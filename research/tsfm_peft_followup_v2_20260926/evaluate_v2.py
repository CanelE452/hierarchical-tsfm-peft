"""Validation-only mode selection and exposed development evaluation for v2.

V1 artifacts are read-only. Every invocation claims a new named attempt and saves
its selection before looking at DEV/E1/E2 predictions. All completed cohort modes
are reported, including modes not selected. No missing RAW mode is treated as a
fair final selection comparison. GPU execution is only in the explicit CLI.
"""

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V1 = ROOT / 'research' / 'tsfm_peft_development_20260926'
OLD_CACHE = ROOT / '.cache' / 'tsfm_peft_development_20260926'
CACHE = ROOT / '.cache' / HERE.name
SEEDS = [92601, 92602]
MODES = ['current', 'fixed_ed', 'slow_ed']
PERIODS = {'electricity': ['dev'], 'bull': ['e1', 'e2']}
BASE_LATENT = {'electricity': 8, 'bull': 4}
SCHEMA_VERSION = 2


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, payload, exclusive=False):
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if exclusive:
        with Path(path).open('x', encoding='utf-8') as stream:
            stream.write(text)
    else:
        temporary = Path(path).with_suffix('.json.tmp')
        temporary.write_text(text, encoding='utf-8')
        for retry in range(6):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if retry == 5:
                    raise
                time.sleep(.05 * 2 ** retry)


def scores(prediction, target, mask):
    prediction, target, mask = np.asarray(prediction), np.asarray(target), np.asarray(mask, dtype=bool)
    if prediction.shape != target.shape or mask.shape != target.shape or prediction.ndim != 3:
        raise ValueError('prediction, target, mask must share [origin,horizon,channel] shape')
    if not np.isfinite(prediction).all() or not np.isfinite(target[mask]).all():
        raise ValueError('nonfinite prediction or observed target')
    count = mask.sum(axis=(0, 1))
    if not np.all(count > 0):
        raise ValueError('no evaluation target for a selected channel')
    error = np.where(mask, prediction.astype(np.float64) - target, 0)
    channel_mse = (error ** 2).sum(axis=(0, 1)) / count
    channel_mae = np.abs(error).sum(axis=(0, 1)) / count
    return dict(mse=float(channel_mse.mean()), mae=float(channel_mae.mean()),
                channel_mse=channel_mse.tolist(), channel_mae=channel_mae.tolist(),
                target_counts=count.tolist(), origins=len(prediction))


def score_period(prediction, target, mask):
    full = scores(prediction, target, mask)
    halves = []
    for part in (slice(None, len(target) // 2), slice(len(target) // 2, None)):
        try:
            halves.append(dict(status='available', **scores(prediction[part], target[part], mask[part])))
        except ValueError as exc:
            if 'no evaluation target' not in str(exc):
                raise
            halves.append(dict(status='unavailable', reason=str(exc)))
    return dict(full=full, halves=halves)


def validate_origins(data, dataset, period, contract):
    key = period if period not in ('e1', 'e2') else 'eval_' + period
    origins = np.asarray(data[key + '_origins'])
    entry = contract['datasets']['electricity_first32' if dataset == 'electricity' else 'bdg2_bull_office']
    if dataset == 'electricity':
        start, end = entry['split_bounds'][key]
    else:
        start, end = [int(np.searchsorted(data['times'], np.datetime64(t, 'ns').astype(np.int64)))
                      for t in entry['split_periods'][key]]
    if (origins.ndim != 1 or not np.issubdtype(origins.dtype, np.integer) or len(origins) == 0
            or np.any(np.diff(origins) <= 0) or np.any(origins < max(512, start))
            or np.any(origins + 48 > end) or end > len(data['x'])):
        raise ValueError(f'illegal or leaking {dataset}/{period} origins')
    if len(origins) != entry['origin_counts'][key]:
        raise ValueError('origin count differs from the unchanged v1 data contract')
    return origins


def windows(data, origins):
    return (np.stack([data['x'][o:o + 48] for o in origins]),
            np.stack([data['finite'][o:o + 48] for o in origins]).astype(bool))


def read_prediction(path, origins, target, mask, expected_hash=None, expected_scores=None):
    actual_hash = digest(path)
    if expected_hash is not None and actual_hash != expected_hash:
        raise ValueError(f'prediction hash mismatch: {path}')
    with np.load(path, allow_pickle=False) as archive:
        if not np.array_equal(archive['origins'], origins):
            raise ValueError(f'prediction origins mismatch: {path}')
        prediction = archive['prediction'].copy()
    computed = scores(prediction, target, mask)
    if expected_scores is not None:
        for metric in ('mse', 'mae'):
            if not np.isclose(computed[metric], expected_scores[metric], rtol=0, atol=1e-8):
                raise ValueError(f'cached {metric} differs from recorded score: {path}')
    return prediction, dict(path=str(Path(path).resolve()), sha256=actual_hash,
                           prior_sha256_verified=expected_hash is not None)


def variant_key(mode, latent):
    if mode not in MODES or type(latent) is not int or latent <= 0:
        raise ValueError('variant requires a supported E/D mode and a positive integer latent count')
    return f'{mode}:k{latent}'


def row_label(row):
    if row['arm'] in ('residual', 'raw_bypass'):
        return row['arm'] + ':' + variant_key(row['ed_mode'], row['latent'])
    return row['arm']


def select_modes(rows):
    """Select (E/D mode, K) by two seed mean VAL, with stable base-K ties."""
    selection = {}
    for dataset in sorted({row['dataset'] for row in rows}):
        selected = {}
        for arm in ('residual', 'raw_bypass'):
            eligible = [r for r in rows if r['dataset'] == dataset and r['arm'] == arm]
            grouped = {}
            for row in eligible:
                mode = row.get('ed_mode', 'current')
                if mode not in MODES or row['seed'] not in SEEDS or not np.isfinite(row['best_val_mse']):
                    raise ValueError('unsupported mode, seed, or validation value')
                variant = variant_key(mode, row['latent'])
                group = grouped.setdefault(variant, {})
                if row['seed'] in group:
                    raise ValueError('multiple fits for one dataset/arm/mode/latent/seed; supply a resolved cohort')
                group[row['seed']] = row
            variant_scores, fit_ids = {}, {}
            for variant, group in grouped.items():
                if sorted(group) != SEEDS:
                    raise ValueError(f'incomplete paired seeds for {dataset}/{arm}/{variant}')
                variant_scores[variant] = float(np.mean([group[s]['best_val_mse'] for s in SEEDS]))
                fit_ids[variant] = [group[s]['id'] for s in SEEDS]
            if not variant_scores:
                selected[arm] = dict(status='pending', available_modes=[], available_variants=[])
                continue
            def preference(variant):
                representative = grouped[variant][SEEDS[0]]
                latent = representative['latent']
                return (latent != BASE_LATENT[dataset], latent, MODES.index(representative.get('ed_mode', 'current')))
            variants = sorted(variant_scores, key=preference)
            best = min(variants, key=lambda v: (variant_scores[v], preference(v)))
            representative = grouped[best][SEEDS[0]]
            selected[arm] = dict(status='selected', mode=representative.get('ed_mode', 'current'),
                                 latent=representative['latent'], variant=best,
                                 mean_best_val_mse=variant_scores[best], variant_scores=variant_scores,
                                 fits=fit_ids[best], variant_fits=fit_ids, available_variants=variants,
                                 available_modes=[m for m in MODES if any(v.startswith(m + ':') for v in variants)])
        left, right = (selected[a]['available_variants'] for a in ('residual', 'raw_bypass'))
        selected['selection_opportunity'] = dict(
            status='matched' if left == right and left else 'pending',
            residual_variants=left, raw_variants=right,
            final_fair_selection_comparison=bool(left == right and left),
            reason='same completed (mode, latent) and paired seed opportunities' if left == right and left
            else 'RAW and RESIDUAL have unequal (mode, latent) opportunities; no final selected-method superiority claim')
        selection[dataset] = selected
    return selection


def paired_effect(predictions, references, target, mask, block=7, draws=2000):
    """V1 non-circular 7-origin block procedure, with masked NaNs excluded."""
    if len(references) not in (1, len(predictions)):
        raise ValueError('pair identical seed order or one fixed reference')
    mt = np.mean([np.where(mask, (p.astype(np.float64) - target) ** 2, 0).sum(axis=1) for p in predictions], axis=0)
    rt = np.mean([np.where(mask, (p.astype(np.float64) - target) ** 2, 0).sum(axis=1) for p in references], axis=0)
    counts = mask.sum(axis=1)
    def gain(index):
        count = counts[index].sum(axis=0)
        if np.any(count == 0):
            raise ValueError('a block draw contains no targets for a channel')
        m, r = [(terms[index].sum(axis=0) / count).mean() for terms in (mt, rt)]
        if r <= 0:
            raise ValueError('relative effect undefined for a zero-loss reference')
        return float(100 * (r - m) / r)
    n = len(target)
    if not 1 <= block <= n:
        raise ValueError('block length must fit the evaluation period')
    try:
        rng = np.random.default_rng(9262026)
        sampled = [gain(np.concatenate([s + np.arange(block) for s in
                   rng.integers(0, n - block + 1, int(np.ceil(n / block)))])[:n]) for _ in range(draws)]
        seed_gains = []
        for i, prediction in enumerate(predictions):
            reference = references[0 if len(references) == 1 else i]
            ref = scores(reference, target, mask)['mse']
            if ref <= 0:
                raise ValueError('relative effect undefined for a zero-loss reference')
            seed_gains.append(100 * (ref - scores(prediction, target, mask)['mse']) / ref)
        return dict(status='available', gain_pct=gain(np.arange(n)), paired_seed_gain_pct=seed_gains,
                    period_halves_gain_pct=[gain(np.arange(n // 2)), gain(np.arange(n // 2, n))],
                    conditional_block95_pct=np.quantile(sampled, [.025, .975]).tolist(),
                    block_origins=block, draws=draws,
                    scope='paired non-circular time blocks, channels together, trained seeds fixed; exposed development and selection uncertainty not represented')
    except ValueError as exc:
        return dict(status='unavailable', reason=str(exc), block_origins=block, draws=draws)


def read_fit(result_path, spec=None):
    row = read_json(result_path)
    if row.get('status') != 'complete':
        raise ValueError(f'fit is not complete: {result_path}')
    if spec is not None and any(row.get(key) != value for key, value in spec.items()):
        raise ValueError(f'fit differs from cohort spec: {result_path}')
    if digest(row['checkpoint']) != row['checkpoint_sha256']:
        raise ValueError(f'checkpoint hash mismatch: {row["id"]}')
    row = dict(row, ed_mode=row.get('ed_mode', 'current'), result_path=str(result_path),
               result_sha256=digest(result_path))
    return row


def legacy_catalog(v1=V1, cache=OLD_CACHE):
    """Paths are derived from actual v1 evaluators and their recorded results."""
    catalog, fits, receipts = [], {}, {}
    for dataset, filename, subdir in (
            ('electricity', 'revision1_rank32_development.json', 'revision1_rank32_evaluation'),
            ('bull', 'protected_evaluation.json', 'protected_evaluation')):
        report_path = v1 / filename
        report = read_json(report_path)
        if report['status'] != 'complete':
            raise ValueError('incomplete v1 reference report')
        receipts[dataset] = dict(path=str(report_path), sha256=digest(report_path), data_sha256=report['data_sha256'])
        for old in report['rows']:
            arm, fit = old['arm'], old['fit']
            if fit is not None and fit not in fits:
                fits[fit] = read_fit(v1 / 'runs' / fit / 'result.json')
            if dataset == 'electricity' and arm in ('factor_linear', 'shared_linear', 'seasonal'):
                linear_path = v1 / report['linear_baseline_report']
                linear = read_json(linear_path)
                if digest(linear_path) != report['linear_baseline_report_sha256']:
                    raise ValueError('linear report hash differs from development report')
                leaf = 'seasonal_dev.npz' if arm == 'seasonal' else f'{arm}_{linear["selected"][arm]["ridge_penalty"]:g}_dev.npz'
                path = cache / linear['cache_subdirectory'] / leaf
            else:
                leaf = old.get('prediction_cache', (fit or 'f0') + '.npz')
                path = cache / subdir / leaf
            catalog.append(dict(dataset=dataset, period=old.get('period', 'dev'), arm=arm,
                                fit=fit, seed=old['seed'], ed_mode='current' if arm in ('residual', 'raw_bypass') else None,
                                latent=fits[fit].get('latent') if fit is not None else None,
                                path=path, expected_hash=old.get('prediction_sha256'), old_scores=old['scores']))
    return catalog, list(fits.values()), receipts


def period_comparisons(rows, arrays, selection, target, mask):
    groups, aggregate_rows = {}, {}
    for row in rows:
        label = row_label(row)
        group = groups.setdefault(label, {})
        if row['seed'] in group:
            raise ValueError('duplicate prediction for a variant and seed')
        group[row['seed']] = arrays[row['key']]
        aggregate_rows.setdefault(label, []).append(row)
    mean_scores = {}
    for label, members in aggregate_rows.items():
        mean_scores[label] = dict(
            arm=members[0]['arm'], ed_mode=members[0]['ed_mode'], latent=members[0].get('latent'),
            seed_loss_mean_mse=float(np.mean([r['scores']['full']['mse'] for r in members])),
            seed_loss_mean_mae=float(np.mean([r['scores']['full']['mae'] for r in members])),
            mean_channel_mse=np.mean([r['scores']['full']['channel_mse'] for r in members], axis=0).tolist(),
            mean_channel_mae=np.mean([r['scores']['full']['channel_mae'] for r in members], axis=0).tolist(),
            per_seed=[dict(seed=r['seed'], fit=r['fit'], mse=r['scores']['full']['mse'], mae=r['scores']['full']['mae'])
                      for r in members],
            aggregation='individual seed losses averaged; not an ensemble')
    def forecasts(label):
        group = groups[label]
        if set(group) == {None}:
            return [group[None]]
        if sorted(group) != SEEDS:
            raise ValueError('comparison requires the complete paired seed cohort')
        return [group[s] for s in SEEDS]
    effects = {}
    for label in groups:
        if label.startswith('residual:'):
            effects[label] = {ref: paired_effect(forecasts(label), forecasts(ref), target, mask)
                              for ref in groups if ref != label and not ref.startswith('residual:')}
            current = 'residual:current:k' + str(BASE_LATENT[rows[0]['dataset']])
            if label != current and current in groups:
                effects[label][current] = paired_effect(forecasts(label), forecasts(current), target, mask)
    chosen = {}
    for arm in ('residual', 'raw_bypass'):
        if selection[arm]['status'] == 'selected':
            chosen[arm] = arm + ':' + selection[arm]['variant']
    selected_effect = dict(status='pending', reason=selection['selection_opportunity']['reason'])
    if selection['selection_opportunity']['final_fair_selection_comparison']:
        selected_effect = paired_effect(forecasts(chosen['residual']), forecasts(chosen['raw_bypass']), target, mask)
        selected_effect['chosen_labels'] = chosen
    matched = {}
    variants = sorted({label.split(':', 1)[1] for label in groups if label.startswith(('residual:', 'raw_bypass:'))})
    for variant in variants:
        pair = ['residual:' + variant, 'raw_bypass:' + variant]
        matched[variant] = (paired_effect(forecasts(pair[0]), forecasts(pair[1]), target, mask)
                            if all(label in groups for label in pair) else dict(status='pending', reason='matched (mode, K) comparison not complete'))
    latent_effects = {}
    for arm in ('residual', 'raw_bypass'):
        larger, reference = arm + ':fixed_ed:k8', arm + ':fixed_ed:k4'
        if larger in groups and reference in groups:
            latent_effects[arm] = dict(
                candidate=larger, reference=reference,
                effect=paired_effect(forecasts(larger), forecasts(reference), target, mask),
                interpretation='fixed E/D at K8 versus K4; increased backbone channel computation as well as changed representation')
    return dict(mean_scores=mean_scores, all_residual_variants_vs=effects, matched_variant_residual_vs_raw=matched,
                fixed_ed_k8_vs_k4=latent_effects,
                selected_residual_vs_selected_raw=selected_effect)


def run_evaluation(cohort, name, reuse_reports=()):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('name must be a safe new artifact stem')
    from run import GPUJob, evaluate, load_data, make_model, source_receipt
    import torch
    paths = {kind: HERE / f'{name}_{kind}.json' for kind in ('attempt', 'selection', 'development')}
    local = CACHE / (name + '_evaluation')
    if any(path.exists() for path in paths.values()) or local.exists():
        raise FileExistsError('never overwrite an evaluation or its selection; preserve failures under their original name')
    specs = read_json(cohort)
    if not isinstance(specs, list) or len({s['id'] for s in specs}) != len(specs):
        raise ValueError('cohort must be a list of unique fit specs')
    new_fits = [read_fit(HERE / 'runs' / spec['id'] / 'result.json', spec) for spec in specs]
    if any(r['arm'] not in ('residual', 'raw_bypass') for r in new_fits):
        raise ValueError('this first-comparison evaluator accepts residual/raw cohort fits only')
    catalog, legacy_fits, legacy_receipts = legacy_catalog()
    fitted = legacy_fits + new_fits
    if len({r['id'] for r in fitted}) != len(fitted):
        raise ValueError('cohort fit collides with a reused v1 fit')
    fit_by_id = {r['id']: r for r in fitted}
    record = dict(status='running', schema_version=SCHEMA_VERSION, pid=os.getpid(), started_utc=time.time(), source=source_receipt(),
                  evaluator_sha256=digest(__file__), cohort=str(Path(cohort).resolve()), cohort_sha256=digest(cohort),
                  legacy_reports=legacy_receipts, result_hashes={r['id']: r['result_sha256'] for r in fitted}, partial_rows=[])
    previous = {}
    record['reused_v2_reports'] = []
    for path in reuse_reports:
        report = read_json(path)
        if report.get('status') != 'complete':
            raise ValueError('only completed v2 evaluation can supply cached predictions')
        record['reused_v2_reports'].append(dict(path=str(Path(path).resolve()), sha256=digest(path)))
        for row in report['rows']:
            if row['fit'] not in {r['id'] for r in new_fits}:
                continue
            fit = next(r for r in new_fits if r['id'] == row['fit'])
            if report['result_hashes'][fit['id']] != fit['result_sha256']:
                raise ValueError('prior v2 result hash changed; cannot reuse prediction')
            if report['data_sha256'][fit['dataset']] != fit['data_sha256']:
                raise ValueError('prior v2 prediction used different data')
            if any(row[k] != fit[k] for k in ('dataset', 'arm', 'seed', 'ed_mode')):
                raise ValueError('prior v2 prediction identity differs from fit')
            if 'latent' in row and row['latent'] != fit['latent']:
                raise ValueError('prior v2 prediction latent differs from its hash-verified fit')
            previous[(row['fit'], row['period'])] = row
    write_json(paths['attempt'], record, exclusive=True)
    started = time.perf_counter()
    try:
        data = {dataset: load_data(dataset) for dataset in PERIODS}
        contract = read_json(V1 / 'data_contract.json')
        validation_receipts = {}
        for dataset, item in data.items():
            data_hash = digest(item['_path'])
            if data_hash != legacy_receipts[dataset]['data_sha256']:
                raise ValueError('data changed from v1 reference report')
            for split in ['train', 'val'] + PERIODS[dataset]:
                validate_origins(item, dataset, split, contract)
            origins = item['val_origins']
            target, mask = windows(item, origins)
            for row in fitted:
                if row['dataset'] != dataset:
                    continue
                if row['data_sha256'] != data_hash:
                    raise ValueError('fit data hash differs from current reference data')
                prediction, receipt = read_prediction(Path(row['checkpoint']).with_name('best_val.npz'), origins, target, mask)
                if not np.isclose(scores(prediction, target, mask)['mse'], row['best_val_mse'], rtol=0, atol=1e-8):
                    raise ValueError('best validation cache does not match selection score')
                validation_receipts[row['id']] = receipt
        selection = select_modes([r for r in fitted if r['arm'] in ('residual', 'raw_bypass')])
        selection_record = dict(schema_version=SCHEMA_VERSION, selection=selection,
                                criterion='two fixed seed losses averaged on VAL; select (mode,K); no prediction ensemble',
                                validation_receipts=validation_receipts, cohort_sha256=digest(cohort),
                                result_hashes=record['result_hashes'], saved_before_development_evaluation=True)
        write_json(paths['selection'], selection_record, exclusive=True)
        record['selection_sha256'] = digest(paths['selection'])
        write_json(paths['attempt'], record)
        local.mkdir(parents=True, exist_ok=False)
        period_arrays, predictions, period_rows = {}, {}, {}
        for dataset, item in data.items():
            for period in PERIODS[dataset]:
                key = dataset + '/' + period
                origins = validate_origins(item, dataset, period, contract)
                period_arrays[key] = (origins, *windows(item, origins))
                predictions[key], period_rows[key] = {}, []
        def add(dataset, period, arm, fit, seed, mode, prediction, receipt, reused):
            key = dataset + '/' + period
            _, target, mask = period_arrays[key]
            row_key = fit or arm
            if row_key in predictions[key]:
                raise ValueError('duplicate prediction key')
            row = dict(dataset=dataset, period=period, arm=arm, fit=fit, seed=seed, ed_mode=mode,
                       latent=fit_by_id[fit].get('latent') if fit is not None else None,
                       key=row_key, scores=score_period(prediction, target, mask), prediction=receipt, reused=reused)
            predictions[key][row_key] = prediction
            period_rows[key].append(row)
            record['partial_rows'].append(row)
            write_json(paths['attempt'], record)
        for old in catalog:
            origins, target, mask = period_arrays[old['dataset'] + '/' + old['period']]
            prediction, receipt = read_prediction(old['path'], origins, target, mask, old['expected_hash'], old['old_scores'])
            add(old['dataset'], old['period'], old['arm'], old['fit'], old['seed'], old['ed_mode'], prediction, receipt, True)
        for (fit_id, period), prior in previous.items():
            origins, target, mask = period_arrays[prior['dataset'] + '/' + period]
            prediction, receipt = read_prediction(prior['prediction']['path'], origins, target, mask,
                                                  prior['prediction']['sha256'], prior['scores']['full'])
            add(prior['dataset'], period, prior['arm'], fit_id, prior['seed'], prior['ed_mode'], prediction, receipt, True)
        pending_fits = [r for r in new_fits if any((r['id'], p) not in previous for p in PERIODS[r['dataset']])]
        gpu_seconds = 0.0
        with GPUJob(name + '_development_evaluation') if pending_fits else nullcontext() as job:
            for row in pending_fits:
                saved = torch.load(row['checkpoint'], map_location='cpu', weights_only=False)
                if any(row.get(k) != v for k, v in saved['spec'].items()):
                    raise ValueError('checkpoint spec differs from completed result')
                model = make_model(row['arm'], saved['basis'], residual_rank=row.get('residual_rank', 32), ed_mode=row['ed_mode'])
                try:
                    model.restore_adapter(saved['state'])
                    for period in PERIODS[row['dataset']]:
                        if (row['id'], period) in previous:
                            continue
                        origins, _, _ = period_arrays[row['dataset'] + '/' + period]
                        _, prediction = evaluate(model, data[row['dataset']], origins, job, row['id'] + ' ' + period)
                        path = local / f'{period}_{row["id"]}.npz'
                        with path.open('xb') as stream:
                            np.savez_compressed(stream, prediction=prediction, origins=origins)
                        receipt = dict(path=str(path), sha256=digest(path), prior_sha256_verified=False)
                        add(row['dataset'], period, row['arm'], row['id'], row['seed'], row['ed_mode'], prediction, receipt, False)
                finally:
                    del model, saved
                    gc.collect()
                    torch.cuda.empty_cache()
        if pending_fits:
            gpu_seconds = job.record['elapsed_s']
        comparisons = {}
        for key, rows in period_rows.items():
            _, target, mask = period_arrays[key]
            comparisons[key] = period_comparisons(rows, predictions[key], selection[key.split('/')[0]], target, mask)
        report = dict(status='complete', schema_version=SCHEMA_VERSION, scope='v2 exposed development only; Bull E1/E2 are not a new confirmation',
                      selection=selection, selection_sha256=record['selection_sha256'], rows=record['partial_rows'],
                      comparisons=comparisons, source=record['source'], evaluator_sha256=record['evaluator_sha256'],
                      legacy_reports=legacy_receipts, data_contract_sha256=digest(V1 / 'data_contract.json'),
                      reused_v2_reports=record['reused_v2_reports'],
                      data_sha256={d: digest(x['_path']) for d, x in data.items()},
                      channels={d: x['columns'].tolist() for d, x in data.items()},
                      result_hashes=record['result_hashes'], gpu_accounted_seconds=gpu_seconds,
                      elapsed_seconds=time.perf_counter() - started,
                      aggregation='mean of individual seed losses; never forecast-ensemble score')
        write_json(paths['development'], report, exclusive=True)
        record.update(status='complete', report_sha256=digest(paths['development']), elapsed_seconds=time.perf_counter() - started)
        write_json(paths['attempt'], record)
        return report
    except BaseException:
        record.update(status='failed', error=traceback.format_exc(), elapsed_seconds=time.perf_counter() - started)
        write_json(paths['attempt'], record)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--reuse', action='append', default=[], help='completed v2 development report to verify and reuse')
    args = parser.parse_args()
    output = run_evaluation(args.cohort, args.name, args.reuse)
    print(json.dumps({'status': output['status'], 'selection': output['selection']}, ensure_ascii=False))
