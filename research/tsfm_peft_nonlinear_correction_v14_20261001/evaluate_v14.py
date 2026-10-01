"""Joint fixed-parent correction comparison with unchanged reference predictions."""
import argparse
import csv
import gc
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from runtime_v14 import HERE, ROOT, CACHE, SEEDS, Job, artifact, digest, import_file, load_data, protocol, read_json, save_json

NEW = ('level_gelu', 'level_linear', 'direct_gelu')
REFERENCES = ('pca_level', 'a', 'f0', 'full_mse', 'native', 'direct', 'group_res')
ROLES = NEW + REFERENCES
PERIODS = ('test_a', 'test_b', 'combined')
LRS = (1e-4, 1e-3)
V13 = ROOT / 'research/tsfm_peft_group_basis_v13_20261001'
V12 = ROOT / 'research/tsfm_peft_a_confirmation_v12_20261001'
V11 = ROOT / 'research/tsfm_peft_internal_vs_subspace_v11_20260930'


@lru_cache(maxsize=1)
def previous_api():
    if str(V12) not in sys.path:
        sys.path.append(str(V12))
    return import_file(V12 / 'evaluate_confirmation_v12.py', 'v14_verified_metric_primitives')


def reuse():
    value = read_json(HERE / 'reuse_manifest.json')
    if set(value['datasets']) != set(protocol()['units']):
        raise ValueError('Reuse manifest must include all four fixed datasets')
    return value


def selected():
    from train_v14 import result_for
    p = protocol()
    fits = {spec['id']: result_for(spec) for spec in p['fits']}
    value = read_json(HERE / 'selected.json')
    if len(fits) != 48 or set(value['selected']) != set(p['units']):
        raise ValueError('All forty-eight fits and all datasets must finish before evaluation')
    for dataset, families in value['selected'].items():
        if set(families) != set(NEW):
            raise ValueError('New families changed')
        for family, item in families.items():
            if item['seeds'] != list(SEEDS) or item['lr'] not in LRS:
                raise ValueError('Fixed seed/LR candidate contract changed')
            lr_scores = {}
            for lr in LRS:
                members = [r for r in fits.values() if r['spec']['dataset'] == dataset
                           and r['spec']['family'] == family and r['spec']['lr'] == lr]
                if sorted(r['spec']['seed'] for r in members) != list(SEEDS):
                    raise ValueError('Both seed fits are required for every LR')
                lr_scores[lr] = float(np.mean([r['best_val_mse'] for r in members]))
            if item['lr'] != min(LRS, key=lambda lr: (lr_scores[lr], LRS.index(lr))):
                raise ValueError('LR selection differs from mean-seed best VAL with fixed tie order')
            for i, seed in enumerate(SEEDS):
                matches = [r for r in fits.values() if r['spec']['dataset'] == dataset
                           and r['spec']['family'] == family and r['spec']['seed'] == seed and r['spec']['lr'] == item['lr']]
                if len(matches) != 1:
                    raise ValueError('Completed matching fit is missing')
                row = matches[0]
                if (item['run_ids'][i] != row['id'] or item['checkpoints'][i] != row['checkpoint']
                        or item['checkpoint_sha256'][i] != row['checkpoint_sha256']):
                    raise ValueError('Selection does not match its completed actual attempt')
                artifact(row['checkpoint'], row['checkpoint_sha256'])
    return value


def evaluation_rows(selection, dataset):
    rows = []
    for family in NEW:
        item = selection['selected'][dataset][family]
        for i, seed in enumerate(SEEDS):
            rows.append({'id': item['run_ids'][i], 'dataset': dataset, 'family': family, 'seed': seed,
                         'checkpoint': artifact(item['checkpoints'][i], item['checkpoint_sha256'][i]),
                         'reused': False})
    for original in reuse()['datasets'][dataset]['references']:
        names = ('id', 'family', 'seed', 'checkpoint', 'prediction', 'source_evaluation',
                 'source_model_id', 'source_generation')
        row = {name: original[name] for name in names if name in original}
        row['source_row'] = {name: original['source_row'][name] for name in
                             ('id', 'dataset', 'family', 'seed', 'checkpoint', 'sources') if name in original['source_row']}
        row.update(dataset=dataset, reused=True)
        rows.append(row)
    if len(rows) != 19 or len({row['id'] for row in rows}) != 19:
        raise ValueError('Six new and thirteen reference model instances are required')
    for family in ROLES:
        seeds = [row['seed'] for row in rows if row['family'] == family]
        if sorted(seeds, key=str) != sorted([None] if family == 'f0' else list(SEEDS), key=str):
            raise ValueError('Missing prescribed reference or seed: ' + family)
    return rows


def seal(job):
    destination = HERE / 'selection_seal.json'
    if destination.exists() or (HERE / 'test_exposure.json').exists():
        raise FileExistsError('Do not overwrite selection or exposure history')
    p, choice = protocol(), selected()
    files = ('protocol.json', 'selected.json', 'reuse_manifest.json', 'data_manifest.json',
             'model_v14.py', 'train_v14.py', 'runtime_v14.py', 'evaluate_v14.py', 'cost_v14.py')
    units = {dataset: {'rows': evaluation_rows(choice, dataset), 'unit': unit}
             for dataset, unit in p['units'].items()}
    value = {'status': 'sealed', 'datasets': units, 'files': {name: digest(HERE / name) for name in files},
             'created_utc': time.time(), 'all_fits_selected_before_new_evaluation': True,
             'independent_confirmation': False, 'scope': 'All four datasets are now exposed development data'}
    save_json(destination, value)
    job.heartbeat('All four datasets and forty-eight fits selected together')
    return value


def ensure_exposure():
    value = read_json(HERE / 'selection_seal.json')
    if set(value['datasets']) != set(protocol()['units']):
        raise ValueError('Selection seal omits a fixed dataset')
    for name, sha in value['files'].items():
        artifact(HERE / name, sha)
    path = HERE / 'test_exposure.json'
    if path.exists():
        if read_json(path)['seal_sha256'] != digest(HERE / 'selection_seal.json'):
            raise ValueError('Exposure belongs to another selection')
    else:
        save_json(path, {'seal_sha256': digest(HERE / 'selection_seal.json'), 'first_new_evaluation_utc': time.time(),
                         'datasets': list(value['datasets']), 'previously_exposed_development_data': True})
    return value


def prediction_array(receipt, origins, shape):
    artifact(receipt['path'], receipt['sha256'])
    with np.load(receipt['path'], allow_pickle=False) as archive:
        if not np.array_equal(archive['origins'], origins):
            raise ValueError('Reference/new origin order differs')
        array = archive['prediction'].copy()
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError('Prediction shape/finiteness differs')
    return array


def predict(job):
    import torch
    from model_v14 import restore_model
    value = ensure_exposure()
    destination = HERE / 'prediction_manifest.json'
    if destination.exists():
        raise FileExistsError('Prediction phase already complete; reuse it')
    folder = CACHE / 'evaluation_predictions'
    folder.mkdir(parents=True, exist_ok=True)
    output = {'status': 'partial', 'seal': artifact(HERE / 'selection_seal.json'), 'datasets': {}}
    for dataset, unit in value['datasets'].items():
        data = load_data(dataset, include_test=True)
        origins = np.concatenate([data['test_a_origins'], data['test_b_origins']]).astype(np.int64)
        shape = (len(origins), 48, data['x'].shape[1])
        records = []
        for row in unit['rows']:
            job.check_limits()
            record = dict(row)
            if row['reused']:
                prediction_array(row['prediction'], origins, shape)
            else:
                path, receipt_path = folder / (row['id'] + '.npz'), folder / (row['id'] + '.json')
                if path.exists() or receipt_path.exists():
                    if not path.exists() or not receipt_path.exists():
                        raise ValueError('Preserve and repair partial prediction explicitly')
                    prior = read_json(receipt_path)
                    if prior['seal_sha256'] != output['seal']['sha256'] or prior['id'] != row['id']:
                        raise ValueError('Partial prediction selection mismatch')
                    prediction_array(prior['prediction'], origins, shape)
                    record.update(prior)
                else:
                    model = restore_model(row['checkpoint']['path'], device='cuda').eval()
                    outputs = []
                    with torch.no_grad():
                        for start in range(0, len(origins), 4):
                            job.check_limits()
                            x = torch.from_numpy(np.stack([data['x'][o - 512:o] for o in origins[start:start + 4]])).to('cuda')
                            outputs.append(model(x).float().cpu().numpy())
                    result = np.concatenate(outputs)
                    if result.shape != shape or not np.isfinite(result).all():
                        raise ValueError('New correction predictions are invalid')
                    np.savez_compressed(path, origins=origins, prediction=result)
                    record.update(prediction=artifact(path), seal_sha256=output['seal']['sha256'])
                    save_json(receipt_path, record)
                    del model, x, result, outputs
                    gc.collect()
                    torch.cuda.empty_cache()
            records.append(record)
            output['datasets'][dataset] = {'models': records, 'shape': list(shape)}
            save_json(HERE / 'prediction_manifest_partial.json', output)
            job.heartbeat(dataset + ': completed/reused ' + row['id'])
    output['status'] = 'complete'
    save_json(destination, output)
    return output


def comparison(role_terms, indices, block, unit_index, job):
    api = previous_api()
    weights = api.bootstrap_weights(indices, block, unit_index)
    draws = {}
    for period, weight in weights.items():
        denominator = weight @ role_terms['f0']['count']
        if np.any(denominator <= 0):
            raise ValueError('A fixed bootstrap sample has no observed channel')
        draws[period] = {}
        for family, terms in role_terms.items():
            job.check_limits()
            draws[period][family] = {metric: np.mean((weight @ terms[key]) / denominator, axis=1)
                                    for metric, key in (('mse', 'squared'), ('mae', 'absolute'))}
    pairs = [('level_gelu', name) for name in ('level_linear', 'direct_gelu') + REFERENCES]
    pairs += [('direct_gelu', 'direct'), ('level_linear', 'pca_level')]
    values = {}
    for left, right in pairs:
        item = {'candidate': left, 'reference': right, 'periods': {}}
        for period, index in indices.items():
            a, b = (api.aggregate_terms(role_terms[family], index) for family in (left, right))
            item['periods'][period] = {}
            for metric in ('mse', 'mae'):
                delta = a[metric] - b[metric]
                da, db = draws[period][left][metric], draws[period][right][metric]
                item['periods'][period][metric] = {'candidate': a[metric], 'reference': b[metric],
                    'absolute_difference': delta, 'relative_change_percent': 100 * delta / b[metric] if b[metric] > 0 else None,
                    'conditional_block95_absolute': np.quantile(da - db, [.025, .975]).tolist(),
                    'conditional_block95_relative_percent': np.quantile(100 * (da - db) / db, [.025, .975]).tolist() if np.all(db > 0) else None}
        values[left + '__vs__' + right] = item
    return values


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@lru_cache(maxsize=None)
def source_evaluation(path, expected):
    artifact(path, expected)
    return read_json(path)


def score(job):
    ensure_exposure()
    api = previous_api()
    destination = HERE / 'evaluation01.json'
    if destination.exists():
        raise FileExistsError('Preserve original evaluation and record corrections separately')
    manifest = read_json(HERE / 'prediction_manifest.json')
    if manifest['status'] != 'complete' or set(manifest['datasets']) != set(protocol()['units']):
        raise ValueError('All prescribed predictions must complete before comparison')
    artifact(manifest['seal']['path'], manifest['seal']['sha256'])
    output = {'status': 'partial', 'datasets': {}, 'prediction_manifest': artifact(HERE / 'prediction_manifest.json'),
              'selection_seal': artifact(HERE / 'selection_seal.json'), 'independent_confirmation': False,
              'scope': 'Four exposed development datasets; mean seed losses, never an ensemble'}
    scalar_rows, channel_rows = [], []
    for unit_index, (dataset, unit) in enumerate(protocol()['units'].items()):
        data = load_data(dataset, include_test=True)
        a, b = data['test_a_origins'], data['test_b_origins']
        origins = np.concatenate([a, b]).astype(np.int64)
        target = np.stack([data['x'][o:o + 48] for o in origins])
        mask = np.stack([data['finite'][o:o + 48] for o in origins]).astype(bool)
        indices = {'test_a': np.arange(len(a)), 'test_b': np.arange(len(a), len(origins)), 'combined': np.arange(len(origins))}
        terms, models, role_terms, summaries = {}, {}, {}, {}
        for row in manifest['datasets'][dataset]['models']:
            prediction = prediction_array(row['prediction'], origins, target.shape)
            term = api.error_terms(prediction, target, mask)
            scores = {period: api.aggregate_terms(term, index) for period, index in indices.items()}
            replay = None
            if row['reused']:
                entry = row['source_evaluation']
                old = source_evaluation(entry['path'], entry['sha256'])['datasets'][dataset]['models'][row['source_model_id']]
                for period in PERIODS:
                    for metric in ('mse', 'mae', 'signed_mean_error'):
                        if not np.isclose(scores[period][metric], old['periods'][period][metric], atol=1e-10, rtol=1e-9):
                            raise ValueError('Stored reference score contract changed: ' + row['id'])
                replay = {'passed': True, 'atol': 1e-10, 'rtol': 1e-9, 'source': entry}
            models[row['id']] = {**row, 'periods': scores, 'reference_score_replay': replay}
            terms[row['id']] = term
            for period, scores in scores.items():
                scalar_rows.append({'dataset': dataset, 'family': row['family'], 'id': row['id'], 'seed': row['seed'],
                    'period': period, 'mse': scores['mse'], 'mae': scores['mae'], 'signed_mean_error': scores['signed_mean_error'],
                    'reused': row['reused']})
                for c, name in enumerate(data['columns']):
                    channel_rows.append({'dataset': dataset, 'family': row['family'], 'id': row['id'], 'seed': row['seed'],
                        'period': period, 'channel': str(name), 'mse': scores['channel_mse'][c], 'mae': scores['channel_mae'][c],
                        'signed_mean_error': scores['channel_bias'][c], 'target_count': scores['target_counts'][c]})
        for family in ROLES:
            ids = [identifier for identifier, row in models.items() if row['family'] == family]
            if len(ids) != (1 if family == 'f0' else 2):
                raise ValueError('Every method and seed is required')
            count = terms[ids[0]]['count']
            if any(not np.array_equal(terms[identifier]['count'], count) for identifier in ids):
                raise ValueError('Reference and candidate masks differ')
            term = {key: np.mean([terms[identifier][key] for identifier in ids], axis=0) for key in ('squared', 'absolute', 'signed')}
            term['count'] = count
            role_terms[family] = term
            summaries[family] = {'ids': ids, 'aggregation': 'Mean seed losses, not ensemble',
                                 'periods': {period: api.aggregate_terms(term, index) for period, index in indices.items()}}
        term_path = CACHE / (dataset + '_origin_terms01.npz')
        np.savez_compressed(term_path, origins=origins, **{identifier + '__' + key: v for identifier, item in terms.items() for key, v in item.items()})
        output['datasets'][dataset] = {'models': models, 'role_summary': summaries,
            'comparisons': comparison(role_terms, indices, unit['block_origins'], unit_index, job),
            'origin_terms': artifact(term_path), 'data': artifact(data['_path']), 'columns': [str(v) for v in data['columns']],
            'period_origin_counts': {period: len(index) for period, index in indices.items()},
            'bootstrap': {'draws': 2000, 'seed': 9262026, 'block_origins': unit['block_origins'],
                          'period_boundaries_crossed': False, 'selection_uncertainty_included': False},
            'aggregation': 'Pool channel errors/counts across periods, then channel macro, then mean seed losses'}
        save_json(HERE / 'evaluation01_partial.json', output)
        job.heartbeat(dataset + ' complete including all references')
    write_csv(HERE / 'evaluation01_models.csv', scalar_rows)
    write_csv(HERE / 'evaluation01_channels.csv', channel_rows)
    output.update(status='complete', tables={name: artifact(HERE / f'evaluation01_{name}.csv') for name in ('models', 'channels')})
    save_json(destination, output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('seal', 'predict', 'score'))
    parser.add_argument('--job', required=True)
    parser.add_argument('--reserve-s', type=float, required=True)
    args = parser.parse_args()
    category = {'seal': 'cpu_check', 'predict': 'gpu', 'score': 'cpu_analysis'}[args.action]
    with Job(category, args.job, reserve_s=args.reserve_s, metadata={'fit_count': 0, 'purpose': 'v14 ' + args.action}) as job:
        if args.action == 'predict':
            from runtime_v14 import configure
            configure()
        {'seal': seal, 'predict': predict, 'score': score}[args.action](job)


if __name__ == '__main__':
    main()
