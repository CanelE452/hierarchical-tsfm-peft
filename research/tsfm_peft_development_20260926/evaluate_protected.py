"""One-shot Bull evaluation, with all artifact checks before loading arrays.

The final seal adds ``evaluation`` with periods ["e1", "e2"], block_origins 7,
linear_penalty 0.001, seasonal_period 24, primary_metric
"train_std_equal_channel_macro_mse", and the development-fixed positive latent.
``evaluation_source_hashes`` must pin the five EVALUATION_SOURCES below.

``preflight(here, root)`` reads JSON and hashes opaque artifact bytes only.
``run_evaluation()`` then claims the fixed attempt path exclusively. There is no
selection or retry option. A failed claim/evaluation must be audited manually;
this script never removes its attempt, partial predictions, or failed report.
The public adaptation receipt is the exact protected_protocol receipt dictionary.
"""

import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

from protected_protocol import validate_protected_evaluation


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EVALUATION_SOURCES = ('evaluate_protected.py', 'evaluate_development.py', 'masked_linear.py',
                      'protected_protocol.py', 'data_prepare.py')
RUN_SPEC_KEYS = {'id', 'dataset', 'arm', 'seed', 'lr', 'latent', 'epochs', 'residual_rank',
                 'loss', 'weight_decay', 'epoch_origins', 'microbatch_origins'}
ATTEMPT_NAME = 'protected_evaluation_attempt.json'
REPORT_NAME = 'protected_evaluation.json'


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _read_json(path, label):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError(f'cannot read {label}: {path}') from exc


def _same_json(left, right):
    options = dict(sort_keys=True, separators=(',', ':'), allow_nan=False)
    return json.dumps(left, **options) == json.dumps(right, **options)


def preflight(here=HERE, root=ROOT, seal_path=None):
    """Verify the immutable cohort without importing models or loading arrays."""
    here, root = Path(here), Path(root)
    seal_path = Path(seal_path) if seal_path else here / 'final_seal.json'
    try:
        seal_bytes = seal_path.read_bytes()
        seal = json.loads(seal_bytes.decode('utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError(f'cannot read final seal: {seal_path}') from exc
    initial_seal_hash = hashlib.sha256(seal_bytes).hexdigest()
    contract = _read_json(here / 'data_contract.json', 'data contract')
    data_path = root / contract['datasets']['bdg2_bull_office']['npz']
    hashes = {name: _digest(here / name) for name in ('data_contract.json', 'protocol.json', 'model.py', 'run.py')}
    hashes['protected_npz'] = _digest(data_path)
    fits = seal.get('allowed_fits', [])
    ids = [spec.get('id') for spec in fits if isinstance(spec, dict)]
    receipt = validate_protected_evaluation(seal_path, ids, hashes)
    if receipt['seal_sha256'] != initial_seal_hash or receipt['seal_sha256'] != _digest(seal_path):
        raise ValueError('seal changed during preflight')
    evaluation = seal.get('evaluation')
    if not isinstance(evaluation, dict):
        raise ValueError('seal requires an evaluation contract')
    latent = evaluation.get('latent')
    if type(latent) is not int or not 1 <= latent <= contract['datasets']['bdg2_bull_office']['selected_count']:
        raise ValueError('evaluation latent must fit the sealed channel count')
    expected_evaluation = dict(periods=['e1', 'e2'], block_origins=7, linear_penalty=0.001,
                               seasonal_period=24, primary_metric='train_std_equal_channel_macro_mse', latent=latent)
    if not _same_json(evaluation, expected_evaluation):
        raise ValueError('unsupported evaluation contract; do not tune protected evaluation')
    source_hashes = {name: _digest(here / name) for name in EVALUATION_SOURCES}
    if not _same_json(seal.get('evaluation_source_hashes'), source_hashes):
        raise ValueError('evaluation source hash mismatch')
    adaptation_path = here / 'protected_adaptation_receipt.json'
    if not _same_json(_read_json(adaptation_path, 'adaptation receipt'), receipt):
        raise ValueError('adaptation receipt does not match final seal')
    results, result_hashes = [], {}
    paired_ids = set()
    for spec in fits:
        if spec['id'] in ('f0', 'shared_linear', 'factor_linear', 'seasonal24'):
            raise ValueError(f'fit id is reserved for a fixed baseline: {spec["id"]}')
        result_path = here / 'runs' / spec['id'] / 'result.json'
        row = _read_json(result_path, 'completed fit result')
        if row.get('status') != 'complete':
            raise ValueError(f'fit result is incomplete: {spec["id"]}')
        recorded_spec = {key: row[key] for key in (RUN_SPEC_KEYS | set(spec)) if key in row}
        if not _same_json(recorded_spec, spec):
            raise ValueError(f'fit result full spec differs from seal: {spec["id"]}')
        if row.get('data_sha256') != hashes['protected_npz']:
            raise ValueError(f'fit result data hash mismatch: {spec["id"]}')
        if not _same_json(row.get('protected_receipt'), receipt):
            raise ValueError(f'fit result protected receipt mismatch: {spec["id"]}')
        try:
            checkpoint = Path(row['checkpoint'])
            actual_checkpoint_hash = _digest(checkpoint)
        except (KeyError, TypeError, OSError) as exc:
            raise ValueError(f'cannot hash checkpoint: {spec["id"]}') from exc
        if row.get('checkpoint_sha256') != actual_checkpoint_hash:
            raise ValueError(f'checkpoint hash mismatch: {spec["id"]}')
        paired_id = (spec['arm'], spec['seed'])
        if paired_id in paired_ids:
            raise ValueError('seal has ambiguous multiple fits for the same arm and seed')
        paired_ids.add(paired_id)
        results.append(row)
        result_hashes[spec['id']] = _digest(result_path)
    validate_protected_evaluation(seal_path, [row['id'] for row in results], hashes)
    return dict(here=here, root=root, data_path=data_path, seal=seal, receipt=receipt,
                results=results, result_hashes=result_hashes, hashes=hashes,
                evaluation_source_hashes=source_hashes,
                adaptation_receipt_sha256=_digest(adaptation_path))


def _write_json(path, payload, exclusive=False):
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(text)
        return
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    for retry in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if retry == 5:
                raise
            time.sleep(0.05 * 2 ** retry)


def run_evaluation(here=HERE, root=ROOT):
    """Run once; every exception preserves the attempt and any cache artifacts."""
    here, root = Path(here), Path(root)
    attempt_path, report_path = here / ATTEMPT_NAME, here / REPORT_NAME
    for path in (attempt_path, report_path):
        if path.exists():
            raise FileExistsError(f'never overwrite protected evaluation: {path}')
    plan = preflight(here, root)
    local = root / '.cache' / 'tsfm_peft_development_20260926' / 'protected_evaluation'
    if local.exists():
        raise FileExistsError(f'preserve existing protected evaluation cache: {local}')
    record = dict(status='running', started_utc=time.time(), pid=os.getpid(),
                  protected_receipt=plan['receipt'], evaluation=plan['seal']['evaluation'],
                  evaluation_source_hashes=plan['evaluation_source_hashes'],
                  result_hashes=plan['result_hashes'],
                  adaptation_receipt_sha256=plan['adaptation_receipt_sha256'], partial_rows=[])
    _write_json(attempt_path, record, exclusive=True)
    started = time.perf_counter()
    try:
        if report_path.exists():
            raise FileExistsError(f'protected report appeared after claim: {report_path}')
        local.mkdir(parents=True, exist_ok=False)
        report = _evaluate_plan(plan, local, record)
        _write_json(report_path, report, exclusive=True)
        record.update(status='complete', elapsed_seconds=time.perf_counter() - started,
                      report_sha256=_digest(report_path))
        _write_json(attempt_path, record)
        return report
    except BaseException:
        record.update(status='failed', elapsed_seconds=time.perf_counter() - started,
                      error=traceback.format_exc())
        _write_json(attempt_path, record)
        raise


def _origin_hash(origins):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(origins, dtype='<i8').tobytes()).hexdigest()


def _save_npz(path, **arrays):
    import numpy as np
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    return _digest(path)


def _windows(data, origins):
    import numpy as np
    return (np.stack([data['x'][o - 512:o] for o in origins]),
            np.stack([data['x'][o:o + 48] for o in origins]),
            np.stack([data['finite'][o:o + 48] for o in origins]))


def _scores(prediction, target, mask):
    import numpy as np
    from run import score_arrays
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError('protected predictions, targets, and masks must have identical shapes')
    if not np.isfinite(prediction).all() or not np.isfinite(target[mask]).all():
        raise ValueError('nonfinite protected prediction or observed target')
    scores = score_arrays(prediction, target, mask)
    scores['channel_mae'] = (np.where(mask, np.abs(prediction.astype(np.float64) - target), 0).sum(axis=(0, 1))
                             / mask.sum(axis=(0, 1))).tolist()
    return scores


def _fit_baselines(data, basis, local, penalty):
    import numpy as np
    from masked_linear import fit_masked_factor, fit_masked_shared
    from threadpoolctl import threadpool_limits
    origins = data['train_origins']
    if (not len(origins) or np.any(origins < max(512, int(data['train_start'])))
            or np.any(origins + 48 > int(data['train_end']))):
        raise ValueError('baseline fit must use legal TRAIN target origins only')
    x, y, mask = _windows(data, origins)
    fits, records = {}, {}
    with threadpool_limits(limits=2):
        for arm in ('shared_linear', 'factor_linear'):
            used = mask.any(axis=2) if arm == 'shared_linear' else mask.all(axis=2)
            used_counts = used.sum(axis=0)
            row = dict(train_origins=int(len(origins)), train_origins_sha256=_origin_hash(origins),
                       used_train_origins=int(used.any(axis=1).sum()),
                       used_train_origins_sha256=_origin_hash(origins[used.any(axis=1)]),
                       used_origin_counts_per_horizon=used_counts.tolist(),
                       used_origin_hashes_per_horizon=[_origin_hash(origins[used[:, h]]) for h in range(48)],
                       penalty=penalty, cpu_threads=2,
                       mask_rule='individual channel targets' if arm == 'shared_linear' else 'fully observed channel vectors')
            if arm == 'factor_linear' and np.any(used_counts == 0):
                row.update(status='unavailable', reason='no fully observed TRAIN target vector for horizons '
                           + str(np.flatnonzero(used_counts == 0).tolist()), fit_seconds=0.0)
                records[arm] = row
                continue
            started = time.perf_counter()
            fit = (fit_masked_shared(x, y, mask, penalty) if arm == 'shared_linear'
                   else fit_masked_factor(x, y, mask, basis, penalty))
            row.update(status='complete', fit_seconds=time.perf_counter() - started, loss=fit['loss'])
            arrays = dict(train_origins=origins, used_origin_mask=used)
            if arm == 'shared_linear':
                arrays.update(weight=fit['weight'], bias=fit['bias'], target_counts=fit['target_counts'])
                row['target_counts_per_horizon'] = fit['target_counts'].tolist()
            else:
                arrays.update(basis=fit['basis'], common_weight=fit['common']['weight'],
                              common_bias=fit['common']['bias'], residual_weight=fit['residual']['weight'],
                              residual_bias=fit['residual']['bias'])
                row['common_target_counts_per_horizon'] = fit['common']['target_counts'].tolist()
                row['residual_target_counts_per_horizon'] = fit['residual']['target_counts'].tolist()
            weights_path = local / f'{arm}_weights.npz'
            row.update(weights_cache=weights_path.name, weights_sha256=_save_npz(weights_path, **arrays))
            records[arm], fits[arm] = row, fit
    return fits, records


def _effects(predictions, specs, target, mask, block):
    from evaluate_development import paired_block_effect
    groups = {}
    for spec in specs:
        groups.setdefault(spec['arm'], {})[spec['seed']] = predictions[spec['id']]
    residual = groups.get('residual', {})
    if not residual:
        return {}
    seeds = sorted(residual)
    effects = {}
    references = {arm: [predictions[arm]] for arm in ('f0', 'shared_linear', 'factor_linear', 'seasonal24')
                  if arm in predictions}
    for arm, forecasts in groups.items():
        if arm == 'residual':
            continue
        if set(forecasts) != set(seeds):
            effects[arm] = dict(status='unavailable', reason='sealed arms do not have matching complete training seeds')
        else:
            references[arm] = [forecasts[seed] for seed in seeds]
    for arm, reference in references.items():
        try:
            effect = paired_block_effect([residual[seed] for seed in seeds], reference, target, mask, block=block)
        except ValueError as exc:
            if 'no targets' not in str(exc) and 'zero-loss' not in str(exc):
                raise
            effects[arm] = dict(status='unavailable', reason=str(exc))
            continue
        except ZeroDivisionError:
            effects[arm] = dict(status='unavailable', reason='relative seed effect undefined for a zero-loss reference')
            continue
        effect.update(status='available', paired_seeds=seeds,
                      scope='conditional non-circular moving time blocks, all channels together; trained seeds fixed; selection uncertainty and pretraining overlap not represented')
        effects[arm] = effect
    return effects


def _evaluate_plan(plan, local, record):
    if plan['here'].resolve() != HERE or plan['root'].resolve() != ROOT:
        raise ValueError('runtime evaluation must use this experiment checkout; custom paths are for preflight only')
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    from evaluate_development import measured_evaluate
    from masked_linear import predict_factor, predict_shared
    from model import make_model, pca_basis
    from run import GPUJob, load_data

    data = load_data('bull')
    if Path(data['_path']).resolve() != plan['data_path'].resolve():
        raise ValueError('loader path differs from preflight data path')
    configuration = plan['seal']['evaluation']
    periods, predictions, period_arrays = {}, {}, {}
    cpu_started = time.perf_counter()
    with threadpool_limits(limits=2):
        basis = pca_basis(data['x'][int(data['train_start']):int(data['train_end'])], configuration['latent'])
        fits, baseline_records = _fit_baselines(data, basis, local, configuration['linear_penalty'])
    cpu_fit_seconds = time.perf_counter() - cpu_started
    record.update(cpu_baseline_fit_seconds=cpu_fit_seconds, baselines=baseline_records)
    _write_json(plan['here'] / ATTEMPT_NAME, record)

    def add_prediction(period, key, arm, fit_id, seed, prediction, cost=None):
        origins, target, mask = period_arrays[period]
        path = local / f'{period}_{key}.npz'
        prediction_hash = _save_npz(path, prediction=prediction, origins=origins)
        row = dict(period=period, arm=arm, fit=fit_id, seed=seed,
                   scores=_scores(prediction, target, mask), prediction_cache=path.name,
                   prediction_sha256=prediction_hash)
        if cost is not None:
            row['evaluation_cost'] = cost
        predictions[period][key] = prediction
        record['partial_rows'].append(row)
        _write_json(plan['here'] / ATTEMPT_NAME, record)

    cpu_prediction_started = time.perf_counter()
    for period in configuration['periods']:
        origins = data[f'eval_{period}_origins']
        if len(origins) < configuration['block_origins']:
            raise ValueError(f'{period} has fewer origins than the sealed block length')
        x, target, mask = _windows(data, origins)
        period_arrays[period] = origins, target, mask
        predictions[period] = {}
        periods[period] = dict(origins=int(len(origins)), origins_sha256=_origin_hash(origins),
                               target_counts=mask.sum(axis=(0, 1)).tolist())
        with threadpool_limits(limits=2):
            for arm, fit in fits.items():
                prediction = predict_shared(x, fit) if arm == 'shared_linear' else predict_factor(x, fit)
                add_prediction(period, arm, arm, None, None, prediction)
        prediction = np.tile(x[:, -configuration['seasonal_period']:, :], (1, 2, 1))
        add_prediction(period, 'seasonal24', 'seasonal24', None, None, prediction)
    cpu_prediction_seconds = time.perf_counter() - cpu_prediction_started

    gpu_started = time.perf_counter()
    with GPUJob('protected_evaluation') as job:
        model = make_model('f0', basis)
        try:
            for period in configuration['periods']:
                _, prediction, cost = measured_evaluate(model, data, period_arrays[period][0], job, f'F0 protected {period}')
                add_prediction(period, 'f0', 'f0', None, None, prediction, cost)
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        for spec, result in zip(plan['seal']['allowed_fits'], plan['results']):
            saved = torch.load(result['checkpoint'], map_location='cpu', weights_only=False)
            if not _same_json(saved['spec'], spec):
                raise ValueError(f'checkpoint spec differs from seal: {spec["id"]}')
            if tuple(saved['basis'].shape) != (data['x'].shape[1], spec['latent']):
                raise ValueError(f'checkpoint basis shape differs from seal: {spec["id"]}')
            model = make_model(spec['arm'], saved['basis'], residual_rank=spec.get('residual_rank', 8))
            try:
                model.restore_adapter(saved['state'])
                for period in configuration['periods']:
                    _, prediction, cost = measured_evaluate(model, data, period_arrays[period][0], job, spec['id'] + ' protected ' + period)
                    add_prediction(period, spec['id'], spec['arm'], spec['id'], spec['seed'], prediction, cost)
            finally:
                del model, saved
                gc.collect()
                torch.cuda.empty_cache()
    gpu_seconds = time.perf_counter() - gpu_started
    for period in configuration['periods']:
        _, target, mask = period_arrays[period]
        periods[period]['residual_vs'] = _effects(predictions[period], plan['seal']['allowed_fits'], target, mask,
                                                 configuration['block_origins'])
        if baseline_records['factor_linear']['status'] == 'unavailable':
            periods[period]['residual_vs']['factor_linear'] = dict(status='unavailable', reason=baseline_records['factor_linear']['reason'])
    return dict(status='complete', scope='Bull Office local protected confirmation; E1/E2 are two sequential periods of one source, not independent datasets; pretraining overlap unknown',
                protected_receipt=plan['receipt'], evaluation=configuration,
                evaluation_source_hashes=plan['evaluation_source_hashes'], result_hashes=plan['result_hashes'],
                adaptation_receipt_sha256=plan['adaptation_receipt_sha256'],
                data_sha256=plan['hashes']['protected_npz'], channels=data['columns'].tolist(),
                baselines=baseline_records, rows=record['partial_rows'], periods=periods,
                cpu_baseline_fit_seconds=cpu_fit_seconds, cpu_baseline_prediction_seconds=cpu_prediction_seconds,
                gpu_job_wall_seconds=gpu_seconds, gpu_accounted_seconds=job.record['elapsed_s'],
                gpu_time_scope='model loading/restoration, GPU inference, transfers, metrics and persistence; CPU baseline fits and paired bootstrap excluded')


if __name__ == '__main__':
    report = run_evaluation()
    print(json.dumps({'status': report['status'], 'periods': report['periods']}, ensure_ascii=False))
