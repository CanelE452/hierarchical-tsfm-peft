"""Independent CPU audits; no experiment imports, training or network access.

EVAL-bearing predictions are refused without explicit post-evaluation audit
authorization. Stored checkpoint scores are checked for choice consistency;
this alone does not independently verify every checkpoint's forward output.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
H, L = 48, 512
SEEDS = (92601, 92602)
ARCHITECTURES = ('DIRECT_PCA', 'ANCHOR_PCA', 'DIRECT_TEACHER', 'ANCHOR_TEACHER',
                 'ANCHOR_POINTWISE', 'ANCHOR_NONLINEAR_D')
METHODS = ('BASE_Q', 'LATENT_FUTURE_MSE', 'SUMMARY_MSE', 'SHUFFLED_SUMMARY',
           'FIXED_REGIME', 'LEARNED_REGIME', 'SHUFFLED_REGIME')
LAMBDAS = (1e-4, 1e-2, 1., 1e2, 1e4)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def checked_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if path.name.lower() in ('test.npz', 'jena_test.npz'):
        raise PermissionError('Protected legacy TEST must not be opened')
    return path


def require_terminal(result):
    if result.get('status') not in ('COMPLETE', 'PARTIAL_FAILED_PATHS', 'FAILED_PRESERVED',
                                    'FAILED_PATH_PRESERVED', 'EVALUATION_COMPLETE',
                                    'COMPLETE_RESOLVED_WITH_DECLARED_PROTOCOL_DEVIATION', 'PARTIAL_RESOLVED'):
        raise ValueError('Only terminal receipts may be audited')


def require_eval_authorization(authorized):
    if not authorized:
        raise PermissionError('EVAL audit requires root authorization after the first evaluation pass')


def authorized_for(authorized, dataset):
    return bool(authorized) if isinstance(authorized, bool) else dataset in authorized


def independent_score(prediction, target, observed):
    prediction, target = np.asarray(prediction), np.asarray(target)
    observed = np.asarray(observed, dtype=bool)
    if prediction.shape != target.shape or observed.shape != target.shape or target.ndim != 3:
        raise ValueError('Forecast/mask shapes disagree')
    if not np.isfinite(prediction).all() or not np.isfinite(target[observed]).all():
        raise ValueError('Nonfinite scored values')
    counts, mses, maes = [], [], []
    for c in range(target.shape[2]):
        mask = observed[:, :, c]
        errors = prediction[:, :, c][mask].astype(np.float64) - target[:, :, c][mask].astype(np.float64)
        counts.append(int(mask.sum()))
        mses.append(float(np.mean(errors ** 2)) if errors.size else 0.)
        maes.append(float(np.mean(np.abs(errors))) if errors.size else 0.)
    usable = np.asarray(counts) > 0
    if not usable.any():
        raise ValueError('No observed evaluation target')
    result = {'mse': float(np.asarray(mses)[usable].mean()), 'mae': float(np.asarray(maes)[usable].mean()),
              'per_channel_mse': mses, 'per_channel_mae': maes, 'observed_counts': counts,
              'horizon_metrics': {}}
    if target.shape[1] == H:
        for name, lo, hi in (('1_8', 0, 8), ('9_24', 8, 24), ('25_48', 24, 48)):
            result['horizon_metrics'][name] = independent_score(prediction[:, lo:hi], target[:, lo:hi], observed[:, lo:hi])
    return result


def independent_checkpoint_choice(checkpoints):
    if not checkpoints:
        raise ValueError('No checkpoints')
    values = [(float(cp['forecast']['tune']['normalized']['mse']), int(cp['update'])) for cp in checkpoints]
    if not np.isfinite([x[0] for x in values]).all():
        raise ValueError('Nonfinite TUNE checkpoint score')
    return min(values)[1]


def complete_development_coverage(cases):
    pairs = [(c['dataset'], c['seed']) for c in cases]
    return len(pairs) == 4 and set(pairs) == {(d, s) for d in ('jena', 'robin') for s in SEEDS}


def independent_window_fill(physical, observed, mean, std):
    physical, observed = np.asarray(physical), np.asarray(observed, bool)
    result = np.empty_like(physical, dtype=np.float32)
    for i in range(len(physical)):
        last = np.asarray(mean, np.float32).copy()
        for t in range(physical.shape[1]):
            last[observed[i, t]] = physical[i, t, observed[i, t]]
            result[i, t] = (last - mean) / std
    return result


def independent_fit_statistics(physical, observed, bounds):
    lo, hi = bounds
    masked = np.where(observed[lo:hi], physical[lo:hi], np.nan)
    mean = np.nanmean(masked, axis=0).astype(np.float32)
    std = np.nanstd(masked, axis=0).astype(np.float32)
    return mean, np.where(std > 1e-6, std, 1.).astype(np.float32)


def independent_raw_source(audit, dataset, contract):
    import io
    import zipfile
    import pandas as pd
    source = audit.asset(contract['source'], dataset + ':raw_source_SHA')
    member = contract['source']['parse']['source_member']
    if ',' in member:
        raise ValueError('Raw audit requires one unambiguous declared CSV member')
    if source.suffix.lower() == '.csv':
        content = source.read_bytes()
        audit.check(dataset + ':raw_source_member', source.name == member)
    else:
        names = member.split('!')
        with zipfile.ZipFile(source) as archive:
            content = archive.read(names[0])
        for name in names[1:]:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                content = archive.read(name)
    frame = pd.read_csv(io.BytesIO(content))
    spec = contract['spec']
    if dataset == 'appliances':
        timestamps = pd.to_datetime(frame['date'], errors='raise')
    elif dataset == 'beijing_aotizhongxin':
        audit.check(dataset + ':raw_station_matches_contract',
                    frame['station'].astype(str).eq(spec['station']).all())
        timestamps = pd.to_datetime({key: frame[key] for key in ('year', 'month', 'day', 'hour')}, errors='raise')
    else:
        raise ValueError('Unsupported independent raw source: ' + dataset)
    order = np.argsort(timestamps.to_numpy())
    frame, timestamps = frame.iloc[order].reset_index(drop=True), timestamps.iloc[order].reset_index(drop=True)
    values = frame[spec['channels']].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float32)
    observed = np.isfinite(values)
    physical = np.where(observed, values, np.nan).astype(np.float32)
    times = timestamps.dt.strftime('%Y-%m-%dT%H:%M:%S').to_numpy()
    intervals = timestamps.diff().dropna().dt.total_seconds().to_numpy()
    audit.check(dataset + ':raw_rows', len(physical) == contract['source']['parse']['rows'])
    audit.check(dataset + ':raw_unique_timestamps', not timestamps.duplicated().any())
    audit.check(dataset + ':raw_interval_contract', np.all(intervals == spec['interval_seconds']))
    audit.check(dataset + ':raw_physical_reduction_layout', physical.flags.f_contiguous,
                {'source_strides': list(physical.strides),
                 'note': 'Pandas column storage is preserved for the FP32 FIT reduction; NPZ physical is C-contiguous'})
    return physical, observed, times


def independent_coordinates(target, observed, last, basis):
    n, horizon, _ = target.shape
    k = basis.shape[1]
    result, valid = np.zeros((n, horizon, k), np.float32), np.zeros((n, horizon), bool)
    for i in range(n):
        for h in range(horizon):
            mask = np.asarray(observed[i, h], bool)
            b = np.asarray(basis[mask], np.float64)
            if mask.sum() < k or np.linalg.matrix_rank(b, tol=1e-8) < k:
                continue
            response = (target[i, h, mask] - last[i, mask]).astype(np.float64)
            result[i, h] = np.linalg.lstsq(b, response, rcond=1e-15)[0]
            valid[i, h] = True
    return result, valid


def independent_summary(coordinates, valid):
    n, horizon, k = coordinates.shape
    values, masks = np.zeros((n, 3 * k), np.float32), np.zeros((n, 3 * k), bool)
    for i in range(n):
        times = np.flatnonzero(valid[i]).astype(np.float64)
        if times.size:
            values[i, :k] = coordinates[i, valid[i]].mean(0)
            masks[i, :k] = True
        if valid[i, -1]:
            values[i, k:2 * k] = coordinates[i, -1]
            masks[i, k:2 * k] = True
        if times.size >= 2:
            centered = times - times.mean()
            values[i, 2 * k:] = (centered[:, None] * coordinates[i, valid[i]]).sum(0) / np.dot(centered, centered)
            masks[i, 2 * k:] = True
    return values, masks


def checkpoint_coverage(records, expected):
    from collections import Counter
    counts = Counter((r['dataset'], r['label'], int(r['update'])) for r in records)
    required = {(dataset, label, int(update)) for (dataset, label), updates in expected.items() for update in updates}
    missing = sorted(required - counts.keys())
    extra = sorted(counts.keys() - required)
    duplicates = sorted(key for key, count in counts.items() if count != 1)
    return {'complete': not missing and not extra and not duplicates,
            'missing': missing, 'extra': extra, 'duplicates': duplicates, 'expected_count': len(required), 'actual_count': len(records)}


def independent_q_apply(before, weight, bias):
    return np.asarray(before) @ np.asarray(weight).T + np.asarray(bias)


def independent_alignment_diagnostics(value, target, valid, latent_std, means, stds, summary_target, complete):
    errors = ((np.asarray(value, np.float64) - target) / latent_std) ** 2
    latent = float(errors[np.asarray(valid, bool)].mean()) if np.asarray(valid).any() else None
    summary, _ = independent_summary(value, valid)
    summary_errors = ((summary - means) / stds)[complete].astype(np.float64) - summary_target[complete]
    return {'latent_future_normalized_mse': latent,
            'complete_summary_normalized_mse': float(np.mean(summary_errors ** 2)) if np.asarray(complete).any() else None}


class Audit:
    def __init__(self):
        self.checks, self.completion_checks, self.started = [], [], time.perf_counter()

    def check(self, name, condition, detail=None):
        if time.perf_counter() - self.started > 7200:
            raise RuntimeError('CPU audit wall cap exceeded')
        self.checks.append({'name': name, 'pass': bool(condition), 'detail': detail})

    def close(self, name, actual, expected, rtol=2e-9, atol=1e-9):
        actual, expected = np.asarray(actual), np.asarray(expected)
        equal = actual.shape == expected.shape and np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False)
        delta = float(np.max(np.abs(actual.astype(float) - expected.astype(float)))) if actual.size and actual.shape == expected.shape else None
        self.check(name, equal, {'max_abs': delta, 'rtol': rtol, 'atol': atol})

    def scope(self, name, condition, detail=None):
        self.completion_checks.append({'name': name, 'complete': bool(condition), 'detail': detail})

    def asset(self, record, name):
        path = checked_path(record['path'])
        actual = file_hash(path) if path.is_file() else None
        self.check(name, actual == record['sha256'], {'path': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                   'expected': record['sha256'], 'actual': actual})
        return path


def audit_score(audit, label, actual, declared, rtol=2e-9, atol=1e-9):
    for key in ('mse', 'mae', 'per_channel_mse', 'per_channel_mae', 'observed_counts'):
        audit.close(label + ':' + key, actual[key], declared[key], rtol=rtol, atol=atol)
    for name, value in actual['horizon_metrics'].items():
        audit_score(audit, label + ':horizon_' + name, value, declared['horizon_metrics'][name], rtol=rtol, atol=atol)


def legacy_data(record):
    source = checked_path(record['source']['path'])
    with np.load(source, allow_pickle=False) as z:
        physical = (z['x'] * z['std'] + z['mean']).astype(np.float32)
        mask = z['finite'].astype(bool)
        physical[~mask] = np.nan
        saved_origins = z['val_origins'].astype(np.int64)
    lo, hi = record['bounds']['fit']
    mean = np.nanmean(physical[lo:hi], axis=0).astype(np.float32)
    std = np.nanstd(physical[lo:hi], axis=0).astype(np.float32)
    std = np.where(std > 1e-6, std, 1.).astype(np.float32)
    filled = physical.copy()
    last = mean.copy()
    for t in range(len(physical)):
        last[mask[t]] = physical[t, mask[t]]
        filled[t] = last
    x = ((filled - mean) / std).astype(np.float32)
    origins = {}
    for split, count in (('fit', 256), ('tune', 64)):
        start, stop = record['bounds'][split]
        possible = np.arange(start + L if split == 'fit' else start, stop - H + 1, dtype=np.int64)
        origins[split] = possible[np.unique(np.linspace(0, len(possible) - 1, min(count, len(possible))).astype(int))]
    origins['val'] = saved_origins
    return {'physical': physical, 'mask': mask, 'x': x, 'mean': mean, 'std': std, 'origins': origins}


def expected_targets(data, split):
    if split in data.get('targets', {}):
        return data['targets'][split], data['target_masks'][split]
    oo = data['origins'][split]
    return np.stack([data['x'][o:o + H] for o in oo]), np.stack([data['mask'][o:o + H] for o in oo])


def result_lineage(audit, result_path):
    result = read(result_path)
    refs = result.get('source_results', [])
    sources = []
    for ref in refs:
        path = audit.asset(ref, 'lineage:' + result_path.parent.name + ':' + Path(ref['path']).parent.name)
        value = read(path)
        require_terminal(value)
        sources.append((path, value))
    return sources


def dataset_owner(audit, result_path, dataset):
    receipt = result_path.parent / (dataset + '_EVAL_FIRST_OPEN.json')
    if receipt.is_file():
        return result_path
    for path, source in result_lineage(audit, result_path):
        if dataset in source.get('evaluation_open_receipts', {}) or (path.parent / (dataset + '_EVAL_FIRST_OPEN.json')).is_file():
            return path
    raise FileNotFoundError('Dataset first-open receipt not present in resolved lineage: ' + dataset)


def fresh_data(audit, dataset, result_path, authorized):
    method_path = HERE / 'METHOD_LOCK.json'
    method = read(method_path)
    method_hash = file_hash(method_path)
    registry_name = dataset + '_' + method_hash + '.json'
    registry_candidates = [base / 'contract' / registry_name for base in
                           (ROOT / '.cache' / HERE.name, ROOT / '.cache' / HERE.name / 'beijing_repair')]
    source_result = read(result_path)
    declared = source_result.get('newdata_contracts', {}).get(dataset, {})
    if not declared and source_result['status'] in ('COMPLETE_RESOLVED_WITH_DECLARED_PROTOCOL_DEVIATION', 'PARTIAL_RESOLVED'):
        for _, source in result_lineage(audit, result_path):
            declared = source.get('newdata_contracts', {}).get(dataset, {})
            if declared:
                break
    registry_candidates = [p for p in registry_candidates if p.is_file() and
                           (not declared or read(p)['contract_sha256'] == declared['contract_sha256'])]
    if len(registry_candidates) != 1:
        raise ValueError('Exactly one source-bound dataset registry required: ' + dataset)
    registry_path = registry_candidates[0]
    registry = read(registry_path)
    contract_path = checked_path(registry['contract_path'])
    audit.check(dataset + ':contract_SHA', file_hash(contract_path) == registry['contract_sha256'])
    contract = read(contract_path)
    audit.check(dataset + ':method_lock_precedes_ingest', method['created_unix'] <= contract['created_unix'])
    audit.check(dataset + ':method_lock_hash', contract['method_lock']['sha256'] == method_hash)
    path = audit.asset(contract['payloads']['fit_tune'], dataset + ':FIT_TUNE_SHA')
    with np.load(path, allow_pickle=False) as z:
        data = {key: z[key].copy(order='K') for key in ('physical', 'mask', 'mean', 'std', 'x', 'columns', 'times')}
        data['origins'] = {split: z['origins_' + split].astype(np.int64) for split in ('fit', 'tune')}
    audit.check(dataset + ':FIT_TUNE_channel_order', data['columns'].tolist() == contract['spec']['channels'])
    raw_physical = raw_mask = raw_times = None
    if authorized_for(authorized, dataset):
        owner = dataset_owner(audit, result_path, dataset)
        audit_new_lock(audit, dataset, owner, contract, registry['contract_sha256'])
        raw_physical, raw_mask, raw_times = independent_raw_source(audit, dataset, contract)
        end = contract['bounds']['tune'][1]
        audit.check(dataset + ':raw_FIT_TUNE_values', np.array_equal(data['physical'], raw_physical[:end], equal_nan=True))
        audit.close(dataset + ':raw_FIT_TUNE_mask', data['mask'], raw_mask[:end], rtol=0, atol=0)
        audit.check(dataset + ':raw_FIT_TUNE_times', np.array_equal(data['times'], raw_times[:end]))
    statistics_physical = raw_physical if raw_physical is not None else np.asfortranarray(data['physical'])
    statistics_mask = raw_mask if raw_mask is not None else np.asfortranarray(data['mask'])
    mean, std = independent_fit_statistics(statistics_physical, statistics_mask, contract['bounds']['fit'])
    c_mean, c_std = independent_fit_statistics(np.ascontiguousarray(data['physical']), data['mask'], contract['bounds']['fit'])
    audit.check(dataset + ':FIT_statistics_numeric_provenance', True,
                {'raw_source_audited': raw_physical is not None,
                 'declared_parser_layout': 'F-contiguous column matrix before storage',
                 'C_layout_mean_max_abs': float(np.max(np.abs(c_mean-mean))),
                 'C_layout_std_max_abs': float(np.max(np.abs(c_std-std))),
                 'statistics_scope': 'FIT rows only; FP32 source layout reduction'})
    audit.close(dataset + ':FIT_only_mean', data['mean'], mean, rtol=0, atol=0)
    audit.close(dataset + ':FIT_only_std', data['std'], std, rtol=0, atol=0)
    audit.check(dataset + ':training_no_EVAL_rows', len(data['physical']) == contract['bounds']['tune'][1])
    for split, maximum in (('fit', 256), ('tune', 64)):
        lo, hi = contract['bounds'][split]
        possible = np.arange(max(L, lo+L) if split == 'fit' else max(L, lo), hi-H+1, dtype=np.int64)
        expected = possible[np.unique(np.linspace(0, len(possible)-1, min(maximum, len(possible))).astype(int))]
        audit.close(dataset + ':independent_origins_' + split, data['origins'][split], expected, rtol=0, atol=0)
        audit.check(dataset + ':origins_SHA_' + split, array_hash(expected) == contract['origins'][split]['sha256'])
    normalized = independent_window_fill(data['physical'][None], data['mask'][None], mean, std)[0]
    audit.close(dataset + ':global_FF_normalization', data['x'], normalized, rtol=0, atol=0)
    if authorized_for(authorized, dataset):
        require_eval_authorization(True)
        target_path = audit.asset(contract['payloads']['eval_future_target'], dataset + ':EVAL_target_SHA')
        input_path = audit.asset(contract['payloads']['eval_past_input'], dataset + ':EVAL_past_SHA')
        with np.load(target_path, allow_pickle=False) as z:
            target, physical, masks, origins = z['target'], z['target_physical'], z['target_mask'], z['origins'].astype(np.int64)
            audit.close(dataset + ':EVAL_target_mean', z['mean'], mean, rtol=0, atol=0)
            audit.close(dataset + ':EVAL_target_std', z['std'], std, rtol=0, atol=0)
            audit.check(dataset + ':EVAL_target_channel_order', np.array_equal(z['columns'], data['columns']))
        with np.load(input_path, allow_pickle=False) as z:
            history, history_physical, history_mask = z['history'], z['physical_history'], z['history_mask']
            audit.close(dataset + ':EVAL_input_target_origins', z['origins'], origins, rtol=0, atol=0)
            audit.close(dataset + ':EVAL_input_mean', z['mean'], mean, rtol=0, atol=0)
            audit.close(dataset + ':EVAL_input_std', z['std'], std, rtol=0, atol=0)
            audit.check(dataset + ':EVAL_input_channel_order', np.array_equal(z['columns'], data['columns']))
            audit.check(dataset + ':raw_EVAL_input_all_timestamps', np.array_equal(z['times'], raw_times))
        data['origins']['eval'] = origins
        data['targets'] = {'eval': target}
        data['target_masks'] = {'eval': masks}
        audit.check(dataset + ':raw_EVAL_target_values', np.array_equal(physical,
                    np.stack([raw_physical[o:o+H] for o in origins]), equal_nan=True))
        audit.close(dataset + ':raw_EVAL_target_mask', masks, np.stack([raw_mask[o:o+H] for o in origins]), rtol=0, atol=0)
        audit.check(dataset + ':raw_EVAL_history_values', np.array_equal(history_physical,
                    np.stack([raw_physical[o-L:o] for o in origins]), equal_nan=True))
        audit.close(dataset + ':raw_EVAL_history_mask', history_mask, np.stack([raw_mask[o-L:o] for o in origins]), rtol=0, atol=0)
        expected = ((np.where(masks, physical, mean).astype(np.float32)-mean)/std).astype(np.float32)
        audit.close(dataset + ':EVAL_observed_target_normalization', target, expected, rtol=0, atol=0)
        audit.close(dataset + ':EVAL_window_FF_reset', history,
                    independent_window_fill(history_physical, history_mask, mean, std), rtol=0, atol=0)
        lo, hi = contract['bounds']['eval']
        expected_origins = np.arange(max(L, lo), hi-H+1, H, dtype=np.int64)
        audit.close(dataset + ':EVAL_all_48stride_origins', origins, expected_origins, rtol=0, atol=0)
        audit.check(dataset + ':EVAL_horizons_nonoverlap', np.all(np.diff(origins) >= H))
    return data


def audit_new_lock(audit, dataset, result_path, contract, contract_hash):
    folder = result_path.parent
    global_path = folder / 'MODEL_SELECTION_LOCK.json'
    per_path = folder / (dataset + '_MODEL_SELECTION_LOCK.json')
    open_path = folder / (dataset + '_EVAL_FIRST_OPEN.json')
    glob, per, opened = read(global_path), read(per_path), read(open_path)
    prefix = dataset + ':EVAL_LOCK'
    audit.check(prefix + ':global_status', glob['status'] == 'ALL_NEW_MODELS_SELECTED_BEFORE_ANY_EVAL_TARGET_OPEN')
    audit.check(prefix + ':perdataset_status', per['status'] == 'MODEL_SELECTION_LOCKED_BEFORE_EVAL_TARGET_OPEN')
    audit.check(prefix + ':temporal_order', glob['created_unix'] <= per['created_unix'] <= opened['first_and_only_open_started_unix'] <= opened['open_completed_unix'])
    audit.check(prefix + ':perdataset_SHA', file_hash(per_path) == opened['model_selection_lock_sha256'])
    audit.check(prefix + ':global_SHA', file_hash(global_path) == per['global_model_selection_lock']['sha256'])
    payload = per['newdata_contract']
    audit.check(prefix + ':contract_SHA', payload['contract_sha256'] == contract_hash)
    audit.check(prefix + ':method_SHA', payload['method_lock_sha256'] == contract['method_lock']['sha256'])
    audit.check(prefix + ':source_SHA', payload['source_sha256'] == contract['source']['sha256'])
    audit.check(prefix + ':target_payload_SHA', opened['eval_target_payload_sha256'] == contract['payloads']['eval_future_target']['sha256'])
    audit.scope(prefix + ':original_joint_two_dataset_barrier', set(glob['newdata_contracts']) == {'appliances', 'beijing_aotizhongxin'},
                {'datasets_actually_sealed': sorted(glob['newdata_contracts']),
                 'note': 'A valid per-dataset first-open lock does not establish the original joint two-dataset barrier'})
    for i, artifact in enumerate(payload['selected_models']):
        audit.asset(artifact, prefix + ':model_' + str(i))
        audit.check(prefix + ':model_before_open_' + str(i), checked_path(artifact['path']).stat().st_mtime <= opened['first_and_only_open_started_unix'])


def audit_cost_result(audit, path):
    from collections import Counter
    path = checked_path(path)
    result = read(path)
    require_terminal(result)
    audit.scope('cost:terminal_complete', result['status'] == 'COMPLETE', result['status'])
    audit.check('cost:no_training', result['new_training_updates'] == 0)
    audit.check('cost:backbone_unchanged', result['bolt_initial_hash'] == result['bolt_final_hash'])
    for source, digest in result['source_hashes'].items():
        audit.check('cost:source:' + Path(source).name, file_hash(checked_path(source)) == digest)
    cases = {}
    for record in result['source_results']:
        source_path = audit.asset(record, 'cost:source_result_SHA')
        source = read(source_path)
        require_terminal(source)
        cases.update({(case['dataset'], case['label']): case for case in source['cases']})
    required = {(d, label, method['method']) for (d, label), case in cases.items() if case.get('stage') != 'RAW'
                for method in case['methods']}
    counted = Counter((r['dataset'], r['label'], r['method']) for r in result['records'])
    audit.check('cost:unique_source_bound_records', all(v == 1 for v in counted.values()) and not (counted.keys()-required))
    audit.scope('cost:full_200_method_scope', len(required) == 200 and counted.keys() == required,
                {'expected': len(required), 'actual': len(result['records']), 'missing': sorted(required-counted.keys())})
    for i, record in enumerate(result['records']):
        prefix = 'cost:' + record['dataset'] + ':' + record['label'] + ':' + record['method']
        case = cases[record['dataset'], record['label']]
        method = next(m for m in case['methods'] if m['method'] == record['method'])
        audit.check(prefix + ':source_prediction_binding', record['prediction'] == method['prediction'])
        selected = case.get('selected_models', case.get('training', {}).get('selected_checkpoint'))
        audit.check(prefix + ':source_model_binding', record['selected_models'] == selected)
        if selected is not None:
            audit.asset(selected, prefix + ':selected_model_SHA')
        science_path = audit.asset(record['prediction'], prefix + ':prediction_SHA')
        split = 'val' if record['dataset'] in ('jena', 'robin') else 'eval'
        with np.load(science_path, allow_pickle=False) as z:
            reference = z[split + '_prediction'] * z['std'] + z['mean']
        timings = record['latency']
        audit.check(prefix + ':both_batches', [r['batch'] for r in timings] == [1, 8])
        for row in timings:
            label = prefix + ':batch' + str(row['batch'])
            values = np.asarray(row['seconds'])
            audit.check(label + ':measurement_contract', row['warmup'] == 2 and row['repeats'] == 7 and len(values) == 7)
            audit.check(label + ':positive_finite_times', np.isfinite(values).all() and np.all(values > 0))
            audit.close(label + ':median', np.median(values), row['median_seconds'], rtol=0, atol=0)
            audit.close(label + ':per_window', np.median(values)/row['batch'], row['median_seconds_per_window'], rtol=0, atol=0)
            audit.check(label + ':prediction_parity', row['parity_pass'])
            declared_diff = row['physical_max_abs_vs_science']
            necessary_bound = float(np.max(2e-3 + 2e-5*np.abs(reference[:row['batch']])))
            audit.check(label + ':finite_diff_within_necessary_parity_bound',
                        np.isfinite(declared_diff) and 0 <= declared_diff <= necessary_bound,
                        {'recorded_max_abs': declared_diff, 'necessary_max_tolerance': necessary_bound,
                         'claim_limit': 'Recorded maximum and source are checked; timed output arrays were not saved'})
            audit.check(label + ':RAM_VRAM_caps', max(row['peak_allocated_vram_bytes'], row['resident_rss_bytes']) <= 8*1024**3)
    aliases = result.get('canonical_aliases', [])
    audit.scope('cost:both_RAW_aliases', {r['dataset'] for r in aliases} == {'jena', 'robin'})
    for alias in aliases:
        first = cases[alias['dataset'], alias['label']]
        canonical = cases[alias['dataset'], alias['canonical_label']]
        a = next(m for m in first['methods'] if m['method'] == alias['method'])
        b = next(m for m in canonical['methods'] if m['method'] == alias['canonical_method'])
        a_path, b_path = audit.asset(a['prediction'], 'cost:alias:original_SHA'), audit.asset(b['prediction'], 'cost:alias:canonical_SHA')
        with np.load(a_path, allow_pickle=False) as left, np.load(b_path, allow_pickle=False) as right:
            for split in ('fit', 'tune', 'val'):
                audit.check('cost:alias:' + alias['dataset'] + ':' + split,
                            np.array_equal(left[split+'_prediction'], right[split+'_prediction']))
    return {'path': str(path.relative_to(ROOT)), 'sha256': file_hash(path), 'status': result['status'],
            'method_records': len(result['records']), 'latency_rows': sum(len(r['latency']) for r in result['records']),
            'parity_scope': 'Science NPZ/model binding, producer source/receipt and necessary diff bound; timed output arrays unavailable',
            'source_failed_paths': result.get('failed_paths', [])}


def audit_checkpoint_result(audit, path, data, authorized):
    import torch
    path = checked_path(path)
    result = read(path)
    require_terminal(result)
    prefix = 'checkpoint_audit:' + path.parent.name
    audit.scope(prefix + ':terminal_complete', result['status'] == 'COMPLETE', result['status'])
    audit.check(prefix + ':no_training', result['new_training_updates'] == 0)
    audit.check(prefix + ':backbone', result.get('bolt_initial_hash') == result.get('bolt_final_hash') and result.get('bolt_final_hash') is not None)
    for source, digest in result['source_hashes'].items():
        audit.check(prefix + ':source:' + Path(source).name, file_hash(checked_path(source)) == digest)
    source_cases = []
    for source in result['source_results']:
        source_path = audit.asset(source, prefix + ':source_result')
        source_result = read(source_path)
        require_terminal(source_result)
        contracts = dict(source_result.get('newdata_contracts', {}))
        for _, lineage in result_lineage(audit, source_path):
            contracts.update(lineage.get('newdata_contracts', {}))
        for dataset in contracts:
            if dataset not in data:
                data[dataset] = fresh_data(audit, dataset, source_path, authorized)
        source_cases.extend(source_result.get('cases', []))
    cases = {(c['dataset'], c['label']): c for c in source_cases}
    cores = {key: case for key, case in cases.items() if 'training' in case}
    natives = {key: (case, next(m for m in case['methods'] if 'native_training' in m))
               for key, case in cases.items() if any('native_training' in m for m in case['methods'])}
    aligned = {key: case for key, case in cores.items() if 'alignment' in case}
    expected = {key: (0, 100, 200, 400, 600) for key in cores}
    native_expected = {key: (0, 100, 200, 400, 600) for key in natives}
    coverage = checkpoint_coverage(result['records'], expected)
    native_coverage = checkpoint_coverage(result['native_records'], native_expected)
    audit.scope(prefix + ':core_CP_coverage', coverage['complete'], coverage)
    audit.scope(prefix + ':native_CP_coverage', native_coverage['complete'], native_coverage)
    audit.check(prefix + ':core_unique_and_bound', not coverage['duplicates'] and not coverage['extra'], coverage)
    audit.check(prefix + ':native_unique_and_bound', not native_coverage['duplicates'] and not native_coverage['extra'], native_coverage)
    audit.scope(prefix + ':full_goal_scope', len(cores) == 120 and len(natives) == 16 and len(aligned) == 80,
                {'core_models': len(cores), 'native_models': len(natives), 'aligned_models': len(aligned)})
    rescored = {'core': {}, 'native': {}}
    for kind, records in (('core', result['records']), ('native', result['native_records'])):
        for record in records:
            key = (record['dataset'], record['label'])
            label = prefix + ':' + kind + ':' + ':'.join(key) + ':u' + str(record['update'])
            try:
                case = cores[key] if kind == 'core' else natives[key][0]
                training = case['training'] if kind == 'core' else natives[key][1]['native_training']
                cp = next(c for c in training['checkpoints'] if c['update'] == record['update'])
                audit.check(label + ':CP_binding', record['checkpoint']['path'] == cp['path'] and record['checkpoint']['sha256'] == cp['sha256'])
                audit.asset(record['checkpoint'], label + ':CP_SHA')
                npz_path = audit.asset(record['prediction'], label + ':NPZ_SHA')
                d = data[key[0]]
                target, mask = expected_targets(d, 'tune')
                with np.load(npz_path, allow_pickle=False) as z:
                    audit.close(label + ':TUNE_target', z['target'], target, rtol=0, atol=0)
                    audit.check(label + ':TUNE_mask', np.array_equal(z['observed'], mask))
                    audit.close(label + ':TUNE_origins', z['origins'], d['origins']['tune'], rtol=0, atol=0)
                    audit.close(label + ':mean', z['mean'], d['mean'], rtol=0, atol=0)
                    audit.close(label + ':std', z['std'], d['std'], rtol=0, atol=0)
                    normalized = independent_score(z['prediction'], z['target'], z['observed'])
                    physical = independent_score(z['prediction']*z['std']+z['mean'], z['target']*z['std']+z['mean'], z['observed'])
                audit_score(audit, label + ':NPZ_normalized', normalized, record['scores']['normalized'])
                audit_score(audit, label + ':NPZ_physical', physical, record['scores'])
                declared = cp['forecast']['tune'] if kind == 'core' else cp['tune']
                audit_score(audit, label + ':reinference_vs_original_CP', normalized, declared['normalized'], rtol=2e-6, atol=1e-6)
                rescored[kind].setdefault(key, []).append((normalized['mse'], record['update']))
            except Exception as error:
                audit.check(label + ':record_executable', False, str(error))
    choices = []
    for kind, models in (('core', cores), ('native', natives)):
        for key, item in models.items():
            training = item['training'] if kind == 'core' else item[1]['native_training']
            scores = rescored[kind].get(key, [])
            if len(scores) != 5 or {r[1] for r in scores} != {0, 100, 200, 400, 600}:
                audit.scope(prefix + ':' + kind + ':' + ':'.join(key) + ':choice_complete', False, scores)
                continue
            reinference_choice = min(scores)[1]
            chosen = training['selected_update']
            chosen_score = next(s for s, u in scores if u == chosen)
            minimum = min(s for s, _ in scores)
            margin = chosen_score - minimum
            exact = chosen == reinference_choice
            within_rounding = margin <= 1e-6 + 2e-6*abs(minimum)
            audit.check(prefix + ':' + kind + ':' + ':'.join(key) + ':choice_reinference', exact or within_rounding,
                        {'original_choice': chosen, 'reinference_argmin': reinference_choice, 'exact_choice_match': exact,
                         'chosen_minus_minimum': margin, 'numerically_indeterminate': not exact and within_rounding})
            choices.append({'dataset': key[0], 'label': key[1], 'kind': kind, 'original_choice': chosen,
                           'reinference_argmin': reinference_choice, 'exact_match': exact, 'margin': margin})
    alignment_seen = []
    for record in result['alignment_records']:
        key = (record['dataset'], record['label'])
        label = prefix + ':alignment:' + ':'.join(key)
        alignment_seen.append(key)
        try:
            case = aligned[key]
            cp = case['training']['selected_checkpoint']
            audit.check(label + ':selected_CP_binding', record['selected_checkpoint']['path'] == cp['path'] and record['selected_checkpoint']['sha256'] == cp['sha256'])
            source_path = audit.asset(record['alignment_source'], label + ':source_SHA')
            prediction_path = audit.asset(record['prediction'], label + ':NPZ_SHA')
            state = torch.load(source_path, map_location='cpu', weights_only=True)
            selected_state = torch.load(checked_path(cp['path']), map_location='cpu', weights_only=True)
            with np.load(prediction_path, allow_pickle=False) as z:
                audit.close(label + ':q_weight_CP', z['q_weight'], selected_state['q']['weight'].numpy(), rtol=0, atol=0)
                audit.close(label + ':q_bias_CP', z['q_bias'], selected_state['q']['bias'].numpy(), rtol=0, atol=0)
                audit.close(label + ':q_weight_source', z['q_weight'], state['q.weight'].numpy(), rtol=0, atol=0)
                audit.close(label + ':q_bias_source', z['q_bias'], state['q.bias'].numpy(), rtol=0, atol=0)
                for name, buffer in (('target_latent','target'), ('valid','valid'), ('means','means'), ('stds','stds'),
                                     ('latent_std','latent_std'), ('summary_target','summary_target')):
                    audit.close(label + ':buffer_' + name, z[name], state[buffer].numpy(), rtol=0, atol=0)
                metadata = case['alignment']['metadata']
                audit.check(label + ':target_truth_SHA', array_hash(z['target_latent']) == metadata['target_sha256'])
                audit.check(label + ':valid_truth_SHA', array_hash(z['valid']) == metadata['valid_sha256'])
                audit.check(label + ':complete_mask', np.array_equal(z['complete_summary_origins'], state['summary_valid'].numpy().all(1)))
                audit.close(label + ':FIT_origins', z['origins'], data[key[0]]['origins']['fit'], rtol=0, atol=0)
                audit.close(label + ':independent_q_apply', z['latent_fit_post_q'], independent_q_apply(z['latent_fit_pre_q'], z['q_weight'], z['q_bias']), rtol=2e-6, atol=2e-6)
                for name, field in (('before_q','latent_fit_pre_q'), ('after_q','latent_fit_post_q')):
                    calculated = independent_alignment_diagnostics(z[field], z['target_latent'], z['valid'], z['latent_std'],
                                   z['means'], z['stds'], z['summary_target'], z['complete_summary_origins'])
                    for metric, value in calculated.items():
                        metric = name + '_' + metric
                        declared = record['diagnostics'][metric]
                        if value is None:
                            audit.check(label + ':' + metric, declared is None)
                        else:
                            audit.close(label + ':' + metric, value, declared, rtol=2e-6, atol=2e-5)
    
        except Exception as error:
            audit.check(label + ':record_executable', False, str(error))
    audit.scope(prefix + ':alignment_coverage', len(alignment_seen) == len(aligned) and set(alignment_seen) == set(aligned),
                {'expected': len(aligned), 'actual': len(alignment_seen)})
    return {'source': str(path.relative_to(ROOT)), 'source_sha256': file_hash(path), 'source_status': result['status'],
            'core_coverage': coverage, 'native_coverage': native_coverage,
            'alignment_records': len(alignment_seen), 'reinference_choices': choices,
            'source_failed_paths': result.get('failed_paths', [])}


def audit_fixture(audit, result, folder):
    if 'normalizer_fixture' not in result:
        audit.check('fixture:new:backbone_state', result['bolt_initial_hash'] == result['bolt_final_hash'])
        for dataset, fixture in result['fixtures'].items():
            prefix = 'fixture:new:' + dataset
            audit.check(prefix + ':identity_parity', fixture['identity_anchor_vs_raw_max_abs'] < 2e-6,
                        fixture['identity_anchor_vs_raw_max_abs'])
            audit.check(prefix + ':active_causal_prefix', fixture['active_causal_prefix_max_abs'] == 0)
            audit.check(prefix + ':finite_gradients', fixture['finite_input_and_parameter_gradients'])
            audit.check(prefix + ':actual_update', fixture['actual_temp_optimizer_step_max_abs'] > 0)
            audit.check(prefix + ':restore', fixture['state_restored'])
            audit.check(prefix + ':frozen_F', fixture['F_hash_unchanged'])
        return
    audit.check('fixture:mode', result['mode'] in ('preflight', 'dev', 'new'))
    audit.check('fixture:backbone_state_unchanged', result['bolt_initial_hash'] == result['bolt_final_hash'])
    normalizer = result['normalizer_fixture']
    audit.check('normalizer:zero_forward_difference', normalizer['all_forward_max_abs'] == 0)
    audit.check('normalizer:positive_gradient_parity', normalizer['positive_variance_gradient_max_abs'] == 0)
    audit.check('normalizer:constant_finite', normalizer['constant_repaired_input_gradient_finite'])
    for dataset, fixture in result['fixtures'].items():
        prefix = 'fixture:' + dataset
        audit.check(prefix + ':identity_raw_parity', fixture['identity_anchor_vs_RAW_BOLT_max_abs'] < 2e-6,
                    fixture['identity_anchor_vs_RAW_BOLT_max_abs'])
        for key in ('causal_prefix_max_abs', 'causal_active_temporal_prefix_max_abs', 'existing_quarter_selected_checkpoint_forward_parity_max_abs'):
            audit.check(prefix + ':' + key, fixture[key] == 0)
        audit.check(prefix + ':window_reset', fixture['missing_window_reset_all_missing_normalized_zero'])
        audit.check(prefix + ':backbone', fixture['backbone_hash_unchanged'])
        audit.asset(fixture['existing_quarter_checkpoint'], prefix + ':old_checkpoint')
        groups = fixture['architecture_gradient_temp_update_fixtures']
        audit.check(prefix + ':all_architectures', {g['architecture'] for g in groups} == set(ARCHITECTURES))
        for group in groups:
            audit.check(prefix + ':' + group['architecture'] + ':gradient_update_restore',
                        group['finite_gradient'] and group['actual_temp_optimizer_step_max_abs'] > 0 and group['state_restored'])


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def audit_training(audit, case):
    import torch
    tr, name = case['training'], case['dataset'] + ':' + case['label']
    audit.check(name + ':full_600', tr['updates'] == 600 and len(tr['curve']) == 600)
    audit.check(name + ':curve_updates', [r['update'] for r in tr['curve']] == list(range(1, 601)))
    for key in ('main_loss', 'aux_loss', 'loss', 'gradient_norm'):
        audit.check(name + ':finite_' + key, np.isfinite([r[key] for r in tr['curve']]).all())
    audit.check(name + ':optimizer_contract', tr['optimizer'] == 'AdamW' and tr['lr'] == .001 and tr['weight_decay'] == 0.)
    audit.check(name + ':checkpoints', [c['update'] for c in tr['checkpoints']] == [0, 100, 200, 400, 600])
    audit.check(name + ':tune_choice', independent_checkpoint_choice(tr['checkpoints']) == tr['selected_update'])
    order = np.asarray(tr['batch_indices'], np.int64)
    audit.check(name + ':batch_order_hash', array_hash(order) == tr['order_sha256'])
    audit.close(name + ':loss_composition', [r['loss'] for r in tr['curve']],
                [r['main_loss'] + case.get('aux_weight', 0.) * r['aux_loss'] for r in tr['curve']],
                rtol=3e-7, atol=2e-6)
    selected_state = initial_state = None
    for cp in tr['checkpoints']:
        path = audit.asset(cp, name + ':cp' + str(cp['update']) + ':sha')
        state = torch.load(path, map_location='cpu', weights_only=True)
        audit.check(name + ':cp' + str(cp['update']) + ':metadata', state['update'] == cp['update'] and state['seed'] == case['seed'])
        if cp['update'] == 0:
            initial_state = state
        if cp['update'] == tr['selected_update']:
            selected_state = state
    audit.check(name + ':initial_state_hash', state_hash(initial_state['codec']) == tr['initial_codec_hash'])
    audit.check(name + ':selected_state_hash', state_hash(selected_state['codec']) == tr['selected_codec_hash'])
    return initial_state, selected_state


def audit_teacher_basis(audit, result_path, dataset, seed, data, cases=()):
    path = result_path.parent / dataset / f'TEACHER_s{seed}.json'
    if path.is_file():
        record = read(path)
    else:
        teacher = next((c for c in cases if c['dataset'] == dataset and c.get('seed') == seed and c.get('stage') == 'S3_TEACHER'), None)
        if teacher is None:
            return None
        record = {**teacher['basis_fitting'], 'checkpoint': teacher['training']['selected_checkpoint'],
                  'fit_output_artifact': teacher['basis_artifact']}
    prefix = f'teacher:{dataset}:{seed}'
    audit.asset(record['checkpoint'], prefix + ':checkpoint')
    artifact = audit.asset(record['fit_output_artifact'], prefix + ':FIT_artifact')
    with np.load(artifact, allow_pickle=False) as z:
        forecast = z['forecast'] if 'forecast' in z else z['fit_forecast']
        last = z['last'] if 'last' in z else z['fit_last']
        basis = z['basis']
    histories = np.stack([data['physical'][o-L:o] for o in data['origins']['fit']])
    masks = np.stack([data['mask'][o-L:o] for o in data['origins']['fit']])
    expected_last = independent_window_fill(histories, masks, data['mean'], data['std'])[:, -1]
    audit.close(prefix + ':past_last', last, expected_last, rtol=0, atol=0)
    changes = (forecast - last[:, None]).reshape(-1, forecast.shape[-1]).astype(np.float64)
    centered = changes - changes.mean(0)
    covariance = centered.T @ centered / len(centered)
    values = np.linalg.eigvalsh(covariance)
    audit.close(prefix + ':eigenvalues', values, record['teacher_delta_eigenvalues'], rtol=1e-8, atol=1e-10)
    audit.close(prefix + ':orthonormal_basis', basis.astype(float).T @ basis, np.eye(basis.shape[1]), rtol=0, atol=2e-7)
    audit.close(prefix + ':topK_captured_variance', np.trace(basis.astype(float).T @ covariance @ basis), values[-basis.shape[1]:].sum(),
                rtol=2e-7, atol=1e-8)
    audit.check(prefix + ':basis_SHA', array_hash(basis) == record['basis_sha256'])
    return basis


def audit_alignment(audit, case, data, basis):
    import torch
    metadata = case['alignment']['metadata']
    prefix = case['dataset'] + ':' + case['label'] + ':alignment'
    artifact = checked_path(case['training']['selected_checkpoint']['path']).parent / 'alignment.pt'
    audit.check(prefix + ':state_exists', artifact.is_file())
    if not artifact.is_file() or basis is None:
        audit.check(prefix + ':independent_coordinates_available', False)
        return
    state = torch.load(artifact, map_location='cpu', weights_only=True)
    coordinates = state['target'].numpy()
    valid = state['valid'].numpy()
    target, observed = expected_targets(data, 'fit')
    physical = np.stack([data['physical'][o-L:o] for o in data['origins']['fit']])
    mask = np.stack([data['mask'][o-L:o] for o in data['origins']['fit']])
    last = independent_window_fill(physical, mask, data['mean'], data['std'])[:, -1]
    expected, expected_valid = independent_coordinates(target, observed, last, basis)
    audit.check(prefix + ':basis_SHA', array_hash(basis) == metadata['basis_sha256'])
    audit.check(prefix + ':valid_SHA', array_hash(valid) == metadata['valid_sha256'])
    audit.check(prefix + ':target_SHA', array_hash(coordinates) == metadata['target_sha256'])
    audit.check(prefix + ':independent_valid', np.array_equal(valid, expected_valid))
    audit.close(prefix + ':independent_coordinates', coordinates, expected, rtol=2e-6, atol=2e-6)
    latent_mean = coordinates[valid].mean(0) if valid.any() else np.zeros(coordinates.shape[-1], np.float32)
    latent_std = coordinates[valid].std(0) if valid.any() else np.ones(coordinates.shape[-1], np.float32)
    latent_std = np.where(latent_std > 1e-6, latent_std, 1).astype(np.float32)
    audit.close(prefix + ':latent_mean', state['latent_mean'].numpy(), latent_mean, rtol=0, atol=0)
    audit.close(prefix + ':latent_std', state['latent_std'].numpy(), latent_std, rtol=0, atol=0)
    summary, summary_mask = independent_summary(coordinates, valid)
    audit.check(prefix + ':summary_mask', np.array_equal(summary_mask, state['summary_valid'].numpy()))
    means = np.array([summary[summary_mask[:, j], j].mean() for j in range(summary.shape[1])], np.float32)
    stds = np.array([summary[summary_mask[:, j], j].std() for j in range(summary.shape[1])], np.float32)
    stds = np.where(stds > 1e-6, stds, 1).astype(np.float32)
    audit.close(prefix + ':summary_mean', state['means'].numpy(), means, rtol=1e-5, atol=2e-6)
    audit.close(prefix + ':summary_std', state['stds'].numpy(), stds, rtol=1e-5, atol=2e-6)
    audit.close(prefix + ':summary_targets', state['summary_target'].numpy(), (summary-means)/stds, rtol=1e-5, atol=1e-5)
    permutation = state['shuffle'].numpy()
    usable = summary_mask.all(1)
    origins = data['origins']['fit']
    audit.check(prefix + ':shuffle_bijection', np.array_equal(np.sort(permutation), np.arange(len(permutation))))
    audit.check(prefix + ':shuffle_SHA', array_hash(permutation) == metadata['shuffle_sha256'])
    for name in ('fixed', 'learned'):
        audit.check(prefix + ':prototype_' + name + '_SHA', array_hash(state[name].numpy()) == metadata['prototype_' + name + '_sha256'])
    audit.check(prefix + ':shuffle_same_loss_mask', np.array_equal(usable, usable[permutation]))
    if metadata.get('shuffle_available', True):
        audit.check(prefix + ':nonoverlap_future_pair', np.all(np.abs(origins[usable] - origins[permutation[usable]]) >= H))
    audit.check(prefix + ':head_removed', metadata['deployment_head_removed'])


def audit_method(audit, case, method, data, authorized):
    if 'eval' in method['scores']:
        require_eval_authorization(authorized_for(authorized, case['dataset']))
    label = case['dataset'] + ':' + case['label'] + ':' + method['method']
    path = audit.asset(method['prediction'], label + ':NPZ_sha')
    with np.load(path, allow_pickle=False) as z:
        mean, std = z['mean'], z['std']
        audit.close(label + ':mean', mean, data['mean'], rtol=0, atol=0)
        audit.close(label + ':std', std, data['std'], rtol=0, atol=0)
        for split, declared in method['scores'].items():
            pred, target, mask, origins = (z[split + suffix] for suffix in ('_prediction', '_target', '_observed', '_origins'))
            audit.close(label + ':' + split + ':origins', origins, data['origins'][split], rtol=0, atol=0)
            expected, expected_mask = expected_targets(data, split)
            audit.close(label + ':' + split + ':target', target, expected, rtol=0, atol=0)
            audit.check(label + ':' + split + ':mask', np.array_equal(mask, expected_mask))
            norm = independent_score(pred, target, mask)
            physical = independent_score(pred * std + mean, target * std + mean, mask)
            audit_score(audit, label + ':' + split + ':normalized', norm, declared['normalized'])
            audit_score(audit, label + ':' + split + ':physical', physical, declared)
            if 'training' in case and method['method'] == 'BOLT' and split == 'tune':
                selected = case['training']['selected_checkpoint']['forecast']['tune']
                audit_score(audit, label + ':selected_TUNE_from_NPZ', norm, selected['normalized'], rtol=2e-6, atol=1e-6)


def audit_native(audit, case, method):
    tr = method['native_training']
    prefix = case['dataset'] + ':' + case['label'] + ':native'
    audit.check(prefix + ':600_updates', tr['updates'] == 600 and len(tr['curve']) == 600)
    audit.check(prefix + ':curve_order', [r['update'] for r in tr['curve']] == list(range(1, 601)))
    audit.check(prefix + ':finite_curve', np.isfinite([[r['loss'], r['gradient_norm']] for r in tr['curve']]).all())
    audit.check(prefix + ':frozen_ED', tr['encoder_decoder_frozen'] and not tr['E_D_optimized_for_native'])
    selected = min((cp['tune']['normalized']['mse'], cp['update']) for cp in tr['checkpoints'])[1]
    audit.check(prefix + ':TUNE_choice', selected == tr['selected_update'])
    for cp in tr['checkpoints']:
        audit.asset(cp, prefix + ':checkpoint_' + str(cp['update']))
    audit_score(audit, prefix + ':selected_TUNE_NPZ', method['scores']['tune']['normalized'],
                tr['selected_checkpoint']['tune']['normalized'], rtol=2e-6, atol=1e-6)


def audit_ridge_choice(audit, case, method):
    prefix = case['dataset'] + ':' + case['label'] + ':' + method['method'] + ':ridge_choice'
    if 'correction' in method:
        selection = method['correction']
        scores = selection['tune_scores']
        chosen = int(np.argmin(scores))
        audit.check(prefix + ':TUNE_grid', len(scores) == len(LAMBDAS) and np.isfinite(scores).all())
        audit.check(prefix + ':lambda', selection['lambda'] == LAMBDAS[chosen])
        audit.check(prefix + ':support', selection['raw_input_support'] == L)
        audit.check(prefix + ':FIT_in_sample_declared', selection['FIT_base_predictions_in_sample'])
    if 'selection' in method and 'selected_lambda' in method['selection']:
        selection = method['selection']
        options = selection['candidates']
        chosen = int(np.argmin([c['tune']['normalized']['mse'] for c in options]))
        audit.check(prefix + ':self_lambda', chosen == selection['selected_index'] and options[chosen]['lambda'] == selection['selected_lambda'])
        audit.check(prefix + ':self_grid', tuple(c['lambda'] for c in options) == LAMBDAS)


def audit_cases(audit, cases, data, teacher_bases, authorized):
    trained = []
    for case in cases:
        if 'training' in case:
            audit_training(audit, case)
            trained.append(case)
        for method in case.get('methods', []):
            audit_method(audit, case, method, data[case['dataset']], authorized)
            if 'native_training' in method:
                audit_native(audit, case, method)
            audit_ridge_choice(audit, case, method)
        if 'alignment' in case:
            audit_alignment(audit, case, data[case['dataset']], teacher_bases.get((case['dataset'], case['seed'])))
    for dataset in data:
        for seed in SEEDS:
            subset = [c for c in trained if c['dataset'] == dataset and c['seed'] == seed]
            audit.check(f'matched:{dataset}:{seed}:order', len({c['training']['order_sha256'] for c in subset}) <= 1)
            aligned = [c for c in subset if c.get('stage') == 'S2' or c.get('stage') == 'S3' and c.get('aux_method') in METHODS]
            audit.check(f'matched:{dataset}:{seed}:aligned_init', len({c['training']['initial_codec_hash'] for c in aligned}) <= 1)
    return trained


def audit_selection(audit, result):
    cases = result.get('cases', [])
    raw = {c['dataset']: c['methods'][0]['scores']['tune']['normalized']['mse'] for c in cases if c.get('stage') == 'RAW'}
    for stage, key in (('S1', 'S1_selection'), ('S2', 'S2_selection')):
        if key not in result:
            continue
        selection, selected_candidates = result[key], []
        for candidate in selection['candidates']:
            valid_weights = []
            for item in candidate['weight_candidates']:
                matched = [c for c in cases if c.get('stage') == stage and
                           (c['architecture'] == candidate['name'] if stage == 'S1' else
                            c['aux_method'] == candidate['name'] and c['aux_weight'] == item['weight'])]
                complete = complete_development_coverage(matched)
                label = 'choice:' + stage + ':' + candidate['name'] + ':w' + str(item['weight'])
                audit.check(label + ':coverage', complete == item['complete'] and len(matched) == item['coverage'])
                if complete:
                    score = float(np.log([c['methods'][0]['scores']['tune']['normalized']['mse'] / raw[c['dataset']] for c in matched]).mean())
                    audit.close(label + ':score', score, item['log_mean_TUNE_RAW_ratio'])
                    valid_weights.append((score, item['weight']))
            if valid_weights:
                best = min(valid_weights, key=lambda p: p[0])
                audit.check('choice:' + stage + ':' + candidate['name'] + ':weight', best[1] == candidate['chosen_weight'])
                selected_candidates.append((candidate['name'], best[0]))
        minimum = min(c[1] for c in selected_candidates)
        selected = next(name for name, score in selected_candidates if score <= minimum + 1e-12)
        audit.check('choice:' + stage + ':global', selected == selection['selected'])


def audit_run(audit, result_path, data, authorized):
    result_path = checked_path(result_path)
    result = read(result_path)
    require_terminal(result)
    resolved = result['status'] in ('COMPLETE_RESOLVED_WITH_DECLARED_PROTOCOL_DEVIATION', 'PARTIAL_RESOLVED')
    audit.scope('run:' + result_path.parent.name + ':terminal_complete', result['status'] == 'COMPLETE', result['status'])
    for name, digest in result.get('source_hashes', {}).items():
        original = checked_path(name)
        snapshot = result_path.parent / 'sources' / original.name
        audit.check('source:' + result_path.parent.name + ':' + original.name + ':original', file_hash(original) == digest)
        if not resolved:
            audit.check('source:' + result_path.parent.name + ':' + original.name + ':snapshot', file_hash(snapshot) == digest)
    lineage = []
    if resolved:
        lineage = result_lineage(audit, result_path)
        audit.check('resolved:immutable_lineage_present', len(lineage) >= 2)
        deviation = result.get('protocol_deviation', result.get('protocol_deviations', {}))
        audit.scope('resolved:original_protocol', False, deviation)
        for source_path, source in lineage:
            for name, digest in source.get('source_hashes', {}).items():
                original = checked_path(name)
                audit.check('resolved:source:' + source_path.parent.name + ':' + original.name, file_hash(original) == digest)
                snapshot = source_path.parent / 'sources' / original.name
                if snapshot.is_file():
                    audit.check('resolved:snapshot:' + source_path.parent.name + ':' + original.name, file_hash(snapshot) == digest)
    if result.get('fixtures'):
        audit_fixture(audit, result, result_path.parent)
    contracts = dict(result.get('newdata_contracts', {}))
    for _, source in lineage:
        contracts.update(source.get('newdata_contracts', {}))
    for dataset in contracts:
        if dataset not in data:
            data[dataset] = fresh_data(audit, dataset, result_path, authorized)
    teacher_bases = {}
    for dataset in data:
        for seed in SEEDS:
            if dataset in data:
                teacher_bases[dataset, seed] = audit_teacher_basis(audit, result_path, dataset, seed, data[dataset], result.get('cases', []))
    trained = audit_cases(audit, result.get('cases', []), data, teacher_bases, authorized)
    audit.check('run:' + result_path.parent.name + ':training_count', len(trained) == result.get('training_fits', len(trained)))
    audit_selection(audit, result)
    if 'method_lock' in result:
        path = audit.asset(result['method_lock'], 'method_lock:SHA:' + result_path.parent.name)
        locked = read(path)
        audit.check('method_lock:status', locked['status'] == 'METHODS_LOCKED_BEFORE_NEW_RAW_DOWNLOAD')
        audit.check('method_lock:constants', locked['updates'] == 600 and locked['L'] == L and locked['H'] == H and tuple(locked['seeds']) == SEEDS)
        audit.check('method_lock:K_rule', locked['K_rule'] == 'max(1,floor(C/4))')
        if 'S1_selection' in result:
            audit.check('method_lock:architecture', locked['architecture'] == result['S1_selection']['selected'])
            audit.check('method_lock:S1_choice_record', locked['S1_selection'] == result['S1_selection'])
            audit.check('method_lock:S2_choice_record', locked['S2_selection'] == result['S2_selection'])
            audit.check('method_lock:weights', locked['weights'] == {c['name']: c['chosen_weight'] for c in result['S2_selection']['candidates']})
        if contracts:
            audit.scope('new_coverage:both_declared_datasets', set(contracts) == {'appliances', 'beijing_aotizhongxin'},
                        {'actual': sorted(contracts), 'failed_paths': result.get('failed_paths', [])})
            for dataset in contracts:
                for seed in SEEDS:
                    subset = [c for c in trained if c['dataset'] == dataset and c['seed'] == seed]
                    teacher = [c for c in subset if c.get('stage') == 'S3_TEACHER']
                    main = [c for c in subset if c.get('stage') == 'S3' and c.get('aux_method') == 'MAIN']
                    auxiliary = [c for c in subset if c.get('stage') == 'S3' and c.get('aux_method') in METHODS]
                    prefix = f'new_coverage:{dataset}:{seed}'
                    audit.check(prefix + ':teacher', len(teacher) == 1)
                    audit.check(prefix + ':base', {c['architecture'] for c in main} == set(('DIRECT_PCA', 'ANCHOR_PCA', locked['architecture'])))
                    audit.check(prefix + ':all_aux', {c['aux_method'] for c in auxiliary} == set(METHODS) and len(auxiliary) == len(METHODS))
                    for case in auxiliary:
                        audit.check(prefix + ':' + case['aux_method'] + ':locked_weight', case['aux_weight'] == locked['weights'][case['aux_method']])
    if 'resources' in result:
        resource = result['resources']
        audit.check('resource:' + result_path.parent.name + ':GPU_wall', resource['cumulative_gpu_process_wall_seconds'] <= 7200)
        audit.check('resource:' + result_path.parent.name + ':memory', max(resource['peak_sampled_rss_bytes'], resource['peak_allocated_vram_bytes']) <= 8*1024**3)
        audit.check('resource:' + result_path.parent.name + ':storage', resource.get('new_bytes', 0) <= 12*1024**3)
    if result.get('mode') == 'dev':
        audit.check('coverage:S1_24', len([c for c in trained if c['stage'] == 'S1']) == 24)
        audit.check('coverage:S2_52', len([c for c in trained if c['stage'] == 'S2']) == 52)
    return result


def audit_completed_cases(audit, folder, stage, data, authorized):
    folder = checked_path(folder)
    paths = sorted(folder.glob('*/' + stage + '_*.json'))
    paths = [p for p in paths if not p.stem.endswith('_FAILED')]
    cases = [read(p) for p in paths]
    for path, case in zip(paths, cases):
        audit.check('completed_case:' + path.parent.name + ':' + path.name, case['training']['updates'] == 600 and bool(case['methods']))
    bases = {(d, s): audit_teacher_basis(audit, folder / 'RESULT.json', d, s, data[d]) for d in ('jena', 'robin') for s in SEEDS}
    trained = audit_cases(audit, cases, data, bases, authorized)
    selection_path = folder / (stage + '_SELECTION.json')
    if selection_path.is_file():
        raw = [read(p) for p in sorted(folder.glob('*/RAW.json'))]
        audit_selection(audit, {'cases': cases + raw, stage + '_selection': read(selection_path)})
    audit.check('completed_stage:' + stage + ':case_count', len(trained) == (24 if stage == 'S1' else 52), len(trained))
    return {'scope': 'Completed immutable case artifacts; enclosing campaign may still be running',
            'stage': stage, 'case_count': len(cases), 'case_files': [{'path': str(p.relative_to(ROOT)), 'sha256': file_hash(p)} for p in paths]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', action='append', default=[])
    parser.add_argument('--completed-case-dir')
    parser.add_argument('--completed-stage', choices=('S1', 'S2'), default='S1')
    parser.add_argument('--cost-result', action='append', default=[])
    parser.add_argument('--checkpoint-result', action='append', default=[])
    parser.add_argument('--output', required=True)
    parser.add_argument('--allow-eval-after-root-authorization', action='store_true')
    parser.add_argument('--allow-eval-dataset', action='append', default=[])
    args = parser.parse_args()
    destination = Path(args.output).resolve()
    if not destination.is_relative_to(HERE) or destination.exists():
        raise ValueError('Fresh audit output directory inside this study required')
    audit, preflight = Audit(), read(HERE / 'PREFLIGHT.json')
    authorized = set(args.allow_eval_dataset) if args.allow_eval_dataset else args.allow_eval_after_root_authorization
    for i, asset in enumerate(preflight['preserve_files']):
        audit.asset(asset, f'preserved:{i}')
    data = {}
    for dataset, record in preflight['legacy_data'].items():
        audit.asset(record['source'], 'legacy:' + dataset + ':source')
        value = legacy_data(record)
        data[dataset] = value
        for field in ('mean', 'std'):
            audit.check('legacy:' + dataset + ':' + field, array_hash(value[field]) == record[field + '_sha256'])
        for split, origins in value['origins'].items():
            audit.check('legacy:' + dataset + ':origins_' + split, array_hash(origins) == record['origin_hashes'][split])
    result_records = []
    for path in args.result:
        result = audit_run(audit, path, data, authorized)
        result_records.append({'path': str(checked_path(path).relative_to(ROOT)), 'sha256': file_hash(checked_path(path)), 'status': result['status']})
    completed = None
    if args.completed_case_dir:
        completed = audit_completed_cases(audit, args.completed_case_dir, args.completed_stage, data, authorized)
    cost_reports = [audit_cost_result(audit, path) for path in args.cost_result]
    checkpoint_reports = []
    for path in args.checkpoint_result:
        checkpoint_reports.append(audit_checkpoint_result(audit, path, data, authorized))
    prior_audits = []
    for path in sorted(HERE.glob('validation_*/audit.json')):
        if path.parent.resolve() == destination:
            continue
        receipt = read(path)
        prior_audits.append({'path': str(path.relative_to(ROOT)), 'sha256': file_hash(path),
                             'cpu_wall_seconds': receipt['cpu_wall_seconds']})
    current_wall = time.perf_counter() - audit.started
    total_wall = sum(r['cpu_wall_seconds'] for r in prior_audits) + current_wall
    audit.check('CPU_audit:terminal_receipt_process_wall_cap', total_wall <= 7200,
                {'counted_seconds': total_wall, 'cap_seconds': 7200,
                 'claim_limit': 'Unique terminal audit process receipts plus this process; excludes brief shell/test/source-inspection tasks'})
    report = {'status': 'PASS' if all(c['pass'] for c in audit.checks) else 'FAIL',
              'created_unix': time.time(), 'checks': audit.checks, 'count': len(audit.checks),
              'failed_count': sum(not c['pass'] for c in audit.checks), 'results': result_records,
              'completed_stage': completed,
              'checkpoint_audits': checkpoint_reports,
              'cost_audits': cost_reports,
              'completion_status': 'COMPLETE_FOR_REQUESTED_AUDIT_SCOPE' if all(c['complete'] for c in audit.completion_checks) else 'INCOMPLETE_OR_PROTOCOL_DEVIATION',
              'completion_checks': audit.completion_checks,
              'cpu_wall_seconds': time.perf_counter() - audit.started,
              'prior_audit_cpu_receipts': prior_audits,
              'counted_audit_cpu_wall_seconds_including_current': total_wall,
              'validator': {'source_sha256': file_hash(Path(__file__)),
                            'test_sha256': file_hash(HERE / 'test_validation_contracts.py')},
              'eval_audit_authorized': sorted(authorized) if isinstance(authorized, set) else authorized,
              'scope': 'Independent NumPy raw/metric/mask/origin, source and CPU checkpoint hashes and TUNE choice',
              'limitations': ['No GPU inference or backbone training',
                             'All-checkpoint NPZ scores are independently audited only when a checkpoint-result is supplied',
                             'Fixture evidence is receipt/source checked, not a new independent GPU rerun',
                             'Timed cost output arrays were not saved; cost parity is source/receipt checked rather than independently rescored',
                             'This audit does not prove exhaustive absence of leakage or independent generalization']}
    destination.mkdir()
    sources = destination / 'sources'
    sources.mkdir()
    for path in (Path(__file__), HERE / 'test_validation_contracts.py'):
        (sources / path.name).write_bytes(path.read_bytes())
    (destination / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('status', 'count', 'failed_count', 'cpu_wall_seconds')}))
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
