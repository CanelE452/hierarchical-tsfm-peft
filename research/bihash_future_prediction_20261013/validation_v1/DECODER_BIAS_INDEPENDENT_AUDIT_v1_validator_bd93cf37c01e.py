"""Independent CPU numerical and causal audits of the primary Bi-Hash study.

Pure NumPy scoring never imports an experiment runner. The optional model
fixture instantiates temporary CPU models with synthetic data and a tiny frozen
forecaster; it does not train an experiment, load Bolt, access TEST, or open EVAL.
Real EVAL-bearing artifacts require explicit post-first-evaluation authorization.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PROTECTED_NAMES = {'test.npz', 'jena_test.npz'}


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_fresh(path, value):
    # Fully serialize before creating any file: a serialization failure leaves
    # no misleading partial terminal receipt.
    payload = json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        handle.write(payload)


def checked_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if path.name.lower() in PROTECTED_NAMES:
        raise PermissionError('PROTECTED_TEST_NOT_AUTHORIZED')
    return path


def file_hash(value):
    digest = hashlib.sha256()
    with checked_path(value).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def macro_score(prediction, target, observed):
    p, y, mask = np.asarray(prediction), np.asarray(target), np.asarray(observed, bool)
    if p.ndim != 3 or p.shape != y.shape or p.shape != mask.shape:
        raise ValueError('FORECAST_MASK_SHAPE_MISMATCH')
    if not np.isfinite(p).all() or not np.isfinite(y[mask]).all():
        raise ValueError('NONFINITE_SCORED_VALUE')
    count, mse, mae = [], [], []
    for c in range(p.shape[-1]):
        active = mask[..., c]
        error = p[..., c][active].astype(np.float64) - y[..., c][active].astype(np.float64)
        count.append(int(error.size))
        mse.append(float(np.mean(error * error)) if error.size else 0.)
        mae.append(float(np.mean(np.abs(error))) if error.size else 0.)
    valid = np.array(count) > 0
    if not valid.any():
        raise ValueError('NO_OBSERVED_TARGET')
    result = {'mse': float(np.array(mse)[valid].mean()), 'mae': float(np.array(mae)[valid].mean()),
              'per_channel_mse': mse, 'per_channel_mae': mae, 'observed_counts': count,
              'horizon_metrics': {}}
    if p.shape[1] == 48:
        for name, lo, hi in (('1_8', 0, 8), ('9_24', 8, 24), ('25_48', 24, 48)):
            if mask[:, lo:hi].any():
                result['horizon_metrics'][name] = macro_score(p[:, lo:hi], y[:, lo:hi], mask[:, lo:hi])
            else:
                result['horizon_metrics'][name] = {'status': 'NO_OBSERVED_TARGET'}
    return result


def mse_difference_bound(first, second, target, observed):
    """Channel-macro |MSE(a)-MSE(b)| bound with actual observed-cell weights."""
    a, b, y = [np.asarray(x, np.float64) for x in (first, second, target)]
    mask = np.asarray(observed, bool)
    bounds = []
    for c in range(a.shape[-1]):
        valid = mask[..., c]
        if valid.any():
            delta = a[..., c][valid] - b[..., c][valid]
            error = b[..., c][valid] - y[..., c][valid]
            bounds.append(float(np.mean(2 * np.abs(error) * np.abs(delta) + delta * delta)))
    if not bounds:
        raise ValueError('NO_OBSERVED_MSE_BOUND')
    return float(np.mean(bounds))


def forecast_score(prediction, target, observed, mean, std):
    p, y = np.asarray(prediction), np.asarray(target)
    mean, std = np.asarray(mean), np.asarray(std)
    if mean.shape != (p.shape[-1],) or std.shape != mean.shape or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError('INVALID_NORMALIZATION_STATISTICS')
    # Preserve saved array dtype for the declared physical inverse transform.
    # This reproduces FP32 conversion rounding without sharing runner scoring.
    result = macro_score(p * std + mean, y * std + mean, observed)
    result['normalized'] = macro_score(p, y, observed)
    result['scale_note'] = 'Physical channel macro within dataset; heterogeneous units are not pooled across datasets'
    return result


def fit_statistics(physical, observed, bounds):
    raw, mask = np.asarray(physical), np.asarray(observed, bool)
    lo, hi = map(int, bounds)
    if raw.shape != mask.shape or raw.ndim != 2 or not 0 <= lo < hi <= len(raw):
        raise ValueError('INVALID_FIT_SLICE')
    means, stds = [], []
    for c in range(raw.shape[1]):
        v = raw[lo:hi, c][mask[lo:hi, c]].astype(np.float64)
        if not v.size or not np.isfinite(v).all():
            raise ValueError('NO_FINITE_FIT_CHANNEL')
        means.append(v.mean())
        stds.append(v.std(ddof=0))
    mean, std64 = np.array(means, np.float32), np.array(stds, np.float64)
    return mean, np.where(std64 > 1e-6, std64, 1.).astype(np.float32)


def raw_windows(physical, observed, origins, mean, std, length=512, horizon=48):
    raw, mask = np.asarray(physical), np.asarray(observed, bool)
    origins = np.asarray(origins, np.int64)
    mean, std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
    if raw.shape != mask.shape or raw.ndim != 2 or np.any(origins - length < 0) or np.any(origins + horizon > len(raw)):
        raise ValueError('RAW_WINDOW_OUT_OF_BOUNDS')
    histories, targets, hmask, tmask = [], [], [], []
    for origin in origins:
        last = mean.copy()
        h = np.empty((length, raw.shape[-1]), np.float32)
        hm = mask[origin - length:origin].copy()
        for index, row in enumerate(range(int(origin) - length, int(origin))):
            last[mask[row]] = raw[row, mask[row]]
            h[index] = (last - mean) / std
        tm = mask[origin:origin + horizon].copy()
        ty = np.zeros((horizon, raw.shape[-1]), np.float32)
        for c in range(raw.shape[-1]):
            ty[tm[:, c], c] = (raw[origin:origin + horizon, c][tm[:, c]] - mean[c]) / std[c]
        histories.append(h); targets.append(ty); hmask.append(hm); tmask.append(tm)
    return np.array(histories), np.array(targets), np.array(hmask), np.array(tmask)


def memory_eligibility(query_origins, bank_origins, length=512, horizon=48):
    origins, memory = np.asarray(query_origins, np.int64), np.asarray(bank_origins, np.int64)
    result = np.zeros((len(origins), length, len(memory)), bool)
    for b, origin in enumerate(origins):
        for row in range(length):
            last_observation = int(origin) - length + row
            result[b, row] = memory + horizon - 1 <= last_observation
    return result


def numpy_attention(query, keys, eligible, temperature=.2):
    q, k, valid = np.asarray(query, np.float64), np.asarray(keys, np.float64), np.asarray(eligible, bool)
    if valid.shape != (*q.shape[:2], len(k)):
        raise ValueError('ATTENTION_SHAPE_MISMATCH')
    answer = np.zeros_like(valid, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-6)
    k = k / np.maximum(np.linalg.norm(k, axis=-1, keepdims=True), 1e-6)
    for b in range(len(q)):
        for row in range(q.shape[1]):
            allowed = np.flatnonzero(valid[b, row])
            if len(allowed):
                scores = k[allowed] @ q[b, row] / temperature
                probabilities = np.exp(scores - scores.max())
                answer[b, row, allowed] = probabilities / probabilities.sum()
    return answer


def checkpoint_choice(checkpoints):
    if not checkpoints:
        raise ValueError('NO_CHECKPOINTS')
    updates = [int(c['update']) for c in checkpoints]
    if len(set(updates)) != len(updates):
        raise ValueError('DUPLICATE_CHECKPOINT_UPDATE')
    values = [(float(c['tune']['normalized']['mse']), int(c['update'])) for c in checkpoints]
    if not np.isfinite([s for s, _ in values]).all():
        raise ValueError('NONFINITE_TUNE_SCORE')
    return min(values)[1]


def batch_contract(origins, indices, expected_updates=600, batch_size=8, horizon=48):
    oo, batches = np.asarray(origins, np.int64), np.asarray(indices, np.int64)
    if batches.shape != (expected_updates, batch_size) or np.any(batches < 0) or np.any(batches >= len(oo)):
        raise ValueError('BATCH_SHAPE_OR_INDEX')
    minimum = None
    for batch in batches:
        if len(np.unique(batch)) != batch_size:
            raise ValueError('BATCH_DUPLICATE_ORIGIN')
        diff = np.diff(np.sort(oo[batch]))
        if np.any(diff < horizon):
            raise ValueError('BATCH_FUTURE_WINDOW_OVERLAP')
        if len(diff):
            minimum = int(diff.min()) if minimum is None else min(minimum, int(diff.min()))
    return {'updates': len(batches), 'batch_size': batch_size, 'minimum_future_gap': minimum,
            'array_sha256': array_hash(batches)}


def load_prediction(path, splits, eval_authorized=False):
    splits = tuple(splits)
    # Check permission before opening or hashing even a non-existent payload.
    if any(s.lower() == 'eval' for s in splits) and not eval_authorized:
        raise PermissionError('EVAL_REQUIRES_POST_FIRST_PASS_AUDIT_AUTHORIZATION')
    with np.load(checked_path(path), allow_pickle=False) as z:
        result = {'mean': z['mean'].copy(), 'std': z['std'].copy()}
        for split in splits:
            result[split] = {key: z[f'{split}_{key}'].copy() for key in ('prediction', 'target', 'observed', 'origins')}
    return result


def joint_barrier(lock, first_open_events, expected_cases, lock_sha256):
    expected = set(expected_cases)
    actual = [(m['dataset'], int(m['seed']), m['method'], float(m.get('weight', 0.))) for m in lock.get('selected_models', [])]
    datasets = {v[0] for v in expected}
    timestamp = lock.get('created_unix')
    reasons = []
    if not lock.get('ALL_SELECTED'):
        reasons.append('ALL_SELECTED_NOT_TRUE')
    if set(actual) != expected or len(actual) != len(expected):
        reasons.append('SELECTED_MODEL_COVERAGE_OR_DUPLICATES')
    if not datasets.issubset(set(lock.get('newdata_contracts', {}))):
        reasons.append('MISSING_NEW_DATA_CONTRACT')
    if not isinstance(timestamp, (int, float)) or not np.isfinite(timestamp):
        reasons.append('MISSING_LOCK_TIMESTAMP')
    else:
        for event in first_open_events:
            if event.get('dataset') not in datasets or event.get('opened_unix', -np.inf) <= timestamp:
                reasons.append('EVAL_OPEN_PRECEDES_JOINT_LOCK')
            if event.get('lock_sha256') != lock_sha256:
                reasons.append('EVAL_OPEN_LOCK_SHA_MISMATCH')
    if {e.get('dataset') for e in first_open_events} != datasets:
        reasons.append('MISSING_FIRST_OPEN_EVENT')
    return {'passed': not reasons, 'reasons': sorted(set(reasons)), 'expected_model_cases': len(expected),
            'actual_model_cases': len(actual), 'claim': 'coverage/timestamp evidence only; files and data SHA must be independently checked'}


class Audit:
    def __init__(self):
        self.checks = []
    def check(self, name, passed, detail=None):
        self.checks.append({'name': name, 'passed': bool(passed), 'detail': jsonable(detail)})
    def asset(self, name, artifact):
        path = checked_path(artifact['path'])
        self.check(name + ':exists', path.is_file(), str(path))
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = file_hash(path)
        self.check(name + ':sha256', actual == artifact.get('sha256'), {'actual': actual, 'declared': artifact.get('sha256')})
        return path
    def numeric(self, name, actual, expected, atol=1e-10, rtol=1e-8):
        a, b = np.asarray(actual), np.asarray(expected)
        same_shape = a.shape == b.shape
        ok = same_shape and np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a, b, atol=atol, rtol=rtol)
        self.check(name, ok, {'max_abs': float(np.max(np.abs(a - b))) if same_shape and a.size else None,
                              'atol': atol, 'rtol': rtol, 'actual_shape': list(a.shape), 'expected_shape': list(b.shape)})
    def close(self, **fields):
        failed = sum(not c['passed'] for c in self.checks)
        return {'status': 'PASS_WITH_LIMITATIONS' if not failed else 'FAIL', 'failed_count': failed,
                'check_count': len(self.checks), 'checks': self.checks, **fields}


def compare_score(audit, name, actual, claimed):
    for key in ('mse', 'mae', 'per_channel_mse', 'per_channel_mae'):
        audit.numeric(name + ':' + key, actual[key], claimed[key])
    audit.check(name + ':observed_counts', actual['observed_counts'] == claimed['observed_counts'])
    for key in actual.get('horizon_metrics', {}):
        if actual['horizon_metrics'][key].get('status') == 'NO_OBSERVED_TARGET':
            audit.check(name + ':' + key + ':no_observed', claimed.get('horizon_metrics', {}).get(key, {}).get('status') == 'NO_OBSERVED_TARGET')
        else:
            compare_score(audit, name + ':horizon:' + key, actual['horizon_metrics'][key], claimed['horizon_metrics'][key])
    if 'normalized' in actual:
        compare_score(audit, name + ':normalized', actual['normalized'], claimed['normalized'])


def audit_prediction(audit, name, artifact, scores, reference=None, eval_authorized=False, require_increasing=True):
    splits = tuple(scores)
    if 'eval' in splits and not eval_authorized:
        raise PermissionError('EVAL_REQUIRES_POST_FIRST_PASS_AUDIT_AUTHORIZATION')
    path = audit.asset(name, artifact)
    arrays = load_prediction(path, splits, eval_authorized=eval_authorized)
    rescored = {}
    for split, claimed in scores.items():
        item = arrays[split]
        actual = forecast_score(item['prediction'], item['target'], item['observed'], arrays['mean'], arrays['std'])
        compare_score(audit, name + ':' + split, actual, claimed)
        rescored[split] = actual
        origins = item['origins']
        audit.check(name + ':' + split + ':origins_unique', len(origins) == len(np.unique(origins)))
        if require_increasing:
            audit.check(name + ':' + split + ':origins_increasing', bool(np.all(np.diff(origins) > 0)))
        audit.check(name + ':' + split + ':mask_boolean', item['observed'].dtype == np.bool_)
        if reference is not None:
            for field in ('mean', 'std'):
                audit.check(name + ':' + split + ':' + field + '_raw_contract', np.array_equal(arrays[field], reference[field]))
            for field in ('target', 'observed', 'origins'):
                audit.check(name + ':' + split + ':' + field + '_raw_contract', np.array_equal(item[field], reference[field][split], equal_nan=True))
    return rescored


def method_selection(cases, expected_groups, method_order):
    groups = set(expected_groups)
    baseline_rows = [c for c in cases if c['method'] == 'NO_ALIGN']
    base = {(c['dataset'], int(c['seed'])): float(c['selected_tune_mse']) for c in baseline_rows}
    if set(base) != groups or len(baseline_rows) != len(groups):
        raise ValueError('BASELINE_SELECTION_COVERAGE')
    bins = {}
    for case in cases:
        if case['method'] != 'NO_ALIGN':
            bins.setdefault((case['method'], float(case['weight'])), []).append(case)
    rows, weights = [], {'NO_ALIGN': 0.}
    for (method, weight), items in sorted(bins.items()):
        values = {(c['dataset'], int(c['seed'])): float(c['selected_tune_mse']) for c in items}
        if set(values) != groups or len(items) != len(groups):
            raise ValueError('METHOD_SELECTION_COVERAGE:' + method)
        ratios = [values[g] / base[g] for g in sorted(groups)]
        if not np.isfinite(ratios).all() or np.any(np.array(ratios) <= 0):
            raise ValueError('INVALID_TUNE_RATIO')
        rows.append({'method': method, 'weight': weight, 'mean_log_tune_ratio': float(np.mean(np.log(ratios)))})
    for method in sorted({m for m, _ in bins}):
        weights[method] = min((r for r in rows if r['method'] == method), key=lambda r: (r['mean_log_tune_ratio'], r['weight']))['weight']
    b = [r for r in rows if r['method'].startswith('B_')]
    if not b:
        raise ValueError('NO_B_SELECTION_CANDIDATES')
    winner = min(b, key=lambda r: (r['mean_log_tune_ratio'], method_order.index(r['method'])))
    return {'weights': weights, 'selected_B_method': winner['method'], 'selection_records': rows,
            'claim': 'Recalculated equal dataset/seed-group TUNE selection; no EVAL or DEVVAL metric consulted'}


def tensor_state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def reference_from_physical(physical, observed, origins, mean, std):
    histories, targets, masks = {}, {}, {}
    for split, oo in origins.items():
        h, y, _hm, ym = raw_windows(physical, observed, oo, mean, std)
        histories[split], targets[split], masks[split] = h, y, ym
    fit_pool = histories['fit'].reshape(-1, physical.shape[-1]).astype(np.float64)
    center = fit_pool.mean(0)
    centered = fit_pool - center
    covariance = centered.T @ centered / len(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    k = max(1, physical.shape[-1] // 4)
    basis = eigenvectors[:, -k:][:, ::-1].copy()
    for column in range(k):
        if basis[np.argmax(np.abs(basis[:, column])), column] < 0:
            basis[:, column] *= -1
    fit_origins = np.asarray(origins['fit'], np.int64)
    bank_indices = np.arange(len(fit_origins)) if len(fit_origins) <= 128 else np.linspace(0, len(fit_origins) - 1, 128).astype(np.int64)
    bank_mask = masks['fit'][bank_indices]
    delta = np.where(bank_mask, targets['fit'][bank_indices] - histories['fit'][bank_indices, -1:], 0.).astype(np.float32)
    return {'mean': mean, 'std': std, 'origins': origins, 'history': histories, 'target': targets, 'observed': masks,
            'raw_history': {s: np.stack([physical[o - 512:o] for o in oo]) for s, oo in origins.items()},
            'history_observed': {s: np.stack([observed[o - 512:o] for o in oo]) for s, oo in origins.items()},
            'pca_mean': center.astype(np.float32), 'basis': basis.astype(np.float32),
            'bank': {'origins': fit_origins[bank_indices], 'delta': delta, 'observed': bank_mask}}


def legacy_reference(audit, dataset, declared=None):
    if dataset not in ('jena', 'robin'):
        raise ValueError('NOT_LEGACY_DATASET')
    contract_path = ROOT / 'research/level_matched_learned_ed_v18_20261006' / ('data_contract_' + dataset + '.json')
    contract = json.loads(contract_path.read_text(encoding='utf-8-sig'))
    source = audit.asset(dataset + ':legacy_TRAIN_DEVVAL', contract['trainval'])
    with np.load(source, allow_pickle=False) as z:
        x, observed = z['x'].astype(np.float32), z['finite'].astype(bool)
        old_mean = z['mean'].astype(np.float32)
        old_std = z['std'].astype(np.float32)
        old_std = np.where(old_std > 1e-6, old_std, 1.).astype(np.float32)
        raw = (x * old_std + old_mean).astype(np.float32)
        raw[~observed] = np.nan
        val_origins = z['val_origins'].astype(np.int64)
        times = z['times'].copy()
    train_lo, train_hi = contract['bounds']['train']
    fit_end = train_lo + int(np.floor(.8 * (train_hi - train_lo)))
    bounds = {'fit': (train_lo, fit_end), 'tune': (fit_end, train_hi)}
    origins = {'val': val_origins}
    for split, maximum in (('fit', 256), ('tune', 64)):
        lo, hi = bounds[split]
        candidates = np.arange(max(512, lo), min(hi - 48, len(raw) - 48) + 1, dtype=np.int64)
        legal = np.array([o for o in candidates if observed[o:o + 48].any(0).all()], np.int64)
        indices = np.arange(len(legal)) if len(legal) <= maximum else np.linspace(0, len(legal) - 1, maximum).astype(np.int64)
        origins[split] = legal[indices]
        audit.check(dataset + ':' + split + ':nonempty_legal_origins', len(origins[split]) > 0)
        audit.check(dataset + ':' + split + ':target_stays_inside_split', bool(np.all((origins[split] >= lo) & (origins[split] + 48 <= hi))))
    mean, std = fit_statistics(raw, observed, bounds['fit'])
    reference = reference_from_physical(raw, observed, origins, mean, std)
    audit.check(dataset + ':raw_timestamp_rows', len(times) == len(raw))
    audit.check(dataset + ':raw_timestamp_unique', len(np.unique(times)) == len(times))
    # Existing prepare_v6/data_prepare_v7 int64_ns functions store epoch ns.
    timestamp_seconds = (times.astype('datetime64[ns]') if np.issubdtype(times.dtype, np.integer) else times).astype('datetime64[s]').astype(np.int64)
    differences = np.diff(timestamp_seconds)
    audit.check(dataset + ':raw_timestamp_strictly_increasing', bool(np.all(differences > 0)))
    if 'interval_seconds' in contract:
        audit.check(dataset + ':raw_timestamp_declared_regular_interval', bool(np.all(differences == contract['interval_seconds'])))
    if declared is not None:
        for field in ('mean', 'std', 'basis', 'pca_mean'):
            audit.check(dataset + ':' + field + ':declared_array_SHA', array_hash(reference[field]) == declared[field + '_sha256'])
        for split, oo in origins.items():
            audit.check(dataset + ':' + split + ':declared_origin_SHA', array_hash(oo) == declared['origin_sha256'][split])
        for field in ('origins', 'delta', 'observed'):
            key = 'bank_origin_sha256' if field == 'origins' else 'bank_' + field + '_sha256'
            audit.check(dataset + ':bank_' + field + ':declared_array_SHA', array_hash(reference['bank'][field]) == declared[key])
        audit.check(dataset + ':fresh_floorK', declared['K'] == max(1, raw.shape[-1] // 4))
        audit.check(dataset + ':declared_FIT_bounds', list(bounds['fit']) == list(declared['contract']['bounds']['fit']))
    return reference


def legacy_loader_fixture():
    """Compare actual exposed legacy loader with raw NPZ-derived NumPy contracts."""
    started = time.perf_counter()
    source = HERE / 'data.py'
    before = file_hash(source)
    spec = importlib.util.spec_from_file_location('independent_primary_data_under_audit', source)
    dt = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = dt
    spec.loader.exec_module(dt)
    audit = Audit()
    summaries = {}
    for dataset in ('jena', 'robin'):
        reference = legacy_reference(audit, dataset)
        data, history, target, observed, history_observed = dt.load_legacy(dataset)
        for field in ('mean', 'std', 'basis', 'pca_mean'):
            audit.check(dataset + ':' + field + ':loader_exact', np.array_equal(reference[field], data[field]),
                        {'independent_sha256': array_hash(reference[field]), 'loader_sha256': array_hash(data[field])})
        for split in ('fit', 'tune', 'val'):
            for field, actual in (('history', history[split]), ('target', target[split]), ('observed', observed[split]), ('origins', data['origins'][split])):
                audit.check(dataset + ':' + split + ':' + field + ':loader_exact', np.array_equal(reference[field][split], actual),
                            {'independent_sha256': array_hash(reference[field][split]), 'loader_sha256': array_hash(actual)})
        for field, key in (('origins', 'bank_origins'), ('delta', 'bank_delta'), ('observed', 'bank_observed')):
            audit.check(dataset + ':bank_' + field + ':loader_exact', np.array_equal(reference['bank'][field], data['bank'][key]))
        summaries[dataset] = {'C': data['C'], 'K': data['K'], 'origin_counts': {s: len(v) for s, v in data['origins'].items()},
            'mean_sha256': array_hash(reference['mean']), 'std_sha256': array_hash(reference['std']),
            'basis_sha256': array_hash(reference['basis']), 'pca_mean_sha256': array_hash(reference['pca_mean']),
            'source_records': data['source_records'], 'contract': data['contract'],
            'reconstructed_physical_precision_limit': True}
    audit.check('data_source_unchanged_during_legacy_fixture', file_hash(source) == before)
    return audit.close(sources={str(source.relative_to(ROOT)): before}, datasets=summaries,
        cpu_wall_seconds=time.perf_counter() - started, protected_TEST_opened=False, fresh_EVAL_opened=False,
        limitations=['Legacy source coordinates reconstructed from existing normalized float32 NPZ; original raw bytes are not independently checked here.',
                     'This is a data contract check on previously exposed development data, not new forecasting evidence.'])


def audit_real_fixture(result_path):
    started = time.perf_counter()
    result_path = checked_path(result_path)
    result = json.loads(result_path.read_text(encoding='utf-8-sig'))
    if result.get('status') != 'COMPLETE' or result.get('stage') != 'fixture':
        raise ValueError('REAL_FIXTURE_REQUIRES_COMPLETED_TERMINAL_FIXTURE')
    audit = Audit()
    audit.check('terminal_no_failed_paths', not result['failed_paths'])
    audit.check('backbone_state_hash_terminal_invariance', result['backbone_before_sha256'] == result['backbone_after_sha256'])
    snapshot_root = ROOT / '.cache' / HERE.name / 'source_snapshots' / result['tag']
    manifest = json.loads((snapshot_root / 'MANIFEST.json').read_text(encoding='utf-8-sig'))
    audit.check('snapshot_manifest_matches_result_source_map', manifest['sha256'] == result['source_hashes'])
    for path, digest in result['source_hashes'].items():
        audit.check('source:' + path + ':snapshot_SHA', file_hash(snapshot_root / path) == digest)
        audit.check('source:' + path + ':current_SHA', file_hash(path) == digest)
    resource_path = audit.asset('terminal_gpu_resource', result['resource_receipt'])
    resource = json.loads(resource_path.read_text(encoding='utf-8-sig'))
    audit.check('resource_status_complete_gpu', resource['status'] == 'COMPLETE' and resource['gpu'] is True)
    audit.check('resource_process_wall_positive_bounded', 0 < resource['wall_seconds'] < 7200)
    audit.check('resource_no_process_RSS_VRAM_over_8GiB', resource['sampled_peak_rss'] <= 8 * 1024**3 and resource['peak_allocated_vram'] <= 8 * 1024**3)
    audit.check('resource_timestamp_brackets_stage', resource['started_unix'] <= result['started_unix'] <= result['finished_unix'] <= resource['finished_unix'])
    fixture = result['fixture']
    reference = legacy_reference(audit, fixture['dataset'])
    indices = np.array(fixture['fit_batch_indices'], np.int64)
    audit.check('actual_FIT_batch_origins_independent', np.array_equal(reference['origins']['fit'][indices], fixture['fit_batch_origins']))
    batch_info = batch_contract(reference['origins']['fit'], indices[None], expected_updates=1)
    selected_reference = {'mean': reference['mean'], 'std': reference['std'],
        'origins': {'fit': reference['origins']['fit'][indices]}, 'target': {'fit': reference['target']['fit'][indices]},
        'observed': {'fit': reference['observed']['fit'][indices]}}
    audit.check('fixture_backbone_before_after_equals_terminal', fixture['backbone_before_sha256'] == fixture['backbone_after_sha256'] == result['backbone_before_sha256'])
    audit.check('frozen_real_Bolt_input_gradient_positive_finite', np.isfinite(fixture['frozen_F_input_gradient_norm']) and fixture['frozen_F_input_gradient_norm'] > 0.)
    audit.check('identity_anchor_matches_RAW_with_declared_FP32_tolerance', 0 <= fixture['identity_anchor_vs_raw_max_abs'] <= 2e-6,
                {'actual': fixture['identity_anchor_vs_raw_max_abs'], 'atol': 2e-6})
    normalization = fixture['normalizer_fixture']
    for field, desired in (('constant_original_input_gradient_finite', False), ('constant_repaired_input_gradient_finite', True), ('pass', True)):
        audit.check('normalizer:' + field, normalization[field] is desired)
    audit.check('normalizer_forward_and_positive_variance_gradient_parity', normalization['all_forward_max_abs'] == normalization['positive_variance_gradient_max_abs'] == 0.)
    summaries = []
    audit.check('both_A_and_B_real_fixtures_present', {r['method'] for r in fixture['fixtures']} == {'B_GUMBEL', 'A_BIHASH_GUMBEL'} and len(fixture['fixtures']) == 2)
    for row in fixture['fixtures']:
        name = row['method']
        score = audit_prediction(audit, name + ':FIT_forecast', row['forecast_artifact'], {'fit': row['forecast_score']},
                                 selected_reference, require_increasing=False)
        steps = row['actual_three_optimizer_steps']
        audit.check(name + ':three_actual_recorded_updates', [s['step'] for s in steps] == [1, 2, 3])
        for step in steps:
            tag = name + ':step' + str(step['step'])
            audit.check(tag + ':all_losses_finite', np.isfinite([step[k] for k in ('main', 'aux', 'rec')]).all())
            for group in ('E', 'D', 'S_projection', 'Ws', 'Wf', 'phi'):
                norm = step['gradient_norms'][group]
                change = step['parameter_max_abs_from_initial'][group]
                audit.check(tag + ':' + group + ':nonzero_finite_gradient', np.isfinite(norm) and norm > 0., norm)
                audit.check(tag + ':' + group + ':nonzero_finite_parameter_update', np.isfinite(change) and change > 0., change)
        if name == 'B_GUMBEL':
            main = row['main_only']
            audit.check(name + ':actual_main_only_optimizer_step', main['actual_main_only_optimizer_step'] is True)
            for group in ('E', 'Ws', 'Wf', 'phi'):
                audit.check(name + ':main_only:' + group + ':gradient_and_update_positive',
                    main['gradient_norms'][group] > 0 and main['parameter_max_abs_from_initial'][group] > 0)
            audit.check(name + ':activated_raw_prefix_exact', row['active_temporal_raw_prefix_max_abs'] == 0.)
            audit.check(name + ':unavailable_future_memory_prefix_FP32_bounded', 0 <= row['unavailable_future_bank_prefix_max_abs'] <= 2e-6,
                {'actual': row['unavailable_future_bank_prefix_max_abs'], 'atol': 2e-6,
                 'scope': 'Producer compares original batch8 with changed batch1; this difference alone is not literal-zero causal proof.'})
            audit.check(name + ':allmask_fallback_exact', row['empty_eligible_memory_fallback_max_abs'] == 0.)
        audit.check(name + ':restored_state_hash', row['initial_hash'] == row['restored_hash'])
        audit.check(name + ':future_target_prediction_invariance_receipt', row['current_future_prediction_invariance'] is True)
        audit.check(name + ':dedicated_noise_CP_continuation_exact', row['noise_calls_checkpoint_continuation_max_abs'] == 0.)
        audit.check(name + ':AE_common_state_clone_receipt', row['future_AE_state_clone_pass'] is True)
        audit.check(name + ':real_Bolt_batch_FP32_tolerance', 0 <= row['batch_vs_single_max_abs'] <= 2e-5,
                    {'actual': row['batch_vs_single_max_abs'], 'atol': 2e-5})
        summaries.append({'method': name, 'fit_normalized_mse': score['fit']['normalized']['mse'],
                          'fit_normalized_mae': score['fit']['normalized']['mae'],
                          'bank_prefix_max_abs': row['unavailable_future_bank_prefix_max_abs'],
                          'source_replay_without_GPU': False})
    return audit.close(result={'path': str(result_path.relative_to(ROOT)), 'sha256': file_hash(result_path)},
        fixture_summaries=summaries, batch_contract=batch_info, resource_receipt=resource,
        cpu_wall_seconds=time.perf_counter() - started, future_EVAL_opened=False, protected_TEST_opened=False,
        generalization_verified=False, performance_improvement_verified=False,
        limitations=['Both saved FIT prediction NPZ files are independently rescored against reconstructed source targets/masks/statistics.',
                     'Real GPU gradient/step/restore/noise/AE-clone checks are checked against frozen producer source and terminal receipts; step tensors were not saved for independent GPU replay.',
                     'Independent synthetic CPU derivatives are exactly zero for ineligible memory; the real batch8/batch1 FP32 prefix test has nonzero bounded roundoff.',
                     'No actual 600-update experiment or independent EVAL result is established by this fixture.'])


def audit_case(audit, case, reference=None, eval_authorized=False):
    """Audit stored CP predictions and selection; no CP forward replay or training."""
    import torch
    name = case['dataset'] + ':' + case['label']
    training = case['training']
    checkpoints = training['checkpoints']
    audit.check(name + ':full600updates', training['updates'] == 600 and len(training['curve']) == 600)
    audit.check(name + ':actual_update_sequence', [r['update'] for r in training['curve']] == list(range(1, 601)))
    audit.check(name + ':all_five_checkpoints', sorted(int(c['update']) for c in checkpoints) == [0, 100, 200, 400, 600])
    audit.check(name + ':optimizer_and_clip_contract', training['optimizer'] == 'AdamW' and training['lr'] == .001 and training['weight_decay'] == 0.
                and training['codec_clip'] == 1. and training['side_clip'] == 1.)
    if all('total_loss' in r for r in training['curve']):
        measured = np.array([r['total_loss'] for r in training['curve']])
        reconstructed = np.array([r['main_loss'] + case['weight'] * r['aux_loss'] + .1 * r['future_rec_loss'] for r in training['curve']])
        audit.numeric(name + ':600_step_total_loss_contract', measured, reconstructed, atol=1e-6, rtol=2e-6)
        audit.check(name + ':600_step_finite_losses_gradient_norms', all(np.isfinite([r[k] for k in
                    ('main_loss', 'aux_loss', 'future_rec_loss', 'codec_unclipped_gradient_norm', 'side_unclipped_gradient_norm')]).all() for r in training['curve']))
        if case['method'] == 'A_BIHASH_DETACH_E':
            gradients = [r['aux_codec_unclipped_gradient_norm'] for r in training['curve'] if 'aux_codec_unclipped_gradient_norm' in r]
            audit.check(name + ':logged_aux_codec_gradient_exact_zero', len(gradients) == 5 and all(v == 0. for v in gradients))
    states = {}
    rescored_cps = []
    for cp in checkpoints:
        tag = name + ':cp' + str(cp['update'])
        path = audit.asset(tag, cp)
        saved = torch.load(path, map_location='cpu', weights_only=True)
        audit.check(tag + ':checkpoint_identity', saved['method'] == case['method'] and saved['seed'] == case['seed']
                    and saved['weight'] == case['weight'] and saved['update'] == cp['update'])
        if cp['update'] in (0, 600, training['selected_update']):
            states[cp['update']] = saved['model']
        score = audit_prediction(audit, tag + ':TUNE_NPZ', cp['tune_prediction'], {'tune': cp['tune']}, reference)
        rescored_cps.append({'update': cp['update'], 'tune': score['tune']})
        if 'bank' in (reference or {}):
            for field, key in (('delta', 'bank_delta'), ('observed', 'bank_observed'), ('origins', 'bank_origins')):
                audit.check(tag + ':' + key + '_raw_contract', np.array_equal(saved['model'][key].numpy(), reference['bank'][field]))
    selected = checkpoint_choice(rescored_cps)
    audit.check(name + ':selected_checkpoint_TUNE_argmin', selected == training['selected_update'], {'independent': selected, 'declared': training['selected_update']})
    chosen = next(c for c in checkpoints if c['update'] == selected)
    audit.check(name + ':selected_checkpoint_SHA_matches_argmin', chosen['sha256'] == training['selected_checkpoint']['sha256'])
    audit.numeric(name + ':selected_TUNE_mse', next(c['tune']['normalized']['mse'] for c in rescored_cps if c['update'] == selected), case['selected_tune_mse'])
    if selected not in states:
        states[selected] = torch.load(checked_path(chosen['path']), map_location='cpu', weights_only=True)['model']
    audit.check(name + ':selected_state_hash', tensor_state_hash(states[selected]) == training['selected_hash'])
    forecast = audit_prediction(audit, name + ':SELECTED_FIT_TUNE', case['predictions'], case['scores'], reference)
    cp_arrays = load_prediction(chosen['tune_prediction']['path'], ('tune',))
    selected_arrays = load_prediction(case['predictions']['path'], ('tune',))
    for field in ('target', 'observed', 'origins'):
        audit.check(name + ':reinference_same_' + field, np.array_equal(cp_arrays['tune'][field], selected_arrays['tune'][field]))
    audit.check(name + ':reinference_same_FIT_statistics', np.array_equal(cp_arrays['mean'], selected_arrays['mean']) and np.array_equal(cp_arrays['std'], selected_arrays['std']))
    a, b = selected_arrays['tune']['prediction'], cp_arrays['tune']['prediction']
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    audit.numeric(name + ':distinct_FP32_inference_array_parity', a, b, atol=1e-5, rtol=2e-6)
    bound = mse_difference_bound(a, b, cp_arrays['tune']['target'], cp_arrays['tune']['observed'])
    measured = abs(forecast['tune']['normalized']['mse'] - next(c['tune']['normalized']['mse'] for c in rescored_cps if c['update'] == selected))
    audit.check(name + ':reinference_normalized_MSE_delta_within_observed_macro_bound', measured <= bound + 1e-12,
                {'actual_abs_difference': measured, 'observed_channel_macro_bound': bound})
    mean, std = cp_arrays['mean'], cp_arrays['std']
    physical_bound = mse_difference_bound(a * std + mean, b * std + mean, cp_arrays['tune']['target'] * std + mean, cp_arrays['tune']['observed'])
    physical_delta = abs(forecast['tune']['mse'] - next(c['tune']['mse'] for c in rescored_cps if c['update'] == selected))
    audit.check(name + ':reinference_physical_MSE_delta_within_observed_macro_bound', physical_delta <= physical_bound + 1e-8,
                {'actual_abs_difference': physical_delta, 'observed_channel_macro_bound': physical_bound})
    if 'evaluation' in case:
        item = case['evaluation']
        audit_prediction(audit, name + ':DEVVAL', item['prediction'], {'val': item['scores']}, reference)
        audit.check(name + ':DEVVAL_declared_development', bool(item.get('repeated_development')))
    if reference is not None:
        batch = batch_contract(reference['origins']['fit'], training['batch_indices'])
        audit.check(name + ':batch_order_SHA', batch['array_sha256'] == training['order_sha256'])
    before, after = states[0], states[600]
    counts = {}
    for group, prefixes in {
        'E': ('codec.codec.project.', 'codec.codec.input.', 'codec.codec.temporal.', 'codec.codec.output.'),
        'D': ('codec.codec.decode.', 'codec.residual.'), 'Wp': ('student_hash.',), 'Wf': ('future_hash.',),
        'phi': ('future_autoencoder.encoder.',), 'fusion': ('student_fusion.', 'memory_fusion.', 'alpha_logit'),
    }.items():
        counts[group] = sum(int((v != after[k]).sum()) for k, v in before.items() if k.startswith(prefixes))
    fixed = 'FIXED' in case['method'] or case['method'] in ('NO_ALIGN', 'B_UNIFORM')
    for group in ('Wp', 'Wf'):
        audit.check(name + ':' + group + '_updates_expected', counts[group] == 0 if fixed else counts[group] > 0, counts[group])
    bias_changes = {key: {'changed_scalar_count': int((before[key] != after[key]).sum()),
                         'max_abs_u0_to_u600': float((before[key] - after[key]).abs().max())}
                    for key in ('codec.codec.decode.bias', 'codec.residual.2.bias') if key in before}
    return {'dataset': case['dataset'], 'label': case['label'], 'rescore_selected_update': selected,
            'changed_parameter_scalar_counts_u0_to_u600': counts,
            'mathematically_cancelled_decoder_bias_actual_changes': bias_changes,
            'bias_note': 'Structural algebraic cancellation does not imply saved FP32 bias values or accumulated optimizer updates are exactly zero.',
            'distinct_inference_parity': {'max_abs_prediction_difference': float(delta.max()), 'mean_abs_prediction_difference': float(delta.mean()),
                'normalized_MSE_difference': measured, 'normalized_MSE_difference_bound': bound, 'physical_MSE_difference': physical_delta,
                'physical_MSE_difference_bound': physical_bound, 'atol': 1e-5, 'rtol': 2e-6,
                'condition_difference': 'CP predicts with adaptation requires_grad True; selected predicts with requires_grad False; both no_grad/batch8.',
                'cause': 'Unestablished. Different parameter flags are a verified condition difference, not a runtime kernel causal proof.',
                'tolerance_basis': 'Existing real Bolt fixture allowed 2e-5 batch absolute drift; distinct-output bound1e-5 is stricter. Jena51 measured max3.4570694e-6.',
                'prior_strict_score_failures_preserved': 'validation_v1/DEV_JENA_51_AUDIT_v1.json',
                'prior_array_candidate_failures_preserved': 'validation_v1/DEV_JENA_SELECTED_REINFERENCE_DIFF_v1.json',
                'same_NPZ_JSON_rescore_remains_strict': True, 'original_CP_TUNE_argmin_unchanged': True},
            'checkpoint_predictions_verified': len(rescored_cps),
            'checkpoint_forward_replayed': False}


def audit_case_manifests(paths, source_tag):
    started = time.perf_counter()
    audit = Audit()
    snapshot_root = ROOT / '.cache' / HERE.name / 'source_snapshots' / source_tag
    manifest = json.loads((snapshot_root / 'MANIFEST.json').read_text(encoding='utf-8-sig'))
    for path, digest in manifest['sha256'].items():
        audit.check('source:' + path + ':snapshot_SHA', file_hash(snapshot_root / path) == digest)
        audit.check('source:' + path + ':current_SHA', file_hash(path) == digest)
    cases = []
    source_artifacts = []
    for path in paths:
        path = checked_path(path)
        case = json.loads(path.read_text(encoding='utf-8-sig'))
        if case.get('dataset') not in ('jena', 'robin'):
            raise PermissionError('FRESH_CASE_ADAPTER_AWAITS_EXPLICIT_DATA_CONTRACT_SCHEMA')
        if 'training' not in case or case['training']['updates'] != 600:
            raise ValueError('ONLY_COMPLETED_600UPDATE_CASE_MANIFESTS')
        audit.check('case:' + case['label'] + ':written_after_source_snapshot', path.stat().st_mtime >= manifest['created_unix'])
        cases.append(case)
        source_artifacts.append({'path': str(path.relative_to(ROOT)), 'sha256': file_hash(path)})
    references = {ds: legacy_reference(audit, ds) for ds in sorted({c['dataset'] for c in cases})}
    details = [audit_case(audit, case, references[case['dataset']]) for case in cases]
    for ds in references:
        for seed in sorted({c['seed'] for c in cases if c['dataset'] == ds}):
            current = [c for c in cases if c['dataset'] == ds and c['seed'] == seed]
            audit.check(ds + ':seed' + str(seed) + ':common_initial_state', len({c['initial_hash'] for c in current}) == 1)
            audit.check(ds + ':seed' + str(seed) + ':common_actual_batch_order', len({c['training']['order_sha256'] for c in current}) == 1)
            audit.check(ds + ':seed' + str(seed) + ':common_future_pretrain', len({c['future_pretrain']['sha256'] for c in current}) == 1)
    return audit.close(case_manifest_artifacts=source_artifacts, cases=details, cpu_wall_seconds=time.perf_counter() - started,
        source_snapshot={'path': str((snapshot_root / 'MANIFEST.json').relative_to(ROOT)), 'sha256': file_hash(snapshot_root / 'MANIFEST.json')},
        artifact_case_scope_complete=True, whole_experiment_complete=False, fresh_EVAL_opened=False, protected_TEST_opened=False,
        limitations=['Checks cover only the explicitly supplied completed case manifests, not unfinished runs or full intended scope.',
                     'Stored CP predictions are independently rescored; the frozen Bolt forward is not replayed on CPU.',
                     'TUNE checkpoint selection is development selection, not evidence of independent generalization.'])


PRIMARY_METHODS = ('NO_ALIGN', 'B_COSINE', 'B_TANH', 'B_GUMBEL', 'B_ST', 'B_FIXED_GUMBEL', 'B_UNIFORM',
                   'A_CONTINUOUS', 'A_BIHASH_GUMBEL', 'A_BIHASH_SHUFFLE', 'A_BIHASH_DETACH_E', 'A_BIHASH_FIXED_MAPS')
PRIMARY_SEEDS = (92601, 92602, 92603)
LEGACY_DATASETS = ('jena', 'robin')
FRESH_DATASETS = ('uci235_household_hourly', 'uci274_sml2010')
PROBE_METHODS = ('NO_ALIGN', 'B_GUMBEL', 'B_COSINE', 'B_UNIFORM', 'A_BIHASH_GUMBEL', 'A_CONTINUOUS')


def exact_coverage(rows, expected, fields):
    actual = [tuple(row.get(field) for field in fields) for row in rows]
    expected = set(expected)
    return {'passed': set(actual) == expected and len(actual) == len(expected),
            'expected_count': len(expected), 'actual_count': len(actual),
            'duplicates': len(actual) - len(set(actual)),
            'missing': sorted(expected - set(actual), key=str), 'unexpected': sorted(set(actual) - expected, key=str)}


def raw_fresh_source(audit, contract):
    """Independent ingestion of declared source bytes; post-EVAL audit only."""
    import io
    import zipfile
    import pandas as pd
    dataset = contract['dataset']
    sources = [audit.asset(dataset + ':raw_source:' + str(i), r) for i, r in enumerate(contract['source_records'])]
    columns = contract['columns']
    if dataset == FRESH_DATASETS[0]:
        if len(sources) != 1:
            raise ValueError('HOUSEHOLD_SOURCE_AMBIGUOUS')
        path = sources[0]
        if path.suffix.lower() == '.zip':
            with zipfile.ZipFile(path) as archive:
                members = [n for n in archive.namelist() if Path(n).name == 'household_power_consumption.txt']
                if len(members) != 1:
                    raise ValueError('HOUSEHOLD_MEMBER_AMBIGUOUS')
                handle = io.BytesIO(archive.read(members[0]))
        else:
            handle = path
        frame = pd.read_csv(handle, sep=';', na_values=['?', ''], low_memory=False)
        times = pd.to_datetime(frame['Date'].astype(str) + ' ' + frame['Time'].astype(str), format='%d/%m/%Y %H:%M:%S', errors='raise').to_numpy()
        sort = np.argsort(times)
        times = times[sort]
        values = frame[columns].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64)[sort]
        minute = times.astype('datetime64[m]').astype(np.int64)
        audit.check(dataset + ':raw_minute_grid_unique_complete', bool(np.all(np.diff(minute) == 1)))
        hour = minute // 60
        starts = np.r_[0, np.flatnonzero(np.diff(hour)) + 1]
        ends = np.r_[starts[1:], len(hour)]
        complete = ends - starts == 60
        output = []
        hour_times = []
        for lo, hi in zip(starts[complete], ends[complete]):
            group = values[lo:hi]
            row = []
            for c in range(len(columns)):
                observed = group[:, c][np.isfinite(group[:, c])]
                row.append(observed.mean() if len(observed) else np.nan)
            output.append(row)
            hour_times.append(np.datetime64(int((hour[lo] + 1) * 60), 'm'))
        physical = np.array(output, np.float32)
        timestamps = np.array(hour_times).astype('datetime64[s]').astype('U19')
        audit.check(dataset + ':complete_hour_count_contract', len(physical) == contract['parser_metadata']['hourly_rows'])
        audit.check(dataset + ':minute_rows_contract', len(values) == contract['parser_metadata']['raw_minute_rows'])
        audit.check(dataset + ':incomplete_boundary_hours_contract', int((~complete).sum()) == contract['parser_metadata']['structural_boundary_hours_excluded'])
    elif dataset == FRESH_DATASETS[1]:
        if len(columns) != 21:
            raise ValueError('SML21_CHANNEL_CONTRACT')
        names = ('NEW-DATA-1.T15.txt', 'NEW-DATA-2.T15.txt')
        tables = []
        if len(sources) == 1 and sources[0].suffix.lower() == '.zip':
            with zipfile.ZipFile(sources[0]) as archive:
                for name in names:
                    matches = [n for n in archive.namelist() if Path(n).name == name]
                    if len(matches) != 1:
                        raise ValueError('SML_MEMBER_AMBIGUOUS')
                    tables.append(pd.read_csv(io.BytesIO(archive.read(matches[0])), comment='#', header=None, sep=r'[;,\s]+', engine='python'))
        else:
            for name in names:
                matches = [p for p in sources if p.name == name]
                if len(matches) != 1:
                    raise ValueError('SML_FILE_AMBIGUOUS')
                tables.append(pd.read_csv(matches[0], comment='#', header=None, sep=r'[;,\s]+', engine='python'))
        frame = pd.concat(tables, ignore_index=True).dropna(axis=1, how='all')
        audit.check(dataset + ':source24columns_excludes_date_time_dayweek', frame.shape[1] == 24 and len(columns) == 21)
        times = pd.to_datetime(frame.iloc[:, 0].astype(str) + ' ' + frame.iloc[:, 1].astype(str), utc=True, errors='raise')
        values = frame.iloc[:, 2:23].apply(pd.to_numeric, errors='coerce').to_numpy(np.float64)
        nanos = times.astype('int64').to_numpy()
        order = np.argsort(nanos, kind='stable'); nanos, values = nanos[order], values[order]
        unique, first, counts = np.unique(nanos, return_index=True, return_counts=True)
        for index, count in zip(first[counts > 1], counts[counts > 1]):
            audit.check(dataset + ':duplicate_values:' + str(int(nanos[index])),
                        all(np.array_equal(values[index], v, equal_nan=True) for v in values[index:index + count]))
        chosen = values[first]
        interval = 15 * 60 * 10**9
        audit.check(dataset + ':source_rows_on_15min_grid', bool(np.all((unique - unique[0]) % interval == 0)))
        grid = np.arange(unique[0], unique[-1] + interval, interval, dtype=np.int64)
        physical = np.full((len(grid), 21), np.nan, np.float32)
        physical[((unique - grid[0]) // interval).astype(int)] = chosen.astype(np.float32)
        timestamps = pd.to_datetime(grid, utc=True).strftime('%Y-%m-%dT%H:%M:%SZ').to_numpy(dtype='U20')
        audit.check(dataset + ':duplicate_count_contract', int((counts - 1).sum()) == contract['parser_metadata']['identical_duplicates_collapsed'])
        audit.check(dataset + ':grid_row_count_contract', len(grid) == contract['parser_metadata']['full_grid_rows'])
    else:
        raise ValueError('UNKNOWN_FRESH_SOURCE')
    observed = np.isfinite(physical)
    return physical, observed, timestamps


def fresh_reference(audit, contract, eval_authorized=False):
    if not eval_authorized:
        raise PermissionError('FRESH_RAW_AND_EVAL_AUDIT_REQUIRES_POST_FIRST_PASS_AUTHORIZATION')
    dataset = contract['dataset']
    raw, observed, times = raw_fresh_source(audit, contract)
    n = len(raw); fit_end, tune_end = int(.6 * n), int(.8 * n)
    bounds = {'fit': (0, fit_end), 'tune': (fit_end, tune_end), 'eval': (tune_end, n)}
    audit.check(dataset + ':row_split60_20_20', all(list(b) == contract['bounds'][s] for s, b in bounds.items()))
    mean, std = fit_statistics(raw, observed, bounds['fit'])
    origins = {}
    for split, (lo, hi) in bounds.items():
        stride = 48 if split == 'eval' else 1
        candidates = np.arange(max(512, lo), hi - 48 + 1, stride, dtype=np.int64)
        legal = np.array([o for o in candidates if observed[o:o + 48].any(0).all()], np.int64)
        maximum = 256 if split == 'fit' else 64
        positions = np.arange(len(legal)) if split == 'eval' or len(legal) <= maximum else np.linspace(0, len(legal) - 1, maximum).astype(np.int64)
        origins[split] = legal[positions]
        audit.check(dataset + ':' + split + ':origin_count', len(origins[split]) == contract['origins'][split]['count'])
        audit.check(dataset + ':' + split + ':origin_array_SHA', array_hash(origins[split]) == contract['origins'][split]['sha256'])
    reference = reference_from_physical(raw, observed, origins, mean, std)
    for field in ('mean', 'std'):
        audit.check(dataset + ':' + field + ':raw_FIT_SHA', array_hash(reference[field]) == contract['statistics'][field + '_sha256'])
    for field in ('basis', 'pca_mean'):
        audit.check(dataset + ':' + field + ':raw_FIT_PCA_SHA', array_hash(reference[field]) == contract['pca_initialization'][field + '_sha256'])
    for name, artifact in contract['payloads'].items():
        path = audit.asset(dataset + ':' + name, artifact)
        with np.load(path, allow_pickle=False) as z:
            for field in ('mean', 'std'):
                audit.check(dataset + ':' + name + ':' + field + ':raw_contract', np.array_equal(z[field], reference[field]))
            if name == 'fit_tune':
                audit.check(dataset + ':' + name + ':physical_source', np.array_equal(z['physical'], raw[:tune_end], equal_nan=True))
                audit.check(dataset + ':' + name + ':mask_source', np.array_equal(z['mask'], observed[:tune_end]))
                audit.check(dataset + ':' + name + ':times_source', np.array_equal(z['times'], times[:tune_end]))
            elif name == 'eval_past_input':
                audit.check(dataset + ':' + name + ':history_raw_contract', np.array_equal(z['history'], reference['history']['eval']))
                audit.check(dataset + ':' + name + ':physical_history_raw', np.array_equal(z['physical_history'], np.stack([raw[o - 512:o] for o in origins['eval']]), equal_nan=True))
                audit.check(dataset + ':' + name + ':past_mask_raw', np.array_equal(z['history_observed'], np.stack([observed[o - 512:o] for o in origins['eval']])))
                audit.check(dataset + ':' + name + ':origins_raw', np.array_equal(z['origins'], origins['eval']))
            elif name == 'eval_future_target':
                for field, stored in (('target', 'target'), ('observed', 'observed'), ('origins', 'origins')):
                    audit.check(dataset + ':' + name + ':' + field + ':raw_contract', np.array_equal(z[stored], reference[field]['eval']))
                audit.check(dataset + ':' + name + ':physical_target_no_fill', np.array_equal(z['target_physical'], np.stack([raw[o:o + 48] for o in origins['eval']]), equal_nan=True))
    return reference


def deployment_parity(actual_physical, science, mean, std):
    actual, science = np.asarray(actual_physical), np.asarray(science)
    mean, std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
    if actual.shape != science.shape or actual.ndim != 3 or not np.isfinite(actual).all() or not np.isfinite(science).all():
        raise ValueError('INVALID_DEPLOYMENT_OUTPUT_ARRAYS')
    reference = science.astype(np.float32) * std + mean
    error = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    allowance = .002 + .00002 * np.abs(reference.astype(np.float64))
    normalized_error = np.abs((actual.astype(np.float64) - mean.astype(np.float64)) / std.astype(np.float64) - science.astype(np.float64))
    return {'physical_parity_pass': bool(np.all(error <= allowance)), 'physical_atol': .002, 'physical_rtol': .00002,
            'physical_max_abs': float(error.max()), 'physical_mean_abs': float(error.mean()),
            'physical_max_error_to_allowance': float((error / allowance).max()), 'failed_physical_cells': int((error > allowance).sum()),
            'normalized_max_abs': float(normalized_error.max()), 'normalized_mean_abs': float(normalized_error.mean()),
            'normalized_per_channel_max_abs': normalized_error.max((0, 1)).tolist(), 'total_output_cells': int(actual.size)}


def audit_deployment_outputs(audit, result_path, expected, references, eval_authorized=False, phase_artifacts=None, cost=None, method_lock=None):
    if not eval_authorized:
        raise PermissionError('DEPLOYMENT_AUDIT_REQUIRES_POST_FIRST_PASS_AUTHORIZATION')
    path = checked_path(result_path)
    result = json.loads(path.read_text(encoding='utf-8-sig'))
    if result.get('status') not in ('COMPLETE', 'FAILED_PRESERVED', 'BUDGET_HELD_PRESERVED'):
        raise ValueError('DEPLOYMENT_RESULT_NOT_TERMINAL')
    audit.check('deployment:terminal_complete', result['status'] == 'COMPLETE' and not result['failed_routes'])
    coverage = exact_coverage(result['records'], expected, ('dataset', 'identity', 'route', 'batch'))
    audit.check('coverage:deployment536:exact', coverage['passed'], coverage)
    audit.check('deployment:no_new_training_selection', result['new_training_updates'] == 0 and result['new_model_or_weight_selection'] is False)
    audit.check('deployment:backbone_state_unchanged', result['backbone_before_sha256'] == result['backbone_after_sha256'])
    audit.check('deployment:producer_script_SHA', file_hash(HERE / 'verify_deployment_outputs.py') == result['script_sha256'])
    if method_lock is not None:
        audit.check('deployment:scientific_source_map', result['scientific_source_hashes'] == method_lock['source_hashes'])
    if phase_artifacts is not None:
        for phase, artifact in phase_artifacts.items():
            key = 'probes' if phase == 'probe' else phase
            audit.check('deployment:source_result:' + phase, result['source_results'][key]['sha256'] == artifact['sha256'])
    cost_rows = {} if cost is None else {(r['dataset'], r['identity'], r['route'], r['batch']): r for r in cost['records']}
    failed_cells, max_ratio = 0, 0.
    for row in result['records']:
        name = 'deployment:' + row['dataset'] + ':' + row['identity'] + ':batch' + str(row['batch'])
        artifact = audit.asset(name + ':actual_output', row['actual_output'])
        ref = references[row['dataset']]; split = row['split']; n = row['batch']
        original = audit.asset(name + ':science_source', row['science_prediction'])
        with np.load(original, allow_pickle=False) as z:
            science = z[split + '_prediction'][:n].copy()
        with np.load(artifact, allow_pickle=False) as z:
            for field in ('mean', 'std'):
                audit.check(name + ':' + field + '_source', np.array_equal(z[field], ref[field]))
            audit.check(name + ':origins_source', np.array_equal(z['origins'], ref['origins'][split][:n]))
            audit.check(name + ':science_subset_exact', np.array_equal(z['science_prediction'], science))
            audit.check(name + ':raw_history_source', np.array_equal(z['raw_history'], ref['raw_history'][split][:n], equal_nan=True))
            audit.check(name + ':past_mask_source', np.array_equal(z['history_observed'], ref['history_observed'][split][:n]))
            audit.check(name + ':source_history_normalized', np.array_equal(z['source_history_normalized'], ref['history'][split][:n]))
            # Recompute FF from the raw arrays without the companion helper.
            filled = np.empty_like(z['raw_history'], dtype=np.float32)
            for b in range(n):
                last = ref['mean'].copy()
                for t in range(512):
                    mask = z['history_observed'][b, t]
                    last[mask] = z['raw_history'][b, t, mask]
                    filled[b, t] = (last - ref['mean']) / ref['std']
            audit.check(name + ':independent_fill_matches_source_and_companion', np.array_equal(filled, z['source_history_normalized']) and np.array_equal(filled, z['independent_history_normalized']))
            parity = deployment_parity(z['actual_physical'], z['science_prediction'], z['mean'], z['std'])
            audit.check(name + ':independent_percell_parity', parity['physical_parity_pass'])
            for field, actual in parity.items():
                if isinstance(actual, bool):
                    audit.check(name + ':reported:' + field, actual == row['independent_CPU_parity'][field])
                else:
                    audit.numeric(name + ':reported:' + field, actual, row['independent_CPU_parity'][field])
            failed_cells += parity['failed_physical_cells']; max_ratio = max(max_ratio, parity['physical_max_error_to_allowance'])
        for model in row['model_artifacts']:
            audit.asset(name + ':model:' + model['path'], model)
        audit.check(name + ':one_posthoc_forward_separate_from_timing', row['new_actual_forward_calls'] == 1 and row['not_added_to_latency_statistics'] is True)
        audit.check(name + ':F_state_matches_record', row['backbone_sha256'] == result['backbone_before_sha256'])
        if cost is not None:
            timing = cost_rows[(row['dataset'], row['identity'], row['route'], n)]
            audit.numeric(name + ':unchanged_seven_repeat_median', row['original_seven_repeat_timing']['median_seconds'], timing['median_seconds'])
        active = row['active_model_metadata']
        audit.check(name + ':active_frozen_F_route', active['frozen_backbone_active_at_route'] == (row['route'] in ('CORE', 'RAW_BOLT', 'RAW_BOLT_RAW_R')))
    return {'passed': coverage['passed'] and failed_cells == 0 and result['status'] == 'COMPLETE',
            'result': {'path': str(path.relative_to(ROOT)), 'sha256': file_hash(path)},
            'saved_actual_batch_outputs': len(result['records']), 'failed_physical_cells': failed_cells,
            'maximum_error_to_declared_allowance': max_ratio,
            'scope': 'Separate post-cost single-call deployment outputs; original seven timed outputs are not persisted.'}


def audit_dev_result(path):
    started = time.perf_counter()
    path = checked_path(path)
    result = json.loads(path.read_text(encoding='utf-8-sig'))
    if result.get('stage') != 'dev' or result.get('status') != 'COMPLETE':
        raise ValueError('COMPLETED_DEV_ONLY')
    import torch
    audit = Audit()
    audit.check('DEV:no_failed_paths', not result['failed_paths'])
    resource_path = audit.asset('DEV:terminal_resource', result['resource_receipt'])
    resource = json.loads(resource_path.read_text(encoding='utf-8-sig'))
    audit.check('DEV:actual_terminal_GPU_receipt', resource['status'] == 'COMPLETE' and resource['gpu'] is True)
    audit.check('DEV:F_weights_unchanged', bool(result['backbone_before_sha256']) and result['backbone_before_sha256'] == result['backbone_after_sha256'])
    method_path = audit.asset('METHOD_LOCK', result['method_lock'])
    method_lock = json.loads(method_path.read_text(encoding='utf-8-sig'))
    audit.check('METHOD_LOCK:locked', method_lock['status'] == 'LOCKED' and method_lock['locked'] is True)
    audit.check('METHOD_LOCK:created_before_DEV_terminal', result['started_unix'] < method_lock['created_unix'] <= result['finished_unix'])
    snapshot = ROOT / '.cache' / HERE.name / 'source_snapshots' / result['tag']
    snapshot_path = snapshot / 'MANIFEST.json'
    manifest = json.loads(snapshot_path.read_text(encoding='utf-8-sig'))
    audit.check('DEV:source_map_snapshot', manifest['sha256'] == result['source_hashes'])
    audit.check('DEV:source_map_METHOD_LOCK', method_lock['source_hashes'] == result['source_hashes'])
    for source, digest in result['source_hashes'].items():
        audit.check('DEV:source:' + source + ':snapshot_SHA', file_hash(snapshot / source) == digest)
        audit.check('DEV:source:' + source + ':current_SHA', file_hash(source) == digest)
    expected = {(ds, seed, method, weight) for ds in LEGACY_DATASETS for seed in PRIMARY_SEEDS for method in PRIMARY_METHODS
                for weight in ((.03, .1) if method.startswith('A_') else (0.,))}
    coverage = exact_coverage(result['cases'], expected, ('dataset', 'seed', 'method', 'weight'))
    audit.check('DEV:coverage102:exact', coverage['passed'], coverage)
    references = {ds: legacy_reference(audit, ds, result['data'][ds]) for ds in LEGACY_DATASETS}
    details, case_artifacts = [], []
    for case in result['cases']:
        case_path = path.parent / case['dataset'] / (case['label'] + '.json')
        saved_case = json.loads(case_path.read_text(encoding='utf-8-sig'))
        original = {k: v for k, v in case.items() if k != 'evaluation'}
        audit.check('DEV:' + case['label'] + ':original_case_json_matches', saved_case == original)
        selected = case['weight'] == method_lock['weights'][case['method']]
        audit.check('DEV:' + case['label'] + ':only_TUNE_selected_case_has_DEVVAL', ('evaluation' in case) == selected)
        if 'evaluation' in case:
            val_path = case_path.with_name(case['label'] + '_DEVVAL.json')
            val = json.loads(val_path.read_text(encoding='utf-8-sig'))
            audit.check('DEV:' + case['label'] + ':separate_DEVVAL_receipt_matches', val == case['evaluation'])
            audit.check('DEV:' + case['label'] + ':DEVVAL_after_METHOD_LOCK', method_lock['created_unix'] < val_path.stat().st_mtime <= result['finished_unix'])
            case_artifacts.append({'path': str(val_path.relative_to(ROOT)), 'sha256': file_hash(val_path)})
        audit.check('DEV:' + case['label'] + ':case_before_METHOD_LOCK', case_path.stat().st_mtime < method_lock['created_unix'])
        case_artifacts.append({'path': str(case_path.relative_to(ROOT)), 'sha256': file_hash(case_path)})
        details.append(audit_case(audit, case, references[case['dataset']]))
    for ds in LEGACY_DATASETS:
        for seed in PRIMARY_SEEDS:
            current = [c for c in result['cases'] if c['dataset'] == ds and c['seed'] == seed]
            for field, key in (('initial_hash', 'initial_hash'), ('batch_order', 'order_sha256'), ('AE_clone', 'sha256')):
                values = [c[key] if field == 'initial_hash' else c['training' if field == 'batch_order' else 'future_pretrain'][key] for c in current]
                audit.check(ds + ':seed' + str(seed) + ':common_' + field, len(set(values)) == 1)
    recomputed = method_selection(result['cases'], {(d, s) for d in LEGACY_DATASETS for s in PRIMARY_SEEDS}, PRIMARY_METHODS)
    audit.check('METHOD_LOCK:weights_TUNE_recomputed', recomputed['weights'] == method_lock['weights'])
    audit.check('METHOD_LOCK:selected_B_TUNE_recomputed', recomputed['selected_B_method'] == method_lock['selected_B_method'])
    pretrains = result['future_pretrains']
    ae_coverage = exact_coverage(pretrains, {(d, s) for d in LEGACY_DATASETS for s in PRIMARY_SEEDS}, ('dataset', 'seed'))
    audit.check('DEV:AE6:exact', ae_coverage['passed'], ae_coverage)
    for row in pretrains:
        name = row['dataset'] + ':AE:' + str(row['seed'])
        audit.check(name + ':actual400curve', row['updates'] == 400 and [r['update'] for r in row['curve']] == list(range(1, 401)))
        audit.check(name + ':fixed_final_selection', row['selection'] == 'fixed final 400, TUNE diagnostic only')
        order = np.random.default_rng(row['seed'] + 10000).integers(0, len(references[row['dataset']]['origins']['fit']), size=(400, 64))
        audit.check(name + ':FIT_order_SHA', array_hash(order) == row['order_sha256'])
        checkpoint_path = audit.asset(name + ':checkpoint', row['checkpoint'])
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        audit.check(name + ':saved400identity', saved['update'] == 400 and saved['seed'] == row['seed'])
        audit.check(name + ':final_state_hash', tensor_state_hash(saved['future_autoencoder']) == row['final_hash'])
        steps = [int(v['step']) for v in saved['optimizer']['state'].values()]
        audit.check(name + ':optimizer_actual400steps', bool(steps) and all(s == 400 for s in steps))
        for case in (c for c in result['cases'] if c['dataset'] == row['dataset'] and c['seed'] == row['seed']):
            cp0 = torch.load(checked_path(case['training']['checkpoints'][0]['path']), map_location='cpu', weights_only=True)['model']
            clone = {k.removeprefix('future_autoencoder.'): v for k, v in cp0.items() if k.startswith('future_autoencoder.')}
            audit.check(name + ':clone_to:' + case['label'], tensor_state_hash(clone) == row['final_hash'])
    return audit.close(phase_artifact={'path': str(path.relative_to(ROOT)), 'sha256': file_hash(path)},
        case_manifest_artifacts=case_artifacts, cases=details, coverage=coverage, AE_coverage=ae_coverage,
        method_lock=result['method_lock'], independently_recomputed_selection=recomputed,
        checkpoint_predictions_verified=sum(d['checkpoint_predictions_verified'] for d in details),
        source_snapshot={'path': str(snapshot_path.relative_to(ROOT)), 'sha256': file_hash(snapshot_path)},
        resource_artifact=result['resource_receipt'], cpu_wall_seconds=time.perf_counter() - started,
        preserved_prior_attempt='validation_v1/DEV_COMPLETE_102_AUDIT_v1.json',
        artifact_DEV_scope_complete=coverage['passed'] and ae_coverage['passed'], whole_experiment_complete=False,
        fresh_EVAL_opened=False, protected_TEST_opened=False, GPU_executed=False,
        limitations=['This receipt covers completed DEV102, CP510 and fixed-final AE6; fresh training, probes, EVAL and costs are outside scope.',
                    'Each saved prediction is rescored independently; no frozen Bolt or historical GPU optimizer forward is replayed.',
                    'Legacy VAL was already exposed and does not constitute independent fresh generalization.',
                    'A auxiliary heads are logically inactive at deployment but remain co-resident in saved checkpoints and GPU objects.'])


def audit_bias_diagnostic(path):
    started = time.perf_counter()
    path = checked_path(path)
    result = json.loads(path.read_text(encoding='utf-8-sig'))
    if result.get('status') != 'COMPLETE' or result.get('GPU_executed') is not False:
        raise ValueError('COMPLETED_CPU_DIAGNOSTIC_ONLY')
    import torch
    torch.set_num_threads(2)
    audit = Audit()
    for key in ('protected_TEST_opened', 'EVAL_opened', 'GPU_executed', 'training_source_changed', 'checkpoint_modified'):
        audit.check('diagnostic:' + key + ':false', result[key] is False)
    audit.check('diagnostic:synthetic_latents_declared', result['synthetic_latent_provenance']['actual_E_or_F_latents'] is False)
    for source, digest in result['source_hashes'].items():
        audit.check('diagnostic:source_SHA:' + source, file_hash(source) == digest)
    artifacts = {key: audit.asset('diagnostic:source:' + key, record) for key, record in result['source_artifacts'].items()}
    checkpoints = {u: torch.load(artifacts['checkpoint_u' + str(u)], map_location='cpu', weights_only=True) for u in (0, 600)}
    input_path = audit.asset('diagnostic:INPUTS', result['input_artifact'])
    with np.load(input_path, allow_pickle=False) as store:
        inputs = {k: store[k].copy() for k in store.files}
    with np.load(artifacts['FIT_prediction_target'], allow_pickle=False) as source:
        indices = np.asarray(result['actual_FIT_indices'], np.int64)
        for field in ('prediction', 'target', 'observed', 'origins'):
            audit.check('diagnostic:actual_FIT_' + field, np.array_equal(inputs['actual_FIT_' + field], source['fit_' + field][indices]))
        for field in ('mean', 'std'):
            audit.check('diagnostic:actual_FIT_' + field, np.array_equal(inputs[field], source[field]))
    generator = np.random.default_rng(result['synthetic_latent_provenance']['seed'])
    shape = inputs['synthetic_anchor_latent'].shape
    anchor = generator.normal(0., .7, shape).astype(np.float32)
    future = (anchor + generator.normal(0., .7, inputs['synthetic_future_latent'].shape).astype(np.float32)).astype(np.float32)
    audit.check('diagnostic:synthetic_anchor_seed', np.array_equal(anchor, inputs['synthetic_anchor_latent']))
    audit.check('diagnostic:synthetic_future_seed', np.array_equal(future, inputs['synthetic_future_latent']))
    spec = importlib.util.spec_from_file_location('independent_bias_parameter_order', HERE / 'model.py')
    bm = importlib.util.module_from_spec(spec); spec.loader.exec_module(bm)
    model = bm.FutureBiHash(np.zeros(len(inputs['mean']), np.float32), checkpoints[0]['model']['codec.fixed_basis'].numpy(), 92601, 'NO_ALIGN')
    parameter_names = [n for n, _ in model.named_parameters()]
    local_to_saved = {'linear_bias': 'codec.codec.decode.bias', 'residual_2_bias': 'codec.residual.2.bias'}
    indexes = {}
    for update, checkpoint in checkpoints.items():
        selected_names = [n for n in checkpoint['model'] if n in dict(model.named_parameters())]
        ids = checkpoint['optimizer']['param_groups'][0]['params']
        audit.check('diagnostic:u' + str(update) + ':optimizer_actual_named_parameter_order', selected_names == parameter_names and len(ids) == len(parameter_names))
        indexes[update] = dict(zip(parameter_names, ids))
        for saved_name in local_to_saved.values():
            actual = (checkpoint['model'][saved_name] - checkpoints[0]['model'][saved_name]).abs()
            if update == 600:
                claim = result['actual_saved_bias_changes_u0_to_u600'][saved_name]
                audit.check(saved_name + ':actual_saved_changed_count', int((actual != 0).sum()) == claim['changed_scalar_count'])
                audit.numeric(saved_name + ':actual_saved_max_delta', float(actual.max()), claim['max_abs'])
    expected = {(u, dtype, batch, mode, form) for u in (0, 600) for dtype in ('torch.float32', 'torch.float64') for batch in (1, 8)
                for mode in ('synthetic_latent_forward', 'saved_FIT_residual_Jacobian_replay') for form in ('original', 'bias_free')}
    coverage = exact_coverage(result['rows'], expected, ('update', 'dtype', 'batch', 'mode', 'form'))
    audit.check('diagnostic:coverage32:exact', coverage['passed'], coverage)
    details = []
    for row in result['rows']:
        name = str((row['update'], row['dtype'], row['batch'], row['mode'], row['form']))
        arrays_path = audit.asset(name + ':arrays', row['arrays'])
        with np.load(arrays_path, allow_pickle=False) as store:
            arrays = {k: store[k].copy() for k in store.files}
        dtype = torch.float32 if row['dtype'] == 'torch.float32' else torch.float64
        upstream = torch.from_numpy(arrays['upstream_prediction_gradient'])
        flat = upstream.reshape(-1, upstream.shape[-1]).sum(0)
        nested = upstream.sum(1).sum(0)
        delta = flat - nested
        for field, expected_array in (('flat_BH_reduction', flat.numpy()), ('nested_H_then_B_reduction', nested.numpy()), ('reduction_difference', delta.numpy())):
            audit.check(name + ':' + field + ':independent_Torch_reduction', np.array_equal(arrays[field], expected_array))
        n = row['batch']; prediction = arrays['prediction']
        if row['mode'] == 'saved_FIT_residual_Jacobian_replay':
            audit.check(name + ':actual_saved_FIT_prediction_exact', np.array_equal(prediction, inputs['actual_FIT_prediction'][:n].astype(prediction.dtype)))
        score = macro_score(prediction, inputs['actual_FIT_target'][:n], inputs['actual_FIT_observed'][:n])
        audit.numeric(name + ':loss_independent_NumPy', score['mse'], row['loss'], atol=2e-7 if dtype == torch.float32 else 1e-14, rtol=2e-6 if dtype == torch.float32 else 1e-12)
        mask = torch.from_numpy(inputs['actual_FIT_observed'][:n]); counts = mask.sum((0, 1))
        residual = torch.from_numpy(prediction) - torch.as_tensor(inputs['actual_FIT_target'][:n], dtype=dtype)
        analytic = torch.where(mask, 2 * residual / counts.clamp_min(1) / (counts > 0).sum(), 0.)
        audit.check(name + ':upstream_masked_macro_gradient', torch.allclose(analytic, upstream, atol=torch.finfo(dtype).eps * 1e-3, rtol=torch.finfo(dtype).eps * 8))
        for local, saved_name in local_to_saved.items():
            gradient = arrays[local + '_gradient']
            expected_gradient = delta.numpy() if row['form'] == 'original' else np.zeros_like(delta.numpy())
            audit.check(name + ':' + local + ':gradient_exact_reduction_contract', np.array_equal(gradient, expected_gradient))
            for mode in ('fresh_zero_moments', 'restored_saved_moments'):
                prefix = mode + '_' + local
                before = arrays[prefix + '_before']
                parameter = torch.nn.Parameter(torch.from_numpy(before.copy()))
                optimizer = torch.optim.AdamW([parameter], lr=.001, weight_decay=0., eps=1e-8)
                if mode == 'restored_saved_moments':
                    state = copy.deepcopy(checkpoints[row['update']]['optimizer']['state'].get(indexes[row['update']][saved_name], {}))
                    for key in ('exp_avg', 'exp_avg_sq'):
                        if key in state:
                            state[key] = state[key].to(dtype=dtype)
                    optimizer.state[parameter] = state
                moment = optimizer.state[parameter].get('exp_avg', torch.zeros_like(parameter)).detach().numpy().copy()
                audit.check(name + ':' + prefix + ':saved_parameter_exact', np.array_equal(before, checkpoints[row['update']]['model'][saved_name].to(dtype=dtype).numpy()))
                audit.check(name + ':' + prefix + ':saved_moment_exact', np.array_equal(moment, arrays[prefix + '_moment_before']))
                parameter.grad = None if row['form'] == 'bias_free' else torch.from_numpy(gradient.copy())
                optimizer.step()
                audit.check(name + ':' + prefix + ':independent_one_bias_AdamW_step_exact', np.array_equal(parameter.detach().numpy(), arrays[prefix + '_after']))
                audit.check(name + ':' + prefix + ':saved_step_delta_exact', np.array_equal(arrays[prefix + '_after'] - before, arrays[prefix + '_step_delta']))
                after_moment = optimizer.state[parameter].get('exp_avg', torch.zeros_like(parameter)).detach().numpy()
                audit.check(name + ':' + prefix + ':independent_updated_moment_exact', np.array_equal(after_moment, arrays[prefix + '_moment_after']))
        numpy_delta = arrays['upstream_prediction_gradient'].reshape(-1, prediction.shape[-1]).sum(0) - arrays['upstream_prediction_gradient'].sum(1).sum(0)
        details.append({'condition': name, 'Torch_reduction_difference_maxabs': float(delta.abs().max()),
                        'NumPy_reduction_difference_gap': float(np.abs(numpy_delta - delta.numpy()).max()),
                        'NumPy_bitwise_matches_Torch': bool(np.array_equal(numpy_delta, delta.numpy()))})
    audit.check('diagnostic:CUDA_not_initialized', torch.cuda.is_initialized() is False)
    return audit.close(result_artifact={'path': str(path.relative_to(ROOT)), 'sha256': file_hash(path)},
        coverage=coverage, independent_conditions=details, cpu_wall_seconds=time.perf_counter() - started,
        GPU_executed=False, fresh_EVAL_opened=False, protected_TEST_opened=False, historical_GPU_gradient_reproduced=False,
        performance_improvement_verified=False, limitations=['Direct CPU reductions and disposable single-bias AdamW steps validate this numerical diagnostic only.',
            'Synthetic latent Jacobian replay is not the actual E/F forward and does not reproduce historical GPU kernels.',
            'No inference about the cause of forecast failure or accuracy/latency benefit of bias removal follows.'])


def full_audit(paths, eval_authorized=False, diagnostic_path=None, deployment_path=None):
    # No path is opened or hashed before the explicit permission check.
    if not eval_authorized:
        raise PermissionError('FINAL_EVAL_AUDIT_REQUIRES_ROOT_POST_FIRST_PASS_AUTHORIZATION')
    started = time.perf_counter()
    import torch
    audit = Audit()
    results = {}
    phase_artifacts = {}
    for phase in ('dev', 'fresh', 'probe', 'evaluation', 'cost'):
        path = checked_path(paths[phase])
        result = json.loads(path.read_text(encoding='utf-8-sig'))
        if result.get('status') not in ('COMPLETE', 'FAILED_PRESERVED', 'BUDGET_HELD_PRESERVED'):
            raise ValueError('NONTERMINAL_PHASE:' + phase)
        results[phase] = result
        phase_artifacts[phase] = {'path': str(path.relative_to(ROOT)), 'sha256': file_hash(path)}
        audit.check('phase:' + phase + ':complete', result['status'] == 'COMPLETE' and not result.get('failed_paths'))
        resource_path = audit.asset('phase:' + phase + ':resource', result['resource_receipt'])
        resource = json.loads(resource_path.read_text(encoding='utf-8-sig'))
        audit.check('phase:' + phase + ':terminal_resource_complete', resource['status'] == 'COMPLETE' and resource['gpu'] is True)
        audit.check('phase:' + phase + ':F_state_unchanged', result.get('backbone_before_sha256') == result.get('backbone_after_sha256') and bool(result.get('backbone_before_sha256')))
    dev, fresh, probe, evaluation, cost = [results[k] for k in ('dev', 'fresh', 'probe', 'evaluation', 'cost')]
    preflight = json.loads((HERE / 'PREFLIGHT.json').read_text(encoding='utf-8-sig'))
    for record in preflight['bolt_snapshot']['files']:
        audit.asset('backbone_original_file:' + record['path'], record)
    adjustment_path = HERE / 'BUDGET_PROFILE_ADJUSTMENT_v1.json'
    adjustment = json.loads(adjustment_path.read_text(encoding='utf-8-sig'))
    profile_path = checked_path(adjustment['profile_path'])
    audit.check('budget_adjustment:profile_SHA', file_hash(profile_path) == adjustment['profile_sha256'])
    profile_result = json.loads(profile_path.read_text(encoding='utf-8-sig'))
    audit.check('budget_adjustment:before_DEV_after_profile', profile_result['finished_unix'] <= adjustment['created_unix'] <= dev['started_unix'])
    audit.check('budget_adjustment:scope_unchanged', adjustment['same_scope'] == {'core_fits': 174, 'core_updates': 600, 'AE_fits': 12, 'AE_updates': 400, 'probes_unique_fits': 108, 'cost_rows': 536})
    historical_source_deltas = {}
    for tag in ('fixture_v1', 'profile_v1'):
        path = HERE / ('run_' + tag) / 'RESULT.json'
        old = json.loads(path.read_text(encoding='utf-8-sig'))
        differences = [k for k in old['source_hashes'] if old['source_hashes'][k] != dev['source_hashes'][k]]
        allowed = {'research/bihash_future_prediction_20261013/PLAN_KO.md', 'research/bihash_future_prediction_20261013/runtime.py'}
        audit.check('historical_source:' + tag + ':only_declared_budget_deltas', set(differences).issubset(allowed), differences)
        snapshot = ROOT / '.cache' / HERE.name / 'source_snapshots' / tag
        for source, digest in old['source_hashes'].items():
            audit.check('historical_source:' + tag + ':' + source + ':original_snapshot', file_hash(snapshot / source) == digest)
        historical_source_deltas[tag] = differences
    method_path = audit.asset('METHOD_LOCK', dev['method_lock'])
    method_lock = json.loads(method_path.read_text(encoding='utf-8-sig'))
    method_sha = file_hash(method_path)
    audit.check('METHOD_LOCK:declared_locked', method_lock.get('locked') is True and method_lock.get('status') == 'LOCKED')
    for phase, result in results.items():
        snapshot = ROOT / '.cache' / HERE.name / 'source_snapshots' / result['tag']
        manifest = json.loads((snapshot / 'MANIFEST.json').read_text(encoding='utf-8-sig'))
        audit.check('source:' + phase + ':snapshot_map', manifest['sha256'] == result['source_hashes'])
        audit.check('source:' + phase + ':METHOD_LOCK_map', result['source_hashes'] == method_lock['source_hashes'])
        for path, digest in result['source_hashes'].items():
            audit.check('source:' + phase + ':' + path + ':snapshot_bytes', file_hash(snapshot / path) == digest)
            audit.check('source:' + phase + ':' + path + ':current_bytes', file_hash(path) == digest)
        if 'method_lock' in result:
            audit.check('METHOD_LOCK:' + phase + ':SHA', result['method_lock']['sha256'] == method_sha)
    audit.check('F_state_common_across_phases', len({r['backbone_before_sha256'] for r in results.values()}) == 1)
    expected_dev = {(ds, seed, method, weight) for ds in LEGACY_DATASETS for seed in PRIMARY_SEEDS for method in PRIMARY_METHODS
                    for weight in ((.03, .1) if method.startswith('A_') else (0.,))}
    expected_fresh = {(ds, seed, method, method_lock['weights'][method]) for ds in FRESH_DATASETS for seed in PRIMARY_SEEDS for method in PRIMARY_METHODS}
    fields = ('dataset', 'seed', 'method', 'weight')
    coverages = {'dev': exact_coverage(dev.get('cases', []), expected_dev, fields),
                 'fresh': exact_coverage(fresh.get('cases', []), expected_fresh, fields)}
    for name, coverage in coverages.items():
        audit.check('coverage:' + name + ':exact', coverage['passed'], coverage)
    references = {ds: legacy_reference(audit, ds, dev['data'][ds]) for ds in LEGACY_DATASETS}
    for ds in FRESH_DATASETS:
        contract = fresh['contracts'][ds]
        audit.check(ds + ':method_lock_before_ingestion', contract['method_lock']['sha256'] == method_sha and method_lock['created_unix'] < contract['created_unix'])
        for record in contract['source_records']:
            source_path = checked_path(record['path'])
            receipts = []
            for receipt_path in source_path.parent.glob('receipt_*.json'):
                candidate = json.loads(receipt_path.read_text(encoding='utf-8-sig'))
                if candidate.get('status') == 'DOWNLOADED' and candidate.get('artifact', {}).get('sha256') == record['sha256']:
                    receipts.append(candidate)
            audit.check(ds + ':raw_download_after_METHOD_LOCK', bool(receipts) and all(r['created_unix'] > method_lock['created_unix'] for r in receipts),
                        {'matching_terminal_download_receipts': len(receipts)})
        references[ds] = fresh_reference(audit, contract, eval_authorized=True)
    case_summaries = [audit_case(audit, c, references[c['dataset']], eval_authorized=True) for c in dev['cases'] + fresh['cases']]
    for ds in LEGACY_DATASETS + FRESH_DATASETS:
        for seed in PRIMARY_SEEDS:
            current = [c for c in dev['cases'] + fresh['cases'] if c['dataset'] == ds and c['seed'] == seed]
            for field in ('initial_hash',):
                audit.check(ds + ':seed' + str(seed) + ':common_' + field, len({c[field] for c in current}) == 1)
            audit.check(ds + ':seed' + str(seed) + ':common_actual_batch', len({c['training']['order_sha256'] for c in current}) == 1)
            audit.check(ds + ':seed' + str(seed) + ':common_AE_clone_SHA', len({c['future_pretrain']['sha256'] for c in current}) == 1)
    recomputed = method_selection(dev['cases'], {(d, s) for d in LEGACY_DATASETS for s in PRIMARY_SEEDS}, PRIMARY_METHODS)
    audit.check('METHOD_LOCK:weights_TUNE_recomputed', recomputed['weights'] == method_lock['weights'])
    audit.check('METHOD_LOCK:selected_B_TUNE_recomputed', recomputed['selected_B_method'] == method_lock['selected_B_method'])
    pretrains = dev['future_pretrains'] + fresh['future_pretrains']
    ae_coverage = exact_coverage(pretrains, {(d, s) for d in LEGACY_DATASETS + FRESH_DATASETS for s in PRIMARY_SEEDS}, ('dataset', 'seed'))
    audit.check('coverage:AE12:exact', ae_coverage['passed'], ae_coverage)
    for row in pretrains:
        name = row['dataset'] + ':AE:' + str(row['seed'])
        audit.check(name + ':actual400curve', row['updates'] == 400 and [r['update'] for r in row['curve']] == list(range(1, 401)))
        audit.check(name + ':fixed_final_selection', row['selection'] == 'fixed final 400, TUNE diagnostic only')
        order = np.random.default_rng(row['seed'] + 10000).integers(0, len(references[row['dataset']]['origins']['fit']), size=(400, 64))
        audit.check(name + ':FIT_order_SHA', array_hash(order) == row['order_sha256'])
        path = audit.asset(name + ':checkpoint', row['checkpoint'])
        saved = torch.load(path, map_location='cpu', weights_only=True)
        audit.check(name + ':saved400identity', saved['update'] == 400 and saved['seed'] == row['seed'])
        audit.check(name + ':final_state_hash', tensor_state_hash(saved['future_autoencoder']) == row['final_hash'])
        steps = [int(v['step']) for v in saved['optimizer']['state'].values()]
        audit.check(name + ':optimizer_actual400steps', bool(steps) and all(s == 400 for s in steps))
        for case in (c for c in dev['cases'] + fresh['cases'] if c['dataset'] == row['dataset'] and c['seed'] == row['seed']):
            cp0 = torch.load(checked_path(case['training']['checkpoints'][0]['path']), map_location='cpu', weights_only=True)['model']
            clone = {k.removeprefix('future_autoencoder.'): v for k, v in cp0.items() if k.startswith('future_autoencoder.')}
            audit.check(name + ':clone_to:' + case['label'], tensor_state_hash(clone) == row['final_hash'])
    selected = [c for c in dev['cases'] if c['weight'] == method_lock['weights'][c['method']]] + fresh['cases']
    selected_ids = {(d, s, m, method_lock['weights'][m]) for d in LEGACY_DATASETS + FRESH_DATASETS for s in PRIMARY_SEEDS for m in PRIMARY_METHODS}
    selected_coverage = exact_coverage(selected, selected_ids, fields)
    audit.check('coverage:selected144:exact', selected_coverage['passed'], selected_coverage)
    readouts = probe['readouts']; controls = probe['controls']
    probe_expected = {(d, s, m) for d in LEGACY_DATASETS + FRESH_DATASETS for s in PRIMARY_SEEDS for m in PROBE_METHODS}
    probe_coverage = exact_coverage(readouts, probe_expected, ('dataset', 'seed', 'source_method'))
    audit.check('coverage:probe72:exact', probe_coverage['passed'], probe_coverage)
    audit.check('coverage:control4datasets', exact_coverage(controls, {(d,) for d in LEGACY_DATASETS + FRESH_DATASETS}, ('dataset',))['passed'])
    for container in controls + readouts:
        methods = container.get('methods', container.get('readouts'))
        expected_methods = {'RAW_BOLT', 'RAW_JOINT_RIDGE', 'RAW_BOLT_RAW_R', 'RAW_SELF33'} if 'methods' in container else {'E_RAW_RIDGE', 'E_ENRICHED_RIDGE'}
        audit.check('coverage:' + container['dataset'] + ':' + container.get('source_label', 'controls') + ':entry_methods',
                    {r['method'] for r in methods} == expected_methods and len(methods) == len(expected_methods))
        for row in methods:
            name = container['dataset'] + ':' + container.get('source_label', 'RAW_CONTROLS') + ':' + row['method']
            audit_prediction(audit, name + ':FIT_TUNE', row['prediction'], row['scores'], references[container['dataset']])
            if 'model' in row:
                audit.asset(name + ':ridge_weights', row['model'])
            for selection in ([row['selection']] if 'selection' in row else row.get('channel_selections', [])):
                choice = min(selection['candidates'], key=lambda c: (c['tune_norm_mse'], c['alpha']))['alpha']
                audit.check(name + ':ridge_alpha_TUNE_argmin', choice == selection['alpha'])
            if row.get('duplicate_of'):
                original = next(r for r in methods if r['method'] == row['duplicate_of'])
                audit.check(name + ':alias_no_additional_fit', row['additional_fit'] is False and row['model']['sha256'] == original['model']['sha256'] and row['prediction']['sha256'] == original['prediction']['sha256'])
    unique_probe = [r for c in readouts for r in c['readouts'] if r.get('additional_fit')]
    audit.check('coverage:probe108unique_fits', len(unique_probe) == 108)
    global_path = audit.asset('GLOBAL_MODEL_LOCK', evaluation['global_model_lock'])
    global_lock = json.loads(global_path.read_text(encoding='utf-8-sig'))
    global_sha = file_hash(global_path)
    audit.check('GLOBAL:METHOD_LOCK_SHA', global_lock['method_lock_sha256'] == method_sha)
    audit.check('GLOBAL:source_map_matches_METHOD_LOCK', global_lock['source_hashes'] == method_lock['source_hashes'])
    for ds, declared in global_lock['newdata_contracts'].items():
        contract_path = audit.asset('GLOBAL:data_contract:' + ds, {'path': declared['contract_path'], 'sha256': declared['contract_sha256']})
        actual_contract = json.loads(contract_path.read_text(encoding='utf-8-sig'))
        audit.check('GLOBAL:data_contract:' + ds + ':matches_used_fresh_contract', actual_contract == fresh['contracts'][ds])
    for row in global_lock['selected_models']:
        path = audit.asset('GLOBAL:model:' + row['path'], row)
        audit.check('GLOBAL:model_written_before_lock:' + row['path'], path.stat().st_mtime < global_lock['created_unix'])
    for phase in ('dev', 'fresh', 'probe'):
        audit.check('GLOBAL:' + phase + ':finished_before_lock', results[phase]['finished_unix'] < global_lock['created_unix'])
    events = [{'dataset': e['dataset'], 'opened_unix': e['opened_unix'], 'lock_sha256': e['global_model_lock']['sha256']} for e in evaluation['evaluation_open_receipts']]
    gate_lock = {**global_lock, 'selected_models': global_lock['core_model_identities']}
    barrier = joint_barrier(gate_lock, events, selected_ids, global_sha)
    audit.check('GLOBAL:joint_fresh_barrier', barrier['passed'], barrier)
    for event in evaluation['evaluation_open_receipts']:
        audit.check('EVAL:first_open:' + event['dataset'] + ':interval', global_lock['created_unix'] < event['opened_unix'] <= event['completed_open_unix'] <= evaluation['finished_unix'])
        audit.check('EVAL:first_open:' + event['dataset'] + ':METHOD_SHA', event['method_lock_sha256'] == method_sha)
        declared = global_lock['newdata_contracts'][event['dataset']]
        for key, payload_name in (('fit_tune_payload_sha256', 'fit_tune'), ('eval_input_payload_sha256', 'eval_past_input'), ('eval_target_payload_sha256', 'eval_future_target')):
            audit.check('GLOBAL:data:' + event['dataset'] + ':' + key, declared[key] == fresh['contracts'][event['dataset']]['payloads'][payload_name]['sha256'])
        audit.check('EVAL:first_open:' + event['dataset'] + ':payload_SHA', event['payload_hashes'] == {
            'eval_past_input': declared['eval_input_payload_sha256'], 'eval_future_target': declared['eval_target_payload_sha256']})
    eval_expected = {(c['dataset'], c['seed'], c['method']) for c in selected}
    eval_coverage = exact_coverage(evaluation['core_evaluation'], eval_expected, ('dataset', 'seed', 'method'))
    audit.check('coverage:core_EVAL144', eval_coverage['passed'], eval_coverage)
    eval_npz_count = 0
    for item in evaluation['core_evaluation']:
        variants = {'regular', 'student', 'memory', 'uniform', 'tanh', 'reset_maps'} if item['method'] in {'B_GUMBEL', method_lock['selected_B_method']} else {'regular'}
        audit.check('EVAL:' + item['source_label'] + ':variant_coverage', {e['variant'] for e in item['entries']} == variants and len(item['entries']) == len(variants))
        audit.check('EVAL:' + item['source_label'] + ':no_reselection', item['no_retraining_or_new_selection'] is True)
        for entry in item['entries']:
            audit_prediction(audit, item['dataset'] + ':' + item['source_label'] + ':' + entry['variant'], entry['prediction'],
                             {item['split']: entry['scores']}, references[item['dataset']], eval_authorized=True)
            eval_npz_count += 1
    for group in ('control_evaluation', 'readout_evaluation'):
        for item in evaluation[group]:
            audit_prediction(audit, item['dataset'] + ':' + group + ':' + item.get('source_label', '') + ':' + item['method'], item['prediction'],
                             {item['split']: item['scores']}, references[item['dataset']], eval_authorized=True)
    control_expected = {(d, m) for d in LEGACY_DATASETS + FRESH_DATASETS for m in ('RAW_BOLT', 'RAW_JOINT_RIDGE', 'RAW_BOLT_RAW_R', 'RAW_SELF33')}
    audit.check('coverage:control_EVAL16', exact_coverage(evaluation['control_evaluation'], control_expected, ('dataset', 'method'))['passed'])
    readout_expected = {(c['dataset'], c['seed'], c['source_label'], m) for c in readouts for m in ('E_RAW_RIDGE', 'E_ENRICHED_RIDGE')}
    audit.check('coverage:readout_EVAL144', exact_coverage(evaluation['readout_evaluation'], readout_expected, ('dataset', 'seed', 'source_label', 'method'))['passed'])
    for item in evaluation['readout_evaluation']:
        if item.get('duplicate_of'):
            original = next(r for r in evaluation['readout_evaluation'] if r['dataset'] == item['dataset'] and r['source_label'] == item['source_label'] and r['method'] == item['duplicate_of'])
            a = load_prediction(item['prediction']['path'], (item['split'],), eval_authorized=True)
            b = load_prediction(original['prediction']['path'], (item['split'],), eval_authorized=True)
            audit.check('EVAL:alias:' + item['dataset'] + ':' + item['source_label'] + ':exact_predictions', np.array_equal(a[item['split']]['prediction'], b[item['split']]['prediction']))
    expected_cost = {(c['dataset'], c['label'], 'CORE', b) for c in selected for b in (1, 8)}
    expected_cost |= {(c['dataset'], r['method'], r['method'], b) for c in controls for r in c['methods'] for b in (1, 8)}
    expected_cost |= {(c['dataset'], c['source_label'] + '_' + r['method'], r['method'], b) for c in readouts for r in c['readouts'] if not r.get('duplicate_of') for b in (1, 8)}
    cost_coverage = exact_coverage(cost['records'], expected_cost, ('dataset', 'identity', 'route', 'batch'))
    audit.check('coverage:cost536:exact', cost_coverage['passed'] and len(expected_cost) == 536, cost_coverage)
    for row in cost['records']:
        name = row['dataset'] + ':' + row['identity'] + ':batch' + str(row['batch'])
        audit.check('cost:' + name + ':2warm7repeat', row['warmup'] == 2 and row['repeats'] == 7 and len(row['seconds']) == 7)
        audit.check('cost:' + name + ':positive_finite_seconds', np.isfinite(row['seconds']).all() and all(s > 0 for s in row['seconds']))
        audit.numeric('cost:' + name + ':median', np.median(row['seconds']), row['median_seconds'])
        audit.numeric('cost:' + name + ':per_window', np.median(row['seconds']) / row['batch'], row['median_seconds_per_window'])
        audit.check('cost:' + name + ':producer_percell_parity', row['parity_pass'] is True and row['physical_parity_atol'] == 2e-3 and row['physical_parity_rtol'] == 2e-5)
    parameter_checks = [c for c in audit.checks if any(s in c['name'] for s in ('updates_expected', 'clone_to', 'actual400steps', 'actual_update_sequence', '600_step', 'common_actual_batch'))]
    numeric_checks = [c for c in audit.checks if any(s in c['name'] for s in (':mse', ':mae', ':per_channel_', ':observed_counts', '_raw_contract', 'TUNE_argmin'))]
    coverage_checks = [c for c in audit.checks if c['name'].startswith('coverage:')]
    gates = {
        'joint_barrier': {'passed': barrier['passed'], 'detail': barrier},
        'parameter_updates': {'passed': all(c['passed'] for c in parameter_checks), 'checks': len(parameter_checks)},
        'numeric_rescore': {'passed': all(c['passed'] for c in numeric_checks), 'status': 'PASS' if all(c['passed'] for c in numeric_checks) else 'FAIL', 'checks': len(numeric_checks), 'cp_TUNE_predictions': sum(c['checkpoint_predictions_verified'] for c in case_summaries), 'checkpoint_forward_replayed': False},
        'scope_coverage': {'passed': all(c['passed'] for c in coverage_checks), 'status': 'PASS' if all(c['passed'] for c in coverage_checks) else 'FAIL', 'core_fits': len(case_summaries), 'AE_fits': len(pretrains), 'selected_core': len(selected), 'unique_probe_fits': len(unique_probe), 'cost_paths': len(expected_cost) // 2, 'cost_rows': len(cost['records'])},
        'causality': {'passed': False, 'status': 'PENDING_CPU_FIXTURE_LINK', 'limitation': 'Fresh target separation and joint barrier checked; CPU model fixture must be linked below.'},
    }
    fixture_path = HERE / 'validation_v1/CPU_FIXTURE_v1.json'
    fixture = json.loads(fixture_path.read_text(encoding='utf-8-sig'))
    model_sha = method_lock['source_hashes']['research/bihash_future_prediction_20261013/model.py']
    gate_ok = fixture['failed_count'] == 0 and fixture['model_source_sha256'] == model_sha
    gates['causality'] = {'passed': gate_ok, 'status': 'CPU_MODEL_FIXTURE_PLUS_FRESH_BARRIER_CHECKED',
                         'evidence': {'path': str(fixture_path.relative_to(ROOT)), 'sha256': file_hash(fixture_path)},
                         'limitation': 'No claim of prospective online retraining, or unseen-data generalization from passing safety checks.'}
    if diagnostic_path is not None:
        path = checked_path(diagnostic_path)
        diagnostics = json.loads(path.read_text(encoding='utf-8-sig'))
        audit.check('diagnostic:terminal', diagnostics.get('status') in ('COMPLETE', 'FAILED_PRESERVED'))
    deployment = None
    if deployment_path is not None:
        deployment = audit_deployment_outputs(audit, deployment_path, expected_cost, references, eval_authorized=True,
                                              phase_artifacts=phase_artifacts, cost=cost, method_lock=method_lock)
        gates['deployment_parity'] = {'passed': deployment['passed'], 'status': 'POSTHOC_SAVED_OUTPUTS_CHECKED', 'scope': deployment['scope']}
    terminal_resources = []
    for resource_path in (HERE / 'resources').glob('*.json'):
        resource = json.loads(resource_path.read_text(encoding='utf-8-sig'))
        if resource.get('status') not in ('COMPLETE', 'FAILED_PRESERVED', 'BUDGET_HELD_PRESERVED'):
            raise ValueError('ACTIVE_RESOURCE_JSON_NOT_ALLOWED')
        terminal_resources.append(resource)
    measured_gpu = sum(r['wall_seconds'] for r in terminal_resources if r['gpu'])
    measured_cpu = sum(r['wall_seconds'] for r in terminal_resources if not r['gpu'])
    audit.check('budget:GPU_terminal_wall_plus_pilot_and_reserve', measured_gpu + 475.5248562004417 + 60. <= adjustment['current_agent_selected_GPU_ceiling_seconds'])
    audit.check('budget:CPU_terminal_wall', measured_cpu + time.perf_counter() - started <= 7200.)
    audit.check('budget:RSS_VRAM_terminal8GiB', all(r['sampled_peak_rss'] <= 8 * 1024**3 and r['peak_allocated_vram'] <= 8 * 1024**3 for r in terminal_resources))
    current_storage = sum(p.stat().st_size for base in (HERE, ROOT / '.cache' / HERE.name,
        ROOT / 'research/bihash_future_alignment_20261013', ROOT / '.cache/bihash_future_alignment_20261013') if base.exists() for p in base.rglob('*') if p.is_file())
    audit.check('budget:study_storage12GiB', current_storage <= 12 * 1024**3)
    audit.check('budget:campaign8h', time.time() - preflight['goal_started_unix'] <= 28800.)
    for name, gate in gates.items():
        audit.check('gate:' + name, gate['passed'])
    return audit.close(gates=gates, phase_artifacts=phase_artifacts, case_summaries=case_summaries,
        whole_experiment_complete=all(c['passed'] for c in audit.checks) and all(r['status'] == 'COMPLETE' for r in results.values()),
        historical_source_deltas=historical_source_deltas,
        resource_summary={'primary_terminal_GPU_process_wall_seconds': measured_gpu, 'prior_pilot_measured_GPU_seconds': 475.5248562004417,
            'prior_pilot_unmetered_fixture_reserve_not_measurement': 60., 'budget_counted_GPU_seconds': measured_gpu + 535.5248562004417,
            'terminal_CPU_process_wall_seconds': measured_cpu, 'storage_bytes': current_storage,
            'CPU_limit_scope_note': 'This receipt includes its own audit body plus runtime terminal CPU processes; historical audit/test subprocess totals are separately reported by root.'},
        cpu_wall_seconds=time.perf_counter() - started, deployment_output_audit=deployment,
        actual_counts={'core': len(case_summaries), 'AE': len(pretrains), 'CP_TUNE_NPZ': sum(c['checkpoint_predictions_verified'] for c in case_summaries),
                       'selected_core': len(selected), 'core_EVAL_NPZ': eval_npz_count, 'cost_rows': len(cost['records'])},
        generalization_claim_verified=False, performance_improvement_claim_verified=False,
        limitations=['Science prediction artifacts are fully rescored, including their masks and source coordinates; CP Bolt forward is not replayed on CPU.',
                     'Timed seven-repeat output arrays are not stored. Per-cell timed parity comes from the sealed producer and receipt; optional posthoc deployment NPZ is a separate replay.',
                     'Legacy physical values are reconstructed from historical normalized FP32 NPZ; original legacy raw bytes are not checked here.',
                     'Integrity, causality and coverage PASS do not establish improvement, statistical significance or independent generalization.'])


def cpu_model_fixture():
    """Real model logic exercised with different synthetic coordinates on CPU."""
    started = time.perf_counter()
    import torch
    from torch import nn
    torch.set_num_threads(2)
    source = HERE / 'model.py'
    before_hash = file_hash(source)
    spec = importlib.util.spec_from_file_location('independent_primary_model', source)
    bm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bm)
    audit = Audit()
    rng = np.random.default_rng(731)
    basis = np.linalg.qr(rng.normal(size=(3, 2)))[0].astype(np.float32)
    bank = rng.normal(size=(5, 4, 3)).astype(np.float32)
    mask = np.ones_like(bank, bool); mask[1, 0, 2] = False
    bank_origins = np.array([0, 24, 64, 100, 200], np.int64)
    x = torch.tensor(rng.normal(size=(3, 32, 3)).astype(np.float32))
    target = torch.tensor(rng.normal(size=(3, 4, 3)).astype(np.float32))
    observed = torch.ones_like(target, dtype=torch.bool); observed[0, :2, 1] = False
    origins = torch.tensor([45, 80, 120])

    class FrozenForecast(nn.Module):
        def __init__(self):
            super().__init__()
            self.matrix = nn.Parameter(torch.tensor([[.13, .02], [-.03, .09]]), requires_grad=False)
        def forward(self, z):
            return z[:, -1:] + torch.einsum('blk,km->blm', z.mean(1, keepdim=True), self.matrix).expand(-1, 4, -1)

    def build(method):
        return bm.FutureBiHash(np.zeros(3, np.float32), basis, 92317, method,
                               bank, mask, bank_origins, horizon=4, representation=5, bits=7)

    def arr(value):
        return value.detach().cpu().numpy()

    def grad_norm(loss, parameters, retain_graph=True):
        gradients = torch.autograd.grad(loss, list(parameters), allow_unused=True, retain_graph=retain_graph)
        return float(sum(g.detach().square().sum() for g in gradients if g is not None).sqrt()) if any(g is not None for g in gradients) else 0.

    frozen = FrozenForecast()
    frozen_before = {k: v.detach().clone() for k, v in frozen.state_dict().items()}
    signature = set(inspect.signature(bm.FutureBiHash.forward).parameters)
    audit.check('forward_signature_has_no_current_future_or_target', not (signature & {'target', 'future_target', 'y', 'observed'}), sorted(signature))
    reference = build('NO_ALIGN')
    for method in bm.METHODS:
        other = build(method)
        audit.check('same_initial_state:' + method, all(torch.equal(v, other.state_dict()[k]) for k, v in reference.state_dict().items()))

    # Independent manual two-class Gumbel algebra with identical RNG, including
    # hard-forward/soft-backward ST rather than a categorical choice across bits.
    logits = torch.tensor([[0., .7, -1.2, .3]], requires_grad=True)
    generator = torch.Generator().manual_seed(91)
    random = torch.rand((1, 4, 2), generator=generator)
    random = random.clamp(torch.finfo(random.dtype).eps, 1 - torch.finfo(random.dtype).eps)
    gumbel = -torch.log(-torch.log(random))
    log_odds = (2 * logits.detach() + gumbel[..., 0] - gumbel[..., 1]) / .65
    expected = 2 * torch.sigmoid(log_odds) - 1
    soft = bm.binary_code(logits, True, .65, 'gumbel', torch.Generator().manual_seed(91))
    hard = bm.binary_code(logits, True, .65, 'st', torch.Generator().manual_seed(91))
    audit.check('gumbel_independent_binary_formula', torch.allclose(soft, expected, atol=2e-7, rtol=2e-6))
    audit.check('st_forward_equals_per_bit_sign', torch.equal(hard.detach(), torch.where(log_odds >= 0, 1., -1.)))
    expected_gradient = (1 - expected.square()) / .65
    actual_gradient = torch.autograd.grad(hard.sum(), logits)[0]
    audit.check('st_gradient_equals_continuous_binary_derivative', torch.allclose(actual_gradient, expected_gradient, atol=5e-7, rtol=2e-6))
    audit.check('eval_zero_tie_positive', torch.equal(bm.binary_code(logits, False, .25, 'gumbel'), torch.tensor([[1., 1., -1., 1.]])))

    model = build('B_GUMBEL').eval()
    # Activate temporal output, so zero-initialized residuals cannot hide leakage.
    with torch.no_grad():
        model.codec.codec.output.weight.fill_(.021)
    augmented, details = model.encode_with_memory(x, origins)
    independent_eligible = memory_eligibility(arr(origins), bank_origins, length=32, horizon=4)
    audit.check('rowwise_eligibility_matches_absolute_coordinate_loop', np.array_equal(arr(details['eligible']), independent_eligible))
    expected_attention = numpy_attention(arr(details['query_code']), arr(details['memory_code']), independent_eligible)
    audit.check('retrieval_matches_independent_numpy_cosine_softmax', np.allclose(arr(details['attention']), expected_attention, atol=2e-7, rtol=2e-6))
    changed_x = x.clone(); changed_x[:, 17:] += 600.
    altered, _ = model.encode_with_memory(changed_x, origins)
    audit.check('activated_encoder_prefix_ignores_later_input', torch.equal(augmented[:, :17], altered[:, :17]))
    model.bank_delta.requires_grad_(True)
    augmented, details = model.encode_with_memory(x, origins)
    derivative = torch.autograd.grad(augmented[:, :9].square().sum(), model.bank_delta)[0]
    never_eligible = ~independent_eligible[:, :9].any((0, 1))
    audit.check('ineligible_memory_exact_zero_derivative', torch.equal(derivative[never_eligible], torch.zeros_like(derivative[never_eligible])),
                {'memory_indices': np.flatnonzero(never_eligible), 'max_abs': float(derivative[never_eligible].abs().max())})
    original = model(x, origins, frozen)
    with torch.no_grad():
        model.bank_delta[-1].add_(10000.)
    changed_prediction = model(x, origins, frozen)
    audit.check('whole_forecast_ignores_always_future_memory', torch.equal(original, changed_prediction))
    allmasked, info = model(x, torch.tensor([-100, -90, -80]), frozen, return_details=True)
    fallback = model.codec.forward_path(x, frozen)
    audit.check('allmasked_memory_fallback_exact', torch.equal(allmasked, fallback))
    audit.check('allmasked_attention_finite_exact_zero', torch.equal(info['attention'], torch.zeros_like(info['attention'])) and bool(torch.isfinite(info['attention']).all()))
    whole = model(x, origins, frozen)
    single = model(x[1:2], origins[1:2], frozen)
    audit.check('query_prediction_batch_independent', torch.allclose(whole[1:2], single, atol=5e-7, rtol=3e-6))

    diagnostics = {}
    for method in ('B_GUMBEL', 'B_COSINE', 'A_BIHASH_GUMBEL', 'A_BIHASH_DETACH_E'):
        temporary = build(method).train()
        groups = {
            'E': [p for n, p in temporary.named_parameters() if n.startswith('codec.') and '.decode.' not in n and '.residual.' not in n],
            'D': [p for n, p in temporary.named_parameters() if n.startswith('codec.') and ('.decode.' in n or '.residual.' in n)],
            'Wp': list(temporary.student_hash.parameters()), 'Wf': list(temporary.future_hash.parameters()),
            'phi': list(temporary.future_autoencoder.parameters()), 'Sprojection': list(temporary.student_projection.parameters()),
        }
        initial = {name: [p.detach().clone() for p in pars] for name, pars in groups.items()}
        optimizer = torch.optim.AdamW(temporary.parameters(), lr=.001, weight_decay=0.)
        record = {'steps': [], 'gradient_paths': {}}
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            losses = temporary.losses(x, origins, target, observed, frozen, update=step)
            if step == 0:
                for name, pars in groups.items():
                    record['gradient_paths'][name] = {kind: grad_norm(losses[kind], pars) for kind in ('main', 'aux', 'rec')}
                independent_loss = macro_score(arr(losses['prediction']), arr(target), arr(observed))['mse']
                audit.check('macro_main_loss_independent:' + method, np.isclose(float(losses['main'].detach()), independent_loss, atol=2e-6, rtol=2e-6))
            total = losses['main'] + .1 * losses['aux'] + .1 * losses['rec']
            total.backward()
            audit.check('finite_actual_gradients:' + method + ':' + str(step), all(bool(torch.isfinite(p.grad).all()) for p in temporary.parameters() if p.grad is not None))
            codec = list(temporary.codec.parameters())
            side = [p for n, p in temporary.named_parameters() if not n.startswith('codec.')]
            torch.nn.utils.clip_grad_norm_(codec, 1.)
            torch.nn.utils.clip_grad_norm_(side, 1.)
            optimizer.step()
            record['steps'].append({'step': step + 1, 'main': float(losses['main'].detach()), 'aux': float(losses['aux'].detach()), 'rec': float(losses['rec'].detach())})
        record['changed_scalar_counts'] = {name: sum(int((before != after.detach()).sum()) for before, after in zip(initial[name], pars)) for name, pars in groups.items()}
        diagnostics[method] = record
        for name in groups:
            audit.check('actual_temporary_optimizer_updates:' + method + ':' + name, record['changed_scalar_counts'][name] > 0, record['changed_scalar_counts'][name])
        if method == 'A_BIHASH_DETACH_E':
            audit.check('detach_aux_encoder_gradient_exact_zero', record['gradient_paths']['E']['aux'] == 0.)
        elif method == 'A_BIHASH_GUMBEL':
            audit.check('aligned_aux_encoder_gradient_positive', record['gradient_paths']['E']['aux'] > 0.)
        if method.startswith('B_'):
            for name in ('Wp', 'Wf', 'E', 'phi'):
                audit.check('B_forecast_loss_trains_branch:' + method + ':' + name, record['gradient_paths'][name]['main'] > 0., record['gradient_paths'][name]['main'])

    # DETACH control must preserve actual codec updates against the no-align
    # model, not just have a zero auxiliary norm before joint gradient clipping.
    base, detach = build('NO_ALIGN').train(), build('A_BIHASH_DETACH_E').train()
    for temporary in (base, detach):
        optimizer = torch.optim.AdamW(temporary.parameters(), lr=.001, weight_decay=0.)
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            losses = temporary.losses(x, origins, target, observed, frozen, update=step)
            (losses['main'] + .1 * losses['aux'] + .1 * losses['rec']).backward()
            torch.nn.utils.clip_grad_norm_(list(temporary.codec.parameters()), 1.)
            torch.nn.utils.clip_grad_norm_([p for n, p in temporary.named_parameters() if not n.startswith('codec.')], 1.)
            optimizer.step()
    audit.check('detach_three_actual_codec_updates_equal_noalign', all(torch.equal(v, detach.codec.state_dict()[k]) for k, v in base.codec.state_dict().items()))
    for method in ('B_GUMBEL', 'A_BIHASH_GUMBEL'):
        temporary = build(method).eval()
        a = temporary.losses(x, origins, target, observed, frozen)
        b = temporary.losses(x, origins, target + 600., observed, frozen)
        audit.check('changing_current_future_cannot_change_prediction:' + method, torch.equal(a['prediction'], b['prediction']))
    audit.check('tiny_forecaster_parameters_remain_frozen', all(not p.requires_grad and p.grad is None for p in frozen.parameters()))
    audit.check('tiny_forecaster_state_unchanged', all(torch.equal(v, frozen.state_dict()[k]) for k, v in frozen_before.items()))
    audit.check('model_source_unchanged_during_cpu_fixture', before_hash == file_hash(source))
    sources = {str(source.relative_to(ROOT)): before_hash,
               str(Path(bm.en.__file__).relative_to(ROOT)): file_hash(bm.en.__file__),
               str(Path(bm.en.ex.__file__).relative_to(ROOT)): file_hash(bm.en.ex.__file__)}
    return audit.close(device='cpu', model_source_sha256=before_hash, sources=sources,
                       diagnostics=diagnostics, cpu_wall_seconds=time.perf_counter() - started,
                       generalization_verified=False, performance_improvement_verified=False,
                       limitations=['Synthetic CPU logic checks do not verify real Bolt gradients or forecasting performance.',
                                    'Actual raw data, stored CP predictions, TUNE selection, source locks and EVAL barrier await terminal artifacts.',
                                    'Train-soft/eval-hard hash behavior is intentional and remains a performance diagnostic.'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu-fixture', action='store_true')
    parser.add_argument('--legacy-loader-fixture', action='store_true')
    parser.add_argument('--fixture-result', type=Path)
    parser.add_argument('--dev-only-result', type=Path)
    parser.add_argument('--bias-diagnostic-result', type=Path)
    parser.add_argument('--case-manifests', nargs='+', type=Path)
    parser.add_argument('--source-tag')
    parser.add_argument('--dev-result', type=Path)
    parser.add_argument('--fresh-result', type=Path)
    parser.add_argument('--probe-result', type=Path)
    parser.add_argument('--evaluation-result', type=Path)
    parser.add_argument('--cost-result', type=Path)
    parser.add_argument('--diagnostic-result', type=Path)
    parser.add_argument('--deployment-output-result', type=Path)
    parser.add_argument('--eval-authorized', action='store_true')
    parser.add_argument('--tag')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    full_paths = {'dev': args.dev_result, 'fresh': args.fresh_result, 'probe': args.probe_result,
                  'evaluation': args.evaluation_result, 'cost': args.cost_result}
    full_mode = any(full_paths.values())
    if sum(bool(v) for v in (args.cpu_fixture, args.legacy_loader_fixture, args.fixture_result, args.case_manifests, args.dev_only_result, args.bias_diagnostic_result, full_mode)) != 1:
        parser.error('Choose exactly one explicit fixture, completed-case, or full terminal audit mode.')
    if full_mode and (not all(full_paths.values()) or not args.tag):
        parser.error('Full audit requires all five result paths and --tag.')
    if args.tag and (not args.tag.replace('_', '').replace('-', '').isalnum()):
        parser.error('--tag must use only letters, digits, underscores or hyphens.')
    if not args.output and not full_mode:
        parser.error('Fixture/completed-case mode requires --output.')
    output = checked_path(args.output or HERE / 'validation_v1' / args.tag / 'AUDIT.json')
    if not output.is_relative_to(HERE / 'validation_v1'):
        raise ValueError('AUDIT_OUTPUT_OUTSIDE_OWNED_VALIDATION_FOLDER')
    audit_started_unix = time.time()
    if full_mode:
        result = full_audit(full_paths, eval_authorized=args.eval_authorized, diagnostic_path=args.diagnostic_result,
                            deployment_path=args.deployment_output_result)
    elif args.cpu_fixture:
        result = cpu_model_fixture()
    elif args.legacy_loader_fixture:
        result = legacy_loader_fixture()
    elif args.fixture_result:
        result = audit_real_fixture(args.fixture_result)
    elif args.dev_only_result:
        result = audit_dev_result(args.dev_only_result)
    elif args.bias_diagnostic_result:
        result = audit_bias_diagnostic(args.bias_diagnostic_result)
    else:
        if not args.source_tag:
            parser.error('--case-manifests requires --source-tag')
        result = audit_case_manifests(args.case_manifests, args.source_tag)
    result['audit_started_unix'] = audit_started_unix
    result['audit_finished_unix'] = time.time()
    result['cpu_wall_scope'] = 'Measured audit function body; output serialization/write and pre-body NumPy startup are not included.'
    source_bytes = Path(__file__).read_bytes()
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    source_copy = output.with_name(output.stem + '_validator_' + source_digest[:12] + '.py')
    result['validator_source'] = {'path': str(Path(__file__).relative_to(ROOT)), 'sha256': source_digest,
                                'immutable_snapshot': str(source_copy.relative_to(ROOT))}
    # Validate the entire receipt before producing either final artifact.
    json.dumps(jsonable(result), ensure_ascii=False, allow_nan=False)
    source_copy.parent.mkdir(parents=True, exist_ok=True)
    with source_copy.open('xb') as handle:
        handle.write(source_bytes)
    write_fresh(output, result)
    print(json.dumps({'status': result['status'], 'failed_count': result['failed_count'], 'check_count': result['check_count'], 'cpu_wall_seconds': result['cpu_wall_seconds'], 'output': str(output)}, ensure_ascii=False))
    return 0 if result['failed_count'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
