"""D1 fixed FIT-RAW projection math and a caller-supplied forecasting path.

Importing this module never imports the old campaign, loads a model, or opens data.
Scientific execution requires caller-verified old-condition inputs and a shared
Bi resource Limit. The CLI only describes this interface.
"""
from dataclasses import dataclass
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / '.cache' / HERE.name
SPLIT_ROLES = {'fit': 'training', 'tune': 'model_selection',
               'val': 'reused_development', 'eval': 'exposed_evaluation'}


@dataclass(frozen=True)
class Basis:
    name: str
    mode: str
    rank_label: str
    matrix: np.ndarray
    mean: np.ndarray
    eigenvalues: np.ndarray

    @property
    def rank(self):
        return self.matrix.shape[1]


def fit_bases(fit_raw_forecast, fit_last):
    """Only FIT RAW forecasts and their exact preprocessed last values enter fit."""
    forecast = np.asarray(fit_raw_forecast, dtype=np.float64)
    last = np.asarray(fit_last, dtype=np.float64)
    if forecast.ndim != 3 or last.shape != (forecast.shape[0], forecast.shape[-1]):
        raise ValueError('FIT_RAW_LAST_SHAPE')
    if forecast.size == 0 or not np.isfinite(forecast).all() or not np.isfinite(last).all():
        raise ValueError('FIT_RAW_LAST_NONFINITE_OR_EMPTY')
    channels = forecast.shape[-1]
    if channels < 2:
        raise ValueError('D1_REQUIRES_AT_LEAST_TWO_CHANNELS')
    delta = (forecast - last[:, None, :]).reshape(-1, channels)
    mean = delta.mean(axis=0)
    ranks = [('quarter', max(1, channels // 4)), ('half', channels // 2),
             ('full_C_rotated', channels)]
    result = [Basis('identity_C', 'identity', 'identity_C', np.eye(channels),
                    np.zeros(channels), np.zeros(channels))]
    for mode in ('through_origin', 'centered'):
        values = delta if mode == 'through_origin' else delta - mean
        eigenvalues, vectors = np.linalg.eigh(values.T @ values / len(values))
        order = np.argsort(-eigenvalues, kind='stable')
        eigenvalues, vectors = np.maximum(eigenvalues[order], 0.), vectors[:, order]
        for index in range(channels):
            pivot = np.argmax(np.abs(vectors[:, index]))
            if vectors[pivot, index] < 0:
                vectors[:, index] *= -1
        for label, rank in ranks:
            result.append(Basis(mode + '_' + label, mode, label, vectors[:, :rank].copy(),
                                mean.copy(), eigenvalues.copy()))
    return result


def mean_perpendicular(basis, dtype=np.float64):
    matrix = basis.matrix.astype(dtype)
    mean = basis.mean.astype(dtype)
    return mean - (mean @ matrix) @ matrix.T if basis.mode == 'centered' else np.zeros_like(mean)


def project_raw_forecast(raw_forecast, raw_last, basis, dtype=np.float64):
    last = np.asarray(raw_last, dtype=dtype)[:, None, :]
    delta = np.asarray(raw_forecast, dtype=dtype) - last
    matrix = basis.matrix.astype(dtype)
    if basis.mode == 'centered':
        mean = basis.mean.astype(dtype)
        return last + mean + ((delta - mean) @ matrix) @ matrix.T
    return last + (delta @ matrix) @ matrix.T


def decode_forecast(latent_future, raw_last, basis, dtype=np.float64):
    last = np.asarray(raw_last, dtype=dtype)[:, None, :]
    matrix = basis.matrix.astype(dtype)
    future = np.asarray(latent_future, dtype=dtype)
    return last + (future - last @ matrix) @ matrix.T + mean_perpendicular(basis, dtype)


def probe_split(history, raw_forecast, basis, predictor, *, batch_size=8, check=None):
    """Run E(X)=XB, caller's F, then the fixed linear anchor decode."""
    history = np.asarray(history, dtype=np.float32)
    raw_forecast = np.asarray(raw_forecast, dtype=np.float32)
    if history.ndim != 3 or raw_forecast.ndim != 3 or history.shape[0] != raw_forecast.shape[0]:
        raise ValueError('PROBE_HISTORY_RAW_SHAPE')
    if history.shape[-1] != raw_forecast.shape[-1] or basis.matrix.shape[0] != history.shape[-1]:
        raise ValueError('PROBE_CHANNEL_SHAPE')
    if batch_size < 1 or not len(history):
        raise ValueError('PROBE_BATCH_OR_EMPTY')
    matrix = basis.matrix.astype(np.float32)
    latent_past = history @ matrix
    chunks = []
    for start in range(0, len(history), batch_size):
        if check is not None:
            check()
        values = np.asarray(predictor(latent_past[start:start + batch_size]), dtype=np.float32)
        shape = (min(batch_size, len(history) - start), raw_forecast.shape[1], basis.rank)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError('PREDICTOR_LATENT_SHAPE_OR_NONFINITE')
        chunks.append(values)
        if check is not None:
            check()
    latent_future = np.concatenate(chunks)
    last = history[:, -1].copy()
    return {'projected_raw_forecast': project_raw_forecast(raw_forecast, last, basis, np.float32),
            'projected_raw_reference_fp64': project_raw_forecast(raw_forecast, last, basis),
            'projected_past_forecast': decode_forecast(latent_future, last, basis, np.float32),
            'latent_past': latent_past, 'latent_last': latent_past[:, -1:].copy(),
            'latent_future': latent_future, 'raw_last': last,
            'reconstructed_history': latent_past @ matrix.T}


def observed_score(prediction, target, observed, std=None):
    error = np.asarray(prediction, np.float64) - np.asarray(target, np.float64)
    if std is not None:
        error = error * std
    error = np.where(observed, error, 0.)
    count = observed.sum(axis=(0, 1))
    active = count > 0
    mse = np.divide((error ** 2).sum((0, 1)), count, out=np.zeros(len(count)), where=active)
    mae = np.divide(np.abs(error).sum((0, 1)), count, out=np.zeros(len(count)), where=active)
    return {'mse': float(mse[active].mean()) if active.any() else None,
            'mae': float(mae[active].mean()) if active.any() else None,
            'per_channel_mse': [float(v) if a else None for v, a in zip(mse, active)],
            'per_channel_mae': [float(v) if a else None for v, a in zip(mae, active)],
            'observed_counts': count.tolist(), 'active_channels': int(active.sum())}


def difference_summary(left, right):
    difference = np.asarray(left, np.float64) - np.asarray(right, np.float64)
    return {'mse': float(np.mean(difference ** 2)), 'mae': float(np.mean(np.abs(difference))),
            'max_abs': float(np.max(np.abs(difference)))}


def array_sha256(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _save(output, name, arrays):
    path = output / name
    with path.open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    return {'path': name, 'sha256': _sha(path), 'bytes': path.stat().st_size,
            'arrays': {key: {'shape': list(value.shape), 'dtype': str(value.dtype),
                             'sha256': array_sha256(value)} for key, value in arrays.items()}}


def _validate_inputs(splits, mean, std, actual):
    if not {'fit', 'tune'} <= set(splits) or set(splits) - set(SPLIT_ROLES):
        raise ValueError('SPLIT_SCOPE: only FIT/TUNE/reused VAL/already-exposed EVAL')
    if 'val' in splits and 'eval' in splits:
        raise ValueError('SPLIT_SCOPE: one reused evaluation role per dataset')
    channels = splits['fit']['history'].shape[-1]
    if mean.shape != (channels,) or std.shape != (channels,) or mean.dtype != np.float32 or std.dtype != np.float32:
        raise ValueError('EXACT_STORED_NORMALIZATION_SHAPE_OR_DTYPE')
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError('NORMALIZATION_NONFINITE_OR_NONPOSITIVE')
    for split, values in splits.items():
        history, raw, target = (values[key] for key in ('history', 'raw_forecast', 'target'))
        observed, past_mask, origins = (values[key] for key in ('observed', 'history_observed', 'origins'))
        if history.ndim != 3 or raw.ndim != 3 or raw.shape != target.shape or raw.shape != observed.shape:
            raise ValueError('INPUT_SHAPE: ' + split)
        if history.shape[0] != raw.shape[0] or history.shape[-1] != channels or raw.shape[-1] != channels:
            raise ValueError('INPUT_CHANNEL_OR_WINDOW_SHAPE: ' + split)
        if past_mask.shape != history.shape or origins.shape != (len(history),) or not len(history):
            raise ValueError('INPUT_MASK_ORIGIN_SHAPE: ' + split)
        if any(v.dtype != np.float32 for v in (history, raw, target)):
            raise ValueError('EXACT_OLD_FP32_INPUT_REQUIRED: ' + split)
        if observed.dtype != np.bool_ or past_mask.dtype != np.bool_ or origins.dtype != np.int64:
            raise ValueError('EXACT_OLD_MASK_ORIGIN_DTYPE: ' + split)
        if not np.isfinite(history).all() or not np.isfinite(raw).all() or not np.isfinite(target[observed]).all():
            raise ValueError('INPUT_NONFINITE: ' + split)
        if actual and (history.shape[1] != 512 or raw.shape[1] != 48):
            raise ValueError('OLD_L512_H48_REQUIRED: ' + split)


class ProcessGuard:
    """Cooperative local cap plus the caller's existing shared resource ledger."""
    def __init__(self, shared_limit):
        import psutil
        self.process = psutil.Process()
        self.shared_limit = shared_limit

    def check(self):
        if time.time() - self.process.create_time() >= 300:
            raise RuntimeError('D1_PROCESS_WALL_300S')
        if self.process.memory_info().rss > 8 * 1024 ** 3:
            raise RuntimeError('D1_PROCESS_RSS_8GIB')
        self.shared_limit.check()


def make_bolt_predictor(bolt):
    """Adapt an already loaded, frozen, source-verified original Bolt to E(X)."""
    import torch

    def predict(history):
        with torch.no_grad():
            value = bolt(torch.as_tensor(history, dtype=torch.float32, device='cuda'))
            return value.detach().cpu().numpy()
    return predict


def run_probe(splits, predictor, mean, std, output, *, provenance, actual=False,
              shared_limit=None, batch_size=8, identity_atol=2e-6, identity_rtol=2e-6):
    """Save every fixed rank; never train, select a rank, or read a source asset.

The caller verifies source/RAW/MAIN CP hashes, original origins/statistics and
exact normalized past BEFORE calling. ``actual=True`` enforces L512/H48,
fresh followup-cache output, the shared Limit and a local process cap.
The caller owns shared Limit creation/terminal receipt and the external watchdog.
"""
    if actual and shared_limit is None:
        raise ValueError('SHARED_LIMIT_REQUIRED')
    mean, std = np.asarray(mean), np.asarray(std)
    _validate_inputs(splits, mean, std, actual)
    output = Path(output).resolve()
    if actual and not output.is_relative_to(CACHE):
        raise ValueError('SCIENTIFIC_OUTPUT_OUTSIDE_FOLLOWUP_CACHE')
    output.mkdir(parents=True, exist_ok=False)
    guard = ProcessGuard(shared_limit) if actual else None
    check = guard.check if guard else None
    result = {'status': 'RUNNING', 'scientific_execution': bool(actual),
              'storage_scale': 'normalized', 'normalization_origin_split': 'fit',
              'split_roles': {s: SPLIT_ROLES[s] for s in splits}, 'provenance': provenance,
              'basis_fit_scope': 'FIT_RAW_forecast_minus_exact_preprocessed_current_last_only',
              'model_selection': 'none; every quarter/half/full-C rotated and identity candidate retained',
              'decode_scope': 'fixed linear B diagnostic; no universal nonlinear-D lower bound',
              'current_state_bypass_C': len(mean), 'strict_K_only_compression': False,
              'runtime_projection_dtype': 'float32', 'basis_eigensystem_dtype': 'float64',
              'identity_tolerance': {'atol': identity_atol, 'rtol': identity_rtol},
              'batch_size': batch_size, 'candidates': [], 'artifacts': []}
    location = {'phase': 'inputs', 'candidate': None, 'split': None, 'path': str(output)}
    try:
        for split, values in splits.items():
            location.update(split=split, path=str(output / ('INPUT_' + split + '.npz')))
            if check:
                check()
            artifact = _save(output, 'INPUT_' + split + '.npz', {
                'history_normalized': values['history'], 'history_observed': values['history_observed'],
                'raw_forecast': values['raw_forecast'], 'target': values['target'],
                'observed': values['observed'], 'origins': values['origins'],
                'raw_last': values['history'][:, -1].copy(), 'mean': mean, 'std': std})
            result['artifacts'].append(artifact)
        location.update(phase='basis_fit', candidate=None, split='fit', path=str(output / 'INPUT_fit.npz'))
        family = fit_bases(splits['fit']['raw_forecast'], splits['fit']['history'][:, -1])
        for basis in family:
            location.update(phase='basis', candidate=basis.name, split=None,
                            path=str(output / (basis.name + '__basis.npz')))
            artifact = _save(output, basis.name + '__basis.npz', {
                'basis': basis.matrix, 'basis_runtime_float32': basis.matrix.astype(np.float32),
                'delta_mean': basis.mean, 'mean_perpendicular': mean_perpendicular(basis),
                'eigenvalues': basis.eigenvalues})
            result['artifacts'].append(artifact)
            candidate = {'name': basis.name, 'mode': basis.mode, 'K': basis.rank,
                         'rank_label': basis.rank_label, 'full_C_is_compression_claim': False,
                         'basis_artifact': artifact, 'splits': {}}
            result['candidates'].append(candidate)
            for split, values in splits.items():
                name = basis.name + '__' + split + '.npz'
                location.update(phase='forecast', split=split, path=str(output / name))
                arrays = probe_split(values['history'], values['raw_forecast'], basis, predictor,
                                     batch_size=batch_size, check=check)
                artifact = _save(output, name, arrays)
                result['artifacts'].append(artifact)
                forecasts = {'raw': values['raw_forecast'], 'projected_raw': arrays['projected_raw_forecast'],
                             'projected_past': arrays['projected_past_forecast']}
                row = {'artifact': artifact,
                       'past_reconstruction': difference_summary(arrays['reconstructed_history'], values['history']),
                       'forecast_noncommutation': difference_summary(forecasts['projected_past'], forecasts['projected_raw']),
                       'raw_projection_vs_original_raw': difference_summary(forecasts['projected_raw'], forecasts['raw']),
                       'runtime_raw_projection_vs_fp64_reference': difference_summary(
                           forecasts['projected_raw'], arrays['projected_raw_reference_fp64']),
                       'scores': {name: {**observed_score(forecast, values['target'], values['observed']),
                                        'physical': observed_score(forecast, values['target'], values['observed'], std)}
                                  for name, forecast in forecasts.items()}}
                candidate['splits'][split] = row
                if basis.mode == 'identity':
                    row['identity_forecast_vs_stored_raw'] = difference_summary(arrays['latent_future'], values['raw_forecast'])
                    row['identity_parity_pass'] = bool(np.allclose(arrays['latent_future'], values['raw_forecast'],
                                                                   atol=identity_atol, rtol=identity_rtol))
                    if not row['identity_parity_pass']:
                        raise ValueError('IDENTITY_FORECAST_VS_STORED_RAW: ' + split)
                if check:
                    check()
        result['status'] = 'COMPLETE'
    except Exception as error:
        result.update(status='FAILED_PRESERVED', failure={**location, 'type': type(error).__name__,
                                                        'message': str(error)})
        raise
    finally:
        with (output / 'RESULT.json').open('x', encoding='utf-8') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
    return result


def describe():
    return {'status': 'PREPARATION_ONLY', 'actual_data_opened': False, 'actual_gpu_executed': False,
            'entrypoint': 'run_probe(splits, make_bolt_predictor(original_bolt), stored_mean, stored_std, fresh_output, provenance=verified_manifest, actual=True, shared_limit=existing_Bi_rt_Limit)',
            'remaining_runner_responsibility': 'Reconstruct and independently verify exact old normalized histories/stats/origins and RAW/MAIN CP/source hashes; root launches only after audited Bi completion.',
            'input_schema': {'splits': 'fit/tune/val OR fit/tune/already-exposed eval; no test',
                             'history': 'float32[N,512,C]', 'raw_forecast': 'float32[N,48,C]',
                             'target': 'float32[N,48,C]', 'observed': 'bool[N,48,C]',
                             'history_observed': 'bool[N,512,C]', 'origins': 'int64[N]',
                             'mean/std': 'exact stored float32[C]'},
            'output': 'INPUT_<split>.npz; <candidate>__basis.npz; <candidate>__<split>.npz; RESULT.json',
            'resource': 'caller shared Limit plus cooperative 300-second process/RSS8GiB guard; caller external watchdog and terminal resource receipt'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(describe(), ensure_ascii=False, indent=2))
