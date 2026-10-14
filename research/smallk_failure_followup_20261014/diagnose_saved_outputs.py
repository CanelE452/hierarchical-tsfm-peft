"""CPU-only saved-output diagnostics; preparation does not open study arrays."""
from pathlib import Path
import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import os
import time
import zipfile

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPLIT_ROLES = {'training', 'model_selection', 'reused_development', 'exposed_evaluation'}
QUANTILES = (.5, .9, .95, .99)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checked_path(value, base=ROOT):
    path = Path(value)
    if not path.is_absolute():
        path = Path(base) / path
    path = path.resolve()
    if any(part.lower() in {'test', 'test.npz', 'jena_test.npz', 'protected_test'} for part in path.parts):
        raise PermissionError('Protected TEST path must not be opened')
    return path


def exact(a, b, label):
    if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b, equal_nan=True):
        raise ValueError('EXACT_CONDITION_MISMATCH:' + label)


def validate_statistics(mean, std):
    if mean.ndim != 1 or std.shape != mean.shape or not np.isfinite(mean).all():
        raise ValueError('Invalid channel statistics')
    if not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError('Stored FIT std must be finite and positive; do not silently refit or replace it')


def validate_split(values, channels):
    pred, target, mask, origins = (values[key] for key in ('prediction', 'target', 'observed', 'origins'))
    if pred.ndim != 3 or pred.shape != target.shape or mask.shape != target.shape or pred.shape[-1] != channels:
        raise ValueError('Forecast/target/mask dimensions disagree')
    if not pred.shape[0] or not pred.shape[1] or mask.dtype != np.dtype(bool):
        raise ValueError('Empty windows or nonboolean observation mask')
    if not np.isfinite(pred).all() or not np.isfinite(target[mask]).all():
        raise ValueError('Nonfinite prediction or observed target')
    if origins.shape != (len(pred),) or not np.issubdtype(origins.dtype, np.integer):
        raise ValueError('Origins must be an integer per window')
    if len(np.unique(origins)) != len(origins) or (np.diff(origins) <= 0).any():
        raise ValueError('Origins must be unique and strictly increasing; duplicates cannot be independent samples')
    if not mask.any():
        raise ValueError('No observed target in split')


@dataclass
class Bundle:
    mean: np.ndarray
    std: np.ndarray
    splits: dict
    storage_scale: str
    source: dict

    def normalized(self, split, field='prediction'):
        value = self.splits[split][field].astype(np.float64)
        return value if self.storage_scale == 'normalized' else (value - self.mean) / self.std

    def physical(self, split, field='prediction'):
        value = self.splits[split][field].astype(np.float64)
        return value if self.storage_scale == 'physical' else value * self.std + self.mean


def load_bundle(path, splits, storage_scale, expected_sha=None, max_bytes=256 * 1024 ** 2, allow_exposed_eval=False):
    if not splits or any(split not in {'fit', 'tune', 'val', 'eval'} for split in splits):
        raise ValueError('Unsupported or protected split request; source must not be read')
    if len(set(splits)) != len(splits):
        raise ValueError('Duplicate split request')
    path = checked_path(path)
    if 'eval' in splits and not allow_exposed_eval:
        raise PermissionError('EVAL read requires an explicit already-exposed evaluation contract')
    if storage_scale not in {'normalized', 'physical'}:
        raise ValueError('Explicit storage_scale must be normalized or physical')
    before = sha256(path)
    if expected_sha and before != expected_sha:
        raise ValueError('SOURCE_SHA_MISMATCH:' + str(path))
    with zipfile.ZipFile(path) as archive:
        if sum(item.file_size for item in archive.infolist()) > max_bytes:
            raise ValueError('NPZ uncompressed size exceeds the declared RAM read limit')
    with np.load(path, allow_pickle=False) as arrays:
        mean, std = arrays['mean'].copy(), arrays['std'].copy()
        validate_statistics(mean, std)
        payload = {}
        for split in splits:
            payload[split] = {field: arrays[split + '_' + field].copy()
                              for field in ('prediction', 'target', 'observed', 'origins')}
            validate_split(payload[split], len(mean))
    after = sha256(path)
    if after != before:
        raise ValueError('Source changed while being read')
    return Bundle(mean, std, payload, storage_scale, {'path': str(path), 'sha256': before})


def validate_pair(raw, candidate):
    if raw.storage_scale != candidate.storage_scale or raw.splits.keys() != candidate.splits.keys():
        raise ValueError('Scale or split-set mismatch')
    exact(raw.mean, candidate.mean, 'mean')
    exact(raw.std, candidate.std, 'std')
    for split in raw.splits:
        for field in ('target', 'observed', 'origins'):
            exact(raw.splits[split][field], candidate.splits[split][field], split + ':' + field)
        if raw.splits[split]['prediction'].shape != candidate.splits[split]['prediction'].shape:
            raise ValueError('Prediction shape mismatch')


def channel_metrics(error, mask):
    counts = mask.sum(axis=(0, 1))
    result = []
    for channel in range(error.shape[-1]):
        values = error[:, :, channel][mask[:, :, channel]]
        row = {'channel_index': channel, 'observed_count': int(counts[channel])}
        for name, value in (('mse', values ** 2), ('mae', np.abs(values)), ('signed_bias', values)):
            row[name] = float(value.mean()) if len(values) else None
        for quantile in QUANTILES:
            row['absolute_error_q' + str(quantile)] = float(np.quantile(np.abs(values), quantile)) if len(values) else None
        result.append(row)
    return result


def score(error, mask):
    rows = channel_metrics(error, mask)
    active = [row for row in rows if row['observed_count']]
    if not active:
        return {'normalized_mse': None, 'normalized_mae': None, 'normalized_signed_bias': None,
                'active_channels': 0, 'observed_count': 0}
    return {'normalized_' + key: float(np.mean([row[key] for row in active]))
            for key in ('mse', 'mae', 'signed_bias')} | {
            'active_channels': len(active), 'observed_count': sum(row['observed_count'] for row in active)}


def contribution(error, selected, full_mask):
    counts = full_mask.sum(axis=(0, 1))
    active = counts > 0
    if not active.any():
        raise ValueError('No observed full-split target')
    numerator = np.where(selected & full_mask, error ** 2, 0.).sum(axis=(0, 1))
    values = np.zeros(len(counts), np.float64)
    values[active] = numerator[active] / counts[active] / active.sum()
    return values


def split_overlap_info(bundle, roles):
    if set(roles) != set(bundle.splits) or any(role not in SPLIT_ROLES for role in roles.values()):
        raise ValueError('Explicit split roles must be training/model_selection/reused_development/exposed_evaluation')
    if 'eval' in roles and roles['eval'] != 'exposed_evaluation':
        raise ValueError('This followup only accepts already exposed EVAL as development evidence')
    if roles.get('fit') != 'training':
        raise ValueError('FIT must be explicitly declared training')
    comparisons = []
    for i, left in enumerate(bundle.splits):
        a = bundle.splits[left]
        for right in list(bundle.splits)[i + 1:]:
            b = bundle.splits[right]
            overlap_pairs = sum(int(((b['origins'] < origin + a['prediction'].shape[1]) &
                                     (b['origins'] + b['prediction'].shape[1] > origin)).sum())
                                for origin in a['origins'])
            comparisons.append({'left': left, 'right': right, 'overlapping_target_window_pairs': overlap_pairs})
    within = []
    for split, values in bundle.splits.items():
        origins = values['origins']
        overlap_pairs = sum(int(np.searchsorted(origins, origin + values['prediction'].shape[1], side='left') - index - 1)
                            for index, origin in enumerate(origins))
        within.append({'split': split, 'overlapping_target_window_pairs': overlap_pairs})
    return {'roles_declared': roles, 'target_interval_overlap': comparisons, 'within_split_target_interval_overlap': within,
            'independent_generalization_established': False,
            'reason': 'Saved arrays, unique origins and nonoverlap do not prove an independent evaluation or selection history.'}


def describe_channels(bundle, split, prediction, target, mask, error, method, seed, channels):
    physical_error = error * bundle.std
    norm_rows, physical_rows = channel_metrics(error, mask), channel_metrics(physical_error, mask)
    rows = []
    for channel, (norm, physical) in enumerate(zip(norm_rows, physical_rows)):
        valid = mask[:, :, channel]
        ys, ps = target[:, :, channel][valid], prediction[:, :, channel][valid]
        row = {'split': split, 'method': method, 'seed': seed, 'channel': channels[channel],
               'channel_index': channel, 'observed_count': norm['observed_count'],
               'stored_fit_mean': float(bundle.mean[channel]), 'stored_fit_std': float(bundle.std[channel])}
        row.update({'normalized_' + key: value for key, value in norm.items() if key not in ('channel_index', 'observed_count')})
        row.update({'physical_' + key: value for key, value in physical.items() if key not in ('channel_index', 'observed_count')})
        for label, values in (('target', ys), ('prediction', ps)):
            row[label + '_normalized_mean'] = float(values.mean()) if len(values) else None
            row[label + '_normalized_std'] = float(values.std()) if len(values) else None
            row[label + '_physical_mean'] = float(values.mean() * bundle.std[channel] + bundle.mean[channel]) if len(values) else None
            row[label + '_physical_std'] = float(values.std() * bundle.std[channel]) if len(values) else None
        rows.append(row)
    return rows


def tail_rows(raw_error, error, full_mask, fit_raw_error, fit_mask, method, seed, split, channels, quantiles):
    rows = []
    for quantile in quantiles:
        for reference in ('same_split_raw', 'same_split_own', 'fit_raw'):
            reference_error = fit_raw_error if reference == 'fit_raw' else error if reference == 'same_split_own' else raw_error
            reference_mask = fit_mask if reference == 'fit_raw' else full_mask
            raw_reference_error = fit_raw_error if reference == 'fit_raw' else raw_error
            cutoff, raw_cutoff = [], []
            for channel in range(error.shape[-1]):
                observed = reference_mask[:, :, channel]
                cutoff.append(float(np.quantile(np.abs(reference_error[:, :, channel][observed]), quantile)) if observed.any() else None)
                raw_cutoff.append(float(np.quantile(np.abs(raw_reference_error[:, :, channel][observed]), quantile)) if observed.any() else None)
            tail = np.zeros_like(full_mask)
            raw_tail = np.zeros_like(full_mask)
            for channel, (threshold, raw_threshold) in enumerate(zip(cutoff, raw_cutoff)):
                if threshold is not None:
                    tail[:, :, channel] = np.abs(error[:, :, channel]) >= threshold
                    if reference != 'same_split_own':
                        tail[:, :, channel] = np.abs(raw_error[:, :, channel]) >= threshold
                if raw_threshold is not None:
                    raw_tail[:, :, channel] = np.abs(raw_error[:, :, channel]) >= raw_threshold
            for region in ('tail', 'body'):
                selected = tail if region == 'tail' else ~tail
                raw_selected = raw_tail if region == 'tail' else ~raw_tail
                parts = contribution(error, selected, full_mask)
                raw_parts = contribution(raw_error, raw_selected, full_mask)
                for channel in range(error.shape[-1]):
                    rows.append({'split': split, 'method': method, 'seed': seed,
                                 'channel_index': channel, 'channel': channels[channel],
                                 'quantile': quantile, 'cutoff_reference': reference, 'region': region,
                                 'cutoff_normalized_abs_error': cutoff[channel], 'raw_cutoff_normalized_abs_error': raw_cutoff[channel],
                                 'selected_observed_count': int((selected[:, :, channel] & full_mask[:, :, channel]).sum()),
                                 'raw_selected_observed_count': int((raw_selected[:, :, channel] & full_mask[:, :, channel]).sum()),
                                 'normalized_mse_contribution': float(parts[channel]),
                                 'raw_normalized_mse_contribution': float(raw_parts[channel]),
                                 'gap_contribution': float(parts[channel] - raw_parts[channel]),
                                 'paired_same_events': reference != 'same_split_own',
                                 'channelmacro_denominator_uses_full_split': True})
    return rows


def partition_rows(error, raw_error, mask, origins, method, seed, split, groups, time_bins):
    rows = []
    partitions = [('horizon', str(index + 1), np.arange(len(origins)), np.array([index])) for index in range(error.shape[1])]
    partitions += [('horizon_group', group['name'], np.arange(len(origins)), np.arange(group['start'], group['stop']))
                   for group in groups]
    partitions += [('origin_bin', str(index + 1), indices, np.arange(error.shape[1]))
                   for index, indices in enumerate(np.array_split(np.arange(len(origins)), min(time_bins, len(origins))))]
    for kind, name, ni, hi in partitions:
        selected = np.zeros_like(mask)
        selected[np.ix_(ni, hi, np.arange(error.shape[-1]))] = True
        parts, raw_parts = contribution(error, selected, mask), contribution(raw_error, selected, mask)
        values = score(error, selected & mask)
        rows.append({'split': split, 'method': method, 'seed': seed, 'partition': kind, 'group': name,
                     'origin_start': int(origins[ni[0]]), 'origin_end_inclusive': int(origins[ni[-1]]),
                     'horizon_start_1based': int(hi[0] + 1), 'horizon_end_inclusive': int(hi[-1] + 1),
                     'window_count': len(ni), **values,
                     'normalized_mse_contribution': float(parts.sum()),
                     'raw_normalized_mse_contribution': float(raw_parts.sum()),
                     'gap_contribution': float((parts - raw_parts).sum())})
    return rows


def default_horizon_groups(horizon):
    boundaries = sorted({0, min(8, horizon), min(24, horizon), horizon})
    return [{'name': str(lo + 1) + '_' + str(hi), 'start': lo, 'stop': hi}
            for lo, hi in zip(boundaries[:-1], boundaries[1:])]


def diagnose_pair(raw, candidate, method, seed, channels=None, tail_quantiles=(.9, .95, .99), time_bins=4):
    validate_pair(raw, candidate)
    channels = channels or [str(i) for i in range(len(raw.mean))]
    if len(channels) != len(raw.mean) or len(set(channels)) != len(channels):
        raise ValueError('Channel labels must be unique and match C')
    if 'fit' not in raw.splits or time_bins < 1 or any(not 0 < q < 1 for q in tail_quantiles):
        raise ValueError('FIT, positive time_bins and interior tail quantiles are required')
    tables = {'summary': [], 'channels': [], 'partitions': [], 'tails': [], 'reconstruction': []}
    fit_raw_error = raw.normalized('fit') - raw.normalized('fit', 'target')
    for split, payload in raw.splits.items():
        mask, origins = payload['observed'], payload['origins']
        raw_error = raw.normalized(split) - raw.normalized(split, 'target')
        prediction, target = candidate.normalized(split), candidate.normalized(split, 'target')
        error = prediction - target
        metrics, raw_metrics = score(error, mask), score(raw_error, mask)
        tables['summary'].append({'split': split, 'method': method, 'seed': seed, **metrics,
                                  'raw_normalized_mse': raw_metrics['normalized_mse'],
                                  'raw_normalized_mae': raw_metrics['normalized_mae'],
                                  'mse_gap': metrics['normalized_mse'] - raw_metrics['normalized_mse'],
                                  'mae_gap': metrics['normalized_mae'] - raw_metrics['normalized_mae']})
        tables['channels'].extend(describe_channels(candidate, split, prediction, target, mask, error, method, seed, channels))
        partitions = partition_rows(error, raw_error, mask, origins, method, seed, split,
                                    default_horizon_groups(error.shape[1]), time_bins)
        tails = tail_rows(raw_error, error, mask, fit_raw_error, raw.splits['fit']['observed'],
                         method, seed, split, channels, tail_quantiles)
        tables['partitions'].extend(partitions)
        tables['tails'].extend(tails)
        for kind in ('horizon', 'horizon_group', 'origin_bin'):
            reconstructed = sum(row['gap_contribution'] for row in partitions if row['partition'] == kind)
            tables['reconstruction'].append({'split': split, 'method': method, 'seed': seed,
                'decomposition': kind, 'mse_gap': metrics['normalized_mse'] - raw_metrics['normalized_mse'],
                'reconstructed_gap': reconstructed, 'absolute_difference': abs(reconstructed - metrics['normalized_mse'] + raw_metrics['normalized_mse'])})
        for quantile in tail_quantiles:
            for reference in ('same_split_raw', 'same_split_own', 'fit_raw'):
                reconstructed = sum(row['gap_contribution'] for row in tails
                                    if row['quantile'] == quantile and row['cutoff_reference'] == reference)
                tables['reconstruction'].append({'split': split, 'method': method, 'seed': seed,
                    'decomposition': reference + ':' + str(quantile), 'mse_gap': metrics['normalized_mse'] - raw_metrics['normalized_mse'],
                    'reconstructed_gap': reconstructed, 'absolute_difference': abs(reconstructed - metrics['normalized_mse'] + raw_metrics['normalized_mse'])})
    if any(row['absolute_difference'] > 1e-10 * max(1., abs(row['mse_gap'])) for row in tables['reconstruction']):
        raise ValueError('Channel-macro gap reconstruction failed')
    return tables


@dataclass
class Projection:
    basis: np.ndarray
    center: np.ndarray
    mode: str
    k: int
    fit_energy: float
    retained_fit_energy_fraction: float | None
    effective_fit_rank: int

    def apply(self, delta):
        delta = np.asarray(delta, np.float64)
        if delta.shape[-1] != len(self.center) or not np.isfinite(delta).all():
            raise ValueError('Projection requires finite full-channel forecast changes')
        return self.center + ((delta - self.center) @ self.basis) @ self.basis.T


def fit_projection(fit_delta, k, mode):
    values = np.asarray(fit_delta, np.float64)
    if values.ndim != 3 or not values.shape[0] or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError('FIT delta must be finite [N,H,C] RAW forecasts, without actual future targets')
    if not isinstance(k, int) or not 1 <= k <= values.shape[-1] or mode not in {'through_origin', 'mean_centered'}:
        raise ValueError('Invalid rank or projection mode')
    flattened = values.reshape(-1, values.shape[-1])
    center = flattened.mean(0) if mode == 'mean_centered' else np.zeros(values.shape[-1])
    centered = flattened - center
    energy_matrix = centered.T @ centered / len(centered)
    eigenvalues, vectors = np.linalg.eigh(energy_matrix)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, vectors = np.maximum(eigenvalues[order], 0.), vectors[:, order]
    basis = vectors[:, :k].copy()
    for col in range(k):
        if basis[np.argmax(np.abs(basis[:, col])), col] < 0:
            basis[:, col] *= -1
    total = float(eigenvalues.sum())
    tolerance = max(centered.shape) * np.finfo(np.float64).eps * (eigenvalues[0] if len(eigenvalues) else 0.)
    return Projection(basis, center, mode, k, total,
                      float(eigenvalues[:k].sum() / total) if total > 0 else None,
                      int((eigenvalues > tolerance).sum()))


def rank_projection_diagnostics(raw, normalized_last, k):
    if set(normalized_last) != set(raw.splits) or 'fit' not in raw.splits:
        raise ValueError('Matching FIT and split-specific last states are required')
    for split, last in normalized_last.items():
        if last.shape != (len(raw.splits[split]['prediction']), len(raw.mean)) or not np.isfinite(last).all():
            raise ValueError('Last-state dimensions or finite values disagree')
    fit_delta = raw.normalized('fit') - normalized_last['fit'][:, None, :]
    projections = [fit_projection(fit_delta, k, mode) for mode in ('through_origin', 'mean_centered')]
    rows, arrays = [], {}
    for projection in projections:
        arrays[projection.mode + '_basis'] = projection.basis
        arrays[projection.mode + '_center'] = projection.center
        for split, payload in raw.splits.items():
            prediction = raw.normalized(split)
            last = normalized_last[split][:, None, :]
            projected = last + projection.apply(prediction - last)
            approximation = score(projected - prediction, np.ones_like(payload['observed']))
            actual = score(projected - raw.normalized(split, 'target'), payload['observed'])
            baseline = score(prediction - raw.normalized(split, 'target'), payload['observed'])
            rows.append({'split': split, 'projection_mode': projection.mode, 'k': k,
                         'fit_basis_uses_actual_target': False, 'basis_fitted_split': 'fit',
                         'fit_energy_definition': 'centered_variance' if projection.mode == 'mean_centered' else 'through_origin_second_moment',
                         'fit_energy': projection.fit_energy, 'retained_fit_energy_fraction': projection.retained_fit_energy_fraction,
                         'effective_fit_rank': projection.effective_fit_rank,
                         'raw_forecast_approximation_normalized_mse': approximation['normalized_mse'],
                         'raw_forecast_approximation_normalized_mae': approximation['normalized_mae'],
                         **actual, 'raw_actual_target_normalized_mse': baseline['normalized_mse'],
                         'actual_target_mse_gap': actual['normalized_mse'] - baseline['normalized_mse'],
                         'requires_full_c_raw_forecast': True, 'nonlinear_decoder_lower_bound': False})
            arrays[split + '_' + projection.mode + '_prediction_normalized'] = projected
    return rows, arrays


def load_last(path, raw, expected_sha=None):
    path = checked_path(path)
    before = sha256(path)
    if expected_sha and before != expected_sha:
        raise ValueError('LAST_SOURCE_SHA_MISMATCH')
    with zipfile.ZipFile(path) as archive:
        if sum(item.file_size for item in archive.infolist()) > 256 * 1024 ** 2:
            raise ValueError('Last-state NPZ uncompressed size exceeds RAM read limit')
    with np.load(path, allow_pickle=False) as payload:
        exact(payload['mean'], raw.mean, 'last:mean')
        exact(payload['std'], raw.std, 'last:std')
        values = {}
        for split in raw.splits:
            exact(payload[split + '_origins'], raw.splits[split]['origins'], split + ':last:origins')
            last = payload[split + '_last'].astype(np.float64)
            values[split] = last if raw.storage_scale == 'normalized' else (last - raw.mean) / raw.std
    if sha256(path) != before:
        raise ValueError('Last-state source changed while being read')
    return values, {'path': str(path), 'sha256': before, 'storage_scale': raw.storage_scale}


def seed_means(rows, identity_fields):
    grouped = {}
    for row in rows:
        if row['seed'] == 'shared_reference':
            continue
        identity = tuple(row.get(key) for key in identity_fields)
        grouped.setdefault(identity, []).append(row)
    result = []
    for records in grouped.values():
        if len({row['seed'] for row in records}) != len(records):
            raise ValueError('Duplicate seed metric identity')
        row = {key: records[0].get(key) for key in identity_fields}
        row.update(seed_aggregation='arithmetic_mean_of_seed_metrics', seed_count=len(records),
                   seeds=json.dumps([record['seed'] for record in records]))
        for key in records[0]:
            if key in identity_fields or key == 'seed':
                continue
            values = [record.get(key) for record in records]
            if all(value is None for value in values):
                row[key] = None
            elif all(isinstance(value, (float, int, np.number)) and not isinstance(value, bool) for value in values):
                row[key] = float(np.mean(values))
            elif all(value == values[0] for value in values):
                row[key] = values[0]
        result.append(row)
    return result


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('x', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def resolve_config(args):
    if args.source_manifest:
        if args.raw or args.candidate:
            raise ValueError('Use explicit NPZ arguments or source manifest, not both')
        manifest_path = checked_path(args.source_manifest)
        config = json.loads(manifest_path.read_text(encoding='utf-8'))
        if config.get('schema') != 'smallk_saved_diagnostics_v1':
            raise ValueError('Unsupported source-manifest schema')
        for source in [config['raw'], *config['candidates']] + ([config['last']] if config.get('last') else []):
            source['path'] = str(checked_path(source['path'], manifest_path.parent))
            if not source.get('sha256'):
                raise ValueError('Manifest sources require SHA256')
        config['source_manifest'] = {'path': str(manifest_path), 'sha256': sha256(manifest_path)}
    else:
        if not args.raw or not args.candidate or not args.storage_scale or not args.split_role:
            raise ValueError('Direct mode requires RAW, candidates, storage scale and explicit split roles')
        roles = dict(pair.split('=', 1) for pair in args.split_role)
        if len(roles) != len(args.split_role):
            raise ValueError('Duplicate split role')
        config = {'schema': 'smallk_saved_diagnostics_v1', 'dataset': args.dataset,
                  'raw': {'path': args.raw}, 'storage_scale': args.storage_scale,
                  'splits': args.splits, 'split_roles': roles, 'normalization_origin_split': 'fit',
                  'candidates': [{'name': name, 'seed': int(seed), 'path': path} for name, seed, path in args.candidate],
                  'time_bins': args.time_bins, 'tail_quantiles': args.tail_quantiles, 'k': args.k}
        if args.last_npz:
            config['last'] = {'path': args.last_npz}
    if config.get('normalization_origin_split') != 'fit':
        raise ValueError('Normalization must be explicitly declared stored FIT statistics')
    if not config.get('dataset') or not config.get('candidates'):
        raise ValueError('Dataset label and candidates are required')
    if len(set(config['splits'])) != len(config['splits']):
        raise ValueError('Duplicate split request')
    identities = [(entry['name'], entry['seed']) for entry in config['candidates']]
    if len(set(identities)) != len(identities) or any(name == 'RAW' for name, _ in identities):
        raise ValueError('Candidate name/seed identities must be unique and RAW is reserved')
    return config


def execute(config, output):
    started = time.perf_counter()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Output must be a fresh directory')
    roles = config['split_roles']
    if set(roles) != set(config['splits']) or any(role not in SPLIT_ROLES for role in roles.values()):
        raise ValueError('Explicit supported split roles are required before opening arrays')
    if roles.get('fit') != 'training' or ('eval' in roles and roles['eval'] != 'exposed_evaluation'):
        raise ValueError('FIT training and already-exposed EVAL contract must be fixed before opening arrays')
    raw = load_bundle(config['raw']['path'], config['splits'], config['storage_scale'], config['raw'].get('sha256'),
                      allow_exposed_eval=roles.get('eval') == 'exposed_evaluation')
    split_information = split_overlap_info(raw, config['split_roles'])
    channels = config.get('channels') or [str(index) for index in range(len(raw.mean))]
    tables = diagnose_pair(raw, raw, 'RAW', 'shared_reference', channels,
                           tuple(config.get('tail_quantiles', (.9, .95, .99))), config.get('time_bins', 4))
    sources = [raw.source]
    for entry in config['candidates']:
        candidate = load_bundle(entry['path'], config['splits'], config['storage_scale'], entry.get('sha256'),
                                allow_exposed_eval=roles.get('eval') == 'exposed_evaluation')
        part = diagnose_pair(raw, candidate, entry['name'], entry['seed'], channels,
                             tuple(config.get('tail_quantiles', (.9, .95, .99))), config.get('time_bins', 4))
        sources.append({**candidate.source, 'method': entry['name'], 'seed': entry['seed']})
        for key, values in part.items():
            tables[key].extend(values)
        del candidate
    rank_rows, rank_arrays, last_source = [], None, None
    if config.get('k') is not None:
        if not config.get('last'):
            raise ValueError('Rank-K change diagnostics require explicit matching last-state NPZ')
        last, last_source = load_last(config['last']['path'], raw, config['last'].get('sha256'))
        rank_rows, rank_arrays = rank_projection_diagnostics(raw, last, config['k'])
    output.mkdir(parents=True, exist_ok=False)
    for key, rows in tables.items():
        write_csv(output / (key + '.csv'), rows)
    for table, keys in [('summary', ['split', 'method']), ('channels', ['split', 'method', 'channel_index', 'channel']),
                        ('partitions', ['split', 'method', 'partition', 'group']),
                        ('tails', ['split', 'method', 'channel_index', 'channel', 'quantile', 'cutoff_reference', 'region'])]:
        write_csv(output / (table + '_seed_metric_means.csv'), seed_means(tables[table], keys))
    if rank_arrays is not None:
        write_csv(output / 'rank_projection.csv', rank_rows)
        with (output / 'rank_projection.npz').open('xb') as handle:
            np.savez_compressed(handle, **rank_arrays)
    with (output / 'REPORT_KO.md').open('x', encoding='utf-8') as handle:
        handle.write('# 저장 예측의 D0/D1 진단\n\n'
                     '[확인] 명시된 저장 배열을 CPU로 읽어 오차를 분해했다. 학습·새 모델 추론·독립 평가를 수행한 보고서가 아니다.\n\n'
                     '- 모든 비교는 mean/std·target·mask·origin이 동일한 조건이다. 통계가 FIT에서 만들어졌다는 출처 선언을 재사용하며 원자료에서 재계산하지 않았다.\n'
                     '- 정규화 지표는 관측값이 있는 채널별 오차의 균등 평균이다. 서로 단위가 다른 원단위 오차는 channels.csv에서 채널별로 보고한다.\n'
                     '- 저장 값을 float64로 변환해 재채점했다. 기존 FP32 변환·reduction의 기록과 차이가 나면 별도 수치로 보존하며 같은 결과로 강제하지 않는다.\n'
                     '- seed_metric_means는 seed별 지표의 산술 평균이며 예측 앙상블, 독립 표본수, 신뢰구간이 아니다. RAW는 공통 참조로 한 번만 저장했다.\n'
                     '- origin_bin은 저장 origin 순서의 동일 개수 구간이다. calendar 시간 구간이 아니며 겹치는 창을 독립 표본으로 해석하지 않는다.\n'
                     '- same_split_raw와 fit_raw는 후보/RAW에 같은 사건을 적용한다. same_split_own은 각 방법의 서로 다른 큰 오차 사건이므로 paired 비교가 아니다. 경계 동률은 모두 포함했다.\n'
                     '- 모든 contribution은 원 split의 채널별 전체 관측 수와 활성 채널 수를 분모로 사용한다. tail+body 및 horizon/time partition의 합은 전체 MSE gap을 재구성한다.\n'
                     '- rank_projection은 FIT의 RAW forecast 변화만으로 선형 basis/mean을 고정한다. RAW forecast 근사오차와 실제 Y 예측오차를 분리하며 비선형 decoder 하한이라고 주장하지 않는다. 이 경로는 full-C RAW 추론을 먼저 필요로 한다.\n\n'
                     '원수치: [summary.csv](summary.csv), [channels.csv](channels.csv), [partitions.csv](partitions.csv), '
                     '[tails.csv](tails.csv), [reconstruction.csv](reconstruction.csv). 출처와 split 경계는 [RESULT.json](RESULT.json)에 기록했다.\n')
    result = {'status': 'COMPLETE_SAVED_OUTPUT_DIAGNOSTICS', 'experiment_complete': False,
              'dataset': config['dataset'], 'storage_scale': config['storage_scale'],
              'sources': sources, 'last_source': last_source, 'source_manifest': config.get('source_manifest'),
              'statistics_provenance': 'declared_stored_FIT_statistics; raw FIT statistics not independently recomputed here',
              'split_information': split_information, 'tables_rows': {key: len(value) for key, value in tables.items()},
              'rank_projection_rows': len(rank_rows), 'runtime_seconds': time.perf_counter() - started,
              'gpu_used': False, 'new_inference': False, 'ci_or_independent_sample_claim': False,
              'artifacts': [{'path': str(path), 'sha256': sha256(path)} for path in sorted(output.iterdir())]}
    with (output / 'RESULT.json').open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-saved-diagnostics', action='store_true', help='Explicit root-directed execution after Bi study')
    parser.add_argument('--source-manifest')
    parser.add_argument('--raw')
    parser.add_argument('--candidate', action='append', nargs=3, metavar=('NAME', 'SEED', 'NPZ'))
    parser.add_argument('--dataset')
    parser.add_argument('--storage-scale', choices=('normalized', 'physical'))
    parser.add_argument('--splits', nargs='+', default=['fit', 'tune', 'val'])
    parser.add_argument('--split-role', action='append', help='split=explicit_role; use already exposed EVAL only')
    parser.add_argument('--time-bins', type=int, default=4)
    parser.add_argument('--tail-quantiles', type=float, nargs='+', default=[.9, .95, .99])
    parser.add_argument('--k', type=int)
    parser.add_argument('--last-npz')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not args.execute_saved_diagnostics:
        parser.error('PREPARATION_ONLY: real array execution requires explicit --execute-saved-diagnostics')
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        result = execute(resolve_config(args), args.output)
    print(json.dumps({'status': result['status'], 'output': str(Path(args.output).resolve()),
                      'experiment_complete': False}, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
