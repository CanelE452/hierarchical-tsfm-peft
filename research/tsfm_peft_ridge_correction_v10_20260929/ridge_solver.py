"""Streaming sufficient statistics for the approved masked macro ridge objective."""
from dataclasses import dataclass, field
import hashlib

import numpy as np


@dataclass
class RidgeStatistics:
    context: int
    horizon: int
    channels: int
    origins: int
    channel_counts: np.ndarray
    horizon_groups: list
    full_gram: np.ndarray
    missing_grams: list
    rhs: dict
    target_ss: dict
    family: str
    mask_sha256: str
    eigen_cache: dict = field(default_factory=dict)

    def gram(self, group):
        missing = self.missing_grams[group]
        return self.full_gram if missing is None else self.full_gram - missing


def build_statistics(blocks, target, mask, parents, *, family, basis=None):
    """Consume (indices, [B,L,C] context) blocks once, sharing Gram across parents.

    The loss weights are 1/(C*N_c), where N_c counts all observed TRAIN targets
    for channel c across origins and horizons. No target projection or intercept
    is used. All missing targets are excluded before arithmetic.
    """
    if family not in ('ridge_p', 'ridge_raw'):
        raise ValueError('Only approved P and RAW inputs are supported')
    target, mask = np.asarray(target), np.asarray(mask, dtype=bool)
    if target.ndim != 3 or target.shape != mask.shape or not parents:
        raise ValueError('Target/mask must share [N,H,C] and parents cannot be empty')
    n, horizon, channels = target.shape
    counts = mask.sum(axis=(0, 1), dtype=np.int64)
    if (counts == 0).any() or not np.isfinite(target[mask]).all():
        raise ValueError('Every channel needs observed finite TRAIN targets')
    parent_arrays = {key: np.asarray(value) for key, value in parents.items()}
    if any(value.shape != target.shape for value in parent_arrays.values()):
        raise ValueError('Parent predictions must align to TRAIN target axes')
    channel_weights = 1. / (channels * counts.astype(np.float64))
    if family == 'ridge_p':
        basis = np.asarray(basis, dtype=np.float64)
        if basis.ndim != 2 or basis.shape[0] != channels or not np.isfinite(basis).all():
            raise ValueError('P input requires the recorded finite [C,K] basis')

    groups, keys = [], {}
    for h in range(horizon):
        pattern = np.packbits(mask[:, h, :], axis=None).tobytes()
        if pattern not in keys:
            keys[pattern] = len(groups)
            groups.append([])
        groups[keys[pattern]].append(h)
    del keys
    seen = np.zeros(n, dtype=bool)
    stats = None
    source = blocks() if callable(blocks) else blocks
    for indices, context in source:
        indices = np.asarray(indices, dtype=np.int64)
        x = np.asarray(context, dtype=np.float64)
        if (indices.ndim != 1 or len(indices) == 0 or x.ndim != 3
                or x.shape[0] != len(indices) or x.shape[2] != channels
                or np.any(indices < 0) or np.any(indices >= n)
                or len(np.unique(indices)) != len(indices) or seen[indices].any()):
            raise ValueError('Context blocks must cover each TRAIN origin exactly once')
        if not np.isfinite(x).all():
            raise ValueError('Context must already be finite under the data contract')
        seen[indices] = True
        if stats is None:
            length = x.shape[1]
            stats = RidgeStatistics(length, horizon, channels, n, counts, groups,
                np.zeros((length, length), dtype=np.float64),
                [None if mask[:, group[0], :].all() else np.zeros((length, length), dtype=np.float64)
                 for group in groups],
                {key: np.zeros((horizon, length), dtype=np.float64) for key in parents},
                {key: 0. for key in parents}, family,
                hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest())
        elif x.shape[1] != stats.context:
            raise ValueError('Context length changed between blocks')
        z = x - x[:, -1:, :]
        if family == 'ridge_p':
            z = (z @ basis) @ basis.T
        examples = z.transpose(0, 2, 1).reshape(-1, stats.context)
        weights = np.tile(channel_weights, len(indices))
        weighted_examples = examples * np.sqrt(weights)[:, None]
        stats.full_gram += weighted_examples.T @ weighted_examples
        local_mask = mask[indices].transpose(0, 2, 1).reshape(-1, horizon)
        for group_index, group in enumerate(groups):
            if stats.missing_grams[group_index] is not None:
                missing = weighted_examples[~local_mask[:, group[0]]]
                if len(missing):
                    stats.missing_grams[group_index] += missing.T @ missing
        local_target = target[indices].transpose(0, 2, 1).reshape(-1, horizon)
        for key, parent in parent_arrays.items():
            local_parent = parent[indices].transpose(0, 2, 1).reshape(-1, horizon)
            if not np.isfinite(local_parent).all():
                raise ValueError('Parent forecasts must all be finite')
            error = np.zeros(local_target.shape, dtype=np.float64)
            np.subtract(local_target, local_parent, out=error, where=local_mask)
            weighted_error = error * weights[:, None]
            stats.rhs[key] += weighted_error.T @ examples
            stats.target_ss[key] += float(np.sum(error * weighted_error))
    if stats is None or not seen.all():
        raise ValueError('Missing TRAIN context origins')
    stats.full_gram = (stats.full_gram + stats.full_gram.T) / 2
    for i, gram in enumerate(stats.missing_grams):
        if gram is not None:
            stats.missing_grams[i] = (gram + gram.T) / 2
    return stats


def objective_from_statistics(stats, parent_key, weight, penalty):
    weight = np.asarray(weight, dtype=np.float64)
    if weight.shape != (stats.horizon, stats.context):
        raise ValueError('Weight must have [H,L] shape')
    data_loss = stats.target_ss[parent_key] - 2 * float(np.sum(weight * stats.rhs[parent_key]))
    for group_index, horizons in enumerate(stats.horizon_groups):
        selected = weight[horizons]
        data_loss += float(np.sum((selected @ stats.gram(group_index)) * selected))
    regularizer = float(penalty) * float(np.sum(weight * weight)) / stats.horizon
    return {'masked_macro_mse': float(data_loss), 'regularizer': regularizer,
            'objective': float(data_loss + regularizer)}


def solve_ridge(stats, parent_key, penalty):
    """One parent/input/penalty fit; caller reserves its attempt before this call."""
    if float(penalty) <= 0 or parent_key not in stats.rhs:
        raise ValueError('Positive ridge penalty and an existing parent are required')
    ridge = float(penalty) / stats.horizon
    result = np.empty((stats.horizon, stats.context), dtype=np.float64)
    residuals, eigen_records = [], []
    for group_index, horizons in enumerate(stats.horizon_groups):
        gram = stats.gram(group_index)
        if group_index not in stats.eigen_cache:
            values, vectors = np.linalg.eigh(gram)
            floor_tolerance = 1e-10 * max(1., float(np.max(np.abs(values))))
            if float(values.min()) < -floor_tolerance:
                raise ValueError('Sufficient-statistic Gram is not positive semidefinite')
            stats.eigen_cache[group_index] = (values, vectors)
        values, vectors = stats.eigen_cache[group_index]
        rhs = stats.rhs[parent_key][horizons].T
        solved = vectors @ ((vectors.T @ rhs) / (np.maximum(values, 0.)[:, None] + ridge))
        result[horizons] = solved.T
        stationarity = gram @ solved + ridge * solved - rhs
        denominator = ((float(np.linalg.norm(gram)) + ridge) * np.linalg.norm(solved, axis=0)
                       + np.linalg.norm(rhs, axis=0))
        relative = np.linalg.norm(stationarity, axis=0) / np.maximum(denominator, 1e-30)
        residuals.extend(float(v) for v in relative)
        eigen_records.append({'horizons': list(horizons), 'min_eigenvalue': float(values.min()),
                              'roundoff_negative_eigenvalues': int((values < 0).sum())})
    if not np.isfinite(result).all() or max(residuals) > 1e-8:
        raise ValueError('Ridge solve did not satisfy the fixed normal-equation check')
    objective = objective_from_statistics(stats, parent_key, result, penalty)
    if objective['objective'] > stats.target_ss[parent_key] + 1e-8 * max(1., stats.target_ss[parent_key]):
        raise ValueError('Ridge objective exceeds the feasible zero correction')
    receipt = {'family': stats.family, 'penalty': float(penalty), 'dtype': 'float64',
               'context': stats.context, 'horizon': stats.horizon, 'channels': stats.channels,
               'origins': stats.origins, 'fitted_coefficients': int(result.size), 'bias': False,
               'horizon_solve_count': stats.horizon, 'unique_mask_gram_count': len(stats.horizon_groups),
               'channel_target_counts': stats.channel_counts.tolist(), 'mask_sha256': stats.mask_sha256,
               'parent_macro_mse': stats.target_ss[parent_key], **objective,
               'max_relative_normal_equation_residual': max(residuals),
               'normal_equation_relative_tolerance': 1e-8, 'eigen_checks': eigen_records,
               'objective_definition': 'observed channel-macro MSE + penalty*sum(W^2)/H',
               'temporal_rank': int(np.linalg.matrix_rank(result))}
    return result, receipt


def predict_correction(context, weight, *, family, basis=None):
    x = np.asarray(context, dtype=np.float64)
    z = x - x[:, -1:, :]
    if family == 'ridge_p':
        basis = np.asarray(basis, dtype=np.float64)
        z = (z @ basis) @ basis.T
    elif family != 'ridge_raw':
        raise ValueError(family)
    return np.einsum('blc,hl->bhc', z, np.asarray(weight, dtype=np.float64), optimize=True)
