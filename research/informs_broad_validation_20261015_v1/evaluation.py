"""Normalized original-channel FP64 metrics and public band statistics.

No data/model loaders. Macro averages use global observed counts per channel,
not equal averaging of origins. Missing targets are never converted to labels.
"""
import hashlib
import math

import numpy as np


def array_fingerprint(array):
    value = np.ascontiguousarray(array)
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest()}


def _arrays(prediction, target, observed):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    observed = np.asarray(observed)
    if prediction.ndim != 3 or prediction.shape != target.shape or prediction.shape != observed.shape:
        raise ValueError("Expected matching [origins,horizon,channels] arrays")
    if min(prediction.shape) < 1 or observed.dtype != np.bool_:
        raise ValueError("Positive shape and original bool observation mask required")
    if not np.isfinite(prediction[observed]).all() or not np.isfinite(target[observed]).all():
        raise ValueError("Nonfinite prediction/target on an observed target")
    return prediction, target, observed


def horizon_bands(horizon):
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("Positive horizon required")
    edges = sorted({0, math.ceil(horizon / 4), math.ceil(horizon / 2), horizon})
    return list(zip(edges[:-1], edges[1:]))


def error_statistics(pred_normalized, y_normalized, observed, origins, run_id, split, cohort, channel_ids):
    prediction, target, observed = _arrays(pred_normalized, y_normalized, observed)
    origins = np.asarray(origins)
    channels = np.asarray(channel_ids)
    n, horizon, width = prediction.shape
    if origins.shape != (n,) or origins.dtype.kind not in "iu" or len(set(map(int, origins))) != n:
        raise ValueError("Unique integer origins matching prediction order required")
    if channels.shape != (width,) or len(set(map(str, channels))) != width or any(not str(x) for x in channels):
        raise ValueError("Unique nonempty channel IDs matching array order required")
    if not all(str(value) for value in (run_id, split, cohort)):
        raise ValueError("run_id/split/cohort must be nonempty")
    counts = observed.sum(axis=(0, 1), dtype=np.int64)
    channel_sse, channel_sae = [], []
    for channel in range(width):
        difference = prediction[:, :, channel][observed[:, :, channel]] - target[:, :, channel][observed[:, :, channel]]
        channel_sse.append(float(np.sum(difference * difference, dtype=np.float64)))
        channel_sae.append(float(np.sum(np.abs(difference), dtype=np.float64)))
    active = np.flatnonzero(counts)
    score = dict(run_id=str(run_id), split=str(split), cohort=str(cohort),
                 mse=math.fsum(channel_sse[c] / int(counts[c]) for c in active) / len(active) if len(active) else None,
                 mae=math.fsum(channel_sae[c] / int(counts[c]) for c in active) / len(active) if len(active) else None,
                 n_observed=int(counts.sum()), n_origins=n, num_channels=width, horizon=horizon,
                 active_channels=int(len(active)), metric_scale="A_normalized_original_channels")
    rows = []
    for index, origin in enumerate(origins):
        for channel, channel_id in enumerate(channels):
            for start, stop in horizon_bands(horizon):
                mask = observed[index, start:stop, channel]
                difference = prediction[index, start:stop, channel][mask] - target[index, start:stop, channel][mask]
                rows.append(dict(run_id=str(run_id), split=str(split), cohort=str(cohort), origin=int(origin),
                                 channel=str(channel_id), horizon_start=start, horizon_end=stop,
                                 sse=float(np.sum(difference * difference, dtype=np.float64)),
                                 sae=float(np.sum(np.abs(difference), dtype=np.float64)), count=int(mask.sum())))
    return score, rows


def fit_global_alpha(g_normalized, b_normalized, y_normalized, observed):
    """M_cal-only closed-form clip[-sum w(g-y)(b-g)/sum w(b-g)^2]."""
    g, target, observed = _arrays(g_normalized, y_normalized, observed)
    b, _, _ = _arrays(b_normalized, y_normalized, observed)
    counts = observed.sum(axis=(0, 1), dtype=np.int64)
    active = np.flatnonzero(counts)
    if not len(active):
        return dict(alpha=0.0, numerator=0.0, denominator=0.0, n_observed=0,
                    active_channels=0, fit_status="NO_OBSERVED_TARGETS")
    numerators, denominators, g_objectives, b_objectives = [], [], [], []
    for channel in active:
        mask = observed[:, :, channel]
        e = g[:, :, channel][mask] - target[:, :, channel][mask]
        delta = b[:, :, channel][mask] - g[:, :, channel][mask]
        numerators.append(-float(np.dot(e, delta)) / int(counts[channel]))
        denominators.append(float(np.dot(delta, delta)) / int(counts[channel]))
        g_objectives.append(float(np.dot(e, e)) / int(counts[channel]))
        b_objectives.append(float(np.dot(e + delta, e + delta)) / int(counts[channel]))
    numerator = math.fsum(numerators) / len(active)
    denominator = math.fsum(denominators) / len(active)
    alpha = float(np.clip(numerator / denominator, 0, 1)) if denominator > 0 else 0.0
    g_objective = math.fsum(g_objectives) / len(active)
    return dict(alpha=alpha, numerator=numerator, denominator=denominator,
                n_observed=int(counts.sum()), active_channels=len(active),
                objective_g=g_objective, objective_b=math.fsum(b_objectives) / len(active),
                objective_alpha=g_objective - 2 * alpha * numerator + alpha * alpha * denominator,
                zero_denominator=denominator == 0, fit_status="COMPLETE_VALID",
                channel_counts=[int(value) for value in counts])
