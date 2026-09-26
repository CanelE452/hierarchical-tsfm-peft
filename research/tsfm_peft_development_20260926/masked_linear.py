from __future__ import annotations

from typing import Any

import numpy as np


def _as_bool_mask(mask: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.shape != shape:
        raise ValueError(f"mask shape {mask.shape} != target shape {shape}")
    return mask.astype(bool, copy=False)


def _check_xy(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError(f"x and y must be rank-3 arrays, got {x.shape} and {y.shape}")
    if x.shape[0] != y.shape[0] or x.shape[2] != y.shape[2]:
        raise ValueError(f"x [B,L,C] and y [B,H,C] batch/channel mismatch: {x.shape}, {y.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must already be finite/imputed by the caller")
    mask = _as_bool_mask(mask, y.shape)
    if not np.all(np.isfinite(y[mask])):
        raise ValueError("observed target values must be finite")
    return x, y, mask


def _ridge_scalar(examples: np.ndarray, target: np.ndarray, penalty: float, label: str) -> tuple[np.ndarray, float]:
    if len(examples) == 0:
        raise ValueError(f"no observed target examples for {label}")
    if penalty < 0:
        raise ValueError("penalty must be non-negative")
    xmean = examples.mean(axis=0)
    ymean = float(target.mean())
    xc = examples - xmean
    yc = target - ymean
    covariance = xc.T @ xc / len(examples)
    cross = xc.T @ yc / len(examples)
    values, vectors = np.linalg.eigh(covariance)
    inv = 1 / np.maximum(values + penalty, 1e-9)
    weight = (vectors * inv) @ (vectors.T @ cross)
    bias = ymean - float(xmean @ weight)
    return weight, bias


def _examples_for_horizon(x: np.ndarray, y_h: np.ndarray, mask_h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    examples = x.transpose(0, 2, 1).reshape(-1, x.shape[1])
    target = y_h.reshape(-1)
    observed = mask_h.reshape(-1)
    return examples[observed], target[observed]


def fit_masked_shared(x: np.ndarray, y: np.ndarray, mask: np.ndarray, penalty: float) -> dict[str, Any]:
    """Fit a shared temporal affine ridge map with horizon-wise target masking.

    For each horizon h, this solves

        mean_{(b,c): mask[b,h,c]} (y[b,h,c] - x[b,:,c] @ w_h - b_h)^2
        + penalty * ||w_h||^2

    The intercept is not penalized. Missing target entries are never read as zero;
    only observed target examples for that horizon enter the fit. If a horizon has
    no observed target examples, the fit is unavailable and this function raises.
    With an all-true mask, the result matches the dense shared temporal ridge used
    by linear_fulltrain.py.
    """

    x, y, mask = _check_xy(x, y, mask)
    _, context, _ = x.shape
    horizon = y.shape[1]
    weight = np.zeros((context, horizon), dtype=np.float64)
    bias = np.zeros(horizon, dtype=np.float64)
    counts = np.zeros(horizon, dtype=np.int64)
    for h in range(horizon):
        examples_h, target_h = _examples_for_horizon(x, y[:, h, :], mask[:, h, :])
        counts[h] = len(target_h)
        weight[:, h], bias[h] = _ridge_scalar(examples_h, target_h, penalty, f"horizon {h}")
    return {
        "kind": "masked_shared",
        "weight": weight,
        "bias": bias,
        "target_counts": counts,
        "penalty": float(penalty),
        "loss": "per-horizon mean squared loss over observed targets plus penalty*||W_h||^2",
    }


def predict_shared(x: np.ndarray, fit: dict[str, Any]) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"x must be [B,L,C], got {x.shape}")
    weight = np.asarray(fit["weight"], dtype=np.float64)
    bias = np.asarray(fit["bias"], dtype=np.float64)
    if weight.shape[0] != x.shape[1]:
        raise ValueError(f"weight context {weight.shape[0]} != x context {x.shape[1]}")
    return np.einsum("blc,lh->bhc", x, weight) + bias[None, :, None]


def fit_masked_factor(x: np.ndarray, y: np.ndarray, mask: np.ndarray, basis: np.ndarray, penalty: float) -> dict[str, Any]:
    """Fit PCA common/residual temporal ridge maps with full-vector target masking.

    PCA targets require the whole channel vector at a horizon. For each horizon h,
    this uses only origins b where every channel target is observed:

        full_vector_mask[b,h] = all_c mask[b,h,c]

    Common and residual maps then solve the same horizon-wise affine ridge objective
    as fit_masked_shared, using latent PCA components and channel residual components
    as repeated examples. Partially observed future vectors are excluded entirely;
    their observed channels are not used and their missing channels are not treated
    as zero. If any horizon has no full target vector, the factor fit is unavailable
    and this function raises rather than dropping channels.
    """

    x, y, mask = _check_xy(x, y, mask)
    basis = np.asarray(basis, dtype=np.float64)
    if basis.ndim != 2 or basis.shape[0] != x.shape[2]:
        raise ValueError(f"basis must be [C,R] with C={x.shape[2]}, got {basis.shape}")
    z = x @ basis
    residual_x = x - z @ basis.T
    batch, context, channels = x.shape
    horizon = y.shape[1]
    rank = basis.shape[1]
    common_weight = np.zeros((context, horizon), dtype=np.float64)
    common_bias = np.zeros(horizon, dtype=np.float64)
    residual_weight = np.zeros((context, horizon), dtype=np.float64)
    residual_bias = np.zeros(horizon, dtype=np.float64)
    full_origin_counts = np.zeros(horizon, dtype=np.int64)
    common_counts = np.zeros(horizon, dtype=np.int64)
    residual_counts = np.zeros(horizon, dtype=np.int64)

    full_vector_mask = mask.all(axis=2)
    for h in range(horizon):
        origins = full_vector_mask[:, h]
        full_origin_counts[h] = int(origins.sum())
        if full_origin_counts[h] == 0:
            raise ValueError(f"factor unavailable: no fully observed target vector for horizon {h}")
        y_full = y[origins, h, :]
        if y_full.shape != (full_origin_counts[h], channels):
            raise AssertionError("unexpected full-vector target shape")
        zy_h = y_full @ basis
        residual_y_h = y_full - zy_h @ basis.T

        common_examples = z[origins].transpose(0, 2, 1).reshape(-1, context)
        common_target = zy_h.reshape(-1)
        residual_examples = residual_x[origins].transpose(0, 2, 1).reshape(-1, context)
        residual_target = residual_y_h.reshape(-1)
        common_counts[h] = len(common_target)
        residual_counts[h] = len(residual_target)
        common_weight[:, h], common_bias[h] = _ridge_scalar(common_examples, common_target, penalty, f"common horizon {h}")
        residual_weight[:, h], residual_bias[h] = _ridge_scalar(
            residual_examples, residual_target, penalty, f"residual horizon {h}"
        )

    return {
        "kind": "masked_factor",
        "basis": basis.copy(),
        "common": {
            "weight": common_weight,
            "bias": common_bias,
            "target_counts": common_counts,
            "full_vector_origin_counts": full_origin_counts.copy(),
        },
        "residual": {
            "weight": residual_weight,
            "bias": residual_bias,
            "target_counts": residual_counts,
            "full_vector_origin_counts": full_origin_counts.copy(),
        },
        "full_vector_origin_counts": full_origin_counts,
        "penalty": float(penalty),
        "loss": "per-horizon mean squared loss over fully observed PCA/residual targets plus penalty*||W_h||^2",
    }


def predict_factor(x: np.ndarray, fit: dict[str, Any]) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    basis = np.asarray(fit["basis"], dtype=np.float64)
    z = x @ basis
    residual_x = x - z @ basis.T
    common = predict_shared(z, fit["common"]) @ basis.T
    residual = predict_shared(residual_x, fit["residual"])
    return common + residual
