"""Past-only closed-form R4 controls; no data/model loading at import time.

The F-swap head shares 3 x 48 parameters across latent streams. With P/E/D
frozen, original-channel predictions are affine in these 144 parameters.
The caller streams chunks into sufficient statistics and chooses alpha/ZERO
using B only. This module never reads targets during deployment prediction.
"""
from __future__ import annotations

import numpy as np
import torch

LAGS = (0, 1, 2, 4, 8, 16, 32)
ALPHAS = (0.0, 1e-4, 1e-2, 1.0)
H = 48


def features(z: torch.Tensor) -> torch.Tensor:
    """Three shared latent-past features [B,3,r], centered at current latent."""
    last = z[:, -1]
    return torch.stack((z[:, -8:].mean(1) - last,
                        z[:, -64:].mean(1) - last,
                        z.mean(1) - last), dim=1)


def swap_future(z: torch.Tensor, weight: torch.Tensor, horizon=H) -> torch.Tensor:
    weight = torch.as_tensor(weight, dtype=z.dtype, device=z.device).reshape(3, horizon)
    return z[:, -1:, :] + torch.einsum("bpr,ph->bhr", features(z), weight)


def swap_predict(z: torch.Tensor, decoder: torch.Tensor, weight: torch.Tensor,
                 current: torch.Tensor | None = None, lags=LAGS, horizon=H) -> torch.Tensor:
    """Efficient deployed original-channel correction, optionally with anchor.

    ``current`` is original X's last row [B,C] or [B,1,C]. Omitting it returns
    only the decoder's correction. No future original/latent value is input.
    """
    future = swap_future(z, weight, horizon=horizon)
    joined = torch.cat((z, future), dim=1)
    length = z.shape[1]
    lagged = torch.cat([
        joined[:, length-lag:length-lag+horizon] - z[:, -1-lag, None]
        for lag in lags
    ], dim=-1)
    result = lagged @ decoder
    if current is not None:
        result = result + (current[:, None] if current.ndim == 2 else current)
    return result


def swap_design(z: torch.Tensor, decoder: torch.Tensor, lags=LAGS, horizon=H):
    """Return offset[B,H,C] and chunk-local design[B,H,C,3*H].

    Original prediction equals ``current_C + offset + design @ W.reshape(-1)``.
    The offset includes lag-reference/past-prefix terms but excludes current C.
    W flattening is feature-major, then future horizon. Never concatenate this
    chunk design across the complete training origin set.
    """
    batch, length, rank = z.shape
    if length <= max(lags) or decoder.shape[0] != len(lags) * rank:
        raise ValueError("Latent history and decoder do not match origin-reference lags")
    channel = decoder.shape[1]
    latent_features = features(z)
    last = z[:, -1]
    offset = z.new_zeros(batch, horizon, channel)
    design = z.new_zeros(batch, horizon, channel, 3, horizon)
    for lag_index, lag in enumerate(lags):
        matrix = decoder[lag_index*rank:(lag_index+1)*rank]
        reference = z[:, -1-lag]
        if lag:
            prefix_horizon = min(lag, horizon)
            prefix = z[:, length-lag:length-lag+prefix_horizon]
            offset[:, :prefix_horizon] += (prefix - reference[:, None]) @ matrix
        if lag < horizon:
            offset[:, lag:] += ((last - reference) @ matrix)[:, None]
            coefficients = torch.einsum("bpr,rc->bcp", latent_features, matrix)
            # A forecast at output h uses W[:,h-lag] for this lag branch.
            active = design[:, lag:, :, :, :horizon-lag]
            active.diagonal(dim1=1, dim2=4).add_(coefficients[..., None])
    return offset, design.reshape(batch, horizon, channel, 3*horizon)


def _numpy64(value) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def sufficient_update(gram, rhs, design, target, mask, counts):
    """Accumulate one F-swap chunk in FP64, with global observed-C weights.

    ``target`` is Y-current_C-offset. ``counts[C]`` must be the global A
    observed-target counts on the exact fitted origin schedule, not per-chunk
    counts. Each scalar target receives 1/count[C]; no additional horizon or
    channel scaling is applied. Gram/rhs are mutated and returned.
    """
    x, y, valid, counts = _numpy64(design), _numpy64(target), np.asarray(
        mask.detach().cpu().numpy() if torch.is_tensor(mask) else mask, dtype=bool
    ), _numpy64(counts)
    if x.shape[:-1] != y.shape or y.shape != valid.shape or counts.shape != (y.shape[-1],):
        raise ValueError("Design/target/mask/global-count dimensions differ")
    if np.any(counts < 0) or not np.isfinite(x).all() or not np.isfinite(y[valid]).all():
        raise ValueError("Non-finite design/observed target or negative count")
    weight = valid / np.where(counts > 0, counts, 1)[None, None, :]
    weight *= (counts > 0)[None, None, :]
    xf = x.reshape(-1, x.shape[-1])
    wf = weight.reshape(-1)
    yf = np.where(valid, y, 0).reshape(-1)
    gram += xf.T @ (xf * wf[:, None])
    rhs += xf.T @ (wf * yf)
    return gram, rhs


def fit_from_sufficient(gram, rhs, alpha: float):
    """FP64 ridge or minimum-norm normal-equation LS and numeric diagnostics.

    A rank-deficient normal equation is explicitly reported. Solving it is not
    proof that the representation/decoder has a unique solution or that an
    Adam comparison converged. The caller records unresolved numeric cases.
    """
    matrix, vector = _numpy64(gram), _numpy64(rhs)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or vector.shape != (len(matrix),):
        raise ValueError("Sufficient-statistic dimensions differ")
    if alpha < 0 or not np.isfinite(matrix).all() or not np.isfinite(vector).all():
        raise ValueError("Non-finite sufficient statistics or negative alpha")
    matrix = (matrix + matrix.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    scale = max(float(np.max(np.abs(eigenvalues), initial=0)), 1.0)
    tolerance = np.finfo(np.float64).eps * max(matrix.shape) * scale
    if eigenvalues.min(initial=0) < -100*tolerance:
        raise ValueError("Gram matrix has materially negative eigenvalues")
    positive = eigenvalues > tolerance
    projection = eigenvectors.T @ vector
    if alpha == 0:
        inverse = np.divide(1.0, eigenvalues, out=np.zeros_like(eigenvalues), where=positive)
    else:
        inverse = 1.0 / (np.maximum(eigenvalues, 0) + alpha)
    weight = eigenvectors @ (inverse * projection)
    residual = (matrix + alpha*np.eye(len(matrix))) @ weight - vector
    rank = int(positive.sum())
    condition = float(eigenvalues[-1] / eigenvalues[positive][0]) if rank else None
    relative = float(np.linalg.norm(residual) / max(np.linalg.norm(vector), 1e-30))
    return weight, {
        "alpha": float(alpha), "parameters": len(weight), "rank": rank,
        "full_rank": rank == len(weight), "positive_subspace_condition": condition,
        "eigenvalue_min": float(eigenvalues.min()),
        "eigenvalue_max": float(eigenvalues.max(initial=0)),
        "rank_tolerance": float(tolerance),
        "normal_equation_relative_residual": relative,
        "finite_solution": bool(np.isfinite(weight).all()),
        "solver": "FP64_eigh_ridge_or_normal_equation_pseudoinverse",
        "numeric_status": "PASS_NORMAL_EQUATIONS" if np.isfinite(weight).all() and relative < 1e-6 else "NUMERIC_UNRESOLVED",
        "rank_note": "Rank-deficient alpha0 is minimum-norm on retained Gram eigenspace; normal equations square design conditioning.",
    }


def direct_features(x: torch.Tensor) -> torch.Tensor:
    """Self-channel 32 contiguous 16-point block means plus current [B,C,33].

    Every feature is centered at current C; consequently the last/current
    feature is identically zero. It is retained to match the prescribed
    33-feature contract, with rank deficiency reported by the LS diagnostic.
    """
    if x.shape[1] != 512:
        raise ValueError("Direct baseline requires original-grid L512 past")
    mean_blocks = x.reshape(x.shape[0], 32, 16, x.shape[2]).mean(2)
    last = x[:, -1:]
    return torch.cat((mean_blocks, last), dim=1).sub(last).transpose(1, 2)


def direct_sufficient_update(gram, rhs, past_features, target_delta, mask, counts):
    """Per-C, per-horizon sufficient statistics for independent 33-feature LS.

    Shapes: gram[C,H,33,33], rhs[C,H,33], features[B,C,33], target/mask[B,H,C].
    Target is observed normalized Y-current_C; global observed-C weighting is
    identical to the swap fit. Updates never form a complete training design.
    """
    x, y, counts = _numpy64(past_features), _numpy64(target_delta), _numpy64(counts)
    valid = np.asarray(mask.detach().cpu().numpy() if torch.is_tensor(mask) else mask, dtype=bool)
    if x.ndim != 3 or y.ndim != 3 or y.shape != valid.shape or x.shape[:2] != (len(y), y.shape[2]):
        raise ValueError("Direct features/target/mask dimensions differ")
    if not np.isfinite(x).all() or not np.isfinite(y[valid]).all():
        raise ValueError("Non-finite direct feature/observed target")
    weight = valid / np.where(counts > 0, counts, 1)[None, None, :]
    weight *= (counts > 0)[None, None, :]
    safe = np.where(valid, y, 0)
    gram += np.einsum("bci,bhc,bcj->chij", x, weight, x, optimize=True)
    rhs += np.einsum("bci,bhc,bhc->chi", x, weight, safe, optimize=True)
    return gram, rhs


def fit_direct_from_sufficient(gram, rhs, alpha: float):
    matrix, vector = _numpy64(gram), _numpy64(rhs)
    if matrix.ndim != 4 or vector.shape != matrix.shape[:3] or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("Direct sufficient-statistic dimensions differ")
    channels, horizon, width = vector.shape
    weight = np.empty((channels, width, horizon), dtype=np.float64)
    diagnostics = []
    for channel in range(channels):
        for step in range(horizon):
            fitted, diagnostic = fit_from_sufficient(matrix[channel, step], vector[channel, step], alpha)
            weight[channel, :, step] = fitted
            diagnostics.append(diagnostic)
    return weight, {
        "alpha": float(alpha), "parameters": int(weight.size),
        "designated_feature_count": width, "horizon": horizon, "channels": channels,
        "min_rank": min(item["rank"] for item in diagnostics),
        "max_rank": max(item["rank"] for item in diagnostics),
        "numeric_unresolved_equations": sum(item["numeric_status"] != "PASS_NORMAL_EQUATIONS" for item in diagnostics),
        "max_normal_equation_relative_residual": max(item["normal_equation_relative_residual"] for item in diagnostics),
        "solver": "independent_C_H_FP64_eigh_ridge_or_normal_equation_pseudoinverse",
        "diagnostics": diagnostics,
    }


def direct_predict(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = torch.as_tensor(weight, dtype=x.dtype, device=x.device)
    return x[:, -1:, :] + torch.einsum("bcf,cfh->bhc", direct_features(x), weight)
