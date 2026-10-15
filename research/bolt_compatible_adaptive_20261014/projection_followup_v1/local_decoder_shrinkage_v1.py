"""Observed-past affine ridge decoder toward a fixed static decoder prior."""
import hashlib
import math

import numpy as np

from local_past_decoder_v1 import LocalPastDecoder

RHOS = (.001, .01, .1, 1., 10.)


def fit_shrunk_local_decoder(history, observed, latent_history, prior_decoder, rho):
    x, mask, z, prior = [np.ascontiguousarray(value) for value in (history, observed, latent_history, prior_decoder)]
    if (x.dtype != np.float32 or x.ndim != 3 or x.shape[1] != 512 or not len(x)
            or not x.shape[2] or not np.isfinite(x).all()):
        raise ValueError("FINITE_FP32_KNOWN_N512C_PAST_REQUIRED")
    if mask.dtype != np.bool_ or mask.shape != x.shape:
        raise ValueError("TRUE_OBSERVED_PAST_MASK_REQUIRED")
    if (z.dtype != np.float32 or z.ndim != 3 or z.shape[:2] != x.shape[:2]
            or not 1 <= z.shape[2] <= x.shape[2] or not np.isfinite(z).all()):
        raise ValueError("FINITE_FP32_KNOWN_N512K_LATENT_PAST_REQUIRED")
    n, _, c = x.shape
    k = z.shape[2]
    if prior.dtype != np.float32 or prior.shape != (k, c) or not np.isfinite(prior).all():
        raise ValueError("FINITE_FIXED_FP32_PRIOR_KC_REQUIRED")
    if not isinstance(rho, (int, float)) or isinstance(rho, bool) or not math.isfinite(rho) or rho <= 0:
        raise ValueError("FIXED_FINITE_POSITIVE_RHO_REQUIRED")
    d = np.broadcast_to(prior.astype(np.float64)[None], (n, k, c)).copy()
    bias = np.zeros((n, c), np.float64)
    counts = mask.sum(axis=1, dtype=np.int64)
    lambdas = np.zeros((n, c), np.float64)
    traces = np.zeros((n, c), np.float64)
    zmean = np.zeros((n, c, k), np.float64)
    ymean = np.zeros((n, c), np.float64)
    past_mse = np.zeros((n, c), np.float64)
    condition = np.zeros((n, c), np.float64)
    stationarity = np.zeros((n, c), np.float64)
    zero_trace = np.zeros((n, c), bool)
    for window in range(n):
        for channel in range(c):
            valid = mask[window, :, channel]
            if counts[window, channel] == 0:
                bias[window, channel] = float(x[window, -1, channel]) - float(z[window, -1].astype(np.float64) @ d[window, :, channel])
                continue
            features = z[window, valid].astype(np.float64)
            target = x[window, valid, channel].astype(np.float64)
            zm, ym = features.mean(axis=0), float(target.mean())
            centered_z, centered_y = features - zm, target - ym
            gram = centered_z.T @ centered_z / len(features)
            linear = centered_z.T @ centered_y / len(features)
            trace = float(np.trace(gram))
            zmean[window, channel], ymean[window, channel] = zm, ym
            traces[window, channel] = trace
            if trace > 0:
                penalty = float(rho) * trace / k
                matrix = gram + penalty * np.eye(k)
                coefficient = d[window, :, channel] + np.linalg.solve(matrix, linear - gram @ d[window, :, channel])
                stationarity[window, channel] = float(np.linalg.norm(gram @ coefficient + penalty * (coefficient - prior[:, channel]) - linear))
                condition[window, channel] = float(np.linalg.cond(matrix))
                lambdas[window, channel] = penalty
                d[window, :, channel] = coefficient
            else:
                zero_trace[window, channel] = True
            bias[window, channel] = ym - float(zm @ d[window, :, channel])
            residual = features @ d[window, :, channel] + bias[window, channel] - target
            past_mse[window, channel] = float(np.mean(residual * residual))
    if not np.isfinite(d).all() or not np.isfinite(bias).all():
        raise ValueError("NONFINITE_SHRUNK_FP64_DECODER")
    with np.errstate(over="ignore", invalid="ignore"):
        d32, b32 = d.astype(np.float32), bias.astype(np.float32)
    if not np.isfinite(d32).all() or not np.isfinite(b32).all():
        raise ValueError("NONFINITE_SHRUNK_FP32_DECODER")
    arrays = {"history": x, "history_observed": mask, "latent_history": z, "prior_decoder": prior}
    metadata = {"route": "OBSERVED_PAST_CENTERED_RIDGE_TOWARD_STATIC_D", "windows": n, "channels": c, "K": k,
        "rho": float(rho), "lambda_rule": "rho*trace(observed_centered_Z.T@Z/n)/K", "slope_singular_cutoff_used": False,
        "true_observed_counts": counts.tolist(), "local_covariance_trace": traces.tolist(), "actual_lambda": lambdas.tolist(),
        "zero_trace_prior_slope_retained": zero_trace.tolist(), "no_observation_prior_with_current_intercept": (counts == 0).tolist(),
        "observed_latent_means_per_C": zmean.tolist(), "observed_raw_C_means": ymean.tolist(),
        "observed_real_past_reconstruction_MSE": past_mse.tolist(), "past_metric_valid": (counts > 0).tolist(),
        "ridge_matrix_condition": condition.tolist(), "ridge_matrix_condition_valid": (lambdas > 0).tolist(),
        "real_stationarity_residual_norm": stationarity.tolist(),
        "max_abs_decoder_per_window": np.max(np.abs(d32.astype(np.float64)), axis=(1, 2)).tolist(),
        "decoder_Frobenius_norm_per_window": np.linalg.norm(d32.astype(np.float64), axis=(1, 2)).tolist(),
        "decoder_operator_norm_per_window": [float(np.linalg.svd(matrix.astype(np.float64), compute_uv=False)[0]) for matrix in d32],
        "fit_input_bindings": {name: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(value.tobytes()).hexdigest()} for name, value in arrays.items()},
        "future_labels_used_in_decoder_fit": False, "absolute_decoder_gain_bound_claimed": False,
        "intercept_analytic_unpenalized": True, "decoder_interface_same_as_static": False,
        "native_head_frozen_intervention_not_reoptimized": True, "strict_K_only_compression": False,
        "inference_time_past_decoder_refitting": True, "scientific_IO_in_module": False}
    return LocalPastDecoder(d, np.ascontiguousarray(d32), bias, np.ascontiguousarray(b32), metadata)
