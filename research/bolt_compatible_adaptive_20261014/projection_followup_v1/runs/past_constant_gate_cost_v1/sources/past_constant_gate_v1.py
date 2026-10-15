"""Deterministic output intervention using only the known causal-filled past."""
import numpy as np


LENGTH = 512
HORIZON = 48
MIN_OBSERVED = 2


def past_constant_gate(history, history_observed):
    x, observed = np.asarray(history), np.asarray(history_observed)
    if (x.dtype != np.float32 or x.ndim != 3 or x.shape[1] != LENGTH
            or not len(x) or not x.shape[2] or not np.isfinite(x).all()):
        raise ValueError("FINITE_NONEMPTY_FP32_N512C_CAUSAL_FILLED_PAST_REQUIRED")
    if observed.dtype != np.bool_ or observed.shape != x.shape:
        raise ValueError("EXACT_BOOL_TRUE_OBSERVED_PAST_MASK_REQUIRED")
    flat = np.all(x == x[:, -1:, :], axis=1)
    counts = observed.sum(axis=1, dtype=np.int64)
    keep_forecast = ~(flat & (counts >= MIN_OBSERVED))
    return {"keep_forecast": np.ascontiguousarray(keep_forecast),
            "exact_flat": np.ascontiguousarray(flat),
            "true_observed_counts": np.ascontiguousarray(counts)}


def apply_output_gate(prediction, raw_last, keep_forecast):
    prediction, current, keep = np.asarray(prediction), np.asarray(raw_last), np.asarray(keep_forecast)
    if (prediction.dtype != np.float32 or prediction.ndim != 3 or prediction.shape[1] != HORIZON
            or not len(prediction) or not prediction.shape[2] or not np.isfinite(prediction).all()):
        raise ValueError("FINITE_NONEMPTY_FP32_N48C_PREDICTION_REQUIRED")
    if (current.dtype != np.float32 or current.shape != (len(prediction), prediction.shape[2])
            or not np.isfinite(current).all()):
        raise ValueError("FINITE_FP32_CURRENT_C_REQUIRED")
    if keep.dtype != np.bool_ or keep.shape != current.shape:
        raise ValueError("EXACT_BOOL_NC_KEEP_FORECAST_GATE_REQUIRED")
    return np.ascontiguousarray(np.where(keep[:, None, :], prediction, current[:, None, :]), dtype=np.float32)
