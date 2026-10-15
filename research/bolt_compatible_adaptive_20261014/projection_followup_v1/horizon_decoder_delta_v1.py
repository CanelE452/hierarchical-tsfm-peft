"""Noncentered, observed-response ridge for equal-size static/ramp decoder deltas."""
import numpy as np

ALPHAS = (0., 1e-4, .001, .01, .1, 1., 10., 100., 1000., 10000.)
ARMS = ("STATIC_DELTA", "RAMP_DELTA")


def require(value, message):
    if not value:
        raise ValueError(message)


def gate(length, arm):
    require(type(length) is int and length > 1 and arm in ARMS, "REGISTERED_HORIZON_ARM")
    return np.ones(length, np.float32) if arm == ARMS[0] else np.arange(length, dtype=np.float32) / np.float32(length-1)


def gated_features(features, arm):
    x = np.asarray(features)
    require(x.dtype == np.float32 and x.ndim == 3 and x.size and np.isfinite(x).all(), "FINITE_FP32_NHP_FEATURES")
    return np.ascontiguousarray(x * gate(x.shape[1], arm)[None, :, None], dtype=np.float32)


def feature_scale(ungated_actual_F_features):
    x = np.asarray(ungated_actual_F_features)
    require(x.dtype == np.float32 and x.ndim == 3 and x.size and np.isfinite(x).all(), "ACTUAL_F_FINITE_FP32_UNGATED_FIT_FEATURES")
    rms = np.sqrt(np.mean(x.astype(np.float64).reshape(-1, x.shape[2]) ** 2, axis=0))
    constant = rms == 0
    return np.where(constant, 1., rms), constant


def fit_path(features, response, observed, scale, constant):
    x, y, m = map(np.asarray, (features, response, observed))
    scale, constant = map(np.asarray, (scale, constant))
    require(x.dtype == np.float32 and x.ndim == 3 and x.size and np.isfinite(x).all(), "FINITE_FP32_GATED_FIT_FEATURES")
    require(y.dtype == np.float64 and y.ndim == 3 and y.shape[:2] == x.shape[:2] and m.dtype == np.bool_ and m.shape == y.shape and np.isfinite(y[m]).all(), "ONLY_TRUE_OBSERVED_FP64_FIT_RESIDUALS")
    require(scale.dtype == np.float64 and scale.shape == (x.shape[2],) and np.isfinite(scale).all() and (scale > 0).all() and constant.dtype == np.bool_ and constant.shape == scale.shape, "SHARED_ACTUAL_F_RMS_AND_ZERO_FEATURE_FLAGS")
    n, h, p = x.shape
    c = y.shape[2]
    design = x.astype(np.float64).reshape(n*h, p) / scale
    design[:, constant] = 0
    y, m = y.reshape(n*h, c), m.reshape(n*h, c)
    states = [{"alpha": alpha, "coefficient_FP64": np.zeros((p, c))} for alpha in ALPHAS]
    groups = {}
    for column in range(c):
        groups.setdefault(m[:, column].tobytes(), []).append(column)
    diagnostics = []
    for columns in groups.values():
        valid = m[:, columns[0]]
        count = int(valid.sum())
        a = design[valid]
        target = y[np.ix_(valid, columns)]
        if count:
            u, singular, vt = np.linalg.svd(a, full_matrices=False)
            keep = singular > singular[0]*1e-8 if singular.size and singular[0] > 0 else np.zeros_like(singular, dtype=bool)
            projected = u.T @ target
            for state in states:
                gain = np.zeros_like(singular)
                if state["alpha"] == 0:
                    gain[keep] = 1/singular[keep]
                else:
                    gain = singular/(singular*singular + count*state["alpha"])
                coefficient = ((vt.T * gain) @ projected) / scale[:, None]
                coefficient[constant] = 0
                state["coefficient_FP64"][:, columns] = coefficient
        else:
            singular, keep = np.array([], np.float64), np.array([], bool)
        diagnostics.append({"response_columns": columns, "observed_count": count,
            "observed_mask": valid.copy(), "singularvalues": singular,
            "retained_rank": int(keep.sum()), "fit_executed": bool(count),
            "fallback": "NO_OBSERVED_RESPONSE_ZERO_CORRECTION" if not count else None})
    for state in states:
        state["coefficient_FP32"] = state["coefficient_FP64"].astype(np.float32)
        require(np.isfinite(state["coefficient_FP64"]).all() and np.isfinite(state["coefficient_FP32"]).all(), "FINITE_FUSED_NONCENTERED_DELTA")
    return states, {"groups": diagnostics, "scale_FP64": scale.copy(), "constant_features": constant.copy(),
        "alpha_grid": list(ALPHAS), "intercept_fitted": False, "feature_centering": False,
        "design_singular_rcond": 1e-8, "positive_alpha_uses_all_singular_directions": True,
        "response_macro_objective": "mean_C(observed residual SSE/N_c + alpha*||standardized slope_c||^2)",
        "coefficient_fused_to_original_feature_units": True, "TUNE_DEV_used_for_fit": False,
        "unobserved_response_placeholders_used": False}


def predict(base, ungated_features, state, arm):
    baseline = np.asarray(base)
    x = gated_features(ungated_features, arm)
    coefficient = np.asarray(state["coefficient_FP32"])
    require(baseline.dtype == coefficient.dtype == np.float32 and baseline.ndim == 3 and baseline.shape[:2] == x.shape[:2] and coefficient.shape == (x.shape[2], baseline.shape[2]) and np.isfinite(baseline).all() and np.isfinite(coefficient).all(), "MATCHED_FP32_BASE_FEATURE_DELTA_STATE")
    pieces = []
    for start in range(0, len(x), 32):
        block = x[start:start+32]
        correction = (block.reshape(-1, block.shape[2]) @ coefficient).reshape(*block.shape[:2], baseline.shape[2])
        pieces.append(baseline[start:start+32] + correction)
    result = np.ascontiguousarray(np.concatenate(pieces), dtype=np.float32)
    require(np.isfinite(result).all(), "FINITE_CANONICAL32_DECODER_CORRECTION")
    return result
