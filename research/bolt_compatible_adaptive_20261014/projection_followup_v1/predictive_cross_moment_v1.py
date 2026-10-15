"""FIT-only spatial directions from centered current/future cross moments."""
import numpy as np


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical_columns(matrix):
    result = matrix.copy()
    for j in range(result.shape[1]):
        if result[np.argmax(np.abs(result[:, j])), j] < 0:
            result[:, j] *= -1
    return result


def fit_directions(history, history_observed, target, observed, k, *, kind):
    x, xm, y, ym = map(np.asarray, (history, history_observed, target, observed))
    require(xm.dtype == ym.dtype == np.bool_ and xm.shape == x.shape and ym.shape == y.shape,
            "MATCHED_BOOL_FIT_OBSERVED_MASKS")
    require(x.dtype == y.dtype == np.float32 and x.ndim == y.ndim == 3
            and x.shape[0] == y.shape[0] and x.shape[2] == y.shape[2]
            and x.shape[1] == 512 and y.shape[1] == 48 and len(x) > 1
            and np.isfinite(x).all() and np.isfinite(y[ym]).all(), "MATCHED_FINITE_FIT_N512C_N48C")
    require(type(k) is int and 1 <= k <= x.shape[-1] and kind in ("PCA", "CROSS_MOMENT"),
            "REGISTERED_INTEGER_K_AND_DIRECTION_KIND")
    flat = x.reshape(-1, x.shape[2]).astype(np.float64)
    center = flat.mean(0)
    centered = flat - center
    covariance = centered.T @ centered / len(flat)
    values, vectors = np.linalg.eigh(covariance)
    require(values[-1] > 0 and np.isfinite(values).all(), "POSITIVE_FIT_HISTORY_VARIANCE")
    retained = np.flatnonzero(values > values[-1] * 1e-8)[::-1]
    require(len(retained) >= k, "FIT_HISTORY_WHITENING_RANK_BELOW_K")
    whitening = canonical_columns(vectors[:, retained]) / np.sqrt(values[retained])[None, :]
    complete = ym.all(axis=2)
    counts = complete.sum(axis=0)
    require((counts >= 2).all(), "EACH_HORIZON_NEEDS_TWO_COMPLETE_C_FIT_RESPONSES")
    current = x[:, -1].astype(np.float64)
    cross = []
    for h in range(48):
        mask = complete[:, h]
        a, b = current[mask], y[mask, h].astype(np.float64)
        cross.append((a-a.mean(0)).T @ (b-b.mean(0)) / len(a))
    cross = np.stack(cross)
    moment = np.concatenate([whitening.T @ c for c in cross], axis=1) / np.sqrt(48.)
    u, singular, _ = np.linalg.svd(moment, full_matrices=False)
    moment_rank = int(np.sum(singular > singular[0] * 1e-8)) if singular[0] > 0 else 0
    if kind == "CROSS_MOMENT":
        require(moment_rank >= k, "CROSS_MOMENT_NUMERIC_RANK_BELOW_K")
        encoder = canonical_columns(whitening @ u[:, :k])
    else:
        encoder = canonical_columns(whitening[:, :k])
    latent_cov = encoder.T @ covariance @ encoder
    require(np.allclose(latent_cov, np.eye(k), atol=1e-7, rtol=0), "FIT_HISTORY_LATENT_COVARIANCE_NOT_IDENTITY")
    center32, encoder32 = center.astype(np.float32), encoder.astype(np.float32)
    require(np.isfinite(center32).all() and np.isfinite(encoder32).all(), "FINITE_DEPLOYMENT_STATE_FP32")
    return {"center": center32, "E": encoder32,
            "center_FP64": center, "encoder_FP64": encoder, "cross_moments_FP64": cross,
            "whitening_FP64": whitening, "singularvalues": singular,
            "metadata": {"kind": kind, "K": k, "C": x.shape[2], "fit_windows": len(x),
                         "whitening_rank": len(retained), "moment_rank": moment_rank,
                         "eigenvalue_relative_cut": 1e-8, "moment_relative_cut": 1e-8,
                         "horizon_complete_C_origins": counts.tolist(),
                         "horizon_equal_squared_moment_weight": 1/48,
                         "latent_history_covariance_FP64": latent_cov.tolist(),
                         "source_current_missing_fraction": float((~xm[:, -1]).mean()),
                         "source_history_missing_fraction": float((~xm).mean()),
                         "history_overlap_positions": len(flat),
                         "future_response_centered_actual_level": True,
                         "current_source_filled_causal": True, "unobserved_targets_used": False,
                         "exact_masked_RRR": False, "TUNE_DEV_used": False,
                         "canonical_sign": "largest absolute coordinate in final E is positive",
                         "self_forecastable_streams_guaranteed": False}}


def encode(history, state):
    x = np.asarray(history)
    require(x.dtype == np.float32 and x.ndim == 3 and np.isfinite(x).all()
            and x.shape[-1] == state["E"].shape[0], "FINITE_FP32_PAST_AND_STATE")
    return np.ascontiguousarray((x-state["center"]) @ state["E"], dtype=np.float32)
