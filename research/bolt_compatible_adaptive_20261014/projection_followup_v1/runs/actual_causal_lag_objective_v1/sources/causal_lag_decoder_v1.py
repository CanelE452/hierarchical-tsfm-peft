"""Pure fixed causal-lag D fitting, deployment and joint-horizon diagnostics."""
import hashlib

import numpy as np

RCOND = 1e-4
JOINT_RCOND = 1e-15
RECON_START = 32


def array_binding(value):
    value = np.asarray(value)
    return {"shape": list(value.shape), "dtype": str(value.dtype),
        "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}


def checked_lags(lags, history_length):
    values = list(lags)
    if (not values or any(type(v) not in (int, np.int32, np.int64) for v in values)
            or values != sorted(set(values)) or values[0] < 0 or values[-1] >= history_length):
        raise ValueError("ASCENDING_DISTINCT_NONNEGATIVE_LAGS_INSIDE_KNOWN_PAST_REQUIRED")
    return tuple(int(v) for v in values)


def latent_pair(history, future):
    x, f = np.asarray(history), np.asarray(future)
    if (x.dtype not in (np.float32, np.float64) or f.dtype != x.dtype or x.ndim != 3 or f.ndim != 3
            or not x.size or not f.size or x.shape[0] != f.shape[0] or x.shape[2] != f.shape[2]
            or not np.isfinite(x).all() or not np.isfinite(f).all()):
        raise ValueError("SAME_DTYPE_FINITE_NLK_NHK_LATENT_PAIR_REQUIRED")
    return x, f


def lagged_delta_features(history, future, lags):
    x, f = latent_pair(history, future)
    lags = checked_lags(lags, x.shape[1])
    n, h, k = f.shape
    features = np.empty((n, h, len(lags), k), dtype=x.dtype)
    for index, lag in enumerate(lags):
        baseline = x[:, -1-lag]
        for step in range(1, h+1):
            relative = step-lag
            value = f[:, relative-1] if relative > 0 else x[:, -1+relative]
            features[:, step-1, index] = value-baseline
    if not np.isfinite(features).all(): raise ValueError("NONFINITE_CAUSAL_LAG_DELTA_FEATURES")
    return features.reshape(n, h, len(lags)*k)


def past_level_features(history, lags, start=RECON_START):
    x = np.asarray(history)
    if x.dtype != np.float32 or x.ndim != 3 or not x.size or not np.isfinite(x).all():
        raise ValueError("FINITE_FP32_NLK_KNOWN_PAST_REQUIRED")
    lags = checked_lags(lags, x.shape[1])
    if type(start) is not int or not lags[-1] <= start < x.shape[1]:
        raise ValueError("COMMON_RECONSTRUCTION_START_MUST_COVER_ALL_LAGS")
    return np.concatenate([x[:, start-lag:x.shape[1]-lag] for lag in lags], axis=2)


def fit_response_columns(features, responses, observed, centered):
    if (features.ndim != 2 or responses.ndim != 2 or features.dtype != np.float64 or responses.dtype != np.float64
            or features.shape[0] != responses.shape[0] or observed.dtype != np.bool_ or observed.shape != responses.shape
            or not np.isfinite(features).all() or not np.isfinite(responses[observed]).all()):
        raise ValueError("FINITE_FP64_DESIGN_AND_TRUE_OBSERVED_RESPONSE_COLUMNS_REQUIRED")
    p, c = features.shape[1], responses.shape[1]
    counts = observed.sum(axis=0, dtype=np.int64)
    decoder, bias = np.zeros((p, c)), np.zeros(c)
    metadata = {key: [] for key in ("rank", "singularvalues", "retained_condition", "direct_residual_MSE",
        "fallback_reason", "fit_executed", "rank_deficient_slope", "full_slope_identified",
        "constant_observed_response", "all_predictor_entries_zero", "zero_centered_predictor_variance")}
    for channel, count in enumerate(counts):
        if count == 0:
            values = (0, [], None, None, "NO_OBSERVED_RESPONSE_UNLEARNED_ZERO_SLOPE", False, None, False, None, None, None)
        else:
            a, y = features[observed[:, channel]], responses[observed[:, channel], channel]
            mean_a, mean_y = a.mean(axis=0), float(y.mean())
            design, target = (a-mean_a, y-mean_y) if centered else (a, y)
            slope, _, rank, singular = np.linalg.lstsq(design, target, rcond=RCOND)
            decoder[:, channel] = slope
            bias[channel] = mean_y-mean_a@slope if centered else 0.
            residual = a@slope+bias[channel]-y
            zero = bool(np.all(design == 0))
            fallback = None
            if centered and count == 1: fallback = "ONE_OBSERVATION_CENTERED_SLOPE_UNIDENTIFIED_MEAN_INTERCEPT"
            elif zero: fallback = "ZERO_DESIGN_MINIMUM_NORM_ZERO_SLOPE"+("_MEAN_INTERCEPT" if centered else "_NO_INTERCEPT")
            values = (int(rank), singular.tolist(), float(singular[0]/singular[rank-1]) if rank else None,
                float(np.mean(residual*residual)), fallback, True, bool(rank < p), bool(rank == p),
                bool(np.all(y == y[0])), bool(np.all(a == 0)), bool(np.all(a-mean_a == 0)))
        for key, value in zip(metadata, values): metadata[key].append(value)
    with np.errstate(over="ignore", invalid="ignore"):
        decoder32, bias32 = decoder.astype(np.float32), bias.astype(np.float32)
    if not all(np.isfinite(value).all() for value in (decoder, bias, decoder32, bias32)):
        raise ValueError("NONFINITE_CAUSAL_LAG_DECODER_STATE")
    metadata.update(true_observed_counts=counts.tolist(), no_observed_response=(counts == 0).tolist(),
        one_observed_response=(counts == 1).tolist(), singular_relative_cutoff=RCOND, centered_design=bool(centered),
        intercept_fitted=bool(centered), slope_parameters_per_C=p,
        constant_response_is_normal_LS_fit=True, noncentered_one_observation_not_forced_to_zero=True)
    return {"decoder_FP64": decoder, "decoder_FP32": decoder32, "bias_FP64": bias, "bias_FP32": bias32, "metadata": metadata}


def fitted_result(fitted, lags, objective, bindings, **scope):
    fitted["lags"] = np.asarray(lags, dtype=np.int64)
    fitted["metadata"].update(objective=objective, lags=list(lags), feature_order="lag-major then latent K",
        input_bindings=bindings, state_bindings={name: array_binding(value) for name, value in fitted.items() if name != "metadata"},
        deployment="FP32 lag-major deltafeatures @ D32 + currentC; reconstruction bias cancels",
        per_origin_fit=False, TUNE_DEV_used_for_fit=False, fit_time_measured_inside_pure=False,
        independent_generalization_claimed=False, **scope)
    return fitted


def fit_reconstruction(history, observed, latent_history, lags):
    x, mask, z = np.asarray(history), np.asarray(observed), np.asarray(latent_history)
    if (x.dtype != np.float32 or x.ndim != 3 or not x.size or x.shape[1] != 512 or not np.isfinite(x).all()
            or mask.dtype != np.bool_ or mask.shape != x.shape or z.dtype != np.float32 or z.ndim != 3
            or z.shape[:2] != x.shape[:2] or not z.shape[2] or not np.isfinite(z).all()):
        raise ValueError("MATCHED_FINITE_FILLED_FP32_N512C_N512K_AND_TRUE_OBSERVED_MASK_REQUIRED")
    lags = checked_lags(lags, 512)
    features = past_level_features(z, lags, RECON_START)
    target, response_mask = x[:, RECON_START:], mask[:, RECON_START:]
    fitted = fit_response_columns(features.reshape(-1, features.shape[2]).astype(np.float64),
        target.reshape(-1, x.shape[2]).astype(np.float64), response_mask.reshape(-1, x.shape[2]), True)
    return fitted_result(fitted, lags, "RECON", {"history": array_binding(x), "observed": array_binding(mask),
        "latent_history": array_binding(z), "actual_level_features_FP32": array_binding(features),
        "response_observed_common_slice": array_binding(response_mask)}, common_history_slice=[32, 512],
        reconstruction_positions_per_window=480, future_FIT_Y_used_as_supervision=False,
        response_scope="FIT true-observed C past levels, centered per response; filled gaps are predictors only")


def fit_forecast(history, future, current, target, observed, lags):
    z, f = latent_pair(history, future)
    current, target, mask = np.asarray(current), np.asarray(target), np.asarray(observed)
    if (z.dtype != np.float32 or z.shape[1] != 512 or current.dtype != np.float32 or current.ndim != 2
            or current.shape[0] != len(z) or not current.shape[1] or not np.isfinite(current).all()
            or target.dtype != np.float32 or target.shape != (*f.shape[:2], current.shape[1])
            or mask.dtype != np.bool_ or mask.shape != target.shape or not np.isfinite(target[mask]).all()):
        raise ValueError("ACTUAL_FP32_F_PREDICTION_CURRENT_C_AND_OBSERVED_FIT_Y_REQUIRED")
    lags = checked_lags(lags, z.shape[1])
    features = lagged_delta_features(z, f, lags)
    innovation = np.zeros(target.shape, np.float64)
    anchored = np.broadcast_to(current.astype(np.float64)[:, None], target.shape)
    innovation[mask] = target[mask].astype(np.float64)-anchored[mask]
    fitted = fit_response_columns(features.reshape(-1, features.shape[2]).astype(np.float64),
        innovation.reshape(-1, current.shape[1]), mask.reshape(-1, current.shape[1]), False)
    return fitted_result(fitted, lags, "FORECAST", {"latent_history": array_binding(z), "actual_F_latent_future": array_binding(f),
        "current_C": array_binding(current), "FIT_target_Y": array_binding(target), "FIT_target_observed": array_binding(mask),
        "actual_deployment_delta_features_FP32": array_binding(features), "FIT_Y_minus_currentC_FP64": array_binding(innovation)},
        future_FIT_Y_used_as_supervision=True, future_Y_used_as_deployment_input=False,
        training_horizons=f.shape[1], intercept_in_forecast=False,
        predictor_numeric_contract="Actual FP32 deployment subtraction/features, cast to FP64 for LS",
        target_numeric_contract="Y32 interpreted as FP64 minus currentC32 interpreted as FP64; no intermediate FP32 subtraction",
        unobserved_innovation_entries_filled_zero=True, unobserved_target_values_used_in_regression_or_statistics=False,
        response_scope="FIT true-observed C future innovation; noncentered direct LS without intercept")


def deploy_inputs(history, future, current, decoder, lags):
    z, f = latent_pair(history, future)
    current, d = np.asarray(current), np.asarray(decoder)
    lags = checked_lags(lags, z.shape[1])
    if (z.dtype != np.float32 or current.dtype != np.float32 or d.dtype != np.float32 or d.ndim != 2
            or d.shape[0] != len(lags)*z.shape[2] or not d.shape[1] or current.shape != (len(z), d.shape[1])
            or not np.isfinite(current).all() or not np.isfinite(d).all()):
        raise ValueError("REGISTERED_FP32_CURRENT_C_AND_LAG_MAJOR_DECODER_REQUIRED")
    return z, f, current, d, lags


def predict(history, future, current, decoder, lags):
    z, f, current, d, lags = deploy_inputs(history, future, current, decoder, lags)
    result = current[:, None]+lagged_delta_features(z, f, lags)@d
    if not np.isfinite(result).all(): raise ValueError("NONFINITE_FP32_CAUSAL_LAG_FORECAST")
    return result


def real_trace(history, future, current, decoder, lags):
    z, f, current, d, lags = deploy_inputs(history, future, current, decoder, lags)
    return current.astype(np.float64)[:, None]+lagged_delta_features(z.astype(np.float64), f.astype(np.float64), lags)@d.astype(np.float64)


def joint_real_map(history, current, decoder, lags, H=48):
    z, current, d = np.asarray(history), np.asarray(current), np.asarray(decoder)
    if type(H) is not int or H < 1: raise ValueError("POSITIVE_INTEGER_JOINT_HORIZON_REQUIRED")
    empty_future = np.zeros((len(z), H, z.shape[2]), np.float32) if z.ndim == 3 else np.empty(0, np.float32)
    z, _, current, d, lags = deploy_inputs(z, empty_future, current, d, lags)
    n, _, k = z.shape; c = d.shape[1]
    z64, d64 = z.astype(np.float64), d.astype(np.float64)
    A = np.zeros((H*c, H*k), np.float64)
    offset = np.broadcast_to(current.astype(np.float64)[:, None], (n, H, c)).copy()
    for index, lag in enumerate(lags):
        block = d64[index*k:(index+1)*k]
        offset -= (z64[:, -1-lag]@block)[:, None]
        for h in range(1, H+1):
            j = h-lag
            if j > 0: A[(h-1)*c:h*c, (j-1)*k:j*k] += block.T
            else: offset[:, h-1] += z64[:, -1+j]@block
    return {"A": A, "offset": offset, "metadata": {"unknown_coordinates": "absolute future z1..zH, horizon-major then K",
        "output_coordinates": "horizon-major then C", "lags": list(lags), "H": H, "K": k, "C": c,
        "A_shape": list(A.shape), "known_prefix_and_negative_lag_baselines_in_offset": True,
        "all_HK_unknowns_retained_before_output_mask": True, "floating_trace_bit_identity_claimed": False}}


def masked_joint_minimum(history, current, decoder, lags, target, observed):
    target, mask = np.asarray(target), np.asarray(observed)
    if (target.dtype != np.float32 or target.ndim != 3 or not target.size or mask.dtype != np.bool_
            or mask.shape != target.shape or not np.isfinite(target[mask]).all()):
        raise ValueError("OBSERVED_FP32_FUTURE_Y_FOR_DIAGNOSTIC_ONLY_REQUIRED")
    mapping = joint_real_map(history, current, decoder, lags, H=target.shape[1])
    A, offset = mapping["A"], mapping["offset"]
    if offset.shape != target.shape: raise ValueError("JOINT_DIAGNOSTIC_TARGET_MODEL_SHAPE_MISMATCH")
    n, H, c = target.shape; k = A.shape[1]//H
    complete = mask.all(axis=2)
    target64, b = target.astype(np.float64).reshape(n, -1), offset.reshape(n, -1)
    optimal, prediction = np.zeros((n, H*k)), b.copy()
    patterns, ranks, certificates = {}, np.zeros(n, np.int64), []
    for origin in range(n): patterns.setdefault(complete[origin].tobytes(), []).append(origin)
    SSE = 0.; nonempty_svd = 0
    for indices in patterns.values():
        rows = np.repeat(complete[indices[0]], c)
        if not rows.any():
            certificates.append({"origins_zero_based": indices, "complete_positions_per_origin": 0,
                "selected_output_rows": 0, "unknown_columns_kept": H*k, "numeric_rank": 0, "singularvalues": [],
                "normal_equation_max_abs": None, "retained_left_space_residual_max_abs": None}); continue
        design = A[rows]
        U, singular, Vt = np.linalg.svd(design, full_matrices=False)
        retained = singular > singular[0]*JOINT_RCOND if len(singular) and singular[0] > 0 else np.zeros(len(singular), bool)
        rank = int(retained.sum()); ranks[indices] = rank; nonempty_svd += 1
        rhs = target64[np.ix_(indices, rows)]-b[np.ix_(indices, rows)]
        inverse = (Vt[retained].T/singular[retained])@U[:, retained].T if rank else np.zeros((H*k, int(rows.sum())))
        fitted = rhs@inverse.T
        optimal[indices] = fitted; prediction[indices] = fitted@A.T+b[indices]
        residual = prediction[np.ix_(indices, rows)]-target64[np.ix_(indices, rows)]
        SSE += float(np.sum(residual*residual, dtype=np.float64))
        certificates.append({"origins_zero_based": indices, "complete_positions_per_origin": int(complete[indices[0]].sum()),
            "selected_output_rows": int(rows.sum()), "unknown_columns_kept": H*k, "numeric_rank": rank,
            "singularvalues": singular.tolist(), "normal_equation_max_abs": float(np.max(np.abs(residual@design))),
            "retained_left_space_residual_max_abs": float(np.max(np.abs(residual@U[:, retained]))) if rank else 0.})
    count = int(complete.sum()); denominator = count*c
    return {"prediction_FP64": prediction.reshape(n, H, c), "optimal_future_FP64": optimal.reshape(n, H, k),
        "complete_horizons": complete, "minimum_normalized_MSE": SSE/denominator if denominator else None,
        "metadata": {"status": "COMPUTED_DIAGNOSTIC_ONLY" if denominator else "NO_COMPLETE_C_TARGET_POSITIONS",
            **mapping["metadata"], "rcond": JOINT_RCOND, "numeric_rank_per_origin": ranks.tolist(),
            "complete_target_positions": count, "complete_target_positions_per_origin": complete.sum(axis=1).tolist(),
            "C_times_complete_positions_denominator": denominator, "SSE_complete_C": SSE,
            "mask_patterns": len(patterns), "nonempty_mask_pattern_SVDs": nonempty_svd, "certificates": certificates,
            "window_MSEs_reaveraged": False, "unknown_latents_deleted_at_missing_output_horizon": False,
            "minimum_only_normalized_MSE_in_real_numeric_joint_A_space": True,
            "minimum_space_definition": "Registered retained numeric left-singular space of each masked A; singular value > largest*1e-15",
            "strict_untruncated_real_rowspace_minimum_claimed": False,
            "all_real_latent_forecasts_satisfy_registered_bound_claimed": False,
            "MAE_or_physical_MSE_minimum_claimed": False, "strict_all_FP32_output_lower_bound_claimed": False,
            "future_labels_for_diagnostic_only": True, "future_labels_used_to_fit_D_or_deploy": False,
            "old_pointwise_or_global_rankK_floor_reused": False}}
