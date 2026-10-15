"""Independent lagged decoder LS/forecast/joint-space verification; CPU artifacts only."""
from pathlib import Path
import argparse
import ast
import json
import sys

import audit_projection_results_v1 as prior
import audit_raw_subset_comparison_v3 as raw_audit

HERE, ROOT = prior.HERE, prior.ROOT
require, exact = prior.require, raw_audit.exact
np = sys.modules.get("numpy")
DATASETS = raw_audit.DATASETS
SPACES = tuple(raw_audit.SPACES)
CONDITIONS = {"L0_RECON": ((0,), "RECON"), "L7_RECON": ((0, 1, 2, 4, 8, 16, 32), "RECON"), "L0_FORECAST": ((0,), "FORECAST"), "L7_FORECAST": ((0, 1, 2, 4, 8, 16, 32), "FORECAST")}
RCOND, SPACE_RCOND = 1e-4, 1e-15
PARENT_SHA = {"raw_subset_diagnostic": "2b271ba9aa6ae359033444b282ddc1becc5d1f60ef063451171e9c93f70db4ff", "raw_subset_audit": "e177652ac34414bccfbfc564da820fca0bde5a462922ceb0edaf04c47081e95d", "parent_report": "49130355afe1d89e43d130b416a536c6f683f36a76ab0d4e844afe11ec1a652b", "parent_report_manifest": "a6d8a837a543712402224d742fcc7c5e08b9eba1333968f6e53baa56a8c45e36", "parent_report_audit": "a184351bcbc503c5308cd3cb74232fb470136f7cdcc8711cc3ac20972c86cb00"}


def initialize_numpy():
    global np
    raw_audit.initialize_numpy(); np = sys.modules["numpy"]


def validate_lags(lags):
    require(isinstance(lags, (list, tuple)) and len(lags) > 0 and lags[0] == 0 and all(type(value) is int and 0 <= value <= 32 for value in lags) and list(lags) == sorted(set(lags)), "ORDERED_UNIQUE_REGISTERED_CAUSAL_LAGS")


def delta_features(history, future, lags, *, real=False):
    validate_lags(lags)
    require(history.dtype == future.dtype == np.float32 and history.ndim == future.ndim == 3 and history.shape[0] == future.shape[0] and history.shape[-1] == future.shape[-1] and history.shape[1] > max(lags) and future.shape[1] > 0 and np.isfinite(history).all() and np.isfinite(future).all(), "FINITE_MATCHED_FP32_CAUSAL_LATENT_HISTORY_AND_FUTURE")
    n, horizon, k = future.shape
    dtype = np.float64 if real else np.float32
    out = np.empty((n, horizon, len(lags) * k), dtype=dtype)
    for hi in range(horizon):
        h = hi + 1
        for li, lag in enumerate(lags):
            s = h - lag
            value = future[:, s - 1] if s >= 1 else history[:, history.shape[1] - 1 + s]
            out[:, hi, li * k:(li + 1) * k] = value.astype(dtype) - history[:, -1 - lag].astype(dtype)
    return out


def anchored_forecast(history, future, current, decoder, lags, *, real=False):
    features = delta_features(history, future, lags, real=real)
    require(current.dtype == decoder.dtype == np.float32 and current.ndim == decoder.ndim == 2 and current.shape[0] == history.shape[0] and decoder.shape == (features.shape[-1], current.shape[-1]) and np.isfinite(current).all() and np.isfinite(decoder).all(), "FINITE_FIXED_LAG_DECODER_AND_CURRENT_C")
    dtype = np.float64 if real else np.float32
    return current.astype(dtype)[:, None] + features @ decoder.astype(dtype)


def independent_fit(history, observed, latent, future, current, target, future_observed, lags, objective):
    validate_lags(lags)
    require(objective in ("RECON", "FORECAST") and history.dtype == latent.dtype == np.float32 and observed.dtype == future_observed.dtype == np.bool_ and history.shape == observed.shape and history.shape[:2] == latent.shape[:2] and history.shape[1] == 512 and target.shape == future_observed.shape and current.shape == history[:, -1].shape, "FIXED_FIT_HISTORY_RESPONSE_AND_OBJECTIVE_CONTRACT")
    c, k = history.shape[-1], latent.shape[-1]
    if objective == "RECON":
        features = np.concatenate([latent[:, 32 - lag:512 - lag] for lag in lags], axis=2).reshape(-1, len(lags) * k).astype(np.float64)
        responses = history[:, 32:].reshape(-1, c).astype(np.float64); mask = observed[:, 32:].reshape(-1, c)
    else:
        features = delta_features(latent, future, lags).reshape(-1, len(lags) * k).astype(np.float64)
        innovations = np.zeros(target.shape, np.float64)
        innovations[future_observed] = target[future_observed].astype(np.float64) - np.broadcast_to(current.astype(np.float64)[:, None], target.shape)[future_observed]
        responses = innovations.reshape(-1, c); mask = future_observed.reshape(-1, c)
    require(np.isfinite(features).all() and np.isfinite(responses[mask]).all(), "FINITE_FIT_PREDICTORS_AND_TRUE_OBSERVED_RESPONSES")
    p = features.shape[-1]; d = np.zeros((p, c), np.float64); b = np.zeros(c, np.float64)
    certificates, counts, ranks, spectra, errors, means_z, means_y = [], mask.sum(axis=0), [], [], [], [], []
    for channel in range(c):
        active = mask[:, channel]; x, y = features[active], responses[active, channel]
        if not active.any():
            ranks.append(0); spectra.append([]); errors.append(None); means_z.append([0.] * p); means_y.append(0.)
            certificates.append({"observed_count": 0, "fit_executed": False, "normal_certificate": None, "minimum_norm_null_component": None}); continue
        mean_x = x.mean(axis=0) if objective == "RECON" else np.zeros(p)
        mean_y = float(y.mean()) if objective == "RECON" else 0.
        design, response = x - mean_x, y - mean_y
        u, singular, vh = np.linalg.svd(design, full_matrices=False)
        keep = singular > singular[0] * RCOND if singular.size else np.zeros(0, bool)
        rank = int(keep.sum())
        slope = vh[keep].T @ ((u[:, keep].T @ response) / singular[keep]) if rank else np.zeros(p)
        direct, _, direct_rank, _ = np.linalg.lstsq(design, response, rcond=RCOND)
        require(rank == int(direct_rank), "CAUSAL_D_INDEPENDENT_SVD_DIRECT_LS_RANK")
        prior.close(slope, direct, "causal_D_SVD_direct_LS", atol=1e-10, rtol=1e-10)
        d[:, channel], b[channel] = slope, mean_y - mean_x @ slope
        residual = design @ slope - response
        normal = vh[keep] @ (design.T @ residual)
        null = (np.eye(p) - vh[keep].T @ vh[keep]) @ slope
        normal_tolerance = 1e-10 * (1 + float(np.linalg.norm(design)) * float(np.linalg.norm(response)))
        require(not rank or float(np.abs(normal).max()) <= normal_tolerance, "CAUSAL_D_RETAINED_NORMAL_EQUATIONS")
        require(float(np.abs(null).max()) <= 1e-10 * (1 + float(np.linalg.norm(slope))), "CAUSAL_D_MINIMUM_NORM_RETAINED_NULLSPACE")
        ranks.append(rank); spectra.append(singular.tolist()); errors.append(float(np.mean((x @ slope + b[channel] - y) ** 2))); means_z.append(mean_x.tolist()); means_y.append(mean_y)
        certificates.append({"observed_count": int(counts[channel]), "fit_executed": True, "normal_certificate": float(np.abs(normal).max()) if rank else 0., "normal_tolerance": normal_tolerance, "minimum_norm_null_component": float(np.abs(null).max()), "one_observation_has_identifiable_noncentered_direction": len(x) == 1 and objective == "FORECAST" and rank > 0})
    return {"decoder_FP64": d, "decoder_FP32": d.astype(np.float32), "bias_FP64": b, "bias_FP32": b.astype(np.float32), "true_observed_counts": counts.tolist(), "retained_rank": ranks, "singularvalues": spectra, "response_MSE_FP64": errors, "predictor_means": means_z, "response_means": means_y, "certificates": certificates, "features_FP64": features, "responses_FP64": responses, "response_mask": mask}


def joint_absolute_map(history, current, decoder, lags, horizon):
    validate_lags(lags)
    n, _, k = history.shape; c = current.shape[-1]
    require(history.dtype == current.dtype == decoder.dtype == np.float32 and decoder.shape == (len(lags) * k, c) and current.shape[0] == n and horizon > 0, "JOINT_REAL_D32_ABSOLUTE_FUTURE_MAP_CONTRACT")
    a = np.zeros((horizon * c, horizon * k), np.float64)
    offset = np.broadcast_to(current.astype(np.float64)[:, None], (n, horizon, c)).copy()
    for hi in range(horizon):
        h = hi + 1
        for li, lag in enumerate(lags):
            block = decoder[li * k:(li + 1) * k].astype(np.float64)
            offset[:, hi] -= history[:, -1 - lag].astype(np.float64) @ block
            s = h - lag
            if s <= 0: offset[:, hi] += history[:, history.shape[1] - 1 + s].astype(np.float64) @ block
            else: a[hi * c:(hi + 1) * c, (s - 1) * k:s * k] += block.T
    return a, offset


def exact_zero_reduced_LS(matrix, response):
    active = np.any(matrix != 0., axis=0)
    coefficients = np.zeros((matrix.shape[1], response.shape[1]), np.float64)
    if active.any():
        reduced, _, rank, singular = np.linalg.lstsq(matrix[:, active], response, rcond=SPACE_RCOND)
        coefficients[active] = reduced
    else: rank, singular = 0, np.empty(0, np.float64)
    require(np.all(matrix[:, ~active] == 0.) and np.all(coefficients[~active] == 0.), "EXACT_ZERO_COLUMN_PROOF_AND_ZERO_REINSERTION")
    certificate = {"full_unknown_columns": matrix.shape[1], "nonzero_columns": int(active.sum()), "exact_zero_column_indices": np.flatnonzero(~active).tolist(), "exact_zero_column_proof_max_abs": 0., "reduced_matrix_shape": [matrix.shape[0], int(active.sum())], "reduced_LS_numeric_rank": int(rank), "reduced_LS_singularvalues": singular.tolist(), "full_HK_solution_reinserted": True, "missing_output_horizons_deleted_as_a_rule": False, "rcond_unchanged": SPACE_RCOND}
    return coefficients, int(rank), certificate


def joint_minimum(values, history, decoder, lags):
    target, observed, current = values["target"], values["observed"], values["raw_last"]
    n, horizon, c = target.shape; k = history.shape[-1]
    complete_positions = observed.all(axis=2); mask = np.broadcast_to(complete_positions[..., None], observed.shape)
    if not complete_positions.any(): return {"status": "NO_COMPLETE_C_TARGET_POSITIONS", "RAW_bound_applied": False}
    a, offset = joint_absolute_map(history, current, decoder, lags, horizon)
    prediction = offset.copy(); best_future = np.zeros((n, horizon * k), np.float64)
    certificates, retained_groups = [], []; groups = {}
    for index in range(n):
        if complete_positions[index].any(): groups.setdefault(complete_positions[index].tobytes(), []).append(index)
    total_sse = 0.
    for indices in groups.values():
        rows = np.repeat(complete_positions[indices[0]], c); matrix = a[rows]
        response = (target.astype(np.float64) - offset)[indices].reshape(len(indices), horizon * c)[:, rows].T
        coefficients, rank, reduction = exact_zero_reduced_LS(matrix, response)
        u, s, _ = np.linalg.svd(matrix, full_matrices=False); keep = s > s[0] * SPACE_RCOND if s.size else np.zeros(0, bool)
        require(int(rank) == int(keep.sum()), "JOINT_EXACT_ZERO_REDUCED_LS_FULL_SVD_NUMERIC_RANK")
        projection = u[:, keep] @ (u[:, keep].T @ response)
        fitted = matrix @ coefficients
        prior.close(fitted, projection, "joint_map_independent_LS_output_projection", atol=1e-10, rtol=1e-10)
        residual = response - fitted; normal = matrix.T @ residual
        tolerance = 1e-10 * (1 + float(np.linalg.norm(matrix)) * float(np.linalg.norm(response)))
        require(float(np.abs(normal).max()) <= tolerance, "JOINT_OUTPUT_NORMAL_CERTIFICATE")
        total_sse += float(np.square(residual).sum()); best_future[indices] = coefficients.T
        prediction[indices] = offset[indices] + (a @ coefficients).T.reshape(len(indices), horizon, c)
        certificates.append({"window_indices": indices, "complete_horizons": complete_positions[indices[0]].tolist(), "selected_output_rows": int(rows.sum()), "unknown_future_coordinates_kept": horizon * k, "numeric_rank": int(rank), "singularvalues": s.tolist(), "normal_residual_max_abs": float(np.abs(normal).max()), "normal_tolerance": tolerance, "exact_zero_reduced_LS": reduction})
        retained_groups.append({"window_indices": indices, "output_rows": rows, "retained_left_basis": u[:, keep]})
    scores = prior.scores(prediction, target, mask, values["std"])
    minimum = total_sse / (int(complete_positions.sum()) * c)
    prior.close(scores["normalized"]["mse"], minimum, "joint_complete_position_weighted_denominator", atol=1e-10, rtol=1e-10)
    return {"status": "COMPUTED_DIAGNOSTIC_ONLY", "minimum_normalized_MSE_real_joint_D32": minimum, "minimum_solution_scores_complete_C": scores, "complete_target_positions": int(complete_positions.sum()), "complete_horizons_per_origin": complete_positions.sum(axis=1).tolist(), "joint_map": a, "known_offset": offset, "minimum_solution_prediction": prediction, "best_absolute_future_latent": best_future.reshape(n, horizon, k), "certificates": certificates, "retained_groups": retained_groups, "minimum_only_normalized_MSE": True, "RAW_bound_applied": False, "old_static_or_global_rankK_floor_reused": False, "strict_all_FP32_output_lower_bound_claimed": False}


def diagnostic_trace(prediction, future, values, history, decoder, lags, minimum):
    if minimum["status"] == "NO_COMPLETE_C_TARGET_POSITIONS": return {"status": minimum["status"]}
    real = anchored_forecast(history, future, values["raw_last"], decoder, lags, real=True)
    a, offset = joint_absolute_map(history, values["raw_last"], decoder, lags, future.shape[1])
    mapped = offset + (future.astype(np.float64).reshape(len(future), -1) @ a.T).reshape(real.shape)
    prior.close(real, mapped, "independent_absolute_joint_A_offset_vs_real_direct_forecast", atol=1e-10, rtol=1e-10)
    mask = prior.oracle_mask(values["observed"])
    actual = prior.scores(prediction, values["target"], mask, values["std"]); real_scores = prior.scores(real, values["target"], mask, values["std"])
    retained_prediction = np.full(real.shape, np.nan, np.float64).reshape(len(real), -1)
    flat_offset, real_flat = offset.reshape(len(real), -1), real.reshape(len(real), -1)
    for group in minimum["retained_groups"]:
        indices, rows, u = group["window_indices"], group["output_rows"], group["retained_left_basis"]
        innovation = (real_flat - flat_offset)[np.ix_(indices, rows)]
        retained_prediction[np.ix_(indices, rows)] = flat_offset[np.ix_(indices, rows)] + (innovation @ u) @ u.T
    retained_prediction = retained_prediction.reshape(real.shape)
    retained_score = prior.scores(retained_prediction, values["target"], mask, values["std"])
    discarded_full = np.zeros(real.shape, np.float64); discarded_full[mask] = (real - retained_prediction)[mask]
    discarded_output = discarded_full[mask]
    rounding = abs(actual["normalized"]["mse"] - real_scores["normalized"]["mse"])
    floor = minimum["minimum_normalized_MSE_real_joint_D32"]
    tolerance = 1e-10 * max(1., abs(floor), abs(real_scores["normalized"]["mse"]), abs(retained_score["normalized"]["mse"]))
    require(retained_score["normalized"]["mse"] + tolerance >= floor, "NEW_JOINT_BOUND_ONLY_ON_REGISTERED_RETAINED_OUTPUT_TRACE")
    return {"scores_complete_C": actual, "real_trace_scores_complete_C": real_scores, "retained_output_real_trace_scores_complete_C": retained_score, "discarded_numeric_map_output_squared_norm": float(np.square(discarded_output).sum()), "discarded_numeric_map_output_max_abs": float(np.abs(discarded_output).max()), "measured_FP32_rounding_normalized_MSE_allowance": rounding, "actual_FP32_minus_minimum": actual["normalized"]["mse"] - floor, "real_trace_minus_minimum": real_scores["normalized"]["mse"] - floor, "numerical_MSE_tolerance": tolerance, "retained_output_trace_bound_inequality_pass": True, "real_trace_bound_inequality_pass": bool(real_scores["normalized"]["mse"] + tolerance >= floor), "FP32_bound_with_measured_allowance_pass": bool(actual["normalized"]["mse"] + rounding + tolerance >= floor), "all_real_latent_forecasts_bound_claimed": False, "_retained_prediction_FP64": retained_prediction, "_discarded_output_FP64": discarded_full}


def raw_binding(value):
    return raw_audit.raw_binding(value)


def check_state(state, metadata, values, latent, future, lags, objective):
    require(set(state) == {"decoder_FP64", "decoder_FP32", "bias_FP64", "bias_FP32", "lags"}, "FIVE_EXACT_CAUSAL_DECODER_STATE_ARRAYS")
    exact(state["lags"], np.asarray(lags, np.int64), "registered_state_lags")
    require(metadata["objective"] == objective and metadata["lags"] == list(lags) and metadata["feature_order"] == "lag-major then latent K" and metadata["singular_relative_cutoff"] == RCOND, "CAUSAL_D_OBJECTIVE_ORDER_AND_REGISTERED_CUTOFF")
    require(metadata["centered_design"] == metadata["intercept_fitted"] == (objective == "RECON") and metadata["per_origin_fit"] is metadata["TUNE_DEV_used_for_fit"] is metadata["independent_generalization_claimed"] is False, "CAUSAL_D_FIT_INFORMATION_AND_INTERCEPT_SCOPE")
    expected = independent_fit(values["history"], values["history_observed"], latent, future, values["raw_last"], values["target"], values["observed"], lags, objective)
    for name in state:
        require(metadata["state_bindings"][name] == raw_binding(state[name]), "EXACT_CAUSAL_D_STATE_ARRAY_SHA:" + name)
        if name.endswith("FP64"): prior.close(state[name], expected[name], "independent_causal_D:" + name, atol=1e-10, rtol=1e-10)
        elif name.endswith("FP32"): exact(state[name], state[name.replace("FP32", "FP64")].astype(np.float32), "exact_causal_D_deployment_cast:" + name)
    if objective == "RECON":
        level_features = np.concatenate([latent[:, 32 - lag:512 - lag] for lag in lags], axis=2)
        bindings = {"history": raw_binding(values["history"]), "observed": raw_binding(values["history_observed"]), "latent_history": raw_binding(latent), "actual_level_features_FP32": raw_binding(level_features), "response_observed_common_slice": raw_binding(values["history_observed"][:, 32:])}
        require(metadata["common_history_slice"] == [32, 512] and metadata["reconstruction_positions_per_window"] == 480 and metadata["future_FIT_Y_used_as_supervision"] is False, "COMMON_FIT_RECON480_NOT_OLD512")
    else:
        bindings = {"latent_history": raw_binding(latent), "actual_F_latent_future": raw_binding(future), "current_C": raw_binding(values["raw_last"]), "FIT_target_Y": raw_binding(values["target"]), "FIT_target_observed": raw_binding(values["observed"]), "actual_deployment_delta_features_FP32": raw_binding(delta_features(latent, future, lags)), "FIT_Y_minus_currentC_FP64": raw_binding(expected["responses_FP64"].reshape(values["target"].shape))}
        require(metadata["future_FIT_Y_used_as_supervision"] is True and metadata["future_Y_used_as_deployment_input"] is metadata["intercept_in_forecast"] is False and metadata["training_horizons"] == future.shape[1], "FORECAST_FIT_FUTURE_TARGET_NOT_DEPLOYMENT_INPUT")
        require(metadata["unobserved_innovation_entries_filled_zero"] is True and metadata["unobserved_target_values_used_in_regression_or_statistics"] is False, "UNOBSERVED_TARGET_PAYLOADS_NOT_USED_IN_FIT_STATISTICS")
    require(metadata["input_bindings"] == bindings, "ALL_TRUE_FIT_INPUT_TARGET_AND_FEATURE_BINDINGS")
    features, responses, mask = expected["features_FP64"], expected["responses_FP64"], expected["response_mask"]
    counts, p = mask.sum(axis=0), features.shape[1]
    flags = {name: [] for name in ("fallback_reason", "fit_executed", "rank_deficient_slope", "full_slope_identified", "constant_observed_response", "all_predictor_entries_zero", "zero_centered_predictor_variance")}
    conditions = []
    for channel, count in enumerate(counts):
        if not count: fields = ("NO_OBSERVED_RESPONSE_UNLEARNED_ZERO_SLOPE", False, None, False, None, None, None); conditions.append(None)
        else:
            x, y = features[mask[:, channel]], responses[mask[:, channel], channel]
            centered = x - x.mean(axis=0); design = centered if objective == "RECON" else x
            zero = bool(np.all(design == 0)); rank = expected["retained_rank"][channel]
            fallback = "ONE_OBSERVATION_CENTERED_SLOPE_UNIDENTIFIED_MEAN_INTERCEPT" if objective == "RECON" and count == 1 else "ZERO_DESIGN_MINIMUM_NORM_ZERO_SLOPE" + ("_MEAN_INTERCEPT" if objective == "RECON" else "_NO_INTERCEPT") if zero else None
            fields = (fallback, True, bool(rank < p), bool(rank == p), bool(np.all(y == y[0])), bool(np.all(x == 0)), bool(np.all(centered == 0)))
            singular = expected["singularvalues"][channel]; conditions.append(singular[0] / singular[rank - 1] if rank else None)
        for key, value in zip(flags, fields): flags[key].append(value)
    exact_values = {**flags, "true_observed_counts": counts.tolist(), "no_observed_response": (counts == 0).tolist(), "one_observed_response": (counts == 1).tolist(), "rank": expected["retained_rank"], "slope_parameters_per_C": p, "constant_response_is_normal_LS_fit": True, "noncentered_one_observation_not_forced_to_zero": True}
    for key, value in exact_values.items(): require(metadata[key] == value, "FIT_D_EXACT_FLAGS_COUNTS_OR_RANK:" + key)
    for key, value in (("singularvalues", expected["singularvalues"]), ("direct_residual_MSE", expected["response_MSE_FP64"]), ("retained_condition", conditions)):
        for recorded, calculated in zip(metadata[key], value):
            if calculated is None: require(recorded is None, "NO_OBSERVED_FIT_METRIC_NONE")
            else: prior.close(recorded, calculated, "causal_D_fit_metadata:" + key, atol=1e-10, rtol=1e-10)
    return {"objective": objective, "lags": list(lags), "observed_counts": counts.tolist(), "retained_rank": expected["retained_rank"], "certificates": expected["certificates"], "independent_SVD_and_direct_LS": True, "TUNE_DEV_fitting_or_new_training_executed": False}


def verified_metadata(record, check):
    check(); path = Path(record["path"])
    path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    require(path.is_file() and prior.file_sha(path) == record["sha256"], "EXACT_METADATA_FILE_SHA")
    if "bytes" in record: require(path.stat().st_size == record["bytes"], "EXACT_METADATA_FILE_BYTES")
    if path.suffix.lower() == ".json": return prior.verified_json({**record, "path": str(path)}, check)
    require(path.suffix.lower() in (".md", ".py"), "JSON_OR_HASHED_TEXT_METADATA_REQUIRED")
    return None


def load_gates(path, check):
    path = Path(path).resolve(); binding = {"path": str(path), "sha256": prior.file_sha(path)}
    result = prior.verified_json(binding, check)
    if result.get("status") not in ("COMPLETE", "PARTIAL_COMPLETED"): return binding, result, {}
    parents = {}
    for key, digest in PARENT_SHA.items():
        require(result[key]["sha256"] == digest, "EXACT_ACCEPTED_PARENT_SHA:" + key)
        parents[key] = verified_metadata(result[key], check)
    source, audit, report_audit = (parents[key] for key in ("raw_subset_diagnostic", "raw_subset_audit", "parent_report_audit"))
    require(source["status"] == "COMPLETE" and audit["status"] == report_audit["status"] == "PASS" and report_audit["files_checked"] == 32 and report_audit["total_CSV_primitive_fields_checked"] == 171848 and report_audit["local_links_checked"] == 47, "EXACT_SCIENCE_AND_PARENT_REPORT_PASS")
    require(audit["diagnostic"] == result["raw_subset_diagnostic"] and report_audit["report"] == result["parent_report"] and report_audit["manifest"] == result["parent_report_manifest"], "PARENT_JSON_REPORT_AND_MANIFEST_BINDINGS")
    require(source["F_before"] == source["F_after"] == result["frozen_F_parent_signatures"]["before"] == result["frozen_F_parent_signatures"]["after"], "PARENT_F_HASH_PROVENANCE_WITHOUT_NEW_F_LOAD")
    require(result["new_F_calls"] == result["new_native_head_training"] == 0, "NO_NEW_F_OR_NATIVE_TRAINING")
    for key in ("new_backbone_training", "protected_TEST_accessed", "independent_holdout", "oracle_used_for_fit_or_selection", "TUNE_DEV_used_for_fit_or_selection", "strict_K_only_compression", "new_representation_learning_claimed", "whole_deployment_cost_measured", "new_F_object_or_state_hash_computed", "individual_exact_human_method_approval_claimed"):
        require(result[key] is False, "REGISTERED_CAUSAL_DECODER_SCOPE:" + key)
    return binding, result, parents


def score_checked(recorded, prediction, values, mask, label):
    if not mask.any():
        require(recorded == {"status": "NO_OBSERVED_TARGETS", "normalized": None, "physical_per_channel_diagnostic": None}, "NO_OBSERVATIONS_EXPLICIT_NO_SCORE")
        return recorded, 0
    expected = prior.scores(prediction, values["target"], mask, values["std"])
    return expected, prior.compare_scores(recorded, expected, label)


def check_joint_diagnostic(entry, arrays, values, latent, decoder, lags, minimum=None):
    minimum = joint_minimum(values, latent, decoder, lags) if minimum is None else minimum
    floor = entry["diagnostics"]["joint_floor"]
    if minimum["status"] == "NO_COMPLETE_C_TARGET_POSITIONS":
        require(floor["status"] == "SKIPPED_NO_COMPLETE_C_TARGETS", "NO_OBSERVED_JOINT_FLOOR_NOT_FABRICATED")
        return {"status": minimum["status"]}, 0
    require(floor["status"] == "COMPLETE" and floor["all_HK_future_variables_retained"] is floor["complete_C_output_rows_only"] is True and floor["RAW_bound_inequality_applied"] is False, "NEW_JOINT_HORIZON_MINIMUM_SCOPE")
    metadata = floor["metadata"]; n, horizon, c = values["target"].shape; k = latent.shape[-1]
    complete = values["observed"].all(axis=2); mask = prior.oracle_mask(values["observed"])
    require(metadata["unknown_coordinates"] == "absolute future z1..zH, horizon-major then K" and metadata["rcond"] == SPACE_RCOND and metadata["A_shape"] == [horizon * c, horizon * k] and metadata["all_HK_unknowns_retained_before_output_mask"] is True, "JOINT_HK_UNKNOWN_COUNT_NOT_HJK")
    for key in ("window_MSEs_reaveraged", "unknown_latents_deleted_at_missing_output_horizon", "MAE_or_physical_MSE_minimum_claimed", "strict_all_FP32_output_lower_bound_claimed", "future_labels_used_to_fit_D_or_deploy", "old_pointwise_or_global_rankK_floor_reused", "strict_untruncated_real_rowspace_minimum_claimed", "all_real_latent_forecasts_satisfy_registered_bound_claimed"):
        require(metadata[key] is False, "JOINT_DIAGNOSTIC_INFORMATION_OR_DENOMINATOR_SCOPE:" + key)
    require(metadata["complete_target_positions"] == int(complete.sum()) and metadata["complete_target_positions_per_origin"] == complete.sum(axis=1).tolist() and metadata["C_times_complete_positions_denominator"] == int(complete.sum()) * c and metadata["mask_patterns"] == len({row.tobytes() for row in complete}), "JOINT_EXACT_MASK_POSITION_DENOMINATOR")
    prior.close(floor["minimum_normalized_MSE"], minimum["minimum_normalized_MSE_real_joint_D32"], "independent_joint_floor_MSE", atol=1e-10, rtol=1e-10)
    prior.close(arrays["joint_minimum_prediction_FP64"], minimum["minimum_solution_prediction"], "independent_joint_SVD_vs_LS_min_prediction", atol=1e-10, rtol=1e-10)
    a, offset = minimum["joint_map"], minimum["known_offset"]
    require(arrays["joint_optimal_future_FP64"].dtype == np.float64 and arrays["joint_optimal_future_FP64"].shape == (n, horizon, k), "ALL_HK_SAVED_ORACLE_COORDINATES")
    mapped = offset + (arrays["joint_optimal_future_FP64"].reshape(n, -1) @ a.T).reshape(n, horizon, c)
    prior.close(arrays["joint_minimum_prediction_FP64"], mapped, "saved_optimal_absolute_future_A_plus_known_offset", atol=1e-10, rtol=1e-10)
    score, checks = score_checked(floor["minimum_solution_scores_complete_mask"], arrays["joint_minimum_prediction_FP64"], values, mask, "saved_joint_minimum")
    prior.close(score["normalized"]["mse"], minimum["minimum_normalized_MSE_real_joint_D32"], "saved_solution_vs_independent_minimum", atol=1e-10, rtol=1e-10)
    prior.close(metadata["SSE_complete_C"], score["normalized"]["mse"] * int(complete.sum()) * c, "joint_SSE_and_global_weighting", atol=1e-10, rtol=1e-10)
    ranks = [0] * n
    by_windows = {tuple(cert["window_indices"]): cert for cert in minimum["certificates"]}
    for certificate in metadata["certificates"]:
        indices = certificate["origins_zero_based"]; require(certificate["unknown_columns_kept"] == horizon * k and certificate["selected_output_rows"] == int(complete[indices[0]].sum()) * c, "ORACLE_GROUP_ALL_UNKNOWN_COLUMNS_AND_OBSERVED_OUTPUT_ROWS")
        if not complete[indices[0]].any():
            require(certificate["numeric_rank"] == 0 and certificate["normal_equation_max_abs"] is None, "EMPTY_ORACLE_PATTERN_NOT_MEASURED"); continue
        expected = by_windows[tuple(indices)]
        require(certificate["numeric_rank"] == expected["numeric_rank"], "MASK_GROUP_SVD_LS_RANK")
        prior.close(certificate["singularvalues"], expected["singularvalues"], "joint_group_singularvalues", atol=1e-10, rtol=1e-10)
        for index in indices: ranks[index] = expected["numeric_rank"]
        rowmask = np.repeat(complete[indices[0]], c); design = a[rowmask]
        residual = (arrays["joint_minimum_prediction_FP64"] - values["target"].astype(np.float64))[indices].reshape(len(indices), -1)[:, rowmask]
        normal = float(np.abs(residual @ design).max())
        prior.close(certificate["normal_equation_max_abs"], normal, "saved_joint_normal_certificate", atol=1e-10, rtol=1e-10)
        require(normal <= expected["normal_tolerance"], "INDEPENDENT_SAVED_JOINT_NORMAL_RESIDUAL")
    require(metadata["numeric_rank_per_origin"] == ranks and metadata["nonempty_mask_pattern_SVDs"] == len(minimum["certificates"]), "JOINT_RANKS_AND_REUSED_MASK_FACTORIZATIONS")
    return {"status": "COMPUTED_DIAGNOSTIC_ONLY", "minimum_normalized_MSE_real_joint_D32": minimum["minimum_normalized_MSE_real_joint_D32"], "minimum_solution_scores_complete_C": minimum["minimum_solution_scores_complete_C"], "complete_target_positions": minimum["complete_target_positions"], "joint_unknown_future_coordinates": horizon * k, "joint_output_coordinates_before_mask": horizon * c, "lag_slope_parameters_per_C": len(lags) * k, "rank_upper_bound_same_HK_for_L0_and_L7": horizon * k, "certificates": minimum["certificates"], "minimum_only_normalized_MSE": True, "RAW_bound_applied": False, "old_static_or_global_rankK_floor_reused": False}, checks


def audit_split(entry, arrays, values, latent, future, encoder, decoder, lags):
    require(entry["status"] == "COMPLETE" and entry["windows"] == len(latent), "SAVED_COMPLETE_CACHED_D_SPLIT")
    complete = values["observed"].all(axis=2); complete_mask = prior.oracle_mask(values["observed"])
    require(entry["complete_target_positions"] == int(complete.sum()) and entry["target_positions"] == complete.size and entry["EY_incomplete_C_horizons_filled_with_currentC"] == int((~complete).sum()), "EXACT_COMPLETE_HORIZONS_AND_EY_IMPUTATION_COUNTS")
    base = {"target": values["target"], "observed": values["observed"], "currentC": values["raw_last"], "origins": values["origins"], "FIT_mean": values["mean"], "FIT_std": values["std"], "complete_horizons": complete, "latent_history": latent, "actual_F_future": future}
    flat = np.ascontiguousarray(np.broadcast_to(latent[:, -1:], future.shape))
    ey_input = np.where(complete[..., None], values["target"], values["raw_last"][:, None]).astype(np.float32)
    ey_future = raw_audit.encode(ey_input, encoder, "BOLT_NUMPY")
    base.update(flat_future=flat, EY_future=ey_future)
    methods = {"ACTUAL_F": future, "FLAT": flat, "EY": ey_future}
    expected_keys = set(base) | {method + suffix for method in methods for suffix in ("_prediction_FP32", "_real_trace_FP64")}
    if complete.any(): expected_keys |= {"joint_minimum_prediction_FP64", "joint_optimal_future_FP64", "joint_A_FP64", "joint_known_offset_FP64"} | {method + suffix for method in methods for suffix in ("_retained_joint_trace_FP64", "_discarded_joint_component_FP64")}
    require(set(arrays) == expected_keys, "EXACT_CACHED_PREDICTION_TRACE_AND_JOINT_ARTIFACT_KEYS")
    for key, value in base.items(): exact(arrays[key], value, "saved_causal_split_source:" + key)
    minimum = joint_minimum(values, latent, decoder, lags)
    if complete.any():
        prior.close(arrays["joint_A_FP64"], minimum["joint_map"], "independent_saved_joint_block_matrix", atol=1e-10, rtol=1e-10)
        prior.close(arrays["joint_known_offset_FP64"], minimum["known_offset"], "independent_saved_known_prefix_offset", atol=1e-10, rtol=1e-10)
    diagnostic, score_fields = check_joint_diagnostic(entry, arrays, values, latent, decoder, lags, minimum)
    output = {"joint_floor": diagnostic, "methods": {}, "current_C_persistence": {}, "RAW_BOLT": {}}
    require(set(entry["diagnostics"]["methods"]) == set(methods), "EXACT_ACTUAL_FLAT_AND_EY_DIAGNOSTIC_METHODS")
    for method, forecast in methods.items():
        predicted = anchored_forecast(latent, forecast, values["raw_last"], decoder, lags)
        real = anchored_forecast(latent, forecast, values["raw_last"], decoder, lags, real=True)
        exact(arrays[method + "_prediction_FP32"], predicted, "independent_FP32_lag_major_cached_prediction:" + method)
        prior.close(arrays[method + "_real_trace_FP64"], real, "independent_lag_real_trace:" + method, atol=1e-10, rtol=1e-10)
        recorded = entry["diagnostics"]["methods"][method]; out = {}
        for key, mask, p in (("scores_all_observed", values["observed"], predicted), ("scores_complete_mask", complete_mask, predicted), ("real_trace_scores_complete_mask", complete_mask, real)):
            out[key], count = score_checked(recorded[key], p, values, mask, "causal_method:" + method + ":" + key); score_fields += count
        if complete.any():
            trace = diagnostic_trace(predicted, forecast, values, latent, decoder, lags, minimum)
            fields = {"real_trace_minus_minimum_normalized_MSE": trace["real_trace_minus_minimum"], "measured_FP32_rounding_normalized_MSE_allowance": trace["measured_FP32_rounding_normalized_MSE_allowance"], "numerical_MSE_tolerance": trace["numerical_MSE_tolerance"], "discarded_numeric_map_output_squared_norm": trace["discarded_numeric_map_output_squared_norm"], "discarded_numeric_map_output_max_abs": trace["discarded_numeric_map_output_max_abs"]}
            for key, value in fields.items(): prior.close(recorded[key], value, "causal_method_trace_metadata:" + key, atol=1e-10, rtol=1e-10)
            for key in ("real_trace_bound_inequality_pass", "FP32_bound_with_measured_allowance_pass", "retained_output_trace_bound_inequality_pass"): require(recorded[key] == trace[key], "RETAINED_VS_UNIVERSAL_BOUND_BOOLEAN:" + key)
            score_fields += prior.compare_scores(recorded["retained_output_real_trace_scores_complete_mask"], trace["retained_output_real_trace_scores_complete_C"], "independent_retained_joint_trace")
            retained = arrays[method + "_retained_joint_trace_FP64"]
            require(np.isnan(retained[~complete_mask]).all(), "RETAINED_TRACE_UNOBSERVED_ROWS_EXPLICIT_NAN_NOT_SCORED")
            prior.close(retained[complete_mask], trace["_retained_prediction_FP64"][complete_mask], "independent_retained_output_trace_array", atol=1e-10, rtol=1e-10)
            prior.close(arrays[method + "_discarded_joint_component_FP64"], trace["_discarded_output_FP64"], "independent_discarded_numeric_component", atol=1e-10, rtol=1e-10)
            out.update({key: value for key, value in trace.items() if not key.startswith("_")})
        output["methods"][method] = out
    require(entry["scores_all_observed"] == entry["diagnostics"]["methods"]["ACTUAL_F"]["scores_all_observed"] and entry["scores_complete_mask"] == entry["diagnostics"]["methods"]["ACTUAL_F"]["scores_complete_mask"], "ACTUAL_F_TOP_SCORE_ALIAS")
    for method, prediction in (("current_C_persistence", np.broadcast_to(values["raw_last"][:, None], values["target"].shape)), ("RAW_BOLT", values["raw_forecast"])):
        for key, mask in (("scores_all_observed", values["observed"]), ("scores_complete_mask", complete_mask)):
            output[method][key], count = score_checked(entry[method][key], prediction, values, mask, method + ":" + key); score_fields += count
    require(entry["RAW_BOLT"]["bound_inequality_applied"] is False, "NO_RAW_JOINT_BOUND")
    perturbation = entry["cached_future_label_perturbation"]
    require(perturbation["status"] == "PASS" and perturbation["windows"] == min(2, len(latent)) and perturbation["target_and_observed_mask_perturbed"] is perturbation["prediction_byte_exact"] is perturbation["fit_state_held_fixed"] is True and perturbation["E_F_forward_replayed"] is False, "NARROW_FIXED_CACHE_LABEL_PERTURBATION_SCOPE")
    changed = {**values, "target": np.full_like(values["target"], np.float32(1e6)), "observed": np.zeros_like(values["observed"])}
    probe_n = perturbation["windows"]
    first = anchored_forecast(latent[:probe_n], future[:probe_n], values["raw_last"][:probe_n], decoder, lags)
    second = anchored_forecast(latent[:probe_n], future[:probe_n], changed["raw_last"][:probe_n], decoder, lags)
    exact(first, second, "fixed_cache_target_mask_not_forward_inputs")
    return output, score_fields


def source_contract(records):
    by_name = {Path(record["path"]).name: Path(record["path"]) for record in records}
    texts = {name: by_name[name].read_text(encoding="utf-8-sig") for name in ("causal_lag_decoder_v1.py", "run_causal_lag_objective_v1.py")}
    pure = ast.parse(texts["causal_lag_decoder_v1.py"]); controller = ast.parse(texts["run_causal_lag_objective_v1.py"])
    constants = {node.targets[0].id: ast.literal_eval(node.value) for node in pure.body if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("RCOND", "JOINT_RCOND", "RECON_START")}
    require(constants == {"RCOND": RCOND, "JOINT_RCOND": SPACE_RCOND, "RECON_START": 32}, "REGISTERED_LS_AND_RETAINED_SPACE_CUTOFFS")
    functions = {node.name: node for node in pure.body if isinstance(node, ast.FunctionDef)}
    require([arg.arg for arg in functions["predict"].args.args] == ["history", "future", "current", "decoder", "lags"], "ACTUAL_CACHED_DEPLOYMENT_API_NO_TARGET_OR_MASK")
    require([arg.arg for arg in functions["fit_reconstruction"].args.args] == ["history", "observed", "latent_history", "lags"], "RECON_NO_FUTURE_TARGET_OR_FIT_F_FORECAST_ARGUMENT")
    require("np.linalg.lstsq(design, target, rcond=RCOND)" in ast.unparse(functions["fit_response_columns"]), "PRODUCER_DIRECT_LS_NOT_NORMAL_EQUATION_EXPANSION")
    forbidden = {"FrozenBolt", "NativeFixedDecoder", "fit_target_subspace", "AdamW", "pipeline_predict"}
    require(not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in forbidden for node in ast.walk(controller)), "NO_NEW_F_ENCODER_OR_NATIVE_MODEL_TRAINING_SOURCE")
    for node in ast.walk(controller):
        if isinstance(node, ast.Import): require(not any(alias.name.split(".")[0] in ("torch", "chronos") for alias in node.names), "CPU_CACHE_ONLY_CONTROLLER")
    return {"cached_F_only_no_actual_F_replay_by_auditor": True, "RECON_common_480_and_FORECAST_observed_FIT_supervision_separate": True, "predictor_API_has_no_future_target": True, "joint_unknowns_HK_not_HJK": True, "all_real_latent_forecasts_bound_claimed": False}


def audit_result(result_path, meter, *, expected_windows=None):
    check = meter.check
    binding, result, parents = load_gates(result_path, check)
    if not parents: return {"status": "NOT_AUDITED_INCOMPLETE_PRODUCER", "diagnostic": binding, "producer_status": result.get("status"), "scientific_assets_opened": False}
    producer_complete = result["status"] == "COMPLETE"
    source = parents["raw_subset_diagnostic"]
    files = [binding, *[result[key] for key in PARENT_SHA], *result["source_records"], *result["source_snapshots"], *result["protected_prior_records"], *result["input_records"], result["plan"]]
    for record in files: check(); require(prior.file_sha(record["path"]) == record["sha256"], "SEALED_SOURCE_CACHE_MODEL_LOCK_AND_PLAN_SHA")
    originals = {Path(record["path"]).name: record for record in result["source_records"]}
    for record in result["source_snapshots"]: require(record["sha256"] == originals[Path(record["path"]).name]["sha256"], "EXECUTED_SOURCE_SNAPSHOT_EXACT_BYTES")
    src_contract = source_contract(result["source_records"])
    registry = prior.verified_json(result["fit_registry"], check); files.append(result["fit_registry"])
    keys = {(data, space, condition) for data in DATASETS for space in SPACES for condition in CONDITIONS}
    locks = {}
    require(registry["status"] == "ALL_24_RESOLVED_BEFORE_TUNE_DEV" and registry["TUNE_DEV_NPZ_opened_before_seal"] is False and result["all_24_locks_before_TUNE_DEV_NPZ"] is True, "ALL_FIXED_D_LOCKS_AND_REGISTRY_BEFORE_TUNE_DEV")
    for row in registry["conditions"]:
        key = row["dataset"], row["space"], row["condition"]
        require(key not in locks and row["fit_lock"] == result["datasets"][key[0]]["spaces"][key[1]]["conditions"][key[2]]["fit_lock"], "UNIQUE_EXACT_FIT_LOCK_BINDING")
        lock = prior.verified_json(row["fit_lock"], check); files.append(row["fit_lock"]); locks[key] = lock
        require(lock["status"] in ("LOCKED_FIT_ONLY", "FAILED_PATH_PRESERVED") and lock["sealed_unix"] <= registry["sealed_unix"], "EACH_FIT_PATH_RESOLVED_BEFORE_REGISTRY_SEAL")
    require(set(locks) == keys and result["resolved_fit_lock_files"] == len(keys), "ALL_NOMINAL_TWENTYFOUR_FIT_PATHS_RESOLVED")
    sealed = [i for i, event in enumerate(result["execution_events"]) if event["event"] == "ALL_24_FIT_LOCKS_SEALED"]
    require(len(sealed) == 1 and result["execution_events"][sealed[0]]["registry"] == result["fit_registry"], "EXACT_ONE_REGISTRY_EVENT")
    events = [event for event in result["execution_events"] if event["event"] == "CONTEXT_OPENED"]
    require(all(i > sealed[0] and event["all_24_locks_already_sealed"] is True for i, event in enumerate(result["execution_events"]) if event["event"] == "CONTEXT_OPENED" and event["split"] != "fit"), "EVERY_TUNE_DEV_CONTEXT_OPEN_AFTER_ALL_FIT_LOCKS")
    totals = {"decoder_states_checked": 0, "fit_lock_files_checked": len(locks), "FIT_source_contexts_checked": 0, "cached_Bolt_source_contexts_checked": 0, "cached_D_split_prediction_arrays_checked": 0, "independent_FP32_method_prediction_arrays_checked": 0, "real_trace_prediction_arrays_checked": 0, "joint_output_diagnostic_cells_checked": 0, "EY_prediction_arrays_checked": 0, "flat_prediction_arrays_checked": 0, "retained_trace_inequalities_checked": 0, "original_real_trace_inequalities_evaluated": 0, "FP32_with_allowance_inequalities_evaluated": 0, "RAW_bound_inequalities_checked": 0, "cached_future_label_perturbation_checks": 0, "independent_fit_response_columns_checked": 0, "score_field_checks": 0, "protected_prior_file_hashes_checked": len(result["protected_prior_records"])}
    report = {"status": "PASS" if producer_complete else "PARTIAL_AUDITED_NO_FULL_PASS", "diagnostic": binding, **{key: result[key] for key in PARENT_SHA}, "plan": result["plan"], "fit_registry": result["fit_registry"], "source_contract": src_contract, "totals": totals, "datasets": {}, "producer_failed_paths": result["failed_paths"], "audit_failed_paths": [], "partial_conditions_not_accepted": [], "protected_TEST_accessed": False, "new_F_native_or_encoder_training": False, "actual_F_forward_independently_replayed": False, "TUNE_DEV_or_oracle_used_for_new_fit_selection": False, "all_real_latent_forecasts_bound_claimed": False, "strict_all_FP32_bound_claimed": False, "MAE_or_physical_MSE_minimum_claimed": False, "strict_K_only_compression": False, "independent_generalization_claimed": False, "whole_deployment_cost_measured": False}
    for dataset, (c, k, dev, dev_n) in DATASETS.items():
        row = result["datasets"][dataset]; parent = source["datasets"][dataset]
        require(row["channels"] == c and row["K"] == k and row["development_split"] == dev, "FIXED_C_K_AND_EXPOSED_DEVELOPMENT_SPLIT")
        out = report["datasets"][dataset] = {"spaces": {}}
        for space in SPACES:
            entry, original = row["spaces"][space], parent["spaces"][space]
            contexts = {}; output = out["spaces"][space] = {"conditions": {}}
            require(entry["E_source"] == original["E_state_artifact"] and entry["FIT_input_source"] == original["inputs"]["fit"] and entry["FIT_Bolt_cache"] == original["Bolt"]["fit"]["artifact"], "FIXED_E_FIT_INPUT_AND_ACTUAL_F_CACHE")
            encoder = prior.verified_arrays(entry["E_source"], check)
            for split, count in (("fit", 256), ("tune", 64), (dev, dev_n)):
                if expected_windows is not None: count = expected_windows[split]
                values = prior.verified_arrays(original["inputs"][split], check); prior.check_raw_split(values, channels=c, windows=count)
                cached = prior.verified_arrays(original["Bolt"][split]["artifact"], check)
                latent = raw_audit.encode(values["history"], encoder, "BOLT_NUMPY")
                exact(latent, values["latent_history_BOLT_NUMPY"], "fixed_numpy_E_input_cache_lineage")
                exact(latent, cached["latent_history"], "actual_F_cache_encoded_past")
                exact(values["origins"], cached["origins"], "actual_F_cache_origin_lineage")
                require(cached["latent_future"].dtype == np.float32 and cached["latent_future"].shape == (count, 48, k) and np.isfinite(cached["latent_future"]).all(), "FINITE_ACTUAL_SAVED_K_STREAM_F_FORECAST")
                contexts[split] = (values, latent, cached["latent_future"])
                if split != "fit":
                    for name in ("mean", "std"): exact(values[name], contexts["fit"][0][name], "fixed_FIT_scaler:" + name)
                totals["cached_Bolt_source_contexts_checked"] += 1
                totals["FIT_source_contexts_checked"] += int(split == "fit")
            for condition, (lags, objective) in CONDITIONS.items():
                saved = entry["conditions"][condition]; lock = locks[(dataset, space, condition)]
                if saved["status"] != "LOCKED_FIT_ONLY":
                    require(not producer_complete, "COMPLETE_PRODUCER_WITH_FAILED_FIT")
                    report["partial_conditions_not_accepted"].append({"dataset": dataset, "space": space, "condition": condition, "metadata": saved}); continue
                try:
                    require({key: value for key, value in saved.items() if key not in ("splits", "fit_lock")} == {key: value for key, value in lock.items() if key != "sealed_unix"}, "FIT_LOCK_EXACT_STATE_METADATA_AND_OBJECTIVE")
                    require(saved["FIT_future_targets_used"] == (objective == "FORECAST") and saved["TUNE_DEV_used_for_fit_or_selection"] is saved["oracle_used_for_fit_or_selection"] is False, "FIT_ONLY_SUPERVISION_SEPARATE_FROM_INFERENCE")
                    state = prior.verified_arrays(saved["state_artifact"], check)
                    fitted = check_state(state, saved["fit_metadata"], *contexts["fit"], lags, objective)
                    checked = output["conditions"][condition] = {"state": saved["state_artifact"], "fit": fitted, "splits": {}}
                    totals["decoder_states_checked"] += 1; totals["independent_fit_response_columns_checked"] += c
                    for split, (values, latent, future) in contexts.items():
                        summary = saved["splits"][split]
                        if summary["status"] != "COMPLETE":
                            require(not producer_complete, "COMPLETE_PRODUCER_WITH_FAILED_SPLIT"); report["partial_conditions_not_accepted"].append(summary); continue
                        require(summary["input_source"] == original["inputs"][split] and summary["cached_Bolt_source"] == original["Bolt"][split]["artifact"] and summary["state_source"] == saved["state_artifact"] and summary["fit_lock"] == saved["fit_lock"], "CACHED_SPLIT_INPUT_F_STATE_AND_FIT_LOCK_BINDINGS")
                        arrays = prior.verified_arrays(summary["artifact"], check)
                        checked["splits"][split], count = audit_split(summary, arrays, values, latent, future, encoder, state["decoder_FP32"], lags)
                        totals["score_field_checks"] += count; totals["cached_D_split_prediction_arrays_checked"] += 1
                        for key in ("EY_prediction_arrays_checked", "flat_prediction_arrays_checked", "cached_future_label_perturbation_checks"): totals[key] += 1
                        for key in ("independent_FP32_method_prediction_arrays_checked", "real_trace_prediction_arrays_checked"): totals[key] += 3
                        eligible = bool(values["observed"].all(axis=2).any()); totals["joint_output_diagnostic_cells_checked"] += int(eligible)
                        for key in ("retained_trace_inequalities_checked", "original_real_trace_inequalities_evaluated", "FP32_with_allowance_inequalities_evaluated"): totals[key] += 3 * int(eligible)
                except Exception as error:
                    report["audit_failed_paths"].append({"dataset": dataset, "space": space, "condition": condition, "failure": {"type": type(error).__name__, "message": str(error)}})
    receipt = prior.verified_json(result["resource_terminal"], check)
    require(receipt["status"] == result["status"] and receipt["elapsed_or_cumulative_budget_caps"] is None and result["source_cache_model_selection_SHA_unchanged"] is True, "ACTUAL_CPU_EXECUTION_RECEIPT_AND_OLD_ASSET_PRESERVATION")
    for record in files: check(); require(prior.file_sha(record["path"]) == record["sha256"], "POST_AUDIT_OLD_SOURCE_MODEL_CACHE_AND_LOCKS_UNCHANGED")
    if report["audit_failed_paths"]: report["status"] = "FAILED_PARTIAL_AUDIT_PRESERVED_NO_FULL_PASS"
    elif producer_complete:
        nominal_states = len(keys); nominal_splits = nominal_states * 3
        require(totals["decoder_states_checked"] == result["decoder_fits_completed"] == nominal_states and totals["cached_D_split_prediction_arrays_checked"] == result["cached_D_predictions_completed"] == nominal_splits and totals["cached_Bolt_source_contexts_checked"] == result["actual_F_cache_splits_reused"] == len(DATASETS) * len(SPACES) * 3 and not result["failed_paths"] and not report["partial_conditions_not_accepted"], "FULL_REGISTERED_CAUSAL_LAG_COHORT_COUNTS")
        require(totals["joint_output_diagnostic_cells_checked"] == result["joint_output_diagnostics_completed"] and totals["EY_prediction_arrays_checked"] == result["EY_diagnostics_completed"] and totals["flat_prediction_arrays_checked"] == result["flat_diagnostics_completed"], "ACTUAL_DIAGNOSTIC_ARRAY_COUNTS")
    return report


def execute(result_path, output_path):
    from continuation_runtime_v1 import Meter, write_new
    output = Path(output_path).resolve(); require(not output.exists(), "FRESH_CAUSAL_LAG_AUDIT_OUTPUT")
    output.parent.mkdir(parents=True, exist_ok=True); meter = Meter("audit_causal_lag_" + output.stem, gpu=False)
    source = {"path": str(Path(__file__).resolve()), "sha256": prior.file_sha(__file__)}; failure = None
    try:
        meter.protected_check(); _, _, parents = load_gates(result_path, meter.check)
        if not parents: report = {"status": "NOT_AUDITED_INCOMPLETE_PRODUCER", "scientific_assets_opened": False}
        else:
            initialize_numpy(); sys.path.insert(0, str(ROOT / "research/smallk_failure_followup_20261014"))
            from prepare_inputs import install_protected_test_guard
            install_protected_test_guard()
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=2): report = audit_result(result_path, meter)
        meter.protected_check(); require(prior.file_sha(__file__) == source["sha256"], "INDEPENDENT_AUDITOR_SOURCE_UNCHANGED"); json.dumps(report, allow_nan=False)
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}; report = {"status": "FAILED_PRESERVED", "failure": failure, "full_scope_PASS_claimed": False}
    finally:
        report["audit_source"] = source; report["resource_terminal"] = meter.close(report["status"], failure); write_new(output, report)
    print(json.dumps({"status": report["status"], "totals": report.get("totals"), "failure": failure, "audit_failed_paths": report.get("audit_failed_paths")}))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--result", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); raise SystemExit(execute(args.result, args.output))
