"""Independent saved calibration audit with direct lag math and augmented least squares; no F/GPU replay."""
from pathlib import Path
import argparse
import ast
import json
import math
import sys

import audit_linear_e_adaptation_v1 as base

HERE, ROOT, prior, require = base.HERE, base.ROOT, base.prior, base.require
STATES = ("RECON_CP0", "E_ONLY_s92601", "E_ONLY_s92602")
FAMILIES = ("SCALE_ONLY", "SCALE_BIAS")
LAGS, LAMBDAS = (0, 1, 2, 4, 8, 16, 32), (0., 1e-6, 1e-4, 1e-2, .1, 1., 10.)
RESULT_SHA = "88ddbbf7287ad89b7bf81e170b3e05ddfc89bead700e2fefa50bb0dd4e85f576"
CONTROLLER_SHA = "2d81492385dc8f2c889993f9c5d4f02b32d1917a8e9e20f1b53c88778173ee04"
MODULE_SHA = "088b9800724b07ced0c42cfa2f514077e430d2c3a7f291dea6f45d1f444cb523"
PARENT_SHA = {"matched_l7_native_diagnostic": "4a6f2d3601c27bf000659aa99a586a2f04d72371766f7558cc78cc16028caa2e", "matched_l7_native_audit": "429296ac7346189953813292291584d0c5ceb0a60028b4c9b955a2a429a1db87", "matched_l7_native_cost": "6fce6c6a462cb0b45f6cc3932f94c615ad02b17892c6aade642bc2ad39a6b51c", "matched_l7_native_cost_audit": "cd8716119609ad5a3148d7f9c620fb08b938f9a5d36f61ba7f111dc3ca880983", "parent_report": "2adaf2ac2104b658e36ec10060e58604ce5b699a17f95d4b6ea1b2ee35b70c62", "parent_report_manifest": "e4ec0a98b24019b37bdd414b9569c322825a3df805bdd084e5d1d843b47e272d", "parent_report_audit": "91da4c088a2328229526f90c83e3b73ad8ba065504036dfb6b32065dc8b51e24"}
COEFFICIENT_ATOL, COEFFICIENT_RTOL = 1e-8, 1e-7
DESIGN_ATOL, DESIGN_RTOL = 1e-10, 5e-12
DEPLOYMENT_ATOL, DEPLOYMENT_RTOL = 1e-5, 2e-6
np = sys.modules.get("numpy")


def initialize_numpy():
    global np
    base.initialize_numpy(); np = sys.modules["numpy"]


def finite_metadata(value):
    if isinstance(value, dict):
        for item in value.values(): finite_metadata(item)
    elif isinstance(value, list):
        for item in value: finite_metadata(item)
    elif isinstance(value, float): require(math.isfinite(value), "NONFINITE_JSON_METADATA")


def same_binding(first, second):
    require(Path(first["path"]).resolve() == Path(second["path"]).resolve() and first["sha256"] == second["sha256"], "EXACT_PARENT_BINDING")


def direct_design(past, future, decoder, family):
    require(family in FAMILIES and past.dtype == future.dtype == decoder.dtype == np.float32 and past.ndim == future.ndim == 3 and decoder.ndim == 2 and past.shape[0] == future.shape[0] > 0 and past.shape[1] > 32 and past.shape[2] == future.shape[2] > 0 and decoder.shape[0] == 7*future.shape[2] and all(np.isfinite(v).all() for v in (past, future, decoder)), "FINITE_DIRECT_DESIGN_INPUTS")
    n, horizon, k = future.shape; columns = np.zeros((n, horizon, decoder.shape[1], k if family == "SCALE_ONLY" else 2*k), np.float64)
    for h in range(horizon):
        for li, lag in enumerate(LAGS):
            if h < lag: continue
            for channel in range(k):
                row = decoder[li*k+channel].astype(np.float64)
                displacement = future[:, h-lag, channel].astype(np.float64)-past[:, -1, channel].astype(np.float64)
                columns[:, h, :, channel] += displacement[:, None]*row
                if family == "SCALE_BIAS": columns[:, h, :, k+channel] += row
    return columns


def direct_decode(past, future, current, decoder):
    n, horizon, k = future.shape; features = np.empty((n, horizon, 7*k), np.float32)
    for h in range(horizon):
        for li, lag in enumerate(LAGS):
            source = future[:, h-lag] if h >= lag else past[:, past.shape[1]+h-lag]
            features[:, h, li*k:(li+1)*k] = source-past[:, -1-lag]
    return current[:, None]+features@decoder


def independent_fit(past, future, decoder, baseline, target, observed, family):
    phi = direct_design(past, future, decoder, family)
    require(baseline.dtype == target.dtype == np.float32 and observed.dtype == np.bool_ and baseline.shape == target.shape == observed.shape == phi.shape[:3] and np.isfinite(baseline).all() and np.isfinite(target[observed]).all(), "OBSERVED_FIT_RESPONSE_ONLY")
    counts = observed.sum((0, 1)); active_channels = counts > 0; require(active_channels.any(), "OBSERVED_FIT_CHANNELS_REQUIRED")
    weights = np.zeros(len(counts), np.float64); weights[active_channels] = 1/(int(active_channels.sum())*counts[active_channels]); w = np.broadcast_to(weights, observed.shape)[observed]
    x = phi[observed]; response = (target[observed].astype(np.float64)-baseline[observed].astype(np.float64))*np.sqrt(w)
    active = np.any(x != 0, axis=0); scales = np.sqrt((w[:, None]*x*x).sum(0)); require(np.all(scales[active] > 0) and np.isfinite(scales).all(), "FINITE_NONZERO_RMS")
    matrix = np.sqrt(w[:, None])*x[:, active]/scales[active]
    singular = np.linalg.svd(matrix, compute_uv=False) if active.any() else np.empty(0, np.float64)
    rank = int((singular > singular[0]*1e-12).sum()) if len(singular) else 0
    solutions = []
    for lam in LAMBDAS:
        if not active.any(): theta = np.empty(0, np.float64)
        elif lam == 0: theta = np.linalg.lstsq(matrix, response, rcond=1e-12)[0]
        else:
            augmented = np.concatenate((matrix, np.sqrt(lam)*np.eye(int(active.sum()))), axis=0)
            rhs = np.concatenate((response, np.zeros(int(active.sum()))))
            theta = np.linalg.lstsq(augmented, rhs, rcond=None)[0]
        coefficients = np.zeros(phi.shape[-1], np.float64); coefficients[active] = theta/scales[active]
        solutions.append({"lambda": lam, "coefficients": coefficients, "theta": theta})
    return {"phi": phi, "scales": scales, "active": active, "singular": singular, "rank": rank, "counts": counts, "row_weight_sum": float(w.sum()), "matrix": matrix, "response": response, "solutions": solutions}


def coefficient_check(candidate, saved, expected, k, family):
    identity = candidate["kind"] == "IDENTITY"; p = k if family == "SCALE_ONLY" else 2*k
    require(candidate["family"] == family and saved["coefficients_FP64"].dtype == np.float64 and saved["coefficients_FP64"].shape == (p,), "REGISTERED_COEFFICIENT_SHAPE_DTYPE")
    if identity: expected = np.zeros(p, np.float64)
    prior.close(saved["coefficients_FP64"], expected, "independent_augmented_LS_coefficients", atol=COEFFICIENT_ATOL, rtol=COEFFICIENT_RTOL)
    base.exact(saved["a_FP64"], saved["coefficients_FP64"][:k], "coefficient_a64_partition")
    base.exact(saved["b_FP64"], saved["coefficients_FP64"][k:] if family == "SCALE_BIAS" else np.zeros(k, np.float64), "coefficient_b64_partition")
    with np.errstate(over="ignore", invalid="ignore"): deployed = saved["coefficients_FP64"].astype(np.float32)
    representable = bool(np.isfinite(saved["coefficients_FP64"]).all() and np.isfinite(deployed).all())
    keys = {"a_FP64", "b_FP64", "coefficients_FP64"}
    if representable:
        require(candidate["status"] in ("COMPLETE", "FAILED_CANDIDATE_PRESERVED"), "FINITE_COEFFICIENT_CANDIDATE_STATUS")
        require(set(saved) == keys|{"a_FP32", "b_FP32", "coefficients_FP32"}, "EXACT_FINITE_COEFFICIENT_MEMBERS")
        base.exact(saved["coefficients_FP32"], deployed, "FP64_to_FP32_coefficient_cast")
        base.exact(saved["a_FP32"], deployed[:k], "coefficient_a32_partition")
        base.exact(saved["b_FP32"], deployed[k:] if family == "SCALE_BIAS" else np.zeros(k, np.float32), "coefficient_b32_partition")
    else: require(set(saved) == keys and candidate["status"] == "FAILED_NONFINITE_COEFFICIENTS_PRESERVED", "FAILED_FP32_CONVERSION_PRESERVED")
    require(base.state_hash(saved) == candidate["coefficient_value_sha256"], "COEFFICIENT_VALUE_HASH")
    return representable


def fit_check(row, saved, calculated, target_shape):
    require(set(saved) == {"active_columns", "column_scales_FP64", "singularvalues_FP64"} and saved["active_columns"].dtype == np.bool_ and saved["column_scales_FP64"].dtype == saved["singularvalues_FP64"].dtype == np.float64, "EXACT_FIT_SOLVER_ARTIFACT_MEMBERS_DTYPES")
    base.exact(saved["active_columns"], calculated["active"], "exact_zero_observed_FIT_columns")
    prior.close(saved["column_scales_FP64"], calculated["scales"], "FIT_RMS_no_center", atol=DESIGN_ATOL, rtol=DESIGN_RTOL)
    prior.close(saved["singularvalues_FP64"], calculated["singular"], "weighted_design_singularvalues", atol=DESIGN_ATOL, rtol=DESIGN_RTOL)
    meta = row["fit_metadata"]
    require(meta["FIT_channel_observed_counts"] == calculated["counts"].tolist() and meta["active_FIT_channels"] == int((calculated["counts"] > 0).sum()) and meta["observed_FIT_rows"] == int(calculated["counts"].sum()) and meta["exact_zero_FIT_columns"] == np.flatnonzero(~calculated["active"]).tolist() and meta["zero_lambda_numeric_rank"] == calculated["rank"] and meta["SVD_fits"] == int(calculated["active"].any()) and meta["coefficient_candidates"] == 7, "FIT_COUNTS_ZERO_COLUMNS_RANK_AND_ACTUAL_SVD_COUNT")
    prior.close(meta["row_weight_sum"], 1., "FIT_MACRO_WEIGHT_SUM_ONE", atol=1e-13, rtol=0.)
    for key in ("FIT_column_RMS_only", "response_is_actual_cached_baseline_residual", "identity_centered_regularization_on_FIT_RMS_scaled_coefficients"): require(meta[key] is True, "FIT_SCOPE:"+key)
    for key in ("column_or_response_centered", "positive_lambda_discards_small_singular_values", "coefficients_identified_unique_latent_causal_effects_claimed", "actual_FP32_equals_linear_surrogate_required"): require(meta[key] is False, "FIT_SCOPE:"+key)
    require(meta["zero_lambda_relative_cutoff"] == 1e-12 and meta["lambdas"] == list(LAMBDAS) and meta["E_D_F_fits"] == 0 and (meta["FP64_design_check_atol"], meta["FP64_design_check_rtol"], meta["independent_FP32_deployment_parity_atol"], meta["independent_FP32_deployment_parity_rtol"]) == (DESIGN_ATOL, DESIGN_RTOL, DEPLOYMENT_ATOL, DEPLOYMENT_RTOL), "FROZEN_SOLVER_AND_NUMERIC_CONTRACT")
    require(target_shape[-1] > 0, "NONEMPTY_ACTUAL_C")


def scalar_diagnostics(candidate, coefficients, calculated):
    if candidate["kind"] == "IDENTITY": require(candidate["metadata"]["identity_baseline_byte_exact_separate_path"] is True, "IDENTITY_SEPARATE_PATH"); return
    meta = candidate["metadata"]; beta = coefficients["coefficients_FP64"]; theta = beta[calculated["active"]]*calculated["scales"][calculated["active"]]
    residual = calculated["response"]-calculated["matrix"]@theta
    expected = {"standardized_coefficient_L2": float(np.linalg.norm(theta)), "coefficient_L2": float(np.linalg.norm(beta)), "coefficient_max_abs": float(np.abs(beta).max()), "surrogate_FIT_macro_MSE": float(residual@residual), "ridge_penalty": float(candidate["lambda"]*(theta@theta))}
    for key, value in expected.items():
        if math.isfinite(value): prior.close(meta[key], value, "saved_solver_scalar:"+key, atol=1e-8, rtol=1e-7)
        else: require(meta[key] is None and key in meta["nonfinite_scalar_fields"], "NONFINITE_SOLVER_SCALAR_PRESERVED")
    require(meta["coefficient_clipping_applied"] is False and meta["negative_total_scale_channels"] == np.flatnonzero(1+coefficients["a_FP64"] < 0).tolist() and meta["singular_filter"] == ("relative_cutoff_1e-12" if candidate["lambda"] == 0 else "all_singular_values_sigma_over_sigma2_plus_lambda"), "IDENTITY_RIDGE_FILTER_UNCLIPPED_GAIN")


def coefficient_metadata(candidate, saved):
    out = {key: candidate[key] for key in ("id", "kind", "family", "lambda", "status", "metadata", "coefficient_artifact", "coefficient_value_sha256")}
    out["nonfinite_coefficient_indices"] = {}
    for key in ("a_FP64", "b_FP64", "a_FP32", "b_FP32", "coefficients_FP64", "coefficients_FP32"):
        out[key] = [float(v) if np.isfinite(v) else None for v in saved[key]] if key in saved else None
        if key in saved and not np.isfinite(saved[key]).all(): out["nonfinite_coefficient_indices"][key] = np.flatnonzero(~np.isfinite(saved[key])).tolist()
    for key in ("FIT", "TUNE", "TUNE_MSE", "failure"):
        if key in candidate: out[key] = candidate[key]
    return out


def prediction_check(entry, saved, candidate, coefficients, cached, raw, decoder, family):
    identity = candidate["kind"] == "IDENTITY"
    require(set(saved) == {"prediction_FP32", "latent_future_calibrated_FP32", "surrogate_prediction_FP64", "actual_minus_surrogate_FP64", "independent_NumPy_future_FP32", "independent_NumPy_prediction_FP32", "origins"} and saved["surrogate_prediction_FP64"].dtype == saved["actual_minus_surrogate_FP64"].dtype == np.float64, "EXACT_SAVED_PREDICTION_AND_FP64_DIAGNOSTIC_MEMBERS")
    for key in ("input_source", "cached_Bolt_source"): require(key in entry, "SAVED_PREDICTION_LINEAGE")
    base.exact(saved["origins"], raw["origins"], "selected_prediction_origins")
    require(saved["prediction_FP32"].dtype == saved["latent_future_calibrated_FP32"].dtype == np.float32 and np.isfinite(saved["prediction_FP32"]).all() and np.isfinite(saved["latent_future_calibrated_FP32"]).all(), "FINITE_SAVED_ACTUAL_FP32")
    expected_future = cached["latent_future"] if identity else cached["latent_future"]+coefficients["a_FP32"]*(cached["latent_future"]-cached["latent_history"][:, -1:])+coefficients["b_FP32"]
    expected = direct_decode(cached["latent_history"], expected_future, raw["raw_last"], decoder)
    base.exact(saved["independent_NumPy_future_FP32"], expected_future, "saved_independent_FP32_q")
    base.exact(saved["independent_NumPy_prediction_FP32"], expected, "saved_independent_FP32_lag_decode")
    parity = {}
    for key, actual, reference in (("latent_future", saved["latent_future_calibrated_FP32"], expected_future), ("C_prediction", saved["prediction_FP32"], expected)):
        recorded = entry["independent_FP32_deployment_parity"][key]
        passed = bool(actual.shape == reference.shape and np.isfinite(actual).all() and np.isfinite(reference).all() and np.allclose(actual, reference, atol=DEPLOYMENT_ATOL, rtol=DEPLOYMENT_RTOL))
        require(recorded["pass"] is passed and (recorded["atol"], recorded["rtol"]) == (DEPLOYMENT_ATOL, DEPLOYMENT_RTOL), "PREDECLARED_INDEPENDENT_DEPLOYMENT_PARITY")
        delta = actual.astype(np.float64)-reference.astype(np.float64)
        prior.close(recorded["max_abs"], float(np.abs(delta).max()), "independent_parity_max_abs", atol=1e-12, rtol=1e-12)
        prior.close(recorded["rmse"], float(np.sqrt(np.mean(delta*delta))), "independent_parity_rmse", atol=1e-12, rtol=1e-12)
        parity[key] = passed
    require(entry["status"] == ("COMPLETE" if all(parity.values()) else "FAILED_DEPLOYMENT_PARITY_PRESERVED"), "NUMERIC_FAILURE_STATUS_PRESERVED")
    if identity:
        base.exact(saved["prediction_FP32"], cached["prediction"], "identity_actual_baseline_bytes"); base.exact(saved["latent_future_calibrated_FP32"], cached["latent_future"], "identity_original_F_future_bytes")
    require(entry["identity_checks"]["identity_byte_exact"] is identity and entry["identity_checks"]["batch_size"] == 32 and entry["identity_checks"]["identity_object_checks"] == (math.ceil(len(raw["raw_last"])/32) if identity else 0), "CANONICAL_IDENTITY_OBJECT_AND_BYTE_RECORD")
    phi = direct_design(cached["latent_history"], cached["latent_future"], decoder, family)
    surrogate = cached["prediction"].astype(np.float64)+phi@coefficients["coefficients_FP64"]
    prior.close(saved["surrogate_prediction_FP64"], surrogate, "independent_FP64_surrogate", atol=DESIGN_ATOL, rtol=DESIGN_RTOL)
    base.exact(saved["actual_minus_surrogate_FP64"], saved["prediction_FP32"].astype(np.float64)-saved["surrogate_prediction_FP64"], "actual_minus_surrogate_array")
    error = saved["prediction_FP32"].astype(np.float64)-surrogate; meta = entry["surrogate_diagnostics"]
    for key, value in (("max_abs_actual_FP32_minus_linear_surrogate", np.abs(error).max()), ("RMS_actual_FP32_minus_linear_surrogate", np.sqrt(np.mean(error*error))), ("per_channel_max_abs", np.max(np.abs(error), axis=(0, 1)))): prior.close(meta[key], value, key, atol=DESIGN_ATOL, rtol=DESIGN_RTOL)
    require(meta["surrogate_zero_error_required"] is meta["linear_surrogate_is_actual_deployment_claimed"] is False and entry["FP64_surrogate_is_not_deployment_parity_target"] is entry["target_future_is_loss_or_evaluation_only"] is True and entry["coefficient_value_sha256"] == base.state_hash(coefficients), "NUMERIC_DIAGNOSTIC_NOT_ZERO_ERROR_OR_PARITY_TARGET")
    counts = prior.compare_scores(entry["scores_all_observed"], prior.scores(saved["prediction_FP32"], raw["target"], raw["observed"], raw["std"]), "split_own_mask_counts")
    complete = np.broadcast_to(raw["observed"].all(2, keepdims=True), raw["observed"].shape)
    if complete.any(): counts += prior.compare_scores(entry["scores_complete_mask"], prior.scores(saved["prediction_FP32"], raw["target"], complete, raw["std"]), "complete_C_own_counts")
    else: require(entry["scores_complete_mask"] == {"status": "NO_COMPLETE_C"}, "NO_COMPLETE_C_EXPLICIT")
    counts += prior.compare_scores(entry["surrogate_scores_all_observed"], prior.scores(surrogate, raw["target"], raw["observed"], raw["std"]), "surrogate_scores_not_actual_deployment")
    return counts


def selected(records):
    valid = [r for r in records if r["status"] == "COMPLETE" and type(r.get("TUNE_MSE")) in (int, float) and math.isfinite(r["TUNE_MSE"])]
    require(bool(valid), "FINITE_TUNE_SELECTION_REQUIRED")
    return min(valid, key=lambda r: (r["TUNE_MSE"], r["kind"] != "IDENTITY", -r["lambda"] if r["kind"] == "RIDGE" else 0))


def source_contract(text):
    nodes = {n.name: n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef)}
    context, run, infer, execute = (ast.unparse(nodes[k]) for k in ("context", "run_complete", "infer_development", "execute"))
    require("SIX_FAMILY_LOCKS_BEFORE_DEV_ARRAY_IO" in context and "module.fit_calibration(cached['latent_history'], cached['latent_future'], initial_states[state]['D'], cached['prediction'], fit['target'], fit['observed'], LAGS, family)" in run and "ALL_SIX_LOCKS_BEFORE_ANY_DEV" in infer and execute.index("resolved = resolve_families(") < execute.index("infer_development(result, resolved, infer)"), "FIT_ONLY_AND_SIX_LOCK_SOURCE_FLOW")
    forbidden = ("FrozenBolt", "from_pretrained", "torch.optim", ".backward(", "train_head(")
    require(not any(token in text for token in forbidden), "NO_F_MODEL_HEAD_OPTIMIZER_OR_BACKWARD_SOURCE")
    return {"FIT_only_response_and_six_locks_before_DEV_source": True, "no_F_model_head_optimizer_or_backward_source": True, "independent_runtime_GPU_or_optimizer_replay": False}


def metadata(path, check):
    require(all(isinstance(v, str) and len(v) == 64 for v in (RESULT_SHA, CONTROLLER_SHA)), "ACTUAL_FIXED_L7_CALIBRATION_NOT_YET_REGISTERED")
    binding = base.file_record(path); require(binding["sha256"] == RESULT_SHA, "EXACT_CALIBRATION_RESULT_SHA")
    result = prior.verified_json(binding, check); finite_metadata(result)
    require(result["status"] in ("COMPLETE", "PARTIAL_COMPLETED") and result["controller_source"]["sha256"] == CONTROLLER_SHA and result["module_source"]["sha256"] == MODULE_SHA and all(result[k]["sha256"] == v for k, v in PARENT_SHA.items()), "EXECUTED_SOURCES_AND_ACCEPTED_PARENT_BUNDLE")
    return binding, result


def receipt_check(receipt, status, check):
    require(all(type(receipt.get(k)) in (int, float) and math.isfinite(receipt[k]) and receipt[k] >= 0 for k in ("process_cpu_seconds", "wall_seconds")), "CANONICAL_PROCESS_CPU_AND_WALL_RECEIPT")
    binding = {key: receipt[key] for key in ("path", "sha256")}; recorded = prior.verified_json(binding, check); finite_metadata(recorded)
    require(recorded == {k: v for k, v in receipt.items() if k not in ("path", "sha256")} and receipt["status"] == status and receipt["gpu_process"] is True and receipt["original_budget_records_unchanged"] is True and receipt["elapsed_or_cumulative_budget_caps"] is None, "EXACT_SOURCE_RESOURCE_RECEIPT_CONTENT_AND_SCOPE")
    authorization = prior.verified_json(receipt["authorization"], check); manifest = prior.verified_json(receipt["continuation_manifest"], check)
    finite_metadata(authorization); finite_metadata(manifest)
    require(authorization["status"] == "APPROVED_HUMAN_UNCAPPED_CONTINUATION" and authorization["latest_human_override"] == "상한내수행하지 말고 쭉해줘" and manifest["authorization_sha256"] == receipt["authorization"]["sha256"] and manifest["runtime_sha256"] == prior.file_sha(HERE/"continuation_runtime_v1.py"), "RECORDED_AUTHORIZATION_MANIFEST_RUNTIME_SHA_CHAIN")
    return [binding, receipt["authorization"], receipt["continuation_manifest"]]


def audit_result(path, meter):
    check = meter.check; binding, result = metadata(path, check)
    parents = {key: prior.verified_json(result[key], check) for key in ("matched_l7_native_diagnostic", "matched_l7_native_audit", "matched_l7_native_cost", "matched_l7_native_cost_audit", "parent_report_manifest", "parent_report_audit", "causal_lag_diagnostic")}
    require(all(value["status"] == ("PASS" if key.endswith("audit") else "COMPLETE") and value.get("protected_TEST_accessed", False) is False for key, value in parents.items()), "ACCEPTED_SCIENCE_COST_REPORT_METADATA")
    native = parents["matched_l7_native_diagnostic"]; science = parents["causal_lag_diagnostic"]
    same_binding(parents["matched_l7_native_audit"]["diagnostic"], result["matched_l7_native_diagnostic"])
    same_binding(parents["matched_l7_native_cost_audit"]["cost"], result["matched_l7_native_cost"])
    for key, parent_key in (("report", "parent_report"), ("manifest", "parent_report_manifest")): same_binding(parents["parent_report_audit"][key], result[parent_key])
    for key, source in (("train", "matched_l7_native_diagnostic"), ("audit", "matched_l7_native_audit"), ("cost", "matched_l7_native_cost"), ("cost_audit", "matched_l7_native_cost_audit")): same_binding(parents["parent_report_audit"]["inputs"][key], result[source])
    same_binding(native["causal_lag_diagnostic"], result["causal_lag_diagnostic"])
    require(result["cached_Bolt_contexts_reused"] == result["cached_baseline_redecode_checks"] <= 9 and (result["status"] != "COMPLETE" or result["cached_Bolt_contexts_reused"] == 9) and result["selected_FIT_TUNE_predictions_reused_without_new_inference"] is True, "MATCHED_BASELINE_CONTEXTS_SELECTED_FIT_TUNE_REUSE")
    receipt = result["resource_terminal"]; receipt_records = receipt_check(receipt, result["status"], check)
    for key in ("new_F_forward_calls", "new_F_model_objects_loaded", "new_optimizer_updates", "new_E_D_training_updates"): require(result[key] == 0, "NO_NEW_F_E_D_OPTIMIZER:"+key)
    for key in ("native_head_weights_loaded", "new_backbone_training", "new_F_state_hash_computed", "whole_deployment_cost_measured", "independent_holdout", "strict_K_only", "protected_TEST_accessed"): require(result[key] is False, "CALIBRATION_SCOPE:"+key)
    require(result["F_parent_signatures_reused"] == native["F_parent_signatures_reused"], "SAME_DECLARED_F_SIGNATURES_NO_REHASH")
    guarded = [binding, result["plan"], *result["source_records"], *result["input_records"], *result["protected_prior_records"], *receipt_records]
    for item in guarded:
        require(prior.file_sha(item["path"]) == item["sha256"], "GUARDED_SOURCE_INPUT_PARENT_SHA")
        if "snapshot_path" in item: require(prior.file_sha(item["snapshot_path"]) == item["sha256"], "EXECUTED_SOURCE_SNAPSHOT_SHA")
    contract = source_contract(Path(result["controller_source"]["path"]).read_text(encoding="utf-8"))
    registry = prior.verified_json(result["selection_registry"], check); guarded.append(result["selection_registry"])
    require(registry["status"] == "ALL_SIX_RESOLVED_BEFORE_DEV" and registry["DEV_opened_before_seal"] is False and len(registry["conditions"]) == result["resolved_selection_locks"] == 6 and result["all_6_locks_before_DEV"] is True, "ALL_SIX_RESOLVED_LOCKS_BEFORE_DEV")
    locks = {}
    for item in registry["conditions"]:
        key = item["state"], item["family"]; require(key in {(s, f) for s in STATES for f in FAMILIES} and key not in locks, "UNIQUE_SIX_FAMILY_LOCKS")
        lock = prior.verified_json(item["selection_lock"], check); guarded.append(item["selection_lock"])
        require(lock["DEV_opened_before_lock"] is False and lock["sealed_unix"] <= registry["sealed_unix"], "SEALED_LOCK_BEFORE_REGISTRY")
        locks[key] = item, lock
    loaded = {}
    def load(item):
        key = item["path"], item["sha256"]
        if key not in loaded: loaded[key] = prior.verified_arrays(item, check); guarded.append(item)
        return loaded[key]
    totals = {k: 0 for k in ("fit_conditions_checked", "actual_SVD_fits_checked", "ridge_candidates_checked", "identity_candidates_checked", "FIT_TUNE_prediction_evaluations_checked", "selected_DEV_prediction_evaluations_checked", "score_fields_checked", "selection_locks_checked")}
    report = {"status": "PASS" if result["status"] == "COMPLETE" else "PARTIAL_AUDITED_NO_FULL_PASS", "diagnostic": binding, **{key: result[key] for key in PARENT_SHA}, "totals": totals, "states": {}, "partial_paths": [], "failed_candidates_preserved": result["failed_candidate_paths"], "source_contract": contract, "source_resource_terminal": receipt, "protected_TEST_accessed": False, "independent_F_model_GPU_timer_optimizer_or_E_D_training_replayed": False, "independent_calibration_solver_recomputed": True, "independent_holdout": False, "strict_K_only": False, "coefficient_comparison_atol": COEFFICIENT_ATOL, "coefficient_comparison_rtol": COEFFICIENT_RTOL, "scope": "Independent direct future-only lag design and FP64 FIT RMS/weights, lambda-zero lstsq and positive-lambda augmented least squares; saved FP32 q/lag predictions and own-split scores, actual-vs-surrogate diagnostic, candidate/lock/source/input hashes. No F/GPU inference, E/D or optimizer replay, new dataset or independent generalization."}
    report.update({key: result[key] for key in ("selection_registry", "controller_source", "module_source", "plan", "F_parent_signatures_reused")})
    recon = science["datasets"]["robin"]["spaces"]["PCA"]["conditions"]["L7_RECON"]["splits"]
    raw = {s: load(recon[s]["input_source"]) for s in ("fit", "tune", "val")}
    for split, n in (("fit", 256), ("tune", 64), ("val", 30)): prior.check_raw_split(raw[split], channels=17, windows=n)
    cached = {}
    for state in STATES:
        owner = result["states"][state]; same_binding(owner["frozen_state_source"], native["states"][state]["frozen_state_source"]); frozen = load(owner["frozen_state_source"])
        require(set(frozen) == {"E", "D", "center"} and frozen["E"].shape == (17, 4) and frozen["D"].shape == (28, 17) and owner["E_previously_supervised_by_FIT_future"] == (state != "RECON_CP0") and owner["D_previously_supervised_by_FIT_future"] is False, "FROZEN_C17_K4_L7_AND_SUPERVISION")
        for split in ("fit", "tune", "val"):
            source = native["states"][state]["Bolt"][split]["artifact"]; values = cached[state, split] = load(source)
            for key in ("target", "observed", "raw_last", "origins", "mean", "std"): base.exact(values[key], raw[split][key], "accepted_Bolt_raw_"+key)
            prior.close(values["latent_history"], (raw[split]["history"]-frozen["center"])@frozen["E"], "independent_frozen_E_encoding")
            prior.close(values["prediction"], direct_decode(values["latent_history"], values["latent_future"], raw[split]["raw_last"], frozen["D"]), "accepted_current_D_baseline")
        report["states"][state] = {}
        for family in FAMILIES:
            row = owner["families"][family]; item, lock = locks[state, family]; same_binding(row["selection_lock"], item["selection_lock"]); totals["selection_locks_checked"] += 1
            require(row["state"] == lock["state"] == state and row["family"] == lock["family"] == family and row["status"] == lock["status"] == item["status"] and row["candidates"] == lock["candidates"], "LOCKED_FIT_CANDIDATES_UNCHANGED")
            if row["status"] != "COMPLETE":
                report["partial_paths"].append({"state": state, "family": family, "status": row["status"]})
                if "fit_metadata" not in row: continue
            same_binding(row["FIT_input_source"], recon["fit"]["input_source"]); same_binding(row["FIT_cached_Bolt_source"], native["states"][state]["Bolt"]["fit"]["artifact"]); same_binding(row["frozen_state_source"], owner["frozen_state_source"])
            fit_cache = cached[state, "fit"]; calculated = independent_fit(fit_cache["latent_history"], fit_cache["latent_future"], frozen["D"], fit_cache["prediction"], raw["fit"]["target"], raw["fit"]["observed"], family); fit_check(row, load(row["fit_artifact"]), calculated, raw["fit"]["target"].shape)
            totals["fit_conditions_checked"] += 1; totals["actual_SVD_fits_checked"] += int(calculated["active"].any())
            expected_candidates = [("IDENTITY", "IDENTITY", None)]+[("RIDGE_"+format(v, "g"), "RIDGE", v) for v in LAMBDAS]
            require(len(row["candidates"]) <= 8 and (row["status"] != "COMPLETE" or len(row["candidates"]) == 8) and [(c["id"], c["kind"], c["lambda"]) for c in row["candidates"]] == expected_candidates[:len(row["candidates"])], "EXACT_IDENTITY_PLUS_SEVEN_LAMBDAS_OR_PRESERVED_FAILED_PREFIX")
            coefficient_states, candidate_metadata = {}, []
            for index, candidate in enumerate(row["candidates"]):
                coefficients = coefficient_states[candidate["id"]] = load(candidate["coefficient_artifact"]); expected = np.zeros(4 if family == "SCALE_ONLY" else 8) if index == 0 else calculated["solutions"][index-1]["coefficients"]
                representable = coefficient_check(candidate, coefficients, expected, 4, family); scalar_diagnostics(candidate, coefficients, calculated); totals["identity_candidates_checked" if index == 0 else "ridge_candidates_checked"] += 1
                for split in ("fit", "tune"):
                    if split.upper() not in candidate: continue
                    entry = candidate[split.upper()]; same_binding(entry["input_source"], recon[split]["input_source"]); same_binding(entry["cached_Bolt_source"], native["states"][state]["Bolt"][split]["artifact"])
                    require(representable, "NO_INFERENCE_FOR_NONFINITE_FP32_COEFFICIENTS")
                    totals["score_fields_checked"] += prediction_check(entry, load(entry["artifact"]), candidate, coefficients, cached[state, split], raw[split], frozen["D"], family)
                    totals["FIT_TUNE_prediction_evaluations_checked"] += int(entry["status"] == "COMPLETE")
                if candidate["status"] == "COMPLETE": require(candidate["FIT"]["status"] == candidate["TUNE"]["status"] == "COMPLETE" and candidate["TUNE_MSE"] == candidate["TUNE"]["scores_all_observed"]["normalized"]["mse"], "ACTUAL_OWN_TUNE_SELECTION_SCORE")
                candidate_metadata.append(coefficient_metadata(candidate, coefficients))
            if row["status"] != "COMPLETE":
                report["states"][state][family] = {"status": row["status"], "fit_metadata": row["fit_metadata"], "candidates": candidate_metadata}; continue
            winner = selected(row["candidates"])
            require(row["selected_candidate"] == lock["selected_candidate"] == winner["id"] and row["selected_kind"] == winner["kind"] and row["selected_lambda"] == winner["lambda"] and row["selected_TUNE_MSE"] == winner["TUNE_MSE"] and row["FIT"] == winner["FIT"] and row["TUNE"] == winner["TUNE"], "IDENTITY_THEN_LARGER_LAMBDA_TUNE_ONLY_LOCK")
            same_binding(row["selected_coefficients"], winner["coefficient_artifact"]); same_binding(lock["selected_coefficients"], winner["coefficient_artifact"])
            require(row["selected_coefficient_value_sha256"] == winner["coefficient_value_sha256"], "SELECTED_COEFFICIENT_HASH_UNCHANGED")
            if "DEV" in row:
                entry = row["DEV"]; same_binding(entry["input_source"], recon["val"]["input_source"]); same_binding(entry["cached_Bolt_source"], native["states"][state]["Bolt"]["val"]["artifact"])
                totals["score_fields_checked"] += prediction_check(entry, load(entry["artifact"]), winner, coefficient_states[winner["id"]], cached[state, "val"], raw["val"], frozen["D"], family)
                require(Path(entry["artifact"]["path"]).stat().st_mtime >= registry["sealed_unix"], "DEV_ARTIFACT_AFTER_SIX_LOCK_REGISTRY")
                totals["selected_DEV_prediction_evaluations_checked"] += int(entry["status"] == "COMPLETE")
            report["states"][state][family] = {"selected_candidate": winner["id"], "selected_lambda": winner["lambda"], "FIT_solver_rank": calculated["rank"], "fit_metadata": row["fit_metadata"], "independently_recomputed_fit": {"column_scales_FP64": calculated["scales"].tolist(), "active_columns": calculated["active"].tolist(), "singularvalues_FP64": calculated["singular"].tolist(), "zero_lambda_numeric_rank": calculated["rank"], "FIT_channel_observed_counts": calculated["counts"].tolist(), "row_weight_sum": calculated["row_weight_sum"]}, "candidates": candidate_metadata, "independent_positive_lambda_solver": "augmented_weighted_lstsq_no_Gram", "selected_scores": {s: row[s]["scores_all_observed"] for s in ("FIT", "TUNE", "DEV") if s in row}}
    mapping = {"fit_conditions_checked": "fit_conditions_completed", "actual_SVD_fits_checked": "SVD_fits_completed", "ridge_candidates_checked": "ridge_candidates_generated", "identity_candidates_checked": "identity_comparisons_generated", "FIT_TUNE_prediction_evaluations_checked": "candidate_FIT_TUNE_evaluations_completed", "selected_DEV_prediction_evaluations_checked": "selected_DEV_evaluations_completed"}
    for key, source in mapping.items(): require(totals[key] == result[source] if result["status"] == "COMPLETE" else totals[key] <= result[source], "ACTUAL_COMPLETED_COUNT:"+source)
    if result["status"] == "COMPLETE": require([totals[k] for k in ("fit_conditions_checked", "ridge_candidates_checked", "identity_candidates_checked", "FIT_TUNE_prediction_evaluations_checked", "selected_DEV_prediction_evaluations_checked")] == [6, 42, 6, 96, 6] and totals["actual_SVD_fits_checked"] <= 6 and not result["failed_paths"] and not result["failed_candidate_paths"], "SIX_FITS_MAX_SIX_SVD_42_COEFFICIENTS_SIX_IDENTITY_96_FT_SIX_DEV")
    for item in guarded: check(); require(prior.file_sha(item["path"]) == item["sha256"], "SOURCE_INPUT_RESULT_BYTES_UNCHANGED")
    totals["protected_source_input_and_result_hashes_checked"] = len(guarded)
    return report


def execute(path, output):
    from continuation_runtime_v1 import Meter, write_new
    output = Path(output).resolve(); require(not output.exists(), "FRESH_CALIBRATION_AUDIT_OUTPUT"); meter = Meter("audit_fixed_L7_calibration_"+output.stem, gpu=False); source = base.file_record(__file__); failure = None
    try:
        metadata(path, meter.check); initialize_numpy(); sys.path.insert(0, str(ROOT/"research/smallk_failure_followup_20261014"))
        from prepare_inputs import install_protected_test_guard
        from threadpoolctl import threadpool_limits
        install_protected_test_guard(); meter.protected_check()
        with threadpool_limits(limits=2): report = audit_result(path, meter)
        meter.protected_check(); require(prior.file_sha(__file__) == source["sha256"], "AUDIT_SOURCE_UNCHANGED"); json.dumps(report, allow_nan=False)
    except Exception as error: failure = {"type": type(error).__name__, "message": str(error)}; report = {"status": "FAILED_PRESERVED", "failure": failure}
    finally: report["audit_source"] = source; report["resource_terminal"] = meter.close(report["status"], failure); write_new(output, report)
    print(json.dumps({"status": report["status"], "totals": report.get("totals"), "failure": failure}), flush=True); return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--result", required=True); parser.add_argument("--output", required=True); args = parser.parse_args(); raise SystemExit(execute(args.result, args.output))
