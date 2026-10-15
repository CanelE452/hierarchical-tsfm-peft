"""Independent augmented-LS past shrinkage and Bolt-only selection audit."""
from pathlib import Path
import argparse
import ast
import json
import sys

import audit_projection_results_v1 as prior
import audit_local_past_decoder_probe_v1 as local_audit

np = sys.modules.get("numpy")
HERE, ROOT = prior.HERE, prior.ROOT
DATASETS = local_audit.DATASETS
RHOS = (.001, .01, .1, 1., 10.)
RHO_LABELS = ("RHO_0.001", "RHO_0.01", "RHO_0.1", "RHO_1", "RHO_10")
CANDIDATE_ORDER = ("STATIC", "LOCAL", *RHO_LABELS)
require, exact = prior.require, local_audit.exact
raw_binding = local_audit.raw_binding


def initialize_numpy():
    global np
    local_audit.initialize_numpy()
    np = sys.modules["numpy"]


def independent_augmented_fit(history, observed, latent, prior32, rho):
    require(history.dtype == latent.dtype == prior32.dtype == np.float32 and history.ndim == latent.ndim == 3
            and history.shape[1] == 512 and len(history) > 0 and history.shape[:2] == latent.shape[:2]
            and 1 <= latent.shape[-1] <= history.shape[-1] and prior32.shape == (latent.shape[-1], history.shape[-1])
            and observed.dtype == np.bool_ and observed.shape == history.shape
            and all(np.isfinite(array).all() for array in (history, latent, prior32))
            and type(rho) in (float, int) and rho > 0 and np.isfinite(rho), "FINITE_PAST_ONLY_SHRINKAGE_INPUTS_AND_POSITIVE_RHO")
    n, _, c = history.shape
    k = latent.shape[-1]
    d = np.broadcast_to(prior32.astype(np.float64)[None], (n, k, c)).copy()
    bias = np.zeros((n, c), np.float64)
    counts = observed.sum(axis=1, dtype=np.int64)
    traces, penalties, mean_y, mse, conditions = (np.zeros((n, c), np.float64) for _ in range(5))
    mean_z = np.zeros((n, c, k), np.float64)
    zero_trace = np.zeros((n, c), bool)
    stationarity = np.zeros((n, c), np.float64)
    for window in range(n):
        for channel in range(c):
            valid = observed[window, :, channel]
            count = int(counts[window, channel])
            old = prior32[:, channel].astype(np.float64)
            if not count:
                bias[window, channel] = float(history[window, -1, channel]) - float(latent[window, -1].astype(np.float64) @ old)
                continue
            z, y = latent[window, valid].astype(np.float64), history[window, valid, channel].astype(np.float64)
            zm, ym = z.mean(axis=0), float(y.mean())
            zc, yc = z - zm, y - ym
            mean_z[window, channel], mean_y[window, channel] = zm, ym
            trace = float(np.square(zc).sum() / count)
            traces[window, channel] = trace
            gram, linear = zc.T @ zc / count, zc.T @ yc / count
            if trace > 0:
                penalty = float(rho) * trace / k
                design = np.vstack((zc / np.sqrt(count), np.sqrt(penalty) * np.eye(k)))
                response = np.concatenate((yc / np.sqrt(count), np.sqrt(penalty) * old))
                coefficient, _, rank, singular = np.linalg.lstsq(design, response, rcond=None)
                require(rank == k, "POSITIVE_PENALTY_AUGMENTED_DESIGN_FULL_RANK")
                d[window, :, channel], penalties[window, channel] = coefficient, penalty
                conditions[window, channel] = float((singular[0] / singular[-1]) ** 2)
                require(conditions[window, channel] <= (1 + k / rho) * (1 + 1e-10), "RELATIVE_RIDGE_CONDITION_CERTIFICATE_NOT_ABSOLUTE_GAIN_BOUND")
            else:
                zero_trace[window, channel] = True
                coefficient = old
            bias[window, channel] = ym - float(zm @ coefficient)
            residual = z @ coefficient + bias[window, channel] - y
            mse[window, channel] = float(np.mean(residual * residual))
            normal = gram @ coefficient + penalties[window, channel] * (coefficient - old) - linear
            stationarity[window, channel] = float(np.linalg.norm(normal))
            require(stationarity[window, channel] <= 1e-10 * (1 + np.linalg.norm(linear) + np.linalg.norm(gram) * np.linalg.norm(coefficient)), "AUGMENTED_LS_RIDGE_STATIONARITY_CERTIFICATE")
    return {"decoder_fit_FP64": d, "decoder_bias_fit_FP64": bias, "true_observed_counts": counts,
            "local_covariance_trace": traces, "actual_lambda": penalties, "zero_trace_prior_slope_retained": zero_trace,
            "no_observation_prior_with_current_intercept": counts == 0, "observed_latent_means_per_C": mean_z, "observed_raw_C_means": mean_y,
            "observed_real_past_reconstruction_MSE": mse, "past_metric_valid": counts > 0, "ridge_matrix_condition": conditions,
            "ridge_matrix_condition_valid": penalties > 0, "independent_augmented_stationarity_norm": stationarity}


def check_state(entry, arrays, values, latent, prior32, rho):
    require(set(arrays) == {"decoder_fit_FP64", "decoder_deployed_FP32", "decoder_bias_fit_FP64", "decoder_bias_FP32", "metadata_json"}, "EXACT_SHRINKAGE_STATE_KEYS")
    metadata = entry["state_metadata"]
    require(json.loads(str(arrays["metadata_json"].item())) == metadata, "SHRINKAGE_EMBEDDED_STATE_METADATA")
    n, _, c = values["history"].shape
    k = latent.shape[-1]
    for key, shape, dtype in (("decoder_fit_FP64", (n, k, c), np.float64), ("decoder_deployed_FP32", (n, k, c), np.float32),
                              ("decoder_bias_fit_FP64", (n, c), np.float64), ("decoder_bias_FP32", (n, c), np.float32)):
        require(arrays[key].shape == shape and arrays[key].dtype == dtype and np.isfinite(arrays[key]).all(), "FINITE_SHRUNK_STATE_TYPE_SHAPE:" + key)
    exact(arrays["decoder_deployed_FP32"], arrays["decoder_fit_FP64"].astype(np.float32), "shrunk_actual_decoder_cast32")
    exact(arrays["decoder_bias_FP32"], arrays["decoder_bias_fit_FP64"].astype(np.float32), "shrunk_actual_intercept_cast32")
    require(metadata["route"] == "OBSERVED_PAST_CENTERED_RIDGE_TOWARD_STATIC_D" and metadata["windows"] == n and metadata["channels"] == c and metadata["K"] == k and metadata["rho"] == rho
            and metadata["lambda_rule"] == "rho*trace(observed_centered_Z.T@Z/n)/K" and metadata["slope_singular_cutoff_used"] is False
            and metadata["future_labels_used_in_decoder_fit"] is metadata["absolute_decoder_gain_bound_claimed"] is metadata["decoder_interface_same_as_static"] is metadata["strict_K_only_compression"] is metadata["scientific_IO_in_module"] is False
            and metadata["intercept_analytic_unpenalized"] is metadata["native_head_frozen_intervention_not_reoptimized"] is metadata["inference_time_past_decoder_refitting"] is True,
            "REGISTERED_PAST_RIDGE_TOWARD_STATIC_PRIOR_WITH_CHANGED_INFORMATION")
    require(metadata["fit_input_bindings"] == {"history": raw_binding(values["history"]), "history_observed": raw_binding(values["history_observed"]), "latent_history": raw_binding(latent), "prior_decoder": raw_binding(prior32)}, "SHRINKAGE_FIT_INPUTS_ONLY_KNOWN_PAST_AND_FINITE_CAST_STATIC_PRIOR")
    for name, key in (("decoder_fit64", "decoder_fit_FP64"), ("decoder32", "decoder_deployed_FP32"), ("bias_fit64", "decoder_bias_fit_FP64"), ("bias32", "decoder_bias_FP32")):
        require(metadata["state_arrays"][name] == raw_binding(arrays[key]), "SHRINKAGE_STATE_RAWBYTE_BINDING:" + name)
    independent = independent_augmented_fit(values["history"], values["history_observed"], latent, prior32, rho)
    for key in ("decoder_fit_FP64", "decoder_bias_fit_FP64"):
        prior.close(arrays[key], independent[key], "independent_augmented_LS:" + key, atol=1e-10, rtol=1e-10)
    for key in ("true_observed_counts", "zero_trace_prior_slope_retained", "no_observation_prior_with_current_intercept", "past_metric_valid", "ridge_matrix_condition_valid"):
        require(metadata[key] == independent[key].tolist(), "SHRINKAGE_EXACT_COUNT_TRACE_FALLBACK_DECISION:" + key)
    for key in ("local_covariance_trace", "actual_lambda", "observed_latent_means_per_C", "observed_raw_C_means", "observed_real_past_reconstruction_MSE", "ridge_matrix_condition"):
        prior.close(metadata[key], independent[key], "independent_shrinkage_fit_metadata:" + key, atol=1e-10, rtol=1e-10)
    counts, penalties = independent["true_observed_counts"], independent["actual_lambda"]
    source_stationarity = np.zeros((n, c), np.float64)
    for window in range(n):
        for channel in range(c):
            valid = values["history_observed"][window, :, channel]
            if not valid.any() or not penalties[window, channel]:
                continue
            z, y = latent[window, valid].astype(np.float64), values["history"][window, valid, channel].astype(np.float64)
            zc, yc = z - z.mean(axis=0), y - y.mean()
            beta = arrays["decoder_fit_FP64"][window, :, channel]
            normal = zc.T @ zc / len(z) @ beta + penalties[window, channel] * (beta - prior32[:, channel].astype(np.float64)) - zc.T @ yc / len(z)
            source_stationarity[window, channel] = np.linalg.norm(normal)
    prior.close(metadata["real_stationarity_residual_norm"], source_stationarity, "saved_shrinkage_normal_equation_certificate", atol=1e-10, rtol=1e-10)
    d64 = arrays["decoder_deployed_FP32"].astype(np.float64)
    for key, expected in (("max_abs_decoder_per_window", np.abs(d64).max(axis=(1, 2))), ("decoder_Frobenius_norm_per_window", np.sqrt(np.square(d64).sum(axis=(1, 2)))),
                          ("decoder_operator_norm_per_window", [float(np.sqrt(max(0., np.linalg.eigvalsh(matrix @ matrix.T)[-1]))) for matrix in d64])):
        prior.close(metadata[key], expected, "actual_shrunk_D32_gain:" + key, atol=1e-10, rtol=1e-10)
    return {"window_fits_checked": n, "channel_slots_checked": n * c, "observed_channel_fits_checked": int((counts > 0).sum()),
            "positive_lambda_solver_calls_checked": int((penalties > 0).sum()), "zero_trace_prior_fallbacks_checked": int(independent["zero_trace_prior_slope_retained"].sum()),
            "no_observation_prior_fallbacks_checked": int((counts == 0).sum()), "max_actual_D32_abs": float(np.abs(d64).max()),
            "zero_trace_prior_is_explicit_choice_not_unique_ridge_solution": True, "no_observation_prior_is_not_future_constant_guarantee": True,
            "absolute_decoder_gain_or_future_error_bound_proven": False}


def independent_selection(scores):
    require(set(scores) == set(CANDIDATE_ORDER) and all(type(value) in (float, int) and np.isfinite(value) and value >= 0 for value in scores.values()), "ALL_SEVEN_FINITE_BOLT_TUNE_PRIMARY_SCORES")
    winner = CANDIDATE_ORDER[0]
    for name in CANDIDATE_ORDER[1:]:
        if scores[name] < scores[winner]:
            winner = name
    return winner


def check_events(events, lock_records):
    require([event["sequence"] for event in events] == list(range(len(events))), "CONTIGUOUS_ACTUAL_SHRINKAGE_EVENT_ORDER")
    saved = [event for event in events if event["kind"] == "SELECTION_LOCK_SAVED"]
    require(len(saved) == 4 and {(event["dataset"], event["route"]) for event in saved} == set(lock_records), "ALL_FOUR_SAVED_SELECTION_LOCK_EVENTS")
    for event in saved:
        require(event["binding"] == lock_records[(event["dataset"], event["route"])], "ACTUAL_SELECTION_EVENT_RECORD_BINDING")
    sealed = [event for event in events if event["kind"] == "ALL_FOUR_SELECTION_LOCK_FILES_SAVED"]
    require(len(sealed) == 1 and sealed[0]["files"] == 4 and max(event["sequence"] for event in saved) < sealed[0]["sequence"], "FOUR_LOCKS_SAVED_BEFORE_GLOBAL_DEV_GATE")
    for event in events:
        if event["kind"] in ("DEV_SPLIT_BEGIN", "DEV_SELECTED_RHO_FIT", "DEV_SELECTED_CACHED_CONTROL"):
            require(event["sequence"] > sealed[0]["sequence"], "NO_DEV_CANDIDATE_COMPUTE_BEFORE_ALL_FOUR_LOCK_FILES")
    return {"all_four_lock_files_before_any_DEV_event": True, "event_count": len(events)}


def check_source_contract(records):
    names = {Path(item["path"]).name: item for item in records}
    require(set(names) == {"run_local_decoder_shrinkage_v1.py", "local_decoder_shrinkage_v1.py", "PLAN_LOCAL_DECODER_SHRINKAGE_KO_v1.md", "local_past_decoder_v1.py", "decoder_metric_probe_v1.py", "fixed_decoder_span_v1.py", "run_fit_target_comparison_v1.py", "continuation_runtime_v1.py"}, "EXACT_EXECUTED_SHRINKAGE_SOURCE_SET")
    tree = ast.parse(Path(names["run_local_decoder_shrinkage_v1.py"]["path"]).read_text(encoding="utf-8-sig"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "fit_shrunk_local_decoder"]
    require(len(calls) == 1 and [ast.unparse(node) for node in calls[0].args] == ["values['history']", "values['history_observed']", "context['latent_history']", "context['prior_decoder32']", "rho"], "NO_FUTURE_LABEL_OR_METRIC_PASSED_TO_SHRUNK_DECODER_FIT")
    return {"past_only_fit_call_AST_checked": True, "scientific_producer_math_imported_by_auditor": False}


def score_method(metadata, prediction, values, label):
    require(metadata["status"] == "COMPLETE", "COMPLETE_SHRINKAGE_FORECAST_METHOD:" + label)
    result, checks = {}, 0
    for field, mask in (("scores_all_observed", values["observed"]), ("scores_complete_mask", prior.oracle_mask(values["observed"]))):
        score, count = local_audit.score_checked(metadata[field], prediction, values["target"], mask, values["std"], label + ":" + field)
        result[field], checks = score, checks + count
    return result, checks


def cached_context(condition, local_backend, control_entry, values, backend, check):
    original_state = prior.verified_arrays(condition["state_artifact"], check)
    require(original_state["decoder"].dtype in (np.float32, np.float64) and np.isfinite(original_state["decoder"]).all(), "FINITE_SERIALIZED_STATIC_PRIOR_PRESERVED")
    prior32 = np.ascontiguousarray(original_state["decoder"].astype(np.float32))
    binding = {"stored_prior_dtype": str(original_state["decoder"].dtype),
        "deployment_cast": "ascontiguousarray(stored decoder.astype(float32)); original serialized prior preserved", "deployed_prior": raw_binding(prior32)}
    sources = local_backend["cached_method_sources"]
    require(control_entry["status"] == "COMPLETE" and control_entry["original_state_source"] == condition["state_artifact"]
            and control_entry["local_state_source"] == local_backend["local_state_artifact"] and control_entry["local_forecast_source"] == local_backend["artifact"]
            and control_entry["cached_method_sources"] == sources and control_entry["static_prior_deployment_binding"] == binding
            and control_entry["native_past_byte_exact_before_D_reuse"] == (backend == "NATIVE_TORCH32") and control_entry["state_refit"] is False,
            "EXACT_REUSED_CONTROL_STATE_SOURCE_AND_FINITE_STATIC_PRIOR_CAST")
    cached = {name: prior.verified_arrays(source, check) for name, source in sources.items()}
    reused = prior.verified_arrays(control_entry["artifact"], check)
    accepted = prior.verified_arrays(local_backend["artifact"], check)
    require(set(reused) == set(accepted), "EXACT_REUSED_LOCAL_AND_STATIC_ARRAY_SET")
    for key in reused:
        exact(reused[key], accepted[key], "original_Local_control_artifact_byte_preservation:" + key)
    for key in ("origins", "target", "observed", "raw_last"):
        exact(reused[key], values[key], "reused_same_original_raw_input:" + key)
    for name, arrays in cached.items():
        exact(arrays["latent_history"], reused["latent_history"], "native_seed_past_bytes_before_shared_ridge_D:" + name)
        exact(arrays["origins"], values["origins"], "shrunk_cached_original_origins:" + name)
        exact(arrays["prediction"], reused[name + "_static_prediction"], "shrunk_control_original_static_prediction:" + name)
        exact(arrays["latent_future"], reused[name + "_cached_future"], "shrunk_control_original_frozen_future:" + name)
    return {"cached": cached, "controls": reused, "prior32": prior32, "prior_binding": binding,
            "sources": sources, "original_state": original_state, "control_record": control_entry["artifact"]}


def audit_positive_backend(entry, values, context, backend, rho, check):
    methods = {"BOLT"} if backend == "BOLT_NUMPY" else {"NATIVE_s92601", "NATIVE_s92602"}
    require(entry["status"] == "COMPLETE" and set(entry["methods"]) == methods
            and entry["cached_method_sources"] == context["sources"] and entry["static_prior_deployment_binding"] == context["prior_binding"]
            and entry["native_past_byte_exact_before_D_reuse"] == (backend == "NATIVE_TORCH32"), "COMPLETE_REGISTERED_RIDGE_BACKEND_AND_SHARED_NATIVE_PRIOR")
    state = prior.verified_arrays(entry["state_artifact"], check)
    latent = context["controls"]["latent_history"]
    state_checks = check_state(entry, state, values, latent, context["prior32"], rho)
    arrays = prior.verified_arrays(entry["artifact"], check)
    expected_keys = {"past_prediction_FP32", "latent_history", "origins", "target", "observed", "raw_last"}
    expected_keys |= {name + suffix for name in methods for suffix in ("_prediction", "_cached_future")}
    require(set(arrays) == expected_keys, "EXACT_SHRUNK_CACHED_FORECAST_ARRAY_SET")
    for key in ("origins", "target", "observed", "raw_last"):
        exact(arrays[key], values[key], "same_shrinkage_target_mask_origin_current:" + key)
    exact(arrays["latent_history"], latent, "shrunk_frozen_cached_E_past")
    d32, b32 = state["decoder_deployed_FP32"], state["decoder_bias_FP32"]
    past = np.einsum("nlk,nkc->nlc", latent, d32, optimize=False) + b32[:, None]
    exact(arrays["past_prediction_FP32"], np.ascontiguousarray(past), "shrunk_actual_past_reconstruction_FP32")
    if values["history_observed"].any():
        past_score = prior.scores(past, values["history"], values["history_observed"], values["std"])
        checks = prior.compare_scores(entry["past_reconstruction_scores_observed_only"], past_score, "shrunk_original_observed_past_reconstruction")
    else:
        require(entry["past_reconstruction_scores_observed_only"] == {"status": "NO_OBSERVED_PAST_TARGETS", "normalized": None}, "EXPLICIT_EMPTY_SHRINKAGE_PAST_SCORE")
        past_score, checks = None, 0
    result_methods = {}
    for name, cached in context["cached"].items():
        metadata, prediction = entry["methods"][name], arrays[name + "_prediction"]
        future, last = cached["latent_future"], latent[:, -1]
        exact(arrays[name + "_cached_future"], future, "shrunk_same_original_cached_F_or_head_future:" + name)
        exact(prediction, local_audit.local_forecast(values["raw_last"], future, last, d32), "shrunk_actual_new_FP32_einsum_forecast:" + name)
        require(metadata["artifact"] == entry["artifact"] and metadata["prediction_member"] == name + "_prediction"
                and metadata["new_latent_forward_or_training"] is False
                and metadata["native_head_frozen_intervention_not_reoptimized"] == (backend == "NATIVE_TORCH32"), "SHRUNK_FORECAST_ARTIFACT_FROZEN_NATIVE_INTERVENTION_BINDING")
        scores, count = score_method(metadata, prediction, values, "shrunk:" + name)
        checks += count
        real = values["raw_last"].astype(np.float64)[:, None] + np.einsum("nhk,nkc->nhc", future.astype(np.float64) - last.astype(np.float64)[:, None], d32.astype(np.float64), optimize=False)
        real_scores = prior.scores(real, values["target"], values["observed"], values["std"])
        result_methods[name] = {**scores, "real_arithmetic_scores_all_observed": real_scores,
            "measured_FP32_rounding_normalized_MSE_difference": abs(scores["scores_all_observed"]["normalized"]["mse"] - real_scores["normalized"]["mse"])}
    return {"state_checks": state_checks, "methods": result_methods, "past_reconstruction_scores_observed_only": past_score,
            "score_field_checks": checks, "future_labels_used_for_fit": False}


def audit_result(result_path, meter):
    check = meter.check
    path = Path(result_path).resolve()
    record = {"path": str(path), "sha256": prior.file_sha(path)}
    result = prior.verified_json(record, check)
    if result.get("status") != "COMPLETE":
        return {"status": "NOT_AUDITED_INCOMPLETE_PRODUCER", "diagnostic": record, "producer_status": result.get("status"),
                "completed_producer_counts": {key: value for key, value in result.items() if key.startswith("completed_")},
                "preserved_failed_paths": result.get("failed_paths"), "scientific_assets_opened": False, "full_scope_PASS_claimed": False}
    require(result["new_E_F_native_head_forward_runs"] == 0
            and result["new_E_F_or_native_head_training"] is result["new_backbone_training"] is result["new_GPU_execution"] is result["protected_TEST_accessed"] is False
            and result["future_labels_used_for_decoder_fit"] is result["native_specific_selection"] is result["development_positive_rho_sweep_executed"] is result["whole_selected_deployment_cost_measured"] is False
            and result["same_decoder_side_information_as_static"] is result["strict_K_only_compression"] is result["old_static_or_global_rankK_floor_reused"] is result["independent_holdout"] is False
            and result["TUNE_future_labels_used_for_normal_hyperparameter_selection"] is result["native_uses_same_Bolt_selected_candidate"] is result["all_selection_lock_files_saved_before_any_DEV_candidate_compute"] is result["fit_uses_TUNE_DEV_known_past_at_origin"] is True
            and result["rho_candidates_predeclared"] == list(RHOS) and result["candidate_tie_order"] == list(CANDIDATE_ORDER)
            and result["failed_paths"] == [] and result["full_research_resolution_claimed"] is False and result["cost_status"].startswith("NOT_MEASURED:"), "COMPLETE_PAST_ONLY_RIDGE_FIT_AND_TUNE_ONLY_SHARED_SELECTION_SCOPE")
    local = prior.verified_json(result["local_decoder_diagnostic"], check)
    accepted = prior.verified_json(result["local_decoder_audit"], check)
    producer = prior.verified_json(result["producer"], check)
    require(local["status"] == producer["status"] == "COMPLETE" and accepted["status"] == "PASS"
            and accepted["diagnostic"] == result["local_decoder_diagnostic"] and accepted["producer"] == result["producer"]
            and accepted["audit_source"]["path"] == str(HERE / "audit_local_past_decoder_probe_v1.py")
            and prior.file_sha(accepted["audit_source"]["path"]) == accepted["audit_source"]["sha256"], "EXACT_ACCEPTED_LOCAL_D_SCIENCE_AND_IMMUTABLE_INDEPENDENT_PASS")
    parent_keys = ("producer", "fit_target_audit", "past_constant_diagnostic", "past_constant_audit", "past_constant_whole_cost", "past_constant_whole_cost_audit")
    for key in parent_keys:
        require(result[key] == local[key] == accepted[key], "SAME_SIX_IMMUTABLE_SCIENCE_PREDECESSOR_BINDINGS:" + key)
        prior.verified_json(result[key], check)
    require(all(accepted["totals"].get(key) == value for key, value in {"dataset_split_cells_checked": 6, "backend_states_checked": 24, "method_pairs_checked": 36,
        "window_matrix_fits_checked": 3264, "channel_slots_checked": 44304, "native_seed_D_reuses_checked": 1632, "score_field_checks": 2592}.items()), "ACCEPTED_FULL_LOCAL_D_AUDIT_COVERAGE")
    scope = prior.verified_json(result["scope"], check)
    binding_keys = ("local_decoder_diagnostic", "local_decoder_audit", *parent_keys)
    require(scope["bindings"] == {key: result[key] for key in binding_keys}
            and scope["status"] == "CONTINUING_GOAL_PAST_DECODER_STABILITY_JUDGMENT" and scope["individual_exact_human_approval_claimed"] is False
            and scope["rho_candidates_predeclared"] == list(RHOS) and scope["decoder_fit_uses_TUNE_DEV_known_past"] is True
            and scope["TUNE_future_labels_used_for_normal_hyperparameter_selection"] is scope["all_four_lock_files_before_any_DEV_candidate_compute"] is scope["static_and_original_local_controls_reused_byte_exact"] is True
            and scope["decoder_fit_uses_future_labels"] is scope["DEV_positive_rho_sweep"] is scope["whole_selected_deployment_cost_measured"] is scope["independent_generalization_proven"] is scope["new_E_F_head_forward_training_GPU_TEST"] is False,
            "SHRINKAGE_CONTINUATION_JUDGMENT_AND_NORMAL_TUNE_LABEL_SELECTION_NOT_LABEL_FREE_SELECTION")
    guarded = [record, result["scope"], accepted["audit_source"], *[result[key] for key in binding_keys], *result["source_records"], *result["input_records"], *result["protected_prior_records"]]
    for item in guarded:
        check()
        require(prior.file_sha(item["path"]) == item["sha256"], "UNCHANGED_SHRINKAGE_SOURCE_PRIOR_INPUT_CP_CACHE_MODEL_OR_SELECTION")
    source_contract = check_source_contract(result["source_records"])
    require(scope["plan"] in result["source_records"], "SHRINKAGE_SCOPE_PLAN_EXACT_SOURCE_BINDING")
    receipt = prior.verified_json(result["resource_terminal"], check)
    require(receipt["status"] == "COMPLETE" and receipt["elapsed_or_cumulative_budget_caps"] is None, "ACTUAL_COMPLETE_CPU_SHRINKAGE_RECEIPT")
    locks = {(dataset, route): result["selection_locks"][dataset][route] for dataset in DATASETS for route in ("PCA_PINV", "FIT_TARGET_PINV")}
    for item in locks.values():
        prior.verified_json(item, check)
        guarded.append(item)
    events = check_events(result["execution_events"], locks)
    all_fit_events = [event for event in result["execution_events"] if event["kind"] == "FIT_TUNE_RHO_FIT"]
    expected_fit_events = {(dataset, route, split, backend, label) for dataset in DATASETS for route in ("PCA_PINV", "FIT_TARGET_PINV")
        for split in ("fit", "tune") for backend in ("BOLT_NUMPY", "NATIVE_TORCH32") for label in RHO_LABELS}
    event_tuple = lambda event: (event["dataset"], event["route"], event["split"], event["backend"], event["candidate"])
    require(len(all_fit_events) == 80 and {event_tuple(event) for event in all_fit_events} == expected_fit_events, "EXACT_ALL_EIGHTY_FIT_TUNE_POSITIVE_RHO_BACKEND_EVENTS")
    expected_dev_events = set()
    expected_cached_dev_events = set()
    for (dataset, route), item in locks.items():
        lock = prior.verified_json(item, check)
        label = lock["selected_candidate"]
        if label in RHO_LABELS:
            expected_dev_events |= {(dataset, route, DATASETS[dataset][2], backend, label) for backend in ("BOLT_NUMPY", "NATIVE_TORCH32")}
        else:
            expected_cached_dev_events.add((dataset, route, label))
    dev_events = [event for event in result["execution_events"] if event["kind"] == "DEV_SELECTED_RHO_FIT"]
    cached_dev_events = [event for event in result["execution_events"] if event["kind"] == "DEV_SELECTED_CACHED_CONTROL"]
    require(len(dev_events) == len(expected_dev_events) and {event_tuple(event) for event in dev_events} == expected_dev_events
            and len(cached_dev_events) == len(expected_cached_dev_events)
            and {(event["dataset"], event["route"], event["candidate"]) for event in cached_dev_events} == expected_cached_dev_events, "ONLY_BOLT_LOCKED_CANDIDATE_EVENTS_AT_DEV_NO_UNSELECTED_SWEEP")
    totals = {"dataset_split_cells_checked": 0, "RAW_control_cells_checked": 0, "reused_control_method_pairs_checked": 0,
              "reused_control_prediction_arrays_byte_preserved": 0, "positive_rho_backend_states_checked": 0,
              "FIT_TUNE_positive_rho_prediction_arrays_checked": 0, "selected_DEV_prediction_arrays_checked": 0,
              "positive_rho_forecast_arrays_checked": 0, "window_fits_checked": 0, "channel_slots_checked": 0, "observed_channel_fits_checked": 0,
              "positive_lambda_solver_calls_checked": 0, "zero_trace_prior_fallbacks_checked": 0, "no_observation_prior_fallbacks_checked": 0,
              "native_seed_D_reuses_checked": 0, "selection_locks_checked": 0, "score_field_checks": 0,
              "protected_prior_file_hashes_checked": len(result["protected_prior_records"])}
    report = {"status": "PASS", "diagnostic": record, **{key: result[key] for key in binding_keys}, "scope": result["scope"],
              "source_contract": source_contract, "selection_event_checks": events, "datasets": {}, "selection_locks": {}, "totals": totals,
              "protected_TEST_accessed": False, "new_F_E_head_forward_or_training": False, "new_GPU_execution": False,
              "future_labels_used_for_decoder_fit": False, "TUNE_future_labels_used_for_hyperparameter_selection": True,
              "native_uses_same_Bolt_selected_candidate": True, "DEV_positive_rho_sweep_executed": False,
              "whole_selected_deployment_cost_measured": False, "past_C_coefficients_added_to_decoder_information": True,
              "same_decoder_side_information_as_static": False, "strict_K_only_compression": False, "independent_generalization_proven": False}
    require(set(result["datasets"]) == set(DATASETS), "EXACT_SHRINKAGE_TWO_DATASETS")
    for dataset, (channels, k, development, dev_count) in DATASETS.items():
        row, original = result["datasets"][dataset], producer["datasets"][dataset]
        require(row["K"] == k and row["channels"] == channels and row["development_split"] == development and set(row["splits"]) == {"fit", "tune", development}, "EXACT_SHRINKAGE_C_K_AND_THREE_SPLITS")
        report["datasets"][dataset] = {"K": k, "channels": channels, "splits": {}}
        for split, count in (("fit", 256), ("tune", 64), (development, dev_count)):
            cell = row["splits"][split]
            require(cell["status"] == "COMPLETE" and cell["input_source"] == original["inputs"][split] and set(cell["routes"]) == {"PCA_PINV", "FIT_TARGET_PINV"}, "COMPLETE_SHRINKAGE_ORIGINAL_INPUT_ROUTE_CELL")
            values = prior.verified_arrays(cell["input_source"], check)
            prior.check_raw_split(values, channels=channels, windows=count)
            raw_scores, checks = score_method(cell["RAW"], values["raw_forecast"], values, "shrunk_RAW_control")
            output_cell = {"RAW": raw_scores, "routes": {}}
            report["datasets"][dataset]["splits"][split] = output_cell
            totals["dataset_split_cells_checked"] += 1; totals["RAW_control_cells_checked"] += 1; totals["score_field_checks"] += checks
            for route in ("PCA_PINV", "FIT_TARGET_PINV"):
                entry, condition = cell["routes"][route], original["projections"][route]
                require(entry["status"] == "COMPLETE" and set(entry["control_backends"]) == {"BOLT_NUMPY", "NATIVE_TORCH32"}
                        and set(entry["controls"]) == {"STATIC", "LOCAL"}, "BOTH_CONTROL_DECODERS_AND_BOTH_CACHED_LATENT_BACKENDS")
                output_route = {"controls": {"STATIC": {"methods": {}}, "LOCAL": {"methods": {}}}, "positive_rho_candidates": {}}
                output_cell["routes"][route] = output_route
                contexts = {}
                for backend in ("BOLT_NUMPY", "NATIVE_TORCH32"):
                    old = local["datasets"][dataset]["splits"][split]["routes"][route]["backends"][backend]
                    context = cached_context(condition, old, entry["control_backends"][backend], values, backend, check)
                    contexts[backend] = context
                    for name in context["cached"]:
                        for control, suffix in (("STATIC", "static"), ("LOCAL", "local")):
                            method = entry["controls"][control]["methods"][name]
                            require(method["artifact"] == context["control_record"] and method["prediction_member"] == name + "_" + suffix + "_prediction"
                                    and method["original_cached_prediction_byte_preserved"] is True and method["native_head_frozen_intervention_not_reoptimized"] == (backend == "NATIVE_TORCH32"), "REUSED_CONTROL_SCORE_POINTS_TO_PRESERVED_ARRAY")
                            scored, checked = score_method(method, context["controls"][name + "_" + suffix + "_prediction"], values, "shrunk_reused:" + control + ":" + name)
                            output_route["controls"][control]["methods"][name] = scored
                            totals["score_field_checks"] += checked; totals["reused_control_prediction_arrays_byte_preserved"] += 1
                        _, checked = score_method(entry["controls"]["STATIC"]["methods"][name]["static_einsum_replay_scores"], context["controls"][name + "_static_einsum_replay"], values, "shrunk_original_static_einsum_control")
                        totals["score_field_checks"] += checked; totals["reused_control_method_pairs_checked"] += 1
                if split == development:
                    selection = prior.verified_json(locks[(dataset, route)], check)
                    candidate = selection["selected_candidate"]
                    require(entry["selection_lock"] == locks[(dataset, route)] and entry["selected_candidate"] == candidate
                            and entry["unselected_positive_rho_DEV_candidates_computed"] is False
                            and set(entry["positive_rho_candidates"]) == ({candidate} if candidate in RHO_LABELS else set()), "ONLY_LOCKED_CANDIDATE_COMPUTED_AT_DEV_FOR_SHARED_BOLT_AND_NATIVE")
                else:
                    require(set(entry["positive_rho_candidates"]) == set(RHO_LABELS), "ALL_FIVE_PREDECLARED_POSITIVE_RHOS_ONLY_FIT_TUNE")
                for label, candidate_entry in entry["positive_rho_candidates"].items():
                    rho = dict(zip(RHO_LABELS, RHOS))[label]
                    require(candidate_entry["rho"] == rho and set(candidate_entry["backends"]) == {"BOLT_NUMPY", "NATIVE_TORCH32"} and set(candidate_entry["methods"]) == {"BOLT", "NATIVE_s92601", "NATIVE_s92602"}, "EXACT_RHO_AND_THREE_METHOD_TWO_BACKEND_CANDIDATE")
                    out_candidate = {"rho": rho, "backends": {}, "methods": {}}
                    output_route["positive_rho_candidates"][label] = out_candidate
                    for backend, backend_entry in candidate_entry["backends"].items():
                        require(backend_entry["static_prior_source"] == condition["state_artifact"], "EXACT_SAVED_STATIC_D_PRIOR_SOURCE_FOR_RIDGE")
                        audited = audit_positive_backend(backend_entry, values, contexts[backend], backend, rho, check)
                        out_candidate["backends"][backend] = audited
                        out_candidate["methods"].update(audited["methods"])
                        totals["positive_rho_backend_states_checked"] += 1; totals["score_field_checks"] += audited["score_field_checks"]
                        for field in ("window_fits_checked", "channel_slots_checked", "observed_channel_fits_checked", "positive_lambda_solver_calls_checked", "zero_trace_prior_fallbacks_checked", "no_observation_prior_fallbacks_checked"):
                            totals[field] += audited["state_checks"][field]
                        if backend == "NATIVE_TORCH32":
                            totals["native_seed_D_reuses_checked"] += count
                        for name, method in backend_entry["methods"].items():
                            require(candidate_entry["methods"][name] == method, "CANDIDATE_TOP_METHOD_IDENTICAL_TO_AUDITED_BACKEND_METHOD")
                        predictions = len(audited["methods"])
                        totals["positive_rho_forecast_arrays_checked"] += predictions
                        if split != development:
                            totals["FIT_TUNE_positive_rho_prediction_arrays_checked"] += predictions
                if split == development:
                    selected_source = entry["controls"][candidate]["methods"] if candidate in ("STATIC", "LOCAL") else entry["positive_rho_candidates"][candidate]["methods"]
                    require(entry["selected_methods"] == selected_source and set(selected_source) == {"BOLT", "NATIVE_s92601", "NATIVE_s92602"}, "SAME_BOLT_SELECTED_CONDITION_APPLIED_TO_BOTH_FROZEN_NATIVE_SEEDS")
                    output_route["selected_candidate"] = candidate
                    output_route["selected_methods"] = output_route["controls"][candidate]["methods"] if candidate in ("STATIC", "LOCAL") else output_route["positive_rho_candidates"][candidate]["methods"]
                    totals["selected_DEV_prediction_arrays_checked"] += 3
        report["selection_locks"][dataset] = {}
        for route in ("PCA_PINV", "FIT_TARGET_PINV"):
            selection = prior.verified_json(locks[(dataset, route)], check)
            tune = row["splits"]["tune"]["routes"][route]
            candidates = {**tune["controls"], **tune["positive_rho_candidates"]}
            literal_scores = {name: candidates[name]["methods"]["BOLT"]["scores_all_observed"]["normalized"]["mse"] for name in CANDIDATE_ORDER}
            scored = report["datasets"][dataset]["splits"]["tune"]["routes"][route]
            recalculated_scores = {name: ({**scored["controls"], **scored["positive_rho_candidates"]})[name]["methods"]["BOLT"]["scores_all_observed"]["normalized"]["mse"] for name in CANDIDATE_ORDER}
            winner = independent_selection(literal_scores)
            require(selection["status"] == "LOCKED" and selection["selected_candidate"] == winner == independent_selection(recalculated_scores)
                    and selection["selected_rho"] == dict(zip(RHO_LABELS, RHOS)).get(winner)
                    and selection["TUNE_Bolt_normalized_MSE_by_candidate"] == literal_scores and selection["selected_TUNE_Bolt_normalized_MSE"] == literal_scores[winner]
                    and selection["candidate_tie_order"] == list(CANDIDATE_ORDER) and selection["selection_predictor"] == "BOLT_NUMPY" and selection["selection_split"] == "TUNE"
                    and selection["selection_mask"] == "all_true_observed" and selection["selection_metric"] == "normalized_macro_channel_MSE"
                    and selection["MAE_used_for_selection"] is selection["native_specific_selection"] is selection["development_or_oracle_scores_used_for_selection"] is False
                    and selection["native_uses_same_Bolt_selected_candidate"] is True
                    and selection["dataset"] == dataset and selection["route"] == route
                    and selection["local_decoder_diagnostic"] == result["local_decoder_diagnostic"] and selection["local_decoder_audit"] == result["local_decoder_audit"]
                    and selection["TUNE_input_source"] == original["inputs"]["tune"], "INDEPENDENT_ALL_SEVEN_BOLT_TUNE_MSE_SELECTION_AND_EXACT_FIRST_TIE_RULE")
            refs = {name: {"artifact": candidates[name]["methods"]["BOLT"]["artifact"], "prediction_member": candidates[name]["methods"]["BOLT"]["prediction_member"]} for name in CANDIDATE_ORDER}
            require(selection["TUNE_Bolt_candidate_prediction_records"] == refs, "ALL_SEVEN_SELECTION_SCORES_TIED_TO_ACTUAL_TUNE_PREDICTION_ARRAYS")
            report["selection_locks"][dataset][route] = {"record": locks[(dataset, route)], "selected_candidate": winner, "selected_rho": selection["selected_rho"],
                "TUNE_Bolt_normalized_MSE_by_candidate": recalculated_scores, "recomputed_winner_equal_registered_winner": True,
                "native_same_Bolt_condition_fixed_intervention": True}
            totals["selection_locks_checked"] += 1
    pairs = (("completed_positive_rho_backend_batches", "positive_rho_backend_states_checked"), ("completed_window_matrix_fits", "window_fits_checked"), ("completed_channel_slots", "channel_slots_checked"),
             ("completed_observed_channel_fits", "observed_channel_fits_checked"), ("completed_ridge_solver_calls", "positive_lambda_solver_calls_checked"),
             ("completed_zero_trace_prior_slope_fallbacks", "zero_trace_prior_fallbacks_checked"), ("completed_no_observation_fallbacks", "no_observation_prior_fallbacks_checked"),
             ("native_seed_D_reuses", "native_seed_D_reuses_checked"), ("completed_FIT_TUNE_positive_rho_method_predictions", "FIT_TUNE_positive_rho_prediction_arrays_checked"),
             ("completed_selected_DEV_method_predictions", "selected_DEV_prediction_arrays_checked"), ("completed_reused_control_method_pairs", "reused_control_method_pairs_checked"),
             ("completed_RAW_control_cells", "RAW_control_cells_checked"), ("completed_selection_locks", "selection_locks_checked"))
    for produced, audited in pairs:
        require(result[produced] == totals[audited], "REGISTERED_ACTUAL_SHRINKAGE_COUNTER:" + produced)
    require(result["attempted_positive_rho_backend_batches"] == totals["positive_rho_backend_states_checked"]
            and all(totals[key] == value for key, value in {"dataset_split_cells_checked": 6, "RAW_control_cells_checked": 6, "reused_control_method_pairs_checked": 36,
                "reused_control_prediction_arrays_byte_preserved": 72, "FIT_TUNE_positive_rho_prediction_arrays_checked": 120,
                "selected_DEV_prediction_arrays_checked": 12, "selection_locks_checked": 4}.items())
            and totals["window_fits_checked"] <= 13504 and totals["channel_slots_checked"] <= 187664 and totals["native_seed_D_reuses_checked"] <= 6752
            and totals["positive_lambda_solver_calls_checked"] + totals["zero_trace_prior_fallbacks_checked"] + totals["no_observation_prior_fallbacks_checked"] == totals["channel_slots_checked"], "COMPLETE_SHRINKAGE_SCIENCE_FORECAST_LOCK_AND_ACTUAL_SOLVER_COUNT_SCOPE")
    require(result["original_source_cache_model_and_selection_SHA_unchanged"] is True, "UNCHANGED_SHRINKAGE_SOURCE_CACHE_MODEL_AND_SELECTION_DECLARED")
    for item in guarded:
        check()
        require(prior.file_sha(item["path"]) == item["sha256"], "AUDITED_SHRINKAGE_SOURCE_INPUT_CHECKPOINT_CACHE_SELECTION_UNCHANGED")
    return report


def execute(result_path, output_path):
    from continuation_runtime_v1 import Meter, write_new
    output = Path(output_path).resolve()
    require(not output.exists(), "FRESH_INDEPENDENT_SHRINKAGE_AUDIT_OUTPUT")
    output.parent.mkdir(parents=True, exist_ok=True)
    meter = Meter("audit_shrinkage_" + output.stem, gpu=False)
    source = {"path": str(Path(__file__).resolve()), "sha256": prior.file_sha(__file__)}
    failure = None
    try:
        meter.protected_check()
        initialize_numpy()
        sys.path.insert(0, str(ROOT / "research/smallk_failure_followup_20261014"))
        from prepare_inputs import install_protected_test_guard
        install_protected_test_guard()
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2):
            report = audit_result(result_path, meter)
        meter.protected_check()
        require(prior.file_sha(__file__) == source["sha256"], "INDEPENDENT_SHRINKAGE_AUDIT_SOURCE_UNCHANGED")
        json.dumps(report, allow_nan=False)
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        report = {"status": "FAILED_PRESERVED", "failure": failure, "full_scope_PASS_claimed": False}
    finally:
        report["audit_source"] = source
        report["resource_terminal"] = meter.close(report["status"], failure)
        write_new(output, report)
    print(json.dumps({"status": report["status"], "output": str(output), "totals": report.get("totals"), "failure": failure}, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    raise SystemExit(execute(arguments.result, arguments.output))
