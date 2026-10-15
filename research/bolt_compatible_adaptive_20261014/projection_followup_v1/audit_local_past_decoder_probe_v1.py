"""Independent retained-SVD local-past decoder checks; no producer/model import."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import sys

import audit_projection_results_v1 as prior
import audit_decoder_metric_probe_v1 as exact_audit

np = sys.modules.get("numpy")
HERE, ROOT = prior.HERE, prior.ROOT
require, exact = prior.require, exact_audit.byte_exact
RCOND = 1e-4
DATASETS = {"robin": (17, 4, "val", 30), "beijing_aotizhongxin": (11, 2, "eval", 146)}


def initialize_numpy():
    global np
    exact_audit.initialize_numpy()
    np = sys.modules["numpy"]


def raw_binding(value):
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}


def independent_centered_slope(features, target):
    features, target = np.asarray(features, np.float64), np.asarray(target, np.float64)
    require(features.ndim == 2 and target.shape == (len(features),) and len(features) > 0
            and features.shape[1] > 0 and np.isfinite(features).all() and np.isfinite(target).all(), "FINITE_OBSERVED_LOCAL_CENTRED_SLOPE_INPUT")
    mean_z, mean_y = features.mean(axis=0), float(target.mean())
    design, response = features - mean_z, target - mean_y
    left, singular, right = np.linalg.svd(design, full_matrices=False)
    retained = singular > singular[0] * RCOND if len(singular) else np.zeros(0, bool)
    rank = int(retained.sum())
    slope = right[retained].T @ ((left[:, retained].T @ response) / singular[retained]) if rank else np.zeros(features.shape[1], np.float64)
    bias = mean_y - float(mean_z @ slope)
    gram = design.T @ design
    eigenvalues, vectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, vectors = eigenvalues[order], vectors[:, order]
    roundoff = float(100 * np.finfo(np.float64).eps * max(1., np.square(design).sum()))
    require(eigenvalues[-1] >= -roundoff, "LOCAL_CENTERED_GRAM_NONNEGATIVE_WITH_ROUNDOFF")
    padded_squared = np.zeros(features.shape[1], np.float64)
    padded_squared[:len(singular)] = singular * singular
    require(np.max(np.abs(eigenvalues - padded_squared)) <= roundoff, "INDEPENDENT_SVD_GRAM_ENERGY_CERTIFICATE")
    threshold = float(singular[0] ** 2 * RCOND ** 2) if len(singular) else 0.
    boundary_resolved = bool(not len(singular) or singular[0] == 0 or np.min(np.abs(eigenvalues - threshold)) > roundoff)
    if boundary_resolved and singular[0] > 0:
        require(int(np.count_nonzero(eigenvalues > eigenvalues[0] * RCOND ** 2)) == rank, "MATCHED_LOCAL_SINGULAR_CUTOFF_SQUARED_COVARIANCE_RANK")
    residual = design @ slope - response
    full_gradient = design.T @ residual / len(design)
    retained_gradient = right[retained] @ full_gradient
    condition = float(singular[0] / singular[rank - 1]) if rank else 0.
    require(condition <= 1. / RCOND, "REGISTERED_RETAINED_RELATIVE_CONDITION_LIMIT")
    require(np.linalg.norm(retained_gradient) <= 1e-10 * (1 + np.linalg.norm(design) * np.linalg.norm(response)),
            "RETAINED_SUBSPACE_NORMAL_RESIDUAL_CERTIFICATE")
    null_projection = slope - right[retained].T @ (right[retained] @ slope) if rank else slope
    require(np.linalg.norm(null_projection) <= 1e-10 * (1 + np.linalg.norm(slope)), "TRUNCATED_MINIMUM_NORM_RETAINED_COORDINATES")
    return {"slope": slope, "bias": bias, "mean_z": mean_z, "mean_y": mean_y, "singularvalues": singular,
            "rank": rank, "condition": condition, "reconstruction_MSE": float(np.mean(residual * residual)),
            "Gram_eigenvalues": eigenvalues, "Gram_boundary_resolved": boundary_resolved, "Gram_roundoff_bound": roundoff,
            "full_normal_gradient": full_gradient, "retained_normal_gradient": retained_gradient,
            "retained_projector": right[retained].T @ right[retained] if rank else np.zeros_like(gram)}


def independently_fit_local(history, observed, latent_history):
    require(history.dtype == latent_history.dtype == np.float32 and history.ndim == latent_history.ndim == 3
            and history.shape[1] == 512 and len(history) > 0 and history.shape[:2] == latent_history.shape[:2]
            and 1 <= latent_history.shape[-1] <= history.shape[-1]
            and observed.dtype == np.bool_ and observed.shape == history.shape
            and np.isfinite(history).all() and np.isfinite(latent_history).all(), "FINITE_FIXED_CACHED_E_ORIGIN_PAST_INPUTS")
    n, _, c = history.shape
    k = latent_history.shape[-1]
    decoder, bias = np.zeros((n, k, c), np.float64), np.zeros((n, c), np.float64)
    counts, ranks = observed.sum(axis=1, dtype=np.int64), np.zeros((n, c), np.int64)
    affine_ranks, conditions, mse = np.zeros_like(ranks), np.zeros((n, c), np.float64), np.zeros((n, c), np.float64)
    means_z, means_y = np.zeros((n, c, k), np.float64), np.zeros((n, c), np.float64)
    spectrum = np.zeros((n, c, k), np.float64)
    summaries = []
    for window in range(n):
        for channel in range(c):
            valid = observed[window, :, channel]
            if not counts[window, channel]:
                bias[window, channel] = float(history[window, -1, channel])
                summaries.append({"window": window, "channel": channel, "observed": 0, "rank": 0,
                                  "fallback": "ZERO_SLOPE_CURRENT_C_BIAS_NO_OBSERVED_INFO"})
                continue
            solution = independent_centered_slope(latent_history[window, valid], history[window, valid, channel])
            decoder[window, :, channel], bias[window, channel] = solution["slope"], solution["bias"]
            ranks[window, channel], affine_ranks[window, channel] = solution["rank"], solution["rank"] + 1
            conditions[window, channel], mse[window, channel] = solution["condition"], solution["reconstruction_MSE"]
            means_z[window, channel], means_y[window, channel] = solution["mean_z"], solution["mean_y"]
            spectrum[window, channel, :len(solution["singularvalues"])] = solution["singularvalues"]
            summaries.append({"window": window, "channel": channel, "observed": int(counts[window, channel]), "rank": solution["rank"],
                              "Gram_cutoff_boundary_resolved": solution["Gram_boundary_resolved"],
                              "full_normal_gradient_max_abs_not_required_zero": float(np.abs(solution["full_normal_gradient"]).max()),
                              "retained_normal_gradient_max_abs": float(np.max(np.abs(solution["retained_normal_gradient"]))) if solution["rank"] else 0.})
    require(np.isfinite(decoder).all() and np.isfinite(bias).all(), "FINITE_INDEPENDENT_LOCAL_SLOPES_AND_BIASES")
    return {"decoder_fit_FP64": decoder, "decoder_bias_fit_FP64": bias, "true_observed_counts": counts,
            "centered_slope_ranks": ranks, "affine_model_ranks_with_analytic_intercept": affine_ranks,
            "retained_condition_number": conditions, "observed_reconstruction_MSE_real_fit": mse,
            "observed_latent_means_per_C": means_z, "observed_raw_C_means": means_y,
            "singularvalues_observed_centered_Z": spectrum, "numerical_certificates": summaries}


def local_forecast(current, future, last, decoder):
    require(current.dtype == future.dtype == last.dtype == decoder.dtype == np.float32
            and current.ndim == 2 and future.shape == (len(current), 48, decoder.shape[1])
            and last.shape == (len(current), decoder.shape[1]) and decoder.shape == (len(current), decoder.shape[1], current.shape[1])
            and all(np.isfinite(value).all() for value in (current, future, last, decoder)), "MATCHED_FINITE_FP32_LOCAL_FORECAST_ARRAYS")
    delta = future - last[:, None]
    innovation = np.einsum("nhk,nkc->nhc", delta, decoder, optimize=False)
    prediction = np.ascontiguousarray(current[:, None] + innovation)
    require(np.isfinite(prediction).all(), "LOCAL_FORECAST_FINITE_FP32_RETURN")
    return prediction


def per_origin_output_minimum(values, decoder32):
    complete = values["observed"].all(axis=-1)
    require(complete.any() and decoder32.dtype == np.float32 and decoder32.ndim == 3
            and decoder32.shape[0] == len(values["target"]) and decoder32.shape[-1] == values["target"].shape[-1],
            "NONEMPTY_COMPLETE_C_TARGET_AND_PAST_FIXED_PER_ORIGIN_D32")
    target, current = values["target"].astype(np.float64), values["raw_last"].astype(np.float64)
    prediction = np.full(target.shape, np.nan, np.float64)
    rank_by_origin, residual_energy, numerical_certificate = [], 0., []
    for origin in range(len(target)):
        d = decoder32[origin].astype(np.float64)
        rows = complete[origin]
        singular = np.linalg.svd(d, compute_uv=False)
        d_rank = int(np.sum(singular > (singular[0] * 1e-15))) if len(singular) else 0
        rank_by_origin.append(d_rank)
        if not rows.any():
            numerical_certificate.append({"origin": origin, "complete_positions": 0, "rank_D": d_rank})
            continue
        innovation = target[origin, rows] - current[origin]
        coordinates, _, rank, _ = np.linalg.lstsq(d.T, innovation.T, rcond=1e-15)
        projection = coordinates.T @ d
        residual = innovation - projection
        tolerance = 1e-10 * (1 + np.linalg.norm(innovation) * np.linalg.norm(d))
        normal = residual @ d.T
        require(int(rank) == d_rank and np.max(np.abs(normal)) <= tolerance, "PER_ORIGIN_FUTURE_OUTPUT_LSTSQ_CERTIFICATE")
        prediction[origin, rows] = current[origin] + projection
        residual_energy += float(np.square(residual).sum())
        numerical_certificate.append({"origin": origin, "complete_positions": int(rows.sum()), "rank_D": d_rank,
                                      "normal_residual_max_abs": float(np.max(np.abs(normal))), "normal_absolute_tolerance": float(tolerance)})
    mask = np.broadcast_to(complete[:, :, None], target.shape)
    result = prior.scores(prediction, target, mask, values["std"])
    close = prior.close
    close(result["normalized"]["mse"], residual_energy / (int(complete.sum()) * target.shape[-1]), "per_origin_output_minimum_energy_denominator", atol=1e-10, rtol=1e-10)
    return {"prediction": prediction, "complete": complete, "scores": result, "rank_D_per_origin": rank_by_origin,
            "certificate_per_origin": numerical_certificate,
            "role": "DIAGNOSTIC_ONLY_NEW_PAST_FIXED_ORIGIN_SPECIFIC_D_OUTPUT_SPACE",
            "minimum_only_normalized_MSE_in_each_known_D32_real_rowspace": True,
            "old_static_or_global_rank_K_minimum_reused": False, "RAW_bound_inequality_applied": False,
            "forecastability_or_generalization_proven": False}


def check_local_state(entry, state, values, latent):
    require(set(state) == {"decoder_fit_FP64", "decoder_deployed_FP32", "decoder_bias_fit_FP64", "decoder_bias_FP32", "metadata_json"}, "EXACT_LOCAL_STATE_ARRAY_KEYS")
    metadata = entry["local_fit_metadata"]
    require(json.loads(str(state["metadata_json"].item())) == metadata, "LOCAL_STATE_EMBEDDED_METADATA_BINDING")
    n, _, c = values["history"].shape
    k = latent.shape[-1]
    for key, shape, dtype in (("decoder_fit_FP64", (n, k, c), np.float64), ("decoder_deployed_FP32", (n, k, c), np.float32),
                              ("decoder_bias_fit_FP64", (n, c), np.float64), ("decoder_bias_FP32", (n, c), np.float32)):
        require(state[key].shape == shape and state[key].dtype == dtype and np.isfinite(state[key]).all(), "FINITE_LOCAL_STATE_DIMENSION_DTYPE:" + key)
    exact(state["decoder_deployed_FP32"], state["decoder_fit_FP64"].astype(np.float32), "local_actual_decoder_cast32")
    exact(state["decoder_bias_FP32"], state["decoder_bias_fit_FP64"].astype(np.float32), "local_actual_bias_cast32")
    require(metadata["windows"] == n and metadata["channels"] == c and metadata["K"] == k
            and metadata["length"] == 512 and metadata["horizon"] == 48 and metadata["singular_relative_cut"] == RCOND
            and metadata["equivalent_centered_covariance_eigen_relative_cut"] == RCOND ** 2
            and metadata["future_labels_used_in_local_decoder_fit"] is metadata["TUNE_or_DEV_threshold_selection"] is False
            and metadata["new_E_F_or_native_head_training"] is metadata["strict_K_only_compression"] is metadata["same_decoder_side_information_as_static"] is False
            and metadata["past_C_condition_metadata_added"] is metadata["inference_time_decoder_refitting"] is True
            and metadata["static_fixed_D_floor_reused_as_dynamic_lower_bound"] is metadata["representation_learning_improvement_claimed"] is False,
            "REGISTERED_PAST_LOCAL_TRUNCATION_AND_CHANGED_DECODER_INFORMATION_SCOPE")
    require(metadata["fit_input_bindings"] == {"history": raw_binding(values["history"]), "history_observed": raw_binding(values["history_observed"]),
                                              "cached_latent_history": raw_binding(latent)}, "LOCAL_FIT_ONLY_KNOWN_PAST_C_AND_CACHED_E_BINDINGS")
    for name, key in (("decoder_fit64", "decoder_fit_FP64"), ("decoder32", "decoder_deployed_FP32"),
                      ("bias_fit64", "decoder_bias_fit_FP64"), ("bias32", "decoder_bias_FP32")):
        require(metadata["state_arrays"][name] == raw_binding(state[key]), "LOCAL_STATE_RAWBYTE_HASH:" + name)
    independent = independently_fit_local(values["history"], values["history_observed"], latent)
    for key in ("decoder_fit_FP64", "decoder_bias_fit_FP64"):
        prior.close(state[key], independent[key], "independent_retained_SVD_local_solution:" + key, atol=1e-10, rtol=1e-10)
    for key in ("true_observed_counts", "centered_slope_ranks", "affine_model_ranks_with_analytic_intercept"):
        require(metadata[key] == independent[key].tolist(), "LOCAL_COUNTS_RANKS_AND_INTERCEPT_RANK:" + key)
    for key in ("retained_condition_number", "observed_reconstruction_MSE_real_fit", "observed_latent_means_per_C", "observed_raw_C_means", "singularvalues_observed_centered_Z"):
        prior.close(metadata[key], independent[key], "local_registered_fit_metadata:" + key, atol=1e-10, rtol=1e-10)
    counts, ranks = independent["true_observed_counts"], independent["centered_slope_ranks"]
    for key, expected in (("rank_deficient_slope_channels", ranks < k), ("observed_count_below_two", counts < 2),
                          ("no_observation_current_C_fallback", counts == 0), ("observed_reconstruction_MSE_valid", counts > 0),
                          ("retained_condition_number_valid", ranks > 0)):
        require(metadata[key] == expected.tolist(), "LOCAL_LOW_OBSERVATION_AND_RECONSTRUCTION_VALIDITY:" + key)
    d64 = state["decoder_deployed_FP32"].astype(np.float64)
    operators = [float(np.sqrt(max(0., np.linalg.eigvalsh(d @ d.T)[-1]))) for d in d64]
    for key, expected in (("decoder_Frobenius_norm_per_window", np.sqrt(np.square(d64).sum(axis=(1, 2)))),
                          ("decoder_operator_norm_per_window", operators), ("max_abs_decoder_per_window", np.abs(d64).max(axis=(1, 2)))):
        prior.close(metadata[key], expected, "local_actual_D32_norm:" + key, atol=1e-10, rtol=1e-10)
    valid = ranks > 0
    maximum_condition = float(independent["retained_condition_number"][valid].max()) if valid.any() else None
    if maximum_condition is None:
        require(metadata["max_retained_relative_condition_number"] is None, "NO_VALID_LOCAL_CONDITION_NUMBER")
    else:
        prior.close(metadata["max_retained_relative_condition_number"], maximum_condition, "max_local_retained_condition", atol=1e-10, rtol=1e-10)
    return {"window_matrix_fits_checked": n, "channel_slots_checked": n * c, "observed_channel_solver_calls_checked": int((counts > 0).sum()),
            "no_observation_fallbacks_checked": int((counts == 0).sum()),
            "rank_zero_channels": int((ranks == 0).sum()), "rank_deficient_channels": int((ranks < k).sum()),
            "Gram_cutoff_ambiguous_channel_checks_resolved_by_direct_SVD": sum(not row.get("Gram_cutoff_boundary_resolved", True) for row in independent["numerical_certificates"]),
            "max_abs_actual_D32": float(np.abs(d64).max()), "max_actual_D32_operator_norm": max(operators),
            "max_retained_condition": maximum_condition,
            "full_design_normal_gradient_not_required_zero": True, "minimum_norm_in_retained_centered_coordinates_only": True}


def audit_backend(entry, arrays, local_state, values, original_state, cached, backend):
    expected_methods = {"BOLT"} if backend == "BOLT_NUMPY" else {"NATIVE_s92601", "NATIVE_s92602"}
    require(entry["status"] in ("COMPLETE", "PARTIAL_COMPLETED") and set(entry["methods"]) == set(cached) == expected_methods, "REGISTERED_LOCAL_BACKEND_METHOD_SET")
    methods = {method for method in expected_methods if entry["methods"][method].get("status") != "FAILED_PATH_PRESERVED"}
    require(entry["status"] != "COMPLETE" or methods == expected_methods, "COMPLETE_LOCAL_BACKEND_ALL_METHODS_PRESENT")
    expected_keys = {"latent_history", "past_prediction_FP32", "origins", "target", "observed", "raw_last"}
    expected_keys |= {method + suffix for method in methods for suffix in ("_static_prediction", "_local_prediction", "_static_einsum_replay", "_cached_future")}
    require(set(arrays) == expected_keys, "EXACT_LOCAL_BACKEND_FORECAST_ARRAY_KEYS")
    for key in ("origins", "target", "observed", "raw_last"):
        exact(arrays[key], values[key], "local_same_original_input:" + key)
    first = next(iter(cached.values()))
    exact(arrays["latent_history"], first["latent_history"], "local_original_cached_backend_E_past")
    for artifact in cached.values():
        exact(artifact["latent_history"], arrays["latent_history"], "native_seeds_identical_past_D_shared")
        exact(artifact["origins"], values["origins"], "local_cached_original_origins")
    require(entry["native_cached_past_byte_exact_and_D_shared"] == (backend == "NATIVE_TORCH32"), "LOCAL_NATIVE_D_REUSE_SCOPE")
    d32, bias32 = local_state["decoder_deployed_FP32"], local_state["decoder_bias_FP32"]
    past = np.einsum("nlk,nkc->nlc", arrays["latent_history"], d32, optimize=False) + bias32[:, None]
    exact(arrays["past_prediction_FP32"], np.ascontiguousarray(past), "actual_FP32_local_past_reconstruction")
    past_score, checks = score_checked(entry["past_reconstruction_deployment_scores_observed_only"], past,
        values["history"], values["history_observed"], values["std"], "local_past_deployment_reconstruction_original_observed_only", empty_status="NO_OBSERVED_PAST_TARGETS")
    complete = prior.oracle_mask(values["observed"])
    diagnostic = per_origin_output_minimum(values, d32) if complete.any() else None
    floor, predictions = diagnostic["scores"]["normalized"]["mse"] if diagnostic is not None else None, {}
    original_d = np.broadcast_to(original_state["decoder"].astype(np.float32)[None], d32.shape).copy()
    for method, artifact in cached.items():
        if method not in methods:
            continue
        original, local, replay = (arrays[method + suffix] for suffix in ("_static_prediction", "_local_prediction", "_static_einsum_replay"))
        future, last = artifact["latent_future"], artifact["latent_history"][:, -1]
        exact(original, artifact["prediction"], "preserved_original_static_prediction:" + method)
        exact(arrays[method + "_cached_future"], future, "same_original_frozen_F_or_head_future:" + method)
        exact(local, local_forecast(values["raw_last"], future, last, d32), "local_new_FP32_einsum_prediction:" + method)
        exact(replay, local_forecast(values["raw_last"], future, last, original_d), "original_static_D_new_FP32_einsum_control:" + method)
        metadata = entry["methods"][method]
        scored = {}
        for key, prediction, mask in (("static_scores_all_observed", original, values["observed"]), ("local_scores_all_observed", local, values["observed"]),
                                      ("static_scores_complete_mask", original, complete), ("local_scores_complete_mask", local, complete)):
            actual, checked = score_checked(metadata[key], prediction, values["target"], mask, values["std"], "local_decoder_actual_forecast:" + method + ":" + key)
            checks += checked
            scored[key] = actual
        arithmetic = metadata["static_D_einsum_arithmetic_replay"]
        delta = replay.astype(np.float64) - original.astype(np.float64)
        prior.close(arithmetic["max_abs_vs_original"], float(np.abs(delta).max()), "static_D_arithmetic_control_max_abs", atol=1e-10, rtol=1e-10)
        prior.close(arithmetic["MSE_vs_original"], float(np.mean(delta * delta)), "static_D_arithmetic_control_MSE", atol=1e-10, rtol=1e-10)
        equal = original.dtype == replay.dtype and prior.array_sha(original) == prior.array_sha(replay)
        require(arithmetic["byte_exact"] == equal and metadata["original_selected_prediction_byte_preserved"] is True
                and metadata["native_head_frozen_intervention_not_reoptimized"] == (backend == "NATIVE_TORCH32"), "ORIGINAL_BYTE_PRESERVATION_AND_FROZEN_HEAD_INTERVENTION")
        _, checked = score_checked(arithmetic["scores_all_observed"], replay, values["target"], values["observed"], values["std"], "static_D_new_arithmetic_scores")
        checks += checked
        real = values["raw_last"].astype(np.float64)[:, None] + np.einsum("nhk,nkc->nhc", future.astype(np.float64) - last.astype(np.float64)[:, None], d32.astype(np.float64), optimize=False)
        real_score = prior.scores(real, values["target"], complete, values["std"]) if diagnostic is not None else None
        actual_mse = scored["local_scores_complete_mask"]["normalized"]["mse"] if diagnostic is not None else None
        real_mse = real_score["normalized"]["mse"] if diagnostic is not None else None
        rounding = abs(actual_mse - real_mse) if diagnostic is not None else None
        if diagnostic is not None:
            require(real_mse + 1e-10 >= floor and actual_mse + rounding + 1e-10 >= floor, "NEW_PER_ORIGIN_OUTPUT_FLOOR_ONLY_WITH_MEASURED_ROUNDING")
        predictions[method] = {**scored, "static_D_einsum_max_abs_vs_original": float(np.abs(delta).max()),
            "new_local_real_arithmetic_scores_complete_mask": real_score, "measured_FP32_rounding_MSE_allowance": rounding,
            "local_real_minus_new_origin_specific_minimum_normalized_MSE": real_mse - floor if diagnostic is not None else None,
            "local_actual_minus_new_origin_specific_minimum_normalized_MSE": actual_mse - floor if diagnostic is not None else None}
    dynamic_scope = {key: value for key, value in diagnostic.items() if key not in ("prediction", "complete")} if diagnostic is not None else {"status": "NO_COMPLETE_C_TARGET_POSITIONS", "RAW_bound_inequality_applied": False}
    dynamic_scope["RAW_scores_same_complete_mask"] = prior.scores(values["raw_forecast"], values["target"], complete, values["std"]) if diagnostic is not None else None
    return {"methods": predictions, "score_field_checks": checks, "new_per_origin_output_space_diagnostic": dynamic_scope,
            "past_reconstruction_deployment_scores_observed_only": past_score,
            "same_encoder_cached_past_future_and_original_current_anchor": True, "local_future_labels_for_fit": False}


def score_checked(recorded, prediction, target, observed, scale, label, *, empty_status="NO_OBSERVED_TARGETS"):
    if not observed.any():
        require(recorded == {"status": empty_status, "normalized": None, "physical_per_channel_diagnostic": None}, "EXPLICIT_EMPTY_OBSERVED_SCORE:" + label)
        return None, 0
    actual = prior.scores(prediction, target, observed, scale)
    return actual, prior.compare_scores(recorded, actual, label)


def check_source_contract(source_records):
    names = {Path(item["path"]).name: item for item in source_records}
    require(set(names) == {"run_local_past_decoder_probe_v1.py", "local_past_decoder_v1.py", "PLAN_LOCAL_PAST_DECODER_KO_v1.md", "decoder_metric_probe_v1.py", "fixed_decoder_span_v1.py", "run_fit_target_comparison_v1.py", "run_past_constant_gate_v1.py", "run_past_constant_gate_cost_v1.py"}, "EXACT_LOCAL_SOURCE_RECORD_SET")
    runner = ast.parse(Path(names["run_local_past_decoder_probe_v1.py"]["path"]).read_text(encoding="utf-8-sig"))
    calls = [node for node in ast.walk(runner) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "fit_local_past_decoder"]
    require(len(calls) == 1 and not calls[0].keywords and [ast.unparse(node) for node in calls[0].args] == ["values['history']", "values['history_observed']", "z"], "ONLY_KNOWN_PAST_C_MASK_CACHED_E_PASSED_TO_LOCAL_FIT")
    module = ast.parse(Path(names["local_past_decoder_v1.py"]["path"]).read_text(encoding="utf-8-sig"))
    fits = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "fit_local_past_decoder"]
    require(len(fits) == 1 and [arg.arg for arg in fits[0].args.args] == ["history", "history_observed", "latent_history"], "LOCAL_FIT_HAS_NO_FUTURE_OR_SPLIT_SELECTION_PARAMETERS")
    return {"past_only_fit_call_AST_checked": True, "fit_API_has_no_future_target": True, "source_records_checked": len(names)}


def audit_result(diagnostic_path, meter):
    check = meter.check
    path = Path(diagnostic_path).resolve()
    record = {"path": str(path), "sha256": prior.file_sha(path)}
    diagnostic = prior.verified_json(record, check)
    status = diagnostic.get("status")
    if status not in ("COMPLETE", "PARTIAL_COMPLETED"):
        return {"status": "NOT_AUDITED_INCOMPLETE_PRODUCER", "diagnostic": record, "producer_status": status, "scientific_assets_opened": False}
    binding_names = ("producer", "fit_target_audit", "past_constant_diagnostic", "past_constant_audit", "past_constant_whole_cost", "past_constant_whole_cost_audit")
    parents = {name: prior.verified_json(diagnostic[name], check) for name in binding_names}
    require([parents[name]["status"] for name in binding_names] == ["COMPLETE", "PASS", "COMPLETE", "PASS", "COMPLETE", "PASS"], "EXACT_COMPLETE_PRIOR_PRODUCERS_AND_PASS_AUDITS")
    require(parents["fit_target_audit"]["producer"] == diagnostic["producer"]
            and parents["past_constant_audit"]["diagnostic"] == diagnostic["past_constant_diagnostic"]
            and parents["past_constant_audit"]["producer"] == diagnostic["producer"]
            and parents["past_constant_audit"]["fit_target_audit"] == diagnostic["fit_target_audit"]
            and parents["past_constant_whole_cost_audit"]["cost"] == diagnostic["past_constant_whole_cost"], "EXACT_PREDECESSOR_AUDIT_BINDINGS")
    for parent_name in ("past_constant_whole_cost", "past_constant_whole_cost_audit"):
        parent = parents[parent_name]
        require(parent["producer"] == diagnostic["producer"] and parent["fit_target_audit"] == diagnostic["fit_target_audit"]
                and parent["gate_diagnostic"] == diagnostic["past_constant_diagnostic"] and parent["gate_audit"] == diagnostic["past_constant_audit"], "SAME_COMPLETE_WHOLE_COST_PREDECESSOR_CHAIN")
    code_names = {"fit_target_audit": "audit_fit_target_comparison_v2.py", "past_constant_audit": "audit_past_constant_gate_v1.py", "past_constant_whole_cost_audit": "audit_past_constant_gate_cost_v2.py"}
    for name, filename in code_names.items():
        code = parents[name]["audit_source"]
        require(Path(code["path"]).name == filename and prior.file_sha(code["path"]) == code["sha256"], "IMMUTABLE_ACCEPTED_PREDECESSOR_AUDIT_SOURCE")
    for name, counts in {"fit_target_audit": {"projection_states_checked": 4, "native_fits_checked": 8, "checkpoint_states_checked": 40, "score_field_checks": 1464, "saved_prediction_arrays_checked": 88, "cost_cells_checked": 20, "recorded_timer_samples_checked": 120, "output_subspace_diagnostic_cells_checked": 12},
                         "past_constant_audit": {"dataset_split_cells_checked": 6, "method_pairs_checked": 42, "score_field_checks": 2016},
                         "past_constant_whole_cost_audit": {"cost_cells_checked": 24, "recorded_timer_samples_checked": 144, "physical_gate_prechecks_checked": 2, "paired_gate_timed_output_arrays_checked": 72, "original_selected_forecast_hashes_checked": 36}}.items():
        require(all(parents[name]["totals"].get(key) == count for key, count in counts.items()), "COMPLETE_ACCEPTED_PREDECESSOR_COUNT:" + name)
    require(diagnostic["new_E_F_native_head_forward_runs"] == 0
            and diagnostic["new_E_F_or_native_head_training"] is diagnostic["new_backbone_training"] is diagnostic["future_labels_used_for_local_fit"] is diagnostic["method_checkpoint_threshold_selection_used"] is False
            and diagnostic["protected_TEST_accessed"] is diagnostic["new_GPU_execution"] is diagnostic["whole_local_deployment_cost_measured"] is diagnostic["same_decoder_side_information_as_static"] is diagnostic["strict_K_only_compression"] is False
            and diagnostic["static_D_or_global_rankK_oracle_reused_as_local_lower_bound"] is diagnostic["independent_holdout"] is False
            and diagnostic["inference_time_past_decoder_refitting"] is diagnostic["fit_uses_TUNE_DEV_known_past"] is diagnostic["past_C_condition_metadata_added"] is True
            and diagnostic["expected_method_pairs"] == 36 and diagnostic["maximum_window_matrix_fits"] == 3264 and diagnostic["maximum_channel_slots"] == 44304
            and diagnostic["cost_status"].startswith("NOT_MEASURED_LOCAL:"), "CACHED_CPU_LOCAL_FIT_SCOPE_WITH_NEW_C_PAST_SIDE_INFORMATION")
    scope = prior.verified_json(diagnostic["scope"], check)
    require(scope["bindings"] == {name: diagnostic[name] for name in binding_names}
            and scope["status"] == "CONTINUING_GOAL_CAUSAL_DIAGNOSTIC_JUDGMENT" and scope["individual_exact_human_approval_claimed"] is False
            and scope["fit_uses_TUNE_DEV_past_at_origin"] is scope["inference_time_past_decoder_refitting"] is True
            and scope["future_labels_used_for_fit"] is scope["threshold_or_lambda_search"] is scope["new_E_F_or_head_training"] is False
            and scope["strict_K_only_compression"] is scope["same_decoder_side_information_as_static"] is scope["whole_local_deployment_cost_measured"] is scope["static_and_global_rankK_oracle_lower_bounds_applicable"] is scope["independent_generalization_proven"] is False,
            "CONTINUING_GOAL_JUDGMENT_NOT_EXACT_INDIVIDUAL_METHOD_APPROVAL")
    guarded = [record, diagnostic["scope"], *[diagnostic[name] for name in binding_names], *[parents[name]["audit_source"] for name in code_names], *diagnostic["source_records"], *diagnostic["input_records"], *diagnostic["protected_prior_records"]]
    for item in guarded:
        check()
        require(prior.file_sha(item["path"]) == item["sha256"], "UNCHANGED_LOCAL_SOURCE_PRIOR_CACHE_CP_MODEL_OR_SELECTION_HASH")
    source_contract = check_source_contract(diagnostic["source_records"])
    require(scope["plan"] in diagnostic["source_records"], "LOCAL_SCOPE_PLAN_EXACT_SOURCE_BINDING")
    receipt = prior.verified_json(diagnostic["resource_terminal"], check)
    require(receipt["status"] == status and receipt["elapsed_or_cumulative_budget_caps"] is None, "ACTUAL_LOCAL_CPU_COMPLETION_RECEIPT")
    producer = parents["producer"]
    require(set(diagnostic["datasets"]) == set(DATASETS), "REGISTERED_TWO_LOCAL_DATASETS")
    totals = {"dataset_split_cells_checked": 0, "RAW_control_cells_checked": 0, "backend_states_checked": 0, "completed_backend_batches_checked": 0, "method_pairs_checked": 0,
              "window_matrix_fits_checked": 0, "channel_slots_checked": 0, "observed_channel_solver_calls_checked": 0, "no_observation_fallbacks_checked": 0,
              "native_seed_D_reuses_checked": 0, "static_prediction_arrays_byte_preserved": 0, "local_prediction_arrays_checked": 0,
              "static_einsum_control_arrays_checked": 0, "cached_future_arrays_byte_preserved": 0, "score_field_checks": 0,
              "new_per_origin_output_diagnostic_backend_cells_checked": 0, "new_per_origin_output_inequalities_checked": 0, "RAW_bound_inequalities_checked": 0,
              "protected_prior_file_hashes_checked": len(diagnostic["protected_prior_records"])}
    report = {"status": "PASS" if status == "COMPLETE" else "PARTIAL_AUDITED_NO_FULL_PASS", "diagnostic": record,
              **{name: diagnostic[name] for name in binding_names}, "scope": diagnostic["scope"], "source_contract": source_contract,
              "producer_status": status, "producer_failed_paths": diagnostic["failed_paths"], "datasets": {}, "totals": totals,
              "protected_TEST_accessed": False, "new_F_E_head_forward_or_training": False, "new_GPU_execution": False,
              "whole_local_deployment_cost_measured": False, "past_C_condition_metadata_added": True, "same_decoder_side_information_as_static": False,
              "strict_K_only_compression": False, "fit_uses_TUNE_DEV_known_past_at_origin": True, "future_labels_used_for_local_fit": False,
              "new_per_origin_output_diagnostic_is_future_informed_only_after_past_fit": True, "old_static_or_global_rankK_lower_bound_reused": False,
              "independent_generalization_proven": False, "partial_branches": []}
    for dataset, (channels, k, development, dev_count) in DATASETS.items():
        row, original = diagnostic["datasets"][dataset], producer["datasets"][dataset]
        require(row["K"] == original["K"] == k and row["channels"] == channels and set(row["splits"]) == {"fit", "tune", development}, "LOCAL_REGISTERED_K_C_AND_THREE_SPLITS")
        output_row = {"K": k, "channels": channels, "splits": {}}
        report["datasets"][dataset] = output_row
        for split, count in (("fit", 256), ("tune", 64), (development, dev_count)):
            cell = row["splits"][split]
            if cell.get("status") == "FAILED_PATH_PRESERVED":
                require(status != "COMPLETE", "COMPLETE_LOCAL_PRODUCER_HAS_FAILED_INPUT")
                report["partial_branches"].append({"dataset": dataset, "split": split, "failure": cell})
                continue
            require(cell["input_source"] == original["inputs"][split] and set(cell["routes"]) == {"PCA_PINV", "FIT_TARGET_PINV"}, "LOCAL_ORIGINAL_RAW_SPLIT_AND_ROUTE_BINDING")
            values = prior.verified_arrays(cell["input_source"], check)
            prior.check_raw_split(values, channels=channels, windows=count)
            exact(values["raw_last"], values["history"][:, -1], "local_actual_currentC_is_last_available_past")
            output_cell = {"routes": {}}
            for field, mask in (("RAW_scores_all_observed", values["observed"]), ("RAW_scores_complete_mask", prior.oracle_mask(values["observed"]))):
                actual, checked = score_checked(cell[field], values["raw_forecast"], values["target"], mask, values["std"], field)
                output_cell[field] = actual
                totals["score_field_checks"] += checked
            output_row["splits"][split] = output_cell
            totals["dataset_split_cells_checked"] += 1
            totals["RAW_control_cells_checked"] += 1
            for route in ("PCA_PINV", "FIT_TARGET_PINV"):
                condition = original["projections"][route]
                require(set(cell["routes"][route]["backends"]) == {"BOLT_NUMPY", "NATIVE_TORCH32"}, "BOTH_LOCAL_CACHED_E_BACKENDS_REGISTERED")
                output_route = {"backends": {}}
                output_cell["routes"][route] = output_route
                original_state = prior.verified_arrays(condition["state_artifact"], check)
                for backend, entry in cell["routes"][route]["backends"].items():
                    check()
                    if "local_state_artifact" not in entry:
                        require(status != "COMPLETE" and entry["status"] == "FAILED_PATH_PRESERVED", "MISSING_LOCAL_STATE_REQUIRES_RECORDED_PARTIAL_FAILURE")
                        report["partial_branches"].append({"dataset": dataset, "route": route, "split": split, "backend": backend, "failure": entry})
                        continue
                    sources = {"BOLT": condition["bolt"]["artifacts"][split]} if backend == "BOLT_NUMPY" else {"NATIVE_s" + str(seed): condition["native"][str(seed)]["artifacts"][split] for seed in (92601, 92602)}
                    require(entry["original_state_source"] == condition["state_artifact"] and entry["cached_method_sources"] == sources, "LOCAL_FIXED_E_D_AND_F_OR_HEAD_CACHE_SOURCE_BINDING")
                    cached = {name: prior.verified_arrays(source, check) for name, source in sources.items()}
                    latent = next(iter(cached.values()))["latent_history"]
                    require(latent.shape == (count, 512, k), "LOCAL_EXACT_CACHED_E_N512K")
                    for artifact in cached.values():
                        exact(artifact["latent_history"], latent, "same_native_seed_past_used_for_shared_localD")
                        exact(artifact["origins"], values["origins"], "same_origin_for_fit_and_cached_forecast")
                    state = prior.verified_arrays(entry["local_state_artifact"], check)
                    state_check = check_local_state(entry, state, values, latent)
                    for key in ("window_matrix_fits_checked", "channel_slots_checked", "observed_channel_solver_calls_checked", "no_observation_fallbacks_checked"):
                        totals[key] += state_check[key]
                    totals["backend_states_checked"] += 1
                    if backend == "NATIVE_TORCH32":
                        totals["native_seed_D_reuses_checked"] += count
                    if "artifact" not in entry:
                        require(status != "COMPLETE", "COMPLETE_LOCAL_PRODUCER_HAS_NO_FORECAST_ARTIFACT")
                        output_route["backends"][backend] = {"status": "STATE_ONLY_PARTIAL_AUDITED", "state_checks": state_check}
                        report["partial_branches"].append({"dataset": dataset, "route": route, "split": split, "backend": backend, "failure": entry.get("failure")})
                        continue
                    arrays = prior.verified_arrays(entry["artifact"], check)
                    backend_check = audit_backend(entry, arrays, state, values, original_state, cached, backend)
                    output_route["backends"][backend] = {"status": "PASS" if entry["status"] == "COMPLETE" else "PARTIAL_AUDITED", "state_checks": state_check, **backend_check}
                    pairs = len(backend_check["methods"])
                    totals["completed_backend_batches_checked"] += int(entry["status"] == "COMPLETE")
                    totals["method_pairs_checked"] += pairs
                    for key in ("static_prediction_arrays_byte_preserved", "local_prediction_arrays_checked", "static_einsum_control_arrays_checked", "cached_future_arrays_byte_preserved"):
                        totals[key] += pairs
                    totals["score_field_checks"] += backend_check["score_field_checks"]
                    has_diagnostic = backend_check["new_per_origin_output_space_diagnostic"].get("status") != "NO_COMPLETE_C_TARGET_POSITIONS"
                    totals["new_per_origin_output_diagnostic_backend_cells_checked"] += int(has_diagnostic)
                    totals["new_per_origin_output_inequalities_checked"] += pairs if has_diagnostic else 0
                    if entry["status"] != "COMPLETE":
                        report["partial_branches"].append({"dataset": dataset, "route": route, "split": split, "backend": backend, "failed_methods": sorted(set(sources) - set(backend_check["methods"]))})
    for field, measured in (("completed_method_pairs", "method_pairs_checked"), ("completed_backend_batches", "completed_backend_batches_checked"),
                            ("completed_window_matrix_fits", "window_matrix_fits_checked"), ("completed_channel_solver_calls", "observed_channel_solver_calls_checked"),
                            ("completed_no_observation_fallbacks", "no_observation_fallbacks_checked"), ("native_seed_D_reuses", "native_seed_D_reuses_checked")):
        if status == "COMPLETE":
            require(diagnostic[field] == totals[measured], "LOCAL_RECORDED_AND_AUDITED_ACTUAL_COUNT:" + field)
        else:
            require(diagnostic[field] >= totals[measured], "PARTIAL_AUDITED_COUNT_CANNOT_EXCEED_RECORDED_COMPLETED:" + field)
            if diagnostic[field] != totals[measured]:
                report["partial_branches"].append({"unverifiable_preserved_artifact_count": field,
                    "producer_recorded": diagnostic[field], "independently_checked": totals[measured]})
    if status == "COMPLETE":
        require(not diagnostic["failed_paths"] and diagnostic["attempted_backend_batches"] == 24 and not report["partial_branches"]
                and all(totals[key] == count for key, count in {"dataset_split_cells_checked": 6, "RAW_control_cells_checked": 6, "backend_states_checked": 24, "completed_backend_batches_checked": 24, "method_pairs_checked": 36,
                "window_matrix_fits_checked": 3264, "channel_slots_checked": 44304, "native_seed_D_reuses_checked": 1632, "score_field_checks": 2592,
                "new_per_origin_output_diagnostic_backend_cells_checked": 24, "new_per_origin_output_inequalities_checked": 36, "RAW_bound_inequalities_checked": 0}.items()), "FULL_REGISTERED_LOCAL_STATE_FORECAST_AND_NEW_DYNAMIC_DIAGNOSTIC_AUDIT_COUNTS")
    require(diagnostic["original_source_cache_selection_SHA_unchanged"] is diagnostic["forecast_finite_for_completed_paths"] is True, "LOCAL_PRODUCER_SOURCE_CACHE_SELECTION_PRESERVATION_AND_FINITE_PATHS")
    for item in guarded:
        check()
        require(prior.file_sha(item["path"]) == item["sha256"], "LOCAL_SOURCE_PRIOR_DATA_CACHE_CHECKPOINT_SELECTION_UNCHANGED_AFTER_AUDIT")
    return report


def execute(diagnostic_path, output_path):
    from continuation_runtime_v1 import Meter, write_new
    output = Path(output_path).resolve()
    require(not output.exists(), "FRESH_INDEPENDENT_LOCAL_PAST_DECODER_AUDIT_OUTPUT")
    output.parent.mkdir(parents=True, exist_ok=True)
    meter = Meter("audit_local_past_" + output.stem, gpu=False)
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
            report = audit_result(diagnostic_path, meter)
        meter.protected_check()
        require(prior.file_sha(__file__) == source["sha256"], "INDEPENDENT_LOCAL_AUDIT_SOURCE_UNCHANGED")
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
    parser.add_argument("--diagnostic", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    raise SystemExit(execute(arguments.diagnostic, arguments.output))
