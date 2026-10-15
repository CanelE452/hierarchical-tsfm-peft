"""Independent FIT-target Gram, deployment, checkpoint and cost audit.

The static forecast/cost audit bodies retain the prior independent equations.
No producer projection, trainer, oracle or timing helper is imported.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import audit_projection_results_v1 as prior
import audit_decoder_metric_probe_v1 as decoder_audit

np = sys.modules.get("numpy")
HERE, BASE, ROOT = prior.HERE, prior.BASE, prior.ROOT
DATASETS, SEEDS, CHECKPOINTS = prior.DATASETS, prior.SEEDS, prior.CHECKPOINTS
COST_ROUTES = ("RAW_BOLT", "PCA_PINV_BOLT", "FIT_TARGET_PINV_BOLT", "PCA_PINV_NATIVE", "FIT_TARGET_PINV_NATIVE")
PROJECTION_ROUTES = ("PCA_PINV", "FIT_TARGET_PINV")
require, close, exact = prior.require, prior.close, decoder_audit.byte_exact
file_sha, array_sha = prior.file_sha, prior.array_sha
verified_arrays, verified_json = prior.verified_arrays, prior.verified_json
encode, anchor_prediction = prior.encode, prior.anchor_prediction
forecast_from_latents, native_latent_forecast = prior.forecast_from_latents, prior.native_latent_forecast
select_checkpoint = prior.select_checkpoint
scores, compare_normalized, compare_optional_scores = prior.scores, prior.compare_normalized, prior.compare_optional_scores
oracle_mask = prior.oracle_mask
causal_physical_preprocess, audit_recorded_times = prior.causal_physical_preprocess, prior.audit_recorded_times
compare_parity_record = prior.compare_parity_record


def initialize_numpy():
    global np
    decoder_audit.initialize_numpy()
    np = sys.modules["numpy"]


def raw_binding(value):
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}


def canonical_columns(value):
    result = np.ascontiguousarray(value, np.float64).copy()
    for component in range(result.shape[1]):
        if result[np.argmax(np.abs(result[:, component])), component] < 0:
            result[:, component] *= -1
    return result


def fit_target_gram(target, observed, current, k):
    require(target.dtype == np.float32 and target.ndim == 3 and target.shape[1] == 48
            and observed.dtype == np.bool_ and observed.shape == target.shape
            and current.dtype == np.float32 and current.shape == (len(target), target.shape[-1]), "FIT_TARGET_CURRENT_MASK_SHAPES")
    require(type(k) is int and 1 <= k <= target.shape[-1], "REGISTERED_INTEGER_K")
    complete = observed.all(axis=-1)
    require(complete.any(), "NONEMPTY_COMPLETE_FIT_TARGET_POSITIONS")
    positions = np.argwhere(complete)
    innovation = np.empty((len(positions), target.shape[-1]), np.float64)
    for row, (window, horizon) in enumerate(positions):
        innovation[row] = target[window, horizon].astype(np.float64) - current[window].astype(np.float64)
    require(np.isfinite(innovation).all(), "FINITE_COMPLETE_FIT_INNOVATIONS")
    gram = innovation.T @ innovation
    eigenvalues, vectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, vectors = eigenvalues[order], vectors[:, order]
    total = float(np.square(innovation).sum())
    numerical_bound = float(100 * np.finfo(np.float64).eps * max(1., total))
    require(eigenvalues[-1] >= -numerical_bound and eigenvalues[k - 1] > numerical_bound,
            "RETAINED_FIT_TARGET_K_ENERGY_ABOVE_GRAM_ROUNDOFF")
    retained = float(eigenvalues[:k].sum())
    threshold = eigenvalues[0] * 1e-30
    certain_rank = int(np.sum(eigenvalues > max(threshold, numerical_bound)))
    boundary_unique = k == target.shape[-1] or eigenvalues[k - 1] - eigenvalues[k] > numerical_bound
    projector = vectors[:, :k] @ vectors[:, :k].T
    return {"innovation": innovation, "complete": complete, "gram": gram, "eigenvalues": eigenvalues,
            "vectors": vectors, "projector": projector, "total_energy": total, "retained_energy": retained,
            "gram_roundoff_energy_bound": numerical_bound, "certain_target_rank": certain_rank,
            "rank_at_1e_minus_15_fully_resolved": certain_rank == target.shape[-1], "retained_boundary_unique": boundary_unique}


def verify_inverse_decoder(state):
    inverse = decoder_audit.independent_encoder_inverse(state["encoder"])
    require(inverse["E_rank_lstsq"] == state["encoder"].shape[1], "FULL_RANK_DEPLOYED_FROZEN_E")
    exact(state["decoder"], inverse["D_inverse32"].astype(np.float64), "independent_lstsq_E32_pinv_D32_serialized64")
    exact(state["decoder_bias"], state["center"], "inverse_decoder_bias_original_FIT_past_center")
    return inverse


def native_state_certificate(state_arrays, projection, metadata=None):
    for key in ("center", "encoder", "decoder", "decoder_bias"):
        exact(state_arrays["projection." + key], np.asarray(projection[key], np.float32), "native_same_frozen_E_D_bytes:" + key)
    return prior.native_state_certificate(state_arrays, projection, metadata)


def output_subspace_cell(values, state, cached):
    """Known future diagnostic in this fixed D32 span, after selection only."""
    helper = decoder_audit.span_audit
    solution = helper.independent_span(values["target"], values["observed"], values["raw_last"], state["decoder"], values["std"])
    complete = solution["complete"]
    floor = solution["scores"]["normalized"]["mse"]
    _, geometry = helper.independent_geometry(state["encoder"], state["decoder"])
    methods = {}
    for name, arrays in cached.items():
        exact(arrays["origins"], values["origins"], "output_space_same_original_origins:" + name)
        last = arrays["latent_last"]
        trace = helper.real_trace_and_rounding(arrays["prediction"], arrays["latent_future"], last,
            values["raw_last"], state["decoder"], values["target"], complete, values["std"])
        saved_mse, real_mse = trace["saved_scores"]["normalized"]["mse"], trace["real_scores"]["normalized"]["mse"]
        require(real_mse + 1e-10 >= floor
                and saved_mse + trace["measured_MSE_allowance"] + 1e-10 >= floor, "FIXED_D_REAL_SPAN_BOUND_WITH_MEASURED_FP32_ROUNDOFF:" + name)
        methods[name] = {"saved_prediction_scores_same_complete_mask": trace["saved_scores"],
                         "real_arithmetic_scores_same_complete_mask": trace["real_scores"],
                         "saved_minus_fixed_D_minimum_normalized_MSE": saved_mse - floor,
                         "real_minus_fixed_D_minimum_normalized_MSE": real_mse - floor,
                         "measured_FP32_rounding_MSE_allowance": trace["measured_MSE_allowance"],
                         "FP32_prediction_vs_real_trace_max_abs": trace["prediction_max_abs"],
                         "uses_fixed_D_span": True}
    raw_scores = helper.complete_scores(values["raw_forecast"], values["target"], complete, values["std"])
    return {"role": "DIAGNOSTIC_ONLY_FIXED_D32_OUTPUT_SUBSPACE", "actual_future_labels_used": True,
            "deployment_or_checkpoint_or_branch_selection_used": False,
            "complete_target_positions": int(complete.sum()), "total_target_positions": int(complete.size),
            "fixed_D_minimum_normalized_MSE": floor,
            "fixed_D_real_arithmetic_solution_scores_same_complete_mask": solution["scores"],
            "RAW_scores_same_complete_mask": raw_scores, "RAW_bound_inequality_applied": False,
            "methods_same_complete_mask": methods,
            "E_Y_minus_fixed_D_minimum_normalized_MSE": methods["E_Y"]["saved_minus_fixed_D_minimum_normalized_MSE"],
            "geometry": geometry, "minimum_claim_only_normalized_MSE_in_fixed_D32_real_linear_span": True,
            "physical_MSE_or_MAE_minimum_claimed": False,
            "past_information_predictability_or_generalization_proven": False}


def output_subspace_diagnostics(condition, state, raw, check):
    cells = {}
    for split, values in raw.items():
        bolt = verified_arrays(condition["bolt"]["artifacts"][split], check)
        cached = {"BOLT": {**bolt, "latent_last": bolt["latent_history"][:, -1]}}
        for seed in SEEDS:
            arrays = verified_arrays(condition["native"][str(seed)]["artifacts"][split], check)
            cached["NATIVE_s" + str(seed)] = {**arrays, "latent_last": arrays["latent_history"][:, -1]}
        ey = verified_arrays(condition["oracle"][split]["artifact"], check)
        exact(ey["evaluation_mask"], oracle_mask(values["observed"]), "output_space_E_Y_complete_C_mask")
        cached["E_Y"] = {**ey, "latent_last": bolt["latent_history"][:, -1]}
        cells[split] = output_subspace_cell(values, state, cached)
    return cells


def check_projection_metadata(metadata, state, fit, original_projection=None):
    c, k = fit["history"].shape[-1], state["encoder"].shape[-1]
    require(metadata["channels"] == c and metadata["K"] == k and metadata["length"] == 512
            and metadata["horizon"] == 48 and metadata["TUNE_or_DEV_used_in_projection_or_decoder_fit"] is False
            and metadata["strict_K_only_compression"] is False and metadata["independent_holdout"] is False,
            "SAME_DIMENSIONS_FIT_ONLY_SUPERVISION_PAST_DEPLOYMENT")
    for key in ("center", "encoder", "decoder", "decoder_bias"):
        require(metadata["state_array_sha256"][key] == array_sha(state[key]), "FIT_TARGET_STATE_HASH:" + key)
    flat = fit["history"].astype(np.float64).reshape(-1, c)
    center = flat.mean(axis=0)
    close(state["center"], center, "global_causal_FIT_past_center", atol=1e-10, rtol=1e-10)
    latent = (flat - center) @ state["encoder"]
    covariance = latent.T @ latent / len(flat)
    close(covariance, np.eye(k), "unit_FIT_covariance_no_latent_collapse", atol=1e-7, rtol=0)
    close(metadata["latent_fit_covariance"], covariance, "recorded_FIT_covariance", atol=1e-10, rtol=1e-10)
    inverse = verify_inverse_decoder(state)
    construction = metadata["decoder_construction"]
    require(construction["pinv_rcond"] == 1e-15 and construction["rank_deployed_encoder"] == k
            and construction["encoder_FP32"] == raw_binding(state["encoder"].astype(np.float32))
            and construction["decoder_FP32"] == raw_binding(state["decoder"].astype(np.float32))
            and construction["future_labels_used_to_construct_inverse_decoder"] is False, "SAME_CONDITIONAL_INVERSE_RULE_CONTENT")
    report = {"components": k, "FIT_rows_with_overlap": len(flat), "latent_covariance": covariance.tolist(),
              "decoder_rule": "independent_lstsq(E32_as64,I_C,rcond=1e-15) cast32 serialized64",
              "FIT_whitening_and_inverse_verified": True, "independent_samples_claimed": False}
    if metadata["route"] == "PCA_ENCODER_PINV":
        require(original_projection is not None and metadata["learning_class"] == "UNSUPERVISED_FIT_PAST_PCA"
                and metadata["actual_future_target_used_in_projection_or_decoder_fit"] is False
                and metadata["encoder_center_and_whitening_unchanged"] is True
                and metadata["native_head_retraining_required_for_optimized_comparison"] is True
                and metadata["original_LS_trained_native_head_is_an_optimized_control"] is False, "REUSED_AUDITED_PCA_E_NEW_MATCHED_NATIVE_CONTROL")
        original_state, original_metadata = original_projection
        for key in ("encoder", "center"):
            exact(state[key], original_state[key], "unchanged_original_PCA_" + key)
        require(metadata["source_original_PCA_state_metadata"] == original_metadata, "ORIGINAL_AUDITED_PCA_METADATA_BINDING")
        return {**report, "E_future_labels_used": False, "D_indirect_FIT_future_dependence_via_E": False}
    require(metadata["route"] == "FIT_TARGET_SUBSPACE" and metadata["learning_class"] == "SUPERVISED_FIT_TARGET_SUBSPACE"
            and metadata["actual_future_target_used_in_projection_or_decoder_fit"] is True
            and metadata["target_output_space_is_FIT_only"] is True
            and metadata["method_checkpoint_branch_selection_used_by_this_fit"] is False
            and metadata["eigenvalue_relative_cut"] == 1e-8 and metadata["target_rank_relative_cut"] == 1e-15,
            "REGISTERED_SUPERVISED_FIT_TARGET_OBJECTIVE")
    expected_bindings = {key: raw_binding(value) for key, value in (
        ("history", fit["history"]), ("history_observed", fit["history_observed"]), ("FIT_target", fit["target"]),
        ("FIT_target_observed", fit["observed"]), ("raw_last", fit["raw_last"]), ("origins", fit["origins"]))}
    require(metadata["fit_input_bindings"] == expected_bindings, "ONLY_ORIGINAL_FIT_ARRAY_CONTENT_IN_SPACE_FIT")
    gram = fit_target_gram(fit["target"], fit["observed"], fit["raw_last"], k)
    require(metadata["fit_windows"] == len(fit["history"]) and metadata["fit_positions_with_overlap"] == len(flat)
            and metadata["fit_complete_future_positions"] == int(gram["complete"].sum())
            and metadata["fit_total_future_positions"] == int(gram["complete"].size)
            and metadata["fit_complete_future_positions_per_horizon"] == gram["complete"].sum(axis=0).tolist()
            and metadata["FIT_history_observed_counts"] == fit["history_observed"].sum(axis=(0, 1)).tolist()
            and metadata["FIT_target_observed_counts"] == fit["observed"].sum(axis=(0, 1)).tolist()
            and metadata["FIT_origins_unique"] == len(np.unique(fit["origins"]))
            and metadata["complete_future_mask"] == raw_binding(gram["complete"])
            and metadata["FIT_target_innovation"] == raw_binding(gram["innovation"]), "COMPLETE_FIT_MASK_HORIZON_AND_OVERLAP_COUNTS")
    directions = np.asarray(metadata["target_output_directions"], np.float64)
    require(directions.shape == (c, k), "DECLARED_FIT_TARGET_CK_DIRECTION_SHAPE")
    close(directions.T @ directions, np.eye(k), "FIT_target_orthogonal_basis", atol=1e-10, rtol=1e-10)
    exact(directions, canonical_columns(directions), "FIT_target_canonical_sign")
    captured = float(np.square(gram["innovation"] @ directions).sum())
    close(captured, gram["retained_energy"], "FIT_target_topK_Gram_energy_optimality", atol=1e-10, rtol=1e-10)
    if gram["retained_boundary_unique"]:
        close(directions @ directions.T, gram["projector"], "independent_unique_Gram_topK_subspace", atol=1e-10, rtol=1e-10)
    close(metadata["FIT_target_innovation_total_energy"], gram["total_energy"], "FIT_target_total_energy", atol=1e-10, rtol=1e-10)
    close(metadata["FIT_target_top_K_retained_energy"], captured, "FIT_target_retained_energy", atol=1e-10, rtol=1e-10)
    close(metadata["FIT_target_top_K_retained_fraction"], captured / gram["total_energy"], "FIT_target_retained_fraction", atol=1e-10, rtol=1e-10)
    singular = np.asarray(metadata["target_singularvalues"], np.float64)
    require(singular.shape == (min(len(gram["innovation"]), c),) and np.isfinite(singular).all()
            and np.all(singular >= 0) and np.all(np.diff(singular) <= 0), "FINITE_ORDERED_DECLARED_TARGET_SINGULAR_SPECTRUM")
    close(np.square(singular).sum(), gram["total_energy"], "declared_spectrum_total_Gram_energy", atol=1e-10, rtol=1e-10)
    close(np.square(singular[:k]), gram["eigenvalues"][:k], "declared_retained_spectrum_Gram_eigenvalues", atol=1e-10, rtol=1e-10)
    declared_rank = int(np.sum(singular > singular[0] * 1e-15))
    require(metadata["target_numerical_rank"] == declared_rank >= gram["certain_target_rank"] >= k,
            "DECLARED_SINGULAR_RANK_AND_INDEPENDENT_RETAINED_K_CERTIFICATE")
    if gram["rank_at_1e_minus_15_fully_resolved"]:
        require(declared_rank == c, "FULL_TARGET_RANK_RESOLVED_INDEPENDENTLY_BY_GRAM")
    projected = (flat - center) @ directions
    restricted = projected.T @ projected / len(flat)
    close(metadata["restricted_past_covariance"], restricted, "target_W_restricted_FIT_past_covariance", atol=1e-10, rtol=1e-10)
    values, vectors = np.linalg.eigh(restricted)
    require(values[-1] > 0 and np.count_nonzero(values > values[-1] * 1e-8) == k
            and metadata["whitening_rank"] == k, "NO_REGULARIZATION_OR_K_CHANGE_RESTRICTED_WHITENING_RANK")
    order = np.argsort(values)[::-1]
    expected = canonical_columns(directions @ vectors[:, order] / np.sqrt(values[order])[None])
    close(state["encoder"], expected, "independent_restricted_whitening_encoder", atol=1e-10, rtol=1e-10)
    close(metadata["retained_whitening_eigenvalues"], values[order], "restricted_whitening_eigenvalues", atol=1e-10, rtol=1e-10)
    close(state["encoder"], directions @ (directions.T @ state["encoder"]), "E_stays_within_FIT_target_output_space", atol=1e-10, rtol=1e-10)
    gap = (float((singular[k - 1] - singular[k]) / singular[0]) if k < len(singular) else None)
    require(metadata["target_K_boundary_relative_singular_gap"] == gap, "DECLARED_K_BOUNDARY_GAP")
    return {**report, "E_future_labels_used": True, "D_indirect_FIT_future_dependence_via_E": True,
            "FIT_complete_future_positions": int(gram["complete"].sum()), "FIT_Gram_eigenvalues": gram["eigenvalues"].tolist(),
            "FIT_Gram_roundoff_energy_bound": gram["gram_roundoff_energy_bound"],
            "target_rank_recorded": declared_rank, "target_rank_independently_resolved": gram["rank_at_1e_minus_15_fully_resolved"],
            "independent_certain_target_rank": gram["certain_target_rank"], "retained_topK_subspace_unique": gram["retained_boundary_unique"],
            "target_energy_preservation_is_not_conditional_future_predictability": True}


def audit_condition(condition_json, folder, bridge_npzs, projection_npzs=None, *, check=lambda: None, original_projection=None):
    """Audit one completed route using the runner's exact registered schema."""
    condition = condition_json
    require(Path(condition["state_artifact"]["path"]).resolve().parent == Path(folder).resolve(),
            "CURRENT_RUN_PROJECTION_ARTIFACT_LOCATION")
    require(condition["status"] == "COMPLETE" and condition["bolt"]["status"] == "COMPLETE",
            "COMPLETED_PROJECTION_AND_BOLT_ROUTE_REQUIRED")
    state = projection_npzs if projection_npzs is not None else verified_arrays(condition["state_artifact"], check)
    require(set(state) == {"center", "encoder", "decoder", "decoder_bias", "metadata_json"}, "PROJECTION_NPZ_KEYS")
    for key in ("center", "encoder", "decoder", "decoder_bias"):
        require(state[key].dtype == np.float64, "SERIALIZED_FP64_PROJECTION_STATE:" + key)
    embedded = json.loads(str(state["metadata_json"].item()))
    require(embedded == {key: value for key, value in condition["fit_metadata"].items() if key != "state_array_sha256"},
            "PROJECTION_EMBEDDED_METADATA_BINDING")
    projection = check_projection_metadata(condition["fit_metadata"], state, bridge_npzs["fit"], original_projection)
    report = {"projection": projection, "bolt": {}, "native": {}, "oracle": {},
              "score_field_checks": 0, "saved_prediction_arrays_checked": 0,
              "F_forward_independently_replayed": False}
    bolt_predictions, native_predictions = {}, {}
    for split, values in bridge_npzs.items():
        check()
        saved = verified_arrays(condition["bolt"]["artifacts"][split], check)
        require(set(saved) == {"prediction", "latent_history", "latent_future", "origins"}, "BOLT_SAVED_ARRAY_KEYS")
        exact(saved["origins"], values["origins"], "bolt_origins:" + split)
        latent = encode(values["history"], state["center"], state["encoder"])
        encoded_error = close(saved["latent_history"], latent, "bolt_encoded_past:" + split)
        require(saved["latent_future"].shape == (len(values["history"]), 48, state["encoder"].shape[1])
                and saved["latent_future"].dtype == np.float32 and np.isfinite(saved["latent_future"]).all(),
                "FP32_N48K_ACTUAL_BOLT_FUTURE")
        anchored = forecast_from_latents(values["raw_last"], saved["latent_history"], saved["latent_future"], state["decoder"])
        anchored_error = close(saved["prediction"], anchored, "bolt_current_C_D_anchor:" + split)
        report["score_field_checks"] += compare_optional_scores(condition["bolt"]["scores"][split], saved["prediction"],
                                                                values, "bolt:" + split)
        report["saved_prediction_arrays_checked"] += 1
        report["bolt"][split] = {"encoder_max_abs": encoded_error, "anchored_decoder_max_abs": anchored_error}
        bolt_predictions[split] = saved["prediction"]
    require(set(condition["native"]) == {str(seed) for seed in SEEDS}, "EXACT_TWO_REGISTERED_NATIVE_SEEDS")
    for seed in SEEDS:
        check()
        entry, tune = condition["native"][str(seed)], bridge_npzs["tune"]
        metadata = entry["metadata"]
        require(entry["status"] == "COMPLETE" and metadata["seed"] == seed and metadata["updates"] == 600
                and metadata["batch_size"] == 4 and metadata["checkpoint_updates"] == list(CHECKPOINTS),
                "NATIVE_SEED_UPDATES_BATCH_CHECKPOINT_CONTRACT")
        require(metadata["optimizer"] == "AdamW" and metadata["learning_rate"] == .001
                and metadata["weight_decay"] == 0 and metadata["gradient_clip"] == 1.0
                and metadata["initial_weights_and_biases_zero"] is True
                and metadata["seed_changes_initial_weights"] is False
                and metadata["future_encoding_in_forward"] is False and metadata["new_backbone_training"] is False,
                "REGISTERED_NATIVE_OPTIMIZER_FORWARD_SCOPE")
        order = np.random.default_rng(seed).integers(0, len(bridge_npzs["fit"]["history"]), size=(600, 4))
        require(metadata["order_sha256"] == hashlib.sha256(order.tobytes()).hexdigest(), "NATIVE_FIT_ONLY_SEEDED_MINIBATCH_ORDER")
        require(len(entry["learning_curve"]) == 600, "ACTUAL_600_UPDATE_CURVE_REQUIRED")
        for update, item in enumerate(entry["learning_curve"], 1):
            require(item["update"] == update and np.isfinite(item["loss"]) and item["loss"] >= 0
                    and np.isfinite(item["gradient_norm_before_clip"]) and item["gradient_norm_before_clip"] >= 0,
                    "FINITE_ACTUAL_NATIVE_LEARNING_CURVE")
        checkpoints = entry["checkpoints"]
        require(tuple(item["update"] for item in checkpoints) == CHECKPOINTS, "EXACT_NATIVE_FIVE_CHECKPOINTS")
        checkpoint_rows, checkpoint_arrays, checkpoint_details = [], {}, []
        for checkpoint in checkpoints:
            check()
            arrays = verified_arrays(checkpoint["state_artifact"], check)
            certificate = native_state_certificate(arrays, state, metadata)
            require(certificate["model_state_sha256"] == checkpoint["state_sha256"], "NATIVE_CHECKPOINT_STATE_DIGEST")
            saved = verified_arrays(checkpoint["tune_prediction_artifact"], check)
            require(set(saved) == {"prediction", "origins"}, "CHECKPOINT_TUNE_SAVED_ARRAY_KEYS")
            exact(saved["origins"], tune["origins"], "native_checkpoint_TUNE_origins")
            latent = encode(tune["history"], state["center"], state["encoder"])
            future = native_latent_forecast(latent, arrays["head.weight"], arrays["head.bias"])
            prediction = forecast_from_latents(tune["raw_last"], latent, future, state["decoder"])
            difference = close(saved["prediction"], prediction, "native_checkpoint_forward_u" + str(checkpoint["update"]))
            recalculated = scores(saved["prediction"], tune["target"], tune["observed"], tune["std"])["normalized"]
            report["score_field_checks"] += compare_normalized(checkpoint["tune"], recalculated, "native_checkpoint_TUNE")
            report["saved_prediction_arrays_checked"] += 1
            checkpoint_rows.append({"update": checkpoint["update"], "tune_mse": recalculated["mse"]})
            checkpoint_arrays[checkpoint["update"]] = arrays
            checkpoint_details.append({"update": checkpoint["update"], "forward_max_abs": difference,
                                       "recalculated_tune_mse": recalculated["mse"]})
            if checkpoint["update"] == 0:
                require(np.count_nonzero(arrays["head.weight"]) == np.count_nonzero(arrays["head.bias"]) == 0,
                        "NATIVE_CP0_ZERO_WEIGHTS_BIASES")
                close(saved["prediction"], np.broadcast_to(tune["raw_last"][:, None], tune["target"].shape),
                      "native_CP0_original_C_persistence")
                require(metadata["initial_model_sha256"] == certificate["model_state_sha256"], "NATIVE_INITIAL_MODEL_DIGEST")
        chosen = select_checkpoint(checkpoint_rows)
        winner = checkpoints[chosen["selected_index"]]
        require(metadata["selected_update"] == chosen["selected_update"]
                and metadata["selected_state_sha256"] == winner["state_sha256"]
                and metadata["selected_tune"] == winner["tune"], "TUNE_EARLIEST_NATIVE_SELECTION_CERTIFICATE")
        selected_state = checkpoint_arrays[chosen["selected_update"]]
        split_details, native_predictions[str(seed)] = {}, {}
        for split, values in bridge_npzs.items():
            saved = verified_arrays(entry["artifacts"][split], check)
            require(set(saved) == {"prediction", "latent_history", "latent_future", "origins"}, "NATIVE_SAVED_ARRAY_KEYS")
            exact(saved["origins"], values["origins"], "native_selected_origins:" + split)
            latent_error = close(saved["latent_history"], encode(values["history"], state["center"], state["encoder"]),
                                 "native_frozen_E:" + split)
            future_error = close(saved["latent_future"], native_latent_forecast(saved["latent_history"],
                                 selected_state["head.weight"], selected_state["head.bias"]), "native_selected_temporal_head:" + split)
            anchored = forecast_from_latents(values["raw_last"], saved["latent_history"], saved["latent_future"], state["decoder"])
            prediction_error = close(saved["prediction"], anchored, "native_same_D_current_C_anchor:" + split)
            report["score_field_checks"] += compare_optional_scores(entry["scores"][split], saved["prediction"], values,
                                                                    "native_selected:" + str(seed) + ":" + split)
            report["saved_prediction_arrays_checked"] += 1
            split_details[split] = {"encoder_max_abs": latent_error, "latent_future_max_abs": future_error,
                                   "anchored_decoder_max_abs": prediction_error}
            native_predictions[str(seed)][split] = saved["prediction"]
        report["native"][str(seed)] = {"selection": chosen, "checkpoints": checkpoint_details, "splits": split_details,
                                       "optimizer_updates_independently_replayed": False}
    for split, values in bridge_npzs.items():
        check()
        entry = condition["oracle"][split]
        require(entry["role"] == "diagnostic_only" and entry["selection_used"] is False,
                "ORACLE_NEVER_DEPLOYMENT_OR_SELECTION")
        saved = verified_arrays(entry["artifact"], check)
        require(set(saved) == {"prediction", "latent_future", "evaluation_mask", "origins"}, "ORACLE_SAVED_ARRAY_KEYS")
        exact(saved["origins"], values["origins"], "oracle_origins:" + split)
        mask = oracle_mask(values["observed"])
        exact(saved["evaluation_mask"], mask, "oracle_complete_positions_mask:" + split)
        count = int(mask[:, :, 0].sum())
        require(entry["complete_target_positions"] == count
                and entry["total_target_positions"] == len(values["history"]) * 48, "ORACLE_COMPLETE_POSITION_COUNTS")
        safe_y = np.where(values["observed"], values["target"], np.float32(0.))
        expected_future = encode(safe_y, state["center"], state["encoder"])
        future_error = close(saved["latent_future"], expected_future, "oracle_only_true_future_diagnostic:" + split)
        expected = anchor_prediction(values["history"], values["raw_last"], saved["latent_future"],
                                     state["center"], state["encoder"], state["decoder"])
        prediction_error = close(saved["prediction"], expected, "oracle_same_D_current_C_anchor:" + split)
        comparisons = (("scores", saved["prediction"]), ("raw_scores_same_mask", values["raw_forecast"]),
                       ("bolt_scores_same_mask", bolt_predictions[split]))
        for key, prediction in comparisons:
            report["score_field_checks"] += compare_optional_scores(entry[key], prediction, values,
                                                                    "oracle_mask_comparison:" + key, mask)
        require(set(entry["native_scores_same_mask"]) == {str(seed) for seed in SEEDS}, "ORACLE_TWO_NATIVE_SAME_MASK")
        for seed in SEEDS:
            report["score_field_checks"] += compare_optional_scores(entry["native_scores_same_mask"][str(seed)],
                native_predictions[str(seed)][split], values, "oracle_mask_native:" + str(seed), mask)
        report["saved_prediction_arrays_checked"] += 1
        report["oracle"][split] = {"complete_target_positions": count, "latent_future_max_abs": future_error,
                                    "anchored_decoder_max_abs": prediction_error,
                                    "deployment_or_selection_use": False,
                                    "strict_lower_bound_for_fixed_decoder_or_all_decoders": False}
    perturbation = condition["future_label_perturbation"]
    require(perturbation["target_and_future_mask_changed"] is True and perturbation["bolt_exact_equal"] is True
            and perturbation["native_exact_equal"] == {str(seed): True for seed in SEEDS}, "RECORDED_ACTUAL_FUTURE_LABEL_FIXTURE")
    report["future_label_perturbation"] = {"recorded_scope": perturbation["scope"],
        "producer_record_and_executed_source_checked": True, "independently_replayed_in_audit": False,
        "whole_raw_future_replay_or_general_leakage_proof": False}
    return report, {"BOLT": bolt_predictions, "NATIVE": native_predictions}


def audit_cost_dataset(cost, physical, development, scientific_predictions, check, *, sealed_unix):
    require(cost["status"] == "COMPLETE" and cost["routes"] == list(COST_ROUTES)
            and cost["expected_cells_this_dataset"] == 10 and cost["expected_cells_two_datasets"] == 20
            and len(cost["rows"]) == 10 and cost["warmup_each_route"] == 2 and cost["repeats"] == 6,
            "COMPLETE_MATCHED_TEN_COST_CELLS")
    require(cost["new_training_updates"] == 0 and cost["encoder_decoder_refit_in_cost"] is False
            and cost["protected_TEST_accessed"] is False and cost["independent_holdout"] is False,
            "COST_NO_REFIT_OR_NEW_HOLDOUT")
    normalized_history = causal_physical_preprocess(physical["history_physical"], physical["history_observed"],
                                                   development["mean"], development["std"])
    close(normalized_history, development["history"], "cost_physical_causal_preprocess")
    exact(physical["history_observed"], development["history_observed"], "cost_original_past_mask")
    if "origins" in physical:
        exact(physical["origins"], development["origins"], "cost_physical_bridge_origins")
    for key, actual in (("physical_past", physical["history_physical"]),
                        ("history_observed", physical["history_observed"]), ("FIT_mean", development["mean"]),
                        ("FIT_std", development["std"]), ("origins", development["origins"])):
        declared = cost["input_array_metadata"][key]
        require(declared == {"shape": list(actual.shape), "dtype": str(actual.dtype),
                              "sha256": hashlib.sha256(np.ascontiguousarray(actual).tobytes()).hexdigest()},
                "COST_PHYSICAL_PAST_FIT_SCALER_INPUT_CONTENT:" + key)
    require(cost["order_seed"] == 20261015, "PREDECLARED_COST_ORDER_SEED")
    generator = np.random.default_rng(20261015)
    order = generator.permutation(5)
    expected_schedule = [[COST_ROUTES[index] for index in np.roll(order, -repeat)] for repeat in range(5)]
    expected_schedule.append([COST_ROUTES[index] for index in generator.permutation(5)])
    require(cost["schedule"] == expected_schedule, "ACTUAL_FIVE_ROTATIONS_PLUS_EXTRA_PERMUTATION")
    table = {(row["route"], row["batch"]): row for row in cost["rows"]}
    require(len(table) == 10 and set(table) == {(route, batch) for route in COST_ROUTES for batch in (1, 8)},
            "EXACT_FIVE_ROUTE_TWO_BATCH_COST_MATRIX")
    source = cost["source_lineage"]
    require(file_sha(HERE / "whole_fit_target_cost_v1.py") == source["whole_cost_source_sha256"]
            and file_sha(source["copied_preprocess_source"]) == source["copied_preprocess_source_sha256"]
            and source["legacy_scientific_runtime_imported"] is False, "WHOLE_COST_SOURCE_LINEAGE")
    details = []
    for (route, batch), row in table.items():
        check()
        require(row["status"] == "COMPLETE" and row["dataset"] == cost["dataset"], "COST_ROW_IDENTITY")
        arithmetic = audit_recorded_times(row, table[("RAW_BOLT", batch)])
        require(row["positions"] == [order.index(route) for order in expected_schedule]
                and sorted(row["positions"][:5]) == list(range(5)), "FIRST_FIVE_COST_POSITIONS_BALANCED")
        require(row["origins"] == development["origins"][:batch].tolist(), "SAME_FIRST_B_COST_ORIGINS")
        reference = verified_arrays(row["reference"], check)
        timed = verified_arrays(row["timed_outputs"], check)
        require(set(reference) == {"normalized", "physical", "scientific_normalized", "origins"}
                and set(timed) == {"normalized", "physical", "block", "position", "origins"}, "COST_OUTPUT_ARRAY_KEYS")
        exact(reference["origins"], development["origins"][:batch], "cost_reference_origins")
        exact(timed["origins"], development["origins"][:batch], "cost_timed_origins")
        exact(timed["block"], np.arange(6, dtype=np.int64), "cost_repeat_block")
        exact(timed["position"], np.asarray(row["positions"], np.int64), "cost_position_indices")
        require(reference["normalized"].shape == reference["physical"].shape == (batch, 48, len(development["mean"]))
                and timed["normalized"].shape == timed["physical"].shape == (6, batch, 48, len(development["mean"]))
                and all(value.dtype == np.float32 for value in (reference["normalized"], reference["physical"],
                                                                timed["normalized"], timed["physical"])),
                "FP32_REGISTERED_REFERENCE_TIMED_OUTPUT_SHAPES")
        exact(reference["scientific_normalized"], scientific_predictions[route][:batch], "cost_canonical_scientific_prediction")
        close(reference["normalized"], reference["scientific_normalized"], "cost_reference_scientific_parity")
        exact(reference["physical"], (reference["normalized"] * development["std"] + development["mean"]).astype(np.float32),
              "cost_reference_actual_CPU_physical_inverse")
        scientific_physical = (reference["scientific_normalized"] * development["std"] + development["mean"]).astype(np.float32)
        require(len(row["samples"]) == len(row["parity"]) == len(row["memory"]) == 6, "SIX_ACTUAL_COST_SAMPLES")
        for repeat, sample in enumerate(row["samples"]):
            check()
            require(sample["repeat"] == sample["block"] == repeat and sample["route"] == route
                    and sample["batch"] == batch and sample["dataset"] == cost["dataset"]
                    and sample["position"] == row["positions"][repeat], "SAME_PAIRED_COST_SAMPLE_IDENTITY")
            require(sample["perf_end_ns"] > sample["perf_start_ns"]
                    and sample["unix_end_ns"] >= sample["unix_start_ns"] > sealed_unix * 1e9,
                    "POSITIVE_COST_TIMER_AFTER_SELECTION_SEAL")
            close(sample["seconds"], (sample["perf_end_ns"] - sample["perf_start_ns"]) / 1e9,
                  "cost_recorded_perf_clock_arithmetic", atol=1e-12, rtol=1e-12)
            require(sample["seconds"] == row["seconds"][repeat]
                    and sample["parity"] == row["parity"][repeat]
                    and sample["memory"] == row["memory"][repeat], "COST_ROW_SAMPLE_BINDING")
            require(all(isinstance(value, int) and value >= 0 for value in sample["memory"].values()),
                    "NONNEGATIVE_RECORDED_SHARED_MEMORY")
            sample_path = Path(sample["sample_artifact"]["path"]).with_suffix(".json")
            literal = verified_json({"path": str(sample_path), "sha256": file_sha(sample_path)}, check)
            require(literal == sample, "COST_LITERAL_SAMPLE_JSON_BINDING")
            saved = verified_arrays(sample["sample_artifact"], check)
            require(set(saved) == {"normalized", "physical", "origins"}, "COST_SAMPLE_SAVED_ARRAY_KEYS")
            exact(saved["origins"], development["origins"][:batch], "cost_sample_origins")
            exact(saved["normalized"], timed["normalized"][repeat], "cost_sample_stack_normalized")
            exact(saved["physical"], timed["physical"][repeat], "cost_sample_stack_physical")
            exact(saved["physical"], (saved["normalized"] * development["std"] + development["mean"]).astype(np.float32),
                  "cost_sample_actual_CPU_physical_inverse")
            for key, actual, expected, atol, rtol in (
                    ("normalized", saved["normalized"], reference["normalized"], 1e-5, 2e-6),
                    ("physical", saved["physical"], reference["physical"], 2e-3, 2e-5),
                    ("scientific_normalized", saved["normalized"], reference["scientific_normalized"], 1e-5, 2e-6),
                    ("scientific_physical", saved["physical"], scientific_physical, 2e-3, 2e-5)):
                compare_parity_record(sample["parity"][key], actual, expected, "cost:" + route + ":" + key,
                                      atol=atol, rtol=rtol)
        require(row["memory_counter_reset_inside_module"] is False and row["stage_decomposition"] is False
                and row["optimized_native_implementation_claim"] is False, "RECORDED_WHOLE_SHARED_MEMORY_COST_SCOPE")
        if route.endswith("_NATIVE"):
            require(row["selected_metadata"]["native_seed"] == 92601 and row["selected_metadata"]["head_device"] == "cpu"
                    and row["selected_metadata"]["E_D_frozen"] is True, "REGISTERED_COST_NATIVE_SEED_FROZEN_CPU_ROUTE")
        details.append({"route": route, "batch": batch, **arithmetic})
    return {"rows": details, "cells_checked": len(details), "recorded_timer_samples_checked": len(details) * 6,
            "clock_memory_or_F_forward_independently_remeasured": False,
            "first_five_rounds_balanced_sixth_extra": True,
            "same_first_B_origins_repeated_not_independent_draws": True}



def audit_result(path, meter):
    check = meter.check
    check()
    path = Path(path).resolve()
    require(path.parent.parent == HERE / "runs" and path.name == "RESULT.json", "EXPLICIT_CURRENT_FOLLOWUP_RUN_RESULT")
    record = {"path": str(path), "sha256": file_sha(path)}
    result = verified_json(record, check)
    if result.get("status") != "COMPLETE":
        return {**prior.incomplete_producer_report(result), "producer": record}
    require(result["scientific_execution"] is True and result["protected_TEST_accessed"] is False
            and result["independent_holdout"] is False and result["new_backbone_training"] is False
            and result["native_training_fits"] == 8 and result["projection_fits"] == 2
            and result["reused_projection_states"] == 2 and result["projection_conditions"] == 4
            and result["hard_elapsed_or_cumulative_resource_caps"] is None, "COMPLETED_REGISTERED_TWO_TARGET_FITS_TWO_CONTROLS_EIGHT_NATIVE_SCOPE")
    completeness = {"projection_fits": 2, "reused_projection_states": 2, "projection_conditions": 4,
                    "completed_projection_conditions": 4, "native_training_fits": 8, "cost_cells": 20, "timer_samples": 120}
    require(result["failed_paths"] == [] and result["full_scope_complete_claimed"] is True
            and result["completeness"]["expected"] == result["completeness"]["actual"] == completeness
            and result["projection_fit_attempts"] == 2 and result["native_training_attempts"] == 8,
            "ACTUAL_MATCHED_COMPLETENESS_WITH_NO_FAILED_OR_RETRIED_SCIENTIFIC_PATHS")
    source_projection = verified_json(result["source_projection"], check)
    source_audit = verified_json(result["source_projection_audit"], check)
    require(source_projection["status"] == "COMPLETE" and source_audit["status"] == "PASS"
            and source_audit["producer"] == result["source_projection"]
            and all(source_audit["totals"].get(key) == expected for key, expected in {
                "projection_states_checked": 4, "native_fits_checked": 8, "checkpoint_states_checked": 40,
                "score_field_checks": 1464, "saved_prediction_arrays_checked": 88,
                "cost_cells_checked": 20, "recorded_timer_samples_checked": 120}.items()), "EXACT_ACCEPTED_ORIGINAL_PROJECTION_PASS_GATE")
    require(result["individual_exact_human_method_approval_claimed"] is False
            and result["continuation_scope_is_documented_judgment"] is True
            and result["encoder_fit_uses_FIT_true_future"] == {"PCA_PINV": False, "FIT_TARGET_PINV": True}
            and result["original_LS_native_heads_reused_as_optimized_inverse_controls"] is False
            and result["canonical_native_batch_size"] == 32
            and result["matched_decoder_rule"] == "pinv(E64cast32 interpreted64,rcond=1e-15)cast32", "FRESH_DOCUMENTED_SCOPE_AND_MATCHED_SUPERVISION_CONTRACT")
    for key, expected in (("scope_addendum", HERE / "FIT_TARGET_CONTINUATION_SCOPE_KO_v1.md"),
                          ("plan", HERE / "PLAN_FIT_TARGET_SUBSPACE_KO_v1.md")):
        require(Path(result[key]["path"]).resolve() == expected and result[key]["sha256"] == file_sha(expected), "FIT_TARGET_FRESH_SCOPE_PLAN_BYTES:" + key)
    require(set(result["datasets"]) == set(result["cost"]) == set(DATASETS), "EXACT_TWO_FOLLOWUP_DATASETS")
    for name, expected_path in (("authorization", HERE / "AUTHORIZATION_SCOPE_v1.json"),
                                ("continuation_manifest", HERE / "CONTINUATION_MANIFEST_v1.json")):
        expected_record = result[name]
        require(Path(expected_record["path"]).resolve() == expected_path
                and expected_record["sha256"] == file_sha(expected_path), "SAME_ACTUAL_CONTINUATION_AUTH_MANIFEST:" + name)
    source_checks = 0
    for entry in result["source_records"] + result["source_snapshots"]:
        check()
        require(file_sha(entry["path"]) == entry["sha256"], "PRODUCER_SOURCE_FILE_BYTES_CHANGED")
        source_checks += 1
        if "snapshot_path" in entry:
            require(file_sha(entry["snapshot_path"]) == entry["sha256"], "EXECUTED_SOURCE_SNAPSHOT_CHANGED")
            source_checks += 1
    require(set(result["F_before"]) == {"old_algorithm", "bi_algorithm"}
            and result["F_before"] == result["F_after"] == source_projection["F_before"] == source_projection["F_after"], "DUAL_F_STATE_SIGNATURE_BEFORE_AFTER")
    for name in ("model_binary", "model_config"):
        before, after = result[name + "_before"], result[name + "_after"]
        require(Path(before["path"]).resolve() == Path(after["path"]).resolve()
                and before["sha256"] == after["sha256"] == file_sha(after["path"]), "ACTUAL_F_FILE_BYTES_UNCHANGED:" + name)
        source_checks += 1
    behavior = result["runtime_behavior_contract"]
    require(behavior["evaluation"] is True and behavior["all_backbone_parameters_frozen"] is True
            and behavior["context_length"] == 2048 and behavior["prediction_length"] == 64
            and behavior["input_patch_size"] == behavior["input_patch_stride"] == 16
            and behavior["median_index"] == 4 and behavior["returned_horizon"] == 48
            and behavior["quantiles"] == [.1, .2, .3, .4, .5, .6, .7, .8, .9]
            and behavior["pipeline_predict_not_used"] is True, "EXACT_RECORDED_FROZEN_BOLT_CONFIG")
    require(behavior["normalizer_before"]["class"] == "InstanceNorm"
            and behavior["normalizer_before"]["eps"] == behavior["normalizer_after"]["eps"]
            and behavior["normalizer_before"]["use_arcsinh"] == behavior["normalizer_after"]["use_arcsinh"]
            and set(behavior["normalizer_fixture_forward_parity"]) == {"standardized", "location", "scale"}
            and all(value["exact_equal_including_nan"] is True for value in behavior["normalizer_fixture_forward_parity"].values()),
            "RECORDED_ACTUAL_NORMALIZER_FORWARD_PARITY")
    seal = verified_json(result["selection_lock"], check)
    require(seal["status"] == "SEALED_BEFORE_NEW_DEV_AND_ORACLE" and set(seal["datasets"]) == set(DATASETS),
            "SELECTION_LOCK_BEFORE_DEVELOPMENT_OR_ORACLE")
    receipt = verified_json(result["resource_terminal"], check)
    require(receipt["status"] == "COMPLETE" and receipt["elapsed_or_cumulative_budget_caps"] is None
            and receipt["started_unix"] <= seal["sealed_unix"] <= receipt["ended_unix"], "COMPLETED_ACTUAL_PRODUCER_RESOURCE_RECEIPT")
    report = {"status": "PASS", "producer": record, "datasets": {}, "cost": {}, "output_subspace_diagnostics": {},
              "source_projection": result["source_projection"], "source_projection_audit": result["source_projection_audit"],
              "scope_addendum": result["scope_addendum"], "plan": result["plan"],
              "F_frozen": {"dual_state_signatures_equal": True, "actual_binary_config_bytes_rehashed": True,
                           "normalizer_forward_producer_fixture_record_checked": True,
                           "F_forward_or_model_tensor_hash_independently_replayed": False},
              "totals": {"projection_states_checked": 0, "native_fits_checked": 0,
                         "checkpoint_states_checked": 0, "score_field_checks": 0,
                         "saved_prediction_arrays_checked": 0, "cost_cells_checked": 0, "recorded_timer_samples_checked": 0},
              "source_file_checks": source_checks, "protected_TEST_accessed": False,
              "new_training_or_GPU_execution": False, "independent_generalization_claimed": False,
              "supervised_FIT_target_space_not_unsupervised_or_past_only_training": True,
              "individual_exact_human_method_approval_claimed": False,
              "oracle_is_fixed_E_Y_diagnostic_not_optimal_latent_or_strict_error_floor": True}
    for dataset, (channels, k, development, development_count, parent_name) in DATASETS.items():
        check()
        row = result["datasets"][dataset]
        require(row["status"] == "COMPLETE" and row["K"] == k
                and set(row["inputs"]) == {"fit", "tune", development}
                and set(row["projections"]) == set(PROJECTION_ROUTES), "REGISTERED_DATASET_K_SPLITS_PROJECTIONS")
        parent_path = BASE / "runs" / parent_name / "RESULT.json"
        parent_record = {"path": str(parent_path), "sha256": file_sha(parent_path)}
        parent = verified_json(parent_record, check)
        require(result["source_parent_results"][dataset] == parent_record, "EXACT_RUN_ORIGINAL_PARENT_RESULT_FILE_BINDING")
        require(parent["inputs"] == source_projection["datasets"][dataset]["inputs"], "ACCEPTED_ORIGINAL_PARENT_INPUT_RECORDS")
        require(parent["status"] == "COMPLETE" and parent["dataset"] == dataset, "COMPLETED_ORIGINAL_INPUT_PRODUCER")
        bridge_record = parent["bridge_provenance"]["input_bridge"]
        bridge = verified_json(bridge_record, check)
        require(bridge["status"] == "COMPLETE" and isinstance(bridge["protected_test_guarded"], list)
                and bool(bridge["protected_test_guarded"]),
                "COMPLETE_ORIGINAL_CANONICAL_INPUT_BRIDGE")
        raw, splits = {}, {}
        for split, count in (("fit", 256), ("tune", 64), (development, development_count)):
            require(row["inputs"][split] == parent["inputs"][split], "EXACT_ORIGINAL_INPUT_RECORD_REUSE:" + split)
            raw[split] = verified_arrays(row["inputs"][split], check)
            splits[split] = prior.check_raw_split(raw[split], channels=channels, windows=count)
            require(row["data_contract"][split]["windows"] == count
                    and row["data_contract"][split]["raw_channels"] == channels
                    and row["data_contract"][split]["latent_channels"] == k, "DECLARED_ORIGINAL_SPLIT_DIMENSION_CONTRACT")
            for key in ("mean", "std"):
                exact(raw[split][key], raw["fit"][key], "same_FIT_only_raw_scaler:" + split + ":" + key)
            require(raw[split]["raw_forecast"].shape == raw[split]["target"].shape
                    and raw[split]["raw_forecast"].dtype == np.float32 and np.isfinite(raw[split]["raw_forecast"]).all(),
                    "ORIGINAL_FP32_RAW_BOLT_SAVED_PREDICTION")
            report["totals"]["score_field_checks"] += compare_optional_scores(row["raw_scores"][split], raw[split]["raw_forecast"],
                                                                              raw[split], "raw_reference:" + dataset + ":" + split)
        for split in ("fit", "tune"):
            parity = row["raw_parity"][split]
            require(parity["pass"] is True and parity["atol"] == 1e-5 and parity["rtol"] == 2e-6
                    and np.isfinite(parity["max_abs"]) and parity["max_abs"] >= 0
                    and np.isfinite(parity["rmse"]) and parity["rmse"] >= 0, "RECORDED_ACTUAL_RAW_PARITY")
        dataset_report = {"original_input_producer": parent_record, "original_canonical_input_bridge": bridge_record,
                          "splits": splits, "projections": {},
                          "raw_model_parity_fixture_independently_replayed": False}
        science = {"RAW_BOLT": raw[development]["raw_forecast"]}
        report["output_subspace_diagnostics"][dataset] = {}
        original_PCA_state = verified_arrays(source_projection["datasets"][dataset]["projections"]["PCA"]["state_artifact"], check)
        for route in PROJECTION_ROUTES:
            if route == "PCA_PINV":
                require(row["projections"][route]["source_original_PCA_state"]
                        == source_projection["datasets"][dataset]["projections"]["PCA"]["state_artifact"], "EXACT_REUSED_PCA_SOURCE_STATE_RECORD")
            condition_report, predictions = audit_condition(row["projections"][route], path.parent, raw, check=check,
                original_projection=(original_PCA_state, source_projection["datasets"][dataset]["projections"]["PCA"]["fit_metadata"]))
            current_state = verified_arrays(row["projections"][route]["state_artifact"], check)
            report["output_subspace_diagnostics"][dataset][route] = output_subspace_diagnostics(row["projections"][route], current_state, raw, check)
            dataset_report["projections"][route] = condition_report
            for seed in SEEDS:
                require(seal["datasets"][dataset][route][str(seed)] == condition_report["native"][str(seed)]["selection"]["selected_update"],
                        "SELECTED_NATIVE_MODEL_MATCHES_PRE_DEVELOPMENT_LOCK")
            science[route + "_BOLT"] = predictions["BOLT"][development]
            science[route + "_NATIVE"] = predictions["NATIVE"]["92601"][development]
            report["totals"]["projection_states_checked"] += 1
            report["totals"]["native_fits_checked"] += 2
            report["totals"]["checkpoint_states_checked"] += 10
            for key in ("score_field_checks", "saved_prediction_arrays_checked"):
                report["totals"][key] += condition_report[key]
        physical_candidates = [entry for entry in bridge["artifacts"]
                               if entry["dataset"] == dataset and entry["split"] == development]
        require(len(physical_candidates) == 1, "EXACT_ORIGINAL_DEVELOPMENT_PHYSICAL_BRIDGE_ARTIFACT")
        canonical_physical = prior.bind_canonical_npz(row["physical_bridge"], physical_candidates[0])
        dataset_report["canonical_physical_bridge"] = canonical_physical
        physical = verified_arrays(canonical_physical, check)
        cost = audit_cost_dataset(result["cost"][dataset], physical, raw[development], science, check, sealed_unix=seal["sealed_unix"])
        report["cost"][dataset] = cost
        report["totals"]["cost_cells_checked"] += cost["cells_checked"]
        report["totals"]["recorded_timer_samples_checked"] += cost["recorded_timer_samples_checked"]
        report["datasets"][dataset] = dataset_report
    require(report["totals"]["projection_states_checked"] == 4 and report["totals"]["native_fits_checked"] == 8
            and report["totals"]["checkpoint_states_checked"] == 40
            and report["totals"]["cost_cells_checked"] == 20 and report["totals"]["recorded_timer_samples_checked"] == 120,
            "COMPLETE_REGISTERED_AUDIT_SCOPE_COUNTS")
    require(result["prior_source_results_caches_and_selection_locks_unchanged"] is True
            and bool(result["protected_prior_records"]), "RECORDED_PRIOR_RUN_SOURCES_CACHES_LOCKS_PRESERVATION")
    for old_record in result["protected_prior_records"]:
        check()
        require(file_sha(old_record["path"]) == old_record["sha256"], "PROTECTED_PRIOR_SOURCE_RESULT_CACHE_LOCK_BYTES")
    report["totals"]["target_projection_fits_checked"] = 2
    report["totals"]["reused_PCA_controls_checked"] = 2
    report["totals"]["protected_prior_file_hashes_checked"] = len(result["protected_prior_records"])
    report["totals"]["output_subspace_diagnostic_cells_checked"] = 12
    report["totals"]["fixed_D_real_method_inequalities_checked"] = 48
    report["totals"]["bound_RAW_inequalities_checked"] = 0
    require(file_sha(path) == record["sha256"], "PRODUCER_RESULT_CHANGED_DURING_AUDIT")
    return report


def execute(result_path, output_path):
    from continuation_runtime_v1 import Meter, write_new
    output_path = Path(output_path).resolve()
    require(not output_path.exists(), "NEW_UNIQUE_FIT_TARGET_AUDIT_OUTPUT")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meter = Meter("audit_FIT_target_" + output_path.stem, gpu=False)
    source = {"path": str(Path(__file__).resolve()), "sha256": file_sha(__file__)}
    error = None
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
        require(file_sha(__file__) == source["sha256"], "INDEPENDENT_FIT_TARGET_AUDIT_SOURCE_UNCHANGED")
    except Exception as failure:
        error = {"type": type(failure).__name__, "message": str(failure)}
        report = {"status": "FAILED_PRESERVED", "failure": error, "full_scope_PASS_claimed": False}
    finally:
        report["audit_source"] = source
        report["independent_metric_IO_helper"] = {"path": str(Path(prior.__file__).resolve()), "sha256": file_sha(prior.__file__)}
        report["resource_terminal"] = meter.close(report["status"], error)
        write_new(output_path, report)
    print(json.dumps({"status": report["status"], "output": str(output_path), "totals": report.get("totals"), "failure": error}, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    raise SystemExit(execute(arguments.result, arguments.output))

