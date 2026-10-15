"""JSON-only raw subset/mixed-coordinate fixed-FIT-D comparison report."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATASETS = {"robin": (17, 4, "val"), "beijing_aotizhongxin": (11, 2, "eval")}
SPACES = ("PCA", "FIT_TARGET", "RAW_SUBSET")
SEEDS = ("92601", "92602")
SCALES = ("normalized", "physical_per_channel_diagnostic")
BACKENDS = {"BOLT_NUMPY": ("BOLT",), "NATIVE_TORCH32": ("NATIVE_s92601", "NATIVE_s92602")}
ROUTES = ("RAW_BOLT", "PCA_BOLT", "FIT_TARGET_BOLT", "RAW_SUBSET_BOLT", "PCA_NATIVE_s92601", "PCA_NATIVE_s92602",
          "FIT_TARGET_NATIVE_s92601", "FIT_TARGET_NATIVE_s92602", "RAW_SUBSET_NATIVE_s92601", "RAW_SUBSET_NATIVE_s92602")
INPUTS = ("science", "audit", "cost", "cost_audit")
FIXED_SHA = {"science": "2b271ba9aa6ae359033444b282ddc1becc5d1f60ef063451171e9c93f70db4ff",
    "audit": "e177652ac34414bccfbfc564da820fca0bde5a462922ceb0edaf04c47081e95d",
    "cost": "60eb05ebdd1a7f24d82fa58d4dcab192c5c4684552882d1f61f175fc0585a23c",
    "cost_audit": "e419db2ff01e133a698eec169e546f0ade17f9ae8983c2b5c99d56534259be10"}
COST_FORMAT_REPAIR = {"path": str(HERE/"RAW_SUBSET_COST_AUDITOR_FORMAT_REPAIR_KO_v2.md"),
    "sha256": "8f2bada1aea0da5467763d0d86c19431fdb88083a5c46d7a5e9bd063e0d381b2"}
PRIOR_SHA = "4240cedea826e052b227ef2fba4e16ff52cdcd6db69281844485eca6c0b0bd6b"
FAILED_AUDITS = {
    "RAW_SUBSET_COMPARISON_AUDIT_v2": {"path": str(HERE/"audits/RAW_SUBSET_COMPARISON_AUDIT_v2.json"),
        "sha256": "9697c162c30c108dbcc9f73ef51d28ec37537e56126abc063b29ba1482b28589"},
    "RAW_SUBSET_AUDIT_FAILURE_TRACE_REEXECUTION_v1": {"path": str(HERE/"audits/RAW_SUBSET_AUDIT_FAILURE_TRACE_REEXECUTION_v1.json"),
        "sha256": "d6f49796b94fe0ed38f7be00701aa77308856aaddc199ff06ef467e04a6690ea"},
    "RAW_SUBSET_AUDIT_EXCEPTION_TRACE_v1": {"path": str(HERE/"audits/RAW_SUBSET_AUDIT_EXCEPTION_TRACE_v1.json"),
        "sha256": "c5d40dfa520087107ff7ffb13e10eb606b6303dd55b57c3e82873303f44ffd1e"},
    "RAW_SUBSET_COST_AUDIT_v1": {"path": str(HERE/"audits/RAW_SUBSET_COST_AUDIT_v1.json"),
        "sha256": "42081d93049e78aaa8b313a00dd28f68e292129f9ac3b036832889137c6fb6c2"}}
PRIOR_REPORTS = {
    "matched_projection_v1": "4ad724017706b690e2d19ce4e8bdc326e6663f8ca6b0bf6f157ca4641ef0d3e2",
    "oracle_rank_addendum_v2": "a75784106b241b7228bd09c6b930b94c2a15097c6fd9c9cde466b12cdcc1d4fe",
    "decoder_metric_intervention_v2": "1958073281ea0a9c7cd14bacab2b97b8b4a95063f1015a8c94b25bafe85b5fe5",
    "FIT_target_comparison_v1": "fc89475c76f43f0c30d6a0029d96c138c1997dd510c35600664af4fb5337f77c",
    "past_constant_gate_v1": "4d53cb0b42f882bc636f4f8e57e5d2c42620382274b7268a2f4345ee4a581fd9",
    "local_past_decoder_v1": "cd15f1e0fb4614870713141b631b929696f6d46f92a2cccd76556ced2d1bd74b",
    "shrinkage_matched_native_v1": "c48f0b8d36b882f935e5090f1034406a859df581a43b042357b0b2c043c94e0a"}
SCIENCE_COUNTS = {"subset_selection_fits_checked": 2, "greedy_candidate_scores_checked": 83,
    "global_decoder_states_checked": 12, "source_encoder_states_checked": 6, "source_split_contexts_checked": 18,
    "native_fits_checked": 12, "training_updates_logged_checked": 7200, "checkpoint_states_checked": 60,
    "checkpoint_TUNE_prediction_arrays_checked": 60, "selected_native_split_prediction_arrays_checked": 36,
    "Bolt_split_prediction_arrays_checked": 18, "actual_subset_F_split_provenance_checks": 6,
    "actual_subset_F_model_batch_calls_registered": 103, "reused_mixed_F_cache_lineages_checked": 12,
    "resolved_native_selection_locks_checked": 12, "cached_future_label_perturbation_checks": 12,
    "new_fixed_D_output_space_cells_computed": 36, "new_E_Y_diagnostic_predictions_computed": 36,
    "new_fixed_D_real_method_inequalities_checked": 90, "RAW_bound_inequalities_checked": 0,
    "current_C_persistence_split_baselines_computed": 6, "score_field_checks": 1800}
COST_COUNTS = {"cost_cells_checked": 40, "recorded_timer_samples_checked": 400,
    "physical_backend_prechecks_checked": 12, "physical_first8_windows_checked": 96,
    "native_first8_prediction_checks": 12, "native_selection_locks_checked": 12,
    "subset_selection_locks_checked": 2, "encoder_state_hashes_checked": 6,
    "native_projection_state_hashes_checked": 6, "global_decoder_state_hashes_checked": 12,
    "native_head_state_hashes_checked": 12, "selected_forecast_file_hashes_checked": 54,
    "canonical_physical_bridges_checked": 2}
DISPLAY = {"robin": "Robin", "beijing_aotizhongxin": "Beijing Aotizhongxin"}
SHORT = {"PCA": "PCA", "FIT_TARGET": "Target", "RAW_SUBSET": "Subset"}


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as handle: return hashlib.file_digest(handle, "sha256").hexdigest()


def record(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT/path).resolve()


def read(path):
    require(Path(path).suffix.lower() == ".json", "JSON_METADATA_ONLY")
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def bound(item, expected, label):
    path = resolve(item["path"])
    require(path.suffix.lower() in (".json", ".md", ".py") and path == resolve(expected["path"])
        and item["sha256"] == expected["sha256"] == sha(path), "EXACT_JSON_BINDING:"+label)


def require_counts(value, expected, label):
    require(all(value.get("totals", {}).get(key) == count for key, count in expected.items()), "FULL_AUDIT_COUNTS:"+label)


def load_verified(paths):
    require(set(paths) == set(INPUTS), "FOUR_METADATA_INPUTS_REQUIRED")
    values = {key: read(path) for key, path in paths.items()}
    bindings = {key: record(path) for key, path in paths.items()}
    require(all(FIXED_SHA[key] is not None and bindings[key]["sha256"] == FIXED_SHA[key] for key in INPUTS),
        "REGISTERED_COMPLETE_ACTUAL_SHA_REQUIRED_NOT_YET_OPEN")
    science, audit, cost, cost_audit = (values[name] for name in INPUTS)
    for name, value in values.items():
        require(value.get("status") == ("PASS" if "audit" in name else "COMPLETE"), "ACTUAL_COMPLETE_OR_PASS:"+name)
        require(value.get("protected_TEST_accessed") is False, "PROTECTED_TEST_NOT_ACCESSED:"+name)
    for name in ("audit", "cost_audit"):
        code = values[name]["audit_source"]
        require(resolve(code["path"]).suffix == ".py" and code == record(resolve(code["path"])), "EXACT_EXECUTED_AUDITOR_SOURCE:"+name)
        bindings[name+"_source"] = code
    bound(audit["diagnostic"], bindings["science"], "science independent PASS")
    for value in (cost, cost_audit):
        bound(value["raw_subset_diagnostic"], bindings["science"], "cost same scientific predictions")
        bound(value["raw_subset_audit"], bindings["audit"], "cost same scientific PASS")
        for key in ("source_producer", "source_audit"):
            bound(value[key], science[key], "same original source "+key)
    bound(cost_audit["cost"], bindings["cost"], "new whole cost independent PASS")
    for key in ("source_producer", "source_audit", "plan", "selection_registry"):
        bound(audit[key], science[key], "science audit predecessor "+key)
    require_counts(audit, SCIENCE_COUNTS, "raw subset, matched heads and new output spaces")
    require_counts(cost_audit, COST_COUNTS, "new ten-route whole cost")
    require(science["completeness"]["actual"] == science["completeness"]["expected"] and not science["failed_paths"]
        and science["F_before"] == science["F_after"] and science["all_twelve_locks_before_any_DEV_input_or_prediction"] is True,
        "COMPLETE_FROZEN_TWELVE_CONDITIONS_AND_LOCK_BARRIER")
    require(science["future_labels_used_for_subset_or_decoder_fit"] is False
        and science["FIT_TARGET_E_already_FIT_future_supervised"] is True
        and science["per_origin_decoder_fit"] is False and science["new_backbone_training"] is False
        and science["oracle_used_for_selection"] is False and science["independent_holdout"] is False
        and science["strict_K_only_compression"] is False, "PREDECLARED_SUPERVISION_AND_CAUSAL_DEPLOYMENT_SCOPE")
    require(audit["new_output_space_diagnostics_computed_by_this_auditor"] is True
        and audit["new_output_minimum_only_normalized_MSE_in_each_new_static_D32_real_rowspace"] is True
        and audit["new_oracle_used_for_selection_or_deployment"] is False
        and audit["MAE_or_physical_MSE_minimum_claimed"] is False
        and audit["strict_universal_FP32_output_lower_bound_claimed"] is False
        and audit["old_output_space_floor_reused"] is False, "NEW_FUTURE_KNOWN_REAL_FIXED_D_DIAGNOSTICS_ONLY")
    require(tuple(cost["routes"]) == ROUTES and cost["completed_cost_cells"] == 40
        and cost["recorded_timer_samples"] == cost["completed_cell_timer_samples"] == 400
        and not cost["failed_paths"] and cost["F_before"] == cost["F_after"]
        and cost["native_deployment_cost_measured"] is True and cost["loaded_native_heads_count"] == 12
        and cost["per_origin_decoder_fit"] is False and cost["fixed_FIT_decoder_loaded_before_timer"] is True
        and cost["all_source_file_state_metadata_hashes_outside_timer"] is True
        and cost["new_encoder_subset_decoder_or_CP_selection"] is False and cost["new_training_updates"] == 0,
        "ACTUAL_TEN_ROUTE_STATIC_FIT_D_WHOLE_PIPELINE")
    bound(cost["selection_registry"], science["selection_registry"], "same twelve selected heads registry")
    registry = read(resolve(science["selection_registry"]["path"]))
    require(len(registry["conditions"]) == 12 and registry["status"] == "ALL_TWELVE_RESOLVED_BEFORE_DEV", "TWELVE_NATIVE_SELECTIONS_RESOLVED")
    for item in registry["conditions"]:
        lock = item["lock"]
        bound(cost["native_selection_locks"][item["dataset"]][item["space"]][item["seed"]], lock, "cost exact selected head")
        require(read(resolve(lock["path"]))["status"] == "LOCKED", "FULL_REPORT_REQUIRES_LOCKED_HEAD")
    for dataset in DATASETS:
        bound(cost["subset_selection_locks"][dataset], science["datasets"][dataset]["subset_selection_lock"], "same FIT-only channel selection")
    prior_binding = science["prior_partial"]
    require(prior_binding["sha256"] == PRIOR_SHA, "PRESERVED_ORIGINAL_PARTIAL_SHA")
    prior = read(resolve(prior_binding["path"])); bound(prior_binding, record(resolve(prior_binding["path"])), "prior partial")
    require(prior["status"] == "PARTIAL_COMPLETED" and prior["native_training_fits"] == 8
        and prior["subset_selection_fits"] == prior["actual_subset_F_split_inferences"] == 0,
        "V1_NUMERIC_FAILURE_NOT_SUBSET_PERFORMANCE_RESULT")
    identity = science["mixed_reexecution_identity"]
    require(identity["status"] == "PASS" and identity["mixed_native_conditions_compared"] == 8
        and identity["checkpoint_states_and_TUNE_arrays_compared"] == 40 and identity["selected_native_split_arrays_compared"] == 24
        and identity["mixed_Bolt_split_arrays_compared"] == 12 and identity["global_D_states_compared"] == 8
        and identity["learning_curves_compared"] == 8 and identity["tolerance_used"] is False
        and identity["total_physical_native_fit_attempts_across_v1_v2"] == 20
        and identity["distinct_registered_native_conditions"] == 12, "EIGHT_REPEATED_MIXED_CONDITIONS_EXACT_IDENTITY")
    independent_identity = audit["mixed_reexecution_identity"]
    require(independent_identity["status"] == "PASS" and independent_identity["actual_old_and_new_NPZ_logical_bytes_checked"] is True
        and independent_identity["new_independent_conditions_from_reexecution"] == 0
        and all(independent_identity[key] == identity[key] for key in ("mixed_native_conditions_compared",
            "checkpoint_states_and_TUNE_arrays_compared", "selected_native_split_arrays_compared", "mixed_Bolt_split_arrays_compared",
            "global_D_states_compared", "learning_curves_compared", "total_physical_native_fit_attempts_across_v1_v2", "distinct_registered_native_conditions")),
        "INDEPENDENT_OLD_NEW_LOGICAL_ARRAY_BYTE_IDENTITY")
    diagnostic = read(resolve(science["Gram_failure_diagnostic"]["path"]))
    bound(diagnostic["partial_result"], prior_binding, "Gram failure prior")
    require(diagnostic["FIT_future_targets_read"] is False and diagnostic["TUNE_DEV_read"] is False,
        "GRAM_DIAGNOSIS_ONLY_FIT_PAST")
    for name in ("prior_partial", "Gram_failure_diagnostic", "numeric_repair_addendum"):
        for value in (cost,):
            require(value[name] == science[name], "SAME_REPAIR_PROVENANCE:"+name)
    values.update(prior_partial=prior, Gram_failure_diagnostic=diagnostic)
    bindings.update({name: science[name] for name in ("prior_partial", "Gram_failure_diagnostic", "numeric_repair_addendum")})
    values["preserved_audit_failures"] = []
    for name, binding in FAILED_AUDITS.items():
        bound(binding, record(resolve(binding["path"])), "preserved auditor failure "+name)
        failure = read(resolve(binding["path"]))
        expected_status = "CAPTURED_UNCHANGED_AUDITOR_EXCEPTION_TRACE" if name.endswith("EXCEPTION_TRACE_v1") else "FAILED_PRESERVED"
        require(failure["status"] == expected_status, "PRESERVED_FAILURE_NOT_ACCEPTED_PASS")
        bindings[name] = binding
        values["preserved_audit_failures"].append({"name": name, **binding, **failure,
            "accepted_scientific_PASS": False, "new_forecast_or_training_executed_for_audit": False,
            "repair_scope": "MD addendum verified as exact path and SHA instead of JSON parsing; no numeric or timer change"
                if name == "RAW_SUBSET_COST_AUDIT_v1" else "producer-complete bool and complete-C mask used different variable names in fresh auditor; no formula/tolerance/model change"})
    bound(COST_FORMAT_REPAIR, record(resolve(COST_FORMAT_REPAIR["path"])), "cost auditor Markdown format repair")
    bindings["cost_auditor_format_repair"] = COST_FORMAT_REPAIR
    return values, bindings


def score_rows(base, scores, role):
    return [{**base, "metric_scale": scale, **scores[scale], "role": role,
        "MSE_is_real_fixed_D_minimum": base["method"] == "NEW_FIXED_D_MINIMUM" and scale == "normalized",
        "MAE_is_minimum": False, "physical_MSE_is_minimum": False, "RAW_bound_applied": False} for scale in SCALES]


def collect_tables(values):
    science, audit, cost = (values[name] for name in ("science", "audit", "cost"))
    tables = {name: [] for name in ("COMPARISON", "RELATIVE_TO_RAW", "GREEDY_SELECTION", "GREEDY_CANDIDATES",
        "GREEDY_RESPONSE_SCORES", "GLOBAL_DECODER_STATES", "GLOBAL_DECODER_CHANNELS", "CHECKPOINTS", "LEARNING_CURVE",
        "NATIVE_SELECTION", "F_CACHE_AND_SUBSET_INFERENCE", "NEW_FIXED_D_DIAGNOSTIC", "NEW_FIXED_D_GEOMETRY",
        "WHOLE_COST", "WHOLE_COST_SAMPLES", "REGISTERED_ADOPTION", "EXECUTION_COST_AND_COUNTS", "FAILURE_AND_REPAIR", "AUDIT_FAILURE_PRESERVATION")}
    for dataset, (channels, k, dev) in DATASETS.items():
        row, checked = science["datasets"][dataset], audit["datasets"][dataset]
        selection_binding = row["subset_selection_lock"]
        selection = read(resolve(selection_binding["path"]))
        selected_names = selection["selected_channel_names"]
        for stage in selection["selection_metadata"]["stages"]:
            base = {"dataset": dataset, "stage": stage["stage"], "chosen_original_index": stage["selected_original_index"]}
            tables["GREEDY_SELECTION"].append({**base, **{key: stage[key] for key in stage if key != "candidates"},
                "candidate_count": len(stage["candidates"]), "final_selected_indices": selection["selected_indices"],
                "final_selected_names": selected_names, "final_greedy_order": selection["selection_order"],
                "selection_source": selection["FIT_source"], "future_forecast_objective_used": False})
            for candidate in stage["candidates"]:
                tables["GREEDY_CANDIDATES"].append({**base, **candidate,
                    "winner": candidate["candidate_original_index"] == stage["selected_original_index"], "exact_tie_rule": "lowest_original_index"})
                for channel in range(channels):
                    tables["GREEDY_RESPONSE_SCORES"].append({**base, "candidate_original_index": candidate["candidate_original_index"],
                        "response_channel_zero_based": channel, "candidate_indices": candidate["indices"],
                        "MSE": candidate["per_channel_MSE"][channel], "observed_count": candidate["true_observed_counts"][channel],
                        "retained_rank": candidate["retained_centered_design_rank"][channel],
                        "retained_condition": candidate["retained_centered_design_condition"][channel],
                        "centered_singularvalues": candidate["centered_design_singularvalues"][channel]})
        for split in ("fit", "tune", dev):
            base = {"dataset": dataset, "space": "NA", "split": split, "backend": "RAW"}
            for method, origin in (("RAW_BOLT", row["raw_scores"][split]), ("CURRENT_C_PERSISTENCE", checked["current_C_persistence_scores"][split])):
                for mask, key in (("all_original_observed", "scores_all_observed"), ("all_C_complete", "scores_complete_mask")):
                    tables["COMPARISON"] += score_rows({**base, "method": method, "mask": mask}, origin[key], "saved_Raw" if method == "RAW_BOLT" else "currentC_repeated_no_forecaster_and_not_timed")
        for space in SPACES:
            condition, verified = row["spaces"][space], checked["spaces"][space]
            for backend in BACKENDS:
                entry, check = condition["backends"][backend], verified["backends"][backend]
                meta = entry["metadata"]; base = {"dataset": dataset, "space": space, "backend": backend}
                require(meta["singular_relative_cutoff"] == 1e-4 and meta["future_TUNE_DEV_or_F_scores_used"] is False,
                    "SAME_OBSERVED_PAST_GLOBAL_D_FITTING_RULE")
                tables["GLOBAL_DECODER_STATES"].append({**base, "C": channels, "K": k, "state": entry["state_artifact"],
                    "metadata": meta, "independent_fit": check["independent_fit"],
                    "deployed_decoder_FP32_values": check["deployed_decoder_FP32_values"],
                    "deployed_decoder_max_abs": check["deployed_decoder_max_abs"],
                    "deployed_decoder_operator_norm": check["deployed_decoder_operator_norm"],
                    "deployed_decoder_Frobenius_norm": check["deployed_decoder_Frobenius_norm"],
                    "encoder_already_FIT_future_supervised": space == "FIT_TARGET", "per_origin_fit": False})
                channel_keys = ("true_observed_counts", "retained_centered_design_rank", "retained_centered_design_condition",
                    "centered_design_singularvalues", "observed_past_reconstruction_MSE_FP64", "fallback_reason", "learned_slope",
                    "zero_centered_predictor_variance", "constant_observed_response", "no_observed_responses", "one_observed_response",
                    "rank_deficient_slope", "slope_identified_in_retained_full_K_space", "centered_lstsq_fit_executed")
                for channel in range(channels):
                    tables["GLOBAL_DECODER_CHANNELS"].append({**base, "response_channel_zero_based": channel,
                        **{key: meta[key][channel] for key in channel_keys}, "constant_response_is_normal_fit": True})
            for split in ("fit", "tune", dev):
                base = {"dataset": dataset, "space": space, "split": split}
                for method in ("BOLT", "NATIVE_s92601", "NATIVE_s92602"):
                    backend = "BOLT_NUMPY" if method == "BOLT" else "NATIVE_TORCH32"
                    origin = condition["Bolt"][split] if method == "BOLT" else condition["native"][method[-5:]]["scores"][split]
                    for mask, key in (("all_original_observed", "scores_all_observed"), ("all_C_complete", "scores_complete_mask")):
                        tables["COMPARISON"] += score_rows({**base, "backend": backend, "method": method, "mask": mask}, origin[key],
                            "fixedE_actual_F_or_fresh_native_actualC_objective_fixedFIT_D")
                    raw = row["raw_scores"][split]["scores_all_observed"]["normalized"]
                    actual = origin["scores_all_observed"]["normalized"]
                    persist = checked["current_C_persistence_scores"][split]["scores_all_observed"]["normalized"]
                    tables["RELATIVE_TO_RAW"].append({**base, "backend": backend, "method": method,
                        "MSE": actual["mse"], "MAE": actual["mae"], "RAW_MSE": raw["mse"], "RAW_MAE": raw["mae"],
                        "MSE_minus_RAW": actual["mse"]-raw["mse"], "MSE_ratio_to_RAW": actual["mse"]/raw["mse"],
                        "MAE_ratio_to_RAW": actual["mae"]/raw["mae"], "currentC_persistence_MSE": persist["mse"],
                        "MSE_minus_currentC_persistence": actual["mse"]-persist["mse"], "independent_generalization_claimed": False})
                bolt = condition["Bolt"][split]
                tables["F_CACHE_AND_SUBSET_INFERENCE"].append({**base, "mixed_F_cache_reused": bolt["mixed_F_cache_reused"],
                    "new_K_F_inference_executed": bolt.get("new_K_F_inference_executed", False),
                    "original_F_cache": bolt.get("original_F_cache"), "full_C_slice_diagnostic": bolt.get("full_C_slice_diagnostic"),
                    "full_C_forecast_slice_used_as_new_inference": False, "artifact": bolt["artifact"]})
                for backend in BACKENDS:
                    diagnostic = verified["new_output_space_diagnostics"][split][backend]
                    require(diagnostic["minimum_only_normalized_MSE"] is True and diagnostic["old_output_space_floor_reused"] is False
                        and diagnostic["RAW_bound_applied"] is False, "NEW_FIXED_D_MINIMUM_SCOPE")
                    diagbase = {**base, "backend": backend, "mask": "all_C_complete"}
                    for method, metrics, role in (("NEW_FIXED_D_MINIMUM", diagnostic["minimum_solution_scores_complete_C"], "known_future_best_real_numeric_fixedD_rowspace"),
                        ("RAW_BOLT", diagnostic["RAW_scores_same_complete_C"], "same_mask_unconstrained_Raw")):
                        tables["NEW_FIXED_D_DIAGNOSTIC"] += score_rows({**diagbase, "method": method}, metrics, role)
                    for method, item in {"E_Y": diagnostic["E_Y"], **diagnostic["actual_methods"]}.items():
                        for row_score in score_rows({**diagbase, "method": method}, item["scores_complete_C"],
                            "known_future_E_Y_diagnostic_only" if method == "E_Y" else "actual_deployable_prediction"):
                            row_score.update(real_trace_scores_complete_C=item["real_trace_scores_complete_C"],
                                measured_FP32_rounding_normalized_MSE_allowance=item["measured_FP32_rounding_normalized_MSE_allowance"],
                                actual_FP32_minus_minimum=item["actual_FP32_minus_minimum"], real_trace_minus_minimum=item["real_trace_minus_minimum"])
                            tables["NEW_FIXED_D_DIAGNOSTIC"].append(row_score)
                    geometry = {key: value for key, value in diagnostic.items()
                        if key not in ("minimum_solution_scores_complete_C", "RAW_scores_same_complete_C", "E_Y", "actual_methods")}
                    tables["NEW_FIXED_D_GEOMETRY"].append({**base, "backend": backend, **geometry})
            for seed in SEEDS:
                head = condition["native"][seed]; meta = head["metadata"]
                tables["NATIVE_SELECTION"].append({"dataset": dataset, "space": space, "seed": seed, "selected_update": meta["selected_update"],
                    "selected_tune": meta["selected_tune"], "state_sha256": meta["selected_state_sha256"], "metadata": meta,
                    "two_random_initializations_claimed": False, "new_head_fits_in_v2": 1, "same_condition_was_executed_in_v1": space != "RAW_SUBSET"})
                for cp in head["checkpoints"]:
                    tables["CHECKPOINTS"].append({"dataset": dataset, "space": space, "seed": seed, "update": cp["update"],
                        **cp["tune"], "selected": cp["update"] == meta["selected_update"], "state_sha256": cp["state_sha256"],
                        "state_artifact": cp["state_artifact"], "exact_scored_TUNE_prediction_artifact": cp["tune_prediction_artifact"]})
                tables["LEARNING_CURVE"] += [{"dataset": dataset, "space": space, "seed": seed, **point} for point in head["learning_curve"]]
        rows = cost["cost"][dataset]["rows"]
        require(len(rows) == 20 and {(r["route"], r["batch"]) for r in rows} == {(route, b) for route in ROUTES for b in (1, 8)}, "ALL_TEN_ROUTES_B1_B8")
        for row in rows:
            raw = next(r for r in rows if r["route"] == "RAW_BOLT" and r["batch"] == row["batch"])
            require(row["repeats"] == len(row["seconds"]) == len(row["samples"]) == 10, "TEN_ACTUAL_SAMPLES_EACH_CELL")
            tables["WHOLE_COST"].append({"dataset": dataset, "route": row["route"], "batch": row["batch"], "repeats": 10,
                "median_seconds": row["median_seconds"], "median_seconds_per_window": row["median_seconds_per_window"],
                "ratio_of_RAW_medians": row["median_seconds"]/raw["median_seconds"], "median_paired_RAW_ratio": row["paired_RAW_median_ratio"],
                "seconds": row["seconds"], "origins": row["origins"], "memory_shared_process": row["memory"],
                "includes": row["whole_includes"], "excludes": row["whole_excludes"], "memory_scope": row["memory_scope"],
                "selected_metadata": row["selected_metadata"], "reference": row["reference"], "timed_outputs": row["timed_outputs"],
                "static_FIT_D_prepared_outside_timer": True, "native_timed": "_NATIVE_s" in row["route"]})
            for index, sample in enumerate(row["samples"]):
                tables["WHOLE_COST_SAMPLES"].append({"dataset": dataset, "route": row["route"], "batch": row["batch"],
                    "repeat": sample["repeat"], "position": sample["position"], "seconds": sample["seconds"],
                    "paired_RAW_ratio": row["paired_RAW_ratios"][index], "sample": sample})
        for route in ROUTES[1:]:
            space = next(name for name in SPACES if route.startswith(name+"_"))
            method = "BOLT" if route.endswith("_BOLT") else "NATIVE_s"+route[-5:]
            actual = next(r for r in tables["RELATIVE_TO_RAW"] if r["dataset"] == dataset and r["space"] == space and r["method"] == method and r["split"] == "tune")
            timing = next(r for r in tables["WHOLE_COST"] if r["dataset"] == dataset and r["route"] == route and r["batch"] == 8)
            tables["REGISTERED_ADOPTION"].append({"dataset": dataset, "route": route,
                "TUNE_MSE_ratio_to_RAW": actual["MSE_ratio_to_RAW"], "TUNE_MAE_ratio_to_RAW": actual["MAE_ratio_to_RAW"],
                "B8_ratio_of_RAW_medians": timing["ratio_of_RAW_medians"], "B8_median_paired_RAW_ratio": timing["median_paired_RAW_ratio"],
                "all_registered_conditions_pass": actual["MSE_ratio_to_RAW"] <= 1.01 and actual["MAE_ratio_to_RAW"] <= 1.01 and timing["ratio_of_RAW_medians"] <= .9,
                "new_selection_used": False, "independent_generalization_claimed": False})
    for name in INPUTS:
        value = values[name]
        tables["EXECUTION_COST_AND_COUNTS"].append({"stage": name, "status": value["status"],
            "wall_seconds": value.get("wall_seconds"), "resource_terminal": value.get("resource_terminal"),
            "totals": value.get("totals", value.get("completeness")), "native_training_wall_seconds_sum": sum(h["metadata"]["train_wall_seconds"] for d in value.get("datasets", {}).values() for s in d.get("spaces", {}).values() for h in s.get("native", {}).values()) if name == "science" else None,
            "separate_global_D_fit_wall_measured": False, "new_independent_generalization_runs": 0})
    tables["EXECUTION_COST_AND_COUNTS"].append({"stage": "preserved_prior_v1", "status": values["prior_partial"]["status"],
        "native_fits": 8, "updates": 4800, "CP": 40, "actual_subset_F_split_inferences": 0,
        "actual_native_fits_across_v1_v2": 20, "distinct_native_conditions": 12, "identity": science["mixed_reexecution_identity"],
        "independent_identity": audit["mixed_reexecution_identity"]})
    for dataset, diagnostic in values["Gram_failure_diagnostic"]["datasets"].items():
        tables["FAILURE_AND_REPAIR"].append({"dataset": dataset, **diagnostic["first_Gram_failure"],
            "failure_scope": "numeric score cancellation before subset selection; no subset forecast result", "old_v1_source_preserved": True,
            "new_score": "direct centered lstsq coefficient plus explicit residual squared mean; same rcond1e-4 and exact tie rule", "tolerance_relaxed": False})
    tables["AUDIT_FAILURE_PRESERVATION"] = values["preserved_audit_failures"]
    expected = {"COMPARISON": 264, "RELATIVE_TO_RAW": 54, "GREEDY_SELECTION": 6, "GREEDY_CANDIDATES": 83,
        "GREEDY_RESPONSE_SCORES": 1285, "GLOBAL_DECODER_STATES": 12, "GLOBAL_DECODER_CHANNELS": 168,
        "CHECKPOINTS": 60, "LEARNING_CURVE": 7200, "NATIVE_SELECTION": 12, "F_CACHE_AND_SUBSET_INFERENCE": 18,
        "NEW_FIXED_D_DIAGNOSTIC": 324, "NEW_FIXED_D_GEOMETRY": 36, "WHOLE_COST": 40, "WHOLE_COST_SAMPLES": 400,
        "REGISTERED_ADOPTION": 18, "EXECUTION_COST_AND_COUNTS": 5, "FAILURE_AND_REPAIR": 2, "AUDIT_FAILURE_PRESERVATION": 4}
    require({name: len(rows) for name, rows in tables.items()} == expected, "FULL_JSON_REPORT_COHORT_COUNTS")
    return tables


def choose(rows, **where):
    return next(row for row in rows if all(row[key] == value for key, value in where.items()))


def rel(path, output):
    return Path(os.path.relpath(Path(path).resolve(), Path(output).resolve())).as_posix()


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                if isinstance(value, (dict, list)) else value for key, value in row.items()})


def figures(tables, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    folder = Path(output)/"figures"; folder.mkdir(exist_ok=False); paths = []
    def save(fig, name):
        for suffix in ("png", "svg"):
            path = folder/(name+"."+suffix); fig.savefig(path, dpi=165 if suffix == "png" else None, bbox_inches="tight"); paths.append(path)
        plt.close(fig)
    def metric(dataset, space, method, split):
        return choose(tables["COMPARISON"], dataset=dataset, space=space, method=method, split=split,
            mask="all_original_observed", metric_scale="normalized")["mse"]
    colors = ("#0072B2", "#D55E00", "#009E73")
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.5), layout="constrained")
    for di, (dataset, (_, _, dev)) in enumerate(DATASETS.items()):
        for si, split in enumerate(("tune", dev)):
            ax = axes[di, si]
            numbers = [metric(dataset, "NA", "RAW_BOLT", split)] + [metric(dataset, s, "BOLT", split) for s in SPACES]
            persist = metric(dataset, "NA", "CURRENT_C_PERSISTENCE", split)
            bars = ax.bar(np.arange(4), numbers, color=("#666666", *colors))
            ax.bar_label(bars, labels=[f"{v:.5f}" for v in numbers], padding=4)
            ax.axhline(persist, color="#882255", linestyle=":", label=f"Current-C persistence: {persist:.5f}")
            ax.set_xticks(np.arange(4), ("Raw Bolt", "PCA / Bolt", "Target / Bolt", "Subset / Bolt"))
            ax.set_ylim(0, max(*numbers, persist)*1.24); ax.grid(axis="y", alpha=.2)
            ax.set_ylabel("Normalized macro MSE / all observed target")
            ax.set_title(f"{DISPLAY[dataset]} / {split.upper()}\nDotted current-C persistence = {persist:.5f}")
    fig.suptitle("Raw-channel subset and mixed spaces: actual frozen-Bolt forecasts", fontsize=13)
    save(fig, "01_actual_Bolt")
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.7), layout="constrained")
    for di, (dataset, (_, _, dev)) in enumerate(DATASETS.items()):
        for si, split in enumerate(("tune", dev)):
            ax = axes[di, si]; x = np.arange(3)
            all_values = []
            for seed_index, seed in enumerate(SEEDS):
                numbers = [metric(dataset, space, "NATIVE_s"+seed, split) for space in SPACES]; all_values += numbers
                bars = ax.bar(x+(seed_index-.5)*.34, numbers, width=.33, color=colors, alpha=1. if seed_index == 0 else .55)
                ax.bar_label(bars, labels=[f"{v:.4f}\ns{seed[-1]}" for v in numbers], padding=3, fontsize=8)
            raw = metric(dataset, "NA", "RAW_BOLT", split); persist = metric(dataset, "NA", "CURRENT_C_PERSISTENCE", split)
            ax.axhline(raw, color="#666666", linestyle="--", label=f"Raw Bolt: {raw:.4f}")
            ax.axhline(persist, color="#882255", linestyle=":", label=f"Current-C persistence: {persist:.4f}")
            ax.set_xticks(x, ("PCA / fresh native", "Target / fresh native", "Subset / fresh native"))
            ax.set_ylim(0, max(*all_values, raw, persist)*1.27); ax.grid(axis="y", alpha=.2)
            ax.set_title(f"{DISPLAY[dataset]} / {split.upper()}\nDashed Raw = {raw:.4f}; dotted current C = {persist:.4f}")
            ax.set_ylabel("Normalized macro MSE / all observed target")
    fig.suptitle("Matched fresh heads: same 600 updates and TUNE checkpoint rule; seed changes batch order", fontsize=12)
    save(fig, "02_actual_native")
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10), layout="constrained")
    queries = (("BOLT_NUMPY", "NEW_FIXED_D_MINIMUM"), ("NATIVE_TORCH32", "NEW_FIXED_D_MINIMUM"),
        ("BOLT_NUMPY", "E_Y"), ("NATIVE_TORCH32", "E_Y"), ("BOLT_NUMPY", "BOLT"),
        ("NATIVE_TORCH32", "NATIVE_s92601"), ("NATIVE_TORCH32", "NATIVE_s92602"))
    for di, (dataset, (_, _, dev)) in enumerate(DATASETS.items()):
        for pi, space in enumerate(SPACES):
            ax = axes[di, pi]
            numbers = [choose(tables["NEW_FIXED_D_DIAGNOSTIC"], dataset=dataset, space=space, split=dev,
                backend=backend, method=method, metric_scale="normalized")["mse"] for backend, method in queries]
            raw = choose(tables["NEW_FIXED_D_DIAGNOSTIC"], dataset=dataset, space=space, split=dev,
                backend="BOLT_NUMPY", method="RAW_BOLT", metric_scale="normalized")["mse"]
            bars = ax.bar(np.arange(7), numbers, color=("#CC79A7", "#882255", "#BBBBBB", "#999999", colors[pi], "#56B4E9", "#E69F00"))
            ax.bar_label(bars, labels=[f"{n:.4f}" for n in numbers], padding=3, fontsize=8)
            ax.axhline(raw, color="#666666", linestyle="--")
            ax.set_ylim(0, max(*numbers, raw)*1.25); ax.grid(axis="y", alpha=.2)
            ax.set_xticks(np.arange(7), ("D min/F", "D min/N", "E(Y)/F", "E(Y)/N", "Bolt", "Native s1", "Native s2"), rotation=23, ha="right")
            ax.set_title(f"{DISPLAY[dataset]} / {SHORT[space]} / {dev.upper()}\nSame-mask Raw={raw:.5f}")
            ax.set_ylabel("Normalized MSE / same complete-C target")
    fig.suptitle("New fixed-D output spaces: known-future diagnostics are separate from achieved forecasts", fontsize=13)
    save(fig, "03_new_fixedD_diagnostic")
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9), layout="constrained")
    for di, (dataset, (channels, _, _)) in enumerate(DATASETS.items()):
        ax = axes[di, 0]
        stages = [r for r in tables["GREEDY_SELECTION"] if r["dataset"] == dataset]
        for stage in stages:
            points = [r for r in tables["GREEDY_CANDIDATES"] if r["dataset"] == dataset and r["stage"] == stage["stage"]]
            line, = ax.plot([r["candidate_original_index"] for r in points], [r["macro_normalized_MSE"] for r in points],
                marker=".", linewidth=1., label=f"Stage {stage['stage']}: add {stage['chosen_original_index']}")
            winner = next(r for r in points if r["winner"])
            ax.scatter([winner["candidate_original_index"]], [winner["macro_normalized_MSE"]], marker="*", s=140, color=line.get_color(), zorder=5)
        ax.set_title(f"{DISPLAY[dataset]}: all FIT-past candidate scores")
        ax.set_xlabel("Original channel index (zero based)"); ax.set_ylabel("Observed-C macro past reconstruction MSE")
        ax.set_xticks(range(channels)); ax.grid(alpha=.2); ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
        ax = axes[di, 1]
        for pi, space in enumerate(SPACES):
            points = [choose(tables["GLOBAL_DECODER_CHANNELS"], dataset=dataset, space=space, backend="BOLT_NUMPY",
                response_channel_zero_based=channel)["observed_past_reconstruction_MSE_FP64"] for channel in range(channels)]
            ax.plot(range(channels), points, marker=".", color=colors[pi], label=SHORT[space])
        ax.set_title(f"{DISPLAY[dataset]}: same global-D fitting rule / NumPy backend")
        ax.set_xlabel("Response channel index (zero based)"); ax.set_ylabel("Observed past reconstruction MSE / channel")
        ax.set_xticks(range(channels)); ax.grid(alpha=.2); ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    fig.suptitle("Past reconstruction selection and fit: these scores do not guarantee future predictability", fontsize=13)
    save(fig, "04_subset_past_selection")
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), layout="constrained")
    for di, (dataset, _) in enumerate(DATASETS.items()):
        for pi, space in enumerate(SPACES):
            ax = axes[di, pi]
            for si, seed in enumerate(SEEDS):
                rows = [r for r in tables["CHECKPOINTS"] if r["dataset"] == dataset and r["space"] == space and r["seed"] == seed]
                line, = ax.plot([r["update"] for r in rows], [r["mse"] for r in rows], marker="o", label="s"+seed)
                selected = next(r for r in rows if r["selected"])
                ax.scatter([selected["update"]], [selected["mse"]], marker="*", s=160, color=line.get_color(), zorder=5)
            ax.set_xticks((0, 100, 200, 400, 600)); ax.set_xlabel("Training updates; star = selected checkpoint")
            ax.set_ylabel("Entire TUNE normalized observed-C macro MSE")
            ax.set_title(f"{DISPLAY[dataset]} / {SHORT[space]}"); ax.grid(alpha=.2)
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    fig.suptitle("All 60 checkpoints; CP0 is current-C persistence, exact ties choose the earlier checkpoint", fontsize=12)
    save(fig, "05_checkpoint_learning")
    fig, axes = plt.subplots(2, 2, figsize=(19, 11), layout="constrained")
    labels = ("Raw F", "PCA F", "Target F", "Subset F", "PCA N1", "PCA N2", "Target N1", "Target N2", "Subset N1", "Subset N2")
    for di, dataset in enumerate(DATASETS):
        for bi, batch in enumerate((1, 8)):
            ax = axes[di, bi]
            rows = [choose(tables["WHOLE_COST"], dataset=dataset, route=route, batch=batch) for route in ROUTES]
            medians = [r["median_seconds"]*1000 for r in rows]
            ax.bar(np.arange(10), medians, color=("#666666", *colors, colors[0], colors[0], colors[1], colors[1], colors[2], colors[2]), alpha=.75)
            maxima = []
            for index, row in enumerate(rows):
                samples = np.asarray(row["seconds"])*1000
                ax.scatter(index+np.linspace(-.12, .12, 10), samples, color="#333333", s=12, alpha=.6, zorder=4)
                maxima.append(max(row["median_seconds"]*1000, float(samples.max())))
            padding = max(max(maxima), .001)*.045
            for index, row in enumerate(rows):
                ax.text(index, maxima[index]+padding, f"{medians[index]:.2f} ms\n{row['median_paired_RAW_ratio']:.2f}x", ha="center", va="bottom", fontsize=7)
            ax.set_ylim(0, max(maxima)+padding*5); ax.set_xticks(np.arange(10), labels, rotation=27, ha="right")
            ax.set_ylabel("Whole pipeline milliseconds / batch (points = all 10 samples)")
            ax.set_title(f"{DISPLAY[dataset]} / B{batch}; labels show median and paired-RAW ratio")
            ax.grid(axis="y", alpha=.2)
    fig.suptitle("Actual whole cost: causal physical input to physical CPU return; static FIT-D preparation excluded", fontsize=13)
    save(fig, "06_whole_cost")
    return paths


def make_text(values, tables, bindings, output):
    science = values["science"]
    lines = ["# 원 채널 subset과 혼합 표현: 같은 고정 FIT 디코더 규칙의 실제 비교", "",
        "[확인] 원 채널을 섞지 않고 작은 K개만 사용하는 대조까지 실제로 수행했다. 아래 답은 두 자료의 기노출 개발 구간과 고정된 K·학습기·손실·선정 규칙에 한정한다. 원자료 보호 TEST를 사용하지 않았고, 독립 일반화나 전체 연구의 해결을 주장하지 않는다.", "",
        "## 먼저 답할 연구 질문", "",
        "### 1. 원 채널 혼합과 frozen Bolt의 호환성이 주원인이었는가?", "",
        "[확인] 다음 원수치는 모두 같은 FIT mean/std·관측 mask·origin·과거512/미래48·current-C anchor와 같은 FIT 관측 과거 global-D 적합 규칙을 사용한다. Raw는 원 C-channel frozen Bolt다. 세 공간의 E와 D 수치가 같다는 뜻은 아니다.", "", "```text",
        "자료/구간                       Raw Bolt       PCA/Bolt       Target/Bolt    Subset/Bolt    Current-C persistence"]
    for dataset, (_, _, dev) in DATASETS.items():
        for split in ("tune", dev):
            def q(space, method):
                return choose(tables["COMPARISON"], dataset=dataset, space=space, split=split, method=method,
                    mask="all_original_observed", metric_scale="normalized")["mse"]
            numbers = [q("NA", "RAW_BOLT"), *[q(space, "BOLT") for space in SPACES], q("NA", "CURRENT_C_PERSISTENCE")]
            lines.append(f"{DISPLAY[dataset]+'/'+split.upper():30s} "+"  ".join(f"{n:13.9f}" for n in numbers))
    lines += ["```", ""]
    for dataset, (_, _, dev) in DATASETS.items():
        point = choose(tables["RELATIVE_TO_RAW"], dataset=dataset, space="RAW_SUBSET", method="BOLT", split=dev)
        lines.append(f"[확인] {DISPLAY[dataset]} {dev.upper()}의 subset/Bolt 정규화 MSE는 {point['MSE']:.9f}, Raw는 {point['RAW_MSE']:.9f}로 {point['MSE_ratio_to_RAW']:.6f}배다. 이 값은 같은 full-observed mask의 비교다.")
    subset_F_diffs = [row["full_C_slice_diagnostic"]["max_abs"] for row in tables["F_CACHE_AND_SUBSET_INFERENCE"] if row["space"] == "RAW_SUBSET"]
    lines.append(f"[확인] 실제 새 K-channel F와 기존 full-C Raw의 선택 채널 forecast slice의 최대 절대 차이는 6개 split에서 {min(subset_F_diffs):.9g}–{max(subset_F_diffs):.9g}였다. 선택된 원 시계열의 F 출력과 C 전체 복원 오차를 구분해야 한다. 이 가까운 수치 일치가 모든 K·batch·채널에서의 byte 동일성을 증명하지는 않는다.")
    lines += ["", "![Actual Bolt comparison](figures/01_actual_Bolt.png)", "",
        "[판단] 이 대조로 혼합 좌표가 실패의 유일 원인이라는 결론을 내릴 수 없다. Subset은 채널 혼합을 없애는 동시에 보존하는 정보와 출력 D 공간도 바꾼다. 선정 기준은 과거 복원이며 미래 예측성이 아니다. 따라서 subset의 성적이 나쁘면 이번 단순 대안의 유용성은 약해지지만, 모든 원 채널 선택·혼합 표현·예측기의 가능성까지 부정할 근거가 되지는 않는다. 동일 K에서 혼합 여부만 바꾼 완전한 단일 인과 개입도 아니다.", "",
        "### 2. 고정 복원 공간과 예측기 중 어디에 문제가 남았는가?", "",
        "[확인] 이번에 새로 적합한 backend별 D32의 실제 수치 rowspace를 각각 사용해 미래를 이미 아는 경우의 최적 정규화 MSE를 다시 계산했다. 기존 static·per-origin·global rank-K oracle를 가져오지 않았다. 다음 표는 같은 complete-C 관측 위치끼리 비교하므로 Beijing의 full-observed Raw 숫자와 섞으면 안 된다.", "", "```text",
        "자료/공간                   같은mask Raw    D 최소/F       실제 Bolt       Bolt-최소      E(Y)/F"]
    for dataset, (_, _, dev) in DATASETS.items():
        for space in SPACES:
            def d(method):
                return choose(tables["NEW_FIXED_D_DIAGNOSTIC"], dataset=dataset, space=space, split=dev, backend="BOLT_NUMPY",
                    method=method, metric_scale="normalized")["mse"]
            raw, minimum, actual, ey = (d(m) for m in ("RAW_BOLT", "NEW_FIXED_D_MINIMUM", "BOLT", "E_Y"))
            lines.append(f"{DISPLAY[dataset]+'/'+SHORT[space]:26s} {raw:13.9f} {minimum:13.9f} {actual:13.9f} {actual-minimum:13.9f} {ey:13.9f}")
    lines += ["```", ""]
    for dataset, (_, _, dev) in DATASETS.items():
        minimums = [choose(tables["NEW_FIXED_D_DIAGNOSTIC"], dataset=dataset, space=space, split=dev,
            backend="BOLT_NUMPY", method="NEW_FIXED_D_MINIMUM", metric_scale="normalized")["mse"] for space in SPACES]
        raw = choose(tables["NEW_FIXED_D_DIAGNOSTIC"], dataset=dataset, space="RAW_SUBSET", split=dev,
            backend="BOLT_NUMPY", method="RAW_BOLT", metric_scale="normalized")["mse"]
        above = sum(value > raw for value in minimums)
        subset_ey = choose(tables["NEW_FIXED_D_DIAGNOSTIC"], dataset=dataset, space="RAW_SUBSET", split=dev,
            backend="BOLT_NUMPY", method="E_Y", metric_scale="normalized")["mse"]
        lines.append(f"[확인] {DISPLAY[dataset]} {dev.upper()}의 세 NumPy 고정 D 중 {above}/3개는 미래 잠재값을 자유롭게 고르는 실수 minimum 자체가 같은 mask Raw({raw:.9f})보다 높다. Subset의 E(Y) 진단은 {subset_ey:.9f}다.")
        if above == 3:
            lines.append("[판단] 이 자료의 현재 세 고정 출력 공간은 그 조건 아래 Raw 수준 도달을 막는 제약이 있다. 실제 예측 오차는 해당 minimum보다도 커서 남은 predictor gap 역시 함께 확인해야 한다.")
        if above == 0:
            lines.append("[판단] 이 자료에서는 세 고정 출력 공간의 minimum만으로 Raw보다 큰 실제 오차를 설명할 수 없다. 낮은 오차의 잠재값이 공간에 존재하는 것과 과거로 예측 가능한 것은 별개다.")
    lines += ["", "![New fixed-D diagnostics](figures/03_new_fixedD_diagnostic.png)", "",
        "[판단] 최소 오차가 Raw보다 크면 이 특정 D 공간은 미래 잠재값을 완벽히 고르더라도 그 Raw 수준에 도달하기 어렵다는 수학적 제약을 드러낸다. 최소 오차가 낮아도 실제 예측기의 gap이 크면 낮은 공간 오차만으로 예측 성공을 보장하지 않는다. 이 차이는 정해진 실수 수치 rowspace의 진단이며, FP32의 모든 가능한 출력에 대한 엄밀 하한이나 단계별 실패 원인의 인과 비율은 아니다. 실제 FP32/real-trace 점수와 측정된 rounding allowance를 CSV에 함께 보존했다.", "",
        "[확인] 최소라는 말은 정규화 C macro MSE에만 해당한다. 그 해의 MAE·이질적 원단위 MSE를 각 목적의 최소라고 부르지 않는다. E(Y)는 이미 알려진 실제 미래를 encoder에 넣는 별도 진단이며 최적 잠재값과 같다는 보장이 없다. 배포·E/D/head fitting·선정에 사용하지 않았다. 공간에 낮은 오차의 해가 있다는 사실은 과거만으로 그 해를 예측할 수 있다는 증명이 아니다.", "",
        "### 3. 새 matched native와 전체 비용까지 보면 작은 K의 효용이 남는가?", "",
        "[확인] 새 native는 각 E/backend의 고정 FIT D32에 맞춰 actual-C 미래 관측 macro MSE로 새로 학습했다. 과거 복원만 학습한 head를 그대로 비교한 결과가 아니다. 원 POINTWISE/MAIN 계열에도 actual-Y 학습이 있었으므로 지금까지의 모든 실패를 ‘재구성 학습만 해서’로 바꾸어 설명할 수 없다. 다음 두 seed는 모두 zero-init이며 minibatch 순서만 다르다.", "", "```text",
        "자료/공간/seed                 선택CP      TUNE MSE      DEV MSE       DEV/Raw"]
    for dataset, (_, _, dev) in DATASETS.items():
        for space in SPACES:
            for seed in SEEDS:
                head = choose(tables["NATIVE_SELECTION"], dataset=dataset, space=space, seed=seed)
                tune = choose(tables["RELATIVE_TO_RAW"], dataset=dataset, space=space, method="NATIVE_s"+seed, split="tune")
                development = choose(tables["RELATIVE_TO_RAW"], dataset=dataset, space=space, method="NATIVE_s"+seed, split=dev)
                lines.append(f"{DISPLAY[dataset]+'/'+SHORT[space]+'/'+seed:31s} {head['selected_update']:6d} {tune['MSE']:13.9f} {development['MSE']:13.9f} {development['MSE_ratio_to_RAW']:11.6f}")
    passed = sum(r["all_registered_conditions_pass"] for r in tables["REGISTERED_ADOPTION"])
    lines += ["```", "", "![Fresh native comparison](figures/02_actual_native.png)", "",
        f"[확인] 원 동시 기준(TUNE MSE/MAE 각각 Raw×1.01 이하, 같은 시행 B8 ratio-of-medians≤0.90)을 통과한 자료·경로는 {passed}/18개다. seed나 DEV 최저값으로 채택 규칙을 바꾸지 않았다. 기준 통과 자체도 독립 일반화 검증은 아니다.", "",
        "[판단] 이 대조는 작은 K에서 고정 Bolt의 정확도와 전체 비용을 함께 유지하려는 목적을 평가한다. 단순 ridge의 이득 유무로 목적을 줄이지 않는다. 같은 affine native class와 600update에서 실패하더라도 모든 native·과거 정보의 예측 불가능성이나 frozen Bolt의 필수성을 증명하지 않는다. Transformer/대조학습/그래프를 추가해야 한다는 결론도 아직 자동으로 따라오지 않는다.", "",
        "## 실제 데이터·학습·선정 계약", "",
        "Robin C17/K4, Beijing Aotizhongxin C11/K2, FIT256/TUNE64, 기노출 Robin VAL30·Beijing EVAL146을 유지했다. causal-filled 정규화 과거는 predictor로 쓰되, 과거 복원 D와 subset score의 response에는 채널별 true-observed 위치만 썼다. 중복 FIT window의 펼친 행을 독립 표본으로 세지 않는다.", "",
        "PCA/FIT-target E는 원 SHA 상태를 재사용했다. FIT-target E는 원 FIT 실제 미래로 감독받은 공간이다. 이것은 정상적인 FIT supervision이며 평가 origin의 미래를 배포 입력으로 준 것과 다르다. 세 E의 감독 정보가 동일하다고 주장하지 않는다. 이번 subset 선정과 새 global D는 FIT 관측 과거만 사용하고, E의 배포식은 시점별 현재까지의 값만 사용한다.", "",
        "Subset은 response별 중심화한 직접 lstsq(rcond1e-4)의 잔차 제곱 평균을 관측 C에 균등 평균해 greedy하게 추가한다. exact FP64 tie만 작은 원 index로 정리한다. 최적 K-subset이나 예측 목적 선정을 보장하지 않는다. 최종 E 열은 원 채널 오름차순이며 실제 값은 직접 FP32 slice로 유지한다. dense one-hot matmul의 signedzero byte 동일성은 주장하지 않는다.", "", "```text", "자료                     최종 원 index / 채널명                    greedy 추가 순서"]
    for dataset in DATASETS:
        stage = next(r for r in tables["GREEDY_SELECTION"] if r["dataset"] == dataset)
        lines.append(f"{DISPLAY[dataset]:23s} {stage['final_selected_indices']} / {', '.join(stage['final_selected_names'])}    {stage['final_greedy_order']}")
    lines += ["```", "", "![Past selection and reconstruction](figures/04_subset_past_selection.png)", "",
        "세 공간의 global D는 같은 중심화·관측 mask·rcond1e-4·분석적 절편 규칙이지만 각 E의 실제 FP32 과거 latent에 따로 적합했다. NumPy Bolt와 Torch CPU2/canonical32 native의 실제 Z를 구분해 D12개를 보존했다. 상수 response의 full-rank zero-slope는 정상 적합이며, 무관측·관측1개·rank-deficient slope와 구분했다.", "",
        "배포식은 `currentC + (predictedZfuture - lastZ) @ D32`다. 복원 절편은 innovation에서 상쇄된다. 평가 origin마다 D를 새로 적합하거나 C 과거 회귀계수를 추가하지 않지만 current-C anchor는 여전히 읽으므로 strict K-only가 아니다. 기존 pinv D나 shrinkage D와 동일한 디코더라는 주장도 하지 않는다.", "",
        "12 native 조건은 zero-init K별 affine512→48, AdamW lr0.001/WD0/clip1/batch4·600update·seed92601/92602·CP0/100/200/400/600을 유지했다. 전체 TUNE 정규화 관측 C macro MSE로 checkpoint를 고르고 exact tie는 빠른 CP다. 12개 선정 잠금과 registry를 모두 저장한 다음 이번 DEV 입력/예측을 열었다. 다만 전체 후속 설계 자체는 이전 DEV·oracle를 본 뒤 진행한 탐색적 대조이므로 독립 holdout으로 바뀌지 않는다.", "",
        "![All checkpoints](figures/05_checkpoint_learning.png)", "",
        "CP0는 현재 C 반복 baseline이며 독립 감사에서 같은 mask로 검산했다. 이 baseline의 실제 전체 배포 비용은 이번 열 경로에 포함하지 않았다. checkpoint 예측은 canonical batch32의 실제 scored array를 저장했다. cached past를 고정하고 미래 target/mask를 바꾸는 12검사는 그 경로의 forward가 label을 읽지 않는 좁은 검사다. raw upstream 재생·E/D 재학습이나 독립 일반화의 증명으로 확대하지 않는다.", "",
        "## 실제 추론과 전체 비용", "",
        "기존 mixed F latent cache12개는 실제 새 E 과거/origin byte lineage가 맞는 경우만 재사용했다. Subset K-channel F는 FIT/TUNE/DEV 총6개 split을 실제 새로 추론했으며 model batch103회를 기록했다. Full-C Raw forecast에서 잘라낸 것을 신규 추론으로 세지 않았다. 그 slice와 새 K-F 결과의 차이는 별도 수치 진단으로 보존했다.", "",
        "비용은 같은 시행에서 자료2×batch1/8×10경로=40셀, warmup2·반복10=400개 실제 타이머 표본이다. 10개 circular rotation으로 위치를 균등하게 사용했다. 같은 첫 B개 DEV origin을 반복하므로 400개의 독립 데이터가 아니다. raw ns start/end·순서·outlier·예측 출력 검산을 보존했다.", "",
        "타이머는 physical window의 causal fill/FIT normalize부터 실제 E→frozen F 또는 선택 native CPU2 head→고정 D/currentC anchor→원단위 CPU return 및 동기화까지다. 고정 FIT D/E/head 로드·FIT 학습·model/state/file hash·setup·warmup·저장·guard·memory snapshot은 밖이다. 이 경로에는 per-origin D fit가 없다. 별도의 global-D fitting 시간은 단독 구간으로 측정하지 않았으므로 만들어내지 않고 캠페인 receipt 및 head 학습 시간만 분리해 제공한다.", "",
        "![Actual whole cost](figures/06_whole_cost.png)", "", "```text", "자료/경로                              B1 batch ms   B8 batch ms   B8 ms/window   B8/Raw median비   B8 paired비 median"]
    for dataset in DATASETS:
        for route in ROUTES:
            b1 = choose(tables["WHOLE_COST"], dataset=dataset, route=route, batch=1)
            b8 = choose(tables["WHOLE_COST"], dataset=dataset, route=route, batch=8)
            lines.append(f"{DISPLAY[dataset]+'/'+route:46s} {b1['median_seconds']*1000:12.5f} {b8['median_seconds']*1000:12.5f} {b8['median_seconds_per_window']*1000:14.7f} {b8['ratio_of_RAW_medians']:15.7f} {b8['median_paired_RAW_ratio']:18.7f}")
    lines += ["```", "",
        "ratio-of-medians와 같은 반복끼리의 paired ratio 중앙값은 서로 다른 통계량이다. 별도 CSV로 모두 보존하며 원 채택 기준은 앞의 통계량이다. B1/B8 실제 native와 canonical32 score의 수치 차이는 등록 precheck 공차(정규화 atol1e-5/rtol2e-6, 원단위 atol2e-3/rtol2e-5)로 검산했다. 혼합 E와 subset direct slice의 계약·원 mask/scaler/origin을 함께 확인했다.", "",
        "[불확실성] 절대 시간은 장비 상태와 순서에 민감하며 전체 시스템 독점이나 완전히 조용한 환경을 보장하지 않는다. 모든 E/D/12head/F가 resident인 공유 process의 RSS/VRAM을 방법 고유 메모리 절감으로 해석하지 않는다. 이전 shrinkage나 다른 cohort의 Raw 시간을 가져와 속도비를 만들지 않았다. 학습 비용·배포 latency·저장량·논리적 정보량은 서로 다른 측정이다.", "",
        "## 실패 기록과 수치 복구", "",
        "[확인] 최초 v1은 PARTIAL_COMPLETED였다. 두 자료 모두 subset 선정 전 Gram/cross/response variance의 상쇄식이 약1e-12 음수 MSE를 만들었다. 원 tolerance를 넘었고 subset 선정0·신규 subset F0이었지만 독립 가능한 mixed8fit/4800update/40CP·24native/12Bolt split은 완료했다. 실패를 subset forecasting 성적이나 방법의 구조적 한계로 부르지 않는다.", "", "```text", "자료       Gram MSE             원 negative tolerance    직접 centered 잔차 MSE"]
    for item in tables["FAILURE_AND_REPAIR"]:
        lines.append(f"{DISPLAY[item['dataset']]:20s} {item['Gram_MSE']: .12e} {item['original_negative_tolerance']: .12e} {item['direct_centered_residual_MSE']: .12e}")
    lines += ["```", "",
        "v2는 원 source를 보존하고 새 파일에서 동일 FIT-past LS 목적/rcond/exact tie 규칙에 직접 centered lstsq 계수와 residual² 평균을 사용했다. 음수 clipping이나 공차 확대를 하지 않았다. finite 계산 경로가 달라 근접 후보 점수/순서가 bit-identical하다고 주장하지 않는다. v1에서는 subset 승자가 없었다.", "",
        "v2에서 원 12-lock-before-DEV 계약을 단순하게 유지하려고 12조건을 새로 학습했다. 그중8mixed는 v1의 반복이며 E/D·40CP·TUNE/DEV 예측·학습곡선·선정update가 공차 없이 동일함을 검산했다. 두 시도의 실제 native 학습은20회/12000update/100CP지만 distinct 등록 조건은12개다. 이 중복을 추가 독립 실험이나 일반화 증거로 세지 않았다. 시간/누적 상한 해제는 기록 초기화가 아니며 원 실패와 모든 receipt를 보존했다.", "",
        "[확인] 독립 auditor v2와 같은 source의 traceback 재호출은 FAILED_PRESERVED로 남았다. producer의 complete bool을 E(Y) 계산의 complete-C mask ndarray가 덮어 최종 if에서 ambiguous truth-value 예외가 났다. 이 두 파일을 PASS로 세지 않는다. ROOT exception trace를 보존했고 새 auditor에서는 변수명만 분리했다. 모델·예측·학습·수식·공차를 바꾼 복구가 아니며, 최종 채택 근거는 별도로 결속된 새 PASS다.", "",
        "[확인] 비용 실험 자체는40셀/400표본 COMPLETE였지만 최초 비용 auditor는 복구 addendum Markdown을 JSON으로 해석해 JSONDecodeError로 FAILED_PRESERVED가 됐다. 새 auditor는 그 파일을 원 path/SHA로 검증한다. Markdown 본문·예측·timer sample·수치/공차를 바꾼 것이 아니다. 이 실패와 최종 비용 PASS를 분리해 원 기록을 링크했다.", "",
        "## 무엇이 남았는가", "",
        "[판단] 보고서의 수치와 새 complete-C 진단을 함께 보면 이번 단순 subset·고정 과거 D·선형 native/F 조합이 어느 기준에서 실패했는지 구분할 수 있다. 좋은 과거 복원이 좋은 미래 표현을 보장하지 않는다는 반론은 여전히 유효하다. 그러나 actual-Y 학습도 이미 수행했으므로 원인을 재구성 목적 하나나 최적화 문제 하나로 확정하지 않는다. 특정 D 공간·예측기의 gap·subset 정보손실·좌표 적합성·학습 class의 표현력이라는 남은 가능성을 구분해야 한다.", "",
        "다음 판단은 K와 비용 목적을 유지하면서 새로운 목적이나 모델을 도입할 때 그 비교가 무엇을 반증하는지 먼저 고정하는 것이다. 이 보고서만으로 graph/contrastive가 필요하다거나 불필요하다거나, 모든 저차원 복원이 불가능하다거나, Chronos-Bolt를 반드시 유지해야 한다고 결론 내리지 않는다. 새로운 독립 평가 자료와 통계 검증도 아직 없다.", "",
        "## 원수치·실행 소스·검산 링크", ""]
    for key, binding in bindings.items():
        lines.append(f"- [{key}]({rel(resolve(binding['path']), output)}) — SHA256 `{binding['sha256']}`")
    lines += ["", "원 보고서는 각 시점의 불확실성과 결과를 유지한다.", ""]
    for name, expected_sha in PRIOR_REPORTS.items():
        path = HERE/"reports"/name/"REPORT_KO.md"
        lines.append(f"- [{name}]({rel(path, output)}) — SHA256 `{expected_sha}`")
    lines += ["", "전체 CSV는 nested metric·관측수·source binding을 JSON cell에 보존한다.", ""]
    for name, rows in tables.items(): lines.append(f"- [{name}](tables/{name}.csv) — {len(rows)} rows")
    lines += ["", "PNG와 같은 figure의 SVG 원본도 `figures/`에 함께 저장했다. 실행·독립 검산·보고/시각화의 완료와 연구 전체 해결을 구분한다.", ""]
    return "\n".join(lines)


def generate(paths, output_path):
    values, bindings = load_verified(paths)
    prior_reports = {}
    for name, expected_sha in PRIOR_REPORTS.items():
        path = HERE/"reports"/name/"REPORT_KO.md"
        require(path.is_file() and sha(path) == expected_sha, "PRESERVED_PRIOR_REPORT:"+name)
        prior_reports[name] = record(path)
    tables = collect_tables(values)
    output = Path(output_path).resolve()
    require(output.parent == (HERE/"reports").resolve() and not output.exists(), "FRESH_LOCAL_REPORT_DIRECTORY_REQUIRED")
    source = record(__file__); bindings["report_source"] = source
    output.mkdir(parents=True, exist_ok=False)
    folder = output/"tables"; folder.mkdir()
    for name, rows in tables.items(): write_csv(folder/(name+".csv"), rows)
    images = figures(tables, output)
    (output/"REPORT_KO.md").write_text(make_text(values, tables, bindings, output), encoding="utf-8", newline="\n")
    require(source == record(__file__), "EXECUTED_REPORT_SOURCE_CHANGED")
    files = [{"path": path.relative_to(output).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size}
        for path in sorted(output.rglob("*")) if path.is_file()]
    manifest = {"status": "COMPLETE", "source": source, "inputs": bindings, "preserved_prior_reports": prior_reports,
        "table_row_counts": {name: len(rows) for name, rows in tables.items()}, "files": files,
        "figure_png_count": sum(path.suffix == ".png" for path in images), "figure_svg_count": sum(path.suffix == ".svg" for path in images),
        "scientific_NPZ_or_model_read": False, "new_fit_F_or_training_executed": False, "protected_TEST_accessed": False,
        "known_future_diagnostics_separate_from_deployment": True, "independent_generalization_claimed": False}
    with (output/"MANIFEST.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")
    print(json.dumps({"status": "COMPLETE", "manifest": str(output/"MANIFEST.json"), "files": len(files),
        "tables": len(tables), "figures": len(images)}, ensure_ascii=False), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (*INPUTS, "output"): parser.add_argument("--"+name.replace("_", "-"), required=True)
    args = vars(parser.parse_args()); output = args.pop("output"); generate(args, output)
