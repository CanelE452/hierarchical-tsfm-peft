"""Independent JSON/CSV/MD/artifact verification; no generator or scientific IO."""
from pathlib import Path
import argparse
import ast
import csv
import json
import math
import statistics
import xml.etree.ElementTree as ET

from verify_fit_target_report_v1 import require, sha, read, resolve, verify_record, leaf_check, links
from audit_shrinkage_matched_native_report_v1 import bind, require_counts, unique

HERE = Path(__file__).resolve().parent
DATASETS = {"robin": (17, 4, "val"), "beijing_aotizhongxin": (11, 2, "eval")}
SPACES, SEEDS = ("PCA", "FIT_TARGET", "RAW_SUBSET"), ("92601", "92602")
BACKENDS = {"BOLT_NUMPY": ("BOLT",), "NATIVE_TORCH32": ("NATIVE_s92601", "NATIVE_s92602")}
SCALES = ("normalized", "physical_per_channel_diagnostic")
ROUTES = ("RAW_BOLT", "PCA_BOLT", "FIT_TARGET_BOLT", "RAW_SUBSET_BOLT", "PCA_NATIVE_s92601", "PCA_NATIVE_s92602", "FIT_TARGET_NATIVE_s92601", "FIT_TARGET_NATIVE_s92602", "RAW_SUBSET_NATIVE_s92601", "RAW_SUBSET_NATIVE_s92602")
INPUTS = ("science", "audit", "cost", "cost_audit")
FIGURES = ("01_actual_Bolt", "02_actual_native", "03_new_fixedD_diagnostic", "04_subset_past_selection", "05_checkpoint_learning", "06_whole_cost")
COUNTS = {"COMPARISON": 264, "RELATIVE_TO_RAW": 54, "GREEDY_SELECTION": 6, "GREEDY_CANDIDATES": 83, "GREEDY_RESPONSE_SCORES": 1285, "GLOBAL_DECODER_STATES": 12, "GLOBAL_DECODER_CHANNELS": 168, "CHECKPOINTS": 60, "LEARNING_CURVE": 7200, "NATIVE_SELECTION": 12, "F_CACHE_AND_SUBSET_INFERENCE": 18, "NEW_FIXED_D_DIAGNOSTIC": 324, "NEW_FIXED_D_GEOMETRY": 36, "WHOLE_COST": 40, "WHOLE_COST_SAMPLES": 400, "REGISTERED_ADOPTION": 18, "EXECUTION_COST_AND_COUNTS": 5, "FAILURE_AND_REPAIR": 2, "AUDIT_FAILURE_PRESERVATION": 4}
GENERATOR_SHA = "1044944e04807200978dbf3a828652b33a85b6b7ec85ca1332b57b0ffa074beb"
FIXED_SHA = {"science": "2b271ba9aa6ae359033444b282ddc1becc5d1f60ef063451171e9c93f70db4ff", "audit": "e177652ac34414bccfbfc564da820fca0bde5a462922ceb0edaf04c47081e95d", "cost": "60eb05ebdd1a7f24d82fa58d4dcab192c5c4684552882d1f61f175fc0585a23c", "cost_audit": "e419db2ff01e133a698eec169e546f0ade17f9ae8983c2b5c99d56534259be10"}
FAILURE_SHAS = {"RAW_SUBSET_COMPARISON_AUDIT_v2": "9697c162c30c108dbcc9f73ef51d28ec37537e56126abc063b29ba1482b28589", "RAW_SUBSET_AUDIT_FAILURE_TRACE_REEXECUTION_v1": "d6f49796b94fe0ed38f7be00701aa77308856aaddc199ff06ef467e04a6690ea", "RAW_SUBSET_AUDIT_EXCEPTION_TRACE_v1": "c5d40dfa520087107ff7ffb13e10eb606b6303dd55b57c3e82873303f44ffd1e", "RAW_SUBSET_COST_AUDIT_v1": "42081d93049e78aaa8b313a00dd28f68e292129f9ac3b036832889137c6fb6c2"}
PRIOR_REPORTS = {"matched_projection_v1": "4ad724017706b690e2d19ce4e8bdc326e6663f8ca6b0bf6f157ca4641ef0d3e2", "oracle_rank_addendum_v2": "a75784106b241b7228bd09c6b930b94c2a15097c6fd9c9cde466b12cdcc1d4fe", "decoder_metric_intervention_v2": "1958073281ea0a9c7cd14bacab2b97b8b4a95063f1015a8c94b25bafe85b5fe5", "FIT_target_comparison_v1": "fc89475c76f43f0c30d6a0029d96c138c1997dd510c35600664af4fb5337f77c", "past_constant_gate_v1": "4d53cb0b42f882bc636f4f8e57e5d2c42620382274b7268a2f4345ee4a581fd9", "local_past_decoder_v1": "cd15f1e0fb4614870713141b631b929696f6d46f92a2cccd76556ced2d1bd74b", "shrinkage_matched_native_v1": "c48f0b8d36b882f935e5090f1034406a859df581a43b042357b0b2c043c94e0a"}
DISPLAY = {"robin": "Robin", "beijing_aotizhongxin": "Beijing Aotizhongxin"}
SHORT = {"PCA": "PCA", "FIT_TARGET": "Target", "RAW_SUBSET": "Subset"}
SCIENCE_COUNTS = {"subset_selection_fits_checked": 2, "greedy_candidate_scores_checked": 83, "global_decoder_states_checked": 12, "source_encoder_states_checked": 6, "source_split_contexts_checked": 18, "native_fits_checked": 12, "training_updates_logged_checked": 7200, "checkpoint_states_checked": 60, "checkpoint_TUNE_prediction_arrays_checked": 60, "selected_native_split_prediction_arrays_checked": 36, "Bolt_split_prediction_arrays_checked": 18, "actual_subset_F_split_provenance_checks": 6, "actual_subset_F_model_batch_calls_registered": 103, "reused_mixed_F_cache_lineages_checked": 12, "resolved_native_selection_locks_checked": 12, "cached_future_label_perturbation_checks": 12, "new_fixed_D_output_space_cells_computed": 36, "new_E_Y_diagnostic_predictions_computed": 36, "new_fixed_D_real_method_inequalities_checked": 90, "RAW_bound_inequalities_checked": 0, "current_C_persistence_split_baselines_computed": 6, "score_field_checks": 1800}
COST_COUNTS = {"cost_cells_checked": 40, "recorded_timer_samples_checked": 400, "physical_backend_prechecks_checked": 12, "physical_first8_windows_checked": 96, "native_first8_prediction_checks": 12, "native_selection_locks_checked": 12, "subset_selection_locks_checked": 2, "encoder_state_hashes_checked": 6, "native_projection_state_hashes_checked": 6, "global_decoder_state_hashes_checked": 12, "native_head_state_hashes_checked": 12, "selected_forecast_file_hashes_checked": 54, "canonical_physical_bridges_checked": 2}


def source_contract(text):
    tree = ast.parse(text); functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    generate = ast.unparse(functions["generate"])
    require("not output.exists()" in generate and "output.mkdir(parents=True, exist_ok=False)" in generate, "REPORT_OVERWRITE_REFUSAL")
    require("FRESH_LOCAL_REPORT_DIRECTORY_REQUIRED" in generate, "FRESH_REPORT_OUTPUT_GUARD")
    for name in ("write_csv", "generate"):
        require(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "open" and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "x" for node in ast.walk(functions[name])), "EXCLUSIVE_CSV_OR_MANIFEST_CREATE:" + name)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [alias.name.split(".")[0] for alias in node.names] if isinstance(node, ast.Import) else [str(node.module).split(".")[0]]
            require(not set(modules) & {"torch", "chronos"}, "NO_GENERATOR_MODEL_IMPORT")
        if isinstance(node, ast.Attribute): require(node.attr not in {"load_state_dict", "savez", "savez_compressed", "fit", "predict"}, "NO_GENERATOR_SCIENTIFIC_FIT_OR_MODEL_CALL")
    module_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    require(not any(isinstance(node, ast.Import) and any(alias.name.split(".")[0] == "numpy" for alias in node.names) for node in module_imports), "NUMPY_ONLY_PLOTTING_SCOPE")
    require("Path(path).suffix.lower() == '.json'" in ast.unparse(functions["read"]), "JSON_ONLY_METADATA_READER")
    return {"overwrite_refusal_checked": True, "exclusive_CSV_manifest_creation": True, "generator_imported_by_verifier": False, "NPZ_model_or_training_calls": False}


def terminal(manifest, fixed_sha=None):
    fixed_sha = FIXED_SHA if fixed_sha is None else fixed_sha
    values, bindings = {}, manifest["inputs"]
    for name in INPUTS:
        path = verify_record(bindings[name], ".json")
        require(fixed_sha.get(name) is not None and bindings[name]["sha256"] == fixed_sha[name], "REGISTERED_ACTUAL_JSON_SHA:" + name)
        values[name] = read(path)
        require(values[name]["status"] == ("PASS" if "audit" in name else "COMPLETE") and values[name]["protected_TEST_accessed"] is False, "COMPLETE_OR_PASS_NO_TEST:" + name)
    science, audit, cost, checked_cost = (values[name] for name in INPUTS)
    bind(audit["diagnostic"], bindings["science"], "science PASS")
    bind(checked_cost["cost"], bindings["cost"], "whole cost PASS")
    for item in (cost, checked_cost):
        bind(item["raw_subset_diagnostic"], bindings["science"], "same actual forecasts")
        bind(item["raw_subset_audit"], bindings["audit"], "same science PASS")
        for name in ("source_producer", "source_audit", "prior_partial", "Gram_failure_diagnostic", "numeric_repair_addendum"): bind(item[name], science[name], "unchanged science parent:" + name)
    for key in ("source_producer", "source_audit", "plan", "selection_registry"): bind(audit[key], science[key], "same scientific provenance:" + key)
    for name, filename in (("audit", "audit_raw_subset_comparison_v3.py"), ("cost_audit", "audit_raw_subset_cost_v2.py")):
        require(verify_record(values[name]["audit_source"], ".py").name == filename, "CANONICAL_NEW_INDEPENDENT_AUDIT_SOURCE")
        bind(bindings[name + "_source"], values[name]["audit_source"], "report audit source binding")
    require_counts(audit, SCIENCE_COUNTS, "science"); require_counts(checked_cost, COST_COUNTS, "whole cost")
    require(science["completeness"]["actual"] == science["completeness"]["expected"] and science["failed_paths"] == [] and science["F_before"] == science["F_after"] and science["all_twelve_locks_before_any_DEV_input_or_prediction"] is True, "FULL_FROZEN_SCIENTIFIC_COHORT_AND_LOCK_BARRIER")
    for key in ("future_labels_used_for_subset_or_decoder_fit", "per_origin_decoder_fit", "new_backbone_training", "oracle_used_for_selection", "independent_holdout", "strict_K_only_compression"): require(science[key] is False, "SCIENTIFIC_SCOPE:" + key)
    require(science["FIT_TARGET_E_already_FIT_future_supervised"] is True, "FIT_TARGET_SUPERVISION_NOT_HIDDEN")
    for flag in ("new_oracle_used_for_selection_or_deployment", "MAE_or_physical_MSE_minimum_claimed", "strict_universal_FP32_output_lower_bound_claimed", "old_output_space_floor_reused"): require(audit[flag] is False, "NEW_REAL_D32_MINIMUM_SCOPE:" + flag)
    require(audit["new_output_space_diagnostics_computed_by_this_auditor"] is audit["new_output_minimum_only_normalized_MSE_in_each_new_static_D32_real_rowspace"] is True, "INDEPENDENT_NEW_FIXED_D_DIAGNOSTICS")
    require(tuple(cost["routes"]) == ROUTES and cost["completed_cost_cells"] == 40 and cost["recorded_timer_samples"] == cost["completed_cell_timer_samples"] == 400 and cost["failed_paths"] == [] and cost["F_before"] == cost["F_after"] and cost["loaded_native_heads_count"] == 12 and cost["native_deployment_cost_measured"] is True, "COMPLETE_TEN_ROUTE_ACTUAL_COST")
    require(cost["per_origin_decoder_fit"] is cost["new_encoder_subset_decoder_or_CP_selection"] is False and cost["new_training_updates"] == 0 and cost["fixed_FIT_decoder_loaded_before_timer"] is cost["all_source_file_state_metadata_hashes_outside_timer"] is True, "STATIC_FIT_D_TIMER_SCOPE")
    registry = read(verify_record(science["selection_registry"], ".json"))
    bind(cost["selection_registry"], science["selection_registry"], "unchanged twelve-head registry")
    require(len(registry["conditions"]) == 12 and registry["status"] == "ALL_TWELVE_RESOLVED_BEFORE_DEV" and registry["failed_paths_at_seal"] == [], "TWELVE_SUCCESSFUL_NATIVE_LOCKS")
    for item in registry["conditions"]:
        bind(cost["native_selection_locks"][item["dataset"]][item["space"]][item["seed"]], item["lock"], "cost selected native")
        require(read(verify_record(item["lock"], ".json"))["status"] == "LOCKED", "SUCCESSFUL_LOCK_NOT_FAILED_PATH")
    for dataset in DATASETS: bind(cost["subset_selection_locks"][dataset], science["datasets"][dataset]["subset_selection_lock"], "fixed FIT-only subset")
    prior = read(verify_record(bindings["prior_partial"], ".json")); values["prior_partial"] = prior
    require(prior["status"] == "PARTIAL_COMPLETED" and prior["native_training_fits"] == 8 and prior["subset_selection_fits"] == prior["actual_subset_F_split_inferences"] == 0, "PRESERVED_SUBSET_ZERO_RESULT_NUMERIC_FAILURE")
    identity = audit["mixed_reexecution_identity"]
    require(identity["status"] == "PASS" and identity["mixed_native_conditions_compared"] == 8 and identity["checkpoint_states_and_TUNE_arrays_compared"] == 40 and identity["selected_native_split_arrays_compared"] == 24 and identity["mixed_Bolt_split_arrays_compared"] == 12 and identity["global_D_states_compared"] == 8 and identity["learning_curves_compared"] == 8 and identity["total_physical_native_fit_attempts_across_v1_v2"] == 20 and identity["distinct_registered_native_conditions"] == 12 and identity["new_independent_conditions_from_reexecution"] == 0 and identity["actual_old_and_new_NPZ_logical_bytes_checked"] is True, "REPEATED_EIGHT_MIXED_NOT_NEW_INDEPENDENT_EVIDENCE")
    values["Gram_failure_diagnostic"] = read(verify_record(bindings["Gram_failure_diagnostic"], ".json"))
    require(values["Gram_failure_diagnostic"]["FIT_future_targets_read"] is values["Gram_failure_diagnostic"]["TUNE_DEV_read"] is False, "GRAM_FAILURE_ONLY_FIT_PAST")
    bind(values["Gram_failure_diagnostic"]["partial_result"], science["prior_partial"], "Gram original partial")
    values["preserved_audit_failures"] = []
    for name, digest in FAILURE_SHAS.items():
        record = bindings[name]; require(record["sha256"] == digest, "EXACT_FAILED_ATTEMPT_PRESERVED_SHA")
        failure = read(verify_record(record, ".json"))
        require(failure["status"] == ("CAPTURED_UNCHANGED_AUDITOR_EXCEPTION_TRACE" if name.endswith("EXCEPTION_TRACE_v1") else "FAILED_PRESERVED"), "FAILED_AUDIT_NOT_COUNTED_AS_PASS")
        repair = "MD addendum verified as exact path and SHA instead of JSON parsing; no numeric or timer change" if name == "RAW_SUBSET_COST_AUDIT_v1" else "producer-complete bool and complete-C mask used different variable names in fresh auditor; no formula/tolerance/model change"
        values["preserved_audit_failures"].append({"name": name, **record, **failure, "accepted_scientific_PASS": False, "new_forecast_or_training_executed_for_audit": False, "repair_scope": repair})
    return values


def score_rows(table, base, scores, role):
    for scale in SCALES: table.append({**base, "metric_scale": scale, **scores[scale], "role": role, "MSE_is_real_fixed_D_minimum": base["method"] == "NEW_FIXED_D_MINIMUM" and scale == "normalized", "MAE_is_minimum": False, "physical_MSE_is_minimum": False, "RAW_bound_applied": False})


def independent_tables(values):
    out = {name: [] for name in COUNTS}; science, audit, cost = (values[name] for name in ("science", "audit", "cost"))
    for dataset, (c, k, dev) in DATASETS.items():
        data, verified = science["datasets"][dataset], audit["datasets"][dataset]
        selection = read(verify_record(data["subset_selection_lock"], ".json"))
        for stage in selection["selection_metadata"]["stages"]:
            base = {"dataset": dataset, "stage": stage["stage"], "chosen_original_index": stage["selected_original_index"]}
            out["GREEDY_SELECTION"].append({**base, **{name: value for name, value in stage.items() if name != "candidates"}, "candidate_count": len(stage["candidates"]), "final_selected_indices": selection["selected_indices"], "final_selected_names": selection["selected_channel_names"], "final_greedy_order": selection["selection_order"], "selection_source": selection["FIT_source"], "future_forecast_objective_used": False})
            for candidate in stage["candidates"]:
                out["GREEDY_CANDIDATES"].append({**base, **candidate, "winner": candidate["candidate_original_index"] == stage["selected_original_index"], "exact_tie_rule": "lowest_original_index"})
                for channel in range(c): out["GREEDY_RESPONSE_SCORES"].append({**base, "candidate_original_index": candidate["candidate_original_index"], "response_channel_zero_based": channel, "candidate_indices": candidate["indices"], "MSE": candidate["per_channel_MSE"][channel], "observed_count": candidate["true_observed_counts"][channel], "retained_rank": candidate["retained_centered_design_rank"][channel], "retained_condition": candidate["retained_centered_design_condition"][channel], "centered_singularvalues": candidate["centered_design_singularvalues"][channel]})
        for split in ("fit", "tune", dev):
            for method, item in (("RAW_BOLT", data["raw_scores"][split]), ("CURRENT_C_PERSISTENCE", verified["current_C_persistence_scores"][split])):
                for mask, key in (("all_original_observed", "scores_all_observed"), ("all_C_complete", "scores_complete_mask")): score_rows(out["COMPARISON"], {"dataset": dataset, "space": "NA", "split": split, "backend": "RAW", "method": method, "mask": mask}, item[key], "saved_Raw" if method == "RAW_BOLT" else "currentC_repeated_no_forecaster_and_not_timed")
        for space in SPACES:
            condition, checked = data["spaces"][space], verified["spaces"][space]
            for backend in BACKENDS:
                entry, independent = condition["backends"][backend], checked["backends"][backend]; meta = entry["metadata"]
                require(meta["singular_relative_cutoff"] == 1e-4 and meta["future_TUNE_DEV_or_F_scores_used"] is False, "PAST_OBSERVED_FIXED_D_RULE")
                base = {"dataset": dataset, "space": space, "backend": backend}
                out["GLOBAL_DECODER_STATES"].append({**base, "C": c, "K": k, "state": entry["state_artifact"], "metadata": meta, "independent_fit": independent["independent_fit"], **{name: independent[name] for name in ("deployed_decoder_FP32_values", "deployed_decoder_max_abs", "deployed_decoder_operator_norm", "deployed_decoder_Frobenius_norm")}, "encoder_already_FIT_future_supervised": space == "FIT_TARGET", "per_origin_fit": False})
                keys = ("true_observed_counts", "retained_centered_design_rank", "retained_centered_design_condition", "centered_design_singularvalues", "observed_past_reconstruction_MSE_FP64", "fallback_reason", "learned_slope", "zero_centered_predictor_variance", "constant_observed_response", "no_observed_responses", "one_observed_response", "rank_deficient_slope", "slope_identified_in_retained_full_K_space", "centered_lstsq_fit_executed")
                for channel in range(c): out["GLOBAL_DECODER_CHANNELS"].append({**base, "response_channel_zero_based": channel, **{key: meta[key][channel] for key in keys}, "constant_response_is_normal_fit": True})
            for split in ("fit", "tune", dev):
                base = {"dataset": dataset, "space": space, "split": split}
                raw = data["raw_scores"][split]["scores_all_observed"]["normalized"]; persistence = verified["current_C_persistence_scores"][split]["scores_all_observed"]["normalized"]
                for method in ("BOLT", "NATIVE_s92601", "NATIVE_s92602"):
                    backend = "BOLT_NUMPY" if method == "BOLT" else "NATIVE_TORCH32"
                    item = condition["Bolt"][split] if method == "BOLT" else condition["native"][method[-5:]]["scores"][split]
                    for mask, key in (("all_original_observed", "scores_all_observed"), ("all_C_complete", "scores_complete_mask")): score_rows(out["COMPARISON"], {**base, "backend": backend, "method": method, "mask": mask}, item[key], "fixedE_actual_F_or_fresh_native_actualC_objective_fixedFIT_D")
                    actual = item["scores_all_observed"]["normalized"]
                    out["RELATIVE_TO_RAW"].append({**base, "backend": backend, "method": method, "MSE": actual["mse"], "MAE": actual["mae"], "RAW_MSE": raw["mse"], "RAW_MAE": raw["mae"], "MSE_minus_RAW": actual["mse"] - raw["mse"], "MSE_ratio_to_RAW": actual["mse"] / raw["mse"], "MAE_ratio_to_RAW": actual["mae"] / raw["mae"], "currentC_persistence_MSE": persistence["mse"], "MSE_minus_currentC_persistence": actual["mse"] - persistence["mse"], "independent_generalization_claimed": False})
                bolt = condition["Bolt"][split]
                out["F_CACHE_AND_SUBSET_INFERENCE"].append({**base, "mixed_F_cache_reused": bolt["mixed_F_cache_reused"], "new_K_F_inference_executed": bolt.get("new_K_F_inference_executed", False), "original_F_cache": bolt.get("original_F_cache"), "full_C_slice_diagnostic": bolt.get("full_C_slice_diagnostic"), "full_C_forecast_slice_used_as_new_inference": False, "artifact": bolt["artifact"]})
                for backend in BACKENDS:
                    diagnostic = checked["new_output_space_diagnostics"][split][backend]
                    require(diagnostic["minimum_only_normalized_MSE"] is True and diagnostic["old_output_space_floor_reused"] is diagnostic["RAW_bound_applied"] is False, "SPECIFIC_NEW_D32_REAL_MINIMUM_SCOPE")
                    diagbase = {**base, "backend": backend, "mask": "all_C_complete"}
                    for method, metric, role in (("NEW_FIXED_D_MINIMUM", diagnostic["minimum_solution_scores_complete_C"], "known_future_best_real_numeric_fixedD_rowspace"), ("RAW_BOLT", diagnostic["RAW_scores_same_complete_C"], "same_mask_unconstrained_Raw")): score_rows(out["NEW_FIXED_D_DIAGNOSTIC"], {**diagbase, "method": method}, metric, role)
                    for method, item in {"E_Y": diagnostic["E_Y"], **diagnostic["actual_methods"]}.items():
                        rows = []; score_rows(rows, {**diagbase, "method": method}, item["scores_complete_C"], "known_future_E_Y_diagnostic_only" if method == "E_Y" else "actual_deployable_prediction")
                        for row in rows: row.update({key: item[key] for key in ("real_trace_scores_complete_C", "measured_FP32_rounding_normalized_MSE_allowance", "actual_FP32_minus_minimum", "real_trace_minus_minimum")})
                        out["NEW_FIXED_D_DIAGNOSTIC"] += rows
                    out["NEW_FIXED_D_GEOMETRY"].append({**base, "backend": backend, **{key: value for key, value in diagnostic.items() if key not in ("minimum_solution_scores_complete_C", "RAW_scores_same_complete_C", "E_Y", "actual_methods")}})
            for seed in SEEDS:
                head = condition["native"][seed]; meta = head["metadata"]
                out["NATIVE_SELECTION"].append({"dataset": dataset, "space": space, "seed": seed, "selected_update": meta["selected_update"], "selected_tune": meta["selected_tune"], "state_sha256": meta["selected_state_sha256"], "metadata": meta, "two_random_initializations_claimed": False, "new_head_fits_in_v2": 1, "same_condition_was_executed_in_v1": space != "RAW_SUBSET"})
                for cp in head["checkpoints"]: out["CHECKPOINTS"].append({"dataset": dataset, "space": space, "seed": seed, "update": cp["update"], **cp["tune"], "selected": cp["update"] == meta["selected_update"], "state_sha256": cp["state_sha256"], "state_artifact": cp["state_artifact"], "exact_scored_TUNE_prediction_artifact": cp["tune_prediction_artifact"]})
                out["LEARNING_CURVE"] += [{"dataset": dataset, "space": space, "seed": seed, **point} for point in head["learning_curve"]]
        rows = cost["cost"][dataset]["rows"]
        for row in rows:
            raw_row = unique(rows, route="RAW_BOLT", batch=row["batch"])
            times = [(sample["perf_end_ns"] - sample["perf_start_ns"]) / 1e9 for sample in row["samples"]]
            raw_times = {sample["repeat"]: (sample["perf_end_ns"] - sample["perf_start_ns"]) / 1e9 for sample in raw_row["samples"]}
            paired = [value / raw_times[sample["repeat"]] for value, sample in zip(times, row["samples"])]
            require(len(times) == 10 and row["seconds"] == times and row["paired_RAW_ratios"] == paired and row["median_seconds"] == statistics.median(times) and row["paired_RAW_median_ratio"] == statistics.median(paired), "INDEPENDENT_TIMER_NS_TWO_MEDIAN_ARITHMETIC")
            out["WHOLE_COST"].append({"dataset": dataset, "route": row["route"], "batch": row["batch"], "repeats": 10, "median_seconds": statistics.median(times), "median_seconds_per_window": statistics.median(times) / row["batch"], "ratio_of_RAW_medians": statistics.median(times) / statistics.median(raw_times.values()), "median_paired_RAW_ratio": statistics.median(paired), "seconds": times, "origins": row["origins"], "memory_shared_process": row["memory"], "includes": row["whole_includes"], "excludes": row["whole_excludes"], "memory_scope": row["memory_scope"], "selected_metadata": row["selected_metadata"], "reference": row["reference"], "timed_outputs": row["timed_outputs"], "static_FIT_D_prepared_outside_timer": True, "native_timed": "_NATIVE_s" in row["route"]})
            for index, sample in enumerate(row["samples"]): out["WHOLE_COST_SAMPLES"].append({"dataset": dataset, "route": row["route"], "batch": row["batch"], "repeat": sample["repeat"], "position": sample["position"], "seconds": times[index], "paired_RAW_ratio": paired[index], "sample": sample})
        for route in ROUTES[1:]:
            space = next(name for name in SPACES if route.startswith(name + "_")); method = "BOLT" if route.endswith("_BOLT") else "NATIVE_s" + route[-5:]
            accuracy = unique(out["RELATIVE_TO_RAW"], dataset=dataset, space=space, method=method, split="tune"); latency = unique(out["WHOLE_COST"], dataset=dataset, route=route, batch=8)
            out["REGISTERED_ADOPTION"].append({"dataset": dataset, "route": route, "TUNE_MSE_ratio_to_RAW": accuracy["MSE_ratio_to_RAW"], "TUNE_MAE_ratio_to_RAW": accuracy["MAE_ratio_to_RAW"], "B8_ratio_of_RAW_medians": latency["ratio_of_RAW_medians"], "B8_median_paired_RAW_ratio": latency["median_paired_RAW_ratio"], "all_registered_conditions_pass": accuracy["MSE_ratio_to_RAW"] <= 1.01 and accuracy["MAE_ratio_to_RAW"] <= 1.01 and latency["ratio_of_RAW_medians"] <= .9, "new_selection_used": False, "independent_generalization_claimed": False})
    for name in INPUTS:
        value = values[name]
        training = sum(head["metadata"]["train_wall_seconds"] for data in value.get("datasets", {}).values() for space in data.get("spaces", {}).values() for head in space.get("native", {}).values()) if name == "science" else None
        out["EXECUTION_COST_AND_COUNTS"].append({"stage": name, "status": value["status"], "wall_seconds": value.get("wall_seconds"), "resource_terminal": value.get("resource_terminal"), "totals": value.get("totals", value.get("completeness")), "native_training_wall_seconds_sum": training, "separate_global_D_fit_wall_measured": False, "new_independent_generalization_runs": 0})
    out["EXECUTION_COST_AND_COUNTS"].append({"stage": "preserved_prior_v1", "status": values["prior_partial"]["status"], "native_fits": 8, "updates": 4800, "CP": 40, "actual_subset_F_split_inferences": 0, "actual_native_fits_across_v1_v2": 20, "distinct_native_conditions": 12, "identity": science["mixed_reexecution_identity"], "independent_identity": audit["mixed_reexecution_identity"]})
    for dataset, record in values["Gram_failure_diagnostic"]["datasets"].items(): out["FAILURE_AND_REPAIR"].append({"dataset": dataset, **record["first_Gram_failure"], "failure_scope": "numeric score cancellation before subset selection; no subset forecast result", "old_v1_source_preserved": True, "new_score": "direct centered lstsq coefficient plus explicit residual squared mean; same rcond1e-4 and exact tie rule", "tolerance_relaxed": False})
    out["AUDIT_FAILURE_PRESERVATION"] = values["preserved_audit_failures"]
    require({name: len(rows) for name, rows in out.items()} == COUNTS, "INDEPENDENT_ALL_NINETEEN_CSV_COHORT_COUNTS")
    return out


def row_key(name, row):
    keys = {"COMPARISON": ("dataset", "space", "split", "backend", "method", "mask", "metric_scale"), "RELATIVE_TO_RAW": ("dataset", "space", "split", "method"), "GREEDY_SELECTION": ("dataset", "stage"), "GREEDY_CANDIDATES": ("dataset", "stage", "candidate_original_index"), "GREEDY_RESPONSE_SCORES": ("dataset", "stage", "candidate_original_index", "response_channel_zero_based"), "GLOBAL_DECODER_STATES": ("dataset", "space", "backend"), "GLOBAL_DECODER_CHANNELS": ("dataset", "space", "backend", "response_channel_zero_based"), "CHECKPOINTS": ("dataset", "space", "seed", "update"), "LEARNING_CURVE": ("dataset", "space", "seed", "update"), "NATIVE_SELECTION": ("dataset", "space", "seed"), "F_CACHE_AND_SUBSET_INFERENCE": ("dataset", "space", "split"), "NEW_FIXED_D_DIAGNOSTIC": ("dataset", "space", "split", "backend", "method", "metric_scale"), "NEW_FIXED_D_GEOMETRY": ("dataset", "space", "split", "backend"), "WHOLE_COST": ("dataset", "route", "batch"), "WHOLE_COST_SAMPLES": ("dataset", "route", "batch", "repeat"), "REGISTERED_ADOPTION": ("dataset", "route"), "EXECUTION_COST_AND_COUNTS": ("stage",), "FAILURE_AND_REPAIR": ("dataset",), "AUDIT_FAILURE_PRESERVATION": ("name",)}[name]
    return tuple(str(row[key]) for key in keys)


def check_csv(path, name, expected):
    csv.field_size_limit(max(csv.field_size_limit(), Path(path).stat().st_size))
    with Path(path).open(encoding="utf-8-sig", newline="") as handle: reader = csv.DictReader(handle); actual, fields = list(reader), reader.fieldnames
    mapping = {row_key(name, row): row for row in expected}
    require(len(mapping) == len(expected) == len(actual) == COUNTS[name], "CSV_COUNT_DUPLICATE_OR_OMITTED_ROW:" + name)
    require(fields is not None and len(fields) == len(set(fields)) and set(fields) == {key for row in expected for key in row}, "EXACT_CSV_COLUMN_SET:" + name)
    seen, total = set(), 0
    for row in actual:
        identity = row_key(name, row); require(identity in mapping and identity not in seen, "CSV_UNKNOWN_OR_DUPLICATE_IDENTITY"); seen.add(identity)
        for key in fields:
            expected_value, raw = mapping[identity].get(key), row[key]
            if isinstance(expected_value, (dict, list)): actual_value = json.loads(raw)
            elif expected_value is None: actual_value = None if raw == "" else raw
            elif isinstance(expected_value, bool): actual_value = True if raw == "True" else False if raw == "False" else raw
            elif isinstance(expected_value, int): actual_value = int(raw)
            elif isinstance(expected_value, float): actual_value = float(raw)
            else: actual_value = raw
            total += leaf_check(actual_value, expected_value, name + ":" + repr(identity) + ":" + key)
    return total


def check_MD_numeric(text, tables):
    count = 0
    def check(value):
        nonlocal count
        require(value in text, "MD_NUMERIC_OR_SCOPE_CHANGED:" + value); count += 1
    for dataset, (_, _, dev) in DATASETS.items():
        for split in ("tune", dev):
            def metric(space, method): return unique(tables["COMPARISON"], dataset=dataset, space=space, split=split, method=method, mask="all_original_observed", metric_scale="normalized")["mse"]
            numbers = [metric("NA", "RAW_BOLT"), *[metric(space, "BOLT") for space in SPACES], metric("NA", "CURRENT_C_PERSISTENCE")]
            check(f"{DISPLAY[dataset]+'/'+split.upper():30s} " + "  ".join(f"{value:13.9f}" for value in numbers))
        relative = unique(tables["RELATIVE_TO_RAW"], dataset=dataset, space="RAW_SUBSET", method="BOLT", split=dev)
        check(f"[확인] {DISPLAY[dataset]} {dev.upper()}의 subset/Bolt 정규화 MSE는 {relative['MSE']:.9f}, Raw는 {relative['RAW_MSE']:.9f}로 {relative['MSE_ratio_to_RAW']:.6f}배다. 이 값은 같은 full-observed mask의 비교다.")
        for space in SPACES:
            def diagnostic(method): return unique(tables["NEW_FIXED_D_DIAGNOSTIC"], dataset=dataset, space=space, split=dev, backend="BOLT_NUMPY", method=method, metric_scale="normalized")["mse"]
            raw, floor, bolt, ey = (diagnostic(method) for method in ("RAW_BOLT", "NEW_FIXED_D_MINIMUM", "BOLT", "E_Y"))
            check(f"{DISPLAY[dataset]+'/'+SHORT[space]:26s} {raw:13.9f} {floor:13.9f} {bolt:13.9f} {bolt-floor:13.9f} {ey:13.9f}")
            for seed in SEEDS:
                head = unique(tables["NATIVE_SELECTION"], dataset=dataset, space=space, seed=seed)
                tune = unique(tables["RELATIVE_TO_RAW"], dataset=dataset, space=space, method="NATIVE_s" + seed, split="tune")
                development = unique(tables["RELATIVE_TO_RAW"], dataset=dataset, space=space, method="NATIVE_s" + seed, split=dev)
                check(f"{DISPLAY[dataset]+'/'+SHORT[space]+'/'+seed:31s} {head['selected_update']:6d} {tune['MSE']:13.9f} {development['MSE']:13.9f} {development['MSE_ratio_to_RAW']:11.6f}")
        same_raw = unique(tables["NEW_FIXED_D_DIAGNOSTIC"], dataset=dataset, space="RAW_SUBSET", split=dev, backend="BOLT_NUMPY", method="RAW_BOLT", metric_scale="normalized")["mse"]
        above = sum(unique(tables["NEW_FIXED_D_DIAGNOSTIC"], dataset=dataset, space=space, split=dev, backend="BOLT_NUMPY", method="NEW_FIXED_D_MINIMUM", metric_scale="normalized")["mse"] > same_raw for space in SPACES)
        subset_ey = unique(tables["NEW_FIXED_D_DIAGNOSTIC"], dataset=dataset, space="RAW_SUBSET", split=dev, backend="BOLT_NUMPY", method="E_Y", metric_scale="normalized")["mse"]
        check(f"[확인] {DISPLAY[dataset]} {dev.upper()}의 세 NumPy 고정 D 중 {above}/3개는 미래 잠재값을 자유롭게 고르는 실수 minimum 자체가 같은 mask Raw({same_raw:.9f})보다 높다. Subset의 E(Y) 진단은 {subset_ey:.9f}다.")
        selection = next(row for row in tables["GREEDY_SELECTION"] if row["dataset"] == dataset)
        check(f"{DISPLAY[dataset]:23s} {selection['final_selected_indices']} / {', '.join(selection['final_selected_names'])}    {selection['final_greedy_order']}")
        for route in ROUTES:
            one = unique(tables["WHOLE_COST"], dataset=dataset, route=route, batch=1); eight = unique(tables["WHOLE_COST"], dataset=dataset, route=route, batch=8)
            check(f"{DISPLAY[dataset]+'/'+route:46s} {one['median_seconds']*1000:12.5f} {eight['median_seconds']*1000:12.5f} {eight['median_seconds_per_window']*1000:14.7f} {eight['ratio_of_RAW_medians']:15.7f} {eight['median_paired_RAW_ratio']:18.7f}")
    diffs = [row["full_C_slice_diagnostic"]["max_abs"] for row in tables["F_CACHE_AND_SUBSET_INFERENCE"] if row["space"] == "RAW_SUBSET"]
    check(f"[확인] 실제 새 K-channel F와 기존 full-C Raw의 선택 채널 forecast slice의 최대 절대 차이는 6개 split에서 {min(diffs):.9g}–{max(diffs):.9g}였다.")
    check(f"[확인] 원 동시 기준(TUNE MSE/MAE 각각 Raw×1.01 이하, 같은 시행 B8 ratio-of-medians≤0.90)을 통과한 자료·경로는 {sum(row['all_registered_conditions_pass'] for row in tables['REGISTERED_ADOPTION'])}/18개다.")
    for row in tables["FAILURE_AND_REPAIR"]: check(f"{DISPLAY[row['dataset']]:20s} {row['Gram_MSE']: .12e} {row['original_negative_tolerance']: .12e} {row['direct_centered_residual_MSE']: .12e}")
    for phrase in ("같은 affine native class와 600update", "current-C anchor는 여전히 읽으므로 strict K-only가 아니다", "이전 DEV·oracle를 본 뒤 진행한 탐색적 대조", "MAE·이질적 원단위 MSE를 각 목적의 최소라고 부르지 않는다", "FP32의 모든 가능한 출력에 대한 엄밀 하한", "400개의 독립 데이터가 아니다", "방법 고유 메모리 절감으로 해석하지 않는다", "20회/12000update/100CP", "distinct 등록 조건은12개", "JSONDecodeError로 FAILED_PRESERVED", "모델·예측·학습·수식·공차를 바꾼 복구가 아니며", "최종 채택 근거는 별도로 결속된 새 PASS"):
        require(phrase in text, "REQUIRED_RESEARCH_SCOPE_OR_FAILURE_PRESERVATION:" + phrase)
    return count


def check_files(folder, manifest):
    assets = {"REPORT_KO.md"} | {"tables/" + name + ".csv" for name in COUNTS} | {"figures/" + name + "." + suffix for name in FIGURES for suffix in ("png", "svg")}
    declared = set()
    for record in manifest["files"]:
        path = (folder / record["path"]).resolve()
        require(path.is_relative_to(folder) and record["path"] not in declared, "ARTIFACT_ESCAPE_OR_DUPLICATE")
        declared.add(record["path"])
        require(path.is_file() and path.stat().st_size == record["bytes"] > 0 and sha(path) == record["sha256"], "ARTIFACT_SHA_BYTES_OR_EXISTENCE:" + record["path"])
    actual = {path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file() and path.name != "MANIFEST.json"}
    require(actual == declared == assets, "ALL_32_REPORT_FILES_EXACT_NO_OMITTED_OR_EXTRA_ARTIFACTS")
    return len(assets)


def relative_links(report_path):
    found, embedded = links(report_path)
    require(all(not Path(row["target"]).is_absolute() and ":" not in row["target"] for row in found), "GITHUB_RELATIVE_LOCAL_LINKS_REQUIRED")
    return found, embedded


def check_figures(folder):
    from PIL import Image
    parsed = []
    for name in FIGURES:
        path = folder / ("figures/" + name + ".png")
        with Image.open(path) as image: require(image.format == "PNG" and min(image.size) > 0, "VALID_PNG_DIMENSIONS"); image.verify()
        with Image.open(path) as image: image.load(); dimensions = list(image.size)
        svg = folder / ("figures/" + name + ".svg"); require(ET.parse(svg).getroot().tag == "{http://www.w3.org/2000/svg}svg", "PARSEABLE_SVG")
        parsed.append({"name": name, "PNG_dimensions": dimensions, "SVG_parsed": True})
    return parsed


def verify(folder, *, fixed_sha=None, generator_sha=GENERATOR_SHA):
    folder = Path(folder).resolve(); manifest_path = folder / "MANIFEST.json"; manifest = read(manifest_path)
    require(manifest["status"] == "COMPLETE" and manifest["scientific_NPZ_or_model_read"] is manifest["new_fit_F_or_training_executed"] is manifest["protected_TEST_accessed"] is manifest["independent_generalization_claimed"] is False and manifest["known_future_diagnostics_separate_from_deployment"] is True, "COMPLETE_REPORT_WITH_EXPLICIT_METADATA_AND_DIAGNOSTIC_SCOPE")
    generator = verify_record(manifest["source"], ".py")
    require(generator.name == "make_raw_subset_report_v1.py" and manifest["source"]["sha256"] == generator_sha, "EXACT_FROZEN_REPORT_GENERATOR")
    source = source_contract(generator.read_text(encoding="utf-8-sig"))
    bind(manifest["inputs"]["report_source"], manifest["source"], "report source provenance")
    for record in manifest["inputs"].values():
        require(Path(record["path"]).suffix.lower() in (".json", ".md", ".py"), "MANIFEST_METADATA_INPUTS_ONLY_NO_NPZ_OR_MODELS")
        verify_record(record)
    require(manifest["inputs"]["cost_auditor_format_repair"]["sha256"] == "8f2bada1aea0da5467763d0d86c19431fdb88083a5c46d7a5e9bd063e0d381b2", "PRESERVED_COST_AUDITOR_FORMAT_REPAIR")
    require(set(manifest["preserved_prior_reports"]) == set(PRIOR_REPORTS), "SEVEN_ORIGINAL_REPORTS_PRESERVED")
    for name, record in manifest["preserved_prior_reports"].items():
        require(record["sha256"] == PRIOR_REPORTS[name], "ORIGINAL_REPORT_FIXED_SHA:" + name)
        verify_record(record, ".md")
    values = terminal(manifest, fixed_sha); tables = independent_tables(values)
    require(manifest["table_row_counts"] == COUNTS and manifest["figure_png_count"] == manifest["figure_svg_count"] == 6, "EXACT_NINETEEN_TABLES_AND_SIX_FIGURE_PAIRS")
    file_count = check_files(folder, manifest)
    fields = {name: check_csv(folder / "tables" / (name + ".csv"), name, rows) for name, rows in tables.items()}
    report_path = folder / "REPORT_KO.md"; text = report_path.read_text(encoding="utf-8-sig"); numeric = check_MD_numeric(text, tables)
    found, embedded = relative_links(report_path); linked = {Path(row["path"]).resolve() for row in found}
    provenance = [*manifest["inputs"].values(), *manifest["preserved_prior_reports"].values()]
    require(all(resolve(record["path"]) in linked and record["sha256"] in text for record in provenance), "ALL_BOUND_JSON_SOURCE_REPAIR_FAILURE_AND_PRIOR_LINKS_WITH_SHA")
    require(all(folder / ("tables/" + name + ".csv") in linked for name in COUNTS), "ALL_NINETEEN_CSV_LINKS")
    require(set(embedded) == {folder / ("figures/" + name + ".png") for name in FIGURES} and len(embedded) == 6, "SIX_EXACT_EMBEDDED_PNGS")
    parsed = check_figures(folder)
    return {"status": "PASS", "report": {"path": str(report_path), "sha256": sha(report_path)}, "manifest": {"path": str(manifest_path), "sha256": sha(manifest_path)}, "audit_source": {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)}, "inputs": manifest["inputs"], "source_contract": source, "files_checked": file_count, "CSV_rows": COUNTS, "CSV_primitive_fields_checked": fields, "total_CSV_primitive_fields_checked": sum(fields.values()), "MD_numeric_rows_and_paragraphs_checked": numeric, "local_links_checked": len(found), "local_links": found, "PNG_count": 6, "SVG_count": 6, "parsed_figures": parsed, "preserved_failed_audit_attempts_checked": 4, "timer_two_ratios_and_adoption_recomputed": True, "report_generator_imported": False, "NPZ_or_model_opened": False, "training_F_or_actual_timer_replayed": False, "protected_TEST_accessed": False, "scope": "Exact bound JSON to typed CSV and independent arithmetic, Markdown figures/numeric/relative provenance links and parsed image containers. Scientific arrays and models are opaque; visual inspection/HTTP rendering are separate."}


def execute(folder, output):
    output = Path(output).resolve(); require(not output.exists(), "FRESH_REPORT_AUDIT_OUTPUT")
    receipt = verify(folder)
    with output.open("x", encoding="utf-8") as handle: json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")
    print(json.dumps({name: receipt[name] for name in ("status", "files_checked", "total_CSV_primitive_fields_checked", "local_links_checked", "PNG_count", "SVG_count")}))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--report", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); raise SystemExit(execute(args.report, args.output))
