"""Independent JSON-to-CSV, scope, links and image audit; no generator/scientific IO."""
from pathlib import Path
import argparse
import ast
import csv
import json
import math
import statistics
import xml.etree.ElementTree as ET

from verify_fit_target_report_v1 import require, sha, read, resolve, verify_record, leaf_check, links

HERE = Path(__file__).resolve().parent
DATASETS = {"robin": (17, 4, "val"), "beijing_aotizhongxin": (11, 2, "eval")}
PROJECTIONS, SEEDS = ("PCA_PINV", "FIT_TARGET_PINV"), ("92601", "92602")
BACKENDS = {"BOLT_NUMPY": ("BOLT",), "NATIVE_TORCH32": ("NATIVE_s92601", "NATIVE_s92602")}
SCALES = ("normalized", "physical_per_channel_diagnostic")
CANDIDATES = ("STATIC", "LOCAL", "RHO_0.001", "RHO_0.01", "RHO_0.1", "RHO_1", "RHO_10")
SELECTED_ROUTES = ("RAW_BOLT", "PCA_PINV_BOLT", "FIT_TARGET_PINV_BOLT", "PCA_SHRUNK_BOLT", "FIT_TARGET_SHRUNK_BOLT")
MATCHED_ROUTES = ("RAW_BOLT", "PCA_SHRUNK_BOLT", "FIT_TARGET_SHRUNK_BOLT", "PCA_SHRUNK_NATIVE_s92601", "PCA_SHRUNK_NATIVE_s92602", "FIT_TARGET_SHRUNK_NATIVE_s92601", "FIT_TARGET_SHRUNK_NATIVE_s92602")
INPUTS = ("shrinkage", "shrinkage_audit", "space", "selected_cost", "selected_cost_audit", "matched", "matched_audit", "matched_cost", "matched_cost_audit", "repeat_matched_cost", "repeat_matched_cost_audit")
FIGURES = ("01_TUNE_rho_selection", "02_actual_Bolt_decoder_comparison", "03_frozen_and_matched_native", "04_selected_output_space_and_forecast_gap", "05_separate_actual_whole_cost_cohorts", "06_native_TUNE_checkpoints")
COUNTS = {"SCORES": 552, "RHO_CANDIDATES": 480, "TUNE_SELECTION": 28, "RHO_LOCKS": 4, "SELECTED_D_BACKENDS": 24,
    "SELECTED_D_WINDOWS": 3264, "SELECTED_PAST_RECONSTRUCTION": 48, "CURRENT_C_PERSISTENCE": 48, "CHECKPOINTS": 40,
    "LEARNING_CURVE": 4800, "NATIVE_TRAINING": 8, "NATIVE_LOCKS": 8, "OUTPUT_DIAGNOSTIC": 264, "OUTPUT_GEOMETRY": 24,
    "SELECTED_F_ONLY_COST": 20, "SELECTED_F_ONLY_COST_SAMPLES": 120, "MATCHED_WHOLE_COST": 28, "MATCHED_WHOLE_COST_SAMPLES": 196,
    "REPEAT_MATCHED_WHOLE_COST": 28, "REPEAT_MATCHED_WHOLE_COST_SAMPLES": 196, "REGISTERED_ADOPTION": 24, "COST_FAILURE_PROVENANCE": 1}
GENERATOR_SHA = "58ea5384e4d2d04b54ad69ac828784b919da251e31a1f82b079e816889178211"
FIXED_SHA = {"shrinkage": "aa553512965bf9253fae1983424e2fcb4f2916cc95ec12e9ed671ae325dbd532",
    "shrinkage_audit": "145999ea8bc7068eba08f08039287a33097cdb4e9ec8dab002f4ebc7630fdda9",
    "space": "6093d1b2cd7b61c1549aa51ac234476404b13a24edc492a42764ccac0c67deff",
    "selected_cost_audit": "43e5986d250a707cdb95958e8887184f46794204f10e4d2767acdc7ab83424c6",
    "matched": "e13587c86a9f28b3dccc5bf445cb85cd855a8a20a0ce1d9caad7dd57b93ba77a",
    "matched_audit": "0eb57a67b2ee3c25a7a5309efbdb62e55f756fb0c9552bd251b50f2ea6505077",
    "matched_cost": "a67df4dacf9fdf354dd8e16cfeebb22017f8d1dcf3551e467b170fefa025457e",
    "matched_cost_audit": "cc2b9e432a354b62d362ede8f8fb681e4a9c5d4fbfd65587190c6dabb48bc31d",
    "repeat_matched_cost": "e1b0db88225100fa05b7d74178801dc0c4eb77c61e03c8f0c9b64fd91489eed8",
    "repeat_matched_cost_audit": "c01026d2314f0700335aa473a85e5bd2dd34b0296af520a20e40612574592af7"}
COST_COUNTS = {"cost_cells_checked": 28, "recorded_timer_samples_checked": 196, "physical_backend_prechecks_checked": 8,
    "physical_first8_window_fits_checked": 64, "native_first8_prediction_checks": 8, "native_selection_locks_checked": 8,
    "old_Bolt_rho_lock_files_checked": 4, "native_and_E_state_hashes_checked": 12, "selected_forecast_file_hashes_checked": 60, "canonical_physical_bridges_checked": 2}


def bind(actual, expected, label):
    require(resolve(actual["path"]) == resolve(expected["path"]) and actual["sha256"] == expected["sha256"], "EXACT_METADATA_BINDING:" + label)
    path = verify_record(actual)
    for value in (actual, expected):
        if "bytes" in value: require(value["bytes"] == path.stat().st_size, "OPTIONAL_BOUND_BYTES_CHANGED:" + label)
    return path


def require_counts(value, expected, label):
    require(all(value.get("totals", {}).get(key) == number for key, number in expected.items()), "INDEPENDENT_COHORT_COUNTS:" + label)


def source_contract(text):
    tree = ast.parse(text)
    funcs = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    generate = ast.unparse(funcs["generate"])
    require("require(not output.exists(), 'FRESH_INTEGRATED_REPORT_DIRECTORY')" in generate and "output.mkdir(parents=True, exist_ok=False)" in generate, "REPORT_OVERWRITE_REFUSAL_SOURCE")
    require(".open('x', encoding='utf-8')" in generate and ".open('x', encoding='utf-8', newline='')" in ast.unparse(funcs["write_csv"]), "EXCLUSIVE_MANIFEST_AND_CSV_CREATION")
    imported = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    require(not any((isinstance(node, ast.Import) and any(alias.name.split('.')[0] in ("numpy", "torch", "chronos") for alias in node.names)) or (isinstance(node, ast.ImportFrom) and node.module and node.module.split('.')[0] in ("numpy", "torch", "chronos")) for node in imported), "NO_SCIENTIFIC_GENERATOR_IMPORTS")
    require(not any(isinstance(node, ast.Attribute) and node.attr in ("load_state_dict", "fit", "predict", "savez", "savez_compressed") for node in ast.walk(tree)), "NO_GENERATOR_MODEL_MATRIX_OR_FIT_CALLS")
    return {"existing_report_directory_refused": True, "exclusive_CSV_and_manifest_creation": True, "scientific_runtime_imported_by_verifier": False}


def terminal(manifest, fixed_sha=None):
    fixed_sha = FIXED_SHA if fixed_sha is None else fixed_sha
    require(set(INPUTS).issubset(manifest["inputs"]), "ELEVEN_BOUND_INPUT_JSONS")
    values = {}
    for name in INPUTS:
        record = manifest["inputs"][name]
        path = verify_record(record, ".json")
        if name in fixed_sha: require(record["sha256"] == fixed_sha[name], "REGISTERED_ACTUAL_JSON_SHA:" + name)
        value = read(path); values[name] = value
        require(value["status"] == ("PASS" if name.endswith("audit") or name == "space" else "COMPLETE") and value["protected_TEST_accessed"] is False, "COMPLETE_OR_PASS_WITHOUT_TEST:" + name)
    shrink, matched = values["shrinkage"], values["matched"]
    bindings = manifest["inputs"]
    for audit, diagnostic, member in (("shrinkage_audit", "shrinkage", "diagnostic"), ("selected_cost_audit", "selected_cost", "cost"), ("matched_audit", "matched", "diagnostic"), ("matched_cost_audit", "matched_cost", "cost"), ("repeat_matched_cost_audit", "repeat_matched_cost", "cost")):
        bind(values[audit][member], bindings[diagnostic], audit)
    for name in INPUTS:
        if name != "shrinkage":
            for key in ("producer", "fit_target_audit"): bind(values[name][key], shrink[key], name + ":" + key)
        if name not in ("shrinkage", "shrinkage_audit"):
            bind(values[name]["shrinkage_diagnostic"], bindings["shrinkage"], name + ":shrinkage")
            bind(values[name]["shrinkage_audit"], bindings["shrinkage_audit"], name + ":shrinkage_audit")
    for name in ("matched_cost", "matched_cost_audit", "repeat_matched_cost", "repeat_matched_cost_audit"):
        bind(values[name]["matched_native_diagnostic"], bindings["matched"], name + ":matched")
        bind(values[name]["matched_native_audit"], bindings["matched_audit"], name + ":matched_audit")
        require(values[name]["old_Bolt_rho_selection_locks"] == shrink["selection_locks"] and values[name]["selection_locks"] == matched["condition_selection_locks"], "SAME_FOUR_RHO_AND_EIGHT_CP_LOCKS")
    codes = {"shrinkage_audit": "audit_local_decoder_shrinkage_v1.py", "space": "audit_selected_shrinkage_output_space_v1.py", "selected_cost_audit": "audit_selected_shrinkage_cost_v1.py", "matched_audit": "audit_matched_shrunk_native_v1.py", "matched_cost_audit": "audit_matched_shrunk_native_cost_v1.py", "repeat_matched_cost_audit": "audit_matched_shrunk_native_cost_v1.py"}
    for name, filename in codes.items(): require(verify_record(values[name]["audit_source"], ".py").name == filename, "CANONICAL_INDEPENDENT_SOURCE:" + name)
    require_counts(values["shrinkage_audit"], {"positive_rho_backend_states_checked": 88, "window_fits_checked": 13504, "selection_locks_checked": 4, "FIT_TUNE_positive_rho_prediction_arrays_checked": 120, "selected_DEV_prediction_arrays_checked": 12, "score_field_checks": 6960}, "shrinkage")
    require_counts(values["space"], {"selected_output_space_cells_checked": 24, "E_Y_diagnostic_predictions_checked": 24, "actual_predictions_checked": 36, "real_output_space_inequalities_checked": 60, "RAW_bound_inequalities_checked": 0, "score_field_checks": 864, "selected_D_origin_matrices_checked": 3264}, "selected D")
    require_counts(values["selected_cost_audit"], {"cost_cells_checked": 20, "recorded_timer_samples_checked": 120, "physical_shrunk_prechecks_checked": 4, "physical_shrunk_first8_window_fits_checked": 32, "selected_rule_lock_files_checked": 4}, "F-only cost")
    for name in ("matched_cost_audit", "repeat_matched_cost_audit"): require_counts(values[name], COST_COUNTS, name)
    require_counts(values["matched_audit"], {"native_fits_checked": 8, "training_updates_logged_checked": 4800, "checkpoint_states_checked": 40, "checkpoint_TUNE_prediction_arrays_checked": 40, "selected_FIT_TUNE_DEV_prediction_arrays_checked": 24, "source_backend_contexts_checked": 12, "preserved_control_prediction_arrays_checked": 48, "persistence_baseline_contexts_computed": 12, "unique_persistence_split_cells_computed": 6, "selection_locks_checked": 8, "resolved_registry_conditions_checked": 8, "future_label_perturbation_scope_checks": 8, "score_field_checks": 2064}, "matched native")
    require(shrink["future_labels_used_for_decoder_fit"] is shrink["development_positive_rho_sweep_executed"] is shrink["native_specific_selection"] is False and shrink["TUNE_future_labels_used_for_normal_hyperparameter_selection"] is True, "PAST_FIT_NORMAL_TUNE_SELECTION_SCOPE")
    space = values["space"]
    require(space["minimum_claim_only_normalized_MSE_in_registered_real_D32_numeric_rowspace"] is True and space["strict_lower_bound_over_all_FP32_outputs_claimed"] is space["MAE_or_physical_MSE_minimum_claimed"] is space["future_labels_used_to_fit_or_modify_D"] is False, "FUTURE_KNOWN_DIAGNOSTIC_SCOPE")
    require(matched["all_eight_condition_locks_saved_before_any_new_DEV_prediction"] is matched["all_conditions_resolved_registry_saved_before_any_DEV_input_load"] is True and matched["rho_selection_repeated"] is matched["rho_selected_by_native"] is matched["new_encoder_decoder_fit"] is matched["new_backbone_training"] is False and not matched["failed_paths"], "FRESH_MATCHED_CP_LOCKS_AND_FIXED_D_RHO")
    for value in (shrink, matched): require(value["independent_holdout"] is False, "EXPOSED_DEV_SCOPE")
    for name, routes, cells, samples in (("selected_cost", SELECTED_ROUTES, 20, 120), ("matched_cost", MATCHED_ROUTES, 28, 196), ("repeat_matched_cost", MATCHED_ROUTES, 28, 196)):
        value = values[name]
        require(tuple(value["routes"]) == routes and value["completed_cost_cells"] == cells and value["recorded_timer_samples"] == value["completed_cell_timer_samples"] == samples and not value["failed_paths"], "SEPARATE_COMPLETE_COST_COHORT:" + name)
        require(value["new_training_updates"] == 0 and value["F_before"] == value["F_after"] and value["local_ridge_and_implemented_diagnostics_inside_timer"] is value["fit_input_binding_hashes_inside_timer"] is True and value["local_coefficients_cached_inside_timed_pipeline"] is False, "REGISTERED_ACTUAL_COST_BOUNDARY:" + name)
    require(values["selected_cost"]["native_heads_loaded_or_timed"] is False and values["matched_cost"]["native_deployment_cost_measured"] is values["repeat_matched_cost"]["native_deployment_cost_measured"] is True, "F_ONLY_AND_MATCHED_NATIVE_DISTINCT")
    first, repeat = values["matched_cost"], values["repeat_matched_cost"]
    require(first["F_before"] == repeat["F_before"] and [{key: item[key] for key in ("path", "sha256")} for item in first["source_snapshots"]] == [{key: item[key] for key in ("path", "sha256")} for item in repeat["source_snapshots"]] and resolve(bindings["matched_cost"]["path"]) != resolve(bindings["repeat_matched_cost"]["path"]), "TWO_PRESERVED_SAME_PROTOCOL_PROCESS_COHORTS")
    for dataset in DATASETS:
        bridge = first["canonical_physical_bridge_bindings"][dataset]
        require(resolve(bridge["canonical_record"]["path"]) == resolve(bridge["original_alias"]["path"]) and bridge["canonical_record"]["sha256"] == bridge["original_alias"]["sha256"] and bridge["same_physical_file_path_and_SHA"] is bridge["complete_array_manifest_preserved"] is True and bridge["loader_file_SHA_member_shape_dtype_SHA_checks_relaxed"] is bridge["numeric_tolerance_changed"] is False, "STRICT_OPAQUE_PHYSICAL_BRIDGE_METADATA_ONLY")
        for projection in PROJECTIONS:
            rho = shrink["selection_locks"][dataset][projection]; verify_record(rho, ".json")
            require(space["selection_locks"][dataset][projection] == rho, "SELECTED_SPACE_RHO_LOCK")
            for seed in SEEDS: verify_record(matched["condition_selection_locks"][dataset][projection][seed], ".json")
    for name in ("failed_matched_cost_v1", "failed_cost_identity", "audit_invocation_failure"):
        values[name] = read(verify_record(bindings[name], ".json"))
    failure, identity, invocation = (values[name] for name in ("failed_matched_cost_v1", "failed_cost_identity", "audit_invocation_failure"))
    require(failure["status"] == "PARTIAL_COMPLETED" and failure["completed_cost_cells"] == failure["recorded_timer_samples"] == 0 and failure["protected_TEST_accessed"] is False, "PRESERVED_ZERO_SAMPLE_IO_FAILURE")
    require(identity["status"] == "CONFIRMED_PRESERVED_IMPLEMENTATION_IO_CONTRACT_FAILURE" and identity["actual_timer_samples"] == 0 and identity["method_or_training_failure_inferred"] is False, "IO_CONTRACT_NOT_PERFORMANCE_FAILURE")
    require(invocation["status"] == "CONFIRMED_PRESERVED_AUDITOR_INVOCATION_PATH_FAILURE" and invocation["scientific_inputs_sources_or_outputs_changed"] is invocation["new_F_native_forward_or_training_or_ridge_fit"] is invocation["new_cost_timing"] is False, "PRESERVED_ABSOLUTE_INVOCATION_REPAIR_WITHOUT_RETIMING")
    bind(invocation["audit_source"], values["matched_cost_audit"]["audit_source"], "unchanged invoked auditor")
    verify_record(invocation["failed_audit"], ".json")
    for name in ("matched_cost", "matched_cost_audit", "repeat_matched_cost", "repeat_matched_cost_audit"): bind(values[name]["prior_failed_cost_attempt"], bindings["failed_matched_cost_v1"], "preserved failed cost")
    for name in ("physical_bridge_repair_addendum", "cost_repeat_plan"): verify_record(bindings[name], ".md")
    return values


def stat(values):
    require(len(values) > 0 and all(type(value) in (int, float) and math.isfinite(value) for value in values), "FINITE_NONEMPTY_METADATA_SUMMARY")
    return {"min": min(values), "median": statistics.median(values), "max": max(values), "mean": statistics.mean(values)}


def metrics(table, base, scores, **flags):
    for scale in SCALES: table.append({**base, "metric_scale": scale, **scores[scale], **flags})


def expected_tables(values):
    out = {name: [] for name in COUNTS}
    shrink, space, matched = (values[name] for name in ("shrinkage", "space", "matched"))
    for dataset, (c, k, dev) in DATASETS.items():
        for projection in PROJECTIONS:
            record = shrink["selection_locks"][dataset][projection]; lock = read(resolve(record["path"]))
            label, rho = lock["selected_candidate"], lock["selected_rho"]
            out["RHO_LOCKS"].append({"dataset": dataset, "projection": projection, "C": c, "K": k, "rho": rho, "selected_candidate": label, "lock_record": record, "lock": lock, "DEV_used": False, "native_specific_selection": False})
            tune = shrink["datasets"][dataset]["splits"]["tune"]["routes"][projection]
            candidates = {**tune["controls"], **tune["positive_rho_candidates"]}
            for name in CANDIDATES:
                score = candidates[name]["methods"]["BOLT"]["scores_all_observed"]["normalized"]
                out["TUNE_SELECTION"].append({"dataset": dataset, "projection": projection, "candidate": name, "selected": name == label, "normalized_MSE": score["mse"], "normalized_MAE": score["mae"], "rho": candidates[name].get("rho"), "selection_metric": "normalized_MSE", "MAE_used_for_selection": False})
            for split in ("fit", "tune", dev):
                cell = shrink["datasets"][dataset]["splits"][split]; route = cell["routes"][projection]
                if projection == "PCA_PINV":
                    for mask, suffix in (("all_original_observed", "all_observed"), ("all_C_complete", "complete_mask")):
                        metrics(out["SCORES"], {"dataset": dataset, "projection": "NA", "split": split, "model": "RAW_BOLT", "variant": "RAW", "mask": mask}, cell["RAW"]["scores_" + suffix])
                selected = route["positive_rho_candidates"][label]
                for name in ("STATIC", "LOCAL", label):
                    candidate = route["controls"][name] if name in ("STATIC", "LOCAL") else selected
                    for method, scored in candidate["methods"].items():
                        for mask, suffix in (("all_original_observed", "all_observed"), ("all_C_complete", "complete_mask")):
                            metrics(out["SCORES"], {"dataset": dataset, "projection": projection, "split": split, "model": method, "variant": name, "mask": mask}, scored["scores_" + suffix], native_head_frozen_intervention_not_reoptimized=method.startswith("NATIVE"))
                if split in ("fit", "tune"):
                    for name, candidate in route["positive_rho_candidates"].items():
                        for method, scored in candidate["methods"].items():
                            for mask, suffix in (("all_original_observed", "all_observed"), ("all_C_complete", "complete_mask")):
                                metrics(out["RHO_CANDIDATES"], {"dataset": dataset, "projection": projection, "split": split, "candidate": name, "rho": candidate["rho"], "model": method, "mask": mask}, scored["scores_" + suffix])
                for backend in BACKENDS:
                    branch = selected["backends"][backend]; meta = branch["state_metadata"]
                    geo = space["datasets"][dataset]["splits"][split]["routes"][projection]["backends"][backend]
                    base = {"dataset": dataset, "projection": projection, "split": split, "backend": backend, "rho": rho}
                    valid = [value for row, flags in zip(meta["ridge_matrix_condition"], meta["ridge_matrix_condition_valid"]) for value, flag in zip(row, flags) if flag]
                    out["SELECTED_D_BACKENDS"].append({**base, "windows": meta["windows"], "C": c, "K": k, "max_abs_D": stat(meta["max_abs_decoder_per_window"]), "operator_norm": stat(meta["decoder_operator_norm_per_window"]), "Frobenius_norm": stat(meta["decoder_Frobenius_norm_per_window"]), "valid_ridge_condition": stat(valid) if valid else None, "zero_trace_prior_count": sum(map(sum, meta["zero_trace_prior_slope_retained"])), "no_observation_prior_count": sum(map(sum, meta["no_observation_prior_with_current_intercept"])), "state_record": branch["state_artifact"], "mask_scope": "true observed past only", "absolute_gain_bound_claimed": False})
                    metrics(out["SELECTED_PAST_RECONSTRUCTION"], base, branch["past_reconstruction_scores_observed_only"], future_forecast_task=False, reconstruction_is_in_sample_known_past=True)
                    for i in range(meta["windows"]):
                        out["SELECTED_D_WINDOWS"].append({**base, "window_index_zero_based": i, "max_abs_D": meta["max_abs_decoder_per_window"][i], "operator_norm": meta["decoder_operator_norm_per_window"][i], "Frobenius_norm": meta["decoder_Frobenius_norm_per_window"][i], "output_numeric_rank": geo["rank_D_per_origin"][i], "observed_count_per_C": meta["true_observed_counts"][i], "lambda_summary": stat(meta["actual_lambda"][i]), "covariance_trace_summary": stat(meta["local_covariance_trace"][i]), "origin_values_not_invented": True})
                    minimum = geo["minimum_normalized_MSE_real_numeric_rowspace"]
                    out["OUTPUT_GEOMETRY"].append({**base, "minimum_normalized_MSE": minimum, "numeric_relative_singular_cutoff": geo["numeric_rowspace_relative_singular_cutoff"], "complete_target_positions": space["datasets"][dataset]["splits"][split]["complete_target_positions"], "per_origin_rank": geo["rank_D_per_origin"], "per_origin_certificates": geo["certificates"], "RAW_bound_applied": False})
                    diagnostic = {"RAW_BOLT": geo["RAW_scores_same_complete_C"], "SELECTED_D_MINIMUM": geo["minimum_solution_scores_complete_C"], "E_Y": geo["E_Y_diagnostic"]["scores_complete_C"], **{name: item["scores_complete_C"] for name, item in geo["actual_methods"].items()}}
                    if backend == "NATIVE_TORCH32":
                        for seed in SEEDS: diagnostic["MATCHED_NATIVE_s" + seed] = matched["datasets"][dataset]["routes"][projection]["native"][seed]["scores"][split]["scores_complete_mask"]
                    for method, scored in diagnostic.items():
                        metrics(out["OUTPUT_DIAGNOSTIC"], {**base, "model": method, "mask": "all_C_complete"}, scored, minimum_only_normalized_MSE=method == "SELECTED_D_MINIMUM", MAE_is_minimum=False, physical_MSE_is_minimum=False, actual_normalized_MSE_minus_known_real_minimum=scored["normalized"]["mse"] - minimum, known_future_diagnostic_not_deployable=method in ("SELECTED_D_MINIMUM", "E_Y"), RAW_bound_applied=False, new_matched_real_trace_inequality_verified_in_report=False)
                persistence = values["matched_audit"]["datasets"][dataset]["routes"][projection]["inputs"][split]["persistence_current_C_baseline"]
                for mask, suffix in (("all_original_observed", "all_observed"), ("all_C_complete", "complete_mask")):
                    metrics(out["CURRENT_C_PERSISTENCE"], {"dataset": dataset, "projection": projection, "split": split, "model": "CURRENT_C_PERSISTENCE", "mask": mask}, persistence["scores_" + suffix], unique_dataset_split_cells=6, repeated_projection_contexts=12, independent_statistical_draws_claimed=False, Raw_Bolt_identical_claimed=False, whole_deployment_cost_measured=False)
                    for seed in SEEDS:
                        scored = matched["datasets"][dataset]["routes"][projection]["native"][seed]["scores"][split]["scores_" + suffix]
                        metrics(out["SCORES"], {"dataset": dataset, "projection": projection, "split": split, "model": "MATCHED_NATIVE_s" + seed, "variant": label, "mask": mask}, scored, native_head_frozen_intervention_not_reoptimized=False)
            for seed in SEEDS:
                native = matched["datasets"][dataset]["routes"][projection]["native"][seed]; meta = native["metadata"]
                base = {"dataset": dataset, "projection": projection, "seed": seed, "rho": rho}
                require([entry["update"] for entry in native["checkpoints"]] == [0, 100, 200, 400, 600] and len(native["learning_curve"]) == 600 and meta["updates"] == 600, "FIVE_CP_AND_600_LOGGED_UPDATES")
                chosen = min(native["checkpoints"], key=lambda entry: (entry["tune"]["mse"], entry["update"]))
                require(chosen["update"] == meta["selected_update"] and meta["old_head_weights_reused_or_warm_started"] is meta["decoder_condition_selected_by_native"] is False, "ORIGINAL_EARLIEST_TUNE_CP_AND_FIXED_RHO")
                for point in native["checkpoints"]: out["CHECKPOINTS"].append({**base, "update": point["update"], "selected": point["update"] == chosen["update"], **point["tune"], "state_sha256": point["state_sha256"], "state_record": point["state_artifact"], "prediction_record": point["tune_prediction_artifact"]})
                for point in native["learning_curve"]: out["LEARNING_CURVE"].append({**base, **point, "loss_scope": "minibatch actual-C masked macro MSE, not full FIT loss"})
                out["NATIVE_TRAINING"].append({**base, "metadata": meta, "fresh_train_wall_seconds": meta["train_wall_seconds"], "parameters": meta["parameters"], "selected_update": meta["selected_update"], "zero_init_seed_changes_batch_order_only": True})
                record = matched["condition_selection_locks"][dataset][projection][seed]
                out["NATIVE_LOCKS"].append({**base, "lock_record": record, "lock": read(resolve(record["path"])), "DEV_used_for_CP_selection": False})
        for name, table, routes, repeats in (("selected_cost", "SELECTED_F_ONLY_COST", SELECTED_ROUTES, 6), ("matched_cost", "MATCHED_WHOLE_COST", MATCHED_ROUTES, 7), ("repeat_matched_cost", "REPEAT_MATCHED_WHOLE_COST", MATCHED_ROUTES, 7)):
            source = values[name]["cost"][dataset]["rows"]
            mapping = {(row["route"], row["batch"]): row for row in source}
            require(len(mapping) == len(source) == 2 * len(routes) and set(mapping) == {(route, batch) for route in routes for batch in (1, 8)}, "EXACT_DISTINCT_COST_GRID")
            for row in source:
                raw = mapping[("RAW_BOLT", row["batch"])]
                times = [(item["perf_end_ns"] - item["perf_start_ns"]) / 1e9 for item in row["samples"]]
                raw_times = [(item["perf_end_ns"] - item["perf_start_ns"]) / 1e9 for item in raw["samples"]]
                require(len(times) == len(raw_times) == repeats and all(math.isfinite(value) and value > 0 for value in times + raw_times), "REGISTERED_FINITE_TIMER_COUNTS")
                paired = [time / base for time, base in zip(times, raw_times)]
                require(row["seconds"] == times and row["median_seconds"] == statistics.median(times) and row["paired_RAW_ratios"] == paired and row["paired_RAW_median_ratio"] == statistics.median(paired), "INDEPENDENT_TIMESTAMP_MEDIAN_AND_PAIRED_RATIO")
                out[table].append({"dataset": dataset, "route": row["route"], "batch": row["batch"], "repeats": repeats, "median_seconds": statistics.median(times), "median_seconds_per_window": statistics.median(times) / row["batch"], "ratio_of_RAW_medians": statistics.median(times) / statistics.median(raw_times), "median_paired_RAW_ratio": statistics.median(paired), "seconds": times, "origins": row["origins"], "shared_process_memory": row["memory"], "includes": row["whole_includes"], "excludes": row["whole_excludes"], "selected_metadata": row["selected_metadata"], "reference": row["reference"], "timed_outputs": row["timed_outputs"], "cohort": name})
                for i, sample in enumerate(row["samples"]): out[table + "_SAMPLES"].append({"dataset": dataset, "route": row["route"], "batch": row["batch"], "repeat": sample["repeat"], "seconds": times[i], "paired_RAW_ratio": paired[i], "sample": sample})
        for route in MATCHED_ROUTES[1:]:
            projection = "PCA_PINV" if route.startswith("PCA") else "FIT_TARGET_PINV"
            if "_NATIVE_s" in route:
                seed = route.rsplit("s", 1)[1]; score = matched["datasets"][dataset]["routes"][projection]["native"][seed]["scores"]["tune"]["scores_all_observed"]["normalized"]
            else:
                label = read(resolve(shrink["selection_locks"][dataset][projection]["path"]))["selected_candidate"]
                score = shrink["datasets"][dataset]["splits"]["tune"]["routes"][projection]["positive_rho_candidates"][label]["methods"]["BOLT"]["scores_all_observed"]["normalized"]
            raw = shrink["datasets"][dataset]["splits"]["tune"]["RAW"]["scores_all_observed"]["normalized"]
            for table in ("MATCHED_WHOLE_COST", "REPEAT_MATCHED_WHOLE_COST"):
                timing = next(row for row in out[table] if row["dataset"] == dataset and row["route"] == route and row["batch"] == 8)
                mse, mae, latency = score["mse"] / raw["mse"], score["mae"] / raw["mae"], timing["ratio_of_RAW_medians"]
                out["REGISTERED_ADOPTION"].append({"dataset": dataset, "route": route, "cost_cohort": table, "TUNE_MSE_ratio_to_RAW": mse, "TUNE_MAE_ratio_to_RAW": mae, "B8_ratio_of_RAW_medians": latency, "all_conditions_pass": mse <= 1.01 and mae <= 1.01 and latency <= .9, "no_new_rho_seed_checkpoint_selection_from_cost_or_DEV": True})
    failed = values["failed_matched_cost_v1"]
    out["COST_FAILURE_PROVENANCE"].append({"stage": "matched_whole_cost_v1", "status": failed["status"], "completed_cells": failed["completed_cost_cells"], "recorded_samples": failed["recorded_timer_samples"], "failed_paths": failed["failed_paths"], "source_snapshots": failed["source_snapshots"], "resource_terminal": failed["resource_terminal"], "performance_failure_claimed": False, "same_encoder_decoder_rho_checkpoints_and_tolerances_preserved": True, "repair_scope": "fresh v2 canonical physical-bridge array metadata binding; prior v1 remains immutable"})
    require({name: len(rows) for name, rows in out.items()} == COUNTS, "EXACT_22_TABLE_COUNTS")
    return out


def row_key(row, table):
    keys = ("dataset", "projection")
    if table in ("SCORES", "RHO_CANDIDATES"):
        keys += ("split", "model", "variant" if table == "SCORES" else "candidate", "mask", "metric_scale")
    elif table == "TUNE_SELECTION": keys += ("candidate",)
    elif table in ("SELECTED_D_BACKENDS", "OUTPUT_GEOMETRY"): keys += ("split", "backend")
    elif table == "SELECTED_D_WINDOWS": keys += ("split", "backend", "window_index_zero_based")
    elif table == "SELECTED_PAST_RECONSTRUCTION": keys += ("split", "backend", "metric_scale")
    elif table == "CURRENT_C_PERSISTENCE": keys += ("split", "mask", "metric_scale")
    elif table in ("CHECKPOINTS", "LEARNING_CURVE"): keys += ("seed", "update")
    elif table in ("NATIVE_TRAINING", "NATIVE_LOCKS"): keys += ("seed",)
    elif table == "OUTPUT_DIAGNOSTIC": keys += ("split", "backend", "model", "metric_scale")
    elif table in ("SELECTED_F_ONLY_COST", "MATCHED_WHOLE_COST", "REPEAT_MATCHED_WHOLE_COST"): keys = ("dataset", "route", "batch")
    elif table.endswith("_COST_SAMPLES"): keys = ("dataset", "route", "batch", "repeat")
    elif table == "REGISTERED_ADOPTION": keys = ("dataset", "route", "cost_cohort")
    elif table == "COST_FAILURE_PROVENANCE": keys = ("stage",)
    return tuple(str(row[key]) for key in keys)


def check_csv(path, name, rows):
    csv.field_size_limit(max(csv.field_size_limit(), Path(path).stat().st_size))
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle); actual, fields = list(reader), reader.fieldnames
    expected = {row_key(row, name): row for row in rows}
    require(len(expected) == len(rows) == len(actual) == COUNTS[name], "CSV_ROW_COUNT_OR_DUPLICATES:" + name)
    require(fields is not None and len(fields) == len(set(fields)) and set(fields) == {key for row in rows for key in row}, "CSV_FIELD_SET:" + name)
    seen, checked = set(), 0
    for row in actual:
        key = row_key(row, name)
        require(key in expected and key not in seen, "UNKNOWN_OR_DUPLICATE_CSV_ROW:" + name); seen.add(key)
        for field in fields:
            source, text = expected[key].get(field), row[field]
            if isinstance(source, (dict, list)): value = json.loads(text)
            elif source is None: value = None if text == "" else text
            elif isinstance(source, bool): value = True if text == "True" else False if text == "False" else text
            elif isinstance(source, int): value = int(text)
            elif isinstance(source, float): value = float(text)
            else: value = text
            checked += leaf_check(value, source, name + ":" + repr(key) + ":" + field)
    return checked


def unique(rows, **where):
    found = [row for row in rows if all(row[key] == value for key, value in where.items())]
    require(len(found) == 1, "UNIQUE_MD_VALUE:" + repr(where))
    return found[0]


def check_MD_numeric(text, tables):
    checked = 0
    for dataset, (_, _, dev) in DATASETS.items():
        for projection in PROJECTIONS:
            label = unique(tables["RHO_LOCKS"], dataset=dataset, projection=projection)["selected_candidate"]
            for split in ("tune", dev):
                def mse(model, variant=label, space=projection):
                    return unique(tables["SCORES"], dataset=dataset, projection=space, split=split, model=model, variant=variant, mask="all_original_observed", metric_scale="normalized")["mse"]
                paragraph = f"[확인] {dataset}/{projection}/{split.upper()} 정규화 MSE: Raw {mse('RAW_BOLT','RAW','NA'):.9f}; static Bolt {mse('BOLT','STATIC'):.9f}; unregularized Bolt {mse('BOLT','LOCAL'):.9f}; selected Bolt {mse('BOLT'):.9f}; old frozen native01/02 {mse('NATIVE_s92601'):.9f}/{mse('NATIVE_s92602'):.9f}; freshly matched native01/02 {mse('MATCHED_NATIVE_s92601'):.9f}/{mse('MATCHED_NATIVE_s92602'):.9f}."
                require(paragraph in text, "MD_ACTUAL_SCORE_PARAGRAPH_CHANGED"); checked += 1
                persistence = unique(tables["CURRENT_C_PERSISTENCE"], dataset=dataset, projection=projection, split=split, mask="all_original_observed", metric_scale="normalized")["mse"]
                line = f"[확인] 같은 mask의 current-C persistence MSE는 {persistence:.9f}, fresh native01/02의 persistence 대비 비율은 {mse('MATCHED_NATIVE_s92601')/persistence:.6f}/{mse('MATCHED_NATIVE_s92602')/persistence:.6f}다. Raw와 persistence는 서로 다른 기준이다."
                require(line in text, "MD_PERSISTENCE_PARAGRAPH_CHANGED"); checked += 1
            def value(backend, model):
                return unique(tables["OUTPUT_DIAGNOSTIC"], dataset=dataset, projection=projection, split=dev, backend=backend, model=model, metric_scale="normalized")["mse"]
            paragraph = f"[확인] {dataset}/{projection}/complete-C {dev.upper()}: D/F real minimum {value('BOLT_NUMPY','SELECTED_D_MINIMUM'):.9f}, E(Y)/F {value('BOLT_NUMPY','E_Y'):.9f}, actual Bolt {value('BOLT_NUMPY','BOLT'):.9f}; D/native real minimum {value('NATIVE_TORCH32','SELECTED_D_MINIMUM'):.9f}, fresh native01/02 {value('NATIVE_TORCH32','MATCHED_NATIVE_s92601'):.9f}/{value('NATIVE_TORCH32','MATCHED_NATIVE_s92602'):.9f}."
            require(paragraph in text, "MD_COMPLETE_MASK_OUTPUT_DIAGNOSTIC_CHANGED"); checked += 1
        old, first, repeat = [unique(tables[table], dataset=dataset, route="RAW_BOLT", batch=8) for table in ("SELECTED_F_ONLY_COST", "MATCHED_WHOLE_COST", "REPEAT_MATCHED_WHOLE_COST")]
        paragraph = f"[확인] {dataset}/Raw B8 절대 ms/window: 앞선 F-only {old['median_seconds_per_window']*1000:.9f}; 첫 matched {first['median_seconds_per_window']*1000:.9f}; 별도 반복 {repeat['median_seconds_per_window']*1000:.9f}. 시행/프로세스 사이 변동 원인은 미확정이며 이를 제거한 speedup을 증명하지 않는다."
        require(paragraph in text, "MD_THREE_SEPARATE_RAW_COHORTS_CHANGED"); checked += 1
    for table in ("SELECTED_F_ONLY_COST", "MATCHED_WHOLE_COST", "REPEAT_MATCHED_WHOLE_COST"):
        require(table in text, "MD_COST_COHORT_NAME_REQUIRED")
        for row in tables[table]:
            if row["batch"] != 8: continue
            require(f"{row['dataset']}/{row['route']} {row['median_seconds_per_window']*1000:.6f} {row['ratio_of_RAW_medians']:.6f} {row['median_paired_RAW_ratio']:.6f}" in text, "MD_COST_ROW_ARITHMETIC_CHANGED"); checked += 1
    for table in ("MATCHED_WHOLE_COST", "REPEAT_MATCHED_WHOLE_COST"):
        rows = [row for row in tables["REGISTERED_ADOPTION"] if row["cost_cohort"] == table]
        paragraph = f"{table}: 동일 Raw TUNE MSE·MAE≤1.01배와 해당 cohort B8 ratio of medians≤.90을 함께 요구한 검산은 {sum(row['all_conditions_pass'] for row in rows)}/{len(rows)}개다."
        require(paragraph in text, "MD_SEPARATE_PRIMARY_REPEAT_ADOPTION_CHANGED"); checked += 1
    for phrase in ("독립 holdout 일반화 검증이 아니다", "빠른 cohort만 고르거나 첫 성공 결과를 반복으로 대체하지 않는다", "MAE·물리 단위 점수는 최소해의 값일 뿐", "Raw에 공간 부등식을 적용하지 않는다", "source/file/state metadata hashes generated outside timed callback", "0셀·0측정", "상대 literal JSON 경로", "독점 메모리 감소를 의미하지 않는다"):
        require(phrase in text, "MD_REQUIRED_SCOPE_PHRASE:" + phrase)
    return checked


def verify(folder, *, fixed_sha=None, generator_sha=GENERATOR_SHA):
    folder = Path(folder).resolve(); manifest_path = folder / "MANIFEST.json"
    manifest = read(manifest_path)
    require(manifest["status"] == "COMPLETE" and manifest["metadata_only"] is True, "COMPLETED_METADATA_ONLY_REPORT")
    for flag in ("NPZ_or_model_opened", "F_head_fit_or_timer_executed_by_report_generator", "protected_TEST_accessed", "same_decoder_side_information_claimed", "strict_K_only_compression", "independent_generalization_claimed", "Bolt_necessity_proven", "new_rho_seed_checkpoint_or_method_selection", "MAE_or_physical_MSE_minimum_claimed", "strict_FP32_universal_output_lower_bound_claimed", "RAW_bound_applied", "automatic_commit_push", "favorable_cost_cohort_selected"):
        require(manifest[flag] is False, "REPORT_SCOPE_CHANGED:" + flag)
    for key, value in {"selected_F_only_cost_cells": 20, "selected_F_only_cost_samples": 120, "new_matched_whole_cost_cells": 28, "new_matched_whole_cost_samples": 196, "repeat_matched_whole_cost_cells": 28, "repeat_matched_whole_cost_samples": 196}.items(): require(manifest[key] == value, "THREE_DISTINCT_COST_COUNTS:" + key)
    generator_path = verify_record(manifest["source"], ".py")
    require(generator_path.name == "make_shrinkage_matched_native_report_v1.py" and manifest["source"]["sha256"] == generator_sha, "EXACT_APPROVED_FROZEN_GENERATOR")
    source = source_contract(generator_path.read_text(encoding="utf-8-sig"))
    for record in manifest["source_bindings"].values():
        require(Path(record["path"]).suffix.lower() in (".py", ".json", ".md"), "SOURCE_BINDINGS_JSON_MD_PY_ONLY")
        verify_record(record)
    require(len(manifest["prior_reports"]) > 0, "EXISTING_PRIOR_REPORT_LINKS_REQUIRED")
    for record in manifest["prior_reports"]: verify_record(record, ".md")
    values = terminal(manifest, fixed_sha)
    for name in ("failed_cost_identity", "physical_bridge_repair_addendum"):
        bind(manifest["inputs"][name], values["matched_cost"]["input_schema_failure_identity" if name == "failed_cost_identity" else "repair_addendum"], name)
    tables = expected_tables(values)
    require(manifest["CSV_rows"] == COUNTS, "EXACT_22_CSV_COUNTS")
    assets = {"REPORT_KO.md"} | {"tables/" + name + ".csv" for name in COUNTS} | {"figures/" + name + "." + suffix for name in FIGURES for suffix in ("png", "svg")}
    declared = set()
    for record in manifest["files"]:
        path = (folder / record["path"]).resolve()
        require(path.is_relative_to(folder) and record["path"] not in declared and path.suffix.lower() in (".csv", ".md", ".png", ".svg"), "ARTIFACT_ESCAPE_OR_DUPLICATE")
        declared.add(record["path"])
        require(path.is_file() and sha(path) == record["sha256"] and path.stat().st_size == record["bytes"] and record["bytes"] > 0, "ARTIFACT_SHA_OR_BYTES_CHANGED:" + record["path"])
    actual = {path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file() and path.name != "MANIFEST.json"}
    require(actual == declared == assets, "EXACT_22_TABLES_12_IMAGES_ONE_MD_NO_UNDECLARED_FILES")
    fields = {name: check_csv(folder / "tables" / (name + ".csv"), name, rows) for name, rows in tables.items()}
    report_path = folder / "REPORT_KO.md"
    text = report_path.read_text(encoding="utf-8-sig"); numeric_lines = check_MD_numeric(text, tables)
    found, embedded = links(report_path)
    linked = {Path(row["path"]).resolve() for row in found}
    require(all(resolve(item["path"]) in linked for item in (*manifest["source_bindings"].values(), *manifest["prior_reports"])), "ALL_BOUND_STAGE_PROVENANCE_PRIOR_AND_SOURCE_LINKS")
    require(all(folder / ("tables/" + name + ".csv") in linked for name in COUNTS), "ALL_22_CSV_RELATIVE_LINKS")
    pictures = {"figures/" + name + "." + suffix for name in FIGURES for suffix in ("png", "svg")}
    require(set(manifest["figures"]) == pictures and len(manifest["figures"]) == 12 and set(embedded) == {folder / item for item in pictures if item.endswith(".png")} and len(embedded) == 6, "EXACT_SIX_EMBEDDED_PNG_AND_SVG_PAIRS")
    from PIL import Image
    parsed = []
    for name in sorted(pictures):
        path = folder / name
        if path.suffix == ".png":
            with Image.open(path) as picture: require(picture.format == "PNG" and min(picture.size) > 0, "VALID_PNG_DIMENSIONS"); picture.verify()
            with Image.open(path) as picture: picture.load(); dimensions = list(picture.size)
            parsed.append({"path": name, "format": "PNG", "dimensions": dimensions})
        else:
            require(ET.parse(path).getroot().tag == "{http://www.w3.org/2000/svg}svg", "PARSEABLE_SVG_REQUIRED")
            parsed.append({"path": name, "format": "SVG"})
    return {"status": "PASS", "report": {"path": str(report_path), "sha256": sha(report_path)}, "manifest": {"path": str(manifest_path), "sha256": sha(manifest_path)}, "audit_source": {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)}, "inputs": manifest["inputs"], "source_contract": source, "files_checked": len(declared), "CSV_rows": COUNTS, "CSV_primitive_fields_checked": fields, "total_CSV_primitive_fields_checked": sum(fields.values()), "MD_numeric_paragraphs_and_cost_rows_checked": numeric_lines, "local_links_checked": len(found), "local_links": found, "PNG_count": 6, "SVG_count": 6, "parsed_figures": parsed, "all_three_cost_cohorts_and_two_adoption_rows_per_method_preserved": True, "timer_ratios_summaries_adoption_recomputed": True, "report_generator_imported": False, "NPZ_or_model_opened": False, "training_or_F_forward": False, "protected_TEST_accessed": False, "scope": "Independent exact bound JSON to typed CSV leaf comparison, MD reported arithmetic, relative file links and decoded image containers. Scientific arrays/models and actual F/timer replay are excluded; visual layout/HTTP review is separate."}


def execute(folder, output):
    output = Path(output).resolve(); require(not output.exists(), "FRESH_REPORT_AUDIT_OUTPUT_REQUIRED")
    receipt = verify(folder)
    with output.open("x", encoding="utf-8") as handle: json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")
    print(json.dumps({name: receipt[name] for name in ("status", "files_checked", "total_CSV_primitive_fields_checked", "local_links_checked", "PNG_count", "SVG_count")}))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--report", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); raise SystemExit(execute(args.report, args.output))
