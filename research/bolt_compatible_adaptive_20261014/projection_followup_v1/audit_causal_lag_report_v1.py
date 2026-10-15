"""Independent JSON-to-report verification; scientific arrays and models stay opaque."""
from pathlib import Path
import argparse
import ast
import csv
import json
import statistics
import xml.etree.ElementTree as ET

from verify_fit_target_report_v1 import require, sha, read, resolve, verify_record, leaf_check, links
from audit_shrinkage_matched_native_report_v1 import bind, require_counts, unique
from audit_causal_lag_cost_v1 import SCIENCE_COUNTS, LAG_SHA, LAG_AUDIT_SHA

HERE = Path(__file__).resolve().parent
DATASETS = {"robin": (17, 4, "val"), "beijing_aotizhongxin": (11, 2, "eval")}
SPACES = ("PCA", "FIT_TARGET", "RAW_SUBSET")
CONDITIONS = ("L0_RECON", "L7_RECON", "L0_FORECAST", "L7_FORECAST")
ROUTES = ("RAW_BOLT",) + tuple(space + "_" + condition + "_BOLT" for space in SPACES for condition in CONDITIONS)
INPUTS = ("science", "audit", "cost", "cost_audit")
SCALES = ("normalized", "physical_per_channel_diagnostic")
FIGURES = ("01_actual_accuracy", "02_factorial_contrasts", "03_joint_future_diagnosis", "04_rank_and_discarded_components", "05_whole_pipeline_cost", "06_accuracy_cost_gate")
TABLES = ("COMPARISON", "DELTA_COMPARISONS", "FACTORIAL_CONTRASTS", "DECODER_STATES", "DECODER_CHANNELS", "DECODER_SINGULARVALUES", "FIT_LOCKS", "F_CACHE_LINEAGE", "JOINT_GEOMETRY", "JOINT_MASK_CERTIFICATES", "FUTURE_DIAGNOSTIC_SCORES", "RETAINED_DISCARDED", "FORECAST_LABEL_PROBES", "WHOLE_COST", "WHOLE_COST_SAMPLES", "REGISTERED_ADOPTION", "EXECUTION_RESOURCES", "PRESERVED_FAILURES")
FIXED_SHA = {"science": LAG_SHA, "audit": LAG_AUDIT_SHA, "cost": "8bdf4cdf05e2be8dfef56406d9b8f7cee46a98571b864707afe83d77431966f4", "cost_audit": "d770b9561263bb99507e1cfe323b3c5785b6806fb3bd80a8c66109c1f5df41a6"}
AUDITOR_SOURCES = {"audit": {"path": str(HERE / "audit_causal_lag_objective_v2.py"), "sha256": "022e9506fe0dbaa593f069e48972cb878a6a2f5cb2f0d062feb0cdde5cbeae30"}, "cost_audit": {"path": str(HERE / "audit_causal_lag_cost_v1.py"), "sha256": "4f0fa97ef682df37d3717812170d8bf8811bd5bb20b6f1f129da482fa74f9f9b"}}
FAILURE_SHA = "7d4999aed252e343dba18c2f009bb3dc20498af5ab2b90c6ac01c0d8c6c8df7a"
REPAIR_SHA = {"joint_zero_column_rank_diagnostic": "9280c189a503b13ccef65c0a259c52958e280f2203458c829f07f9bd6c6b50c6", "joint_auditor_repair_addendum": "89474245c07e7e1d07a24079862ce71dc605cbcf59b76b0f000f4eb800a33286"}
COST_COUNTS = {"cost_cells_checked": 52, "recorded_timer_samples_checked": 676, "physical_backend_prechecks_checked": 6, "physical_first8_windows_checked": 48, "fixed_D_first8_prediction_checks": 24, "fit_lock_files_checked": 24, "encoder_state_hashes_checked": 6, "fixed_decoder_state_hashes_checked": 24, "scientific_prediction_file_hashes_checked": 72, "cached_Bolt_file_hashes_checked": 18, "canonical_physical_bridges_checked": 2}
GENERATOR_SHA = "be3ac864a53beb8e1149c7500bf1d14532373e643866a75fbf95492b0973edf2"
MANIFEST_SHA = "e4e04e2a86a58f02da2fed28cda4eb2e59c0f1973db98d1f57c470645e0965ea"
REPORT_SHA = "d8c7b1e09c2b1677bc40dea8bccc5ec081756ca9cbf088fccb9d8ec52fcf3928"
SHORT = {"PCA": "PCA", "FIT_TARGET": "Target", "RAW_SUBSET": "Subset"}
DISPLAY = {"robin": "Robin", "beijing_aotizhongxin": "Beijing Aotizhongxin"}


def source_contract(text):
    tree = ast.parse(text); functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    generate = ast.unparse(functions["generate"])
    require("not output.exists()" in generate and "output.mkdir(parents=True, exist_ok=False)" in generate and "FRESH_LOCAL_REPORT_DIRECTORY_REQUIRED" in generate, "REPORT_OVERWRITE_REFUSAL")
    for name in ("write_csv", "generate"):
        require(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "open" and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "x" for node in ast.walk(functions[name])), "EXCLUSIVE_CREATE:" + name)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [alias.name.split(".")[0] for alias in node.names] if isinstance(node, ast.Import) else [str(node.module).split(".")[0]]
            require(not set(modules) & {"torch", "chronos"}, "NO_MODEL_IMPORT_IN_REPORT")
        if isinstance(node, ast.Attribute): require(node.attr not in {"load_state_dict", "savez", "savez_compressed", "fit", "predict"}, "NO_SCIENTIFIC_ARRAY_MODEL_OR_FIT_CALL")
    require("path.suffix.lower() == '.json'" in ast.unparse(functions["read"]), "JSON_ONLY_INPUT_READER")
    return {"overwrite_refusal": True, "exclusive_csv_manifest_creation": True, "generator_imported_by_verifier": False, "scientific_NPZ_model_or_training_read": False}


def terminal(manifest, registration):
    bindings = manifest["inputs"]; values = {}
    for name in INPUTS:
        require(bindings[name]["sha256"] == registration["inputs"][name], "EXACT_REGISTERED_INPUT_SHA:" + name)
        values[name] = read(verify_record(bindings[name], ".json"))
        require(values[name]["status"] == ("PASS" if "audit" in name else "COMPLETE") and values[name]["protected_TEST_accessed"] is False, "COMPLETE_PASS_AND_NO_TEST:" + name)
    science, audit, cost, checked = (values[name] for name in INPUTS)
    bind(audit["diagnostic"], bindings["science"], "science PASS"); bind(checked["cost"], bindings["cost"], "whole cost PASS")
    for name in ("audit", "cost_audit"):
        bind(values[name]["audit_source"], registration["auditors"][name], "canonical independent auditor")
        bind(bindings[name + "_source"], values[name]["audit_source"], "manifest audit source")
        verify_record(values[name]["audit_source"], ".py")
    require_counts(audit, SCIENCE_COUNTS, "complete science"); require_counts(checked, COST_COUNTS, "whole cost")
    for stage in (cost, checked):
        bind(stage["causal_lag_diagnostic"], bindings["science"], "same frozen science")
        bind(stage["causal_lag_audit"], bindings["audit"], "same full science PASS")
        for key in ("fit_registry", "raw_subset_diagnostic", "raw_subset_audit", "parent_report", "parent_report_manifest", "parent_report_audit"): bind(stage[key], science[key], "unchanged metadata parent " + key)
    require(science["decoder_fits_completed"] == science["resolved_fit_lock_files"] == 24 and all(science[key] == 72 for key in ("cached_D_predictions_completed", "joint_output_diagnostics_completed", "EY_diagnostics_completed", "flat_diagnostics_completed")) and science["actual_F_cache_splits_reused"] == 18 and science["all_24_locks_before_TUNE_DEV_NPZ"] is True and science["failed_paths"] == [], "FULL_24_D_72_PREDICTIONS_AND_LOCK_BARRIER")
    require(science["new_F_calls"] == science["new_native_head_training"] == 0, "CACHED_CPU_SCIENCE_NOT_NEW_F_OR_NATIVE")
    for key in ("new_backbone_training", "TUNE_DEV_used_for_fit_or_selection", "oracle_used_for_fit_or_selection", "independent_holdout", "strict_K_only_compression", "new_representation_learning_claimed", "whole_deployment_cost_measured"): require(science[key] is False, "RESEARCH_SCOPE:" + key)
    require(tuple(cost["routes"]) == ROUTES and cost["completed_cost_cells"] == 52 and cost["recorded_timer_samples"] == cost["completed_cell_timer_samples"] == 676 and not cost["failed_paths"] and cost["F_before"] == cost["F_after"] and cost["fixed_states_before"] == cost["fixed_states_after"], "COMPLETE_NEW_ACTUAL_COST_COHORT")
    require(cost["new_training_updates"] == cost["new_native_head_training"] == 0 and cost["fixed_FIT_decoder_loaded_before_timer"] is cost["all_source_file_state_metadata_hashes_outside_timer"] is True, "NO_NEW_FIT_IN_COST")
    for key in ("new_backbone_training", "native_heads_loaded_or_timed", "per_origin_decoder_fit", "cached_future_returned_instead_of_actual_F_inside_timer", "new_D_or_lag_objective_selection"): require(cost[key] is False, "ACTUAL_COST_SCOPE:" + key)
    registry = read(verify_record(science["fit_registry"], ".json")); seen = set()
    require(registry["status"] == "ALL_24_RESOLVED_BEFORE_TUNE_DEV" and len(registry["conditions"]) == 24 and registry["TUNE_DEV_NPZ_opened_before_seal"] is False and registry["failed_fit_paths"] == 0, "ALL_24_FIT_LOCKS_BEFORE_EVALUATION")
    for row in registry["conditions"]:
        key = row["dataset"], row["space"], row["condition"]
        require(key not in seen, "DUPLICATE_CONDITION_LOCK"); seen.add(key)
        entry = science["datasets"][key[0]]["spaces"][key[1]]["conditions"][key[2]]
        bind(row["fit_lock"], entry["fit_lock"], "registered D lock"); bind(cost["fit_locks"][key[0]][key[1]][key[2]], entry["fit_lock"], "cost D lock")
        lock = read(verify_record(entry["fit_lock"], ".json"))
        require(lock["status"] == entry["status"] == "LOCKED_FIT_ONLY" and lock["TUNE_DEV_used_for_fit_or_selection"] is lock["oracle_used_for_fit_or_selection"] is False, "FIT_ONLY_SUCCESSFUL_D_LOCK")
        require(resolve(lock["state_artifact"]["path"]) == resolve(entry["state_artifact"]["path"]) and lock["state_artifact"]["sha256"] == entry["state_artifact"]["sha256"], "SAME_OPAQUE_FIXED_D_METADATA_WITHOUT_ARRAY_FILE_READ")
    require(seen == {(d, s, c) for d in DATASETS for s in SPACES for c in CONDITIONS}, "EXACT_24_LOCK_KEYS")
    failure_record = bindings["science_audit_v1"]; require(failure_record["sha256"] == registration["failure"], "ORIGINAL_FAILED_AUDIT_SHA")
    failure = read(verify_record(failure_record, ".json")); require(failure["status"] != "PASS", "FAILURE_NOT_REPLACED_WITH_PASS")
    values["preserved_failures"] = [{"name": "science_audit_v1", "binding": failure_record, "original_status": failure["status"], "totals": failure.get("totals"), "failed_count": failure.get("failed_count"), "failures": failure.get("audit_failed_paths", failure.get("failures", failure.get("failed_paths")))}]
    for key, expected in registration["repair"].items(): require(bindings[key]["sha256"] == expected, "EXACT_REPAIR_EVIDENCE_SHA:" + key); verify_record(bindings[key])
    d2 = read(verify_record(bindings["D2_prior_audit"], ".json"))
    require(d2["status"] == "PASS_WITHOUT_INDEPENDENT_F_FORWARD_REPLAY" and d2["checks"] == 28365 and d2["failed_count"] == 0, "PREVIOUS_PREDICTIVE_D_NOT_ERASED")
    read(verify_record(bindings["joint_Y_prior_audit"], ".json"))
    verify_record(bindings["D2_prior_source"], ".py"); verify_record(bindings["joint_Y_prior_source"], ".py")
    return values


def independent_tables(values):
    tables = {name: [] for name in TABLES}; science, cost = values["science"], values["cost"]
    def put(name, row): tables[name].append(row)
    def metric_rows(name, identity, score, role):
        for scale in SCALES: put(name, dict(identity, scale=scale, role=role, **score[scale]))
    for dataset, (channels, k, dev) in DATASETS.items():
        spaces = science["datasets"][dataset]["spaces"]
        for split in ("fit", "tune", dev):
            reference = spaces["PCA"]["conditions"]["L0_RECON"]["splits"][split]
            for name, source in (("RAW_BOLT", "RAW_BOLT"), ("CURRENT_C_PERSISTENCE", "current_C_persistence")):
                for mask, field in (("all_observed", "scores_all_observed"), ("complete_C", "scores_complete_mask")):
                    metric_rows("COMPARISON", {"dataset": dataset, "space": "NA", "condition": "NA", "split": split, "method": name, "mask": mask}, reference[source][field], "same original raw baseline" if name == "RAW_BOLT" else "baseline; persistence whole cost not measured")
        for space in SPACES:
            owner = spaces[space]
            for condition in CONDITIONS:
                entry = owner["conditions"][condition]; base = {"dataset": dataset, "space": space, "condition": condition}; meta = entry["fit_metadata"]
                put("DECODER_STATES", {**base, "C": channels, "K": k, "lags": entry["lags"], "objective": entry["objective"], "slope_parameters_per_C": meta["slope_parameters_per_C"], "total_slope_parameters": channels * meta["slope_parameters_per_C"], "state_artifact": entry["state_artifact"], "fit_lock": entry["fit_lock"], "fit_wall_seconds": entry["fit_wall_seconds"], "fit_metadata": meta, "coefficient_values_read_by_report": False})
                put("FIT_LOCKS", {**base, **{field: entry[field] for field in ("status", "fit_lock", "FIT_input_source", "FIT_Bolt_cache", "FIT_future_targets_used", "TUNE_DEV_used_for_fit_or_selection", "oracle_used_for_fit_or_selection")}})
                for channel in range(channels):
                    channel_fields = ("rank", "retained_condition", "direct_residual_MSE", "fallback_reason", "fit_executed", "rank_deficient_slope", "full_slope_identified", "constant_observed_response", "all_predictor_entries_zero", "zero_centered_predictor_variance", "true_observed_counts", "no_observed_response", "one_observed_response")
                    put("DECODER_CHANNELS", {**base, "channel_zero_based": channel, **{field: meta[field][channel] for field in channel_fields}, **{field: meta[field] for field in ("singular_relative_cutoff", "centered_design", "intercept_fitted", "slope_parameters_per_C")}})
                    spectrum = meta["singularvalues"][channel]
                    for index, singular in enumerate(spectrum): put("DECODER_SINGULARVALUES", {**base, "channel_zero_based": channel, "singular_index_zero_based": index, "singularvalue": singular, "relative_cutoff": meta["singular_relative_cutoff"], "retained": singular > spectrum[0] * meta["singular_relative_cutoff"]})
                for split in ("fit", "tune", dev):
                    case = entry["splits"][split]; identity = dict(base, split=split)
                    require(case["status"] == "COMPLETE" and not case["diagnostic_failures"], "ALL_PRIMARY_AND_DIAGNOSTIC_CASES_COMPLETE")
                    actual = case["scores_all_observed"]["normalized"]; raw = case["RAW_BOLT"]["scores_all_observed"]["normalized"]; l0 = owner["conditions"]["L0_RECON"]["splits"][split]["scores_all_observed"]["normalized"]
                    for mask, field in (("all_observed", "scores_all_observed"), ("complete_C", "scores_complete_mask")): metric_rows("COMPARISON", dict(identity, method="ACTUAL_F", mask=mask), case[field], "deployable fixed cached F and FIT D")
                    put("DELTA_COMPARISONS", {**identity, "MSE": actual["mse"], "MAE": actual["mae"], "RAW_MSE": raw["mse"], "RAW_MAE": raw["mae"], "MSE_minus_RAW": actual["mse"] - raw["mse"], "MAE_minus_RAW": actual["mae"] - raw["mae"], "MSE_ratio_to_RAW": actual["mse"] / raw["mse"], "MAE_ratio_to_RAW": actual["mae"] / raw["mae"], "MSE_minus_L0_RECON": actual["mse"] - l0["mse"], "MAE_minus_L0_RECON": actual["mae"] - l0["mae"], "full_observed_counts": actual["observed_counts"], "complete_target_positions": case["complete_target_positions"]})
                    if condition == "L0_RECON": put("F_CACHE_LINEAGE", {"dataset": dataset, "space": space, "split": split, "input_source": case["input_source"], "cached_Bolt_source": case["cached_Bolt_source"], "E_source": owner["E_source"], "cached_history_array": case["artifact"]["arrays"]["latent_history"], "cached_future_array": case["artifact"]["arrays"]["actual_F_future"], "origins_array": case["artifact"]["arrays"]["origins"], "new_F_calls_in_science": 0})
                    floor = case["diagnostics"]["joint_floor"]; require(floor["status"] == "COMPLETE", "JOINT_DIAGNOSTIC_COMPLETE")
                    put("JOINT_GEOMETRY", {**identity, "minimum_normalized_MSE": floor["minimum_normalized_MSE"], "metadata": floor["metadata"], "all_HK_future_variables_retained": floor["all_HK_future_variables_retained"], "complete_C_output_rows_only": floor["complete_C_output_rows_only"], "all_real_latent_forecasts_satisfy_registered_bound_claimed": False, "RAW_bound_inequality_applied": False, "source_artifact": case["artifact"]})
                    for index, certificate in enumerate(floor["metadata"]["certificates"]): put("JOINT_MASK_CERTIFICATES", dict(identity, mask_pattern_index=index, **certificate))
                    metric_rows("FUTURE_DIAGNOSTIC_SCORES", dict(identity, method="JOINT_RETAINED_MINIMUM", mask="complete_C", arithmetic="FP64_numeric_retained_solution"), floor["minimum_solution_scores_complete_mask"], "future-known diagnosis; only normalized MSE is minimized")
                    for method, diagnostic in case["diagnostics"]["methods"].items():
                        roles = {"ACTUAL_F": "deployable latent forecast", "EY": "true-future diagnostic, unavailable at deployment", "FLAT": "known-past flat-latent diagnostic; deployment-feasible control, whole cost not separately timed"}
                        for mask, arithmetic, field in (("all_observed", "FP32", "scores_all_observed"), ("complete_C", "FP32", "scores_complete_mask"), ("complete_C", "real_trace_FP64", "real_trace_scores_complete_mask"), ("complete_C", "retained_output_trace_FP64", "retained_output_real_trace_scores_complete_mask")): metric_rows("FUTURE_DIAGNOSTIC_SCORES", dict(identity, method=method, mask=mask, arithmetic=arithmetic), diagnostic[field], roles[method])
                        flags = ("real_trace_minus_minimum_normalized_MSE", "measured_FP32_rounding_normalized_MSE_allowance", "numerical_MSE_tolerance", "real_trace_bound_inequality_pass", "FP32_bound_with_measured_allowance_pass", "retained_output_trace_bound_inequality_pass", "discarded_numeric_map_output_squared_norm", "discarded_numeric_map_output_max_abs")
                        put("RETAINED_DISCARDED", {**identity, "method": method, **{field: diagnostic[field] for field in flags}, "real_MSE": diagnostic["real_trace_scores_complete_mask"]["normalized"]["mse"], "retained_MSE": diagnostic["retained_output_real_trace_scores_complete_mask"]["normalized"]["mse"], "minimum_MSE": floor["minimum_normalized_MSE"]})
                    put("FORECAST_LABEL_PROBES", dict(identity, **case["cached_future_label_perturbation"]))
            for split in ("fit", "tune", dev):
                comparisons = {"LAG_RECON": ("L0_RECON", "L7_RECON"), "LAG_FORECAST": ("L0_FORECAST", "L7_FORECAST"), "OBJECTIVE_L0": ("L0_RECON", "L0_FORECAST"), "OBJECTIVE_L7": ("L7_RECON", "L7_FORECAST")}
                for contrast, (before, after) in comparisons.items():
                    a, b = [owner["conditions"][c]["splits"][split]["scores_all_observed"]["normalized"] for c in (before, after)]
                    put("FACTORIAL_CONTRASTS", {"dataset": dataset, "space": space, "split": split, "contrast": contrast, "before_condition": before, "after_condition": after, "before_MSE": a["mse"], "after_MSE": b["mse"], "MSE_delta_after_minus_before": b["mse"] - a["mse"], "before_MAE": a["mae"], "after_MAE": b["mae"], "MAE_delta_after_minus_before": b["mae"] - a["mae"], "independent_generalization_claimed": False})
        rows = cost["cost"][dataset]["rows"]
        require(len(rows) == 26 and {(r["route"], r["batch"]) for r in rows} == {(r, b) for r in ROUTES for b in (1, 8)}, "ALL_COST_CONDITIONS_NO_FILTERING")
        for row in rows:
            base = unique(rows, route="RAW_BOLT", batch=row["batch"])
            seconds = [(s["perf_end_ns"] - s["perf_start_ns"]) / 1e9 for s in row["samples"]]
            raw_seconds = {s["repeat"]: (s["perf_end_ns"] - s["perf_start_ns"]) / 1e9 for s in base["samples"]}
            paired = [time / raw_seconds[s["repeat"]] for time, s in zip(seconds, row["samples"])]
            require(row["status"] == "COMPLETE" and len(seconds) == row["repeats"] == 13, "ALL_13_COST_SAMPLES")
            leaf_check(row["seconds"], seconds, "independent ns times"); leaf_check(row["paired_RAW_ratios"], paired, "independent paired ratios")
            leaf_check(row["median_seconds"], statistics.median(seconds), "independent median"); leaf_check(row["paired_RAW_median_ratio"], statistics.median(paired), "independent paired median")
            put("WHOLE_COST", {"dataset": dataset, **row, "ratio_of_RAW_medians": statistics.median(seconds) / statistics.median(raw_seconds.values()), "median_paired_RAW_ratio": statistics.median(paired), "memory_is_shared_not_exclusive": True})
            for index, sample in enumerate(row["samples"]): put("WHOLE_COST_SAMPLES", {"dataset": dataset, "route": row["route"], "batch": row["batch"], "repeat": sample["repeat"], "position": sample["position"], "seconds": sample["seconds"], "paired_RAW_ratio": paired[index], "sample": sample})
        for space in SPACES:
            for condition in CONDITIONS:
                quality = unique(tables["DELTA_COMPARISONS"], dataset=dataset, space=space, condition=condition, split="tune")
                speed = unique(tables["WHOLE_COST"], dataset=dataset, route=space + "_" + condition + "_BOLT", batch=8)
                mse, mae, latency = quality["MSE_ratio_to_RAW"] <= 1.01, quality["MAE_ratio_to_RAW"] <= 1.01, speed["ratio_of_RAW_medians"] <= .9
                put("REGISTERED_ADOPTION", {"dataset": dataset, "space": space, "condition": condition, "TUNE_MSE_ratio_to_RAW": quality["MSE_ratio_to_RAW"], "TUNE_MAE_ratio_to_RAW": quality["MAE_ratio_to_RAW"], "B8_ratio_of_RAW_medians": speed["ratio_of_RAW_medians"], "B8_median_paired_RAW_ratio": speed["median_paired_RAW_ratio"], "TUNE_MSE_pass": mse, "TUNE_MAE_pass": mae, "B8_cost_pass": latency, "all_registered_conditions_pass": mse and mae and latency, "DEV_used_for_selection": False, "new_method_selection_performed_by_report": False})
    for name in INPUTS:
        value = values[name]
        put("EXECUTION_RESOURCES", {"stage": name, "status": value["status"], "resource_terminal": value.get("resource_terminal"), "wall_seconds": value.get("wall_seconds"), "totals": value.get("totals"), "new_F_calls": value.get("new_F_calls"), "decoder_fits_completed": value.get("decoder_fits_completed"), "fit_wall_seconds_sum": sum(r["fit_wall_seconds"] for r in tables["DECODER_STATES"]) if name == "science" else None})
    tables["PRESERVED_FAILURES"] = values["preserved_failures"]
    expected = {"COMPARISON": 336, "DELTA_COMPARISONS": 72, "FACTORIAL_CONTRASTS": 72, "DECODER_STATES": 24, "DECODER_CHANNELS": 336, "FIT_LOCKS": 24, "F_CACHE_LINEAGE": 18, "JOINT_GEOMETRY": 72, "FUTURE_DIAGNOSTIC_SCORES": 1872, "RETAINED_DISCARDED": 216, "FORECAST_LABEL_PROBES": 72, "WHOLE_COST": 52, "WHOLE_COST_SAMPLES": 676, "REGISTERED_ADOPTION": 24, "EXECUTION_RESOURCES": 4, "PRESERVED_FAILURES": 1}
    require(all(len(tables[name]) == number for name, number in expected.items()), "ALL_FIXED_COHORT_COUNTS")
    return tables


def row_key(table, row):
    base = ("dataset", "space", "condition")
    keys = {"COMPARISON": base + ("split", "method", "mask", "scale"), "DELTA_COMPARISONS": base + ("split",), "FACTORIAL_CONTRASTS": ("dataset", "space", "split", "contrast"), "DECODER_STATES": base, "DECODER_CHANNELS": base + ("channel_zero_based",), "DECODER_SINGULARVALUES": base + ("channel_zero_based", "singular_index_zero_based"), "FIT_LOCKS": base, "F_CACHE_LINEAGE": ("dataset", "space", "split"), "JOINT_GEOMETRY": base + ("split",), "JOINT_MASK_CERTIFICATES": base + ("split", "mask_pattern_index"), "FUTURE_DIAGNOSTIC_SCORES": base + ("split", "method", "mask", "arithmetic", "scale"), "RETAINED_DISCARDED": base + ("split", "method"), "FORECAST_LABEL_PROBES": base + ("split",), "WHOLE_COST": ("dataset", "route", "batch"), "WHOLE_COST_SAMPLES": ("dataset", "route", "batch", "repeat"), "REGISTERED_ADOPTION": base, "EXECUTION_RESOURCES": ("stage",), "PRESERVED_FAILURES": ("name",)}[table]
    return tuple(str(row[key]) for key in keys)


def check_csv(path, table, expected):
    csv.field_size_limit(max(csv.field_size_limit(), Path(path).stat().st_size))
    with Path(path).open(encoding="utf-8-sig", newline="") as handle: reader = csv.DictReader(handle); actual, fields = list(reader), reader.fieldnames
    mapping = {row_key(table, row): row for row in expected}
    require(len(actual) == len(mapping) == len(expected), "CSV_DUPLICATE_MISSING_OR_EXTRA_ROWS:" + table)
    require(fields is not None and len(fields) == len(set(fields)) and set(fields) == {key for row in expected for key in row}, "EXACT_CSV_COLUMN_SET:" + table)
    checked, seen = 0, set()
    for row in actual:
        identity = row_key(table, row); require(identity in mapping and identity not in seen, "CSV_UNKNOWN_OR_DUPLICATE_IDENTITY"); seen.add(identity)
        for key in fields:
            expected_value, raw = mapping[identity].get(key), row[key]
            if isinstance(expected_value, (dict, list)): value = json.loads(raw)
            elif expected_value is None: value = None if raw == "" else raw
            elif isinstance(expected_value, bool): value = True if raw == "True" else False if raw == "False" else raw
            elif isinstance(expected_value, int): value = int(raw)
            elif isinstance(expected_value, float): value = float(raw)
            else: value = raw
            checked += leaf_check(value, expected_value, table + ":" + repr(identity) + ":" + key)
    return checked


def check_MD(text, tables):
    checked = 0
    def contains(fragment):
        nonlocal checked
        require(fragment in text, "MD_NUMERIC_OR_SCOPE_MISMATCH:" + fragment); checked += 1
    def point(dataset, space, condition, split): return unique(tables["DELTA_COMPARISONS"], dataset=dataset, space=space, condition=condition, split=split)
    passed = sum(row["all_registered_conditions_pass"] for row in tables["REGISTERED_ADOPTION"])
    better = sum(row["MSE_minus_RAW"] < 0 for row in tables["DELTA_COMPARISONS"] if row["split"] in ("val", "eval"))
    contrasts = tables["FACTORIAL_CONTRASTS"]
    tune = sum(row["MSE_delta_after_minus_before"] < 0 for row in contrasts if row["split"] == "tune" and row["contrast"].startswith("OBJECTIVE"))
    dev = sum(row["MSE_delta_after_minus_before"] < 0 for row in contrasts if row["split"] in ("val", "eval") and row["contrast"].startswith("OBJECTIVE"))
    contains(f"Raw보다 full-observed 정규화 MSE가 작은 조건은 {better}/24이며, 사전 정확도·비용 동시 기준을 통과한 조건은 {passed}/24다.")
    contains(f"MSE를 줄인 비교는 TUNE {tune}/12, 기노출 DEV {dev}/12다.")
    for dataset, (_, _, development) in DATASETS.items():
        for space in SPACES:
            for condition in CONDITIONS:
                a, b = [point(dataset, space, condition, split)["MSE"] for split in ("tune", development)]
                contains(f"{('Robin' if dataset == 'robin' else 'Beijing'):<9} {SHORT[space]:<7} {condition:<13} {a:.9f}   {b:.9f}")
        raw = unique(tables["COMPARISON"], dataset=dataset, space="NA", condition="NA", split=development, method="RAW_BOLT", mask="all_observed", scale="normalized")
        contains(f"{DISPLAY[dataset]} Raw exposed_DEV MSE={raw['mse']:.9f}")
        rows = [row for row in tables["RETAINED_DISCARDED"] if row["dataset"] == dataset]
        contains(f"{DISPLAY[dataset]}: 원 real-trace 부등식 거짓 {sum(not row['real_trace_bound_inequality_pass'] for row in rows)}/{len(rows)}, FP32+allowance 부등식 거짓 {sum(not row['FP32_bound_with_measured_allowance_pass'] for row in rows)}/{len(rows)}.")
        for route in ROUTES:
            timing = unique(tables["WHOLE_COST"], dataset=dataset, route=route, batch=8)
            contains(f"{('Robin' if dataset == 'robin' else 'Beijing'):<9} {route:<31} {timing['median_seconds_per_window']*1000:.6f}   {timing['ratio_of_RAW_medians']:.6f}   {timing['median_paired_RAW_ratio']:.6f}")
    before, after = [point("robin", "PCA", condition, "val")["MSE"] for condition in ("L0_RECON", "L7_RECON")]
    f0, f7 = [unique(tables["JOINT_GEOMETRY"], dataset="robin", space="PCA", condition=condition, split="val")["minimum_normalized_MSE"] for condition in ("L0_RECON", "L7_RECON")]
    contains(f"DEV MSE가 {before:.9f}→{after:.9f}, 동일 complete-C mask의 수치 유지 공간 minimum은 {f0:.9f}→{f7:.9f}다.")
    contains(f"통과 {passed}/24이며 DEV 최저 조건을 채택하거나 원수치를 선택적으로 제외하지 않았다.")
    for phrase in ("E 자체, F, native head를 새로 학습하지 않았다", "FORECAST D는 FIT의 실제 미래 Y를 학습 target으로 쓰는 지도학습", "72개의 좁은 perturbation 검사", "독립 일반화 검증으로 확대하지 않는다", "FIT_TARGET E는 이전부터 FIT 미래 감독을 사용했다", "공통 과거32:512의 480시점", "target support·분포·감독 정보가 다르다", "lag7은 slope 수가 7배", "중첩된 행을 독립 표본 수로 세지 않는다", "strict K-only가 아니다", "flat 미래 latent도", "미래 latent 자유변수는 horizon마다 K개인 HK", "관측이 없는 horizon의 잠재 미지수도 남겼다", "SSE/(complete위치수×C)", "불완전 C horizon 전체를 currentC로 채운", "유지된 수치 공간", "유지 공간으로 투영한 trace에만", "MAE·물리 단위 오차는 그 해의 값이며 각각의 최소가 아니다", "이전 pointwise/global-rank-K floor는 재사용하지 않았다", "정확한 zero 열만 LS 계산에서 제외", "원 D·예측·조건·cutoff·공차를 바꾼 개선이 아니다", "실패한 최초 감사와 원 source", "676개의 독립 데이터가 아니다", "RSS/VRAM을 방법별 독점 메모리나 장비 독점으로 해석하지 않는다", "기존 native는 이전 D에 맞춘 모델", "독립 검산28,365/실패0", "이번은 최초 predictive D가 아니라", "ODPC 전체 구현이나 새로운 lag 구조의 발견으로 주장하지 않는다", "기존 joint-Y 4조건도", "E/D를 600 update 학습", "F 입력까지의 nonzero gradient", "논문/PPT 작성이나 Git 게시 완료를 뜻하지 않는다"):
        contains(phrase)
    return checked


def check_files(folder, manifest):
    expected = {"REPORT_KO.md"} | {"tables/" + table + ".csv" for table in TABLES} | {"figures/" + name + "." + extension for name in FIGURES for extension in ("png", "svg")}
    declared = set()
    for item in manifest["files"]:
        path = (folder / item["path"]).resolve(); require(path.is_relative_to(folder) and item["path"] not in declared, "FILE_ESCAPE_OR_DUPLICATE")
        declared.add(item["path"]); require(path.is_file() and path.stat().st_size == item["bytes"] > 0 and sha(path) == item["sha256"], "FILE_SHA_BYTES_OR_EXISTENCE:" + item["path"])
    actual = {path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file() and path.name != "MANIFEST.json"}
    require(expected == actual == declared, "EXACT_31_FILES_NO_OMISSIONS_OR_EXTRAS")
    return len(expected)


def check_figures(folder):
    from PIL import Image
    result = []
    for name in FIGURES:
        png, svg = [folder / ("figures/" + name + "." + suffix) for suffix in ("png", "svg")]
        with Image.open(png) as image: require(image.format == "PNG" and min(image.size) > 0, "VALID_PNG_CONTAINER"); image.verify()
        with Image.open(png) as image: image.load(); size = list(image.size)
        require(ET.parse(svg).getroot().tag == "{http://www.w3.org/2000/svg}svg", "VALID_SVG_XML")
        result.append({"name": name, "PNG_dimensions": size, "SVG_parsed": True})
    return result


def verify(folder, registration=None):
    if registration is None: registration = {"generator": GENERATOR_SHA, "manifest": MANIFEST_SHA, "report": REPORT_SHA, "inputs": FIXED_SHA, "auditors": AUDITOR_SOURCES, "failure": FAILURE_SHA, "repair": REPAIR_SHA}
    require(all(isinstance(registration.get(key), str) and len(registration[key]) == 64 for key in ("generator", "manifest", "report")), "ACTUAL_REPORT_MANIFEST_SOURCE_NOT_YET_REGISTERED")
    folder = Path(folder).resolve(); manifest_path, report_path = folder / "MANIFEST.json", folder / "REPORT_KO.md"
    require(sha(manifest_path) == registration["manifest"] and sha(report_path) == registration["report"], "EXACT_ACTUAL_REPORT_AND_MANIFEST_SHA")
    manifest = read(manifest_path)
    require(manifest["status"] == "COMPLETE" and all(manifest[key] is False for key in ("scientific_NPZ_or_model_read", "new_fit_F_or_training_executed", "protected_TEST_accessed", "independent_generalization_claimed")) and all(manifest[key] is True for key in ("future_known_diagnostics_separate", "all_four_conditions_and_72_predictions_preserved", "all_cost_routes_and_two_ratios_preserved")), "COMPLETE_METADATA_ONLY_REPORT_SCOPE")
    generator = verify_record(manifest["source"], ".py")
    require(generator.name == "make_causal_lag_report_v1.py" and manifest["source"]["sha256"] == registration["generator"], "EXACT_FROZEN_REPORT_GENERATOR")
    source = source_contract(generator.read_text(encoding="utf-8-sig")); bind(manifest["inputs"]["report_source"], manifest["source"], "report source provenance")
    for record in manifest["inputs"].values(): require(Path(record["path"]).suffix.lower() in (".json", ".md", ".py"), "METADATA_BINDINGS_ONLY"); verify_record(record)
    values = terminal(manifest, registration); tables = independent_tables(values); counts = {key: len(rows) for key, rows in tables.items()}
    require(manifest["table_row_counts"] == counts and manifest["figure_png_count"] == manifest["figure_svg_count"] == 6, "ALL_18_TABLE_COUNTS_AND_SIX_FIGURE_PAIRS")
    file_count = check_files(folder, manifest)
    leaves = {table: check_csv(folder / ("tables/" + table + ".csv"), table, rows) for table, rows in tables.items()}
    text = report_path.read_text(encoding="utf-8-sig"); md = check_MD(text, tables)
    found, embedded = links(report_path); linked = {Path(row["path"]).resolve() for row in found}
    require(all(not Path(row["target"]).is_absolute() and ":" not in row["target"] for row in found), "RELATIVE_GITHUB_LOCAL_LINKS")
    require(all(resolve(record["path"]) in linked and record["sha256"] in text for record in manifest["inputs"].values()), "ALL_INPUT_SOURCE_FAILURE_REPAIR_PRIOR_LINKS_AND_SHA")
    require(all(folder / ("tables/" + name + ".csv") in linked for name in TABLES), "ALL_18_CSV_LINKS")
    require(set(embedded) == {folder / ("figures/" + name + ".png") for name in FIGURES} and len(embedded) == 6, "EXACT_SIX_EMBEDDED_PNGS")
    parsed = check_figures(folder)
    return {"status": "PASS", "report": {"path": str(report_path), "sha256": sha(report_path)}, "manifest": {"path": str(manifest_path), "sha256": sha(manifest_path)}, "audit_source": {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)}, "inputs": manifest["inputs"], "source_contract": source, "files_checked": file_count, "CSV_rows": counts, "CSV_primitive_fields_checked": leaves, "total_CSV_primitive_fields_checked": sum(leaves.values()), "MD_numeric_rows_and_scope_fragments_checked": md, "local_links_checked": len(found), "local_links": found, "PNG_count": 6, "SVG_count": 6, "parsed_figures": parsed, "original_failed_science_audit_preserved": True, "exact_zero_column_repair_evidence_preserved": True, "D2_and_joint_Y_prior_preserved": True, "two_timer_ratios_and_adoption_recomputed": True, "report_generator_imported": False, "scientific_NPZ_or_model_opened": False, "training_F_or_actual_timers_replayed": False, "protected_TEST_accessed": False, "scope": "Exact bound JSON to every typed CSV primitive and independent arithmetic; Markdown numbers/scope/relative provenance links and parsed image containers. No scientific array/model access, timer replay, or independent generalization claim."}


def execute(folder, output):
    output = Path(output).resolve(); require(not output.exists(), "FRESH_REPORT_AUDIT_OUTPUT")
    receipt = verify(folder)
    with output.open("x", encoding="utf-8") as handle: json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")
    print(json.dumps({key: receipt[key] for key in ("status", "files_checked", "total_CSV_primitive_fields_checked", "local_links_checked", "PNG_count", "SVG_count")}))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--report", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); raise SystemExit(execute(args.report, args.output))
