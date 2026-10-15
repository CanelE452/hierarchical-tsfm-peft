"""JSON-only report of fixed Bolt, causal-lag D structure and FIT objectives."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import statistics


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATASETS = {"robin": (17, 4, "val"), "beijing_aotizhongxin": (11, 2, "eval")}
SPACES = ("PCA", "FIT_TARGET", "RAW_SUBSET")
CONDITIONS = ("L0_RECON", "L7_RECON", "L0_FORECAST", "L7_FORECAST")
ROUTES = ("RAW_BOLT",) + tuple(space+"_"+condition+"_BOLT" for space in SPACES for condition in CONDITIONS)
INPUTS = ("science", "audit", "cost", "cost_audit")
FIXED_SHA = {"science": "4168cd05962e479130a722919310f4c192c1bab03d9b78cb7983754f2e502c17",
             "audit": "d395a82e76a31e38520db1dd4d56642963ca4e362f358a6d86480049d402a31d",
             "cost": "8bdf4cdf05e2be8dfef56406d9b8f7cee46a98571b864707afe83d77431966f4",
             "cost_audit": "d770b9561263bb99507e1cfe323b3c5785b6806fb3bd80a8c66109c1f5df41a6"}
AUDITOR_SOURCES = {"audit": {"path": str(HERE/"audit_causal_lag_objective_v2.py"),
    "sha256": "022e9506fe0dbaa593f069e48972cb878a6a2f5cb2f0d062feb0cdde5cbeae30"},
    "cost_audit": {"path": str(HERE/"audit_causal_lag_cost_v1.py"), "sha256": "4f0fa97ef682df37d3717812170d8bf8811bd5bb20b6f1f129da482fa74f9f9b"}}
PRESERVED_FAILURES = {"science_audit_v1": {"path": str(HERE/"audits/CAUSAL_LAG_OBJECTIVE_AUDIT_v1.json"),
    "sha256": "7d4999aed252e343dba18c2f009bb3dc20498af5ab2b90c6ac01c0d8c6c8df7a"}}
REPAIR_EVIDENCE = {"joint_zero_column_rank_diagnostic": {"path": str(HERE/"audits/CAUSAL_LAG_RANK_BOUNDARY_DIAGNOSTIC_ROOT_v1.json"),
    "sha256": "9280c189a503b13ccef65c0a259c52958e280f2203458c829f07f9bd6c6b50c6"},
    "joint_auditor_repair_addendum": {"path": str(HERE/"CAUSAL_LAG_JOINT_AUDITOR_ZERO_COLUMN_REPAIR_KO_v2.md"),
    "sha256": "89474245c07e7e1d07a24079862ce71dc605cbcf59b76b0f000f4eb800a33286"}}
SCIENCE_COUNTS = {"decoder_states_checked": 24, "fit_lock_files_checked": 24, "FIT_source_contexts_checked": 6,
    "cached_Bolt_source_contexts_checked": 18, "cached_D_split_prediction_arrays_checked": 72,
    "independent_FP32_method_prediction_arrays_checked": 216, "real_trace_prediction_arrays_checked": 216,
    "joint_output_diagnostic_cells_checked": 72, "EY_prediction_arrays_checked": 72, "flat_prediction_arrays_checked": 72,
    "retained_trace_inequalities_checked": 216, "original_real_trace_inequalities_evaluated": 216,
    "FP32_with_allowance_inequalities_evaluated": 216, "RAW_bound_inequalities_checked": 0,
    "cached_future_label_perturbation_checks": 72, "independent_fit_response_columns_checked": 336,
    "score_field_checks": 14688}
COST_COUNTS = {"cost_cells_checked": 52, "recorded_timer_samples_checked": 676,
    "physical_backend_prechecks_checked": 6, "physical_first8_windows_checked": 48, "fixed_D_first8_prediction_checks": 24,
    "fit_lock_files_checked": 24, "encoder_state_hashes_checked": 6, "fixed_decoder_state_hashes_checked": 24,
    "scientific_prediction_file_hashes_checked": 72, "cached_Bolt_file_hashes_checked": 18, "canonical_physical_bridges_checked": 2}
SCALES = ("normalized", "physical_per_channel_diagnostic")
DISPLAY = {"robin": "Robin", "beijing_aotizhongxin": "Beijing Aotizhongxin"}
SHORT = {"PCA": "PCA", "FIT_TARGET": "Target", "RAW_SUBSET": "Subset"}
CONDITION_SHORT = {"L0_RECON": "R0", "L7_RECON": "R7", "L0_FORECAST": "F0", "L7_FORECAST": "F7"}


def require(value, message):
    if not value: raise ValueError(message)


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha(path):
    with Path(path).open("rb") as handle: return hashlib.file_digest(handle, "sha256").hexdigest()


def record(path):
    return {"path": str(resolve(path)), "sha256": sha(resolve(path))}


def read(path):
    path = resolve(path)
    require(path.suffix.lower() == ".json", "JSON_METADATA_ONLY_INPUT")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def bound(item, expected=None):
    path = resolve(item["path"])
    require(path.suffix.lower() in (".json", ".md", ".py") and sha(path) == item["sha256"], "EXACT_METADATA_FILE_BINDING")
    if expected is not None:
        require(path == resolve(expected["path"]) and item["sha256"] == expected["sha256"], "SAME_EXACT_METADATA_BINDING")
    if "bytes" in item: require(path.stat().st_size == item["bytes"], "OPTIONAL_METADATA_BYTES")
    return path


def same_opaque(item, expected):
    require(resolve(item["path"]) == resolve(expected["path"]) and item["sha256"] == expected["sha256"], "SAME_OPAQUE_SCIENTIFIC_ARTIFACT")
    if "bytes" in item and "bytes" in expected: require(item["bytes"] == expected["bytes"], "SAME_OPAQUE_BYTES")


def load_verified(paths):
    require(set(paths) == set(INPUTS), "FOUR_EXACT_STAGE_INPUTS_REQUIRED")
    require(all(isinstance(FIXED_SHA[key], str) and len(FIXED_SHA[key]) == 64 for key in INPUTS)
        and all(isinstance(value, dict) for value in (*AUDITOR_SOURCES.values(), *PRESERVED_FAILURES.values(), *REPAIR_EVIDENCE.values())),
        "ACTUAL_SCIENCE_COST_PASS_AND_FAILURE_RECORDS_NOT_YET_REGISTERED")
    values, bindings = {key: read(path) for key, path in paths.items()}, {key: record(path) for key, path in paths.items()}
    for key in INPUTS:
        require(bindings[key]["sha256"] == FIXED_SHA[key] and values[key].get("status") == ("PASS" if "audit" in key else "COMPLETE"), "REGISTERED_ACTUAL_COMPLETE_PASS:"+key)
        require(values[key].get("protected_TEST_accessed") is False, "NO_PROTECTED_TEST:"+key)
    science, audit, cost, cost_audit = (values[key] for key in INPUTS)
    bound(audit["diagnostic"], bindings["science"]); bound(cost_audit["cost"], bindings["cost"])
    for key in ("audit", "cost_audit"):
        bound(values[key]["audit_source"], AUDITOR_SOURCES[key]); bindings[key+"_source"] = values[key]["audit_source"]
    for value in (cost, cost_audit):
        bound(value["causal_lag_diagnostic"], bindings["science"])
        bound(value["causal_lag_audit"], bindings["audit"])
        bound(value["fit_registry"], science["fit_registry"])
        for key in ("raw_subset_diagnostic", "raw_subset_audit", "parent_report", "parent_report_manifest", "parent_report_audit"):
            bound(value[key], science[key])
    for value, expected in ((audit, SCIENCE_COUNTS), (cost_audit, COST_COUNTS)):
        require(all(value.get("totals", {}).get(key) == count for key, count in expected.items()), "FULL_INDEPENDENT_AUDIT_COVERAGE")
    require(science["decoder_fits_completed"] == science["resolved_fit_lock_files"] == 24
        and science["cached_D_predictions_completed"] == science["joint_output_diagnostics_completed"] == science["EY_diagnostics_completed"] == science["flat_diagnostics_completed"] == 72
        and science["actual_F_cache_splits_reused"] == 18 and science["all_24_locks_before_TUNE_DEV_NPZ"] is True and not science["failed_paths"], "COMPLETE_24_BY_THREE_SCIENCE")
    require(science["new_F_calls"] == science["new_native_head_training"] == 0
        and all(science[key] is False for key in ("new_backbone_training", "TUNE_DEV_used_for_fit_or_selection", "oracle_used_for_fit_or_selection",
            "independent_holdout", "strict_K_only_compression", "new_representation_learning_claimed", "whole_deployment_cost_measured")), "EXACT_FIT_SUPERVISION_AND_CACHE_ONLY_SCOPE")
    require(tuple(cost["routes"]) == ROUTES and cost["completed_cost_cells"] == 52
        and cost["recorded_timer_samples"] == cost["completed_cell_timer_samples"] == 676 and not cost["failed_paths"]
        and cost["F_before"] == cost["F_after"] and cost["fixed_states_before"] == cost["fixed_states_after"], "FULL_NEW_THIRTEEN_ROUTE_WHOLE_COST")
    require(cost["new_training_updates"] == cost["new_native_head_training"] == 0 and all(cost[key] is False for key in (
        "new_backbone_training", "native_heads_loaded_or_timed", "per_origin_decoder_fit", "cached_future_returned_instead_of_actual_F_inside_timer", "new_D_or_lag_objective_selection"))
        and cost["fixed_FIT_decoder_loaded_before_timer"] is True and cost["all_source_file_state_metadata_hashes_outside_timer"] is True, "ACTUAL_F_TIME_NOT_CACHED_CPU_REPLAY")
    registry = read(bound(science["fit_registry"]))
    require(registry["status"] == "ALL_24_RESOLVED_BEFORE_TUNE_DEV" and registry["TUNE_DEV_NPZ_opened_before_seal"] is False
        and len(registry["conditions"]) == 24 and registry["failed_fit_paths"] == 0, "ALL_24_FIT_LOCK_BARRIER")
    seen = set()
    for item in registry["conditions"]:
        data, space, condition = key = item["dataset"], item["space"], item["condition"]
        require(key not in seen, "UNIQUE_FIT_CONDITION"); seen.add(key)
        entry = science["datasets"][data]["spaces"][space]["conditions"][condition]
        bound(item["fit_lock"], entry["fit_lock"]); bound(cost["fit_locks"][data][space][condition], entry["fit_lock"])
        lock = read(bound(entry["fit_lock"]))
        require(lock["status"] == entry["status"] == "LOCKED_FIT_ONLY" and lock["TUNE_DEV_used_for_fit_or_selection"] is False
            and lock["oracle_used_for_fit_or_selection"] is False, "SUCCESSFUL_FIXED_FIT_ONLY_D")
        same_opaque(lock["state_artifact"], entry["state_artifact"])
    require(seen == {(d, s, c) for d in DATASETS for s in SPACES for c in CONDITIONS}, "EXACT_PREDECLARED_24_KEYS")
    values["preserved_failures"] = []
    for name, item in PRESERVED_FAILURES.items():
        value = read(bound(item)); require(value.get("status") != "PASS", "ORIGINAL_FAILED_AUDIT_MUST_REMAIN_FAILED")
        bindings[name] = item; values["preserved_failures"].append({"name": name, "binding": item, "original_status": value.get("status"),
            "totals": value.get("totals"), "failed_count": value.get("failed_count"),
            "failures": value.get("audit_failed_paths", value.get("failures", value.get("failed_paths")))})
    for name, item in REPAIR_EVIDENCE.items():
        bound(item); bindings[name] = item
    for key in ("plan", "raw_subset_diagnostic", "raw_subset_audit", "parent_report", "parent_report_manifest", "parent_report_audit"):
        bound(science[key]); bindings[key] = science[key]
    bindings["cost_plan"] = cost["cost_plan"]; bound(bindings["cost_plan"])
    d2 = ROOT / "research/smallk_failure_followup_20261014/validation_v1/D2_ARRAY_AUDIT_v1.json"
    values["D2_prior_audit"] = read(d2); bindings["D2_prior_audit"] = record(d2)
    require(values["D2_prior_audit"]["status"] == "PASS_WITHOUT_INDEPENDENT_F_FORWARD_REPLAY"
        and values["D2_prior_audit"]["checks"] == 28365 and values["D2_prior_audit"]["failed_count"] == 0, "PRESERVED_PREVIOUS_PREDICTIVE_DECODER_EXPERIMENT")
    bindings["D2_prior_source"] = record(ROOT / "research/smallk_failure_followup_20261014/frozen_decoder_refit.py")
    bindings["joint_Y_prior_source"] = record(ROOT / "research/smallk_failure_followup_20261014/run_joint_objectives_v2.py")
    bindings["joint_Y_prior_audit"] = record(ROOT / "research/smallk_failure_followup_20261014/validation_v1/JOINT_ARRAY_AUDIT_v1.json")
    return values, bindings


def choose(rows, **where):
    matches = [row for row in rows if all(row.get(key) == value for key, value in where.items())]
    require(len(matches) == 1, "UNIQUE_REPORT_ROW:"+str(where)); return matches[0]


def score_rows(base, score, role):
    require(score.get("normalized") is not None, "OBSERVED_REPORT_SCORE_REQUIRED")
    return [{**base, "scale": scale, "role": role, **score[scale]} for scale in SCALES]


def collect_tables(values):
    science, cost = values["science"], values["cost"]
    names = ("COMPARISON", "DELTA_COMPARISONS", "FACTORIAL_CONTRASTS", "DECODER_STATES", "DECODER_CHANNELS",
        "DECODER_SINGULARVALUES", "FIT_LOCKS", "F_CACHE_LINEAGE", "JOINT_GEOMETRY", "JOINT_MASK_CERTIFICATES",
        "FUTURE_DIAGNOSTIC_SCORES", "RETAINED_DISCARDED", "FORECAST_LABEL_PROBES", "WHOLE_COST", "WHOLE_COST_SAMPLES",
        "REGISTERED_ADOPTION", "EXECUTION_RESOURCES", "PRESERVED_FAILURES")
    tables = {name: [] for name in names}
    channel_keys = ("rank", "retained_condition", "direct_residual_MSE", "fallback_reason", "fit_executed", "rank_deficient_slope",
        "full_slope_identified", "constant_observed_response", "all_predictor_entries_zero", "zero_centered_predictor_variance",
        "true_observed_counts", "no_observed_response", "one_observed_response")
    for data, (channels, k, dev) in DATASETS.items():
        for split in ("fit", "tune", dev):
            case = science["datasets"][data]["spaces"]["PCA"]["conditions"]["L0_RECON"]["splits"][split]
            for method, entry in (("RAW_BOLT", case["RAW_BOLT"]), ("CURRENT_C_PERSISTENCE", case["current_C_persistence"])):
                for mask, key in (("all_observed", "scores_all_observed"), ("complete_C", "scores_complete_mask")):
                    tables["COMPARISON"] += score_rows({"dataset": data, "space": "NA", "condition": "NA", "split": split,
                        "method": method, "mask": mask}, entry[key], "baseline; persistence whole cost not measured" if method != "RAW_BOLT" else "same original raw baseline")
        for space in SPACES:
            parent = science["datasets"][data]["spaces"][space]
            for condition in CONDITIONS:
                entry, base = parent["conditions"][condition], {"dataset": data, "space": space, "condition": condition}
                meta = entry["fit_metadata"]
                tables["DECODER_STATES"].append({**base, "C": channels, "K": k, "lags": entry["lags"], "objective": entry["objective"],
                    "slope_parameters_per_C": meta["slope_parameters_per_C"], "total_slope_parameters": channels*meta["slope_parameters_per_C"],
                    "state_artifact": entry["state_artifact"], "fit_lock": entry["fit_lock"], "fit_wall_seconds": entry["fit_wall_seconds"], "fit_metadata": meta,
                    "coefficient_values_read_by_report": False})
                tables["FIT_LOCKS"].append({**base, **{key: entry[key] for key in ("status", "fit_lock", "FIT_input_source", "FIT_Bolt_cache", "FIT_future_targets_used",
                    "TUNE_DEV_used_for_fit_or_selection", "oracle_used_for_fit_or_selection")}})
                for channel in range(channels):
                    tables["DECODER_CHANNELS"].append({**base, "channel_zero_based": channel,
                        **{key: meta[key][channel] for key in channel_keys}, "singular_relative_cutoff": meta["singular_relative_cutoff"],
                        "centered_design": meta["centered_design"], "intercept_fitted": meta["intercept_fitted"], "slope_parameters_per_C": meta["slope_parameters_per_C"]})
                    for index, singular in enumerate(meta["singularvalues"][channel]):
                        tables["DECODER_SINGULARVALUES"].append({**base, "channel_zero_based": channel, "singular_index_zero_based": index,
                            "singularvalue": singular, "relative_cutoff": meta["singular_relative_cutoff"],
                            "retained": singular > meta["singularvalues"][channel][0]*meta["singular_relative_cutoff"]})
                for split in ("fit", "tune", dev):
                    case, identity = entry["splits"][split], {**base, "split": split}
                    require(case["status"] == "COMPLETE" and not case["diagnostic_failures"], "SUCCESSFUL_PRIMARY_AND_DIAGNOSTIC_CASE")
                    for mask, key in (("all_observed", "scores_all_observed"), ("complete_C", "scores_complete_mask")):
                        tables["COMPARISON"] += score_rows({**identity, "method": "ACTUAL_F", "mask": mask}, case[key], "deployable fixed cached F and FIT D")
                    raw, actual = case["RAW_BOLT"]["scores_all_observed"]["normalized"], case["scores_all_observed"]["normalized"]
                    reference = parent["conditions"]["L0_RECON"]["splits"][split]["scores_all_observed"]["normalized"]
                    tables["DELTA_COMPARISONS"].append({**identity, "MSE": actual["mse"], "MAE": actual["mae"],
                        "RAW_MSE": raw["mse"], "RAW_MAE": raw["mae"], "MSE_minus_RAW": actual["mse"]-raw["mse"], "MAE_minus_RAW": actual["mae"]-raw["mae"],
                        "MSE_ratio_to_RAW": actual["mse"]/raw["mse"], "MAE_ratio_to_RAW": actual["mae"]/raw["mae"],
                        "MSE_minus_L0_RECON": actual["mse"]-reference["mse"], "MAE_minus_L0_RECON": actual["mae"]-reference["mae"],
                        "full_observed_counts": actual["observed_counts"], "complete_target_positions": case["complete_target_positions"]})
                    if condition == "L0_RECON":
                        tables["F_CACHE_LINEAGE"].append({"dataset": data, "space": space, "split": split,
                            "input_source": case["input_source"], "cached_Bolt_source": case["cached_Bolt_source"], "E_source": parent["E_source"],
                            "cached_history_array": case["artifact"]["arrays"]["latent_history"], "cached_future_array": case["artifact"]["arrays"]["actual_F_future"],
                            "origins_array": case["artifact"]["arrays"]["origins"], "new_F_calls_in_science": 0})
                    floor = case["diagnostics"]["joint_floor"]
                    require(floor["status"] == "COMPLETE", "COMPLETE_NEW_JOINT_DIAGNOSTIC")
                    tables["JOINT_GEOMETRY"].append({**identity, "minimum_normalized_MSE": floor["minimum_normalized_MSE"],
                        "metadata": floor["metadata"], "all_HK_future_variables_retained": floor["all_HK_future_variables_retained"],
                        "complete_C_output_rows_only": floor["complete_C_output_rows_only"], "all_real_latent_forecasts_satisfy_registered_bound_claimed": False,
                        "RAW_bound_inequality_applied": False, "source_artifact": case["artifact"]})
                    for index, certificate in enumerate(floor["metadata"]["certificates"]):
                        tables["JOINT_MASK_CERTIFICATES"].append({**identity, "mask_pattern_index": index, **certificate})
                    tables["FUTURE_DIAGNOSTIC_SCORES"] += score_rows({**identity, "method": "JOINT_RETAINED_MINIMUM", "mask": "complete_C", "arithmetic": "FP64_numeric_retained_solution"},
                        floor["minimum_solution_scores_complete_mask"], "future-known diagnosis; only normalized MSE is minimized")
                    for method, result in case["diagnostics"]["methods"].items():
                        role = {"ACTUAL_F": "deployable latent forecast", "EY": "true-future diagnostic, unavailable at deployment",
                            "FLAT": "known-past flat-latent diagnostic; deployment-feasible control, whole cost not separately timed"}[method]
                        for mask, arithmetic, key in (("all_observed", "FP32", "scores_all_observed"), ("complete_C", "FP32", "scores_complete_mask"),
                            ("complete_C", "real_trace_FP64", "real_trace_scores_complete_mask"), ("complete_C", "retained_output_trace_FP64", "retained_output_real_trace_scores_complete_mask")):
                            tables["FUTURE_DIAGNOSTIC_SCORES"] += score_rows({**identity, "method": method, "mask": mask, "arithmetic": arithmetic}, result[key], role)
                        keys = ("real_trace_minus_minimum_normalized_MSE", "measured_FP32_rounding_normalized_MSE_allowance", "numerical_MSE_tolerance",
                            "real_trace_bound_inequality_pass", "FP32_bound_with_measured_allowance_pass", "retained_output_trace_bound_inequality_pass",
                            "discarded_numeric_map_output_squared_norm", "discarded_numeric_map_output_max_abs")
                        tables["RETAINED_DISCARDED"].append({**identity, "method": method, **{key: result[key] for key in keys},
                            "real_MSE": result["real_trace_scores_complete_mask"]["normalized"]["mse"],
                            "retained_MSE": result["retained_output_real_trace_scores_complete_mask"]["normalized"]["mse"], "minimum_MSE": floor["minimum_normalized_MSE"]})
                    tables["FORECAST_LABEL_PROBES"].append({**identity, **case["cached_future_label_perturbation"]})
            for split in ("fit", "tune", dev):
                for name, first, second in (("LAG_RECON", "L0_RECON", "L7_RECON"), ("LAG_FORECAST", "L0_FORECAST", "L7_FORECAST"),
                    ("OBJECTIVE_L0", "L0_RECON", "L0_FORECAST"), ("OBJECTIVE_L7", "L7_RECON", "L7_FORECAST")):
                    before, after = (parent["conditions"][condition]["splits"][split]["scores_all_observed"]["normalized"] for condition in (first, second))
                    tables["FACTORIAL_CONTRASTS"].append({"dataset": data, "space": space, "split": split, "contrast": name,
                        "before_condition": first, "after_condition": second, "before_MSE": before["mse"], "after_MSE": after["mse"],
                        "MSE_delta_after_minus_before": after["mse"]-before["mse"], "before_MAE": before["mae"], "after_MAE": after["mae"],
                        "MAE_delta_after_minus_before": after["mae"]-before["mae"], "independent_generalization_claimed": False})
        rows = cost["cost"][data]["rows"]
        require(len(rows) == 26 and {(row["route"], row["batch"]) for row in rows} == {(route, batch) for route in ROUTES for batch in (1, 8)}, "ALL_THIRTEEN_COST_ROUTES_B1_B8")
        for row in rows:
            raw = choose(rows, route="RAW_BOLT", batch=row["batch"])
            require(row["status"] == "COMPLETE" and row["repeats"] == len(row["seconds"]) == len(row["samples"]) == 13, "THIRTEEN_ACTUAL_REPEATS")
            ratio = row["median_seconds"]/raw["median_seconds"]
            tables["WHOLE_COST"].append({"dataset": data, **row, "ratio_of_RAW_medians": ratio,
                "median_paired_RAW_ratio": row["paired_RAW_median_ratio"], "memory_is_shared_not_exclusive": True})
            for index, sample in enumerate(row["samples"]):
                tables["WHOLE_COST_SAMPLES"].append({"dataset": data, "route": row["route"], "batch": row["batch"],
                    "repeat": sample["repeat"], "position": sample["position"], "seconds": sample["seconds"],
                    "paired_RAW_ratio": row["paired_RAW_ratios"][index], "sample": sample})
        for space in SPACES:
            for condition in CONDITIONS:
                point = choose(tables["DELTA_COMPARISONS"], dataset=data, space=space, condition=condition, split="tune")
                timing = choose(tables["WHOLE_COST"], dataset=data, route=space+"_"+condition+"_BOLT", batch=8)
                tables["REGISTERED_ADOPTION"].append({"dataset": data, "space": space, "condition": condition,
                    "TUNE_MSE_ratio_to_RAW": point["MSE_ratio_to_RAW"], "TUNE_MAE_ratio_to_RAW": point["MAE_ratio_to_RAW"],
                    "B8_ratio_of_RAW_medians": timing["ratio_of_RAW_medians"], "B8_median_paired_RAW_ratio": timing["median_paired_RAW_ratio"],
                    "TUNE_MSE_pass": point["MSE_ratio_to_RAW"] <= 1.01, "TUNE_MAE_pass": point["MAE_ratio_to_RAW"] <= 1.01,
                    "B8_cost_pass": timing["ratio_of_RAW_medians"] <= .9,
                    "all_registered_conditions_pass": point["MSE_ratio_to_RAW"] <= 1.01 and point["MAE_ratio_to_RAW"] <= 1.01 and timing["ratio_of_RAW_medians"] <= .9,
                    "DEV_used_for_selection": False, "new_method_selection_performed_by_report": False})
    for key in INPUTS:
        value = values[key]
        tables["EXECUTION_RESOURCES"].append({"stage": key, "status": value["status"], "resource_terminal": value.get("resource_terminal"),
            "wall_seconds": value.get("wall_seconds"), "totals": value.get("totals"), "new_F_calls": value.get("new_F_calls"),
            "decoder_fits_completed": value.get("decoder_fits_completed"), "fit_wall_seconds_sum": sum(row["fit_wall_seconds"] for row in tables["DECODER_STATES"]) if key == "science" else None})
    tables["PRESERVED_FAILURES"] = values["preserved_failures"]
    return tables


def rel(path, output):
    return Path(os.path.relpath(resolve(path), Path(output).resolve())).as_posix()


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                if isinstance(value, (dict, list)) else value for key, value in row.items()})


def figures(tables, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    folder = Path(output) / "figures"; folder.mkdir(exist_ok=False); images = []
    colors = ("#0072B2", "#D55E00", "#009E73", "#AA4499")
    def save(fig, name):
        for extension in ("png", "svg"):
            path = folder / (name + "." + extension)
            fig.savefig(path, dpi=170 if extension == "png" else None, bbox_inches="tight"); images.append(path)
        plt.close(fig)
    def metric(data, space, condition, split):
        return choose(tables["DELTA_COMPARISONS"], dataset=data, space=space, condition=condition, split=split)["MSE"]
    def baseline(data, split, method="RAW_BOLT", mask="all_observed"):
        return choose(tables["COMPARISON"], dataset=data, space="NA", condition="NA", split=split,
            method=method, mask=mask, scale="normalized")["mse"]
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.4), layout="constrained")
    for di, (data, (_, _, dev)) in enumerate(DATASETS.items()):
        for si, split in enumerate(("tune", dev)):
            ax = axes[di, si]
            for ci, condition in enumerate(CONDITIONS):
                points = [metric(data, space, condition, split) for space in SPACES]
                bars = ax.bar(np.arange(3) + (ci-1.5)*.19, points, width=.19, color=colors[ci], label=condition)
                ax.bar_label(bars, labels=[f"{point:.4f}" for point in points], rotation=90, fontsize=7, padding=2)
            ax.axhline(baseline(data, split), color="black", linestyle="--", label="Raw Bolt")
            ax.axhline(baseline(data, split, "CURRENT_C_PERSISTENCE"), color="#777777", linestyle=":", label="Current-C persistence")
            ax.set_xticks(range(3), [SHORT[s] for s in SPACES]); ax.set_ylabel("Full-observed channel-macro normalized MSE")
            ax.set_title(DISPLAY[data]+" / "+("TUNE" if split == "tune" else "exposed DEV")); ax.margins(y=.16)
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle("Same fixed E and Bolt cache: temporal decoder x FIT objective\nRECON: 480 observed past levels; FORECAST: 48 forecast innovations; no native refit")
    save(fig, "01_actual_accuracy")

    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5), layout="constrained")
    for di, (data, (_, _, dev)) in enumerate(DATASETS.items()):
        for ci, contrasts in enumerate((("LAG_RECON", "LAG_FORECAST"), ("OBJECTIVE_L0", "OBJECTIVE_L7"))):
            ax = axes[di, ci]
            for pi, (contrast, split) in enumerate((contrast, split) for contrast in contrasts for split in ("tune", dev)):
                points = [choose(tables["FACTORIAL_CONTRASTS"], dataset=data, space=space, split=split, contrast=contrast)["MSE_delta_after_minus_before"] for space in SPACES]
                ax.bar(np.arange(3)+(pi-1.5)*.19, points, width=.19, color=colors[pi//2], hatch="//" if split == dev else None,
                    label=contrast+" / "+("TUNE" if split == "tune" else "exposed DEV"))
            ax.axhline(0, color="black", linewidth=.8); ax.set_xticks(range(3), [SHORT[s] for s in SPACES])
            ax.set_ylabel("MSE after - before (negative is lower error)")
            ax.set_title(DISPLAY[data]+(" / add causal lags" if ci == 0 else " / forecast-trained D minus reconstruction D")); ax.legend(fontsize=7)
    fig.suptitle("Matched-cache contrasts: no independent repetitions or statistical confidence intervals\nLag adds 7x slopes; objective changes target/support as well as supervision")
    save(fig, "02_factorial_contrasts")

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.3), layout="constrained")
    for di, (data, (_, _, dev)) in enumerate(DATASETS.items()):
        for si, space in enumerate(SPACES):
            ax = axes[di, si]
            for mi, (method, arithmetic) in enumerate((("JOINT_RETAINED_MINIMUM", "FP64_numeric_retained_solution"),
                ("ACTUAL_F", "real_trace_FP64"), ("EY", "real_trace_FP64"), ("FLAT", "real_trace_FP64"))):
                points = [choose(tables["FUTURE_DIAGNOSTIC_SCORES"], dataset=data, space=space, condition=condition, split=dev,
                    method=method, mask="complete_C", arithmetic=arithmetic, scale="normalized")["mse"] for condition in CONDITIONS]
                ax.bar(np.arange(4)+(mi-1.5)*.18, points, width=.18, color=colors[mi], label=method)
            ax.axhline(baseline(data, dev, mask="complete_C"), color="black", linestyle="--", label="Raw: identical complete-C mask")
            ax.set_xticks(range(4), [CONDITION_SHORT[c] for c in CONDITIONS]); ax.set_ylabel("Same-mask normalized MSE")
            ax.set_title(DISPLAY[data]+" / "+SHORT[space]); ax.margins(y=.12)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("Exposed DEV joint-horizon diagnosis: shared HK future coordinates + known-prefix offset\nRetained numerical minimum, actual latent forecast, true-future E(Y), flat latent; EY is unavailable at deployment")
    save(fig, "03_joint_future_diagnosis")

    fig, axes = plt.subplots(2, 2, figsize=(16, 8.8), layout="constrained")
    labels = [SHORT[space]+"-"+CONDITION_SHORT[c] for space in SPACES for c in CONDITIONS]
    for di, (data, (_, _, dev)) in enumerate(DATASETS.items()):
        ax = axes[di, 0]
        for index, (space, condition) in enumerate((space, condition) for space in SPACES for condition in CONDITIONS):
            rows = [row for row in tables["DECODER_CHANNELS"] if row["dataset"] == data and row["space"] == space and row["condition"] == condition]
            ratios = [row["rank"]/row["slope_parameters_per_C"] for row in rows]
            center = statistics.mean(ratios)
            ax.errorbar(index, center, yerr=[[center-min(ratios)], [max(ratios)-center]], color=colors[CONDITIONS.index(condition)], fmt="o", capsize=3)
        ax.set_xticks(range(12), labels, rotation=55, ha="right"); ax.set_ylim(-.03, 1.08)
        ax.set_ylabel("Retained rank / coefficient features"); ax.set_title(DISPLAY[data]+" / response-channel rank mean and range")
        ax = axes[di, 1]
        for mi, method in enumerate(("ACTUAL_F", "EY", "FLAT")):
            points = [choose(tables["RETAINED_DISCARDED"], dataset=data, space=space, condition=condition, split=dev, method=method)["discarded_numeric_map_output_max_abs"]
                for space in SPACES for condition in CONDITIONS]
            ax.plot(range(12), points, marker="o", color=colors[mi], label=method)
        ax.set_xticks(range(12), labels, rotation=55, ha="right"); ax.set_yscale("symlog", linthresh=1e-12)
        ax.set_ylabel("Max absolute discarded map-output component"); ax.set_title(DISPLAY[data]+" / exposed DEV cutoff effects"); ax.legend(fontsize=8)
    fig.suptitle("Rank truncation is coordinate/support dependent; ranges are across channels, not confidence intervals\nFit cutoff: singular relative 1e-4. Joint diagnostic cutoff: 1e-15; no universal full-space floor claim")
    save(fig, "04_rank_and_discarded_components")

    fig, axes = plt.subplots(2, 2, figsize=(17, 9), layout="constrained")
    route_labels = ["Raw"] + [SHORT[s]+"-"+CONDITION_SHORT[c] for s in SPACES for c in CONDITIONS]
    for di, data in enumerate(DATASETS):
        for bi, batch in enumerate((1, 8)):
            ax = axes[di, bi]
            points = [choose(tables["WHOLE_COST"], dataset=data, route=route, batch=batch)["median_seconds_per_window"]*1000 for route in ROUTES]
            ax.bar(range(13), points, color=["#555555"]+[colors[CONDITIONS.index(c)] for s in SPACES for c in CONDITIONS])
            ax.set_xticks(range(13), route_labels, rotation=55, ha="right"); ax.set_ylabel("Median whole pipeline ms / window")
            ax.set_title(DISPLAY[data]+" / batch "+str(batch))
    fig.suptitle("New actual 13-route cohort: physical fill/normalize -> E -> GPU Bolt -> fixed lag D -> physical CPU return\n13 full circular rotations; repeated first B origins, shared residency, no exclusive-device claim")
    save(fig, "05_whole_pipeline_cost")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.7), layout="constrained")
    markers = ("o", "s", "^", "D")
    for di, data in enumerate(DATASETS):
        ax = axes[di]
        for row in [r for r in tables["REGISTERED_ADOPTION"] if r["dataset"] == data]:
            color, marker = colors[SPACES.index(row["space"])], markers[CONDITIONS.index(row["condition"])]
            x, paired, y = row["B8_ratio_of_RAW_medians"], row["B8_median_paired_RAW_ratio"], row["TUNE_MSE_ratio_to_RAW"]
            ax.plot([x, paired], [y, y], color=color, linewidth=.8)
            ax.scatter(x, y, color=color, marker=marker, s=60)
            ax.scatter(paired, y, edgecolors=color, facecolors="none", marker=marker, s=60)
            ax.annotate(CONDITION_SHORT[row["condition"]], (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.scatter(1, 1, color="black", marker="*", s=120); ax.axhline(1.01, color="black", linestyle="--"); ax.axvline(.9, color="black", linestyle=":")
        ax.set_title(DISPLAY[data]); ax.set_xlabel("B8 ratio to same-cohort Raw (filled: ratio of medians; open: paired median)")
        ax.set_ylabel("TUNE full-observed normalized MSE / Raw")
        ax.legend(handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[i], markeredgecolor=colors[i], label=SHORT[s]) for i, s in enumerate(SPACES)], fontsize=8)
    passed = sum(row["all_registered_conditions_pass"] for row in tables["REGISTERED_ADOPTION"])
    fig.suptitle(f"Declared accuracy + cost gate: {passed}/24 pass (MSE and MAE <= Raw x1.01; B8 ratio-of-medians <= .90)\nAll four conditions reported; DEV never selects a winner; MAE gate remains separate from this MSE axis")
    save(fig, "06_accuracy_cost_gate")
    return images


def make_text(values, tables, bindings, output):
    def metric(data, space, condition, split):
        return choose(tables["DELTA_COMPARISONS"], dataset=data, space=space, condition=condition, split=split)
    passed = sum(row["all_registered_conditions_pass"] for row in tables["REGISTERED_ADOPTION"])
    dev_better = sum(row["MSE_minus_RAW"] < 0 for row in tables["DELTA_COMPARISONS"] if row["split"] in ("val", "eval"))
    objective_tune = sum(row["MSE_delta_after_minus_before"] < 0 for row in tables["FACTORIAL_CONTRASTS"] if row["split"] == "tune" and row["contrast"].startswith("OBJECTIVE"))
    objective_dev = sum(row["MSE_delta_after_minus_before"] < 0 for row in tables["FACTORIAL_CONTRASTS"] if row["split"] in ("val", "eval") and row["contrast"].startswith("OBJECTIVE"))
    lines = ["# 고정 Bolt·표현에서 lag와 디코더 학습 목적을 나눈 결과", "",
        f"이번 2×2 대조는 **시간별 독립 복원만 바꾸면 전체 문제가 해결된다는 결과를 주지 않았다.** 기노출 DEV의 24조건 중 Raw보다 full-observed 정규화 MSE가 작은 조건은 {dev_better}/24이며, 사전 정확도·비용 동시 기준을 통과한 조건은 {passed}/24다. 예측 목적 D가 같은 lag의 재구성 D보다 MSE를 줄인 비교는 TUNE {objective_tune}/12, 기노출 DEV {objective_dev}/12다. 이것은 현재 세 E·작은 K·고정 Bolt와 새 선형 D 안의 결과이며 다른 표현·예측기의 불가능성을 뜻하지 않는다.", "",
        "**현재 확인한 것은 표현을 새로 학습한 성공이 아니라 디코더의 구조·학습 목적 개입이다.** 같은 E와 실제 Bolt 잠재 미래를 고정하고 lag0/lag7 및 과거 재구성/FIT 미래 예측 목적을 교차했다. E 자체, F, native head를 새로 학습하지 않았다. 과거의 관계를 시간적으로 복원하는 여지는 일부 있었으나, 출력 공간에 여지가 생기는 것과 실제 잠재 미래가 그 공간의 적절한 좌표에 도달하는 것은 별개다.", "",
        "**미래 누출과 정상 감독을 구분했다.** FORECAST D는 FIT의 실제 미래 Y를 학습 target으로 쓰는 지도학습이다. 배포용 특징은 origin에서 알려진 과거와 동결 Bolt 예측뿐이다. 24 D의 성공·실패 잠금 및 registry를 만든 뒤 TUNE·DEV NPZ를 열었고, 그 값으로 D를 재적합하거나 네 조건을 고르지 않았다. 72개의 좁은 perturbation 검사는 고정 E/F 캐시·D에서 scoring target/mask를 바꿔도 ACTUAL_F 예측이 같음을 확인한 것이다. E/F 전체를 독립 재실행한 누출 검사나 독립 일반화 검증으로 확대하지 않는다. FIT_TARGET E는 이전부터 FIT 미래 감독을 사용했다.", "",
        "**2×2의 비교 조건.** Robin C17/K4와 Beijing C11/K2, L512/H48, FIT256/TUNE64와 이미 노출된 VAL30/EVAL146, 원 scaler·origin·observed mask를 유지했다. L0와 L7 RECON은 모두 공통 과거32:512의 480시점에서 채널별 관측 level을 centered LS로 학습한다. FORECAST는 48개 실제 F 예측의 배포 delta-Z에서 Y−currentC를 비중심·무절편 LS로 적합한다. 두 목적은 같은 lag에서 배포 계수 수와 입력이 같지만 target support·분포·감독 정보가 다르다. lag7은 slope 수가 7배다. 두 비교 모두 singular 상대 cutoff 1e−4를 고정했지만 유효 rank·좌표별 정규화가 같다는 뜻은 아니다. 중첩된 행을 독립 표본 수로 세지 않는다.", "",
        "배포식은 `currentC + Σ_l (z_combined(h−l)−z_past(−l)) B_l`이다. `z_past(−l)`의 실제 index는 `-1-l`이며, 상대시점≤0은 알려진 과거, >0은 예측이다. lag의 단위는 원 자료의 sampling step이며 채널 간 인과 그래프를 학습하거나 인과성을 증명하는 실험이 아니다. 입력은 K streams·K 과거 buffer·현재 C anchor이며 strict K-only가 아니다. origin별 C 과거로 D를 새로 적합하지 않는다. flat 미래 latent도 알려진 lag prefix와 기준값 때문에 current-C persistence와 다를 수 있어 별도 진단했다. RECON의 재구성 잔차 r_t는 anchored 예측에서 r_0−r_h로 들어가므로 level 재구성 목적이 미래 innovation 오차를 직접 최소화하지는 않는다.", "",
        "아래 값은 모두 full-observed channel-macro 정규화 MSE다. 마스크가 다른 공간 진단과 섞지 않는다.", "", "```text", "dataset  E       condition       TUNE           exposed_DEV"]
    for data, (_, _, dev) in DATASETS.items():
        for space in SPACES:
            for condition in CONDITIONS:
                lines.append(f"{('Robin' if data == 'robin' else 'Beijing'):<9} {SHORT[space]:<7} {condition:<13} {metric(data,space,condition,'tune')['MSE']:.9f}   {metric(data,space,condition,dev)['MSE']:.9f}")
        raw = choose(tables["COMPARISON"], dataset=data, space="NA", condition="NA", split=dev, method="RAW_BOLT", mask="all_observed", scale="normalized")
        lines.append(f"{DISPLAY[data]} Raw exposed_DEV MSE={raw['mse']:.9f}")
    lines += ["```", "", "![실제 예측 비교](figures/01_actual_accuracy.png)", "[SVG](figures/01_actual_accuracy.svg)", "",
        "![같은 표현과 캐시의 2×2 차이](figures/02_factorial_contrasts.png)", "[SVG](figures/02_factorial_contrasts.svg)", ""]
    before = metric("robin", "PCA", "L0_RECON", "val"); after = metric("robin", "PCA", "L7_RECON", "val")
    f0 = choose(tables["JOINT_GEOMETRY"], dataset="robin", space="PCA", condition="L0_RECON", split="val")
    f7 = choose(tables["JOINT_GEOMETRY"], dataset="robin", space="PCA", condition="L7_RECON", split="val")
    lines += [f"**시간 구조의 효과는 조건부다.** 사전 비교인 Robin/PCA/RECON에서는 L0→L7에 따라 DEV MSE가 {before['MSE']:.9f}→{after['MSE']:.9f}, 동일 complete-C mask의 수치 유지 공간 minimum은 {f0['minimum_normalized_MSE']:.9f}→{f7['minimum_normalized_MSE']:.9f}다. 시간적 D가 일부 공간 제약을 완화할 수 있다는 단서지만, 더 큰 계수 수·과거 buffer·변경된 offset을 함께 사용한 비교다. 실제 F 결과가 Raw보다 좋은지와 분리해야 한다. FORECAST가 TUNE에서 좋아도 DEV에서 같은 방향으로 바뀌는지는 네 조건 전체 원수치로 판단한다.", "",
        "**공동 미래 진단은 배포 성능이나 보편적 하한이 아니다.** 미래 latent 자유변수는 horizon마다 K개인 HK이며 H×7K개의 독립 lag 특징이 아니다. HC×HK block causal map와 origin별 알려진 offset을 구성하고, complete-C horizon에 해당하는 출력 행만 선택했다. 관측이 없는 horizon의 잠재 미지수도 남겼다. 출력값의 SSE/(complete위치수×C)를 합산하며 창 평균을 다시 평균하지 않았다. 실제 미래 E(Y)는 불완전 C horizon 전체를 currentC로 채운 후 고정 E를 적용한 진단이며, lag를 통해 뒤 horizon에도 영향을 줄 수 있다. 최적 latent·배포 입력·선정 근거가 아니다.", "",
        "minimum은 masked map의 singular 상대 cutoff1e−15로 등록한 **유지된 수치 공간**의 정규화 MSE 최소다. 아주 작은 실제 방향을 버릴 수 있으므로 원 실제 trace/FP32(+실측 rounding allowance)와의 부등식은 참·거짓을 그대로 저장하고, 유지 공간으로 투영한 trace에만 최소 부등식을 필수 검산한다. MAE·물리 단위 오차는 그 해의 값이며 각각의 최소가 아니다. 이전 pointwise/global-rank-K floor는 재사용하지 않았다.", "",
        "최초 독립 감사는 일부 L7 masked map에서 LS와 SVD의 수치 rank가 달라 통과하지 못했다. 저장 A/관측 mask에 대한 별도 진단에서는 불일치 LS rank가 exact nonzero-column rank 상한을 초과하는 문제가 확인됐다. 새 감사는 정확한 zero 열만 LS 계산에서 제외하고 전체 HK 좌표로 복원하는 대수적으로 동등한 검산 경로를 사용한다. 선택된 관측 출력에 기여하지 않는 exact-null 열의 내부 처리이며, 해당 horizon의 Y가 결측이라는 이유만으로 미래 변수를 삭제한 것이 아니다. 원 D·예측·조건·cutoff·공차를 바꾼 개선이 아니다. 실패한 최초 감사와 원 source, 별도 진단 및 repair 문서는 모두 보존한다. 이를 학습 실패나 미래 누출로 단정하지 않는다.", ""]
    for data, (_, _, dev) in DATASETS.items():
        rows = [r for r in tables["RETAINED_DISCARDED"] if r["dataset"] == data]
        lines.append(f"{DISPLAY[data]}: 원 real-trace 부등식 거짓 {sum(not r['real_trace_bound_inequality_pass'] for r in rows)}/{len(rows)}, FP32+allowance 부등식 거짓 {sum(not r['FP32_bound_with_measured_allowance_pass'] for r in rows)}/{len(rows)}. 이 수치는 cutoff 정의와 함께 읽으며 버린 성분을 공차 확대나 PASS로 숨기지 않는다.")
    lines += ["", "![공동 미래 출력 진단](figures/03_joint_future_diagnosis.png)", "[SVG](figures/03_joint_future_diagnosis.svg)", "",
        "![학습 rank와 버린 출력 성분](figures/04_rank_and_discarded_components.png)", "[SVG](figures/04_rank_and_discarded_components.svg)", "",
        "**비용은 새 전체 실행으로 확인했다.** 과학 단계는 이미 실제로 실행된 F 캐시18개를 재사용한 CPU D 적합·추론·진단이며 새 F 호출0이다. 이를 F latency로 세지 않았다. 별도 비용은 13경로×2자료×B1/B8의52조건, warmup2회·13완전순환의676타이머 표본에서 실제 E→GPU F→새 D를 다시 실행했다. physical causal fill/FIT normalize부터 물리 단위 FP32 CPU 반환·최종 sync까지 포함하고, 고정 E/D 준비·FIT 적합·source/file/state SHA·guard·warmup·저장·memory snapshot은 제외한다. callback에 포함된 pure 배포 연산과 검증 비용은 포함된다. 같은 첫 B origins의 반복이므로676개의 독립 데이터가 아니다.", "",
        "아래는 B8 ms/window와 같은 cohort Raw 분모의 두 비율이다. ratio-of-medians와 반복별 paired ratio의 중앙값은 서로 다른 통계다. B1과 전체 ns timestamp·표본·parity·메모리는 CSV에 모두 보존한다. 공유 process에 F와 여러 E/D가 resident이므로 RSS/VRAM을 방법별 독점 메모리나 장비 독점으로 해석하지 않는다.", "", "```text", "dataset  route                         B8_ms/window  ratio_medians  median_paired"]
    for data in DATASETS:
        for route in ROUTES:
            row = choose(tables["WHOLE_COST"], dataset=data, route=route, batch=8)
            lines.append(f"{('Robin' if data == 'robin' else 'Beijing'):<9} {route:<31} {row['median_seconds_per_window']*1000:.6f}   {row['ratio_of_RAW_medians']:.6f}   {row['median_paired_RAW_ratio']:.6f}")
    lines += ["```", "", "![전체 배포 비용](figures/05_whole_pipeline_cost.png)", "[SVG](figures/05_whole_pipeline_cost.svg)", "",
        f"사전 동시 기준은 각 자료·12경로의 TUNE full-observed normalized MSE와 MAE가 각각 Raw×1.01 이하이고 B8 ratio-of-medians≤.90인지다. 통과 {passed}/24이며 DEV 최저 조건을 채택하거나 원수치를 선택적으로 제외하지 않았다. 일부 accuracy 또는 cost 개선을 동시 성공으로 바꾸지 않는다.", "",
        "![정확도와 비용의 동시 기준](figures/06_accuracy_cost_gate.png)", "[SVG](figures/06_accuracy_cost_gate.svg)", "",
        "**남는 판단.** 현재 비교는 decoder-only 통제이며 새 표현·동적 그래프·대조학습의 필요성을 입증하지 않는다. 큰 모델을 쓰면 해결된다는 증거도 없다. 기존 native는 이전 D에 맞춘 모델이어서 주비교에서 제외했으며 이를 새 D에서 최적화된 native 비교라고 부르지 않는다. 향후 새 D에 맞춘 predictor 검토가 필요하더라도 실제 F와 출력공간 minimum의 gap만으로 F 자체의 유일한 원인을 확정하면 안 된다. 이미 노출된 두 자료와 제한된 K에 대한 적응적 탐색이며 독립 일반화 검증은 남아 있다.", "",
        "**기존 결과와 방법의 위치.** 고정 E/F/GELU hidden의 outgoing D를 실제-Y innovation에 맞춘 이전 D2도 이미 실행됐고 독립 검산28,365/실패0이었다. 이번은 최초 predictive D가 아니라 같은 세 선형 E·cache·lag별 용량에서 학습 목적을 교차한 통제다. 잠재 성분의 lag로 선형 복원하고 성분별 예측을 복원식에 넣는 구조는 [ODPC §2·§2.2](https://arxiv.org/html/1708.04705v1)에 있다. 이번을 ODPC 전체 구현이나 새로운 lag 구조의 발견으로 주장하지 않는다.", "",
        "기존 joint-Y 4조건도 frozen F를 통과한 actual-C 예측 손실로 E/D를 600 update 학습했고, F 입력까지의 nonzero gradient 검사와 F 동결 hash를 기록했다. 따라서 이전 연구를 재구성만 학습한 것으로 설명하거나, F를 통한 E 업데이트 자체를 아직 시도하지 않은 해결책이라고 쓰지 않는다. 이번은 현재 세 선형 E와 캐시를 고정한 D-only 대조이며 그 기존 공동 학습과 다른 조건이다.", "",
        f"[기존 D2 source]({rel(bindings['D2_prior_source']['path'], output)}), [기존 D2 검산]({rel(bindings['D2_prior_audit']['path'], output)}), [기존 joint-Y source]({rel(bindings['joint_Y_prior_source']['path'], output)}), [기존 joint-Y 검산]({rel(bindings['joint_Y_prior_audit']['path'], output)}), [직전 raw-subset 보고서]({rel(bindings['parent_report']['path'], output)})를 보존한다.", "",
        "원수치·재개 근거는 다음 파일에 있다.", ""]
    for name in tables: lines.append(f"- [{name}.csv](tables/{name}.csv): {len(tables[name])}행")
    lines.append("")
    for name, binding in bindings.items(): lines.append(f"- [{name}]({rel(binding['path'], output)}): `{binding['sha256']}`")
    lines += ["", "계수 원값과 예측 배열은 원 scientific artifact에 있으며 보고서 생성기는 NPZ·모델을 열지 않는다. 상태 shape/dtype/SHA·관측수/rank/SVD 및 모든 정규화·물리 단위 점수는 metadata 그대로 CSV에 보존했다. 이 보고서는 논문/PPT 작성이나 Git 게시 완료를 뜻하지 않는다.", ""]
    return "\n".join(lines)


def generate(paths, output_path):
    output = resolve(output_path)
    require(output.parent == (HERE / "reports").resolve() and not output.exists(), "FRESH_LOCAL_REPORT_DIRECTORY_REQUIRED")
    values, bindings = load_verified(paths)
    tables = collect_tables(values)
    source = record(__file__); bindings["report_source"] = source
    output.mkdir(parents=True, exist_ok=False); folder = output / "tables"; folder.mkdir()
    for name, rows in tables.items(): write_csv(folder / (name + ".csv"), rows)
    images = figures(tables, output)
    with (output / "REPORT_KO.md").open("x", encoding="utf-8", newline="\n") as handle: handle.write(make_text(values, tables, bindings, output))
    require(record(__file__) == source, "EXECUTED_REPORT_SOURCE_CHANGED")
    for binding in bindings.values(): bound(binding)
    files = [{"path": path.relative_to(output).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size}
        for path in sorted(output.rglob("*")) if path.is_file()]
    manifest = {"status": "COMPLETE", "source": source, "inputs": bindings,
        "table_row_counts": {name: len(rows) for name, rows in tables.items()}, "files": files,
        "figure_png_count": sum(path.suffix == ".png" for path in images), "figure_svg_count": sum(path.suffix == ".svg" for path in images),
        "scientific_NPZ_or_model_read": False, "new_fit_F_or_training_executed": False, "protected_TEST_accessed": False,
        "future_known_diagnostics_separate": True, "independent_generalization_claimed": False,
        "all_four_conditions_and_72_predictions_preserved": True, "all_cost_routes_and_two_ratios_preserved": True}
    with (output / "MANIFEST.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")
    print(json.dumps({"status": "COMPLETE", "manifest": str(output / "MANIFEST.json"), "tables": len(tables), "files": len(files), "images": len(images)}), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (*INPUTS, "output"): parser.add_argument("--" + name.replace("_", "-"), required=True)
    arguments = vars(parser.parse_args()); destination = arguments.pop("output")
    generate(arguments, destination)
