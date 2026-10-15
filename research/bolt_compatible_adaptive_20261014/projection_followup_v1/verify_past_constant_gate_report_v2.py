"""Independent gate-report JSON/CSV arithmetic and local artifact verification."""
from pathlib import Path
import argparse
import csv
import json
import statistics

from verify_fit_target_report_v1 import require, sha, read, verify_record, leaf_check, links

HERE = Path(__file__).resolve().parent
DATASETS = {"robin": (17, 4, "val"), "beijing_aotizhongxin": (11, 2, "eval")}
METHODS = ("RAW_BOLT", "PCA_PINV_BOLT", "PCA_PINV_NATIVE_s92601", "PCA_PINV_NATIVE_s92602",
           "FIT_TARGET_PINV_BOLT", "FIT_TARGET_PINV_NATIVE_s92601", "FIT_TARGET_PINV_NATIVE_s92602")
ROUTES = ("RAW_BOLT", "RAW_BOLT_GATED", "PCA_PINV_BOLT", "PCA_PINV_BOLT_GATED", "FIT_TARGET_PINV_BOLT", "FIT_TARGET_PINV_BOLT_GATED")
SCALES = ("normalized", "physical_per_channel_diagnostic")
COUNTS = {"SCORES": 336, "SIGNED_EFFECT": 168, "GATE_CHANNELS": 84, "GATE_CASES": 6,
          "GATE_METHODS": 42, "WHOLE_COST": 24, "WHOLE_COST_SAMPLES": 144,
          "PHYSICAL_PREFLIGHT": 2, "REGISTERED_ADOPTION": 10, "AUDIT_REPAIR_PROVENANCE": 1}
PRIOR = (("matched_projection_v1", "4ad724017706b690e2d19ce4e8bdc326e6663f8ca6b0bf6f157ca4641ef0d3e2"),
         ("oracle_rank_addendum_v2", "a75784106b241b7228bd09c6b930b94c2a15097c6fd9c9cde466b12cdcc1d4fe"),
         ("decoder_metric_intervention_v2", "1958073281ea0a9c7cd14bacab2b97b8b4a95063f1015a8c94b25bafe85b5fe5"),
         ("FIT_target_comparison_v1", "fc89475c76f43f0c30d6a0029d96c138c1997dd510c35600664af4fb5337f77c"))


def same_bound_record(item, expected):
    if any(item.get(key) != expected.get(key) for key in ("path", "sha256")):
        return False
    path = verify_record(item)
    return "bytes" not in item or item["bytes"] == path.stat().st_size


def terminal(manifest):
    cpu, audit, cost, cost_pass, repair = [read(verify_record(manifest[key], ".json"))
        for key in ("CPU_producer", "CPU_audit", "cost", "cost_audit", "failure_identity")]
    require(cpu["status"] == cost["status"] == "COMPLETE" and audit["status"] == cost_pass["status"] == "PASS", "ALL_TERMINAL_ARTIFACTS")
    require(audit["diagnostic"] == manifest["CPU_producer"] and cost_pass["cost"] == manifest["cost"]
            and cost["gate_diagnostic"] == cost_pass["gate_diagnostic"] == manifest["CPU_producer"]
            and cost["gate_audit"] == cost_pass["gate_audit"] == manifest["CPU_audit"], "EXACT_COST_CPU_AUDIT_CHAIN")
    require(cost["producer"] == audit["producer"] == cost_pass["producer"] == cpu["producer"]
            and cost["fit_target_audit"] == audit["fit_target_audit"] == cost_pass["fit_target_audit"] == cpu["independent_FIT_target_audit"], "COMMON_FIT_SOURCE")
    verify_record(cpu["producer"], ".json"); verify_record(cpu["independent_FIT_target_audit"], ".json")
    for key, count in {"dataset_split_cells_checked": 6, "method_pairs_checked": 42,
                       "ungated_prediction_arrays_checked": 42, "gated_prediction_arrays_checked": 42, "score_field_checks": 2016}.items():
        require(audit["totals"][key] == count, "CPU_FULL_COVERAGE:" + key)
    for key, count in {"cost_cells_checked": 24, "recorded_timer_samples_checked": 144,
                       "physical_gate_prechecks_checked": 2, "paired_gate_timed_output_arrays_checked": 72,
                       "original_selected_forecast_hashes_checked": 36}.items():
        require(cost_pass["totals"][key] == count, "COST_FULL_COVERAGE:" + key)
    require(cost_pass["audit_source"]["path"].endswith("audit_past_constant_gate_cost_v2.py"), "ACCEPTED_COST_V2")
    verify_record(cost_pass["audit_source"], ".py")
    for artifact in (cpu, audit, cost, cost_pass): require(artifact["protected_TEST_accessed"] is False, "NO_TEST")
    require(cpu["new_training"] is False and cpu["new_encoder_forecaster_or_head_forward"] is False
            and cpu["E_F_native_head_forward_runs"] == 0 and cpu["future_labels_used_for_gate"] is False
            and cpu["method_selection_used"] is False and cpu["source_cache_selection_SHA_unchanged"] is True, "CACHED_PAST_ONLY_GATE_SCOPE")
    require(cost["new_training_updates"] == cost["projection_fits"] == 0 and cost["native_deployment_cost_measured"] is False
            and cost["new_native_head_forward"] is False and tuple(cost["routes"]) == ROUTES
            and cost["F_before"] == cost["F_after"] and cost["source_cache_selection_SHA_unchanged"] is True, "F_ONLY_COST_SCOPE")
    require(cpu["past_C_condition_metadata_added"] is True and cpu["same_decoder_side_information_as_ungated"] is False
            and cpu["static_original_fixed_D_floor_reused_as_gated_lower_bound"] is False, "ADDED_C_INFORMATION")
    require(repair["status"] == "CONFIRMED_AUDITOR_REGISTERED_SEED_LITERAL_ERROR_REPAIRED"
            and same_bound_record(repair["fresh_audit"], manifest["cost_audit"]) and same_bound_record(repair["cost_producer"], manifest["cost"])
            and repair["cost_experiment_reexecuted"] is False and repair["F_or_native_head_called"] is False
            and repair["original_failed_source_audit_and_receipt_preserved"] is True
            and repair["source_difference"]["exactly_one_seed_guard_literal_restoration_verified"] is True
            and repair["source_difference"]["prediction_metric_tolerance_counts_and_cost_measurement_unchanged"] is True, "PRESERVED_FAILURE_NO_RERUN")
    verify_record(repair["failed_audit"], ".json")
    return cpu, audit, cost, cost_pass, repair


def timer_statistics(row, raw):
    values = [(sample["perf_end_ns"] - sample["perf_start_ns"]) / 1e9 for sample in row["samples"]]
    raw_values = [(sample["perf_end_ns"] - sample["perf_start_ns"]) / 1e9 for sample in raw["samples"]]
    require(len(values) == len(raw_values) == 6 and all(value > 0 for value in values + raw_values), "SIX_POSITIVE_TIMERS")
    require(values == row["seconds"] and raw_values == row["paired_RAW_seconds"], "TIMER_TIMESTAMP_BINDING")
    require(sorted(row["positions"]) == list(range(6)) and row["repeats"] == 6, "LATIN_SIX_POSITION_COVERAGE")
    ratios = [value / base for value, base in zip(values, raw_values)]
    require(ratios == row["paired_RAW_ratios"] and statistics.median(values) == row["median_seconds"]
            and statistics.median(values) / row["batch"] == row["median_seconds_per_window"]
            and statistics.median(ratios) == row["paired_RAW_median_ratio"], "RECORDED_TIMING_AGGREGATES")
    return statistics.median(values) / statistics.median(raw_values), statistics.median(ratios)


def expected_tables(cpu, audit, cost, cost_pass, repair):
    tables = {name: [] for name in COUNTS}
    arrays = ("exact_flat_windows_per_channel", "gate_eligible_windows_per_channel", "eligible_windows_with_any_observed_future_per_channel",
              "eligible_windows_with_observed_future_change_per_channel", "eligible_observed_future_positions_per_channel", "eligible_observed_future_changed_positions_per_channel")
    for dataset, (channels, k, development) in DATASETS.items():
        for split in ("fit", "tune", development):
            cell = cpu["datasets"][dataset]["splits"][split]
            require(cell["status"] == "COMPLETE" and set(cell["methods"]) == set(METHODS), "COMPLETE_SEVEN_METHOD_CELL")
            base = {"dataset": dataset, "split": split, "C": channels, "K": k, "windows": cell["windows"],
                    "complete_target_positions": cell["complete_target_positions"], "total_target_positions": cell["total_target_positions"]}
            tables["GATE_CASES"].append({**base, "eligible_window_channel_pairs": sum(cell["gate_eligible_windows_per_channel"]),
                "eligible_window_channel_pairs_with_observed_future_change": sum(cell["eligible_windows_with_observed_future_change_per_channel"]),
                **{key: cell[key] for key in ("input_source", "artifact", "cached_prediction_sources", "future_change_diagnostic_scope")},
                "independent_sample_count_claimed": False})
            for channel in range(channels):
                tables["GATE_CHANNELS"].append({**base, "channel_index_zero_based": channel, **{key: cell[key][channel] for key in arrays},
                    "future_changes_used_for_gate": False, "past_flatness_guarantees_future_flatness": False})
            for method, value in cell["methods"].items():
                tables["GATE_METHODS"].append({**base, "method": method,
                    **{key: value[key] for key in ("unchanged_coordinates_byte_exact", "gated_coordinates_current_anchor_byte_exact",
                                                 "changed_prediction_coordinates_per_channel", "native_head_frozen_intervention")},
                    "optimized_gated_native_head_claimed": False})
                for mask, suffix in (("all_original_observed", "all_observed"), ("all_C_complete", "complete_mask")):
                    for scale in SCALES:
                        old, new = value["ungated_scores_" + suffix][scale], value["gated_scores_" + suffix][scale]
                        require(old["observed_counts"] == new["observed_counts"], "IDENTICAL_SCORING_MASK_COUNTS")
                        for variant, score in (("UNGATED", old), ("GATED", new)):
                            tables["SCORES"].append({**base, "method": method, "variant": variant, "mask": mask, "metric_scale": scale, **score,
                                "role": "cached_native_frozen_intervention" if "NATIVE" in method else "cached_F_output_intervention",
                                "new_E_F_head_forward": False, "physical_units_comparable": False})
                        tables["SIGNED_EFFECT"].append({**base, "method": method, "mask": mask, "metric_scale": scale,
                            "ungated_MSE": old["mse"], "gated_MSE": new["mse"], "gated_minus_ungated_MSE": new["mse"] - old["mse"],
                            "ungated_MAE": old["mae"], "gated_MAE": new["mae"], "gated_minus_ungated_MAE": new["mae"] - old["mae"],
                            "observed_counts": old["observed_counts"], "statistical_improvement_claimed": False})
        pre, independent = cost["datasets"][dataset]["physical_gate_precheck"], cost_pass["datasets"][dataset]["physical_gate_precheck"]
        require(pre["status"] == "PASS" and pre["all_three_gate_arrays_byte_exact"] is True
                and pre["tolerance_parity_used_to_accept_gate_decision"] is False, "PHYSICAL_BYTE_EXACT_GATE_PREFLIGHT")
        eligible = independent["physical_gate_eligible_window_channel_pairs_first8"]
        tables["PHYSICAL_PREFLIGHT"].append({"dataset": dataset, "actual_precheck": pre, "independent_precheck": independent,
            "eligible_window_channel_pairs_first8": eligible, "gate_output_change_possible_in_timed_prefix": eligible > 0, "exclusive_method_memory_measured": False})
        rows = cost["cost"][dataset]["rows"]
        require(len(rows) == 12 and {(row["route"], row["batch"]) for row in rows} == {(route, batch) for route in ROUTES for batch in (1, 8)}, "TWELVE_COST_CELLS")
        for row in rows:
            raw = next(item for item in rows if item["route"] == "RAW_BOLT" and item["batch"] == row["batch"])
            ratio, paired = timer_statistics(row, raw)
            tables["WHOLE_COST"].append({"dataset": dataset, "route": row["route"], "batch": row["batch"], "repeats": 6,
                "median_seconds": row["median_seconds"], "median_seconds_per_window": row["median_seconds_per_window"],
                "ratio_of_RAW_medians": ratio, "median_paired_RAW_ratio": paired, "seconds": row["seconds"], "origins": row["origins"],
                "memory_shared_process": row["memory"], "includes": row["whole_includes"], "excludes": row["whole_excludes"],
                **{key: row[key] for key in ("memory_scope", "selected_metadata", "timed_outputs", "reference")},
                "native_timed": False, "full_six_rotation_Latin": True, "gate_caused_speed_gain_claimed": False})
            for sample in row["samples"]:
                index = sample["repeat"]
                tables["WHOLE_COST_SAMPLES"].append({"dataset": dataset, "route": row["route"], "batch": row["batch"],
                    "repeat": index, "position": sample["position"], "seconds": sample["seconds"],
                    "paired_RAW_ratio": row["seconds"][index] / raw["seconds"][index], "sample": sample})
        raw_tune = cpu["datasets"][dataset]["splits"]["tune"]["methods"]["RAW_BOLT"]["ungated_scores_all_observed"]["normalized"]
        for route in ROUTES[1:]:
            gated = route.endswith("_GATED"); method = route[:-6] if gated else route
            metric = cpu["datasets"][dataset]["splits"]["tune"]["methods"][method][("gated" if gated else "ungated") + "_scores_all_observed"]["normalized"]
            mse, mae = metric["mse"] / raw_tune["mse"], metric["mae"] / raw_tune["mae"]
            latency = next(item["ratio_of_RAW_medians"] for item in tables["WHOLE_COST"] if item["dataset"] == dataset and item["route"] == route and item["batch"] == 8)
            tables["REGISTERED_ADOPTION"].append({"dataset": dataset, "route": route, "TUNE_MSE_ratio_to_ungated_RAW": mse,
                "TUNE_MAE_ratio_to_ungated_RAW": mae, "B8_ratio_of_ungated_RAW_medians": latency,
                "all_registered_conditions_pass": mse <= 1.01 and mae <= 1.01 and latency <= .9,
                "actual_method_selection_used": False, "gated_RAW_control_included": True})
    tables["AUDIT_REPAIR_PROVENANCE"].append({**{key: repair[key] for key in ("failed_audit", "fresh_audit", "failed_audit_source",
        "fresh_audit_source", "registered_producer_seed", "incorrect_v1_auditor_seed_literal", "source_difference", "regression_tests")},
        "cost_experiment_reexecuted": False, "F_or_native_head_called_in_repair": False, "original_failed_artifacts_preserved": True})
    return tables


def row_key(row, name):
    names = {"SCORES": ("dataset", "split", "method", "variant", "mask", "metric_scale"),
             "SIGNED_EFFECT": ("dataset", "split", "method", "mask", "metric_scale"),
             "GATE_CHANNELS": ("dataset", "split", "channel_index_zero_based"), "GATE_CASES": ("dataset", "split"),
             "GATE_METHODS": ("dataset", "split", "method"), "WHOLE_COST": ("dataset", "route", "batch"),
             "WHOLE_COST_SAMPLES": ("dataset", "route", "batch", "repeat"), "PHYSICAL_PREFLIGHT": ("dataset",),
             "REGISTERED_ADOPTION": ("dataset", "route"), "AUDIT_REPAIR_PROVENANCE": ("registered_producer_seed",)}[name]
    return tuple(str(row[key]) for key in names)


def check_csv(path, name, expected):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle); rows, fields = list(reader), reader.fieldnames
    mapping = {row_key(row, name): row for row in expected}
    require(len(rows) == len(expected) == len(mapping) == COUNTS[name], "CSV_COUNT_OR_DUPLICATE:" + name)
    require(set(fields) == {key for row in expected for key in row}, "CSV_COLUMNS:" + name)
    seen, checks = set(), 0
    for row in rows:
        key = row_key(row, name); require(key in mapping and key not in seen, "CSV_UNKNOWN_OR_DUPLICATE_ROW:" + name); seen.add(key)
        for field in fields:
            value, cell = mapping[key].get(field), row[field]
            if isinstance(value, (dict, list)): parsed = json.loads(cell)
            elif value is None: parsed = None if cell == "" else cell
            elif isinstance(value, bool): parsed = True if cell == "True" else False if cell == "False" else cell
            elif isinstance(value, int): parsed = int(cell)
            elif isinstance(value, float): parsed = float(cell)
            else: parsed = cell
            checks += leaf_check(parsed, value, name + ":" + repr(key) + ":" + field)
    return checks


def verify(folder):
    folder = Path(folder).resolve(); manifest = read(folder / "MANIFEST.json")
    require(manifest["status"] == "COMPLETE" and manifest["metadata_only"] is True and manifest["decoder_side_information_changed"] is True, "EXPLICIT_REPORT_SCOPE")
    for flag in ("NPZ_or_model_opened", "training_or_F_forward", "protected_TEST_accessed", "new_native_deployment_cost_measured",
                 "strict_K_only_compression", "bit_packed_transmission_cost_measured", "static_fixed_D_floor_applied_to_gate",
                 "independent_generalization_claimed", "gate_caused_speed_gain_claimed"):
        require(manifest[flag] is False, "REPORT_SCOPE_CHANGED:" + flag)
    verify_record(manifest["source"], ".py")
    for item in manifest["source_bindings"].values(): verify_record(item)
    require(len(manifest["prior_reports"]) == 4, "FOUR_PRIOR_REPORTS")
    for item, (version, digest) in zip(manifest["prior_reports"], PRIOR):
        require(Path(item["path"]).resolve() == HERE / "reports" / version / "REPORT_KO.md" and item["sha256"] == digest, "PRECISE_PRIOR_VERSION")
        verify_record(item, ".md")
    sources = terminal(manifest); expected = expected_tables(*sources)
    require(manifest["CSV_rows"] == COUNTS, "MANIFEST_CSV_COUNTS")
    declared = set()
    for item in manifest["files"]:
        path = (folder / item["path"]).resolve()
        require(path.is_relative_to(folder) and item["path"] not in declared, "ARTIFACT_ESCAPE_OR_DUPLICATE")
        declared.add(item["path"])
        require(path.is_file() and path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], "ARTIFACT_SHA_OR_BYTES:" + item["path"])
    actual = {path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file() and path.name != "MANIFEST.json"}
    require(actual == declared, "UNDECLARED_OR_MISSING_FILES")
    fields = {name: check_csv(folder / "tables" / (name + ".csv"), name, rows) for name, rows in expected.items()}
    found, images = links(folder / "REPORT_KO.md")
    require(len(images) == 4 and all(path.suffix == ".png" for path in images), "FOUR_PNG_EMBEDS")
    require(all(item["path"] in {row["path"] for row in found} for item in manifest["prior_reports"]), "FOUR_PRIOR_REPORT_LINKS")
    figures = {path.relative_to(folder).as_posix() for path in (folder / "figures").iterdir() if path.is_file()}
    require(figures == set(manifest["figures"]) and len(figures) == 8
            and sum(Path(path).suffix == ".png" for path in figures) == 4
            and sum(Path(path).suffix == ".svg" for path in figures) == 4, "FOUR_PNG_SVG_PAIRS")
    require(set(images) == {folder / path for path in figures if Path(path).suffix == ".png"}, "EXACT_FIGURE_EMBEDS")
    return {"status": "PASS", "report": {"path": str(folder / "REPORT_KO.md"), "sha256": sha(folder / "REPORT_KO.md")},
        "manifest": {"path": str(folder / "MANIFEST.json"), "sha256": sha(folder / "MANIFEST.json")},
        "verifier": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__))},
        **{key: manifest[key] for key in ("CPU_producer", "CPU_audit", "cost", "cost_audit", "failure_identity", "prior_reports")},
        "files_checked": len(declared), "CSV_rows": COUNTS, "CSV_primitive_fields_checked": fields,
        "total_CSV_primitive_fields_checked": sum(fields.values()), "local_links": found, "local_links_checked": len(found),
        "PNG_count": 4, "SVG_count": 4, "adoption_all_conditions_pass": sum(row["all_registered_conditions_pass"] for row in expected["REGISTERED_ADOPTION"]),
        "timer_aggregates_signed_differences_adoption_recomputed": True, "NPZ_or_model_opened": False,
        "training_or_F_forward": False, "protected_TEST_accessed": False,
        "scope": "JSON/CSV every primitive field, timing arithmetic, retained failure identity, local paths/bytes/SHA. Visual layout and HTTP separate."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--report", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); receipt = verify(args.report)
    with args.output.open("x", encoding="utf-8") as handle: json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")
    print(json.dumps({key: receipt[key] for key in ("status", "files_checked", "total_CSV_primitive_fields_checked", "local_links_checked", "PNG_count", "SVG_count", "adoption_all_conditions_pass")}))
