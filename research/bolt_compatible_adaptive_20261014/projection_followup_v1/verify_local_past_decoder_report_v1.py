"""Independent metadata/CSV/artifact verification without generator or scientific IO."""
from pathlib import Path
import argparse
import csv
import json
import statistics
import xml.etree.ElementTree as ET

from verify_fit_target_report_v1 import require, sha, read, resolve, verify_record, leaf_check, links

HERE = Path(__file__).resolve().parent
DATASETS = {"robin": (17, 4, "val"), "beijing_aotizhongxin": (11, 2, "eval")}
PROJECTIONS = ("PCA_PINV", "FIT_TARGET_PINV")
BACKENDS = {"BOLT_NUMPY": ("BOLT",), "NATIVE_TORCH32": ("NATIVE_s92601", "NATIVE_s92602")}
ROUTES = ("RAW_BOLT", "PCA_PINV_BOLT", "FIT_TARGET_PINV_BOLT", "PCA_LOCAL_BOLT", "FIT_TARGET_LOCAL_BOLT")
SCALES = ("normalized", "physical_per_channel_diagnostic")
COUNTS = {"COMPARISON": 312, "SIGNED_EFFECT": 144, "STATIC_EINSUM_CONTROL": 36, "PAST_RECONSTRUCTION": 48,
    "LOCAL_BACKEND_METADATA": 24, "LOCAL_WINDOW_GEOMETRY": 3264, "LOCAL_CHANNEL_SOLVERS": 44304,
    "NEW_PER_ORIGIN_DIAGNOSTIC": 168, "NEW_PER_ORIGIN_GEOMETRY": 24, "WHOLE_COST": 20,
    "WHOLE_COST_SAMPLES": 120, "REGISTERED_ADOPTION": 8}
PRIOR = (("matched_projection_v1", "4ad724017706b690e2d19ce4e8bdc326e6663f8ca6b0bf6f157ca4641ef0d3e2"),
    ("oracle_rank_addendum_v2", "a75784106b241b7228bd09c6b930b94c2a15097c6fd9c9cde466b12cdcc1d4fe"),
    ("decoder_metric_intervention_v2", "1958073281ea0a9c7cd14bacab2b97b8b4a95063f1015a8c94b25bafe85b5fe5"),
    ("FIT_target_comparison_v1", "fc89475c76f43f0c30d6a0029d96c138c1997dd510c35600664af4fb5337f77c"),
    ("past_constant_gate_v1", "4d53cb0b42f882bc636f4f8e57e5d2c42620382274b7268a2f4345ee4a581fd9"))
FIGURES = ("01_actual_Bolt_comparison", "02_new_conditional_space_and_forecast_gap", "03_local_gain_and_rank",
           "04_past_reconstruction_vs_future_forecast", "05_actual_local_whole_cost")


def same_record(item, expected, suffix=".json"):
    path = verify_record(item, suffix)
    require(resolve(expected["path"]) == path and expected["sha256"] == item["sha256"], "EXACT_METADATA_PATH_SHA_BINDING")
    if "bytes" in item:
        require(item["bytes"] == path.stat().st_size, "BOUND_OPTIONAL_BYTES_CHANGED")
    if "bytes" in expected:
        require(expected["bytes"] == path.stat().st_size, "EXPECTED_OPTIONAL_BYTES_CHANGED")


def terminal(manifest):
    cpu, audit, cost, cost_audit = [read(verify_record(manifest[key], ".json"))
        for key in ("CPU_producer", "CPU_audit", "cost", "cost_audit")]
    require(cpu["status"] == cost["status"] == "COMPLETE" and audit["status"] == cost_audit["status"] == "PASS", "COMPLETE_CPU_COST_AND_EXACT_INDEPENDENT_PASSES")
    for item, expected in ((audit["diagnostic"], manifest["CPU_producer"]), (cost_audit["cost"], manifest["cost"]),
        (cost["local_diagnostic"], manifest["CPU_producer"]), (cost["local_audit"], manifest["CPU_audit"]),
        (cost_audit["local_diagnostic"], manifest["CPU_producer"]), (cost_audit["local_audit"], manifest["CPU_audit"])):
        same_record(item, expected)
    for key in ("producer", "fit_target_audit", "past_constant_diagnostic", "past_constant_audit", "past_constant_whole_cost", "past_constant_whole_cost_audit"):
        same_record(audit[key], cpu[key])
    for key in ("producer", "fit_target_audit"):
        same_record(cost[key], cpu[key]); same_record(cost_audit[key], cpu[key])
    for item, name in ((audit["audit_source"], "audit_local_past_decoder_probe_v1.py"),
                       (cost_audit["audit_source"], "audit_local_past_decoder_cost_v1.py")):
        require(verify_record(item, ".py").name == name, "ACCEPTED_INDEPENDENT_AUDIT_SOURCE")
    cpu_counts = {"dataset_split_cells_checked": 6, "RAW_control_cells_checked": 6,
        "backend_states_checked": 24, "completed_backend_batches_checked": 24, "method_pairs_checked": 36,
        "window_matrix_fits_checked": 3264, "channel_slots_checked": 44304, "native_seed_D_reuses_checked": 1632,
        "static_prediction_arrays_byte_preserved": 36, "local_prediction_arrays_checked": 36,
        "static_einsum_control_arrays_checked": 36, "cached_future_arrays_byte_preserved": 36, "score_field_checks": 2592,
        "new_per_origin_output_diagnostic_backend_cells_checked": 24, "new_per_origin_output_inequalities_checked": 36,
        "RAW_bound_inequalities_checked": 0}
    cost_counts = {"cost_cells_checked": 20, "recorded_timer_samples_checked": 120, "physical_local_prechecks_checked": 4,
        "physical_local_first8_window_fits_checked": 32, "selected_forecast_file_hashes_checked": 36, "canonical_physical_bridges_checked": 2}
    for value, expected in ((audit, cpu_counts), (cost_audit, cost_counts)):
        require(all(value["totals"].get(key) == count for key, count in expected.items()), "ALL_LOCAL_INDEPENDENT_AUDIT_COUNTS")
    for key, count in (("completed_method_pairs", 36), ("completed_backend_batches", 24), ("completed_window_matrix_fits", 3264), ("native_seed_D_reuses", 1632)):
        require(cpu[key] == count, "LOCAL_CPU_RECORDED_COMPLETE_COUNTS")
    require(cpu["completed_channel_solver_calls"] == audit["totals"]["observed_channel_solver_calls_checked"]
            and cpu["completed_no_observation_fallbacks"] == audit["totals"]["no_observation_fallbacks_checked"]
            and cpu["completed_channel_solver_calls"] + cpu["completed_no_observation_fallbacks"] == 44304, "OBSERVED_SOLVERS_VS_CHANNEL_SLOTS")
    require(not cpu["failed_paths"] and cpu["new_E_F_native_head_forward_runs"] == 0
            and cpu["new_E_F_or_native_head_training"] is False and cpu["new_backbone_training"] is False
            and cpu["future_labels_used_for_local_fit"] is False
            and cpu["method_checkpoint_threshold_selection_used"] is False, "LOCAL_KNOWN_PAST_FIT_SCOPE")
    require(cpu["inference_time_past_decoder_refitting"] is True and cpu["fit_uses_TUNE_DEV_known_past"] is True
            and cpu["same_decoder_side_information_as_static"] is False and cpu["past_C_condition_metadata_added"] is True
            and cpu["strict_K_only_compression"] is False and cpu["static_D_or_global_rankK_oracle_reused_as_local_lower_bound"] is False,
            "CHANGED_C_PAST_INFORMATION_AND_CONDITIONAL_SPACE")
    require(cost["new_training_updates"] == 0 and cost["new_backbone_training"] is False
            and cost["F_before"] == cost["F_after"] and tuple(cost["routes"]) == ROUTES and not cost["failed_paths"]
            and cost["completed_cost_cells"] == 20 and cost["recorded_timer_samples"] == cost["completed_cell_timer_samples"] == 120
            and cost["local_LS_and_implemented_diagnostics_inside_timer"] is True
            and cost["native_heads_loaded_or_timed"] is False and cost["local_coefficients_cached_inside_timed_pipeline"] is False,
            "ACTUAL_F_ONLY_LOCAL_LS_INSIDE_WHOLE_TIMERS")
    require(cpu["original_source_cache_selection_SHA_unchanged"] is True and cpu["independent_holdout"] is False
            and cost["source_cache_selection_SHA_unchanged"] is True and cost["original_selected_forecasts_and_locks_unchanged"] is True,
            "IMMUTABLE_SCIENCE_AND_EXPOSED_DEV")
    for value in (cpu, audit, cost, cost_audit):
        require(value["protected_TEST_accessed"] is False, "NO_PROTECTED_TEST")
    return cpu, audit, cost, cost_audit


def metric_entries(base, method, variant, mask, scores, role, extra=None):
    return [{**base, "method": method, "variant": variant, "mask": mask, "metric_scale": scale, **scores[scale],
        "role": role, "this_MSE_is_a_known_per_origin_D_minimum": method == "LOCAL_D_MINIMUM" and scale == "normalized",
        "MAE_is_minimum": False, "physical_MSE_is_minimum": False, "RAW_bound_applied": False, **(extra or {})} for scale in SCALES]


def expected_tables(cpu, audit, cost):
    """Extract values independently; timestamp arithmetic never trusts CSV summaries."""
    tables = {key: [] for key in COUNTS}
    channel_keys = ("true_observed_counts", "centered_slope_ranks", "affine_model_ranks_with_analytic_intercept",
        "rank_deficient_slope_channels", "observed_count_below_two", "no_observation_current_C_fallback", "retained_condition_number",
        "retained_condition_number_valid", "observed_reconstruction_MSE_real_fit", "observed_reconstruction_MSE_valid")
    for dataset, (channels, k, development) in DATASETS.items():
        row = cpu["datasets"][dataset]
        require(row["channels"] == channels and row["K"] == k, "REGISTERED_DATASET_DIMENSIONS")
        for split, n in (("fit", 256), ("tune", 64), (development, 30 if dataset == "robin" else 146)):
            cell = row["splits"][split]
            base = {"dataset": dataset, "projection": "NA", "split": split, "backend": "RAW"}
            for mask, suffix in (("all_original_observed", "all_observed"), ("all_C_complete", "complete_mask")):
                tables["COMPARISON"] += metric_entries(base, "RAW_BOLT", "STATIC", mask, cell["RAW_scores_" + suffix], "saved_Raw_baseline")
            for projection in PROJECTIONS:
                for backend, methods in BACKENDS.items():
                    branch = cell["routes"][projection]["backends"][backend]
                    checked = audit["datasets"][dataset]["splits"][split]["routes"][projection]["backends"][backend]
                    meta, diag = branch["local_fit_metadata"], checked["new_per_origin_output_space_diagnostic"]
                    base = {"dataset": dataset, "projection": projection, "split": split, "backend": backend}
                    require(meta["windows"] == n and meta["singular_relative_cut"] == 1e-4
                            and meta["future_labels_used_in_local_decoder_fit"] is False and meta["TUNE_or_DEV_threshold_selection"] is False,
                            "REGISTERED_OBSERVED_CENTERED_CUTOFF_AND_COUNT")
                    require(diag["minimum_only_normalized_MSE_in_each_known_D32_real_rowspace"] is True
                            and diag["old_static_or_global_rank_K_minimum_reused"] is False, "NEW_KNOWN_LOCAL_SPACE_DIAGNOSTIC_ONLY")
                    tables["LOCAL_BACKEND_METADATA"].append({**base, "C": channels, "K": k, "local_fit_metadata": meta,
                        **{key: branch[key] for key in ("local_state_artifact", "original_state_source", "cached_method_sources", "native_cached_past_byte_exact_and_D_shared", "artifact")},
                        "new_forecast_training": False, "same_side_information_as_static": False})
                    for window in range(n):
                        tables["LOCAL_WINDOW_GEOMETRY"].append({**base, "window_index_zero_based": window,
                            "max_abs_D": meta["max_abs_decoder_per_window"][window], "D_Frobenius_norm": meta["decoder_Frobenius_norm_per_window"][window],
                            "D_operator_norm": meta["decoder_operator_norm_per_window"][window], "new_output_D_rank": diag["rank_D_per_origin"][window],
                            "absolute_origin_value_not_assumed": True, "relative_condition_cap_is_absolute_gain_bound": False})
                        for channel in range(channels):
                            tables["LOCAL_CHANNEL_SOLVERS"].append({**base, "window_index_zero_based": window, "channel_index_zero_based": channel,
                                **{key: meta[key][window][channel] for key in channel_keys},
                                "singularvalues_observed_centered_Z": meta["singularvalues_observed_centered_Z"][window][channel]})
                    tables["PAST_RECONSTRUCTION"] += metric_entries(base, "OBSERVED_PAST_RECONSTRUCTION", "LOCAL", "past_original_observed",
                        branch["past_reconstruction_deployment_scores_observed_only"], "observed_past_fit_only_not_future_objective")
                    tables["NEW_PER_ORIGIN_GEOMETRY"].append({**base, "rank_D_per_origin": diag["rank_D_per_origin"],
                        "certificate_per_origin": diag["certificate_per_origin"], "minimum_only_normalized_MSE_in_each_known_D32_real_rowspace": True,
                        "old_static_or_global_rank_K_minimum_reused": False})
                    tables["NEW_PER_ORIGIN_DIAGNOSTIC"] += metric_entries(base, "LOCAL_D_MINIMUM", "LOCAL", "all_C_complete", diag["scores"], "known_future_given_known_local_D_real_minimum")
                    tables["NEW_PER_ORIGIN_DIAGNOSTIC"] += metric_entries(base, "RAW_BOLT", "STATIC", "all_C_complete", diag["RAW_scores_same_complete_mask"], "same_complete_mask_unconstrained_Raw")
                    require(set(branch["methods"]) == set(checked["methods"]) == set(methods), "COMPLETE_BACKEND_METHODS")
                    for method in methods:
                        values, audited = branch["methods"][method], checked["methods"][method]
                        for mask, suffix in (("all_original_observed", "all_observed"), ("all_C_complete", "complete_mask")):
                            for variant, key in (("STATIC", "static"), ("LOCAL", "local")):
                                tables["COMPARISON"] += metric_entries(base, method, variant, mask, values[key + "_scores_" + suffix],
                                    "frozen_native_head_intervention" if method.startswith("NATIVE") else "cached_Bolt_D_intervention")
                            for scale in SCALES:
                                old, new = values["static_scores_" + suffix][scale], values["local_scores_" + suffix][scale]
                                tables["SIGNED_EFFECT"].append({**base, "method": method, "mask": mask, "metric_scale": scale,
                                    "static_MSE": old["mse"], "local_MSE": new["mse"], "local_minus_static_MSE": new["mse"] - old["mse"],
                                    "static_MAE": old["mae"], "local_MAE": new["mae"], "local_minus_static_MAE": new["mae"] - old["mae"],
                                    "statistical_improvement_claimed": False})
                        tables["STATIC_EINSUM_CONTROL"].append({**base, "method": method, "static_arithmetic_replay": values["static_D_einsum_arithmetic_replay"],
                            "local_D_effect_includes_arithmetic_change": True})
                        extras = {key: audited[key] for key in ("measured_FP32_rounding_MSE_allowance", "local_real_minus_new_origin_specific_minimum_normalized_MSE",
                            "local_actual_minus_new_origin_specific_minimum_normalized_MSE", "new_local_real_arithmetic_scores_complete_mask")}
                        tables["NEW_PER_ORIGIN_DIAGNOSTIC"] += metric_entries(base, method, "LOCAL", "all_C_complete", audited["local_scores_complete_mask"], "actual_local_prediction_same_complete_mask", extras)
        rows = cost["cost"][dataset]["rows"]
        require(len(rows) == 10 and {(r["route"], r["batch"]) for r in rows} == {(route, b) for route in ROUTES for b in (1, 8)}, "TEN_UNIQUE_F_ONLY_COST_CELLS")
        for r in rows:
            raw = next(value for value in rows if value["route"] == "RAW_BOLT" and value["batch"] == r["batch"])
            seconds = [(s["perf_end_ns"] - s["perf_start_ns"]) / 1e9 for s in r["samples"]]
            raw_seconds = [(s["perf_end_ns"] - s["perf_start_ns"]) / 1e9 for s in raw["samples"]]
            require(len(seconds) == r["repeats"] == 6 and all(value > 0 for value in seconds)
                    and [s["repeat"] for s in r["samples"]] == list(range(6)), "SIX_ACTUAL_ORDERED_TIMER_SAMPLES")
            ratios = [value / baseline for value, baseline in zip(seconds, raw_seconds)]
            median, raw_median, paired = statistics.median(seconds), statistics.median(raw_seconds), statistics.median(ratios)
            require(r["seconds"] == seconds and r["median_seconds"] == median and r["median_seconds_per_window"] == median / r["batch"]
                    and r["paired_RAW_seconds"] == raw_seconds and r["paired_RAW_ratios"] == ratios and r["paired_RAW_median_ratio"] == paired,
                    "INDEPENDENT_TIMESTAMP_MEDIAN_PAIRED_RATIO_ARITHMETIC")
            for s, value in zip(r["samples"], seconds):
                require(s["seconds"] == value and s["unix_end_ns"] >= s["unix_start_ns"] and s["dataset"] == dataset
                        and s["route"] == r["route"] and s["batch"] == r["batch"], "EXACT_TIMER_SAMPLE_IDENTITIES")
            tables["WHOLE_COST"].append({"dataset": dataset, "route": r["route"], "batch": r["batch"], "repeats": 6,
                "median_seconds": median, "median_seconds_per_window": median / r["batch"], "ratio_of_RAW_medians": median / raw_median,
                "median_paired_RAW_ratio": paired, "seconds": seconds, "origins": r["origins"], "memory_shared_process": r["memory"],
                "includes": r["whole_includes"], "excludes": r["whole_excludes"], "memory_scope": r["memory_scope"],
                "selected_metadata": r["selected_metadata"], "reference": r["reference"], "timed_outputs": r["timed_outputs"], "native_timed": False})
            for index, s in enumerate(r["samples"]):
                tables["WHOLE_COST_SAMPLES"].append({"dataset": dataset, "route": r["route"], "batch": r["batch"], "repeat": s["repeat"],
                    "position": s["position"], "seconds": seconds[index], "paired_RAW_ratio": ratios[index], "sample": s})
        for route in ROUTES[1:]:
            projection = "PCA_PINV" if route.startswith("PCA") else "FIT_TARGET_PINV"
            variant = "local" if "LOCAL" in route else "static"
            values = row["splits"]["tune"]["routes"][projection]["backends"]["BOLT_NUMPY"]["methods"]["BOLT"]
            score, raw_score = values[variant + "_scores_all_observed"]["normalized"], row["splits"]["tune"]["RAW_scores_all_observed"]["normalized"]
            benchmark = next(value for value in tables["WHOLE_COST"] if value["dataset"] == dataset and value["route"] == route and value["batch"] == 8)
            mse, mae, latency = score["mse"] / raw_score["mse"], score["mae"] / raw_score["mae"], benchmark["ratio_of_RAW_medians"]
            tables["REGISTERED_ADOPTION"].append({"dataset": dataset, "route": route, "TUNE_MSE_ratio_to_RAW": mse,
                "TUNE_MAE_ratio_to_RAW": mae, "B8_ratio_of_RAW_medians": latency,
                "all_registered_conditions_pass": mse <= 1.01 and mae <= 1.01 and latency <= .9, "new_method_selection_used": False})
    return tables


def row_key(row, table):
    names = ("dataset", "projection", "split", "backend")
    if table in ("COMPARISON", "NEW_PER_ORIGIN_DIAGNOSTIC"):
        names += ("method", "variant", "mask", "metric_scale")
    elif table == "SIGNED_EFFECT": names += ("method", "mask", "metric_scale")
    elif table == "STATIC_EINSUM_CONTROL": names += ("method",)
    elif table == "PAST_RECONSTRUCTION": names += ("method", "mask", "metric_scale")
    elif table == "LOCAL_WINDOW_GEOMETRY": names += ("window_index_zero_based",)
    elif table == "LOCAL_CHANNEL_SOLVERS": names += ("window_index_zero_based", "channel_index_zero_based")
    elif table == "WHOLE_COST": names = ("dataset", "route", "batch")
    elif table == "WHOLE_COST_SAMPLES": names = ("dataset", "route", "batch", "repeat")
    elif table == "REGISTERED_ADOPTION": names = ("dataset", "route")
    return tuple(str(row[name]) for name in names)


def check_csv(path, table, expected):
    csv.field_size_limit(max(csv.field_size_limit(), Path(path).stat().st_size))
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle); actual, fields = list(reader), reader.fieldnames
    mapping = {row_key(row, table): row for row in expected}
    require(len(mapping) == len(expected) == len(actual) == COUNTS[table], "CSV_COUNT_OR_DUPLICATE:" + table)
    require(fields is not None and len(fields) == len(set(fields)) and set(fields) == {key for row in expected for key in row}, "CSV_COLUMNS_CHANGED:" + table)
    seen, checked = set(), 0
    for row in actual:
        key = row_key(row, table)
        require(key in mapping and key not in seen, "UNKNOWN_OR_DUPLICATE_CSV_ROW:" + table); seen.add(key)
        for field in fields:
            expected_value, text = mapping[key].get(field), row[field]
            if isinstance(expected_value, (dict, list)): value = json.loads(text)
            elif expected_value is None: value = None if text == "" else text
            elif isinstance(expected_value, bool): value = True if text == "True" else False if text == "False" else text
            elif isinstance(expected_value, int): value = int(text)
            elif isinstance(expected_value, float): value = float(text)
            else: value = text
            checked += leaf_check(value, expected_value, table + ":" + repr(key) + ":" + field)
    return checked


def verify(folder, prior_reports=None):
    folder = Path(folder).resolve()
    manifest = read(folder / "MANIFEST.json")
    require(manifest["status"] == "COMPLETE" and manifest["metadata_only"] is True and manifest["decoder_side_information_changed"] is True, "COMPLETED_METADATA_ONLY_CHANGED_SIDE_INFORMATION")
    for key in ("NPZ_or_model_opened", "training_or_F_forward", "protected_TEST_accessed", "strict_K_only_compression",
                "native_reoptimized_for_local_D", "new_native_deployment_cost_measured", "old_static_or_global_K_floor_reused",
                "MAE_or_physical_MSE_minimum_claimed", "RAW_bound_applied", "independent_generalization_claimed"):
        require(manifest[key] is False, "REPORT_SCOPE_CHANGED:" + key)
    require(verify_record(manifest["source"], ".py").name == "make_local_past_decoder_report_v1.py", "CANONICAL_REPORT_SOURCE")
    for item in manifest["source_bindings"].values():
        require(Path(item["path"]).suffix.lower() in (".py", ".md", ".json"), "REPORT_BINDINGS_METADATA_ONLY")
        verify_record(item)
    if prior_reports is None:
        prior_reports = tuple((HERE / "reports" / name / "REPORT_KO.md", digest) for name, digest in PRIOR)
    require(len(manifest["prior_reports"]) == len(prior_reports) == 5, "FIVE_APPROVED_PRIOR_REPORTS")
    for item, (path, digest) in zip(manifest["prior_reports"], prior_reports):
        require(resolve(item["path"]) == Path(path).resolve() and item["sha256"] == digest, "PRIOR_REPORT_VERSION_CHANGED")
        verify_record(item, ".md")
    cpu, audit, cost, cost_audit = terminal(manifest)
    tables = expected_tables(cpu, audit, cost)
    require(manifest["CSV_rows"] == COUNTS and {name: len(rows) for name, rows in tables.items()} == COUNTS, "EXACT_TWELVE_CSV_COUNTS")
    declared = set()
    for item in manifest["files"]:
        path = (folder / item["path"]).resolve()
        require(path.is_relative_to(folder) and item["path"] not in declared and path.suffix.lower() in (".md", ".csv", ".png", ".svg"), "ARTIFACT_ESCAPE_DUPLICATE_OR_SCIENTIFIC_FILE")
        declared.add(item["path"])
        require(path.is_file() and path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], "ARTIFACT_SHA_OR_BYTES_CHANGED:" + item["path"])
    actual = {path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file() and path.name != "MANIFEST.json"}
    require(actual == declared, "UNDECLARED_OR_MISSING_REPORT_FILES")
    expected_assets = {"REPORT_KO.md"} | {"tables/" + name + ".csv" for name in COUNTS}
    expected_assets |= {"figures/" + stem + "." + suffix for stem in FIGURES for suffix in ("png", "svg")}
    require(declared == expected_assets, "EXACT_TWELVE_TABLES_TEN_IMAGES_ONE_MD")
    fields = {name: check_csv(folder / "tables" / (name + ".csv"), name, rows) for name, rows in tables.items()}
    found, images = links(folder / "REPORT_KO.md")
    require(all(str(Path(path).resolve()) in {row["path"] for row in found} for path, _ in prior_reports), "FIVE_PRIOR_REPORT_LINKS")
    figures = {"figures/" + stem + "." + suffix for stem in FIGURES for suffix in ("png", "svg")}
    require(set(manifest["figures"]) == figures and len(manifest["figures"]) == 10
            and set(images) == {folder / path for path in figures if path.endswith(".png")} and len(images) == 5, "EXACT_FIVE_EMBEDDED_PNG_SVG_PAIRS")
    from PIL import Image
    parsed_figures = []
    for item in sorted(figures):
        path = folder / item
        if path.suffix == ".png":
            with Image.open(path) as picture:
                require(picture.format == "PNG" and min(picture.size) > 0, "PARSEABLE_PNG_REQUIRED")
                picture.verify()
            with Image.open(path) as picture:
                picture.load(); size = list(picture.size)
            parsed_figures.append({"path": item, "format": "PNG", "dimensions": size})
        else:
            document = ET.parse(path)
            require(document.getroot().tag == "{http://www.w3.org/2000/svg}svg", "PARSEABLE_SVG_REQUIRED")
            parsed_figures.append({"path": item, "format": "SVG"})
    return {"status": "PASS", "report": {"path": str(folder / "REPORT_KO.md"), "sha256": sha(folder / "REPORT_KO.md")},
        "manifest": {"path": str(folder / "MANIFEST.json"), "sha256": sha(folder / "MANIFEST.json")},
        "verifier": {"path": str(Path(__file__)), "sha256": sha(__file__)},
        **{key: manifest[key] for key in ("CPU_producer", "CPU_audit", "cost", "cost_audit", "prior_reports")},
        "files_checked": len(declared), "CSV_rows": COUNTS, "CSV_primitive_fields_checked": fields,
        "total_CSV_primitive_fields_checked": sum(fields.values()), "local_links_checked": len(found), "local_links": found,
        "PNG_count": 5, "SVG_count": 5, "parsed_figures": parsed_figures,
        "timer_medians_paired_ratios_adoption_recomputed": True, "report_generator_imported": False,
        "NPZ_or_model_opened": False, "training_or_F_forward": False, "protected_TEST_accessed": False,
        "scope": "Independent exact JSON-to-typed-CSV primitive leaf comparison, timer/adoption arithmetic, metadata SHA/bytes, fixed five prior reports, relative links, PNG decoding and SVG parsing. No scientific matrix/model load; visual layout and HTTP are separate."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    receipt = verify(args.report)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")
    print(json.dumps({key: receipt[key] for key in ("status", "files_checked", "total_CSV_primitive_fields_checked", "local_links_checked", "PNG_count", "SVG_count")}))
