"""Independently check D-only report CSV metadata, bytes and local links."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from urllib.parse import unquote, urlsplit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATASETS = {"robin": (17, 4, "val"), "beijing_aotizhongxin": (11, 2, "eval")}
DECODERS = ("ORIGINAL_PAST_LS", "ENCODER_PINV")
HEADS = ("BOLT", "NATIVE_s92601", "NATIVE_s92602")
ROUTES = ("RAW_BOLT", "PCA_PAST_LS_BOLT", "FORECA_PAST_LS_BOLT", "PCA_ENCODER_PINV_BOLT", "FORECA_ENCODER_PINV_BOLT")
COUNTS = {"DEPLOYMENT_AND_ORACLE_SCORES": 192, "FIXED_D_DIAGNOSTIC": 24, "GEOMETRY": 24,
          "PAST_RECONSTRUCTION": 24, "SOURCE_LINEAGE": 12, "WHOLE_COST": 20, "WHOLE_COST_SAMPLES": 120, "REGISTERED_ADOPTION": 8}
PRIOR = ((HERE / "reports/matched_projection_v1/REPORT_KO.md", "4ad724017706b690e2d19ce4e8bdc326e6663f8ca6b0bf6f157ca4641ef0d3e2"),
         (HERE / "reports/oracle_rank_addendum_v2/REPORT_KO.md", "a75784106b241b7228bd09c6b930b94c2a15097c6fd9c9cde466b12cdcc1d4fe"))


def require(condition, label):
    if not condition:
        raise ValueError(label)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def verify_record(record, *, suffix=None):
    path = resolve(record["path"])
    require(path.is_file() and (suffix is None or path.suffix.lower() == suffix), "METADATA_RECORD_FILE_REQUIRED:" + str(path))
    require(sha(path) == record["sha256"], "METADATA_RECORD_SHA_CHANGED:" + str(path))
    return path


def check_terminal_bindings(manifest):
    paths = {key: verify_record(manifest[key], suffix=".json") for key in ("decoder", "decoder_audit", "cost", "cost_audit")}
    d, da, c, ca = (read(paths[key]) for key in ("decoder", "decoder_audit", "cost", "cost_audit"))
    require(d["status"] == c["status"] == "COMPLETE" and da["status"] == ca["status"] == "PASS", "BOTH_ACCEPTED_TERMINAL_PRODUCERS_REQUIRED")
    require(da["diagnostic"] == manifest["decoder"] and ca["cost"] == manifest["cost"], "EXACT_TERMINAL_AUDIT_BINDING")
    require(c["decoder_diagnostic"] == manifest["decoder"] and c["decoder_audit"] == manifest["decoder_audit"], "COST_BINDS_SAME_DECODER")
    for key in ("producer", "projection_audit", "span_diagnostic", "span_audit"):
        require(d[key] == da[key], "DECODER_AUDIT_SOURCE_CHAIN:" + key)
        verify_record(d[key], suffix=".json")
    for key in ("producer", "projection_audit"):
        require(c[key] == ca[key] == d[key], "COST_AUDIT_SOURCE_CHAIN:" + key)
    for key in ("decoder_diagnostic", "decoder_audit"):
        require(c[key] == ca[key], "EXACT_DECODER_COST_CHAIN:" + key)
    require(all(da["totals"][key] == expected for key, expected in {
        "base_cells_checked": 12, "decoder_cells_checked": 24, "cached_forecast_arrays_checked": 96,
        "original_predictions_byte_exact_checked": 48}.items()), "DECODER_AUDIT_COMPLETE_COUNTS")
    require(ca["totals"]["cost_cells_checked"] == 20 and ca["totals"]["recorded_timer_samples_checked"] == 120,
            "COST_AUDIT_COMPLETE_COUNTS")
    require(d["completed_cells"] == 12 and d["new_training"] is False and d["new_encoder_F_or_native_head_forward"] is False
            and d["future_labels_used_to_fit_alternative_D"] is False and d["method_checkpoint_branch_selection_used"] is False,
            "FIXED_CACHED_FORECAST_LABEL_FREE_INTERVENTION")
    require(c["new_training_updates"] == c["projection_fits"] == c["alternative_decoder_refits"] == 0
            and c["F_before"] == c["F_after"] and tuple(c["routes"]) == ROUTES, "FIXED_F_FIVE_COST_ROUTES")
    require(all(data["protected_TEST_accessed"] is False for data in (d, da, c, ca)), "PROTECTED_TEST_SCOPE")
    require(d["original_selected_forecasts_and_locks_unchanged"] is True and c["original_selected_forecasts_and_locks_unchanged"] is True,
            "ORIGINAL_RESULTS_PRESERVED")
    return d, c


def metrics(source):
    n, p = source["normalized"], source["physical_per_channel_diagnostic"]
    mappings = {"normalized_MSE": (n, "mse"), "normalized_MAE": (n, "mae"),
        "physical_macro_MSE_diagnostic": (p, "mse"), "physical_macro_MAE_diagnostic": (p, "mae"),
        "observed_counts": (n, "observed_counts"), "active_channels": (n, "active_channels"),
        "normalized_per_channel_MSE": (n, "per_channel_mse"), "normalized_per_channel_MAE": (n, "per_channel_mae"),
        "physical_per_channel_MSE": (p, "per_channel_mse"), "physical_per_channel_MAE": (p, "per_channel_mae")}
    return {column: value[field] for column, (value, field) in mappings.items()}


def independent_expected_tables(decoder, cost):
    expected = {name: [] for name in COUNTS}
    for dataset, (channels, k, dev) in DATASETS.items():
        for projection in ("PCA", "FORECA"):
            branch = decoder["datasets"][dataset]["projections"][projection]
            for split in ("fit", "tune", dev):
                cell = branch["splits"][split]
                base = {"dataset": dataset, "projection": projection, "split": split, "C": channels, "K": k,
                        "complete_target_positions": cell["complete_target_positions"], "total_target_positions": cell["total_target_positions"]}
                for mask, source_key in (("all_original_observed", "RAW_scores_all_observed"), ("all_C_complete", "RAW_scores_complete_mask")):
                    expected["DEPLOYMENT_AND_ORACLE_SCORES"].append({**base, "decoder": "NA", "predictor": "RAW_BOLT", "mask": mask,
                        "role": "saved_baseline_repeated_per_projection", **metrics(cell[source_key])})
                expected["SOURCE_LINEAGE"].append({**base, "source_input": cell["source_input"], "source_state": branch["source_state"],
                    "cached_BOLT": cell["cached_future_sources"]["BOLT"], "decoder_probe_artifact": cell["artifact"],
                    "alternative_construction": cell["alternative_decoder_construction"], "new_E_F_head_forward": False, "D_fitted_to_future": False})
                for variant in DECODERS:
                    row = cell["decoders"][variant]
                    for head in HEADS:
                        for mask, key in (("all_original_observed", "scores_all_observed"), ("all_C_complete", "scores_complete_mask")):
                            expected["DEPLOYMENT_AND_ORACLE_SCORES"].append({**base, "decoder": variant, "predictor": head, "mask": mask,
                                "role": "fixed_cached_forecast_D_intervention", **metrics(row["forecasts"][head][key])})
                    expected["DEPLOYMENT_AND_ORACLE_SCORES"].append({**base, "decoder": variant, "predictor": "E_Y", "mask": "all_C_complete",
                        "role": "actual_future_diagnostic_only", **metrics(row["forecasts"]["E_Y"]["scores_complete_mask"])})
                    floor = row["fixed_D_real_arithmetic_diagnostic_same_complete_mask"]
                    expected["FIXED_D_DIAGNOSTIC"].append({**base, "decoder": variant, "mask": "all_C_complete",
                        "role": "real_arithmetic_fixed_D_span_only", **metrics(floor["scores"]), "normalized_MSE_is_minimum": True,
                        "MAE_is_minimum": False, "physical_MSE_is_minimum": False, "Raw_is_constrained_to_this_D": False,
                        "independent_linear_checks": floor["independent_linear_checks"]})
                    expected["GEOMETRY"].append({**base, "decoder": variant, **row["geometry"],
                        "native_heads_reoptimized_for_new_D": False, "causal_obliqueness_only_effect": False})
                    expected["PAST_RECONSTRUCTION"].append({**base, "decoder": variant, "mask": "past_original_observed",
                        **metrics(row["past_reconstruction_observed_only"]), "bias_rule": row["past_reconstruction_bias_rule"],
                        "future_prediction_guaranteed": False})
        rows = cost["cost"][dataset]["rows"]
        require(len(rows) == 10, "TEN_COST_CELLS_PER_DATASET")
        for row in rows:
            raw = next(source for source in rows if source["route"] == "RAW_BOLT" and source["batch"] == row["batch"])
            require(len(row["samples"]) == len(row["seconds"]) == 6 and row["repeats"] == 6 and row["warmup_each_route"] == 2,
                    "ACTUAL_SIX_SAMPLES_TWO_WARMUPS")
            seconds = [(sample["perf_end_ns"] - sample["perf_start_ns"]) / 1e9 for sample in row["samples"]]
            raw_seconds = [(sample["perf_end_ns"] - sample["perf_start_ns"]) / 1e9 for sample in raw["samples"]]
            require(seconds == row["seconds"] and raw_seconds == row["paired_RAW_seconds"], "SOURCE_TIMER_ARITHMETIC")
            median, raw_median = statistics.median(seconds), statistics.median(raw_seconds)
            ratios = [value / baseline for value, baseline in zip(seconds, raw_seconds)]
            paired = statistics.median(ratios)
            require(median == row["median_seconds"] and paired == row["paired_RAW_median_ratio"], "SOURCE_MEDIAN_ARITHMETIC")
            expected["WHOLE_COST"].append({"dataset": dataset, "route": row["route"], "batch": row["batch"], "repeats": 6,
                "median_whole_seconds": median, "median_seconds_per_window": median / row["batch"], "ratio_of_RAW_medians": median / raw_median,
                "median_of_paired_RAW_ratios": paired, "seconds": seconds, "paired_RAW_seconds": raw_seconds,
                "positions": row["positions"], "origins": row["origins"], "warmup_each_route": 2, "shared_memory_scope": row["memory_scope"],
                "peak_RSS_bytes_shared_process": max(item["resident_rss_bytes"] for item in row["memory"]),
                "peak_reserved_VRAM_bytes_shared_process": max(item["peak_reserved_vram_bytes"] for item in row["memory"]),
                "whole_includes": row["whole_includes"], "whole_excludes": row["whole_excludes"], "stage_decomposition": False,
                "native_inference_cost_measured": False, "selected_metadata": row["selected_metadata"],
                "timed_outputs": row["timed_outputs"], "reference": row["reference"]})
            for index, sample in enumerate(row["samples"]):
                expected["WHOLE_COST_SAMPLES"].append({"dataset": dataset, "route": row["route"], "batch": row["batch"],
                    "repeat": sample["repeat"], "position": sample["position"], "whole_seconds": seconds[index],
                    "RAW_same_repeat_seconds": raw_seconds[index], "paired_RAW_ratio": ratios[index],
                    "perf_start_ns": sample["perf_start_ns"], "perf_end_ns": sample["perf_end_ns"],
                    "all_four_parity_checks_pass": all(check["pass"] for check in sample["parity"].values()),
                    "parity": sample["parity"], "shared_process_memory": sample["memory"], "sample_artifact": sample["sample_artifact"]})
        raw_tune = decoder["datasets"][dataset]["projections"]["PCA"]["splits"]["tune"]["RAW_scores_all_observed"]["normalized"]
        for route in ROUTES[1:]:
            projection = "PCA" if route.startswith("PCA") else "FORECA"
            variant = "ENCODER_PINV" if "ENCODER_PINV" in route else "ORIGINAL_PAST_LS"
            actual = decoder["datasets"][dataset]["projections"][projection]["splits"]["tune"]["decoders"][variant]["forecasts"]["BOLT"]["scores_all_observed"]["normalized"]
            benchmark = next(row for row in expected["WHOLE_COST"] if row["dataset"] == dataset and row["route"] == route and row["batch"] == 8)
            latency, mse, mae = benchmark["ratio_of_RAW_medians"], actual["mse"] / raw_tune["mse"], actual["mae"] / raw_tune["mae"]
            expected["REGISTERED_ADOPTION"].append({"dataset": dataset, "route": route, "registered_B8_ratio_of_medians": latency,
                "TUNE_MSE_ratio_to_RAW": mse, "TUNE_MAE_ratio_to_RAW": mae, "latency_le_0_90": latency <= .9,
                "MSE_le_1_01": mse <= 1.01, "MAE_le_1_01": mae <= 1.01,
                "all_registered_conditions_pass": latency <= .9 and mse <= 1.01 and mae <= 1.01,
                "actual_model_or_method_selection_used": False})
    return expected


def leaf_check(actual, expected, label):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), "NESTED_FIELDS_CHANGED:" + label)
        return sum(leaf_check(actual[key], value, label + "." + key) for key, value in expected.items())
    if isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), "NESTED_VECTOR_CHANGED:" + label)
        return sum(leaf_check(value, expected[index], label + f"[{index}]") for index, value in enumerate(actual))
    require(type(actual) is type(expected) and actual == expected and (not isinstance(actual, float) or math.isfinite(actual)), "CSV_SOURCE_VALUE_MISMATCH:" + label)
    return 1


def row_key(row, table):
    if table == "WHOLE_COST": names = ("dataset", "route", "batch")
    elif table == "WHOLE_COST_SAMPLES": names = ("dataset", "route", "batch", "repeat")
    elif table == "REGISTERED_ADOPTION": names = ("dataset", "route")
    elif table == "SOURCE_LINEAGE": names = ("dataset", "projection", "split")
    elif table == "DEPLOYMENT_AND_ORACLE_SCORES": names = ("dataset", "projection", "split", "decoder", "predictor", "mask")
    else: names = ("dataset", "projection", "split", "decoder")
    return tuple(str(row[key]) for key in names)


def check_csv(path, table, expected):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows, columns = list(reader), reader.fieldnames
    mapping = {row_key(row, table): row for row in expected}
    require(len(mapping) == len(expected) == len(rows) == COUNTS[table], "CSV_COUNTS_OR_EXPECTED_DUPLICATES:" + table)
    require(set(columns) == set(key for row in expected for key in row), "CSV_COLUMNS_CHANGED:" + table)
    seen, checked = set(), 0
    for row in rows:
        key = row_key(row, table)
        require(key not in seen and key in mapping, "CSV_DUPLICATE_OR_UNKNOWN_ROW:" + table)
        seen.add(key)
        source = mapping[key]
        for name in columns:
            value = source.get(name)
            text = row[name]
            if isinstance(value, (dict, list)): parsed = json.loads(text)
            elif value is None: parsed = None if text == "" else text
            elif isinstance(value, bool): parsed = True if text == "True" else False if text == "False" else text
            elif isinstance(value, int): parsed = int(text)
            elif isinstance(value, float): parsed = float(text)
            else: parsed = text
            checked += leaf_check(parsed, value, table + ":" + repr(key) + ":" + name)
    require(seen == set(mapping), "CSV_MISSING_ROWS:" + table)
    return checked


def check_links(report):
    links, images = [], []
    for match in re.finditer(r"(!?)\[[^\]]*\]\(([^)]+)\)", report.read_text(encoding="utf-8-sig")):
        target = match.group(2).strip().strip("<>")
        parts = urlsplit(target)
        if parts.scheme or not parts.path:
            continue
        path = (report.parent / unquote(parts.path)).resolve()
        require(path.is_file(), "MISSING_LOCAL_LINK:" + target)
        links.append({"target": target, "path": str(path)})
        if match.group(1): images.append(path)
    require(len(images) == 3 and all(path.suffix == ".png" for path in images), "THREE_ACTUAL_PNG_EMBEDS_REQUIRED")
    return links, images


def verify(folder, prior_reports=PRIOR):
    folder = Path(folder).resolve()
    manifest = read(folder / "MANIFEST.json")
    require(manifest["status"] == "COMPLETE" and manifest["metadata_only"] is True, "COMPLETED_METADATA_REPORT_REQUIRED")
    for key in ("scientific_NPZ_or_model_opened", "training_or_F_forward", "protected_TEST_accessed", "independent_generalization_claimed",
                "E_Y_full_observed_zero_filled_future_included_in_major_comparison", "native_head_reoptimized_for_new_D",
                "obliqueness_only_causal_effect_claimed", "cost_native_inference_measured"):
        require(manifest[key] is False, "REPORT_SCOPE_FLAG_CHANGED:" + key)
    verify_record(manifest["source"], suffix=".py")
    for item in manifest["source_bindings"].values(): verify_record(item)
    require(len(manifest["prior_reports"]) == len(prior_reports) == 2, "EXACT_TWO_PRIOR_REPORT_BINDINGS")
    for item, (path, digest) in zip(manifest["prior_reports"], prior_reports):
        require(resolve(item["path"]) == Path(path).resolve() and item["sha256"] == digest, "WRONG_PRIOR_REPORT_VERSION")
        verify_record(item, suffix=".md")
    decoder, cost = check_terminal_bindings(manifest)
    declared = set()
    for item in manifest["files"]:
        path = (folder / item["path"]).resolve()
        require(path.is_relative_to(folder) and item["path"] not in declared, "MANIFEST_ESCAPE_OR_DUPLICATE")
        declared.add(item["path"])
        require(path.is_file() and path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], "MANIFEST_BYTES_OR_SHA_CHANGED:" + item["path"])
    actual = {path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file() and path.name != "MANIFEST.json"}
    require(actual == declared, "UNDECLARED_OR_MISSING_REPORT_FILES")
    require(manifest["counts"] == COUNTS, "EXACT_CSV_MANIFEST_COUNTS")
    expected = independent_expected_tables(decoder, cost)
    field_checks = {name: check_csv(folder / "tables" / (name + ".csv"), name, rows) for name, rows in expected.items()}
    links, images = check_links(folder / "REPORT_KO.md")
    link_paths = {item["path"] for item in links}
    require(all(str(Path(path).resolve()) in link_paths for path, _ in prior_reports), "PRIOR_REPORT_MARKDOWN_LINKS_REQUIRED")
    figures = [path.relative_to(folder).as_posix() for path in sorted((folder / "figures").iterdir()) if path.is_file()]
    require(set(figures) == set(manifest["figures"]) and len(figures) == 6
            and sum(Path(path).suffix == ".png" for path in figures) == 3 and sum(Path(path).suffix == ".svg" for path in figures) == 3,
            "EXACT_THREE_PNG_THREE_SVG")
    require(set(images) == {folder / path for path in figures if Path(path).suffix == ".png"}, "EMBEDDED_IMAGE_SET_CHANGED")
    return {"status": "PASS", "report": {"path": str(folder / "REPORT_KO.md"), "sha256": sha(folder / "REPORT_KO.md")},
        "manifest": {"path": str(folder / "MANIFEST.json"), "sha256": sha(folder / "MANIFEST.json")},
        "verifier": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__))},
        "decoder": manifest["decoder"], "decoder_audit": manifest["decoder_audit"], "cost": manifest["cost"], "cost_audit": manifest["cost_audit"],
        "prior_reports": manifest["prior_reports"], "files_checked": len(declared), "CSV_rows": COUNTS,
        "CSV_primitive_fields_checked": field_checks, "total_CSV_primitive_fields_checked": sum(field_checks.values()),
        "local_links": links, "local_links_checked": len(links), "PNG_count": 3, "SVG_count": 3,
        "timer_and_median_arithmetic_recomputed_from_metadata": True, "ratio_of_medians_adoption_used": True,
        "scientific_NPZ_or_model_opened": False, "training_or_F_forward": False, "protected_TEST_accessed": False,
        "scope": "Independent CSV/JSON leaf values, scalar timing arithmetic, manifest bytes and SHA, approved prior report versions, local Markdown links. Image layout and external HTTP are separate checks."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    receipt = verify(args.report)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: receipt[key] for key in ("status", "files_checked", "total_CSV_primitive_fields_checked", "local_links_checked", "PNG_count", "SVG_count")}))
