"""Independent JSON/CSV leaf and artifact verification, with no scientific runtime."""
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
PROJECTIONS = ("PCA_PINV", "FIT_TARGET_PINV")
SEEDS = ("92601", "92602")
SCALES = ("normalized", "physical_per_channel_diagnostic")
ROUTES = ("RAW_BOLT", "PCA_PINV_BOLT", "FIT_TARGET_PINV_BOLT", "PCA_PINV_NATIVE", "FIT_TARGET_PINV_NATIVE")
COUNTS = {"COMPARISON": 84, "MATCHED_COMPLETE_DIAGNOSTIC": 144, "OUTPUT_SPACE_GEOMETRY": 12,
          "CHECKPOINTS": 40, "LEARNING_CURVE": 4800, "TRAINING_COST": 8, "FIT_SPACE_METADATA": 4,
          "FIT_TARGET_SPECTRUM": 28, "RESTRICTED_PAST_VARIANCE": 6, "WHOLE_COST": 20, "WHOLE_COST_SAMPLES": 120, "REGISTERED_ADOPTION": 8}
PRIOR = ((HERE / "reports/matched_projection_v1/REPORT_KO.md", "4ad724017706b690e2d19ce4e8bdc326e6663f8ca6b0bf6f157ca4641ef0d3e2"),
         (HERE / "reports/oracle_rank_addendum_v2/REPORT_KO.md", "a75784106b241b7228bd09c6b930b94c2a15097c6fd9c9cde466b12cdcc1d4fe"),
         (HERE / "reports/decoder_metric_intervention_v2/REPORT_KO.md", "1958073281ea0a9c7cd14bacab2b97b8b4a95063f1015a8c94b25bafe85b5fe5"))


def require(condition, label):
    if not condition: raise ValueError(label)


def sha(path):
    with Path(path).open("rb") as handle: return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    require(Path(path).suffix.lower() == ".json", "JSON_ONLY_METADATA")
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def verify_record(item, suffix=None):
    path = resolve(item["path"])
    require(path.is_file() and (suffix is None or path.suffix.lower() == suffix), "BOUND_FILE_REQUIRED:" + str(path))
    require(sha(path) == item["sha256"], "BOUND_FILE_SHA_CHANGED:" + str(path))
    return path


def terminal(manifest):
    result = read(verify_record(manifest["producer"], ".json")); audit = read(verify_record(manifest["audit"], ".json"))
    require(result["status"] == "COMPLETE" and audit["status"] == "PASS" and audit["producer"] == manifest["producer"], "ACCEPTED_EXACT_PRODUCER_AUDIT")
    expected = {"projection_states_checked": 4, "target_projection_fits_checked": 2, "reused_PCA_controls_checked": 2,
        "native_fits_checked": 8, "checkpoint_states_checked": 40, "cost_cells_checked": 20, "recorded_timer_samples_checked": 120,
        "saved_prediction_arrays_checked": 88, "output_subspace_diagnostic_cells_checked": 12,
        "fixed_D_real_method_inequalities_checked": 48, "bound_RAW_inequalities_checked": 0}
    require(all(audit["totals"].get(key) == value for key, value in expected.items()), "ALL_AUDIT_COHORT_COUNTS")
    require(result["projection_fits"] == 2 and result["reused_projection_states"] == 2 and result["projection_conditions"] == 4
            and result["native_training_fits"] == 8, "ACTUAL_TRAINING_COUNTS")
    for key in ("source_projection", "source_projection_audit"):
        require(result[key] == audit[key], "ORIGINAL_SOURCE_BINDING:" + key); verify_record(result[key], ".json")
    require(result["encoder_fit_uses_FIT_true_future"] == {"PCA_PINV": False, "FIT_TARGET_PINV": True}
            and result["original_LS_native_heads_reused_as_optimized_inverse_controls"] is False, "SUPERVISED_FIT_AND_FRESH_HEAD_SCOPE")
    require(result["canonical_native_batch_size"] == 32 and result["F_before"] == result["F_after"]
            and result["prior_source_results_caches_and_selection_locks_unchanged"] is True, "FIXED_F_AND_PRIOR_PRESERVATION")
    require(result["protected_TEST_accessed"] is False and audit["protected_TEST_accessed"] is False
            and result["new_backbone_training"] is False and result["independent_holdout"] is False, "PROTECTED_TEST_AND_EXPOSED_DEV")
    for dataset in DATASETS:
        data = result["datasets"][dataset]
        require(data["status"] == "COMPLETE" and set(data["projections"]) == set(PROJECTIONS), "FULL_DATASET_ROUTE_SCOPE")
        for route in PROJECTIONS:
            meta = data["projections"][route]["fit_metadata"]
            require(meta["TUNE_or_DEV_used_in_projection_or_decoder_fit"] is False and meta["actual_future_target_used_in_projection_or_decoder_fit"] == (route == "FIT_TARGET_PINV"), "NO_DEV_FIT_AND_EXPLICIT_LABEL_USE")
            for seed in SEEDS:
                n = data["projections"][route]["native"][seed]
                require(n["status"] == "COMPLETE" and n["metadata"]["updates"] == 600
                        and n["metadata"]["initial_weights_and_biases_zero"] is True and n["metadata"]["seed_changes_initial_weights"] is False,
                        "ACTUAL_ZERO_INITIALIZED_SIX_HUNDRED_UPDATE_HEADS")
    return result, audit


def metric_rows(base, predictor, seed, mask, role, scores, extra=None):
    values = []
    for scale in SCALES:
        values.append({**base, "predictor": predictor, "seed": seed, "mask": mask, "role": role, "metric_scale": scale,
            "this_MSE_is_a_fixed_D_minimum": predictor == "FIXED_D_MINIMUM" and scale == "normalized",
            **scores[scale], **(extra or {})})
    return values


def independent_expected_tables(result, audit):
    out = {table: [] for table in COUNTS}
    for dataset, (channels, k, dev) in DATASETS.items():
        data = result["datasets"][dataset]
        for split in ("fit", "tune", dev):
            out["COMPARISON"] += metric_rows({"dataset": dataset, "projection": "NA", "split": split}, "RAW_BOLT", None,
                "all_original_observed", "saved_raw_baseline", data["raw_scores"][split])
        for route in PROJECTIONS:
            condition = data["projections"][route]; meta = condition["fit_metadata"]
            out["FIT_SPACE_METADATA"].append({"dataset": dataset, "projection": route, "C": channels, "K": k,
                "supervised_FIT_future_training": route == "FIT_TARGET_PINV", "past_only_training": route == "PCA_PINV",
                "deployment_past_only": True, "TUNE_DEV_fit_used": False, "fit_metadata": meta,
                "state_artifact": condition["state_artifact"], "output_space_guarantees_predictability": False})
            if route == "FIT_TARGET_PINV":
                for index, singular in enumerate(meta["target_singularvalues"]):
                    energy = singular * singular
                    out["FIT_TARGET_SPECTRUM"].append({"dataset": dataset, "component": index + 1, "K": k,
                        "singular_value": singular, "squared_energy": energy, "energy_fraction": energy / meta["FIT_target_innovation_total_energy"],
                        "retained_top_K": index < k, "FIT_complete_positions": meta["fit_complete_future_positions"],
                        "FIT_total_positions": meta["fit_total_future_positions"], "not_predictability_score": True})
                for index, eigenvalue in enumerate(meta["retained_whitening_eigenvalues"]):
                    out["RESTRICTED_PAST_VARIANCE"].append({"dataset": dataset, "component": index + 1, "FIT_past_restricted_eigenvalue": eigenvalue,
                        "K": k, "relative_cut": meta["eigenvalue_relative_cut"], "post_whitening_latent_covariance": meta["latent_fit_covariance"], "not_forecast_accuracy": True})
            for split in ("fit", "tune", dev):
                out["COMPARISON"] += metric_rows({"dataset": dataset, "projection": route, "split": split}, "BOLT", None,
                    "all_original_observed", "deployed_past_only", condition["bolt"]["scores"][split])
            for seed in SEEDS:
                native = condition["native"][seed]; metadata = native["metadata"]
                for split in ("fit", "tune", dev):
                    out["COMPARISON"] += metric_rows({"dataset": dataset, "projection": route, "split": split}, "NATIVE", seed,
                        "all_original_observed", "fresh_matched_head", native["scores"][split], {"selected_update": metadata["selected_update"]})
                checkpoints = native["checkpoints"]
                require([row["update"] for row in checkpoints] == [0, 100, 200, 400, 600], "FIVE_ACTUAL_CHECKPOINTS")
                selected = min(checkpoints, key=lambda row: (row["tune"]["mse"], row["update"]))
                require(selected["update"] == metadata["selected_update"], "EARLIEST_MINIMUM_TUNE_CHECKPOINT")
                for row in checkpoints:
                    out["CHECKPOINTS"].append({"dataset": dataset, "projection": route, "seed": seed, "update": row["update"],
                        "selected": row["update"] == selected["update"], **row["tune"], "state_sha256": row["state_sha256"],
                        "state_artifact": row["state_artifact"], "prediction_artifact": row["tune_prediction_artifact"]})
                require([row["update"] for row in native["learning_curve"]] == list(range(1, 601)), "SIX_HUNDRED_LEARNING_RECORDS")
                for point in native["learning_curve"]:
                    out["LEARNING_CURVE"].append({"dataset": dataset, "projection": route, "seed": seed, **point,
                        "loss_scope": "minibatch per-channel observed macro MSE; active channels", "not_full_FIT_loss": True})
                out["TRAINING_COST"].append({"dataset": dataset, "projection": route, "seed": seed,
                    "train_wall_seconds": metadata["train_wall_seconds"], "updates": metadata["updates"], "parameters": metadata["parameters"],
                    "batch_size": metadata["batch_size"], "metadata": metadata, "original_LS_head_reused": False, "seed_changes_initial_weights": False})
            for split in ("fit", "tune", dev):
                diag = audit["output_subspace_diagnostics"][dataset][route][split]
                base = {"dataset": dataset, "projection": route, "split": split}
                scope = {"complete_target_positions": diag["complete_target_positions"], "total_target_positions": diag["total_target_positions"],
                    "fixed_D_minimum_objective_is_normalized_MSE": True, "MAE_is_minimum": False, "physical_MSE_is_minimum": False, "RAW_bound_applied": False}
                out["MATCHED_COMPLETE_DIAGNOSTIC"] += metric_rows(base, "RAW_BOLT", None, "all_C_complete", "saved_raw_same_mask", diag["RAW_scores_same_complete_mask"], scope)
                out["MATCHED_COMPLETE_DIAGNOSTIC"] += metric_rows(base, "FIXED_D_MINIMUM", None, "all_C_complete", "future_informed_fixed_D_real_minimum",
                    diag["fixed_D_real_arithmetic_solution_scores_same_complete_mask"], scope)
                require(set(diag["methods_same_complete_mask"]) == {"BOLT", "NATIVE_s92601", "NATIVE_s92602", "E_Y"}, "FOUR_MATCHED_DIAGNOSTIC_METHODS")
                for method, value in diag["methods_same_complete_mask"].items():
                    extra = {**scope, "saved_minus_fixed_D_minimum_normalized_MSE": value["saved_minus_fixed_D_minimum_normalized_MSE"],
                        "real_minus_fixed_D_minimum_normalized_MSE": value["real_minus_fixed_D_minimum_normalized_MSE"],
                        "measured_FP32_rounding_MSE_allowance": value["measured_FP32_rounding_MSE_allowance"]}
                    out["MATCHED_COMPLETE_DIAGNOSTIC"] += metric_rows(base, method, None, "all_C_complete",
                        "future_E_Y_diagnostic_only" if method == "E_Y" else "deployed_same_mask", value["saved_prediction_scores_same_complete_mask"], extra)
                out["OUTPUT_SPACE_GEOMETRY"].append({**base, **scope, "fixed_D_minimum_normalized_MSE": diag["fixed_D_minimum_normalized_MSE"],
                    "E_Y_minus_fixed_D_minimum_normalized_MSE": diag["E_Y_minus_fixed_D_minimum_normalized_MSE"], "geometry": diag["geometry"]})
        rows = result["cost"][dataset]["rows"]
        require({(row["route"], row["batch"]) for row in rows} == {(route, batch) for route in ROUTES for batch in (1, 8)} and len(rows) == 10, "EXACT_TEN_COST_CELLS")
        for row in rows:
            raw = next(value for value in rows if value["route"] == "RAW_BOLT" and value["batch"] == row["batch"])
            seconds = [(sample["perf_end_ns"] - sample["perf_start_ns"]) / 1e9 for sample in row["samples"]]
            raw_seconds = [(sample["perf_end_ns"] - sample["perf_start_ns"]) / 1e9 for sample in raw["samples"]]
            ratios = [value / baseline for value, baseline in zip(seconds, raw_seconds)]
            median, paired = statistics.median(seconds), statistics.median(ratios)
            require(len(seconds) == 6 and seconds == row["seconds"] and median == row["median_seconds"]
                    and ratios == row["paired_RAW_ratios"] and paired == row["paired_RAW_median_ratio"], "TIMER_AND_MEDIAN_SOURCE_ARITHMETIC")
            out["WHOLE_COST"].append({"dataset": dataset, "route": row["route"], "batch": row["batch"], "repeats": row["repeats"],
                "median_seconds": median, "median_seconds_per_window": median / row["batch"], "ratio_of_RAW_medians": median / statistics.median(raw_seconds),
                "median_paired_RAW_ratio": paired, "seconds": seconds, "origins": row["origins"], "memory_shared_process": row["memory"],
                "includes": row["whole_includes"], "excludes": row["whole_excludes"], "memory_scope": row["memory_scope"],
                "selected_metadata": row["selected_metadata"], "timed_outputs": row["timed_outputs"], "reference": row["reference"]})
            for index, sample in enumerate(row["samples"]):
                out["WHOLE_COST_SAMPLES"].append({"dataset": dataset, "route": row["route"], "batch": row["batch"], "repeat": sample["repeat"],
                    "position": sample["position"], "seconds": seconds[index], "paired_RAW_ratio": ratios[index], "sample": sample})
        for route in ROUTES[1:]:
            projection = "PCA_PINV" if route.startswith("PCA_PINV") else "FIT_TARGET_PINV"
            branch = data["projections"][projection]
            score = (branch["bolt"]["scores"]["tune"] if route.endswith("_BOLT") else branch["native"]["92601"]["scores"]["tune"])["normalized"]
            raw = data["raw_scores"]["tune"]["normalized"]
            benchmark = next(value for value in out["WHOLE_COST"] if value["dataset"] == dataset and value["route"] == route and value["batch"] == 8)
            mse, mae, latency = score["mse"] / raw["mse"], score["mae"] / raw["mae"], benchmark["ratio_of_RAW_medians"]
            out["REGISTERED_ADOPTION"].append({"dataset": dataset, "route": route, "TUNE_MSE_ratio": mse, "TUNE_MAE_ratio": mae,
                "B8_ratio_of_medians": latency, "all_registered_conditions_pass": mse <= 1.01 and mae <= 1.01 and latency <= .9,
                "actual_posthoc_method_selection_used": False})
    return out


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
    if table in ("COMPARISON", "MATCHED_COMPLETE_DIAGNOSTIC"): names = ("dataset", "projection", "split", "predictor", "seed", "metric_scale")
    elif table == "OUTPUT_SPACE_GEOMETRY": names = ("dataset", "projection", "split")
    elif table == "CHECKPOINTS": names = ("dataset", "projection", "seed", "update")
    elif table == "LEARNING_CURVE": names = ("dataset", "projection", "seed", "update")
    elif table == "TRAINING_COST": names = ("dataset", "projection", "seed")
    elif table == "FIT_SPACE_METADATA": names = ("dataset", "projection")
    elif table in ("FIT_TARGET_SPECTRUM", "RESTRICTED_PAST_VARIANCE"): names = ("dataset", "component")
    elif table == "WHOLE_COST": names = ("dataset", "route", "batch")
    elif table == "WHOLE_COST_SAMPLES": names = ("dataset", "route", "batch", "repeat")
    else: names = ("dataset", "route")
    return tuple("" if row[name] is None else str(row[name]) for name in names)


def check_csv(path, table, expected):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle); rows, fields = list(reader), reader.fieldnames
    mapping = {row_key(row, table): row for row in expected}
    require(len(rows) == len(expected) == len(mapping) == COUNTS[table], "CSV_COUNT_OR_DUPLICATE:" + table)
    require(set(fields) == {field for row in expected for field in row}, "CSV_COLUMNS_CHANGED:" + table)
    seen, checks = set(), 0
    for row in rows:
        key = row_key(row, table); require(key in mapping and key not in seen, "UNKNOWN_OR_DUPLICATE_CSV_ROW:" + table); seen.add(key)
        for field in fields:
            value, text = mapping[key].get(field), row[field]
            if isinstance(value, (dict, list)): parsed = json.loads(text)
            elif value is None: parsed = None if text == "" else text
            elif isinstance(value, bool): parsed = True if text == "True" else False if text == "False" else text
            elif isinstance(value, int): parsed = int(text)
            elif isinstance(value, float): parsed = float(text)
            else: parsed = text
            checks += leaf_check(parsed, value, table + ":" + repr(key) + ":" + field)
    return checks


def links(report):
    found, images = [], []
    for match in re.finditer(r"(!?)\[[^\]]*\]\(([^)]+)\)", report.read_text(encoding="utf-8-sig")):
        target = match.group(2).strip().strip("<>"); parts = urlsplit(target)
        if parts.scheme or not parts.path: continue
        path = (report.parent / unquote(parts.path)).resolve(); require(path.is_file(), "MISSING_LOCAL_LINK:" + target)
        found.append({"target": target, "path": str(path)})
        if match.group(1): images.append(path)
    return found, images


def verify(folder, prior_reports=PRIOR):
    folder = Path(folder).resolve(); manifest = read(folder / "MANIFEST.json")
    require(manifest["status"] == "COMPLETE" and manifest["metadata_only"] is True
            and manifest["supervised_FIT_future_training"] is True and manifest["deployment_past_only"] is True, "EXPLICIT_SUPERVISED_COMPLETED_REPORT")
    for flag in ("NPZ_or_model_opened", "training_or_F_forward", "protected_TEST_accessed", "past_only_or_unsupervised_training_claimed",
                 "independent_generalization_claimed", "new_model_or_method_selected_posthoc", "fixed_D_MAE_or_physical_MSE_minimum_claimed", "RAW_bound_applied"):
        require(manifest[flag] is False, "REPORT_SCOPE_CHANGED:" + flag)
    verify_record(manifest["source"], ".py")
    for item in manifest["source_bindings"].values(): verify_record(item)
    require(len(manifest["prior_reports"]) == len(prior_reports) == 3, "THREE_PRECISE_PRIOR_REPORTS")
    for item, (path, digest) in zip(manifest["prior_reports"], prior_reports):
        require(resolve(item["path"]) == Path(path).resolve() and item["sha256"] == digest, "PRIOR_REPORT_VERSION_CHANGED"); verify_record(item, ".md")
    result, audit = terminal(manifest)
    expected = independent_expected_tables(result, audit)
    require(manifest["CSV_rows"] == COUNTS, "MANIFEST_CSV_COUNTS")
    declared = set()
    for item in manifest["files"]:
        path = (folder / item["path"]).resolve(); require(path.is_relative_to(folder) and item["path"] not in declared, "ARTIFACT_ESCAPE_OR_DUPLICATE")
        declared.add(item["path"]); require(path.is_file() and path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], "ARTIFACT_SHA_OR_BYTES_CHANGED:" + item["path"])
    actual = {path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file() and path.name != "MANIFEST.json"}
    require(actual == declared, "UNDECLARED_OR_MISSING_REPORT_FILES")
    fields = {table: check_csv(folder / "tables" / (table + ".csv"), table, rows) for table, rows in expected.items()}
    found, images = links(folder / "REPORT_KO.md")
    require(len(images) == 5 and all(path.suffix == ".png" for path in images), "FIVE_PNG_EMBEDS")
    require(all(str(Path(path).resolve()) in {row["path"] for row in found} for path, _ in prior_reports), "PRIOR_MD_LINKS_REQUIRED")
    figures = {path.relative_to(folder).as_posix() for path in (folder / "figures").iterdir() if path.is_file()}
    require(figures == set(manifest["figures"]) and len(figures) == 10 and sum(Path(path).suffix == ".png" for path in figures) == 5
            and sum(Path(path).suffix == ".svg" for path in figures) == 5, "FIVE_PNG_SVG_PAIRS")
    require(set(images) == {folder / path for path in figures if Path(path).suffix == ".png"}, "EXACT_EMBEDDED_FIGURES")
    return {"status": "PASS", "report": {"path": str(folder / "REPORT_KO.md"), "sha256": sha(folder / "REPORT_KO.md")},
        "manifest": {"path": str(folder / "MANIFEST.json"), "sha256": sha(folder / "MANIFEST.json")},
        "verifier": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__))},
        "producer": manifest["producer"], "audit": manifest["audit"], "prior_reports": manifest["prior_reports"],
        "files_checked": len(declared), "CSV_rows": COUNTS, "CSV_primitive_fields_checked": fields,
        "total_CSV_primitive_fields_checked": sum(fields.values()), "local_links": found, "local_links_checked": len(found),
        "PNG_count": 5, "SVG_count": 5, "timer_medians_ratios_checkpoint_choice_recomputed_from_metadata": True,
        "NPZ_or_model_opened": False, "training_or_F_forward": False, "protected_TEST_accessed": False,
        "scope": "Independent terminal JSON leaf extraction and CSV comparison; arithmetic for timers, medians, paired ratios, gates and CP selection; artifact bytes/SHA and prior versions/local links. Visual layout and HTTP are separate."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--report", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); receipt = verify(args.report)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")
    print(json.dumps({key: receipt[key] for key in ("status", "files_checked", "total_CSV_primitive_fields_checked", "local_links_checked", "PNG_count", "SVG_count")}))
