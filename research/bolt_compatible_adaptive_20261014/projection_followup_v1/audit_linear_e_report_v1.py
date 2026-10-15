"""Independent metadata/CSV report checks; no generator, models, NPZ or scientific replay."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
from urllib.parse import unquote
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATASETS = {"robin": (17, 4, "val"), "beijing_aotizhongxin": (11, 2, "eval")}
LAGS, SEEDS = ("L0", "L7"), (92601, 92602)
INPUTS = ("probe", "train", "audit", "cost", "cost_audit")
REPORT_SHA = "eff21bf19811b2f5a09945494aeca5817d22a2dd4f6d7da8b406a46a8f467230"
MANIFEST_SHA = "67b73f080dedb654b26bb49c14e445ab7914af6d4900b504a2bf45aa0654a488"
REPORT_SOURCE_SHA = "32e2460ea4724144067f264ab1ceb0332cd04c5fce2296838024c119c9572466"
INPUT_SHA = {"probe": "00440bf6e09ac31db4666ffc656ec348cc4b5f698b18600c65d0ff99cf0d1b8d",
    "train": "3f0cbd9176f2e7882778452e537564aab7445e1b88c1534fd48b1bce7ef639a4",
    "audit": "9796b4ed53f74186bc2513db56fe321c2d450cdd9bffbb7c5f341766e73ee36d",
    "cost": "5038bc1cd6c52fadf4b9397b3eabb9f1bcf3085c9901d954222d43835c05a97d", "cost_audit": "7ddd11e9d201d4dad9e051a99e4fc0d52af7d0441a0ea3fef10c646f87827a8f"}
TABLES = ("SCORES", "PAIRED_E", "CHANNEL_COMPARISONS", "CHECKPOINTS", "UPDATES", "CONDITIONS", "PROBES", "SELECTION_LOCKS",
    "WHOLE_COST", "WHOLE_COST_SAMPLES", "ACCURACY_COST_REFERENCE", "EXECUTION_RESOURCES", "PROVENANCE", "PRESERVED_FAILURES")


def require(value, message):
    if not value: raise ValueError(message)


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT/path).resolve()


def record(path):
    path = resolve(path)
    with path.open("rb") as handle: digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


def parse(text):
    return json.loads(text, parse_constant=lambda token: (_ for _ in ()).throw(ValueError("NONFINITE_JSON:"+token)))


def metadata(binding):
    path = resolve(binding["path"])
    require(path.suffix.lower() in (".json", ".jsonl", ".md", ".py"), "METADATA_ONLY_BINDING")
    actual = record(path); require(actual["sha256"] == binding["sha256"], "EXACT_METADATA_SHA")
    if "bytes" in binding: require(actual["bytes"] == binding["bytes"], "EXACT_METADATA_BYTES")
    return path


def find(rows, **fields):
    selected = [row for row in rows if all(row.get(key) == value for key, value in fields.items())]
    require(len(selected) == 1, "UNIQUE_SOURCE_ROW:"+str(fields)); return selected[0]


def relative_values(value, folder):
    if isinstance(value, list): return [relative_values(v, folder) for v in value]
    if isinstance(value, dict):
        return {k: os.path.relpath(resolve(v), folder).replace("\\", "/") if k in ("path", "snapshot_path") and isinstance(v, str)
                else relative_values(v, folder) for k, v in value.items()}
    return value


def equal(actual, expected, label):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(expected).issubset(actual), label)
        return sum(equal(actual.get(k), v, label+"."+k) for k, v in expected.items())
    if isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label)
        return sum(equal(a, b, label) for a, b in zip(actual, expected))
    if isinstance(expected, bool) or expected is None or isinstance(expected, str): require(actual == expected, label)
    elif isinstance(expected, (int, float)):
        require(isinstance(actual, (int, float)) and not isinstance(actual, bool) and math.isfinite(actual)
                and actual == expected, label)
    else: raise ValueError("UNSUPPORTED_METADATA_TYPE")
    return 1


def check_table(path, expected):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle: rows = list(csv.DictReader(handle))
    require(len(rows) == len(expected), "CSV_ROW_COUNT:"+Path(path).name)
    count = 0
    for index, (actual, source) in enumerate(zip(rows, expected)):
        for key, value in source.items():
            text = actual.get(key); label = "CSV_CELL:"+Path(path).name+":"+str(index)+":"+key
            require(text is not None, label)
            if isinstance(value, bool): require(text in ("True", "False"), label)
            parsed = parse(text) if isinstance(value, (dict, list)) else (text == "True" if isinstance(value, bool) else None if value is None and text == "" else
                int(text) if isinstance(value, int) else float(text) if isinstance(value, float) else text)
            count += equal(parsed, value, label)
    return count


def expected_tables(values):
    t, result = values["train"], {name: [] for name in TABLES}
    def append_scores(base, item):
        for mask, field in (("all_observed", "scores_all_observed"), ("complete_C", "scores_complete_mask")):
            for scale, metrics in item[field].items():
                if scale in ("normalized", "physical_per_channel_diagnostic"):
                    result["SCORES"].append({**base, "mask": mask, "scale": scale, **metrics})
    for data, (_, _, dev) in DATASETS.items():
        for lag in LAGS:
            result["PROBES"].append({"dataset": data, "lag": lag, **values["probe"]["cases"][data][lag]})
            for split, item in t["CP0_LS_and_RAW_baselines"][data][lag].items():
                append_scores({"dataset": data, "lag": lag, "seed": 0, "learn_E": False, "method": "CP0_LS", "split": split}, item["CP0_LS"])
                if lag == "L0":
                    result["SCORES"] += [{"dataset": data, "lag": "NA", "seed": 0, "learn_E": False, "method": "RAW_BOLT", "split": split,
                        "mask": "all_observed", "scale": scale, **metric} for scale, metric in item["RAW_scores"].items()]
        for condition, row in t["datasets"][data]["conditions"].items():
            base = {"dataset": data, "condition": condition, **{key: row[key] for key in ("lag", "learn_E", "seed")}}
            cps = row["checkpoints"]; require([cp["update"] for cp in cps] == [0, 100, 200, 400, 600], "FIVE_CHECKPOINTS")
            selected = min(cps, key=lambda cp: (cp["tune"]["mse"], cp["update"]))
            require(selected["update"] == row["selected_update"] and selected["state"] == row["selected_state"]
                    and selected["tune"] == row["selected_tune"], "CHECKPOINT_SELECTION")
            result["CHECKPOINTS"] += [{**base, "selected": cp["update"] == selected["update"], **cp} for cp in cps]
            for split, item in (("fit", row["FIT"]), ("tune", selected), (dev, row["DEV"])):
                append_scores({**base, "method": "SELECTED_ADAPTER", "split": split}, item)
            if not row["learn_E"] and row["seed"] == SEEDS[0]:
                append_scores({"dataset": data, "lag": row["lag"], "seed": 0, "learn_E": False, "method": "CP0_LS", "split": "tune"}, row["CP0_TUNE"])
                if row["lag"] == "L0":
                    result["SCORES"] += [{"dataset": data, "lag": "NA", "seed": 0, "learn_E": False, "method": "RAW_BOLT", "split": "tune",
                        "mask": "all_observed", "scale": scale, **metric} for scale, metric in row["RAW_TUNE_scores"].items()]
            updates = [parse(line) for line in metadata(row["updates_log"]).read_text(encoding="utf-8").splitlines()]
            require([r["update"] for r in updates] == list(range(1, 601)), "EXACT_600_UPDATE_SEQUENCE")
            require(all(math.isfinite(r[key]) for r in updates for key in ("loss", "E_gradient_norm_before_clip", "D_gradient_norm_before_clip", "global_gradient_norm_before_clip")), "FINITE_UPDATE_LOGS")
            require(sum(r["clipped"] for r in updates)/600 == row["global_clipping_fraction"], "GLOBAL_CLIP_FRACTION")
            result["UPDATES"] += [{**base, **r} for r in updates]
            condition_fields = ("status", "selected_update", "selected_tune", "selected_state", "selected_state_sha256", "initial_state_sha256",
                "order_sha256", "FIT_channel_observed_counts", "global_clipping_fraction", "updates_log", "selection_lock", "training_updates_completed", "checkpoint_count")
            result["CONDITIONS"].append({**base, **{key: row[key] for key in condition_fields if key in row},
                "selected_parameter_diagnostics": selected["parameter_diagnostics"], "selected_TUNE_latent_variance": selected["TUNE_latent_variance"],
                "independent_parameter_change": values["audit"].get("datasets", {}).get(data, {}).get(condition, {}).get("parameter_change"),
                "parameter_change_missing_is_unmeasured_not_zero": True})
            if "selection_lock" in row: result["SELECTION_LOCKS"].append({**base, "selected_update": selected["update"], "selection_lock": row["selection_lock"]})
        for lag in LAGS:
            for seed in SEEDS:
                for split in ("fit", "tune", dev):
                    for mask in ("all_observed", "complete_C"):
                        for scale in ("normalized", "physical_per_channel_diagnostic"):
                            common = {"dataset": data, "lag": lag, "seed": seed, "split": split, "mask": mask, "scale": scale}
                            fixed, learn = (find(result["SCORES"], **common, method="SELECTED_ADAPTER", learn_E=flag) for flag in (False, True))
                            result["PAIRED_E"].append({**common, "fixed_MSE": fixed["mse"], "learn_MSE": learn["mse"], "MSE_learn_minus_fixed": learn["mse"]-fixed["mse"],
                                "fixed_MAE": fixed["mae"], "learn_MAE": learn["mae"], "MAE_learn_minus_fixed": learn["mae"]-fixed["mae"], "paired_order_seed_only_not_independent_initialization": True})
                            if mask != "all_observed" or scale != "normalized": continue
                            initial = find(result["SCORES"], dataset=data, lag=lag, split=split, method="CP0_LS", mask=mask, scale=scale)
                            raw = find(result["SCORES"], dataset=data, split=split, method="RAW_BOLT", mask=mask, scale=scale)
                            require(fixed["observed_counts"] == learn["observed_counts"] == initial["observed_counts"] == raw["observed_counts"], "CHANNEL_COUNT_PARITY")
                            for channel, n in enumerate(raw["observed_counts"]):
                                a, b, c, d = (x["per_channel_mse"][channel] for x in (fixed, learn, initial, raw))
                                result["CHANNEL_COMPARISONS"].append({"dataset": data, "lag": lag, "seed": seed, "split": split, "channel_index": channel,
                                    "observed_count": n, "fixed_MSE": a, "learn_MSE": b, "CP0_LS_MSE": c, "RAW_MSE": d,
                                    "learn_minus_fixed": b-a if n else None, "learn_minus_CP0": b-c if n else None, "learn_minus_RAW": b-d if n else None,
                                    "horizon_errors_measured_in_metadata": False})
        rows = values["cost"]["cost"][data]["rows"]
        for row in rows:
            raw = find(rows, route="RAW_BOLT", batch=row["batch"]); sec = row["seconds"]
            require(len(sec) == len(row["samples"]) == 9 and statistics.median(sec) == row["median_seconds"]
                and row["median_seconds_per_window"] == row["median_seconds"]/row["batch"], "COST_MEDIAN")
            ratios = [a/b for a, b in zip(sec, raw["seconds"])]
            require(ratios == row["paired_RAW_ratios"] and statistics.median(ratios) == row["paired_RAW_median_ratio"], "PAIRED_RATIO")
            result["WHOLE_COST"].append({"dataset": data, **row, "ratio_of_RAW_medians": row["median_seconds"]/raw["median_seconds"], "median_paired_RAW_ratio": statistics.median(ratios)})
            for index, sample in enumerate(row["samples"]):
                require(sample["seconds"] == (sample["perf_end_ns"]-sample["perf_start_ns"])/1e9 == sec[index], "TIMER_NS")
                result["WHOLE_COST_SAMPLES"].append({"dataset": data, "route": row["route"], "batch": row["batch"], "paired_RAW_ratio": ratios[index], **sample})
        for row in t["datasets"][data]["conditions"].values():
            route = row["lag"]+"_"+("E_LEARN" if row["learn_E"] else "E_FIXED")+"_s"+str(row["seed"])
            selected = next(cp for cp in row["checkpoints"] if cp["update"] == row["selected_update"])["scores_all_observed"]["normalized"]; raw = row["RAW_TUNE_scores"]["normalized"]; timing = find(result["WHOLE_COST"], dataset=data, route=route, batch=8)
            result["ACCURACY_COST_REFERENCE"].append({"dataset": data, **{key: row[key] for key in ("lag", "learn_E", "seed")}, "route": route,
                "TUNE_MSE_over_RAW": selected["mse"]/raw["mse"], "TUNE_MAE_over_RAW": selected["mae"]/raw["mae"],
                "B8_ratio_of_RAW_medians": timing["ratio_of_RAW_medians"], "B8_median_paired_RAW_ratio": timing["median_paired_RAW_ratio"],
                "prior_bundle_reference_criteria_met": selected["mse"] <= raw["mse"]*1.01 and selected["mae"] <= raw["mae"]*1.01 and timing["ratio_of_RAW_medians"] <= .90})
    for stage in ("probe", "train", "cost"):
        result["EXECUTION_RESOURCES"].append({"stage": stage, **values[stage].get("resource", values[stage].get("resource_terminal", {}))})
        result["PRESERVED_FAILURES"] += [{"stage": stage, **row} for row in values[stage].get("failed_paths", [])]
    for stage in INPUTS:
        fields = ("status", "mode", "F_before", "F_after", "new_backbone_training", "protected_TEST_accessed", "completed_cost_cells", "recorded_timer_samples", "cost_split")
        result["PROVENANCE"].append({"stage": stage, "metadata": {key: values[stage][key] for key in fields if key in values[stage]}})
    return result


def check_manifest_files(folder, files):
    paths = []
    for item in files:
        path = (folder/item["path"]).resolve(); require(path.is_relative_to(folder.resolve()), "REPORT_FILE_CONTAINMENT")
        actual = record(path); require(actual["sha256"] == item["sha256"] and actual["bytes"] == item["bytes"], "REPORT_FILE_SHA_BYTES")
        paths.append(path)
    require(len(set(paths)) == len(paths), "UNIQUE_MANIFEST_PATHS"); return paths


def inspect_images(paths):
    from PIL import Image
    png = svg = 0
    for path in paths:
        if path.suffix == ".png":
            with Image.open(path) as image: require(image.format == "PNG", "PNG_FORMAT"); image.verify()
            png += 1
        if path.suffix == ".svg":
            require(ET.parse(path).getroot().tag.split("}")[-1] == "svg", "SVG_ROOT"); svg += 1
    return png, svg


def check_reference_paths(value, folder):
    if isinstance(value, list): return sum(check_reference_paths(item, folder) for item in value)
    if not isinstance(value, dict): return 0
    count = 0
    for key, item in value.items():
        if key in ("path", "snapshot_path") and isinstance(item, str):
            require(not re.match(r"^[A-Za-z]:|^/", item) and (folder/item).resolve().is_file(), "RELATIVE_EXISTING_ARTIFACT_PATH")
            count += 1
        else: count += check_reference_paths(item, folder)
    return count


def check_markdown(folder, tables):
    text = (folder/"REPORT_KO.md").read_text(encoding="utf-8"); links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text)
    for target in links:
        require(not re.match(r"^[A-Za-z]:|^/|^file:", target), "RELATIVE_MARKDOWN_LINK_REQUIRED")
        if not re.match(r"https?://|#", target): require((folder/unquote(target.split("#", 1)[0])).resolve().is_file(), "EXISTING_LOCAL_LINK:"+target)
    primary = [row for row in tables["PAIRED_E"] if row["mask"] == "all_observed" and row["scale"] == "normalized"]
    for split, label in (("tune", "TUNE"), ("dev", "기노출 DEV")):
        n = sum(row["MSE_learn_minus_fixed"] < 0 for row in primary if (row["split"] == "tune") == (split == "tune") and row["split"] != "fit")
        require(label+" "+str(n)+"/8" in text, "MD_PAIRED_COUNT")
    cp_rows = cost_rows = 0
    for line in text.splitlines():
        words = line.split()
        if not words or words[0] not in ("Robin", "Beijing"): continue
        data = "robin" if words[0] == "Robin" else "beijing_aotizhongxin"
        if len(words) == 9 and words[1] in LAGS:
            cp_rows += 1
            lag, seed = words[1], int(words[2]); f, l = (find(tables["CONDITIONS"], dataset=data, lag=lag, seed=seed, learn_E=flag) for flag in (False, True))
            tune = find(primary, dataset=data, lag=lag, seed=seed, split="tune"); dev = find(primary, dataset=data, lag=lag, seed=seed, split=DATASETS[data][2])
            require(words[3:5] == [str(f["selected_update"]), str(l["selected_update"])], "MD_CP_VALUES")
            require(words[5:] == [f"{v:.9f}" for v in (tune["fixed_MSE"], tune["learn_MSE"], dev["fixed_MSE"], dev["learn_MSE"])], "MD_PRIMARY_NUMBERS")
        elif len(words) == 5:
            cost_rows += 1
            row = find(tables["WHOLE_COST"], dataset=data, route=words[1], batch=8)
            require(words[2:] == [f"{v:.6f}" for v in (row["median_seconds_per_window"]*1000, row["ratio_of_RAW_medians"], row["median_paired_RAW_ratio"])], "MD_COST_NUMBERS")
    require(cp_rows == 8 and cost_rows == 18, "ALL_MD_NUMERIC_ROWS")
    last_cp = sum(row["learn_E"] and row["selected_update"] == 600 for row in tables["CONDITIONS"])
    better = sum(row["mse"] < find(tables["SCORES"], dataset=row["dataset"], split=row["split"], method="RAW_BOLT", mask="all_observed", scale="normalized")["mse"]
        for row in tables["SCORES"] if row["method"] == "SELECTED_ADAPTER" and row["split"] in ("val", "eval") and row["mask"] == "all_observed" and row["scale"] == "normalized")
    require(f"선정 조건은 {better}/16" in text and f"중 {last_cp}/8은 마지막 CP600" in text, "MD_RAW_AND_LAST_CP_COUNTS")
    for batch, phrase, predicate in ((1, "B1이 더 느린", lambda r: r > 1), (8, "B8이 더 빠른", lambda r: r < 1)):
        count = sum(predicate(row["ratio_of_RAW_medians"]) for row in tables["WHOLE_COST"] if row["route"] != "RAW_BOLT" and row["batch"] == batch)
        require(f"{phrase} 조건은 {count}/16" in text, "MD_BATCH_COST_COUNT")
    for key, rows in tables.items(): require(f"{key}.csv](tables/{key}.csv): {len(rows)}행" in text, "MD_CSV_ROW_COUNTS")
    for token in ("strict K-only가 아니다", "독립 일반화는 확인하지 않았다", "독립 VJP replay", "minibatch 순서만", "DEV timing은 하지 않았다", "처음 예측 목적으로 바꾼 시도로 포장하지 않는다"):
        require(token in text, "MD_SCOPE:"+token)
    return len(links)


def verify(folder):
    require(all(isinstance(v, str) and len(v) == 64 for v in (REPORT_SHA, MANIFEST_SHA, REPORT_SOURCE_SHA, *INPUT_SHA.values())), "ACTUAL_REPORT_GATE_NOT_REGISTERED")
    folder = resolve(folder); manifest_path = folder/"MANIFEST.json"; report_path = folder/"REPORT_KO.md"
    require(record(manifest_path)["sha256"] == MANIFEST_SHA and record(report_path)["sha256"] == REPORT_SHA, "EXACT_ACTUAL_REPORT_MANIFEST")
    manifest = parse(manifest_path.read_text(encoding="utf-8")); require(manifest["status"] == "COMPLETE", "COMPLETE_REPORT_MANIFEST")
    require(Path(manifest["source"]["path"]).name == "make_linear_e_adaptation_report_v2.py"
            and manifest["source"]["sha256"] == REPORT_SOURCE_SHA, "CANONICAL_V2_REPORT_SOURCE")
    metadata(manifest["source"]); values = {}
    for stage in INPUTS:
        binding = manifest["inputs"][stage]; require(binding["sha256"] == INPUT_SHA[stage], "EXACT_STAGE_SHA:"+stage)
        values[stage] = parse(metadata(binding).read_text(encoding="utf-8-sig"))
        require(values[stage]["status"] == ("PASS" if "audit" in stage else "COMPLETE"), "ACCEPTED_STAGE_STATUS")
    registry = parse(metadata(values["train"]["selection_registry"]).read_text(encoding="utf-8"))
    require(registry["status"] == "ALL_16_RESOLVED_BEFORE_DEV" and registry["DEV_opened_before_seal"] is False and len(registry["conditions"]) == 16, "ALL_16_SELECTION_LOCKS")
    for item in registry["conditions"]:
        lock = parse(metadata(item["selection_lock"]).read_text(encoding="utf-8")); require(lock["DEV_opened_before_lock"] is False and lock["status"] == "COMPLETE", "SEALED_SELECTED_LOCK")
    tables = expected_tables(values); fields, artifact_paths = {}, 0
    for key in TABLES:
        require(manifest["table_row_counts"][key] == len(tables[key]), "MANIFEST_TABLE_COUNT:"+key)
        expected = relative_values(tables[key], folder)
        fields[key] = check_table(folder/"tables"/(key+".csv"), expected)
        artifact_paths += check_reference_paths(expected, folder)
    require([len(tables[k]) for k in ("SCORES", "CHECKPOINTS", "UPDATES", "WHOLE_COST", "WHOLE_COST_SAMPLES")] == [252, 80, 9600, 36, 324], "FULL_CORE_REPORT_COUNTS")
    paths = check_manifest_files(folder, manifest["files"]); png, svg = inspect_images(paths)
    require(png == svg == 6 and len(paths) == 27, "FULL_14CSV_12FIGURE_REPORT")
    require({p.resolve() for p in folder.rglob("*") if p.is_file()} == set(paths)|{manifest_path.resolve()}, "EXACT_REPORT_DIRECTORY_FILES")
    return {"status": "PASS", "report": record(report_path), "manifest": record(manifest_path), "source": manifest["source"], "inputs": {key: manifest["inputs"][key] for key in INPUTS},
        "table_rows_checked": {key: len(rows) for key, rows in tables.items()}, "CSV_primitive_fields_compared": fields,
        "files_checked": len(paths), "PNG_count": png, "SVG_count": svg, "local_links_checked": check_markdown(folder, tables), "relative_artifact_path_existence_checks": artifact_paths,
        "verification_scope": "All score/paired/channel/CP/update/probe/cost/sample source fields plus selected condition/geometry/provenance fields; not all nested provenance or all condition metadata",
        "scientific_array_model_training_F_timer_or_VJP_replay": False, "figure_visual_review_performed_by_this_script": False,
        "protected_TEST_accessed": False, "audit_source": record(__file__)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--report-dir", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try: result = verify(args.report_dir)
    except Exception as error: result = {"status": "FAILED_PRESERVED", "failure": {"type": type(error).__name__, "message": str(error)}, "audit_source": record(__file__)}
    with resolve(args.output).open("x", encoding="utf-8") as handle: json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")
    print(json.dumps({"status": result["status"], "output": str(resolve(args.output))}), flush=True)
    raise SystemExit(0 if result["status"] == "PASS" else 2)
