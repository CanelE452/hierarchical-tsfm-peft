"""JSON/JSONL and published-file checks only; no generator/scientific replay."""
from pathlib import Path
import argparse
import json
import math
import re
import statistics
from urllib.parse import unquote

import audit_linear_e_report_v1 as common
from audit_linear_e_report_v1 import require, resolve, record, parse, metadata, find, relative_values, check_table

HERE = Path(__file__).resolve().parent
DATASETS, SEEDS = {"robin": (17, 4, "val"), "beijing_aotizhongxin": (11, 2, "eval")}, (92601, 92602)
INPUTS = ("train", "audit", "cost", "cost_audit")
TABLES = ("SCORES", "PAIRED", "CHECKPOINTS", "UPDATES", "CLIP_SUMMARY", "PROBES", "SELECTION", "WHOLE_COST", "WHOLE_COST_SAMPLES", "RESOURCES", "PROVENANCE")
SCALES = ("normalized", "physical_per_channel_diagnostic")
REPORT_SHA = "f8eb895004fbcd1036616b9378cd36352eca38bc4aa306cd9ad0eab385ec23df"
MANIFEST_SHA = "f8984db0a81c99b871201376943a869aa9f670a9de570c38f085f67b0c929a69"
REPORT_SOURCE_SHA = "db79167b70b69db0b76e5c63e8cf679cb2be91f84cb9577f24cf3c22263e0cbb"
INPUT_SHA = {"train": "e27989fb23716be8bbf507442d44033884b28522d641a010d18b426a0eb6d10a", "audit": "64f19658ac8bc8cee1d6fa3713113a7146ba282ebb8ad2b9eb49498d91e6418f", "cost": "354cbeb47e298c1f313c08c52a61abeb19020396dfa1980e4c12b41647e2c9de", "cost_audit": "0a70e0a1be13044846bca8bbb1414794857ee7e6eb6f76f43d679c4070f835dc"}
COMMON_SHA = "c87fe5ada4de37e0f46fc61303ea0d18fb0bd9178d4ddc1771e92e6f26acf005"
PARENT_SHA = "3f0cbd9176f2e7882778452e537564aab7445e1b88c1534fd48b1bce7ef639a4"


def winner(row):
    require([cp["update"] for cp in row["checkpoints"]] == [0, 100, 200, 400, 600], "FIVE_CP")
    cp = min(row["checkpoints"], key=lambda value: (value["tune"]["mse"], value["update"]))
    require(cp["update"] == row["selected_update"] and cp["state"] == row["selected_state"], "EARLIEST_TUNE_ONLY_CP")
    return cp


def expected_tables(values):
    train, parent, cost = (values[key] for key in ("train", "parent", "cost")); out = {key: [] for key in TABLES}
    def scores(base, item):
        for mask, field in (("all_observed", "scores_all_observed"), ("complete_C", "scores_complete_mask")):
            for scale in SCALES:
                if scale in item[field]: out["SCORES"].append({**base, "mask": mask, "scale": scale, **item[field][scale], "artifact": item.get("artifact")})
    for data, (_, _, dev) in DATASETS.items():
        for seed in SEEDS:
            full = parent["datasets"][data]["conditions"][f"{data}_L0_E_LEARN_s{seed}"]; stop = train["datasets"][data]["conditions"][str(seed)]
            require(full["order_sha256"] == stop["order_sha256"], "SAME_PAIRED_BATCH_ORDER")
            fixed = parent["datasets"][data]["conditions"][f"{data}_L0_E_FIXED_s{seed}"]
            for method, row in (("FULL_GRADIENT", full), ("STOP_GRADIENT", stop), ("E_FIXED", fixed)):
                cp = winner(row); base = {"dataset": data, "seed": seed, "method": method, "new_execution": method == "STOP_GRADIENT"}
                for split, entry in (("fit", row["FIT"]), ("tune", cp), (dev, row["DEV"])): scores({**base, "split": split}, entry)
                if method == "E_FIXED": continue
                out["CHECKPOINTS"] += [{**base, **{key: item[key] for key in ("update", "tune", "state", "scores_all_observed", "scores_complete_mask", "artifact")}, "selected": item["update"] == row["selected_update"]} for item in row["checkpoints"]]
                out["SELECTION"].append({**base, **{key: row[key] for key in ("selected_update", "selected_state", "selection_lock", "order_sha256", "updates_log")}})
                updates = [parse(line) for line in metadata(row["updates_log"]).read_text(encoding="utf-8").splitlines()]
                require([r["update"] for r in updates] == list(range(1, 601)), "SIX_HUNDRED_LOGGED_UPDATES")
                fields = ("stage", "update", "loss", "E_gradient_norm_before_clip", "D_gradient_norm_before_clip", "global_gradient_norm_before_clip", "clipped")
                rows = []
                for item in updates:
                    require(item["stage"] == "TRAINING_UPDATE" and all(math.isfinite(item[key]) and item[key] >= 0 for key in fields[2:6]) and item["clipped"] == (item["global_gradient_norm_before_clip"] > 1.), "FINITE_LOGGED_LOSS_NORMS_CLIP")
                    scale = min(1., 1. / (item["global_gradient_norm_before_clip"] + 1e-6))
                    rows.append({**base, **{key: item[key] for key in fields}, "global_clip_scale_from_logged_norm": scale, "D_gradient_norm_after_clip_formula": scale * item["D_gradient_norm_before_clip"], "D_gradient_scaling_is_AdamW_step_norm": False})
                fraction = sum(r["clipped"] for r in rows) / 600; require(fraction == row["global_clipping_fraction"], "CLIP_LOG_FRACTION")
                out["UPDATES"] += rows
                out["CLIP_SUMMARY"].append({**base, "clipping_fraction": fraction, "mean_reconstructed_clip_scale": statistics.mean(r["global_clip_scale_from_logged_norm"] for r in rows), "mean_D_preclip_norm": statistics.mean(r["D_gradient_norm_before_clip"] for r in rows), "mean_D_afterclip_formula_norm": statistics.mean(r["D_gradient_norm_after_clip_formula"] for r in rows), "actual_per_step_AdamW_parameter_update_norm_measured": False})
            p = stop["probe"]; require(p["status"] == "PASS" and p["optimizer_updates"] == 0 and p["forward_byte_exact"] is p["finite_gradient_vectors"] is True and all(v["pass"] for v in p["gradient_checks"].values()), "FOUR_ZERO_UPDATE_PROBES")
            out["PROBES"].append({"dataset": data, "seed": seed, **{key: p[key] for key in ("status", "optimizer_updates", "forward_byte_exact", "finite_gradient_vectors", "gradient_checks", "gradient_norms", "artifact", "initial_state_sha256")}, "CP0_TUNE_byte_checks": stop["CP0_TUNE_byte_checks"], "CP0_TUNE_parity_artifact": stop["CP0_TUNE_parity_artifact"]})
        baseline = parent["CP0_LS_and_RAW_baselines"][data]["L0"]
        for split in ("fit", "tune", dev):
            initial = full["CP0_TUNE"] if split == "tune" else baseline[split]["CP0_LS"]; raw = full["RAW_TUNE_scores"] if split == "tune" else baseline[split]["RAW_scores"]
            scores({"dataset": data, "seed": 0, "method": "CP0_LS", "split": split, "new_execution": False}, initial)
            out["SCORES"] += [{"dataset": data, "seed": 0, "method": "RAW_BOLT", "split": split, "mask": "all_observed", "scale": scale, "new_execution": False, **raw[scale]} for scale in SCALES]
        for seed in SEEDS:
            for split in ("fit", "tune", dev):
                for mask in ("all_observed", "complete_C"):
                    for scale in SCALES:
                        where = {"dataset": data, "seed": seed, "split": split, "mask": mask, "scale": scale}; a, b = (find(out["SCORES"], **where, method=m) for m in ("FULL_GRADIENT", "STOP_GRADIENT"))
                        out["PAIRED"].append({**where, "full_MSE": a["mse"], "stop_MSE": b["mse"], "MSE_stop_minus_full": b["mse"] - a["mse"], "full_MAE": a["mae"], "stop_MAE": b["mae"], "MAE_stop_minus_full": b["mae"] - a["mae"]})
        rows = cost["cost"][data]["rows"]
        for row in rows:
            raw = find(rows, route="RAW_BOLT", batch=row["batch"]); sec = row["seconds"]; ratios = [a/b for a, b in zip(sec, raw["seconds"])]
            require(len(sec) == len(row["samples"]) == 5 and statistics.median(sec) == row["median_seconds"] and row["median_seconds_per_window"] == row["median_seconds"]/row["batch"] and ratios == row["paired_RAW_ratios"] and statistics.median(ratios) == row["paired_RAW_median_ratio"], "COST_FIVE_SAMPLES_MEDIANS_PAIRED_RATIO")
            out["WHOLE_COST"].append({"dataset": data, **row, "ratio_of_RAW_medians": row["median_seconds"]/raw["median_seconds"], "median_paired_RAW_ratio": statistics.median(ratios)})
            for index, sample in enumerate(row["samples"]):
                require(sample["seconds"] == sec[index] == (sample["perf_end_ns"] - sample["perf_start_ns"])/1e9, "TIMER_INTEGER_NS")
                out["WHOLE_COST_SAMPLES"].append({"dataset": data, "route": row["route"], "batch": row["batch"], **sample, "paired_RAW_ratio": ratios[index]})
    out["RESOURCES"] = [{"stage": key, **values[key].get("resource_terminal", {})} for key in ("train", "cost")]
    out["PROVENANCE"] = [{"stage": key, "status": values[key]["status"], "totals": values[key].get("totals"), "failed_paths": values[key].get("failed_paths", [])} for key in INPUTS]
    return out


def check_markdown(folder, tables):
    text = (folder / "REPORT_KO.md").read_text(encoding="utf-8"); links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text)
    for target in links:
        require(not re.match(r"^[A-Za-z]:|^/|^file:", target), "RELATIVE_MD_LINK")
        if not re.match(r"https?://|#", target): require((folder / unquote(target.split("#", 1)[0])).resolve().is_file(), "MD_LOCAL_LINK_EXISTS")
    primary = [r for r in tables["PAIRED"] if r["mask"] == "all_observed" and r["scale"] == "normalized"]
    for label, splits in (("TUNE", ("tune",)), ("기노출 DEV", ("val", "eval"))):
        require(f"{label} {sum(r['MSE_stop_minus_full'] > 0 for r in primary if r['split'] in splits)}/4" in text, "MD_PAIRED_COUNTS")
    better = sum(row["mse"] < find(tables["SCORES"], dataset=row["dataset"], method="RAW_BOLT", split=row["split"], mask="all_observed", scale="normalized")["mse"] for row in tables["SCORES"] if row["method"] in ("FULL_GRADIENT", "STOP_GRADIENT") and row["split"] in ("val", "eval") and row["mask"] == "all_observed" and row["scale"] == "normalized")
    require(f"Raw보다 DEV MSE가 작은 조건은 {better}/8" in text, "MD_RAW_COUNT")
    paired_rows = cost_rows = 0
    for line in text.splitlines():
        words = line.split()
        if not words or words[0] not in ("Robin", "Beijing"): continue
        data = "robin" if words[0] == "Robin" else "beijing_aotizhongxin"
        if len(words) == 8:
            seed = int(words[1]); tune = find(primary, dataset=data, seed=seed, split="tune"); dev = find(primary, dataset=data, seed=seed, split=DATASETS[data][2])
            require(words[2:4] == [str(find(tables["SELECTION"], dataset=data, seed=seed, method=m)["selected_update"]) for m in ("FULL_GRADIENT", "STOP_GRADIENT")], "MD_PAIRED_CP")
            require(words[4:] == [f"{v:.9f}" for v in (tune["full_MSE"], tune["stop_MSE"], dev["full_MSE"], dev["stop_MSE"])], "MD_PAIRED_NUMBERS"); paired_rows += 1
        elif len(words) == 5:
            row = find(tables["WHOLE_COST"], dataset=data, route=words[1], batch=8)
            require(words[2:] == [f"{v:.6f}" for v in (row["median_seconds_per_window"]*1000, row["ratio_of_RAW_medians"], row["median_paired_RAW_ratio"])], "MD_WHOLE_COST_NUMBERS"); cost_rows += 1
    require(paired_rows == 4 and cost_rows == 10, "FOUR_PAIRED_TEN_COST_MD_ROWS")
    for name, rows in tables.items(): require(f"{name}.csv](tables/{name}.csv): {len(rows)}행" in text, "MD_CSV_COUNTS")
    for token in ("F 예측값은 제거하지 않았다", "strict K-only 압축이 아니다", "독립 초기화나 독립 표본이 아니다", "실제 F/VJP/backward/optimizer를 재실행한 검사는 아니다", "실제 per-step AdamW parameter update norm을 측정한 것은 아니다", "새 학습은 4×600=2,400 update, 20 CP", "2,400 update·20 CP 로그", "probe는 optimizer update 0회", "독립 일반화", "총20조건·100표본"):
        require(token in text, "MD_SCOPE_AND_EXECUTION:" + token)
    return len(links)


def verify(folder):
    require(all(isinstance(value, str) and len(value) == 64 for value in (REPORT_SHA, MANIFEST_SHA, REPORT_SOURCE_SHA, *INPUT_SHA.values())), "ACTUAL_REPORT_COST_NOT_YET_REGISTERED")
    require(record(common.__file__)["sha256"] == COMMON_SHA, "IMMUTABLE_METADATA_CSV_HELPERS")
    folder = resolve(folder); report_path, manifest_path = folder / "REPORT_KO.md", folder / "MANIFEST.json"
    require(record(report_path)["sha256"] == REPORT_SHA and record(manifest_path)["sha256"] == MANIFEST_SHA, "EXACT_ACTUAL_REPORT_MANIFEST")
    manifest = parse(manifest_path.read_text(encoding="utf-8")); require(manifest["status"] == "COMPLETE" and manifest["source"]["sha256"] == REPORT_SOURCE_SHA and Path(manifest["source"]["path"]).name == "make_f_input_gradient_report_v1.py", "ACTUAL_REPORT_SOURCE")
    metadata(manifest["source"]); values = {}
    for key in INPUTS:
        binding = manifest["inputs"][key]; require(binding["sha256"] == INPUT_SHA[key], "EXACT_INPUT_SHA:" + key)
        values[key] = parse(metadata(binding).read_text(encoding="utf-8-sig")); require(values[key]["status"] == ("PASS" if "audit" in key else "COMPLETE") and values[key]["protected_TEST_accessed"] is False, "COMPLETE_PASS_INPUTS_NO_TEST")
    for value, field, wanted in ((values["audit"], "diagnostic", "train"), (values["cost_audit"], "cost", "cost"), (values["cost"], "stop_gradient_diagnostic", "train"), (values["cost"], "stop_gradient_audit", "audit")):
        require(all(value[field][key] == manifest["inputs"][wanted][key] for key in ("path", "sha256")), "EXACT_SCIENCE_COST_AUDIT_CHAIN")
    parent = values["train"]["full_gradient_parent"]; require(parent["sha256"] == PARENT_SHA, "EXACT_REUSED_PARENT")
    values["parent"] = parse(metadata(parent).read_text(encoding="utf-8-sig"))
    registry = parse(metadata(values["train"]["selection_registry"]).read_text(encoding="utf-8"))
    require(registry["status"] == "ALL_FOUR_RESOLVED_BEFORE_DEV" and registry["DEV_opened_before_seal"] is False and len(registry["conditions"]) == 4, "FOUR_SELECTION_LOCKS_BEFORE_DEV")
    for item in registry["conditions"]:
        lock = parse(metadata(item["selection_lock"]).read_text(encoding="utf-8")); require(lock["status"] == "COMPLETE" and lock["DEV_opened_before_lock"] is False, "COMPLETE_SEALED_SELECTION_LOCK")
    require(all(manifest[key] == value for key, value in (("new_updates", 2400), ("reused_full_gradient_updates", 2400), ("probe_optimizer_updates", 0), ("scientific_NPZ_or_model_read", False), ("new_scientific_execution", False), ("protected_TEST_accessed", False))), "REPORT_SCOPE_AND_REUSE_COUNTS")
    tables = expected_tables(values); fields, refs = {}, 0
    for key in TABLES:
        require(manifest["table_row_counts"][key] == len(tables[key]), "MANIFEST_TABLE_COUNT")
        expected = relative_values(tables[key], folder); fields[key] = check_table(folder / "tables" / (key + ".csv"), expected); refs += common.check_reference_paths(expected, folder)
    require([len(tables[k]) for k in TABLES] == [180, 48, 40, 4800, 8, 4, 8, 20, 100, 2, 4], "EXACT_PRIMARY_REPORT_COUNTS")
    require(sum(r["new_execution"] for r in tables["UPDATES"]) == 2400 and sum(r["new_execution"] for r in tables["CHECKPOINTS"]) == 20 and sum(r["new_execution"] for r in tables["SELECTION"]) == 4, "NEW_VERSUS_REUSED_UPDATES_CP_SELECTION")
    paths = common.check_manifest_files(folder, manifest["files"]); png, svg = common.inspect_images(paths)
    require(png == svg == 4 and len(paths) == 20 and {p.resolve() for p in folder.rglob("*") if p.is_file()} == set(paths) | {manifest_path.resolve()}, "ELEVEN_CSV_EIGHT_FIGURES_EXACT_FILES")
    return {"status": "PASS", "report": record(report_path), "manifest": record(manifest_path), "source": manifest["source"], "inputs": {key: manifest["inputs"][key] for key in INPUTS}, "table_rows_checked": {key: len(rows) for key, rows in tables.items()}, "CSV_primitive_fields_compared": fields, "files_checked": len(paths), "PNG_count": png, "SVG_count": svg, "local_links_checked": check_markdown(folder, tables), "relative_artifact_path_existence_checks": refs, "verification_scope": "All primary/paired/per-channel score fields; CP scores/selection/refs; logged loss/norm/clip scalars; recorded probe vectors metadata, cost/ns/two ratios, resources and selected provenance. Not all nested CP geometry, probe or provenance metadata; no scientific-array/F/VJP/optimizer/timer replay or visual review.", "scientific_array_model_training_F_timer_or_VJP_replay": False, "figure_visual_review_performed_by_this_script": False, "protected_TEST_accessed": False, "audit_source": record(__file__)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--report-dir", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try: result = verify(args.report_dir)
    except Exception as error: result = {"status": "FAILED_PRESERVED", "failure": {"type": type(error).__name__, "message": str(error)}, "audit_source": record(__file__)}
    with resolve(args.output).open("x", encoding="utf-8") as handle: json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")
    print(json.dumps({"status": result["status"], "output": str(resolve(args.output))}), flush=True)
    raise SystemExit(0 if result["status"] == "PASS" else 2)
