"""Independent JSON/JSONL report verification; no generator or scientific replay."""
from pathlib import Path
import argparse
import json
import re
from urllib.parse import unquote

import audit_linear_e_report_v1 as common
from audit_linear_e_report_v1 import require, resolve, record, parse, metadata, find, relative_values, check_table

SEEDS = (92601, 92602)
TABLES = ("SCORES", "PAIRED", "CHECKPOINTS", "UPDATES", "CONDITIONS", "PROBES", "SPACE_CELLS", "SPACE_METHODS", "SVD_CERTIFICATES", "RESOURCE")
INPUT_SHA = {"train": "4b1559216ea99937ca8e7e0f6cdac9885abc29479a6050d82336dcfa0305bb2b", "audit": "f79324dc9452192601ac16b483e4b19c296f3c252f5814062c7c05f7d4f9e85b"}
REPORT_SHA = "8c38be2568a1b06c2272a4fc231e1fdf63e41ff4e16373a22381f400c4cac29d"
MANIFEST_SHA = "ef507cdf9dbda5dccb0165ce2be4428b0fc847732f38b0cfb9abd850ae380f6d"
SOURCE_SHA = "99b8e6638637b8e5e27f1092d2567f832273f350e9dce79d4a894c6e033ff392"
COMMON_SHA = "c87fe5ada4de37e0f46fc61303ea0d18fb0bd9178d4ddc1771e92e6f26acf005"
FLAGS = {"trainable_parameter_names": ["E"], "D_requires_grad": False, "D_grad_is_None": True, "center_requires_grad": False}


def selected(row):
    require([cp["update"] for cp in row["checkpoints"]] == [0, 100, 200, 400, 600], "FIVE_CP")
    cp = min(row["checkpoints"], key=lambda p: (p["tune"]["mse"], p["update"]))
    require(cp["update"] == row["selected_update"] and cp["state"] == row["selected_state"], "TUNE_ONLY_EARLIEST_CP"); return cp


def expected_tables(values):
    train, joint, audit = (values[key] for key in ("train", "joint", "audit")); out = {key: [] for key in TABLES}
    def scores(base, item):
        for mask, field in (("all_observed", "scores_all_observed"), ("complete_C", "scores_complete_mask")):
            for scale in ("normalized", "physical_per_channel_diagnostic"): out["SCORES"].append({**base, "mask": mask, "scale": scale, **item[field][scale], "artifact": item.get("artifact")})
    for seed in SEEDS:
        new, old = (values[key]["datasets"]["robin"]["conditions"][str(seed)] for key in ("train", "joint")); require(new["order_sha256"] == old["order_sha256"], "SAME_PAIRED_MINIBATCH_ORDER")
        for method, row in (("E_ONLY", new), ("JOINT_E_D", old)):
            cp = selected(row); fresh = method == "E_ONLY"; base = {"method": method, "seed": seed, "new_execution": fresh}
            if fresh: require(row["D_trainable"] is False and row["fixed_decoder_final_status"] == FLAGS and row["D_frozen_after_optimizer_step_checks"] == 600 and row["D_and_center_checkpoint_byte_checks"] == 5 and row["D_bytes_checked_at_checkpoints_and_final_only"] is True and row["independent_per_update_D_bytes_replayed"] is False and all(p["fixed_decoder_status"] == FLAGS and p["D_and_center_exact_initial_bytes"] is True for p in row["checkpoints"]), "FIXED_D_RECORDS_CP_AND_FINAL")
            for split, item in (("fit", row["FIT"]), ("tune", cp), ("val", row["DEV"])): scores({**base, "split": split}, item)
            out["CHECKPOINTS"] += [{**base, **{key: item[key] for key in ("update", "tune", "state", "artifact", "scores_all_observed", "scores_complete_mask")}, "selected": item["update"] == cp["update"]} for item in row["checkpoints"]]
            logs = [parse(line) for line in metadata(row["updates_log"]).read_text(encoding="utf-8").splitlines()]; require([p["update"] for p in logs] == list(range(1, 601)) and sum(p["clipped"] for p in logs)/600 == row["global_clipping_fraction"], "SIX_HUNDRED_UPDATE_LOG_AND_CLIP_FRACTION")
            if fresh: require(all(p["D_gradient_norm_before_clip"] == 0. and p["gradient_clipping_parameter_scope"] == "E_ONLY" and all(p[key] == value for key, value in FLAGS.items()) for p in logs), "RECORDED_E_ONLY_FLAGS_ZERO_D_NORM")
            out["UPDATES"] += [{**base, **item} for item in logs]
            condition = {**base, **{key: row[key] for key in ("selected_update", "selected_state", "selection_lock", "order_sha256", "updates_log", "global_clipping_fraction")}, "gradient_clipping_parameter_scope": "E_ONLY" if fresh else "E_AND_D", "selected_TUNE_latent_variance": cp["TUNE_latent_variance"]}
            if fresh: condition.update({key: row[key] for key in ("D_frozen_after_optimizer_step_checks", "D_and_center_checkpoint_byte_checks", "fixed_decoder_final_status", "initial_D_value_sha256", "D_bytes_checked_at_checkpoints_and_final_only", "independent_per_update_D_bytes_replayed")})
            out["CONDITIONS"].append(condition)
        require(new["probe"]["status"] == "PASS" and new["probe"]["optimizer_updates"] == 0 and new["probe"]["component_byte_exact"] == {key: True for key in ("prediction", "latent_history", "latent_future", "lag_features")} and new["probe"]["fixed_decoder_status"] == FLAGS, "ZERO_UPDATE_CP0_PARITY_FLAGS"); out["PROBES"].append({"seed": seed, **new["probe"]})
    recon = values["science"]["datasets"]["robin"]["spaces"]["PCA"]["conditions"]["L7_RECON"]["splits"]
    for split in ("fit", "tune", "val"):
        for method, item in (("RECON_CP0", recon[split]), ("RAW_BOLT", recon[split]["RAW_BOLT"])): scores({"method": method, "seed": 0, "split": split, "new_execution": False}, item)
    for seed in SEEDS:
        for split in ("fit", "tune", "val"):
            for mask in ("all_observed", "complete_C"):
                for scale in ("normalized", "physical_per_channel_diagnostic"):
                    where = {"seed": seed, "split": split, "mask": mask, "scale": scale}; a, b = (find(out["SCORES"], **where, method=method) for method in ("E_ONLY", "JOINT_E_D"))
                    out["PAIRED"].append({**where, "E_ONLY_MSE": a["mse"], "JOINT_MSE": b["mse"], "MSE_E_ONLY_minus_JOINT": a["mse"]-b["mse"], "E_ONLY_MAE": a["mae"], "JOINT_MAE": b["mae"], "MAE_E_ONLY_minus_JOINT": a["mae"]-b["mae"]})
        for method, source in (("E_ONLY", audit), ("JOINT_E_D", values["joint_audit"])):
            for split in ("tune", "val"):
                cell = source["new_output_space_diagnostics"][str(seed)][split]; base = {"method": method, "seed": seed, "split": split, "computed_during_current_audit": method == "E_ONLY"}
                require(cell["status"] == "COMPLETE" and cell["independent_reduced_LS_vs_SVD_verified"] is True and cell["all_real_latent_or_all_FP32_lower_bound_claimed"] is cell["diagnostic_future_labels_used_for_new_fit_or_selection"] is False, "NEW_VERSUS_REUSED_NUMERIC_SPACE_SCOPE")
                e_sup = cell.get("selected_E_previously_supervised_by_FIT_future_targets", cell.get("selected_E_D_previously_supervised_by_FIT_future_targets")); d_sup = cell.get("selected_D_previously_supervised_by_FIT_future_targets", cell.get("selected_E_D_previously_supervised_by_FIT_future_targets"))
                if method == "E_ONLY": require(d_sup is False and e_sup == (train["datasets"]["robin"]["conditions"][str(seed)]["selected_update"] > 0), "SEPARATE_SELECTED_E_AND_D_SUPERVISION")
                row = {**base, **{key: cell[key] for key in ("minimum_normalized_MSE", "complete_target_positions", "target_positions", "joint_unknown_future_coordinates", "joint_output_coordinates_before_mask", "minimum_solution_scores_complete_C", "EY_encoding_backend", "EY_incomplete_C_horizons_filled_with_current_C", "artifact", "input_source", "selected_forecast_source", "current_C_persistence_scores_complete_mask")}, "selected_E_previously_supervised_by_FIT_future_targets": e_sup, "selected_D_previously_supervised_by_FIT_future_targets": d_sup}
                row.update({label: cell["methods"][name]["scores_complete_mask"]["normalized"]["mse"] for label, name in (("actual_complete_MSE", "ACTUAL_F"), ("EY_complete_MSE", "EY"), ("FLAT_complete_MSE", "FLAT"))}); row["RAW_complete_MSE"] = cell["RAW_BOLT_scores_complete_mask"]["normalized"]["mse"]; out["SPACE_CELLS"].append(row)
                out["SPACE_METHODS"] += [{**base, "future_method": name, **value} for name, value in cell["methods"].items()]
                keys = ("window_indices", "complete_horizons", "selected_output_rows", "unknown_future_coordinates_kept", "numeric_rank", "singularvalues", "normal_residual_max_abs", "normal_tolerance")
                out["SVD_CERTIFICATES"].append({**base, "certificates": [{key: cert[key] for key in keys} for cert in cell["independent_LS_certificates"]]})
    out["RESOURCE"] = [{"stage": "new E-only two-fit training" if key == "train" else "saved-training audit plus NEW selected-space computation", "new_deployment_timing": False, **values[key].get("resource_terminal" if key == "train" else "resource", {})} for key in ("train", "audit")]
    return out


def check_markdown(folder, tables):
    text = (folder/"REPORT_KO.md").read_text(encoding="utf-8"); links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text)
    for target in links:
        require(not re.match(r"^[A-Za-z]:|^/|^file:", target), "RELATIVE_MD_LINK")
        if not re.match(r"https?://|#", target): require((folder/unquote(target.split("#", 1)[0])).resolve().is_file(), "MD_LOCAL_LINK_EXISTS")
    def mse(method, seed, split): return find(tables["SCORES"], method=method, seed=seed, split=split, mask="all_observed", scale="normalized")["mse"]
    better = sum(mse("E_ONLY", s, "val") < mse("JOINT_E_D", s, "val") for s in SEEDS); worse = sum(mse("E_ONLY", s, "tune") > mse("JOINT_E_D", s, "tune") for s in SEEDS)
    require(f"JOINT_E_D보다 {better}/2 조건이 좋았지만 TUNE는 {worse}/2 조건에서 나빴다" in text and f"RECON CP0보다 {sum(mse('E_ONLY', s, 'val') > mse('RECON_CP0', 0, 'val') for s in SEEDS)}/2, Raw보다 {sum(mse('E_ONLY', s, 'val') > mse('RAW_BOLT', 0, 'val') for s in SEEDS)}/2 조건이 나빴다" in text, "MD_TUNE_DEV_OPPOSITE_ORDER_COUNTS")
    require(f"RECON CP0 TUNE {mse('RECON_CP0',0,'tune'):.10f}, DEV {mse('RECON_CP0',0,'val'):.10f}; Raw TUNE {mse('RAW_BOLT',0,'tune'):.10f}, DEV {mse('RAW_BOLT',0,'val'):.10f}." in text, "MD_ORIGINAL_BASELINE_NUMBERS")
    for seed in SEEDS:
        a, b = (find(tables["CONDITIONS"], method=m, seed=seed) for m in ("E_ONLY", "JOINT_E_D"))
        require(f"seed {seed}: 새 E_ONLY CP{a['selected_update']}, TUNE {mse('E_ONLY',seed,'tune'):.10f}, DEV {mse('E_ONLY',seed,'val'):.10f}; 재사용 JOINT_E_D CP{b['selected_update']}, TUNE {mse('JOINT_E_D',seed,'tune'):.10f}, DEV {mse('JOINT_E_D',seed,'val'):.10f}." in text, "MD_TWO_SEED_CP_AND_SCORE_ROWS")
    for row in tables["SPACE_CELLS"]:
        require(f"{row['method']} / {row['seed']} / {row['split']}: actual {row['actual_complete_MSE']:.10f}, E(Y) {row['EY_complete_MSE']:.10f}, minimum {row['minimum_normalized_MSE']:.10f}, Raw {row['RAW_complete_MSE']:.10f}; complete {row['complete_target_positions']}/{row['target_positions']}." in text, "MD_EIGHT_SPACE_ROWS")
    cells = [r for r in tables["SPACE_CELLS"] if r["method"] == "E_ONLY"]
    require(f"TUNE {sum(r['minimum_normalized_MSE'] > r['RAW_complete_MSE'] for r in cells if r['split']=='tune')}/2 조건에서 Raw보다 높고, DEV {sum(r['minimum_normalized_MSE'] < r['RAW_complete_MSE'] for r in cells if r['split']=='val')}/2 조건에서 Raw보다 낮았다" in text, "MD_CONDITIONAL_NUMERIC_SPACE_COUNTS")
    for key, rows in tables.items(): require(f"{key}.csv](tables/{key}.csv): {len(rows)}행" in text, "MD_CSV_COUNTS")
    for token in ("새 2fit 1,200 update·10CP", "2fit 1,200 update·10CP 재사용", "단일 인과효과로 부르지 않는다", "매 update bytes를 독립 재생했다는 뜻은 아니다", "학습 감사 중 새 계산", "세 번째 외부 재현은 아니다", "MAE·물리 단위 값은 이 MSE 해의 점수", "false-bound flags", "strict K-only도 아니다", "새 deployment 비용은 현재 미측정", "독립 일반화 검증이 아니다"):
        require(token in text, "MD_SCIENTIFIC_SCOPE:"+token)
    return len(links)


def verify(folder):
    require(all(isinstance(v, str) and len(v) == 64 for v in (REPORT_SHA, MANIFEST_SHA, SOURCE_SHA, *INPUT_SHA.values())), "ACTUAL_REPORT_NOT_YET_REGISTERED"); require(record(common.__file__)["sha256"] == COMMON_SHA, "IMMUTABLE_METADATA_HELPERS")
    folder = resolve(folder); report_path, manifest_path = folder/"REPORT_KO.md", folder/"MANIFEST.json"; require(record(report_path)["sha256"] == REPORT_SHA and record(manifest_path)["sha256"] == MANIFEST_SHA, "EXACT_ACTUAL_REPORT_MANIFEST")
    manifest = parse(manifest_path.read_text(encoding="utf-8")); require(manifest["status"] == "COMPLETE" and manifest["source"]["sha256"] == SOURCE_SHA and Path(manifest["source"]["path"]).name == "make_fixed_recon_d_e_adaptation_report_v1.py", "EXACT_EXECUTED_REPORT_SOURCE"); metadata(manifest["source"])
    values = {}
    for key, sha in INPUT_SHA.items():
        binding = manifest["inputs"][key]; require(binding["sha256"] == sha, "EXACT_INPUT_SHA"); values[key] = parse(metadata(binding).read_text(encoding="utf-8-sig")); require(values[key]["status"] == ("PASS" if key == "audit" else "COMPLETE") and values[key]["protected_TEST_accessed"] is False, "COMPLETE_PASS_NO_TEST")
    require(all(values["audit"]["diagnostic"][key] == manifest["inputs"]["train"][key] for key in ("path", "sha256")), "EXACT_SCIENTIFIC_AUDIT_CHAIN")
    for key, source in (("joint", "joint_recon_diagnostic"), ("joint_audit", "joint_recon_audit"), ("science", "causal_lag_diagnostic")): values[key] = parse(metadata(values["train"][source]).read_text(encoding="utf-8-sig"))
    require(all(manifest[k] == v for k, v in (("scientific_NPZ_or_model_read", False), ("new_updates_during_generation", 0), ("new_deployment_cost_measured", False), ("new_E_only_training_updates", 1200), ("reused_joint_training_updates", 1200), ("independent_per_update_D_bytes_replayed", False), ("protected_TEST_accessed", False))), "REPORT_NEW_REUSED_AND_DIAGNOSTIC_SCOPE")
    tables = expected_tables(values); fields, refs = {}, 0
    for key in TABLES:
        require(manifest["table_row_counts"][key] == len(tables[key]), "MANIFEST_TABLE_COUNTS"); expected = relative_values(tables[key], folder); fields[key] = check_table(folder/"tables"/(key+".csv"), expected); refs += common.check_reference_paths(expected, folder)
    require([len(tables[k]) for k in TABLES] == [72, 24, 20, 2400, 4, 2, 8, 24, 8, 2] and sum(r["new_execution"] for r in tables["UPDATES"]) == 1200 and sum(r["new_execution"] for r in tables["CHECKPOINTS"]) == 10, "EXACT_TWO_NEW_TWO_REUSED_CORE_COUNTS")
    paths = common.check_manifest_files(folder, manifest["files"]); png, svg = common.inspect_images(paths); require(len(paths) == 17 and png == svg == 3 and {p.resolve() for p in folder.rglob("*") if p.is_file()} == set(paths)|{manifest_path.resolve()}, "TEN_CSV_SIX_IMAGES_EXACT_FILES")
    return {"status": "PASS", "report": record(report_path), "manifest": record(manifest_path), "source": manifest["source"], "inputs": {key: manifest["inputs"][key] for key in INPUT_SHA}, "table_rows_checked": {key: len(rows) for key, rows in tables.items()}, "CSV_primitive_fields_compared": fields, "files_checked": len(paths), "PNG_count": png, "SVG_count": svg, "local_links_checked": check_markdown(folder, tables), "relative_artifact_path_existence_checks": refs, "verification_scope": "All primary score/mask/per-channel and paired fields; CP scores/selection/refs, logs/recorded fixed-D flags, probe bytes metadata, selected space scores/method traces/supervision, selected SVD certificate fields and resource values. Not all nested CP geometry or joint/SVD metadata; no scientific arrays, F/backward/optimizer/per-step-D-byte replay or visual review.", "scientific_array_model_F_or_training_replayed": False, "figure_visual_review_performed_by_this_script": False, "protected_TEST_accessed": False, "audit_source": record(__file__)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--report-dir", required=True); parser.add_argument("--output", required=True); args = parser.parse_args()
    try: result = verify(args.report_dir)
    except Exception as error: result = {"status": "FAILED_PRESERVED", "failure": {"type": type(error).__name__, "message": str(error)}, "audit_source": record(__file__)}
    with resolve(args.output).open("x", encoding="utf-8") as handle: json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")
    print(json.dumps({"status": result["status"], "output": str(resolve(args.output))}), flush=True); raise SystemExit(0 if result["status"] == "PASS" else 2)
