"""Independent JSON-to-report numeric checks; no NPZ, generator or science replay."""
from pathlib import Path
import argparse
import json
import re
from urllib.parse import unquote

import audit_linear_e_report_v1 as common
from audit_linear_e_report_v1 import require, resolve, record, parse, metadata, find, relative_values, check_table

DATASETS = {"robin": (17, 4, "val"), "beijing_aotizhongxin": (11, 2, "eval")}
TABLES = ("CELLS", "SCORES", "TRACES", "SVD_GROUPS", "MATCHED_L0_FULL_STOP", "FAILURES", "RESOURCE")
INPUT_SHA = {"diagnostic": "cf042cc764ff36cbb877211777721783efadbaab6ba8741268eaccba26cbb34b", "audit": "1b0dbacb711becce70ac877825cd53b84069918ff62ac79dc9b9b5d8ac410738"}
REPORT_SHA = "a057dc5f202c146156044f7e4aea5987c93ea2c0b44e5ea3161b19aab0ea2e0c"
MANIFEST_SHA = "d231904818058a0c8cc3b7ed8e62b6264ab8f6ce28fa7c6f8ae4ed7ed068ee5e"
REPORT_SOURCE_SHA = "c086fd012e022729595a7b1fd869a11b2cad5aa989cfb04ed75b3160f671adeb"
COMMON_SHA = "c87fe5ada4de37e0f46fc61303ea0d18fb0bd9178d4ddc1771e92e6f26acf005"


def reference(value): return {key: value[key] for key in ("path", "sha256", "bytes") if key in value}


def expected_tables(result):
    tables = {key: [] for key in TABLES}
    def scores(base, value):
        if value.get("normalized") is not None:
            for scale in ("normalized", "physical_per_channel_diagnostic"): tables["SCORES"].append({**base, "scale": scale, **value[scale]})
    for data, (_, _, dev) in DATASETS.items():
        for condition, owner in result["datasets"][data]["conditions"].items():
            for split in ("tune", dev):
                base = {"dataset": data, "condition": condition, "split": split, **{key: owner[key] for key in ("origin", "seed", "selected_update")}}; cell = owner["splits"][split]
                if cell["status"] != "COMPLETE":
                    tables["FAILURES"].append({**base, "status": cell["status"], "failure": cell.get("failure"), "original_state": reference(owner["selected_state"])}); continue
                ranks = cell["joint_metadata"]["numeric_rank_per_origin"]
                require(cell["joint_metadata"]["rcond"] == 1e-15 and cell["RAW_bound_applied"] is cell["all_real_latent_or_all_FP32_lower_bound_claimed"] is False, "RETAINED_NUMERIC_COMPLETE_C_SCOPE")
                row = {**base, **{key: cell[key] for key in ("status", "complete_target_positions", "target_positions", "lag_slope_parameters_per_C", "minimum_normalized_MSE", "EY_encoding_backend", "selected_E_D_previously_supervised_by_FIT_future_targets")}, "HK_future_unknowns": cell["joint_unknown_future_coordinates"], "HC_outputs_before_mask": cell["joint_output_coordinates_before_mask"], "rcond": 1e-15, "numeric_rank_per_origin": ranks, "numeric_rank_min": min(ranks), "numeric_rank_max": max(ranks), "minimum_space_definition": cell["joint_metadata"]["minimum_space_definition"], "EY_filled_incomplete_C_horizons": cell["EY_incomplete_C_horizons_filled_with_current_C"], "EY_fill_can_affect_later_L7": cell["EY_fill_can_affect_later_L7_complete_outputs"]}
                row.update({label: cell["methods"][method]["scores_complete_mask"]["normalized"]["mse"] for label, method in (("actual_complete_MSE", "ACTUAL_F"), ("EY_complete_MSE", "EY"), ("FLAT_complete_MSE", "FLAT"))})
                row.update({label: cell[key]["normalized"]["mse"] for label, key in (("RAW_complete_MSE", "RAW_BOLT_scores_complete_mask"), ("RAW_all_observed_MSE", "RAW_BOLT_scores_all_observed"), ("current_C_persistence_complete_MSE", "current_C_persistence_scores_complete_mask"))})
                row.update({key: reference(value) for key, value in (("selected_state", owner["selected_state"]), ("selection_lock", owner["selection_lock"]), ("input_source", cell["input_source"]), ("selected_forecast_source", cell["selected_forecast_source"]), ("artifact", cell["artifact"]))}); tables["CELLS"].append(row)
                for method, value in cell["methods"].items():
                    common_base = {**base, "method": method, "mask": "complete_C", "role": value["role"]}
                    for phase, key in (("FP32", "scores_complete_mask"), ("REAL_TRACE", "real_trace_scores_complete_C"), ("RETAINED_TRACE", "retained_output_real_trace_scores_complete_C")): scores({**common_base, "phase": phase}, value[key])
                    if method == "ACTUAL_F": scores({**common_base, "phase": "FP32", "mask": "all_observed"}, value["scores_all_observed"])
                    tables["TRACES"].append({**base, "method": method, **{key: item for key, item in value.items() if isinstance(item, (str, int, float, bool)) or item is None}})
                for method, key, mask in (("RAW_BOLT", "RAW_BOLT_scores_complete_mask", "complete_C"), ("RAW_BOLT", "RAW_BOLT_scores_all_observed", "all_observed"), ("CURRENT_C_PERSISTENCE", "current_C_persistence_scores_complete_mask", "complete_C"), ("MINIMUM_SOLUTION", "minimum_solution_scores_complete_C", "complete_C")):
                    role = "Normalized MSE numeric-space solution; MAE/physical are values of this solution, not separate minima" if method == "MINIMUM_SOLUTION" else "same-original-mask baseline"
                    scores({**base, "method": method, "phase": "REAL_MINIMUM_SOLUTION" if method == "MINIMUM_SOLUTION" else "FP32", "mask": mask, "role": role}, cell[key])
                for index, cert in enumerate(cell["independent_LS_certificates"]):
                    rank, s = cert["numeric_rank"], cert["singularvalues"]
                    tables["SVD_GROUPS"].append({**base, "mask_group": index, **{key: cert[key] for key in ("window_indices", "complete_horizons", "selected_output_rows", "unknown_future_coordinates_kept", "numeric_rank", "normal_residual_max_abs", "normal_tolerance")}, "largest_singular": s[0] if s else None, "last_retained_singular": s[rank-1] if rank else None, "first_discarded_singular": s[rank] if rank < len(s) else None})
        for seed in (92601, 92602):
            for split in ("tune", dev):
                a = [r for r in tables["CELLS"] if (r["dataset"], r["condition"], r["split"]) == (data, f"FULL_L0_s{seed}", split)]; b = [r for r in tables["CELLS"] if (r["dataset"], r["condition"], r["split"]) == (data, f"STOP_L0_s{seed}", split)]
                tables["MATCHED_L0_FULL_STOP"].append({"dataset": data, "seed": seed, "split": split, "status": "COMPLETE" if a and b else "UNAVAILABLE_PARTIAL", **{key+"_stop_minus_full": b[0][key]-a[0][key] if a and b else None for key in ("minimum_normalized_MSE", "actual_complete_MSE", "EY_complete_MSE")}, "same_learned_E_D_claimed": False})
    require(len(tables["CELLS"]) == result["completed_diagnostic_cells"], "ACTUAL_CELL_COUNTS")
    tables["RESOURCE"] = [{"stage": "saved-output diagnosis; not deployment timing", **result.get("resource_terminal", {})}]
    return tables


def check_markdown(folder, tables):
    text = (folder / "REPORT_KO.md").read_text(encoding="utf-8"); links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text)
    for target in links:
        require(not re.match(r"^[A-Za-z]:|^/|^file:", target), "RELATIVE_MD_LINK")
        if not re.match(r"https?://|#", target): require((folder / unquote(target.split("#", 1)[0])).resolve().is_file(), "EXISTING_MD_LINK")
    seen = set()
    for line in text.splitlines():
        words = line.split()
        if len(words) != 8 or words[0] not in ("Robin", "Beijing"): continue
        data = "robin" if words[0] == "Robin" else "beijing_aotizhongxin"; key = data, words[1], words[2]
        row = find(tables["CELLS"], dataset=data, split=words[1], condition=words[2]); require(key not in seen, "UNIQUE_MD_CELL"); seen.add(key)
        require(words[3:] == [f"{row[k]:.9f}" for k in ("minimum_normalized_MSE", "actual_complete_MSE", "EY_complete_MSE", "RAW_complete_MSE", "RAW_all_observed_MSE")], "MD_TWENTY_FOUR_CELL_NUMBERS")
    require(len(seen) == len(tables["CELLS"]) == 24, "ALL_24_MD_NUMERIC_ROWS")
    for data, (_, _, dev) in DATASETS.items():
        label = "Robin" if data == "robin" else "Beijing"; rows = [r for r in tables["CELLS"] if r["dataset"] == data and r["split"] == dev]
        above = sum(r["minimum_normalized_MSE"] > r["RAW_complete_MSE"] for r in rows)
        if data == "robin":
            tune = [r for r in tables["CELLS"] if r["dataset"] == data and r["split"] == "tune"]
            require(f"minimum>Raw가 TUNE {sum(r['minimum_normalized_MSE'] > r['RAW_complete_MSE'] for r in tune)}/{len(tune)} → 기노출 DEV {above}/{len(rows)}" in text, "MD_ROBIN_TUNE_DEV_CONDITIONAL_COUNTS")
        else: require(f"minimum>Raw {above}/{len(rows)}, E(Y)<Raw {sum(r['EY_complete_MSE'] < r['RAW_complete_MSE'] for r in rows)}/{len(rows)}, actual F>Raw {sum(r['actual_complete_MSE'] > r['RAW_complete_MSE'] for r in rows)}/{len(rows)}" in text, "MD_BEIJING_SPACE_EY_ACTUAL_COUNTS")
        for split in ("tune", dev):
            group = [r for r in tables["CELLS"] if r["dataset"] == data and r["split"] == split]
            require(len({(r["RAW_complete_MSE"], r["RAW_all_observed_MSE"], r["input_source"]["sha256"]) for r in group}) == 1, "FOUR_UNIQUE_RAW_BASELINES_REUSED")
            r = find(group, condition="FULL_L0_s92601")
            require(f"대표 FULL_L0/seed92601 {split}: complete-C normalized MSE actual F={r['actual_complete_MSE']:.9f}, E(Y)={r['EY_complete_MSE']:.9f}, minimum={r['minimum_normalized_MSE']:.9f}, Raw={r['RAW_complete_MSE']:.9f}." in text, "MD_REPRESENTATIVE_FOUR_NUMBERS")
    for key, rows in tables.items(): require(f"{key}.csv](tables/{key}.csv): {len(rows)}행" in text, "MD_CSV_COUNTS")
    for token in ("새 학습·F 추론·새 선정·배포 비용 측정은 모두0", "rcond1e−15", "보편 하한도 아니다", "MAE·물리 MSE는 각각의 minimum이 아니며", "E(Y)는 배포 입력이 아니다", "strict K-only도 아니다", "동일 고정 공간의 단독 인과효과로 부르지 않는다", "24개의 독립 평가가 아니다", "과거 E/D의 FIT 미래 감독은 인정", "독립 일반화", "예측 목적 표현의 첫 시도가 아니다"):
        require(token in text, "MD_SCIENTIFIC_SCOPE:"+token)
    return len(links)


def verify(folder):
    require(all(isinstance(v, str) and len(v) == 64 for v in (REPORT_SHA, MANIFEST_SHA, REPORT_SOURCE_SHA, *INPUT_SHA.values())), "ACTUAL_REPORT_NOT_YET_REGISTERED")
    require(record(common.__file__)["sha256"] == COMMON_SHA, "IMMUTABLE_METADATA_HELPERS")
    folder = resolve(folder); manifest_path, report_path = folder / "MANIFEST.json", folder / "REPORT_KO.md"
    require(record(manifest_path)["sha256"] == MANIFEST_SHA and record(report_path)["sha256"] == REPORT_SHA, "EXACT_ACTUAL_REPORT_MANIFEST")
    manifest = parse(manifest_path.read_text(encoding="utf-8")); require(manifest["status"] == "COMPLETE" and Path(manifest["source"]["path"]).name == "make_learned_output_space_report_v2.py" and manifest["source"]["sha256"] == REPORT_SOURCE_SHA, "CANONICAL_V2_REPORT_SOURCE")
    metadata(manifest["source"]); values = {}
    for key, sha in INPUT_SHA.items():
        binding = manifest["inputs"][key]; require(binding["sha256"] == sha, "EXACT_INPUT_SHA")
        values[key] = parse(metadata(binding).read_text(encoding="utf-8-sig")); require(values[key]["status"] == ("PASS" if key == "audit" else "COMPLETE") and values[key]["protected_TEST_accessed"] is False, "COMPLETE_DIAGNOSIS_AND_PASS_NO_TEST")
    require(all(values["audit"]["diagnostic"][key] == manifest["inputs"]["diagnostic"][key] for key in ("path", "sha256")), "EXACT_SCIENTIFIC_AUDIT_CHAIN")
    for key in ("scientific_NPZ_or_model_read", "new_F_or_training_or_timing", "protected_TEST_accessed", "all_FP32_bound_or_past_predictability_claimed"): require(manifest[key] is False, "METADATA_ONLY_REPORT_SCOPE")
    for data, owner in values["diagnostic"]["datasets"].items():
        for name, condition in owner["conditions"].items():
            checked = values["audit"]["datasets"][data]["conditions"][name]; require(condition["selected_state"] == checked["selected_state"] and condition["selected_update"] == checked["selected_update"], "SAME_AUDITED_SELECTED_STATE")
    tables = expected_tables(values["diagnostic"]); fields, refs = {}, 0
    for name in TABLES:
        require(manifest["table_row_counts"][name] == len(tables[name]), "MANIFEST_TABLE_COUNTS")
        expected = relative_values(tables[name], folder); fields[name] = check_table(folder / "tables" / (name+".csv"), expected); refs += common.check_reference_paths(expected, folder)
    require([len(tables[k]) for k in ("CELLS", "SCORES", "TRACES", "MATCHED_L0_FULL_STOP", "FAILURES", "RESOURCE")] == [24, 672, 72, 8, 0, 1], "COMPLETE_PRIMARY_REPORT_COUNTS")
    paths = common.check_manifest_files(folder, manifest["files"]); png, svg = common.inspect_images(paths)
    require(len(paths) == 16 and png == svg == 4 and {p.resolve() for p in folder.rglob("*") if p.is_file()} == set(paths) | {manifest_path.resolve()}, "SEVEN_CSV_EIGHT_FIGURES_EXACT_FILES")
    return {"status": "PASS", "report": record(report_path), "manifest": record(manifest_path), "source": manifest["source"], "inputs": {key: manifest["inputs"][key] for key in INPUT_SHA}, "table_rows_checked": {key: len(rows) for key, rows in tables.items()}, "CSV_primitive_fields_compared": fields, "files_checked": len(paths), "PNG_count": png, "SVG_count": svg, "local_links_checked": check_markdown(folder, tables), "relative_artifact_path_existence_checks": refs, "verification_scope": "Primary cell/phase/mask/per-channel scores, scalar trace values/booleans, matched differences, state refs, failures/resources and selected SVD group rank/singular/certificate summaries. Not all nested SVD certificates, original arrays or any F/training/timer replay; script checks image format rather than visual layout.", "scientific_array_model_F_training_replay": False, "figure_visual_review_performed_by_this_script": False, "protected_TEST_accessed": False, "audit_source": record(__file__)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--report-dir", required=True); parser.add_argument("--output", required=True); args = parser.parse_args()
    try: result = verify(args.report_dir)
    except Exception as error: result = {"status": "FAILED_PRESERVED", "failure": {"type": type(error).__name__, "message": str(error)}, "audit_source": record(__file__)}
    with resolve(args.output).open("x", encoding="utf-8") as handle: json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")
    print(json.dumps({"status": result["status"], "output": str(resolve(args.output))}), flush=True); raise SystemExit(0 if result["status"] == "PASS" else 2)
