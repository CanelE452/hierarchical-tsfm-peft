"""Independent matched-L7 report metadata audit; never import its generator or open scientific arrays."""
from pathlib import Path
import argparse
import csv
import json
import math
import os
import re
import statistics
from urllib.parse import unquote

import audit_linear_e_report_v1 as common

require, record, find, relative_values = common.require, common.record, common.find, common.relative_values
ROOT = common.ROOT
STATES = ("RECON_CP0", "E_ONLY_s92601", "E_ONLY_s92602")
SEEDS = (92601, 92602)
ROUTES = ("RAW_BOLT",) + tuple(r for s in STATES for r in (s+"_BOLT", s+"_NATIVE_s92601", s+"_NATIVE_s92602"))
TABLES = ("SCORES", "PAIRED", "CHECKPOINTS", "UPDATES", "CONDITIONS", "PROBES", "SPACE_CELLS", "SPACE_METHODS", "CONTEXT_PARITY", "FAILURES", "WHOLE_COST", "WHOLE_COST_SAMPLES", "RESOURCES")
COUNTS = (120, 72, 30, 3600, 6, 6, 12, 36, 3, 0, 20, 100, 4)
INPUT_SHA = {"train": "4a6f2d3601c27bf000659aa99a586a2f04d72371766f7558cc78cc16028caa2e", "audit": "429296ac7346189953813292291584d0c5ceb0a60028b4c9b955a2a429a1db87", "cost": "6fce6c6a462cb0b45f6cc3932f94c615ad02b17892c6aade642bc2ad39a6b51c", "cost_audit": "cd8716119609ad5a3148d7f9c620fb08b938f9a5d36f61ba7f111dc3ca880983"}
REPORT_SHA = "2adaf2ac2104b658e36ec10060e58604ce5b699a17f95d4b6ea1b2ee35b70c62"
MANIFEST_SHA = "e4ec0a98b24019b37bdd414b9569c322825a3df805bdd084e5d1d843b47e272d"
SOURCE_SHA = "b401543919ada3d1633df67d909a7ad4152bae5567d479ea6b440ee30033bc20"
COMMON_SHA = "c87fe5ada4de37e0f46fc61303ea0d18fb0bd9178d4ddc1771e92e6f26acf005"
PRIOR_KEYS = ("E_only_diagnostic", "E_only_audit", "joint_recon_diagnostic", "joint_recon_audit", "full_gradient_parent", "full_gradient_parent_audit", "causal_lag_diagnostic", "causal_lag_audit", "parent_report", "parent_report_manifest", "parent_report_audit", "selection_registry", "plan")
FIGURES = ("01_actual_errors", "02_CP_and_clipping", "03_output_space_diagnostics", "04_same_CUDA_whole_cost")
FLAGS = {"trainable_parameter_names": ["head.weight", "head.bias"], "E_requires_grad": False, "D_requires_grad": False, "center_requires_grad": False, "E_grad_is_None": True, "D_grad_is_None": True}


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT/path).resolve()


def local_file(path, base=None):
    path = Path(path)
    path = path if path.is_absolute() else (base or ROOT)/path
    resolved = path.resolve()
    require(resolved.is_relative_to(ROOT.resolve()), "LOCAL_PATH_CONTAINMENT")
    require(not any(p.is_symlink() for p in (path, *path.parents)), "SYMLINK_PATH_REJECTED")
    require(resolved.is_file(), "EXISTING_LOCAL_FILE:"+str(path))
    return resolved


def finite(value):
    if isinstance(value, dict):
        for item in value.values(): finite(item)
    elif isinstance(value, list):
        for item in value: finite(item)
    elif isinstance(value, float): require(math.isfinite(value), "NONFINITE_METADATA")


def parse(text):
    value = common.parse(text); finite(value); return value


def metadata(binding):
    require(Path(binding["path"]).suffix.lower() in (".json", ".jsonl", ".md", ".py"), "METADATA_ONLY_BINDING")
    path = local_file(binding["path"]); actual = record(path)
    require(actual["sha256"] == binding["sha256"], "EXACT_METADATA_SHA")
    if "bytes" in binding: require(type(binding["bytes"]) is int and actual["bytes"] == binding["bytes"], "EXACT_METADATA_BYTES")
    return path


def same(first, second):
    require(resolve(first["path"]) == resolve(second["path"]) and first["sha256"] == second["sha256"], "EXACT_PARENT_BINDING")


def selected(row):
    require([p["update"] for p in row["checkpoints"]] == [0, 100, 200, 400, 600], "FIVE_NATIVE_CHECKPOINTS")
    cp = min(row["checkpoints"], key=lambda p: (p["tune"]["mse"], p["update"]))
    require(cp["update"] == row["selected_update"], "TUNE_ONLY_EARLIEST_CP"); same(cp["state"], row["selected_state"])
    return cp


def expected_tables(values):
    finite(values); train, audit, cost = (values[k] for k in ("train", "audit", "cost")); out = {key: [] for key in TABLES}
    def scores(base, item):
        for mask, field in (("all_observed", "scores_all_observed"), ("complete_C", "scores_complete_mask")):
            for scale in ("normalized", "physical_per_channel_diagnostic"):
                out["SCORES"].append({**base, "mask": mask, "scale": scale, **item[field][scale], "artifact": item.get("artifact")})
    require(set(train["states"]) == set(STATES), "EXACT_THREE_FROZEN_STATES")
    for state in STATES:
        owner = train["states"][state]; require(set(owner["native"]) == {str(s) for s in SEEDS}, "EXACT_TWO_HEAD_SEEDS")
        for split in ("fit", "tune", "val"): scores({"state": state, "predictor": "BOLT", "seed": 0, "split": split, "new_head_fit": False}, owner["Bolt"][split])
        for seed in SEEDS:
            row = owner["native"][str(seed)]; cp = selected(row); base = {"state": state, "predictor": "NATIVE", "seed": seed, "new_head_fit": True}
            require(row["post_optimizer_frozen_flag_checks"] == 600 and row["frozen_state_checkpoint_byte_checks"] == 5 and row["final_frozen_status"] == FLAGS and row["frozen_bytes_checked_at_CP_and_final_only"] is True and row["independent_per_update_frozen_bytes_replayed"] is False and row["gradient_clipping_parameter_scope"] == "HEAD_ONLY" and row["training_head_values_changed"] is True, "RECORDED_FROZEN_HEAD_TRAINING_SCOPE")
            for split, item in (("fit", row["FIT"]), ("tune", cp), ("val", row["DEV"])): scores({**base, "split": split}, item)
            out["CHECKPOINTS"] += [{**base, **{k: v for k, v in item.items() if k != "state"}, "checkpoint_state": item["state"], "selected": item["update"] == cp["update"]} for item in row["checkpoints"]]
            log = metadata(row["updates_log"]); require(log.suffix == ".jsonl", "JSONL_UPDATE_METADATA_ONLY")
            events = [parse(line) for line in log.read_text(encoding="utf-8").splitlines()]
            require([p["update"] for p in events] == list(range(1, 601)) and all(type(p["clipped"]) is bool for p in events) and sum(p["clipped"] for p in events)/600 == row["global_clipping_fraction"], "SIX_HUNDRED_UPDATE_SEQUENCE_AND_CLIPPING")
            require(all(p["gradient_clipping_parameter_scope"] == "HEAD_ONLY" and all(p[k] == v for k, v in FLAGS.items()) for p in events), "RECORDED_HEAD_ONLY_FLAGS")
            out["UPDATES"] += [{**base, **p} for p in events]
            keys = ("selected_update", "selected_state", "selected_head_state_sha256", "order_sha256", "updates_log", "selection_lock", "loss", "FIT_channel_observed_counts", "global_clipping_fraction", "post_optimizer_frozen_flag_checks", "frozen_state_checkpoint_byte_checks", "final_frozen_status", "training_head_values_changed", "selected_head_nonzero", "frozen_bytes_checked_at_CP_and_final_only", "independent_per_update_frozen_bytes_replayed", "gradient_clipping_parameter_scope")
            out["CONDITIONS"].append({**base, **{k: row[k] for k in keys}, "frozen_state_source": owner["frozen_state_source"], "E_previously_supervised_by_FIT_future": owner["E_previously_supervised_by_FIT_future"], "D_previously_supervised_by_FIT_future": owner["D_previously_supervised_by_FIT_future"]})
            require(row["probe"]["status"] == "PASS" and row["probe"]["optimizer_updates"] == 0, "ZERO_UPDATE_CP0_PROBE")
            out["PROBES"].append({**base, **row["probe"]})
            for split in ("tune", "val"):
                cell = audit["new_output_space_diagnostics"][state][str(seed)][split]; where = {"state": state, "seed": seed, "split": split}
                require(cell["status"] == "COMPLETE" and set(cell["methods"]) == {"ACTUAL_NATIVE", "EY", "FLAT"} and cell["all_real_latent_or_all_FP32_lower_bound_claimed"] is False and cell["independent_external_replay_of_new_diagnostic_claimed"] is False, "NUMERIC_SPACE_SCOPE")
                require(cell["selected_E_previously_supervised_by_FIT_future_targets"] == owner["E_previously_supervised_by_FIT_future"] and cell["selected_D_previously_supervised_by_FIT_future_targets"] is False and cell["selected_native_head_previously_supervised_by_FIT_future_targets"] == (cp["update"] > 0) and cell["native_head_training_procedure_used_FIT_future_targets"] is True, "SEPARATE_E_D_AND_SELECTED_HEAD_SUPERVISION")
                keys = ("minimum_normalized_MSE", "complete_target_positions", "target_positions", "joint_unknown_future_coordinates", "joint_output_coordinates_before_mask", "lag_slope_parameters_per_C", "joint_metadata", "minimum_solution_scores_complete_C", "RAW_BOLT_scores_complete_mask", "current_C_persistence_scores_complete_mask", "selected_E_previously_supervised_by_FIT_future_targets", "selected_D_previously_supervised_by_FIT_future_targets", "selected_native_head_previously_supervised_by_FIT_future_targets", "native_head_training_procedure_used_FIT_future_targets", "EY_encoding_backend", "EY_incomplete_C_horizons_filled_with_current_C", "artifact", "selected_state", "selected_forecast_source", "input_source")
                summary = {**where, **{k: cell[k] for k in keys}, "computed_during_current_audit": True, "RAW_complete_MSE": cell["RAW_BOLT_scores_complete_mask"]["normalized"]["mse"], "matched_Bolt_complete_MSE": owner["Bolt"][split]["scores_complete_mask"]["normalized"]["mse"]}
                summary.update({label: cell["methods"][method]["scores_complete_mask"]["normalized"]["mse"] for label, method in (("native_complete_MSE", "ACTUAL_NATIVE"), ("EY_complete_MSE", "EY"), ("FLAT_complete_MSE", "FLAT"))})
                out["SPACE_CELLS"].append(summary); out["SPACE_METHODS"] += [{**where, "method": method, **entry} for method, entry in cell["methods"].items()]
    raw = values["science"]["datasets"]["robin"]["spaces"]["PCA"]["conditions"]["L7_RECON"]["splits"]
    for split in ("fit", "tune", "val"): scores({"state": "RAW_C", "predictor": "RAW_BOLT", "seed": 0, "split": split, "new_head_fit": False}, raw[split]["RAW_BOLT"])
    for state in STATES:
        for seed in SEEDS:
            for split in ("fit", "tune", "val"):
                for mask in ("all_observed", "complete_C"):
                    for scale in ("normalized", "physical_per_channel_diagnostic"):
                        where = {"state": state, "split": split, "mask": mask, "scale": scale}
                        a = find(out["SCORES"], **where, predictor="NATIVE", seed=seed); b = find(out["SCORES"], **where, predictor="BOLT", seed=0); r = find(out["SCORES"], state="RAW_C", split=split, mask=mask, scale=scale, predictor="RAW_BOLT", seed=0)
                        out["PAIRED"].append({**where, "seed": seed, "native_MSE": a["mse"], "Bolt_MSE": b["mse"], "RAW_MSE": r["mse"], "native_minus_Bolt_MSE": a["mse"]-b["mse"], "native_minus_RAW_MSE": a["mse"]-r["mse"], "native_MAE": a["mae"], "Bolt_MAE": b["mae"], "RAW_MAE": r["mae"], "native_minus_Bolt_MAE": a["mae"]-b["mae"]})
    out["CONTEXT_PARITY"] = [dict(p) for p in audit["optional_NumPy_context_checks"]]
    require(len(out["CONTEXT_PARITY"]) == 3 and {r["split"] for r in out["CONTEXT_PARITY"]} == {"fit", "tune", "val"} and all(r["primary_gate"] is False for r in out["CONTEXT_PARITY"]), "THREE_OPTIONAL_CONTEXTS_NOT_PRIMARY_GATES")
    for stage, field in (("train", "failed_paths"), ("audit", "partial_paths"), ("audit", "new_output_space_diagnostic_failures"), ("cost", "failed_paths")):
        out["FAILURES"] += [{"stage": stage, "collection": field, **p} for p in values[stage][field]]
    rows = cost["cost"]["robin"]["rows"]
    require(len(rows) == 20 and {(r["route"], r["batch"]) for r in rows} == {(r, b) for r in ROUTES for b in (1, 8)}, "EXACT_TWENTY_COST_CELLS")
    for row in rows:
        route = row["route"]; matching = "RAW_BOLT" if route == "RAW_BOLT" else route.split("_NATIVE_")[0]+"_BOLT" if "_NATIVE_" in route else route
        require(row["matched_Bolt_route"] == matching, "SAME_E_BOLT_DENOMINATOR")
        seconds = row["seconds"]; samples = row["samples"]
        require(row["status"] == "COMPLETE" and len(samples) == len(seconds) == 5 and all(type(s) in (int, float) and s > 0 for s in seconds) and row["median_seconds"] == statistics.median(seconds) and row["median_seconds_per_window"] == statistics.median(seconds)/row["batch"], "POSITIVE_FIVE_SAMPLE_COST_MEDIANS")
        require([s["repeat"] for s in samples] == list(range(5)), "EXACT_REPEAT_SEQUENCE")
        for field, denominator_route in (("RAW", "RAW_BOLT"), ("matched_Bolt", matching)):
            denominator = find(rows, route=denominator_route, batch=row["batch"]); by_repeat = {s["repeat"]: s["seconds"] for s in denominator["samples"]}
            paired = [by_repeat[s["repeat"]] for s in samples]; ratios = [s/d for s, d in zip(seconds, paired)]
            require(row["paired_"+field+"_seconds"] == paired and row["paired_"+field+"_ratios"] == ratios and row["paired_"+field+"_median_ratio"] == statistics.median(ratios) and row["ratio_of_"+field+"_medians"] == statistics.median(seconds)/statistics.median(denominator["seconds"]), "PAIRED_AND_RATIO_OF_MEDIANS")
        for index, sample in enumerate(samples):
            require(sample["route"] == route and sample["batch"] == row["batch"] and type(sample["perf_start_ns"]) is type(sample["perf_end_ns"]) is int and (sample["perf_end_ns"]-sample["perf_start_ns"])/1e9 == sample["seconds"] == seconds[index], "INTEGER_NS_SAMPLE_BINDING")
        out["WHOLE_COST"].append({k: v for k, v in row.items() if k != "samples"}); out["WHOLE_COST_SAMPLES"] += samples
    for stage in INPUT_SHA:
        resource = values[stage]["resource" if stage == "audit" else "resource_terminal"]
        require(isinstance(resource, dict) and type(resource.get("wall_seconds")) in (int, float) and resource["wall_seconds"] >= 0, "FOUR_NONEMPTY_RESOURCE_RECEIPTS")
        out["RESOURCES"].append({"stage": stage, "new_deployment_timing": stage == "cost", **resource})
    require(tuple(len(out[k]) for k in TABLES) == COUNTS, "EXACT_THIRTEEN_TABLE_COUNTS_ZERO_FAILURES")
    return out


def exact(actual, expected, label):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), label)
        return sum(exact(actual[k], v, label+"."+k) for k, v in expected.items())
    if isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label)
        return sum(exact(a, b, label) for a, b in zip(actual, expected))
    if isinstance(expected, bool): require(type(actual) is bool and actual is expected, label)
    elif isinstance(expected, int): require(type(actual) is int and actual == expected, label)
    elif isinstance(expected, float): require(type(actual) in (int, float) and math.isfinite(actual) and actual == expected, label)
    else: require(type(actual) is type(expected) and actual == expected, label)
    return 1


def check_table(path, expected):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle); rows = list(reader); fields = reader.fieldnames
    names = set(k for row in expected for k in row) or {"no_failed_paths"}
    require(fields is not None and len(fields) == len(set(fields)) and set(fields) == names, "EXACT_CSV_COLUMNS")
    require(len(rows) == len(expected), "CSV_ROW_COUNT")
    count = 0
    for index, (row, source) in enumerate(zip(rows, expected)):
        require(None not in row, "EXACT_CSV_ROW_WIDTH")
        for key in names:
            text = row[key]; value = source.get(key); label = "CSV_CELL:"+Path(path).name+":"+str(index)+":"+key
            require(isinstance(text, str), label)
            if key not in source or value is None: require(text == "", label); count += 1; continue
            if isinstance(value, (dict, list)): actual = parse(text)
            elif isinstance(value, bool): require(text in ("True", "False"), label); actual = text == "True"
            elif isinstance(value, int): actual = int(text)
            elif isinstance(value, float): actual = float(text)
            else: actual = text
            count += exact(actual, value, label)
    return count


def check_reference_paths(value, folder):
    if isinstance(value, list): return sum(check_reference_paths(v, folder) for v in value)
    if not isinstance(value, dict): return 0
    total = 0
    for key, item in value.items():
        if key in ("path", "snapshot_path") and isinstance(item, str):
            require(not re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:|^[/\\]", item), "RELATIVE_ARTIFACT_PATH")
            local_file(item, folder); total += 1
        else: total += check_reference_paths(item, folder)
    return total


def check_files(folder, files):
    expected = {"REPORT_KO.md"} | {"tables/"+k+".csv" for k in TABLES} | {"figures/"+name+"."+ext for name in FIGURES for ext in ("png", "svg")}
    require(len(files) == 22 and {p["path"] for p in files} == expected, "EXACT_TWENTYTWO_MANIFEST_FILES")
    paths = []
    for item in files:
        path = local_file(item["path"], folder); require(path.is_relative_to(folder), "REPORT_FILE_CONTAINMENT")
        actual = record(path); require(actual["sha256"] == item["sha256"] and type(item["bytes"]) is int and actual["bytes"] == item["bytes"], "REPORT_FILE_SHA_BYTES"); paths.append(path)
    require(len(set(paths)) == 22, "UNIQUE_REPORT_FILES")
    all_paths = list(folder.rglob("*")); require(not any(p.is_symlink() for p in all_paths), "SYMLINK_REPORT_ENTRY")
    require({p.resolve() for p in all_paths if p.is_file()} == set(paths)|{folder/"MANIFEST.json"}, "EXACT_REPORT_DIRECTORY_FILES")
    png, svg = common.inspect_images(paths); require(png == svg == 4, "FOUR_PNG_FOUR_SVG")
    return paths


def check_markdown(folder, tables, values, bindings=None):
    text = (folder/"REPORT_KO.md").read_text(encoding="utf-8"); links = re.findall(r"(!?)\[[^\]]*\]\(([^)]+)\)", text)
    images = []
    for marker, target in links:
        target = unquote(target.split("#", 1)[0])
        require(target and not re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:|^[/\\]", target), "RELATIVE_MARKDOWN_LINK")
        local_file(target, folder)
        if marker: images.append(target)
    require(len(images) == 4 and set(images) == {"figures/"+name+".png" for name in FIGURES}, "FOUR_MARKDOWN_IMAGES")
    def paired(state, seed, split): return find(tables["PAIRED"], state=state, seed=seed, split=split, mask="all_observed", scale="normalized")
    for split, label in (("tune", "TUNE"), ("val", "기노출 DEV")):
        n = sum(paired(s, seed, split)["native_MSE"] > paired(s, seed, split)["Bolt_MSE"] for s in STATES for seed in SEEDS)
        require(f"{label} {n}/6" in text, "MD_NATIVE_BOLT_COUNTS")
    n = sum(paired(s, seed, "val")["native_MSE"] < paired(s, seed, "val")["RAW_MSE"] for s in STATES for seed in SEEDS)
    require(f"Raw보다 DEV MSE가 낮은 native는 {n}/6" in text, "MD_NATIVE_RAW_COUNT")
    for state in STATES:
        for seed in SEEDS:
            row = find(tables["CONDITIONS"], state=state, seed=seed); t, d = (paired(state, seed, split) for split in ("tune", "val"))
            require(f"- {state} / {seed}: CP{row['selected_update']}; TUNE native {t['native_MSE']:.10f}, Bolt {t['Bolt_MSE']:.10f}; DEV native {d['native_MSE']:.10f}, Bolt {d['Bolt_MSE']:.10f}, Raw {d['RAW_MSE']:.10f}." in text, "MD_SIX_SELECTED_CP_SCORE_ROWS")
    clips = [r["global_clipping_fraction"] for r in tables["CONDITIONS"]]
    require(f"선정 CP100은 {sum(r['selected_update']==100 for r in tables['CONDITIONS'])}/6이고 clipping fraction은 {min(clips):.6f}~{max(clips):.6f}" in text, "MD_SELECTION_AND_CLIPPING")
    require(f"NumPy 문맥 검사 {values['train']['optional_NumPy_context_parity_checks']}개, 기록된 문맥 실패 {len(values['train']['optional_NumPy_context_parity_failures'])}개" in text, "MD_OPTIONAL_CONTEXT_COUNTS")
    for r in tables["SPACE_CELLS"]:
        require(f"- {r['state']} / {r['seed']} / {r['split']}: native {r['native_complete_MSE']:.10f}, Bolt {r['matched_Bolt_complete_MSE']:.10f}, E(Y) {r['EY_complete_MSE']:.10f}, minimum {r['minimum_normalized_MSE']:.10f}, Raw {r['RAW_complete_MSE']:.10f}; complete {r['complete_target_positions']}/{r['target_positions']}." in text, "MD_TWELVE_SPACE_ROWS")
    tune = [find(tables["SPACE_CELLS"], state=s, seed=SEEDS[0], split="tune") for s in STATES]
    dev = [find(tables["SPACE_CELLS"], state=s, seed=SEEDS[0], split="val") for s in STATES]
    require(f"TUNE retained numeric minimum이 Raw보다 높은 고정 E/D는 {sum(r['minimum_normalized_MSE']>r['RAW_complete_MSE'] for r in tune)}/3이고, DEV minimum이 Raw보다 낮은 고정 E/D는 {sum(r['minimum_normalized_MSE']<r['RAW_complete_MSE'] for r in dev)}/3" in text, "MD_OPPOSITE_TUNE_DEV_RETAINED_SPACES")
    for r in tables["WHOLE_COST"]:
        require(f"- B{r['batch']} {r['route']}: {r['median_seconds_per_window']*1000:.6f} ms/window; Raw paired {r['paired_RAW_median_ratio']:.6f}, medians {r['ratio_of_RAW_medians']:.6f}; same-E Bolt paired {r['paired_matched_Bolt_median_ratio']:.6f}, medians {r['ratio_of_matched_Bolt_medians']:.6f}." in text, "MD_TWENTY_COST_ROWS")
    for key, rows in tables.items(): require(f"[{key}.csv](tables/{key}.csv): {len(rows)}행" in text, "MD_THIRTEEN_CSV_COUNTS")
    for key, item in (bindings or {}).items():
        path = os.path.relpath(resolve(item["path"]), folder).replace("\\", "/")
        require(f"[{key}]({path}): `{item['sha256']}`" in text, "MD_SOURCE_RECEIPT_BINDING")
    for token in ("새 6fit×600=3,600 update·30CP·선정 FIT/DEV12예측", "E/D/F 학습은0", "독립 가중치 초기화가 아니다", "같은 전체 학습 계산량 대조도 아니다", "고정 전체 FIT 채널 관측수", "모든6개 성공/실패 잠금", "최종 C는 현재 C persistence와 같지 않다", "z_past[-1-l]", "C_current + sum_l (z_last - z_past[-1-l]) D_l", "새 F forward는0", "새 matched RECON C decode3개", "C byte 보존6개", "매 update bytes·loss·backward·optimizer를 독립 재현한 검사는 아니다", "12개 독립 자료가 아니다", "세 번째 외부 재현은 아니다", "각 지표의 minimum이 아니다", "real/FP32 false-bound flags", "명시적 warmup2는 모든 비측정 forward가2회라는 뜻이 아니다", "5순환은 10개 위치의 완전 균형이 아니다", "내부 finite/shape 검사·sync는 timer 안", "median of paired ratios와 ratio of medians", "독립 자료나 독점 peak로 바꾸지 않는다", "독립 일반화 검증이 아니다", "strict K-only도 아니다", "JSON/JSONL metadata만 읽고"):
        require(token in text, "MD_SCIENTIFIC_SCOPE:"+token)
    return len(links)


def load_inputs(manifest):
    values = {}
    for key, digest in INPUT_SHA.items():
        binding = manifest["inputs"][key]; require(binding["sha256"] == digest, "EXACT_STAGE_SHA:"+key)
        require(Path(binding["path"]).suffix == ".json", "FOUR_JSON_STAGE_INPUTS")
        values[key] = parse(metadata(binding).read_text(encoding="utf-8-sig"))
        require(values[key]["status"] == ("PASS" if "audit" in key else "COMPLETE") and values[key]["protected_TEST_accessed"] is False, "COMPLETE_PASS_NO_TEST")
    train, audit, cost, cost_audit = (values[k] for k in INPUT_SHA)
    same(audit["diagnostic"], manifest["inputs"]["train"]); same(cost_audit["cost"], manifest["inputs"]["cost"])
    for key in ("audit", "cost_audit"): same(values[key]["audit_source"], manifest["inputs"][key+"_source"])
    for value in (cost, cost_audit):
        for key, target in (("matched_l7_native_diagnostic", "train"), ("matched_l7_native_audit", "audit")): same(value[key], manifest["inputs"][target])
        same(value["selection_registry"], train["selection_registry"])
    require(audit["training_verification_status"] == "PASS" and audit["new_space_diagnostic_status"] == "COMPLETE" and not audit["partial_paths"] and not audit["new_output_space_diagnostic_failures"], "TRAINING_AND_SPACE_AUDIT_COMPLETE")
    require(train["native_training_fits"] == train["resolved_selection_locks"] == 6 and train["training_updates_completed"] == 3600 and train["checkpoint_states_saved"] == 30 and train["all_6_locks_before_DEV"] is True and not train["failed_paths"], "SIX_HEADS_THIRTY_CP_ALL_LOCKS")
    require(train["matched_Bolt_C_decodes"] == train["cached_Bolt_contexts_reused"] == 9 and train["E0_new_matched_RECON_C_decodes"] == 3 and train["selected_E_only_Bolt_C_byte_checks"] == 6 and all(train[k] == 0 for k in ("new_F_forward_calls", "new_F_model_objects_loaded", "new_E_training_updates", "new_D_training_updates", "probe_optimizer_updates")), "CACHED_F_AND_MATCHED_DECODER_SCOPE")
    require(tuple(cost["routes"]) == ROUTES and cost["cost_split"] == "tune" and cost["completed_cost_cells"] == 20 and cost["recorded_timer_samples"] == 100 and not cost["failed_paths"] and cost["native_predictor_loaded_or_timed"] is True and cost["cached_future_returned_instead_of_actual_F_inside_timer"] is False and cost["new_training_updates"] == 0, "ACTUAL_TWENTY_CELL_HUNDRED_SAMPLE_COST")
    require(cost["F_before"] == cost["F_after"] == train["F_parent_signatures_reused"]["before"] == train["F_parent_signatures_reused"]["after"] and cost["selected_states_before"] == cost["selected_states_after"] and len(cost["selected_states_after"]) == 9 and cost["native_heads_before"] == cost["native_heads_after"] and len(cost["native_heads_after"]) == 6, "F_ADAPTERS_AND_HEADS_FROZEN")
    for key in PRIOR_KEYS:
        metadata(train[key]); same(audit[key], train[key]); same(manifest["inputs"][key], train[key])
    for key, item in (("cost_plan", cost["cost_plan"]), ("controller_source", train["controller_source"]), ("native_source", train["native_source"])): same(manifest["inputs"][key], item)
    registry = parse(metadata(train["selection_registry"]).read_text(encoding="utf-8-sig"))
    require(registry["status"] == "ALL_SIX_RESOLVED_BEFORE_DEV" and registry["DEV_opened_before_seal"] is False and len(registry["conditions"]) == 6, "SIX_SEALED_LOCKS")
    seen = set()
    for item in registry["conditions"]:
        key = item["state"], item["seed"]; require(key in {(s, seed) for s in STATES for seed in SEEDS} and key not in seen, "UNIQUE_SIX_LOCKS"); seen.add(key)
        row = train["states"][key[0]]["native"][str(key[1])]; same(item["selection_lock"], row["selection_lock"])
        lock = parse(metadata(item["selection_lock"]).read_text(encoding="utf-8-sig")); cp = selected(row)
        require(lock["status"] == row["status"] == item["status"] == "COMPLETE" and lock["DEV_opened_before_lock"] is False and lock["selected_update"] == cp["update"], "TUNE_ONLY_SEALED_LOCK")
        same(lock["selected_state"], row["selected_state"])
    values["science"] = parse(metadata(train["causal_lag_diagnostic"]).read_text(encoding="utf-8-sig")); require(values["science"]["status"] == "COMPLETE", "COMPLETE_RAW_BASELINE_METADATA")
    return values


def verify(folder):
    require(all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for v in (REPORT_SHA, MANIFEST_SHA, SOURCE_SHA, *INPUT_SHA.values())), "ACTUAL_REPORT_NOT_YET_REGISTERED")
    require(record(common.__file__)["sha256"] == COMMON_SHA, "IMMUTABLE_METADATA_HELPERS")
    folder = resolve(folder); report_path, manifest_path = local_file(folder/"REPORT_KO.md"), local_file(folder/"MANIFEST.json")
    require(record(report_path)["sha256"] == REPORT_SHA and record(manifest_path)["sha256"] == MANIFEST_SHA, "EXACT_ACTUAL_REPORT_MANIFEST")
    manifest = parse(manifest_path.read_text(encoding="utf-8")); require(manifest["status"] == "COMPLETE" and manifest["source"]["sha256"] == SOURCE_SHA and Path(manifest["source"]["path"]).name == "make_matched_l7_native_report_v2.py", "EXACT_EXECUTED_REPORT_SOURCE")
    metadata(manifest["source"]); same(manifest["source"], manifest["inputs"]["report_source"])
    require(set(manifest["inputs"]) == set(INPUT_SHA)|set(PRIOR_KEYS)|{"audit_source", "cost_audit_source", "cost_plan", "controller_source", "native_source", "report_utility", "report_source"}, "EXACT_MANIFEST_INPUT_BINDINGS")
    for binding in manifest["inputs"].values(): metadata(binding)
    for key, value in {"PNG_count": 4, "SVG_count": 4, "scientific_NPZ_or_model_read": False, "new_updates_during_generation": 0, "new_head_fits_in_source": 6, "new_head_updates_in_source": 3600, "new_F_forward_calls_in_scientific_source": 0, "cost_uses_live_F_and_native": True, "geometry_cells_reuse_only_three_frozen_E_D_states": True, "independent_holdout": False, "strict_K_only": False, "protected_TEST_accessed": False}.items(): exact(manifest[key], value, "MANIFEST_SCOPE:"+key)
    values = load_inputs(manifest); tables = expected_tables(values); fields, refs = {}, 0
    require(set(manifest["table_row_counts"]) == set(TABLES), "EXACT_MANIFEST_TABLE_NAMES")
    paths = check_files(folder, manifest["files"])
    for key in TABLES:
        require(manifest["table_row_counts"][key] == len(tables[key]), "MANIFEST_TABLE_COUNT")
        expected = relative_values(tables[key], folder); fields[key] = check_table(folder/"tables"/(key+".csv"), expected); refs += check_reference_paths(expected, folder)
    links = check_markdown(folder, tables, values, manifest["inputs"])
    return {"status": "PASS", "report": record(report_path), "manifest": record(manifest_path), "source": manifest["source"], "inputs": {k: manifest["inputs"][k] for k in INPUT_SHA}, "table_rows_checked": {k: len(v) for k, v in tables.items()}, "CSV_primitive_fields_compared": fields, "files_checked": len(paths), "PNG_count": 4, "SVG_count": 4, "local_links_checked": links, "relative_artifact_path_existence_checks": refs, "verification_scope": "Independent JSON/JSONL reconstruction of all 13 CSV tables and nested primitives; selection/log/flags, actual-cost integer-ns samples and both denominators, four resource receipts, exact sources/files/relative Markdown numbers and links. No scientific arrays, F/GPU/timer/backward/optimizer or per-update frozen bytes replay; image structure only, no visual review.", "scientific_array_model_F_training_or_timer_replayed": False, "figure_visual_review_performed_by_this_script": False, "protected_TEST_accessed": False, "audit_source": record(__file__)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--report-dir", required=True); parser.add_argument("--output", required=True); args = parser.parse_args()
    try: result = verify(args.report_dir)
    except Exception as error: result = {"status": "FAILED_PRESERVED", "failure": {"type": type(error).__name__, "message": str(error)}, "audit_source": record(__file__)}
    with resolve(args.output).open("x", encoding="utf-8") as handle: json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")
    print(json.dumps({"status": result["status"], "output": str(resolve(args.output))}), flush=True); raise SystemExit(0 if result["status"] == "PASS" else 2)
