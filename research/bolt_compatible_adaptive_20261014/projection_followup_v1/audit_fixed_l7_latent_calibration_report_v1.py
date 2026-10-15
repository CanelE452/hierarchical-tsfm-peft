"""Independent JSON/JSONL-to-CSV/Markdown audit; never import the report generator or open NPZ/models."""
from pathlib import Path
import argparse
import json
import math
import os
import re
import statistics
from urllib.parse import unquote

import audit_matched_l7_native_report_v1 as common

require, record, resolve, parse, metadata, find = common.require, common.record, common.resolve, common.parse, common.metadata, common.find
relative_values, check_table, local_file = common.relative_values, common.check_table, common.local_file
STATES = ("RECON_CP0", "E_ONLY_s92601", "E_ONLY_s92602")
FAMILIES = ("SCALE_ONLY", "SCALE_BIAS")
LAMBDAS = (0., 1e-6, 1e-4, 1e-2, .1, 1., 10.)
ROUTES = ("RAW_BOLT",)+tuple(r for s in STATES for r in (s+"_BOLT", s+"_SCALE_ONLY", s+"_SCALE_BIAS"))
TABLES = ("CANDIDATE_SCORES", "SELECTED_SCORES", "BASELINES", "PAIRED", "CANDIDATES", "COEFFICIENTS", "SOLVERS", "DIAGNOSTICS", "SELECTION", "SPACE_CONTEXT", "FAILURES", "WHOLE_COST", "WHOLE_COST_SAMPLES", "RESOURCES")
COUNTS = (384, 72, 48, 72, 48, 48, 6, 102, 6, 6, 0, 20, 100, 4)
PRIOR_KEYS = ("parent_report", "parent_report_manifest", "parent_report_audit", "matched_l7_native_diagnostic", "matched_l7_native_audit", "matched_l7_native_cost", "matched_l7_native_cost_audit", "native_selection_registry", "native_bundle_metadata_reader", "parent_report_auditor_source", "E_only_diagnostic", "E_only_audit", "joint_recon_diagnostic", "joint_recon_audit", "causal_lag_diagnostic", "causal_lag_audit")
EXTRA_BINDINGS = ("controller_source", "module_source", "selection_registry", "plan", "cost_plan", "report_utility")
FIGURES = ("01_selected_errors", "02_lambda_curves", "03_channel_and_surrogate_context", "04_whole_cost")
INPUT_SHA = {"train": "88ddbbf7287ad89b7bf81e170b3e05ddfc89bead700e2fefa50bb0dd4e85f576", "audit": "b31b742d3947a205dbaf01cd84239b3cde95e1ea9baf3331a3c08c48a0aa5ba0", "cost": "d8db7445a7a5a750361149cde9a3ad653d4926e871d72c387fe8cbd285aca17b", "cost_audit": "4a54f5b738beef15bc02a17687624afb8c4493f5df7463aeb7b8e4e35b619c40"}
REPORT_SHA = "e8f1036dcc27ee2a460b6d87a0f9d1d03368067dab5cb3abd32ec2ec8e368c4f"
MANIFEST_SHA = "e5a17e7f5131ed611b87beca2d49ec4a05e093c2eaf1d0f42d0343b5f0350198"
SOURCE_SHA = "3b91764dc6864dd09f1f68a07bd94f11094a002f7f508fc2f97d388e147ce1ad"
AUDIT_SOURCE_SHA = {"audit": "208816a2656800afa190a9f713abe3e51c4f8ec88cfc1e6116bd4e4788b2464d", "cost_audit": "4a78b4b8ecc0e000021941f613360b2e94457a3a2784ff104602be44d98e06d0"}
RUNTIME_SHA = "83db8dc961cdf66dc2b0b499c717f89651faebddbbc4f56f939d5d890793b2d5"
HELPER_SHA = "5f6b7bd4e397603c5441a7b8c765f8032a7f0274c4852bef14b57b8218891980"
AUDIT_COUNTS = {"fit_conditions_checked": 6, "actual_SVD_fits_checked": 6, "ridge_candidates_checked": 42, "identity_candidates_checked": 6, "FIT_TUNE_prediction_evaluations_checked": 96, "selected_DEV_prediction_evaluations_checked": 6, "score_fields_checked": 3672, "selection_locks_checked": 6, "protected_source_input_and_result_hashes_checked": 226}
COST_COUNTS = {"cost_cells_checked": 20, "recorded_timer_samples_checked": 100, "selected_full_state_hashes_checked": 9, "selected_q_coefficient_config_hashes_checked": 6, "selection_locks_checked": 6, "physical_prechecks_checked": 9, "physical_first8_windows_checked": 72, "composed_q_references_checked": 6, "independent_canonical_q_and_lag_D_checks": 6, "canonical_TUNE_bridges_checked": 1, "protected_prior_file_hashes_checked": 1923}


def same(first, second):
    require(resolve(first["path"]) == resolve(second["path"]) and first["sha256"] == second["sha256"], "EXACT_PARENT_BINDING")


def selected(row):
    valid = [c for c in row["candidates"] if c["status"] == "COMPLETE" and type(c.get("TUNE_MSE")) in (int, float) and math.isfinite(c["TUNE_MSE"])]
    require(bool(valid), "FINITE_ACTUAL_TUNE_CANDIDATE")
    winner = min(valid, key=lambda c: (c["TUNE_MSE"], c["kind"] != "IDENTITY", -c["lambda"] if c["kind"] == "RIDGE" else 0))
    require(row["selected_candidate"] == winner["id"] and row["selected_kind"] == winner["kind"] and row["selected_lambda"] == winner["lambda"] and row["selected_TUNE_MSE"] == winner["TUNE_MSE"], "IDENTITY_THEN_LARGER_LAMBDA_EXACT_TIE")
    same(row["selected_coefficients"], winner["coefficient_artifact"]); require(row["selected_coefficient_value_sha256"] == winner["coefficient_value_sha256"], "SELECTED_COEFFICIENT_HASH")
    return winner


def expected_tables(values):
    common.finite(values); train, audit, cost, native = (values[k] for k in ("train", "audit", "cost", "native")); out = {k: [] for k in TABLES}
    def scores(table, base, entry):
        for mask, field in (("all_observed", "scores_all_observed"), ("complete_C", "scores_complete_mask")):
            for scale in ("normalized", "physical_per_channel_diagnostic"): out[table].append({**base, "mask": mask, "scale": scale, **entry[field][scale], "prediction_artifact": entry.get("artifact")})
    def diagnostics(base, entry):
        require(entry["FP64_surrogate_is_not_deployment_parity_target"] is True and entry["surrogate_diagnostics"]["surrogate_zero_error_required"] is False, "FP32_PARITY_AND_MEASURED_SURROGATE_SEPARATE")
        require(all(p["pass"] is True and (p["atol"], p["rtol"]) == (1e-5, 2e-6) for p in entry["independent_FP32_deployment_parity"].values()), "PREDECLARED_FP32_DEPLOYMENT_PARITY")
        out["DIAGNOSTICS"].append({**base, **{k: entry[k] for k in ("status", "surrogate_diagnostics", "independent_FP32_deployment_parity", "FP64_surrogate_is_not_deployment_parity_target", "identity_checks", "coefficient_value_sha256", "input_source", "cached_Bolt_source")}, "prediction_artifact": entry["artifact"]})
    require(set(train["states"]) == set(STATES), "THREE_FIXED_STATES")
    for state in STATES:
        for split in ("fit", "tune", "val"): scores("BASELINES", {"state_label": state, "predictor": "BOLT", "split": split}, native["states"][state]["Bolt"][split])
        require(set(train["states"][state]["families"]) == set(FAMILIES), "TWO_FIXED_FAMILIES")
        for family in FAMILIES:
            row = train["states"][state]["families"][family]; winner = selected(row); verified = audit["states"][state][family]; base = {"state_label": state, "family": family}
            require([(c["id"], c["kind"], c["lambda"]) for c in row["candidates"]] == [("IDENTITY", "IDENTITY", None)]+[("RIDGE_"+format(v, "g"), "RIDGE", v) for v in LAMBDAS], "IDENTITY_AND_SEVEN_REGISTERED_LAMBDAS")
            require(verified["selected_candidate"] == winner["id"] and verified["selected_lambda"] == winner["lambda"] and verified["FIT_solver_rank"] == verified["independently_recomputed_fit"]["zero_lambda_numeric_rank"], "INDEPENDENT_SOLVER_SELECTION_RANK")
            out["SOLVERS"].append({**base, "fit_metadata": row["fit_metadata"], "fit_artifact": row["fit_artifact"], "independently_recomputed_fit": verified["independently_recomputed_fit"], "FIT_solver_rank": verified["FIT_solver_rank"], "FIT_input_source": row["FIT_input_source"], "FIT_cached_Bolt_source": row["FIT_cached_Bolt_source"], "frozen_state_source": row["frozen_state_source"]})
            for c in row["candidates"]:
                vc = find(verified["candidates"], id=c["id"]); require(all(vc[k] == c[k] for k in ("kind", "family", "lambda", "status", "coefficient_value_sha256")), "AUDITED_CANDIDATE_COEFFICIENT_BINDING"); same(vc["coefficient_artifact"], c["coefficient_artifact"])
                group = {**base, "candidate_id": c["id"], "kind": c["kind"], "lambda": c["lambda"], "selected": c["id"] == winner["id"]}
                out["CANDIDATES"].append({**group, **{k: c[k] for k in ("status", "metadata", "coefficient_artifact", "coefficient_value_sha256")}, "TUNE_MSE": c.get("TUNE_MSE"), "FIT_prediction_artifact": c.get("FIT", {}).get("artifact"), "TUNE_prediction_artifact": c.get("TUNE", {}).get("artifact")})
                out["COEFFICIENTS"].append({**group, **{k: vc[k] for k in ("a_FP64", "b_FP64", "a_FP32", "b_FP32", "coefficients_FP64", "coefficients_FP32", "nonfinite_coefficient_indices")}, "coefficient_artifact": c["coefficient_artifact"]})
                if c["status"] == "COMPLETE": require(c["TUNE_MSE"] == c["TUNE"]["scores_all_observed"]["normalized"]["mse"], "ACTUAL_TUNE_OWN_SCORE")
                for label, split in (("FIT", "fit"), ("TUNE", "tune")):
                    if label in c: scores("CANDIDATE_SCORES", {**group, "split": split}, c[label]); diagnostics({**group, "phase": "CANDIDATE", "split": split}, c[label])
            require(row["FIT"] == winner["FIT"] and row["TUNE"] == winner["TUNE"], "SELECTED_FIT_TUNE_ARTIFACTS_REUSED")
            out["SELECTION"].append({**base, **{k: row[k] for k in ("selected_candidate", "selected_kind", "selected_lambda", "selected_TUNE_MSE", "selected_coefficients", "selected_coefficient_value_sha256", "selection_lock", "selection_rule")}, "E_previously_supervised_by_FIT_future": train["states"][state]["E_previously_supervised_by_FIT_future"], "D_previously_supervised_by_FIT_future": False})
            for label, split in (("FIT", "fit"), ("TUNE", "tune"), ("DEV", "val")): scores("SELECTED_SCORES", {**base, "candidate_id": winner["id"], "split": split}, row[label])
            diagnostics({**base, "candidate_id": winner["id"], "kind": winner["kind"], "lambda": winner["lambda"], "selected": True, "phase": "SELECTED", "split": "val"}, row["DEV"])
        for split in ("tune", "val"):
            cell = values["native_audit"]["new_output_space_diagnostics"][state]["92601"][split]
            out["SPACE_CONTEXT"].append({"state_label": state, "split": split, "minimum_normalized_MSE": cell["minimum_normalized_MSE"], "RAW_complete_MSE": cell["RAW_BOLT_scores_complete_mask"]["normalized"]["mse"], "complete_target_positions": cell["complete_target_positions"], "target_positions": cell["target_positions"], "source_audit": train["matched_l7_native_audit"], "new_geometry_computed": False, "universal_all_FP32_bound": False})
    raw = values["science"]["datasets"]["robin"]["spaces"]["PCA"]["conditions"]["L7_RECON"]["splits"]
    for split in ("fit", "tune", "val"): scores("BASELINES", {"state_label": "RAW_C", "predictor": "RAW_BOLT", "split": split}, raw[split]["RAW_BOLT"])
    for state in STATES:
        for family in FAMILIES:
            for split in ("fit", "tune", "val"):
                for mask in ("all_observed", "complete_C"):
                    for scale in ("normalized", "physical_per_channel_diagnostic"):
                        where = {"state_label": state, "family": family, "split": split, "mask": mask, "scale": scale}; q = find(out["SELECTED_SCORES"], **where); b = find(out["BASELINES"], state_label=state, predictor="BOLT", split=split, mask=mask, scale=scale); r = find(out["BASELINES"], state_label="RAW_C", predictor="RAW_BOLT", split=split, mask=mask, scale=scale)
                        out["PAIRED"].append({**where, "q_MSE": q["mse"], "baseline_MSE": b["mse"], "RAW_MSE": r["mse"], "q_minus_baseline_MSE": q["mse"]-b["mse"], "q_relative_baseline_MSE_change": q["mse"]/b["mse"]-1 if b["mse"] else None, "q_MAE": q["mae"], "baseline_MAE": b["mae"], "RAW_MAE": r["mae"], "q_minus_baseline_MAE": q["mae"]-b["mae"]})
    for stage, field in (("train", "failed_paths"), ("train", "failed_candidate_paths"), ("audit", "partial_paths"), ("cost", "failed_paths")): out["FAILURES"] += [{"stage": stage, "collection": field, **v} for v in values[stage][field]]
    rows = cost["cost"]["robin"]["rows"]; require(len(rows) == 20 and {(r["route"], r["batch"]) for r in rows} == {(r, b) for r in ROUTES for b in (1, 8)}, "TWENTY_COST_CELLS")
    for row in rows:
        sec = row["seconds"]; samples = row["samples"]; route = row["route"]; matching = route.split("_SCALE_")[0]+"_BOLT" if "_SCALE_" in route else route
        require(row["status"] == "COMPLETE" and len(sec) == len(samples) == 5 and all(type(v) in (int, float) and v > 0 for v in sec) and row["median_seconds"] == statistics.median(sec) and row["median_seconds_per_window"] == statistics.median(sec)/row["batch"] and row["matched_Bolt_route"] == matching, "FIVE_POSITIVE_SAMPLES_SAME_E_MEDIANS")
        require([s["repeat"] for s in samples] == list(range(5)), "EXACT_COST_REPEAT_SEQUENCE")
        for field, denominator_route in (("RAW", "RAW_BOLT"), ("matched_Bolt", matching)):
            denominator = find(rows, route=denominator_route, batch=row["batch"]); by_repeat = {s["repeat"]: s["seconds"] for s in denominator["samples"]}; paired = [by_repeat[s["repeat"]] for s in samples]; ratios = [a/b for a, b in zip(sec, paired)]
            require(row["paired_"+field+"_seconds"] == paired and row["paired_"+field+"_ratios"] == ratios and row["paired_"+field+"_median_ratio"] == statistics.median(ratios) and row["ratio_of_"+field+"_medians"] == statistics.median(sec)/statistics.median(denominator["seconds"]), "INDEPENDENT_PAIRED_AND_MEDIAN_RATIOS")
        for index, sample in enumerate(samples):
            require(sample["route"] == route and sample["batch"] == row["batch"] and type(sample["perf_start_ns"]) is type(sample["perf_end_ns"]) is int and (sample["perf_end_ns"]-sample["perf_start_ns"])/1e9 == sample["seconds"] == sec[index], "INTEGER_NS_SAMPLE_BINDING")
            require(type(sample["unix_start_ns"]) is type(sample["unix_end_ns"]) is int and cost["resource_terminal"]["started_unix"]*1e9 <= sample["unix_start_ns"] <= sample["unix_end_ns"] <= cost["resource_terminal"]["ended_unix"]*1e9, "TIMERS_WITHIN_COST_PROCESS_RECEIPT")
        out["WHOLE_COST"].append({k: v for k, v in row.items() if k != "samples"}); out["WHOLE_COST_SAMPLES"] += samples
    for stage in INPUT_SHA:
        resource = values[stage]["resource_terminal"]; require(all(type(resource.get(k)) in (int, float) and resource[k] >= 0 for k in ("process_cpu_seconds", "wall_seconds")), "FOUR_CANONICAL_RESOURCE_RECEIPTS")
        out["RESOURCES"].append({"stage": stage, **resource})
    require(tuple(len(out[k]) for k in TABLES) == COUNTS, "EXACT_FOURTEEN_TABLE_COUNTS")
    return out


def check_files(folder, files):
    expected = {"REPORT_KO.md"}|{"tables/"+k+".csv" for k in TABLES}|{"figures/"+n+"."+ext for n in FIGURES for ext in ("png", "svg")}
    require(len(files) == 23 and {r["path"] for r in files} == expected, "EXACT_TWENTYTHREE_MANIFEST_FILES")
    paths = []
    for item in files:
        path = local_file(item["path"], folder); require(path.is_relative_to(folder), "REPORT_FILE_CONTAINMENT"); actual = record(path)
        require(actual["sha256"] == item["sha256"] and type(item["bytes"]) is int and actual["bytes"] == item["bytes"], "REPORT_FILE_SHA_BYTES"); paths.append(path)
    entries = list(folder.rglob("*")); require(not any(p.is_symlink() for p in entries) and len(set(paths)) == 23, "NO_SYMLINK_OR_DUPLICATE_REPORT_FILES")
    require({p.resolve() for p in entries if p.is_file()} == set(paths)|{folder/"MANIFEST.json"}, "EXACT_REPORT_DIRECTORY_FILES")
    png, svg = common.common.inspect_images(paths); require(png == svg == 4, "FOUR_PNG_FOUR_SVG")
    return paths


def check_markdown(folder, tables, values, bindings=None):
    text = (folder/"REPORT_KO.md").read_text(encoding="utf-8"); links = re.findall(r"(!?)\[[^\]]*\]\(([^)]+)\)", text); images = []
    for marker, target in links:
        target = unquote(target.split("#", 1)[0]); require(target and not re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:|^[/\\]", target), "RELATIVE_MARKDOWN_LINK"); local_file(target, folder)
        if marker: images.append(target)
    require(len(images) == 4 and set(images) == {"figures/"+n+".png" for n in FIGURES}, "FOUR_MARKDOWN_IMAGES")
    def paired(s, f, split): return find(tables["PAIRED"], state_label=s, family=f, split=split, mask="all_observed", scale="normalized")
    identity = sum(r["selected_kind"] == "IDENTITY" for r in tables["SELECTION"]); tune = sum(paired(s, f, "tune")["q_minus_baseline_MSE"] < 0 for s in STATES for f in FAMILIES); dev = sum(paired(s, f, "val")["q_minus_baseline_MSE"] < 0 for s in STATES for f in FAMILIES); raw = sum(paired(s, f, "val")["q_MSE"] < paired(s, f, "val")["RAW_MSE"] for s in STATES for f in FAMILIES)
    require(f"identity 선정은 {identity}/6" in text and f"TUNE {tune}/6, 기노출 DEV {dev}/6" in text and f"Raw보다 DEV MSE가 낮은 조건은 {raw}/6" in text, "MD_IDENTITY_AND_NEGATIVE_EVIDENCE_COUNTS")
    for row in tables["SELECTION"]:
        s, f = row["state_label"], row["family"]; t, d = (paired(s, f, split) for split in ("tune", "val"))
        require(f"- {s} / {f}: {row['selected_candidate']}; TUNE q {t['q_MSE']:.12f}, baseline {t['baseline_MSE']:.12f}, delta {t['q_minus_baseline_MSE']:+.12f} ({100*t['q_relative_baseline_MSE_change']:+.6f}%); DEV q {d['q_MSE']:.12f}, baseline {d['baseline_MSE']:.12f}, delta {d['q_minus_baseline_MSE']:+.12f} ({100*d['q_relative_baseline_MSE_change']:+.6f}%), Raw {d['RAW_MSE']:.12f}." in text, "MD_SIX_SELECTION_NUMBERS_AND_TINY_PERCENT_EFFECTS")
    fit_better = 0
    for s in STATES:
        for f in FAMILIES:
            rows = [r for r in tables["CANDIDATE_SCORES"] if r["state_label"] == s and r["family"] == f and r["split"] == "fit" and r["mask"] == "all_observed" and r["scale"] == "normalized"]
            fit_better += min(r["mse"] for r in rows) < find(rows, candidate_id="IDENTITY")["mse"]
    require(f"actual FIT MSE의 후보 최솟값이 identity보다 낮은 조건은 {fit_better}/6" in text, "MD_ACTUAL_FIT_VERSUS_TRANSFER")
    train = values["train"]; require(f"실제 {train['fit_conditions_completed']} fit 조건에서 SVD 호출은 {train['SVD_fits_completed']}회" in text and f"FIT/TUNE actual FP32 평가는 {train['candidate_FIT_TUNE_evaluations_completed']}개, 선정 DEV는 {train['selected_DEV_evaluations_completed']}개" in text, "MD_ACTUAL_EXECUTION_COUNTS")
    diagnostics = tables["DIAGNOSTICS"]
    require(f"현재 보존된 actual-minus-surrogate max absolute error의 최댓값은 {max(r['surrogate_diagnostics']['max_abs_actual_FP32_minus_linear_surrogate'] for r in diagnostics):.12g}, RMS의 최댓값은 {max(r['surrogate_diagnostics']['RMS_actual_FP32_minus_linear_surrogate'] for r in diagnostics):.12g}다. 독립 FP32 C parity max error의 최댓값은 {max(r['independent_FP32_deployment_parity']['C_prediction']['max_abs'] for r in diagnostics):.12g}" in text, "MD_THREE_DISTINCT_NUMERIC_DIAGNOSTICS")
    for r in tables["SPACE_CONTEXT"]: require(f"- {r['state_label']} / {r['split']}: retained numeric minimum {r['minimum_normalized_MSE']:.12f}, Raw {r['RAW_complete_MSE']:.12f}; complete {r['complete_target_positions']}/{r['target_positions']}; 이전 감사 문맥 재사용." in text, "MD_SIX_REUSED_SPACE_ROWS")
    for r in tables["WHOLE_COST"]: require(f"- B{r['batch']} {r['route']}: {r['median_seconds_per_window']*1000:.6f} ms/window; Raw paired {r['paired_RAW_median_ratio']:.6f}, medians {r['ratio_of_RAW_medians']:.6f}; matching Bolt paired {r['paired_matched_Bolt_median_ratio']:.6f}, medians {r['ratio_of_matched_Bolt_medians']:.6f}." in text, "MD_TWENTY_COST_ROWS_TWO_DENOMINATORS")
    for key, rows in tables.items(): require(f"[{key}.csv](tables/{key}.csv): {len(rows)}행" in text, "MD_FOURTEEN_CSV_COUNTS")
    for key, item in (bindings or {}).items():
        path = os.path.relpath(resolve(item["path"]), folder).replace("\\", "/"); require(f"[{key}]({path}): `{item['sha256']}`" in text, "MD_ALL_SOURCE_PARENT_BINDINGS")
    for token in ("이 작은 보정이 Raw gap을 해결했다고 볼 수 없다", "독립 일반화나 유의한 개선의 근거가 아니다", "실제 FP32 MSE의 정확한 최적해".replace("실제", "actual"), "FIT 미래는 정상 회귀 response", "별도 C intercept나 새 D는 없다", "lag별 z_past[-1-lag] 기준", "a=-1,b=0", "byte-collapse를 보장하지 않는다", "별도 byte-exact 경로", "전체 FIT feature가 정확0", "같은 세 상태의 두 family를6개 독립 자료로 세지 않는다", "lambda0 상대 cutoff1e-12", "계수나 음수 gain을 사후 clip하지 않았다", "모든6잠금·registry 뒤 DEV", "FIT 가중치를 TUNE/DEV 분모로 재사용하지 않는다", "FP64 baseline+Xβ는 이 parity 대상이 아니다", "0으로 가정하지 않는다", "rank가 부족하면 계수를 고유한 잠재 의미나 인과효과로 해석하지 않는다", "새 공간 적합은0", "실제 동결 Bolt를 실행", "5순환은10위치 완전 균형이 아니며", "warm2가 전체 비측정 forward2라는 뜻도 아니다", "내부 finite/shape 검사·sync도 남아 있다", "cached 과학 출력을 반환해 F 비용을 없애지 않았다", "paired median ratio와 ratio of medians", "중복 route의 시간 차이를 서로 다른 학습 방법의 속도 승리로 세지 않는다", "무료 보정으로 부르지 않는다", "Raw-F teacher 증류는 이미 수행", "독립 일반화 검증이 아니다", "strict K-only도 아니다", "JSON metadata만 읽고 NPZ/model/F/학습/TEST/Git을 실행하지 않았다"):
        require(token in text, "MD_SCIENTIFIC_SCOPE:"+token)
    return len(links)


def check_receipts(values):
    checked = []
    require(record(Path(__file__).parent/"continuation_runtime_v1.py")["sha256"] == RUNTIME_SHA, "IMMUTABLE_RECEIPT_RUNTIME")
    for stage in INPUT_SHA:
        receipt = values[stage]["resource_terminal"]
        require(all(type(receipt.get(k)) in (int, float) and math.isfinite(receipt[k]) and receipt[k] >= 0 for k in ("process_cpu_seconds", "wall_seconds", "started_unix", "ended_unix")), "CANONICAL_RESOURCE_RECEIPT_FIELDS")
        saved = parse(metadata(receipt).read_text(encoding="utf-8-sig"))
        common.exact(saved, {k: v for k, v in receipt.items() if k not in ("path", "sha256")}, "EXACT_RECORDED_RECEIPT_CONTENT")
        require(receipt["status"] == values[stage]["status"] and receipt["error"] is None and receipt["gpu_process"] is (stage in ("train", "cost")) and receipt["ended_unix"] >= receipt["started_unix"] and receipt["original_budget_records_unchanged"] is True and receipt["elapsed_or_cumulative_budget_caps"] is None, "RESOURCE_PROCESS_STATUS_AND_SCOPE")
        authorization = parse(metadata(receipt["authorization"]).read_text(encoding="utf-8-sig")); continuation = parse(metadata(receipt["continuation_manifest"]).read_text(encoding="utf-8-sig"))
        require(authorization["status"] == "APPROVED_HUMAN_UNCAPPED_CONTINUATION" and authorization["latest_human_override"] == "상한내수행하지 말고 쭉해줘" and continuation["authorization_sha256"] == receipt["authorization"]["sha256"] and continuation["runtime_sha256"] == RUNTIME_SHA, "RECORDED_AUTHORIZATION_MANIFEST_RUNTIME_CHAIN")
        checked.append({"stage": stage, "receipt": record(receipt["path"]), "authorization": receipt["authorization"], "continuation_manifest": receipt["continuation_manifest"]})
    return checked


def load_inputs(manifest):
    values = {}
    for key, digest in INPUT_SHA.items():
        binding = manifest["inputs"][key]; require(binding["sha256"] == digest and Path(binding["path"]).suffix == ".json", "EXACT_JSON_STAGE_SHA:"+key); values[key] = parse(metadata(binding).read_text(encoding="utf-8-sig")); require(values[key]["status"] == ("PASS" if "audit" in key else "COMPLETE") and values[key]["protected_TEST_accessed"] is False, "FOUR_COMPLETE_PASS_NO_TEST")
    train, audit, cost, cost_audit = (values[k] for k in INPUT_SHA)
    same(audit["diagnostic"], manifest["inputs"]["train"]); same(cost_audit["cost"], manifest["inputs"]["cost"])
    for key, counts in (("audit", AUDIT_COUNTS), ("cost_audit", COST_COUNTS)):
        same(values[key]["audit_source"], manifest["inputs"][key+"_source"])
        require(values[key]["audit_source"]["sha256"] == AUDIT_SOURCE_SHA[key] and all(type(values[key]["totals"].get(k)) is int and values[key]["totals"][k] == v for k, v in counts.items()), "EXACT_AUDITOR_SOURCE_AND_COVERAGE")
    for value in (cost, cost_audit):
        same(value["latent_calibration_diagnostic"], manifest["inputs"]["train"]); same(value["latent_calibration_audit"], manifest["inputs"]["audit"]); same(value["selection_registry"], train["selection_registry"])
    require(train["fit_conditions_completed"] == train["resolved_selection_locks"] == train["selected_conditions_completed"] == train["selected_DEV_evaluations_completed"] == 6 and 0 <= train["SVD_fits_completed"] <= 6 and train["ridge_candidates_generated"] == 42 and train["identity_comparisons_generated"] == 6 and train["candidate_FIT_TUNE_evaluations_completed"] == 96 and train["all_6_locks_before_DEV"] is True and not train["failed_paths"] and not train["failed_candidate_paths"], "SIX_FITS_MAX_SIX_SVD_FORTYEIGHT_CANDIDATES_96FT_SIXDEV")
    require(all(train[k] == 0 for k in ("new_F_forward_calls", "new_F_model_objects_loaded", "new_optimizer_updates", "new_E_D_training_updates")) and all(train[k] is False for k in ("native_head_weights_loaded", "new_backbone_training", "new_F_state_hash_computed", "whole_deployment_cost_measured", "independent_holdout", "strict_K_only")), "CACHED_F_SCIENCE_SCOPE")
    require(tuple(cost["routes"]) == ROUTES and cost["cost_split"] == "tune" and cost["completed_cost_cells"] == 20 and cost["recorded_timer_samples"] == 100 and not cost["failed_paths"] and cost["F_before"] == cost["F_after"] == train["F_parent_signatures_reused"]["before"] == train["F_parent_signatures_reused"]["after"], "ACTUAL_SAME_F_TUNE_COST_20_100")
    for key in PRIOR_KEYS: same(manifest["inputs"][key], train[key])
    for key in EXTRA_BINDINGS:
        if key == "cost_plan": same(manifest["inputs"][key], cost["cost_plan"])
        elif key != "report_utility": same(manifest["inputs"][key], train[key])
    for key, source in (("native", "matched_l7_native_diagnostic"), ("native_audit", "matched_l7_native_audit"), ("science", "causal_lag_diagnostic")):
        values[key] = parse(metadata(train[source]).read_text(encoding="utf-8-sig")); require(values[key]["status"] == ("PASS" if key == "native_audit" else "COMPLETE"), "ACCEPTED_BASELINE_AND_SPACE_METADATA")
    registry = parse(metadata(train["selection_registry"]).read_text(encoding="utf-8-sig")); require(registry["status"] == "ALL_SIX_RESOLVED_BEFORE_DEV" and registry["DEV_opened_before_seal"] is False and len(registry["conditions"]) == 6, "SIX_LOCKS_BEFORE_DEV")
    seen = set()
    for item in registry["conditions"]:
        key = item["state"], item["family"]; require(key in {(s, f) for s in STATES for f in FAMILIES} and key not in seen, "UNIQUE_SIX_LOCKS"); seen.add(key)
        row = train["states"][key[0]]["families"][key[1]]; same(item["selection_lock"], row["selection_lock"]); lock = parse(metadata(item["selection_lock"]).read_text(encoding="utf-8-sig"))
        require(item["status"] == row["status"] == lock["status"] == "COMPLETE" and lock["DEV_opened_before_lock"] is False and lock["selected_candidate"] == selected(row)["id"], "TUNE_ONLY_SEALED_LOCK")
        same(lock["selected_coefficients"], row["selected_coefficients"])
    return values


def verify(folder):
    require(all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for v in (REPORT_SHA, MANIFEST_SHA, SOURCE_SHA, *INPUT_SHA.values(), *AUDIT_SOURCE_SHA.values())), "ACTUAL_CALIBRATION_REPORT_NOT_YET_REGISTERED")
    require(record(common.__file__)["sha256"] == HELPER_SHA and record(common.common.__file__)["sha256"] == common.COMMON_SHA, "IMMUTABLE_REPORT_METADATA_HELPERS")
    folder = resolve(folder); report_path, manifest_path = local_file(folder/"REPORT_KO.md"), local_file(folder/"MANIFEST.json")
    require(record(report_path)["sha256"] == REPORT_SHA and record(manifest_path)["sha256"] == MANIFEST_SHA, "EXACT_ACTUAL_REPORT_MANIFEST")
    manifest = parse(manifest_path.read_text(encoding="utf-8")); require(manifest["status"] == "COMPLETE" and manifest["source"]["sha256"] == SOURCE_SHA and Path(manifest["source"]["path"]).name == "make_fixed_l7_latent_calibration_report_v1.py", "EXACT_EXECUTED_REPORT_SOURCE")
    metadata(manifest["source"]); same(manifest["source"], manifest["inputs"]["report_source"])
    require(set(manifest["inputs"]) == set(INPUT_SHA)|set(PRIOR_KEYS)|set(EXTRA_BINDINGS)|{"audit_source", "cost_audit_source", "report_source"}, "EXACT_REPORT_INPUT_BINDINGS")
    for binding in manifest["inputs"].values(): metadata(binding)
    for key, value in {"PNG_count": 4, "SVG_count": 4, "scientific_NPZ_or_model_read": False, "new_fit_or_training_during_generation": 0, "new_geometry_computed": False, "FP64_surrogate_is_not_FP32_deployment": True, "actual_cost_has_live_F": True, "protected_TEST_accessed": False, "independent_holdout": False, "strict_K_only": False}.items(): common.exact(manifest[key], value, "MANIFEST_SCOPE:"+key)
    values = load_inputs(manifest); receipts = check_receipts(values); tables = expected_tables(values); require(set(manifest["table_row_counts"]) == set(TABLES), "EXACT_MANIFEST_TABLE_NAMES"); paths = check_files(folder, manifest["files"]); fields, refs = {}, 0
    for key, rows in tables.items():
        require(manifest["table_row_counts"][key] == len(rows), "MANIFEST_TABLE_COUNT"); expected = relative_values(rows, folder); fields[key] = check_table(folder/"tables"/(key+".csv"), expected); refs += common.check_reference_paths(expected, folder)
    links = check_markdown(folder, tables, values, manifest["inputs"])
    return {"status": "PASS", "report": record(report_path), "manifest": record(manifest_path), "source": manifest["source"], "inputs": {k: manifest["inputs"][k] for k in INPUT_SHA}, "table_rows_checked": {k: len(v) for k, v in tables.items()}, "CSV_primitive_fields_compared": fields, "files_checked": len(paths), "PNG_count": 4, "SVG_count": 4, "local_links_checked": links, "relative_artifact_path_existence_checks": refs, "resource_receipts_checked": receipts, "verification_scope": "Independent reconstruction of all 14 tables/nested CSV primitives, actual FP32 own-TUNE selection and tiny signed effects, audited coefficients/scales/rank and measured surrogate context, repeated identity cost limitations, integer-ns timings with both independently recomputed denominators, four canonical resource receipt contents and authorization/manifest/runtime bindings, exact metadata/source/file/relative Markdown bindings. No scientific NPZ/model/F/GPU/solver/training/timing replay or image visual review.", "scientific_array_model_F_GPU_solver_training_or_timer_replayed": False, "figure_visual_review_performed_by_this_script": False, "protected_TEST_accessed": False, "audit_source": record(__file__)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--report-dir", required=True); parser.add_argument("--output", required=True); args = parser.parse_args()
    try: result = verify(args.report_dir)
    except Exception as error: result = {"status": "FAILED_PRESERVED", "failure": {"type": type(error).__name__, "message": str(error)}, "audit_source": record(__file__)}
    with resolve(args.output).open("x", encoding="utf-8") as handle: json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")
    print(json.dumps({"status": result["status"], "output": str(resolve(args.output))}), flush=True); raise SystemExit(0 if result["status"] == "PASS" else 2)
