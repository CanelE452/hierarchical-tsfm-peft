"""Independent synthetic metadata packages; no report generator, scientific arrays or live inference."""
from pathlib import Path
from unittest.mock import patch
import copy
import csv
import json
import os
import statistics
import tempfile
import unittest

import audit_fixed_l7_latent_calibration_report_v1 as audit


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8"); return audit.record(path)


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row)) or ["no_failed_paths"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader()
        for row in rows: writer.writerow({k: json.dumps(v, allow_nan=False) if isinstance(v, (dict, list)) else "" if v is None else v for k, v in row.items()})


def score(value, count=100):
    unit = {"mse": value, "mae": value/2, "per_channel_mse": [value]*3, "per_channel_mae": [value/2]*3, "observed_counts": [count]*3, "active_channels": 3}
    return {"normalized": unit, "physical_per_channel_diagnostic": {**unit, "mse": value*4, "mae": value, "per_channel_mse": [value*4]*3, "per_channel_mae": [value]*3}}


def fixture(root):
    payload = root/"unopened.npz"; payload.write_bytes(b"Synthetic existence sentinel; never open this payload")
    binding = {"path": str(payload), "sha256": "a"*64}
    def entry(value, identity=False):
        parity = {"max_abs": 3e-7, "rmse": 1e-8, "pass": True, "atol": 1e-5, "rtol": 2e-6}
        return {"status": "COMPLETE", "artifact": binding, "scores_all_observed": score(value), "scores_complete_mask": score(value+.1, 70), "surrogate_diagnostics": {"max_abs_actual_FP32_minus_linear_surrogate": 0. if identity else 2e-6, "RMS_actual_FP32_minus_linear_surrogate": 0. if identity else 2e-7, "per_channel_max_abs": [0. if identity else 2e-6]*3, "surrogate_zero_error_required": False}, "independent_FP32_deployment_parity": {"latent_future": parity.copy(), "C_prediction": parity.copy()}, "FP64_surrogate_is_not_deployment_parity_target": True, "identity_checks": {"identity_byte_exact": identity, "batch_size": 32}, "coefficient_value_sha256": "b"*64, "input_source": binding, "cached_Bolt_source": binding}
    values = {"train": {"states": {}, "failed_paths": [], "failed_candidate_paths": []}, "audit": {"states": {}, "partial_paths": []}, "native": {"states": {}}, "native_audit": {"new_output_space_diagnostics": {}}, "science": {"status": "COMPLETE", "datasets": {"robin": {"spaces": {"PCA": {"conditions": {"L7_RECON": {"splits": {s: {"RAW_BOLT": entry(.25)} for s in ("fit", "tune", "val")}}}}}}}}, "cost": {"cost": {"robin": {"rows": []}}, "failed_paths": []}, "cost_audit": {}}
    values["train"]["matched_l7_native_audit"] = binding
    for index, state in enumerate(audit.STATES):
        baseline = {"fit": .2+index*.01, "tune": .1+index*.001, "val": .5+index*.01}
        values["native"]["states"][state] = {"Bolt": {split: entry(value) for split, value in baseline.items()}}
        owner = values["train"]["states"][state] = {"E_previously_supervised_by_FIT_future": index > 0, "families": {}}
        values["audit"]["states"][state] = {}; values["native_audit"]["new_output_space_diagnostics"][state] = {"92601": {}}
        for split in ("tune", "val"):
            values["native_audit"]["new_output_space_diagnostics"][state]["92601"][split] = {"minimum_normalized_MSE": .36 if split == "tune" else .3, "RAW_BOLT_scores_complete_mask": score(.35, 70), "complete_target_positions": 70, "target_positions": 100}
        for family in audit.FAMILIES:
            special = index == 2 and family == "SCALE_ONLY"; candidates, verified = [], []
            for ci, lam in enumerate((None, *audit.LAMBDAS)):
                identity = ci == 0; tune = baseline["tune"]+(0. if identity else .002+ci*.0001)
                if special and ci == 7: tune = baseline["tune"]-1e-5
                c = {"id": "IDENTITY" if identity else "RIDGE_"+format(lam, "g"), "kind": "IDENTITY" if identity else "RIDGE", "family": family, "lambda": lam, "status": "COMPLETE", "metadata": {"state": binding, "negative_gain_clipped": False}, "coefficient_artifact": binding, "coefficient_value_sha256": "b"*64, "FIT": entry(baseline["fit"]-(0. if identity else .001+ci*.0001), identity), "TUNE": entry(tune, identity), "TUNE_MSE": tune}
                candidates.append(c); a = [0.]*4 if identity else [-.001, .002, -.003, -.006]; b = [0.]*4
                verified.append({**{k: c[k] for k in ("id", "kind", "family", "lambda", "status", "coefficient_artifact", "coefficient_value_sha256")}, "a_FP64": a, "b_FP64": b, "a_FP32": a[:], "b_FP32": b[:], "coefficients_FP64": a+(b if family == "SCALE_BIAS" else []), "coefficients_FP32": a+(b if family == "SCALE_BIAS" else []), "nonfinite_coefficient_indices": []})
            winner = candidates[-1] if special else candidates[0]; width = 4 if family == "SCALE_ONLY" else 8
            owner["families"][family] = {"status": "COMPLETE", "candidates": candidates, "selected_candidate": winner["id"], "selected_kind": winner["kind"], "selected_lambda": winner["lambda"], "selected_TUNE_MSE": winner["TUNE_MSE"], "selected_coefficients": binding, "selected_coefficient_value_sha256": "b"*64, "FIT": winner["FIT"], "TUNE": winner["TUNE"], "DEV": entry(baseline["val"]+(2e-4 if special else 0.), not special), "fit_metadata": {"SVD_fits": 1, "state": binding}, "fit_artifact": binding, "FIT_input_source": binding, "FIT_cached_Bolt_source": binding, "frozen_state_source": binding, "selection_lock": binding, "selection_rule": "actual TUNE; exact ties identity then larger lambda"}
            values["audit"]["states"][state][family] = {"selected_candidate": winner["id"], "selected_lambda": winner["lambda"], "candidates": verified, "FIT_solver_rank": width, "independently_recomputed_fit": {"column_scales_FP64": [1.]*width, "active_columns": [True]*width, "singularvalues_FP64": [1.]*width, "zero_lambda_numeric_rank": width, "FIT_channel_observed_counts": [100, 90, 80], "row_weight_sum": 1.}}
    rows = values["cost"]["cost"]["robin"]["rows"]
    for batch in (1, 8):
        for route in audit.ROUTES:
            duration = (5, 1, 3, 4, 2) if route == "RAW_BOLT" else (4, 2, 6, 3, 1) if route.endswith("_BOLT") else (2, 2, 3, 4, 1)
            sec = [n/1000 for n in duration]; matched = route.split("_SCALE_")[0]+"_BOLT" if "_SCALE_" in route else route
            samples = [{"route": route, "batch": batch, "repeat": i, "seconds": n/1000, "perf_start_ns": 10**18+i*10**8, "perf_end_ns": 10**18+i*10**8+n*10**6, "unix_start_ns": 10**18+i*10**8, "unix_end_ns": 10**18+i*10**8+n*10**6} for i, n in enumerate(duration)]
            rows.append({"route": route, "batch": batch, "status": "COMPLETE", "seconds": sec, "samples": samples, "median_seconds": statistics.median(sec), "median_seconds_per_window": statistics.median(sec)/batch, "matched_Bolt_route": matched})
    for row in rows:
        for field, route in (("RAW", "RAW_BOLT"), ("matched_Bolt", row["matched_Bolt_route"])):
            denominator = audit.find(rows, route=route, batch=row["batch"]); ratios = [a/b for a, b in zip(row["seconds"], denominator["seconds"])]
            row["paired_"+field+"_seconds"] = denominator["seconds"][:]; row["paired_"+field+"_ratios"] = ratios; row["paired_"+field+"_median_ratio"] = statistics.median(ratios); row["ratio_of_"+field+"_medians"] = statistics.median(row["seconds"])/statistics.median(denominator["seconds"])
    for stage, wall in zip(audit.INPUT_SHA, (2., 1., 3., .5)):
        values[stage]["resource_terminal"] = {"process_cpu_seconds": wall/2, "wall_seconds": wall, "started_unix": 10**9, "ended_unix": 10**9+2}
    values["train"].update(fit_conditions_completed=6, SVD_fits_completed=6, candidate_FIT_TUNE_evaluations_completed=96, selected_DEV_evaluations_completed=6)
    return values


def markdown(folder, tables, values, bindings):
    def paired(state, family, split): return audit.find(tables["PAIRED"], state_label=state, family=family, split=split, mask="all_observed", scale="normalized")
    rows = tables["SELECTION"]; identities = sum(r["selected_kind"] == "IDENTITY" for r in rows)
    text = [f"identity 선정은 {identities}/6; TUNE 1/6, 기노출 DEV 0/6; Raw보다 DEV MSE가 낮은 조건은 0/6", "이 작은 보정이 Raw gap을 해결했다고 볼 수 없다. 독립 일반화나 유의한 개선의 근거가 아니다."]
    for row in rows:
        s, f = row["state_label"], row["family"]; t, d = (paired(s, f, split) for split in ("tune", "val"))
        text.append(f"- {s} / {f}: {row['selected_candidate']}; TUNE q {t['q_MSE']:.12f}, baseline {t['baseline_MSE']:.12f}, delta {t['q_minus_baseline_MSE']:+.12f} ({100*t['q_relative_baseline_MSE_change']:+.6f}%); DEV q {d['q_MSE']:.12f}, baseline {d['baseline_MSE']:.12f}, delta {d['q_minus_baseline_MSE']:+.12f} ({100*d['q_relative_baseline_MSE_change']:+.6f}%), Raw {d['RAW_MSE']:.12f}.")
    text += ["actual FIT MSE의 후보 최솟값이 identity보다 낮은 조건은 6/6", "실제 6 fit 조건에서 SVD 호출은 6회; FIT/TUNE actual FP32 평가는 96개, 선정 DEV는 6개", "actual FP32 MSE의 정확한 최적해 아님; FIT 미래는 정상 회귀 response; 별도 C intercept나 새 D는 없다; lag별 z_past[-1-lag] 기준; a=-1,b=0; byte-collapse를 보장하지 않는다; 별도 byte-exact 경로; 전체 FIT feature가 정확0; 같은 세 상태의 두 family를6개 독립 자료로 세지 않는다; lambda0 상대 cutoff1e-12; 계수나 음수 gain을 사후 clip하지 않았다; 모든6잠금·registry 뒤 DEV; FIT 가중치를 TUNE/DEV 분모로 재사용하지 않는다; FP64 baseline+Xβ는 이 parity 대상이 아니다; surrogate 차이를0으로 가정하지 않는다; rank가 부족하면 계수를 고유한 잠재 의미나 인과효과로 해석하지 않는다; 새 공간 적합은0; 실제 동결 Bolt를 실행; 5순환은10위치 완전 균형이 아니며; warm2가 전체 비측정 forward2라는 뜻도 아니다; 내부 finite/shape 검사·sync도 남아 있다; cached 과학 출력을 반환해 F 비용을 없애지 않았다; paired median ratio와 ratio of medians; 중복 route의 시간 차이를 서로 다른 학습 방법의 속도 승리로 세지 않는다; 무료 보정으로 부르지 않는다; Raw-F teacher 증류는 이미 수행; 독립 일반화 검증이 아니다; strict K-only도 아니다; JSON metadata만 읽고 NPZ/model/F/학습/TEST/Git을 실행하지 않았다"]
    diagnostics = tables["DIAGNOSTICS"]
    text.append(f"현재 보존된 actual-minus-surrogate max absolute error의 최댓값은 {max(r['surrogate_diagnostics']['max_abs_actual_FP32_minus_linear_surrogate'] for r in diagnostics):.12g}, RMS의 최댓값은 {max(r['surrogate_diagnostics']['RMS_actual_FP32_minus_linear_surrogate'] for r in diagnostics):.12g}다. 독립 FP32 C parity max error의 최댓값은 {max(r['independent_FP32_deployment_parity']['C_prediction']['max_abs'] for r in diagnostics):.12g}")
    for r in tables["SPACE_CONTEXT"]: text.append(f"- {r['state_label']} / {r['split']}: retained numeric minimum {r['minimum_normalized_MSE']:.12f}, Raw {r['RAW_complete_MSE']:.12f}; complete {r['complete_target_positions']}/{r['target_positions']}; 이전 감사 문맥 재사용.")
    for r in tables["WHOLE_COST"]: text.append(f"- B{r['batch']} {r['route']}: {r['median_seconds_per_window']*1000:.6f} ms/window; Raw paired {r['paired_RAW_median_ratio']:.6f}, medians {r['ratio_of_RAW_medians']:.6f}; matching Bolt paired {r['paired_matched_Bolt_median_ratio']:.6f}, medians {r['ratio_of_matched_Bolt_medians']:.6f}.")
    text += [f"![Figure](figures/{name}.png)" for name in audit.FIGURES]
    text += [f"[{key}.csv](tables/{key}.csv): {len(rows)}행" for key, rows in tables.items()]
    text += [f"[{key}]({os.path.relpath(item['path'], folder).replace(chr(92), '/')}): `{item['sha256']}`" for key, item in bindings.items()]
    return "\n".join(text)


def package(root):
    from PIL import Image
    values = fixture(root); source = root/"make_fixed_l7_latent_calibration_report_v1.py"; source.write_text("# frozen synthetic report source", encoding="utf-8"); report_source = audit.record(source)
    prior = write_json(root/"prior.json", {"status": "COMPLETE"}); authorization = write_json(root/"authorization.json", {"status": "APPROVED_HUMAN_UNCAPPED_CONTINUATION", "latest_human_override": "상한내수행하지 말고 쭉해줘"})
    continuation = write_json(root/"continuation.json", {"authorization_sha256": authorization["sha256"], "runtime_sha256": audit.RUNTIME_SHA})
    for stage in audit.INPUT_SHA:
        resource = values[stage]["resource_terminal"]; resource.update(status="PASS" if "audit" in stage else "COMPLETE", error=None, gpu_process=stage in ("train", "cost"), original_budget_records_unchanged=True, elapsed_or_cumulative_budget_caps=None, authorization=authorization, continuation_manifest=continuation)
        saved = write_json(root/(stage+"_receipt.json"), resource); values[stage]["resource_terminal"] = {"path": saved["path"], "sha256": saved["sha256"], **resource}
    conditions = []
    for state in audit.STATES:
        for family in audit.FAMILIES:
            row = values["train"]["states"][state]["families"][family]
            row["selection_lock"] = write_json(root/(state+family+"_lock.json"), {"status": "COMPLETE", "DEV_opened_before_lock": False, "selected_candidate": row["selected_candidate"], "selected_coefficients": row["selected_coefficients"]})
            conditions.append({"state": state, "family": family, "status": "COMPLETE", "selection_lock": row["selection_lock"]})
    registry = write_json(root/"registry.json", {"status": "ALL_SIX_RESOLVED_BEFORE_DEV", "DEV_opened_before_seal": False, "conditions": conditions})
    native = write_json(root/"native.json", {"status": "COMPLETE", **values["native"]}); native_audit = write_json(root/"native_audit.json", {"status": "PASS", **values["native_audit"]}); science = write_json(root/"science.json", values["science"])
    train, scientific, cost, cost_audit = (values[k] for k in audit.INPUT_SHA)
    for key in audit.PRIOR_KEYS: train[key] = native if key == "matched_l7_native_diagnostic" else native_audit if key == "matched_l7_native_audit" else science if key == "causal_lag_diagnostic" else prior
    train.update(status="COMPLETE", protected_TEST_accessed=False, resolved_selection_locks=6, selected_conditions_completed=6, ridge_candidates_generated=42, identity_comparisons_generated=6, all_6_locks_before_DEV=True, selection_registry=registry, F_parent_signatures_reused={"before": "same", "after": "same"}, controller_source=report_source, module_source=report_source, plan=prior)
    train.update({k: 0 for k in ("new_F_forward_calls", "new_F_model_objects_loaded", "new_optimizer_updates", "new_E_D_training_updates")}); train.update({k: False for k in ("native_head_weights_loaded", "new_backbone_training", "new_F_state_hash_computed", "whole_deployment_cost_measured", "independent_holdout", "strict_K_only")})
    train_binding = write_json(root/"train.json", train)
    audit_source = root/"science_auditor.py"; audit_source.write_text("# frozen synthetic science auditor", encoding="utf-8"); audit_source = audit.record(audit_source)
    scientific.update(status="PASS", protected_TEST_accessed=False, diagnostic=train_binding, audit_source=audit_source, totals=audit.AUDIT_COUNTS.copy()); audit_binding = write_json(root/"audit.json", scientific)
    cost.update(status="COMPLETE", protected_TEST_accessed=False, latent_calibration_diagnostic=train_binding, latent_calibration_audit=audit_binding, selection_registry=registry, routes=list(audit.ROUTES), cost_split="tune", completed_cost_cells=20, recorded_timer_samples=100, F_before="same", F_after="same", cost_plan=prior)
    cost_binding = write_json(root/"cost.json", cost)
    cost_source = root/"cost_auditor.py"; cost_source.write_text("# frozen synthetic cost auditor", encoding="utf-8"); cost_source = audit.record(cost_source)
    cost_audit.update(status="PASS", protected_TEST_accessed=False, cost=cost_binding, latent_calibration_diagnostic=train_binding, latent_calibration_audit=audit_binding, selection_registry=registry, audit_source=cost_source, totals=audit.COST_COUNTS.copy())
    bindings = {"train": train_binding, "audit": audit_binding, "cost": cost_binding, "cost_audit": write_json(root/"cost_audit.json", cost_audit), **{k: train[k] for k in audit.PRIOR_KEYS}, **{k: train[k] for k in ("controller_source", "module_source", "selection_registry", "plan")}, "cost_plan": prior, "report_utility": report_source, "audit_source": audit_source, "cost_audit_source": cost_source, "report_source": report_source}
    folder = root/"report"; folder.mkdir(); (folder/"tables").mkdir(); (folder/"figures").mkdir(); tables = audit.expected_tables(values)
    for key, rows in tables.items(): write_csv(folder/"tables"/(key+".csv"), audit.relative_values(rows, folder))
    for name in audit.FIGURES:
        Image.new("RGB", (2, 2), "white").save(folder/"figures"/(name+".png")); (folder/"figures"/(name+".svg")).write_text('<svg xmlns="http://www.w3.org/2000/svg"/>', encoding="utf-8")
    (folder/"REPORT_KO.md").write_text(markdown(folder, tables, values, bindings), encoding="utf-8")
    manifest = {"status": "COMPLETE", "source": report_source, "inputs": bindings, "files": [{"path": p.relative_to(folder).as_posix(), "sha256": audit.record(p)["sha256"], "bytes": p.stat().st_size} for p in sorted(folder.rglob("*")) if p.is_file()], "table_row_counts": {k: len(v) for k, v in tables.items()}, "PNG_count": 4, "SVG_count": 4, "scientific_NPZ_or_model_read": False, "new_fit_or_training_during_generation": 0, "new_geometry_computed": False, "FP64_surrogate_is_not_FP32_deployment": True, "actual_cost_has_live_F": True, "protected_TEST_accessed": False, "independent_holdout": False, "strict_K_only": False}
    write_json(folder/"MANIFEST.json", manifest)
    pins = {"REPORT_SHA": audit.record(folder/"REPORT_KO.md")["sha256"], "MANIFEST_SHA": audit.record(folder/"MANIFEST.json")["sha256"], "SOURCE_SHA": report_source["sha256"], "INPUT_SHA": {k: bindings[k]["sha256"] for k in audit.INPUT_SHA}, "AUDIT_SOURCE_SHA": {"audit": audit_source["sha256"], "cost_audit": cost_source["sha256"]}}
    return folder, values, tables, manifest, pins


class FixedCalibrationReportAuditTests(unittest.TestCase):
    def test_closed_gate_and_metadata_payload_guard_precede_reads(self):
        with patch.object(audit, "REPORT_SHA", None), patch.object(audit, "record", side_effect=AssertionError("gate read")), self.assertRaisesRegex(ValueError, "NOT_YET_REGISTERED"): audit.verify("missing")
        with patch.object(audit.common, "record", side_effect=AssertionError("payload opened")), self.assertRaisesRegex(ValueError, "METADATA_ONLY"): audit.metadata({"path": "forbidden.npz", "sha256": "a"*64})

    def test_fourteen_tables_coefficient_rank_masks_identity_and_two_denominators(self):
        with tempfile.TemporaryDirectory() as temp:
            values = fixture(Path(temp)); tables = audit.expected_tables(values)
            self.assertEqual(tuple(len(tables[k]) for k in audit.TABLES), audit.COUNTS)
            self.assertEqual(sum(r["selected_kind"] == "IDENTITY" for r in tables["SELECTION"]), 5)
            self.assertEqual([r["FIT_solver_rank"] for r in tables["SOLVERS"]], [4, 8]*3)
            self.assertEqual(audit.find(tables["RESOURCES"], stage="cost_audit")["wall_seconds"], .5)
            complete = audit.find(tables["CANDIDATE_SCORES"], state_label="RECON_CP0", family="SCALE_ONLY", candidate_id="IDENTITY", split="fit", mask="complete_C", scale="normalized"); self.assertEqual(complete["observed_counts"], [70]*3)
            row = audit.find(tables["WHOLE_COST"], route="RECON_CP0_SCALE_ONLY", batch=8)
            self.assertEqual(row["paired_RAW_median_ratio"], 1.); self.assertEqual(row["ratio_of_RAW_medians"], 2/3)
            self.assertEqual(row["paired_matched_Bolt_median_ratio"], 1.); self.assertEqual(row["ratio_of_matched_Bolt_medians"], 2/3)
            self.assertEqual(tables["WHOLE_COST_SAMPLES"][0]["perf_start_ns"], 10**18)

    def test_identity_then_larger_lambda_exact_ties_and_failed_candidate_exclusion(self):
        with tempfile.TemporaryDirectory() as temp:
            row = fixture(Path(temp))["train"]["states"][audit.STATES[0]]["families"][audit.FAMILIES[0]]
            for c in row["candidates"]: c["TUNE_MSE"] = .1
            row["selected_TUNE_MSE"] = .1; self.assertEqual(audit.selected(row)["id"], "IDENTITY")
            row["candidates"][0]["status"] = "FAILED_DEPLOYMENT_PARITY_PRESERVED"; row.update(selected_candidate="RIDGE_10", selected_kind="RIDGE", selected_lambda=10.)
            self.assertEqual(audit.selected(row)["lambda"], 10.)
            row["selected_lambda"] = 1.
            with self.assertRaisesRegex(ValueError, "EXACT_TIE"): audit.selected(row)

    def test_ratio_integer_ns_rank_parity_and_canonical_resource_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            original = fixture(Path(temp))
            changes = ((lambda v: v["cost"]["cost"]["robin"]["rows"][2].update(paired_RAW_median_ratio=99.), "PAIRED"), (lambda v: v["cost"]["cost"]["robin"]["rows"][2].update(matched_Bolt_route="E_ONLY_s92602_BOLT"), "SAME_E"), (lambda v: v["cost"]["cost"]["robin"]["rows"][0]["samples"][0].update(perf_end_ns=10**18+5000001), "INTEGER_NS"), (lambda v: v["cost"]["cost"]["robin"]["rows"][0]["samples"][0].update(unix_start_ns=0), "RECEIPT"), (lambda v: v["audit"]["states"][audit.STATES[0]][audit.FAMILIES[0]].update(FIT_solver_rank=3), "RANK"), (lambda v: v["cost_audit"].update(resource_terminal={"cpu_seconds": .2, "wall_seconds": .5}), "RESOURCE"), (lambda v: v["train"]["states"][audit.STATES[0]]["families"][audit.FAMILIES[0]]["candidates"][0]["FIT"]["independent_FP32_deployment_parity"]["C_prediction"].update(atol=1.), "PARITY"))
            for change, label in changes:
                wrong = copy.deepcopy(original); change(wrong)
                with self.subTest(label=label), self.assertRaisesRegex(ValueError, label): audit.expected_tables(wrong)
            for raw in ('{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'):
                with self.subTest(raw=raw), self.assertRaises(ValueError): audit.parse(raw)

    def test_csv_exact_nested_primitives_no_extra_columns_bool_or_ns_rounding(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"x.csv"; rows = [{"nested": {"flag": False, "v": [1., None]}, "value": 2., "ns": 10**18}]; write_csv(path, rows); self.assertGreater(audit.check_table(path, rows), 0)
            for row in ({**rows[0], "nested": {"flag": 0, "v": [1., None]}}, {**rows[0], "ns": 10**18+1}, {**rows[0], "extra": 1}, {**rows[0], "value": 9.}):
                write_csv(path, [row])
                with self.assertRaises(ValueError): audit.check_table(path, rows)
            write_csv(path, rows); path.write_text(path.read_text(encoding="utf-8-sig").replace("2.0", "nan"), encoding="utf-8")
            with self.assertRaises(ValueError): audit.check_table(path, rows)
            write_csv(path, []); self.assertEqual(audit.check_table(path, []), 0)

    def test_full_package_verification_without_opening_scientific_payload(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(audit.common, "ROOT", Path(temp)):
            folder, values, tables, manifest, pins = package(Path(temp)); original_open = Path.open
            def guarded(path, *args, **kwargs):
                if path.suffix == ".npz": raise AssertionError("Scientific payload opened")
                return original_open(path, *args, **kwargs)
            with patch.multiple(audit, **pins), patch.object(Path, "open", guarded): result = audit.verify(folder)
            self.assertEqual(result["status"], "PASS"); self.assertEqual((result["files_checked"], result["PNG_count"], result["SVG_count"]), (23, 4, 4)); self.assertEqual(len(result["resource_receipts_checked"]), 4)
            self.assertFalse(result["scientific_array_model_F_GPU_solver_training_or_timer_replayed"])
            Path(manifest["source"]["path"]).write_text("changed", encoding="utf-8")
            with patch.multiple(audit, **pins), self.assertRaisesRegex(ValueError, "METADATA_SHA"): audit.verify(folder)

    def test_markdown_numbers_limits_relative_links_and_four_images(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(audit.common, "ROOT", Path(temp)):
            folder, values, tables, manifest, pins = package(Path(temp)); report = folder/"REPORT_KO.md"; text = report.read_text(encoding="utf-8")
            self.assertGreater(audit.check_markdown(folder, tables, values, manifest["inputs"]), 40)
            changes = (("TUNE 1/6", "TUNE 0/6"), ("0.500000000000", "0.510000000000"), ("(-0.009804%)", "(-9.804000%)"), ("(tables/PAIRED.csv)", "(%2Foutside.csv)"), ("figures/01_selected_errors.png", "../missing.png"), ("별도 byte-exact 경로", ""), ("중복 route의 시간 차이를 서로 다른 학습 방법의 속도 승리로 세지 않는다", ""))
            for before, after in changes:
                self.assertIn(before, text); report.write_text(text.replace(before, after, 1), encoding="utf-8")
                with self.subTest(before=before), self.assertRaises(ValueError): audit.check_markdown(folder, tables, values, manifest["inputs"])

    def test_resource_contents_and_authorization_chain_are_bound(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(audit.common, "ROOT", Path(temp)):
            folder, values, tables, manifest, pins = package(Path(temp)); self.assertEqual(len(audit.check_receipts(values)), 4)
            wrong = copy.deepcopy(values); wrong["audit"]["resource_terminal"]["wall_seconds"] = .9
            with self.assertRaisesRegex(ValueError, "RECEIPT_CONTENT"): audit.check_receipts(wrong)
            wrong = copy.deepcopy(values); wrong["cost_audit"]["resource_terminal"].pop("process_cpu_seconds")
            with self.assertRaisesRegex(ValueError, "CANONICAL"): audit.check_receipts(wrong)
            path = Path(values["train"]["resource_terminal"]["authorization"]["path"]); path.write_text('{"status":"wrong"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "METADATA_SHA"): audit.check_receipts(values)

    def test_independent_auditor_coverage_and_parent_source_bindings(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(audit.common, "ROOT", Path(temp)):
            folder, values, tables, manifest, pins = package(Path(temp))
            with patch.multiple(audit, **pins): self.assertEqual(audit.load_inputs(manifest)["audit"]["status"], "PASS")
            for stage, field, value in (("audit", "score_fields_checked", 1), ("cost_audit", "canonical_TUNE_bridges_checked", True)):
                wrong_values = copy.deepcopy(values[stage]); wrong_values["totals"][field] = value
                wrong_manifest = copy.deepcopy(manifest); binding = write_json(Path(temp)/(stage+"_wrong.json"), wrong_values); wrong_manifest["inputs"][stage] = binding
                wrong_pins = copy.deepcopy(pins); wrong_pins["INPUT_SHA"][stage] = binding["sha256"]
                with self.subTest(stage=stage), patch.multiple(audit, **wrong_pins), self.assertRaisesRegex(ValueError, "COVERAGE"): audit.load_inputs(wrong_manifest)
            wrong = copy.deepcopy(manifest); wrong["inputs"]["module_source"] = wrong["inputs"]["cost_audit_source"]
            with patch.multiple(audit, **pins), self.assertRaisesRegex(ValueError, "PARENT_BINDING"): audit.load_inputs(wrong)

    def test_manifest_paths_image_format_and_symlinks_rejected(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(audit.common, "ROOT", Path(temp)):
            root = Path(temp); folder, values, tables, manifest, pins = package(root)
            wrong = copy.deepcopy(manifest["files"]); wrong[0]["path"] = "../prior.json"
            with self.assertRaisesRegex(ValueError, "TWENTYTHREE"): audit.check_files(folder, wrong)
            with self.assertRaisesRegex(ValueError, "CONTAINMENT"): audit.local_file(root.parent/"outside.json")
            image = folder/"figures"/(audit.FIGURES[0]+".png"); image.write_bytes(b"not PNG")
            wrong = copy.deepcopy(manifest["files"])
            for item in wrong:
                if item["path"] == image.relative_to(folder).as_posix(): item.update(sha256=audit.record(image)["sha256"], bytes=image.stat().st_size)
            with self.assertRaises(Exception): audit.check_files(folder, wrong)
            link = root/"link.json"
            try: link.symlink_to(root/"prior.json")
            except (OSError, NotImplementedError): return
            with self.assertRaisesRegex(ValueError, "SYMLINK"): audit.metadata({"path": str(link), "sha256": audit.record(root/"prior.json")["sha256"]})


if __name__ == "__main__": unittest.main()
