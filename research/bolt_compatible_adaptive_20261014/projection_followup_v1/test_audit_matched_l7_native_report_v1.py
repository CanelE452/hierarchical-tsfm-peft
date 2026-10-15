"""Synthetic metadata and file fixtures only; no generator, model, scientific NPZ or live cost execution."""
from pathlib import Path
from unittest.mock import patch
import copy
import csv
import json
import os
import statistics
import tempfile
import unittest

import audit_matched_l7_native_report_v1 as audit


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8"); return audit.record(path)


def score(value, count=100):
    unit = {"mse": value, "mae": value/2, "per_channel_mse": [value]*3, "per_channel_mae": [value/2]*3, "observed_counts": [count]*3, "active_channels": 3}
    return {"normalized": unit, "physical_per_channel_diagnostic": {**unit, "mse": value*4, "mae": value, "per_channel_mse": [value*4]*3, "per_channel_mae": [value]*3}}


def fixture(folder):
    path = folder/"unopened.npz"; path.write_text("Existence only; must never be opened", encoding="utf-8"); binding = {"path": str(path), "sha256": "a"*64}
    def entry(value): return {"artifact": binding, "scores_all_observed": score(value), "scores_complete_mask": score(value+.1, 70)}
    values = {"train": {"states": {}, "failed_paths": [], "optional_NumPy_context_parity_checks": 3, "optional_NumPy_context_parity_failures": [], "resource_terminal": {"wall_seconds": 2., "cpu_seconds": 1.5}}, "audit": {"new_output_space_diagnostics": {}, "partial_paths": [], "new_output_space_diagnostic_failures": [], "optional_NumPy_context_checks": [{"split": s, "registered_tolerance_pass": s != "val", "primary_gate": False} for s in ("fit", "tune", "val")], "resource": {"wall_seconds": 1.}}, "cost": {"cost": {"robin": {"rows": []}}, "failed_paths": [], "resource_terminal": {"wall_seconds": 3.}}, "cost_audit": {"resource_terminal": {"wall_seconds": .5, "cpu_seconds": .25}}, "science": {"status": "COMPLETE", "datasets": {"robin": {"spaces": {"PCA": {"conditions": {"L7_RECON": {"splits": {s: {"RAW_BOLT": entry(.25)} for s in ("fit", "tune", "val")}}}}}}}}}
    for index, state in enumerate(audit.STATES):
        owner = values["train"]["states"][state] = {"native": {}, "Bolt": {s: entry(.5+index*.02) for s in ("fit", "tune", "val")}, "frozen_state_source": binding, "E_previously_supervised_by_FIT_future": index > 0, "D_previously_supervised_by_FIT_future": False}
        values["audit"]["new_output_space_diagnostics"][state] = {}
        for seed in audit.SEEDS:
            log = folder/(state+str(seed)+".jsonl")
            events = [{"stage": "TRAINING_UPDATE", "update": n, "loss": .2, "head_weight_gradient_norm_before_clip": .3, "head_bias_gradient_norm_before_clip": .4, "global_gradient_norm_before_clip": .5, "clipped": False, "gradient_clipping_parameter_scope": "HEAD_ONLY", **audit.FLAGS} for n in range(1, 601)]
            log.write_text("\n".join(json.dumps(v) for v in events), encoding="utf-8")
            cps = [{"update": n, "state": binding, "tune": score(value)["normalized"], "frozen_status": copy.deepcopy(audit.FLAGS), **entry(value)} for n, value in zip((0, 100, 200, 400, 600), (3., 1., .8, .8, 1.2))]
            row = {"status": "COMPLETE", "checkpoints": cps, "selected_update": 200, "selected_state": binding, "selected_head_state_sha256": "b"*64, "order_sha256": "c"*64, "updates_log": audit.record(log), "selection_lock": binding, "loss": "fixed entire FIT counts", "FIT_channel_observed_counts": [100, 90, 80], "global_clipping_fraction": 0., "post_optimizer_frozen_flag_checks": 600, "frozen_state_checkpoint_byte_checks": 5, "final_frozen_status": copy.deepcopy(audit.FLAGS), "training_head_values_changed": True, "selected_head_nonzero": True, "frozen_bytes_checked_at_CP_and_final_only": True, "independent_per_update_frozen_bytes_replayed": False, "gradient_clipping_parameter_scope": "HEAD_ONLY", "probe": {"status": "PASS", "optimizer_updates": 0, "artifact": binding}, "FIT": entry(.9), "DEV": entry(1.+index*.02)}
            owner["native"][str(seed)] = row; values["audit"]["new_output_space_diagnostics"][state][str(seed)] = {}
            for split in ("tune", "val"):
                cell = {"status": "COMPLETE", "minimum_normalized_MSE": .3, "complete_target_positions": 20, "target_positions": 24, "joint_unknown_future_coordinates": 48, "joint_output_coordinates_before_mask": 144, "lag_slope_parameters_per_C": 7, "joint_metadata": {"numeric_rank_per_origin": [2, 2], "rcond": 1e-15}, "minimum_solution_scores_complete_C": score(.3, 70), "RAW_BOLT_scores_complete_mask": score(.25, 70), "current_C_persistence_scores_complete_mask": score(.9, 70), "selected_E_previously_supervised_by_FIT_future_targets": index > 0, "selected_D_previously_supervised_by_FIT_future_targets": False, "selected_native_head_previously_supervised_by_FIT_future_targets": True, "native_head_training_procedure_used_FIT_future_targets": True, "EY_encoding_backend": "NumPy FP32 GEMM", "EY_incomplete_C_horizons_filled_with_current_C": 4, "artifact": binding, "selected_state": binding, "selected_forecast_source": binding, "input_source": binding, "all_real_latent_or_all_FP32_lower_bound_claimed": False, "independent_external_replay_of_new_diagnostic_claimed": False, "methods": {}}
                for method, value in (("ACTUAL_NATIVE", 1.), ("EY", .5), ("FLAT", .7)):
                    cell["methods"][method] = {"scores_complete_mask": score(value, 70), "scores_all_observed": score(value+.01), "real_trace_bound_inequality_pass": False, "FP32_bound_with_measured_allowance_pass": True}
                values["audit"]["new_output_space_diagnostics"][state][str(seed)][split] = cell
    rows = values["cost"]["cost"]["robin"]["rows"]
    for batch in (1, 8):
        for route in audit.ROUTES:
            duration = (5, 1, 3, 4, 2) if route == "RAW_BOLT" else (4, 2, 6, 3, 1) if "_NATIVE_" not in route else (2, 2, 3, 4, 1)
            seconds = [v/1000 for v in duration]; matching = "RAW_BOLT" if route == "RAW_BOLT" else route.split("_NATIVE_")[0]+"_BOLT" if "_NATIVE_" in route else route
            samples = [{"route": route, "batch": batch, "repeat": i, "seconds": v/1000, "perf_start_ns": 10**18+i*10**8, "perf_end_ns": 10**18+i*10**8+v*10**6} for i, v in enumerate(duration)]
            rows.append({"route": route, "batch": batch, "status": "COMPLETE", "seconds": seconds, "samples": samples, "median_seconds": statistics.median(seconds), "median_seconds_per_window": statistics.median(seconds)/batch, "matched_Bolt_route": matching})
    for row in rows:
        for field, route in (("RAW", "RAW_BOLT"), ("matched_Bolt", row["matched_Bolt_route"])):
            denominator = audit.find(rows, route=route, batch=row["batch"]); ratios = [a/b for a, b in zip(row["seconds"], denominator["seconds"])]
            row["paired_"+field+"_seconds"] = denominator["seconds"][:]; row["paired_"+field+"_ratios"] = ratios; row["paired_"+field+"_median_ratio"] = statistics.median(ratios); row["ratio_of_"+field+"_medians"] = statistics.median(row["seconds"])/statistics.median(denominator["seconds"])
    return values


def write_csv(path, rows):
    names = list(dict.fromkeys(k for row in rows for k in row)) or ["no_failed_paths"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names); writer.writeheader()
        for row in rows: writer.writerow({k: json.dumps(v, allow_nan=False) if isinstance(v, (list, dict)) else "" if v is None else v for k, v in row.items()})


def markdown(folder, tables, values, bindings=None):
    text = ["TUNE 6/6, 기노출 DEV 6/6; Raw보다 DEV MSE가 낮은 native는 0/6", "선정 CP100은 0/6이고 clipping fraction은 0.000000~0.000000", "NumPy 문맥 검사 3개, 기록된 문맥 실패 0개"]
    text.append("새 6fit×600=3,600 update·30CP·선정 FIT/DEV12예측; E/D/F 학습은0; 독립 가중치 초기화가 아니다; 같은 전체 학습 계산량 대조도 아니다; 고정 전체 FIT 채널 관측수; 모든6개 성공/실패 잠금; 최종 C는 현재 C persistence와 같지 않다; 새 F forward는0; 새 matched RECON C decode3개; C byte 보존6개; 매 update bytes·loss·backward·optimizer를 독립 재현한 검사는 아니다; 12개 독립 자료가 아니다; 세 번째 외부 재현은 아니다; 각 지표의 minimum이 아니다; real/FP32 false-bound flags; 명시적 warmup2는 모든 비측정 forward가2회라는 뜻이 아니다; 5순환은 10개 위치의 완전 균형이 아니다; 내부 finite/shape 검사·sync는 timer 안; median of paired ratios와 ratio of medians; 독립 자료나 독점 peak로 바꾸지 않는다; 독립 일반화 검증이 아니다; strict K-only도 아니다; JSON/JSONL metadata만 읽고")
    text.append("z_past[-1-l]; C_current + sum_l (z_last - z_past[-1-l]) D_l")
    tune = [audit.find(tables["SPACE_CELLS"], state=s, seed=audit.SEEDS[0], split="tune") for s in audit.STATES]
    dev = [audit.find(tables["SPACE_CELLS"], state=s, seed=audit.SEEDS[0], split="val") for s in audit.STATES]
    text.append(f"TUNE retained numeric minimum이 Raw보다 높은 고정 E/D는 {sum(r['minimum_normalized_MSE']>r['RAW_complete_MSE'] for r in tune)}/3이고, DEV minimum이 Raw보다 낮은 고정 E/D는 {sum(r['minimum_normalized_MSE']<r['RAW_complete_MSE'] for r in dev)}/3")
    for state in audit.STATES:
        for seed in audit.SEEDS:
            row = audit.find(tables["CONDITIONS"], state=state, seed=seed)
            t, d = (audit.find(tables["PAIRED"], state=state, seed=seed, split=s, mask="all_observed", scale="normalized") for s in ("tune", "val"))
            text.append(f"- {state} / {seed}: CP{row['selected_update']}; TUNE native {t['native_MSE']:.10f}, Bolt {t['Bolt_MSE']:.10f}; DEV native {d['native_MSE']:.10f}, Bolt {d['Bolt_MSE']:.10f}, Raw {d['RAW_MSE']:.10f}.")
    for r in tables["SPACE_CELLS"]: text.append(f"- {r['state']} / {r['seed']} / {r['split']}: native {r['native_complete_MSE']:.10f}, Bolt {r['matched_Bolt_complete_MSE']:.10f}, E(Y) {r['EY_complete_MSE']:.10f}, minimum {r['minimum_normalized_MSE']:.10f}, Raw {r['RAW_complete_MSE']:.10f}; complete {r['complete_target_positions']}/{r['target_positions']}.")
    for r in tables["WHOLE_COST"]: text.append(f"- B{r['batch']} {r['route']}: {r['median_seconds_per_window']*1000:.6f} ms/window; Raw paired {r['paired_RAW_median_ratio']:.6f}, medians {r['ratio_of_RAW_medians']:.6f}; same-E Bolt paired {r['paired_matched_Bolt_median_ratio']:.6f}, medians {r['ratio_of_matched_Bolt_medians']:.6f}.")
    text += [f"![Figure](figures/{name}.png)" for name in audit.FIGURES]
    text += [f"[{key}.csv](tables/{key}.csv): {len(rows)}행" for key, rows in tables.items()]
    text += [f"[{key}]({os.path.relpath(item['path'], folder).replace(chr(92), '/')}): `{item['sha256']}`" for key, item in (bindings or {}).items()]
    return "\n".join(text)


def package(root):
    from PIL import Image
    values = fixture(root); source = root/"make_matched_l7_native_report_v2.py"; source.write_text("# frozen synthetic source", encoding="utf-8"); source_binding = audit.record(source)
    dummy = root/"prior.json"; prior = write_json(dummy, {"status": "COMPLETE"}); science = write_json(root/"science.json", values["science"])
    registry = []
    for state in audit.STATES:
        for seed in audit.SEEDS:
            row = values["train"]["states"][state]["native"][str(seed)]
            row["selection_lock"] = write_json(root/(state+str(seed)+"_lock.json"), {"status": "COMPLETE", "DEV_opened_before_lock": False, "selected_update": row["selected_update"], "selected_state": row["selected_state"]})
            registry.append({"state": state, "seed": seed, "status": "COMPLETE", "selection_lock": row["selection_lock"]})
    registry_binding = write_json(root/"registry.json", {"status": "ALL_SIX_RESOLVED_BEFORE_DEV", "DEV_opened_before_seal": False, "conditions": registry})
    train, scientific, cost, cost_audit = (values[k] for k in audit.INPUT_SHA)
    for key in audit.PRIOR_KEYS: train[key] = registry_binding if key == "selection_registry" else science if key == "causal_lag_diagnostic" else prior
    train.update(status="COMPLETE", protected_TEST_accessed=False, native_training_fits=6, resolved_selection_locks=6, training_updates_completed=3600, checkpoint_states_saved=30, all_6_locks_before_DEV=True, matched_Bolt_C_decodes=9, cached_Bolt_contexts_reused=9, E0_new_matched_RECON_C_decodes=3, selected_E_only_Bolt_C_byte_checks=6, F_parent_signatures_reused={"before": "same", "after": "same"}, controller_source=source_binding, native_source=source_binding)
    train.update({k: 0 for k in ("new_F_forward_calls", "new_F_model_objects_loaded", "new_E_training_updates", "new_D_training_updates", "probe_optimizer_updates")})
    train_binding = write_json(root/"train.json", train)
    scientific.update(status="PASS", protected_TEST_accessed=False, diagnostic=train_binding, audit_source=source_binding, training_verification_status="PASS", new_space_diagnostic_status="COMPLETE", **{k: train[k] for k in audit.PRIOR_KEYS})
    audit_binding = write_json(root/"audit.json", scientific)
    cost.update(status="COMPLETE", protected_TEST_accessed=False, matched_l7_native_diagnostic=train_binding, matched_l7_native_audit=audit_binding, selection_registry=registry_binding, routes=list(audit.ROUTES), cost_split="tune", completed_cost_cells=20, recorded_timer_samples=100, native_predictor_loaded_or_timed=True, cached_future_returned_instead_of_actual_F_inside_timer=False, new_training_updates=0, F_before="same", F_after="same", selected_states_before={str(i): "same" for i in range(9)}, selected_states_after={str(i): "same" for i in range(9)}, native_heads_before={str(i): "same" for i in range(6)}, native_heads_after={str(i): "same" for i in range(6)}, cost_plan=prior)
    cost_binding = write_json(root/"cost.json", cost)
    cost_audit.update(status="PASS", protected_TEST_accessed=False, cost=cost_binding, matched_l7_native_diagnostic=train_binding, matched_l7_native_audit=audit_binding, selection_registry=registry_binding, audit_source=source_binding)
    bindings = {"train": train_binding, "audit": audit_binding, "cost": cost_binding, "cost_audit": write_json(root/"cost_audit.json", cost_audit), "audit_source": source_binding, "cost_audit_source": source_binding, **{k: train[k] for k in audit.PRIOR_KEYS}, "cost_plan": prior, "controller_source": source_binding, "native_source": source_binding, "report_utility": source_binding, "report_source": source_binding}
    folder = root/"report"; folder.mkdir(); (folder/"tables").mkdir(); (folder/"figures").mkdir(); tables = audit.expected_tables(values)
    for key, rows in tables.items(): write_csv(folder/"tables"/(key+".csv"), audit.relative_values(rows, folder))
    for name in audit.FIGURES:
        Image.new("RGB", (2, 2), "white").save(folder/"figures"/(name+".png")); (folder/"figures"/(name+".svg")).write_text('<svg xmlns="http://www.w3.org/2000/svg"/>', encoding="utf-8")
    (folder/"REPORT_KO.md").write_text(markdown(folder, tables, values, bindings), encoding="utf-8")
    manifest = {"status": "COMPLETE", "source": source_binding, "inputs": bindings, "files": [{"path": p.relative_to(folder).as_posix(), "sha256": audit.record(p)["sha256"], "bytes": p.stat().st_size} for p in sorted(folder.rglob("*")) if p.is_file()], "table_row_counts": {k: len(v) for k, v in tables.items()}, "PNG_count": 4, "SVG_count": 4, "scientific_NPZ_or_model_read": False, "new_updates_during_generation": 0, "new_head_fits_in_source": 6, "new_head_updates_in_source": 3600, "new_F_forward_calls_in_scientific_source": 0, "cost_uses_live_F_and_native": True, "geometry_cells_reuse_only_three_frozen_E_D_states": True, "independent_holdout": False, "strict_K_only": False, "protected_TEST_accessed": False}
    write_json(folder/"MANIFEST.json", manifest)
    pins = {"REPORT_SHA": audit.record(folder/"REPORT_KO.md")["sha256"], "MANIFEST_SHA": audit.record(folder/"MANIFEST.json")["sha256"], "SOURCE_SHA": source_binding["sha256"], "INPUT_SHA": {k: bindings[k]["sha256"] for k in audit.INPUT_SHA}}
    return folder, values, tables, manifest, pins


class MatchedNativeReportAuditTests(unittest.TestCase):
    def test_closed_gate_precedes_reads_and_npz_rejected_before_open(self):
        with patch.object(audit, "REPORT_SHA", None), patch.object(audit, "record", side_effect=AssertionError("closed")), self.assertRaisesRegex(ValueError, "NOT_YET_REGISTERED"): audit.verify("missing")
        with patch.object(audit, "record", side_effect=AssertionError("NPZ opened")), self.assertRaisesRegex(ValueError, "METADATA_ONLY"): audit.metadata({"path": "forbidden.npz", "sha256": "a"*64})

    def test_tables_counts_masks_flags_false_bounds_and_resources(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(audit, "ROOT", Path(temp)):
            tables = audit.expected_tables(fixture(Path(temp))); self.assertEqual(tuple(len(tables[k]) for k in audit.TABLES), audit.COUNTS)
            self.assertTrue(all(type(row["state"]) is str and type(row["checkpoint_state"]) is dict for row in tables["CHECKPOINTS"]))
            row = audit.find(tables["SCORES"], state="RECON_CP0", predictor="NATIVE", seed=92601, split="tune", mask="complete_C", scale="normalized"); self.assertEqual(row["observed_counts"], [70]*3)
            self.assertFalse(tables["SPACE_METHODS"][0]["real_trace_bound_inequality_pass"]); self.assertEqual(tables["RESOURCES"][-1]["wall_seconds"], .5)
            cost = audit.find(tables["WHOLE_COST"], route="RECON_CP0_NATIVE_s92601", batch=8); self.assertEqual(cost["ratio_of_RAW_medians"], 2/3); self.assertEqual(cost["paired_RAW_median_ratio"], 1.)

    def test_source_invariants_cost_denominators_ns_and_resource_tamper(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(audit, "ROOT", Path(temp)):
            values = fixture(Path(temp))
            changes = ((lambda v: v["train"]["states"]["RECON_CP0"]["native"]["92601"].update(selected_update=400), "EARLIEST"), (lambda v: v["cost"]["cost"]["robin"]["rows"][2].update(matched_Bolt_route="E_ONLY_s92601_BOLT"), "SAME_E"), (lambda v: v["cost"]["cost"]["robin"]["rows"][0]["samples"][0].update(perf_end_ns=10**18+5000001), "INTEGER_NS"), (lambda v: v["cost_audit"].update(resource_terminal={}), "RESOURCE"), (lambda v: v["audit"]["optional_NumPy_context_checks"][0].update(primary_gate=True), "OPTIONAL_CONTEXT"), (lambda v: v["cost"]["cost"]["robin"]["rows"][0]["paired_RAW_ratios"].__setitem__(0, 99.), "PAIRED"))
            for change, label in changes:
                wrong = copy.deepcopy(values); change(wrong)
                with self.subTest(label=label), self.assertRaisesRegex(ValueError, label): audit.expected_tables(wrong)
            for text in ('{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'):
                with self.subTest(text=text), self.assertRaises(ValueError): audit.parse(text)

    def test_exact_csv_nested_primitives_header_bool_and_nan(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); rows = [{"nested": {"flag": False, "v": [1., None]}, "value": 2., "ns": 10**18}]; path = root/"x.csv"; write_csv(path, rows)
            self.assertGreater(audit.check_table(path, rows), 0)
            for wrong in ([{**rows[0], "nested": {"flag": 0, "v": [1., None]}}], [{**rows[0], "nested": {"flag": False, "v": [1., None], "extra": 1}}], [{**rows[0], "value": 3.}], [{**rows[0], "ns": 10**18+1}]):
                write_csv(path, wrong)
                with self.assertRaises(ValueError): audit.check_table(path, rows)
            write_csv(path, rows); path.write_text(path.read_text(encoding="utf-8-sig").replace("2.0", "nan"), encoding="utf-8")
            with self.assertRaises(ValueError): audit.check_table(path, rows)
            write_csv(path, []); self.assertEqual(audit.check_table(path, []), 0)

    def test_complete_synthetic_package_no_scientific_payload_open_and_tamper(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(audit, "ROOT", Path(temp)):
            folder, values, tables, manifest, pins = package(Path(temp)); original_open = Path.open
            def guarded_open(path, *args, **kwargs):
                if path.suffix == ".npz": raise AssertionError("Scientific payload was opened")
                return original_open(path, *args, **kwargs)
            with patch.multiple(audit, **pins), patch.object(Path, "open", guarded_open):
                result = audit.verify(folder)
            self.assertEqual(result["status"], "PASS"); self.assertEqual((result["files_checked"], result["PNG_count"], result["SVG_count"]), (22, 4, 4))
            source = Path(manifest["source"]["path"]); source.write_text("tampered", encoding="utf-8")
            with patch.multiple(audit, **pins), self.assertRaisesRegex(ValueError, "METADATA_SHA"): audit.verify(folder)

    def test_manifest_files_links_unsafe_paths_and_symlinks(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(audit, "ROOT", Path(temp)):
            root = Path(temp); folder, values, tables, manifest, pins = package(root)
            self.assertGreater(audit.check_markdown(folder, tables, values, manifest["inputs"]), 17)
            text = (folder/"REPORT_KO.md").read_text(encoding="utf-8")
            for wrong in (text.replace("TUNE 6/6", "TUNE 5/6", 1), text.replace("0.8000000000", "0.8100000000", 1), text.replace("(tables/SCORES.csv)", "(%2Foutside.csv)", 1), text.replace("[Figure]", "[Other]", 1).replace("figures/01_actual_errors.png", "../missing.png", 1), text.replace("독립 일반화 검증이 아니다", ""), text.replace("높은 고정 E/D는 3/3", "높은 고정 E/D는 0/3")):
                (folder/"REPORT_KO.md").write_text(wrong, encoding="utf-8")
                with self.assertRaises(ValueError): audit.check_markdown(folder, tables, values, manifest["inputs"])
            with self.assertRaisesRegex(ValueError, "CONTAINMENT"): audit.local_file(root.parent/"outside.json")
            bad = copy.deepcopy(manifest["files"]); bad[0]["path"] = "../prior.json"
            with self.assertRaisesRegex(ValueError, "TWENTYTWO"): audit.check_files(folder, bad)
            link = root/"link.json"
            try: link.symlink_to(root/"prior.json")
            except (OSError, NotImplementedError): return
            with self.assertRaisesRegex(ValueError, "SYMLINK"): audit.metadata({"path": str(link), "sha256": audit.record(root/"prior.json")["sha256"]})


if __name__ == "__main__": unittest.main()
