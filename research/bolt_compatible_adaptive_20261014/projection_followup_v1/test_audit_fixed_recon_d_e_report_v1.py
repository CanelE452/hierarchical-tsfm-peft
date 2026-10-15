"""Synthetic metadata checks only; no generator or scientific payload reads."""
from pathlib import Path
from unittest.mock import patch
import copy
import json
import tempfile
import unittest

import audit_fixed_recon_d_e_report_v1 as audit
from test_audit_f_input_gradient_report_v1 import fixtures as gradient_fixture, score, write_csv
from test_audit_learned_output_space_report_v1 import fixture as space_fixture


def fixture(folder):
    prior = gradient_fixture(folder); spaces = space_fixture(folder)
    values = {"train": {"datasets": {"robin": {"conditions": {}}}, "resource_terminal": {"cpu_seconds": 3.}}, "joint": {"datasets": {"robin": {"conditions": {}}}}, "audit": {"new_output_space_diagnostics": {}, "resource": {"cpu_seconds": 2.}}, "joint_audit": {"new_output_space_diagnostics": {}}, "science": {"datasets": {"robin": {"spaces": {"PCA": {"conditions": {"L7_RECON": {"splits": {}}}}}}}}}
    def entry(value, artifact): return {"artifact": artifact, "scores_all_observed": score(value), "scores_complete_mask": score(value+.1, (70, 70, 70))}
    for seed in audit.SEEDS:
        old = copy.deepcopy(prior["parent"]["datasets"]["robin"]["conditions"][f"robin_L0_E_LEARN_s{seed}"])
        new = copy.deepcopy(prior["train"]["datasets"]["robin"]["conditions"][str(seed)])
        for fresh, row in ((False, old), (True, new)):
            for cp in row["checkpoints"]: cp["TUNE_latent_variance"] = [.2, .3]
            if fresh:
                row.update(D_trainable=False, fixed_decoder_final_status=copy.deepcopy(audit.FLAGS), D_frozen_after_optimizer_step_checks=600, D_and_center_checkpoint_byte_checks=5, D_bytes_checked_at_checkpoints_and_final_only=True, independent_per_update_D_bytes_replayed=False, initial_D_value_sha256="2"*64)
                for cp in row["checkpoints"]: cp.update(fixed_decoder_status=copy.deepcopy(audit.FLAGS), D_and_center_exact_initial_bytes=True)
                row["probe"] = {"status": "PASS", "optimizer_updates": 0, "component_byte_exact": {key: True for key in ("prediction", "latent_history", "latent_future", "lag_features")}, "fixed_decoder_status": copy.deepcopy(audit.FLAGS), "artifact": row["selected_state"]}
                path = Path(row["updates_log"]["path"]); logs = [json.loads(line) for line in path.read_text().splitlines()]
                for event in logs: event.update(audit.FLAGS, D_gradient_norm_before_clip=0., gradient_clipping_parameter_scope="E_ONLY")
                path.write_text("\n".join(json.dumps(item) for item in logs), encoding="utf-8"); row["updates_log"] = audit.record(path)
        # Fresh E-only is worse on TUNE but better on DEV, still worse than CP0/Raw.
        new["DEV"] = entry(1., new["selected_state"]); old["DEV"] = entry(1.3, old["selected_state"])
        values["train"]["datasets"]["robin"]["conditions"][str(seed)] = new
        values["joint"]["datasets"]["robin"]["conditions"][str(seed)] = old
        values["audit"]["new_output_space_diagnostics"][str(seed)] = {}
        values["joint_audit"]["new_output_space_diagnostics"][str(seed)] = {}
        for split in ("tune", "val"):
            cell = spaces["datasets"]["robin"]["conditions"][f"FULL_L7_s{seed}"]["splits"][split]
            for owner, fresh in (("audit", True), ("joint_audit", False)):
                item = copy.deepcopy(cell); item.update(independent_reduced_LS_vs_SVD_verified=True, diagnostic_future_labels_used_for_new_fit_or_selection=False)
                if fresh:
                    item.pop("selected_E_D_previously_supervised_by_FIT_future_targets")
                    item.update(selected_E_previously_supervised_by_FIT_future_targets=True, selected_D_previously_supervised_by_FIT_future_targets=False)
                values[owner]["new_output_space_diagnostics"][str(seed)][split] = item
    artifact = new["selected_state"]
    for split in ("fit", "tune", "val"):
        baseline = entry(.5, artifact); baseline["RAW_BOLT"] = entry(.25, artifact)
        values["science"]["datasets"]["robin"]["spaces"]["PCA"]["conditions"]["L7_RECON"]["splits"][split] = baseline
    return values


def markdown(folder, tables):
    def mse(method, seed, split): return audit.find(tables["SCORES"], method=method, seed=seed, split=split, mask="all_observed", scale="normalized")["mse"]
    lines = ["JOINT_E_D보다 2/2 조건이 좋았지만 TUNE는 2/2 조건에서 나빴다. RECON CP0보다 2/2, Raw보다 2/2 조건이 나빴다.", f"RECON CP0 TUNE {mse('RECON_CP0',0,'tune'):.10f}, DEV {mse('RECON_CP0',0,'val'):.10f}; Raw TUNE {mse('RAW_BOLT',0,'tune'):.10f}, DEV {mse('RAW_BOLT',0,'val'):.10f}.", "TUNE 0/2 조건에서 Raw보다 높고, DEV 0/2 조건에서 Raw보다 낮았다", "새 2fit 1,200 update·10CP / 2fit 1,200 update·10CP 재사용", "단일 인과효과로 부르지 않는다; 매 update bytes를 독립 재생했다는 뜻은 아니다; 학습 감사 중 새 계산; 세 번째 외부 재현은 아니다; MAE·물리 단위 값은 이 MSE 해의 점수; false-bound flags; strict K-only도 아니다; 새 deployment 비용은 현재 미측정; 독립 일반화 검증이 아니다"]
    for seed in audit.SEEDS:
        a, b = (audit.find(tables["CONDITIONS"], method=m, seed=seed) for m in ("E_ONLY", "JOINT_E_D"))
        lines.append(f"seed {seed}: 새 E_ONLY CP{a['selected_update']}, TUNE {mse('E_ONLY',seed,'tune'):.10f}, DEV {mse('E_ONLY',seed,'val'):.10f}; 재사용 JOINT_E_D CP{b['selected_update']}, TUNE {mse('JOINT_E_D',seed,'tune'):.10f}, DEV {mse('JOINT_E_D',seed,'val'):.10f}.")
    for r in tables["SPACE_CELLS"]:
        lines.append(f"{r['method']} / {r['seed']} / {r['split']}: actual {r['actual_complete_MSE']:.10f}, E(Y) {r['EY_complete_MSE']:.10f}, minimum {r['minimum_normalized_MSE']:.10f}, Raw {r['RAW_complete_MSE']:.10f}; complete {r['complete_target_positions']}/{r['target_positions']}.")
    (folder/"tables").mkdir()
    for key, rows in tables.items():
        (folder/"tables"/(key+".csv")).write_text("Exists", encoding="utf-8"); lines.append(f"[{key}.csv](tables/{key}.csv): {len(rows)}행")
    return "\n".join(lines)


class FixedDReportAuditTests(unittest.TestCase):
    def test_primary_masks_pairing_new_reused_counts_and_numeric_space_scope(self):
        with tempfile.TemporaryDirectory() as folder:
            tables = audit.expected_tables(fixture(Path(folder)))
            self.assertEqual([len(tables[k]) for k in audit.TABLES], [72,24,20,2400,4,2,8,24,8,2])
            pair = audit.find(tables["PAIRED"], seed=92601, split="tune", mask="all_observed", scale="normalized")
            self.assertEqual(pair["MSE_E_ONLY_minus_JOINT"], 1.2-1.)
            self.assertLess(audit.find(tables["PAIRED"], seed=92601, split="val", mask="all_observed", scale="normalized")["MSE_E_ONLY_minus_JOINT"], 0.)
            row = audit.find(tables["SCORES"], seed=92601, method="E_ONLY", split="tune", mask="complete_C", scale="normalized")
            self.assertEqual(row["observed_counts"], [70]*3); self.assertEqual(row["per_channel_mse"], [1.3]*3)
            self.assertEqual(sum(r["new_execution"] for r in tables["UPDATES"]), 1200); self.assertEqual(sum(r["selected"] for r in tables["CHECKPOINTS"]), 4)
            self.assertTrue(all(r["selected_D_previously_supervised_by_FIT_future_targets"] is False for r in tables["SPACE_CELLS"] if r["method"]=="E_ONLY"))
            self.assertFalse(tables["SPACE_METHODS"][0]["real_trace_bound_inequality_pass"])

    def test_all_requested_CSV_fields_roundtrip_and_perchannel_falseflag_tamper(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); tables = audit.expected_tables(fixture(root))
            for key, rows in tables.items():
                expected = audit.relative_values(rows, root); path = root/(key+".csv"); write_csv(path, expected)
                self.assertGreater(audit.check_table(path, expected), 0); audit.common.check_reference_paths(expected, root)
            for key, field in (("SCORES", "per_channel_mse"), ("SPACE_METHODS", "real_trace_bound_inequality_pass")):
                expected = audit.relative_values(tables[key], root); changed = copy.deepcopy(expected)
                if field=="per_channel_mse": changed[0][field][0] += .01
                else: changed[0][field] = True
                path=root/(key+".csv"); write_csv(path, changed)
                with self.assertRaises(ValueError): audit.check_table(path, expected)

    def test_CP_tie_pair_order_fixed_D_and_log_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            values = fixture(Path(folder))
            for field, value, label in (("selected_update",400,"EARLIEST"),("order_sha256","changed","MINIBATCH"),("D_trainable",True,"FIXED_D")):
                wrong=copy.deepcopy(values); wrong["train"]["datasets"]["robin"]["conditions"]["92601"][field]=value
                with self.subTest(field=field), self.assertRaisesRegex(ValueError,label): audit.expected_tables(wrong)
            row=values["train"]["datasets"]["robin"]["conditions"]["92601"]; path=Path(row["updates_log"]["path"]); logs=[json.loads(line) for line in path.read_text().splitlines()]; logs[0]["D_gradient_norm_before_clip"] = .1
            path.write_text("\n".join(json.dumps(item) for item in logs), encoding="utf-8"); row["updates_log"]=audit.record(path)
            with self.assertRaisesRegex(ValueError,"ZERO_D_NORM"): audit.expected_tables(values)

    def test_selected_CP0_E_supervision_is_separate_from_fixed_D(self):
        with tempfile.TemporaryDirectory() as folder:
            values=fixture(Path(folder)); row=values["train"]["datasets"]["robin"]["conditions"]["92601"]; row["checkpoints"][0]["tune"]["mse"] = .01; row["selected_update"]=0
            for cell in values["audit"]["new_output_space_diagnostics"]["92601"].values(): cell["selected_E_previously_supervised_by_FIT_future_targets"]=False
            tables=audit.expected_tables(values); cell=audit.find(tables["SPACE_CELLS"],method="E_ONLY",seed=92601,split="val")
            self.assertIs(cell["selected_E_previously_supervised_by_FIT_future_targets"],False); self.assertIs(cell["selected_D_previously_supervised_by_FIT_future_targets"],False)
            values["audit"]["new_output_space_diagnostics"]["92601"]["val"]["selected_D_previously_supervised_by_FIT_future_targets"]=True
            with self.assertRaisesRegex(ValueError,"SUPERVISION"): audit.expected_tables(values)

    def test_markdown_primary_numeric_rows_opposite_order_counts_and_scope(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); tables=audit.expected_tables(fixture(root)); text=markdown(root,tables); path=root/"REPORT_KO.md"; path.write_text(text,encoding="utf-8")
            self.assertEqual(audit.check_markdown(root,tables),10)
            for wrong in (text.replace("TUNE는 2/2","TUNE는 1/2"),text.replace("1.2000000000","1.2500000000",1),text.replace("세 번째 외부 재현은 아니다","unverified"),text.replace("complete 20/24","complete 19/24",1)):
                path.write_text(wrong,encoding="utf-8")
                with self.subTest(wrong=wrong[-40:]), self.assertRaises(ValueError): audit.check_markdown(root,tables)
            path.write_text(text,encoding="utf-8"); (root/"tables/SCORES.csv").unlink()
            with self.assertRaisesRegex(ValueError,"MD_LOCAL_LINK"): audit.check_markdown(root,tables)

    def test_closed_gate_and_scientific_payload_rejection_before_open(self):
        with patch.object(audit,"REPORT_SHA",None),patch.object(audit,"record",side_effect=AssertionError("opened")),self.assertRaisesRegex(ValueError,"NOT_YET_REGISTERED"): audit.verify("missing")
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"forbidden.npz"; path.write_text("unopened")
            with self.assertRaisesRegex(ValueError,"METADATA_ONLY"): audit.metadata({"path":str(path),"sha256":"0"*64})


if __name__=="__main__": unittest.main()
