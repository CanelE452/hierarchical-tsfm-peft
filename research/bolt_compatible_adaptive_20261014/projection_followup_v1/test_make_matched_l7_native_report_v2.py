"""Synthetic JSON/JSONL metadata and report text only; no scientific payloads or figures."""
from pathlib import Path
from unittest.mock import patch
import copy
import json
import statistics
import tempfile
import unittest

import make_matched_l7_native_report_v2 as report
from test_audit_f_input_gradient_report_v1 import score
from test_audit_learned_output_space_report_v1 import fixture as space_fixture


def fixture(folder):
    asset=folder/"unopened.npz"; asset.write_text("Existence only; report must not open",encoding="utf-8"); binding={"path":str(asset),"sha256":"a"*64}
    def entry(value): return {"artifact":binding,"scores_all_observed":score(value),"scores_complete_mask":score(value+.1,(70,70,70))}
    values={"train":{"states":{},"failed_paths":[],"optional_NumPy_context_parity_checks":3,"optional_NumPy_context_parity_failures":[],"resource_terminal":{"wall_seconds":2.}},"audit":{"new_output_space_diagnostics":{},"partial_paths":[],"new_output_space_diagnostic_failures":[],"optional_NumPy_context_checks":[{"split":s,"registered_tolerance_pass":True,"primary_gate":False} for s in ("fit","tune","val")],"resource":{"wall_seconds":1.}},"cost":{"cost":{"robin":{"rows":[]}},"failed_paths":[],"resource_terminal":{"wall_seconds":3.}},"cost_audit":{"resource_terminal":{"wall_seconds":.5}},"science":{"datasets":{"robin":{"spaces":{"PCA":{"conditions":{"L7_RECON":{"splits":{s:{"RAW_BOLT":entry(.25)} for s in ("fit","tune","val")}}}}}}}}}
    diagnostic=space_fixture(folder)
    for index,state in enumerate(report.STATES):
        owner=values["train"]["states"][state]={"native":{},"Bolt":{s:entry(.5+index*.02) for s in ("fit","tune","val")},"frozen_state_source":binding,"E_previously_supervised_by_FIT_future":index>0,"D_previously_supervised_by_FIT_future":False}
        values["audit"]["new_output_space_diagnostics"][state]={}
        for seed in report.SEEDS:
            log=folder/(state+str(seed)+".jsonl"); events=[{"stage":"TRAINING_UPDATE","update":i+1,"loss":.2,"head_weight_gradient_norm_before_clip":.3,"head_bias_gradient_norm_before_clip":.4,"global_gradient_norm_before_clip":.5,"clipped":False} for i in range(600)]
            log.write_text("\n".join(json.dumps(p) for p in events),encoding="utf-8")
            cps=[{"update":u,"state":binding,"tune":score(v)["normalized"],**entry(v)} for u,v in zip((0,100,200,400,600),(3.,1.,.8,.8,1.2))]
            row={"checkpoints":cps,"selected_update":200,"selected_state":binding,"selected_head_state_sha256":"b"*64,"order_sha256":"sameorder","updates_log":report.record(log),"selection_lock":binding,"loss":"fixed entire FIT counts","FIT_channel_observed_counts":[100,90,80],"global_clipping_fraction":0.,"post_optimizer_frozen_flag_checks":600,"frozen_state_checkpoint_byte_checks":5,"final_frozen_status":{"trainable_parameter_names":["head.weight","head.bias"]},"training_head_values_changed":True,"selected_head_nonzero":True,"frozen_bytes_checked_at_CP_and_final_only":True,"independent_per_update_frozen_bytes_replayed":False,"gradient_clipping_parameter_scope":"HEAD_ONLY","probe":{"status":"PASS","optimizer_updates":0,"artifact":binding},"FIT":entry(.9),"DEV":entry(1.+index*.02)}
            owner["native"][str(seed)]=row; values["audit"]["new_output_space_diagnostics"][state][str(seed)]={}
            for split in ("tune","val"):
                cell=copy.deepcopy(diagnostic["datasets"]["robin"]["conditions"][f"FULL_L7_s{seed}"]["splits"][split]); cell["methods"]["ACTUAL_NATIVE"]=cell["methods"].pop("ACTUAL_F"); cell.update(independent_external_replay_of_new_diagnostic_claimed=False,selected_E_previously_supervised_by_FIT_future_targets=index>0,selected_D_previously_supervised_by_FIT_future_targets=False,selected_native_head_previously_supervised_by_FIT_future_targets=True,native_head_training_procedure_used_FIT_future_targets=True,selected_state=binding)
                values["audit"]["new_output_space_diagnostics"][state][str(seed)][split]=cell
    rows=values["cost"]["cost"]["robin"]["rows"]
    for batch in (1,8):
        for route in report.ROUTES:
            duration=(5,1,3,4,2) if route=="RAW_BOLT" else (4,2,6,3,1) if "_NATIVE_" not in route else (2,2,3,4,1)
            sec=[v/1000 for v in duration]; matched="RAW_BOLT" if route=="RAW_BOLT" else route.split("_NATIVE_")[0]+"_BOLT" if "_NATIVE_" in route else route
            samples=[{"route":route,"batch":batch,"repeat":i,"seconds":s,"perf_start_ns":10**18+i*10**8,"perf_end_ns":10**18+i*10**8+v*10**6} for i,(s,v) in enumerate(zip(sec,duration))]
            rows.append({"route":route,"batch":batch,"status":"COMPLETE","seconds":sec,"samples":samples,"median_seconds":statistics.median(sec),"median_seconds_per_window":statistics.median(sec)/batch,"matched_Bolt_route":matched})
    for row in rows:
        for field,route in (("RAW","RAW_BOLT"),("matched_Bolt",row["matched_Bolt_route"])):
            denominator=report.choose(rows,route=route,batch=row["batch"]); ratios=[a/b for a,b in zip(row["seconds"],denominator["seconds"])]; row["paired_"+field+"_median_ratio"]=statistics.median(ratios); row["ratio_of_"+field+"_medians"]=statistics.median(row["seconds"])/statistics.median(denominator["seconds"])
    return values


class MatchedNativeReportTests(unittest.TestCase):
    def test_checkpoint_state_label_collision_repair_and_all_other_tables_unchanged(self):
        import make_matched_l7_native_report_v1 as preserved
        with tempfile.TemporaryDirectory() as folder:
            values=fixture(Path(folder)); old=preserved.collect_tables(values); new=report.collect_tables(values)
            self.assertTrue(all(isinstance(p["state"],dict) for p in old["CHECKPOINTS"]))
            for state in report.STATES:
                for seed in report.SEEDS:
                    rows=[p for p in new["CHECKPOINTS"] if p["state"]==state and p["seed"]==seed]
                    self.assertEqual([p["update"] for p in rows],[0,100,200,400,600]); self.assertEqual(sum(p["selected"] for p in rows),1)
                    for row,source in zip(rows,values["train"]["states"][state]["native"][str(seed)]["checkpoints"]):
                        self.assertEqual(row["checkpoint_state"],source["state"]); self.assertEqual(row["artifact"],source["artifact"])
            for key in report.TABLES:
                if key!="CHECKPOINTS": self.assertEqual(old[key],new[key],key)

    def test_all_four_synthetic_figures_render_PNG_and_SVG_after_CP_grouping(self):
        import xml.etree.ElementTree as ET
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); tables=report.collect_tables(fixture(root)); paths=report.figures(tables,root)
            self.assertEqual(len(paths),8); self.assertEqual(len(set(p.name for p in paths)),8)
            for path in paths:
                self.assertGreater(path.stat().st_size,1000)
                if path.suffix==".png":
                    with Image.open(path) as image: self.assertGreater(image.width,500); self.assertGreater(image.height,500); image.verify()
                else: self.assertTrue(ET.parse(path).getroot().tag.endswith("svg"))

    def test_all_table_counts_primary_units_masks_pairing_new_logs_and_context(self):
        with tempfile.TemporaryDirectory() as folder:
            values=fixture(Path(folder)); tables=report.collect_tables(values)
            self.assertEqual([len(tables[k]) for k in report.TABLES],[120,72,30,3600,6,6,12,36,3,0,20,100,4])
            paired=report.choose(tables["PAIRED"],state="RECON_CP0",seed=92601,split="tune",mask="all_observed",scale="normalized"); self.assertEqual(paired["native_minus_Bolt_MSE"],.8-.5)
            masked=report.choose(tables["SCORES"],state="RECON_CP0",predictor="NATIVE",seed=92601,split="tune",mask="complete_C",scale="normalized"); self.assertEqual(masked["observed_counts"],[70]*3); self.assertEqual(masked["per_channel_mse"],[.9]*3)
            self.assertEqual(report.choose(tables["RESOURCES"],stage="cost_audit")["wall_seconds"],.5)
            self.assertEqual(sum(p["selected"] for p in tables["CHECKPOINTS"]),6); self.assertTrue(all(p["new_head_fit"] for p in tables["UPDATES"]))
            self.assertEqual(set(p["method"] for p in tables["SPACE_METHODS"]),{"ACTUAL_NATIVE","EY","FLAT"})
            self.assertFalse(tables["SPACE_METHODS"][0]["real_trace_bound_inequality_pass"])

    def test_same_E_Bolt_denominator_and_raw_ratios_are_distinct_from_median_ratios(self):
        with tempfile.TemporaryDirectory() as folder:
            tables=report.collect_tables(fixture(Path(folder))); row=report.choose(tables["WHOLE_COST"],route="RECON_CP0_NATIVE_s92601",batch=8)
            self.assertEqual(row["ratio_of_RAW_medians"],2/3); self.assertEqual(row["paired_RAW_median_ratio"],1.)
            self.assertEqual(row["ratio_of_matched_Bolt_medians"],2/3); self.assertEqual(row["paired_matched_Bolt_median_ratio"],1.)
            self.assertEqual(tables["WHOLE_COST_SAMPLES"][0]["perf_start_ns"],10**18)

    def test_changed_CP_tie_update_sequence_cost_ratio_and_integer_ns_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            values=fixture(Path(folder)); wrong=copy.deepcopy(values); wrong["train"]["states"]["RECON_CP0"]["native"]["92601"]["selected_update"]=400
            with self.assertRaisesRegex(ValueError,"EARLIEST"): report.collect_tables(wrong)
            for change,label in ((lambda v:v["cost"]["cost"]["robin"]["rows"][1].update(paired_RAW_median_ratio=99.),"PAIRED"),(lambda v:v["cost"]["cost"]["robin"]["rows"][0]["samples"][0].update(perf_end_ns=10**18+5000001),"INTEGER_NS")):
                wrong=copy.deepcopy(values); change(wrong)
                with self.subTest(label=label),self.assertRaisesRegex(ValueError,label): report.collect_tables(wrong)
            row=values["train"]["states"]["RECON_CP0"]["native"]["92601"]; path=Path(row["updates_log"]["path"]); lines=path.read_text().splitlines(); lines[0]=lines[1]; path.write_text("\n".join(lines),encoding="utf-8"); row["updates_log"]=report.record(path)
            with self.assertRaisesRegex(ValueError,"UPDATES"): report.collect_tables(values)

    def test_new_native_geometry_labels_and_no_universal_bound_or_third_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            values=fixture(Path(folder))
            for key,value in (("all_real_latent_or_all_FP32_lower_bound_claimed",True),("independent_external_replay_of_new_diagnostic_claimed",True)):
                wrong=copy.deepcopy(values); wrong["audit"]["new_output_space_diagnostics"]["RECON_CP0"]["92601"]["tune"][key]=value
                with self.subTest(key=key),self.assertRaisesRegex(ValueError,"GEOMETRY_LABEL"): report.collect_tables(wrong)

    def test_korean_report_contains_raw_numbers_opposite_evidence_and_exact_reuse_scope(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); values=fixture(root); tables=report.collect_tables(values); text=report.make_text(values,tables,{},root)
            self.assertIn("TUNE 6/6, 기노출 DEV 6/6",text); self.assertIn("낮은 native는 0/6",text); self.assertIn("CP100은 0/6",text)
            self.assertIn("TUNE retained numeric minimum이 Raw보다 높은 고정 E/D는 0/3",text); self.assertIn("DEV minimum이 Raw보다 낮은 고정 E/D는 0/3",text)
            self.assertIn("각 lag에서 빼는 기준 z_past[-1-l]",text); self.assertIn("prefix가 모두 끝난 뒤에도",text); self.assertIn("plateau가 남을 수 있다",text)
            self.assertIn("TUNE native 0.8000000000, Bolt 0.5000000000; DEV native 1.0000000000, Bolt 0.5000000000, Raw 0.2500000000",text)
            for token in ("새 F forward는0","새 matched RECON C decode3개","C byte 보존6개","12개 독립 자료가 아니다","독립 일반화 검증이 아니다","strict K-only도 아니다","5순환은 10개 위치의 완전 균형이 아니다","매 update bytes·loss·backward·optimizer를 독립 재현한 검사는 아니다","원 real/FP32 false-bound"):
                self.assertIn(token.replace("원 real/FP32","real/FP32"),text)
            self.assertEqual(text.count("ms/window; Raw paired"),20); self.assertEqual(text.count("; complete "),12)
            for key,rows in tables.items(): self.assertIn(f"{key}.csv](tables/{key}.csv): {len(rows)}행",text)

    def test_closed_cost_gate_precedes_metadata_or_payload_reads(self):
        with patch.dict(report.FIXED_SHA,{"cost_audit":None}),patch.object(report,"read",side_effect=AssertionError("closed read")),patch.object(report,"bound",side_effect=AssertionError("closed file hash")),self.assertRaisesRegex(ValueError,"NOT_YET_REGISTERED"):
            report.load_verified({key:"missing" for key in report.FIXED_SHA})
        with patch.object(report,"FIXED_SHA",{k:"a"*64 for k in report.FIXED_SHA}),patch.object(report,"AUDIT_SOURCES",{"audit":{},"cost_audit":{}}),patch.object(report,"COST_COUNTS",{"cost_cells_checked":20}),patch.object(report,"read",side_effect=AssertionError("NPZ opened")),patch.object(report,"bound",side_effect=AssertionError("NPZ hashed")),self.assertRaisesRegex(ValueError,"JSON_METADATA_INPUTS_ONLY"):
            report.load_verified({key:"forbidden.npz" for key in report.FIXED_SHA})


if __name__=="__main__": unittest.main()
