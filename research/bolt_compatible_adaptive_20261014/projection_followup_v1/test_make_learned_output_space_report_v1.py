"""Generated metadata only; no actual arrays, checkpoint/model IO or fitting."""
from pathlib import Path
import importlib.util
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch


SOURCE=Path(__file__).with_name("make_learned_output_space_report_v1.py")


def module():
    if not SOURCE.exists(): raise AssertionError("Learned output-space report source is missing")
    spec=importlib.util.spec_from_file_location("learned_space_report_fixture",SOURCE)
    result=importlib.util.module_from_spec(spec); spec.loader.exec_module(result); return result


def fixture_values(folder):
    r=module(); folder.mkdir(parents=True,exist_ok=True)
    def opaque(name): return {"path":str(folder/(name+".npz")),"sha256":"a"*64,"bytes":123}
    def score(mse,c):
        return {scale:{"mse":mse*f,"mae":(mse*f)**.5,"per_channel_mse":[mse*f]*c,"per_channel_mae":[(mse*f)**.5]*c,"observed_counts":[96]*c,"active_channels":c} for scale,f in (("normalized",1),("physical_per_channel_diagnostic",4))}
    d={"status":"COMPLETE","datasets":{},"failed_paths":[],"completed_diagnostic_cells":24,"expected_diagnostic_cells":24,"selected_states_diagnosed":12,
        "saved_ACTUAL_F_predictions_diagnosed":24,"EY_diagnostics":24,"latent_persistence_diagnostics":24,"independent_LS_SVD_crosschecks":24,
        "new_F_calls":0,"new_model_objects_loaded":0,"new_training_updates":0,"new_E_D_fit":False,"CP_or_branch_reselection":False,"oracle_used_for_fit_or_selection":False,
        "protected_TEST_accessed":False,"independent_holdout":False,"strict_K_only":False,"new_F_state_hash_computed":False,"whole_deployment_cost_measured":False,
        "numerical_crosscheck_is_not_independent_generalization":True,"resource_terminal":{"wall_seconds":1.,"process_cpu_seconds":.5}}
    for data,(c,k,dev) in r.DATASETS.items():
        d["datasets"][data]={"conditions":{}}
        for name in r.CONDITIONS:
            lag="L7" if "L7" in name else "L0"; seed=92601 if name.endswith("1") else 92602
            owner={"status":"COMPLETE","origin":"STOP_REUSED" if name.startswith("STOP") else "FULL_REUSED","seed":seed,"selected_update":400,
                "selected_state":opaque(data+name+"STATE"),"selection_lock":{"path":str(folder/(data+name+"LOCK.json")),"sha256":"b"*64},"splits":{}}
            d["datasets"][data]["conditions"][name]=owner
            for split in ("tune",dev):
                minimum=.03 if split=="tune" else .43 if data=="robin" else .39
                raw=.057 if split=="tune" else .34 if data=="robin" else .60
                methods={}
                for method,value in (("ACTUAL_F",.58 if data=="robin" else .66),("EY",.49 if data=="robin" else .41),("FLAT",1.2)):
                    methods[method]={"scores_all_observed":score(value-.01,c),"scores_complete_mask":score(value,c),"real_trace_scores_complete_C":score(value,c),
                        "retained_output_real_trace_scores_complete_C":score(value,c),"actual_FP32_minus_minimum":value-minimum,"real_trace_minus_minimum":value-minimum,
                        "measured_FP32_rounding_normalized_MSE_allowance":0.,"discarded_numeric_map_output_squared_norm":0.,"discarded_numeric_map_output_max_abs":0.,
                        "retained_output_trace_bound_inequality_pass":True,"real_trace_bound_inequality_pass":True,"FP32_bound_with_measured_allowance_pass":True,"numerical_MSE_tolerance":1e-10,
                        "role":"DIAGNOSTIC_FUTURE_Y_NOT_DEPLOYABLE" if method=="EY" else "PAST_ONLY_LATENT_PERSISTENCE" if method=="FLAT" else "ACTUAL_DEPLOYMENT_SAVED_FUTURE"}
                owner["splits"][split]={"status":"COMPLETE","methods":methods,"minimum_normalized_MSE":minimum,"minimum_solution_scores_complete_C":score(minimum,c),
                    "RAW_BOLT_scores_all_observed":score(raw-.01,c),"RAW_BOLT_scores_complete_mask":score(raw,c),"current_C_persistence_scores_complete_mask":score(1.3,c),
                    "complete_target_positions":96,"target_positions":96,"joint_unknown_future_coordinates":48*k,"joint_output_coordinates_before_mask":48*c,
                    "lag_slope_parameters_per_C":(7 if lag=="L7" else 1)*k,"lags":[0,1,2,4,8,16,32] if lag=="L7" else [0],
                    "joint_metadata":{"rcond":1e-15,"numeric_rank_per_origin":[2,2],"all_HK_unknowns_retained_before_output_mask":True,"minimum_space_definition":"retained numeric singular space"},
                    "independent_LS_certificates":[{"window_indices":[0,1],"complete_horizons":[True]*48,"selected_output_rows":48*c,"unknown_future_coordinates_kept":48*k,"numeric_rank":2,"singularvalues":[2.,1.,0.],"normal_residual_max_abs":1e-15,"normal_tolerance":1e-10}],
                    "minimum_only_normalized_MSE":True,"RAW_bound_applied":False,"all_real_latent_or_all_FP32_lower_bound_claimed":False,"old_initial_D_minimum_reused":False,
                    "diagnostic_future_labels_used_for_new_fit_or_selection":False,"selected_E_D_previously_supervised_by_FIT_future_targets":True,"independent_reduced_LS_vs_SVD_verified":True,
                    "EY_encoding_backend":"NumPy FP32 selected E","EY_incomplete_C_horizons_filled_with_current_C":0,"EY_fill_can_affect_later_L7_complete_outputs":lag=="L7",
                    "input_source":opaque(data+split+"INPUT"),"selected_forecast_source":opaque(data+name+split+"F"),"artifact":opaque(data+name+split+"DIAG")}
    return {"diagnostic":d,"audit":{"status":"PASS","protected_TEST_accessed":False,"totals":{},"partial_paths":[]}}


def gate_fixture(folder):
    r=module(); values=fixture_values(folder)
    def put(name,value):
        path=folder/name; path.write_text(json.dumps(value,allow_nan=False),encoding="utf-8"); return r.record(path)
    producer=folder/"run_learned_output_space_diagnostic_v1.py"; producer.write_text("# generated source\n",encoding="utf-8")
    auditor=folder/"auditor.py"; auditor.write_text("# generated audit\n",encoding="utf-8")
    d=values["diagnostic"]; d["source_records"]=[r.record(producer)]
    keys=("stop_gradient_diagnostic","stop_gradient_audit","full_gradient_parent","full_gradient_parent_audit","causal_lag_diagnostic","causal_lag_audit","selection_registry","full_gradient_selection_registry","plan")
    for key in keys: d[key]=put(key+".json",{"status":"PASS"})
    diagonal=put("DIAGNOSTIC.json",d)
    values["audit"].update(diagnostic=diagonal,audit_source=r.record(auditor),totals={"diagnostic_cells_checked":24},**{k:d[k] for k in keys})
    audit=put("AUDIT.json",values["audit"])
    return {"diagnostic":Path(diagonal["path"]),"audit":Path(audit["path"])},{"FIXED_SHA":{"diagnostic":diagonal["sha256"],"audit":audit["sha256"]},
        "PRODUCER_SOURCE":r.record(producer),"AUDITOR_SOURCE":r.record(auditor),"EXPECTED_COUNTS":{"diagnostic_cells_checked":24}}


class ReportTests(unittest.TestCase):
    def setUp(self): self.r=module(); self.temp=tempfile.TemporaryDirectory(dir=SOURCE.parent); self.folder=Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()
    def test_closed_actual_gate_reads_nothing(self):
        with patch.object(self.r,"FIXED_SHA",{"diagnostic":None,"audit":None}),patch.object(self.r,"read",side_effect=AssertionError("closed IO")),self.assertRaisesRegex(ValueError,"NOT_YET_REGISTERED"):
            self.r.load_verified({"diagnostic":"missing","audit":"missing"})
    def test_24_cells_masks_and_672_scores_are_distinct(self):
        tables=self.r.collect_tables(fixture_values(self.folder))
        self.assertEqual(len(tables["CELLS"]),24); self.assertEqual(len(tables["SCORES"]),672); self.assertEqual(len(tables["TRACES"]),72)
        self.assertEqual(len(tables["MATCHED_L0_FULL_STOP"]),8)
        where={"dataset":"beijing_aotizhongxin","condition":"FULL_L0_s92601","split":"eval","method":"RAW_BOLT","scale":"normalized","phase":"FP32"}
        complete=self.r.choose(tables["SCORES"],**where,mask="complete_C"); full=self.r.choose(tables["SCORES"],**where,mask="all_observed")
        self.assertNotEqual(complete["mse"],full["mse"])
    def test_positive_metadata_gate_and_modified_input_rejection(self):
        paths,patches=gate_fixture(self.folder)
        with patch.multiple(self.r,**patches):
            values,bindings=self.r.load_verified(paths); self.assertEqual(values["diagnostic"]["completed_diagnostic_cells"],24)
            paths["audit"].write_text(paths["audit"].read_text()+" ")
            with self.assertRaisesRegex(ValueError,"EXACT_ACTUAL_INPUT"): self.r.load_verified(paths)
    def test_original_false_real_or_FP32_flags_are_not_forced_to_pass(self):
        values=fixture_values(self.folder); row=values["diagnostic"]["datasets"]["robin"]["conditions"]["FULL_L0_s92601"]["splits"]["val"]["methods"]["ACTUAL_F"]
        row["real_trace_bound_inequality_pass"]=False; row["FP32_bound_with_measured_allowance_pass"]=False
        traces=self.r.collect_tables(values)["TRACES"]
        found=self.r.choose(traces,dataset="robin",condition="FULL_L0_s92601",split="val",method="ACTUAL_F")
        self.assertFalse(found["real_trace_bound_inequality_pass"]); self.assertTrue(found["retained_output_trace_bound_inequality_pass"])
    def test_partial_failed_cell_is_preserved_and_not_complete(self):
        values=fixture_values(self.folder); d=values["diagnostic"]; d["status"]="PARTIAL_COMPLETED"
        cell=d["datasets"]["robin"]["conditions"]["FULL_L0_s92601"]["splits"]["val"]
        cell.clear(); cell.update(status="FAILED_PATH_PRESERVED",failure={"message":"rank mismatch"}); d["completed_diagnostic_cells"]=23
        tables=self.r.collect_tables(values); self.assertEqual(len(tables["FAILURES"]),1)
        text=self.r.make_text(values,tables,{},self.folder); self.assertIn("미완료",text); self.assertIn("rank mismatch",text)
    def test_missing_expected_condition_is_rejected(self):
        values=fixture_values(self.folder); values["diagnostic"]["datasets"]["robin"]["conditions"].pop("STOP_L0_s92602")
        with self.assertRaisesRegex(ValueError,"TWELVE"): self.r.collect_tables(values)
    def test_text_keeps_numeric_minimum_and_future_diagnostic_scope(self):
        values=fixture_values(self.folder); text=self.r.make_text(values,self.r.collect_tables(values),{},self.folder)
        for phrase in ("complete-C","FP32","과거", "NumPy", "E(Y)", "L7", "current-C", "독립 일반화", "기노출", "새 선정", "Robin", "Beijing"):
            self.assertIn(phrase,text)
        self.assertIn("minimum>Raw가 TUNE 0/6 → 기노출 DEV 6/6",text)
        self.assertIn("minimum>Raw 0/6, E(Y)<Raw 6/6, actual F>Raw 6/6",text)
        self.assertIn("분포 shift가 원인이라고 확정하지",text)
    def test_generated_only_report_csv_images_and_no_overwrite(self):
        values=fixture_values(self.folder/"metadata"); output=self.folder/"reports"/"toy"
        with patch.object(self.r,"HERE",self.folder),patch.object(self.r,"load_verified",return_value=(values,{})):
            manifest=self.r.generate({},output); self.assertEqual(manifest["PNG_count"],4); self.assertEqual(manifest["SVG_count"],4)
            for item in manifest["files"]:
                path=output/item["path"]; self.assertEqual(self.r.sha(path),item["sha256"])
                if path.suffix==".png": self.assertEqual(path.read_bytes()[:8],b"\x89PNG\r\n\x1a\n")
                if path.suffix==".svg": ET.parse(path)
            with self.assertRaisesRegex(ValueError,"FRESH"): self.r.generate({},output)


if __name__=="__main__": unittest.main()
