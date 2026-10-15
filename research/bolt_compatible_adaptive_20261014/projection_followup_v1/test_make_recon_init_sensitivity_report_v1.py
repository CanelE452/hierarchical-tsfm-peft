"""Synthetic JSON/JSONL only: no scientific arrays, model, or fitting."""
from pathlib import Path
import importlib.util
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch


SOURCE=Path(__file__).with_name("make_recon_init_sensitivity_report_v1.py")


def module():
    if not SOURCE.exists(): raise AssertionError("RECON initialization report source missing")
    spec=importlib.util.spec_from_file_location("recon_report_fixture",SOURCE)
    result=importlib.util.module_from_spec(spec); spec.loader.exec_module(result); return result


def fixture_values(folder):
    folder.mkdir(parents=True,exist_ok=True); r=module()
    def opaque(name): return {"path":str(folder/(name+".npz")),"sha256":"a"*64}
    def score(mse):
        return {s:{"mse":mse*f,"mae":(mse*f)**.5,"per_channel_mse":[mse*f]*17,"per_channel_mae":[(mse*f)**.5]*17,"observed_counts":[48]*17,"active_channels":17} for s,f in (("normalized",1),("physical_per_channel_diagnostic",4))}
    def entry(mse,name): return {"scores_all_observed":score(mse),"scores_complete_mask":score(mse+.001),"artifact":opaque(name)}
    def condition(seed,new):
        cps=[]
        for i,update in enumerate((0,100,200,400,600)):
            mse=(.103 if new else .11)-.005*i
            cps.append({"update":update,"tune":score(mse)["normalized"],"state":opaque(str(seed)+str(new)+str(update)),"parameter_diagnostics":{"E_l2_norm":1.+i*.1,"D_l2_norm":2.+i*.2},"TUNE_latent_variance":[1.]*4,**entry(mse,"CP")})
        path=folder/(str(seed)+str(new)+"_UPDATES.jsonl")
        path.write_text("".join(json.dumps({"stage":"TRAINING_UPDATE","update":i,"loss":1./i,"E_gradient_norm_before_clip":2.,"D_gradient_norm_before_clip":1.,"global_gradient_norm_before_clip":2.3,"clipped":True})+"\n" for i in range(1,601)),encoding="utf-8")
        result={"status":"COMPLETE","selected_update":600,"selected_state":cps[-1]["state"],"selected_tune":cps[-1]["tune"],"CP0_TUNE":cps[0],"checkpoints":cps,"FIT":entry(.7,"FIT"),"DEV":entry(.54 if new else .59,"DEV"),"updates_log":r.record(path),"order_sha256":"b"*64,"training_updates_completed":600,"global_clipping_fraction":1.,"selection_lock":{"path":str(folder/(str(seed)+str(new)+"LOCK.json")),"sha256":"c"*64}}
        if new: result["probe"]={"status":"PASS","optimizer_updates":0,"checks":{k:{"pass":True,"atol":1e-5,"rtol":2e-6} for k in ("prediction","latent_history","latent_future","lag_features")},"artifact":opaque("PROBE")}
        return result
    train={"status":"COMPLETE","datasets":{"robin":{"conditions":{str(s):condition(s,True) for s in r.SEEDS}}},"failed_paths":[],"resource_terminal":{"wall_seconds":10.,"process_cpu_seconds":9.}}
    parent={"datasets":{"robin":{"conditions":{"robin_L7_E_LEARN_s"+str(s):condition(s,False) for s in r.SEEDS}}},"CP0_LS_and_RAW_baselines":{"robin":{"L7":{s:{"CP0_LS":entry(.6,"FORECAST_CP0")} for s in ("fit","val")}}}}
    recon_splits={s:{**entry(.103 if s=="tune" else .488,"RECON_CP0"),"RAW_BOLT":entry(.057 if s=="tune" else .34,"RAW")} for s in ("fit","tune","val")}
    science={"datasets":{"robin":{"spaces":{"PCA":{"conditions":{"L7_RECON":{"splits":recon_splits}}}}}}}
    def cell(new):
        methods={}
        for name,mse in (("ACTUAL_F",.54 if new else .59),("EY",.46),("FLAT",1.2)):
            methods[name]={**entry(mse,name),"real_trace_scores_complete_C":score(mse),"retained_output_real_trace_scores_complete_C":score(mse),"role":name,"actual_FP32_minus_minimum":mse-.4,"real_trace_minus_minimum":mse-.4,"retained_output_trace_bound_inequality_pass":True,"real_trace_bound_inequality_pass":False,"FP32_bound_with_measured_allowance_pass":False,"discarded_numeric_map_output_squared_norm":1e-16,"discarded_numeric_map_output_max_abs":1e-8,"measured_FP32_rounding_normalized_MSE_allowance":1e-8}
        return {"status":"COMPLETE","minimum_normalized_MSE":.4 if new else .43,"minimum_solution_scores_complete_C":score(.4 if new else .43),"methods":methods,"RAW_BOLT_scores_complete_mask":score(.34),"current_C_persistence_scores_complete_mask":score(1.3),"complete_target_positions":48,"target_positions":48,"joint_unknown_future_coordinates":192,"joint_output_coordinates_before_mask":816,"joint_metadata":{"rcond":1e-15,"numeric_rank_per_origin":[192],"minimum_space_definition":"registered retained numerical space"},"independent_LS_certificates":[{"numeric_rank":192,"singularvalues":[2.,1.],"normal_residual_max_abs":1e-12}],"independent_reduced_LS_vs_SVD_verified":True,"diagnostic_future_labels_used_for_new_fit_or_selection":False,"all_real_latent_or_all_FP32_lower_bound_claimed":False,"minimum_only_normalized_MSE":True,"EY_encoding_backend":"NumPy FP32","EY_incomplete_C_horizons_filled_with_current_C":0,"artifact":opaque("SPACE"),"input_source":opaque("INPUT"),"selected_forecast_source":opaque("SELECTED")}
    audit={"status":"PASS","training_verification_status":"PASS","new_space_diagnostic_status":"COMPLETE","new_output_space_diagnostics":{str(s):{split:cell(True) for split in ("tune","val")} for s in r.SEEDS},"new_output_space_diagnostic_failures":[],"partial_paths":[],"datasets":{"robin":{"conditions":{str(s):{"parameter_change":{"E_same_bytes":False}} for s in r.SEEDS}}},"resource":{"wall_seconds":1.}}
    space={"datasets":{"robin":{"conditions":{"FULL_L7_s"+str(s):{"splits":{split:cell(False) for split in ("tune","val")}} for s in r.SEEDS}}}}
    return {"train":train,"audit":audit,"parent":parent,"science":science,"space":space}


class ReportTests(unittest.TestCase):
    def setUp(self): self.r=module(); self.temp=tempfile.TemporaryDirectory(dir=SOURCE.parent); self.folder=Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()
    def test_closed_gate_reads_no_metadata(self):
        with patch.object(self.r,"FIXED_SHA",{"train":None,"audit":None}),patch.object(self.r,"read",side_effect=AssertionError("closed read")),self.assertRaisesRegex(ValueError,"NOT_YET_REGISTERED"):
            self.r.load_verified({"train":"absent","audit":"absent"})
    def test_positive_exact_metadata_gate_and_input_mutation_rejected(self):
        r=self.r; values=fixture_values(self.folder); train,audit=values["train"],values["audit"]
        def put(name,value):
            path=self.folder/name; path.write_text(json.dumps(value,allow_nan=False),encoding="utf-8"); return r.record(path)
        controller=self.folder/"controller.py"; controller.write_text("# synthetic source\n",encoding="utf-8")
        auditor=self.folder/"auditor.py"; auditor.write_text("# synthetic auditor\n",encoding="utf-8")
        for key in ("parent","science","space"): values[key].update(status="COMPLETE",F_before={"a":"same"},F_after={"a":"same"})
        train.update(protected_TEST_accessed=False,F_before={"a":"same"},F_after={"a":"same"},controller_source=r.record(controller),recon_initialization_training_fits=2,resolved_selection_locks=2,training_updates_completed=1200,checkpoint_states_saved=10,forecast_initialization_conditions_reused=2,probe_optimizer_updates=0,new_forecast_initialization_fits=0,all_2_locks_before_DEV=True)
        for key in ("new_backbone_training","new_D_initialization_fit","independent_holdout","strict_K_only","whole_deployment_cost_measured"): train[key]=False
        locks=[]
        for seed in r.SEEDS:
            row=train["datasets"]["robin"]["conditions"][str(seed)]
            row["selection_lock"]=put(str(seed)+"LOCK.json",{**row,"DEV_opened_before_lock":False})
            locks.append({"dataset":"robin","seed":seed,"status":"COMPLETE","selection_lock":row["selection_lock"]})
        train["selection_registry"]=put("REGISTRY.json",{"status":"ALL_TWO_RESOLVED_BEFORE_DEV","DEV_opened_before_seal":False,"conditions":locks})
        for key,target in (("full_gradient_parent","parent"),("causal_lag_diagnostic","science"),("learned_output_space_diagnostic","space")):
            train[key]=put(key+".json",values[target])
        for key in ("full_gradient_parent_audit","learned_output_space_audit","causal_lag_audit","plan"): train[key]=put(key+".json",{"status":"PASS"})
        audited_keys=("full_gradient_parent","full_gradient_parent_audit","learned_output_space_diagnostic","learned_output_space_audit","causal_lag_diagnostic","causal_lag_audit","selection_registry","plan")
        train_ref=put("TRAIN.json",train)
        audit.update(diagnostic=train_ref,audit_source=r.record(auditor),protected_TEST_accessed=False,totals={"checkpoint_states_checked":10},**{k:train[k] for k in audited_keys})
        audit_ref=put("AUDIT.json",audit); paths={"train":Path(train_ref["path"]),"audit":Path(audit_ref["path"])}
        failure=put("FAILED_AUDIT.json",{"status":"FAILED_PRESERVED"})
        with patch.multiple(r,FIXED_SHA={"train":train_ref["sha256"],"audit":audit_ref["sha256"]},AUDITOR_SOURCE=r.record(auditor),AUDIT_COUNTS={"checkpoint_states_checked":10},CONTROLLER_SHA=r.record(controller)["sha256"],PARENT_SHA=train["full_gradient_parent"]["sha256"],PRESERVED_AUDIT_FAILURE=failure):
            checked,_=r.load_verified(paths); self.assertEqual(checked["train"]["recon_initialization_training_fits"],2)
            paths["train"].write_text(paths["train"].read_text(encoding="utf-8")+" ",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"EXACT_COMPLETE_PASS"): r.load_verified(paths)
    def test_full_scores_new_and_reused_updates_and_masks(self):
        tables=self.r.collect_tables(fixture_values(self.folder))
        self.assertEqual({k:len(tables[k]) for k in ("SCORES","PAIRED","CHECKPOINTS","UPDATES","SPACE_CELLS","SPACE_METHODS")},{"SCORES":84,"PAIRED":24,"CHECKPOINTS":20,"UPDATES":2400,"SPACE_CELLS":8,"SPACE_METHODS":24})
        self.assertEqual(sum(r["new_execution"] for r in tables["UPDATES"]),1200)
        a=self.r.choose(tables["SCORES"],method="RECON_INIT_TRAINED",seed=92601,split="val",mask="all_observed",scale="normalized")
        b=self.r.choose(tables["SCORES"],method="RECON_INIT_TRAINED",seed=92601,split="val",mask="complete_C",scale="normalized")
        self.assertNotEqual(a["mse"],b["mse"])
    def test_TUNE_selection_cannot_be_replaced_by_DEV(self):
        values=fixture_values(self.folder); values["train"]["datasets"]["robin"]["conditions"]["92601"]["selected_update"]=100
        with self.assertRaisesRegex(ValueError,"TUNE"): self.r.collect_tables(values)
    def test_false_original_inequalities_and_new_diagnostic_role_are_preserved(self):
        tables=self.r.collect_tables(fixture_values(self.folder)); row=self.r.choose(tables["SPACE_METHODS"],initialization="RECON",seed=92601,split="val",method="ACTUAL_F")
        self.assertFalse(row["real_trace_bound_inequality_pass"]); self.assertTrue(row["retained_output_trace_bound_inequality_pass"])
        self.assertTrue(row["computed_during_current_audit"])
    def test_lead_text_separates_TUNE_decrease_from_DEV_initial_baseline_failure(self):
        values=fixture_values(self.folder); text=self.r.make_text(values,self.r.collect_tables(values),{},self.folder)
        for word in ("TUNE", "2/2", "RECON CP0", "DEV", "1,200", "재사용", "미측정", "strict K-only", "일반화", "감사 중 새 계산", "AdamW", "E(Y)","고정 D 대조","bytes", "v1", "v2"):
            self.assertIn(word,text)
    def test_synthetic_report_parse_relative_links_and_refuse_overwrite(self):
        values=fixture_values(self.folder/"metadata"); output=self.folder/"reports"/"toy"
        with patch.object(self.r,"HERE",self.folder),patch.object(self.r,"load_verified",return_value=(values,{})):
            manifest=self.r.generate({},output); self.assertEqual((manifest["PNG_count"],manifest["SVG_count"]),(3,3))
            for item in manifest["files"]:
                path=output/item["path"]; self.assertEqual(self.r.sha(path),item["sha256"])
                if path.suffix==".svg": ET.parse(path)
                if path.suffix==".png": self.assertEqual(path.read_bytes()[:8],b"\x89PNG\r\n\x1a\n")
            self.assertNotIn("E:/",(output/"REPORT_KO.md").read_text(encoding="utf-8"))
            with self.assertRaisesRegex(ValueError,"FRESH"): self.r.generate({},output)


if __name__=="__main__": unittest.main()
