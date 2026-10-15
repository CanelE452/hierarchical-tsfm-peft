"""Generated JSON/JSONL only; scientific NPZ paths intentionally do not exist."""
from pathlib import Path
import importlib.util
import json
import statistics
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch


SOURCE=Path(__file__).with_name("make_f_input_gradient_report_v1.py")


def module():
    if not SOURCE.exists(): raise AssertionError("F-input gradient report implementation is missing")
    spec=importlib.util.spec_from_file_location("f_gradient_report_fixture",SOURCE)
    result=importlib.util.module_from_spec(spec); spec.loader.exec_module(result); return result


def fixture_values(folder):
    r=module(); folder.mkdir(parents=True,exist_ok=True)
    def put(name,value):
        path=folder/name; path.write_text(json.dumps(value,allow_nan=False),encoding="utf-8"); return r.record(path)
    def opaque(name): return {"path":str(folder/(name+".npz")),"sha256":"a"*64,"bytes":42,"arrays":{"E":{"shape":[2,1],"dtype":"float32","sha256":"b"*64}}}
    def score(value,c):
        return {scale:{"mse":value*f,"mae":(value*f)**.5,"per_channel_mse":[value*f]*c,"per_channel_mae":[(value*f)**.5]*c,"observed_counts":[48]*c,"active_channels":c} for scale,f in (("normalized",1),("physical_per_channel_diagnostic",4))}
    def forecast(value,c,name): return {"scores_all_observed":score(value,c),"scores_complete_mask":score(value+.01,c),"artifact":opaque(name)}
    parent={"status":"COMPLETE","protected_TEST_accessed":False,"datasets":{},"CP0_LS_and_RAW_baselines":{},"F_before":{"model":"same"}}
    train={"status":"COMPLETE","protected_TEST_accessed":False,"datasets":{},"failed_paths":[],"stop_gradient_training_fits":4,"training_updates_completed":2400,
        "checkpoint_states_saved":20,"probe_optimizer_updates":0,"full_gradient_conditions_reused":4,"new_full_gradient_fits":0,
        "new_backbone_training":False,"independent_holdout":False,"strict_K_only":False,"whole_deployment_cost_measured":False,
        "all_4_locks_before_DEV":True,"resolved_selection_locks":4,"F_before":{"model":"same"},"F_after":{"model":"same"},
        "D_step_scope":"Preclip scalar reconstructed scaling; not actual per-step AdamW updates","resource_terminal":{"wall_seconds":2.,"process_cpu_seconds":1.}}
    locks=[]
    for data,(c,_,dev) in r.DATASETS.items():
        parent["datasets"][data]={"conditions":{}}; parent["CP0_LS_and_RAW_baselines"][data]={"L0":{split:{"CP0_LS":forecast(.9,c,data+split+"CP0"),"RAW_scores":score(.6,c)} for split in ("fit",dev)}}
        train["datasets"][data]={"conditions":{}}
        for seed in r.SEEDS:
            for method in ("E_FIXED","FULL_GRADIENT","STOP_GRADIENT"):
                name=data+"_"+method+"_s"+str(seed); cps=[]
                for update in (0,100,200,400,600):
                    value=.9 if update==0 else (.75 if method=="FULL_GRADIENT" else .78 if method=="STOP_GRADIENT" else .85)+(0 if update in (200,400) else .03)
                    cps.append({"update":update,"tune":score(value,c)["normalized"],"state":opaque(name+str(update)+"S"),
                        "parameter_diagnostics":{"E_l2_norm":1.,"D_l2_norm":2.,"E_singularvalues":[1.],"D_feature_to_C_operator_gain":2.},**forecast(value,c,name+str(update))})
                row={"status":"COMPLETE","dataset":data,"seed":seed,"lag":"L0","learn_E":method!="E_FIXED","selected_update":200,
                    "selected_state":cps[2]["state"],"selected_tune":cps[2]["tune"],"checkpoints":cps,"checkpoint_count":5,
                    "FIT":forecast(.7,c,name+"FIT"),"DEV":forecast(.8 if method=="FULL_GRADIENT" else .82,c,name+"DEV"),"CP0_TUNE":cps[0],
                    "RAW_TUNE_scores":score(.6,c),"training_updates_completed":600,"global_clipping_fraction":.5,"order_sha256":"c"*64}
                if method!="E_FIXED":
                    log=folder/(name+".jsonl")
                    with log.open("w",encoding="utf-8") as handle:
                        for update in range(1,601): handle.write(json.dumps({"stage":"TRAINING_UPDATE","update":update,"loss":1/update,"E_gradient_norm_before_clip":.4,
                            "D_gradient_norm_before_clip":.6,"global_gradient_norm_before_clip":2. if update%2 else .5,"clipped":bool(update%2)})+"\n")
                    row["updates_log"]=r.record(log)
                if method=="STOP_GRADIENT":
                    row["probe"]={"status":"PASS","optimizer_updates":0,"forward_byte_exact":True,"finite_gradient_vectors":True,"artifact":opaque(name+"PROBE"),
                        "gradient_checks":{key:{"pass":True,"max_abs":0.,"atol":1e-5,"rtol":2e-6} for key in ("stop_E_equals_full_minus_F_path","stop_D_equals_full_D")},
                        "gradient_norms":{"full_E_gradient":.5,"F_path_E_gradient":.3,"direct_E_gradient":.4,"stop_E_gradient":.4,"full_D_gradient":.6,"stop_D_gradient":.6},
                        "F_parameters_frozen_and_grad_none":True,"parameter_grad_fields_none":True}
                    lock=put(name+"LOCK.json",{**row,"DEV_opened_before_lock":False,"sealed_unix":1.}); row["selection_lock"]=lock
                    train["datasets"][data]["conditions"][str(seed)]=row; locks.append({"dataset":data,"seed":seed,"status":"COMPLETE","selection_lock":lock})
                else:
                    lock=put(name+"LOCK.json",{"status":"COMPLETE","selected_update":200}); row["selection_lock"]=lock
                    parent["datasets"][data]["conditions"][f"{data}_L0_{'E_LEARN' if method=='FULL_GRADIENT' else 'E_FIXED'}_s{seed}"]=row
    train["selection_registry"]=put("REGISTRY.json",{"status":"ALL_FOUR_RESOLVED_BEFORE_DEV","conditions":locks,"DEV_opened_before_seal":False})
    train["full_gradient_parent"]=put("PARENT.json",parent); train["full_gradient_parent_audit"]=put("PARENT_AUDIT.json",{"status":"PASS","protected_TEST_accessed":False})
    cost={"status":"COMPLETE","protected_TEST_accessed":False,"cost_split":"tune","routes":list(r.ROUTES),"cost":{},"completed_cost_cells":20,"recorded_timer_samples":100,"failed_paths":[],
        "F_before":{"model":"same"},"F_after":{"model":"same"},"selected_states_before":{},"selected_states_after":{},"new_training_updates":0,"new_backbone_training":False,
        "cached_future_returned_instead_of_actual_F_inside_timer":False,"resource_terminal":{"wall_seconds":1.}}
    for data in r.DATASETS:
        rows=[]
        for batch in (1,8):
            for position,route in enumerate(r.ROUTES):
                seconds=[(.01+i*.001)*(1 if route=="RAW_BOLT" else .8) for i in range(5)]
                samples=[{"repeat":i,"position":(position-i)%5,"seconds":value,"perf_start_ns":1000000000+i*100000000,"perf_end_ns":1000000000+i*100000000+round(value*1e9)} for i,value in enumerate(seconds)]
                rows.append({"route":route,"batch":batch,"seconds":seconds,"samples":samples,"median_seconds":statistics.median(seconds),"median_seconds_per_window":statistics.median(seconds)/batch,
                    "paired_RAW_ratios":[1. if route=="RAW_BOLT" else .8]*5,"paired_RAW_median_ratio":1. if route=="RAW_BOLT" else .8})
        cost["cost"][data]={"rows":rows}
    return {"train":train,"parent":parent,"audit":{"status":"PASS","protected_TEST_accessed":False,"totals":{}},"cost":cost,"cost_audit":{"status":"PASS","protected_TEST_accessed":False,"totals":{}}}


def gate_fixture(folder):
    r=module(); values=fixture_values(folder)
    def put(name,value):
        path=folder/name; path.write_text(json.dumps(value,allow_nan=False),encoding="utf-8"); return r.record(path)
    plan=folder/"PLAN.md"; plan.write_text("Generated scope\n",encoding="utf-8")
    source=folder/"SOURCE.py"; source.write_text("# generated audit source\n",encoding="utf-8")
    report=folder/"PRIOR.md"; report.write_text("Generated prior report\n",encoding="utf-8")
    prior=put("PRIOR.json",{"status":"COMPLETE"}); train=values["train"]
    train.update(controller_source=r.record(source),plan=r.record(plan),parent_report=r.record(report),parent_report_manifest=prior,parent_report_audit=prior)
    train_ref=put("TRAIN.json",train)
    parents={key:train[key] for key in ("full_gradient_parent","full_gradient_parent_audit")}
    audit=values["audit"]; audit.update(diagnostic=train_ref,**parents,audit_source=r.record(source),totals={"stop_gradient_fits_checked":4})
    audit_ref=put("AUDIT.json",audit)
    common={**parents,"stop_gradient_diagnostic":train_ref,"stop_gradient_audit":audit_ref,"selection_registry":train["selection_registry"]}
    cost=values["cost"]; cost.update(**common,cost_plan=r.record(plan)); cost_ref=put("COST.json",cost)
    values["cost_audit"].update(**common,cost=cost_ref,audit_source=r.record(source),totals={"cost_cells_checked":20}); put("COST_AUDIT.json",values["cost_audit"])
    paths={key:folder/name for key,name in (("train","TRAIN.json"),("audit","AUDIT.json"),("cost","COST.json"),("cost_audit","COST_AUDIT.json"))}
    patches={"FIXED_SHA":{key:r.sha(path) for key,path in paths.items()},"AUDITOR_SOURCES":{key:r.record(source) for key in ("audit","cost_audit")},
        "AUDIT_COUNTS":{"stop_gradient_fits_checked":4},"COST_COUNTS":{"cost_cells_checked":20},"PARENT_SHA":train["full_gradient_parent"]["sha256"],"PARENT_AUDIT_SHA":train["full_gradient_parent_audit"]["sha256"]}
    return paths,patches


class ReportTests(unittest.TestCase):
    def setUp(self): self.r=module(); self.temp=tempfile.TemporaryDirectory(dir=SOURCE.parent); self.folder=Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()
    def test_unregistered_actual_gate_reads_nothing(self):
        with patch.object(self.r,"FIXED_SHA",{key:None for key in self.r.INPUTS}),patch.object(self.r,"read",side_effect=AssertionError("no read")),self.assertRaisesRegex(ValueError,"NOT_YET_REGISTERED"):
            self.r.load_verified({key:"missing" for key in self.r.INPUTS})
    def test_four_pairs_all_masks_scales_new_and_reused_counts(self):
        values=fixture_values(self.folder); tables=self.r.collect_tables(values)
        self.assertEqual(len(tables["PAIRED"]),48); self.assertEqual(len(tables["SCORES"]),180)
        self.assertEqual(len(tables["CHECKPOINTS"]),40); self.assertEqual(len(tables["UPDATES"]),4800)
        self.assertEqual(sum(r["new_execution"] for r in tables["UPDATES"]),2400)
        pair=next(row for row in tables["PAIRED"] if row["split"]=="tune" and row["mask"]=="all_observed" and row["scale"]=="normalized")
        self.assertAlmostEqual(pair["MSE_stop_minus_full"],.03)
        self.assertEqual(len(tables["PROBES"]),4); self.assertEqual(len(tables["WHOLE_COST_SAMPLES"]),100)
    def test_positive_exact_metadata_gate_and_modified_file_rejection(self):
        paths,patches=gate_fixture(self.folder)
        with patch.multiple(self.r,**patches):
            values,bindings=self.r.load_verified(paths); self.assertEqual(values["train"]["stop_gradient_training_fits"],4)
            paths["audit"].write_text(paths["audit"].read_text()+" ")
            with self.assertRaisesRegex(ValueError,"EXACT_COMPLETE"): self.r.load_verified(paths)
    def test_later_tied_checkpoint_or_missing_seed_rejected(self):
        values=fixture_values(self.folder); values["train"]["datasets"]["robin"]["conditions"]["92601"]["selected_update"]=400
        with self.assertRaisesRegex(ValueError,"EARLIEST_TUNE"): self.r.collect_tables(values)
        values=fixture_values(self.folder); values["train"]["datasets"]["robin"]["conditions"].pop("92602")
        with self.assertRaisesRegex(ValueError,"EXACT_FOUR"): self.r.collect_tables(values)
    def test_probe_update_or_failed_relation_cannot_be_reported_as_pass(self):
        values=fixture_values(self.folder); values["train"]["datasets"]["robin"]["conditions"]["92601"]["probe"]["optimizer_updates"]=1
        with self.assertRaisesRegex(ValueError,"ZERO_UPDATE"): self.r.collect_tables(values)
    def test_clip_scaling_is_formula_not_adamw_update(self):
        tables=self.r.collect_tables(fixture_values(self.folder)); row=tables["UPDATES"][0]
        self.assertAlmostEqual(row["global_clip_scale_from_logged_norm"],1/(2+1e-6)); self.assertFalse(row["D_gradient_scaling_is_AdamW_step_norm"])
        self.assertAlmostEqual(tables["CLIP_SUMMARY"][0]["clipping_fraction"],.5)
    def test_two_cost_ratios_and_whole_cohort_coverage(self):
        values=fixture_values(self.folder); row=values["cost"]["cost"]["robin"]["rows"][1]
        row["seconds"]=[.001,.001,.001,.1,.1]; row["median_seconds"]=.001; row["median_seconds_per_window"]=.001
        raw=values["cost"]["cost"]["robin"]["rows"][0]["seconds"]; row["paired_RAW_ratios"]=[a/b for a,b in zip(row["seconds"],raw)]; row["paired_RAW_median_ratio"]=statistics.median(row["paired_RAW_ratios"])
        result=self.r.collect_tables(values)["WHOLE_COST"][1]; self.assertNotEqual(result["ratio_of_RAW_medians"],result["median_paired_RAW_ratio"])
    def test_text_distinguishes_F_gradient_values_and_exploratory_scope(self):
        values=fixture_values(self.folder); text=self.r.make_text(values,self.r.collect_tables(values),{},self.folder)
        for phrase in ("2,400","probe", "기노출", "minibatch", "clipping", "AdamW", "필요성", "F 예측값", "독립 일반화", "재사용"):
            self.assertIn(phrase,text)
    def test_generated_only_report_images_and_fresh_output(self):
        values=fixture_values(self.folder/"metadata"); output=self.folder/"reports"/"toy"
        with patch.object(self.r,"HERE",self.folder),patch.object(self.r,"load_verified",return_value=(values,{})):
            result=self.r.generate({},output); self.assertEqual(result["PNG_count"],4); self.assertEqual(result["SVG_count"],4)
            for item in result["files"]:
                path=output/item["path"]; self.assertEqual(self.r.sha(path),item["sha256"])
                if path.suffix==".png": self.assertEqual(path.read_bytes()[:8],b"\x89PNG\r\n\x1a\n")
                if path.suffix==".svg": ET.parse(path)
            with self.assertRaisesRegex(ValueError,"FRESH"): self.r.generate({},output)


if __name__=="__main__": unittest.main()
