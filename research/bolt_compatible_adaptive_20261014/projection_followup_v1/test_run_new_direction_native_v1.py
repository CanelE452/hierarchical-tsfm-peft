"""Synthetic metadata and small CPU native fits; no actual scientific payload or F."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

import run_new_direction_native_v1 as run
import lagged_native_head_v1 as native
from native_head_v1 import state_hash,score_predictions

torch.set_num_threads(2)


def fixture(k=2):
    rng = np.random.default_rng(3092); c,length,h = 4,40,5
    frozen = {"E":rng.normal(scale=.2,size=(c,k)).astype(np.float32),"D":rng.normal(scale=.1,size=(7*k,c)).astype(np.float32),"center":rng.normal(scale=.1,size=c).astype(np.float32)}
    model = native.LaggedNativeForecaster(frozen["center"],frozen["E"],frozen["D"],run.LAGS,L=length,H=h).eval()
    history = rng.normal(size=(8,length,c)).astype(np.float32); baseline = run.old.predict_components(model,None,history,lambda x:None)
    target = baseline["prediction"]+rng.normal(scale=.05,size=baseline["prediction"].shape).astype(np.float32)
    observed = np.ones_like(target,bool); observed[::2,:,0] = False; target[~observed] = np.nan
    raw = {"history":history,"target":target,"observed":observed,"raw_last":history[:,-1].copy(),"origins":np.arange(8,dtype=np.int64)+512,"mean":np.arange(c,dtype=np.float32),"std":np.arange(1,c+1,dtype=np.float32),"raw_forecast":baseline["prediction"]+np.float32(.1)}
    return model,raw,{key:value.copy() for key,value in baseline.items()},frozen


def result_fixture():
    return {"datasets":{data:{"conditions":{name:{"native":{}} for d,name in run.STATES if d == data}} for data in run.DATASETS},"failed_paths":[],"training_updates_completed":0,"checkpoint_states_saved":0,"native_training_fits":0,"DEV_evaluations_completed":0}


def metadata_fixture(folder):
    def put(name,value):
        path = folder/name; path.write_text(json.dumps(value,allow_nan=False),encoding="utf-8"); return run.old.record(path)
    state = put("STATE.json",{"synthetic_not_an_array":True}); registry = put("REGISTRY.json",{})
    refs = {"parent_ref":put("PARENT.json",{"synthetic":True})}; signature = {"same":"F"}; contract = {"median":"unchanged"}
    parent = {"status":"COMPLETE","protected_TEST_accessed":False,"failed_paths":[],"controller_source":run.old.record(run.cross.__file__),"module_source":run.old.record(run.HERE/"predictive_cross_moment_v1.py"),"state_registry":registry,"direction_fits_completed":8,"decoder_fits_completed":8,"resolved_state_locks":8,"FIT_TUNE_evaluations_completed":16,"DEV_evaluations_completed":8,"all_8_locks_before_DEV":True,"new_F_forward_calls":104,"new_F_forward_attempts":104,"new_F_model_objects_loaded":1,"F_before":signature,"F_after":signature,"runtime_behavior_contract":contract,**refs,"datasets":{data:{"conditions":{}} for data in run.DATASETS}}
    for data,name in run.STATES:
        kind,k = name.rsplit("_K",1); parent["datasets"][data]["conditions"][name] = {"status":"COMPLETE","kind":kind,"K":int(k),"C":run.DATASETS[data][0],"E_FIT_future_supervised":kind == "CROSS_MOMENT","D_FIT_future_supervised":False,"state_artifact":state,"splits":{split:{"status":"COMPLETE"} for split in ("fit","tune",run.DATASETS[data][2])}}
    binding = put("CROSS.json",parent)
    audit = {"status":"PASS","protected_TEST_accessed":False,"partial_paths":[],"diagnostic":binding,"audit_source":run.old.record(run.HERE/"audit_predictive_cross_moment_v1.py"),"totals":run.CROSS_AUDIT_COUNTS.copy(),"state_registry":registry,"datasets":{data:{"conditions":{name:{"state_artifact":state} for d,name in run.STATES if d == data}} for data in run.DATASETS}}
    ab = put("AUDIT.json",audit); args = SimpleNamespace(cross_result=binding["path"],cross_audit=ab["path"])
    calibration = {"F_parent_signatures_reused":{"before":signature,"after":signature}}
    return args,parent,audit,binding,ab,(calibration,{}, {},{"runtime_behavior_contract":contract},refs)


class NewDirectionNativeTests(unittest.TestCase):
    def test_eight_frozen_states_sixteen_fits_and_unchanged_recipe(self):
        self.assertEqual(len(run.STATES),8); self.assertEqual(len(set(run.CONDITIONS)),16)
        self.assertEqual({name for data,name in run.STATES if data == "robin"},{"PCA_K4","PCA_K8","CROSS_MOMENT_K4","CROSS_MOMENT_K8"})
        self.assertEqual({name for data,name in run.STATES if data == "beijing_aotizhongxin"},{"PCA_K2","PCA_K4","CROSS_MOMENT_K2","CROSS_MOMENT_K4"})
        self.assertEqual(run.RECIPE["updates_per_condition"],600); self.assertEqual(run.RECIPE["checkpoint_updates"],[0,100,200,400,600]); self.assertTrue(run.RECIPE["head_zero_initialized"])

    def test_actual_cross_audit_closed_gate_precedes_source_parent_and_array_IO(self):
        with patch.object(run,"CROSS_AUDIT_SHA",None),patch.object(run.old,"bound",side_effect=AssertionError("closed source IO")),patch.object(run.cross,"load_gates",side_effect=AssertionError("closed parent IO")):
            with self.assertRaisesRegex(ValueError,"NOT_YET_REGISTERED"): run.load_gates(SimpleNamespace())

    def test_synthetic_metadata_gate_accepts_exact_bindings_and_rejects_mutations(self):
        with tempfile.TemporaryDirectory(dir=run.HERE) as temp:
            values = metadata_fixture(Path(temp)); args,parent,audit,binding,ab,parents = values
            with patch.object(run,"CROSS_RESULT_SHA",binding["sha256"]),patch.object(run,"CROSS_AUDIT_SHA",ab["sha256"]),patch.object(run.cross,"load_gates",return_value=copy.deepcopy(parents)):
                returned = run.load_gates(args); self.assertEqual(len(returned),6); self.assertEqual(returned[-1]["predictive_cross_diagnostic"],binding)
            for kind in ("digest","audit_count","state_binding","F","partial","K","split","TEST"):
                p,a = copy.deepcopy((parent,audit))
                if kind == "audit_count": a["totals"]["states_checked"] = 7
                elif kind == "state_binding": a["datasets"]["robin"]["conditions"]["PCA_K4"]["state_artifact"]["sha256"] = "0"*64
                elif kind == "F": p["F_after"] = {"different":"F"}
                elif kind == "partial": a["partial_paths"] = [{"synthetic":"failure"}]
                elif kind == "K": p["datasets"]["robin"]["conditions"]["PCA_K4"]["K"] = 8
                elif kind == "split": p["datasets"]["beijing_aotizhongxin"]["conditions"]["PCA_K2"]["splits"]["eval"]["status"] = "FAILED"
                elif kind == "TEST": p["protected_TEST_accessed"] = True
                Path(binding["path"]).write_text(json.dumps(p),encoding="utf-8"); tb = run.old.record(binding["path"]); a["diagnostic"] = tb
                Path(ab["path"]).write_text(json.dumps(a),encoding="utf-8"); ta = run.old.record(ab["path"])
                pin = "0"*64 if kind == "digest" else tb["sha256"]
                with self.subTest(kind=kind),patch.object(run,"CROSS_RESULT_SHA",pin),patch.object(run,"CROSS_AUDIT_SHA",ta["sha256"]),patch.object(run.cross,"load_gates",return_value=copy.deepcopy(parents)),self.assertRaises(ValueError): run.load_gates(args)

    def test_fresh_native_forward_rejects_F_and_has_no_future_label_arguments(self):
        model,raw,cached,frozen = fixture()
        with self.assertRaisesRegex(ValueError,"NO_F_CALLBACK"): model(torch.from_numpy(raw["history"]),lambda x:x)
        initial = run.old.predict_components(model,None,raw["history"],lambda x:None)
        raw["target"][:] = np.inf; raw["observed"][:] = False
        repeat = run.old.predict_components(model,None,raw["history"],lambda x:None)
        for key in initial: self.assertEqual(initial[key].tobytes(),repeat[key].tobytes())
        self.assertEqual([name for name,p in model.named_parameters() if p.requires_grad],["head.weight","head.bias"])

    def test_actual_saved_native_latents_old_lag_D_scores_masks_and_CP0_probe(self):
        for k in (1,2):
            model,raw,cached,frozen = fixture(k); outputs = run.old.predict_components(model,None,raw["history"],lambda x:None)
            entry,independent = run.evaluate_arrays(outputs,raw,frozen,cached); self.assertEqual(entry["status"],"COMPLETE"); self.assertTrue(entry["same_D_and_current_C_anchor"])
            self.assertEqual(entry["scores_all_observed"]["normalized"]["observed_counts"],raw["observed"].sum((0,1)).tolist()); self.assertEqual(set(independent),{"independent_NumPy_prediction","independent_NumPy_lag_features"})
            probe,arrays = run.matched.cp0_probe(outputs,cached,frozen,raw); self.assertEqual(probe["status"],"PASS"); self.assertFalse(probe["latent_persistence_is_current_C_persistence_claimed"])
            self.assertFalse(np.array_equal(outputs["prediction"],np.broadcast_to(raw["raw_last"][:,None],outputs["prediction"].shape)))
            for kind in ("history","C","features"):
                changed = copy.deepcopy(outputs); changed[{"history":"latent_history","C":"prediction","features":"lag_features"}[kind]][0,0,0] += .1
                wrong,_ = run.evaluate_arrays(changed,raw,frozen,cached)
                with self.subTest(k=k,kind=kind): self.assertEqual(wrong["status"],"FAILED_PARITY_PRESERVED")
            json.dumps(entry,allow_nan=False)

    def test_unobserved_labels_no_effect_and_empty_complete_mask_is_explicit(self):
        model,raw,cached,frozen = fixture(); outputs = run.old.predict_components(model,None,raw["history"],lambda x:None)
        original,_ = run.evaluate_arrays(outputs,raw,frozen,cached); raw["target"][~raw["observed"]] = np.inf
        changed,_ = run.evaluate_arrays(outputs,raw,frozen,cached); self.assertEqual(original,changed)
        raw["observed"][:,:,3] = False; raw["target"][:,:,3] = np.nan
        entry,_ = run.evaluate_arrays(outputs,raw,frozen,cached); self.assertEqual(entry["scores_complete_mask"],{"status":"NO_COMPLETE_C"}); self.assertEqual(entry["complete_target_positions"],0)

    def test_tiny_reused_trainer_changes_only_head_and_restores_own_selected_state(self):
        for seed in run.SEEDS:
            model,raw,cached,frozen = fixture(); initial = run.old.state_arrays(model); events,checkpoints = [],[]
            def checkpoint(update,state,score,outputs):
                entry,_ = run.evaluate_arrays(outputs,raw,frozen,cached); run.check_score_link(score,entry)
                self.assertEqual(entry["status"],"COMPLETE"); checkpoints.append({"update":update,"state":copy.deepcopy(state),"tune":score})
                return entry
            trained = native.train_head(model,raw,raw,seed,events.append,checkpoint,updates=3)
            selected = min(checkpoints,key=lambda cp:(cp["tune"]["mse"],cp["update"]))
            with self.subTest(seed=seed):
                self.assertEqual(trained["selected_update"],selected["update"]); self.assertEqual(trained["selected_state_sha256"],state_hash(selected["state"]))
                self.assertEqual(trained["FIT_channel_observed_counts"],raw["observed"].sum((0,1)).tolist()); self.assertEqual(trained["post_optimizer_frozen_flag_checks"],3)
                for key in frozen: self.assertEqual(initial[key].tobytes(),run.old.state_arrays(model)[key].tobytes())
                updates = [e for e in events if e["stage"] == "TRAINING_UPDATE"]; self.assertEqual(len(updates),3)
                self.assertTrue(all(e["gradient_clipping_parameter_scope"] == "HEAD_ONLY" and e["E_grad_is_None"] and e["D_grad_is_None"] for e in updates))
                expected_order = np.random.default_rng(seed).integers(0,len(raw["history"]),size=(3,4)); self.assertEqual(trained["order_sha256"],hashlib.sha256(expected_order.tobytes()).hexdigest())

    def test_zero_decoder_keeps_exact_CP0_tie_in_reused_trainer(self):
        model,raw,cached,frozen = fixture(); model.D.data.zero_(); events=[]
        trained = native.train_head(model,raw,raw,92601,events.append,lambda *values:{},updates=2)
        self.assertEqual(trained["selected_update"],0); self.assertFalse(trained["selected_head_nonzero"])

    def test_score_reduction_link_has_fixed_tolerance_and_does_not_rewrite_selection(self):
        model,raw,cached,frozen = fixture(); outputs = run.old.predict_components(model,None,raw["history"],lambda x:None)
        entry,_ = run.evaluate_arrays(outputs,raw,frozen,cached); score = score_predictions(outputs["prediction"],raw["target"],raw["observed"]); before = copy.deepcopy(score)
        run.check_score_link(score,entry); self.assertEqual(score,before)
        entry["scores_all_observed"]["normalized"]["mse"] += 1e-13; run.check_score_link(score,entry)
        entry["scores_all_observed"]["normalized"]["mae"] += .1
        with self.assertRaisesRegex(ValueError,"TUNE_SCORE_LINK"): run.check_score_link(score,entry)

    def test_sixteen_failure_or_success_locks_precede_any_DEV_and_preserve_partial_updates(self):
        result = result_fixture(); events=[]
        def execute(data,name,seed):
            if (data,name,seed) == run.CONDITIONS[0]:
                result["datasets"][data]["conditions"][name]["native"][str(seed)] = {"training_updates_completed":37,"checkpoint_count":2,"checkpoints":[{"artifact":{"path":"preserved"}}]}
                raise RuntimeError("synthetic update38 failure")
            return {"status":"COMPLETE","training_updates_completed":600,"checkpoint_count":5}
        def seal(data,name,seed,row): events.append(("lock",data,name,seed)); return {"path":data+name+str(seed)}
        def registry(rows): events.append(("registry",len(rows))); return {"path":"registry"}
        resolved = run.resolve_all(result,execute,seal,registry)
        self.assertEqual(len(resolved),15); self.assertEqual(events[-1],("registry",16)); self.assertEqual(result["training_updates_completed"],9037); self.assertEqual(result["checkpoint_states_saved"],77)
        data,name,seed = run.CONDITIONS[0]; failed = result["datasets"][data]["conditions"][name]["native"][str(seed)]; self.assertEqual(failed["checkpoints"][0]["artifact"]["path"],"preserved")
        def development(data,name,seed,row):
            self.assertEqual(sum(e[0] == "lock" for e in events),16); self.assertEqual(events[-1],("registry",16)); return {"status":"COMPLETE"}
        run.infer_development(result,resolved,development); self.assertEqual(result["DEV_evaluations_completed"],15)
        with self.assertRaisesRegex(ValueError,"SIXTEEN"): run.infer_development({"all_16_locks_before_DEV":False},resolved,development)

    def test_all_sixteen_successes_count_9600_updates_eighty_CPs_and_sixteen_DEV(self):
        result = result_fixture(); events=[]
        def seal(data,name,seed,row): events.append("lock"); return {"path":data+name+str(seed)}
        def registry(rows): self.assertEqual(len(rows),16); events.append("registry"); return {"path":"registry"}
        resolved = run.resolve_all(result,lambda *args:{"status":"COMPLETE","training_updates_completed":600,"checkpoint_count":5},seal,registry)
        def development(*args): self.assertEqual(events,["lock"]*16+["registry"]); return {"status":"COMPLETE"}
        run.infer_development(result,resolved,development)
        self.assertEqual((result["native_training_fits"],result["training_updates_completed"],result["checkpoint_states_saved"],result["DEV_evaluations_completed"]),(16,9600,80,16))

    def test_import_and_help_do_not_import_numpy_torch_or_open_scientific_inputs(self):
        command = "import sys;sys.path.insert(0,"+repr(str(run.HERE))+");import run_new_direction_native_v1;assert 'numpy' not in sys.modules and 'torch' not in sys.modules;print('PURE_IMPORT_PASS')"
        result = subprocess.run([sys.executable,"-c",command],capture_output=True,text=True,check=True); self.assertIn("PURE_IMPORT_PASS",result.stdout)
        result = subprocess.run([sys.executable,run.__file__,"--help"],capture_output=True,text=True,check=True); self.assertIn("--cross-audit",result.stdout)


if __name__ == "__main__": unittest.main()
