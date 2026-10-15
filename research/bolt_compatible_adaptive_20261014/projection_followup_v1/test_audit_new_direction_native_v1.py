"""Independent saved native arithmetic on generated CPU arrays and metadata only."""
from pathlib import Path
from unittest.mock import patch
import ast
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

import audit_new_direction_native_v1 as audit
import run_new_direction_native_v1 as producer
from lagged_native_head_v1 import LaggedNativeForecaster

audit.initialize_numpy(); torch.set_num_threads(2)


def fixture(*,zero=False,k=2):
    rng = np.random.default_rng(8932); c,n,length,h = 4,4,40,48
    frozen = {"E":rng.normal(scale=.2,size=(c,k)).astype(np.float32),"D":rng.normal(scale=.1,size=(7*k,c)).astype(np.float32),"center":rng.normal(scale=.1,size=c).astype(np.float32)}
    model = LaggedNativeForecaster(frozen["center"],frozen["E"],frozen["D"],audit.LAGS,L=length,H=h).eval()
    history = rng.normal(size=(n,length,c)).astype(np.float32); baseline = producer.old.predict_components(model,None,history,lambda x:None)
    if not zero:
        with torch.no_grad():
            model.head.weight.copy_(torch.from_numpy(rng.normal(scale=.002,size=(k,h,length)).astype(np.float32)))
            model.head.bias.copy_(torch.from_numpy(rng.normal(scale=.002,size=(k,h)).astype(np.float32)))
    outputs = producer.old.predict_components(model,None,history,lambda x:None)
    target = outputs["prediction"]+rng.normal(scale=.03,size=outputs["prediction"].shape).astype(np.float32); observed = np.ones_like(target,bool); observed[::2,:,0] = False; target[~observed] = np.nan
    raw = {"history":history,"target":target,"observed":observed,"raw_last":history[:,-1].copy(),"origins":np.arange(n,dtype=np.int64)+512,"mean":np.arange(c,dtype=np.float32),"std":np.arange(1,c+1,dtype=np.float32),"raw_forecast":baseline["prediction"]+np.float32(.1)}
    row,independent = producer.evaluate_arrays(outputs,raw,frozen,baseline); row.update(artifact={"path":"synthetic_forecast","sha256":"a"*64},independent_artifact={"path":"synthetic_independent","sha256":"b"*64})
    arrays = {**outputs,**{key:raw[key].copy() for key in ("target","observed","raw_last","origins","mean","std")}}
    return row,arrays,independent,raw,baseline,producer.old.state_arrays(model)


def probe_fixture():
    row,arrays,independent,raw,cached,state = fixture(zero=True); frozen = {k:state[k] for k in ("E","D","center")}
    outputs = {key:arrays[key] for key in ("prediction","latent_history","latent_future","lag_features")}
    probe,values = producer.matched.cp0_probe(outputs,cached,frozen,raw); probe["artifact"] = {"path":"synthetic_probe","sha256":"c"*64}; values["origins"] = raw["origins"].copy()
    return probe,{key:value.copy() for key,value in values.items()},arrays,cached,raw,frozen


def selection_fixture():
    _,_,_,_,_,base_state = fixture(zero=True); states = {}; checkpoints=[]
    for update in audit.CPS:
        state = copy.deepcopy(base_state)
        if update: state["head.weight"][:] = np.float32(update*1e-5)
        states[update] = state; head = {key.split(".",1)[1]:state[key] for key in ("head.weight","head.bias")}
        checkpoints.append({"update":update,"tune":{"mse":.1,"mae":.2},"state":{"path":"synthetic_CP"+str(update),"sha256":str(update).zfill(64)},"head_state_sha256":audit.base.state_hash(head),"frozen_status":audit.native.FLAGS.copy(),"E_D_center_exact_initial_bytes":True,"canonical_score_link_atol":1e-12,"canonical_score_link_rtol":1e-12})
    selected = checkpoints[0]
    row = {"checkpoints":checkpoints,"checkpoint_count":5,"CP0_TUNE":checkpoints[0],"selected_update":0,"selected_tune":selected["tune"].copy(),"selected_state":selected["state"].copy(),"selected_state_sha256":audit.base.state_hash(states[0]),"selected_head_state_sha256":selected["head_state_sha256"],"initial_full_state_sha256":audit.base.state_hash(states[0]),"initial_head_state_sha256":selected["head_state_sha256"],"TUNE":{k:v for k,v in selected.items() if k not in ("state","frozen_status","head_state_sha256","E_D_center_exact_initial_bytes")},"training_head_values_changed":True,"selected_head_nonzero":False}
    return row,states


def registry_fixture():
    result = {"all_16_locks_before_DEV":True,"resolved_selection_locks":16,"datasets":{data:{"conditions":{name:{"native":{}} for d,name in audit.STATES if d == data}} for data in audit.DATASETS}}
    registry = {"status":"ALL_SIXTEEN_RESOLVED_BEFORE_DEV","DEV_opened_before_seal":False,"sealed_unix":20.,"conditions":[]}; locks={}
    for data,name in audit.STATES:
        for seed in audit.SEEDS:
            path = data+name+str(seed); binding={"path":path,"sha256":"a"*64}; row={"status":"COMPLETE","dataset":data,"condition":name,"seed":seed,"selected_update":100,"selection_lock":binding,"DEV":{"status":"COMPLETE"}}
            result["datasets"][data]["conditions"][name]["native"][str(seed)] = row
            registry["conditions"].append({"dataset":data,"condition":name,"seed":seed,"status":"COMPLETE","selection_lock":binding})
            locks[path] = {k:v for k,v in row.items() if k not in ("DEV","selection_lock")}|{"DEV_opened_before_lock":False,"sealed_unix":10.}
    return result,registry,locks


def logs_fixture(folder):
    _,_,_,raw,_,_ = fixture(); path = folder/"SYNTHETIC_UPDATES.jsonl"
    logs = [{"stage":"TRAINING_UPDATE","update":n,"loss":.1,"head_weight_gradient_norm_before_clip":.3,"head_bias_gradient_norm_before_clip":.4,"global_gradient_norm_before_clip":.5,"clipped":False,"gradient_clipping_parameter_scope":"HEAD_ONLY",**audit.native.FLAGS} for n in range(1,601)]
    path.write_text("\n".join(json.dumps(v) for v in logs),encoding="utf-8"); digest=hashlib.sha256(np.random.default_rng(92601).integers(0,len(raw["history"]),size=(600,4)).tobytes()).hexdigest()
    row = {"updates_log":audit.base.file_record(path),"training_updates_completed":600,"seed":92601,"order_sha256":digest,"FIT_channel_observed_counts":raw["observed"].sum((0,1)).tolist(),"post_optimizer_frozen_flag_checks":600,"frozen_state_checkpoint_byte_checks":5,"final_frozen_status":audit.native.FLAGS.copy(),"frozen_bytes_checked_at_CP_and_final_only":True,"independent_per_update_frozen_bytes_replayed":False,"gradient_clipping_parameter_scope":"HEAD_ONLY","global_clipping_fraction":0.,"loss":"N_fit/B/C_active * sum_c SSE_batch_c/N_fit_c; fixed entire-FIT counts"}
    return row,raw,digest,logs,path


class NewDirectionNativeAuditTests(unittest.TestCase):
    def test_independent_per_stream_affine_future_old_lag_C_and_observed_macro_scores(self):
        for k in (1,2):
            values = fixture(k=k); result = audit.check_cell(*values)
            with self.subTest(k=k):
                self.assertTrue(result["independent_NumPy_head_future_parity"]["pass"]); self.assertEqual(result["score_fields_checked"],72); self.assertFalse(result["new_F_forward_replayed"]); json.dumps(result,allow_nan=False)
            state,past = values[-1],values[1]["latent_history"]; expected=np.empty((4,48,k),np.float32)
            for n in range(4):
                for stream in range(k):
                    for h in range(48): expected[n,h,stream] = past[n,-1,stream]+((past[n,:,stream]-past[n,-1,stream])@state["head.weight"][stream,h]+state["head.bias"][stream,h])
            np.testing.assert_allclose(audit.native.head_future(state,past),expected,atol=1e-5,rtol=2e-6)

    def test_head_future_C_features_original_context_and_scores_tampering_rejected(self):
        for kind in ("future","C","history","features","independent","head","origins","score","count","tol","dtype"):
            v=fixture()
            if kind == "future": v[1]["latent_future"][0,0,0] += .1
            elif kind == "C": v[1]["prediction"][0,0,0] += .1
            elif kind == "history": v[1]["latent_history"][0,0,0] += .1
            elif kind == "features": v[1]["lag_features"][0,0,0] += .1
            elif kind == "independent": v[2]["independent_NumPy_lag_features"][0,0,0] += .1
            elif kind == "head": v[5]["head.bias"][0,0] += .1
            elif kind == "origins": v[1]["origins"][0] += 1
            elif kind == "score": v[0]["RAW_BOLT_scores"]["all_observed"]["normalized"]["mse"] += .1
            elif kind == "count": v[0]["complete_target_positions"] += 1
            elif kind == "tol": v[0]["checks"]["independent_NumPy_old_lag_C"]["atol"] = .1
            else: v[1]["latent_future"] = v[1]["latent_future"].astype(np.float64)
            with self.subTest(kind=kind),self.assertRaises(ValueError): audit.check_cell(*v)

    def test_auxiliary_fresh_E_failure_is_recomputed_without_replacing_cached_history_gate(self):
        row,arrays,independent,raw,cached,state = fixture(); frozen = {k:state[k] for k in ("E","D","center")}
        arrays["latent_history"] += np.float32(.1); cached["latent_history"] = arrays["latent_history"].copy(); arrays["latent_future"] = audit.native.head_future(state,arrays["latent_history"])
        arrays["lag_features"] = audit.base.delta_features(arrays["latent_history"],arrays["latent_future"],audit.LAGS); arrays["prediction"] = raw["raw_last"][:,None]+arrays["lag_features"]@state["D"]
        outputs = {k:arrays[k] for k in ("prediction","latent_history","latent_future","lag_features")}; row,independent = producer.evaluate_arrays(outputs,raw,frozen,cached); row.update(artifact={"path":"synthetic","sha256":"a"*64},independent_artifact={"path":"synthetic_i","sha256":"b"*64})
        result = audit.check_cell(row,arrays,independent,raw,cached,state); self.assertFalse(result["NumPy_fresh_E_auxiliary_parity"]["pass"])
        row["NumPy_fresh_E_parity"]["pass"] = True
        with self.assertRaisesRegex(ValueError,"PARITY_FLAGS"): audit.check_cell(row,arrays,independent,raw,cached,state)

    def test_unobserved_labels_have_no_effect_and_empty_complete_mask_is_explicit(self):
        v=fixture(); expected=audit.check_cell(*v); v[3]["target"][~v[3]["observed"]]=np.inf; v[1]["target"] = v[3]["target"].copy(); self.assertEqual(expected,audit.check_cell(*v))
        v[3]["observed"][:,:,3] = False; v[3]["target"][:,:,3]=np.nan; v[1]["observed"] = v[3]["observed"].copy(); v[1]["target"] = v[3]["target"].copy()
        row,independent = producer.evaluate_arrays({k:v[1][k] for k in ("prediction","latent_history","latent_future","lag_features")},v[3],{k:v[5][k] for k in ("E","D","center")},v[4]); row.update(artifact=v[0]["artifact"],independent_artifact=v[0]["independent_artifact"])
        result=audit.check_cell(row,v[1],independent,*v[3:]); self.assertEqual(result["score_fields_checked"],36); self.assertEqual(result["scores_complete_mask"],{"status":"NO_COMPLETE_C"})

    def test_CP0_probe_binds_saved_CP0_and_flat_future_without_current_C_persistence_claim(self):
        values=probe_fixture(); result=audit.check_probe(*values); self.assertFalse(result["latent_persistence_is_current_C_persistence_claimed"])
        self.assertFalse(np.array_equal(values[2]["prediction"],np.broadcast_to(values[4]["raw_last"][:,None],values[2]["prediction"].shape)))
        for kind in ("flat","C","cache","origins","scope","tol"):
            v=copy.deepcopy(values)
            if kind == "flat": v[1]["flat_future"][0,0,0] += .1
            elif kind == "C": v[1]["NumPy_flat_prediction"][0,0,0] += .1
            elif kind == "cache": v[1]["cached_Torch_latent_history"][0,0,0] += .1
            elif kind == "origins": v[1]["origins"][0] += 1
            elif kind == "scope": v[0]["latent_persistence_is_current_C_persistence_claimed"] = True
            else: v[0]["checks"]["NumPy_lag_prediction"]["rtol"] = .1
            with self.subTest(kind=kind),self.assertRaises(ValueError): audit.check_probe(*v)

    def test_exact_tie_selects_earliest_CP_and_selected_TUNE_is_reused(self):
        row,states=selection_fixture(); self.assertEqual(audit.check_selection(row,states)["update"],0)
        row["checkpoints"][-1]["tune"]["mse"] -= 1e-16
        with self.assertRaisesRegex(ValueError,"EXACT_EARLIER"): audit.check_selection(row,states)
        for kind in ("CP0","statehash","headhash","TUNE","headchange","zero"):
            r,s=copy.deepcopy(selection_fixture())
            if kind == "CP0": r["CP0_TUNE"] = r["checkpoints"][1]
            elif kind == "statehash": r["selected_state_sha256"] = "0"*64
            elif kind == "headhash": r["selected_head_state_sha256"] = "0"*64
            elif kind == "TUNE": r["TUNE"]["update"] = 600
            elif kind == "headchange": r["training_head_values_changed"] = False
            else: s[0]["head.bias"][0,0] = .1
            with self.subTest(kind=kind),self.assertRaises(ValueError): audit.check_selection(r,s)

    def test_sixteen_complete_locks_bind_all_pre_DEV_metadata(self):
        result,registry,locks=registry_fixture(); self.assertEqual(len(audit.check_registry(result,registry,lambda item:locks[item["path"]])),16)
        for kind in ("duplicate","early","CPchanged","late","failed"):
            r,g,l=copy.deepcopy(registry_fixture())
            if kind == "duplicate": g["conditions"][-1] = g["conditions"][0]
            elif kind == "early": g["DEV_opened_before_seal"] = True
            elif kind == "CPchanged": next(iter(l.values()))["selected_update"] = 600
            elif kind == "late": next(iter(l.values()))["sealed_unix"] = 21.
            else: g["conditions"][0]["status"] = "FAILED"
            with self.subTest(kind=kind),self.assertRaises(ValueError): audit.check_registry(r,g,lambda item:l[item["path"]])

    def test_logs_reuse_full_FIT_counts_head_flags_and_same_parent_seed_order(self):
        with tempfile.TemporaryDirectory() as temp:
            row,fit,digest,logs,path=logs_fixture(Path(temp)); self.assertEqual(audit.check_logs(row,fit,digest),600)
            for kind in ("flag","loss","clip","order","scope","count"):
                r,l=copy.deepcopy((row,logs))
                if kind == "flag": l[0]["D_grad_is_None"] = False
                elif kind == "loss": l[0]["loss"] = float("nan")
                elif kind == "clip": l[0]["clipped"] = True
                elif kind == "order": r["order_sha256"] = "0"*64
                elif kind == "scope": r["independent_per_update_frozen_bytes_replayed"] = True
                else: r["FIT_channel_observed_counts"][0] += 1
                path.write_text("\n".join(json.dumps(v) for v in l),encoding="utf-8"); r["updates_log"] = audit.base.file_record(path)
                with self.subTest(kind=kind),self.assertRaises(ValueError): audit.check_logs(r,fit,digest)

    def test_original_TUNE_reduction_has_exact_mask_counts_and_scalar_for_selection(self):
        from native_head_v1 import score_predictions
        _,arrays,_,raw,_,_=fixture()
        self.assertEqual(audit.selection_score(arrays["prediction"],raw["target"],raw["observed"]),score_predictions(arrays["prediction"],raw["target"],raw["observed"]))

    def test_closed_actual_pin_precedes_IO_and_import_has_no_scientific_execution(self):
        with patch.object(audit,"RESULT_SHA",None),patch.object(audit.base,"file_record",side_effect=AssertionError("IO forbidden")):
            with self.assertRaisesRegex(ValueError,"NOT_YET_REGISTERED"): audit.metadata("missing",lambda:None)
        with patch.object(audit.base,"file_record",return_value={"path":"synthetic","sha256":"0"*64}),patch.object(audit.prior,"verified_json",side_effect=AssertionError("actual IO forbidden")):
            with self.assertRaisesRegex(ValueError,"EXACT_NEW_DIRECTION"): audit.metadata("synthetic",lambda:None)
        tree=ast.parse(Path(audit.__file__).read_text(encoding="utf-8")); imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]+[a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        self.assertFalse(any(name in ("run_new_direction_native_v1","lagged_native_head_v1","torch") for name in imports))
        command="import sys;sys.path.insert(0,"+repr(str(audit.HERE))+");import audit_new_direction_native_v1;assert 'numpy' not in sys.modules and 'torch' not in sys.modules;print('PURE_IMPORT_PASS')"
        completed=subprocess.run([sys.executable,"-c",command],capture_output=True,text=True,check=True); self.assertIn("PURE_IMPORT_PASS",completed.stdout)


if __name__ == "__main__": unittest.main()
