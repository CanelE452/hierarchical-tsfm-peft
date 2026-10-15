"""SOURCE, generated metadata and CPU tensors only; no actual assets or timing."""
from pathlib import Path
import ast
import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_fixed_l7_latent_calibration_cost_v1 as run
from calibrated_l7_forecaster_v1 import CalibratedL7Forecaster
from linear_e_adaptation_v1 import LinearLagAdapter


class CalibrationCostControllerTests(unittest.TestCase):
    def test_closed_gate_precedes_any_actual_metadata_or_asset_read(self):
        with patch.object(run,"Q_AUDIT_SHA",None),patch.object(run,"record",side_effect=AssertionError("closed IO")),self.assertRaisesRegex(ValueError,"NOT_YET_REGISTERED"): run.load_gates("missing","missing")

    def test_ten_routes_and_three_existing_physical_helpers_are_reused(self):
        import run_linear_e_cost_v1 as old
        self.assertEqual(len(run.ROUTES),10); self.assertEqual(sum("_SCALE_" in r for r in run.ROUTES),6)
        self.assertIs(run.canonical_tune_bridge,old.canonical_tune_bridge); self.assertIs(run.adapter_predictor,old.adapter_predictor); self.assertIs(run.physical_precheck,old.physical_precheck)

    def test_explicit_q_hashes_detect_both_coefficients_and_python_config(self):
        model=CalibratedL7Forecaster(np.zeros(2,np.float32),np.eye(2,1,dtype=np.float32),np.ones((7,2),np.float32),np.ones(1,np.float32),np.zeros(1,np.float32),family="SCALE_ONLY")
        models={"toy_SCALE_ONLY":model}; before=run.calibration_hashes(models)
        with torch.no_grad(): model.a[0]=2
        changed=run.calibration_hashes(models); self.assertNotEqual(before,changed)
        model.identity=True; after=run.calibration_hashes(models); self.assertNotEqual(changed,after); self.assertEqual(changed["toy_SCALE_ONLY"]["coefficients_state_sha256"],after["toy_SCALE_ONLY"]["coefficients_state_sha256"])

    def test_composed_reference_preserves_saved_C_and_checked_past_components(self):
        rng=np.random.default_rng(12); p=rng.normal(size=(3,512,2)).astype(np.float32); f=rng.normal(size=(3,48,2)).astype(np.float32); prediction=rng.normal(size=(3,48,3)).astype(np.float32)
        baseline={"latent_history":p,"latent_future":f,"prediction":np.zeros_like(prediction),"origins":np.arange(3,dtype=np.int64)}
        selected={"prediction_FP32":prediction,"latent_future_calibrated_FP32":f+.25,"origins":baseline["origins"].copy()}
        reference=run.compose_reference(baseline,selected)
        self.assertIs(reference["prediction"],prediction); self.assertIs(reference["latent_history"],p); self.assertIs(reference["latent_future"],selected["latent_future_calibrated_FP32"])
        self.assertEqual(reference["lag_features"].shape,(3,48,14)); selected["origins"][0]+=1
        with self.assertRaises(ValueError): run.compose_reference(baseline,selected)

    def test_actual_callback_and_composed_reference_pass_physical_precheck_on_generated_arrays(self):
        rng=np.random.default_rng(66); x=rng.normal(size=(8,512,3)).astype(np.float32); mask=np.ones_like(x,bool); origins=np.arange(8,dtype=np.int64)
        raw={"history":x,"history_observed":mask,"raw_last":x[:,-1].copy(),"target":np.zeros((8,48,3),np.float32),"observed":np.ones((8,48,3),bool),"mean":np.zeros(3,np.float32),"std":np.ones(3,np.float32),"origins":origins}; physical={"history_physical":x,"history_observed":mask,"origins":origins}
        e=rng.normal(scale=.1,size=(3,2)).astype(np.float32); d=rng.normal(scale=.1,size=(14,3)).astype(np.float32); center=np.zeros(3,np.float32)
        base=LinearLagAdapter(center,e,d,run.LAGS,learn_E=False).eval().requires_grad_(False); calibrated=CalibratedL7Forecaster(center,e,d,np.array([.2,-.3],np.float32),np.array([.1,.15],np.float32),family="SCALE_BIAS")
        calls=[]
        def toyF(z): calls.append(len(z)); return z[:,-1:].expand(-1,48,-1)+.5
        with torch.no_grad():
            p,z,u=base(torch.as_tensor(x),toyF,return_components=True); qpred,_,q=calibrated(torch.as_tensor(x),toyF,return_components=True)
        baseline={**raw,"prediction":p.numpy().copy(),"latent_history":z.numpy().copy(),"latent_future":u.numpy().copy(),"lag_features":base.lag_features(z,u).numpy().copy()}
        reference=run.compose_reference(baseline,{"prediction_FP32":qpred.numpy().copy(),"latent_future_calibrated_FP32":q.numpy().copy(),"origins":origins})
        entry,_=run.physical_precheck(physical,raw,reference,calibrated,toyF,lambda value:torch.as_tensor(value))
        self.assertTrue(entry["input_backend_pass"] and entry["prediction_backend_pass"]); self.assertEqual(calls,[8,8,8])
        callback=run.adapter_predictor(calibrated,toyF,lambda value:torch.as_tensor(value)); self.assertEqual(callback(x[:1],mask[:1]).shape,(1,48,3)); self.assertEqual(calls[-1],1)

    def test_six_selected_family_locks_bind_before_model_import(self):
        def ref(name,sha="e"*64): return {"path":name,"sha256":sha}
        calibration={"status":"COMPLETE","protected_TEST_accessed":False,"failed_paths":[],"failed_candidate_paths":[],"controller_source":ref("controller","c"*64),"module_source":ref("module","d"*64),"fit_conditions_completed":6,"selected_conditions_completed":6,"resolved_selection_locks":6,"selected_DEV_evaluations_completed":6,"ridge_candidates_generated":42,"identity_comparisons_generated":6,"candidate_FIT_TUNE_evaluations_completed":96,"all_6_locks_before_DEV":True,"cached_Bolt_contexts_reused":9,"cached_baseline_redecode_checks":9,"new_F_forward_calls":0,"new_F_model_objects_loaded":0,"new_optimizer_updates":0,"new_E_D_training_updates":0,"native_head_weights_loaded":False,"new_backbone_training":False,"F_parent_signatures_reused":{"before":{"toy":1},"after":{"toy":1}},"states":{s:{"families":{}} for s in run.STATES}}
        keys=("matched_l7_native_diagnostic","matched_l7_native_audit","matched_l7_native_cost","matched_l7_native_cost_audit","native_selection_registry","native_bundle_metadata_reader","parent_report_auditor_source","parent_report","parent_report_manifest","parent_report_audit","E_only_diagnostic","E_only_audit","joint_recon_diagnostic","joint_recon_audit","causal_lag_diagnostic","causal_lag_audit","selection_registry","plan")
        calibration.update({key:ref(key) for key in keys}); registry={"status":"ALL_SIX_RESOLVED_BEFORE_DEV","DEV_opened_before_seal":False,"conditions":[]}; reads={"q":calibration,"selection_registry":registry,"subset":{"source_producer":ref("original")},"original":{"F_before":{"toy":1},"F_after":{"toy":1}}}
        accepted={"status":"PASS","protected_TEST_accessed":False,"totals":{"fixture":6},"diagnostic":ref("q","a"*64),"audit_source":ref("auditor")}; reads["audit"]=accepted
        for state in run.STATES:
            for family in run.FAMILIES:
                name=state+"_"+family; row={"status":"COMPLETE","selected_candidate":"IDENTITY","selected_kind":"IDENTITY","selected_lambda":None,"selected_TUNE_MSE":.5,"selected_coefficient_value_sha256":"f"*64,"selected_coefficients":ref(name+"_coefs"),"selection_lock":ref(name+"_lock")}; calibration["states"][state]["families"][family]=row; reads[name+"_lock"]={**row,"DEV_opened_before_lock":False}; registry["conditions"].append({"state":state,"family":family,"status":"COMPLETE","selection_lock":row["selection_lock"]})
        native={"F_parent_signatures_reused":calibration["F_parent_signatures_reused"]}; science={"raw_subset_diagnostic":ref("subset")}
        import run_fixed_l7_latent_calibration_v1 as parent
        with patch.multiple(run,Q_RESULT_SHA="a"*64,Q_AUDIT_SHA="b"*64,Q_CONTROLLER_SHA="c"*64,Q_MODULE_SHA="d"*64,Q_AUDITOR_SOURCE=ref("auditor"),AUDIT_COUNTS={"fixture":6}),patch.object(run,"record",side_effect=lambda name:ref(name,"a"*64 if name=="q" else "b"*64)),patch.object(run,"read",side_effect=lambda name:reads[name]),patch.object(run,"bound",side_effect=lambda item,*other:item["path"]),patch.object(parent,"load_gates",return_value=(native,science,{})):
            values=run.load_gates("q","audit"); self.assertEqual(set(values[-1]),{s+"_"+f for s in run.STATES for f in run.FAMILIES})

    def test_controller_has_no_fit_or_optimizer_and_forward_calls_actual_F(self):
        source=Path(run.__file__).read_text(encoding="utf-8"); tree=ast.parse(source); calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertFalse(any(name.endswith(("fit_calibration","backward","step","train_head")) for name in calls))
        wrapper=Path(run.__file__).with_name("calibrated_l7_forecaster_v1.py").read_text(encoding="utf-8"); self.assertIn("original_future=bolt(zpast)",wrapper); self.assertNotIn("predict_calibrated(",wrapper)


if __name__=="__main__": unittest.main()
