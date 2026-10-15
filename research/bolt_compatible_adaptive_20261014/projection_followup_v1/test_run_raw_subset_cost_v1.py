"""Synthetic JSON gates and fixed-state callback arithmetic; no actual science assets."""
from pathlib import Path
import ast
import builtins
import copy
import inspect
import json
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

import run_raw_subset_cost_v1 as run
import whole_raw_subset_cost_v1 as wc


def write(path, value):
    path.write_text(json.dumps(value,allow_nan=False),encoding="utf-8"); return run.record(path)


def gate_fixture(folder):
    source=write(folder/"PARENT.json",{"status":"COMPLETE","protected_TEST_accessed":False})
    parent_audit=write(folder/"PARENT_PASS.json",{"status":"PASS","protected_TEST_accessed":False})
    auditor=folder/run.SCIENCE_AUDITOR; auditor.write_text("# synthetic never imported\n",encoding="utf-8")
    state=write(folder/"STATE_METADATA_ONLY.json",{"synthetic":True})
    conditions=[]; datasets={}
    for dataset,(_,k,_,_) in run.DATASETS.items():
        subset=write(folder/(dataset+"_SUBSET_LOCK.json"),{"status":"LOCKED_FIT_PAST_ONLY","selected_indices":list(range(k))})
        bridge={"canonical_record":{"path":"not_opened_physical.npz","sha256":"synthetic","arrays":{"opaque":True}},
            "same_physical_file_path_and_SHA":True,"complete_array_manifest_preserved":True,
            "loader_file_SHA_member_shape_dtype_SHA_checks_relaxed":False,"numeric_tolerance_changed":False}
        row={"subset_selection_lock":subset,"canonical_physical_bridge_binding":bridge,"physical_bridge_canonical":bridge["canonical_record"],"spaces":{}}
        for space in run.SPACES:
            entry={"status":"COMPLETE","fixed_E_hashes_before":{"unchanged":True},"fixed_E_hashes_after":{"unchanged":True},"native":{}}
            for seed in run.SEEDS:
                key=(dataset,space,str(seed))
                lock=write(folder/(dataset+space+str(seed)+"_LOCK.json"),{"dataset":dataset,"space":space,"seed":str(seed),"status":"LOCKED",
                    "DEV_used_for_CP_selection":False,"selected_update":100,"selected_state_artifact":state})
                conditions.append({"dataset":dataset,"space":space,"seed":str(seed),"lock":lock})
                entry["native"][str(seed)]={"status":"COMPLETE","metadata":{"updates":600,"selected_update":100,
                    "canonical_prediction_batch_size":32,"fixed_global_FIT_decoder":True},"checkpoints":[{"update":100,"state_artifact":state}]}
            row["spaces"][space]=entry
        datasets[dataset]=row
    registry=write(folder/"REGISTRY.json",{"status":"ALL_TWELVE_RESOLVED_BEFORE_DEV","conditions":conditions,"failed_paths_at_seal":[]})
    science={**run.SCIENCE_EXPECTED,"status":"COMPLETE","protected_TEST_accessed":False,"independent_holdout":False,
        "new_backbone_training":False,"existing_head_weights_reused":False,"future_labels_used_for_subset_or_decoder_fit":False,
        "per_origin_decoder_fit":False,"whole_deployment_cost_measured":False,"oracle_used_for_selection":False,"strict_K_only_compression":False,
        "failed_paths":[],"all_twelve_locks_before_any_DEV_input_or_prediction":True,"F_before":{"same":True},"F_after":{"same":True},
        "source_producer":source,"source_audit":parent_audit,"selection_registry":registry,"datasets":datasets}
    science["prior_partial"]=write(folder/"PRIOR_PARTIAL.json",{"status":"PARTIAL_COMPLETED","subset_selection_fits":0,"actual_subset_F_split_inferences":0,
        "native_training_fits":8,"training_updates":4800,"checkpoint_states_saved":40})
    science["Gram_failure_diagnostic"]=write(folder/"GRAM_FAILURE.json",{"synthetic":True})
    addendum=folder/"REPAIR.md"; addendum.write_text("Direct residual numeric repair.\n",encoding="utf-8"); science["numeric_repair_addendum"]=run.record(addendum)
    science["mixed_reexecution_identity"]={"status":"PASS","mismatches":[],"mixed_native_conditions_compared":8,"checkpoint_states_and_TUNE_arrays_compared":40,
        "selected_native_split_arrays_compared":24,"mixed_Bolt_split_arrays_compared":12,"global_D_states_compared":8,"learning_curves_compared":8,
        "total_physical_native_fit_attempts_across_v1_v2":20,"distinct_registered_native_conditions":12,
        "tolerance_used":False,"new_heads_warm_started_or_reused":False,"two_runs_independent_conditions_claimed":False}
    science_path,audit_path=folder/"SCIENCE.json",folder/"SCIENCE_AUDIT.json"
    binding=write(science_path,science)
    audit={"status":"PASS","protected_TEST_accessed":False,"diagnostic":binding,"totals":{"synthetic_full_coverage":12},
        "audit_source":run.record(auditor),"source_producer":source,"source_audit":parent_audit}
    write(audit_path,audit)
    return science_path,audit_path


class RawSubsetCostControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.folder=Path(self.temp.name)

    def gates(self, paths):
        return patch.multiple(run,ROOT=self.folder,SCIENCE_RESULT_SHA=run.record(paths[0])["sha256"],
            SCIENCE_AUDIT_SHA=run.record(paths[1])["sha256"],SCIENCE_AUDIT_COUNTS={"synthetic_full_coverage":12})

    def test_unregistered_actual_PASS_stops_before_any_scientific_import(self):
        imported=[]; original=builtins.__import__
        def track(name,*args,**kwargs):
            if name in ("torch","numpy","continuation_runtime_v1","run_condition_probes"): imported.append(name)
            return original(name,*args,**kwargs)
        with patch.multiple(run,SCIENCE_RESULT_SHA=None,SCIENCE_AUDIT_SHA=None,SCIENCE_AUDIT_COUNTS={}),patch("builtins.__import__",side_effect=track):
            with self.assertRaisesRegex(ValueError,"NOT_YET_REGISTERED"): run.execute("never_run",self.folder/"no_source.json",self.folder/"no_pass.json")
        self.assertEqual(imported,[])

    def test_full_synthetic_JSON_gate_resolves_twelve_locks_without_array_load(self):
        paths=gate_fixture(self.folder)
        with self.gates(paths):
            _,_,bindings,locks,subset=run.load_gates(*paths)
        self.assertEqual(len(locks),12); self.assertEqual(len(subset),2); self.assertEqual(set(bindings),{
            "raw_subset_diagnostic","raw_subset_audit","source_producer","source_audit","prior_partial","Gram_failure_diagnostic","numeric_repair_addendum"})
        self.assertFalse((self.folder/"not_opened_physical.npz").exists())

    def test_partial_science_or_audit_missing_coverage_rejected(self):
        paths=gate_fixture(self.folder); value=run.read(paths[0]); value["status"]="PARTIAL_COMPLETED"; write(paths[0],value)
        with self.gates(paths),self.assertRaisesRegex(ValueError,"COMPLETE_SCIENCE"): run.load_gates(*paths)
        paths=gate_fixture(self.folder); value=run.read(paths[1]); value["totals"]["synthetic_full_coverage"]=11; write(paths[1],value)
        with self.gates(paths),self.assertRaisesRegex(ValueError,"AUDIT_COVERAGE"): run.load_gates(*paths)

    def test_selected_state_mutation_or_changed_CP_lock_rejected(self):
        paths=gate_fixture(self.folder)
        with self.gates(paths):
            (self.folder/"STATE_METADATA_ONLY.json").write_text("{}",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"EXACT_LOCAL_SOURCE_SHA"): run.load_gates(*paths)

    def test_arrayless_physical_alias_cannot_replace_manifest(self):
        paths=gate_fixture(self.folder); value=run.read(paths[0]); value["datasets"]["robin"]["physical_bridge_canonical"].pop("arrays")
        write(paths[0],value); audit=run.read(paths[1]); audit["diagnostic"]=run.record(paths[0]); write(paths[1],audit)
        with self.gates(paths),self.assertRaisesRegex(ValueError,"CANONICAL_PHYSICAL_ARRAY_MANIFEST"): run.load_gates(*paths)

    def test_protected_TEST_scope_and_registry_missing_seed_rejected(self):
        paths=gate_fixture(self.folder); value=run.read(paths[0]); value["protected_TEST_accessed"]=True; write(paths[0],value)
        audit=run.read(paths[1]); audit["diagnostic"]=run.record(paths[0]); write(paths[1],audit)
        with self.gates(paths),self.assertRaisesRegex(ValueError,"SCIENCE_SCOPE"): run.load_gates(*paths)

    def test_marking_one_head_unavailable_keeps_bolt_and_other_seed(self):
        routes=("RAW_BOLT","PCA_BOLT","PCA_NATIVE_s92601","PCA_NATIVE_s92602")
        prepared={"predictors":dict.fromkeys(routes,object()),"references":dict.fromkeys(routes,object()),"metadata":dict.fromkeys(routes,{}),"unavailable":{}}
        run.mark_unavailable(prepared,["PCA_NATIVE_s92601"],{"status":"FAILED_PATH_PRESERVED"})
        self.assertEqual(set(prepared["predictors"]),set(routes)-{"PCA_NATIVE_s92601"})
        self.assertEqual(set(prepared["metadata"]),set(routes)); self.assertEqual(set(prepared["unavailable"]),{"PCA_NATIVE_s92601"})

    def test_bolt_callback_really_uses_K_streams_global_D_and_current_anchor(self):
        fake=types.ModuleType("run_raw_subset_comparison_v2")
        def encode(state,indices,h): return np.ascontiguousarray(h[...,indices])
        fake.encode_numpy=encode
        history=np.ones((2,512,3),np.float32); history[...,1]=2; history[...,2]=3
        D=np.array([[1.,2.,3.],[4.,5.,6.]],np.float32); calls=[]
        def F(z): calls.append(z.copy()); return np.repeat(z[:,-1:]+np.float32(1),48,axis=1)
        with patch.dict(sys.modules,{fake.__name__:fake}):
            callback=run.bolt_predictor(None,[0,2],D,F); result=callback(history,np.ones_like(history,dtype=bool))
        self.assertEqual(calls[0].shape,(2,512,2))
        np.testing.assert_array_equal(result,np.repeat(history[:,-1:]+D.sum(axis=0),48,axis=1))
        np.testing.assert_array_equal(history,np.broadcast_to(np.array([1,2,3],np.float32),history.shape))

    def test_fixed_D_pipeline_does_not_import_training_or_shrinkage_fit(self):
        text=Path(run.__file__).read_text(encoding="utf-8")
        self.assertNotIn("train_native_fixed(",text); self.assertNotIn("fit_global_observed_decoder(",text); self.assertNotIn("fit_shrunk_local_decoder",text)
        tree=ast.parse(text)
        top=[node for node in tree.body if isinstance(node,(ast.Import,ast.ImportFrom))]
        names=[node.module if isinstance(node,ast.ImportFrom) else alias.name for node in top for alias in (node.names if isinstance(node,ast.Import) else [types.SimpleNamespace(name="")])]
        self.assertFalse({"torch","numpy","chronos"}&set(names))

    def test_prior_mixed_reexecution_not_counted_as_new_independent_conditions(self):
        paths=gate_fixture(self.folder); value=run.read(paths[0]); value["mixed_reexecution_identity"]["two_runs_independent_conditions_claimed"]=True
        write(paths[0],value); audit=run.read(paths[1]); audit["diagnostic"]=run.record(paths[0]); write(paths[1],audit)
        with self.gates(paths),self.assertRaisesRegex(ValueError,"REEXECUTION_IDENTITY"): run.load_gates(*paths)

    def test_physical_native_precheck_one_failed_seed_keeps_other_seed_checks(self):
        future=types.ModuleType("run_raw_subset_comparison_v2")
        future.encode_numpy=lambda state,indices,h:np.ascontiguousarray(h[...,indices])
        future.encode_native=lambda projection,indices,h:np.ascontiguousarray(h[...,indices])
        native=types.ModuleType("raw_subset_native_head_v1")
        def predict(model,z,current,batch_size):
            if model=="bad": raise ValueError("one synthetic selected head failed")
            return np.repeat(current[:,None],48,axis=1),np.repeat(z[:,-1:],48,axis=1)
        native.predict_components=predict
        h=np.ones((8,512,3),np.float32); mask=np.ones_like(h,dtype=bool); z=np.ascontiguousarray(h[...,[0,2]])
        raw={"history":h,"history_observed":mask,"raw_last":h[:,-1],"origins":np.arange(8,dtype=np.int64)}
        fixed={"latent_history_NATIVE_TORCH32":z,"origins":raw["origins"]}
        expected=np.repeat(raw["raw_last"][:,None],48,axis=1)
        with patch.dict(sys.modules,{future.__name__:future,native.__name__:native}):
            result,arrays=run.physical_backend_precheck(h,mask,np.zeros(3,np.float32),np.ones(3,np.float32),raw,fixed,None,None,[0,2],
                np.ones((2,3),np.float32),"NATIVE_TORCH32",{"92601":expected,"92602":expected},models={"92601":"bad","92602":"good"})
        self.assertTrue(result["input_backend_pass"]); self.assertFalse(result["prediction_checks"]["92601"]["pass"])
        self.assertTrue(result["prediction_checks"]["92602"]["pass"]); self.assertIn("native_s92602_prediction",arrays)
        self.assertNotIn("native_s92601_prediction",arrays)

    def test_actual_small_CPU_native_B1_B8_match_canonical32_reference_with_fixed_D(self):
        import torch
        from raw_subset_native_head_v1 import NativeFixedDecoder,predict_components
        torch.set_num_threads(2)
        rng=np.random.default_rng(92601); h=rng.normal(size=(8,512,3)).astype(np.float32)
        d=np.array([[1.,.2,0.],[-.1,.3,1.]],np.float32)
        model=NativeFixedDecoder(d).cpu().eval().requires_grad_(False)
        with torch.no_grad(): model.head.weight.fill_(.001); model.head.bias.fill_(.02)
        z=np.ascontiguousarray(h[...,[0,2]]); expected,_=predict_components(model,z,h[:,-1],batch_size=32)
        with patch.object(sys.modules.get("run_raw_subset_comparison_v2") or __import__("run_raw_subset_comparison_v2"),"encode_native",
                side_effect=lambda projection,indices,history:np.ascontiguousarray(history[...,indices])):
            callback=run.native_predictor(None,[0,2],model)
            for batch in (1,8):
                actual=callback(h[:batch],np.ones_like(h[:batch],dtype=bool))
                self.assertTrue(wc.parity(actual,expected[:batch])["pass"])
        self.assertEqual(model.decoder.device.type,"cpu")
        np.testing.assert_array_equal(model.decoder.detach().numpy(),d)

    def test_full_FrozenProjection_hash_includes_unused_decoder_and_bias(self):
        import torch
        from projection_math_v1 import ProjectionState
        from native_head_v1 import FrozenProjection,clone_state,state_hash
        center=np.zeros(3); encoder=np.array([[1.,0.],[0.,1.],[0.,0.]])
        original=FrozenProjection(ProjectionState(center,encoder,np.ones((2,3)),np.ones(3),{})).cpu()
        wrong=FrozenProjection(ProjectionState(center,encoder,np.zeros((2,3)),np.zeros(3),{})).cpu()
        h=torch.ones((2,512,3),dtype=torch.float32)
        torch.testing.assert_close(original.encode(h),wrong.encode(h),rtol=0,atol=0)
        self.assertNotEqual(state_hash(clone_state(original)),state_hash(clone_state(wrong)))


if __name__=="__main__": unittest.main()
