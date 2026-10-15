"""Generated JSON/CPU tensors only; no research arrays, F, GPU or real cost replay."""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import ast
import copy
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import run_new_direction_cost_v1 as runner
import whole_new_direction_cost_v1 as whole


def gate_fixture(folder):
    folder=Path(folder)
    def save(name,value):
        path=folder/name;path.write_text(json.dumps(value),encoding="utf-8");return runner.record(path)
    source=folder/"fixture_source.py";source.write_text("# Synthetic source only\n",encoding="utf-8");source=runner.record(source)
    f={"fixture_model":"not a real model"};original=save("original.json",{"F_before":f,"F_after":f});subset=save("subset.json",{"source_producer":original});science={"raw_subset_diagnostic":subset}
    cross={"F_before":f,"F_after":f}
    for key in ("latent_calibration_diagnostic","latent_calibration_audit","latent_calibration_cost","latent_calibration_cost_audit","parent_report","parent_report_manifest","parent_report_audit"):cross[key]=save(key+".json",{})
    cross_binding=save("cross.json",cross);cross_audit=save("cross_audit.json",{});state=save("synthetic_state.json",{"fixture":"not NPZ or model state"})
    native={"status":"COMPLETE","protected_TEST_accessed":False,"failed_paths":[],"controller_source":source,"native_training_fits":16,"resolved_selection_locks":16,"training_updates_completed":9600,"checkpoint_states_saved":80,"DEV_evaluations_completed":16,"all_16_locks_before_DEV":True,"cached_actual_F_contexts_reused":24,"new_F_forward_calls":0,"new_F_model_objects_loaded":0,"new_backbone_training":False,"whole_deployment_cost_measured":False,"predictive_cross_diagnostic":cross_binding,"predictive_cross_audit":cross_audit,"F_parent_signatures_reused":{"before":f,"after":f},"datasets":{d:{"conditions":{}} for d in runner.DATASETS}}
    ridge={"status":"COMPLETE","protected_TEST_accessed":False,"failed_paths":[],"controller_source":source,"module_source":source,"ridge_fits_completed":12,"resolved_selection_locks":12,"alpha_candidates_completed":84,"FIT_TUNE_prediction_evaluations_completed":168,"selected_DEV_evaluations_completed":12,"all_12_locks_before_DEV":True,"new_F_forward_calls":0,"new_F_model_objects_loaded":0,"new_backbone_training":False,"whole_deployment_cost_measured":False,"cross_moment_diagnostic":cross_binding,"cross_moment_audit":cross_audit,"F_parent_signatures_reused":{"before":f,"after":f},"datasets":{d:{"routes":{}} for d in runner.DATASETS}}
    native_locks=[];ridge_locks=[]
    for data in runner.DATASETS:
        for name in whole.state_names(data):
            native["datasets"][data]["conditions"][name]={"native":{}}
            for seed in (92601,92602):
                lock=save(data+name+str(seed)+".json",{"status":"COMPLETE","DEV_opened_before_lock":False,"selected_update":100,"selected_state":state})
                native["datasets"][data]["conditions"][name]["native"][str(seed)]={"selection_lock":lock,"selected_update":100,"selected_state":state}
                native_locks.append({"dataset":data,"condition":name,"seed":seed,"status":"COMPLETE","selection_lock":lock})
        for route in ("RAW_CURRENT_C","RAW_HISTORY512",*(name+"_HISTORY512" for name in whole.state_names(data))):
            lock=save(data+route+".json",{"status":"COMPLETE","DEV_opened_before_lock":False,"selected_alpha":.1,"selected_coefficients":state})
            ridge["datasets"][data]["routes"][route]={"selection_lock":lock,"selected_alpha":.1,"selected_coefficients":state};ridge_locks.append({"dataset":data,"route":route,"status":"COMPLETE","selection_lock":lock})
    native["selection_registry"]=save("native_registry.json",{"status":"ALL_SIXTEEN_RESOLVED_BEFORE_DEV","DEV_opened_before_seal":False,"conditions":native_locks})
    ridge["selection_registry"]=save("ridge_registry.json",{"status":"ALL_TWELVE_RESOLVED_BEFORE_DEV","DEV_opened_before_seal":False,"conditions":ridge_locks})
    cross["state_registry"]=save("cross_registry.json",{})
    n=save("native.json",native);r=save("ridge.json",ridge)
    na=save("native_audit.json",{"status":"PASS","protected_TEST_accessed":False,"diagnostic":n,"audit_source":source,"selection_registry":native["selection_registry"],"totals":{"native_fits_checked":16},"partial_paths":[]})
    ra=save("ridge_audit.json",{"status":"PASS","protected_TEST_accessed":False,"diagnostic":r,"audit_source":source,"selection_registry":ridge["selection_registry"],"totals":{"ridge_fits_checked":12},"partial_paths":[]})
    values={"NATIVE_RESULT_SHA":n["sha256"],"NATIVE_AUDIT_SHA":na["sha256"],"RIDGE_RESULT_SHA":r["sha256"],"RIDGE_AUDIT_SHA":ra["sha256"],"NATIVE_AUDITOR_SOURCE":source,"RIDGE_AUDITOR_SOURCE":source,"NATIVE_CONTROLLER_SHA":source["sha256"],"RIDGE_CONTROLLER_SHA":source["sha256"],"RIDGE_MODULE_SHA":source["sha256"],"NATIVE_AUDIT_COUNTS":{"native_fits_checked":16},"RIDGE_AUDIT_COUNTS":{"ridge_fits_checked":12}}
    return [v["path"] for v in (n,na,r,ra)],values,cross,science,source


def generated_past(encoded=False):
    rng=np.random.default_rng(82);physical=rng.normal(size=(8,512,3)).astype(np.float32);mask=np.ones_like(physical,bool);mask[:,:3,0]=False;physical[~mask]=np.nan
    mean=np.array([2.,-1.,.4],np.float32);std=np.array([3.,.5,1.5],np.float32);history=whole.preprocess(physical,mask,mean,std);origins=np.arange(8,dtype=np.int64)+512
    raw={"history":history,"raw_last":history[:,-1].copy(),"mean":mean,"std":std,"history_observed":mask,"origins":origins}
    values={"history_physical":physical,"history_observed":mask,"origins":origins}
    encoder=None
    if encoded:
        from linear_e_adaptation_v1 import LinearLagAdapter
        encoder=LinearLagAdapter(np.array([.1,.2,-.2],np.float32),np.array([[.3],[.5],[-.2]],np.float32),np.ones((7,3),np.float32),runner.LAGS,learn_E=False).eval().requires_grad_(False)
    return raw,values,encoder


def coefficients(p,c=3):
    value=np.zeros((p,48*c),np.float32);value[0,:c]=[.05,.1,-.02];value[-1,-c:]=[.03,-.01,.07]
    intercept=np.linspace(-.1,.1,48*c,dtype=np.float32)
    return {"coefficient_FP64":value.astype(np.float64),"intercept_FP64":intercept.astype(np.float64),"coefficient_FP32":value,"intercept_FP32":intercept}


class NewDirectionCostControllerTests(unittest.TestCase):
    def test_closed_gate_precedes_asset_reads_and_scientific_runtime(self):
        with patch.object(runner,"NATIVE_AUDIT_SHA",None),patch.object(runner,"record",side_effect=AssertionError("closed before IO")),self.assertRaisesRegex(ValueError,"NOT_YET_REGISTERED"):
            runner.load_gates("missing","missing","missing","missing")
        with patch.object(runner,"load_gates",side_effect=ValueError("closed")),patch.object(runner.old,"fresh_output",side_effect=AssertionError("must not create output")),self.assertRaisesRegex(ValueError,"closed"):
            runner.execute(SimpleNamespace(native_result="n",native_audit="na",ridge_result="r",ridge_audit="ra",tag="not-created"))

    def test_exact_generated_gate_and_all_twenty_eight_selection_bindings(self):
        import run_new_direction_native_v1 as ns
        import run_new_direction_direct_ridge_v1 as rs
        with tempfile.TemporaryDirectory(dir=runner.HERE,prefix="synthetic_cost_gate_") as folder:
            args,values,cross,science,_=gate_fixture(folder)
            with patch.multiple(runner,**values),patch.object(ns,"load_gates",return_value=(cross,{}, {},science,{},{})),patch.object(rs,"load_gates",return_value=(copy.deepcopy(cross),copy.deepcopy(science),{})):
                result=runner.load_gates(*args)
                self.assertEqual(len(result[-1]["native"]),16);self.assertEqual(len(result[-1]["ridge"]),12)
                self.assertEqual(result[2],cross);self.assertEqual(result[3],science)

    def test_result_and_bound_lock_mutation_are_rejected(self):
        import run_new_direction_native_v1 as ns
        import run_new_direction_direct_ridge_v1 as rs
        for mutation in ("result","lock"):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory(dir=runner.HERE,prefix="synthetic_cost_gate_") as folder:
                args,values,cross,science,_=gate_fixture(folder)
                path=Path(args[0]) if mutation=="result" else next(Path(folder).glob("robinPCA_K492601.json"));path.write_text("{}",encoding="utf-8")
                with patch.multiple(runner,**values),patch.object(ns,"load_gates",return_value=(cross,{},{},science,{},{})),patch.object(rs,"load_gates",return_value=(cross,science,{})),self.assertRaisesRegex(ValueError,"EXACT_NEW_NATIVE_RIDGE_RESULTS|EXACT_SOURCE_SHA"):
                    runner.load_gates(*args)

    def test_raw_current_level_and_history_innovation_match_fused_fp32(self):
        import run_new_direction_direct_ridge_v1 as source
        raw,_,_=generated_past()
        for route in ("RAW_CURRENT_C_RIDGE","RAW_HISTORY512_RIDGE"):
            with self.subTest(route=route):
                name=runner.ridge_route(route);feature,anchor=source.feature_payload(raw,name);state=coefficients(feature.shape[1]);callback=runner.ridge_predictor(route,state)
                pred=callback(raw["history"],raw["history_observed"]);expected=(feature@state["coefficient_FP32"]+state["intercept_FP32"]).reshape(8,48,3)+anchor[:,None]
                np.testing.assert_array_equal(pred,expected);self.assertEqual(pred.dtype,np.float32)
                self.assertEqual(bool(np.count_nonzero(anchor)),name!="RAW_CURRENT_C")

    def test_encoded_ridge_reencodes_each_callback_and_bypasses_D(self):
        raw,_,encoder=generated_past(True);state=coefficients(512);callback=runner.ridge_predictor("PCA_K4_RIDGE_HISTORY512",state,encoder)
        with patch.object(encoder,"encode",wraps=encoder.encode) as calls:
            one=callback(raw["history"][:1],raw["history_observed"][:1]);eight=callback(raw["history"],raw["history_observed"])
            self.assertEqual([call.args[0].shape[0] for call in calls.call_args_list],[1,8])
        with torch.no_grad():encoder.D.mul_(1e5)
        np.testing.assert_array_equal(eight,callback(raw["history"],raw["history_observed"]))
        np.testing.assert_allclose(one,eight[:1],atol=whole.NORM_ATOL,rtol=whole.NORM_RTOL)
        self.assertNotIn("target",runner.ridge_components.__code__.co_varnames)

    def test_ridge_first8_physical_boundary_and_independent_features(self):
        from run_predictive_cross_moment_v1 import encode_torch
        raw,physical,encoder=generated_past(True);state=coefficients(512);route="CROSS_MOMENT_K4_RIDGE_HISTORY512";reference={"TUNE_prediction_FP32":runner.ridge_predictor(route,state,encoder)(raw["history"],raw["history_observed"]),"TUNE_origins":raw["origins"]}
        check,arrays=runner.ridge_physical_precheck(physical,raw,reference,state,route,encoder,{"latent_history":encode_torch(encoder,raw["history"])})
        self.assertTrue(check["input_backend_pass"] and check["prediction_backend_pass"])
        self.assertTrue(all(v["pass"] for v in check["intermediate_checks"].values()))
        self.assertFalse(check["intermediate_parity_required_for_acceptance"]);self.assertEqual(arrays["features"].shape,(8,512));self.assertEqual(arrays["prediction"].shape,(8,48,3))
        bad=copy.deepcopy(reference);bad["TUNE_prediction_FP32"]+=1.
        check,_=runner.ridge_physical_precheck(physical,raw,bad,state,route,encoder,{"latent_history":encode_torch(encoder,raw["history"])})
        self.assertTrue(check["input_backend_pass"]);self.assertFalse(check["prediction_backend_pass"])

    def test_precheck_rejects_origin_and_filled_history_lineage_mutation(self):
        raw,physical,_=generated_past();state=coefficients(3);route="RAW_CURRENT_C_RIDGE";reference={"TUNE_prediction_FP32":runner.ridge_predictor(route,state)(raw["history"],raw["history_observed"]),"TUNE_origins":raw["origins"].copy()}
        reference["TUNE_origins"][0]+=1
        check,_=runner.ridge_physical_precheck(physical,raw,reference,state,route);self.assertFalse(check["input_backend_pass"])
        reference["TUNE_origins"]=raw["origins"].copy();changed=copy.deepcopy(raw);changed["history"][0,0,0]+=1
        check,_=runner.ridge_physical_precheck(physical,changed,reference,state,route);self.assertFalse(check["input_backend_pass"])

    def test_native_predictor_none_and_f_callback_are_distinct_actual_cpu_toys(self):
        from lagged_native_head_v1 import LaggedNativeForecaster
        from linear_e_adaptation_v1 import LinearLagAdapter
        from native_head_v1 import state_hash
        raw,_,_=generated_past();e=np.array([[.3],[.5],[-.2]],np.float32);d=np.full((7,3),.1,np.float32);center=np.zeros(3,np.float32)
        native=LaggedNativeForecaster(center,e,d,runner.LAGS).eval().requires_grad_(False);adapter=LinearLagAdapter(center,e,d,runner.LAGS,learn_E=False).eval().requires_grad_(False);calls=[]
        def actual_toy_f(z):calls.append(z.shape);return z[:,-1:].expand(-1,48,-1)+.2
        before=state_hash(native.state_dict());n=runner.adapter_predictor(native,None,torch.as_tensor)(raw["history"][:1],raw["history_observed"][:1]);f=runner.adapter_predictor(adapter,actual_toy_f,torch.as_tensor)(raw["history"][:1],raw["history_observed"][:1])
        self.assertEqual(len(calls),1);self.assertEqual(before,state_hash(native.state_dict()));self.assertFalse(np.array_equal(n,f));self.assertTrue(all(not p.requires_grad for p in native.parameters()))

    def test_route_mapping_and_helpers_preserve_registered_boundary(self):
        self.assertIs(runner.canonical_tune_bridge,runner.old.canonical_tune_bridge);self.assertIs(runner.adapter_predictor,runner.old.adapter_predictor);self.assertIs(runner.physical_precheck,runner.old.physical_precheck)
        for data in runner.DATASETS:
            routes=[r for r in whole.routes_for(data) if "RIDGE" in r];self.assertEqual(len(routes),6)
            self.assertEqual(len({runner.ridge_route(route) for route in routes}),6)
        with self.assertRaisesRegex(ValueError,"REGISTERED_RIDGE"):runner.ridge_route("PCA_K4_BOLT")

    def test_no_training_or_future_labels_inside_ridge_callback_and_gate_before_imports(self):
        tree=ast.parse(Path(runner.__file__).read_text(encoding="utf-8"));calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertFalse(any(value.endswith(("train_head","frozen_status","backward","fit_path","optimizer.step")) for value in calls))
        execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="execute")
        first=ast.unparse(execute.body[0]);self.assertIn("load_gates",first)
        scientific_imports=[n for n in execute.body if isinstance(n,(ast.Import,ast.ImportFrom))];self.assertTrue(scientific_imports)
        for import_node in scientific_imports:self.assertGreater(import_node.lineno,execute.body[0].lineno)
        callback=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="ridge_components")
        text=ast.unparse(callback);self.assertNotIn('"target"',text);self.assertNotIn('"observed"',text);self.assertNotIn("meter.check",text)


if __name__=="__main__":unittest.main()
