"""SOURCE-only controller checks with generated metadata and tiny CPU tensors."""
from pathlib import Path
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SOURCE=Path(__file__).with_name("run_matched_l7_native_v1.py")


def module():
    if not SOURCE.exists(): raise AssertionError("matched L7 native controller missing")
    spec=importlib.util.spec_from_file_location("matched_l7_fixture",SOURCE); value=importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


class ControllerTests(unittest.TestCase):
    def setUp(self): self.r=module()
    def test_closed_report_gate_precedes_any_file_or_scientific_read(self):
        with patch.object(self.r,"FIXED_SHA",{k:None for k in self.r.FIXED_SHA}),patch.object(self.r.old,"record",side_effect=AssertionError("closed IO")),self.assertRaisesRegex(ValueError,"NOT_YET_REGISTERED"):
            self.r.load_gates("a","b","c","d","e")
    def test_three_states_and_six_unique_head_orders(self):
        self.assertEqual(self.r.STATES,("RECON_CP0","E_ONLY_s92601","E_ONLY_s92602")); self.assertEqual(len(self.r.CONDITIONS),6)
        self.assertEqual({seed for _,seed in self.r.CONDITIONS},{92601,92602})
    def test_positive_metadata_gate_and_mutation_rejection_without_scientific_IO(self):
        r=self.r
        with tempfile.TemporaryDirectory(dir=SOURCE.parent) as temp:
            folder=Path(temp)
            def put(name,value):
                path=folder/name; path.write_text(json.dumps(value,allow_nan=False),encoding="utf-8"); return r.old.record(path)
            code=folder/"controller.py"; code.write_text("# fixture controller\n",encoding="utf-8")
            auditor=folder/"auditor.py"; auditor.write_text("# fixture auditor\n",encoding="utf-8")
            source=r.old.record(code); audit_source=r.old.record(auditor)
            train={"status":"COMPLETE","protected_TEST_accessed":False,"controller_source":source,"F_before":{"hash":"same"},"F_after":{"hash":"same"},"fixed_D_E_only_training_fits":2,"all_2_locks_before_DEV":True,"failed_paths":[],"selection_registry":put("REGISTRY.json",{})}
            for key in ("joint_recon_diagnostic","joint_recon_audit","full_gradient_parent","full_gradient_parent_audit","causal_lag_diagnostic","causal_lag_audit"):
                train[key]=put(key+".json",{"status":"PASS" if "audit" in key else "COMPLETE","F_before":{"hash":"same"},"protected_TEST_accessed":False})
            t=put("TRAIN.json",train)
            audit={"status":"PASS","protected_TEST_accessed":False,"training_verification_status":"PASS","new_space_diagnostic_status":"COMPLETE","partial_paths":[],"new_output_space_diagnostic_failures":[],"diagnostic":t,"audit_source":audit_source,"totals":dict(r.AUDIT_COUNTS),**{k:train[k] for k in train if k.endswith(("diagnostic","audit","parent"))}}
            a=put("AUDIT.json",audit); report=folder/"REPORT.md"; report.write_text("generated fixture",encoding="utf-8"); rp=r.old.record(report)
            mf=put("MANIFEST.json",{"status":"COMPLETE","protected_TEST_accessed":False,"inputs":{"train":t,"audit":a},"files":[]})
            ra=put("REPORT_AUDIT.json",{"status":"PASS","protected_TEST_accessed":False,"report":rp,"manifest":mf})
            paths=[t["path"],a["path"],rp["path"],mf["path"],ra["path"]]; pins=dict(zip(r.FIXED_SHA,[t["sha256"],a["sha256"],rp["sha256"],mf["sha256"],ra["sha256"]]))
            with patch.multiple(r,FIXED_SHA=pins,CONTROLLER_SHA=source["sha256"],AUDITOR_SHA=audit_source["sha256"]):
                values=r.load_gates(*paths); self.assertEqual(values[0]["fixed_D_E_only_training_fits"],2)
                Path(t["path"]).write_text(json.dumps(train)+" ",encoding="utf-8")
                with self.assertRaisesRegex(ValueError,"EXACT_CURRENT"): r.load_gates(*paths)
    def test_E0_FIT_DEV_sources_are_actual_Torch_latents_not_FORECAST_D_C_prediction(self):
        joint={"datasets":{"robin":{"conditions":{"92601":{"CP0_TUNE":{"state":{"path":"E0"},"artifact":{"path":"TUNE_TORCH"}}}}}}}
        full={"CP0_LS_and_RAW_baselines":{"robin":{"L7":{s:{"CP0_LS":{"artifact":{"path":s+"_FORECAST_D"}}} for s in ("fit","val")}}}}
        only={"datasets":{"robin":{"conditions":{"92601":{"selected_state":{"path":"E1"},"selected_update":400,"checkpoints":[{"update":400,"artifact":{"path":"E1_TUNE"}}],"FIT":{"artifact":{"path":"E1_FIT"}},"DEV":{"artifact":{"path":"E1_VAL"}}}}}}}
        self.assertEqual(self.r.cache_source(only,joint,full,"RECON_CP0","val")["path"],"val_FORECAST_D")
        self.assertEqual(self.r.cache_source(only,joint,full,"RECON_CP0","tune")["path"],"TUNE_TORCH")
        self.assertEqual(self.r.cache_source(only,joint,full,"E_ONLY_s92601","tune")["path"],"E1_TUNE")
        self.assertEqual(self.r.frozen_source(only,joint,"RECON_CP0")["path"],"E0")
    def test_all_six_locks_registry_precede_development_even_when_one_fit_fails(self):
        events=[]; report={"states":{s:{"native":{}} for s in self.r.STATES},"failed_paths":[],"native_training_fits":0,"training_updates_completed":0,"checkpoint_states_saved":0}
        def run(state,seed):
            events.append(("fit",state,seed))
            if (state,seed)==self.r.CONDITIONS[0]: raise ValueError("synthetic path failure")
            return {"status":"COMPLETE","training_updates_completed":600,"checkpoint_count":5}
        def seal(state,seed,row): events.append(("lock",state,seed)); return {"path":state+str(seed)}
        def registry(rows): events.append(("registry",len(rows))); return {"path":"REGISTRY"}
        resolved=self.r.resolve_all(report,run,seal,registry)
        self.assertEqual(events[-1],("registry",6)); self.assertEqual(len(resolved),5); self.assertEqual(len(report["failed_paths"]),1)
        self.assertTrue(report["all_6_locks_before_DEV"]); self.assertEqual(report["resolved_selection_locks"],6)
    def test_CP0_probe_preserves_numeric_failure_and_L7_flat_reference(self):
        import torch
        from lagged_native_head_v1 import LaggedNativeForecaster
        from run_linear_e_adaptation_v1 import predict_components
        rng=np.random.default_rng(7); raw={"history":rng.normal(size=(2,40,3)).astype(np.float32)}; raw["raw_last"]=raw["history"][:,-1]
        state={"E":rng.normal(size=(3,1)).astype(np.float32),"center":np.zeros(3,np.float32),"D":rng.normal(size=(7,3)).astype(np.float32)}
        model=LaggedNativeForecaster(state["center"],state["E"],state["D"],self.r.LAGS,L=40,H=5)
        outputs=predict_components(model,None,raw["history"],lambda _:None); cache={"latent_history":outputs["latent_history"].copy()}
        probe,arrays=self.r.cp0_probe(outputs,cache,state,raw)
        self.assertEqual(probe["status"],"PASS"); self.assertEqual(probe["optimizer_updates"],0); self.assertIn("NumPy_flat_prediction",arrays)
        outputs["prediction"][0,0,0]+=.1; probe,arrays=self.r.cp0_probe(outputs,cache,state,raw)
        self.assertEqual(probe["status"],"FAILED_PARITY_PRESERVED"); self.assertIn("native_prediction",arrays)
    def test_cached_Bolt_decode_uses_current_RECON_D_and_no_head_or_F_forward(self):
        import torch
        from lagged_native_head_v1 import LaggedNativeForecaster
        from unittest.mock import patch
        rng=np.random.default_rng(8); state={"E":rng.normal(size=(3,1)).astype(np.float32),"center":np.zeros(3,np.float32),"D":rng.normal(size=(7,3)).astype(np.float32)}
        model=LaggedNativeForecaster(state["center"],state["E"],state["D"],self.r.LAGS,L=40,H=5)
        cache={"latent_history":rng.normal(size=(2,40,1)).astype(np.float32),"latent_future":rng.normal(size=(2,5,1)).astype(np.float32),"prediction":np.full((2,5,3),999.,np.float32)}; raw={"raw_last":np.zeros((2,3),np.float32)}
        with patch.object(model.head,"forward",side_effect=AssertionError("head must not run for cached Bolt")):
            output=self.r.decode_cached_Bolt(model,cache,raw,lambda _:None)
        self.assertFalse(np.array_equal(output["prediction"],cache["prediction"]))
        from causal_lag_decoder_v1 import predict
        np.testing.assert_allclose(output["prediction"],predict(cache["latent_history"],cache["latent_future"],raw["raw_last"],state["D"],self.r.LAGS),atol=1e-5,rtol=2e-6)


if __name__=="__main__": unittest.main()
