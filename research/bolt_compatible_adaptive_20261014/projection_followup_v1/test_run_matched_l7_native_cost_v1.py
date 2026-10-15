"""SOURCE/metadata and generated CPU tensors only; no actual assets or F."""
from pathlib import Path
import ast
import importlib.util
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).with_name("run_matched_l7_native_cost_v1.py")


def module():
    if not SOURCE.exists(): raise AssertionError("matched L7 native cost controller missing")
    spec=importlib.util.spec_from_file_location("l7_cost_controller_fixture",SOURCE); value=importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


class CostControllerTests(unittest.TestCase):
    def setUp(self): self.r=module()
    def test_unregistered_science_audit_closes_before_any_asset_read(self):
        with patch.object(self.r,"NATIVE_AUDIT_SHA",None),patch.object(self.r.old,"record",side_effect=AssertionError("closed IO")),self.assertRaisesRegex(ValueError,"NOT_YET_REGISTERED"):
            self.r.load_gates("missing","missing")
    def test_routes_and_existing_input_boundary_helpers_are_reused(self):
        import run_linear_e_cost_v1 as old
        self.assertEqual(len(self.r.ROUTES),10); self.assertEqual(self.r.STATES,("RECON_CP0","E_ONLY_s92601","E_ONLY_s92602"))
        self.assertIs(self.r.canonical_tune_bridge,old.canonical_tune_bridge); self.assertIs(self.r.adapter_predictor,old.adapter_predictor); self.assertIs(self.r.physical_precheck,old.physical_precheck)
    def test_native_inference_uses_None_predictor_and_all_weights_frozen(self):
        import numpy as np
        import torch
        from lagged_native_head_v1 import LaggedNativeForecaster
        from native_head_v1 import state_hash
        rng=np.random.default_rng(8); e=rng.normal(size=(3,1)).astype(np.float32); d=rng.normal(size=(7,3)).astype(np.float32)
        model=LaggedNativeForecaster(np.zeros(3,np.float32),e,d,self.r.LAGS,L=40,H=5).eval().requires_grad_(False)
        history=rng.normal(size=(2,40,3)).astype(np.float32); before=state_hash(model.state_dict())
        callback=self.r.adapter_predictor(model,None,lambda x:torch.as_tensor(x))
        result=callback(history,np.ones_like(history,bool)); self.assertEqual(result.shape,(2,5,3)); self.assertEqual(before,state_hash(model.state_dict()))
        self.assertEqual(len(self.r.native_head_hashes({"toy_NATIVE_s92601":model})),1)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
    def test_embedded_head_hash_detects_head_only_mutation(self):
        import numpy as np
        import torch
        from lagged_native_head_v1 import LaggedNativeForecaster
        model=LaggedNativeForecaster(np.zeros(2,np.float32),np.ones((2,1),np.float32),np.ones((7,2),np.float32),self.r.LAGS)
        before=self.r.native_head_hashes({"toy_NATIVE_s92601":model})
        with torch.no_grad(): model.head.bias[0,0]=1.
        self.assertNotEqual(before,self.r.native_head_hashes({"toy_NATIVE_s92601":model}))
    def test_runner_has_no_training_or_train_only_guard_call(self):
        tree=ast.parse(SOURCE.read_text(encoding="utf-8")); calls=[]
        for n in ast.walk(tree):
            if isinstance(n,ast.Call): calls.append(ast.unparse(n.func))
        self.assertFalse(any(x.endswith(("train_head","frozen_status","backward","step","fit")) for x in calls))


if __name__=="__main__": unittest.main()
