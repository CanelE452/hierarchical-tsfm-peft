"""Small synthetic CPU fixtures only; no original assets or actual F."""
from pathlib import Path
import importlib.util
import unittest
from unittest.mock import patch

import numpy as np
import torch

SOURCE=Path(__file__).with_name("lagged_native_head_v1.py")


def module():
    if not SOURCE.exists(): raise AssertionError("lagged native head source missing")
    spec=importlib.util.spec_from_file_location("lag_native_fixture",SOURCE); value=importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


class LaggedNativeTests(unittest.TestCase):
    def setUp(self):
        self.m=module(); torch.set_num_threads(2); rng=np.random.default_rng(19)
        self.x=rng.normal(size=(3,40,3)).astype(np.float32); self.E=rng.normal(size=(3,1)).astype(np.float32); self.D=rng.normal(size=(7,3)).astype(np.float32)
        self.model=self.m.LaggedNativeForecaster(np.zeros(3,np.float32),self.E,self.D,(0,1,2,4,8,16,32),L=40,H=5)
    def test_reuses_legacy_head_and_linear_lag_adapter(self):
        from native_head_v1 import NativeHead
        from linear_e_adaptation_v1 import LinearLagAdapter
        self.assertIsInstance(self.model,LinearLagAdapter); self.assertIsInstance(self.model.head,NativeHead)
        self.assertEqual(self.m.frozen_status(self.model)["trainable_parameter_names"],["head.weight","head.bias"])
    def test_zero_head_is_latent_persistence_but_L7_C_is_not_current_persistence(self):
        x=torch.from_numpy(self.x); p,z,f=self.model(x,return_components=True)
        self.assertEqual(f.detach().numpy().tobytes(),np.ascontiguousarray(np.broadcast_to(z[:,-1:].detach().numpy(),f.shape)).tobytes())
        self.assertFalse(torch.equal(p,x[:,-1:,].expand_as(p)))
        h0=self.model.decode(x[:,-1],z,f,include_origin=True); self.assertTrue(torch.equal(h0[:,0],x[:,-1]))
    def test_numpy_lag_formula_matches_and_no_F_callback_is_called(self):
        import causal_lag_decoder_v1 as math
        x=torch.from_numpy(self.x); p,z,f=self.model(x,return_components=True)
        expected=math.predict(z.detach().numpy(),f.detach().numpy(),self.x[:,-1],self.D,self.model.lags)
        np.testing.assert_allclose(p.detach().numpy(),expected,atol=1e-5,rtol=2e-6)
        with self.assertRaisesRegex(ValueError,"NO_F"): self.model(x,lambda _: (_ for _ in ()).throw(AssertionError("actualF")))
    def test_gradients_only_head_and_unobserved_NaNs_are_safe(self):
        from linear_e_adaptation_v1 import global_fit_macro_mse
        p=self.model(torch.from_numpy(self.x)); y=torch.zeros_like(p); mask=torch.ones_like(p,dtype=torch.bool); mask[0,0,0]=False; y[0,0,0]=float("nan")
        loss=global_fit_macro_mse(p,y,mask,mask.sum((0,1)),3); loss.backward()
        self.assertTrue(torch.isfinite(loss)); self.assertIsNone(self.model.E.grad); self.assertIsNone(self.model.D.grad)
        self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in self.model.head.parameters()))
    def test_two_updates_preserve_frozen_bytes_and_save_CP_before_selection(self):
        from run_linear_e_adaptation_v1 import state_arrays
        initial=state_arrays(self.model); y=np.zeros((3,5,3),np.float32); observed=np.ones_like(y,dtype=bool); payload={"history":self.x,"target":y,"observed":observed}; cps=[]; events=[]
        result=self.m.train_head(self.model,payload,payload,92601,events.append,lambda step,state,score,values:cps.append({"update":step,"state":state,"tune":score}),updates=2)
        self.assertEqual(result["training_updates_completed"],2)
        self.assertEqual([c["update"] for c in cps],[0,2])
        for key in ("E","D","center"): self.assertEqual(state_arrays(self.model)[key].tobytes(),initial[key].tobytes())
        self.assertEqual(result["post_optimizer_frozen_flag_checks"],2); self.assertEqual(result["frozen_state_checkpoint_byte_checks"],2)
        self.assertEqual(result["selected_update"],min(cps,key=lambda c:(c["tune"]["mse"],c["update"]))["update"])
        self.assertTrue(result["training_head_values_changed"]); self.assertFalse(result["independent_per_update_frozen_bytes_replayed"])
    def test_order_seed_is_only_sampling_and_matches_existing_rule(self):
        p={"history":self.x,"target":np.zeros((3,5,3),np.float32),"observed":np.ones((3,5,3),bool)}
        result=self.m.train_head(self.model,p,p,92602,lambda _:None,lambda *args:None,updates=1)
        import hashlib
        expected=np.random.default_rng(92602).integers(0,3,size=(1,4))
        self.assertEqual(result["order_sha256"],hashlib.sha256(expected.tobytes()).hexdigest())
    def test_equal_TUNE_scores_restore_earliest_CP0_even_after_nonzero_training(self):
        p={"history":self.x,"target":np.zeros((3,5,3),np.float32),"observed":np.ones((3,5,3),bool)}
        with patch.object(self.m,"score_predictions",return_value={"mse":1.,"mae":1.}):
            result=self.m.train_head(self.model,p,p,92601,lambda _:None,lambda *args:None,updates=2)
        self.assertEqual(result["selected_update"],0); self.assertTrue(result["training_head_values_changed"])
        self.assertFalse(result["selected_head_nonzero"]); self.assertTrue(all(torch.count_nonzero(p)==0 for p in self.model.head.parameters()))
    def test_stale_frozen_gradient_and_extra_trainable_parameter_are_rejected(self):
        self.model.D.grad=torch.zeros_like(self.model.D)
        with self.assertRaisesRegex(ValueError,"FROZEN"): self.m.frozen_status(self.model)
        self.model.D.grad=None; self.model.E.requires_grad_(True)
        with self.assertRaisesRegex(ValueError,"FROZEN"): self.m.frozen_status(self.model)


if __name__=="__main__": unittest.main()
