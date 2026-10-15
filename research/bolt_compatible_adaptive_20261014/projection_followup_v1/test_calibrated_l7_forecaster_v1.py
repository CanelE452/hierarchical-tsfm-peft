"""Generated CPU E/F/q/D only; toy F is a callable, no real models or data."""
import unittest

import numpy as np
import torch

from calibrated_l7_forecaster_v1 import CalibratedL7Forecaster
from linear_e_adaptation_v1 import LinearLagAdapter
from fixed_l7_latent_calibration_v1 import LAGS,apply_calibration


def fixture():
    rng=np.random.default_rng(536); c,k=3,2
    center=rng.normal(size=c).astype(np.float32); e=rng.normal(scale=.1,size=(c,k)).astype(np.float32); d=rng.normal(scale=.1,size=(14,c)).astype(np.float32); x=rng.normal(size=(4,512,c)).astype(np.float32)
    return center,e,d,torch.as_tensor(x)


class CalibratedForecasterTests(unittest.TestCase):
    def test_identity_calls_actual_F_on_each_new_input_and_decodes_without_cached_C(self):
        center,e,d,x=fixture(); model=CalibratedL7Forecaster(center,e,d,np.zeros(2,np.float32),np.zeros(2,np.float32),family="SCALE_ONLY",identity=True); plain=LinearLagAdapter(center,e,d,LAGS,learn_E=False).eval().requires_grad_(False)
        calls=[]; outputs=[]
        def toyF(z):
            calls.append(z.detach().clone()); future=z[:,-1:].expand(-1,48,-1)+np.float32(.125); outputs.append(future); return future
        first,past,future=model(x,toyF,return_components=True); self.assertIs(future,outputs[0]); self.assertEqual(len(calls),1)
        expected=plain.decode(x[:,-1],past,outputs[0]); np.testing.assert_array_equal(first.numpy(),expected.numpy())
        second=model(x+1,toyF); self.assertEqual(len(calls),2); self.assertFalse(torch.equal(first,second))
        self.assertFalse(model.calibration_config()["cached_baseline_returned"])

    def test_both_families_use_pure_FP32_q_and_original_lag_decode(self):
        center,e,d,x=fixture(); a=np.array([.2,-.4],np.float32)
        for family,b in (("SCALE_ONLY",np.zeros(2,np.float32)),("SCALE_BIAS",np.array([.1,-.3],np.float32))):
            with self.subTest(family=family):
                model=CalibratedL7Forecaster(center,e,d,a,b,family=family); captured=[]
                def toyF(z): captured.append(z[:,-1:].expand(-1,48,-1)+.5); return captured[-1]
                p,z,q=model(x,toyF,return_components=True); expected=apply_calibration(z,captured[0],a,b)
                np.testing.assert_array_equal(q.numpy(),expected.numpy()); np.testing.assert_array_equal(p.numpy(),model.decode(x[:,-1],z,expected).numpy())

    def test_frozen_state_contains_GPU_setup_buffers_and_roundtrips(self):
        center,e,d,x=fixture(); model=CalibratedL7Forecaster(center,e,d,np.ones(2,np.float32),np.zeros(2,np.float32),family="SCALE_ONLY")
        self.assertEqual(set(model.state_dict()),{"E","D","center","a","b"}); self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
        other=CalibratedL7Forecaster(center,e,d,np.zeros(2,np.float32),np.zeros(2,np.float32),family="SCALE_ONLY"); other.load_state_dict(model.state_dict())
        for key in model.state_dict(): self.assertTrue(torch.equal(model.state_dict()[key],other.state_dict()[key]))

    def test_bad_identity_or_family_bias_and_nonfinite_coefficients_fail(self):
        center,e,d,_=fixture()
        cases=[(np.ones(2,np.float32),np.zeros(2,np.float32),"SCALE_ONLY",True),(np.zeros(2,np.float32),np.ones(2,np.float32),"SCALE_ONLY",False),(np.full(2,np.nan,np.float32),np.zeros(2,np.float32),"SCALE_BIAS",False)]
        for a,b,family,identity in cases:
            with self.subTest(family=family,identity=identity),self.assertRaises(ValueError): CalibratedL7Forecaster(center,e,d,a,b,family=family,identity=identity)


if __name__=="__main__": unittest.main()
