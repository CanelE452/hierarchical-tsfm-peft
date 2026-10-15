"""Generated FP32 arithmetic only; no research payload, model, F or GPU."""
import json
import unittest

import numpy as np
import horizon_correction_error_terms_v1 as terms


class CorrectionErrorTermsTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(709);b=rng.normal(size=(5,48,3)).astype(np.float32);p=(b+rng.normal(scale=.2,size=b.shape).astype(np.float32)).astype(np.float32);y=rng.normal(size=b.shape).astype(np.float32);mask=rng.random(b.shape)>.35
        return b,p,y,mask

    def test_direct_observed_channel_loop_and_signed_identity(self):
        b,p,y,mask=self.fixture();result=terms.decompose(b,p,y,mask);expected=[]
        for j in range(3):
            values=[]
            for i,h in zip(*np.where(mask[:,:,j])):
                e0=float(b[i,h,j])-float(y[i,h,j]);c=float(p[i,h,j])-float(b[i,h,j]);e1=float(p[i,h,j])-float(y[i,h,j]);values.append((e0*e0,e1*e1,2*e0*c,c*c))
            expected.append(np.mean(values,axis=0))
        expected=np.mean(expected,axis=0)
        for name,value in zip(("baseline_MSE","selected_MSE","interaction_term","correction_energy"),expected):self.assertAlmostEqual(result[name],value,places=13)
        self.assertAlmostEqual(result["delta_MSE"],expected[1]-expected[0],places=13);self.assertTrue(result["closure_pass"]);self.assertGreaterEqual(result["correction_energy"],0.)

    def test_same_channel_macro_is_not_pooled_or_mean_horizon_macro(self):
        b=np.zeros((2,48,2),np.float32);p=np.zeros_like(b);p[:,:,0]=1;p[:,:,1]=3;y=np.zeros_like(b);mask=np.zeros_like(b,bool);mask[:,:,0]=True;mask[0,0,1]=True
        result=terms.diagnose(b,p,y,mask);self.assertEqual(result["full"]["selected_MSE"],5.);self.assertNotAlmostEqual(result["full"]["selected_MSE"],float(np.mean(p[mask].astype(np.float64)**2)))
        horizon_mean=np.mean([row["selected_MSE"] for row in result["per_horizon"]]);self.assertNotEqual(result["full"]["selected_MSE"],horizon_mean)

    def test_absent_channels_and_intervals_are_null_and_unknown_labels_do_not_enter(self):
        b,p,y,mask=self.fixture();mask[:,:,2]=False;mask[:,16:32]=False;y[~mask]=np.nan
        first=terms.diagnose(b,p,y,mask);y[~mask]=np.inf;second=terms.diagnose(b,p,y,mask);self.assertEqual(first,second)
        self.assertIsNone(first["full"]["per_channel"][2]["baseline_MSE"]);self.assertEqual(first["registered_intervals"]["17_32"]["status"],"NO_OBSERVED_TARGETS");self.assertIsNone(first["registered_intervals"]["17_32"]["delta_MSE"]);json.dumps(first,allow_nan=False)

    def test_cast_endpoints_before_subtraction_and_actual_return_rounding(self):
        b=np.full((1,48,1),1e8,np.float32);p=b+np.float32(8);y=np.ones_like(b);mask=np.ones_like(b,bool);result=terms.decompose(b,p,y,mask)
        self.assertEqual(result["interaction_term"],2*(100000000.-1.)*8.)
        rounded=b+np.float32(.5);zero=terms.diagnose(b,rounded,y,mask);self.assertTrue(zero["prediction_baseline_byte_equal"]);self.assertEqual(zero["full"]["correction_energy"],0.);self.assertEqual(zero["full"]["interaction_term"],0.)

    def test_ZERO_requires_original_baseline_bytes_including_signed_zero(self):
        b,p,y,mask=self.fixture();p=b.copy();result=terms.diagnose(b,p,y,mask,require_zero=True);self.assertTrue(result["prediction_baseline_byte_equal"])
        b[0,0,0]=np.float32(-0.);p=b.copy();p[0,0,0]=np.float32(0.)
        with self.assertRaisesRegex(ValueError,"ZERO_ORIGINAL_BASELINE_BYTES"):terms.diagnose(b,p,y,mask,require_zero=True)

    def test_fixed_three_interval_boundaries_and_each_own_counts(self):
        b=np.zeros((1,48,1),np.float32);p=np.arange(1,49,dtype=np.float32)[None,:,None];y=np.zeros_like(b);mask=np.ones_like(b,bool);mask[:,16,0]=False
        result=terms.diagnose(b,p,y,mask);self.assertEqual(len(result["per_horizon"]),48)
        for key,start,stop in (("1_16",0,16),("17_32",16,32),("33_48",32,48)):
            expected=terms.decompose(b[:,start:stop],p[:,start:stop],y[:,start:stop],mask[:,start:stop]);self.assertEqual(result["registered_intervals"][key],expected)
        self.assertEqual(result["registered_intervals"]["17_32"]["observed_counts"],[15])

    def test_nonfinite_return_and_non_FP32_endpoints_fail(self):
        b,p,y,mask=self.fixture();p[0,0,0]=np.nan
        with self.assertRaisesRegex(ValueError,"FINITE_FP32"):terms.decompose(b,p,y,mask)
        p=b.copy()
        with self.assertRaisesRegex(ValueError,"FINITE_FP32"):terms.decompose(b.astype(np.float64),p,y,mask)


if __name__=="__main__":unittest.main()
