"""Small synthetic CPU numerical contracts; no actual assets, F or GPU."""
import inspect
import json
import unittest

import numpy as np
import torch

import fixed_l7_latent_calibration_v1 as cal
from linear_e_adaptation_v1 import LinearLagAdapter


def fixture(seed=7,k=2,c=3,n=4):
    rng=np.random.default_rng(seed); past=rng.normal(size=(n,40,k)).astype(np.float32); future=rng.normal(size=(n,48,k)).astype(np.float32); d=rng.normal(scale=.2,size=(7*k,c)).astype(np.float32)
    return past,future,d


def real_decode(past,future,current,d):
    n,h,k=future.shape; result=np.broadcast_to(current[:,None],(n,h,d.shape[1])).astype(np.float64).copy()
    for hi in range(h):
        for li,lag in enumerate(cal.LAGS):
            source=future[:,hi-lag] if hi>=lag else past[:,past.shape[1]+hi-lag]
            result[:,hi]+=(source.astype(np.float64)-past[:,-1-lag].astype(np.float64))@d[li*k:(li+1)*k].astype(np.float64)
    return result


def adapter(d):
    k=d.shape[0]//7; c=d.shape[1]
    return LinearLagAdapter(np.zeros(c,np.float32),np.eye(c,k,dtype=np.float32),d,cal.LAGS,learn_E=False).eval().requires_grad_(False)


class FixedL7CalibrationTests(unittest.TestCase):
    def test_independent_direct_decode_design_at_every_lag_boundary(self):
        p,u,d=fixture(); current=np.zeros((len(p),d.shape[1]),np.float32); a=np.array([.7,-1.3]); b=np.array([-.2,.4])
        changed=u.astype(np.float64)+a*(u.astype(np.float64)-p[:,-1:].astype(np.float64))+b
        for family,coeff in (("SCALE_ONLY",a),("SCALE_BIAS",np.r_[a,b])):
            v=changed if family=="SCALE_BIAS" else u.astype(np.float64)+a*(u.astype(np.float64)-p[:,-1:].astype(np.float64))
            expected=real_decode(p,v,current,d)-real_decode(p,u,current,d); actual=cal.design_columns(p,u,d,family=family)@coeff
            np.testing.assert_allclose(actual,expected,atol=cal.DESIGN_ATOL,rtol=cal.DESIGN_RTOL)
            np.testing.assert_allclose(actual[:,[0,1,31,32]],expected[:,[0,1,31,32]],atol=cal.DESIGN_ATOL,rtol=cal.DESIGN_RTOL)

    def test_future_only_lag32_columns_leave_known_prefix_unchanged(self):
        p,u,d=fixture(k=1,c=1); d[:]=0; d[-1]=2; phi=cal.design_columns(p,u,d,family="SCALE_BIAS")
        np.testing.assert_array_equal(phi[:,:32],0)
        np.testing.assert_array_equal(phi[:,32,0,0],2*(u[:,0,0].astype(np.float64)-p[:,-1,0].astype(np.float64)))
        np.testing.assert_array_equal(phi[:,32:,0,1],2)

    def test_known_solution_with_unequal_channel_masks_and_empty_channel(self):
        rng=np.random.default_rng(18); p=rng.integers(-4,5,size=(4,40,1)).astype(np.float32); u=rng.integers(-4,5,size=(4,48,1)).astype(np.float32); d=np.ones((7,3),np.float32)
        phi=cal.design_columns(p,u,d,family="SCALE_BIAS"); baseline=np.zeros((4,48,3),np.float32); target=(phi@np.array([.25,-.5])).astype(np.float32); observed=np.ones_like(target,bool); observed[:3,:,1]=False; observed[:,:,2]=False; target[~observed]=np.nan
        fit=cal.fit_calibration(p,u,d,baseline,target,observed,family="SCALE_BIAS"); candidate=fit["candidates"][0]
        np.testing.assert_allclose(candidate["coefficients_FP64"],[.25,-.5],atol=1e-12,rtol=1e-12)
        self.assertEqual(fit["metadata"]["FIT_channel_observed_counts"],[192,48,0]); self.assertEqual(fit["metadata"]["active_FIT_channels"],2)
        counts=observed.sum((0,1)); weights=np.broadcast_to(np.array([1/(2*192),1/(2*48),0]),target.shape)[observed]; expected=np.linalg.lstsq(phi[observed]*np.sqrt(weights[:,None]),target[observed]*np.sqrt(weights),rcond=1e-12)[0]
        np.testing.assert_allclose(candidate["coefficients_FP64"],expected,atol=1e-12,rtol=1e-12)

    def test_masked_nan_or_arbitrary_labels_never_enter_fit(self):
        p,u,d=fixture(); base=np.zeros((4,48,3),np.float32); mask=np.ones_like(base,bool); mask[0,:,1]=False
        target=base.copy(); target[~mask]=np.nan; first=cal.fit_calibration(p,u,d,base,target,mask)
        target[~mask]=1e30; second=cal.fit_calibration(p,u,d,base,target,mask)
        np.testing.assert_array_equal(first["column_scales_FP64"],second["column_scales_FP64"])
        for x,y in zip(first["candidates"],second["candidates"]): np.testing.assert_array_equal(x["coefficients_FP64"],y["coefficients_FP64"])

    def test_exact_zero_columns_reinserted_and_proportional_columns_not_causal_unique(self):
        p,u,d=fixture(k=3); p[:,:,1]=p[:,:,0]; u[:,:,1]=u[:,:,0]; d[1::3]=d[0::3]; d[2::3]=0
        base=np.zeros((4,48,3),np.float32); target=(cal.design_columns(p,u,d)@np.array([.1,.1,0])).astype(np.float32); fit=cal.fit_calibration(p,u,d,base,target,np.ones_like(base,bool))
        self.assertEqual(fit["metadata"]["exact_zero_FIT_columns"],[2]); self.assertEqual(fit["metadata"]["zero_lambda_numeric_rank"],1)
        for candidate in fit["candidates"]: self.assertEqual(candidate["coefficients_FP64"][2],0)
        self.assertFalse(fit["metadata"]["coefficients_identified_unique_latent_causal_effects_claimed"])

    def test_near_rank_zero_lambda_cutoff_and_positive_lambda_all_singular_filter(self):
        p,u,d=fixture(k=2,c=1); p[:,:,1]=p[:,:,0]; p[:,-1]=0; u[:,:,1]=u[:,:,0]; u[0,0]=0; u[0,0,1]=1e-13; d[1::2]=d[0::2]
        base=np.zeros((4,48,1),np.float32); target=np.ones_like(base); fit=cal.fit_calibration(p,u,d,base,target,np.ones_like(base,bool))
        self.assertEqual(fit["metadata"]["zero_lambda_relative_cutoff"],1e-12); self.assertEqual(len(fit["singularvalues_FP64"]),2); self.assertEqual(fit["metadata"]["zero_lambda_numeric_rank"],1)
        self.assertEqual(fit["candidates"][0]["metadata"]["singular_filter"],"relative_cutoff_1e-12")
        for candidate in fit["candidates"][1:]: self.assertEqual(candidate["metadata"]["singular_filter"],"all_singular_values_sigma_over_sigma2_plus_lambda")
        u[0,0,1]=1e-5; above=cal.fit_calibration(p,u,d,base,target,np.ones_like(base,bool)); self.assertEqual(above["metadata"]["zero_lambda_numeric_rank"],2)

    def test_positive_ridge_matches_independent_penalized_normal_equations(self):
        p,u,d=fixture(); phi=cal.design_columns(p,u,d,family="SCALE_BIAS"); base=np.zeros((4,48,3),np.float32); target=np.arange(base.size,dtype=np.float32).reshape(base.shape)/base.size; mask=np.ones_like(base,bool); mask[:3,:,1]=False
        fit=cal.fit_calibration(p,u,d,base,target,mask,family="SCALE_BIAS"); counts=mask.sum((0,1)); w=np.broadcast_to(1/(3*counts),mask.shape)[mask]; a=np.sqrt(w[:,None])*phi[mask]/fit["column_scales_FP64"]; r=np.sqrt(w)*target[mask]
        for candidate in fit["candidates"][1:]:
            expected=np.linalg.solve(a.T@a+candidate["lambda"]*np.eye(a.shape[1]),a.T@r)
            np.testing.assert_allclose(candidate["coefficients_FP64"]*fit["column_scales_FP64"],expected,atol=1e-10,rtol=1e-10)

    def test_nonfinite_FP32_candidate_is_preserved_with_JSONsafe_metadata(self):
        p=np.zeros((2,40,1),np.float32); u=np.ones((2,48,1),np.float32); d=np.zeros((7,1),np.float32); d[0]=np.nextafter(np.float32(0),np.float32(1)); base=np.zeros((2,48,1),np.float32)
        fit=cal.fit_calibration(p,u,d,base,np.ones_like(base),np.ones_like(base,bool))
        self.assertEqual(len(fit["candidates"]),7); self.assertEqual(fit["identity"]["status"],"COMPLETE")
        for row in fit["candidates"]:
            self.assertEqual(row["status"],"FAILED_NONFINITE_COEFFICIENTS_PRESERVED"); self.assertIsNone(row["a_FP32"]); self.assertTrue(np.isfinite(row["coefficients_FP64"]).all()); json.dumps(row["metadata"],allow_nan=False)

    def test_all_zero_observed_FIT_columns_have_no_fabricated_SVD(self):
        p,u,d=fixture(); d[:]=0; base=np.zeros((4,48,3),np.float32)
        fit=cal.fit_calibration(p,u,d,base,np.ones_like(base),np.ones_like(base,bool))
        self.assertEqual(fit["metadata"]["SVD_fits"],0); self.assertEqual(fit["metadata"]["zero_lambda_numeric_rank"],0); self.assertFalse(fit["active_columns"].any())
        for row in fit["candidates"]: np.testing.assert_array_equal(row["coefficients_FP64"],0.)

    def test_fit_rms_is_not_centered_and_no_tune_dev_inputs_exist(self):
        p=np.zeros((2,40,1),np.float32); u=np.ones((2,48,1),np.float32)*3; d=np.zeros((7,1),np.float32); d[0]=2; base=np.zeros((2,48,1),np.float32)
        fit=cal.fit_calibration(p,u,d,base,np.ones_like(base),np.ones_like(base,bool)); np.testing.assert_allclose(fit["column_scales_FP64"],[6.],atol=1e-14,rtol=1e-14)
        self.assertFalse(fit["metadata"]["column_or_response_centered"])
        self.assertFalse(any(key.lower().startswith(("tune","dev")) for key in inspect.signature(cal.fit_calibration).parameters))
        foreign_tune=np.zeros(100); first=fit["candidates"][0]["coefficients_FP64"].copy(); foreign_tune[:]=np.nan
        again=cal.fit_calibration(p,u,d,base,np.ones_like(base),np.ones_like(base,bool)); np.testing.assert_array_equal(first,again["candidates"][0]["coefficients_FP64"])

    def test_identity_is_byte_exact_baseline_and_original_future_without_arithmetic(self):
        p,u,d=fixture(); model=adapter(d); current=torch.zeros((4,3)); baseline=torch.zeros((4,48,3)); baseline[0,0,0]=-0.
        future=torch.as_tensor(u); prediction,unchanged=cal.predict_calibrated(model,current,torch.as_tensor(p),future,cal.identity_candidate(2,"SCALE_ONLY"),baseline)
        self.assertIs(prediction,baseline); self.assertIs(unchanged,future); self.assertTrue(np.signbit(prediction.numpy()[0,0,0]))

    def test_exact_tie_identity_then_larger_lambda_and_failed_path_excluded(self):
        records=[{"id":str(v),"kind":"RIDGE","lambda":v,"TUNE_MSE":1.,"status":"COMPLETE"} for v in cal.LAMBDAS]
        self.assertEqual(cal.select_candidate(records)["lambda"],10.)
        records.append({"id":"IDENTITY","kind":"IDENTITY","lambda":None,"TUNE_MSE":1.,"status":"COMPLETE"}); self.assertEqual(cal.select_candidate(records)["kind"],"IDENTITY")
        records.append({"id":"failed","kind":"RIDGE","lambda":0.,"TUNE_MSE":0.,"status":"FAILED_PATH_PRESERVED"}); self.assertEqual(cal.select_candidate(records)["kind"],"IDENTITY")

    def test_flat_latent_future_has_non_current_C_plateau_and_origin_is_exact(self):
        p=np.arange(40,dtype=np.float32)[None,:,None]; u=np.full((1,48,1),100,np.float32); d=np.ones((7,1),np.float32); model=adapter(d); current=torch.tensor([[20.]],dtype=torch.float32)
        calibrated=cal.apply_calibration(torch.as_tensor(p),torch.as_tensor(u),np.array([-1.],np.float32),np.array([0.],np.float32))
        np.testing.assert_array_equal(calibrated.numpy(),39.)
        prediction=model.decode(current,torch.as_tensor(p),calibrated); np.testing.assert_array_equal(prediction.numpy()[:,32:],83.)
        origin=model.decode(current,torch.as_tensor(p),calibrated,include_origin=True); self.assertEqual(origin[0,0,0],current[0,0])

    def test_FP32_stress_error_is_measured_not_zero_or_tolerance_relaxed(self):
        p,u,d=fixture(); p*=1e7; u*=1e7; model=adapter(d); current=torch.zeros((4,3)); baseline=model.decode(current,torch.as_tensor(p),torch.as_tensor(u))
        candidate={"kind":"RIDGE","status":"COMPLETE","a_FP32":np.array([.3,-.2],np.float32),"b_FP32":np.array([.01,-.02],np.float32)}
        prediction,changed=cal.predict_calibrated(model,current,torch.as_tensor(p),torch.as_tensor(u),candidate,baseline)
        info=cal.surrogate_diagnostics(prediction.numpy(),baseline.numpy(),cal.design_columns(p,u,d,family="SCALE_BIAS"),np.array([.3,-.2,.01,-.02]))
        self.assertGreater(info["max_abs_actual_FP32_minus_linear_surrogate"],0.); self.assertFalse(info["surrogate_zero_error_required"])
        expected=u+candidate["a_FP32"]*(u-p[:,-1:])+candidate["b_FP32"]; np.testing.assert_array_equal(changed.numpy(),expected)
        self.assertEqual((cal.DEPLOYMENT_ATOL,cal.DEPLOYMENT_RTOL),(1e-5,2e-6))

    def test_ordinary_FP32_C_deployment_matches_independent_numpy_q_and_original_lag_decoder(self):
        from causal_lag_decoder_v1 import predict as original_numpy_predict
        p,u,d=fixture(seed=918); model=adapter(d); current=np.array([[3.,-2.,1.],[-4.,1.,2.],[.5,2.,-3.],[6.,-5.,.1]],np.float32)
        baseline=model.decode(torch.as_tensor(current),torch.as_tensor(p),torch.as_tensor(u))
        for family,bias in (("SCALE_ONLY",np.zeros(2,np.float32)),("SCALE_BIAS",np.array([.15,-.25],np.float32))):
            with self.subTest(family=family):
                candidate={"kind":"RIDGE","family":family,"status":"COMPLETE","a_FP32":np.array([.3,-.8],np.float32),"b_FP32":bias}
                actual,q=cal.predict_calibrated(model,torch.as_tensor(current),torch.as_tensor(p),torch.as_tensor(u),candidate,baseline)
                q_numpy=u+candidate["a_FP32"]*(u-p[:,-1:])+bias
                expected=original_numpy_predict(p,q_numpy,current,d,cal.LAGS)
                pieces=[]
                for lag in cal.LAGS:
                    values=np.stack([q_numpy[:,h-lag] if h>=lag else p[:,p.shape[1]+h-lag] for h in range(48)],axis=1)
                    pieces.append(values-p[:,None,-1-lag])
                direct=current[:,None]+np.concatenate(pieces,axis=2)@d
                np.testing.assert_array_equal(expected,direct)
                np.testing.assert_allclose(q.numpy(),q_numpy,atol=cal.DEPLOYMENT_ATOL,rtol=cal.DEPLOYMENT_RTOL)
                np.testing.assert_allclose(actual.numpy(),expected,atol=cal.DEPLOYMENT_ATOL,rtol=cal.DEPLOYMENT_RTOL)
                np.testing.assert_allclose(actual.numpy()[:,[0,1,31,32]],direct[:,[0,1,31,32]],atol=cal.DEPLOYMENT_ATOL,rtol=cal.DEPLOYMENT_RTOL)

    def test_actual_cached_baseline_is_response_reference_not_remade_common_origin(self):
        p,u,d=fixture(k=1,c=1); phi=cal.design_columns(p,u,d); baseline=np.full((4,48,1),5.,np.float32); target=(baseline+phi@np.array([-.4])).astype(np.float32); mask=np.ones_like(baseline,bool)
        correct=cal.fit_calibration(p,u,d,baseline,target,mask); wrong=cal.fit_calibration(p,u,d,np.zeros_like(baseline),target,mask)
        self.assertAlmostEqual(correct["candidates"][0]["a_FP64"][0],-.4,places=6); self.assertNotAlmostEqual(wrong["candidates"][0]["a_FP64"][0],-.4,places=3)

    def test_negative_total_scale_is_not_clipped(self):
        p,u,d=fixture(k=1,c=1); phi=cal.design_columns(p,u,d); base=np.zeros((4,48,1),np.float32); target=(phi@np.array([-2.])).astype(np.float32)
        result=cal.fit_calibration(p,u,d,base,target,np.ones_like(base,bool)); self.assertAlmostEqual(result["candidates"][0]["a_FP64"][0],-2,places=6)
        self.assertEqual(result["candidates"][0]["metadata"]["negative_total_scale_channels"],[0]); self.assertFalse(result["candidates"][0]["metadata"]["coefficient_clipping_applied"])

    def test_no_observations_nonfinite_observed_or_bad_family_fail_explicitly(self):
        p,u,d=fixture(); base=np.zeros((4,48,3),np.float32)
        with self.assertRaisesRegex(ValueError,"NO_OBSERVED"): cal.fit_calibration(p,u,d,base,base,np.zeros_like(base,bool))
        target=base.copy(); target[0,0,0]=np.nan
        with self.assertRaises(ValueError): cal.fit_calibration(p,u,d,base,target,np.ones_like(base,bool))
        with self.assertRaises(ValueError): cal.design_columns(p,u,d,family="unregistered")


if __name__=="__main__": unittest.main()
