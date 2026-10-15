"""Independent reference math on generated CPU arrays; no research files/F/GPU."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import audit_horizon_decoder_diagnostic_v1 as audit
import horizon_decoder_delta_v1 as producer


class HorizonDecoderAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):audit.initialize_numpy()

    def fixture(self):
        rng=np.random.default_rng(631);history=rng.normal(size=(4,40,2)).astype(np.float32);future=rng.normal(size=(4,48,2)).astype(np.float32);phi=audit.old_lag_features(history,future);baseline=rng.normal(size=(4,48,3)).astype(np.float32);target=rng.normal(size=baseline.shape).astype(np.float32);mask=np.ones_like(target,bool);mask[::2,1::3,1]=False;scale,constant=producer.feature_scale(phi)
        return phi,baseline,target,mask,scale,constant

    def test_independent_old_lag_index_includes_known_prefix_and_lag_specific_reference(self):
        from causal_lag_decoder_v1 import lagged_delta_features
        history=np.arange(80,dtype=np.float32).reshape(1,40,2);future=np.arange(96,dtype=np.float32).reshape(1,48,2)+1000
        actual=audit.old_lag_features(history,future);np.testing.assert_array_equal(actual,lagged_delta_features(history,future,audit.LAGS))
        np.testing.assert_array_equal(actual[:,0,2:4],history[:,-1]-history[:,-2]);np.testing.assert_array_equal(actual[:,0,-2:],history[:,-32]-history[:,-33]);np.testing.assert_array_equal(actual[:,32,-2:],future[:,0]-history[:,-33])

    def test_first_future_and_last_ramp_endpoints_match_registered_fp32_gate(self):
        phi,*_=self.fixture()
        for arm in audit.ARMS:
            gate,features=audit.gate_and_features(phi,arm);np.testing.assert_array_equal(gate,producer.gate(48,arm));np.testing.assert_array_equal(features,producer.gated_features(phi,arm))
        gate,features=audit.gate_and_features(phi,"RAMP_DELTA");self.assertEqual(gate[0],0.);self.assertEqual(gate[-1],1.);np.testing.assert_array_equal(features[:,0],0.);np.testing.assert_array_equal(features[:,-1],phi[:,-1])

    def test_actual_F_uncentered_RMS_common_prior_and_exact_zero_column(self):
        phi,*_=self.fixture();phi[:,:,0]=0;phi[:,:,1]+=10.;scale,constant=audit.actual_F_scale(phi);expected,flags=producer.feature_scale(phi)
        np.testing.assert_allclose(scale,expected,atol=1e-12,rtol=1e-12);np.testing.assert_array_equal(constant,flags);self.assertEqual(scale[0],1.);self.assertGreater(scale[1],phi[:,:,1].astype(np.float64).std())

    def test_observed_residual_is_fp64_endpoint_difference_and_missing_labels_irrelevant(self):
        _,baseline,target,mask,*_=self.fixture();changed=target.copy();changed[~mask]=np.nan;a=audit.observed_residual(target,baseline,mask);b=audit.observed_residual(changed,baseline,mask);np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a[~mask],0.);np.testing.assert_array_equal(a[mask],target[mask].astype(np.float64)-baseline[mask].astype(np.float64))

    def test_selected_alpha_zero_and_positive_independent_solutions_match_svd(self):
        phi,baseline,target,mask,scale,constant=self.fixture();response=audit.observed_residual(target,baseline,mask)
        for arm in audit.ARMS:
            _,x=audit.gate_and_features(phi,arm);states,_=producer.fit_path(x,response,mask,scale,constant)
            for index in (0,1,4,9):
                with self.subTest(arm=arm,alpha=states[index]["alpha"]):
                    coefficient,meta=audit.selected_fit(x,response,mask,scale,constant,states[index]["alpha"]);np.testing.assert_allclose(coefficient,states[index]["coefficient_FP64"],atol=1e-8,rtol=1e-7);self.assertFalse(meta["feature_centering"]);self.assertFalse(meta["intercept_fitted"])

    def test_macro_count_normalization_replication_and_zero_response_fallback(self):
        phi,baseline,target,mask,scale,constant=self.fixture();response=audit.observed_residual(target,baseline,mask);alpha=.1
        first,_=audit.selected_fit(phi,response,mask,scale,constant,alpha);second,_=audit.selected_fit(np.concatenate((phi,phi)),np.concatenate((response,response)),np.concatenate((mask,mask)),scale,constant,alpha);np.testing.assert_allclose(first,second,atol=1e-12,rtol=1e-12)
        mask[:,:,2]=False;response[:,:,2]=np.nan;coefficient,meta=audit.selected_fit(phi,response,mask,scale,constant,alpha);np.testing.assert_array_equal(coefficient[:,2],0.);self.assertEqual(meta["groups"][-1]["fallback"],"NO_OBSERVED_RESPONSE_ZERO_CORRECTION")

    def test_no_intercept_and_no_centering_are_substantive_constraints(self):
        x=np.ones((3,4,1),np.float32);response=np.full((3,4,1),2.,np.float64);mask=np.ones_like(response,bool);coefficient,_=audit.selected_fit(x,response,mask,np.ones(1,np.float64),np.zeros(1,bool),.1)
        self.assertAlmostEqual(coefficient[0,0],2/1.1);self.assertNotEqual(coefficient[0,0],0.)

    def test_deployed_fp32_matches_separate_correction_and_ZERO_byte_bypass(self):
        phi,baseline,target,mask,scale,constant=self.fixture();response=audit.observed_residual(target,baseline,mask)
        for arm in audit.ARMS:
            _,x=audit.gate_and_features(phi,arm);state=producer.fit_path(x,response,mask,scale,constant)[0][4];actual=audit.deployed_prediction(baseline,phi,state["coefficient_FP32"],arm);np.testing.assert_array_equal(actual,producer.predict(baseline,phi,state,arm))
            if arm=="RAMP_DELTA":np.testing.assert_array_equal(actual[:,0],baseline[:,0])
        baseline[0,0,0]=np.float32(-0.);zero=audit.deployed_prediction(baseline,phi,np.zeros((phi.shape[2],3),np.float32),"STATIC_DELTA",zero=True);self.assertEqual(zero.tobytes(),baseline.tobytes())

    def test_oracle_mask_requires_all_consumed_positive_future_references(self):
        mask=np.ones((1,48,3),bool);mask[0,1,0]=False;valid=audit.oracle_dependency_mask(mask)
        for lag in audit.LAGS:self.assertFalse(valid[0,1+lag])
        self.assertTrue(valid[0,0]);self.assertTrue(valid[0,47]);self.assertFalse(valid[0,1]);self.assertTrue(mask[0,2].all());self.assertFalse(valid[0,2])

    def test_candidate_ties_keep_registered_alpha_order_before_ZERO(self):
        menu=[{"id":"ALPHA_"+format(alpha,"g"),"status":"COMPLETE","TUNE_MSE":1.} for alpha in audit.ALPHAS]+[{"id":"ZERO_DELTA","status":"COMPLETE","TUNE_MSE":1.}]
        self.assertEqual(audit.select_candidate(menu)["id"],"ALPHA_0");menu[-1]["TUNE_MSE"]=.9;self.assertEqual(audit.select_candidate(menu)["id"],"ZERO_DELTA")

    def test_actual_gate_remains_closed_before_any_file_read(self):
        with patch.object(audit,"RESULT_SHA",None),patch.object(audit.prior,"verified_json",side_effect=AssertionError("no actual IO")),self.assertRaisesRegex(ValueError,"NOT_YET_REGISTERED"):audit.audit_result("missing",None)

    def test_generated_full_controller_to_independent_auditor_24_path_schema(self):
        import sys
        import run_horizon_decoder_diagnostic_v1 as run
        import fixed_decoder_span_v1 as io
        import continuation_runtime_v1 as rt
        from threadpoolctl import threadpool_limits
        sys.path.insert(0,str(run.OLD))
        import prepare_inputs
        import run_fit_target_comparison_v1 as inputs
        class Meter:
            def __init__(self,*args,**kwargs):pass
            def check(self,*args,**kwargs):pass
            def protected_check(self):pass
            def close(self,status,failure):return {"status":status,"synthetic_only":True}
        datasets={"robin":(3,1,"val",2),"beijing_aotizhongxin":(3,1,"eval",2)}
        with tempfile.TemporaryDirectory(dir=run.HERE,prefix="synthetic_horizon_audit_") as directory:
            folder=Path(directory);output=folder/"run";cross={"status":"COMPLETE","F_before":{"old_algorithm":"synthetic","bi_algorithm":"synthetic"},"F_after":{"old_algorithm":"synthetic","bi_algorithm":"synthetic"},"datasets":{}};science={"datasets":{}}
            for data,(_,k,dev,dn) in datasets.items():
                science["datasets"][data]={"spaces":{"PCA":{"conditions":{"L7_RECON":{"splits":{}}}}}};cross["datasets"][data]={"conditions":{}}
                for width in (1,2):
                    state={"E":np.eye(3,dtype=np.float32)[:,:width].copy(),"D":np.full((7*width,3),.01,np.float32),"center":np.zeros(3,np.float32)};name="PCA_K"+str(width);cross["datasets"][data]["conditions"][name]={"state_artifact":io.save_npz(folder/(data+name+"_state.npz"),state),"splits":{}}
                for split,n in (("fit",256),("tune",64),(dev,dn)):
                    current=np.broadcast_to(np.array([[.1,.2,.3]],np.float32),(n,3)).copy();history=np.broadcast_to(current[:,None],(n,512,3)).copy();future=np.broadcast_to(current[:,None],(n,48,3)).copy();target=future.copy();observed=np.ones_like(target,bool);observed[0,1,0]=False;target[~observed]=np.nan;raw={"history":history,"history_observed":np.ones_like(history,bool),"target":target,"observed":observed,"raw_last":current,"origins":np.arange(n,dtype=np.int64)+512,"mean":np.zeros(3,np.float32),"std":np.ones(3,np.float32),"raw_forecast":future};source=io.save_npz(folder/(data+split+"_raw.npz"),raw);science["datasets"][data]["spaces"]["PCA"]["conditions"]["L7_RECON"]["splits"][split]={"input_source":source}
                    for width in (1,2):
                        name="PCA_K"+str(width);owner=cross["datasets"][data]["conditions"][name];z=history[:,:,:width].copy();f=future[:,:,:width].copy();phi=audit.old_lag_features(z,f);cache={"latent_history":z,"latent_future":f,"lag_features":phi,"prediction":future.copy(),"FLAT_prediction":future.copy(),"origins":raw["origins"]};owner["splits"][split]={"input_source":source,"artifact":io.save_npz(folder/(data+name+split+"_cache.npz"),cache)}
            def save_json(name,value):
                path=folder/name;path.write_text(json.dumps(value,allow_nan=False),encoding="utf-8");return run.old.record(path)
            cross_record=save_json("cross.json",cross);cross_audit=save_json("cross_audit.json",{"status":"PASS","diagnostic":cross_record});science_record=save_json("science.json",science);refs={"cross_moment_diagnostic":cross_record,"cross_moment_audit":cross_audit,"causal_lag_diagnostic":science_record}
            conditions=tuple((d,s,m,a) for d in datasets for s in ("RAW_C","PCA_K1","PCA_K2") for m in ("F","FLAT") for a in audit.ARMS)
            with patch.object(run,"load_gates",return_value=(cross,science,refs)),patch.object(run.old,"fresh_output",return_value=output),patch.object(run,"DATASETS",datasets),patch.object(run,"CONDITIONS",conditions),patch.object(rt,"Meter",Meter),patch.object(inputs,"prior_file_records",return_value=[]),patch.object(prepare_inputs,"install_protected_test_guard"),threadpool_limits(limits=2):
                self.assertEqual(run.execute(SimpleNamespace(tag="synthetic-horizon-audit")),0)
            actual=output/"RESULT.json";binding=run.old.record(actual)
            with patch.object(audit,"RESULT_SHA",binding["sha256"]),patch.object(audit,"CROSS_RESULT_SHA",cross_record["sha256"]),patch.object(audit,"CROSS_AUDIT_SHA",cross_audit["sha256"]),patch.object(audit,"DATASETS",datasets),patch.object(audit.ridge,"receipt_check",return_value=[]),threadpool_limits(limits=2):
                result=audit.audit_result(actual,Meter())
            self.assertEqual(result["status"],"PASS");self.assertEqual(result["totals"]["decoder_fit_paths_checked"],24);self.assertEqual(result["totals"]["alpha_candidates_checked"],240);self.assertEqual(result["totals"]["FIT_TUNE_prediction_evaluations_checked"],528);self.assertEqual(result["totals"]["selected_DEV_prediction_evaluations_checked"],24);self.assertEqual(result["totals"]["source_contexts_checked"],18);json.dumps(result,allow_nan=False)


if __name__=="__main__":unittest.main()
