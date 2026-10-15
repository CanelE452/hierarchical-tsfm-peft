"""Tiny generated arrays only; no actual research payload, F or audit execution."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import audit_purged_readout_stability_v1 as audit
import horizon_decoder_delta_v1 as decoder
import horizon_correction_error_terms_v1 as terms


class PurgedAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):audit.initialize_numpy()

    def test_future_origin_first_timestamp_strict_purge_and_original_index_order(self):
        origins=np.arange(24,dtype=np.int64)[::-1].copy();p=audit.partition_reference(origins,a_count=8,b_count=8,horizon=4)
        np.testing.assert_array_equal(origins[p['A_indices']],np.arange(5));np.testing.assert_array_equal(origins[p['B_indices']],np.arange(8,13));np.testing.assert_array_equal(origins[p['C_indices']],np.arange(16,24));np.testing.assert_array_equal(origins[p['A_purged_indices']],np.arange(5,8))
        self.assertEqual(origins[p['A_indices'][-1]]+3,7);self.assertEqual(origins[p['B_candidate_indices'][0]],8)

    def test_duplicate_origin_and_empty_purged_cohort_are_rejected(self):
        origins=np.arange(24,dtype=np.int64);origins[2]=origins[1]
        with self.assertRaisesRegex(ValueError,'UNIQUE_SORTED'):audit.partition_reference(origins,a_count=8,b_count=8,horizon=4)
        with self.assertRaisesRegex(ValueError,'NONEMPTY'):audit.partition_reference(np.arange(24,dtype=np.int64),a_count=8,b_count=8,horizon=48)

    def test_response_timestamp_occurrences_are_not_independent_samples(self):
        mask=np.ones((3,4,2),bool);mask[0,0,1]=False;value=audit.timestamp_reference(np.array([0,1,5],np.int64),mask)
        self.assertEqual(value['response_timestamp_occurrences'],12);self.assertEqual(value['unique_response_timestamps'],9);self.assertEqual(value['maximum_response_multiplicity'],2);self.assertEqual(value['observed_target_counts_per_channel'],[12,11]);self.assertEqual(value['last_response_timestamp'],8)

    def test_A_only_uncentered_RMS_does_not_pool_later_cohorts(self):
        a=np.arange(12,dtype=np.float32).reshape(2,3,2);a[:,:,0]=0.;b=np.full_like(a,10000.);scale,zero=audit.old.actual_F_scale(a);pooled,_=audit.old.actual_F_scale(np.concatenate((a,b)))
        self.assertEqual(scale[0],1.);self.assertTrue(zero[0]);self.assertGreater(pooled[1],scale[1]*100);np.testing.assert_allclose(scale,decoder.feature_scale(a)[0],atol=1e-12,rtol=1e-12)

    def test_selected_independent_primal_and_lstsq_unequal_observation_counts(self):
        rng=np.random.default_rng(92);x=rng.normal(size=(3,48,4)).astype(np.float32);baseline=np.zeros((3,48,3),np.float32);target=(x@rng.normal(size=(4,3))).astype(np.float32);mask=np.ones_like(target,bool);mask[::2,::3,1]=False;mask[:,:,2]=False;target[~mask]=np.nan;response=audit.old.observed_residual(target,baseline,mask);scale,zero=audit.old.actual_F_scale(x);states,_=decoder.fit_path(x,response,mask,scale,zero)
        for index in (0,4):
            with self.subTest(alpha=states[index]['alpha']):
                expected,metadata=audit.old.selected_fit(x,response,mask,scale,zero,states[index]['alpha']);np.testing.assert_allclose(expected,states[index]['coefficient_FP64'],atol=1e-8,rtol=1e-7);np.testing.assert_array_equal(expected[:,2],0.)
        self.assertEqual(metadata['groups'][-1]['fallback'],'NO_OBSERVED_RESPONSE_ZERO_CORRECTION')

    def test_actual_endpoint_terms_mask_macro_intervals_and_rounding(self):
        rng=np.random.default_rng(33);baseline=rng.normal(size=(2,48,3)).astype(np.float32);prediction=(baseline+np.float32(.1)).astype(np.float32);target=rng.normal(size=baseline.shape).astype(np.float32);mask=np.ones_like(target,bool);mask[0,::2,1]=False;mask[:,:,2]=False;target[~mask]=np.nan;raw={'target':target,'observed':mask};saved=terms.diagnose(baseline,prediction,target,mask)
        self.assertEqual(audit.endpoint_check(saved,baseline,prediction,raw),52);wanted=audit.endpoint_reference(baseline,prediction,target,mask);self.assertEqual(wanted['active_channels'],2);self.assertLess(abs(wanted['closure_residual']),1e-12)
        changed=copy.deepcopy(saved);changed['full']['interaction_term']+=.1
        with self.assertRaises(ValueError):audit.endpoint_check(changed,baseline,prediction,raw)

    def test_B_selection_ties_do_not_prefer_ZERO_or_last_alpha(self):
        candidates=[{'id':'ALPHA_'+format(a,'g'),'status':'COMPLETE','B_MSE':1.} for a in audit.old.ALPHAS]+[{'id':'ZERO_DELTA','status':'COMPLETE','B_MSE':1.}]
        self.assertEqual(audit.select_reference(candidates)['id'],'ALPHA_0');candidates[-1]['B_MSE']=.8;self.assertEqual(audit.select_reference(candidates)['id'],'ZERO_DELTA');candidates.pop()
        with self.assertRaisesRegex(ValueError,'ELEVEN'):audit.select_reference(candidates)

    def test_actual_result_gate_closed_before_file_hash_or_payload_access(self):
        with patch.object(audit,'RESULT_SHA',None),patch.object(audit.base,'file_record',side_effect=AssertionError('no actual IO')),self.assertRaisesRegex(ValueError,'NOT_YET_REGISTERED'):audit.metadata_gate('missing',None)

    def test_real_execute_entry_reaches_metadata_gate_without_shadow_or_array_load(self):
        import continuation_runtime_v1 as rt
        class Sentinel(Exception):pass
        class Meter:
            def __init__(self,*a,**k):pass
            def check(self,*a,**k):pass
            def close(self,status,failure):return {'status':status,'failure':failure,'synthetic_only':True}
        with tempfile.TemporaryDirectory() as folder,patch.object(rt,'Meter',Meter),patch.object(rt,'write_new') as write,patch.object(audit,'metadata_gate',side_effect=Sentinel('ENTRY_REACHED')) as gate,patch.object(audit,'initialize_numpy',side_effect=AssertionError('no scientific initialization')):
            self.assertEqual(audit.execute('missing',Path(folder)/'audit.json'),2);gate.assert_called_once();saved=write.call_args.args[1];self.assertEqual(saved['status'],'FAILED_PRESERVED');self.assertEqual(saved['failure'],{'type':'Sentinel','message':'ENTRY_REACHED'});json.dumps(saved,allow_nan=False)

    def test_generated_full_controller_saved_arrays_to_independent_audit(self):
        import sys
        import run_purged_readout_stability_v1 as run
        import fixed_decoder_span_v1 as io
        import continuation_runtime_v1 as rt
        import run_fit_target_comparison_v1 as inputs
        from threadpoolctl import threadpool_limits
        sys.path.insert(0,str(run.OLD))
        import prepare_inputs
        class Meter:
            def __init__(self,*a,**k):pass
            def check(self,*a,**k):pass
            def protected_check(self):pass
            def close(self,status,failure):return {'status':status,'synthetic_only':True}
        with tempfile.TemporaryDirectory(dir=run.HERE,prefix='synthetic_purged_audit_') as directory:
            folder=Path(directory);n,c,k=256,17,8;history=np.full((n,512,c),.1,np.float32);future=np.full((n,48,c),.1,np.float32);target=np.full_like(future,.2);observed=np.ones_like(target,bool);observed[0,0,0]=False;target[~observed]=np.nan;raw={'history':history,'history_observed':np.ones_like(history,bool),'target':target,'observed':observed,'raw_last':history[:,-1].copy(),'origins':np.arange(n,dtype=np.int64)+512,'mean':np.zeros(c,np.float32),'std':np.ones(c,np.float32),'raw_forecast':future};source=io.save_npz(folder/'original_FIT.npz',raw)
            E=np.eye(c,dtype=np.float32)[:,:k].copy();D=np.zeros((7*k,c),np.float32);D[:k,:k]=np.eye(k,dtype=np.float32);state=io.save_npz(folder/'original_PCA_STATE.npz',{'E':E,'D':D,'center':np.zeros(c,np.float32)});z=history[:,:,:k].copy();u=future[:,:,:k].copy();cache=io.save_npz(folder/'original_PCA_FIT.npz',{'latent_history':z,'latent_future':u,'prediction':future.copy(),'FLAT_prediction':future.copy(),'origins':raw['origins'],'lag_features':audit.old.old_lag_features(z,u)})
            def saved(name,value):
                path=folder/name;path.write_text(json.dumps(value,allow_nan=False),encoding='utf-8');return run.old.record(path)
            horizon={'status':'COMPLETE','datasets':{'robin':{'spaces':{s:{'contexts':{'fit':{'input_source':source,'cached_latent_source':cache if s=='PCA_K8' else None,'frozen_state_source':state if s=='PCA_K8' else None}}} for s in audit.ROUTES}}}};cross={'status':'COMPLETE','datasets':{'robin':{'conditions':{'PCA_K8':{'state_artifact':state,'splits':{'fit':{'input_source':source,'artifact':cache}}}}}}};science={'datasets':{'robin':{'spaces':{'PCA':{'conditions':{'L7_RECON':{'splits':{'fit':{'input_source':source}}}}}}}}};h=saved('horizon.json',horizon);ha=saved('horizon_audit.json',{'status':'PASS','diagnostic':h});cr=saved('cross.json',cross);ca=saved('cross_audit.json',{'status':'PASS','diagnostic':cr});refs={'horizon_diagnostic':h,'horizon_audit':ha,'cross_moment_diagnostic':cr,'cross_moment_audit':ca,'causal_lag_diagnostic':saved('causal.json',science)};output=folder/'run'
            with patch.object(run,'load_gates',return_value=(horizon,{},cross,science,refs)),patch.object(run.old,'fresh_output',return_value=output),patch.object(rt,'Meter',Meter),patch.object(inputs,'prior_file_records',return_value=[]),patch.object(prepare_inputs,'install_protected_test_guard'),threadpool_limits(limits=2):self.assertEqual(run.execute(SimpleNamespace(tag='synthetic-purged-audit')),0)
            actual=output/'RESULT.json';binding=run.old.record(actual)
            with patch.object(audit,'RESULT_SHA',binding['sha256']),patch.object(audit,'CONTROLLER_SHA',run.old.record(run.__file__)['sha256']),patch.object(audit,'HORIZON_SHA',h['sha256']),patch.object(audit,'HORIZON_AUDIT_SHA',ha['sha256']),patch.object(audit,'CROSS_SHA',cr['sha256']),patch.object(audit,'CROSS_AUDIT_SHA',ca['sha256']),patch.object(audit.ridge,'receipt_check',return_value=[]),threadpool_limits(limits=2):result=audit.audit_result(actual,Meter())
            self.assertEqual(result['status'],'PASS');self.assertEqual(result['totals']['readout_fit_paths_checked'],2);self.assertEqual(result['totals']['A_B_prediction_evaluations_checked'],44);self.assertEqual(result['totals']['selected_C_prediction_evaluations_checked'],2);self.assertEqual(result['totals']['FP32_arithmetic_predictions_checked'],46);self.assertEqual(result['totals']['endpoint_diagnostic_cells_checked'],312);self.assertFalse(result['independent_OS_C_access_events_verified']);json.dumps(result,allow_nan=False)


if __name__=='__main__':unittest.main()
