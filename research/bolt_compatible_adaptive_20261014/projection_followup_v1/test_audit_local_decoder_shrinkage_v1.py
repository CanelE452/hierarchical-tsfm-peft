"""Independent ridge, fallback, provenance, selection and saved forecast fixtures."""
import copy
import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import audit_local_decoder_shrinkage_v1 as audit

audit.initialize_numpy()


def past_case():
    z = np.tile(np.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]],np.float32),(128,1))[None]
    x = (z @ np.array([[2.,1.,-1.],[3.,-2.,.5]],np.float32) + np.array([7.,3.,-2.],np.float32)).astype(np.float32)
    mask = np.ones_like(x,bool)
    prior = np.array([[.5,.2,1.],[-.5,.4,-2.]],np.float32)
    return x,mask,z,prior


def state_fixture(x,mask,z,prior,rho):
    solved=audit.independent_augmented_fit(x,mask,z,prior,rho)
    arrays={"decoder_fit_FP64":solved["decoder_fit_FP64"],"decoder_deployed_FP32":solved["decoder_fit_FP64"].astype(np.float32),
            "decoder_bias_fit_FP64":solved["decoder_bias_fit_FP64"],"decoder_bias_FP32":solved["decoder_bias_fit_FP64"].astype(np.float32)}
    metadata={key:solved[key].tolist() for key in ("true_observed_counts","local_covariance_trace","actual_lambda","zero_trace_prior_slope_retained","no_observation_prior_with_current_intercept","observed_latent_means_per_C","observed_raw_C_means","observed_real_past_reconstruction_MSE","past_metric_valid","ridge_matrix_condition","ridge_matrix_condition_valid")}
    metadata.update(route="OBSERVED_PAST_CENTERED_RIDGE_TOWARD_STATIC_D",windows=len(x),channels=x.shape[-1],K=z.shape[-1],rho=rho,
        lambda_rule="rho*trace(observed_centered_Z.T@Z/n)/K",slope_singular_cutoff_used=False,future_labels_used_in_decoder_fit=False,
        absolute_decoder_gain_bound_claimed=False,decoder_interface_same_as_static=False,strict_K_only_compression=False,scientific_IO_in_module=False,
        intercept_analytic_unpenalized=True,native_head_frozen_intervention_not_reoptimized=True,inference_time_past_decoder_refitting=True,
        fit_input_bindings={"history":audit.raw_binding(x),"history_observed":audit.raw_binding(mask),"latent_history":audit.raw_binding(z),"prior_decoder":audit.raw_binding(prior)},
        state_arrays={name:audit.raw_binding(arrays[key]) for name,key in (("decoder_fit64","decoder_fit_FP64"),("decoder32","decoder_deployed_FP32"),("bias_fit64","decoder_bias_fit_FP64"),("bias32","decoder_bias_FP32"))},
        real_stationarity_residual_norm=solved["independent_augmented_stationarity_norm"].tolist(),
        max_abs_decoder_per_window=np.abs(arrays["decoder_deployed_FP32"].astype(np.float64)).max(axis=(1,2)).tolist(),
        decoder_Frobenius_norm_per_window=np.linalg.norm(arrays["decoder_deployed_FP32"].astype(np.float64),axis=(1,2)).tolist(),
        decoder_operator_norm_per_window=[float(np.linalg.svd(d.astype(np.float64),compute_uv=False)[0]) for d in arrays["decoder_deployed_FP32"]])
    arrays["metadata_json"]=np.array(json.dumps(metadata,allow_nan=False))
    return metadata,arrays


def save_npz(path,arrays):
    np.savez_compressed(path,**arrays)
    return {"path":str(path.resolve()),"sha256":audit.prior.file_sha(path),"arrays":{key:audit.raw_binding(value) for key,value in arrays.items()}}


def positive_backend_fixture(folder,*,noobs=False):
    x,mask,z,prior=past_case()
    if noobs:mask[:]=False
    rho=.1
    metadata,state=state_fixture(x,mask,z,prior,rho)
    future=np.repeat(z[:,-1:],48,axis=1)+np.float32(.5)
    prediction=audit.local_audit.local_forecast(x[:,-1],future,z[:,-1],state["decoder_deployed_FP32"])
    target=prediction+np.float32(.125)
    values={"history":x,"history_observed":mask,"target":target,"observed":np.ones_like(target,bool),"std":np.ones(x.shape[-1],np.float32),"raw_last":x[:,-1],"origins":np.array([512],np.int64)}
    past=np.einsum('nlk,nkc->nlc',z,state['decoder_deployed_FP32'],optimize=False)+state['decoder_bias_FP32'][:,None]
    artifacts={"past_prediction_FP32":past,"latent_history":z,"origins":values["origins"],"target":target,"observed":values["observed"],"raw_last":values["raw_last"],"BOLT_prediction":prediction,"BOLT_cached_future":future}
    source={"BOLT":{"path":"synthetic_original_cache","sha256":"fixture"}}
    prior_binding={"stored_prior_dtype":"float64","deployment_cast":"fixture","deployed_prior":audit.raw_binding(prior)}
    state_record=save_npz(folder/'state.npz',state)
    record=save_npz(folder/'predictions.npz',artifacts)
    scored={"status":"COMPLETE","scores_all_observed":audit.prior.scores(prediction,target,values['observed'],values['std']),"scores_complete_mask":audit.prior.scores(prediction,target,values['observed'],values['std']),"artifact":record,"prediction_member":"BOLT_prediction","new_latent_forward_or_training":False,"native_head_frozen_intervention_not_reoptimized":False}
    past_scores=audit.prior.scores(past,x,mask,values['std']) if mask.any() else {"status":"NO_OBSERVED_PAST_TARGETS","normalized":None}
    entry={"status":"COMPLETE","methods":{"BOLT":scored},"state_metadata":metadata,"state_artifact":state_record,"artifact":record,"cached_method_sources":source,"static_prior_deployment_binding":prior_binding,"native_past_byte_exact_before_D_reuse":False,"past_reconstruction_scores_observed_only":past_scores}
    context={"controls":{"latent_history":z},"prior32":prior,"prior_binding":prior_binding,"sources":source,"cached":{"BOLT":{"latent_future":future}}}
    return entry,values,context,'BOLT_NUMPY',rho


class IndependentShrinkageFixtures(unittest.TestCase):
    def test_augmented_LS_matches_diagonal_closed_form_and_intercept(self):
        x,mask,z,prior=past_case();rho=.1
        fit=audit.independent_augmented_fit(x,mask,z,prior,rho)
        truth=np.array([[2.,1.,-1.],[3.,-2.,.5]])
        expected=(truth+rho*prior.astype(np.float64))/(1+rho)
        np.testing.assert_allclose(fit['decoder_fit_FP64'][0],expected,atol=1e-13)
        np.testing.assert_allclose(fit['decoder_bias_fit_FP64'][0],[7.,3.,-2.],atol=1e-13)
        self.assertLess(fit['independent_augmented_stationarity_norm'].max(),1e-13)

    def test_filled_raw_responses_excluded_from_mean_and_ridge_fit(self):
        x,mask,z,prior=past_case();mask[:,::3,1]=False
        before=audit.independent_augmented_fit(x,mask,z,prior,.1);x[~mask]=1e10
        after=audit.independent_augmented_fit(x,mask,z,prior,.1)
        np.testing.assert_array_equal(before['decoder_fit_FP64'],after['decoder_fit_FP64'])
        np.testing.assert_array_equal(before['decoder_bias_fit_FP64'],after['decoder_bias_fit_FP64'])

    def test_zero_trace_is_prior_choice_not_unique_ridge_minimum(self):
        x,mask,z,prior=past_case();z[:]=2.
        fit=audit.independent_augmented_fit(x,mask,z,prior,.1)
        np.testing.assert_array_equal(fit['decoder_fit_FP64'][0],prior.astype(np.float64))
        self.assertTrue(fit['zero_trace_prior_slope_retained'].all());self.assertEqual(fit['actual_lambda'].max(),0.)
        self.assertEqual(fit['ridge_matrix_condition'].max(),0.)

    def test_no_observation_retains_prior_and_current_intercept_not_constant_future(self):
        x,mask,z,prior=past_case();mask[:]=False
        fit=audit.independent_augmented_fit(x,mask,z,prior,.1)
        np.testing.assert_array_equal(fit['decoder_fit_FP64'][0],prior.astype(np.float64))
        np.testing.assert_allclose(fit['decoder_bias_fit_FP64'],x[:,-1].astype(np.float64)-z[:,-1].astype(np.float64)@prior.astype(np.float64),atol=1e-13)
        prediction=audit.local_audit.local_forecast(x[:,-1],np.repeat(z[:,-1:],48,axis=1)+1,z[:,-1],fit['decoder_fit_FP64'].astype(np.float32))
        self.assertGreater(np.abs(prediction-x[:,-1,None]).max(),0.)

    def test_rank_deficient_ridge_retains_prior_unidentified_component(self):
        x,mask,z,prior=past_case();z[:,:,1]=z[:,:,0];prior[:,0]=[2.,-2.]
        fit=audit.independent_augmented_fit(x,mask,z,prior,.1)
        self.assertAlmostEqual(fit['decoder_fit_FP64'][0,0,0]-fit['decoder_fit_FP64'][0,1,0],4.,places=12)

    def test_relative_ridge_condition_does_not_bound_gain(self):
        signal=np.tile(np.array([-1.,1.],np.float32),256)
        z=(signal*1e-7)[None,:,None];x=np.stack([signal,2*signal],axis=1)[None]
        fit=audit.independent_augmented_fit(x,np.ones_like(x,bool),z,np.zeros((1,2),np.float32),.001)
        self.assertGreater(fit['decoder_fit_FP64'].max(),1e7)
        np.testing.assert_allclose(fit['ridge_matrix_condition'],1.,atol=1e-14)

    def test_unpenalized_intercept_moves_with_raw_mean_only(self):
        x,mask,z,prior=past_case()
        a=audit.independent_augmented_fit(x,mask,z,prior,1.);b=audit.independent_augmented_fit(x+5,mask,z,prior,1.)
        np.testing.assert_allclose(a['decoder_fit_FP64'],b['decoder_fit_FP64'],atol=1e-12)
        np.testing.assert_allclose(b['decoder_bias_fit_FP64']-a['decoder_bias_fit_FP64'],5.,atol=1e-12)

    def test_fit_has_no_future_and_uses_augmented_LS_not_normal_solve(self):
        self.assertEqual(set(inspect.signature(audit.independent_augmented_fit).parameters),{'history','observed','latent','prior32','rho'})
        self.assertIn('np.linalg.lstsq',inspect.getsource(audit.independent_augmented_fit))
        self.assertNotIn('np.linalg.solve',inspect.getsource(audit.independent_augmented_fit))

    def test_invalid_rho_and_unconverted_FP64_prior_rejected(self):
        x,mask,z,prior=past_case()
        for rho in (0,-1,True,float('nan'),float('inf')):
            with self.assertRaises(ValueError):audit.independent_augmented_fit(x,mask,z,prior,rho)
        with self.assertRaises(ValueError):audit.independent_augmented_fit(x,mask,z,prior.astype(np.float64),.1)

    def test_exact_first_tie_and_finite_Bolt_only_selection(self):
        scores={key:1. for key in audit.CANDIDATE_ORDER}
        self.assertEqual(audit.independent_selection(scores),'STATIC')
        scores['LOCAL']=.5;scores['RHO_0.001']=.5
        self.assertEqual(audit.independent_selection(scores),'LOCAL')
        scores['RHO_1']=.1
        self.assertEqual(audit.independent_selection(scores),'RHO_1')
        scores['STATIC']=float('nan')
        with self.assertRaises(ValueError):audit.independent_selection(scores)

    def test_four_locks_must_precede_first_DEV_event(self):
        records={(dataset,route):{'path':dataset+route,'sha256':'fixture'} for dataset in audit.DATASETS for route in ('PCA_PINV','FIT_TARGET_PINV')}
        events=[{'sequence':n,'kind':'SELECTION_LOCK_SAVED','dataset':data,'route':route,'binding':record} for n,((data,route),record) in enumerate(records.items())]
        events += [{'sequence':4,'kind':'ALL_FOUR_SELECTION_LOCK_FILES_SAVED','files':4},{'sequence':5,'kind':'DEV_SPLIT_BEGIN'}]
        self.assertTrue(audit.check_events(events,records)['all_four_lock_files_before_any_DEV_event'])
        events[0]={'sequence':0,'kind':'DEV_SPLIT_BEGIN'}
        with self.assertRaises(ValueError):audit.check_events(events,records)

    def test_state_coefficients_and_normal_certificate_JSON_serialization(self):
        x,mask,z,prior=past_case();metadata,state=state_fixture(x,mask,z,prior,.1)
        result=audit.check_state({'state_metadata':metadata},state,{'history':x,'history_observed':mask},z,prior,.1)
        self.assertEqual(result['positive_lambda_solver_calls_checked'],3)
        json.loads(json.dumps(result,allow_nan=False))
        state['decoder_deployed_FP32'][0,0,0]+=.01
        with self.assertRaisesRegex(ValueError,'BYTE'):audit.check_state({'state_metadata':metadata},state,{'history':x,'history_observed':mask},z,prior,.1)

    def test_positive_backend_saved_future_scores_and_FP32_rounding(self):
        with tempfile.TemporaryDirectory(prefix='shrunk_decoder_audit_fixture_') as directory:
            args=positive_backend_fixture(Path(directory))
            result=audit.audit_positive_backend(*args,lambda:None)
            self.assertEqual(result['score_field_checks'],36)
            self.assertEqual(result['methods']['BOLT']['scores_all_observed']['normalized']['mse'],.015625)
            json.loads(json.dumps(result,allow_nan=False))

    def test_noobs_backend_forecasts_not_silently_skipped(self):
        with tempfile.TemporaryDirectory(prefix='shrunk_decoder_audit_fixture_') as directory:
            args=positive_backend_fixture(Path(directory),noobs=True)
            result=audit.audit_positive_backend(*args,lambda:None)
            self.assertEqual(result['state_checks']['no_observation_prior_fallbacks_checked'],3)
            self.assertEqual(result['score_field_checks'],24);self.assertIsNone(result['past_reconstruction_scores_observed_only'])

    def test_future_target_change_leaves_past_state_but_rejects_stale_scores(self):
        with tempfile.TemporaryDirectory(prefix='shrunk_decoder_audit_fixture_') as directory:
            args=positive_backend_fixture(Path(directory));args[1]['target']=args[1]['target']+1
            with self.assertRaisesRegex(ValueError,'BYTE'):audit.audit_positive_backend(*args,lambda:None)

    def test_macro_channel_score_is_not_pooled_error(self):
        prediction=np.zeros((1,48,2),np.float32);target=np.zeros_like(prediction);mask=np.zeros_like(prediction,bool)
        target[:,:,0]=1;target[0,0,1]=3;mask[:,:,0]=True;mask[0,0,1]=True
        score=audit.prior.scores(prediction,target,mask,np.ones(2))
        self.assertEqual(score['normalized']['mse'],5.)
        self.assertNotEqual(score['normalized']['mse'],float(((prediction-target)[mask]**2).mean()))

    def test_actual_source_AST_fit_call_has_no_future_targets(self):
        names=('run_local_decoder_shrinkage_v1.py','local_decoder_shrinkage_v1.py','PLAN_LOCAL_DECODER_SHRINKAGE_KO_v1.md','local_past_decoder_v1.py','decoder_metric_probe_v1.py','fixed_decoder_span_v1.py','run_fit_target_comparison_v1.py','continuation_runtime_v1.py')
        self.assertTrue(audit.check_source_contract([{'path':str(audit.HERE/name)} for name in names])['past_only_fit_call_AST_checked'])


if __name__=='__main__':unittest.main()
