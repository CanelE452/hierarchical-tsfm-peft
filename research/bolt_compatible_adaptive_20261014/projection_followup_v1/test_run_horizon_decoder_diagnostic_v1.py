"""Synthetic-only stream, shared-scale, selection and complete controller fixtures."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import copy
import json
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import causal_lag_decoder_v1 as lag
import horizon_decoder_delta_v1 as module
import run_horizon_decoder_diagnostic_v1 as run


def fixture(n=7,c=3,k=2):
    rng=np.random.default_rng(522+n+c+k);history=rng.normal(scale=.1,size=(n,512,c)).astype(np.float32);current=history[:,-1].copy();forecast=current[:,None]+rng.normal(scale=.01,size=(n,48,c)).astype(np.float32)
    e=np.eye(c,dtype=np.float32)[:,:k].copy();state={"E":e,"D":rng.normal(scale=.02,size=(7*k,c)).astype(np.float32),"center":np.zeros(c,np.float32)};z=history@e;future=forecast@e;features=lag.lagged_delta_features(z,future,run.LAGS);flat=np.broadcast_to(z[:,-1:],future.shape).copy()
    cache={"latent_history":z,"latent_future":future,"lag_features":features,"prediction":lag.predict(z,future,current,state["D"],run.LAGS),"FLAT_prediction":lag.predict(z,flat,current,state["D"],run.LAGS)}
    target=current[:,None]+rng.normal(scale=.02,size=(n,48,c)).astype(np.float32);observed=np.ones_like(target,dtype=bool);observed[0,1,0]=False;target[~observed]=np.nan
    raw={"history":history,"raw_last":current,"raw_forecast":forecast,"target":target,"observed":observed,"origins":np.arange(n,dtype=np.int64)+512,"mean":np.zeros(c,np.float32),"std":np.ones(c,np.float32),"history_observed":np.ones_like(history,dtype=bool)}
    return raw,state,{**cache,"origins":raw["origins"].copy()}


class HorizonDecoderRunnerTests(unittest.TestCase):
    def test_PCA_fixed_stream_lags_flat_plateau_and_no_target_forward(self):
        raw,state,cache=fixture();original={k:v.tobytes() for k,v in state.items()};streams={mode:run.stream_payload(raw["history"],raw["raw_last"],raw["raw_forecast"],"PCA_K2",mode,cache,state) for mode in run.MODES}
        for mode in run.MODES:
            self.assertTrue(all(v["pass"] for v in streams[mode]["checks"].values()));self.assertEqual(streams[mode]["features"].dtype,np.float32)
        self.assertEqual(streams['F']['baseline'].tobytes(),cache['prediction'].tobytes());self.assertEqual(streams['FLAT']['baseline'].tobytes(),cache['FLAT_prediction'].tobytes())
        self.assertFalse(np.all(streams['FLAT']['baseline']==raw['raw_last'][:,None]));np.testing.assert_array_equal(streams['FLAT']['features'][:,33],streams['FLAT']['features'][:,47])
        changed=copy.deepcopy(cache);changed['prediction'][0,0,0]+=.1
        with self.assertRaisesRegex(ValueError,'BASELINE_AND_LAG'):run.stream_payload(raw['history'],raw['raw_last'],raw['raw_forecast'],'PCA_K2','F',changed,state)
        self.assertEqual(original,{k:v.tobytes() for k,v in state.items()})

    def test_RAW_baseline_bytes_do_not_depend_on_FP32_identity_reassociation(self):
        history=np.full((2,512,2),np.float32(1e8));current=history[:,-1].copy();future=np.ones((2,48,2),np.float32)
        stream=run.stream_payload(history,current,future,'RAW_C','F');self.assertEqual(stream['baseline'].tobytes(),future.tobytes());self.assertFalse(stream['checks']['same_old_decoder']['pass']);self.assertFalse(stream['checks']['same_old_decoder']['required'])
        flat=run.stream_payload(history,current,future,'RAW_C','FLAT');self.assertEqual(flat['baseline'].tobytes(),np.broadcast_to(current[:,None],future.shape).copy().tobytes())

    def test_four_arms_share_actualF_ungated_RMS_and_pack_arrays_without_hidden_intercept(self):
        raw,state,cache=fixture();streams={m:run.stream_payload(raw['history'],raw['raw_last'],raw['raw_forecast'],'PCA_K2',m,cache,state) for m in run.MODES};scale,constant=module.feature_scale(streams['F']['features'])
        for mode in run.MODES:
            for arm in run.ARMS:
                with self.subTest(mode=mode,arm=arm):
                    stream=streams[mode];response=run.fit_response(raw['target'],raw['observed'],stream['baseline']);states,meta=module.fit_path(module.gated_features(stream['features'],arm),response,raw['observed'],scale,constant);summary,arrays=run.fit_info(meta);json.dumps(summary,allow_nan=False)
                    self.assertEqual(arrays['scale_FP64'].tobytes(),scale.tobytes());self.assertFalse(summary['intercept_fitted']);self.assertFalse(summary['feature_centering']);self.assertEqual(len(states),10)
                    prediction,parity=run.predict_candidate(stream['baseline'],stream['features'],states[4],arm,'ALPHA');self.assertTrue(parity['pass']);self.assertEqual(prediction.dtype,np.float32)
                    if arm=='RAMP_DELTA':self.assertEqual(prediction[:,0].tobytes(),stream['baseline'][:,0].tobytes())
        self.assertNotEqual(module.feature_scale(module.gated_features(streams['F']['features'],'RAMP_DELTA'))[0].tobytes(),scale.tobytes())

    def test_zero_copy_nan_placeholder_exclusion_and_registered_alpha_ties(self):
        raw,state,cache=fixture();baseline=cache['prediction'];response=run.fit_response(raw['target'],raw['observed'],baseline);changed=raw['target'].copy();changed[~raw['observed']]=np.inf
        np.testing.assert_array_equal(response,run.fit_response(changed,raw['observed'],baseline));self.assertTrue(np.isfinite(response).all())
        zero={'coefficient_FP64':np.zeros((14,3)),'coefficient_FP32':np.zeros((14,3),np.float32)}
        prediction,check=run.predict_candidate(baseline,cache['lag_features'],zero,'RAMP_DELTA','ZERO_DELTA');self.assertEqual(prediction.tobytes(),baseline.tobytes());self.assertIsNot(prediction,baseline);self.assertTrue(check['ZERO_original_baseline_byte_exact'])
        rows=[{'id':'ALPHA_'+format(a,'g'),'alpha':a,'status':'COMPLETE','TUNE_MSE':1.} for a in run.ALPHAS]+[{'id':'ZERO_DELTA','alpha':None,'status':'COMPLETE','TUNE_MSE':1.}]
        self.assertEqual(run.select_candidate(rows)['id'],'ALPHA_0');rows[-1]['TUNE_MSE']=.9;self.assertEqual(run.select_candidate(rows)['id'],'ZERO_DELTA')

    def test_24_lock_registry_before_DEV_with_one_failed_condition_isolation(self):
        result={'datasets':{d:{'spaces':{s:{'conditions':{}} for dd,s,_,_ in run.CONDITIONS if dd==d}} for d in run.DATASETS},'failed_paths':[],'selected_DEV_evaluations_completed':0};events=[]
        def fit(data,space,mode,arm):
            if (data,space,mode,arm)==run.CONDITIONS[1]:raise RuntimeError('synthetic failed solve')
            return {'status':'COMPLETE'}
        def seal(*args):events.append(('lock',args[:4]));return {'path':'lock'}
        def registry(rows):events.append(('registry',len(rows)));return {'path':'registry'}
        resolved=run.resolve_paths(result,fit,seal,registry);self.assertEqual(len(resolved),23);self.assertEqual(len(result['failed_paths']),1)
        def dev(*args):self.assertEqual(sum(x[0]=='lock' for x in events),24);self.assertEqual(events[-1],('registry',24));return {'status':'COMPLETE'}
        run.infer_development(result,resolved,dev);self.assertEqual(result['selected_DEV_evaluations_completed'],23)
        with self.assertRaisesRegex(ValueError,'24_LOCKS'):run.infer_development({'all_24_locks_before_DEV':False},resolved,dev)

    def test_latent_complete_C_and_horizon_macro_diagnostics_have_explicit_scope(self):
        raw,state,cache=fixture();streams={m:run.stream_payload(raw['history'],raw['raw_last'],raw['raw_forecast'],'PCA_K2',m,cache,state) for m in run.MODES};value=run.latent_errors(streams,raw['target'],raw['observed'],state)
        complete=raw['observed'].all(2);truth=(raw['target'][complete]-state['center'])@state['E'];expected=streams['F']['latent_future'][complete].astype(np.float64)-truth.astype(np.float64)
        self.assertAlmostEqual(value['F']['mse'],float(np.mean(expected**2)));self.assertFalse(value['incomplete_C_vectors_encoded']);self.assertEqual(value['C_lag_output_EY_or_floor_status'],'NOT_MEASURED_IN_THIS_STAGE')
        temporal=run.temporal_scores(cache['prediction'],raw);self.assertEqual(len(temporal['per_horizon']),48);self.assertEqual(temporal['per_horizon'][1]['observed_counts'],[6,7,7]);self.assertTrue(temporal['horizon_averages_not_equal_full_observed_channel_macro'])

    def test_full_toy_controller_24_paths_528_predictions_selection_archives_and_zero_baselines(self):
        import fixed_decoder_span_v1 as io
        import continuation_runtime_v1 as rt
        sys.path.insert(0,str(run.OLD))
        import prepare_inputs
        import run_fit_target_comparison_v1 as inputs
        class Meter:
            def __init__(self,*args,**kwargs):pass
            def check(self,*args,**kwargs):pass
            def protected_check(self):pass
            def close(self,status,failure):return {'status':status,'synthetic_only':True}
        def fake_fit(features,response,observed,scale,constant):
            shape=(features.shape[2],response.shape[2]);states=[{'alpha':a,'coefficient_FP64':np.zeros(shape),'coefficient_FP32':np.zeros(shape,np.float32)} for a in module.ALPHAS]
            return states,{'groups':[],'scale_FP64':scale.copy(),'constant_features':constant.copy(),'alpha_grid':list(module.ALPHAS),'synthetic_no_fit':True}
        with tempfile.TemporaryDirectory(prefix='synthetic_horizon_',dir=run.ROOT/'.cache') as directory:
            folder=Path(directory);output=folder/'run';cross={'datasets':{},'F_before':{'synthetic':True},'F_after':{'synthetic':True}};science={'datasets':{}};datasets={'toy_a':(2,1,'val',3),'toy_b':(2,1,'eval',4)}
            for data,(c,k,dev,dn) in datasets.items():
                cross['datasets'][data]={'conditions':{}};science['datasets'][data]={'spaces':{'PCA':{'conditions':{'L7_RECON':{'splits':{}}}}}}
                for split,n in (('fit',256),('tune',64),(dev,dn)):
                    raw,_,_=fixture(n=n,c=c,k=k);rawrecord=io.save_npz(folder/(data+'_'+split+'_input.npz'),raw);science['datasets'][data]['spaces']['PCA']['conditions']['L7_RECON']['splits'][split]={'input_source':rawrecord}
                    for width in (k,2*k):
                        _,state,cache=fixture(n=n,c=c,k=width);name='PCA_K'+str(width);owner=cross['datasets'][data]['conditions'].setdefault(name,{'K':width,'splits':{}})
                        # All spaces must use the exact same C history/forecast and origins.
                        z=raw['history']@state['E'];future=raw['raw_forecast']@state['E'];cache={'latent_history':z,'latent_future':future,'lag_features':lag.lagged_delta_features(z,future,run.LAGS),'prediction':lag.predict(z,future,raw['raw_last'],state['D'],run.LAGS),'FLAT_prediction':lag.predict(z,np.broadcast_to(z[:,-1:],future.shape).copy(),raw['raw_last'],state['D'],run.LAGS),'origins':raw['origins']}
                        if 'state_artifact' not in owner:owner['state_artifact']=io.save_npz(folder/(data+'_'+name+'_state.npz'),state)
                        owner['splits'][split]={'input_source':rawrecord,'artifact':io.save_npz(folder/(data+'_'+name+'_'+split+'.npz'),cache)}
            # Make D identical across split caches, as in an accepted FIT-only state.
            for data in datasets:
                for owner in cross['datasets'][data]['conditions'].values():
                    with np.load(owner['state_artifact']['path'],allow_pickle=False) as values:d=values['D'].copy()
                    for split,entry in owner['splits'].items():
                        rawrecord=entry['input_source']
                        with np.load(rawrecord['path'],allow_pickle=False) as values:raw={k:values[k].copy() for k in values.files}
                        with np.load(entry['artifact']['path'],allow_pickle=False) as values:cache={k:values[k].copy() for k in values.files}
                        cache['prediction']=lag.predict(cache['latent_history'],cache['latent_future'],raw['raw_last'],d,run.LAGS);cache['FLAT_prediction']=lag.predict(cache['latent_history'],np.broadcast_to(cache['latent_history'][:,-1:],cache['latent_future'].shape).copy(),raw['raw_last'],d,run.LAGS)
                        entry['artifact']=io.save_npz(folder/(data+'_correct_'+str(owner['K'])+'_'+split+'.npz'),cache)
            conditions=tuple((d,s,m,a) for d,(_,k,_,_) in datasets.items() for s in ('RAW_C','PCA_K1','PCA_K2') for m in run.MODES for a in run.ARMS)
            with patch.object(run,'load_gates',return_value=(cross,science,{})),patch.object(run.old,'fresh_output',return_value=output),patch.object(run,'DATASETS',datasets),patch.object(run,'CONDITIONS',conditions),patch.object(rt,'Meter',Meter),patch.object(inputs,'prior_file_records',return_value=[]),patch.object(prepare_inputs,'install_protected_test_guard'),patch.object(module,'fit_path',side_effect=fake_fit):
                status=run.execute(SimpleNamespace(tag='synthetic'));detail=json.loads((output/'RESULT.json').read_text(encoding='utf-8'));self.assertEqual(status,0,detail.get('failed_paths',detail.get('failure')))
            result=json.loads((output/'RESULT.json').read_text(encoding='utf-8'));self.assertEqual(result['status'],'COMPLETE');self.assertEqual([result[k] for k in ('fits_completed','alpha_candidates_completed','ZERO_candidates_completed','candidate_FIT_TUNE_evaluations_completed','selected_DEV_evaluations_completed','resolved_selection_locks')],[24,240,24,528,24,24]);self.assertFalse(result['new_backbone_training']);self.assertEqual(result['new_F_forward_calls'],0)
            registry=json.loads(Path(result['selection_registry']['path']).read_text(encoding='utf-8'));self.assertEqual(len(registry['conditions']),24)
            for item in registry['conditions']:
                row=result['datasets'][item['dataset']]['spaces'][item['space']]['conditions'][item['mode']+'_'+item['arm']];self.assertEqual(row['selected_candidate'],'ALPHA_0');self.assertEqual(len(row['candidates']),11)
                self.assertGreaterEqual(Path(row['DEV']['artifact']['path']).stat().st_mtime,registry['sealed_unix']);self.assertEqual(set(row['selected_coefficients']['arrays']),{'coefficient_FP64','coefficient_FP32'})

    def test_closed_source_gate_precedes_array_import_and_io(self):
        with patch.object(run,'MODULE_SHA',None),patch.object(run.old,'bound',side_effect=AssertionError('forbidden IO')):
            with self.assertRaisesRegex(ValueError,'NOT_REGISTERED'):run.load_gates(SimpleNamespace())
        code='import sys;sys.path.insert(0,'+repr(str(run.HERE))+');import run_horizon_decoder_diagnostic_v1;assert "numpy" not in sys.modules and "torch" not in sys.modules;print("STDLIB_HORIZON_IMPORT_PASS")'
        completed=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,check=True);self.assertIn('STDLIB_HORIZON_IMPORT_PASS',completed.stdout)


if __name__=='__main__':unittest.main()
