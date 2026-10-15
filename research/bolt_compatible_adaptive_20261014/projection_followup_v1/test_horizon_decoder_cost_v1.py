"""Synthetic CPU/Torch callbacks and timer plumbing; no actual model, data or GPU."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
import causal_lag_decoder_v1 as lag
import horizon_decoder_delta_v1 as delta
from linear_e_adaptation_v1 import LinearLagAdapter
import run_horizon_decoder_diagnostic_v1 as science
import run_horizon_decoder_cost_v1 as run
import whole_horizon_decoder_cost_v1 as wc
import whole_f_input_gradient_cost_v1 as timer


def fixture():
    rng=np.random.default_rng(4482);history=rng.normal(scale=.1,size=(9,512,3)).astype(np.float32);e=np.eye(3,dtype=np.float32)[:,:2];d=rng.normal(scale=.02,size=(14,3)).astype(np.float32);center=np.zeros(3,np.float32)
    model=LinearLagAdapter(center,e,d,run.LAGS,learn_E=False).eval().requires_grad_(False);state={"E":e,"D":d,"center":center};coef64=rng.normal(scale=.001,size=(14,3));coefficient={"coefficient_FP64":coef64,"coefficient_FP32":coef64.astype(np.float32)}
    calls=[]
    def tensor(x):return torch.as_tensor(x,dtype=torch.float32)
    def bolt(z):calls.append(tuple(z.shape));return z[:,-1:].expand(-1,48,-1)+.01
    return history,model,state,coefficient,tensor,bolt,calls


class HorizonCostTests(unittest.TestCase):
    def test_routes_18_and_five_distinct_positions_not_full_balance(self):
        for data,k in wc.DATASETS.items():
            routes=wc.routes_for(data);schedule=wc.balanced_schedule(data);self.assertEqual(len(routes),18);self.assertEqual(len(set(routes)),18);self.assertEqual(len(schedule),5)
            self.assertIn('PCA_K'+str(k)+'_FLAT_RAMP_DELTA',routes);self.assertIn('PCA_K'+str(k*2)+'_F_STATIC_DELTA',routes)
            for order in schedule:self.assertEqual(set(order),set(routes))
            for route in routes:self.assertEqual(len({order.index(route) for order in schedule}),5)
            self.assertEqual(wc.matched_baseline('PCA_K'+str(k)+'_FLAT_RAMP_DELTA'),'PCA_K'+str(k)+'_FLAT_BASELINE')
        self.assertIs(wc.whole_predict,timer.whole_predict);self.assertIs(wc.preprocess,timer.preprocess)
        old=(wc.HERE/'whole_new_direction_cost_v1.py').read_text(encoding='utf-8');new=Path(wc.__file__).read_text(encoding='utf-8');start='synchronize(); result["attempted_timer_calls"] += 1';end='seconds = (perf_end_ns-perf_start_ns)/1e9'
        self.assertEqual(old.split(start)[1].split(end)[0],new.split(start)[1].split(end)[0])

    def test_raw_F_direct_output_and_FLAT_do_not_reassociate_or_call_encoder(self):
        history=np.full((2,512,2),np.float32(1e8));calls=[]
        def tensor(x):return torch.as_tensor(x)
        def bolt(x):calls.append(1);return torch.ones((len(x),48,2),dtype=torch.float32)
        p=run.forecast_components(history,'RAW_C','F','BASELINE',None,bolt,tensor);self.assertEqual(p.tobytes(),np.ones_like(p).tobytes());self.assertEqual(len(calls),1)
        def forbidden(*args):raise AssertionError('F/E must not run for RawFLAT baseline')
        p=run.forecast_components(history,'RAW_C','FLAT','BASELINE',None,forbidden,forbidden);self.assertEqual(p.tobytes(),np.broadcast_to(history[:,-1:],p.shape).copy().tobytes())

    def test_live_PCA_F_and_FLAT_baselines_and_active_CPU_correction(self):
        history,model,state,coefficient,tensor,bolt,calls=fixture();original={k:v.detach().numpy().tobytes() for k,v in model.state_dict().items()}
        for mode in ('F','FLAT'):
            for arm in ('STATIC_DELTA','RAMP_DELTA'):
                with self.subTest(mode=mode,arm=arm):
                    before=len(calls);out=run.forecast_components(history[:8],'PCA_K2',mode,arm,model,bolt,tensor,coefficient,'ALPHA',capture=True);self.assertEqual(len(calls)-before,int(mode=='F'))
                    expected=delta.predict(out['baseline'],out['features'],coefficient,arm);self.assertEqual(out['prediction'].tobytes(),expected.tobytes());self.assertEqual(out['features'].dtype,np.float32)
                    z=(history[:8]-state['center'])@state['E'];np.testing.assert_array_equal(out['latent_history'],z)
                    if arm=='RAMP_DELTA':self.assertEqual(out['prediction'][:,0].tobytes(),out['baseline'][:,0].tobytes())
        self.assertEqual(original,{k:v.detach().numpy().tobytes() for k,v in model.state_dict().items()})
        def forbidden(*args):raise AssertionError('PCA FLAT may not call F')
        p=run.forecast_components(history[:1],'PCA_K2','FLAT','BASELINE',model,forbidden,tensor);self.assertFalse(np.all(p==history[:1,-1:]))

    def test_ZERO_skips_correction_and_extra_host_lag_features_but_keeps_live_F(self):
        history,model,state,coefficient,tensor,bolt,calls=fixture()
        baseline=run.forecast_components(history[:1],'PCA_K2','F','BASELINE',model,bolt,tensor)
        with patch.object(delta,'predict',side_effect=AssertionError('ZERO cannot matmul')),patch.object(lag,'lagged_delta_features',side_effect=AssertionError('ZERO must skip extra CPU lag features')):
            before=len(calls);zero=run.forecast_components(history[:1],'PCA_K2','F','RAMP_DELTA',model,bolt,tensor,coefficient,'ZERO_DELTA');self.assertEqual(len(calls)-before,1);self.assertEqual(zero.tobytes(),baseline.tobytes())

    def test_active_raw_correction_uses_raw_past_and_actual_future(self):
        history,_,_,_,tensor,bolt,calls=fixture();coef64=np.arange(63,dtype=np.float64).reshape(21,3)*1e-5;coefficient={'coefficient_FP64':coef64,'coefficient_FP32':coef64.astype(np.float32)}
        for mode in ('F','FLAT'):
            before=len(calls);out=run.forecast_components(history[:8],'RAW_C',mode,'STATIC_DELTA',None,bolt,tensor,coefficient,'ALPHA',capture=True);self.assertEqual(len(calls)-before,int(mode=='F'))
            np.testing.assert_array_equal(out['features'],lag.lagged_delta_features(history[:8],out['latent_future'],run.LAGS));self.assertEqual(out['baseline'].tobytes(),out['latent_future'].tobytes())
            self.assertEqual(out['prediction'].tobytes(),delta.predict(out['baseline'],out['features'],coefficient,'STATIC_DELTA').tobytes())

    def test_physical_precheck_reference_normalized_physical_and_intermediates(self):
        history,model,state,coefficient,tensor,bolt,calls=fixture();n=len(history);origins=np.arange(n,dtype=np.int64)+512;observed=np.ones_like(history,dtype=bool);future=(history[:,-1:]+.01).repeat(48,axis=1)
        raw={'history':history,'raw_last':history[:,-1].copy(),'raw_forecast':future,'history_observed':observed,'origins':origins,'mean':np.zeros(3,np.float32),'std':np.ones(3,np.float32)};physical={**raw,'history_physical':history.copy()}
        z=history@state['E'];f=future@state['E'];cache={'latent_history':z,'latent_future':f,'lag_features':lag.lagged_delta_features(z,f,run.LAGS),'prediction':lag.predict(z,f,raw['raw_last'],state['D'],run.LAGS),'FLAT_prediction':lag.predict(z,np.broadcast_to(z[:,-1:],f.shape).copy(),raw['raw_last'],state['D'],run.LAGS)}
        stream=science.stream_payload(history,raw['raw_last'],future,'PCA_K2','F',cache,state);reference=delta.predict(stream['baseline'],stream['features'],coefficient,'STATIC_DELTA');check,arrays=run.physical_precheck(physical,raw,reference,'PCA_K2','F','STATIC_DELTA',model,bolt,tensor,coefficient,'ALPHA',stream)
        self.assertTrue(check['input_backend_pass']);self.assertTrue(check['prediction_backend_pass']);self.assertTrue(check['capture_diagnostics_and_extra_host_transfers_outside_timer']);self.assertEqual(arrays['prediction'].shape,(8,48,3))
        changed=reference.copy();changed[0,0,0]+=.1;bad,_=run.physical_precheck(physical,raw,changed,'PCA_K2','F','STATIC_DELTA',model,bolt,tensor,coefficient,'ALPHA',stream);self.assertFalse(bad['prediction_backend_pass'])

    def test_full_synthetic_timer_36_cells_180_samples_two_denominators_and_partial_preservation(self):
        physical=np.arange(8*512*2,dtype=np.float32).reshape(8,512,2)*1e-5;observed=np.ones_like(physical,dtype=bool);mean=np.zeros(2,np.float32);std=np.ones(2,np.float32);origins=np.arange(8,dtype=np.int64)+512
        routes=wc.routes_for('robin');predictors={route:(lambda x,m:np.broadcast_to(x[:,-1:],(len(x),48,2)).copy()) for route in routes};reference={r:predictors[r](physical,observed) for r in routes};metadata={r:{'synthetic_only':True} for r in routes}
        with tempfile.TemporaryDirectory() as directory:
            result=wc.measure_cost('robin',physical,observed,mean,std,origins,predictors,reference,metadata,Path(directory)/'complete',lambda:None,lambda:None,lambda:{'scope':'synthetic shared'})
            self.assertEqual((result['status'],result['completed_cost_cells'],result['recorded_timer_samples']),('COMPLETE',36,180));self.assertIn('not complete',result['schedule_scope'])
            for row in result['rows']:
                self.assertEqual(len(row['samples']),5);self.assertEqual(row['matched_baseline_route'],wc.matched_baseline(row['route']));self.assertEqual(row['repeats'],5);self.assertFalse(row['native_head_loaded_or_timed'])
                for sample in row['samples']:self.assertEqual(sample['seconds'],(sample['perf_end_ns']-sample['perf_start_ns'])/1e9);self.assertTrue(all(p['pass'] for p in sample['parity'].values()))
                if row['route']=='RAW_C_F_BASELINE':self.assertEqual(row['paired_RAW_median_ratio'],1.)
                if row['route'].endswith('_BASELINE'):self.assertEqual(row['paired_matched_baseline_median_ratio'],1.)
            unavailable=routes[-1];partial=wc.measure_cost('robin',physical,observed,mean,std,origins,{k:v for k,v in predictors.items() if k!=unavailable},{k:v for k,v in reference.items() if k!=unavailable},metadata,Path(directory)/'partial',lambda:None,lambda:None,lambda:{},unavailable_routes={unavailable:'synthetic preflight failure'})
            self.assertEqual(partial['status'],'PARTIAL_COMPLETED');self.assertEqual(partial['completed_cost_cells'],34);self.assertEqual(partial['recorded_timer_samples'],170);self.assertEqual(len(partial['rows']),36);self.assertTrue(any(p['route']==unavailable for p in partial['failed_paths']))

    def test_closed_audit_gate_before_scientific_IO_and_fresh_import_stdlib_only(self):
        with patch.object(run,'AUDIT_SHA',None),patch.object(run,'record',side_effect=AssertionError('forbidden IO')):
            with self.assertRaisesRegex(ValueError,'NOT_REGISTERED'):run.load_gates('missing','missing')
        code='import sys;sys.path.insert(0,'+repr(str(run.HERE))+');import run_horizon_decoder_cost_v1;assert "numpy" not in sys.modules and "torch" not in sys.modules;print("STDLIB_COST_IMPORT_PASS")'
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,check=True);self.assertIn('STDLIB_COST_IMPORT_PASS',result.stdout)


if __name__=='__main__':unittest.main()
