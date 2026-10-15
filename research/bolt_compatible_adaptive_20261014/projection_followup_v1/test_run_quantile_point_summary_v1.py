"""Synthetic CPU full-quantile model, lag decoding, selection, and closed metadata gates."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import copy
import inspect
import json
import subprocess
import sys
import unittest

import numpy as np
import torch

import run_quantile_point_summary_v1 as run
from linear_e_adaptation_v1 import LinearLagAdapter


class ToyQuantileModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.marker=torch.nn.Parameter(torch.zeros(()),requires_grad=False);self.seen=[]
    def forward(self,*,context):
        self.seen.append(context.detach().clone())
        future=context[:,-1:]+.1*context.mean(1,keepdim=True)+context.new_tensor(range(64))[None]*.002
        offsets=context.new_tensor([-.5,-.4,-.3,-.2,0.,.15,.3,.7,1.4])
        return SimpleNamespace(quantile_preds=future[:,None,:]+offsets[None,:,None])


def fixture(raw_route=False):
    rng=np.random.default_rng(765);history=rng.normal(scale=.3,size=(3,512,6)).astype(np.float32)
    frozen={"center":rng.normal(scale=.02,size=6).astype(np.float32),"E":rng.normal(scale=.1,size=(6,4)).astype(np.float32),"D":rng.normal(scale=.05,size=(28,6)).astype(np.float32)}
    model=None if raw_route else LinearLagAdapter(frozen["center"],frozen["E"],frozen["D"],run.LAGS,learn_E=False).eval().requires_grad_(False)
    bolt=SimpleNamespace(model=ToyQuantileModel().eval());x=torch.from_numpy(history)
    with torch.no_grad():
        past=x if model is None else model.encode(x);n,length,k=past.shape
        q=bolt.model(context=past.transpose(1,2).reshape(n*k,length)).quantile_preds
        future=q[:,4,:48].reshape(n,k,48).transpose(1,2).contiguous()
        prediction=future if model is None else model.decode(x[:,-1],past,future)
    raw={"history":history,"raw_last":history[:,-1].copy(),"target":rng.normal(size=(3,48,6)).astype(np.float32),"observed":np.ones((3,48,6),dtype=bool),"std":np.arange(1,7,dtype=np.float32),"mean":np.arange(6,dtype=np.float32),"origins":np.arange(3,dtype=np.int64)+512}
    raw["observed"][0,0,0]=False;raw["target"][~raw["observed"]]=np.nan
    raw["raw_forecast"]=prediction.numpy().copy() if raw_route else prediction.numpy().copy()+np.float32(.2)
    cached={"prediction":prediction.numpy().copy(),"latent_history":past.numpy().copy(),"latent_future":future.numpy().copy()};bolt.model.seen.clear()
    return model,bolt,raw,cached,frozen


def empty_result():return {"routes":{},"failed_paths":[],"resolved_selection_locks":0,"all_4_locks_before_DEV":False,"DEV_diagnostic_summary_evaluations_completed":0,"selected_DEV_evaluations_completed":0}


def synthetic_route():
    return {"status":"COMPLETE","splits":{split:{"status":"COMPLETE","summaries":{summary:{"status":"COMPLETE","scores_all_observed":{"normalized":{"mse":.2}}} for summary in run.SUMMARIES}} for split in ("fit","tune")}}


class QuantileSummaryRunnerTests(unittest.TestCase):
    def test_raw_and_latent_full64_layout_actual_rows_independent_decode_and_median_cache(self):
        for raw_route in (False,True):
            with self.subTest(raw_route=raw_route):
                model,bolt,raw,cached,frozen=fixture(raw_route);events=[]
                outputs,calls=run.infer_full_quantiles(model,bolt,raw["history"],events.append,batch_size=2)
                row,independent=run.evaluate_arrays(outputs,raw,cached,frozen,"RAW_C" if raw_route else "RECON_CP0")
                self.assertEqual(row['status'],'COMPLETE');self.assertEqual(calls,2);self.assertEqual(len(events),2)
                self.assertEqual(outputs['full_quantiles_FP32'].shape,(3,64,6 if raw_route else 4,9))
                for start,seen in zip((0,2),bolt.model.seen):
                    past=torch.from_numpy(outputs['latent_history'][start:start+2]);self.assertTrue(torch.equal(seen,past.transpose(1,2).reshape(-1,512)))
                for key in ('median_future_vs_accepted_cache','median_C_vs_accepted_cache'):self.assertTrue(row['checks'][key]['pass'])
                self.assertEqual(set(independent),{s+suffix for s in run.SUMMARIES for suffix in ('_NumPy_future','_NumPy_prediction')})
                self.assertGreater(float(np.max(np.abs(outputs['NINE_MEAN_PROXY_future']-outputs['MEDIAN_future']))),.01)
                self.assertFalse(row['pipeline_mean_used']);self.assertFalse(row['true_conditional_mean_or_C_quantile_claimed'])
                json.dumps(row,allow_nan=False)

    def test_behind_horizon_48_stored_but_never_changes_evaluated_summary(self):
        model,bolt,raw,cached,frozen=fixture();outputs,_=run.infer_full_quantiles(model,bolt,raw['history'],lambda value:None)
        original,_=run.evaluate_arrays(outputs,raw,cached,frozen,'RECON_CP0');outputs['full_quantiles_FP32'][:,48:]+=100
        changed,_=run.evaluate_arrays(outputs,raw,cached,frozen,'RECON_CP0')
        self.assertEqual(original['summaries'],changed['summaries']);self.assertEqual(original['evaluated48_quantile_diagnostics'],changed['evaluated48_quantile_diagnostics']);self.assertNotEqual(original['full64_quantile_diagnostics']['maximum'],changed['full64_quantile_diagnostics']['maximum'])

    def test_wrong_median_baseline_or_one_summary_decode_fails_preserved_checks(self):
        model,bolt,raw,cached,frozen=fixture();outputs,_=run.infer_full_quantiles(model,bolt,raw['history'],lambda value:None)
        wrong=copy.deepcopy(cached);wrong['prediction'][0,0,0]+=.1
        row,_=run.evaluate_arrays(outputs,raw,wrong,frozen,'RECON_CP0');self.assertEqual(row['status'],'FAILED_PARITY_PRESERVED');self.assertFalse(row['checks']['median_C_vs_accepted_cache']['pass'])
        outputs['TAIL_CLAMPED_PROXY_prediction'][0,0,0]+=.1
        row,_=run.evaluate_arrays(outputs,raw,cached,frozen,'RECON_CP0');self.assertEqual(row['summaries']['TAIL_CLAMPED_PROXY']['status'],'FAILED_PARITY_PRESERVED');self.assertEqual(row['summaries']['MEDIAN']['status'],'COMPLETE')

    def test_labels_never_enter_forward_and_unobserved_nan_or_inf_are_ignored(self):
        self.assertEqual(tuple(inspect.signature(run.infer_full_quantiles).parameters),('model','bolt','history','check','batch_size'))
        model,bolt,raw,cached,frozen=fixture();outputs,_=run.infer_full_quantiles(model,bolt,raw['history'],lambda value:None)
        row,_=run.evaluate_arrays(outputs,raw,cached,frozen,'RECON_CP0');changed=copy.deepcopy(raw);changed['target'][~changed['observed']]=np.inf
        other,_=run.evaluate_arrays(outputs,changed,cached,frozen,'RECON_CP0');self.assertEqual(row,other)
        changed['target'][changed['observed']]+=1
        other,_=run.evaluate_arrays(outputs,changed,cached,frozen,'RECON_CP0');self.assertEqual(row['checks'],other['checks']);self.assertNotEqual(row['summaries']['MEDIAN']['scores_all_observed'],other['summaries']['MEDIAN']['scores_all_observed'])
        repeat,_=run.infer_full_quantiles(model,bolt,raw['history'],lambda value:None)
        for key in outputs:self.assertEqual(outputs[key].tobytes(),repeat[key].tobytes())

    def test_four_actual_menus_exact_ties_raw_control_and_all_locks_before_any_DEV(self):
        result=empty_result();events=[]
        def execute(route):
            row=synthetic_route()
            if route=='E_ONLY_s92601':row['splits']['tune']['summaries']['NINE_MEAN_PROXY']['scores_all_observed']['normalized']['mse']=.1
            return row
        def seal(route,row):events.append(('lock',route));return {'path':route,'sha256':'a'*64}
        def registry(rows):events.append(('registry',len(rows)));return {'path':'registry','sha256':'b'*64}
        resolved=run.resolve_routes(result,execute,seal,registry);self.assertEqual(len(resolved),4)
        self.assertEqual(result['routes']['RAW_C']['selected_summary'],'MEDIAN');self.assertEqual(result['routes']['E_ONLY_s92601']['selected_summary'],'NINE_MEAN_PROXY')
        def develop(route,row):
            self.assertEqual(sum(p[0]=='lock' for p in events),4);self.assertEqual(events[-1],('registry',4));return {'status':'COMPLETE','summaries':{s:{'status':'COMPLETE'} for s in run.SUMMARIES}}
        run.infer_development(result,resolved,develop)
        self.assertEqual(result['DEV_diagnostic_summary_evaluations_completed'],12);self.assertEqual(result['selected_DEV_evaluations_completed'],4)
        for row in resolved.values():self.assertEqual(sum(p['selected'] for p in row['splits']['val']['summaries'].values()),1)

    def test_failure_stays_on_its_route_no_stale_row_and_other_paths_continue(self):
        result=empty_result();events=[]
        def execute(route):
            if route==run.ROUTES[1]:raise RuntimeError('synthetic route failure')
            return synthetic_route()
        resolved=run.resolve_routes(result,execute,lambda route,row:{'path':route},lambda rows:{'path':'registry'})
        self.assertEqual(len(resolved),3);self.assertNotIn('splits',result['routes'][run.ROUTES[1]])
        run.infer_development(result,resolved,lambda route,row:{'status':'COMPLETE','summaries':{s:{} for s in run.SUMMARIES}})
        self.assertEqual(result['selected_DEV_evaluations_completed'],3);self.assertEqual(len(result['failed_paths']),1)
        with self.assertRaisesRegex(ValueError,'FOUR'):run.infer_development(empty_result(),{},lambda *args:self.fail('DEV touched'))

    def test_wrong_full_quantile_shape_is_not_a_pipeline_or_mean_fallback(self):
        model,bolt,raw,_,_=fixture()
        class Wrong(ToyQuantileModel):
            def forward(self,*,context):return SimpleNamespace(quantile_preds=torch.zeros((len(context),8,64),dtype=torch.float32))
        bolt.model=Wrong()
        with self.assertRaisesRegex(ValueError,'NINE_64'):run.infer_full_quantiles(model,bolt,raw['history'],lambda value:None)

    def test_frozen_state_no_grad_and_closed_source_gate_before_any_IO(self):
        model,bolt,raw,_,_=fixture();before={k:v.detach().clone() for k,v in model.state_dict().items()}
        run.infer_full_quantiles(model,bolt,raw['history'],lambda value:None)
        for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,before[k]))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in (*model.parameters(),*bolt.model.parameters())))
        with patch.object(run,'MODULE_SHA',None),patch.object(run.old,'bound',side_effect=AssertionError('IO forbidden')):
            with self.assertRaisesRegex(ValueError,'NOT_REGISTERED'):run.load_gates(SimpleNamespace())
        code='import sys;sys.path.insert(0,'+repr(str(run.HERE))+');import run_quantile_point_summary_v1;assert "numpy" not in sys.modules and "torch" not in sys.modules;print("STDLIB_GATE_IMPORT_PASS")'
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,check=True);self.assertIn('STDLIB_GATE_IMPORT_PASS',result.stdout)


if __name__=='__main__':unittest.main()
