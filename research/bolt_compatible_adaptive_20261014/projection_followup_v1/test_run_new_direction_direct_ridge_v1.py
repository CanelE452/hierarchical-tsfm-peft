"""Small synthetic CPU ridge paths and metadata locks; no actual cohort/model/fit replay."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import copy
import inspect
import json
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import masked_direct_ridge_v1 as ridge
import run_new_direction_direct_ridge_v1 as run


def fixture():
    rng=np.random.default_rng(92911);history=rng.normal(scale=.2,size=(12,512,3)).astype(np.float32)
    target=np.broadcast_to(history[:,-1:],(12,48,3)).copy()+rng.normal(scale=.03,size=(12,48,3)).astype(np.float32)
    observed=np.ones_like(target,dtype=bool);observed[0,2,0]=False;observed[1,7,1:]=False;target[~observed]=np.nan
    raw={"history":history,"raw_last":history[:,-1].copy(),"target":target,"observed":observed,"origins":np.arange(12,dtype=np.int64)+512}
    z=np.ascontiguousarray(history[:,:,[0,2]],dtype=np.float32);return raw,z


class NewDirectionRidgeRunnerTests(unittest.TestCase):
    def test_three_feature_contracts_and_currentC_anchor_are_exact_FP32(self):
        raw,z=fixture()
        for route in ('RAW_CURRENT_C','RAW_HISTORY512','PCA_K2_HISTORY512'):
            x,anchor=run.feature_payload(raw,route,z)
            if route=='RAW_CURRENT_C':np.testing.assert_array_equal(x,raw['raw_last']);np.testing.assert_array_equal(anchor,np.zeros_like(anchor))
            else:
                source=raw['history'] if route=='RAW_HISTORY512' else z
                self.assertEqual(x.tobytes(),(source-source[:,-1:]).reshape(12,-1).tobytes());self.assertEqual(anchor.tobytes(),raw['raw_last'].tobytes())
                self.assertEqual(x.dtype,np.float32);self.assertTrue(x.flags.c_contiguous)
        self.assertEqual(tuple(inspect.signature(run.feature_payload).parameters),('raw','route','latent_history'))

    def test_current_horizon_complete_mask_vs_history_per_response_and_FP64_innovation(self):
        raw,z=fixture();current,cmask=run.response_payload(raw,'RAW_CURRENT_C');history,hmask=run.response_payload(raw,'RAW_HISTORY512')
        current=current.reshape(raw['target'].shape);history=history.reshape(raw['target'].shape);cmask=cmask.reshape(raw['target'].shape);hmask=hmask.reshape(raw['target'].shape)
        self.assertFalse(cmask[0,2].any());self.assertEqual(hmask[0,2].tolist(),[False,True,True])
        self.assertEqual(current.dtype,np.float64);self.assertEqual(history.dtype,np.float64)
        np.testing.assert_array_equal(current[cmask],raw['target'][cmask].astype(np.float64))
        anchor=np.broadcast_to(raw['raw_last'].astype(np.float64)[:,None],history.shape)
        np.testing.assert_array_equal(history[hmask],raw['target'][hmask].astype(np.float64)-anchor[hmask]);self.assertTrue((history[~hmask]==0).all())
        other=copy.deepcopy(raw);other['target'][~other['observed']]=np.inf
        for route in ('RAW_CURRENT_C','RAW_HISTORY512'):
            a,m=run.response_payload(raw,route);b,n=run.response_payload(other,route);np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(m,n)

    def test_three_actual_tiny_fit_paths_all_alpha_and_own_fused_FP32_predictions(self):
        raw,z=fixture()
        for route in ('RAW_CURRENT_C','RAW_HISTORY512','PCA_K2_HISTORY512'):
            with self.subTest(route=route):
                features,anchor=run.feature_payload(raw,route,z);y,m=run.response_payload(raw,route);normalization='PER_RESPONSE_MASK' if route=='RAW_CURRENT_C' else 'GLOBAL_FIT'
                states,meta=ridge.fit_path(features,y,m,normalization=normalization);self.assertEqual(tuple(s['alpha'] for s in states),run.ALPHAS)
                for state in states:
                    p=run.predict(features,state,anchor,batch_size=4)
                    expected=np.concatenate([(features[i:i+4]@state['coefficient_FP32']+state['intercept_FP32']).reshape(-1,48,3)+anchor[i:i+4,None] for i in range(0,12,4)])
                    self.assertEqual(p.tobytes(),expected.tobytes());self.assertEqual(p.dtype,np.float32);self.assertTrue(np.isfinite(p).all())
                summary,arrays=run.fit_info(meta);json.dumps(summary,allow_nan=False);self.assertEqual(summary['group_count'],len(meta['groups']));self.assertIn('global_feature_scale_FP64',arrays)
                self.assertTrue(all(isinstance(value,np.ndarray) for value in arrays.values()))

    def test_fit_info_and_selected_coefficient64_32_roundtrip_hash_without_alpha(self):
        raw,z=fixture();x,anchor=run.feature_payload(raw,'RAW_CURRENT_C');y,m=run.response_payload(raw,'RAW_CURRENT_C');states,meta=ridge.fit_path(x,y,m,normalization='PER_RESPONSE_MASK')
        summary,arrays=run.fit_info(meta);expected,digest=run.coefficient_records(states[3])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'synthetic_selected.npz';np.savez_compressed(path,**{k:v for k,v in states[3].items() if k!='alpha'})
            with np.load(path,allow_pickle=False) as payload:loaded={k:payload[k].copy() for k in payload.files}
            self.assertEqual(run.coefficient_records(loaded),(expected,digest));self.assertEqual(set(loaded),{'coefficient_FP64','intercept_FP64','coefficient_FP32','intercept_FP32'})
        self.assertFalse(any(isinstance(v,np.ndarray) for v in summary.values()));self.assertEqual(len(summary['groups']),len(meta['groups']))
        for group in summary['groups']:
            for member in group['array_members'].values():self.assertIn(member,arrays)

    def test_canonical32_forward_has_no_target_or_mask_and_preserves_anchor(self):
        self.assertEqual(tuple(inspect.signature(run.predict).parameters),('features','state','anchor','check','batch_size'))
        x=np.zeros((65,2),np.float32);state={'coefficient_FP32':np.zeros((2,144),np.float32),'intercept_FP32':np.zeros(144,np.float32)};anchor=np.arange(195,dtype=np.float32).reshape(65,3);events=[]
        p=run.predict(x,state,anchor,events.append)
        self.assertEqual([p['start'] for p in events],[0,32,64]);np.testing.assert_array_equal(p,np.broadcast_to(anchor[:,None],(65,48,3)))

    def test_exact_alpha_tie_uses_registered_order_and_skips_failed_candidates(self):
        rows=[{'alpha':a,'TUNE_MSE':.2,'status':'COMPLETE'} for a in run.ALPHAS];self.assertEqual(run.select_alpha(rows)['alpha'],0.)
        rows[0]['status']='FAILED_CANDIDATE_PRESERVED';self.assertEqual(run.select_alpha(rows)['alpha'],1e-4)
        rows[-1]['TUNE_MSE']=.1;self.assertEqual(run.select_alpha(rows)['alpha'],10.)

    def test_twelve_resolved_locks_before_any_DEV_and_failed_fit_is_local(self):
        result={'datasets':{d:{'routes':{}} for d in run.DATASETS},'failed_paths':[],'selected_DEV_evaluations_completed':0};events=[]
        def execute(data,route):
            if (data,route)==run.CONDITIONS[1]:
                result['datasets'][data]['routes'][route]={'fit_artifact':{'path':'preserved_fit'}};raise RuntimeError('synthetic SVD failure')
            return {'status':'COMPLETE','selected_alpha':0.}
        def seal(data,route,row):events.append(('lock',data,route));return {'path':route}
        def registry(rows):events.append(('registry',len(rows)));return {'path':'registry'}
        resolved=run.resolve_paths(result,execute,seal,registry);self.assertEqual(len(resolved),11);data,route=run.CONDITIONS[1];self.assertIn('fit_artifact',result['datasets'][data]['routes'][route])
        def dev(data,route,row):self.assertEqual(sum(e[0]=='lock' for e in events),12);self.assertEqual(events[-1],('registry',12));return {'status':'COMPLETE'}
        run.infer_development(result,resolved,dev);self.assertEqual(result['selected_DEV_evaluations_completed'],11);self.assertEqual(len(result['failed_paths']),1)
        with self.assertRaisesRegex(ValueError,'TWELVE'):run.infer_development({'all_12_locks_before_DEV':False},resolved,dev)

    def test_closed_gate_and_stdlib_import_before_any_actual_arrays_or_models(self):
        with patch.object(run,'CROSS_AUDIT_SHA',None),patch.object(run.old,'record',side_effect=AssertionError('IO forbidden')):
            with self.assertRaisesRegex(ValueError,'NOT_REGISTERED'):run.load_gates(SimpleNamespace())
        code='import sys;sys.path.insert(0,'+repr(str(run.HERE))+');import run_new_direction_direct_ridge_v1;assert "numpy" not in sys.modules and "torch" not in sys.modules;print("STDLIB_RIDGE_IMPORT_PASS")'
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,check=True);self.assertIn('STDLIB_RIDGE_IMPORT_PASS',result.stdout)


if __name__=='__main__':unittest.main()
