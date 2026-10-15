"""Small CPU FIT-only direction/D integration and conservative causal oracle masks."""
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

import run_predictive_cross_moment_v1 as run
from linear_e_adaptation_v1 import LinearLagAdapter
from causal_lag_decoder_v1 import array_binding


def fixture():
    rng=np.random.default_rng(92803);x=rng.normal(scale=.3,size=(12,512,4)).astype(np.float32)
    y=np.broadcast_to(x[:,-1:,[1,2,3,0]],(12,48,4)).copy();y+=rng.normal(scale=.03,size=y.shape).astype(np.float32)
    mask=np.ones_like(y,dtype=bool);mask[0,2,0]=False;mask[1,4,2]=False;y[~mask]=np.nan
    return {"history":x,"history_observed":np.ones_like(x,dtype=bool),"target":y,"observed":mask,"raw_last":x[:,-1].copy(),
        "raw_forecast":np.broadcast_to(x[:,-1:],y.shape).copy(),"std":np.arange(1,5,dtype=np.float32),"mean":np.arange(4,dtype=np.float32),"origins":np.arange(12,dtype=np.int64)+512}


def toy_F(z):return z[:,-1:].expand(-1,48,-1)+.1*z.mean(1,keepdim=True)


def outputs_for(raw,state):
    model=LinearLagAdapter(state['center'],state['E'],state['D'],run.LAGS,learn_E=False).eval().requires_grad_(False)
    outputs=run.old.predict_components(model,toy_F,raw['history'],lambda value:None)
    complete=raw['observed'].all(2);safe=np.where(complete[:,:,None],raw['target'],raw['raw_last'][:,None]).astype(np.float32)
    ey=run.encode_torch(model,safe);z=torch.from_numpy(outputs['latent_history']);current=torch.from_numpy(raw['raw_last'])
    with torch.no_grad():
        eyp=model.decode(current,z,torch.from_numpy(ey)).numpy().copy()
        flatp=model.decode(current,z,z[:,-1:].expand(-1,48,-1)).numpy().copy()
    return model,outputs,ey,eyp,flatp


class PredictiveCrossMomentRunnerTests(unittest.TestCase):
    def test_same_actual_FP32_encoder_and_RECON_D_rule_for_both_directions_and_K(self):
        raw=fixture()
        for kind in run.KINDS:
            for k in (1,2):
                with self.subTest(kind=kind,k=k):
                    direction,fitted,state,z,timing=run.fit_state(raw,kind,k,device='cpu')
                    self.assertEqual(state['E'].shape,(4,k));self.assertEqual(state['D'].shape,(7*k,4))
                    self.assertEqual(fitted['metadata']['input_bindings']['latent_history'],array_binding(z))
                    self.assertEqual(fitted['metadata']['common_history_slice'],[32,512]);self.assertEqual(fitted['metadata']['singular_relative_cutoff'],1e-4)
                    self.assertFalse(fitted['metadata']['future_FIT_Y_used_as_supervision']);self.assertTrue(all(v>=0 for v in timing.values()))
                    model,outputs,ey,eyp,flatp=outputs_for(raw,state);row,independent=run.evaluate_arrays(outputs,raw,state,ey,eyp,flatp)
                    self.assertEqual(row['status'],'COMPLETE');self.assertEqual(set(row['methods']),{'ACTUAL_F','RAW_BOLT','FLAT','CURRENT_C','E_Y'})
                    self.assertTrue(row['methods']['E_Y']['non_oracle_mask_scores_not_reported']);self.assertNotIn('scores_all_observed',row['methods']['E_Y'])
                    self.assertEqual(row['output_space_minimum_status'],'NOT_MEASURED_IN_THIS_STAGE');self.assertFalse(row['oracle_is_deployment_or_predictability_proof'])
                    np.testing.assert_array_equal(independent['oracle_valid_horizon_mask'],run.oracle_dependency_mask(raw['observed']))
                    self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()));json.dumps(row,allow_nan=False)

    def test_oracle_requires_earlier_future_references_and_never_masks_known_past(self):
        observed=np.ones((1,8,3),dtype=bool);observed[0,2,1]=False
        mask=run.oracle_dependency_mask(observed,(0,1,4))
        np.testing.assert_array_equal(np.flatnonzero(~mask[0]),[2,3,6])
        self.assertTrue(observed[0,3].all());self.assertFalse(mask[0,3])
        observed[:]=True;observed[0,0,0]=False;mask=run.oracle_dependency_mask(observed,(0,4))
        np.testing.assert_array_equal(np.flatnonzero(~mask[0]),[0,4]);self.assertTrue(mask[0,1:4].all())
        with self.assertRaisesRegex(ValueError,'LAG_ZERO'):run.oracle_dependency_mask(observed,(1,4))

    def test_unobserved_future_placeholders_do_not_change_E_or_oracle_or_scores(self):
        raw=fixture();direction,fitted,state,z,_=run.fit_state(raw,'CROSS_MOMENT',2,device='cpu')
        model,outputs,ey,eyp,flatp=outputs_for(raw,state);row,_=run.evaluate_arrays(outputs,raw,state,ey,eyp,flatp)
        changed=copy.deepcopy(raw);changed['target'][~changed['observed']]=np.inf
        direction2,_,state2,_,_=run.fit_state(changed,'CROSS_MOMENT',2,device='cpu')
        np.testing.assert_array_equal(direction['E'],direction2['E']);np.testing.assert_array_equal(state['D'],state2['D'])
        _,outputs2,ey2,eyp2,flatp2=outputs_for(changed,state)
        for name in outputs:np.testing.assert_array_equal(outputs[name],outputs2[name])
        np.testing.assert_array_equal(ey,ey2);other,_=run.evaluate_arrays(outputs2,changed,state,ey2,eyp2,flatp2);self.assertEqual(row,other)

    def test_DEPLOYED_forward_has_only_history_and_state_not_future_or_mask(self):
        self.assertEqual(tuple(inspect.signature(run.encode_torch).parameters),('model','history','batch_size'))
        raw=fixture();_,_,state,_,_=run.fit_state(raw,'PCA',1,device='cpu');model,outputs,_,_,_=outputs_for(raw,state)
        raw['target'][:]=1e8;raw['observed'][:]=False
        repeat=run.old.predict_components(model,toy_F,raw['history'],lambda value:None)
        for key in outputs:self.assertEqual(outputs[key].tobytes(),repeat[key].tobytes())

    def test_oracle_common_mask_is_shared_by_Raw_actual_flat_current_and_EY(self):
        raw=fixture();_,_,state,_,_=run.fit_state(raw,'PCA',1,device='cpu');_,outputs,ey,eyp,flatp=outputs_for(raw,state)
        row,_=run.evaluate_arrays(outputs,raw,state,ey,eyp,flatp);counts=[]
        for method in row['methods'].values():counts.append(method['scores_oracle_common_mask']['normalized']['observed_counts'])
        self.assertTrue(all(c==counts[0] for c in counts));self.assertEqual(counts[0],[row['oracle_valid_horizons']]*4)
        self.assertLess(row['oracle_valid_horizons'],row['complete_C_horizons'])

    def test_timestamp_occurrences_count_duplicates_not_independent_samples(self):
        raw=fixture();meta=run.timestamp_metadata(raw)
        self.assertEqual(meta['history']['occurrences'],12*512);self.assertEqual(meta['history']['unique_timestamps'],523)
        self.assertGreater(meta['history']['maximum_multiplicity'],1);self.assertEqual(meta['future']['occurrences'],12*48)
        self.assertEqual(meta['complete_C_response']['occurrences'],12*48-2)
        self.assertTrue(meta['overlapping_occurrences_not_independent_samples']);self.assertFalse(meta['new_raw_data_split_boundary_reconstruction'])

    def test_all_eight_states_and_failure_locks_resolve_before_any_DEV(self):
        result={'datasets':{d:{'conditions':{}} for d in run.DATASETS},'failed_paths':[],'DEV_evaluations_completed':0};events=[]
        def execute(data,kind,k):
            name=kind+'_K'+str(k)
            if (data,kind,k)==run.CONDITIONS[1]:
                result['datasets'][data]['conditions'][name]={'direction_artifact':{'path':'preserved_direction'},'splits':{}}
                raise RuntimeError('synthetic decoder failure')
            return {'status':'COMPLETE','splits':{'fit':{'status':'COMPLETE'},'tune':{'status':'COMPLETE'}}}
        def seal(data,name,row):events.append(('lock',data,name));return {'path':name}
        def registry(rows):events.append(('registry',len(rows)));return {'path':'registry'}
        resolved=run.resolve_states(result,execute,seal,registry);self.assertEqual(len(resolved),7)
        data,kind,k=run.CONDITIONS[1];self.assertIn('direction_artifact',result['datasets'][data]['conditions'][kind+'_K'+str(k)])
        def development(data,name,row):self.assertEqual(sum(e[0]=='lock' for e in events),8);self.assertEqual(events[-1],('registry',8));return {'status':'COMPLETE'}
        run.infer_development(result,resolved,development);self.assertEqual(result['DEV_evaluations_completed'],7)
        self.assertEqual(len(result['failed_paths']),1)
        with self.assertRaisesRegex(ValueError,'EIGHT'):run.infer_development({'all_8_locks_before_DEV':False},resolved,development)

    def test_independent_decode_violation_is_recorded_as_failed_not_as_performance_gate(self):
        raw=fixture();_,_,state,_,_=run.fit_state(raw,'PCA',1,device='cpu');_,outputs,ey,eyp,flatp=outputs_for(raw,state)
        outputs['prediction'][0,0,0]+=.1;row,_=run.evaluate_arrays(outputs,raw,state,ey,eyp,flatp)
        self.assertEqual(row['status'],'FAILED_PARITY_PRESERVED');self.assertFalse(row['checks']['independent_NumPy_old_lag_C']['pass'])
        self.assertIn('scores_all_observed',row['methods']['ACTUAL_F'])

    def test_stdlib_import_and_closed_source_gate_precede_any_scientific_IO(self):
        with patch.object(run,'MODULE_SHA',None),patch.object(run.old,'bound',side_effect=AssertionError('IO forbidden')):
            with self.assertRaisesRegex(ValueError,'NOT_REGISTERED'):run.load_gates(SimpleNamespace())
        code='import sys;sys.path.insert(0,'+repr(str(run.HERE))+');import run_predictive_cross_moment_v1;assert "numpy" not in sys.modules and "torch" not in sys.modules;print("STDLIB_IMPORT_PASS")'
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,check=True);self.assertIn('STDLIB_IMPORT_PASS',result.stdout)


if __name__=='__main__':unittest.main()
