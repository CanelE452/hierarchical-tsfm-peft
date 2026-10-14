"""CPU checks for preserving timings and using independent science references."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import copy
import tempfile
import unittest
import numpy as np
import torch
import batch_science_reference as bs
import resume_cost_measurement as rc


def old_row(identity, batch=1):
    return {'dataset': 'sml', 'identity': identity, 'route': 'CORE', 'split': 'eval', 'batch': batch,
        'warmup': 2, 'repeats': 7, 'seconds': [.01]*7, 'median_seconds': .01,
        'median_seconds_per_window': .01/batch, 'parity_pass': True,
        'physical_parity_atol': .002, 'physical_parity_rtol': 2e-5}


class RecoveryTests(unittest.TestCase):
    def test_positive_unique_original_subset_is_reused_without_mutation(self):
        routes = [{'dataset': 'sml', 'identity': label, 'route': 'CORE', 'split': 'eval'} for label in ('one','two','three')]
        rows = [old_row(label, b) for label in ('one','two') for b in (1,8)]
        original = {'status': 'FAILED_PRESERVED', 'records': rows}
        before = copy.deepcopy(original)
        reused, missing = rc.validate_reused_rows(original, routes, expected_count=4, expected_total=6)
        self.assertEqual(original, before)
        self.assertIs(reused[0], rows[0])
        self.assertEqual(missing, {('sml','three','CORE',1),('sml','three','CORE',8)})
        for alteration in ('duplicate','negative','wrong_repeats','false_parity','different_gate','wrong_median'):
            bad=copy.deepcopy(original)
            if alteration=='duplicate': bad['records'][1]=copy.deepcopy(bad['records'][0])
            elif alteration=='negative': bad['records'][0]['seconds'][0]=-.01
            elif alteration=='wrong_repeats': bad['records'][0]['repeats']=8
            elif alteration=='false_parity': bad['records'][0]['parity_pass']=False
            elif alteration=='different_gate': bad['records'][0]['physical_parity_atol']=.01
            else: bad['records'][0]['median_seconds']=.02
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                rc.validate_reused_rows(bad,routes,expected_count=4,expected_total=6)

    def test_matched_physical_gate_remains_strict_and_bulk_can_fail(self):
        bulk=np.full((1,48,1),-2.,np.float32)
        matched=np.nextafter(bulk,np.float32(-np.inf)); mean=np.array([65536.],np.float32); std=np.array([32768.],np.float32)
        actual=matched*std+mean
        self.assertTrue(bs.physical_parity(actual,matched,mean,std)['parity_pass'])
        self.assertFalse(bs.physical_parity(actual,bulk,mean,std)['parity_pass'])
        self.assertTrue(bs.normalized_reinference(matched,bulk)['gate_pass'])
        self.assertFalse(bs.normalized_reinference(matched+.01,bulk)['gate_pass'])
        invalid=actual.copy(); invalid[0,0,0]=np.nan
        with self.assertRaises(ValueError): bs.physical_parity(invalid,matched,mean,std)

    def test_science_readout_has_a_separate_mutable_memory_model(self):
        class SmallModel(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.w=torch.nn.Parameter(torch.ones(1)); self.register_buffer('memory',torch.zeros(1))
        deployment_model=SmallModel(); science_model=SmallModel()
        pack={'history':{'eval':np.zeros((8,512,1),np.float32)},'origins':{'eval':np.arange(8)}}
        route={'route':'E_ENRICHED_RIDGE','split':'eval','case':{},'readout':{'method':'E_ENRICHED_RIDGE'}}
        def evaluate(thin,split,case,model,readout,rt,limit):
            self.assertEqual(set(thin),{'history','origins'})
            self.assertEqual(len(thin['history'][split]),1)
            model.memory.add_(1)
            return np.ones((1,48,1),np.float32)
        with patch.object(bs,'load_selected_model',return_value=science_model), patch.object(bs.pr,'evaluate_readout',side_effect=evaluate):
            reference,meta=bs.science_reference(pack,route,1,None,None,device='cpu')
        self.assertEqual(reference.shape,(1,48,1))
        self.assertEqual(float(deployment_model.memory),0.)
        self.assertTrue(meta['registered_parameters_unchanged'])
        self.assertFalse(meta['selected_full_state_unchanged'])
        self.assertIn('evaluate_readout',meta['API'])

    def test_raw_joint_reference_uses_original_numpy_ridge_path(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory); path=base/'weights.npz'
            weight=np.zeros((33,48),np.float32); weight[-1]=2.
            np.savez(path,weight=weight,bias=np.zeros(48,np.float32),horizon=np.asarray(48),channels=np.asarray(1))
            artifact={'path':'weights.npz','sha256':bs.rt.sha(path)}
            pack={'history':{'eval':np.full((8,512,1),3.,np.float32)},'origins':{'eval':np.arange(8)}}
            route={'route':'RAW_JOINT_RIDGE','split':'eval','control':{'method':'RAW_JOINT_RIDGE','model':artifact}}
            with patch.object(bs.rt,'ROOT',base):
                prediction,meta=bs.science_reference(pack,route,1,None,None,device='cpu')
            self.assertTrue(np.array_equal(prediction,np.full((1,48,1),9.,np.float32)))
            self.assertIn('evaluate_raw_control',meta['API'])
            self.assertIsNone(meta['selected_model_before_sha256'])

    def test_numeric_failure_keeps_seven_arrays_and_does_not_raise(self):
        pack={'dataset':'sml','mean':np.zeros(1,np.float32),'std':np.ones(1,np.float32),
            'history':{'eval':np.zeros((8,512,1),np.float32)},
            'raw_history':{'eval':np.zeros((8,512,1),np.float32)},
            'raw_history_observed':{'eval':np.ones((8,512,1),bool)},'origins':{'eval':np.arange(8)}}
        route={'dataset':'sml','identity':'one','route':'CORE','split':'eval'}
        ref=np.zeros((1,48,1),np.float32)
        times=iter(np.arange(14)*.01)
        with patch.object(rc.bench,'deployment',return_value=np.full((1,48,1),.1,np.float32)) as deployment, \
                patch.object(rc.torch.cuda,'synchronize'), patch.object(rc.torch.cuda,'reset_peak_memory_stats'), \
                patch.object(rc.torch.cuda,'max_memory_allocated',return_value=0), patch.object(rc.time,'perf_counter',side_effect=lambda:next(times)):
            row,arrays=rc.measure_batch(pack,route,1,ref,ref,None,None,None)
        self.assertEqual(deployment.call_count,9)
        self.assertFalse(row['parity_pass'])
        self.assertEqual(len(row['seconds']),7)
        self.assertEqual(arrays['all_7_physical_outputs'].shape,(7,1,48,1))
        self.assertEqual(row['warmup'],2)
        self.assertEqual(row['physical_parity_atol'],.002)

    def test_original_science_core_receives_requested_batch_and_no_targets(self):
        model=torch.nn.Linear(1,1)
        pack={'history':{'eval':np.zeros((8,512,1),np.float32)},'origins':{'eval':np.arange(8)}}
        route={'route':'CORE','split':'eval','case':{}}
        def predict(actual_model,history,origins,forecaster,limit,batch):
            self.assertIs(actual_model,model); self.assertEqual(batch,1)
            self.assertEqual(len(history),1); self.assertTrue(np.array_equal(origins,np.array([0])))
            return np.zeros((1,48,1),np.float32)
        with patch.object(bs,'load_selected_model',return_value=model),patch.object(bs.tr,'predict',side_effect=predict):
            _,meta=bs.science_reference(pack,route,1,None,None,device='cpu')
        self.assertTrue(meta['selected_full_state_unchanged'])
        self.assertIn('train.predict',meta['API'])

    def test_past_loader_ignores_nonexistent_future_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory); path=base/'past.npz'
            history=np.zeros((8,512,1),np.float32); origins=np.arange(8)
            mean=np.zeros(1,np.float32);std=np.ones(1,np.float32)
            np.savez(path,history=history,physical_history=history,history_observed=np.ones_like(history,bool),
                origins=origins,mean=mean,std=std)
            contract={'C':1,'K':1,'columns':['channel'],'method_lock':{'sha256':'locked'},
                'statistics':{field+'_sha256':bs.ru.dt.array_hash(value) for field,value in (('mean',mean),('std',std))},
                'payloads':{'eval_past_input':{'path':'past.npz','sha256':bs.rt.sha(path)},
                    'eval_future_target':{'path':'DO_NOT_OPEN_TARGET.npz','sha256':'absent'}}}
            evaluation={'global_model_lock':{'sha256':'global'},'evaluation_open_receipts':[{'dataset':'sml',
                'opened_only_after_all_selected':True,'method_lock_sha256':'locked','global_model_lock':{'sha256':'global'}}]}
            with patch.object(bs.rt,'ROOT',base),patch.object(bs.ru,'FRESH',('sml',)):
                pack,meta=bs.load_past_only('sml',{'contracts':{'sml':contract}},evaluation,{'sha256':'locked'})
            self.assertNotIn('target',pack)
            self.assertEqual(meta['accessed_members'],['history','physical_history','history_observed','origins','mean','std'])
            self.assertFalse((base/'DO_NOT_OPEN_TARGET.npz').exists())

    def test_asset_sha_and_process_wall_limit_are_real_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory); path=base/'asset';path.write_bytes(b'original')
            artifact={'path':'asset','sha256':bs.rt.sha(path)}
            with patch.object(bs.rt,'ROOT',base):
                self.assertEqual(bs.verify_artifact(artifact),path)
                path.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'ARTIFACT_SHA'):bs.verify_artifact(artifact)
        fake=SimpleNamespace(check=lambda:{'elapsed_seconds':299.})
        self.assertEqual(rc.CappedLimit(fake).check()['elapsed_seconds'],299.)
        fake.check=lambda:{'elapsed_seconds':300.001}
        with self.assertRaisesRegex(rc.rt.BudgetHeld,'300SECONDS'):rc.CappedLimit(fake).check()


if __name__=='__main__': unittest.main()
