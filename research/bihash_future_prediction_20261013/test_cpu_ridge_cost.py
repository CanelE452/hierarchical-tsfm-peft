"""Focused CPU tests for uniform preloaded heads and canonical cost lineage."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import copy
import tempfile
import unittest
import numpy as np
import torch
import cpu_ridge_deployment as cpu
import merge_canonical_cost as merge
import measure_cpu_ridge_cost as measure


class CPUHeadTests(unittest.TestCase):
    def test_preloaded_raw_joint_and_self_heads_need_no_disk_or_GPU(self):
        x=np.full((2,512,2),3,np.float32);mask=np.ones_like(x,bool)
        pack={'mean':np.zeros(2,np.float32),'std':np.ones(2,np.float32)}
        joint={'weight':np.zeros((66,96),np.float32),'bias':np.ones(96,np.float32),'horizon':48,'channels':2}
        per={'weight':np.zeros((2,33,48),np.float32),'bias':np.ones((2,48),np.float32)}
        with cpu.blas_context(),patch.object(cpu.np,'load',side_effect=AssertionError('timed disk access')):
            for route,ridge in [('RAW_JOINT_RIDGE',joint),('RAW_SELF33',per)]:
                y=cpu.deployment(pack,x,mask,np.array([512,520]),route,None,ridge=ridge)
                self.assertTrue(np.array_equal(y,np.full((2,48,2),4,np.float32)))
            changed=cpu.deployment(pack,x+2,mask,np.array([512,520]),'RAW_JOINT_RIDGE',None,ridge=joint)
            self.assertTrue(np.array_equal(changed,np.full((2,48,2),6,np.float32)))

    def test_cpu_coefficients_loader_preserves_dtype_and_sha_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);path=base/'ridge.npz'
            np.savez(path,weight=np.zeros((33,48),np.float32),bias=np.zeros(48,np.float32),horizon=np.asarray(48),channels=np.asarray(1))
            import runtime as rt
            record={'model':{'path':'ridge.npz','sha256':rt.sha(path)}}
            with patch.object(rt,'ROOT',base):
                ridge=cpu.load_ridge(record,rt)
                self.assertIsInstance(ridge['weight'],np.ndarray);self.assertEqual(ridge['weight'].dtype,np.float32)
                path.write_bytes(b'changed')
                with self.assertRaises(ValueError):cpu.load_ridge(record,rt)

    def test_existing_frozen_memory_is_not_recomputed(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.w=torch.nn.Parameter(torch.ones(1));self.register_buffer('memory_is_frozen',torch.tensor(True))
                self.register_buffer('memory_embedding',torch.ones(3,2));self.register_buffer('memory_hash_logits',torch.ones(3,2))
            def freeze_memory(self):raise AssertionError('cache recomputation in prepared model')
        m=Model();meta=cpu.prepare_memory(m)
        self.assertFalse(meta['cache_recomputed']);self.assertTrue(meta['registered_parameters_unchanged'])
        self.assertEqual(meta['model_state_before_sha256'],meta['model_state_after_sha256'])

    def test_uniform_merge_preserves_296_objects_and_rejects_failed_or_duplicate_rows(self):
        cuda=[{'dataset':'ds','identity':'core','route':'CORE','batch':b,'parity_pass':True} for b in (1,8)]
        head=[{'dataset':'ds','identity':'ridge','route':'E_RAW_RIDGE','batch':b,'parity_pass':True,
            'deployment_implementation':cpu.IMPLEMENTATION_ID} for b in (1,8)]
        before=copy.deepcopy(cuda)
        expected={merge.key(r) for r in cuda+head}
        merged=merge.combine_rows(cuda,head,expected,cuda_expected=2,cpu_expected=2)
        self.assertIs(merged[0],cuda[0]);self.assertEqual(cuda,before)
        bad=copy.deepcopy(head);bad[1]=copy.deepcopy(bad[0])
        with self.assertRaises(ValueError):merge.combine_rows(cuda,bad,expected,cuda_expected=2,cpu_expected=2)
        bad=copy.deepcopy(head);bad[0]['parity_pass']=False
        with self.assertRaises(ValueError):merge.combine_rows(cuda,bad,expected,cuda_expected=2,cpu_expected=2)
        bad=copy.deepcopy(head);bad[0]['deployment_implementation']='unknown'
        with self.assertRaises(ValueError):merge.combine_rows(cuda,bad,expected,cuda_expected=2,cpu_expected=2)

    def test_actual_cpu_head_measurer_keeps_all_seven_outputs_and_excludes_disk(self):
        raw=np.zeros((8,512,1),np.float32)
        pack={'dataset':'ds','mean':np.zeros(1,np.float32),'std':np.ones(1,np.float32),
            'raw_history':{'val':raw},'raw_history_observed':{'val':np.ones_like(raw,bool)},
            'history':{'val':raw},'origins':{'val':np.arange(8)}}
        route={'dataset':'ds','identity':'joint','route':'RAW_JOINT_RIDGE','split':'val'}
        ridge={'weight':np.zeros((33,48),np.float32),'bias':np.ones(48,np.float32),'horizon':48,'channels':1}
        matched=np.ones((1,48,1),np.float32);clock=iter(np.arange(14)*.01)
        with cpu.blas_context(),patch.object(cpu.np,'load',side_effect=AssertionError('timed disk access')), \
            patch.object(measure.torch.cuda,'synchronize'),patch.object(measure.torch.cuda,'reset_peak_memory_stats'), \
            patch.object(measure.torch.cuda,'max_memory_allocated',return_value=0), \
            patch.object(measure.time,'perf_counter',side_effect=lambda:next(clock)):
            row,arrays=measure.measure(pack,route,1,matched,matched,None,None,ridge)
        self.assertTrue(row['parity_pass']);self.assertEqual(len(row['seconds']),7)
        self.assertEqual(row['deployment_implementation'],cpu.IMPLEMENTATION_ID)
        self.assertTrue(row['implementation_metadata']['all_BLAS_pools_have_two_threads'])
        self.assertTrue(np.array_equal(arrays['all_7_physical_outputs'],np.ones((7,1,48,1),np.float32)))

    def test_merge_requires_positive_timings_and_unchanged_physical_tolerance(self):
        r={'warmup':2,'repeats':7,'seconds':[.01]*7,'median_seconds':.01,'median_seconds_per_window':.01,
            'batch':1,'physical_parity_atol':.002,'physical_parity_rtol':2e-5}
        merge.verify_rows([r])
        for field,value in [('seconds',[0.]*7),('physical_parity_atol',.003),('median_seconds',.02)]:
            bad={**r,field:value}
            with self.subTest(field=field),self.assertRaises(ValueError):merge.verify_rows([bad])


if __name__=='__main__':unittest.main()
