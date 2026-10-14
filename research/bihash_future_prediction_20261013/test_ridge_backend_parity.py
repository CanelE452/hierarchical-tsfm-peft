"""CPU checks for the bounded dynamic-input CPU-head prototype."""
from pathlib import Path
from unittest.mock import patch
import tempfile
import unittest
import numpy as np
import diagnose_ridge_backend_parity as d


class RidgeBackendTests(unittest.TestCase):
    def test_cpu_head_recomputes_from_raw_inputs_with_original_numpy_ops(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);path=base/'ridge.npz'
            w=np.zeros((33,48),np.float32);w[-1]=2
            np.savez(path,weight=w,bias=np.zeros(48,np.float32),horizon=np.asarray(48),channels=np.asarray(1))
            route={'route':'RAW_JOINT_RIDGE','split':'eval','control':{'method':'RAW_JOINT_RIDGE',
                'model':{'path':'ridge.npz','sha256':d.rt.sha(path)}}}
            pack={'mean':np.zeros(1,np.float32),'std':np.ones(1,np.float32)}
            raw=np.full((1,512,1),3,np.float32);mask=np.ones_like(raw,bool);origins=np.array([512])
            with patch.object(d.rt,'ROOT',base):
                norm,physical,stages=d.cpu_head_deployment(pack,raw,mask,origins,route,None)
                other,_,_=d.cpu_head_deployment(pack,raw+1,mask,origins,route,None)
            self.assertTrue(np.array_equal(norm,np.full((1,48,1),9,np.float32)))
            self.assertTrue(np.array_equal(other,np.full((1,48,1),12,np.float32)))
            self.assertTrue(np.array_equal(norm,physical))
            self.assertEqual(stages['features_CPU'].shape,(1,33))
            self.assertTrue(np.array_equal(stages['delta_CPU'],np.full((1,48,1),6,np.float32)))

    def test_capture_wrappers_restore_original_functions_after_error(self):
        originals=(d.pr.block_features,d.pr.ridge_predict,d.pr.representation_features)
        raw=np.zeros((1,512,1),np.float32)
        route={'route':'RAW_JOINT_RIDGE','split':'eval','control':{'method':'RAW_JOINT_RIDGE'}}
        with patch.object(d.pr,'evaluate_raw_control',side_effect=ValueError('intentional_fixture_error')):
            with self.assertRaisesRegex(ValueError,'intentional'):
                d.cpu_head_deployment({'mean':np.zeros(1,np.float32),'std':np.ones(1,np.float32)},
                    raw,np.ones_like(raw,bool),np.array([512]),route,None)
        self.assertEqual(originals,(d.pr.block_features,d.pr.ridge_predict,d.pr.representation_features))

    def test_decomposition_keeps_unexplained_historical_residual(self):
        cpu=np.full((1,48,1),-2,np.float32);gpu=np.nextafter(cpu,np.float32(-np.inf))
        mean=np.array([65536],np.float32);std=np.array([32768],np.float32)
        old=gpu*std+mean
        summary,arrays=d.stage_decomposition(old,cpu,gpu,mean,std)
        self.assertEqual(summary['historical_minus_current_GPU_stage_physical_max_abs'],0)
        self.assertEqual(summary['decomposition_residual_max_abs'],0)
        self.assertGreater(summary['normalized_stage_delta_times_std_max_abs'],0)
        summary,_=d.stage_decomposition(old+.5,cpu,gpu,mean,std)
        self.assertEqual(summary['historical_minus_current_GPU_stage_physical_max_abs'],.5)
        self.assertTrue(np.array_equal(arrays['historical_vs_current_GPU_stage_physical'],np.zeros_like(old)))


if __name__=='__main__':unittest.main()
