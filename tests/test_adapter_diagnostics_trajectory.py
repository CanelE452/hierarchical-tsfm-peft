"""Stored sampler evidence and masked fixed-decoder scoring need no model execution."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]/"reports/adapter_diagnostics_20261017_v2/code/trajectory_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("trajectory_diagnostics", SOURCE)
trajectory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trajectory)


class TrajectoryTests(unittest.TestCase):
    def test_actual_order_rng_and_cursor_fingerprints_reject_same_exposure_different_order(self):
        state=dict(order=[2,0,1],position=1,exposures=4,epochs_completed=1,rng=dict(state=123),origins_sha="a"*64)
        a=trajectory.sampler_record(dict(sampler=state))
        b=trajectory.sampler_record(dict(sampler=dict(state,order=[1,0,2])))
        self.assertNotEqual(a["sampler_sha256"],b["sampler_sha256"])
        self.assertNotEqual(a["order_sha256"],b["order_sha256"])
        self.assertEqual(a["exposures"],b["exposures"])
        with self.assertRaisesRegex(ValueError,"arithmetic"):
            trajectory.sampler_record(dict(sampler=dict(state,position=2)))

    def test_streamed_masked_decoder_scores_keep_macro_counts_and_native_fp32_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"features.npz"
            phi=np.asarray([[[1,0],[0,1]],[[1,1],[2,0]]],np.float32)
            target=np.zeros((2,2,2),np.float32)
            mask=np.asarray([[[1,1],[1,0]],[[1,0],[0,0]]],bool)
            np.savez(path,phi=phi,target=target,mask=mask,anchor=np.zeros((2,1,2),np.float32))
            scores=trajectory.fixed_decoder_scores(path,np.asarray([[2,0],[0,1]],np.float32),chunk_origins=1)
            self.assertAlmostEqual(scores["float64"]["MSE"],4/3)
            self.assertAlmostEqual(scores["float64"]["MAE"],2/3)
            self.assertEqual(scores["float64"]["observed_count"],4)
            self.assertEqual(scores["float64"],scores["native_float32"])
            np.savez(path,phi=phi,target=target,mask=np.zeros_like(mask),anchor=np.zeros((2,1,2),np.float32))
            with self.assertRaisesRegex(ValueError,"No observed"):
                trajectory.fixed_decoder_scores(path,np.eye(2,dtype=np.float32))

    def test_decoder_storage_is_narrow_and_unlisted_pickle_global_never_executes(self):
        import os
        import pickle
        import zipfile
        from unittest.mock import patch
        class Forbidden:
            def __reduce__(self):
                return os.system,("must not execute",)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"bundle.pt";weights=np.arange(4,dtype=np.float32).reshape(2,2)
            with zipfile.ZipFile(path,"w") as archive:
                archive.writestr("bundle/data.pkl",pickle.dumps({}))
                archive.writestr("bundle/data/0",weights.tobytes())
                archive.writestr("bundle/data/1",b"frozen backbone storage must stay unopened")
            tensor=trajectory._Tensor(dict(key="0",dtype="<f4",count=4),0,(2,2),(2,1))
            np.testing.assert_array_equal(trajectory.tensor_array(path,tensor),weights)
            with zipfile.ZipFile(path,"w") as archive:
                archive.writestr("bundle/data.pkl",pickle.dumps(Forbidden()))
            with patch.object(os,"system") as system:
                with self.assertRaisesRegex(ValueError,"Unsupported trajectory metadata global"):
                    trajectory.checkpoint_metadata(path)
                system.assert_not_called()
            binding=dict(reference="cache://bundle.pt",sha256="0"*64)
            with self.assertRaisesRegex(ValueError,"SHA binding rejected"):
                trajectory.bound_path(dict(cache=Path(directory)),binding)


if __name__ == "__main__":
    unittest.main()
