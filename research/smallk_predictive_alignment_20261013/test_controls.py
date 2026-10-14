"""Pure/native contract tests without foundation loads or new EVAL payloads."""
import unittest
import numpy as np
import torch
import engine as en
import controls
import new_run


class ControlTests(unittest.TestCase):
    def test_native_persistence_anchor_initial(self):
        rng=np.random.default_rng(0); basis=np.linalg.qr(rng.normal(size=(7,2)))[0].astype(np.float32)
        codec=en.Codec(np.zeros(7,np.float32),basis,92601,'ANCHOR_TEACHER')
        model=en.ex.NativeShared(512)
        x=torch.randn(3,512,7)
        p=codec.forward_path(x,model)
        self.assertTrue(torch.allclose(p,x[:,-1:].expand_as(p),atol=1e-6))
    def test_new_dataset_keys(self):
        self.assertEqual(set(new_run.NEW_DATASETS),set(new_run.newdata.DATASETS))
    def test_selected_artifacts_deduplicated(self):
        # Test requires true existing/hash-matched file, not a fabricated model name.
        path=en.HERE/'test_controls.py'
        artifact={'path':str(path.relative_to(en.ROOT)),'sha256':en.ex.sha(path)}
        cases=[{'dataset':'fixture','training':{'selected_checkpoint':artifact},'selected_models':artifact,'methods':[]}]
        result=new_run.selected_artifacts(cases,'fixture')
        self.assertEqual(len(result),1)
    def test_same_shared_native_parameters(self):
        a,b=en.ex.NativeShared(512),en.ex.NativeShared(512)
        self.assertEqual(en.ex.previous.state_hash(a),en.ex.previous.state_hash(b))
        self.assertEqual(sum(p.numel() for p in a.parameters()),512*48+48)


if __name__=='__main__': unittest.main()
