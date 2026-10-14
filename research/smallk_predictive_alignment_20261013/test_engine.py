"""Small contract fixtures before real Bolt preflight, no training campaign writes."""
import unittest
import numpy as np
import torch
import engine as en
from alignment import future_coordinates, summary_numpy, summary_torch, nonoverlap_permutation


class ContractTests(unittest.TestCase):
    def test_identity_anchor_algebra(self):
        model=en.Codec(np.zeros(3,np.float32),np.eye(3,dtype=np.float32),92601,'ANCHOR_PCA')
        x=torch.randn(2,32,3)
        f=lambda z: z[:,-1:]+torch.arange(48)[None,:,None]*.01
        self.assertTrue(torch.allclose(model.forward_path(x,f),f(x),atol=2e-7))
    def test_common_initialization_and_zero_residual(self):
        basis=np.linalg.qr(np.random.default_rng(0).normal(size=(7,2)))[0].astype(np.float32)
        models={a:en.Codec(np.zeros(7,np.float32),basis,92601,a) for a in en.ARCHITECTURES}
        x=torch.randn(2,64,7)
        initial=models['ANCHOR_TEACHER'].encode(x)
        for m in models.values(): self.assertTrue(torch.equal(m.encode(x),initial))
        z=torch.randn(2,48,2)
        self.assertTrue(torch.equal(models['ANCHOR_NONLINEAR_D'].decode(z),models['ANCHOR_TEACHER'].decode(z)))
    def test_projection_complete_observed_parity(self):
        rng=np.random.default_rng(2)
        basis=np.linalg.qr(rng.normal(size=(6,2)))[0].astype(np.float32)
        target=rng.normal(size=(4,48,6)).astype(np.float32); last=rng.normal(size=(4,6)).astype(np.float32)
        actual,valid=future_coordinates(target,np.ones_like(target,dtype=bool),last,basis)
        self.assertTrue(valid.all())
        self.assertLess(np.max(np.abs(actual-(target-last[:,None])@basis)),5e-7)
    def test_missing_not_filled_target(self):
        basis=np.eye(3,dtype=np.float32)[:,:2]
        y=np.zeros((2,48,3),np.float32); m=np.zeros_like(y,dtype=bool)
        m[:,:,0]=True
        _,valid=future_coordinates(y,m,np.zeros((2,3),np.float32),basis)
        self.assertFalse(valid.any())
    def test_summary_mask_and_torch_parity(self):
        rng=np.random.default_rng(0); z=rng.normal(size=(5,48,2)).astype(np.float32)
        valid=rng.random((5,48))>.2
        a,mask=summary_numpy(z,valid)
        b=summary_torch(torch.from_numpy(z),torch.from_numpy(valid)).numpy()
        self.assertLess(np.max(np.abs(a-b)),2e-7)
        self.assertTrue(np.array_equal(mask[:,2:4],np.repeat(valid[:,-1:],2,1)))
    def test_shuffle_nonoverlap_bijection(self):
        origins=np.arange(256)*16
        p=nonoverlap_permutation(origins,92601)
        self.assertEqual(len(np.unique(p)),256)
        self.assertGreaterEqual(np.abs(origins-origins[p]).min(),48)
    def test_two_adjacent_observed_points_slope(self):
        z=np.zeros((1,48,1),np.float32); z[:,0]=3; z[:,1]=5
        valid=np.zeros((1,48),bool); valid[:,:2]=True
        a,_=summary_numpy(z,valid)
        b=summary_torch(torch.from_numpy(z),torch.from_numpy(valid)).numpy()
        self.assertEqual(float(a[0,-1]),2)
        self.assertEqual(float(b[0,-1]),2)


if __name__=='__main__': unittest.main()
