"""Numerical contracts that must hold before costly Stage B runs."""
import unittest
import numpy as np
import torch
import component_experiment as ex
from common import Ridge


class ComponentContracts(unittest.TestCase):
    def setUp(self):
        ex.DEVICE='cpu'
        torch.set_num_threads(2)

    def test_causal_receptive_field_with_active_nonlinear_branch(self):
        codec=ex.Codec(np.zeros(8,np.float32),np.eye(8,2,dtype=np.float32),92601)
        with torch.no_grad():
            codec.output.weight.fill_(.1)
        x=torch.randn(2,512,8)
        changed=x.clone();changed[:,256:]+=100
        torch.testing.assert_close(codec.encode(x)[:,:256],codec.encode(changed)[:,:256],rtol=0,atol=1e-6)
        earlier=x.clone();earlier[:,:-127]+=100
        torch.testing.assert_close(codec.encode(x)[:,-1],codec.encode(earlier)[:,-1],rtol=0,atol=1e-6)
        torch.testing.assert_close(codec.encode(x)[:,-1],codec.encode(x[:,-127:])[:,-1],rtol=0,atol=1e-6)

    def test_masked_macro_loss_weights_channels_equally(self):
        prediction=torch.tensor([[[2.,3.],[4.,9.]]])
        observed=torch.tensor([[[True,True],[True,False]]])
        self.assertEqual(float(ex.macro_loss(prediction,torch.zeros_like(prediction),observed)),9.5)

    def test_full_history_block_features_and_numpy_torch_parity(self):
        x=np.arange(2*512*3,dtype=np.float32).reshape(2,512,3)
        np.testing.assert_array_equal(ex.block_features(x),ex.block_features(torch.from_numpy(x)).numpy())
        self.assertEqual(ex.block_features(x).shape,(2,99))
        altered=x.copy();altered[:,0]+=160
        self.assertEqual(float((ex.block_features(altered)-ex.block_features(x))[0,0]),10.)

    def test_preprocessing_restarts_at_window_boundary_and_ignores_missing_payload(self):
        x=np.ones((1,512,2),np.float32)*8
        mask=np.ones_like(x,dtype=bool);mask[:,0,0]=False;mask[:,1,1]=False
        x[:,0,0]=np.nan;x[:,1,1]=1e8
        actual=ex.preprocess(x,mask,np.array([2,4],np.float32),np.array([2,2],np.float32))
        np.testing.assert_array_equal(actual[0,:2],np.array([[0,2],[3,2]],np.float32))

    def test_grouped_ridge_effective_float32_coefficients(self):
        rng=np.random.default_rng(2);x=rng.normal(size=(12,20));y=rng.normal(size=(12,96))
        left=Ridge(1.).fit(x[:9],y[:9,:48]);right=Ridge(.01).fit(x[2:],y[2:,48:])
        model=ex.TorchRidge([(list(range(48)),left),(list(range(48,96)),right)],20,2)
        expected=np.concatenate((left.predict(x),right.predict(x)),axis=1)
        actual=model(torch.tensor(x,dtype=torch.float32)).reshape(12,-1).numpy()
        np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=2e-6)


if __name__=='__main__':
    unittest.main()
