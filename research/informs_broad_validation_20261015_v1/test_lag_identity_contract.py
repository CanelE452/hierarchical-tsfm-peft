import unittest
import numpy as np
import torch
import fit_base
class LagIdentityContract(unittest.TestCase):
 def test_registered_K_equals_C_lag_initialization_is_identity(self):
  rng=np.random.default_rng(13)
  task=dict(physical=rng.normal(size=(200,3)).astype(np.float32),observed=np.ones((200,3),bool),L=64,H=8)
  for family in ('LAG_CAUSAL','LAG_BIDIR'):
   with self.subTest(family=family):
    model,norm=fit_base.create_model(dict(family=family,latent_k=3,seed=7,device='cpu'),task,120,lambda x:x[:,-8:])
    torch.testing.assert_close(model.E,torch.eye(3),atol=0,rtol=0)
    self.assertEqual(norm['basis_initialization'],'identity_at_K_equals_C')
    np.testing.assert_array_equal(norm['pca_center'],np.zeros(3,np.float32))
if __name__=='__main__':unittest.main()
