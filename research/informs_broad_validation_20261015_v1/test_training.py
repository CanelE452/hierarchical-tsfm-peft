"""Synthetic indexing, macro weights and exact training-resume regression."""
import unittest
import numpy as np
import torch
try:
 from training import PermutationStream, FastWindows, macro_loss, verify_contract
except ImportError:
 PermutationStream=FastWindows=macro_loss=verify_contract=None

class TrainingTests(unittest.TestCase):
 def test_permutation_resume_and_no_skipped_tail(self):
  self.assertIsNotNone(PermutationStream,'training primitive not implemented')
  source=np.arange(31,dtype=np.int64)
  a=PermutationStream(source,9);first=a.draw(7);state=a.state_dict();expected=a.draw(55)
  b=PermutationStream(source,9);b.load_state_dict(state)
  np.testing.assert_array_equal(expected,b.draw(55))
  c=PermutationStream(source,9);whole=c.draw(31)
  self.assertEqual(set(whole),set(source));self.assertEqual(len(first),7)
 def test_window_reset_never_uses_pre_window_or_future(self):
  self.assertIsNotNone(FastWindows,'training primitive not implemented')
  task={'physical':np.array([[99.,0],[0,3],[0,0],[5,0],[0,8],[7,0]],np.float32),'observed':np.array([[1,1],[0,1],[0,0],[1,0],[0,1],[1,0]],bool),'L':3,'H':2}
  w=FastWindows(task,np.array([1,2],np.float32),np.ones(2,np.float32),'cpu')
  x,y,m=w.get(np.array([4]))
  np.testing.assert_array_equal(x.numpy(),[[[0,1],[0,1],[4,1]]])
  self.assertEqual(float(y[0,0,1]),6)
  self.assertTrue(bool(m[0,0,1]));self.assertFalse(bool(m[0,0,0]))
 def test_global_macro_loss_unbiased_partition_and_gradient(self):
  self.assertIsNotNone(macro_loss,'training primitive not implemented')
  p=torch.randn(5,3,2,dtype=torch.float64,requires_grad=True);y=torch.randn_like(p);m=torch.ones_like(p,dtype=torch.bool);m[:3,:,1]=False
  counts=m.sum((0,1));whole=macro_loss(p,y,m,counts,5,5)
  parts=sum(macro_loss(p[i:i+1],y[i:i+1],m[i:i+1],counts,5,5) for i in range(5))
  torch.testing.assert_close(whole,parts)
  expected=torch.stack([(p[:,:,c][m[:,:,c]]-y[:,:,c][m[:,:,c]]).square().mean() for c in range(2)]).mean()
  torch.testing.assert_close(whole,expected)
  torch.testing.assert_close(torch.autograd.grad(whole,p,retain_graph=True)[0],torch.autograd.grad(parts,p)[0])
 def test_contract_change_refuses_resume(self):
  self.assertIsNotNone(verify_contract,'training primitive not implemented')
  a={'data_sha':'x','source_sha':'a','config_sha':'b','model_sha':'c','scaler_sha':'d'}
  verify_contract(a,a)
  for key in a:
   b=dict(a);b[key]='CHANGED'
   with self.subTest(key=key), self.assertRaises(ValueError):verify_contract(a,b)

if __name__=='__main__':unittest.main()
