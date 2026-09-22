import copy
import math
import unittest
import torch
from torch import nn
from reference_core import (
    factor_rank, activation_factors, FactorizedLinear, AdapterLinear,
    anchored_raw, raw_to_native, native_loss_from_raw, response_loss,
    emloc_correct, FixedBudgetResult, common_quality_target, outcome,
)

torch.set_num_threads(1)

class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(913)

    def test_rank_saves_parameters(self):
        for m,n in [(768,768),(3072,768),(768,3072)]:
            r=factor_rank(m,n)
            self.assertLessEqual(r*(m+n),.5*m*n)

    def test_bad_compression_raises(self):
        with self.assertRaises(ValueError): factor_rank(1,5)
        with self.assertRaises(ValueError): factor_rank(5,5,1.)

    def test_full_rank_weighted_reconstruction(self):
        w=torch.randn(6,8,dtype=torch.double)
        x=torch.randn(50,8,dtype=torch.double)
        l,r=activation_factors(w,x.T@x,6)
        torch.testing.assert_close(l@r,w,atol=1e-10,rtol=1e-10)

    def test_weighted_rank_is_no_worse_than_plain_svd(self):
        w=torch.randn(7,8,dtype=torch.double)
        x=torch.randn(100,8,dtype=torch.double)*torch.arange(1,9,dtype=torch.double)
        l,r=activation_factors(w,x.T@x,3)
        u,s,vh=torch.linalg.svd(w,full_matrices=False)
        plain=(u[:,:3]*s[:3])@vh[:3]
        self.assertLessEqual(float(((w-l@r)@x.T).square().sum()),
                             float(((w-plain)@x.T).square().sum())+1e-8)

    def test_zero_moment_raises(self):
        with self.assertRaises(ValueError):
            activation_factors(torch.randn(3,4),torch.zeros(4,4),2)

    def test_factor_forward_matches_dense_and_bias(self):
        l,r=torch.randn(5,2),torch.randn(2,4)
        b=torch.randn(5); x=torch.randn(3,4)
        f=FactorizedLinear(l,r,b)
        torch.testing.assert_close(f(x),x@(l@r).T+b)

    def test_factor_scale_grad_and_freeze(self):
        f=FactorizedLinear(torch.randn(5,2),torch.randn(2,4))
        f.set_calibration(True); f(torch.randn(3,4)).square().mean().backward()
        self.assertGreater(float(f.gamma.grad.abs().sum()),0)
        f.set_calibration(False)
        self.assertFalse(f.gamma.requires_grad)

    def test_scale_clamp(self):
        f=FactorizedLinear(torch.randn(5,2),torch.randn(2,4))
        with torch.no_grad(): f.gamma.copy_(torch.tensor([-100.,100.]))
        f.clamp_calibration()
        torch.testing.assert_close(f.gamma.exp(),torch.tensor([.5,2.]))

    def test_initial_lora_matches_base(self):
        b=nn.Linear(4,5); a=AdapterLinear(b,rank=2,alpha=4.)
        x=torch.randn(7,4)
        torch.testing.assert_close(a(x),b(x),atol=0,rtol=0)
        self.assertFalse(b.weight.requires_grad)

    def test_lora_output_factor_receives_gradient_at_zero(self):
        a=AdapterLinear(nn.Linear(4,5),2,4.)
        a(torch.randn(3,4)).square().mean().backward()
        self.assertGreater(float(a.lora_B.weight.grad.abs().sum()),0)

    def test_anchor_initial_parity(self):
        f=torch.randn(2,3,4); e=torch.randn_like(f)
        torch.testing.assert_close(anchored_raw(f,e,e),f,atol=0,rtol=0)

    def test_anchor_only_live_prediction_has_gradient(self):
        f=torch.randn(2,3,4,requires_grad=True)
        e=torch.randn(2,3,4,requires_grad=True)
        e0=torch.randn(2,3,4,requires_grad=True)
        anchored_raw(f,e,e0).sum().backward()
        self.assertIsNone(f.grad); self.assertIsNone(e0.grad)
        torch.testing.assert_close(e.grad,torch.ones_like(e))

    def test_transfer_error_identity(self):
        f0,e0,f,e=[torch.randn(2,3,4,dtype=torch.double) for _ in range(4)]
        torch.testing.assert_close(f-anchored_raw(f0,e,e0),(f-f0)-(e-e0))

    def test_raw_native_round_trip(self):
        n=torch.randn(2,3,4); loc=torch.randn(2,1); s=torch.rand(2,1)+.5
        raw=n.sinh()*s[:,None,:]+loc[:,None,:]
        torch.testing.assert_close(raw_to_native(raw,loc,s),n)

    def test_nan_target_has_finite_gradients(self):
        q=torch.randn(2,3,4,requires_grad=True)
        y=torch.randn(2,4); y[0,0]=float('nan')
        loss=native_loss_from_raw(q,y,torch.zeros(2,1),torch.ones(2,1),torch.tensor([.1,.5,.9]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss)); self.assertTrue(torch.isfinite(q.grad).all())
        torch.testing.assert_close(q.grad[0,:,0],torch.zeros(3))

    def test_response_zero_when_equal(self):
        p,m=torch.randn(2,3,4),torch.randn(2,3,4)
        self.assertEqual(float(response_loss(p,m,p,m,torch.ones(2))),0.)

    def test_response_gradient_not_detached(self):
        ep=torch.randn(2,3,4,requires_grad=True); em=torch.randn(2,3,4,requires_grad=True)
        fp,fm=torch.randn_like(ep),torch.randn_like(em)
        response_loss(ep,em,fp,fm,torch.ones(2)).backward()
        self.assertGreater(float(ep.grad.abs().sum()),0.)
        self.assertGreater(float(em.grad.abs().sum()),0.)

    def test_emloc_corrects_on_active_subspace(self):
        wf,we=torch.randn(6,8,dtype=torch.double),torch.randn(6,8,dtype=torch.double)
        a,b=torch.randn(3,8,dtype=torch.double),torch.randn(6,3,dtype=torch.double)
        aa,bb=emloc_correct(wf,we,a,b,scaling=2.,correction_limit=math.inf)
        u,_,_=torch.linalg.svd(a.T,full_matrices=False)
        torch.testing.assert_close((wf+2*bb@aa)@u,(we+2*b@a)@u,atol=1e-10,rtol=1e-10)

    def test_emloc_zero_init_no_nan(self):
        w=torch.randn(6,8); a=torch.randn(3,8); b=torch.zeros(6,3)
        aa,bb=emloc_correct(w,w*.4,a,b)
        self.assertTrue(torch.isfinite(bb).all())
        torch.testing.assert_close(bb,b,atol=0,rtol=0)
        torch.testing.assert_close(aa,a,atol=0,rtol=0)

    def test_emloc_zero_limit_preserves_update(self):
        wf,we=torch.randn(6,8,dtype=torch.double),torch.randn(6,8,dtype=torch.double)
        a,b=torch.randn(3,8,dtype=torch.double),torch.randn(6,3,dtype=torch.double)
        aa,bb=emloc_correct(wf,we,a,b,correction_limit=0.)
        torch.testing.assert_close(bb@aa,b@a,atol=1e-10,rtol=1e-10)

    def test_same_weights_transfer_remains_same(self):
        w=torch.randn(6,8,dtype=torch.double); a=torch.randn(3,8,dtype=torch.double)
        b=torch.randn(6,3,dtype=torch.double)
        aa,bb=emloc_correct(w,w,a,b)
        torch.testing.assert_close(bb@aa,b@a,atol=1e-10,rtol=1e-10)

    def test_target_uses_best_not_deteriorated_last(self):
        t,already=common_quality_target([.25,.24,.26],.25)
        self.assertAlmostEqual(t,.24*1.005)
        self.assertFalse(already)

    def test_prep_charged_and_no_hindsight_shortcut(self):
        r=FixedBudgetResult('x',1,16,.2,90.,20.,3.)
        self.assertEqual(r.total_seconds,113.)

    def test_go_is_quality_and_cost_and_added_value(self):
        def pts(a,e,t):return [FixedBudgetResult(a,s,64,e,t,0.,0.) for s in [1,2]]
        d=outcome(pts('FULL',.175,100),pts('RESPONSE',.1751,60),
                  {'VALUE_DELTA':pts('VALUE',.177,55)})
        self.assertEqual(d['status'],'GO_METHOD_SCREEN')

    def test_dominated_is_not_new_method_success(self):
        def pts(a,e,t):return [FixedBudgetResult(a,s,64,e,t,0.,0.) for s in [1,2]]
        d=outcome(pts('FULL',.175,100),pts('RESPONSE',.1751,70),
                  {'VALUE_DELTA':pts('VALUE',.1750,60)})
        self.assertEqual(d['status'],'GO_STANDARD_ONLY')

if __name__=='__main__': unittest.main(verbosity=2)
