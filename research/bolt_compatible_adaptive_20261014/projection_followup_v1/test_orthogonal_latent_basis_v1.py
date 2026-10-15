"""CPU synthetic basis/transport/gradient checks; no actual assets, F or GPU."""
import copy
import json
import unittest

import numpy as np
import torch
from torch import nn

from causal_lag_decoder_v1 import predict
from linear_e_adaptation_v1 import LinearLagAdapter
from orthogonal_latent_basis_v1 import (OrthogonalLatentBasisAdapter,orthogonal_from_theta,
    fixed_orthogonal_controls,analytic_transport,FP64_ATOL,FP64_RTOL,FP32_ATOL,FP32_RTOL)

LAGS=(0,1,2,4,8,16,32)


def fixture(learn=False,identity=False,Q=None):
    rng=np.random.default_rng(821)
    center=rng.normal(size=6).astype(np.float32)
    e=rng.normal(scale=.2,size=(6,4)).astype(np.float32)
    d=rng.normal(scale=.1,size=(28,6)).astype(np.float32)
    x=torch.from_numpy(rng.normal(size=(2,512,6)).astype(np.float32))
    return OrthogonalLatentBasisAdapter(center,e,d,LAGS,Q,learn_Q=learn,identity=identity),x


def temporal_linear(z):
    delta=(z[:,-1:]-z[:,-2:-1])*.1
    return z[:,-1:]+torch.arange(1,49,dtype=z.dtype,device=z.device)[None,:,None]*delta


def nonlinear(z):
    return z[:,-1:].square().expand(-1,48,-1)+z[:,-2:-1].sin()


class OrthogonalLatentBasisTests(unittest.TestCase):
    def test_fixed_controls_reproducible_canonical_QR_SO_and_no_alias(self):
        a,b=fixed_orthogonal_controls(),fixed_orthogonal_controls()
        self.assertEqual(tuple(a),('IDENTITY','PERMUTATION','SIGN_FLIP','DENSE_SO_s92711','DENSE_SO_s92712'))
        for name,row in a.items():
            with self.subTest(name=name):
                np.testing.assert_array_equal(row['Q'],b[name]['Q'])
                np.testing.assert_allclose(row['Q'].T@row['Q'],np.eye(4),atol=FP64_ATOL,rtol=FP64_RTOL)
                if name.startswith('DENSE'):
                    self.assertGreater(np.linalg.det(row['Q']),0)
                    seed=row['metadata']['seed']; q,r=np.linalg.qr(np.random.default_rng(seed).normal(size=(4,4)))
                    q*=np.where(np.diag(r)<0,-1.,1.)[None,:]
                    if np.linalg.det(q)<0:q[:,-1]*=-1
                    np.testing.assert_array_equal(row['Q'],q)
                json.dumps(row['metadata'],allow_nan=False)
        self.assertLess(np.linalg.det(a['SIGN_FLIP']['Q']),0)
        a['IDENTITY']['Q'][0,0]=2
        self.assertEqual(fixed_orthogonal_controls()['IDENTITY']['Q'][0,0],1)

    def test_six_theta_FP64_exponential_is_SO_and_inverse(self):
        theta=torch.tensor([.2,-.3,.4,.1,.7,-.6],dtype=torch.float64,requires_grad=True)
        q=orthogonal_from_theta(theta)
        torch.testing.assert_close(q.T@q,torch.eye(4,dtype=torch.float64),atol=FP64_ATOL,rtol=FP64_RTOL)
        torch.testing.assert_close(orthogonal_from_theta(-theta),q.T,atol=FP64_ATOL,rtol=FP64_RTOL)
        self.assertAlmostEqual(float(torch.linalg.det(q.detach())),1.,places=10)

    def test_fixed_identity_bypasses_both_matrix_multiplies_and_preserves_bytes(self):
        model,x=fixture(identity=True); parent=LinearLagAdapter(model.center,model.E,model.D,LAGS,learn_E=False)
        seen=[]
        def observe(z): seen.append(z); return nonlinear(z)
        result,past,future=model(x,observe,return_components=True)
        self.assertIs(seen[0],past)
        expected=parent(x,nonlinear)
        self.assertEqual(result.detach().numpy().tobytes(),expected.detach().numpy().tobytes())
        self.assertEqual(future.detach().numpy().tobytes(),nonlinear(past).detach().numpy().tobytes())

    def test_learned_zero_is_not_bypassed_and_has_theta_gradient(self):
        model,x=fixture(learn=True); seen=[]
        def observe(z): seen.append(z); return nonlinear(z)
        result,past,_=model(x,observe,return_components=True)
        self.assertIsNot(seen[0],past); self.assertTrue(seen[0].requires_grad)
        result.square().mean().backward()
        self.assertIsNotNone(model.theta.grad); self.assertGreater(float(model.theta.grad.norm()),1e-7)
        self.assertFalse(model.identity); self.assertTrue(model.frozen_status()['theta_trainable'])

    def test_E_D_center_original_bytes_and_flags_survive_theta_optimizer(self):
        model,x=fixture(learn=True); original={k:model.state_dict()[k].clone() for k in ('E','D','center')}
        optimizer=torch.optim.AdamW([model.theta],lr=.02)
        model(x,nonlinear).square().mean().backward(); optimizer.step()
        self.assertGreater(float(model.theta.detach().norm()),0)
        for name,value in original.items():self.assertTrue(torch.equal(value,model.state_dict()[name]))
        self.assertIsNone(model.E.grad);self.assertIsNone(model.D.grad);model.frozen_status()

    def test_actual_F_input_rotated_old_known_past_and_D_preserved(self):
        Q=fixed_orthogonal_controls()['DENSE_SO_s92711']['Q']; model,x=fixture(Q=Q); seen=[]
        def observe(z):seen.append(z.clone());return nonlinear(z)
        result,past,future=model(x,observe,return_components=True)
        torch.testing.assert_close(seen[0],past@model.basis(),atol=0,rtol=0)
        torch.testing.assert_close(future,nonlinear(past@model.basis())@model.basis().T,atol=0,rtol=0)
        expected=predict(past.detach().numpy(),future.detach().numpy(),x[:,-1].numpy(),model.D.detach().numpy(),LAGS)
        np.testing.assert_allclose(result.detach().numpy(),expected,atol=FP32_ATOL,rtol=FP32_RTOL)
        self.assertTrue(torch.equal(past,model.encode(x)))

    def test_analytic_transport_preserves_future_and_old_decoder_with_registered_tolerance(self):
        rng=np.random.default_rng(16); u=torch.from_numpy(rng.normal(size=(2,48,4)))
        for name,row in fixed_orthogonal_controls().items():
            q=torch.from_numpy(row['Q']); restored=analytic_transport(u,q)
            torch.testing.assert_close(restored,u,atol=FP64_ATOL,rtol=FP64_RTOL)
            model,x=fixture(Q=q); p=model.encode(x); future=analytic_transport(u.float(),q.float())
            torch.testing.assert_close(model.decode(x[:,-1],p,future),model.decode(x[:,-1],p,u.float()),atol=FP32_ATOL,rtol=FP32_RTOL)

    def test_rotating_all_past_and_future_requires_block_decoder_transport(self):
        model,x=fixture(); p=model.encode(x); u=nonlinear(p); q=torch.from_numpy(fixed_orthogonal_controls()['DENSE_SO_s92712']['Q']).float()
        transported=torch.cat([q.T@block for block in model.D.split(4,dim=0)],dim=0)
        expected=model.decode(x[:,-1],p,u)
        actual=x[:,-1,None,:]+model.lag_features(p@q,u@q)@transported
        torch.testing.assert_close(actual,expected,atol=FP32_ATOL,rtol=FP32_RTOL)
        wrong=x[:,-1,None,:]+model.lag_features(p@q,u@q)@model.D
        self.assertGreater(float((wrong-expected).abs().max()),1e-3)

    def test_h0_exact_anchor_and_known_prefix_flat_future_is_not_C_persistence(self):
        model,x=fixture(); past=model.encode(x); flat=past[:,-1:].expand(-1,48,-1)
        features=model.lag_features(past,flat,include_origin=True)
        self.assertTrue(torch.equal(features[:,0],torch.zeros_like(features[:,0])))
        prediction=model.decode(x[:,-1],past,flat,include_origin=True)
        self.assertTrue(torch.equal(prediction[:,0],x[:,-1]))
        self.assertGreater(float((prediction[:,1:]-x[:,-1,None]).abs().max()),1e-3)
        for h in (1,31,32):
            expected=torch.cat([(past[:,511+h-lag] if h<=lag else flat[:,h-lag-1])-past[:,511-lag] for lag in LAGS],dim=1)
            torch.testing.assert_close(features[:,h],expected,atol=0,rtol=0)

    def test_temporal_linear_F_commutes_with_all_controls_but_nonlinear_dense_does_not(self):
        model,x=fixture(); z=model.encode(x)
        for row in fixed_orthogonal_controls().values():
            q=torch.from_numpy(row['Q']).float()
            torch.testing.assert_close(temporal_linear(z@q)@q.T,temporal_linear(z),atol=FP32_ATOL,rtol=FP32_RTOL)
        q=torch.from_numpy(fixed_orthogonal_controls()['DENSE_SO_s92711']['Q']).float()
        self.assertGreater(float((nonlinear(z@q)@q.T-nonlinear(z)).abs().max()),1e-2)

    def test_permutation_equivariant_channelwise_F_and_sign_null_control(self):
        model,x=fixture(); z=model.encode(x); controls=fixed_orthogonal_controls()
        q=torch.from_numpy(controls['PERMUTATION']['Q']).float()
        self.assertTrue(torch.equal(nonlinear(z@q)@q.T,nonlinear(z)))
        q=torch.from_numpy(controls['SIGN_FLIP']['Q']).float()
        self.assertGreater(float((nonlinear(z@q)@q.T-nonlinear(z)).abs().max()),1e-2)

    def test_theta_FP64_finite_difference_gradient_and_zero_gradient_linear_null(self):
        generator=torch.Generator().manual_seed(82); z=torch.randn((2,10,4),dtype=torch.float64,generator=generator)
        target=torch.randn((2,48,4),dtype=torch.float64,generator=generator); theta=torch.tensor([.1,-.2,.03,.07,.08,-.11],dtype=torch.float64,requires_grad=True)
        def loss(t,F):q=orthogonal_from_theta(t);return (F(z@q)@q.T-target).square().mean()
        gradient=torch.autograd.grad(loss(theta,nonlinear),theta)[0]; numerical=[]; step=1e-6
        for index in range(6):
            plus,minus=theta.detach().clone(),theta.detach().clone();plus[index]+=step;minus[index]-=step
            numerical.append((loss(plus,nonlinear)-loss(minus,nonlinear))/(2*step))
        torch.testing.assert_close(gradient,torch.stack(numerical),atol=1e-8,rtol=1e-7)
        null=torch.autograd.grad(loss(theta,temporal_linear),theta)[0]
        self.assertLess(float(null.abs().max()),FP64_ATOL)

    def test_frozen_toy_F_has_no_parameter_gradient_but_Q_gradient_is_finite(self):
        class FrozenF(nn.Module):
            def __init__(self):super().__init__();self.scale=nn.Parameter(torch.tensor([.2,.4,.6,.8]),requires_grad=False)
            def forward(self,z):return nonlinear(z)*self.scale
        F=FrozenF(); model,x=fixture(learn=True);model(x,F).square().mean().backward()
        self.assertIsNone(F.scale.grad);self.assertIsNotNone(model.theta.grad);self.assertTrue(torch.isfinite(model.theta.grad).all())
        json.dumps(model.basis_diagnostics(),allow_nan=False)

    def test_state_roundtrip_and_bad_Q_modes_or_nonfinite_rejected(self):
        model,x=fixture(learn=True)
        with torch.no_grad():model.theta.copy_(torch.tensor([.2,.1,-.1,.05,.04,.03]))
        clone,_=fixture(learn=True);clone.load_state_dict(copy.deepcopy(model.state_dict()))
        self.assertTrue(torch.equal(model(x,nonlinear),clone(x,nonlinear)))
        for kwargs in ({'Q':np.ones((4,4))},{'Q':np.diag([-1.,1.,1.,1.]),'learn':True},{'identity':True,'learn':True}):
            with self.subTest(kwargs=tuple(kwargs)),self.assertRaises(ValueError):fixture(**kwargs)
        with self.assertRaises(ValueError):orthogonal_from_theta(torch.full((6,),float('nan')))
        with self.assertRaises(ValueError):orthogonal_from_theta(torch.zeros(5))

    def test_FP32_stress_is_measured_separately_from_ordinary_tolerance(self):
        q=torch.from_numpy(fixed_orthogonal_controls()['DENSE_SO_s92711']['Q']).float()
        u=torch.tensor([[[1e8,1.,-1e8,3.]]],dtype=torch.float32)
        difference=float((analytic_transport(u,q)-u).abs().max())
        self.assertTrue(np.isfinite(difference));self.assertGreater(difference,0)


if __name__=='__main__':unittest.main()
