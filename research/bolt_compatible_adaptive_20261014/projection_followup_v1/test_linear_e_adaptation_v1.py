"""CPU generated fixtures for linear E/lag D and gradient-path accounting."""
import itertools
import json
import unittest

import numpy as np
import torch

from causal_lag_decoder_v1 import lagged_delta_features, predict

try:
    import linear_e_adaptation_v1 as adaptation
except ModuleNotFoundError as error:
    if error.name != "linear_e_adaptation_v1": raise
    adaptation = None


class LinearEAdaptationTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(adaptation, "LinearLagAdapter implementation has not been written")
        return adaptation

    def adapter(self, learn=True, lags=(0, 1)):
        api = self.api()
        e = np.array([[1.], [0.]], np.float32)
        d = np.zeros((len(lags), 2), np.float32); d[0, 0] = 1.
        if len(lags) > 1: d[1, 1] = 1.
        return api.LinearLagAdapter(np.zeros(2, np.float32), e, d, lags, learn_E=learn)

    def test_delayed_copy_and_known_prefix_use_origin_reference(self):
        model = self.adapter()
        first = torch.arange(40, dtype=torch.float32)
        x = torch.stack((first, torch.cat((first[:1], first[:-1]))), dim=-1)[None]
        z = model.encode(x); future = torch.tensor([[[40.], [41.], [42.]]])
        expected = torch.tensor([[[40., 39.], [41., 40.], [42., 41.]]])
        self.assertTrue(torch.equal(model.decode(x[:, -1], z, future), expected))

    def test_flat_future_does_not_remove_known_prefix_effect(self):
        model = self.adapter(); x = torch.stack((torch.arange(40.), torch.arange(40.)-1), dim=-1)[None]
        z = model.encode(x); future = z[:, -1:].expand(-1, 3, -1)
        actual = model.decode(x[:, -1], z, future)
        self.assertEqual(float(actual[0, 0, 1].detach()), 39.)
        self.assertFalse(torch.equal(actual, x[:, -1:, :].expand_as(actual)))

    def test_origin_zero_and_numpy_lag_parity(self):
        api = self.api(); rng = np.random.default_rng(937)
        x = rng.normal(size=(3, 512, 4)).astype(np.float32)
        e = rng.normal(size=(4, 2)).astype(np.float32); d = rng.normal(size=(14, 4)).astype(np.float32)
        model = api.LinearLagAdapter(np.zeros(4, np.float32), e, d, (0, 1, 2, 4, 8, 16, 32))
        z = model.encode(torch.from_numpy(x)); future = torch.from_numpy(rng.normal(size=(3, 48, 2)).astype(np.float32))
        features = model.lag_features(z, future)
        self.assertTrue(np.array_equal(features.detach().numpy(), lagged_delta_features(z.detach().numpy(), future.numpy(), model.lags)))
        actual = model.decode(torch.from_numpy(x[:, -1].copy()), z, future)
        np.testing.assert_allclose(actual.detach().numpy(), predict(z.detach().numpy(), future.numpy(), x[:, -1], d, model.lags), atol=1e-5, rtol=2e-6)
        origin = model.decode(torch.from_numpy(x[:, -1].copy()), z, future, include_origin=True)
        self.assertTrue(torch.equal(origin[:, 0], torch.from_numpy(x[:, -1])))
        self.assertTrue(torch.equal(model.lag_features(z, future, include_origin=True)[:, 0], torch.zeros((3, 14))))

    def test_full_FIT_weighted_loss_is_unbiased_across_minibatches(self):
        api = self.api()
        prediction = torch.tensor([[[1., 2., 9.]], [[2., 4., 9.]], [[3., 8., 9.]], [[4., 16., 9.]]])
        target = torch.zeros_like(prediction)
        observed = torch.tensor([[[1, 1, 0]], [[1, 0, 0]], [[1, 0, 0]], [[1, 0, 0]]], dtype=torch.bool)
        counts = observed.sum((0, 1))
        full = api.global_fit_macro_mse(prediction, target, observed, counts, 4)
        sampled = [api.global_fit_macro_mse(prediction[list(indices)], target[list(indices)], observed[list(indices)], counts, 4)
            for indices in itertools.combinations(range(4), 2)]
        self.assertAlmostEqual(float(full), float(torch.stack(sampled).mean()), places=6)
        self.assertAlmostEqual(float(full), ((1+4+9+16)/4+4)/2, places=6)

    def test_unobserved_NaN_has_zero_finite_gradient(self):
        api = self.api(); prediction = torch.tensor([[[2., 3.]]], requires_grad=True)
        target = torch.tensor([[[float("nan"), 1.]]]); observed = torch.tensor([[[False, True]]])
        loss = api.global_fit_macro_mse(prediction, target, observed, torch.tensor([0, 1]), 1)
        loss.backward(); self.assertEqual(float(loss.detach()), 4.)
        self.assertTrue(torch.equal(prediction.grad, torch.tensor([[[0., 4.]]])))

    def test_empty_global_channels_and_observed_nonfinite_are_explicit(self):
        api = self.api(); p = torch.ones(1, 2, 2); y = torch.zeros_like(p); m = torch.ones_like(p, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "ACTIVE_FIT_CHANNEL"): api.global_fit_macro_mse(p,y,torch.zeros_like(m),torch.zeros(2,dtype=torch.int64),1)
        with self.assertRaisesRegex(ValueError, "GLOBALLY_EMPTY"): api.global_fit_macro_mse(p,y,m,torch.tensor([0,2]),1)
        y[0,0,0] = float("nan")
        with self.assertRaisesRegex(ValueError, "OBSERVED_TARGET"): api.global_fit_macro_mse(p,y,m,torch.tensor([2,2]),1)

    def test_gradient_F_and_direct_paths_sum_without_consuming_backward(self):
        api = self.api(); model = self.adapter(); x = torch.arange(24,dtype=torch.float32).reshape(2,6,2)/10
        class ToyFrozenF(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.scale = torch.nn.Parameter(torch.tensor(1.7), requires_grad=False)
            def forward(self,z): return self.scale*z[:, -1:].expand(-1,3,-1)+.2*z.mean(1,keepdim=True)
        F = ToyFrozenF(); prediction,z,f = model(x,F,return_components=True); loss = prediction.square().mean()
        diagnostic = api.gradient_diagnostics(model,loss,z,f)
        self.assertTrue(diagnostic["F_path_available"])
        self.assertGreater(diagnostic["E_F_path_gradient_norm"],0);self.assertGreater(diagnostic["E_direct_past_gradient_norm"],0)
        self.assertLess(diagnostic["E_gradient_sum_residual_max_abs"],1e-5)
        self.assertIsNone(model.E.grad);self.assertIsNone(model.D.grad)
        loss.backward();self.assertIsNotNone(model.E.grad);self.assertIsNone(F.scale.grad)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(model.E.grad)),diagnostic["E_total_gradient_norm"],places=5)
        json.dumps(diagnostic,allow_nan=False)

    def test_detached_F_is_identified_despite_nonzero_direct_E_gradient(self):
        api = self.api(); model = self.adapter(); x = torch.arange(24,dtype=torch.float32).reshape(2,6,2)/10
        p,z,f = model(x,lambda z: (2*z[:, -1:].expand(-1,3,-1)).detach(),return_components=True)
        diagnostic = api.gradient_diagnostics(model,p.square().mean(),z,f)
        self.assertFalse(diagnostic["F_path_available"]);self.assertIsNone(diagnostic["E_F_path_gradient_norm"])
        self.assertGreater(diagnostic["E_total_gradient_norm"],0)
        p.square().mean().backward();self.assertIsNotNone(model.E.grad)

    def test_frozen_E_has_no_gradient_and_CP0_matches_learning_mode(self):
        fixed,learned = self.adapter(False),self.adapter(True);x = torch.arange(24,dtype=torch.float32).reshape(2,6,2)/10
        F = lambda z: 1.2*z[:, -1:].expand(-1,3,-1)
        one,two = fixed(x,F),learned(x,F)
        self.assertTrue(torch.equal(one.view(torch.int32),two.view(torch.int32)))
        one.square().mean().backward();self.assertIsNone(fixed.E.grad);self.assertIsNotNone(fixed.D.grad)
        self.assertFalse(fixed.center.requires_grad);self.assertEqual(set(fixed.state_dict()),{"E","D","center"})

    def test_forward_and_decode_do_not_accept_future_targets(self):
        model = self.adapter();x = torch.zeros(1,40,2);F = lambda z: z[:,-1:].expand(-1,3,-1)
        with self.assertRaises(TypeError):model(x,F,target=torch.zeros(1,3,2))
        with self.assertRaises(TypeError):model.decode(x[:,-1],model.encode(x),F(model.encode(x)),target=torch.zeros(1,3,2))

    def test_parameter_diagnostics_are_finite_and_gain_is_feature_operator(self):
        api = self.api();model = self.adapter();z = model.encode(torch.ones(2,40,2))
        diagnostic = api.parameter_diagnostics(model,z)
        self.assertTrue(diagnostic["finite"]);self.assertEqual(diagnostic["latent_variance"],[0.])
        self.assertAlmostEqual(diagnostic["D_feature_to_C_operator_gain"],1.,places=6)
        self.assertEqual(len(diagnostic["E_singularvalues"]),1);json.dumps(diagnostic,allow_nan=False)
        with torch.no_grad():model.D[0,0] = float("nan")
        with self.assertRaisesRegex(ValueError,"NONFINITE"):api.parameter_diagnostics(model,z)

    def test_shape_lag_and_nonfinite_input_contracts(self):
        api = self.api()
        with self.assertRaises(ValueError):api.LinearLagAdapter([0.,0.],[[1.],[0.]],[[1.,1.]],(1,0))
        model = self.adapter()
        with self.assertRaises(ValueError):model.encode(torch.ones(1,3,3))
        with self.assertRaises(ValueError):model.encode(torch.full((1,3,2),float("nan")))
        with self.assertRaises(ValueError):model.decode(torch.ones(1,2),torch.ones(1,3,1),torch.ones(1,2,2))


if __name__ == "__main__": unittest.main()
