"""CPU-only synthetic checks; no payload, foundation model, or GPU access."""

import unittest

import torch

try:
    from models import DirectMixer, TemporalFront, parameter_count
except ModuleNotFoundError:
    DirectMixer = TemporalFront = parameter_count = None


class TemporalModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        torch.set_num_threads(1)
        self.assertIsNotNone(TemporalFront, "TemporalFront has not been implemented")

    def test_cp0_preserves_original_grid_and_input_gradient(self):
        for kind in ("B0", "B1", "B2", "B3", "B4"):
            with self.subTest(kind=kind):
                model = TemporalFront(kind, length=16, channels=3, hidden=5, kernel=3)
                x = torch.randn(2, 16, 3, requires_grad=True)
                coefficient = torch.randn_like(x)
                y = model(x)
                self.assertEqual(y.shape, x.shape)
                torch.testing.assert_close(y, x, atol=1e-7, rtol=1e-6)
                (y * coefficient).sum().backward()
                torch.testing.assert_close(x.grad, coefficient)
        self.assertEqual(parameter_count(TemporalFront("B0", length=16, channels=3)), 0)

    def test_zero_output_layers_open_gradients_after_first_update(self):
        for kind in ("B1", "B2", "B3", "B4"):
            with self.subTest(kind=kind):
                model = TemporalFront(kind, length=16, channels=3, hidden=5, kernel=3)
                initial_zero_names = {
                    name for name, p in model.named_parameters() if torch.count_nonzero(p) == 0
                }
                self.assertTrue(initial_zero_names)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
                x = torch.randn(3, 16, 3)
                target = torch.randn_like(x)
                (model(x) - target).square().mean().backward()
                for name, p in model.named_parameters():
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(torch.isfinite(p.grad).all(), name)
                    if name in initial_zero_names:
                        self.assertGreater(torch.count_nonzero(p.grad).item(), 0, name)
                    else:
                        self.assertEqual(torch.count_nonzero(p.grad).item(), 0, name)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                (model(x) - target).square().mean().backward()
                for name, p in model.named_parameters():
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(torch.isfinite(p.grad).all(), name)
                    self.assertGreater(torch.count_nonzero(p.grad).item(), 0, name)

    def test_b1_keeps_channel_independence_after_learning(self):
        model = TemporalFront("B1", length=16, channels=3, hidden=5)
        x = torch.randn(2, 16, 3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        (model(x) - torch.randn_like(x)).square().mean().backward()
        optimizer.step()
        changed = x.clone()
        changed[:, :, 0] += 2
        torch.testing.assert_close(model(x)[:, :, 1:], model(changed)[:, :, 1:])
        permutation = torch.tensor([2, 0, 1])
        torch.testing.assert_close(model(x[:, :, permutation]), model(x)[:, :, permutation])

    def test_multiscale_pooling_and_symmetric_decomposition_are_explicit(self):
        model = TemporalFront("B4", length=8, channels=1, hidden=3, kernel=3)
        x = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1)
        scales = model.multiscale.scale_inputs(x)
        torch.testing.assert_close(scales[1].flatten(), torch.tensor([0.5, 2.5, 4.5, 6.5]))
        torch.testing.assert_close(scales[2].flatten(), torch.tensor([1.5, 5.5]))
        residual, trend = model.multiscale.decompose(x)
        torch.testing.assert_close(trend.flatten(), torch.tensor([1/3, 1., 2., 3., 4., 5., 6., 20/3]))
        torch.testing.assert_close(residual + trend, x)

    def test_cross_scale_dependencies_follow_the_declared_direction(self):
        for bottom_up in (True, False):
            core = TemporalFront("B4", length=16, channels=3, hidden=5, kernel=3,
                                 decomposition=False, seasonal_mix=bottom_up,
                                 trend_mix=not bottom_up).multiscale
            scales = [torch.randn(2, n, 3, requires_grad=True) for n in (16, 8, 4)]
            # Independent scale-boundary inputs isolate mixing from pooling.
            core.scale_inputs = lambda _: scales
            outputs = core(None)
            if bottom_up:
                allowed = torch.autograd.grad(outputs[-1].sum(), scales[0], retain_graph=True)[0]
                forbidden = torch.autograd.grad(outputs[0].sum(), scales[-1], allow_unused=True)[0]
            else:
                allowed = torch.autograd.grad(outputs[0].sum(), scales[-1], retain_graph=True)[0]
                forbidden = torch.autograd.grad(outputs[-1].sum(), scales[0], allow_unused=True)[0]
            self.assertGreater(torch.count_nonzero(allowed).item(), 0)
            self.assertIsNone(forbidden)

    def test_b3_is_exact_capacity_matched_and_b4_ablation_removes_live_edges(self):
        common = dict(length=16, channels=3, hidden=5, kernel=3)
        b3 = TemporalFront("B3", **common)
        b4 = TemporalFront("B4", **common)
        self.assertEqual(parameter_count(b3), parameter_count(b4))
        no_decomposition = TemporalFront("B4", decomposition=False, **common)
        no_decomposition.load_state_dict(b3.state_dict())
        with torch.no_grad():
            for p in b3.parameters():
                p.add_(0.01 * torch.randn_like(p))
        no_decomposition.load_state_dict(b3.state_dict())
        x = torch.randn(2, 16, 3)
        torch.testing.assert_close(b3(x), no_decomposition(x))
        for option in ("seasonal_mix", "trend_mix"):
            ablated = TemporalFront("B4", **common, **{option: False})
            self.assertLess(parameter_count(ablated), parameter_count(b4))
            optimizer = torch.optim.SGD(ablated.parameters(), lr=0.03)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                (ablated(x) - 1).square().mean().backward()
                optimizer.step()
            self.assertTrue(all(p.grad is not None for p in ablated.parameters()))

    def test_multiscale_mapping_uses_no_future_or_padded_time_concatenation(self):
        for kind in ("B3", "B4"):
            model = TemporalFront(kind, length=16, channels=3, hidden=5)
            with torch.no_grad():
                for p in model.parameters():
                    if torch.count_nonzero(p) == 0:
                        p.normal_(std=0.1)
            x = torch.randn(2, 16, 3)
            altered = x.clone()
            altered[:, -1] += 5
            self.assertEqual(model(x).shape, (2, 16, 3))
            self.assertFalse(torch.allclose(model(x)[:, 0], model(altered)[:, 0]))
            torch.testing.assert_close(model(x)[0], model(x[:1])[0])

    def test_direct_mixers_predict_horizon_with_live_trainable_parameters(self):
        for kind in ("B2", "B4"):
            model = DirectMixer(kind, length=16, horizon=7, channels=3, hidden=5, kernel=3)
            x = torch.randn(2, 16, 3, requires_grad=True)
            y = model(x)
            self.assertEqual(y.shape, (2, 7, 3))
            if kind == "B4":
                y.square().mean().backward()
                for name, p in model.named_parameters():
                    self.assertGreater(torch.count_nonzero(p.grad).item(), 0, name)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                (model(x) - 1).square().mean().backward()
                optimizer.step()
            self.assertTrue(torch.isfinite(x.grad).all())
            for name, p in model.named_parameters():
                self.assertIsNotNone(p.grad, name)
                self.assertGreater(torch.count_nonzero(p.grad).item(), 0, name)

    def test_invalid_grid_or_kind_is_rejected(self):
        for kwargs in ({"kind": "B9"}, {"kind": "B4", "scales": (1, 3)},
                       {"kind": "B4", "kernel": 4}):
            with self.assertRaises(ValueError):
                TemporalFront(length=16, channels=3, **kwargs)
        model = TemporalFront("B1", length=16, channels=3)
        with self.assertRaises(ValueError):
            model(torch.randn(2, 15, 3))


if __name__ == "__main__":
    unittest.main()
