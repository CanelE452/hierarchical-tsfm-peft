"""CPU synthetic verification of coordinates, losses, filling, and readouts.

No dataset archive, checkpoint, foundation-model object, or GPU is opened.
The vectorized Windows.get/deployment fill formula is tested standalone because
those production entry points deliberately select CUDA. This is not a real
Windows.get GPU execution or a full LoRA-hook/Bolt integration test.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import unittest

import numpy as np
import torch
from torch import nn

import data as dio
import runtime as rt
import campaign as cp
import deployment as dp
import native_ridge as nr
from chronos.chronos_bolt import InstanceNorm


def _training_fill_formula(physical, observed, origins, length, mean, std):
    """Standalone CPU translation of campaign.Windows.get's index formula."""
    values = np.where(observed, (physical - mean) / std, 0).astype(np.float32)
    previous = np.maximum.accumulate(
        np.where(observed, np.arange(len(values))[:, None], -1), axis=0)
    rows = origins[:, None] + np.arange(-length, 0)[None]
    prev = previous[rows]
    x = values[np.maximum(prev, 0), np.arange(values.shape[1])]
    return np.where(prev >= (origins - length)[:, None, None], x, 0)


def _deployment_fill_formula(physical_windows, observed_windows, mean, std):
    """Standalone preprocessing formula from deployment.Route.predict_physical."""
    length = physical_windows.shape[1]
    index = np.maximum.accumulate(np.where(
        observed_windows, np.arange(length)[None, :, None], -1), axis=1)
    filled = np.take_along_axis(physical_windows, index.clip(0), axis=1)
    filled = np.where(index >= 0, filled, mean)
    return ((filled - mean) / std).astype(np.float32)


class RuntimeSyntheticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(41015)
        self.rng = np.random.default_rng(41015)

    def test_lag_coordinates_match_brute_prefix_and_forecast_mapping(self):
        z = torch.randn(2, 40, 3, dtype=torch.float64)
        future = 100 + torch.randn(2, 48, 3, dtype=torch.float64)
        actual = rt.lag_features(z, future)
        brute = torch.empty_like(actual)
        for batch in range(2):
            for h in range(48):
                for branch, lag in enumerate(rt.LAGS):
                    value = z[batch, 40-lag+h] if h < lag else future[batch, h-lag]
                    brute[batch, h, branch*3:(branch+1)*3] = value - z[batch, 39-lag]
        torch.testing.assert_close(actual, brute, atol=0, rtol=0)
        altered = future.clone()
        altered[:, 21:] += 1000
        torch.testing.assert_close(rt.lag_features(z, altered)[:, :21], actual[:, :21])

    def test_global_mask_loss_and_gradient_equal_partition_expectation(self):
        inputs = torch.randn(7, 5, 2, dtype=torch.float64)
        target = torch.randn(7, 5, 3, dtype=torch.float64)
        mask = torch.tensor(self.rng.uniform(size=target.shape) > .4)
        mask[:, :, 2] = False
        target[~mask] = 1e6
        counts = mask.sum((0, 1))
        weight = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        prediction = inputs @ weight
        full = rt.loss(prediction, target, mask, counts, 7)
        expected = torch.stack([
            (prediction[:, :, c][mask[:, :, c]] - target[:, :, c][mask[:, :, c]]).square().mean()
            for c in (0, 1)
        ]).mean()
        torch.testing.assert_close(full, expected, atol=1e-12, rtol=1e-12)
        full_grad, = torch.autograd.grad(full, weight, retain_graph=True)
        partition = prediction.new_zeros(())
        for lo, hi in ((0, 2), (2, 5), (5, 7)):
            partition = partition + (hi-lo)/7 * rt.loss(
                prediction[lo:hi], target[lo:hi], mask[lo:hi], counts, 7)
        partition_grad, = torch.autograd.grad(partition, weight, retain_graph=True)
        torch.testing.assert_close(partition, full, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(partition_grad, full_grad, atol=1e-12, rtol=1e-12)
        # Uniformly selecting one origin has the same expected gradient.
        per_origin = torch.stack([
            rt.loss(prediction[i:i+1], target[i:i+1], mask[i:i+1], counts, 7)
            for i in range(7)
        ]).mean()
        expected_grad, = torch.autograd.grad(per_origin, weight)
        torch.testing.assert_close(expected_grad, full_grad, atol=1e-12, rtol=1e-12)

    def test_all_missing_minibatch_is_zero_with_positive_global_counts(self):
        prediction = torch.randn(2, 4, 3, requires_grad=True)
        target = torch.full_like(prediction, 9000)
        mask = torch.zeros_like(prediction, dtype=torch.bool)
        loss = rt.loss(prediction, target, mask, torch.tensor([20, 11, 0]), 10)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(loss.detach()), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertEqual(torch.count_nonzero(prediction.grad).item(), 0)

    def test_entire_task_without_any_observed_label_is_undefined(self):
        # An empty global task is rejected by train's isfinite-loss guard; it
        # is not assigned a fictitious perfect zero score.
        prediction = torch.randn(2, 4, 3, requires_grad=True)
        mask = torch.zeros_like(prediction, dtype=torch.bool)
        loss = rt.loss(prediction, torch.zeros_like(prediction), mask)
        self.assertFalse(torch.isfinite(loss))

    def test_vectorized_per_origin_reset_matches_sequential_history_helper(self):
        physical = self.rng.normal(size=(31, 4)).astype(np.float32)
        observed = self.rng.uniform(size=physical.shape) > .35
        observed[:5, 0] = False
        observed[7:15, 1] = False
        observed[:, 3] = False
        observed[1, 3] = True
        physical[~observed] = np.nan
        data = {"physical": physical, "observed": observed}
        mean, std, _ = dio.fit_statistics(data, (0, 16))
        origins = np.array([8, 12, 15, 23, 31], dtype=np.int64)
        length = 7
        reference = np.stack([
            dio.history_window(data, int(o), mean, std, length)[0] for o in origins
        ])
        training = _training_fill_formula(physical, observed, origins, length, mean, std)
        histories = np.stack([physical[o-length:o] for o in origins])
        masks = np.stack([observed[o-length:o] for o in origins])
        deployed = _deployment_fill_formula(histories, masks, mean, std)
        np.testing.assert_allclose(training, reference, atol=1e-7, rtol=1e-6)
        np.testing.assert_allclose(deployed, reference, atol=1e-7, rtol=1e-6)
        # A real past observation before the window must not carry into it.
        np.testing.assert_array_equal(training[1, :, 3], 0)
        changed = physical.copy()
        changed[23:] = 1e9
        changed_fill = _training_fill_formula(changed, observed, np.array([23]), length, mean, std)
        np.testing.assert_array_equal(changed_fill[0], training[3])

    def test_predeclared_folds_purge_target_and_share_long_context_cohorts(self):
        primary = dio.make_fold(10000, fold="primary")
        secondary = dio.make_fold(10000, fold="secondary")
        self.assertEqual(primary["bounds"], {"A": (0, 6000), "B": (6000, 8000), "C": (8000, 10000)})
        self.assertEqual(secondary["bounds"], {"A": (0, 5000), "B": (5000, 7000), "C": (7000, 10000)})
        for fold in (primary, secondary):
            self.assertTrue(all(fold["purge_receipt"].values()))
            for before, after in (("A", "B"), ("B", "C")):
                self.assertLess(fold["origins"][before][-1] + 47, fold["origins"][after][0])
            common = dio.common_origins(fold, (512, 1024, 2048))
            self.assertEqual(common["A"][0], 2048)
            np.testing.assert_array_equal(common["B"], fold["origins"]["B"])
            np.testing.assert_array_equal(common["C"], fold["origins"]["C"])
        short = dio.common_origins(dio.make_fold(1593), (512, 2048))
        self.assertTrue(all(len(origins) == 0 for origins in short.values()))

    def test_adapter_gradient_crosses_frozen_toy_predictor(self):
        front = nn.Linear(3, 3, bias=False)
        adapter = rt.Adapter(front, np.zeros(3, np.float32), np.eye(3, dtype=np.float32)[:, :2])
        frozen_head = nn.Linear(40, 48, bias=False).requires_grad_(False)

        def frozen_predict(z):
            return frozen_head(z.transpose(1, 2)).transpose(1, 2)

        x = torch.randn(2, 40, 3)
        pred = adapter(x, frozen_predict)
        (pred - torch.randn_like(pred)).square().mean().backward()
        for parameter in (front.weight, adapter.E, adapter.D):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(torch.count_nonzero(parameter.grad).item(), 0)
        self.assertIsNone(frozen_head.weight.grad)

    def test_safe_instance_norm_forward_and_positive_variance_gradient_parity(self):
        for use_arcsinh in (False, True):
            with self.subTest(use_arcsinh=use_arcsinh):
                x = torch.randn(3, 19, requires_grad=True)
                reference_input = x.detach().clone().requires_grad_(True)
                safe = rt.SafeInstanceNorm(eps=1e-5, use_arcsinh=use_arcsinh)
                original = InstanceNorm(eps=1e-5, use_arcsinh=use_arcsinh)
                a, stats_a = safe(x)
                b, stats_b = original(reference_input)
                torch.testing.assert_close(a, b, atol=0, rtol=0)
                for first, second in zip(stats_a, stats_b):
                    torch.testing.assert_close(first, second, atol=0, rtol=0)
                coefficient = torch.randn_like(a)
                (a * coefficient).sum().backward()
                (b * coefficient).sum().backward()
                torch.testing.assert_close(x.grad, reference_input.grad, atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(safe.inverse(a, stats_a), x.detach(), atol=2e-6, rtol=2e-6)

    def test_safe_instance_norm_constant_forward_and_gradient_are_finite(self):
        x = torch.full((2, 16), 7., requires_grad=True)
        normalized, (_, scale) = rt.SafeInstanceNorm()(x)
        self.assertTrue(torch.isfinite(normalized).all())
        self.assertEqual(torch.count_nonzero(normalized).item(), 0)
        torch.testing.assert_close(scale, torch.full_like(scale, 1e-5))
        (normalized * torch.randn_like(normalized)).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_native_actual_y_design_chunk_fit_and_deployment_parity(self):
        z = torch.randn(31, 80, 3, dtype=torch.float64)
        decoder = torch.randn(12, 2, dtype=torch.float64)
        true_weight = torch.randn(3, 6, dtype=torch.float64)
        current = torch.randn(31, 2, dtype=torch.float64)
        lags = (0, 1, 2, 4)
        offset, design = nr.swap_design(z, decoder, lags=lags, horizon=6)
        actual_y = nr.swap_predict(z, decoder, true_weight, current, lags=lags, horizon=6)
        explicit = current[:, None] + offset + torch.einsum("bhcp,p->bhc", design, true_weight.flatten())
        torch.testing.assert_close(actual_y, explicit, atol=1e-12, rtol=1e-12)
        mask = self.rng.uniform(size=actual_y.shape) > .2
        counts = mask.sum((0, 1))
        target_delta = (actual_y - current[:, None] - offset).numpy()
        target_delta[~mask] = np.nan
        gram, rhs = np.zeros((18, 18)), np.zeros(18)
        for lo in range(0, len(z), 5):
            nr.sufficient_update(gram, rhs, design[lo:lo+5], target_delta[lo:lo+5], mask[lo:lo+5], counts)
        fitted, meta = nr.fit_from_sufficient(gram, rhs, 0)
        self.assertEqual(meta["numeric_status"], "PASS_NORMAL_EQUATIONS")
        np.testing.assert_allclose(fitted, true_weight.numpy().flatten(), atol=1e-9, rtol=1e-9)
        prediction = nr.swap_predict(z, decoder, fitted, current, lags=lags, horizon=6)
        torch.testing.assert_close(prediction, actual_y, atol=1e-9, rtol=1e-9)
        for alpha in nr.ALPHAS:
            solved, diagnostic = nr.fit_from_sufficient(np.eye(3), np.ones(3), alpha)
            np.testing.assert_allclose(solved, np.ones(3)/(1+alpha), atol=1e-12)
            self.assertEqual(diagnostic["numeric_status"], "PASS_NORMAL_EQUATIONS")

    def test_full_144_parameter_native_design_matches_h48_lag7_route(self):
        z = torch.randn(3, 512, 5, dtype=torch.float64)
        decoder = torch.randn(35, 21, dtype=torch.float64)
        weight = torch.randn(3, 48, dtype=torch.float64)
        offset, design = nr.swap_design(z, decoder)
        self.assertEqual(design.shape, (3, 48, 21, 144))
        deployed = nr.swap_predict(z, decoder, weight)
        affine = offset + torch.einsum("bhcp,p->bhc", design, weight.flatten())
        torch.testing.assert_close(deployed, affine, atol=1e-11, rtol=1e-11)

    def test_self_channel_direct_ridge_chunk_fit_preserves_declared_rank(self):
        x = torch.randn(91, 512, 2, dtype=torch.float64)
        true_weight = torch.randn(2, 33, 4, dtype=torch.float64)
        actual_y = nr.direct_predict(x, true_weight)
        past = nr.direct_features(x)
        self.assertEqual(torch.count_nonzero(past[:, :, -1]).item(), 0)
        mask = self.rng.uniform(size=actual_y.shape) > .1
        counts = mask.sum((0, 1))
        target_delta = (actual_y-x[:, -1:]).numpy()
        target_delta[~mask] = np.nan
        gram, rhs = np.zeros((2, 4, 33, 33)), np.zeros((2, 4, 33))
        for lo in range(0, len(x), 11):
            nr.direct_sufficient_update(gram, rhs, past[lo:lo+11], target_delta[lo:lo+11], mask[lo:lo+11], counts)
        fitted, meta = nr.fit_direct_from_sufficient(gram, rhs, 0)
        self.assertEqual(meta["min_rank"], 32)
        self.assertEqual(meta["numeric_unresolved_equations"], 0)
        torch.testing.assert_close(nr.direct_predict(x, fitted), actual_y, atol=1e-9, rtol=1e-9)
        # No cross-channel input can enter the independent channel readout.
        changed = x.clone()
        changed[:, :, 1] += 1000*torch.randn_like(changed[:, :, 1])
        torch.testing.assert_close(nr.direct_predict(x, fitted)[:, :, 0], nr.direct_predict(changed, fitted)[:, :, 0])

    def test_lora_pair_cp0_parity_and_two_update_gradient_opening(self):
        layer = nn.Linear(9, 7, bias=False).requires_grad_(False)
        pair = dp.LoRAPair(layer)
        x = torch.randn(5, 9)
        base = layer(x).detach()
        torch.testing.assert_close(base + pair(x), base, atol=0, rtol=0)
        initial_down = pair.down.weight.detach().clone()
        optimizer = torch.optim.SGD(pair.parameters(), lr=.03)
        target = torch.randn_like(base)
        (base + pair(x) - target).square().mean().backward()
        self.assertEqual(torch.count_nonzero(pair.down.weight.grad).item(), 0)
        self.assertGreater(torch.count_nonzero(pair.up.weight.grad).item(), 0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self.assertGreater(torch.count_nonzero(pair(x)).item(), 0)
        (base + pair(x) - target).square().mean().backward()
        self.assertGreater(torch.count_nonzero(pair.down.weight.grad).item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(pair.down.weight, initial_down))
        self.assertIsNone(layer.weight.grad)

    @unittest.skip("Full LoRA constructor requires CUDA and genuine 294912-parameter q/v topology; CPU-only test does not bypass its assertion")
    def test_full_lora_hook_close_and_real_bolt_topology(self):
        pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
