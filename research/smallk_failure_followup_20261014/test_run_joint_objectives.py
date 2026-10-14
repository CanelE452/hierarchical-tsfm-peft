"""Meaningful CPU fixtures for objective fairness and frozen-F gradient flow."""
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_joint_objectives as joint


class ObjectiveContractTests(unittest.TestCase):
    def test_import_and_describe_do_not_open_actual_assets(self):
        record = joint.describe()
        self.assertFalse(record['actual_data_opened'])
        self.assertFalse(record['GPU_executed'])

    def test_batch_order_same_for_all_objectives_and_differs_between_seeds(self):
        orders = [joint.training_order(92601, 256) for _ in joint.OBJECTIVES]
        for order in orders[1:]:
            np.testing.assert_array_equal(order, orders[0])
        self.assertEqual(orders[0].shape, (600, 4))
        self.assertTrue(np.all((orders[0] >= 0) & (orders[0] < 256)))
        self.assertFalse(np.array_equal(orders[0], joint.training_order(92602, 256)))

    def test_observed_loss_balances_channels_instead_of_pooled_samples(self):
        import torch
        prediction = torch.tensor([[[2., 3., 900.], [0., 3., 900.]]])
        target = torch.zeros_like(prediction)
        observed = torch.tensor([[[True, True, False], [False, True, False]]])
        self.assertAlmostEqual(float(joint.observed_macro_loss(prediction, target, observed)), (4. + 9.) / 2)
        self.assertNotEqual(float(joint.observed_macro_loss(prediction, target, observed)), (4. + 9. + 9.) / 3)

    def test_RAW_training_gradient_is_unchanged_by_actual_Y_values(self):
        import torch
        values = torch.tensor([[[1., 2.], [3., 4.]]])
        teacher = torch.tensor([[[.1, .2], [.3, .4]]])
        observed = torch.tensor([[[True, False], [True, True]]])
        gradients = []
        losses = []
        for actual in (torch.zeros_like(values), torch.ones_like(values) * 100):
            prediction = values.clone().requires_grad_(True)
            training, _, _ = joint.objective_losses(prediction, actual, teacher, observed, 'RAW')
            gradients.append(torch.autograd.grad(training, prediction)[0])
            losses.append(float(training.detach()))
        self.assertEqual(losses[0], losses[1])
        self.assertTrue(torch.equal(gradients[0], gradients[1]))
        self.assertEqual(float(gradients[0][0, 0, 1]), 0.)

    def test_MIXED_reports_half_losses_and_matches_mixture_target_gradient(self):
        import torch
        prediction = torch.tensor([[[1., 2.], [3., 4.]]], requires_grad=True)
        actual, teacher = torch.zeros_like(prediction), torch.ones_like(prediction) * 5
        observed = torch.ones_like(prediction, dtype=torch.bool)
        mixed, y, raw = joint.objective_losses(prediction, actual, teacher, observed, 'MIXED')
        self.assertEqual(float(mixed.detach()), .5 * float(y.detach()) + .5 * float(raw.detach()))
        mixture_loss = joint.observed_macro_loss(prediction, .5 * actual + .5 * teacher, observed)
        mixed_gradient = torch.autograd.grad(mixed, prediction, retain_graph=True)[0]
        mixture_gradient = torch.autograd.grad(mixture_loss, prediction)[0]
        self.assertTrue(torch.equal(mixed_gradient, mixture_gradient))
        self.assertNotEqual(float(mixed.detach()), float(mixture_loss.detach()))

    def test_selection_uses_actual_Y_TUNE_and_preserves_initial_earliest_tie(self):
        rows = [{'new_update': 0, 'tune_actual_Y_mse': 1., 'teacher_mse': 99.},
                {'new_update': 100, 'tune_actual_Y_mse': 2., 'teacher_mse': 0.},
                {'new_update': 200, 'tune_actual_Y_mse': 1., 'teacher_mse': 0.}]
        self.assertEqual(joint.choose_checkpoint(rows), 0)
        rows[2]['tune_actual_Y_mse'] = .9
        self.assertEqual(joint.choose_checkpoint(rows), 2)

    def test_nonfinite_selection_and_empty_observed_batch_are_rejected(self):
        import torch
        with self.assertRaisesRegex(ValueError, 'FINITE_ACTUAL_Y'):
            joint.choose_checkpoint([{'tune_actual_Y_mse': float('nan')}])
        values = torch.zeros(2, 3, 2)
        with self.assertRaisesRegex(ValueError, 'NO_OBSERVED_TARGET'):
            joint.observed_macro_loss(values, values, torch.zeros_like(values, dtype=torch.bool))

    def test_missing_target_nan_does_not_poison_backward_or_change_counts(self):
        import torch
        prediction = torch.tensor([[[1., 2., 4.], [3., 4., 5.]]], requires_grad=True)
        target = torch.tensor([[[0., float('nan'), float('nan')], [1., 1., float('nan')]]])
        observed = torch.tensor([[[True, False, False], [True, True, False]]])
        loss = joint.observed_macro_loss(prediction, target, observed)
        self.assertAlmostEqual(float(loss.detach()), ((1. + 4.) / 2. + 9.) / 2.)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(prediction.grad).all()))
        self.assertEqual(float(prediction.grad[0, 0, 1]), 0.)
        self.assertTrue(torch.equal(prediction.grad[:, :, 2], torch.zeros_like(prediction.grad[:, :, 2])))

    def test_finite_masked_loss_matches_original_macro_loss_exactly(self):
        import torch
        torch.manual_seed(714)
        prediction = torch.randn(4, 48, 17, requires_grad=True)
        target = torch.randn_like(prediction)
        observed = torch.rand_like(prediction) > .4
        observed[:, :, 2] = False
        count = observed.sum((0, 1))
        error = torch.where(observed, (prediction - target).square(), 0.).sum((0, 1))
        original = (error / count.clamp_min(1))[count > 0].mean()
        safe = joint.observed_macro_loss(prediction, target, observed)
        self.assertTrue(torch.equal(original, safe))
        original_gradient = torch.autograd.grad(original, prediction, retain_graph=True)[0]
        safe_gradient = torch.autograd.grad(safe, prediction)[0]
        self.assertTrue(torch.equal(original_gradient, safe_gradient))

    def test_parser_requires_concrete_new_fit_scope(self):
        with self.assertRaises(SystemExit):
            joint.parse_args(['--execute'])
        with self.assertRaises(SystemExit):
            joint.parse_args(['--describe', '--tag', '../existing'])
        args = joint.parse_args(['--execute', '--dataset', 'robin', '--seed', '92601',
                                '--objective', 'RAW', '--input-bridge', 'synthetic.json', '--tag', 'joint_fixture'])
        self.assertEqual(args.objective, 'RAW')


class FrozenGradientTests(unittest.TestCase):
    def models(self):
        import torch
        from torch import nn
        torch.manual_seed(912)
        class SyntheticCodec(nn.Module):
            def __init__(self):
                super().__init__()
                self.codec = nn.Module()
                self.codec.project = nn.Linear(3, 2)
                self.codec.decode = nn.Linear(2, 3)
                self.residual = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, 3))
                self.register_buffer('fixed_basis', torch.zeros(3, 2))
            def encode(self, x):
                return self.codec.project(x)
            def decode(self, z):
                return self.codec.decode(z) + self.residual(z)
        class SyntheticFrozenF(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(6, 2)
            def forward(self, z):
                return self.layer(z.transpose(1, 2)).transpose(1, 2)
        return SyntheticCodec(), SyntheticFrozenF().eval().requires_grad_(False)

    def test_actual_CPU_autograd_updates_E_D_but_freezes_F_weights(self):
        import torch
        codec, frozen = self.models()
        initial = {name: value.detach().clone() for name, value in codec.state_dict().items()}
        original_f = {name: value.detach().clone() for name, value in frozen.state_dict().items()}
        optimizer = torch.optim.AdamW(codec.parameters(), lr=.001, weight_decay=0.)
        x = torch.randn(4, 6, 3)
        actual, teacher = torch.randn(4, 2, 3), torch.randn(4, 2, 3)
        observed = torch.ones_like(actual, dtype=torch.bool)
        optimizer.zero_grad(set_to_none=True)
        prediction, z, future = joint.forward_trace(codec, frozen, x)
        loss, _, _ = joint.objective_losses(prediction, actual, teacher, observed, 'MIXED')
        proof = joint.parameter_gradient_proof(codec, loss, z, future)
        self.assertGreater(proof['frozen_F_input_gradient_norm'], 0.)
        self.assertTrue(proof['frozen_F_input_gradient_finite'])
        for group in ('E', 'D'):
            self.assertGreater(proof['E_D_gradients'][group]['norm'], 0.)
            self.assertTrue(proof['E_D_gradients'][group]['finite'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.)
        optimizer.step()
        updated = joint.parameter_change_proof(initial, codec.state_dict())
        self.assertGreater(updated['E']['changed_parameter_tensors'], 0)
        self.assertGreater(updated['D']['changed_parameter_tensors'], 0)
        self.assertTrue(torch.equal(codec.fixed_basis, initial['fixed_basis']))
        for name, value in frozen.state_dict().items():
            self.assertTrue(torch.equal(value, original_f[name]))
        self.assertTrue(all(parameter.grad is None for parameter in frozen.parameters()))

    def test_F_branch_gradient_excludes_anchor_shortcut(self):
        import torch
        codec, frozen = self.models()
        x = torch.randn(4, 6, 3)
        prediction, z, future = joint.forward_trace(codec, frozen, x)
        observed = torch.ones_like(prediction, dtype=torch.bool)
        loss = joint.observed_macro_loss(prediction, torch.zeros_like(prediction), observed)
        full_input_gradient = torch.autograd.grad(loss, z, retain_graph=True)[0]
        upstream = torch.autograd.grad(loss, future, retain_graph=True)[0]
        branch = torch.autograd.grad(future, z, grad_outputs=upstream, retain_graph=True)[0]
        proof = joint.parameter_gradient_proof(codec, loss, z, future)
        self.assertAlmostEqual(proof['frozen_F_input_gradient_norm'], float(branch.norm()), places=7)
        self.assertFalse(torch.equal(full_input_gradient, branch))

    def test_structurally_cancelled_D_bias_can_remain_unchanged(self):
        import torch
        codec, frozen = self.models()
        x = torch.randn(4, 6, 3)
        prediction, _, _ = joint.forward_trace(codec, frozen, x)
        loss = prediction.square().mean()
        loss.backward()
        torch.testing.assert_close(codec.codec.decode.bias.grad, torch.zeros_like(codec.codec.decode.bias), atol=1e-7, rtol=0.)
        torch.testing.assert_close(codec.residual[2].bias.grad, torch.zeros_like(codec.residual[2].bias), atol=1e-7, rtol=0.)
        self.assertEqual(joint.parameter_group('codec.project.weight'), 'E')
        self.assertEqual(joint.parameter_group('codec.temporal.0.weight'), 'E')
        self.assertEqual(joint.parameter_group('codec.decode.bias'), 'D')
        self.assertEqual(joint.parameter_group('residual.0.weight'), 'D')


if __name__ == '__main__':
    unittest.main()
