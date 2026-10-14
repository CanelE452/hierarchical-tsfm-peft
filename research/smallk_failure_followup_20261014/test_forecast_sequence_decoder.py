"""CPU-only class, causality and autograd examples; no real assets or F model."""
import inspect
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import forecast_sequence_decoder as sequence


def identity_channels(head):
    """GELU(u)-GELU(-u)=u constructs an explicit linear lag response."""
    with torch.no_grad():
        head.hidden.weight.zero_()
        head.hidden.bias.zero_()
        head.output.weight.zero_()
        head.output.bias.zero_()
        if isinstance(head, sequence.TemporalResidualHead):
            for channel, lag in ((0, 2), (2, 1)):
                head.hidden.weight[channel, 0, lag] = 1.
                head.hidden.weight[channel + 1, 0, lag] = -1.
                head.output.weight[channel // 2, channel, 0] = 1.
                head.output.weight[channel // 2, channel + 1, 0] = -1.
        else:
            head.hidden.weight[0, 0] = 1.
            head.hidden.weight[1, 0] = -1.
            head.output.weight[0, 0] = 1.
            head.output.weight[0, 1] = -1.


class SyntheticCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.E = nn.Linear(3, 2)
        self.D = nn.Linear(2, 3)

    def encode(self, history):
        return self.E(history)

    def decode(self, latents):
        return self.D(latents)


class SyntheticFrozenF(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(6, 4)

    def forward(self, past_z):
        return self.layer(past_z.transpose(1, 2)).transpose(1, 2)


class SequenceContractTests(unittest.TestCase):
    def test_exact_eight_locked_conditions_and_budget_without_asset_reads(self):
        with mock.patch.object(Path, 'open', side_effect=AssertionError('ASSET_READ')):
            record = sequence.describe()
        self.assertEqual(len(record['conditions']), 8)
        self.assertEqual(len({(r['dataset'], r['seed'], r['head']) for r in record['conditions']}), 8)
        self.assertFalse(record['actual_data_opened'])
        self.assertFalse(record['actual_training_executed'])
        self.assertFalse(torch.cuda.is_initialized())
        self.assertEqual(record['proposed_process_caps']['fits_gpu_process_seconds'], 1200)
        self.assertEqual(record['proposed_process_caps']['each_fit_seconds'], 150)
        self.assertIn('resource extension required', record['execution_gate'])

    def test_registered_and_active_capacity_disclosure(self):
        for dataset, expected in (('robin', (501, 497)), ('beijing_aotizhongxin', (305, 299))):
            counts = sequence.capacity_summary(dataset)
            self.assertEqual(tuple(counts['registered'][h] for h in sequence.HEADS), expected)
            self.assertLess(counts['registered_symmetric_relative_difference'], .02)
            for head in sequence.HEADS:
                module = sequence.make_head(dataset, head, 92601)
                self.assertEqual(sum(p.numel() for p in module.parameters()), counts['registered'][head])
                self.assertEqual(counts['potentially_active'][head], counts['registered'][head] - sequence.DATASETS[dataset]['C'])
        self.assertGreater(sequence.capacity_summary('beijing_aotizhongxin')['active_symmetric_relative_difference'], .02)

    def test_zero_initial_head_preserves_original_MAIN_prediction_exactly(self):
        torch.manual_seed(213)
        codec, frozen = SyntheticCodec(), SyntheticFrozenF().requires_grad_(False)
        history = torch.randn(4, 6, 3)
        for head in (sequence.StaticResidualHead(2, 3, 5), sequence.TemporalResidualHead(2, 3)):
            predicted, past, future, residual = sequence.forecast_sequence_trace(codec, frozen, head, history)
            original = history[:, -1:] + codec.decode(future) - codec.decode(past[:, -1:])
            self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))
            self.assertTrue(torch.equal(predicted, original))

    def test_constant_forecast_residual_is_exact_zero_after_nonzero_weights(self):
        torch.manual_seed(34)
        past = torch.randn(3, 9, 2)
        future = past[:, -1:].expand(-1, 48, -1)
        for head in (sequence.StaticResidualHead(2, 3, 5), sequence.TemporalResidualHead(2, 3)):
            nn.init.normal_(head.output.weight)
            nn.init.normal_(head.output.bias)
            residual = head(past, future)
            self.assertEqual(tuple(residual.shape), (3, 48, 3))
            self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))

    def test_output_bias_cancels_and_whole_anchor_has_disclosed_FP_rounding(self):
        head = sequence.StaticResidualHead(1, 1, 2)
        identity_channels(head)
        past, future = torch.zeros(1, 2, 1), torch.tensor([[[1.], [2.]]])
        original = head(past, future)
        with torch.no_grad():
            head.output.bias.fill_(3.)
        torch.testing.assert_close(head(past, future), original, atol=1e-6, rtol=0.)
        anchor, decoded = torch.tensor(1e-7), torch.tensor(1.)
        self.assertNotEqual(float(anchor + decoded - decoded), float(anchor))
        self.assertIn('FP rounding', sequence.describe()['parity']['constant_future'])

    def test_API_has_no_target_or_extra_raw_future_argument(self):
        self.assertEqual(list(inspect.signature(sequence.forecast_sequence_trace).parameters),
                         ['codec', 'frozen_forecaster', 'head', 'history'])
        for cls in (sequence.StaticResidualHead, sequence.TemporalResidualHead):
            self.assertEqual(list(inspect.signature(cls.forward).parameters), ['self', 'past_z', 'predicted_future_z'])

    def test_temporal_RF3_uses_two_prefix_points_and_no_later_forecast(self):
        head = sequence.TemporalResidualHead(1, 1)
        with torch.no_grad():
            head.hidden.weight.zero_(); head.hidden.bias.zero_()
            head.output.weight.zero_(); head.output.bias.zero_()
            head.hidden.weight[0, 0] = torch.tensor([1., 2., 3.])
            head.output.weight[0, 0, 0] = 1.
        past = torch.zeros(1, 6, 1, requires_grad=True)
        future = torch.ones(1, 48, 1, requires_grad=True)
        output = head.response(past, future)
        self.assertEqual(tuple(output.shape), (1, 48, 1))
        prefix_gradient, future_gradient = torch.autograd.grad(output[0, 0, 0], (past, future), retain_graph=True)
        self.assertTrue(torch.equal(prefix_gradient[0, :-2], torch.zeros_like(prefix_gradient[0, :-2])))
        self.assertTrue(bool((prefix_gradient[0, -2:] != 0).all()))
        self.assertNotEqual(float(future_gradient[0, 0, 0]), 0.)
        self.assertTrue(torch.equal(future_gradient[0, 1:], torch.zeros_like(future_gradient[0, 1:])))
        second_prefix, second_future = torch.autograd.grad(output[0, 2, 0], (past, future))
        self.assertTrue(torch.equal(second_prefix, torch.zeros_like(second_prefix)))
        self.assertTrue(bool((second_future[0, :3] != 0).all()))
        self.assertTrue(torch.equal(second_future[0, 3:], torch.zeros_like(second_future[0, 3:])))

    def test_pointwise_control_has_only_current_predicted_latent_support(self):
        head = sequence.StaticResidualHead(1, 1, 2)
        identity_channels(head)
        past = torch.zeros(1, 6, 1, requires_grad=True)
        future = torch.ones(1, 48, 1, requires_grad=True)
        gradient = torch.autograd.grad(head(past, future)[0, 7, 0], future)[0]
        self.assertNotEqual(float(gradient[0, 7, 0]), 0.)
        others = torch.cat((gradient[:, :7], gradient[:, 8:]), dim=1)
        self.assertTrue(torch.equal(others, torch.zeros_like(others)))

    def test_independent_innovations_toy_exhibits_lag_function_class_gap(self):
        past = torch.zeros(2, 2, 1)
        future = torch.tensor([[[-1.], [.4], [.8]], [[1.], [.4], [.8]]])
        lagged = torch.cat((past[:, -1:], future[:, :-1]), dim=1)
        target = torch.cat((future, lagged), dim=2)
        temporal = sequence.TemporalResidualHead(1, 2)
        identity_channels(temporal)
        torch.testing.assert_close(temporal(past, future), target, atol=3e-7, rtol=1e-6)
        static = sequence.StaticResidualHead(1, 2, 16)
        nn.init.normal_(static.output.weight)
        output = static(past, future)
        self.assertTrue(torch.equal(output[0, 1], output[1, 1]))
        self.assertNotEqual(float(target[0, 1, 1]), float(target[1, 1, 1]))
        common_prediction = output[:, 1, 1]
        pair_mse = (common_prediction - target[:, 1, 1]).square().mean()
        self.assertGreaterEqual(float(pair_mse.detach()), 1. - 1e-6)

    def test_invalid_latent_shapes_rejected(self):
        head = sequence.TemporalResidualHead(2, 3)
        for past, future in ((torch.zeros(2, 1, 2), torch.zeros(2, 48, 2)),
                             (torch.zeros(2, 3, 2), torch.zeros(1, 48, 2)),
                             (torch.zeros(2, 3, 2), torch.zeros(2, 48, 3))):
            with self.assertRaises(ValueError):
                head(past, future)


class SequenceGradientTests(unittest.TestCase):
    def test_masked_missing_targets_leave_finite_gradients_in_new_head_path(self):
        from run_joint_objectives_v2 import observed_macro_loss
        torch.manual_seed(119)
        codec, frozen = SyntheticCodec(), SyntheticFrozenF().requires_grad_(False)
        history = torch.randn(4, 6, 3)
        target = torch.randn(4, 4, 3)
        observed = torch.ones_like(target, dtype=torch.bool)
        observed[:, :, 2] = False
        target[:, :, 2] = float('nan')
        for head in (sequence.StaticResidualHead(2, 3, 5), sequence.TemporalResidualHead(2, 3)):
            codec.zero_grad(set_to_none=True)
            prediction, _, _, _ = sequence.forecast_sequence_trace(codec, frozen, head, history)
            loss = observed_macro_loss(prediction, target, observed)
            self.assertTrue(bool(torch.isfinite(loss)))
            loss.backward()
            self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                                for p in list(codec.parameters()) + list(head.parameters())))
            self.assertTrue(all(p.grad is None for p in frozen.parameters()))

    def test_CPU_autograd_updates_E_D_head_through_frozen_F(self):
        for head_factory in (lambda: sequence.StaticResidualHead(2, 3, 5),
                             lambda: sequence.TemporalResidualHead(2, 3)):
            torch.manual_seed(883)
            codec, frozen, head = SyntheticCodec(), SyntheticFrozenF().eval().requires_grad_(False), head_factory()
            original_f = {name: value.detach().clone() for name, value in frozen.state_dict().items()}
            initial = {group: [p.detach().clone() for p in module.parameters()]
                       for group, module in (('E', codec.E), ('D', codec.D), ('Head', head))}
            optimizer = torch.optim.AdamW(list(codec.parameters()) + list(head.parameters()), lr=.001, weight_decay=0.)
            history, target = torch.randn(4, 6, 3), torch.randn(4, 4, 3)
            for update in range(2):
                optimizer.zero_grad(set_to_none=True)
                prediction, past, future, residual = sequence.forecast_sequence_trace(codec, frozen, head, history)
                loss = (prediction - target).square().mean()
                upstream = torch.autograd.grad(loss, future, retain_graph=True)[0]
                through_F = torch.autograd.grad(future, past, grad_outputs=upstream, retain_graph=True)[0]
                self.assertTrue(bool(torch.isfinite(through_F).all()))
                self.assertGreater(float(through_F.norm()), 0.)
                loss.backward()
                for module in (codec.E, codec.D, head):
                    self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in module.parameters()))
                if update == 0:
                    self.assertTrue(torch.equal(head.hidden.weight.grad, torch.zeros_like(head.hidden.weight.grad)))
                    self.assertGreater(float(head.output.weight.grad.norm()), 0.)
                else:
                    self.assertGreater(float(head.hidden.weight.grad.norm()), 0.)
                torch.nn.utils.clip_grad_norm_(list(codec.parameters()) + list(head.parameters()), 1.)
                optimizer.step()
            for group, module in (('E', codec.E), ('D', codec.D), ('Head', head)):
                self.assertTrue(any(not torch.equal(before, after) for before, after in zip(initial[group], module.parameters())))
            for name, value in frozen.state_dict().items():
                self.assertTrue(torch.equal(value, original_f[name]))
            self.assertTrue(all(p.grad is None for p in frozen.parameters()))
            self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
