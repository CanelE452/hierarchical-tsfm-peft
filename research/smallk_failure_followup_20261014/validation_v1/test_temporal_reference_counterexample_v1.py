"""CPU synthetic counterexamples only; no scientific data, CP, F, or training.

Prepared during the live approved eight fits. Do not execute concurrently with
their GPU learning/cost measurements. A toy pass is not real-data improvement.
"""
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import forecast_sequence_decoder as frozen_sequence
from temporal_origin_reference_v1 import origin_reference_residual


def linear_response_head(taps):
    """Exact linear response via GELU(s)-GELU(-s)=s, without training."""
    outputs = len(taps)
    head = frozen_sequence.TemporalResidualHead(1, outputs, width=2 * outputs).double()
    with torch.no_grad():
        head.hidden.weight.zero_()
        head.hidden.bias.zero_()
        head.output.weight.zero_()
        head.output.bias.zero_()
        for channel, weights in enumerate(taps):
            values = torch.tensor(weights, dtype=torch.float64)
            head.hidden.weight[2 * channel, 0] = values
            head.hidden.weight[2 * channel + 1, 0] = -values
            head.output.weight[channel, 2 * channel, 0] = 1.
            head.output.weight[channel, 2 * channel + 1, 0] = -1.
    return head


class TemporalReferenceCounterexampleTests(unittest.TestCase):
    def test_flat_future_current_reference_erases_known_one_step_lag_shift(self):
        head = linear_response_head(((0., 0., 1.), (0., 1., 0.)))
        past = torch.tensor([[[-1.], [2.], [5.]]], dtype=torch.float64)
        future = torch.full((1, 3, 1), 5., dtype=torch.float64)
        raw_current = torch.tensor([[[5., 2.]]], dtype=torch.float64)
        base = raw_current.expand(-1, 3, -1)
        true_future = torch.full((1, 3, 2), 5., dtype=torch.float64)
        current = head(past, future)
        alternative = origin_reference_residual(head, past, future)
        self.assertTrue(torch.equal(current, torch.zeros_like(current)))
        torch.testing.assert_close(alternative, torch.tensor([[[0., 3.]]], dtype=torch.float64).expand(-1, 3, -1),
                                   atol=1e-12, rtol=0.)
        torch.testing.assert_close(base + alternative, true_future, atol=1e-12, rtol=0.)
        self.assertEqual(float((base + current - true_future).square().mean()), 4.5)
        self.assertLessEqual(float((base + alternative - true_future).square().mean()), 1e-24)

    def test_linear_current_reference_cancels_actual_prefix_terms(self):
        head = linear_response_head(((1., 10., 100.),))
        past_a = torch.tensor([[[7.], [2.], [5.]]], dtype=torch.float64)
        past_b = torch.tensor([[[-9.], [100.], [5.]]], dtype=torch.float64)
        future = torch.tensor([[[6.], [8.], [9.]]], dtype=torch.float64)
        expected = torch.tensor([[[100.], [310.], [431.]]], dtype=torch.float64)
        torch.testing.assert_close(head(past_a, future), expected, atol=1e-10, rtol=0.)
        torch.testing.assert_close(head(past_b, future), expected, atol=1e-10, rtol=0.)

    def test_nonlinear_current_reference_can_modulate_nonzero_innovations_by_past(self):
        head = frozen_sequence.TemporalResidualHead(1, 1, width=1).double()
        with torch.no_grad():
            head.hidden.weight.copy_(torch.tensor([[[1., 0., 1.]]], dtype=torch.float64))
            head.hidden.bias.zero_()
            head.output.weight.fill_(1.)
            head.output.bias.zero_()
        past_a = torch.tensor([[[0.], [-5.], [5.]]], dtype=torch.float64)
        past_b = torch.tensor([[[0.], [5.], [5.]]], dtype=torch.float64)
        nonflat = torch.tensor([[[6.]]], dtype=torch.float64)
        self.assertGreater(float((head(past_a, nonflat) - head(past_b, nonflat)).abs().max()), .1)
        flat = torch.tensor([[[5.]]], dtype=torch.float64)
        self.assertTrue(torch.equal(head(past_a, flat), torch.zeros_like(flat)))
        self.assertTrue(torch.equal(head(past_b, flat), torch.zeros_like(flat)))

    def test_global_constant_past_and_future_preserve_current_anchor_in_both_references(self):
        head = linear_response_head(((1., 10., 100.), (0., 1., 0.)))
        past = torch.full((2, 6, 1), 5., dtype=torch.float64)
        future = torch.full((2, 4, 1), 5., dtype=torch.float64)
        for residual in (head(past, future), origin_reference_residual(head, past, future)):
            self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))

    def test_origin_reference_retains_actual_past_shift_with_no_future_innovation(self):
        head = linear_response_head(((1., 0., 0.),))
        past = torch.tensor([[[-1.], [2.], [5.]]], dtype=torch.float64)
        future = torch.full((1, 3, 1), 5., dtype=torch.float64)
        torch.testing.assert_close(origin_reference_residual(head, past, future),
                                   torch.tensor([[[3.], [6.], [6.]]], dtype=torch.float64), atol=1e-12, rtol=0.)
        self.assertTrue(torch.equal(head(past, future), torch.zeros((1, 3, 1), dtype=torch.float64)))

    def test_origin_reference_is_causal_and_has_no_later_forecast_support(self):
        head = linear_response_head(((1., 10., 100.),))
        past = torch.tensor([[[7.], [2.], [5.]]], dtype=torch.float64, requires_grad=True)
        future = torch.tensor([[[6.], [8.], [9.], [11.]]], dtype=torch.float64, requires_grad=True)
        residual = origin_reference_residual(head, past, future)
        past_gradient, future_gradient = torch.autograd.grad(residual[0, 0, 0], (past, future))
        torch.testing.assert_close(past_gradient, torch.tensor([[[-1.], [-9.], [-90.]]], dtype=torch.float64),
                                   atol=1e-12, rtol=0.)
        torch.testing.assert_close(future_gradient, torch.tensor([[[100.], [0.], [0.], [0.]]], dtype=torch.float64),
                                   atol=1e-12, rtol=0.)

    def test_zero_output_initialization_preserves_original_prediction_for_origin_reference(self):
        head = frozen_sequence.TemporalResidualHead(1, 2, width=4).double()
        past = torch.tensor([[[-1.], [2.], [5.]]], dtype=torch.float64)
        future = torch.tensor([[[6.], [8.], [9.]]], dtype=torch.float64)
        residual = origin_reference_residual(head, past, future)
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))

    def test_origin_reference_requires_three_actual_past_latents(self):
        head = frozen_sequence.TemporalResidualHead(1, 2, width=4).double()
        with self.assertRaisesRegex(ValueError, 'THREE_ACTUAL_PAST'):
            origin_reference_residual(head, torch.zeros(1, 2, 1, dtype=torch.float64),
                                      torch.zeros(1, 3, 1, dtype=torch.float64))

    def test_scope_is_cpu_only(self):
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
