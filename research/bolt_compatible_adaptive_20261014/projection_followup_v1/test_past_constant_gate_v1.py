"""Small analytic controls; no actual time series, model or checkpoint IO."""
import inspect
import unittest

import numpy as np

from past_constant_gate_v1 import past_constant_gate, apply_output_gate


class PastConstantGateTests(unittest.TestCase):
    def values(self):
        history = np.zeros((2, 512, 3), dtype=np.float32)
        history[:, :, 1] = np.arange(512, dtype=np.float32)
        history[:, :, 2] = 7.
        observed = np.ones(history.shape, dtype=bool)
        return history, observed

    def test_flat_only_with_at_least_two_true_observations(self):
        history, observed = self.values()
        observed[0, :, 0] = False
        observed[1, :, 2] = False
        observed[1, 5, 2] = True
        observed[0, :, 2] = False
        observed[0, [7, 27], 2] = True
        gate = past_constant_gate(history, observed)
        np.testing.assert_array_equal(gate["keep_forecast"], [[True, True, False], [False, True, True]])
        self.assertEqual(gate["true_observed_counts"][0, 2], 2)

    def test_one_FP32_ulp_change_prevents_gate(self):
        history, observed = self.values()
        history[0, 10, 2] = np.nextafter(np.float32(7.), np.float32(8.))
        self.assertTrue(past_constant_gate(history, observed)["keep_forecast"][0, 2])

    def test_unknown_filled_labels_do_not_count_as_observed(self):
        history = np.ones((1, 512, 1), dtype=np.float32)
        observed = np.zeros(history.shape, dtype=bool)
        self.assertTrue(past_constant_gate(history, observed)["keep_forecast"][0, 0])
        observed[:, [0, 511]] = True
        self.assertFalse(past_constant_gate(history, observed)["keep_forecast"][0, 0])

    def test_unchanged_coordinates_preserve_bits_including_signed_zero(self):
        history, observed = self.values()
        gate = past_constant_gate(history, observed)["keep_forecast"]
        prediction = np.arange(2 * 48 * 3, dtype=np.float32).reshape(2, 48, 3)
        prediction[:, :, 1] = np.float32(-0.)
        current = history[:, -1].copy()
        result = apply_output_gate(prediction, current, gate)
        kept = np.broadcast_to(gate[:, None], prediction.shape)
        np.testing.assert_array_equal(result.view(np.uint32)[kept], prediction.view(np.uint32)[kept])
        expected = np.broadcast_to(current[:, None], prediction.shape)
        np.testing.assert_array_equal(result.view(np.uint32)[~kept], expected.view(np.uint32)[~kept])

    def test_no_eligible_channels_returns_entire_prediction_byte_exact(self):
        history, observed = self.values()
        observed[:] = False
        gate = past_constant_gate(history, observed)["keep_forecast"]
        prediction = np.random.default_rng(3).normal(size=(2, 48, 3)).astype(np.float32)
        result = apply_output_gate(prediction, history[:, -1], gate)
        self.assertEqual(result.tobytes(), prediction.tobytes())

    def test_no_future_or_target_in_gate_or_output_API(self):
        self.assertEqual(list(inspect.signature(past_constant_gate).parameters), ["history", "history_observed"])
        self.assertEqual(list(inspect.signature(apply_output_gate).parameters), ["prediction", "raw_last", "keep_forecast"])

    def test_past_flat_future_change_can_make_gate_worse(self):
        history = np.zeros((1, 512, 1), dtype=np.float32)
        gate = past_constant_gate(history, np.ones_like(history, dtype=bool))["keep_forecast"]
        correct_future_prediction = np.ones((1, 48, 1), dtype=np.float32)
        gated = apply_output_gate(correct_future_prediction, history[:, -1], gate)
        self.assertEqual(float(np.mean((gated - correct_future_prediction) ** 2)), 1.)

    def test_gate_is_outside_original_fixed_D_span(self):
        history = np.zeros((1, 512, 2), dtype=np.float32)
        history[0, :, 1] = np.linspace(1., 0., 512, dtype=np.float32)
        gate = past_constant_gate(history, np.ones_like(history, dtype=bool))["keep_forecast"]
        decoder = np.array([[1., 2.]], dtype=np.float64)
        prediction = np.tile(np.array([1., 2.], dtype=np.float32), (1, 48, 1))
        target = np.tile(np.array([0., 2.], dtype=np.float64), (1, 48, 1))
        best_original = target @ np.linalg.pinv(decoder) @ decoder
        original_floor = float(np.mean((target - best_original) ** 2))
        gated = apply_output_gate(prediction, history[:, -1], gate)
        self.assertGreater(original_floor, .39)
        self.assertEqual(float(np.mean((target - gated) ** 2)), 0.)
        self.assertFalse(np.allclose(gated.astype(np.float64), gated @ np.linalg.pinv(decoder) @ decoder))

    def test_last_signed_zero_is_the_replacement_anchor(self):
        history = np.zeros((1, 512, 1), dtype=np.float32)
        history[:, -1] = np.float32(-0.)
        gate = past_constant_gate(history, np.ones_like(history, dtype=bool))["keep_forecast"]
        prediction = np.ones((1, 48, 1), dtype=np.float32)
        result = apply_output_gate(prediction, history[:, -1], gate)
        np.testing.assert_array_equal(result.view(np.uint32), np.full(result.shape, 0x80000000, dtype=np.uint32))

    def test_gate_does_not_change_past_representation_input(self):
        history, observed = self.values()
        original = history.copy()
        past_constant_gate(history, observed)
        np.testing.assert_array_equal(history, original)

    def test_later_known_past_can_change_origin_gate(self):
        history = np.zeros((1, 512, 1), dtype=np.float32)
        observed = np.ones_like(history, dtype=bool)
        before = past_constant_gate(history, observed)["keep_forecast"]
        history[:, -1] = 1.
        after = past_constant_gate(history, observed)["keep_forecast"]
        self.assertFalse(before.item())
        self.assertTrue(after.item())

    def test_FP32_normalization_can_hide_distinct_physical_values(self):
        physical = np.ones((1, 512, 1), dtype=np.float32)
        physical[:, 1::2] = np.nextafter(np.float32(1.), np.float32(2.))
        self.assertNotEqual(physical[0, 0, 0], physical[0, 1, 0])
        mean, std = np.array([-1e8], dtype=np.float32), np.array([1e8], dtype=np.float32)
        normalized = ((physical - mean) / std).astype(np.float32)
        gate = past_constant_gate(normalized, np.ones_like(physical, dtype=bool))
        self.assertFalse(gate["keep_forecast"].item())
        self.assertTrue(gate["exact_flat"].item())

    def test_invalid_history_mask_prediction_contracts(self):
        history, observed = self.values()
        with self.assertRaises(ValueError):
            past_constant_gate(history.astype(np.float64), observed)
        with self.assertRaises(ValueError):
            past_constant_gate(history[:, :511], observed[:, :511])
        with self.assertRaises(ValueError):
            past_constant_gate(history, observed.astype(np.uint8))
        history[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            past_constant_gate(history, observed)
        with self.assertRaises(ValueError):
            apply_output_gate(np.zeros((2, 47, 3), dtype=np.float32), np.zeros((2, 3), dtype=np.float32),
                              np.ones((2, 3), dtype=bool))

    def test_C_and_Fortran_layout_have_identical_values(self):
        history, observed = self.values()
        a = past_constant_gate(history, observed)
        b = past_constant_gate(np.asfortranarray(history), np.asfortranarray(observed))
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])


if __name__ == "__main__":
    unittest.main()
