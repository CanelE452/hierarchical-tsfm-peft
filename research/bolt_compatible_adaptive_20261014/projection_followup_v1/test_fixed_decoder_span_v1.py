"""Analytic synthetic geometry; no original NPZ/checkpoint or model reads."""
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np

import fixed_decoder_span_v1 as span


class SpanTests(unittest.TestCase):
    def simple(self, target, encoder, decoder, z, *, raw_last=None):
        target = np.asarray(target, np.float32).reshape(1, -1, 2)
        observed = np.ones(target.shape, bool)
        raw_last = np.zeros((1, 2), np.float32) if raw_last is None else np.asarray(raw_last, np.float32).reshape(1, 2)
        encoder, decoder = np.asarray(encoder, np.float32).reshape(2, 1), np.asarray(decoder, np.float32).reshape(1, 2)
        z = np.asarray(z, np.float32).reshape(1, -1, 1)
        last = np.zeros((1, 1), np.float32)
        prediction = raw_last[:, None] + z @ decoder
        return span.diagnose_span(target, observed, raw_last, encoder, decoder,
                                 {"FIXED": prediction}, {"FIXED": z}, {"FIXED": last}, np.ones(2, np.float32))

    def test_orthogonal_example_exact_bound(self):
        report, arrays = self.simple([[1, 3]], [1, 0], [1, 0], [1])
        self.assertAlmostEqual(report["fixed_D_bound_scores_same_mask"]["normalized"]["mse"], 4.5)
        np.testing.assert_allclose(arrays["least_squares_latent_innovation"], [[[1]]], atol=1e-14)
        np.testing.assert_allclose(arrays["fixed_D_bound_prediction"], [[[1, 0]]], atol=1e-14)
        self.assertEqual(report["geometry"]["E_D_symmetry_max_abs"], 0)

    def test_oblique_encoder_oracle_is_not_lower_bound(self):
        report, arrays = self.simple([[0, 1]], [1, 0], [1, 2], [.4])
        self.assertAlmostEqual(report["fixed_D_bound_scores_same_mask"]["normalized"]["mse"], .1)
        self.assertAlmostEqual(np.mean((np.array([0., 0.]) - np.array([0., 1.])) ** 2), .5)
        self.assertGreater(report["geometry"]["E_D_symmetry_max_abs"], 0)
        self.assertAlmostEqual(report["geometry"]["E_D_idempotence_max_abs"], 0)
        self.assertAlmostEqual(report["geometry"]["D_E_minus_I_max_abs"], 0)
        np.testing.assert_allclose(arrays["complete_projection_residual"] @ np.array([[1.], [2.]]), 0, atol=1e-14)

    def test_bound_matches_independent_lstsq_and_normal_equation(self):
        report, arrays = self.simple([[1, 3], [2, -1]], [1, 0], [1, 2], [1, 1])
        expected = np.linalg.lstsq(np.array([[1.], [2.]]), np.array([[1., 3.], [2., -1.]]).T, rcond=1e-15)[0].T
        np.testing.assert_allclose(arrays["least_squares_latent_innovation"][0], expected, atol=1e-14)
        self.assertTrue(report["independent_linear_checks"]["normal_equation_pass"])
        self.assertLess(report["independent_linear_checks"]["residual_orthogonality_max_abs"], 1e-14)

    def test_current_c_anchor_is_preserved(self):
        report, arrays = self.simple([[11, 23]], [1, 0], [1, 0], [1], raw_last=[10, 20])
        np.testing.assert_allclose(arrays["fixed_D_bound_prediction"], [[[11, 20]]], atol=1e-14)
        self.assertAlmostEqual(report["fixed_D_bound_scores_same_mask"]["normalized"]["mse"], 4.5)

    def test_fp32_rounding_can_beat_real_arithmetic_bound(self):
        report, _ = self.simple([[1e8 + 8, 12]], [1, 0], [1, 2], [6], raw_last=[1e8, 0])
        entry = report["comparisons"]["FIXED"]
        self.assertAlmostEqual(entry["saved_prediction_scores_same_mask"]["normalized"]["mse"], 0)
        self.assertGreater(report["fixed_D_bound_scores_same_mask"]["normalized"]["mse"], 1)
        self.assertTrue(entry["bound_le_saved_FP32_mse_plus_measured_allowance"])
        self.assertAlmostEqual(entry["measured_saved_FP32_vs_real_MSE_allowance"], 2)

    def test_incomplete_positions_excluded_without_filled_future_targets(self):
        target = np.array([[[1, 3], [np.nan, 1000]]], np.float32)
        mask = np.array([[[True, True], [False, True]]])
        report, arrays = span.diagnose_span(target, mask, np.zeros((1, 2), np.float32),
            np.array([[1], [0]], np.float32), np.array([[1, 0]], np.float32),
            {"RAW_BOLT": np.zeros_like(target)}, {}, {}, np.ones(2, np.float32))
        self.assertEqual(report["complete_target_positions"], 1)
        self.assertAlmostEqual(report["fixed_D_bound_scores_same_mask"]["normalized"]["mse"], 4.5)
        self.assertTrue(np.isnan(arrays["fixed_D_bound_prediction"][0, 1]).all())
        self.assertFalse(report["RAW_bound_inequality_applied"])

    def test_no_complete_positions_rejected(self):
        with self.assertRaisesRegex(ValueError, "NO_ALL_C"):
            span.diagnose_span(np.zeros((1, 1, 2)), np.zeros((1, 1, 2), bool), np.zeros((1, 2)),
                               np.array([[1], [0]]), np.array([[1, 0]]), {}, {}, {}, np.ones(2))

    def test_zero_decoder_span_is_current_anchor(self):
        report, arrays = self.simple([[1, 3]], [1, 0], [0, 0], [7])
        self.assertEqual(report["geometry"]["rank_D"], 0)
        np.testing.assert_array_equal(arrays["fixed_D_bound_prediction"], np.zeros((1, 1, 2)))
        self.assertAlmostEqual(report["fixed_D_bound_scores_same_mask"]["normalized"]["mse"], 5)

    def test_invalid_encoder_decoder_and_masks_rejected(self):
        with self.assertRaisesRegex(ValueError, "FINITE_CK"):
            span.fixed_decoder_geometry(np.ones((2, 1)), np.ones((2, 2)))
        with self.assertRaisesRegex(ValueError, "SHAPES"):
            span.diagnose_span(np.zeros((1, 1, 2)), np.ones((1, 1, 2), np.uint8), np.zeros((1, 2)),
                               np.ones((2, 1)), np.ones((1, 2)), {}, {}, {}, np.ones(2))

    def test_full_rank_d_recovers_arbitrary_target_exactly(self):
        rng = np.random.default_rng(601)
        y = rng.normal(size=(2, 3, 3)).astype(np.float32)
        raw = rng.normal(size=(2, 3)).astype(np.float32)
        report, arrays = span.diagnose_span(y, np.ones_like(y, bool), raw, np.eye(3), np.eye(3), {}, {}, {}, np.ones(3))
        np.testing.assert_allclose(arrays["fixed_D_bound_prediction"], y, atol=1e-14)
        self.assertAlmostEqual(report["fixed_D_bound_scores_same_mask"]["normalized"]["mse"], 0)

    def test_nonfinite_complete_targets_rejected(self):
        with self.assertRaisesRegex(ValueError, "NONFINITE_COMPLETE"):
            span.diagnose_span(np.full((1, 1, 2), np.nan), np.ones((1, 1, 2), bool), np.zeros((1, 2)),
                               np.ones((2, 1)), np.ones((1, 2)), {}, {}, {}, np.ones(2))

    def test_predictions_and_state_not_mutated(self):
        y = np.array([[[1, 3]]], np.float32)
        d = np.array([[1, 0]], np.float32)
        original_y, original_d = y.copy(), d.copy()
        span.diagnose_span(y, np.ones_like(y, bool), np.zeros((1, 2), np.float32),
                           d.T, d, {"RAW_BOLT": y}, {}, {}, np.ones(2))
        np.testing.assert_array_equal(y, original_y)
        np.testing.assert_array_equal(d, original_d)

    def test_report_explicitly_excludes_selection_deployment_and_nonlinear_bound(self):
        report, _ = self.simple([[1, 3]], [1, 0], [1, 0], [1])
        self.assertFalse(report["deployment_used"])
        self.assertFalse(report["checkpoint_or_branch_selection_used"])
        self.assertTrue(report["actual_future_labels_used"])
        self.assertIn("excludes FP32 rounding", report["scope"])

    def test_cli_requires_exact_sha_pass_audit_before_payload_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary) / "producer.json"
            audit = Path(temporary) / "audit.json"
            result.write_text(json.dumps({"status": "COMPLETE", "protected_TEST_accessed": False,
                                          "independent_holdout": False, "datasets": {name: {} for name in span.DATASETS}}))
            audit.write_text(json.dumps({"status": "PASS", "producer": {"path": str(result), "sha256": "0" * 64}}))
            with self.assertRaisesRegex(ValueError, "EXACT_SHA_BOUND"):
                span.verify_audit_binding(result, audit)
            audit.write_text(json.dumps({"status": "PASS", "producer": {"path": str(result), "sha256": span.sha(result)}}))
            producer, _, _ = span.verify_audit_binding(result, audit)
            self.assertEqual(producer["status"], "COMPLETE")


if __name__ == "__main__":
    unittest.main()
