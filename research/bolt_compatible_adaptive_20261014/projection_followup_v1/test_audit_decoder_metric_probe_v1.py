"""Small independent decoder counterexamples; no real payload or model reads."""
import inspect
import unittest

import numpy as np

import audit_decoder_metric_probe_v1 as audit


class IndependentDecoderMetricFixtures(unittest.TestCase):
    def test_affine_oblique_decoder_differs_from_encoder_Euclidean_inverse(self):
        e = np.array([[1.], [0.]], np.float64)
        original = np.array([[1., 2.]], np.float32)
        inverse = audit.independent_encoder_inverse(e)
        np.testing.assert_array_equal(inverse["D_inverse32"], [[1., 0.]])
        self.assertEqual(audit.geometry(e, original)["DE_minus_I_max_abs"], 0.)
        self.assertEqual(audit.geometry(e, original)["ED_symmetry_max_abs"], 2.)
        self.assertEqual(audit.geometry(e, inverse["D_inverse32"])["ED_symmetry_max_abs"], 0.)

    def test_orthogonal_inverse_reconstructs_rotated_rank_one_coordinate(self):
        inverse = audit.independent_encoder_inverse(np.array([[1.], [2.]]))
        np.testing.assert_allclose(inverse["D_inverse64"], [[.2, .4]], atol=1e-14)
        np.testing.assert_allclose(inverse["real_projector"], [[.2, .4], [.4, .8]], atol=1e-14)

    def test_forecast_current_C_anchor_survives_zero_latent_innovation(self):
        last = np.array([[11., 99.]], np.float32)
        zlast = np.array([[3.]], np.float32)
        future = np.full((1, 4, 1), 3., np.float32)
        d = np.array([[1., 2.]], np.float32)
        predicted = audit.cached_forecast(last, future, zlast, d, backend="numpy_full_N")
        np.testing.assert_array_equal(predicted, np.broadcast_to(last[:, None], (1, 4, 2)))

    def test_cached_decoder_has_no_target_or_future_label_parameter(self):
        names = set(inspect.signature(audit.cached_forecast).parameters)
        self.assertEqual(names, {"raw_last", "latent_future", "latent_last", "decoder", "backend"})
        self.assertNotIn("target", names)
        self.assertNotIn("observed", names)

    def test_full_observed_macro_and_complete_C_mask_are_different_contracts(self):
        prediction = np.array([[[2., 4.], [10., 8.]]], np.float32)
        target = np.zeros_like(prediction)
        observed = np.array([[[True, True], [True, False]]])
        full = audit.prior.scores(prediction, target, observed, [1., 2.])
        complete = audit.span_audit.complete_scores(prediction, target, observed.all(axis=-1), [1., 2.])
        self.assertEqual(full["normalized"]["observed_counts"], [2, 1])
        self.assertEqual(complete["normalized"]["observed_counts"], [1, 1])
        self.assertEqual(full["normalized"]["mse"], 34.)
        self.assertEqual(complete["normalized"]["mse"], 10.)

    def test_byte_exact_check_detects_signed_zero_that_value_equality_misses(self):
        positive, negative = np.array([0.], np.float32), np.array([-0.], np.float32)
        self.assertTrue(np.array_equal(positive, negative))
        with self.assertRaisesRegex(audit.prior.AuditFailure, "BYTE_EXACT"):
            audit.byte_exact(positive, negative, "signed_zero")

    def test_native_CPU32_replay_uses_cached_latents_without_head(self):
        last = np.tile(np.array([[10., 20.]], np.float32), (35, 1))
        future = np.tile(np.array([[[2.], [3.]]], np.float32), (35, 1, 1))
        zlast = np.ones((35, 1), np.float32)
        d = np.array([[1., 2.]], np.float32)
        numpy_value = audit.cached_forecast(last, future, zlast, d, backend="numpy_full_N")
        torch_value = audit.cached_forecast(last, future, zlast, d, backend="torch_CPU2_chunk32")
        audit.byte_exact(torch_value, numpy_value, "analytic_simple_two_chunks")

    def test_shape_or_dtype_change_cannot_be_hidden_by_matching_macro_score(self):
        last, future, previous = np.zeros((1, 2), np.float32), np.zeros((1, 1, 1), np.float32), np.zeros((1, 1), np.float32)
        with self.assertRaisesRegex(audit.prior.AuditFailure, "FINITE_FP32"):
            audit.cached_forecast(last, future, previous, np.ones((1, 2), np.float64), backend="numpy_full_N")


if __name__ == "__main__":
    unittest.main()
