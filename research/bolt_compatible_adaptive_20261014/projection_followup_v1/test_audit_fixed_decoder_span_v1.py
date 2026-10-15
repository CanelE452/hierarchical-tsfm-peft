"""Independent analytic fixtures; no diagnostic producer import or real assets."""
import unittest

import numpy as np

import audit_fixed_decoder_span_v1 as audit


class IndependentFixedDSpanFixtures(unittest.TestCase):
    def test_oblique_E_Y_reconstruction_can_be_worse_than_optimal_fixed_D(self):
        target = np.array([[[0., 1.]]], np.float32)
        e, d = np.array([[1.], [0.]]), np.array([[1., 2.]])
        arrays, geometry = audit.independent_geometry(e, d)
        solution = audit.independent_span(target, np.ones_like(target, bool), np.zeros((1, 2), np.float32), d, np.ones(2))
        np.testing.assert_allclose(solution["prediction"], [[[.4, .8]]], atol=1e-14)
        self.assertAlmostEqual(solution["scores"]["normalized"]["mse"], .1)
        ey = target @ e @ d
        self.assertAlmostEqual(float(np.mean((ey - target) ** 2)), .5)
        self.assertGreater(geometry["E_D_symmetry_max_abs"], 0)
        np.testing.assert_allclose(arrays["orthogonal_output_projector"].T, arrays["orthogonal_output_projector"], atol=1e-14)

    def test_complete_positions_change_original_channel_denominators(self):
        y = np.array([[[1., 3.], [100., np.nan]]], np.float32)
        observed = np.array([[[True, True], [True, False]]])
        solution = audit.independent_span(y, observed, np.zeros((1, 2), np.float32), np.array([[1., 0.]]), [1., 2.])
        self.assertEqual(solution["scores"]["normalized"]["observed_counts"], [1, 1])
        self.assertAlmostEqual(solution["scores"]["normalized"]["mse"], 4.5)
        self.assertTrue(np.isnan(solution["prediction"][0, 1]).all())
        full = audit.prior.scores(np.zeros_like(y), y, observed, [1., 2.])
        self.assertEqual(full["normalized"]["observed_counts"], [2, 1])
        self.assertNotEqual(full["normalized"]["mse"], 4.5)

    def test_fp32_rounding_can_beat_real_span_minimum_and_allowance_is_measured(self):
        target = np.array([[[1e8 + 8, 12.]]], np.float32)
        last = np.array([[1e8, 0.]], np.float32)
        d = np.array([[1., 2.]], np.float32)
        future = np.array([[[6.]]], np.float32)
        previous = np.zeros((1, 1, 1), np.float32)
        saved = last[:, None] + (future - previous) @ d
        trace = audit.real_trace_and_rounding(saved, future, previous, last, d, target, np.ones((1, 1), bool), np.ones(2))
        solution = audit.independent_span(target, np.ones_like(target, bool), last, d, np.ones(2))
        self.assertEqual(trace["saved_scores"]["normalized"]["mse"], 0.)
        self.assertAlmostEqual(trace["measured_MSE_allowance"], 2.)
        self.assertGreater(solution["scores"]["normalized"]["mse"], 1.)
        self.assertLessEqual(solution["scores"]["normalized"]["mse"], trace["measured_MSE_allowance"])

    def test_same_current_C_anchor_makes_bound_on_innovation(self):
        y = np.array([[[11., 23.]]], np.float32)
        last = np.array([[10., 20.]], np.float32)
        solution = audit.independent_span(y, np.ones_like(y, bool), last, np.array([[1., 0.]]), np.ones(2))
        np.testing.assert_allclose(solution["prediction"], [[[11., 20.]]], atol=1e-14)
        np.testing.assert_array_equal(solution["innovation"], [[1., 3.]])

    def test_full_rank_fixed_D_recovers_all_complete_targets(self):
        y = np.array([[[1., 2., 3.], [4., 5., 6.]]], np.float32)
        last = np.array([[.5, 10., -9.]], np.float32)
        solution = audit.independent_span(y, np.ones_like(y, bool), last, np.eye(3), np.ones(3))
        np.testing.assert_allclose(solution["prediction"], y, atol=1e-14)
        self.assertEqual(solution["scores"]["normalized"]["mse"], 0)

    def test_zero_D_span_is_persistence_not_an_unrestricted_predictor_bound(self):
        y = np.array([[[1., 3.]]], np.float32)
        solution = audit.independent_span(y, np.ones_like(y, bool), np.zeros((1, 2), np.float32), np.zeros((1, 2)), np.ones(2))
        self.assertEqual(solution["rank_lstsq"], 0)
        np.testing.assert_array_equal(solution["prediction"], np.zeros_like(y))
        self.assertEqual(solution["scores"]["normalized"]["mse"], 5.)

    def test_normal_residual_and_output_orthogonality_are_separate_saved_checks(self):
        y = np.array([[[1., 3.], [2., -1.]]], np.float32)
        d = np.array([[1., 2.]])
        solution = audit.independent_span(y, np.ones_like(y, bool), np.zeros((1, 2), np.float32), d, np.ones(2))
        np.testing.assert_allclose(solution["normal"], 0., atol=1e-14)
        np.testing.assert_allclose(solution["orthogonality"], 0., atol=1e-14)
        np.testing.assert_allclose(solution["residual"] @ d.T, 0., atol=1e-14)

    def test_physical_and_MAE_scores_are_not_the_normalized_LS_objective(self):
        y = np.array([[[0., 1.]]], np.float32)
        solution = audit.independent_span(y, np.ones_like(y, bool), np.zeros((1, 2), np.float32), np.array([[1., 2.]]), [1., 10.])
        self.assertAlmostEqual(solution["scores"]["normalized"]["mse"], .1)
        self.assertAlmostEqual(solution["scores"]["physical_per_channel_diagnostic"]["mse"], 2.08)
        self.assertAlmostEqual(solution["scores"]["normalized"]["mae"], .3)

    def test_nonfinite_complete_labels_and_empty_complete_mask_rejected(self):
        y = np.array([[[np.nan, 1.]]], np.float32)
        with self.assertRaisesRegex(audit.prior.AuditFailure, "FINITE_NONEMPTY"):
            audit.independent_span(y, np.ones_like(y, bool), np.zeros((1, 2), np.float32), np.array([[1., 0.]]), np.ones(2))
        with self.assertRaisesRegex(audit.prior.AuditFailure, "FINITE_NONEMPTY"):
            audit.independent_span(y, np.zeros_like(y, bool), np.zeros((1, 2), np.float32), np.array([[1., 0.]]), np.ones(2))

    def test_original_arrays_are_unchanged_by_diagnostic_arithmetic(self):
        y, d = np.array([[[1., 3.]]], np.float32), np.array([[1., 2.]], np.float64)
        y0, d0 = y.copy(), d.copy()
        audit.independent_span(y, np.ones_like(y, bool), np.zeros((1, 2), np.float32), d, np.ones(2))
        np.testing.assert_array_equal(y, y0)
        np.testing.assert_array_equal(d, d0)


if __name__ == "__main__":
    unittest.main()
