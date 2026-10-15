"""Analytic Gram fixtures; no SVD or diagnostic producer import and no real assets."""
import unittest

import numpy as np

import audit_oracle_rank_ceiling_v1 as audit


class IndependentRankOracleFixtures(unittest.TestCase):
    def solve(self, matrix, k, std=None):
        matrix = np.asarray(matrix, np.float32)
        y = matrix[None]
        return audit.gram_oracle(y, np.ones_like(y, bool), np.zeros((1, matrix.shape[-1]), np.float32), k,
                                 np.ones(matrix.shape[-1]) if std is None else std)

    def test_rotated_rank_one_has_zero_tail_without_vector_sign_choice(self):
        solution = self.solve([[1., 2.], [2., 4.], [-3., -6.]], 1)
        np.testing.assert_allclose(solution["reconstruction"], [[1., 2.], [2., 4.], [-3., -6.]], atol=1e-14)
        self.assertLess(solution["normalized_mse_minimum"], 1e-14)
        self.assertTrue(solution["unique_retained_boundary"])

    def test_diagonal_spectrum_discarded_energy_M_times_C(self):
        solution = self.solve([[3., 0., 0.], [0., 2., 0.], [0., 0., 1.]], 1)
        self.assertAlmostEqual(solution["discarded_energy"], 5.)
        self.assertAlmostEqual(solution["normalized_mse_minimum"], 5. / 9.)
        np.testing.assert_array_equal(solution["gram"], np.diag([9., 4., 1.]))

    def test_unscaled_constant_innovation_is_not_centered_or_given_intercept(self):
        solution = self.solve([[1., 2.], [1., 2.], [1., 2.]], 1)
        self.assertAlmostEqual(solution["total_energy"], 15.)
        np.testing.assert_allclose(solution["reconstruction"], [[1., 2.]] * 3, atol=1e-14)
        self.assertLess(solution["discarded_energy"], 1e-12)

    def test_rank_C_oracle_recovers_every_known_future(self):
        solution = self.solve([[1., 2., 3.], [3., -1., 4.], [2., 0., -1.]], 3)
        self.assertEqual(solution["normalized_mse_minimum"], 0.)
        np.testing.assert_allclose(solution["prediction"][0], [[1., 2., 3.], [3., -1., 4.], [2., 0., -1.]], atol=1e-13)

    def test_zero_energy_zero_tail_is_finite_and_has_no_unique_boundary(self):
        solution = self.solve([[0., 0.], [0., 0.]], 1)
        self.assertEqual(solution["normalized_mse_minimum"], 0.)
        self.assertFalse(solution["unique_retained_boundary"])

    def test_degenerate_boundary_does_not_require_unique_eigenvector_or_projection(self):
        solution = self.solve([[1., 0.], [0., 1.]], 1)
        self.assertFalse(solution["unique_retained_boundary"])
        self.assertAlmostEqual(solution["normalized_mse_minimum"], .25)
        first = np.array([[1., 0.], [0., 0.]])
        second = np.array([[0., 0.], [0., 1.]])
        original = np.eye(2)
        self.assertEqual(float(((original - first) ** 2).sum()), float(((original - second) ** 2).sum()))

    def test_complete_positions_have_equal_Nc_and_missing_values_are_not_filled(self):
        y = np.array([[[3., 0.], [np.nan, 1e6], [0., 2.]]], np.float32)
        observed = np.array([[[True, True], [False, True], [True, True]]])
        solution = audit.gram_oracle(y, observed, np.zeros((1, 2), np.float32), 1, [1., 10.])
        self.assertEqual(solution["scores"]["normalized"]["observed_counts"], [2, 2])
        self.assertAlmostEqual(solution["discarded_energy"], 4.)
        self.assertAlmostEqual(solution["normalized_mse_minimum"], 1.)
        self.assertTrue(np.isnan(solution["prediction"][0, 1]).all())

    def test_same_current_C_anchor_is_subtracted_before_uncentered_Gram(self):
        y = np.array([[[11., 22.], [12., 24.]]], np.float32)
        solution = audit.gram_oracle(y, np.ones_like(y, bool), np.array([[10., 20.]], np.float32), 1, np.ones(2))
        np.testing.assert_array_equal(solution["innovation"], [[1., 2.], [2., 4.]])
        self.assertAlmostEqual(solution["total_energy"], 25.)

    def test_physical_scoring_does_not_change_normalized_rank_objective(self):
        first = self.solve([[3., 0.], [0., 2.]], 1, [1., 10.])
        second = self.solve([[3., 0.], [0., 2.]], 1, [100., 1.])
        self.assertEqual(first["normalized_mse_minimum"], second["normalized_mse_minimum"])
        np.testing.assert_array_equal(first["reconstruction"], second["reconstruction"])
        self.assertNotEqual(first["scores"]["physical_per_channel_diagnostic"]["mse"], second["scores"]["physical_per_channel_diagnostic"]["mse"])

    def test_invalid_K_empty_complete_and_nonfinite_complete_rejected(self):
        y = np.zeros((1, 2, 2), np.float32)
        for k in (0, 3, True):
            with self.assertRaisesRegex(audit.prior.AuditFailure, "INTEGER_K"):
                audit.gram_oracle(y, np.ones_like(y, bool), np.zeros((1, 2)), k, np.ones(2))
        with self.assertRaisesRegex(audit.prior.AuditFailure, "FINITE_NONEMPTY"):
            audit.gram_oracle(y, np.zeros_like(y, bool), np.zeros((1, 2)), 1, np.ones(2))


if __name__ == "__main__":
    unittest.main()
