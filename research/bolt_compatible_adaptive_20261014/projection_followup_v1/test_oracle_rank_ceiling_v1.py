"""Analytic future-informed rank bounds using small synthetic matrices only."""
import unittest

import numpy as np

import oracle_rank_ceiling_v1 as oracle
import fixed_decoder_span_v1 as span


class RankOracleTests(unittest.TestCase):
    def analyze(self, matrix, k, raw=None, mask=None):
        target = np.asarray(matrix, np.float32)[None]
        c = target.shape[2]
        raw = np.zeros((1, c), np.float32) if raw is None else np.asarray(raw, np.float32).reshape(1, c)
        mask = np.ones(target.shape, bool) if mask is None else np.asarray(mask, bool)[None]
        return oracle.rank_oracle(target, mask, raw, k, np.ones(c, np.float32))

    def test_rotated_analytic_rank_one(self):
        theta = .6
        rotation = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        matrix = np.diag([3., 2., 1.]) @ rotation
        report, _ = self.analyze(matrix, 1)
        self.assertAlmostEqual(report["minimum_normalized_MSE_from_discarded_energy"], 5 / 9, places=7)

    def test_discarded_energy_conservation(self):
        report, arrays = self.analyze([[3, 0, 0], [0, 2, 0], [0, 0, 1]], 1)
        certificate = report["energy_certificate"]
        self.assertAlmostEqual(certificate["total_innovation_energy"], 14)
        self.assertAlmostEqual(certificate["retained_energy"], 9)
        self.assertAlmostEqual(certificate["discarded_energy"], 5)
        self.assertAlmostEqual(float(np.sum(arrays["rank_K_projection_residual"] ** 2)), 5)
        self.assertTrue(certificate["checks_pass"])

    def test_through_origin_differs_from_centered_plus_future_intercept(self):
        report, _ = self.analyze([[10, 1], [10, -1]], 1)
        self.assertAlmostEqual(report["minimum_normalized_MSE_from_discarded_energy"], .5)
        centered = np.array([[0., 1.], [0., -1.]])
        self.assertEqual(np.linalg.matrix_rank(centered), 1)
        self.assertFalse(report["centered_or_future_intercept_used"])

    def test_incomplete_target_positions_not_filled_or_used(self):
        report, arrays = self.analyze([[3, 0], [np.nan, 10000], [0, 2]], 1,
                                      mask=[[True, True], [False, True], [True, True]])
        self.assertEqual(report["complete_target_positions"], 2)
        self.assertAlmostEqual(report["minimum_normalized_MSE_from_discarded_energy"], 1)
        self.assertTrue(np.isnan(arrays["rank_K_prediction"][0, 1]).all())

    def test_no_complete_positions_rejected(self):
        with self.assertRaisesRegex(ValueError, "COMPLETE_FUTURE"):
            self.analyze([[1, 2]], 1, mask=[[False, True]])

    def test_current_anchor_not_future_mean_changes_innovation(self):
        report, arrays = self.analyze([[11, 20], [10, 22]], 1, raw=[10, 20])
        np.testing.assert_array_equal(arrays["complete_innovation"], [[1, 0], [0, 2]])
        self.assertAlmostEqual(report["minimum_normalized_MSE_from_discarded_energy"], .25)

    def test_svd_matches_gram_eigen_tail_certificate(self):
        rng = np.random.default_rng(601)
        report, arrays = self.analyze(rng.normal(size=(9, 4)), 2)
        gram_floor = np.linalg.eigvalsh(arrays["innovation_gram"])[:2].sum() / (9 * 4)
        self.assertAlmostEqual(gram_floor, report["minimum_normalized_MSE_from_discarded_energy"], places=12)
        np.testing.assert_allclose(arrays["squared_singularvalues"], arrays["gram_eigenvalues"], atol=1e-12)

    def test_rank_oracle_below_fixed_d_and_saved_real_forecasts(self):
        y = np.array([[[3, 0], [0, 2]]], np.float32)
        fixed, _ = span.diagnose_span(y, np.ones_like(y, bool), np.zeros((1, 2), np.float32),
            np.array([[0], [1]], np.float32), np.array([[0, 1]], np.float32),
            {"RAW_BOLT": y, "FIXED": np.zeros_like(y)}, {"FIXED": np.zeros((1, 2, 1), np.float32)},
            {"FIXED": np.zeros((1, 1), np.float32)}, np.ones(2, np.float32))
        report, _ = self.analyze(y[0], 1)
        comparison = oracle.compare_fixed_span(report, {"PCA": fixed})["PCA"]
        self.assertAlmostEqual(report["minimum_normalized_MSE_from_discarded_energy"], 1)
        self.assertAlmostEqual(comparison["fixed_D_floor_MSE_same_mask"], 2.25)
        self.assertTrue(comparison["rank_K_oracle_le_fixed_D_floor"])
        self.assertTrue(comparison["saved_fixed_D_real_predictions"]["FIXED"]["rank_K_oracle_le_real_arithmetic_MSE"])

    def test_full_rank_oracle_zero_residual(self):
        report, arrays = self.analyze([[1, 3], [2, -1]], 2)
        self.assertAlmostEqual(report["minimum_normalized_MSE_from_discarded_energy"], 0)
        np.testing.assert_allclose(arrays["rank_K_prediction"], [[[1, 3], [2, -1]]], atol=1e-14)

    def test_rank_exceeding_number_of_positions_handles_rectangular_matrix(self):
        report, arrays = self.analyze([[1, 2, 3]], 3)
        self.assertAlmostEqual(report["minimum_normalized_MSE_from_discarded_energy"], 0)
        self.assertEqual(arrays["retained_right_singular_vectors"].shape, (1, 3))

    def test_zero_future_innovation(self):
        report, arrays = self.analyze([[1, 2], [1, 2]], 1, raw=[1, 2])
        self.assertEqual(report["numerical_rank_future_matrix"], 0)
        self.assertEqual(report["minimum_normalized_MSE_from_discarded_energy"], 0)
        np.testing.assert_array_equal(arrays["rank_K_prediction"], [[[1, 2], [1, 2]]])

    def test_future_directions_are_unusable_and_not_generalization(self):
        report, _ = self.analyze([[3, 0], [0, 2]], 1)
        self.assertTrue(report["directions_fitted_to_known_future_of_this_split"])
        self.assertFalse(report["deployment_used"])
        self.assertFalse(report["method_checkpoint_branch_selection_used"])
        self.assertFalse(report["one_shared_output_space_across_splits_assumed"])
        self.assertIn("not separate minima", report["scope"])

    def test_invalid_k_and_nonfinite_current_state(self):
        for k in (0, True, 3):
            with self.assertRaises(ValueError):
                self.analyze([[1, 2]], k)
        with self.assertRaises(ValueError):
            self.analyze([[1, 2]], 1, raw=[np.nan, 0])

    def test_oracle_does_not_mutate_target_or_mask(self):
        target = np.array([[[1, 2], [3, 4]]], np.float32)
        mask = np.ones(target.shape, bool)
        original, original_mask = target.copy(), mask.copy()
        oracle.rank_oracle(target, mask, np.zeros((1, 2), np.float32), 1, np.ones(2, np.float32))
        np.testing.assert_array_equal(target, original)
        np.testing.assert_array_equal(mask, original_mask)


if __name__ == "__main__":
    unittest.main()
