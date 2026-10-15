"""Analytic and adversarial CPU fixtures only; no actual scientific assets."""
import inspect
import unittest

import numpy as np

from local_past_decoder_v1 import fit_local_past_decoder, SINGULAR_RELATIVE_CUT


def signs():
    s = np.tile(np.array([-1., 1.], np.float32), 256)
    t = np.tile(np.array([-1., -1., 1., 1.], np.float32), 128)
    return s, t


def near_collinear(epsilon=2e-6):
    s, t = signs()
    z = np.stack((s, s + np.float32(epsilon) * t), axis=-1)[None]
    raw = np.concatenate((z, t[None, :, None]), axis=2)
    return raw, np.ones_like(raw, dtype=bool), z


class LocalPastDecoderTests(unittest.TestCase):
    def fixture(self):
        s, t = signs()
        z = np.tile(np.stack((s, t), axis=1)[None], (2, 1, 1))
        matrix = np.array([[1., 2., -1.], [0., .5, 1.]], dtype=np.float32)
        bias = np.array([3., -2., .75], dtype=np.float32)
        raw = z @ matrix + bias
        return raw, np.ones_like(raw, dtype=bool), z, matrix, bias

    def test_observed_affine_past_is_recovered_and_bias_cancels_forecast(self):
        raw, mask, z, matrix, bias = self.fixture()
        state = fit_local_past_decoder(raw, mask, z)
        np.testing.assert_allclose(state.decoder32, np.broadcast_to(matrix, state.decoder32.shape), atol=1e-7)
        np.testing.assert_allclose(state.bias32, np.broadcast_to(bias, state.bias32.shape), atol=1e-7)
        np.testing.assert_allclose(state.reconstruct(z), raw, atol=1e-6)
        future = np.repeat(z[:, -1:, :], 48, axis=1) + np.float32(1.)
        prediction = state.forecast(raw[:, -1], future, z[:, -1])
        expected = raw[:, -1:, :] + np.ones((2, 48, 2), dtype=np.float32) @ matrix
        np.testing.assert_array_equal(prediction, expected)

    def test_zero_latent_innovation_is_exact_current_C_value(self):
        raw, mask, z, *_ = self.fixture()
        state = fit_local_past_decoder(raw, mask, z)
        future = np.repeat(z[:, -1:, :], 48, axis=1)
        np.testing.assert_array_equal(state.forecast(raw[:, -1], future, z[:, -1]),
                                      np.repeat(raw[:, -1:, :], 48, axis=1))

    def test_unobserved_filled_responses_are_not_LS_targets(self):
        raw, mask, z, *_ = self.fixture()
        mask[0, :10, 2] = False
        changed = raw.copy()
        changed[0, :10, 2] = 1e8
        a = fit_local_past_decoder(raw, mask, z)
        b = fit_local_past_decoder(changed, mask, z)
        np.testing.assert_array_equal(a.decoder32, b.decoder32)
        np.testing.assert_array_equal(a.bias32, b.bias32)
        self.assertEqual(a.metadata()["true_observed_counts"][0][2], 502)

    def test_rank_deficient_duplicate_latents_use_centered_slope_minimum_norm(self):
        s, _ = signs()
        z = np.stack((s, s), axis=1)[None]
        raw = np.stack((2. * s + 3., -s + .5), axis=1)[None]
        state = fit_local_past_decoder(raw, np.ones_like(raw, bool), z)
        np.testing.assert_allclose(state.decoder32[0], [[1., -.5], [1., -.5]], atol=1e-7)
        self.assertEqual(state.metadata()["centered_slope_ranks"], [[1, 1]])
        self.assertEqual(state.metadata()["affine_model_ranks_with_analytic_intercept"], [[2, 2]])

    def test_zero_observation_fallback_and_one_observation_are_distinct(self):
        raw, mask, z, *_ = self.fixture()
        mask[0, :, 0] = False
        mask[0, :, 1] = False
        mask[0, 7, 1] = True
        state = fit_local_past_decoder(raw, mask, z)
        np.testing.assert_array_equal(state.decoder32[0, :, :2], 0.)
        self.assertEqual(state.bias32[0, 0], raw[0, -1, 0])
        self.assertEqual(state.bias32[0, 1], raw[0, 7, 1])
        meta = state.metadata()
        self.assertTrue(meta["no_observation_current_C_fallback"][0][0])
        self.assertFalse(meta["no_observation_current_C_fallback"][0][1])
        self.assertEqual(meta["affine_model_ranks_with_analytic_intercept"][0][:2], [0, 1])

    def test_minimum_norm_is_coordinate_dependent_unlike_residual_objective(self):
        raw = np.full((1, 512, 1), 3., dtype=np.float32)
        z = np.full((1, 512, 1), 2., dtype=np.float32)
        mask = np.zeros_like(raw, bool)
        mask[0, 0, 0] = True
        state = fit_local_past_decoder(raw, mask, z)
        uncentered = np.linalg.lstsq(np.array([[2., 1.]]), np.array([3.]), rcond=1e-4)[0]
        self.assertEqual(state.decoder32.item(), 0.)
        self.assertEqual(state.bias32.item(), 3.)
        self.assertAlmostEqual(uncentered[0], 1.2)
        self.assertAlmostEqual(float(np.array([2., 1.]) @ uncentered), 3.)

    def test_rank_deficient_minimum_norm_changes_under_nonorthogonal_coordinates(self):
        s, _ = signs()
        raw = np.stack((s, s), axis=1)[None]
        z = raw.copy()
        transformed = z * np.array([1., 2.], np.float32)
        a = fit_local_past_decoder(raw, np.ones_like(raw, bool), z)
        b = fit_local_past_decoder(raw, np.ones_like(raw, bool), transformed)
        np.testing.assert_allclose(a.decoder32[0, :, 0], [.5, .5], atol=1e-7)
        np.testing.assert_allclose(b.decoder32[0, :, 0], [.2, .4], atol=1e-7)
        self.assertEqual(float(np.mean((a.reconstruct(z) - raw) ** 2)), 0.)
        np.testing.assert_allclose(b.reconstruct(transformed), raw, atol=1e-7)
        delta = np.array([1., 0.], np.float32)
        self.assertAlmostEqual(float(delta @ a.decoder32[0, :, 0]), .5)
        self.assertAlmostEqual(float(delta @ b.decoder32[0, :, 0]), .2, places=7)

    def test_window_permutation_equivariance(self):
        raw, mask, z, *_ = self.fixture()
        raw[1, :, 2] *= -1.
        order = np.array([1, 0])
        a = fit_local_past_decoder(raw, mask, z)
        b = fit_local_past_decoder(raw[order], mask[order], z[order])
        np.testing.assert_array_equal(b.decoder32, a.decoder32[order])
        future = np.repeat(z[:, -1:, :], 48, axis=1) + .5
        np.testing.assert_array_equal(b.forecast(raw[order, -1], future[order], z[order, -1]),
                                      a.forecast(raw[:, -1], future, z[:, -1])[order])

    def test_future_labels_are_not_fit_API_or_state_dependency(self):
        raw, mask, z, *_ = self.fixture()
        unused_future = np.zeros((2, 48, 3), np.float32)
        a = fit_local_past_decoder(raw, mask, z)
        unused_future[:] = 1e9
        b = fit_local_past_decoder(raw, mask, z)
        np.testing.assert_array_equal(a.decoder32, b.decoder32)
        self.assertEqual(list(inspect.signature(fit_local_past_decoder).parameters),
                         ["history", "history_observed", "latent_history"])

    def test_perfect_past_and_perfect_latent_future_do_not_ensure_raw_future(self):
        s, _ = signs()
        z = s[None, :, None]
        raw = np.repeat(z, 2, axis=2)
        state = fit_local_past_decoder(raw, np.ones_like(raw, bool), z)
        np.testing.assert_allclose(state.reconstruct(z), raw, atol=1e-7)
        future = np.repeat(z[:, -1:], 48, axis=1) + 1.
        truth = raw[:, -1:, :] + np.tile(np.array([1., -1.], np.float32), (1, 48, 1))
        prediction = state.forecast(raw[:, -1], future, z[:, -1])
        self.assertEqual(float(np.mean((prediction - truth) ** 2)), 2.)

    def test_conditional_decoder_leaves_old_global_output_space(self):
        s, _ = signs()
        z = np.tile(s[None, :, None], (2, 1, 1))
        raw = np.concatenate((z, z.copy()), axis=2)
        raw[1, :, 1] *= -1.
        state = fit_local_past_decoder(raw, np.ones_like(raw, bool), z)
        old_global_decoder = np.array([[1., 0.]], np.float64)
        innovation = state.decoder32[:, 0, :].astype(np.float64)
        projected = innovation @ np.linalg.pinv(old_global_decoder) @ old_global_decoder
        self.assertFalse(np.allclose(innovation, projected))
        np.testing.assert_allclose(innovation, [[1., 1.], [1., -1.]], atol=1e-7)

    def test_centered_singular_cut_and_covariance_eigen_cut_have_same_rank(self):
        raw, mask, z = near_collinear()
        state = fit_local_past_decoder(raw, mask, z)
        centered = z[0].astype(np.float64) - z[0].mean(axis=0, dtype=np.float64)
        eigen = np.linalg.eigvalsh(centered.T @ centered / len(centered))
        covariance_rank = int(np.sum(eigen > eigen[-1] * SINGULAR_RELATIVE_CUT ** 2))
        self.assertEqual(covariance_rank, state.metadata()["centered_slope_ranks"][0][2])
        self.assertEqual(covariance_rank, 1)

    def test_nearly_collinear_1e_minus_8_solver_amplifies_small_forecast_error(self):
        raw, mask, z = near_collinear()
        centered = z[0].astype(np.float64)
        target = raw[0, :, 2].astype(np.float64)
        weak_d, _, rank, _ = np.linalg.lstsq(centered, target, rcond=1e-8)
        self.assertEqual(rank, 2)
        self.assertGreater(np.linalg.norm(weak_d), 1e5)
        state = fit_local_past_decoder(raw, mask, z)
        self.assertLess(np.linalg.norm(state.decoder32[0, :, 2]), 1e-5)
        erroneous_delta = np.array([0., .001], np.float64)
        self.assertGreater(abs(float(erroneous_delta @ weak_d)), 100.)
        self.assertLess(abs(float(erroneous_delta @ state.decoder32[0, :, 2])), 1e-7)

    def test_discarding_weak_direction_can_lose_true_predictive_signal(self):
        raw, mask, z = near_collinear()
        weak_d = np.linalg.lstsq(z[0].astype(np.float64), raw[0, :, 2].astype(np.float64), rcond=1e-8)[0]
        state = fit_local_past_decoder(raw, mask, z)
        exact_delta = np.array([0., float(z[0, 0, 1] - z[0, 0, 0])], np.float64)
        self.assertAlmostEqual(float(exact_delta @ weak_d), -1., places=6)
        self.assertLess(abs(float(exact_delta @ state.decoder32[0, :, 2])), 1e-8)

    def test_truncated_certificate_is_retained_direction_not_full_gradient(self):
        raw, mask, z = near_collinear()
        state = fit_local_past_decoder(raw, mask, z)
        features = z[0].astype(np.float64)
        target = raw[0, :, 2].astype(np.float64)
        error = features @ state.decoder_fit64[0, :, 2] + state.bias_fit64[0, 2] - target
        full_gradient = features.T @ error
        _, singular, right = np.linalg.svd(features, full_matrices=False)
        retained = right[singular > singular[0] * SINGULAR_RELATIVE_CUT]
        self.assertGreater(np.linalg.norm(full_gradient), 1e-5)
        np.testing.assert_allclose(retained @ full_gradient, 0., atol=1e-10)

    def test_fixed_cutoff_has_a_discontinuous_rank_boundary(self):
        s, t = signs()
        ranks, norms = [], []
        for epsilon in (.999e-4, 1.001e-4):
            z = np.stack((s, np.float32(epsilon) * t), axis=1)[None]
            raw = np.concatenate((z, t[None, :, None]), axis=2)
            state = fit_local_past_decoder(raw, np.ones_like(raw, bool), z)
            ranks.append(state.metadata()["centered_slope_ranks"][0][2])
            norms.append(float(np.linalg.norm(state.decoder32[0, :, 2])))
        self.assertEqual(ranks, [1, 2])
        self.assertLess(norms[0], 1e-12)
        self.assertGreater(norms[1], 9000.)

    def test_relative_condition_limit_does_not_bound_absolute_D_gain(self):
        s, _ = signs()
        z = (s * np.float32(1e-8))[None, :, None]
        raw = np.concatenate((s[None, :, None], z), axis=2)
        state = fit_local_past_decoder(raw, np.ones_like(raw, bool), z)
        self.assertEqual(state.metadata()["max_retained_relative_condition_number"], 1.)
        self.assertGreater(state.metadata()["max_abs_decoder_per_window"][0], 9e7)

    def test_FP32_overflow_rejects_path_instead_of_changing_regularization(self):
        s, _ = signs()
        z = (s * np.float32(1e-30))[None, :, None]
        raw = np.concatenate(((s * np.float32(1e20))[None, :, None], z), axis=2)
        with self.assertRaisesRegex(ValueError, "NONFINITE_FP32_LOCAL_DECODER_CAST"):
            fit_local_past_decoder(raw, np.ones_like(raw, bool), z)

    def test_nonfinite_future_prediction_is_reported(self):
        s, _ = signs()
        z = (s * np.float32(1e-10))[None, :, None]
        raw = np.concatenate((s[None, :, None], z), axis=2)
        state = fit_local_past_decoder(raw, np.ones_like(raw, bool), z)
        future = np.full((1, 48, 1), 1e30, np.float32)
        with self.assertRaisesRegex(ValueError, "NONFINITE_FP32_LOCAL_DECODER_FORECAST"):
            state.forecast(raw[:, -1], future, z[:, -1])

    def test_latent_translation_is_absorbed_by_analytic_intercept(self):
        raw, mask, z, *_ = self.fixture()
        a = fit_local_past_decoder(raw, mask, z)
        b = fit_local_past_decoder(raw, mask, z + np.float32(5.))
        np.testing.assert_array_equal(a.decoder32, b.decoder32)
        future = np.repeat(z[:, -1:, :], 48, axis=1) + np.float32(.5)
        np.testing.assert_array_equal(a.forecast(raw[:, -1], future, z[:, -1]),
                                      b.forecast(raw[:, -1], future + 5., z[:, -1] + 5.))

    def test_input_arrays_and_F_order_are_preserved(self):
        raw, mask, z, *_ = self.fixture()
        before = [value.tobytes() for value in (raw, mask, z)]
        a = fit_local_past_decoder(raw, mask, z)
        b = fit_local_past_decoder(*(np.asfortranarray(value) for value in (raw, mask, z)))
        self.assertEqual(before, [value.tobytes() for value in (raw, mask, z)])
        np.testing.assert_array_equal(a.decoder32, b.decoder32)
        self.assertEqual(a.metadata()["fit_input_bindings"], b.metadata()["fit_input_bindings"])

    def test_metadata_and_arrays_are_pure_serializable_state(self):
        raw, mask, z, *_ = self.fixture()
        state = fit_local_past_decoder(raw, mask, z)
        arrays = state.to_npz()
        self.assertEqual(arrays["decoder_deployed_FP32"].dtype, np.float32)
        self.assertTrue(state.metadata()["inference_time_decoder_refitting"])
        self.assertFalse(state.metadata()["future_labels_used_in_local_decoder_fit"])
        self.assertFalse(state.metadata()["static_fixed_D_floor_reused_as_dynamic_lower_bound"])
        self.assertFalse(state.metadata()["same_decoder_side_information_as_static"])
        self.assertEqual(state.metadata()["singular_relative_cut"], 1e-4)


if __name__ == "__main__":
    unittest.main()
