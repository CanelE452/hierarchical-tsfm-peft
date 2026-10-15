"""Independent analytic FIT supervision and preservation counterexamples."""
import copy
import inspect
import unittest

import numpy as np

import audit_fit_target_comparison_v1 as audit

audit.initialize_numpy()


def gram(matrix, k=1):
    matrix = np.asarray(matrix, np.float32)
    y = np.resize(matrix, (1, 48, matrix.shape[-1])).copy()
    return audit.fit_target_gram(y, np.ones_like(y, bool), np.zeros((1, matrix.shape[-1]), np.float32), k)


def analytic_fit():
    rng = np.random.default_rng(61115)
    history = rng.normal(size=(2, 512, 2)).astype(np.float32)
    history[:, -1] = 0.
    target = np.tile(np.array([[[3., 0.], [0., 2.]]], np.float32), (2, 24, 1))
    fit = {"history": history, "history_observed": np.ones_like(history, bool), "target": target,
           "observed": np.ones_like(target, bool), "raw_last": history[:, -1].copy(), "origins": np.array([512, 520], np.int64)}
    flat = history.astype(np.float64).reshape(-1, 2)
    center = flat.mean(axis=0)
    variance = float(np.square(flat[:, 0] - center[0]).mean())
    encoder = np.array([[1. / np.sqrt(variance)], [0.]], np.float64)
    decoder = audit.decoder_audit.independent_encoder_inverse(encoder)["D_inverse32"].astype(np.float64)
    state = {"center": center, "encoder": encoder, "decoder": decoder, "decoder_bias": center.copy()}
    complete = fit["observed"].all(axis=-1)
    innovation = target.astype(np.float64).reshape(-1, 2)
    spectrum = [np.sqrt(432.), np.sqrt(192.)]
    latent_cov = ((flat - center) @ encoder).T @ ((flat - center) @ encoder) / len(flat)
    metadata = {"route": "FIT_TARGET_SUBSPACE", "learning_class": "SUPERVISED_FIT_TARGET_SUBSPACE", "channels": 2, "K": 1,
                "length": 512, "horizon": 48, "TUNE_or_DEV_used_in_projection_or_decoder_fit": False,
                "strict_K_only_compression": False, "independent_holdout": False,
                "state_array_sha256": {key: audit.array_sha(value) for key, value in state.items()},
                "latent_fit_covariance": latent_cov.tolist(),
                "decoder_construction": {"pinv_rcond": 1e-15, "rank_deployed_encoder": 1,
                    "encoder_FP32": audit.raw_binding(encoder.astype(np.float32)), "decoder_FP32": audit.raw_binding(decoder.astype(np.float32)),
                    "future_labels_used_to_construct_inverse_decoder": False},
                "actual_future_target_used_in_projection_or_decoder_fit": True, "target_output_space_is_FIT_only": True,
                "method_checkpoint_branch_selection_used_by_this_fit": False, "eigenvalue_relative_cut": 1e-8, "target_rank_relative_cut": 1e-15,
                "fit_input_bindings": {key: audit.raw_binding(value) for key, value in (
                    ("history", history), ("history_observed", fit["history_observed"]), ("FIT_target", target),
                    ("FIT_target_observed", fit["observed"]), ("raw_last", fit["raw_last"]), ("origins", fit["origins"]))},
                "fit_windows": 2, "fit_positions_with_overlap": 1024, "fit_complete_future_positions": 96,
                "fit_total_future_positions": 96, "fit_complete_future_positions_per_horizon": [2] * 48,
                "FIT_history_observed_counts": [1024, 1024], "FIT_target_observed_counts": [96, 96], "FIT_origins_unique": 2,
                "complete_future_mask": audit.raw_binding(complete), "FIT_target_innovation": audit.raw_binding(innovation),
                "target_output_directions": [[1.], [0.]], "FIT_target_innovation_total_energy": 624.,
                "FIT_target_top_K_retained_energy": 432., "FIT_target_top_K_retained_fraction": 432. / 624.,
                "target_singularvalues": spectrum, "target_numerical_rank": 2,
                "restricted_past_covariance": [[variance]], "whitening_rank": 1,
                "retained_whitening_eigenvalues": [variance],
                "target_K_boundary_relative_singular_gap": (spectrum[0] - spectrum[1]) / spectrum[0]}
    return fit, state, metadata


class IndependentFitTargetFixtures(unittest.TestCase):
    def test_rotated_top_direction_uses_Gram_without_SVD_or_producer_import(self):
        solution = gram([[1., 2.], [2., 4.], [-3., -6.]])
        np.testing.assert_allclose(solution["projector"], np.array([[1., 2.], [2., 4.]]) / 5., atol=1e-14)
        self.assertTrue(solution["retained_boundary_unique"])
        source = inspect.getsource(audit.fit_target_gram)
        self.assertNotIn("svd", source.lower())
        self.assertNotIn("future_target_projection", source)

    def test_target_mean_is_retained_as_signal_through_origin(self):
        solution = gram([[10., 1.], [10., -1.]])
        np.testing.assert_allclose(solution["projector"], [[1., 0.], [0., 0.]], atol=1e-14)
        self.assertAlmostEqual(solution["total_energy"], 48. * 101.)

    def test_incomplete_position_all_channels_excluded_and_Nc_is_not_pooled(self):
        target = np.zeros((1, 48, 2), np.float32)
        target[0, 0], target[0, 1], target[0, 2] = [3., 0.], [np.nan, 1e20], [0., 2.]
        observed = np.zeros_like(target, bool)
        observed[0, :3] = True
        observed[0, 1, 0] = False
        result = audit.fit_target_gram(target, observed, np.zeros((1, 2), np.float32), 1)
        self.assertEqual(result["complete"].sum(), 2)
        np.testing.assert_array_equal(result["innovation"], [[3., 0.], [0., 2.]])
        self.assertEqual(result["total_energy"], 13.)

    def test_large_independent_future_noise_can_dominate_predictable_signal(self):
        # Balanced enumeration makes the second coordinate a deterministic past label
        # while the independent first-coordinate noise has higher target energy.
        future = [[10., 1.], [-10., 1.], [10., -1.], [-10., -1.]]
        result = gram(future)
        np.testing.assert_array_equal(result["projector"], [[1., 0.], [0., 0.]])
        self.assertGreater(result["retained_energy"], result["total_energy"] / 2.)
        self.assertEqual(sum(row[0] * row[1] for row in future), 0.)
        # A zero innovation forecaster in selected noise direction cannot recover
        # coordinate two even though a past feature equal to it predicts it exactly.
        self.assertGreater(np.mean(np.square(np.asarray(future)[:, 1])), 0.)

    def test_tied_boundary_preserves_energy_without_unique_vectors_claim(self):
        solution = gram([[1., 0.], [0., 1.]])
        self.assertFalse(solution["retained_boundary_unique"])
        self.assertAlmostEqual(solution["retained_energy"], 24.)

    def test_current_anchor_subtracted_before_FIT_target_Gram(self):
        target = np.full((1, 48, 2), [11., 22.], np.float32)
        result = audit.fit_target_gram(target, np.ones_like(target, bool), np.array([[10., 20.]], np.float32), 1)
        np.testing.assert_array_equal(result["innovation"], np.tile([1., 2.], (48, 1)))

    def test_unresolved_Gram_rank_does_not_claim_exact_1e_minus15_rank(self):
        result = gram([[1., 0.], [0., 1e-12]])
        self.assertEqual(result["certain_target_rank"], 1)
        self.assertFalse(result["rank_at_1e_minus_15_fully_resolved"])

    def test_zero_or_unresolved_retained_space_cannot_be_silently_regularized(self):
        with self.assertRaisesRegex(audit.prior.AuditFailure, "ABOVE_GRAM_ROUNDOFF"):
            gram([[0., 0.], [0., 0.]])
        with self.assertRaisesRegex(audit.prior.AuditFailure, "ABOVE_GRAM_ROUNDOFF"):
            gram([[1., 0.], [0., 1e-12]], k=2)

    def test_full_analytic_supervised_state_contract_and_indirect_D_dependence(self):
        fit, state, metadata = analytic_fit()
        result = audit.check_projection_metadata(metadata, state, fit)
        self.assertTrue(result["E_future_labels_used"])
        self.assertTrue(result["D_indirect_FIT_future_dependence_via_E"])
        self.assertTrue(result["target_rank_independently_resolved"])

    def test_same_shape_TUNE_target_or_wrong_origin_cannot_replace_FIT_content(self):
        fit, state, metadata = analytic_fit()
        for field in ("target", "origins"):
            altered = {key: value.copy() for key, value in fit.items()}
            altered[field].flat[0] += 1
            with self.assertRaisesRegex(audit.prior.AuditFailure, "ORIGINAL_FIT_ARRAY"):
                audit.check_projection_metadata(metadata, state, altered)

    def test_supervised_space_cannot_be_labelled_unsupervised(self):
        fit, state, metadata = analytic_fit()
        metadata["actual_future_target_used_in_projection_or_decoder_fit"] = False
        with self.assertRaisesRegex(audit.prior.AuditFailure, "SUPERVISED_FIT_TARGET"):
            audit.check_projection_metadata(metadata, state, fit)

    def test_mask_count_mismatch_is_rejected_even_when_scalar_energy_matches(self):
        fit, state, metadata = analytic_fit()
        metadata["fit_complete_future_positions_per_horizon"][0] = 1
        with self.assertRaisesRegex(audit.prior.AuditFailure, "OVERLAP_COUNTS"):
            audit.check_projection_metadata(metadata, state, fit)

    def test_output_span_diagnostic_has_same_complete_mask_and_no_RAW_bound(self):
        fit, state, _ = analytic_fit()
        values = {**fit, "std": np.array([1., 10.], np.float32), "raw_forecast": fit["target"].copy()}
        values["observed"] = fit["observed"].copy()
        values["observed"][0, 0, 1] = False
        latent = audit.encode(values["history"], state["center"], state["encoder"])
        future = audit.encode(values["target"], state["center"], state["encoder"])
        prediction = audit.anchor_prediction(values["history"], values["raw_last"], future, state["center"], state["encoder"], state["decoder"])
        cached = {name: {"prediction": prediction, "latent_future": future, "latent_last": latent[:, -1], "origins": values["origins"]}
                  for name in ("BOLT", "NATIVE_s92601", "NATIVE_s92602", "E_Y")}
        output = audit.output_subspace_cell(values, state, cached)
        self.assertEqual(output["complete_target_positions"], 95)
        self.assertEqual(output["RAW_scores_same_complete_mask"]["normalized"]["observed_counts"], [95, 95])
        self.assertGreater(output["fixed_D_minimum_normalized_MSE"], 0.)
        self.assertEqual(output["RAW_scores_same_complete_mask"]["normalized"]["mse"], 0.)
        self.assertFalse(output["RAW_bound_inequality_applied"])
        self.assertFalse(output["physical_MSE_or_MAE_minimum_claimed"])
        self.assertFalse(output["deployment_or_checkpoint_or_branch_selection_used"])

    def test_native_frozen_buffers_are_byte_identical_including_zero_sign(self):
        _, projection, _ = analytic_fit()
        arrays = {"projection." + key: value.astype(np.float32) for key, value in projection.items()}
        arrays["head.weight"] = np.zeros((1, 48, 512), np.float32)
        arrays["head.bias"] = np.zeros((1, 48), np.float32)
        audit.native_state_certificate(arrays, projection)
        arrays["projection.encoder"][1, 0] = -0.
        with self.assertRaisesRegex(audit.prior.AuditFailure, "frozen_E_D_bytes"):
            audit.native_state_certificate(arrays, projection)


if __name__ == "__main__":
    unittest.main()
