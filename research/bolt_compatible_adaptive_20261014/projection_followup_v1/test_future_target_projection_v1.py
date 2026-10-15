"""Small CPU fixtures; no actual data, model, checkpoint or experiment IO."""
import inspect
import unittest

import numpy as np

from projection_math_v1 import fit_projection, ProjectionState
from future_target_projection_v1 import fit_target_subspace, pca_encoder_pinv_control


def fixture(channels=3, windows=3):
    rng = np.random.default_rng(61015)
    history = rng.normal(size=(windows, 512, channels)).astype(np.float32)
    history *= np.arange(channels, 0, -1, dtype=np.float32)[None, None]
    past_mask = np.ones(history.shape, dtype=bool)
    last = history[:, -1].copy()
    direction = np.arange(1, channels + 1, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    amplitude = rng.normal(size=(windows, 48, 1))
    target = (last[:, None].astype(np.float64) + amplitude * direction).astype(np.float32)
    observed = np.ones(target.shape, dtype=bool)
    origins = np.arange(windows, dtype=np.int64) * 7 + 512
    return history, past_mask, target, observed, last, origins


class TargetProjectionTests(unittest.TestCase):
    def fit(self, values=None, k=1):
        return fit_target_subspace(*(fixture() if values is None else values), k=k)

    def test_future_space_differs_from_high_variance_input_pca(self):
        values = fixture()
        h, hm, y, ym, last, origins = values
        direction = np.array([0., 1., 1.]) / np.sqrt(2.)
        amplitude = np.linspace(-2., 3., h.shape[0] * 48).reshape(h.shape[0], 48, 1)
        y = (last[:, None] + amplitude * direction).astype(np.float32)
        state = self.fit((h, hm, y, ym, last, origins))
        projector = state.encoder @ np.linalg.pinv(state.encoder)
        np.testing.assert_allclose(projector, np.outer(direction, direction), atol=2e-8)
        pca = fit_projection("PCA", h, hm, 1)
        self.assertLess(abs(float(pca.encoder[:, 0] @ direction)) / np.linalg.norm(pca.encoder[:, 0]), .15)

    def test_whitened_two_dimensional_space(self):
        h, hm, y, ym, last, origins = fixture()
        rng = np.random.default_rng(15)
        changes = rng.normal(size=(len(h), 48, 2))
        directions = np.array([[1., 0., 0.], [0., 1., 0.]])
        y = (last[:, None] + changes @ directions).astype(np.float32)
        state = self.fit((h, hm, y, ym, last, origins), k=2)
        flat = h.astype(np.float64).reshape(-1, 3) - state.center
        covariance = (flat @ state.encoder).T @ (flat @ state.encoder) / len(flat)
        np.testing.assert_allclose(covariance, np.eye(2), atol=1e-10)
        np.testing.assert_allclose(state.encoder[2], 0., atol=1e-14)

    def test_encoder_deployment_is_pointwise_prefix_invariant(self):
        state = self.fit()
        history = fixture()[0]
        altered = history.copy()
        altered[:, 200:] += 1000.
        np.testing.assert_array_equal(state.encode(history)[:, :200], state.encode(altered)[:, :200])
        self.assertEqual(list(inspect.signature(state.encode).parameters), ["history"])

    def test_tune_future_is_not_an_argument_or_global_dependency(self):
        values = fixture()
        tune_future = np.zeros((7, 48, 3), dtype=np.float32)
        first = self.fit(values)
        tune_future[:] = 1e10
        second = self.fit(values)
        np.testing.assert_array_equal(first.encoder, second.encoder)
        self.assertFalse(any("tune" in key.lower() or "dev" in key.lower()
                             for key in inspect.signature(fit_target_subspace).parameters))

    def test_incomplete_future_positions_are_entirely_excluded(self):
        h, hm, y, ym, last, origins = fixture()
        ym[0, 3, 0] = False
        changed = y.copy()
        changed[0, 3] = [np.nan, 1e20, -1e20]
        a = self.fit((h, hm, y, ym, last, origins))
        b = self.fit((h, hm, changed, ym, last, origins))
        np.testing.assert_array_equal(a.encoder, b.encoder)
        self.assertEqual(b.metadata()["fit_complete_future_positions"], len(h) * 48 - 1)

    def test_observed_future_nonfinite_is_rejected(self):
        values = list(fixture())
        values[2][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "FINITE_COMPLETE"):
            self.fit(values)

    def test_through_origin_target_is_not_centered_future_pca(self):
        h, hm, y, ym, last, origins = fixture()
        changes = np.zeros(y.shape, dtype=np.float64)
        changes[:, :, 0] = 10.
        changes[:, :, 1] = np.tile(np.array([-1., 1.]), (len(h), 24))
        y = (last[:, None] + changes).astype(np.float32)
        state = self.fit((h, hm, y, ym, last, origins))
        v = (y.astype(np.float64) - last[:, None].astype(np.float64)).reshape(-1, 3)
        _, _, centered_right = np.linalg.svd(v - v.mean(axis=0), full_matrices=False)
        target_projector = state.encoder @ np.linalg.pinv(state.encoder)
        self.assertGreater(target_projector[0, 0], .999999)
        self.assertGreater(centered_right[0, 1] ** 2, .999999)

    def test_zero_future_signal_rejected(self):
        values = list(fixture())
        values[2] = np.repeat(values[4][:, None], 48, axis=1)
        with self.assertRaisesRegex(ValueError, "TARGET_INNOVATION_RANK"):
            self.fit(values)

    def test_target_direction_with_no_past_variance_rejected(self):
        values = list(fixture())
        values[0][:, :, 1:] = 0.
        values[4] = values[0][:, -1].copy()
        values[2] = np.repeat(values[4][:, None], 48, axis=1)
        values[2][:, :, 1] = np.arange(48, dtype=np.float32)[None]
        with self.assertRaisesRegex(ValueError, "RESTRICTED_PAST"):
            self.fit(values)

    def test_pinverse_matches_independent_lstsq_and_orthogonal_projector(self):
        state = self.fit()
        e = state.encoder.astype(np.float32).astype(np.float64)
        independent = np.linalg.lstsq(e, np.eye(e.shape[0]), rcond=1e-15)[0].astype(np.float32)
        np.testing.assert_allclose(state.decoder.astype(np.float32), independent, atol=1e-7, rtol=1e-7)
        projector = e @ state.decoder
        np.testing.assert_allclose(projector, projector.T, atol=1e-7)
        np.testing.assert_allclose(state.decoder @ e, np.eye(1), atol=1e-7)

    def test_zero_innovation_keeps_current_C_anchor(self):
        state = self.fit()
        history = fixture()[0]
        latent_last = state.encode(history)[:, -1]
        future = np.repeat(latent_last[:, None], 48, axis=1)
        prediction = state.forecast(history[:, -1], future, latent_last)
        np.testing.assert_array_equal(prediction, np.repeat(history[:, -1, None], 48, axis=1))

    def test_pca_control_preserves_encoder_and_center(self):
        h, hm, *_ = fixture()
        original = fit_projection("PCA", h, hm, 2)
        control = pca_encoder_pinv_control(original)
        np.testing.assert_array_equal(original.encoder, control.encoder)
        np.testing.assert_array_equal(original.center, control.center)
        self.assertEqual(control.metadata()["route"], "PCA_ENCODER_PINV")
        self.assertFalse(control.metadata()["actual_future_target_used_in_projection_or_decoder_fit"])

    def test_metadata_binds_masks_origins_targets_without_claiming_unsupervised(self):
        values = list(fixture())
        values[1][0, 2, 1] = False
        state = self.fit(values)
        metadata = state.metadata()
        self.assertEqual(metadata["learning_class"], "SUPERVISED_FIT_TARGET_SUBSPACE")
        self.assertTrue(metadata["actual_future_target_used_in_projection_or_decoder_fit"])
        self.assertFalse(metadata["independent_holdout"])
        self.assertEqual(set(metadata["fit_input_bindings"]),
                         {"history", "history_observed", "FIT_target", "FIT_target_observed", "raw_last", "origins"})
        self.assertEqual(metadata["FIT_history_observed_counts"][1], len(values[0]) * 512 - 1)

    def test_serialization_and_layout_are_stable(self):
        values = fixture()
        a = self.fit(values)
        b = self.fit(tuple(np.asfortranarray(value) for value in values))
        restored = ProjectionState.from_npz(a.to_npz())
        np.testing.assert_array_equal(a.encoder, b.encoder)
        np.testing.assert_array_equal(a.encode(values[0]), restored.encode(values[0]))
        self.assertEqual(a.metadata()["fit_input_bindings"], b.metadata()["fit_input_bindings"])

    def test_invalid_contracts_rejected(self):
        values = fixture()
        with self.assertRaisesRegex(ValueError, "INTEGER_K"):
            self.fit(values, k=True)
        altered = list(values)
        altered[5] = altered[5].astype(np.int32)
        with self.assertRaisesRegex(ValueError, "ORIGINS"):
            self.fit(altered)
        altered = list(values)
        altered[2] = altered[2][:, :47]
        altered[3] = altered[3][:, :47]
        with self.assertRaisesRegex(ValueError, "N48C"):
            self.fit(altered)


if __name__ == "__main__":
    unittest.main()
