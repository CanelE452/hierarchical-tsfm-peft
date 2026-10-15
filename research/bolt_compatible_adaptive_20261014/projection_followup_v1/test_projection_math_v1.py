"""Small synthetic fixtures only; no real files, model, or runtime reads."""
from io import BytesIO
import inspect
import unittest

import numpy as np

import projection_math_v1 as projection


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(61014)
        self.history = self.rng.normal(size=(3, 512, 3)).astype(np.float32)
        self.observed = np.ones(self.history.shape, dtype=bool)

    def test_pca_unit_variance_and_orthogonality(self):
        state = projection.fit_projection("PCA", self.history, self.observed, 2)
        latent = state.encode(self.history).reshape(-1, 2).astype(np.float64)
        np.testing.assert_allclose(latent.mean(0), 0, atol=1e-7)
        np.testing.assert_allclose(latent.T @ latent / len(latent), np.eye(2), atol=3e-7)

    def test_duplicate_channels_rank_one_exact_reconstruction(self):
        base = self.history[:, :, :1]
        x = np.concatenate((base, base, 3 * base + 2), axis=-1)
        state = projection.fit_projection("PCA", x, self.observed, 1)
        self.assertEqual(state.metadata()["whitening_rank"], 1)
        np.testing.assert_allclose(state.decode(state.encode(x)), x, atol=2e-6)

    def test_duplicate_rank_below_k_rejected(self):
        x = np.repeat(self.history[:, :, :1], 3, axis=-1)
        with self.assertRaisesRegex(ValueError, "WHITENING_RANK_BELOW_K"):
            projection.fit_projection("FORECA", x, self.observed, 2)

    def test_static_variance_zero_rejected(self):
        with self.assertRaisesRegex(ValueError, "ZERO_OR_NONFINITE"):
            projection.fit_projection("PCA", np.ones_like(self.history), self.observed, 1)

    def test_nonfinite_history_rejected(self):
        x = self.history.copy()
        x[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "FINITE_NONEMPTY"):
            projection.fit_projection("PCA", x, self.observed, 1)

    def test_fit_dtype_shape_and_mask_contract(self):
        with self.assertRaises(ValueError):
            projection.fit_projection("PCA", self.history.astype(np.float64), self.observed, 1)
        with self.assertRaises(ValueError):
            projection.fit_projection("PCA", self.history[:, :511], self.observed[:, :511], 1)
        with self.assertRaisesRegex(ValueError, "BOOL_OBSERVED"):
            projection.fit_projection("PCA", self.history, self.observed.astype(np.uint8), 1)

    def test_future_targets_absent_from_fit_api(self):
        self.assertEqual(tuple(inspect.signature(projection.fit_projection).parameters),
                         ("route", "history", "history_observed", "k"))
        with self.assertRaises(TypeError):
            projection.fit_projection("PCA", self.history, self.observed, 1, future_target=self.history)

    def test_decoder_mask_matches_independent_observed_least_squares(self):
        mask = self.observed.copy()
        mask[0, :400, 1] = False
        state = projection.fit_projection("PCA", self.history, mask, 2)
        latent = (self.history.astype(np.float64).reshape(-1, 3) - state.center) @ state.encoder
        design = np.column_stack((latent, np.ones(len(latent))))
        keep = mask.reshape(-1, 3)[:, 1]
        expected = np.linalg.lstsq(design[keep], self.history.reshape(-1, 3)[keep, 1].astype(np.float64), rcond=None)[0]
        np.testing.assert_allclose(state.decoder[:, 1], expected[:-1], atol=1e-12)
        np.testing.assert_allclose(state.decoder_bias[1], expected[-1], atol=1e-12)
        self.assertEqual(state.metadata()["decoder_observed_counts"][1], int(keep.sum()))

    def test_all_unobserved_decoder_channel_rejected(self):
        mask = self.observed.copy()
        mask[:, :, 2] = False
        with self.assertRaisesRegex(ValueError, "WITHOUT_OBSERVED"):
            projection.fit_projection("PCA", self.history, mask, 1)

    def test_hann_dc_nyquist_and_interior_power_weights(self):
        times = np.arange(512)
        x = np.stack((np.ones(512), (-1.) ** times, np.cos(2 * np.pi * times * 31 / 512)), axis=-1)[None]
        spectrum = projection.one_sided_cross_spectrum(x)
        hann = np.hanning(512)
        transformed = np.fft.rfft(x * hann[None, :, None], axis=1)
        base = np.abs(transformed[0]) ** 2 / (512 * np.sum(hann * hann))
        diagonal = np.diagonal(spectrum, axis1=1, axis2=2)
        np.testing.assert_allclose(diagonal[0], base[0], atol=1e-15)
        np.testing.assert_allclose(diagonal[-1], base[-1], atol=1e-15)
        np.testing.assert_allclose(diagonal[1:-1], 2 * base[1:-1], atol=1e-15)
        np.testing.assert_allclose(diagonal.sum(0), np.sum((x[0] * hann[:, None]) ** 2, axis=0) / np.sum(hann * hann), atol=1e-12)

    def test_spectral_reflect_smoothing(self):
        values = np.arange(257, dtype=np.float64)[:, None, None]
        smooth = projection.smooth_cross_spectrum(values)
        self.assertAlmostEqual(smooth[0, 0, 0], (3 + 2 + 1 + 0 + 1 + 2 + 3) / 7)
        self.assertAlmostEqual(smooth[100, 0, 0], 100)

    def test_spectral_entropy_scale_invariance_and_gradient(self):
        spectrum = projection.smooth_cross_spectrum(projection.one_sided_cross_spectrum(self.history))
        direction = np.array((.2, -.3, .7))
        value, derivative = projection.spectral_entropy(spectrum, direction, gradient=True)
        self.assertAlmostEqual(value, projection.spectral_entropy(spectrum, 7 * direction), places=12)
        numeric = []
        for index in range(3):
            step = np.zeros(3); step[index] = 1e-6
            numeric.append((projection.spectral_entropy(spectrum, direction + step) - projection.spectral_entropy(spectrum, direction - step)) / 2e-6)
        np.testing.assert_allclose(derivative, numeric, atol=2e-8, rtol=1e-5)

    def test_foreca_picks_periodic_signal_and_two_starts_are_monotone(self):
        times = np.arange(512)
        signal = np.sin(2 * np.pi * times / 32)
        x = np.stack((np.tile(signal, (3, 1)), self.rng.normal(size=(3, 512))), axis=-1).astype(np.float32)
        state = projection.fit_projection("FORECA", x, np.ones(x.shape, dtype=bool), 1)
        latent = state.encode(x).reshape(-1)
        correlation = np.corrcoef(latent, np.tile(signal, 3))[0, 1]
        self.assertGreater(abs(correlation), .98)
        starts = state.metadata()["foreca_search"][0]["starts"]
        self.assertEqual(len(starts), 2)
        self.assertTrue(all(row["final_entropy"] <= row["initial_entropy"] + 1e-12 for row in starts))
        self.assertTrue(all(row["iterations"] <= 200 for row in starts))

    def test_foreca_all_directions_unit_variance_and_determinism(self):
        a = projection.fit_projection("FORECA", self.history, self.observed, 3)
        b = projection.fit_projection("FORECA", self.history, self.observed, 3)
        np.testing.assert_array_equal(a.encoder, b.encoder)
        latent = a.encode(self.history).reshape(-1, 3).astype(np.float64)
        np.testing.assert_allclose(latent.T @ latent / len(latent), np.eye(3), atol=4e-7)
        np.testing.assert_allclose(a.decode(a.encode(self.history)), self.history, atol=2e-6)

    def test_serialization_and_memory_layout_preserve_deployment(self):
        a = projection.fit_projection("PCA", self.history, self.observed, 2)
        b = projection.fit_projection("PCA", np.asfortranarray(self.history), np.asfortranarray(self.observed), 2)
        np.testing.assert_array_equal(a.encoder, b.encoder)
        buffer = BytesIO()
        np.savez(buffer, **a.to_npz()); buffer.seek(0)
        with np.load(buffer, allow_pickle=False) as saved:
            restored = projection.ProjectionState.from_npz(saved)
        np.testing.assert_array_equal(a.encode(self.history), restored.encode(self.history))
        np.testing.assert_array_equal(a.decode(a.encode(self.history)), restored.decode(restored.encode(self.history)))
        self.assertEqual(a.metadata(), restored.metadata())

    def test_forecast_anchor_bias_cancellation_and_fp32(self):
        state = projection.fit_projection("PCA", self.history + 3, self.observed, 2)
        z = state.encode(self.history + 3)
        last = (self.history + 3)[:, -1:]
        constant_future = np.repeat(z[:, -1:], 48, axis=1)
        forecast = state.forecast(last, constant_future, z[:, -1:])
        self.assertEqual(forecast.dtype, np.float32)
        np.testing.assert_allclose(forecast, np.repeat(last, 48, axis=1), atol=1e-6)
        self.assertFalse(state.metadata()["strict_K_only_compression"])
        self.assertFalse(state.metadata()["actual_future_target_used_in_projection_or_decoder_fit"])

    def test_invalid_k_route_and_serialized_state_rejected(self):
        for route, k in (("BAD", 1), ("PCA", True), ("PCA", 0), ("PCA", 4)):
            with self.assertRaises(ValueError):
                projection.fit_projection(route, self.history, self.observed, k)
        state = projection.fit_projection("PCA", self.history, self.observed, 2)
        saved = state.to_npz(); saved["decoder"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "FINITE_ERROR"):
            projection.ProjectionState.from_npz(saved)


if __name__ == "__main__":
    unittest.main()
