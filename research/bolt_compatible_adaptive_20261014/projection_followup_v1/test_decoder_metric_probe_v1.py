"""Analytic and cache-backend synthetic tests; no real model, data or checkpoint."""
import unittest

import numpy as np

import decoder_metric_probe_v1 as probe


class DecoderMetricTests(unittest.TestCase):
    def test_oblique_original_inverse_becomes_orthogonal_encoder_inverse(self):
        e = np.array([[1.], [0.]], np.float32)
        original = np.array([[1., 2.]], np.float32)
        alternative, metadata = probe.decoder_from_encoder(e)
        np.testing.assert_array_equal(alternative, [[1, 0]])
        self.assertGreater(np.max(np.abs(e @ original - (e @ original).T)), 0)
        np.testing.assert_array_equal(e @ alternative, (e @ alternative).T)
        np.testing.assert_array_equal(alternative @ e, np.eye(1))
        self.assertFalse(metadata["future_labels_used_to_construct_D"])

    def test_pseudoinverse_matches_independent_lstsq(self):
        rng = np.random.default_rng(615)
        encoder = rng.normal(size=(7, 3)).astype(np.float32)
        decoder, _ = probe.decoder_from_encoder(encoder)
        independent = np.linalg.lstsq(encoder.astype(np.float64), np.eye(7), rcond=1e-15)[0].astype(np.float32)
        np.testing.assert_array_equal(decoder, independent)
        np.testing.assert_allclose(decoder.astype(np.float64) @ encoder.astype(np.float64), np.eye(3), atol=1e-7)

    def test_rank_deficient_encoder_rejected(self):
        with self.assertRaisesRegex(ValueError, "FULL_COLUMN_RANK"):
            probe.decoder_from_encoder(np.array([[1, 1], [2, 2]], np.float32))

    def test_future_labels_not_in_decoder_fit_api(self):
        import inspect
        self.assertEqual(tuple(inspect.signature(probe.decoder_from_encoder).parameters), ("encoder",))
        with self.assertRaises(TypeError):
            probe.decoder_from_encoder(np.ones((2, 1), np.float32), target=np.ones((1, 48, 2)))

    def test_numpy_cached_forecast_exact_known_affine_innovation(self):
        raw = np.array([[10, 20]], np.float32)
        future = np.array([[[3], [5]]], np.float32)
        last = np.array([[1]], np.float32)
        decoder = np.array([[1, 2]], np.float32)
        predicted = probe.cached_forecast(raw, future, last, decoder, "numpy_FP32_full_N")
        np.testing.assert_array_equal(predicted, [[[12, 24], [14, 28]]])

    def test_zero_delta_current_anchor_exact_all_backends(self):
        rng = np.random.default_rng(616)
        raw = rng.normal(size=(37, 3)).astype(np.float32)
        last = rng.normal(size=(37, 2)).astype(np.float32)
        future = np.repeat(last[:, None], 48, axis=1)
        decoder = rng.normal(size=(2, 3)).astype(np.float32)
        for backend in ("numpy_FP32_full_N", "torch_CPU_FP32_threads2_chunk32"):
            np.testing.assert_array_equal(probe.cached_forecast(raw, future, last, decoder, backend), np.repeat(raw[:, None], 48, axis=1))

    def test_native_cached_batch32_replays_direct_torch_chunk32(self):
        import torch
        torch.set_num_threads(2)
        rng = np.random.default_rng(617)
        raw = rng.normal(size=(67, 5)).astype(np.float32)
        future = rng.normal(size=(67, 48, 3)).astype(np.float32)
        last = rng.normal(size=(67, 3)).astype(np.float32)
        decoder = rng.normal(size=(3, 5)).astype(np.float32)
        expected = []
        with torch.no_grad():
            for start in range(0, 67, 32):
                expected.append((torch.from_numpy(raw[start:start + 32, None]) +
                    (torch.from_numpy(future[start:start + 32]) - torch.from_numpy(last[start:start + 32, None])) @ torch.from_numpy(decoder)).numpy())
        actual = probe.cached_forecast(raw, future, last, decoder, "torch_CPU_FP32_threads2_chunk32")
        np.testing.assert_array_equal(actual, np.concatenate(expected))

    def test_masked_scores_ignore_unobserved_future_nan(self):
        target = np.array([[[1, np.nan], [3, 4]]], np.float32)
        prediction = np.array([[[2, 1000], [1, 5]]], np.float32)
        mask = np.array([[[True, False], [True, True]]])
        scored = probe.masked_scores(prediction, target, mask, np.ones(2, np.float32))
        self.assertAlmostEqual(scored["normalized"]["mse"], 1.75)
        self.assertEqual(scored["normalized"]["observed_counts"], [2, 1])

    def test_backend_shape_and_invalid_tag_rejected(self):
        for backend in ("BAD", "numpy_FP32_full_N"):
            with self.assertRaises(ValueError):
                probe.cached_forecast(np.zeros((1, 2), np.float32), np.zeros((1, 48, 1), np.float32),
                                      np.zeros((1, 1), np.float32), np.zeros((2, 2), np.float32), backend)

    def test_nonfinite_encoder_rejected(self):
        with self.assertRaisesRegex(ValueError, "FINITE_CK"):
            probe.decoder_from_encoder(np.array([[np.nan], [1]], np.float32))

    def test_decoder_construction_and_prediction_do_not_change_cache(self):
        rng = np.random.default_rng(618)
        encoder = rng.normal(size=(3, 2)).astype(np.float32)
        future = rng.normal(size=(2, 48, 2)).astype(np.float32)
        original_encoder, original_future = encoder.copy(), future.copy()
        decoder, _ = probe.decoder_from_encoder(encoder)
        probe.cached_forecast(np.zeros((2, 3), np.float32), future, np.zeros((2, 2), np.float32), decoder, "numpy_FP32_full_N")
        np.testing.assert_array_equal(encoder, original_encoder)
        np.testing.assert_array_equal(future, original_future)


if __name__ == "__main__":
    unittest.main()
