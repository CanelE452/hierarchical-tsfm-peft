"""Small synthetic-only arithmetic and coordinate checks for R1."""

import unittest

import numpy as np
from scipy.interpolate import CubicSpline
import torch

from transforms import (
    _natural_cubic_weights,
    _sprint_trend,
    r1_metadata,
    r1_predict,
    trailing_mean,
)


class LinearContinuation:
    def __call__(self, x, horizon=48):
        step = x[:, -1:] - x[:, -2:-1]
        t = torch.arange(1, horizon + 1, dtype=x.dtype, device=x.device)
        return x[:, -1:] + step * t[None, :, None]


class LastValue:
    def __call__(self, x, horizon=48):
        return x[:, -1:].expand(-1, horizon, -1)


class TestTransforms(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)

    def test_trailing_is_past_only_and_additive(self):
        x = torch.randn(2, 32, 3)
        for width in (5, 25):
            trend = trailing_mean(x, width)
            for i in range(x.shape[1]):
                expected = x[:, max(0, i - width + 1) : i + 1].mean(dim=1)
                torch.testing.assert_close(trend[:, i], expected, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(trend + (x - trend), x, atol=3e-7, rtol=1e-6)
            changed = x.clone()
            changed[:, 17:] += 100
            torch.testing.assert_close(trailing_mean(changed, width)[:, :17], trend[:, :17])

    def test_endpoint_physical_coordinates(self):
        x = torch.arange(512, dtype=torch.float32)[None, :, None]
        expected = torch.arange(512, 560, dtype=torch.float32)[None, :, None]
        for variant in ("A7", "A8"):
            torch.testing.assert_close(r1_predict(x, LinearContinuation(), variant), expected, atol=0, rtol=0)

    def test_block_mean_is_not_endpoint(self):
        x = torch.arange(512, dtype=torch.float32)[None, :, None]
        for variant, stride in (("A9", 2), ("A10", 4)):
            expected = torch.arange(512, 560, dtype=torch.float32).reshape(-1, stride).mean(dim=1).repeat_interleave(stride)
            actual = r1_predict(x, LinearContinuation(), variant)[0, :, 0]
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            self.assertNotEqual(actual[0].item(), 512.0)

    def test_component_anchoring_with_last_value_predictor(self):
        x = torch.randn(2, 512, 3)
        expected = x[:, -1:].expand(-1, 48, -1)
        for variant in ("A1", "A2", "A3", "A4", "A5", "A6"):
            torch.testing.assert_close(r1_predict(x, LastValue(), variant), expected, atol=3e-7, rtol=1e-6)

    def test_fixed_horizon_callable_fallback(self):
        def fixed(x):
            return x[:, -1:].expand(-1, 48, -1)

        x = torch.ones(1, 512, 2)
        torch.testing.assert_close(r1_predict(x, fixed, "A8"), torch.ones(1, 48, 2))

    def test_natural_cubic_matches_scipy(self):
        for length, stride in ((480, 6), (432, 36), (512, 2)):
            horizon = 48
            coarse_future = horizon // stride + 2
            coordinates = np.concatenate((np.arange(-length, 0, stride), np.arange(coarse_future) * stride))
            y = torch.randn(2, len(coordinates), 3, dtype=torch.float64)
            weights = _natural_cubic_weights(length, stride, coarse_future, horizon)
            actual = torch.einsum("qn,bnc->bqc", weights, y).numpy()
            expected = CubicSpline(coordinates, y.numpy(), axis=1, bc_type="natural")(np.arange(horizon))
            np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)

    def test_sprint_manual_reference_and_trim(self):
        x = torch.randn(2, 512, 3, dtype=torch.float64)
        period, horizon = 24, 48
        length = 504
        z = x[:, -length:]
        mean = z.mean(dim=1, keepdim=True)
        std = (z.var(dim=1, keepdim=True) + 1e-5).sqrt()
        z = (z - mean) / std
        trend = _sprint_trend(z, period)
        seasonal = z - trend
        cycles = length // period
        weight = 0.9 ** torch.arange(cycles, 0, -1, dtype=z.dtype)
        seasonal_reference = (seasonal.reshape(2, cycles, period, 3) * (weight / weight.sum())[None, :, None, None]).sum(dim=1)
        seasonal_reference = seasonal_reference.repeat(1, 2, 1)
        stride = 6
        coarse = trend[:, ::stride]
        coarse_future = horizon // stride + 2
        forecast = LastValue()(coarse, coarse_future)
        support = torch.cat((coarse, forecast), dim=1).numpy()
        coordinates = np.concatenate((np.arange(-length, 0, stride), np.arange(coarse_future) * stride))
        interpolated = CubicSpline(coordinates, support, axis=1, bc_type="natural")(np.arange(horizon))
        expected = (torch.from_numpy(interpolated) + seasonal_reference) * std + mean
        actual = r1_predict(x, LastValue(), "A11", period=period)
        torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
        metadata = r1_metadata("A11", period=period, channels=3)
        self.assertEqual(metadata["effective_history_length"], 504)
        self.assertEqual(metadata["discarded_left_rows"], 8)
        self.assertEqual(metadata["foundation_requested_horizon"], 10)

    def test_sprint_constant_is_constant(self):
        x = torch.full((2, 512, 3), 7.25)
        for period in (24, 144):
            actual = r1_predict(x, LastValue(), "A11", period=period)
            torch.testing.assert_close(actual, torch.full((2, 48, 3), 7.25), atol=1e-6, rtol=0)

    def test_shapes_and_stream_metadata(self):
        x = torch.randn(2, 512, 3)
        for i in range(14):
            variant = f"A{i}"
            actual = r1_predict(x, LastValue(), variant, period=24)
            self.assertEqual(actual.shape, (2, 48, 3))
            self.assertTrue(torch.isfinite(actual).all())
            info = r1_metadata(variant, channels=3, period=24)
            self.assertEqual(info["streams_per_origin"], 6 if i in (5, 6) else 3)

    def test_invalid_sprint_period_is_explicit(self):
        for period in (None, 0, 4, 7, 13, 24.5, 1024):
            with self.assertRaises(ValueError):
                r1_predict(torch.ones(1, 512, 1), LastValue(), "A11", period=period)


if __name__ == "__main__":
    unittest.main()
