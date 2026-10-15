"""Training-free R1 transforms on a fixed physical forecast horizon.

SPRINT-on-Bolt reimplements the MIT-licensed official SPRINT workflow:
Copyright 2026 Tsinghua University and ByteDance. SPDX-License-Identifier: MIT.
Pinned upstream commit: 1cc1274719458f0cc820d922b974ae2b2e040306.
This is a backbone port, not reproduction of the official Chronos-T5 results.

Upstream MIT permission notice:
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from functools import lru_cache
import inspect
import math

import torch
import torch.nn.functional as functional


SPRINT_COMMIT = "1cc1274719458f0cc820d922b974ae2b2e040306"
SPRINT_BLOB = "623494941d55f1b537c8feafcb3757191c316de8"
CHRONOS_BLOB = "cb89b8f54c0fda5804e25f781f082d65108f5464"


def trailing_mean(x, width):
    """At each historical row, average only that row and earlier rows."""
    if width < 1:
        raise ValueError("Moving-average width must be positive")
    length = x.shape[1]
    prefix = torch.cat((torch.zeros_like(x[:, :1]), x.cumsum(dim=1)), dim=1)
    end = torch.arange(1, length + 1, device=x.device)
    start = (end - width).clamp_min(0)
    return (prefix[:, end] - prefix[:, start]) / (end - start)[None, :, None]


def _forecast(bolt, x, horizon):
    callable_signature = inspect.signature(bolt.forward if hasattr(bolt, "forward") else bolt)
    parameters = callable_signature.parameters
    if "horizon" in parameters:
        result = bolt(x, horizon=horizon)
    elif "prediction_length" in parameters:
        result = bolt(x, prediction_length=horizon)
    else:
        result = bolt(x)
    if result.ndim != 3 or result.shape[0] != x.shape[0] or result.shape[2] != x.shape[2]:
        raise ValueError("The Bolt callable must return [batch, horizon, channels]")
    if result.shape[1] < horizon:
        raise ValueError(f"Bolt returned {result.shape[1]} rows; {horizon} required")
    return result[:, :horizon]


def endpoint_restore(current, coarse_prediction, stride, horizon):
    """Coarse samples end at o-1, so future support is o+s-1, o+2s-1, ..."""
    if stride < 1 or coarse_prediction.shape[1] * stride < horizon:
        raise ValueError("Insufficient endpoint support for the physical horizon")
    support = torch.cat((current[:, None], coarse_prediction), dim=1)
    physical_step = torch.arange(1, horizon + 1, device=support.device)
    left = torch.div(physical_step, stride, rounding_mode="floor")
    right = (left + 1).clamp_max(support.shape[1] - 1)
    fraction = (physical_step % stride).to(support.dtype) / stride
    return support[:, left] * (1 - fraction[None, :, None]) + support[:, right] * fraction[None, :, None]


def _sprint_trend(x, period):
    # Exact official MA padding: even kernel has one fewer repeated row on the left.
    width = period // 2
    front = x[:, :1].expand(-1, width // 2 - 1, -1)
    back = x[:, -1:].expand(-1, width // 2, -1)
    padded = torch.cat((front, x, back), dim=1).transpose(1, 2)
    return functional.avg_pool1d(padded, kernel_size=width, stride=1).transpose(1, 2)


@lru_cache(maxsize=32)
def _natural_cubic_weights(length, stride, future_coarse, horizon):
    """CPU double-precision interpolation weights; no learned/data statistics."""
    history_coordinates = torch.arange(-length, 0, stride, dtype=torch.float64)
    forecast_coordinates = torch.arange(future_coarse, dtype=torch.float64) * stride
    support = torch.cat((history_coordinates, forecast_coordinates))
    intervals = support.diff()
    count = support.numel()
    system = torch.zeros((count, count), dtype=torch.float64)
    difference = torch.zeros_like(system)
    system[0, 0] = 1
    system[-1, -1] = 1
    rows = torch.arange(1, count - 1)
    system[rows, rows - 1] = intervals[:-1]
    system[rows, rows] = 2 * (intervals[:-1] + intervals[1:])
    system[rows, rows + 1] = intervals[1:]
    difference[rows, rows - 1] = 6 / intervals[:-1]
    difference[rows, rows] = -6 * (1 / intervals[:-1] + 1 / intervals[1:])
    difference[rows, rows + 1] = 6 / intervals[1:]
    second_derivative_map = torch.linalg.solve(system, difference)
    query = torch.arange(horizon, dtype=torch.float64)
    left = torch.searchsorted(support, query, right=True).sub(1).clamp(0, count - 2)
    right = left + 1
    widths = support[right] - support[left]
    right_fraction = (query - support[left]) / widths
    left_fraction = 1 - right_fraction
    weights = torch.zeros((horizon, count), dtype=torch.float64)
    query_rows = torch.arange(horizon)
    weights[query_rows, left] = left_fraction
    weights[query_rows, right] = right_fraction
    weights += ((left_fraction**3 - left_fraction) * widths**2 / 6)[:, None] * second_derivative_map[left]
    weights += ((right_fraction**3 - right_fraction) * widths**2 / 6)[:, None] * second_derivative_map[right]
    return weights


@lru_cache(maxsize=64)
def _device_cubic_weights(length, stride, future_coarse, horizon, device, dtype):
    return _natural_cubic_weights(length, stride, future_coarse, horizon).to(device=device, dtype=dtype)


def sprint_predict(x, bolt, period, horizon=48, use_revin=True):
    """Official SPRINT decomposition/replication/spline, with frozen Bolt replacing T5."""
    if period is None or int(period) != period or period < 8 or period % 4:
        raise ValueError("SPRINT requires a predeclared period >=8 divisible by four")
    period = int(period)
    length = (x.shape[1] // period) * period
    if length < period:
        raise ValueError("SPRINT has no complete observed period")
    x = x[:, -length:]
    if use_revin:
        mean = x.mean(dim=1, keepdim=True)
        # Upstream torch.var uses correction=1; preserve that convention.
        standard_deviation = (x.var(dim=1, keepdim=True, correction=1) + 1e-5).sqrt()
        x = (x - mean) / standard_deviation
    trend = _sprint_trend(x, period)
    seasonal = x - trend
    cycles = length // period
    weights = 0.9 ** torch.arange(cycles, 0, -1, device=x.device, dtype=x.dtype)
    weights = weights / weights.sum()
    pattern = (seasonal.reshape(x.shape[0], cycles, period, x.shape[2]) * weights[None, :, None, None]).sum(dim=1)
    seasonal_future = pattern.repeat(1, math.ceil(horizon / period), 1)[:, :horizon]
    stride = period // 4
    downsampled_trend = trend[:, ::stride]
    future_coarse = horizon // stride + 2
    predicted_trend = _forecast(bolt, downsampled_trend, future_coarse)
    support = torch.cat((downsampled_trend, predicted_trend), dim=1)
    interpolation = _device_cubic_weights(length, stride, future_coarse, horizon, x.device, x.dtype)
    result = torch.einsum("qn,bnc->bqc", interpolation, support) + seasonal_future
    if use_revin:
        result = result * standard_deviation + mean
    return result


def r1_predict(x, bolt, variant="A0", horizon=48, period=None, use_revin=True):
    """Return original-grid future predictions; x contains strictly pre-origin data."""
    if x.ndim != 3 or x.shape[1] < 1:
        raise ValueError("x must have shape [batch, observed past length, channels]")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if variant in ("A0", "A12", "A13"):
        return _forecast(bolt, x, horizon)
    if variant in ("A1", "A2", "A3", "A4", "A5", "A6"):
        width = 5 if variant in ("A1", "A3", "A5") else 25
        trend = trailing_mean(x, width)
        residual = x - trend
        if variant in ("A1", "A2"):
            return _forecast(bolt, trend, horizon) + residual[:, -1:]
        if variant in ("A3", "A4"):
            return _forecast(bolt, residual, horizon) + trend[:, -1:]
        return _forecast(bolt, trend, horizon) + _forecast(bolt, residual, horizon)
    if variant in ("A7", "A8", "A9", "A10"):
        stride = 2 if variant in ("A7", "A9") else 4
        if x.shape[1] % stride:
            raise ValueError("R1 uniform/block transforms require length divisible by stride")
        coarse_horizon = math.ceil(horizon / stride)
        if variant in ("A7", "A8"):
            coarse = x[:, stride - 1 :: stride]
            prediction = _forecast(bolt, coarse, coarse_horizon)
            return endpoint_restore(x[:, -1], prediction, stride, horizon)
        coarse = x.reshape(x.shape[0], -1, stride, x.shape[2]).mean(dim=2)
        prediction = _forecast(bolt, coarse, coarse_horizon)
        return prediction.repeat_interleave(stride, dim=1)[:, :horizon]
    if variant == "A11":
        return sprint_predict(x, bolt, period, horizon, use_revin)
    raise ValueError(f"Unknown R1 variant: {variant}")


def r1_metadata(variant, length=512, channels=1, horizon=48, period=None, use_revin=True):
    result = {
        "variant": variant,
        "physical_history_length": int(length),
        "physical_horizon": int(horizon),
        "streams_per_origin": int(channels * (2 if variant in ("A5", "A6") else 1)),
        "foundation_calls_per_origin_batch": 2 if variant in ("A5", "A6") else 1,
        "target_grid": "original physical grid, all methods scored against unchanged Y",
        "future_information_used": False,
        "effective_history_length": int(length),
        "foundation_context_length": int(length),
        "foundation_requested_horizon": int(horizon),
        "foundation_actual_head_compute": "caller must record actual Bolt head length; requested crop is not compute reduction",
    }
    if variant in ("A7", "A8", "A9", "A10"):
        stride = 2 if variant in ("A7", "A9") else 4
        result.update(
            stride=stride,
            foundation_context_length=int(length // stride),
            foundation_requested_horizon=int(math.ceil(horizon / stride)),
            coarse_observation_semantics="endpoint" if variant in ("A7", "A8") else "nonoverlapping block mean",
            restoration="linear from last observed endpoint at o-1" if variant in ("A7", "A8") else "repeat forecast block mean across its future block",
        )
    if variant in ("A1", "A2", "A3", "A4", "A5", "A6"):
        result["moving_average_width"] = 5 if variant in ("A1", "A3", "A5") else 25
        result["decomposition"] = "past-only trailing mean, truncated window at left edge; residual=x-trend"
    if variant == "A11":
        if period is None or int(period) != period or period < 8 or period % 4:
            raise ValueError("SPRINT requires a predeclared period >=8 divisible by four")
        period = int(period)
        effective_length = (length // period) * period
        if effective_length < period:
            raise ValueError("SPRINT has no complete observed period")
        result.update(
            period=period,
            period_source="must be set from A only or pre-existing sampling metadata",
            effective_history_length=int(effective_length),
            discarded_left_rows=int(length - effective_length),
            stride=period // 4,
            foundation_context_length=int(effective_length // (period // 4)),
            foundation_requested_horizon=int(horizon // (period // 4) + 2),
            restoration="natural cubic spline on official support coordinates plus weighted period replication",
            use_revin=bool(use_revin),
            sprint_commit=SPRINT_COMMIT,
            sprint_blob=SPRINT_BLOB,
            original_chronos_blob=CHRONOS_BLOB,
            interpretation="SPRINT-on-Bolt backbone port; not official Chronos-T5 reproduction",
            required_control=f"Raw Bolt on same {effective_length} historical rows and origins",
        )
    return result
