"""SOURCE prepared numeric fixtures; execute only under the fresh runtime gate."""
import inspect
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_stage_native_core_v4 as core


def test_independent_scalar_formula_matches_declared_fp64_then_fp32_order():
    value = np.array([-6., -2.3, -.1, 0., .1, 1.3, 6.], np.float32)
    expected = np.array([.5 * float(v) * (1. + math.erf(float(v) / math.sqrt(2.)))
                         for v in value], np.float32)
    np.testing.assert_array_equal(core.gelu_scalar_formula(value), expected)
    source = inspect.getsource(core.gelu_scalar_formula)
    assert 'stage_separation_math' not in source
    assert 'torch' not in source
    assert core.ATOL == 1e-5 and core.RTOL == 2e-6


def test_torch_ulp_can_be_amplified_by_large_calibration_outgoing_weight():
    # A deterministic bounded toy searches ULP differences, without assets or fit.
    grid = np.linspace(-7., 7., 4097, dtype=np.float32)
    old = core.gelu_torch(grid)
    exact = core.gelu_scalar_formula(grid)
    index = int(np.argmax(np.abs(old.astype(np.float64) - exact)))
    assert old[index] != exact[index]
    z = np.full((1, 48, 1), grid[index], np.float32)
    last = np.zeros((1, 1, 1), np.float32)
    weight = np.zeros((32, 1), np.float32); weight[0, 0] = 1.
    bias = np.zeros(32, np.float32)
    outgoing = np.zeros((33, 1), np.float32); outgoing[1, 0] = 1e7
    old_features = np.concatenate((z-last, core.gelu_torch(z @ weight.T + bias)
                                   - core.gelu_torch(last @ weight.T + bias)), axis=2)
    new_features = core.decoder_features(z, last, {'hidden_weight': weight, 'hidden_bias': bias})
    raw_last = -(new_features @ outgoing)[:, 0]
    old_prediction = raw_last[:, None] + old_features @ outgoing
    new_prediction = raw_last[:, None] + new_features @ outgoing
    # Absolute forecast differences can exceed 1e-5 even for one GELU ULP.
    assert float(np.abs(old_prediction - new_prediction).max()) > core.ATOL
    assert not np.allclose(old_prediction, new_prediction, atol=core.ATOL, rtol=core.RTOL)


def test_calibration_arithmetic_known_synthetic_and_zero_innovation_anchor():
    future = np.full((2, 48, 1), 1., np.float32)
    last = np.zeros((2, 1, 1), np.float32)
    raw_last = np.array([[2.], [-3.]], np.float32)
    weight = np.zeros((32, 1), np.float32); weight[0, 0] = 1.
    bias = np.zeros(32, np.float32)
    outgoing = np.zeros((33, 1), np.float32); outgoing[0, 0] = .5; outgoing[1, 0] = 1.
    arrays = {'hidden_weight': weight, 'hidden_bias': bias, 'effective_weight': outgoing}
    hidden = np.float32(.5 * (1. + math.erf(1. / math.sqrt(2.))))
    expected_delta = np.float32(np.float32(.5) + hidden)
    expected = np.repeat(raw_last[:, None], 48, axis=1) + expected_delta
    np.testing.assert_array_equal(core.decoder_calibration_prediction(future, last, raw_last, arrays), expected)
    np.testing.assert_array_equal(core.decoder_calibration_prediction(np.repeat(last, 48, axis=1), last, raw_last, arrays),
                                  np.repeat(raw_last[:, None], 48, axis=1))


def test_original_decoder_replay_retains_torch_operation_order():
    source = inspect.getsource(core.original_anchored_prediction)
    assert 'F.gelu(F.linear' in source
    assert 'gelu_scalar_formula' not in source
    assert 'last[:, None] + decode(zf) - decode(zl)' in source
