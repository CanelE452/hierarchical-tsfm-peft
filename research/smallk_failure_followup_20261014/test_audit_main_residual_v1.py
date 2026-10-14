"""Synthetic contract tests for audit_main_residual_v1.

These tests use only toy arrays. They do not open producer RESULT files,
scientific NPZs, checkpoints, models, CUDA, or old campaign data.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import audit_main_residual_v1 as audit


def _rows(*values):
    rows = []
    for index, candidate_id in enumerate(audit.CANDIDATE_IDS):
        alpha = None if index < 2 else audit.ALPHAS[index - 2]
        rows.append({"candidate_id": candidate_id, "alpha": alpha, "status": "COMPLETE", "tune_actual_Y_mse": float(values[index])})
    return rows


def test_select_candidate_preserves_zero_mean_then_alpha_order_and_earliest_tie():
    rows = _rows(1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
    assert audit.select_candidate(rows) == 0

    rows = _rows(5.0, 4.0, 3.0, 2.0, 2.0, 7.0, 8.0, 9.0, 10.0)
    assert audit.select_candidate(rows) == 3

    bad = _rows(*range(9))
    bad[3] = {**bad[3], "alpha": 999.0}
    with pytest.raises(audit.AuditFailure, match="ZERO_MEAN_THEN_SEVEN_ALPHA_ORDER_REQUIRED"):
        audit.select_candidate(bad)


def test_observed_score_is_channel_macro_with_nan_safe_mask():
    prediction = np.array([[[1.0, 10.0], [2.0, 20.0]]], dtype=np.float32)
    target = np.array([[[0.0, np.nan], [0.0, 10.0]]], dtype=np.float32)
    observed = np.array([[[True, False], [True, True]]], dtype=np.bool_)
    score = audit.observed_score(prediction, target, observed)
    assert score["observed_counts"] == [2, 1]
    assert math.isclose(score["per_channel_mse"][0], (1.0 + 4.0) / 2.0)
    assert math.isclose(score["per_channel_mse"][1], 100.0)
    assert math.isclose(score["mse"], (((1.0 + 4.0) / 2.0) + 100.0) / 2.0)


def test_apply_model_zero_preserves_main_bits_and_mean_uses_fp32_correction():
    main = np.zeros((1, 48, 2), dtype=np.float32)
    main[0, 0, 1] = np.float32(-0.0)
    main[0, 1:, :] = np.arange(94, dtype=np.float32).reshape(47, 2)
    past = np.zeros((1, 512, 2), dtype=np.float32)
    meta_zero = {"candidate_id": audit.ZERO}
    zero = audit.apply_model("RAW33C", meta_zero, {}, None, None, past, main)
    assert zero.dtype == np.float32
    assert np.array_equal(zero, main)
    assert np.signbit(zero[0, 0, 1]) == np.signbit(main[0, 0, 1])
    plus_zero = main.copy()
    plus_zero[0, 0, 1] = np.float32(0.0)
    with pytest.raises(audit.AuditFailure, match="ARRAY_EXACT_MISMATCH"):
        audit.exact_array(plus_zero, main, "signed-zero")

    correction = np.full((48, 2), np.float32(0.25), dtype=np.float32)
    meta_mean = {"candidate_id": audit.MEAN}
    mean = audit.apply_model("RAW33C", meta_mean, {}, None, {"correction": correction}, past, main)
    assert mean.dtype == np.float32
    assert np.array_equal(mean, main + correction)


def test_feature_hash_contract_includes_dtype_shape_and_fit_raw_bytes():
    center = np.array([1.0, 2.0], dtype=np.float64)
    scale = np.array([3.0, 4.0], dtype=np.float64)
    fit_raw = np.array([[1.0, 2.0], [5.0, 6.0]], dtype=np.float32)
    raw_hash = audit.array_sha_state(fit_raw)
    state_hash = audit.state_sha("MAIN_RESIDUAL_FIT_FEATURE_STATE", "RAW33C", center, scale, 2, len(fit_raw), raw_hash)
    mutated_dtype_hash = audit.array_sha_state(fit_raw.astype(np.float64))
    assert raw_hash != mutated_dtype_hash
    assert state_hash != audit.state_sha("MAIN_RESIDUAL_FIT_FEATURE_STATE", "RAW33C", center, scale, 2, len(fit_raw), mutated_dtype_hash)


def test_coefficient_contract_requires_denominator_nc_not_nhc():
    arrays = {
        "coefficients": np.zeros((3, 48, 2), dtype=np.float64),
        "intercept": np.zeros((48, 2), dtype=np.float64),
        "observed_counts": np.ones((48, 2), dtype=np.int64),
        "channel_counts": np.array([5, 7], dtype=np.int64),
        "normal_equation_denominators": np.broadcast_to(np.array([[5.0, 7.0]], dtype=np.float64), (48, 2)).copy(),
        "design_ranks": np.ones((48, 2), dtype=np.int64),
        "regularized_gram_conditions": np.ones((48, 2), dtype=np.float64),
    }
    audit.verify_coeff_contract("RAW33C", {"candidate_id": "RIDGE_ALPHA_0"}, arrays, channels=2, feature_count=3)

    bad = {**arrays, "normal_equation_denominators": arrays["normal_equation_denominators"] * 48.0}
    with pytest.raises(audit.AuditFailure, match="DENOMINATOR_MUST_BE_NC_NOT_NHC"):
        audit.verify_coeff_contract("RAW33C", {"candidate_id": "RIDGE_ALPHA_0"}, bad, channels=2, feature_count=3)

