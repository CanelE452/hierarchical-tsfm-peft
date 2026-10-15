"""Toy-only lag orientation, observed LS, joint coupling and missing-horizon tests."""
import inspect
import json

import numpy as np
import pytest

import causal_lag_decoder_v1 as lag


def test_delta_features_are_lag_major_with_correct_past_orientation():
    history = np.arange(40*2, dtype=np.float32).reshape(1, 40, 2)
    future = np.arange(100, 112, dtype=np.float32).reshape(1, 6, 2)
    actual = lag.lagged_delta_features(history, future, [0, 1, 2, 4])
    expected = np.empty((1, 6, 8), np.float32)
    for h in range(1, 7):
        pieces = []
        for ell in (0, 1, 2, 4):
            value = future[:, h-ell-1] if h > ell else history[:, 39+h-ell]
            pieces.append(value-history[:, 39-ell])
        expected[:, h-1] = np.concatenate(pieces, axis=1)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[:, 0, 2:4], history[:, -1]-history[:, -2])


def test_L0_prediction_is_original_FP32_current_anchor_byte_exact():
    rng = np.random.default_rng(711)
    z, future, current, d = (rng.normal(size=shape).astype(np.float32)
        for shape in ((3, 512, 2), (3, 48, 2), (3, 4), (2, 4)))
    expected = current[:, None]+(future-z[:, -1:])@d
    assert lag.predict(z, future, current, d, [0]).tobytes() == expected.tobytes()


def test_flat_future_known_prefix_is_not_currentC_persistence_for_lagged_D():
    z = np.arange(512, dtype=np.float32)[None, :, None]
    flat = np.repeat(z[:, -1:], 6, axis=1)
    current = np.array([[10.]], np.float32)
    d = np.array([[0.], [1.]], np.float32)
    prediction = lag.predict(z, flat, current, d, [0, 4])
    np.testing.assert_array_equal(prediction[0, :, 0], [11., 12., 13., 14., 14., 14.])
    np.testing.assert_array_equal(lag.predict(z, flat, current, np.array([[2.]], np.float32), [0]), np.full((1, 6, 1), 10., np.float32))


def test_delayed_copy_reconstruction_identifies_direction_and_common480_rows():
    rng = np.random.default_rng(99)
    z = rng.normal(size=(2, 512, 1)).astype(np.float32)
    history = np.concatenate((z, np.roll(z, 4, axis=1)), axis=2)
    mask = np.ones(history.shape, bool)
    fitted = lag.fit_reconstruction(history, mask, z, [0, 4])
    np.testing.assert_allclose(fitted["decoder_FP64"], np.eye(2), atol=2e-15)
    np.testing.assert_allclose(fitted["bias_FP64"], np.zeros(2), atol=1e-15)
    assert max(fitted["metadata"]["direct_residual_MSE"]) < 1e-28
    point = lag.fit_reconstruction(history, mask, z, [0])
    assert point["metadata"]["direct_residual_MSE"][1] > .5
    assert fitted["metadata"]["true_observed_counts"] == point["metadata"]["true_observed_counts"] == [960, 960]
    changed = history.copy(); changed[:, :32] = 1e6
    for lags in ([0], [0, 4]):
        original = lag.fit_reconstruction(history, mask, z, lags)
        replay = lag.fit_reconstruction(changed, mask, z, lags)
        assert replay["decoder_FP64"].tobytes() == original["decoder_FP64"].tobytes()
        assert replay["bias_FP64"].tobytes() == original["bias_FP64"].tobytes()
        assert replay["metadata"]["common_history_slice"] == [32, 512]


def test_RECON_observation_masks_constant_response_and_noobs_are_distinct():
    rng = np.random.default_rng(343)
    z = rng.normal(size=(1, 512, 2)).astype(np.float32)
    history = rng.normal(size=(1, 512, 5)).astype(np.float32)
    history[:, :, 3] = 7.
    mask = rng.random(history.shape) > .2
    mask[:, :, 1] = False; mask[:, :, 2] = False; mask[0, 100, 2] = True; mask[:, :, 3] = True
    fitted = lag.fit_reconstruction(history, mask, z, [0]); meta = fitted["metadata"]
    assert meta["no_observed_response"][1] and not meta["fit_executed"][1]
    assert meta["one_observed_response"][2] and meta["rank"][2] == 0
    assert "ONE_OBSERVATION" in meta["fallback_reason"][2] and fitted["bias_FP64"][2] == history[0, 100, 2]
    assert meta["constant_observed_response"][3] and meta["full_slope_identified"][3]
    assert meta["fallback_reason"][3] is None and meta["rank"][3] == 2
    np.testing.assert_array_equal(fitted["decoder_FP64"][:, 3], np.zeros(2))
    changed = history.copy(); changed[~mask] = 1e20
    replay = lag.fit_reconstruction(changed, mask, z, [0])
    assert fitted["decoder_FP64"].tobytes() == replay["decoder_FP64"].tobytes()
    assert fitted["bias_FP64"].tobytes() == replay["bias_FP64"].tobytes()
    features = z[:, 32:].reshape(-1, 2).astype(np.float64)
    for channel in (0, 4):
        valid = mask[:, 32:, channel].reshape(-1); y = history[:, 32:, channel].reshape(-1).astype(np.float64)[valid]
        a = features[valid]; centered = a-a.mean(axis=0)
        slope, _, rank, _ = np.linalg.lstsq(centered, y-y.mean(), rcond=1e-4)
        np.testing.assert_array_equal(fitted["decoder_FP64"][:, channel], slope)
        assert meta["rank"][channel] == rank


def test_FORECAST_equals_direct_noncentered_actual_delta_LS_per_mask():
    rng = np.random.default_rng(84)
    history = rng.integers(-4, 5, size=(2, 512, 2)).astype(np.float32)
    future = rng.integers(-4, 5, size=(2, 7, 2)).astype(np.float32)
    current = np.array([[2., 3., 4.], [-1., 0., 1.]], np.float32)
    features = lag.lagged_delta_features(history, future, [0, 1, 4])
    true_D = rng.integers(-2, 3, size=(6, 3)).astype(np.float32)
    target = current[:, None]+features@true_D
    observed = rng.random(target.shape) > .15
    fitted = lag.fit_forecast(history, future, current, target, observed, [0, 1, 4])
    for channel in range(3):
        valid = observed[:, :, channel].reshape(-1)
        a = features.reshape(-1, 6)[valid].astype(np.float64)
        y = (target.astype(np.float64)-current.astype(np.float64)[:, None]).reshape(-1, 3)[valid, channel]
        expected, _, rank, singular = np.linalg.lstsq(a, y, rcond=1e-4)
        np.testing.assert_array_equal(fitted["decoder_FP64"][:, channel], expected)
        assert fitted["metadata"]["rank"][channel] == rank
        np.testing.assert_array_equal(fitted["metadata"]["singularvalues"][channel], singular)
    np.testing.assert_array_equal(fitted["bias_FP64"], np.zeros(3))
    assert fitted["metadata"]["future_FIT_Y_used_as_supervision"] is True
    assert fitted["metadata"]["TUNE_DEV_used_for_fit"] is False


def test_FORECAST_target_difference_is_FP64_and_constant_nonzero_design_is_identifiable():
    history, future = np.zeros((1, 512, 1), np.float32), np.ones((1, 2, 1), np.float32)
    current, target = np.array([[1e8]], np.float32), np.ones((1, 2, 1), np.float32)
    fitted = lag.fit_forecast(history, future, current, target, np.ones(target.shape, bool), [0])
    assert fitted["decoder_FP64"][0, 0] == pytest.approx(-99999999., abs=1e-6)
    assert fitted["decoder_FP32"][0, 0] == np.float32(-1e8)
    meta = fitted["metadata"]
    assert meta["zero_centered_predictor_variance"] == [True]
    assert meta["all_predictor_entries_zero"] == [False] and meta["rank"] == [1]
    assert meta["full_slope_identified"] == [True] and meta["fallback_reason"] == [None]
    assert meta["intercept_fitted"] is False and "no intermediate FP32 subtraction" in meta["target_numeric_contract"]


def test_noncentered_one_observation_can_fit_rank_one_without_forced_zero_and_noobs_unlearned():
    history = np.zeros((1, 512, 2), np.float32)
    future = np.array([[[1., 2.], [0., 0.]]], np.float32)
    current = np.zeros((1, 2), np.float32)
    target = np.array([[[3., np.nan], [np.nan, np.nan]]], np.float32)
    observed = np.array([[[True, False], [False, False]]])
    fitted = lag.fit_forecast(history, future, current, target, observed, [0]); meta = fitted["metadata"]
    np.testing.assert_allclose(fitted["decoder_FP64"][:, 0], [.6, 1.2], atol=1e-15)
    assert meta["one_observed_response"][0] and meta["rank"][0] == 1
    assert meta["fallback_reason"][0] is None and not meta["full_slope_identified"][0]
    assert meta["no_observed_response"][1] and not meta["fit_executed"][1]
    np.testing.assert_array_equal(fitted["decoder_FP64"][:, 1], np.zeros(2))


def test_zero_forecast_design_does_not_fit_an_intercept_to_target_innovation():
    history = np.zeros((1, 512, 1), np.float32); future = np.zeros((1, 3, 1), np.float32)
    fitted = lag.fit_forecast(history, future, np.zeros((1, 1), np.float32),
        np.full((1, 3, 1), 4., np.float32), np.ones((1, 3, 1), bool), [0])
    assert fitted["metadata"]["rank"] == [0]
    assert fitted["metadata"]["direct_residual_MSE"] == [16.]
    assert fitted["metadata"]["fallback_reason"] == ["ZERO_DESIGN_MINIMUM_NORM_ZERO_SLOPE_NO_INTERCEPT"]
    np.testing.assert_array_equal(fitted["bias_FP64"], np.zeros(1))


def test_joint_A_offset_matches_independent_loop_and_direct_real_trace():
    rng = np.random.default_rng(423)
    z, future, current, d = (rng.normal(size=shape).astype(np.float32)
        for shape in ((2, 512, 2), (2, 6, 2), (2, 3), (8, 3)))
    lags = [0, 1, 2, 4]; model = lag.joint_real_map(z, current, d, lags, H=6)
    A, b = np.zeros((18, 12)), np.zeros((2, 6, 3))
    for h in range(1, 7):
        for c in range(3):
            b[:, h-1, c] = current[:, c].astype(np.float64)
            for ell_index, ell in enumerate(lags):
                coeff = d[ell_index*2:(ell_index+1)*2, c].astype(np.float64)
                b[:, h-1, c] -= z[:, -1-ell].astype(np.float64)@coeff
                if h-ell > 0: A[(h-1)*3+c, (h-ell-1)*2:(h-ell)*2] = coeff
                else: b[:, h-1, c] += z[:, -1+h-ell].astype(np.float64)@coeff
    np.testing.assert_array_equal(model["A"], A)
    np.testing.assert_allclose(model["offset"], b, atol=4e-15, rtol=1e-15)
    mapped = (future.astype(np.float64).reshape(2, -1)@A.T).reshape(2, 6, 3)+model["offset"]
    np.testing.assert_allclose(mapped, lag.real_trace(z, future, current, d, lags), atol=1e-14, rtol=2e-15)
    assert model["metadata"]["unknown_coordinates"].startswith("absolute future")


def test_joint_shared_latents_have_positive_minimum_when_pointwise_features_would_fake_zero():
    history, current = np.zeros((1, 512, 1), np.float32), np.zeros((1, 2), np.float32)
    decoder = np.array([[1., 0.], [0., 1.]], np.float32)
    target = np.array([[[0., 0.], [2., 1.], [0., 3.]]], np.float32)
    actual = lag.masked_joint_minimum(history, current, decoder, [0, 1], target, np.ones(target.shape, bool))
    assert actual["minimum_normalized_MSE"] == pytest.approx(1/6, abs=1e-15)
    np.testing.assert_allclose(actual["optimal_future_FP64"][0, :, 0], [.5, 2.5, 0.], atol=1e-15)
    assert actual["metadata"]["numeric_rank_per_origin"] == [3]
    # An invalid per-horizon free feature pair would independently solve identity D.
    fake_pointwise_residual = target.reshape(-1, 2)@np.linalg.pinv(decoder.astype(np.float64))@decoder-target.reshape(-1, 2)
    assert np.mean(fake_pointwise_residual**2) == 0
    assert actual["metadata"]["certificates"][0]["unknown_columns_kept"] == 3


def test_missing_output_horizon_latent_still_controls_later_output():
    history, current = np.zeros((1, 512, 1), np.float32), np.zeros((1, 1), np.float32)
    decoder = np.array([[1.]], np.float32)
    target = np.array([[[np.nan], [7.], [np.nan]]], np.float32)
    observed = np.array([[[False], [True], [False]]])
    result = lag.masked_joint_minimum(history, current, decoder, [1], target, observed)
    assert result["minimum_normalized_MSE"] == 0 and result["optimal_future_FP64"].shape == (1, 3, 1)
    assert result["optimal_future_FP64"][0, 0, 0] == 7.
    assert result["metadata"]["certificates"][0]["unknown_columns_kept"] == 3
    A = lag.joint_real_map(history, current, decoder, [1], H=3)["A"]
    invalid_dropped_columns_design = A[[1]][:, [1]]
    assert np.all(invalid_dropped_columns_design == 0) and (7.-0.)**2 == 49.
    # Filling the unavailable E(Y) at horizon1 with currentC is diagnostic, not an optimum.
    teacher = np.where(observed, target, current[:, None]).astype(np.float32)
    assert lag.real_trace(history, teacher, current, decoder, [1])[0, 1, 0] == 0.


def test_joint_minimum_weights_global_complete_positions_not_window_means():
    history, current = np.zeros((2, 512, 1), np.float32), np.zeros((2, 1), np.float32)
    decoder = np.zeros((1, 1), np.float32)
    target = np.array([[[4.], [np.nan], [np.nan]], [[1.], [1.], [1.]]], np.float32)
    observed = np.isfinite(target)
    result = lag.masked_joint_minimum(history, current, decoder, [0], target, observed)
    assert result["metadata"]["SSE_complete_C"] == 19.
    assert result["metadata"]["C_times_complete_positions_denominator"] == 4
    assert result["minimum_normalized_MSE"] == 19/4 and result["minimum_normalized_MSE"] != (16+1)/2
    assert not result["metadata"]["window_MSEs_reaveraged"]


def test_mask_patterns_share_SVD_and_empty_complete_positions_are_explicit():
    history, current = np.zeros((4, 512, 1), np.float32), np.zeros((4, 2), np.float32)
    d = np.array([[1., 2.]], np.float32); target = np.ones((4, 3, 2), np.float32)
    mask = np.ones(target.shape, bool); mask[:2, 1] = False; mask[3] = False
    result = lag.masked_joint_minimum(history, current, d, [0], target, mask)
    assert result["metadata"]["mask_patterns"] == 3 and result["metadata"]["nonempty_mask_pattern_SVDs"] == 2
    assert result["metadata"]["numeric_rank_per_origin"][-1] == 0
    empty = lag.masked_joint_minimum(history, current, d, [0], target, np.zeros(target.shape, bool))
    assert empty["minimum_normalized_MSE"] is None and empty["metadata"]["status"] == "NO_COMPLETE_C_TARGET_POSITIONS"
    assert empty["metadata"]["nonempty_mask_pattern_SVDs"] == 0


def test_masked_joint_solution_matches_independent_lstsq_and_actual_flat_teacher_real_scores():
    rng = np.random.default_rng(247)
    z, future, current, d = (rng.normal(size=shape).astype(np.float32)
        for shape in ((3, 512, 2), (3, 5, 2), (3, 3), (6, 3)))
    target = rng.normal(size=(3, 5, 3)).astype(np.float32); observed = rng.random(target.shape) > .12
    result = lag.masked_joint_minimum(z, current, d, [0, 1, 2], target, observed)
    mapping = lag.joint_real_map(z, current, d, [0, 1, 2], H=5); SSE = 0.
    for n in range(3):
        rows = np.repeat(observed[n].all(axis=1), 3)
        if not rows.any(): continue
        a, rhs = mapping["A"][rows], (target[n].astype(np.float64)-mapping["offset"][n]).reshape(-1)[rows]
        solved, _, rank, _ = np.linalg.lstsq(a, rhs, rcond=1e-15)
        SSE += float(np.sum((a@solved-rhs)**2))
        np.testing.assert_allclose(result["optimal_future_FP64"][n].reshape(-1), solved, rtol=1e-12, atol=1e-12)
        assert result["metadata"]["numeric_rank_per_origin"][n] == rank
    assert result["minimum_normalized_MSE"] == pytest.approx(SSE/result["metadata"]["C_times_complete_positions_denominator"], abs=1e-13)
    complete = observed.all(axis=2)
    for candidate in (future, np.repeat(z[:, -1:], 5, axis=1),
            np.where(complete[:, :, None], target, current[:, None])[:, :, :2].copy()):
        real = lag.real_trace(z, candidate, current, d, [0, 1, 2])
        saved = lag.predict(z, candidate, current, d, [0, 1, 2])
        real_mse = float(np.mean((real-target.astype(np.float64))[complete]**2))
        saved_mse = float(np.mean((saved.astype(np.float64)-target.astype(np.float64))[complete]**2))
        assert real_mse+1e-12 >= result["minimum_normalized_MSE"]
        assert np.isfinite(abs(saved_mse-real_mse))


def test_metadata_JSON_safe_preserved_inputs_and_predict_has_no_future_labels():
    rng = np.random.default_rng(18)
    z = rng.normal(size=(1, 512, 2)).astype(np.float32); future = rng.normal(size=(1, 3, 2)).astype(np.float32)
    x = rng.normal(size=(1, 512, 3)).astype(np.float32); mask = np.ones(x.shape, bool)
    target = rng.normal(size=(1, 3, 3)).astype(np.float32); observed = np.ones(target.shape, bool)
    before = [a.tobytes() for a in (z, future, x, mask, target, observed)]
    for state in (lag.fit_reconstruction(x, mask, z, [0, 1]), lag.fit_forecast(z, future, x[:, -1], target, observed, [0, 1])):
        json.dumps(state["metadata"], allow_nan=False)
        assert state["decoder_FP32"].shape == (4, 3) and state["lags"].dtype == np.int64
        assert state["metadata"]["singular_relative_cutoff"] == 1e-4
    diagnostic = lag.masked_joint_minimum(z, x[:, -1], state["decoder_FP32"], [0, 1], target, observed)
    json.dumps(diagnostic["metadata"], allow_nan=False)
    assert diagnostic["metadata"]["rcond"] == 1e-15
    assert not diagnostic["metadata"]["strict_untruncated_real_rowspace_minimum_claimed"]
    assert set(inspect.signature(lag.predict).parameters) == {"history", "future", "current", "decoder", "lags"}
    assert before == [a.tobytes() for a in (z, future, x, mask, target, observed)]


@pytest.mark.parametrize("lags", [[], [0, 0], [1, 0], [-1], [512], [False]])
def test_invalid_lag_contracts_are_rejected(lags):
    with pytest.raises(ValueError): lag.lagged_delta_features(np.zeros((1, 512, 1), np.float32), np.zeros((1, 3, 1), np.float32), lags)


def test_reconstruction_zero_design_and_rank_deficiency_remain_explicit():
    z = np.ones((1, 512, 2), np.float32); x = np.full((1, 512, 1), 5., np.float32)
    state = lag.fit_reconstruction(x, np.ones(x.shape, bool), z, [0])
    assert state["metadata"]["rank"] == [0] and state["metadata"]["rank_deficient_slope"] == [True]
    assert state["metadata"]["zero_centered_predictor_variance"] == [True]
    assert state["metadata"]["fallback_reason"] == ["ZERO_DESIGN_MINIMUM_NORM_ZERO_SLOPE_MEAN_INTERCEPT"]
    assert state["bias_FP64"][0] == 5.


def test_unobserved_forecast_NaN_or_arbitrary_values_do_not_change_fit_or_innovation_statistics():
    history = np.zeros((1, 512, 1), np.float32)
    future = np.array([[[1.], [2.], [3.]]], np.float32)
    current = np.array([[4.]], np.float32)
    target = np.array([[[6.], [np.nan], [10.]]], np.float32)
    mask = np.array([[[True], [False], [True]]])
    a = lag.fit_forecast(history, future, current, target, mask, [0])
    target[:, 1] = np.float32(1e30)
    b = lag.fit_forecast(history, future, current, target, mask, [0])
    assert a["decoder_FP64"].tobytes() == b["decoder_FP64"].tobytes()
    assert a["metadata"]["direct_residual_MSE"] == b["metadata"]["direct_residual_MSE"]
    assert a["metadata"]["input_bindings"]["FIT_Y_minus_currentC_FP64"] == b["metadata"]["input_bindings"]["FIT_Y_minus_currentC_FP64"]
    assert a["metadata"]["input_bindings"]["FIT_target_Y"] != b["metadata"]["input_bindings"]["FIT_target_Y"]
    assert a["metadata"]["unobserved_innovation_entries_filled_zero"]
    assert not a["metadata"]["unobserved_target_values_used_in_regression_or_statistics"]
    json.dumps(a["metadata"], allow_nan=False)


def test_registered_numeric_retained_space_is_not_untruncated_real_universal_lower_bound():
    history, current = np.zeros((1, 512, 2), np.float32), np.zeros((1, 2), np.float32)
    decoder = np.diag(np.array([1., 1e-16], np.float32))
    target = np.array([[[0., 1.]]], np.float32); mask = np.ones(target.shape, bool)
    registered = lag.masked_joint_minimum(history, current, decoder, [0], target, mask)
    assert registered["metadata"]["numeric_rank_per_origin"] == [1]
    assert registered["minimum_normalized_MSE"] == .5
    actual_future = np.array([[[0., 1e16]]], np.float32)
    actual_trace = lag.real_trace(history, actual_future, current, decoder, [0])
    real_MSE = float(np.mean((actual_trace-target.astype(np.float64))**2))
    assert real_MSE < 1e-12 < registered["minimum_normalized_MSE"]
    assert not registered["metadata"]["all_real_latent_forecasts_satisfy_registered_bound_claimed"]
    assert not registered["metadata"]["strict_untruncated_real_rowspace_minimum_claimed"]
