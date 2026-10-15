"""Independent direct-LS synthetic checks; no real selection/data/F/head execution."""
import inspect
import itertools
import json

import numpy as np
import pytest

import raw_subset_projection_v2 as subset


def history_fixture():
    rng = np.random.default_rng(817)
    history = rng.normal(size=(2, 512, 4)).astype(np.float32)
    history[:, :, 3] = np.float32(.4)*history[:, :, 0] + np.float32(.8)*history[:, :, 1]
    observed = rng.random(history.shape) > .2
    observed[:, :51, 1] = False
    observed[0, :141, 2] = False
    return history, observed


def direct_scores(history, observed, indices):
    x, mask = history.reshape(-1, history.shape[-1]).astype(np.float64), observed.reshape(-1, observed.shape[-1])
    mse, ranks = [], []
    for channel in range(x.shape[1]):
        valid = mask[:, channel]
        if not valid.any(): mse.append(None); ranks.append(None); continue
        predictors, response = x[valid][:, indices], x[valid, channel]
        centered, target = predictors-predictors.mean(axis=0), response-response.mean()
        slope, _, rank, _ = np.linalg.lstsq(centered, target, rcond=1e-4)
        mse.append(float(np.mean((centered@slope-target)**2))); ranks.append(int(rank))
    return mse, ranks, float(np.mean([value for value in mse if value is not None]))


def test_response_specific_direct_scores_equal_independent_centered_lstsq_for_all_small_subsets():
    history, observed = history_fixture()
    stats = subset.precompute_observed_stats(history, observed)
    for size in (1, 2, 3):
        for indices in itertools.combinations(range(4), size):
            expected, ranks, macro = direct_scores(history, observed, list(indices))
            actual = subset.score_subset(stats, list(indices))
            np.testing.assert_allclose(actual["per_channel_MSE"], expected, rtol=1e-9, atol=2e-12)
            np.testing.assert_allclose(actual["macro_normalized_MSE"], macro, rtol=1e-9, atol=2e-12)
            assert actual["retained_centered_design_rank"] == ranks
    assert not stats.features.flags.writeable and not stats.observed.flags.writeable
    assert stats.metadata["response_specific_observed_mask_for_centered_statistics"]


@pytest.mark.parametrize("epsilon,rank", [(5e-5, 1), (2e-4, 2)])
def test_direct_singular_cutoff_preserves_registered_rank_on_both_sides(epsilon, rank):
    s = np.tile(np.array([-1., 1.], np.float32), 256)
    t = np.tile(np.array([-1., -1., 1., 1.], np.float32), 128)
    history = np.stack((s, np.float32(epsilon)*t, .3*s+.7*np.float32(epsilon)*t), axis=1)[None]
    observed = np.ones(history.shape, bool)
    actual = subset.score_subset(subset.precompute_observed_stats(history, observed), [0, 1])
    expected, ranks, _ = direct_scores(history, observed, [0, 1])
    assert actual["retained_centered_design_rank"] == ranks == [rank]*3
    assert actual["Gram_eigen_relative_cutoff_equivalent_in_real_arithmetic_only"] == 1e-8
    assert actual["Gram_cross_variance_expansion_used_for_scores"] is False
    np.testing.assert_allclose(actual["per_channel_MSE"], expected, atol=2e-12, rtol=1e-6)


def test_greedy_matches_independent_candidate_scores_and_preserves_all_trace():
    history, observed = history_fixture()
    result = subset.greedy_raw_subset(history, observed, 2)
    order = []
    for stage in result["metadata"]["stages"]:
        candidates = {index: direct_scores(history, observed, sorted([*order, index]))[2] for index in range(4) if index not in order}
        winner = min(candidates, key=lambda index: (candidates[index], index))
        assert stage["selected_original_index"] == winner
        assert len(stage["candidates"]) == 4-len(order)
        for row in stage["candidates"]:
            np.testing.assert_allclose(row["macro_normalized_MSE"], candidates[row["candidate_original_index"]], atol=2e-12, rtol=1e-9)
        order.append(winner)
    assert result["selection_order"] == order and result["selected_indices"] == sorted(order)
    assert not result["metadata"]["optimal_subset_guaranteed"]


def test_exact_tie_chooses_lowest_original_index_and_tracks_greedy_order():
    signal = np.tile(np.array([-1., 1.], np.float32), 256)
    history = np.broadcast_to(signal[None, :, None], (1, 512, 4)).copy()
    result = subset.greedy_raw_subset(history, np.ones(history.shape, bool), 3)
    assert result["selection_order"] == result["selected_indices"] == [0, 1, 2]
    assert all(len({row["macro_normalized_MSE"] for row in stage["candidates"]}) == 1 for stage in result["metadata"]["stages"])


def test_inactive_response_excluded_count1_included_and_rank_deficiency_retained():
    history, observed = history_fixture()
    observed[:, :, 2] = False
    observed[:, :, 3] = False; observed[0, 0, 3] = True
    actual = subset.score_subset(subset.precompute_observed_stats(history, observed), [0])
    assert actual["active_channels"] == 3 and actual["per_channel_MSE"][2] is None
    assert actual["per_channel_MSE"][3] == 0 and actual["retained_centered_design_rank"][3] == 0
    expected, _, macro = direct_scores(history, observed, [0])
    np.testing.assert_allclose(actual["macro_normalized_MSE"], macro, rtol=1e-9, atol=2e-12)


def test_onehot_encoder_is_center0_and_direct_slice_preserves_signed_zero_bytes():
    history = np.zeros((1, 512, 4), np.float32); history[0, 0, 3] = np.float32(-0.)
    selected = subset.encode_subset(history, [1, 3])
    center, encoder = subset.subset_encoder(4, [1, 3])
    assert center.dtype == encoder.dtype == selected.dtype == np.float32
    np.testing.assert_array_equal(center, np.zeros(4, np.float32))
    np.testing.assert_array_equal(encoder[[1, 3]], np.eye(2, dtype=np.float32))
    assert selected.tobytes() == np.ascontiguousarray(history[..., [1, 3]]).tobytes()
    dense = (history-center)@encoder
    np.testing.assert_array_equal(selected, dense)
    assert selected.tobytes() != dense.tobytes()


def test_global_decoder_equals_independent_per_response_observed_affine_lstsq():
    history, observed = history_fixture()
    z = subset.encode_subset(history, [0, 1])
    fitted = subset.fit_global_observed_decoder(history, observed, z)
    flat_z, flat_x, flat_mask = z.reshape(-1, 2).astype(np.float64), history.reshape(-1, 4).astype(np.float64), observed.reshape(-1, 4)
    for channel in range(4):
        features, target = flat_z[flat_mask[:, channel]], flat_x[flat_mask[:, channel], channel]
        zm, ym = features.mean(axis=0), target.mean()
        slope, _, rank, _ = np.linalg.lstsq(features-zm, target-ym, rcond=1e-4)
        np.testing.assert_allclose(fitted["decoder_FP64"][:, channel], slope, rtol=0, atol=0)
        np.testing.assert_allclose(fitted["bias_FP64"][channel], ym-zm@slope, rtol=0, atol=0)
        assert fitted["metadata"]["retained_centered_design_rank"][channel] == rank
    assert fitted["decoder_FP32"].dtype == fitted["bias_FP32"].dtype == np.float32


def test_constant_response_fullrank_is_normal_identified_zero_slope_fit():
    history, observed = history_fixture()
    history[:, :, 3] = np.float32(7)
    fitted = subset.fit_global_observed_decoder(history, observed, subset.encode_subset(history, [0, 1]))
    meta = fitted["metadata"]
    assert meta["constant_observed_response"][3] and meta["fallback_reason"][3] is None
    assert meta["learned_slope"][3] and meta["slope_identified_in_retained_full_K_space"][3]
    assert not meta["rank_deficient_slope"][3]
    np.testing.assert_array_equal(fitted["decoder_FP64"][:, 3], np.zeros(2))
    assert fitted["bias_FP64"][3] == 7


def test_no_observations_one_observation_zero_predictors_and_rank_deficient_are_distinct():
    history, observed = history_fixture()
    observed[:, :, 2] = False; observed[:, :, 3] = False; observed[0, 0, 3] = True
    z = np.ones((2, 512, 2), np.float32)
    fitted = subset.fit_global_observed_decoder(history, observed, z); meta = fitted["metadata"]
    assert meta["no_observed_responses"][2] and meta["rank_deficient_slope"][2] is None
    assert not meta["centered_lstsq_fit_executed"][2] and fitted["bias_FP64"][2] == 0
    assert meta["one_observed_response"][3] and meta["rank_deficient_slope"][3]
    assert "ONE_OBSERVED" in meta["fallback_reason"][3]
    assert meta["zero_centered_predictor_variance"][0] and "ZERO_CENTERED_PREDICTOR" in meta["fallback_reason"][0]
    np.testing.assert_array_equal(fitted["decoder_FP64"], np.zeros((2, 4)))
    assert fitted["bias_FP64"][3] == history[0, 0, 3]
    missing = subset.fit_global_observed_decoder(history, np.zeros(history.shape, bool), z)
    assert missing["metadata"]["no_observed_responses"] == [True]*4
    with pytest.raises(ValueError, match="NO_ACTIVE"): subset.greedy_raw_subset(history, np.zeros(history.shape, bool), 2)


def test_rank_deficient_nonconstant_slopes_are_minimum_norm_but_not_full_K_identified():
    history, observed = history_fixture()
    z = np.repeat(history[:, :, :1], 2, axis=2)
    fitted = subset.fit_global_observed_decoder(history, observed, z); meta = fitted["metadata"]
    assert meta["retained_centered_design_rank"] == [1]*4
    assert meta["rank_deficient_slope"] == [True]*4 and not any(meta["slope_identified_in_retained_full_K_space"])
    np.testing.assert_allclose(fitted["decoder_FP64"][0], fitted["decoder_FP64"][1], atol=1e-12)


def test_affine_bias_cancels_in_current_anchor_innovation_math():
    history, observed = history_fixture()
    z = subset.encode_subset(history, [0, 1])
    fitted = subset.fit_global_observed_decoder(history, observed, z)
    last = z[:, -1:]; future = np.repeat(last, 48, axis=1).astype(np.float64)+.25
    current = history[:, -1:]
    expression = current+(future-last)@fitted["decoder_FP64"]
    decode_difference = current+(future@fitted["decoder_FP64"]+fitted["bias_FP64"])-(last@fitted["decoder_FP64"]+fitted["bias_FP64"])
    np.testing.assert_allclose(expression, decode_difference, rtol=0, atol=2e-15)
    np.testing.assert_array_equal(current+(last-last)@fitted["decoder_FP32"], current)


def test_APIs_have_no_future_TUNE_DEV_or_F_scores_and_preserve_inputs():
    history, observed = history_fixture(); before, mask_before = history.tobytes(), observed.tobytes()
    result = subset.greedy_raw_subset(history, observed, 2)
    fitted = subset.fit_global_observed_decoder(history, observed, subset.encode_subset(history, result["selected_indices"]))
    assert history.tobytes() == before and observed.tobytes() == mask_before
    assert set(inspect.signature(subset.greedy_raw_subset).parameters) == {"history", "observed", "K"}
    assert set(inspect.signature(subset.fit_global_observed_decoder).parameters) == {"history", "observed", "latent_history"}
    assert not result["metadata"]["future_TUNE_DEV_or_F_scores_used"] and not fitted["metadata"]["future_TUNE_DEV_or_F_scores_used"]
    assert not result["metadata"]["direct_slice_dense_onehot_byte_identity_claimed"]


def test_nondegenerate_global_and_greedy_metadata_are_strict_JSON_serializable():
    history, observed = history_fixture()
    selection = subset.greedy_raw_subset(history, observed, 2)
    fitted = subset.fit_global_observed_decoder(history, observed, subset.encode_subset(history, selection["selected_indices"]))
    assert all(type(value) is bool for value in fitted["metadata"]["learned_slope"])
    assert any(fitted["metadata"]["learned_slope"])
    for metadata in (selection["metadata"], fitted["metadata"]):
        assert json.loads(json.dumps(metadata, allow_nan=False)) == metadata


@pytest.mark.parametrize("change", ["history_FP64", "observed_int", "history_nonfinite", "latent_FP64", "K0"])
def test_invalid_past_contracts_are_rejected(change):
    history, observed = history_fixture(); z = history[:, :, :2].copy()
    if change == "history_FP64": history = history.astype(np.float64)
    elif change == "observed_int": observed = observed.astype(np.int64)
    elif change == "history_nonfinite": history[0, 0, 0] = np.nan
    elif change == "latent_FP64": z = z.astype(np.float64)
    if change == "K0":
        with pytest.raises(ValueError): subset.greedy_raw_subset(history, observed, 0)
    else:
        with pytest.raises(ValueError): subset.fit_global_observed_decoder(history, observed, z)


def test_long_self_response_uses_explicit_nonnegative_residual_and_never_clips():
    rng = np.random.default_rng(8942)
    history = rng.normal(size=(64, 512, 3)).astype(np.float32)
    observed = rng.random(history.shape) > .13
    stats = subset.precompute_observed_stats(history, observed)
    for index in range(3):
        actual = subset.score_subset(stats, [index])
        expected, ranks, macro = direct_scores(history, observed, [index])
        assert actual["per_channel_MSE"][index] >= 0
        assert actual["per_channel_MSE"][index] < 1e-20
        assert actual["negative_roundoff_MSE_values_clipped_to_zero"] == 0
        assert actual["Gram_cross_variance_expansion_used_for_scores"] is False
        np.testing.assert_array_equal(actual["per_channel_MSE"], expected)
        assert actual["retained_centered_design_rank"] == ranks
        assert actual["macro_normalized_MSE"] == macro


def test_global_decoder_is_exact_immutable_v1_function_and_score_has_no_Gram_expansion():
    import raw_subset_projection_v1 as original
    assert subset.fit_global_observed_decoder is original.fit_global_observed_decoder
    source = inspect.getsource(subset.score_subset)
    assert "np.linalg.lstsq" in source and "np.mean(residual*residual)" in source
    assert "stats.gram" not in source and "stats.cross" not in source
    assert "max(0., error)" not in source


def test_direct_score_selection_preserves_every_candidate_exact_small_fixture():
    history, observed = history_fixture()
    selection = subset.greedy_raw_subset(history, observed, 3)
    candidates = 0
    for stage in selection["metadata"]["stages"]:
        for row in stage["candidates"]:
            expected, rank, macro = direct_scores(history, observed, row["indices"])
            assert row["per_channel_MSE"] == expected
            assert row["macro_normalized_MSE"] == macro
            assert row["retained_centered_design_rank"] == rank
            candidates += 1
    assert candidates == 4+3+2
    assert selection["metadata"]["global_decoder_fitter_unchanged_v1_function"]
    assert "ordering can differ" in selection["metadata"]["numeric_repair_scope"]
