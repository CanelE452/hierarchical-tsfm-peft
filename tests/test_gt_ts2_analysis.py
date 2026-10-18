import importlib
import json

import numpy as np
import pytest


def analysis_module():
    try:
        return importlib.import_module("research.gt_ts2_fastcheck_20261018_v1.analysis")
    except ModuleNotFoundError:
        pytest.fail("The independent FP64 fastcheck analysis implementation is missing")


def prediction_bundle(value=1.):
    return {
        "prediction": np.full((2, 2, 2), value, dtype=np.float64),
        "target": np.zeros((2, 2, 2), dtype=np.float64),
        "mask": np.ones((2, 2, 2), dtype=bool),
        "origins": np.array([100, 124], dtype=np.int64),
        "mean": np.array([10., 20.]),
        "std": np.array([2., 4.]),
    }


def test_score_aggregates_channel_sufficient_statistics_before_macro_average():
    module = analysis_module()
    prediction = np.array([[[1., 3.], [1., 3.]], [[5., 9.], [5., 9.]]])
    mask = np.array([[[True, True], [True, False]], [[True, True], [False, False]]])
    target = np.where(mask, 0., np.nan)
    result = module.score(prediction, target, mask)
    assert result["channel_count"] == [3, 2]
    assert result["channel_sse"] == [27., 90.]
    assert result["channel_sae"] == [7., 12.]
    assert result["channel_mse"] == [9., 45.]
    assert result["mse"] == 27.
    assert result["mae"] == pytest.approx(25 / 6)
    assert result["mse"] != 117 / 5
    assert result["mse"] != (5 + 53) / 2
    json.dumps(result, allow_nan=False)


def test_score_casts_to_fp64_before_subtraction_and_square():
    module = analysis_module()
    prediction = np.array([[[16777218.]]], np.float32)
    target = np.array([[[-16777216.]]], np.float32)
    result = module.score(prediction, target, np.ones((1, 1, 1), bool))
    assert result["mse"] == 33554434. ** 2
    assert result["mae"] == 33554434.


def test_score_rejects_overflow_in_the_final_channel_average():
    with pytest.raises(ValueError):
        analysis_module().score(np.full((1, 1, 2), 1e154), np.zeros((1, 1, 2)),
                                np.ones((1, 1, 2), bool))


@pytest.mark.parametrize("change", ["shape", "rank", "empty", "mask_dtype", "prediction_nan",
                                    "observed_target_inf", "empty_channel", "overflow"])
def test_score_rejects_invalid_or_undefined_metrics(change):
    bundle = prediction_bundle()
    prediction, target, mask = bundle["prediction"], bundle["target"], bundle["mask"]
    if change == "shape":
        target = target[:, :1]
    elif change == "rank":
        prediction, target, mask = prediction[0], target[0], mask[0]
    elif change == "empty":
        prediction, target, mask = prediction[:0], target[:0], mask[:0]
    elif change == "mask_dtype":
        mask = mask.astype(np.uint8)
    elif change == "prediction_nan":
        prediction[0, 0, 0] = np.nan
    elif change == "observed_target_inf":
        target[0, 0, 0] = np.inf
    elif change == "empty_channel":
        mask[:, :, 0] = False
    elif change == "overflow":
        prediction[:] = 1e308
    with pytest.raises(ValueError):
        analysis_module().score(prediction, target, mask)


def test_statistics_total_reconstructs_from_disjoint_horizon_bands():
    module = analysis_module()
    prediction = np.tile(np.repeat([[1., 2.], [3., 4.], [5., 6.]], 16, axis=0)[None], (2, 1, 1))
    target, mask = np.zeros_like(prediction), np.ones_like(prediction, dtype=bool)
    rows, origins = module.statistics(prediction, target, mask, np.array([100, 124]), "candidate")
    assert len(rows) == 8 and len(origins) == 2
    for channel, total_sse in ((0, 1120.), (1, 1792.)):
        channel_rows = [row for row in rows if row["channel"] == channel]
        total = next(row for row in channel_rows if row["horizon_band"] == "TOTAL")
        bands = [row for row in channel_rows if row["horizon_band"] != "TOTAL"]
        assert [row["horizon_band"] for row in bands] == ["1-16", "17-32", "33-48"]
        assert total["sse"] == total_sse == sum(row["sse"] for row in bands)
        assert total["sae"] == sum(row["sae"] for row in bands)
        assert total["count"] == 96 == sum(row["count"] for row in bands)
        assert total["mse"] == total_sse / 96
    assert [row["origin"] for row in origins] == [100, 124]
    assert all(row["run_id"] == "candidate" and row["count"] == 96 for row in origins)
    assert all(row["mse"] == pytest.approx(91 / 6) for row in origins)


def test_statistics_rejects_empty_band_channel_and_origin_cohort_shape():
    module = analysis_module()
    prediction = np.ones((2, 48, 2))
    mask = np.ones_like(prediction, bool)
    mask[:, :16, 1] = False
    with pytest.raises(ValueError):
        module.statistics(prediction, np.zeros_like(prediction), mask, [100, 124], "run")
    with pytest.raises(ValueError):
        module.statistics(prediction, np.zeros_like(prediction), np.ones_like(mask), [100], "run")


def test_paired_reports_candidate_minus_reference_and_negative_improvement():
    summary, rows = analysis_module().paired(prediction_bundle(.5), prediction_bundle())
    assert summary["candidate_mse"] == .25
    assert summary["reference_mse"] == 1.
    assert summary["mse_delta"] == -.75
    assert summary["mse_relative_percent"] == -75.
    assert summary["mae_relative_percent"] == -50.
    assert summary["origin_mse_improved_fraction"] == 1.
    assert [row["origin"] for row in rows] == [100, 124]
    assert all(row["mse_delta"] == -.75 and row["mae_delta"] == -.5 for row in rows)
    assert summary["origins_are_independent_samples"] is False
    json.dumps((summary, rows), allow_nan=False)


def test_paired_relative_change_uses_unrounded_original_precision():
    summary, _ = analysis_module().paired(prediction_bundle(1.00003), prediction_bundle(1.00004))
    expected = ((1.00003 ** 2) - (1.00004 ** 2)) / (1.00004 ** 2) * 100
    assert summary["mse_relative_percent"] == pytest.approx(expected, abs=1e-13)
    assert summary["mse_relative_percent"] < 0


@pytest.mark.parametrize("field", ["target", "mask", "origins", "mean", "std", "dtype", "prediction_shape"])
def test_paired_refuses_different_cohorts_or_scaling_coordinates(field):
    reference, candidate = prediction_bundle(), prediction_bundle(.5)
    if field == "mask":
        candidate[field][0, 0, 0] = False
    elif field == "dtype":
        candidate["target"] = candidate["target"].astype(np.float32)
    elif field == "prediction_shape":
        candidate["prediction"] = candidate["prediction"][:, :1]
    else:
        candidate[field].flat[0] += 1
    with pytest.raises(ValueError):
        analysis_module().paired(candidate, reference)


def test_paired_npz_inputs_have_identical_results_to_array_inputs(tmp_path):
    module = analysis_module()
    candidate, reference = prediction_bundle(.5), prediction_bundle()
    candidate_path, reference_path = tmp_path / "candidate.npz", tmp_path / "reference.npz"
    np.savez(candidate_path, **candidate)
    np.savez(reference_path, **reference)
    assert module.paired(candidate_path, reference_path) == module.paired(candidate, reference)


def test_paired_rejects_zero_reference_relative_denominator():
    with pytest.raises(ValueError):
        analysis_module().paired(prediction_bundle(.5), prediction_bundle(0.))


def test_paired_rejects_nonfinite_relative_percent_even_with_finite_scores():
    with pytest.raises(ValueError):
        analysis_module().paired(prediction_bundle(1e10), prediction_bundle(1e-150))


def test_schedule_visits_all_197_origins_and_keeps_last_small_batch():
    schedule = analysis_module().make_schedule(92601, 197, 50)
    assert len(schedule) == 50
    assert [len(batch) for batch in schedule[:25]] == [8] * 24 + [5]
    assert [len(batch) for batch in schedule[25:]] == [8] * 24 + [5]
    np.testing.assert_array_equal(np.sort(np.concatenate(schedule[:25])), np.arange(197))
    np.testing.assert_array_equal(np.sort(np.concatenate(schedule[25:])), np.arange(197))
    assert not np.array_equal(np.concatenate(schedule[:25]), np.concatenate(schedule[25:]))


def test_schedule_is_reproducible_for_all_arms_seed_specific_and_has_no_512_cap():
    module = analysis_module()
    first, second = module.make_schedule(92601, 600, 75), module.make_schedule(92601, 600, 75)
    other = module.make_schedule(92602, 600, 75)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert any(not np.array_equal(a, b) for a, b in zip(first, other))
    np.testing.assert_array_equal(np.sort(np.concatenate(first)), np.arange(600))
    assert module.make_schedule(92601, 197, 0) == []


@pytest.mark.parametrize("pool_size,updates,batch", [(0, 3, 8), (197, -1, 8), (197, 3, 0),
                                                      (197.5, 3, 8), (197, 3.5, 8)])
def test_schedule_rejects_invalid_pool_or_update_requests(pool_size, updates, batch):
    with pytest.raises(ValueError):
        analysis_module().make_schedule(92601, pool_size, updates, batch)
