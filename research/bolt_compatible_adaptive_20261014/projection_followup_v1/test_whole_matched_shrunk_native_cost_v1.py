"""Synthetic seven-route timer fixtures; no actual model/data/GPU timing."""
from pathlib import Path
import inspect

import numpy as np
import pytest

import whole_matched_shrunk_native_cost_v1 as cost
import whole_selected_shrinkage_cost_v1 as previous


def fixture():
    rng = np.random.default_rng(771)
    physical = rng.normal(size=(8, 512, 3)).astype(np.float32)
    observed = np.ones(physical.shape, bool); observed[:, :2, 1] = False
    physical[~observed] = np.nan
    mean, std = np.array([1., 2., 3.], np.float32), np.array([2., 3., 4.], np.float32)
    origins = np.arange(512, 520, dtype=np.int64)
    history = cost.preprocess(physical, observed, mean, std)
    calls = {name: [] for name in cost.ROUTES}
    predictors, saved, metadata = {}, {}, {}
    for index, name in enumerate(cost.ROUTES):
        def predict(history, mask, index=index, name=name):
            calls[name].append((history.shape, mask.copy()))
            return np.repeat(history[:, -1:, :] + np.float32(index*.01), 48, axis=1)
        predictors[name] = predict
        saved[name] = np.repeat(history[:, -1:, :] + np.float32(index*.01), 48, axis=1)
        metadata[name] = {"synthetic_fixture": True, "native_head_loaded_or_timed": "_NATIVE_s" in name}
    return physical, observed, mean, std, origins, predictors, saved, metadata, calls


def test_full_seven_rotations_balance_each_route_at_each_position():
    schedule = cost.balanced_schedule()
    assert len(schedule) == cost.REPEATS == 7 and cost.WARMUP == 2
    assert schedule == cost.balanced_schedule()
    for order in schedule: assert set(order) == set(cost.ROUTES) and len(order) == 7
    for name in cost.ROUTES: assert sorted(order.index(name) for order in schedule) == list(range(7))
    assert cost.ORDER_SEED == 20261015


def test_exact_routes_and_original_tolerances_are_preserved():
    assert len(cost.ROUTES) == 7 and cost.ROUTES[:3] == ("RAW_BOLT", "PCA_SHRUNK_BOLT", "FIT_TARGET_SHRUNK_BOLT")
    assert sum("_NATIVE_s" in name for name in cost.ROUTES) == 4
    assert (cost.NORM_ATOL, cost.NORM_RTOL, cost.PHYSICAL_ATOL, cost.PHYSICAL_RTOL) == (1e-5, 2e-6, 2e-3, 2e-5)


def test_preprocess_and_actual_timer_bounds_match_previous_source():
    for name in ("fill_vectorized", "preprocess", "parity", "whole_predict"):
        assert inspect.getsource(getattr(cost, name)) == inspect.getsource(getattr(previous, name))
    def bounds(module):
        source = inspect.getsource(module.measure_cost)
        return source[source.index("unix_start_ns ="):source.index("seconds = (perf_end_ns")]
    assert bounds(cost) == bounds(previous)


def test_causal_fill_and_physical_return_are_materialized_FP32():
    physical, observed, mean, std, _, predictors, _, _, _ = fixture()
    normalized, physical_prediction = cost.whole_predict(predictors["RAW_BOLT"], physical, observed, mean, std)
    assert normalized.dtype == physical_prediction.dtype == np.float32
    assert np.isfinite(physical_prediction).all() and physical_prediction.flags.owndata
    filled = cost.fill_vectorized(physical, observed, mean)
    np.testing.assert_array_equal(filled[:, :2, 1], np.full((8, 2), mean[1], np.float32))
    changed = physical.copy(); changed[:, 300:] = 1e5
    np.testing.assert_array_equal(cost.preprocess(physical, observed, mean, std)[:, :300], cost.preprocess(changed, observed, mean, std)[:, :300])


def test_two_synthetic_datasets_record_exact_28_cells_and_196_samples(tmp_path, monkeypatch):
    counter = iter(range(1000, 1000000, 1000))
    monkeypatch.setattr(cost.time, "perf_counter_ns", lambda: next(counter))
    all_rows, all_samples = [], 0
    for dataset in ("synthetic_robin", "synthetic_beijing"):
        physical, observed, mean, std, origins, predictors, saved, metadata, calls = fixture()
        result = cost.measure_cost(dataset, physical, observed, mean, std, origins, predictors, saved,
            metadata, tmp_path / dataset, lambda *_: None, lambda: None,
            lambda: {"resident_rss_bytes": 100, "peak_allocated_vram_bytes": 0, "peak_reserved_vram_bytes": 0})
        assert result["status"] == "COMPLETE" and len(result["rows"]) == 14
        assert result["recorded_timer_samples"] == result["attempted_timer_calls"] == 98
        assert all(len(values) == 20 for values in calls.values())
        for row in result["rows"]:
            assert row["repeats"] == 7 and sorted(row["positions"]) == list(range(7))
            assert row["origins"] == origins[:row["batch"]].tolist()
            assert row["paired_RAW_median_ratio"] == 1.
            assert row["native_head_loaded_or_timed"] == ("_NATIVE_s" in row["route"])
            assert all(sample["seconds"] == 1e-6 for sample in row["samples"])
            assert all(check["pass"] for sample in row["samples"] for check in sample["parity"].values())
            assert "not isolated" in row["memory_scope"]
        all_rows.extend(result["rows"]); all_samples += result["recorded_timer_samples"]
    assert len(all_rows) == 28 and all_samples == 196


@pytest.mark.parametrize("tamper", ["missing_route", "reference_FP64", "stats_FP64", "observed_nonfinite"])
def test_bad_scientific_comparison_inputs_are_rejected(tamper):
    physical, observed, mean, std, origins, predictors, saved, metadata, _ = fixture()
    if tamper == "missing_route": predictors.pop(cost.ROUTES[-1])
    elif tamper == "reference_FP64": saved[cost.ROUTES[0]] = saved[cost.ROUTES[0]].astype(np.float64)
    elif tamper == "stats_FP64": std = std.astype(np.float64)
    else: physical[0, -1, 0] = np.nan
    with pytest.raises(ValueError): cost.validate_inputs(physical, observed, mean, std, origins, predictors, saved, metadata)


def test_failed_first_timed_parity_preserves_actual_partial_sample(tmp_path):
    physical, observed, mean, std, origins, predictors, saved, metadata, _ = fixture()
    name = cost.balanced_schedule()[0][0]; ordinary = predictors[name]; count = [0]
    def fail_fourth(history, mask):
        count[0] += 1
        prediction = ordinary(history, mask)
        return prediction + np.float32(1) if count[0] == 4 else prediction
    predictors[name] = fail_fourth
    output = tmp_path / "partial"
    with pytest.raises(ValueError, match="TIMED_OUTPUT_PARITY_FAILED"):
        cost.measure_cost("synthetic", physical, observed, mean, std, origins, predictors, saved, metadata,
            output, lambda *_: None, lambda: None, lambda: {"resident_rss_bytes": 100})
    import json
    receipt = json.loads((output / "RESULT.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "FAILED_PRESERVED" and receipt["recorded_timer_samples"] == 1
    assert receipt["rows"] == [] and (output / "B1" / (name + "_r0.json")).is_file()
    assert (output / "B1" / (name + "_r0.npz")).is_file()


def test_existing_timer_directory_and_artifacts_are_not_overwritten(tmp_path):
    physical, observed, mean, std, origins, predictors, saved, metadata, _ = fixture()
    old = tmp_path / "old"; old.mkdir(); marker = old / "keep.txt"; marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        cost.measure_cost("synthetic", physical, observed, mean, std, origins, predictors, saved, metadata,
            old, lambda *_: None, lambda: None, lambda: {})
    assert marker.read_text(encoding="utf-8") == "preserve"
