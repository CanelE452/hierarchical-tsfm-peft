"""Synthetic nine-route schedule, whole-boundary and failure preservation."""
from pathlib import Path
import ast
import inspect
import statistics

import numpy as np
import pytest

import whole_linear_e_cost_v1 as cost


def fixture():
    rng = np.random.default_rng(9)
    physical = rng.normal(size=(8, 512, 2)).astype(np.float32)
    observed = np.ones_like(physical, bool)
    observed[:, :3, 0] = False; physical[~observed] = np.nan
    mean, std = np.array([2., 3.], np.float32), np.array([4., 5.], np.float32)
    normalized = cost.preprocess(physical, observed, mean, std)
    pred = np.repeat(normalized[:, -1:], 48, 1)
    callbacks = {route: lambda x, mask: np.repeat(x[:, -1:], 48, 1) for route in cost.ROUTES}
    return [physical, observed, mean, std, np.arange(8, dtype=np.int64), callbacks,
            {route: pred.copy() for route in cost.ROUTES}, {route: {"synthetic": True} for route in cost.ROUTES}]


def run(folder, values=None):
    return cost.measure_cost("synthetic", *(fixture() if values is None else values), folder,
        lambda: None, lambda: None, lambda: {"shared": True, "RSS_bytes": 12})


def test_nine_complete_circular_rotations_are_position_balanced():
    assert cost.ORDER_SEED == 20261015 and cost.WARMUP == 2 and cost.REPEATS == 9
    schedule = cost.balanced_schedule()
    assert schedule == cost.balanced_schedule(20261015) and len(schedule) == 9
    assert all(set(row) == set(cost.ROUTES) for row in schedule)
    assert all(sorted(row.index(route) for row in schedule) == list(range(9)) for route in cost.ROUTES)
    assert len(cost.ROUTES) == 9 and not any("NATIVE" in route for route in cost.ROUTES)


def test_two_toy_datasets_save_exact_36_cells_324_samples(tmp_path):
    runs = [run(tmp_path / name) for name in ("one", "two")]
    assert sum(value["completed_cost_cells"] for value in runs) == 36
    assert sum(value["recorded_timer_samples"] for value in runs) == 324
    for result in runs:
        assert result["status"] == "COMPLETE"
        for row in result["rows"]:
            assert len(row["samples"]) == row["repeats"] == 9
            assert row["median_seconds"] == statistics.median(row["seconds"])
            assert row["paired_RAW_median_ratio"] == statistics.median(row["paired_RAW_ratios"])
            for item in row["samples"]:
                assert item["seconds"] == (item["perf_end_ns"] - item["perf_start_ns"]) / 1e9
                assert all(value["pass"] for value in item["parity"].values())
                assert Path(item["sample_artifact"]["path"]).exists()


def test_one_parity_failure_keeps_other_eight_routes_and_slots(tmp_path):
    values = fixture(); bad = cost.ROUTES[1]; values[6][bad] += np.float32(.1)
    result = run(tmp_path / "partial", values)
    assert result["status"] == "PARTIAL_COMPLETED" and len(result["rows"]) == 18
    assert result["completed_cost_cells"] == 16 and result["recorded_timer_samples"] == 144
    assert {row["route"] for row in result["failed_paths"]} == {bad}
    assert all(row["declared_route_deleted"] is False for row in result["rows"])


def test_failure_after_two_timed_calls_preserves_partial_samples(tmp_path):
    values, counts = fixture(), {1: 0, 8: 0}
    def fail(x, mask):
        counts[len(x)] += 1
        if counts[len(x)] == 6: raise RuntimeError("two timed samples then failure")
        return np.repeat(x[:, -1:], 48, 1)
    values[5][cost.ROUTES[-1]] = fail
    result = run(tmp_path / "partial", values)
    assert result["completed_cost_cells"] == 16 and result["recorded_timer_samples"] == 148
    assert [row["repeats"] for row in result["rows"] if row["route"] == cost.ROUTES[-1]] == [2, 2]


def test_exact_legacy_preprocessing_and_ns_critical_region():
    import whole_causal_lag_cost_v1 as old
    a, b = (ast.parse(Path(value.__file__).read_text(encoding="utf-8")) for value in (old, cost))
    for name in ("fill_vectorized", "preprocess", "parity", "whole_predict"):
        functions = [next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name) for tree in (a, b)]
        assert ast.dump(functions[0], include_attributes=False) == ast.dump(functions[1], include_attributes=False)
    def region(module):
        return inspect.getsource(module.measure_cost).split('synchronize(); result["attempted_timer_calls"] += 1', 1)[1].split("seconds =", 1)[0]
    assert region(old) == region(cost)


def test_forward_fill_and_materialized_physical_return():
    physical = np.array([[[np.nan, 3.], [7., np.nan], [np.nan, 8.]]], np.float32)
    observed = np.isfinite(physical)
    mean, std = np.array([2., 4.], np.float32), np.ones(2, np.float32)
    filled = cost.fill_vectorized(physical, observed, mean)
    np.testing.assert_array_equal(filled, [[[2., 3.], [7., 3.], [7., 8.]]])
    actual, returned = cost.whole_predict(lambda x, mask: np.repeat(x[:, -1:], 48, 1), physical, observed, mean, std)
    assert returned.dtype == np.float32 and returned.flags.owndata and not np.shares_memory(actual, returned)
    np.testing.assert_array_equal(returned, actual * std + mean)


def test_registered_tolerances_and_exclusive_output(tmp_path):
    assert (cost.NORM_ATOL, cost.NORM_RTOL, cost.PHYSICAL_ATOL, cost.PHYSICAL_RTOL) == (1e-5, 2e-6, 2e-3, 2e-5)
    one = np.ones((1, 48, 2), np.float32)
    assert not cost.parity(one + .001, one)["pass"]
    run(tmp_path / "existing")
    with pytest.raises(FileExistsError): run(tmp_path / "existing")
