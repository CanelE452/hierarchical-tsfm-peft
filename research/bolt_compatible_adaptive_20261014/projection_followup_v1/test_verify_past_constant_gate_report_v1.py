import copy
import csv
import json

import pytest

import verify_past_constant_gate_report_v1 as verifier


def timers():
    seconds = [value / 1e9 for value in (100, 200, 300, 400, 500, 600)]
    row = {"samples": [{"perf_start_ns": 1000, "perf_end_ns": 1000 + value} for value in (100, 200, 300, 400, 500, 600)],
           "seconds": seconds, "paired_RAW_seconds": seconds, "paired_RAW_ratios": [1.0] * 6,
           "positions": list(range(6)), "repeats": 6, "batch": 8,
           "median_seconds": (seconds[2] + seconds[3]) / 2,
           "median_seconds_per_window": ((seconds[2] + seconds[3]) / 2) / 8,
           "paired_RAW_median_ratio": 1.0}
    return row


def test_timer_recomputation_uses_timestamp_differences():
    row = timers()
    assert verifier.timer_statistics(row, row) == (1.0, 1.0)


@pytest.mark.parametrize("field,value,match", [
    ("median_seconds", 1.0, "RECORDED_TIMING_AGGREGATES"),
    ("paired_RAW_ratios", [2.0] * 6, "RECORDED_TIMING_AGGREGATES"),
    ("positions", [0] * 6, "LATIN_SIX_POSITION_COVERAGE"),
    ("seconds", [1.0] * 6, "TIMER_TIMESTAMP_BINDING"),
])
def test_wrong_timer_metadata_is_rejected(field, value, match):
    raw = timers(); altered = copy.deepcopy(raw); altered[field] = value
    with pytest.raises(ValueError, match=match): verifier.timer_statistics(altered, raw)


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader()
        for row in rows: writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def sample_row():
    return {"dataset": "synthetic", "split": "tune", "method": "CONTROL", "variant": "GATED",
            "mask": "all_original_observed", "metric_scale": "normalized", "mse": .25,
            "observed_counts": [2, 3], "scope": {"future_labels_used": False}}


def test_nested_csv_leaves_preserve_boolean_and_counts(tmp_path, monkeypatch):
    row = sample_row(); path = tmp_path / "scores.csv"; write_rows(path, [row])
    monkeypatch.setitem(verifier.COUNTS, "SCORES", 1)
    assert verifier.check_csv(path, "SCORES", [row]) == 10


def test_csv_numeric_corruption_is_rejected(tmp_path, monkeypatch):
    row = sample_row(); altered = copy.deepcopy(row); altered["mse"] += .01
    path = tmp_path / "scores.csv"; write_rows(path, [altered]); monkeypatch.setitem(verifier.COUNTS, "SCORES", 1)
    with pytest.raises(ValueError, match="CSV_SOURCE_VALUE_MISMATCH"): verifier.check_csv(path, "SCORES", [row])


def test_csv_duplicate_scientific_condition_is_rejected(tmp_path, monkeypatch):
    row = sample_row(); path = tmp_path / "scores.csv"; write_rows(path, [row, row])
    monkeypatch.setitem(verifier.COUNTS, "SCORES", 1)
    with pytest.raises(ValueError, match="CSV_COUNT_OR_DUPLICATE"): verifier.check_csv(path, "SCORES", [row])
