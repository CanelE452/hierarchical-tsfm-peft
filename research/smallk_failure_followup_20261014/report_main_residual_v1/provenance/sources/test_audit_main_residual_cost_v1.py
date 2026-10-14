"""Synthetic contract tests for audit_main_residual_cost_v1.

No real producer RESULT, NPZ, checkpoint, CUDA, or timing data is opened.
"""
from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import audit_main_residual_cost_v1 as audit


def test_schedule_is_latin6_with_each_route_once_per_block():
    schedule = audit.expected_schedule(np)
    assert len(schedule) == audit.REPEATS == 6
    assert all(sorted(row) == sorted(audit.ROUTES) for row in schedule)
    for route in audit.ROUTES:
        assert sorted(row.index(route) for row in schedule) == list(range(len(audit.ROUTES)))


def test_close_arrays_uses_scale_specific_tolerances_and_rejects_nan():
    ref = np.array([0.0, 1000.0], dtype=np.float32)
    assert audit.close_arrays(ref + np.array([1e-5, 1e-3], dtype=np.float32), ref, "normalized")
    assert not audit.close_arrays(ref + np.array([3e-5, 0.0], dtype=np.float32), ref, "normalized")
    assert audit.close_arrays(ref + np.array([1e-3, 1e-2], dtype=np.float32), ref, "physical")
    assert not audit.close_arrays(np.array([np.nan, 1.0], dtype=np.float32), ref, "physical")


def test_masked_score_is_channel_macro_and_physical_scaling():
    prediction = np.array([[[2.0, 3.0, np.nan], [2.0, np.nan, np.nan]]], dtype=np.float32)
    target = np.zeros((1, 2, 3), dtype=np.float32)
    observed = np.array([[[True, True, False], [True, False, False]]], dtype=np.bool_)
    score = audit.masked_score(prediction, target, observed)
    assert score["mse"] == pytest.approx(6.5)
    assert score["mae"] == pytest.approx(2.5)
    assert score["active_channels"] == 2
    assert score["per_channel_counts"] == [2, 1, 0]
    physical = audit.masked_score(prediction, target, observed, np.array([2.0, 1.0, 5.0]))
    assert physical["mse"] == pytest.approx(12.5)
    assert physical["mae"] == pytest.approx(3.5)


def test_timing_checks_detect_wrong_paired_raw_seconds():
    schedule = audit.expected_schedule(np)
    route = "E33K"
    raw = {"route": "RAW_BOLT", "block": list(range(6)), "seconds": np.arange(1, 7, dtype=float).tolist()}
    row = {
        "route": route,
        "status": "COMPLETE",
        "warmup_each_route": 2,
        "repeats": 6,
        "batch": 8,
        "seconds": (np.arange(1, 7, dtype=float) * 2.0).tolist(),
        "block": list(range(6)),
        "position": [order.index(route) for order in schedule],
        "median_seconds": float(np.median(np.arange(1, 7, dtype=float) * 2.0)),
        "median_seconds_per_window": float(np.median(np.arange(1, 7, dtype=float) * 2.0) / 8.0),
        "paired_RAW_seconds": raw["seconds"],
        "paired_RAW_ratios": [2.0] * 6,
        "paired_RAW_median_ratio": 2.0,
    }
    failures = []
    audit.check_timing(row, raw, schedule, failures, "fixture")
    assert failures == []
    row["paired_RAW_seconds"] = list(reversed(raw["seconds"]))
    audit.check_timing(row, raw, schedule, failures, "fixture")
    assert any("paired_raw_seconds" in item for item in failures)


def test_scientific_audit_binding_requires_four_pass_conditions_and_seed92601_cost_producers(tmp_path):
    producers = []
    for dataset in audit.DATASETS:
        path = tmp_path / (dataset + "_producer.json")
        path.write_text(json.dumps({"dataset": dataset, "seed": audit.SEED}), encoding="utf-8")
        producers.append(audit.file_record(path))
    conditions = []
    for dataset in audit.DATASETS:
        for seed in (92601, 92602):
            sha = next((record["sha256"] for record in producers if dataset in record["path"]), "dummy" + str(seed))
            conditions.append({"dataset": dataset, "seed": seed, "producer_sha256": sha, "status": "PASS"})
    science = {
        "status": "PASS",
        "failed_count": 0,
        "conditions": conditions,
        "candidate_rows_checked": 108,
        "new_ridge_candidates_checked": 84,
        "selected_predictions_checked": 36,
        "source_before_after_SHA_pass": True,
    }
    assert len(audit.bind_scientific_audit(science, {}, producers, [])) == 4
    science["conditions"][0]["status"] = "FAILED_PRESERVED"
    with pytest.raises(audit.CostAuditFailure, match="ROW_NOT_PASS"):
        audit.bind_scientific_audit(science, {}, producers, [])


def test_source_declares_runtime_before_numpy_assets_and_no_torch_import():
    source = (HERE / "audit_main_residual_cost_v1.py").read_text(encoding="utf-8-sig")
    assert "main_residual_runtime_v1" in source
    assert "import torch" not in source
    assert "profile_main_residual_cost_v1" not in source
    audit_body = source[source.index("def audit") :]
    assert audit_body.index("runtime.ContinuationLimit") < audit_body.index("import numpy as np")
