"""Report gate and CSV fixtures; real arrays/models are never read."""

import json

import pytest

from make_projection_report_v1 import load_verified, load_diagnostic, score_row, sha


def inputs(tmp_path, *, audit_status="PASS", producer_status="COMPLETE", wrong_sha=False):
    result_path, audit_path = tmp_path / "RESULT.json", tmp_path / "AUDIT.json"
    result_path.write_text(json.dumps({"status": producer_status, "protected_TEST_accessed": False,
                                       "new_backbone_training": False}), encoding="utf-8")
    audit_path.write_text(json.dumps({"status": audit_status,
                                      "producer": {"sha256": "bad" if wrong_sha else sha(result_path)},
                                      "totals": {"projection_states_checked": 4, "native_fits_checked": 8,
                                                 "checkpoint_states_checked": 40, "cost_cells_checked": 20,
                                                 "recorded_timer_samples_checked": 120}}), encoding="utf-8")
    return result_path, audit_path


def test_complete_and_bound_pass_audit_required(tmp_path):
    result, audit = load_verified(*inputs(tmp_path))
    assert result["status"] == "COMPLETE" and audit["status"] == "PASS"


@pytest.mark.parametrize("arguments", [{"audit_status": "FAILED_PRESERVED"},
                                       {"producer_status": "FAILED"}, {"wrong_sha": True}])
def test_failed_or_stale_audit_cannot_create_report(tmp_path, arguments):
    with pytest.raises(ValueError): load_verified(*inputs(tmp_path, **arguments))


def test_oracle_csv_role_mask_coverage_not_generalization():
    metric = {"mse": .872, "mae": .6, "active_channels": 2,
              "observed_counts": [10, 10], "per_channel_mse": [.5, 1.244],
              "per_channel_mae": [.4, .8]}
    row = score_row("robin", "EY_ORACLE", "val", "normalized", metric,
                    role="future_EY_diagnostic_matched_complete_mask", complete=10, total=12)
    assert row["complete_target_positions"] == 10 and row["total_target_positions"] == 12
    assert row["role"].startswith("future_EY") and row["independent_holdout"] is False
    assert row["mse"] == .872


def test_channel_physical_aggregate_marked_incomparable():
    metric = {"mse": 10., "mae": 2., "active_channels": 1,
              "observed_counts": [3], "per_channel_mse": [10.], "per_channel_mae": [2.]}
    row = score_row("fixture", "PCA_BOLT", "tune", "physical_per_channel_diagnostic", metric)
    assert row["physical_channel_units_comparable"] is False
    assert row["split_role"] == "selection"


def test_fixed_D_diagnostic_needs_independent_exact_SHA_gate(tmp_path):
    result, audit = inputs(tmp_path)
    diag, diag_audit = tmp_path / "DIAG.json", tmp_path / "DIAG_AUDIT.json"
    diag.write_text(json.dumps({"status": "COMPLETE", "producer": {"sha256": sha(result)},
                               "independent_projection_audit": {"sha256": sha(audit)},
                               "protected_TEST_accessed": False,
                               "checkpoint_or_method_or_branch_selection_changed": False,
                               "original_producer_selected_forecasts_changed": False,
                               "diagnostic_cells": 12}), encoding="utf-8")
    diag_audit.write_text(json.dumps({"status": "PASS", "diagnostic": {"sha256": sha(diag)},
                                     "producer": {"sha256": sha(result)},
                                     "projection_audit": {"sha256": sha(audit)}}), encoding="utf-8")
    assert load_diagnostic(result, audit, diag, diag_audit)["status"] == "COMPLETE"
    with pytest.raises(ValueError): load_diagnostic(result, audit, diag, None)
    data = json.loads(diag_audit.read_text()); data["diagnostic"]["sha256"] = "stale"
    diag_audit.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError): load_diagnostic(result, audit, diag, diag_audit)
