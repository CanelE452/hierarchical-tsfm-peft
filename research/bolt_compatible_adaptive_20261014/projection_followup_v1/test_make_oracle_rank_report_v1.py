"""Synthetic metadata fixtures for the oracle-only addendum gate."""

import json

import pytest

from make_oracle_rank_report_v1 import load_verified, collect_tables, sha


def fixture(tmp_path):
    bindings = {}
    for key in ("producer", "projection_audit", "fixed_span", "fixed_span_audit"):
        path = tmp_path / (key + ".json")
        path.write_text(json.dumps({"fixture": key}), encoding="utf-8")
        bindings[key] = {"path": str(path), "sha256": sha(path)}
    diagnostic = tmp_path / "DIAGNOSTIC.json"
    data = {"status": "COMPLETE", **bindings, "protected_TEST_accessed": False,
            "deployment_or_method_checkpoint_branch_selection_used": False,
            "original_selected_forecasts_and_selection_lock_unchanged": True}
    diagnostic.write_text(json.dumps(data), encoding="utf-8")
    audit = tmp_path / "AUDIT.json"
    audit.write_text(json.dumps({"status": "PASS", "diagnostic": {"sha256": sha(diagnostic)},
                                "producer": bindings["producer"], "projection_audit": bindings["projection_audit"],
                                "span_diagnostic": bindings["fixed_span"], "span_audit": bindings["fixed_span_audit"],
                                "totals": {"dataset_split_cells_checked": 6}}), encoding="utf-8")
    return diagnostic, audit


def test_exact_metadata_chain_required(tmp_path):
    diagnostic, audit = load_verified(*fixture(tmp_path))
    assert diagnostic["status"] == "COMPLETE" and audit["status"] == "PASS"


@pytest.mark.parametrize("change", ["stale_diagnostic", "audit_failed", "linked_producer_changed", "oracle_selection_used"])
def test_stale_or_deployment_misuse_gate_closed(tmp_path, change):
    diagnostic, audit = fixture(tmp_path)
    if change == "stale_diagnostic":
        diagnostic.write_text(diagnostic.read_text() + " ", encoding="utf-8")
    elif change == "audit_failed":
        data = json.loads(audit.read_text()); data["status"] = "FAILED"
        audit.write_text(json.dumps(data), encoding="utf-8")
    elif change == "linked_producer_changed":
        (tmp_path / "producer.json").write_text('{"changed":true}', encoding="utf-8")
    else:
        data = json.loads(diagnostic.read_text()); data["deployment_or_method_checkpoint_branch_selection_used"] = True
        diagnostic.write_text(json.dumps(data), encoding="utf-8")
        checked = json.loads(audit.read_text()); checked["diagnostic"]["sha256"] = sha(diagnostic)
        audit.write_text(json.dumps(checked), encoding="utf-8")
    with pytest.raises(ValueError): load_verified(diagnostic, audit)


def synthetic_cells():
    metric = {"mse": 1.25, "mae": .8, "observed_counts": [10,10], "active_channels": 2,
              "per_channel_mse": [1.,1.5], "per_channel_mae": [.7,.9]}
    cell = {"directions_fitted_to_known_future_of_this_split": True,
            "one_shared_output_space_across_splits_assumed": False,
            "deployment_used": False, "method_checkpoint_branch_selection_used": False,
            "new_deployable_training": False, "centered_or_future_intercept_used": False,
            "channels": 2, "K": 1, "complete_target_positions": 10, "total_target_positions": 12,
            "minimum_normalized_MSE_from_discarded_energy": 1.25,
            "RAW_scores_same_mask": {"normalized": {"mse": 2.}},
            "fixed_D_and_RAW_comparisons_same_mask": {"PCA": {"fixed_D_floor_MSE_same_mask": 1.5},
                                                      "FORECA": {"fixed_D_floor_MSE_same_mask": 1.8}},
            "energy_certificate": {"total_innovation_energy": 100., "retained_energy": 75., "discarded_energy": 25.},
            "future_matrix_shape": [10,2], "numerical_rank_future_matrix": 2,
            "rank_K_oracle_scores_same_mask": {"normalized": metric, "physical_per_channel_diagnostic": metric}}
    return {"datasets": {dataset: {"splits": {split: json.loads(json.dumps(cell)) for split in splits}}
                          for dataset, splits in (("robin", ("fit","tune","val")),
                                                  ("beijing_aotizhongxin", ("fit","tune","eval")))}}


def test_future_fitted_scope_and_non_MSE_minima_are_preserved():
    tables = collect_tables(synthetic_cells(), {"cells": []})
    assert len(tables["COMPARISON"]) == 6
    assert all(row["future_informed_direction_selection"] and not row["deployable"]
               and not row["one_D_across_splits"] for row in tables["COMPARISON"])
    assert all(not row["MAE_is_minimum"] and not row["physical_MSE_is_minimum"]
               for row in tables["ORACLE_SCORE_VALUES"])
    assert all(row["MSE_is_minimum"] == (row["metric_scale"] == "normalized")
               for row in tables["ORACLE_SCORE_VALUES"])


def test_one_shared_future_space_cannot_be_assumed():
    data = synthetic_cells()
    data["datasets"]["robin"]["splits"]["fit"]["one_shared_output_space_across_splits_assumed"] = True
    with pytest.raises(ValueError): collect_tables(data, {"cells": []})
