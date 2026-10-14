import pytest

import make_stage_separation_report_v1 as report


def _score(mse, mae):
    return {"normalized": {"mse": mse, "mae": mae}, "physical_per_channel_diagnostic": {"mse": mse * 2, "mae": mae * 2}}


def _native(dataset="robin", seed=92601, sha="A" * 64):
    final = report.FINAL_SPLIT[dataset]
    return {
        "status": "COMPLETE",
        "dataset": dataset,
        "seed": seed,
        "protected_TEST_accessed": False,
        "F_loaded": False,
        "F_forward_replayed": False,
        "E_forward_replayed": False,
        "independent_holdout": False,
        "references": {
            "fit": {"RAW": _score(1.0, 1.0), "ORIGINAL_MAIN": _score(0.8, 0.8)},
            "tune": {"RAW": _score(1.1, 1.1), "ORIGINAL_MAIN": _score(0.9, 0.9)},
            final: {"RAW": _score(1.2, 1.2), "ORIGINAL_MAIN": _score(1.0, 1.0)},
        },
        "controls": {
            "CURRENT_C_PERSISTENCE": {"splits": {final: _score(1.3, 1.3)}},
            "FIT_INTERCEPT_ONLY": {"model_sha256": sha, "splits": {final: _score(1.1, 1.1)}},
        },
        "SAME_MAIN_NATIVE_existing": {
            "status": "COMPLETE",
            "new_training": False,
            "lineage": {"native_prediction": {"path": "native_old.npz", "sha256": sha}},
            "splits": {final: {"normalized": {"mse": 0.77, "mae": 0.66}, "physical": {"mse": 7.7, "mae": 6.6}}},
        },
        "D2_existing_F_decoder_refits": {
            "status": "COMPLETE",
            "source": {"path": "d2_refit.json", "sha256": sha},
            "refits": {"Y": {"selected_alpha": 0.1, "selected_index": 1,
                             "splits": {final: {"normalized": {"mse": 0.55, "mae": 0.44},
                                                "physical": {"mse": 5.5, "mae": 4.4}}}}},
        },
        "O0_ORACLE_OLD_D": {"splits": {final: _score(0.4, 0.4)}},
        "families": {
            "O1": {"status": "COMPLETE", "diagnostic_only": True, "zero_observed_output_counts": 0,
                   "selected_index": 0, "selected_alpha": 0.0, "selected_model_sha256": sha,
                   "alpha_grid": [{"alpha": 0.0, "model_sha256": sha, "tune_actual_Y_mse": 0.5,
                                   "fit_scores": _score(0.4, 0.4), "tune_scores": _score(0.5, 0.5)}],
                   "splits": {final: _score(0.6, 0.6)}},
            "N1": {"status": "COMPLETE", "diagnostic_only": False, "zero_observed_output_counts": 2,
                   "selected_index": 0, "selected_alpha": 0.0, "selected_model_sha256": sha,
                   "alpha_grid": [{"alpha": 0.0, "model_sha256": sha, "tune_actual_Y_mse": 0.7,
                                   "fit_scores": _score(0.6, 0.6), "tune_scores": _score(0.7, 0.7)}],
                   "splits": {final: _score(0.9, 0.9)}},
            "N2": {"status": "COMPLETE", "diagnostic_only": False, "zero_observed_output_counts": 0,
                   "selected_index": 0, "selected_alpha": 0.0, "selected_model_sha256": sha,
                   "alpha_grid": [{"alpha": 0.0, "model_sha256": sha, "tune_actual_Y_mse": 0.8,
                                   "fit_scores": _score(0.7, 0.7), "tune_scores": _score(0.8, 0.8)}],
                   "splits": {final: _score(0.95, 0.95)}},
            "N3": {"status": "COMPLETE", "diagnostic_only": False, "zero_observed_output_counts": 0,
                   "selected_index": 0, "selected_alpha": 0.0, "selected_model_sha256": sha,
                   "alpha_grid": [{"alpha": 0.0, "model_sha256": sha, "tune_actual_Y_mse": 0.85,
                                   "fit_scores": _score(0.75, 0.75), "tune_scores": _score(0.85, 0.85)}],
                   "splits": {final: _score(0.98, 0.98)}},
        },
        "teacher_contract": {"fit_counts_by_horizon_K": [[1, 2], [2, 3]], "actual_Y_fit_channel_counts": [10, 11]},
    }


def test_collect_native_separates_deploy_oracle_controls_and_reused_raw():
    tables = report.empty_tables()
    native = _native()
    record = {"path": "native.json", "sha256": "A" * 64}
    report.collect_native(native, record, tables)
    audit = {"reused_raw_controls": [{
        "dataset": "robin",
        "status": "COMPLETE",
        "new_training": False,
        "matched_readout_capacity_or_anchor": False,
        "source": {"path": "raw_controls.json", "sha256": "B" * 64},
        "methods": {"RAW_SELF33": {"status": "COMPLETE", "splits": {"val": _score(0.7, 0.7)}}},
    }]}
    report.collect_reused_raw_controls(audit, tables)

    by_method = {row["method"]: row for row in tables["scores"] if row["split"] == "val"}
    assert by_method["N1"]["role"] == "deploy_candidate"
    assert by_method["O1"]["role"] == "future_oracle_diagnostic"
    assert by_method["O0_ORACLE_OLD_D"]["deployable"] is False
    assert by_method["CURRENT_C_PERSISTENCE"]["role"] == "control"
    assert by_method["RAW_SELF33"]["role"] == "reused_RAW_control"
    assert by_method["RAW_SELF33"]["new_training"] is False
    assert by_method["RAW_SELF33"]["matched_readout_capacity_or_anchor"] is False
    assert by_method["OLD_SAME_MAIN_NATIVE"]["role"] == "old_reference"
    assert by_method["OLD_F_REFIT_D_Y"]["physical_mse_diagnostic"] == 5.5
    coverage = [row for row in tables["coverage"] if row["coverage_type"] == "N1_teacher_complete_future_prefix_fit"]
    assert coverage[0]["horizon"] == 1
    assert coverage[0]["count"] == 1
    assert coverage[0]["raw_counts_json"] == [1, 2]


def test_native_audit_binding_requires_exact_four_dataset_seed_sha_conditions():
    records = [
        {"dataset": "robin", "seed": 92601, "sha256": "A" * 64},
        {"dataset": "robin", "seed": 92602, "sha256": "B" * 64},
        {"dataset": "beijing_aotizhongxin", "seed": 92601, "sha256": "C" * 64},
        {"dataset": "beijing_aotizhongxin", "seed": 92602, "sha256": "D" * 64},
    ]
    good = {"status": "PASS", "failed_count": 0, "conditions": [
        {"dataset": row["dataset"], "seed": row["seed"], "producer_sha256": row["sha256"]} for row in records
    ]}
    assert report.validate_native_audit_binding(good, records)["passed"] is True
    bad = {**good, "conditions": good["conditions"][:-1] + [{"dataset": "beijing_aotizhongxin", "seed": 92602, "producer_sha256": "E" * 64}]}
    with pytest.raises(report.ReportFailure, match="EXACT4_NATIVE_AUDIT"):
        report.validate_native_audit_binding(bad, records)


def test_cost_table_marks_unmeasured_without_replaying_bulk_flags():
    cost_gate = {"status": "FAILED_PRESERVED", "reason": "preserved timing failure"}
    rows = report.collect_cost({"status": "FAILED_PRESERVED", "rows": []}, cost_gate)
    assert rows == [{
        "status": "FAILED_PRESERVED",
        "reason": "preserved timing failure",
        "route": None,
        "dataset": None,
        "batch": None,
        "method": None,
        "seed": None,
        "median_ms": None,
        "median_ms_per_window": None,
        "paired_RAW_median_ratio": None,
        "repeats": 0,
        "original_bulk_flags_reported_only": True,
    }]


def test_markdown_states_future_oracle_limitations_and_avoids_cache_links():
    tables = report.empty_tables()
    base = {"dataset": "robin", "seed": 92601, "producer_sha256": "A" * 64, "producer_path": "native.json"}
    report.add_score(tables["scores"], base, "ORIGINAL_MAIN", "val", _score(1.0, 1.0), "fixture")
    report.add_score(tables["scores"], base, "N2", "val", _score(0.9, 0.9), "fixture")
    report.add_score(tables["scores"], base, "O1", "val", _score(0.5, 0.5), "fixture")
    gates = {"native_audit": {"status": "PASS", "failed_count": 0}, "cost": {"status": "NOT_MEASURED", "audit_passed": False}}
    text = report.render_markdown(tables, gates)
    assert "O0/O1은 actual future" in text
    assert "먼저 답해야 할 여섯 질문" in text
    assert "N2 우세가 말하는 범위" in text
    assert "![deploy 점수]" in text
    assert ".cache" not in text


def test_final_manifest_inventory_excludes_manifest_and_uses_actual_result_status(tmp_path):
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / "scores.csv").write_text("x\n1\n", encoding="utf-8")
    (tmp_path / "RESULT.json").write_text('{"status":"FAILED_PRESERVED"}\n', encoding="utf-8")
    result = {"status": "FAILED_PRESERVED"}
    manifest = report.build_final_manifest(result, {"native_audit_gate": {"status": "PASS"}}, tmp_path)
    paths = {row["path"] for row in manifest["artifacts"]}
    assert manifest["generator_result_status"] == "FAILED_PRESERVED"
    assert "RESULT.json" in paths
    assert "tables/scores.csv" in paths
    assert "MANIFEST.json" not in paths
    assert all("sha256" in row and "bytes" in row for row in manifest["artifacts"])
