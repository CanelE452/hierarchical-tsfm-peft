"""Synthetic metadata contract and figure fixtures; no real experiment execution."""
import copy
import json

import pytest

import make_fit_target_report_v1 as report
from test_make_decoder_metric_report_v1 import score, synthetic_metadata as old_metadata


def synthetic_metadata():
    result = {"datasets": {}, "cost": {}}
    audit = {"output_subspace_diagnostics": {}}
    _, old_cost = old_metadata()
    record = {"path": "not_opened.npz", "sha256": "0" * 64}
    for dataset, (channels, k, dev) in report.DATASETS.items():
        data = {"status": "COMPLETE", "K": k, "raw_scores": {split: score(.8, channels) for split in ("fit", "tune", dev)}, "projections": {}}
        audit["output_subspace_diagnostics"][dataset] = {}
        for route in report.PROJECTIONS:
            supervised = route == "FIT_TARGET_PINV"
            meta = {"actual_future_target_used_in_projection_or_decoder_fit": supervised,
                "TUNE_or_DEV_used_in_projection_or_decoder_fit": False, "latent_fit_covariance": [[float(i == j) for j in range(k)] for i in range(k)]}
            if supervised:
                singular = [1. / (i + 1) for i in range(channels)]
                meta.update(target_singularvalues=singular, FIT_target_innovation_total_energy=sum(s * s for s in singular),
                    fit_complete_future_positions=100, fit_total_future_positions=110, retained_whitening_eigenvalues=[.5 / (i + 1) for i in range(k)], eigenvalue_relative_cut=1e-8)
            condition = {"fit_metadata": meta, "state_artifact": record,
                "bolt": {"scores": {split: score(.7, channels) for split in ("fit", "tune", dev)}}, "native": {}}
            for seed in report.SEEDS:
                native = {"status": "COMPLETE", "metadata": {"updates": 600, "initial_weights_and_biases_zero": True,
                    "seed_changes_initial_weights": False, "selected_update": 200, "train_wall_seconds": 1.2,
                    "parameters": k * 48 * 513, "batch_size": 4}, "scores": {split: score(.6, channels) for split in ("fit", "tune", dev)},
                    "checkpoints": [{"update": update, "tune": score(.6 if update == 200 else .8, channels)["normalized"],
                        "state_sha256": "0" * 64, "state_artifact": record, "tune_prediction_artifact": record} for update in (0, 100, 200, 400, 600)],
                    "learning_curve": [{"update": update, "loss": 1. / update, "gradient_norm_before_clip": .1} for update in range(1, 601)]}
                condition["native"][seed] = native
            data["projections"][route] = condition
            audit["output_subspace_diagnostics"][dataset][route] = {}
            for split in ("fit", "tune", dev):
                methods = {method: {"saved_prediction_scores_same_complete_mask": score(.6, channels),
                    "real_arithmetic_scores_same_complete_mask": score(.6, channels),
                    "saved_minus_fixed_D_minimum_normalized_MSE": .4, "real_minus_fixed_D_minimum_normalized_MSE": .4,
                    "measured_FP32_rounding_MSE_allowance": 1e-8} for method in ("BOLT", "NATIVE_s92601", "NATIVE_s92602", "E_Y")}
                audit["output_subspace_diagnostics"][dataset][route][split] = {
                    "complete_target_positions": 100, "total_target_positions": 110,
                    "fixed_D_minimum_normalized_MSE": .2,
                    "fixed_D_real_arithmetic_solution_scores_same_complete_mask": score(.2, channels),
                    "RAW_scores_same_complete_mask": score(.9, channels), "methods_same_complete_mask": methods,
                    "E_Y_minus_fixed_D_minimum_normalized_MSE": .4, "geometry": {"rank_D": k}}
        result["datasets"][dataset] = data
        rows = copy.deepcopy(old_cost["cost"][dataset]["rows"])
        old_routes = ("RAW_BOLT", "PCA_PAST_LS_BOLT", "FORECA_PAST_LS_BOLT", "PCA_ENCODER_PINV_BOLT", "FORECA_ENCODER_PINV_BOLT")
        for row in rows: row["route"] = report.ROUTES[old_routes.index(row["route"])]
        result["cost"][dataset] = {"rows": rows}
    return result, audit


def gate_fixture(tmp_path):
    def write(name, value):
        path = tmp_path / name; path.write_text(json.dumps(value), encoding="utf-8"); return path
    source = report.record(write("source_projection.json", {"status": "COMPLETE"}))
    source_audit = report.record(write("source_audit.json", {"status": "PASS"}))
    result, audit = synthetic_metadata()
    result.update(status="COMPLETE", projection_fits=2, reused_projection_states=2, projection_conditions=4, native_training_fits=8,
        source_projection=source, source_projection_audit=source_audit,
        encoder_fit_uses_FIT_true_future={"PCA_PINV": False, "FIT_TARGET_PINV": True},
        original_LS_native_heads_reused_as_optimized_inverse_controls=False, canonical_native_batch_size=32,
        F_before={"hash": "frozen"}, F_after={"hash": "frozen"}, new_backbone_training=False,
        prior_source_results_caches_and_selection_locks_unchanged=True, protected_TEST_accessed=False, independent_holdout=False)
    result_path = write("RESULT.json", result)
    audit.update(status="PASS", producer=report.record(result_path), source_projection=source, source_projection_audit=source_audit,
        protected_TEST_accessed=False, totals={"projection_states_checked": 4, "target_projection_fits_checked": 2,
        "reused_PCA_controls_checked": 2, "native_fits_checked": 8, "checkpoint_states_checked": 40,
        "cost_cells_checked": 20, "recorded_timer_samples_checked": 120, "saved_prediction_arrays_checked": 88,
        "output_subspace_diagnostic_cells_checked": 12, "fixed_D_real_method_inequalities_checked": 48, "bound_RAW_inequalities_checked": 0})
    return result_path, write("AUDIT.json", audit)


def test_complete_exact_accepted_metadata(tmp_path):
    result, audit = report.load_verified(*gate_fixture(tmp_path))
    assert result["projection_fits"] == 2 and audit["status"] == "PASS"


@pytest.mark.parametrize("change", ["pending", "stale_producer", "stale_original", "past_only_training_claim", "old_native_reused", "missing_span_cell", "RAW_bound"])
def test_incomplete_or_misrepresented_report_gate_is_closed(tmp_path, change):
    result_path, audit_path = gate_fixture(tmp_path)
    if change == "pending":
        audit = report.read(audit_path); audit["status"] = "PENDING"; audit_path.write_text(json.dumps(audit), encoding="utf-8")
    elif change == "stale_producer": result_path.write_text(result_path.read_text() + " ", encoding="utf-8")
    elif change == "stale_original": (tmp_path / "source_projection.json").write_text('{"changed":true}', encoding="utf-8")
    elif change in ("missing_span_cell", "RAW_bound"):
        audit = report.read(audit_path); audit["totals"]["output_subspace_diagnostic_cells_checked" if change == "missing_span_cell" else "bound_RAW_inequalities_checked"] = 11 if change == "missing_span_cell" else 1
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
    else:
        result = report.read(result_path)
        if change == "past_only_training_claim": result["encoder_fit_uses_FIT_true_future"]["FIT_TARGET_PINV"] = False
        else: result["original_LS_native_heads_reused_as_optimized_inverse_controls"] = True
        result_path.write_text(json.dumps(result), encoding="utf-8")
        audit = report.read(audit_path); audit["producer"] = report.record(result_path); audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError): report.load_verified(result_path, audit_path)


def test_supervised_training_and_matched_mask_minimum_scope_kept():
    tables = report.collect_tables(*synthetic_metadata())
    assert {name: len(rows) for name, rows in tables.items()} == {
        "COMPARISON": 84, "MATCHED_COMPLETE_DIAGNOSTIC": 144, "OUTPUT_SPACE_GEOMETRY": 12, "CHECKPOINTS": 40,
        "LEARNING_CURVE": 4800, "TRAINING_COST": 8, "FIT_SPACE_METADATA": 4, "FIT_TARGET_SPECTRUM": 28,
        "RESTRICTED_PAST_VARIANCE": 6, "WHOLE_COST": 20, "WHOLE_COST_SAMPLES": 120, "REGISTERED_ADOPTION": 8}
    target = [row for row in tables["FIT_SPACE_METADATA"] if row["projection"] == "FIT_TARGET_PINV"]
    assert all(row["supervised_FIT_future_training"] and not row["past_only_training"] and row["deployment_past_only"] for row in target)
    assert all(row["mask"] == "all_C_complete" and not row["MAE_is_minimum"] and not row["physical_MSE_is_minimum"]
               and not row["RAW_bound_applied"] for row in tables["MATCHED_COMPLETE_DIAGNOSTIC"])
    assert all(row["this_MSE_is_a_fixed_D_minimum"] == (row["predictor"] == "FIXED_D_MINIMUM" and row["metric_scale"] == "normalized")
               for row in tables["MATCHED_COMPLETE_DIAGNOSTIC"])
    assert all(not row["original_LS_head_reused"] and not row["seed_changes_initial_weights"] for row in tables["TRAINING_COST"])


def test_figures_use_five_standalone_png_svg_pairs(tmp_path):
    paths = report.figures(report.collect_tables(*synthetic_metadata()), tmp_path)
    assert len(paths) == 10 and sum(path.suffix == ".png" for path in paths) == 5
    assert all(path.is_file() and path.stat().st_size > 1000 for path in paths)


def test_report_marks_adaptive_design_and_indirect_D_supervision(tmp_path):
    result, audit = synthetic_metadata(); tables = report.collect_tables(result, audit)
    text = report.text_report(result, audit, tables, tmp_path, {})
    assert "탐색적·적응적 대조" in text and "D 역시 FIT 미래의 간접 감독" in text
    assert "독립 잡음" in text and "원래 사용자의 문장 그대로 승인받았다고 주장" in text
    assert "checkpoint gate에 oracle" in text and "전체 방법 선정 동기" in text
