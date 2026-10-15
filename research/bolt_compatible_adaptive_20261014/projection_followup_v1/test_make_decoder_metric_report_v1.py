"""Synthetic JSON and plotting fixtures; no actual scientific inputs are opened."""
from pathlib import Path
import copy
import json

import pytest

import make_decoder_metric_report_v1 as report


def score(value, channels=2):
    metric = {"mse": value, "mae": value / 2, "observed_counts": [10] * channels,
              "active_channels": channels, "per_channel_mse": [value] * channels,
              "per_channel_mae": [value / 2] * channels}
    return {"normalized": metric, "physical_per_channel_diagnostic": copy.deepcopy(metric)}


def synthetic_metadata():
    decoder, cost = {"datasets": {}}, {"cost": {}}
    source = {"path": "not_opened.npz", "sha256": "a" * 64}
    for dataset, (channels, k, development) in report.DATASETS.items():
        conditions = {}
        for projection in ("PCA", "FORECA"):
            splits = {}
            for split in ("fit", "tune", development):
                cell = {"complete_target_positions": 10, "total_target_positions": 12,
                    "RAW_scores_all_observed": score(.8, channels), "RAW_scores_complete_mask": score(.9, channels),
                    "source_input": source, "artifact": source, "cached_future_sources": {"BOLT": source},
                    "alternative_decoder_construction": {"future_labels_used_to_construct_D": False}, "decoders": {}}
                for variant in report.DECODERS:
                    forecasts = {predictor: {"scores_all_observed": score(.7, channels), "scores_complete_mask": score(.75, channels)}
                                 for predictor in report.DEPLOY}
                    forecasts["E_Y"] = {"scores_all_observed": score(999., channels), "scores_complete_mask": score(.15, channels)}
                    cell["decoders"][variant] = {"forecasts": forecasts,
                        "geometry": {"rank_D": k, "E_D_symmetry_max_abs": .01, "D_E_minus_I_max_abs": .001},
                        "fixed_D_real_arithmetic_diagnostic_same_complete_mask": {"scores": score(.1, channels),
                            "independent_linear_checks": {"orthogonality_pass": True}},
                        "past_reconstruction_observed_only": score(.2, channels), "past_reconstruction_bias_rule": "synthetic",
                        "native_heads_optimized_for_this_D": variant == "ORIGINAL_PAST_LS"}
                splits[split] = cell
            conditions[projection] = {"source_state": source, "splits": splits}
        decoder["datasets"][dataset] = {"K": k, "development_split": development, "projections": conditions}
        rows = []
        for batch in (1, 8):
            for ri, route in enumerate(report.ROUTES):
                median = 1. if ri == 0 else 1.2 if ri == 1 else .8
                seconds = [median] * 6
                memory = {"resident_rss_bytes": 1024, "peak_reserved_vram_bytes": 2048}
                samples = [{"repeat": repeat, "position": (repeat + ri) % 5, "seconds": median,
                            "perf_start_ns": repeat * 10**10, "perf_end_ns": repeat * 10**10 + int(median * 1e9),
                            "parity": {name: {"pass": True} for name in ("normalized", "physical", "scientific_normalized", "scientific_physical")},
                            "memory": memory, "sample_artifact": source} for repeat in range(6)]
                rows.append({"route": route, "batch": batch, "repeats": 6, "median_seconds": median,
                    "median_seconds_per_window": median / batch, "paired_RAW_median_ratio": .2 if ri else 1.,
                    "seconds": seconds, "paired_RAW_seconds": [1.] * 6, "paired_RAW_ratios": [.2 if ri else 1.] * 6,
                    "positions": [sample["position"] for sample in samples], "origins": list(range(batch)),
                    "warmup_each_route": 2, "memory_scope": "shared process", "memory": [memory] * 6,
                    "whole_includes": "actual forward", "whole_excludes": "load/save", "samples": samples,
                    "selected_metadata": {"synthetic": True}, "timed_outputs": source, "reference": source})
        cost["cost"][dataset] = {"rows": rows}
    return decoder, cost


def gate_fixture(tmp_path):
    def write(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path
    chain = {key: report.record(write(key + ".json", {"fixture": key})) for key in
             ("producer", "projection_audit", "span_diagnostic", "span_audit")}
    decoder = {"status": "COMPLETE", **chain, "completed_cells": 12, "failed_paths": [],
        "new_training": False, "new_encoder_F_or_native_head_forward": False,
        "future_labels_used_to_fit_alternative_D": False, "method_checkpoint_branch_selection_used": False,
        "protected_TEST_accessed": False, "original_selected_forecasts_and_locks_unchanged": True}
    decoder_path = write("DECODER.json", decoder)
    audit = {"status": "PASS", **chain, "diagnostic": report.record(decoder_path), "protected_TEST_accessed": False,
             "totals": {"base_cells_checked": 12, "decoder_cells_checked": 24,
                        "cached_forecast_arrays_checked": 96, "original_predictions_byte_exact_checked": 48}}
    audit_path = write("DECODER_AUDIT.json", audit)
    cost = {"status": "COMPLETE", "producer": chain["producer"], "projection_audit": chain["projection_audit"],
        "decoder_diagnostic": report.record(decoder_path), "decoder_audit": report.record(audit_path),
        "new_training_updates": 0, "projection_fits": 0, "alternative_decoder_refits": 0,
        "method_checkpoint_branch_selection_used": False, "F_before": {"hash": "frozen"}, "F_after": {"hash": "frozen"},
        "expected_cost_cells": 20, "recorded_timer_samples": 120, "routes": list(report.ROUTES),
        "protected_TEST_accessed": False, "original_selected_forecasts_and_locks_unchanged": True}
    cost_path = write("COST.json", cost)
    cost_audit = {"status": "PASS", "cost": report.record(cost_path), "producer": chain["producer"],
        "projection_audit": chain["projection_audit"], "decoder_diagnostic": report.record(decoder_path),
        "decoder_audit": report.record(audit_path), "protected_TEST_accessed": False,
        "totals": {"cost_cells_checked": 20, "recorded_timer_samples_checked": 120}}
    cost_audit_path = write("COST_AUDIT.json", cost_audit)
    return decoder_path, audit_path, cost_path, cost_audit_path


def test_both_actual_audits_and_exact_metadata_chain_required(tmp_path):
    values = report.load_verified(*gate_fixture(tmp_path))
    assert values[0]["status"] == values[2]["status"] == "COMPLETE"


@pytest.mark.parametrize("change", ["cost_pending", "decoder_stale", "producer_changed", "missing_sample", "training_used", "native_cost_mislabeled"])
def test_gate_stays_closed_for_unverified_or_changed_scope(tmp_path, change):
    paths = gate_fixture(tmp_path)
    if change == "cost_pending":
        value = report.read(paths[-1]); value["status"] = "PENDING"
        paths[-1].write_text(json.dumps(value), encoding="utf-8")
    elif change == "decoder_stale":
        paths[0].write_text(paths[0].read_text() + " ", encoding="utf-8")
    elif change == "producer_changed":
        (tmp_path / "producer.json").write_text('{"changed":true}', encoding="utf-8")
    elif change == "missing_sample":
        value = report.read(paths[-1]); value["totals"]["recorded_timer_samples_checked"] = 119
        paths[-1].write_text(json.dumps(value), encoding="utf-8")
    else:
        value = report.read(paths[2])
        if change == "training_used": value["new_training_updates"] = 1
        else: value["routes"][-1] = "FORECA_NATIVE"
        paths[2].write_text(json.dumps(value), encoding="utf-8")
        audit = report.read(paths[-1]); audit["cost"] = report.record(paths[2])
        paths[-1].write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError): report.load_verified(*paths)


def test_oracle_mask_minimum_and_frozen_head_scope_are_separated():
    tables = report.collect_tables(*synthetic_metadata())
    assert {key: len(value) for key, value in tables.items()} == {
        "DEPLOYMENT_AND_ORACLE_SCORES": 192, "FIXED_D_DIAGNOSTIC": 24, "GEOMETRY": 24,
        "PAST_RECONSTRUCTION": 24, "SOURCE_LINEAGE": 12, "WHOLE_COST": 20,
        "WHOLE_COST_SAMPLES": 120, "REGISTERED_ADOPTION": 8}
    oracle = [row for row in tables["DEPLOYMENT_AND_ORACLE_SCORES"] if row["predictor"] == "E_Y"]
    assert len(oracle) == 24 and all(row["mask"] == "all_C_complete" and row["normalized_MSE"] == .15 for row in oracle)
    assert all(row["normalized_MSE"] != 999. for row in tables["DEPLOYMENT_AND_ORACLE_SCORES"])
    assert all(row["normalized_MSE_is_minimum"] and not row["MAE_is_minimum"]
               and not row["physical_MSE_is_minimum"] and not row["Raw_is_constrained_to_this_D"] for row in tables["FIXED_D_DIAGNOSTIC"])
    assert all(not row["native_heads_reoptimized_for_new_D"] and not row["causal_obliqueness_only_effect"] for row in tables["GEOMETRY"])
    assert all(not row["native_inference_cost_measured"] for row in tables["WHOLE_COST"])


def test_registered_cost_gate_uses_ratio_of_medians_not_paired_ratio():
    tables = report.collect_tables(*synthetic_metadata())
    row = next(row for row in tables["REGISTERED_ADOPTION"] if row["route"] == "PCA_PAST_LS_BOLT")
    assert row["registered_B8_ratio_of_medians"] == 1.2 and not row["latency_le_0_90"]
    assert not row["all_registered_conditions_pass"]
    diagnostic = next(row for row in tables["WHOLE_COST"] if row["route"] == "PCA_PAST_LS_BOLT" and row["batch"] == 8)
    assert diagnostic["median_of_paired_RAW_ratios"] == .2


def test_figures_write_exactly_three_png_and_three_svg_from_synthetic_metadata(tmp_path):
    tables = report.collect_tables(*synthetic_metadata())
    files = report.figure_outputs(tables, tmp_path)
    assert len(files) == 6 and sum(path.suffix == ".png" for path in files) == 3
    assert all(path.is_file() and path.stat().st_size > 1000 for path in files)
    with pytest.raises(FileExistsError): report.figure_outputs(tables, tmp_path)


def test_csv_keeps_original_numeric_fields_and_scope_flags(tmp_path):
    tables = report.collect_tables(*synthetic_metadata())
    path = tmp_path / "scores.csv"
    report.write_csv(path, tables["DEPLOYMENT_AND_ORACLE_SCORES"])
    assert "normalized_MSE" in path.read_text(encoding="utf-8")
    assert "all_C_complete" in path.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError): report.write_csv(path, tables["DEPLOYMENT_AND_ORACLE_SCORES"])
