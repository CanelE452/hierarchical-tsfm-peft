"""Synthetic report contracts and scope fixtures; never scientific data/runtime."""
import copy
import json

import pytest

import make_past_constant_gate_report_v1 as report
from test_make_decoder_metric_report_v1 import score, synthetic_metadata as old_metadata


def synthetic_metadata():
    cpu = {"datasets": {}}; cost = {"datasets": {}, "cost": {}}; cost_audit = {"datasets": {}}
    _, templates = old_metadata()
    binding = {"path": "not_opened.npz", "sha256": "a" * 64}
    for dataset, (channels, k, dev) in report.DATASETS.items():
        cpu["datasets"][dataset] = {"channels": channels, "K": k, "splits": {}}
        for split, n in (("fit", 256), ("tune", 64), (dev, 30 if dataset == "robin" else 146)):
            eligible = 0 if split == "fit" else 3
            changed = 1 if split == dev else 0
            arrays = {"exact_flat_windows_per_channel": [eligible] * channels,
                "gate_eligible_windows_per_channel": [eligible] * channels,
                "eligible_windows_with_any_observed_future_per_channel": [eligible] * channels,
                "eligible_windows_with_observed_future_change_per_channel": [changed] * channels,
                "eligible_observed_future_positions_per_channel": [eligible * 48] * channels,
                "eligible_observed_future_changed_positions_per_channel": [changed] * channels}
            methods = {}
            for method in report.METHODS:
                methods[method] = {"ungated_scores_all_observed": score(.8, channels), "gated_scores_all_observed": score(.79, channels),
                    "ungated_scores_complete_mask": score(.9, channels), "gated_scores_complete_mask": score(.89, channels),
                    "unchanged_coordinates_byte_exact": True, "gated_coordinates_current_anchor_byte_exact": True,
                    "changed_prediction_coordinates_per_channel": [eligible * 48] * channels,
                    "native_head_frozen_intervention": "NATIVE" in method, "optimized_gated_native_head_claimed": False}
            cpu["datasets"][dataset]["splits"][split] = {"windows": n, "channels": channels,
                "complete_target_positions": n * 48, "total_target_positions": n * 48, "methods": methods, **arrays,
                "input_source": binding, "artifact": binding, "cached_prediction_sources": {method: binding for method in report.METHODS},
                "future_change_diagnostic_scope": "Normalized FP32 observed future diagnostic only"}
        pre = {"status": "PASS", "all_three_gate_arrays_byte_exact": True, "tolerance_parity_used_to_accept_gate_decision": False}
        cost["datasets"][dataset] = {"physical_gate_precheck": pre}
        cost_audit["datasets"][dataset] = {"physical_gate_precheck": {
            "first_B1_B8_gate_arrays_byte_exact": True, "gate_decision_tolerance_used": False,
            "physical_gate_eligible_window_channel_pairs_first8": 59 if dataset == "robin" else 0}}
        rows = []
        for batch in (1, 8):
            template = next(row for row in templates["cost"][dataset]["rows"] if row["route"] == "RAW_BOLT" and row["batch"] == batch)
            for ri, route in enumerate(report.ROUTES):
                row = copy.deepcopy(template); row["route"] = route
                value = (1., 1.05, .75, .8, .85, .9)[ri]
                row.update(seconds=[value] * 6, median_seconds=value, median_seconds_per_window=value / batch,
                    paired_RAW_ratios=[value] * 6, paired_RAW_median_ratio=value, positions=[(ri + repeat) % 6 for repeat in range(6)])
                for repeat, sample in enumerate(row["samples"]):
                    sample.update(seconds=value, position=(ri + repeat) % 6,
                        perf_end_ns=sample["perf_start_ns"] + int(value * 1e9))
                rows.append(row)
        cost["cost"][dataset] = {"rows": rows}
    identity = {"failed_audit": binding, "fresh_audit": binding, "failed_audit_source": binding, "fresh_audit_source": binding,
        "registered_producer_seed": 20261015, "incorrect_v1_auditor_seed_literal": 24261015,
        "source_difference": {"exactly_one_seed_guard_literal_restoration_verified": True,
            "prediction_metric_tolerance_counts_and_cost_measurement_unchanged": True}, "regression_tests": {"fresh_tests_passed": 14}}
    return cpu, {}, cost, cost_audit, identity


def gate_fixture(tmp_path):
    def write(name, value):
        path = tmp_path / name; path.write_text(json.dumps(value), encoding="utf-8"); return path
    p = report.record(write("FIT_RESULT.json", {"status": "COMPLETE"}))
    fa = report.record(write("FIT_AUDIT.json", {"status": "PASS"}))
    cpu, audit, cost, cost_audit, identity = synthetic_metadata()
    cpu.update(status="COMPLETE", producer=p, independent_FIT_target_audit=fa, protected_TEST_accessed=False,
        new_training=False, new_encoder_forecaster_or_head_forward=False, E_F_native_head_forward_runs=0,
        future_labels_used_for_gate=False, method_selection_used=False, past_C_condition_metadata_added=True,
        same_decoder_side_information_as_ungated=False, static_original_fixed_D_floor_reused_as_gated_lower_bound=False,
        source_cache_selection_SHA_unchanged=True)
    cpu_path = write("CPU.json", cpu)
    audit.update(status="PASS", producer=p, fit_target_audit=fa, diagnostic=report.record(cpu_path), protected_TEST_accessed=False,
        totals={"dataset_split_cells_checked": 6, "method_pairs_checked": 42,
            "ungated_prediction_arrays_checked": 42, "gated_prediction_arrays_checked": 42, "score_field_checks": 2016})
    audit_path = write("CPU_AUDIT.json", audit)
    cost.update(status="COMPLETE", producer=p, fit_target_audit=fa, gate_diagnostic=report.record(cpu_path), gate_audit=report.record(audit_path),
        protected_TEST_accessed=False, new_training_updates=0, projection_fits=0, new_native_head_forward=False,
        native_deployment_cost_measured=False, routes=list(report.ROUTES), F_before={"hash": "frozen"}, F_after={"hash": "frozen"},
        source_cache_selection_SHA_unchanged=True)
    cost_path = write("COST.json", cost)
    cost_audit.update(status="PASS", cost=report.record(cost_path), producer=p, fit_target_audit=fa,
        gate_diagnostic=report.record(cpu_path), gate_audit=report.record(audit_path), protected_TEST_accessed=False,
        totals={"cost_cells_checked": 24, "recorded_timer_samples_checked": 144, "physical_gate_prechecks_checked": 2,
            "paired_gate_timed_output_arrays_checked": 72, "original_selected_forecast_hashes_checked": 36})
    cost_audit_path = write("COST_AUDIT_v2.json", cost_audit)
    identity.update(status="CONFIRMED_AUDITOR_REGISTERED_SEED_LITERAL_ERROR_REPAIRED", cost_experiment_reexecuted=False,
        F_or_native_head_called=False, original_failed_source_audit_and_receipt_preserved=True,
        fresh_audit=report.record(cost_audit_path), cost_producer=report.record(cost_path),
        failed_audit=report.record(write("FAILED_COST_AUDIT_v1.json", {"status": "FAILED_PRESERVED"})))
    identity_path = write("IDENTITY.json", identity)
    return cpu_path, audit_path, cost_path, cost_audit_path, identity_path


def test_exact_accepted_metadata_and_auditor_repair_receipt(tmp_path):
    values = report.load_verified(*gate_fixture(tmp_path))
    assert values[0]["status"] == values[2]["status"] == "COMPLETE"


@pytest.mark.parametrize("change", ["pending", "stale_CPU", "missing_timer", "native_timed", "same_interface", "tolerance_gate", "cost_rerun"])
def test_gate_rejects_incomplete_or_wrong_comparison_scope(tmp_path, change):
    paths = gate_fixture(tmp_path)
    if change == "pending":
        value = report.read(paths[3]); value["status"] = "PENDING"; paths[3].write_text(json.dumps(value), encoding="utf-8")
    elif change == "stale_CPU": paths[0].write_text(paths[0].read_text() + " ", encoding="utf-8")
    elif change == "missing_timer":
        value = report.read(paths[3]); value["totals"]["recorded_timer_samples_checked"] = 143; paths[3].write_text(json.dumps(value), encoding="utf-8")
    elif change == "cost_rerun":
        value = report.read(paths[4]); value["cost_experiment_reexecuted"] = True; paths[4].write_text(json.dumps(value), encoding="utf-8")
    else:
        index = 0 if change == "same_interface" else 2
        value = report.read(paths[index])
        if change == "same_interface": value["same_decoder_side_information_as_ungated"] = True
        elif change == "native_timed": value["native_deployment_cost_measured"] = True
        else: value["datasets"]["robin"]["physical_gate_precheck"]["tolerance_parity_used_to_accept_gate_decision"] = True
        paths[index].write_text(json.dumps(value), encoding="utf-8")
        if index == 2:
            a = report.read(paths[3]); a["cost"] = report.record(paths[2]); paths[3].write_text(json.dumps(a), encoding="utf-8")
            i = report.read(paths[4]); i["fresh_audit"] = report.record(paths[3]); i["cost_producer"] = report.record(paths[2]); paths[4].write_text(json.dumps(i), encoding="utf-8")
        else:
            a = report.read(paths[1]); a["diagnostic"] = report.record(paths[0]); paths[1].write_text(json.dumps(a), encoding="utf-8")
            c = report.read(paths[2]); c.update(gate_diagnostic=report.record(paths[0]), gate_audit=report.record(paths[1])); paths[2].write_text(json.dumps(c), encoding="utf-8")
            a = report.read(paths[3]); a.update(cost=report.record(paths[2]), gate_diagnostic=report.record(paths[0]), gate_audit=report.record(paths[1])); paths[3].write_text(json.dumps(a), encoding="utf-8")
            i = report.read(paths[4]); i.update(fresh_audit=report.record(paths[3]), cost_producer=report.record(paths[2])); paths[4].write_text(json.dumps(i), encoding="utf-8")
    with pytest.raises(ValueError): report.load_verified(*paths)


def test_counts_signed_effect_and_no_native_speed_or_static_floor_claim():
    tables = report.collect_tables(*synthetic_metadata())
    assert {name: len(rows) for name, rows in tables.items()} == {"SCORES": 336, "SIGNED_EFFECT": 168,
        "GATE_CHANNELS": 84, "GATE_CASES": 6, "GATE_METHODS": 42, "WHOLE_COST": 24,
        "WHOLE_COST_SAMPLES": 144, "PHYSICAL_PREFLIGHT": 2, "REGISTERED_ADOPTION": 10, "AUDIT_REPAIR_PROVENANCE": 1}
    assert all(row["gated_minus_ungated_MSE"] == row["gated_MSE"] - row["ungated_MSE"] for row in tables["SIGNED_EFFECT"])
    assert all(not row["native_timed"] and not row["gate_caused_speed_gain_claimed"] for row in tables["WHOLE_COST"])
    assert all(not row["statistical_improvement_claimed"] for row in tables["SIGNED_EFFECT"])
    assert not report.choose(tables["PHYSICAL_PREFLIGHT"], dataset="beijing_aotizhongxin")["gate_output_change_possible_in_timed_prefix"]


def test_report_distinguishes_C_byte_storage_from_logical_bits_and_Beijing_jitter(tmp_path):
    cpu, audit, cost, cost_audit, identity = synthetic_metadata()
    tables = report.collect_tables(cpu, audit, cost, cost_audit, identity)
    text = report.text_report(cpu, cost, cost_audit, identity, tables, {}, tmp_path)
    for phrase in ("bit-packed 전송을 측정한 것이 아니다", "flag 하나당 byte", "gate-eligible 좌표가0개",
                   "signed zero", "dynamic", "탐색적 설계", "비용 실험 재실행이나 모델 재학습은0회"):
        assert phrase in text


def test_four_actual_plot_fixtures_write_png_and_svg(tmp_path):
    paths = report.figures(report.collect_tables(*synthetic_metadata()), tmp_path)
    assert len(paths) == 8 and sum(path.suffix == ".png" for path in paths) == 4
    assert all(path.is_file() and path.stat().st_size > 1000 for path in paths)
