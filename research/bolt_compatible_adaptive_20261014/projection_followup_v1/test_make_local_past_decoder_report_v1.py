"""Full-cohort synthetic LocalD metadata; no actual matrices/models/fit are read."""
import copy
import json

import pytest

import make_local_past_decoder_report_v1 as report
from test_make_decoder_metric_report_v1 import score, synthetic_metadata as old_metadata


def synthetic_metadata():
    cpu = {"datasets": {}}; audit = {"datasets": {}}; cost = {"cost": {}}
    _, templates = old_metadata()
    binding = {"path": "not_opened.npz", "sha256": "a" * 64}
    for dataset, (c, k, dev) in report.DATASETS.items():
        cpu["datasets"][dataset] = {"channels": c, "K": k, "splits": {}}
        audit["datasets"][dataset] = {"splits": {}}
        for split, n in (("fit", 256), ("tune", 64), (dev, 30 if dataset == "robin" else 146)):
            cell = {"RAW_scores_all_observed": score(.8, c), "RAW_scores_complete_mask": score(.9, c), "routes": {}}
            audit_cell = {"routes": {}}
            for route in report.PROJECTIONS:
                cell["routes"][route] = {"backends": {}}; audit_cell["routes"][route] = {"backends": {}}
                for backend, methods in report.BACKENDS.items():
                    meta = {"windows": n, "singular_relative_cut": 1e-4, "future_labels_used_in_local_decoder_fit": False,
                        "TUNE_or_DEV_threshold_selection": False,
                        "max_abs_decoder_per_window": [1. + index / 10 for index in range(n)],
                        "decoder_Frobenius_norm_per_window": [2.] * n, "decoder_operator_norm_per_window": [1.] * n,
                        "true_observed_counts": [[512] * c for _ in range(n)], "centered_slope_ranks": [[k] * c for _ in range(n)],
                        "affine_model_ranks_with_analytic_intercept": [[k + 1] * c for _ in range(n)],
                        "rank_deficient_slope_channels": [[False] * c for _ in range(n)], "observed_count_below_two": [[False] * c for _ in range(n)],
                        "no_observation_current_C_fallback": [[False] * c for _ in range(n)],
                        "retained_condition_number": [[2.] * c for _ in range(n)], "retained_condition_number_valid": [[True] * c for _ in range(n)],
                        "observed_reconstruction_MSE_real_fit": [[.1] * c for _ in range(n)], "observed_reconstruction_MSE_valid": [[True] * c for _ in range(n)],
                        "singularvalues_observed_centered_Z": [[[1. / (index + 1) for index in range(k)] for _ in range(c)] for _ in range(n)]}
                    values = {method: {"static_scores_all_observed": score(.5, c), "local_scores_all_observed": score(.6, c),
                        "static_scores_complete_mask": score(.55, c), "local_scores_complete_mask": score(.65, c),
                        "static_D_einsum_arithmetic_replay": {"max_abs_vs_original": 1e-8, "byte_exact": False,
                            "scores_all_observed": score(.5, c)}} for method in methods}
                    branch = {"local_fit_metadata": meta, "local_state_artifact": binding, "original_state_source": binding,
                        "cached_method_sources": {method: binding for method in methods}, "native_cached_past_byte_exact_and_D_shared": backend == "NATIVE_TORCH32",
                        "artifact": binding, "methods": values, "past_reconstruction_deployment_scores_observed_only": score(.1, c)}
                    cell["routes"][route]["backends"][backend] = branch
                    verified_methods = {method: {"local_scores_complete_mask": score(.65, c), "measured_FP32_rounding_MSE_allowance": 1e-8,
                        "local_real_minus_new_origin_specific_minimum_normalized_MSE": .45,
                        "local_actual_minus_new_origin_specific_minimum_normalized_MSE": .45,
                        "new_local_real_arithmetic_scores_complete_mask": score(.65, c)} for method in methods}
                    diag = {"minimum_only_normalized_MSE_in_each_known_D32_real_rowspace": True, "old_static_or_global_rank_K_minimum_reused": False,
                        "scores": score(.2, c), "RAW_scores_same_complete_mask": score(.9, c), "rank_D_per_origin": [k] * n,
                        "certificate_per_origin": [{"fixture_window": index, "pass": True} for index in range(n)]}
                    audit_cell["routes"][route]["backends"][backend] = {"methods": verified_methods, "new_per_origin_output_space_diagnostic": diag}
            cpu["datasets"][dataset]["splits"][split] = cell; audit["datasets"][dataset]["splits"][split] = audit_cell
        rows = copy.deepcopy(templates["cost"][dataset]["rows"])
        old_routes = ("RAW_BOLT", "PCA_PAST_LS_BOLT", "FORECA_PAST_LS_BOLT", "PCA_ENCODER_PINV_BOLT", "FORECA_ENCODER_PINV_BOLT")
        for row in rows: row["route"] = report.ROUTES[old_routes.index(row["route"])]
        cost["cost"][dataset] = {"rows": rows}
    cpu.update(completed_method_pairs=36, completed_backend_batches=24, completed_window_matrix_fits=3264,
               completed_channel_solver_calls=44304, completed_no_observation_fallbacks=0)
    return cpu, audit, cost, {}


def gate_fixture(tmp_path):
    def write(name, value):
        path = tmp_path / name; path.write_text(json.dumps(value), encoding="utf-8"); return path
    bindings = {key: report.record(write(key + ".json", {"fixture": key})) for key in
        ("producer", "fit_target_audit", "past_constant_diagnostic", "past_constant_audit", "past_constant_whole_cost", "past_constant_whole_cost_audit")}
    cpu, audit, cost, cost_audit = synthetic_metadata()
    cpu.update(status="COMPLETE", **bindings, failed_paths=[], new_E_F_native_head_forward_runs=0,
        new_E_F_or_native_head_training=False, future_labels_used_for_local_fit=False, method_checkpoint_threshold_selection_used=False,
        inference_time_past_decoder_refitting=True, fit_uses_TUNE_DEV_known_past=True, same_decoder_side_information_as_static=False,
        past_C_condition_metadata_added=True, static_D_or_global_rankK_oracle_reused_as_local_lower_bound=False,
        protected_TEST_accessed=False, original_source_cache_selection_SHA_unchanged=True, independent_holdout=False)
    cpu_path = write("CPU.json", cpu)
    audit.update(status="PASS", diagnostic=report.record(cpu_path), **bindings, protected_TEST_accessed=False,
        totals={"dataset_split_cells_checked": 6, "backend_states_checked": 24, "completed_backend_batches_checked": 24,
        "method_pairs_checked": 36, "window_matrix_fits_checked": 3264, "channel_slots_checked": 44304, "native_seed_D_reuses_checked": 1632,
        "static_prediction_arrays_byte_preserved": 36, "local_prediction_arrays_checked": 36, "static_einsum_control_arrays_checked": 36,
        "cached_future_arrays_byte_preserved": 36, "score_field_checks": 2592, "new_per_origin_output_diagnostic_backend_cells_checked": 24,
        "new_per_origin_output_inequalities_checked": 36, "RAW_bound_inequalities_checked": 0})
    audit_path = write("CPU_AUDIT.json", audit)
    cost.update(status="COMPLETE", producer=bindings["producer"], fit_target_audit=bindings["fit_target_audit"],
        local_diagnostic=report.record(cpu_path), local_audit=report.record(audit_path), new_training_updates=0, new_backbone_training=False,
        F_before={"hash": "frozen"}, F_after={"hash": "frozen"}, routes=list(report.ROUTES), protected_TEST_accessed=False,
        completed_cost_cells=20, recorded_timer_samples=120, local_LS_and_implemented_diagnostics_inside_timer=True,
        native_heads_loaded_or_timed=False, local_coefficients_cached_inside_timed_pipeline=False)
    cost_path = write("COST.json", cost)
    cost_audit.update(status="PASS", cost=report.record(cost_path), producer=bindings["producer"], fit_target_audit=bindings["fit_target_audit"],
        local_diagnostic=report.record(cpu_path), local_audit=report.record(audit_path), protected_TEST_accessed=False,
        totals={"cost_cells_checked": 20, "recorded_timer_samples_checked": 120, "physical_local_prechecks_checked": 4,
            "physical_local_first8_window_fits_checked": 32, "selected_forecast_file_hashes_checked": 36, "canonical_physical_bridges_checked": 2})
    return cpu_path, audit_path, cost_path, write("COST_AUDIT.json", cost_audit)


def test_complete_exact_metadata_gate(tmp_path):
    values = report.load_verified(*gate_fixture(tmp_path))
    assert values[0]["completed_channel_solver_calls"] == 44304


@pytest.mark.parametrize("change", ["pending_cost", "stale_CPU", "missing_new_floor", "future_fit", "same_interface", "old_floor", "missing_timer", "cached_D", "excluded_solver_diagnostics", "native_timed"])
def test_wrong_or_unfinished_scope_rejects_report_generation(tmp_path, change):
    paths = gate_fixture(tmp_path)
    if change == "pending_cost":
        value = report.read(paths[3]); value["status"] = "PENDING"; paths[3].write_text(json.dumps(value), encoding="utf-8")
    elif change == "stale_CPU": paths[0].write_text(paths[0].read_text() + " ", encoding="utf-8")
    elif change in ("missing_new_floor", "missing_timer"):
        index, key = (1, "new_per_origin_output_diagnostic_backend_cells_checked") if change == "missing_new_floor" else (3, "recorded_timer_samples_checked")
        value = report.read(paths[index]); value["totals"][key] -= 1; paths[index].write_text(json.dumps(value), encoding="utf-8")
        if index == 1:
            cost = report.read(paths[2]); cost["local_audit"] = report.record(paths[1]); paths[2].write_text(json.dumps(cost), encoding="utf-8")
            ca = report.read(paths[3]); ca.update(cost=report.record(paths[2]), local_audit=report.record(paths[1])); paths[3].write_text(json.dumps(ca), encoding="utf-8")
    elif change in ("cached_D", "excluded_solver_diagnostics", "native_timed"):
        cost = report.read(paths[2])
        key = {"cached_D": "local_coefficients_cached_inside_timed_pipeline", "excluded_solver_diagnostics": "local_LS_and_implemented_diagnostics_inside_timer", "native_timed": "native_heads_loaded_or_timed"}[change]
        cost[key] = change != "excluded_solver_diagnostics"; paths[2].write_text(json.dumps(cost), encoding="utf-8")
        ca = report.read(paths[3]); ca["cost"] = report.record(paths[2]); paths[3].write_text(json.dumps(ca), encoding="utf-8")
    else:
        cpu = report.read(paths[0]); key = {"future_fit": "future_labels_used_for_local_fit", "same_interface": "same_decoder_side_information_as_static", "old_floor": "static_D_or_global_rankK_oracle_reused_as_local_lower_bound"}[change]
        cpu[key] = True; paths[0].write_text(json.dumps(cpu), encoding="utf-8")
        a = report.read(paths[1]); a["diagnostic"] = report.record(paths[0]); paths[1].write_text(json.dumps(a), encoding="utf-8")
        c = report.read(paths[2]); c.update(local_diagnostic=report.record(paths[0]), local_audit=report.record(paths[1])); paths[2].write_text(json.dumps(c), encoding="utf-8")
        ca = report.read(paths[3]); ca.update(cost=report.record(paths[2]), local_diagnostic=report.record(paths[0]), local_audit=report.record(paths[1])); paths[3].write_text(json.dumps(ca), encoding="utf-8")
    with pytest.raises(ValueError): report.load_verified(*paths)


def test_full_cohort_and_minimum_scope(tmp_path):
    tables = report.collect_tables(*synthetic_metadata())
    assert {name: len(rows) for name, rows in tables.items()} == {"COMPARISON": 312, "SIGNED_EFFECT": 144,
        "STATIC_EINSUM_CONTROL": 36, "PAST_RECONSTRUCTION": 48, "LOCAL_BACKEND_METADATA": 24,
        "LOCAL_WINDOW_GEOMETRY": 3264, "LOCAL_CHANNEL_SOLVERS": 44304, "NEW_PER_ORIGIN_DIAGNOSTIC": 168,
        "NEW_PER_ORIGIN_GEOMETRY": 24, "WHOLE_COST": 20, "WHOLE_COST_SAMPLES": 120, "REGISTERED_ADOPTION": 8}
    assert all(row["this_MSE_is_a_known_per_origin_D_minimum"] == (row["method"] == "LOCAL_D_MINIMUM" and row["metric_scale"] == "normalized")
               and not row["MAE_is_minimum"] and not row["physical_MSE_is_minimum"] and not row["RAW_bound_applied"]
               for row in tables["NEW_PER_ORIGIN_DIAGNOSTIC"])
    assert all(not row["native_timed"] for row in tables["WHOLE_COST"])


def test_scope_and_remaining_failure_are_explicit(tmp_path):
    values = synthetic_metadata(); tables = report.collect_tables(*values)
    text = report.text_report(values[0], values[1], values[2], tables, {}, tmp_path)
    for phrase in ("unrestricted full-design least squares", "상대 cutoff", "signed-zero", "native head는 원 static D",
                   "새로운 backbone/표현 학습", "각 이미 fitting된 D32[n]", "과거 전용 local fitting", "독립 holdout",
                   "input-array byte hash", "state-array hash", "quiet/exclusive benchmark"):
        assert phrase in text


def test_five_png_svg_pairs_from_synthetic_metadata(tmp_path):
    files = report.figures(report.collect_tables(*synthetic_metadata()), tmp_path)
    assert len(files) == 10 and sum(path.suffix == ".png" for path in files) == 5
    assert all(path.is_file() and path.stat().st_size > 1000 for path in files)
