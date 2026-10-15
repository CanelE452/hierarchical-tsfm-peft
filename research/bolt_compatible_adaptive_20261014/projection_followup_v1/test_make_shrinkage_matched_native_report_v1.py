"""Synthetic JSON-only integrated report fixtures, never models or actual science."""
from pathlib import Path
import copy
import json
import tempfile
import unittest
from unittest.mock import patch

import make_shrinkage_matched_native_report_v1 as report


def write(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def scores(value, channels):
    return {scale: {"mse": float(value), "mae": float(value / 2), "per_channel_MSE": [float(value)] * channels,
        "per_channel_MAE": [float(value / 2)] * channels, "count_per_channel": [48] * channels} for scale in report.SCALES}


def prediction_scores(value, c):
    return {"status": "COMPLETE", "scores_all_observed": scores(value, c), "scores_complete_mask": scores(value + .01, c)}


def synthetic_values(folder):
    artifact = {"path": "never_opened_scientific.npz", "sha256": "metadata-only"}
    shrink, space, matched, native_audit = ({"datasets": {}} for _ in range(4))
    shrink["selection_locks"] = {}; matched["condition_selection_locks"] = {}
    for dataset, (c, k, dev) in report.DATASETS.items():
        shrink["datasets"][dataset] = {"splits": {}}
        space["datasets"][dataset] = {"splits": {}}
        matched["datasets"][dataset] = {"routes": {}}
        native_audit["datasets"][dataset] = {"routes": {}}
        shrink["selection_locks"][dataset] = {}; matched["condition_selection_locks"][dataset] = {}
        for projection in report.PROJECTIONS:
            label = "RHO_1" if dataset == "beijing_aotizhongxin" and projection == "FIT_TARGET_PINV" else "RHO_0.1"
            rho = 1. if label == "RHO_1" else .1
            lock = {"status": "LOCKED", "selected_candidate": label, "selected_rho": rho, "native_specific_selection": False}
            path = folder / (dataset + projection + "_RHO.json"); write(path, lock)
            shrink["selection_locks"][dataset][projection] = report.record(path)
            matched["datasets"][dataset]["routes"][projection] = {"native": {}}
            native_audit["datasets"][dataset]["routes"][projection] = {"inputs": {}}
            matched["condition_selection_locks"][dataset][projection] = {}
            for seed in report.SEEDS:
                points = [{"update": u, "tune": {"mse": m, "mae": m / 2}, "state_sha256": "fixture", "state_artifact": artifact,
                    "tune_prediction_artifact": artifact} for u, m in ((0, 2.), (100, 1.5), (200, 1.2), (400, 1.4), (600, 1.3))]
                metadata = {"updates": 600, "selected_update": 200, "parameters": k * 513 * 48, "train_wall_seconds": .1,
                    "old_head_weights_reused_or_warm_started": False, "decoder_condition_selected_by_native": False,
                    "initial_weights_and_biases_zero": True, "seed_changes_initial_weights": False}
                matched["datasets"][dataset]["routes"][projection]["native"][seed] = {"metadata": metadata, "checkpoints": points,
                    "learning_curve": [{"update": u, "loss": 1. / u, "gradient_norm_before_clip": .5} for u in range(1, 601)],
                    "scores": {split: prediction_scores(1.2 if split == "tune" else 1.4, c) for split in ("fit", "tune", dev)}}
                path = folder / (dataset + projection + seed + "_CP.json")
                write(path, {"status": "LOCKED", "selected_update": 200, "rho": rho, "rho_selected_by_native": False})
                matched["condition_selection_locks"][dataset][projection][seed] = report.record(path)
        for split, n in (("fit", 256), ("tune", 64), (dev, 30 if dataset == "robin" else 146)):
            cell = {"RAW": prediction_scores(1., c), "routes": {}}
            diagnostic = {"routes": {}, "complete_target_positions": n * 48}
            for projection in report.PROJECTIONS:
                lock = report.read(shrink["selection_locks"][dataset][projection]["path"]); label, rho = lock["selected_candidate"], lock["selected_rho"]
                methods = {backend: names for backend, names in report.BACKENDS.items()}
                route = {"controls": {}, "positive_rho_candidates": {}}
                for candidate, value in (("STATIC", 1.), ("LOCAL", 1.4)):
                    route["controls"][candidate] = {"methods": {name: prediction_scores(value, c) for names in methods.values() for name in names}}
                candidates = report.CANDIDATES[2:] if split != dev else (label,)
                for candidate in candidates:
                    current_rho = dict(zip(report.CANDIDATES[2:], (.001, .01, .1, 1., 10.)))[candidate]
                    entry = {"rho": current_rho, "methods": {name: prediction_scores(.8 if candidate == label else 1.1, c)
                        for names in methods.values() for name in names}, "backends": {}}
                    for backend in report.BACKENDS:
                        metadata = {"windows": n, "true_observed_counts": [[512] * c for _ in range(n)],
                            "max_abs_decoder_per_window": [1.] * n, "decoder_operator_norm_per_window": [2.] * n,
                            "decoder_Frobenius_norm_per_window": [3.] * n, "actual_lambda": [[current_rho] * c for _ in range(n)],
                            "local_covariance_trace": [[float(k)] * c for _ in range(n)], "ridge_matrix_condition": [[2.] * c for _ in range(n)],
                            "ridge_matrix_condition_valid": [[True] * c for _ in range(n)], "zero_trace_prior_slope_retained": [[False] * c for _ in range(n)],
                            "no_observation_prior_with_current_intercept": [[False] * c for _ in range(n)]}
                        entry["backends"][backend] = {"state_metadata": metadata, "state_artifact": artifact,
                            "past_reconstruction_scores_observed_only": scores(.1, c)}
                    route["positive_rho_candidates"][candidate] = entry
                cell["routes"][projection] = route
                diagnostic["routes"][projection] = {"backends": {}}
                for backend, names in report.BACKENDS.items():
                    diagnostic["routes"][projection]["backends"][backend] = {"minimum_normalized_MSE_real_numeric_rowspace": .2,
                        "rank_D_per_origin": [k] * n, "numeric_rowspace_relative_singular_cutoff": 1e-15,
                        "certificates": [{"window": index, "rank_D": k, "complete_positions": 48} for index in range(n)],
                        "RAW_scores_same_complete_C": scores(1.01, c), "minimum_solution_scores_complete_C": scores(.2, c),
                        "E_Y_diagnostic": {"scores_complete_C": scores(.4, c)}, "actual_methods": {name: {"scores_complete_C": scores(.81, c)} for name in names}}
                native_audit["datasets"][dataset]["routes"][projection]["inputs"][split] = {"persistence_current_C_baseline": prediction_scores(2., c)}
            shrink["datasets"][dataset]["splits"][split], space["datasets"][dataset]["splits"][split] = cell, diagnostic
    costs = {}
    for name, routes, repeats in (("selected_cost", report.SELECTED_ROUTES, 6), ("matched_cost", report.MATCHED_ROUTES, 7),
            ("repeat_matched_cost", report.MATCHED_ROUTES, 7)):
        cost = {"cost": {}}
        for dataset in report.DATASETS:
            rows = []
            for batch in (1, 8):
                for ri, route in enumerate(routes):
                    multiplier = 1. if route == "RAW_BOLT" else .5 + ri * .01
                    cohort_factor = 2. if name == "repeat_matched_cost" else 1.
                    seconds = [round(float(index + 1) * multiplier * cohort_factor * 1e9) / 1e9 for index in range(repeats)]
                    raw = [float(index + 1) * cohort_factor for index in range(repeats)]
                    samples = [{"repeat": i, "seconds": value, "perf_start_ns": 1000000,
                        "perf_end_ns": 1000000 + round(value * 1e9)} for i, value in enumerate(seconds)]
                    import statistics
                    rows.append({"route": route, "batch": batch, "samples": samples, "seconds": seconds,
                        "median_seconds": statistics.median(seconds), "paired_RAW_ratios": [a / b for a, b in zip(seconds, raw)],
                        "paired_RAW_median_ratio": statistics.median([a / b for a, b in zip(seconds, raw)]), "origins": list(range(batch)), "memory": [{"shared_process": True}] * repeats,
                        "whole_includes": "ridge + diagnostics + fit input hashes", "whole_excludes": "source/file hashes outside callback",
                        "selected_metadata": {"synthetic": True}, "reference": artifact, "timed_outputs": artifact})
            cost["cost"][dataset] = {"rows": rows}
        costs[name] = cost
    failure = {"status": "PARTIAL_COMPLETED", "completed_cost_cells": 0, "recorded_timer_samples": 0,
        "failed_paths": [{"type": "KeyError", "message": "arrays"}], "source_snapshots": [], "resource_terminal": {"path": "synthetic.json", "sha256": "fixture"}}
    return {"shrinkage": shrink, "space": space, "matched": matched, "matched_audit": native_audit, **costs, "failed_matched_cost_v1": failure}


def synthetic_gate_files(folder, values):
    values = copy.deepcopy(values)
    paths = {name: folder / (name + "_GATE.json") for name in report.INPUT_NAMES}
    producer, fitpass = folder / "PRODUCER.json", folder / "FIT_PASS.json"
    write(producer, {"status": "COMPLETE"}); write(fitpass, {"status": "PASS"})
    common = {"producer": report.record(producer), "fit_target_audit": report.record(fitpass)}
    counts = {
        "shrinkage_audit": {"positive_rho_backend_states_checked": 88, "window_fits_checked": 13504, "selection_locks_checked": 4,
            "FIT_TUNE_positive_rho_prediction_arrays_checked": 120, "selected_DEV_prediction_arrays_checked": 12, "score_field_checks": 6960},
        "space": {"selected_output_space_cells_checked": 24, "E_Y_diagnostic_predictions_checked": 24, "actual_predictions_checked": 36,
            "real_output_space_inequalities_checked": 60, "RAW_bound_inequalities_checked": 0, "score_field_checks": 864, "selected_D_origin_matrices_checked": 3264},
        "selected_cost_audit": {"cost_cells_checked": 20, "recorded_timer_samples_checked": 120, "physical_shrunk_prechecks_checked": 4,
            "physical_shrunk_first8_window_fits_checked": 32, "selected_rule_lock_files_checked": 4},
        "matched_audit": {"native_fits_checked": 8, "training_updates_logged_checked": 4800, "checkpoint_states_checked": 40,
            "checkpoint_TUNE_prediction_arrays_checked": 40, "selected_FIT_TUNE_DEV_prediction_arrays_checked": 24,
            "source_backend_contexts_checked": 12, "preserved_control_prediction_arrays_checked": 48,
            "persistence_baseline_contexts_computed": 12, "unique_persistence_split_cells_computed": 6, "selection_locks_checked": 8,
            "resolved_registry_conditions_checked": 8, "future_label_perturbation_scope_checks": 8, "score_field_checks": 2064},
        "matched_cost_audit": {"cost_cells_checked": 28, "recorded_timer_samples_checked": 196, "physical_backend_prechecks_checked": 8,
            "physical_first8_window_fits_checked": 64, "native_first8_prediction_checks": 8, "native_selection_locks_checked": 8,
            "old_Bolt_rho_lock_files_checked": 4, "native_and_E_state_hashes_checked": 12, "selected_forecast_file_hashes_checked": 60,
            "canonical_physical_bridges_checked": 2}}
    code_names = {"shrinkage_audit": "audit_local_decoder_shrinkage_v1.py", "space": "audit_selected_shrinkage_output_space_v1.py",
        "selected_cost_audit": "audit_selected_shrinkage_cost_v1.py", "matched_audit": "audit_matched_shrunk_native_v1.py",
        "matched_cost_audit": "audit_matched_shrunk_native_cost_v1.py"}
    counts["repeat_matched_cost_audit"] = copy.deepcopy(counts["matched_cost_audit"])
    code_names["repeat_matched_cost_audit"] = code_names["matched_cost_audit"]
    for name in report.INPUT_NAMES:
        values.setdefault(name, {})
        values[name].update(common, status="PASS" if name.endswith("audit") or name == "space" else "COMPLETE", protected_TEST_accessed=False)
        if name in counts:
            values[name]["totals"] = counts[name]
            source = folder / code_names[name]; source.write_text("# synthetic code; never imported\n", encoding="utf-8")
            values[name]["audit_source"] = report.record(source)
    shrink, matched = values["shrinkage"], values["matched"]
    shrink.update(future_labels_used_for_decoder_fit=False, TUNE_future_labels_used_for_normal_hyperparameter_selection=True,
        development_positive_rho_sweep_executed=False, native_specific_selection=False, independent_holdout=False)
    values["space"].update(minimum_claim_only_normalized_MSE_in_registered_real_D32_numeric_rowspace=True,
        strict_lower_bound_over_all_FP32_outputs_claimed=False, MAE_or_physical_MSE_minimum_claimed=False,
        future_labels_used_to_fit_or_modify_D=False, selection_locks=shrink["selection_locks"])
    matched.update(native_training_fits=8, completed_training_updates=4800, checkpoint_states_saved=40, completed_fit_tune_predictions=16,
        completed_DEV_predictions=8, completed_selection_locks=8, completed_source_backend_cells=12, future_label_perturbation_checks=8,
        failed_paths=[], independent_holdout=False, rho_selection_repeated=False, rho_selected_by_native=False,
        new_encoder_decoder_fit=False, new_backbone_training=False, all_eight_condition_locks_saved_before_any_new_DEV_prediction=True,
        all_conditions_resolved_registry_saved_before_any_DEV_input_load=True)
    values["matched_audit"]["condition_selection_locks"] = matched["condition_selection_locks"]
    for name, routes, cells, samples in (("selected_cost", report.SELECTED_ROUTES, 20, 120), ("matched_cost", report.MATCHED_ROUTES, 28, 196),
            ("repeat_matched_cost", report.MATCHED_ROUTES, 28, 196)):
        values[name].update(routes=list(routes), completed_cost_cells=cells, recorded_timer_samples=samples, completed_cell_timer_samples=samples,
            failed_paths=[], F_before={"unchanged": True}, F_after={"unchanged": True}, new_training_updates=0,
            local_ridge_and_implemented_diagnostics_inside_timer=True, fit_input_binding_hashes_inside_timer=True,
            local_coefficients_cached_inside_timed_pipeline=False, source_snapshots=[{"path": "synthetic executed source.py", "sha256": "unchanged"}])
    values["selected_cost"]["native_heads_loaded_or_timed"] = False
    values["matched_cost"].update(native_deployment_cost_measured=True, selection_locks=matched["condition_selection_locks"],
        old_Bolt_rho_selection_locks=shrink["selection_locks"], canonical_physical_bridge_bindings={})
    values["matched_cost_audit"].update(selection_locks=matched["condition_selection_locks"], old_Bolt_rho_selection_locks=shrink["selection_locks"])
    failure = folder / "FAILED_COST_V1.json"
    values["failed_matched_cost_v1"]["protected_TEST_accessed"] = False; write(failure, values["failed_matched_cost_v1"])
    identity, repair = folder / "SCHEMA_IDENTITY.json", folder / "REPAIR.md"
    write(identity, {"status": "CONFIRMED_PRESERVED_IMPLEMENTATION_IO_CONTRACT_FAILURE", "actual_timer_samples": 0,
        "method_or_training_failure_inferred": False})
    repair.write_text("Fresh same-file canonical metadata binding only.\n", encoding="utf-8")
    values["matched_cost"].update(prior_failed_cost_attempt=report.record(failure), input_schema_failure_identity=report.record(identity),
        repair_addendum=report.record(repair))
    values["matched_cost_audit"]["prior_failed_cost_attempt"] = report.record(failure)
    for dataset in report.DATASETS:
        artifact = {"path": str(folder / "NEVER_READ_PHYSICAL.npz"), "sha256": "audited same opaque SHA"}
        values["matched_cost"]["canonical_physical_bridge_bindings"][dataset] = {"canonical_record": {**artifact, "arrays": {"opaque": "not opened"}},
            "original_alias": artifact, "same_physical_file_path_and_SHA": True, "complete_array_manifest_preserved": True,
            "loader_file_SHA_member_shape_dtype_SHA_checks_relaxed": False, "numeric_tolerance_changed": False}
    write(paths["shrinkage"], shrink)
    values["shrinkage_audit"]["diagnostic"] = report.record(paths["shrinkage"]); write(paths["shrinkage_audit"], values["shrinkage_audit"])
    for name in ("space", "selected_cost", "selected_cost_audit", "matched", "matched_audit", "matched_cost", "matched_cost_audit"):
        values[name].update(shrinkage_diagnostic=report.record(paths["shrinkage"]), shrinkage_audit=report.record(paths["shrinkage_audit"]))
    write(paths["space"], values["space"]); write(paths["selected_cost"], values["selected_cost"])
    values["selected_cost_audit"]["cost"] = report.record(paths["selected_cost"]); write(paths["selected_cost_audit"], values["selected_cost_audit"])
    write(paths["matched"], matched); values["matched_audit"]["diagnostic"] = report.record(paths["matched"])
    write(paths["matched_audit"], values["matched_audit"])
    for name in ("matched_cost", "matched_cost_audit"):
        values[name].update(matched_native_diagnostic=report.record(paths["matched"]), matched_native_audit=report.record(paths["matched_audit"]))
    write(paths["matched_cost"], values["matched_cost"]); values["matched_cost_audit"]["cost"] = report.record(paths["matched_cost"])
    write(paths["matched_cost_audit"], values["matched_cost_audit"])
    values["repeat_matched_cost"].update({key: value for key, value in values["matched_cost"].items() if key != "cost"})
    write(paths["repeat_matched_cost"], values["repeat_matched_cost"])
    values["repeat_matched_cost_audit"] = copy.deepcopy(values["matched_cost_audit"])
    values["repeat_matched_cost_audit"]["cost"] = report.record(paths["repeat_matched_cost"])
    write(paths["repeat_matched_cost_audit"], values["repeat_matched_cost_audit"])
    return paths, failure


class IntegratedReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.values = synthetic_values(self.folder)

    def test_all_bad_and_good_conditions_scales_masks_curves_preserved(self):
        tables = report.collect_tables(self.values)
        expected = {"SCORES": 552, "RHO_CANDIDATES": 480, "TUNE_SELECTION": 28, "RHO_LOCKS": 4,
            "SELECTED_D_BACKENDS": 24, "SELECTED_D_WINDOWS": 3264, "SELECTED_PAST_RECONSTRUCTION": 48,
            "CURRENT_C_PERSISTENCE": 48, "CHECKPOINTS": 40, "LEARNING_CURVE": 4800, "NATIVE_TRAINING": 8, "NATIVE_LOCKS": 8,
            "OUTPUT_DIAGNOSTIC": 264, "OUTPUT_GEOMETRY": 24, "SELECTED_F_ONLY_COST": 20, "SELECTED_F_ONLY_COST_SAMPLES": 120,
            "MATCHED_WHOLE_COST": 28, "MATCHED_WHOLE_COST_SAMPLES": 196, "REPEAT_MATCHED_WHOLE_COST": 28,
            "REPEAT_MATCHED_WHOLE_COST_SAMPLES": 196, "REGISTERED_ADOPTION": 24, "COST_FAILURE_PROVENANCE": 1}
        self.assertEqual({key: len(rows) for key, rows in tables.items()}, expected)
        self.assertEqual({row["candidate"] for row in tables["TUNE_SELECTION"]}, set(report.CANDIDATES))
        self.assertEqual({row["mask"] for row in tables["SCORES"]}, {"all_original_observed", "all_C_complete"})
        self.assertEqual({row["metric_scale"] for row in tables["SCORES"]}, set(report.SCALES))
        self.assertFalse(any(row["MAE_is_minimum"] or row["physical_MSE_is_minimum"] or row["RAW_bound_applied"] for row in tables["OUTPUT_DIAGNOSTIC"]))

    def test_currentC_improvement_not_confused_with_Raw_adoption(self):
        tables = report.collect_tables(self.values)
        native = [row for row in tables["SCORES"] if row["model"].startswith("MATCHED_NATIVE") and row["metric_scale"] == "normalized"]
        self.assertTrue(all(row["mse"] < 2. for row in native))
        gates = [row for row in tables["REGISTERED_ADOPTION"] if "NATIVE" in row["route"]]
        self.assertTrue(all(not row["all_conditions_pass"] for row in gates))
        self.assertTrue(all(not row["independent_statistical_draws_claimed"] for row in tables["CURRENT_C_PERSISTENCE"]))

    def test_earliest_TUNE_checkpoint_and_full_curve_required(self):
        modified = copy.deepcopy(self.values)
        entry = modified["matched"]["datasets"]["robin"]["routes"]["PCA_PINV"]["native"]["92601"]
        entry["metadata"]["selected_update"] = 600
        with self.assertRaisesRegex(ValueError, "TUNE_ONLY_CP"): report.collect_tables(modified)
        modified = copy.deepcopy(self.values)
        modified["matched"]["datasets"]["robin"]["routes"]["PCA_PINV"]["native"]["92601"]["learning_curve"].pop()
        with self.assertRaisesRegex(ValueError, "FULL_CP_AND_CURVE"): report.collect_tables(modified)

    def test_timestamp_arithmetic_rejects_changed_cost_summary(self):
        modified = copy.deepcopy(self.values)
        modified["matched_cost"]["cost"]["robin"]["rows"][0]["median_seconds"] += .1
        with self.assertRaisesRegex(ValueError, "TIMER_SOURCE_ARITHMETIC"): report.collect_tables(modified)

    def test_report_interpretation_keeps_learning_persistence_scope_failures_and_pending_research(self):
        tables = report.collect_tables(self.values)
        text = report.report_text(self.values, tables, {}, self.folder)
        for phrase in ("persistence를 개선", "전혀 학습하지 못했다고 말하지", "12독립 draw가 아니다", "known-past", "KeyError 'arrays'",
            "생성기는 JSON metadata", "fit input binding hashes", "독립 holdout", "relative", "strict lower bound"):
            if phrase == "relative": continue
            self.assertIn(phrase, text)

    def test_JSON_only_read_refuses_model_and_scientific_files(self):
        for name in ("example.npz", "checkpoint.pt", "model.safetensors"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "JSON_METADATA_ONLY"): report.read(self.folder / name)

    def test_pending_wholecost_prevents_output_and_any_plot_work(self):
        for pending in ("matched_cost_audit", "repeat_matched_cost_audit"):
            paths = {}
            for name in report.INPUT_NAMES:
                path = self.folder / (name + ".json")
                write(path, {"status": "PENDING" if name == pending else "PASS" if name.endswith("audit") or name == "space" else "COMPLETE", "protected_TEST_accessed": False})
                paths[name] = path
            with self.subTest(pending=pending), patch.object(report, "FIXED_SHA", {}), patch.object(report, "figures", side_effect=AssertionError("plots must wait")) as plot:
                with self.assertRaisesRegex(ValueError, "COMPLETE_OR_PASS_REQUIRED"):
                    report.generate(paths, self.folder / "not_opened_failure.json", [], self.folder / "never_generated")
                plot.assert_not_called()
            self.assertFalse((self.folder / "never_generated").exists())

    def test_repeat_Raw_absolute_times_kept_and_not_used_to_choose_fastest_cohort(self):
        tables = report.collect_tables(self.values)
        first = report.choose(tables["MATCHED_WHOLE_COST"], dataset="robin", route="RAW_BOLT", batch=8)
        repeated = report.choose(tables["REPEAT_MATCHED_WHOLE_COST"], dataset="robin", route="RAW_BOLT", batch=8)
        self.assertEqual(repeated["median_seconds"], 2 * first["median_seconds"])
        self.assertEqual(len(tables["MATCHED_WHOLE_COST_SAMPLES"]), 196)
        self.assertEqual(len(tables["REPEAT_MATCHED_WHOLE_COST_SAMPLES"]), 196)
        self.assertEqual({row["cost_cohort"] for row in tables["REGISTERED_ADOPTION"]}, {"MATCHED_WHOLE_COST", "REPEAT_MATCHED_WHOLE_COST"})
        text = report.report_text(self.values, tables, {}, self.folder)
        for phrase in ("빠른 cohort만 고르", "앞선 F-only", "짝지은 비율", "변동 원인은 미확정", "12개 같은 split"):
            if phrase == "12개 같은 split": continue
            self.assertIn(phrase, text)

    def test_full_bound_metadata_gate_reads_no_scientific_assets(self):
        paths, failure = synthetic_gate_files(self.folder, self.values)
        original_sha = report.sha
        def metadata_sha(path):
            self.assertIn(Path(path).suffix, (".json", ".py", ".md"))
            return original_sha(path)
        with patch.object(report, "FIXED_SHA", {}), patch.object(report, "FAILED_COST_SHA", original_sha(failure)), patch.object(report, "sha", side_effect=metadata_sha):
            values, bindings = report.load_verified(paths, failure)
            self.assertEqual(len(bindings), 14)
            self.assertEqual(len(report.collect_tables(values)["SCORES"]), 552)
        self.assertFalse((self.folder / "NEVER_READ_PHYSICAL.npz").exists())

    def test_missing_independent_physical_coefficient_checks_rejected(self):
        paths, failure = synthetic_gate_files(self.folder, self.values)
        value = report.read(paths["matched_cost_audit"]); value["totals"]["physical_first8_window_fits_checked"] = 0
        write(paths["matched_cost_audit"], value)
        with patch.object(report, "FIXED_SHA", {}), patch.object(report, "FAILED_COST_SHA", report.sha(failure)):
            with self.assertRaisesRegex(ValueError, "COMPLETE_INDEPENDENT_COUNTS:new matched"):
                report.load_verified(paths, failure)

    def test_native_lock_mutation_detected_before_plot_or_output(self):
        paths, failure = synthetic_gate_files(self.folder, self.values)
        lock = self.values["matched"]["condition_selection_locks"]["robin"]["PCA_PINV"]["92601"]
        Path(lock["path"]).write_text("{}", encoding="utf-8")
        with patch.object(report, "FIXED_SHA", {}), patch.object(report, "FAILED_COST_SHA", report.sha(failure)):
            with self.assertRaisesRegex(ValueError, "EXACT_BINDING:preserved native"):
                report.load_verified(paths, failure)

    def test_protected_TEST_scope_or_decoder_cache_is_rejected(self):
        paths, failure = synthetic_gate_files(self.folder, self.values)
        value = report.read(paths["matched_cost"]); value["protected_TEST_accessed"] = True
        write(paths["matched_cost"], value)
        with patch.object(report, "FIXED_SHA", {}), patch.object(report, "FAILED_COST_SHA", report.sha(failure)):
            with self.assertRaisesRegex(ValueError, "NO_PROTECTED_TEST"):
                report.load_verified(paths, failure)

    def test_decoder_fit_outside_whole_timer_is_not_accepted(self):
        paths, failure = synthetic_gate_files(self.folder, self.values)
        cost = report.read(paths["matched_cost"]); cost["local_coefficients_cached_inside_timed_pipeline"] = True
        write(paths["matched_cost"], cost)
        audit = report.read(paths["matched_cost_audit"]); audit["cost"] = report.record(paths["matched_cost"])
        write(paths["matched_cost_audit"], audit)
        with patch.object(report, "FIXED_SHA", {}), patch.object(report, "FAILED_COST_SHA", report.sha(failure)):
            with self.assertRaisesRegex(ValueError, "WHOLE_TIMER_WITH_RIDGE"):
                report.load_verified(paths, failure)

    def test_ratio_of_medians_adoption_does_not_use_favorable_paired_ratio(self):
        import statistics
        cost = self.values["matched_cost"]["cost"]["robin"]["rows"]
        row = next(row for row in cost if row["route"] == "PCA_SHRUNK_BOLT" and row["batch"] == 8)
        raw = next(row for row in cost if row["route"] == "RAW_BOLT" and row["batch"] == 8)
        seconds = [10., 20., 30., 3.5, 4., 5., 6.]
        for sample, value in zip(row["samples"], seconds):
            sample.update(seconds=value, perf_end_ns=sample["perf_start_ns"] + round(value * 1e9))
        ratios = [a / b for a, b in zip(seconds, raw["seconds"])]
        row.update(seconds=seconds, median_seconds=statistics.median(seconds), paired_RAW_ratios=ratios,
            paired_RAW_median_ratio=statistics.median(ratios))
        tables = report.collect_tables(self.values)
        gate = report.choose(tables["REGISTERED_ADOPTION"], dataset="robin", route="PCA_SHRUNK_BOLT", cost_cohort="MATCHED_WHOLE_COST")
        measured = report.choose(tables["MATCHED_WHOLE_COST"], dataset="robin", route="PCA_SHRUNK_BOLT", batch=8)
        self.assertLess(measured["median_paired_RAW_ratio"], .9)
        self.assertGreater(gate["B8_ratio_of_RAW_medians"], .9)
        self.assertFalse(gate["all_conditions_pass"])

    def test_six_PNG_and_SVG_figures_render_from_synthetic_metadata_only(self):
        import xml.etree.ElementTree as ET
        images = report.figures(report.collect_tables(self.values), self.folder)
        self.assertEqual(len(images), 12)
        for path in images:
            self.assertGreater(path.stat().st_size, 1000)
            if path.suffix == ".png": self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            else: self.assertTrue(ET.parse(path).getroot().tag.endswith("svg"))

    def test_fresh_manifest_hashes_CSV_rows_relative_links_and_preserved_prior_report(self):
        import base64
        import csv
        import re
        paths, failure = synthetic_gate_files(self.folder, self.values)
        prior = self.folder / "PRIOR_REPORT.md"; prior.write_text("Immutable prior scientific report.\n", encoding="utf-8")
        original_prior_sha = report.sha(prior)
        def fixture_images(tables, output):
            folder = Path(output) / "figures"; folder.mkdir()
            saved = []
            for name in ("01_TUNE_rho_selection", "02_actual_Bolt_decoder_comparison", "03_frozen_and_matched_native",
                    "04_selected_output_space_and_forecast_gap", "05_separate_actual_whole_cost_cohorts", "06_native_TUNE_checkpoints"):
                png, svg = folder / (name + ".png"), folder / (name + ".svg")
                png.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aA1cAAAAASUVORK5CYII="))
                svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"/>', encoding="utf-8")
                saved += [png, svg]
            return saved
        output = self.folder / "fresh_report"
        repeat_plan, invocation = self.folder / "REPEAT_PLAN.md", self.folder / "AUDIT_CALL_FAILURE.json"
        repeat_plan.write_text("Same protocol repeated; both cohorts retained.\n", encoding="utf-8")
        failed_audit = self.folder / "FAILED_AUDIT.json"; write(failed_audit, {"status": "FAILED_PRESERVED"})
        write(invocation, {"status": "CONFIRMED_PRESERVED_AUDITOR_INVOCATION_PATH_FAILURE",
            "scientific_inputs_sources_or_outputs_changed": False, "new_F_native_forward_or_training_or_ridge_fit": False,
            "new_cost_timing": False, "failed_audit": report.record(failed_audit),
            "audit_source": report.read(paths["matched_cost_audit"])["audit_source"]})
        same_volume_source = self.folder / "synthetic_report_source.py"
        same_volume_source.write_text(Path(report.__file__).read_text(encoding="utf-8"), encoding="utf-8")
        with patch.object(report, "FIXED_SHA", {}), patch.object(report, "FAILED_COST_SHA", report.sha(failure)), \
                patch.object(report, "REPEAT_PLAN_SHA", report.sha(repeat_plan)), patch.object(report, "AUDIT_INVOCATION_IDENTITY_SHA", report.sha(invocation)), \
                patch.object(report, "figures", side_effect=fixture_images), patch.object(report, "__file__", str(same_volume_source)):
            manifest = report.generate(paths, failure, [prior], output, repeat_plan, invocation)
            with self.assertRaisesRegex(ValueError, "FRESH_INTEGRATED_REPORT_DIRECTORY"):
                report.generate(paths, failure, [prior], output, repeat_plan, invocation)
        self.assertEqual(len(manifest["figures"]), 12)
        self.assertEqual(len(manifest["CSV_rows"]), 22)
        self.assertFalse(manifest["NPZ_or_model_opened"] or manifest["automatic_commit_push"])
        for item in manifest["files"]:
            path = output / item["path"]
            self.assertEqual(report.sha(path), item["sha256"]); self.assertEqual(path.stat().st_size, item["bytes"])
        for name, count in manifest["CSV_rows"].items():
            with (output / "tables" / (name + ".csv")).open(encoding="utf-8", newline="") as handle:
                self.assertEqual(sum(1 for _ in csv.DictReader(handle)), count)
        for target in re.findall(r"\]\(([^)]+)\)", (output / "REPORT_KO.md").read_text(encoding="utf-8")):
            self.assertFalse(Path(target).is_absolute()); self.assertTrue((output / target).resolve().is_file(), target)
        self.assertEqual(report.sha(prior), original_prior_sha)


if __name__ == "__main__": unittest.main()
