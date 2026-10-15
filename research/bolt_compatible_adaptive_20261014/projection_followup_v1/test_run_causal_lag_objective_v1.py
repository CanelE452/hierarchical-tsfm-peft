"""Synthetic gates, seal-before-evaluation routing and a complete tiny 24/72 bundle."""
from pathlib import Path
import ast
import builtins
import copy
import inspect
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import run_causal_lag_objective_v1 as run


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, allow_nan=False)
    return run.record(path)


def rewrite(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return run.record(path)


def gate_fixture(folder):
    science = {"status": "COMPLETE", "failed_paths": [], "protected_TEST_accessed": False,
        "saved_Bolt_split_predictions": 18, "F_before": {"opaque_synthetic": True}, "F_after": {"opaque_synthetic": True},
        "all_twelve_locks_before_any_DEV_input_or_prediction": True, "independent_holdout": False, "new_backbone_training": False}
    source_path = folder / "SCIENCE.json"; source = write_new(source_path, science)
    code = folder / "audit_raw_subset_comparison_v3.py"; code.write_text("# synthetic metadata-only source\n", encoding="utf-8")
    audit_path = folder / "SCIENCE_PASS.json"
    audit = write_new(audit_path, {"status": "PASS", "diagnostic": source, "totals": run.SCIENCE_COUNTS,
        "protected_TEST_accessed": False, "audit_source": run.record(code)})
    report_dir = folder / "report"; report_dir.mkdir()
    report = report_dir / "REPORT_KO.md"; report.write_text("Synthetic report.\n", encoding="utf-8")
    files = []
    for i in range(32):
        path = report_dir / (str(i) + ".txt"); path.write_text(str(i), encoding="utf-8")
        files.append({**run.record(path), "path": path.name, "bytes": path.stat().st_size})
    manifest = write_new(report_dir / "MANIFEST.json", {"status": "COMPLETE", "files": files,
        "inputs": {"science": source, "audit": audit}})
    verifier = folder / "audit_raw_subset_report_v1.py"; verifier.write_text("# synthetic verifier identity\n", encoding="utf-8")
    report_audit_path = folder / "REPORT_PASS.json"
    write_new(report_audit_path, {"status": "PASS", "protected_TEST_accessed": False, "report": run.record(report),
        "manifest": manifest, "audit_source": run.record(verifier), "inputs": {"science": source, "audit": audit},
        "files_checked": 32, "total_CSV_primitive_fields_checked": 171848, "local_links_checked": 47, "PNG_count": 6, "SVG_count": 6})
    return source_path, audit_path, report_audit_path


def gate_patch(folder, paths):
    report_audit = run.read(paths[2])
    return patch.multiple(run, ROOT=folder, SCIENCE_SHA=run.record(paths[0])["sha256"],
        SCIENCE_AUDIT_SHA=run.record(paths[1])["sha256"], REPORT_AUDIT_SHA=run.record(paths[2])["sha256"],
        REPORT_SHA=report_audit["report"]["sha256"], REPORT_MANIFEST_SHA=report_audit["manifest"]["sha256"],
        SCIENCE_AUDITOR_SHA=run.read(paths[1])["audit_source"]["sha256"], REPORT_AUDITOR_SHA=report_audit["audit_source"]["sha256"])


def empty_report():
    return {"datasets": {}, "failed_paths": [], "decoder_fits_completed": 0, "resolved_fit_lock_files": 0,
        "cached_D_predictions_completed": 0, "joint_output_diagnostics_completed": 0,
        "EY_diagnostics_completed": 0, "flat_diagnostics_completed": 0, "all_24_locks_before_TUNE_DEV_NPZ": False}


def small_case():
    history = np.zeros((1, 512, 2), np.float32)
    current = np.zeros((1, 2), np.float32)
    future = np.ones((1, 3, 2), np.float32)
    target = future.copy(); observed = np.ones_like(target, dtype=np.bool_)
    values = {"target": target, "observed": observed, "raw_last": current, "origins": np.array([512], np.int64),
        "mean": np.zeros(2, np.float32), "std": np.ones(2, np.float32), "raw_forecast": target.copy()}
    return {"values": values, "z": history, "future": future, "encode": lambda x: x.copy()}, {
        "lags": np.array([0], np.int64), "decoder_FP32": np.eye(2, dtype=np.float32)}


def build_full_fixture(folder):
    """Write only tiny generated arrays through the real fit/evaluate/save orchestration."""
    import causal_lag_decoder_v1 as math
    import fixed_decoder_span_v1 as io
    from projection_math_v1 import ProjectionState
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    paths = gate_fixture(folder)
    output = folder / "tiny_run"; output.mkdir()
    data = {"robin": (3, 1, "val", 1), "beijing_aotizhongxin": (3, 1, "eval", 1)}
    contexts, parent = {}, run.read(paths[0]); parent["datasets"] = {}
    rng = np.random.default_rng(426)
    for dataset, (c, k, dev, _) in data.items():
        parent["datasets"][dataset] = {"spaces": {}}
        values_by_split = {}
        for split, n in (("fit", 2), ("tune", 1), (dev, 1)):
            history = np.ascontiguousarray(rng.normal(size=(n, 512, c)), dtype=np.float32)
            observed_history = np.ones_like(history, dtype=np.bool_); observed_history[0, 35, 1] = False
            current = history[:, -1].copy()
            target = np.ascontiguousarray(current[:, None] + .1 * rng.normal(size=(n, 48, c)), dtype=np.float32)
            observed = np.ones_like(target, dtype=np.bool_); observed[0, 1, 2] = False
            values_by_split[split] = {"history": history, "history_observed": observed_history, "target": target,
                "observed": observed, "raw_last": current, "origins": np.arange(n, dtype=np.int64) * 48 + 512,
                "mean": np.zeros(c, np.float32), "std": np.ones(c, np.float32), "raw_forecast": current[:, None] + np.zeros_like(target)}
        for index, space in enumerate(run.SPACES):
            e = np.eye(c, dtype=np.float64)[:, index:index+1]
            state = ProjectionState(np.zeros(c), e, e.T.copy(), np.zeros(c), {"channels": c, "K": k})
            indices = [index] if space == "RAW_SUBSET" else None
            def encode(x, e=e, indices=indices):
                return np.ascontiguousarray(x[..., indices] if indices is not None else x @ e.astype(np.float32), dtype=np.float32)
            e_source = io.save_npz(folder / (dataset+space+"_E.npz"), {"center": state.center, "encoder": state.encoder,
                "selected_indices": np.array(indices if indices is not None else [], np.int64)})
            condition = {"status": "COMPLETE", "E_state_artifact": e_source, "inputs": {}, "Bolt": {}}
            if indices is None: condition["original_E_state"] = io.save_npz(folder/(dataset+space+"_ORIGINAL_E.npz"), state.to_npz())
            parent["datasets"][dataset]["spaces"][space] = condition
            for split, values in values_by_split.items():
                z = encode(values["history"])
                future = np.ascontiguousarray(z[:, -1:] + .05 * rng.normal(size=(len(z), 48, k)), dtype=np.float32)
                inputs = io.save_npz(folder/(dataset+space+split+"_INPUT.npz"), {**values, "latent_history_BOLT_NUMPY": z})
                cached = io.save_npz(folder/(dataset+space+split+"_BOLT.npz"), {"latent_history": z,
                    "latent_future": future, "prediction": values["raw_last"][:, None] + (future-z[:, -1:]) @ e.T.astype(np.float32), "origins": values["origins"]})
                condition["inputs"][split], condition["Bolt"][split] = inputs, {"artifact": cached}
                contexts[(dataset, space, split)] = {"values": values, "z": z, "future": future, "encode": encode,
                    "input_source": inputs, "cached_Bolt_source": cached, "E_source": e_source}
    source_binding = rewrite(paths[0], parent)
    audit = run.read(paths[1]); audit["diagnostic"] = source_binding; audit_binding = rewrite(paths[1], audit)
    manifest_path = folder / "report/MANIFEST.json"; manifest = run.read(manifest_path)
    manifest["inputs"] = {"science": source_binding, "audit": audit_binding}; manifest_binding = rewrite(manifest_path, manifest)
    report_audit = run.read(paths[2]); report_audit["inputs"] = manifest["inputs"]; report_audit["manifest"] = manifest_binding
    report_audit_binding = rewrite(paths[2], report_audit)
    report = empty_report()
    opened = []
    def load(dataset, space, split):
        if split != "fit":
            if not report["all_24_locks_before_TUNE_DEV_NPZ"]: raise AssertionError("FUTURE_ARRAY_OPEN_BEFORE_SEAL")
            if len(run.read(report["fit_registry"]["path"])["conditions"]) != 24: raise AssertionError("INCOMPLETE_REGISTRY")
        opened.append((dataset, space, split)); return contexts[(dataset, space, split)]
    def fit(context, lags, objective):
        values = context["values"]
        return math.fit_reconstruction(values["history"], values["history_observed"], context["z"], lags) if objective == "RECON" else math.fit_forecast(
            context["z"], context["future"], values["raw_last"], values["target"], values["observed"], lags)
    with patch.multiple(run, ROOT=folder, DATASETS=data, SPLIT_COUNTS={"fit": 2, "tune": 1}):
        states = run.fit_all_conditions(report, output, load, fit, io.save_npz, write_new, lambda event: None)
        run.evaluate_all_conditions(report, states, load, run.evaluate_case, io.save_npz, output, lambda event: None)
    report.update(status="COMPLETE" if report["cached_D_predictions_completed"] == 72 and not report["failed_paths"] else "PARTIAL_COMPLETED",
        raw_subset_diagnostic=source_binding, raw_subset_audit=audit_binding, parent_report=report_audit["report"],
        parent_report_manifest=manifest_binding, parent_report_audit=report_audit_binding,
        protected_TEST_accessed=False, independent_holdout=False, new_F_calls=0, new_backbone_training=False,
        new_native_head_training=0, TUNE_DEV_used_for_fit_or_selection=False, oracle_used_for_fit_or_selection=False,
        whole_deployment_cost_measured=False, actual_F_cache_splits_reused=18,
        fixture_scope="Generated small CPU arrays only; not scientific results", fixture_open_order=opened,
        protected_prior_records=[], input_records=[], source_records=[], source_snapshots=[])
    write_new(output / "RESULT.json", report)
    return {"result": output / "RESULT.json", "paths": paths, "datasets": data, "contexts": contexts}


class CausalLagControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.folder = Path(self.temp.name)

    def test_complete_gate_is_json_and_opaque_asset_only(self):
        paths = gate_fixture(self.folder)
        with gate_patch(self.folder, paths): source, bindings = run.load_gates(*paths)
        self.assertEqual(source["status"], "COMPLETE"); self.assertEqual(len(bindings["parent_report_file_records"]), 32)
        self.assertEqual(list(self.folder.rglob("*.npz")), [])

    def test_partial_science_and_missing_audit_coverage_rejected(self):
        paths = gate_fixture(self.folder)
        value = run.read(paths[0]); value["status"] = "PARTIAL_COMPLETED"; rewrite(paths[0], value)
        with gate_patch(self.folder, paths), self.assertRaisesRegex(ValueError, "COMPLETE_SCIENCE"): run.load_gates(*paths)
        value["status"] = "COMPLETE"; rewrite(paths[0], value)
        audit = run.read(paths[1]); audit["diagnostic"] = run.record(paths[0]); audit["totals"] = {}; rewrite(paths[1], audit)
        with gate_patch(self.folder, paths), self.assertRaisesRegex(ValueError, "AUDIT_COVERAGE"): run.load_gates(*paths)

    def test_report_PASS_must_bind_exact_same_science(self):
        paths = gate_fixture(self.folder); value = run.read(paths[2]); value["inputs"]["science"]["sha256"] = "0" * 64; rewrite(paths[2], value)
        with gate_patch(self.folder, paths), self.assertRaisesRegex(ValueError, "IMMUTABLE_FILE_SHA"): run.load_gates(*paths)

    def test_report_file_mutation_is_rejected(self):
        paths = gate_fixture(self.folder); (self.folder / "report/4.txt").write_text("changed", encoding="utf-8")
        with gate_patch(self.folder, paths), self.assertRaisesRegex(ValueError, "IMMUTABLE_FILE_SHA"): run.load_gates(*paths)

    def test_protected_TEST_or_wrong_report_counts_are_rejected(self):
        paths = gate_fixture(self.folder)
        for change, message in (({"protected_TEST_accessed": True}, "PROTECTED_TEST"), ({"files_checked": 31}, "REPORT_VERIFICATION")):
            value = run.read(paths[2]); original = copy.deepcopy(value); value.update(change); rewrite(paths[2], value)
            with gate_patch(self.folder, paths), self.assertRaisesRegex(ValueError, message): run.load_gates(*paths)
            rewrite(paths[2], original)

    def test_pending_gates_stop_before_science_import_or_output_creation(self):
        imported = []; original = builtins.__import__
        def track(name, *args, **kwargs):
            if name in ("numpy", "causal_lag_decoder_v1", "fixed_decoder_span_v1", "prepare_inputs", "continuation_runtime_v1"): imported.append(name)
            return original(name, *args, **kwargs)
        with patch.object(run, "HERE", self.folder), patch("builtins.__import__", side_effect=track):
            with self.assertRaises(FileNotFoundError): run.execute("fresh", self.folder/"absent.json", self.folder/"no.json", self.folder/"pending.json")
        self.assertEqual(imported, []); self.assertFalse((self.folder / "runs/fresh").exists())

    def test_fresh_output_rejects_existing_and_escape(self):
        (self.folder / "runs").mkdir(); (self.folder / "runs/old").mkdir()
        with patch.object(run, "HERE", self.folder):
            for tag in ("old", "../escape", ".", ""):
                with self.subTest(tag=tag), self.assertRaises(ValueError): run.fresh_output(tag)

    def test_failed_fit_resolves_its_lock_and_preserves_other_23(self):
        report = empty_report(); output = self.folder / "run"; output.mkdir(); seen = []
        context = {"input_source": {"synthetic": True}, "E_source": {"synthetic": True}, "cached_Bolt_source": {"synthetic": True}}
        def fitter(ctx, lags, objective):
            seen.append((lags, objective))
            if len(seen) == 2: raise ValueError("TOY_RANK_FAILURE")
            return {**{name: np.zeros(1) for name in ("decoder_FP64", "decoder_FP32", "bias_FP64", "bias_FP32", "lags")}, "metadata": {}}
        def saver(path, arrays): path.write_bytes(b"synthetic state"); return run.record(path)
        with patch.object(run, "ROOT", self.folder):
            states = run.fit_all_conditions(report, output, lambda *args: context, fitter, saver, write_new, lambda event: None)
        self.assertEqual(len(states), 23); self.assertEqual(report["resolved_fit_lock_files"], 24)
        registry = run.read(report["fit_registry"]["path"])
        self.assertEqual(registry["failed_fit_paths"], 1); self.assertEqual(len(registry["conditions"]), 24)
        self.assertTrue(report["all_24_locks_before_TUNE_DEV_NPZ"])

    def test_evaluation_cannot_start_without_full_registry(self):
        calls = []
        with self.assertRaisesRegex(ValueError, "ALL_24_FIT_LOCKS"):
            run.evaluate_all_conditions(empty_report(), {}, lambda *args: calls.append(args), None, None, self.folder, lambda event: None)
        self.assertEqual(calls, [])

    def test_main_source_has_no_F_model_training_or_timer_path(self):
        tree = ast.parse(inspect.getsource(run)); calls = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
        self.assertFalse(any(name in calls for name in ("FrozenBolt", "torch.load", "fit_native", "measure_cost")))
        execute = inspect.getsource(run.execute)
        self.assertLess(execute.index("fit_all_conditions("), execute.index("prior_file_records(source)"))
        self.assertLess(execute.index("prior_file_records(source)"), execute.index("evaluate_all_conditions("))

    def test_joint_diagnostic_failure_keeps_primary_prediction_and_flat_EY(self):
        import causal_lag_decoder_v1 as math
        context, fitted = small_case()
        with patch.object(math, "masked_joint_minimum", side_effect=ValueError("TOY_DIAGNOSTIC_ONLY_FAILURE")):
            summary, arrays = run.evaluate_case(context, fitted)
        self.assertEqual(summary["status"], "COMPLETE_PREDICTION_WITH_DIAGNOSTIC_FAILURES")
        self.assertEqual(summary["diagnostics"]["joint_floor"]["status"], "FAILED_PATH_PRESERVED")
        self.assertEqual(summary["scores_all_observed"]["normalized"]["mse"], 0.)
        self.assertTrue(all(name+"_prediction_FP32" in arrays for name in ("ACTUAL_F", "EY", "FLAT")))

    def test_discarded_weak_direction_is_reported_without_false_floor_failure(self):
        context, fitted = small_case()
        context["future"][:] = np.array([0., 1e16], np.float32)
        context["values"]["target"][:] = np.array([0., 1.], np.float32)
        fitted["decoder_FP32"] = np.diag(np.array([1., 1e-16], np.float32))
        summary, arrays = run.evaluate_case(context, fitted)
        actual = summary["diagnostics"]["methods"]["ACTUAL_F"]
        self.assertEqual(summary["status"], "COMPLETE")
        self.assertFalse(actual["real_trace_bound_inequality_pass"])
        self.assertFalse(actual["FP32_bound_with_measured_allowance_pass"])
        self.assertTrue(actual["retained_output_trace_bound_inequality_pass"])
        self.assertGreater(actual["discarded_numeric_map_output_squared_norm"], 1.)
        self.assertFalse(summary["diagnostics"]["joint_floor"]["all_real_latent_forecasts_satisfy_registered_bound_claimed"])
        self.assertIn("ACTUAL_F_discarded_joint_component_FP64", arrays)

    def test_changed_scoring_targets_preserve_fixed_cache_actual_forward(self):
        context, fitted = small_case(); _, first = run.evaluate_case(context, fitted)
        context["values"]["target"] += np.float32(7.)
        context["values"]["observed"][0, 1, 0] = False
        summary, second = run.evaluate_case(context, fitted)
        self.assertTrue(run.byte_equal(first["ACTUAL_F_prediction_FP32"], second["ACTUAL_F_prediction_FP32"]))
        self.assertEqual(summary["cached_future_label_perturbation"]["status"], "PASS")
        self.assertFalse(run.byte_equal(first["EY_future"], second["EY_future"]))

    def test_full_tiny_real_helpers_generate_24_fits_72_predictions_after_seal(self):
        bundle = build_full_fixture(self.folder / "bundle"); result = run.read(bundle["result"])
        self.assertEqual(result["status"], "COMPLETE", result["failed_paths"])
        self.assertEqual(result["decoder_fits_completed"], 24); self.assertEqual(result["cached_D_predictions_completed"], 72)
        self.assertEqual(result["joint_output_diagnostics_completed"], 72)
        self.assertEqual(len(list(bundle["result"].parent.glob("*_FIT_LOCK.json"))), 24)
        self.assertEqual(len(list(bundle["result"].parent.glob("*_PREDICTION_DIAGNOSTIC.npz"))), 72)
        self.assertEqual(result["new_F_calls"], 0)
        case = result["datasets"]["robin"]["spaces"]["PCA"]["conditions"]["L7_FORECAST"]["splits"]["fit"]
        self.assertEqual(case["cached_future_label_perturbation"]["status"], "PASS")
        self.assertEqual(case["EY_incomplete_C_horizons_filled_with_currentC"], 1)
        self.assertTrue(case["diagnostics"]["joint_floor"]["all_HK_future_variables_retained"])
        for entry in case["diagnostics"]["methods"].values(): self.assertTrue(entry["real_trace_bound_inequality_pass"])


if __name__ == "__main__": unittest.main()
