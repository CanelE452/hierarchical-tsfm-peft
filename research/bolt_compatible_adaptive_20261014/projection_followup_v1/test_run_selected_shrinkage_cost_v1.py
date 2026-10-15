"""Selected-lock gates and pure synthetic shrinkage deployment parity fixtures."""
from pathlib import Path
import copy
import inspect
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import run_selected_shrinkage_cost_v1 as run
from local_decoder_shrinkage_v1 import fit_shrunk_local_decoder
from projection_math_v1 import ProjectionState


def write(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        folder = Path(self.temporary.name)
        parents = {}
        for key in run.PARENT_KEYS:
            path = folder / (key + ".json"); write(path, {"synthetic": True}); parents[key] = run.record(path)
        source = folder / "local_decoder_shrinkage_v1.py"; source.write_text("synthetic sealed source", encoding="utf-8")
        shrink = {"status": "COMPLETE", **parents, "source_records": [run.record(source)], "datasets": {}, "selection_locks": {},
            "completed_positive_rho_backend_batches": 88, "completed_window_matrix_fits": 13504, "completed_channel_slots": 187664,
            "completed_observed_channel_fits": 187664, "completed_ridge_solver_calls": 187664,
            "completed_zero_trace_prior_slope_fallbacks": 0, "completed_no_observation_fallbacks": 0,
            "native_seed_D_reuses": 6752, "completed_FIT_TUNE_positive_rho_method_predictions": 120,
            "completed_selected_DEV_method_predictions": 12, "completed_reused_control_method_pairs": 36,
            "completed_RAW_control_cells": 6, "completed_selection_locks": 4, "new_E_F_native_head_forward_runs": 0, "failed_paths": [],
            **{key: False for key in ("new_E_F_or_native_head_training", "new_backbone_training", "new_GPU_execution", "protected_TEST_accessed",
                "future_labels_used_for_decoder_fit", "native_specific_selection", "development_positive_rho_sweep_executed",
                "whole_selected_deployment_cost_measured", "same_decoder_side_information_as_static", "strict_K_only_compression",
                "old_static_or_global_rankK_floor_reused", "independent_holdout")},
            **{key: True for key in ("TUNE_future_labels_used_for_normal_hyperparameter_selection", "native_uses_same_Bolt_selected_candidate",
                "all_selection_lock_files_saved_before_any_DEV_candidate_compute", "fit_uses_TUNE_DEV_known_past_at_origin", "original_source_cache_model_and_selection_SHA_unchanged")}}
        audit = {"status": "PASS", **parents, "selection_locks": {}, "totals": dict(run.AUDIT_COUNTS), "protected_TEST_accessed": False,
            "future_labels_used_for_decoder_fit": False}
        self.producer = {"datasets": {}}
        for dataset, (c, k, dev) in run.DATASETS.items():
            source = {"K": k, "inputs": {split: {"path": split + ".npz", "sha256": "metadata-only"} for split in ("fit", "tune", dev)}}
            row = {"K": k, "channels": c, "development_split": dev, "splits": {split: {"status": "COMPLETE", "input_source": source["inputs"][split], "routes": {}} for split in ("fit", "tune", dev)}}
            shrink["selection_locks"][dataset], audit["selection_locks"][dataset] = {}, {}
            for projection in run.PROJECTIONS:
                label, rho = "RHO_0.1", .1
                scores = {name: float(index + 1) for index, name in enumerate(run.CANDIDATE_ORDER)}
                scores[label] = .5
                lock = {"status": "LOCKED", "dataset": dataset, "route": projection, "selected_candidate": label, "selected_rho": rho,
                    "selection_predictor": "BOLT_NUMPY", "selection_split": "TUNE", "selection_metric": "normalized_macro_channel_MSE",
                    "selection_mask": "all_true_observed", "MAE_used_for_selection": False, "native_specific_selection": False,
                    "development_or_oracle_scores_used_for_selection": False, "native_uses_same_Bolt_selected_candidate": True,
                    "candidate_tie_order": list(run.CANDIDATE_ORDER), "TUNE_Bolt_normalized_MSE_by_candidate": scores,
                    "selected_TUNE_Bolt_normalized_MSE": .5}
                path = folder / (dataset + projection + "_LOCK.json"); write(path, lock); item = run.record(path)
                shrink["selection_locks"][dataset][projection] = item
                audit["selection_locks"][dataset][projection] = {"record": item, "selected_candidate": label, "selected_rho": rho,
                    "recomputed_winner_equal_registered_winner": True}
                artifact = {"path": "never_opened_scientific.npz", "sha256": "metadata-only"}
                branch = {"status": "COMPLETE", "state_metadata": {"rho": rho}, "artifact": artifact}
                row["splits"][dev]["routes"][projection] = {"status": "COMPLETE", "selection_lock": item, "selected_candidate": label,
                    "unselected_positive_rho_DEV_candidates_computed": False, "positive_rho_candidates": {label: {"backends": {"BOLT_NUMPY": branch}}},
                    "selected_methods": {"BOLT": {"artifact": artifact, "prediction_member": "BOLT_prediction"}}}
            self.producer["datasets"][dataset], shrink["datasets"][dataset] = source, row
        self.result, self.audit = folder / "SHRINK.json", folder / "AUDIT.json"
        write(self.result, shrink)
        code = folder / "audit_local_decoder_shrinkage_v1.py"; code.write_text("independent synthetic source", encoding="utf-8")
        audit.update(diagnostic=run.record(self.result), audit_source=run.record(code)); write(self.audit, audit)
        self.sha_patch = patch.multiple(run, SHRINK_SHA=run.sha(self.result), SHRINK_AUDIT_SHA=run.sha(self.audit))
        self.sha_patch.start(); self.addCleanup(self.sha_patch.stop)
        self.parent_patch = patch("run_local_decoder_shrinkage_v1.load_gates", return_value=({}, self.producer, parents))
        self.parent_patch.start(); self.addCleanup(self.parent_patch.stop)

    def reseal(self):
        audit = run.read(self.audit); audit["diagnostic"] = run.record(self.result); write(self.audit, audit)
        run.SHRINK_SHA, run.SHRINK_AUDIT_SHA = run.sha(self.result), run.sha(self.audit)

    def test_all_exact_passes_sources_and_four_locks_before_any_arrays(self):
        with patch.object(np, "load", side_effect=AssertionError("no actual arrays")):
            source, shrink, audit, locks = run.load_gates(self.result, self.audit)
        self.assertEqual(source, self.producer)
        self.assertEqual(sum(len(row) for row in locks.values()), 4)
        self.assertTrue(all(item["rho"] == .1 for row in locks.values() for item in row.values()))

    def test_pending_and_stale_science_gates_precede_scientific_import(self):
        value = run.read(self.audit); value["status"] = "PENDING"; write(self.audit, value)
        calls, original = [], __import__
        def forbid(name, *args, **kwargs):
            if name.split(".")[0] in ("numpy", "torch", "local_decoder_shrinkage_v1", "continuation_runtime_v1"):
                calls.append(name); raise AssertionError("science before accepted locks")
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=forbid), self.assertRaisesRegex(ValueError, "BEFORE_SCIENCE"):
            run.execute("never_started", self.result, self.audit)
        self.assertEqual(calls, [])

    def test_exact_result_source_and_lock_SHA_changes_are_rejected(self):
        for path in (self.result, Path(run.read(self.result)["source_records"][0]["path"]), Path(run.read(self.result)["selection_locks"]["robin"]["PCA_PINV"]["path"])):
            old = path.read_bytes(); path.write_bytes(old + b"\n")
            with self.subTest(path=path.name), self.assertRaises(ValueError): run.load_gates(self.result, self.audit)
            path.write_bytes(old)

    def test_every_registered_audit_count_is_required(self):
        original = run.read(self.audit)
        for key in run.AUDIT_COUNTS:
            value = copy.deepcopy(original); value["totals"][key] -= 1; write(self.audit, value)
            run.SHRINK_AUDIT_SHA = run.sha(self.audit)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "SHRINKAGE_AUDIT_COUNTS"):
                run.load_gates(self.result, self.audit)
        write(self.audit, original); run.SHRINK_AUDIT_SHA = run.sha(self.audit)

    def test_TEST_native_specific_future_fit_and_unlocked_DEV_sweep_rejected(self):
        original = run.read(self.result)
        for key in ("protected_TEST_accessed", "future_labels_used_for_decoder_fit", "native_specific_selection", "development_positive_rho_sweep_executed"):
            value = copy.deepcopy(original); value[key] = True; write(self.result, value); self.reseal()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "SEALED_SHRINKAGE_SCOPE"):
                run.load_gates(self.result, self.audit)
        write(self.result, original); self.reseal()

    def test_missing_fourth_lock_and_modified_rho_do_not_trigger_model_or_selection(self):
        shrink, audit = run.read(self.result), run.read(self.audit)
        del shrink["selection_locks"]["robin"]["FIT_TARGET_PINV"]
        with self.assertRaisesRegex(ValueError, "TWO_LOCKED_PROJECTION_RULES"):
            run.selected_locks(shrink, audit)
        shrink = run.read(self.result)
        binding = shrink["selection_locks"]["robin"]["PCA_PINV"]
        path = Path(binding["path"]); value = run.read(path); value["selected_rho"] = 10.; write(path, value)
        changed = run.record(path); shrink["selection_locks"]["robin"]["PCA_PINV"] = changed
        audit["selection_locks"]["robin"]["PCA_PINV"].update(record=changed, selected_rho=10.)
        with self.assertRaisesRegex(ValueError, "REGISTERED_POSITIVE_SELECTED_RHO"):
            run.selected_locks(shrink, audit)


class CallbackFixtures(unittest.TestCase):
    def setUp(self):
        s = np.tile(np.array([-1., 1.], np.float32), 256)
        t = np.tile(np.array([-1., -1., 1., 1.], np.float32), 128)
        self.history = np.tile(np.stack((s, t, s + 2 * t), axis=1)[None], (8, 1, 1))
        self.mask = np.ones(self.history.shape, bool)
        self.state = ProjectionState(np.zeros(3), np.array([[1., 0.], [0., 1.], [0., 0.]]), np.array([[.5, 0., .5], [0., .5, 1.]]), np.zeros(3), {})
        self.z = self.state.encode(self.history)
        self.fitted = fit_shrunk_local_decoder(self.history, self.mask, self.z, self.state.decoder.astype(np.float32), .1)
        self.values = {"history": self.history, "history_observed": self.mask, "raw_last": self.history[:, -1], "origins": np.arange(512, 520, dtype=np.int64)}

    def precheck(self, metadata=None, arrays=None):
        return run.physical_shrunk_precheck(self.history, self.mask, np.zeros(3, np.float32), np.ones(3, np.float32),
            self.values, self.state, {"latent_history": self.z}, self.fitted.to_npz() if arrays is None else arrays,
            self.fitted.metadata() if metadata is None else metadata, .1)

    def test_full_physical_prefix_coefficient_lambda_trace_and_six_decisions(self):
        checked, arrays = self.precheck()
        self.assertEqual(checked["status"], "PASS")
        self.assertEqual(len(checked["exact_decisions"]), 6)
        self.assertTrue(all(checked["exact_decisions"].values()))
        np.testing.assert_array_equal(arrays["prior_decoder_FP32"], self.state.decoder.astype(np.float32))

    def test_lambda_numeric_or_zerotrace_decision_mismatch_cannot_hide_in_output_parity(self):
        metadata = copy.deepcopy(self.fitted.metadata()); metadata["actual_lambda"][0][0] += .1
        checked, _ = self.precheck(metadata=metadata)
        self.assertFalse(checked["checks"]["actual_lambda"]["pass"])
        metadata = copy.deepcopy(self.fitted.metadata()); metadata["zero_trace_prior_slope_retained"][0][0] = True
        checked, _ = self.precheck(metadata=metadata)
        self.assertFalse(checked["exact_decisions"]["zero_trace_prior_slope_retained"])
        self.assertEqual(checked["status"], "FAILED_PRECHECK_PRESERVED")

    def test_deployment_prior_cast_rho_and_trace_rule_are_exact(self):
        metadata = copy.deepcopy(self.fitted.metadata()); metadata["fit_input_bindings"]["prior_decoder"]["sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "STORED_STATIC_PRIOR_DEPLOYMENT_CAST"): self.precheck(metadata=metadata)
        metadata = copy.deepcopy(self.fitted.metadata()); metadata["rho"] = 1.
        with self.assertRaisesRegex(ValueError, "LOCKED_RHO_TRACE_LAMBDA"): self.precheck(metadata=metadata)

    def test_one_ULP_past_change_switches_zero_trace_and_positive_lambda_despite_input_parity(self):
        import whole_selected_shrinkage_cost_v1 as wc
        history = np.ones((8, 512, 3), np.float32)
        observed = np.ones(history.shape, bool)
        latent = self.state.encode(history)
        original = fit_shrunk_local_decoder(history, observed, latent, self.state.decoder.astype(np.float32), .1)
        values = {"history": history, "history_observed": observed, "raw_last": history[:, -1], "origins": self.values["origins"]}
        altered = history.copy(); altered[:, 0, 0] = np.nextafter(np.float32(1), np.float32(2))
        self.assertTrue(wc.parity(altered, history)["pass"])
        checked, _ = run.physical_shrunk_precheck(altered, observed, np.zeros(3, np.float32), np.ones(3, np.float32),
            values, self.state, {"latent_history": latent}, original.to_npz(), original.metadata(), .1)
        self.assertTrue(checked["checks"]["normalized_history"]["pass"])
        self.assertTrue(checked["checks"]["encoded_past"]["pass"])
        self.assertFalse(checked["exact_decisions"]["zero_trace_prior_slope_retained"])
        self.assertFalse(checked["exact_decisions"]["positive_lambda_solver_decision"])
        self.assertEqual(checked["status"], "FAILED_PRECHECK_PRESERVED")

    def test_factory_refits_with_locked_rho_prior_mask_every_call(self):
        calls = []
        def fitted(history, observed, z, prior, rho):
            calls.append((observed.copy(), prior.copy(), rho))
            return fit_shrunk_local_decoder(history, observed, z, prior, rho)
        def F(z): return np.repeat(z[:, -1:] + np.float32(.2), 48, axis=1)
        with patch("local_decoder_shrinkage_v1.fit_shrunk_local_decoder", side_effect=fitted):
            callback = run.shrunk_predictor(self.state, F, .1)
            a, b = callback(self.history, self.mask), callback(self.history, self.mask)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(item[2] == .1 for item in calls))
        np.testing.assert_array_equal(calls[0][0], self.mask)
        np.testing.assert_array_equal(calls[0][1], self.state.decoder.astype(np.float32))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(set(inspect.signature(callback).parameters), {"history", "history_observed"})

    def test_static_control_keeps_same_original_matmul_order(self):
        def F(z): return np.repeat(z[:, -1:] + np.float32(.2), 48, axis=1)
        expected = self.state.forecast(self.history[:, -1], F(self.z), self.z[:, -1])
        np.testing.assert_array_equal(run.static_predictor(self.state, F)(self.history, self.mask), expected)


if __name__ == "__main__": unittest.main()
