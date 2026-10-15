"""Synthetic source gates and physical/local-coefficient parity, never real assets."""
from pathlib import Path
import copy
import inspect
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import run_local_past_decoder_cost_v1 as run
from local_past_decoder_v1 import fit_local_past_decoder
from projection_math_v1 import ProjectionState


def write(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        folder = Path(self.temporary.name)
        paths = {}
        for key in run.PARENT_KEYS:
            path = folder / (key + ".json")
            write(path, {"status": "PASS" if "audit" in key else "COMPLETE", "synthetic": True})
            paths[key] = run.record(path)
        self.bindings = paths
        self.producer = {"datasets": {}}
        local = {"status": "COMPLETE", **paths, "datasets": {}, "failed_paths": [],
            "completed_method_pairs": 36, "completed_backend_batches": 24, "completed_window_matrix_fits": 3264,
            "native_seed_D_reuses": 1632, "completed_channel_solver_calls": 44304,
            "completed_no_observation_fallbacks": 0, "protected_TEST_accessed": False,
            "new_E_F_native_head_forward_runs": 0, "new_E_F_or_native_head_training": False,
            "new_backbone_training": False, "future_labels_used_for_local_fit": False,
            "method_checkpoint_threshold_selection_used": False, "original_source_cache_selection_SHA_unchanged": True,
            "same_decoder_side_information_as_static": False, "past_C_condition_metadata_added": True,
            "strict_K_only_compression": False, "static_D_or_global_rankK_oracle_reused_as_local_lower_bound": False}
        for dataset, (c, k, dev) in run.DATASETS.items():
            source = {"K": k, "inputs": {}, "projections": {}}
            row = {"K": k, "channels": c, "splits": {}}
            for projection in run.PROJECTIONS:
                source["projections"][projection] = {"state_artifact": {"path": "synthetic_state.npz", "sha256": "metadata-only"}, "bolt": {"artifacts": {}}}
            for split in ("fit", "tune", dev):
                source["inputs"][split] = {"path": split + ".npz", "sha256": "metadata-only"}
                cell = {"status": "COMPLETE", "input_source": source["inputs"][split], "routes": {}}
                for projection in run.PROJECTIONS:
                    cache = {"path": projection + split + ".npz", "sha256": "metadata-only"}
                    source["projections"][projection]["bolt"]["artifacts"][split] = cache
                    cell["routes"][projection] = {"backends": {"BOLT_NUMPY": {"status": "COMPLETE",
                        "original_state_source": source["projections"][projection]["state_artifact"], "cached_method_sources": {"BOLT": cache}}}}
                row["splits"][split] = cell
            self.producer["datasets"][dataset], local["datasets"][dataset] = source, row
        self.local = folder / "LOCAL.json"
        self.audit = folder / "LOCAL_AUDIT.json"
        write(self.local, local)
        code = folder / "audit_local_past_decoder_probe_v1.py"
        code.write_text("independent synthetic source", encoding="utf-8")
        write(self.audit, {"status": "PASS", **paths, "diagnostic": run.record(self.local), "audit_source": run.record(code),
            "protected_TEST_accessed": False, "totals": {**run.LOCAL_AUDIT_COUNTS,
                "observed_channel_solver_calls_checked": 44304, "no_observation_fallbacks_checked": 0}})
        self.predecessor = patch("run_local_past_decoder_probe_v1.load_gates", return_value=(self.producer, {}, {}, self.bindings))
        self.predecessor.start()
        self.addCleanup(self.predecessor.stop)

    def update_local(self, change):
        value = run.read(self.local)
        change(value)
        write(self.local, value)
        audit = run.read(self.audit)
        audit["diagnostic"] = run.record(self.local)
        write(self.audit, audit)

    def test_exact_complete_PASS_accepts_without_scientific_array_or_model_load(self):
        with patch.object(np, "load", side_effect=AssertionError("real assets forbidden")):
            producer, local, audit = run.load_gates(self.local, self.audit)
        self.assertEqual(local["completed_method_pairs"], 36)
        self.assertEqual(producer, self.producer)
        self.assertEqual(audit["status"], "PASS")

    def test_pending_audit_stops_before_scientific_import_and_fresh_output(self):
        value = run.read(self.audit)
        value["status"] = "PENDING"
        write(self.audit, value)
        imports, real_import = [], __import__
        def forbid(name, *args, **kwargs):
            if name.split(".")[0] in ("numpy", "torch", "continuation_runtime_v1", "whole_local_past_decoder_cost_v1"):
                imports.append(name)
                raise AssertionError("scientific import before PASS")
            return real_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=forbid):
            with self.assertRaisesRegex(ValueError, "COMPLETE_LOCAL_PRODUCER_AND_INDEPENDENT_PASS"):
                run.execute("never_started", self.local, self.audit)
        self.assertEqual(imports, [])

    def test_each_required_audit_count_cannot_be_partial(self):
        original = run.read(self.audit)
        for key in run.LOCAL_AUDIT_COUNTS:
            changed = copy.deepcopy(original)
            changed["totals"][key] += 1
            write(self.audit, changed)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "FULL_LOCAL_PAST_DECODER"):
                run.load_gates(self.local, self.audit)
        write(self.audit, original)

    def test_local_or_predecessor_or_auditor_SHA_mutation_is_rejected(self):
        for path in (self.local, run.resolved(self.bindings["producer"]), Path(run.read(self.audit)["audit_source"]["path"])):
            before = path.read_bytes()
            path.write_bytes(before + b"\n")
            with self.subTest(path=path.name), self.assertRaisesRegex(ValueError, "EXACT_SOURCE_BINDING"):
                run.load_gates(self.local, self.audit)
            path.write_bytes(before)

    def test_TEST_futurefit_and_false_same_information_claims_rejected(self):
        for key, value in (("protected_TEST_accessed", True), ("future_labels_used_for_local_fit", True),
                           ("method_checkpoint_threshold_selection_used", True), ("same_decoder_side_information_as_static", True)):
            before = run.read(self.local)
            audit_before = run.read(self.audit)
            self.update_local(lambda v: v.update({key: value}))
            with self.subTest(key=key), self.assertRaises(ValueError):
                run.load_gates(self.local, self.audit)
            write(self.local, before)
            write(self.audit, audit_before)

    def test_channel_slots_are_not_hardcoded_as_observed_solvers(self):
        self.update_local(lambda v: v.update(completed_channel_solver_calls=44303, completed_no_observation_fallbacks=1))
        audit = run.read(self.audit)
        audit["totals"].update(observed_channel_solver_calls_checked=44303, no_observation_fallbacks_checked=1)
        write(self.audit, audit)
        self.assertEqual(run.load_gates(self.local, self.audit)[1]["completed_no_observation_fallbacks"], 1)
        audit["totals"]["observed_channel_solver_calls_checked"] += 1
        write(self.audit, audit)
        with self.assertRaisesRegex(ValueError, "SEPARATE_FROM_CHANNEL_SLOTS"):
            run.load_gates(self.local, self.audit)

    def test_local_input_or_numpy_backend_state_mismatch_rejected(self):
        self.update_local(lambda v: v["datasets"]["robin"]["splits"]["val"].update(input_source={"path": "different.npz", "sha256": "different"}))
        with self.assertRaisesRegex(ValueError, "SAME_LOCAL_KNOWN_PAST_ORIGINS"):
            run.load_gates(self.local, self.audit)


class CallbackAndPrecheckTests(unittest.TestCase):
    def setUp(self):
        s = np.tile(np.array([-1., 1.], np.float32), 256)
        t = np.tile(np.array([-1., -1., 1., 1.], np.float32), 128)
        z = np.tile(np.stack((s, t), axis=1)[None], (8, 1, 1))
        self.history = np.concatenate((z, (z[:, :, :1] + 2 * z[:, :, 1:])), axis=2)
        self.mask = np.ones(self.history.shape, bool)
        self.mask[:, :20, 2] = False
        self.state = ProjectionState(np.zeros(3), np.array([[1., 0.], [0., 1.], [0., 0.]]),
            np.array([[1., 0., 1.], [0., 1., 2.]]), np.zeros(3), {})
        self.z = self.state.encode(self.history)
        self.local = fit_local_past_decoder(self.history, self.mask, self.z)
        self.values = {"history": self.history, "history_observed": self.mask,
            "raw_last": self.history[:, -1], "origins": np.arange(512, 520, dtype=np.int64)}

    def precheck(self, metadata=None, arrays=None, physical=None):
        return run.physical_local_precheck(self.history if physical is None else physical, self.mask,
            np.zeros(3, np.float32), np.ones(3, np.float32), self.values, self.state,
            {"latent_history": self.z}, self.local.to_npz() if arrays is None else arrays,
            self.local.metadata() if metadata is None else metadata)

    def test_actual_physical_prefix_ff_controls_match_solver_reference(self):
        import whole_local_past_decoder_cost_v1 as wc
        filled = wc.preprocess(self.history, self.mask, np.zeros(3, np.float32), np.ones(3, np.float32))
        self.values["history"] = filled
        self.z = self.state.encode(filled)
        self.local = fit_local_past_decoder(filled, self.mask, self.z)
        metadata, arrays = self.precheck()
        self.assertEqual(metadata["status"], "PASS")
        self.assertTrue(all(metadata["exact_decisions"].values()))
        self.assertEqual(arrays["decoder_deployed_FP32"].shape, (8, 2, 3))

    def test_rank_decision_mismatch_is_not_hidden_by_prediction_tolerance(self):
        metadata = self.local.metadata()
        metadata["centered_slope_ranks"][0][0] = 1
        check, _ = self.precheck(metadata=metadata)
        self.assertEqual(check["status"], "FAILED_PRECHECK_PRESERVED")
        self.assertFalse(check["exact_decisions"]["centered_slope_ranks"])

    def test_decoder_coefficient_mismatch_with_same_rank_is_rejected(self):
        arrays = self.local.to_npz()
        arrays["decoder_deployed_FP32"][0, 0, 0] += .1
        check, _ = self.precheck(arrays=arrays)
        self.assertFalse(check["checks"]["decoder_FP32"]["pass"])
        self.assertEqual(check["status"], "FAILED_PRECHECK_PRESERVED")

    def test_cutoff_mismatch_cannot_be_accepted_by_output_parity(self):
        metadata = self.local.metadata()
        metadata["singular_relative_cut"] = 1e-8
        with self.assertRaisesRegex(ValueError, "EXACT_PREDECLARED_LOCAL_SINGULAR_CUTOFF"):
            self.precheck(metadata=metadata)

    def test_one_ULP_normalized_past_change_crosses_singular_cutoff_despite_input_parity(self):
        import whole_local_past_decoder_cost_v1 as wc
        s = np.tile(np.array([-1., 1.], np.float32), 256)
        t = np.tile(np.array([-1., -1., 1., 1.], np.float32), 128)
        weak = np.float32(1e-4)
        history = np.tile(np.stack((s, weak * t, t), axis=1)[None], (8, 1, 1))
        observed = np.ones(history.shape, bool)
        latent = self.state.encode(history)
        original = fit_local_past_decoder(history, observed, latent)
        changed = history.copy()
        changed[:, :, 1] = np.nextafter(weak, np.float32(np.inf)) * t
        values = {"history": history, "history_observed": observed, "raw_last": history[:, -1], "origins": self.values["origins"]}
        self.assertTrue(wc.parity(changed, history)["pass"])
        checked, _ = run.physical_local_precheck(changed, observed, np.zeros(3, np.float32), np.ones(3, np.float32),
            values, self.state, {"latent_history": latent}, original.to_npz(), original.metadata())
        self.assertTrue(checked["checks"]["normalized_history"]["pass"])
        self.assertTrue(checked["checks"]["encoded_past"]["pass"])
        self.assertFalse(checked["exact_decisions"]["centered_slope_ranks"])
        self.assertEqual(checked["status"], "FAILED_PRECHECK_PRESERVED")

    def test_static_callback_exactly_uses_original_projection_contract(self):
        def F(z):
            return np.repeat(z[:, -1:] + np.float32(.3), 48, axis=1)
        callback = run.static_predictor(self.state, F)
        expected = self.state.forecast(self.history[:, -1], F(self.z), self.z[:, -1])
        np.testing.assert_array_equal(callback(self.history, self.mask), expected)
        self.assertEqual(set(inspect.signature(callback).parameters), {"history", "history_observed"})

    def test_local_callback_refits_every_call_uses_mask_and_never_futurelabels(self):
        import local_past_decoder_v1 as math
        calls = []
        def fit(x, mask, z):
            calls.append((x.copy(), mask.copy(), z.copy()))
            return fit_local_past_decoder(x, mask, z)
        def F(z):
            return np.repeat(z[:, -1:] + np.float32(.3), 48, axis=1)
        callback = run.local_predictor(self.state, F)
        # Factory binds the fit function once; patch during factory creation.
        with patch.object(math, "fit_local_past_decoder", side_effect=fit):
            callback = run.local_predictor(self.state, F)
            a = callback(self.history, self.mask)
            b = callback(self.history, self.mask)
        self.assertEqual(len(calls), 2)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(calls[0][1], self.mask)
        self.assertEqual(set(inspect.signature(callback).parameters), {"history", "history_observed"})


if __name__ == "__main__":
    unittest.main()
