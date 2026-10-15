"""Source gates and frozen-cache helper fixtures; no actual assets or CLI run."""
from pathlib import Path
import copy
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import run_local_past_decoder_probe_v1 as run
from local_past_decoder_v1 import fit_local_past_decoder


FIT_COUNTS = {"projection_states_checked": 4, "native_fits_checked": 8, "checkpoint_states_checked": 40,
    "score_field_checks": 1464, "saved_prediction_arrays_checked": 88, "cost_cells_checked": 20,
    "recorded_timer_samples_checked": 120, "target_projection_fits_checked": 2,
    "reused_PCA_controls_checked": 2, "output_subspace_diagnostic_cells_checked": 12}
CPU_COUNTS = {"dataset_split_cells_checked": 6, "method_pairs_checked": 42,
    "ungated_prediction_arrays_checked": 42, "gated_prediction_arrays_checked": 42, "score_field_checks": 2016}
COST_COUNTS = {"cost_cells_checked": 24, "recorded_timer_samples_checked": 144, "physical_gate_prechecks_checked": 2,
    "paired_gate_timed_output_arrays_checked": 72, "original_selected_forecast_hashes_checked": 36}


def write(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.paths = tuple(root / name for name in ("producer.json", "fit_pass.json", "cpu.json", "cpu_pass.json", "cost.json", "cost_pass.json"))
        fit_code, cost_code = root / "audit_fit_target_comparison_v2.py", root / "audit_past_constant_gate_cost_v2.py"
        fit_code.write_text("synthetic independent FIT source", encoding="utf-8")
        cost_code.write_text("synthetic independent cost source", encoding="utf-8")
        producer = {"status": "COMPLETE", "protected_TEST_accessed": False, "new_backbone_training": False,
            "native_training_fits": 8, "projection_fits": 2, "reused_projection_states": 2,
            "projection_conditions": 4, "failed_paths": [], "F_before": {"state": "frozen"}, "F_after": {"state": "frozen"},
            "prior_source_results_caches_and_selection_locks_unchanged": True, "datasets": {}}
        cpu = {"status": "COMPLETE", "protected_TEST_accessed": False, "completed_dataset_split_cells": 6,
            "completed_method_pairs": 42, "failed_paths": [], "new_training": False,
            "new_encoder_forecaster_or_head_forward": False, "E_F_native_head_forward_runs": 0,
            "future_labels_used_for_gate": False, "method_selection_used": False,
            "past_C_condition_metadata_added": True, "same_decoder_side_information_as_ungated": False,
            "strict_K_only_compression": False, "static_original_fixed_D_floor_reused_as_gated_lower_bound": False,
            "source_cache_selection_SHA_unchanged": True, "datasets": {}}
        for dataset, (channels, k, dev, _) in run.DATASETS.items():
            input_record = {"path": "synthetic_past_metadata_only.npz", "sha256": "no-actual-array-IO"}
            producer["datasets"][dataset] = {"status": "COMPLETE", "K": k,
                "inputs": {dev: input_record}, "projections": {route: {} for route in run.ROUTES}}
            cpu["datasets"][dataset] = {"K": k, "channels": channels, "splits": {dev: {"status": "COMPLETE",
                "input_source": input_record, "methods": {name: {"unchanged_coordinates_byte_exact": True,
                    "gated_coordinates_current_anchor_byte_exact": True}
                    for name in ("RAW_BOLT", "PCA_PINV_BOLT", "FIT_TARGET_PINV_BOLT")}}}}
        write(self.paths[0], producer)
        fit = {"status": "PASS", "protected_TEST_accessed": False, "producer": run.record(self.paths[0]),
               "audit_source": run.record(fit_code), "totals": FIT_COUNTS}
        write(self.paths[1], fit)
        cpu.update(producer=run.record(self.paths[0]), independent_FIT_target_audit=run.record(self.paths[1]))
        write(self.paths[2], cpu)
        cpu_pass = {"status": "PASS", "protected_TEST_accessed": False, "diagnostic": run.record(self.paths[2]),
                    "producer": run.record(self.paths[0]), "fit_target_audit": run.record(self.paths[1]), "totals": CPU_COUNTS}
        write(self.paths[3], cpu_pass)
        source_refs = {"producer": run.record(self.paths[0]), "fit_target_audit": run.record(self.paths[1]),
                       "gate_diagnostic": run.record(self.paths[2]), "gate_audit": run.record(self.paths[3])}
        cost = {"status": "COMPLETE", **source_refs, "expected_cost_cells": 24, "recorded_timer_samples": 144,
            "new_training_updates": 0, "projection_fits": 0, "protected_TEST_accessed": False,
            "new_backbone_training": False, "source_cache_selection_SHA_unchanged": True,
            "original_selected_forecasts_and_locks_unchanged": True}
        write(self.paths[4], cost)
        cost_pass = {"status": "PASS", **source_refs, "cost": run.record(self.paths[4]),
                     "audit_source": run.record(cost_code), "totals": COST_COUNTS, "protected_TEST_accessed": False}
        write(self.paths[5], cost_pass)

    def test_full_three_stage_complete_PASS_gate_accepts_without_array_load(self):
        with patch.object(np, "load", side_effect=AssertionError("actual arrays forbidden")):
            producer, cpu, cost, bindings = run.load_gates(*self.paths)
        self.assertEqual((producer["status"], cpu["status"], cost["status"]), ("COMPLETE",) * 3)
        self.assertEqual(len(bindings), 6)

    def test_pending_any_prerequisite_precedes_scientific_import_or_output(self):
        for path in self.paths:
            original = run.read(path)
            changed = {**original, "status": "RUNNING"}
            write(path, changed)
            called, real_import = [], __import__
            def forbid_scientific(name, *args, **kwargs):
                if name.split(".")[0] in ("numpy", "torch", "fixed_decoder_span_v1", "continuation_runtime_v1"):
                    called.append(name)
                    raise AssertionError("science before completed prerequisites")
                return real_import(name, *args, **kwargs)
            with self.subTest(path=path.name):
                with patch("builtins.__import__", side_effect=forbid_scientific):
                    with self.assertRaisesRegex(ValueError, "COMPLETE_FIT_CONSTANT_AND_WHOLE_COST"):
                        run.execute(*self.paths, Path(self.temporary.name) / "never_started")
                self.assertEqual(called, [])
            write(path, original)

    def test_whole_cost_each_coverage_count_is_required(self):
        audit = run.read(self.paths[5])
        for key in COST_COUNTS:
            changed = copy.deepcopy(audit)
            changed["totals"][key] -= 1
            write(self.paths[5], changed)
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "FULL_BASELINE_WHOLE_COST"):
                    run.load_gates(*self.paths)

    def test_whole_cost_SHA_and_v2_code_required(self):
        self.paths[4].write_text(self.paths[4].read_text() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "EXACT_BASELINE_WHOLE_COST"):
            run.load_gates(*self.paths)

    def test_protected_TEST_change_cannot_use_completed_cost(self):
        cost, audit = run.read(self.paths[4]), run.read(self.paths[5])
        cost["protected_TEST_accessed"] = True
        write(self.paths[4], cost)
        audit["cost"] = run.record(self.paths[4])
        write(self.paths[5], audit)
        with self.assertRaisesRegex(ValueError, "NO_TEST_BASELINE_WHOLE_COST"):
            run.load_gates(*self.paths)


class CachedHelpers(unittest.TestCase):
    def fixture(self):
        s = np.tile(np.array([-1., 1.], np.float32), 256)
        t = np.tile(np.array([-1., -1., 1., 1.], np.float32), 128)
        z = np.tile(np.stack((s, t), axis=1)[None], (2, 1, 1))
        d = np.array([[1., 2., 0.], [0., 1., 1.]], np.float32)
        bias = np.array([3., -1., 2.], np.float32)
        history = z @ d + bias
        future = np.repeat(z[:, -1:], 48, axis=1) + np.float32(.25)
        prediction = history[:, -1:] + (future - z[:, -1:]) @ d
        origins = np.array([512, 1024], np.int64)
        cached = {"latent_history": z, "latent_future": future, "prediction": prediction, "origins": origins}
        values = {"history": history, "history_observed": np.ones_like(history, bool), "target": prediction.copy(),
                  "observed": np.ones_like(prediction, bool), "raw_last": history[:, -1].copy(),
                  "origins": origins, "mean": np.zeros(3, np.float32), "std": np.array([1., 2., 3.], np.float32)}
        state = {"decoder": d.astype(np.float64), "decoder_bias": bias.astype(np.float64)}
        return values, state, cached

    def test_native_past_shared_only_when_bytes_and_origins_match(self):
        _, _, cached = self.fixture()
        second = {key: value.copy() for key, value in cached.items()}
        np.testing.assert_array_equal(run.shared_native_past(cached, second), cached["latent_history"])
        second["latent_history"][0, 0, 0] = np.nextafter(second["latent_history"][0, 0, 0], np.float32(10.))
        with self.assertRaisesRegex(ValueError, "NOT_BYTE_IDENTICAL"):
            run.shared_native_past(cached, second)

    def test_static_bytes_preserved_and_static_new_einsum_control_is_distinct(self):
        values, original_state, cached = self.fixture()
        local = fit_local_past_decoder(values["history"], values["history_observed"], cached["latent_history"])
        before = cached["prediction"].tobytes()
        output = run.backend_predictions(local, original_state, values, {"BOLT": cached})["BOLT"]
        self.assertEqual(output["original"].tobytes(), before)
        np.testing.assert_allclose(output["static_replay"], cached["prediction"], atol=1e-6)
        scores = run.score_pair(values, **{"original": output["original"], "local": output["local"], "static_replay": output["static_replay"]})
        self.assertAlmostEqual(scores["local_scores_all_observed"]["normalized"]["mse"], 0.)
        self.assertIn("static_D_einsum_arithmetic_replay", scores)

    def test_other_native_seed_future_does_not_refit_shared_decoder(self):
        values, original_state, cached = self.fixture()
        second = {key: value.copy() for key, value in cached.items()}
        second["latent_future"] += .5
        z = run.shared_native_past(cached, second)
        local = fit_local_past_decoder(values["history"], values["history_observed"], z)
        result = run.backend_predictions(local, original_state, values, {"NATIVE_s92601": cached, "NATIVE_s92602": second})
        self.assertFalse(np.array_equal(result["NATIVE_s92601"]["local"], result["NATIVE_s92602"]["local"]))

    def test_future_label_alteration_does_not_change_local_D_or_forecast(self):
        values, original_state, cached = self.fixture()
        a = fit_local_past_decoder(values["history"], values["history_observed"], cached["latent_history"])
        first = run.backend_predictions(a, original_state, values, {"BOLT": cached})
        values["target"][:] = 1e7
        values["observed"][:] = False
        b = fit_local_past_decoder(values["history"], values["history_observed"], cached["latent_history"])
        second = run.backend_predictions(b, original_state, values, {"BOLT": cached})
        np.testing.assert_array_equal(a.decoder32, b.decoder32)
        np.testing.assert_array_equal(first["BOLT"]["local"], second["BOLT"]["local"])

    def test_bad_origin_rejects_only_that_cached_helper_call(self):
        values, original_state, cached = self.fixture()
        local = fit_local_past_decoder(values["history"], values["history_observed"], cached["latent_history"])
        bad = {key: value.copy() for key, value in cached.items()}
        bad["origins"][0] += 1
        with self.assertRaisesRegex(ValueError, "ORIGINS"):
            run.backend_predictions(local, original_state, values, {"BOLT": bad})
        self.assertIn("BOLT", run.backend_predictions(local, original_state, values, {"BOLT": cached}))

    def test_one_method_failure_preserves_independent_seed_prediction(self):
        values, original_state, cached = self.fixture()
        local = fit_local_past_decoder(values["history"], values["history_observed"], cached["latent_history"])
        bad = {key: value.copy() for key, value in cached.items()}
        bad["origins"][0] += 1
        predictions, failures = run.isolated_method_predictions(local, original_state, values,
            {"NATIVE_s92601": bad, "NATIVE_s92602": cached})
        self.assertEqual(set(predictions), {"NATIVE_s92602"})
        self.assertEqual(set(failures), {"NATIVE_s92601"})
        self.assertEqual(failures["NATIVE_s92601"]["failure"]["type"], "ValueError")


if __name__ == "__main__":
    unittest.main()
