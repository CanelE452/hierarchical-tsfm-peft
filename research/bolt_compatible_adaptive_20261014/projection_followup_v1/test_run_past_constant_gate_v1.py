"""Temporary exact-audit gates and small cached-output scalar fixtures only."""
from pathlib import Path
import copy
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import run_past_constant_gate_v1 as run


COVERAGE = {"projection_states_checked": 4, "native_fits_checked": 8,
            "checkpoint_states_checked": 40, "score_field_checks": 1464,
            "saved_prediction_arrays_checked": 88, "cost_cells_checked": 20,
            "recorded_timer_samples_checked": 120, "target_projection_fits_checked": 2,
            "reused_PCA_controls_checked": 2, "output_subspace_diagnostic_cells_checked": 12}


def write_json(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")


class AcceptanceGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.source = self.folder / "RESULT.json"
        self.audit = self.folder / "PASS_v2.json"
        self.audit_code = self.folder / "audit_fit_target_comparison_v2.py"
        self.audit_code.write_text("synthetic audited source", encoding="utf-8")
        self.producer = {"status": "COMPLETE", "protected_TEST_accessed": False,
                         "new_backbone_training": False, "native_training_fits": 8,
                         "projection_fits": 2, "reused_projection_states": 2,
                         "projection_conditions": 4, "failed_paths": []}
        write_json(self.source, self.producer)
        self.certificate = {"status": "PASS", "producer": run.record(self.source),
                            "audit_source": run.record(self.audit_code), "totals": COVERAGE,
                            "protected_TEST_accessed": False}
        write_json(self.audit, self.certificate)

    def test_exact_complete_v2_PASS_is_accepted(self):
        source, binding, audit, code = run.accepted_gate(self.source, self.audit)
        self.assertEqual(source, self.producer)
        self.assertEqual(binding, run.record(self.source))
        self.assertEqual(audit, run.record(self.audit))
        self.assertEqual(code, run.record(self.audit_code))

    def test_all_coverage_counters_required(self):
        for key in COVERAGE:
            changed = copy.deepcopy(self.certificate)
            changed["totals"][key] -= 1
            write_json(self.audit, changed)
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "FULL_ACCEPTED"):
                    run.accepted_gate(self.source, self.audit)

    def test_source_or_audit_code_SHA_mutation_rejected(self):
        self.source.write_text(self.source.read_text() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "EXACT_FIT_TARGET"):
            run.accepted_gate(self.source, self.audit)
        write_json(self.source, self.producer)
        self.audit_code.write_text("mutated source", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "EXACT_INDEPENDENT_V2"):
            run.accepted_gate(self.source, self.audit)

    def test_v1_audit_cannot_be_silently_accepted_as_v2(self):
        old = self.folder / "audit_fit_target_comparison_v1.py"
        old.write_text("same scientific contents", encoding="utf-8")
        self.certificate["audit_source"] = run.record(old)
        write_json(self.audit, self.certificate)
        with self.assertRaisesRegex(ValueError, "EXACT_INDEPENDENT_V2"):
            run.accepted_gate(self.source, self.audit)

    def test_pending_audit_stops_before_numpy_Meter_or_arrays(self):
        self.certificate["status"] = "RUNNING"
        write_json(self.audit, self.certificate)
        real_import, called = __import__, []
        def forbid_science(name, *args, **kwargs):
            if name.split(".")[0] in ("numpy", "torch", "fixed_decoder_span_v1", "continuation_runtime_v1"):
                called.append(name)
                raise AssertionError("actual science must not begin before PASS")
            return real_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=forbid_science):
            with self.assertRaisesRegex(ValueError, "COMPLETE_FIT_TARGET_AND_V2_PASS"):
                run.execute(self.source, self.audit, self.folder / "never_started")
        self.assertEqual(called, [])
        self.assertFalse((self.folder / "never_started").exists())

    def test_TEST_false_and_complete_original_training_scope_required(self):
        for field, value in (("protected_TEST_accessed", True), ("native_training_fits", 7),
                             ("new_backbone_training", True), ("failed_paths", [{"failure": "fixture"}])):
            changed = {**self.producer, field: value}
            write_json(self.source, changed)
            certificate = {**self.certificate, "producer": run.record(self.source)}
            write_json(self.audit, certificate)
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "NO_TEST_FROZEN_FIT_TARGET"):
                    run.accepted_gate(self.source, self.audit)


class CachedCaseTests(unittest.TestCase):
    def fixture(self):
        history = np.zeros((2, 512, 2), dtype=np.float32)
        history[:, :, 1] = np.linspace(-1., 0., 512, dtype=np.float32)
        observed_past = np.ones_like(history, dtype=bool)
        target = np.zeros((2, 48, 2), dtype=np.float32)
        target[1, :, 0] = 1.
        values = {"history": history, "history_observed": observed_past, "target": target,
                  "observed": np.ones_like(target, dtype=bool), "raw_last": history[:, -1].copy(),
                  "origins": np.array([512, 1024], dtype=np.int64),
                  "std": np.array([2., 3.], dtype=np.float32), "mean": np.zeros(2, dtype=np.float32)}
        methods = {name: np.ones((2, 48, 2), dtype=np.float32) for name in run.METHODS}
        return values, methods

    def test_seven_methods_include_raw_and_two_frozen_native_seeds_each(self):
        values, methods = self.fixture()
        summary, arrays = run.evaluate_cached_case(values, methods)
        self.assertEqual(set(summary["methods"]), set(run.METHODS))
        self.assertEqual(len(summary["methods"]), 7)
        self.assertEqual(summary["gate_eligible_windows_per_channel"], [2, 0])
        self.assertEqual(summary["eligible_windows_with_observed_future_change_per_channel"], [1, 0])
        for name in run.METHODS:
            np.testing.assert_array_equal(arrays[name + "_gated_prediction"][:, :, 0], 0.)
            np.testing.assert_array_equal(arrays[name + "_gated_prediction"][:, :, 1], 1.)

    def test_future_labels_change_scores_not_gate_or_predictions(self):
        values, methods = self.fixture()
        _, first = run.evaluate_cached_case(values, methods)
        values["target"][:] = 50.
        values["observed"][0, :10] = False
        _, second = run.evaluate_cached_case(values, methods)
        np.testing.assert_array_equal(first["keep_forecast"], second["keep_forecast"])
        for name in run.METHODS:
            np.testing.assert_array_equal(first[name + "_gated_prediction"], second[name + "_gated_prediction"])

    def test_observed_macro_MSE_and_physical_scales_hand_calculation(self):
        values, methods = self.fixture()
        summary, _ = run.evaluate_cached_case(values, methods)
        raw = summary["methods"]["RAW_BOLT"]
        self.assertEqual(raw["ungated_scores_all_observed"]["normalized"]["mse"], .75)
        self.assertEqual(raw["gated_scores_all_observed"]["normalized"]["mse"], .75)
        self.assertEqual(raw["ungated_scores_all_observed"]["physical_per_channel_diagnostic"]["mse"], 5.5)
        self.assertTrue(raw["unchanged_coordinates_byte_exact"])
        self.assertFalse(raw["optimized_gated_native_head_claimed"])

    def test_incomplete_future_is_not_counted_as_known_unchanged(self):
        values, methods = self.fixture()
        values["observed"][0, :, 0] = False
        values["target"][0, :, 0] = np.nan
        summary, _ = run.evaluate_cached_case(values, methods)
        self.assertEqual(summary["eligible_windows_with_any_observed_future_per_channel"], [1, 0])
        self.assertEqual(summary["eligible_observed_future_positions_per_channel"], [48, 0])
        self.assertEqual(summary["complete_target_positions"], 48)

    def test_cached_inputs_and_native_predictions_are_not_modified(self):
        values, methods = self.fixture()
        before_values = {key: value.tobytes() for key, value in values.items()}
        before_predictions = {key: value.tobytes() for key, value in methods.items()}
        run.evaluate_cached_case(values, methods)
        self.assertEqual(before_values, {key: value.tobytes() for key, value in values.items()})
        self.assertEqual(before_predictions, {key: value.tobytes() for key, value in methods.items()})


if __name__ == "__main__":
    unittest.main()
