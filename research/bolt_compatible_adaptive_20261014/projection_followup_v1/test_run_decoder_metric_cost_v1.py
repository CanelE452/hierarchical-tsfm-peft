"""Filesystem-safe synthetic deployment fixtures, never real inputs or models."""
from pathlib import Path
import inspect
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import whole_decoder_metric_cost_v1 as cost
import run_decoder_metric_cost_v1 as runner


def gate_fixture(folder):
    def write(name, value):
        path = folder / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path
    state = runner.file_record(write("state.json", {"fixture": True}))
    future = runner.file_record(write("future.json", {"fixture": True}))
    span, span_audit = write("span.json", {}), write("span_audit.json", {})
    producer = {"status": "COMPLETE", "protected_TEST_accessed": False, "datasets": {}}
    decoder = {"status": "COMPLETE", "protected_TEST_accessed": False, "datasets": {},
               "completed_cells": 12, "failed_paths": [], "new_training": False,
               "future_labels_used_to_fit_alternative_D": False,
               "method_checkpoint_branch_selection_used": False,
               "original_selected_forecasts_and_locks_unchanged": True,
               "span_diagnostic": runner.file_record(span), "span_audit": runner.file_record(span_audit)}
    for dataset, (c, k, split) in runner.DATASETS.items():
        producer["datasets"][dataset] = {"status": "COMPLETE", "K": k,
            "data_contract": {split: {"raw_channels": c}}, "projections": {
                route: {"state_artifact": state, "bolt": {"artifacts": {split: future}}} for route in ("PCA", "FORECA")}}
        decoder["datasets"][dataset] = {"K": k, "development_split": split, "projections": {
            route: {"status": "COMPLETE", "source_state": state, "splits": {
                split: {"status": "COMPLETE", "cached_future_sources": {"BOLT": future}}}} for route in ("PCA", "FORECA")}}
    producer_path = write("producer.json", producer)
    projection_audit_path = write("projection_audit.json", {
        "status": "PASS", "producer": runner.file_record(producer_path), "protected_TEST_accessed": False,
        "totals": {"projection_states_checked": 4, "native_fits_checked": 8, "checkpoint_states_checked": 40,
                   "cost_cells_checked": 20, "recorded_timer_samples_checked": 120}})
    decoder.update(producer=runner.file_record(producer_path), projection_audit=runner.file_record(projection_audit_path))
    decoder_path = write("decoder.json", decoder)
    decoder_audit_path = write("decoder_audit.json", {
        "status": "PASS", "diagnostic": runner.file_record(decoder_path), "producer": runner.file_record(producer_path),
        "projection_audit": runner.file_record(projection_audit_path), "span_diagnostic": runner.file_record(span),
        "span_audit": runner.file_record(span_audit), "protected_TEST_accessed": False,
        "totals": {"base_cells_checked": 12, "decoder_cells_checked": 24, "cached_forecast_arrays_checked": 96,
                   "original_predictions_byte_exact_checked": 48}})
    return producer_path, projection_audit_path, decoder_path, decoder_audit_path


class RunnerTests(unittest.TestCase):
    def test_helper_is_original_timer_with_only_authentic_routes_changed(self):
        original = (runner.HERE / "whole_cost_v1.py").read_text(encoding="utf-8")
        old = 'ROUTES = ("RAW_BOLT", "PCA_BOLT", "FORECA_BOLT", "PCA_NATIVE", "FORECA_NATIVE")'
        new = 'ROUTES = ("RAW_BOLT", "PCA_PAST_LS_BOLT", "FORECA_PAST_LS_BOLT",\n          "PCA_ENCODER_PINV_BOLT", "FORECA_ENCODER_PINV_BOLT")'
        self.assertEqual((runner.HERE / "whole_decoder_metric_cost_v1.py").read_text(encoding="utf-8"), original.replace(old, new))
        self.assertEqual(tuple(cost.ROUTES), runner.ROUTES)
        self.assertTrue(all("NATIVE" not in route for route in cost.ROUTES))

    def test_exact_pass_bindings_and_full_audit_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = gate_fixture(Path(temporary))
            self.assertEqual(runner.load_gates(*paths)[2]["completed_cells"], 12)
            audit = runner.read(paths[-1])
            audit["totals"]["cached_forecast_arrays_checked"] -= 1
            paths[-1].write_text(json.dumps(audit), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "COMPLETE_DECODER_AUDIT_COVERAGE"):
                runner.load_gates(*paths)

    def test_no_scientific_execution_before_both_independent_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = gate_fixture(Path(temporary))
            audit = runner.read(paths[-1])
            audit["status"] = "PENDING"
            paths[-1].write_text(json.dumps(audit), encoding="utf-8")
            with patch.object(runner, "make_predictor", side_effect=AssertionError("must not run")) as call:
                with self.assertRaisesRegex(ValueError, "BOTH_INDEPENDENT_PASS"):
                    runner.execute("synthetic_never_started", *paths)
                call.assert_not_called()

    def test_changed_decoder_result_fails_exact_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = gate_fixture(Path(temporary))
            paths[2].write_text(paths[2].read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "EXACT_METADATA_BINDING"):
                runner.load_gates(*paths)

    def test_label_fitted_or_selected_intervention_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = gate_fixture(Path(temporary))
            decoder = runner.read(paths[2])
            decoder["future_labels_used_to_fit_alternative_D"] = True
            paths[2].write_text(json.dumps(decoder), encoding="utf-8")
            audit = runner.read(paths[-1])
            audit["diagnostic"] = runner.file_record(paths[2])
            paths[-1].write_text(json.dumps(audit), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "LABEL_FREE_FIXED_MODEL"):
                runner.load_gates(*paths)

    def test_real_encode_F_D_formula_and_changed_decoder_only(self):
        rng = np.random.default_rng(5015)
        history = rng.normal(size=(8, 512, 3)).astype(np.float32)
        center = np.array((2, -3, 4), np.float32)
        encoder = rng.normal(size=(3, 2)).astype(np.float32)
        decoder = rng.normal(size=(2, 3)).astype(np.float32)
        encoder_original, decoder_original = encoder.copy(), decoder.copy()
        calls = []
        def frozen_F(z):
            calls.append(z.copy())
            return np.repeat(z[:, -1:, :] + np.float32(.25), 48, axis=1)
        predict = runner.make_predictor(center, encoder, decoder, frozen_F)
        prediction = predict(history)
        z = (history - center) @ encoder
        future = np.repeat(z[:, -1:] + np.float32(.25), 48, axis=1)
        np.testing.assert_array_equal(prediction, history[:, -1:] + (future - z[:, -1:]) @ decoder)
        np.testing.assert_array_equal(calls[0], z)
        np.testing.assert_array_equal(encoder, encoder_original)
        np.testing.assert_array_equal(decoder, decoder_original)
        second = runner.make_predictor(center, encoder, decoder * np.float32(2), frozen_F)
        second_prediction = second(history)
        np.testing.assert_array_equal(calls[-1], z)
        np.testing.assert_array_equal(second_prediction, history[:, -1:] + (future - z[:, -1:]) @ (decoder * np.float32(2)))
        self.assertFalse(np.array_equal(prediction, second_prediction))

    def test_zero_innovation_is_exact_currentC_and_no_future_target_parameter(self):
        rng = np.random.default_rng(5016)
        history = rng.normal(size=(8, 512, 3)).astype(np.float32)
        prediction = runner.make_predictor(np.ones(3), rng.normal(size=(3, 2)), rng.normal(size=(2, 3)),
            lambda z: np.repeat(z[:, -1:], 48, axis=1))(history)
        np.testing.assert_array_equal(prediction, np.repeat(history[:, -1:], 48, axis=1))
        self.assertEqual(set(inspect.signature(runner.make_predictor).parameters), {"center", "encoder", "decoder", "frozen_F"})
        self.assertEqual(set(inspect.signature(runner.make_predictor(np.zeros(3), np.ones((3, 2)), np.ones((2, 3)), lambda z: z)).parameters), {"history"})


class CostTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(61015)
        self.physical = rng.normal(size=(16, 512, 2)).astype(np.float32)
        self.observed = np.ones(self.physical.shape, dtype=bool)
        self.observed[:, :7, 0] = False
        self.physical[:, :7, 0] = np.nan
        self.mean = np.array((2, -3), np.float32)
        self.std = np.array((.5, 2), np.float32)
        self.origins = np.arange(512, 528, dtype=np.int64)
        self.calls = {route: 0 for route in cost.ROUTES}
        self.guard_calls = 0
        self.sync_calls = 0
        self.memory_calls = 0
        self.predictors = {}
        for index, route in enumerate(cost.ROUTES):
            def predict(x, name=route, offset=index):
                self.calls[name] += 1
                return (np.repeat(x[:, -1:], 48, axis=1) + np.float32(offset * .1)).astype(np.float32)
            self.predictors[route] = predict
        history = cost.preprocess(self.physical, self.observed, self.mean, self.std)
        self.references = {route: self.predictors[route](history) for route in cost.ROUTES}
        self.calls = {route: 0 for route in cost.ROUTES}
        self.metadata = {route: {"synthetic": True, "seed": 92601, "no_model_loaded": True} for route in cost.ROUTES}

    def check(self):
        self.guard_calls += 1

    def synchronize(self):
        self.sync_calls += 1

    def memory(self):
        self.memory_calls += 1
        return {"rss_bytes": 12345, "shared_gpu_reserved_bytes": 0, "fixture": True}

    def measure(self, output, dataset="synthetic_a", references=None):
        return cost.measure_cost(dataset, self.physical, self.observed, self.mean, self.std,
                                 self.origins, self.predictors,
                                 self.references if references is None else references,
                                 self.metadata, output, self.check, self.synchronize, self.memory)

    def test_five_rotation_positions_and_extra_repeat_scope(self):
        schedule = cost.balanced_schedule()
        self.assertEqual(len(schedule), 6)
        for route in cost.ROUTES:
            self.assertEqual(sorted(order.index(route) for order in schedule[:5]), list(range(5)))
        self.assertTrue(all(set(order) == set(cost.ROUTES) for order in schedule))
        self.assertEqual(schedule, cost.balanced_schedule())

    def test_forward_fill_uses_only_observed_past_with_fit_mean_start(self):
        actual = cost.preprocess(self.physical, self.observed, self.mean, self.std)
        last = np.broadcast_to(self.mean, (16, 2)).copy()
        expected = np.empty_like(self.physical)
        for time_index in range(512):
            last = np.where(self.observed[:, time_index], self.physical[:, time_index], last)
            expected[:, time_index] = (last - self.mean) / self.std
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_array_equal(actual[:, :7, 0], np.zeros((16, 7), np.float32))

    def test_false_mask_values_have_no_forward_fill_effect(self):
        altered = self.physical.copy()
        altered[~self.observed] = 1e30
        np.testing.assert_array_equal(cost.preprocess(altered, self.observed, self.mean, self.std),
                                      cost.preprocess(self.physical, self.observed, self.mean, self.std))

    def test_future_targets_are_absent_from_timer_api(self):
        parameters = inspect.signature(cost.measure_cost).parameters
        self.assertNotIn("target", parameters)
        self.assertNotIn("future_observed", parameters)
        self.assertIn("saved_normalized_predictions", parameters)

    def test_twenty_cells_across_two_datasets_and_saved_timing_arithmetic(self):
        with tempfile.TemporaryDirectory() as temporary:
            a = self.measure(Path(temporary) / "a")
            b = self.measure(Path(temporary) / "b", "synthetic_b")
            self.assertEqual(len(a["rows"]) + len(b["rows"]), 20)
            self.assertEqual(a["status"], "COMPLETE")
            self.assertEqual(self.memory_calls, 120)
            self.assertEqual(set((row["route"], row["batch"]) for row in a["rows"]),
                             set((route, batch) for route in cost.ROUTES for batch in (1, 8)))
            for row in a["rows"]:
                self.assertEqual(row["repeats"], 6)
                self.assertEqual(row["warmup_each_route"], 2)
                self.assertAlmostEqual(row["median_seconds_per_window"], np.median(row["seconds"]) / row["batch"])
                self.assertEqual(row["origins"], self.origins[:row["batch"]].tolist())
                self.assertIn("shared resident", row["memory_scope"])
                self.assertFalse(row["memory_counter_reset_inside_module"])
                for sample in row["samples"]:
                    self.assertEqual(sample["seconds"], (sample["perf_end_ns"] - sample["perf_start_ns"]) / 1e9)
                    self.assertGreaterEqual(sample["seconds"], 0)
                    self.assertTrue(all(value["pass"] for value in sample["parity"].values()))
                    artifact = sample["sample_artifact"]
                    self.assertEqual(cost.sha(artifact["path"]), artifact["sha256"])
                    with np.load(artifact["path"], allow_pickle=False) as saved:
                        self.assertEqual(saved["normalized"].shape, (row["batch"], 48, 2))
                        np.testing.assert_array_equal(saved["physical"], saved["normalized"] * self.std + self.mean)
                        np.testing.assert_array_equal(saved["origins"], self.origins[:row["batch"]])
            self.assertTrue(all(count == 36 for count in self.calls.values()))

    def test_wrong_scientific_prediction_fails_and_preserves_receipt(self):
        references = {route: value.copy() for route, value in self.references.items()}
        references["RAW_BOLT"] += np.float32(.1)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bad"
            with self.assertRaisesRegex(ValueError, "SCIENTIFIC_REFERENCE_PARITY_FAILED"):
                self.measure(output, references=references)
            result = json.loads((output / "RESULT.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "FAILED_PRESERVED")
            self.assertTrue((output / "B1/RAW_BOLT_REFERENCE.npz").exists())

    def test_output_directory_is_exclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileExistsError):
                self.measure(Path(temporary))

    def test_observed_nonfinite_past_rejected(self):
        self.physical[0, 20, 0] = np.nan
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "OBSERVED_FINITE"):
                self.measure(Path(temporary) / "bad")

    def test_wrong_stats_and_origins_are_rejected(self):
        self.std[0] = 0
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "FIT_STATS"):
                self.measure(Path(temporary) / "bad")
        self.std[0] = .5
        self.origins = self.origins.astype(np.int32)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "INT64_ORIGINS"):
                self.measure(Path(temporary) / "bad")

    def test_route_coverage_and_reference_shape_are_rejected(self):
        predictor = self.predictors.pop("PCA_ENCODER_PINV_BOLT")
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "EXACT_FIVE"):
                self.measure(Path(temporary) / "bad")
        self.predictors["PCA_ENCODER_PINV_BOLT"] = predictor
        self.references["PCA_ENCODER_PINV_BOLT"] = self.references["PCA_ENCODER_PINV_BOLT"][:, :47]
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "FP32_N48C_SAVED"):
                self.measure(Path(temporary) / "bad")

    def test_callback_wrong_dtype_rejected(self):
        self.predictors["RAW_BOLT"] = lambda x: np.zeros((len(x), 48, 2), np.float64)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "NORMALIZED_FP32"):
                self.measure(Path(temporary) / "bad")

    def test_callback_drift_fails_after_sample_saved(self):
        count = 0
        original = self.predictors["RAW_BOLT"]
        def drifting(x):
            nonlocal count
            count += 1
            prediction = original(x)
            return prediction if count <= 3 else prediction + np.float32(.2)
        self.predictors["RAW_BOLT"] = drifting
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bad"
            with self.assertRaisesRegex(ValueError, "TIMED_OUTPUT_PARITY_FAILED"):
                self.measure(output)
            self.assertTrue((output / "B1/RAW_BOLT_r0.json").exists())
            self.assertTrue((output / "B1/RAW_BOLT_r0.npz").exists())

    def test_physical_parity_enforced_even_when_normalized_parity_passes(self):
        self.mean[:] = 0
        self.std[:] = 1e4
        self.references = {route: np.zeros((16, 48, 2), np.float32) for route in cost.ROUTES}
        self.predictors = {route: lambda x: np.full((len(x), 48, 2), 5e-6, np.float32) for route in cost.ROUTES}
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "SCIENTIFIC_REFERENCE_PARITY_FAILED"):
                self.measure(Path(temporary) / "bad")

    def test_parity_shape_nonfinite_and_exact_tolerances(self):
        source = np.array((0., 1.), np.float32)
        self.assertTrue(cost.parity(source, source)["pass"])
        self.assertFalse(cost.parity(source, source[:1])["pass"])
        self.assertFalse(cost.parity(np.array((np.nan, 1), np.float32), source)["pass"])
        self.assertFalse(cost.parity(source + .1, source)["pass"])
        self.assertEqual(cost.parity(source, source)["atol"], 1e-5)
        self.assertEqual(cost.parity(source, source)["rtol"], 2e-6)


if __name__ == "__main__":
    unittest.main()
