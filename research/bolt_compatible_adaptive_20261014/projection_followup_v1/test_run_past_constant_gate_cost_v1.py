"""Filesystem-safe synthetic deployment fixtures, never real inputs or models."""
from pathlib import Path
import inspect
import json
import tempfile
import unittest

import numpy as np

import whole_past_constant_gate_cost_v1 as cost
import run_past_constant_gate_cost_v1 as runner
from unittest.mock import patch


def gate_fixture(folder):
    def write(name, value):
        path = folder / name; path.write_text(json.dumps(value), encoding="utf-8"); return path
    code = runner.record(write("audit_fit_target_comparison_v2.py", {"synthetic": True}))
    producer = {"status": "COMPLETE", "protected_TEST_accessed": False, "new_backbone_training": False,
        "F_before": {"hash": "frozen"}, "F_after": {"hash": "frozen"}, "prior_source_results_caches_and_selection_locks_unchanged": True,
        "datasets": {}}
    cpu = {"status": "COMPLETE", "protected_TEST_accessed": False, "completed_dataset_split_cells": 6,
        "completed_method_pairs": 42, "failed_paths": [], "new_training": False, "new_encoder_forecaster_or_head_forward": False,
        "E_F_native_head_forward_runs": 0, "future_labels_used_for_gate": False, "method_selection_used": False,
        "past_C_condition_metadata_added": True, "same_decoder_side_information_as_ungated": False,
        "strict_K_only_compression": False, "static_original_fixed_D_floor_reused_as_gated_lower_bound": False,
        "source_cache_selection_SHA_unchanged": True, "datasets": {}}
    input_record = runner.record(write("never_opened_arrays.npz", {"synthetic": True}))
    for dataset, (c, k, dev) in runner.DATASETS.items():
        producer["datasets"][dataset] = {"status": "COMPLETE", "K": k, "inputs": {dev: input_record},
            "projections": {route: {} for route in runner.PROJECTIONS}}
        cpu["datasets"][dataset] = {"channels": c, "K": k, "splits": {dev: {"status": "COMPLETE", "input_source": input_record,
            "methods": {route: {"unchanged_coordinates_byte_exact": True, "gated_coordinates_current_anchor_byte_exact": True}
                        for route in ("RAW_BOLT", "PCA_PINV_BOLT", "FIT_TARGET_PINV_BOLT")}}}}
    p = write("PRODUCER.json", producer)
    fit = write("FIT_AUDIT.json", {"status": "PASS", "protected_TEST_accessed": False, "producer": runner.record(p),
        "audit_source": code, "totals": {"projection_states_checked": 4, "target_projection_fits_checked": 2, "reused_PCA_controls_checked": 2,
            "native_fits_checked": 8, "checkpoint_states_checked": 40, "cost_cells_checked": 20, "recorded_timer_samples_checked": 120,
            "saved_prediction_arrays_checked": 88, "output_subspace_diagnostic_cells_checked": 12}})
    cpu.update(producer=runner.record(p), independent_FIT_target_audit=runner.record(fit))
    q = write("CPU.json", cpu)
    qa = write("CPU_AUDIT.json", {"status": "PASS", "protected_TEST_accessed": False, "diagnostic": runner.record(q),
        "producer": runner.record(p), "fit_target_audit": runner.record(fit), "totals": {"dataset_split_cells_checked": 6,
        "method_pairs_checked": 42, "ungated_prediction_arrays_checked": 42, "gated_prediction_arrays_checked": 42, "score_field_checks": 2016}})
    return p, fit, q, qa


class RunnerTests(unittest.TestCase):
    def test_both_exact_passes_and_complete_gate_cohort(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = gate_fixture(Path(temporary))
            self.assertEqual(runner.load_gates(*paths)[2]["completed_method_pairs"], 42)
            value = runner.read(paths[-1]); value["totals"]["gated_prediction_arrays_checked"] = 41
            paths[-1].write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "FULL_CPU_GATE_AUDIT"):
                runner.load_gates(*paths)

    def test_pending_cpu_audit_stops_before_any_predictor_or_gpu_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = gate_fixture(Path(temporary)); value = runner.read(paths[-1]); value["status"] = "PENDING"
            paths[-1].write_text(json.dumps(value), encoding="utf-8")
            with patch.object(runner, "projected_predictor", side_effect=AssertionError("must not run")) as call:
                with self.assertRaisesRegex(ValueError, "BOTH_INDEPENDENT_PASS"):
                    runner.execute("never_started", *paths)
                call.assert_not_called()

    def test_changed_cpu_and_same_side_information_claim_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = gate_fixture(Path(temporary)); value = runner.read(paths[2]); value["same_decoder_side_information_as_ungated"] = True
            paths[2].write_text(json.dumps(value), encoding="utf-8")
            audit = runner.read(paths[-1]); audit["diagnostic"] = runner.record(paths[2]); paths[-1].write_text(json.dumps(audit), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "EXPLICIT_CHANGED_DECODER_SIDE"):
                runner.load_gates(*paths)

    def test_gate_uses_actual_explicit_observed_mask_and_does_not_skip_F(self):
        history = np.ones((8, 512, 2), np.float32)
        history[:, 0, 1] = 2.
        observed = np.ones(history.shape, bool)
        observed[:, :-1, 0] = False
        calls = []
        def ungated(x, mask):
            calls.append(mask.copy())
            return np.repeat(x[:, -1:] + np.float32(3), 48, axis=1)
        gated = runner.gated_predictor(ungated)
        before = gated(history, observed)
        eligible = observed.copy(); eligible[:, 0, 0] = True
        after = gated(history, eligible)
        self.assertEqual(len(calls), 2)
        np.testing.assert_array_equal(calls[0], observed)
        np.testing.assert_array_equal(before, ungated(history, observed))
        np.testing.assert_array_equal(after[:, :, 0], np.ones((8, 48), np.float32))
        np.testing.assert_array_equal(after[:, :, 1], before[:, :, 1])
        self.assertEqual(set(inspect.signature(gated).parameters), {"history", "history_observed"})

    def test_projected_predictor_uses_same_frozen_E_F_D_without_future_or_mask_capture(self):
        rng = np.random.default_rng(1510)
        history = rng.normal(size=(8, 512, 3)).astype(np.float32); observed = np.ones(history.shape, bool)
        center = rng.normal(size=3).astype(np.float32); encoder = rng.normal(size=(3, 2)).astype(np.float32); decoder = rng.normal(size=(2, 3)).astype(np.float32)
        def frozen_F(z): return np.repeat(z[:, -1:] + np.float32(.25), 48, axis=1)
        callback = runner.projected_predictor(center, encoder, decoder, frozen_F)
        latent = (history - center) @ encoder
        expected = history[:, -1:] + (frozen_F(latent) - latent[:, -1:]) @ decoder
        np.testing.assert_array_equal(callback(history, observed), expected)
        changed = observed.copy(); changed[:] = False
        np.testing.assert_array_equal(callback(history, changed), expected)
        self.assertEqual(set(inspect.signature(callback).parameters), {"history", "history_observed"})

    def test_timer_passes_original_mask_after_preprocessing(self):
        physical = np.ones((8, 512, 2), np.float32); mask = np.ones(physical.shape, bool); mask[:, :5, 0] = False
        seen = []
        def callback(history, observed):
            seen.append((history.copy(), observed.copy()))
            return np.repeat(history[:, -1:], 48, axis=1)
        cost.whole_predict(callback, physical, mask, np.zeros(2, np.float32), np.ones(2, np.float32))
        np.testing.assert_array_equal(seen[0][1], mask)
        self.assertEqual(seen[0][0].dtype, np.float32)
        np.testing.assert_array_equal(seen[0][0][:, :5, 0], np.zeros((8, 5), np.float32))

    def test_one_ULP_past_difference_can_change_gate_even_with_output_tolerance_parity(self):
        from past_constant_gate_v1 import past_constant_gate
        source = np.ones((8, 512, 2), np.float32); observed = np.ones(source.shape, bool)
        altered = source.copy(); altered[:, 0, 0] = np.nextafter(np.float32(1), np.float32(2))
        self.assertTrue(cost.parity(altered, source)["pass"])
        a, b = past_constant_gate(source, observed), past_constant_gate(altered, observed)
        self.assertFalse(np.array_equal(a["keep_forecast"], b["keep_forecast"]))
        self.assertFalse(np.array_equal(a["exact_flat"], b["exact_flat"]))
        np.testing.assert_array_equal(a["true_observed_counts"], b["true_observed_counts"])


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
            def predict(x, history_observed, name=route, offset=index):
                self.calls[name] += 1
                return (np.repeat(x[:, -1:], 48, axis=1) + np.float32(offset * .1)).astype(np.float32)
            self.predictors[route] = predict
        history = cost.preprocess(self.physical, self.observed, self.mean, self.std)
        self.references = {route: self.predictors[route](history, self.observed) for route in cost.ROUTES}
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

    def test_six_rotation_positions_are_fully_balanced(self):
        schedule = cost.balanced_schedule()
        self.assertEqual(len(schedule), 6)
        for route in cost.ROUTES:
            self.assertEqual(sorted(order.index(route) for order in schedule[:6]), list(range(6)))
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

    def test_twenty_four_cells_across_two_datasets_and_saved_timing_arithmetic(self):
        with tempfile.TemporaryDirectory() as temporary:
            a = self.measure(Path(temporary) / "a")
            b = self.measure(Path(temporary) / "b", "synthetic_b")
            self.assertEqual(len(a["rows"]) + len(b["rows"]), 24)
            self.assertEqual(a["status"], "COMPLETE")
            self.assertEqual(self.memory_calls, 144)
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
        predictor = self.predictors.pop("PCA_PINV_BOLT_GATED")
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "EXACT_SIX"):
                self.measure(Path(temporary) / "bad")
        self.predictors["PCA_PINV_BOLT_GATED"] = predictor
        self.references["PCA_PINV_BOLT_GATED"] = self.references["PCA_PINV_BOLT_GATED"][:, :47]
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "FP32_N48C_SAVED"):
                self.measure(Path(temporary) / "bad")

    def test_callback_wrong_dtype_rejected(self):
        self.predictors["RAW_BOLT"] = lambda x, history_observed: np.zeros((len(x), 48, 2), np.float64)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "NORMALIZED_FP32"):
                self.measure(Path(temporary) / "bad")

    def test_callback_drift_fails_after_sample_saved(self):
        count = 0
        original = self.predictors["RAW_BOLT"]
        def drifting(x, history_observed):
            nonlocal count
            count += 1
            prediction = original(x, history_observed)
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
        self.predictors = {route: lambda x, history_observed: np.full((len(x), 48, 2), 5e-6, np.float32) for route in cost.ROUTES}
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
