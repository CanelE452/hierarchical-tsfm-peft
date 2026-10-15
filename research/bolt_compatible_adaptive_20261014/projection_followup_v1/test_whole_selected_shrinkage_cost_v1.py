"""Filesystem-safe synthetic deployment fixtures, never real inputs or models."""
from pathlib import Path
import inspect
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import whole_selected_shrinkage_cost_v1 as cost


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
            self.assertEqual(a["recorded_timer_samples"], 60)
            self.assertEqual(a["attempted_timer_calls"], 60)
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
        predictor = self.predictors.pop("PCA_SHRUNK_BOLT")
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "EXACT_FIVE"):
                self.measure(Path(temporary) / "bad")
        self.predictors["PCA_SHRUNK_BOLT"] = predictor
        self.references["PCA_SHRUNK_BOLT"] = self.references["PCA_SHRUNK_BOLT"][:, :47]
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
            receipt = json.loads((output / "RESULT.json").read_text(encoding="utf-8"))
            saved_samples = len(list((output / "B1").glob("*_r*.json")))
            self.assertEqual(receipt["recorded_timer_samples"], saved_samples)
            self.assertEqual(receipt["attempted_timer_calls"], saved_samples)

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

    def test_local_fit_runs_inside_each_timed_callback_and_mask_is_explicit(self):
        from projection_math_v1 import ProjectionState
        from local_decoder_shrinkage_v1 import fit_shrunk_local_decoder
        from run_selected_shrinkage_cost_v1 import shrunk_predictor
        state = ProjectionState(np.zeros(2), np.array([[1.], [0.]]), np.array([[1., 0.]]), np.zeros(2), {})
        events = []
        def local_fit(history, observed, latent, prior, rho):
            events.append(("fit", history.shape[0]))
            np.testing.assert_array_equal(observed, self.observed[:len(history)])
            return fit_shrunk_local_decoder(history, observed, latent, prior, rho)
        def frozen_F(latent):
            events.append(("F", latent.shape[0]))
            return np.repeat(latent[:, -1:] + np.float32(.1), 48, axis=1)
        callback = shrunk_predictor(state, frozen_F, .1)
        history = cost.preprocess(self.physical, self.observed, self.mean, self.std)
        for route in ("PCA_SHRUNK_BOLT", "FIT_TARGET_SHRUNK_BOLT"):
            self.predictors[route] = callback
            self.references[route] = callback(history, self.observed)
        events.clear()
        original_clock = cost.time.perf_counter_ns
        def clock():
            events.append(("clock", 0))
            return original_clock()
        with tempfile.TemporaryDirectory() as temporary:
            with patch("local_decoder_shrinkage_v1.fit_shrunk_local_decoder", side_effect=local_fit), patch.object(cost.time, "perf_counter_ns", side_effect=clock):
                for route in ("PCA_SHRUNK_BOLT", "FIT_TARGET_SHRUNK_BOLT"):
                    self.predictors[route] = shrunk_predictor(state, frozen_F, .1)
                result = self.measure(Path(temporary) / "actual_synthetic_local")
        self.assertTrue(result["local_decoder_fit_inside_timed_callback"])
        self.assertTrue(result["implemented_local_fit_diagnostics_included"])
        self.assertTrue(result["fit_input_binding_hashes_inside_timer"])
        self.assertTrue(all("fit input binding hashes" in row["whole_includes"]
                            and "source/file/state metadata hashes generated outside timed callback" in row["whole_excludes"] for row in result["rows"]))
        for batch in (1, 8):
            self.assertEqual(sum(event == ("fit", batch) for event in events), 18)
        pairs = [index for index, event in enumerate(events) if event[0] == "clock"]
        self.assertEqual(len(pairs), 120)
        local_segments = 0
        for index in range(0, len(pairs), 2):
            inside = events[pairs[index] + 1:pairs[index + 1]]
            if inside:
                self.assertEqual([event[0] for event in inside], ["fit", "F"])
                local_segments += 1
        self.assertEqual(local_segments, 24)


if __name__ == "__main__":
    unittest.main()
