"""Synthetic physical/selected-head and seven-route saved-clock evidence only."""
from pathlib import Path
import copy
import json
import tempfile
import unittest

import numpy as np

import audit_matched_shrunk_native_cost_v1 as audit
from test_audit_selected_shrinkage_cost_v1 import physical_case, parity

audit.initialize_numpy()


def texts():
    return [(audit.HERE / name).read_text(encoding="utf-8-sig") for name in ("whole_selected_shrinkage_cost_v1.py", "whole_matched_shrunk_native_cost_v1.py", "run_matched_shrunk_native_cost_v2.py", "local_decoder_shrinkage_v1.py", "run_matched_shrunk_native_cost_v1.py")]


def physical_fixture(backend):
    entry, arrays, physical, dev, state, cache, saved, metadata, _, rho = physical_case()
    sources = {name: {"path": "synthetic", "sha256": "fixture"} for name in ("source_selected_state", "source_selected_forecasts", "source_fixed_E", "old_Bolt_rho_selection_lock")}
    entry = {key: value for key, value in entry.items() if key not in ("source_shrunk_state", "source_shrunk_forecast", "selection_lock", "source_state", "source_BOLT_cache", "selected_candidate", "deployed_static_prior")}
    entry.update(sources); entry.update(backend=backend, matched_native_head_prediction_checks={})
    heads, predictions = {}, {}
    if backend == "NATIVE_TORCH32":
        for i, seed in enumerate(audit.SEEDS):
            seed = str(seed)
            heads[seed] = {"head.weight": np.zeros((2, 48, 512), np.float32), "head.bias": np.full((2, 48), .1 * (i + 1), np.float32)}
            prediction, _ = audit.matched_audit.independent_head_forward({"latent_history": cache["latent_history"], "decoder": saved["decoder_deployed_FP32"], "raw_last": dev["raw_last"]}, heads[seed])
            predictions[seed] = prediction
            arrays["MATCHED_NATIVE_s" + seed + "_prediction"] = prediction
            entry["matched_native_head_prediction_checks"][seed] = parity(prediction, prediction)
    return entry, arrays, physical, dev, state, saved, cache, metadata, sources, rho, backend, heads, predictions


def complete_synthetic_cost(folder):
    import whole_matched_shrunk_native_cost_v1 as helper
    physical = np.tile(np.arange(512, dtype=np.float32)[None, :, None], (8, 1, 3))
    mask = np.ones(physical.shape, bool)
    mask[:, 3:9, 0] = False
    physical[~mask] = np.nan
    mean, std = np.zeros(3, np.float32), np.ones(3, np.float32)
    normalized = audit.prior.causal_physical_preprocess(physical, mask, mean, std)
    origins = np.arange(512, 520, dtype=np.int64)
    prediction = np.broadcast_to(normalized[:, -1:, :], (8, 48, 3)).copy()
    predictions = {route: prediction.copy() for route in audit.ROUTES}
    metadata = {}
    for route in audit.ROUTES:
        native = "_NATIVE_s" in route
        item = {"native_head_loaded_or_timed": native, "local_fit": route != "RAW_BOLT", "reference_prediction_array": "raw_forecast" if route == "RAW_BOLT" else "prediction"}
        if native: item.update(head_device="cpu", head_threads=2, canonical_prediction_batch_size=32, F_forward_in_this_route=False, native_seed_D_cache_shared_inside_timer=False)
        else: item["shared_resident_frozen_F"] = True
        if route != "RAW_BOLT": item.update(cached_local_D_used_for_timing=False, fit_input_binding_hashes_inside_timer=True)
        metadata[route] = item
    counters = {"resident_rss_bytes": 100, "peak_allocated_vram_bytes": 0, "peak_reserved_vram_bytes": 0}
    callbacks = {route: lambda x, observed: np.broadcast_to(x[:, -1:], (len(x), 48, 3)).copy() for route in audit.ROUTES}
    cost = helper.measure_cost("synthetic", physical, mask, mean, std, origins, callbacks, predictions, metadata, folder, lambda: None, lambda: None, lambda: counters.copy())
    dev = {"history": normalized, "history_observed": mask, "mean": mean, "std": std, "origins": origins}
    return cost, {"history_physical": physical, "history_observed": mask, "origins": origins}, dev, predictions, metadata


class IndependentMatchedWholeCostTests(unittest.TestCase):
    def test_source_true_mask_fresh_ridge_native_call_and_unchanged_timer(self):
        result = audit.source_contract(*texts())
        self.assertTrue(result["no_timed_decoder_cache_or_seed_sharing"])
        self.assertFalse(result["independent_F_or_timer_replay"])

    def test_source_wrong_order_seed_rejected(self):
        source = texts(); source[1] = source[1].replace("ORDER_SEED = 20261015", "ORDER_SEED = 24261015")
        with self.assertRaisesRegex(ValueError, "ORDER_SEED"): audit.source_contract(*source)

    def test_source_cached_D_missing_mask_or_fit_hash_rejected(self):
        for index, old, new in ((2, "fitted = fit_shrunk_local_decoder(history, history_observed, z, prior32, rho)", "fitted = cached_D"), (1, "predictor(normalized_history, observed)", "predictor(normalized_history)"), (3, "hashlib.sha256(value.tobytes())", "hashlib.sha256(b'cached')")):
            source = texts(); self.assertIn(old, source[index]); source[index] = source[index].replace(old, new)
            with self.subTest(index=index), self.assertRaises(ValueError): audit.source_contract(*source)

    def test_full_Latin7_each_route_each_position_once(self):
        rows = audit.independent_schedule()
        self.assertEqual(len(rows), 7)
        for route in audit.ROUTES: self.assertEqual(sorted(row.index(route) for row in rows), list(range(7)))
        self.assertEqual(rows[0], [audit.ROUTES[index] for index in np.random.default_rng(20261015).permutation(7)])

    def test_recorded_ratios_use_paired_samples_not_ratio_of_medians(self):
        raw = {"batch": 8, "seconds": [1., 2., 3., 4., 9., 10., 11.]}
        row = {"batch": 8, "seconds": [2., 3., 4., 8., 9., 10., 22.], "repeats": 7, "warmup_each_route": 2, "paired_RAW_seconds": raw["seconds"]}
        row.update(paired_RAW_ratios=[a / b for a, b in zip(row["seconds"], raw["seconds"])], median_seconds=8., median_seconds_per_window=1., paired_RAW_median_ratio=1.5)
        result = audit.recorded_times(row, raw)
        self.assertEqual(result["samples_checked"], 7)
        row["paired_RAW_median_ratio"] = 2.
        with self.assertRaises(ValueError): audit.recorded_times(row, raw)

    def test_numpy_physical_independent_augmented_LS(self):
        result = audit.independent_physical_precheck(*physical_fixture("BOLT_NUMPY"))
        self.assertEqual(result["state_checks"]["window_fits_checked"], 8)
        self.assertEqual(result["native_first8_prediction_checks"], 0)
        json.dumps(result, allow_nan=False)

    def test_native_physical_independent_encoding_two_selected_heads(self):
        result = audit.independent_physical_precheck(*physical_fixture("NATIVE_TORCH32"))
        self.assertEqual(result["native_first8_prediction_checks"], 2)
        self.assertEqual(result["state_checks"]["channel_slots_checked"], 24)

    def test_wrong_physical_count_mask_lambda_and_saved_D_rejected(self):
        for what in ("count", "mask", "lambda", "D"):
            args = list(physical_fixture("BOLT_NUMPY"))
            if what == "count": args[0]["exact_decisions"]["true_observed_counts"] = False
            elif what == "mask": args[3]["history_observed"] = args[3]["history_observed"].copy(); args[3]["history_observed"][0, 0, 0] = False
            elif what == "lambda": args[7] = copy.deepcopy(args[7]); args[7]["actual_lambda"][0][0] += .1
            else: args[1]["decoder_fit_FP64"] = args[1]["decoder_fit_FP64"].copy(); args[1]["decoder_fit_FP64"][0, 0, 0] += .1
            with self.subTest(what=what), self.assertRaises(ValueError): audit.independent_physical_precheck(*args)

    def test_native_saved_prediction_mutation_rejected_independently(self):
        args = list(physical_fixture("NATIVE_TORCH32"))
        key = "MATCHED_NATIVE_s92601_prediction"
        args[1][key] = args[1][key].copy(); args[1][key][0, 0, 0] += .01
        with self.assertRaises(ValueError): audit.independent_physical_precheck(*args)

    def test_synthetic14cells98saved_samples_outputs_clocks_and_IO(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = audit.audit_cost_dataset(*complete_synthetic_cost(Path(temporary) / "synthetic"), lambda: None, sealed_unix=0.)
        self.assertEqual((result["cells_checked"], result["recorded_timer_samples_checked"]), (14, 98))
        self.assertFalse(result["clock_memory_or_F_forward_independently_remeasured"])

    def test_bad_saved_sample_count_and_wrong_seventh_order_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = list(complete_synthetic_cost(Path(temporary) / "synthetic"))
            args[0]["recorded_timer_samples"] = 97
            with self.assertRaises(ValueError): audit.audit_cost_dataset(*args, lambda: None, sealed_unix=0.)
            args[0]["recorded_timer_samples"] = 98; args[0]["schedule"][-1] = args[0]["schedule"][0]
            with self.assertRaises(ValueError): audit.audit_cost_dataset(*args, lambda: None, sealed_unix=0.)

    def test_sample_file_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = complete_synthetic_cost(Path(temporary) / "synthetic")
            artifact = Path(args[0]["rows"][0]["samples"][0]["sample_artifact"]["path"])
            with artifact.open("ab") as handle: handle.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "SHA"): audit.audit_cost_dataset(*args, lambda: None, sealed_unix=0.)

    def test_missing_pending_completion_does_not_read_npz(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "PENDING.json"; path.write_text('{"status":"RUNNING"}', encoding="utf-8")
            _, result, parents = audit.metadata_gates(path, lambda: None)
        self.assertEqual(result["status"], "RUNNING"); self.assertEqual(parents, {})

    def test_native_route_metadata_disallows_seed_shared_D(self):
        route = "PCA_SHRUNK_NATIVE_s92601"
        entry = {"native_head_loaded_or_timed": True, "local_fit": True, "head_device": "cpu", "head_threads": 2, "canonical_prediction_batch_size": 32, "F_forward_in_this_route": False, "native_seed_D_cache_shared_inside_timer": True, "cached_local_D_used_for_timing": False, "fit_input_binding_hashes_inside_timer": True}
        with self.assertRaises(ValueError): audit.validate_route_metadata(route, entry, entry)


if __name__ == "__main__": unittest.main()
