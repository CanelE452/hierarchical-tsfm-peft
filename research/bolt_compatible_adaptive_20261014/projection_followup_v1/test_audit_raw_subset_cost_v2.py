"""Synthetic cost records, physical/static-D and cached CPU head fixtures only."""
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import statistics
import tempfile
import unittest

import numpy as np

import audit_raw_subset_cost_v2 as audit
import whole_raw_subset_cost_v1 as timer_fixture
import run_raw_subset_cost_v1 as callback_fixture

audit.initialize_numpy()


def physical_fixture():
    rng = np.random.default_rng(776)
    x = rng.normal(size=(8, 512, 3)).astype(np.float32)
    mask = np.ones(x.shape, bool); mask[:, :3, 0] = False; mask[:, 10:12, 2] = False; x[~mask] = np.nan
    mean = np.array([1., 2., 3.], np.float32); std = np.array([2., 3., 4.], np.float32)
    history = audit.prior.causal_physical_preprocess(x, mask, mean, std)
    origins = np.arange(8, dtype=np.int64) + 512
    raw = {"history": history, "history_observed": mask, "raw_last": history[:, -1], "mean": mean, "std": std, "origins": origins}
    return {"history_physical": x, "history_observed": mask, "origins": origins}, raw


def memory():
    return {"RSS_bytes": 4096, "allocated_VRAM_bytes": 100, "reserved_VRAM_bytes": 200, "max_allocated_VRAM_bytes": 300, "max_reserved_VRAM_bytes": 400, "scope": "shared resident F/E/D/all native heads"}


def synthetic_cost(folder, *, broken=None):
    physical, raw = physical_fixture()
    p = np.repeat(raw["history"][:, -1:], 48, axis=1)
    predictions = {route: p.copy() for route in audit.ROUTES}
    callbacks = {route: lambda h, m: np.repeat(h[:, -1:], 48, axis=1) for route in audit.ROUTES}
    metadata = {route: {"synthetic_source": route} for route in audit.ROUTES}
    if broken is not None:
        counts = {1: 0, 8: 0}
        def break_after_two(h, m):
            counts[len(h)] += 1
            if counts[len(h)] == 6: raise ValueError("synthetic failure after two saved samples")
            return np.repeat(h[:, -1:], 48, axis=1)
        callbacks[broken] = break_after_two
    result = timer_fixture.measure_cost("synthetic", physical["history_physical"], physical["history_observed"], raw["mean"], raw["std"], raw["origins"], callbacks, predictions, metadata, folder, lambda: None, lambda: None, memory)
    return result, physical, raw, predictions, metadata


class RawSubsetCostAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(); cls.addClassCleanup(cls.temp.cleanup)
        cls.complete = synthetic_cost(Path(cls.temp.name) / "complete")
        cls.partial = synthetic_cost(Path(cls.temp.name) / "partial", broken=audit.ROUTES[-1])

    def test_complete_twenty_cells_two_hundred_samples_all_arrays_clock_and_memory(self):
        result, physical, raw, predictions, metadata = self.complete
        verified = audit.audit_cost_dataset(result, physical, raw, predictions, metadata, lambda: None, 0)
        self.assertEqual(verified["cells_checked"], 20); self.assertEqual(verified["recorded_timer_samples_checked"], 200)
        self.assertFalse(verified["actual_timers_or_F_remeasured"])
        json.dumps(verified, allow_nan=False)

    def test_partial_retains_successful_samples_and_other_declared_routes(self):
        result, physical, raw, predictions, metadata = self.partial
        verified = audit.audit_cost_dataset(result, physical, raw, predictions, metadata, lambda: None, 0)
        self.assertEqual(verified["cells_checked"], 18); self.assertEqual(verified["recorded_timer_samples_checked"], 184)
        self.assertEqual(verified["declared_cells_checked"], 20)
        self.assertEqual([row["samples_checked"] for row in verified["rows"] if row["route"] == audit.ROUTES[-1]], [2, 2])

    def test_schedule_true_seed_every_route_at_each_position_and_wrong_seed_rejected(self):
        schedule = audit.independent_schedule()
        self.assertEqual(schedule, timer_fixture.balanced_schedule(20261015))
        self.assertNotEqual(schedule, timer_fixture.balanced_schedule(24261015))
        for route in audit.ROUTES: self.assertEqual(sorted(order.index(route) for order in schedule), list(range(10)))
        args = copy.deepcopy(self.complete); args[0]["order_seed"] = 24261015
        with self.assertRaisesRegex(ValueError, "EXACT_TEN_LATIN"): audit.audit_cost_dataset(*args, lambda: None, 0)

    def test_paired_median_is_distinct_from_ratio_of_medians(self):
        raw = {"status": "PARTIAL_OR_FAILED_PATH_PRESERVED", "samples": [{"repeat": i, "perf_start_ns": 0, "perf_end_ns": int(v * 1e9)} for i, v in enumerate((1., 10., 100.))]}
        seconds = [2., 10., 300.]
        row = {"samples": [{"repeat": i, "perf_start_ns": 0, "perf_end_ns": int(v * 1e9)} for i, v in enumerate(seconds)], "repeats": 3, "warmup_each_route": 2, "seconds": seconds, "batch": 1, "status": "PARTIAL_OR_FAILED_PATH_PRESERVED", "median_seconds": 10., "median_seconds_per_window": 10., "paired_RAW_seconds": [1., 10., 100.], "paired_RAW_ratios": [2., 1., 3.], "paired_RAW_median_ratio": 2.}
        result = audit.recorded_times(row, raw)
        self.assertEqual(result["paired_RAW_median_ratio"], 2.); self.assertEqual(result["ratio_of_RAW_medians"], 1.)

    def test_missing_raw_pair_stays_none_and_is_not_fabricated(self):
        result = copy.deepcopy(self.partial[0]); raw = next(row for row in result["rows"] if row["route"] == "RAW_BOLT" and row["batch"] == 1)
        row = next(row for row in result["rows"] if row["route"] == audit.ROUTES[-1] and row["batch"] == 1)
        raw["samples"] = raw["samples"][1:]; raw["status"] = "PARTIAL_OR_FAILED_PATH_PRESERVED"
        row["paired_RAW_seconds"][0] = row["paired_RAW_ratios"][0] = row["paired_RAW_median_ratio"] = None
        self.assertIsNone(audit.recorded_times(row, raw)["paired_RAW_median_ratio"])

    def test_changed_ns_identity_or_positions_or_source_counts_fail(self):
        for field in ("ns", "position", "count", "route_metadata"):
            args = copy.deepcopy(self.complete); row = args[0]["rows"][0]
            if field == "ns": row["samples"][0]["perf_end_ns"] += 1
            elif field == "position": row["positions"][0] = (row["positions"][0] + 1) % 10
            elif field == "count": args[0]["recorded_timer_samples"] -= 1
            else: row["selected_metadata"] = {"synthetic_source": "wrong"}
            with self.subTest(field=field), self.assertRaises(ValueError): audit.audit_cost_dataset(*args, lambda: None, 0)

    def test_wrong_reference_head_or_saved_prediction_is_not_accepted(self):
        args = copy.deepcopy(self.complete); args[3][audit.ROUTES[-1]][0, 0, 0] += .2
        with self.assertRaisesRegex(ValueError, "exact_selected_scientific"): audit.audit_cost_dataset(*args, lambda: None, 0)

    def test_registered_parity_tolerance_cannot_be_relaxed(self):
        x = np.ones((2, 3), np.float32); y = x.copy(); y[0, 0] += .1
        recorded = timer_fixture.parity(x, y, atol=1., rtol=1.)
        with self.assertRaisesRegex(ValueError, "UNCHANGED_TOLERANCES"): audit.parity_checked(recorded, x, y)

    def test_memory_schema_shared_scope_and_peak_relationships(self):
        audit.memory_checked(memory())
        for field, value in (("scope", "exclusive per route"), ("reserved_VRAM_bytes", 1), ("max_allocated_VRAM_bytes", 1)):
            wrong = memory(); wrong[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): audit.memory_checked(wrong)

    def test_source_physical_model_callbacks_and_timer_boundaries(self):
        texts = [(audit.HERE / name).read_text(encoding="utf-8-sig") for name in ("whole_raw_subset_cost_v1.py", "run_raw_subset_cost_v1.py", "whole_matched_shrunk_native_cost_v1.py")]
        result = audit.source_contract(*texts)
        self.assertTrue(result["all_source_file_state_metadata_hashes_outside_timer"])
        for index, before, after in ((0, "REPEATS = 2, 10", "REPEATS = 2, 9"), (1, "z = encode_native(projection, indices, history)", "z = encode_native(projection, indices, history[::-1])")):
            changed = list(texts); changed[index] = changed[index].replace(before, after)
            self.assertNotEqual(changed[index], texts[index])
            with self.subTest(index=index), self.assertRaises(ValueError): audit.source_contract(*changed)

    def test_preflight_real_native_equations_static_D_and_two_seed_arrays(self):
        self.preflight("NATIVE_TORCH32")

    def test_preflight_cached_Bolt_future_static_D_without_F_reinference(self):
        self.preflight("BOLT_NUMPY")

    def preflight(self, backend):
        import torch
        from raw_subset_native_head_v1 import NativeFixedDecoder, predict_components
        physical, raw = physical_fixture(); indices = [0, 1]
        e = {"center": np.zeros(3, np.float32), "encoder": np.eye(3, dtype=np.float32)[:, indices], "selected_indices": np.array(indices, np.int64)}
        z = np.ascontiguousarray(raw["history"][..., indices]); d = np.array([[1., 0., .3], [0., 1., .7]], np.float32)
        fixed = {"origins": raw["origins"], "latent_history_" + backend: z}
        f = np.repeat(z[:, -1:], 48, axis=1) + np.float32(.1)
        models, states, predictions = {}, {}, {}
        for seed in ("92601", "92602"):
            model = NativeFixedDecoder(d)
            with torch.no_grad(): model.head.bias.fill_(.1 if seed == "92601" else .2)
            models[seed] = model; states[seed] = {key: value.detach().numpy().copy() for key, value in model.state_dict().items()}
            predictions[seed] = predict_components(model, z, raw["raw_last"])[0]
        bolt = raw["raw_last"][:, None] + (f - z[:, -1:]) @ d
        expected = {"BOLT": bolt} if backend == "BOLT_NUMPY" else predictions
        entry, arrays = callback_fixture.physical_backend_precheck(physical["history_physical"], physical["history_observed"], raw["mean"], raw["std"], raw, fixed, SimpleNamespace(**e), None, indices, d, backend, expected, lambda z: f, models, f)
        actual = audit.physical_precheck(entry, arrays, physical, raw, fixed, e, d, backend, states, expected, f)
        self.assertEqual(actual["independent_native_prediction_checks"], 2 if backend == "NATIVE_TORCH32" else 0)
        self.assertFalse(actual["F_future_reinference_by_auditor"])
        wrong = dict(arrays); wrong["fixed_decoder_FP32"] = d + np.float32(.1)
        with self.assertRaises(ValueError): audit.physical_precheck(entry, wrong, physical, raw, fixed, e, d, backend, states, expected, f)

    def test_running_producer_does_not_open_missing_assets(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "RUNNING.json"; path.write_text('{"status":"RUNNING"}', encoding="utf-8")
            _, _, parents = audit.metadata_gates(path, lambda: None)
        self.assertEqual(parents, {})


class JSONAndMarkdownParentRegressionTests(unittest.TestCase):
    def fixture(self, folder):
        cost = {"status": "COMPLETE", "routes": list(audit.ROUTES), "expected_cost_cells": 40, "expected_timer_samples": 400, "new_training_updates": 0, "hard_elapsed_or_cumulative_resource_caps": None, "fixed_FIT_decoder_loaded_before_timer": True, "all_source_file_state_metadata_hashes_outside_timer": True}
        for key in ("new_backbone_training", "new_encoder_subset_decoder_or_CP_selection", "per_origin_decoder_fit", "protected_TEST_accessed", "independent_holdout", "strict_K_only_compression"): cost[key] = False
        for key in audit.PARENT_KEYS:
            path = Path(folder) / (key + (".md" if key == "numeric_repair_addendum" else ".json"))
            path.write_text("# 실제 Markdown 형식\n\nJSON 객체가 아닌 수치 복구 근거.\n" if key == "numeric_repair_addendum" else json.dumps({"synthetic_parent": key}), encoding="utf-8")
            cost[key] = {"path": str(path.resolve()), "sha256": audit.prior.file_sha(path), "bytes": path.stat().st_size}
        return cost

    def test_original_v1_actual_MD_parse_error_reproduced_in_metadata_gate(self):
        import audit_raw_subset_cost_v1 as old
        with tempfile.TemporaryDirectory() as folder:
            cost = self.fixture(folder); path = Path(folder) / "cost.json"; path.write_text(json.dumps(cost), encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError): old.metadata_gates(path, lambda: None)

    def test_v2_JSON_parents_and_MD_path_SHA_bytes_checked_without_JSON_parse(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            cost = self.fixture(folder); opened = []; original = audit.prior.verified_json
            def observe(record, check):
                opened.append(Path(record["path"]).suffix)
                return original(record, check)
            with patch.object(audit.prior, "verified_json", side_effect=observe): parents = audit.load_parent_metadata(cost, lambda: None)
            self.assertEqual(opened, [".json"] * 6)
            self.assertEqual(set(parents), set(audit.PARENT_KEYS) - {"numeric_repair_addendum"})
            self.assertEqual(parents["raw_subset_diagnostic"], {"synthetic_parent": "raw_subset_diagnostic"})

    def test_wrong_JSON_SHA_MD_SHA_or_MD_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            original = self.fixture(folder)
            for what in ("json_sha", "md_sha", "md_bytes"):
                cost = copy.deepcopy(original)
                if what == "json_sha": cost["raw_subset_diagnostic"]["sha256"] = "wrong"
                elif what == "md_sha": cost["numeric_repair_addendum"]["sha256"] = "wrong"
                else: cost["numeric_repair_addendum"]["bytes"] += 1
                with self.subTest(what=what), self.assertRaises(ValueError): audit.load_parent_metadata(cost, lambda: None)

    def test_v2_only_parent_loader_changed_no_numeric_function_changes(self):
        import ast
        old = ast.parse((audit.HERE / "audit_raw_subset_cost_v1.py").read_text(encoding="utf-8"))
        new = ast.parse(Path(audit.__file__).read_text(encoding="utf-8"))
        old_functions = {node.name: node for node in old.body if isinstance(node, ast.FunctionDef)}
        new_functions = {node.name: node for node in new.body if isinstance(node, ast.FunctionDef)}
        self.assertEqual(set(new_functions) - set(old_functions), {"load_parent_metadata"})
        for name in old_functions:
            if name != "metadata_gates": self.assertEqual(ast.dump(old_functions[name], include_attributes=False), ast.dump(new_functions[name], include_attributes=False), name)


if __name__ == "__main__": unittest.main()



