"""Full saved synthetic whole-cost path plus physical/lag and tamper regressions."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import copy
import json
import tempfile
import unittest

import numpy as np

import audit_causal_lag_cost_v1 as audit
import run_causal_lag_cost_v1 as producer
import whole_causal_lag_cost_v1 as timer
from projection_math_v1 import ProjectionState

audit.initialize_numpy()


def memory():
    return {"RSS_bytes": 4096, "allocated_VRAM_bytes": 100, "reserved_VRAM_bytes": 200, "max_allocated_VRAM_bytes": 300, "max_reserved_VRAM_bytes": 400, "scope": "shared resident frozen F, six E and all fixed D; no method-exclusive allocation"}


def full_toy(folder, dataset="synthetic"):
    rng = np.random.default_rng(411)
    physical = rng.normal(size=(8, 512, 3)).astype(np.float32)
    observed = np.ones(physical.shape, bool); observed[:, :4, 0] = False; observed[:, 13:16, 2] = False; physical[~observed] = np.nan
    mean = np.array([1., 2., 3.], np.float32); std = np.array([2., 3., 4.], np.float32)
    history = audit.prior.causal_physical_preprocess(physical, observed, mean, std); origins = np.arange(8, dtype=np.int64) + 512
    raw = {"history": history, "history_observed": observed, "raw_last": history[:, -1], "origins": origins, "mean": mean, "std": std}
    inputs = {"history_physical": physical, "history_observed": observed, "origins": origins}
    def fake_F(z): return np.repeat(z[:, -1:], 48, axis=1) + np.float32(.05)
    predictors = {"RAW_BOLT": lambda h, m: fake_F(h)}; predictions = {"RAW_BOLT": fake_F(history)}
    metadata = {route: {"synthetic_route": route} for route in audit.ROUTES}; prechecks = {}
    for space in audit.SPACES:
        indices = [0] if space == "RAW_SUBSET" else None
        e = {"center": np.zeros(3, np.float64), "encoder": np.array([[1.], [0.], [0.]], np.float64), "selected_indices": np.array([0], np.int64)}
        state = ProjectionState(e["center"], e["encoder"], np.zeros((1, 3)), np.zeros(3), {}); z = audit.lag_audit.raw_audit.encode(history, e, "BOLT_NUMPY")
        fixed = {"origins": origins, "latent_history_BOLT_NUMPY": z}; cached = {"latent_future": fake_F(z)}
        decoders, values = {}, {}
        for condition, (lags, _) in audit.CONDITIONS.items():
            d = rng.normal(scale=.1, size=(len(lags), 3)).astype(np.float32)
            decoders[condition] = {"decoder_FP32": d, "lags": np.asarray(lags, np.int64)}
            callback = producer.bolt_predictor(state, indices, d, list(lags), fake_F)
            values[condition] = callback(history, observed)
            route = space + "_" + condition + "_BOLT"; predictors[route] = callback; predictions[route] = values[condition]
        entry, arrays = producer.physical_precheck(physical, observed, mean, std, raw, fixed, cached, state, indices, {name: (value["decoder_FP32"], value["lags"].tolist()) for name, value in decoders.items()}, values, fake_F)
        prechecks[space] = (entry, arrays, inputs, raw, fixed, cached, e, decoders, values)
    result = timer.measure_cost(dataset, physical, observed, mean, std, origins, predictors, predictions, metadata, folder, lambda: None, lambda: None, memory)
    return result, inputs, raw, predictions, metadata, prechecks


def full_orchestration_fixture(folder):
    folder.mkdir(); cost = {"status": "COMPLETE", "datasets": {}, "cost": {}, "source_records": [], "source_snapshots": [], "protected_prior_records": [], "input_records": [], "failed_paths": [], "fit_locks": {}, "completed_cost_cells": 52, "recorded_timer_samples": 676, "completed_cell_timer_samples": 676}
    science, subset = {"datasets": {}}, {"datasets": {}}
    original = {"source_parent_results": {}, "datasets": {}}
    parents = {"causal_lag_diagnostic": science, "raw_subset_diagnostic": subset, "fit_registry": {"sealed_unix": 0.}}
    def record(path): return {"path": str(path.resolve()), "sha256": audit.prior.file_sha(path)}
    def save_json(name, value):
        path = folder / (name + ".json"); path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8"); return record(path)
    def save_arrays(name, values): return timer.save_outputs(folder / (name + ".npz"), values)
    hashes = {"E": {}, "D_states": {}}
    for dataset, dev in (("robin", "val"), ("beijing_aotizhongxin", "eval")):
        value, physical, raw, predictions, _, prechecks = full_toy(folder / ("cost_" + dataset), dataset)
        raw.update(target=predictions["RAW_BOLT"].copy(), observed=np.ones(predictions["RAW_BOLT"].shape, bool), raw_forecast=predictions["RAW_BOLT"])
        raw_record = save_arrays(dataset + "_raw", raw)
        physical.update(mean=raw["mean"], std=raw["std"]); physical_record = {**save_arrays(dataset + "_physical", physical), "dataset": dataset, "split": dev}
        alias = {key: val for key, val in physical_record.items() if key != "arrays"}
        bridge_record = save_json(dataset + "_bridge", {"status": "COMPLETE", "actual_status": {"protected_test_opened": False}, "artifacts": [{**physical_record, "dataset": dataset, "split": dev}]})
        first_parent = save_json(dataset + "_parent", {"status": "COMPLETE", "protected_TEST_accessed": False, "bridge_provenance": {"input_bridge": bridge_record}})
        original["source_parent_results"][dataset] = first_parent; original["datasets"][dataset] = {"physical_bridge": alias}
        proof = {"original_first_parent": first_parent, "canonical_input_bridge": bridge_record, "original_alias": alias, "canonical_record": physical_record, "same_physical_file_path_and_SHA": True, "complete_array_manifest_preserved": True, "loader_file_SHA_member_shape_dtype_SHA_checks_relaxed": False, "numeric_tolerance_changed": False}
        srow = science["datasets"][dataset] = {"spaces": {}}
        urow = subset["datasets"][dataset] = {"raw_input_sources": {dev: raw_record}, "spaces": {}, "physical_bridge_canonical": physical_record, "canonical_physical_bridge_binding": proof}
        row = cost["datasets"][dataset] = {"canonical_physical_bridge": proof, "spaces": {}, "physical_prechecks": {}}
        locks = cost["fit_locks"][dataset] = {}
        for space, arguments in prechecks.items():
            pre, arrays, _, _, fixed, cached, e, decoders, by_condition = arguments
            e_record = save_arrays(dataset + space + "_E", e)
            fixed_record = save_arrays(dataset + space + "_fixed", {**raw, **fixed})
            cached_record = save_arrays(dataset + space + "_cached", {**cached, "origins": raw["origins"]})
            urow["spaces"][space] = {"E_state_artifact": e_record, "inputs": {dev: fixed_record}, "Bolt": {split: {"artifact": cached_record} for split in ("fit", "tune", dev)}}
            branch = srow["spaces"][space] = {"conditions": {}}; locks[space] = {}
            row["spaces"][space] = {"E_state": e_record, "indices": [0] if space == "RAW_SUBSET" else None, "available_conditions": list(audit.CONDITIONS)}
            pre = copy.deepcopy(pre); pre["artifact"] = save_arrays(dataset + space + "_precheck", arrays); row["physical_prechecks"][space] = pre
            state = {"center": e["center"], "encoder": e["encoder"], "decoder": np.zeros((1, 3)), "decoder_bias": np.zeros(3)}
            hashes["E"][str((dataset, space))] = {name: audit.array_record(array)["sha256"] for name, array in state.items()}
            for condition, d in decoders.items():
                lags, objective = audit.CONDITIONS[condition]
                d = {**d, "decoder_FP64": d["decoder_FP32"].astype(np.float64), "bias_FP64": np.zeros(3), "bias_FP32": np.zeros(3, np.float32)}
                d_record = save_arrays(dataset + space + condition + "_D", d)
                locks[space][condition] = save_json(dataset + space + condition + "_lock", {"sealed_unix": 0.})
                branch["conditions"][condition] = {"state_artifact": d_record, "lags": list(lags), "objective": objective, "splits": {}}
                for split in ("fit", "tune", dev):
                    branch["conditions"][condition]["splits"][split] = {"artifact": save_arrays(dataset + space + condition + split, {"ACTUAL_F_prediction_FP32": by_condition[condition], "origins": raw["origins"]})}
                hashes["D_states"][str((dataset, space, condition))] = {name: audit.array_record(array)["sha256"] for name, array in d.items()}
        metadata = audit.route_metadata(srow, urow, dev, locks); value["source_metadata"] = metadata
        for timer_row in value["rows"]: timer_row["selected_metadata"] = metadata[timer_row["route"]]
        path = folder / ("cost_" + dataset) / "RESULT.json"
        path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8"); cost["cost"][dataset] = value
    signature = {"old_algorithm": "synthetic-old", "bi_algorithm": "synthetic-new"}
    subset.update(F_before=signature, F_after=signature); original["F_before"] = signature
    science["frozen_F_parent_signatures"] = {"before": signature, "after": signature}; cost.update(F_before=signature, F_after=signature, fixed_states_before=hashes, fixed_states_after=hashes)
    for name in ("model_binary", "model_config"):
        path = folder / (name + ".txt"); path.write_text("synthetic immutable model metadata", encoding="utf-8")
        cost[name + "_before"] = cost[name + "_after"] = original[name + "_before"] = record(path)
    runtime_keys = ("evaluation", "all_backbone_parameters_frozen", "context_length", "prediction_length", "input_patch_size", "input_patch_stride", "median_index", "returned_horizon", "quantiles", "pipeline_predict_not_used", "normalizer_before", "normalizer_after", "normalizer_fixture_forward_parity")
    cost["runtime_behavior_contract"] = original["runtime_behavior_contract"] = {key: "synthetic" for key in runtime_keys}
    subset["source_producer"] = save_json("source_producer", original)
    parents["causal_lag_audit"] = {"audit_source": record(Path(audit.__file__))}
    for key in audit.PARENT_KEYS:
        cost[key] = save_json(key, parents.get(key, {"synthetic_metadata": key}))
    cost["cost_plan"] = cost["plan"]
    cost["resource_terminal"] = save_json("resource", {"status": "COMPLETE", "elapsed_or_cumulative_budget_caps": None})
    binding = save_json("RESULT", cost)
    return binding, cost, parents


class CausalLagCostAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory(); cls.addClassCleanup(cls.folder.cleanup)
        cls.toy = full_toy(Path(cls.folder.name) / "whole")

    def test_full_actual_synthetic_callbacks_physical_preflight_whole_timer_and_all_saved_arrays(self):
        value, physical, raw, predictions, metadata, prechecks = self.toy
        verified_prechecks = {space: audit.physical_precheck(*arguments) for space, arguments in prechecks.items()}
        result = audit.audit_cost_dataset(value, physical, raw, predictions, metadata, lambda: None, 0)
        self.assertEqual(result["cells_checked"], 26); self.assertEqual(result["recorded_timer_samples_checked"], 338)
        self.assertEqual(sum(row["fixed_D_predictions_checked"] for row in verified_prechecks.values()), 12)
        self.assertFalse(result["actual_timers_or_F_remeasured"])
        json.dumps({"prechecks": verified_prechecks, "cost": result}, allow_nan=False)

    def test_wrong_saved_science_reference_is_rejected(self):
        value, physical, raw, predictions, metadata, _ = copy.deepcopy(self.toy)
        predictions[audit.ROUTES[-1]][0, 0, 0] += np.float32(.1)
        with self.assertRaisesRegex(ValueError, "exact_selected_scientific_output_reference"):
            audit.audit_cost_dataset(value, physical, raw, predictions, metadata, lambda: None, 0)

    def test_wrong_lag_or_fixed_D_cannot_pass_physical_preflight(self):
        for what in ("lag", "decoder"):
            args = copy.deepcopy(self.toy[-1]["PCA"])
            if what == "lag": args[1]["L7_RECON_lags"][1] = 3
            else: args[1]["L7_RECON_decoder_FP32"][0, 0] += np.float32(.1)
            with self.subTest(what=what), self.assertRaises(ValueError): audit.physical_precheck(*args)

    def test_exact_seed_13_full_rotations_and_changed_order_rejected(self):
        self.assertEqual(audit.independent_schedule(), timer.balanced_schedule(20261015))
        for route in audit.ROUTES: self.assertEqual(sorted(order.index(route) for order in audit.independent_schedule()), list(range(13)))
        value, physical, raw, predictions, metadata, _ = copy.deepcopy(self.toy); value["order_seed"] = 24261015
        with self.assertRaisesRegex(ValueError, "EXACT_THIRTEEN_LATIN"): audit.audit_cost_dataset(value, physical, raw, predictions, metadata, lambda: None, 0)

    def test_wrong_ns_or_metadata_and_memory_scope_rejected(self):
        for what in ("timer", "metadata", "memory"):
            value, physical, raw, predictions, metadata, _ = copy.deepcopy(self.toy)
            if what == "timer": value["rows"][0]["samples"][0]["perf_end_ns"] += 1
            elif what == "metadata": metadata[audit.ROUTES[0]] = {"wrong": True}
            else: value["rows"][0]["memory"][0]["scope"] = "exclusive per method"
            with self.subTest(what=what), self.assertRaises(ValueError): audit.audit_cost_dataset(value, physical, raw, predictions, metadata, lambda: None, 0)

    def test_actual_E_F_lag_callback_and_timer_AST_contract(self):
        texts = [(audit.HERE / name).read_text(encoding="utf-8-sig") for name in ("whole_causal_lag_cost_v1.py", "run_causal_lag_cost_v1.py", "whole_raw_subset_cost_v1.py")]
        self.assertTrue(audit.source_contract(*texts)["thirteen_complete_order_rotations"])
        for index, before, after in ((0, "REPEATS = 2, 13", "REPEATS = 2, 12"), (1, "future = frozen_F(z)", "future = z[:, -48:]")):
            changed = list(texts); changed[index] = changed[index].replace(before, after)
            self.assertNotEqual(changed[index], texts[index])
            with self.subTest(index=index), self.assertRaises(ValueError): audit.source_contract(*changed)

    def test_text_MD_parent_and_JSON_parent_validation_no_format_confusion(self):
        with tempfile.TemporaryDirectory() as folder:
            cost = {}
            for key in audit.PARENT_KEYS:
                path = Path(folder) / (key + (".md" if key in ("parent_report", "plan") else ".json"))
                path.write_text("# 실제 Markdown\n\nJSON으로 읽지 않는 근거 문서." if path.suffix == ".md" else json.dumps({"synthetic": key}), encoding="utf-8")
                cost[key] = {"path": str(path.resolve()), "sha256": audit.prior.file_sha(path), "bytes": path.stat().st_size}
            parents = audit.load_parent_metadata(cost, lambda: None)
            self.assertIsNone(parents["parent_report"]); self.assertEqual(parents["causal_lag_audit"], {"synthetic": "causal_lag_audit"})
            for key in ("plan", "causal_lag_audit"):
                bad = copy.deepcopy(cost); bad[key]["sha256"] = "wrong"
                with self.subTest(key=key), self.assertRaises(ValueError): audit.load_parent_metadata(bad, lambda: None)

    def test_running_producer_does_not_open_assets(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "RUNNING.json"; path.write_text('{"status":"RUNNING"}', encoding="utf-8")
            self.assertEqual(audit.metadata_gates(path, lambda: None)[-1], {})

    def test_whole_two_dataset_orchestration_through_final_counts_and_strict_JSON(self):
        with tempfile.TemporaryDirectory() as folder:
            binding, cost, parents = full_orchestration_fixture(Path(folder) / "full")
            datasets = {"robin": (3, 1, "val", 8), "beijing_aotizhongxin": (3, 1, "eval", 8)}
            with patch.object(audit, "metadata_gates", return_value=(binding, cost, parents)), patch.object(audit, "DATASETS", datasets):
                report = audit.audit_result(binding["path"], SimpleNamespace(check=lambda: None))
            self.assertEqual(report["status"], "PASS"); self.assertEqual(report["totals"]["cost_cells_checked"], 52)
            self.assertEqual(report["totals"]["recorded_timer_samples_checked"], 676)
            self.assertEqual(report["totals"]["scientific_prediction_file_hashes_checked"], 72)
            json.dumps(report, allow_nan=False)


if __name__ == "__main__": unittest.main()
