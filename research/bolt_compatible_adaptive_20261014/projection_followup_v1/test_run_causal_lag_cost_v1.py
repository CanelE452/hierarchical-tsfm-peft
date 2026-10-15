"""Synthetic closed-gate, real-callback and isolated controller lifecycle checks."""
from pathlib import Path
from types import SimpleNamespace
import ast
import inspect
import json
import sys

import numpy as np
import pytest
import torch

import causal_lag_decoder_v1 as math
import run_causal_lag_cost_v1 as run
import whole_causal_lag_cost_v1 as whole
from projection_math_v1 import ProjectionState


def write(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return run.record(path)


def gate_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "ROOT", tmp_path); monkeypatch.setattr(run, "HERE", tmp_path)
    pure = tmp_path/"causal_lag_decoder_v1.py"; pure.write_text("# synthetic immutable pure source\n")
    monkeypatch.setattr(run, "PURE_SHA", run.record(pure)["sha256"])
    source = write(tmp_path/"ORIGINAL.json", {"status": "COMPLETE", "F_before": {"value": 1}, "F_after": {"value": 1}})
    subset = write(tmp_path/"SUBSET.json", {"status": "COMPLETE", "F_before": {"value": 1}, "F_after": {"value": 1}, "source_producer": source})
    registry = {"status": "ALL_24_RESOLVED_BEFORE_TUNE_DEV", "failed_fit_paths": 0,
        "TUNE_DEV_NPZ_opened_before_seal": False, "conditions": []}
    science = {"status": "COMPLETE", "decoder_fits_completed": 24, "resolved_fit_lock_files": 24,
        "cached_D_predictions_completed": 72, "joint_output_diagnostics_completed": 72, "EY_diagnostics_completed": 72,
        "flat_diagnostics_completed": 72, "actual_F_cache_splits_reused": 18, "failed_paths": [],
        "all_24_locks_before_TUNE_DEV_NPZ": True, "protected_TEST_accessed": False, "new_F_calls": 0,
        "new_native_head_training": 0, "new_backbone_training": False, "independent_holdout": False,
        "TUNE_DEV_used_for_fit_or_selection": False, "oracle_used_for_fit_or_selection": False,
        "raw_subset_diagnostic": subset, "frozen_F_parent_signatures": {"before": {"value": 1}, "after": {"value": 1}},
        "source_records": [run.record(pure)], "datasets": {}}
    for key in ("raw_subset_audit", "parent_report", "parent_report_manifest", "parent_report_audit", "plan"):
        science[key] = write(tmp_path/(key+".json"), {"synthetic": True})
    for dataset in run.DATASETS:
        science["datasets"][dataset] = {"spaces": {}}
        for space in run.SPACES:
            parent = {"conditions": {}}; science["datasets"][dataset]["spaces"][space] = parent
            for condition, (lags, objective) in run.CONDITIONS.items():
                entry = {"status": "LOCKED_FIT_ONLY", "dataset": dataset, "space": space, "condition": condition,
                    "lags": lags, "objective": objective, "state_artifact": {"path": "not-opened.npz", "sha256": "1"*64},
                    "TUNE_DEV_used_for_fit_or_selection": False, "oracle_used_for_fit_or_selection": False}
                lock = write(tmp_path/(dataset+space+condition+"_LOCK.json"), entry)
                entry = {**entry, "fit_lock": lock, "splits": {split: {"status": "COMPLETE"} for split in ("fit", "tune", run.DATASETS[dataset][2])}}
                parent["conditions"][condition] = entry
                registry["conditions"].append({"dataset": dataset, "space": space, "condition": condition, "fit_lock": lock})
    science["fit_registry"] = write(tmp_path/"REGISTRY.json", registry)
    path = tmp_path/"SCIENCE.json"; write(path, science)
    code = tmp_path/"audit_causal_lag_objective_v1.py"; code.write_text("# synthetic auditor\n")
    audit = {"status": "PASS", "diagnostic": run.record(path), "totals": run.SCIENCE_AUDIT_COUNTS.copy(),
        "protected_TEST_accessed": False, "audit_source": run.record(code),
        **{key: science[key] for key in ("raw_subset_diagnostic", "raw_subset_audit", "parent_report", "parent_report_manifest", "parent_report_audit", "plan", "fit_registry")}}
    audit_path = tmp_path/"AUDIT.json"; write(audit_path, audit)
    monkeypatch.setattr(run, "SCIENCE_RESULT_SHA", run.record(path)["sha256"])
    monkeypatch.setattr(run, "SCIENCE_AUDIT_SHA", run.record(audit_path)["sha256"])
    return path, audit_path, science, audit


def test_default_gate_closed_before_any_asset_or_metadata_read(monkeypatch):
    monkeypatch.setattr(run, "SCIENCE_AUDIT_SHA", None)
    monkeypatch.setattr(run, "read", lambda *_: pytest.fail("closed gate opened a file"))
    with pytest.raises(ValueError, match="NOT_YET_REGISTERED"): run.load_gates("not-opened", "not-opened")


def test_metadata_only_exact_complete_PASS_and_24_fit_locks(tmp_path, monkeypatch):
    paths = gate_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(np, "load", lambda *_: pytest.fail("metadata gate opened scientific arrays"))
    science, _, _, binding, locks = run.load_gates(*paths[:2])
    assert science["decoder_fits_completed"] == 24 and len(locks) == 24
    assert binding["causal_lag_diagnostic"] == run.record(paths[0])


@pytest.mark.parametrize("change", ["count", "TEST", "newF", "barrier", "oracle", "registry", "lock", "lags", "pure_source", "audit_pending"])
def test_wrong_scope_or_unresolved_fit_path_is_rejected(tmp_path, monkeypatch, change):
    path, audit_path, science, audit = gate_fixture(tmp_path, monkeypatch)
    if change == "count": audit["totals"]["cached_D_split_prediction_arrays_checked"] = 71
    elif change == "TEST": science["protected_TEST_accessed"] = True
    elif change == "newF": science["new_F_calls"] = 1
    elif change == "barrier": science["all_24_locks_before_TUNE_DEV_NPZ"] = False
    elif change == "oracle": science["oracle_used_for_fit_or_selection"] = True
    elif change == "audit_pending": audit["status"] = "PENDING"
    elif change == "pure_source": (tmp_path/"causal_lag_decoder_v1.py").write_text("# changed")
    else:
        registry = run.read(science["fit_registry"]["path"])
        if change == "registry": registry["conditions"].pop()
        else:
            item = registry["conditions"][0]; lock = run.read(item["fit_lock"]["path"])
            if change == "lock": lock["status"] = "FAILED_PATH_PRESERVED"
            else: lock["lags"] = [0, 1]
            item["fit_lock"] = write(item["fit_lock"]["path"], lock)
        science["fit_registry"] = write(science["fit_registry"]["path"], registry)
        audit["fit_registry"] = science["fit_registry"]
    write(path, science); audit["diagnostic"] = run.record(path)
    write(audit_path, audit)
    monkeypatch.setattr(run, "SCIENCE_RESULT_SHA", run.record(path)["sha256"])
    monkeypatch.setattr(run, "SCIENCE_AUDIT_SHA", run.record(audit_path)["sha256"])
    with pytest.raises(ValueError): run.load_gates(path, audit_path)


def test_actual_bolt_callback_runs_F_each_time_and_matches_frozen_lag_formula():
    rng = np.random.default_rng(99)
    history = rng.normal(size=(2, 512, 3)).astype(np.float32)
    state = ProjectionState(np.zeros(3), np.array([[1., 0.], [0., 1.], [.2, .3]]), np.zeros((2, 3)), np.zeros(3), {})
    d = rng.normal(size=(6, 3)).astype(np.float32); calls = []
    def F(z):
        calls.append(z.copy()); return np.repeat(z[:, -1:], 48, axis=1)+np.float32(.1*len(calls))
    callback = run.bolt_predictor(state, None, d, [0, 1, 4], F)
    first = callback(history, np.ones(history.shape, bool)); second = callback(history, np.zeros(history.shape, bool))
    assert len(calls) == 2 and not np.array_equal(first, second)
    expected = math.predict(calls[0], np.repeat(calls[0][:, -1:], 48, axis=1)+np.float32(.1), history[:, -1], d, [0, 1, 4])
    assert first.tobytes() == expected.tobytes()


def test_physical_first8_precheck_checks_original_history_E_F_and_each_D():
    rng = np.random.default_rng(281)
    x = rng.normal(size=(8, 512, 3)).astype(np.float32); observed = rng.random(x.shape) > .1
    x[~observed] = np.nan
    mean, std = np.zeros(3, np.float32), np.ones(3, np.float32)
    h = whole.preprocess(x, observed, mean, std)
    state = ProjectionState(np.zeros(3), np.eye(3, 2), np.zeros((2, 3)), np.zeros(3), {})
    z = state.encode(h); future = np.repeat(z[:, -1:], 48, axis=1)+np.float32(.01)
    raw = {"history": h, "history_observed": observed, "raw_last": h[:, -1], "origins": np.arange(8, dtype=np.int64)}
    fixed = {"latent_history_BOLT_NUMPY": z, "origins": raw["origins"]}
    decoders = {"L0_RECON": (np.ones((2, 3), np.float32), [0]), "L7_FORECAST": (np.ones((6, 3), np.float32), [0, 1, 4])}
    refs = {name: math.predict(z, future, h[:, -1], d, lags) for name, (d, lags) in decoders.items()}
    count = []
    def F(z): count.append(1); return future.copy()
    result, arrays = run.physical_precheck(x, observed, mean, std, raw, fixed, {"latent_future": future}, state, None, decoders, refs, F)
    assert result["status"] == "PASS" and len(count) == 1
    assert set(result["prediction_checks"]) == set(decoders) and "L7_FORECAST_decoder_FP32" in arrays
    changed, _ = run.physical_precheck(x, observed, mean, std, raw, fixed, {"latent_future": future+.1}, state, None, decoders, refs, F)
    assert not changed["input_backend_pass"] and changed["status"] == "FAILED_PRECHECK_PRESERVED"


def test_source_imports_and_timed_callback_have_no_fit_or_cached_F_return():
    module = ast.parse(Path(run.__file__).read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [node.module] if isinstance(node, ast.ImportFrom) else [x.name for x in node.names]
            assert not any(name and name.split(".")[0] in ("numpy", "torch", "chronos", "causal_lag_decoder_v1") for name in names)
    callback = inspect.getsource(run.bolt_predictor)
    assert "frozen_F(z)" in callback and "return predict(" in callback and "fit_" not in callback
    source = inspect.getsource(run.execute)
    assert source.index("load_gates(") < source.index("import numpy") < source.index("FrozenBolt()")
    assert "canonical_physical_bridge" in source and 'arrays["decoder_FP32"]' in source


@pytest.mark.parametrize("fail_one", [False, True])
def test_mocked_controller_lifecycle_keeps_other_routes_after_one_D_failure(tmp_path, monkeypatch, fail_one):
    import continuation_runtime_v1 as runtime
    import fixed_decoder_span_v1 as io
    import run_matched_shrunk_native_cost_v2 as canonical
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(run, "HERE", tmp_path); monkeypatch.setattr(run, "ROOT", tmp_path)
    base, old = tmp_path/"base", tmp_path/"old"; base.mkdir(); old.mkdir()
    monkeypatch.setattr(run, "BASE", base); monkeypatch.setattr(run, "OLD", old)
    monkeypatch.setattr(run, "DATASETS", {name: (3, 1, dev, 8) for name, (_, _, dev, _) in run.DATASETS.items()})
    for name in ("whole_causal_lag_cost_v1.py", "causal_lag_decoder_v1.py", "run_causal_lag_objective_v1.py",
            "run_raw_subset_comparison_v2.py", "raw_subset_projection_v2.py", "raw_subset_projection_v1.py", "projection_math_v1.py",
            "fixed_decoder_span_v1.py", "run_matched_shrunk_native_cost_v2.py", "PLAN_CAUSAL_LAG_WHOLE_COST_KO_v1.md", "continuation_runtime_v1.py"):
        (tmp_path/name).write_text("# synthetic immutable snapshot\n")
    (base/"cost_readout_v1.py").write_text("# synthetic snapshot\n")
    for name in ("run_condition_probes.py", "prepare_inputs.py"): (old/name).write_text("# synthetic snapshot\n")
    arrays, locks = {}, {}
    def cached(name, value):
        path = tmp_path/(name+".npz"); path.write_bytes(b"synthetic array binding file; no model asset")
        binding = {**run.record(path), "fixture_id": name}; arrays[name] = value; return binding
    source = {"F_before": {"frozen": 1}, "model_binary_before": {"synthetic": "no weight opened"}, "model_config_before": {"synthetic": "no config opened"}}
    subset, science = {"datasets": {}}, {"datasets": {}}
    for dataset, (c, k, dev, _) in run.DATASETS.items():
        history = np.zeros((8, 512, c), np.float32); observed = np.ones(history.shape, bool)
        origins = np.arange(8, dtype=np.int64); future = np.zeros((8, 48, k), np.float32)
        raw = {"history": history, "history_observed": observed, "origins": origins, "raw_last": history[:, -1],
            "mean": np.zeros(c, np.float32), "std": np.ones(c, np.float32), "raw_forecast": np.zeros((8, 48, c), np.float32)}
        physical = cached(dataset+"_physical", {"history_physical": history, "history_observed": observed,
            "origins": origins, "mean": raw["mean"], "std": raw["std"]})
        subset["datasets"][dataset] = {"raw_input_sources": {dev: cached(dataset+"_raw", raw)}, "physical_bridge_canonical": physical, "spaces": {}}
        science["datasets"][dataset] = {"spaces": {}}
        for space in run.SPACES:
            subset["datasets"][dataset]["spaces"][space] = {"E_state_artifact": cached(dataset+space+"_E", {
                "center": np.zeros(c), "encoder": np.eye(c, k), "selected_indices": np.array([0] if space == "RAW_SUBSET" else [], np.int64)}),
                "inputs": {dev: cached(dataset+space+"_fixed", {"latent_history_BOLT_NUMPY": history[:, :, :k], "origins": origins})},
                "Bolt": {dev: {"artifact": cached(dataset+space+"_F", {"latent_future": future, "origins": origins})}}}
            condition_entries = {}; science["datasets"][dataset]["spaces"][space] = {"conditions": condition_entries}
            for condition, (lags, _) in run.CONDITIONS.items():
                state = cached(dataset+space+condition+"_D", {"decoder_FP64": np.ones((len(lags)*k, c)),
                    "decoder_FP32": np.ones((len(lags)*k, c), np.float32), "bias_FP64": np.zeros(c), "bias_FP32": np.zeros(c, np.float32), "lags": np.array(lags, np.int64)})
                lock_path = tmp_path/(dataset+space+condition+"_LOCK.json")
                locks[(dataset, space, condition)] = write(lock_path, {"state_artifact": state})
                artifact = cached(dataset+space+condition+"_prediction", {"ACTUAL_F_prediction_FP32": raw["raw_forecast"].copy(), "origins": origins})
                condition_entries[condition] = {"state_artifact": state, "splits": {dev: {"artifact": artifact}}}
    monkeypatch.setattr(run, "load_gates", lambda *_: (science, subset, source, {}, locks))
    monkeypatch.setattr(run, "bound", lambda item, expected=None: Path(item["path"]))
    monkeypatch.setattr(canonical, "canonical_physical_bridge", lambda source, data: {"canonical_record": subset["datasets"][data]["physical_bridge_canonical"]})
    def load(binding, meter, checked):
        if fail_one and binding["fixture_id"] == "robinPCAL0_RECON_D": raise ValueError("synthetic one D load failure")
        return {key: value.copy() for key, value in arrays[binding["fixture_id"]].items()}
    monkeypatch.setattr(io, "checked_payload", load)
    class Meter:
        def __init__(self, *a, **kw): pass
        def check(self, *a): pass
        def protected_check(self): pass
        def close(self, status, error=None): return {"status": status, "synthetic": True}
    monkeypatch.setattr(runtime, "Meter", Meter)
    F_calls = []
    class FakeModel:
        def eval(self): return self
        def requires_grad_(self, *a): return self
    class FakeBolt:
        def __init__(self): self.model = FakeModel()
        def __call__(self, z): F_calls.append(tuple(z.shape)); return z[:, -1:].repeat(1, 48, 1)
    monkeypatch.setitem(sys.modules, "run_condition_probes", SimpleNamespace(import_old_engine=lambda: SimpleNamespace(
        ex=SimpleNamespace(previous=SimpleNamespace(FrozenBolt=FakeBolt)), tensor=torch.from_numpy),
        verified_file=lambda x: x, model_hashes=lambda x: {"frozen": 1}))
    monkeypatch.setitem(sys.modules, "cost_readout_v1", SimpleNamespace(bolt_runtime_contract=lambda *a: {"synthetic": True}))
    monkeypatch.setitem(sys.modules, "run_fit_target_comparison_v1", SimpleNamespace(prior_file_records=lambda *a: []))
    monkeypatch.setitem(sys.modules, "prepare_inputs", SimpleNamespace(install_protected_test_guard=lambda: []))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    def measure(dataset, physical, mask, mean, std, origins, callbacks, references, metadata, output, check, sync, memory, unavailable):
        rows = []
        assert set(callbacks) == set(whole.ROUTES)-set(unavailable)
        for batch in (1, 8):
            for route in whole.ROUTES:
                if route not in unavailable:
                    prediction = callbacks[route](whole.preprocess(physical[:batch], mask[:batch], mean, std), mask[:batch])
                    np.testing.assert_array_equal(prediction, references[route][:batch])
                rows.append({"route": route, "batch": batch, "status": "COMPLETE" if route not in unavailable else "FAILED_PATH_PRESERVED",
                    "samples": [{"synthetic_not_actual_timer": True}]*13 if route not in unavailable else []})
        return {"status": "PARTIAL_COMPLETED" if unavailable else "COMPLETE", "rows": rows, "failed_paths": []}
    monkeypatch.setattr(whole, "measure_cost", measure)
    output = tmp_path/"runs"/"synthetic_controller"
    audit = tmp_path/"AUDIT.json"; write(audit, {"synthetic": True})
    code = run.execute(output.name, "not-read-source", audit)
    result = run.read(output/"RESULT.json")
    assert code == (2 if fail_one else 0) and result["status"] == ("PARTIAL_COMPLETED" if fail_one else "COMPLETE")
    assert result["completed_cost_cells"] == (50 if fail_one else 52)
    assert result["recorded_timer_samples"] == (650 if fail_one else 676)
    assert result["F_before"] == result["F_after"] and result["fixed_states_before"] == result["fixed_states_after"]
    assert len(F_calls) > 50 and not result["native_heads_loaded_or_timed"]
