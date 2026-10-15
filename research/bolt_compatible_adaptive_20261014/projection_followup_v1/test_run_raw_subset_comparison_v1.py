"""Synthetic exact-gate, encoder and twelve-lock controller fixtures only."""
from pathlib import Path
import ast
import inspect
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import run_raw_subset_comparison_v1 as run
from native_head_v1 import FrozenProjection
from projection_math_v1 import ProjectionState


def write(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return run.record(path)


def gate_fixture(tmp_path, monkeypatch):
    source_path, audit_path = tmp_path / "SOURCE.json", tmp_path / "AUDIT.json"
    source = {"status": "COMPLETE", "protected_TEST_accessed": False, "independent_holdout": False,
        "F_before": {"synthetic": 1}, "F_after": {"synthetic": 1}}
    write(source_path, source)
    code = tmp_path / "audit_source.py"; code.write_text("# independent synthetic auditor\n", encoding="utf-8")
    audit = {"status": "PASS", "producer": run.record(source_path), "protected_TEST_accessed": False,
        "audit_source": run.record(code), "totals": {"projection_states_checked": 4, "target_projection_fits_checked": 2,
            "reused_PCA_controls_checked": 2, "native_fits_checked": 8, "checkpoint_states_checked": 40,
            "cost_cells_checked": 20, "recorded_timer_samples_checked": 120, "saved_prediction_arrays_checked": 88,
            "output_subspace_diagnostic_cells_checked": 12, "fixed_D_real_method_inequalities_checked": 48, "bound_RAW_inequalities_checked": 0}}
    write(audit_path, audit)
    monkeypatch.setattr(run, "SOURCE_SHA", run.record(source_path)["sha256"])
    monkeypatch.setattr(run, "SOURCE_AUDIT_SHA", run.record(audit_path)["sha256"])
    return source_path, audit_path, source, audit


def test_exact_accepted_metadata_gate_precedes_scientific_IO(tmp_path, monkeypatch):
    source_path, audit_path, _, _ = gate_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(np, "load", lambda *_: pytest.fail("gate opened scientific arrays"))
    assert run.load_gates(source_path, audit_path)[0]["status"] == "COMPLETE"


@pytest.mark.parametrize("change", ["pending", "count", "hash", "TEST", "Fchanged"])
def test_gate_rejects_incomplete_or_modified_evidence(tmp_path, monkeypatch, change):
    source_path, audit_path, source, audit = gate_fixture(tmp_path, monkeypatch)
    if change == "pending": audit["status"] = "PENDING"
    elif change == "count": audit["totals"]["native_fits_checked"] = 7
    elif change == "hash": audit["producer"]["sha256"] = "0"*64
    else:
        if change == "TEST": source["protected_TEST_accessed"] = True
        else: source["F_after"] = {"synthetic": 2}
        write(source_path, source); monkeypatch.setattr(run, "SOURCE_SHA", run.record(source_path)["sha256"])
        audit["producer"] = run.record(source_path)
    write(audit_path, audit); monkeypatch.setattr(run, "SOURCE_AUDIT_SHA", run.record(audit_path)["sha256"])
    with pytest.raises(ValueError): run.load_gates(source_path, audit_path)


def lock_fixture(tmp_path):
    locks = {}
    for dataset in run.DATASETS:
        for space in run.SPACES:
            for seed in run.SEEDS:
                key = (dataset, space, str(seed))
                locks[key] = write(tmp_path / ("_".join(key)+".json"), {"dataset": dataset, "space": space, "seed": str(seed),
                    "status": "LOCKED", "selected_update": 100, "DEV_used_for_CP_selection": False})
    return locks


def test_all_twelve_locks_required_and_partial_path_is_resolved_not_selected(tmp_path):
    locks = lock_fixture(tmp_path)
    assert len(run.resolved_locks(locks)) == 12
    key = next(iter(locks)); value = run.read(locks[key]["path"])
    value.update(status="FAILED_PATH_PRESERVED", failure={"type": "ValueError", "message": "synthetic unavailable path"})
    locks[key] = write(locks[key]["path"], value)
    assert sum(item["status"] == "LOCKED" for item in run.resolved_locks(locks).values()) == 11
    del locks[key]
    with pytest.raises(ValueError, match="ALL_TWELVE"): run.resolved_locks(locks)


def test_changed_lock_before_DEV_is_rejected(tmp_path):
    locks = lock_fixture(tmp_path); item = next(iter(locks.values())); value = run.read(item["path"])
    value["selected_update"] = 600; write(item["path"], value)
    with pytest.raises(ValueError, match="LOCK_CHANGED"): run.resolved_locks(locks)


def test_subset_numpy_and_native_direct_index_preserve_raw_order_and_signed_zero():
    torch.set_num_threads(2)
    history = np.arange(65*512*4, dtype=np.float32).reshape(65, 512, 4)
    history[0, 0, 3] = np.float32(-0.)
    indices = [1, 3]
    a, b = run.encode_numpy(None, indices, history), run.encode_native(None, indices, history)
    expected = np.ascontiguousarray(history[..., indices])
    assert a.tobytes() == b.tobytes() == expected.tobytes()
    assert a.shape == (65, 512, 2)


def test_mixed_native_matches_original_Torch32_encode_with_fixed_E():
    torch.set_num_threads(2)
    rng = np.random.default_rng(971); history = rng.normal(size=(65, 512, 3)).astype(np.float32)
    state = ProjectionState(np.array([.1, .2, .3]), np.array([[1., .2], [.3, .8], [.2, .5]]), np.zeros((2, 3)), np.zeros(3), {})
    projection = FrozenProjection(state)
    expected = np.concatenate([projection.encode(torch.from_numpy(history[start:start+32])).detach().numpy() for start in range(0, 65, 32)])
    assert run.encode_native(projection, None, history).tobytes() == expected.tobytes()
    assert run.encode_numpy(state, None, history).tobytes() == state.encode(history).tobytes()


def test_scientific_split_and_barrier_contract_is_explicit_in_source():
    module = ast.parse(Path(run.__file__).read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [node.module] if isinstance(node, ast.ImportFrom) else [item.name for item in node.names]
            assert not any(name and name.split(".")[0] in ("numpy", "torch", "chronos", "prepare_inputs") for name in names)
    source = inspect.getsource(run.execute)
    assert source.index("load_gates(") < source.index("import numpy")
    assert source.index('event("ALL_TWELVE_LOCKS_AND_REGISTRY_SAVED"') < source.index('event("DEV_INPUT_LOAD_AFTER_ALL_TWELVE_LOCKS"')
    assert 'subset.greedy_raw_subset(fit["history"], fit["history_observed"], k)' in source
    assert 'subset.fit_global_observed_decoder(fit["history"], fit["history_observed"]' in source
    assert 'actual_subset_F_model_batch_calls' in source and 'full_C_forecast_slice_used_as_new_inference' in source
    assert 'per_origin_decoder_fit' in source and 'whole_deployment_cost_measured' in source
    assert 'FIT_TUNE_DEPENDENT_CONDITION' in source


@pytest.mark.parametrize("fail_one", [False, True])
def test_mocked_lifecycle_preserves_twelve_lock_barrier_and_path_isolation(tmp_path, monkeypatch, fail_one):
    """Only synthetic arrays/fake F and CP callbacks; no model/data/training job."""
    import continuation_runtime_v1 as runtime
    import fixed_decoder_span_v1 as io
    import decoder_metric_probe_v1 as metric
    import raw_subset_native_head_v1 as native
    import raw_subset_projection_v1 as subset_math
    import run_matched_shrunk_native_cost_v2 as cost_source
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setitem(sys.modules, "decoder_metric_probe_v1", metric)
    monkeypatch.setitem(sys.modules, "raw_subset_projection_v1", subset_math)
    monkeypatch.setattr(run, "HERE", tmp_path)
    base, old = tmp_path / "base", tmp_path / "old"; base.mkdir(); old.mkdir()
    monkeypatch.setattr(run, "BASE", base); monkeypatch.setattr(run, "OLD", old)
    monkeypatch.setattr(run, "DATASETS", {name: (3, 2, dev, 8) for name, (_, _, dev, _) in run.DATASETS.items()})
    for name in ("raw_subset_projection_v1.py", "raw_subset_native_head_v1.py", "native_head_v1.py", "projection_math_v1.py",
                 "fixed_decoder_span_v1.py", "decoder_metric_probe_v1.py", "run_matched_shrunk_native_cost_v2.py",
                 "PLAN_RAW_CHANNEL_SUBSET_KO_v1.md", "continuation_runtime_v1.py"):
        (tmp_path/name).write_text("# synthetic snapshot fixture\n", encoding="utf-8")
    (base/"cost_readout_v1.py").write_text("# synthetic snapshot fixture\n", encoding="utf-8")
    for name in ("run_condition_probes.py", "prepare_inputs.py", "matched_readout_contract_v1.py"):
        (old/name).write_text("# synthetic snapshot fixture\n", encoding="utf-8")
    output = tmp_path / "runs" / "synthetic_raw_subset"
    arrays, source, train_calls, F_calls = {}, {"datasets": {}, "F_before": {"synthetic": 1}, "F_after": {"synthetic": 1},
        "model_binary_before": {"path": "synthetic_weights", "sha256": "not-opened"}, "model_config_before": {"path": "synthetic_config", "sha256": "not-opened"}}, [], []
    def cached(name, value): arrays[name] = value; return {"fixture_id": name}
    bridge_entries = []
    for dataset, (c, k, dev, count) in run.DATASETS.items():
        row = {"inputs": {}, "projections": {}}; source["datasets"][dataset] = row
        for split, n in (("fit", 256), ("tune", 64), (dev, count)):
            history = np.zeros((n, 512, c), np.float32); target = np.full((n, 48, c), .04, np.float32)
            values = {"history": history, "history_observed": np.ones(history.shape, bool), "target": target,
                "observed": np.ones(target.shape, bool), "raw_last": history[:, -1].copy(), "origins": np.arange(n, dtype=np.int64),
                "mean": np.zeros(c, np.float32), "std": np.ones(c, np.float32), "raw_forecast": np.zeros(target.shape, np.float32)}
            row["inputs"][split] = cached(dataset+":"+split+":RAW", values)
            if split == "fit": bridge_entries.append({"dataset": dataset, "split": "fit", **cached(dataset+":fit:PHYSICAL", {
                **values, "columns": np.array(["a", "b", "c"])})})
        for space in ("PCA_PINV", "FIT_TARGET_PINV"):
            state = ProjectionState(np.zeros(c), np.eye(c, k), np.zeros((k, c)), np.zeros(c), {"channels": c, "K": k})
            condition = {"state_artifact": cached(dataset+space+":state", state.to_npz()), "bolt": {"artifacts": {}},
                "native": {str(seed): {"artifacts": {}} for seed in run.SEEDS}}
            row["projections"][space] = condition
            for split, n in (("fit", 256), ("tune", 64), (dev, count)):
                z = np.zeros((n, 512, k), np.float32); future = np.full((n, 48, k), .05, np.float32)
                stored = {"latent_history": z, "latent_future": future, "prediction": np.zeros((n, 48, c), np.float32), "origins": np.arange(n, dtype=np.int64)}
                condition["bolt"]["artifacts"][split] = cached(dataset+":"+split+":"+space+":BOLT", stored)
                for seed in run.SEEDS: condition["native"][str(seed)]["artifacts"][split] = cached(dataset+":"+split+":"+space+":NATIVE"+str(seed), stored)
    bridge_path = tmp_path/"BRIDGE.json"; write(bridge_path, {"artifacts": bridge_entries})
    monkeypatch.setattr(cost_source, "canonical_physical_bridge", lambda *_: {
        "canonical_input_bridge": {"path": str(bridge_path)}, "canonical_record": {"synthetic_fixture": True}})
    monkeypatch.setattr(run, "load_gates", lambda *_: (source, {}))
    def checked_payload(binding, meter, records):
        name = binding["fixture_id"]
        if ":val:" in name or ":eval:" in name:
            assert len(list(output.glob("*_s9260[12]_SELECTION_LOCK.json"))) == 12
            assert len(run.read(output/"SELECTION_REGISTRY.json")["conditions"]) == 12
        return {name: value.copy() for name, value in arrays[name].items()}
    monkeypatch.setattr(io, "checked_payload", checked_payload)
    class Meter:
        def __init__(self, *args, **kwargs): pass
        def check(self, *_): pass
        def protected_check(self): pass
        def close(self, status, error=None): return {"status": status, "synthetic_fixture": True}
    monkeypatch.setattr(runtime, "Meter", Meter)
    class FakeModel:
        def eval(self): return self
        def requires_grad_(self, *_): return self
    class FakeBolt:
        def __init__(self): self.model = FakeModel()
        def __call__(self, z): F_calls.append(tuple(z.shape)); return z[:, -1:].repeat(1, 48, 1)+.05
    probes = SimpleNamespace(import_old_engine=lambda: SimpleNamespace(ex=SimpleNamespace(previous=SimpleNamespace(FrozenBolt=FakeBolt)), tensor=torch.from_numpy),
        verified_file=lambda value: value, model_hashes=lambda _: {"synthetic": 1})
    monkeypatch.setitem(sys.modules, "run_condition_probes", probes)
    monkeypatch.setitem(sys.modules, "cost_readout_v1", SimpleNamespace(bolt_runtime_contract=lambda *_: {"synthetic": True}))
    monkeypatch.setitem(sys.modules, "matched_readout_contract_v1", SimpleNamespace(validate_raw_split=lambda *_: {}))
    monkeypatch.setitem(sys.modules, "run_fit_target_comparison_v1", SimpleNamespace(prior_file_records=lambda *_: []))
    monkeypatch.setitem(sys.modules, "prepare_inputs", SimpleNamespace(install_protected_test_guard=lambda: []))
    def fake_train(d, fit, tune, seed, progress, callback):
        assert not (output/"SELECTION_REGISTRY.json").exists()
        train_calls.append(seed)
        if fail_one and len(train_calls) == 1: raise ValueError("synthetic first native condition failure")
        model = native.NativeFixedDecoder(d).eval().requires_grad_(False)
        prediction, _ = native.predict_components(model, tune["latent_history"], tune["raw_last"])
        score = native.score_predictions(prediction, tune["target"], tune["observed"])
        for update in (0, 100, 200, 400, 600): callback(update, native.clone_state(model), score, prediction.copy())
        return {"model": model, "metadata": {"selected_update": 0}, "learning_curve": [{"update": n+1, "loss": .1, "gradient_norm_before_clip": .2} for n in range(600)]}
    monkeypatch.setattr(native, "train_native_fixed", fake_train)
    audit_path = tmp_path/"AUDIT.json"; write(audit_path, {"synthetic": True})
    result = run.execute("synthetic_source", audit_path, output)
    report = run.read(output/"RESULT.json")
    assert result == (2 if fail_one else 0) and report["status"] == ("PARTIAL_COMPLETED" if fail_one else "COMPLETE")
    assert len(train_calls) == 12 and report["native_training_fits"] == (11 if fail_one else 12)
    assert report["checkpoint_states_saved"] == (55 if fail_one else 60)
    assert report["saved_native_split_predictions"] == (33 if fail_one else 36)
    assert report["global_decoder_fits"] == 12 and report["actual_subset_F_split_inferences"] == 6
    assert report["reused_mixed_F_split_caches"] == 12 and all(shape[-1] == 2 for shape in F_calls)
    barrier = next(row["sequence"] for row in report["execution_events"] if row["kind"] == "ALL_TWELVE_LOCKS_AND_REGISTRY_SAVED")
    assert all(row["sequence"] > barrier for row in report["execution_events"] if row["kind"].startswith("DEV_"))
    assert report["cost_status"].startswith("NOT_MEASURED") and report["F_before"] == report["F_after"]
