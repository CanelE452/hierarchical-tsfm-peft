"""Synthetic JSON, payload and mocked lifecycle fixtures; no real science assets."""
from pathlib import Path
from types import SimpleNamespace
import ast
import copy
import inspect
import json
import sys

import numpy as np
import pytest

import run_matched_shrunk_native_v1 as runner


def write(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return runner.record(path)


def gate_fixture(tmp_path, monkeypatch):
    audit_source = tmp_path / "audit_local_decoder_shrinkage_v1.py"
    audit_source.write_text("# Synthetic audit source fixture\n", encoding="utf-8")
    head = tmp_path / "native_shrunk_decoder_head_v1.py"; head.write_text("# Synthetic head fixture\n", encoding="utf-8")
    plan = tmp_path / "PLAN_MATCHED_SHRUNK_NATIVE_KO_v1.md"; plan.write_text("Synthetic plan fixture\n", encoding="utf-8")
    monkeypatch.setattr(runner, "HERE", tmp_path)
    monkeypatch.setattr(runner, "HEAD_SHA", runner.record(head)["sha256"])
    monkeypatch.setattr(runner, "PLAN_SHA", runner.record(plan)["sha256"])
    producer_binding = write(tmp_path / "producer.json", {"status": "COMPLETE", "protected_TEST_accessed": False})
    parent_audit_binding = write(tmp_path / "parent_audit.json", {"status": "PASS", "protected_TEST_accessed": False, "producer": producer_binding})
    monkeypatch.setattr(runner, "PARENT_SHAS", {"producer": producer_binding["sha256"], "fit_target_audit": parent_audit_binding["sha256"]})
    locks = {dataset: {} for dataset in runner.DATASETS}
    for dataset in runner.DATASETS:
        for route in runner.ROUTES:
            label, rho = runner.SELECTED[(dataset, route)]
            locks[dataset][route] = write(tmp_path / (dataset + "_" + route + "_OLD_LOCK.json"), {
                "status": "LOCKED", "selected_candidate": label, "selected_rho": rho,
                "native_specific_selection": False, "development_or_oracle_scores_used_for_selection": False})
    shrink = {"status": "COMPLETE", "producer": producer_binding, "fit_target_audit": parent_audit_binding,
        "protected_TEST_accessed": False, "new_E_F_or_native_head_training": False, "new_E_F_native_head_forward_runs": 0,
        "new_backbone_training": False, "new_GPU_execution": False, "future_labels_used_for_decoder_fit": False,
        "native_specific_selection": False, "native_uses_same_Bolt_selected_candidate": True,
        "all_selection_lock_files_saved_before_any_DEV_candidate_compute": True,
        "development_positive_rho_sweep_executed": False, "independent_holdout": False,
        "original_source_cache_model_and_selection_SHA_unchanged": True, "failed_paths": [],
        "datasets": {dataset: {} for dataset in runner.DATASETS}, "selection_locks": locks}
    shrink_path = tmp_path / "shrink.json"; shrink_binding = write(shrink_path, shrink)
    monkeypatch.setattr(runner, "SHRINK_SHA", shrink_binding["sha256"])
    audit = {"status": "PASS", "diagnostic": shrink_binding, "totals": copy.deepcopy(runner.SHRINK_COUNTS),
        "protected_TEST_accessed": False, "new_F_E_head_forward_or_training": False,
        "future_labels_used_for_decoder_fit": False, "DEV_positive_rho_sweep_executed": False,
        "audit_source": runner.record(audit_source), "producer": producer_binding, "fit_target_audit": parent_audit_binding}
    audit_path = tmp_path / "audit.json"; audit_binding = write(audit_path, audit)
    monkeypatch.setattr(runner, "SHRINK_AUDIT_SHA", audit_binding["sha256"])
    return shrink_path, audit_path, shrink, audit


def test_gate_accepts_exact_bound_pass_without_science_io(tmp_path, monkeypatch):
    shrink_path, audit_path, _, _ = gate_fixture(tmp_path, monkeypatch)
    shrink, producer, bindings = runner.load_gates(shrink_path, audit_path)
    assert shrink["status"] == "COMPLETE" and producer["status"] == "COMPLETE"
    assert bindings["shrinkage_diagnostic"] == runner.record(shrink_path)


@pytest.mark.parametrize("tamper", ["hash", "status", "count", "futurefit", "rho", "head"])
def test_gate_rejects_changed_evidence_or_scope(tmp_path, monkeypatch, tamper):
    shrink_path, audit_path, shrink, audit = gate_fixture(tmp_path, monkeypatch)
    if tamper == "head":
        (tmp_path / "native_shrunk_decoder_head_v1.py").write_text("changed", encoding="utf-8")
    elif tamper == "rho":
        binding = shrink["selection_locks"]["robin"]["PCA_PINV"]
        path = Path(binding["path"]); value = runner.read(path); value["selected_rho"] = .01
        binding.update(write(path, value)); write(shrink_path, shrink)
        monkeypatch.setattr(runner, "SHRINK_SHA", runner.record(shrink_path)["sha256"])
        audit["diagnostic"] = runner.record(shrink_path); write(audit_path, audit)
        monkeypatch.setattr(runner, "SHRINK_AUDIT_SHA", runner.record(audit_path)["sha256"])
    else:
        if tamper == "hash": audit["diagnostic"]["sha256"] = "0" * 64
        elif tamper == "status": audit["status"] = "FAILED"
        elif tamper == "count": audit["totals"]["selection_locks_checked"] -= 1
        else: audit["future_labels_used_for_decoder_fit"] = True
        write(audit_path, audit)
        monkeypatch.setattr(runner, "SHRINK_AUDIT_SHA", runner.record(audit_path)["sha256"])
    with pytest.raises(ValueError): runner.load_gates(shrink_path, audit_path)


def registry_fixture(tmp_path):
    locks = {}
    for dataset in runner.DATASETS:
        for route in runner.ROUTES:
            for seed in runner.SEEDS:
                key = (dataset, route, str(seed))
                locks[key] = write(tmp_path / (dataset + "_" + route + "_" + str(seed) + ".json"), {
                    "status": "LOCKED", "dataset": dataset, "route": route, "seed": str(seed),
                    "selected_update": 100, "rho_selected_by_native": False, "DEV_predictions_computed_before_lock": False})
    return locks


def test_registry_requires_all_eight_conditions_before_development(tmp_path):
    locks = registry_fixture(tmp_path)
    assert len(runner.resolved_registry_before_development(locks)) == 8
    locks.pop(next(iter(locks)))
    with pytest.raises(ValueError, match="ALL_EIGHT"): runner.resolved_registry_before_development(locks)


def test_registry_supports_preserved_partial_path_but_does_not_select_it(tmp_path):
    locks = registry_fixture(tmp_path); key = next(iter(locks)); path = Path(locks[key]["path"])
    value = {"status": "FAILED_PATH_PRESERVED", "dataset": key[0], "route": key[1], "seed": key[2],
        "failure": {"type": "ValueError", "message": "Synthetic one-condition failure"}}
    locks[key] = write(path, value)
    resolved = runner.resolved_registry_before_development(locks)
    assert len(resolved) == 8 and sum(row["status"] == "LOCKED" for row in resolved.values()) == 7
    value["status"] = "TRAINING"; locks[key] = write(path, value)
    with pytest.raises(ValueError, match="SEALED_OR_EXPLICITLY_FAILED"): runner.resolved_registry_before_development(locks)


def test_registry_detects_mutated_lock_before_development(tmp_path):
    locks = registry_fixture(tmp_path); path = Path(next(iter(locks.values()))["path"])
    value = runner.read(path); value["selected_update"] = 600; write(path, value)
    with pytest.raises(ValueError, match="LOCK_CHANGED"): runner.resolved_registry_before_development(locks)


def small_payload():
    rng = np.random.default_rng(902)
    history = rng.normal(size=(3, 512, 3)).astype(np.float32)
    z = rng.normal(size=(3, 512, 2)).astype(np.float32)
    target = np.zeros((3, 48, 3), np.float32)
    values = {"history": history, "target": target, "observed": np.ones(target.shape, bool),
        "raw_last": history[:, -1].copy(), "origins": np.arange(3, dtype=np.int64)}
    candidate = {name: value.copy() for name, value in values.items() if name != "history"}; candidate["latent_history"] = z
    state = {"decoder_deployed_FP32": rng.normal(size=(3, 2, 3)).astype(np.float32)}
    cache = {"origins": values["origins"].copy(), "latent_history": z.copy()}
    return values, candidate, state, cache


def test_payload_uses_native_backend_D_and_exact_shared_seed_past():
    values, candidate, state, cache = small_payload()
    payload = runner.match_payload(values, candidate, state, cache, copy.deepcopy(cache), 3, 3, 2)
    assert payload["decoder"] is state["decoder_deployed_FP32"]
    assert payload["latent_history"] is candidate["latent_history"]


@pytest.mark.parametrize("tamper", ["seedpast", "origin", "futuretarget", "decoder", "current"])
def test_payload_rejects_wrong_lineage_dtype_or_origin_pairing(tamper):
    values, candidate, state, cache = small_payload(); second = copy.deepcopy(cache)
    if tamper == "seedpast": second["latent_history"][0, 0, 0] += np.float32(1)
    elif tamper == "origin": second["origins"] = second["origins"][::-1].copy()
    elif tamper == "futuretarget": candidate["target"][0, 0, 0] = np.float32(1)
    elif tamper == "decoder": state["decoder_deployed_FP32"] = state["decoder_deployed_FP32"].astype(np.float64)
    else: values["raw_last"][0, 0] += np.float32(1); candidate["raw_last"] = values["raw_last"].copy()
    with pytest.raises(ValueError): runner.match_payload(values, candidate, state, cache, second, 3, 3, 2)


def test_future_label_perturbation_checks_actual_fixed_cached_forward_inputs():
    values, candidate, state, cache = small_payload()
    payload = runner.match_payload(values, candidate, state, cache, cache, 3, 3, 2)
    called = []
    def predictor(model, z, current, decoder, batch_size):
        called.append((z.copy(), current.copy(), decoder.copy(), batch_size))
        return np.repeat(current[:, None], 48, axis=1)
    result = runner.future_label_perturbation(None, payload, predictor)
    assert result["exact_prediction_bytes_equal"] and len(called) == 2
    assert all(np.array_equal(a, b) for a, b in zip(called[0][:-1], called[1][:-1]))
    assert "does not replay" in result["scope"]


def test_controller_imports_no_scientific_runtime_and_places_barrier_before_dev():
    module = ast.parse(Path(runner.__file__).read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [node.module] if isinstance(node, ast.ImportFrom) else [alias.name for alias in node.names]
            assert not any(name and name.split(".")[0] in ("torch", "numpy", "chronos", "fixed_decoder_span_v1", "native_shrunk_decoder_head_v1") for name in names)
    source = inspect.getsource(runner.execute)
    assert source.index('event("ALL_EIGHT_CONDITION_LOCKS_AND_REGISTRY_SAVED"') < source.index('event("DEV_INPUT_LOAD_AFTER_ALL_LOCKS"') < source.index('event("DEV_NATIVE_PREDICTION_AFTER_ALL_LOCKS"')
    assert "updates=600, batch_size=4" in source and "new_encoder_decoder_fit" in source


def test_output_rejects_existing_or_outside_run_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "HERE", tmp_path)
    runs = tmp_path / "runs"; runs.mkdir(); existing = runs / "old"; existing.mkdir()
    assert runner.fresh_output_path(runs / "new") == (runs / "new").resolve()
    with pytest.raises(ValueError): runner.fresh_output_path(existing)
    with pytest.raises(ValueError): runner.fresh_output_path(tmp_path / "outside")


@pytest.mark.parametrize("fail_one", [False, True])
def test_mocked_full_lifecycle_seals_all_conditions_before_any_dev_read(tmp_path, monkeypatch, fail_one):
    """Real controller branching/saves with synthetic caches and no gradient training."""
    import torch
    import continuation_runtime_v1 as runtime
    import fixed_decoder_span_v1 as io
    import native_shrunk_decoder_head_v1 as native
    import decoder_metric_probe_v1 as metric
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setitem(sys.modules, "decoder_metric_probe_v1", metric)
    datasets = {name: (3, 2, development, 2) for name, (_, _, development, _) in runner.DATASETS.items()}
    monkeypatch.setattr(runner, "DATASETS", datasets)
    monkeypatch.setattr(runner, "HERE", tmp_path)
    old = tmp_path / "old"; old.mkdir(); monkeypatch.setattr(runner, "OLD", old)
    for name in ("native_shrunk_decoder_head_v1.py", "native_head_v1.py", "PLAN_MATCHED_SHRUNK_NATIVE_KO_v1.md",
                 "fixed_decoder_span_v1.py", "decoder_metric_probe_v1.py", "continuation_runtime_v1.py", "run_fit_target_comparison_v1.py"):
        (tmp_path / name).write_text("# Synthetic source snapshot fixture\n", encoding="utf-8")
    (old / "prepare_inputs.py").write_text("# Synthetic guard fixture\n", encoding="utf-8")
    output = tmp_path / "runs" / "synthetic_lifecycle"
    arrays, source, shrink, training_calls, loaded = {}, {"datasets": {}}, {"datasets": {}, "selection_locks": {}}, [], []
    def payload_record(name, value):
        arrays[name] = value
        return {"fixture_id": name}
    for dataset, (c, k, development, dev_count) in datasets.items():
        source["datasets"][dataset] = {"inputs": {}}
        shrink["datasets"][dataset] = {"splits": {}}
        shrink["selection_locks"][dataset] = {}
        for route in runner.ROUTES: shrink["selection_locks"][dataset][route] = {"fixture_id": dataset + route + "oldlock"}
        for split, count in (("fit", 256), ("tune", 64), (development, dev_count)):
            history = np.zeros((count, 512, c), np.float32)
            target = np.full((count, 48, c), .04, np.float32)
            values = {"history": history, "history_observed": np.ones(history.shape, bool), "target": target,
                "observed": np.ones(target.shape, bool), "raw_last": history[:, -1].copy(),
                "origins": np.arange(count, dtype=np.int64), "mean": np.zeros(c), "std": np.ones(c),
                "raw_forecast": np.zeros(target.shape, np.float32)}
            source["datasets"][dataset]["inputs"][split] = payload_record(dataset + ":" + split + ":raw", values)
            split_row = {"routes": {}}; shrink["datasets"][dataset]["splits"][split] = split_row
            for route in runner.ROUTES:
                label, rho = runner.SELECTED[(dataset, route)]
                z = np.zeros((count, 512, k), np.float32)
                cache = {"origins": values["origins"].copy(), "latent_history": z.copy()}
                cached = {key: value.copy() for key, value in values.items() if key in ("target", "observed", "raw_last", "origins")}
                cached.update(latent_history=z)
                for seed in runner.SEEDS: cached["NATIVE_s" + str(seed) + "_prediction"] = values["raw_forecast"].copy()
                base = dataset + ":" + split + ":" + route
                native_backend = {"status": "COMPLETE", "native_past_byte_exact_before_D_reuse": True,
                    "state_metadata": {"rho": rho, "fit_input_bindings": {key: {"sha256": io.array_sha(value)}
                        for key, value in (("history", history), ("history_observed", values["history_observed"]), ("latent_history", z))}},
                    "state_artifact": payload_record(base + ":state", {"decoder_deployed_FP32": np.ones((count, k, c), np.float32)}),
                    "artifact": payload_record(base + ":native", cached),
                    "cached_method_sources": {"NATIVE_s" + str(seed): payload_record(base + ":seed" + str(seed), copy.deepcopy(cache)) for seed in runner.SEEDS}}
                bolt_saved = {key: value.copy() for key, value in cached.items() if key in ("target", "observed", "raw_last", "origins")}
                bolt_saved["BOLT_prediction"] = values["raw_forecast"].copy()
                split_row["routes"][route] = {"selected_candidate": label, "positive_rho_candidates": {label: {"backends": {
                    "NATIVE_TORCH32": native_backend, "BOLT_NUMPY": {"artifact": payload_record(base + ":bolt", bolt_saved)}}}}}
    monkeypatch.setattr(runner, "load_gates", lambda *_: (shrink, source, {"shrinkage_diagnostic": {"fixture": True}}))
    def checked_payload(binding, meter, records):
        name = binding["fixture_id"]
        if ":val:" in name or ":eval:" in name:
            assert len(list(output.glob("*_SELECTION_LOCK.json"))) == 8
            registry = runner.read(output / "SELECTION_LOCK.json")
            assert len(registry["conditions"]) == 8 and registry["expected_conditions"] == 8
        loaded.append(name)
        return {name: value.copy() for name, value in arrays[name].items()}
    monkeypatch.setattr(io, "checked_payload", checked_payload)
    class FakeMeter:
        def __init__(self, *args, **kwargs): self.events = []
        def check(self, event=None): self.events.append(event)
        def protected_check(self): pass
        def close(self, status, error=None): return {"status": status, "synthetic_fixture": True}
    monkeypatch.setattr(runtime, "Meter", FakeMeter)
    monkeypatch.setitem(sys.modules, "prepare_inputs", SimpleNamespace(install_protected_test_guard=lambda: []))
    monkeypatch.setitem(sys.modules, "run_fit_target_comparison_v1", SimpleNamespace(prior_file_records=lambda *_: []))
    def predict(model, z, current, decoder, batch_size=32):
        future = z[:, -1:, :] + np.full((len(z), 48, z.shape[2]), model.offset, np.float32)
        prediction = current[:, None] + np.einsum("nhk,nkc->nhc", future-z[:, -1:, :], decoder)
        return prediction.astype(np.float32), future.astype(np.float32)
    monkeypatch.setattr(native, "predict_cached_components", predict)
    monkeypatch.setattr(native, "predict_cached_latents", lambda *args, **kwargs: predict(*args, **kwargs)[0])
    def fake_train(fit, tune, seed, progress, checkpoint, *, updates, batch_size):
        assert updates == 600 and batch_size == 4
        assert not (output / "SELECTION_LOCK.json").exists()
        training_calls.append(seed)
        if fail_one and len(training_calls) == 1: raise ValueError("Synthetic first condition failure")
        checkpoints = []
        for update in (0, 100, 200, 400, 600):
            model = SimpleNamespace(offset=np.float32(update/20000))
            prediction, _ = predict(model, tune["latent_history"], tune["raw_last"], tune["decoder"])
            score = native.score_predictions(prediction, tune["target"], tune["observed"])
            state = {"head.synthetic_fixture": torch.tensor([update], dtype=torch.float32)}
            checkpoint(update, state, score, prediction)
            checkpoints.append({"update": update, "tune": score})
        selected = native.select_checkpoint(checkpoints)
        return {"model": SimpleNamespace(offset=np.float32(selected["update"]/20000)),
            "metadata": {"selected_update": selected["update"]},
            "learning_curve": [{"update": update, "loss": .1, "gradient_norm_before_clip": .2} for update in range(1, 601)]}
    monkeypatch.setattr(native, "train_native_shrunk", fake_train)
    synthetic_audit = tmp_path / "synthetic_audit.json"
    write(synthetic_audit, {"status": "PASS", "synthetic_fixture": True})
    code = runner.execute("synthetic_shrink", synthetic_audit, output)
    report = runner.read(output / "RESULT.json")
    assert code == (2 if fail_one else 0) and len(training_calls) == 8
    assert report["native_training_fits"] == (7 if fail_one else 8)
    assert report["checkpoint_states_saved"] == (35 if fail_one else 40)
    assert report["completed_DEV_predictions"] == (7 if fail_one else 8)
    assert report["resolved_condition_lock_files"] == 8
    assert report["status"] == ("PARTIAL_COMPLETED" if fail_one else "COMPLETE")
    events = report["execution_events"]
    barrier = next(row["sequence"] for row in events if row["kind"] == "ALL_EIGHT_CONDITION_LOCKS_AND_REGISTRY_SAVED")
    assert all(row["sequence"] > barrier for row in events if row["kind"].startswith("DEV_"))
    assert report["cost_status"].startswith("NOT_MEASURED") and report["new_E_F_forward_or_training"] is False
    assert any(":val:" in name for name in loaded) and any(":eval:" in name for name in loaded)
