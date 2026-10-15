"""Synthetic gate, backend, per-call refit and physical-decision fixtures only."""
from pathlib import Path
import ast
import copy
import inspect
import json

import numpy as np
import pytest
import torch

import run_matched_shrunk_native_cost_v1 as run
from local_decoder_shrinkage_v1 import fit_shrunk_local_decoder
from native_head_v1 import FrozenProjection
from native_shrunk_decoder_head_v1 import NativeShrunkForecaster, predict_cached_latents
from projection_math_v1 import ProjectionState


def write(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return run.record(path)


def gate_fixture(tmp_path, monkeypatch):
    import run_matched_shrunk_native_v1 as original
    parents = {}
    for name in ("shrinkage_diagnostic", "shrinkage_audit", "producer", "fit_target_audit"):
        parents[name] = write(tmp_path / (name + ".json"), {"synthetic": True})
    registry = write(tmp_path / "REGISTRY.json", {"complete_fit_tune_conditions": 8})
    matched = {"status": "COMPLETE", **parents, "selection_lock": registry, "condition_selection_locks": {},
        "failed_paths": [], "datasets": {}, "native_training_fits": 8, "completed_training_updates": 4800,
        "checkpoint_states_saved": 40, "completed_fit_tune_predictions": 16, "completed_DEV_predictions": 8,
        "completed_selection_locks": 8, "resolved_condition_lock_files": 8, "completed_source_backend_cells": 12,
        "future_label_perturbation_checks": 8,
        **{key: False for key in ("protected_TEST_accessed", "independent_holdout", "new_backbone_training", "new_E_F_forward_or_training",
            "new_encoder_decoder_fit", "new_GPU_execution", "rho_selection_repeated", "rho_selected_by_native",
            "same_Bolt_native_lambda_or_D_bytes_claimed", "native_original_static_head_weights_reused",
            "whole_matched_native_deployment_cost_measured", "old_static_or_global_rankK_floor_reused")},
        **{key: True for key in ("native_backend_cached_D_required", "shared_Bolt_TUNE_selected_rho",
            "all_eight_condition_locks_saved_before_any_new_DEV_prediction",
            "all_conditions_resolved_registry_saved_before_any_DEV_input_load", "original_source_cache_model_and_selection_SHA_unchanged")}}
    shrink = {"selection_locks": {}}; resolved = {}
    for dataset, (c, k, dev) in run.DATASETS.items():
        matched["datasets"][dataset] = {"channels": c, "K": k, "development_split": dev, "routes": {}}
        matched["condition_selection_locks"][dataset], shrink["selection_locks"][dataset] = {}, {}
        for route in run.PROJECTIONS:
            label, rho = original.SELECTED[(dataset, route)]
            old_lock = write(tmp_path / (dataset + route + "_OLD.json"), {"fixture": True})
            shrink["selection_locks"][dataset][route] = old_lock
            condition = {"rho": rho, "selected_candidate": label, "old_Bolt_rho_selection_lock": old_lock, "native": {}}
            matched["datasets"][dataset]["routes"][route] = condition
            matched["condition_selection_locks"][dataset][route] = {}
            for seed in run.SEEDS:
                artifact = {"path": str(tmp_path / "synthetic_not_opened_CP.npz"), "sha256": "metadata-only"}
                lock = {"status": "LOCKED", "selected_update": 100, "selected_tune": {"mse": .1}, "selected_state_artifact": artifact}
                key = (dataset, route, str(seed)); resolved[key] = lock
                matched["condition_selection_locks"][dataset][route][str(seed)] = write(tmp_path / (dataset + route + str(seed) + ".json"), lock)
                condition["native"][str(seed)] = {"status": "COMPLETE", "metadata": {
                    "updates": 600, "selected_update": 100, "selected_tune": lock["selected_tune"], "canonical_checkpoint_prediction_batch_size": 32},
                    "checkpoints": [{"update": 100, "state_artifact": artifact}]}
    result_path = tmp_path / "MATCHED.json"; write(result_path, matched)
    audit_source = tmp_path / "audit_matched_shrunk_native_v1.py"; audit_source.write_text("# synthetic independent fixture\n", encoding="utf-8")
    counts = {"native_fits_checked": 8, "selection_locks_checked": 8}
    audit = {"status": "PASS", "diagnostic": run.record(result_path), **parents,
        "totals": counts, "audit_source": run.record(audit_source), "protected_TEST_accessed": False}
    audit_path = tmp_path / "AUDIT.json"; write(audit_path, audit)
    monkeypatch.setattr(run, "MATCHED_SHA", run.record(result_path)["sha256"])
    monkeypatch.setattr(run, "MATCHED_AUDIT_SHA", run.record(audit_path)["sha256"])
    monkeypatch.setattr(run, "MATCHED_AUDIT_COUNTS", counts.copy())
    monkeypatch.setattr(original, "load_gates", lambda *_: (shrink, {"synthetic_producer": True}, parents))
    monkeypatch.setattr(original, "resolved_registry_before_development", lambda *_: resolved)
    return result_path, audit_path, matched, audit


def test_closed_or_pending_gate_precedes_scientific_import_and_file_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "MATCHED_AUDIT_SHA", None)
    with pytest.raises(ValueError, match="NOT_YET_REGISTERED"):
        run.execute("never_run", tmp_path / "absent.json", tmp_path / "absent_audit.json")
    assert list(tmp_path.iterdir()) == []


def test_positive_synthetic_gate_reads_no_scientific_npz(tmp_path, monkeypatch):
    result_path, audit_path, _, _ = gate_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(np, "load", lambda *_: pytest.fail("scientific IO before accepted gate"))
    matched, _, _, bindings, resolved = run.load_gates(result_path, audit_path)
    assert matched["status"] == "COMPLETE" and len(resolved) == 8
    assert bindings["matched_native_diagnostic"] == run.record(result_path)


@pytest.mark.parametrize("tamper", ["audit_status", "count", "binding", "native_reuse", "newrho", "headCP", "registry"])
def test_tampered_completion_selection_or_audit_rejected(tmp_path, monkeypatch, tamper):
    result_path, audit_path, matched, audit = gate_fixture(tmp_path, monkeypatch)
    if tamper == "audit_status": audit["status"] = "PENDING"
    elif tamper == "count": audit["totals"]["selection_locks_checked"] = 7
    elif tamper == "binding": audit["producer"]["sha256"] = "0" * 64
    elif tamper == "registry":
        path = matched["selection_lock"]["path"]; write(path, {"complete_fit_tune_conditions": 7})
    else:
        if tamper == "native_reuse": matched["native_original_static_head_weights_reused"] = True
        elif tamper == "newrho": matched["rho_selection_repeated"] = True
        else: matched["datasets"]["robin"]["routes"]["PCA_PINV"]["native"]["92601"]["metadata"]["selected_update"] = 600
        write(result_path, matched); monkeypatch.setattr(run, "MATCHED_SHA", run.record(result_path)["sha256"])
        audit["diagnostic"] = run.record(result_path)
    write(audit_path, audit); monkeypatch.setattr(run, "MATCHED_AUDIT_SHA", run.record(audit_path)["sha256"])
    with pytest.raises(ValueError): run.load_gates(result_path, audit_path)


@pytest.fixture
def toy():
    torch.set_num_threads(2)
    s = np.tile(np.array([-1., 1.], np.float32), 256)
    t = np.tile(np.array([-1., -1., 1., 1.], np.float32), 128)
    history = np.tile(np.stack((s, t, s+2*t), axis=1)[None], (8, 1, 1))
    observed = np.ones(history.shape, bool)
    state = ProjectionState(np.zeros(3), np.array([[1., 0.], [0., 1.], [0., 0.]]),
        np.array([[.5, 0., .5], [0., .5, 1.]]), np.zeros(3), {})
    projection = FrozenProjection(state)
    models = {str(seed): NativeShrunkForecaster(2).eval().requires_grad_(False) for seed in run.SEEDS}
    for index, model in enumerate(models.values()):
        with torch.no_grad(): model.head.bias.fill_(.1*(index+1))
    values = {"history": history, "history_observed": observed, "raw_last": history[:, -1].copy(), "origins": np.arange(512, 520, dtype=np.int64)}
    return history, observed, state, projection, models, values


def test_native_uses_actual_Torch_backend_and_canonical32(toy):
    history, _, _, projection, _, _ = toy
    actual = run.native_latents(projection, history)
    expected = projection.encode(torch.from_numpy(history)).detach().numpy()
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32
    assert "range(0, len(history), 32)" in inspect.getsource(run.native_latents)


def test_every_native_seed_callback_refits_D_with_original_mask_and_prior(toy, monkeypatch):
    history, observed, state, projection, models, _ = toy
    import local_decoder_shrinkage_v1 as shrink
    calls = []
    def fit(history, mask, z, prior, rho):
        calls.append((mask.copy(), prior.copy(), rho, z.copy()))
        return fit_shrunk_local_decoder(history, mask, z, prior, rho)
    monkeypatch.setattr(shrink, "fit_shrunk_local_decoder", fit)
    callbacks = [run.native_predictor(projection, state.decoder.astype(np.float32), .1, model) for model in models.values()]
    first, second, repeated = callbacks[0](history, observed), callbacks[1](history, observed), callbacks[0](history, observed)
    assert len(calls) == 3
    assert not np.array_equal(first, second)
    np.testing.assert_array_equal(first, repeated)
    for mask, prior, rho, z in calls:
        np.testing.assert_array_equal(mask, observed); np.testing.assert_array_equal(prior, state.decoder.astype(np.float32))
        np.testing.assert_array_equal(z, run.native_latents(projection, history)); assert rho == .1
    assert set(inspect.signature(callbacks[0]).parameters) == {"history", "history_observed"}


def test_bolt_callback_runs_real_supplied_F_and_refits_each_call(toy, monkeypatch):
    history, observed, state, _, _, _ = toy
    calls = []; fits = []
    def F(z): calls.append(z.copy()); return np.repeat(z[:, -1:] + np.float32(.2), 48, axis=1)
    import local_decoder_shrinkage_v1 as shrink
    def fit(*args): fits.append(args[-1]); return fit_shrunk_local_decoder(*args)
    monkeypatch.setattr(shrink, "fit_shrunk_local_decoder", fit)
    callback = run.bolt_predictor(state, F, .1)
    a, b = callback(history, observed), callback(history, observed)
    assert len(calls) == len(fits) == 2 and fits == [.1, .1]
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(calls[0], state.encode(history))


def precheck(toy, backend="NATIVE_TORCH32", metadata_change=None, physical_change=None, prediction_change=None):
    history, observed, state, projection, models, values = toy
    z = state.encode(history) if backend == "BOLT_NUMPY" else run.native_latents(projection, history)
    fitted = fit_shrunk_local_decoder(history, observed, z, state.decoder.astype(np.float32), .1)
    metadata = copy.deepcopy(fitted.metadata())
    if metadata_change: metadata_change(metadata)
    physical = history.copy()
    if physical_change: physical_change(physical)
    predictions = {seed: predict_cached_latents(model, z, values["raw_last"], fitted.decoder32) for seed, model in models.items()}
    if prediction_change: prediction_change(predictions)
    return run.physical_backend_precheck(physical, observed, np.zeros(3, np.float32), np.ones(3, np.float32),
        values, state, projection, backend, {"latent_history": z}, fitted.to_npz(), metadata, .1,
        models if backend == "NATIVE_TORCH32" else None, predictions if backend == "NATIVE_TORCH32" else None)


@pytest.mark.parametrize("backend", ["BOLT_NUMPY", "NATIVE_TORCH32"])
def test_physical_precheck_validates_coefficients_trace_lambda_and_decisions(toy, backend):
    result, payload = precheck(toy, backend)
    assert result["status"] == "PASS" and len(result["exact_decisions"]) == 6
    assert all(result["exact_decisions"].values()) and not result["decision_acceptance_by_output_tolerance_only"]
    assert len(result["matched_native_head_prediction_checks"]) == (2 if backend == "NATIVE_TORCH32" else 0)
    assert payload["decoder_deployed_FP32"].shape == (8, 2, 3)


def test_lambda_and_fallback_decision_mismatch_cannot_pass_by_output(toy):
    result, _ = precheck(toy, metadata_change=lambda meta: meta["actual_lambda"][0].__setitem__(0, meta["actual_lambda"][0][0]+.1))
    assert result["status"] == "FAILED_PRECHECK_PRESERVED" and not result["checks"]["actual_lambda"]["pass"]
    result, _ = precheck(toy, metadata_change=lambda meta: meta["zero_trace_prior_slope_retained"][0].__setitem__(0, True))
    assert result["status"] == "FAILED_PRECHECK_PRESERVED" and not result["exact_decisions"]["zero_trace_prior_slope_retained"]


def test_selected_head_prediction_mismatch_stops_native_preflight(toy):
    result, _ = precheck(toy, prediction_change=lambda values: values["92601"].__setitem__((0, 0, 0), values["92601"][0, 0, 0]+1))
    assert result["status"] == "FAILED_PRECHECK_PRESERVED"
    assert not result["matched_native_head_prediction_checks"]["92601"]["pass"]


def test_one_ULP_flat_past_changes_positive_lambda_decision_despite_input_tolerance(toy):
    _, _, state, projection, models, _ = toy
    history = np.ones((8, 512, 3), np.float32); observed = np.ones(history.shape, bool)
    values = {"history": history, "history_observed": observed, "raw_last": history[:, -1], "origins": np.arange(8, dtype=np.int64)}
    z = run.native_latents(projection, history)
    fitted = fit_shrunk_local_decoder(history, observed, z, state.decoder.astype(np.float32), .1)
    physical = history.copy(); physical[:, 0, 0] = np.nextafter(np.float32(1), np.float32(2))
    predictions = {seed: predict_cached_latents(model, z, values["raw_last"], fitted.decoder32) for seed, model in models.items()}
    result, _ = run.physical_backend_precheck(physical, observed, np.zeros(3, np.float32), np.ones(3, np.float32),
        values, state, projection, "NATIVE_TORCH32", {"latent_history": z}, fitted.to_npz(), fitted.metadata(), .1, models, predictions)
    assert result["checks"]["normalized_history"]["pass"] and result["checks"]["encoded_past"]["pass"]
    assert not result["exact_decisions"]["zero_trace_prior_slope_retained"]
    assert not result["exact_decisions"]["positive_lambda_solver_decision"] and result["status"] == "FAILED_PRECHECK_PRESERVED"


def test_cost_source_import_is_not_a_training_model_or_cache_execution():
    module = ast.parse(Path(run.__file__).read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [node.module] if isinstance(node, ast.ImportFrom) else [alias.name for alias in node.names]
            assert not any(name and name.split(".")[0] in ("torch", "numpy", "chronos", "prepare_inputs") for name in names)
    source = inspect.getsource(run.execute)
    assert source.index("load_gates(") < source.index("import numpy") < source.index("FrozenBolt()")
    assert "train_native_shrunk(" not in source and "native_seed_D_cache_shared_inside_timer" in source
