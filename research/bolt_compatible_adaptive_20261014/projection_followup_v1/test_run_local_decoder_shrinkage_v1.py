"""Synthetic producer gates, frozen-cache contracts and selection ordering only."""
from pathlib import Path
import copy
import json
import sys
import types
from unittest.mock import patch

import numpy as np
import pytest

import run_local_decoder_shrinkage_v1 as run
from local_decoder_shrinkage_v1 import fit_shrunk_local_decoder
from local_past_decoder_v1 import LocalPastDecoder
from decoder_metric_probe_v1 import masked_scores


def write(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


@pytest.fixture
def gates(tmp_path, monkeypatch):
    code = tmp_path / "audit_local_past_decoder_probe_v1.py"
    code.write_text("synthetic independent audit source only", encoding="utf-8")
    parents, shas = {}, {}
    for index, name in enumerate(run.PRIOR_SHAS):
        path = tmp_path / (name + ".json")
        write(path, {"status": "COMPLETE" if index % 2 == 0 else "PASS", "protected_TEST_accessed": False})
        parents[name], shas[name] = run.record(path), run.record(path)["sha256"]
    local_path, audit_path = tmp_path / "local.json", tmp_path / "audit.json"
    local = {"status": "COMPLETE", **parents, "completed_method_pairs": 36, "completed_backend_batches": 24,
        "completed_window_matrix_fits": 3264, "completed_channel_solver_calls": 44304,
        "completed_no_observation_fallbacks": 0, "native_seed_D_reuses": 1632,
        "new_E_F_native_head_forward_runs": 0, "protected_TEST_accessed": False,
        "new_E_F_or_native_head_training": False, "new_backbone_training": False, "new_GPU_execution": False,
        "original_source_cache_selection_SHA_unchanged": True, "future_labels_used_for_local_fit": False,
        "method_checkpoint_threshold_selection_used": False, "whole_local_deployment_cost_measured": False,
        "independent_holdout": False, "failed_paths": []}
    write(local_path, local)
    audit = {"status": "PASS", **parents, "diagnostic": run.record(local_path), "totals": run.LOCAL_AUDIT_COUNTS.copy(),
        "audit_source": run.record(code), "protected_TEST_accessed": False,
        "new_F_E_head_forward_or_training": False, "old_static_or_global_rankK_lower_bound_reused": False}
    write(audit_path, audit)
    monkeypatch.setattr(run, "LOCAL_SHA", run.record(local_path)["sha256"])
    monkeypatch.setattr(run, "LOCAL_AUDIT_SHA", run.record(audit_path)["sha256"])
    monkeypatch.setattr(run, "PRIOR_SHAS", shas)
    return local_path, audit_path


def refresh(gates, monkeypatch):
    local, audit = gates
    value = run.read(audit)
    value["diagnostic"] = run.record(local)
    write(audit, value)
    monkeypatch.setattr(run, "LOCAL_SHA", run.record(local)["sha256"])
    monkeypatch.setattr(run, "LOCAL_AUDIT_SHA", run.record(audit)["sha256"])


def test_complete_exact_metadata_gate_has_no_array_IO(gates):
    with patch.object(np, "load", side_effect=AssertionError("No actual arrays in gate fixture")):
        local, producer, bindings = run.load_gates(*gates)
    assert local["status"] == producer["status"] == "COMPLETE"
    assert len(bindings) == 8


@pytest.mark.parametrize("index", [0, 1])
def test_pending_gate_precedes_science_import_cache_or_output(gates, tmp_path, index):
    value = run.read(gates[index])
    value["status"] = "RUNNING"
    write(gates[index], value)
    imported, original = [], __import__
    def refuse(name, *args, **kwargs):
        if name.split(".")[0] in ("numpy", "torch", "fixed_decoder_span_v1", "local_decoder_shrinkage_v1", "continuation_runtime_v1"):
            imported.append(name)
            raise AssertionError("Science must wait for accepted metadata")
        return original(name, *args, **kwargs)
    output = tmp_path / "never_created"
    with patch("builtins.__import__", side_effect=refuse):
        with pytest.raises(ValueError, match="BEFORE_SCIENCE"):
            run.execute(*gates, output)
    assert imported == [] and not output.exists()


@pytest.mark.parametrize("index", [0, 1])
def test_exact_immutable_SHA_change_is_rejected(gates, index):
    gates[index].write_text(gates[index].read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="PASS_SHA"):
        run.load_gates(*gates)


@pytest.mark.parametrize("key", list(run.LOCAL_AUDIT_COUNTS))
def test_every_independent_coverage_count_required(gates, monkeypatch, key):
    value = run.read(gates[1]); value["totals"][key] += 1
    write(gates[1], value); refresh(gates, monkeypatch)
    with pytest.raises(ValueError, match="AUDIT_COVERAGE"):
        run.load_gates(*gates)


def test_TEST_false_and_no_new_forward_training_are_required(gates, monkeypatch):
    value = run.read(gates[0]); value["protected_TEST_accessed"] = True
    write(gates[0], value); refresh(gates, monkeypatch)
    with pytest.raises(ValueError, match="NO_TEST_LOCAL_D_SCOPE"):
        run.load_gates(*gates)


def test_immutable_parent_json_bytes_required(gates):
    parent = run.read(gates[0])["producer"]
    Path(parent["path"]).write_text(Path(parent["path"]).read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="PARENT_JSON_BINDING"):
        run.load_gates(*gates)


def test_independent_audit_source_bytes_required(gates):
    code = run.read(gates[1])["audit_source"]["path"]
    Path(code).write_text("changed synthetic code", encoding="utf-8")
    with pytest.raises(ValueError, match="AUDIT_SOURCE"):
        run.load_gates(*gates)


def candidates(scores=None):
    scores = scores or dict(zip(run.CANDIDATE_ORDER, (9., 8., 7., 6., 5., 4., 3.)))
    return {name: {"methods": {"BOLT": {"status": "COMPLETE", "scores_all_observed": {
        "normalized": {"mse": value, "mae": 100. - value}}},
        "NATIVE_s92601": {"status": "COMPLETE", "scores_all_observed": {"normalized": {"mse": 100. - value}}},
        "NATIVE_s92602": {"status": "COMPLETE", "scores_all_observed": {"normalized": {"mse": -999.}}}}}
        for name, value in scores.items()}


def test_only_Bolt_TUNE_primary_selects_and_native_MAE_do_not():
    lock = run.select_bolt_only(candidates())
    assert lock["selected_candidate"] == "RHO_10" and lock["selected_rho"] == 10.
    assert lock["native_specific_selection"] is False
    assert lock["MAE_used_for_selection"] is False
    assert lock["development_or_oracle_scores_used_for_selection"] is False


@pytest.mark.parametrize("first", ["STATIC", "LOCAL", "RHO_0.001", "RHO_1"])
def test_predeclared_tie_order(first):
    offset = run.CANDIDATE_ORDER.index(first)
    scores = {name: 1. if index >= offset else 2. for index, name in enumerate(run.CANDIDATE_ORDER)}
    assert run.select_bolt_only(candidates(scores))["selected_candidate"] == first


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -.1, True])
def test_missing_or_invalid_candidate_cannot_be_silently_excluded(bad):
    values = candidates()
    values["RHO_0.1"]["methods"]["BOLT"]["scores_all_observed"]["normalized"]["mse"] = bad
    with pytest.raises(ValueError, match="ALL_SEVEN"):
        run.select_bolt_only(values)


def selection_records(tmp_path, skipped=None):
    records = {}
    for dataset in run.DATASETS:
        for route in run.ROUTES:
            path = tmp_path / (dataset + route + ".json")
            value = {"status": "LOCKED", "selected_candidate": "RHO_0.01", "native_specific_selection": False}
            if (dataset, route) == skipped:
                value["status"] = "SKIPPED_MISSING_COMPLETE_TUNE_CANDIDATE_SET"
            write(path, value); records[(dataset, route)] = run.record(path)
    return records


def test_any_DEV_requires_all_four_saved_lock_files(tmp_path):
    records = selection_records(tmp_path)
    records.pop(next(iter(records)))
    with pytest.raises(ValueError, match="ALL_FOUR"):
        run.locked_candidate_for_development(records, next(iter(records)))


def test_missing_selection_blocks_only_dependent_DEV_after_all_files_saved(tmp_path):
    skipped = ("robin", "PCA_PINV")
    records = selection_records(tmp_path, skipped)
    assert run.locked_candidate_for_development(records, skipped) is None
    assert run.locked_candidate_for_development(records, ("beijing_aotizhongxin", "FIT_TARGET_PINV")) == "RHO_0.01"


def test_changed_selection_file_blocks_DEPENDENT_DEV(tmp_path):
    records = selection_records(tmp_path)
    path = Path(next(iter(records.values()))["path"])
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="LOCK_CHANGED"):
        run.locked_candidate_for_development(records, ("robin", "PCA_PINV"))


def numerical_fixture():
    rng = np.random.default_rng(335)
    z = rng.normal(size=(2, 512, 2)).astype(np.float32)
    prior = np.array([[1., .2, -.3], [0., 1., .7]], np.float32)
    x = z @ prior + np.array([2., -1., 3.], np.float32)
    values = {"history": x, "history_observed": np.ones_like(x, bool), "raw_last": x[:, -1].copy(),
        "origins": np.array([512, 600], np.int64), "std": np.ones(3),
        "target": np.zeros((2, 48, 3), np.float32), "observed": np.ones((2, 48, 3), bool), "split": "tune"}
    cached = {"latent_history": z, "latent_future": np.repeat(z[:, -1:], 48, axis=1) + np.float32(.1),
        "origins": values["origins"].copy(), "prediction": np.zeros((2, 48, 3), np.float32)}
    return values, prior, cached


def test_native_past_bytes_and_origins_required_before_D_share():
    _, _, a = numerical_fixture(); b = {name: value.copy() for name, value in a.items()}
    assert run.shared_native_past(a, b).tobytes() == a["latent_history"].tobytes()
    b["latent_history"][0, 0, 0] = np.nextafter(b["latent_history"][0, 0, 0], np.float32(100.))
    with pytest.raises(ValueError, match="BYTE_IDENTICAL"):
        run.shared_native_past(a, b)
    b = {name: value.copy() for name, value in a.items()}; b["origins"][0] += 1
    with pytest.raises(ValueError, match="BYTE_IDENTICAL"):
        run.shared_native_past(a, b)


def test_future_targets_masks_and_other_native_future_do_not_refit_D():
    values, prior, cached = numerical_fixture()
    state = fit_shrunk_local_decoder(values["history"], values["history_observed"], cached["latent_history"], prior, .1)
    first, _ = run.candidate_predictions(state, values, {"NATIVE_s92601": cached})
    values["target"][:] = 1e9; values["observed"][:] = False
    second_state = fit_shrunk_local_decoder(values["history"], values["history_observed"], cached["latent_history"], prior, .1)
    second, _ = run.candidate_predictions(second_state, values, {"NATIVE_s92601": cached})
    assert run.byte_equal(state.decoder32, second_state.decoder32)
    assert run.byte_equal(first["NATIVE_s92601"], second["NATIVE_s92601"])
    changed = {name: value.copy() for name, value in cached.items()}; changed["latent_future"] += 1.
    third, _ = run.candidate_predictions(state, values, {"NATIVE_s92602": changed})
    assert not run.byte_equal(first["NATIVE_s92601"], third["NATIVE_s92602"])


def test_true_observed_response_only_changes_decoder_when_observed():
    values, prior, cached = numerical_fixture()
    values["history_observed"][:, ::2, 1] = False
    state = fit_shrunk_local_decoder(values["history"], values["history_observed"], cached["latent_history"], prior, 1.)
    values["history"][~values["history_observed"]] = 1e8
    other = fit_shrunk_local_decoder(values["history"], values["history_observed"], cached["latent_history"], prior, 1.)
    assert run.byte_equal(state.decoder32, other.decoder32)
    assert run.byte_equal(state.bias32, other.bias32)


def test_one_failed_method_preserves_independent_seed_and_failure_reason():
    values, prior, cached = numerical_fixture()
    state = fit_shrunk_local_decoder(values["history"], values["history_observed"], cached["latent_history"], prior, .01)
    bad = {name: value.copy() for name, value in cached.items()}; bad["origins"][0] += 1
    predictions, errors = run.candidate_predictions(state, values, {"NATIVE_s92601": bad, "NATIVE_s92602": cached})
    assert set(predictions) == {"NATIVE_s92602"} and set(errors) == {"NATIVE_s92601"}
    assert errors["NATIVE_s92601"]["status"] == "FAILED_PATH_PRESERVED"


def backend_fixture(stored_dtype):
    values, prior, cached = numerical_fixture()
    original = {name: values[name].copy() for name in ("origins", "target", "observed", "raw_last")}
    original["latent_history"] = cached["latent_history"].copy()
    for name in ("BOLT",):
        for suffix in ("static_prediction", "local_prediction", "static_einsum_replay"):
            original[name + "_" + suffix] = cached["prediction"].copy()
        original[name + "_cached_future"] = cached["latent_future"].copy()
    stored = prior.astype(stored_dtype)
    if stored_dtype == np.float64:
        stored[0, 0] += 1e-9
    payloads = {"state": {"decoder": stored}, "cached": cached, "controls": original}
    io = types.SimpleNamespace(checked_payload=lambda item, *args: payloads[item["path"]])
    condition = {"K": 2, "state_artifact": {"path": "state"}, "bolt": {"artifacts": {"tune": {"path": "cached"}}}}
    local = {"artifact": {"path": "controls"}, "local_state_artifact": {"path": "synthetic_old_state"}}
    return values, cached, stored, original, io, condition, local


@pytest.mark.parametrize("stored_dtype", [np.float32, np.float64])
def test_frozen_cache_control_bytes_and_explicit_deployment_prior_cast(stored_dtype):
    values, cached, stored, original, io, condition, local = backend_fixture(stored_dtype)
    before = stored.tobytes()
    context = run.cached_backend(values, condition, local, "BOLT_NUMPY", io, None, {})
    assert run.byte_equal(context["controls"]["BOLT_static_prediction"], cached["prediction"])
    assert stored.tobytes() == before and context["prior"]["decoder"].dtype == stored_dtype
    assert run.byte_equal(context["prior_decoder32"], np.ascontiguousarray(stored.astype(np.float32)))
    assert context["static_prior_deployment_binding"]["stored_prior_dtype"] == str(np.dtype(stored_dtype))
    assert context["static_prior_deployment_binding"]["deployed_prior"]["dtype"] == "float32"
    assert not np.shares_memory(context["prior_decoder32"], stored)
    # Known constant latent past retains exactly the explicitly cast static prior.
    z = np.zeros_like(cached["latent_history"])
    state = fit_shrunk_local_decoder(values["history"], values["history_observed"], z, context["prior_decoder32"], .1)
    np.testing.assert_array_equal(state.decoder32, np.broadcast_to(context["prior_decoder32"][None], state.decoder32.shape))
    expected = values["raw_last"][:, None] + np.einsum("nhk,nkc->nhc", cached["latent_future"], state.decoder32, optimize=False)
    np.testing.assert_array_equal(state.forecast(values["raw_last"], cached["latent_future"], z[:, -1]), expected)
    original["BOLT_cached_future"][0, 0, 0] += 1.
    with pytest.raises(ValueError, match="BYTE_PRESERVED"):
        run.cached_backend(values, condition, local, "BOLT_NUMPY", io, None, {})


@pytest.mark.parametrize("stored_dtype", [np.int64, np.float16])
def test_arbitrary_serialized_prior_dtypes_are_rejected(stored_dtype):
    values, _, _, _, io, condition, local = backend_fixture(stored_dtype)
    with pytest.raises(ValueError, match="STATIC_PRIOR_DECODER_SHAPES"):
        run.cached_backend(values, condition, local, "BOLT_NUMPY", io, None, {})


def test_finite_FP64_prior_that_overflows_deployment_cast_is_rejected():
    values, _, stored, _, io, condition, local = backend_fixture(np.float64)
    stored[0, 0] = 1e100
    with pytest.raises(ValueError, match="DEPLOYED_STATIC_PRIOR_CAST"):
        run.cached_backend(values, condition, local, "BOLT_NUMPY", io, None, {})


@pytest.mark.parametrize("inject_failure", [False, True])
def test_synthetic_staged_producer_locks_before_DEV_and_isolates_missing_selection(tmp_path, monkeypatch, inject_failure):
    """Mock science and metering; real control flow/JSON exclusive writes only."""
    datasets = {"robin": (3, 2, "val", 2), "beijing_aotizhongxin": (3, 2, "eval", 3)}
    monkeypatch.setattr(run, "HERE", tmp_path)
    monkeypatch.setattr(run, "DATASETS", datasets)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    for name in ("run_local_decoder_shrinkage_v1.py", "local_decoder_shrinkage_v1.py", "PLAN_LOCAL_DECODER_SHRINKAGE_KO_v1.md",
            "local_past_decoder_v1.py", "decoder_metric_probe_v1.py", "fixed_decoder_span_v1.py",
            "run_fit_target_comparison_v1.py", "continuation_runtime_v1.py"):
        (tmp_path / name).write_text("synthetic source metadata fixture only", encoding="utf-8")
    payloads, producer, local = {}, {"datasets": {}}, {"datasets": {}}
    def reference(name, arrays):
        payloads[name] = arrays
        return {"path": name, "sha256": "synthetic-in-memory-no-file"}
    for dataset, (c, k, dev, dev_count) in datasets.items():
        parent = {"inputs": {}, "projections": {route: {"K": k, "state_artifact": reference(dataset + route + "state",
            {"decoder": np.ones((k, c), np.float64)}), "bolt": {"artifacts": {}},
            "native": {str(seed): {"artifacts": {}} for seed in run.SEEDS}} for route in run.ROUTES}}
        producer["datasets"][dataset] = parent
        local["datasets"][dataset] = {"splits": {}}
        for split, n in (("fit", 256), ("tune", 64), (dev, dev_count)):
            values = {"history": np.zeros((n, 512, c), np.float32), "history_observed": np.ones((n, 512, c), bool),
                "raw_last": np.zeros((n, c), np.float32), "raw_forecast": np.zeros((n, 48, c), np.float32),
                "target": np.zeros((n, 48, c), np.float32), "observed": np.ones((n, 48, c), bool),
                "origins": np.arange(n, dtype=np.int64) + 512, "std": np.ones(c)}
            parent["inputs"][split] = reference(dataset + split + "input", values)
            local_cell = {"routes": {}}
            local["datasets"][dataset]["splits"][split] = local_cell
            for route in run.ROUTES:
                condition = parent["projections"][route]
                cached = {"latent_history": np.zeros((n, 512, k), np.float32), "latent_future": np.full((n, 48, k), .5, np.float32),
                    "origins": values["origins"].copy(), "prediction": np.ones((n, 48, c), np.float32)}
                bolt_ref = reference(dataset + route + split + "Bolt", cached)
                condition["bolt"]["artifacts"][split] = bolt_ref
                for seed in run.SEEDS:
                    condition["native"][str(seed)]["artifacts"][split] = reference(dataset + route + split + str(seed), copy.deepcopy(cached))
                local_cell["routes"][route] = {"backends": {}}
                for backend, methods in (("BOLT_NUMPY", ("BOLT",)), ("NATIVE_TORCH32", ("NATIVE_s92601", "NATIVE_s92602"))):
                    arrays = {name: values[name].copy() for name in ("origins", "target", "observed", "raw_last")}
                    arrays["latent_history"] = cached["latent_history"].copy()
                    for method in methods:
                        arrays[method + "_static_prediction"] = cached["prediction"].copy()
                        arrays[method + "_local_prediction"] = np.full_like(cached["prediction"], 2.)
                        arrays[method + "_static_einsum_replay"] = cached["prediction"].copy()
                        arrays[method + "_cached_future"] = cached["latent_future"].copy()
                    local_cell["routes"][route]["backends"][backend] = {
                        "artifact": reference(dataset + route + split + backend + "controls", arrays),
                        "local_state_artifact": {"path": "synthetic-old-state", "sha256": "not-loaded"}}
    local_audit = tmp_path / "metadata_only.json"; write(local_audit, {})
    monkeypatch.setattr(run, "load_gates", lambda *args: (local, producer, {"local_decoder_diagnostic": {}, "local_decoder_audit": {}}))
    output = tmp_path / "synthetic_output"
    class FakeMeter:
        latest = None
        def __init__(self, *args, **kwargs): pass
        def protected_check(self): pass
        def check(self, event=None):
            if event is not None: FakeMeter.latest = event
        def close(self, status, failure): return {"status": status, "synthetic_fixture_not_resource_measurement": True}
    def write_new(path, value):
        with Path(path).open("x", encoding="utf-8") as handle: json.dump(value, handle, allow_nan=False)
    def save_npz(path, arrays):
        # JSON envelope represents synthetic array bindings; no actual .npz is read/written.
        fixture_path = Path(str(path) + ".synthetic.json")
        write_new(fixture_path, {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in arrays.items()})
        return {**run.record(fixture_path), "synthetic_no_actual_array_file": True}
    def fake_fit(x, observed, z, prior, rho):
        assert prior.dtype == np.float32 and prior.flags.c_contiguous
        e = FakeMeter.latest or {}
        if (inject_failure and e.get("dataset") == "robin" and e.get("route") == "PCA_PINV"
                and e.get("split") == "tune" and e.get("backend") == "BOLT_NUMPY" and rho == .01):
            raise ValueError("SYNTHETIC_ONE_CANDIDATE_SOLVER_FAILURE")
        n, _, c = x.shape; k = z.shape[2]
        d64 = np.zeros((n, k, c), np.float64); b64 = np.zeros((n, c), np.float64)
        metadata = {"true_observed_counts": observed.sum(axis=1).tolist(), "actual_lambda": np.ones((n, c)).tolist(),
            "zero_trace_prior_slope_retained": np.zeros((n, c), bool).tolist(), "rho": rho}
        return LocalPastDecoder(d64, d64.astype(np.float32), b64, b64.astype(np.float32), metadata)
    fake_io = types.SimpleNamespace(checked_payload=lambda item, *args: {name: value.copy() for name, value in payloads[item["path"]].items()}, save_npz=save_npz)
    monkeypatch.setitem(sys_modules := sys.modules, "fixed_decoder_span_v1", fake_io)
    monkeypatch.setitem(sys_modules, "decoder_metric_probe_v1", types.SimpleNamespace(masked_scores=masked_scores))
    monkeypatch.setitem(sys_modules, "continuation_runtime_v1", types.SimpleNamespace(Meter=FakeMeter, write_new=write_new))
    monkeypatch.setitem(sys_modules, "run_fit_target_comparison_v1", types.SimpleNamespace(prior_file_records=lambda *args: []))
    monkeypatch.setitem(sys_modules, "prepare_inputs", types.SimpleNamespace(install_protected_test_guard=lambda: ["synthetic guard"]))
    monkeypatch.setitem(sys_modules, "local_decoder_shrinkage_v1", types.SimpleNamespace(fit_shrunk_local_decoder=fake_fit))
    exit_code = run.execute(local_audit, local_audit, output)
    result = run.read(output / "RESULT.json")
    expected_status = "PARTIAL_COMPLETED" if inject_failure else "COMPLETE"
    assert exit_code == (2 if inject_failure else 0) and result["status"] == expected_status
    events = result["execution_events"]
    saved = [event["sequence"] for event in events if event["kind"] == "SELECTION_LOCK_SAVED"]
    dev = [event["sequence"] for event in events if event["kind"] == "DEV_SELECTED_RHO_FIT"]
    assert len(saved) == 4 and max(saved) < min(dev)
    assert all(event["candidate"] == "RHO_0.001" for event in events if event["kind"] == "DEV_SELECTED_RHO_FIT")
    assert result["development_positive_rho_sweep_executed"] is False
    assert result["native_specific_selection"] is False
    assert result["completed_reused_control_method_pairs"] == 36
    assert result["completed_RAW_control_cells"] == 6
    assert len(result["selection_locks"]) == 2
    if inject_failure:
        assert result["completed_selection_locks"] == 3 and result["completed_selected_DEV_method_predictions"] == 9
        assert result["datasets"]["robin"]["splits"]["val"]["routes"]["PCA_PINV"]["status"] == "SKIPPED_DEPENDENT_DEV_NO_SELECTION"
        assert result["datasets"]["beijing_aotizhongxin"]["splits"]["eval"]["status"] == "COMPLETE"
        assert any("SYNTHETIC_ONE_CANDIDATE_SOLVER_FAILURE" in row["failure"]["message"] for row in result["failed_paths"])
    else:
        assert result["completed_selection_locks"] == 4
        assert result["completed_FIT_TUNE_positive_rho_method_predictions"] == 120
        assert result["completed_selected_DEV_method_predictions"] == 12
