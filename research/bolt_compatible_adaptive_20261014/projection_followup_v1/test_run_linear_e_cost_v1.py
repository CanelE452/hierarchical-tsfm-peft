"""No pretrained model or campaign data: closed gates and toy Torch deployment."""
from pathlib import Path
from types import SimpleNamespace
import inspect
import json

import numpy as np
import pytest
import torch

import run_linear_e_cost_v1 as runner
from linear_e_adaptation_v1 import LinearLagAdapter


def test_none_actual_audit_gate_stops_before_any_path_access(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "TRAIN_AUDIT_SHA", None)
    with pytest.raises(ValueError, match="NOT_REGISTERED"):
        runner.load_gates(tmp_path / "missing.json", tmp_path / "missing_audit.json")


def test_registered_wrong_complete_hash_stays_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "TRAIN_RESULT_SHA", "0" * 64)
    monkeypatch.setattr(runner, "TRAIN_AUDIT_SHA", "1" * 64)
    monkeypatch.setattr(runner, "TRAIN_AUDIT_COUNTS", {"adaptation_fits_checked": 16})
    first, second = tmp_path / "producer.json", tmp_path / "audit.json"
    first.write_text('{"status":"COMPLETE"}', encoding="utf-8")
    second.write_text('{"status":"PASS"}', encoding="utf-8")
    with pytest.raises(ValueError, match="EXACT_COMPLETE"):
        runner.load_gates(first, second)


def test_nine_names_map_to_eight_exact_registered_condition_keys():
    for data in runner.DATASETS:
        keys = runner.dataset_keys(data)
        assert len(keys) == 8 and len({runner.route(key) for key in keys}) == 8
        assert set(runner.ROUTES) == {"RAW_BOLT", *(runner.route(key) for key in keys)}
    assert runner.route(("robin", "L7", True, 92602)) == "L7_E_LEARN_s92602"


def test_same_seed_variants_are_not_collapsed_during_deployment():
    assert runner.route(("robin", "L0", False, 92601)) != runner.route(("robin", "L0", False, 92602))


def model():
    return LinearLagAdapter(np.zeros(2, np.float32), np.array([[1.], [.5]], np.float32),
        np.array([[.2, .4], [.3, -.1]], np.float32), (0, 1), learn_E=False).eval().requires_grad_(False)


def test_callback_executes_real_toy_F_each_time_without_cached_future():
    calls = []
    def frozen(z):
        calls.append(z.detach().clone())
        return (z[:, -1:] + z.mean(1, keepdim=True) * .1).expand(-1, 48, -1)
    history = np.random.default_rng(6).normal(size=(8, 512, 2)).astype(np.float32)
    selected = model()
    callback = runner.adapter_predictor(selected, frozen, lambda x: torch.as_tensor(x, dtype=torch.float32))
    first = callback(history, np.ones_like(history, bool))
    second = callback(history + .1, np.ones_like(history, bool))
    assert len(calls) == 2 and not np.array_equal(first, second)
    assert first.dtype == np.float32 and first.shape == (8, 48, 2)
    assert not any(value.requires_grad for value in selected.parameters())


def test_first8_physical_precheck_matches_saved_toy_components():
    import whole_linear_e_cost_v1 as whole
    rng = np.random.default_rng(7)
    x = rng.normal(size=(8, 512, 2)).astype(np.float32)
    obs = np.ones_like(x, bool); obs[:, :4, 0] = False; x[~obs] = np.nan
    mean, std = np.array([1., 2.], np.float32), np.array([2., 3.], np.float32)
    normalized = whole.preprocess(x, obs, mean, std)
    m = model()
    F = lambda z: (z[:, -1:] + .1).expand(-1, 48, -1)
    with torch.no_grad():
        p, z, f = m(torch.from_numpy(normalized), F, return_components=True)
        features = m.lag_features(z, f)
    raw = {"history": normalized, "raw_last": normalized[:, -1].copy(), "history_observed": obs,
           "origins": np.arange(8, dtype=np.int64), "mean": mean, "std": std}
    physical = {"history_physical": x, "history_observed": obs, "origins": raw["origins"]}
    reference = {"prediction": p.numpy(), "latent_history": z.numpy(), "latent_future": f.numpy(),
                 "lag_features": features.numpy(), "origins": raw["origins"]}
    check, arrays = runner.physical_precheck(physical, raw, reference, m, F, torch.from_numpy)
    assert check["input_backend_pass"] and check["prediction_backend_pass"]
    assert all(check["exact_lineage"].values())
    assert all(value["pass"] for value in check["prediction_checks"].values())
    assert arrays["prediction"].dtype == np.float32
    bad = dict(reference, prediction=reference["prediction"] + .1)
    check, _ = runner.physical_precheck(physical, raw, bad, m, F, torch.from_numpy)
    assert check["input_backend_pass"] and not check["prediction_backend_pass"]


def test_precheck_does_not_accept_wrong_origin_even_identical_outputs():
    import whole_linear_e_cost_v1 as whole
    x = np.ones((8, 512, 2), np.float32)
    obs = np.ones_like(x, bool); origins = np.arange(8, dtype=np.int64)
    raw = {"history": x, "raw_last": x[:, -1], "history_observed": obs, "origins": origins,
           "mean": np.zeros(2, np.float32), "std": np.ones(2, np.float32)}
    physical = {"history_physical": x, "history_observed": obs, "origins": origins}
    m = model(); F = lambda z: z[:, -1:].expand(-1, 48, -1)
    with torch.no_grad(): p, z, f = m(torch.from_numpy(x), F, return_components=True)
    reference = {"prediction": p.numpy(), "latent_history": z.numpy(), "latent_future": f.numpy(),
                 "lag_features": m.lag_features(z, f).numpy(), "origins": origins + 1}
    check, _ = runner.physical_precheck(physical, raw, reference, m, F, torch.from_numpy)
    assert not check["input_backend_pass"] and check["prediction_backend_pass"]


def test_deployment_uses_no_grad_and_never_accepts_targets():
    assert set(inspect.signature(runner.adapter_predictor).parameters) == {"model", "bolt", "tensor"}
    source = inspect.getsource(runner.adapter_predictor)
    assert "torch.no_grad" in source and "bolt" in source
    assert "target" not in source and "latent_future" not in source


def test_new_whole_state_hash_is_independent_of_grad_flag():
    m = model(); first = runner.adapter_hashes({("fixture",): m})
    m.E.requires_grad_(True)
    assert runner.adapter_hashes({("fixture",): m}) == first
    with torch.no_grad(): m.D.add_(.01)
    assert runner.adapter_hashes({("fixture",): m}) != first


def canonical_fixture(tmp_path, monkeypatch, change=None):
    import run_linear_e_adaptation_v1 as upstream
    monkeypatch.setattr(upstream, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    shapes = {"history": [64, 512, 17], "history_observed": [64, 512, 17], "target": [64, 48, 17],
        "observed": [64, 48, 17], "raw_last": [64, 17], "origins": [64], "mean": [17], "std": [17], "raw_forecast": [64, 48, 17]}
    arrays = {field: {"shape": shape, "dtype": "bool" if "observed" in field else "int64" if field == "origins" else "float32", "sha256": field}
              for field, shape in shapes.items()}
    raw = {"path": str(tmp_path / "RAW_TUNE.npz"), "sha256": "raw", "arrays": json.loads(json.dumps(arrays))}
    arrays["history_physical"] = {"shape": [64, 512, 17], "dtype": "float32", "sha256": "physical"}
    physical = tmp_path / "physical_tune.npz"; physical.write_bytes(b"never parsed or opened as NPZ")
    canonical = {"dataset": "robin", "split": "tune", "path": str(physical), "sha256": "preserved-physical-sha", "arrays": arrays}
    bridge = {"status": "COMPLETE", "actual_status": {"protected_test_opened": False}, "artifacts": [canonical]}
    if change: change(bridge, raw)
    bridge_path = tmp_path / "BRIDGE.json"; bridge_path.write_text(json.dumps(bridge), encoding="utf-8")
    parent_path = tmp_path / "PARENT.json"
    parent_path.write_text(json.dumps({"status": "COMPLETE", "protected_TEST_accessed": False,
        "bridge_provenance": {"input_bridge": runner.record(bridge_path)}}), encoding="utf-8")
    return {"source_parent_results": {"robin": runner.record(parent_path)}, "datasets": {"robin": {"inputs": {"tune": raw}}}}


def test_existing_TUNE_bridge_keeps_full_manifest_without_NPZ_loader(tmp_path, monkeypatch):
    producer = canonical_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("metadata helper must not load NPZ"))
    proof = runner.canonical_tune_bridge(producer, "robin")
    assert proof["split"] == "tune" and proof["canonical_record"]["sha256"] == "preserved-physical-sha"
    assert proof["complete_array_manifest_preserved"] and not proof["inverse_normalization_used"] and not proof["DEV_physical_alias_used"]


@pytest.mark.parametrize("change,reason", [
    (lambda bridge, raw: bridge["artifacts"].append(dict(bridge["artifacts"][0])), "EXACTLY_ONE"),
    (lambda bridge, raw: bridge["artifacts"][0]["arrays"].pop("history_physical"), "FULL_PHYSICAL"),
    (lambda bridge, raw: bridge["artifacts"][0]["arrays"]["origins"].update(sha256="different"), "RAW_ARRAY_BINDING"),
    (lambda bridge, raw: bridge["artifacts"][0].update(split="val"), "EXACTLY_ONE"),
])
def test_canonical_TUNE_refuses_missing_or_mismatched_metadata(tmp_path, monkeypatch, change, reason):
    producer = canonical_fixture(tmp_path, monkeypatch, change)
    with pytest.raises(ValueError, match=reason): runner.canonical_tune_bridge(producer, "robin")


def test_cost_reference_is_exact_selected_TUNE_CP_not_DEV():
    state = {"path": "fixture", "sha256": "selected"}
    condition = {"selected_update": 100, "selected_state": state, "selected_tune": {"mse": .1},
        "checkpoints": [{"update": 0, "state": {}, "tune": {"mse": .2}, "artifact": "CP0"},
                        {"update": 100, "state": state, "tune": {"mse": .1}, "artifact": "SELECTED_TUNE"}], "DEV": {"artifact": "wrong"}}
    assert runner.selected_TUNE_checkpoint(condition)["artifact"] == "SELECTED_TUNE"
    condition["selected_state"] = {"path": "other", "sha256": "changed"}
    with pytest.raises(ValueError, match="EXACT_REGISTERED_SELECTED_TUNE"):
        runner.selected_TUNE_checkpoint(condition)
