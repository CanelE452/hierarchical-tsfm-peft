"""Synthetic controller checks; never open campaign payloads or a real model."""
from pathlib import Path
import importlib.util
import json

import numpy as np
import pytest
import torch


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("linear_controller_under_test", HERE / "run_linear_e_adaptation_v1.py")
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def test_gate_stops_before_array_or_model_import_on_wrong_report(tmp_path, monkeypatch):
    paths = [tmp_path / name for name in ("science.json", "audit.json", "REPORT.md", "MANIFEST.json", "report_audit.json")]
    for path in paths: path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(controller, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="EXACT_ACCEPTED_REPORT"):
        controller.load_gates(*paths)


def test_binding_rejects_protected_test_before_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="NON_TEST"):
        controller.bound({"path": str(tmp_path / "test.npz"), "sha256": "unknown"})


def test_fresh_output_rejects_existing_and_path_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, "HERE", tmp_path)
    (tmp_path / "runs" / "already").mkdir(parents=True)
    for tag in ("already", "../escape", ".", ""):
        with pytest.raises(ValueError): controller.fresh_output(tag)


def test_exact_same_backend_parity_rejects_signed_zero_difference():
    first = np.array([0.], np.float32)
    assert controller.byte_equal(first, first.copy())
    assert not controller.byte_equal(first, -first)


def test_registered_parity_does_not_relax_failed_element():
    actual = np.array([1., 2.], np.float32)
    assert controller.parity(actual, actual)["pass"]
    assert not controller.parity(actual + .001, actual)["pass"]
    with pytest.raises(ValueError, match="SAME"):
        controller.parity(actual, actual[:1])


def test_full_manifest_io_helpers_are_reused_without_optional_array_validation():
    source = (HERE / "run_linear_e_adaptation_v1.py").read_text(encoding="utf-8")
    assert "io.checked_payload" in source and "io.save_npz" in source
    assert "np.load(" not in source and "torch.load(" not in source


def test_all_sixteen_resolutions_precede_any_DEV_even_one_failure():
    report = {"datasets": {}, "failed_paths": [], "adaptation_training_fits": 0,
              "training_updates_completed": 0, "checkpoint_states_saved": 0}
    events = []
    def condition(key):
        events.append(("train", key))
        if key == controller.CONDITION_KEYS[2]: raise RuntimeError("one independent path fails")
        return {"status": "COMPLETE", "selected_update": 0, "training_updates_completed": 600, "checkpoint_count": 5}
    def seal(key, row):
        events.append(("lock", key)); return {"path": str(key), "sha256": "fixture"}
    def registry(rows):
        assert len(rows) == 16
        events.append(("registry", len(rows))); return {"path": "registry", "sha256": "fixture"}
    resolved = controller.train_all_conditions(report, condition, seal, registry, lambda *args: None)
    assert len(resolved) == 15 and report["resolved_selection_locks"] == 16
    assert report["all_16_locks_before_any_DEV"] is True
    assert report["adaptation_training_fits"] == 15 and report["checkpoint_states_saved"] == 75
    assert len(report["failed_paths"]) == 1
    controller.infer_development(report, resolved, lambda key, row: events.append(("DEV", key)))
    registry_index = next(i for i, row in enumerate(events) if row[0] == "registry")
    assert all(i > registry_index for i, row in enumerate(events) if row[0] == "DEV")


def test_failed_probe_blocks_only_its_four_conditions():
    probe = {"cases": {"robin": {"L0": {"status": "FAILED_PATH_PRESERVED"}, "L7": {"status": "PASS"}},
                       "beijing_aotizhongxin": {"L0": {"status": "PASS"}, "L7": {"status": "PASS"}}}}
    assert sum(controller.probe_allows(probe, key) for key in controller.CONDITION_KEYS) == 12


def test_DEV_guard_prevents_partial_registry():
    with pytest.raises(ValueError, match="ALL_16"):
        controller.infer_development({"resolved_selection_locks": 15, "all_16_locks_before_any_DEV": False}, {}, lambda *args: None)


def test_cli_requires_probe_result_for_training():
    with pytest.raises(SystemExit): controller.parse_args(["--mode", "train", "--tag", "fixture"])
    args = controller.parse_args(["--mode", "probe", "--tag", "fixture"])
    assert args.mode == "probe" and args.probe_result is None


class ToyAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.E = torch.nn.Parameter(torch.tensor([[.5], [.1]], dtype=torch.float32))
        self.D = torch.nn.Parameter(torch.tensor([[.3, .4]], dtype=torch.float32))
        self.register_buffer("center", torch.zeros(2))
        self.lags, self.C, self.K = (0,), 2, 1

    def lag_features(self, z, future): return future-z[:, -1:]

    def forward(self, x, bolt, return_components=False):
        z = (x-self.center) @ self.E
        future = bolt(z)
        p = x[:, -1:] + self.lag_features(z, future) @ self.D
        return (p, z, future) if return_components else p


def toy_payload(n=8):
    rng = np.random.default_rng(19)
    history = rng.normal(size=(n, 512, 2)).astype(np.float32)
    target = (history[:, -1:] + .2).repeat(48, axis=1).astype(np.float32)
    observed = np.ones_like(target, bool)
    observed[::2, :, 1] = False
    target[~observed] = np.nan
    return {"history": history, "target": target, "observed": observed, "raw_last": history[:, -1].copy(),
            "origins": np.arange(n, dtype=np.int64), "std": np.ones(2, np.float32), "mean": np.zeros(2, np.float32)}


def toy_F(z): return z[:, -1:].expand(-1, 48, -1) + .1


def test_canonical_predict_saves_exact_scored_array_and_all_components():
    payload, model = toy_payload(), ToyAdapter()
    outputs = controller.predict_components(model, toy_F, payload["history"], lambda *args: None)
    assert set(outputs) == {"prediction", "latent_history", "latent_future", "lag_features"}
    assert outputs["prediction"].shape == (8, 48, 2)
    assert all(value.dtype == np.float32 and np.isfinite(value).all() for value in outputs.values())
    assert not model.E.grad


def test_train_uses_full_FIT_counts_and_preserves_checkpoint_scored_bytes(monkeypatch):
    payload, model, seen_counts, callback_arrays = toy_payload(), ToyAdapter(), [], []
    import linear_e_adaptation_v1 as pure
    original = pure.global_fit_macro_mse
    def capture(prediction, target, observed, fit_counts, fit_windows):
        seen_counts.append((fit_counts.detach().cpu().numpy().copy(), fit_windows, len(prediction)))
        return original(prediction, target, observed, fit_counts, fit_windows)
    monkeypatch.setattr(pure, "global_fit_macro_mse", capture)
    result = controller.train_condition(model, toy_F, payload, payload, 92601, lambda *args: None,
        lambda update, state, score, arrays: callback_arrays.append((update, score, arrays["prediction"].copy())), updates=2)
    assert result["training_updates_completed"] == 2 and len(result["learning_curve"]) == 2
    assert [row[0] for row in callback_arrays] == [0, 2]
    expected = payload["observed"].sum((0, 1))
    assert all(np.array_equal(row[0], expected) and row[1:] == (8, 4) for row in seen_counts)
    assert all(np.isfinite(row[1]["mse"]) for row in callback_arrays)
    assert result["selected_update"] == min(result["checkpoints"], key=lambda row: (row["tune"]["mse"], row["update"]))["update"]


def test_two_seeds_start_with_identical_E_D_and_change_only_order():
    payload, captures, results = toy_payload(), [], []
    for seed in (92601, 92602):
        start = []
        result = controller.train_condition(ToyAdapter(), toy_F, payload, payload, seed, lambda *args: None,
            lambda update, state, score, arrays: start.append({name: value.clone() for name, value in state.items()}) if update == 0 else None,
            updates=2)
        captures.append(start[0]); results.append(result)
    assert all(torch.equal(captures[0][name], captures[1][name]) for name in captures[0])
    assert results[0]["order_sha256"] != results[1]["order_sha256"]


def test_safe_checkpoint_state_is_owned_CPU_copy():
    model = ToyAdapter()
    state = controller.state_arrays(model)
    before = state["E"].copy()
    with torch.no_grad(): model.E.add_(1)
    assert np.array_equal(state["E"], before)


def test_probe_result_must_resolve_all_four_paths_before_train():
    with pytest.raises(ValueError, match="FOUR"):
        controller.validate_probe_registry({"mode": "probe", "cases": {"robin": {"L0": {"status": "PASS"}}}})


class ToyFrozenF:
    def __init__(self, frozen=True):
        self.model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad(): self.model.weight.fill_(.7)
        self.model.eval().requires_grad_(not frozen)

    def __call__(self, z):
        return self.model(z.mean(1, keepdim=True)).expand(-1, 48, -1)


def test_real_torch_F_parameter_stays_frozen_while_input_E_learns():
    model, frozen, payload = ToyAdapter(), ToyFrozenF(), toy_payload()
    before = frozen.model.weight.detach().clone()
    e_before = model.E.detach().clone()
    controller.train_condition(model, frozen, payload, payload, 92601, lambda *args: None, lambda *args: None, updates=2)
    assert torch.equal(frozen.model.weight, before) and frozen.model.weight.grad is None
    assert not torch.equal(model.E, e_before)


def test_train_rejects_accidentally_trainable_F_before_optimizer_step():
    with pytest.raises(ValueError, match="F_FROZEN_EXCLUDED"):
        controller.train_condition(ToyAdapter(), ToyFrozenF(frozen=False), toy_payload(), toy_payload(), 92601,
            lambda *args: None, lambda *args: None, updates=1)


def test_all_resolved_failed_paths_remain_partial_without_DEV_calls():
    report = {"datasets": {}, "failed_paths": [], "adaptation_training_fits": 0,
              "training_updates_completed": 0, "checkpoint_states_saved": 0}
    resolved = controller.train_all_conditions(report,
        lambda key: {"status": "FAILED_PATH_PRESERVED", "training_updates_completed": 3, "checkpoint_count": 1},
        lambda *args: {"path": "fixture", "sha256": "fixture"}, lambda *args: {"path": "fixture", "sha256": "fixture"}, lambda *args: None)
    seen = []
    controller.infer_development(report, resolved, lambda *args: seen.append(args))
    assert resolved == {} and seen == [] and len(report["failed_paths"]) == 16
    assert report["adaptation_training_fits"] == 0 and report["training_updates_completed"] == 48
