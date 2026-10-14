from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_stage_oracle_extract_v1 as runner  # noqa: E402


class FakeBudgetHeld(RuntimeError):
    pass


class FakeLimit:
    def __init__(self, *, terminal_status="COMPLETE", close_error=None):
        self.authorization_record = {"path": "synthetic_auth.json", "sha256": "a" * 64}
        self.run_manifest_record = {"path": "synthetic_manifest.json", "sha256": "b" * 64}
        self.shared = SimpleNamespace(check=lambda: {"scope": "fast"})
        self.terminal_status = terminal_status
        self.close_error = close_error
        self.checks = 0

    def check(self):
        self.checks += 1
        return {"scope": "full", "checks": self.checks}

    def close(self, status, error=None):
        return {
            "terminal_status": self.terminal_status,
            "terminal_error": self.close_error,
            "requested_status": status,
            "caller_error": error,
            "path": "synthetic_resource.json",
            "sha256": "c" * 64,
        }


class FakeCodec:
    def __init__(self):
        self.grad_param = SimpleNamespace(grad=None)

    def parameters(self):
        return [self.grad_param]


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def write_json_new(path: Path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def make_split(n=2, c=17, k=4):
    history = np.arange(n * 512 * c, dtype=np.float32).reshape(n, 512, c) / np.float32(1000.0)
    target = np.repeat(history[:, -1:], 48, axis=1).astype(np.float32, copy=False)
    observed = np.ones((n, 48, c), dtype=np.bool_)
    return {
        "history": history,
        "history_observed": np.ones_like(history, dtype=np.bool_),
        "raw_last": history[:, -1].copy(),
        "raw_forecast": target.copy(),
        "target": target,
        "observed": observed,
        "origins": np.arange(n, dtype=np.int64) + 512,
        "latent_past": np.zeros((n, 512, k), dtype=np.float32),
        "z_future": np.zeros((n, 48, k), dtype=np.float32),
        "z_last": np.zeros((n, 1, k), dtype=np.float32),
    }


def install_fixture_modules(
    monkeypatch,
    tmp_path,
    *,
    d2_status="COMPLETE",
    cached_target_offset=0.0,
    prefix_pass=True,
    terminal_status="COMPLETE",
    runtime_constructor_error=None,
):
    fake_here = tmp_path / "source"
    fake_here.mkdir()
    for name in (
        "stage_separation_runtime_v1.py",
        "stage_separation_oracle_v1.py",
        "METHOD_LOCK_STAGE_SEPARATION_v1.md",
        "PLAN_STAGE_SEPARATION_KO_v1.md",
        "run_condition_probes.py",
        "prepare_inputs.py",
        "fast_storage_scan_v1.py",
    ):
        (fake_here / name).write_text(f"synthetic {name}\n", encoding="utf-8")
    monkeypatch.setattr(runner, "HERE", fake_here)
    monkeypatch.setattr(runner, "CACHE", tmp_path / "cache")

    splits = {name: make_split() for name in ("fit", "tune", "val")}
    mean = np.zeros(17, dtype=np.float32)
    std = np.ones(17, dtype=np.float32)
    bridge = {"declared_asset_hashes": {"source": {}}}
    provenance = {
        "input_artifacts": [{"path": str(tmp_path / "bridge_fit.npz"), "sha256": "1" * 64}],
        "input_bridge": {"path": str(tmp_path / "bridge_RESULT.json"), "sha256": "2" * 64},
    }
    d2_result = tmp_path / "D2_RESULT.json"
    d2_result.write_text("synthetic D2 result", encoding="utf-8")
    d2 = {
        "status": d2_status,
        "stage": "D2",
        "dataset": "robin",
        "seed": 92601,
        "F_frozen_pass": True,
        "MAIN_codec_frozen_pass": True,
        "F_before": {"old_algorithm": "old-f-hash", "bi_algorithm": "bi-f-hash"},
        "F_after": {"old_algorithm": "old-f-hash", "bi_algorithm": "bi-f-hash"},
        "MAIN_checkpoint": {"checkpoint": {"path": str(tmp_path / "main.pt"), "sha256": "d" * 64}},
        "source_snapshot": [],
        "artifacts": [
            {"path": str(tmp_path / f"ABSOLUTE_LATENTS_{split}.npz"), "sha256": str(i) * 64}
            for i, split in enumerate(("fit", "tune", "val"), start=3)
        ],
    }
    calls = {"load_bridge": 0, "import_old_engine": 0, "oracle_extract": [], "save_npz": []}

    class FakeProbes:
        OLD_F_HASH = "old-f-hash"
        BI_F_HASH = "bi-f-hash"

        @staticmethod
        def resolve(path):
            return Path(path)

        @staticmethod
        def sha(path):
            path = Path(path)
            return file_sha(path) if path.exists() else "f" * 64

        @staticmethod
        def read(path):
            if Path(path) == d2_result:
                return d2
            return json.loads(Path(path).read_text(encoding="utf-8"))

        @staticmethod
        def verified_file(record, role=None):
            return dict(record)

        @staticmethod
        def load_bridge(input_bridge, dataset, barrier):
            calls["load_bridge"] += 1
            return bridge, splits, mean, std, provenance

        @staticmethod
        def verify_source_records(_bridge):
            return [{"path": str(fake_here / "run_condition_probes.py"), "sha256": file_sha(fake_here / "run_condition_probes.py")}]

        @staticmethod
        def check_original_raw(_bridge, _dataset, _splits, _mean, _std):
            return {"path": str(tmp_path / "raw.npz"), "sha256": "4" * 64}

        @staticmethod
        def import_old_engine():
            calls["import_old_engine"] += 1
            return SimpleNamespace(tensor=lambda value: value)

        @staticmethod
        def load_main_codec(_engine, _bridge, _dataset, _seed):
            state = {"fixed_basis": np.zeros((17, 4), dtype=np.float32)}
            cp = {
                "checkpoint": {"path": str(tmp_path / "main.pt"), "sha256": "d" * 64},
                "case": {"path": str(tmp_path / "case.json"), "sha256": "e" * 64},
            }
            return FakeCodec(), state, cp

        @staticmethod
        def load_checked_npz(record):
            split = Path(record["path"]).stem.replace("ABSOLUTE_LATENTS_", "")
            values = splits[split]
            cached = {
                "latent_past": values["latent_past"].copy(),
                "z_future": values["z_future"].copy(),
                "z_last": values["z_last"].copy(),
                "source_gpu_prediction": values["target"].copy(),
                "original_prediction": values["target"].copy(),
                "raw_forecast": values["raw_forecast"].copy(),
                "target": values["target"].copy() + np.float32(cached_target_offset),
                "observed": values["observed"].copy(),
                "origins": values["origins"].copy(),
                "raw_last": values["raw_last"].copy(),
                "history_observed": values["history_observed"].copy(),
                "mean": mean.copy(),
                "std": std.copy(),
            }
            return cached, dict(record)

        @staticmethod
        def save_npz(path, arrays):
            calls["save_npz"].append((Path(path).name, sorted(arrays)))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic oracle npz placeholder")
            return {"path": str(path), "sha256": file_sha(path), "arrays": {key: list(value.shape) for key, value in arrays.items()}}

        @staticmethod
        def old_state_hash(_codec):
            return "old-codec-hash"

        @staticmethod
        def bi_state_hash(_codec):
            return "bi-codec-hash"

    class FakeOracle:
        @staticmethod
        def assert_no_protected_test_path(path, role=""):
            if "TEST" in str(path):
                raise AssertionError("protected TEST path reached")

        @staticmethod
        def validate_d2_cached_latent_payload(cached):
            required = {
                "latent_past",
                "z_future",
                "z_last",
                "source_gpu_prediction",
                "original_prediction",
                "raw_forecast",
                "target",
                "observed",
                "origins",
                "raw_last",
                "history_observed",
                "mean",
                "std",
            }
            if not required.issubset(cached):
                raise AssertionError("cached latent schema changed")

        @staticmethod
        def compare_arrays(left, right):
            return {"pass": bool(np.array_equal(left, right)), "max_abs": float(np.max(np.abs(left - right))) if left.size else 0.0}

        @staticmethod
        def extract_normalized_oracle(codec, history, target, observed, *, history_observed, tensor, no_grad, check, synchronize, batch_size):
            calls["oracle_extract"].append(
                {
                    "history_shape": history.shape,
                    "target_shape": target.shape,
                    "observed_shape": observed.shape,
                    "history_observed_shape": history_observed.shape,
                    "batch_size": batch_size,
                }
            )
            check()
            n = len(history)
            return {
                "oracle_z": np.zeros((n, 48, 4), dtype=np.float32),
                "current_past_latent": np.zeros((n, 512, 4), dtype=np.float32),
                "current_z_last": np.zeros((n, 1, 4), dtype=np.float32),
                "oracle_past_prefix_latent": np.zeros((n, 512, 4), dtype=np.float32),
                "prefix_check": {"pass": bool(prefix_pass)},
                "first_sample_forward_invariance": {"pass": True, "status": "TESTED_CONTEXT_CHANGED_PREFIX_INVARIANCE"},
                "future_filled": target.copy(),
                "future_fill_source": np.ones(target.shape, dtype=np.uint8),
                "complete_future_prefix_mask": np.ones((n, 48), dtype=np.bool_),
                "complete_future_prefix_counts": [int(n)] * 48,
                "normalization_contract": {"refit_or_renormalized_here": False},
            }

    def make_limit(tag, gpu=True):
        if runtime_constructor_error is not None:
            raise runtime_constructor_error
        return FakeLimit(terminal_status=terminal_status)

    fake_runtime = SimpleNamespace(
        ContinuationLimit=make_limit,
        BudgetHeld=FakeBudgetHeld,
        write_new=write_json_new,
    )
    fake_inputs = SimpleNamespace(
        install_protected_test_guard=lambda: calls.setdefault("guard_installed", True),
        verify_phase_barrier=lambda *args: {"barrier": "synthetic"},
    )
    fake_torch = SimpleNamespace(
        no_grad=lambda: SimpleNamespace(__enter__=lambda self: None, __exit__=lambda self, exc_type, exc, tb: None),
        cuda=SimpleNamespace(synchronize=lambda: calls.setdefault("cuda_sync_seen", True)),
    )
    monkeypatch.setitem(sys.modules, "stage_separation_runtime_v1", fake_runtime)
    monkeypatch.setitem(sys.modules, "prepare_inputs", fake_inputs)
    monkeypatch.setitem(sys.modules, "run_condition_probes", FakeProbes)
    monkeypatch.setitem(sys.modules, "stage_separation_oracle_v1", FakeOracle)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    args = SimpleNamespace(
        dataset="robin",
        seed=92601,
        input_bridge=str(tmp_path / "bridge_RESULT.json"),
        d2_result=str(d2_result),
        tag="synthetic_oracle_v1",
    )
    return args, calls


def test_runtime_authorization_failure_blocks_bridge_and_model_reads(tmp_path, monkeypatch):
    args, calls = install_fixture_modules(
        monkeypatch,
        tmp_path,
        runtime_constructor_error=RuntimeError("AUTH_REQUIRED_BEFORE_DATA"),
    )

    with pytest.raises(RuntimeError, match="AUTH_REQUIRED_BEFORE_DATA"):
        runner.extract(args)

    assert calls["load_bridge"] == 0
    assert calls["import_old_engine"] == 0


def test_extract_complete_uses_cached_d2_and_normalized_oracle_without_f_loading(tmp_path, monkeypatch):
    args, calls = install_fixture_modules(monkeypatch, tmp_path)

    result = runner.extract(args)

    assert result["status"] == "COMPLETE"
    assert result["F_loaded"] is False
    assert result["F_forward_replayed"] is False
    assert result["future_observations_used"] is True
    assert result["past_only_forecasting_result"] is False
    assert result["original_codec_frozen_pass"] is True
    assert result["codec_before"] == result["codec_after"] == result["codec_terminal_hash"]
    assert sorted(result["artifacts"]) == ["fit", "tune", "val"]
    assert [name for name, _ in calls["save_npz"]] == [
        "ORACLE_DIAGNOSTIC_fit.npz",
        "ORACLE_DIAGNOSTIC_tune.npz",
        "ORACLE_DIAGNOSTIC_val.npz",
    ]
    assert len(calls["oracle_extract"]) == 3
    assert all(call["target_shape"] == (2, 48, 17) for call in calls["oracle_extract"])


def test_bad_d2_lineage_rejected_before_old_engine_import(tmp_path, monkeypatch):
    args, calls = install_fixture_modules(monkeypatch, tmp_path, d2_status="FAILED_PRESERVED")

    result = runner.extract(args)

    assert result["status"] == "FAILED_PRESERVED"
    assert result["error"]["message"] == "ORIGINAL_D2_COMPLETE_CP_AND_F_LINEAGE_REQUIRED"
    assert calls["import_old_engine"] == 0
    assert calls["oracle_extract"] == []


def test_mismatched_d2_cached_input_array_fails_before_oracle_extraction(tmp_path, monkeypatch):
    args, calls = install_fixture_modules(monkeypatch, tmp_path, cached_target_offset=1.0)

    result = runner.extract(args)

    assert result["status"] == "FAILED_PRESERVED"
    assert "OLD_D2_SAME_INPUT_ARRAY_REQUIRED" in result["error"]["message"]
    assert calls["oracle_extract"] == []


def test_prefix_parity_failure_is_not_complete(tmp_path, monkeypatch):
    args, calls = install_fixture_modules(monkeypatch, tmp_path, prefix_pass=False)

    result = runner.extract(args)

    assert result["status"] == "FAILED_PRESERVED"
    assert "ACTUAL_ENCODER_ORACLE_PREFIX_OR_CACHED_PARITY_NOT_PASSED" in result["error"]["message"]
    assert len(calls["oracle_extract"]) == 1


def test_resource_terminal_budget_status_overrides_local_complete(tmp_path, monkeypatch):
    args, _ = install_fixture_modules(monkeypatch, tmp_path, terminal_status="BUDGET_HELD_PRESERVED")

    result = runner.extract(args)

    assert result["status"] == "BUDGET_HELD_PRESERVED"
    assert result["resource_receipt"]["terminal_status"] == "BUDGET_HELD_PRESERVED"


def test_describe_cli_is_source_only_and_requires_no_actual_modules(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_stage_oracle_extract_v1.py", "--describe"])

    runner.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "SOURCE_ONLY_NOT_EXECUTED"
    assert payload["actual_assets_opened"] is False
    assert payload["scope"] == "frozen E future-oracle diagnostic only; cached F reused"
