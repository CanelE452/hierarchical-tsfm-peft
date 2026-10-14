from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from research.smallk_failure_followup_20261014 import prepare_appliances_inputs as pi


def test_guard_blocks_test_path_before_hash(tmp_path, monkeypatch):
    protected = tmp_path / "TEST" / "payload.npz"
    protected.parent.mkdir()
    protected.write_bytes(b"not-read")
    called = False

    def fail_if_called(path):
        nonlocal called
        called = True
        raise AssertionError(f"hash should not run for protected TEST path: {path}")

    monkeypatch.setattr(pi, "sha256_path", fail_if_called)

    with pytest.raises(PermissionError, match="PROTECTED_TEST_PATH_DENIED"):
        pi.checked_npz({"path": str(protected), "sha256": "unused"}, role="protected_payload")

    assert called is False


def test_import_source_registers_module_for_dataclass(tmp_path):
    source = tmp_path / "dataclass_source.py"
    source.write_text(
        "from dataclasses import dataclass\n"
        "@dataclass(frozen=True)\n"
        "class Record:\n"
        "    value: int\n"
        "EXAMPLE = Record(3)\n",
        encoding="utf-8",
    )

    module = pi.import_source(
        pi.FileAsset(str(source), hashlib.sha256(source.read_bytes()).hexdigest(), "synthetic_dataclass_source"),
        "synthetic_dataclass_source_for_appliance_bridge_test",
    )

    assert module.EXAMPLE.value == 3


def test_training_reconstruction_uses_physical_stats_and_mask():
    rng = np.random.default_rng(7)
    total = pi.L + pi.H + 3
    mean = np.linspace(0.2, 2.7, pi.C, dtype=np.float32)
    std = np.linspace(1.1, 3.6, pi.C, dtype=np.float32)
    physical = rng.normal(size=(total, pi.C)).astype(np.float32)
    mask = rng.random((total, pi.C)) > 0.2
    physical_with_missing = physical.copy()
    physical_with_missing[~mask] = np.nan
    origins = np.array([pi.L, pi.L + 1], dtype=np.int64)
    normalized = ((np.where(mask, physical, mean) - mean) / std).astype(np.float32)

    split = pi.reconstruct_training_split(
        {
            "physical": physical_with_missing,
            "mask": mask,
            "x": normalized,
            "origins": {"fit": origins},
        },
        "fit",
        mean,
        std,
    )

    expected_physical = np.stack([physical_with_missing[o - pi.L : o] for o in origins]).astype(np.float32)
    expected_observed = np.stack([mask[o - pi.L : o] for o in origins]).astype(bool)
    expected_history = pi.old_window_preprocess(expected_physical, expected_observed, mean, std)

    assert pi.arrays_equal_exact(split["history_physical"], expected_physical)
    assert pi.arrays_equal_exact(split["history_observed"], expected_observed)
    assert pi.arrays_equal_exact(split["history"], expected_history)
    assert pi.arrays_equal_exact(split["target"], np.stack([normalized[o : o + pi.H] for o in origins]).astype(np.float32))
    assert pi.arrays_equal_exact(split["observed"], np.stack([mask[o : o + pi.H] for o in origins]).astype(bool))
    assert pi.arrays_equal_exact(split["raw_last"], expected_history[:, -1].astype(np.float32))


def test_validate_reference_arrays_rejects_target_mismatch():
    reference = {"mean": np.zeros(pi.C, dtype=np.float32), "std": np.ones(pi.C, dtype=np.float32)}
    candidate = {"mean": reference["mean"].copy(), "std": reference["std"].copy()}
    for split, count in pi.EXPECTED_COUNTS.items():
        reference[f"{split}_target"] = np.zeros((count, pi.H, pi.C), dtype=np.float32)
        reference[f"{split}_observed"] = np.ones((count, pi.H, pi.C), dtype=bool)
        reference[f"{split}_origins"] = np.arange(count, dtype=np.int64)
        candidate[f"{split}_target"] = reference[f"{split}_target"].copy()
        candidate[f"{split}_observed"] = reference[f"{split}_observed"].copy()
        candidate[f"{split}_origins"] = reference[f"{split}_origins"].copy()
    candidate["tune_target"][0, 0, 0] = 1.0

    with pytest.raises(pi.ApplianceBridgeError, match="ARRAY_EXACT_MISMATCH:LEARNED:tune:target"):
        pi.validate_reference_arrays(reference, candidate, label="LEARNED")


def test_method_plan_locks_loss_comparison_without_auxiliary_selection_change():
    plan = pi.METHOD_PLAN

    assert plan["dataset"] == "appliances"
    assert plan["loss_candidates"] == ["MSE", "MAE", "HUBER_DELTA1"]
    assert plan["huber_delta"] == 1.0
    assert plan["updates"] == 600
    assert plan["batch_size"] == 4
    assert plan["checkpoints"] == [0, 100, 200, 400, 600]
    assert plan["selection_includes_u0"] is True
    assert "MSE" in plan["selection_primary_metric"]
    assert "MAE" not in plan["selection_primary_metric"].split("MSE", 1)[0]
    assert plan["future_alignment_auxiliary"] == "disabled_for_all_new_loss_candidates"
    assert "not loaded" in plan["q_aux_state"]


def test_load_prepared_bundle_verifies_npz_array_contract(tmp_path):
    arrays = {
        "history": np.zeros((1, pi.L, pi.C), dtype=np.float32),
        "history_physical": np.zeros((1, pi.L, pi.C), dtype=np.float32),
        "history_observed": np.ones((1, pi.L, pi.C), dtype=bool),
        "target": np.zeros((1, pi.H, pi.C), dtype=np.float32),
        "observed": np.ones((1, pi.H, pi.C), dtype=bool),
        "origins": np.array([pi.L], dtype=np.int64),
        "raw_last": np.zeros((1, pi.C), dtype=np.float32),
        "raw_forecast": np.zeros((1, pi.H, pi.C), dtype=np.float32),
        "raw_plus_r_forecast": np.zeros((1, pi.H, pi.C), dtype=np.float32),
        "mean": np.zeros(pi.C, dtype=np.float32),
        "std": np.ones(pi.C, dtype=np.float32),
        "learned_92601_forecast": np.zeros((1, pi.H, pi.C), dtype=np.float32),
        "learned_92602_forecast": np.zeros((1, pi.H, pi.C), dtype=np.float32),
    }
    artifacts = []
    for split in pi.SPLITS:
        path = tmp_path / f"appliances_{split}_inputs.npz"
        with path.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        artifacts.append({"split": split, "dataset": pi.DATASET, **pi.array_manifest(path, arrays)})

    result_path = tmp_path / "RESULT.json"
    result_path.write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "artifacts": artifacts,
                "columns": [f"c{i}" for i in range(pi.C)],
                "checkpoint_cases": {"92601": {}, "92602": {}},
                "raw_references": {"RAW_BOLT": {}, "RAW_BOLT_RAW_R": {}, "LEARNED_BOLT": {}},
                "method_plan": pi.METHOD_PLAN,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    bundle = pi.load_prepared_bundle(result_path)

    assert set(bundle["splits"]) == set(pi.SPLITS)
    assert bundle["mean"].dtype == np.float32
    assert bundle["std"].dtype == np.float32
    assert bundle["method_plan"]["loss_candidates"] == ["MSE", "MAE", "HUBER_DELTA1"]


def test_load_codec_checkpoint_state_cpu_and_q_preserved(tmp_path):
    checkpoint = tmp_path / "u600.pt"
    checkpoint.write_bytes(b"checkpoint-payload")
    calls = []

    class FakeTorch:
        @staticmethod
        def load(path, *, map_location, weights_only):
            calls.append((Path(path), map_location, weights_only))
            return {
                "codec": {"fixed_basis": np.eye(2, dtype=np.float32)},
                "q": {"unused": np.ones(2, dtype=np.float32)},
                "update": 600,
                "seed": 92601,
                "architecture": "ANCHOR_NONLINEAR_D",
                "method": "LEARNED_REGIME",
                "weight": 0.1,
            }

    payload = pi.load_codec_checkpoint_state(
        {"path": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
        torch_module=FakeTorch,
    )

    assert calls == [(checkpoint, "cpu", True)]
    assert payload["codec"]["fixed_basis"].shape == (2, 2)
    assert payload["q"]["unused"].shape == (2,)
    assert payload["q_aux_deployment_use"] == "preserved_not_used"
