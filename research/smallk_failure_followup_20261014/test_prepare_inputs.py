"""Synthetic CPU checks for the old-condition input bridge.

These tests do not open previous scientific NPZ/checkpoint/model assets.
"""
from types import SimpleNamespace
import json

import numpy as np
import pytest

from research.smallk_failure_followup_20261014 import prepare_inputs as pi
from research.smallk_failure_followup_20261014 import followup_runtime as fr


def make_raw_prediction_bundle(spec, mean=None, std=None):
    rng = np.random.default_rng(92601 + spec.channels)
    mean = np.zeros(spec.channels, np.float32) if mean is None else np.asarray(mean, np.float32)
    std = np.ones(spec.channels, np.float32) if std is None else np.asarray(std, np.float32)
    split_payloads = {}
    base_origin = pi.L + 10
    for index, split in enumerate(spec.splits):
        n = spec.counts[split]
        origins = base_origin + 20000 * index + np.arange(n, dtype=np.int64) * pi.H
        prediction = rng.normal(size=(n, pi.H, spec.channels)).astype(np.float32)
        target = rng.normal(size=(n, pi.H, spec.channels)).astype(np.float32)
        observed = np.ones((n, pi.H, spec.channels), bool)
        if n:
            observed[0, 0, 0] = False
            target[~observed] = np.float32(123.0)
        split_payloads[split] = {
            "prediction": prediction,
            "target": target,
            "observed": observed,
            "origins": origins,
        }
    return {"mean": mean, "std": std, "split_payloads": split_payloads}


def make_physical_source(spec, raw, mean, std):
    max_origin = max(int(raw["split_payloads"][split]["origins"][-1]) for split in spec.splits)
    rows = max_origin + pi.H + 5
    channels = spec.channels
    physical = np.zeros((rows, channels), np.float32)
    mask = np.ones((rows, channels), bool)
    for t in range(rows):
        physical[t] = np.arange(channels, dtype=np.float32) + t / 1000.0
    mask[:, 0] = True
    for split in spec.splits:
        for i, origin in enumerate(raw["split_payloads"][split]["origins"]):
            target = raw["split_payloads"][split]["target"][i]
            observed = raw["split_payloads"][split]["observed"][i]
            physical[origin : origin + pi.H] = target * std + mean
            mask[origin : origin + pi.H] = observed
    return physical, mask


def test_robin_uses_stored_raw_stats_and_old_window_preprocess():
    spec = pi.ROBIN
    stored_mean = np.linspace(-3, 3, spec.channels, dtype=np.float32)
    stored_std = np.linspace(1, 2, spec.channels, dtype=np.float32)
    raw = make_raw_prediction_bundle(spec, stored_mean, stored_std)
    physical, mask = make_physical_source(spec, raw, stored_mean, stored_std)
    first_origin = raw["split_payloads"]["fit"]["origins"][0]
    mask[first_origin - pi.L, 3] = False
    physical[first_origin - pi.L, 3] = np.nan
    common_mean = stored_mean + np.float32(100.0)
    common_std = stored_std + np.float32(10.0)
    data = {
        "physical": physical,
        "mask": mask,
        "mean": common_mean,
        "std": common_std,
        "origins": {split: raw["split_payloads"][split]["origins"].copy() for split in spec.splits},
        "columns": [f"c{i}" for i in range(spec.channels)],
        "source": "synthetic",
        "source_hash": "synthetic_sha",
    }
    common = SimpleNamespace(load_data=lambda *args, **kwargs: data)
    ex = SimpleNamespace(preprocess=pi.old_window_preprocess)
    bundle = pi.reconstruct_robin_inputs(raw=raw, common_module=common, experiment_module=ex)
    fit = bundle["split_payloads"]["fit"]
    expected_history = pi.old_window_preprocess(
        np.stack([physical[o - pi.L : o] for o in raw["split_payloads"]["fit"]["origins"]]),
        np.stack([mask[o - pi.L : o] for o in raw["split_payloads"]["fit"]["origins"]]),
        stored_mean,
        stored_std,
    )
    assert np.array_equal(bundle["mean"], stored_mean)
    assert np.array_equal(bundle["std"], stored_std)
    assert np.array_equal(bundle["common_load_mean"], common_mean)
    assert np.array_equal(fit["history"], expected_history)
    assert np.array_equal(
        fit["physical_past"],
        np.stack([physical[o - pi.L : o] for o in raw["split_payloads"]["fit"]["origins"]]),
        equal_nan=True,
    )
    assert np.array_equal(fit["history_physical"], fit["physical_past"], equal_nan=True)
    assert np.array_equal(fit["physical_past_mask"], np.stack([mask[o - pi.L : o] for o in raw["split_payloads"]["fit"]["origins"]]))
    assert np.array_equal(fit["raw_last"], fit["history"][:, -1])
    assert fit["history"].dtype == np.float32


def test_target_mask_and_origins_are_preserved_from_canonical_raw():
    spec = pi.ROBIN
    raw = make_raw_prediction_bundle(spec)
    physical, mask = make_physical_source(spec, raw, raw["mean"], raw["std"])
    payload = pi.build_split_from_physical(
        split="fit",
        origins=raw["split_payloads"]["fit"]["origins"],
        physical=physical,
        mask=mask,
        raw_prediction_payload=raw["split_payloads"]["fit"],
        mean=raw["mean"],
        std=raw["std"],
    )
    assert np.array_equal(payload["target"], raw["split_payloads"]["fit"]["target"])
    assert np.array_equal(payload["observed"], raw["split_payloads"]["fit"]["observed"])
    assert np.array_equal(payload["origins"], raw["split_payloads"]["fit"]["origins"])


def test_test_split_is_rejected_by_dataset_validation():
    raw = make_raw_prediction_bundle(pi.ROBIN)
    split = raw["split_payloads"]["fit"]
    payload = {
        "history": np.zeros((1, pi.L, pi.ROBIN.channels), np.float32),
        "history_physical": np.zeros((1, pi.L, pi.ROBIN.channels), np.float32),
        "physical_past": np.zeros((1, pi.L, pi.ROBIN.channels), np.float32),
        "raw_forecast": split["prediction"][:1],
        "target": split["target"][:1],
        "observed": split["observed"][:1],
        "history_observed": np.ones((1, pi.L, pi.ROBIN.channels), bool),
        "physical_past_mask": np.ones((1, pi.L, pi.ROBIN.channels), bool),
        "origins": split["origins"][:1],
        "raw_last": np.zeros((1, pi.ROBIN.channels), np.float32),
    }
    bundle = {
        "splits": ("fit", "tune", "test"),
        "mean": raw["mean"],
        "std": raw["std"],
        "split_payloads": {"fit": payload, "tune": payload, "test": payload},
    }
    with pytest.raises(pi.InputBridgeError, match="SPLIT_ORDER"):
        pi.validate_dataset_bundle(pi.ROBIN, bundle)


def test_reference_equivalence_detects_target_or_stat_drift(monkeypatch):
    spec = pi.ROBIN
    raw = make_raw_prediction_bundle(spec)
    candidate = {
        "asset": pi.FileAsset("synthetic"),
        "mean": raw["mean"].copy(),
        "std": raw["std"].copy(),
        "split_payloads": {
            split: {key: value.copy() for key, value in payload.items()}
            for split, payload in raw["split_payloads"].items()
        },
    }
    candidate["split_payloads"]["val"]["target"][0, 0, 1] += np.float32(1.0)
    monkeypatch.setattr(pi, "load_prediction_bundle", lambda asset, spec: candidate)
    monkeypatch.setattr(pi, "ensure_file_hash", lambda asset: {"path": "synthetic", "sha256": "x"})
    with pytest.raises(pi.InputBridgeError, match="REFERENCE_TARGET_MISMATCH"):
        pi.compare_prediction_reference(spec, raw, pi.FileAsset("synthetic"))


def test_reference_equivalence_allows_matching_masked_target_nans(monkeypatch):
    spec = pi.ROBIN
    raw = make_raw_prediction_bundle(spec)
    raw["split_payloads"]["val"]["target"][0, 0, 0] = np.nan
    candidate = {
        "asset": pi.FileAsset("synthetic"),
        "mean": raw["mean"].copy(),
        "std": raw["std"].copy(),
        "split_payloads": {
            split: {key: value.copy() for key, value in payload.items()}
            for split, payload in raw["split_payloads"].items()
        },
    }
    monkeypatch.setattr(pi, "load_prediction_bundle", lambda asset, spec: candidate)
    monkeypatch.setattr(pi, "ensure_file_hash", lambda asset: {"path": "synthetic", "sha256": "x"})
    record = pi.compare_prediction_reference(spec, raw, pi.FileAsset("synthetic"))
    assert record["asset"]["sha256"] == "x"


def test_phase_barrier_blocks_incomplete_bi_before_work(tmp_path):
    audit = tmp_path / "audit.json"
    manifest = tmp_path / "manifest.json"
    report = tmp_path / "report.md"
    audit.write_text(json.dumps({"status": "PASS", "whole_experiment_complete": False, "failed_count": 0}), encoding="utf-8")
    manifest.write_text(json.dumps({"status": "GENERATED_FROM_ACTUAL_TERMINAL_ARTIFACTS", "files": {}}), encoding="utf-8")
    report.write_text("done", encoding="utf-8")
    with pytest.raises(pi.PhaseBarrierError, match="WHOLE_EXPERIMENT_INCOMPLETE"):
        pi.verify_phase_barrier(audit, report, manifest)


def test_phase_barrier_accepts_completed_audit_report_manifest(tmp_path):
    audit = tmp_path / "audit.json"
    manifest = tmp_path / "manifest.json"
    report = tmp_path / "report.md"
    audit.write_text(json.dumps({"status": "PASS", "whole_experiment_complete": True, "failed_count": 0, "checks": [{"passed": True}]}), encoding="utf-8")
    report.write_text("[확인] completed; historic FAILED_PATH_PRESERVED rows remain documented", encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "status": "GENERATED_FROM_ACTUAL_TERMINAL_ARTIFACTS",
                "files": {"audit.json": pi.sha256_path(audit), "report.md": pi.sha256_path(report)},
            }
        ),
        encoding="utf-8",
    )
    barrier = pi.verify_phase_barrier(audit, report, manifest)
    assert barrier["audit"]["sha256"] == pi.sha256_path(audit)
    assert barrier["manifest_report_sha256"] == pi.sha256_path(report)


def test_phase_barrier_rejects_manifest_sha_mismatch(tmp_path):
    audit = tmp_path / "AUDIT_RESULT.json"
    manifest = tmp_path / "MANIFEST.json"
    report = tmp_path / "REPORT_KO.md"
    audit.write_text(json.dumps({"status": "PASS", "whole_experiment_complete": True, "failed_count": 0, "checks": []}), encoding="utf-8")
    report.write_text("complete", encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "status": "GENERATED_FROM_ACTUAL_TERMINAL_ARTIFACTS",
                "files": {"AUDIT_RESULT.json": "bad", "REPORT_KO.md": pi.sha256_path(report)},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(pi.PhaseBarrierError, match="MANIFEST_AUDIT_SHA_MISMATCH"):
        pi.verify_phase_barrier(audit, report, manifest)


def test_common_source_uses_original_path_and_keeps_archive_provenance():
    assert pi.SOURCE_ASSETS["common"].path == "research/past_only_repr_diagnosis_20261012/common.py"
    assert pi.SOURCE_ASSETS["common_archive"].path == "research/smallk_predictive_alignment_20261013/run_dev_v1/sources/common.py"


def test_beijing_repair_physical_payload_must_preexist(monkeypatch, tmp_path):
    input_path = tmp_path / "eval_past_input_payload.npz"
    input_path.write_bytes(b"placeholder")
    contract = {"payloads": {"eval_past_input": {"path": str(input_path)}}}
    repair = SimpleNamespace(base=SimpleNamespace(sha256_path=pi.sha256_path))
    with pytest.raises(pi.InputBridgeError, match="NOT_PREEXISTING"):
        pi.ensure_existing_beijing_repair_payloads(repair, contract)


def test_beijing_reconstruction_uses_stored_eval_history_without_open_evaluation(monkeypatch, tmp_path):
    spec = pi.BEIJING
    raw = make_raw_prediction_bundle(spec)
    physical, mask = make_physical_source(spec, raw, raw["mean"], raw["std"])
    eval_history = np.zeros((spec.counts["eval"], pi.L, spec.channels), np.float32)
    eval_history_observed = np.ones_like(eval_history, bool)
    eval_physical = np.where(eval_history_observed, np.float32(7.0), np.nan).astype(np.float32)
    eval_origins = raw["split_payloads"]["eval"]["origins"]
    input_path = tmp_path / "eval_past_input_payload.npz"
    physical_payload = tmp_path / "repair_eval_past_physical_payload.npz"
    manifest = tmp_path / "repair_eval_past_input_manifest.json"
    input_path.write_bytes(b"input")
    physical_payload.write_bytes(b"physical")
    manifest.write_text(json.dumps({"history_physical_sha256": pi.sha256_path(physical_payload)}), encoding="utf-8")

    class FakeRepair:
        base = SimpleNamespace(sha256_path=pi.sha256_path)

        def get_contract(self, *args, **kwargs):
            return {
                "contract_sha256": "contract",
                "payloads": {
                    "eval_past_input": {"path": str(input_path)},
                    "fit_tune": {"sha256": "fit_tune"},
                },
            }

        def load_training(self, *args, **kwargs):
            return {
                "physical": physical,
                "mask": mask,
                "mean": raw["mean"],
                "std": raw["std"],
                "columns": [f"b{i}" for i in range(spec.channels)],
                "fit_tune_payload_sha256": "fit_tune",
            }

        def load_evaluation_inputs(self, *args, **kwargs):
            return {
                "history": eval_history,
                "history_observed": eval_history_observed,
                "history_physical": eval_physical,
                "origins": eval_origins,
                "mean": raw["mean"],
                "std": raw["std"],
                "eval_input_payload_sha256": "eval_input",
                "repair_history_physical_payload_sha256": pi.sha256_path(physical_payload),
            }

        def open_evaluation(self, *args, **kwargs):  # pragma: no cover - should never be called.
            raise AssertionError("open_evaluation must not be called")

    bundle = pi.reconstruct_beijing_inputs(raw=raw, repair_module=FakeRepair(), experiment_module=SimpleNamespace(preprocess=pi.old_window_preprocess))
    assert np.array_equal(bundle["split_payloads"]["eval"]["history"], eval_history)
    assert np.array_equal(bundle["split_payloads"]["eval"]["history_physical"], eval_physical, equal_nan=True)
    assert np.array_equal(bundle["split_payloads"]["eval"]["physical_past"], eval_physical, equal_nan=True)
    assert np.array_equal(bundle["split_payloads"]["eval"]["target"], raw["split_payloads"]["eval"]["target"])
    assert bundle["lineage"]["eval_target_source"].startswith("canonical exposed RAW")


def test_save_split_npz_is_exclusive_and_records_hashes(tmp_path):
    raw = make_raw_prediction_bundle(pi.ROBIN)
    physical, mask = make_physical_source(pi.ROBIN, raw, raw["mean"], raw["std"])
    split_payload = pi.build_split_from_physical(
        split="fit",
        origins=raw["split_payloads"]["fit"]["origins"],
        physical=physical,
        mask=mask,
        raw_prediction_payload=raw["split_payloads"]["fit"],
        mean=raw["mean"],
        std=raw["std"],
    )
    bundle = {
        "mean": raw["mean"],
        "std": raw["std"],
        "columns": np.array([f"c{i}" for i in range(pi.ROBIN.channels)]),
        "common_load_mean": raw["mean"],
        "common_load_std": raw["std"],
        "split_payloads": {"fit": split_payload},
    }
    record = pi.save_split_npz(tmp_path, "robin", "fit", bundle)
    assert (tmp_path / "robin_fit_input.npz").is_file()
    assert record["arrays"]["history"]["sha256"] == pi.array_sha256(split_payload["history"])
    assert record["arrays"]["history_physical"]["sha256"] == pi.array_sha256(split_payload["history_physical"])
    assert record["arrays"]["physical_past"]["sha256"] == pi.array_sha256(split_payload["physical_past"])
    with pytest.raises(FileExistsError):
        pi.save_split_npz(tmp_path, "robin", "fit", bundle)


def test_execute_after_bi_uses_followup_limit_and_preserves_runner_keys(monkeypatch, tmp_path):
    output = tmp_path / "cache" / "input_bridge" / "run"
    monkeypatch.setattr(pi, "CACHE", tmp_path / "cache" / "input_bridge")
    raw_by_key = {pi.ROBIN.key: make_raw_prediction_bundle(pi.ROBIN), pi.BEIJING.key: make_raw_prediction_bundle(pi.BEIJING)}
    monkeypatch.setattr(pi, "install_protected_test_guard", lambda: ["guarded"])
    monkeypatch.setattr(pi, "verify_declared_assets_for_actual", lambda: {"source": {}, "predictions": {}, "checkpoints_file_hash_only": {}})
    monkeypatch.setattr(pi, "load_prediction_bundle", lambda asset, spec: raw_by_key[spec.key])
    monkeypatch.setattr(pi, "verify_reference_assets", lambda spec, raw: [{"name": "main_92601"}])

    def fake_bundle(spec):
        raw = raw_by_key[spec.key]
        split_payloads = {}
        for split, values in raw["split_payloads"].items():
            n = len(values["origins"])
            split_payloads[split] = {
                "history": np.zeros((n, pi.L, spec.channels), np.float32),
                "history_physical": np.full((n, pi.L, spec.channels), np.nan, np.float32),
                "physical_past": np.full((n, pi.L, spec.channels), np.nan, np.float32),
                "raw_forecast": values["prediction"],
                "target": values["target"],
                "observed": values["observed"],
                "history_observed": np.zeros((n, pi.L, spec.channels), bool),
                "physical_past_mask": np.zeros((n, pi.L, spec.channels), bool),
                "origins": values["origins"],
                "raw_last": np.zeros((n, spec.channels), np.float32),
            }
        return {
            "dataset": spec.key,
            "splits": spec.splits,
            "mean": raw["mean"],
            "std": raw["std"],
            "columns": np.array([f"c{i}" for i in range(spec.channels)]),
            "split_payloads": split_payloads,
            "common_load_mean": raw["mean"],
            "common_load_std": raw["std"],
            "lineage": {},
        }

    monkeypatch.setattr(pi, "reconstruct_robin_inputs", lambda raw: fake_bundle(pi.ROBIN))
    monkeypatch.setattr(pi, "reconstruct_beijing_inputs", lambda raw: fake_bundle(pi.BEIJING))
    calls = []

    class FakeLimit:
        def __init__(self, tag, gpu, followup_roots=()):
            calls.append(("init", tag, gpu, tuple(followup_roots)))

        def check(self):
            calls.append(("check",))
            return {"budget_counted_seconds": 1.0}

        def close(self, status="COMPLETE", error=None):
            calls.append(("close", status, error))
            return {"path": "research/bihash_future_prediction_20261013/resources/fake.json", "sha256": "resource"}

    monkeypatch.setattr(pi, "load_followup_limit_class", lambda: FakeLimit)
    result = pi.execute_after_bi(
        output,
        {"audit": {"path": "a", "sha256": "a"}, "report": {"path": "r", "sha256": "r"}, "manifest": {"path": "m", "sha256": "m"}},
    )
    assert result["declared_asset_hashes"].keys() >= {"source", "predictions", "checkpoints_file_hash_only"}
    assert result["resource_terminal"]["sha256"] == "resource"
    assert result["artifacts"][0]["arrays"]["history_physical"]["dtype"] == "float32"
    assert result["artifacts"][0]["arrays"]["physical_past"]["dtype"] == "float32"
    assert calls[0][0:3] == ("init", "input_bridge", False)


def test_followup_limit_counts_supplement_and_returns_jsonable_receipt(monkeypatch, tmp_path):
    class BudgetHeld(RuntimeError):
        pass

    class FakeSharedLimit:
        def __init__(self, tag, gpu=True):
            self.tag = tag
            self.gpu = gpu

        def check(self):
            return {"elapsed_seconds": 1.0, "budget_counted_seconds": 10.0, "storage_bytes": 20}

        def close(self, status="COMPLETE", error=None):
            return {"path": "research/bihash_future_prediction_20261013/resources/fake.json", "sha256": "abc", "status": status, "error": error}

    class FakeRuntime:
        HERE = tmp_path / "bi"
        ROOT = tmp_path
        CPU_SECONDS = 100.0
        STORAGE_BYTES = 1000
        Limit = FakeSharedLimit

        @staticmethod
        def read(path):
            return {}

    FakeRuntime.BudgetHeld = BudgetHeld
    (FakeRuntime.HERE / "resources").mkdir(parents=True)
    monkeypatch.setattr(fr, "load_bi_runtime", lambda: FakeRuntime)
    limit = fr.FollowupLimit(
        "probe",
        False,
        followup_roots=(tmp_path / "missing",),
        supplementary_cpu_seconds=5.0,
        include_report_body_seconds=False,
    )
    check = limit.check()
    assert check["cpu_budget_counted_with_supplement"] == 15.0
    receipt = limit.close("COMPLETE", {"ok": True})
    json.dumps(receipt)
    assert receipt["sha256"] == "abc"


def test_followup_limit_counts_report_bodies_once(monkeypatch, tmp_path):
    class BudgetHeld(RuntimeError):
        pass

    class FakeSharedLimit:
        def __init__(self, tag, gpu=True):
            self.tag = tag
            self.gpu = gpu

        def check(self):
            return {"elapsed_seconds": 1.0, "budget_counted_seconds": 10.0, "storage_bytes": 20}

        def close(self, status="COMPLETE", error=None):
            return {"path": "resource.json", "sha256": "abc", "status": status, "error": error}

    class FakeRuntime:
        HERE = tmp_path / "bi"
        ROOT = tmp_path
        CPU_SECONDS = 1000.0
        STORAGE_BYTES = 1000
        Limit = FakeSharedLimit

        @staticmethod
        def read(path):
            return {}

    calls = []
    FakeRuntime.BudgetHeld = BudgetHeld
    (FakeRuntime.HERE / "resources").mkdir(parents=True)
    monkeypatch.setattr(fr, "load_bi_runtime", lambda: FakeRuntime)
    monkeypatch.setattr(fr, "verified_report_body_seconds", lambda: calls.append("reports") or 100.0)
    limit = fr.FollowupLimit("probe", False, followup_roots=(tmp_path / "missing",), supplementary_cpu_seconds=5.0)
    first = limit.check()
    second = limit.check()
    assert first["cpu_budget_counted_with_supplement"] == 115.0
    assert second["supplementary_cpu_seconds_counted"] == 105.0
    assert calls == ["reports"]


def test_followup_limit_constructor_check_failure_closes_shared_once(monkeypatch, tmp_path):
    class BudgetHeld(RuntimeError):
        pass

    closed = []

    class FakeSharedLimit:
        def __init__(self, tag, gpu=True):
            self.tag = tag
            self.gpu = gpu

        def check(self):
            raise BudgetHeld("constructor check budget")

        def close(self, status="COMPLETE", error=None):
            closed.append((status, error))
            return {"path": "resource.json", "sha256": "abc", "status": status, "error": error}

    class FakeRuntime:
        HERE = tmp_path / "bi"
        ROOT = tmp_path
        CPU_SECONDS = 1000.0
        STORAGE_BYTES = 1000
        Limit = FakeSharedLimit

        @staticmethod
        def read(path):
            return {}

    FakeRuntime.BudgetHeld = BudgetHeld
    (FakeRuntime.HERE / "resources").mkdir(parents=True)
    monkeypatch.setattr(fr, "load_bi_runtime", lambda: FakeRuntime)
    with pytest.raises(BudgetHeld):
        fr.FollowupLimit("probe", False, followup_roots=(tmp_path / "missing",), include_report_body_seconds=False)
    assert len(closed) == 1
    assert closed[0][0] == "BUDGET_HELD_PRESERVED"
    assert closed[0][1]["phase"] == "constructor_check"
    assert fr.LAST_CONSTRUCTOR_TERMINAL["terminal_error"]["phase"] == "constructor_check"


def test_followup_limit_constructor_limit_failure_cannot_close(monkeypatch, tmp_path):
    class BudgetHeld(RuntimeError):
        pass

    class ExplodingLimit:
        closed = 0

        def __init__(self, tag, gpu=True):
            raise RuntimeError("limit constructor failure")

        def close(self, status="COMPLETE", error=None):  # pragma: no cover - should not be reachable.
            type(self).closed += 1

    class FakeRuntime:
        HERE = tmp_path / "bi"
        ROOT = tmp_path
        CPU_SECONDS = 1000.0
        STORAGE_BYTES = 1000
        Limit = ExplodingLimit

        @staticmethod
        def read(path):
            return {}

    FakeRuntime.BudgetHeld = BudgetHeld
    (FakeRuntime.HERE / "resources").mkdir(parents=True)
    monkeypatch.setattr(fr, "load_bi_runtime", lambda: FakeRuntime)
    with pytest.raises(RuntimeError, match="limit constructor failure"):
        fr.FollowupLimit("probe", False, followup_roots=(tmp_path / "missing",), include_report_body_seconds=False)
    assert ExplodingLimit.closed == 0


def test_followup_limit_close_check_failure_returns_terminal_failure(monkeypatch, tmp_path):
    class BudgetHeld(RuntimeError):
        pass

    closed = []

    class FakeSharedLimit:
        def __init__(self, tag, gpu=True):
            self.tag = tag
            self.gpu = gpu
            self.check_count = 0

        def check(self):
            self.check_count += 1
            if self.check_count >= 2:
                raise BudgetHeld("final check budget")
            return {"elapsed_seconds": 1.0, "budget_counted_seconds": 10.0, "storage_bytes": 20}

        def close(self, status="COMPLETE", error=None):
            closed.append((status, error))
            return {"path": "resource.json", "sha256": "abc", "status": status, "error": error}

    class FakeRuntime:
        HERE = tmp_path / "bi"
        ROOT = tmp_path
        CPU_SECONDS = 1000.0
        STORAGE_BYTES = 1000
        Limit = FakeSharedLimit

        @staticmethod
        def read(path):
            return {}

    FakeRuntime.BudgetHeld = BudgetHeld
    (FakeRuntime.HERE / "resources").mkdir(parents=True)
    monkeypatch.setattr(fr, "load_bi_runtime", lambda: FakeRuntime)
    limit = fr.FollowupLimit("probe", False, followup_roots=(tmp_path / "missing",), include_report_body_seconds=False)
    receipt = limit.close("COMPLETE", {"ok": True})
    json.dumps(receipt)
    assert receipt["terminal_status"] == "BUDGET_HELD_PRESERVED"
    assert receipt["terminal_error"]["check_failure"]["type"] == "BudgetHeld"
    assert receipt["check_failure"]["type"] == "BudgetHeld"
    assert closed[0][0] == "BUDGET_HELD_PRESERVED"


def test_describe_is_preparation_only():
    desc = pi.describe()
    assert desc["status"] == "PREPARATION_ONLY"
    assert desc["actual_status"]["old_npz_opened"] is False
    assert "test" not in desc["datasets"]["robin"]["splits"]
    assert desc["runtime_api"] == "followup_runtime.FollowupLimit(tag, gpu)"
