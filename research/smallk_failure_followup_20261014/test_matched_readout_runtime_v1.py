from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import matched_readout_runtime_v1 as rt  # noqa: E402


class FakeProcess:
    def __init__(self, *, created_unix: float = 1001.5, rss: int = 4096):
        self._created_unix = created_unix
        self._rss = rss

    def create_time(self):
        return self._created_unix

    def memory_info(self):
        return SimpleNamespace(rss=self._rss)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def rehome_runtime(monkeypatch, tmp_path):
    plan = tmp_path / "PLAN_MATCHED_READOUT_INFORMATION_KO_v1.md"
    method = tmp_path / "METHOD_LOCK_MATCHED_READOUT_v1.md"
    core = tmp_path / "matched_readout_runtime_v1.py"
    scanner = tmp_path / "fast_storage_scan_v1.py"
    plan.write_text("synthetic matched readout plan", encoding="utf-8")
    method.write_text("synthetic method lock", encoding="utf-8")
    core.write_text("synthetic runtime source", encoding="utf-8")
    scanner.write_text("def storage_bytes(roots): return 0\n", encoding="utf-8")
    monkeypatch.setattr(rt, "HERE", tmp_path)
    monkeypatch.setattr(rt, "ROOT", tmp_path)
    monkeypatch.setattr(rt, "APPROVED_PLAN", plan)
    monkeypatch.setattr(rt, "METHOD_LOCK", method)
    monkeypatch.setattr(rt, "CORE_RUNTIME", core)
    monkeypatch.setattr(rt, "AUTHORIZATION", tmp_path / "matched_readout_authorization_v1.json")
    monkeypatch.setattr(rt, "RUN_MANIFEST", tmp_path / "matched_readout_run_manifest_v1.json")
    monkeypatch.setattr(rt, "RESOURCES", tmp_path / "matched_readout_resources_v1")
    monkeypatch.setattr(rt, "ACTIVE", tmp_path / "matched_readout_resources_v1" / "ACTIVE.json")
    monkeypatch.setattr(rt, "LAST_CONSTRUCTOR_TERMINAL", None)
    monkeypatch.setattr(rt, "storage_bytes", lambda roots: 0)


def install_fake_psutil(monkeypatch, *, created_unix: float = 1001.5, rss: int = 4096):
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process=lambda: FakeProcess(created_unix=created_unix, rss=rss)))


def write_authorization_and_manifest(monkeypatch, tmp_path, *, limits=None, approved_unix=1000.0,
                                     started_unix=1001.0, protected_prior_receipts=None,
                                     storage_roots=None):
    rehome_runtime(monkeypatch, tmp_path)
    limits = dict(rt.MAX_LIMITS if limits is None else limits)
    storage_root = tmp_path / "matched_outputs"
    storage_root.mkdir(parents=True, exist_ok=True)
    auth = {
        "schema": "smallk_matched_readout_authorization_v1",
        "status": "APPROVED",
        "origin": "human_user_response",
        "response_identifier": "synthetic-call:0",
        "user_response": "synthetic human approved matched readout resources",
        "phase": "matched_readout_information_v1",
        "approved_unix": approved_unix,
        "approved_plan_sha256": rt.sha(rt.APPROVED_PLAN),
        "method_lock_sha256": rt.sha(rt.METHOD_LOCK),
        "runtime_source_sha256": rt.sha(rt.CORE_RUNTIME),
        "gpu_allowed": False,
        "cuda_allowed": False,
        "F_access_allowed": False,
        "limits": limits,
    }
    write_json(rt.AUTHORIZATION, auth)
    if protected_prior_receipts is None:
        prior = tmp_path / "protected_prior_receipts" / "stage_report_receipt.json"
        write_json(prior, {"status": "COMPLETE", "tag": "prior_stage", "gpu": False, "wall_seconds": 0.125})
        protected_prior_receipts = [{"path": str(prior), "sha256": rt.sha(prior)}]
    if storage_roots is None:
        storage_roots = [str(storage_root)]
    run = {
        "schema": "smallk_matched_readout_run_manifest_v1",
        "phase": "matched_readout_information_v1",
        "authorization_sha256": rt.sha(rt.AUTHORIZATION),
        "runtime_source_sha256": rt.sha(rt.CORE_RUNTIME),
        "started_unix": started_unix,
        "expires_unix": started_unix + limits["campaign_wall_seconds"],
        "baseline_storage_bytes": 0,
        "storage_roots": storage_roots,
        "protected_prior_receipts": protected_prior_receipts,
        "gpu_allowed": False,
        "cuda_allowed": False,
        "F_access_allowed": False,
    }
    write_json(rt.RUN_MANIFEST, run)
    return auth, run


def install_clock(monkeypatch, *, now=1002.0, perf=0.0, ns=123456789):
    monkeypatch.setattr(rt.time, "time", lambda: now)
    monkeypatch.setattr(rt.time, "perf_counter", lambda: perf)
    monkeypatch.setattr(rt.time, "time_ns", lambda: ns)


def test_missing_authorization_records_constructor_terminal_without_receipt(tmp_path, monkeypatch):
    rehome_runtime(monkeypatch, tmp_path)
    with pytest.raises(rt.AuthorizationRequired, match="EXPLICIT_HUMAN_APPROVAL"):
        rt.ContinuationLimit("missing_auth", gpu=False)
    assert rt.LAST_CONSTRUCTOR_TERMINAL["terminal_status"] == "FAILED_PRESERVED"
    assert "pre-authorization" in rt.LAST_CONSTRUCTOR_TERMINAL["scope"]


def test_gpu_true_and_nonzero_gpu_limit_are_rejected(tmp_path, monkeypatch):
    write_authorization_and_manifest(monkeypatch, tmp_path)
    install_fake_psutil(monkeypatch)
    install_clock(monkeypatch)
    with pytest.raises(ValueError, match="GPU_FORBIDDEN"):
        rt.ContinuationLimit("gpu_true", gpu=True)

    limits = dict(rt.MAX_LIMITS)
    limits["gpu_process_wall_seconds"] = 1
    write_authorization_and_manifest(monkeypatch, tmp_path, limits=limits)
    with pytest.raises(rt.AuthorizationRequired, match="EXPLICIT_HUMAN_APPROVAL_RECORD_INVALID"):
        rt.authorized_records()


def test_auth_manifest_bind_human_plan_method_runtime_and_exact_expiry(tmp_path, monkeypatch):
    auth, run = write_authorization_and_manifest(monkeypatch, tmp_path)
    auth["origin"] = "agent_summary"
    write_json(rt.AUTHORIZATION, auth)
    with pytest.raises(rt.AuthorizationRequired, match="EXPLICIT_HUMAN_APPROVAL_RECORD_INVALID"):
        rt.authorized_records()

    auth["origin"] = "human_user_response"
    auth["runtime_source_sha256"] = "0" * 64
    write_json(rt.AUTHORIZATION, auth)
    with pytest.raises(rt.AuthorizationRequired, match="EXPLICIT_HUMAN_APPROVAL_RECORD_INVALID"):
        rt.authorized_records()

    auth["runtime_source_sha256"] = rt.sha(rt.CORE_RUNTIME)
    write_json(rt.AUTHORIZATION, auth)
    run["expires_unix"] += 1
    write_json(rt.RUN_MANIFEST, run)
    with pytest.raises(rt.AuthorizationRequired, match="FIXED_AUTHORIZATION_BOUND"):
        rt.authorized_records()


def test_empty_protected_receipts_or_roots_rejected(tmp_path, monkeypatch):
    write_authorization_and_manifest(monkeypatch, tmp_path, protected_prior_receipts=[])
    with pytest.raises(rt.AuthorizationRequired, match="PRESERVED_PRIOR_RECEIPTS"):
        rt.authorized_records()
    write_authorization_and_manifest(monkeypatch, tmp_path, storage_roots=[])
    with pytest.raises(rt.AuthorizationRequired, match="PRESERVED_PRIOR_RECEIPTS"):
        rt.authorized_records()


def test_prior_stage_or_continuation_active_lease_blocks_without_private_active(tmp_path, monkeypatch):
    write_authorization_and_manifest(monkeypatch, tmp_path)
    install_fake_psutil(monkeypatch)
    install_clock(monkeypatch)
    write_json(tmp_path / "stage_separation_resources_v1" / "ACTIVE.json", {"pid": 44})
    with pytest.raises(RuntimeError, match="PRIOR_PHASE_ACTIVE_LEASE"):
        rt.ContinuationLimit("stage_active", gpu=False)
    assert not rt.ACTIVE.exists()

    (tmp_path / "stage_separation_resources_v1" / "ACTIVE.json").unlink()
    write_json(tmp_path / "continuation_resources_v1" / "ACTIVE.json", {"pid": 45})
    with pytest.raises(RuntimeError, match="PRIOR_PHASE_ACTIVE_LEASE"):
        rt.ContinuationLimit("continuation_active", gpu=False)
    assert not rt.ACTIVE.exists()


def test_mutated_protected_prior_receipt_closes_constructor_failure_and_unlinks_active(tmp_path, monkeypatch):
    write_authorization_and_manifest(monkeypatch, tmp_path)
    install_fake_psutil(monkeypatch)
    install_clock(monkeypatch)
    run = json.loads(rt.RUN_MANIFEST.read_text(encoding="utf-8"))
    Path(run["protected_prior_receipts"][0]["path"]).write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="PROTECTED_PRIOR_RECEIPT_CHANGED"):
        rt.ContinuationLimit("mutated_prior", gpu=False)
    assert rt.LAST_CONSTRUCTOR_TERMINAL["terminal_status"] == "FAILED_PRESERVED"
    assert not rt.ACTIVE.exists()


def test_previous_receipts_must_bind_auth_manifest_and_are_counted(tmp_path, monkeypatch):
    write_authorization_and_manifest(monkeypatch, tmp_path)
    install_fake_psutil(monkeypatch)
    install_clock(monkeypatch)
    rt.RESOURCES.mkdir(parents=True, exist_ok=True)
    write_json(rt.RESOURCES / "previous.json", {
        "status": "COMPLETE",
        "tag": "previous",
        "gpu": False,
        "wall_seconds": 3.0,
        "authorization": {"path": str(rt.AUTHORIZATION), "sha256": rt.sha(rt.AUTHORIZATION)},
        "run_manifest": {"path": str(rt.RUN_MANIFEST), "sha256": rt.sha(rt.RUN_MANIFEST)},
    })
    limit = rt.ContinuationLimit("previous_bound", gpu=False)
    receipt = limit.close("COMPLETE")
    assert receipt["terminal_status"] == "COMPLETE"
    saved = json.loads((rt.ROOT / receipt["path"]).read_text(encoding="utf-8"))
    assert saved["previous_receipts"] == 1
    assert receipt["continuation_limit"]["budget_counted_seconds"] >= 3.0

    write_authorization_and_manifest(monkeypatch, tmp_path)
    install_fake_psutil(monkeypatch)
    install_clock(monkeypatch, ns=223456789)
    rt.RESOURCES.mkdir(parents=True, exist_ok=True)
    write_json(rt.RESOURCES / "wrong_previous.json", {
        "status": "COMPLETE",
        "gpu": False,
        "wall_seconds": 1.0,
        "authorization": {"path": str(rt.AUTHORIZATION), "sha256": "0" * 64},
        "run_manifest": {"path": str(rt.RUN_MANIFEST), "sha256": rt.sha(rt.RUN_MANIFEST)},
    })
    with pytest.raises(RuntimeError, match="OTHER_APPROVAL_OR_MANIFEST"):
        rt.ContinuationLimit("previous_wrong", gpu=False)


def test_forbidden_cuda_or_f_access_and_budget_close_cannot_complete(tmp_path, monkeypatch):
    write_authorization_and_manifest(monkeypatch, tmp_path)
    install_fake_psutil(monkeypatch)
    install_clock(monkeypatch)
    limit = rt.ContinuationLimit("forbidden_access", gpu=False)
    with pytest.raises(RuntimeError, match="CUDA_OR_F_ACCESS_FORBIDDEN"):
        limit.flag_forbidden_access("F")
    receipt = limit.close("COMPLETE")
    assert receipt["terminal_status"] == "FAILED_PRESERVED"
    assert receipt["terminal_error"]["type"] == "RuntimeError"


def test_storage_budget_close_cannot_complete(tmp_path, monkeypatch):
    limits = dict(rt.MAX_LIMITS)
    limits["additional_storage_bytes"] = 1
    write_authorization_and_manifest(monkeypatch, tmp_path, limits=limits)
    install_fake_psutil(monkeypatch)
    install_clock(monkeypatch)
    storage = {"bytes": 0}
    monkeypatch.setattr(rt, "storage_bytes", lambda roots: storage["bytes"])

    limit = rt.ContinuationLimit("storage_terminal_guard", gpu=False)
    storage["bytes"] = 10
    receipt = limit.close("COMPLETE", {"type": "SyntheticFailure", "message": "caller failed"})
    assert receipt["terminal_status"] == "BUDGET_HELD_PRESERVED"
    assert receipt["requested_status"] == "COMPLETE"
    assert receipt["terminal_error"]["type"] == "BudgetHeld"
