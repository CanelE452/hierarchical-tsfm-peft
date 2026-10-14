from __future__ import annotations

from pathlib import Path
import json
import sys
import time
from types import SimpleNamespace

import pytest

import main_residual_runtime_v1 as runtime


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _configure_runtime(tmp_path, monkeypatch, *, auth_updates=None, run_updates=None, prior_receipts=()):
    plan = tmp_path / "PLAN_MAIN_FORECAST_RESIDUAL_KO_v1.md"
    method = tmp_path / "METHOD_LOCK_MAIN_RESIDUAL_v1.md"
    auth = tmp_path / "main_residual_authorization_v1.json"
    run = tmp_path / "main_residual_run_manifest_v1.json"
    resources = tmp_path / "main_residual_resources_v1"
    storage = tmp_path / "storage"
    protected_dir = tmp_path / "protected"
    resources.mkdir()
    storage.mkdir()
    protected_dir.mkdir()
    plan.write_text("plan", encoding="utf-8")
    method.write_text("method", encoding="utf-8")
    protected = protected_dir / "old_receipt.json"
    _write_json(protected, {"status": "COMPLETE", "wall_seconds": 1.0, "gpu": False})
    approved = time.time() - 20.0
    started = approved + 1.0
    auth_value = {
        "schema": "smallk_main_residual_authorization_v1",
        "status": "APPROVED",
        "origin": "human_user_response",
        "response_identifier": runtime.EXPECTED_RESPONSE_IDENTIFIER,
        "user_response": runtime.EXPECTED_USER_RESPONSE,
        "phase": runtime.PHASE,
        "approved_unix": approved,
        "approved_plan_sha256": runtime.sha(plan),
        "method_lock_sha256": runtime.sha(method),
        "runtime_source_sha256": runtime.sha(runtime.CORE_RUNTIME),
        "gpu_allowed": True,
        "cuda_allowed_for_gpu_cost": True,
        "cpu_cuda_allowed": False,
        "F_access_allowed": True,
        "training_allowed": False,
        "limits": dict(runtime.MAX_LIMITS),
    }
    if auth_updates:
        auth_value.update(auth_updates)
    _write_json(auth, auth_value)
    protected_records = [{"path": str(protected), "sha256": runtime.sha(protected)}]
    run_value = {
        "schema": "smallk_main_residual_run_manifest_v1",
        "authorization_sha256": runtime.sha(auth),
        "runtime_source_sha256": runtime.sha(runtime.CORE_RUNTIME),
        "phase": runtime.PHASE,
        "started_unix": started,
        "expires_unix": started + runtime.MAX_LIMITS["campaign_wall_seconds"],
        "baseline_storage_bytes": 0,
        "storage_roots": [str(storage)],
        "protected_prior_receipts": protected_records,
        "gpu_allowed": True,
        "cuda_allowed_for_gpu_cost": True,
        "cpu_cuda_allowed": False,
        "F_access_allowed": True,
        "training_allowed": False,
    }
    if run_updates:
        run_value.update(run_updates)
    _write_json(run, run_value)
    monkeypatch.setattr(runtime, "AUTHORIZATION", auth)
    monkeypatch.setattr(runtime, "RUN_MANIFEST", run)
    monkeypatch.setattr(runtime, "APPROVED_PLAN", plan)
    monkeypatch.setattr(runtime, "METHOD_LOCK", method)
    monkeypatch.setattr(runtime, "RESOURCES", resources)
    monkeypatch.setattr(runtime, "ACTIVE", resources / "ACTIVE.json")
    monkeypatch.setattr(runtime, "PRIOR_ACTIVE_LEASES", tuple())
    monkeypatch.setattr(runtime, "storage_bytes", lambda roots: 0)
    for name, receipt in prior_receipts:
        receipt = dict(receipt)
        receipt.setdefault("status", "COMPLETE")
        receipt.setdefault("authorization", {"sha256": runtime.sha(auth)})
        receipt.setdefault("run_manifest", {"sha256": runtime.sha(run)})
        _write_json(resources / name, receipt)
    return SimpleNamespace(auth=auth, run=run, plan=plan, method=method, protected=protected, resources=resources, storage=storage)


def test_constructor_requires_exact_human_reply(tmp_path, monkeypatch):
    _configure_runtime(tmp_path, monkeypatch, auth_updates={"user_response": "추가승인"})
    with pytest.raises(runtime.AuthorizationRequired):
        runtime.ContinuationLimit("main_runtime_bad_auth", gpu=False)
    assert runtime.LAST_CONSTRUCTOR_TERMINAL["terminal_status"] == "FAILED_PRESERVED"


def test_cpu_limit_forbids_f_access_and_cuda_initialization(tmp_path, monkeypatch):
    _configure_runtime(tmp_path, monkeypatch)
    limit = runtime.ContinuationLimit("main_cpu_fixture", gpu=False)
    try:
        with pytest.raises(runtime.ForbiddenAccess):
            limit.flag_forbidden_access("F")
    finally:
        receipt = limit.close("FAILED_PRESERVED", {"type": "expected", "message": "cpu F forbidden"})
    assert receipt["terminal_status"] == "FAILED_PRESERVED"

    _configure_runtime(tmp_path / "cuda", monkeypatch)
    limit = runtime.ContinuationLimit("main_cpu_cuda_fixture", gpu=False)
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    with pytest.raises(runtime.ForbiddenAccess):
        limit.checkfull()
    receipt = limit.close("FAILED_PRESERVED", {"type": "expected", "message": "cpu cuda forbidden"})
    assert receipt["terminal_status"] == "FAILED_PRESERVED"


def test_gpu_limit_allows_only_frozen_f_forward_scope_and_rejects_training(tmp_path, monkeypatch):
    _configure_runtime(tmp_path, monkeypatch)
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    limit = runtime.ContinuationLimit("main_gpu_cost", gpu=True)
    assert limit.declare_gpu_forward_scope("frozen_F")["training_allowed"] is False
    assert limit.checkfull()["gpu"] is True
    with pytest.raises(runtime.ForbiddenAccess):
        limit.declare_gpu_forward_scope("frozen_EFD")
    with pytest.raises(runtime.ForbiddenAccess):
        limit.flag_training_access("optimizer_step")
    receipt = limit.close("FAILED_PRESERVED", {"type": "expected", "message": "training forbidden"})
    assert receipt["terminal_status"] == "FAILED_PRESERVED"


def test_protected_prior_receipt_sha_is_rechecked(tmp_path, monkeypatch):
    handles = _configure_runtime(tmp_path, monkeypatch)
    limit = runtime.ContinuationLimit("main_protection", gpu=False)
    handles.protected.write_text(json.dumps({"status": "COMPLETE", "changed": True}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="PROTECTED_PRIOR_RECEIPT_CHANGED"):
        limit.checkfull()
    receipt = limit.close("FAILED_PRESERVED", {"type": "expected", "message": "protected changed"})
    assert receipt["terminal_status"] == "FAILED_PRESERVED"


def test_cpu_and_gpu_receipts_use_separate_process_budgets(tmp_path, monkeypatch):
    prior = (
        ("prior_cpu.json", {"status": "COMPLETE", "gpu": False, "wall_seconds": 400.0}),
        ("prior_gpu.json", {"status": "COMPLETE", "gpu": True, "wall_seconds": 10.0}),
    )
    _configure_runtime(tmp_path, monkeypatch, prior_receipts=prior)
    gpu_limit = runtime.ContinuationLimit("main_gpu_budget", gpu=True)
    try:
        value = gpu_limit.checkfull()
        assert value["budget_counted_seconds"] < 30.0
    finally:
        gpu_limit.close("COMPLETE")

    _configure_runtime(tmp_path / "cpu", monkeypatch, prior_receipts=prior)
    cpu_limit = runtime.ContinuationLimit("main_cpu_budget", gpu=False)
    try:
        value = cpu_limit.checkfull()
        assert value["budget_counted_seconds"] >= 400.0
    finally:
        cpu_limit.close("COMPLETE")
