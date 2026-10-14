from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage_separation_runtime_v1 as rt  # noqa: E402


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
    plan = tmp_path / "PLAN_STAGE_SEPARATION_KO_v1.md"
    method = tmp_path / "METHOD_LOCK_STAGE_SEPARATION_v1.md"
    scanner = tmp_path / "fast_storage_scan_v1.py"
    plan.write_text("synthetic stage plan", encoding="utf-8")
    method.write_text("synthetic method lock", encoding="utf-8")
    scanner.write_text("def storage_bytes(roots): return 0\n", encoding="utf-8")
    monkeypatch.setattr(rt, "HERE", tmp_path)
    monkeypatch.setattr(rt, "ROOT", tmp_path)
    monkeypatch.setattr(rt, "APPROVED_PLAN", plan)
    monkeypatch.setattr(rt, "METHOD_LOCK", method)
    monkeypatch.setattr(rt, "AUTHORIZATION", tmp_path / "stage_separation_authorization_v1.json")
    monkeypatch.setattr(rt, "RUN_MANIFEST", tmp_path / "stage_separation_run_manifest_v1.json")
    monkeypatch.setattr(rt, "RESOURCES", tmp_path / "stage_separation_resources_v1")
    monkeypatch.setattr(rt, "ACTIVE", tmp_path / "stage_separation_resources_v1" / "ACTIVE.json")
    monkeypatch.setattr(rt, "LAST_CONSTRUCTOR_TERMINAL", None)
    monkeypatch.setattr(rt, "storage_bytes", lambda roots: 0)


def install_fake_psutil(monkeypatch, *, created_unix: float = 1001.5, rss: int = 4096):
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(Process=lambda: FakeProcess(created_unix=created_unix, rss=rss)),
    )


def write_authorization_and_manifest(
    monkeypatch,
    tmp_path,
    *,
    limits=None,
    approved_unix: float = 1000.0,
    started_unix: float = 1001.0,
    protected_prior_receipts=None,
    storage_roots=None,
):
    rehome_runtime(monkeypatch, tmp_path)
    limits = dict(rt.MAX_LIMITS if limits is None else limits)
    auth = {
        "schema": "smallk_stage_separation_authorization_v1",
        "status": "APPROVED",
        "origin": "human_user_response",
        "response_identifier": "synthetic-human-call-id:0",
        "user_response": "synthetic approval fixture",
        "phase": "stage_separation_diagnostics_v1",
        "oracle_scope": "diagnostic_only_not_deployment",
        "approved_unix": approved_unix,
        "approved_plan_sha256": rt.sha(rt.APPROVED_PLAN),
        "method_lock_sha256": rt.sha(rt.METHOD_LOCK),
        "limits": limits,
    }
    write_json(rt.AUTHORIZATION, auth)
    run = {
        "schema": "smallk_stage_separation_run_manifest_v1",
        "authorization_sha256": rt.sha(rt.AUTHORIZATION),
        "runtime_source_sha256": rt.sha(rt.CORE_RUNTIME),
        "started_unix": started_unix,
        "expires_unix": started_unix + limits["campaign_wall_seconds"],
        "baseline_storage_bytes": 0,
        "storage_roots": [str(tmp_path / "stage_outputs")] if storage_roots is None else storage_roots,
        "protected_prior_receipts": [] if protected_prior_receipts is None else protected_prior_receipts,
    }
    write_json(rt.RUN_MANIFEST, run)
    return auth, run


def test_missing_authorization_is_constructor_terminal_without_resource_receipt(tmp_path, monkeypatch):
    rehome_runtime(monkeypatch, tmp_path)

    with pytest.raises(rt.AuthorizationRequired, match="EXPLICIT_HUMAN_APPROVAL"):
        rt.ContinuationLimit("synthetic_missing_auth", gpu=False)

    assert rt.LAST_CONSTRUCTOR_TERMINAL["terminal_status"] == "FAILED_PRESERVED"
    assert "pre-authorization failure" in rt.LAST_CONSTRUCTOR_TERMINAL["scope"]


def test_rejects_old_schema_and_missing_human_response(tmp_path, monkeypatch):
    auth, _ = write_authorization_and_manifest(monkeypatch, tmp_path)
    auth["schema"] = "smallk_continuation_authorization_v1"
    write_json(rt.AUTHORIZATION, auth)

    with pytest.raises(rt.AuthorizationRequired, match="EXPLICIT_HUMAN_APPROVAL_RECORD_INVALID"):
        rt.authorized_records()

    auth["schema"] = "smallk_stage_separation_authorization_v1"
    auth["origin"] = "agent_summary"
    auth["response_identifier"] = ""
    write_json(rt.AUTHORIZATION, auth)

    with pytest.raises(rt.AuthorizationRequired, match="EXPLICIT_HUMAN_APPROVAL_RECORD_INVALID"):
        rt.authorized_records()


@pytest.mark.parametrize(
    "limit_name,too_large",
    [
        ("gpu_process_wall_seconds", 601),
        ("cpu_process_wall_seconds", 901),
        ("campaign_wall_seconds", 2401),
        ("additional_storage_bytes", 1024**3 + 1),
    ],
)
def test_limits_cannot_exceed_human_approved_caps(tmp_path, monkeypatch, limit_name, too_large):
    limits = dict(rt.MAX_LIMITS)
    limits[limit_name] = too_large
    write_authorization_and_manifest(monkeypatch, tmp_path, limits=limits)

    with pytest.raises(rt.AuthorizationRequired, match="LIMITS_EXCEED_APPROVED"):
        rt.authorized_records()


def test_manifest_binds_authorization_runtime_and_exact_expiry(tmp_path, monkeypatch):
    write_authorization_and_manifest(monkeypatch, tmp_path)
    run = json.loads(rt.RUN_MANIFEST.read_text(encoding="utf-8"))
    run["authorization_sha256"] = "0" * 64
    write_json(rt.RUN_MANIFEST, run)

    with pytest.raises(rt.AuthorizationRequired, match="FIXED_AUTHORIZATION_BOUND"):
        rt.authorized_records()

    run["authorization_sha256"] = rt.sha(rt.AUTHORIZATION)
    run["runtime_source_sha256"] = "1" * 64
    write_json(rt.RUN_MANIFEST, run)

    with pytest.raises(rt.AuthorizationRequired, match="FIXED_AUTHORIZATION_BOUND"):
        rt.authorized_records()

    run["runtime_source_sha256"] = rt.sha(rt.CORE_RUNTIME)
    run["expires_unix"] += 1
    write_json(rt.RUN_MANIFEST, run)

    with pytest.raises(rt.AuthorizationRequired, match="FIXED_AUTHORIZATION_BOUND"):
        rt.authorized_records()


def test_campaign_expiry_constructor_closes_budget_terminal_and_removes_active(tmp_path, monkeypatch):
    limits = dict(rt.MAX_LIMITS)
    limits["campaign_wall_seconds"] = 10
    write_authorization_and_manifest(monkeypatch, tmp_path, limits=limits, started_unix=1001.0)
    install_fake_psutil(monkeypatch, created_unix=1002.0)
    monkeypatch.setattr(rt.time, "time", lambda: 1012.5)
    monkeypatch.setattr(rt.time, "perf_counter", lambda: 0.0)
    monkeypatch.setattr(rt.time, "time_ns", lambda: 123456789)

    with pytest.raises(rt.BudgetHeld, match="STAGE_SEPARATION_CAMPAIGN_WALL_BUDGET"):
        rt.ContinuationLimit("synthetic_expired", gpu=False)

    assert rt.LAST_CONSTRUCTOR_TERMINAL["terminal_status"] == "BUDGET_HELD_PRESERVED"
    assert rt.LAST_CONSTRUCTOR_TERMINAL["terminal_error"]["type"] == "BudgetHeld"
    assert not rt.ACTIVE.exists()


def test_prior_phase_active_lease_blocks_new_runtime_before_private_active_write(tmp_path, monkeypatch):
    write_authorization_and_manifest(monkeypatch, tmp_path)
    install_fake_psutil(monkeypatch)
    write_json(tmp_path / "continuation_resources_v1" / "ACTIVE.json", {"pid": 999, "tag": "prior"})
    monkeypatch.setattr(rt.time, "time", lambda: 1002.0)
    monkeypatch.setattr(rt.time, "perf_counter", lambda: 0.0)
    monkeypatch.setattr(rt.time, "time_ns", lambda: 223456789)

    with pytest.raises(RuntimeError, match="PRIOR_PHASE_ACTIVE_LEASE"):
        rt.ContinuationLimit("synthetic_prior_active", gpu=False)

    assert not rt.ACTIVE.exists()


def test_existing_previous_terminal_receipts_are_counted_and_bound_into_close(tmp_path, monkeypatch):
    write_authorization_and_manifest(monkeypatch, tmp_path)
    install_fake_psutil(monkeypatch, created_unix=1001.5)
    rt.RESOURCES.mkdir(parents=True, exist_ok=True)
    write_json(
        rt.RESOURCES / "previous.json",
        {"tag": "previous_cpu", "gpu": False, "status": "COMPLETE", "wall_seconds": 5.0},
    )
    now = {"value": 1002.0}
    monkeypatch.setattr(rt.time, "time", lambda: now["value"])
    monkeypatch.setattr(rt.time, "perf_counter", lambda: now["value"] - 1002.0)
    monkeypatch.setattr(rt.time, "time_ns", lambda: 323456789)

    limit = rt.ContinuationLimit("synthetic_previous_binding", gpu=False)
    receipt = limit.close("COMPLETE")

    assert receipt["terminal_status"] == "COMPLETE"
    assert receipt["continuation_limit"]["budget_counted_seconds"] >= 5.0
    saved = json.loads((rt.ROOT / receipt["path"]).read_text(encoding="utf-8"))
    assert saved["previous_receipts"] == 1


def test_terminal_error_and_storage_growth_cannot_be_reported_complete(tmp_path, monkeypatch):
    limits = dict(rt.MAX_LIMITS)
    limits["additional_storage_bytes"] = 1
    write_authorization_and_manifest(monkeypatch, tmp_path, limits=limits)
    install_fake_psutil(monkeypatch)
    monkeypatch.setattr(rt.time, "time", lambda: 1002.0)
    monkeypatch.setattr(rt.time, "perf_counter", lambda: 0.0)
    monkeypatch.setattr(rt.time, "time_ns", lambda: 423456789)
    storage = {"bytes": 0}
    monkeypatch.setattr(rt, "storage_bytes", lambda roots: storage["bytes"])

    limit = rt.ContinuationLimit("synthetic_storage_terminal", gpu=False)
    storage["bytes"] = 10
    receipt = limit.close("COMPLETE", {"type": "SyntheticFailure", "message": "caller failed"})

    assert receipt["terminal_status"] == "BUDGET_HELD_PRESERVED"
    assert receipt["requested_status"] == "COMPLETE"
    assert receipt["terminal_error"]["type"] == "BudgetHeld"
