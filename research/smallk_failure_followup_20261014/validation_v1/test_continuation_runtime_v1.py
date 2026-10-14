from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.smallk_failure_followup_20261014 import continuation_runtime_v1 as rt


class FakeProcess:
    def __init__(self, created_unix=1000.0, rss=1024):
        self._created_unix = created_unix
        self._rss = rss

    def create_time(self):
        return self._created_unix

    def memory_info(self):
        return SimpleNamespace(rss=self._rss)


def install_fake_psutil(monkeypatch, *, created_unix=1000.0, rss=1024):
    monkeypatch.setitem(
        __import__("sys").modules,
        "psutil",
        SimpleNamespace(Process=lambda: FakeProcess(created_unix=created_unix, rss=rss)),
    )


def rehome_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(rt, "ROOT", tmp_path)
    auth = tmp_path / "continuation_authorization_v1.json"
    run = tmp_path / "continuation_run_manifest_v1.json"
    resources = tmp_path / "continuation_resources_v1"
    monkeypatch.setattr(rt, "AUTHORIZATION", auth)
    monkeypatch.setattr(rt, "RUN_MANIFEST", run)
    monkeypatch.setattr(rt, "RESOURCES", resources)
    monkeypatch.setattr(rt, "ACTIVE", resources / "ACTIVE.json")
    monkeypatch.setattr(rt, "LAST_CONSTRUCTOR_TERMINAL", None)
    return auth, run, resources


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_authorized_manifest(monkeypatch, tmp_path, *, limits=None, started=1010.0, approved=1005.0, protected_prior_receipts=None, storage_roots=None):
    auth_path, run_path, resources = rehome_runtime(monkeypatch, tmp_path)
    limits = dict(rt.MAX_LIMITS if limits is None else limits)
    auth = {
        "schema": "smallk_continuation_authorization_v1",
        "status": "APPROVED",
        "origin": "human_user_response",
        "response_identifier": "synthetic-human-reply-id",
        "user_response": "approved in synthetic test fixture",
        "approved_unix": approved,
        "limits": limits,
    }
    write_json(auth_path, auth)
    run = {
        "schema": "smallk_continuation_run_manifest_v1",
        "authorization_sha256": rt.sha(auth_path),
        "runtime_source_sha256": rt.sha(Path(rt.__file__)),
        "started_unix": started,
        "expires_unix": started + limits["campaign_wall_seconds"],
        "baseline_storage_bytes": 0,
        "storage_roots": [str(tmp_path / "storage")],
        "protected_prior_receipts": [] if protected_prior_receipts is None else protected_prior_receipts,
    }
    if storage_roots is not None:
        run["storage_roots"] = storage_roots
    write_json(run_path, run)
    return auth_path, run_path, resources


def test_authorization_missing_is_required(tmp_path, monkeypatch):
    rehome_runtime(monkeypatch, tmp_path)

    with pytest.raises(rt.AuthorizationRequired, match="EXPLICIT_HUMAN_APPROVAL"):
        rt.authorized_records()


def test_manifest_must_bind_authorization_and_runtime_sha(tmp_path, monkeypatch):
    auth_path, run_path, _ = write_authorized_manifest(monkeypatch, tmp_path)
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["authorization_sha256"] = "0" * 64
    write_json(run_path, run)

    with pytest.raises(rt.AuthorizationRequired, match="FIXED_AUTHORIZATION_BOUND_RUN_MANIFEST_INVALID"):
        rt.authorized_records()

    run["authorization_sha256"] = rt.sha(auth_path)
    run["runtime_source_sha256"] = "1" * 64
    write_json(run_path, run)

    with pytest.raises(rt.AuthorizationRequired, match="FIXED_AUTHORIZATION_BOUND_RUN_MANIFEST_INVALID"):
        rt.authorized_records()


def test_campaign_wall_exceeded_becomes_constructor_terminal_budget(tmp_path, monkeypatch):
    limits = dict(rt.MAX_LIMITS)
    limits["campaign_wall_seconds"] = 10
    write_authorized_manifest(monkeypatch, tmp_path, limits=limits, started=1010.0, approved=1005.0)
    install_fake_psutil(monkeypatch, created_unix=1011.0)
    monkeypatch.setattr(rt.time, "time", lambda: 1021.5)
    monkeypatch.setattr(rt.time, "perf_counter", lambda: 0.0)
    monkeypatch.setattr(rt.time, "time_ns", lambda: 123456789)

    with pytest.raises(rt.BudgetHeld, match="CONTINUATION_CAMPAIGN_WALL_BUDGET"):
        rt.ContinuationLimit("synthetic_budget", gpu=False)

    assert rt.LAST_CONSTRUCTOR_TERMINAL["terminal_status"] == "BUDGET_HELD_PRESERVED"
    assert not rt.ACTIVE.exists()


def test_terminal_error_cannot_be_reported_as_complete(tmp_path, monkeypatch):
    write_authorized_manifest(monkeypatch, tmp_path)
    install_fake_psutil(monkeypatch, created_unix=1011.0)
    now = {"value": 1012.0}
    monkeypatch.setattr(rt.time, "time", lambda: now["value"])
    monkeypatch.setattr(rt.time, "perf_counter", lambda: now["value"] - 1012.0)
    monkeypatch.setattr(rt.time, "time_ns", lambda: 223456789)

    limit = rt.ContinuationLimit("synthetic_terminal_error", gpu=False)
    receipt = limit.close("COMPLETE", {"type": "SyntheticFailure", "message": "must not complete"})

    assert receipt["terminal_status"] != "COMPLETE"
    assert receipt["terminal_error"]["type"] == "SyntheticFailure"


def test_fresh_receipt_write_and_active_lease_cleanup(tmp_path, monkeypatch):
    write_authorized_manifest(monkeypatch, tmp_path)
    install_fake_psutil(monkeypatch, created_unix=1011.0)
    monkeypatch.setattr(rt.time, "time", lambda: 1012.0)
    monkeypatch.setattr(rt.time, "perf_counter", lambda: 0.25)
    monkeypatch.setattr(rt.time, "time_ns", lambda: 323456789)

    limit = rt.ContinuationLimit("synthetic_receipt", gpu=False)
    assert rt.ACTIVE.exists()
    receipt = limit.close("FAILED_PRESERVED", {"type": "SyntheticFailure"})

    assert receipt["terminal_status"] == "FAILED_PRESERVED"
    assert Path(rt.ROOT / receipt["path"]).is_file()
    assert not rt.ACTIVE.exists()


def test_active_lease_blocks_second_constructor(tmp_path, monkeypatch):
    _, _, resources = write_authorized_manifest(monkeypatch, tmp_path)
    resources.mkdir(exist_ok=True)
    write_json(resources / "ACTIVE.json", {"pid": 999, "tag": "other"})
    install_fake_psutil(monkeypatch, created_unix=1011.0)
    monkeypatch.setattr(rt.time, "time", lambda: 1012.0)
    monkeypatch.setattr(rt.time, "perf_counter", lambda: 0.0)

    with pytest.raises(RuntimeError, match="EXISTING_ACTIVE_LEASE"):
        rt.ContinuationLimit("synthetic_lease", gpu=False)


def test_contract_exposes_checkfast_and_checkfull_api(tmp_path, monkeypatch):
    write_authorized_manifest(monkeypatch, tmp_path)
    install_fake_psutil(monkeypatch, created_unix=1011.0)
    monkeypatch.setattr(rt.time, "time", lambda: 1012.0)
    monkeypatch.setattr(rt.time, "perf_counter", lambda: 0.0)
    monkeypatch.setattr(rt.time, "time_ns", lambda: 423456789)

    limit = rt.ContinuationLimit("synthetic_api", gpu=False)
    try:
        assert hasattr(limit.shared, "checkfast")
        assert hasattr(limit, "checkfull")
        assert limit.rt.BudgetHeld is rt.BudgetHeld
    finally:
        if not limit.closed:
            limit.close("FAILED_PRESERVED", {"type": "SyntheticTeardown"})
