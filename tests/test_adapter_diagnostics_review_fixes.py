"""Failure-path regressions for the campaign audit entry points."""
import json

import pytest

from research.adapter_diagnostics_20261017_v2 import run


@pytest.mark.parametrize(
    "question,audit_result,expected_status",
    [
        (
            "A01",
            {"status": "PARTIAL_OR_FAILED", "synthetic_G1_G4": {"status": "PASS"}},
            "FAILED",
        ),
        ("A01", {"synthetic_G1_G4": {"status": "PASS"}}, "FAILED"),
        (
            "A01",
            {"status": "PASS_WITH_M5_BLOCK", "synthetic_G1_G4": {"status": "PASS"}},
            "COMPLETE",
        ),
        (
            "A04",
            {"status": "PARTIAL_OR_FAILED", "synthetic_G1_G4": {"status": "PASS"}},
            "COMPLETE",
        ),
        (
            "A04",
            {"status": "PASS_WITH_M5_BLOCK", "synthetic_G1_G4": {"status": "FAILED"}},
            "FAILED",
        ),
        ("A04", {"status": "PASS_WITH_M5_BLOCK"}, "FAILED"),
    ],
    ids=[
        "dataset-failure-with-passing-solver",
        "missing-dataset-audit-status",
        "registered-m5-access-block-allowed",
        "solver-success-independent-of-dataset-failure",
        "solver-failure-with-passing-datasets",
        "missing-solver-status",
    ],
)
def test_audit_job_status_reflects_its_own_evidence(
    tmp_path, question, audit_result, expected_status
):
    artifact = tmp_path / "DATA_AUDIT.json"
    artifact.write_text(json.dumps(audit_result), encoding="utf-8")

    result = run.handler({"question_id": question}, {"report": tmp_path})

    assert result["status"] == expected_status
    assert result["artifact_paths"] == [str(artifact)]
    assert result["result"] == audit_result
    if expected_status == "FAILED":
        assert result["reason"]


@pytest.mark.parametrize(
    "audit_result,expected_success",
    [
        ({"status": "PARTIAL_OR_FAILED"}, False),
        ({}, False),
        ({"status": "PASS_WITH_M5_BLOCK"}, True),
    ],
    ids=["partial-audit-exits-nonzero", "missing-status-exits-nonzero", "m5-block-exits-zero"],
)
def test_standalone_audit_exit_status(monkeypatch, audit_result, expected_success):
    sentinel_context = object()
    called = []
    monkeypatch.setattr(run, "context", lambda args: sentinel_context)

    def audit(ctx):
        assert ctx is sentinel_context
        called.append(ctx)
        return audit_result

    monkeypatch.setattr(run, "audit", audit)
    # The module's __main__ calls main() directly: a returned integer alone
    # cannot set the process exit code.
    exit_status = 0
    try:
        run.main(["--mode", "audit"])
    except SystemExit as error:
        exit_status = error.code

    assert called == [sentinel_context]
    assert (exit_status in (None, 0)) is expected_success
