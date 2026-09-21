import argparse
import json
import traceback
from hier_peft.audit import ROOT, RESULTS, sha256, utc_now, write_json
from hier_peft.data import DataFailure
from hier_peft.screen import Screen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["A", "B"], required=True)
    stage = parser.parse_args().stage
    seal = json.loads((RESULTS / "PROTOCOL_SEAL.json").read_text())
    assert seal["preflight_status"] == "PASS"
    for path, expected in seal["code_sha256"].items():
        assert sha256(ROOT / path) == expected, f"Unsealed source change: {path}"
    assert sha256(RESULTS / "SMOKE_AUDIT.json") == seal["smoke_audit_sha256"]
    for group, expected in seal["data_audit_sha256"].items():
        assert sha256(RESULTS / "data" / group / "DATA_AUDIT.json") == expected
    assert sha256(RESULTS / "PREFLIGHT_TESTS.xml") == seal["test_report_sha256"]
    if stage == "B":
        decision = json.loads((RESULTS / "CONTINUATION_DECISION.json").read_text())
        assert decision["decision"] == "PROCEED_TO_TOURISM_CONFIRMATION", "TourismLarge training not authorized by project gate"
    out = RESULTS if stage == "A" else RESULTS / "tourism_confirmation"
    ledger = out / "UPDATE_LEDGER.jsonl"
    if ledger.exists():
        assert not any(json.loads(line)["kind"] == "main" for line in ledger.read_text().splitlines()), "Existing main updates: automatic rerun forbidden"
    budget = out / "TRAINING_BUDGET.json"
    if budget.exists():
        assert json.loads(budget.read_text())["fits_started"] == 0, "Existing fit attempts: automatic rerun forbidden"
    try:
        Screen(stage).run()
    except Exception as exc:
        failure_type = "data failure" if isinstance(exc, DataFailure) else "resource block" if "out of memory" in str(exc).lower() else "implementation failure"
        write_json(out / "FAILURE.json", {"stage": stage, "timestamp": utc_now(), "failure_type": failure_type,
                   "error": repr(exc), "traceback": traceback.format_exc(), "scientific_result": "NOT_ASSESSED"})
        write_json(out / "PROGRESS.json", {"status": "FAILED", "failure_type": failure_type, "error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
