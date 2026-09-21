import json
import subprocess
import sys
from hier_peft.audit import ROOT, RESULTS, sha256, utc_now, write_json


def main():
    env = json.loads((RESULTS / "ENVIRONMENT.json").read_text())
    assert env["cuda_available"] and env["pip_check"]["returncode"] == 0
    for group in ["Labour", "TourismLarge"]:
        assert json.loads((RESULTS / "data" / group / "DATA_AUDIT.json").read_text())["status"] == "PASS"
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "--junitxml=" + str(RESULTS / "PREFLIGHT_TESTS.xml")], cwd=ROOT)
    if result.returncode:
        raise SystemExit(result.returncode)
    smoke = json.loads((RESULTS / "SMOKE_AUDIT.json").read_text())
    assert smoke["status"] == "PASS"
    files = [*ROOT.glob("src/hier_peft/*.py"), *ROOT.glob("configs/*.yaml"),
             ROOT / "requirements-lock.txt", ROOT / "experiments/hier_peft_screen_v1/PROTOCOL.md",
             *ROOT.glob("experiments/hier_peft_screen_v1/*.py")]
    write_json(RESULTS / "PROTOCOL_SEAL.json", {"sealed_at": utc_now(), "preflight_status": "PASS",
               "code_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in files},
               "smoke_audit_sha256": sha256(RESULTS / "SMOKE_AUDIT.json"),
               "data_audit_sha256": {g: sha256(RESULTS / "data" / g / "DATA_AUDIT.json") for g in ["Labour", "TourismLarge"]},
               "test_report_sha256": sha256(RESULTS / "PREFLIGHT_TESTS.xml")})
    print("Preflight PASS; protocol sealed; Stage A may run", flush=True)


if __name__ == "__main__":
    main()
