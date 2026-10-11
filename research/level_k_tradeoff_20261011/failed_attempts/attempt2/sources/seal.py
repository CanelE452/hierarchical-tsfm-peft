"""Preserve preceding studies and seal each new stage before science."""

from pathlib import Path
import argparse
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = ROOT / "research/level_budget_recheck_20261011"
HISTORY = ROOT / "_docs/history/2026-10-11.md"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def resolve(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def write(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def capture():
    if (HERE / "parent_preservation.before.json").exists():
        raise FileExistsError("Existing parent preservation receipt")
    prior = read(PARENT / "final_verification.json")
    if prior["status"] != "PASS":
        raise RuntimeError("Prior final audit did not pass")
    expected = {str(resolve(r["path"])): r["sha256"] for r in prior["artifacts"]}
    expected.update({str(resolve(r["path"])): r["sha256"] for r in
                     read(PARENT / "parent_preservation.before.json")["artifacts"]})
    paths = {Path(p) for p in expected}
    paths.update(p for folder in (PARENT, ROOT / ".cache" / PARENT.name) for p in folder.rglob("*"))
    paths.add(PARENT / "final_verification.json")
    rows = []
    for p in sorted(paths):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        actual = sha(p)
        if str(p) in expected and actual != expected[str(p)]:
            raise RuntimeError(f"Prior evidence changed: {p}")
        if p == HISTORY:
            continue
        rows.append({"path": p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p),
                     "sha256": actual, "bytes": p.stat().st_size})
    history = HISTORY.read_bytes()
    write("parent_preservation.before.json", {"status": "PASS", "artifacts": rows,
          "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
          "history_append_only": {"path": HISTORY.relative_to(ROOT).as_posix(), "bytes": len(history),
                                  "prefix_sha256": hashlib.sha256(history).hexdigest()}})
    print(f"Preserved {len(rows)} preceding artifacts; history protected by append-only prefix")


def check():
    before = read(HERE / "parent_preservation.before.json")
    checks = [{**r, "actual_sha256": sha(resolve(r["path"])),
               "passed": sha(resolve(r["path"])) == r["sha256"]} for r in before["artifacts"]]
    history = before["history_append_only"]
    prefix = resolve(history["path"]).read_bytes()[:history["bytes"]]
    prefix_passed = hashlib.sha256(prefix).hexdigest() == history["prefix_sha256"]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    passed = all(r["passed"] for r in checks) and prefix_passed and head == before["head"]
    write("parent_preservation.after.json", {"status": "PASS" if passed else "FAIL", "checks": checks,
          "history_prefix_unchanged": prefix_passed, "head_unchanged": head == before["head"]})
    print(f"Preservation {'PASS' if passed else 'FAIL'}: {len(checks)} artifacts")
    return 0 if passed else 1


def source(stage):
    output = HERE / f"source_seal.{stage}.json"
    if output.exists():
        raise FileExistsError(output)
    names = ["PLAN.md", "protocol.json", "sweep.py", "benchmark.py", "test_sweep.py", "seal.py"]
    if stage == "method":
        names.extend(["method_protocol.json", "method_run.py", "ts2vec_adapter.py", "test_ts2vec.py"])
    paths = [HERE / n for n in names]
    if stage == "method":
        paths.extend(p for p in (HERE / "vendor/ts2vec").rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    write(output.name, {"phase": "before_stage_science", "stage": stage,
          "artifacts": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in sorted(paths)]})
    print(f"Sealed {len(paths)} {stage} stage sources")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["capture", "check", "source-k", "source-method"])
    args = parser.parse_args()
    if args.action == "capture":
        capture()
    elif args.action == "check":
        raise SystemExit(check())
    else:
        source("k" if args.action == "source-k" else "method")
