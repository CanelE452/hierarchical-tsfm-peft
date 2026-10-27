"""Followup-only paths, frozen-input guards, and preregistration admission."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

NAME = "comparative_ablation_followup_20261027_v1"
OLD_NAME = "comparative_ablation_20261025_v1"
REPO = Path(__file__).resolve().parents[2]
REPORT = REPO / "reports" / NAME
CACHE = REPO / ".cache" / NAME
OLD_REPORT = REPO / "reports" / OLD_NAME
OLD_CACHE = REPO / ".cache" / OLD_NAME
BASE = "06f142dbba1ab0370fb69366eb6cd28b721d5505"
PYTHON = Path("E:/CODING/proj/hierarchical-tsfm-peft/.venv/Scripts/python.exe")
SOURCE_DOCUMENT = Path("C:/Users/User/Downloads/comparative_ablation_followup_executor_instructions.md")
LIMIT_SECONDS = 13 * 3600
LAST_START = "2026-10-28T00:00:00+09:00"
DATASETS = ("jena", "ETTh1", "ETTh2_safe17400", "appliances")
SEEDS = tuple(range(92601, 92611))
KST = timezone(timedelta(hours=9))

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def now_utc():
    return datetime.now(timezone.utc).isoformat()

def now_kst():
    return datetime.now(KST).isoformat()

def _new_path(path):
    path = Path(path).resolve()
    if not any(path.is_relative_to(root.resolve()) for root in (REPORT, CACHE)):
        raise ValueError(f"Followup writer outside new namespace: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def write_new(path, value):
    path = _new_path(path)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")

def atomic(path, value):
    path = _new_path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in (5, 32, 33) or attempt == 5:
                raise
            time.sleep(0.1 * 2 ** attempt)

def selected_jobs():
    plan = read(OLD_REPORT / "RUN_PLAN.json")
    registered = plan["jobs"] + plan["extension_jobs"]
    chosen = [row for row in registered
              if (row["method_id"] == "A4" and not row["extended"])
              or (row["method_id"] == "A1" and row["dataset"] == "jena" and row["extended"])]
    order = {name: i for i, name in enumerate(DATASETS)}
    chosen.sort(key=lambda x: (x["method_id"] != "A4", x["seed"], order[x["dataset"]]))
    if len(chosen) != 50 or len({x["job_id"] for x in chosen}) != 50:
        raise ValueError("Followup must have exactly 50 unique registered jobs")
    if any(set(x) != {"job_id", "method_id", "dataset", "K", "seed", "extended"} for x in chosen):
        raise ValueError("Registered job fields changed")
    if [x["seed"] for x in chosen if x["method_id"] == "A1"] != list(SEEDS):
        raise ValueError("A1 extension seeds changed")
    for dataset in DATASETS:
        if [x["seed"] for x in chosen if x["method_id"] == "A4" and x["dataset"] == dataset] != list(SEEDS):
            raise ValueError("A4 seeds changed")
    return chosen

def all_results():
    results = [read(path) for path in sorted((OLD_REPORT / "jobs").glob("*/RESULT.json"))]
    if any(row["status"] != "COMPLETE" for row in results):
        raise ValueError("Non-complete RESULT in analysis inputs")
    if len({row["job"]["job_id"] for row in results}) != len(results):
        raise ValueError("Duplicate analysis inputs")
    return results

def guard_preserved():
    baseline = read(REPORT / "PRESERVATION_BASELINE.json")
    changed = []
    for row in baseline["files"]:
        path = Path(row["path"])
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            changed.append(row["path"])
    if changed:
        raise RuntimeError("FG1: pre-existing files changed: " + ", ".join(changed))
    return {"status": "PASS", "checked": len(baseline["files"]), "changed": [],
            "scope": baseline["scope"]}

def install_write_guard():
    baseline = read(REPORT / "PRESERVATION_BASELINE.json")
    protected = {os.path.normcase(os.path.abspath(row["path"])) for row in baseline["files"]}
    denied = []
    def normalize(path):
        if not isinstance(path, (str, bytes, os.PathLike)):
            return None
        return os.path.normcase(os.path.abspath(os.fsdecode(path)))
    def reject(event, paths):
        for path in paths:
            normalized = normalize(path)
            if normalized in protected:
                denied.append({"event": event, "path": normalized})
                raise PermissionError(f"Followup forbids changing pre-existing file: {normalized}")
    def audit(event, args):
        if event == "open":
            path, mode, flags = args
            writing = ((isinstance(mode, str) and any(x in mode for x in "wax+"))
                       or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
            if writing:
                reject(event, (path,))
        elif event in ("os.remove", "os.unlink", "os.truncate", "os.chmod", "os.utime"):
            reject(event, args[:1])
        elif event in ("os.rename", "os.replace"):
            reject(event, args[:2])
    sys.addaudithook(audit)
    return {"denied": denied, "protected_files": len(protected)}

def _git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO)

def validate_admission():
    admission = read(REPORT / "ADMISSION.json")
    if admission["status"] != "PASS":
        raise RuntimeError("Followup admission has not passed")
    gates = read(REPORT / "GATES.json")
    if any(gates[f"FG{i}"]["status"] != "PASS" for i in range(1, 9)):
        raise RuntimeError("Followup gate incomplete")
    receipt = read(REPORT / "PREREGISTRATION_COMMIT.json")
    commit = receipt["commit"]
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=REPO, check=True, capture_output=True)
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE, commit], cwd=REPO, check=True, capture_output=True)
    if receipt["commit_unix"] > time.time():
        raise RuntimeError("Preregistration commit is after current time")
    for row in receipt["files"]:
        content = _git("show", f"{commit}:{row['path']}")
        if hashlib.sha256(content).hexdigest() != row["sha256"] or sha(REPO / row["path"]) != row["sha256"]:
            raise RuntimeError("Followup preregistration bytes changed")
    if sha(SOURCE_DOCUMENT) != admission["instruction_sha256"]:
        raise RuntimeError("Followup instruction file changed")
    for row in admission["new_sources"]:
        if sha(REPO / row["path"]) != row["sha256"]:
            raise RuntimeError(f"Followup admitted source changed: {row['path']}")
    guard_preserved()
    return {"status": "PASS", "commit": commit, "commit_unix": receipt["commit_unix"],
            "limit_seconds": LIMIT_SECONDS, "latest_start": LAST_START}

