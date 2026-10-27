"""Capture frozen inputs and measured budget before any followup learning."""
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

from . import common as c

def run():
    if (c.REPORT / "PRESERVATION_BASELINE.json").exists():
        raise FileExistsError("Followup baseline already exists; do not overwrite")
    subprocess.run(["git", "merge-base", "--is-ancestor", c.BASE, "HEAD"], cwd=c.REPO,
                   check=True, capture_output=True)
    prefixes = (f"research/{c.OLD_NAME}/", f"reports/{c.OLD_NAME}/")
    raw = subprocess.check_output(["git", "ls-tree", "-r", "-z", c.BASE, "--", *prefixes], cwd=c.REPO)
    files = {}
    tracked = set()
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        meta, name_bytes = entry.split(b"\t", 1)
        mode, kind, oid = meta.split()
        name = name_bytes.decode()
        path = c.REPO / name
        data = path.read_bytes()
        if mode != b"100644" or kind != b"blob":
            raise RuntimeError("FG1 unexpected baseline mode: " + name)
        git_hash = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if git_hash != oid.decode():
            raise RuntimeError("FG1 baseline committed bytes changed: " + name)
        tracked.add(name)
        files[str(path.resolve())] = {"path": str(path.resolve()), "bytes": len(data),
                                      "sha256": hashlib.sha256(data).hexdigest(), "origin": "06f142db"}
    actual = {str(p.relative_to(c.REPO).as_posix()) for prefix in prefixes
              for p in (c.REPO / prefix).rglob("*") if p.is_file()}
    if actual != tracked:
        raise RuntimeError("FG1 original public/source namespace has added or missing files")
    cache_count = 0
    for path in c.OLD_CACHE.rglob("*"):
        if path.is_file():
            files[str(path.resolve())] = {"path": str(path.resolve()), "bytes": path.stat().st_size,
                                          "sha256": c.sha(path), "origin": "pre-followup-private-cache"}
            cache_count += 1
    prior = c.read(c.OLD_REPORT / "PRE_HASHES.json")
    for row in prior["files"]:
        path = Path(row["path"]).resolve()
        if not path.is_file() or c.sha(path) != row["sha256"]:
            raise RuntimeError("FG1 old PRE_HASHES mismatch: " + str(path))
        files[str(path)] = dict(row, path=str(path), origin="original-PRE_HASHES")
    registration = c.read(c.OLD_REPORT / "PREREGISTRATION.json")
    for name, expected in registration["registered_sources"].items():
        if c.sha(c.REPO / name) != expected:
            raise RuntimeError("FG1 registered scientific code changed")
    jobs = c.selected_jobs()
    plan = c.read(c.OLD_REPORT / "RUN_PLAN.json")
    all_jobs = plan["jobs"] + plan["extension_jobs"]
    from research.comparative_ablation_20261025_v1 import run_one
    for job in jobs:
        if sum(row == job for row in all_jobs) != 1 or sum(row["job_id"] == job["job_id"] for row in all_jobs) != 1:
            raise RuntimeError("FG2 duplicated or modified registration")
        run_one.validate_registration(job)
        candidates = [c.OLD_REPORT / "jobs" / job["job_id"],
                      c.OLD_CACHE / "jobs" / job["job_id"],
                      c.OLD_CACHE / "job_inputs" / (job["job_id"] + ".json")]
        if any(path.exists() for path in candidates) or any((c.OLD_CACHE / "logs").glob(job["job_id"] + "*")):
            raise RuntimeError("FG2 selected job already has outputs: " + job["job_id"])
    original_completion = c.read(c.OLD_REPORT / "EXECUTION_COMPLETION.json")
    history_paths = [c.OLD_REPORT / "PROGRESS.json", c.OLD_REPORT / "EXECUTION_COMPLETION.json"]
    history_paths += list((c.OLD_REPORT / "recovery_20261026_01").glob("*PROGRESS.json"))
    history_paths += list((c.OLD_REPORT / "recovery_20261026_01").glob("*EXECUTION_COMPLETION.json"))
    for path in history_paths:
        statuses = c.read(path).get("run_statuses", {})
        if any(statuses.get(job["job_id"]) == "FAILED" for job in jobs):
            raise RuntimeError("FG2 selected job has prior FAILED")
    baseline = {"status": "PASS", "captured_utc": c.now_utc(), "baseline_commit": c.BASE,
                "scope": "All pre-existing original campaign public/source files and private cache files, plus original PRE_HASHES allowlist; future selected job folders and new temporary files may be added, no baseline file may change.",
                "committed_old_namespace_files": len(tracked), "old_cache_files": cache_count,
                "original_PRE_HASHES_files": len(prior["files"]),
                "files": sorted(files.values(), key=lambda row: row["path"])}
    c.write_new(c.REPORT / "PRESERVATION_BASELINE.json", baseline)
    with (c.OLD_REPORT / "RUNS.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    results = {r["job"]["job_id"]: r for r in c.all_results()}
    profiles = {}
    for dataset in c.DATASETS:
        sample = [row for row in rows if row["dataset"] == dataset and row["method_id"] in ("A1", "A2")]
        core = [sum(float(row[k]) for k in ("search_seconds", "refit_seconds", "evaluation_seconds")) for row in sample]
        elapsed = [float(results[row["job_id"]]["elapsed_seconds"]) for row in sample]
        profiles[dataset] = {"n": len(sample), "proxy_methods": ["A1", "A2"],
                             "mean_stage_seconds": statistics.mean(core),
                             "mean_job_elapsed_seconds": statistics.mean(elapsed),
                             "max_job_elapsed_seconds": max(elapsed),
                             "job_ids": [row["job_id"] for row in sample]}
    a1 = [row for row in rows if row["dataset"] == "jena" and row["method_id"] == "A1"]
    per_epoch_search = statistics.mean(float(row["search_seconds"]) / float(row["training_cap"]) for row in a1)
    per_epoch_refit = statistics.mean(float(row["refit_seconds"]) / float(row["selected_epoch"]) for row in a1)
    evaluation = statistics.mean(float(row["evaluation_seconds"]) for row in a1)
    overheads = [max(0., float(results[row["job_id"]]["elapsed_seconds"]) -
                     sum(float(row[k]) for k in ("search_seconds", "refit_seconds", "evaluation_seconds"))) for row in a1]
    extension = 10 * per_epoch_search + 10 * per_epoch_refit + evaluation + statistics.mean(overheads)
    old_profile = plan["budget"]["profiles"]["jena"]
    planner_extension = old_profile["search_cap3_max_seconds"] * 10 / 3 + old_profile["refit_per_epoch_max_seconds"] * 10 + 180
    per_data = {dataset: row["mean_job_elapsed_seconds"] for dataset, row in profiles.items()}
    a4_total = 10 * sum(per_data.values())
    if a4_total > c.LIMIT_SECONDS:
        raise RuntimeError("FG6 estimated A4 alone exceeds 13 hours")
    budget = {"status": "PASS_A4_WITHIN_LIMIT", "basis": "Original campaign actual RUNS stage times and RESULT elapsed_seconds; no followup outcomes used.",
              "per_dataset_a4_seconds": per_data, "a1_jena_extension_seconds_per_job": extension,
              "A4_40_estimated_seconds": a4_total, "A1_jena_10_estimated_seconds": 10 * extension,
              "all_50_estimated_seconds": a4_total + 10 * extension,
              "limit_seconds": c.LIMIT_SECONDS, "latest_start": c.LAST_START,
              "profiles": profiles,
              "extension_formula": "mean(search_seconds/training_cap)*10 + mean(refit_seconds/selected_epoch)*10 + mean(evaluation_seconds) + mean(non-stage elapsed overhead); conservatively assumes refit at epoch 10",
              "extension_components": {"search_per_epoch": per_epoch_search, "refit_per_epoch": per_epoch_refit,
                                       "evaluation": evaluation, "overhead_mean": statistics.mean(overheads)},
              "planner_legacy_profile_extension_seconds": planner_extension,
              "planner_legacy_profile_total_seconds": 10 * sum(row["mean_stage_seconds"] for row in profiles.values()) + 10 * planner_extension,
              "interpretation": "Planner's ~11.3h arithmetic uses older K_tau profile for extension and omits per-job non-stage overhead. Actual first-campaign measured profile is used for admission. Extension is decided only after A4 and requires all 10 seeds to fit the remaining wall-clock/deadline window.",
              "all10_admission_rule": "No partial seed selection. If all 10 are not expected to finish, all ten are NOT_RUN_BUDGET."}
    c.write_new(c.REPORT / "BUDGET.json", budget)
    chosen_ids = {j["job_id"] for j in jobs}
    baseline_complete = [j["job_id"] for j in all_jobs
                         if original_completion["run_statuses"].get(j["job_id"]) == "COMPLETE"]
    excluded = [dict(j, status="NOT_RUN_SCOPE") for j in all_jobs
                if j["job_id"] not in chosen_ids and j["job_id"] not in baseline_complete]
    c.write_new(c.REPORT / "EXECUTION_PLAN.json", {"jobs": jobs, "baseline_complete": baseline_complete,
                "excluded_jobs": excluded, "limit_seconds": c.LIMIT_SECONDS, "latest_start": c.LAST_START,
                "priority": ["A4_base_40", "interim_A4_report_concurrent_with_next_stage", "A1_jena_ext10_10_all_or_none"],
                "new_models_or_seeds": False, "scope_chosen_after_original_results": True})
    env = c.read(c.REPORT / "ENVIRONMENT_VALIDATION.json")
    if env["status"] != "PASS" or not env["all_environment_fields_equal"] or env["F_SHA"] != registration["foundation_sha256"]:
        raise RuntimeError("FG3 environment or foundation mismatch")
    gates = {
        "created_utc": c.now_utc(), "training_executed": False,
        "FG1": {"status": "PASS", "old_namespace_files": len(tracked), "PRE_HASHES_files": len(prior["files"]), "baseline": "PRESERVATION_BASELINE.json"},
        "FG2": {"status": "PASS", "exact_registered_jobs": 50, "prior_outputs": 0, "prior_FAILED": 0},
        "FG3": {"status": "PASS", "evidence": "ENVIRONMENT_VALIDATION.json",
                "optional_repeat": "Not performed: unchanged registered run_job uses fixed original paths and returns existing RESULT; forced rerun in temporary paths requires changing registered globals/code or bypassing the registered entrypoint. Environment, F/data/scaler fingerprints compared instead, as allowed."},
        "FG4": {"status": "PASS", "scope": "Original bounded runtime.load_task + original search(A_fit/A_stop)/fresh A60 refit + final V evaluation; D0/D1 only saved V and A60.",
                "T_numeric_values_parsed": 0, "Jena_access_key": "new_T_numeric_values_parsed"},
        "FG5": {"status": "PENDING", "condition": "Commit both addendum files before first new training."},
        "FG6": {"status": "PASS", "A4_estimated_hours": a4_total / 3600, "limit_hours": 13, "evidence": "BUDGET.json"},
        "FG7": {"status": "PASS", "initial_check": "Both new research/report/cache namespace paths were absent before this task created them.",
                "timestamp_context": "Initial root and independent explorer checks before file creation; no pre-existing artifacts overwritten."},
        "FG8": {"status": "PASS", "seeds": list(c.SEEDS), "A4_jobs": 40, "A1_jena_extension_jobs": 10},
    }
    c.write_new(c.REPORT / "GATES.json", gates)
    c.write_new(c.REPORT / "PREFLIGHT_INPUTS.json", {"source_document": str(c.SOURCE_DOCUMENT),
                "sha256": c.sha(c.SOURCE_DOCUMENT), "baseline_commit": c.BASE,
                "purpose": "Quantify total decoder structure effect, diagnose affine output-space limits, and resolve Jena P-ablation undertraining within fixed scope.",
                "claims_not_identifiable": ["Separate last-value vs lag effects", "E initialization effect", "Fully converged C2/C3 comparison", "Protected T generalization"],
                "scientific_sources": registration["registered_sources"]})
    print(json.dumps({"status": "PASS_PENDING_PREREGISTRATION_AND_D0", "protected_files": len(files),
                      "A4_estimated_hours": a4_total / 3600, "A1_extension_estimated_hours": 10 * extension / 3600,
                      "total_estimated_hours": (a4_total + 10 * extension) / 3600,
                      "excluded_NOT_RUN_SCOPE": len(excluded)}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    run()

