"""Immutable scientific identities and explicit finite campaign accounting."""
from pathlib import Path
import time

from research.adapter_diagnostics_20261017_v2.core import atomic_json, digest, read, sha, seal_plan, utc

NAME = "adapter_improvements_20261018_v1"
REQUEST_SHA256 = "55af78e3ade0d4cebcbc09f0f453b457afc7f748b2a2316486e4eadf8869e679"
DEFAULT_LIMITS = dict(wall_seconds=21600, gpu_seconds=14400, wall_reserve=4320,
                      gpu_reserve=2880, cache_bytes=30 * 1024**3,
                      download_bytes=5 * 1024**3, cpu_workers=1, gpu_workers=1,
                      blas_threads=2, stop_new_fraction=.8)
SUCCESS = {"COMPLETE", "REUSED_VERIFIED"}
STATES = SUCCESS | {"PLANNED", "RUNNING", "FAILED", "SKIPPED_BUDGET", "PARTIAL_CHECKPOINTED",
                    "BLOCKED_ASSET", "BLOCKED_IMPLEMENTATION", "BLOCKED_DATA_CONTRACT",
                    "BLOCKED_MISSING_A_FIT_CHECKPOINT", "NOT_CONVERGED", "NOT_RUN_DEPENDENCY",
                    "TIMEOUT", "FAILED_NUMERICAL"}
ROOTS = ("cache", "previous_cache", "report", "asset", "repo")


def seal_job(recipe, sources):
    body = dict(recipe, scientific_source_hashes=dict(sources))
    return dict(body, scientific_identity=digest(body))


def validate_plan(plan, repo=None):
    if seal_plan(plan)["plan_sha256"] != plan.get("plan_sha256"):
        raise ValueError("Sealed plan identity mismatch")
    seen = set()
    for job in plan["jobs"]:
        if job["job_id"] in seen:
            raise ValueError("Duplicate job identity")
        body = {key: value for key, value in job.items() if key != "scientific_identity"}
        if digest(body) != job["scientific_identity"]:
            raise ValueError("Scientific job identity mismatch")
        if not set(job.get("dependency_ids", [])).issubset(seen):
            raise ValueError("Dependency missing or not topologically ordered")
        seen.add(job["job_id"])
    if repo is not None:
        for relative, expected in plan["source_hashes"].items():
            target = (Path(repo) / relative).resolve()
            if not target.is_relative_to(Path(repo).resolve()) or sha(target) != expected:
                raise ValueError("Sealed scientific source changed: " + relative)


def select_jobs(plan, only=None):
    selected = set(only or [])
    registered = {job["job_id"] for job in plan["jobs"]}
    if selected - registered:
        raise ValueError("--only must name registered exact job IDs: " + ",".join(sorted(selected-registered)))
    return [job for job in plan["jobs"] if not selected or job["job_id"] in selected]


def binding(path, ctx):
    path = Path(path).resolve()
    for name in ROOTS:
        if name in ctx:
            root = Path(ctx[name]).resolve()
            if path.is_relative_to(root):
                return dict(reference=name + "://" + path.relative_to(root).as_posix(),
                            sha256=sha(path), bytes=path.stat().st_size)
    raise ValueError("Artifact is outside the registered private/public roots")


def resolve_binding(value, ctx):
    scheme, relative = value["reference"].split("://", 1)
    if scheme not in ROOTS or scheme not in ctx or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Unregistered or escaping artifact binding")
    root = Path(ctx[scheme]).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
        raise ValueError("Artifact binding is missing or outside its root")
    if sha(path) != value["sha256"]:
        raise ValueError("Artifact hash mismatch: " + value["reference"])
    return path


def campaign_ledger(path, started_unix=None, lineage=None, limits=None):
    path = Path(path)
    if path.exists():
        value = read(path)
        if value.get("campaign") != NAME:
            raise ValueError("Campaign ledger belongs to another campaign")
        if limits is not None and value["limits"] != limits:
            raise ValueError("A resumed campaign cannot reset its limits")
        return value
    value = dict(campaign=NAME, started_unix=time.time() if started_unix is None else started_unix,
                 limits=dict(DEFAULT_LIMITS if limits is None else limits),
                 lineage=[] if lineage is None else lineage, gpu_seconds=0., cpu_worker_seconds=0.,
                 process_cpu_seconds=0., download_reserved_bytes=0, created_utc=utc())
    atomic_json(path, value)
    return value


def admit_bundle(ledger, expected_wall, expected_gpu, now=None, final=False):
    if min(expected_wall, expected_gpu) < 0:
        raise ValueError("Admission requires nonnegative complete-bundle estimates")
    now = time.time() if now is None else now
    limits = ledger["limits"]
    elapsed = now - ledger["started_unix"]
    used_gpu = ledger["gpu_seconds"]
    if not final and (elapsed >= limits["wall_seconds"] * limits["stop_new_fraction"] or
                      used_gpu >= limits["gpu_seconds"] * limits["stop_new_fraction"]):
        return False
    return (expected_wall <= limits["wall_seconds"] - elapsed - (0 if final else limits["wall_reserve"]) and
            expected_gpu <= limits["gpu_seconds"] - used_gpu - (0 if final else limits["gpu_reserve"]))


def verify_references(value, ctx):
    if isinstance(value, dict):
        if "reference" in value and "sha256" in value:
            resolve_binding(value, ctx)
        for child in value.values():
            verify_references(child, ctx)
    elif isinstance(value, list):
        for child in value:
            verify_references(child, ctx)


def reusable(path, job, ctx):
    if not Path(path).exists():
        return False
    receipt = read(path)
    if receipt.get("status") not in SUCCESS or receipt.get("scientific_identity") != job["scientific_identity"]:
        return False
    artifacts = receipt.get("artifact_hashes", {})
    if not artifacts:
        return False
    for relative, expected in artifacts.items():
        resolve_binding(dict(reference="report://" + relative, sha256=expected), ctx)
    verify_references(receipt.get("result", {}), ctx)
    return True


def finish_job(path, job, result, ctx):
    value = dict(result)
    if value["status"] not in STATES:
        raise ValueError("Unregistered terminal job status: " + value["status"])
    artifacts = {}
    for item in value.pop("artifact_paths", []):
        target = Path(item).resolve()
        relative = target.relative_to(Path(ctx["report"]).resolve()).as_posix()
        artifacts[relative] = sha(target)
    if value["status"] in SUCCESS and not artifacts:
        raise ValueError("A successful job needs actual bound public evidence")
    value.update(job_id=job["job_id"], scientific_identity=job["scientific_identity"],
                 plan_sha256=ctx.get("plan",{}).get("plan_sha256"),
                 artifact_hashes=artifacts, finished_utc=utc())
    atomic_json(path, value)
    return value
