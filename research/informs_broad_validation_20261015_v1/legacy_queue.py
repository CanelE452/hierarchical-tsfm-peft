"""Central-leased fresh-process E0 queue; no CUDA/model/data imports here."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import legacy_continuation as lc
from resources import BudgetExceeded, GPUReservation, ResourceBusy, check_storage_budget


def worker_command(*, python_exe, asset_root, cache_root, legacy_id, endpoint, result_path, resume_checkpoint=None):
    command = [str(python_exe), "-X", "utf8", "-B", str(Path(__file__).resolve()), "worker",
               "--asset-root", str(asset_root), "--cache-root", str(cache_root), "--id", legacy_id,
               "--endpoint", str(endpoint), "--result-json", str(result_path), "--execute-training"]
    if resume_checkpoint is not None:
        command.extend(["--resume-budget-checkpoint", str(resume_checkpoint)])
    return command


def run_queue(asset_root, cache_root, *, ids=None, include_controls=True, endpoints=None,
              total_gpu_seconds=43200, reserve_seconds=3600, wall_deadline=1792050350 + 172800,
              python_exe=None, execute=False, resume_checkpoints=None):
    """Root owns one lease over each child until exit; no detached GPU worker.

    Endpoints are breadth-first, preserving every registered unresolved/control
    route. A budget stop leaves later jobs pending instead of claiming complete.
    ``resume_checkpoints`` keys are ``id:target_endpoint`` with relative paths.
    """
    plan = lc.queue_plan(asset_root, ids, include_controls)
    endpoints = list(lc.ENDPOINTS if endpoints is None else endpoints)
    if endpoints != sorted(set(endpoints)) or any(value not in lc.ENDPOINTS for value in endpoints):
        raise ValueError("Increasing registered endpoint schedule required")
    jobs = [dict(id=row["id"], endpoint=endpoint, role=row["role"]) for endpoint in endpoints for row in plan["routes"]]
    if not execute:
        return dict(status="PLAN_ONLY", jobs=jobs, plan=plan, actual_gpu_children=0)
    cache_root, asset_root = Path(cache_root).resolve(), Path(asset_root).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    queue_dir = cache_root / "legacy_queue" / str(time.time_ns())
    queue_dir.mkdir(parents=True, exist_ok=False)
    results = []
    status = "QUEUE_FINISHED"
    for index, job in enumerate(jobs):
        public_endpoint = lc.HERE / "legacy_e0" / "fits" / f"{job['id']}_{job['endpoint']}.json"
        if public_endpoint.exists():
            prior = lc.read_json(public_endpoint)
            if prior.get("status") != "CONTINUED_ENDPOINT_COMPLETE" or prior.get("new_runner_source_sha256") != lc.sha(Path(lc.__file__)):
                raise ValueError("Existing E0 endpoint source/status differs; preserve and audit before reuse")
            old_lock, sources = lc.validate_sources(asset_root)
            last_path = cache_root / "legacy_e0" / "training" / job["id"] / f"last_{job['endpoint']}.pt"
            selected_path = cache_root / "legacy_e0" / "selected" / f"{job['id']}_{job['endpoint']}.pt"
            if (prior.get("id") != job["id"] or prior.get("endpoint") != job["endpoint"] or
                prior.get("source_hashes") != sources or prior.get("statistics_sha256") != old_lock["statistics_hash"] or
                not last_path.is_file() or lc.sha(last_path) != prior.get("last_file_sha256") or
                not selected_path.is_file() or lc.sha(selected_path) != prior.get("selected_file_sha256")):
                raise ValueError("Existing E0 endpoint model/data/source evidence differs")
            results.append(dict(job, status="REUSED_COMPLETE_ENDPOINT", receipt_relative_path=public_endpoint.relative_to(lc.HERE).as_posix()))
            continue
        job_dir = queue_dir / f"{index:03d}_{job['id']}_{job['endpoint']}"
        job_dir.mkdir()
        result_path = job_dir / "RESULT.json"
        try:
            check_storage_budget(cache_root)
            with GPUReservation(cache_root, total_gpu_seconds=total_gpu_seconds, reserve_seconds=reserve_seconds,
                                wall_deadline=wall_deadline, purpose="training", kind=f"E0_{job['id']}_{job['endpoint']}") as lease:
                lease.check()
                if lease.remaining <= 5:
                    raise BudgetExceeded("Insufficient remaining lease to initialize a safe E0 child")
                env = dict(os.environ, **lease.child_environment())
                env.update(INFORMS_ASSET_ROOT=str(asset_root), INFORMS_CACHE_ROOT=str(cache_root),
                           OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
                command = worker_command(python_exe=python_exe or sys.executable, asset_root=asset_root, cache_root=cache_root,
                        legacy_id=job["id"], endpoint=job["endpoint"], result_path=result_path,
                        resume_checkpoint=(resume_checkpoints or {}).get(f"{job['id']}:{job['endpoint']}"))
                with (job_dir / "stdout.log").open("xb") as output, (job_dir / "stderr.log").open("xb") as errors:
                    child = subprocess.Popen(command, env=env, cwd=str(lc.HERE), stdout=output, stderr=errors)
                    # The OS lease remains owned until the child exits, including
                    # model loading, carry serialization and teardown wall time.
                    returncode = child.wait()
                if returncode != 0:
                    results.append(dict(job, status="FAILED_CHILD", exit_code=returncode,
                                        private_job_relative_path=job_dir.relative_to(cache_root).as_posix()))
                    continue
                if not result_path.exists():
                    raise ValueError("Successful E0 child wrote no result receipt")
                result = lc.read_json(result_path)
                if result.get("id") != job["id"] or result.get("endpoint") != job["endpoint"]:
                    raise ValueError("E0 child result identity differs")
                results.append(result)
                if result.get("status") in {"BUDGET_LIMIT", "RESOURCE_GUARD_STOP"}:
                    status = result["status"]
                    break
        except (BudgetExceeded, ResourceBusy) as error:
            results.append(dict(job, status="PENDING_BUDGET_OR_GPU_OWNER", reason=str(error)))
            status = "BUDGET_OR_OWNER_LIMIT"
            break
    completed = {f"{row['id']}:{row['endpoint']}" for row in results if row['status'] in {"CONTINUED_ENDPOINT_COMPLETE", "REUSED_COMPLETE_ENDPOINT"}}
    pending = [job for job in jobs if f"{job['id']}:{job['endpoint']}" not in completed]
    if status == "QUEUE_FINISHED" and pending:
        status = "QUEUE_PARTIAL"
    receipt = dict(status=status, jobs_requested=len(jobs), complete_endpoints=len(completed),
                   results=results, incomplete_or_pending=pending, initial_continuation_fidelity=lc.FIDELITY,
                   accounting="Central GPUReservation exclusive full process wall including authenticated fresh children",
                   reserve_seconds=reserve_seconds, protected_TEST_accessed=False, C_scored=False,
                   research_success=False, source_sha256=lc.sha(Path(__file__)),
                   resource_source_sha256=lc.sha(Path(__file__).with_name("resources.py")),
                   runner_source_sha256=lc.sha(Path(lc.__file__)))
    lc.write_json(queue_dir / "QUEUE_RECEIPT.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "worker"))
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--id", action="append", dest="ids")
    parser.add_argument("--endpoint", action="append", type=int, choices=lc.ENDPOINTS)
    parser.add_argument("--include-controls", action="store_true")
    parser.add_argument("--execute-training", action="store_true")
    parser.add_argument("--result-json")
    parser.add_argument("--resume-budget-checkpoint")
    parser.add_argument("--total-gpu-seconds", type=float, default=43200)
    parser.add_argument("--reserve-seconds", type=float, default=3600)
    parser.add_argument("--wall-deadline", type=float, default=1792050350 + 172800)
    args = parser.parse_args()
    if args.command == "worker":
        if not args.execute_training or len(args.ids or []) != 1 or len(args.endpoint or []) != 1 or not args.result_json:
            parser.error("worker requires one id/endpoint/result-json and --execute-training")
        cache = Path(args.cache_root).resolve()
        result_path = Path(args.result_json).resolve()
        if not result_path.is_relative_to(cache / "legacy_queue"):
            parser.error("worker result must remain in the private legacy_queue directory")
        plan = lc.queue_plan(args.asset_root, args.ids, False)
        try:
            result = lc.run_one(args.asset_root, cache, args.ids[0], args.endpoint[0], plan,
                                resume_budget_checkpoint=args.resume_budget_checkpoint)
        except (BudgetExceeded, ResourceBusy) as error:
            result = dict(id=args.ids[0], endpoint=args.endpoint[0],
                          status="BUDGET_LIMIT" if isinstance(error, BudgetExceeded) else "RESOURCE_GUARD_STOP",
                          initialized=False, no_optimizer_update=True, reason=str(error), endpoint_complete=False,
                          research_success=False, continuation_fidelity=lc.FIDELITY)
        lc.write_json(result_path, result)
    else:
        if args.command == "run" and not args.execute_training:
            parser.error("run requires --execute-training")
        resumes = ({f"{args.ids[0]}:{args.endpoint[0]}":args.resume_budget_checkpoint}
                   if args.resume_budget_checkpoint and len(args.ids or []) == 1 and len(args.endpoint or []) == 1 else None)
        if args.resume_budget_checkpoint and resumes is None:
            parser.error("explicit resume requires one id and target endpoint")
        result = run_queue(args.asset_root, args.cache_root, ids=args.ids, include_controls=args.include_controls,
                            endpoints=args.endpoint, total_gpu_seconds=args.total_gpu_seconds,
                            reserve_seconds=args.reserve_seconds, wall_deadline=args.wall_deadline,
                            execute=args.command == "run", resume_checkpoints=resumes)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
