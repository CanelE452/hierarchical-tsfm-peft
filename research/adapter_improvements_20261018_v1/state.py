"""Public resource/queue snapshots; never changes the private execution ledger."""
import argparse
from collections import Counter
from pathlib import Path
import sys
import time

from .contracts import NAME,SUCCESS,atomic_json,read,sha,utc


def snapshot(ctx):
    import psutil
    repo,cache,report=(Path(ctx[key]) for key in ("repo","cache","report"))
    plan=read(report/"EXPERIMENT_PLAN.json")
    plan_paths=[report/"EXPERIMENT_PLAN.json"]+sorted((report/"SUPPLEMENTAL_PLANS").glob("*.json"))
    plans=[plan]+[read(path) for path in plan_paths[1:]]
    ledger=read(cache/"CAMPAIGN.json")
    sys.path.insert(0,str(repo/"research/informs_broad_validation_20261015_v1"))
    import resources
    charged=resources.gpu_usage_seconds(cache)
    active_seconds=0.
    active_verified=False
    active_path=cache/"ACTIVE_GPU_LEASE.json"
    if active_path.exists():
        owner=read(active_path)
        try:
            active_verified=abs(psutil.Process(owner["pid"]).create_time()-owner["create_time"])<.001
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            pass
        if active_verified:
            active_seconds=max(0.,min(time.time(),owner["deadline"])-owner["started_unix"])
    official=read(report/"OFFICIAL_SOURCE_MANIFEST.json")
    downloads=sum(source["download_bytes"] for source in official["sources"])
    cache_bytes=sum(path.stat().st_size for path in cache.rglob("*") if path.is_file())
    counts=Counter()
    queue=[]
    jobs={}
    owning_plans={}
    for plan_path,registered_plan in zip(plan_paths,plans):
        plan_name=plan_path.relative_to(report).as_posix()
        for job in registered_plan["jobs"]:
            if job["job_id"] in jobs:
                raise ValueError("Supplemental plan duplicates an existing job")
            jobs[job["job_id"]]=job
            owning_plans[job["job_id"]]=plan_name
    receipts={}
    for identifier in jobs:
        path=report/"receipts"/(identifier+".json")
        if path.exists():
            receipts[identifier]=read(path)
    for job in jobs.values():
        receipt=receipts.get(job["job_id"],{})
        status=receipt.get("status","PLANNED")
        counts[status]+=1
        if status in SUCCESS:
            continue
        resumable=status in ("PLANNED","PARTIAL_CHECKPOINTED","RUNNING")
        original="python -m research."+NAME+".run --mode execute --plan reports/"+NAME+"/"+owning_plans[job["job_id"]]+" --asset-root ASSET_ROOT --cache-root PRIVATE_CACHE --report-root REPORT_ROOT --only "+job["job_id"]
        queue.append(dict(job_id=job["job_id"],plan=owning_plans[job["job_id"]],status=status,reason=receipt.get("reason","Registered work has not yet reached a terminal receipt"),
            dependencies=[dict(job_id=identifier,status=receipts.get(identifier,{}).get("status","PLANNED")) for identifier in job["dependency_ids"]],
            expected_remaining_seconds=job["expected_cost"]["wall_seconds"],estimate_scope="Pre-profile conservative registration; consult group profiles for measured complete-bundle admission",
            same_identity_resume_allowed=resumable,
            command=original if resumable else None,
            conditional_command_template=original.replace(owning_plans[job["job_id"]],"APPROVED_NEW_PLAN.json").replace("--only "+job["job_id"],"--only APPROVED_NEW_JOB_ID") if not resumable else None,
            minimum_condition=("Verify existing persistent ledger and checkpoint/source contract; live workers must finish or be confirmed terminal before resume" if resumable else
                "Preserve this terminal receipt. Resolve the recorded asset/numerical/dependency/budget cause and register an explicitly approved new plan/job identity before retry; no automatic grid/solver/budget extension")))
    value=dict(ledger,plan_sha256=plan["plan_sha256"],supplemental_plan_sha256=[item["plan_sha256"] for item in plans[1:]],snapshot_utc=utc(),elapsed_wall_seconds=time.time()-ledger["started_unix"],
        gpu_seconds=charged,gpu_pending_live_process_seconds=active_seconds,gpu_live_owner_verified=active_verified,
        download_actual_bytes=downloads,download_source_manifest_sha256=sha(report/"OFFICIAL_SOURCE_MANIFEST.json"),
        cache_actual_bytes=cache_bytes,retained_plus_new_cache_bytes=ledger.get("retained_ancestral_cache_bytes",0)+cache_bytes,
        job_counts=dict(counts),
        resource_scopes=dict(wall="Whole campaign since original start, including preparation, reporting and publication",
            GPU="Charged reserved process wall, plus explicitly charged conservative legacy regression; live pending time shown separately",
            CPU_worker="Completed CPU job process wall; overlaps campaign wall and GPU activities",
            process_CPU="Handler parent-process CPU only; subprocess and auxiliary test CPU are not a complete aggregate",
            cache="All new campaign bytes, bounded by conservative retained-ancestor30GiB allowance",
            downloads="Pinned source files and GitHub tree responses; no new data/model downloads",
            publication="Remote commit proof is verified separately after normal push"))
    value["remaining"]=dict(wall_seconds=max(0.,ledger["limits"]["wall_seconds"]-value["elapsed_wall_seconds"]),
        GPU_seconds_including_live_pending=max(0.,ledger["limits"]["gpu_seconds"]-charged-active_seconds),
        new_training_wall_allowance_seconds=max(0.,ledger["limits"]["wall_seconds"]-value["elapsed_wall_seconds"]-ledger["limits"]["wall_reserve"]),
        new_training_GPU_allowance_seconds=max(0.,ledger["limits"]["gpu_seconds"]-charged-active_seconds-ledger["limits"]["gpu_reserve"]),
        new_cache_bytes=max(0,ledger["limits"]["cache_bytes"]-cache_bytes),
        source_download_bytes=max(0,ledger["limits"]["download_bytes"]-downloads))
    value["known_lineage_plus_new_GPU_seconds"]=sum(float(item.get("gpu_seconds",0.)) for item in ledger.get("lineage",[]))+charged+active_seconds
    value["known_lineage_sum_scope"]="Arithmetic sum of separately scoped historical campaign records and this campaign, including pending live time; historical limits and ledgers remain unchanged."
    atomic_json(report/"RESOURCE_LEDGER.json",value)
    atomic_json(report/"RESUME_QUEUE.json",dict(campaign=NAME,plan_sha256=plan["plan_sha256"],snapshot_utc=utc(),
        status="EMPTY" if not queue else "REGISTERED_WORK_REMAINS",items=queue,
        immutable_terminal_policy="Failed/blocked/budget-skipped receipts are never overwritten by an implicit retry. PARTIAL checkpoints can resume under the same verified identity."))
    return value


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument("--cache-root",type=Path)
    parser.add_argument("--report-root",type=Path)
    args=parser.parse_args(argv)
    repo=args.repo.resolve()
    value=snapshot(dict(repo=repo,cache=args.cache_root or repo/".cache"/NAME,
                        report=args.report_root or repo/"reports"/NAME))
    print(dict(status="SNAPSHOT",job_counts=value["job_counts"],elapsed_wall_seconds=value["elapsed_wall_seconds"],
               gpu_seconds=value["gpu_seconds"],pending_GPU_seconds=value["gpu_pending_live_process_seconds"],
               new_cache_bytes=value["cache_actual_bytes"],downloads=value["download_actual_bytes"]))


if __name__=="__main__":
    main()
