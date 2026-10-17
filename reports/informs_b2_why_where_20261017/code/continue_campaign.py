"""Finite continuation of registered B2 mechanism jobs; default does no asset IO.

Seal a metadata-only plan, review it, then explicitly execute it. A restarted
controller attaches to its recorded process identities and never retries jobs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


NAMESPACE = "informs_b2_why_where_20261017"
ORIGINAL = "informs_broad_validation_20261015_v1"
ORIGINAL_SOURCES = ("fit_base.py", "training.py", "networks.py", "data_contract.py",
                    "gpu_runtime.py", "evaluation.py")
NEW_SOURCES = ("mechanism_runner.py", "cap_control.py", "linear_probe.py",
               "pems_resume.py", "pems_cost.py", "continue_campaign.py")
WORKER_NAMES = {"mechanism_runner.py", "pems_cost.py", "linear_probe.py"}
GIB = 1024**3


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def create(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def identity(process):
    return dict(pid=process.pid, create_time=process.create_time(),
                cmdline=process.cmdline(), parent_pid=process.ppid())


def same_process(psutil, item):
    try:
        process = psutil.Process(item["pid"])
        if abs(process.create_time()-item["create_time"]) > 0.001:
            return None
        if process.cmdline() != item["cmdline"]:
            raise RuntimeError("Observed PID/create identity changed its command line")
        return process if process.is_running() else None
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return None


def workers(psutil):
    result = []
    for process in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            info = process.info
            if "python" not in (info["name"] or "").lower():
                continue
            if any(Path(arg).name in WORKER_NAMES for arg in info["cmdline"] or []):
                result.append(identity(process))
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    return result


def tasks():
    rows = []
    for seed in (92601, 92602):
        rows.append(dict(key=f"train-cap_B2_CAP_POINTWISE_{seed}", phase="CAP",
                         job="train-cap", seed=seed, family="B2_CAP_POINTWISE", attached=True))
    for seed in (92601, 92602):
        rows.append(dict(key=f"evaluate-search_B2_{seed}", phase="SAME_UPDATE",
                         job="evaluate-search", seed=seed, family="B2", attached=False,
                         requires=[f"train-cap_B2_CAP_POINTWISE_{seed}"]))
    for seed in (92601, 92602):
        for family in ("B0", "B2", "B2_CAP_POINTWISE"):
            rows.append(dict(key=f"probe_{family}_{seed}", phase="PROBE", job="probe",
                             seed=seed, family=family, attached=False,
                             requires=[f"train-cap_B2_CAP_POINTWISE_{seed}"] if family == "B2_CAP_POINTWISE" else []))
    for family in ("B0", "B2"):
        rows.append(dict(key=f"resume-pems_{family}_92601", phase="PEMS", job="resume-pems",
                         seed=92601, family=family, attached=False))
    for family in ("B0", "B2"):
        for batch in (1, 8):
            rows.append(dict(key=f"cost_{family}_b{batch}", phase="COST", job="cost",
                             family=family, batch=batch, attached=False,
                             requires=[f"resume-pems_{family}_92601"]))
    return rows


def locations(repo, asset):
    repo, asset = Path(repo).resolve(), Path(asset).resolve()
    public = repo/"reports"/NAMESPACE
    private = asset/".cache"/NAMESPACE
    return dict(repo=repo, asset=asset, public=public, private=private,
                code=public/"code", old_source=repo/"research"/ORIGINAL,
                old_cache=asset/".cache"/ORIGINAL, control=private/"CONTINUATION_v1")


def source_check(plan, paths):
    for key, expected in plan["new_sources"].items():
        if digest(paths["code"]/key) != expected:
            raise RuntimeError("Common source violation: " + key)
    for key, expected in plan["original_sources"].items():
        if digest(paths["old_source"]/key) != expected:
            raise RuntimeError("Common original source violation: " + key)
    for name, expected in plan["contract_sources"].items():
        if digest(paths["private"]/name) != expected:
            raise RuntimeError("Common finite contract violation: " + name)


def storage_check(paths):
    roots = (paths["old_source"], paths["old_cache"], paths["private"], paths["public"])
    total = 0
    for root in roots:
        for folder, directories, files in os.walk(root, followlinks=False):
            directories[:] = [name for name in directories if not (Path(folder)/name).is_symlink()]
            for name in files:
                try:
                    total += (Path(folder)/name).stat().st_size
                except FileNotFoundError:
                    continue
    reserve = 256*1024**2
    if total+reserve > 30*GIB:
        raise RuntimeError("Combined R/cache/public storage exceeds30GiB including reserve")
    for root in roots:
        if shutil.disk_usage(root).free < reserve:
            raise RuntimeError("Filesystem free space below256MiB reserve")
    return dict(combined_storage_bytes=total, limit_bytes=30*GIB, reserve_bytes=reserve)


def seal(paths, psutil):
    control = paths["control"]
    if control.exists():
        raise FileExistsError("Continuation directory already exists; inspect existing plan")
    campaign = read(paths["private"]/"CAMPAIGN_v1.json")
    bounds = read(paths["private"]/"EXECUTION_BOUNDS_v1.json")
    if (campaign["CPU_threads_per_worker"] != 2 or campaign["max_model_workers"] != 2
            or campaign["RSS_per_worker_bytes"] != 3*GIB
            or campaign["RSS_model_sum_bytes"] != 6*GIB or campaign["new_GPU_seconds"] != 0):
        raise ValueError("Registered unchanged resource contract differs")
    if time.time() >= bounds["deadline_unix"]:
        raise TimeoutError("New finite campaign deadline has expired")
    original_receipt = paths["old_source"]/"priority_core_followup_20261016_v1/runs/priority_seed2_jena_B2_k6_s92602_v1/SOURCE_CONTRACT.json"
    original = read(original_receipt)["files"]
    if set(original) != set(ORIGINAL_SOURCES):
        raise ValueError("Original six-source contract differs")
    plan = dict(schema="FINITE_B2_CONTINUATION_PLAN_v1", sealed_unix=time.time(),
                repo=str(paths["repo"]), asset=str(paths["asset"]),
                deadline_unix=bounds["deadline_unix"], original_sources=original,
                original_source_receipt_sha256=digest(original_receipt),
                new_sources={name:digest(paths["code"]/name) for name in NEW_SOURCES},
                contract_sources={name:digest(paths["private"]/name) for name in (
                    "CAMPAIGN_v1.json", "EXECUTION_BOUNDS_v1.json", "LAUNCH_INVENTORY_v1.json")},
                phase_order=["CAP", "SAME_UPDATE", "PROBE", "PEMS", "COST"],
                max_model_workers=2, cost_workers=1, CPU_threads=2,
                worker_RSS_bytes=3*GIB, combined_RSS_bytes=6*GIB,
                protected_T_numeric_access=False, old_queue_restart=False,
                retry_failed_or_existing_outputs=False, jobs=tasks(),
                completion_is_not_hypothesis_support=True, initial_storage=storage_check(paths))
    attached = {}
    for row in plan["jobs"]:
        if not row["attached"]:
            continue
        started_path = paths["private"]/"jobs"/row["key"]/"STARTED.json"
        started = read(started_path)
        process = psutil.Process(started["pid"])
        current = identity(process)
        if (started["job"] != row["job"] or started["seed"] != row["seed"]
                or started["family"] != row["family"]
                or started["CPU_threads"] != 2 or started["GPU_used"] is not False
                or started["root_scope_sha256"] != plan["contract_sources"]["CAMPAIGN_v1.json"]
                or started["runner_sha256"] != plan["new_sources"]["mechanism_runner.py"]
                or abs(started["started_unix"]-current["create_time"]) > 60
                or not any(Path(arg).name == "mechanism_runner.py" for arg in current["cmdline"])):
            raise RuntimeError("Existing CAP does not match registered started/process identity")
        tree = [current] + [identity(child) for child in process.children(recursive=True)]
        parent = process.parent()
        if parent is not None and any(Path(arg).name == "mechanism_runner.py" for arg in parent.cmdline()):
            tree.append(identity(parent))
        attached[row["key"]] = dict(started_sha256=digest(started_path), process=current, observed_tree=tree)
    plan["attached_CAP"] = attached
    expected = {(r["pid"], r["create_time"]) for v in attached.values() for r in v["observed_tree"]}
    unexpected = [r for r in workers(psutil) if (r["pid"],r["create_time"]) not in expected]
    if unexpected:
        raise RuntimeError("Unregistered scientific workers exist; do not duplicate or overlap")
    source_check(plan, paths)
    control.mkdir(exist_ok=False)
    create(control/"PLAN.json",plan)
    print(json.dumps(dict(status="SEALED_METADATA_ONLY_NO_DISPATCH", plan=str(control/"PLAN.json"),
                          sha256=digest(control/"PLAN.json"), attached_CAP=2)))


def outputs(row, paths):
    if row["job"] == "cost":
        return [paths["private"]/"PEMS_cost"/f'{row["family"]}_b{row["batch"]}',
                paths["public"]/"costs"/f'pems_{row["family"]}_b{row["batch"]}']
    items = [paths["private"]/"jobs"/row["key"]]
    if row["job"] == "evaluate-search":
        items += [base/"same_update"/f's{row["seed"]}' for base in (paths["private"],paths["public"])]
    elif row["job"] == "probe":
        items += [base/"probes"/f'{row["family"]}_s{row["seed"]}' for base in (paths["private"],paths["public"])]
    elif row["job"] == "resume-pems":
        items += [base/"runs"/f'pems_{row["family"]}' for base in (paths["private"],paths["public"])]
    return items


def result_path(row, paths):
    if row["job"] == "cost":
        return paths["public"]/"costs"/f'pems_{row["family"]}_b{row["batch"]}'/"RESULT.json"
    return paths["private"]/"jobs"/row["key"]/"RESULT.json"


def successful(row, result):
    if row["job"] == "cost":
        return result.get("status") == "COMPLETE_PROTOCOL"
    if result.get("status") != "JOB_COMPLETE":
        return False
    return result.get("result",{}).get("status") in {
        "COMPLETE_SELECTED_V", "COMPLETE_SAME_UPDATE", "COMPLETE"}


def registered_live_rows(plan, state, psutil):
    """Recheck owned PID/create identities before any common-stop closeout.

    A command-line or source violation does not establish process termination.
    Access-denied observations remain unresolved and prevent a closed claim.
    """
    live=[]
    for row in plan["jobs"]:
        record=state["jobs"][row["key"]]
        if not record["status"].startswith("RUNNING"):
            continue
        observed=list(record.get("observed_tree",[]))
        if record.get("process") is not None:
            observed.append(record["process"])
        for item in observed:
            try:
                process=psutil.Process(item["pid"])
                if abs(process.create_time()-item["create_time"])<=0.001 and process.is_running():
                    live.append(row)
                    break
            except (psutil.NoSuchProcess,psutil.ZombieProcess):
                continue
            except psutil.AccessDenied:
                live.append(row)
                break
    return live


def command(row, paths, deadline):
    script = "pems_cost.py" if row["job"] == "cost" else "mechanism_runner.py"
    argv = [sys.executable,str(paths["code"]/script),"--execute","--family",row["family"],
            "--repo",str(paths["repo"]),"--asset",str(paths["asset"])]
    if row["job"] == "cost":
        argv += ["--batch",str(row["batch"]),"--deadline",str(deadline)]
    else:
        argv += ["--job",row["job"],"--seed",str(row["seed"])]
    return argv


def execute(paths, psutil, poll_seconds):
    control = paths["control"]
    plan = read(control/"PLAN.json")
    if str(paths["repo"]) != plan["repo"] or str(paths["asset"]) != plan["asset"]:
        raise ValueError("Explicit roots differ from sealed plan")
    source_check(plan,paths)
    for lease_file in control.glob("PROCESS_*.json"):
        if same_process(psutil,read(lease_file)["process"]) is not None:
            raise RuntimeError("Another continuation controller with the same process identity is active")
    own = identity(psutil.Process())
    create(control/f'PROCESS_{own["pid"]}_{int(own["create_time"]*1000)}.json',
           dict(process=own, plan_sha256=digest(control/"PLAN.json"), started_unix=time.time()))
    state_path=control/"STATE.json"
    if state_path.exists():
        state=read(state_path)
        if state["plan_sha256"] != digest(control/"PLAN.json"):
            raise RuntimeError("Continuation state was bound to another plan")
    else:
        state=dict(schema="FINITE_B2_CONTINUATION_STATE_v1",plan_sha256=digest(control/"PLAN.json"),
                   jobs={}, stop_new_dispatch=False, stop_reasons=[], finished=False)
        for row in plan["jobs"]:
            record=dict(status="PENDING", observed_tree=[])
            if row["attached"]:
                record.update(status="RUNNING_ATTACHED", **plan["attached_CAP"][row["key"]])
            state["jobs"][row["key"]]=record
    handles={}

    def save(event, **fields):
        state["updated_unix"]=time.time()
        temporary=control/"STATE.tmp"
        temporary.write_text(json.dumps(state,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
        temporary.replace(state_path)
        with (control/"JOURNAL.jsonl").open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(dict(time_unix=time.time(),event=event,**fields),ensure_ascii=False,allow_nan=False)+"\n")

    def stop(reason):
        state["stop_new_dispatch"]=True
        if reason not in state["stop_reasons"]:
            state["stop_reasons"].append(reason)
            request=control/"SAFE_STOP_REQUEST.json"
            if not request.exists():
                create(request,dict(reason=reason,created_unix=time.time(),action="save_optimizer_sampler_at_next_safe_update_then_exit",kill_requested=False))
            save("COMMON_STOP_NO_NEW_DISPATCH",reason=reason)

    save("CONTROLLER_ATTACHED",pid=own["pid"])
    while True:
        active=[]
        try:
            if time.time() >= plan["deadline_unix"]:
                stop("New finite campaign deadline reached")
            known=set()
            total_rss=0
            for row in plan["jobs"]:
                record=state["jobs"][row["key"]]
                if not record["status"].startswith("RUNNING"):
                    if record["status"]=="LAUNCH_REQUESTED":
                        stop("Controller interrupted during launch gap; manual identity audit required: "+row["key"])
                    continue
                live=[]
                observed={ (r["pid"],r["create_time"]):r for r in record["observed_tree"] }
                for item in list(observed.values()):
                    process=same_process(psutil,item)
                    if process is not None:
                        for child in process.children(recursive=True):
                            info=identity(child)
                            observed[(info["pid"],info["create_time"])]=info
                record["observed_tree"]=list(observed.values())
                for item in record["observed_tree"]:
                    process=same_process(psutil,item)
                    if process is not None:
                        live.append(item)
                        known.add((item["pid"],item["create_time"]))
                        total_rss+=process.memory_info().rss
                if live:
                    active.append(row)
                    record["last_live_tree"]=live
                    rss=sum(psutil.Process(v["pid"]).memory_info().rss for v in live if psutil.pid_exists(v["pid"]))
                    record["last_tree_RSS_bytes"]=rss
                    if rss>3*GIB:
                        stop("Owned worker tree exceeds3GiB; awaiting safe checkpoint: "+row["key"])
                    continue
                handle=handles.pop(row["key"],None)
                exit_code=handle.wait(timeout=5) if handle is not None else None
                receipt_path=result_path(row,paths)
                record.update(closed_unix=time.time(),all_observed_processes_closed=True,exit_code=exit_code)
                if receipt_path.exists():
                    result=read(receipt_path)
                    record.update(result_sha256=digest(receipt_path),result_status=result.get("status"),
                                  status="COMPLETE" if successful(row,result) and exit_code in (None,0) else "FAILED_OR_PARTIAL")
                    if result.get("error_type") in ("MemoryError",) or "Common source violation" in result.get("error",""):
                        stop("Worker reported common resource/source failure: "+row["key"])
                else:
                    record["status"]="FAILED_NO_RESULT"
                closure=control/(row["key"]+".CLOSED.json")
                if not closure.exists():
                    create(closure,dict(key=row["key"],status=record["status"],
                        closed_unix=record["closed_unix"],observed_tree=record["observed_tree"],
                        all_observed_processes_closed=True,exit_code=exit_code,
                        exit_code_observed=handle is not None,
                        result_sha256=record.get("result_sha256"),
                        attached_worker_exit_code_not_invented=handle is None))
                save("WORKER_TREE_CLOSED",key=row["key"],status=record["status"],exit_code=exit_code)
            unexpected=[v for v in workers(psutil) if (v["pid"],v["create_time"]) not in known]
            if unexpected:
                stop("Unregistered scientific process observed; no duplicate dispatch")
                state["unexpected_scientific_workers"]=unexpected
            if len(active)>2 or total_rss>6*GIB:
                stop("Aggregate model concurrency/RSS contract exceeded")
            source_check(plan,paths)
        except (psutil.NoSuchProcess,psutil.ZombieProcess):
            save("PROCESS_EXIT_DURING_OBSERVATION_RECHECK")
            time.sleep(poll_seconds)
            continue
        except (RuntimeError,ValueError,OSError) as error:
            stop(str(error))
        if state["stop_new_dispatch"]:
            # Re-derive this after an incomplete observation or source exception;
            # an empty local list alone is never evidence of worker termination.
            by_key={r["key"]:r for r in active+registered_live_rows(plan,state,psutil)}
            active=list(by_key.values())
            if active:
                save("WAITING_SAFE_CLOSE_NO_KILL",active=[r["key"] for r in active])
                time.sleep(poll_seconds)
                continue
            for record in state["jobs"].values():
                if record["status"] in ("PENDING","LAUNCH_REQUESTED"):
                    record.update(status="NOT_EXECUTED_COMMON_STOP",reason=state["stop_reasons"])
            state.update(finished=True,status="PARTIAL_COMMON_STOP",owned_models_observed_active_at_close=False,
                         unregistered_models_observed_active_at_close=bool(state.get("unexpected_scientific_workers")))
            save("CONTROLLER_FINISHED_PARTIAL")
            return 2
        pending=[r for r in plan["jobs"] if state["jobs"][r["key"]]["status"]=="PENDING"]
        if not active and not pending:
            state.update(finished=True,status="FINITE_JOBS_FINISHED_AGGREGATION_PENDING",models_observed_active_at_close=False,
                         hypothesis_supported_not_assumed=True,report_aggregation_executed=False)
            save("CONTROLLER_FINISHED")
            print(json.dumps(dict(status=state["status"],jobs={k:v["status"] for k,v in state["jobs"].items()})))
            return 0
        phase=next(name for name in plan["phase_order"] if any(r["phase"]==name for r in active+pending))
        limit=1 if phase=="COST" else 2
        for row in [r for r in pending if r["phase"]==phase]:
            if len(active)>=limit:
                break
            dependencies=[state["jobs"][key] for key in row.get("requires",[])]
            if any(d["status"].startswith("RUNNING") or d["status"]=="PENDING" for d in dependencies):
                continue
            record=state["jobs"][row["key"]]
            if any(d["status"]!="COMPLETE" for d in dependencies):
                record.update(status="NOT_EXECUTED_DEPENDENCY",reason="Required completed artifact unavailable")
                save("DEPENDENCY_SKIPPED",key=row["key"])
                continue
            try:
                source_check(plan,paths)
                record["pre_dispatch_storage"]=storage_check(paths)
                if time.time()>=plan["deadline_unix"]:
                    raise TimeoutError("Finite campaign deadline reached before dispatch")
                if any(path.exists() for path in outputs(row,paths)):
                    record.update(status="NOT_EXECUTED_EXISTING_OUTPUT",reason="Fresh outputs required; no automatic retry")
                    save("EXISTING_OUTPUT_PRESERVED",key=row["key"])
                    continue
                argv=command(row,paths,plan["deadline_unix"])
                record.update(status="LAUNCH_REQUESTED",argv=argv,launch_requested_unix=time.time())
                save("LAUNCH_REQUESTED",key=row["key"])
                environment=os.environ.copy()
                for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
                    environment[name]="2"
                environment.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
                with (control/(row["key"]+".stdout.txt")).open("xb") as stdout, (control/(row["key"]+".stderr.txt")).open("xb") as stderr:
                    child=subprocess.Popen(argv,cwd=paths["repo"],env=environment,stdout=stdout,stderr=stderr,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
                handles[row["key"]]=child
                try:
                    info=identity(psutil.Process(child.pid))
                    tree=[info]
                except psutil.NoSuchProcess:
                    child.wait(timeout=5)
                    info=None
                    tree=[]
                record.update(status="RUNNING_LAUNCHED",process=info,observed_tree=tree,started_unix=time.time())
                active.append(row)
                save("WORKER_DISPATCHED",key=row["key"],process=info)
            except (RuntimeError,TimeoutError) as error:
                stop("Pre-dispatch common failure: "+str(error))
                break
            except (OSError,ValueError,psutil.Error) as error:
                record.update(status="FAILED_TO_LAUNCH",error_class=type(error).__name__,error=str(error))
                save("ROUTE_LAUNCH_FAILED_NO_RETRY",key=row["key"],error=str(error))
        save("OBSERVATION",active=[r["key"] for r in active],phase=phase)
        time.sleep(poll_seconds)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--seal",action="store_true")
    mode.add_argument("--execute",action="store_true")
    parser.add_argument("--repo",type=Path)
    parser.add_argument("--asset",type=Path)
    parser.add_argument("--poll-seconds",type=float,default=15)
    args=parser.parse_args()
    if not args.seal and not args.execute:
        print(json.dumps(dict(status="SOURCE_ONLY_NO_ASSET_IO",input_reads=0,writes=0,model_dispatches=0)))
        return 0
    if args.repo is None or args.asset is None:
        parser.error("Explicit repository and asset roots required")
    if not 1<=args.poll_seconds<=60:
        parser.error("Poll interval must be1..60seconds")
    import psutil
    paths=locations(args.repo,args.asset)
    if args.seal:
        seal(paths,psutil)
        return 0
    return execute(paths,psutil,args.poll_seconds)


if __name__=="__main__":
    raise SystemExit(main())
