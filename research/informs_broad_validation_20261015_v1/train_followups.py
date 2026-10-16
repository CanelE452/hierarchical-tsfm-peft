"""Sequential, receipt-driven followups; children own every GPU reservation.

No models/data are loaded here. Planning never starts a child. Actual execution
requires --execute and must be launched after the preceding controller exits.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
from resources import gpu_usage_seconds

HERE = Path(__file__).resolve().parent
DATASETS = ("beijing", "jena", "electricity", "pems08")
TOTAL_GPU = 43200
DESIRED_FINAL_RESERVE = 9000
START_LIMIT = TOTAL_GPU-DESIRED_FINAL_RESERVE
WALL_DEADLINE = 1792050350+172800
SOURCE_FILES = ("train_followups.py", "postfit.py", "deepen.py", "legacy_queue.py", "legacy_continuation.py",
                "learning_extension.py", "fit_base.py", "fit_control.py", "training.py", "networks.py",
                "data_contract.py", "gpu_runtime.py", "resources.py", "combination.py", "raw_calibration.py",
                "prediction_cache.py", "ordered_pilot.py", "registry_extension.py", "evaluation.py", "CONFIG_LOCK.json")


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)


def inside(root, relative):
    root,relative = Path(root).resolve(),Path(relative)
    path = (root/relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root) or path == root:raise ValueError("Relative path escapes its declared root")
    return path


def sources():
    return {name:sha(HERE/name) for name in SOURCE_FILES}


def registry(public):
    with (Path(public)/"EXPERIMENT_REGISTRY.csv").open(encoding="utf-8-sig",newline="") as stream:rows=list(csv.DictReader(stream))
    if len({row["experiment_id"] for row in rows}) != len(rows):raise ValueError("Duplicate scientific registry IDs")
    return rows


def complete(run_id, public, cache):
    path = Path(public)/"runs"/run_id/"BASE_FIT_RECEIPT.json"
    if not path.is_file():return None
    value = read(path)
    if value.get("run_id") != run_id or value.get("status") != "COMPLETE_VALID":raise ValueError("Existing fit receipt is not this complete run")
    artifact = inside(cache,value["artifact_cache_relative_path"])
    if not artifact.is_file() or sha(artifact) != value["artifact_sha256"]:raise ValueError("Fit selected artifact SHA differs")
    contract = path.with_name("SOURCE_CONTRACT.json")
    if not contract.is_file():raise ValueError("Fit has no immutable source contract")
    original = read(contract)
    if original.get("source_sha") != value.get("source_sha") or original.get("spec",{}).get("run_id") != run_id:
        raise ValueError("Fit receipt/source/run identity differs")
    return dict(run_id=run_id,receipt_public_relative_path=path.relative_to(public).as_posix(),receipt_sha256=sha(path),
                artifact_cache_relative_path=value["artifact_cache_relative_path"],artifact_sha256=value["artifact_sha256"],
                source_contract_public_relative_path=contract.relative_to(public).as_posix(),source_contract_sha256=sha(contract),
                source_sha=value["source_sha"],epochs=value["epochs"])


def target_state(run_id, public, cache):
    proof = complete(run_id,public,cache)
    if proof:return "COMPLETE_VALID",proof
    if (Path(public)/"runs"/run_id).exists() or (Path(cache)/"runs"/run_id).exists():return "PARTIAL_REQUIRES_EXPLICIT_RESUME",None
    return "NOT_STARTED",None


def _predict(dataset):
    return dict(label="save_V_"+dataset,kind="inference",dataset=dataset,
                arguments=["postfit.py","--phase","predict","--dataset",dataset])


def _extend_group(base_ids, epoch, public, cache, omissions, label):
    parents, pending, targets, reused = [],[],[],[]
    for base in base_ids:
        parent = base if epoch == 1 else base+"__e1"
        proof = complete(parent,public,cache)
        if proof is None:
            omissions.append(dict(label=label,run_id=parent,status="MISSING_COMPLETE_PARENT",reason="No eligible complete parent receipt; no assumed result"))
            return []
        target = parent+"__e"+str(epoch)
        state,existing = target_state(target,public,cache)
        if state == "PARTIAL_REQUIRES_EXPLICIT_RESUME":
            omissions.append(dict(label=label,run_id=target,status=state,reason="Existing files preserved; automatic continuation is not requested by this controller"))
            return []
        parents.append(proof);targets.append(target)
        if existing:reused.append(existing)
        else:pending.append(parent)
    if not pending:
        omissions.append(dict(label=label,status="ALREADY_COMPLETE_VERIFIED",run_ids=targets))
        return []
    arguments = ["deepen.py"]
    for parent in pending:arguments += ["--parent-id",parent]
    arguments += ["--max-epoch",str(epoch)]
    return [dict(label=label,kind="training",arguments=arguments,parent_proofs=parents,
                 requested_parent_ids=[proof["run_id"] for proof in parents],requested_target_ids=targets,
                 new_target_ids=[parent+"__e"+str(epoch) for parent in pending],
                 reused_complete_targets=reused)]


def build_stage(stage, public, cache, used, endpoint=None):
    public,cache = Path(public).resolve(),Path(cache).resolve()
    rows = registry(public);omissions=[];commands=[]
    selected = [row for row in rows if row["fold"] == "primary" and row["dataset"] in DATASETS and int(row["seed"]) == 92601 and "__" not in row["experiment_id"]]
    lookup = {(row["dataset"],row["block"],row["family"],int(row["latent_k"])):row for row in selected}
    channels = {row["dataset"]:int(row["num_channels"]) for row in selected}
    if stage == "recovery":
        for phase in ("predict","combine","ordered","controls"):
            args=["postfit.py","--phase",phase,"--dataset","beijing"]
            if phase in {"ordered","controls"}:args += ["--epochs",".25"]
            commands.append(dict(label="beijing_recovery_"+phase,kind="training" if phase in {"combine","ordered","controls"} else "inference",arguments=args))
    elif stage in {"e0","e0_next"}:
        endpoint = endpoint or 4800
        cap = min(START_LIMIT,float(used)+7200)
        args=["legacy_queue.py","run","--asset-root","{ASSET_ROOT}","--cache-root","{CACHE_ROOT}",
              "--endpoint",str(endpoint),"--include-controls","--execute-training","--total-gpu-seconds",str(cap),
              "--reserve-seconds","0","--wall-deadline",str(WALL_DEADLINE)]
        if endpoint != 4800:
            prior = 4800 if endpoint == 9600 else 9600
            with (public/"UNRESOLVED_RESOLUTION.csv").open(encoding="utf-8-sig",newline="") as stream:groups=list(csv.DictReader(stream))
            ready=[]
            for group in groups:
                ids=list(dict.fromkeys([group["legacy_id"]]+[value for value in group["matched_control_ids"].split(";") if value]))
                good=True
                for legacy_id in ids:
                    path=public/"legacy_e0/fits"/f"{legacy_id}_{prior}.json"
                    if not path.is_file():good=False;break
                    value=read(path)
                    if value.get("status") != "CONTINUED_ENDPOINT_COMPLETE":good=False;break
                    for relative,key in ((f"legacy_e0/training/{legacy_id}/last_{prior}.pt","last_file_sha256"),
                                         (f"legacy_e0/selected/{legacy_id}_{prior}.pt","selected_file_sha256")):
                        file=inside(cache,relative)
                        if not file.is_file() or sha(file) != value.get(key):good=False;break
                    if not good:break
                if good:ready += ids
                else:omissions.append(dict(label="E0_"+str(endpoint),legacy_id=group["legacy_id"],status="MATCHED_PREVIOUS_ENDPOINT_NOT_COMPLETE"))
            ready=list(dict.fromkeys(ready))
            if not ready:return commands,omissions
            args.remove("--include-controls")  # Every selected matched group is already fully enumerated.
            for legacy_id in ready:args += ["--id",legacy_id]
        commands.append(dict(label="legacy_E0_"+str(endpoint),kind="training",arguments=args,E0_total_gpu_subcap=cap,
                             source_fidelity="RESTARTED_NOT_EXACT_RESUME; existing endpoint/budget evidence remains authoritative"))
    else:
        epoch=1 if stage == "deepen1" else 3
        planned=set()
        def group(ds,block,families,k,label):
            found=[lookup.get((ds,block,family,k)) for family in families]
            if any(row is None for row in found):
                omissions.append(dict(label=label,status="REGISTERED_COMPARISON_ABSENT",dataset=ds,families=list(families),k=k));return
            ids=[row["experiment_id"] for row in found];planned.update(ids)
            actions=_extend_group(ids,epoch,public,cache,omissions,label)
            if actions:commands.extend(actions);commands.append(_predict(ds))
        if epoch == 1:
            for ds in DATASETS:group(ds,"E1",("DIRECT_G",),0,"E1_G_"+ds+"_epoch1")
            for ratio in (4,2,1):
                for ds in DATASETS:
                    if ds in channels:
                        k=math.ceil(channels[ds]/ratio)
                        group(ds,"E1",("B0","B2"),k,f"E1_B0_B2_{ds}_K{k}_epoch1")
            for ratio in (4,1):
                for ds in DATASETS:
                    if ds in channels:
                        k=math.ceil(channels[ds]/ratio);group(ds,"E1",("B4",),k,f"E1_B4_{ds}_K{k}_epoch1")
            allowed={"B0","B2","B4","DIRECT_G","DIRECT_B2","DIRECT_B4","LORA_R8","ADAPTS_BOLT_PORT"}
            for ds in DATASETS:
                for row in selected:
                    if row["dataset"] == ds and row["block"] == "E1" and row["family"] in allowed and row["requirement"] == "MANDATORY_MINIMUM" and row["experiment_id"] not in planned:
                        group(ds,"E1",(row["family"],),int(row["latent_k"]),"E1_remaining_"+row["experiment_id"])
            for ratio in (4,2,1):
                for ds in DATASETS:
                    if ds in channels:
                        k=math.ceil(channels[ds]/ratio);group(ds,"E3",("LAG_CAUSAL","LAG_BIDIR"),k,f"E3_lag_pair_{ds}_K{k}_epoch1")
            for ds in DATASETS:
                if ds in channels:group(ds,"E4",("ORDERED_MULTI_K","MAXK_PCA_TRAINED"),channels[ds],"E4_shared_max_"+ds+"_epoch1")
        else:
            for ds in ("beijing","jena","pems08","electricity"):
                group(ds,"E1",("DIRECT_G",),0,"E1_G_"+ds+"_epoch3")
                for ratio in (4,2,1):
                    if ds in channels:
                        k=math.ceil(channels[ds]/ratio);group(ds,"E1",("B0","B2"),k,f"E1_B0_B2_{ds}_K{k}_epoch3")
    return commands,omissions


def no_active_owner(cache):
    path=Path(cache)/"ACTIVE_GPU_LEASE.json"
    if not path.is_file():return
    value=read(path)
    try:live=abs(psutil.Process(int(value["pid"])).create_time()-float(value["create_time"])) < .001
    except psutil.NoSuchProcess:live=False
    if live:raise RuntimeError("Another GPU owner is active; this controller never waits, interrupts, or kills it")


def save_plan(stage, public, cache, commands, omissions, used):
    stamp=str(time.time_ns());path=Path(public)/"execution_batches"/("followup_plan_"+stage+"_"+stamp+".json")
    plan=dict(stage=stage,commands=commands,omissions=omissions,source_files=sources(),created_unix=time.time(),
              closed_GPU_seconds_at_plan=used,start_training_limit=START_LIMIT,desired_final_reserve=DESIRED_FINAL_RESERVE,
              wall_deadline=WALL_DEADLINE,no_numeric_performance_gate=True,no_outer_GPU_lease=True,
              child_commands=[["python","-X","utf8","-u"]+row["arguments"] for row in commands],
              resume_instruction="Use this saved --resume-plan with --from-command; do not index a regenerated command list")
    write(path,plan);return path,plan


def execute_plan(path, plan, public, cache, asset_root, start=0):
    stamp=str(time.time_ns());private=Path(cache)/"followup_commands"/stamp;private.mkdir(parents=True,exist_ok=False)
    output=Path(public)/"execution_batches"/("followup_execution_"+stamp);output.mkdir(parents=True,exist_ok=False)
    remaining=[];records=[]
    for index,item in enumerate(plan["commands"]):
        if index < start:continue
        if sources() != plan["source_files"]:raise ValueError("Followup/dependent source changed after immutable command registration")
        used=gpu_usage_seconds(cache)
        if time.time() >= WALL_DEADLINE or (item["kind"] == "training" and used >= START_LIMIT):
            remaining=plan["commands"][index:];break
        no_active_owner(cache)
        skip=None
        for proof in item.get("parent_proofs",[]):
            if complete(proof["run_id"],public,cache) != proof:raise ValueError("Registered parent receipt/artifact changed before child")
        for proof in item.get("reused_complete_targets",[]):
            if complete(proof["run_id"],public,cache) != proof:raise ValueError("Verified complete extension changed before child")
        if item.get("new_target_ids"):
            states=[target_state(run,public,cache)[0] for run in item["new_target_ids"]]
            if all(state == "COMPLETE_VALID" for state in states):skip="SKIPPED_COMPLETE_EXTENSION_VERIFIED"
            elif any(state != "NOT_STARTED" for state in states):skip="SKIPPED_EXISTING_EXTENSION_REPLAN_OR_EXPLICIT_RESUME"
        if skip:
            record=dict(command_index=index,label=item["label"],status=skip,scientific_completion_inferred=False)
            write(output/f"COMMAND_{index:03d}.json",record);records.append(record);continue
        arguments=[value.replace("{CACHE_ROOT}",str(cache)).replace("{ASSET_ROOT}",str(asset_root)) for value in item["arguments"]]
        env=dict(os.environ,INFORMS_CACHE_ROOT=str(cache),INFORMS_ASSET_ROOT=str(asset_root),
                 OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")
        command=[sys.executable,"-X","utf8","-u",str(HERE/arguments[0])]+arguments[1:]
        started=time.time();log=private/f"{index:03d}.log"
        write(output/f"COMMAND_{index:03d}_PRELAUNCH.json",dict(command_index=index,label=item["label"],
              human_command=plan["child_commands"][index],source_files=plan["source_files"],registered_unix=started,
              closed_GPU_seconds=used,parent_proofs=item.get("parent_proofs",[])))
        print(json.dumps(dict(event="FOLLOWUP_COMMAND_START",index=index,label=item["label"])),flush=True)
        with log.open("xb") as stream:code=subprocess.Popen(command,cwd=HERE,env=env,stdout=stream,stderr=subprocess.STDOUT).wait()
        record=dict(command_index=index,label=item["label"],returncode=code,
                    status="EXIT_ZERO_REQUIRES_SCIENTIFIC_RECEIPT_REVIEW" if code == 0 else "PROCESS_FAILED_OTHER_COMMANDS_CONTINUE",
                    started_unix=started,completed_unix=time.time(),private_log_sha256=sha(log),
                    closed_GPU_seconds_after=gpu_usage_seconds(cache),T_final_accessed_by_this_controller=False,
                    scientific_completion_inferred=False,outer_GPU_lease=False)
        write(output/f"COMMAND_{index:03d}.json",record);records.append(record)
        print(json.dumps(dict(event="FOLLOWUP_COMMAND_DONE",index=index,returncode=code)),flush=True)
    summary=dict(status="DEFERRED_RESOURCE" if remaining else "COMMAND_LIST_FINISHED_REQUIRES_RECEIPT_REVIEW",
                 plan_public_relative_path=Path(path).relative_to(public).as_posix(),plan_sha256=sha(path),
                 records=records,remaining_commands=remaining,closed_GPU_seconds=gpu_usage_seconds(cache),
                 research_completion_claimed=False,automatic_T_lock=False,automatic_push=False)
    write(output/"EXECUTION_RECEIPT.json",summary);return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("recovery","e0","deepen1","deepen3","all"),default="all")
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--resume-plan")
    parser.add_argument("--from-command",type=int,default=0)
    args=parser.parse_args()
    cache=Path(os.environ.get("INFORMS_CACHE_ROOT",str(HERE.parents[1]/".cache"/HERE.name))).resolve()
    asset=Path(os.environ.get("INFORMS_ASSET_ROOT",str(HERE.parents[1]))).resolve()
    if args.from_command < 0 or (args.from_command and not args.resume_plan):parser.error("--from-command requires a saved --resume-plan")
    if args.resume_plan:
        path=inside(HERE,args.resume_plan);plan=read(path)
        if sources() != plan["source_files"]:raise ValueError("Saved plan source differs; preserve it and create a fresh plan")
        if args.execute:execute_plan(path,plan,HERE,cache,asset,args.from_command)
        else:print(json.dumps(dict(status="PLAN_ONLY",commands=plan["child_commands"],no_child_started=True)))
        return
    stages=["recovery","e0","deepen1","deepen3","e0_next9600","e0_next19200"] if args.stage == "all" else (["deepen3","e0_next9600","e0_next19200"] if args.stage == "deepen3" else [args.stage])
    for stage in stages:
        used=gpu_usage_seconds(cache)
        actual="e0_next" if stage.startswith("e0_next") else stage
        endpoint=int(stage[len("e0_next"):]) if actual == "e0_next" else None
        commands,omissions=build_stage(actual,HERE,cache,used,endpoint)
        path,plan=save_plan(stage,HERE,cache,commands,omissions,used)
        print(json.dumps(dict(event="FOLLOWUP_PLAN_REGISTERED",plan=path.relative_to(HERE).as_posix(),commands=len(commands),omissions=len(omissions))),flush=True)
        if args.execute:
            result=execute_plan(path,plan,HERE,cache,asset)
            if result["status"] == "DEFERRED_RESOURCE":break


if __name__ == "__main__":main()
