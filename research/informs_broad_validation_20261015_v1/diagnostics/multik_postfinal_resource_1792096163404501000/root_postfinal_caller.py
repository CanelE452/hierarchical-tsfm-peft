"""Explicit ROOT postfinal cost allocation; default is metadata preparation only."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

PLAN_REL="diagnostics/multik_postfinal_resource_1792096163404501000/PLAN.json"
PLAN_SHA="45908dfa6bf5354c655bc879143d81a440484927885acda8ea4989cbf9808601"


def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def new(path,value):
    with Path(path).open("x",encoding="utf-8") as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def metadata(public,cache):
    if sha(public/PLAN_REL)!=PLAN_SHA:raise ValueError("Postfinal plan changed")
    plan=read(public/PLAN_REL);folder=(public/PLAN_REL).parent
    for name,digest in plan["new_code_sha256"].items():
        if sha(folder/name)!=digest:raise ValueError("Postfinal code changed")
    sys.path.insert(0,str(folder))
    spec=importlib.util.spec_from_file_location("postfinal_gate",folder/"postfinal_gate.py")
    gate=importlib.util.module_from_spec(spec);sys.modules["postfinal_gate"]=gate;spec.loader.exec_module(gate)
    gate.preparation(public,cache)
    return plan,gate


def resume_metadata(gate,public,path,digest,plan):
    path=Path(path).resolve()
    if not path.is_relative_to(public) or len(digest)!=64 or gate.sha(path)!=digest:
        raise ValueError("Explicit sealed latest public partial batch is required")
    prior=gate.read(path)
    if (prior["pins_sha256"]!=plan["original_pins"]["sha256"]
            or [x["scope_id"] for x in prior["scope_ledger"]]!=[x["scope_id"] for x in plan["full_scope_queue"]]):
        raise ValueError("Latest partial batch must preserve the original20 inputs and queue")
    return prior


@contextmanager
def reservation(factory,cache,out):
    with factory(cache,total_gpu_seconds=43200,reserve_seconds=0,wall_deadline=1792223150,
                 purpose="audit",kind="POSTFINAL_registered_multik_costs_reserve0") as owner:
        try:yield owner
        finally:
            import psutil
            pending=[]
            for row in owner._authorized_children.values():
                try:
                    process=psutil.Process(row["pid"])
                    if abs(process.create_time()-row["create_time"])<.001:pending.append(process)
                except psutil.NoSuchProcess:pass
            if pending:
                new(out/"JOIN_CHILDREN_BEFORE_CLOSE.json",dict(pids=[p.pid for p in pending],termination_requested=False))
                for process in pending:process.wait()


def execute(args,public,cache,plan,gate):
    if not args.root_triggered:raise ValueError("ROOT review and explicit trigger required")
    if any(key.startswith("INFORMS_GPU_") for key in os.environ):raise ValueError("No inherited or nested GPU owner")
    proof=gate.validate_evidence(public,cache,args.final_proof_json,args.final_proof_sha256)
    prior=resume_metadata(gate,public,args.resume_receipt_json,args.resume_receipt_sha256,plan)
    storage=gate.storage_preflight(public,cache,plan)
    sys.path.insert(0,str(public));from resources import GPUReservation,gpu_usage_seconds
    if gpu_usage_seconds(cache)>=43200:raise ValueError("Original cumulative43200 budget exhausted")
    folder=(public/PLAN_REL).parent;stamp=str(time.time_ns());out=folder/("actual_"+stamp);out.mkdir(exist_ok=False)
    private=cache/"final_execution"/("multik_postfinal_"+stamp);private.mkdir(exist_ok=False)
    registry_sha=sha(public/"EXPERIMENT_REGISTRY.csv");selection_sha=sha(public/"SELECTION_LOCK.json")
    os.environ["INFORMS_POSTFINAL_PROOF_JSON"]=str(Path(args.final_proof_json).resolve())
    os.environ["INFORMS_POSTFINAL_PROOF_SHA256"]=args.final_proof_sha256
    spec=importlib.util.spec_from_file_location("postfinal_multik",folder/"postfinal_multik_launcher.py")
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    owner=None;result=None;fatal=None;closure=None;closed_before=gpu_usage_seconds(cache)
    new(out/"RESOURCE_PHASE.json",dict(schema="EXPLICIT_POSTFINAL_ALLOCATION_v1",pre_final_cumulative_ceiling_seconds=39600,
        post_final_cumulative_ceiling_seconds=43200,global_GPU_ceiling_seconds=43200,reserved_final_seconds_now=0,
        previously_closed_GPU_seconds=closed_before,hard_wall_deadline_unix=1792223150,
        reason="Locked T46 and all55 E0 checkpoint inference/rescore attempts terminated before filling preregistered missing costs",
        proof=proof,resume_receipt_public_relative_path=Path(args.resume_receipt_json).resolve().relative_to(public).as_posix(),
        resume_receipt_sha256=args.resume_receipt_sha256,prior_complete_scopes=prior["complete_scopes"],storage_bytes_before=storage,
        plan_sha256=PLAN_SHA,caller_sha256=sha(__file__),queue=plan["full_scope_queue"],new_scientific_conditions=0))
    try:
        with reservation(GPUReservation,cache,out) as owner:
            new(out/"OWNER_SAFE_FIELDS.json",dict(lease_id=owner.lease_id,pid=owner.pid,create_time=owner.create_time,
                started_unix=owner.started_unix,deadline=owner.deadline,total_gpu_seconds=owner.total_gpu_seconds,
                purpose=owner.purpose,pre_final_ceiling_seconds=39600,post_final_ceiling_seconds=43200,
                authentication_values_published=False))
            result=helper.run_with_owner(owner,Path(os.environ["INFORMS_ASSET_ROOT"]),cache,public,
                max_seconds=None,resume_from=Path(args.resume_receipt_json).resolve())
    except BaseException as error:
        fatal=type(error).__name__
        with (private/"CONTROLLER_ERROR.txt").open("x",encoding="utf-8") as stream:stream.write(traceback.format_exc())
    finally:
        os.environ.pop("INFORMS_POSTFINAL_PROOF_JSON",None);os.environ.pop("INFORMS_POSTFINAL_PROOF_SHA256",None)
        if owner is not None:
            receipt=cache/"gpu_receipts"/(owner.lease_id+".json")
            if receipt.exists():
                raw=read(receipt);closure={k:raw.get(k) for k in ("lease_id","status","charged_process_seconds","deadline_overrun_seconds","exception_class")}
                closure.update(cache_relative_path=receipt.relative_to(cache).as_posix(),sha256=sha(receipt))
        preserved=sha(public/"EXPERIMENT_REGISTRY.csv")==registry_sha and sha(public/"SELECTION_LOCK.json")==selection_sha
        code_preserved=all(gate.sha(public/n)==d for n,d in plan["source39"].items())
        valid=bool(closure and closure["status"]=="CLOSED" and closure["deadline_overrun_seconds"]==0 and not (cache/"ACTIVE_GPU_LEASE.json").exists())
        complete=bool(result and result["status"]=="COMPLETE_EXACT20_SCOPES" and valid and preserved and code_preserved and fatal is None)
        new(out/"RESULT.json",dict(status="COMPLETE_K20_POSTFINAL_RESOURCE_PHASE" if complete else "PARTIAL_OR_FAILED_FULL20_PRESERVED",
            original20_queue=plan["full_scope_queue"],multik_batch_result=result,central_closure=closure,resource_valid=valid,
            immutable_selection_and_registry_preserved=preserved,source39_preserved=code_preserved,fatal_error_type=fatal,
            global43200_budget_not_reset=True,model_or_method_reselection=False,new_training_T_or_Git_writes=0))
    print(json.dumps(dict(public_result_relative_path=(out/"RESULT.json").relative_to(public).as_posix(),complete=complete,fatal_error_type=fatal)))
    return 0 if complete else 1


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--execute",action="store_true");p.add_argument("--root-triggered",action="store_true")
    p.add_argument("--final-proof-json",type=Path);p.add_argument("--final-proof-sha256")
    p.add_argument("--resume-receipt-json",type=Path);p.add_argument("--resume-receipt-sha256")
    args=p.parse_args();public=Path(os.environ["INFORMS_PUBLIC_ROOT"]).resolve();cache=Path(os.environ["INFORMS_CACHE_ROOT"]).resolve()
    plan,gate=metadata(public,cache)
    if args.execute:
        if not all([args.final_proof_json,args.final_proof_sha256,args.resume_receipt_json,args.resume_receipt_sha256]):p.error("Explicit final proof and latest partial receipt paths+SHA are required")
        raise SystemExit(execute(args,public,cache,plan,gate))
    print(json.dumps(dict(status="PREPARATION_ONLY_FINAL_PROOF_NOT_YET_VERIFIED",pre_final_ceiling=39600,post_final_ceiling=43200,
        original_queue_scopes=len(plan["full_scope_queue"]),required_T_routes=46,required_E0_checkpoint_attempts=55,
        GPU_owner_created=False,GPU_children_started=0,source39_unchanged=True,
        framework_imports=[n for n in ("numpy","torch","pandas") if n in sys.modules],actual_final_proof_or_arrays_read=False)))


if __name__=="__main__":main()
