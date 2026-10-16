"""Seal terminal metadata for the unchanged postfinal gate; never run GPU work."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

PREPARATION_REL="diagnostics/multik_postfinal_resource_1792096163404501000"
PLAN_SHA="45908dfa6bf5354c655bc879143d81a440484927885acda8ea4989cbf9808601"
GATE_SHA="240301a5e74d88430cee7b1b150c11e9b846ebfdd5606829aaa800601c78ce9c"
DRIVER_REL="final_execution/ROOT_final_T46_E0B55_1792097100000.py"
DRIVER_SHA="bc512e55ad50de138b4f43206a5f34d0d29f085237d6681a425c4c857495d975"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def new(path,value):
    with Path(path).open("x",encoding="utf-8") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")


def metadata(public,cache):
    folder=public/PREPARATION_REL
    if sha(folder/"PLAN.json")!=PLAN_SHA or sha(folder/"postfinal_gate.py")!=GATE_SHA:
        raise ValueError("Official postfinal preparation changed")
    if sha(cache/DRIVER_REL)!=DRIVER_SHA:raise ValueError("ROOT final driver changed")
    spec=importlib.util.spec_from_file_location("proof_builder_original_gate",folder/"postfinal_gate.py")
    gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
    gate.preparation(public,cache)
    return gate


def pinned_json(path,public):
    path=Path(path).resolve()
    if not path.is_relative_to(public):raise ValueError("Final metadata must be public-relative")
    raw=path.read_bytes()
    return json.loads(raw.decode("utf-8-sig")),dict(public_relative_path=path.relative_to(public).as_posix(),
        sha256=hashlib.sha256(raw).hexdigest())


def assemble(result,safe_binding,safe,lock_binding,t_binding,e_binding):
    if (result.get("status")!="TERMINAL_ACTUAL_PHASES_REVIEW_REQUIRED"
            or result.get("closure_status")!="CLOSED"
            or any(result.get(k) is not True for k in ("source39_unchanged","registry_bytes_unchanged","selection_lock_unchanged"))):
        raise ValueError("Actual final RESULT is not a preserved terminal record")
    if (result["owner_safe_fields"]!=safe_binding or result["phase_outputs"].get("T_final")!=t_binding
            or result["phase_outputs"].get("E0_B")!=e_binding):
        raise ValueError("Explicit final paths do not match ROOT output bindings")
    processes=[dict(pid=safe["pid"],create_time=safe["create_time"])]+[
        dict(pid=x["pid"],create_time=x["create_time"]) for x in safe["authorized_children"]]
    if not safe["authorized_children"] or len(processes)!=len({(x["pid"],x["create_time"]) for x in processes}):
        raise ValueError("Owner/authenticated child process set is missing or duplicated")
    return dict(schema="POSTFINAL_RESOURCE_EVIDENCE_v1",status="SEALED_ACTUAL_METADATA_CANDIDATE",
        selection_lock=lock_binding,T_batch=t_binding,E0_B_registry=e_binding,
        phase_owners=[dict(phase=phase,output_binding=value,owner_safe_fields=safe_binding,
            central_closure=result["central_closure"]) for phase,value in [("T_final",t_binding),("E0_B",e_binding)]],
        prior_processes=processes)


def verify_processes_terminated(processes):
    import psutil
    for item in processes:
        try:
            if abs(psutil.Process(item["pid"]).create_time()-item["create_time"])<.001:
                raise ValueError("Original final runtime or authenticated child has not terminated")
        except psutil.NoSuchProcess:pass


def execute_proof(args,public,cache,gate):
    result,result_binding=pinned_json(args.root_result_json,public)
    safe,safe_binding=pinned_json(args.owner_safe_json,public)
    _,lock_binding=pinned_json(args.selection_lock_json,public)
    _,t_binding=pinned_json(args.T_batch_json,public)
    _,e_binding=pinned_json(args.E0_registry_json,public)
    if lock_binding["public_relative_path"]!="SELECTION_LOCK.json":raise ValueError("Canonical selection lock required")
    proof=assemble(result,safe_binding,safe,lock_binding,t_binding,e_binding)
    verify_processes_terminated(proof["prior_processes"])
    if (cache/"ACTIVE_GPU_LEASE.json").exists():raise ValueError("An active GPU owner remains")
    binding=result["central_closure"];closure_path=gate.inside(cache,binding["cache_relative_path"])
    if sha(closure_path)!=binding["sha256"]:raise ValueError("ROOT central closure changed")
    closure=read(closure_path)
    if (closure["lease_id"]!=safe["lease_id"] or closure["status"]!="CLOSED"
            or closure["deadline_overrun_seconds"]!=0 or closure["started_unix"]!=safe["started_unix"]):
        raise ValueError("Actual ROOT owner has no clean bounded closure")
    inputs=[result_binding,safe_binding,lock_binding,t_binding,e_binding]
    for item in inputs:
        if sha(gate.inside(public,item["public_relative_path"]))!=item["sha256"]:
            raise ValueError("Final metadata changed during capture")
    stamp=str(time.time_ns());folder=public/"diagnostics"/("postfinal_resource_evidence_"+stamp)
    folder.mkdir(exist_ok=False)
    proof["provenance"]=dict(ROOT_terminal_RESULT=result_binding,builder_sha256=sha(__file__),
        ROOT_final_driver_sha256=DRIVER_SHA,official_gate_sha256=GATE_SHA,
        actual_owner_and_authenticated_children_observed_terminated=True,model_arrays_GPU_or_T_execution=0)
    path=folder/"FINAL_PROOF.json";new(path,proof);digest=sha(path)
    new(folder/"INPUT_BINDINGS.json",dict(public_bindings=inputs,central_closure=binding,source39_plan_sha256=PLAN_SHA))
    validation=None;error_type=None
    try:
        validation=gate.validate_evidence(public,cache,path,digest)
    except Exception as error:
        error_type=type(error).__name__
        with (cache/"final_execution"/("postfinal_proof_error_"+stamp+".txt")).open("x",encoding="utf-8") as stream:
            stream.write(traceback.format_exc())
    status="PASS_OFFICIAL_POSTFINAL_METADATA_GATE" if validation else "FAILED_PRESERVED_METADATA_CANDIDATE"
    new(folder/"VALIDATION_RESULT.json",dict(status=status,proof_public_relative_path=path.relative_to(public).as_posix(),
        proof_sha256=digest,official_gate_sha256=GATE_SHA,validation=validation,error_type=error_type,
        no_GPU_owner_child_training_lock_T_arrays_or_models_created=True))
    print(json.dumps(dict(status=status,proof_public_relative_path=path.relative_to(public).as_posix(),
        proof_sha256=digest,validation_public_relative_path=(folder/"VALIDATION_RESULT.json").relative_to(public).as_posix(),
        validation_sha256=sha(folder/"VALIDATION_RESULT.json")),ensure_ascii=False))
    return 0 if validation else 1


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--execute-proof",action="store_true")
    p.add_argument("--root-result-json",type=Path);p.add_argument("--owner-safe-json",type=Path)
    p.add_argument("--selection-lock-json",type=Path);p.add_argument("--T-batch-json",dest="T_batch_json",type=Path)
    p.add_argument("--E0-registry-json",dest="E0_registry_json",type=Path)
    args=p.parse_args();public=Path(os.environ["INFORMS_PUBLIC_ROOT"]).resolve();cache=Path(os.environ["INFORMS_CACHE_ROOT"]).resolve()
    gate=metadata(public,cache)
    if args.execute_proof:
        if not all([args.root_result_json,args.owner_safe_json,args.selection_lock_json,args.T_batch_json,args.E0_registry_json]):
            p.error("All five explicit final metadata paths are required")
        raise SystemExit(execute_proof(args,public,cache,gate))
    print(json.dumps(dict(status="PREPARATION_ONLY_NO_ACTUAL_PHASE_JSON_READ",builder_sha256=sha(__file__),
        ROOT_final_driver_sha256=DRIVER_SHA,official_gate_sha256=GATE_SHA,source39_unchanged=True,
        actual_phase_JSON_reads=0,OS_process_queries=0,artifact_or_array_loads=0,GPU_owner_or_child_created=False,
        framework_imports=[name for name in ("numpy","torch","pandas","psutil") if name in sys.modules])))


if __name__=="__main__":main()
