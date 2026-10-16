"""Metadata-only release of the final reserve after locked T and E0 attempts."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

HERE = Path(__file__).resolve().parent
MUTABLE = {"execution_status","training_status","validity_status","scientific_status",
           "publication_status","reason","resume_command"}
DONE_E0 = {"REPLAY_RESCORE_COMPLETE","REUSED_RESCORED","CURVE_SCORE_MISMATCH"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):value.update(block)
    return value.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def inside(root,relative):
    root=Path(root).resolve();value=Path(relative);path=(root/value).resolve()
    if value.is_absolute() or path==root or not path.is_relative_to(root):
        raise ValueError("Unsafe metadata binding")
    return path


def bound(public,binding):
    path=inside(public,binding["public_relative_path"])
    if sha(path)!=binding["sha256"]:raise ValueError("Sealed final evidence changed")
    return read(path)


def unique_records(records,keys):
    values=[tuple(row[key] for key in keys) for row in records]
    if len(values)!=len(set(values)):raise ValueError("Duplicate final evidence identity")
    return set(values)


def validate_T_documents(plan,lock,batch):
    ids=plan["selected_run_ids"]
    if len(ids)!=46 or len(set(ids))!=46 or lock.get("locked") is not True:
        raise ValueError("Exact locked T46 is required")
    if lock["allowed_run_ids"]!=ids or lock["source_bindings"]!=plan["source18"]:
        raise ValueError("Final selection identity or sources differ")
    if (batch.get("status")!="COMPLETE_SELECTED_ROUTES" or batch.get("phase")!="T_final"
            or batch.get("run_ids")!=ids or batch.get("selection_sha256")!=lock["selection_sha256"]
            or batch.get("succeeded")!=46 or batch.get("failed")!=0 or batch.get("deferred_run_ids")!=[]):
        raise ValueError("Locked T46 has not completed")
    if unique_records(batch["records"],["run_id"])!={(i,) for i in ids} or any(row["status"]!="PASS" for row in batch["records"]):
        raise ValueError("All exact T46 routes must have terminal PASS")
    return 46


def validate_E0_records(plan,summary):
    expected={(x["id"],x["update"]) for x in plan["legacy_checkpoints"]}
    rows=summary["results"]
    if len(expected)!=55 or summary.get("available_checkpoint_count")!=55:
        raise ValueError("The exact55 legacy inventory is required")
    if summary.get("protected_TEST_accessed") is not False or summary.get("C_scored") is not False:
        raise ValueError("Legacy evaluation scope changed")
    if unique_records(rows,["id","update"])!=expected:
        raise ValueError("Legacy attempts have missing or extra checkpoints")
    completed=failed=0
    for row in rows:
        if row["status"] in DONE_E0 and "score" in row:completed+=1
        elif row["status"]=="REPLAY_FAILED" and int(row.get("actual_F_calls",0))>0:failed+=1
        else:raise ValueError("Legacy inventory/load/resource stop is not an actual inference attempt")
    return dict(rescored_records=completed,failed_inference_attempts=failed,curve_match_required=False)


def preparation(public,cache):
    public,cache=Path(public).resolve(),Path(cache).resolve();plan=read(HERE/"PLAN.json")
    if plan["pre_final_cumulative_ceiling_seconds"]!=39600 or plan["post_final_cumulative_ceiling_seconds"]!=43200:
        raise ValueError("Explicit pre/post allocation differs")
    for name,digest in plan["source39"].items():
        if sha(inside(public,name))!=digest:raise ValueError("Original source39 changed")
    for root,item in [(cache,plan["original_launcher"]),(public,plan["original_pins"]),
                      (public,plan["selection_review46"]),(public,plan["legacy_inventory"])]:
        if sha(inside(root,item["relative_path"]))!=item["sha256"]:raise ValueError("Preparation input changed")
    if read(inside(public,plan["original_pins"]["relative_path"]))["queue"]!=plan["full_scope_queue"]:
        raise ValueError("Original full20 queue changed")
    return plan


def storage_preflight(public,cache,plan):
    roots=[Path(public),Path(cache)];pending=plan["pending_storage_allowance_bytes"]
    size=sum(p.stat().st_size for root in roots for p in root.rglob("*")
             if p.is_file() and not p.is_symlink())
    if size+pending>plan["combined_storage_limit_bytes"] or min(shutil.disk_usage(p).free for p in roots)<pending:
        raise ValueError("Original combined30GiB/pending/free-space boundary exhausted")
    return size


def validate_phase_owner(owner,closure,phase_started,phase_finished):
    if (owner["lease_id"]!=closure["lease_id"] or closure.get("status")!="CLOSED"
            or closure.get("deadline_overrun_seconds")!=0
            or closure["started_unix"]!=owner["started_unix"]
            or not closure["started_unix"]<=phase_started<=phase_finished<=closure["finished_unix"]):
        raise ValueError("Actual final phase is not bound to its clean owner closure")
    return [(owner["pid"],owner["create_time"])]+[(x["pid"],x["create_time"]) for x in owner["authorized_children"]]


def validate_evidence(public,cache,proof_path,expected_sha,owner=None):
    public,cache=Path(public).resolve(),Path(cache).resolve();plan=preparation(public,cache)
    proof_path=Path(proof_path).resolve()
    if not proof_path.is_relative_to(public) or len(expected_sha)!=64 or sha(proof_path)!=expected_sha:
        raise ValueError("Explicit sealed public final proof is required")
    proof=read(proof_path)
    if proof.get("schema")!="POSTFINAL_RESOURCE_EVIDENCE_v1":raise ValueError("Unknown evidence schema")
    lock=bound(public,proof["selection_lock"]);batch=bound(public,proof["T_batch"])
    validate_T_documents(plan,lock,batch)
    if fingerprint({k:v for k,v in lock.items() if k!="selection_sha256"})!=lock["selection_sha256"]:
        raise ValueError("Official selection fingerprint differs")
    with (public/"EXPERIMENT_REGISTRY.csv").open(encoding="utf-8-sig",newline="") as stream:rows=list(csv.DictReader(stream))
    if len({x["experiment_id"] for x in rows})!=len(rows) or any(x["execution_status"]=="RUNNING" for x in rows):
        raise ValueError("Registry is not quiescent")
    science=sorted(({k:v for k,v in x.items() if k not in MUTABLE} for x in rows),key=lambda x:x["experiment_id"])
    if fingerprint(science)!=lock["registry_scientific_sha"]:raise ValueError("Locked registry science changed")
    T_reference_times=[]
    for item in batch["records"]:
        m=bound(public,dict(public_relative_path=item["manifest_public_relative_path"],sha256=item["manifest_sha256"]))
        run=item["run_id"];binding=lock["route_bindings"][run];meta=m["metadata"]
        reference=read(inside(public,item["manifest_public_relative_path"]).with_name("REFERENCE_CONTRACT.json"))
        T_reference_times.append(reference["created_unix"])
        if (m["run_id"]!=run or m["split"]!="T_final" or m["data_sha"]!=binding["data_sha"]
                or m["scaler_sha"]!=binding["scaler_sha"] or m["source_sha"]!=plan["route_source_sha"]
                or meta["route_sha"]!=lock["allowed_route_sha"][run]
                or fingerprint(meta["route_spec"])!=lock["allowed_route_sha"][run]
                or meta["artifact_bindings"]!=binding["artifact_bindings"] or meta.get("T_gate_checked") is not True):
            raise ValueError("Actual T metadata differs from the exact locked route")
    legacy=bound(public,proof["E0_B_registry"]);attempts=validate_E0_records(plan,legacy)
    if (legacy["source_hashes"]!=plan["legacy_source_hashes"]
            or legacy["statistics_sha256"]!=plan["legacy_statistics_sha256"]
            or legacy["inventory"]["source_hashes"]!=plan["legacy_source_hashes"]
            or legacy["inventory"]["foundation_sha256"]!=plan["legacy_foundation_sha256"]
            or legacy["evaluation_source_sha256"]!=plan["source39"]["legacy_evaluation.py"]):
        raise ValueError("Old source or statistics changed")
    actual_inventory={}
    for route in legacy["inventory"]["routes"]:
        for cp in route.get("checkpoints",[]):
            key=(route["id"],cp["update"])
            if key in actual_inventory:raise ValueError("Duplicate E0 inventory checkpoint")
            actual_inventory[key]=cp
    expected={(x["id"],x["update"]):x for x in plan["legacy_checkpoints"]}
    if set(actual_inventory)!=set(expected):raise ValueError("E0 execution inventory differs")
    for key,pin in expected.items():
        cp=actual_inventory[key]
        if any(cp.get(k)!=pin[k] for k in ("scope","checkpoint_relative_path","file_sha256","state_sha256")):
            raise ValueError("E0 selected checkpoint identity changed")
        if fingerprint(cp.get("expected_B"))!=pin["recorded_curve_sha256"]:
            raise ValueError("E0 recorded curve binding changed")
    for row in legacy["results"]:
        if row["status"] not in DONE_E0:continue
        pin=expected[(row["id"],row["update"])];p=inside(public,row["manifest_relative_path"])
        m=read(p);reference=read(p.with_name("REFERENCE_CONTRACT.json"));rescore=read(p.with_name("INDEPENDENT_RESCORE.json"))
        identity=m["identity"]
        if (identity["legacy_id"]!=pin["id"] or identity["checkpoint_update"]!=pin["update"]
                or identity["model_file_sha256"]!=pin["file_sha256"] or identity["model_state_sha256"]!=pin["state_sha256"]
                or identity["checkpoint_relative_path"]!=pin["checkpoint_relative_path"] or identity["model_path_scope"]!=pin["scope"]
                or identity["evaluation_source_sha256"]!=plan["source39"]["legacy_evaluation.py"]
                or identity["continuation_source_sha256"]!=plan["source39"]["legacy_continuation.py"]
                or identity["recorded_curve_sha256"]!=pin["recorded_curve_sha256"]
                or fingerprint(m.get("expected_B"))!=pin["recorded_curve_sha256"]
                or identity["source_hashes"]!=plan["legacy_source_hashes"]
                or identity["statistics_sha256"]!=plan["legacy_statistics_sha256"]
                or identity["foundation_sha256"]!=plan["legacy_foundation_sha256"]
                or m["split"]!="legacy_B_development" or reference["identity"]!=identity
                or fingerprint(reference)!=m["reference_sha256"] or reference.get("created_before_inference") is not True
                or reference["created_unix"]>m["inference_started_unix"]
                or rescore.get("status")!="RESCORE_COMPLETE" or rescore.get("independent_FP64_rescore") is not True
                or rescore["archive_sha256"]!=m["archive_sha256"] or rescore["reference_sha256"]!=m["reference_sha256"]
                or rescore["score"]!=m["score"] or row["score"]!=m["score"]):
            raise ValueError("Actual E0 inference/rescore metadata identity differs")
    import psutil
    if not proof.get("phase_owners") or not proof.get("prior_processes"):
        raise ValueError("Explicit old owner and process terminal evidence is required")
    if unique_records(proof["phase_owners"],["phase"])!={("T_final",),("E0_B",)}:
        raise ValueError("Both actual final phases must have owner evidence")
    expected_processes=set()
    for item in proof["phase_owners"]:
        expected_output=proof["T_batch"] if item["phase"]=="T_final" else proof["E0_B_registry"]
        if item["output_binding"]!=expected_output:raise ValueError("Owner receipt binds a different final phase output")
        safe=bound(public,item["owner_safe_fields"])
        closure_binding=item["central_closure"]
        p=inside(cache,closure_binding["cache_relative_path"])
        if sha(p)!=closure_binding["sha256"]:raise ValueError("Old owner closure changed")
        bounds=(min(T_reference_times),batch["created_unix"]) if item["phase"]=="T_final" else (legacy["started_unix"],legacy["finished_unix"])
        expected_processes.update(validate_phase_owner(safe,read(p),*bounds))
    if {(x["pid"],x["create_time"]) for x in proof["prior_processes"]}!=expected_processes:
        raise ValueError("Final owner/authorized child process coverage differs")
    for item in proof["prior_processes"]:
        try:
            if abs(psutil.Process(item["pid"]).create_time()-item["create_time"])<.001:
                raise ValueError("A previous final owner or child is alive")
        except psutil.NoSuchProcess:pass
    active=cache/"ACTIVE_GPU_LEASE.json"
    if active.exists() and (owner is None or read(active)["lease_id"]!=owner.lease_id):
        raise ValueError("Another GPU owner remains active")
    if time.time()>=plan["hard_wall_deadline_unix"]:raise ValueError("Original48h wall exhausted")
    return dict(status="PASS_POSTFINAL_METADATA_EVIDENCE",T_routes=46,legacy_checkpoint_attempts=55,
        **attempts,selection_sha256=lock["selection_sha256"],proof_sha256=expected_sha,
        proof_public_relative_path=proof_path.relative_to(public).as_posix(),metadata_does_not_replace_independent_numeric_audit=True)


def validate_active_owner(lease,pins,cache):
    if (lease is None or getattr(lease,"purpose",None)!="audit" or not getattr(lease,"_entered",False)
            or getattr(getattr(lease,"lock",None),"file",None) is None or getattr(lease,"pid",None)!=os.getpid()):
        raise ValueError("Same-PID active postfinal audit owner required")
    public=Path(os.environ["INFORMS_PUBLIC_ROOT"]).resolve()
    validate_evidence(public,cache,os.environ["INFORMS_POSTFINAL_PROOF_JSON"],
                      os.environ["INFORMS_POSTFINAL_PROOF_SHA256"],owner=lease)
    plan=preparation(public,cache);lease.check()
    from resources import gpu_usage_seconds
    closed=gpu_usage_seconds(cache)
    latest=lease.started_unix+max(0.,43200-closed)
    if (pins["owner_audit_cumulative_ceiling_seconds"]!=39600 or pins["global_GPU_ceiling_seconds"]!=43200
            or lease.total_gpu_seconds!=43200 or lease.reserve_seconds!=0
            or lease.wall_deadline>plan["hard_wall_deadline_unix"]
            or lease.deadline>min(latest,plan["hard_wall_deadline_unix"])+.001):
        raise ValueError("Explicit postfinal43200 phase or original39600 inputs differ")
    return closed
