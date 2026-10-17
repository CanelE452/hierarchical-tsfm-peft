"""ROOT-triggered stdlib audit of two saved-point refits and six V-only routes.

Default is zero I/O. Only public sufficient statistics/trials and small receipts
are decoded; selected artifacts/original checkpoints are opaque byte pins.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import types

NAME="priority_multik_resume_20261016_v1"
OLD="postT_V_only_cpu_multik_20261016_v1"
PLAN_SHA="886ee23f88dcc4520fda9a409f883193bd8199bc879bbd4f8f0b66f1884a21cc"
OLD_PLAN_SHA="2051573b8dee48200eb34e96e37e7af0d197442619ea1ba52a36a80ad4e27aec"
RUNNER_SHA="b2c2a6500ddd9bee54e8528e7c930558a58c06edb791b5777f6c0fb72807b4c9"
DISPATCH="priority_pause_validate_20261016_v1/FOLLOWUP_ROOT_v1/MULTIK_RESUME_DISPATCH_PREP_v1"
POST="priority_pause_validate_20261016_v1/FOLLOWUP_ROOT_v1/MULTIK_RESUME_POST_PREP_v1"
DISPATCH_SHA="61ee53b08b899602cbd7cc3a69f05e35552392ec9a0f5d46ff91ac714598e233"
POST_SHA="49d9e70e7f86b3295f593e6746873cb39fe04c0fa3186731fa3511fbf841b101"
HELPERS={
    "arithmetic":("priority_valid_minima_20261016_v1/independent_beijing_audit.py","933883a052b02e6a41ccdadaa5ec45e80b977dbfaada007607ac1093f518dd04"),
    "inputs":("priority_valid_minima_20261016_v1/build_beijing_report.py","8bb730a55e0f0f924de0616bf917ca4b47e4117503027cfc4656c6ef502750dd"),
    "guard":("priority_valid_family_20261016_v1/audit_initial_family.py","f92a3a7b4bc03677f67d3bb7c42758765a95f49a378e2dd9d768ffc7f4de6241")}
GUARD_REL="postT_cpu_lag_preparation_20261016_v1/cpu_group_supervisor.py"
GUARD_SHA="82a837ea0cfee8b52765f4ac1c7325d564e886c9e94857bd0f957e4a40c75f8b"
F_SHA="c2b4edf7db22d3f18e58ca79299a5c7e864e9b8f277e4e29e2f88d07dbb35874"
DATA_SHA="b39023bcae04736eff9a0fd9a3d56e10dc7f32bd668a53d681290cb9193d8269"
SCALER="3348bcf1fa76d382f23cf20597fa90de6e21dac587d4ca9f6883f235a10f93fb"
SEARCH_SCALER="cb394e93fec460e933a56c46cfc43645ba787fd765f064668788e33d9d55bb9d"
ORIGINS=tuple(range(14726,16743,48))
STAGES=tuple("stage_"+x+"_seconds" for x in ("CPU_preprocess","host_to_device","P_E","F","D","inverse_and_CPU"))
DEADLINE=1792222550


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):h.update(block)
    return h.hexdigest()


def digest(value):return isinstance(value,str) and len(value)==64 and all(c in "0123456789abcdef" for c in value)


def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def parents():
    return [dict(kind=kind,run_id="postT_cpu_beijing_E4_"+role+"_k11_s92601_v1",id=tag,grid=grid)
        for kind,role,tag,grid in (("shared","ORDERED_MULTI_K","multik_resume_ordered_parent_root_v1",[3,6,11]),
            ("max_only","MAXK_PCA_TRAINED","multik_resume_maxk_parent_root_v1",[11]))]


def outputs():
    result=[]
    for parent in parents():
        for k in (3,6,11):
            role=("ORDERED_MULTI_K" if parent["kind"]=="shared" else "MAXK_PCA_TRAINED") if k==11 else ("SHARED_PREFIX" if parent["kind"]=="shared" else "POSTHOC_PREFIX")
            result.append(dict(run_id="postT_cpu_beijing_E4_"+role+"_k"+str(k)+"_s92601_v1",parent_run_id=parent["run_id"],
                kind=parent["kind"],latent_k=k,split="V_select",variant="shared" if parent["kind"]=="shared" else ("max_only" if k==11 else "posthoc")))
    return result


def expected_steps():
    return {"multik_post_"+r["kind"]+"_k"+str(r["latent_k"])+"_"+mode+("_b"+str(batch) if batch else "")+"_root_v1":
        dict(run_id=r["run_id"],mode=mode,batch=batch) for r in outputs()
        for mode,batch in (("predict",None),("rescore",None),("replay",None),("cost",1),("cost",8))}


def validate_post(post):
    expected=expected_steps();rows=post.get("steps",[])
    if (post.get("status")!="COMPLETE_SIX_PREFIXES_TWELVE_VALIDATIONS_TWELVE_COSTS" or post.get("skipped")!=[] or post.get("error") is not None
            or len(rows)!=30 or len({r.get("id") for r in rows})!=30 or {r.get("id") for r in rows}!=set(expected)
            or any(r.get("success") is not True or any(r.get(k)!=v for k,v in expected[r["id"]].items()) for r in rows)):
        raise ValueError("Exact thirty successful actual POST steps required; partial is NOT_COMPLETE")


def require_argv(command,flag,want):
    if command.count(flag)!=1 or command.index(flag)+1>=len(command) or command[command.index(flag)+1]!=want:
        raise ValueError("Actual unique argv binding differs: "+flag)


def validate_deadline(command,recorded,resume):
    if command.count("--deadline")!=1 or command.index("--deadline")+1>=len(command):raise ValueError("One actual original deadline argv required")
    actual=float(command[command.index("--deadline")+1])
    if not math.isfinite(actual) or actual!=recorded or not 0<actual<=DEADLINE or (not resume and actual!=DEADLINE):
        raise ValueError("Actual argv/STARTED/original worker deadline differs")


def validate_parent(v,search,refit,condition):
    want=dict(status="COMPLETE_VALID",run_id=condition["run_id"],family="B0",seed=92601,epochs=[.25,1.,3.],selected_epoch=3.,
        training_status="UNDERTRAINED_OR_BUDGET_LIMIT",full_V_completed=False,actual_saved_point_resume=True,
        search_updates=1868,search_exposures=29880,refit_updates=2262,refit_exposures=36189,search_fit_end=10519,refit_fit_end=12622,
        search_train_origins=9960,refit_train_origins=12063,search_new_optimizer_updates=0,search_new_F_calls=0,refit_scaler_sha=SCALER,
        source_stage_API="original_fit_base._stage",old_unsaved_interrupted_progress="UNKNOWN_NOT_LOGGED",search_total_wall_seconds_known=False,
        full_interrupted_instant_resume=False,uninterrupted_bitwise_parity_claimed=False,independent_K_training_completed=False,new_T_target_operations=0)
    if any(v.get(k)!=x for k,x in want.items()) or not all(digest(v.get(k)) for k in ("artifact_sha256","model_state_sha")):
        raise ValueError("Exact saved-point parent/no-new-search/undertrained/no-V receipt required")
    for stage,expected in ((search,dict(status="SEGMENT_COMPLETE",selected_epoch=3.,completed_epochs=[0.,.25,1.,3.],train_origins=9960,
            updates=1868,exposures=29880,F_session_logical_streams=0,F_session_physical_streams=0)),
            (refit,dict(status="SEGMENT_COMPLETE",selected_epoch=3.,completed_epochs=[0.,3.],train_origins=12063,updates=2262,exposures=36189))):
        if any(stage.get(k)!=x for k,x in expected.items()):raise ValueError("Original search reconstruction and selected three-epoch refit activity differ")
    if search.get("contract",{}).get("scaler_sha")!=SEARCH_SCALER or refit.get("contract",{}).get("scaler_sha")!=SCALER:
        raise ValueError("Original A50/A60 scaler lineage differs")
    for key in ("search_seconds_this_resume_call","refit_seconds_this_resume_call","historical_saved_search_segment_seconds"):
        if not math.isfinite(float(v[key])) or v[key]<0:raise ValueError("Separately recorded saved/new timing required")


def route_spec(output,receipt):
    return dict(run_id=output["run_id"],source_parent_run_id=output["parent_run_id"],dataset="beijing",family="B0",fold="primary",canonical_fit_end=12622,
        latent_k=output["latent_k"],prefix_variant=output["variant"],artifact_cache_relative_path="runs/"+output["parent_run_id"]+"/selected_bundle.pt",
        artifact_sha256=receipt["artifact_sha256"],foundation_sha=F_SHA,
        expected_prediction_model_sha=fingerprint(dict(artifact_sha=receipt["artifact_sha256"],model_state_sha=receipt["model_state_sha"],normalization=SCALER)))


def validate_validation(v,mode,row,artifact,manifest):
    if v.get("run_id")!=row["run_id"] or v.get("artifact_sha256")!=artifact or v.get("manifest_sha256")!=manifest:
        raise ValueError("Exact alias validation selected parent/artifact/manifest required")
    if mode=="rescore":
        if (v.get("status")!="PASS_INDEPENDENT_FULL_V_FP64" or v.get("full_V_origins")!=43 or v.get("n_observed")!=20539
                or v.get("public_stats_rows")!=1419 or v.get("saved_predictions_rescored") is not True or v.get("new_model_inference") is not False
                or v.get("torch_imported") is not False or v.get("protected_T_target_operations")!=0):
            raise ValueError("Actual independent full saved-V FP64 rescore required")
    elif mode=="replay":
        oo=v.get("origins",[]);k=row["latent_k"]
        if (v.get("status")!="PASS_FRESH_CPU6_REPLAY" or v.get("executed_now") is not True or v.get("microbatches")!=[4,2]
                or len(oo)!=6 or len(set(oo))!=6 or not set(oo).issubset(ORIGINS) or not {ORIGINS[0],ORIGINS[21],ORIGINS[-1]}.issubset(oo)
                or v.get("F_counters")!=dict(streams=6*k,physical_streams=6*k,row_calls=2,physical_calls=2)
                or v.get("atol")!=1e-5 or v.get("rtol")!=1e-4 or v.get("new_T_target_operations")!=0):
            raise ValueError("Actual CPU6/4+2 original alias Route replay required")
    else:raise ValueError("Only actual rescore/replay validations")


def validate_cost(v,row,batch,artifact,manifest,quiet_sha):
    want=dict(status="COMPLETE_PROTOCOL",run_id=row["run_id"],artifact_sha256=artifact,manifest_sha256=manifest,device="cpu",CPU_threads=2,
        batch_size=batch,warmups=10,timed_trials=20,stage_trials=20,actual_trial_rows=50,quiet_window_proof_sha256=quiet_sha,
        cached_Z_or_forecast_reads=0,training_peak_included=False,new_T_target_operations=0,
        source_protocol="original_measure_costs.measure_balanced + original deployment.Route constructor",stage_probe_separate_intrusive_scope=True,
        origins=list(ORIGINS[:batch]))
    if batch not in (1,8) or any(v.get(k)!=x for k,x in want.items()) or not all(digest(v.get(k)) for k in ("trials_sha256","physical_input_sha256","observation_mask_sha256")):
        raise ValueError("Native N7 whole physical CPU2/10+20+20 protocol/input/quiet differs")
    memory=v.get("memory") or {}
    if memory.get("foundation_instances_created")!=1 or not math.isfinite(float(memory.get("resident_weights_cpu_MiB",math.nan))) or memory["resident_weights_cpu_MiB"]<=0:
        raise ValueError("Actual original single-foundation resident-memory receipt required")


def aggregate_trials(rows,identifier,batch,k):
    if batch not in (1,8) or k not in (3,6,11):raise ValueError("Only original six N7 B0 prefix routes")
    phases=dict(warmup=10,timed=20,stage_probe=20);expected={(p,i) for p,n in phases.items() for i in range(n)};seen={};hashes=set();totals=[0,0,0]
    fields={"run_id","phase","trial","order_position","batch_size","whole_seconds","F_logical_streams","F_physical_streams","F_calls","prediction_sha256","row_microbatch","stage_instrumentation",*STAGES}
    for row in rows:
        key=(row.get("phase"),int(row.get("trial",-1)));instrumented=key[0]=="stage_probe"
        if (set(row)!=fields or key not in expected or key in seen or row["run_id"]!=identifier or int(row["batch_size"])!=batch
                or int(row["order_position"])!=0 or int(row["row_microbatch"])!=128 or row["stage_instrumentation"]!=str(instrumented)
                or not digest(row["prediction_sha256"])):raise ValueError("Exact original B0 six-stage fifty-trial fields/keys differ")
        counters=tuple(int(row[key]) for key in ("F_logical_streams","F_physical_streams","F_calls"))
        if counters!=(batch*k,batch*k,1):raise ValueError("Actual saved trial K-row stream/call accounting differs")
        value=float(row["whole_seconds"])
        if not math.isfinite(value) or value<=0:raise ValueError("Finite positive whole route time required")
        for field in STAGES:
            if instrumented:
                number=float(row[field])
                if not math.isfinite(number) or number<0:raise ValueError("Finite nonnegative recorded stage timing required")
            elif row[field]!="":raise ValueError("Ordinary whole trial cannot contain stage timings")
        seen[key]=value;hashes.add(row["prediction_sha256"]);totals=[a+b for a,b in zip(totals,counters)]
    if set(seen)!=expected:raise ValueError("Complete10warm/20timed/20stage required")
    summaries=[]
    for phase in ("timed","stage_probe"):
        values=sorted(v for key,v in seen.items() if key[0]==phase);position=(len(values)-1)*.9;low=math.floor(position);fraction=position-low
        summaries.append(dict(run_id=identifier,phase=phase,batch_size=batch,trials=20,median_seconds=statistics.median(values),mean_seconds=math.fsum(values)/len(values),
            min_seconds=values[0],max_seconds=values[-1],p90_seconds=values[low]+(values[low+1]-values[low])*fraction,
            stage_instrumentation=phase=="stage_probe",stage_sum_equals_whole_claimed=False))
    return dict(trial_rows=len(seen),phase_rows=phases,F_logical_streams=totals[0],F_physical_streams=totals[1],F_calls=totals[2],unique_prediction_hashes=len(hashes),summaries=summaries)


def load_helpers(public):
    modules={}
    for key,(name,expected) in HELPERS.items():
        path=Path(public)/name
        if sha(path)!=expected:raise ValueError("Pinned stdlib audit helper changed")
        m=types.ModuleType("n7_pinned_"+key);m.__file__=str(path);exec(compile(path.read_text(encoding="utf-8"),str(path),"exec"),m.__dict__);modules[key]=m
    return modules


def collect(inputs,helpers,bindings):
    h,b,g=helpers["arithmetic"],helpers["inputs"],helpers["guard"]
    for key,(path,digest_value) in HELPERS.items():inputs.capture("public",path,digest_value)
    inputs.capture("public",NAME+"/multik_resume_runner.py",RUNNER_SHA)
    inputs.capture("cache",DISPATCH+"/RUN_ROOT_DRAFT_v2.ps1",DISPATCH_SHA);inputs.capture("cache",POST+"/RUN_ROOT_DRAFT_v2.ps1",POST_SHA)
    inputs.capture("cache",GUARD_REL,GUARD_SHA)
    plan=inputs.json("public",NAME+"/PLAN.json",bindings["plan_sha"]);old=inputs.json("public",OLD+"/PLAN.json",OLD_PLAN_SHA)
    if (bindings["plan_sha"]!=PLAN_SHA or plan.get("status")!="SEALED_BY_ROOT" or plan.get("runner_sha256")!=RUNNER_SHA or plan.get("outputs")!=outputs()
            or plan.get("protected_T_numeric_allowed") is not False or plan.get("search_retraining_allowed") is not False or plan.get("deadline_unix")!=DEADLINE
            or len(plan.get("conditions",[]))!=2 or len(old["source39"])!=39 or len(old["immutable_public"])!=9
            or len(set(old["source39"])|set(old["immutable_public"]))!=48):raise ValueError("Exact sealed original two-parent/six-output/source48/T-lock contract required")
    for path,digest_value in {**old["source39"],**old["immutable_public"],**plan["dependencies"]}.items():inputs.capture("public",path,digest_value)
    for path,digest_value in plan["cache_metadata"].items():inputs.capture("cache",path,digest_value)
    checkpoint_count=0
    for condition,parent in zip(plan["conditions"],parents()):
        if condition["run_id"]!=parent["run_id"] or condition["kind"]!=parent["kind"] or condition["spec"]["k_grid"]!=parent["grid"]:
            raise ValueError("Original ordered/max-only PCA grid identity differs")
        for row in condition["checkpoints"]:
            for field in ("cache_relative_path","backup_cache_relative_path"):
                path=inputs.capture("cache",row[field],row["sha256"])
                if path.stat().st_size!=row["bytes"]:raise ValueError("Original/backup opaque checkpoint byte count differs")
                checkpoint_count+=1
    if checkpoint_count!=20:raise ValueError("Exactly twenty original/backup opaque checkpoint pins required")
    dispatch=inputs.json("cache",DISPATCH+"/RESULT.json",bindings["dispatch_sha"])
    post=inputs.json("cache",POST+"/RESULT.json",bindings["post_sha"]);validate_post(post)
    for obj in (dispatch,post):
        if obj.get("pins_before_after_unchanged") is not True or obj.get("changed_or_unreadable_pinned_paths")!=[] or obj.get("error") is not None or obj.get("plan_sha256")!=PLAN_SHA:
            raise ValueError("Actual terminal source/PLAN closure required")
    if (dispatch.get("status")!="COMPLETE_TWO_SAVED_POINT_REFITS_NO_V_YET" or dispatch.get("strong_observed_tree_closure_verified") is not True
            or dispatch.get("source39_immutable9_count")!=48 or dispatch.get("original_backup_checkpoint_byte_pins")!=20 or dispatch.get("full_V_completed") is not False
            or post.get("dispatch_result_sha256")!=bindings["dispatch_sha"] or post.get("dispatch_SPEC_sha256")!=bindings["dispatch_spec_sha"]
            or post.get("dispatch_guard_sha256")!=bindings["dispatch_guard_sha"] or post.get("parent_BASE_full_V_completed_unchanged_false") is not True
            or post.get("original_source39_immutable9_count")!=48 or post.get("original_backup_checkpoint_byte_pins")!=20 or post.get("new_training")!=0
            or post.get("whole_MASTER_complete") is not False or post.get("original_worker_deadline")!=DEADLINE or post.get("original_campaign_deadline")!=1792223150):
        raise ValueError("Two saved refits remain no-V while exact thirty-step POST completes separately")
    started=inputs.json("cache",DISPATCH+"/STARTED.json")
    if started.get("script_sha256")!=DISPATCH_SHA or started.get("plan_sha256")!=PLAN_SHA or started.get("full_V_completed") is not False:
        raise ValueError("Actual two-parent dispatch STARTED differs")
    post_started=inputs.json("cache",POST+"/STARTED.json")
    if any(post_started.get(k)!=v for k,v in dict(script_sha256=POST_SHA,dispatch_result_sha256=bindings["dispatch_sha"],dispatch_SPEC_sha256=bindings["dispatch_spec_sha"],dispatch_guard_sha256=bindings["dispatch_guard_sha"],plan_sha256=PLAN_SHA,new_training=0,parent_BASE_full_V_completed_unchanged_false=True).items()):
        raise ValueError("Actual POST STARTED does not bind exact resume execution")
    spec=inputs.json("cache",DISPATCH+"/SPEC_ROOT_v1.json",bindings["dispatch_spec_sha"])
    guard=inputs.json("cache",DISPATCH+"/GUARD_ROOT_v1/RESULT.json",bindings["dispatch_guard_sha"])
    if len(spec.get("processes",[]))!=2 or len(dispatch.get("conditions",[]))!=2:raise ValueError("Exact two actual resume children required")
    reference=inputs.json("public",plan["V_reference"]["relative_path"],plan["V_reference"]["sha256"]);b.validate_cohort(reference)
    channels=old["task_contract"]["columns"]
    if len(channels)!=11 or len(set(channels))!=11:raise ValueError("Exact eleven original channels required")
    steps={r["id"]:r for r in post["steps"]};records=[];cost_intervals=[];other_intervals=[];parent_rows=[];accuracy=[];costs=[];rows=[];physical={};parent_by_id={};replay_origins=None

    def bound(tag,identifier,mode,status,record,resume=False):
        folder=NAME+"/diagnostics/"+tag
        outer=inputs.json("public",folder+"/RESULT.json",record["outer_sha256"]);runtime=inputs.json("public",folder+"/STARTED.json")
        want=dict(status=status,run_id=identifier,mode=mode,plan_sha256=PLAN_SHA,error_class=None,postflight_error_class=None,
            original_source39_immutable9_original_and_backup_checkpoints_preserved=True,new_T_target_operations=0,
            original_full_task_identity_inherited_not_recomputed=True,independent_generalization_claimed=False)
        if any(outer.get(k)!=v for k,v in want.items()) or any(runtime.get(k)!=v for k,v in dict(run_id=identifier,mode=mode,plan_sha256=PLAN_SHA).items()):
            raise ValueError("Exact actual alias/parent outer/runtime/source/T closure required")
        if resume:sp,gp,actual_spec,actual_guard=DISPATCH+"/SPEC_ROOT_v1.json",DISPATCH+"/GUARD_ROOT_v1/RESULT.json",spec,guard
        else:
            sp,gp=POST+"/SPEC_"+tag+".json",NAME+"/supervisors/"+tag+"/RESULT.json"
            actual_spec=inputs.json("cache",sp,record["spec_sha256"]);actual_guard=inputs.json("cache",gp,record["guard_sha256"])
            if record.get("strong_observed_tree_closure_verified") is not True or record.get("guard_exit_code")!=0:raise ValueError("Actual single POST guard must be cleanly closed")
        matches=[p for p in actual_spec["processes"] if p["id"]==tag]
        if len(matches)!=1:raise ValueError("Exactly one original actual argv for selected child")
        command=matches[0]["command"]
        for flag,value in (("--run-id",identifier),("--mode",mode),("--plan-sha",PLAN_SHA)):require_argv(command,flag,value)
        for flag,path in (("--public-root",inputs.roots["public"]),("--cache-root",inputs.roots["cache"]),("--output",inputs.roots["public"]/folder)):
            if command.count(flag)!=1 or Path(command[command.index(flag)+1]).resolve()!=path.resolve():raise ValueError("Exact actual public/cache/output argv paths differ")
        if sum(Path(v).resolve()==(inputs.roots["public"]/NAME/"multik_resume_runner.py").resolve() for v in command)!=1:
            raise ValueError("Exact frozen original N7 entry required")
        validate_deadline(command,runtime["deadline_unix"],resume)
        selected=g.validate_guard(actual_guard,actual_spec,inputs.files[("cache",sp)],tag,command,runtime)
        if not resume and (record.get("outer_status")!=status or record.get("worker_returncode")!=0 or record.get("worker_stop_reason") is not None):raise ValueError("Actual POST child/status metadata differs")
        if mode!="rescore":
            access=inputs.json("public",folder+"/PREFIX_ACCESS_RECEIPT.json")
            if (access.get("new_T_numeric_values_parsed")!=0 or access.get("materialized_rows")!=16830 or access.get("inherited_identity_status")!="INHERITED_NOT_RECOMPUTED"
                    or access.get("inherited_full_task_data_sha256")!=DATA_SHA):raise ValueError("Actual prefix access must preserve inherited identity and zero new T numeric values")
        interval=dict(id=tag,start=actual_guard["start_unix"],end=actual_guard["end_unix"])
        (cost_intervals if mode=="cost" else other_intervals).append(interval)
        record_meta=dict(id=tag,guard_cache_relative_path=gp,guard_sha256=inputs.files[("cache",gp)],spec_sha256=inputs.files[("cache",sp)],
            outer_sha256=inputs.files[("public",folder+"/RESULT.json")],STARTED_sha256=inputs.files[("public",folder+"/STARTED.json")],
            sampled_peak_RSS_bytes=selected["sampled_peak_RSS_bytes"],observed_owned_identities=len(selected["owned_processes"]),observed_tree_join_confirmed=True,current_PID_query_performed=False)
        records.append(record_meta)
        return folder,outer,command,record_meta

    for parent,condition in zip(parents(),plan["conditions"]):
        run=NAME+"/runs/"+parent["run_id"];private=NAME+"/runs/"+parent["run_id"]
        recorded=[r for r in dispatch["conditions"] if r.get("run_id")==parent["run_id"]]
        if len(recorded)!=1 or recorded[0].get("id")!=parent["id"] or recorded[0].get("parent_refit_complete") is not True or recorded[0].get("full_V_completed") is not False:
            raise ValueError("Each exact saved-point refit must actually complete; partial is not PASS")
        record=recorded[0];receipt=inputs.json("public",run+"/BASE_FIT_RECEIPT.json",record["base_receipt_sha256"])
        search=inputs.json("public",run+"/search/RECEIPT.json");refit=inputs.json("public",run+"/refit/RECEIPT.json");validate_parent(receipt,search,refit,parent)
        for stage,label in ((search,"search"),(refit,"refit")):
            if stage["contract"]!=condition["checkpoint_contracts"][label]:raise ValueError("Original source/config/scaler/init stage contract changed")
        source=inputs.json("public",run+"/SOURCE_CONTRACT.json",condition["source_contract_sha256"])
        inputs.capture("public",condition["source_contract_public_relative_path"],condition["source_contract_sha256"])
        original_contract=inputs.json("cache",condition["run_contract_cache_relative_path"],condition["run_contract_sha256"])
        copied_contract=inputs.json("cache",private+"/RUN_CONTRACT.json",condition["run_contract_sha256"])
        if source.get("spec")!=condition["spec"] or copied_contract!=original_contract or any(original_contract.get(k)!=v for k,v in condition["core_contract"].items()):
            raise ValueError("Original complete parent spec/source/run contracts differ")
        artifact=inputs.capture("cache",private+"/selected_bundle.pt",receipt["artifact_sha256"])
        if receipt["artifact_cache_relative_path"]!="runs/"+parent["run_id"]+"/selected_bundle.pt" or record.get("artifact_sha256")!=receipt["artifact_sha256"]:
            raise ValueError("Exact selected parent opaque artifact differs")
        provenance=inputs.json("public",run+"/RESUME_PROVENANCE.json",record["resume_provenance_sha256"])
        if (provenance.get("plan_sha256")!=PLAN_SHA or provenance.get("adapter_sha256")!=RUNNER_SHA or provenance.get("original_run_id")!=parent["run_id"]
                or provenance.get("parent_checkpoints")!=condition["checkpoints"] or provenance.get("original_and_backup_bytes_preserved") is not True
                or provenance.get("checkpoint_contract_modified") is not False or provenance.get("resume_saved_point")!="refit/checkpoint_000000.pt"
                or provenance.get("contracts_run_ID_stage_label_spec_provenance_unchanged") is not True or provenance.get("interrupted_unsaved_work_recovered") is not False):
            raise ValueError("Exact old checkpoint lineage and saved-point-only provenance required")
        _,outer,_,_=bound(parent["id"],parent["run_id"],"resume","COMPLETE_SAVED_POINT_REFIT_NO_V_YET",record,True)
        science=outer["scientific_result"]
        if science.get("fit_receipt")!=receipt or science.get("full_V_completed") is not False or science.get("F_final_state_sha256")!=F_SHA:
            raise ValueError("Actual parent output must retain BASE false/no-V and restored frozen F")
        if receipt["search_seconds_this_resume_call"]!=search["seconds"] or receipt["refit_seconds_this_resume_call"]!=refit["seconds"]:
            raise ValueError("New reconstruction/refit timing differs from actual stage receipt")
        curves=inputs.csv("public",run+"/search/CURVE.csv")
        if math.fsum(float(r.get("segment_seconds") or 0) for r in curves)!=receipt["historical_saved_search_segment_seconds"]:
            raise ValueError("Historical saved-segment time differs; do not call reconstruction overhead original training time")
        parent_rows.append(dict(receipt,kind=parent["kind"],original_grid=parent["grid"],stage_F_counters={label:{key:stage.get(key) for key in
            ("F_session_logical_streams","F_session_physical_streams","F_total_logical_streams","F_total_physical_streams")} for label,stage in (("search",search),("refit",refit))}))
        parent_by_id[parent["run_id"]]=receipt

    for output in outputs():
        identifier=output["run_id"];receipt=parent_by_id[output["parent_run_id"]];artifact=receipt["artifact_sha256"]
        base="multik_post_"+output["kind"]+"_k"+str(output["latent_k"])+"_"
        prediction=NAME+"/predictions/"+identifier+"/V_select"
        manifest=inputs.json("public",prediction+"/PREDICTION_MANIFEST.json");b.validate_cohort(manifest)
        manifest_sha=inputs.files[("public",prediction+"/PREDICTION_MANIFEST.json")];canonical=route_spec(output,receipt)
        expected_model=fingerprint(dict(artifact_sha=artifact,model_state_sha=receipt["model_state_sha"],latent_k=output["latent_k"],variant=output["variant"]))
        if (any(manifest.get(k)!=v for k,v in dict(run_id=identifier,parent_run_id=output["parent_run_id"],registered_role=output["kind"],variant=output["variant"],latent_k=output["latent_k"],family="B0",device="cpu",seed=92601,artifact_sha256=artifact,new_T_target_operations=0).items())
                or manifest.get("actual_route_spec")!=canonical or manifest.get("model_sha")!=expected_model or manifest.get("config_sha")!=fingerprint(canonical)
                or manifest.get("source_sha")!=fingerprint(dict(adapter=RUNNER_SHA,**plan["dependencies"]))
                or manifest.get("F_counters")!=dict(streams=43*output["latent_k"],row_calls=11,physical_streams=43*output["latent_k"],physical_calls=11)
                or manifest.get("arrays",{}).get("pred_normalized",{}).get("shape")!=[43,48,11] or manifest["arrays"]["pred_normalized"].get("dtype")!="float32"):
            raise ValueError("Actual parent/prefix model/spec/full CPU V43 F accounting differs")
        for key in ("target_normalized","observed","origins","channel_ids"):
            if manifest["arrays"][key]!=reference["arrays"][key]:raise ValueError("Exact shared V fingerprints differ")
        reference_contract=inputs.json("public",prediction+"/REFERENCE_CONTRACT.json")
        if any(reference_contract.get(k)!=manifest.get(k) for k in ("run_id","split","cohort","source_sha","config_sha","data_sha","model_sha","scaler_sha")):
            raise ValueError("Saved alias reference identity differs")
        if reference_contract.get("arrays")!={k:manifest["arrays"][k] for k in ("target_normalized","observed","origins","channel_ids")}:
            raise ValueError("Saved alias reference target/mask/origin/channel fingerprints differ")
        archive_path=NAME+"/predictions/"+identifier+"/V_select.npz"
        inputs.capture("cache",archive_path,manifest["archive_sha256"])
        tag=base+"predict_root_v1";_,pred,_,_=bound(tag,identifier,"predict","COMPLETE_FULL_V43",steps[tag])
        if pred["scientific_result"].get("manifest_sha256")!=manifest_sha or pred["scientific_result"].get("V_score")!=manifest["score"] or pred["scientific_result"].get("F_final_state_sha256")!=F_SHA:
            raise ValueError("Actual predict outer/fullV/frozen F binding differs")
        proofs={};rescore=None
        for mode,status in (("rescore","PASS_INDEPENDENT_FULL_V_FP64"),("replay","PASS_FRESH_CPU6_REPLAY")):
            tag=base+mode+"_root_v1";folder,outer,_,_=bound(tag,identifier,mode,status,steps[tag]);science=outer["scientific_result"]
            validate_validation(science,mode,output,artifact,manifest_sha)
            saved=inputs.json("public",folder+("/RESCORE_RECEIPT.json" if mode=="rescore" else "/REPLAY_RECEIPT.json"))
            if any(science.get(k)!=v for k,v in saved.items()):raise ValueError("Native validation receipt and outer differ")
            if mode=="rescore":rescore=science
            else:
                if science.get("F_final_state_sha256")!=F_SHA:raise ValueError("Fresh replay did not preserve original frozen F")
                actual_origins=tuple(science["origins"])
                if replay_origins is not None and replay_origins!=actual_origins:raise ValueError("Actual common CPU6 origin order differs across aliases")
                if post.get("actual_common_CPU6_origins")!="|".join(map(str,actual_origins)):raise ValueError("POST common CPU6 origin binding differs")
                replay_origins=actual_origins
            proofs[mode]=dict(path=inputs.roots["public"]/folder/"RESULT.json",sha256=inputs.files[("public",folder+"/RESULT.json")])
        stats=h.gzip_rows(inputs,"public",prediction+"/PUBLIC_ERROR_STATS.csv.gz",1419)
        aggregate=h.aggregate_statistics(stats,identifier,channels,ORIGINS,((0,12),(12,24),(24,48)),20539)
        score_rows=inputs.csv("public",prediction+"/SCORES.csv")
        if len(score_rows)!=1:raise ValueError("One exact original score row required")
        if any(score_rows[0].get(k)!=manifest.get(k) for k in ("run_id","split","cohort")):raise ValueError("Original SCORES alias/cohort identity differs")
        delta=h.compare_accuracy(aggregate,manifest,score_rows[0],rescore)
        accuracy.append(dict(output,**aggregate,metric_deltas=delta,artifact_sha256=artifact,manifest_sha256=manifest_sha,accuracy_reused=False))
        row=dict(output,mse=aggregate["mse"],mae=aggregate["mae"],artifact_sha256=artifact,manifest_sha256=manifest_sha,accuracy_reused=False,cost_reused=False)
        for batch in (1,8):
            tag=base+"cost_b"+str(batch)+"_root_v1";folder,outer,command,guard_record=bound(tag,identifier,"cost","COMPLETE_PROTOCOL",steps[tag])
            cost=inputs.json("public",folder+"/COST_RECEIPT.json")
            if any(outer["scientific_result"].get(k)!=v for k,v in cost.items()) or outer["scientific_result"].get("F_final_state_sha256")!=F_SHA:
                raise ValueError("Actual native COST_RECEIPT/outer/frozen state differs")
            for mode,proof in proofs.items():
                require_argv(command,"--"+mode+"-sha",proof["sha256"])
                if command.count("--"+mode+"-proof")!=1 or Path(command[command.index("--"+mode+"-proof")+1]).resolve()!=proof["path"].resolve():raise ValueError("Actual cost argv must bind exact alias validation proofs")
            quiet_path=POST+"/QUIET_"+tag+".json";quiet=inputs.json("cache",quiet_path,cost["quiet_window_proof_sha256"])
            require_argv(command,"--quiet-sha",cost["quiet_window_proof_sha256"])
            if command.count("--quiet-proof")!=1 or Path(command[command.index("--quiet-proof")+1]).resolve()!=inputs.path("cache",quiet_path).resolve():raise ValueError("Actual argv quiet path differs")
            runtime=inputs.json("public",folder+"/STARTED.json")
            actual_guard=inputs.json("cache",NAME+"/supervisors/"+tag+"/RESULT.json",steps[tag]["guard_sha256"])
            if (quiet.get("status")!="QUIET_CPU_COST_WINDOW" or quiet.get("active_model_workers")!=[] or quiet.get("active_cost_workers")!=[]
                    or not 0<=actual_guard["start_unix"]-quiet["captured_unix"]<=120 or quiet.get("run_id")!=identifier
                    or quiet.get("artifact_sha256")!=artifact or quiet.get("manifest_sha256")!=manifest_sha or quiet.get("plan_sha256")!=PLAN_SHA
                    or quiet.get("dispatch_result_sha256")!=bindings["dispatch_sha"]):raise ValueError("Fresh matching actual quiet/source/proof window required")
            validate_cost(cost,output,batch,artifact,manifest_sha,cost["quiet_window_proof_sha256"])
            trial_rows=h.gzip_rows(inputs,"public",folder+"/COST_TRIALS.csv.gz",50,cost["trials_sha256"])
            summed=aggregate_trials(trial_rows,identifier,batch,output["latent_k"]);difference=h.compare_summaries(summed["summaries"],cost["summaries"])
            input_key=(cost["physical_input_sha256"],cost["observation_mask_sha256"],tuple(cost["origins"]))
            if batch in physical and physical[batch]!=input_key:raise ValueError("Same-batch physical input/mask/origins differ across aliases")
            physical[batch]=input_key
            cost_row=dict(run_id=identifier,parent_run_id=output["parent_run_id"],kind=output["kind"],latent_k=output["latent_k"],batch=batch,
                public_diagnostic_relative_path=folder,**summed,max_summary_difference_seconds=difference,memory_scope=cost.get("memory"),
                guard_sampled_peak_RSS_bytes=guard_record["sampled_peak_RSS_bytes"],physical_input_sha256=cost["physical_input_sha256"],observation_mask_sha256=cost["observation_mask_sha256"],
                quiet_proof_sha256=cost["quiet_window_proof_sha256"],validation_proof_SHA_from_actual_argv={mode:proof["sha256"] for mode,proof in proofs.items()},
                derived_trial_F_counters=dict(F_logical_streams=summed["F_logical_streams"],F_physical_streams=summed["F_physical_streams"],F_calls=summed["F_calls"],scope="INDEPENDENTLY_SUMMED_FROM_50_PUBLIC_TRIAL_ROWS"),
                raw_receipt_total_F_counters=None,receipt_validation_proof_SHA_map=None,physical_output_dtype_from_receipt=None,raw_receipt_field_scope="NOT_RECORDED_BY_ORIGINAL_N7_RECEIPT")
            row["cost_B"+str(batch)]=cost_row;costs.append(cost_row)
        rows.append(row)
    h.require_same_counts(accuracy);h.validate_intervals(cost_intervals,other_intervals)
    for row in accuracy:row["band_count_binding_sha256"]=fingerprint(row.pop("count_by_key"))
    return dict(parents=parent_rows,accuracy=accuracy,cost_conditions=costs,rows=rows,reference_rows=[],recorded_guards=records,
        recorded_cost_intervals=cost_intervals,recorded_costs_disjoint_from_all_recorded_model_validation_cost_intervals=True,
        totals=dict(parent_refits=2,new_search_optimizer_updates=0,new_search_F_calls=0,alias_accuracy_conditions=6,public_stat_rows=8514,
            observed_labels_per_alias=20539,unique_V_observed_labels=20539,cost_conditions=12,trial_rows=600,summary_groups=24,summary_numeric_fields_checked=120,
            derived_trial_F_logical_streams=sum(r["F_logical_streams"] for r in costs),derived_trial_F_calls=sum(r["F_calls"] for r in costs)))


def execute(public,cache,bindings,audit_sha,output):
    output=Path(output).absolute()
    if output.exists():raise FileExistsError("Existing audit output preserved before any input read")
    public,cache=Path(public).resolve(),Path(cache).resolve()
    if not output.resolve().is_relative_to(public/NAME) or output.suffix!=".json":raise ValueError("One new JSON inside original N7 namespace required")
    required={"plan_sha","dispatch_sha","dispatch_spec_sha","dispatch_guard_sha","post_sha"}
    if set(bindings)!=required or not all(digest(v) for v in bindings.values()) or bindings["plan_sha"]!=PLAN_SHA or not digest(audit_sha):
        raise ValueError("Exact ROOT plan/future actual dispatch+SPEC+guard+POST/current audit SHA required")
    if Path(__file__).resolve()!=public/NAME/"audit_multik_resume.py" or sha(__file__)!=audit_sha:raise ValueError("Current reviewed auditor SHA differs")
    helpers=load_helpers(public);inputs=helpers["inputs"].Inputs(public,cache);inputs.capture("public",NAME+"/audit_multik_resume.py",audit_sha)
    result={};error=None
    try:result=collect(inputs,helpers,bindings)
    except (ValueError,KeyError,TypeError,OverflowError,OSError,csv.Error,IndexError) as failure:
        error=dict(error_class=type(failure).__name__,reason=str(failure) if isinstance(failure,ValueError) else "Bound metadata, closure or arithmetic evidence unavailable or rejected")
    try:preserved=inputs.preserved()
    except OSError:preserved=False
    if not preserved:error=error or dict(error_class="InputPreservationFailed",reason="Captured input changed before output")
    receipt=dict(schema="INDEPENDENT_MULTIK_RESUME_AUDIT_v1",status="PASS" if error is None and preserved else "NOT_COMPLETE_AUDIT_EVIDENCE_REJECTED",
        audit_source_sha256=audit_sha,ROOT_bound_execution_sha256=dict(bindings,dispatch_source_sha=DISPATCH_SHA,post_source_sha=POST_SHA),
        inputs=[dict(root=root,relative_path=name,sha256=value) for (root,name),value in sorted(inputs.files.items())],input_count=len(inputs.files),
        input_SHA_before_after_equal=preserved,source48_preserved=error is None and preserved,error=error,**result,
        accuracy_comparison_tolerance=dict(atol=1e-9,rtol=1e-7,source="Original priority_runner.independent_rescore and frozen933883 arithmetic"),
        cost_comparison_policy="Exact fsum mean/median/min/max/linear p90. Any difference rejects completion; no new tolerance.",
        new_Task_NPZ_raw_checkpoint_tensor_model_training_inference_cost_process_operations=0,current_PID_query_performed=False,
        original_checkpoint_scope="20 opaque original/backup SHA+size checks; no tensor deserialization. Saved selected artifacts and V archives are opaque byte bindings.",
        parent_BASE_full_V_completed_unchanged_false=True,whole_E4_complete=False,whole_MASTER_complete=False,independent_generalization_claimed=False,
        limitations=["Two saved-point parent refits, six reused-parent prefix routes; not six independent training runs or independent observations.",
            "Original completed search is reconstructed with zero new optimizer/F activity; historical cumulative counters and unsaved interrupted work retain separate scopes.",
            "Single seed92601, repeated development V, selected three epochs and explicit UNDERTRAINED_OR_BUDGET_LIMIT. Convergence and generalization remain unresolved.",
            "Recorded guard/tree/PID-create evidence only; ROOT supervisor performs current absence checks. Auditor makes no current idleness claim.",
            "Native cost omits raw total F counters/proof map/output dtype. Independently derived trial counters and actual argv proof/quiet bindings are separate evidence.",
            "Full-array FP64 and CPU6 are actual earlier receipts bound here; this auditor only recomputes public sufficient statistics/trials, never model predictions.",
            "Compression family capacity/grid/compute differs. Audit PASS is not method superiority or completed E4/MASTER."])
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf-8",newline="\n") as stream:json.dump(receipt,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--execute",action="store_true");parser.add_argument("--root-triggered",action="store_true")
    for flag in ("public-root","cache-root","plan-sha","dispatch-sha","dispatch-spec-sha","dispatch-guard-sha","post-sha","audit-sha","output"):parser.add_argument("--"+flag)
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="PREPARATION_ONLY_NOT_EXECUTED",input_reads=0,files_written=0,scientific_operations=0,
            numpy_imported="numpy" in sys.modules,torch_imported="torch" in sys.modules,future_dispatch_SHA=None,future_POST_SHA=None)));return 0
    fields=("public_root","cache_root","plan_sha","dispatch_sha","dispatch_spec_sha","dispatch_guard_sha","post_sha","audit_sha","output")
    if not args.root_triggered or any(getattr(args,k) is None for k in fields):parser.error("Explicit ROOT trigger, roots, actual complete execution SHA bindings and fresh output required")
    try:value=execute(args.public_root,args.cache_root,{k:getattr(args,k) for k in ("plan_sha","dispatch_sha","dispatch_spec_sha","dispatch_guard_sha","post_sha")},args.audit_sha,args.output)
    except (ValueError,FileExistsError,FileNotFoundError) as failure:
        print(json.dumps(dict(status="NOT_COMPLETE_AUDIT_EVIDENCE_REJECTED",error_class=type(failure).__name__)));return 1
    print(json.dumps(dict(status=value["status"],input_count=value["input_count"],files_written=1)));return 0 if value["status"]=="PASS" else 1


if __name__=="__main__":raise SystemExit(main())
