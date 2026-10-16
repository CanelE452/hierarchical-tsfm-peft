"""ROOT-only initial N5 audit; default0 I/O and no model/NPZ operations.

Only two initial Jena family minima. Partials need honest ROOT manual reporting.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

NAME="priority_valid_family_20261016_v1"
OLD="postT_V_only_cpu_family_minima_20261016_v1"
OLD_PLAN_SHA="e34033935916dcaded0885ccde8e1aa084f5cc0df994024d0827510d6db53b10"
RUNNER_SHA="12462b8f1085e346750a24cd37fd52191f8780799e2f9e5903b4a57bcbc570c1"
D2="priority_pause_validate_20261016_v1/FOLLOWUP_ROOT_v1"
FIT=D2+"/FAMILY_INITIAL_DISPATCH_PREP_v1"
POST=D2+"/FAMILY_INITIAL_POST_PREP_v1"
FIT_SOURCE_SHA="67dfce149dc8c3f8b84221d5b7c8da1e3f2233a3e7058ee2182c09fdc63cc1cc"
POST_SOURCE_SHA="57cf98eda8f8d1b2c6303ba336860733a41454c470dd28c0e18485d81e257982"
GUARD_REL="postT_cpu_lag_preparation_20261016_v1/cpu_group_supervisor.py"
GUARD_SHA="82a837ea0cfee8b52765f4ac1c7325d564e886c9e94857bd0f957e4a40c75f8b"
F_SHA="c2b4edf7db22d3f18e58ca79299a5c7e864e9b8f277e4e29e2f88d07dbb35874"
MATH_SOURCE_SHA="933883a052b02e6a41ccdadaa5ec45e80b977dbfaada007607ac1093f518dd04"
BUILDER_SOURCE_SHA="8bb730a55e0f0f924de0616bf917ca4b47e4117503027cfc4656c6ef502750dd"
COHORT="jena:primary:V_select:stride48"
ORIGINS=tuple(range(14675,16739,48))
BANDS=((0,12),(12,24),(24,48))
OBSERVED=43344
STAGE_FIELDS=tuple("stage_"+x+"_seconds" for x in ("CPU_preprocess","host_to_device","P_E","F","D","inverse_and_CPU"))
TARGETS=(
 dict(role="LORA_R8_C21",run_id="postT_cpu_jena_E1_LORA_R8_k0_s92601_family_v1",family="LORA",registered_K=0,K=21,lr=.0001,fit_id="family_initial_jena_lora_r8_fit_root_v1"),
 dict(role="B4_K6",run_id="postT_cpu_jena_E1_B4_k6_s92601_family_v1",family="B4",registered_K=6,K=6,lr=.001,fit_id="family_initial_jena_b4_k6_fit_root_v1"))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""):h.update(chunk)
    return h.hexdigest()


def load_helpers(public):
    modules=[]
    for filename,digest in (("independent_beijing_audit.py",MATH_SOURCE_SHA),("build_beijing_report.py",BUILDER_SOURCE_SHA)):
        path=Path(public)/"priority_valid_minima_20261016_v1"/filename
        if sha(path)!=digest:raise ValueError("Frozen stdlib arithmetic/input helper changed")
        spec=importlib.util.spec_from_file_location("n5_initial_"+filename[:-3],path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);modules.append(module)
    modules[0].COHORT=COHORT
    return tuple(modules)


def expected_steps():
    return {t["fit_id"].replace("_fit_root_v1","_"+mode+("_b"+str(batch) if batch else "")+"_root_v1"):(t["run_id"],mode,batch)
        for t in TARGETS for mode,batch in (("rescore",None),("replay",None),("cost",1),("cost",8))}


def validate_post(v):
    rows=v.get("steps",[])
    if (v.get("status")!="COMPLETE_TWO_VALIDATIONS_AND_FOUR_COSTS" or v.get("skipped")!=[] or v.get("error") is not None
            or v.get("pins_before_after_unchanged") is not True or v.get("changed_or_unreadable_pinned_paths")!=[] or len(rows)!=8
            or {r.get("id") for r in rows}!=set(expected_steps()) or any(r.get("success") is not True or r.get("strong_observed_tree_closure_verified") is not True for r in rows)):
        raise ValueError("Eight exact successful POST steps required; partial uses ROOT manual honest reporting")


def aggregate_trials(rows,identifier,batch,k):
    if k not in (6,21) or batch not in (1,8):raise ValueError("Only admitted B4 K6 / LoRA C21 trials")
    phases=dict(warmup=10,timed=20,stage_probe=20)
    expected={(phase,i) for phase,n in phases.items() for i in range(n)};seen={};hashes=set()
    fields={"run_id","phase","trial","order_position","batch_size","whole_seconds","F_logical_streams","F_physical_streams","F_calls","prediction_sha256","row_microbatch","stage_instrumentation",*STAGE_FIELDS}
    for row in rows:
        key=(row.get("phase"),int(row.get("trial",-1)));stage=key[0]=="stage_probe"
        if (set(row)!=fields or key not in expected or key in seen or row["run_id"]!=identifier or int(row["batch_size"])!=batch or int(row["order_position"])!=0
                or (int(row["F_logical_streams"]),int(row["F_physical_streams"]),int(row["F_calls"]))!=(batch*k,batch*k,math.ceil(batch*k/128))
                or int(row["row_microbatch"])!=128 or row["stage_instrumentation"]!=str(stage) or len(row["prediction_sha256"])!=64 or any(c not in "0123456789abcdef" for c in row["prediction_sha256"])):raise ValueError("Exact50 physical trial keys/phase/K/counters differ")
        seconds=float(row["whole_seconds"])
        if not math.isfinite(seconds) or seconds<=0:raise ValueError("Finite positive whole time required")
        for field in STAGE_FIELDS:
            if stage:
                number=float(row[field])
                if not math.isfinite(number) or number<0:raise ValueError("Finite nonnegative intrusive-stage time required")
            elif row[field]!="":raise ValueError("Ordinary whole trials cannot contain intrusive stage timings")
        seen[key]=seconds;hashes.add(row["prediction_sha256"])
    if set(seen)!=expected:raise ValueError("Complete10warm/20timed/20stage required")
    summaries=[]
    for phase in ("timed","stage_probe"):
        values=sorted(v for key,v in seen.items() if key[0]==phase);position=(len(values)-1)*.9;low=math.floor(position);fraction=position-low
        summaries.append(dict(run_id=identifier,phase=phase,batch_size=batch,trials=20,median_seconds=statistics.median(values),mean_seconds=math.fsum(values)/len(values),min_seconds=values[0],max_seconds=values[-1],p90_seconds=values[low]+(values[low+1]-values[low])*fraction,stage_instrumentation=phase=="stage_probe",stage_sum_equals_whole_claimed=False))
    return dict(trial_rows=50,phase_rows=phases,F_logical_streams=50*batch*k,F_physical_streams=50*batch*k,F_calls=50*math.ceil(batch*k/128),unique_prediction_hashes=len(hashes),summaries=summaries)


def validate_guard(guard,spec,spec_sha,identifier,command,started):
    if (guard.get("schema")!="CPU_GROUP_SUPERVISOR_RESULT_v1" or guard.get("status") not in {"COMPLETE","STOPPED_RESOURCE_OR_WORKER_FAILURE"} or guard.get("helper_sha256")!=GUARD_SHA
            or guard.get("helper_unchanged") is not True or guard.get("spec_file_sha256")!=spec_sha or guard.get("supervisor_failure_class") is not None or guard.get("remaining_owned_processes")!=[]):raise ValueError("Recorded exact guard/spec/helper closure required")
    processes=guard.get("processes",[]);effective=guard.get("effective_spec",{})
    if len(processes) not in (1,2) or len(spec.get("processes",[]))!=len(processes) or {r["id"] for r in processes}!={r["id"] for r in spec["processes"]}:raise ValueError("Exact guard process identity set required")
    if (effective.get("max_each_RSS")!=3221225472 or effective.get("max_group_RSS") not in (3221225472,6442450944) or effective.get("deadline_unix")!=1792222550):raise ValueError("Original CPU worker/group/deadline limits required")
    for field in ("schema","deadline_unix","max_each_RSS","max_group_RSS","poll_seconds","terminate_grace_seconds","kill_grace_seconds"):
        if effective.get(field)!=spec.get(field):raise ValueError("Effective and original guard setting differ")
    for field in ("asset_root","cache_root","output_root"):
        if Path(effective[field]).resolve()!=Path(spec[field]).resolve():raise ValueError("Effective and original guard path differ")
    if guard.get("sampled_group_peak_RSS_bytes",math.inf)>effective["max_group_RSS"]:raise ValueError("Observed group RSS exceeded original guard limit")
    for row in processes:
        if (row.get("returncode") is None or row.get("end_unix") is None or not row.get("owned_processes") or row.get("remaining_owned_processes")!=[] or row.get("process_tree_errors")!=[] or row.get("observed_tree_join_confirmed") is not True or row.get("sampled_peak_RSS_bytes",math.inf)>3221225472):raise ValueError("Every observed worker tree must be terminal without errors")
        root=row.get("root",{})
        if sum(p.get("pid")==root.get("pid") and p.get("create_time")==root.get("create_time") for p in row["owned_processes"])!=1:raise ValueError("Root exact identity missing from recorded ownership")
    selected=[r for r in processes if r["id"]==identifier]
    if len(selected)!=1 or selected[0]["returncode"]!=0 or selected[0].get("stop_reason") is not None or selected[0].get("signals")!=[]:raise ValueError("Selected actual child failed or was signalled")
    row=selected[0]
    if not any(p["pid"]==started.get("actual_pid") and abs(p["create_time"]-started.get("actual_create_time",-1))<=1e-6 for p in row["owned_processes"]):raise ValueError("Actual STARTED PID/create_time not in recorded owned tree")
    for collection in (spec["processes"],effective.get("processes",[])):
        matches=[p for p in collection if p.get("id")==identifier]
        if len(matches)!=1 or matches[0].get("command")!=command:raise ValueError("Original SPEC and effective argv differ")
    original=next(p for p in spec["processes"] if p["id"]==identifier);actual=next(p for p in effective["processes"] if p["id"]==identifier)
    if original["max_seconds"]!=actual["max_seconds"] or Path(original["cwd"]).resolve()!=Path(actual["cwd"]).resolve():raise ValueError("Effective and original worker deadline/cwd differ")
    return row


def validate_validation(v,mode,artifact,manifest_sha,k):
    if v.get("artifact_sha256")!=artifact or v.get("manifest_sha256")!=manifest_sha:raise ValueError("Validation selected artifact/manifest differs")
    if mode=="rescore":
        if (v.get("status")!="PASS_INDEPENDENT_FULL_V_FP64" or v.get("saved_predictions_rescored") is not True or v.get("new_model_inference") is not False or v.get("n_observed")!=OBSERVED or v.get("full_V_origins")!=43):raise ValueError("Actual fullV FP64 saved-array rescore required")
    elif mode=="replay":
        if (v.get("status")!="PASS_FRESH_CPU6_REPLAY" or v.get("executed_now") is not True or v.get("microbatches")!=[4,2] or len(v.get("origins",[]))!=6
                or v.get("F_counters")!=dict(streams=6*k,physical_streams=6*k,physical_calls=2) or v.get("atol")!=1e-5 or v.get("rtol")!=1e-4
                or v.get("base_restored_after_original_Route_close") is not True or v.get("F_final_state_sha256")!=F_SHA):raise ValueError("Actual CPU6/4+2 replay/base restoration required")
        if k==21 and (v.get("lora_merge_parity",{}).get("status")!="PASS" or v.get("lora_merge_parity_F_counters")!=dict(streams=42,physical_streams=42,physical_calls=2)):
            raise ValueError("LoRA parity forward activity must remain separate from CPU6 replay")
        if k==6 and (v.get("lora_merge_parity")!={"status":"NOT_LORA"} or v.get("lora_merge_parity_F_counters")!={}):raise ValueError("B4 must not inherit LoRA parity activity")
    else:raise ValueError("Only rescore/replay supported")


def validate_training(v,search,refit,t,scaler):
    epoch=v.get("selected_epoch");nsearch=10482-512-48+1;nrefit=12579-512-48+1
    targets=[0]+[math.ceil(e*nsearch) for e in (.25,1.,3.)]
    updates=sum(math.ceil((b-a)/16) for a,b in zip(targets,targets[1:]))
    if (v.get("status")!="COMPLETE_VALID" or v.get("run_id")!=t["run_id"] or v.get("family")!=t["family"] or v.get("seed")!=92601
            or v.get("epochs")!=[.25,1.,3.] or epoch not in (0.,.25,1.,3.) or v.get("refit_scaler_sha")!=scaler
            or v.get("search_fit_end")!=10482 or v.get("refit_fit_end")!=12579):raise ValueError("Exact Jena search3/fresh selected A60 refit required")
    if v.get("training_status")!=("UNDERTRAINED_OR_BUDGET_LIMIT" if epoch==3 else "SCREEN_COMPLETE_CONVERGENCE_UNRESOLVED"):raise ValueError("Original selected-endpoint training uncertainty must remain explicit")
    wants=(dict(status="SEGMENT_COMPLETE",completed_epochs=[0.,.25,1.,3.],train_origins=nsearch,exposures=3*nsearch,updates=updates,selected_epoch=epoch),
           dict(status="SEGMENT_COMPLETE",completed_epochs=[0.]+([epoch] if epoch else []),train_origins=nrefit,exposures=math.ceil(epoch*nrefit),updates=math.ceil(math.ceil(epoch*nrefit)/16)))
    for stage,want in zip((search,refit),wants):
        if any(stage.get(k)!=x for k,x in want.items()):raise ValueError("Actual original stage exposure/update/endpoint differs")
    for label,stage in (("search",search),("refit",refit)):
        if any(v.get(label+"_"+k)!=stage[k] for k in ("updates","exposures")):raise ValueError("Fit/stage activity differs")


def validate_cohort(v,reference):
    if (v.get("split")!="V_select" or v.get("cohort")!=COHORT or any(v.get(k)!=reference[k] for k in ("data_sha","scaler_sha"))
            or any(v.get("arrays",{}).get(k)!=reference["arrays"][k] for k in ("target_normalized","observed","origins","channel_ids"))
            or any(v.get("score",{}).get(k)!=x for k,x in dict(n_observed=OBSERVED,n_origins=43,num_channels=21,horizon=48,active_channels=21).items())):
        raise ValueError("Exact Jena V43/43344 observed labels/A60/mask/target/origin/channel fingerprints required")


def collect(inputs,h,b,bindings):
    plan_sha,fit_spec_sha,dispatch_sha,post_sha,post_source_sha=[bindings[k] for k in ("plan_sha","fit_spec_sha","dispatch_sha","post_sha","post_source_sha")]
    if post_source_sha!=POST_SOURCE_SHA:raise ValueError("Exact reviewed initial family POST source required")
    for root,path,digest in (("public",NAME+"/family_minimum_runner.py",RUNNER_SHA),("cache",FIT+"/RUN_ROOT_DRAFT_v1.ps1",FIT_SOURCE_SHA),
            ("cache",POST+"/RUN_ROOT_DRAFT_v1.ps1",post_source_sha),("cache",GUARD_REL,GUARD_SHA)):
        inputs.capture(root,path,digest)
    plan=inputs.json("public",NAME+"/PLAN.json",plan_sha);old=inputs.json("public",OLD+"/PLAN.json",OLD_PLAN_SHA)
    if (plan.get("status")!="SEALED_BY_ROOT" or plan.get("runner_sha256")!=RUNNER_SHA or plan.get("new_T_target_operations_allowed") is not False
            or plan.get("protected_tail_numeric_parse_allowed") is not False or plan.get("original_family_plan_sha256")!=OLD_PLAN_SHA or plan.get("deadline_unix")!=1792222550):raise ValueError("Exact new ROOT sealed original family recipe plan required")
    if len(old["source39"])!=39 or len(old["immutable_public"])!=9 or len(set(old["source39"])|set(old["immutable_public"]))!=48:raise ValueError("Original source48 required")
    for path,digest in {**old["source39"],**old["immutable_public"],**plan["dependencies"]}.items():inputs.capture("public",path,digest)
    for path,digest in plan["original_family_files"].items():inputs.capture("public",OLD+"/"+path,digest)
    dispatch=inputs.json("cache",FIT+"/RESULT.json",dispatch_sha);post=inputs.json("cache",POST+"/RESULT.json",post_sha);validate_post(post)
    if (dispatch.get("status")!="COMPLETE_TWO_FITS_AND_FULL_V" or dispatch.get("error") is not None or dispatch.get("strong_observed_tree_closure_verified") is not True
            or dispatch.get("pins_before_after_unchanged") is not True or dispatch.get("changed_or_unreadable_pinned_paths")!=[] or dispatch.get("plan_sha256")!=plan_sha
            or len(dispatch.get("conditions",[]))!=2 or {r["run_id"] for r in dispatch["conditions"]}!={t["run_id"] for t in TARGETS}
            or post.get("fit_dispatch_result_sha256")!=dispatch_sha or post.get("fit_dispatch_SPEC_sha256")!=fit_spec_sha or post.get("plan_sha256")!=plan_sha):
        raise ValueError("Two actual complete fits and bound POST required; partial remains separately reportable")
    step_map={r["id"]:r for r in post["steps"]};guards=[];others=[];cost_intervals=[];rows=[];accuracy=[];cost_records=[];physical={}

    def bound(t,tag,mode,status,batch=None):
        folder=NAME+"/diagnostics/"+tag;outer=inputs.json("public",folder+"/RESULT.json");science=outer.get("scientific_result") or {}
        if (outer.get("run_id")!=t["run_id"] or outer.get("mode")!=mode or outer.get("status")!=status or outer.get("plan_sha256")!=plan_sha
                or outer.get("error_class") is not None or outer.get("postflight_error_class") is not None
                or outer.get("original_source39_immutable9_family7_probe4_recipe4_and_model_pins_preserved") is not True or outer.get("new_T_target_operations")!=0):raise ValueError("Actual original family outer/source/postflight closure differs")
        if mode=="fit":
            gp,sp=FIT+"/GUARD_ROOT_v1/RESULT.json",FIT+"/SPEC_ROOT_v1.json";record=next(r for r in dispatch["conditions"] if r["run_id"]==t["run_id"])
            if record.get("worker_science_complete") is not True or record.get("id")!=tag:raise ValueError("Selected fit must have actually completed")
        else:
            gp,sp=NAME+"/supervisors/"+tag+"/RESULT.json",POST+"/SPEC_"+tag+".json";record=step_map[tag]
            if (record.get("run_id"),record.get("mode"),record.get("batch"))!=(t["run_id"],mode,batch) or record.get("guard_exit_code")!=0:raise ValueError("POST step identity differs")
        if record.get("outer_sha256")!=inputs.files[("public",folder+"/RESULT.json")]:raise ValueError("Dispatch/POST outer SHA differs")
        guard=inputs.json("cache",gp);spec=inputs.json("cache",sp,fit_spec_sha if mode=="fit" else guard["spec_file_sha256"])
        if mode=="fit" and post.get("fit_guard_sha256")!=inputs.files[("cache",gp)]:raise ValueError("POST must bind the actual whole fit guard")
        entries=[r for r in spec["processes"] if r["id"]==tag]
        if len(entries)!=1:raise ValueError("One actual worker argv required")
        command=entries[0]["command"]
        for flag,value in (("--run-id",t["run_id"]),("--mode",mode),("--plan-sha",plan_sha)):
            if command.count(flag)!=1 or command[command.index(flag)+1]!=value:raise ValueError("Actual argv scientific identity differs")
        for flag,want in (("--output",inputs.roots["public"]/folder),("--public-root",inputs.roots["public"]),("--cache-root",inputs.roots["cache"])):
            if command.count(flag)!=1 or Path(command[command.index(flag)+1]).resolve()!=want.resolve():raise ValueError("Actual argv output/root path differs")
        runner=inputs.roots["public"]/NAME/"family_minimum_runner.py"
        if sum(Path(x).resolve()==runner.resolve() for x in command if not x.startswith("-"))!=1:raise ValueError("Actual worker must invoke the frozen N5 runner")
        started=inputs.json("public",folder+"/STARTED.json")
        if any(started.get(k)!=x for k,x in dict(run_id=t["run_id"],mode=mode,plan_sha256=plan_sha).items()):raise ValueError("STARTED scientific identity differs")
        child=validate_guard(guard,spec,inputs.files[("cache",sp)],tag,command,started)
        if mode!="fit" and (record.get("guard_sha256")!=inputs.files[("cache",gp)] or record.get("spec_sha256")!=inputs.files[("cache",sp)]):raise ValueError("Actual POST guard/SPEC SHA differs")
        if mode=="cost":
            for flag in ("--quiet-proof","--quiet-sha"):
                if command.count(flag)!=1:raise ValueError("Actual quiet command required")
            qp=Path(command[command.index("--quiet-proof")+1]).resolve();cache=inputs.roots["cache"];qs=command[command.index("--quiet-sha")+1]
            if not qp.is_relative_to(cache):raise ValueError("Quiet proof must stay private")
            quiet=inputs.json("cache",qp.relative_to(cache).as_posix(),qs)
            if quiet.get("status")!="QUIET_CPU_COST_WINDOW" or quiet.get("active_model_workers")!=[] or quiet.get("active_cost_workers")!=[] or not 0<=guard["start_unix"]-quiet["captured_unix"]<=120 or science.get("quiet_proof_sha256")!=qs:raise ValueError("Fresh quiet window/proof binding differs")
        (cost_intervals if mode=="cost" else others).append(dict(id=tag,start=guard["start_unix"],end=guard["end_unix"]))
        guards.append(dict(id=tag,cache_relative_path=gp,guard_sha256=inputs.files[("cache",gp)],selected_returncode=0,sampled_peak_RSS_bytes=child["sampled_peak_RSS_bytes"],current_PID_query_performed=False))
        return folder,outer,child

    for t in TARGETS:
        identifier,k=t["run_id"],t["K"];condition=next(r for r in plan["conditions"] if r["run_id"]==identifier);spec=condition["spec"]
        if (condition["family"],condition["effective_F_width"],condition["registered_latent_k"],spec["lr"],spec["lora_rank"],spec["lora_alpha"])!=(t["family"],k,t["registered_K"],t["lr"],8,16):raise ValueError("Exact original family/rank/learning rate/registered K required")
        reference=inputs.json("public",condition["V_reference"]["relative_path"],condition["V_reference"]["sha256"])
        validate_cohort(reference,reference);channels=old["task_contracts"]["jena"]["columns"]
        if len(channels)!=21 or len(set(channels))!=21:raise ValueError("Original21 unique channels required")
        run=NAME+"/runs/"+identifier;pred=NAME+"/predictions/"+identifier+"/V_select"
        manifest=inputs.json("public",pred+"/PREDICTION_MANIFEST.json");validate_cohort(manifest,reference);msha=inputs.files[("public",pred+"/PREDICTION_MANIFEST.json")]
        scores=inputs.csv("public",pred+"/SCORES.csv")
        if len(scores)!=1:raise ValueError("One exact full-V score row required")
        score=scores[0];receipt=inputs.json("public",run+"/BASE_FIT_RECEIPT.json");search=inputs.json("public",run+"/search/RECEIPT.json");refit=inputs.json("public",run+"/refit/RECEIPT.json")
        validate_training(receipt,search,refit,t,reference["scaler_sha"])
        want=dict(run_id=identifier,family=t["family"],seed=92601,device="cpu",CPU_threads=2,artifact_sha256=receipt["artifact_sha256"],config_sha=b.fingerprint(spec),source_sha=b.fingerprint(dict(runner=RUNNER_SHA,**plan["dependencies"])),model_sha=b.fingerprint(dict(artifact_sha=receipt["artifact_sha256"],model_state_sha=receipt["model_state_sha"])),F_counters=dict(streams=43*k,physical_streams=43*k,physical_calls=11))
        if any(manifest.get(key)!=value for key,value in want.items()) or any(score.get(key)!=manifest[key] for key in ("run_id","split","cohort")):raise ValueError("FullCPU V identity/stream/scaler/source binding differs")
        _,outer,_=bound(t,t["fit_id"],"fit","COMPLETE_FIT_AND_FULL_V")
        if outer["scientific_result"].get("fit_receipt")!=receipt or outer["scientific_result"].get("V_score")!=manifest["score"] or outer["scientific_result"].get("full_V_completed") is not True or outer["scientific_result"].get("F_final_state_sha256")!=F_SHA:raise ValueError("Actual complete fit/fullV/unmerged base restoration required")
        proof_shas={};rescored=None
        for mode,status in (("rescore","PASS_INDEPENDENT_FULL_V_FP64"),("replay","PASS_FRESH_CPU6_REPLAY")):
            tag=t["fit_id"].replace("_fit_root_v1","_"+mode+"_root_v1");folder,outer,_=bound(t,tag,mode,status);v=outer["scientific_result"]
            validate_validation(v,mode,receipt["artifact_sha256"],msha,k);saved=inputs.json("public",folder+("/RESCORE_RECEIPT.json" if mode=="rescore" else "/REPLAY_RECEIPT.json"))
            if any(v.get(key)!=value for key,value in saved.items()):raise ValueError("Validation outer and saved receipt differ")
            if mode=="rescore":rescored=v
            else:
                extra=42 if k==21 else 0;calls=4 if k==21 else 2
                if any(outer.get("actual_F_counters",{}).get(key)!=value for key,value in dict(streams=6*k+extra,physical_streams=6*k+extra,physical_calls=calls).items()):raise ValueError("Outer replay total must include separate LoRA parity activity")
            proof_shas[mode]=inputs.files[("public",folder+"/RESULT.json")]
        stats=h.gzip_rows(inputs,"public",pred+"/PUBLIC_ERROR_STATS.csv.gz",2709)
        aggregate=h.aggregate_statistics(stats,identifier,channels,ORIGINS,BANDS,OBSERVED);deltas=h.compare_accuracy(aggregate,manifest,score,rescored)
        accuracy.append(dict(run_id=identifier,**aggregate,metric_deltas=deltas,manifest_sha256=msha,artifact_sha256=receipt["artifact_sha256"]))
        row=dict(role=t["role"],run_id=identifier,registered_K=t["registered_K"],effective_F_width=k,MSE=manifest["score"]["mse"],MAE=manifest["score"]["mae"],accuracy_device="cpu",training_device="cpu",CPU_threads=2,observed_labels=OBSERVED,V_origins=43,search_max_epoch=3,selected_refit_epoch=receipt["selected_epoch"],search_seconds=search["seconds"],refit_seconds=refit["seconds"],training_status=receipt["training_status"],lr=t["lr"],lora_rank=8 if k==21 else None,unmerged_F_base_sha256=F_SHA,F_resident_bytes=None,F_resident_bytes_scope="NOT_SEPARATELY_RECORDED",accuracy_reused=False)
        for key in ("search_updates","refit_updates","search_exposures","refit_exposures"):row[key]=receipt[key]
        for batch in (1,8):
            tag=t["fit_id"].replace("_fit_root_v1","_cost_b"+str(batch)+"_root_v1");folder,outer,child=bound(t,tag,"cost","COMPLETE_PROTOCOL",batch)
            cost=inputs.json("public",folder+"/COST_RECEIPT.json");validate_cost(cost,t,batch,receipt["artifact_sha256"],msha,proof_shas)
            if any(outer["scientific_result"].get(key)!=value for key,value in cost.items()) or outer["scientific_result"].get("F_final_state_sha256")!=F_SHA:raise ValueError("Cost outer/effective/unmerged base restoration differs")
            pair=(cost["physical_input_sha256"],cost["observation_mask_sha256"])
            if batch in physical and physical[batch]!=pair:raise ValueError("Cost histories/mask differ across family routes")
            physical[batch]=pair;trials=h.gzip_rows(inputs,"public",folder+"/COST_TRIALS.csv.gz",50,cost["trials_sha256"])
            totals=aggregate_trials(trials,identifier,batch,k);h.compare_summaries(totals["summaries"],cost["summaries"])
            cost_records.append(dict(run_id=identifier,batch=batch,**totals,effective_F_state_sha256_during_measurement=cost["effective_F_state_sha256_during_measurement"]))
            timed=next(s for s in cost["summaries"] if s["phase"]=="timed");memory=cost["memory"]
            row.update({f"B{batch}_median_seconds":timed["median_seconds"],f"B{batch}_p90_seconds":timed["p90_seconds"],f"B{batch}_CPU_weights_MiB":memory["resident_weights_cpu_MiB"],f"B{batch}_observed_tree_peak_RSS_bytes":child["sampled_peak_RSS_bytes"],f"B{batch}_F_instances":memory["foundation_instances_created"],f"B{batch}_F_streams_per_trial":batch*k,f"B{batch}_F_calls_per_trial":math.ceil(batch*k/128),f"B{batch}_effective_F_state_sha256":cost["effective_F_state_sha256_during_measurement"],f"B{batch}_model_load_seconds":cost["model_load_seconds"],f"B{batch}_LoRA_merge_and_model_loading_in_whole_trial_timer":cost["LoRA_merge_and_model_loading_in_whole_trial_timer"],f"B{batch}_CPU_restore_snapshot_bytes_in_Route_memory_state":cost["CPU_restore_snapshot_bytes_in_Route_memory_state"]})
            row[f"B{batch}_effective_F_state_scope"]=cost["effective_F_state_scope"]
        rows.append(row)
    h.require_same_counts(accuracy);h.validate_intervals(cost_intervals,others)
    return dict(rows=rows,accuracy=accuracy,cost_conditions=cost_records,recorded_guards=guards,totals=dict(new_accuracy_conditions=2,new_public_stats_rows=sum(r["public_stats_rows"] for r in accuracy),unique_observed_V_labels=OBSERVED,new_trial_rows=sum(r["trial_rows"] for r in cost_records),new_cost_conditions=len(cost_records),new_summary_groups=sum(len(r["summaries"]) for r in cost_records),new_summary_numeric_fields_checked=5*sum(len(r["summaries"]) for r in cost_records),F_logical_streams=sum(r["F_logical_streams"] for r in cost_records),F_physical_streams=sum(r["F_physical_streams"] for r in cost_records),F_calls=sum(r["F_calls"] for r in cost_records)),baseline_comparison_included=False)


def validate_cost(v,t,batch,artifact,manifest,proofs):
    k=t["K"];calls=math.ceil(batch*k/128)
    want=dict(status="COMPLETE_PROTOCOL",run_id=t["run_id"],device="cpu",CPU_threads=2,batch_size=batch,warmups=10,timed_trials=20,stage_trials=20,actual_trial_rows=50,artifact_sha256=artifact,accuracy_manifest_sha256=manifest,validation_result_sha256=proofs,F_logical_streams_per_trial=batch*k,F_calls_per_trial=calls,cached_Z_or_forecast_reads=0,physical_output_and_inverse_dtype="float64",stage_probe_separate_intrusive_scope=True,training_peak_included=False,new_T_target_operations=0,unmerged_F_base_sha256=F_SHA,base_restored_after_original_Route_close=True,LoRA_merge_and_model_loading_in_whole_trial_timer=False)
    if any(v.get(key)!=value for key,value in want.items()) or v.get("origins")!=list(ORIGINS)[:batch] or v.get("F_counters")!=dict(streams=50*batch*k,physical_streams=50*batch*k,physical_calls=50*calls) or v.get("memory",{}).get("foundation_instances_created")!=1:raise ValueError("Actual physical CPU2/B1B8/stream-call/memory protocol differs")
    effective=v.get("effective_F_state_sha256_during_measurement")
    if not isinstance(effective,str) or len(effective)!=64 or any(c not in "0123456789abcdef" for c in effective):raise ValueError("Actual effective F hash required")
    if k==6 and effective!=F_SHA:raise ValueError("B4 frozen unmerged F changed")
    if k==21 and v.get("CPU_restore_snapshot_bytes_in_Route_memory_state") is not False:raise ValueError("LoRA snapshot exclusion must be explicit; RSS is a separate scope")


def execute(public,cache,bindings,audit_sha,output):
    output=Path(output).absolute()
    if output.exists():raise FileExistsError("Existing audit preserved before input reads")
    public,cache=Path(public).resolve(),Path(cache).resolve()
    if not output.resolve().is_relative_to(public/NAME) or output.suffix!=".json":raise ValueError("Fresh explicit N5 JSON output required")
    if set(bindings)!={"plan_sha","fit_spec_sha","dispatch_sha","post_sha","post_source_sha"} or any(not isinstance(s,str) or len(s)!=64 or any(c not in "0123456789abcdef" for c in s) for s in (*bindings.values(),audit_sha)):raise ValueError("ROOT must provide actual future PLAN/SPEC/dispatch/POST/source SHAs")
    if Path(__file__).resolve()!=public/NAME/"audit_initial_family.py" or sha(__file__)!=audit_sha:raise ValueError("Current ROOT-pinned auditor required")
    h,b=load_helpers(public);inputs=b.Inputs(public,cache);inputs.capture("public",NAME+"/audit_initial_family.py",audit_sha)
    for filename,digest in (("independent_beijing_audit.py",MATH_SOURCE_SHA),("build_beijing_report.py",BUILDER_SOURCE_SHA)):inputs.capture("public","priority_valid_minima_20261016_v1/"+filename,digest)
    result={};error=None
    try:result=collect(inputs,h,b,bindings)
    except (ValueError,FileNotFoundError,KeyError,TypeError,OverflowError,csv.Error,OSError,StopIteration) as exc:error=dict(error_class=type(exc).__name__,reason=str(exc) if isinstance(exc,ValueError) else "Missing/incomplete relative evidence; ROOT manual partial report required")
    try:preserved=inputs.preserved()
    except OSError:preserved=False
    if not preserved:error=error or dict(error_class="InputPreservationFailed",reason="Captured inputs changed")
    receipt=dict(schema="INDEPENDENT_INITIAL_FAMILY_AUDIT_v1",status="PASS" if error is None and preserved else "NOT_COMPLETE_AUDIT_EVIDENCE_REJECTED",audit_source_sha256=audit_sha,ROOT_bound_execution_sha256=bindings,input_SHA_before_after_equal=preserved,source48_preserved=error is None and preserved,inputs=[dict(root=root,relative_path=name,sha256=digest) for (root,name),digest in sorted(inputs.files.items())],error=error,**result,whole_MASTER_complete=False,independent_generalization_claimed=False,new_models_NPZ_raw_T_numeric_training_inference_cost_or_process_operations=0,accuracy_comparison_tolerance=dict(atol=1e-9,rtol=1e-7),cost_comparison_policy="Original exact fsum/median/min/max/linear-p90; no new tolerance",limitations=["Single seed92601; repeated development V; search3 and selected refit lengths differ; convergence unresolved.","Inherited legacy whole-TRAIN normalization and old quarter/probe full-loader ancestry are preserved; prefix-only entry does not repair that ancestry or establish independent causality.","LoRA merged effective F hash is separate from the restored unmerged original base hash.","Whole CPU weights exclude separately held LoRA restore snapshots; observed tree RSS remains separate; F-only resident bytes not separately measured.","No complete AdaPTS paper reproduction or automatic admission of the other two family conditions.","Partials require ROOT manual honest reporting; completed builder rejects them."])
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf-8",newline="\n") as f:json.dump(receipt,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
    return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--execute",action="store_true");p.add_argument("--root-triggered",action="store_true")
    for name in ("public-root","cache-root","plan-sha","fit-spec-sha","dispatch-sha","post-sha","post-source-sha","audit-sha","output"):p.add_argument("--"+name)
    v=p.parse_args(argv)
    if not v.execute:
        print(json.dumps(dict(status="PREPARATION_ONLY_NOT_EXECUTED",input_reads=0,files_written=0,scientific_or_child_operations=0,numpy_imported="numpy" in sys.modules,torch_imported="torch" in sys.modules)));return 0
    if not v.root_triggered or any(getattr(v,k) is None for k in ("public_root","cache_root","plan_sha","fit_spec_sha","dispatch_sha","post_sha","post_source_sha","audit_sha","output")):p.error("ROOT exact future actual SHA bindings/current source/fresh output required")
    r=execute(v.public_root,v.cache_root,{k:getattr(v,k) for k in ("plan_sha","fit_spec_sha","dispatch_sha","post_sha","post_source_sha")},v.audit_sha,v.output)
    print(json.dumps(dict(status=r["status"],files_written=1)));return 0 if r["status"]=="PASS" else 1


if __name__=="__main__":raise SystemExit(main())


