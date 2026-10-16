"""ROOT-only stdlib audit of the two new Beijing core contrasts.

Default0 I/O. No model/raw/NPZ/checkpoint decoding, process query or dispatch.
Partial evidence produces NOT_COMPLETE; original N4 B2 evidence is reused.
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

NAME="priority_valid_minima_20261016_v1"
D2="priority_pause_validate_20261016_v1/FOLLOWUP_ROOT_v1"
FIT=D2+"/BEIJING_CORE_CONTRAST_DISPATCH_PREP_v1"
POST=D2+"/BEIJING_CORE_CONTRAST_POST_PREP_v1"
GUARD_REL="postT_cpu_lag_preparation_20261016_v1/cpu_group_supervisor.py"
GUARD_SHA="82a837ea0cfee8b52765f4ac1c7325d564e886c9e94857bd0f957e4a40c75f8b"
FIT_SOURCE_SHA="3e48df1d2c6ee9cfdca0d2be425486c92e829d7a6682f1e2e743c55d023ee882"
FIT_SPEC_SHA="3faf463d4c89269a47d20e83929d76b5ae7a23ae6c7766377aa2e92e9d765390"
POST_SOURCE_SHA="b4b7f4cdb00835b333c1b491bf9c514b4ba032a6370ea4d1dbf67a871708f192"
OLD_AUDIT_SHA="72c8174301959f0c9de7eeeae34212b54452f7dda95192785898b7a525c04a68"
OLD_AUDIT=NAME+"/INDEPENDENT_ARITHMETIC_REVIEW_ROOT_v1.json"
OLD_REPORT_SHA="563f1140b673af0e7a187b1a05338d5e5d1b32b22de7eb7c7ff225c1d0ea74cc"
OLD_BINDING_SHA="ce51ecfb61308a8833013fc2be2a2124b51ad2f8b5409904e7964616503c1f6d"
OLD_MANIFEST_SHA="2462712b8596c76400980a20733705eab68950da572b7c6dc4e79da95e912a73"
OLD_RUN="postT_cpu_beijing_E1_B2_k6_s92601_required_v1"
F_SHA="c2b4edf7db22d3f18e58ca79299a5c7e864e9b8f277e4e29e2f88d07dbb35874"
MATH_SOURCE_SHA="933883a052b02e6a41ccdadaa5ec45e80b977dbfaada007607ac1093f518dd04"
BUILDER_SOURCE_SHA="8bb730a55e0f0f924de0616bf917ca4b47e4117503027cfc4656c6ef502750dd"
STAGE_FIELDS=tuple("stage_"+x+"_seconds" for x in ("CPU_preprocess","host_to_device","P_E","F","D","inverse_and_CPU"))
TARGETS=(
 dict(role="B0_K6",namespace="priority_multik_independent_20261016_v1",run_id="postT_cpu_beijing_E4_INDEPENDENT_B0_k6_s92601_v1",family="B0",K=6,fit_id="beijing_core_independent_b0_k6_fit_root_v1",runner="multik_independent_runner.py",source_sha="a1a58875ccf1715d51dce036618768b9521a623f7adf0b78ad6d43562495125d",plan_sha="55e40b087c2a1104e5e290053f61ab4c6fc4dd5d88ab8f446ddc27c511d3eba6",closure="original_source39_immutable9_registered3_and_model_pins_preserved"),
 dict(role="B2_K11",namespace="priority_fullC_b2_20261016_v1",run_id="postT_cpu_beijing_E1_B2_k11_s92601_required_v1",family="B2",K=11,fit_id="beijing_core_b2_fullC_k11_fit_root_v1",runner="fullC_b2_runner.py",source_sha="e8fbb4a57144c470b8b49ae147a84f45689f34b3ba78d46325407606e7e94e7e",plan_sha="78b0a614d0b96c342a6996673f0cd735733718e259072f35bde9b983f7d813aa",closure="original_source39_immutable9_registered2_GPU_refs_and_model_pins_preserved"))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""):h.update(chunk)
    return h.hexdigest()


def load_helpers(public):
    modules=[]
    for filename,digest in (("independent_beijing_audit.py",MATH_SOURCE_SHA),("build_beijing_report.py",BUILDER_SOURCE_SHA)):
        path=Path(public)/NAME/filename
        if sha(path)!=digest:raise ValueError("Frozen stdlib audit/report helper changed")
        spec=importlib.util.spec_from_file_location("core_pinned_"+filename[:-3],path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);modules.append(module)
    return tuple(modules)


def expected_steps():
    return {t["fit_id"].replace("_fit_root_v1", "_"+mode+("_b"+str(batch) if batch else "")+"_root_v1"):(t["run_id"],mode,batch)
        for t in TARGETS for mode,batch in (("rescore",None),("replay",None),("cost",1),("cost",8))}


def validate_post(value):
    rows=value.get("steps",[])
    if (value.get("status")!="COMPLETE_TWO_VALIDATIONS_AND_FOUR_COSTS" or value.get("skipped")!=[] or value.get("error") is not None
            or value.get("pins_before_after_unchanged") is not True or value.get("changed_or_unreadable_pinned_paths")!=[] or len(rows)!=8
            or {r.get("id") for r in rows}!=set(expected_steps()) or any(r.get("success") is not True or r.get("strong_observed_tree_closure_verified") is not True for r in rows)):
        raise ValueError("Eight exact completed POST steps required; partial needs ROOT manual partial reporting")


def aggregate_trials(rows,identifier,batch,k):
    if k not in (6,11) or batch not in (1,8):raise ValueError("Only admitted K6/K11 B1/B8 trials")
    phases=dict(warmup=10,timed=20,stage_probe=20)
    expected={(phase,i) for phase,n in phases.items() for i in range(n)};seen={};hashes=set()
    fields={"run_id","phase","trial","order_position","batch_size","whole_seconds","F_logical_streams","F_physical_streams","F_calls","prediction_sha256","row_microbatch","stage_instrumentation",*STAGE_FIELDS}
    for row in rows:
        key=(row.get("phase"),int(row.get("trial",-1)));stage=key[0]=="stage_probe"
        if (set(row)!=fields or key not in expected or key in seen or row["run_id"]!=identifier or int(row["batch_size"])!=batch or int(row["order_position"])!=0
                or (int(row["F_logical_streams"]),int(row["F_physical_streams"]),int(row["F_calls"]))!=(batch*k,batch*k,1)
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
    return dict(trial_rows=50,phase_rows=phases,F_logical_streams=50*batch*k,F_physical_streams=50*batch*k,F_calls=50,unique_prediction_hashes=len(hashes),summaries=summaries)


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


def validate_validation(value,mode,artifact,manifest_sha,k):
    if value.get("artifact_sha256")!=artifact or value.get("manifest_sha256")!=manifest_sha:raise ValueError("Selected artifact/manifest mismatch in validation")
    if mode=="rescore":
        if (value.get("status")!="PASS_INDEPENDENT_FULL_V_FP64" or value.get("saved_predictions_rescored") is not True or value.get("new_model_inference") is not False or value.get("n_observed")!=20539 or value.get("full_V_origins")!=43):raise ValueError("Actual fullV FP64 saved-array rescore required")
    elif mode=="replay":
        if (value.get("status")!="PASS_FRESH_CPU6_REPLAY" or value.get("executed_now") is not True or value.get("microbatches")!=[4,2] or len(value.get("origins",[]))!=6
                or value.get("F_counters")!=dict(streams=6*k,physical_streams=6*k,physical_calls=2) or value.get("atol")!=1e-5 or value.get("rtol")!=1e-4):raise ValueError("Actual CPU6/4+2 dynamic-K replay required")
    else:raise ValueError("Only rescore/replay validation supported")


def require_count_binding(new,baseline):
    if not new or any(row["band_count_binding_sha256"]!=baseline["band_count_binding_sha256"] for row in new):raise ValueError("Exact paired observed band counts differ from reused N4 baseline")


def collect(inputs,h,b,dispatch_sha,post_sha):
    for root,name,digest in (("public",NAME+"/independent_beijing_audit.py",MATH_SOURCE_SHA),("public",NAME+"/build_beijing_report.py",BUILDER_SOURCE_SHA),
            ("cache",FIT+"/RUN_ROOT_DRAFT_v1.ps1",FIT_SOURCE_SHA),("cache",FIT+"/SPEC_ROOT_v1.json",FIT_SPEC_SHA),("cache",POST+"/RUN_ROOT_DRAFT_v2.ps1",POST_SOURCE_SHA),("cache",GUARD_REL,GUARD_SHA)):
        inputs.capture(root,name,digest)
    dispatch=inputs.json("cache",FIT+"/RESULT.json",dispatch_sha);post=inputs.json("cache",POST+"/RESULT.json",post_sha);validate_post(post)
    if (dispatch.get("status")!="COMPLETE_TWO_FITS_AND_FULL_V" or dispatch.get("strong_observed_tree_closure_verified") is not True or dispatch.get("pins_before_after_unchanged") is not True or dispatch.get("changed_or_unreadable_pinned_paths")!=[] or dispatch.get("error") is not None or len(dispatch.get("conditions",[]))!=2 or {r.get("run_id") for r in dispatch["conditions"]}!={t["run_id"] for t in TARGETS}):raise ValueError("Both fits must actually complete; partial needs separate ROOT manual report")
    oldplan=inputs.json("public","postT_V_only_cpu_multik_20261016_v1/PLAN.json","2051573b8dee48200eb34e96e37e7af0d197442619ea1ba52a36a80ad4e27aec")
    if len(oldplan["source39"])!=39 or len(oldplan["immutable_public"])!=9:raise ValueError("Original48 bindings required")
    for name,digest in {**oldplan["source39"],**oldplan["immutable_public"]}.items():inputs.capture("public",name,digest)
    oldaudit=inputs.json("public",OLD_AUDIT,OLD_AUDIT_SHA)
    if oldaudit.get("status")!="PASS" or oldaudit.get("schema")!="INDEPENDENT_N4_BEIJING_AUDIT_v1" or oldaudit.get("input_SHA_before_after_equal") is not True:raise ValueError("Exact previously PASS N4 audit required")
    inputs.capture("public",NAME+"/RESULTS_WITH_COST_v1.csv",OLD_REPORT_SHA);inputs.capture("public",NAME+"/REPORT_INPUT_BINDINGS_v1.json",OLD_BINDING_SHA)
    baseline=next(a for a in oldaudit["accuracy"] if a["run_id"]==OLD_RUN)
    oldbindings={(r["root"],r["relative_path"]):r["sha256"] for r in oldaudit["inputs"]}
    baseline_folder=NAME+"/predictions/"+OLD_RUN+"/V_select";manifest=inputs.json("public",baseline_folder+"/PREDICTION_MANIFEST.json",OLD_MANIFEST_SHA);b.validate_cohort(manifest)
    baseline_costs=[a for a in oldaudit["cost_conditions"] if a["run_id"]==OLD_RUN]
    if len(baseline_costs)!=2:raise ValueError("Two existing N4 B2 costs required")
    oldrun=NAME+"/runs/"+OLD_RUN
    oldfit=inputs.json("public",oldrun+"/BASE_FIT_RECEIPT.json",oldbindings[("public",oldrun+"/BASE_FIT_RECEIPT.json")])
    oldsearch=inputs.json("public",oldrun+"/search/RECEIPT.json",oldbindings[("public",oldrun+"/search/RECEIPT.json")]);oldrefit=inputs.json("public",oldrun+"/refit/RECEIPT.json",oldbindings[("public",oldrun+"/refit/RECEIPT.json")])
    b.validate_training(oldfit,oldsearch,oldrefit,OLD_RUN,"B2")
    rows=[report_row("B2_K6",OLD_RUN,6,manifest,oldfit,oldsearch,oldrefit,True)];physical={};cost_records=[];accuracy=[];intervals=[];cost_intervals=[];guards=[]
    for cost in baseline_costs:
        batch=cost["batch"];folder=cost["public_diagnostic_relative_path"]
        receipt=inputs.json("public",folder+"/COST_RECEIPT.json",oldbindings[("public",folder+"/COST_RECEIPT.json")]);inputs.capture("public",folder+"/RESULT.json",oldbindings[("public",folder+"/RESULT.json")]);inputs.capture("public",folder+"/COST_TRIALS.csv.gz",receipt["trials_sha256"])
        if receipt["accuracy_manifest_sha256"]!=OLD_MANIFEST_SHA or receipt["artifact_sha256"]!=oldfit["artifact_sha256"] or receipt["device"]!="cpu" or receipt["CPU_threads"]!=2:raise ValueError("Reused CPU baseline cost/accuracy artifact differs")
        physical[batch]=(receipt["physical_input_sha256"],receipt["observation_mask_sha256"])
        prior=next(g for g in oldaudit["recorded_guards"] if g["id"]==Path(folder).name)
        add_cost_row(rows[0],batch,receipt,prior["sampled_peak_RSS_bytes"])
    step_map={r["id"]:r for r in post["steps"]}

    def bound(target,tag,mode,status,batch=None):
        folder=target["namespace"]+"/diagnostics/"+tag;outer=inputs.json("public",folder+"/RESULT.json")
        if (outer.get("run_id")!=target["run_id"] or outer.get("mode")!=mode or outer.get("status")!=status or outer.get("plan_sha256")!=target["plan_sha"] or outer.get("error_class") is not None or outer.get("postflight_error_class") is not None or outer.get(target["closure"]) is not True or outer.get("new_T_target_operations")!=0):raise ValueError("Exact clean new outer/source closure required")
        if mode=="fit":
            gp,sp=FIT+"/GUARD_ROOT_v1/RESULT.json",FIT+"/SPEC_ROOT_v1.json";record=next(r for r in dispatch["conditions"] if r["run_id"]==target["run_id"])
            if record.get("id")!=tag or record.get("outer_sha256")!=inputs.files[("public",folder+"/RESULT.json")] or record.get("worker_science_complete") is not True:raise ValueError("Actual two-fit dispatch outer binding differs")
        else:
            gp=target["namespace"]+"/supervisors/"+tag+"/RESULT.json";sp=POST+"/SPEC_"+tag+".json";record=step_map[tag]
            if record.get("run_id")!=target["run_id"] or record.get("mode")!=mode or record.get("batch")!=batch or record.get("guard_exit_code")!=0 or record.get("outer_sha256")!=inputs.files[("public",folder+"/RESULT.json")]:raise ValueError("Actual POST selected outer identity differs")
        guard=inputs.json("cache",gp);spec=inputs.json("cache",sp,guard["spec_file_sha256"])
        candidates=[r for r in spec["processes"] if r["id"]==tag]
        if len(candidates)!=1:raise ValueError("One exact recorded command required")
        cmd=candidates[0]["command"]
        for flag,value in (("--run-id",target["run_id"]),("--mode",mode),("--plan-sha",target["plan_sha"])):
            if cmd.count(flag)!=1 or cmd[cmd.index(flag)+1]!=value:raise ValueError("Actual guard argv scientific identity differs")
        started=inputs.json("public",folder+"/STARTED.json");child=validate_guard(guard,spec,inputs.files[("cache",sp)],tag,cmd,started)
        if mode!="fit" and (record["guard_sha256"]!=inputs.files[("cache",gp)] or record["spec_sha256"]!=inputs.files[("cache",sp)]):raise ValueError("POST exact guard/SPEC SHA differs")
        if mode=="cost":
            for flag in ("--quiet-proof","--quiet-sha"):
                if cmd.count(flag)!=1:raise ValueError("Actual quiet proof command missing")
            qpath=Path(cmd[cmd.index("--quiet-proof")+1]).resolve();base=inputs.roots["cache"]
            if not qpath.is_relative_to(base):raise ValueError("Quiet proof must stay under cache")
            qsha=cmd[cmd.index("--quiet-sha")+1];quiet=inputs.json("cache",qpath.relative_to(base).as_posix(),qsha)
            if quiet.get("status")!="QUIET_CPU_COST_WINDOW" or quiet.get("active_model_workers")!=[] or quiet.get("active_cost_workers")!=[] or not 0<=guard["start_unix"]-quiet["captured_unix"]<=120 or outer["scientific_result"].get("quiet_proof_sha256")!=qsha:raise ValueError("Actual fresh quiet/input-based CPU cost binding differs")
        interval=dict(id=tag,start=guard["start_unix"],end=guard["end_unix"]);(cost_intervals if mode=="cost" else intervals).append(interval)
        guards.append(dict(id=tag,cache_relative_path=gp,guard_sha256=inputs.files[("cache",gp)],selected_returncode=child["returncode"],sampled_peak_RSS_bytes=child["sampled_peak_RSS_bytes"],observed_owned_identities=len(child["owned_processes"]),current_identity_rechecked=False))
        return folder,outer,child

    for target in TARGETS:
        ns,identifier,k=target["namespace"],target["run_id"],target["K"]
        plan=inputs.json("public",ns+"/PLAN.json",target["plan_sha"]);inputs.capture("public",ns+"/"+target["runner"],target["source_sha"])
        if plan.get("status")!="SEALED_BY_ROOT" or plan.get("ROOT_initial_admitted_run_ids")!=[identifier] or plan.get("runner_sha256")!=target["source_sha"] or plan.get("new_T_target_operations_allowed") is not False:raise ValueError("Current exact ROOT admitted plan required")
        for path,digest in plan["dependencies"].items():inputs.capture("public",path,digest)
        spec=next(r["spec"] for r in plan["conditions"] if r["run_id"]==identifier)
        run=ns+"/runs/"+identifier;pred=ns+"/predictions/"+identifier+"/V_select"
        manifest=inputs.json("public",pred+"/PREDICTION_MANIFEST.json");b.validate_cohort(manifest);score=b.score_csv(inputs,pred,manifest);msha=inputs.files[("public",pred+"/PREDICTION_MANIFEST.json")]
        receipt=inputs.json("public",run+"/BASE_FIT_RECEIPT.json");search=inputs.json("public",run+"/search/RECEIPT.json");refit=inputs.json("public",run+"/refit/RECEIPT.json");b.validate_training(receipt,search,refit,identifier,target["family"])
        want=dict(run_id=identifier,family=target["family"],seed=92601,device="cpu",CPU_threads=2,artifact_sha256=receipt["artifact_sha256"],config_sha=b.fingerprint(spec),source_sha=b.fingerprint(dict(runner=target["source_sha"],**plan["dependencies"])),model_sha=b.fingerprint(dict(artifact_sha=receipt["artifact_sha256"],model_state_sha=receipt["model_state_sha"])),F_counters=dict(streams=43*k,physical_streams=43*k,physical_calls=11))
        if any(manifest.get(key)!=value for key,value in want.items()):raise ValueError("New fullCPU V43 selected bundle/source/K identity differs")
        initializations=receipt.get("initializations",[]);kind="identity_at_K_equals_C" if k==11 else "A_fitted_centered_PCA"
        if len(initializations)!=2 or {r["fit_end"] for r in initializations}!={10519,12622} or any(r.get("basis_initialization")!=kind or not h.is_sha(r.get("basis_sha256")) or not h.is_sha(r.get("center_sha256")) for r in initializations):raise ValueError("Original single-K stage-specific initialization evidence differs")
        _,outer,_=bound(target,target["fit_id"],"fit","COMPLETE_FIT_AND_FULL_V")
        if outer["scientific_result"].get("fit_receipt")!=receipt or outer["scientific_result"].get("V_score")!=manifest["score"] or outer["scientific_result"].get("full_V_completed") is not True or outer["scientific_result"].get("F_final_state_sha256")!=F_SHA:raise ValueError("Actual fit/fullV/frozen F mismatch")
        proof_shas={};rescored=None
        for mode,status in (("rescore","PASS_INDEPENDENT_FULL_V_FP64"),("replay","PASS_FRESH_CPU6_REPLAY")):
            tag=target["fit_id"].replace("_fit_root_v1","_"+mode+"_root_v1");folder,outer,_=bound(target,tag,mode,status);science=outer["scientific_result"]
            validate_validation(science,mode,receipt["artifact_sha256"],msha,k);saved=inputs.json("public",folder+("/RESCORE_RECEIPT.json" if mode=="rescore" else "/REPLAY_RECEIPT.json"))
            if any(science.get(key)!=value for key,value in saved.items()):raise ValueError("Actual validation receipt differs from outer")
            if mode=="rescore":rescored=science
            elif science.get("F_final_state_sha256")!=F_SHA:raise ValueError("Fresh CPU6 frozen F proof missing")
            proof_shas[mode]=inputs.files[("public",folder+"/RESULT.json")]
        stats=h.gzip_rows(inputs,"public",pred+"/PUBLIC_ERROR_STATS.csv.gz",1419)
        aggregate=h.aggregate_statistics(stats,identifier,oldplan["task_contract"]["columns"],h.ORIGINS,h.BANDS,20539);deltas=h.compare_accuracy(aggregate,manifest,score,rescored)
        count_sha=b.fingerprint(aggregate.pop("count_by_key"));accuracy.append(dict(run_id=identifier,**aggregate,metric_deltas=deltas,band_count_binding_sha256=count_sha,manifest_sha256=msha,artifact_sha256=receipt["artifact_sha256"]))
        row=report_row(target["role"],identifier,k,manifest,receipt,search,refit,False)
        for batch in (1,8):
            tag=target["fit_id"].replace("_fit_root_v1","_cost_b"+str(batch)+"_root_v1");folder,outer,child=bound(target,tag,"cost","COMPLETE_PROTOCOL",batch)
            cost=inputs.json("public",folder+"/COST_RECEIPT.json");trials=h.gzip_rows(inputs,"public",folder+"/COST_TRIALS.csv.gz",50,cost["trials_sha256"])
            validate_cost(cost,identifier,k,batch,receipt["artifact_sha256"],msha,proof_shas)
            if any(outer["scientific_result"].get(key)!=value for key,value in cost.items()) or outer["scientific_result"].get("F_final_state_sha256")!=F_SHA:raise ValueError("Actual cost/frozen F receipt mismatch")
            pair=(cost["physical_input_sha256"],cost["observation_mask_sha256"])
            if pair!=physical[batch]:raise ValueError("New and reused baseline physical cost input/mask differ")
            totals=aggregate_trials(trials,identifier,batch,k);h.compare_summaries(totals["summaries"],cost["summaries"])
            cost_records.append(dict(run_id=identifier,batch=batch,public_diagnostic_relative_path=folder,**totals,physical_input_sha256=pair[0],observation_mask_sha256=pair[1],memory_scope=cost["memory"]))
            add_cost_row(row,batch,cost,child["sampled_peak_RSS_bytes"])
        rows.append(row)
    require_count_binding(accuracy,baseline);h.validate_intervals(cost_intervals,intervals)
    rows.sort(key=lambda r:("B0_K6","B2_K6","B2_K11").index(r["role"]))
    return dict(rows=rows,accuracy=accuracy,reused_baseline=dict(run_id=OLD_RUN,accuracy_recomputed_now=False,old_audit_sha256=OLD_AUDIT_SHA,old_manifest_sha256=OLD_MANIFEST_SHA,band_count_binding_sha256=baseline["band_count_binding_sha256"],cost_conditions_reused=2),cost_conditions=cost_records,recorded_guards=guards,totals=dict(new_accuracy_conditions=2,reused_accuracy_conditions=1,new_public_stats_rows=2838,unique_observed_V_labels=20539,new_trial_rows=200,new_cost_conditions=4,reused_cost_conditions=2,new_summary_groups=8,new_summary_numeric_fields_checked=40,F_logical_streams=sum(r["F_logical_streams"] for r in cost_records),F_physical_streams=sum(r["F_physical_streams"] for r in cost_records),F_calls=sum(r["F_calls"] for r in cost_records)))


def validate_cost(value,identifier,k,batch,artifact,manifest,proofs):
    want=dict(status="COMPLETE_PROTOCOL",run_id=identifier,device="cpu",CPU_threads=2,batch_size=batch,warmups=10,timed_trials=20,stage_trials=20,actual_trial_rows=50,artifact_sha256=artifact,accuracy_manifest_sha256=manifest,validation_result_sha256=proofs,F_logical_streams_per_trial=batch*k,F_calls_per_trial=1,cached_Z_or_forecast_reads=0,physical_output_and_inverse_dtype="float64",stage_probe_separate_intrusive_scope=True,training_peak_included=False,new_T_target_operations=0)
    if any(value.get(key)!=expected for key,expected in want.items()) or value.get("origins")!=list(range(14726,16783,48))[:batch] or value.get("F_counters")!=dict(streams=50*batch*k,physical_streams=50*batch*k,physical_calls=50):raise ValueError("Whole CPU2/B1B8 dynamic-K full physical protocol differs")
    if value.get("memory",{}).get("foundation_instances_created")!=1:raise ValueError("Actual frozen F residency instance required; B0 is not noBolt")


def report_row(role,identifier,k,manifest,receipt,search,refit,reused):
    row=dict(role=role,run_id=identifier,K=k,MSE=manifest["score"]["mse"],MAE=manifest["score"]["mae"],accuracy_reused=reused,CPU_threads=2,accuracy_device="cpu",training_device="cpu",search_max_epoch=3,selected_refit_epoch=receipt["selected_epoch"],search_seconds=search["seconds"],refit_seconds=refit["seconds"],training_status=receipt["training_status"],observed_labels=20539,V_origins=43,F_used=True,frozen_F_sha256=F_SHA,F_resident_bytes=None,F_resident_bytes_scope="NOT_SEPARATELY_RECORDED",initialization="identity_at_K_equals_C" if k==11 else "A_fitted_centered_PCA",parameter_count_scope="NOT_REMEASURED",source_of_values="REUSED_N4_VERIFIED" if reused else "NEW_TWO_CORE_CONDITIONS")
    for key in ("search_updates","refit_updates","search_exposures","refit_exposures"):row[key]=receipt[key]
    return row


def add_cost_row(row,batch,cost,peak):
    timed=next(s for s in cost["summaries"] if s["phase"]=="timed");memory=cost["memory"]
    row.update({f"B{batch}_median_seconds":timed["median_seconds"],f"B{batch}_p90_seconds":timed["p90_seconds"],f"B{batch}_CPU_weights_MiB":memory["resident_weights_cpu_MiB"],f"B{batch}_observed_tree_peak_RSS_bytes":peak,f"B{batch}_F_instances":memory["foundation_instances_created"],f"B{batch}_F_streams_per_trial":cost["F_logical_streams_per_trial"],f"B{batch}_F_calls_per_trial":cost["F_calls_per_trial"]})


def execute(public,cache,dispatch_sha,post_sha,audit_sha,output):
    output=Path(output).absolute()
    if output.exists():raise FileExistsError("Existing audit output preserved before inputs")
    public,cache=Path(public).resolve(),Path(cache).resolve()
    if not output.resolve().is_relative_to(public/NAME) or output.suffix!=".json":raise ValueError("Fresh explicit JSON output under N4 namespace required")
    if any(not isinstance(s,str) or len(s)!=64 or any(c not in "0123456789abcdef" for c in s) for s in (dispatch_sha,post_sha,audit_sha)):raise ValueError("ROOT exact actual dispatch/POST/current source SHA required")
    if Path(__file__).resolve()!=public/NAME/"audit_core_contrast.py" or sha(__file__)!=audit_sha:raise ValueError("Current ROOT-pinned auditor source required")
    h,b=load_helpers(public);inputs=b.Inputs(public,cache);inputs.capture("public",NAME+"/audit_core_contrast.py",audit_sha);result={};error=None
    try:result=collect(inputs,h,b,dispatch_sha,post_sha)
    except (ValueError,FileNotFoundError,KeyError,TypeError,OverflowError,csv.Error,OSError,StopIteration) as exc:error=dict(error_class=type(exc).__name__,reason=str(exc) if isinstance(exc,ValueError) else "Required relative evidence absent or incomplete; ROOT manual partial report needed")
    try:preserved=inputs.preserved()
    except OSError:preserved=False
    if not preserved:error=error or dict(error_class="InputPreservationFailed",reason="Captured relative evidence changed")
    receipt=dict(schema="INDEPENDENT_BEIJING_CORE_CONTRAST_v1",status="PASS" if error is None and preserved else "NOT_COMPLETE_AUDIT_EVIDENCE_REJECTED",audit_source_sha256=audit_sha,fit_dispatch_result_sha256=dispatch_sha,post_result_sha256=post_sha,input_SHA_before_after_equal=preserved,inputs=[dict(root=root,relative_path=name,sha256=digest) for (root,name),digest in sorted(inputs.files.items())],error=error,**result,
        accuracy_comparison_tolerance=dict(atol=1e-9,rtol=1e-7),cost_comparison_policy="Exact original saved-vs-stdlib fsum/median/min/max/linear-p90 policy; any ULP difference means NOT_COMPLETE, no new tolerance.",new_models_NPZ_raw_T_numeric_training_inference_cost_or_process_operations=0,whole_MASTER_complete=False,independent_generalization_claimed=False,
        limitations=["One seed92601, repeated development V, limited search3; selected refit lengths/exposures/capacity may differ.","All three rows use the same20539 observed labels, not independent new generalization trials.","B0 and B2 both use frozen Bolt; no new noBolt contrast.","K6 PCA vs K11 identity initialization and different K/P/parameters are not a pure information-loss or isolated-encoder effect.","F-only resident bytes not separately recorded; whole CPU weights and foundation instance count retain their scopes.","Prior actual process closure is bound from receipts; no current PID query or unseen-descendant absence proof.","Any partial requires ROOT manual partial reporting rather than a completed three-row builder."])
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf-8",newline="\n") as f:json.dump(receipt,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
    return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--execute",action="store_true");p.add_argument("--root-triggered",action="store_true")
    for flag in ("public-root","cache-root","dispatch-sha","post-sha","audit-sha","output"):p.add_argument("--"+flag)
    a=p.parse_args(argv)
    if not a.execute:
        print(json.dumps(dict(status="PREPARATION_ONLY_NOT_EXECUTED",input_reads=0,files_written=0,scientific_or_child_operations=0,numpy_imported="numpy" in sys.modules,torch_imported="torch" in sys.modules)));return 0
    if not a.root_triggered or any(getattr(a,k) is None for k in ("public_root","cache_root","dispatch_sha","post_sha","audit_sha","output")):p.error("ROOT trigger/exact actual dispatch+POST+current source SHA/roots/fresh output required")
    try:r=execute(a.public_root,a.cache_root,a.dispatch_sha,a.post_sha,a.audit_sha,a.output)
    except (ValueError,FileNotFoundError,FileExistsError):print(json.dumps(dict(status="NOT_COMPLETE_AUDIT_EVIDENCE_REJECTED")));return 1
    print(json.dumps(dict(status=r["status"],files_written=1)));return 0 if r["status"]=="PASS" else 1


if __name__=="__main__":raise SystemExit(main())
