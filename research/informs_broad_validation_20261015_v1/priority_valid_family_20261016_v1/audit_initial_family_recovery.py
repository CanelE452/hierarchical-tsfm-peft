"""CreateNew audit of original validation evidence and separate recovery costs.

The original PARTIAL receipt and four false wrapper rows are never rewritten.
Default invocation performs no input reads, science, process queries or writes.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

NAME="priority_valid_family_20261016_v1"
D2="priority_pause_validate_20261016_v1/FOLLOWUP_ROOT_v1"
FIT=D2+"/FAMILY_INITIAL_DISPATCH_PREP_v1"
POST=D2+"/FAMILY_INITIAL_POST_PREP_v1"
RECOVERY=D2+"/FAMILY_INITIAL_POST_RECOVERY_v1"
ORIGINAL_AUDITOR_SHA="f92a3a7b4bc03677f67d3bb7c42758765a95f49a378e2dd9d768ffc7f4de6241"
ORIGINAL_POST_SHA="28911ed451a95ce4af157dd863914f9078f5b0f3f11ab82d0ee5ae2efd6d6080"
ORIGINAL_POST_SOURCE_SHA="57cf98eda8f8d1b2c6303ba336860733a41454c470dd28c0e18485d81e257982"
B2_BASE="postT_V_only_cpu_lag_20261016_v1/runs/postT_cpu_jena_E3_B2_k6_s92601_v1/BASE_FIT_RECEIPT.json"
B2_BASE_SHA="0100fa0551f4865cfb047716ed5ab12bea3315612f3d11126856bdb6ebed2acf"
RUNS=("postT_cpu_jena_E1_LORA_R8_k0_s92601_family_v1","postT_cpu_jena_E1_B4_k6_s92601_family_v1")
ACTIVITY=dict(selected_epoch=1,search_updates=1863,refit_updates=752,search_exposures=29769,refit_exposures=12020,search_fit_end=10482,refit_fit_end=12579)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):h.update(block)
    return h.hexdigest()


def digest(value):return isinstance(value,str) and len(value)==64 and all(c in "0123456789abcdef" for c in value)


def normalized_relative(value):
    if not isinstance(value,str) or not value or value.startswith(("/","\\")) or ":" in value:raise ValueError("Root-relative metadata path required")
    parts=value.replace("\\","/").split("/")
    if any(p in ("",".","..") for p in parts):raise ValueError("Metadata path must not escape its root")
    return "/".join(parts)


def steps_expected():
    return {f"family_initial_jena_{stem}_{mode}{'_b'+str(batch) if batch else ''}_root_v1":(run,mode,batch)
        for stem,run in zip(("lora_r8","b4_k6"),RUNS) for mode,batch in (("rescore",None),("replay",None),("cost",1),("cost",8))}


def validate_recovery(original,recovery,fit_guard_sha):
    expected=steps_expected();validation={tag for tag,(_,mode,_) in expected.items() if mode!="cost"};costs=set(expected)-validation
    oldrows=original.get("steps",[])
    if (original.get("status")!="PARTIAL_VALIDATIONS_COSTS_PRESERVED" or original.get("error") is not None
            or original.get("fit_guard_sha256") is not None or original.get("pins_before_after_unchanged") is not True
            or original.get("changed_or_unreadable_pinned_paths")!=[] or len(oldrows)!=4 or {r.get("id") for r in oldrows}!=validation
            or any(r.get("success") is not False or r.get("guard_exit_code")!=0 or r.get("strong_observed_tree_closure_verified") is not True for r in oldrows)):
        raise ValueError("The exact original PARTIAL/four false rows/null fit guard must remain preserved")
    skips=original.get("skipped",[])
    if len(skips)!=2 or {r.get("run_id") for r in skips}!=set(RUNS) or any(r.get("reason")!="ACTUAL_RESCORE_REPLAY_NOT_BOTH_PASS_COST_NOT_ADMITTED" for r in skips):raise ValueError("Original two skipped-cost records must remain intact")
    rows=recovery.get("steps",[])
    if (recovery.get("schema")!="INITIAL_FAMILY_POST_MIXED_SLASH_RECOVERY_v1" or recovery.get("status")!="COMPLETE_RECOVERED_FOUR_VALIDATIONS_AND_FOUR_COSTS"
            or recovery.get("original_POST_sha256")!=ORIGINAL_POST_SHA or recovery.get("original_POST_source_sha256")!=ORIGINAL_POST_SOURCE_SHA
            or recovery.get("original_POST_status")!=original["status"] or recovery.get("original_wrapper_false_rows_preserved") is not True
            or recovery.get("original_fit_guard_sha256_was_null") is not True or recovery.get("actual_fit_guard_sha256")!=fit_guard_sha
            or recovery.get("error") is not None or recovery.get("pins_before_after_unchanged") is not True or recovery.get("changed_or_unreadable_pinned_paths")!=[]
            or recovery.get("skipped",[])!=[] or len(rows)!=8 or {r.get("id") for r in rows}!=set(expected)):
        raise ValueError("A separate complete recovery receipt bound to the original bytes/actual fit guard is required")
    for key in ("fit_dispatch_result_sha256","fit_dispatch_SPEC_sha256","plan_sha256"):
        if recovery.get(key)!=original.get(key):raise ValueError("Recovery must bind the same original fit and PLAN")
    oldmap={r["id"]:r for r in oldrows};result={}
    for row in rows:
        tag=row["id"];run,mode,batch=expected[tag]
        if (row.get("run_id"),row.get("mode"),row.get("batch"))!=(run,mode,batch) or row.get("success") is not True or row.get("guard_exit_code")!=0 or row.get("strong_observed_tree_closure_verified") is not True:raise ValueError("Every combined step must retain exact actual identity and successful guard closure")
        if any(not digest(row.get(key)) for key in ("spec_sha256","guard_sha256","outer_sha256","STARTED_sha256")):raise ValueError("Exact step/SPEC/guard/STARTED SHA bindings required")
        source="RECOVERY_NEW_COST" if mode=="cost" else "ORIGINAL_57CF_VALIDATION_RECONCILED"
        spec_root=RECOVERY if mode=="cost" else POST
        paths=dict(spec_cache_relative_path=spec_root+"/SPEC_"+tag+".json",guard_cache_relative_path=NAME+"/supervisors/"+tag+"/RESULT.json",outer_public_relative_path=NAME+"/diagnostics/"+tag+"/RESULT.json")
        if row.get("evidence_source")!=source or any(normalized_relative(row.get(key))!=value for key,value in paths.items()):raise ValueError("Original validation and new-cost evidence namespaces must stay separate")
        if mode!="cost" and (row.get("original_wrapper_success") is not False or any(row.get(key)!=oldmap[tag].get(key) for key in ("spec_sha256","guard_sha256","outer_sha256"))):raise ValueError("Original false wrapper and actual original proof bytes must be preserved")
        result[tag]=row
    for key,ids in (("validation_reconciliation",validation),("new_cost_steps",costs)):
        subset=recovery.get(key,[])
        if len(subset)!=4 or {r.get("id") for r in subset}!=ids or any(r!=result[r["id"]] for r in subset):raise ValueError("Recovery sections and combined step records differ")
    return result


def validate_matched_activity(rows):
    if len(rows)!=3 or any(any(row.get(key)!=value for key,value in ACTIVITY.items()) for row in rows):raise ValueError("Actual LoRA/B4/existing B2 search3 and selected-refit1 activities must match")


def load_original(public):
    path=Path(public)/NAME/"audit_initial_family.py"
    if sha(path)!=ORIGINAL_AUDITOR_SHA:raise ValueError("Frozen original auditor changed")
    spec=importlib.util.spec_from_file_location("frozen_initial_family_recovery_helpers",path);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    return a


def collect(inputs,a,h,b,bindings):
    plan_sha,fit_spec_sha,dispatch_sha=map(bindings.get,("plan_sha","fit_spec_sha","dispatch_sha"))
    if bindings["post_sha"]!=ORIGINAL_POST_SHA or bindings["post_source_sha"]!=ORIGINAL_POST_SOURCE_SHA:raise ValueError("Only the immutable original mixed-slash PARTIAL is supported")
    for root,path,want in (("public",NAME+"/audit_initial_family.py",ORIGINAL_AUDITOR_SHA),("public",NAME+"/family_minimum_runner.py",a.RUNNER_SHA),("cache",FIT+"/RUN_ROOT_DRAFT_v1.ps1",a.FIT_SOURCE_SHA),("cache",POST+"/RUN_ROOT_DRAFT_v1.ps1",ORIGINAL_POST_SOURCE_SHA),("cache",RECOVERY+"/RUN_ROOT_DRAFT_v1.ps1",bindings["recovery_source_sha"]),("cache",a.GUARD_REL,a.GUARD_SHA)):
        inputs.capture(root,path,want)
    plan=inputs.json("public",NAME+"/PLAN.json",plan_sha);old=inputs.json("public",a.OLD+"/PLAN.json",a.OLD_PLAN_SHA)
    if (plan.get("status")!="SEALED_BY_ROOT" or plan.get("runner_sha256")!=a.RUNNER_SHA or plan.get("new_T_target_operations_allowed") is not False or plan.get("protected_tail_numeric_parse_allowed") is not False or plan.get("original_family_plan_sha256")!=a.OLD_PLAN_SHA or plan.get("deadline_unix")!=1792222550):raise ValueError("Exact sealed original recipe/T/deadline contract required")
    if len(old["source39"])!=39 or len(old["immutable_public"])!=9 or len(set(old["source39"])|set(old["immutable_public"]))!=48:raise ValueError("Original source48 required")
    for path,want in {**old["source39"],**old["immutable_public"],**plan["dependencies"]}.items():inputs.capture("public",normalized_relative(path),want)
    for path,want in plan["original_family_files"].items():inputs.capture("public",a.OLD+"/"+normalized_relative(path),want)
    dispatch=inputs.json("cache",FIT+"/RESULT.json",dispatch_sha);original=inputs.json("cache",POST+"/RESULT.json",ORIGINAL_POST_SHA)
    recovery=inputs.json("cache",RECOVERY+"/RESULT.json",bindings["recovery_sha"])
    fitguard=inputs.json("cache",FIT+"/GUARD_ROOT_v1/RESULT.json");fitguardsha=inputs.files[("cache",FIT+"/GUARD_ROOT_v1/RESULT.json")]
    recovery_started=inputs.json("cache",RECOVERY+"/STARTED.json")
    started_expected=dict(script_sha256=bindings["recovery_source_sha"],original_POST_sha256=ORIGINAL_POST_SHA,plan_sha256=plan_sha,actual_fit_guard_sha256=fitguardsha,validation_reruns=0,new_training=0,maximum_new_costs=4)
    if any(recovery_started.get(key)!=value for key,value in started_expected.items()):raise ValueError("Actual recovery STARTED must bind reviewed source/original partial/same fit PLAN")
    if any(recovery.get(key)!=value for key,value in dict(new_training=0,validation_reruns=0,maximum_new_costs=4,GPU_added_seconds=0,original_worker_deadline=1792222550,whole_MASTER_complete=False,independent_generalization_claimed=False).items()):raise ValueError("Recovery must preserve original no-training/T/resource scope")
    if (dispatch.get("status")!="COMPLETE_TWO_FITS_AND_FULL_V" or dispatch.get("error") is not None or dispatch.get("strong_observed_tree_closure_verified") is not True or dispatch.get("pins_before_after_unchanged") is not True or dispatch.get("changed_or_unreadable_pinned_paths")!=[] or dispatch.get("plan_sha256")!=plan_sha or len(dispatch.get("conditions",[]))!=2 or {r["run_id"] for r in dispatch["conditions"]}!=set(RUNS)):
        raise ValueError("Two actual complete fits with original recorded resource/source closure required")
    if original.get("fit_dispatch_result_sha256")!=dispatch_sha or original.get("fit_dispatch_SPEC_sha256")!=fit_spec_sha or original.get("plan_sha256")!=plan_sha:raise ValueError("Original PARTIAL must bind the same actual fits")
    records=validate_recovery(original,recovery,fitguardsha)
    guards=[];others=[];intervals=[];rows=[];accuracy=[];cost_records=[];physical={};activity=[]

    def bound(t,tag,mode,status,batch=None):
        folder=NAME+"/diagnostics/"+tag;record=next(r for r in dispatch["conditions"] if r["run_id"]==t["run_id"]) if mode=="fit" else records[tag]
        outer=inputs.json("public",folder+"/RESULT.json",record["outer_sha256"]);science=outer.get("scientific_result") or {}
        if (outer.get("run_id")!=t["run_id"] or outer.get("mode")!=mode or outer.get("status")!=status or outer.get("plan_sha256")!=plan_sha or outer.get("error_class") is not None or outer.get("postflight_error_class") is not None or outer.get("original_source39_immutable9_family7_probe4_recipe4_and_model_pins_preserved") is not True or outer.get("new_T_target_operations")!=0):raise ValueError("Actual original runner identity/postflight differs")
        if mode=="fit":
            gp,sp=FIT+"/GUARD_ROOT_v1/RESULT.json",FIT+"/SPEC_ROOT_v1.json"
            if record.get("worker_science_complete") is not True or record.get("id")!=tag:raise ValueError("Actual fit completion required")
            guard=fitguard;spec=inputs.json("cache",sp,fit_spec_sha)
        else:
            gp,sp=normalized_relative(record["guard_cache_relative_path"]),normalized_relative(record["spec_cache_relative_path"])
            guard=inputs.json("cache",gp,record["guard_sha256"]);spec=inputs.json("cache",sp,record["spec_sha256"])
            if len(spec.get("processes",[]))!=1 or spec["processes"][0].get("max_seconds")!=600:raise ValueError("POST must retain one bounded600-second CPU worker")
        entries=[r for r in spec["processes"] if r["id"]==tag]
        if len(entries)!=1:raise ValueError("One exact actual worker argv required")
        command=entries[0]["command"]
        for flag,value in (("--run-id",t["run_id"]),("--mode",mode),("--plan-sha",plan_sha)):
            if command.count(flag)!=1 or command[command.index(flag)+1]!=value:raise ValueError("Actual worker scientific argv differs")
        for flag,want in (("--output",inputs.roots["public"]/folder),("--public-root",inputs.roots["public"]),("--cache-root",inputs.roots["cache"])):
            if command.count(flag)!=1 or Path(command[command.index(flag)+1]).resolve()!=want.resolve():raise ValueError("Actual worker root/output argv differs")
        runner=inputs.roots["public"]/NAME/"family_minimum_runner.py"
        if sum(Path(x).resolve()==runner.resolve() for x in command if not x.startswith("-"))!=1:raise ValueError("Frozen original N5 runner required")
        started=inputs.json("public",folder+"/STARTED.json",None if mode=="fit" else record["STARTED_sha256"])
        if any(started.get(key)!=value for key,value in dict(run_id=t["run_id"],mode=mode,plan_sha256=plan_sha).items()) or (mode!="fit" and started.get("deadline_unix")!=1792222550):raise ValueError("Actual STARTED scientific identity/deadline differs")
        child=a.validate_guard(guard,spec,inputs.files[("cache",sp)],tag,command,started)
        if mode=="cost":
            for kind in ("rescore","replay"):
                proof_tag=t["fit_id"].replace("_fit_root_v1","_"+kind+"_root_v1");proof=records[proof_tag]
                for flag,value in (("--"+kind+"-sha",proof["outer_sha256"]),("--"+kind+"-proof",str(inputs.roots["public"]/proof["outer_public_relative_path"]))):
                    if command.count(flag)!=1 or (Path(command[command.index(flag)+1]).resolve()!=Path(value).resolve() if flag.endswith("-proof") else command[command.index(flag)+1]!=value):raise ValueError("Actual cost argv must bind original successful validation proof bytes")
            if command.count("--batch")!=1 or command[command.index("--batch")+1]!=str(batch):raise ValueError("Actual cost batch argv differs")
            if command.count("--quiet-proof")!=1 or command.count("--quiet-sha")!=1:raise ValueError("Actual quiet argv required")
            qp=Path(command[command.index("--quiet-proof")+1]).resolve();cache=inputs.roots["cache"];qs=command[command.index("--quiet-sha")+1]
            if not qp.is_relative_to(cache):raise ValueError("Actual quiet proof must remain private")
            quiet=inputs.json("cache",qp.relative_to(cache).as_posix(),qs)
            if quiet.get("status")!="QUIET_CPU_COST_WINDOW" or quiet.get("active_model_workers")!=[] or quiet.get("active_cost_workers")!=[] or not 0<=guard["start_unix"]-quiet["captured_unix"]<=120 or science.get("quiet_proof_sha256")!=qs:raise ValueError("Fresh quiet/actual cost proof differs")
        (intervals if mode=="cost" else others).append(dict(id=tag,start=guard["start_unix"],end=guard["end_unix"]))
        guards.append(dict(id=tag,cache_relative_path=gp,guard_sha256=inputs.files[("cache",gp)],selected_returncode=0,sampled_peak_RSS_bytes=child["sampled_peak_RSS_bytes"],current_PID_query_performed=False))
        return folder,outer,child

    for t in a.TARGETS:
        identifier,k=t["run_id"],t["K"];condition=next(r for r in plan["conditions"] if r["run_id"]==identifier);spec=condition["spec"]
        if (condition["family"],condition["effective_F_width"],condition["registered_latent_k"],spec["lr"],spec["lora_rank"],spec["lora_alpha"])!=(t["family"],k,t["registered_K"],t["lr"],8,16):raise ValueError("Original recipe/rank/width/lr differs")
        reference=inputs.json("public",condition["V_reference"]["relative_path"],condition["V_reference"]["sha256"]);a.validate_cohort(reference,reference);channels=old["task_contracts"]["jena"]["columns"]
        if len(channels)!=21 or len(set(channels))!=21:raise ValueError("Original21 channels required")
        run=NAME+"/runs/"+identifier;pred=NAME+"/predictions/"+identifier+"/V_select"
        manifest=inputs.json("public",pred+"/PREDICTION_MANIFEST.json");a.validate_cohort(manifest,reference);msha=inputs.files[("public",pred+"/PREDICTION_MANIFEST.json")]
        scores=inputs.csv("public",pred+"/SCORES.csv")
        if len(scores)!=1:raise ValueError("One V-only score required")
        score=scores[0];receipt=inputs.json("public",run+"/BASE_FIT_RECEIPT.json");search=inputs.json("public",run+"/search/RECEIPT.json");refit=inputs.json("public",run+"/refit/RECEIPT.json")
        a.validate_training(receipt,search,refit,t,reference["scaler_sha"]);activity.append(receipt)
        want=dict(run_id=identifier,family=t["family"],seed=92601,device="cpu",CPU_threads=2,artifact_sha256=receipt["artifact_sha256"],config_sha=b.fingerprint(spec),source_sha=b.fingerprint(dict(runner=a.RUNNER_SHA,**plan["dependencies"])),model_sha=b.fingerprint(dict(artifact_sha=receipt["artifact_sha256"],model_state_sha=receipt["model_state_sha"])),F_counters=dict(streams=43*k,physical_streams=43*k,physical_calls=11))
        if any(manifest.get(key)!=value for key,value in want.items()) or any(score.get(key)!=manifest[key] for key in ("run_id","split","cohort")):raise ValueError("Actual V source/CPU/selected-artifact identity differs")
        inputs.capture("cache",normalized_relative(receipt["artifact_cache_relative_path"]),receipt["artifact_sha256"])
        _,outer,_=bound(t,t["fit_id"],"fit","COMPLETE_FIT_AND_FULL_V")
        if outer["scientific_result"].get("fit_receipt")!=receipt or outer["scientific_result"].get("V_score")!=manifest["score"] or outer["scientific_result"].get("full_V_completed") is not True or outer["scientific_result"].get("F_final_state_sha256")!=a.F_SHA:raise ValueError("Actual fit/fullV/original-base restoration differs")
        proofs={};rescored=None
        for mode,status in (("rescore","PASS_INDEPENDENT_FULL_V_FP64"),("replay","PASS_FRESH_CPU6_REPLAY")):
            tag=t["fit_id"].replace("_fit_root_v1","_"+mode+"_root_v1");folder,outer,_=bound(t,tag,mode,status);v=outer["scientific_result"]
            a.validate_validation(v,mode,receipt["artifact_sha256"],msha,k);saved=inputs.json("public",folder+("/RESCORE_RECEIPT.json" if mode=="rescore" else "/REPLAY_RECEIPT.json"))
            if any(v.get(key)!=value for key,value in saved.items()):raise ValueError("Actual validation receipt/outer differs")
            if mode=="rescore":rescored=v
            elif any(outer.get("actual_F_counters",{}).get(key)!=value for key,value in dict(streams=6*k+(42 if k==21 else 0),physical_streams=6*k+(42 if k==21 else 0),physical_calls=4 if k==21 else 2).items()):raise ValueError("Actual replay/parity activity differs")
            proofs[mode]=inputs.files[("public",folder+"/RESULT.json")]
        stats=h.gzip_rows(inputs,"public",pred+"/PUBLIC_ERROR_STATS.csv.gz",2709);aggregate=h.aggregate_statistics(stats,identifier,channels,a.ORIGINS,a.BANDS,a.OBSERVED);deltas=h.compare_accuracy(aggregate,manifest,score,rescored)
        accuracy.append(dict(run_id=identifier,**aggregate,metric_deltas=deltas,manifest_sha256=msha,artifact_sha256=receipt["artifact_sha256"]))
        row=dict(role=t["role"],run_id=identifier,registered_K=t["registered_K"],effective_F_width=k,MSE=manifest["score"]["mse"],MAE=manifest["score"]["mae"],accuracy_device="cpu",training_device="cpu",CPU_threads=2,observed_labels=a.OBSERVED,V_origins=43,search_max_epoch=3,selected_refit_epoch=receipt["selected_epoch"],search_seconds=search["seconds"],refit_seconds=refit["seconds"],training_status=receipt["training_status"],lr=t["lr"],lora_rank=8 if k==21 else None,unmerged_F_base_sha256=a.F_SHA,F_resident_bytes=None,F_resident_bytes_scope="NOT_SEPARATELY_RECORDED",accuracy_reused=False)
        for key in ("search_updates","refit_updates","search_exposures","refit_exposures"):row[key]=receipt[key]
        for batch in (1,8):
            tag=t["fit_id"].replace("_fit_root_v1","_cost_b"+str(batch)+"_root_v1");folder,outer,child=bound(t,tag,"cost","COMPLETE_PROTOCOL",batch)
            cost=inputs.json("public",folder+"/COST_RECEIPT.json");a.validate_cost(cost,t,batch,receipt["artifact_sha256"],msha,proofs)
            if any(outer["scientific_result"].get(key)!=value for key,value in cost.items()) or outer["scientific_result"].get("F_final_state_sha256")!=a.F_SHA:raise ValueError("Actual cost/effective/base-restoration receipt differs")
            pair=(cost["physical_input_sha256"],cost["observation_mask_sha256"])
            if batch in physical and physical[batch]!=pair:raise ValueError("Same-batch histories/mask differ")
            physical[batch]=pair;trials=h.gzip_rows(inputs,"public",folder+"/COST_TRIALS.csv.gz",50,cost["trials_sha256"]);totals=a.aggregate_trials(trials,identifier,batch,k);h.compare_summaries(totals["summaries"],cost["summaries"])
            cost_records.append(dict(run_id=identifier,batch=batch,**totals,effective_F_state_sha256_during_measurement=cost["effective_F_state_sha256_during_measurement"]))
            timed=next(s for s in cost["summaries"] if s["phase"]=="timed");memory=cost["memory"]
            values=dict(median_seconds=timed["median_seconds"],p90_seconds=timed["p90_seconds"],CPU_weights_MiB=memory["resident_weights_cpu_MiB"],observed_tree_peak_RSS_bytes=child["sampled_peak_RSS_bytes"],F_instances=memory["foundation_instances_created"],F_streams_per_trial=batch*k,F_calls_per_trial=math.ceil(batch*k/128),effective_F_state_sha256=cost["effective_F_state_sha256_during_measurement"],effective_F_state_scope=cost["effective_F_state_scope"],model_load_seconds=cost["model_load_seconds"],LoRA_merge_and_model_loading_in_whole_trial_timer=cost["LoRA_merge_and_model_loading_in_whole_trial_timer"],CPU_restore_snapshot_bytes_in_Route_memory_state=cost["CPU_restore_snapshot_bytes_in_Route_memory_state"])
            row.update({f"B{batch}_"+key:value for key,value in values.items()})
        rows.append(row)
    b2=inputs.json("public",B2_BASE,B2_BASE_SHA);validate_matched_activity(activity+[b2]);h.require_same_counts(accuracy);h.validate_intervals(intervals,others)
    provenance=dict(original_POST_sha256=ORIGINAL_POST_SHA,original_POST_status=original["status"],original_POST_source_sha256=ORIGINAL_POST_SOURCE_SHA,original_false_rows_count=4,original_receipt_mutated=False,original_fit_guard_sha256=None,actual_fit_guard_sha256=fitguardsha,recovery_RESULT_sha256=bindings["recovery_sha"],recovery_source_sha256=bindings["recovery_source_sha"],validation_reconciliation_source="Original four science PASS/guard0/STARTED proofs independently rechecked; original wrapper false remains false",new_cost_conditions=4,new_validation_inference_conditions=0,whole_MASTER_complete=False)
    totals=dict(new_accuracy_conditions=2,new_public_stats_rows=sum(r["public_stats_rows"] for r in accuracy),unique_observed_V_labels=a.OBSERVED,new_trial_rows=sum(r["trial_rows"] for r in cost_records),new_cost_conditions=len(cost_records),new_summary_groups=sum(len(r["summaries"]) for r in cost_records),new_summary_numeric_fields_checked=5*sum(len(r["summaries"]) for r in cost_records),F_logical_streams=sum(r["F_logical_streams"] for r in cost_records),F_physical_streams=sum(r["F_physical_streams"] for r in cost_records),F_calls=sum(r["F_calls"] for r in cost_records))
    return dict(rows=rows,accuracy=accuracy,cost_conditions=cost_records,recorded_guards=guards,totals=totals,baseline_comparison_included=False,recovery_provenance=provenance,matched_training_activity=dict(**ACTIVITY,methods=[*RUNS,b2["run_id"]],existing_B2_accuracy_reused=True,matching_exposure_is_not_equal_F_compute_or_LR=True))


def execute(public,cache,bindings,audit_sha,output):
    public,cache=Path(public).resolve(),Path(cache).resolve();output=Path(output).absolute()
    if output.exists():raise FileExistsError("Existing audit preserved before input reads")
    if not output.resolve().is_relative_to(public/NAME) or output.suffix!=".json":raise ValueError("New public N5 JSON output required")
    required={"plan_sha","fit_spec_sha","dispatch_sha","post_sha","post_source_sha","recovery_sha","recovery_source_sha"}
    if set(bindings)!=required or any(not digest(v) for v in (*bindings.values(),audit_sha)):raise ValueError("ROOT must bind actual execution/current recovery producer SHA values")
    if Path(__file__).resolve()!=public/NAME/"audit_initial_family_recovery.py" or sha(__file__)!=audit_sha:raise ValueError("Exact ROOT-pinned recovery auditor required")
    a=load_original(public);h,b=a.load_helpers(public);inputs=b.Inputs(public,cache);inputs.capture("public",NAME+"/audit_initial_family_recovery.py",audit_sha)
    for filename,want in (("independent_beijing_audit.py",a.MATH_SOURCE_SHA),("build_beijing_report.py",a.BUILDER_SOURCE_SHA)):inputs.capture("public","priority_valid_minima_20261016_v1/"+filename,want)
    result={};error=None
    try:result=collect(inputs,a,h,b,bindings)
    except (ValueError,FileNotFoundError,KeyError,TypeError,OverflowError,csv.Error,OSError,StopIteration) as exc:error=dict(error_class=type(exc).__name__,reason=str(exc) if isinstance(exc,ValueError) else "Incomplete relative evidence; preserve partial and report honestly")
    try:preserved=inputs.preserved()
    except OSError:preserved=False
    if not preserved:error=error or dict(error_class="InputPreservationFailed",reason="Captured inputs changed")
    receipt=dict(schema="INDEPENDENT_INITIAL_FAMILY_AUDIT_v1",status="PASS" if error is None and preserved else "NOT_COMPLETE_AUDIT_EVIDENCE_REJECTED",audit_source_sha256=audit_sha,ROOT_bound_execution_sha256=bindings,input_SHA_before_after_equal=preserved,source48_preserved=error is None and preserved,inputs=[dict(root=root,relative_path=name,sha256=want) for (root,name),want in sorted(inputs.files.items())],error=error,**result,whole_MASTER_complete=False,independent_generalization_claimed=False,new_models_NPZ_raw_T_numeric_training_inference_cost_or_process_operations=0,accuracy_comparison_tolerance=dict(atol=1e-9,rtol=1e-7),cost_comparison_policy="Frozen exact fsum/median/min/max/linear-p90; unchanged",limitations=["Single seed; repeated development V; matched search3/refit1 exposure does not establish convergence or equal F computation/LR/capacity.","Original PARTIAL/four false wrapper rows/null guard key are immutable; separate recovery independently checks actual evidence.","Inherited whole-TRAIN normalization/full-loader ancestry remains; no independent-generalization claim.","LoRA merged F and restored base hashes are separate; cold merge/load and restore snapshots keep their original cost scope.","Other family conditions and whole MASTER remain incomplete."])
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf-8",newline="\n") as stream:json.dump(receipt,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")
    return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--execute",action="store_true");p.add_argument("--root-triggered",action="store_true")
    names=("public-root","cache-root","plan-sha","fit-spec-sha","dispatch-sha","post-sha","post-source-sha","recovery-sha","recovery-source-sha","audit-sha","output")
    for name in names:p.add_argument("--"+name)
    v=p.parse_args(argv)
    if not v.execute:
        print(json.dumps(dict(status="PREPARATION_ONLY_NOT_EXECUTED",input_reads=0,files_written=0,scientific_or_child_operations=0,numpy_imported="numpy" in sys.modules,torch_imported="torch" in sys.modules)));return 0
    if not v.root_triggered or any(getattr(v,name.replace("-","_")) is None for name in names):p.error("ROOT actual original/recovery/current-source SHA bindings and fresh output required")
    keys=("plan_sha","fit_spec_sha","dispatch_sha","post_sha","post_source_sha","recovery_sha","recovery_source_sha")
    result=execute(v.public_root,v.cache_root,{key:getattr(v,key) for key in keys},v.audit_sha,v.output);print(json.dumps(dict(status=result["status"],files_written=1)));return 0 if result["status"]=="PASS" else 1


if __name__=="__main__":raise SystemExit(main())
