"""ROOT-only stdlib audit of N4 public sufficient statistics and saved trials.

Default reads/writes nothing. Actual execution requires whole-three-fit and
POST closure, exact source pins and a fresh output. No NPZ, raw values,
checkpoint/model deserialization, process inspection or experiment dispatch.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import types

NAME="priority_valid_minima_20261016_v1"
PLAN_SHA="6f7baaad7200c7c6601bce5a0cb6f6fdff7c3497e3a26b36906408c2844ae0f3"
BUILDER_SHA="8bb730a55e0f0f924de0616bf917ca4b47e4117503027cfc4656c6ef502750dd"
POST_ROOT="priority_pause_validate_20261016_v1/FOLLOWUP_ROOT_v1/BEIJING_MINIMA_POST_ROOT_v1"
POST_SOURCE_SHA="bcfd0a1947964296c17a58f107871178552b3a787f5cdc73805b9c61cdb856ea"
QUEUE_ROOT="priority_pause_validate_20261016_v1/FOLLOWUP_ROOT_v1/BEIJING_MINIMA_EXECUTION_ROOT_v1"
QUEUE_SOURCE_SHA="a6e593f8d431cf8559dc50fa3c8431208af327146fda68c5936a3d0ece00ad04"
GUARD_REL="postT_cpu_lag_preparation_20261016_v1/cpu_group_supervisor.py"
GUARD_SHA="82a837ea0cfee8b52765f4ac1c7325d564e886c9e94857bd0f957e4a40c75f8b"
COHORT="beijing:primary:V_select:stride48"
ORIGINS=tuple(range(14726,16783,48))
BANDS=((0,12),(12,24),(24,48))
STAGE_FIELDS=tuple("stage_"+name+"_seconds" for name in ("CPU_preprocess","host_to_device","P_E","F","D","inverse_and_CPU"))
SUMMARY_FIELDS=("median_seconds","mean_seconds","min_seconds","max_seconds","p90_seconds")
ACCURACY_ATOL,ACCURACY_RTOL=1e-9,1e-7
TARGETS=(("b2","B2","postT_cpu_beijing_E1_B2_k6_s92601_required_v1"),
    ("causal","LAG_CAUSAL","postT_cpu_beijing_E3_LAG_CAUSAL_k6_s92601_required_v1"),
    ("bidir","LAG_BIDIR","postT_cpu_beijing_E3_LAG_BIDIR_k6_s92601_required_v1"))


def sha(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):value.update(block)
    return value.hexdigest()


def is_sha(value):
    return isinstance(value,str) and len(value)==64 and all(char in "0123456789abcdef" for char in value)


def finite(value,positive=False):
    number=float(value)
    if not math.isfinite(number) or number<0 or (positive and number==0):raise ValueError("Finite nonnegative statistic or positive timing required")
    return number


def gzip_rows(inputs,root,name,maximum,expected=None):
    path=inputs.capture(root,name,expected);rows=[]
    with gzip.open(path,"rt",encoding="utf-8-sig",newline="") as stream:
        for row in csv.DictReader(stream):
            if len(rows)>=maximum or None in row:raise ValueError("Unexpected extra sufficient-statistic/trial rows or columns")
            rows.append(row)
    return rows


def aggregate_statistics(rows,identifier,channels,origins,bands,expected_observed):
    expected={(int(origin),str(channel),start,stop) for origin in origins for channel in channels for start,stop in bands}
    if len(expected)!=len(origins)*len(channels)*len(bands):raise ValueError("Unique exact cohort axes required")
    fields={"run_id","split","cohort","origin","channel","horizon_start","horizon_end","sse","sae","count"}
    seen={}
    for row in rows:
        if set(row)!=fields or (row["run_id"],row["split"],row["cohort"])!=(identifier,"V_select",COHORT):raise ValueError("Public statistic identity or producer fields differ")
        key=(int(row["origin"]),row["channel"],int(row["horizon_start"]),int(row["horizon_end"]))
        count=int(row["count"]);sse,sae=finite(row["sse"]),finite(row["sae"])
        if key not in expected or key in seen or not 0<=count<=key[3]-key[2] or (count==0 and (sse!=0 or sae!=0)):
            raise ValueError("Missing/foreign/duplicate band key, invalid count, or nonzero empty band")
        seen[key]=(sse,sae,count)
    if set(seen)!=expected:raise ValueError("All origin-channel-band rows including zero-count rows required")
    per_channel=[]
    for channel in channels:
        group=[value for key,value in seen.items() if key[1]==channel];count=sum(row[2] for row in group)
        sse=math.fsum(row[0] for row in group);sae=math.fsum(row[1] for row in group)
        per_channel.append(dict(channel=channel,count=count,sse=sse,sae=sae,mse=sse/count if count else None,mae=sae/count if count else None))
    active=[row for row in per_channel if row["count"]]
    observed=sum(row["count"] for row in per_channel)
    if not active or observed!=expected_observed:raise ValueError("Actual observed-label count differs")
    return dict(mse=math.fsum(row["mse"] for row in active)/len(active),mae=math.fsum(row["mae"] for row in active)/len(active),
        n_observed=observed,public_stats_rows=len(seen),active_channels=len(active),per_channel=per_channel,
        count_by_key=[dict(origin=key[0],channel=key[1],horizon_start=key[2],horizon_end=key[3],count=value[2]) for key,value in sorted(seen.items())])


def require_same_counts(groups):
    if not groups or any(group["count_by_key"]!=groups[0]["count_by_key"] for group in groups):raise ValueError("Exact paired observed band counts differ across conditions")


def compare_accuracy(actual,manifest,score,rescore):
    if (int(score["n_observed"])!=actual["n_observed"] or manifest["score"].get("n_observed")!=actual["n_observed"]
            or int(score["active_channels"])!=actual["active_channels"] or manifest["score"].get("active_channels")!=actual["active_channels"]
            or rescore.get("n_observed")!=actual["n_observed"] or rescore.get("public_stats_rows")!=actual["public_stats_rows"]):
        raise ValueError("Observed labels or rescore public-statistic row count differ")
    deltas={}
    for name in ("mse","mae"):
        saved=finite(manifest["score"][name]);independent=finite(rescore[name])
        if float(score[name])!=saved or not math.isclose(actual[name],saved,abs_tol=ACCURACY_ATOL,rel_tol=ACCURACY_RTOL) or not math.isclose(actual[name],independent,abs_tol=ACCURACY_ATOL,rel_tol=ACCURACY_RTOL):
            raise ValueError("Public band aggregate, original full-array FP64 rescore and exact SCORES binding differ")
        deltas[name]=dict(public_minus_manifest=actual[name]-saved,public_minus_saved_array_rescore=actual[name]-independent)
    return deltas


def aggregate_trials(rows,identifier,batch):
    phase_counts=dict(warmup=10,timed=20,stage_probe=20)
    fields={"run_id","phase","trial","order_position","batch_size","whole_seconds","F_logical_streams","F_physical_streams","F_calls","prediction_sha256","row_microbatch","stage_instrumentation",*STAGE_FIELDS}
    expected={(phase,trial) for phase,n in phase_counts.items() for trial in range(n)};seen={};totals=[0,0,0];prediction_hashes=set()
    for row in rows:
        if set(row)!=fields:raise ValueError("Original physical cost trial fields differ")
        key=(row["phase"],int(row["trial"]));instrumented=key[0]=="stage_probe"
        counters=tuple(int(row[name]) for name in ("F_logical_streams","F_physical_streams","F_calls"))
        if (key not in expected or key in seen or row["run_id"]!=identifier or int(row["batch_size"])!=batch or int(row["order_position"])!=0
                or int(row["row_microbatch"])!=128 or counters!=(batch*6,batch*6,1) or row["stage_instrumentation"]!=str(instrumented) or not is_sha(row["prediction_sha256"])):
            raise ValueError("Exact50 trial keys/CPU row128/K6 counters/phase binding differ")
        seconds=finite(row["whole_seconds"],positive=True)
        for field in STAGE_FIELDS:
            if instrumented:finite(row[field])
            elif row[field]!="":raise ValueError("Ordinary whole trials must not contain intrusive stage measurements")
        seen[key]=seconds;totals=[left+right for left,right in zip(totals,counters)];prediction_hashes.add(row["prediction_sha256"])
    if set(seen)!=expected:raise ValueError("Complete10 warmup20 timed20 stage trials required")
    summaries=[]
    for phase in ("timed","stage_probe"):
        values=sorted(value for key,value in seen.items() if key[0]==phase);position=(len(values)-1)*.9;low=math.floor(position);fraction=position-low
        p90=values[low]+(values[low+1]-values[low])*fraction
        summaries.append(dict(run_id=identifier,phase=phase,batch_size=batch,trials=20,median_seconds=statistics.median(values),mean_seconds=math.fsum(values)/len(values),
            min_seconds=values[0],max_seconds=values[-1],p90_seconds=p90,stage_instrumentation=phase=="stage_probe",stage_sum_equals_whole_claimed=False))
    return dict(trial_rows=len(seen),phase_rows=phase_counts,F_logical_streams=totals[0],F_physical_streams=totals[1],F_calls=totals[2],
        unique_prediction_hashes=len(prediction_hashes),summaries=summaries)


def compare_summaries(actual,saved):
    if len(saved)!=2 or {row["phase"] for row in saved}!={"timed","stage_probe"}:raise ValueError("Two exact original cost summaries required")
    largest=0.
    for row in actual:
        reference=next(value for value in saved if value["phase"]==row["phase"])
        if any(reference.get(key)!=row[key] for key in ("run_id","phase","batch_size","trials","stage_instrumentation","stage_sum_equals_whole_claimed")):
            raise ValueError("Cost summary protocol differs")
        for field in SUMMARY_FIELDS:
            difference=finite(reference[field],positive=True)-row[field];largest=max(largest,abs(difference))
            if difference!=0:raise ValueError("Cost summary arithmetic differs without a new acceptance tolerance: "+row["phase"]+"/"+field+" delta="+repr(difference))
    return largest


def validate_intervals(costs,others):
    for row in costs+others:
        start,end=finite(row["start"]),finite(row["end"])
        if end<start:raise ValueError("Actual terminal guard interval required")
    for index,left in enumerate(costs):
        for right in costs[index+1:]+others:
            if max(left["start"],right["start"])<min(left["end"],right["end"]):raise ValueError("Recorded cost interval overlaps another recorded model/validation/cost guard")


def expected_steps():
    return {"beijing_"+label+"_"+mode+("_b"+str(batch) if batch else "")+"_root_v1":(identifier,mode,batch)
        for label,_,identifier in TARGETS for mode,batch in (("rescore",None),("replay",None),("cost",1),("cost",8))}


def validate_post(post):
    steps=post.get("steps",[])
    if (post.get("status")!="COMPLETE_THREE_VALIDATIONS_AND_SIX_COSTS" or post.get("skipped")!=[] or len(steps)!=12
            or {row.get("id") for row in steps}!=set(expected_steps()) or any(row.get("success") is not True for row in steps)):
        raise ValueError("Exact complete twelve POST steps required; partial evidence is not complete")


def load_builder(public):
    path=Path(public)/NAME/"build_beijing_report.py"
    if sha(path)!=BUILDER_SHA:raise ValueError("Pinned stdlib evidence helper changed")
    value=types.ModuleType("n4_pinned_closed_evidence_helpers");value.__file__=str(path)
    exec(compile(path.read_text(encoding="utf-8"),str(path),"exec"),value.__dict__)
    return value


def collect(inputs,helper,plan_sha,post_sha):
    plan=inputs.json("public",NAME+"/PLAN.json",plan_sha)
    if plan_sha!=PLAN_SHA or plan.get("status")!="SEALED_BY_ROOT" or plan.get("runner_sha256")!=helper.RUNNER_SHA:raise ValueError("Exact original sealed N4 plan required")
    inputs.capture("public",NAME+"/beijing_minimum_runner.py",helper.RUNNER_SHA)
    inputs.capture("public",NAME+"/build_beijing_report.py",BUILDER_SHA)
    inputs.capture("cache",POST_ROOT+"/RUN_ROOT_v1.ps1",POST_SOURCE_SHA);inputs.capture("cache",QUEUE_ROOT+"/RUN_ROOT_v1.ps1",QUEUE_SOURCE_SHA);inputs.capture("cache",GUARD_REL,GUARD_SHA)
    old=inputs.json("public",helper.OLD+"/PLAN.json",plan["original_required_plan_sha256"])
    if len(old["source39"])!=39 or len(old["immutable_public"])!=9:raise ValueError("Exact original48 source/immutable inputs required")
    for name,digest in {**old["source39"],**old["immutable_public"],**plan["dependencies"]}.items():inputs.capture("public",name,digest)
    reference=inputs.json("public",plan["V_reference_public_relative_path"] ,plan["V_reference_sha256"]);helper.validate_cohort(reference)
    queue=inputs.json("cache",QUEUE_ROOT+"/RESULT.json");post=inputs.json("cache",POST_ROOT+"/RESULT.json",post_sha);validate_post(post)
    if (queue.get("status") not in {"COMPLETE_THREE_FITS_AND_FULL_V","PARTIAL_FITS_PRESERVED"} or len(queue.get("conditions",[]))!=3
            or {row.get("run_id") for row in queue["conditions"]}!={identifier for _,_,identifier in TARGETS}):raise ValueError("Whole original three-fit queue must be terminal and exact")
    channels=old["task_contract"]["columns"]
    if len(channels)!=11 or len(set(channels))!=11:raise ValueError("Original exact eleven-channel metadata required")
    step_map={row["id"]:row for row in post["steps"]};accuracy=[];costs=[];others=[];cost_intervals=[];physical={};guards=[]

    def bound_outer(tag,identifier,mode,status):
        folder=NAME+"/diagnostics/"+tag;outer=inputs.json("public",folder+"/RESULT.json");helper.validate_outer(outer,identifier,mode,status)
        guard,child,guard_path=helper.guard_for(inputs,folder,outer)
        interval=dict(id=tag,start=guard["start_unix"],end=guard["end_unix"])
        if mode=="fit":
            row=next(row for row in queue["conditions"] if row["run_id"]==identifier)
            if row.get("id")!=tag or row.get("outer_status")!=status or row.get("outer_sha256")!=inputs.files[("public",folder+"/RESULT.json")]:raise ValueError("Original terminal fit queue outer binding differs")
        else:
            step=step_map[tag]
            if (step.get("run_id")!=identifier or step.get("mode")!=mode or step.get("guard_exit_code")!=0 or step.get("guard_status")!="COMPLETE"
                    or step.get("batch")!=expected_steps()[tag][2]
                    or step.get("outer_status")!=status or step.get("guard_sha256")!=inputs.files[("cache",guard_path)] or step.get("outer_sha256")!=inputs.files[("public",folder+"/RESULT.json")]):raise ValueError("Actual POST/guard/outer exact SHA or child identity differs")
        guards.append(dict(id=tag,cache_relative_path=guard_path,selected_returncode=child["returncode"],observed_owned_identities=len(child["owned_processes"]),
            observed_tree_join_confirmed=child["observed_tree_join_confirmed"],sampled_peak_RSS_bytes=child["sampled_peak_RSS_bytes"],current_identity_rechecked=False))
        (cost_intervals if mode=="cost" else others).append(interval)
        return folder,outer

    for label,family,identifier in TARGETS:
        run=NAME+"/runs/"+identifier;prediction=NAME+"/predictions/"+identifier+"/V_select"
        manifest=inputs.json("public",prediction+"/PREDICTION_MANIFEST.json");helper.validate_cohort(manifest);score=helper.score_csv(inputs,prediction,manifest)
        manifest_sha=inputs.files[("public",prediction+"/PREDICTION_MANIFEST.json")]
        receipt=inputs.json("public",run+"/BASE_FIT_RECEIPT.json");search=inputs.json("public",run+"/search/RECEIPT.json");refit=inputs.json("public",run+"/refit/RECEIPT.json")
        helper.validate_training(receipt,search,refit,identifier,family);spec=next(row["spec"] for row in plan["conditions"] if row["run_id"]==identifier)
        if (manifest.get("run_id")!=identifier or manifest.get("family")!=family or manifest.get("seed")!=92601 or manifest.get("device")!="cpu"
                or manifest.get("artifact_sha256")!=receipt["artifact_sha256"] or manifest.get("config_sha")!=helper.fingerprint(spec)
                or manifest.get("source_sha")!=helper.fingerprint(dict(runner=helper.RUNNER_SHA,**plan["dependencies"]))
                or manifest.get("model_sha")!=helper.fingerprint(dict(artifact_sha=receipt["artifact_sha256"],model_state_sha=receipt["model_state_sha"]))
                or manifest.get("F_counters")!=dict(streams=258,physical_streams=258,physical_calls=11)):raise ValueError("Actual full CPU V43/selected checkpoint/source identity differs")
        _,fit_outer=bound_outer("beijing_"+label+"_fit_root_v1",identifier,"fit","COMPLETE_FIT_AND_FULL_V")
        science=fit_outer["scientific_result"]
        if science.get("fit_receipt")!=receipt or science.get("V_score")!=manifest["score"] or science.get("full_V_completed") is not True or science.get("F_final_state_sha256")!=helper.F_SHA:raise ValueError("Actual fullV fit/frozen-state receipt differs")
        proof_shas={};rescore=None
        for mode,status in (("rescore","PASS_INDEPENDENT_FULL_V_FP64"),("replay","PASS_FRESH_CPU6_REPLAY")):
            folder,outer=bound_outer("beijing_"+label+"_"+mode+"_root_v1",identifier,mode,status);science=outer["scientific_result"]
            helper.validate_validation(science,mode,receipt["artifact_sha256"],manifest_sha)
            saved=inputs.json("public",folder+("/RESCORE_RECEIPT.json" if mode=="rescore" else "/REPLAY_RECEIPT.json"))
            if any(science.get(key)!=value for key,value in saved.items()):raise ValueError("Actual validation receipt and outer scientific fields differ")
            if mode=="rescore":rescore=science
            elif science.get("F_final_state_sha256")!=helper.F_SHA:raise ValueError("Actual replay frozen F identity missing")
            proof_shas[mode]=inputs.files[("public",folder+"/RESULT.json")]
        stats=gzip_rows(inputs,"public",prediction+"/PUBLIC_ERROR_STATS.csv.gz",1419)
        aggregate=aggregate_statistics(stats,identifier,channels,ORIGINS,BANDS,20539);deltas=compare_accuracy(aggregate,manifest,score,rescore)
        accuracy.append(dict(run_id=identifier,family=family,**aggregate,metric_deltas=deltas,selected_refit_epoch=receipt["selected_epoch"],
            search_exposures=receipt["search_exposures"],refit_exposures=receipt["refit_exposures"],artifact_sha256=receipt["artifact_sha256"],manifest_sha256=manifest_sha))
        for batch in (1,8):
            tag="beijing_"+label+"_cost_b"+str(batch)+"_root_v1";folder,outer=bound_outer(tag,identifier,"cost","COMPLETE_PROTOCOL")
            cost_receipt=inputs.json("public",folder+"/COST_RECEIPT.json");trials=gzip_rows(inputs,"public",folder+"/COST_TRIALS.csv.gz",50,cost_receipt["trials_sha256"])
            helper.validate_cost(cost_receipt,identifier,batch,receipt["artifact_sha256"],manifest_sha,inputs.files[("public",folder+"/COST_TRIALS.csv.gz")],proof_shas)
            if any(outer["scientific_result"].get(key)!=value for key,value in cost_receipt.items()) or outer["scientific_result"].get("F_final_state_sha256")!=helper.F_SHA:raise ValueError("Actual cost receipt/outer/frozen F differs")
            pair=(cost_receipt["physical_input_sha256"],cost_receipt["observation_mask_sha256"])
            if not all(is_sha(value) for value in pair) or (batch in physical and physical[batch]!=pair):raise ValueError("Exact same physical input/mask fingerprints required across methods")
            physical[batch]=pair;aggregate=aggregate_trials(trials,identifier,batch);delta=compare_summaries(aggregate["summaries"],cost_receipt["summaries"])
            costs.append(dict(run_id=identifier,batch=batch,public_diagnostic_relative_path=folder,**aggregate,max_summary_difference_seconds=delta,
                physical_input_sha256=pair[0],observation_mask_sha256=pair[1],physical_output_and_inverse_dtype=cost_receipt["physical_output_and_inverse_dtype"],memory_scope=cost_receipt["memory"]))
    require_same_counts(accuracy);validate_intervals(cost_intervals,others)
    for row in accuracy:row["band_count_binding_sha256"]=helper.fingerprint(row.pop("count_by_key"))
    return dict(accuracy=accuracy,cost_conditions=costs,recorded_guards=guards,recorded_cost_intervals=cost_intervals,
        all_six_recorded_cost_intervals_disjoint_from_known_model_validation_and_cost_intervals=True,
        totals=dict(accuracy_conditions=3,public_stat_rows=sum(row["public_stats_rows"] for row in accuracy),observed_labels_per_condition=20539,
            unique_V_observed_labels=20539,trial_rows=sum(row["trial_rows"] for row in costs),cost_conditions=6,summary_groups=12,summary_numeric_fields_checked=60,
            F_logical_streams=sum(row["F_logical_streams"] for row in costs),F_physical_streams=sum(row["F_physical_streams"] for row in costs),F_calls=sum(row["F_calls"] for row in costs)))


def execute(public,cache,plan_sha,post_sha,audit_sha,output):
    output=Path(output).absolute()
    if output.exists():raise FileExistsError("Existing audit output preserved before any input read")
    public,cache=Path(public).resolve(),Path(cache).resolve()
    if not output.resolve().is_relative_to(public/NAME) or output.suffix!=".json":raise ValueError("One new explicit JSON output inside N4 namespace required")
    if plan_sha!=PLAN_SHA or not is_sha(post_sha) or not is_sha(audit_sha):raise ValueError("Exact PLAN/POST/auditor SHA required")
    if Path(__file__).resolve()!=public/NAME/"independent_beijing_audit.py" or sha(__file__)!=audit_sha:raise ValueError("ROOT-pinned current audit source differs")
    helper=load_builder(public);inputs=helper.Inputs(public,cache);inputs.capture("public",NAME+"/independent_beijing_audit.py",audit_sha)
    result={};error=None
    try:result=collect(inputs,helper,plan_sha,post_sha)
    except (ValueError,FileNotFoundError,KeyError,TypeError,OverflowError,csv.Error,OSError) as failure:error=dict(error_class=type(failure).__name__,reason=str(failure))
    preservation_error=None
    try:preserved=inputs.preserved()
    except OSError as failure:preserved=False;preservation_error=type(failure).__name__
    if not preserved:error=error or dict(error_class="InputPreservationFailed",reason="Captured input changed or became unavailable before output")
    receipt=dict(schema="INDEPENDENT_N4_BEIJING_AUDIT_v1",status="PASS" if error is None and preserved else "NOT_COMPLETE_AUDIT_EVIDENCE_REJECTED",
        audit_source_sha256=audit_sha,plan_sha256=plan_sha,post_result_sha256=post_sha,input_SHA_before_after_equal=preserved,input_preservation_error_class=preservation_error,input_count=len(inputs.files),
        inputs=[dict(root=root,relative_path=name,sha256=digest) for (root,name),digest in sorted(inputs.files.items())],error=error,**result,
        accuracy_comparison_tolerance=dict(atol=ACCURACY_ATOL,rtol=ACCURACY_RTOL,source="priority_b2_validation_20261016_v1/priority_runner.py:310-317"),
        cost_comparison_policy="Exact saved-vs-stdlib fsum mean/median/min/max/linear p90; no new cost acceptance tolerance. Any difference rejects completion.",
        new_Task_NPZ_raw_checkpoint_tensor_model_training_inference_cost_or_process_operations=0,
        arithmetic_scope="Public sufficient-statistic aggregation; existing full saved-array FP64 rescore/CPU6 receipts are bound and reused, not re-executed.",
        guard_scope="Recorded exact PID/create-time/tree-join/spec/quiet evidence reused; no current process query or global idleness claim.",
        memory_scope="CPU resident weights, sampled process-tree RSS, GPU memory fields and excluded training/loading peaks retain separate scopes.",
        whole_MASTER_complete=False,independent_generalization_claimed=False,
        limitations=["Single seed92601, limited search/fresh selected refit and repeatedly inspected development V.",
            "Three methods share the same20539 observed labels; their repeated counts are not independent observations.",
            "A passing arithmetic/integrity audit is not evidence of forecast improvement, convergence, or generalization.",
            "Band statistics cannot recover individual residuals or individual point wins.",
            "No raw/NPZ/checkpoint/model bytes or tensors are decoded; frozen F checks are prior actual runner evidence."])
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf-8",newline="\n") as stream:json.dump(receipt,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--execute",action="store_true");parser.add_argument("--root-triggered",action="store_true")
    for flag in ("public-root","cache-root","plan-sha","post-sha","audit-sha","output"):parser.add_argument("--"+flag)
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="PREPARATION_ONLY_NOT_EXECUTED",input_reads=0,files_written=0,scientific_or_child_operations=0,numpy_imported="numpy" in sys.modules,torch_imported="torch" in sys.modules)));return 0
    if not args.root_triggered or any(getattr(args,flag) is None for flag in ("public_root","cache_root","plan_sha","post_sha","audit_sha","output")):
        parser.error("Explicit ROOT trigger, roots, exact PLAN/actual complete POST/current audit SHA and fresh output required")
    try:result=execute(args.public_root,args.cache_root,args.plan_sha,args.post_sha,args.audit_sha,args.output)
    except (ValueError,FileNotFoundError,FileExistsError) as error:
        print(json.dumps(dict(status="NOT_COMPLETE_AUDIT_EVIDENCE_REJECTED",error_class=type(error).__name__)));return 1
    print(json.dumps(dict(status=result["status"],input_count=result["input_count"],files_written=1)));return 0 if result["status"]=="PASS" else 1


if __name__=="__main__":raise SystemExit(main())
