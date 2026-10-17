"""ROOT-only stdlib audit of remaining Jena AdaPTS / Beijing B4 routes.

Default zero I/O. Actual arithmetic decodes public band statistics and saved
trials only. No NPZ/tensor/raw/model/process operations or experiment dispatch.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import types

NAME="priority_valid_family_20261016_v1"
OLD="postT_V_only_cpu_family_minima_20261016_v1"
PLAN_SHA="6995939bfb190c5cbd100f5914015f7e5365b8f9a2b451a2fb2156978724bee1"
OLD_SHA="e34033935916dcaded0885ccde8e1aa084f5cc0df994024d0827510d6db53b10"
RUNNER_SHA="12462b8f1085e346750a24cd37fd52191f8780799e2f9e5903b4a57bcbc570c1"
PREP="priority_pause_validate_20261016_v1/FOLLOWUP_ROOT_v1/FAMILY_REMAINING_PREP_v1"
DISPATCH_SHA="0d7c14821f01698614995b9b1e98ddefa60cf096bb8dd5b63cefff87c33b9ad8"
POST_SHA="5442cd39cbaaf5defc05ea968d0504c4d6be4a9712bd033844b4182da8e7930a"
DRAFT_SHA="afa56cb7c8f0745c6a0fce47e0adf7635364afe8db307208bd7164fe64fca58e"
F_SHA="c2b4edf7db22d3f18e58ca79299a5c7e864e9b8f277e4e29e2f88d07dbb35874"
GUARD_REL="postT_cpu_lag_preparation_20261016_v1/cpu_group_supervisor.py"
GUARD_SHA="82a837ea0cfee8b52765f4ac1c7325d564e886c9e94857bd0f957e4a40c75f8b"
HELPERS={
 "inputs":("priority_valid_minima_20261016_v1/build_beijing_report.py","8bb730a55e0f0f924de0616bf917ca4b47e4117503027cfc4656c6ef502750dd"),
 "arithmetic":("priority_valid_minima_20261016_v1/independent_beijing_audit.py","933883a052b02e6a41ccdadaa5ec45e80b977dbfaada007607ac1093f518dd04"),
 "guard":(NAME+"/audit_initial_family.py","f92a3a7b4bc03677f67d3bb7c42758765a95f49a378e2dd9d768ffc7f4de6241"),
 "cost":("priority_multik_resume_20261016_v1/audit_multik_resume.py","0c22445d86e75610d78caeda0dc645c0fa7c09d3ed46f6a2b9ed44878ea25146")}
DEADLINE=1792222550
BANDS=((0,12),(12,24),(24,48))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):h.update(block)
    return h.hexdigest()


def digest(value):return isinstance(value,str) and len(value)==64 and all(c in "0123456789abcdef" for c in value)


def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def targets():
    return [dict(dataset=d,family=family,latent_k=k,C=c,run_id=identifier,id=tag,observed=n,stat_rows=43*c*3,
        search_end=a50,refit_end=a60,prefix_end=end,origins=tuple(start+48*i for i in range(43)),cohort=d+":primary:V_select:stride48")
        for d,family,k,c,identifier,tag,n,a50,a60,start,end in (
            ("jena","ADAPTS",6,21,"postT_cpu_jena_E1_ADAPTS_BOLT_PORT_k6_s92601_family_v1","family_remaining_jena_adapts_k6_fit_root_v1",43344,10482,12579,14675,16772),
            ("beijing","B4",11,11,"postT_cpu_beijing_E1_B4_k11_s92601_family_v1","family_remaining_beijing_b4_k11_fit_root_v1",20539,10519,12622,14726,16830))]


def expected_steps():
    return {t["id"].replace("_fit_root_v1","")+"_"+mode+("_b"+str(batch) if batch else "")+"_root_v1":dict(run_id=t["run_id"],mode=mode,batch=batch)
        for t in targets() for mode,batch in (("rescore",None),("replay",None),("cost",1),("cost",8))}


def validate_post(v):
    steps=v.get("steps",[]);expected=expected_steps()
    if (v.get("status")!="COMPLETE_REMAINING_TWO_VALIDATIONS_AND_FOUR_COSTS" or v.get("skipped")!=[] or v.get("error") is not None
            or len(steps)!=8 or len({r.get("id") for r in steps})!=8 or {r.get("id") for r in steps}!=set(expected)
            or any(r.get("success") is not True or any(r.get(k)!=x for k,x in expected[r["id"]].items()) for r in steps)):
        raise ValueError("Exact eight completed POST steps required; partial remains NOT_COMPLETE")


def deadline_arg(command,recorded,fit):
    if command.count("--deadline")!=1 or command.index("--deadline")+1>=len(command):raise ValueError("One actual worker deadline argv required")
    actual=float(command[command.index("--deadline")+1])
    if not math.isfinite(actual) or actual!=recorded or not 0<actual<=DEADLINE or (not fit and actual!=DEADLINE):
        raise ValueError("Actual argv/STARTED/original worker deadline differs")


def validate_training(v,search,refit,t,scaler):
    epoch=v.get("selected_epoch");n=t["search_end"]-559;nr=t["refit_end"]-559
    if epoch not in (0.,.25,1.,3.):raise ValueError("One actual original selected endpoint including0 required")
    points=[0]+[math.ceil(e*n) for e in (.25,1.,3.)];updates=sum(math.ceil((b-a)/16) for a,b in zip(points,points[1:]))
    want=dict(status="COMPLETE_VALID",run_id=t["run_id"],family=t["family"],seed=92601,epochs=[.25,1.,3.],selected_epoch=epoch,
        training_status="UNDERTRAINED_OR_BUDGET_LIMIT" if epoch==3 else "SCREEN_COMPLETE_CONVERGENCE_UNRESOLVED",optimizer="AdamW",weight_decay=0.,
        search_fit_end=t["search_end"],refit_fit_end=t["refit_end"],refit_scaler_sha=scaler,
        search_updates=updates,search_exposures=3*n,refit_updates=math.ceil(math.ceil(epoch*nr)/16),refit_exposures=math.ceil(epoch*nr))
    if any(v.get(k)!=x for k,x in want.items()):raise ValueError("Dataset-specific actual endpoint/update/exposure/training uncertainty differs")
    for stage,expected in ((search,dict(status="SEGMENT_COMPLETE",completed_epochs=[0.,.25,1.,3.],selected_epoch=epoch,train_origins=n,updates=updates,exposures=3*n)),
            (refit,dict(status="SEGMENT_COMPLETE",completed_epochs=[0.]+([epoch] if epoch else []),selected_epoch=epoch,train_origins=nr,
                updates=math.ceil(math.ceil(epoch*nr)/16),exposures=math.ceil(epoch*nr)))):
        if any(stage.get(k)!=x for k,x in expected.items()):raise ValueError("Actual original search/full selected refit stage arithmetic differs")


def aggregate_statistics(rows,identifier,channels,origins,observed,cohort):
    expected={(int(o),str(c),a,b) for o in origins for c in channels for a,b in BANDS};seen={}
    fields={"run_id","split","cohort","origin","channel","horizon_start","horizon_end","sse","sae","count"}
    for row in rows:
        if set(row)!=fields or (row["run_id"],row["split"],row["cohort"])!=(identifier,"V_select",cohort):raise ValueError("Original same-dataset sufficient-statistic identity/schema differs")
        key=(int(row["origin"]),row["channel"],int(row["horizon_start"]),int(row["horizon_end"]));n=int(row["count"]);sse,sae=float(row["sse"]),float(row["sae"])
        if (key not in expected or key in seen or not 0<=n<=key[3]-key[2] or not all(math.isfinite(x) and x>=0 for x in (sse,sae)) or (n==0 and (sse!=0 or sae!=0))):raise ValueError("Foreign/duplicate/missing band key/count or invalid sufficient statistic")
        seen[key]=(sse,sae,n)
    if set(seen)!=expected:raise ValueError("All original origin/channel/band rows including count0 required")
    per_channel=[]
    for c in channels:
        group=[v for k,v in seen.items() if k[1]==c];n=sum(v[2] for v in group);sse=math.fsum(v[0] for v in group);sae=math.fsum(v[1] for v in group)
        per_channel.append(dict(channel=c,count=n,sse=sse,sae=sae,mse=sse/n if n else None,mae=sae/n if n else None))
    active=[r for r in per_channel if r["count"]];n=sum(r["count"] for r in per_channel)
    if not active or n!=observed:raise ValueError("Actual same-dataset observed count differs")
    return dict(mse=math.fsum(r["mse"] for r in active)/len(active),mae=math.fsum(r["mae"] for r in active)/len(active),n_observed=n,
        active_channels=len(active),public_stats_rows=len(seen),per_channel=per_channel,
        count_by_key=[dict(origin=k[0],channel=k[1],horizon_start=k[2],horizon_end=k[3],count=v[2]) for k,v in sorted(seen.items())])


def load_helpers(public):
    modules={}
    for key,(name,expected) in HELPERS.items():
        p=Path(public)/name
        if sha(p)!=expected:raise ValueError("Frozen stdlib audit helper changed")
        m=types.ModuleType("remaining_pinned_"+key);m.__file__=str(p);exec(compile(p.read_text(encoding="utf-8"),str(p),"exec"),m.__dict__);modules[key]=m
    return modules


def collect(inputs,helpers,bindings):
    h,g=helpers["arithmetic"],helpers["guard"]
    for name,expected in HELPERS.values():inputs.capture("public",name,expected)
    for root,path,d in (("public",NAME+"/family_minimum_runner.py",RUNNER_SHA),("cache",PREP+"/RUN_DISPATCH_DRAFT_v1.ps1",DISPATCH_SHA),
            ("cache",PREP+"/RUN_POST_DRAFT_v1.ps1",POST_SHA),("cache",PREP+"/PLAN_DRAFT.json",DRAFT_SHA),("cache",GUARD_REL,GUARD_SHA)):
        inputs.capture(root,path,d)
    plan=inputs.json("public",NAME+"/PLAN.json",bindings["plan_sha"]);old=inputs.json("public",OLD+"/PLAN.json",OLD_SHA)
    if (bindings["plan_sha"]!=PLAN_SHA or plan.get("status")!="SEALED_BY_ROOT" or plan.get("runner_sha256")!=RUNNER_SHA or len(plan.get("conditions",[]))!=4
            or plan.get("deadline_unix")!=DEADLINE or plan.get("new_T_target_operations_allowed") is not False or plan.get("protected_tail_numeric_parse_allowed") is not False
            or len(old["source39"])!=39 or len(old["immutable_public"])!=9 or len(set(old["source39"])|set(old["immutable_public"]))!=48):raise ValueError("Exact unchanged four-family PLAN/source48/T lock required")
    initial={"postT_cpu_jena_E1_LORA_R8_k0_s92601_family_v1","postT_cpu_jena_E1_B4_k6_s92601_family_v1"}
    if set(plan.get("ROOT_initial_admitted_run_ids",[]))!=initial:raise ValueError("Original initial admission history must stay unchanged")
    for path,d in {**old["source39"],**old["immutable_public"],**plan["dependencies"],**old["upstream_recipe_metadata"]}.items():inputs.capture("public",path,d)
    for path,d in plan["original_family_files"].items():inputs.capture("public",OLD+"/"+path,d)
    for condition in plan["conditions"]:
        for ref in (condition["probe"],condition["V_reference"]):inputs.capture("public",ref["relative_path"],ref["sha256"])
    fit=inputs.json("cache",PREP+"/RESULT.json",bindings["dispatch_sha"]);post=inputs.json("cache",PREP+"/POST_RESULT.json",bindings["post_sha"]);validate_post(post)
    if (fit.get("schema")!="REMAINING_FAMILY_TWO_FIT_RESULT_v1" or fit.get("status")!="COMPLETE_REMAINING_TWO_FITS_AND_FULL_V" or fit.get("strong_observed_tree_closure_verified") is not True
            or len(fit.get("conditions",[]))!=2 or {r.get("run_id") for r in fit["conditions"]}!={r["run_id"] for r in targets()}
            or post.get("schema")!="REMAINING_FAMILY_POST_RESULT_v1" or post.get("dispatcher_source_sha256")!=DISPATCH_SHA
            or post.get("fit_dispatch_result_sha256")!=bindings["dispatch_sha"] or post.get("fit_dispatch_SPEC_sha256")!=bindings["dispatch_spec_sha"]
            or post.get("actual_fit_guard_sha256")!=bindings["dispatch_guard_sha"]):raise ValueError("Exact actual remaining-two complete fits and bound POST required; partial is NOT_COMPLETE")
    for v in (fit,post):
        if (v.get("error") is not None or v.get("pins_before_after_unchanged") is not True or v.get("changed_or_unreadable_pinned_paths")!=[]
                or v.get("plan_sha256")!=PLAN_SHA or v.get("source39_immutable9_count")!=48 or v.get("initial_two_retrained") is not False
                or v.get("original_worker_deadline")!=DEADLINE or v.get("original_campaign_deadline")!=1792223150):raise ValueError("Actual terminal shared source/resource/original deadline closure required")
    if fit.get("original_initial_admission_preserved") is not True:raise ValueError("Original initial admission must remain historical")
    dispatch_started=inputs.json("cache",PREP+"/STARTED.json");post_started=inputs.json("cache",PREP+"/POST_STARTED.json")
    if any(dispatch_started.get(k)!=v for k,v in dict(script_sha256=DISPATCH_SHA,plan_sha256=PLAN_SHA,admission_sha256=fit["admission_sha256"],predecessor_sha256=fit["predecessor_sha256"],original_initial_admission_preserved=True,initial_two_retrained=False).items()):raise ValueError("Actual ROOT additional-two dispatch STARTED differs")
    if any(post_started.get(k)!=v for k,v in dict(script_sha256=POST_SHA,dispatcher_source_sha256=DISPATCH_SHA,fit_dispatch_result_sha256=bindings["dispatch_sha"],fit_dispatch_SPEC_sha256=bindings["dispatch_spec_sha"],actual_fit_guard_sha256=bindings["dispatch_guard_sha"],plan_sha256=PLAN_SHA,new_training=0).items()):raise ValueError("Actual ROOT POST start binding differs")
    fit_spec=inputs.json("cache",PREP+"/SPEC_ROOT_v1.json",bindings["dispatch_spec_sha"]);fit_guard=inputs.json("cache",PREP+"/GUARD_ROOT_v1/RESULT.json",bindings["dispatch_guard_sha"])
    if len(fit_spec.get("processes",[]))!=2:raise ValueError("Both actual fit worker trees required")
    steps={r["id"]:r for r in post["steps"]};guards=[];others=[];cost_intervals=[];accuracy=[];costs=[];rows=[]

    def bound(t,tag,mode,status,record,fit_mode=False):
        folder=NAME+"/diagnostics/"+tag;outer=inputs.json("public",folder+"/RESULT.json",record["outer_sha256"])
        started=inputs.json("public",folder+"/STARTED.json",record.get("STARTED_sha256") if fit_mode else None)
        want=dict(status=status,run_id=t["run_id"],mode=mode,plan_sha256=PLAN_SHA,error_class=None,postflight_error_class=None,
            original_source39_immutable9_family7_probe4_recipe4_and_model_pins_preserved=True,new_T_target_operations=0,independent_generalization_claimed=False,old_full_task_identity_inherited_not_recomputed=True)
        if any(outer.get(k)!=v for k,v in want.items()) or any(started.get(k)!=v for k,v in dict(run_id=t["run_id"],mode=mode,plan_sha256=PLAN_SHA).items()):raise ValueError("Exact remaining-family actual outer/STARTED/source/T identity required")
        if fit_mode:sp,gp,spec,guard=PREP+"/SPEC_ROOT_v1.json",PREP+"/GUARD_ROOT_v1/RESULT.json",fit_spec,fit_guard
        else:
            sp,gp=PREP+"/SPEC_"+tag+".json",NAME+"/supervisors/"+tag+"/RESULT.json"
            spec=inputs.json("cache",sp,record["spec_sha256"]);guard=inputs.json("cache",gp,record["guard_sha256"])
            if record.get("strong_observed_tree_closure_verified") is not True or record.get("guard_exit_code")!=0 or record.get("guard_status")!="COMPLETE":raise ValueError("Actual clean single POST guard required")
        found=[p for p in spec["processes"] if p["id"]==tag]
        if len(found)!=1:raise ValueError("One exact actual selected child command required")
        command=found[0]["command"]
        for flag,want in (("--run-id",t["run_id"]),("--mode",mode),("--plan-sha",PLAN_SHA)):
            if command.count(flag)!=1 or command[command.index(flag)+1]!=want:raise ValueError("Exact actual argv scientific identity differs")
        for flag,path in (("--output",inputs.roots["public"]/folder),("--public-root",inputs.roots["public"]),("--cache-root",inputs.roots["cache"])):
            if command.count(flag)!=1 or Path(command[command.index(flag)+1]).resolve()!=path.resolve():raise ValueError("Exact actual worker root/output argv differs")
        if sum(Path(v).resolve()==(inputs.roots["public"]/NAME/"family_minimum_runner.py").resolve() for v in command)!=1:raise ValueError("Exact frozen family entry source required")
        deadline_arg(command,started["deadline_unix"],fit_mode)
        child=g.validate_guard(guard,spec,inputs.files[("cache",sp)],tag,command,started)
        if mode!="rescore":
            access=inputs.json("public",folder+"/PREFIX_ACCESS_RECEIPT.json")
            if (access.get("materialized_rows")!=t["prefix_end"] or access.get("new_T_numeric_values_parsed")!=0 or access.get("inherited_identity_status")!="INHERITED_NOT_RECOMPUTED"
                    or access.get("actual_A50_A60_and_V43_original_fingerprints_matched") is not True or access.get("old_probe_is_reused_not_reexecuted") is not True):raise ValueError("Exact same-dataset pre-T prefix/A50/A60/V43 fingerprint checks required")
        interval=dict(id=tag,start=guard["start_unix"],end=guard["end_unix"]);(cost_intervals if mode=="cost" else others).append(interval)
        evidence=dict(id=tag,guard_cache_relative_path=gp,guard_sha256=inputs.files[("cache",gp)],spec_sha256=inputs.files[("cache",sp)],outer_sha256=inputs.files[("public",folder+"/RESULT.json")],
            STARTED_sha256=inputs.files[("public",folder+"/STARTED.json")],sampled_peak_RSS_bytes=child["sampled_peak_RSS_bytes"],observed_owned_identities=len(child["owned_processes"]),current_PID_query_performed=False)
        guards.append(evidence)
        return folder,outer,command,evidence

    for t in targets():
        selected=[c for c in plan["conditions"] if c["run_id"]==t["run_id"]]
        if len(selected)!=1:raise ValueError("Exact original remaining recipe required")
        condition=selected[0];original=old["task_contracts"][t["dataset"]];channels=original["columns"]
        if condition["family"]!=t["family"] or condition["effective_F_width"]!=t["latent_k"] or len(channels)!=t["C"] or len(set(channels))!=t["C"]:raise ValueError("Same-dataset family/C/K contract differs")
        record=[r for r in fit["conditions"] if r["run_id"]==t["run_id"]]
        if len(record)!=1 or record[0].get("id")!=t["id"] or record[0].get("worker_science_complete") is not True:raise ValueError("Each selected peer must actually complete; partial audit cannot be PASS")
        run=NAME+"/runs/"+t["run_id"];prediction=NAME+"/predictions/"+t["run_id"]+"/V_select"
        receipt=inputs.json("public",run+"/BASE_FIT_RECEIPT.json");search=inputs.json("public",run+"/search/RECEIPT.json");refit=inputs.json("public",run+"/refit/RECEIPT.json")
        validate_training(receipt,search,refit,t,original["refit_scaler_sha256"])
        if not digest(receipt.get("artifact_sha256")) or not digest(receipt.get("model_state_sha")):raise ValueError("Actual selected artifact/model state SHA required")
        inputs.capture("cache",NAME+"/runs/"+t["run_id"]+"/selected_bundle.pt",receipt["artifact_sha256"])
        if receipt.get("artifact_cache_relative_path")!=NAME+"/runs/"+t["run_id"]+"/selected_bundle.pt":raise ValueError("Original exact selected artifact path differs")
        source=inputs.json("public",run+"/SOURCE_CONTRACT.json")
        config=fingerprint(dict(spec=condition["spec"],L=512,H=48,C=t["C"],columns=channels,inherited_array_sha256=original["array_sha256"],materialized_prefix_sha256=receipt["actual_prefix_sha256"]))
        if (source.get("spec")!=condition["spec"] or source.get("config_sha")!=config or source.get("source_sha")!=fingerprint(source["files"]) or receipt.get("source_sha")!=source["source_sha"]
                or source.get("data_sha")!=original["data_sha256"] or source.get("model_sha")!=F_SHA or source.get("original_task_identity")!="INHERITED_NOT_RECOMPUTED"
                or source.get("actual_prefix_sha256")!=receipt["actual_prefix_sha256"] or receipt.get("probe_weights_optimizer_or_sampler_reused") is not False or receipt.get("original_probe")!=condition["probe"]):raise ValueError("Actual A-only recipe/source/prefix/scaler lineage differs")
        for path,d in source["files"].items():inputs.capture("public",path,d)
        for stage,key,scaler in ((search,"search",original["search_scaler_sha256"]),(refit,"refit",original["refit_scaler_sha256"])):
            if stage.get("contract",{}).get("scaler_sha")!=scaler or any(stage["contract"].get(k)!=source[k] for k in ("source_sha","config_sha","data_sha","model_sha")):raise ValueError("Original dataset-specific A50/A60 stage contract differs")
        _,fit_outer,_,_=bound(t,t["id"],"fit","COMPLETE_FIT_AND_FULL_V",record[0],True)
        science=fit_outer["scientific_result"]
        if science.get("fit_receipt")!=receipt or science.get("full_V_completed") is not True or science.get("F_final_state_sha256")!=F_SHA:raise ValueError("Actual complete fit/fullV/restored F receipt differs")
        manifest=inputs.json("public",prediction+"/PREDICTION_MANIFEST.json");manifest_sha=inputs.files[("public",prediction+"/PREDICTION_MANIFEST.json")]
        reference=inputs.json("public",condition["V_reference"]["relative_path"],condition["V_reference"]["sha256"])
        want=dict(run_id=t["run_id"],family=t["family"],seed=92601,split="V_select",cohort=t["cohort"],device="cpu",CPU_threads=2,
            artifact_sha256=receipt["artifact_sha256"],data_sha=original["data_sha256"],scaler_sha=receipt["refit_scaler_sha"],source_sha=fingerprint(dict(runner=RUNNER_SHA,**plan["dependencies"])),
            config_sha=fingerprint(condition["spec"]),model_sha=fingerprint(dict(artifact_sha=receipt["artifact_sha256"],model_state_sha=receipt["model_state_sha"])),new_T_target_operations=0)
        if (any(manifest.get(k)!=v for k,v in want.items()) or manifest.get("F_counters")!=dict(streams=43*t["latent_k"],physical_streams=43*t["latent_k"],physical_calls=11)
                or any(manifest["score"].get(k)!=v for k,v in dict(n_origins=43,n_observed=t["observed"],horizon=48,num_channels=t["C"],metric_scale="A_normalized_original_channels").items())
                or manifest["arrays"]["pred_normalized"].get("shape")!=[43,48,t["C"]] or manifest["arrays"]["pred_normalized"].get("dtype")!="float32" or science.get("V_score")!=manifest["score"]):raise ValueError("Actual fullV43 same-dataset family/model/spec/F accounting differs")
        for key in ("target_normalized","observed","origins","channel_ids"):
            if manifest["arrays"][key]!=reference["arrays"][key]:raise ValueError("Same-dataset actual V target/mask/origin/channel fingerprint differs")
        saved_reference=inputs.json("public",prediction+"/REFERENCE_CONTRACT.json")
        if (any(saved_reference.get(k)!=manifest.get(k) for k in ("run_id","split","cohort","data_sha","scaler_sha","source_sha","config_sha","model_sha"))
                or saved_reference.get("arrays")!={k:manifest["arrays"][k] for k in ("target_normalized","observed","origins","channel_ids")}):raise ValueError("Saved reference contract differs from exact actual alias manifest")
        inputs.capture("cache",NAME+"/predictions/"+t["run_id"]+"/V_select.npz",manifest["archive_sha256"])
        proofs={};rescore=None
        for mode,status in (("rescore","PASS_INDEPENDENT_FULL_V_FP64"),("replay","PASS_FRESH_CPU6_REPLAY")):
            tag=t["id"].replace("_fit_root_v1","")+"_"+mode+"_root_v1";folder,outer,_,_=bound(t,tag,mode,status,steps[tag]);v=outer["scientific_result"]
            if v.get("run_id")!=t["run_id"] or v.get("artifact_sha256")!=receipt["artifact_sha256"] or v.get("manifest_sha256")!=manifest_sha:raise ValueError("Actual validation selected artifact/manifest differs")
            saved=inputs.json("public",folder+("/RESCORE_RECEIPT.json" if mode=="rescore" else "/REPLAY_RECEIPT.json"))
            if any(v.get(k)!=x for k,x in saved.items()):raise ValueError("Native validation receipt and actual outer differ")
            if mode=="rescore":
                if (v.get("full_V_origins")!=43 or v.get("n_observed")!=t["observed"] or v.get("public_stats_rows")!=t["stat_rows"] or v.get("saved_predictions_rescored") is not True or v.get("new_model_inference") is not False or outer.get("torch_imported") is not False or v.get("protected_T_target_operations")!=0):raise ValueError("Actual independent dataset-specific saved-array FP64 rescore required")
                rescore=v
            else:
                oo=v.get("origins",[])
                if (v.get("executed_now") is not True or v.get("microbatches")!=[4,2] or len(oo)!=6 or len(set(oo))!=6 or not set(oo).issubset(t["origins"]) or not {t["origins"][0],t["origins"][21],t["origins"][-1]}.issubset(oo)
                        or v.get("F_counters")!=dict(streams=6*t["latent_k"],physical_streams=6*t["latent_k"],physical_calls=2) or v.get("lora_merge_parity")!={"status":"NOT_LORA"}
                        or v.get("lora_merge_parity_F_counters")!={} or v.get("base_restored_after_original_Route_close") is not True or v.get("F_final_state_sha256")!=F_SHA or v.get("atol")!=1e-5 or v.get("rtol")!=1e-4):raise ValueError("Actual non-LoRA CPU6/4+2 replay/source restoration required")
            proofs[mode]=dict(path=inputs.roots["public"]/folder/"RESULT.json",sha256=inputs.files[("public",folder+"/RESULT.json")])
        statistics_rows=h.gzip_rows(inputs,"public",prediction+"/PUBLIC_ERROR_STATS.csv.gz",t["stat_rows"])
        aggregate=aggregate_statistics(statistics_rows,t["run_id"],channels,t["origins"],t["observed"],t["cohort"])
        scores=inputs.csv("public",prediction+"/SCORES.csv")
        if len(scores)!=1 or any(scores[0].get(k)!=manifest.get(k) for k in ("run_id","split","cohort")):raise ValueError("One exact same-dataset original score row required")
        deltas=h.compare_accuracy(aggregate,manifest,scores[0],rescore)
        accuracy.append(dict(dataset=t["dataset"],run_id=t["run_id"],**aggregate,metric_deltas=deltas))
        row=dict(receipt,dataset=t["dataset"],latent_k=t["latent_k"],C=t["C"],mse=aggregate["mse"],mae=aggregate["mae"],n_observed=t["observed"],
            manifest_sha256=manifest_sha,fit_receipt=receipt,search_stage=search,refit_stage=refit,accuracy_reused=False,cost_reused=False,F_resident_bytes=None,F_resident_bytes_scope="NOT_SEPARATELY_RECORDED")
        for batch in (1,8):
            tag=t["id"].replace("_fit_root_v1","")+"_cost_b"+str(batch)+"_root_v1";folder,outer,command,evidence=bound(t,tag,"cost","COMPLETE_PROTOCOL",steps[tag]);v=inputs.json("public",folder+"/COST_RECEIPT.json")
            for mode,proof in proofs.items():
                for flag,want in (("--"+mode+"-sha",proof["sha256"]),):
                    if command.count(flag)!=1 or command[command.index(flag)+1]!=want:raise ValueError("Actual cost argv proof SHA differs")
                flag="--"+mode+"-proof"
                if command.count(flag)!=1 or Path(command[command.index(flag)+1]).resolve()!=proof["path"].resolve():raise ValueError("Actual cost argv proof path differs")
            if command.count("--batch")!=1 or command[command.index("--batch")+1]!=str(batch):raise ValueError("Actual whole-batch cost argv differs")
            quiet_rel=PREP+"/QUIET_"+tag+".json";quiet=inputs.json("cache",quiet_rel,v["quiet_proof_sha256"])
            if command.count("--quiet-sha")!=1 or command[command.index("--quiet-sha")+1]!=v["quiet_proof_sha256"] or command.count("--quiet-proof")!=1 or Path(command[command.index("--quiet-proof")+1]).resolve()!=inputs.path("cache",quiet_rel).resolve():raise ValueError("Actual fresh quiet argv binding differs")
            guard=inputs.json("cache",NAME+"/supervisors/"+tag+"/RESULT.json",steps[tag]["guard_sha256"])
            if (quiet.get("status")!="QUIET_CPU_COST_WINDOW" or quiet.get("active_model_workers")!=[] or quiet.get("active_cost_workers")!=[] or not 0<=guard["start_unix"]-quiet["captured_unix"]<=120
                    or any(quiet.get(k)!=x for k,x in dict(fit_dispatch_result_sha256=bindings["dispatch_sha"],plan_sha256=PLAN_SHA,run_id=t["run_id"],artifact_sha256=receipt["artifact_sha256"],manifest_sha256=manifest_sha).items())):raise ValueError("Actual fresh matching quiet proof required")
            want=dict(status="COMPLETE_PROTOCOL",run_id=t["run_id"],device="cpu",CPU_threads=2,batch_size=batch,warmups=10,timed_trials=20,stage_trials=20,actual_trial_rows=50,
                accuracy_manifest_sha256=manifest_sha,artifact_sha256=receipt["artifact_sha256"],validation_result_sha256={k:p["sha256"] for k,p in proofs.items()},
                F_logical_streams_per_trial=batch*t["latent_k"],F_calls_per_trial=1,F_counters=dict(streams=50*batch*t["latent_k"],physical_streams=50*batch*t["latent_k"],physical_calls=50),
                origins=list(t["origins"][:batch]),cached_Z_or_forecast_reads=0,physical_output_and_inverse_dtype="float64",stage_probe_separate_intrusive_scope=True,training_peak_included=False,
                base_restored_after_original_Route_close=True,unmerged_F_base_sha256=F_SHA,effective_F_state_sha256_during_measurement=F_SHA,effective_F_state_scope="original frozen unmerged F",new_T_target_operations=0)
            if (any(v.get(k)!=x for k,x in want.items()) or any(outer["scientific_result"].get(k)!=x for k,x in v.items()) or outer["scientific_result"].get("F_final_state_sha256")!=F_SHA
                    or not digest(v.get("physical_input_sha256")) or not digest(v.get("observation_mask_sha256")) or v.get("memory",{}).get("foundation_instances_created")!=1):raise ValueError("Actual native full physical cost/precision/counters/proof/F restore differs")
            trial_rows=h.gzip_rows(inputs,"public",folder+"/COST_TRIALS.csv.gz",50,v["trials_sha256"])
            summed=helpers["cost"].aggregate_trials(trial_rows,t["run_id"],batch,t["latent_k"]);delta=h.compare_summaries(summed["summaries"],v["summaries"])
            cost=dict(dataset=t["dataset"],run_id=t["run_id"],latent_k=t["latent_k"],batch=batch,receipt=v,**summed,
                memory_scope=v["memory"],guard_sampled_peak_RSS_bytes=evidence["sampled_peak_RSS_bytes"],max_summary_difference_seconds=delta,
                public_diagnostic_relative_path=folder,physical_input_sha256=v["physical_input_sha256"],observation_mask_sha256=v["observation_mask_sha256"],
                derived_trial_F_counters=dict(F_logical_streams=summed["F_logical_streams"],F_physical_streams=summed["F_physical_streams"],F_calls=summed["F_calls"],scope="INDEPENDENTLY_SUMMED_FROM_50_PUBLIC_TRIAL_ROWS"),
                F_resident_bytes=None,F_resident_bytes_scope="NOT_SEPARATELY_RECORDED")
            row["cost_B"+str(batch)]=cost;costs.append(cost)
        rows.append(row)
    h.validate_intervals(cost_intervals,others)
    for a in accuracy:a["band_count_binding_sha256"]=fingerprint(a.pop("count_by_key"))
    return dict(rows=rows,accuracy=accuracy,cost_conditions=costs,recorded_guards=guards,recorded_cost_intervals=cost_intervals,
        costs_disjoint_from_recorded_fit_validation_and_cost_intervals=True,dataset_cohorts_compared_separately=True,
        totals=dict(new_fit_conditions=2,accuracy_conditions=2,public_stat_rows=4128,cost_conditions=4,trial_rows=200,summary_groups=8,summary_numeric_fields_checked=40),initial_two_recomputed=False,initial_two_retrained=False)


def execute(public,cache,bindings,audit_sha,output):
    output=Path(output).absolute()
    if output.exists():raise FileExistsError("Existing audit output preserved before inputs")
    public,cache=Path(public).resolve(),Path(cache).resolve()
    required={"plan_sha","dispatch_sha","dispatch_spec_sha","dispatch_guard_sha","post_sha"}
    if not output.resolve().is_relative_to(public/NAME) or output.suffix!=".json" or set(bindings)!=required or not all(digest(v) for v in bindings.values()) or bindings["plan_sha"]!=PLAN_SHA:
        raise ValueError("New namespace JSON and exact original PLAN/future actual dispatch+SPEC+guard+POST SHA required")
    if Path(__file__).resolve()!=public/NAME/"audit_remaining_family.py" or not digest(audit_sha) or sha(__file__)!=audit_sha:raise ValueError("Current ROOT-reviewed audit source SHA required")
    helpers=load_helpers(public);inputs=helpers["inputs"].Inputs(public,cache);inputs.capture("public",NAME+"/audit_remaining_family.py",audit_sha)
    result={};error=None
    try:result=collect(inputs,helpers,bindings)
    except (ValueError,KeyError,TypeError,IndexError,OverflowError,OSError,csv.Error) as failure:error=dict(error_class=type(failure).__name__,reason=str(failure) if isinstance(failure,ValueError) else "Bound metadata, closure or arithmetic unavailable/rejected")
    try:preserved=inputs.preserved()
    except OSError:preserved=False
    if not preserved:error=error or dict(error_class="InputPreservationFailed",reason="Captured input bytes changed before output")
    receipt=dict(schema="INDEPENDENT_REMAINING_FAMILY_AUDIT_v1",status="PASS" if error is None and preserved else "NOT_COMPLETE_AUDIT_EVIDENCE_REJECTED",audit_source_sha256=audit_sha,
        ROOT_bound_execution_sha256=dict(bindings,dispatch_source_sha=DISPATCH_SHA,post_source_sha=POST_SHA),source48_preserved=error is None and preserved,input_SHA_before_after_equal=preserved,
        input_count=len(inputs.files),inputs=[dict(root=root,relative_path=path,sha256=d) for (root,path),d in sorted(inputs.files.items())],error=error,**result,
        accuracy_comparison_tolerance=dict(atol=1e-9,rtol=1e-7,source="Original priority_runner.independent_rescore; frozen933883 compare_accuracy"),
        cost_comparison_policy="Exact original fsum mean/median/min/max/linear p90; any difference rejects completion without a new tolerance.",
        current_PID_query_performed=False,new_Task_raw_NPZ_checkpoint_tensor_model_training_inference_cost_process_operations=0,whole_MASTER_complete=False,independent_generalization_claimed=False,
        limitations=["Jena and Beijing have separate origins, mask/target/channel fingerprints and channel-macro metrics; do not pool or treat counts as identical.",
            "Single seed and repeatedly inspected development V; selected0/.25/1/3 is read from actual receipts. Search3 does not imply selected refit3, convergence or generalization.",
            "Actual full-array FP64 and fresh CPU6 proofs are bound prior executions; this auditor only recomputes public sufficient statistics/trials.",
            "Recorded guard/SPEC/STARTED/tree closure only; ROOT performs current PID-creation-time absence and campaign quiet checks.",
            "Family native receipts record counters/proofmap/dtype. Keep those scopes separate from trial-derived totals and F-only resident memory, which is not separately measured.",
            "Native execute validates/loads the saved V archive before entering cost(). The50 actual physical Route forwards do not use cached forecasts; cached_Z_or_forecast_reads=0 is not a claim of zero archive I/O for the whole worker.",
            "Original probe/full-task deserialization ancestry remains historical. Current prefix receipts require zero newly parsed T numeric values.",
            "No initial-two result recomputation/retraining, protected T scoring or automatic peer dispatch. Partial means NOT_COMPLETE and does not cancel independent execution."])
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf-8",newline="\n") as stream:json.dump(receipt,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")
    return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--execute",action="store_true");p.add_argument("--root-triggered",action="store_true")
    keys=("public-root","cache-root","plan-sha","dispatch-sha","dispatch-spec-sha","dispatch-guard-sha","post-sha","audit-sha","output")
    for key in keys:p.add_argument("--"+key)
    a=p.parse_args(argv)
    if not a.execute:
        print(json.dumps(dict(status="PREPARATION_ONLY_NOT_EXECUTED",input_reads=0,files_written=0,scientific_operations=0,numpy_imported="numpy" in sys.modules,torch_imported="torch" in sys.modules,future_dispatch_SHA=None,future_POST_SHA=None)));return 0
    if not a.root_triggered or any(getattr(a,k.replace("-","_")) is None for k in keys):p.error("ROOT explicit roots/current source and future actual closed execution SHA bindings required")
    try:r=execute(a.public_root,a.cache_root,{k:getattr(a,k) for k in ("plan_sha","dispatch_sha","dispatch_spec_sha","dispatch_guard_sha","post_sha")},a.audit_sha,a.output)
    except (ValueError,FileExistsError,FileNotFoundError) as error:
        print(json.dumps(dict(status="NOT_COMPLETE_AUDIT_EVIDENCE_REJECTED",error_class=type(error).__name__)));return 1
    print(json.dumps(dict(status=r["status"],input_count=r["input_count"],files_written=1)));return 0 if r["status"]=="PASS" else 1


if __name__=="__main__":raise SystemExit(main())
