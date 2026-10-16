"""Build N4 reports only after actual full-V, FP64, CPU6 and B1/B8 closure.

Preparation source only: default0 I/O, stdlib fixtures only before ROOT's
three-fit queue closes. No model/NPZ/statistic decoding or experiment dispatch.
Missing, failed or ambiguous evidence produces no completed report.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

NAME="priority_valid_minima_20261016_v1"
OLD="postT_V_only_cpu_required_minima_20261016_v1"
N3="priority_nested_cost_20261016_v1"
MULTIK="postT_V_only_cpu_multik_20261016_v1"
QUEUE_RESULT="priority_pause_validate_20261016_v1/FOLLOWUP_ROOT_v1/BEIJING_MINIMA_EXECUTION_ROOT_v1/RESULT.json"
POST_ROOT="priority_pause_validate_20261016_v1/FOLLOWUP_ROOT_v1/BEIJING_MINIMA_POST_ROOT_v1"
PLAN_SHA="6f7baaad7200c7c6601bce5a0cb6f6fdff7c3497e3a26b36906408c2844ae0f3"
RUNNER_SHA="b84f1473ff4d562890910f7555ef3925823dd921539c8d757118c95504513b9b"
DATA_SHA="b39023bcae04736eff9a0fd9a3d56e10dc7f32bd668a53d681290cb9193d8269"
SCALER_SHA="3348bcf1fa76d382f23cf20597fa90de6e21dac587d4ca9f6883f235a10f93fb"
F_SHA="c2b4edf7db22d3f18e58ca79299a5c7e864e9b8f277e4e29e2f88d07dbb35874"
V_FINGERPRINTS={
    "target_normalized":dict(shape=[43,48,11],dtype="float32",sha256="1b03cad2bb2fca92dbc19ab4f588a4e194b191323b2890bddfe4530951689fa7"),
    "observed":dict(shape=[43,48,11],dtype="bool",sha256="b0775eee2288b021f260db23f36bbbe6364231390a28b2f810d05bd2ed79e7c2"),
    "origins":dict(shape=[43],dtype="int64",sha256="596d5fac3452cb6d00167e76156fcfd05331410d86b10dadcc896037c25ca0a6"),
    "channel_ids":dict(shape=[11],dtype="<U5",sha256="4688e20fc5d503ff58a436744491abce205c1cd6930f54f881eccb27430854ac"),
}
TARGETS=(
    ("B2","postT_cpu_beijing_E1_B2_k6_s92601_required_v1","beijing_b2_fit_root_v1"),
    ("LAG_CAUSAL","postT_cpu_beijing_E3_LAG_CAUSAL_k6_s92601_required_v1","beijing_causal_fit_root_v1"),
    ("LAG_BIDIR","postT_cpu_beijing_E3_LAG_BIDIR_k6_s92601_required_v1","beijing_bidir_fit_root_v1"),
)
BASELINES=(
    ("RAW_SMALL","primary_beijing_E1_RAW_BOLT_SMALL_k0_s92601","9a1aaa1ce8f448555543a2082fe79c608d11f055694820f9701777ffec96b481","RAW"),
    ("G_INITIAL025","primary_beijing_E1_DIRECT_G_k0_s92601","9ade5a0284fef8e5dfa7aa3dd03ed07645c02794defcb1241074c9b806d76029","DIRECT_G"),
    ("G_DEEP_E3","primary_beijing_E1_DIRECT_G_k0_s92601__e1__e3","30952d235af0bc4e249b45e04f1e3099b8a5160b1a4426572ca71d70ebd746db","DIRECT_G"),
    ("RIDGE_NATIVE","primary_beijing_E1_RAW_RIDGE_k0_s92601","114ed74bf1d8257df0e8d879f783d64f39a9cea54648399041acf5af8b24107c","RAW_RIDGE"),
)
OUTPUTS=("RESULTS_WITH_COST_v1.csv","RESULTS_KO_v1.md","REPORT_INPUT_BINDINGS_v1.json")
FIELDS=("role","run_id","K","MSE","MAE","accuracy_device","comparison_scope","training_device","search_max_epoch","selected_refit_epoch",
    "search_updates","refit_updates","search_exposures","refit_exposures","training_status","search_seconds","refit_seconds",
    "B1_median_seconds","B1_p90_seconds","B8_median_seconds","B8_p90_seconds","cost_device","cost_scope","B1_CPU_weights_MiB","B8_CPU_weights_MiB",
    "B1_observed_tree_peak_RSS_bytes","B8_observed_tree_peak_RSS_bytes","observed_labels","V_origins","notes")


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


class Inputs:
    def __init__(self,public,cache):self.roots={"public":Path(public).resolve(),"cache":Path(cache).resolve()};self.files={}

    def path(self,root,name):
        base=self.roots[root];path=base/name
        if (Path(name).is_absolute() or ".." in Path(name).parts or path.is_symlink() or not path.resolve().is_relative_to(base)):
            raise ValueError("Explicit root-relative nonsymlink report inputs required")
        return path

    def capture(self,root,name,expected=None):
        path=self.path(root,name);digest=sha(path);key=(root,str(name).replace("\\","/"))
        if expected is not None and digest!=expected:raise ValueError("Pinned report input changed: "+str(name))
        if key in self.files and self.files[key]!=digest:raise ValueError("Report input changed during reading")
        self.files[key]=digest;return path

    def json(self,root,name,expected=None):
        path=self.capture(root,name,expected)
        if path.stat().st_size>8*1024**2:raise ValueError("Small JSON evidence only")
        return json.loads(path.read_text(encoding="utf-8-sig"))

    def csv(self,root,name):
        with self.capture(root,name).open(encoding="utf-8-sig",newline="") as stream:return list(csv.DictReader(stream))

    def preserved(self):return all(sha(self.path(root,name))==digest for (root,name),digest in self.files.items())


def validate_cohort(manifest):
    score=manifest["score"]
    if (manifest.get("split")!="V_select" or manifest.get("cohort")!="beijing:primary:V_select:stride48"
            or manifest.get("data_sha")!=DATA_SHA or manifest.get("scaler_sha")!=SCALER_SHA
            or any(manifest.get("arrays",{}).get(key)!=value for key,value in V_FINGERPRINTS.items())
            or any(score.get(key)!=value for key,value in dict(n_origins=43,n_observed=20539,num_channels=11,horizon=48,metric_scale="A_normalized_original_channels").items())):
        raise ValueError("Same original V43/mask/target/origins/channels/A60 normalization required")
    if any(not math.isfinite(float(score[key])) or float(score[key])<0 for key in ("mse","mae")):raise ValueError("Finite nonnegative V errors required")


def validate_outer(value,identifier,mode,status):
    if (value.get("run_id")!=identifier or value.get("mode")!=mode or value.get("status")!=status or value.get("plan_sha256")!=PLAN_SHA
            or value.get("error_class") is not None or value.get("postflight_error_class") is not None
            or value.get("original_source39_immutable9_preserved") is not True or value.get("new_T_target_operations")!=0):
        raise ValueError("Exact successful outer RESULT and postflight source/T closure required")


def validate_validation(science,mode,artifact,manifest_sha):
    if science.get("artifact_sha256")!=artifact or science.get("manifest_sha256")!=manifest_sha:raise ValueError("Validation belongs to another selected artifact or V manifest")
    if mode=="rescore":
        if (science.get("status")!="PASS_INDEPENDENT_FULL_V_FP64" or science.get("saved_predictions_rescored") is not True
                or science.get("new_model_inference") is not False or science.get("n_observed")!=20539 or science.get("full_V_origins")!=43):raise ValueError("Actual complete saved-V FP64 rescore required")
    elif mode=="replay":
        if (science.get("status")!="PASS_FRESH_CPU6_REPLAY" or science.get("executed_now") is not True or science.get("microbatches")!=[4,2]
                or science.get("F_counters")!=dict(streams=36,physical_streams=36,physical_calls=2)
                or science.get("atol")!=1e-5 or science.get("rtol")!=1e-4 or len(science.get("origins",[]))!=6):raise ValueError("Actual freshCPU6/4+2/K6 replay required")
    else:raise ValueError("Only actual rescore/replay proofs accepted")


def validate_training(receipt,search,refit,identifier,family):
    epoch=receipt.get("selected_epoch")
    if (epoch not in (0.,.25,1.,3.) or receipt.get("status")!="COMPLETE_VALID" or receipt.get("run_id")!=identifier or receipt.get("family")!=family
            or receipt.get("seed")!=92601 or receipt.get("epochs")!=[.25,1.,3.] or receipt.get("refit_scaler_sha")!=SCALER_SHA):raise ValueError("Exact completed three-epoch search/fresh selected refit receipt required")
    expected=(dict(status="SEGMENT_COMPLETE",completed_epochs=[0.,.25,1.,3.],train_origins=9960,exposures=29880,updates=1868,selected_epoch=epoch),
        dict(status="SEGMENT_COMPLETE",completed_epochs=[0.]+([epoch] if epoch else []),train_origins=12063,exposures=math.ceil(epoch*12063),updates=math.ceil(math.ceil(epoch*12063)/16)))
    for stage,want in zip((search,refit),expected):
        if any(stage.get(key)!=value for key,value in want.items()):raise ValueError("Actual stage endpoint/exposure/update record differs")
    if any(receipt.get(label+"_"+key)!=stage[key] for label,stage in (("search",search),("refit",refit)) for key in ("exposures","updates")):raise ValueError("Fit receipt and original stage activity disagree")


def validate_cost(receipt,identifier,batch,artifact,manifest_sha,trials_sha,proof_shas):
    want=dict(status="COMPLETE_PROTOCOL",run_id=identifier,device="cpu",CPU_threads=2,batch_size=batch,warmups=10,timed_trials=20,stage_trials=20,actual_trial_rows=50,
        accuracy_manifest_sha256=manifest_sha,artifact_sha256=artifact,trials_sha256=trials_sha,validation_result_sha256=proof_shas,
        F_logical_streams_per_trial=batch*6,F_calls_per_trial=1,cached_Z_or_forecast_reads=0,physical_output_and_inverse_dtype="float64",stage_probe_separate_intrusive_scope=True,training_peak_included=False,new_T_target_operations=0)
    if any(receipt.get(key)!=value for key,value in want.items()):raise ValueError("Whole physical CPU2/B1B8 protocol or artifact/validation binding differs")
    if receipt.get("F_physical_streams_per_trial",batch*6)!=batch*6:raise ValueError("Physical F streams differ")
    if receipt.get("origins")!=list(range(14726,16783,48))[:batch]:raise ValueError("Cost origins differ from exact first V origins")
    for phase in ("timed","stage_probe"):
        group=[s for s in receipt.get("summaries",[]) if s.get("phase")==phase]
        if len(group)!=1 or group[0].get("trials")!=20 or group[0].get("batch_size")!=batch:raise ValueError("Original20 timed and20 separate stage trial summaries required")
        if any(not math.isfinite(float(group[0][key])) or float(group[0][key])<=0 for key in ("median_seconds","p90_seconds")):raise ValueError("Finite positive cost summary required")


def validate_guard(guard,identifier):
    if (guard.get("status") not in {"COMPLETE","STOPPED_RESOURCE_OR_WORKER_FAILURE"} or guard.get("helper_unchanged") is not True or guard.get("remaining_owned_processes")!=[]
            or guard.get("supervisor_failure_class") is not None):raise ValueError("Actual external supervisor closure required")
    processes=guard.get("processes",[])
    if sum(p.get("id")==identifier for p in processes)!=1:raise ValueError("One exact guarded child identity required")
    if any(p.get("returncode") is None or p.get("remaining_owned_processes")!=[] or p.get("observed_tree_join_confirmed") is not True or p.get("process_tree_errors")!=[] for p in processes):raise ValueError("Unclosed observed process tree cannot support report")
    selected=next(p for p in processes if p["id"]==identifier)
    if selected.get("returncode")!=0 or selected.get("stop_reason") is not None or selected.get("signals")!=[]:raise ValueError("Exact selected child failed or was signalled")


def select_result(candidates,identifier,mode,batch=None):
    found=[(path,value) for path,value in candidates if value.get("run_id")==identifier and value.get("mode")==mode
        and (mode!="cost" or (value.get("scientific_result") or {}).get("batch_size")==batch)]
    if len(found)!=1:raise ValueError("Missing or ambiguous actual per-condition mode evidence; not completed")
    return found[0]


def guard_for(inputs,diagnostic,outer):
    target=(inputs.roots["public"]/diagnostic).resolve();matches=[]
    for path in sorted((inputs.roots["cache"]/NAME/"supervisors").glob("*/RESULT.json")):
        relative=path.relative_to(inputs.roots["cache"]).as_posix();guard=inputs.json("cache",relative)
        for entry in guard.get("effective_spec",{}).get("processes",[]):
            cmd=entry.get("command",[])
            if "--output" in cmd and Path(cmd[cmd.index("--output")+1]).resolve()==target:
                matches.append((relative,guard,entry))
    if len(matches)!=1:raise ValueError("Actual unique supervisor command/output binding missing")
    path,guard,entry=matches[0];validate_guard(guard,entry["id"]);cmd=entry["command"]
    if guard.get("schema")!="CPU_GROUP_SUPERVISOR_RESULT_v1":raise ValueError("Original full supervisor schema required")
    spec_root=QUEUE_RESULT.rsplit("/",1)[0] if outer["mode"]=="fit" else POST_ROOT
    spec=inputs.json("cache",spec_root+"/SPEC_"+Path(path).parent.name+".json",guard["spec_file_sha256"])
    if ({p["id"] for p in spec["processes"]}!={p["id"] for p in guard["processes"]}
            or len(spec["processes"])!=len(guard["processes"])
            or next(p["command"] for p in spec["processes"] if p["id"]==entry["id"])!=cmd):raise ValueError("Actual guard worker IDs/command differ from original exact spec")
    if any(p.get("end_unix") is None or not p.get("owned_processes") for p in guard["processes"]):raise ValueError("All original guarded children need terminal identity evidence")
    for flag,want in (("--run-id",outer["run_id"]),("--mode",outer["mode"]),("--plan-sha",PLAN_SHA)):
        if flag not in cmd or cmd[cmd.index(flag)+1]!=want:raise ValueError("Actual guarded command has different scientific identity")
    start=inputs.json("public",diagnostic+"/STARTED.json");child=next(p for p in guard["processes"] if p["id"]==entry["id"])
    if (not any(p["pid"]==start["actual_pid"] and abs(p["create_time"]-start["actual_create_time"])<=1e-6 for p in child["owned_processes"])
            or guard["helper_sha256"]!="82a837ea0cfee8b52765f4ac1c7325d564e886c9e94857bd0f957e4a40c75f8b"
            or guard["effective_spec"]["max_each_RSS"]!=3*1024**3 or guard["effective_spec"]["max_group_RSS"]>6*1024**3
            or child["sampled_peak_RSS_bytes"]>=3*1024**3):raise ValueError("Exact worker creation-time/source/RSS guard differs")
    if outer["mode"]=="cost":
        if "--quiet-proof" not in cmd or "--quiet-sha" not in cmd:raise ValueError("Actual cost quiet binding missing")
        quiet_path=Path(cmd[cmd.index("--quiet-proof")+1]).resolve()
        if not quiet_path.is_relative_to(inputs.roots["cache"]):raise ValueError("Private quiet proof outside cache")
        quiet_sha=cmd[cmd.index("--quiet-sha")+1];quiet=inputs.json("cache",quiet_path.relative_to(inputs.roots["cache"]).as_posix(),quiet_sha)
        if (quiet.get("status")!="QUIET_CPU_COST_WINDOW" or quiet.get("active_model_workers")!=[] or quiet.get("active_cost_workers")!=[]
                or not 0<=guard["start_unix"]-quiet["captured_unix"]<=120 or outer["scientific_result"]["quiet_proof_sha256"]!=quiet_sha):raise ValueError("Actual fresh quiet proof inconsistent")
    return guard,child,path


def baseline_scope(manifest,family):
    if family=="RAW_RIDGE":return "CPU_NATIVE_NUMPY_SOURCE_TRACED"
    if manifest.get("device")=="cpu":return "REUSED_CPU_FULL_V"
    return "REFERENCE_DEVICE_NOT_RECORDED" if manifest.get("device") is None else "REFERENCE_DEVICE_NOT_CPU"


def score_csv(inputs,folder,manifest):
    scores=inputs.csv("public",folder+"/SCORES.csv")
    if len(scores)!=1 or any(scores[0][key]!=manifest[key] for key in ("run_id","split","cohort")) or any(float(scores[0][key])!=manifest["score"][key] for key in ("mse","mae")):raise ValueError("Exact manifest/score CSV identity required")
    return scores[0]


def baseline_rows(inputs,physical_inputs):
    rows=[]
    for role,identifier,digest,family in BASELINES:
        folder="predictions/"+identifier+"/V_select";manifest=inputs.json("public",folder+"/PREDICTION_MANIFEST.json",digest);validate_cohort(manifest);score_csv(inputs,folder,manifest)
        if manifest.get("metadata",{}).get("family")!=family:raise ValueError("Exact distinct baseline family required")
        row=dict(role=role,run_id=identifier,K=11 if family=="RAW" else 0,MSE=manifest["score"]["mse"],MAE=manifest["score"]["mae"],accuracy_device="cpu_NATIVE_NUMPY_SOURCE_TRACED" if family=="RAW_RIDGE" else manifest.get("device","NOT_RECORDED_IN_MANIFEST"),comparison_scope=baseline_scope(manifest,family),observed_labels=20539,V_origins=43,cost_scope="NOT_INCLUDED_NO_EXACT_CPU_COST_BINDING",notes="Reused saved original V; device and training scope remain separate")
        if family=="DIRECT_G":
            receipt=inputs.json("public","runs/"+identifier+"/BASE_FIT_RECEIPT.json");recipe=inputs.json("public","runs/"+identifier+"/SOURCE_CONTRACT.json")
            row.update(training_device=recipe["spec"]["device"],search_max_epoch=max(receipt["epochs"]),selected_refit_epoch=receipt["selected_epoch"],training_status=receipt["training_status"])
            for key in ("search_updates","refit_updates","search_exposures","refit_exposures"):row[key]=receipt[key]
            row["notes"]="Exact G ID retained: initial.25 and extended3-search are separate; original V device is not explicitly recorded"
        else:row.update(training_device="cpu_CLOSED_FORM" if family=="RAW_RIDGE" else "NO_DATASET_OPTIMIZER",training_status="CLOSED_FORM_ALL_LEGAL_A_ORIGINS" if family=="RAW_RIDGE" else "FROZEN_OFFICIAL_MODEL")
        rows.append(row)
    rid="postT_cpu_beijing_E4_NESTED_PCA_k6_s92601_v1";folder=MULTIK+"/ordered_predictions/"+rid
    manifest=inputs.json("public",folder+"/PREDICTION_MANIFEST.json","b624e18f2586a71c288d55c9b2811addb2955bebf2eb9f98c495e88a584e14fb");validate_cohort(manifest);score_csv(inputs,folder,manifest)
    audit=inputs.json("public",N3+"/INDEPENDENT_COST_REVIEW_ROOT_v1.json","2d00f17d0520d6cd9482d304f4b403b5cf7da613d5faec9a9cd92483ac090c5b")
    if audit.get("status")!="PASS" or audit.get("schema")!="INDEPENDENT_NESTED_COST_REVIEW_v1":raise ValueError("Existing exact nested PCA CPU cost audit required")
    row=dict(role="PCA_FIXED_NESTED_K6",run_id=rid,K=6,MSE=manifest["score"]["mse"],MAE=manifest["score"]["mae"],accuracy_device="cpu",comparison_scope="REUSED_CPU_V43_FP64_AND_CPU6",training_device="NO_NEURAL_OPTIMIZER",training_status="FIXED_PCA_A60_FIT_TIME_NOT_RECORDED",search_updates=0,refit_updates=0,cost_device="cpu",cost_scope="REUSED_AUDITED_CPU2_FULL_PHYSICAL",observed_labels=20539,V_origins=43,notes="Fixed basis inverse+center differs from learned B2 lag decoder/current anchor")
    for batch in (1,8):
        proof=next(c for c in audit["cost_conditions"] if c["latent_k"]==6 and c["batch"]==batch);base=proof["public_diagnostic_relative_path"]
        receipt=inputs.json("public",base+"/COST_RECEIPT.json",proof["input_receipt_SHA256"]["COST_RECEIPT.json"])
        if ((receipt["physical_input_sha256"],receipt["observation_mask_sha256"])!=physical_inputs[batch] or receipt["accuracy_manifest_sha256"]!=inputs.files[("public",folder+"/PREDICTION_MANIFEST.json")]
                or receipt["device"]!="cpu" or receipt["CPU_threads"]!=2):raise ValueError("PCA cost physical input/mask/CPU/fullV binding differs from new comparisons")
        inputs.capture("public",base+"/RESULT.json",proof["input_receipt_SHA256"]["RESULT.json"]);inputs.capture("public",base+"/COST_TRIALS.csv.gz",receipt["trials_sha256"])
        timed=next(s for s in receipt["summaries"] if s["phase"]=="timed");row.update({f"B{batch}_median_seconds":timed["median_seconds"],f"B{batch}_p90_seconds":timed["p90_seconds"],f"B{batch}_CPU_weights_MiB":receipt["memory"]["resident_weights_cpu_MiB"],f"B{batch}_observed_tree_peak_RSS_bytes":proof["guard"]["sampled_group_peak_RSS_bytes"]})
    rows.append(row);return rows


def assemble(inputs,queue,post):
    plan=inputs.json("public",NAME+"/PLAN.json",PLAN_SHA);inputs.capture("public",NAME+"/beijing_minimum_runner.py",RUNNER_SHA)
    if plan.get("status")!="SEALED_BY_ROOT" or plan.get("runner_sha256")!=RUNNER_SHA or [(r["family"],r["run_id"]) for r in plan["conditions"]]!=[(f,r) for f,r,_ in TARGETS]:raise ValueError("Exact sealed N4 three-condition plan required")
    old=inputs.json("public",OLD+"/PLAN.json",plan["original_required_plan_sha256"])
    if len(old["source39"])!=39 or len(old["immutable_public"])!=9:raise ValueError("Original48 source/immutable pins required")
    for name,digest in {**old["source39"],**old["immutable_public"],**plan["dependencies"]}.items():inputs.capture("public",name,digest)
    reference=inputs.json("public",plan["V_reference_public_relative_path"],plan["V_reference_sha256"]);validate_cohort(reference)
    candidates=[(p.parent.relative_to(inputs.roots["public"]).as_posix(),inputs.json("public",p.relative_to(inputs.roots["public"]).as_posix())) for p in sorted((inputs.roots["public"]/NAME/"diagnostics").glob("*/RESULT.json"))]
    rows=[];cost_input={};evidence=[];post_steps={r["id"]:r for r in post["steps"]}
    if len(post_steps)!=12:raise ValueError("Twelve unique actual postprocessing steps required")
    for family,identifier,fit_tag in TARGETS:
        selected={};proof_shas={};run=NAME+"/runs/"+identifier;folder=NAME+"/predictions/"+identifier+"/V_select"
        manifest=inputs.json("public",folder+"/PREDICTION_MANIFEST.json");validate_cohort(manifest);score_csv(inputs,folder,manifest);manifest_sha=inputs.files[("public",folder+"/PREDICTION_MANIFEST.json")]
        receipt=inputs.json("public",run+"/BASE_FIT_RECEIPT.json");search=inputs.json("public",run+"/search/RECEIPT.json");refit=inputs.json("public",run+"/refit/RECEIPT.json");validate_training(receipt,search,refit,identifier,family)
        spec=next(c["spec"] for c in plan["conditions"] if c["run_id"]==identifier)
        if (manifest.get("run_id")!=identifier or manifest.get("family")!=family or manifest.get("seed")!=92601 or manifest.get("device")!="cpu"
                or manifest.get("artifact_sha256")!=receipt["artifact_sha256"] or manifest.get("config_sha")!=fingerprint(spec)
                or manifest.get("source_sha")!=fingerprint(dict(runner=RUNNER_SHA,**plan["dependencies"]))
                or manifest.get("model_sha")!=fingerprint(dict(artifact_sha=receipt["artifact_sha256"],model_state_sha=receipt["model_state_sha"]))
                or manifest.get("F_counters")!=dict(streams=258,physical_streams=258,physical_calls=11)):raise ValueError("Actual fullCPU V43 selected-checkpoint identity differs")
        for mode,status in (("fit","COMPLETE_FIT_AND_FULL_V"),("rescore","PASS_INDEPENDENT_FULL_V_FP64"),("replay","PASS_FRESH_CPU6_REPLAY")):
            path,outer=select_result(candidates,identifier,mode);validate_outer(outer,identifier,mode,status);guard,child,gpath=guard_for(inputs,path,outer);selected[mode]=(path,outer)
            proof_shas[mode]=inputs.files[("public",path+"/RESULT.json")];evidence.append(dict(run_id=identifier,mode=mode,public_result=path+"/RESULT.json",guard_cache_relative_path=gpath))
            if mode!="fit":
                validate_validation(outer["scientific_result"],mode,receipt["artifact_sha256"],manifest_sha)
                step=post_steps.get(Path(path).name,{})
                if step.get("success") is not True or step.get("outer_sha256")!=proof_shas[mode] or step.get("guard_sha256")!=inputs.files[("cache",gpath)]:raise ValueError("Actual POST step closure/SHA disagrees")
        fitpath,fitouter=selected["fit"];science=fitouter["scientific_result"]
        if fitpath!=NAME+"/diagnostics/"+fit_tag or science.get("full_V_completed") is not True or science.get("fit_receipt")!=receipt or science.get("V_score")!=manifest["score"]:raise ValueError("Actual queue fit/fullV receipt mismatch")
        queue_row=next((r for r in queue["conditions"] if r.get("run_id")==identifier),{})
        if queue_row.get("outer_sha256")!=proof_shas["fit"] or queue_row.get("id")!=fit_tag or queue_row.get("outer_status")!="COMPLETE_FIT_AND_FULL_V":raise ValueError("Original actual fit queue per-child outer/SHA differs")
        if science.get("F_final_state_sha256")!=F_SHA or selected["replay"][1]["scientific_result"].get("F_final_state_sha256")!=F_SHA:raise ValueError("Actual final frozen F state proof missing")
        row=dict(role=family,run_id=identifier,K=6,MSE=manifest["score"]["mse"],MAE=manifest["score"]["mae"],accuracy_device="cpu",comparison_scope="NEW_CPU_V43_FP64_CPU6",training_device="cpu",search_max_epoch=3,selected_refit_epoch=receipt["selected_epoch"],training_status=receipt["training_status"],search_seconds=search["seconds"],refit_seconds=refit["seconds"],cost_device="cpu",cost_scope="NEW_CPU2_FULL_PHYSICAL_PROTOCOL",observed_labels=20539,V_origins=43,notes="Repeated development V; frozen F exact; source-fixed recipes, selected epoch may differ")
        for key in ("search_updates","refit_updates","search_exposures","refit_exposures"):row[key]=receipt[key]
        for batch in (1,8):
            path,outer=select_result(candidates,identifier,"cost",batch);validate_outer(outer,identifier,"cost","COMPLETE_PROTOCOL");guard,child,gpath=guard_for(inputs,path,outer)
            cost_receipt=inputs.json("public",path+"/COST_RECEIPT.json");inputs.capture("public",path+"/COST_TRIALS.csv.gz",cost_receipt["trials_sha256"])
            step=post_steps.get(Path(path).name,{})
            if step.get("success") is not True or step.get("outer_sha256")!=inputs.files[("public",path+"/RESULT.json")] or step.get("guard_sha256")!=inputs.files[("cache",gpath)]:raise ValueError("Actual cost POST step/SHA disagrees")
            validate_cost(cost_receipt,identifier,batch,receipt["artifact_sha256"],manifest_sha,inputs.files[("public",path+"/COST_TRIALS.csv.gz")],{k:proof_shas[k] for k in ("rescore","replay")})
            if any(outer["scientific_result"].get(k)!=v for k,v in cost_receipt.items()) or outer["scientific_result"].get("F_final_state_sha256")!=F_SHA:raise ValueError("Actual cost receipt/outer/frozen F mismatch")
            input_pair=(cost_receipt["physical_input_sha256"],cost_receipt["observation_mask_sha256"])
            if batch in cost_input and cost_input[batch]!=input_pair:raise ValueError("New B2/lag physical inputs or masks differ")
            cost_input[batch]=input_pair;timed=next(s for s in cost_receipt["summaries"] if s["phase"]=="timed")
            row.update({f"B{batch}_median_seconds":timed["median_seconds"],f"B{batch}_p90_seconds":timed["p90_seconds"],f"B{batch}_CPU_weights_MiB":cost_receipt["memory"]["resident_weights_cpu_MiB"],f"B{batch}_observed_tree_peak_RSS_bytes":child["sampled_peak_RSS_bytes"]})
            evidence.append(dict(run_id=identifier,mode="cost",batch=batch,public_result=path+"/RESULT.json",guard_cache_relative_path=gpath))
        rows.append(row)
    return rows+baseline_rows(inputs,cost_input),evidence


def csv_text(rows,fields):
    stream=io.StringIO(newline="");writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n");writer.writeheader();writer.writerows({k:r.get(k,"") for k in fields} for r in rows);return stream.getvalue()


def compose(rows):
    primary=rows[:3];base=primary[0];deltas="\n".join(f"{r['role']:<12} MSE {100*(r['MSE']/base['MSE']-1):+.6g}%  MAE {100*(r['MAE']/base['MAE']-1):+.6g}%  B1 {100*(r['B1_median_seconds']/base['B1_median_seconds']-1):+.6g}%  B8 {100*(r['B8_median_seconds']/base['B8_median_seconds']-1):+.6g}%" for r in primary[1:])
    values="\n".join(f"{r['role']:<20} {r['MSE']:.10g} {r['MAE']:.10g} {r.get('accuracy_device','')} / {r['comparison_scope']}" for r in rows)
    training="\n".join(f"{r['role']:<12} {r['selected_refit_epoch']} {r['search_exposures']}/{r['refit_exposures']} {r['search_updates']}/{r['refit_updates']} {r['B1_median_seconds']:.9g}/{r['B1_p90_seconds']:.9g} {r['B8_median_seconds']:.9g}/{r['B8_p90_seconds']:.9g}" for r in primary)
    md=f"""# Beijing K6: B2와 과거 창 지연 encoder의 최소 비교

**[확인] 아래 새3조건은 같은 CPU V43의 실제 전체 추론, 별도 FP64 재점수, freshCPU6 checkpoint replay 및 B1/B8 전체 물리입력 비용 증거가 갖춰진 범위다. 전체 MASTER·충분 학습·독립 일반화 완료를 뜻하지 않는다.**

[원정밀도 CSV](RESULTS_WITH_COST_v1.csv) · [입력·실제증거 SHA](REPORT_INPUT_BINDINGS_v1.json) · [봉인 계획](PLAN.json) · [실행기](beijing_minimum_runner.py)

B2를 기준으로 한 같은 K6/seed92601/CPU의 조건부 차이는 다음과 같다. 음수는 해당 오차 또는 latency가 작다는 뜻이며, 자동 모델 순위나 유의한 일반화 개선을 뜻하지 않는다.

```text
{deltas}
```

```text
role         selected-refit  search/refit-exposures search/refit-updates  B1 median/p90 s  B8 median/p90 s
{training}
```

새3조건은 원 A50[0,10519), A_stop[10519,12622), fresh A60[0,12622), 반복 개발 V[14726,16830)의43 origins·20539 observed labels·C11/L512/H48를 유지했다. 검색0/.25/1/3에서 A_stop으로 epoch를 고른 뒤 새 모델을 그 길이만 refit했다. 최대 search 일정은 같아도 선택된 refit 길이·실제 exposures와 구조별 용량은 다를 수 있으므로 encoder만의 인과 효과나 같은 compute 비교로 확대하지 않는다. 원 channel macro 정규화 MSE를 주지표로 사용하며 MAE 차이도 함께 남긴다.

각 actual fit/replay/cost의 마지막 F tensor hash는 동결 Small의 원 hash와 같고 source48·선택 artifact/scaler·V 네 입력 지문이 연결됐다. 이는 구현·입력·동결 검사 통과이며 성능 개선이나 충분한 수렴의 증거가 아니다. causal encoder와 bidirectional encoder 모두 예측 origin 이전512-step 창만 사용한다. **bidir는 그 과거 창 안에서 뒤쪽 시점을 이용할 수 있으므로 시점별 과거 전용 표현이나 directed causal graph를 검증한 것으로 부르지 않는다.**

```text
role                 MSE          MAE          accuracy device / scope
{values}
```

기존 Raw/G/Ridge/PCA 행은 exact ID의 같은 V/scaler/mask/target/origin/channel일 때만 인용했다. G 초기.25와 extended3-search는 별도 행이며 서로 대체하지 않는다. 원 Raw/G manifest에 장치가 명시되지 않으면 `NOT_RECORDED_IN_MANIFEST`와 reference scope로 남기고 CPU 동등 정확도 비교로 합치지 않았다. G의 training_device=cuda는 원 학습 recipe의 기록이며 그 자체로 전체 V의 실제 추론 장치를 증명하지 않는다. Ridge는 원 NumPy native 경로를 source-traced CPU로 따로 표시한다. 기존 nested PCA K6는 원 CPU V43 FP64/CPU6 및 N3 비용 감사의 재사용 행이며, 고정 basis inverse+center가 B2의 학습된 lag D/current anchor와 다르다. PCA 새 optimizer0회는 PCA 제조 비용0초를 뜻하지 않는다.

새 비용은 CPU2/row128, 각10warm·20timed·별도20intrusive stage이며 physical history/mask→과거 채움/정규화→P/E/F/D→FP64 inverse/CPU output을 매 trial 계산했다. cached Z/forecast는 쓰지 않았다. 같은 B1/B8 물리입력·mask SHA를 세 새조건에서 대사했다. 표의 median/p90은 실제 저장된 ordinary whole20회 요약이며 이 builder가 비용을 다시 측정하거나 통계 gzip을 해석한 것은 아니다. 단계 합이 whole과 같다고 하지 않는다. CPU resident weights·관측된 process-tree RSS·OS peak·GPU peak 및 학습/로드 비용은 별개의 범위다. 기존 Raw/G/Ridge의 정확한 CPU 비용 결속이 이 보고에 없으면 `NOT_INCLUDED`로 남겼으며 다른 장치 비용으로 채우지 않았다.

ROOT의 실제 fit큐 종료와 각 outer postflight/observed-tree return0·remaining[]·quiet 증거를 연결했다. 이 보고는 프로세스를 조회·종료하지 않으며 OS 전체 idleness나 미관측 후손 부재를 증명하지 않는다. 현재 numeric prefix0:16830과 inherited full Task SHA를 구분하고, 원 probe/full-loader의 역사와 반복 개발 V 관찰을 지우지 않는다. 새 protected T 수치/평가·새 모델/NPZ 해석·학습·비용 재실행은 보고 생성 과정에서0이다.

남는 판단은 이 반복 V/한 seed/제한된3epoch 검색에서 지연 encoder의 추가 가치와 비용이 어떻게 바뀌었는가이다. 대조학습·동적 graph 필요성, Bolt 유지의 일반적 우월성, 다른 자료/K/seed와 충분 학습·수렴은 별도 미확정이다. 후순위 B4/ADAPTS/LoRA family4와 기존 중단 shared/max 및 남은 MASTER 항목은 이3조건 완료에 포함하지 않는다.
"""
    return {"RESULTS_WITH_COST_v1.csv":csv_text(rows,FIELDS),"RESULTS_KO_v1.md":md}


def execute(public,cache):
    inputs=Inputs(public,cache);scope=inputs.roots["public"]/NAME
    if any((scope/name).exists() for name in OUTPUTS):raise FileExistsError("Preserve existing output; all report paths must be fresh")
    queue=inputs.json("cache",QUEUE_RESULT)
    if queue.get("status") not in {"COMPLETE_THREE_FITS_AND_FULL_V","PARTIAL_FITS_PRESERVED"} or len(queue.get("conditions",[]))!=3:raise ValueError("Actual original three-fit queue is not terminal; no completed report emitted")
    if {r.get("run_id") for r in queue["conditions"]}!={r for _,r,_ in TARGETS}:raise ValueError("Terminal fit queue contains foreign or duplicate targets")
    post=inputs.json("cache",POST_ROOT+"/RESULT.json")
    if post.get("status")!="COMPLETE_THREE_VALIDATIONS_AND_SIX_COSTS" or len(post.get("steps",[]))!=12 or post.get("skipped")!=[]:raise ValueError("Actual three validation pairs and six standalone costs incomplete")
    inputs.capture("public",NAME+"/build_beijing_report.py");rows,evidence=assemble(inputs,queue,post);texts=compose(rows)
    if not inputs.preserved():raise ValueError("Report input changed; outputs not emitted")
    bindings=dict(schema="N4_BEIJING_REPORT_BINDINGS_v1",status="REPORT_FROM_COMPLETED_THREE_MINIMA",input_SHA_before_after_equal=True,input_count=len(inputs.files),
        inputs=[dict(root=root,relative_path=name,sha256=digest) for (root,name),digest in sorted(inputs.files.items())],
        actual_mode_evidence=evidence,outputs=[dict(relative_path=NAME+"/"+name,sha256=hashlib.sha256(text.encode()).hexdigest()) for name,text in texts.items()],
        new_models_NPZ_raw_T_numeric_training_inference_cost_or_process_operations=0,source48_preserved=True,new_complete_conditions=3,
        baseline_rows_reused=len(rows)-3,cost_summaries_reused_not_retimed=True,whole_MASTER_complete=False,independent_generalization_claimed=False)
    texts["REPORT_INPUT_BINDINGS_v1.json"]=json.dumps(bindings,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    for name,text in texts.items():
        with (scope/name).open("x",encoding="utf-8",newline="\n") as stream:stream.write(text)
    return dict(status="REPORT_CREATED_FROM_ACTUAL_COMPLETE_EVIDENCE",files_written=3,new_conditions=3,reference_rows=len(rows)-3,input_count=len(inputs.files))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--execute",action="store_true");parser.add_argument("--public-root");parser.add_argument("--cache-root");args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="PREPARATION_ONLY_NOT_EXECUTED",input_reads=0,files_written=0,scientific_or_child_operations=0)));return 0
    if not args.public_root or not args.cache_root:parser.error("Explicit --execute --public-root --cache-root required")
    try:result=execute(args.public_root,args.cache_root)
    except (ValueError,FileNotFoundError,FileExistsError,KeyError) as error:
        print(json.dumps(dict(status="NOT_REPORTABLE_EVIDENCE_INCOMPLETE_OR_CHANGED",error_class=type(error).__name__,report_completion_claimed=False)));return 1
    print(json.dumps(result));return 0


if __name__=="__main__":raise SystemExit(main())
