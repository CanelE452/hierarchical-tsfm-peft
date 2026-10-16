"""Two originally registered full-C B2 CPU minima; preparation by default.

ROOT admits a fresh safe-prefix fit only after N4 report/main and a new seal.
No original result/queue is resumed. Actual resource feasibility is unmeasured.
"""
import argparse
import contextlib
import copy
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

NAME="priority_fullC_b2_20261016_v1"
CORE="postT_V_only_cpu_core_remaining_20261016_v1"
REQUIRED="postT_V_only_cpu_required_minima_20261016_v1"
DEADLINE,L,H,SEED=1792222550,512,48,92601
BOUNDS={"jena":dict(A_fit=[0,10482],A_stop=[10482,12579],M_cal=[12579,14675],V_select=[14675,16772],T_final=[16772,20965]),
    "beijing":dict(A_fit=[0,10519],A_stop=[10519,12622],M_cal=[12622,14726],V_select=[14726,16830],T_final=[16830,21038])}
DEPENDENCIES={
    "priority_valid_family_20261016_v1/family_minimum_runner.py":"12462b8f1085e346750a24cd37fd52191f8780799e2f9e5903b4a57bcbc570c1",
    "priority_valid_minima_20261016_v1/beijing_minimum_runner.py":"b84f1473ff4d562890910f7555ef3925823dd921539c8d757118c95504513b9b",
    "priority_b2_validation_20261016_v1/priority_runner.py":"f3e4279ddd250a51b09ce88938f8f202dfd02e3b24d1adff3bb36c567dc7ef07",
    "priority_b2_validation_20261016_v1/safe_jena_prefix.py":"6b496dcbbb6cef48acb9010c630ed81359a2d60bff89aaec9e9e77c4b6d0b541",
    "priority_core_followup_20261016_v1/jena_seed_runner.py":"ad4b76acf51fb416dea4f69c36940507db1f92e223c620388ded45381d6e28e7",
    "priority_nested_cost_20261016_v1/prefix_nested_cost_runner.py":"d64d543499ed21c5a294ace19967c555d05bcfde0cd5ea6c98cbe8ddb23a4ae5"}
REGISTRATION_PINS={
    CORE+"/PLAN.json":"d4c93a9bc00736dc059fda14de39c062e68abab47801434ebc0f1379b779f851",
    CORE+"/REGISTRY.csv":"d5c3ecffb04b0ce89a32a90fde10514f3f8e0fa8363f001617a9427213b8237f",
    CORE+"/core_remaining_runner.py":"6cb1d8ae02fbb2e73dafdf3b18a50b3fe67011f7c9238b5b328f86a4be7d9fc2",
    REQUIRED+"/PLAN.json":"04acc4aabffd02a7266f16fee5c6a3884313fac6ca4b5b46ef0db45eb2fe5152",
    REQUIRED+"/REGISTRY.csv":"fc5f4012a6ced7a1849fcd6474f67d81b600dadaa0014f56df4cda4b8a140f29",
    REQUIRED+"/EXPERIMENT_REGISTRY.csv":"fc5f4012a6ced7a1849fcd6474f67d81b600dadaa0014f56df4cda4b8a140f29",
    REQUIRED+"/required_minima_runner.py":"e11c34918862a698e962cf2a3c151ea6f7484032985f52d25cb7220da6b5d37b",
    "plan/MASTER_PLAN.md":"34d242626f163a8c510d4f6750170bdf9a34af87a610b6aee9df7781fb708569"}
REFERENCES={
    "jena":dict(upstream="primary_jena_E1_B2_k21_s92601",source_sha="764e71f58851d821cb001d5b1935f93a16291701902a7f6b071e85b5c5592267",fit_sha="1ff605e9fed9abd33a979e34c355c030d3f338f1d8358f57eb911f1502959474",V_sha="ccbba525acab00d51fd3914a15b2515454585983b6d33ed158017e6f148586bc"),
    "beijing":dict(upstream="primary_beijing_E1_B2_k11_s92601",source_sha="83c03d9ce63915d5d3b1971e2d71efa657a5578fbc1ac1928148b31b67996b5d",fit_sha="bf48f3639d081739d6a9d2f91edce591f81e7896e38f1fb71f1e05f7aa1ea70c",V_sha="8090e36b853122381db8503f1bb948577712e10e047df0c3f72f02a080f429a3")}
OLD_PROBE=dict(relative_path=REQUIRED+"/diagnostics/probe_root_v1_04/RESULT.json",sha256="3c5e01417db0320151d803439d065e34e065f01b41b560ecaeafa7e01125d352")


def conditions():
    result=[]
    for dataset,width,suffix in (("jena",21,"remaining_v1"),("beijing",11,"required_v1")):
        identifier=f"postT_cpu_{dataset}_E1_B2_k{width}_s92601_{suffix}";reference=REFERENCES[dataset]
        provenance=dict(upstream_registered_id=reference["upstream"])
        if dataset=="jena":provenance.update(old_weights_optimizer_RNG_or_state_reused=False,training_microbatch_changed=False,bitwise_training_equivalence_claimed=False)
        spec=dict(run_id=identifier,dataset=dataset,family="B2",latent_k=width,seed=SEED,fold="primary",epochs=[.25,1.,3.],lr=.001,device="cpu",microbatch=4,effective_batch=16,
            F_row_microbatch=128,checkpoint_inputs=False,lora_rank=8,lora_alpha=16,stage_label="postT_cpu_fresh_V_only_"+("remaining_core29" if dataset=="jena" else "required_minima"),provenance=provenance)
        result.append(dict(run_id=identifier,dataset=dataset,family="B2",latent_k=width,effective_F_width=width,upstream_registered_id=reference["upstream"],spec=spec,
            original_namespace=CORE if dataset=="jena" else REQUIRED,V_reference=dict(relative_path="predictions/"+reference["upstream"]+"/V_select/PREDICTION_MANIFEST.json",sha256=reference["V_sha"]),
            historical_CPU_probe=None if dataset=="jena" else copy.deepcopy(OLD_PROBE)))
    return result


def draft_plan():
    return dict(schema="PRIORITY_FULLC_B2_PREFIX_v1",namespace=NAME,status="DRAFT_NOT_EXECUTABLE",runner_sha256=None,ROOT_after_N4_report_and_main_admitted=False,
        conditions=conditions(),maximum_new_training_conditions=2,dependencies=copy.deepcopy(DEPENDENCIES),registration_pins=copy.deepcopy(REGISTRATION_PINS),GPU_quarter_references=copy.deepcopy(REFERENCES),
        old_Beijing_discarded_probe=copy.deepcopy(OLD_PROBE),original_primary_split_ranges=copy.deepcopy(BOUNDS),materialized_prefix_rows=dict(jena=16772,beijing=16830),
        protected_tail_numeric_parse_allowed=False,new_T_target_operations_allowed=False,deadline_unix=DEADLINE,
        optimizer=dict(name="AdamW",lr=.001,weight_decay=0.,gradient_clip=1.,effective_batch=16,microbatch=4),
        resources=dict(CPU_threads=2,max_worker_RSS_bytes=3*1024**3,maximum_model_workers=2,total_owned_RSS_limit_bytes=6*1024**3,
            storage_limit_bytes=30*1024**3,admission_reserve_bytes=256*1024**2,GPU_process_seconds=0,costs_must_not_overlap_training=True,ROOT_CPU_supervisor_required=True),
        costs=dict(batches=[1,8],warmups=10,timed_trials=20,stage_trials=20,quiet_max_age_seconds=120,prior_actual_rescore_replay_outer_proof_SHA_required=True),
        stage="POST_T_V_ONLY_DEVELOPMENT",scope="Existing mandatory K=C B2 minima, not the unstarted K9/16/241/128 full-grid eight and not new first-Jena4 slots.",
        initialization="Original create_model identity_at_K_equals_C, center0 and no k_grid; trained E may subsequently change. Not nested fixed PCA full-C rotation.",
        original_GPU_screen_reused_as_new_CPU_fit=False,exact_continuation_or_bitwise_equivalence_claimed=False,probe_mode_implemented=False,
        CPU_probe_scope=dict(jena="NO_EXISTING_FULLC_CPU_PROBE",beijing="OLD_DISCARDED_THREE_UPDATES_NOT_FIT"),
        resource_feasibility="Actual full-C CPU fit/forward peak and time unmeasured; ROOT admission and guard still required.",
        interruption="Original stage optimizer-boundary/deadline checkpoints only; no signal autosave or implicit partial resume.",
        actual_Task_model_probe_fit_forward_rescore_cost_or_dispatch_operations=0,
        ROOT_seal_required="After N4 report and main, reassess original purpose/time/resources; preserve this draft and explicitly seal new PLAN with runner SHA and positive ROOT admission. No automatic execution.")


def validate_plan(plan):
    if (plan.get("status")!="SEALED_BY_ROOT" or plan.get("ROOT_after_N4_report_and_main_admitted") is not True
            or not isinstance(plan.get("runner_sha256"),str) or len(plan["runner_sha256"])!=64 or any(char not in "0123456789abcdef" for char in plan["runner_sha256"])):
        raise ValueError("Positive post-N4 ROOT admission and new sealed PLAN/current runner SHA required")
    for key,value in draft_plan().items():
        if key not in {"status","runner_sha256","ROOT_after_N4_report_and_main_admitted","actual_Task_model_probe_fit_forward_rescore_cost_or_dispatch_operations"} and plan.get(key)!=value:raise ValueError("Exact two original full-C recipes/prefix/resource scope required: "+key)


def choose(plan,identifier):
    found=[row for row in plan["conditions"] if row["run_id"]==identifier]
    if len(found)!=1:raise ValueError("One exact originally registered full-C CPU ID required")
    return found[0]


def require_unstarted(public,cache,identifier):
    for root in (Path(public),Path(cache),Path(public)/CORE,Path(cache)/CORE,Path(public)/REQUIRED,Path(cache)/REQUIRED,Path(public)/NAME,Path(cache)/NAME):
        for kind in ("runs","predictions"):
            if (root/kind/identifier).exists():raise FileExistsError("Original or new full/partial result preserved; no implicit resume")
        if (root/"claims"/("fit_"+identifier+".json")).exists():raise FileExistsError("Existing fit claim preserved")


def validate_initialization(normalization,channels,np):
    if (normalization.get("basis_initialization")!="identity_at_K_equals_C" or not np.array_equal(normalization["basis"],np.eye(channels,dtype=np.float32))
            or not np.array_equal(normalization["pca_center"],np.zeros(channels,dtype=np.float32))):raise ValueError("Original full-C identity basis/zero center, not PCA rotation, required")


def metadata(public,cache,asset,plan_sha,identifier,process_api):
    path=relative(public,NAME+"/PLAN.json")
    if not plan_sha or sha(path)!=plan_sha:raise ValueError("Exact new ROOT-sealed PLAN SHA required")
    plan=read(path);validate_plan(plan);condition=choose(plan,identifier)
    if sha(relative(public,NAME+"/fullC_b2_runner.py"))!=plan["runner_sha256"]:raise ValueError("New adapter source changed")
    check_pins(public,DEPENDENCIES);check_pins(public,REGISTRATION_PINS)
    libraries={name:module(relative(public,relative_path),"fullC_"+name) for name,relative_path in (
        ("common","priority_valid_family_20261016_v1/family_minimum_runner.py"),("modelcheck","priority_valid_minima_20261016_v1/beijing_minimum_runner.py"),("priority","priority_b2_validation_20261016_v1/priority_runner.py"),
        ("primitive","priority_b2_validation_20261016_v1/safe_jena_prefix.py"),("seed","priority_core_followup_20261016_v1/jena_seed_runner.py"),("nested","priority_nested_cost_20261016_v1/prefix_nested_cost_runner.py"))}
    p1=libraries["priority"];p1.metadata(public,cache,"475f983058871ff36b4112b6aa56dc252b17aea47eb6ae1744ff119c2ad5138f",process_api)
    core,required=read(relative(public,CORE+"/PLAN.json")),read(relative(public,REQUIRED+"/PLAN.json"))
    if (len(core["source39"])!=39 or len(core["immutable_public"])!=9 or core["source39"]!=required["source39"] or core["immutable_public"]!=required["immutable_public"]):raise ValueError("Original48 immutable/source identities differ")
    check_pins(public,{**core["source39"],**core["immutable_public"]})
    keys=("model_asset_relative_path","model_revision","foundation_tensor_sha256","instance_norm_source_relative_path","instance_norm_source_sha256")
    if any(core[key]!=required[key] for key in keys):raise ValueError("Original model/normalization identities differ")
    old={key:core[key] for key in keys};old["task_contracts"]=dict(jena=core["task_contracts"]["jena"],beijing=required["task_contract"])
    for current in conditions():
        parent=core if current["dataset"]=="jena" else required
        matches=[row for row in parent["conditions"] if row["run_id"]==current["run_id"]]
        if len(matches)!=1 or any(matches[0][key]!=current[key] for key in ("run_id","family","latent_k","spec","upstream_registered_id")):raise ValueError("Original registered CPU condition changed")
        with relative(public,current["original_namespace"]+"/REGISTRY.csv").open(encoding="utf-8-sig",newline="") as stream:rows=list(csv.DictReader(stream))
        row=[row for row in rows if row["experiment_id"]==current["run_id"]]
        if len(row)!=1 or any(row[0][key]!=value for key,value in dict(dataset=current["dataset"],family="B2",latent_k=str(current["latent_k"]),seed="92601",device="cpu",length="512",horizon="48",epochs="0.25|1|3").items()):raise ValueError("Original exact CPU registry row differs")
        reference=REFERENCES[current["dataset"]];upstream=reference["upstream"]
        for name,digest in (("runs/"+upstream+"/SOURCE_CONTRACT.json",reference["source_sha"]),("runs/"+upstream+"/BASE_FIT_RECEIPT.json",reference["fit_sha"]),(current["V_reference"]["relative_path"],reference["V_sha"])):
            if sha(relative(public,name))!=digest:raise ValueError("Original GPU-training quarter-screen/V reference metadata changed")
    if sha(relative(public,OLD_PROBE["relative_path"]))!=OLD_PROBE["sha256"]:raise ValueError("Historical discarded Beijing full-C probe changed")
    probe=read(relative(public,OLD_PROBE["relative_path"]))
    if probe.get("run_id")!=conditions()[1]["run_id"] or probe.get("status")!="PROBE_EFFECTIVE_UPDATES_COMPLETE_DISCARDED" or probe["scientific_result"].get("completed_fit") is not False:raise ValueError("Old discarded probe is not a completed fit")
    for dataset,contract in old["task_contracts"].items():
        meta=contract["metadata"];source=relative(asset,meta["source_reference"]);contract_name=libraries["primitive"].CONTRACT_REL if dataset=="jena" else str(Path(meta["source_reference"]).with_name("contract.json"))
        if source.stat().st_size!=meta["source_bytes"] or sha(source)!=meta["source_sha256"] or sha(relative(asset,contract_name))!=meta["source_contract_sha256"]:raise ValueError("Original opaque raw asset/contract changed")
    libraries["modelcheck"].check_model_snapshot(asset,old["model_asset_relative_path"],libraries["common"].MODEL_BINDINGS)
    if (sha(relative(asset,old["instance_norm_source_relative_path"]))!=old["instance_norm_source_sha256"]
            or sha(relative(public.parent,"temporal_frontend_campaign_20261015_v1/runtime.py"))!=old["instance_norm_source_sha256"]):raise ValueError("Original norm implementation changed")
    return plan,old,condition,libraries


def safe_task(asset,old,condition,libraries,context):
    original=old["task_contracts"][condition["dataset"]];common=libraries["common"];np,data,fit=context["np"],context["data"],context["fit"]
    if condition["dataset"]=="jena":task,access=libraries["primitive"].load_jena_prefix(asset,original)
    else:task,access=libraries["nested"].safe_task(asset,dict(task_contract=original),libraries["primitive"],np)
    common.validate_prefix_shape(task,original,condition["dataset"])
    if access["new_T_numeric_values_parsed"]!=0:raise ValueError("Protected T numerical decoding forbidden")
    bounds=BOUNDS[condition["dataset"]];mean,std,count=data.fit_statistics(task,bounds["A_fit"][1]);a50={key:data.array_sha256(value) for key,value in (("mean",mean),("std",std),("count",count))};a50scale=fit.scaler_fingerprint(mean,std)
    mean,std,_=data.fit_statistics(task,bounds["A_stop"][1]);common.validate_statistics(original,a50,a50scale,fit.scaler_fingerprint(mean,std))
    arrays=context["priority"].cohort_arrays(task,bounds,mean,std,data,np);context["priority"].validate_cohort(arrays,context["reference"],context["evaluation"])
    access.update(actual_A50_A60_and_original_V43_fingerprints_matched=True,old_full_task_identity_inherited_not_recomputed=True,historical_reference_device_scope="Original training cuda; original fullV inference device not recorded")
    return task,access


def fit_and_save(condition,task,foundation,context,public_run,private_run,deadline):
    fit,p1,common,np=context["fit"],context["priority"],context["common"],context["np"];spec=condition["spec"];bounds=BOUNDS[condition["dataset"]];initializations=[]
    for end in (bounds["A_fit"][1],bounds["A_stop"][1]):
        model,norm=fit.create_model(spec,task,end,foundation)
        try:
            validate_initialization(norm,condition["latent_k"],np)
            initializations.append(dict(fit_end=end,basis_initialization=norm["basis_initialization"],basis_sha256=context["data"].array_sha256(norm["basis"]),center_sha256=context["data"].array_sha256(norm["pca_center"])))
        finally:
            if hasattr(model,"close"):model.close()
            del model
    source,files=fit._source_contract();contract=dict(source_sha=source,config_sha=fingerprint(dict(spec=spec,L=L,H=H,C=len(task["columns"]),columns=task["columns"],inherited_array_sha256=task["array_sha256"],materialized_prefix_sha256=task["materialized_prefix_sha256"])),data_sha=task["data_sha256"],model_sha=fit._model_sha(foundation))
    new(public_run/"SOURCE_CONTRACT.json",dict(**contract,files=files,spec=spec,original_task_identity="INHERITED_NOT_RECOMPUTED",actual_prefix_sha256=task["materialized_prefix_sha256"],original_registered_CPU_namespace=condition["original_namespace"],fresh_fit_not_GPU_continuation=True,initialization_checks=initializations))
    stages=common.fit_stages(fit,condition,task,foundation,contract,public_run,private_run,deadline)
    if stages["status"]!="COMPLETE_STAGES":return stages
    search,refit,epoch=stages["search"],stages["refit"],stages["selected_epoch"];common.validate_stages(condition,search["summary"],refit["summary"])
    for label,stage in (("search",search),("refit",refit)):p1.write_stage(public_run/label,stage)
    bundle=dict(model=refit["model"],normalization=refit["normalization"],spec=spec,contract=refit["summary"]["contract"],selected_search_epoch=epoch,
        search_summary=search["summary"],refit_summary=refit["summary"],model_state_sha=p1.state_sha(refit["model"],context["data"]),plan_sha256=context["plan_sha"],actual_prefix_sha256=task["materialized_prefix_sha256"])
    artifact=private_run/"selected_bundle.pt";p1.save_bundle(artifact,bundle,context["torch"])
    receipt=dict(status="COMPLETE_VALID",run_id=spec["run_id"],family="B2",seed=SEED,epochs=[.25,1.,3.],selected_epoch=epoch,
        training_status="UNDERTRAINED_OR_BUDGET_LIMIT" if epoch==3 else "SCREEN_COMPLETE_CONVERGENCE_UNRESOLVED",optimizer="AdamW",weight_decay=0.,
        artifact_sha256=sha(artifact),model_state_sha=bundle["model_state_sha"],refit_scaler_sha=fit.scaler_fingerprint(bundle["normalization"]["mean"],bundle["normalization"]["std"]),
        artifact_cache_relative_path=NAME+"/runs/"+spec["run_id"]+"/selected_bundle.pt",search_fit_end=bounds["A_fit"][1],refit_fit_end=bounds["A_stop"][1],
        search_updates=search["summary"]["updates"],refit_updates=refit["summary"]["updates"],search_exposures=search["summary"]["exposures"],refit_exposures=refit["summary"]["exposures"],
        source_sha=source,source_stage_API="original_fit_base._stage",original_registered_CPU_namespace=condition["original_namespace"],fresh_fit_not_GPU_continuation=True,
        initializations=initializations,initial_identity_does_not_freeze_learned_E=True,old_weights_optimizer_or_sampler_reused=False,reused_complete_artifact=False,
        original_full_task_identity="INHERITED_NOT_RECOMPUTED",actual_prefix_sha256=task["materialized_prefix_sha256"],post_T_development=True,protected_T_target_operations=0)
    new(public_run/"BASE_FIT_RECEIPT.json",receipt);return dict(status="COMPLETE_VALID",receipt=receipt,artifact=artifact)


def saved_arrays(public_folder,private_folder,condition,receipt,context):
    return context["common"].saved_arrays(public_folder,private_folder,condition,receipt,context)


def rescore(public_folder,private_folder,condition,receipt,context,output):
    return context["common"].rescore(public_folder,private_folder,condition,receipt,context,output)


def validate_saved_manifest(manifest,condition,receipt,context):
    expected=dict(run_id=condition["run_id"],family="B2",seed=SEED,split="V_select",cohort=condition["dataset"]+":primary:V_select:stride48",device="cpu",
        data_sha=context["original"]["data_sha256"],artifact_sha256=receipt["artifact_sha256"],scaler_sha=receipt["refit_scaler_sha"],source_sha=fingerprint(context["new_sources"]),
        config_sha=fingerprint(condition["spec"]),model_sha=fingerprint(dict(artifact_sha=receipt["artifact_sha256"],model_state_sha=receipt["model_state_sha"])))
    if any(manifest.get(key)!=value for key,value in expected.items()):raise ValueError("Exact new CPU V43 selected-checkpoint manifest required")
    for key in ("target_normalized","observed","origins","channel_ids"):
        if manifest["arrays"].get(key)!=context["reference"]["arrays"][key]:raise ValueError("Original exact V cohort fingerprint differs")
    prediction=manifest["arrays"]["pred_normalized"];width=condition["latent_k"]
    if prediction.get("shape")!=[43,H,width] or prediction.get("dtype")!="float32":raise ValueError("Exact safe fullV prediction geometry required")
    score=manifest["score"]
    if any(score.get(key)!=value for key,value in dict(n_origins=43,n_observed=43344 if condition["dataset"]=="jena" else 20539,num_channels=width,horizon=48).items()):raise ValueError("Original full V43 score geometry differs")


def sha(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):value.update(block)
    return value.hexdigest()


def read(path):
    if Path(path).stat().st_size > 8*1024**2:raise ValueError("Small JSON metadata required")
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")


def relative(root,name):
    root=Path(root).absolute();path=root/name
    if Path(name).is_absolute() or ".." in Path(name).parts or any(p.is_symlink() or getattr(p,"is_junction",lambda:False)() for p in (path,*path.parents)):
        raise ValueError("Explicit nonsymlink root-relative path required")
    root,path=root.resolve(),path.resolve()
    if not path.is_relative_to(root):raise ValueError("Path escapes declared root")
    return path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def check_pins(root,pins):
    for path,digest in pins.items():
        if sha(relative(root,path)) != digest:raise ValueError("Pinned input/source changed: "+path)


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def require_quiet(proof,now):
    stamp=proof.get("captured_unix")
    if (proof.get("status")!="QUIET_CPU_COST_WINDOW" or proof.get("active_model_workers")!=[] or proof.get("active_cost_workers")!=[]
            or isinstance(stamp,bool) or not isinstance(stamp,(int,float)) or not math.isfinite(stamp) or not 0<=now-stamp<=120):
        raise ValueError("Fresh120s ROOT quiet proof with both active lists empty required")


def validate_origins(origins,dataset):
    lo,hi=BOUNDS[dataset]["V_select"]
    if not len(origins) or any(int(o)<lo or int(o)+H>hi for o in origins):raise ValueError("Origins must stay in exact original V without T")


def expected_counters(width,batches):
    if not isinstance(width,int) or width<1 or not batches or any(not isinstance(b,int) or b<1 for b in batches):raise ValueError("Positive stream width/batches required")
    streams=sum(b*width for b in batches)
    return dict(streams=streams,physical_streams=streams,physical_calls=sum(math.ceil(b*width/128) for b in batches))


def canonical_route(condition,receipt):
    return dict(run_id=condition["run_id"],dataset=condition["dataset"],family=condition["family"],fold="primary",
        latent_k=condition["effective_F_width"],canonical_fit_end=BOUNDS[condition["dataset"]]["A_stop"][1],
        artifact_cache_relative_path=NAME+"/runs/"+condition["run_id"]+"/selected_bundle.pt",artifact_sha256=receipt["artifact_sha256"],row_microbatch=128)


@contextlib.contextmanager
def route_lifetime(route,foundation):
    try:
        yield route
    finally:
        route.close()
        foundation.assert_frozen_unchanged()


def completed(public_run,private_run,condition,original):
    receipt=read(public_run/"BASE_FIT_RECEIPT.json");source=read(public_run/"SOURCE_CONTRACT.json");artifact=private_run/"selected_bundle.pt"
    if (receipt.get("status")!="COMPLETE_VALID" or receipt.get("run_id")!=condition["run_id"] or receipt.get("family")!=condition["family"] or receipt.get("seed")!=SEED
            or receipt.get("epochs")!=[.25,1.,3.] or receipt.get("selected_epoch") not in (0.,.25,1.,3.) or receipt.get("refit_scaler_sha")!=original["refit_scaler_sha256"]
            or source["spec"]!=condition["spec"] or sha(artifact)!=receipt["artifact_sha256"]):raise ValueError("Exact completed new-entry artifact/recipe required")
    return dict(receipt=receipt,artifact=artifact)


def actual_route(fitted,condition,task,pool,context):
    receipt=fitted["receipt"]
    route=context["deployment"].Route(canonical_route(condition,receipt),task,context["cache"],device="cpu",foundation_pool=pool)
    try:
        if route.family!=condition["family"] or route.latent_k!=condition["effective_F_width"]:raise ValueError("Actual bundle family/F width differs from admitted condition")
        bundle=route.bundle
        if (bundle["spec"]!=condition["spec"] or bundle["model_state_sha"]!=receipt["model_state_sha"] or bundle["plan_sha256"]!=context["plan_sha"]
                or bundle["contract"]["data_sha"]!=context["original"]["data_sha256"] or bundle["contract"]["model_sha"]!=context["foundation_sha"]
                or bundle["normalization"]["fit_end"]!=BOUNDS[condition["dataset"]]["A_stop"][1]
                or context["fit"].scaler_fingerprint(route.mean,route.std)!=receipt["refit_scaler_sha"]):raise ValueError("Actual selected bundle state/recipe/plan/A60 differs from public accuracy identity")
        return route
    except BaseException:
        route.close();pool.foundation.assert_frozen_unchanged();raise


def full_prediction(fitted,condition,task,pool,context,public_folder,private_folder,deadline):
    np,p1=context["np"],context["priority"];bounds=BOUNDS[condition["dataset"]];width=len(task["columns"])
    route=actual_route(fitted,condition,task,pool,context);predictions=[]
    with route_lifetime(route,pool.foundation):
        arrays=p1.cohort_arrays(task,bounds,route.mean,route.std,context["data"],np);p1.validate_cohort(arrays,context["reference"],context["evaluation"]);validate_origins(arrays["origins"],condition["dataset"])
        before=pool.counters()
        for start in range(0,43,4):
            p1.check_deadline(deadline);origins=arrays["origins"][start:start+4]
            physical=np.stack([task["physical"][int(o)-L:int(o)] for o in origins]);observed=np.stack([task["observed"][int(o)-L:int(o)] for o in origins])
            raw=route.predict_physical(physical,observed);predictions.append(((raw-route.mean.astype(np.float64))/route.std.astype(np.float64)).astype(np.float32))
        arrays["pred_normalized"]=np.concatenate(predictions);delta={key:pool.counters()[key]-before[key] for key in ("streams","physical_streams","physical_calls")}
        if arrays["pred_normalized"].shape!=(43,H,width) or delta!=expected_counters(condition["effective_F_width"],[4]*10+[3]):raise ValueError("Full V43 actual output/counters differ")
    public_folder.mkdir(parents=True,exist_ok=False);private_folder.mkdir(parents=True,exist_ok=False)
    receipt=fitted["receipt"];identity=dict(run_id=condition["run_id"],split="V_select",cohort=condition["dataset"]+":primary:V_select:stride48",data_sha=task["data_sha256"],scaler_sha=receipt["refit_scaler_sha"],source_sha=fingerprint(context["new_sources"]),config_sha=fingerprint(condition["spec"]),model_sha=fingerprint(dict(artifact_sha=receipt["artifact_sha256"],model_state_sha=receipt["model_state_sha"])))
    new(public_folder/"REFERENCE_CONTRACT.json",dict(**identity,arrays={key:context["evaluation"].array_fingerprint(value) for key,value in arrays.items() if key!="pred_normalized"}))
    archive=private_folder/"V_select.npz"
    with archive.open("xb") as stream:np.savez_compressed(stream,**arrays)
    score,rows=context["evaluation"].error_statistics(arrays["pred_normalized"],arrays["target_normalized"],arrays["observed"],arrays["origins"],condition["run_id"],"V_select",identity["cohort"],arrays["channel_ids"])
    with gzip.open(public_folder/"PUBLIC_ERROR_STATS.csv.gz","xt",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    p1.csv_new(public_folder/"SCORES.csv",[score]);manifest=dict(**identity,archive_sha256=sha(archive),arrays={key:context["evaluation"].array_fingerprint(value) for key,value in arrays.items()},score=score,family=condition["family"],seed=SEED,artifact_sha256=receipt["artifact_sha256"],device="cpu",F_counters=delta,post_T_development=True,new_T_target_operations=0,CPU_threads=2,route="original_deployment.Route_full_constructor")
    new(public_folder/"PREDICTION_MANIFEST.json",manifest)
    return manifest


def replay(fitted,condition,arrays,task,pool,context,output,deadline):
    np,p1=context["np"],context["priority"];indices=p1.replay_indices(43,np);origins=arrays["origins"][indices];route=actual_route(fitted,condition,task,pool,context);predictions=[]
    with route_lifetime(route,pool.foundation):
        before=pool.counters()
        for start in (0,4):
            p1.check_deadline(deadline);oo=origins[start:start+4];physical=np.stack([task["physical"][int(o)-L:int(o)] for o in oo]);observed=np.stack([task["observed"][int(o)-L:int(o)] for o in oo])
            raw=route.predict_physical(physical,observed);predictions.append(((raw-route.mean.astype(np.float64))/route.std.astype(np.float64)).astype(np.float32))
        actual=np.concatenate(predictions);delta={key:pool.counters()[key]-before[key] for key in ("streams","physical_streams","physical_calls")}
        if delta!=expected_counters(condition["effective_F_width"],[4,2]):raise ValueError("CPU6 original stream width/4+2 call accounting differs")
        passed=bool(np.allclose(actual,arrays["pred_normalized"][indices],atol=1e-5,rtol=1e-4))
        result=dict(status="PASS_FRESH_CPU6_REPLAY" if passed else "FAIL_REPLAY_PARITY",executed_now=True,run_id=condition["run_id"],artifact_sha256=fitted["receipt"]["artifact_sha256"],origins=origins.tolist(),microbatches=[4,2],F_counters=delta,base_frozen_after_original_Route_close=True,atol=1e-5,rtol=1e-4,max_abs=float(np.abs(actual.astype(np.float64)-arrays["pred_normalized"][indices].astype(np.float64)).max()),new_T_target_operations=0)
    new(output/"REPLAY_RECEIPT.json",result)
    if not passed:raise ValueError("Actual fresh CPU6 replay parity failed")
    return result


def verify_cost_prerequisites(condition,receipt,manifest_sha,proofs):
    for mode,status in (("rescore","PASS_INDEPENDENT_FULL_V_FP64"),("replay","PASS_FRESH_CPU6_REPLAY")):
        path,digest=proofs.get(mode,(None,None))
        if not path or not digest or sha(path)!=digest:raise ValueError("Exact actual validation RESULT/SHA required")
        value=read(path);science=value.get("scientific_result") or {}
        if (value.get("run_id")!=condition["run_id"] or value.get("mode")!=mode or value.get("error_class") is not None or value.get("postflight_error_class") is not None
                or value.get("status")!=status or science.get("status")!=status or science.get("artifact_sha256")!=receipt["artifact_sha256"] or science.get("manifest_sha256")!=manifest_sha):raise ValueError("Same selected artifact's successful actual rescore/replay required")
        if mode=="replay" and science.get("executed_now") is not True:raise ValueError("Cached replay cannot replace fresh CPU checkpoint forward")


def cost(fitted,condition,task,pool,context,manifest_sha,output,batch,deadline,quiet_path,quiet_sha,proofs):
    if batch not in (1,8) or not quiet_path or not quiet_sha or sha(quiet_path)!=quiet_sha:raise ValueError("B1/B8 with exact fresh ROOT quiet proof required")
    require_quiet(read(quiet_path),time.time());verify_cost_prerequisites(condition,fitted["receipt"],manifest_sha,proofs)
    import measure_costs
    np,p1=context["np"],context["priority"];origins=context["data"].origins(BOUNDS[condition["dataset"]]["V_select"],L,H,H)[:batch];validate_origins(origins,condition["dataset"])
    physical=np.stack([task["physical"][int(o)-L:int(o)] for o in origins]);observed=np.stack([task["observed"][int(o)-L:int(o)] for o in origins]);route=actual_route(fitted,condition,task,pool,context)
    class Check:
        def check(self):p1.check_deadline(deadline)
    with route_lifetime(route,pool.foundation):
        before=pool.counters();rows=measure_costs.measure_balanced([route],physical,observed,warmups=10,timed_trials=20,stage_trials=20,lease=Check());want=expected_counters(condition["effective_F_width"],[batch])
        if len(rows)!=50 or any((row["F_logical_streams"],row["F_physical_streams"],row["F_calls"])!=(want["streams"],want["physical_streams"],want["physical_calls"]) for row in rows):raise ValueError("Original50trial per-family stream/call protocol differs")
        delta={key:pool.counters()[key]-before[key] for key in ("streams","physical_streams","physical_calls")}
        if delta!={key:value*50 for key,value in want.items()}:raise ValueError("Actual total cost counters differ")
        memory=route.memory_state();effective_state=context["gpu_runtime"].state_hash(pool.foundation.model)
        if sha(quiet_path)!=quiet_sha or any(sha(path)!=digest for path,digest in proofs.values()):raise ValueError("Cost proof changed during measurement")
    trials=output/"COST_TRIALS.csv.gz";measure_costs._write_trials(trials,rows)
    result=dict(status="COMPLETE_PROTOCOL",run_id=condition["run_id"],device="cpu",CPU_threads=2,batch_size=batch,warmups=10,timed_trials=20,stage_trials=20,actual_trial_rows=50,
        summaries=measure_costs.summarize(rows),memory=memory,trials_sha256=sha(trials),accuracy_manifest_sha256=manifest_sha,artifact_sha256=fitted["receipt"]["artifact_sha256"],quiet_proof_sha256=quiet_sha,
        model_load_seconds=route.load_seconds,model_loading_in_whole_trial_timer=False,
        validation_result_sha256={key:value[1] for key,value in proofs.items()},physical_input_sha256=hashlib.sha256(physical.tobytes()).hexdigest(),observation_mask_sha256=hashlib.sha256(observed.tobytes()).hexdigest(),origins=origins.tolist(),F_counters=delta,
        F_logical_streams_per_trial=want["streams"],F_calls_per_trial=want["physical_calls"],cached_Z_or_forecast_reads=0,physical_output_and_inverse_dtype="float64",stage_probe_separate_intrusive_scope=True,training_peak_included=False,
        frozen_F_base_sha256=context["foundation_sha"],effective_F_state_sha256_during_measurement=effective_state,effective_F_state_scope="original frozen F",
        base_restored_after_original_Route_close=True,new_T_target_operations=0)
    new(output/"COST_RECEIPT.json",result);return result


def execute(public,cache,asset,plan_sha,identifier,mode,deadline,output,batch=None,quiet_path=None,quiet_sha=None,proofs=None):
    import psutil
    public,cache,asset=map(lambda p:relative(p,"."),(public,cache,asset));plan,old,condition,libraries=metadata(public,cache,asset,plan_sha,identifier,psutil)
    if mode not in ("fit","rescore","replay","cost") or not isinstance(deadline,(int,float)) or not time.time()<deadline<=DEADLINE:raise ValueError("Explicit mode and live original deadline required")
    if (mode=="cost" and batch not in (1,8)) or (mode!="cost" and batch is not None):raise ValueError("Only cost takes B1/B8")
    p1=libraries["priority"];p1.require_environment();scope,private=relative(public,NAME),relative(cache,NAME)
    if not private.is_dir():raise FileNotFoundError("ROOT must prepare new private namespace")
    output=Path(output).absolute();output=relative(scope,os.path.relpath(output,scope))
    if output.exists() or not output.is_relative_to(scope/"diagnostics"):raise FileExistsError("New exclusive diagnostic output required")
    if mode=="fit":require_unstarted(public,cache,identifier)
    if mode=="cost":
        quiet_path=relative(cache,os.path.relpath(Path(quiet_path).absolute(),cache)) if quiet_path else None
        proofs={key:(relative(scope,os.path.relpath(Path(path).absolute(),scope)),digest) for key,(path,digest) in (proofs or {}).items() if path and digest}
        if not quiet_path or not quiet_sha or sha(quiet_path)!=quiet_sha:raise ValueError("Exact private fresh quiet proof required")
        require_quiet(read(quiet_path),time.time())
    libraries["seed"].resource_preflight(public,cache,plan,psutil);output.mkdir(parents=True,exist_ok=False);started=time.time()
    new(output/"STARTED.json",dict(run_id=identifier,mode=mode,plan_sha256=plan_sha,actual_pid=os.getpid(),actual_create_time=psutil.Process().create_time(),deadline_unix=deadline))
    public_run,private_run=scope/"runs"/identifier,private/"runs"/identifier;public_prediction,private_prediction=scope/"predictions"/identifier/"V_select",private/"predictions"/identifier
    result=context=pool=None;failure=post_error=None;post_path=None
    try:
        sys.path.insert(0,str(public))
        import numpy as np
        import data_contract as data
        import evaluation
        if Path(data.__file__).resolve()!=public/"data_contract.py":raise ValueError("Original import root differs")
        context=dict(np=np,data=data,evaluation=evaluation,priority=p1,common=libraries["common"],cache=cache,plan_sha=plan_sha,new_sources=dict(runner=plan["runner_sha256"],**DEPENDENCIES),
            original=old["task_contracts"][condition["dataset"]],reference=read(relative(public,condition["V_reference"]["relative_path"])),foundation_sha=old["foundation_tensor_sha256"])
        if mode=="rescore":
            fitted=completed(public_run,private_run,condition,context["original"]);result=rescore(public_prediction,private_prediction,condition,fitted["receipt"],context,output)
        else:
            import torch
            import fit_base as fit
            import deployment
            import gpu_runtime
            torch.set_num_threads(2)
            if torch.cuda.is_initialized():raise RuntimeError("CPU route initialized CUDA")
            context.update(torch=torch,fit=fit,deployment=deployment,gpu_runtime=gpu_runtime)
            task,access=safe_task(asset,old,condition,libraries,context);new(output/"PREFIX_ACCESS_RECEIPT.json",access)
            pool=deployment.FoundationPool(relative(asset,old["model_asset_relative_path"]),"cpu");foundation=pool.get(H)
            if (gpu_runtime.state_hash(foundation.model)!=old["foundation_tensor_sha256"] or foundation.row_microbatch!=128 or foundation.checkpoint_inputs is not False
                    or foundation.instance_norm_source_sha256!=old["instance_norm_source_sha256"] or any(p.requires_grad or p.grad is not None for p in foundation.model.parameters())):raise ValueError("Actual frozen Small/CPU/row128/normalization identity differs")
            if mode=="fit":
                new(scope/"claims"/("fit_"+identifier+".json"),dict(run_id=identifier,plan_sha256=plan_sha,created_unix=time.time(),prefix_only_new_entry=True))
                public_run.mkdir(parents=True,exist_ok=False);private_run.mkdir(parents=True,exist_ok=False);fitted=fit_and_save(condition,task,foundation,context,public_run,private_run,deadline)
                if "receipt" not in fitted:result=fitted
                else:
                    manifest=full_prediction(fitted,condition,task,pool,context,public_prediction,private_prediction,deadline)
                    result=dict(status="COMPLETE_FIT_AND_FULL_V",fit_receipt=fitted["receipt"],V_score=manifest["score"],full_V_completed=True,independent_rescore_pending=True,checkpoint_replay_pending=True)
            else:
                fitted=completed(public_run,private_run,condition,context["original"])
                if mode=="replay":arrays,manifest=saved_arrays(public_prediction,private_prediction,condition,fitted["receipt"],context)
                else:manifest=read(public_prediction/"PREDICTION_MANIFEST.json")
                validate_saved_manifest(manifest,condition,fitted["receipt"],context);manifest_sha=sha(public_prediction/"PREDICTION_MANIFEST.json")
                if mode=="replay":result=replay(fitted,condition,arrays,task,pool,context,output,deadline);result["manifest_sha256"]=manifest_sha
                else:result=cost(fitted,condition,task,pool,context,manifest_sha,output,batch,deadline,quiet_path,quiet_sha,proofs or {})
            foundation.assert_frozen_unchanged();result["F_final_state_sha256"]=gpu_runtime.state_hash(foundation.model)
        if result.get("status")!="BUDGET_LIMIT_PRESERVED":p1.check_deadline(deadline)
    except Exception as error:
        failure=type(error).__name__;path=private/"failures"/(mode+"_"+identifier+"_"+str(time.time_ns())+".json")
        new(path,dict(error_class=failure,error_message=str(error),traceback=traceback.format_exc()));result=dict(status="FAILED_PRESERVED",error_class=failure,private_failure_cache_relative_path=path.relative_to(cache).as_posix())
    finally:
        final_counters=None if pool is None else pool.counters()
        foundation_instances=0 if pool is None else pool.instances_created
        try:
            if pool is not None:pool.close()
            metadata(public,cache,asset,plan_sha,identifier,psutil);libraries["seed"].resource_preflight(public,cache,plan,psutil)
            if context is not None and "torch" in context and context["torch"].cuda.is_initialized():raise RuntimeError("CPU operation initialized CUDA")
        except Exception as error:
            post_error=type(error).__name__;post_path=private/"failures"/("post_"+mode+"_"+identifier+"_"+str(time.time_ns())+".json")
            new(post_path,dict(error_class=post_error,error_message=str(error),traceback=traceback.format_exc()))
        new(output/"RESULT.json",dict(status="FAILED_PRESERVED" if failure or post_error else (result or {}).get("status","INCOMPLETE"),run_id=identifier,mode=mode,plan_sha256=plan_sha,
            scientific_result=result,error_class=failure,postflight_error_class=post_error,postflight_private_failure_cache_relative_path=None if post_path is None else post_path.relative_to(cache).as_posix(),
            elapsed_wall_seconds=time.time()-started,original_source39_immutable9_registered2_GPU_refs_and_model_pins_preserved=post_error is None,new_T_target_operations=0,
            actual_F_counters=final_counters,foundation_instances_created=foundation_instances,numpy_imported="numpy" in sys.modules,torch_imported="torch" in sys.modules,
            actual_CPU_process_seconds=sum(psutil.Process().cpu_times()[:2]),RSS_bytes=psutil.Process().memory_info().rss,OS_peak_working_set_bytes=getattr(psutil.Process().memory_info(),"peak_wset",None),
            old_full_task_identity_inherited_not_recomputed=True,independent_generalization_claimed=False,ROOT_RSS_CPU2_2slot_6GiB_deadline_and_quiet_supervisor_required=True,historical_GPU_screen_and_Beijing_probe_loader_ancestry_preserved=True))
    if failure or post_error:raise RuntimeError("Full-C prefix operation failed; private traceback and original evidence preserved")
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--execute",action="store_true");parser.add_argument("--root-triggered",action="store_true")
    for key in ("public-root","cache-root","asset-root","plan-sha","run-id","output","quiet-proof","quiet-sha","rescore-proof","rescore-sha","replay-proof","replay-sha"):parser.add_argument("--"+key)
    parser.add_argument("--mode",choices=("fit","rescore","replay","cost"));parser.add_argument("--deadline",type=float);parser.add_argument("--batch",type=int,choices=(1,8));args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="DRAFT_NOT_EXECUTED",input_reads=0,writes=0,actual_training=0,actual_new_training_conditions=0,scientific_or_child_operations=0,numpy_imported="numpy" in sys.modules,torch_imported="torch" in sys.modules,candidate_ids=[c["run_id"] for c in conditions()])))
        return 0
    if not args.root_triggered or any(getattr(args,key) is None for key in ("public_root","cache_root","asset_root","plan_sha","run_id","mode","deadline","output")):
        parser.error("Explicit ROOT trigger, post-N4 new sealed PLAN SHA, roots, exact ID/mode/deadline and fresh output required")
    result=execute(args.public_root,args.cache_root,args.asset_root,args.plan_sha,args.run_id,args.mode,args.deadline,args.output,args.batch,args.quiet_proof,args.quiet_sha,dict(rescore=(args.rescore_proof,args.rescore_sha),replay=(args.replay_proof,args.replay_sha)))
    print(json.dumps(dict(status=result["status"],run_id=args.run_id,mode=args.mode)))
    return 0 if result["status"].startswith(("COMPLETE","PASS")) else 1


if __name__=="__main__":raise SystemExit(main())
