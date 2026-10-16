"""Prepare four existing mandatory family minima; ROOT alone admits execution.

Default is stdlib-only with no input reads. Fresh prefix-only CPU fits,
saved-V rescore, actual CPU6 replay and standalone costs have explicit modes.
Original scientific sources, old probes/screens and interrupted runs are kept.
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

NAME = "priority_valid_family_20261016_v1"
OLD = "postT_V_only_cpu_family_minima_20261016_v1"
OLD_PLAN_SHA = "e34033935916dcaded0885ccde8e1aa084f5cc0df994024d0827510d6db53b10"
OLD_RUNNER_SHA = "16e3a30d678b61ac49328ca1d3f31386bb03cbc4947f2c396b0fc1622258fbd9"
UPSTREAM_PLAN_SHA = "475f983058871ff36b4112b6aa56dc252b17aea47eb6ae1744ff119c2ad5138f"
DEADLINE, L, H, SEED = 1792222550, 512, 48, 92601
BOUNDS = {
    "jena":dict(A_fit=[0,10482],A_stop=[10482,12579],M_cal=[12579,14675],V_select=[14675,16772],T_final=[16772,20965]),
    "beijing":dict(A_fit=[0,10519],A_stop=[10519,12622],M_cal=[12622,14726],V_select=[14726,16830],T_final=[16830,21038]),
}
FAMILY_PINS = {
    "PLAN.json":OLD_PLAN_SHA,
    "REGISTRY.csv":"631f294006c88a1487b9dc14f731beb4437279e8c3b8d8f1a26d79762b0ed6b2",
    "EXPERIMENT_REGISTRY.csv":"631f294006c88a1487b9dc14f731beb4437279e8c3b8d8f1a26d79762b0ed6b2",
    "family_minima_runner.py":OLD_RUNNER_SHA,
    "test_family_minima_runner.py":"ae87c2c68da360e25a1de640611a0604798c42e19a28bc767674a9fdd5316d11",
    "verify_family_minima_outputs.py":"907488e3bf92e3ac13e90f0fc30c4dac09e241e0b51f7ba7c0b274e654dc45fa",
    "test_verify_family_minima_outputs.py":"d11761151b8e4b79cd35808d7965ece9234ceeb4d325680ed02a959799fc6872",
}
DEPENDENCIES = {
    "priority_valid_minima_20261016_v1/beijing_minimum_runner.py":"b84f1473ff4d562890910f7555ef3925823dd921539c8d757118c95504513b9b",
    "priority_b2_validation_20261016_v1/priority_runner.py":"f3e4279ddd250a51b09ce88938f8f202dfd02e3b24d1adff3bb36c567dc7ef07",
    "priority_b2_validation_20261016_v1/safe_jena_prefix.py":"6b496dcbbb6cef48acb9010c630ed81359a2d60bff89aaec9e9e77c4b6d0b541",
    "priority_core_followup_20261016_v1/jena_seed_runner.py":"ad4b76acf51fb416dea4f69c36940507db1f92e223c620388ded45381d6e28e7",
    "priority_nested_cost_20261016_v1/prefix_nested_cost_runner.py":"d64d543499ed21c5a294ace19967c555d05bcfde0cd5ea6c98cbe8ddb23a4ae5",
}
TARGETS = (
    ("jena","B4",6,"B4",6,"367c572107d05a9fb82c2b22cbf7e63f0eff545c47303d73f5d89cca9fa5162d","14a7f8e6b58060520f0d9ff0f4207fe038eb32b98904d0269b0da9e5b0a7a773"),
    ("jena","ADAPTS_BOLT_PORT",6,"ADAPTS",6,"b89b1b1f813647d0a4d7af66fa92db96be4faab8ea079ccb5eef7ba009781224","807db1e494d5f2bbe0487f49b02a27362cf9a75a9294976677225545469fda14"),
    ("jena","LORA_R8",0,"LORA",21,"7da56fa3ab2e023552e8f297c16d28a4fa60c527bfcc6ebed2f30180bc561cda","661dd5912cd18e3b0558ec31042fe47995bfc5ca260a14338f6cacce681b557d"),
    ("beijing","B4",11,"B4",11,"41158c5c260b1e0c0b01406440a17b8163f98d2bf4b0cb7b0992f728c20ca8dd","f3f1d07902dd1e29a1c50e88fd4ad4050565f221341befa16686fa489c545ba5"),
)
MODEL_ROOT=".cache/huggingface/models--amazon--chronos-bolt-small/snapshots/772f3d25d38aec6d914c8949dab4462e2d46f5d8"
MODEL_BINDINGS=[
    dict(snapshot_relative_path=MODEL_ROOT+"/config.json",resolved_blob_relative_path=".cache/huggingface/models--amazon--chronos-bolt-small/blobs/d8f9b65552d4d211e28d332ad5666484092bbf77",sha256="9ca0ebbe049ccebacb0546361e8aa728f0cf104cd18d1838d78d296aa317bd21",bytes=1121),
    dict(snapshot_relative_path=MODEL_ROOT+"/model.safetensors",resolved_blob_relative_path=".cache/huggingface/blobs/61/61439ff1becff564d030cdbe1ce9dfce0673fc0ebc4ca0a009d3fe02ce60bba0",sha256="06a6a19bbe74bc10a9cd193bd4bf2bf638ae07f7e0d51653ae7ab8ea968a21dd",bytes=190888824),
]


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


def conditions():
    rows=[]
    for index,(dataset,label,registered_k,family,width,probe_sha,reference_sha) in enumerate(TARGETS,1):
        upstream=f"primary_{dataset}_E1_{label}_k{registered_k}_s92601"
        identifier=f"postT_cpu_{dataset}_E1_{label}_k{registered_k}_s92601_family_v1"
        spec=dict(run_id=identifier,dataset=dataset,family=family,latent_k=width,seed=SEED,fold="primary",epochs=[.25,1.,3.],
            lr=.0001 if family=="LORA" else .001,device="cpu",microbatch=4,effective_batch=16,F_row_microbatch=128,
            checkpoint_inputs=False,lora_rank=8,lora_alpha=16,stage_label="postT_cpu_fresh_V_only_family_minima",
            provenance=dict(upstream_registered_id=upstream,old_weights_optimizer_RNG_or_state_reused=False))
        rows.append(dict(run_id=identifier,dataset=dataset,family=family,registered_family=label,registered_latent_k=registered_k,
            latent_k=width,effective_F_width=width,upstream_registered_id=upstream,spec=spec,
            probe=dict(relative_path=OLD+f"/diagnostics/probe_root_v1_{index:02d}/RESULT.json",sha256=probe_sha),
            V_reference=dict(relative_path="predictions/"+upstream+"/V_select/PREDICTION_MANIFEST.json",sha256=reference_sha)))
    return rows


def draft_plan():
    return dict(schema="PRIORITY_VALID_FAMILY_MINIMA_v1",namespace=NAME,status="DRAFT_NOT_EXECUTABLE",runner_sha256=None,
        conditions=conditions(),stage="POST_T_V_ONLY_DEVELOPMENT",maximum_new_training_conditions=4,
        original_family_plan_sha256=OLD_PLAN_SHA,original_family_files=copy.deepcopy(FAMILY_PINS),dependencies=copy.deepcopy(DEPENDENCIES),
        upstream_priority_plan_sha256=UPSTREAM_PLAN_SHA,original_primary_split_ranges=copy.deepcopy(BOUNDS),
        materialized_prefix_rows=dict(jena=16772,beijing=16830),protected_tail_numeric_parse_allowed=False,
        new_T_target_operations_allowed=False,deadline_unix=DEADLINE,
        optimizer=dict(name="AdamW",weight_decay=0.,effective_batch=16,microbatch=4,gradient_clip=1.,lr_by_family=dict(B4=.001,ADAPTS=.001,LORA=.0001)),
        resources=dict(CPU_threads=2,max_worker_RSS_bytes=3*1024**3,maximum_model_workers=2,total_owned_RSS_limit_bytes=6*1024**3,
            storage_limit_bytes=30*1024**3,admission_reserve_bytes=256*1024**2,costs_must_not_overlap_training=True,ROOT_CPU_supervisor_required=True),
        authorization_scope="Existing Priority section12 deferred minimum comparisons, after N4 reports and ROOT admission; not additional slots under first-Jena maximum4.",
        model_snapshot_binding_source="priority_nested_cost_20261016_v1/PLAN.json",
        model_snapshot_bindings=copy.deepcopy(MODEL_BINDINGS),
        original_source39_immutable9_reference=dict(relative_path=OLD+"/PLAN.json",sha256=OLD_PLAN_SHA,source_count=39,immutable_count=9),
        task_asset_bindings={
            "jena":dict(relative_path=".cache/tsfm_peft_practical_controls_v7_20260928/data/jena_trainval.npz",sha256="29cfae52d7cd7128e551ef85da11138ed4d3f7b3ccf27af359c4fc1dedf4a933",bytes=1561064,contract_relative_path="research/level_matched_learned_ed_v18_20261006/data_contract_jena.json",contract_sha256="678bd4cbd63265290c0eeed8a78b09545a05219b666fd3b818682ce385906820",source_rows=35135,permitted_rows=20965,prefix_rows=16772),
            "beijing":dict(relative_path=".cache/smallk_predictive_alignment_20261013/beijing_repair/raw_dataset/beijing_aotizhongxin/b04da438b2f331ac/fit_tune_payload.npz",sha256="78063851dd3e65512e3e3ced22210839697553f2d652fc0be9513736af598277",bytes=1524625,contract_relative_path=".cache/smallk_predictive_alignment_20261013/beijing_repair/raw_dataset/beijing_aotizhongxin/b04da438b2f331ac/contract.json",contract_sha256="9116703c22b3dea317cb1ff750fd45d05c5d6e4a226946a41334e0794484849f",source_rows=28051,permitted_rows=21038,prefix_rows=16830)},
        old_probe_scope="Reused discarded3-update probes after exact recipe/source/A50/prefix validation; historical full-loader/T-byte ancestry remains.",
        interruption="Original _stage optimizer-boundary/deadline checkpoint behavior retained; arbitrary signal safety not assumed; no partial resume in this entry.",
        costs=dict(batches=[1,8],warmups=10,timed_trials=20,stage_trials=20,quiet_max_age_seconds=120,
            prior_actual_rescore_replay_outer_proof_SHA_required=True,physical_output_inverse_dtype="float64",row_microbatch=128),
        actual_training_prediction_replay_cost_or_dispatch_operations=0,
        ROOT_seal_required="Review source/tests; preserve DRAFT, create new PLAN.json with SEALED_BY_ROOT and exact runner SHA. No automatic execution or queue.")


def validate_plan(plan):
    expected=draft_plan()
    if plan.get("status")!="SEALED_BY_ROOT" or not isinstance(plan.get("runner_sha256"),str) or len(plan["runner_sha256"])!=64 or any(c not in "0123456789abcdef" for c in plan["runner_sha256"]):
        raise ValueError("Explicit ROOT-sealed new PLAN and nonnull exact runner SHA required")
    for field in ("schema","namespace","conditions","stage","maximum_new_training_conditions","original_family_plan_sha256","original_family_files",
                  "dependencies","upstream_priority_plan_sha256","original_primary_split_ranges","materialized_prefix_rows","protected_tail_numeric_parse_allowed",
                  "new_T_target_operations_allowed","deadline_unix","optimizer","resources","costs","authorization_scope","model_snapshot_binding_source",
                  "model_snapshot_bindings","task_asset_bindings","original_source39_immutable9_reference"):
        if plan.get(field)!=expected[field]:raise ValueError("Only exact four existing family recipes/prefix/resource contracts allowed")


def choose(plan,identifier):
    found=[c for c in plan["conditions"] if c["run_id"]==identifier]
    if len(found)!=1:raise ValueError("One exact new-entry family target ID required")
    return found[0]


def require_unstarted(public,cache,identifier):
    for root in (Path(public),Path(cache),Path(public)/OLD,Path(cache)/OLD,Path(public)/NAME,Path(cache)/NAME):
        for kind in ("runs","predictions"):
            if (root/kind/identifier).exists():raise FileExistsError("Preserve existing full/partial attempt; no automatic reuse, rename or resume")
        if (root/"claims"/("fit_"+identifier+".json")).exists():raise FileExistsError("Existing fit claim preserved")


def require_quiet(proof,now):
    stamp=proof.get("captured_unix")
    if (proof.get("status")!="QUIET_CPU_COST_WINDOW" or proof.get("active_model_workers")!=[] or proof.get("active_cost_workers")!=[]
            or isinstance(stamp,bool) or not isinstance(stamp,(int,float)) or not math.isfinite(stamp) or not 0<=now-stamp<=120):
        raise ValueError("Fresh120s ROOT quiet proof with both active lists empty required")


def validate_prefix_shape(task,original,dataset):
    width=21 if dataset=="jena" else 11;stop=BOUNDS[dataset]["V_select"][1]
    if (task["physical"].shape!=(stop,width) or original["shape"]!=[BOUNDS[dataset]["T_final"][1],width]
            or task["split_ranges"]!=BOUNDS[dataset] or task["data_sha256"]!=original["data_sha256"]):
        raise ValueError("Original bounds/inherited identity and exact pre-T prefix shape required")


def validate_origins(origins,dataset):
    lo,hi=BOUNDS[dataset]["V_select"]
    if not len(origins) or any(int(o)<lo or int(o)+H>hi for o in origins):raise ValueError("Origins must stay in exact original V without T")


def validate_statistics(original,a50_fingerprints,a50_scaler,a60_scaler):
    if (a50_fingerprints!=original["A_fit_statistics_fingerprints"] or a50_scaler!=original["search_scaler_sha256"] or a60_scaler!=original["refit_scaler_sha256"]):
        raise ValueError("Actual A50 or A60 prefix statistics differ from original probe/Task")


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
        if route.family=="LORA" and not hasattr(route,"_merged_weights"):raise ValueError("Original LoRA Route merge-state lifecycle required")
        yield route
    finally:
        route.close()
        foundation.assert_frozen_unchanged()


def fit_stages(fit,condition,task,foundation,contract,public_run,private_run,deadline):
    bounds=BOUNDS[condition["dataset"]];spec=condition["spec"]
    search=fit._stage(spec,task,bounds["A_fit"][1],bounds["A_stop"],[.25,1.,3.],private_run/"search",public_run/"search",foundation,deadline,contract)
    if search["summary"]["status"]!="SEGMENT_COMPLETE":return dict(status="BUDGET_LIMIT_PRESERVED",phase="search",summary=search["summary"])
    epoch=search["summary"]["selected_epoch"]
    if epoch not in (0.,.25,1.,3.):raise ValueError("Only original A_stop-selected0/.25/1/3 endpoints allowed")
    refit=fit._stage(spec,task,bounds["A_stop"][1],None,[epoch] if epoch else [],private_run/"refit",public_run/"refit",foundation,deadline,contract)
    if refit["summary"]["status"]!="SEGMENT_COMPLETE":return dict(status="BUDGET_LIMIT_PRESERVED",phase="refit",summary=refit["summary"])
    return dict(status="COMPLETE_STAGES",search=search,refit=refit,selected_epoch=epoch)


def validate_stages(condition,search,refit):
    bounds=BOUNDS[condition["dataset"]];nsearch=bounds["A_fit"][1]-L-H+1;nrefit=bounds["A_stop"][1]-L-H+1;epoch=search.get("selected_epoch")
    if epoch not in (0.,.25,1.,3.):raise ValueError("Original selected search endpoint required")
    targets=[0]+[math.ceil(endpoint*nsearch) for endpoint in (.25,1.,3.)]
    search_updates=sum(math.ceil((end-start)/16) for start,end in zip(targets,targets[1:]))
    for value,want in ((search,dict(status="SEGMENT_COMPLETE",completed_epochs=[0.,.25,1.,3.],train_origins=nsearch,exposures=3*nsearch,updates=search_updates)),
                       (refit,dict(status="SEGMENT_COMPLETE",completed_epochs=[0.]+([epoch] if epoch else []),train_origins=nrefit,exposures=math.ceil(epoch*nrefit),updates=math.ceil(math.ceil(epoch*nrefit)/16)))):
        if any(value.get(k)!=v for k,v in want.items()):raise ValueError("Original full search and fresh selected A60 refit activity required")


def metadata(public,cache,asset,plan_sha,identifier,process_api):
    path=relative(public,NAME+"/PLAN.json")
    if not plan_sha or sha(path)!=plan_sha:raise ValueError("Explicit new ROOT-sealed PLAN SHA required")
    plan=read(path);validate_plan(plan);condition=choose(plan,identifier)
    if sha(relative(public,NAME+"/family_minimum_runner.py"))!=plan["runner_sha256"]:raise ValueError("New runner source changed")
    check_pins(public,DEPENDENCIES);check_pins(relative(public,OLD),FAMILY_PINS)
    libraries={key:module(relative(public,path),"family_prefix_"+key) for key,path in (
        ("common","priority_valid_minima_20261016_v1/beijing_minimum_runner.py"),
        ("priority","priority_b2_validation_20261016_v1/priority_runner.py"),
        ("seed","priority_core_followup_20261016_v1/jena_seed_runner.py"),
        ("primitive","priority_b2_validation_20261016_v1/safe_jena_prefix.py"),
        ("nested","priority_nested_cost_20261016_v1/prefix_nested_cost_runner.py"),
        ("family",OLD+"/family_minima_runner.py"))}
    p1=libraries["priority"].metadata(public,cache,UPSTREAM_PLAN_SHA,process_api);old=read(relative(public,OLD+"/PLAN.json"))
    if len(old["source39"])!=39 or len(old["immutable_public"])!=9 or old["source39"]!=p1["source39"] or old["immutable_public"]!=p1["immutable_public"]:raise ValueError("Original source39/immutable9 pins differ")
    check_pins(public,{**old["source39"],**old["immutable_public"],**old["dependencies"]})
    original_condition=next(c for c in old["conditions"] if c["run_id"]==identifier)
    if original_condition!={k:v for k,v in condition.items() if k not in {"probe","V_reference"}}:raise ValueError("Exact original family recipe differs")
    for co in plan["conditions"]:
        original=old["task_contracts"][co["dataset"]];recipe="runs/"+co["upstream_registered_id"]+"/SOURCE_CONTRACT.json"
        if sha(relative(public,recipe))!=old["upstream_recipe_metadata"][recipe]:raise ValueError("Upstream recipe changed")
        libraries["family"].validate_original(read(relative(public,recipe)),co,old["source39"],original,old["foundation_tensor_sha256"])
        probe=co["probe"]
        if sha(relative(public,probe["relative_path"]))!=probe["sha256"]:raise ValueError("Historical own probe changed")
        libraries["family"].validate_probe_proof(read(relative(public,probe["relative_path"])),co,original,OLD_PLAN_SHA,OLD_RUNNER_SHA)
        ref=co["V_reference"]
        if sha(relative(public,ref["relative_path"]))!=ref["sha256"]:raise ValueError("Pinned original V reference changed")
    for dataset,contract in old["task_contracts"].items():
        meta=contract["metadata"];source=relative(asset,meta["source_reference"])
        cp=libraries["primitive"].CONTRACT_REL if dataset=="jena" else str(Path(meta["source_reference"]).with_name("contract.json"))
        pin=plan["task_asset_bindings"][dataset]
        if (pin["relative_path"]!=meta["source_reference"] or pin["sha256"]!=meta["source_sha256"] or pin["bytes"]!=meta["source_bytes"] or pin["contract_sha256"]!=meta["source_contract_sha256"]
                or relative(asset,pin["contract_relative_path"])!=relative(asset,cp) or source.stat().st_size!=pin["bytes"] or sha(source)!=pin["sha256"] or sha(relative(asset,cp))!=pin["contract_sha256"]):raise ValueError("Original opaque raw asset/contract changed")
    nested_plan=relative(public,plan["model_snapshot_binding_source"])
    if sha(nested_plan)!="73318f37a2505b42271362180d4c20570770c2f6ff7b4346c5fe29605358fc0a":raise ValueError("Original exact2 snapshot binding source changed")
    if read(nested_plan)["model_snapshot_bindings"]!=plan["model_snapshot_bindings"]:raise ValueError("Original exact2 model bindings differ")
    libraries["common"].check_model_snapshot(asset,old["model_asset_relative_path"],plan["model_snapshot_bindings"])
    if (sha(relative(asset,old["instance_norm_source_relative_path"]))!=old["instance_norm_source_sha256"]
            or sha(relative(public.parent,"temporal_frontend_campaign_20261015_v1/runtime.py"))!=old["instance_norm_source_sha256"]):raise ValueError("Actual normalization source changed")
    return plan,old,condition,libraries


def safe_task(asset,old,condition,libraries,context):
    dataset=condition["dataset"];original=old["task_contracts"][dataset];np,data,fit=context["np"],context["data"],context["fit"]
    if dataset=="jena":task,access=libraries["primitive"].load_jena_prefix(asset,original)
    else:task,access=libraries["nested"].safe_task(asset,dict(task_contract=original),libraries["primitive"],np)
    validate_prefix_shape(task,original,dataset)
    if access["new_T_numeric_values_parsed"]!=0:raise ValueError("Protected tail numeric decoding forbidden")
    bounds=BOUNDS[dataset];mean,std,count=data.fit_statistics(task,bounds["A_fit"][1]);a50={key:data.array_sha256(value) for key,value in (("mean",mean),("std",std),("count",count))};a50scale=fit.scaler_fingerprint(mean,std)
    mean,std,_=data.fit_statistics(task,bounds["A_stop"][1]);validate_statistics(original,a50,a50scale,fit.scaler_fingerprint(mean,std))
    arrays=libraries["priority"].cohort_arrays(task,bounds,mean,std,data,np);libraries["priority"].validate_cohort(arrays,context["reference"],context["evaluation"])
    access.update(actual_A50_A60_and_V43_original_fingerprints_matched=True,old_probe_historical_full_loader_scope_retained=True,old_probe_is_reused_not_reexecuted=True)
    return task,access


def fit_and_save(condition,task,foundation,context,public_run,private_run,deadline):
    fit,p1=context["fit"],context["priority"];spec=condition["spec"];dataset=condition["dataset"];bounds=BOUNDS[dataset]
    source,files=fit._source_contract();contract=dict(source_sha=source,config_sha=fingerprint(dict(spec=spec,L=L,H=H,C=len(task["columns"]),columns=task["columns"],inherited_array_sha256=task["array_sha256"],materialized_prefix_sha256=task["materialized_prefix_sha256"])),data_sha=task["data_sha256"],model_sha=fit._model_sha(foundation))
    new(public_run/"SOURCE_CONTRACT.json",dict(**contract,files=files,spec=spec,original_task_identity="INHERITED_NOT_RECOMPUTED",actual_prefix_sha256=task["materialized_prefix_sha256"],original_probe=condition["probe"],prefix_only_new_entry=True))
    stages=fit_stages(fit,condition,task,foundation,contract,public_run,private_run,deadline)
    if stages["status"]!="COMPLETE_STAGES":return stages
    search,refit,epoch=stages["search"],stages["refit"],stages["selected_epoch"];validate_stages(condition,search["summary"],refit["summary"])
    for label,stage in (("search",search),("refit",refit)):p1.write_stage(public_run/label,stage)
    bundle=dict(model=refit["model"],normalization=refit["normalization"],spec=spec,contract=refit["summary"]["contract"],selected_search_epoch=epoch,
        search_summary=search["summary"],refit_summary=refit["summary"],model_state_sha=p1.state_sha(refit["model"],context["data"]),plan_sha256=context["plan_sha"],actual_prefix_sha256=task["materialized_prefix_sha256"])
    artifact=private_run/"selected_bundle.pt";p1.save_bundle(artifact,bundle,context["torch"])
    receipt=dict(status="COMPLETE_VALID",run_id=spec["run_id"],family=spec["family"],seed=SEED,epochs=[.25,1.,3.],selected_epoch=epoch,
        training_status="UNDERTRAINED_OR_BUDGET_LIMIT" if epoch==3 else "SCREEN_COMPLETE_CONVERGENCE_UNRESOLVED",optimizer="AdamW",weight_decay=0.,
        artifact_sha256=sha(artifact),model_state_sha=bundle["model_state_sha"],refit_scaler_sha=fit.scaler_fingerprint(bundle["normalization"]["mean"],bundle["normalization"]["std"]),
        artifact_cache_relative_path=NAME+"/runs/"+spec["run_id"]+"/selected_bundle.pt",search_fit_end=bounds["A_fit"][1],refit_fit_end=bounds["A_stop"][1],
        search_updates=search["summary"]["updates"],refit_updates=refit["summary"]["updates"],search_exposures=search["summary"]["exposures"],refit_exposures=refit["summary"]["exposures"],
        source_sha=source,source_stage_API="original_fit_base._stage",original_probe=condition["probe"],probe_weights_optimizer_or_sampler_reused=False,
        original_full_task_identity="INHERITED_NOT_RECOMPUTED",actual_prefix_sha256=task["materialized_prefix_sha256"],post_T_development=True,protected_T_target_operations=0,reused_complete_artifact=False)
    new(public_run/"BASE_FIT_RECEIPT.json",receipt)
    return dict(status="COMPLETE_VALID",receipt=receipt,artifact=artifact)


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


def saved_arrays(public_folder,private_folder,condition,receipt,context):
    manifest=read(public_folder/"PREDICTION_MANIFEST.json");reference=read(public_folder/"REFERENCE_CONTRACT.json")
    want=dict(run_id=condition["run_id"],family=condition["family"],seed=SEED,split="V_select",data_sha=context["original"]["data_sha256"],artifact_sha256=receipt["artifact_sha256"],scaler_sha=receipt["refit_scaler_sha"],source_sha=fingerprint(context["new_sources"]),config_sha=fingerprint(condition["spec"]),model_sha=fingerprint(dict(artifact_sha=receipt["artifact_sha256"],model_state_sha=receipt["model_state_sha"])))
    if any(manifest.get(key)!=value for key,value in want.items()) or manifest.get("device")!="cpu":raise ValueError("Foreign or changed saved-V identity")
    archive=private_folder/"V_select.npz"
    if sha(archive)!=manifest["archive_sha256"]:raise ValueError("Saved V bytes changed")
    with context["np"].load(archive,allow_pickle=False) as z:
        if set(z.files)!={"pred_normalized","target_normalized","observed","origins","channel_ids"}:raise ValueError("Exact safe V members required")
        arrays={key:z[key] for key in z.files}
    for key,value in arrays.items():
        if context["evaluation"].array_fingerprint(value)!=manifest["arrays"][key]:raise ValueError("Saved-V member fingerprint differs")
    if arrays["pred_normalized"].shape!=(43,H,len(context["original"]["columns"])):raise ValueError("Original V43 geometry required")
    for original in (reference,context["reference"]):context["priority"].validate_cohort(arrays,original,context["evaluation"])
    if any(reference[key]!=manifest[key] for key in ("run_id","split","cohort","data_sha","scaler_sha","source_sha","config_sha","model_sha")):raise ValueError("Saved reference identity differs")
    validate_origins(arrays["origins"],condition["dataset"])
    return arrays,manifest


def rescore(public_folder,private_folder,condition,receipt,context,output):
    arrays,manifest=saved_arrays(public_folder,private_folder,condition,receipt,context)
    with (public_folder/"SCORES.csv").open(encoding="utf-8",newline="") as stream:scores=list(csv.DictReader(stream))
    if len(scores)!=1 or any(scores[0][key]!=manifest[key] for key in ("run_id","split","cohort")):raise ValueError("One exact score required")
    with gzip.open(public_folder/"PUBLIC_ERROR_STATS.csv.gz","rt",encoding="utf-8",newline="") as stream:rows=list(csv.DictReader(stream))
    result=context["priority"].independent_rescore(arrays,manifest,rows,scores[0],context["np"]);result.update(run_id=condition["run_id"],artifact_sha256=receipt["artifact_sha256"],manifest_sha256=sha(public_folder/"PREDICTION_MANIFEST.json"),full_V_origins=43)
    new(output/"RESCORE_RECEIPT.json",result);return result


def replay(fitted,condition,arrays,task,pool,context,output,deadline):
    np,p1=context["np"],context["priority"];indices=p1.replay_indices(43,np);origins=arrays["origins"][indices];route=actual_route(fitted,condition,task,pool,context);predictions=[];parity=dict(status="NOT_LORA");parity_delta={}
    with route_lifetime(route,pool.foundation):
        if condition["family"]=="LORA":
            o=int(origins[0]);before=pool.counters();parity=route.check_lora_parity(task["physical"][o-L:o][None],task["observed"][o-L:o][None],atol=1e-5,rtol=1e-4)
            parity_delta={key:pool.counters()[key]-before[key] for key in ("streams","physical_streams","physical_calls")}
            if parity.get("status")!="PASS" or parity_delta!=expected_counters(21,[1,1]):raise ValueError("Original LoRA merge parity/counters failed")
        before=pool.counters()
        for start in (0,4):
            p1.check_deadline(deadline);oo=origins[start:start+4];physical=np.stack([task["physical"][int(o)-L:int(o)] for o in oo]);observed=np.stack([task["observed"][int(o)-L:int(o)] for o in oo])
            raw=route.predict_physical(physical,observed);predictions.append(((raw-route.mean.astype(np.float64))/route.std.astype(np.float64)).astype(np.float32))
        actual=np.concatenate(predictions);delta={key:pool.counters()[key]-before[key] for key in ("streams","physical_streams","physical_calls")}
        if delta!=expected_counters(condition["effective_F_width"],[4,2]):raise ValueError("CPU6 original stream width/4+2 call accounting differs")
        passed=bool(np.allclose(actual,arrays["pred_normalized"][indices],atol=1e-5,rtol=1e-4))
        result=dict(status="PASS_FRESH_CPU6_REPLAY" if passed else "FAIL_REPLAY_PARITY",executed_now=True,run_id=condition["run_id"],artifact_sha256=fitted["receipt"]["artifact_sha256"],origins=origins.tolist(),microbatches=[4,2],F_counters=delta,lora_merge_parity=parity,lora_merge_parity_F_counters=parity_delta,base_restored_after_original_Route_close=True,atol=1e-5,rtol=1e-4,max_abs=float(np.abs(actual.astype(np.float64)-arrays["pred_normalized"][indices].astype(np.float64)).max()),new_T_target_operations=0)
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
        model_load_seconds=route.load_seconds,LoRA_merge_and_model_loading_in_whole_trial_timer=False,
        CPU_restore_snapshot_bytes_in_Route_memory_state=False if condition["family"]=="LORA" else "NOT_APPLICABLE",
        validation_result_sha256={key:value[1] for key,value in proofs.items()},physical_input_sha256=hashlib.sha256(physical.tobytes()).hexdigest(),observation_mask_sha256=hashlib.sha256(observed.tobytes()).hexdigest(),origins=origins.tolist(),F_counters=delta,
        F_logical_streams_per_trial=want["streams"],F_calls_per_trial=want["physical_calls"],cached_Z_or_forecast_reads=0,physical_output_and_inverse_dtype="float64",stage_probe_separate_intrusive_scope=True,training_peak_included=False,
        unmerged_F_base_sha256=context["foundation_sha"],effective_F_state_sha256_during_measurement=effective_state,effective_F_state_scope="LoRA merged adapter effective weights" if condition["family"]=="LORA" else "original frozen unmerged F",
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
        context=dict(np=np,data=data,evaluation=evaluation,priority=p1,cache=cache,plan_sha=plan_sha,new_sources=dict(runner=plan["runner_sha256"],**DEPENDENCIES),
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
                fitted=completed(public_run,private_run,condition,context["original"]);arrays,manifest=saved_arrays(public_prediction,private_prediction,condition,fitted["receipt"],context);manifest_sha=sha(public_prediction/"PREDICTION_MANIFEST.json")
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
            elapsed_wall_seconds=time.time()-started,original_source39_immutable9_family7_probe4_recipe4_and_model_pins_preserved=post_error is None,new_T_target_operations=0,
            actual_F_counters=final_counters,foundation_instances_created=foundation_instances,numpy_imported="numpy" in sys.modules,torch_imported="torch" in sys.modules,
            actual_CPU_process_seconds=sum(psutil.Process().cpu_times()[:2]),RSS_bytes=psutil.Process().memory_info().rss,OS_peak_working_set_bytes=getattr(psutil.Process().memory_info(),"peak_wset",None),
            old_full_task_identity_inherited_not_recomputed=True,independent_generalization_claimed=False,ROOT_RSS_CPU2_2slot_6GiB_deadline_and_quiet_supervisor_required=True,historical_old_probe_T_deserialization_scope_preserved=True))
    if failure or post_error:raise RuntimeError("Family prefix operation failed; private traceback and original evidence preserved")
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--execute",action="store_true");parser.add_argument("--root-triggered",action="store_true")
    for key in ("public-root","cache-root","asset-root","plan-sha","run-id","output","quiet-proof","quiet-sha","rescore-proof","rescore-sha","replay-proof","replay-sha"):parser.add_argument("--"+key)
    parser.add_argument("--mode",choices=("fit","rescore","replay","cost"));parser.add_argument("--deadline",type=float);parser.add_argument("--batch",type=int,choices=(1,8));args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="DRAFT_NOT_EXECUTED",input_reads=0,writes=0,actual_training=0,scientific_or_child_operations=0,numpy_imported="numpy" in sys.modules,torch_imported="torch" in sys.modules,candidate_ids=[c["run_id"] for c in conditions()])))
        return 0
    if not args.root_triggered or any(getattr(args,key) is None for key in ("public_root","cache_root","asset_root","plan_sha","run_id","mode","deadline","output")):
        parser.error("Explicit ROOT trigger, sealed new PLAN SHA, roots, exact ID/mode/deadline and fresh output required")
    result=execute(args.public_root,args.cache_root,args.asset_root,args.plan_sha,args.run_id,args.mode,args.deadline,args.output,args.batch,args.quiet_proof,args.quiet_sha,dict(rescore=(args.rescore_proof,args.rescore_sha),replay=(args.replay_proof,args.replay_sha)))
    print(json.dumps(dict(status=result["status"],run_id=args.run_id,mode=args.mode)))
    return 0 if result["status"].startswith(("COMPLETE","PASS")) else 1


if __name__=="__main__":raise SystemExit(main())
