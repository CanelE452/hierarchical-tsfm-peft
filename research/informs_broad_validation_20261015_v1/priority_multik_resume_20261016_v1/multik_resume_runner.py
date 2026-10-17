"""Explicit saved-point multi-K continuation; default is stdlib-only and zero I/O.

ROOT must independently review/seal this adapter and admit each operation.
Original run/spec/checkpoint contracts are preserved inside a new namespace.
No old queue is launched and no protected-T numeric interface is provided.
"""
import argparse
import copy
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

NAME = "priority_multik_resume_20261016_v1"
OLD = "postT_V_only_cpu_multik_20261016_v1"
OLD_PLAN_SHA = "2051573b8dee48200eb34e96e37e7af0d197442619ea1ba52a36a80ad4e27aec"
P1_PLAN_SHA = "475f983058871ff36b4112b6aa56dc252b17aea47eb6ae1744ff119c2ad5138f"
SOURCE_SHA = "c58ed5ed690a812dd326f9c77a9c941268faeddafd1aea78c55e0036142df5e5"
DATA_SHA = "b39023bcae04736eff9a0fd9a3d56e10dc7f32bd668a53d681290cb9193d8269"
FOUNDATION_SHA = "c2b4edf7db22d3f18e58ca79299a5c7e864e9b8f277e4e29e2f88d07dbb35874"
SCALERS = {"search":"cb394e93fec460e933a56c46cfc43645ba787fd765f064668788e33d9d55bb9d",
           "refit":"3348bcf1fa76d382f23cf20597fa90de6e21dac587d4ca9f6883f235a10f93fb"}
INITIALIZATIONS = {"search":"a878e0df875f7c2154d907b445a209556a1e3b7231a309e555141499851709a7",
                   "refit":"88eb4e66400421d3e65d30256ac4cbda01458da5b279bbfae54f1af4d29a914d"}
DEADLINE, L, H, C, PREFIX = 1792222550, 512, 48, 11, 16830
BOUNDS = dict(A_fit=[0,10519],A_stop=[10519,12622],M_cal=[12622,14726],V_select=[14726,16830],T_final=[16830,21038])
BACKUP_ROOT = "priority_pause_validate_20261016_v1/checkpoint_backup_before_worker_stop_v1"
DEPENDENCIES = {
    OLD+"/multik_runner.py":"44cd5e3fed5f4ad8a2110c54c9118063c6838b68d9059dea46f6b33bc32be10a",
    "priority_b2_validation_20261016_v1/priority_runner.py":"f3e4279ddd250a51b09ce88938f8f202dfd02e3b24d1adff3bb36c567dc7ef07",
    "priority_b2_validation_20261016_v1/safe_jena_prefix.py":"6b496dcbbb6cef48acb9010c630ed81359a2d60bff89aaec9e9e77c4b6d0b541",
    "priority_nested_cost_20261016_v1/prefix_nested_cost_runner.py":"d64d543499ed21c5a294ace19967c555d05bcfde0cd5ea6c98cbe8ddb23a4ae5",
    "priority_core_followup_20261016_v1/jena_seed_runner.py":"ad4b76acf51fb416dea4f69c36940507db1f92e223c620388ded45381d6e28e7",
}
CACHE_METADATA = {
    "priority_pause_validate_20261016_v1/CHECKPOINT_STATE_INSPECTION_ROOT_v1.json":"70efa1db99f76262d66e8a3bf40328247c5c87d24beaaa4348c4cf0b0e357060",
    "priority_pause_validate_20261016_v1/CHECKPOINT_BACKUPS_ROOT_v1.json":"2a1f71cda2584e938eaa03b5cdfdf0a5a7c7f70f732680c8e4283d2c94436a30",
}
REFERENCE = "predictions/primary_beijing_E1_B2_k6_s92601/V_select/PREDICTION_MANIFEST.json"
REFERENCE_SHA = "4c017a501b0eb7b9842878162a62bfc4355c97e008b1a12eecc791d0519b23e9"
MODEL_REL = ".cache/huggingface/models--amazon--chronos-bolt-small/snapshots/772f3d25d38aec6d914c8949dab4462e2d46f5d8"
MODEL_BINDINGS = [
    dict(snapshot_relative_path=MODEL_REL+"/config.json",resolved_blob_relative_path=".cache/huggingface/models--amazon--chronos-bolt-small/blobs/d8f9b65552d4d211e28d332ad5666484092bbf77",
         sha256="9ca0ebbe049ccebacb0546361e8aa728f0cf104cd18d1838d78d296aa317bd21",bytes=1121),
    dict(snapshot_relative_path=MODEL_REL+"/model.safetensors",resolved_blob_relative_path=".cache/huggingface/blobs/61/61439ff1becff564d030cdbe1ce9dfce0673fc0ebc4ca0a009d3fe02ce60bba0",
         sha256="06a6a19bbe74bc10a9cd193bd4bf2bf638ae07f7e0d51653ae7ab8ea968a21dd",bytes=190888824),
]


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""): digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def new(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False); stream.write("\n")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def relative(root,name):
    root=Path(root).absolute(); path=root/name
    if Path(name).is_absolute() or ".." in Path(name).parts or any(p.is_symlink() or getattr(p,"is_junction",lambda:False)() for p in (path,*path.parents)):
        raise ValueError("Nonsymlink root-relative path required")
    root,path=root.resolve(),path.resolve()
    if not path.is_relative_to(root): raise ValueError("Path escapes declared root")
    return path


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path); value=importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def conditions():
    values=[]
    records=(
        ("shared","ORDERED_MULTI_K",[3,6,11],"2c45fbf51fa9fc221c80af22d95245f516ba49368b1a1cb074bd9a9a8ee6990f",
         "25ee6fa1365b5f0d12e2f5d63cb6f3c653283ed73a4af9670f609060fefb4085","ba456c0ce6222704eadca3a903d2ab861a59b4e50c9bc7138b7d659e1fc674f2",
         ["cf3cf3070911e355bfc0f7d65b5d5add01fcf83429b09007f78d5e333f7ec83b","77f76f20cf1bdf3437ce27850e0ff184fc7bbdee317093535d62ab2bc80c76ea",
          "b8cd41680ebf12ef1c7f554bfbee0325522e4986944638208283bc973b51f824","9e474289d608c5f3ffc558c50203a982ba4b0f6b1b99114b357f3d375c9cba4b",
          "5a3de1da04dde5c995a2236dbe2b2855ce4b85ab8e5f05f8170a39389e8ead4c"]),
        ("max_only","MAXK_PCA_TRAINED",[11],"00346aa84e49744b6480e9a57a2e97c27fe3ab64f388e4984c00be298b5f80ad",
         "2652dd283cb69816eac42db271b49333afe64013f5fe18028e3643d8ac9a3dbe","2b0f29175f5c75625f740b0318d06df9fe30b77cecef15b643bad97252c05b82",
         ["8c2ec007ee33d49d8adaf9e91ac688d939159226c07382fb45fc7bcd9ce4c236","f8c2ff919a50f7cff5f1432a39daf3842470639806685f260b27ea157f28427f",
          "c77c443e03bb8da1db9dd129dcd1b6bbc2e36980a55a3df453a1fddadf1aee1b","c265072fff68a22be52de404c1d2c6b0893621986773e915de71c5510375125d",
          "298ff62d4eef72a9dec5bcacbea216c3228da33ca5394b66b5605d0f7940f6e1"]))
    for kind,role,grid,config,source_sha,run_sha,digests in records:
        identifier=f"postT_cpu_beijing_E4_{role}_k11_s92601_v1"
        upstream=f"primary_beijing_E4_{role}_k11_s92601"
        spec=dict(family="B0",latent_k=11,seed=92601,epochs=[.25,1.,3.],lr=.001,fold="primary",microbatch=4,effective_batch=16,
            device="cpu",family_lock_sha="3289f56491ab68572a1b1651d31c55c58ad58f8a489caa21dfe96a32e0c6fd4f",run_id=identifier,k_grid=grid,
            dataset="beijing",F_row_microbatch=128,checkpoint_inputs=False,stage_label="postT_cpu_fresh_V_only_multik",provenance=dict(upstream_registered_id=upstream))
        core=dict(source_sha=SOURCE_SHA,config_sha=config,data_sha=DATA_SHA,model_sha=FOUNDATION_SHA)
        checkpoints=[]
        for index,digest in enumerate(digests):
            phase="search" if index<4 else "refit"; basename=f"checkpoint_{index if index<4 else 0:06d}.pt"
            checkpoints.append(dict(phase=phase,basename=basename,cache_relative_path=f"{OLD}/runs/{identifier}/{phase}/{basename}",
                backup_cache_relative_path=f"{BACKUP_ROOT}/{identifier}/{phase}/{basename}",bytes=[46929,56124,56188,56252,53201][index],sha256=digest))
        values.append(dict(kind=kind,run_id=identifier,role=role,spec=spec,core_contract=core,checkpoint_contracts={phase:dict(core,scaler_sha=SCALERS[phase],initialization_sha=INITIALIZATIONS[phase]) for phase in ("search","refit")},
            source_contract_public_relative_path=f"{OLD}/runs/{identifier}/SOURCE_CONTRACT.json",source_contract_sha256=source_sha,
            run_contract_cache_relative_path=f"{OLD}/runs/{identifier}/RUN_CONTRACT.json",run_contract_sha256=run_sha,checkpoints=checkpoints))
    return values


def outputs():
    result=[]
    for condition in conditions():
        for k in (3,6,11):
            role=condition["role"] if k==11 else ("SHARED_PREFIX" if condition["kind"]=="shared" else "POSTHOC_PREFIX")
            result.append(dict(run_id=f"postT_cpu_beijing_E4_{role}_k{k}_s92601_v1",parent_run_id=condition["run_id"],kind=condition["kind"],
                latent_k=k,split="V_select",variant="shared" if condition["kind"]=="shared" else ("max_only" if k==11 else "posthoc")))
    return result


def draft_plan():
    return dict(schema="PRIORITY_MULTI_K_SAVED_POINT_RESUME_v1",status="DRAFT_NOT_EXECUTABLE",namespace=NAME,runner_sha256=None,
        stage="POST_T_V_ONLY_DEVELOPMENT",original_multik_plan_sha256=OLD_PLAN_SHA,priority_plan_sha256=P1_PLAN_SHA,
        conditions=conditions(),outputs=outputs(),dependencies=copy.deepcopy(DEPENDENCIES),cache_metadata=copy.deepcopy(CACHE_METADATA),model_snapshot_bindings=copy.deepcopy(MODEL_BINDINGS),
        V_reference=dict(relative_path=REFERENCE,sha256=REFERENCE_SHA),original_primary_split_ranges=copy.deepcopy(BOUNDS),materialized_prefix_rows=PREFIX,
        protected_T_numeric_allowed=False,original_run_ids_specs_contracts_preserved=True,search_retraining_allowed=False,
        deadline_unix=DEADLINE,resources=dict(CPU_threads=2,max_worker_RSS_bytes=3*1024**3,maximum_model_workers=2,total_owned_RSS_limit_bytes=6*1024**3,
            storage_limit_bytes=30*1024**3,admission_reserve_bytes=256*1024**2,GPU_process_seconds=0,costs_must_not_overlap_training=True,ROOT_CPU_supervisor_required=True),
        lost_progress="UNKNOWN_NOT_LOGGED; restart from saved refit checkpoint_000000.pt only, not the arbitrary interrupted instant",
        RNG_scope="Sampler NumPy Generator and Torch CPU RNG are saved/restored. This does not prove bitwise parity with uninterrupted execution; global Python/NumPy RNG not saved.",
        timing_scope="Restored-search call seconds are new reconstruction overhead only; saved segment seconds and prior cumulative F counters remain separate; interrupted unsaved work is unknown.",
        independent_K_training_completed_by_this_adapter=False,nested_PCA_rerun_allowed=False,
        preparation_actual_checkpoint_tensor_Task_model_fit_inference_cost_child_operations=0,
        ROOT_seal_required="Review code and tests, metadata pins, current original budget and quiet admission. Preserve draft and create NEW PLAN.json with status SEALED_BY_ROOT and runner_sha256. No automatic admission.")


def validate_plan(plan):
    expected=draft_plan()
    if plan.get("status")!="SEALED_BY_ROOT" or not isinstance(plan.get("runner_sha256"),str) or len(plan["runner_sha256"])!=64:
        raise ValueError("ROOT-sealed separate PLAN and adapter SHA required")
    for key in expected:
        if key not in {"status","runner_sha256","ROOT_seal_required"} and plan.get(key)!=expected[key]: raise ValueError("Exact original resume/data/K/resource contract changed: "+key)


def validate_lineage(condition,source,run_contract,task):
    axes=dict(L=task["L"],H=task["H"],C=len(task["columns"]),columns=list(task["columns"]),fold="primary",source_array_fingerprints=task["array_sha256"])
    expected=dict(source_sha=SOURCE_SHA,config_sha=fingerprint(dict(spec=condition["spec"],axes=axes)),data_sha=DATA_SHA,model_sha=FOUNDATION_SHA)
    if source.get("spec")!=condition["spec"] or run_contract!=expected or any(source.get(key)!=value for key,value in expected.items()):
        raise ValueError("Original complete spec/axes/source/data/model contract changed; never rename or rewrite checkpoint contract")
    return expected


def validate_checkpoint(checkpoint,condition,phase):
    search=phase=="search"
    if phase not in ("search","refit") or checkpoint.get("contract")!=condition["checkpoint_contracts"][phase]: raise ValueError("Checkpoint initialization/scaler and all original contracts must match exactly")
    extra=checkpoint.get("extra",{}); sampler=checkpoint.get("sampler",{})
    if (checkpoint.get("updates")!=(1868 if search else 0) or sampler.get("exposures")!=(29880 if search else 0)
            or extra.get("completed_epochs")!=([0.,.25,1.,3.] if search else [0.]) or extra.get("best_epoch")!=(3. if search else 0.)
            or "rng" not in sampler or "torch_rng" not in checkpoint or checkpoint.get("cuda_rng")!=[]
            or "optimizer" not in checkpoint or (not search and checkpoint["optimizer"].get("state")!={})):
        raise ValueError("Only inspected completed search and untouched refit CP0 with optimizer/sampler/RNG are admissible")


def check_transport(cache,public,condition):
    for root,name,digest in ((public,condition["source_contract_public_relative_path"],condition["source_contract_sha256"]),
                            (cache,condition["run_contract_cache_relative_path"],condition["run_contract_sha256"])):
        if sha(relative(root,name))!=digest: raise ValueError("Original run/source metadata changed")
    for row in condition["checkpoints"]:
        for key in ("cache_relative_path","backup_cache_relative_path"):
            path=relative(cache,row[key])
            if not path.is_file() or path.stat().st_size!=row["bytes"] or sha(path)!=row["sha256"]: raise ValueError("Original and backup checkpoint bytes/SHA must both match")


def copy_chain(cache,public,condition,private_run,public_run):
    private_run,public_run=Path(private_run),Path(public_run)
    if private_run.exists() or public_run.exists(): raise FileExistsError("Fresh separate resume destination required")
    check_transport(cache,public,condition)
    private_run.mkdir(parents=True,exist_ok=False); public_run.mkdir(parents=True,exist_ok=False)
    pairs=[(relative(cache,condition["run_contract_cache_relative_path"]),private_run/"RUN_CONTRACT.json"),
           (relative(public,condition["source_contract_public_relative_path"]),public_run/"SOURCE_CONTRACT.json")]
    pairs.extend((relative(cache,row["cache_relative_path"]),private_run/row["phase"]/row["basename"]) for row in condition["checkpoints"])
    for source,target in pairs:
        target.parent.mkdir(parents=True,exist_ok=True)
        with source.open("rb") as input_stream,target.open("xb") as output_stream: shutil.copyfileobj(input_stream,output_stream)
        if sha(source)!=sha(target): raise ValueError("Byte-exact checkpoint/metadata transport failed")
    for label in ("search","refit"): (public_run/label).mkdir(exist_ok=False)
    check_transport(cache,public,condition)
    return dict(status="EXACT_SAVED_CHECKPOINT_CHAIN_COPIED",checkpoint_files_copied=len(condition["checkpoints"]),
        original_run_id=condition["run_id"],original_and_backup_bytes_preserved=True,checkpoint_contract_modified=False,
        unsaved_interrupted_progress="UNKNOWN_NOT_LOGGED",resume_saved_point="refit/checkpoint_000000.pt")


def counters(foundation):
    return {key:int(getattr(foundation,key,0)) for key in ("streams","row_calls","physical_streams","physical_calls")}


def directory_hashes(path):
    return {file.relative_to(path).as_posix():sha(file) for file in sorted(Path(path).rglob("*")) if file.is_file()}


def validate_stages(search,refit):
    expected_search=dict(status="SEGMENT_COMPLETE",selected_epoch=3.,completed_epochs=[0.,.25,1.,3.],train_origins=9960,exposures=29880,updates=1868,
        F_session_logical_streams=0,F_session_physical_streams=0)
    expected_refit=dict(status="SEGMENT_COMPLETE",selected_epoch=3.,completed_epochs=[0.,3.],train_origins=12063,exposures=36189,updates=2262)
    if any(search.get(key)!=value for key,value in expected_search.items()) or any(refit.get(key)!=value for key,value in expected_refit.items()):
        raise ValueError("Completed saved search/no new F and exact selected three-epoch A60 refit required")


def resume_stages(fit,condition,task,foundation,contract,public_run,private_run,deadline):
    spec=condition["spec"]; before=counters(foundation)
    previous={label:directory_hashes(root/"search") for label,root in (("private",private_run),("public",public_run))}
    search=fit._stage(spec,task,10519,[10519,12622],[.25,1.,3.],private_run/"search",public_run/"search",foundation,deadline,contract)
    after=counters(foundation)
    if after!=before or any(previous[label]!=directory_hashes(root/"search") for label,root in (("private",private_run),("public",public_run))):
        raise ValueError("Restoring completed search must execute zero F/training/A_stop calls and write zero checkpoints/events")
    if any(search["summary"].get(key)!=value for key,value in dict(status="SEGMENT_COMPLETE",selected_epoch=3.,updates=1868,exposures=29880,completed_epochs=[0.,.25,1.,3.]).items()):
        raise ValueError("Inspected completed original search selection changed")
    refit=fit._stage(spec,task,12622,None,[3.],private_run/"refit",public_run/"refit",foundation,deadline,contract)
    if refit["summary"].get("status")!="SEGMENT_COMPLETE": return dict(status="BUDGET_LIMIT_PRESERVED",phase="refit",summary=refit["summary"],search_new_optimizer_updates=0,search_new_F_calls=0)
    validate_stages(search["summary"],refit["summary"])
    return dict(status="COMPLETE_STAGES",search=search,refit=refit,search_new_optimizer_updates=0,search_new_F_calls=0)


def require_quiet(proof,now):
    captured=proof.get("captured_unix")
    if (proof.get("status")!="QUIET_CPU_COST_WINDOW" or proof.get("active_model_workers")!=[] or proof.get("active_cost_workers")!=[]
            or isinstance(captured,bool) or not isinstance(captured,(int,float)) or not math.isfinite(captured) or not 0<=now-captured<=120):
        raise ValueError("Fresh120s ROOT proof with no active training or cost worker required")


def check_model_snapshot(asset,bindings):
    asset=relative(asset,".")
    for row in bindings:
        snapshot=asset/row["snapshot_relative_path"]; blob=relative(asset,row["resolved_blob_relative_path"])
        resolved=snapshot.resolve(strict=True)
        if not resolved.is_relative_to(asset) or resolved!=blob or blob.stat().st_size!=row["bytes"] or sha(blob)!=row["sha256"]:
            raise ValueError("Actual foundation config/weight snapshot target differs")


def metadata(public,cache,asset,plan_sha,process_api):
    plan_path=relative(public,NAME+"/PLAN.json")
    if not plan_sha or sha(plan_path)!=plan_sha: raise ValueError("Exact ROOT sealed PLAN SHA required")
    plan=read(plan_path); validate_plan(plan)
    if sha(relative(public,NAME+"/multik_resume_runner.py"))!=plan["runner_sha256"]: raise ValueError("New adapter changed")
    for name,digest in DEPENDENCIES.items():
        if sha(relative(public,name))!=digest: raise ValueError("Reviewed helper source changed: "+name)
    for name,digest in CACHE_METADATA.items():
        if sha(relative(cache,name))!=digest: raise ValueError("Inspection/backup metadata changed")
    old_path=relative(public,OLD+"/PLAN.json")
    if sha(old_path)!=OLD_PLAN_SHA: raise ValueError("Original multi-K PLAN changed")
    old=read(old_path)
    if old["model_asset_relative_path"]!=MODEL_REL: raise ValueError("Original Small model snapshot changed")
    check_model_snapshot(asset,plan["model_snapshot_bindings"])
    libraries={key:module(relative(public,name),"resume_"+key) for key,name in (("old",OLD+"/multik_runner.py"),
        ("priority","priority_b2_validation_20261016_v1/priority_runner.py"),("primitive","priority_b2_validation_20261016_v1/safe_jena_prefix.py"),
        ("nested","priority_nested_cost_20261016_v1/prefix_nested_cost_runner.py"),("seed","priority_core_followup_20261016_v1/jena_seed_runner.py"))}
    libraries["old"].metadata(public,cache,asset,OLD_PLAN_SHA)
    p1=libraries["priority"].metadata(public,cache,P1_PLAN_SHA,process_api)
    if old["source39"]!=p1["source39"] or old["immutable_public"]!=p1["immutable_public"]: raise ValueError("Original source39/immutable9/T lock differ")
    for condition in conditions():
        original=next(row for row in old["fits"] if row["run_id"]==condition["run_id"])
        if original["spec"]!=condition["spec"]: raise ValueError("Original fit recipe differs")
        check_transport(cache,public,condition)
    if sha(relative(public,REFERENCE))!=REFERENCE_SHA: raise ValueError("Original V43 cohort reference changed")
    return plan,old,libraries


def safe_task(asset,old,libraries,context):
    task,access=libraries["nested"].safe_task(asset,old,libraries["primitive"],context["np"])
    data,fit=context["data"],context["fit"]
    if task["physical"].shape!=(PREFIX,C) or access["new_T_numeric_values_parsed"]!=0 or task["data_sha256"]!=DATA_SHA: raise ValueError("Exact safe prefix/full inherited identity required")
    for phase,end in (("search",10519),("refit",12622)):
        mean,std,_=data.fit_statistics(task,end)
        if fit.scaler_fingerprint(mean,std)!=SCALERS[phase]: raise ValueError("Original A50/A60 normalization changed")
    arrays=libraries["priority"].cohort_arrays(task,BOUNDS,mean,std,data,context["np"])
    libraries["priority"].validate_cohort(arrays,context["reference"],context["evaluation"])
    return task,access


def check_initialization(fit,condition,task,foundation,context):
    for phase,end in (("search",10519),("refit",12622)):
        model,norm=fit.create_model(condition["spec"],task,end,foundation)
        try:
            actual=fingerprint(dict(scaler_sha=fit.scaler_fingerprint(norm["mean"],norm["std"]),fit_end=end,
                basis_sha=context["data"].array_sha256(norm["basis"]),center_sha=context["data"].array_sha256(norm["pca_center"])))
            if actual!=INITIALIZATIONS[phase] or norm["basis_initialization"]!="A_fitted_centered_PCA": raise ValueError("Original seeded PCA initialization changed; full-C max grid must remain [11]")
        finally:
            if callable(getattr(model,"close",None)): model.close()
            del model


def resume_and_save(condition,task,foundation,context,public_run,private_run,deadline):
    fit,p1=context["fit"],context["priority"]
    source=read(relative(context["public"],condition["source_contract_public_relative_path"]))
    core=validate_lineage(condition,source,read(relative(context["cache"],condition["run_contract_cache_relative_path"])),task)
    if fit._source_contract()[0]!=core["source_sha"] or fit._model_sha(foundation)!=core["model_sha"]: raise ValueError("Actual original training/F source identity changed")
    check_initialization(fit,condition,task,foundation,context)
    for phase,index in (("search",3),("refit",4)):
        path=relative(context["cache"],condition["checkpoints"][index]["cache_relative_path"])
        validate_checkpoint(context["torch"].load(path,map_location="cpu",weights_only=False),condition,phase)
    transport=copy_chain(context["cache"],context["public"],condition,private_run,public_run)
    new(public_run/"RESUME_PROVENANCE.json",dict(transport,plan_sha256=context["plan_sha"],adapter_sha256=context["plan"]["runner_sha256"],
        parent_checkpoints=condition["checkpoints"],original_source_contract_sha256=condition["source_contract_sha256"],actual_prefix_sha256=task["materialized_prefix_sha256"],
        contracts_run_ID_stage_label_spec_provenance_unchanged=True,interrupted_unsaved_work_recovered=False,uninterrupted_bitwise_parity_claimed=False))
    stage=resume_stages(fit,condition,task,foundation,core,public_run,private_run,deadline)
    if stage["status"]!="COMPLETE_STAGES": return stage
    search,refit=stage["search"],stage["refit"]
    for label,result in (("search",search),("refit",refit)): p1.write_stage(public_run/label,result)
    bundle=dict(model=refit["model"],normalization=refit["normalization"],spec=condition["spec"],contract=refit["summary"]["contract"],
        selected_search_epoch=3.,search_summary=search["summary"],refit_summary=refit["summary"],model_state_sha=p1.state_sha(refit["model"],context["data"]),
        resume_adapter_plan_sha256=context["plan_sha"],actual_prefix_sha256=task["materialized_prefix_sha256"])
    artifact=private_run/"selected_bundle.pt"; p1.save_bundle(artifact,bundle,context["torch"])
    receipt=dict(status="COMPLETE_VALID",run_id=condition["run_id"],family="B0",seed=92601,epochs=[.25,1.,3.],selected_epoch=3.,
        selected_astop_mse=search["summary"]["selected_astop_mse"],training_status="UNDERTRAINED_OR_BUDGET_LIMIT",last_cp_best=True,
        search_updates=1868,search_exposures=29880,refit_updates=2262,refit_exposures=36189,search_fit_end=10519,refit_fit_end=12622,
        search_train_origins=9960,refit_train_origins=12063,source_sha=SOURCE_SHA,refit_scaler_sha=SCALERS["refit"],
        artifact_cache_relative_path=artifact.relative_to(context["private_scope"]).as_posix(),artifact_sha256=sha(artifact),model_state_sha=bundle["model_state_sha"],
        source_stage_API="original_fit_base._stage",search_new_optimizer_updates=0,search_new_F_calls=0,
        search_seconds_this_resume_call=search["summary"]["seconds"],refit_seconds_this_resume_call=refit["summary"]["seconds"],
        historical_saved_search_segment_seconds=math.fsum(row.get("segment_seconds",0.) for row in search["curves"]),
        old_unsaved_interrupted_progress="UNKNOWN_NOT_LOGGED",search_total_wall_seconds_known=False,reused_complete_artifact=False,
        actual_saved_point_resume=True,full_interrupted_instant_resume=False,uninterrupted_bitwise_parity_claimed=False,post_T_development=True,new_T_target_operations=0,
        independent_K_training_completed=False,full_V_completed=False)
    new(public_run/"BASE_FIT_RECEIPT.json",receipt)
    return dict(status="COMPLETE_SAVED_POINT_REFIT_NO_V_YET",fit_receipt=receipt,full_V_completed=False)


def completed(condition,public_scope,private_scope,context):
    public_run=public_scope/"runs"/condition["run_id"]; private_run=private_scope/"runs"/condition["run_id"]
    receipt=read(public_run/"BASE_FIT_RECEIPT.json"); artifact=private_run/"selected_bundle.pt"
    if (receipt.get("status")!="COMPLETE_VALID" or receipt.get("run_id")!=condition["run_id"] or receipt.get("selected_epoch")!=3.
            or receipt.get("refit_updates")!=2262 or receipt.get("refit_exposures")!=36189 or sha(artifact)!=receipt["artifact_sha256"]
            or sha(public_run/"SOURCE_CONTRACT.json")!=condition["source_contract_sha256"]): raise ValueError("Exact completed saved-point parent required")
    return receipt,artifact


def route_spec(output,condition,receipt):
    if output["parent_run_id"]!=condition["run_id"] or output["latent_k"] not in (3,6,11) or output["split"]!="V_select": raise ValueError("One of six exact trained-prefix routes required")
    return dict(run_id=output["run_id"],source_parent_run_id=condition["run_id"],dataset="beijing",family="B0",fold="primary",canonical_fit_end=12622,
        latent_k=output["latent_k"],prefix_variant=output["variant"],artifact_cache_relative_path="runs/"+condition["run_id"]+"/selected_bundle.pt",
        artifact_sha256=receipt["artifact_sha256"],foundation_sha=FOUNDATION_SHA,
        expected_prediction_model_sha=fingerprint(dict(artifact_sha=receipt["artifact_sha256"],model_state_sha=receipt["model_state_sha"],normalization=SCALERS["refit"])))


def route_for(output,condition,receipt,task,pool,context):
    import deployment
    return deployment.Route(route_spec(output,condition,receipt),task,context["private_scope"],device="cpu",foundation_pool=pool)


def full_prediction(output,condition,receipt,task,pool,context,public_folder,private_folder,deadline):
    np,p1=context["np"],context["priority"]
    route=route_for(output,condition,receipt,task,pool,context)
    try:
        arrays=p1.cohort_arrays(task,BOUNDS,route.mean,route.std,context["data"],np)
        p1.validate_cohort(arrays,context["reference"],context["evaluation"])
        before=pool.counters(); predictions=[]
        for start in range(0,43,4):
            p1.check_deadline(deadline); origins=arrays["origins"][start:start+4]
            physical=np.stack([task["physical"][int(o)-L:int(o)] for o in origins]); observed=np.stack([task["observed"][int(o)-L:int(o)] for o in origins])
            raw=route.predict_physical(physical,observed)
            predictions.append(((raw-route.mean.astype(np.float64))/route.std.astype(np.float64)).astype(np.float32))
        arrays["pred_normalized"]=np.concatenate(predictions)
        delta={key:pool.counters()[key]-value for key,value in before.items()}
        if arrays["pred_normalized"].shape!=(43,H,C) or delta!=dict(streams=43*output["latent_k"],row_calls=11,physical_streams=43*output["latent_k"],physical_calls=11): raise ValueError("Actual full V43 and K-row original Route inference required")
    finally: route.close()
    public_folder.mkdir(parents=True,exist_ok=False); private_folder.mkdir(parents=True,exist_ok=False)
    identity=dict(run_id=output["run_id"],split="V_select",cohort="beijing:primary:V_select:stride48",data_sha=DATA_SHA,scaler_sha=SCALERS["refit"],
        source_sha=fingerprint(context["new_sources"]),config_sha=fingerprint(route_spec(output,condition,receipt)),
        model_sha=fingerprint(dict(artifact_sha=receipt["artifact_sha256"],model_state_sha=receipt["model_state_sha"],latent_k=output["latent_k"],variant=output["variant"])))
    new(public_folder/"REFERENCE_CONTRACT.json",dict(**identity,arrays={key:context["evaluation"].array_fingerprint(value) for key,value in arrays.items() if key!="pred_normalized"}))
    archive=private_folder/"V_select.npz"
    with archive.open("xb") as stream: np.savez_compressed(stream,**arrays)
    score,statistics=context["evaluation"].error_statistics(arrays["pred_normalized"],arrays["target_normalized"],arrays["observed"],arrays["origins"],output["run_id"],"V_select",identity["cohort"],arrays["channel_ids"])
    with gzip.open(public_folder/"PUBLIC_ERROR_STATS.csv.gz","xt",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(statistics[0])); writer.writeheader(); writer.writerows(statistics)
    p1.csv_new(public_folder/"SCORES.csv",[score])
    manifest=dict(**identity,archive_sha256=sha(archive),arrays={key:context["evaluation"].array_fingerprint(value) for key,value in arrays.items()},score=score,
        artifact_sha256=receipt["artifact_sha256"],parent_run_id=condition["run_id"],family="B0",registered_role=output["kind"],variant=output["variant"],latent_k=output["latent_k"],
        actual_route_spec=route_spec(output,condition,receipt),F_counters=delta,device="cpu",seed=92601,post_T_development=True,new_T_target_operations=0,
        independent_generalization_claimed=False,full_V_checkpoint_replay_claimed=False)
    new(public_folder/"PREDICTION_MANIFEST.json",manifest)
    return dict(status="COMPLETE_FULL_V43",V_score=score,manifest_sha256=sha(public_folder/"PREDICTION_MANIFEST.json"),artifact_sha256=receipt["artifact_sha256"])


def validate_saved_identity(manifest,output,condition,receipt,new_sources):
    model_sha=fingerprint(dict(artifact_sha=receipt["artifact_sha256"],model_state_sha=receipt["model_state_sha"],latent_k=output["latent_k"],variant=output["variant"]))
    if (manifest.get("run_id")!=output["run_id"] or manifest.get("split")!="V_select" or manifest.get("cohort")!="beijing:primary:V_select:stride48"
            or manifest.get("source_sha")!=fingerprint(new_sources) or manifest.get("model_sha")!=model_sha
            or manifest.get("config_sha")!=fingerprint(route_spec(output,condition,receipt)) or manifest.get("actual_route_spec")!=route_spec(output,condition,receipt)
            or manifest.get("artifact_sha256")!=receipt["artifact_sha256"] or manifest.get("scaler_sha")!=SCALERS["refit"] or manifest.get("data_sha")!=DATA_SHA
            or manifest.get("latent_k")!=output["latent_k"] or manifest.get("variant")!=output["variant"] or manifest.get("parent_run_id")!=condition["run_id"]):
        raise ValueError("Exact selected parent/prefix/scaler/source identity required")


def saved_arrays(output,condition,receipt,public_folder,private_folder,context):
    manifest=read(public_folder/"PREDICTION_MANIFEST.json"); reference=read(public_folder/"REFERENCE_CONTRACT.json"); archive=private_folder/"V_select.npz"
    validate_saved_identity(manifest,output,condition,receipt,context["new_sources"])
    if sha(archive)!=manifest.get("archive_sha256"): raise ValueError("Saved private V archive bytes changed")
    with context["np"].load(archive,allow_pickle=False) as source:
        if set(source.files)!={"pred_normalized","target_normalized","observed","origins","channel_ids"}: raise ValueError("Exact V archive members required")
        arrays={key:source[key] for key in source.files}
    for key,value in arrays.items():
        if context["evaluation"].array_fingerprint(value)!=manifest["arrays"][key]: raise ValueError("Saved V member bytes changed")
    if arrays["pred_normalized"].shape!=(43,H,C): raise ValueError("Full registered V43/C11/H48 required")
    for item in (reference,context["reference"]): context["priority"].validate_cohort(arrays,item,context["evaluation"])
    if any(reference.get(key)!=manifest.get(key) for key in ("run_id","split","cohort","source_sha","config_sha","data_sha","model_sha","scaler_sha")):
        raise ValueError("Saved reference identity differs from manifest")
    return arrays,manifest


def rescore(output,condition,receipt,public_folder,private_folder,context,destination):
    arrays,manifest=saved_arrays(output,condition,receipt,public_folder,private_folder,context)
    with gzip.open(public_folder/"PUBLIC_ERROR_STATS.csv.gz","rt",encoding="utf-8",newline="") as stream: statistics=list(csv.DictReader(stream))
    with (public_folder/"SCORES.csv").open(encoding="utf-8",newline="") as stream: scores=list(csv.DictReader(stream))
    if len(scores)!=1: raise ValueError("Exactly one saved V score required")
    result=context["priority"].independent_rescore(arrays,manifest,statistics,scores[0],context["np"])
    result.update(run_id=output["run_id"],artifact_sha256=receipt["artifact_sha256"],manifest_sha256=sha(public_folder/"PREDICTION_MANIFEST.json"),full_V_origins=43,torch_imported="torch" in sys.modules)
    if result["torch_imported"]: raise ValueError("Independent FP64 rescore must be a separate no-Torch process")
    new(destination/"RESCORE_RECEIPT.json",result)
    return result


def replay(output,condition,receipt,arrays,manifest_sha,task,pool,context,destination,deadline):
    np,p1=context["np"],context["priority"]; indices=p1.replay_indices(43,np); origins=arrays["origins"][indices]
    route=route_for(output,condition,receipt,task,pool,context); before=pool.counters(); predictions=[]
    try:
        for start in range(0,6,4):
            p1.check_deadline(deadline); selected=origins[start:start+4]
            for index in indices[start:start+4]:
                target,mask=context["data"].target_window(task,int(arrays["origins"][index]),route.mean,route.std)
                if not np.array_equal(target,arrays["target_normalized"][index]) or not np.array_equal(mask,arrays["observed"][index]): raise ValueError("Replay original Task target/mask differs")
            physical=np.stack([task["physical"][int(o)-L:int(o)] for o in selected]); observed=np.stack([task["observed"][int(o)-L:int(o)] for o in selected])
            raw=route.predict_physical(physical,observed); predictions.append(((raw-route.mean.astype(np.float64))/route.std.astype(np.float64)).astype(np.float32))
        actual=np.concatenate(predictions); delta={key:pool.counters()[key]-value for key,value in before.items()}
        if delta!=dict(streams=6*output["latent_k"],row_calls=2,physical_streams=6*output["latent_k"],physical_calls=2): raise ValueError("Fresh CPU6 4+2 K-row accounting differs")
        passed=bool(np.allclose(actual,arrays["pred_normalized"][indices],atol=1e-5,rtol=1e-4))
        result=dict(status="PASS_FRESH_CPU6_REPLAY" if passed else "FAIL_REPLAY_PARITY",run_id=output["run_id"],executed_now=True,
            artifact_sha256=receipt["artifact_sha256"],manifest_sha256=manifest_sha,origins=origins.tolist(),microbatches=[4,2],F_counters=delta,
            max_abs=float(np.abs(actual.astype(np.float64)-arrays["pred_normalized"][indices].astype(np.float64)).max()),atol=1e-5,rtol=1e-4,
            full_V_reinference_claimed=False,new_T_target_operations=0)
    finally: route.close()
    new(destination/"REPLAY_RECEIPT.json",result)
    if not passed: raise ValueError("Fresh selected checkpoint replay failed")
    return result


def require_prior_proofs(output,receipt,manifest_sha,proofs):
    for mode,status in (("rescore","PASS_INDEPENDENT_FULL_V_FP64"),("replay","PASS_FRESH_CPU6_REPLAY")):
        path,digest=proofs.get(mode,(None,None))
        if not path or not digest or sha(path)!=digest: raise ValueError("Exact previous rescore/replay RESULT SHA required")
        value=read(path); science=value.get("scientific_result") or {}
        if (value.get("run_id")!=output["run_id"] or value.get("mode")!=mode or value.get("status")!=status or value.get("error_class") is not None
                or value.get("postflight_error_class") is not None or science.get("status")!=status or science.get("artifact_sha256")!=receipt["artifact_sha256"]
                or science.get("manifest_sha256")!=manifest_sha or (mode=="replay" and science.get("executed_now") is not True)):
            raise ValueError("Completed exact selected-prefix FP64 rescore and actual checkpoint replay required")


def cost(output,condition,receipt,manifest_sha,task,pool,context,destination,batch,deadline,quiet_path,quiet_sha,proofs):
    if batch not in (1,8) or not quiet_path or not quiet_sha or sha(quiet_path)!=quiet_sha: raise ValueError("B1/B8 exact quiet proof required")
    require_quiet(read(quiet_path),time.time()); require_prior_proofs(output,receipt,manifest_sha,proofs)
    import measure_costs
    np,p1=context["np"],context["priority"]; origins=context["data"].origins(BOUNDS["V_select"],L,H,H)[:batch]
    physical=np.stack([task["physical"][int(o)-L:int(o)] for o in origins]); observed=np.stack([task["observed"][int(o)-L:int(o)] for o in origins])
    route=route_for(output,condition,receipt,task,pool,context)
    class Lease:
        def check(self): p1.check_deadline(deadline)
    try:
        rows=measure_costs.measure_balanced([route],physical,observed,warmups=10,timed_trials=20,stage_trials=20,lease=Lease())
        if len(rows)!=50 or any((row["F_logical_streams"],row["F_physical_streams"],row["F_calls"])!=(batch*output["latent_k"],batch*output["latent_k"],1) for row in rows): raise ValueError("Actual original balanced whole-path B1/B8 protocol differs")
        memory=route.memory_state()
    finally: route.close()
    if sha(quiet_path)!=quiet_sha or any(sha(path)!=digest for path,digest in proofs.values()): raise ValueError("Cost prerequisites changed")
    trials=destination/"COST_TRIALS.csv.gz"; measure_costs._write_trials(trials,rows)
    result=dict(status="COMPLETE_PROTOCOL",run_id=output["run_id"],artifact_sha256=receipt["artifact_sha256"],manifest_sha256=manifest_sha,device="cpu",CPU_threads=2,
        batch_size=batch,warmups=10,timed_trials=20,stage_trials=20,actual_trial_rows=len(rows),summaries=measure_costs.summarize(rows),memory=memory,
        physical_input_sha256=hashlib.sha256(physical.tobytes()).hexdigest(),observation_mask_sha256=hashlib.sha256(observed.tobytes()).hexdigest(),origins=origins.tolist(),
        trials_sha256=sha(trials),quiet_window_proof_sha256=quiet_sha,cached_Z_or_forecast_reads=0,training_peak_included=False,new_T_target_operations=0,
        source_protocol="original_measure_costs.measure_balanced + original deployment.Route constructor",stage_probe_separate_intrusive_scope=True)
    new(destination/"COST_RECEIPT.json",result)
    return result


def execute(public,cache,asset,plan_sha,identifier,mode,deadline,destination,batch=None,quiet_path=None,quiet_sha=None,proofs=None):
    import psutil
    public,cache,asset=map(lambda path:relative(path,"."),(public,cache,asset))
    plan,old,libraries=metadata(public,cache,asset,plan_sha,psutil); p1=libraries["priority"]
    if mode not in ("resume","predict","rescore","replay","cost") or not isinstance(deadline,(int,float)) or not time.time()<deadline<=DEADLINE: raise ValueError("Exact mode and live original deadline required")
    if (mode=="cost" and batch not in (1,8)) or (mode!="cost" and batch is not None): raise ValueError("Only cost accepts B1/B8")
    p1.require_environment(); scope,private=relative(public,NAME),relative(cache,NAME)
    if not private.is_dir(): raise FileNotFoundError("ROOT must prepare the empty private namespace")
    output=None if mode=="resume" else next((row for row in outputs() if row["run_id"]==identifier),None)
    condition=next((row for row in conditions() if row["run_id"]==(identifier if mode=="resume" else (output or {}).get("parent_run_id"))),None)
    if condition is None: raise ValueError("One exact parent resume or six exact prefix outputs required")
    destination=relative(scope,os.path.relpath(Path(destination).absolute(),scope))
    if destination.exists() or not destination.is_relative_to(scope/"diagnostics"): raise FileExistsError("Exclusive new diagnostic path required")
    public_run,private_run=scope/"runs"/condition["run_id"],private/"runs"/condition["run_id"]
    claim=scope/"claims"/(mode+"_"+identifier+".json")
    if mode=="resume" and (public_run.exists() or private_run.exists() or claim.exists()): raise FileExistsError("Existing resume attempt preserved; no implicit restart")
    public_folder,private_folder=(None,None) if output is None else (scope/"predictions"/identifier/"V_select",private/"predictions"/identifier)
    if mode=="predict" and (public_folder.exists() or private_folder.exists()): raise FileExistsError("Existing full V output preserved; no implicit overwrite")
    libraries["seed"].resource_preflight(public,cache,plan,psutil)
    destination.mkdir(parents=True,exist_ok=False); started=time.time()
    new(destination/"STARTED.json",dict(run_id=identifier,mode=mode,plan_sha256=plan_sha,actual_pid=os.getpid(),actual_create_time=psutil.Process().create_time(),deadline_unix=deadline))
    context,pool,result,failure,post_error=None,None,None,None,None
    try:
        sys.path.insert(0,str(public))
        import numpy as np
        import data_contract as data
        import evaluation
        if Path(data.__file__).resolve()!=public/"data_contract.py": raise ValueError("Original scientific import root differs")
        context=dict(np=np,data=data,evaluation=evaluation,priority=p1,reference=read(relative(public,REFERENCE)),plan=plan,plan_sha=plan_sha,
            public=public,cache=cache,private_scope=private,new_sources=dict(adapter=plan["runner_sha256"],**DEPENDENCIES))
        if mode=="rescore":
            receipt,_=completed(condition,scope,private,context)
            result=rescore(output,condition,receipt,public_folder,private_folder,context,destination)
        else:
            import torch
            import fit_base as fit
            import training
            import deployment
            import gpu_runtime
            torch.set_num_threads(2)
            if torch.cuda.is_initialized(): raise RuntimeError("CPU-only operation initialized CUDA")
            context.update(torch=torch,fit=fit,training=training)
            task,access=safe_task(asset,old,libraries,context); new(destination/"PREFIX_ACCESS_RECEIPT.json",access)
            pool=deployment.FoundationPool(relative(asset,old["model_asset_relative_path"]),"cpu"); foundation=pool.get(H)
            if (gpu_runtime.state_hash(foundation.model)!=FOUNDATION_SHA or foundation.row_microbatch!=128 or foundation.checkpoint_inputs is not False
                    or foundation.instance_norm_source_sha256!=old["instance_norm_source_sha256"] or any(p.requires_grad or p.grad is not None for p in foundation.model.parameters())):
                raise ValueError("Actual frozen Bolt tensor/runtime identity changed")
            if mode=="resume":
                new(claim,dict(run_id=identifier,plan_sha256=plan_sha,saved_point_only=True,created_unix=time.time()))
                result=resume_and_save(condition,task,foundation,context,public_run,private_run,deadline)
            else:
                receipt,artifact=completed(condition,scope,private,context)
                bundle=torch.load(artifact,map_location="cpu",weights_only=False)
                if (bundle.get("resume_adapter_plan_sha256")!=plan_sha or bundle["spec"]!=condition["spec"] or p1.state_sha(bundle["model"],data)!=receipt["model_state_sha"]
                        or bundle["contract"]!=condition["checkpoint_contracts"]["refit"]): raise ValueError("Actual selected resumed parent tensors or source contract changed")
                if mode=="predict": result=full_prediction(output,condition,receipt,task,pool,context,public_folder,private_folder,deadline)
                else:
                    manifest=read(public_folder/"PREDICTION_MANIFEST.json")
                    validate_saved_identity(manifest,output,condition,receipt,context["new_sources"])
                    manifest_sha=sha(public_folder/"PREDICTION_MANIFEST.json")
                    if mode=="replay":
                        arrays,_=saved_arrays(output,condition,receipt,public_folder,private_folder,context)
                        result=replay(output,condition,receipt,arrays,manifest_sha,task,pool,context,destination,deadline)
                    else: result=cost(output,condition,receipt,manifest_sha,task,pool,context,destination,batch,deadline,quiet_path,quiet_sha,proofs or {})
            foundation.assert_frozen_unchanged(); result["F_final_state_sha256"]=gpu_runtime.state_hash(foundation.model)
        if result.get("status")!="BUDGET_LIMIT_PRESERVED": p1.check_deadline(deadline)
    except Exception as error:
        failure=type(error).__name__; path=private/"failures"/(mode+"_"+identifier+"_"+str(time.time_ns())+".json")
        new(path,dict(error_class=failure,error_message=str(error),traceback=traceback.format_exc()))
        result=dict(status="FAILED_PRESERVED",private_failure_cache_relative_path=path.relative_to(cache).as_posix())
    finally:
        try:
            if pool is not None: pool.close()
            metadata(public,cache,asset,plan_sha,psutil); libraries["seed"].resource_preflight(public,cache,plan,psutil)
            if context is not None and "torch" in context and context["torch"].cuda.is_initialized(): raise RuntimeError("Unexpected CUDA initialization")
        except Exception as error:
            post_error=type(error).__name__; path=private/"failures"/("post_"+mode+"_"+identifier+"_"+str(time.time_ns())+".json")
            new(path,dict(error_class=post_error,error_message=str(error),traceback=traceback.format_exc()))
        new(destination/"RESULT.json",dict(status="FAILED_PRESERVED" if failure or post_error else (result or {}).get("status","INCOMPLETE"),run_id=identifier,mode=mode,
            plan_sha256=plan_sha,scientific_result=result,error_class=failure,postflight_error_class=post_error,elapsed_wall_seconds=time.time()-started,
            original_source39_immutable9_original_and_backup_checkpoints_preserved=post_error is None,new_T_target_operations=0,
            original_full_task_identity_inherited_not_recomputed=True,independent_generalization_claimed=False,uninterrupted_bitwise_parity_claimed=False,
            ROOT_CPU2_RSS3GiB_two_model_6GiB_supervisor_and_cost_quiet_required=True,original_GPU_coverage_changed=False))
    if failure or post_error: raise RuntimeError("Saved-point adapter failed; all original evidence and new partials preserved")
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true"); parser.add_argument("--root-triggered",action="store_true")
    for key in ("public-root","cache-root","asset-root","plan-sha","run-id","output","quiet-proof","quiet-sha","rescore-proof","rescore-sha","replay-proof","replay-sha"): parser.add_argument("--"+key)
    parser.add_argument("--mode",choices=("resume","predict","rescore","replay","cost")); parser.add_argument("--deadline",type=float); parser.add_argument("--batch",type=int,choices=(1,8))
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="PREPARATION_NOT_EXECUTED",input_reads=0,writes=0,scientific_operations=0,numpy_imported="numpy" in sys.modules,torch_imported="torch" in sys.modules,
            actual_resume_fit_inference_cost_or_child_operations=0,original_parent_conditions=2,planned_trained_prefix_outputs=6,independent_K_training_completed=False)))
        return 0
    if not args.root_triggered or any(getattr(args,key) is None for key in ("public_root","cache_root","asset_root","plan_sha","run_id","mode","deadline","output")):
        parser.error("Explicit ROOT trigger, sealed PLAN SHA, three roots, one parent/prefix ID, mode, live original deadline and new output required")
    value=execute(args.public_root,args.cache_root,args.asset_root,args.plan_sha,args.run_id,args.mode,args.deadline,args.output,args.batch,args.quiet_proof,args.quiet_sha,
        dict(rescore=(args.rescore_proof,args.rescore_sha),replay=(args.replay_proof,args.replay_sha)))
    print(json.dumps(dict(status=value["status"],run_id=args.run_id,mode=args.mode)))
    return 0 if value["status"].startswith(("COMPLETE","PASS")) else 1


if __name__=="__main__": raise SystemExit(main())
