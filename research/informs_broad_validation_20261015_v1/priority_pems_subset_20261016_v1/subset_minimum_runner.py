"""Four original PeMS C32 conditions; prefix-only CPU preparation.

Default invocation opens no inputs. ROOT alone seals a new PLAN and dispatches
one explicit condition under the original external resource/quiet guards.
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


NAME = "priority_pems_subset_20261016_v1"
PRIOR = "priority_b2_validation_20261016_v1"
PRIOR_SHA = "475f983058871ff36b4112b6aa56dc252b17aea47eb6ae1744ff119c2ad5138f"
DEADLINE, L, H, C, K, PREFIX, SEED = 1792222550, 512, 12, 32, 8, 14284, 92601
BOUNDS = dict(A_fit=[0,8928], A_stop=[8928,10713], M_cal=[10713,12499], V_select=[12499,14284], T_final=[14284,17856])
DEPENDENCY_PATHS = {
    "priority": PRIOR+"/priority_runner.py",
    "primitive": PRIOR+"/safe_jena_prefix.py",
    "pems": "priority_core_followup_20261016_v1/pems_fixed_head_runner.py",
    "minima": "priority_valid_minima_20261016_v1/beijing_minimum_runner.py",
}
MODEL_REL = ".cache/huggingface/models--amazon--chronos-bolt-small/snapshots/772f3d25d38aec6d914c8949dab4462e2d46f5d8"
MODEL_BINDINGS = [
    dict(snapshot_relative_path=MODEL_REL+"/config.json", resolved_blob_relative_path=".cache/huggingface/models--amazon--chronos-bolt-small/blobs/d8f9b65552d4d211e28d332ad5666484092bbf77",sha256="9ca0ebbe049ccebacb0546361e8aa728f0cf104cd18d1838d78d296aa317bd21",bytes=1121),
    dict(snapshot_relative_path=MODEL_REL+"/model.safetensors", resolved_blob_relative_path=".cache/huggingface/blobs/61/61439ff1becff564d030cdbe1ce9dfce0673fc0ebc4ca0a009d3fe02ce60bba0",sha256="06a6a19bbe74bc10a9cd193bd4bf2bf638ae07f7e0d51653ae7ab8ea968a21dd",bytes=190888824),
]
RESOURCES = dict(CPU_threads=2,max_worker_RSS_bytes=3*1024**3,maximum_model_workers=2,total_owned_RSS_limit_bytes=6*1024**3,
    storage_limit_bytes=30*1024**3,admission_reserve_bytes=256*1024**2,GPU_seconds_allowed=0,
    ROOT_CPU_supervisor_required=True,costs_must_not_overlap_training=True)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""): digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False,ensure_ascii=False).encode("utf-8")).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")


def relative(root,name):
    root=Path(root).absolute();part=Path(name);path=root/part
    if part.is_absolute() or any(p.is_symlink() or getattr(p,"is_junction",lambda:False)() for p in (path,*path.parents)):
        raise ValueError("Explicit nonsymlink relative path required")
    root,path=root.resolve(),path.resolve()
    if not path.is_relative_to(root): raise ValueError("Path escapes declared root")
    return path


def module(path,label):
    spec=importlib.util.spec_from_file_location("pems_subset_"+label,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def select_columns(columns):
    if (len(columns)!=170 or len(set(columns))!=170 or any(not isinstance(x,str) or not x for x in columns)):
        raise ValueError("Exactly170 unique metadata sensor IDs required")
    order=sorted(range(170),key=lambda i:(hashlib.sha256(columns[i].encode("utf-8")).digest(),columns[i].encode("utf-8")))
    return dict(schema="UNSALTED_METADATA_ID_SHA256_ASCENDING_v1",salt="",encoding="UTF-8",
        hash_input="ID bytes only; no salt, newline, values, mask, time, performance or learning seed",
        ordering="(SHA256 digest bytes,ID UTF-8 bytes) ascending; output in this hash order",
        full_order_indices=order,full_order_ids=[columns[i] for i in order],
        selected_indices=order[:C],selected_ids=[columns[i] for i in order[:C]],
        nested_prefix_sizes_defined_only=[32,64,128],only_C32_executable=True,
        decision_provenance="ROOT_IMPLEMENTATION_DECISION_FOR_THIS_FOLLOWUP_20261016",
        previously_sealed_or_historical_selection_claimed=False)


def subset_identity_payload(parent_sha,selection,hashes):
    return dict(schema="PEMS_PRE_T_SUBSET_TASK_IDENTITY_v1",parent_full_C170_data_sha256_inherited=parent_sha,
        parent_identity_recomputed=False,column_selection=copy.deepcopy(selection),
        actual_subset_prefix_array_sha256=copy.deepcopy(hashes),actual_C=C,materialized_prefix_rows=PREFIX,
        original_split_ranges=copy.deepcopy(BOUNDS),L=L,H=H,interval_seconds=300,
        mask_rule="isfinite(feature0.astype(float32)); finite zero observed; nonfinite physical stored NaN",
        full_subset_T_identity_computed=False,new_T_numeric_values_parsed=0)


def conditions():
    rows=[]
    for family in ("RAW","DIRECT_G","B0","B2"):
        registered_k=K if family in ("B0","B2") else 0
        middle=f"pems08_E1_SUBSET_{family}_k{registered_k}_s92601_C32"
        spec=dict(run_id="postT_cpu_"+middle+"_priority_v1",dataset="pems08",family=family,latent_k=K if registered_k else 0,
            seed=SEED,fold="primary",epochs=[] if family=="RAW" else [.25,1.,3.],lr=.001,device="cpu",microbatch=4,effective_batch=16,
            F_row_microbatch=128,checkpoint_inputs=False,lora_rank=8,lora_alpha=16,stage_label="postT_prefix_only_C32_original_minimum")
        rows.append(dict(run_id=spec["run_id"],family=family,registered_latent_k=registered_k,actual_channels=C,
            upstream_registered_id="primary_"+middle,spec=spec,new_training_condition=family!="RAW"))
    return rows


def draft_plan(columns,contract_sha,upstream,dependencies):
    return dict(schema="PRIORITY_PEMS_SUBSET_MINIMA_v1",namespace=NAME,status="DRAFT_NOT_EXECUTABLE",runner_sha256=None,
        stage="POST_T_V_ONLY_DEVELOPMENT",upstream_priority_plan_sha256=PRIOR_SHA,
        data_contract_public_sha256=contract_sha,parent_task_contract=copy.deepcopy(upstream["parent_task_contract"]),
        source39=copy.deepcopy(upstream["source39"]),immutable_public=copy.deepcopy(upstream["immutable_public"]),
        dependencies=copy.deepcopy(dependencies),model_snapshot_bindings=copy.deepcopy(MODEL_BINDINGS),
        foundation_tensor_sha256=upstream["foundation_tensor_sha256"],model_asset_relative_path=upstream["model_asset_relative_path"],
        model_revision=upstream["model_revision"],instance_norm_source_relative_path=upstream["instance_norm_source_relative_path"],
        instance_norm_source_sha256=upstream["instance_norm_source_sha256"],
        column_selection=select_columns(columns),conditions=conditions(),maximum_new_training_conditions=3,
        original_primary_split_ranges=copy.deepcopy(BOUNDS),materialized_prefix_rows=PREFIX,
        actual_subset_identity="Derived at execution from inherited parent identity+new explicit selection+actual subset prefix array SHA+bounds/L/H/mask rule; never full-C identity reuse",
        resources=copy.deepcopy(RESOURCES),deadline_unix=DEADLINE,protected_tail_numeric_parse_allowed=False,new_T_target_operations_allowed=False,
        fullC_reference_scope="Old C170 selected1epoch and new C32 search3epoch differ in training; not a C-only causal contrast",
        raw_policy="Accuracy and cost row128; original full-versus128 physical-output parity per batch B1/B8 is mandatory for Raw cost only; original historical Raw failures are preserved",
        epoch_accounting=dict(search_train_origins=8405,search_exposures=25215,search_updates=1577,
            explanation="ceil(segment exposure differences/16) summed at .25/1/3; not ceil(25215/16)=1576",
            refit_train_origins=10190,refit_updates_by_selected_epoch={"0":0,"0.25":160,"1":637,"3":1911}),
        actual_training_prediction_replay_cost_or_child_operations=0,ROOT_seal_required=True,
        external_guard_and_fresh_quiet_required=True,optimizer_resume_or_old_queue_restart_allowed=False)


def require_sealed(plan):
    if plan.get("status")!="SEALED_BY_ROOT" or plan.get("runner_sha256")!=sha(__file__):
        raise ValueError("Exact new ROOT-sealed PLAN and runner SHA required")


def require_fresh(public_scope,private_scope,identifier):
    for root in (Path(public_scope),Path(private_scope)):
        if any((root/kind/identifier).exists() for kind in ("runs","predictions")) or (root/"claims"/("fit_"+identifier+".json")).exists():
            raise FileExistsError("Existing attempt preserved; no resume, rename or implicit overwrite")


def metadata(public,cache,asset,plan_sha,identifier,process_api):
    public,cache,asset=map(lambda p:relative(p,"."),(public,cache,asset))
    path=relative(public,NAME+"/PLAN.json")
    if not plan_sha or sha(path)!=plan_sha: raise ValueError("Explicit new PLAN SHA required")
    plan=read(path);require_sealed(plan)
    if (plan.get("schema")!="PRIORITY_PEMS_SUBSET_MINIMA_v1" or plan.get("namespace")!=NAME or plan.get("stage")!="POST_T_V_ONLY_DEVELOPMENT"
            or plan.get("conditions")!=conditions() or plan.get("maximum_new_training_conditions")!=3
            or plan.get("resources")!=RESOURCES or plan.get("deadline_unix")!=DEADLINE
            or plan.get("original_primary_split_ranges")!=BOUNDS or plan.get("materialized_prefix_rows")!=PREFIX
            or plan.get("protected_tail_numeric_parse_allowed") is not False or plan.get("new_T_target_operations_allowed") is not False
            or plan.get("model_snapshot_bindings")!=MODEL_BINDINGS or set(plan.get("dependencies",{}))!=set(DEPENDENCY_PATHS.values())):
        raise ValueError("Only the exact original four C32 recipes/CPU resources/prefix contract permitted")
    for name,digest in plan["dependencies"].items():
        if sha(relative(public,name))!=digest: raise ValueError("Reviewed dependency changed")
    libraries={key:module(relative(public,name),key) for key,name in DEPENDENCY_PATHS.items()}
    p1=libraries["priority"].metadata(public,cache,PRIOR_SHA,process_api)
    if plan["source39"]!=p1["source39"] or plan["immutable_public"]!=p1["immutable_public"]: raise ValueError("Original source39/immutable9 differ")
    contract_path=relative(public,"DATA_CONTRACT.json")
    if sha(contract_path)!=plan["data_contract_public_sha256"]: raise ValueError("Original data contract changed")
    parent=read(contract_path)["assets"]["pems08"]
    if (parent!=plan["parent_task_contract"] or parent["shape"]!=[17856,170] or parent["L"]!=L or parent["H"]!=H
            or parent["primary_split_ranges"]!=BOUNDS or plan["column_selection"]!=select_columns(parent["columns"])):
        raise ValueError("Parent metadata/new channel rule differs")
    with relative(public,"EXPERIMENT_REGISTRY.csv").open(encoding="utf-8-sig",newline="") as stream:
        registered={row["experiment_id"]:row for row in csv.DictReader(stream)}
    for row in conditions():
        old=registered[row["upstream_registered_id"]]
        if (old["family"]!="SUBSET_"+row["family"] or old["latent_k"]!=str(row["registered_latent_k"])
                or old["num_channels"]!="170" or old["length"]!="512" or old["horizon"]!="12" or old["seed"]!="92601"):
            raise ValueError("Original registered subset condition differs")
    for name in ("foundation_tensor_sha256","model_asset_relative_path","model_revision","instance_norm_source_relative_path","instance_norm_source_sha256"):
        if plan[name]!=p1[name]: raise ValueError("Original F/norm identity differs")
    libraries["minima"].check_model_snapshot(asset,plan["model_asset_relative_path"],plan["model_snapshot_bindings"])
    for root,name in ((asset,plan["instance_norm_source_relative_path"]),(public.parent,"temporal_frontend_campaign_20261015_v1/runtime.py")):
        if sha(relative(root,name))!=plan["instance_norm_source_sha256"]: raise ValueError("Actual runtime normalization changed")
    match=[row for row in conditions() if row["run_id"]==identifier]
    if len(match)!=1: raise ValueError("One exact ROOT-selected condition required")
    libraries["priority"].check_deadline(DEADLINE)
    return plan,match[0],libraries


def resource_preflight(public,cache,process_api):
    used=sum(path.stat().st_size for root in (public,cache) for path in Path(root).rglob("*") if path.is_file())
    reserve=RESOURCES["admission_reserve_bytes"]
    if used+reserve>RESOURCES["storage_limit_bytes"] or shutil.disk_usage(cache).free<reserve: raise RuntimeError("Original storage/reserve limit reached")
    if process_api.Process().memory_info().rss>RESOURCES["max_worker_RSS_bytes"]: raise MemoryError("Own RSS exceeds3GiB")
    return used


def subset_task(parent_task,plan,context):
    np,data=context["np"],context["data"]
    selection=plan["column_selection"];indices=selection["selected_indices"]
    if tuple(parent_task["physical"].shape)!=(PREFIX,170) or tuple(parent_task["observed"].shape)!=(PREFIX,170):
        raise ValueError("Exact legal parent prefix required")
    task=dict(parent_task,physical=parent_task["physical"][:,indices].copy(),observed=parent_task["observed"][:,indices].copy(),
        columns=copy.deepcopy(selection["selected_ids"]),split_ranges=copy.deepcopy(BOUNDS))
    hashes={key:data.array_sha256(value) for key,value in (("physical",task["physical"]),("observed",task["observed"]),
        ("times",task["times"]),("columns",np.asarray(task["columns"],dtype=str)))}
    identity=subset_identity_payload(parent_task["data_sha256"],selection,hashes)
    task.update(data_sha256=fingerprint(identity),array_sha256=hashes,materialized_prefix_array_sha256=hashes,
        materialized_prefix_sha256=fingerprint(hashes),data_identity_provenance="ACTUAL_PRE_T_C32_IDENTITY_DERIVED_NOT_FULL_SUBSET",
        parent_full_C170_data_sha256_inherited=parent_task["data_sha256"],subset_identity_payload=identity)
    task["metadata"]=dict(parent_task["metadata"],source_channels=170,actual_channels=C,
        new_subset_rule_decision=True,subset_full_T_identity_not_computed=True,column_selection=selection)
    return task


def fit_stages(fit,spec,task,foundation,contract,public_run,private_run,deadline):
    search=fit._stage(spec,task,8928,BOUNDS["A_stop"],[.25,1.,3.],private_run/"search",public_run/"search",foundation,deadline,contract)
    if search["summary"]["status"]!="SEGMENT_COMPLETE": return dict(status="BUDGET_LIMIT_PRESERVED",phase="search",summary=search["summary"])
    epoch=search["summary"]["selected_epoch"]
    if epoch not in (0.,.25,1.,3.): raise ValueError("Undeclared A_stop-selected epoch")
    refit=fit._stage(spec,task,10713,None,[epoch] if epoch else [],private_run/"refit",public_run/"refit",foundation,deadline,contract)
    if refit["summary"]["status"]!="SEGMENT_COMPLETE": return dict(status="BUDGET_LIMIT_PRESERVED",phase="refit",summary=refit["summary"])
    return dict(status="COMPLETE_STAGES",search=search,refit=refit,selected_epoch=epoch)


def validate_stages(search,refit):
    epoch=search.get("selected_epoch")
    if epoch not in (0.,.25,1.,3.): raise ValueError("Original selected epoch required")
    exposures=math.ceil(epoch*10190)
    expected_search=dict(status="SEGMENT_COMPLETE",completed_epochs=[0.,.25,1.,3.],train_origins=8405,exposures=25215,updates=1577)
    expected_refit=dict(status="SEGMENT_COMPLETE",completed_epochs=[0.]+([epoch] if epoch else []),train_origins=10190,exposures=exposures,updates=math.ceil(exposures/16))
    if any(search.get(k)!=v for k,v in expected_search.items()) or any(refit.get(k)!=v for k,v in expected_refit.items()):
        raise ValueError("Exact segmented original search/fresh refit exposure and updates required")


def expected_counts(family,batches):
    width={"RAW":32,"DIRECT_G":0,"B0":8,"B2":8}[family]
    streams=sum(batches)*width
    return dict(streams=streams,physical_streams=streams,physical_calls=sum(math.ceil(batch*width/128) for batch in batches))


def canonical_route(condition,receipt,foundation_sha):
    result=dict(run_id=condition["run_id"],family=condition["family"],canonical_fit_end=10713,device="cpu",
        latent_k=32 if condition["family"]=="RAW" else (8 if condition["family"] in ("B0","B2") else 0),row_microbatch=128)
    if condition["family"]=="RAW": result["foundation_sha"]=foundation_sha
    else: result.update(artifact_cache_relative_path="runs/"+condition["run_id"]+"/selected_bundle.pt",artifact_sha256=receipt["artifact_sha256"])
    return result


def make_route(condition,receipt,task,pool,context):
    return context["deployment"].Route(canonical_route(condition,receipt,context["foundation_sha"]),task,context["private_scope"],
        model_path=context["model_path"],device="cpu",foundation_pool=pool)


def delta_counters(pool,before):
    now=pool.counters();return {name:now[name]-before[name] for name in ("streams","physical_streams","physical_calls")}


def physical_batch(task,origins,np):
    return (np.stack([task["physical"][int(o)-L:int(o)] for o in origins]),np.stack([task["observed"][int(o)-L:int(o)] for o in origins]))


def observe_backward(foundation,torch):
    proof=dict(trainable_F_input_tensors=0,F_input_gradient_callbacks=0,finite_callbacks=0,nonzero_callbacks=0,
        F_parameter_gradients_allowed=False,first_update_all_gradients_nonzero_required=False)
    if foundation is None:return proof,None
    def before(module,args,kwargs):
        value=kwargs.get("context",args[0] if args else None)
        if value is not None and torch.is_grad_enabled() and value.requires_grad:
            proof["trainable_F_input_tensors"]+=1
            def gradient(grad):
                proof["F_input_gradient_callbacks"]+=1
                if bool(torch.isfinite(grad).all()):proof["finite_callbacks"]+=1
                else:raise ValueError("Nonfinite gradient through frozen F input")
                if bool(torch.any(grad!=0)):proof["nonzero_callbacks"]+=1
                return grad
            value.register_hook(gradient)
    return proof,foundation.model.register_forward_pre_hook(before,with_kwargs=True)


def fit_condition(condition,task,foundation,context,public_run,private_run,deadline):
    fit,p1=context["fit"],context["priority"];spec=condition["spec"];started=time.perf_counter()
    source,files=fit._source_contract()
    contract=dict(source_sha=source,config_sha=fingerprint(dict(spec=spec,subset_identity=task["subset_identity_payload"])),
        data_sha=task["data_sha256"],model_sha=fit._model_sha(foundation))
    new(public_run/"SOURCE_CONTRACT.json",dict(**contract,files=files,spec=spec,plan_sha256=context["plan_sha"],
        subset_identity_payload=task["subset_identity_payload"],original_registered_id=condition["upstream_registered_id"],prefix_only_new_entry=True))
    if condition["family"]=="RAW":
        mean,std,count=context["data"].fit_statistics(task,10713)
        norm=dict(mean=mean,std=std,observed_counts=count,fit_end=10713,train_end=10713,basis_initialization="NOT_APPLICABLE_RAW",basis=None,pca_center=None)
        raw_contract=dict(**contract,canonical_fit_end=10713,scaler_sha=fit.scaler_fingerprint(mean,std),foundation_sha256=context["foundation_sha"],
            plan_sha256=context["plan_sha"],original_registered_id=condition["upstream_registered_id"],optimizer_updates=0)
        artifact=public_run/"RAW_CONTRACT.json";new(artifact,raw_contract)
        receipt=dict(status="COMPLETE_VALID",run_id=spec["run_id"],family="RAW",selected_epoch=None,search_updates=0,refit_updates=0,
            search_exposures=0,refit_exposures=0,training_status="NO_TRAINING",artifact_sha256=sha(artifact),model_state_sha=context["foundation_sha"],
            artifact_kind="RAW_METADATA_IDENTITY_NO_TRAINED_ADAPTER",trained_adapter_checkpoint_exists=False)
    else:
        stage=fit_stages(fit,spec,task,foundation,contract,public_run,private_run,deadline)
        if stage["status"]!="COMPLETE_STAGES": return stage
        search,refit,epoch=stage["search"],stage["refit"],stage["selected_epoch"]
        validate_stages(search["summary"],refit["summary"])
        backward=context["F_backward_observation"]
        if condition["family"] in ("B0","B2") and (backward["finite_callbacks"]<1 or backward["finite_callbacks"]!=backward["F_input_gradient_callbacks"]):
            raise ValueError("Actual finite backward through frozen F input not observed")
        for label,value in (("search",search),("refit",refit)): p1.write_stage(public_run/label,value)
        norm=refit["normalization"]
        bundle=dict(model=refit["model"],normalization=norm,spec=spec,contract=refit["summary"]["contract"],
            selected_search_epoch=epoch,search_summary=search["summary"],refit_summary=refit["summary"],
            model_state_sha=p1.state_sha(refit["model"],context["data"]),plan_sha256=context["plan_sha"],
            subset_identity_payload=task["subset_identity_payload"])
        artifact=private_run/"selected_bundle.pt";p1.save_bundle(artifact,bundle,context["torch"])
        receipt=dict(status="COMPLETE_VALID",run_id=spec["run_id"],family=spec["family"],selected_epoch=epoch,
            search_updates=search["summary"]["updates"],refit_updates=refit["summary"]["updates"],
            search_exposures=search["summary"]["exposures"],refit_exposures=refit["summary"]["exposures"],
            training_status="UNDERTRAINED_OR_BUDGET_LIMIT" if epoch==3 else "SCREEN_COMPLETE_CONVERGENCE_UNRESOLVED",
            artifact_sha256=sha(artifact),model_state_sha=bundle["model_state_sha"],source_stage_API="original_fit_base._stage",artifact_kind="TRAINED_SUBSET_BUNDLE")
        receipt["initialization_by_stage"]={}
        for label,normalization in (("A50",search["normalization"]),("A60",norm)):
            receipt["initialization_by_stage"][label]=dict(basis_initialization=normalization["basis_initialization"],
                fingerprints={key:None if normalization[key] is None else context["data"].array_sha256(normalization[key])
                    for key in ("mean","std","observed_counts","basis","pca_center")})
    receipt.update(seed=SEED,actual_C=C,L=L,H=H,registered_latent_k=condition["registered_latent_k"],source_sha=source,
        data_sha256=task["data_sha256"],refit_scaler_sha=fit.scaler_fingerprint(norm["mean"],norm["std"]),plan_sha256=context["plan_sha"],
        actual_subset_prefix_sha256=task["materialized_prefix_sha256"],subset_identity_payload=task["subset_identity_payload"],
        A50_end=8928,A60_end=10713,full_parent_C=170,full_parent_identity_inherited=True,new_T_numeric_values_parsed=0,
        independent_generalization_claimed=False,old_C170_training_budget_matched=False,
        fit_component_seconds_including_stage_model_initialization=time.perf_counter()-started,
        PCA_fit_time_separately_recorded=False,F_backward_observation=copy.deepcopy(context["F_backward_observation"]))
    new(public_run/"BASE_FIT_RECEIPT.json",receipt)
    return dict(status="COMPLETE_VALID",receipt=receipt,normalization=norm,artifact=artifact)


def cohort_arrays(task,norm,context):
    np,data=context["np"],context["data"]
    origins=data.origins(BOUNDS["V_select"],L,H,H)
    if len(origins)!=148 or int(origins[-1])+H>PREFIX: raise ValueError("Exact complete V148 required")
    values=[data.target_window(task,int(o),norm["mean"],norm["std"]) for o in origins]
    return dict(target_normalized=np.stack([v[0] for v in values]),observed=np.stack([v[1] for v in values]),
        origins=origins,channel_ids=np.asarray(task["columns"],dtype=str))


def full_prediction(condition,fitted,task,pool,context,public_folder,private_folder,deadline):
    np,p1=context["np"],context["priority"]
    norm=fitted["normalization"];receipt=fitted["receipt"];arrays=cohort_arrays(task,norm,context)
    public_folder.mkdir(parents=True,exist_ok=False);private_folder.mkdir(parents=True,exist_ok=False)
    canonical=canonical_route(condition,receipt,context["foundation_sha"])
    identity=dict(run_id=condition["run_id"],split="V_select",cohort="pems08:C32_hash:primary:V_select:stride12",data_sha=task["data_sha256"],
        scaler_sha=receipt["refit_scaler_sha"],source_sha=fingerprint(context["new_sources"]),config_sha=fingerprint(condition["spec"]),
        model_sha=fingerprint(dict(artifact_sha=receipt["artifact_sha256"],model_state_sha=receipt["model_state_sha"])))
    new(public_folder/"REFERENCE_CONTRACT.json",dict(**identity,arrays={n:context["evaluation"].array_fingerprint(v) for n,v in arrays.items()},
        canonical_route=canonical,canonical_route_sha256=fingerprint(canonical),subset_identity_payload=task["subset_identity_payload"]))
    route=make_route(condition,receipt,task,pool,context);before=pool.counters();predicted=[]
    try:
        for start in range(0,len(arrays["origins"]),4):
            p1.check_deadline(deadline);physical,observed=physical_batch(task,arrays["origins"][start:start+4],np)
            actual=route.predict_physical(physical,observed)
            predicted.append(((actual-route.mean.astype(np.float64))/route.std.astype(np.float64)).astype(np.float32))
        delta=delta_counters(pool,before)
        if delta!=expected_counts(condition["family"],[4]*37): raise ValueError("Actual full V148 stream/call geometry differs")
    finally: route.close()
    arrays["pred_normalized"]=np.concatenate(predicted)
    if arrays["pred_normalized"].shape!=(148,H,C): raise ValueError("Actual complete C32/H12/V148 output required")
    archive=private_folder/"V_select.npz"
    with archive.open("xb") as stream:np.savez_compressed(stream,**arrays)
    score,rows=context["evaluation"].error_statistics(arrays["pred_normalized"],arrays["target_normalized"],arrays["observed"],arrays["origins"],
        condition["run_id"],"V_select",identity["cohort"],arrays["channel_ids"])
    with gzip.open(public_folder/"PUBLIC_ERROR_STATS.csv.gz","xt",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    p1.csv_new(public_folder/"SCORES.csv",[score])
    manifest=dict(**identity,archive_sha256=sha(archive),arrays={n:context["evaluation"].array_fingerprint(v) for n,v in arrays.items()},
        score=score,device="cpu",family=condition["family"],seed=SEED,actual_C=C,H=H,K=condition["registered_latent_k"],
        artifact_sha256=receipt["artifact_sha256"],canonical_route_sha256=fingerprint(canonical),F_counters=delta,
        prefix_rows=PREFIX,new_T_numeric_values_parsed=0,old_fullC_training_budget_matched=False,independent_generalization_claimed=False)
    new(public_folder/"PREDICTION_MANIFEST.json",manifest);return manifest


def completed(scope,private,condition,context):
    rid=condition["run_id"];receipt=read(scope/"runs"/rid/"BASE_FIT_RECEIPT.json");source=read(scope/"runs"/rid/"SOURCE_CONTRACT.json")
    artifact=scope/"runs"/rid/"RAW_CONTRACT.json" if condition["family"]=="RAW" else private/"runs"/rid/"selected_bundle.pt"
    if (receipt.get("status")!="COMPLETE_VALID" or receipt.get("run_id")!=rid or receipt.get("family")!=condition["family"]
            or receipt.get("seed")!=SEED or receipt.get("actual_C")!=C or receipt.get("plan_sha256")!=context["plan_sha"]
            or source.get("spec")!=condition["spec"] or source.get("plan_sha256")!=context["plan_sha"] or sha(artifact)!=receipt["artifact_sha256"]
            or receipt.get("data_sha256")!=fingerprint(receipt["subset_identity_payload"])
            or source.get("subset_identity_payload")!=receipt["subset_identity_payload"]
            or receipt["subset_identity_payload"]["column_selection"]!=context["plan"]["column_selection"]):
        raise ValueError("Exact new subset artifact/recipe/derived task identity required")
    return receipt,artifact


def saved_metadata(public_folder,condition,receipt,context):
    manifest=read(public_folder/"PREDICTION_MANIFEST.json");reference=read(public_folder/"REFERENCE_CONTRACT.json")
    expected=dict(run_id=condition["run_id"],split="V_select",cohort="pems08:C32_hash:primary:V_select:stride12",data_sha=receipt["data_sha256"],
        scaler_sha=receipt["refit_scaler_sha"],artifact_sha256=receipt["artifact_sha256"],config_sha=fingerprint(condition["spec"]),
        source_sha=fingerprint(context["new_sources"]),canonical_route_sha256=fingerprint(canonical_route(condition,receipt,context["foundation_sha"])))
    if any(manifest.get(k)!=v for k,v in expected.items()): raise ValueError("Saved accuracy does not bind the exact subset route")
    if any(reference.get(k)!=manifest.get(k) for k in ("run_id","split","cohort","data_sha","scaler_sha","source_sha","config_sha","model_sha","canonical_route_sha256")):
        raise ValueError("Pre-inference reference and final accuracy identity differ")
    return manifest,reference


def saved_arrays(public_folder,private_folder,condition,receipt,context):
    manifest,reference=saved_metadata(public_folder,condition,receipt,context);np=context["np"]
    archive=private_folder/"V_select.npz"
    if sha(archive)!=manifest["archive_sha256"]:raise ValueError("Saved V archive changed")
    with np.load(archive,allow_pickle=False) as stored:
        if set(stored.files)!={"pred_normalized","target_normalized","observed","origins","channel_ids"}:raise ValueError("Only exact V members allowed")
        arrays={key:stored[key] for key in stored.files}
    for name,value in arrays.items():
        if context["evaluation"].array_fingerprint(value)!=manifest["arrays"][name]:raise ValueError("Saved V array fingerprint differs")
    if arrays["pred_normalized"].shape!=(148,H,C) or arrays["channel_ids"].tolist()!=context["plan"]["column_selection"]["selected_ids"]:
        raise ValueError("Exact C32/H12/V148 and hash-ordered columns required")
    if arrays["origins"].tolist()!=list(range(12499,14284-H+1,H)):raise ValueError("Original V stride12 cohort differs")
    context["priority"].validate_cohort(arrays,reference,context["evaluation"])
    return arrays,manifest


def result_identity(condition,receipt,manifest_sha,context):
    return dict(run_id=condition["run_id"],artifact_sha256=receipt["artifact_sha256"],manifest_sha256=manifest_sha,
        data_sha256=receipt["data_sha256"],canonical_route_sha256=fingerprint(canonical_route(condition,receipt,context["foundation_sha"])))


def rescore(condition,receipt,context,public_folder,private_folder,output):
    arrays,manifest=saved_arrays(public_folder,private_folder,condition,receipt,context)
    with (public_folder/"SCORES.csv").open(encoding="utf-8",newline="") as stream:scores=list(csv.DictReader(stream))
    if len(scores)!=1:raise ValueError("One exact full-V score required")
    with gzip.open(public_folder/"PUBLIC_ERROR_STATS.csv.gz","rt",encoding="utf-8",newline="") as stream:rows=list(csv.DictReader(stream))
    result=context["priority"].independent_rescore(arrays,manifest,rows,scores[0],context["np"])
    result.update(**result_identity(condition,receipt,sha(public_folder/"PREDICTION_MANIFEST.json"),context),full_V_origins=148,actual_C=C,H=H,
        new_T_numeric_values_parsed=0,torch_imported="torch" in sys.modules)
    if result["torch_imported"]:raise RuntimeError("Separate rescore must not import Torch")
    new(output/"RESCORE_RECEIPT.json",result);return result


def replay(condition,receipt,arrays,task,pool,context,output,deadline,manifest_sha):
    np,p1=context["np"],context["priority"];indices=p1.replay_indices(148,np);origins=arrays["origins"][indices]
    route=make_route(condition,receipt,task,pool,context);before=pool.counters();predicted=[]
    try:
        for start in range(0,6,4):
            p1.check_deadline(deadline);physical,observed=physical_batch(task,origins[start:start+4],np);value=route.predict_physical(physical,observed)
            predicted.append(((value-route.mean.astype(np.float64))/route.std.astype(np.float64)).astype(np.float32))
        actual=np.concatenate(predicted);delta=delta_counters(pool,before)
        if delta!=expected_counts(condition["family"],[4,2]):raise ValueError("Actual CPU6 4+2 F counters differ")
        passed=bool(np.allclose(actual,arrays["pred_normalized"][indices],atol=1e-5,rtol=1e-4))
        result=dict(status="PASS_FRESH_CPU6_REPLAY" if passed else "FAIL_REPLAY_PARITY",executed_now=True,
            **result_identity(condition,receipt,manifest_sha,context),origins=origins.tolist(),microbatches=[4,2],replay_selection_seed=92601,
            F_counters=delta,atol=1e-5,rtol=1e-4,max_abs=float(np.abs(actual.astype(np.float64)-arrays["pred_normalized"][indices].astype(np.float64)).max()),
            full_V_reinference_claimed=False,new_T_numeric_values_parsed=0)
    finally:route.close()
    new(output/"REPLAY_RECEIPT.json",result)
    if not passed:raise ValueError("Actual selected checkpoint replay failed")
    return result


def validate_cost_proofs(family,identity,values,batch):
    expected=[("rescore","PASS_INDEPENDENT_FULL_V_FP64"),("replay","PASS_FRESH_CPU6_REPLAY")]
    if family=="RAW":expected.append(("raw_parity","PASS_RAW_ROW128_VS_FULL"))
    for mode,status in expected:
        value=values.get(mode,{}) or {};science=value.get("scientific_result") or {}
        if (value.get("run_id")!=identity["run_id"] or value.get("mode")!=mode or value.get("status")!=status
                or value.get("error_class") is not None or value.get("postflight_error_class") is not None
                or value.get("original_source39_immutable9_preserved") is not True or science.get("status")!=status
                or any(science.get(k)!=v for k,v in identity.items())):raise ValueError("Exact successful per-route validation RESULT required")
        if mode=="replay" and science.get("executed_now") is not True:raise ValueError("Actual fresh CPU6 required")
        if mode=="raw_parity":
            variants=[dict(row_microbatch="full",F_counters=dict(streams=batch*C,physical_streams=batch*C,physical_calls=1)),
                dict(row_microbatch=128,F_counters=expected_counts("RAW",[batch]))]
            if (science.get("all_passed") is not True or science.get("batch_size")!=batch or science.get("actual_raw_forward") is not True
                    or science.get("per_variant_F_counters")!=variants):
                raise ValueError("This Raw batch's actual full-vs128 outputs/counters must pass")


def observe_raw_parity(route,pool,measure,physical,observed,lease):
    variants=[]
    class Observation:
        def __getattr__(self,name):return getattr(route,name)
        def predict_physical(self,values,mask):
            before=pool.counters();predicted=route.predict_physical(values,mask)
            variants.append(dict(row_microbatch=route.spec["row_microbatch"],F_counters=delta_counters(pool,before)))
            return predicted
    proof=measure.raw_row_parity(Observation(),physical,observed,candidates=("full",128),atol=1e-5,rtol=1e-4,lease=lease)
    batch=len(physical)
    expected=[dict(streams=batch*C,physical_streams=batch*C,physical_calls=1),expected_counts("RAW",[batch])]
    if len(variants)!=2 or [v["row_microbatch"] for v in variants]!=["full",128] or [v["F_counters"] for v in variants]!=expected:
        raise ValueError("Each original Raw policy must have its actual expected stream/call geometry")
    return proof,variants


def raw_parity(condition,receipt,task,pool,context,output,batch,deadline,manifest_sha):
    if condition["family"]!="RAW" or batch not in (1,8):raise ValueError("Only Raw B1/B8 actual row parity supported")
    import measure_costs
    origins=context["data"].origins(BOUNDS["V_select"],L,H,H)[:batch];physical,observed=physical_batch(task,origins,context["np"])
    route=make_route(condition,receipt,task,pool,context);before=pool.counters()
    class Lease:
        def check(self):context["priority"].check_deadline(deadline)
    try:
        proof,variants=observe_raw_parity(route,pool,measure_costs,physical,observed,Lease())
        passed=proof.get("all_passed") is True and [row["row_microbatch"] for row in proof["variants"]]==["full",128]
        counters=delta_counters(pool,before)
        if counters!=dict(streams=batch*C*2,physical_streams=batch*C*2,physical_calls=1+math.ceil(batch*C/128)):
            raise ValueError("Actual full/128 Raw parity F counters differ")
        result=dict(status="PASS_RAW_ROW128_VS_FULL" if passed else "FAIL_RAW_ROW_PARITY",**result_identity(condition,receipt,manifest_sha,context),
            batch_size=batch,all_passed=passed,original_raw_row_parity=proof,actual_raw_forward=True,actual_optimizer_updates=0,
            F_counters=counters,per_variant_F_counters=variants,historical_original_Raw_failures_resolved=False,new_T_numeric_values_parsed=0)
    finally:route.close()
    new(output/"RAW_PARITY_RECEIPT.json",result)
    if not passed:raise ValueError("Raw parity failed; only this Raw cost path is ineligible")
    return result


def cost(condition,receipt,task,pool,context,output,batch,deadline,manifest_sha,quiet,proof_paths):
    if batch not in (1,8):raise ValueError("Only B1/B8 cost supported")
    quiet_path,quiet_sha=quiet
    if not quiet_path or not quiet_sha or sha(quiet_path)!=quiet_sha:raise ValueError("Exact fresh ROOT quiet proof required")
    context["libraries"]["minima"].require_quiet(read(quiet_path),time.time())
    required_modes={"rescore","replay"}|({"raw_parity"} if condition["family"]=="RAW" else set())
    proof_paths={key:value for key,value in proof_paths.items() if key in required_modes};values={}
    for mode,(path,digest) in proof_paths.items():
        if path and digest:
            if sha(path)!=digest:raise ValueError("Validation proof changed")
            values[mode]=read(path)
            if values[mode].get("plan_sha256")!=context["plan_sha"]:raise ValueError("Validation proof belongs to another plan")
    validate_cost_proofs(condition["family"],result_identity(condition,receipt,manifest_sha,context),values,batch)
    import measure_costs
    origins=context["data"].origins(BOUNDS["V_select"],L,H,H)[:batch];physical,observed=physical_batch(task,origins,context["np"])
    route=make_route(condition,receipt,task,pool,context)
    class Lease:
        def check(self):context["priority"].check_deadline(deadline)
    try:
        rows=measure_costs.measure_balanced([route],physical,observed,warmups=10,timed_trials=20,stage_trials=20,lease=Lease())
        expected=expected_counts(condition["family"],[batch])
        if len(rows)!=50 or any((row["F_logical_streams"],row["F_physical_streams"],row["F_calls"])!=(expected["streams"],expected["physical_streams"],expected["physical_calls"]) for row in rows):
            raise ValueError("Actual full physical-pipeline F counters/protocol differ")
        memory=route.memory_state();load_seconds=pool.load_seconds+route.load_seconds
    finally:route.close()
    if sha(quiet_path)!=quiet_sha or any(path and digest and sha(path)!=digest for path,digest in proof_paths.values()):raise ValueError("Cost prerequisites changed during measurement")
    trials=output/"COST_TRIALS.csv.gz";measure_costs._write_trials(trials,rows)
    result=dict(status="COMPLETE_PROTOCOL",**result_identity(condition,receipt,manifest_sha,context),device="cpu",CPU_threads=2,batch_size=batch,
        warmups=10,timed_trials=20,stage_trials=20,actual_trial_rows=50,summaries=measure_costs.summarize(rows),memory=memory,trials_sha256=sha(trials),
        origins=origins.tolist(),physical_input_sha256=hashlib.sha256(physical.tobytes()).hexdigest(),observation_mask_sha256=hashlib.sha256(observed.tobytes()).hexdigest(),
        quiet_proof_sha256=quiet_sha,validation_result_sha256={mode:digest for mode,(path,digest) in proof_paths.items() if path and digest},
        F_counters_per_trial=expected,row_microbatch=128,physical_output_and_inverse_dtype="float64",cached_Z_or_forecast_reads=0,
        training_peak_included=False,stage_probe_separate_intrusive_scope=True,new_T_numeric_values_parsed=0,model_load_seconds=load_seconds)
    new(output/"COST_RECEIPT.json",result);return result


def execute(public,cache,asset,plan_sha,identifier,mode,deadline,output,batch=None,quiet=(None,None),proof_paths=None):
    import psutil
    public,cache,asset=map(lambda p:relative(p,"."),(public,cache,asset))
    plan,condition,libraries=metadata(public,cache,asset,plan_sha,identifier,psutil)
    if (mode not in ("fit","rescore","replay","raw_parity","cost") or type(deadline) not in (int,float) or not time.time()<deadline<=DEADLINE
            or ((mode in ("raw_parity","cost"))!=(batch in (1,8)))):raise ValueError("Explicit mode, live original deadline and only B1/B8 parity/cost required")
    if mode=="raw_parity" and condition["family"]!="RAW":raise ValueError("Raw parity cannot run another family")
    p1=libraries["priority"];p1.require_environment()
    scope,private=relative(public,NAME),relative(cache,NAME)
    if not private.is_dir():raise FileNotFoundError("ROOT must create the fresh private namespace")
    output=relative(scope,os.path.relpath(Path(output).absolute(),scope))
    if output.exists() or not output.is_relative_to(scope/"diagnostics"):raise FileExistsError("Exclusive new diagnostic output required")
    if mode=="fit":require_fresh(scope,private,identifier)
    bound_proofs={}
    for label,(path,digest) in (proof_paths or {}).items():
        if path is not None:
            resolved=relative(scope,os.path.relpath(Path(path).absolute(),scope))
            if not resolved.is_relative_to(scope/"diagnostics") or resolved.name!="RESULT.json":raise ValueError("Validation RESULT must remain in this namespace's diagnostics")
            bound_proofs[label]=(resolved,digest)
        else:bound_proofs[label]=(None,digest)
    proof_paths=bound_proofs
    if quiet[0] is not None:quiet=(relative(cache,os.path.relpath(Path(quiet[0]).absolute(),cache)),quiet[1])
    resource_preflight(public,cache,psutil);output.mkdir(parents=True,exist_ok=False)
    started=time.time();new(output/"STARTED.json",dict(run_id=identifier,mode=mode,plan_sha256=plan_sha,actual_pid=os.getpid(),actual_create_time=psutil.Process().create_time(),deadline_unix=deadline))
    public_run,private_run=scope/"runs"/identifier,private/"runs"/identifier
    public_prediction,private_prediction=scope/"predictions"/identifier/"V_select",private/"predictions"/identifier
    pool,context,result,failure,post_error=None,None,None,None,None
    try:
        sys.path.insert(0,str(public))
        import numpy as np
        import data_contract as data
        import evaluation
        if Path(data.__file__).resolve()!=public/"data_contract.py":raise ValueError("Scientific import root differs")
        context=dict(np=np,data=data,evaluation=evaluation,priority=p1,libraries=libraries,plan=plan,plan_sha=plan_sha,
            private_scope=private,model_path=relative(asset,plan["model_asset_relative_path"]),foundation_sha=plan["foundation_tensor_sha256"],
            new_sources=dict(runner=plan["runner_sha256"],**plan["dependencies"]))
        if mode=="rescore":
            receipt,artifact=completed(scope,private,condition,context)
            result=rescore(condition,receipt,context,public_prediction,private_prediction,output)
        else:
            import torch
            import fit_base as fit
            import deployment
            import gpu_runtime
            torch.set_num_threads(2)
            if torch.cuda.is_initialized():raise RuntimeError("CPU entry initialized CUDA")
            context.update(torch=torch,fit=fit,deployment=deployment)
            parent,access=libraries["pems"].load_prefix(cache,plan["parent_task_contract"],p1,libraries["primitive"])
            task=subset_task(parent,plan,context)
            access.update(actual_C=C,column_selection=plan["column_selection"],subset_identity_payload=task["subset_identity_payload"],derived_subset_data_sha256=task["data_sha256"],
                actual_subset_prefix_array_sha256=task["array_sha256"],full_subset_T_identity_computed=False)
            new(output/"PREFIX_ACCESS_RECEIPT.json",access)
            pool=deployment.FoundationPool(context["model_path"],"cpu")
            foundation=None if condition["family"]=="DIRECT_G" else pool.get(H)
            if foundation is not None and (gpu_runtime.state_hash(foundation.model)!=plan["foundation_tensor_sha256"] or foundation.row_microbatch!=128
                    or foundation.checkpoint_inputs is not False or foundation.instance_norm_source_sha256!=plan["instance_norm_source_sha256"]
                    or any(p.requires_grad or p.grad is not None for p in foundation.model.parameters())):raise ValueError("Actual frozen CPU F/runtime changed")
            if mode=="fit":
                new(scope/"claims"/("fit_"+identifier+".json"),dict(run_id=identifier,plan_sha256=plan_sha,created_unix=time.time(),new_subset_rule=True))
                public_run.mkdir(parents=True,exist_ok=False);private_run.mkdir(parents=True,exist_ok=False)
                backward,handle=observe_backward(foundation,torch);context["F_backward_observation"]=backward
                try:fitted=fit_condition(condition,task,foundation,context,public_run,private_run,deadline)
                finally:
                    if handle is not None:handle.remove()
                new(output/"F_BACKWARD_OBSERVATION.json",backward)
                if condition["family"] in ("B0","B2") and fitted["status"]=="COMPLETE_VALID" and (backward["finite_callbacks"]<1 or backward["finite_callbacks"]!=backward["F_input_gradient_callbacks"]):
                    raise ValueError("Actual finite backward path through frozen F input not observed")
                if fitted["status"]!="COMPLETE_VALID":result=fitted
                else:
                    manifest=full_prediction(condition,fitted,task,pool,context,public_prediction,private_prediction,deadline)
                    result=dict(status="COMPLETE_FIT_AND_FULL_V",fit_receipt=fitted["receipt"],V_score=manifest["score"],full_V_origins=148,
                        independent_rescore_pending=True,checkpoint_replay_pending=True,data_sha256=task["data_sha256"])
            else:
                receipt,artifact=completed(scope,private,condition,context)
                if receipt["data_sha256"]!=task["data_sha256"] or receipt["actual_subset_prefix_sha256"]!=task["materialized_prefix_sha256"]:
                    raise ValueError("Actual prefix subset differs from selected artifact")
                manifest,_=saved_metadata(public_prediction,condition,receipt,context)
                manifest_sha=sha(public_prediction/"PREDICTION_MANIFEST.json")
                if mode=="replay":
                    arrays,_=saved_arrays(public_prediction,private_prediction,condition,receipt,context)
                    result=replay(condition,receipt,arrays,task,pool,context,output,deadline,manifest_sha)
                elif mode=="raw_parity":result=raw_parity(condition,receipt,task,pool,context,output,batch,deadline,manifest_sha)
                else:result=cost(condition,receipt,task,pool,context,output,batch,deadline,manifest_sha,quiet,proof_paths or {})
            if foundation is not None:
                foundation.assert_frozen_unchanged()
                if any(p.requires_grad or p.grad is not None for p in foundation.model.parameters()) or foundation.row_microbatch!=128:
                    raise ValueError("Frozen F flags/gradient or restored row policy changed")
                result["F_final_state_sha256"]=gpu_runtime.state_hash(foundation.model)
            elif condition["family"]=="DIRECT_G" and (pool.instances_created or pool.foundation is not None or any(pool.counters().values())):
                raise ValueError("Direct G must not instantiate or call a foundation")
            result["foundation_instances_created"]=pool.instances_created
        if result.get("status")!="BUDGET_LIMIT_PRESERVED": p1.check_deadline(deadline)
    except Exception as error:
        failure=type(error).__name__;path=private/"failures"/(mode+"_"+identifier+"_"+str(time.time_ns())+".json")
        new(path,dict(error_class=failure,error_message=str(error),traceback=traceback.format_exc()))
        result=dict(status="FAILED_PRESERVED",error_class=failure,private_failure_cache_relative_path=path.relative_to(cache).as_posix())
    finally:
        try:
            if pool is not None:pool.close()
            metadata(public,cache,asset,plan_sha,identifier,psutil);resource_preflight(public,cache,psutil)
            if context is not None and "torch" in context and context["torch"].cuda.is_initialized():raise RuntimeError("Unexpected CUDA initialization")
        except Exception as error:
            post_error=type(error).__name__;new(private/"failures"/("post_"+mode+"_"+identifier+"_"+str(time.time_ns())+".json"),dict(error_class=post_error,error_message=str(error),traceback=traceback.format_exc()))
        memory=psutil.Process().memory_info()
        new(output/"RESULT.json",dict(status="FAILED_PRESERVED" if failure or post_error else (result or {}).get("status","INCOMPLETE"),
            run_id=identifier,mode=mode,plan_sha256=plan_sha,scientific_result=result,error_class=failure,postflight_error_class=post_error,
            original_source39_immutable9_preserved=post_error is None,elapsed_wall_seconds=time.time()-started,new_T_numeric_values_parsed=0,
            OS_peak_working_set_bytes=max(memory.rss,getattr(memory,"peak_wset",0)),process_RSS_bytes=memory.rss,
            actual_CPU_threads=2,actual_GPU_seconds=0,ROOT_external_RSS_CPU2_2slot_6GiB_deadline_and_quiet_guard_required=True,
            original_registry_changed=False,independent_generalization_claimed=False))
    if failure or post_error:raise RuntimeError("Selected subset path failed; original evidence preserved; other conditions remain independent")
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--execute",action="store_true");parser.add_argument("--root-triggered",action="store_true")
    for name in ("public-root","cache-root","asset-root","plan-sha","run-id","output","quiet-proof","quiet-sha","rescore-proof","rescore-sha","replay-proof","replay-sha","raw-parity-proof","raw-parity-sha"):
        parser.add_argument("--"+name)
    parser.add_argument("--mode",choices=("fit","rescore","replay","raw_parity","cost"));parser.add_argument("--deadline",type=float);parser.add_argument("--batch",type=int,choices=(1,8))
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="DRAFT_NOT_EXECUTED",input_reads=0,files_written=0,numpy_imported="numpy" in sys.modules,torch_imported="torch" in sys.modules,
            actual_training_prediction_replay_cost_or_child_operations=0)));return 0
    if not args.root_triggered or any(getattr(args,name) is None for name in ("public_root","cache_root","asset_root","plan_sha","run_id","mode","deadline","output")):
        parser.error("Explicit ROOT trigger, sealed PLAN SHA, roots, one run/mode/deadline and fresh output required")
    proofs=dict(rescore=(args.rescore_proof,args.rescore_sha),replay=(args.replay_proof,args.replay_sha),raw_parity=(args.raw_parity_proof,args.raw_parity_sha))
    result=execute(args.public_root,args.cache_root,args.asset_root,args.plan_sha,args.run_id,args.mode,args.deadline,args.output,args.batch,(args.quiet_proof,args.quiet_sha),proofs)
    print(json.dumps(dict(status=result["status"],run_id=args.run_id,mode=args.mode)));return 0 if result["status"].startswith(("COMPLETE","PASS")) else 1


if __name__=="__main__":raise SystemExit(main())
