"""Preparation-only safe-prefix adapter for the three registered independent B0 fits.

No default input reads or scientific imports. ROOT alone admits/seals execution.
The original independent single-K recipes and original scientific APIs are kept.
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

NAME = "priority_multik_independent_20261016_v1"
OLD = "postT_V_only_cpu_multik_20261016_v1"
OLD_PLAN_SHA = "2051573b8dee48200eb34e96e37e7af0d197442619ea1ba52a36a80ad4e27aec"
DEADLINE, L, H, C, SEED = 1792222550, 512, 48, 11, 92601
BOUNDS = dict(A_fit=[0,10519],A_stop=[10519,12622],M_cal=[12622,14726],V_select=[14726,16830],T_final=[16830,21038])
DEPENDENCIES = {
    "priority_valid_family_20261016_v1/family_minimum_runner.py":"12462b8f1085e346750a24cd37fd52191f8780799e2f9e5903b4a57bcbc570c1",
    "priority_valid_minima_20261016_v1/beijing_minimum_runner.py":"b84f1473ff4d562890910f7555ef3925823dd921539c8d757118c95504513b9b",
    "priority_b2_validation_20261016_v1/priority_runner.py":"f3e4279ddd250a51b09ce88938f8f202dfd02e3b24d1adff3bb36c567dc7ef07",
    "priority_b2_validation_20261016_v1/safe_jena_prefix.py":"6b496dcbbb6cef48acb9010c630ed81359a2d60bff89aaec9e9e77c4b6d0b541",
    "priority_core_followup_20261016_v1/jena_seed_runner.py":"ad4b76acf51fb416dea4f69c36940507db1f92e223c620388ded45381d6e28e7",
    "priority_nested_cost_20261016_v1/prefix_nested_cost_runner.py":"d64d543499ed21c5a294ace19967c555d05bcfde0cd5ea6c98cbe8ddb23a4ae5"}
REGISTRATION_PINS = {
    OLD+"/PLAN.json":OLD_PLAN_SHA,
    OLD+"/REGISTRY.csv":"7bb634cf469105c455d552ffedeadbfcb324c2803a95829891589c7fa10c5144",
    OLD+"/EXPERIMENT_REGISTRY.csv":"7bb634cf469105c455d552ffedeadbfcb324c2803a95829891589c7fa10c5144",
    OLD+"/multik_runner.py":"44cd5e3fed5f4ad8a2110c54c9118063c6838b68d9059dea46f6b33bc32be10a",
    "ordered/primary_beijing/FAMILY_LOCK.json":"6954164639efcef4605bdcd22af91a48d262a838ab78bb52852f13478a6ff747",
    "postT_V_only_cpu_fixed_pca_20261016_v1/PLAN.json":"b4207edec63331cf95b24909c24c692091251dd8e29316f8867b9f66c65a0930",
    "postT_V_only_cpu_lag_20261016_v1/cpu_lag_runner.py":"36c75b38cd2c072bac4466129249cc5213c2df679ad80232d4ac511ab6d7e7f1",
    "postT_V_only_cpu_fixed_pca_20261016_v1/fixed_pca_runner.py":"d73d866a6d03061d00fb7ca015a8f3b41e273b73011b22f0c1cd0fbbfb073dd9",
    "plan/MASTER_PLAN.md":"34d242626f163a8c510d4f6750170bdf9a34af87a610b6aee9df7781fb708569"}
REFERENCES = {
    3:dict(manifest="d747c2e00f18e1c1c7c249edea43723e0969d5f36c8afd578318e942f84fc557",source="ea2488f24917da3c13d3bf7b31dd7a2a42921a911603642ed8f9102b8569f637",fit="714ff95bbb05e08b468fb2087bbd7f74a53dff8666a9ad030b208e3651c0bd96"),
    6:dict(manifest="9d10c15eb03784551ae0748c77cb0f394fdc6a69abe3aaec3790982308429ebc",source="f2b0a8821c4d9f18514843da6c5265498d01c653365a0b4a687fe65121eae98d",fit="c36fdbb653667dcd1e664c29d5917c8ce75e31a242af58e81951b4ec6d14aef0"),
    11:dict(manifest="3e01dc4c422240bb6a8017495c1b7caaab6129592cfc5a19614124ee9a64dbfd",source="b6c26ba47727556bb34c0015399360e343925918271625f75bf7b651ec075103",fit="515ad3a10ecfc1190752b56f3c201b2a7d622da2ee888adeedb16012489af8ec")}


def sha(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):value.update(block)
    return value.hexdigest()


def read(path):
    if Path(path).stat().st_size>8*1024**2:raise ValueError("Small JSON metadata required")
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


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def check_pins(root,pins):
    for path,digest in pins.items():
        if sha(relative(root,path))!=digest:raise ValueError("Pinned source/input changed: "+path)


def conditions():
    values=[]
    for k in (3,6,11):
        ident=f"postT_cpu_beijing_E4_INDEPENDENT_B0_k{k}_s92601_v1";upstream=f"primary_beijing_E1_B0_k{k}_s92601"
        spec=dict(run_id=ident,dataset="beijing",family="B0",latent_k=k,seed=SEED,fold="primary",epochs=[.25,1.,3.],lr=.001,device="cpu",microbatch=4,effective_batch=16,
            F_row_microbatch=128,checkpoint_inputs=False,lora_rank=8,lora_alpha=16,stage_label="postT_cpu_fresh_V_only_multik",provenance=dict(upstream_registered_id=upstream))
        values.append(dict(run_id=ident,role="INDEPENDENT_B0",dataset="beijing",family="B0",latent_k=k,effective_F_width=k,spec=spec,upstream_registered_id=upstream,
            V_reference=dict(relative_path="predictions/"+upstream+"/V_select/PREDICTION_MANIFEST.json",sha256=REFERENCES[k]["manifest"])))
    return values


def draft_plan():
    return dict(schema="PRIORITY_MULTIK_INDEPENDENT_PREFIX_v1",namespace=NAME,status="DRAFT_NOT_EXECUTABLE",runner_sha256=None,
        ROOT_after_N4_report_and_main_admitted=False,conditions=conditions(),maximum_new_training_conditions=3,
        dependencies=copy.deepcopy(DEPENDENCIES),registration_pins=copy.deepcopy(REGISTRATION_PINS),original_reference_pins={str(k):copy.deepcopy(value) for k,value in REFERENCES.items()},
        original_source39_and_immutable9_bound_by_exact_parent_PLAN=True,original_primary_split_ranges=copy.deepcopy(BOUNDS),materialized_prefix_rows=16830,
        new_T_target_operations_allowed=False,protected_tail_numeric_parse_allowed=False,deadline_unix=DEADLINE,stage="POST_T_V_ONLY_DEVELOPMENT",
        optimizer=dict(name="AdamW",lr=.001,weight_decay=0.,gradient_clip=1.,effective_batch=16,microbatch=4),
        resources=dict(CPU_threads=2,max_worker_RSS_bytes=3*1024**3,maximum_model_workers=2,total_owned_RSS_limit_bytes=6*1024**3,
            storage_limit_bytes=30*1024**3,admission_reserve_bytes=256*1024**2,GPU_process_seconds=0,ROOT_CPU_supervisor_required=True),
        costs=dict(batches=[1,8],warmups=10,timed_trials=20,stage_trials=20,quiet_max_age_seconds=120,actual_rescore_replay_outer_result_SHA_required=True),
        initialization="Original single-K no k_grid: K3/6 A_fitted_centered_PCA, K11=C identity/zero center; search/refit fit their own A50/A60 normalization.",
        initialization_confound="Shared/max-only K11 has a grid and starts from PCA; independent K11 identity is not a pure shared-training contrast.",
        raw_asset_binding=dict(source_relative_path=".cache/smallk_predictive_alignment_20261013/beijing_repair/raw_dataset/beijing_aotizhongxin/b04da438b2f331ac/fit_tune_payload.npz",
            sha256="78063851dd3e65512e3e3ced22210839697553f2d652fc0be9513736af598277",bytes=1524625,contract_sha256="9116703c22b3dea317cb1ff750fd45d05c5d6e4a226946a41334e0794484849f",
            source_rows=28051,original_permitted_rows=21038,numeric_prefix_rows=16830,source_header_contract="nested dependency HEADERS: physical<f4 C-order(28051,11), maskbool, times<U19, columns<U5"),
        model_binding=dict(parent_PLAN_sha256=OLD_PLAN_SHA,foundation_tensor_sha256="c2b4edf7db22d3f18e58ca79299a5c7e864e9b8f277e4e29e2f88d07dbb35874",
            snapshot_revision="772f3d25d38aec6d914c8949dab4462e2d46f5d8",config_and_weight_path_size_SHA_bindings="Pinned common.MODEL_BINDINGS with resolved blob under asset root; checked before and after actual operations"),
        original_scalers=dict(search_A50="cb394e93fec460e933a56c46cfc43645ba787fd765f064668788e33d9d55bb9d",refit_A60="3348bcf1fa76d382f23cf20597fa90de6e21dac587d4ca9f6883f235a10f93fb"),
        scope="Only the three originally registered E4 independent fresh B0 fits. N7 saved-parent resume and N3 fixed-PCA inverse are separate.",
        probe_mode_implemented=False,old_weights_optimizer_RNG_or_checkpoint_reused=False,partial_resume_implemented=False,
        interruption="Original _stage optimizer-boundary/deadline checkpoints preserved; no signal autosave promise; partial completion remains BUDGET_LIMIT_PRESERVED.",
        original_full_task_hash="INHERITED_NOT_RECOMPUTED",actual_prefix_statistics_and_V_fingerprints_must_match=True,
        prefix_io_scope="Only 0:16830 numeric conversion; whole compressed opaque SHA/ZIP buffering may read tail bytes. Not zero tail I/O.",
        actual_Task_model_fit_forward_rescore_cost_or_dispatch_operations=0,
        ROOT_seal_required="After N4 report/main reassess original purpose and remaining resources, explicitly admit and seal a new PLAN/current source SHA. No automatic queue.")


def validate_plan(plan):
    digest=plan.get("runner_sha256")
    if plan.get("status")!="SEALED_BY_ROOT" or plan.get("ROOT_after_N4_report_and_main_admitted") is not True or not isinstance(digest,str) or len(digest)!=64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("New ROOT seal/current source SHA and positive post-N4 admission required")
    for key,value in draft_plan().items():
        if key not in {"status","runner_sha256","ROOT_after_N4_report_and_main_admitted","actual_Task_model_fit_forward_rescore_cost_or_dispatch_operations"} and plan.get(key)!=value:
            raise ValueError("Exactly three original single-K recipes and prefix/resource scope required: "+key)


def choose(plan,identifier):
    found=[row for row in plan["conditions"] if row["run_id"]==identifier]
    if len(found)!=1:raise ValueError("One exact original independent CPU ID required")
    return found[0]


def require_unstarted(public,cache,identifier):
    for root in (Path(public),Path(cache),Path(public)/OLD,Path(cache)/OLD,Path(public)/NAME,Path(cache)/NAME):
        for kind in ("runs","predictions","ordered_predictions"):
            if (root/kind/identifier).exists():raise FileExistsError("Existing original/new complete or partial result preserved; no resume")
        if (root/"claims"/("fit_"+identifier+".json")).exists():raise FileExistsError("Existing fit claim preserved")


def expected_counters(k,batches):
    if type(k) is not int or k not in (3,6,11) or not batches or any(type(b) is not int or b<1 for b in batches):raise ValueError("Registered K3/6/11 and positive batches required")
    streams=sum(b*k for b in batches)
    return dict(streams=streams,physical_streams=streams,physical_calls=sum(math.ceil(b*k/128) for b in batches))


def validate_origins(origins):
    if not len(origins) or any(int(o)<14726 or int(o)+H>16830 for o in origins):raise ValueError("Original pre-T V origin boundary required")


def require_quiet(proof,now):
    stamp=proof.get("captured_unix")
    if proof.get("status")!="QUIET_CPU_COST_WINDOW" or proof.get("active_model_workers")!=[] or proof.get("active_cost_workers")!=[] or isinstance(stamp,bool) or not isinstance(stamp,(int,float)) or not math.isfinite(stamp) or not 0<=now-stamp<=120:
        raise ValueError("Fresh120s ROOT quiet proof with both active lists empty required")


def metadata(public,cache,asset,plan_sha,identifier,process_api):
    path=relative(public,NAME+"/PLAN.json")
    if not plan_sha or sha(path)!=plan_sha:raise ValueError("Exact new ROOT-sealed PLAN SHA required")
    plan=read(path);validate_plan(plan);condition=choose(plan,identifier)
    if sha(relative(public,NAME+"/multik_independent_runner.py"))!=plan["runner_sha256"]:raise ValueError("Current adapter source changed")
    check_pins(public,DEPENDENCIES);check_pins(public,REGISTRATION_PINS)
    libraries={key:module(relative(public,path),"independent_"+key) for key,path in (
        ("common","priority_valid_family_20261016_v1/family_minimum_runner.py"),("modelcheck","priority_valid_minima_20261016_v1/beijing_minimum_runner.py"),
        ("priority","priority_b2_validation_20261016_v1/priority_runner.py"),("primitive","priority_b2_validation_20261016_v1/safe_jena_prefix.py"),
        ("seed","priority_core_followup_20261016_v1/jena_seed_runner.py"),("nested","priority_nested_cost_20261016_v1/prefix_nested_cost_runner.py"),
        ("old",OLD+"/multik_runner.py"))}
    old=libraries["old"].metadata(public,cache,asset,OLD_PLAN_SHA)
    libraries["priority"].metadata(public,cache,"475f983058871ff36b4112b6aa56dc252b17aea47eb6ae1744ff119c2ad5138f",process_api)
    if len(old["source39"])!=39 or len(old["immutable_public"])!=9:raise ValueError("Original source39/immutable9 required")
    for current in conditions():
        found=[row for row in old["fits"] if row["run_id"]==current["run_id"]]
        if len(found)!=1 or any(found[0][key]!=current[key] for key in ("run_id","role","upstream_registered_id","spec")):raise ValueError("Original exact independent spec changed")
        upstream=current["upstream_registered_id"];ref=REFERENCES[current["latent_k"]]
        check_pins(public,{current["V_reference"]["relative_path"]:ref["manifest"],"runs/"+upstream+"/SOURCE_CONTRACT.json":ref["source"],"runs/"+upstream+"/BASE_FIT_RECEIPT.json":ref["fit"]})
    contract=old["task_contract"];meta=contract["metadata"];source=relative(asset,meta["source_reference"])
    if contract["primary_split_ranges"]!=BOUNDS or source.stat().st_size!=meta["source_bytes"] or sha(source)!=meta["source_sha256"] or sha(source.with_name("contract.json"))!=meta["source_contract_sha256"]:raise ValueError("Original task/raw opaque binding changed")
    libraries["modelcheck"].check_model_snapshot(asset,old["model_asset_relative_path"],libraries["common"].MODEL_BINDINGS)
    return plan,old,condition,libraries


def safe_task(asset,old,condition,libraries,context):
    np,data,fit=context["np"],context["data"],context["fit"]
    task,access=libraries["nested"].safe_task(asset,old,libraries["primitive"],np)
    libraries["common"].validate_prefix_shape(task,old["task_contract"],"beijing")
    if access["new_T_numeric_values_parsed"]!=0:raise ValueError("Protected T numeric conversion forbidden")
    mean,std,count=data.fit_statistics(task,10519)
    a50={key:data.array_sha256(value) for key,value in (("mean",mean),("std",std),("count",count))}
    a50scale=fit.scaler_fingerprint(mean,std);mean,std,_=data.fit_statistics(task,12622)
    libraries["common"].validate_statistics(old["task_contract"],a50,a50scale,fit.scaler_fingerprint(mean,std))
    arrays=context["priority"].cohort_arrays(task,BOUNDS,mean,std,data,np)
    context["priority"].validate_cohort(arrays,context["reference"],context["evaluation"])
    access.update(actual_A50_A60_and_original_V43_fingerprints_matched=True,old_full_task_identity_inherited_not_recomputed=True)
    return task,access


def validate_initialization(norm,k,end,task,data,np):
    if norm.get("fit_end")!=end:raise ValueError("Original stage-specific fit_end required")
    if k==C:
        basis,center,kind=np.eye(C,dtype=np.float32),np.zeros(C,dtype=np.float32),"identity_at_K_equals_C"
    else:
        pca=data.initialize_pca(task,end,k,norm["mean"],norm["std"]);basis,center,kind=pca["E"],pca["pca_center"],"A_fitted_centered_PCA"
    if norm.get("basis_initialization")!=kind or not np.array_equal(norm["basis"],basis) or not np.array_equal(norm["pca_center"],center):raise ValueError("Original independent single-K initialization differs")


def fit_and_save(condition,task,foundation,context,public_run,private_run,deadline):
    fit,p1,common=context["fit"],context["priority"],context["common"];spec=condition["spec"]
    source,files=fit._source_contract()
    contract=dict(source_sha=source,config_sha=fingerprint(dict(spec=spec,L=L,H=H,C=C,columns=task["columns"],inherited_array_sha256=task["array_sha256"],materialized_prefix_sha256=task["materialized_prefix_sha256"])),data_sha=task["data_sha256"],model_sha=fit._model_sha(foundation))
    new(public_run/"SOURCE_CONTRACT.json",dict(**contract,files=files,spec=spec,original_task_identity="INHERITED_NOT_RECOMPUTED",actual_prefix_sha256=task["materialized_prefix_sha256"],original_registered_CPU_namespace=OLD,fresh_independent_not_parent_resume=True))
    stages=common.fit_stages(fit,condition,task,foundation,contract,public_run,private_run,deadline)
    if stages["status"]!="COMPLETE_STAGES":return stages
    search,refit,epoch=stages["search"],stages["refit"],stages["selected_epoch"];common.validate_stages(condition,search["summary"],refit["summary"])
    initializations=[]
    for label,stage,end in (("search",search,10519),("refit",refit,12622)):
        norm=stage["normalization"];validate_initialization(norm,condition["latent_k"],end,task,context["data"],context["np"])
        initializations.append(dict(stage=label,fit_end=end,basis_initialization=norm["basis_initialization"],basis_sha256=context["data"].array_sha256(norm["basis"]),center_sha256=context["data"].array_sha256(norm["pca_center"])))
        p1.write_stage(public_run/label,stage)
    bundle=dict(model=refit["model"],normalization=refit["normalization"],spec=spec,contract=refit["summary"]["contract"],selected_search_epoch=epoch,search_summary=search["summary"],refit_summary=refit["summary"],model_state_sha=p1.state_sha(refit["model"],context["data"]),plan_sha256=context["plan_sha"],actual_prefix_sha256=task["materialized_prefix_sha256"])
    artifact=private_run/"selected_bundle.pt";p1.save_bundle(artifact,bundle,context["torch"])
    receipt=dict(status="COMPLETE_VALID",run_id=condition["run_id"],family="B0",role="INDEPENDENT_B0",seed=SEED,epochs=[.25,1.,3.],selected_epoch=epoch,artifact_sha256=sha(artifact),model_state_sha=bundle["model_state_sha"],refit_scaler_sha=fit.scaler_fingerprint(bundle["normalization"]["mean"],bundle["normalization"]["std"]),
        training_status="UNDERTRAINED_OR_BUDGET_LIMIT" if epoch==3 else "SCREEN_COMPLETE_CONVERGENCE_UNRESOLVED",search_fit_end=10519,refit_fit_end=12622,search_updates=search["summary"]["updates"],refit_updates=refit["summary"]["updates"],search_exposures=search["summary"]["exposures"],refit_exposures=refit["summary"]["exposures"],initializations=initializations,
        artifact_cache_relative_path=NAME+"/runs/"+condition["run_id"]+"/selected_bundle.pt",original_registered_CPU_namespace=OLD,source_stage_API="original_fit_base._stage",old_weights_optimizer_RNG_or_state_reused=False,reused_complete_artifact=False,original_full_task_identity="INHERITED_NOT_RECOMPUTED",actual_prefix_sha256=task["materialized_prefix_sha256"],post_T_development=True,new_T_target_operations=0)
    new(public_run/"BASE_FIT_RECEIPT.json",receipt)
    return dict(receipt=receipt,artifact=artifact)


def canonical_route(condition,receipt):
    return dict(run_id=condition["run_id"],dataset="beijing",family="B0",fold="primary",latent_k=condition["latent_k"],canonical_fit_end=12622,
        artifact_cache_relative_path=NAME+"/runs/"+condition["run_id"]+"/selected_bundle.pt",artifact_sha256=receipt["artifact_sha256"],row_microbatch=128)


def actual_route(fitted,condition,task,pool,context):
    receipt=fitted["receipt"];route=context["deployment"].Route(canonical_route(condition,receipt),task,context["cache"],device="cpu",foundation_pool=pool)
    try:
        bundle=route.bundle
        if (route.family!="B0" or route.latent_k!=condition["latent_k"] or bundle["spec"]!=condition["spec"] or bundle["model_state_sha"]!=receipt["model_state_sha"] or bundle["plan_sha256"]!=context["plan_sha"]
                or bundle["contract"]["data_sha"]!=context["original"]["data_sha256"] or bundle["contract"]["model_sha"]!=context["foundation_sha"] or bundle["normalization"]["fit_end"]!=12622 or context["fit"].scaler_fingerprint(route.mean,route.std)!=receipt["refit_scaler_sha"]):raise ValueError("Exact selected independent artifact/family/K/A60/plan identity required")
        return route
    except BaseException:
        route.close();pool.foundation.assert_frozen_unchanged();raise


@contextlib.contextmanager
def route_lifetime(route,foundation):
    try:yield route
    finally:route.close();foundation.assert_frozen_unchanged()


def physical_predictions(route,origins,task,context,deadline):
    np=context["np"];values=[]
    for start in range(0,len(origins),4):
        context["priority"].check_deadline(deadline);oo=origins[start:start+4]
        physical=np.stack([task["physical"][int(o)-L:int(o)] for o in oo]);observed=np.stack([task["observed"][int(o)-L:int(o)] for o in oo])
        raw=route.predict_physical(physical,observed)
        values.append(((raw-route.mean.astype(np.float64))/route.std.astype(np.float64)).astype(np.float32))
    return np.concatenate(values)


def full_prediction(fitted,condition,task,pool,context,public_folder,private_folder,deadline):
    p1,np=context["priority"],context["np"];route=actual_route(fitted,condition,task,pool,context)
    with route_lifetime(route,pool.foundation):
        arrays=p1.cohort_arrays(task,BOUNDS,route.mean,route.std,context["data"],np);p1.validate_cohort(arrays,context["reference"],context["evaluation"]);validate_origins(arrays["origins"])
        before=pool.counters();arrays["pred_normalized"]=physical_predictions(route,arrays["origins"],task,context,deadline)
        delta={key:pool.counters()[key]-before[key] for key in ("streams","physical_streams","physical_calls")}
        if arrays["pred_normalized"].shape!=(43,H,C) or delta!=expected_counters(condition["latent_k"],[4]*10+[3]):raise ValueError("Full V43 output/counters differ")
    public_folder.mkdir(parents=True,exist_ok=False);private_folder.mkdir(parents=True,exist_ok=False);receipt=fitted["receipt"]
    identity=dict(run_id=condition["run_id"],split="V_select",cohort="beijing:primary:V_select:stride48",data_sha=task["data_sha256"],scaler_sha=receipt["refit_scaler_sha"],source_sha=fingerprint(context["new_sources"]),config_sha=fingerprint(condition["spec"]),model_sha=fingerprint(dict(artifact_sha=receipt["artifact_sha256"],model_state_sha=receipt["model_state_sha"])))
    new(public_folder/"REFERENCE_CONTRACT.json",dict(**identity,arrays={key:context["evaluation"].array_fingerprint(value) for key,value in arrays.items() if key!="pred_normalized"}))
    archive=private_folder/"V_select.npz"
    with archive.open("xb") as stream:np.savez_compressed(stream,**arrays)
    score,rows=context["evaluation"].error_statistics(arrays["pred_normalized"],arrays["target_normalized"],arrays["observed"],arrays["origins"],condition["run_id"],"V_select",identity["cohort"],arrays["channel_ids"])
    with gzip.open(public_folder/"PUBLIC_ERROR_STATS.csv.gz","xt",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    p1.csv_new(public_folder/"SCORES.csv",[score])
    manifest=dict(**identity,archive_sha256=sha(archive),arrays={key:context["evaluation"].array_fingerprint(value) for key,value in arrays.items()},score=score,family="B0",seed=SEED,artifact_sha256=receipt["artifact_sha256"],device="cpu",CPU_threads=2,F_counters=delta,post_T_development=True,new_T_target_operations=0,route="original_deployment.Route_full_constructor")
    validate_saved_manifest(manifest,condition,receipt,context);new(public_folder/"PREDICTION_MANIFEST.json",manifest)
    return manifest


def validate_saved_manifest(manifest,condition,receipt,context):
    expected=dict(run_id=condition["run_id"],family="B0",seed=SEED,split="V_select",cohort="beijing:primary:V_select:stride48",device="cpu",data_sha=context["original"]["data_sha256"],artifact_sha256=receipt["artifact_sha256"],scaler_sha=receipt["refit_scaler_sha"],source_sha=fingerprint(context["new_sources"]),config_sha=fingerprint(condition["spec"]),model_sha=fingerprint(dict(artifact_sha=receipt["artifact_sha256"],model_state_sha=receipt["model_state_sha"])))
    if any(manifest.get(key)!=value for key,value in expected.items()):raise ValueError("Same selected new CPU B0/V identity required")
    for key in ("target_normalized","observed","origins","channel_ids"):
        if manifest["arrays"].get(key)!=context["reference"]["arrays"][key]:raise ValueError("Original full V fingerprint differs")
    pred=manifest["arrays"]["pred_normalized"]
    if pred.get("shape")!=[43,H,C] or pred.get("dtype")!="float32" or any(manifest["score"].get(key)!=value for key,value in dict(n_origins=43,n_observed=20539,num_channels=C,horizon=H).items()):raise ValueError("Full V43/observed20539 geometry required")


def replay(fitted,condition,arrays,task,pool,context,output,deadline):
    np=context["np"];indices=context["priority"].replay_indices(43,np);origins=arrays["origins"][indices];route=actual_route(fitted,condition,task,pool,context)
    with route_lifetime(route,pool.foundation):
        before=pool.counters();actual=physical_predictions(route,origins,task,context,deadline)
        delta={key:pool.counters()[key]-before[key] for key in ("streams","physical_streams","physical_calls")}
        if actual.shape!=(6,H,C) or delta!=expected_counters(condition["latent_k"],[4,2]):raise ValueError("Fresh CPU6 geometry/stream accounting differs")
        expected=arrays["pred_normalized"][indices];passed=bool(np.allclose(actual,expected,atol=1e-5,rtol=1e-4))
        result=dict(status="PASS_FRESH_CPU6_REPLAY" if passed else "FAIL_REPLAY_PARITY",executed_now=True,run_id=condition["run_id"],artifact_sha256=fitted["receipt"]["artifact_sha256"],origins=origins.tolist(),microbatches=[4,2],F_counters=delta,base_frozen_after_original_Route_close=True,atol=1e-5,rtol=1e-4,max_abs=float(np.abs(actual.astype(np.float64)-expected.astype(np.float64)).max()),new_T_target_operations=0)
    new(output/"REPLAY_RECEIPT.json",result)
    if not passed:raise ValueError("Fresh original CPU6 checkpoint parity failed")
    return result


def cost(fitted,condition,task,pool,context,manifest_sha,output,batch,deadline,quiet_path,quiet_sha,proofs):
    if batch not in (1,8) or not quiet_path or not quiet_sha or sha(quiet_path)!=quiet_sha:raise ValueError("B1/B8 exact quiet proof required")
    require_quiet(read(quiet_path),time.time());context["common"].verify_cost_prerequisites(condition,fitted["receipt"],manifest_sha,proofs)
    import measure_costs
    np=context["np"];origins=context["data"].origins(BOUNDS["V_select"],L,H,H)[:batch];validate_origins(origins)
    physical=np.stack([task["physical"][int(o)-L:int(o)] for o in origins]);observed=np.stack([task["observed"][int(o)-L:int(o)] for o in origins]);route=actual_route(fitted,condition,task,pool,context)
    class Check:
        def check(self):context["priority"].check_deadline(deadline)
    with route_lifetime(route,pool.foundation):
        before=pool.counters();rows=measure_costs.measure_balanced([route],physical,observed,warmups=10,timed_trials=20,stage_trials=20,lease=Check());want=expected_counters(condition["latent_k"],[batch])
        if len(rows)!=50 or any((row["F_logical_streams"],row["F_physical_streams"],row["F_calls"])!=(want["streams"],want["physical_streams"],want["physical_calls"]) for row in rows):raise ValueError("Original50trial K-specific full-route counters differ")
        delta={key:pool.counters()[key]-before[key] for key in ("streams","physical_streams","physical_calls")}
        if delta!={key:value*50 for key,value in want.items()}:raise ValueError("Actual total cost stream/call counters differ")
        memory=route.memory_state()
        if sha(quiet_path)!=quiet_sha or any(sha(path)!=digest for path,digest in proofs.values()):raise ValueError("Prerequisite evidence changed during cost")
    trials=output/"COST_TRIALS.csv.gz";measure_costs._write_trials(trials,rows)
    result=dict(status="COMPLETE_PROTOCOL",run_id=condition["run_id"],device="cpu",CPU_threads=2,batch_size=batch,warmups=10,timed_trials=20,stage_trials=20,actual_trial_rows=50,summaries=measure_costs.summarize(rows),memory=memory,trials_sha256=sha(trials),accuracy_manifest_sha256=manifest_sha,artifact_sha256=fitted["receipt"]["artifact_sha256"],quiet_proof_sha256=quiet_sha,validation_result_sha256={key:value[1] for key,value in proofs.items()},model_load_seconds=route.load_seconds,model_loading_in_whole_trial_timer=False,physical_input_sha256=hashlib.sha256(physical.tobytes()).hexdigest(),observation_mask_sha256=hashlib.sha256(observed.tobytes()).hexdigest(),origins=origins.tolist(),F_counters=delta,F_logical_streams_per_trial=want["streams"],F_calls_per_trial=want["physical_calls"],cached_Z_or_forecast_reads=0,physical_output_and_inverse_dtype="float64",stage_probe_separate_intrusive_scope=True,training_peak_included=False,frozen_F_base_sha256=context["foundation_sha"],base_restored_after_original_Route_close=True,new_T_target_operations=0)
    new(output/"COST_RECEIPT.json",result);return result


def execute(public,cache,asset,plan_sha,identifier,mode,deadline,output,batch=None,quiet_path=None,quiet_sha=None,proofs=None):
    import psutil
    public,cache,asset=map(lambda p:relative(p,"."),(public,cache,asset));plan,old,condition,libraries=metadata(public,cache,asset,plan_sha,identifier,psutil)
    if mode not in ("fit","rescore","replay","cost") or not isinstance(deadline,(int,float)) or not time.time()<deadline<=DEADLINE:raise ValueError("Explicit mode and live original deadline required")
    if (mode=="cost" and batch not in (1,8)) or (mode!="cost" and batch is not None):raise ValueError("Only cost takes B1/B8")
    p1,common=libraries["priority"],libraries["common"];p1.require_environment();scope,private=relative(public,NAME),relative(cache,NAME)
    if not private.is_dir():raise FileNotFoundError("ROOT must prepare new private namespace")
    output=relative(scope,os.path.relpath(Path(output).absolute(),scope))
    if output.exists() or not output.is_relative_to(scope/"diagnostics"):raise FileExistsError("Fresh exclusive diagnostic output required")
    if mode=="fit":require_unstarted(public,cache,identifier)
    if mode=="cost":
        quiet_path=relative(cache,os.path.relpath(Path(quiet_path).absolute(),cache)) if quiet_path else None
        proofs={key:(relative(scope,os.path.relpath(Path(path).absolute(),scope)),digest) for key,(path,digest) in (proofs or {}).items() if path and digest}
        if not quiet_path or not quiet_sha or sha(quiet_path)!=quiet_sha:raise ValueError("Exact private fresh quiet proof required")
        require_quiet(read(quiet_path),time.time())
    libraries["seed"].resource_preflight(public,cache,plan,psutil);output.mkdir(parents=True,exist_ok=False);started=time.time()
    new(output/"STARTED.json",dict(run_id=identifier,role="INDEPENDENT_B0",mode=mode,plan_sha256=plan_sha,actual_pid=os.getpid(),actual_create_time=psutil.Process().create_time(),deadline_unix=deadline))
    public_run,private_run=scope/"runs"/identifier,private/"runs"/identifier;public_prediction,private_prediction=scope/"predictions"/identifier/"V_select",private/"predictions"/identifier
    result=context=pool=None;failure=post_error=post_path=None
    try:
        sys.path.insert(0,str(public))
        import numpy as np
        import data_contract as data
        import evaluation
        if Path(data.__file__).resolve()!=public/"data_contract.py":raise ValueError("Original import root differs")
        context=dict(np=np,data=data,evaluation=evaluation,priority=p1,common=common,cache=cache,plan_sha=plan_sha,new_sources=dict(runner=plan["runner_sha256"],**DEPENDENCIES),original=old["task_contract"],reference=read(relative(public,condition["V_reference"]["relative_path"])),foundation_sha=old["foundation_tensor_sha256"])
        if mode=="rescore":
            fitted=common.completed(public_run,private_run,condition,context["original"])
            validate_saved_manifest(read(public_prediction/"PREDICTION_MANIFEST.json"),condition,fitted["receipt"],context)
            result=common.rescore(public_prediction,private_prediction,condition,fitted["receipt"],context,output)
        else:
            import torch
            import fit_base as fit
            import deployment
            import gpu_runtime
            torch.set_num_threads(2)
            if torch.cuda.is_initialized():raise RuntimeError("CPU operation initialized CUDA")
            context.update(torch=torch,fit=fit,deployment=deployment,gpu_runtime=gpu_runtime)
            task,access=safe_task(asset,old,condition,libraries,context);new(output/"PREFIX_ACCESS_RECEIPT.json",access)
            pool=deployment.FoundationPool(relative(asset,old["model_asset_relative_path"]),"cpu");foundation=pool.get(H)
            if (gpu_runtime.state_hash(foundation.model)!=old["foundation_tensor_sha256"] or foundation.row_microbatch!=128 or foundation.checkpoint_inputs is not False or foundation.instance_norm_source_sha256!=old["instance_norm_source_sha256"] or any(p.requires_grad or p.grad is not None for p in foundation.model.parameters())):raise ValueError("Original frozen Small/CPU/row128/normalization differs")
            if mode=="fit":
                new(scope/"claims"/("fit_"+identifier+".json"),dict(run_id=identifier,plan_sha256=plan_sha,created_unix=time.time(),prefix_only_fresh_independent=True))
                public_run.mkdir(parents=True,exist_ok=False);private_run.mkdir(parents=True,exist_ok=False);fitted=fit_and_save(condition,task,foundation,context,public_run,private_run,deadline)
                if "receipt" not in fitted:result=fitted
                else:
                    manifest=full_prediction(fitted,condition,task,pool,context,public_prediction,private_prediction,deadline)
                    result=dict(status="COMPLETE_FIT_AND_FULL_V",fit_receipt=fitted["receipt"],V_score=manifest["score"],full_V_completed=True,independent_rescore_pending=True,checkpoint_replay_pending=True)
            else:
                fitted=common.completed(public_run,private_run,condition,context["original"])
                if mode=="replay":arrays,manifest=common.saved_arrays(public_prediction,private_prediction,condition,fitted["receipt"],context)
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
        counters=None if pool is None else pool.counters();instances=0 if pool is None else pool.instances_created
        try:
            if pool is not None:pool.close()
            metadata(public,cache,asset,plan_sha,identifier,psutil);libraries["seed"].resource_preflight(public,cache,plan,psutil)
            if context is not None and "torch" in context and context["torch"].cuda.is_initialized():raise RuntimeError("CPU operation initialized CUDA")
        except Exception as error:
            post_error=type(error).__name__;post_path=private/"failures"/("post_"+mode+"_"+identifier+"_"+str(time.time_ns())+".json")
            new(post_path,dict(error_class=post_error,error_message=str(error),traceback=traceback.format_exc()))
        new(output/"RESULT.json",dict(status="FAILED_PRESERVED" if failure or post_error else (result or {}).get("status","INCOMPLETE"),run_id=identifier,role="INDEPENDENT_B0",mode=mode,plan_sha256=plan_sha,scientific_result=result,error_class=failure,postflight_error_class=post_error,postflight_private_failure_cache_relative_path=None if post_path is None else post_path.relative_to(cache).as_posix(),elapsed_wall_seconds=time.time()-started,
            original_source39_immutable9_registered3_and_model_pins_preserved=post_error is None,new_T_target_operations=0,actual_F_counters=counters,foundation_instances_created=instances,numpy_imported="numpy" in sys.modules,torch_imported="torch" in sys.modules,actual_CPU_process_seconds=sum(psutil.Process().cpu_times()[:2]),RSS_bytes=psutil.Process().memory_info().rss,OS_peak_working_set_bytes=getattr(psutil.Process().memory_info(),"peak_wset",None),old_full_task_identity_inherited_not_recomputed=True,independent_generalization_claimed=False,ROOT_RSS_CPU2_2slot_6GiB_deadline_and_quiet_supervisor_required=True))
    if failure or post_error:raise RuntimeError("Independent prefix operation failed; private traceback and evidence preserved")
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--execute",action="store_true");parser.add_argument("--root-triggered",action="store_true")
    for key in ("public-root","cache-root","asset-root","plan-sha","run-id","output","quiet-proof","quiet-sha","rescore-proof","rescore-sha","replay-proof","replay-sha"):parser.add_argument("--"+key)
    parser.add_argument("--mode",choices=("fit","rescore","replay","cost"));parser.add_argument("--deadline",type=float);parser.add_argument("--batch",type=int,choices=(1,8));args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="DRAFT_NOT_EXECUTED",input_reads=0,writes=0,actual_training=0,actual_new_training_conditions=0,scientific_or_child_operations=0,numpy_imported="numpy" in sys.modules,torch_imported="torch" in sys.modules,candidate_ids=[c["run_id"] for c in conditions()])))
        return 0
    if not args.root_triggered or any(getattr(args,key) is None for key in ("public_root","cache_root","asset_root","plan_sha","run_id","mode","deadline","output")):parser.error("Explicit ROOT trigger, new sealed PLAN SHA, roots, exact ID/mode/live deadline/fresh output required")
    result=execute(args.public_root,args.cache_root,args.asset_root,args.plan_sha,args.run_id,args.mode,args.deadline,args.output,args.batch,args.quiet_proof,args.quiet_sha,dict(rescore=(args.rescore_proof,args.rescore_sha),replay=(args.replay_proof,args.replay_sha)))
    print(json.dumps(dict(status=result["status"],run_id=args.run_id,mode=args.mode)))
    return 0 if result["status"].startswith(("COMPLETE","PASS")) else 1


if __name__=="__main__":raise SystemExit(main())
