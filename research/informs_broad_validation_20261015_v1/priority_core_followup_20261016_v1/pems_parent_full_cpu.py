"""Same-checkpoint PeMS B2 CPU148 inference and separate saved FP64 rescore."""
import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

NAME = "priority_core_followup_20261016_v1"
PRIOR_PLAN_SHA = "e27ae2c357fb648288809a4213b051ba08d74eb494a51eed351f42d2a4a135c9"
PRIOR_RUNNER_SHA = "e49457c0598d390822418e06b67a241a38a74afcac036a52b3e340b2248aa6b4"
DIAGNOSTIC_ID = "diagnostic_pems08_B2_parent_full_CPU_V148_k43_s92601_v1"
VARIANT = "same_checkpoint_CPU_full_V148"
L,H,C,K,N,PREFIX = 512,12,170,43,148,14284


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):
            value.update(block)
    return value.hexdigest()


def require_pin(path,digest):
    if not digest or sha(path) != digest:
        raise ValueError("Exact bound original/new metadata or source bytes changed")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def module(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    value = importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def require_full_contract(plan):
    if (plan.get("axes") != dict(L=L,H=H,C=C,K=K,V_origins=N,observed_targets=301920)
            or plan.get("materialized_prefix_end_exclusive") != PREFIX or plan.get("microbatch") != 4
            or plan.get("row_microbatch") != 128 or plan.get("checkpoint_inputs") is not False
            or plan.get("deadline_unix") != 1792222550):
        raise ValueError("Exact original full V148/H12/C170/K43 and pre-T prefix contract required")


def batch_slices(count):
    if type(count) is not int or count != N:
        raise ValueError("Exactly148 original origins required")
    return [(start,start+4) for start in range(0,N,4)]


def require_full_counters(counters):
    if counters != dict(streams=6364,row_calls=74,physical_streams=6364,physical_calls=74):
        raise ValueError("Expected37 batches4x43 streams and two128-row F calls per batch")


def compare_GPU(actual,expected,np):
    if (actual.shape != (N,H,C) or expected.shape != (N,H,C) or actual.dtype != np.float32 or expected.dtype != np.float32
            or not np.isfinite(actual).all() or not np.isfinite(expected).all()):
        raise ValueError("Two finite original normalized FP32 full148/H12/C170 forecasts required")
    passed = bool(np.allclose(actual,expected,atol=1e-5,rtol=1e-4))
    return dict(status="PASS_ALL148_GPU_CPU_PARITY" if passed else "FAIL_ALL148_GPU_CPU_PARITY",atol=1e-5,rtol=1e-4,
        max_abs=float(np.abs(actual.astype(np.float64)-expected.astype(np.float64)).max()),
        outside_tolerance_values=int((~np.isclose(actual,expected,atol=1e-5,rtol=1e-4)).sum()),total_prediction_values=301920,
        full148_checked=True,parity_is_not_an_independent_generalization_test=True)


def require_cpu_manifest(manifest,identity,parent,plan_sha):
    if (any(manifest.get(key) != value for key,value in identity.items()) or manifest.get("parent_run_id") != parent["run_id"]
            or manifest.get("parent_artifact_sha256") != parent["artifact_sha256"] or manifest.get("device") != "cpu"
            or manifest.get("inference_variant") != VARIANT or manifest.get("full_cpu_plan_sha256") != plan_sha
            or manifest.get("full_CPU_inference_completed") is not True):
        raise ValueError("Stored CPU forecast belongs to another parent/diagnostic/plan/device")


def output_paths(public,cache,run_id,output,mode):
    if not run_id or Path(run_id).name != run_id or run_id in (".","..") or mode not in ("inference","rescore"):
        raise ValueError("Explicit safe CPU diagnostic identity and supported mode required")
    namespace = public/NAME
    output = Path(output).resolve()
    if output == namespace or not output.is_relative_to(namespace):
        raise ValueError("New diagnostic output must stay in the existing priority namespace")
    if output.exists():
        raise FileExistsError("Existing diagnostic evidence must be preserved")
    paths = dict(public_prediction=namespace/"parent_full_cpu"/run_id/"V_select",
        private_prediction=cache/NAME/"parent_full_cpu"/run_id/"V_select.npz",output=output)
    if mode == "inference" and (paths["public_prediction"].exists() or paths["private_prediction"].exists()):
        raise FileExistsError("Existing CPU full-V forecasts must not be overwritten or resumed")
    return paths


def metadata(public,cache,asset,path,digest,process_api):
    path = Path(path).resolve()
    if not path.is_relative_to(public/NAME):
        raise ValueError("New ROOT-sealed parent full-CPU plan must stay in its priority namespace")
    require_pin(path,digest)
    plan = read(path);require_full_contract(plan)
    if (plan.get("schema") != "PEMS_PARENT_FULL_CPU_PLAN_v1" or plan.get("status") != "SEALED_BY_ROOT"
            or plan.get("diagnostic_run_id") != DIAGNOSTIC_ID or plan.get("runner_sha256") != sha(__file__)
            or plan.get("prior_pems_plan") != dict(public_relative_path=NAME+"/PEMS_FIXED_HEAD_PLAN.json",sha256=PRIOR_PLAN_SHA)
            or plan.get("pems_runner_sha256") != PRIOR_RUNNER_SHA
            or plan.get("limits") != dict(CPU_threads=2,max_RSS_bytes=3221225472,maximum_model_workers=2,total_owned_RSS_limit_bytes=6442450944,
                storage_limit_bytes=32212254720,storage_reserve_bytes=268435456)):
        raise ValueError("Exact same-parent CPU148 diagnostic/resource/source plan required")
    if set(plan["dependencies"]) != {NAME+"/pems_fixed_head_runner.py","priority_b2_validation_20261016_v1/priority_runner.py",
            "priority_b2_validation_20261016_v1/safe_jena_prefix.py"}:
        raise ValueError("All three unchanged reviewed helper sources must be explicitly pinned")
    pems_path = public/NAME/"pems_fixed_head_runner.py";require_pin(pems_path,PRIOR_RUNNER_SHA)
    pems = module("pems_fullCPU_original_helper",pems_path)
    helper_path = public/"priority_b2_validation_20261016_v1/priority_runner.py"
    require_pin(helper_path,plan["dependencies"]["priority_b2_validation_20261016_v1/priority_runner.py"])
    path_helper = module("pems_fullCPU_pinned_path_helper",helper_path)
    paths = dict(public=public,cache=cache,asset=asset)
    for name,value in plan["dependencies"].items():
        require_pin(path_helper.inside(public,name),value)
    for row in plan["input_bindings"]:
        require_pin(path_helper.inside(paths[row["root"]],row["relative_path"]),row["sha256"])
    for row in plan["model_snapshot_bindings"]:
        snapshot = asset/row["snapshot_relative_path"];blob = path_helper.inside(asset,row["resolved_blob_relative_path"])
        if (not snapshot.resolve().is_relative_to(asset) or not blob.is_relative_to(asset) or snapshot.resolve() != blob
                or blob.stat().st_size != row["bytes"]):
            raise ValueError("Pinned snapshot-to-blob target/path/size differs")
        require_pin(blob,row["sha256"])
    spec,helper,old_GPU = pems.metadata(public,cache,public/plan["prior_pems_plan"]["public_relative_path"],PRIOR_PLAN_SHA,process_api)
    expected_files = {spec["parent_cpu"]["model_asset_relative_path"]+"/"+name for name in ("config.json","model.safetensors")}
    if len(plan["model_snapshot_bindings"]) != 2 or {row["snapshot_relative_path"] for row in plan["model_snapshot_bindings"]} != expected_files:
        raise ValueError("Exact original small config/weight blob bindings required")
    if sha(cache/old_GPU["cache_relative_path"]) != old_GPU["archive_sha256"]:
        raise ValueError("Old GPU safe-V archive bytes changed")
    return plan,pems,spec,helper,old_GPU


def identity(plan,spec,old_GPU,plan_sha,helper):
    sources = dict(plan["dependencies"],pems_parent_full_cpu_py=plan["runner_sha256"])
    return dict(run_id=plan["diagnostic_run_id"],split="V_select",cohort="pems08:primary:V_select:stride12",
        data_sha=spec["task_contract"]["data_sha256"],scaler_sha=spec["parent"]["scaler_sha256"],
        source_sha=helper.fingerprint(sources),config_sha=plan_sha,
        model_sha=helper.fingerprint(dict(parent_model_sha=old_GPU["model_sha"],device="cpu",inference_variant=VARIANT)))


def save_CPU_forecast(arrays,comparison,ident,plan,spec,ctx,paths,foundation,batches):
    helper,np,evaluation = ctx["helper"],ctx["np"],ctx["evaluation"]
    archive = paths["private_prediction"]
    with archive.open("xb") as stream:
        np.savez_compressed(stream,**arrays)
    score,statistics = evaluation.error_statistics(arrays["pred_normalized"],arrays["target_normalized"],arrays["observed"],arrays["origins"],
        ident["run_id"],ident["split"],ident["cohort"],arrays["channel_ids"])
    if len(statistics) != 75480 or score["n_observed"] != 301920:
        raise ValueError("Exact original complete PeMS V sufficient-statistic coverage differs")
    folder = paths["public_prediction"]
    with gzip.open(folder/"PUBLIC_ERROR_STATS.csv.gz","xt",encoding="utf-8",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(statistics[0]));writer.writeheader();writer.writerows(statistics)
    helper.csv_new(folder/"SCORES.csv",[score]);helper.csv_new(folder/"INFERENCE_BATCHES.csv",batches)
    manifest = dict(**ident,parent_run_id=spec["parent"]["run_id"],parent_artifact_sha256=spec["parent"]["artifact_sha256"],
        parent_model_state_sha256=spec["parent"]["model_state_sha"],device="cpu",inference_variant=VARIANT,
        full_cpu_plan_sha256=ctx["full_plan_sha"],full_CPU_inference_completed=True,archive_sha256=sha(archive),
        cache_relative_path=archive.relative_to(ctx["cache"]).as_posix(),arrays={key:evaluation.array_fingerprint(value) for key,value in arrays.items()},
        score=score,GPU_comparison=comparison,foundation=foundation,actual_F_streams=6364,F_physical_streams=6364,F_calls=74,
        parent_checkpoint_retrained=False,post_T_development=True,new_T_numeric_values_parsed=0,independent_generalization_claimed=False,
        old_GPU_parent_manifest_sha256=spec["parent"]["V_manifest_sha256"],old_GPU_parent_archive_sha256=ctx["old_GPU"]["archive_sha256"],
        reference_contract_sha256=sha(folder/"REFERENCE_CONTRACT.json"))
    helper.new(folder/"PREDICTION_MANIFEST.json",manifest)
    return manifest


def inference(plan,pems,spec,ctx,paths,ident,asset,guard):
    import deployment
    helper,np = ctx["helper"],ctx["np"]
    task,access = pems.load_prefix(ctx["cache"],spec["task_contract"],helper,ctx["primitive"])
    helper.new(paths["output"]/"PREFIX_ACCESS_RECEIPT.json",access)
    mean,std,count = ctx["data"].fit_statistics(task,8928)
    if {key:ctx["data"].array_sha256(value) for key,value in (("mean",mean),("std",std),("count",count))} != spec["task_contract"]["A_fit_statistics_fingerprints"]:
        raise ValueError("Actual prefix A50 fingerprints differ")
    mean,std,_ = ctx["data"].fit_statistics(task,10713)
    if ctx["fit"].scaler_fingerprint(mean,std) != spec["parent"]["scaler_sha256"]:
        raise ValueError("Actual prefix A60 scaler differs")
    parent = pems.load_parent(spec,ctx["cache"],ctx)
    if ctx["torch"].get_num_threads() != 2:
        raise ValueError("Actual CPU2 execution required")
    arrays = pems.cohort(task,parent["normalization"],ctx["old_GPU"],ctx)
    old_arrays = pems.load_parent_arrays(ctx["old_GPU"],ctx["cache"],ctx)
    canonical = pems.canonical_parent_route(spec,ctx["old_GPU"],helper)
    if helper.fingerprint(canonical) != spec["parent_cpu"]["canonical_route_sha256"]:
        raise ValueError("Exact original canonical parent Route fingerprint differs")
    folder = paths["public_prediction"];folder.mkdir(parents=True,exist_ok=False)
    paths["private_prediction"].parent.mkdir(parents=True,exist_ok=False)
    helper.new(folder/"REFERENCE_CONTRACT.json",dict(**ident,parent_run_id=spec["parent"]["run_id"],device="cpu",inference_variant=VARIANT,
        arrays={key:ctx["evaluation"].array_fingerprint(value) for key,value in arrays.items()},created_before_inference=True,
        parent_artifact_sha256=spec["parent"]["artifact_sha256"],canonical_route_sha256=helper.fingerprint(canonical),expected_F_streams=6364,expected_F_calls=74))
    helper.new(paths["output"]/"INPUT_BINDINGS.json",dict(files=plan["input_bindings"],source_dependencies=plan["dependencies"],
        canonical_route=canonical,canonical_route_sha256=helper.fingerprint(canonical),old_GPU_archive_sha256=ctx["old_GPU"]["archive_sha256"],
        actual_A60_scaler_sha256=ctx["fit"].scaler_fingerprint(mean,std),old_full_task_identity_inherited_not_recomputed=True,
        actual_prefix_sha256=access["materialized_prefix_sha256"],actual_V_arrays={key:ctx["evaluation"].array_fingerprint(value) for key,value in arrays.items()}))
    guard.check()
    route = deployment.Route(canonical,task,ctx["cache"],helper.inside(asset,spec["parent_cpu"]["model_asset_relative_path"]),device="cpu")
    try:
        foundation = pems.parent_foundation(route,spec,ctx)
        if (any(route.pool.counters().values()) or route.prediction_model_sha != ctx["old_GPU"]["model_sha"]
                or helper.state_sha(route.model.state_dict(),ctx["data"]) != spec["parent"]["model_state_sha"]
                or not np.array_equal(route.mean,mean) or not np.array_equal(route.std,std)):
            raise ValueError("Fresh same-parent CPU Route identity/scaler/counters differ")
        predictions,batches = [],[]
        for start,stop in batch_slices(len(arrays["origins"])):
            guard.check();origins = arrays["origins"][start:stop]
            physical = np.stack([task["physical"][int(o)-L:int(o)] for o in origins])
            observed = np.stack([task["observed"][int(o)-L:int(o)] for o in origins])
            raw = route.predict_physical(physical,observed)
            if raw.dtype != np.float64:
                raise ValueError("Original FP64 physical inverse transform/CPU return required")
            value = ((raw-mean.astype(np.float64))/std.astype(np.float64)).astype(np.float32)
            if route.last_receipt["F_logical_streams"] != 172 or route.last_receipt["F_physical_streams"] != 172 or route.last_receipt["F_calls"] != 2:
                raise ValueError("One batch4 must execute172 logical/physical streams and2 calls")
            predictions.append(value)
            batches.append(dict(start_index=start,stop_index=stop,origins=4,F_logical_streams=172,F_physical_streams=172,F_calls=2,
                physical_input_sha256=hashlib.sha256(physical.tobytes()).hexdigest(),observation_mask_sha256=hashlib.sha256(observed.tobytes()).hexdigest()))
        require_full_counters(route.pool.counters())
        pems.parent_foundation(route,spec,ctx)
        if helper.state_sha(route.model.state_dict(),ctx["data"]) != spec["parent"]["model_state_sha"]:
            raise ValueError("Parent P/E/D changed during full CPU inference")
        arrays["pred_normalized"] = np.concatenate(predictions)
        comparison = compare_GPU(arrays["pred_normalized"],old_arrays["pred_normalized"],np)
        guard.check()
        manifest = save_CPU_forecast(arrays,comparison,ident,plan,spec,ctx,paths,foundation,batches)
        result = dict(status="COMPLETE_FULL_CPU_V148",parent_run_id=spec["parent"]["run_id"],diagnostic_run_id=ident["run_id"],V_score=manifest["score"],
            full_CPU_inference_completed=True,GPU_comparison=comparison,actual_F_streams=6364,F_physical_streams=6364,F_calls=74,
            actual_origin_batches=37,microbatch=4,row_microbatch=128,checkpoint_inputs=False,foundation_instances_created=1,
            foundation=foundation,memory=route.memory_state(),full_CPU_manifest_sha256=sha(folder/"PREDICTION_MANIFEST.json"),
            archive_sha256=manifest["archive_sha256"],independent_FP64_rescore_pending=True,old_GPU_preserved=True,
            new_T_numeric_values_parsed=0,new_training=0,new_cost_trials=0,additional_independent_seed_experiment=False,
            scope="Same limited-screen checkpoint on repeated development V; GPU parity failure is recorded separately from completed CPU inference")
        helper.new(paths["output"]/"INFERENCE_RECEIPT.json",result)
        return result
    finally:
        route.close()


def rescore(plan,pems,spec,ctx,paths,ident):
    helper,np,evaluation = ctx["helper"],ctx["np"],ctx["evaluation"]
    folder = paths["public_prediction"];manifest = read(folder/"PREDICTION_MANIFEST.json")
    require_cpu_manifest(manifest,ident,spec["parent"],ctx["full_plan_sha"])
    if manifest["cache_relative_path"] != paths["private_prediction"].relative_to(ctx["cache"]).as_posix():
        raise ValueError("CPU archive location differs from new diagnostic contract")
    require_pin(paths["private_prediction"],manifest["archive_sha256"])
    require_pin(folder/"REFERENCE_CONTRACT.json",manifest["reference_contract_sha256"])
    reference = read(folder/"REFERENCE_CONTRACT.json")
    with np.load(paths["private_prediction"],allow_pickle=False) as archive:
        if set(archive.files) != {"pred_normalized","target_normalized","observed","origins","channel_ids"}:
            raise ValueError("Unexpected saved CPU148 archive members")
        arrays = {key:archive[key] for key in archive.files}
    for key,value in arrays.items():
        if evaluation.array_fingerprint(value) != manifest["arrays"][key]:
            raise ValueError("Saved CPU148 array fingerprint changed")
    helper.validate_cohort(arrays,ctx["old_GPU"],evaluation);helper.validate_cohort(arrays,reference,evaluation)
    if any(reference.get(key) != value for key,value in ident.items()):
        raise ValueError("CPU pre-inference reference identity differs")
    old_arrays = pems.load_parent_arrays(ctx["old_GPU"],ctx["cache"],ctx)
    comparison = compare_GPU(arrays["pred_normalized"],old_arrays["pred_normalized"],np)
    if comparison != manifest["GPU_comparison"] or manifest["actual_F_streams"] != 6364 or manifest["F_calls"] != 74:
        raise ValueError("Stored all148 comparison/inference accounting differs")
    with (folder/"SCORES.csv").open(encoding="utf-8",newline="") as stream:scores = list(csv.DictReader(stream))
    with gzip.open(folder/"PUBLIC_ERROR_STATS.csv.gz","rt",encoding="utf-8",newline="") as stream:statistics = list(csv.DictReader(stream))
    if len(scores) != 1 or len(statistics) != 75480:
        raise ValueError("Exactly one complete PeMS score and75480 sufficient-statistic rows required")
    for key,value in manifest["score"].items():
        if (isinstance(value,(int,float)) and float(scores[0][key]) != value) or (not isinstance(value,(int,float)) and scores[0][key] != str(value)):
            raise ValueError("CPU score CSV differs from its completed manifest")
    result = helper.independent_rescore(arrays,manifest,statistics,scores[0],np)
    if result["n_observed"] != 301920:
        raise ValueError("Full PeMS observed count differs")
    result.update(status="PASS_FULL_CPU148_INDEPENDENT_FP64_RESCORE",diagnostic_run_id=ident["run_id"],parent_run_id=spec["parent"]["run_id"],
        GPU_comparison=comparison,actual_F_streams=0,foundation_instances_created=0,new_model_inference=False,
        source_manifest_sha256=sha(folder/"PREDICTION_MANIFEST.json"),archive_sha256=manifest["archive_sha256"],
        old_GPU_preserved=True,new_T_numeric_values_parsed=0,independent_generalization_claimed=False)
    helper.new(paths["output"]/"RESCORE_RECEIPT.json",result)
    return result


def execute(public,cache,asset,plan_path,plan_sha,mode,output,deadline):
    import psutil
    public,cache,asset = map(lambda value:Path(value).resolve(),(public,cache,asset))
    plan,pems,spec,helper,old_GPU = metadata(public,cache,asset,plan_path,plan_sha,psutil)
    if mode not in ("inference","rescore") or deadline is None or not time.time() < deadline <= plan["deadline_unix"]:
        raise ValueError("Explicit inference/rescore and original live worker deadline required")
    helper.require_environment()
    paths = output_paths(public,cache,plan["diagnostic_run_id"],output,mode)
    helper.inside(public,paths["output"].relative_to(public))
    def storage_check():
        used = sum(path.stat().st_size for root in (public,cache) for path in root.rglob("*") if path.is_file())
        if used+plan["limits"]["storage_reserve_bytes"] > plan["limits"]["storage_limit_bytes"]:
            raise ValueError("Original30GiB combined storage and256MiB reserve reached")
    storage_check();paths["output"].mkdir(parents=True,exist_ok=False)
    started = time.time();process = psutil.Process();before = process.cpu_times()
    helper.new(paths["output"]/"STARTED.json",dict(role="PARENT_B2_FULL_CPU_V148",mode=mode,parent_run_id=spec["parent"]["run_id"],
        diagnostic_run_id=plan["diagnostic_run_id"],full_cpu_plan_sha256=plan_sha,actual_pid=process.pid,actual_create_time=process.create_time(),
        deadline_unix=deadline,new_T_numeric_values_parsed=0,new_training=0))
    result,error,post_error,ctx = None,None,None,None
    class Guard:
        def check(self):
            if time.time() >= deadline:raise TimeoutError("Original worker deadline reached")
            memory = process.memory_info()
            if max(memory.rss,getattr(memory,"peak_wset",0)) > plan["limits"]["max_RSS_bytes"]:
                raise MemoryError("Original3GiB worker RSS/observed peak limit reached")
    guard = Guard()
    try:
        ctx = pems.context(public,cache,spec,PRIOR_PLAN_SHA,helper,mode == "inference")
        ctx.update(full_plan_sha=plan_sha,cache=cache,old_GPU=old_GPU)
        ident = identity(plan,spec,old_GPU,plan_sha,helper)
        guard.check()
        result = inference(plan,pems,spec,ctx,paths,ident,asset,guard) if mode == "inference" else rescore(plan,pems,spec,ctx,paths,ident)
        guard.check()
    except BaseException as failure:
        error = failure
        helper.new(paths["output"]/"ERROR.json",dict(error_class=type(failure).__name__,mode=mode,existing_results_preserved=True))
    finally:
        try:
            metadata(public,cache,asset,plan_path,plan_sha,psutil);storage_check();guard.check()
            if mode == "inference" and ctx is not None and ctx["torch"].cuda.is_initialized():raise RuntimeError("Unexpected CUDA initialization")
            if mode == "rescore" and "torch" in sys.modules:raise RuntimeError("Separate rescore must not import Torch or load a checkpoint")
        except BaseException as failure:post_error = failure
        after = process.cpu_times();memory = process.memory_info()
        helper.new(paths["output"]/"RESULT.json",dict(status="FAILED_PRESERVED" if error or post_error else (result or {}).get("status","INCOMPLETE"),
            role="PARENT_B2_FULL_CPU_V148",mode=mode,parent_run_id=spec["parent"]["run_id"],diagnostic_run_id=plan["diagnostic_run_id"],scientific_result=result,
            error_class=None if error is None else type(error).__name__,postflight_error_class=None if post_error is None else type(post_error).__name__,
            actual_F_streams=(result or {}).get("actual_F_streams"),foundation_instances_created=(result or {}).get("foundation_instances_created"),
            elapsed_process_wall_seconds=time.time()-started,actual_CPU_process_seconds=after.user+after.system-before.user-before.system,
            RSS_bytes=memory.rss,OS_peak_working_set_bytes=getattr(memory,"peak_wset",None),full_cpu_plan_sha256=plan_sha,
            original_source39_immutable9_and_input_bytes_preserved=post_error is None,old_full_task_identity_inherited_not_recomputed=True,
            new_T_numeric_values_parsed=0,new_training=0,new_cost_trials=0,ROOT_external_process_tree_guard_required=True,
            torch_imported="torch" in sys.modules,independent_generalization_claimed=False))
    if error is not None:raise error
    if post_error is not None:raise post_error
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument("--execute",action="store_true")
    for key in ("public-root","cache-root","asset-root","plan","plan-sha","output"):parser.add_argument("--"+key)
    parser.add_argument("--mode",choices=("inference","rescore"));parser.add_argument("--deadline",type=float)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="METADATA_ONLY_NOT_EXECUTED",input_reads=0,files_written=0,numpy_imported="numpy" in sys.modules,
            torch_imported="torch" in sys.modules,actual_Task_checkpoint_forward_training_cost_or_child_operations=0)))
        return 0
    if not all((args.public_root,args.cache_root,args.asset_root,args.plan,args.plan_sha,args.output,args.mode,args.deadline)):
        parser.error("Explicit ROOT-sealed newPLAN/SHA, roots, mode, deadline and fresh diagnostic output required")
    result = execute(args.public_root,args.cache_root,args.asset_root,args.plan,args.plan_sha,args.mode,args.output,args.deadline)
    print(json.dumps(dict(status=result["status"],mode=args.mode)))
    return 0 if result["status"].startswith(("COMPLETE","PASS")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
