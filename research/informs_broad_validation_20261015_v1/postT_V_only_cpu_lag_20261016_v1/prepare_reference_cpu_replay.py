"""Prepare opaque private shadows, then explicitly replay one CPU V reference.

Default metadata performs no copy/model/array operation. Preparation and actual
six-origin replay are separate modes; original scientific APIs stay unchanged.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
import uuid

HERE = Path(__file__).resolve().parent
COST_HELPER_SHA = "62a89fde89402847e762f2a54fdcbd148158ddeb2a89679cc8e7555196828884"
NAME = "postT_V_only_cpu_lag_20261016_v1"
SOURCE_NAMES = ("route_catalog.py","route_predictions.py","deployment.py","data_contract.py","evaluation.py","training.py","networks.py","gpu_runtime.py","ridge_baseline.py","combination.py","fit_base.py")


def load_reference():
    path = HERE/"prepare_reference_cpu_costs.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != COST_HELPER_SHA: raise ValueError("Frozen original-reference metadata helper changed")
    spec = importlib.util.spec_from_file_location("frozen_original_reference_cost_metadata",path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def safe(path):
    path = Path(path)
    if any(p.is_symlink() or getattr(p,"is_junction",lambda:False)() for p in [path]+list(path.parents)):
        raise ValueError("Shadow/source symlink or junction is not permitted")
    return path.resolve()


def inside(root, relative):
    root,relative = safe(root),Path(relative)
    if relative.is_absolute() or ".." in relative.parts: raise ValueError("Explicit safe relative path required")
    result = safe(root/relative)
    if result == root or not result.is_relative_to(root): raise ValueError("Path escapes declared root")
    return result


def metadata(public,cache,asset,run_id,plan_sha=None):
    helper = load_reference()
    state = helper.metadata(public,cache,asset,run_id,plan_sha)
    return helper,state


def inventory(helper,state,public,cache):
    if not state["ready"]: raise ValueError("Only the explicitly requested reference must have approved metadata/assets")
    plan,binding = state["plan"],state["binding"]
    records = [dict(root="public",relative_path="EXPERIMENT_REGISTRY.csv",sha256=plan["immutable_public"]["EXPERIMENT_REGISTRY.csv"]),
               dict(root="public",relative_path=binding["manifest_relative"],sha256=binding["metadata_pins"][binding["manifest_relative"]]),
               dict(root="cache",relative_path=binding["manifest"]["cache_relative_path"],sha256=binding["manifest"]["archive_sha256"])]
    if binding["receipt"] is None:
        records.append(dict(root="public",relative_path="PROBE_ACTUAL_GPU.json",sha256=helper.PROBE_SHA))
    else:
        receipt = binding["receipt"]
        records.extend([dict(root="public",relative_path=binding["receipt_relative"],sha256=binding["metadata_pins"][binding["receipt_relative"]]),
                        dict(root="cache",relative_path=receipt["artifact_cache_relative_path"],sha256=receipt["artifact_sha256"])])
    roots = dict(public=public,cache=cache)
    for row in records:
        path = inside(roots[row["root"]],row["relative_path"])
        if not path.is_file(): raise FileNotFoundError("Approved source is absent")
        row["bytes"] = path.stat().st_size
    return records


def copy_one(helper,source,target,row):
    source,target = safe(source),safe(target)
    if source.stat().st_size != row["bytes"] or helper.sha(source) != row["sha256"]: raise ValueError("Original opaque source bytes changed")
    if target.exists(): raise FileExistsError("Existing shadow file is preserved; no replacement or alias")
    target.parent.mkdir(parents=True,exist_ok=True)
    with source.open("rb") as original,target.open("xb") as copied: shutil.copyfileobj(original,copied,length=1048576)
    if target.stat().st_size != row["bytes"] or helper.sha(target) != row["sha256"] or helper.sha(source) != row["sha256"]:
        raise ValueError("Source/copy bytes changed during opaque transfer")


def check_files(helper,records,sources,destinations=None):
    for row in records:
        for roots in (sources,destinations) if destinations is not None else (sources,):
            path = inside(roots[row["root"]],row["relative_path"])
            if not path.is_file() or path.stat().st_size != row["bytes"] or helper.sha(path) != row["sha256"]:
                raise ValueError("Original or shadow artifact/archive/metadata bytes differ")


def empty_replay_paths(public,cache,run_id):
    paths = (public/"route_replays"/run_id,cache/"route_replays"/run_id)
    if any(path.exists() for path in paths): raise FileExistsError("Fresh CPU proof forbids every existing replay receipt/reference/archive")


def environment(cache,asset):
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or any(key.startswith("INFORMS_GPU_") for key in os.environ)
            or any(os.environ.get(key) != "2" for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"))
            or any(os.environ.get(key) != "1" for key in ("HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE"))
            or Path(os.environ.get("INFORMS_ASSET_ROOT","")).resolve() != asset
            or Path(os.environ.get("INFORMS_CACHE_ROOT","")).resolve() != cache/NAME):
        raise ValueError("Explicit original-asset/separate-cache CPU2 offline environment required")


def admission(public,cache,post,output,plan,deadline,extra_bytes):
    output = safe(output)
    if output == post or not output.is_relative_to(post) or output.exists(): raise FileExistsError("Fresh public diagnostic output required")
    if deadline is None or not time.time() < deadline <= plan["deadline_unix"]: raise ValueError("Live deadline inside original campaign wall required")
    used = sum(path.stat().st_size for root in (public,cache) for path in root.rglob("*") if path.is_file())
    reserve = 256*1024**2
    if used+extra_bytes+reserve > plan["storage_limit_bytes"] or shutil.disk_usage(cache).free < extra_bytes+reserve:
        raise ValueError("Original combined 30GiB/free-space admission failed")


class Watch:
    def __init__(self,runner,plan,deadline,output):
        import psutil
        self.process,self.stop = psutil.Process(),threading.Event()
        def guard():
            while not self.stop.wait(.25):
                rss = self.process.memory_info().rss
                if rss > plan["max_RSS_bytes"] or time.time() >= deadline:
                    runner.new(output/"RESOURCE_STOP.json",dict(status="RESOURCE_STOP",rss_bytes=rss,deadline_unix=deadline));os._exit(98)
        self.thread = threading.Thread(target=guard,daemon=True);self.thread.start()

    def close(self):
        self.stop.set();self.thread.join(timeout=2)


def prepare(public,cache,asset,plan_sha,run_id,output,deadline):
    helper,state = metadata(public,cache,asset,run_id,plan_sha)
    if plan_sha != helper.PLAN_SHA: raise ValueError("Explicit frozen PLAN SHA required")
    runner,plan = helper.load_runner(),state["plan"]
    records = inventory(helper,state,public,cache)
    environment(cache,asset)
    post,private = safe(public/NAME),safe(cache/NAME)
    admission(public,cache,post,output,plan,deadline,sum(row["bytes"] for row in records))
    token = "reference_replay_"+str(time.time_ns())+"_"+uuid.uuid4().hex
    shadow = inside(private,"reference_replay_shadows/"+token)
    if shadow.exists(): raise FileExistsError("Fresh per-run private shadow required")
    shadow.mkdir(parents=True,exist_ok=False);output.mkdir(parents=True,exist_ok=False)
    source_sha = helper.sha(__file__)
    runner.new(output/"STARTED.json",dict(mode="prepare",run_id=run_id,helper_sha256=source_sha,plan_sha256=plan_sha,started_unix=time.time(),actual_replay=0))
    watch = Watch(runner,plan,deadline,output)
    roots = dict(public=public,cache=cache)
    destinations = dict(public=shadow/"public",cache=shadow/"cache")
    copied,failures,error,post_error,preserved,config = [],[],None,None,False,None
    try:
        for path in destinations.values(): path.mkdir(exist_ok=False)
        for row in records:
            if time.time() >= deadline: raise TimeoutError("Opaque shadow copy deadline reached")
            copy_one(helper,inside(roots[row["root"]],row["relative_path"]),inside(destinations[row["root"]],row["relative_path"]),row)
            copied.append(row)
        check_files(helper,records,roots,destinations)
        empty_replay_paths(destinations["public"],destinations["cache"],run_id)
        config = dict(schema="ORIGINAL_REFERENCE_CPU_REPLAY_SHADOW_v1",token=token,run_id=run_id,plan_sha256=plan_sha,
            helper_sha256=source_sha,metadata_helper_sha256=COST_HELPER_SHA,original_roots={key:str(value) for key,value in dict(public=public,cache=cache,asset=asset).items()},
            files=records,original_source39=plan["source39"],original_immutable_public=plan["immutable_public"],
            selected_metadata=state["binding"]["metadata_pins"],source_api_root=str(public),model_asset_relative_path=plan["model_asset_relative_path"],
            truth_arrays={key:value for key,value in state["binding"]["manifest"]["arrays"].items() if key != "pred_normalized"},
            source_sha=helper.fingerprint({name:plan["source39"][name] for name in SOURCE_NAMES}),cpu_device="cpu",atol=1e-5,rtol=1e-4,replay_origins=6)
        runner.new(shadow/"SHADOW_CONFIG.json",config)
    except Exception as failure:
        error = type(failure).__name__
        failures.append(runner.save_private_failure(private,cache,"reference_replay",run_id,"opaque_preparation",failure))
    finally:
        try:
            _,after = metadata(public,cache,asset,run_id,plan_sha)
            if not after["ready"] or after["binding"] != state["binding"] or helper.sha(__file__) != source_sha:
                raise ValueError("Original source/registry/selection/V metadata/helper changed")
            check_files(helper,records,roots)
            preserved = True
        except Exception as failure:
            post_error = type(failure).__name__
            failures.append(runner.save_private_failure(private,cache,"reference_replay",run_id,"preparation_postflight",failure))
        watch.close()
        success = error is None and post_error is None and preserved and config is not None
        config_path = shadow/"SHADOW_CONFIG.json"
        result = dict(status="PREPARED_FRESH_PRIVATE_CPU_REPLAY_SHADOW" if success else "FAILED_PRESERVED",run_id=run_id,
            plan_sha256=plan_sha,helper_sha256=source_sha,shadow_cache_relative_root=shadow.relative_to(cache).as_posix(),
            shadow_config_cache_relative_path=config_path.relative_to(cache).as_posix(),shadow_config_sha256=helper.sha(config_path) if config_path.is_file() else None,
            files=records,copied_files=len(copied),copied_bytes=sum(row["bytes"] for row in copied),source_code_or_foundation_model_copies=0,
            selected_adapter_or_ridge_artifact_copies=sum(row["root"] == "cache" and row["relative_path"] == (state["binding"]["receipt"] or {}).get("artifact_cache_relative_path") for row in copied),
            source_api_original_public_relative_path="route_predictions.py",original_source39=plan["source39"],original_immutable_public=plan["immutable_public"],
            original_inputs_preserved=preserved,error_class=error,postflight_error_class=post_error,private_failures=failures,
            actual_subprocesses_started=0,actual_model_construction_or_training_or_forward_or_array_deserialization=0,
            new_T_target_or_scoring=False,old_cached_replay_artifacts_copied=0,
            replay_command_arguments=["{PYTHON}","-X","utf8","-u","-B","{ORIGINAL_PUBLIC_ROOT}/"+NAME+"/prepare_reference_cpu_replay.py",
                "--execute","--mode","replay","--run-id",run_id,"--public-root","{ORIGINAL_PUBLIC_ROOT}","--cache-root","{ORIGINAL_CACHE_ROOT}",
                "--asset-root","{ASSET_ROOT}","--plan-sha",plan_sha,"--shadow-config","{ORIGINAL_CACHE_ROOT}/"+config_path.relative_to(cache).as_posix(),
                "--shadow-config-sha",helper.sha(config_path) if config_path.is_file() else "UNAVAILABLE","--deadline","{DEADLINE}","--output","{NEW_PUBLIC_REPLAY_OUTPUT}"],
            ROOT_must_enforce_prior_chain_closure_and_quiet_window=True,external_CPU_tree_guard_required=True,created_unix=time.time())
        runner.new(output/"SHADOW_PREPARATION.json",result)
    return result


def load_shadow(helper,state,public,cache,asset,run_id,config_path,config_sha):
    config_path = safe(config_path)
    private = safe(cache/NAME)
    if not config_sha or helper.sha(config_path) != config_sha: raise ValueError("Explicit immutable private shadow config SHA required")
    config = helper.load_runner().read(config_path)
    token = config.get("token","")
    if not re.fullmatch(r"reference_replay_[0-9]+_[0-9a-f]{32}",token): raise ValueError("Approved fresh shadow identity required")
    shadow = inside(private,"reference_replay_shadows/"+token)
    if config_path != shadow/"SHADOW_CONFIG.json": raise ValueError("Config must belong to this exact private shadow")
    expected = dict(schema="ORIGINAL_REFERENCE_CPU_REPLAY_SHADOW_v1",run_id=run_id,plan_sha256=helper.PLAN_SHA,helper_sha256=helper.sha(__file__),metadata_helper_sha256=COST_HELPER_SHA,
        original_roots={key:str(value) for key,value in dict(public=public,cache=cache,asset=asset).items()},source_api_root=str(public),
        original_source39=state["plan"]["source39"],original_immutable_public=state["plan"]["immutable_public"],selected_metadata=state["binding"]["metadata_pins"],
        model_asset_relative_path=state["plan"]["model_asset_relative_path"],files=inventory(helper,state,public,cache),
        source_sha=helper.fingerprint({name:state["plan"]["source39"][name] for name in SOURCE_NAMES}),
        truth_arrays={key:value for key,value in state["binding"]["manifest"]["arrays"].items() if key != "pred_normalized"},cpu_device="cpu",atol=1e-5,rtol=1e-4,replay_origins=6)
    if any(config.get(key) != value for key,value in expected.items()): raise ValueError("Shadow sources/roots/selected identities or fixed CPU replay contract differ")
    return config,shadow,dict(public=shadow/"public",cache=shadow/"cache")


class ReplayRuntime:
    def __init__(self,public):
        sys.path.insert(0,str(public))
        import torch
        import data_contract
        import fit_base
        import route_predictions
        for module in (data_contract,fit_base,route_predictions):
            if Path(module.__file__).resolve().parent != public: raise ValueError("Scientific API must come from pinned original source root")
        torch.set_num_threads(2)
        if torch.cuda.is_initialized(): raise RuntimeError("CPU-only replay required")
        self.data,self.fit,self.api,self.torch = data_contract,fit_base,route_predictions.replay_route,torch

    def task(self,runner,plan,cache):
        task = self.data.load_task("jena",cache_dir=cache)
        return task,runner.validate_task(task,plan,self.data,self.fit)

    def replay(self,run_id,task,roots,model,deadline):
        return self.api(run_id,task,roots["cache"],roots["public"],model,device="cpu",deadline=deadline,lease=None,batch_size=4,atol=1e-5,rtol=1e-4)


def replay_evidence(attempted,result):
    returned = attempted and result is not None
    executed = result.get("executed_now") if returned else None
    return dict(actual_checkpoint_forward_in_this_command=executed if isinstance(executed,bool) else (None if attempted else False),
                checkpoint_replay_call_attempted=attempted,checkpoint_replay_receipt_returned=returned)


def validate_replay(result,run_id,width):
    if (result.get("run_id") != run_id or result.get("executed_now") is not True or len(result.get("origins",[])) != 6 or len(set(result["origins"])) != 6
            or result.get("actual_F_streams") != 6*width or result.get("expected_F_streams") != 6*width or result.get("F_physical_streams") != 6*width
            or result.get("atol") != 1e-5 or result.get("rtol") != 1e-4 or result.get("T_final_accessed") is not False):
        raise ValueError("Fresh CPU forward, original six-origin/6K counter and fixed parity tolerance required")


def replay(public,cache,asset,plan_sha,run_id,output,deadline,config_path,config_sha,*,_runtime=None):
    helper,state = metadata(public,cache,asset,run_id,plan_sha)
    if plan_sha != helper.PLAN_SHA or not state["ready"]: raise ValueError("One approved original reference and exact PLAN required")
    runner,plan = helper.load_runner(),state["plan"]
    config,shadow,destinations = load_shadow(helper,state,public,cache,asset,run_id,config_path,config_sha)
    environment(cache,asset)
    post,private = safe(public/NAME),safe(cache/NAME)
    admission(public,cache,post,output,plan,deadline,0)
    check_files(helper,config["files"],dict(public=public,cache=cache),destinations)
    empty_replay_paths(destinations["public"],destinations["cache"],run_id)
    runner.new(shadow/"CPU_REPLAY_ATTEMPT.json",dict(run_id=run_id,config_sha256=config_sha,helper_sha256=helper.sha(__file__),created_unix=time.time()))
    output.mkdir(parents=True,exist_ok=False)
    runner.new(output/"STARTED.json",dict(mode="replay",run_id=run_id,device="cpu",plan_sha256=plan_sha,shadow_config_sha256=config_sha,actual_pid=os.getpid(),started_unix=time.time()))
    watch = Watch(runner,plan,deadline,output)
    result,error,post_error,failures,attempted,preserved = None,None,None,[],False,False
    try:
        runtime = _runtime if _runtime is not None else ReplayRuntime(public)
        task,task_binding = runtime.task(runner,plan,cache)
        runner.new(output/"TASK_BINDING.json",task_binding)
        empty_replay_paths(destinations["public"],destinations["cache"],run_id)
        attempted = True
        result = runtime.replay(run_id,task,destinations,asset/plan["model_asset_relative_path"],deadline)
        validate_replay(result,run_id,state["binding"]["F_width"])
        reference_path = destinations["public"]/"route_replays"/run_id/"PRE_INFERENCE_REFERENCE.json"
        receipt_path = reference_path.with_name("REPLAY_RECEIPT.json")
        reference,receipt = runner.read(reference_path),runner.read(receipt_path)
        expected_archive = "route_replays/"+run_id+"/V_subset.npz"
        if (dict(receipt,executed_now=True) != result or reference.get("source_sha") != config["source_sha"]
                or reference.get("standard_manifest_sha256") != state["binding"]["metadata_pins"][state["binding"]["manifest_relative"]]
                or reference.get("standard_archive_sha256") != state["binding"]["manifest"]["archive_sha256"]
                or reference.get("created_before_inference") is not True or result.get("reference_sha") != helper.fingerprint(reference)
                or result.get("cache_relative_path") != expected_archive
                or helper.sha(inside(destinations["cache"],expected_archive)) != result.get("archive_sha256")):
            raise ValueError("Fresh replay receipt/source/standard-V/pre-inference identity differs")
        if result["status"] != "PASS": raise ValueError("CPU parity failed at unchanged original tolerance; numerical mismatch is preserved")
    except Exception as failure:
        error = type(failure).__name__
        failures.append(runner.save_private_failure(private,cache,"reference_replay",run_id,"actual_CPU_replay",failure))
    finally:
        try:
            _,after = metadata(public,cache,asset,run_id,plan_sha)
            if not after["ready"] or after["binding"] != state["binding"] or helper.sha(config_path) != config_sha or helper.sha(__file__) != config["helper_sha256"]:
                raise ValueError("Original selected inputs/source/config/helper changed")
            check_files(helper,config["files"],dict(public=public,cache=cache),destinations)
            preserved = True
        except Exception as failure:
            post_error = type(failure).__name__
            failures.append(runner.save_private_failure(private,cache,"reference_replay",run_id,"replay_postflight",failure))
        watch.close()
        success = error is None and post_error is None and preserved and result is not None and result.get("status") == "PASS"
        proof = dict(status="PASS_FRESH_CPU_SIX_ORIGIN_PARITY" if success else "FAILED_OR_PARITY_MISMATCH_PRESERVED",run_id=run_id,device="cpu",
            scientific_result=result,**replay_evidence(attempted,result),plan_sha256=plan_sha,helper_sha256=helper.sha(__file__),shadow_config_sha256=config_sha,
            original_source_registry_scores_cost_T_and_selected_inputs_preserved=preserved,error_class=error,postflight_error_class=post_error,private_failures=failures,
            shadow_cache_relative_root=shadow.relative_to(cache).as_posix(),fresh_shadow_initial_replay_paths_absent=True,
            old_cached_replay_receipts_or_archives_copied=False,atol=1e-5,rtol=1e-4,registered_V_origins=43,replay_origins=6,
            expected_F_streams=6*state["binding"]["F_width"],CPU_threads=2,max_RSS_bytes=plan["max_RSS_bytes"],deadline_unix=deadline,
            original_training_status=state["binding"]["original_training_status"],new_training=False,new_T_target_or_scoring=False,
            original_GPU_minimum_gap_completed=False,independent_generalization_claimed=False,cost_timing_performed=False,
            receipt_scope="Original six selected V origins only; CPU parity does not certify full-V rerun, retraining or independent generalization",
            asset_scope="Original hash-bound Jena trainval FIT loader may decompress old TUNE/VAL and campaign T bytes; protected old TEST file is not opened",
            source_api_scope="Original source __file__ remains in original R; shadow contains only approved metadata/artifacts/standard V",
            actual_pid=watch.process.pid,actual_create_time=watch.process.create_time(),finished_unix=time.time())
        runner.new(output/"CPU_REPLAY_RESULT.json",proof)
    return proof


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("public","cache","asset"): parser.add_argument("--"+key+"-root")
    parser.add_argument("--run-id");parser.add_argument("--plan-sha");parser.add_argument("--execute",action="store_true")
    parser.add_argument("--mode",choices=("prepare","replay"));parser.add_argument("--deadline",type=float);parser.add_argument("--output")
    parser.add_argument("--shadow-config");parser.add_argument("--shadow-config-sha")
    args = parser.parse_args(argv)
    roots = (args.public_root,args.cache_root,args.asset_root)
    if args.execute:
        if not all(roots+(args.run_id,args.plan_sha,args.mode,args.deadline,args.output)): parser.error("Explicit one-ID mode, roots, PLAN, deadline and fresh output required")
        public,cache,asset = (safe(value) for value in roots)
        if args.mode == "prepare": result = prepare(public,cache,asset,args.plan_sha,args.run_id,safe(args.output),args.deadline)
        else:
            if not args.shadow_config or not args.shadow_config_sha: parser.error("Replay requires immutable private --shadow-config and --shadow-config-sha")
            result = replay(public,cache,asset,args.plan_sha,args.run_id,safe(args.output),args.deadline,Path(args.shadow_config),args.shadow_config_sha)
        print(json.dumps(dict(status=result["status"],run_id=result["run_id"],new_T_scoring=False)))
        return 0 if result["status"] in {"PREPARED_FRESH_PRIVATE_CPU_REPLAY_SHADOW","PASS_FRESH_CPU_SIX_ORIGIN_PARITY"} else 1
    helper = load_reference()
    states = []
    if any(roots) or args.run_id:
        if not all(roots): parser.error("All original roots are required for metadata checks")
        public,cache,asset = (safe(value) for value in roots)
        for run_id in [args.run_id] if args.run_id else helper.REFERENCES:
            _,state = metadata(public,cache,asset,run_id,args.plan_sha)
            states.append(dict(run_id=run_id,ready=state["ready"],missing=state["missing"],route_error_class=state["route_error_class"]))
    print(json.dumps(dict(status="METADATA_ONLY_REFERENCE_CPU_REPLAY_PREPARATION",requested_states=states,numpy_imported="numpy" in sys.modules,
        torch_imported="torch" in sys.modules,files_written=0,actual_copy_or_model_or_array_or_replay_or_child_operations=0,helper_sha256=helper.sha(__file__))),flush=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
