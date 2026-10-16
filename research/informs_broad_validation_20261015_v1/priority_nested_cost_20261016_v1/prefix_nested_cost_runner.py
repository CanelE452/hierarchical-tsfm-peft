"""One ROOT-controlled prefix-only Beijing nested-PCA CPU cost condition."""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import zipfile

NAME = "priority_nested_cost_20261016_v1"
OLD = "postT_V_only_cpu_multik_20261016_v1"
OLD_PLAN_SHA = "2051573b8dee48200eb34e96e37e7af0d197442619ea1ba52a36a80ad4e27aec"
ARTIFACT_SHA = "daf45933e8df040390c3c7237e892c16134e61ec1c23e8cebec00a64f29faf73"
PRIMITIVE = "priority_b2_validation_20261016_v1/safe_jena_prefix.py"
PREFIX, L, H, C = 16830, 512, 48, 11
BOUNDS = dict(A_fit=[0,10519],A_stop=[10519,12622],M_cal=[12622,14726],V_select=[14726,16830],T_final=[16830,21038])
HEADERS = dict(physical=dict(descr="<f4",shape=[28051,11],fortran_order=False),
    mask=dict(descr="|b1",shape=[28051,11],fortran_order=False),
    times=dict(descr="<U19",shape=[28051],fortran_order=False),columns=dict(descr="<U5",shape=[11],fortran_order=False))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def new(path,value):
    with Path(path).open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def relative(root,name):
    root = Path(root).resolve();path = root/name
    if Path(name).is_absolute() or path.is_symlink() or any(p.is_symlink() or getattr(p,"is_junction",lambda:False)() for p in path.parents):
        raise ValueError("Explicit nonsymlink relative path required")
    path = path.resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("Path escapes declared root")
    return path


def module(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    value = importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def read_prefix_members(archive,primitive,rows,headers):
    if type(rows) is not int or rows < 0 or set(headers) != {"physical","mask","times","columns"}:
        raise ValueError("Explicit nonnegative prefix and exact four source members required")
    for name,expected in headers.items():
        info = archive.getinfo(name+".npy")
        with archive.open(name+".npy") as stream:
            actual = primitive.read_header(stream)
        if (any(actual[key] != value for key,value in expected.items())
                or actual["fortran_order"] is not False
                or info.file_size != actual["header_total_bytes"]+actual["payload_bytes"]
                or name != "columns" and rows > actual["shape"][0]):
            raise ValueError("Pinned C-order primitive source header or permitted prefix differs")
    values,evidence = {},[]
    for name in headers:
        values[name],entry = primitive.read_member(archive,name+".npy",None if name == "columns" else rows)
        evidence.append(entry)
    return values,evidence


def require_quiet(proof,now):
    captured = proof.get("captured_unix")
    if (proof.get("status") != "QUIET_CPU_COST_WINDOW" or proof.get("active_model_workers") != []
            or proof.get("active_cost_workers") != [] or isinstance(captured,bool) or not isinstance(captured,(int,float))
            or not math.isfinite(captured) or not 0 <= now-captured <= 120):
        raise ValueError("Fresh ROOT quiet proof with both active worker lists empty required")


def require_condition(condition,batch):
    k = condition.get("latent_k")
    if (type(k) is not int or k not in (3,6,11) or condition.get("kind") != "nested"
            or condition.get("run_id") != f"postT_cpu_beijing_E4_NESTED_PCA_k{k}_s92601_v1"
            or type(batch) is not int or batch not in (1,8)):
        raise ValueError("Exactly one existing nested K3/6/11 and B1/B8 condition required")


def canonical_spec(condition,old_plan,binding):
    require_condition(condition,1)
    return dict(run_id=condition["run_id"],dataset="beijing",family="NESTED_PCA",fold="primary",
        artifact_cache_relative_path="ordered/primary_beijing/nested_pca.npz",artifact_sha256=ARTIFACT_SHA,
        foundation_sha=old_plan["foundation_tensor_sha256"],canonical_fit_end=12622,latent_k=condition["latent_k"],
        ordered_manifest_public_relative_path=binding["manifest_relative"],ordered_manifest_sha256=binding["manifest_sha256"],model_size="small")


def require_cpu_replay(proof,receipt,k,verifier_sha):
    expected = dict(status="PASS",mode="replay",run_id=receipt["run_id"],actual_checkpoint_forward_in_this_command=True,
        checkpoint_replay_call_attempted=True,checkpoint_replay_receipt_returned=True,cuda_initialized=False,
        new_T_target_or_scoring=False,original_source_registry_selection_T_and_captured_inputs_preserved=True,
        error_class=None,postflight_error_class=None,latent_k=k,plan_sha256=OLD_PLAN_SHA,verifier_sha256=verifier_sha)
    result = proof.get("scientific_result",{})
    if (any(proof.get(key) != value for key,value in expected.items()) or result.get("executed_now") is not True
            or any(result.get(key) != value for key,value in receipt.items()) or receipt.get("status") != "PASS"
            or receipt.get("actual_F_streams") != 6*k or receipt.get("F_physical_streams") != 6*k
            or receipt.get("F_calls") != 2 or receipt.get("atol") != 1e-5 or receipt.get("rtol") != 1e-4
            or not math.isfinite(receipt.get("max_abs",float("nan")))):
        raise ValueError("Existing actual six-origin CPU forward/source closure is not established")


def require_replay_archive(receipt,run_id,actual_sha):
    if receipt.get("cache_relative_path") != f"route_replays/{run_id}/V_subset.npz" or receipt.get("archive_sha256") != actual_sha:
        raise ValueError("Old PASS replay receipt differs from its exact private archive bytes")


def validate_trials(rows,k,batch):
    expected = {(phase,trial) for phase,count in (("warmup",10),("timed",20),("stage_probe",20)) for trial in range(count)}
    seen = set()
    for row in rows:
        key = row["phase"],row["trial"]
        if (key in seen or key not in expected or row["batch_size"] != batch or row["F_logical_streams"] != batch*k
                or row["F_physical_streams"] != batch*k or row["F_calls"] != 1 or row["row_microbatch"] != 128
                or row["stage_instrumentation"] != (row["phase"] == "stage_probe")):
            raise ValueError("Actual cost trial phase/streams/physical call accounting differs")
        seen.add(key)
    if seen != expected or len({row["prediction_sha256"] for row in rows}) != 1:
        raise ValueError("Complete deterministic original10/20/20 trial ledger required")


def metadata(public,cache,asset,plan_path,plan_sha,k,batch):
    path = Path(plan_path).resolve()
    if not path.is_relative_to(public/NAME) or not plan_sha or sha(path) != plan_sha:
        raise ValueError("Explicit ROOT-sealed new cost PLAN path/SHA required")
    plan = read(path)
    limits = dict(CPU_threads=2,max_RSS_bytes=3221225472,storage_limit_bytes=32212254720,storage_reserve_bytes=268435456,deadline_unix=1792222550)
    if (plan.get("schema") != "PREFIX_NESTED_CPU_COST_PLAN_v1" or plan.get("status") != "SEALED_BY_ROOT"
            or plan.get("namespace") != NAME or plan.get("limits") != limits or plan.get("source_headers") != HEADERS
            or plan.get("protocol") != dict(warmups=10,timed_trials=20,stage_trials=20,row_microbatch=128,checkpoint_inputs=False)
            or plan.get("runner_sha256") != sha(__file__) or plan.get("original_plan_sha256") != OLD_PLAN_SHA):
        raise ValueError("Exact original nested condition/resource/source contract required")
    required_dependencies = {OLD+"/"+name for name in ("prepare_multik_cpu_costs.py","verify_multik_outputs.py","multik_runner.py","test_multik_runner.py")}|{PRIMITIVE}
    if set(plan["dependencies"]) != required_dependencies:
        raise ValueError("All five explicitly pinned existing dependencies required")
    for name,digest in plan["dependencies"].items():
        if sha(relative(public,name)) != digest:
            raise ValueError("Bound existing dependency changed")
    prep = module("nested_original_cost_prepare",relative(public,OLD+"/prepare_multik_cpu_costs.py"))
    verifier,old_runner,old_plan,condition,state = prep.metadata(public,cache,asset,
        f"postT_cpu_beijing_E4_NESTED_PCA_k{k}_s92601_v1",OLD_PLAN_SHA)
    require_condition(condition,batch)
    if not state["ready"]:
        raise ValueError("Selected original nested artifacts/replay metadata absent")
    matches = [row for row in plan["conditions"] if row["run_id"] == condition["run_id"] and row["latent_k"] == k]
    if len(matches) != 1:
        raise ValueError("One ROOT-bound selected existing output required")
    selected = matches[0]
    roots = dict(public=public,cache=cache,asset=asset)
    for row in plan["shared_bindings"]+selected["bindings"]:
        if sha(relative(roots[row["root"]],row["relative_path"])) != row["sha256"]:
            raise ValueError("Captured existing selected input/model/source bytes changed")
    model_bindings = plan["model_snapshot_bindings"]
    expected_files = {old_plan["model_asset_relative_path"]+"/"+name for name in ("config.json","model.safetensors")}
    if len(model_bindings) != 2 or {row["snapshot_relative_path"] for row in model_bindings} != expected_files:
        raise ValueError("Exact local snapshot config and model blob bindings required")
    for row in model_bindings:
        snapshot = asset/row["snapshot_relative_path"]
        blob = relative(asset,row["resolved_blob_relative_path"])
        if (not snapshot.resolve().is_relative_to(asset) or snapshot.resolve() != blob or blob.stat().st_size != row["bytes"]
                or sha(blob) != row["sha256"]):
            raise ValueError("Pinned local HF snapshot file/explicit blob identity differs")
    return plan,selected,prep,verifier,old_runner,old_plan,condition


def safe_task(asset,old_plan,primitive,np):
    old = old_plan["task_contract"];meta = old["metadata"]
    source = relative(asset,meta["source_reference"])
    contract_path = relative(asset,str(Path(meta["source_reference"]).with_name("contract.json")).replace("\\","/"))
    if (old["shape"] != [21038,C] or old["primary_split_ranges"] != BOUNDS or old["L"] != L or old["H"] != H
            or source.stat().st_size != meta["source_bytes"] or sha(source) != meta["source_sha256"]
            or sha(contract_path) != meta["source_contract_sha256"] or read(contract_path)["bounds"]["fit"] != [0,21038]):
        raise ValueError("Exact original permitted Beijing source ancestry differs")
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate raw member names forbidden")
        values,members = read_prefix_members(archive,primitive,PREFIX,HEADERS)
    physical = values["physical"].astype(np.float32,copy=True);observed = values["mask"].astype(bool,copy=True)
    times = values["times"].astype("datetime64[ns]").astype(np.int64);columns = values["columns"].tolist()
    if columns != old["columns"] or not np.isfinite(physical[observed]).all() or not np.all(np.diff(times) == 3600*1000000000):
        raise ValueError("Actual source prefix channel order/time grid/observed values differ")
    physical[~observed] = np.nan
    hashes = dict(physical=primitive.array_sha(physical),observed=primitive.array_sha(observed),times=primitive.array_sha(times),
        columns=hashlib.sha256(json.dumps(columns,ensure_ascii=False).encode()).hexdigest())
    prefix_sha = hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    task = dict(name="beijing",dataset="beijing",physical=physical,observed=observed,times=times,columns=columns,L=L,H=H,
        interval_seconds=3600,period=24,data_sha256=old["data_sha256"],array_sha256=copy.deepcopy(old["array_sha256"]),
        data_identity_provenance="INHERITED_NOT_RECOMPUTED",materialized_prefix_sha256=prefix_sha,materialized_prefix_array_sha256=hashes,
        split_ranges=copy.deepcopy(BOUNDS),secondary_split_ranges=copy.deepcopy(old["secondary_split_ranges"]),metadata=copy.deepcopy(meta))
    return task,dict(status="MATERIALIZED_PRE_T_PREFIX_ONLY",source_asset_relative_path=meta["source_reference"],source_sha256=meta["source_sha256"],
        source_rows=28051,original_permitted_rows=21038,materialized_rows=PREFIX,members=members,new_T_numeric_values_parsed=0,
        inherited_full_task_data_sha256=old["data_sha256"],inherited_full_task_array_sha256=old["array_sha256"],
        inherited_identity_status="INHERITED_NOT_RECOMPUTED",materialized_prefix_sha256=prefix_sha,materialized_prefix_array_sha256=hashes,
        original_primary_split_ranges=BOUNDS,raw_values_published=False,
        IO_scope="Opaque whole-file compressed SHA and ZIP buffering may read tail bytes; only rows0:16830 are numerically decoded")


def check_environment():
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "-1"
            or any(os.environ.get(key) != "2" for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"))
            or any(os.environ.get(key) != "1" for key in ("HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE"))
            or any(key.startswith("INFORMS_GPU_") for key in os.environ)):
        raise ValueError("Explicit offline CPU2 environment without a GPU lease required")


def execute(public,cache,asset,plan_path,plan_sha,k,batch,deadline,quiet_path,quiet_sha,output):
    public,cache,asset,output = map(lambda value:Path(value).resolve(),(public,cache,asset,output))
    plan,selected,prep,verifier,old_runner,old_plan,condition = metadata(public,cache,asset,plan_path,plan_sha,k,batch)
    check_environment()
    quiet_path = Path(quiet_path).resolve()
    if not quiet_path.is_relative_to(cache) or not quiet_sha or sha(quiet_path) != quiet_sha:
        raise ValueError("Explicit ROOT quiet proof inside private cache with exact SHA required")
    started = time.time();require_quiet(read(quiet_path),started)
    if deadline is None or not started < deadline <= plan["limits"]["deadline_unix"]:
        raise ValueError("Explicit live original worker deadline required")
    if output.exists() or output == public/NAME or not output.is_relative_to(public/NAME):
        raise ValueError("Fresh output inside the separate cost namespace required")
    relative(public,output.relative_to(public))
    used = sum(path.stat().st_size for root in (public,cache) for path in root.rglob("*") if path.is_file())
    if used+plan["limits"]["storage_reserve_bytes"] > plan["limits"]["storage_limit_bytes"]:
        raise ValueError("Original combined storage cap and256MiB reserve reached")
    output.mkdir(parents=True,exist_ok=False)
    new(output/"STARTED.json",dict(run_id=condition["run_id"],latent_k=k,batch=batch,device="cpu",plan_sha256=plan_sha,
        quiet_proof_sha256=quiet_sha,actual_pid=os.getpid(),started_unix=started,new_T_target_operations=0,actual_training=0))
    result,error,post_error,route,phase = None,None,None,None,"selected_metadata_binding"
    import psutil
    process = psutil.Process();before_cpu = process.cpu_times()
    class Guard:
        def check(self):
            if time.time() >= deadline:
                raise TimeoutError("Original worker deadline reached")
            info = process.memory_info()
            if max(info.rss,getattr(info,"peak_wset",0)) > plan["limits"]["max_RSS_bytes"]:
                raise MemoryError("Original3GiB RSS/observed peak limit reached")
            if sha(quiet_path) != quiet_sha:
                raise ValueError("ROOT quiet proof changed during cost worker")
    guard = Guard()
    try:
        values,score,paths,bindings = verifier.prepare_inputs(old_runner,old_plan,public,cache,condition["run_id"],condition)
        canonical = canonical_spec(condition,old_plan,selected)
        binding = dict(condition=condition,parent=values["nested_contract"],manifest=values["manifest"],
            artifact_relative="ordered/primary_beijing/nested_pca.npz",manifest_relative=selected["manifest_relative"],
            manifest_sha256=selected["manifest_sha256"],receipt_relative=None,receipt_sha256=None)
        cost_spec = prep.cost_spec(canonical,"small",binding,old_plan,fingerprint)
        post,private = public/OLD,cache/OLD
        reference = read(post/"route_replays"/condition["run_id"]/"PRE_INFERENCE_REFERENCE.json")
        receipt = read(post/"route_replays"/condition["run_id"]/"REPLAY_RECEIPT.json")
        require_replay_archive(receipt,condition["run_id"],sha(relative(private,f"route_replays/{condition['run_id']}/V_subset.npz")))
        prep.validate_replay(old_runner,old_plan,binding,canonical,reference,receipt)
        require_cpu_replay(read(relative(public,selected["cpu_replay_result_public_relative_path"])),receipt,k,
            plan["dependencies"][OLD+"/verify_multik_outputs.py"])
        if fingerprint(canonical) != selected["canonical_route_sha256"]:
            raise ValueError("Canonical route differs from ROOT-bound original pre-inference fingerprint")
        phase = "safe_prefix_materialization"
        sys.path.insert(0,str(public))
        import numpy as np
        import data_contract as data
        import evaluation
        import fit_base as fit
        import deployment
        import measure_costs
        import torch
        if any(Path(value.__file__).resolve() != public/(value.__name__+".py") for value in (data,evaluation,fit,deployment,measure_costs)):
            raise ValueError("Original scientific modules must resolve to the pinned source root")
        torch.set_num_threads(2)
        if torch.cuda.is_initialized() or torch.get_num_threads() != 2:
            raise ValueError("Actual CPU2/no-CUDA execution required")
        primitive = module("nested_safe_primitive",relative(public,PRIMITIVE))
        guard.check()
        task,access = safe_task(asset,old_plan,primitive,np)
        new(output/"PREFIX_ACCESS_RECEIPT.json",access)
        mean,std,count = data.fit_statistics(task,10519)
        if {key:data.array_sha256(value) for key,value in (("mean",mean),("std",std),("count",count))} != old_plan["task_contract"]["A_fit_statistics_fingerprints"]:
            raise ValueError("Actual A50 prefix statistics differ")
        mean,std,_ = data.fit_statistics(task,12622)
        if fit.scaler_fingerprint(mean,std) != old_plan["task_contract"]["refit_scaler_sha256"]:
            raise ValueError("Actual A60 prefix scaler differs")
        origins = data.origins(BOUNDS["V_select"],L,H,H)
        truths = [data.target_window(task,int(o),mean,std) for o in origins]
        arrays = dict(origins=origins,observed=np.stack([value[1] for value in truths]),
            target_normalized=np.stack([value[0] for value in truths]),channel_ids=np.asarray(task["columns"],dtype=str))
        if len(origins) != 43 or any(evaluation.array_fingerprint(value) != values["manifest"]["arrays"][key] for key,value in arrays.items()):
            raise ValueError("Actual safe prefix V43 targets/masks/origins/channel identity differs")
        nested = verifier.validate_nested_arrays(old_runner,old_plan,values["manifest"],values["nested_contract"],paths["artifact"],np,evaluation,fit)
        new(output/"INPUT_BINDINGS.json",dict(files=plan["shared_bindings"]+selected["bindings"],original_prepare_bindings=bindings,
            canonical_route=canonical,canonical_route_sha256=fingerprint(canonical),cost_route=cost_spec,
            actual_prefix_V_arrays={key:evaluation.array_fingerprint(value) for key,value in arrays.items()},actual_A60_scaler_sha256=fit.scaler_fingerprint(mean,std),
            nested_fixed_basis_evidence=nested,quiet_proof_sha256=quiet_sha,original_CPU6_replay_reused=True,
            old_CPU2_evidence="Pinned verifier source sets2threads and replay_call passes CPU; old actual RESULT has no explicit CPU_threads field",
            no_source_load_task_no_truncated_split_or_catalog_call=True))
        phase = "original_frozen_CPU_route_loading"
        guard.check()
        route = deployment.Route(cost_spec,task,private,relative(asset,old_plan["model_asset_relative_path"]),device="cpu")
        foundation = route.pool.foundation
        def frozen_check():
            guard.check()
            if (route.pool.instances_created != 1 or foundation is None or foundation.device != "cpu"
                    or foundation.row_microbatch != 128 or foundation.checkpoint_inputs is not False
                    or foundation.initial_hash != old_plan["foundation_tensor_sha256"]
                    or foundation.instance_norm_source_sha256 != old_plan["instance_norm_source_sha256"]
                    or fit._model_sha(foundation) != old_plan["foundation_tensor_sha256"]
                    or evaluation.array_fingerprint(route.model.E.detach().cpu().numpy()) != nested["basis"]
                    or evaluation.array_fingerprint(route.model.center.detach().cpu().numpy()) != nested["center"]
                    or not np.array_equal(route.mean,mean) or not np.array_equal(route.std,std)):
                raise ValueError("Exact frozen small/row128/fixed basis/scaler identity differs")
            foundation.assert_frozen_unchanged()
        frozen_check()
        if any(route.pool.counters().values()):
            raise ValueError("Fresh cost Route may not consume any cached/extra forecasts")
        chosen = origins[:batch]
        physical = np.stack([task["physical"][int(o)-L:int(o)] for o in chosen])
        observed = np.stack([task["observed"][int(o)-L:int(o)] for o in chosen])
        phase = "physical_CPU_cost_trials"
        rows = measure_costs.measure_balanced([route],physical,observed,warmups=10,timed_trials=20,stage_trials=20,lease=guard)
        validate_trials(rows,k,batch)
        if route.pool.counters() != dict(streams=50*batch*k,row_calls=50,physical_streams=50*batch*k,physical_calls=50):
            raise ValueError("Actual total50trial foundation counters differ")
        frozen_check()
        path = output/"COST_TRIALS.csv.gz";measure_costs._write_trials(path,rows)
        result = dict(status="COMPLETE_PROTOCOL",run_id=condition["run_id"],device="cpu",CPU_threads=torch.get_num_threads(),batch_size=batch,latent_k=k,
            origins=chosen.tolist(),L=L,H=H,C=C,warmups=10,timed_trials=20,stage_trials=20,actual_trial_rows=len(rows),trials_sha256=sha(path),
            summaries=measure_costs.summarize(rows),memory=route.memory_state(),actual_F_streams=50*batch*k,F_physical_streams=50*batch*k,F_calls=50,
            expected_F_streams_per_trial=batch*k,expected_F_calls_per_trial=1,foundation_instances_created=1,
            foundation_tensor_sha256=foundation.initial_hash,instance_norm_source_sha256=foundation.instance_norm_source_sha256,
            artifact_sha256=ARTIFACT_SHA,canonical_route_sha256=fingerprint(canonical),accuracy_manifest_sha256=selected["manifest_sha256"],
            physical_input_sha256=hashlib.sha256(physical.tobytes()).hexdigest(),observation_mask_sha256=hashlib.sha256(observed.tobytes()).hexdigest(),
            physical_output_dtype="float64",inverse_transform_dtype="float64",cached_Z_or_forecast_reads=0,new_T_numeric_values_parsed=0,
            actual_training=0,decoder="Fixed basis transpose plus center; no learned lag decoder or current anchor",
            stage_probe_is_intrusive_separate_scope=True,training_peak_included=False,old_CPU6_replay_reused=True,
            new_full_V_accuracy_evaluation=False,actual_deployment_forward_trials=50,
            historical_old_replay_full_deserialization_scope_preserved=True,external_ROOT_process_tree_quiet_RSS_guard_required=True,
            model_load_seconds=route.load_seconds,quiet_proof_sha256=quiet_sha)
        new(output/"COST_RECEIPT.json",result)
    except BaseException as failure:
        error = failure
        new(output/"ERROR.json",dict(error_class=type(failure).__name__,phase=phase,existing_files_preserved=True))
    finally:
        if route is not None:
            try:
                route.close()
            except BaseException as failure:
                if error is None:error = failure
        try:
            metadata(public,cache,asset,plan_path,plan_sha,k,batch)
            if sha(quiet_path) != quiet_sha:
                raise ValueError("Quiet proof changed")
            require_quiet(read(quiet_path),started)
            guard.check()
            used = sum(path.stat().st_size for root in (public,cache) for path in root.rglob("*") if path.is_file())
            if used+plan["limits"]["storage_reserve_bytes"] > plan["limits"]["storage_limit_bytes"]:
                raise ValueError("Original combined storage cap reached during cost worker")
            if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
                raise ValueError("Unexpected CUDA initialization")
        except BaseException as failure:
            post_error = failure
        after_cpu = process.cpu_times();memory = process.memory_info()
        new(output/"RESULT.json",dict(status="FAILED_PRESERVED" if error or post_error else (result or {}).get("status","INCOMPLETE"),
            run_id=condition["run_id"],latent_k=k,batch=batch,scientific_result=result,error_class=None if error is None else type(error).__name__,
            postflight_error_class=None if post_error is None else type(post_error).__name__,plan_sha256=plan_sha,quiet_proof_sha256=quiet_sha,
            elapsed_process_wall_seconds=time.time()-started,actual_CPU_process_seconds=after_cpu.user+after_cpu.system-before_cpu.user-before_cpu.system,
            RSS_bytes=memory.rss,OS_peak_working_set_bytes=getattr(memory,"peak_wset",None),actual_F_streams=(result or {}).get("actual_F_streams"),
            foundation_instances_created=(result or {}).get("foundation_instances_created"),new_T_numeric_values_parsed=0,new_training=0,
            original_source39_immutable9_and_selected_inputs_preserved=post_error is None,old_full_task_identity_inherited_not_recomputed=True,
            independent_generalization_claimed=False,ROOT_owns_external_guard_and_job_dispatch=True))
    if error is not None:raise error
    if post_error is not None:raise post_error
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    for name in ("public-root","cache-root","asset-root","plan","plan-sha","quiet-proof","quiet-sha","output"):
        parser.add_argument("--"+name)
    parser.add_argument("--k",type=int,choices=(3,6,11));parser.add_argument("--batch",type=int,choices=(1,8))
    parser.add_argument("--deadline",type=float)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="METADATA_ONLY_NOT_EXECUTED",input_reads=0,files_written=0,numpy_imported="numpy" in sys.modules,
            torch_imported="torch" in sys.modules,actual_Task_checkpoint_forward_training_cost_or_child_operations=0)))
        return 0
    if not all((args.public_root,args.cache_root,args.asset_root,args.plan,args.plan_sha,args.quiet_proof,args.quiet_sha,args.output,args.k,args.batch,args.deadline)):
        parser.error("Explicit ROOT-sealed PLAN/SHA, roots, K/B, quiet proof/SHA, fresh output and original deadline required")
    result = execute(args.public_root,args.cache_root,args.asset_root,args.plan,args.plan_sha,args.k,args.batch,args.deadline,args.quiet_proof,args.quiet_sha,args.output)
    print(json.dumps(dict(status=result["status"],k=args.k,batch=args.batch)))
    return 0 if result["status"] == "COMPLETE_PROTOCOL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
