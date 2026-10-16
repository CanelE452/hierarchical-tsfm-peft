"""One completed Beijing condition: full saved-V FP64 audit or CPU V6 replay.

Default is stdlib metadata only. Original scientific APIs and tolerances remain
unchanged. Replay uses one batch of six, separately from old batch-four replays.
"""
import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
RUNNER_SHA = "e11c34918862a698e962cf2a3c151ea6f7484032985f52d25cb7220da6b5d37b"
PLAN_SHA = "04acc4aabffd02a7266f16fee5c6a3884313fac6ca4b5b46ef0db45eb2fe5152"
REGISTRY_SHA = "fc5f4012a6ced7a1849fcd6474f67d81b600dadaa0014f56df4cda4b8a140f29"
PREDICTION_SOURCES = ("prediction_cache.py","fit_base.py","data_contract.py","training.py","evaluation.py","networks.py","gpu_runtime.py")


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""): result.update(block)
    return result.hexdigest()


def plain_path(path):
    path = Path(path).absolute()
    if any(item.is_symlink() for item in [path]+list(path.parents)):
        raise ValueError("Nonsymlink explicit paths required before resolution")
    return path.resolve()


def read(path):
    path = plain_path(path)
    if path.stat().st_size > 8*1024**2: raise ValueError("Small JSON metadata limit exceeded")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def new(path,value):
    with Path(path).open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")


def load_runner():
    path = HERE/"required_minima_runner.py"
    if sha(path) != RUNNER_SHA: raise ValueError("Sealed six-condition runner changed")
    spec = importlib.util.spec_from_file_location("required_minima_frozen_verification_runner",path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def metadata(runner,public,cache,asset,plan_sha):
    if plan_sha != PLAN_SHA: raise ValueError("Exact ROOT-sealed required-minima PLAN SHA required")
    plan = runner.metadata(public,cache,asset,plan_sha)
    if plan["runner_sha256"] != RUNNER_SHA or plan["postT_registry_sha256"] != REGISTRY_SHA:
        raise ValueError("Sealed runner/registry identity differs")
    return plan


def choose(runner,plan,run_id,mode):
    if mode not in {"rescore","replay"}: raise ValueError("Only saved V rescore or checkpoint V replay supported")
    return runner.choose(plan,run_id)


def require_new_output(runner,output,public):
    output,public = Path(output).absolute(),plain_path(public)
    if output == public or not output.is_relative_to(public): raise ValueError("Fresh output below the separate public namespace required")
    runner.relative(public,output.relative_to(public))
    if output.exists(): raise FileExistsError("Existing verification outputs must be preserved")


def registry_projection(plan,public):
    rows = []
    for condition in plan["conditions"]:
        path = public/"runs"/condition["run_id"]/"BASE_FIT_RECEIPT.json"
        status,reason = "NO_COMPLETE_BASE_FIT_RECEIPT","No completed BASE receipt; may be pending, running or partial"
        if path.is_file():
            try:
                receipt = read(path)
                if (not isinstance(receipt,dict) or any(not isinstance(receipt.get(key),str) or not receipt[key].strip()
                        for key in ("run_id","family","status"))):
                    raise ValueError("BASE receipt requires an object with nonempty string identity and status")
                if receipt.get("run_id") != condition["run_id"] or receipt.get("family") != condition["family"]:
                    status,reason = "INVALID_BASE_RECEIPT_IDENTITY","Receipt scientific identity differs"
                else:
                    status,reason = receipt["status"],"Actual BASE status; convergence and scientific success remain separate"
            except (OSError,ValueError,KeyError) as error:
                status,reason = "UNREADABLE_BASE_FIT_RECEIPT","Metadata incomplete: "+type(error).__name__
        rows.append(dict(run_id=condition["run_id"],execution_status=status,reason=reason))
    return rows


def ensure_registry_alias(runner,public,plan):
    source = runner.relative(public,plan["postT_registry_relative_path"])
    expected = plan["postT_registry_sha256"]
    if sha(source) != expected: raise ValueError("Separate sealed registry changed")
    target = runner.relative(public,"EXPERIMENT_REGISTRY.csv")
    created = not target.exists()
    if created:
        with target.open("xb") as stream: stream.write(source.read_bytes())
    if sha(target) != expected or sha(source) != expected: raise ValueError("Existing route registry alias differs; no file overwritten")
    return dict(file=target.name,sha256=expected,created_now=created,source_file=source.name,scientific_rows_or_execution_status_changed=False)


def check_bindings(public,cache,bindings):
    roots = dict(public=public,cache=cache)
    runner = load_runner()
    for label,files in bindings.items():
        for name,digest in files.items():
            if sha(runner.relative(roots[label],name)) != digest: raise ValueError("Captured fit, V or replay evidence changed")


def capture_paths(runner,root,paths):
    result = {}
    for path in paths:
        path = runner.relative(root,Path(path).relative_to(root))
        if not path.is_file(): raise FileNotFoundError("Required exact evidence file absent")
        result[path.relative_to(root).as_posix()] = sha(path)
    return result


def validate_source(runner,source,base,search,refit,condition,plan):
    contract,spec = plan["task_contract"],condition["spec"]
    files = {key:plan["source39"][key] for key in runner.FIT_SOURCES}
    axes = dict(L=512,H=48,C=11,columns=contract["columns"],fold="primary",source_array_fingerprints=contract["array_sha256"])
    expected = dict(spec=spec,files=files,source_sha=runner.fingerprint(files),config_sha=runner.fingerprint(dict(spec=spec,axes=axes)),
        data_sha=contract["data_sha256"],model_sha=plan["foundation_tensor_sha256"])
    if source != expected or base.get("source_sha") != expected["source_sha"]:
        raise ValueError("Selected fit exact CPU recipe/source/axes/data/foundation binding differs")
    for stage,scaler in ((search,contract["search_scaler_sha256"]),(refit,contract["refit_scaler_sha256"])):
        expected_stage = {key:expected[key] for key in ("source_sha","config_sha","data_sha","model_sha")}
        expected_stage["scaler_sha"] = scaler
        actual = stage.get("contract") or {}
        if (set(actual) != set(expected_stage)|{"initialization_sha"} or any(actual.get(key)!=value for key,value in expected_stage.items())
                or not re.fullmatch("[0-9a-f]{64}",actual.get("initialization_sha", ""))):
            raise ValueError("Search/refit normalization, initialization or scientific contract differs")


def validate_saved_v(runner,manifest,reference,base,source,condition,plan):
    contract,k = plan["task_contract"],condition["latent_k"]
    expected_model = runner.fingerprint(dict(artifact_sha=base["artifact_sha256"],model_state_sha=base["model_state_sha"],normalization=contract["refit_scaler_sha256"]))
    prediction_files = {name:plan["source39"][name] for name in PREDICTION_SOURCES}
    expected = dict(run_id=condition["run_id"],split="V_select",cohort="beijing:primary:V_select:stride48",
        source_sha=runner.fingerprint(dict(deployment_sources=prediction_files,training_source_sha=source["source_sha"])),
        config_sha=runner.fingerprint(dict(parent_spec=condition["spec"],dense=False,fold="primary",role="base_parent")),
        data_sha=contract["data_sha256"],scaler_sha=contract["refit_scaler_sha256"],model_sha=expected_model)
    if any(manifest.get(key) != value for key,value in expected.items()): raise ValueError("Saved full V task, model, scaler or cohort differs")
    if any(not manifest.get(key) or reference.get(key) != manifest[key] for key in
            ("run_id","split","cohort","source_sha","config_sha","data_sha","model_sha","scaler_sha")):
        raise ValueError("Independent pre-inference reference identity differs")
    expected_meta = dict(artifact_sha256=base["artifact_sha256"],canonical_scaler_fit_end=12622,parent_fit_end=12622,
        parent_normalization_fit_end=12622,family=condition["family"],fold="primary",dense=False,role="base_parent",
        training_source_sha=source["source_sha"],source_files=prediction_files,reused_prediction_cache=False)
    if any(manifest["metadata"].get(key) != value for key,value in expected_meta.items()): raise ValueError("Fresh A60 parent and V metadata differ")
    shapes = dict(pred_normalized=[43,48,11],target_normalized=[43,48,11],observed=[43,48,11],origins=[43],channel_ids=[11])
    if set(manifest["arrays"]) != set(shapes) or set(reference["arrays"]) != set(shapes)-{"pred_normalized"}:
        raise ValueError("Fixed full-V array keys differ")
    if any(manifest["arrays"][key]["shape"] != shape for key,shape in shapes.items()): raise ValueError("Full V43/H48/C11 array dimensions differ")
    if any(reference["arrays"][key] != manifest["arrays"][key] for key in reference["arrays"]): raise ValueError("Truth, mask, origin or channel reference differs")
    expected_cost = dict(run_id=condition["run_id"],batch=43,latent_k=k,actual_F_streams=43*k,expected_F_streams=43*k,F_calls=1)
    if manifest["cost_record"] != expected_cost or reference.get("expected_F_streams") != 43*k:
        raise ValueError("Saved full V43 K-width ledger differs; its F_calls=1 is original bundle bookkeeping")


def prepare_inputs(runner,plan,old_public,old_cache,run_id,condition):
    public,private = old_public/runner.NAME,old_cache/runner.NAME
    run,folder = public/"runs"/run_id,public/"predictions"/run_id/"V_select"
    paths = dict(fit_receipt=run/"BASE_FIT_RECEIPT.json",source_contract=run/"SOURCE_CONTRACT.json",
        search=run/"search/RECEIPT.json",refit=run/"refit/RECEIPT.json",manifest=folder/"PREDICTION_MANIFEST.json",
        reference=folder/"REFERENCE_CONTRACT.json",score=folder/"SCORES.csv")
    values = {key:read(runner.relative(old_public,path.relative_to(old_public))) for key,path in paths.items() if key!="score"}
    if any(not isinstance(value,dict) for value in values.values()): raise ValueError("Selected fit and V metadata must be JSON objects")
    base,source,manifest = values["fit_receipt"],values["source_contract"],values["manifest"]
    runner.validate_complete_fit(base,values["search"],values["refit"],condition,plan)
    validate_source(runner,source,base,values["search"],values["refit"],condition,plan)
    validate_saved_v(runner,manifest,values["reference"],base,source,condition,plan)
    with paths["score"].open(encoding="utf-8-sig",newline="") as stream: scores=list(csv.DictReader(stream))
    if len(scores)!=1: raise ValueError("Exactly one original per-run V score required")
    score = scores[0]
    for key,value in manifest["score"].items():
        if isinstance(value,(int,float)):
            if not math.isfinite(float(score[key])) or float(score[key])!=value: raise ValueError("Original V score CSV numeric value differs")
        elif score[key] != str(value): raise ValueError("Original V score CSV identity differs")
    paths["statistics"] = runner.relative(public,manifest["public_stats_relative_path"])
    paths["artifact"] = runner.relative(private,base["artifact_cache_relative_path"])
    paths["archive"] = runner.relative(private,manifest["cache_relative_path"])
    if paths["artifact"] != private/"runs"/run_id/"selected_bundle.pt" or paths["archive"] != private/"predictions"/run_id/"V_select.npz":
        raise ValueError("Selected fresh run artifact or saved V archive path differs")
    if sha(paths["artifact"])!=base["artifact_sha256"] or sha(paths["archive"])!=manifest["archive_sha256"]:
        raise ValueError("Selected artifact or saved V archive bytes changed")
    bindings=dict(public=capture_paths(runner,old_public,[p for key,p in paths.items() if key not in {"artifact","archive"}]),
        cache=capture_paths(runner,old_cache,[paths["artifact"],paths["archive"]]))
    return values,score,paths,bindings


def rescore_call(api,manifest,reference,score,statistics,registry,expected_ids,private):
    return api(manifest,reference,score,statistics,registry,expected_ids,[manifest["cost_record"]],private)


def replay_call(api,run_id,task,private,public,model_path,deadline):
    return api(run_id,task,private,public,model_path,device="cpu",deadline=deadline,lease=None,batch_size=6,atol=1e-5,rtol=1e-4)


def validate_replay(value,run_id,k):
    expected = dict(run_id=run_id,actual_F_streams=6*k,expected_F_streams=6*k,F_physical_streams=6*k,
        F_calls=math.ceil(6*k/128),atol=1e-5,rtol=1e-4,T_final_accessed=False,independent_training_rerun=False,model_size="small",
        scope="actual_physical_input_selected_checkpoint_V_subset_replay")
    if (any(value.get(key)!=item for key,item in expected.items()) or len(value.get("origins",[]))!=6
            or len(set(value["origins"]))!=6 or not isinstance(value.get("executed_now"),bool)):
        raise ValueError("Actual original V6 single-batch CPU scope, logical/physical F counters or fixed tolerances differ")


def forward_facts(attempted,value):
    returned = attempted and value is not None
    forward = False if not attempted else (value.get("executed_now") if returned and isinstance(value.get("executed_now"),bool) else None)
    return dict(actual_checkpoint_forward_in_this_command=forward,checkpoint_replay_call_attempted=attempted,checkpoint_replay_receipt_returned=returned)


def validate_replay_bindings(runner,value,values,paths,private,public):
    run_id=values["fit_receipt"]["run_id"]
    receipt_path=runner.relative(public,"route_replays/"+run_id+"/REPLAY_RECEIPT.json")
    reference_path=runner.relative(public,"route_replays/"+run_id+"/PRE_INFERENCE_REFERENCE.json")
    receipt,reference=read(receipt_path),read(reference_path)
    if receipt != {key:item for key,item in value.items() if key!="executed_now"}:
        raise ValueError("Returned replay receipt differs from original immutable public receipt")
    expected_binding={"route":dict(cache_relative_path=values["fit_receipt"]["artifact_cache_relative_path"],sha256=values["fit_receipt"]["artifact_sha256"])}
    expected_standard=paths["manifest"].relative_to(public).as_posix()
    if (value["selected_artifact_bindings"] != expected_binding or reference["artifact_bindings"] != expected_binding
            or value["standard_manifest_public_relative_path"]!=expected_standard or value["standard_manifest_sha256"]!=sha(paths["manifest"])
            or reference["standard_manifest_sha256"]!=sha(paths["manifest"]) or reference["standard_archive_sha256"]!=values["manifest"]["archive_sha256"]
            or value["reference_sha"]!=runner.fingerprint(reference) or reference.get("created_before_inference") is not True
            or reference["run_id"]!=run_id or reference["split"]!="V_select" or reference["origins"]!=value["origins"]
            or value["route_sha"]!=reference["route_sha"] or reference["expected_F_streams"]!=value["expected_F_streams"]
            or reference.get("selection_policy")!="first/middle/last plus 3 remaining origins from seed92601; sorted unique"):
        raise ValueError("Actual replay is not bound to this exact selected artifact and full saved V")
    archive=runner.relative(private,value["cache_relative_path"])
    if archive != private/"route_replays"/run_id/"V_subset.npz" or sha(archive)!=value["archive_sha256"]:
        raise ValueError("Actual replay archive path or bytes differ")
    return receipt_path,reference_path,archive


def require_original_module(module,public):
    if Path(module.__file__).resolve()!=public/(module.__name__+".py"):
        raise ValueError("Original hash-pinned scientific module root required")


def execute(old_public,old_cache,asset,plan_sha,mode,run_id,deadline,output):
    runner=load_runner()
    plan=metadata(runner,old_public,old_cache,asset,plan_sha)
    condition=choose(runner,plan,run_id,mode)
    if deadline is None or not time.time()<deadline<=plan["deadline_unix"]: raise ValueError("A live deadline within the sealed campaign required")
    runner.environment(old_cache,asset)
    public,private=old_public/runner.NAME,old_cache/runner.NAME
    require_new_output(runner,output,public)
    used=sum(path.stat().st_size for root in (old_public,old_cache) for path in root.rglob("*") if path.is_file())
    if used+plan["admission_reserve_bytes"]>plan["storage_limit_bytes"]: raise ValueError("Original combined storage cap/reserve reached")
    utility=runner.load_dependency(old_public,runner.UTILITY_PATH,runner.UTILITY_SHA,"required_verifier_stdlib_utility")
    output.mkdir(parents=True,exist_ok=False)
    source_sha=sha(__file__)
    new(output/"STARTED.json",dict(mode=mode,run_id=run_id,plan_sha256=plan_sha,verifier_sha256=source_sha,actual_pid=os.getpid(),
        started_unix=time.time(),new_training=False,new_T_scoring=False,requested_device="cpu",CPU_threads=2,replay_batch_size=6))
    error_class,postflight_error,value,alias=None,None,None,None
    failures,bindings=[],dict(public={},cache={})
    replay_attempted,source_post=False,False
    process,torch,watchdog=None,None,None
    resource_before,resource_after=None,None
    stop=threading.Event();started=time.perf_counter()
    try:
        sys.path.insert(0,str(old_public))
        import psutil
        import torch
        process=psutil.Process();torch.set_num_threads(2);resource_before=utility.rss_record(process)
        if torch.get_num_threads()!=2: raise RuntimeError("Actual Torch CPU thread count differs")
        def guard():
            while not stop.wait(.25):
                rss=process.memory_info().rss
                if rss>plan["max_RSS_bytes"] or time.time()>=deadline:
                    new(output/"RESOURCE_STOP.json",dict(status="RESOURCE_STOP",rss_bytes=rss,deadline_unix=deadline));os._exit(98)
        watchdog=threading.Thread(target=guard,daemon=True);watchdog.start()
        if torch.cuda.is_initialized() or process.memory_info().rss>plan["max_RSS_bytes"]: raise RuntimeError("CPU/RSS contract violated before verification")
        values,score,paths,bindings=prepare_inputs(runner,plan,old_public,old_cache,run_id,condition)
        projection=registry_projection(plan,public)
        new(output/"INPUT_BINDINGS.json",dict(files_sha256_by_original_root=bindings,registry_projection=projection,
            expected_run_ids=[row["run_id"] for row in plan["conditions"]],selected_artifact_sha256=values["fit_receipt"]["artifact_sha256"],
            saved_V_origins=43,horizon=48,num_channels=11,latent_k=condition["latent_k"],full_three_epoch_search_verified=True,
            selected_fresh_A60_refit_epoch=values["fit_receipt"]["selected_epoch"],zero_selected_refit_is_allowed=True,
            original_training_status=values["fit_receipt"].get("training_status"),convergence_established=False,stage="POST_T_V_ONLY_DEVELOPMENT"))
        if mode=="rescore":
            import audit_predictions
            require_original_module(audit_predictions,old_public)
            with gzip.open(paths["statistics"],"rt",encoding="utf-8",newline="") as stream: statistics=list(csv.DictReader(stream))
            value=rescore_call(audit_predictions.audit_prediction_bundle,values["manifest"],values["reference"],score,statistics,
                projection,[row["run_id"] for row in plan["conditions"]],private)
            if (value.get("status")!="PASS" or value.get("n_origins")!=43 or value.get("horizon")!=48 or value.get("num_channels")!=11
                    or value.get("checkpoint_replay") is not False or value.get("atol")!=1e-9 or value.get("rtol")!=1e-7):
                raise ValueError("Original independent full-V FP64 rescore scope or tolerance differs")
        else:
            alias=ensure_registry_alias(runner,public,plan)
            bindings["public"][(public/alias["file"]).relative_to(old_public).as_posix()]=alias["sha256"]
            import data_contract as data
            import fit_base
            import route_predictions
            for module in (data,fit_base,route_predictions):require_original_module(module,old_public)
            task=data.load_task("beijing",cache_dir=old_cache)
            validator=runner.load_dependency(old_public,runner.TASK_VALIDATOR_PATH,runner.TASK_VALIDATOR_SHA,"required_verifier_task_validator")
            task_binding=validator.validate_task(task,dict(task_contracts=dict(beijing=plan["task_contract"])),"beijing",data,fit_base)
            new(output/"TASK_BINDING.json",task_binding)
            replay_attempted=True
            value=replay_call(route_predictions.replay_route,run_id,task,private,public,asset/plan["model_asset_relative_path"],deadline)
            validate_replay(value,run_id,condition["latent_k"])
            extra=validate_replay_bindings(runner,value,values,paths,private,public)
            bindings["public"].update(capture_paths(runner,old_public,extra[:2]))
            bindings["cache"].update(capture_paths(runner,old_cache,extra[2:]))
            new(output/"REPLAY_BINDINGS.json",dict(files_sha256_by_original_root=bindings,executed_now=value["executed_now"],
                requested_device="cpu",CPU_threads=2,replay_batch_size=6,original_F_row_microbatch=128,
                cached_receipt_has_no_device_field=True,cache_hit_is_not_new_CPU_execution=True,
                foundation_checks="Original route foundation hash binding and FoundationPool.close frozen-weight/parameter-gradient checks"))
        if torch.cuda.is_initialized(): raise RuntimeError("Unexpected CUDA initialization")
    except Exception as error:
        error_class=type(error).__name__
        failures.append(utility.save_private_failure(private,old_cache,mode,run_id,"verification",error))
    finally:
        try:
            metadata(runner,old_public,old_cache,asset,plan_sha)
            if sha(__file__)!=source_sha: raise ValueError("Verifier source changed")
            check_bindings(old_public,old_cache,bindings)
            source_post=True
        except Exception as error:
            postflight_error=type(error).__name__
            failures.append(utility.save_private_failure(private,old_cache,mode,run_id,"postflight",error))
        stop.set()
        if watchdog is not None:watchdog.join(timeout=1)
        if process is not None:resource_after=utility.rss_record(process)
        cuda=bool(torch.cuda.is_initialized()) if torch is not None else None
        facts=forward_facts(replay_attempted,value)
        passed=error_class is None and postflight_error is None and source_post and value is not None and value.get("status")=="PASS" and cuda is False
        new(output/"RESULT.json",dict(status="PASS" if passed else "FAILED_OR_NUMERICAL_MISMATCH_PRESERVED",run_id=run_id,mode=mode,
            scientific_result=value,**facts,actual_fresh_CPU_replay_verified=passed and mode=="replay" and facts["actual_checkpoint_forward_in_this_command"] is True,
            saved_full_V_rescoring=mode=="rescore",subset_checkpoint_replay=mode=="replay",training_rerun=False,
            full_V_origin_count=43,replay_origin_count=6 if mode=="replay" else 0,dataset="beijing",latent_k=condition["latent_k"],horizon=48,num_channels=11,
            requested_device="cpu",CPU_threads=2,actual_Torch_CPU_threads=torch.get_num_threads() if torch is not None else None,
            replay_batch_size=6 if mode=="replay" else None,F_row_microbatch=128,
            cached_replay_receipt_device_unknown=mode=="replay" and facts["actual_checkpoint_forward_in_this_command"] is False,
            prior_batch_four_replay_is_not_same_call_count_protocol=True,original_saved_V_F_calls_is_bundle_bookkeeping=True,
            error_class=error_class,postflight_error_class=postflight_error,private_failures=failures,metadata_alias=alias,
            actual_pid=os.getpid(),actual_create_time=process.create_time() if process is not None else None,
            original_source_registry_selection_T_and_captured_inputs_preserved=source_post,cuda_initialized=cuda,
            new_T_target_or_scoring=False,post_T_development=True,independent_generalization_claimed=False,
            original_GPU_minimum_gap_completed=False,original_registry_statuses_updated=False,
            archive_deserialization_scope="Original safeFIT Task may deserialize/hash T bytes; only A/V computation requested, no new T target or evaluation API.",
            parent_must_enforce_prior_queue_closure_cost_quiet_and_owned_process_closure=True,
            parent_must_report_forced_exit_without_final_result=True,elapsed_process_wall_seconds=time.perf_counter()-started,
            resource_before=resource_before,resource_after=resource_after,plan_sha256=plan_sha,verifier_sha256=source_sha,finished_unix=time.time()))
    return 0 if passed else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for root in ("public","cache","asset"):parser.add_argument("--"+root+"-root")
    parser.add_argument("--execute",action="store_true");parser.add_argument("--mode",choices=("rescore","replay"))
    parser.add_argument("--run-id");parser.add_argument("--plan-sha");parser.add_argument("--deadline",type=float);parser.add_argument("--output")
    args=parser.parse_args(argv)
    roots=(args.public_root,args.cache_root,args.asset_root)
    if args.execute or any(roots):
        if not all(roots):parser.error("All three explicit original roots required")
        public,cache,asset=(plain_path(value) for value in roots)
        if args.execute:
            if not all((args.mode,args.run_id,args.plan_sha,args.deadline,args.output)):parser.error("Explicit mode/one ID/sealed PLAN/deadline/fresh output required")
            return execute(public,cache,asset,args.plan_sha,args.mode,args.run_id,args.deadline,Path(args.output).absolute())
        runner=load_runner();metadata(runner,public,cache,asset,args.plan_sha)
    print(json.dumps(dict(status="METADATA_ONLY_VERIFIER_NOT_EXECUTED",registered_conditions=6,files_written=0,
        plan_sha256=PLAN_SHA,verifier_sha256=sha(__file__),numpy_imported="numpy" in sys.modules,torch_imported="torch" in sys.modules,
        psutil_imported="psutil" in sys.modules,actual_Task_array_model_or_child_operations=0,training_or_replay_or_T_executed=False)),flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
