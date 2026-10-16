"""One completed family condition: saved V43 audit or actual CPU V6 replay.

Default is stdlib metadata only. ROOT must separately seal the family PLAN and
complete the selected fit; no other condition is required to have finished.
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
import shutil
import sys
import threading
import time

HERE=Path(__file__).resolve().parent
RUNNER_SHA="16e3a30d678b61ac49328ca1d3f31386bb03cbc4947f2c396b0fc1622258fbd9"
CORE_REL="postT_V_only_cpu_required_minima_20261016_v1/verify_required_minima_outputs.py"
CORE_SHA="c48cfc2bb738798b8e316fdc86de6d160f954ed634ed0831c5427afdcb3616a5"
PREDICTION_SOURCES=("prediction_cache.py","fit_base.py","data_contract.py","training.py","evaluation.py","networks.py","gpu_runtime.py")


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):h.update(block)
    return h.hexdigest()


def plain_path(path):
    path=Path(path).absolute()
    if any(p.is_symlink() for p in [path]+list(path.parents)):raise ValueError("Nonsymlink paths required before resolution")
    return path.resolve()


def read(path):
    path=plain_path(path)
    if path.stat().st_size>8*1024**2:raise ValueError("Small JSON metadata required")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def new(path,value):
    with Path(path).open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write("\n")


def module_at(path,digest,name):
    path=plain_path(path)
    if sha(path)!=digest:raise ValueError("Frozen verification dependency changed")
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def load_runner():
    return module_at(HERE/"family_minima_runner.py",RUNNER_SHA,"frozen_family_verification_runner")


def load_base():
    return module_at(HERE.parent/CORE_REL,CORE_SHA,"frozen_required_verification_core")


def metadata(runner,public,cache,asset,plan_sha):
    if not re.fullmatch("[0-9a-f]{64}",plan_sha or ""):raise ValueError("Explicit future ROOT-sealed family PLAN SHA required")
    plan=runner.metadata(public,cache,asset,plan_sha)
    if plan["runner_sha256"]!=RUNNER_SHA or len(plan["conditions"])!=4:raise ValueError("Frozen family runner/four-condition contract differs")
    load_base()
    return plan


def choose(runner,plan,run_id,mode):
    if mode not in {"rescore","replay"}:raise ValueError("Only saved V rescore or actual V subset replay supported")
    return runner.choose(plan,run_id)


def registry_projection(plan,public):
    return load_base().registry_projection(plan,public)


def ensure_registry_alias(runner,public,plan):
    return load_base().ensure_registry_alias(runner,public,plan)


def rescore_call(api,manifest,reference,score,statistics,registry,expected_ids,private):
    return load_base().rescore_call(api,manifest,reference,score,statistics,registry,expected_ids,private)


def replay_call(api,run_id,task,private,public,model_path,deadline):
    return load_base().replay_call(api,run_id,task,private,public,model_path,deadline)


def validate_replay(value,run_id,k):
    return load_base().validate_replay(value,run_id,k)


def forward_facts(attempted,value):
    return load_base().forward_facts(attempted,value)


def validate_replay_bindings(runner,value,values,paths,private,public):
    return load_base().validate_replay_bindings(runner,value,values,paths,private,public)


def capture_paths(runner,root,paths):
    return load_base().capture_paths(runner,root,paths)


def check_bindings(runner,public,cache,bindings):
    for label,files in bindings.items():
        root=public if label=="public" else cache
        if label not in {"public","cache"}:raise ValueError("Unknown captured input root")
        for name,digest in files.items():
            if sha(runner.relative(root,name))!=digest:raise ValueError("Captured selected input changed")


def validate_source(runner,source,base,search,refit,condition,plan):
    task=plan["task_contracts"][condition["dataset"]];spec=condition["spec"]
    files={key:plan["source39"][key] for key in runner.SOURCES}
    axes=dict(L=512,H=48,C=len(task["columns"]),columns=task["columns"],fold="primary",source_array_fingerprints=task["array_sha256"])
    expected=dict(spec=spec,files=files,source_sha=runner.fingerprint(files),config_sha=runner.fingerprint(dict(spec=spec,axes=axes)),
        data_sha=task["data_sha256"],model_sha=plan["foundation_tensor_sha256"])
    if source!=expected or base.get("source_sha")!=expected["source_sha"]:raise ValueError("Selected exact family CPU recipe/source/data/F axes differ")
    for stage,scaler in ((search,task["search_scaler_sha256"]),(refit,task["refit_scaler_sha256"])):
        required={key:expected[key] for key in ("source_sha","config_sha","data_sha","model_sha")};required["scaler_sha"]=scaler
        actual=stage.get("contract") or {}
        if (set(actual)!=set(required)|{"initialization_sha"} or any(actual.get(k)!=v for k,v in required.items())
                or not re.fullmatch("[0-9a-f]{64}",actual.get("initialization_sha",""))):
            raise ValueError("Selected search/refit scale, initialization or scientific contract differs")


def validate_saved_v(runner,manifest,reference,base,source,condition,plan):
    task=plan["task_contracts"][condition["dataset"]];c=len(task["columns"]);k=condition["effective_F_width"]
    prediction_files={name:plan["source39"][name] for name in PREDICTION_SOURCES}
    model=runner.fingerprint(dict(artifact_sha=base["artifact_sha256"],model_state_sha=base["model_state_sha"],normalization=task["refit_scaler_sha256"]))
    expected=dict(run_id=condition["run_id"],split="V_select",cohort=condition["dataset"]+":primary:V_select:stride48",
        source_sha=runner.fingerprint(dict(deployment_sources=prediction_files,training_source_sha=source["source_sha"])),
        config_sha=runner.fingerprint(dict(parent_spec=condition["spec"],dense=False,fold="primary",role="base_parent")),
        data_sha=task["data_sha256"],scaler_sha=task["refit_scaler_sha256"],model_sha=model)
    if any(manifest.get(key)!=v for key,v in expected.items()):raise ValueError("Full V family/model/scaler/cohort identity differs")
    if any(reference.get(key)!=v for key,v in expected.items()):raise ValueError("Pre-inference V reference identity differs")
    end=task["primary_split_ranges"]["A_stop"][1]
    meta=dict(artifact_sha256=base["artifact_sha256"],canonical_scaler_fit_end=end,parent_fit_end=end,parent_normalization_fit_end=end,
        family=condition["family"],fold="primary",dense=False,role="base_parent",training_source_sha=source["source_sha"],
        source_files=prediction_files,reused_prediction_cache=False)
    if any(manifest["metadata"].get(key)!=v for key,v in meta.items()):raise ValueError("Fresh selected A60 family V metadata differs")
    shapes=dict(pred_normalized=[43,48,c],target_normalized=[43,48,c],observed=[43,48,c],origins=[43],channel_ids=[c])
    if set(manifest["arrays"])!=set(shapes) or set(reference["arrays"])!=set(shapes)-{"pred_normalized"}:raise ValueError("Standard V array keys differ")
    if any(manifest["arrays"][key]["shape"]!=v for key,v in shapes.items()):raise ValueError("V43/H48 and task C dimensions differ")
    if any(reference["arrays"][key]!=manifest["arrays"][key] for key in reference["arrays"]):raise ValueError("V truth/mask/origin/channel metadata differs")
    cost=dict(run_id=condition["run_id"],batch=43,latent_k=k,actual_F_streams=43*k,expected_F_streams=43*k,F_calls=1)
    if manifest["cost_record"]!=cost or reference.get("expected_F_streams")!=43*k:
        raise ValueError("Original full-V effective F width ledger differs; F_calls1 is bundle bookkeeping")


def prepare_inputs(runner,plan,old_public,old_cache,run_id,condition):
    public,private=old_public/runner.NAME,old_cache/runner.NAME
    run,folder=public/"runs"/run_id,public/"predictions"/run_id/"V_select"
    paths=dict(fit_receipt=run/"BASE_FIT_RECEIPT.json",source_contract=run/"SOURCE_CONTRACT.json",search=run/"search/RECEIPT.json",
        refit=run/"refit/RECEIPT.json",manifest=folder/"PREDICTION_MANIFEST.json",reference=folder/"REFERENCE_CONTRACT.json",score=folder/"SCORES.csv")
    values={key:read(runner.relative(old_public,path.relative_to(old_public))) for key,path in paths.items() if key!="score"}
    if any(not isinstance(v,dict) for v in values.values()):raise ValueError("Selected fit/V metadata must be JSON objects")
    base,source,manifest=values["fit_receipt"],values["source_contract"],values["manifest"]
    runner.validate_complete_fit(base,values["search"],values["refit"],condition,plan["task_contracts"][condition["dataset"]])
    validate_source(runner,source,base,values["search"],values["refit"],condition,plan)
    validate_saved_v(runner,manifest,values["reference"],base,source,condition,plan)
    with paths["score"].open(encoding="utf-8-sig",newline="") as stream:scores=list(csv.DictReader(stream))
    if len(scores)!=1:raise ValueError("Exactly one per-run V score required")
    score=scores[0]
    for key,value in manifest["score"].items():
        if isinstance(value,(int,float)):
            if not math.isfinite(float(score[key])) or float(score[key])!=value:raise ValueError("Original V score CSV number differs")
        elif score[key]!=str(value):raise ValueError("Original V score CSV identity differs")
    paths["statistics"]=runner.relative(public,manifest["public_stats_relative_path"])
    paths["artifact"]=runner.relative(private,base["artifact_cache_relative_path"])
    paths["archive"]=runner.relative(private,manifest["cache_relative_path"])
    if paths["artifact"]!=private/"runs"/run_id/"selected_bundle.pt" or paths["archive"]!=private/"predictions"/run_id/"V_select.npz":
        raise ValueError("This selected fresh run artifact/archive path required")
    if sha(paths["artifact"])!=base["artifact_sha256"] or sha(paths["archive"])!=manifest["archive_sha256"]:raise ValueError("Selected artifact/archive bytes changed")
    bindings=dict(public=capture_paths(runner,old_public,[p for key,p in paths.items() if key not in {"artifact","archive"}]),
        cache=capture_paths(runner,old_cache,[paths["artifact"],paths["archive"]]))
    return values,score,paths,bindings


def execute(old_public,old_cache,asset,plan_sha,mode,run_id,deadline,output):
    runner=load_runner();core=load_base();plan=metadata(runner,old_public,old_cache,asset,plan_sha)
    condition=choose(runner,plan,run_id,mode);task_contract=plan["task_contracts"][condition["dataset"]]
    channels=len(task_contract["columns"]);width=condition["effective_F_width"]
    if deadline is None or not time.time()<deadline<=plan["deadline_unix"]:raise ValueError("Live deadline within original campaign required")
    runner.environment(old_cache,asset);public,private=old_public/runner.NAME,old_cache/runner.NAME
    core.require_new_output(runner,output,public)
    used=sum(path.stat().st_size for root in (old_public,old_cache) for path in root.rglob("*") if path.is_file())
    if (used+plan["admission_reserve_bytes"]>plan["storage_limit_bytes"]
            or any(shutil.disk_usage(root).free<plan["admission_reserve_bytes"] for root in (old_public,old_cache))):raise ValueError("Original storage cap/reserve unavailable")
    utility=runner.dependency(old_public,runner.UTILITY,runner.UTILITY_SHA,"family_verifier_private_failure_utility")
    output.mkdir(parents=True,exist_ok=False);source_sha=sha(__file__)
    new(output/"STARTED.json",dict(mode=mode,run_id=run_id,plan_sha256=plan_sha,verifier_sha256=source_sha,actual_pid=os.getpid(),
        started_unix=time.time(),new_training=False,new_T_scoring=False,requested_device="cpu",CPU_threads=2,replay_batch_size=6))
    error_class=post_error=value=alias=None;failures=[];bindings=dict(public={},cache={})
    replay_attempted=source_post=False;process=torch=watchdog=None;before=after=None;stop=threading.Event();started=time.perf_counter()
    try:
        sys.path.insert(0,str(old_public))
        import psutil
        import torch
        process=psutil.Process();torch.set_num_threads(2);before=utility.rss_record(process)
        def guard():
            while not stop.wait(.25):
                rss=process.memory_info().rss
                if rss>plan["max_RSS_bytes"] or time.time()>=deadline:
                    new(output/"RESOURCE_STOP.json",dict(status="RESOURCE_STOP",rss_bytes=rss,deadline_unix=deadline));os._exit(98)
        watchdog=threading.Thread(target=guard,daemon=True);watchdog.start()
        if torch.get_num_threads()!=2 or torch.cuda.is_initialized() or process.memory_info().rss>plan["max_RSS_bytes"]:raise RuntimeError("Actual CPU2/RSS contract violated")
        values,score,paths,bindings=prepare_inputs(runner,plan,old_public,old_cache,run_id,condition)
        projection=registry_projection(plan,public)
        new(output/"INPUT_BINDINGS.json",dict(files_sha256_by_original_root=bindings,registry_projection=projection,
            expected_run_ids=[row["run_id"] for row in plan["conditions"]],selected_artifact_sha256=values["fit_receipt"]["artifact_sha256"],
            saved_V_origins=43,horizon=48,num_channels=channels,registered_family=condition["registered_family"],actual_family=condition["family"],
            registered_latent_k=condition["registered_latent_k"],effective_F_width=width,full_three_epoch_search_verified=True,
            selected_fresh_A60_refit_epoch=values["fit_receipt"]["selected_epoch"],zero_selected_refit_is_allowed=True,
            original_training_status=values["fit_receipt"].get("training_status"),convergence_established=False))
        if mode=="rescore":
            import audit_predictions
            core.require_original_module(audit_predictions,old_public)
            with gzip.open(paths["statistics"],"rt",encoding="utf-8",newline="") as stream:statistics=list(csv.DictReader(stream))
            value=rescore_call(audit_predictions.audit_prediction_bundle,values["manifest"],values["reference"],score,statistics,projection,
                [row["run_id"] for row in plan["conditions"]],private)
            if (value.get("status")!="PASS" or value.get("n_origins")!=43 or value.get("horizon")!=48 or value.get("num_channels")!=channels
                    or value.get("checkpoint_replay") is not False or value.get("atol")!=1e-9 or value.get("rtol")!=1e-7):raise ValueError("Independent full V FP64 scope/tolerance differs")
        else:
            alias=ensure_registry_alias(runner,public,plan)
            bindings["public"][(public/alias["file"]).relative_to(old_public).as_posix()]=alias["sha256"]
            import data_contract as data
            import fit_base
            import route_predictions
            for module in (data,fit_base,route_predictions):core.require_original_module(module,old_public)
            task=data.load_task(condition["dataset"],cache_dir=old_cache)
            validator=runner.dependency(old_public,runner.VALIDATOR,runner.VALIDATOR_SHA,"family_verifier_task_validator")
            new(output/"TASK_BINDING.json",validator.validate_task(task,plan,condition["dataset"],data,fit_base))
            replay_attempted=True
            value=replay_call(route_predictions.replay_route,run_id,task,private,public,asset/plan["model_asset_relative_path"],deadline)
            validate_replay(value,run_id,width)
            extra=validate_replay_bindings(runner,value,values,paths,private,public)
            bindings["public"].update(capture_paths(runner,old_public,extra[:2]));bindings["cache"].update(capture_paths(runner,old_cache,extra[2:]))
            new(output/"REPLAY_BINDINGS.json",dict(files_sha256_by_original_root=bindings,executed_now=value["executed_now"],requested_device="cpu",
                CPU_threads=2,replay_batch_size=6,original_F_row_microbatch=128,cached_receipt_has_no_device_field=True,
                cache_hit_is_not_new_CPU_execution=True,foundation_checks="Original Route.close/deactivate (LoRA adapter close) then FoundationPool.close frozen hash/base-gradient checks; no added input-gradient gate"))
        if torch.cuda.is_initialized():raise RuntimeError("Unexpected CUDA initialization")
    except Exception as error:
        error_class=type(error).__name__;failures.append(utility.save_private_failure(private,old_cache,mode,run_id,"verification",error))
    finally:
        try:
            metadata(runner,old_public,old_cache,asset,plan_sha)
            if sha(__file__)!=source_sha:raise ValueError("Verifier changed during execution")
            check_bindings(runner,old_public,old_cache,bindings);source_post=True
        except Exception as error:
            post_error=type(error).__name__;failures.append(utility.save_private_failure(private,old_cache,mode,run_id,"postflight",error))
        stop.set()
        if watchdog is not None:watchdog.join(timeout=1)
        if process is not None:after=utility.rss_record(process)
        cuda=bool(torch.cuda.is_initialized()) if torch is not None else None;facts=forward_facts(replay_attempted,value)
        passed=error_class is None and post_error is None and source_post and value is not None and value.get("status")=="PASS" and cuda is False
        new(output/"RESULT.json",dict(status="PASS" if passed else "FAILED_OR_NUMERICAL_MISMATCH_PRESERVED",mode=mode,run_id=run_id,scientific_result=value,**facts,
            actual_fresh_CPU_replay_verified=passed and mode=="replay" and facts["actual_checkpoint_forward_in_this_command"] is True,
            saved_full_V_rescoring=mode=="rescore",subset_checkpoint_replay=mode=="replay",training_rerun=False,
            full_V_origin_count=43,replay_origin_count=6 if mode=="replay" else 0,dataset=condition["dataset"],num_channels=channels,horizon=48,
            registered_family=condition["registered_family"],actual_family=condition["family"],registered_latent_k=condition["registered_latent_k"],effective_F_width=width,
            requested_device="cpu",CPU_threads=2,actual_Torch_CPU_threads=torch.get_num_threads() if torch is not None else None,
            replay_batch_size=6 if mode=="replay" else None,F_row_microbatch=128,cached_replay_receipt_device_unknown=mode=="replay" and facts["actual_checkpoint_forward_in_this_command"] is False,
            original_saved_V_F_calls_is_bundle_bookkeeping=True,error_class=error_class,postflight_error_class=post_error,private_failures=failures,metadata_alias=alias,
            actual_pid=os.getpid(),actual_create_time=process.create_time() if process is not None else None,
            original_source_registry_selection_T_and_captured_inputs_preserved=source_post,cuda_initialized=cuda,new_T_target_or_scoring=False,
            post_T_development=True,independent_generalization_claimed=False,original_GPU_minimum_gap_completed=False,original_registry_statuses_updated=False,
            parent_must_enforce_prior_closure_quiet_and_owned_process_closure=True,parent_must_report_forced_exit_without_final_result=True,
            archive_deserialization_scope="Original safeFIT Task may physically deserialize T bytes; no new T target or scoring API requested",
            elapsed_process_wall_seconds=time.perf_counter()-started,resource_before=before,resource_after=after,
            plan_sha256=plan_sha,verifier_sha256=source_sha,required_verifier_core_sha256=CORE_SHA,finished_unix=time.time()))
    return 0 if passed else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for root in ("public","cache","asset"):parser.add_argument("--"+root+"-root")
    parser.add_argument("--execute",action="store_true");parser.add_argument("--mode",choices=("rescore","replay"))
    parser.add_argument("--run-id");parser.add_argument("--plan-sha");parser.add_argument("--deadline",type=float);parser.add_argument("--output")
    args=parser.parse_args(argv);roots=(args.public_root,args.cache_root,args.asset_root)
    if args.execute or any(roots):
        if not all(roots):parser.error("All three original roots required")
        public,cache,asset=(plain_path(value) for value in roots)
        if args.execute:
            if not all((args.mode,args.run_id,args.plan_sha,args.deadline,args.output)):parser.error("One mode/ID/exact PLAN/deadline/newoutput required")
            return execute(public,cache,asset,args.plan_sha,args.mode,args.run_id,args.deadline,Path(args.output).absolute())
        metadata(load_runner(),public,cache,asset,args.plan_sha)
    print(json.dumps(dict(status="METADATA_ONLY_VERIFIER_NOT_EXECUTED",registered_conditions=4,files_written=0,
        verifier_sha256=sha(__file__),runner_sha256=RUNNER_SHA,required_verifier_core_sha256=CORE_SHA,
        numpy_imported="numpy" in sys.modules,torch_imported="torch" in sys.modules,psutil_imported="psutil" in sys.modules,
        actual_Task_array_model_or_child_operations=0,training_or_replay_or_T_executed=False)))
    return 0


if __name__=="__main__":raise SystemExit(main())
