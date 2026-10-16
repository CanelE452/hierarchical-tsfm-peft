"""One explicitly sealed fresh CPU condition; bare invocation reads no inputs."""
import argparse
import csv
import gzip
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import threading
import time

NAME = "postT_V_only_cpu_core_remaining_20261016_v1"
MODES = {"probe", "fit", "rescore", "replay"}


def proposer():
    path = Path(__file__).with_name("propose_core_remaining_plan.py")
    spec = importlib.util.spec_from_file_location("remaining29_proposer", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def dependency(public, relative, plan):
    p = proposer(); path = p.relative(public, relative)
    if p.sha(path) != plan["dependencies"][relative]:
        raise ValueError("Frozen helper dependency changed")
    spec = importlib.util.spec_from_file_location("remaining29_" + path.stem, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def choose(plan, run_id):
    candidates = [c for c in plan["conditions"] if isinstance(c, dict) and c.get("run_id") == run_id]
    if len(candidates) != 1 or candidates[0].get("preflight_status") != "SOURCE_METADATA_COMPATIBLE":
        raise ValueError("One exact supported sealed condition required")
    return candidates[0]


def f_counters(foundation):
    return {key: int(getattr(foundation, key)) for key in ("streams", "row_calls", "physical_streams", "physical_calls")}


def expected_calls(origins, microbatch, width):
    return sum(math.ceil(min(microbatch, origins-start)*width/128) for start in range(0, origins, microbatch))


def validate_probe(science, condition):
    micro, width = condition["spec"]["microbatch"], condition["effective_F_width"]
    calls = expected_calls(48, micro, width)
    expected = dict(streams=48*width, row_calls=calls, physical_streams=48*width, physical_calls=calls)
    segment = science["segment"]
    if (segment.get("status") != "SEGMENT_COMPLETE" or segment.get("updates") != 3 or segment.get("exposures") != 48
            or science.get("F_counter_delta") != expected or science.get("probe_state_discarded") is not True
            or science.get("saved_model_or_optimizer") is not False or science.get("base_hash_restored_after_close") is not True
            or science.get("no_foundation_parameter_gradients") is not True
            or science.get("actual_parameter_gradients_finite") is not True or science.get("actual_parameter_gradient_nonzero") is not True):
        raise ValueError("Three effective updates, actual gradients, F counters and discarded state required")
    if condition["family"] == "LORA":
        if science.get("gradient_scope") != "LORA_adapter_parameters_only" or science.get("adapter_hooks_removed_after_close") is not True:
            raise ValueError("LoRA adapter-only gradient and close proof required")
        return
    forwards, backwards = science.get("forward_events"), science.get("backward_events")
    count = 48//micro
    if not isinstance(forwards, list) or not isinstance(backwards, list) or len(forwards) != count or len(backwards) != count:
        raise ValueError("Every original microbatch must have an actual F-module backward event")
    for f, b in zip(forwards, backwards):
        value = b.get("absmax")
        if (f.get("shape") != [micro, 512, width] or f.get("requires_grad") is not True or f.get("grad_enabled") is not True
                or b.get("shape") != [micro, 512, width] or b.get("finite") is not True
                or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
            raise ValueError("Actual finite F-branch gradient shape differs")
    if not any(b["absmax"] > 0 for b in backwards):
        raise ValueError("All actual F-branch input gradients are zero")


def validate_fit(base, search, refit, condition, task_contract):
    n50 = task_contract["primary_split_ranges"]["A_fit"][1]-512-task_contract["H"]+1
    n60 = task_contract["primary_split_ranges"]["A_stop"][1]-512-task_contract["H"]+1
    target, updates = 0, 0
    for epoch in (.25, 1., 3.):
        end = math.ceil(epoch*n50); updates += math.ceil((end-target)/16); target = end
    selected = base.get("selected_epoch")
    if selected not in (0., .25, 1., 3.):
        raise ValueError("A_stop-selected refit epoch outside registered endpoints")
    exposures = math.ceil(selected*n60)
    expected_base = dict(family=condition["family"],seed=92601,normalization_fit_end=task_contract["primary_split_ranges"]["A_stop"][1],
        search_exposures=target,search_updates=updates,refit_exposures=exposures,refit_updates=math.ceil(exposures/16))
    if (base.get("status") != "COMPLETE_VALID" or base.get("run_id") != condition["run_id"]
            or any(base.get(key) != value for key,value in expected_base.items())
            or base.get("reused_complete_artifact") is not False or base.get("epochs") != [.25,1.,3.]
            or search.get("status") != "SEGMENT_COMPLETE" or search.get("completed_epochs") != [0.,.25,1.,3.]
            or search.get("exposures") != target or search.get("updates") != updates
            or search.get("train_origins") != n50 or refit.get("train_origins") != n60
            or refit.get("status") != "SEGMENT_COMPLETE" or refit.get("exposures") != exposures
            or refit.get("updates") != math.ceil(exposures/16)
            or refit.get("completed_epochs") != [0.]+([selected] if selected else [])
            or base.get("search_fit_end") != task_contract["primary_split_ranges"]["A_fit"][1]
            or base.get("refit_fit_end") != task_contract["primary_split_ranges"]["A_stop"][1]
            or base.get("refit_scaler_sha") != task_contract["refit_scaler_sha256"]):
        raise ValueError("Original full search3/permutation exposure or selected fresh A60 refit incomplete")


def require_fresh(public, cache, run_id, mode):
    if mode not in MODES:
        raise ValueError("No T or resumed mode exists")
    if mode in {"probe", "fit"}:
        for root in (public, cache, public/NAME, cache/NAME):
            if any((root/kind/run_id).exists() for kind in ("runs", "predictions")):
                raise FileExistsError("Existing fresh attempt/result preserved; no resume")
        if (public/NAME/"claims"/(mode+"_"+run_id+".json")).exists():
            raise FileExistsError("Existing attempt claim preserved")


def prior_closure(path, digest, p, process_api):
    if path is None or digest is None or p.sha(path) != digest:
        raise ValueError("Explicit ROOT prior-closure proof/SHA required")
    value = p.read(path)
    if (value.get("status") != "ALL_PRIOR_CPU_OWNERS_CLOSED" or value.get("remaining_owned_processes") != []
            or not isinstance(value.get("owned_bindings"), list) or not value["owned_bindings"]):
        raise ValueError("Prior fits/cost/audit closure has not been reconciled by ROOT")
    for row in value["owned_bindings"]:
        try:
            process = process_api.Process(row["pid"])
            if process.create_time() == row["create_time"]:
                raise ValueError("A prior exact CPU owner is still live")
        except process_api.NoSuchProcess:
            pass


def load_science(public, plan, name):
    module = __import__(name)
    p = proposer()
    if Path(module.__file__).resolve() != (public/(name+".py")).resolve() or p.sha(module.__file__) != plan["source39"][name+".py"]:
        raise ValueError("Original scientific module path/source differs")
    return module


def validate_saved(public, condition, task, p):
    folder = public/"predictions"/condition["run_id"]/"V_select"
    manifest = p.read(folder/"PREDICTION_MANIFEST.json")
    n, h, c = task["legal_origin_counts_primary"]["V_select"], task["H"], len(task["columns"])
    if (manifest.get("run_id") != condition["run_id"] or manifest.get("split") != "V_select"
            or manifest.get("data_sha") != task["data_sha256"] or manifest.get("scaler_sha") != task["refit_scaler_sha256"]
            or manifest["arrays"]["target_normalized"]["shape"] != [n,h,c]):
        raise ValueError("Saved V data/scaler/axes differ")
    return folder, manifest


def validate_source_contract(source, base, condition, plan, task):
    expected_files={name:plan["source39"][name] for name in proposer().SOURCE6}
    if (source.get("spec")!=condition["spec"] or source.get("files")!=expected_files
            or source.get("data_sha")!=task["data_sha256"] or source.get("model_sha")!=plan["foundation_tensor_sha256"]
            or source.get("source_sha")!=base.get("source_sha")):
        raise ValueError("Actual fresh SOURCE_CONTRACT differs from sealed science/data/model")


def execute(public, cache, asset, plan_sha, mode, run_id, deadline, output, probe_result=None, probe_sha=None, closure=None, closure_sha=None):
    p = proposer(); public, cache, asset, output = map(p.plain, (public, cache, asset, output))
    plan = p.metadata(public, cache, asset, plan_sha, run_id); condition = choose(plan, run_id)
    if mode not in MODES or deadline is None or not time.time() < deadline <= plan["deadline_unix"]:
        raise ValueError("One explicit live-deadline V-only mode required")
    post, private = public/NAME, cache/NAME
    if output.exists() or not output.is_relative_to(post/"diagnostics"):
        raise FileExistsError("New namespace diagnostic output required")
    require_fresh(public, cache, run_id, mode)
    env = os.environ
    if (env.get("CUDA_VISIBLE_DEVICES") != "-1" or any(k.startswith("INFORMS_GPU_") for k in env)
            or any(env.get(k) != "2" for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"))
            or any(env.get(k) != "1" for k in ("HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE"))
            or Path(env.get("INFORMS_CACHE_ROOT", "")).resolve() != private.resolve()
            or Path(env.get("INFORMS_ASSET_ROOT", "")).resolve() != asset.resolve()):
        raise ValueError("Explicit CPU2/offline/new-cache environment without GPU authorization required")
    used = sum(f.stat().st_size for root in (public,cache) for f in root.rglob("*") if f.is_file())
    if used+plan["admission_reserve_bytes"] > plan["storage_limit_bytes"] or any(shutil.disk_usage(root).free < plan["admission_reserve_bytes"] for root in (public,cache)):
        raise ValueError("Original storage cap/reserve unavailable")
    proof = None
    if mode == "fit":
        probe_result = p.plain(probe_result) if probe_result is not None else None
        if probe_result is not None and (not probe_result.is_relative_to(post/"diagnostics") or probe_result.name != "RESULT.json"):
            raise ValueError("Own namespace actual probe RESULT path required")
        if probe_result is None or probe_sha is None or p.sha(probe_result) != probe_sha:
            raise ValueError("Own actual discarded probe proof/SHA required")
        proof = p.read(probe_result)
        if (proof.get("status") != "PROBE_EFFECTIVE_UPDATES_COMPLETE_DISCARDED" or proof.get("run_id") != run_id
                or proof.get("plan_sha256") != plan_sha or proof.get("runner_sha256") != p.sha(__file__)
                or proof.get("original_source_and_inputs_preserved") is not True):
            raise ValueError("Probe identity or provenance differs")
        validate_probe(proof["scientific_result"], condition)
    sys.path.insert(0,str(public))
    import psutil
    prior_closure(closure, closure_sha, p, psutil)
    utility = dependency(public, "postT_V_only_cpu_lag_20261016_v1/cpu_lag_runner.py", plan)
    output.mkdir(parents=True,exist_ok=False)
    if mode in {"probe","fit"}:
        (post/"claims").mkdir(exist_ok=True)
        p.new(post/"claims"/(mode+"_"+run_id+".json"),dict(run_id=run_id,mode=mode,plan_sha256=plan_sha))
    p.new(output/"STARTED.json",dict(run_id=run_id,mode=mode,plan_sha256=plan_sha,runner_sha256=p.sha(__file__),
        actual_pid=os.getpid(),actual_create_time=psutil.Process().create_time(),started_unix=time.time(),new_T_scoring=False,
        external_two_slot_and_quiet_supervisor_required=True))
    error_class=post_error=science=None; failures=[]; process=psutil.Process(); stop=threading.Event(); model=foundation=None
    before_cpu=sum(process.cpu_times()[:2]); peak_rss=process.memory_info().rss; frozen=False
    phase="scientific_import"; replay_attempted=False; forward=None; status="FAILED_PRESERVED"; monitor=None; started=time.perf_counter()
    input_pins={};torch=None
    try:
        import torch
        torch.set_num_threads(2)
        def watch():
            nonlocal peak_rss
            while not stop.wait(.25):
                rss=process.memory_info().rss
                peak_rss=max(peak_rss,rss)
                if rss>plan["max_RSS_bytes"] or time.time()>=deadline:
                    p.new(output/"RESOURCE_STOP.json",dict(status="RESOURCE_STOP",rss_bytes=rss));os._exit(98)
        monitor=threading.Thread(target=watch,daemon=True);monitor.start()
        if torch.cuda.is_initialized() or process.memory_info().rss>plan["max_RSS_bytes"]:
            raise RuntimeError("CPU/RSS preflight violated")
        fit=load_science(public,plan,"fit_base");data=load_science(public,plan,"data_contract")
        contract=plan["task_contracts"][condition["dataset"]]
        if mode in {"probe","fit","replay"}:
            task=data.load_task(condition["dataset"],cache_dir=cache)
            validator=dependency(public,"postT_V_only_cpu_fixed_pca_20261016_v1/fixed_pca_runner.py",plan)
            p.new(output/"TASK_BINDING.json",validator.validate_task(task,plan,condition["dataset"],data,fit))
        if mode in {"probe","fit"}:
            runtime=load_science(public,plan,"gpu_runtime");training=load_science(public,plan,"training")
            foundation=runtime.FrozenBolt(asset/plan["model_asset_relative_path"],contract["H"],128,False,device="cpu")
            if foundation.initial_hash!=plan["foundation_tensor_sha256"] or foundation.instance_norm_source_sha256!=plan["instance_norm_source_sha256"]:
                raise ValueError("Actual frozen foundation or norm differs")
            if mode=="probe":
                phase="discarded_fresh_A50_probe";end=contract["primary_split_ranges"]["A_fit"][1]
                model,norm=fit.create_model(condition["spec"],task,end,foundation)
                if fit.scaler_fingerprint(norm["mean"],norm["std"])!=contract["search_scaler_sha256"]:
                    raise ValueError("Probe A50 normalization differs")
                windows=training.FastWindows(dict(task,physical=task["physical"][:end],observed=task["observed"][:end]),norm["mean"],norm["std"],"cpu")
                legal=data.origins((0,end),512,contract["H"],1);sampler=training.PermutationStream(legal,92601)
                optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=condition["spec"]["lr"],weight_decay=0)
                helper=dependency(public,"postT_V_only_cpu_required_minima_20261016_v1/required_minima_runner.py",plan)
                forwards,backwards=[],[];handles=();before=f_counters(foundation)
                if condition["family"]!="LORA":handles=helper.attach_probe_hooks(foundation,torch,forwards,backwards)
                try:
                    segment=training.train_segment(model,windows,legal,sampler,optimizer,48,
                        lambda x,k:fit.predict_model(model,x,foundation,condition["family"],k),effective_batch=16,
                        microbatch=condition["spec"]["microbatch"],starting_updates=0,budget_deadline=deadline)
                    grads=[v.grad for v in model.parameters() if v.requires_grad and v.grad is not None]
                    if not grads or any(not torch.isfinite(v).all().item() for v in grads):raise FloatingPointError("Probe parameter gradients absent/nonfinite")
                    maxima=[float(v.detach().abs().max().item()) for v in grads]
                    science=dict(segment=segment,F_counter_delta={k:v-before[k] for k,v in f_counters(foundation).items()},
                        probe_state_discarded=True,saved_model_or_optimizer=False,actual_parameter_gradients_finite=True,
                        no_foundation_parameter_gradients=all(v.grad is None for v in foundation.model.parameters()),
                        actual_parameter_gradient_nonzero=any(v>0 for v in maxima),full_A50_legal_origins=len(legal),
                        forward_events=forwards,backward_events=backwards,gradient_scope="F_module_grad_input_only")
                    if condition["family"]=="LORA":
                        model.close();science.update(gradient_scope="LORA_adapter_parameters_only",adapter_hooks_removed_after_close=bool(model._closed and not model._handles))
                    foundation.assert_frozen_unchanged();science["base_hash_restored_after_close"]=True;validate_probe(science,condition)
                finally:
                    for handle in handles:handle.remove()
                if callable(getattr(model,"close",None)):model.close()
                del model,optimizer,windows,sampler;model=None;status="PROBE_EFFECTIVE_UPDATES_COMPLETE_DISCARDED"
            else:
                phase="fresh_search_and_selected_refit"
                receipt=fit.fit_base(condition["spec"],task,private,post,foundation,deadline)
                science=dict(fit_receipt=receipt,full_V_completed=False)
                status=receipt.get("status","FAILED_PRESERVED")
                if status=="COMPLETE_VALID":
                    folder=post/"runs"/run_id;validate_fit(receipt,p.read(folder/"search/RECEIPT.json"),p.read(folder/"refit/RECEIPT.json"),condition,contract)
                    validate_source_contract(p.read(folder/"SOURCE_CONTRACT.json"),receipt,condition,plan,contract)
                    phase="fresh_full_V";before=f_counters(foundation)
                    predict=load_science(public,plan,"prediction_cache")
                    prediction=predict.predict_parent(run_id,task,"V_select",private,post,foundation,receipt=receipt,fold="primary",deadline=deadline)
                    n=contract["legal_origin_counts_primary"]["V_select"];width=condition["effective_F_width"]
                    delta={k:v-before[k] for k,v in f_counters(foundation).items()};calls=expected_calls(n,condition["spec"]["microbatch"],width)
                    if (prediction["score"]["n_origins"]!=n or prediction["score"]["horizon"]!=contract["H"] or prediction["score"]["num_channels"]!=len(contract["columns"])
                            or prediction["metadata"].get("reused_prediction_cache") or delta!=dict(streams=n*width,row_calls=calls,physical_streams=n*width,physical_calls=calls)):
                        raise ValueError("Actual full-V dimensions/F streams/row calls differ")
                    science.update(full_V_completed=True,V_score=prediction["score"],F_counter_delta=delta,
                        V_manifest_sha256=p.sha(post/"predictions"/run_id/"V_select/PREDICTION_MANIFEST.json"),convergence_claimed=False,
                        independent_rescore_completed=False,checkpoint_subset_replay_completed=False,
                        saved_bundle_F_calls_is_bookkeeping_not_actual_row_calls=True)
                    status="COMPLETE_FRESH_CPU_FIT_AND_FULL_V_PENDING_INDEPENDENT_VALIDATION"
        else:
            base=p.read(post/"runs"/run_id/"BASE_FIT_RECEIPT.json")
            validate_fit(base,p.read(post/"runs"/run_id/"search/RECEIPT.json"),p.read(post/"runs"/run_id/"refit/RECEIPT.json"),condition,contract)
            validate_source_contract(p.read(post/"runs"/run_id/"SOURCE_CONTRACT.json"),base,condition,plan,contract)
            folder,manifest=validate_saved(post,condition,contract,p)
            if manifest["metadata"]["artifact_sha256"]!=base["artifact_sha256"]:raise ValueError("V selected artifact differs")
            names=[post/"runs"/run_id/name for name in ("BASE_FIT_RECEIPT.json","SOURCE_CONTRACT.json","search/RECEIPT.json","refit/RECEIPT.json")]
            names += [folder/name for name in ("PREDICTION_MANIFEST.json","REFERENCE_CONTRACT.json","SCORES.csv","PUBLIC_ERROR_STATS.csv.gz")]
            input_pins={name.relative_to(post).as_posix():p.sha(name) for name in names}
            p.new(output/"INPUT_BINDINGS.json",dict(files=input_pins,selected_artifact_sha256=base["artifact_sha256"],
                actual_checkpoint_artifact_byte_validation_delegated_to_original_replay_API=mode=="replay",
                rescore_checks_saved_prediction_and_statistics_not_checkpoint_bytes=mode=="rescore"))
            if mode=="rescore":
                phase="independent_saved_V_FP64"
                audit=load_science(public,plan,"audit_predictions")
                with (folder/"SCORES.csv").open(encoding="utf-8-sig",newline="") as f:scores=list(csv.DictReader(f))
                if len(scores)!=1:raise ValueError("One saved score required")
                with gzip.open(folder/"PUBLIC_ERROR_STATS.csv.gz","rt",encoding="utf-8",newline="") as f:stats=list(csv.DictReader(f))
                projection=[dict(run_id=c["run_id"],execution_status=base["status"] if c["run_id"]==run_id else "NOT_ASSESSED",reason="Selected actual BASE status only") for c in plan["conditions"]]
                science=audit.audit_prediction_bundle(manifest,p.read(folder/"REFERENCE_CONTRACT.json"),scores[0],stats,projection,
                    [c["run_id"] for c in plan["conditions"]],[manifest["cost_record"]],private)
                if science.get("status")!="PASS" or science.get("n_origins")!=contract["legal_origin_counts_primary"]["V_select"]:raise ValueError("FP64 scope differs")
                status="PASS"
            else:
                phase="actual_CPU_checkpoint_V_subset_replay";registry=post/"REGISTRY.csv";alias=post/"EXPERIMENT_REGISTRY.csv"
                if alias.exists():
                    if alias.read_bytes()!=registry.read_bytes():raise ValueError("Existing registry alias differs")
                else:
                    with alias.open("xb") as stream:stream.write(registry.read_bytes())
                replay=load_science(public,plan,"route_predictions");replay_attempted=True
                science=replay.replay_route(run_id,task,private,post,asset/plan["model_asset_relative_path"],device="cpu",deadline=deadline,
                    lease=None,batch_size=condition["spec"]["microbatch"],atol=1e-5,rtol=1e-4)
                forward=science.get("executed_now")
                if (science.get("status")!="PASS" or len(science.get("origins",[]))!=6 or not isinstance(forward,bool)
                        or science.get("actual_F_streams")!=6*condition["effective_F_width"] or science.get("T_final_accessed") is not False
                        or science.get("F_physical_streams")!=6*condition["effective_F_width"]
                        or science.get("F_calls")!=expected_calls(6,condition["spec"]["microbatch"],condition["effective_F_width"])
                        or science.get("atol")!=1e-5 or science.get("rtol")!=1e-4
                        or science.get("standard_manifest_sha256")!=p.sha(folder/"PREDICTION_MANIFEST.json")):
                    raise ValueError("CPU6 replay identity/counters/scope differs")
                status="PASS_FRESH_CPU_REPLAY" if forward else "PASS_REUSED_REPLAY_NOT_NEW_CPU_EXECUTION"
    except Exception as error:
        error_class=type(error).__name__;failures.append(utility.save_private_failure(private,cache,mode,run_id,phase,error))
    finally:
        try:
            if callable(getattr(model,"close",None)):model.close()
            if foundation is not None:foundation.assert_frozen_unchanged();frozen=True
        except Exception as error:
            post_error=type(error).__name__;failures.append(utility.save_private_failure(private,cache,mode,run_id,"model_foundation_postflight",error))
        stop.set()
        if monitor is not None:monitor.join(timeout=1)
        try:
            p.metadata(public,cache,asset,plan_sha,run_id)
            if mode=="fit" and p.sha(probe_result)!=probe_sha:raise ValueError("Own actual probe proof changed during fit")
            if p.sha(closure)!=closure_sha:raise ValueError("Prior closure proof changed during execution")
            if mode in {"rescore","replay"}:p.check_pins(post,input_pins)
            if torch is not None and torch.cuda.is_initialized():raise RuntimeError("Unexpected CUDA initialization")
        except Exception as error:
            post_error=type(error).__name__;failures.append(utility.save_private_failure(private,cache,mode,run_id,"metadata_postflight",error))
    result=dict(status=status if error_class is None and post_error is None else "FAILED_PRESERVED",run_id=run_id,mode=mode,
        plan_sha256=plan_sha,runner_sha256=p.sha(__file__),scientific_result=science,error_class=error_class,postflight_error_class=post_error,
        private_failures=failures,original_source_and_inputs_preserved=post_error is None,actual_checkpoint_forward_in_this_command=forward,
        replay_call_attempted=replay_attempted,new_T_target_or_scoring=False,convergence_claimed=False,original_GPU_completion_claimed=False,
        elapsed_process_wall_seconds=time.perf_counter()-started,actual_CPU_process_seconds=sum(process.cpu_times()[:2])-before_cpu,
        rss_bytes=process.memory_info().rss,peak_observed_RSS_bytes=peak_rss,frozen_foundation_unchanged=frozen,
        archive_deserialization_scope="Original safeFIT loader may physically deserialize T bytes; requested computation is A50/A_stop/A60/V only",
        ROOT_must_preserve_actual_forced_exit_without_final_RESULT=True,finished_unix=time.time())
    p.new(output/"RESULT.json",result);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("public-root","cache-root","asset-root","output","probe-result","closure-proof"):
        parser.add_argument("--"+key,type=Path)
    for key in ("plan-sha","run-id","probe-sha","closure-sha"):parser.add_argument("--"+key)
    parser.add_argument("--mode",choices=sorted(MODES));parser.add_argument("--deadline",type=float);parser.add_argument("--execute",action="store_true")
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="DRAFT_TOOL_ONLY",experiment_inputs_read=0,writes=0,scientific_imports=False)));return 0
    if any(getattr(args,key) is None for key in ("public_root","cache_root","asset_root","output","plan_sha","run_id","mode","deadline","closure_proof","closure_sha")):
        parser.error("Execute requires sealed roots/PLAN, one ID/mode, new output, deadline and ROOT prior closure proof")
    result=execute(args.public_root,args.cache_root,args.asset_root,args.plan_sha,args.mode,args.run_id,args.deadline,args.output,
        args.probe_result,args.probe_sha,args.closure_proof,args.closure_sha)
    print(json.dumps(dict(status=result["status"],run_id=args.run_id,error_class=result["error_class"])))
    complete={"PROBE_EFFECTIVE_UPDATES_COMPLETE_DISCARDED","COMPLETE_FRESH_CPU_FIT_AND_FULL_V_PENDING_INDEPENDENT_VALIDATION",
              "PASS","PASS_FRESH_CPU_REPLAY","PASS_REUSED_REPLAY_NOT_NEW_CPU_EXECUTION"}
    return 0 if result["error_class"] is None and result["postflight_error_class"] is None and result["status"] in complete else 1


if __name__=="__main__":raise SystemExit(main())
