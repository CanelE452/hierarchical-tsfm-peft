"""Four original family minima, prepared for fresh post-T CPU development.

Default is stdlib metadata only. Original fit_base, training, prediction and
frozen foundation APIs execute only after ROOT seals a new PLAN and triggers.
"""
import argparse
import copy
import csv
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

NAME = "postT_V_only_cpu_family_minima_20261016_v1"
REFERENCE = "postT_V_only_cpu_fixed_pca_20261016_v1/PLAN.json"
REFERENCE_SHA = "b4207edec63331cf95b24909c24c692091251dd8e29316f8867b9f66c65a0930"
REQUIRED = "postT_V_only_cpu_required_minima_20261016_v1/required_minima_runner.py"
REQUIRED_SHA = "e11c34918862a698e962cf2a3c151ea6f7484032985f52d25cb7220da6b5d37b"
UTILITY = "postT_V_only_cpu_lag_20261016_v1/cpu_lag_runner.py"
UTILITY_SHA = "36c75b38cd2c072bac4466129249cc5213c2df679ad80232d4ac511ab6d7e7f1"
VALIDATOR = "postT_V_only_cpu_fixed_pca_20261016_v1/fixed_pca_runner.py"
VALIDATOR_SHA = "d73d866a6d03061d00fb7ca015a8f3b41e273b73011b22f0c1cd0fbbfb073dd9"
SOURCES = ("fit_base.py","training.py","networks.py","data_contract.py","gpu_runtime.py","evaluation.py")
IMMUTABLE = {"EXPERIMENT_REGISTRY.csv","SELECTION_LOCK.json","DATA_CONTRACT.json","SCORES.csv","COST.csv","COSTS.csv","FIT_RECEIPTS.csv","E0_LEDGER.csv","T_FIXED_ROLE_SCORES.csv"}
DEADLINE = 1792222550
TARGETS = (("jena","B4",6,"B4",21),("jena","ADAPTS_BOLT_PORT",6,"ADAPTS",21),
           ("jena","LORA_R8",0,"LORA",21),("beijing","B4",11,"B4",11))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""): h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def read(path):
    if Path(path).stat().st_size > 8*1024**2: raise ValueError("Small JSON metadata required")
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def new(path,value):
    with Path(path).open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False); stream.write("\n")


def relative(root,name):
    root,name = Path(root).resolve(),Path(name)
    path = root/name
    if name.is_absolute() or ".." in name.parts or any(p.is_symlink() for p in [path]+list(path.parents)):
        raise ValueError("Safe nonsymlink root-relative path required")
    path = path.resolve()
    if path == root or not path.is_relative_to(root): raise ValueError("Path escapes declared root")
    return path


def check_pins(root,pins):
    if any(sha(relative(root,path)) != digest for path,digest in pins.items()):
        raise ValueError("Pinned source or original input changed")


def dependency(public,path,digest,name):
    source = relative(public,path)
    if sha(source) != digest: raise ValueError("Frozen public dependency changed")
    spec = importlib.util.spec_from_file_location(name,source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def original_spec(dataset,label,k,family,channels):
    return dict(run_id=f"primary_{dataset}_E1_{label}_k{k}_s92601",dataset=dataset,family=family,
        latent_k=channels if family=="LORA" else k,seed=92601,fold="primary",epochs=[.25],
        lr=.0001 if family=="LORA" else .001,device="cuda",microbatch=4,effective_batch=16,
        F_row_microbatch=128,checkpoint_inputs=False,lora_rank=8,lora_alpha=16,stage_label="registered_initial_breadth")


def conditions(upstream_map):
    result = []
    for dataset,label,k,family,channels in TARGETS:
        original = original_spec(dataset,label,k,family,channels)
        old = original["run_id"]
        if upstream_map[old]["spec"] != original: raise ValueError("Original quarter-screen recipe differs")
        run_id = f"postT_cpu_{dataset}_E1_{label}_k{k}_s92601_family_v1"
        spec = copy.deepcopy(upstream_map[old]["spec"])
        spec.update(run_id=run_id,device="cpu",epochs=[.25,1.,3.],stage_label="postT_cpu_fresh_V_only_family_minima",
            provenance=dict(upstream_registered_id=old,old_weights_optimizer_RNG_or_state_reused=False))
        result.append(dict(run_id=run_id,dataset=dataset,family=family,registered_family=label,registered_latent_k=k,
            latent_k=spec["latent_k"],effective_F_width=channels if family=="LORA" else k,
            upstream_registered_id=old,spec=spec))
    return result


def validate_original(value,condition,source39,task,foundation):
    entry = next(t for t in TARGETS if condition["upstream_registered_id"]==original_spec(*t)["run_id"])
    spec = original_spec(*entry)
    files = {name:source39[name] for name in SOURCES}
    axes = dict(L=512,H=48,C=len(task["columns"]),columns=task["columns"],fold="primary",source_array_fingerprints=task["array_sha256"])
    expected = dict(spec=spec,files=files,source_sha=fingerprint(files),config_sha=fingerprint(dict(spec=spec,axes=axes)),
        data_sha=task["data_sha256"],model_sha=foundation)
    if value != expected: raise ValueError("Original recipe/source/data/model/axes contract differs")


def proposal(public,cache,asset):
    if not cache.is_dir(): raise FileNotFoundError("Original cache absent")
    check_pins(public,{REFERENCE:REFERENCE_SHA,REQUIRED:REQUIRED_SHA,UTILITY:UTILITY_SHA,VALIDATOR:VALIDATOR_SHA})
    base = read(relative(public,REFERENCE))
    if len(base["source39"]) != 39 or set(base["immutable_public"]) != IMMUTABLE: raise ValueError("Original39/9 contract required")
    check_pins(public,base["source39"]); check_pins(public,base["immutable_public"])
    if read(public/"SELECTION_LOCK.json").get("locked") is not True: raise ValueError("Observed-T selection must remain locked")
    model = relative(asset,base["model_asset_relative_path"])
    if not model.is_dir() or model.name != base["model_revision"]: raise FileNotFoundError("Pinned local F revision absent")
    check_pins(asset,{base["instance_norm_source_relative_path"]:base["instance_norm_source_sha256"]})
    if sha(public.parent/"temporal_frontend_campaign_20261015_v1/runtime.py") != base["instance_norm_source_sha256"]:
        raise ValueError("Actual worktree normalization source differs")
    assets = read(public/"DATA_CONTRACT.json")["assets"]
    tasks = {ds:base["task_contracts"][ds] for ds in ("jena","beijing")}
    for ds,contract in tasks.items():
        if ({k:v for k,v in contract.items() if k not in {"search_scaler_sha256","refit_scaler_sha256"}} != assets[ds]
                or contract["L"] != 512 or contract["H"] != 48 or contract["legal_origin_counts_primary"]["V_select"] != 43
                or len(contract["columns"]) != (21 if ds=="jena" else 11)):
            raise ValueError("Original task, chronology or V43 axes differ")
    with (public/"EXPERIMENT_REGISTRY.csv").open(encoding="utf-8-sig",newline="") as stream: old_rows = list(csv.DictReader(stream))
    upstream = {row["experiment_id"]:row for row in old_rows}
    if len(upstream) != len(old_rows): raise ValueError("Duplicate original ID")
    originals,pins = {},{}
    for target in TARGETS:
        old = original_spec(*target)["run_id"]
        name = "runs/"+old+"/SOURCE_CONTRACT.json"
        originals[old] = read(relative(public,name)); pins[name] = sha(relative(public,name))
    rows = conditions(originals)
    for row in rows:
        old = upstream.get(row["upstream_registered_id"],{})
        expected = dict(dataset=row["dataset"],family=row["registered_family"],latent_k=str(row["registered_latent_k"]),seed="92601",
            fold="primary",num_channels=str(len(tasks[row["dataset"]]["columns"])),length="512",horizon="48",
            requirement="MANDATORY_MINIMUM",planned_epoch_endpoints="0.25|1|3")
        if any(old.get(k)!=v for k,v in expected.items()) or row["run_id"] in upstream: raise ValueError("Original mandatory identity differs")
        validate_original(originals[row["upstream_registered_id"]],row,base["source39"],tasks[row["dataset"]],base["foundation_tensor_sha256"])
    helper = dependency(public,REQUIRED,REQUIRED_SHA,"family_required_metadata")
    registry = helper.registry_bytes(rows,upstream)
    fields = ("model_asset_relative_path","model_revision","foundation_tensor_sha256","instance_norm_source_relative_path","instance_norm_source_sha256")
    plan = {key:base[key] for key in fields}
    plan.update(schema="POST_T_FAMILY_MINIMA_CPU_PLAN_v1",stage="POST_T_V_ONLY_DEVELOPMENT",runner_sha256=sha(__file__),
        output_public_relative_root=NAME,output_cache_relative_root=NAME,source39=base["source39"],immutable_public=base["immutable_public"],
        task_contracts=tasks,conditions=rows,upstream_recipe_metadata=pins,source_reference=dict(relative_path=REFERENCE,sha256=REFERENCE_SHA),
        dependencies={REQUIRED:REQUIRED_SHA,UTILITY:UTILITY_SHA,VALIDATOR:VALIDATOR_SHA},postT_registry_relative_path="REGISTRY.csv",
        postT_registry_sha256=hashlib.sha256(registry).hexdigest(),CPU_threads=2,max_RSS_bytes=3*1024**3,
        maximum_concurrent_workers=2,total_owned_RSS_limit_bytes=6*1024**3,storage_limit_bytes=30*1024**3,
        admission_reserve_bytes=256*1024**2,deadline_unix=DEADLINE,probe_updates=3,each_fit_requires_its_own_actual_probe=True,
        external_prior_queue_closure_cost_quiet_and_two_slots_required=True,independent_V_rescore_replay_and_cost_required=True)
    return plan,registry


def metadata(public,cache,asset,plan_sha):
    if not re.fullmatch("[0-9a-f]{64}",plan_sha or ""): raise ValueError("Exact sealed PLAN SHA required")
    expected,registry = proposal(public,cache,asset)
    path = relative(public,NAME+"/PLAN.json")
    if sha(path) != plan_sha or read(path) != expected: raise ValueError("Sealed PLAN differs from original science/guards")
    if relative(public,NAME+"/REGISTRY.csv").read_bytes() != registry: raise ValueError("Frozen four-row registry changed")
    return expected


def seal_plan(public,cache,asset,output):
    output = Path(output).absolute()
    if output.exists() or output==cache or not output.is_relative_to(cache): raise FileExistsError("New private candidate folder required")
    relative(cache,output.relative_to(cache))
    plan,registry = proposal(public,cache,asset)
    after,again = proposal(public,cache,asset)
    if plan != after or registry != again: raise ValueError("Metadata changed during seal")
    output.mkdir(parents=True,exist_ok=False)
    new(output/"PLAN.json",plan)
    with (output/"REGISTRY.csv").open("xb") as stream: stream.write(registry)
    result = dict(status="METADATA_ONLY_PRIVATE_PLAN_CANDIDATE",runner_sha256=sha(__file__),plan_sha256=sha(output/"PLAN.json"),
        registry_sha256=sha(output/"REGISTRY.csv"),files_written=3,public_promotion=False,science_or_child_operations=0)
    new(output/"SEAL_RECEIPT.json",result)
    return result


def choose(plan,run_id):
    found = [row for row in plan["conditions"] if row["run_id"]==run_id]
    if len(found) != 1: raise ValueError("One of four explicit original family targets required")
    return found[0]


def require_fresh(public,cache,mode,run_id):
    if mode not in {"probe","fit"}: raise ValueError("Only probe/fit supported")
    for root in (public,cache,public/NAME,cache/NAME):
        for kind in ("runs","predictions"):
            if (root/kind/run_id).exists(): raise FileExistsError("Existing output preserved; no checkpoint resume or implicit reuse")
    if (public/NAME/"claims"/(mode+"_"+run_id+".json")).exists(): raise FileExistsError("Existing claimed attempt preserved")


def validate_probe_science(science,condition,task):
    u,k = 3,condition["effective_F_width"]
    calls = 4*u*math.ceil(4*k/128)
    expected = dict(probe_updates=3,F_counter_delta=dict(streams=16*u*k,row_calls=calls,physical_streams=16*u*k,physical_calls=calls),
        full_A50_legal_origins=task["primary_split_ranges"]["A_fit"][1]-559,full_A50_loss_normalization_preserved=True,
        probe_state_discarded=True,probe_state_reused_for_fit=False,saved_model_or_optimizer=False,completed_fit=False,V_scoring=False,
        actual_parameter_gradients_finite=True,actual_parameter_gradient_nonzero=True,no_foundation_parameter_gradients=True,
        base_hash_restored_after_close=True)
    if any(science.get(key)!=value for key,value in expected.items()): raise ValueError("Actual family probe activity/gradient/discard evidence differs")
    segment = science.get("segment") or {}
    if any(segment.get(key)!=value for key,value in dict(status="SEGMENT_COMPLETE",updates=3,exposures=48,optimizer_updates_this_segment=3,k_exposures={}).items()):
        raise ValueError("Three original effective probe updates required")
    if condition["family"]=="LORA":
        if (science.get("gradient_scope")!="LORA_adapter_parameters_only" or science.get("adapter_hooks_removed_after_close") is not True
                or "actual_backward_input_gradient_events" in science or "foundation_input_requires_grad_events" in science):
            raise ValueError("LoRA needs adapter gradients, not an F-input gradient gate")
        values = science.get("adapter_parameter_gradient_absmax",[])
        if not values or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in values) or not any(v>0 for v in values):
            raise ValueError("Actual finite nonzero LoRA adapter gradients required")
    else:
        events = science.get("actual_backward_input_gradient_events",[])
        forwards = science.get("foundation_input_requires_grad_events",[])
        if science.get("gradient_scope")!="F_module_grad_input_only" or len(events)!=12 or len(forwards)!=12:
            raise ValueError("Every original microbatch needs actual F-module backward evidence")
        if any(e!=dict(shape=[4,512,k],requires_grad=True,grad_enabled=True) for e in forwards): raise ValueError("F-input forward evidence differs")
        if any(set(e)!={"shape","finite","absmax"} or e["shape"]!=[4,512,k] or e["finite"] is not True
                or isinstance(e["absmax"],bool) or not isinstance(e["absmax"],(int,float)) or not math.isfinite(e["absmax"]) or e["absmax"]<0 for e in events):
            raise ValueError("F module actual input gradients must be finite")
        if not any(e["absmax"]>0 for e in events): raise ValueError("All F-module input gradients are zero")


def validate_probe_proof(value,condition,task,plan_sha,runner_sha):
    expected = dict(status="PROBE_EFFECTIVE_UPDATES_COMPLETE_DISCARDED",mode="probe",run_id=condition["run_id"],
        plan_sha256=plan_sha,runner_sha256=runner_sha,frozen_foundation_unchanged=True,
        original_source_registry_selection_T_pins_preserved=True,cuda_initialized=False,new_T_target_or_scoring=False)
    if any(value.get(k)!=v for k,v in expected.items()): raise ValueError("Own actual successful family probe RESULT required")
    validate_probe_science(value.get("scientific_result") or {},condition,task)


def validate_complete_fit(base,search,refit,condition,task):
    helper=dependency(Path(__file__).resolve().parent.parent,REQUIRED,REQUIRED_SHA,"family_complete_fit_validation")
    return helper.validate_complete_fit(base,search,refit,condition,dict(task_contract=task))


def predict_call(api,condition,task,cache,public,foundation,receipt,deadline,*,split="V_select"):
    helper=dependency(Path(__file__).resolve().parent.parent,REQUIRED,REQUIRED_SHA,"family_prediction_validation")
    return helper.predict_call(api,condition,task,cache,public,foundation,receipt,deadline,split=split)


def environment(cache,asset):
    if (os.environ.get("CUDA_VISIBLE_DEVICES")!="-1" or any(k.startswith("INFORMS_GPU_") for k in os.environ)
            or any(os.environ.get(k)!="2" for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"))
            or any(os.environ.get(k)!="1" for k in ("HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE"))
            or Path(os.environ.get("INFORMS_ASSET_ROOT","")).resolve()!=asset
            or Path(os.environ.get("INFORMS_CACHE_ROOT","")).resolve()!=cache/NAME):
        raise ValueError("Explicit offline CPU2 new-cache environment without GPU authorization required")


def execute(public,cache,asset,plan_sha,mode,run_id,deadline,output,probe_result=None,probe_result_sha=None):
    plan = metadata(public,cache,asset,plan_sha); condition = choose(plan,run_id)
    environment(cache,asset)
    if mode not in {"probe","fit"} or deadline is None or not time.time()<deadline<=DEADLINE: raise ValueError("Explicit live probe/fit deadline required")
    output = Path(output).absolute()
    if output.exists() or output==public/NAME or not output.is_relative_to(public/NAME): raise FileExistsError("Fresh owned public diagnostic output required")
    relative(public,output.relative_to(public)); require_fresh(public,cache,mode,run_id)
    proof_path = None; task_contract = plan["task_contracts"][condition["dataset"]]
    if mode=="fit":
        if not re.fullmatch("[0-9a-f]{64}",probe_result_sha or "") or probe_result is None: raise ValueError("Own actual probe RESULT/SHA required")
        proof_path = Path(probe_result).absolute()
        if not proof_path.is_relative_to(public/NAME): raise ValueError("Probe evidence must belong to this namespace")
        proof_path = relative(public,proof_path.relative_to(public))
        if proof_path.name!="RESULT.json" or sha(proof_path)!=probe_result_sha: raise ValueError("Probe bytes changed")
        validate_probe_proof(read(proof_path),condition,task_contract,plan_sha,sha(__file__))
    used = sum(p.stat().st_size for root in (public,cache) for p in root.rglob("*") if p.is_file())
    if used+plan["admission_reserve_bytes"]>plan["storage_limit_bytes"]: raise ValueError("Original combined30GiB storage cap reached")
    if any(shutil.disk_usage(root).free<plan["admission_reserve_bytes"] for root in (public,cache)):
        raise ValueError("Physical free-space reserve unavailable")
    helper = dependency(public,REQUIRED,REQUIRED_SHA,"family_required_runtime")
    utility = dependency(public,UTILITY,UTILITY_SHA,"family_private_failure_utility")
    validator = dependency(public,VALIDATOR,VALIDATOR_SHA,"family_task_validator")
    output.mkdir(parents=True,exist_ok=False)
    claims = public/NAME/"claims"; claims.mkdir(exist_ok=True)
    new(claims/(mode+"_"+run_id+".json"),dict(mode=mode,run_id=run_id,runner_sha256=sha(__file__),plan_sha256=plan_sha))
    sys.path.insert(0,str(public))
    import psutil
    import torch
    import data_contract as data
    import fit_base as fit
    import training
    import prediction_cache
    import gpu_runtime
    torch.set_num_threads(2)
    for module in (data,fit,training,prediction_cache,gpu_runtime):
        if Path(module.__file__).resolve()!=public/(module.__name__+".py"): raise ValueError("Original scientific module root required")
    process,stop = psutil.Process(),threading.Event()
    def guard():
        while not stop.wait(.25):
            rss = process.memory_info().rss
            if rss>plan["max_RSS_bytes"] or time.time()>=deadline:
                new(output/"RESOURCE_STOP.json",dict(status="RESOURCE_STOP",rss_bytes=rss,deadline_unix=deadline)); os._exit(98)
    watcher = threading.Thread(target=guard,daemon=True); watcher.start()
    before,started = utility.rss_record(process),time.perf_counter()
    new(output/"STARTED.json",dict(actual_pid=process.pid,actual_create_time=process.create_time(),mode=mode,run_id=run_id,
        plan_sha256=plan_sha,runner_sha256=sha(__file__),deadline_unix=deadline,resource_before=before))
    foundation=probe_model=science=None
    status,error_class,post_error,phase = "FAILED_PRESERVED",None,None,"Task_binding"
    failures=[]; frozen=preserved=False
    try:
        task = data.load_task(condition["dataset"],cache_dir=cache)
        new(output/"TASK_BINDING.json",validator.validate_task(task,plan,condition["dataset"],data,fit))
        if torch.cuda.is_initialized() or process.memory_info().rss>plan["max_RSS_bytes"]: raise RuntimeError("CPU/RSS contract violated")
        phase="frozen_F_loading"
        foundation = gpu_runtime.FrozenBolt(asset/plan["model_asset_relative_path"],48,128,False,device="cpu")
        if foundation.initial_hash!=plan["foundation_tensor_sha256"] or foundation.instance_norm_source_sha256!=plan["instance_norm_source_sha256"]:
            raise ValueError("Actual F/norm identity differs")
        foundation.assert_frozen_unchanged(); initial=helper.counters(foundation)
        if mode=="probe":
            phase="fresh_full_A50_three_update_probe"
            end=task_contract["primary_split_ranges"]["A_fit"][1]
            probe_model,norm=fit.create_model(condition["spec"],task,end,foundation)
            if fit.scaler_fingerprint(norm["mean"],norm["std"])!=task_contract["search_scaler_sha256"]: raise ValueError("A50 scale differs")
            windows=training.FastWindows(dict(task,physical=task["physical"][:end],observed=task["observed"][:end]),norm["mean"],norm["std"],"cpu")
            legal=data.origins((0,end),512,48,1); sampler=training.PermutationStream(legal,92601)
            optimizer=torch.optim.AdamW([p for p in probe_model.parameters() if p.requires_grad],lr=condition["spec"]["lr"],weight_decay=0)
            forwards,backwards=[],[]; handles=()
            if condition["family"]!="LORA": handles=helper.attach_probe_hooks(foundation,torch,forwards,backwards)
            try:
                segment=training.train_segment(probe_model,windows,legal,sampler,optimizer,48,
                    lambda x,k:fit.predict_model(probe_model,x,foundation,condition["family"],k),
                    effective_batch=16,microbatch=4,starting_updates=0,budget_deadline=deadline)
                grads=[p.grad for p in probe_model.parameters() if p.requires_grad and p.grad is not None]
                if not grads or any(not torch.isfinite(g).all().item() for g in grads): raise FloatingPointError("Actual parameter gradients absent or nonfinite")
                maxima=[float(g.detach().abs().max().item()) for g in grads]
                if not any(v>0 for v in maxima): raise FloatingPointError("All actual trainable gradients zero")
                delta={k:v-initial[k] for k,v in helper.counters(foundation).items()}
                foundation.assert_frozen_unchanged()
                science=dict(segment=segment,probe_updates=3,F_counter_delta=delta,full_A50_legal_origins=len(legal),
                    full_A50_loss_normalization_preserved=True,probe_state_discarded=True,probe_state_reused_for_fit=False,
                    saved_model_or_optimizer=False,completed_fit=False,V_scoring=False,actual_parameter_gradients_finite=True,
                    actual_parameter_gradient_nonzero=True,no_foundation_parameter_gradients=True)
                if condition["family"]=="LORA":
                    probe_model.close()
                    science.update(gradient_scope="LORA_adapter_parameters_only",adapter_parameter_gradient_absmax=maxima,
                        adapter_hooks_removed_after_close=bool(probe_model._closed and not probe_model._handles))
                else:
                    science.update(gradient_scope="F_module_grad_input_only",foundation_input_requires_grad_events=forwards,
                        actual_backward_input_gradient_events=backwards)
                foundation.assert_frozen_unchanged(); science["base_hash_restored_after_close"]=True
                validate_probe_science(science,condition,task_contract)
            finally:
                for handle in handles: handle.remove()
            if callable(getattr(probe_model,"close",None)): probe_model.close()
            del probe_model,optimizer,windows,sampler; probe_model=None
            status="PROBE_EFFECTIVE_UPDATES_COMPLETE_DISCARDED"
        else:
            phase="original_fit_base_fresh_A50_search_A60_refit"
            receipt=fit.fit_base(condition["spec"],task,cache/NAME,public/NAME,foundation,deadline)
            science=dict(fit_receipt=receipt,full_V_completed=False,independent_rescore_completed=False,checkpoint_subset_replay_completed=False)
            if receipt.get("status")=="COMPLETE_VALID":
                folder=public/NAME/"runs"/run_id
                validate_complete_fit(receipt,read(folder/"search/RECEIPT.json"),read(folder/"refit/RECEIPT.json"),condition,task_contract)
                phase="original_full_V"
                prediction=predict_call(prediction_cache.predict_parent,condition,task,cache/NAME,public/NAME,foundation,receipt,deadline)
                path=public/NAME/"predictions"/run_id/"V_select/PREDICTION_MANIFEST.json"
                science.update(full_V_completed=True,V_score=prediction["score"],V_cost_record=prediction["cost_record"],
                    V_manifest_relative_path=path.relative_to(public).as_posix(),V_manifest_sha256=sha(path),
                    convergence_established=False,selected_refit_epoch=receipt["selected_epoch"],
                    V_cost_F_calls_is_bundle_bookkeeping_not_actual_row_calls=True)
                status="COMPLETE_FRESH_CPU_FIT_AND_FULL_V_PENDING_INDEPENDENT_VALIDATION"
            else: status=receipt.get("status","FAILED_PRESERVED")
        science["session_F_counter_delta"]={k:v-initial[k] for k,v in helper.counters(foundation).items()}
        if torch.cuda.is_initialized(): raise RuntimeError("Unexpected CUDA initialization")
    except Exception as error:
        error_class=type(error).__name__; failures.append(utility.save_private_failure(cache/NAME,cache,mode,run_id,phase,error))
    finally:
        try:
            if probe_model is not None and callable(getattr(probe_model,"close",None)): probe_model.close()
            if foundation is not None: foundation.assert_frozen_unchanged(); frozen=True
            metadata(public,cache,asset,plan_sha)
            if proof_path is not None and sha(proof_path)!=probe_result_sha: raise ValueError("Actual probe evidence changed during fit")
            preserved=True
        except Exception as error:
            post_error=type(error).__name__; failures.append(utility.save_private_failure(cache/NAME,cache,mode,run_id,"postflight",error))
        after=utility.rss_record(process); stop.set(); watcher.join(timeout=1)
        success=error_class is None and post_error is None and preserved
        new(output/"RESULT.json",dict(status=status if success else "FAILED_PRESERVED",mode=mode,run_id=run_id,stage=plan["stage"],
            scientific_result=science,error_class=error_class,postflight_error_class=post_error,private_failures=failures,
            actual_pid=process.pid,actual_create_time=process.create_time(),resource_after=after,
            elapsed_process_wall_seconds=time.perf_counter()-started,actual_CPU_process_seconds=after["CPU_process_seconds"]-before["CPU_process_seconds"],
            frozen_foundation_unchanged=frozen,original_source_registry_selection_T_pins_preserved=preserved,cuda_initialized=torch.cuda.is_initialized(),
            new_T_target_or_scoring=False,post_T_development=True,original_GPU_coverage_completed=False,original_registry_statuses_updated=False,
            parent_must_enforce_prior_closure_quiet_cost_and_global_two_slots=True,parent_must_report_forced_exit_without_final_result=True,
            archive_deserialization_scope="Original safeFIT loader may physically deserialize T bytes; only A50/A_stop/A60/V computation requested",
            independent_validation_required=mode=="fit",plan_sha256=plan_sha,runner_sha256=sha(__file__),finished_unix=time.time()))
    return 0 if success and status in {"PROBE_EFFECTIVE_UPDATES_COMPLETE_DISCARDED","COMPLETE_FRESH_CPU_FIT_AND_FULL_V_PENDING_INDEPENDENT_VALIDATION"} else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for root in ("public","cache","asset"): parser.add_argument("--"+root+"-root")
    parser.add_argument("--plan-sha"); parser.add_argument("--seal-plan",action="store_true"); parser.add_argument("--execute",action="store_true")
    parser.add_argument("--mode",choices=("probe","fit")); parser.add_argument("--run-id"); parser.add_argument("--deadline",type=float)
    parser.add_argument("--probe-result"); parser.add_argument("--probe-result-sha"); parser.add_argument("--output")
    args=parser.parse_args(argv); roots=(args.public_root,args.cache_root,args.asset_root)
    if args.seal_plan and args.execute: parser.error("Separate metadata seal and scientific execution")
    if args.seal_plan or args.execute or any(roots):
        if not all(roots): parser.error("All three original roots required")
        public,cache,asset=(Path(v).resolve() for v in roots)
    if args.seal_plan:
        if not args.output: parser.error("New private seal candidate output required")
        print(json.dumps(seal_plan(public,cache,asset,Path(args.output).absolute()))); return 0
    if args.execute:
        if not all((args.plan_sha,args.mode,args.run_id,args.deadline,args.output)): parser.error("Exact PLAN/one ID/mode/deadline/newoutput required")
        return execute(public,cache,asset,args.plan_sha,args.mode,args.run_id,args.deadline,Path(args.output).absolute(),args.probe_result,args.probe_result_sha)
    if any(roots):
        if args.plan_sha: metadata(public,cache,asset,args.plan_sha)
        else: proposal(public,cache,asset)
    print(json.dumps(dict(status="METADATA_ONLY_PREPARATION_NOT_EXECUTED",registered_conditions=4,files_written=0,
        numpy_imported="numpy" in sys.modules,torch_imported="torch" in sys.modules,psutil_imported="psutil" in sys.modules,
        actual_Task_model_array_training_or_child_operations=0,runner_sha256=sha(__file__),new_T_scoring=False)))
    return 0


if __name__=="__main__": raise SystemExit(main())
