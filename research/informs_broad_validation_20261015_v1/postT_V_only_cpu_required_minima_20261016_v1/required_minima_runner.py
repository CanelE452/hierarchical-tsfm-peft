"""Six explicit Beijing CPU fresh fits; default is stdlib metadata only.

Original scientific modules are imported without edits. Probe state is discarded;
completed V generation still requires separate independent rescore and replay.
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time

NAME = "postT_V_only_cpu_required_minima_20261016_v1"
REFERENCE_PATH = "postT_V_only_cpu_fixed_pca_20261016_v1/PLAN.json"
REFERENCE_SHA = "b4207edec63331cf95b24909c24c692091251dd8e29316f8867b9f66c65a0930"
UTILITY_PATH = "postT_V_only_cpu_lag_20261016_v1/cpu_lag_runner.py"
UTILITY_SHA = "36c75b38cd2c072bac4466129249cc5213c2df679ad80232d4ac511ab6d7e7f1"
TASK_VALIDATOR_PATH = "postT_V_only_cpu_fixed_pca_20261016_v1/fixed_pca_runner.py"
TASK_VALIDATOR_SHA = "d73d866a6d03061d00fb7ca015a8f3b41e273b73011b22f0c1cd0fbbfb073dd9"
FIT_SOURCES = ("fit_base.py","training.py","networks.py","data_contract.py","gpu_runtime.py","evaluation.py")
IMMUTABLE = {"EXPERIMENT_REGISTRY.csv","SELECTION_LOCK.json","DATA_CONTRACT.json","SCORES.csv","COST.csv","COSTS.csv","FIT_RECEIPTS.csv","E0_LEDGER.csv","T_FIXED_ROLE_SCORES.csv"}
DEADLINE = 1792222550


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""): value.update(block)
    return value.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def read(path):
    path = Path(path)
    if path.stat().st_size > 8*1024**2: raise ValueError("Small JSON metadata limit exceeded")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def new(path,value):
    with Path(path).open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False); stream.write("\n")


def relative(root,name):
    root,name = Path(root).resolve(),Path(name)
    path = root/name
    if name.is_absolute() or ".." in name.parts or any(p.is_symlink() for p in [path]+list(path.parents)):
        raise ValueError("Safe nonsymlink root-relative path required")
    path = path.resolve()
    if path == root or not path.is_relative_to(root): raise ValueError("Path escapes its explicit root")
    return path


def check_pins(root,pins):
    if any(sha(relative(root,name)) != digest for name,digest in pins.items()):
        raise ValueError("Frozen original metadata or source bytes changed")


def load_dependency(public,path,digest,name):
    source = relative(public,path)
    if sha(source) != digest: raise ValueError("Frozen stdlib/task utility changed")
    spec = importlib.util.spec_from_file_location(name,source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def conditions(campaign_sha):
    result = []
    for k in (6,11):
        for family in ("B2","LAG_CAUSAL","LAG_BIDIR"):
            block = "E1" if family == "B2" else "E3"
            old = f"primary_beijing_{block}_{family}_k{k}_s92601"
            run_id = f"postT_cpu_beijing_{block}_{family}_k{k}_s92601_required_v1"
            spec = dict(run_id=run_id,dataset="beijing",family=family,latent_k=k,seed=92601,fold="primary",epochs=[.25,1.,3.],
                lr=.001,device="cpu",microbatch=4,effective_batch=16,F_row_microbatch=128,checkpoint_inputs=False,lora_rank=8,lora_alpha=16,
                stage_label="postT_cpu_fresh_V_only_required_minima",provenance=dict(upstream_registered_id=old))
            result.append(dict(run_id=run_id,family=family,latent_k=k,upstream_registered_id=old,spec=spec,
                recipe_reference=dict(relative_path="campaign.py",sha256=campaign_sha)))
    return result


def original_spec(condition):
    value = {key:item for key,item in condition["spec"].items() if key not in {"stage_label","provenance"}}
    return dict(value,run_id=condition["upstream_registered_id"],epochs=[.25],device="cuda",stage_label="registered_initial_breadth")


def validate_original_recipe(value,condition,source39,contract,foundation_sha):
    spec = original_spec(condition)
    files = {key:source39[key] for key in FIT_SOURCES}
    axes = dict(L=512,H=48,C=11,columns=contract["columns"],fold="primary",source_array_fingerprints=contract["array_sha256"])
    expected = dict(spec=spec,files=files,source_sha=fingerprint(files),config_sha=fingerprint(dict(spec=spec,axes=axes)),
        data_sha=contract["data_sha256"],model_sha=foundation_sha)
    if value != expected: raise ValueError("Actual original quarter-screen recipe/source/axes/model differs")


def registry_bytes(rows,upstream):
    records = []
    for condition in rows:
        row = dict(upstream[condition["upstream_registered_id"]])
        row.update(experiment_id=condition["run_id"],execution_status="PENDING_FRESH_CPU_EXECUTION",training_status="NOT_RUN",
            validity_status="PENDING_INDEPENDENT_VALIDATION",scientific_status="UNASSESSED",publication_status="PREPARATION_ONLY",
            reason="Fresh post-T CPU development of the original required minimum; no prior checkpoint reuse or T selection",
            resume_command="",device="cpu",stage="POST_T_V_ONLY_DEVELOPMENT",epochs="0.25|1|3",
            upstream_registered_id=condition["upstream_registered_id"],upstream_requirement=row["requirement"],
            requirement="POST_T_CPU_ORIGINAL_REQUIRED_MINIMUM")
        records.append(row)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer,fieldnames=list(records[0]),lineterminator="\n")
    writer.writeheader(); writer.writerows(records)
    return buffer.getvalue().encode("utf-8")


def proposal(public,cache,asset):
    if not cache.is_dir(): raise FileNotFoundError("Explicit original campaign cache directory absent")
    reference = relative(public,REFERENCE_PATH)
    if sha(reference) != REFERENCE_SHA: raise ValueError("Original sealed PCA task/guard reference changed")
    base = read(reference)
    if len(base["source39"]) != 39 or set(base["immutable_public"]) != IMMUTABLE:
        raise ValueError("Exact original source39 and nine public controls required")
    check_pins(public,base["source39"]); check_pins(public,base["immutable_public"])
    if read(public/"SELECTION_LOCK.json").get("locked") is not True: raise ValueError("Original T selection must stay locked")
    check_pins(public,{UTILITY_PATH:UTILITY_SHA,TASK_VALIDATOR_PATH:TASK_VALIDATOR_SHA})
    contract = base["task_contracts"]["beijing"]
    original = read(public/"DATA_CONTRACT.json")["assets"]["beijing"]
    if ({key:value for key,value in contract.items() if key not in {"search_scaler_sha256","refit_scaler_sha256"}} != original
            or contract["shape"] != [21038,11] or contract["L"] != 512 or contract["H"] != 48
            or contract["primary_split_ranges"]["A_stop"][1] != 12622 or contract["legal_origin_counts_primary"]["V_select"] != 43):
        raise ValueError("Original Beijing task/full axes/A60/V43 contract differs")
    model = relative(asset,base["model_asset_relative_path"])
    if not model.is_dir() or model.name != base["model_revision"]: raise FileNotFoundError("Pinned local Small snapshot/revision absent")
    check_pins(asset,{base["instance_norm_source_relative_path"]:base["instance_norm_source_sha256"]})
    if sha(public.parent/"temporal_frontend_campaign_20261015_v1/runtime.py") != base["instance_norm_source_sha256"]:
        raise ValueError("Actual worktree normalization source changed")
    with (public/"EXPERIMENT_REGISTRY.csv").open(encoding="utf-8-sig",newline="") as stream: original_rows = list(csv.DictReader(stream))
    upstream = {row["experiment_id"]:row for row in original_rows}
    if len(upstream) != len(original_rows): raise ValueError("Original registry contains duplicate identities")
    rows = conditions(base["source39"]["campaign.py"])
    recipe_pins = {}
    for condition in rows:
        old = upstream.get(condition["upstream_registered_id"],{})
        expected = dict(dataset="beijing",family=condition["family"],latent_k=str(condition["latent_k"]),seed="92601",fold="primary",
            num_channels="11",length="512",horizon="48",requirement="MANDATORY_MINIMUM",planned_epoch_endpoints="0.25|1|3")
        if any(old.get(key) != value for key,value in expected.items()) or condition["run_id"] in upstream:
            raise ValueError("Exact original six mandatory rows or fresh identities differ")
        name = "runs/"+condition["upstream_registered_id"]+"/SOURCE_CONTRACT.json"
        path = relative(public,name)
        validate_original_recipe(read(path),condition,base["source39"],contract,base["foundation_tensor_sha256"])
        recipe_pins[name] = sha(path)
    registry = registry_bytes(rows,upstream)
    fields = ("model_asset_relative_path","model_revision","foundation_tensor_sha256","instance_norm_source_relative_path","instance_norm_source_sha256")
    plan = {key:base[key] for key in fields}
    plan.update(schema="POST_T_REQUIRED_MINIMA_CPU_PLAN_v1",stage="POST_T_V_ONLY_DEVELOPMENT",runner_sha256=sha(__file__),
        output_public_relative_root=NAME,output_cache_relative_root=NAME,source39=base["source39"],immutable_public=base["immutable_public"],
        upstream_recipe_metadata=recipe_pins,task_contract=contract,conditions=rows,
        source_reference=dict(relative_path=REFERENCE_PATH,sha256=REFERENCE_SHA),
        shared_utility=dict(relative_path=UTILITY_PATH,sha256=UTILITY_SHA),task_validator=dict(relative_path=TASK_VALIDATOR_PATH,sha256=TASK_VALIDATOR_SHA),
        postT_registry_relative_path="REGISTRY.csv",postT_registry_sha256=hashlib.sha256(registry).hexdigest(),
        CPU_threads=2,max_RSS_bytes=3*1024**3,maximum_concurrent_workers=2,total_owned_RSS_limit_bytes=6*1024**3,
        storage_limit_bytes=30*1024**3,admission_reserve_bytes=256*1024**2,deadline_unix=DEADLINE,
        probe_updates_allowed=[1,2,3],each_fit_requires_its_own_actual_probe=True,never_concurrent_with_cost_measurements=True,
        independent_V_rescore_and_checkpoint_replay_required=True,B0_reuse_assumed_complete=False,original_registry_statuses_updated=False)
    return plan,registry


def metadata(public,cache,asset,plan_sha):
    if not re.fullmatch("[0-9a-f]{64}",plan_sha or ""): raise ValueError("Explicit ROOT-sealed PLAN SHA required")
    path = relative(public,NAME+"/PLAN.json")
    expected,registry = proposal(public,cache,asset)
    plan = read(path)
    if sha(path) != plan_sha or plan != expected: raise ValueError("Sealed six-condition PLAN differs from the exact source/recipe/guard proposal")
    if relative(public,NAME+"/REGISTRY.csv").read_bytes() != registry: raise ValueError("Separate sealed six-row registry bytes changed")
    return plan


def seal_plan(public,cache,asset,output):
    output = Path(output).absolute()
    if output == cache or not output.is_relative_to(cache) or output.exists(): raise FileExistsError("Fresh private plan-candidate output required")
    relative(cache,output.relative_to(cache))
    plan,registry = proposal(public,cache,asset)
    after,registry_after = proposal(public,cache,asset)
    if plan != after or registry != registry_after: raise ValueError("Source/metadata changed while preparing the metadata-only seal")
    output.mkdir(parents=True,exist_ok=False)
    new(output/"PLAN.json",plan)
    with (output/"REGISTRY.csv").open("xb") as stream: stream.write(registry)
    result = dict(status="METADATA_ONLY_PLAN_CANDIDATE_NOT_EXECUTED",plan_sha256=sha(output/"PLAN.json"),registry_sha256=sha(output/"REGISTRY.csv"),
        runner_sha256=sha(__file__),source_and_original_controls_preserved=True,actual_Task_model_array_child_or_training_operations=0,
        files_written=3,public_PLAN_or_registry_promoted=False,ROOT_review_and_explicit_public_seal_required=True)
    new(output/"SEAL_RECEIPT.json",result)
    return result


def choose(plan,run_id):
    matches = [row for row in plan["conditions"] if row["run_id"] == run_id]
    if len(matches) != 1: raise ValueError("One of the six explicit Beijing conditions is required")
    return matches[0]


def require_fresh(public,cache,mode,run_id):
    if mode not in {"probe","fit"}: raise ValueError("Only fresh probe or fit mode is supported")
    for root in (public,cache,public/NAME,cache/NAME):
        for kind in ("runs","predictions"):
            if (root/kind/run_id).exists(): raise FileExistsError("Existing run/prediction cannot be resumed or overwritten")
    if (public/NAME/"claims"/(mode+"_"+run_id+".json")).exists(): raise FileExistsError("Existing claimed attempt preserved")


def counters(foundation):
    return {key:int(getattr(foundation,key)) for key in ("streams","row_calls","physical_streams","physical_calls")}


def probe_call(api,model,windows,legal,sampler,optimizer,predict,foundation,condition,deadline,updates):
    if updates not in (1,2,3): raise ValueError("Only one through three effective probe updates allowed")
    before = counters(foundation)
    foundation.assert_frozen_unchanged()
    try:
        segment = api(model,windows,legal,sampler,optimizer,16*updates,predict,effective_batch=16,microbatch=4,
            starting_updates=0,budget_deadline=deadline)
        delta = {key:value-before[key] for key,value in counters(foundation).items()}
        expected = dict(streams=16*updates*condition["latent_k"],row_calls=4*updates,physical_streams=16*updates*condition["latent_k"],physical_calls=4*updates)
        if (segment.get("status") != "SEGMENT_COMPLETE" or segment.get("updates") != updates or segment.get("exposures") != 16*updates
                or segment.get("optimizer_updates_this_segment") != updates or segment.get("k_exposures") != {} or delta != expected
                or any(not isinstance(segment.get(key),(int,float)) or not math.isfinite(segment[key]) for key in ("gradient_norm_min","gradient_norm_max"))):
            raise ValueError("Original single-K effective updates, full-population loss or actual logical/physical F activity differs")
        return dict(segment=segment,probe_updates=updates,F_counter_delta=delta,full_A50_legal_origins=len(legal),
            full_A50_loss_normalization_preserved=True,probe_state_discarded=True,probe_state_reused_for_fit=False,
            saved_model_or_optimizer=False,completed_fit=False,V_scoring=False)
    finally: foundation.assert_frozen_unchanged()


def validate_input_gradient_events(events,k,updates):
    if len(events) != 4*updates or any(row != dict(shape=[4,512,k],requires_grad=True,grad_enabled=True) for row in events):
        raise ValueError("Actual frozen F must receive gradient-enabled K-dimensional inputs at every original probe microbatch")


def attach_probe_hooks(foundation,torch,forward_events,backward_events):
    def observe_forward(module,inputs):
        value = inputs[0]
        forward_events.append(dict(shape=list(value.shape),requires_grad=bool(value.requires_grad),grad_enabled=bool(torch.is_grad_enabled())))
        if not value.requires_grad: raise ValueError("Probe F input has no gradient path")
    def observe_backward(module,grad_input,grad_output):
        gradient = grad_input[0] if len(grad_input) == 1 else None
        if gradient is None:
            backward_events.append(dict(shape=[],finite=False,absmax=None))
        else:
            detached = gradient.detach()
            backward_events.append(dict(shape=list(detached.shape),finite=bool(torch.isfinite(detached).all().item()),
                absmax=float(detached.abs().max().item())))
        return None
    forward_handle = foundation.register_forward_pre_hook(observe_forward)
    try: backward_handle = foundation.register_full_backward_hook(observe_backward)
    except BaseException:
        forward_handle.remove()
        raise
    return forward_handle,backward_handle


def validate_backward_gradient_events(events,k,updates):
    if len(events) != 4*updates: raise ValueError("Every probe microbatch must have an actual F-input backward gradient event")
    for row in events:
        value = row.get("absmax")
        if (set(row) != {"shape","finite","absmax"} or row["shape"] != [4,512,k] or row["finite"] is not True
                or isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value < 0):
            raise ValueError("Actual backward input gradients must have the expected shape and finite absolute maximum")
    nonzero = sum(row["absmax"] > 0 for row in events)
    if not nonzero: raise ValueError("All observed F-input backward gradients are zero")
    return dict(actual_backward_input_gradients_observed=True,actual_backward_input_gradient_count=len(events),
        actual_backward_input_gradients_all_finite=True,actual_backward_input_gradient_nonzero_count=nonzero,
        actual_backward_input_gradient_absmax=max(row["absmax"] for row in events),
        actual_backward_input_gradient_scope="F_module_grad_input_only")


def validate_complete_fit(base,search,refit,condition,plan):
    bounds = plan["task_contract"]["primary_split_ranges"]
    n50,n60 = bounds["A_fit"][1]-559,bounds["A_stop"][1]-559
    targets = [0]+[math.ceil(epoch*n50) for epoch in (.25,1,3)]
    updates = sum(math.ceil((b-a)/16) for a,b in zip(targets,targets[1:]))
    epoch = base.get("selected_epoch")
    if epoch not in (0.,.25,1.,3.): raise ValueError("A_stop-selected original search endpoint required")
    exposures = math.ceil(epoch*n60)
    expected = dict(status="COMPLETE_VALID",run_id=condition["run_id"],family=condition["family"],epochs=[.25,1.,3.],seed=92601,
        reused_complete_artifact=False,search_exposures=3*n50,search_updates=updates,refit_exposures=exposures,refit_updates=math.ceil(exposures/16),
        search_fit_end=bounds["A_fit"][1],refit_fit_end=bounds["A_stop"][1],normalization_fit_end=bounds["A_stop"][1],
        refit_scaler_sha=plan["task_contract"]["refit_scaler_sha256"])
    if (any(base.get(key) != value for key,value in expected.items())
            or any(search.get(key) != value for key,value in dict(status="SEGMENT_COMPLETE",completed_epochs=[0.,.25,1.,3.],exposures=3*n50,updates=updates,train_origins=n50).items())
            or any(refit.get(key) != value for key,value in dict(status="SEGMENT_COMPLETE",completed_epochs=[0.]+([epoch] if epoch else []),exposures=exposures,updates=math.ceil(exposures/16),train_origins=n60).items())):
        raise ValueError("Complete fresh three-epoch search and selected fresh A60 refit required; zero refit remains valid")


def predict_call(api,condition,task,cache,public,foundation,receipt,deadline,*,split="V_select"):
    if split != "V_select": raise ValueError("Only full original V_select may be predicted")
    value = api(condition["run_id"],task,split,cache,public,foundation,receipt=receipt,fold="primary",deadline=deadline)
    cost = value["cost_record"]
    if (value["score"]["n_origins"] != 43 or value["metadata"].get("reused_prediction_cache")
            or cost.get("actual_F_streams") != 43*condition["latent_k"] or cost.get("expected_F_streams") != 43*condition["latent_k"]):
        raise ValueError("Fresh full V43 and actual K6/K11 foundation streams required")
    return value


def validate_probe_proof(value,condition,plan,plan_sha):
    expected = dict(status="PROBE_EFFECTIVE_UPDATES_COMPLETE_DISCARDED",mode="probe",run_id=condition["run_id"],
        plan_sha256=plan_sha,runner_sha256=sha(__file__),frozen_foundation_unchanged=True,
        original_source_registry_selection_T_pins_preserved=True,cuda_initialized=False,new_T_target_or_scoring=False)
    science = value.get("scientific_result") or {}
    updates = science.get("probe_updates")
    if updates not in (1,2,3) or any(value.get(key) != item for key,item in expected.items()): raise ValueError("This condition's actual successful frozen CPU probe RESULT is required")
    delta = dict(streams=16*updates*condition["latent_k"],row_calls=4*updates,physical_streams=16*updates*condition["latent_k"],physical_calls=4*updates)
    facts = dict(F_counter_delta=delta,full_A50_legal_origins=plan["task_contract"]["primary_split_ranges"]["A_fit"][1]-559,
        full_A50_loss_normalization_preserved=True,probe_state_discarded=True,probe_state_reused_for_fit=False,saved_model_or_optimizer=False,
        completed_fit=False,V_scoring=False,actual_parameter_gradients_finite=True,no_foundation_parameter_gradients=True,
        foundation_input_requires_grad_observed=True)
    if any(science.get(key) != item for key,item in facts.items()): raise ValueError("Actual probe F/gradient/full-loss/discard evidence differs")
    validate_input_gradient_events(science.get("foundation_input_requires_grad_events",[]),condition["latent_k"],updates)
    backward = validate_backward_gradient_events(science.get("actual_backward_input_gradient_events",[]),condition["latent_k"],updates)
    if any(science.get(key) != item for key,item in backward.items()):
        raise ValueError("Actual backward input-gradient summary differs from its recorded events")
    segment = science.get("segment") or {}
    if any(segment.get(key) != item for key,item in dict(status="SEGMENT_COMPLETE",updates=updates,exposures=16*updates,optimizer_updates_this_segment=updates,k_exposures={}).items()):
        raise ValueError("Actual effective probe updates are incomplete")


def environment(cache,asset):
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or any(key.startswith("INFORMS_GPU_") for key in os.environ)
            or any(os.environ.get(key) != "2" for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"))
            or any(os.environ.get(key) != "1" for key in ("HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE"))
            or Path(os.environ.get("INFORMS_ASSET_ROOT","")).resolve() != asset
            or Path(os.environ.get("INFORMS_CACHE_ROOT","")).resolve() != cache/NAME):
        raise ValueError("Explicit offline CPU2 separate-cache environment without GPU authorization required")


def execute(public,cache,asset,plan_sha,mode,run_id,deadline,output,*,probe_updates=3,probe_result=None,probe_result_sha=None):
    plan = metadata(public,cache,asset,plan_sha)
    condition = choose(plan,run_id)
    if mode not in {"probe","fit"} or probe_updates not in (1,2,3): raise ValueError("Explicit probe/fit and one through three probe updates required")
    environment(cache,asset)
    if deadline is None or not time.time() < deadline <= DEADLINE: raise ValueError("Live deadline within the original campaign wall required")
    output = Path(output).absolute()
    if output == public/NAME or not output.is_relative_to(public/NAME) or output.exists(): raise FileExistsError("Fresh diagnostic output below the separate public namespace required")
    relative(public,output.relative_to(public)); require_fresh(public,cache,mode,run_id)
    proof_path = None
    if mode == "fit":
        if probe_result is None or not re.fullmatch("[0-9a-f]{64}",probe_result_sha or ""): raise ValueError("Explicit per-condition actual probe RESULT and SHA required before fit")
        proof_path = Path(probe_result).absolute()
        if not proof_path.is_relative_to(public/NAME): raise ValueError("Probe RESULT must stay in this separate namespace")
        proof_path = relative(public,proof_path.relative_to(public))
        if proof_path.name != "RESULT.json" or sha(proof_path) != probe_result_sha: raise ValueError("Probe RESULT byte identity differs")
        validate_probe_proof(read(proof_path),condition,plan,plan_sha)
    used = sum(path.stat().st_size for root in (public,cache) for path in root.rglob("*") if path.is_file())
    if used+plan["admission_reserve_bytes"] > plan["storage_limit_bytes"]: raise ValueError("Original combined 30GiB storage cap including reserve reached")
    utility = load_dependency(public,UTILITY_PATH,UTILITY_SHA,"required_minima_private_utility")
    output.mkdir(parents=True,exist_ok=False)
    claims = public/NAME/"claims"; claims.mkdir(exist_ok=True)
    new(claims/(mode+"_"+run_id+".json"),dict(run_id=run_id,mode=mode,plan_sha256=plan_sha,runner_sha256=sha(__file__),started_unix=time.time()))
    sys.path.insert(0,str(public))
    import psutil
    import torch
    import data_contract as data
    import fit_base as fit
    import training
    import prediction_cache
    from gpu_runtime import FrozenBolt
    torch.set_num_threads(2)
    for module in (data,fit,training,prediction_cache):
        if Path(module.__file__).resolve() != public/(module.__name__+".py"): raise ValueError("Original scientific source root required")
    process,stop = psutil.Process(),threading.Event()
    def guard():
        while not stop.wait(.25):
            rss = process.memory_info().rss
            if rss > plan["max_RSS_bytes"] or time.time() >= deadline:
                new(output/"RESOURCE_STOP.json",dict(status="RESOURCE_STOP",rss_bytes=rss,deadline_unix=deadline)); os._exit(98)
    watcher = threading.Thread(target=guard,daemon=True); watcher.start()
    before,started = utility.rss_record(process),time.perf_counter()
    new(output/"STARTED.json",dict(actual_pid=process.pid,actual_create_time=process.create_time(),run_id=run_id,mode=mode,plan_sha256=plan_sha,
        runner_sha256=sha(__file__),deadline_unix=deadline,resource_before=before,new_T_scoring=False,no_model_state_reuse=True))
    foundation,science = None,None
    status,error_class,post_error,phase = "FAILED_PRESERVED",None,None,"Task_binding"
    failures = []; frozen,source_post = False,False
    try:
        task = data.load_task("beijing",cache_dir=cache)
        validator = load_dependency(public,TASK_VALIDATOR_PATH,TASK_VALIDATOR_SHA,"required_minima_original_task_validator")
        task_binding = validator.validate_task(task,dict(task_contracts=dict(beijing=plan["task_contract"])),"beijing",data,fit)
        new(output/"TASK_BINDING.json",task_binding)
        if torch.cuda.is_initialized() or process.memory_info().rss > plan["max_RSS_bytes"]: raise RuntimeError("CPU/RSS contract violated before foundation load")
        phase = "frozen_foundation_loading"
        foundation = FrozenBolt(asset/plan["model_asset_relative_path"],48,128,False,device="cpu")
        if foundation.initial_hash != plan["foundation_tensor_sha256"] or foundation.instance_norm_source_sha256 != plan["instance_norm_source_sha256"]:
            raise ValueError("Actual foundation/normalization identity differs")
        foundation.assert_frozen_unchanged(); initial = counters(foundation)
        spec = condition["spec"]
        if mode == "probe":
            phase = "fresh_A50_probe_state_discarded"
            end = data.split_ranges(task,"primary")["A_fit"][1]
            model,norm = fit.create_model(spec,task,end,foundation)
            if fit.scaler_fingerprint(norm["mean"],norm["std"]) != plan["task_contract"]["search_scaler_sha256"]: raise ValueError("Probe A50 normalization differs")
            windows = training.FastWindows(dict(task,physical=task["physical"][:end],observed=task["observed"][:end]),norm["mean"],norm["std"],"cpu")
            legal = data.origins((0,end),512,48,1)
            sampler = training.PermutationStream(legal,92601)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=0)
            events,backward_events = [],[]
            forward_handle,backward_handle = attach_probe_hooks(foundation,torch,events,backward_events)
            try:
                science = probe_call(training.train_segment,model,windows,legal,sampler,optimizer,
                    lambda x,k:fit.predict_model(model,x,foundation,condition["family"],k),foundation,condition,deadline,probe_updates)
                validate_input_gradient_events(events,condition["latent_k"],probe_updates)
                backward = validate_backward_gradient_events(backward_events,condition["latent_k"],probe_updates)
                grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
                if not grads or any(not torch.isfinite(g).all() for g in grads): raise FloatingPointError("Actual trainable-parameter gradients missing or nonfinite")
                foundation.assert_frozen_unchanged()
                science.update(actual_parameter_gradients_finite=True,no_foundation_parameter_gradients=True,
                    foundation_input_requires_grad_observed=True,foundation_input_requires_grad_events=events,
                    actual_backward_input_gradient_events=backward_events,**backward,basis_initialization=norm["basis_initialization"])
            finally:
                forward_handle.remove()
                backward_handle.remove()
            del model,optimizer,windows,sampler
            status = "PROBE_EFFECTIVE_UPDATES_COMPLETE_DISCARDED"
        else:
            phase = "original_fresh_search_and_A60_refit"
            receipt = fit.fit_base(spec,task,cache/NAME,public/NAME,foundation,deadline)
            science = dict(fit_receipt=receipt,full_V_completed=False,independent_rescore_completed=False,checkpoint_subset_replay_completed=False)
            if receipt.get("status") == "COMPLETE_VALID":
                folder = public/NAME/"runs"/run_id
                validate_complete_fit(receipt,read(folder/"search/RECEIPT.json"),read(folder/"refit/RECEIPT.json"),condition,plan)
                phase = "original_full_V_prediction"
                prediction = predict_call(prediction_cache.predict_parent,condition,task,cache/NAME,public/NAME,foundation,receipt,deadline)
                path = public/NAME/"predictions"/run_id/"V_select/PREDICTION_MANIFEST.json"
                science.update(full_V_completed=True,V_score=prediction["score"],V_cost_record=prediction["cost_record"],
                    V_manifest_relative_path=path.relative_to(public).as_posix(),V_manifest_sha256=sha(path),
                    strict_search_depth_complete=True,selected_refit_epoch=receipt["selected_epoch"],original_training_status=receipt.get("training_status"),
                    convergence_established=False,V_cost_F_calls_is_bundle_bookkeeping_not_actual_row_calls=True)
                status = "COMPLETE_FRESH_CPU_FIT_AND_FULL_V_PENDING_INDEPENDENT_VALIDATION"
            else: status = receipt.get("status","FAILED_PRESERVED")
        science["session_F_counter_delta"] = {key:value-initial[key] for key,value in counters(foundation).items()}
        foundation.assert_frozen_unchanged()
        if torch.cuda.is_initialized(): raise RuntimeError("Unexpected CUDA initialization")
    except Exception as error:
        error_class = type(error).__name__
        failures.append(utility.save_private_failure(cache/NAME,cache,mode,run_id,phase,error))
    finally:
        try:
            if foundation is not None: foundation.assert_frozen_unchanged(); frozen=True
            metadata(public,cache,asset,plan_sha)
            if proof_path is not None and sha(proof_path) != probe_result_sha: raise ValueError("Actual probe evidence changed during fresh fit")
            source_post = True
        except Exception as error:
            post_error = type(error).__name__
            failures.append(utility.save_private_failure(cache/NAME,cache,mode,run_id,"postflight",error))
        after = utility.rss_record(process); stop.set(); watcher.join(timeout=1)
        success = error_class is None and post_error is None and source_post
        new(output/"RESULT.json",dict(status=status if success else "FAILED_PRESERVED",run_id=run_id,mode=mode,stage=plan["stage"],
            scientific_result=science,error_class=error_class,postflight_error_class=post_error,private_failures=failures,
            actual_pid=process.pid,actual_create_time=process.create_time(),elapsed_process_wall_seconds=time.perf_counter()-started,
            actual_CPU_process_seconds=after["CPU_process_seconds"]-before["CPU_process_seconds"],resource_after=after,
            frozen_foundation_unchanged=frozen,original_source_registry_selection_T_pins_preserved=source_post,cuda_initialized=torch.cuda.is_initialized(),
            new_T_target_or_scoring=False,post_T_development=True,observations_not_a_new_independent_holdout=True,
            original_GPU_coverage_completed=False,original_registry_statuses_updated=False,B0_reference_reuse_assumed=False,
            independent_validation_required=mode=="fit",parent_must_report_forced_exit_without_final_result=True,
            parent_must_enforce_prior_queues_closed_cost_quiet_and_owned_process_closure=True,
            archive_deserialization_scope="Original safeFIT Task may physically deserialize/hash campaign T bytes; only A50/A_stop/A60/V computation is requested",
            plan_sha256=plan_sha,runner_sha256=sha(__file__),finished_unix=time.time()))
    return 0 if success and status in {"PROBE_EFFECTIVE_UPDATES_COMPLETE_DISCARDED","COMPLETE_FRESH_CPU_FIT_AND_FULL_V_PENDING_INDEPENDENT_VALIDATION"} else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for root in ("public","cache","asset"): parser.add_argument("--"+root+"-root")
    parser.add_argument("--plan-sha"); parser.add_argument("--seal-plan",action="store_true"); parser.add_argument("--execute",action="store_true")
    parser.add_argument("--mode",choices=("probe","fit")); parser.add_argument("--run-id"); parser.add_argument("--deadline",type=float)
    parser.add_argument("--probe-updates",type=int,choices=(1,2,3),default=3); parser.add_argument("--probe-result"); parser.add_argument("--probe-result-sha")
    parser.add_argument("--output"); args = parser.parse_args(argv)
    roots = (args.public_root,args.cache_root,args.asset_root)
    if args.seal_plan and args.execute: parser.error("Metadata seal and scientific execution are separate explicit modes")
    if args.seal_plan or args.execute or any(roots):
        if not all(roots): parser.error("All three explicit original roots required")
        public,cache,asset = (Path(value).resolve() for value in roots)
    if args.seal_plan:
        if not args.output: parser.error("A fresh private candidate output is required")
        print(json.dumps(seal_plan(public,cache,asset,Path(args.output).absolute())))
        return 0
    if args.execute:
        if not all((args.plan_sha,args.mode,args.run_id,args.deadline,args.output)): parser.error("Explicit sealed PLAN/one mode/one ID/deadline/new output required")
        return execute(public,cache,asset,args.plan_sha,args.mode,args.run_id,args.deadline,Path(args.output).absolute(),
            probe_updates=args.probe_updates,probe_result=args.probe_result,probe_result_sha=args.probe_result_sha)
    if any(roots):
        if args.plan_sha: metadata(public,cache,asset,args.plan_sha)
        else: proposal(public,cache,asset)
    print(json.dumps(dict(status="METADATA_ONLY_PREPARATION_NOT_EXECUTED",registered_conditions=6,files_written=0,
        numpy_imported="numpy" in sys.modules,torch_imported="torch" in sys.modules,psutil_imported="psutil" in sys.modules,
        actual_Task_model_array_training_or_child_operations=0,runner_sha256=sha(__file__),new_T_scoring=False)),flush=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
