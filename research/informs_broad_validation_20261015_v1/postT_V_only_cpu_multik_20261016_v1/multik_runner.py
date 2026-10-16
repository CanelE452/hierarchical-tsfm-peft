"""Prepared post-T CPU multi-K development: five fresh fits and twelve V outputs.

Original scientific APIs are called unchanged. Default execution is stdlib-only;
the ROOT-sealed PLAN and explicit --execute are required for real work.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

NAME = "postT_V_only_cpu_multik_20261016_v1"
UTILITY_PATH = "postT_V_only_cpu_lag_20261016_v1/cpu_lag_runner.py"
UTILITY_SHA = "36c75b38cd2c072bac4466129249cc5213c2df679ad80232d4ac511ab6d7e7f1"
PCA_RUNNER_PATH = "postT_V_only_cpu_fixed_pca_20261016_v1/fixed_pca_runner.py"
PCA_RUNNER_SHA = "d73d866a6d03061d00fb7ca015a8f3b41e273b73011b22f0c1cd0fbbfb073dd9"
LOCK_PATH = "ordered/primary_beijing/FAMILY_LOCK.json"
LOCK_SHA = "6954164639efcef4605bdcd22af91a48d262a838ab78bb52852f13478a6ff747"
GRID = [3, 6, 11]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def relative(root, name):
    root, name = Path(root).resolve(), Path(name)
    path = root / name
    if name.is_absolute() or ".." in name.parts or any(p.is_symlink() for p in [path]+list(path.parents)):
        raise ValueError("Safe root-relative nonsymlink path required")
    path = path.resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("Path escapes declared root")
    return path


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def identity(role, k):
    return f"postT_cpu_beijing_E4_{role}_k{k}_s92601_v1"


def original_ordered_specs(lock):
    common = dict(family="B0", latent_k=11, seed=92601, epochs=[.25, 1., 3.], lr=.001, fold="primary",
                  microbatch=4, effective_batch=16, device="cpu", family_lock_sha=fingerprint(lock))
    return dict(shared=dict(common, run_id="primary_beijing_E4_ORDERED_MULTI_K_k11_s92601", k_grid=list(GRID)),
                max_only=dict(common, run_id="primary_beijing_E4_MAXK_PCA_TRAINED_k11_s92601", k_grid=[11]))


def recipe_manifest(lock, source39=None):
    """Return explicit planned records; this function executes no scientific API."""
    sources, fits = source39 or {}, []
    for kind, role in (("shared", "ORDERED_MULTI_K"), ("max_only", "MAXK_PCA_TRAINED")):
        original = original_ordered_specs(lock)[kind]
        old = original["run_id"]
        spec = dict(original, run_id=identity(role, 11), dataset="beijing", F_row_microbatch=128, checkpoint_inputs=False,
                    stage_label="postT_cpu_fresh_V_only_multik", provenance=dict(upstream_registered_id=old))
        fits.append(dict(run_id=spec["run_id"], role=role, upstream_registered_id=old, spec=spec,
                         recipe_reference=dict(relative_path="ordered_pilot.py", sha256=sources.get("ordered_pilot.py"))))
    for k in GRID:
        old = f"primary_beijing_E1_B0_k{k}_s92601"
        spec = dict(run_id=identity("INDEPENDENT_B0", k), dataset="beijing", family="B0", latent_k=k, seed=92601,
            fold="primary", epochs=[.25, 1., 3.], lr=.001, device="cpu", microbatch=4, effective_batch=16,
            F_row_microbatch=128, checkpoint_inputs=False, lora_rank=8, lora_alpha=16,
            stage_label="postT_cpu_fresh_V_only_multik", provenance=dict(upstream_registered_id=old))
        fits.append(dict(run_id=spec["run_id"], role="INDEPENDENT_B0", upstream_registered_id=old, spec=spec,
                         recipe_reference=dict(relative_path="campaign.py", sha256=sources.get("campaign.py"))))
    outputs = []
    for k in GRID:
        outputs.append(dict(run_id=identity("INDEPENDENT_B0", k), role="INDEPENDENT_B0", latent_k=k,
            kind="independent", variant=None, parent_run_id=identity("INDEPENDENT_B0", k),
            upstream_registered_id=f"primary_beijing_E1_B0_k{k}_s92601", split="V_select"))
        shared_id = identity("ORDERED_MULTI_K", 11) if k == 11 else identity("SHARED_PREFIX", k)
        old_shared = "primary_beijing_E4_ORDERED_MULTI_K_k11_s92601" + ("" if k == 11 else f"__prefix_k{k}")
        outputs.append(dict(run_id=shared_id, role="ORDERED_MULTI_K" if k == 11 else "SHARED_PREFIX", latent_k=k,
            kind="ordered", variant="shared", parent_run_id=identity("ORDERED_MULTI_K", 11),
            upstream_registered_id=old_shared, split="V_select"))
        outputs.append(dict(run_id=identity("MAXK_PCA_TRAINED", 11) if k == 11 else identity("POSTHOC_PREFIX", k),
            role="MAXK_PCA_TRAINED" if k == 11 else "POSTHOC_PREFIX", latent_k=k, kind="ordered",
            variant="max_only" if k == 11 else "posthoc", parent_run_id=identity("MAXK_PCA_TRAINED", 11),
            upstream_registered_id="primary_beijing_E4_MAXK_PCA_TRAINED_k11_s92601" if k == 11 else f"primary_beijing_E4_POSTHOC_PREFIX_k{k}_s92601",
            split="V_select"))
        outputs.append(dict(run_id=identity("NESTED_PCA", k), role="NESTED_PCA", latent_k=k, kind="nested", variant="nested_pca",
            parent_run_id=None, upstream_registered_id=f"primary_beijing_E4_NESTED_PCA_k{k}_s92601", split="V_select"))
    return fits, outputs


def build_specs(make_specs_api, task, lock, source39=None):
    actual = make_specs_api(task, lock, epochs=[.25, 1., 3.], device="cpu")
    if actual != original_ordered_specs(lock):
        raise ValueError("Original make_specs output differs; max-only [C] must never be removed")
    fits, _ = recipe_manifest(lock, source39)
    specs = {row["run_id"]: dict(row["spec"]) for row in fits}
    for label, role in (("shared", "ORDERED_MULTI_K"), ("max_only", "MAXK_PCA_TRAINED")):
        preserved = dict(actual[label])
        for key in ("run_id", "dataset", "F_row_microbatch", "checkpoint_inputs", "stage_label", "provenance"):
            preserved[key] = specs[identity(role, 11)][key]
        if preserved != specs[identity(role, 11)]:
            raise ValueError("Scientific E4 fields changed beyond declared fresh CPU provenance")
        specs[identity(role, 11)] = preserved
    return specs


def metadata(public, cache, asset, plan_sha=None):
    path = Path(public) / NAME / "PLAN.json"
    if plan_sha is not None and sha(path) != plan_sha:
        raise ValueError("Explicit PLAN SHA differs")
    plan = read(path)
    fixed = dict(stage="POST_T_V_ONLY_DEVELOPMENT", output_public_relative_root=NAME, output_cache_relative_root=NAME,
        CPU_threads=2, max_RSS_bytes=3221225472, maximum_concurrent_workers=2, total_owned_RSS_limit_bytes=6442450944,
        storage_limit_bytes=32212254720, deadline_unix=1792222550)
    if any(plan.get(key) != value for key, value in fixed.items()) or plan["runner_sha256"] != sha(__file__):
        raise ValueError("Runner, separate namespace or resource contract differs")
    if len(plan["source39"]) != 39 or not {"EXPERIMENT_REGISTRY.csv", "SELECTION_LOCK.json", "DATA_CONTRACT.json", "SCORES.csv"}.issubset(plan["immutable_public"]):
        raise ValueError("Original source39, registry, observed-T lock, data and scores must remain pinned")
    for pins in (plan["source39"], plan["immutable_public"]):
        for name, digest in pins.items():
            if sha(relative(public, name)) != digest:
                raise ValueError("Original scientific/public bytes changed: " + name)
    if read(Path(public) / "SELECTION_LOCK.json").get("locked") is not True:
        raise ValueError("Original T selection must remain locked")
    for key, expected_path, expected_sha in (("shared_utility", UTILITY_PATH, UTILITY_SHA), ("pca_runner", PCA_RUNNER_PATH, PCA_RUNNER_SHA)):
        if (plan[key+"_relative_path"] != expected_path or plan[key+"_sha256"] != expected_sha
                or sha(relative(public, expected_path)) != expected_sha):
            raise ValueError("Pinned stdlib utility dependency changed")
    if plan["family_lock"] != dict(relative_path=LOCK_PATH, sha256=LOCK_SHA) or sha(relative(public, LOCK_PATH)) != LOCK_SHA:
        raise ValueError("Original pre-E4 family lock must be reused exactly without reselection")
    lock = read(relative(public, LOCK_PATH))
    expected_lock = dict(locked=True, chosen_before_E4_training=True, chosen_family="B0", dataset="beijing", fold="primary", seed=92601, k_grid=GRID)
    if any(lock.get(key) != value for key, value in expected_lock.items()) or len(lock["source_files"]) != 8:
        raise ValueError("Pretraining chosen family/seed/grid/source8 differs")
    if any(plan["source39"].get(name) != digest for name, digest in lock["source_files"].items()):
        raise ValueError("Pretraining source8 differs from current frozen source39")
    reference = plan["task_contract_reference"]
    if reference["relative_path"] != "postT_V_only_cpu_fixed_pca_20261016_v1/PLAN.json" or sha(relative(public, reference["relative_path"])) != reference["sha256"]:
        raise ValueError("Original fixed-PCA task/scaler reference changed")
    pca_plan = read(relative(public, reference["relative_path"]))
    if (plan["source39"] != pca_plan["source39"]
            or any(plan["immutable_public"].get(key) != value for key,value in pca_plan["immutable_public"].items())):
        raise ValueError("Original source39 and full registry/selection/score/cost/T guard set must be retained")
    contract = plan["task_contract"]
    original = read(Path(public) / "DATA_CONTRACT.json")["assets"]["beijing"]
    preserved = {key:value for key,value in contract.items() if key not in {"search_scaler_sha256", "refit_scaler_sha256"}}
    if (preserved != original or contract != pca_plan["task_contracts"]["beijing"]
            or contract["shape"] != [21038,11] or contract["L"] != 512 or contract["H"] != 48 or len(contract["columns"]) != 11
            or contract["legal_origin_counts_primary"]["V_select"] != 43 or lock["data_sha"] != contract["data_sha256"]
            or lock["scaler_sha"] != contract["refit_scaler_sha256"]):
        raise ValueError("Beijing full task/order/array/scaler/family-lock binding differs")
    model = relative(asset, plan["model_asset_relative_path"])
    if not model.is_dir() or model.name != plan["model_revision"]:
        raise ValueError("Pinned local foundation snapshot absent")
    if (sha(relative(asset, plan["instance_norm_source_relative_path"])) != plan["instance_norm_source_sha256"]
            or sha(Path(public).parent / "temporal_frontend_campaign_20261015_v1/runtime.py") != plan["instance_norm_source_sha256"]):
        raise ValueError("Actual instance normalization source differs")
    fits, outputs = recipe_manifest(lock, plan["source39"])
    if plan["fits"] != fits or plan["outputs"] != outputs:
        raise ValueError("Exactly five original recipes and twelve distinct output identities required")
    original_rows = rows(Path(public) / "EXPERIMENT_REGISTRY.csv")
    old = {row["experiment_id"]:row for row in original_rows}
    if len(old) != len(original_rows):
        raise ValueError("Original registry contains duplicate identities")
    for fit_row in fits:
        old_row = old.get(fit_row["upstream_registered_id"], {})
        original_family = "B0" if fit_row["role"] == "INDEPENDENT_B0" else fit_row["role"]
        expected = dict(dataset="beijing", family=original_family, latent_k=str(fit_row["spec"]["latent_k"]), seed="92601", fold="primary", length="512", horizon="48")
        if any(old_row.get(key) != value for key,value in expected.items()):
            raise ValueError("Original registered parent identity differs")
    registry = relative(Path(public)/NAME, plan["postT_registry_relative_path"])
    if sha(registry) != plan["postT_registry_sha256"]:
        raise ValueError("Separate twelve-row preregistration changed")
    registered = rows(registry)
    by_id = {row["experiment_id"]:row for row in registered}
    if len(registered) != 12 or set(by_id) != {row["run_id"] for row in outputs} or set(by_id) & set(old):
        raise ValueError("Exactly twelve distinct new registered IDs required")
    for output in outputs:
        expected = dict(dataset="beijing", family=output["role"], latent_k=str(output["latent_k"]), seed="92601", fold="primary",
                        length="512", horizon="48", device="cpu", stage="POST_T_V_ONLY_DEVELOPMENT", upstream_registered_id=output["upstream_registered_id"])
        if any(by_id[output["run_id"]].get(key) != value for key,value in expected.items()):
            raise ValueError("Registered role differs; core model family remains B0 separately")
    return plan


def choose(plan, mode, run_id):
    if mode not in {"probe", "fit", "prefix", "nested"}:
        raise ValueError("Only explicit probe/fit/V-prefix/nested modes are supported")
    collection = plan["fits"] if mode in {"probe", "fit"} else plan["outputs"]
    matches = [row for row in collection if row["run_id"] == run_id]
    if len(matches) != 1:
        raise ValueError("One explicit preregistered identity required")
    row = matches[0]
    if mode == "probe" and row["role"] not in {"ORDERED_MULTI_K", "MAXK_PCA_TRAINED"}:
        raise ValueError("Three-update probes are shared/max-only controls only")
    if mode in {"prefix", "nested"} and (row.get("split") != "V_select" or (row["kind"] == "nested") != (mode == "nested")):
        raise ValueError("This mode does not match the registered V output kind")
    return row


def require_fresh(public, cache, mode, run_id):
    for root in (Path(public), Path(cache), Path(public)/NAME, Path(cache)/NAME):
        kinds = ("runs", "predictions", "ordered_predictions") if mode in {"fit", "probe"} else ("predictions", "ordered_predictions")
        if any((root/kind/run_id).exists() for kind in kinds):
            raise FileExistsError("Existing run/output preserved; no old partial resumption")
    if (Path(public)/NAME/"claims"/(mode+"_"+run_id+".json")).exists():
        raise FileExistsError("Existing claimed attempt preserved")


def counters(foundation):
    return {key:int(getattr(foundation,key,0)) for key in ("streams","row_calls","physical_streams","physical_calls")}


def fit_call(api, spec, task, cache, public, foundation, deadline):
    receipt = api(spec, task, cache, public, foundation, deadline)
    if receipt.get("reused_complete_artifact"):
        raise ValueError("A fresh independent CPU fit cannot reuse an old selected bundle")
    return receipt


def validate_complete_fit(base, search, refit, fit_row):
    if (base.get("status") != "COMPLETE_VALID" or base.get("run_id") != fit_row["run_id"] or base.get("family") != "B0"
            or base.get("reused_complete_artifact") or search.get("status") != "SEGMENT_COMPLETE"
            or search.get("completed_epochs") != [0.,.25,1.,3.] or search.get("actual_epochs") != 3.
            or search.get("exposures") != 3*search.get("train_origins",-1) or base.get("search_exposures") != search.get("exposures")
            or refit.get("status") != "SEGMENT_COMPLETE" or refit.get("selected_epoch") != base.get("selected_epoch")):
        raise ValueError("Full original search endpoints/three-epoch exposure or selected refit are incomplete")
    selected = base["selected_epoch"]
    if (selected not in [0.,.25,1.,3.] or base.get("refit_exposures") != refit.get("exposures")
            or base.get("refit_updates") != refit.get("updates")
            or (selected == 0 and (refit.get("exposures") != 0 or refit.get("updates") != 0))
            or (selected > 0 and (not isinstance(refit.get("train_origins"),int) or refit["train_origins"] < 1
                or refit.get("exposures") != math.ceil(selected*refit["train_origins"])))):
        raise ValueError("Selected zero-epoch refit must remain an actual zero-update initialization")


def predict_call(standard_api, prefix_api, output, task, cache, public, foundation, receipt, deadline):
    if output.get("split") != "V_select" or output["latent_k"] not in GRID:
        raise ValueError("Only the registered full V cohort and K3/6/11 are supported")
    before = counters(foundation)
    foundation.assert_frozen_unchanged()
    try:
        if output["kind"] == "independent":
            value = standard_api(output["run_id"], task, "V_select", cache, public, foundation, receipt=receipt, fold="primary", deadline=deadline)
        else:
            value = prefix_api(output["run_id"], task, output["latent_k"], cache, public, foundation,
                parent_receipt=receipt, variant=output["variant"], fold="primary", deadline=deadline)
        expected = 43*output["latent_k"]
        if (value["score"]["n_origins"] != 43 or value["metadata"].get("reused_prediction_cache")
                or value["cost_record"].get("actual_F_streams") != expected or value["cost_record"].get("expected_F_streams") != expected
                or counters(foundation)["streams"]-before["streams"] != expected):
            raise ValueError("Fresh full V inference and actual requested K-row foundation streams required")
        return value
    finally:
        foundation.assert_frozen_unchanged()


def probe_call(segment_api, model, windows, legal, sampler, optimizer, predict, foundation, fit_row, deadline):
    role = fit_row["role"]
    if role not in {"ORDERED_MULTI_K", "MAXK_PCA_TRAINED"}:
        raise ValueError("Only shared/max controls have an approved three-update probe")
    grid = GRID if role == "ORDERED_MULTI_K" else [11]
    expected_sequence = [k for k in (GRID if len(grid)==3 else [11,11,11]) for _ in range(4)]
    seen, before = [], counters(foundation)
    def recorded_predict(x, k):
        seen.append(k)
        return predict(x,k)
    foundation.assert_frozen_unchanged()
    try:
        segment = segment_api(model, windows, legal, sampler, optimizer, 48, recorded_predict,
            effective_batch=16, microbatch=4, starting_updates=0, budget_deadline=deadline, k_grid=list(grid))
        delta = {key:value-before[key] for key,value in counters(foundation).items()}
        expected_streams = 320 if len(grid)==3 else 528
        expected_exposures = {"3":16,"6":16,"11":16} if len(grid)==3 else {"11":48}
        if (segment.get("status") != "SEGMENT_COMPLETE" or segment.get("updates") != 3 or segment.get("exposures") != 48
                or segment.get("optimizer_updates_this_segment") != 3 or segment.get("k_exposures") != expected_exposures
                or seen != expected_sequence or delta != dict(streams=expected_streams, row_calls=12, physical_streams=expected_streams, physical_calls=12)
                or any(not isinstance(segment.get(key),(int,float)) or not math.isfinite(segment[key]) for key in ("gradient_norm_min","gradient_norm_max"))):
            raise ValueError("Original three effective updates, K cycle, finite gradients or actual F execution differ")
        return dict(segment=segment, observed_microbatch_K=seen, F_counter_delta=delta, full_A50_legal_origins=len(legal),
            full_A50_loss_normalization_preserved=True, probe_state_discarded=True, probe_state_reused_for_fit=False,
            saved_model_or_optimizer=False, V_scoring=False)
    finally:
        foundation.assert_frozen_unchanged()


def load_dependency(public, path, name):
    spec = importlib.util.spec_from_file_location(name, relative(public,path))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def environment(cache, asset):
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or any(key.startswith("INFORMS_GPU_") for key in os.environ)
            or any(os.environ.get(key) != "2" for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"))
            or any(os.environ.get(key) != "1" for key in ("HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE"))
            or Path(os.environ.get("INFORMS_ASSET_ROOT", "")).resolve() != asset
            or Path(os.environ.get("INFORMS_CACHE_ROOT", "")).resolve() != cache/NAME):
        raise ValueError("Explicit separate-cache offline CPU2 process without GPU authorization required")


def execute(public, cache, asset, plan_sha, mode, run_id, deadline, output):
    if not plan_sha:
        raise ValueError("Explicit ROOT-sealed PLAN SHA required")
    plan = metadata(public,cache,asset,plan_sha)
    condition = choose(plan,mode,run_id)
    if deadline is None or not time.time() < deadline <= plan["deadline_unix"]:
        raise ValueError("Live deadline inside original campaign wall required")
    environment(cache,asset)
    require_fresh(public,cache,mode,run_id)
    post_public, post_cache = public/NAME, cache/NAME
    if output == post_public or not output.is_relative_to(post_public) or output.exists():
        raise ValueError("New diagnostic output inside separate namespace required")
    used = sum(p.stat().st_size for root in (public,cache) for p in root.rglob("*") if p.is_file())
    if used+256*1024**2 > plan["storage_limit_bytes"]:
        raise ValueError("Combined original/new storage cap reached")
    output.mkdir(parents=True,exist_ok=False)
    post_cache.mkdir(parents=True,exist_ok=True)
    (post_public/"claims").mkdir(exist_ok=True)
    new(post_public/"claims"/(mode+"_"+run_id+".json"),dict(mode=mode,run_id=run_id,plan_sha256=plan_sha,started_unix=time.time()))
    utility = load_dependency(public,UTILITY_PATH,"multik_stdlib_utility")
    pca_helper = load_dependency(public,PCA_RUNNER_PATH,"multik_task_validator")
    sys.path.insert(0,str(public))
    import psutil
    import torch
    import data_contract as data
    import fit_base as fit
    import ordered_pilot as ordered
    import prediction_cache
    import training
    from gpu_runtime import FrozenBolt
    torch.set_num_threads(2)
    process, stop = psutil.Process(os.getpid()), threading.Event()
    def guard():
        while not stop.wait(.25):
            rss = process.memory_info().rss
            if rss > plan["max_RSS_bytes"] or time.time() >= deadline:
                new(output/"RESOURCE_STOP.json",dict(status="RESOURCE_STOP",rss_bytes=rss,deadline_unix=deadline,actual_pid=process.pid,at_unix=time.time()))
                os._exit(98)
    watcher = threading.Thread(target=guard,daemon=True)
    watcher.start()
    before, started = utility.rss_record(process), time.perf_counter()
    new(output/"STARTED.json",dict(mode=mode,run_id=run_id,stage=plan["stage"],plan_sha256=plan_sha,runner_sha256=sha(__file__),
        actual_pid=process.pid,actual_create_time=process.create_time(),deadline_unix=deadline,resource_before=before,
        original_GPU_gap_completed=False,new_T_scoring=False,
        archive_deserialization_scope="Whole approved safeFIT task including campaign-T bytes is loaded/hashed; only original A50/A_stop/A60/V computations requested."))
    foundation, science, frozen, source_post = None, None, False, False
    error_class, postflight_error, failures, status, phase = None, None, [], "FAILED_PRESERVED", "task_loading"
    try:
        task = data.load_task("beijing",cache_dir=cache)
        binding = pca_helper.validate_task(task,dict(task_contracts=dict(beijing=plan["task_contract"])),"beijing",data,fit)
        new(output/"TASK_BINDING.json",binding)
        if torch.cuda.is_initialized() or process.memory_info().rss > plan["max_RSS_bytes"]:
            raise RuntimeError("CPU visibility/RSS violation before foundation load")
        lock = read(relative(public,LOCK_PATH))
        specs = build_specs(ordered.make_specs,task,lock,plan["source39"])
        phase = "foundation_loading"
        foundation = FrozenBolt(asset/plan["model_asset_relative_path"],48,128,False,device="cpu")
        if foundation.initial_hash != plan["foundation_tensor_sha256"] or foundation.instance_norm_source_sha256 != plan["instance_norm_source_sha256"]:
            raise ValueError("Actual frozen foundation or normalization identity differs")
        foundation.assert_frozen_unchanged()
        initial_counters = counters(foundation)
        if mode == "probe":
            phase = "fresh_A50_three_update_probe"
            end = data.split_ranges(task,"primary")["A_fit"][1]
            model,norm = fit.create_model(specs[run_id],task,end,foundation)
            if fit.scaler_fingerprint(norm["mean"],norm["std"]) != plan["task_contract"]["search_scaler_sha256"]:
                raise ValueError("Probe A50 normalization differs")
            prefix = dict(task,physical=task["physical"][:end],observed=task["observed"][:end])
            windows = training.FastWindows(prefix,norm["mean"],norm["std"],"cpu")
            legal = data.origins((0,end),512,48,1)
            sampler = training.PermutationStream(legal,92601)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=0)
            predict = lambda x,k: fit.predict_model(model,x,foundation,"B0",k)
            science = probe_call(training.train_segment,model,windows,legal,sampler,optimizer,predict,foundation,condition,deadline)
            gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            if not gradients or any(not torch.isfinite(g).all() for g in gradients):
                raise FloatingPointError("Actual probe parameter gradients missing or nonfinite")
            science.update(actual_parameter_gradients_finite=True,no_foundation_parameter_gradients=True,
                basis_initialization=norm["basis_initialization"],convergence_claimed=False)
            del model,optimizer,sampler,windows
            status = "COMPLETE_THREE_UPDATE_PROBE_DISCARDED"
        elif mode == "fit":
            phase = "original_fresh_search_and_refit"
            receipt = fit_call(fit.fit_base,specs[run_id],task,post_cache,post_public,foundation,deadline)
            science = dict(fit_receipt=receipt,registered_role=condition["role"],actual_core_family="B0",convergence_claimed=False)
            if receipt["status"] == "COMPLETE_VALID":
                search = read(post_public/"runs"/run_id/"search/RECEIPT.json")
                refit = read(post_public/"runs"/run_id/"refit/RECEIPT.json")
                validate_complete_fit(receipt,search,refit,condition)
                science.update(strict_search_depth_complete=True,selected_epoch_zero_preserved=receipt["selected_epoch"]==0,
                    selected_refit_epoch=receipt["selected_epoch"],last_checkpoint_best=receipt.get("last_cp_best"),
                    original_training_status=receipt.get("training_status"),full_V_completed=False)
                status = "COMPLETE_FRESH_CPU_FIT_NO_V_YET"
            else:
                status = receipt["status"]
        else:
            phase = "parent_binding"
            receipt = None
            parent = condition["parent_run_id"]
            if parent is not None:
                fit_row = next(row for row in plan["fits"] if row["run_id"]==parent)
                receipt = read(post_public/"runs"/parent/"BASE_FIT_RECEIPT.json")
                validate_complete_fit(receipt,read(post_public/"runs"/parent/"search/RECEIPT.json"),
                    read(post_public/"runs"/parent/"refit/RECEIPT.json"),fit_row)
                if read(post_public/"runs"/parent/"SOURCE_CONTRACT.json")["spec"] != specs[parent]:
                    raise ValueError("Prefix parent scientific recipe differs")
            phase = "original_full_V_prediction"
            value = predict_call(prediction_cache.predict_parent,ordered.predict_prefix,condition,task,post_cache,post_public,foundation,receipt,deadline)
            manifest = post_public/("predictions" if condition["kind"]=="independent" else "ordered_predictions")/run_id
            manifest = manifest/("V_select/PREDICTION_MANIFEST.json" if condition["kind"]=="independent" else "PREDICTION_MANIFEST.json")
            science = dict(V_score=value["score"],V_cost_record=value["cost_record"],full_V_completed=True,
                V_manifest_relative_path=manifest.relative_to(public).as_posix(),V_manifest_sha256=sha(manifest),
                registered_role=condition["role"],variant=condition["variant"],parent_run_id=parent,
                prefix_in_registered_training_objective=condition["kind"]=="independent" or condition["variant"] in {"shared","max_only"},
                selected_parent_refit_epochs=None if receipt is None else receipt["selected_epoch"],
                nested_fixed_transform=condition["kind"]=="nested",convergence_claimed=False,
                V_cost_F_calls_is_bundle_bookkeeping_not_actual_row_calls=True)
            status = "COMPLETE_FULL_V_ONLY"
        science["session_F_counter_delta"] = {key:value-initial_counters[key] for key,value in counters(foundation).items()}
        foundation.assert_frozen_unchanged()
        if torch.cuda.is_initialized():
            raise RuntimeError("Unexpected CUDA initialization")
    except Exception as error:
        error_class = type(error).__name__
        failures.append(utility.save_private_failure(post_cache,cache,mode,run_id,phase,error))
    finally:
        try:
            if foundation is not None:
                foundation.assert_frozen_unchanged()
                frozen = True
            metadata(public,cache,asset,plan_sha)
            source_post = True
        except Exception as error:
            postflight_error = type(error).__name__
            failures.append(utility.save_private_failure(post_cache,cache,mode,run_id,"postflight",error))
        after = utility.rss_record(process)
        stop.set(); watcher.join(timeout=1)
        success = error_class is None and postflight_error is None and source_post
        new(output/"RESULT.json",dict(status=status if success else "FAILED_PRESERVED",mode=mode,run_id=run_id,stage=plan["stage"],
            scientific_result=science,error_class=error_class,postflight_error_class=postflight_error,private_failures=failures,
            actual_pid=process.pid,actual_create_time=process.create_time(),elapsed_process_wall_seconds=time.perf_counter()-started,
            actual_CPU_process_seconds=after["CPU_process_seconds"]-before["CPU_process_seconds"],resource_after=after,
            frozen_foundation_unchanged=frozen,original_source_registry_selection_T_pins_preserved=source_post,
            cuda_initialized=torch.cuda.is_initialized(),new_T_scoring=False,post_T_development=True,
            original_GPU_minimum_gap_completed=False,independent_generalization_claimed=False,
            scheduler_must_enforce_prior_chain_completion_and_cost_quiet=True,parent_must_report_forced_exit_without_RESULT=True,
            plan_sha256=plan_sha,runner_sha256=sha(__file__),finished_unix=time.time()))
    return 0 if success and status.startswith("COMPLETE_") else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("public","cache","asset"):
        parser.add_argument("--"+key+"-root")
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--mode",choices=("probe","fit","prefix","nested"))
    parser.add_argument("--run-id")
    parser.add_argument("--plan-sha")
    parser.add_argument("--deadline",type=float)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    roots = (args.public_root,args.cache_root,args.asset_root)
    if args.execute:
        if not all(roots+(args.mode,args.run_id,args.plan_sha,args.deadline,args.output)):
            parser.error("Execution requires original roots,mode,ID,PLAN SHA,deadline,new output")
        public,cache,asset = (Path(x).resolve() for x in roots)
        return execute(public,cache,asset,args.plan_sha,args.mode,args.run_id,args.deadline,Path(args.output).resolve())
    if any(roots):
        if not all(roots):
            parser.error("All three roots required for PLAN metadata validation")
        metadata(*(Path(x).resolve() for x in roots),args.plan_sha)
    print(json.dumps(dict(status="METADATA_ONLY_PREPARATION",fit_conditions=5,V_outputs=12,numpy_imported="numpy" in sys.modules,
        torch_imported="torch" in sys.modules,actual_model_array_training_inference_or_child_operations=0,
        runner_sha256=sha(__file__),new_T_scoring=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
