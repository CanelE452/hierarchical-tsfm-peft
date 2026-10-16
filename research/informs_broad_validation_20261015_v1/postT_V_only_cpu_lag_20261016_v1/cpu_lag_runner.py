"""Explicit post-T CPU development; metadata only unless --execute is supplied.

Original scientific modules are imported and called, never copied or patched.
The original selection, T results, registry and earlier partial fits stay frozen.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import traceback

NAME = "postT_V_only_cpu_lag_20261016_v1"
RUN_IDS = {family: "postT_cpu_jena_E3_" + family + "_k6_s92601_v1"
           for family in ("B0", "B2", "LAG_CAUSAL", "LAG_BIDIR")}
UPSTREAM_IDS = {family: "primary_jena_" + ("E1_" if family in {"B0", "B2"} else "E3_") + family + "_k6_s92601"
                for family in RUN_IDS}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def relative(root, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Expected a root-relative path")
    result = (Path(root).resolve() / path).resolve()
    if not result.is_relative_to(Path(root).resolve()):
        raise ValueError("Path leaves its declared root")
    return result


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def metadata(public, cache, asset, plan_sha=None):
    public, cache, asset = map(Path, (public, cache, asset))
    path = public / NAME / "PLAN.json"
    if plan_sha is not None and sha(path) != plan_sha:
        raise ValueError("Plan SHA changed")
    plan = read(path)
    if (plan["runner_sha256"] != sha(__file__) or plan["stage"] != "POST_T_V_ONLY_DEVELOPMENT"
            or plan["output_public_relative_root"] != NAME or plan["output_cache_relative_root"] != NAME
            or len(plan["source39"]) != 39 or plan["CPU_threads"] != 2 or plan["max_RSS_bytes"] != 4294967296
            or plan["maximum_concurrent_workers"] != 4 or plan["total_owned_RSS_limit_bytes"] != 8589934592
            or plan["storage_limit_bytes"] != 32212254720 or plan["deadline_unix"] != 1792222550):
        raise ValueError("Frozen runner, stage, output or resource contract differs")
    required = {"EXPERIMENT_REGISTRY.csv", "SELECTION_LOCK.json", "DATA_CONTRACT.json"}
    if not required.issubset(plan["immutable_public"]):
        raise ValueError("Original registry, T selection and data contract must be pinned")
    for pins in (plan["source39"], plan["immutable_public"]):
        for name, expected in pins.items():
            if sha(relative(public, name)) != expected:
                raise ValueError("Original public input or scientific source changed")
    if read(public / "SELECTION_LOCK.json").get("locked") is not True:
        raise ValueError("This separate development requires the original T selection to remain locked")
    if sha(relative(asset, plan["instance_norm_source_relative_path"])) != plan["instance_norm_source_sha256"]:
        raise ValueError("Original instance normalization source changed")
    actual_norm = public.parent / "temporal_frontend_campaign_20261015_v1/runtime.py"
    if sha(actual_norm) != plan["instance_norm_source_sha256"]:
        raise ValueError("Scientific module's actual worktree instance normalization source changed")
    model_path = relative(asset, plan["model_asset_relative_path"])
    if not model_path.is_dir() or model_path.name != plan["model_revision"]:
        raise ValueError("Pinned local foundation snapshot absent or revision differs")
    task = plan["task_contract"]
    constants = dict(dataset="jena", L=512, H=48, C=21, K=6, seed=92601, fold="primary", expected_V_origins=43)
    if any(task.get(key) != value for key, value in constants.items()) or len(task["columns"]) != 21:
        raise ValueError("Jena task axes or comparison contract changed")
    if len(plan["conditions"]) != 4 or {row["run_id"] for row in plan["conditions"]} != set(RUN_IDS.values()):
        raise ValueError("Exactly the four declared conditions must be retained")
    upstream = {row["experiment_id"]: row for row in csv_rows(public / "EXPERIMENT_REGISTRY.csv")}
    post = public / NAME
    registry = relative(post, plan["postT_registry_relative_path"])
    if sha(registry) != plan["postT_registry_sha256"]:
        raise ValueError("Separate post-T preregistration changed")
    registered = csv_rows(registry)
    if len(registered) != 4 or {row["experiment_id"] for row in registered} != set(RUN_IDS.values()):
        raise ValueError("Separate preregistration must contain exactly four new IDs")
    for condition in plan["conditions"]:
        family = condition["family"]
        if family not in RUN_IDS or condition["run_id"] != RUN_IDS[family] or condition["upstream_registered_id"] != UPSTREAM_IDS[family]:
            raise ValueError("Family or original preregistered identity changed")
        old = upstream.get(condition["upstream_registered_id"], {})
        expected_old = dict(dataset="jena", family=family, fold="primary", latent_k="6", seed="92601", length="512", horizon="48")
        if any(old.get(key) != value for key, value in expected_old.items()) or condition["run_id"] in upstream:
            raise ValueError("Upstream scientific identity differs or new ID already exists in old registry")
        spec = condition["spec"]
        expected = dict(run_id=RUN_IDS[family], dataset="jena", family=family, latent_k=6, seed=92601, fold="primary",
                        epochs=[.25, 1., 3.], lr=.001, device="cpu", microbatch=4, effective_batch=16,
                        F_row_microbatch=128, checkpoint_inputs=False, lora_rank=8, lora_alpha=16)
        if (any(spec.get(key) != value for key, value in expected.items())
                or set(spec) - set(expected) != {"stage_label", "provenance"}
                or not isinstance(spec["provenance"], dict) or not str(spec["stage_label"]).lower().startswith("post")):
            raise ValueError("Campaign template recipe differs beyond declared fresh CPU provenance")
        recipe = condition["recipe_reference"]
        if recipe != dict(relative_path="campaign.py", sha256=plan["source39"]["campaign.py"]):
            raise ValueError("Original campaign template reference changed")
    return plan


def select_condition(plan, run_id, mode):
    matches = [condition for condition in plan["conditions"] if condition["run_id"] == run_id]
    if len(matches) != 1 or mode not in {"probe", "fit"}:
        raise ValueError("One explicit preregistered condition and probe/fit mode are required")
    if mode == "probe" and matches[0]["family"] != "B0":
        raise ValueError("The first-update feasibility probe is the B0 control only")
    return matches[0]


def require_fresh_paths(plan, public, cache, run_id):
    for root in (Path(public) / plan["output_public_relative_root"], Path(cache) / plan["output_cache_relative_root"], Path(public), Path(cache)):
        for kind in ("runs", "predictions"):
            if (root / kind / run_id).exists():
                raise FileExistsError("Existing run or prediction must be preserved; fresh execution refused")


def fit_and_predict(fit_api, predict_api, spec, task, cache, public, foundation, deadline, *, split="V_select"):
    if split != "V_select":
        raise ValueError("Only V_select may be computed in this post-T development")
    receipt = fit_api(spec, task, cache, public, foundation, deadline)
    if receipt.get("status") != "COMPLETE_VALID":
        return dict(fit_receipt=receipt, prediction=None)
    if receipt.get("reused_complete_artifact"):
        raise ValueError("Fresh CPU execution unexpectedly reused a previous artifact")
    prediction = predict_api(spec["run_id"], task, "V_select", cache, public, foundation,
                             receipt=receipt, fold="primary", deadline=deadline)
    if prediction["score"]["n_origins"] != 43 or prediction["metadata"].get("reused_prediction_cache"):
        raise ValueError("Full fresh V cohort of 43 origins is required")
    if prediction["cost_record"].get("actual_F_streams") != 258 or prediction["cost_record"].get("expected_F_streams") != 258:
        raise ValueError("Actual full V foundation stream count differs from 43 times K6")
    return dict(fit_receipt=receipt, prediction=prediction)


def probe_update(segment_api, model, windows, legal, sampler, optimizer, predict, deadline):
    return segment_api(model, windows, legal, sampler, optimizer, 16, predict,
                       effective_batch=16, microbatch=4, starting_updates=0, budget_deadline=deadline)


def validate_task(task, plan, data, fit):
    contract = plan["task_contract"]
    actual = dict(dataset=task["dataset"], L=task["L"], H=task["H"], C=task["physical"].shape[1],
                  data_sha256=task["data_sha256"], columns=list(task["columns"]), array_sha256=task["array_sha256"],
                  primary_split_ranges={key: list(value) for key, value in data.split_ranges(task, "primary").items()})
    if any(actual[key] != contract[key] for key in actual):
        raise ValueError("Actually loaded task, axes, data, columns or split fingerprints differ")
    bounds = data.split_ranges(task, "primary")
    mean, std, count = data.fit_statistics(task, bounds["A_fit"][1])
    fingerprints = {key: data.array_sha256(value) for key, value in (("mean", mean), ("std", std), ("count", count))}
    if fingerprints != contract["A_fit_statistics_fingerprints"] or fit.scaler_fingerprint(mean, std) != contract["search_scaler_sha256"]:
        raise ValueError("A50 training statistics differ")
    refit_mean, refit_std, _ = data.fit_statistics(task, bounds["A_stop"][1])
    if fit.scaler_fingerprint(refit_mean, refit_std) != contract["refit_scaler_sha256"]:
        raise ValueError("A60 refit/canonical statistics differ")
    if len(data.origins(bounds["V_select"], 512, 48, 48)) != 43:
        raise ValueError("Original full V cohort differs")
    return actual


def rss_record(process):
    memory = process.memory_info()
    cpu = process.cpu_times()
    return dict(rss_bytes=memory.rss, OS_peak_working_set_bytes=getattr(memory, "peak_wset", None),
                CPU_process_seconds=cpu.user+cpu.system)


def save_private_failure(post_cache, original_cache, mode, run_id, phase, error):
    directory = post_cache / "failures"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (mode+"_"+run_id+"_"+str(time.time_ns())+".json")
    new(path, dict(run_id=run_id, mode=mode, phase=phase, error_class=type(error).__name__,
                   error=str(error), traceback=traceback.format_exc(), recorded_unix=time.time()))
    return dict(cache_relative_path=path.relative_to(original_cache).as_posix(), error_class=type(error).__name__, phase=phase)


def execute(public, cache, asset, plan_sha, mode, run_id, deadline, output):
    if not plan_sha:
        raise ValueError("Execution requires an explicit immutable plan SHA")
    plan = metadata(public, cache, asset, plan_sha)
    condition = select_condition(plan, run_id, mode)
    if deadline is None or not time.time() < deadline <= plan["deadline_unix"]:
        raise ValueError("Explicit live child deadline within the fixed post-T campaign limit required")
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or any(key.startswith("INFORMS_GPU_") for key in os.environ)
            or any(os.environ.get(key) != "2" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
            or any(os.environ.get(key) != "1" for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))):
        raise ValueError("Offline CPU-only environment, two threads and no inherited GPU lease required")
    if (Path(os.environ.get("INFORMS_ASSET_ROOT", "")).resolve() != asset
            or Path(os.environ.get("INFORMS_CACHE_ROOT", "")).resolve() != cache / NAME):
        raise ValueError("Runtime asset root and separate post-T cache environment must match the explicit roots")
    post_public, post_cache = public / NAME, cache / NAME
    if not output.is_relative_to(post_public) or output == post_public:
        raise ValueError("New diagnostic output must stay in the separate public post-T root")
    require_fresh_paths(plan, public, cache, run_id)
    if output.exists():
        raise FileExistsError("Diagnostic output already exists")
    used = sum(path.stat().st_size for root in (public, cache) for path in root.rglob("*") if path.is_file())
    if used+256*1024**2 > plan["storage_limit_bytes"]:
        raise ValueError("Combined original and new campaign storage limit reached")
    output.mkdir(parents=True, exist_ok=False)
    claims = post_public / "claims"
    claims.mkdir(exist_ok=True)
    new(claims / (mode+"_"+run_id+".json"), dict(run_id=run_id, mode=mode, plan_sha256=plan_sha, started_unix=time.time()))
    sys.path.insert(0, str(public))
    import psutil
    import torch
    import data_contract as data
    import fit_base as fit
    import training
    import prediction_cache
    from gpu_runtime import FrozenBolt
    torch.set_num_threads(2)
    process = psutil.Process(os.getpid())
    stopped = threading.Event()
    def guard():
        while not stopped.wait(.25):
            memory = process.memory_info().rss
            if memory > plan["max_RSS_bytes"] or time.time() >= deadline:
                new(output / "RESOURCE_STOP.json", dict(status="RESOURCE_STOP", rss_bytes=memory,
                    deadline_unix=deadline, at_unix=time.time(), actual_pid=process.pid))
                os._exit(98)
    watchdog = threading.Thread(target=guard, daemon=True)
    watchdog.start()
    before = rss_record(process)
    new(output / "STARTED.json", dict(actual_pid=process.pid, actual_create_time=process.create_time(), run_id=run_id,
        mode=mode, plan_sha256=plan_sha, runner_sha256=sha(__file__), stage=plan["stage"], deadline_unix=deadline,
        resource_before=before, new_T_scoring=False, no_model_state_reuse=True,
        archive_deserialization_scope="The approved whole safeFIT loader can deserialize T bytes; only A50/A_stop/A60/V computations are requested."))
    foundation = None
    status, error_class, postflight_error_class, science = "FAILED_PRESERVED", None, None, None
    source_post, frozen_unchanged = False, False
    private_failures = []
    started = time.perf_counter()
    try:
        task = data.load_task("jena", cache_dir=cache)
        actual_task = validate_task(task, plan, data, fit)
        new(output / "TASK_BINDING.json", actual_task)
        if torch.cuda.is_initialized() or process.memory_info().rss > plan["max_RSS_bytes"]:
            raise RuntimeError("CPU visibility or worker RSS contract violated before foundation load")
        foundation = FrozenBolt(asset / plan["model_asset_relative_path"], 48, 128, False, device="cpu")
        if foundation.initial_hash != plan["foundation_tensor_sha256"] or foundation.instance_norm_source_sha256 != plan["instance_norm_source_sha256"]:
            raise ValueError("Actual frozen foundation or normalization source differs")
        foundation.assert_frozen_unchanged()
        spec = dict(condition["spec"])
        counters_before = fit._f_counters(foundation)
        if mode == "probe":
            fit_end = data.split_ranges(task, "primary")["A_fit"][1]
            model, norm = fit.create_model(spec, task, fit_end, foundation)
            prefix = dict(task, physical=task["physical"][:fit_end], observed=task["observed"][:fit_end])
            windows = training.FastWindows(prefix, norm["mean"], norm["std"], "cpu")
            legal = data.origins((0, fit_end), 512, 48, 1)
            sampler = training.PermutationStream(legal, 92601)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=0)
            predict = lambda x, k: fit.predict_model(model, x, foundation, "B0", k)
            probe_started = time.perf_counter()
            segment = probe_update(training.train_segment, model, windows, legal, sampler, optimizer, predict, deadline)
            delta = {key: value-counters_before[key] for key, value in fit._f_counters(foundation).items()}
            gradient_finite = all(isinstance(segment.get(key), (int, float)) and math.isfinite(segment[key])
                                  for key in ("gradient_norm_min", "gradient_norm_max"))
            if segment["status"] != "SEGMENT_COMPLETE" or segment["updates"] != 1 or segment["exposures"] != 16 or delta["streams"] != 96 or not gradient_finite:
                raise ValueError("Original first effective update or its gradient/stream contract was not completed")
            science = dict(status="PROBE_FIRST_EFFECTIVE_UPDATE_COMPLETE", segment=segment,
                effective_update_wall_seconds=time.perf_counter()-probe_started, F_counter_delta=delta,
                gradient_finite=gradient_finite, full_A50_legal_origins=len(legal), probe_weights_discarded=True,
                probe_state_reused_for_fit=False, completed_fit=False, V_scoring=False)
            del model, optimizer, windows, sampler
            status = science["status"]
        else:
            outcome = fit_and_predict(fit.fit_base, prediction_cache.predict_parent, spec, task, post_cache, post_public, foundation, deadline)
            prediction = outcome["prediction"]
            science = dict(fit_receipt=outcome["fit_receipt"], full_V_completed=prediction is not None,
                           F_counter_delta={key: value-counters_before[key] for key, value in fit._f_counters(foundation).items()})
            if prediction is not None:
                cost = prediction["cost_record"]
                manifest = post_public / "predictions" / run_id / "V_select/PREDICTION_MANIFEST.json"
                science.update(V_score=prediction["score"], V_cost_record=cost, V_manifest_relative_path=manifest.relative_to(public).as_posix(),
                               V_manifest_sha256=sha(manifest), convergence_established=False,
                               V_cost_F_calls_is_bundle_bookkeeping_not_actual_row_calls=True)
                status = "COMPLETE_FRESH_CPU_FIT_AND_FULL_V"
            else:
                status = outcome["fit_receipt"]["status"]
        foundation.assert_frozen_unchanged()
        if torch.cuda.is_initialized():
            raise RuntimeError("Unexpected CUDA initialization in CPU development")
    except Exception as error:
        error_class = type(error).__name__
        private_failures.append(save_private_failure(post_cache, cache, mode, run_id, "execution", error))
    finally:
        try:
            if foundation is not None:
                foundation.assert_frozen_unchanged()
                frozen_unchanged = True
            metadata(public, cache, asset, plan_sha)
            source_post = True
        except Exception as error:
            postflight_error_class = type(error).__name__
            private_failures.append(save_private_failure(post_cache, cache, mode, run_id, "postflight", error))
        after = rss_record(process)
        stopped.set()
        watchdog.join(timeout=1)
        success = error_class is None and postflight_error_class is None and source_post
        result = dict(status=status if success else "FAILED_PRESERVED", run_id=run_id, mode=mode, stage=plan["stage"],
            actual_pid=process.pid, actual_create_time=process.create_time(), elapsed_process_wall_seconds=time.perf_counter()-started,
            actual_CPU_process_seconds=after["CPU_process_seconds"]-before["CPU_process_seconds"], resource_after=after,
            error_class=error_class, postflight_error_class=postflight_error_class, private_failures=private_failures,
            scientific_result=science, frozen_foundation_unchanged=frozen_unchanged,
            original_source_registry_selection_T_pins_preserved=source_post, cuda_initialized=torch.cuda.is_initialized(),
            new_T_target_or_scoring=False, original_GPU_coverage_completed=False, post_T_development=True,
            observations_not_a_new_independent_holdout=True, parent_must_report_forced_exit_without_final_result=True,
            plan_sha256=plan_sha, runner_sha256=sha(__file__), finished_unix=time.time())
        new(output / "RESULT.json", result)
    return 0 if success and status in {"PROBE_FIRST_EFFECTIVE_UPDATE_COMPLETE", "COMPLETE_FRESH_CPU_FIT_AND_FULL_V"} else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for root in ("public", "cache", "asset"):
        parser.add_argument("--"+root+"-root", default=os.environ.get("INFORMS_"+root.upper()+"_ROOT"))
    parser.add_argument("--plan-sha")
    parser.add_argument("--mode", choices=("probe", "fit"))
    parser.add_argument("--run-id")
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--output")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not all((args.public_root, args.cache_root, args.asset_root)):
        parser.error("Explicit original public/cache/asset roots required")
    public, cache, asset = (Path(value).resolve() for value in (args.public_root, args.cache_root, args.asset_root))
    if args.execute:
        if not all((args.mode, args.run_id, args.output, args.plan_sha, args.deadline)):
            parser.error("Execution requires mode, one run ID, new output, plan SHA and deadline")
        raise SystemExit(execute(public, cache, asset, args.plan_sha, args.mode, args.run_id, args.deadline, Path(args.output).resolve()))
    plan = metadata(public, cache, asset, args.plan_sha)
    print(json.dumps(dict(status="METADATA_ONLY_POST_T_DEVELOPMENT", registered_conditions=len(plan["conditions"]),
        plan_sha256=sha(public / NAME / "PLAN.json"), runner_sha256=sha(__file__), numpy_imported="numpy" in sys.modules,
        torch_imported="torch" in sys.modules, science_modules_imported=any(name in sys.modules for name in ("fit_base", "training", "data_contract", "gpu_runtime")),
        actual_training_or_model_or_array_access=0, new_T_scoring=False)), flush=True)


if __name__ == "__main__":
    main()
