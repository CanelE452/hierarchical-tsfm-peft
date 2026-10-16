"""Fresh fixed-PCA CPU development; original fit/prediction APIs, V scoring only.

The original [.25,1,3] schedule performs fixed-transform evaluations, not learning:
there are no trainable parameters, optimizer updates or training exposures.
Default invocation imports only the standard library and opens no data/model.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time

NAME = "postT_V_only_cpu_fixed_pca_20261016_v1"
UTILITY_RELATIVE_PATH = "postT_V_only_cpu_lag_20261016_v1/cpu_lag_runner.py"
AXES = {"beijing": (21038, 11, 48), "jena": (20965, 21, 48),
        "electricity": (24240, 321, 48), "pems08": (17856, 170, 12)}
CONDITIONS = {f"postT_cpu_{ds}_E1_PCA_FIXED_BOLT_k{k}_s92601_v1": (ds, k)
              for ds, ks in (("beijing", (9,)), ("jena", (16,)), ("electricity", (41, 241)), ("pems08", (22, 128)))
              for k in ks}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1048576), b""):
            value.update(part)
    return value.hexdigest()


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
    root = Path(root).resolve()
    result = root / path
    if any(part.is_symlink() for part in [result] + list(result.parents)):
        raise ValueError("Symlink input paths are not accepted")
    result = result.resolve()
    if not result.is_relative_to(root):
        raise ValueError("Path leaves its declared root")
    return result


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def upstream_id(dataset, k):
    return f"primary_{dataset}_E1_PCA_FIXED_BOLT_k{k}_s92601"


def expected_spec(run_id, dataset, k):
    return dict(run_id=run_id, dataset=dataset, family="PCA_FIXED_BOLT", latent_k=k, seed=92601, fold="primary",
        epochs=[.25, 1., 3.], lr=.001, device="cpu", microbatch=4, effective_batch=16,
        F_row_microbatch=128, checkpoint_inputs=False, lora_rank=8, lora_alpha=16,
        stage_label="postT_cpu_fresh_V_only_fixed_PCA", provenance=dict(upstream_registered_id=upstream_id(dataset, k)))


def metadata(public, cache, asset, plan_sha=None):
    public, cache, asset = map(Path, (public, cache, asset))
    plan_path = public / NAME / "PLAN.json"
    if plan_sha is not None and sha(plan_path) != plan_sha:
        raise ValueError("Explicit PLAN SHA changed")
    plan = read(plan_path)
    fixed = dict(stage="POST_T_V_ONLY_DEVELOPMENT", output_public_relative_root=NAME, output_cache_relative_root=NAME,
        CPU_threads=2, max_RSS_bytes=3221225472, maximum_concurrent_workers=1,
        storage_limit_bytes=32212254720, deadline_unix=1792222550)
    if any(plan.get(key) != value for key, value in fixed.items()) or plan["runner_sha256"] != sha(__file__):
        raise ValueError("Runner, post-T namespace or resource contract differs")
    required = {"campaign.py", "data_contract.py", "fit_base.py", "prediction_cache.py", "networks.py", "training.py", "gpu_runtime.py"}
    if len(plan["source39"]) != 39 or not required.issubset(plan["source39"]):
        raise ValueError("Original source39 must include the unchanged scientific APIs")
    if not {"EXPERIMENT_REGISTRY.csv", "SELECTION_LOCK.json", "DATA_CONTRACT.json"}.issubset(plan["immutable_public"]):
        raise ValueError("Original registry, selection lock and data contract must be pinned")
    for pins in (plan["source39"], plan["immutable_public"]):
        for name, expected in pins.items():
            if sha(relative(public, name)) != expected:
                raise ValueError("Original public/source identity changed: " + name)
    if read(public / "SELECTION_LOCK.json").get("locked") is not True:
        raise ValueError("Original observed-T selection must remain locked")
    if (plan.get("shared_utility_relative_path") != UTILITY_RELATIVE_PATH
            or sha(relative(public, UTILITY_RELATIVE_PATH)) != plan["shared_utility_sha256"]):
        raise ValueError("Shared stdlib utility dependency differs")
    if sha(relative(asset, plan["instance_norm_source_relative_path"])) != plan["instance_norm_source_sha256"]:
        raise ValueError("Asset instance normalization source changed")
    norm = public.parent / "temporal_frontend_campaign_20261015_v1/runtime.py"
    if sha(norm) != plan["instance_norm_source_sha256"]:
        raise ValueError("Original module's actual worktree normalization source changed")
    model_path = relative(asset, plan["model_asset_relative_path"])
    if not model_path.is_dir() or model_path.name != plan["model_revision"]:
        raise ValueError("Pinned foundation snapshot directory/revision absent")
    original = read(public / "DATA_CONTRACT.json")["assets"]
    if set(plan["task_contracts"]) != set(AXES):
        raise ValueError("Exactly the four original task contracts are required")
    for dataset, (rows, channels, horizon) in AXES.items():
        task = plan["task_contracts"][dataset]
        preserved = {key: value for key, value in task.items() if key not in {"search_scaler_sha256", "refit_scaler_sha256"}}
        if (preserved != original[dataset] or task["status"] != "READY_PINNED" or task["shape"] != [rows, channels]
                or task["L"] != 512 or task["H"] != horizon or len(task["columns"]) != channels
                or any(len(task.get(key, "")) != 64 for key in ("search_scaler_sha256", "refit_scaler_sha256"))):
            raise ValueError("Task identity, original metadata, horizon or scaler pin differs")
    upstream = csv_rows(public / "EXPERIMENT_REGISTRY.csv")
    by_id = {row["experiment_id"]: row for row in upstream}
    if len(by_id) != len(upstream):
        raise ValueError("Duplicate original registry ID")
    registry = relative(public / NAME, plan["postT_registry_relative_path"])
    if sha(registry) != plan["postT_registry_sha256"]:
        raise ValueError("Separate immutable preregistration changed")
    registered = csv_rows(registry)
    new_by_id = {row["experiment_id"]: row for row in registered}
    if len(registered) != 6 or set(new_by_id) != set(CONDITIONS):
        raise ValueError("Exactly six distinct new fixed-PCA IDs are required")
    conditions = plan["conditions"]
    if len(conditions) != 6 or {row["run_id"] for row in conditions} != set(CONDITIONS):
        raise ValueError("Exactly the six declared conditions must be retained")
    for condition in conditions:
        run_id = condition["run_id"]
        dataset, k = CONDITIONS[run_id]
        old_id = upstream_id(dataset, k)
        if condition["family"] != "PCA_FIXED_BOLT" or condition["upstream_registered_id"] != old_id or run_id in by_id:
            raise ValueError("Family/upstream/new identity differs")
        science = dict(dataset=dataset, family="PCA_FIXED_BOLT", fold="primary", latent_k=str(k), seed="92601", length="512", horizon=str(AXES[dataset][2]))
        old, post = by_id.get(old_id, {}), new_by_id[run_id]
        if (any(old.get(key) != value or post.get(key) != value for key, value in science.items())
                or old.get("execution_status") != "PENDING_QUEUE" or post.get("upstream_registered_id") != old_id
                or post.get("device") != "cpu" or post.get("epochs") != "0.25|1|3"
                or post.get("stage") != "POST_T_V_ONLY_DEVELOPMENT"):
            raise ValueError("Original missing condition or separate preregistration differs")
        if condition["spec"] != expected_spec(run_id, dataset, k):
            raise ValueError("Original campaign recipe differs beyond the explicit new CPU provenance")
        if condition["recipe_reference"] != dict(relative_path="campaign.py", sha256=plan["source39"]["campaign.py"]):
            raise ValueError("Original recipe source pin differs")
    return plan


def require_fresh_paths(public, cache, run_id):
    if run_id not in CONDITIONS:
        raise ValueError("Only the six preregistered new IDs may be executed")
    for root in (Path(public), Path(cache), Path(public) / NAME, Path(cache) / NAME):
        for kind in ("runs", "predictions"):
            if (root / kind / run_id).exists():
                raise FileExistsError("Existing run/prediction preserved; fresh execution only")
    if (Path(public) / NAME / "claims" / ("fit_" + run_id + ".json")).exists():
        raise FileExistsError("Existing claimed attempt cannot be resumed")


def require_environment(public, cache, asset, env=None):
    env = os.environ if env is None else env
    if (env.get("CUDA_VISIBLE_DEVICES") != "-1" or any(key.startswith("INFORMS_GPU_") for key in env)
            or any(env.get(key) != "2" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
            or any(env.get(key) != "1" for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))
            or Path(env.get("INFORMS_ASSET_ROOT", "")).resolve() != Path(asset).resolve()
            or Path(env.get("INFORMS_CACHE_ROOT", "")).resolve() != (Path(cache) / NAME).resolve()):
        raise ValueError("Explicit separate-cache offline CPU2 environment without GPU authorization required")


def fit_and_predict(fit_api, predict_api, spec, task, cache, public, foundation, deadline, expected_origins, *, split="V_select"):
    if split != "V_select":
        raise ValueError("Only full V_select may be evaluated")
    receipt = fit_api(spec, task, cache, public, foundation, deadline)
    if receipt.get("status") != "COMPLETE_VALID":
        return dict(fit_receipt=receipt, prediction=None)
    if receipt.get("reused_complete_artifact") or any(receipt.get(key) != 0 for key in ("search_updates", "refit_updates", "search_exposures", "refit_exposures")):
        raise ValueError("Fixed PCA must be fresh with exactly zero optimizer updates/training exposures")
    prediction = predict_api(spec["run_id"], task, "V_select", cache, public, foundation, receipt=receipt, fold="primary", deadline=deadline)
    cost, expected = prediction["cost_record"], expected_origins * spec["latent_k"]
    if (prediction["score"]["n_origins"] != expected_origins or prediction["metadata"].get("reused_prediction_cache")
            or cost.get("actual_F_streams") != expected or cost.get("expected_F_streams") != expected):
        raise ValueError("Fresh full original V cohort and exact K-row F streams required")
    return dict(fit_receipt=receipt, prediction=prediction)


def validate_task(task, plan, dataset, data, fit):
    contract = plan["task_contracts"][dataset]
    actual = dict(shape=list(task["physical"].shape), L=task["L"], H=task["H"], columns=list(task["columns"]),
        data_sha256=task["data_sha256"], array_sha256=task["array_sha256"],
        primary_split_ranges={key: list(value) for key, value in data.split_ranges(task, "primary").items()},
        secondary_split_ranges={key: list(value) for key, value in data.split_ranges(task, "secondary").items()}, metadata=task["metadata"])
    if task["dataset"] != dataset or any(contract[key] != value for key, value in actual.items()):
        raise ValueError("Loaded original task, axes, time bounds or fingerprints differ")
    bounds = data.split_ranges(task, "primary")
    mean, std, count = data.fit_statistics(task, bounds["A_fit"][1])
    fingerprints = {key: data.array_sha256(value) for key, value in (("mean", mean), ("std", std), ("count", count))}
    if fingerprints != contract["A_fit_statistics_fingerprints"] or fit.scaler_fingerprint(mean, std) != contract["search_scaler_sha256"]:
        raise ValueError("Original A50 statistics/scaler differ")
    mean, std, _ = data.fit_statistics(task, bounds["A_stop"][1])
    if fit.scaler_fingerprint(mean, std) != contract["refit_scaler_sha256"]:
        raise ValueError("Original A60 canonical/refit scaler differs")
    count = len(data.origins(bounds["V_select"], 512, task["H"], task["H"]))
    if count != contract["legal_origin_counts_primary"]["V_select"]:
        raise ValueError("Full V origin count differs from original contract")
    return dict(actual, dataset=dataset, expected_V_origins=count,
                search_scaler_sha256=contract["search_scaler_sha256"], refit_scaler_sha256=contract["refit_scaler_sha256"])


def execute(public, cache, asset, plan_sha, run_id, deadline, output):
    if not plan_sha:
        raise ValueError("Execution requires explicit immutable PLAN SHA")
    plan = metadata(public, cache, asset, plan_sha)
    if run_id not in CONDITIONS or deadline is None or not time.time() < deadline <= plan["deadline_unix"]:
        raise ValueError("One preregistered ID and live deadline within original wall required")
    require_environment(public, cache, asset)
    require_fresh_paths(public, cache, run_id)
    post_public, post_cache = public / NAME, cache / NAME
    if output == post_public or not output.is_relative_to(post_public) or output.exists():
        raise ValueError("A new diagnostic path inside the separate public namespace required")
    used = sum(path.stat().st_size for root in (public, cache) for path in root.rglob("*") if path.is_file())
    if used + 256 * 1024**2 > plan["storage_limit_bytes"]:
        raise ValueError("Combined original/new storage cap reached")
    output.mkdir(parents=True, exist_ok=False)
    post_cache.mkdir(parents=True, exist_ok=True)
    (post_public / "claims").mkdir(exist_ok=True)
    new(post_public / "claims" / ("fit_" + run_id + ".json"), dict(run_id=run_id, plan_sha256=plan_sha, started_unix=time.time()))
    utility_spec = importlib.util.spec_from_file_location("fixed_pca_stdlib_utilities", relative(public, UTILITY_RELATIVE_PATH))
    utilities = importlib.util.module_from_spec(utility_spec)
    utility_spec.loader.exec_module(utilities)
    sys.path.insert(0, str(public))
    import psutil
    import torch
    import data_contract as data
    import fit_base as fit
    import prediction_cache
    from gpu_runtime import FrozenBolt
    torch.set_num_threads(2)
    process, stopped = psutil.Process(os.getpid()), threading.Event()
    def guard():
        while not stopped.wait(.25):
            rss = process.memory_info().rss
            if rss > plan["max_RSS_bytes"] or time.time() >= deadline:
                new(output / "RESOURCE_STOP.json", dict(status="RESOURCE_STOP", rss_bytes=rss, deadline_unix=deadline, at_unix=time.time(), actual_pid=process.pid))
                os._exit(98)
    watchdog = threading.Thread(target=guard, daemon=True)
    watchdog.start()
    before, started = utilities.rss_record(process), time.perf_counter()
    new(output / "STARTED.json", dict(run_id=run_id, mode="fit", stage=plan["stage"], plan_sha256=plan_sha,
        runner_sha256=sha(__file__), actual_pid=process.pid, actual_create_time=process.create_time(), deadline_unix=deadline,
        fixed_transform=True, expected_optimizer_updates=0, resource_before=before, new_T_scoring=False,
        archive_deserialization_scope="Approved whole safeFIT task is loaded/hashed, including campaign-T bytes; only original A50/A_stop/A60/V computations requested."))
    dataset, k = CONDITIONS[run_id]
    condition = next(row for row in plan["conditions"] if row["run_id"] == run_id)
    foundation, science, error_class, postflight_error = None, None, None, None
    frozen, source_post, failures = False, False, []
    status = "FAILED_PRESERVED"
    phase = "task_loading"
    try:
        task = data.load_task(dataset, cache_dir=cache)
        binding = validate_task(task, plan, dataset, data, fit)
        new(output / "TASK_BINDING.json", binding)
        if torch.cuda.is_initialized() or process.memory_info().rss > plan["max_RSS_bytes"]:
            raise RuntimeError("CPU-only/RSS contract violated before foundation loading")
        phase = "foundation_loading"
        foundation = FrozenBolt(asset / plan["model_asset_relative_path"], task["H"], 128, False, device="cpu")
        if foundation.initial_hash != plan["foundation_tensor_sha256"] or foundation.instance_norm_source_sha256 != plan["instance_norm_source_sha256"]:
            raise ValueError("Actual original frozen model/normalization identity differs")
        foundation.assert_frozen_unchanged()
        phase = "fixed_transform_validation"
        spec = dict(condition["spec"])
        model, norm = fit.create_model(spec, task, data.split_ranges(task, "primary")["A_fit"][1], foundation)
        buffers = dict(model.named_buffers())
        if list(model.parameters()) or set(buffers) != {"center", "E"} or tuple(buffers["center"].shape) != (AXES[dataset][1],) or tuple(buffers["E"].shape) != (AXES[dataset][1], k):
            raise ValueError("Original fixed PCA must contain only center/E buffers and zero parameters")
        if any(not torch.isfinite(value).all() for value in buffers.values()):
            raise ValueError("Nonfinite original fixed-PCA buffer")
        initialization = dict(trainable_parameters=0, total_parameters=0,
            buffers={name: dict(shape=list(value.shape), sha256=data.array_sha256(value.detach().cpu().numpy())) for name, value in buffers.items()},
            scaler_sha256=fit.scaler_fingerprint(norm["mean"], norm["std"]))
        if initialization["scaler_sha256"] != plan["task_contracts"][dataset]["search_scaler_sha256"]:
            raise ValueError("create_model A50 normalization differs")
        del model, buffers
        new(output / "FIXED_PCA_INITIALIZATION.json", initialization)
        counters = fit._f_counters(foundation)
        phase = "original_fixed_transform_fit"
        def infer_V(*args, **kwargs):
            nonlocal phase
            phase = "original_full_V_prediction"
            return prediction_cache.predict_parent(*args, **kwargs)
        outcome = fit_and_predict(fit.fit_base, infer_V, spec, task, post_cache, post_public,
                                  foundation, deadline, binding["expected_V_origins"])
        prediction = outcome["prediction"]
        science = dict(fit_receipt=outcome["fit_receipt"], full_V_completed=prediction is not None,
            fixed_transform=True, learned_epochs=0, optimizer_updates=0, training_exposures=0,
            original_evaluation_schedule_preserved=[.25, 1., 3.], schedule_is_not_learning_epochs=True,
            F_counter_delta={key: value - counters[key] for key, value in fit._f_counters(foundation).items()})
        if prediction is not None:
            manifest = post_public / "predictions" / run_id / "V_select/PREDICTION_MANIFEST.json"
            science.update(V_score=prediction["score"], V_cost_record=prediction["cost_record"],
                V_manifest_relative_path=manifest.relative_to(public).as_posix(), V_manifest_sha256=sha(manifest),
                V_cost_F_calls_is_bundle_bookkeeping_not_actual_row_calls=True, convergence_claimed=False)
            status = "COMPLETE_FRESH_CPU_FIXED_PCA_AND_FULL_V"
        else:
            status = outcome["fit_receipt"]["status"]
        foundation.assert_frozen_unchanged()
        if torch.cuda.is_initialized():
            raise RuntimeError("Unexpected CUDA initialization")
    except Exception as error:
        error_class = type(error).__name__
        failures.append(utilities.save_private_failure(post_cache, cache, "fit", run_id, phase, error))
    finally:
        try:
            if foundation is not None:
                foundation.assert_frozen_unchanged()
                frozen = True
            metadata(public, cache, asset, plan_sha)
            source_post = True
        except Exception as error:
            postflight_error = type(error).__name__
            failures.append(utilities.save_private_failure(post_cache, cache, "fit", run_id, "postflight", error))
        after = utilities.rss_record(process)
        stopped.set()
        watchdog.join(timeout=1)
        success = error_class is None and postflight_error is None and source_post
        new(output / "RESULT.json", dict(status=status if success else "FAILED_PRESERVED", run_id=run_id, mode="fit",
            stage=plan["stage"], dataset=dataset, latent_k=k, fixed_transform=True,
            actual_pid=process.pid, actual_create_time=process.create_time(), elapsed_process_wall_seconds=time.perf_counter() - started,
            actual_CPU_process_seconds=after["CPU_process_seconds"] - before["CPU_process_seconds"], resource_after=after,
            error_class=error_class, postflight_error_class=postflight_error, private_failures=failures,
            scientific_result=science, frozen_foundation_unchanged=frozen, original_source_registry_selection_T_pins_preserved=source_post,
            cuda_initialized=torch.cuda.is_initialized(), new_T_scoring=False, T_targets_used_for_model_selection=False,
            whole_permitted_task_deserialization_disclosed=True, original_GPU_coverage_completed=False,
            post_T_development=True, observations_not_a_new_independent_holdout=True,
            parent_must_report_forced_exit_without_final_result=True, plan_sha256=plan_sha,
            runner_sha256=sha(__file__), finished_unix=time.time()))
    return 0 if success and status == "COMPLETE_FRESH_CPU_FIXED_PCA_AND_FULL_V" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for root in ("public", "cache", "asset"):
        parser.add_argument("--" + root + "-root")
    parser.add_argument("--mode", choices=("fit",), default="fit")
    parser.add_argument("--run-id")
    parser.add_argument("--plan-sha")
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--output")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    roots = (args.public_root, args.cache_root, args.asset_root)
    if args.execute:
        if not all(roots + (args.run_id, args.plan_sha, args.deadline, args.output)):
            parser.error("Execution requires original roots, one run ID, PLAN SHA, live deadline and new output")
        public, cache, asset = (Path(value).resolve() for value in roots)
        return execute(public, cache, asset, args.plan_sha, args.run_id, args.deadline, Path(args.output).resolve())
    if any(roots):
        if not all(roots):
            parser.error("Supply all three original roots for PLAN metadata verification")
        metadata(*(Path(value).resolve() for value in roots), args.plan_sha)
    print(json.dumps(dict(status="METADATA_ONLY_FIXED_PCA", registered_conditions=6, runner_sha256=sha(__file__),
        numpy_imported="numpy" in sys.modules, torch_imported="torch" in sys.modules,
        actual_model_array_or_fit_access=0, new_T_scoring=False, fixed_transform=True, optimizer_updates=0)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
