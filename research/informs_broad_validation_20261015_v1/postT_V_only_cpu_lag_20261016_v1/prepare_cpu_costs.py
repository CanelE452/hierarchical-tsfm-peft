"""Prepare eight original CPU cost workers; never start a subprocess or a timer."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
FIT_SOURCES = ("fit_base.py", "training.py", "networks.py", "data_contract.py", "gpu_runtime.py", "evaluation.py")
IDENTITY = ("run_id", "split", "cohort", "source_sha", "config_sha", "data_sha", "model_sha", "scaler_sha")
SHAPES = {"pred_normalized": [43, 48, 21], "target_normalized": [43, 48, 21], "observed": [43, 48, 21], "origins": [43], "channel_ids": [21]}


def load_runner():
    spec = importlib.util.spec_from_file_location("cpu_cost_frozen_runner", HERE / "cpu_lag_runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def registry_alias(runner, plan, public):
    source = runner.relative(public, plan["postT_registry_relative_path"])
    alias = public / "EXPERIMENT_REGISTRY.csv"
    if runner.sha(source) != plan["postT_registry_sha256"]:
        raise ValueError("Immutable four-condition registration changed")
    if not alias.exists():
        return False
    if not alias.is_file() or alias.read_bytes() != source.read_bytes():
        raise ValueError("Portable-route registry alias must be the exact four-condition registration")
    return True


def fit_binding(runner, plan, public, private, condition):
    run_id = condition["run_id"]
    folder = public / "predictions" / run_id / "V_select"
    paths = dict(receipt=public / "runs" / run_id / "BASE_FIT_RECEIPT.json",
                 source=public / "runs" / run_id / "SOURCE_CONTRACT.json",
                 manifest=folder / "PREDICTION_MANIFEST.json", reference=folder / "REFERENCE_CONTRACT.json")
    missing = [path.relative_to(public).as_posix() for path in paths.values() if not path.is_file()]
    if missing:
        return None, missing
    values = {key: runner.read(path) for key, path in paths.items()}
    receipt, source, manifest, reference = (values[key] for key in ("receipt", "source", "manifest", "reference"))
    task = plan["task_contract"]
    if (receipt.get("status") != "COMPLETE_VALID" or receipt.get("run_id") != run_id
            or receipt.get("family") != condition["family"] or receipt.get("reused_complete_artifact")
            or receipt.get("epochs") != [.25, 1., 3.] or source.get("spec") != condition["spec"]
            or source.get("data_sha") != task["data_sha256"] or source.get("model_sha") != plan["foundation_tensor_sha256"]
            or receipt.get("source_sha") != source.get("source_sha")
            or source.get("files") != {name: plan["source39"][name] for name in FIT_SOURCES}
            or receipt.get("search_fit_end") != 10482 or receipt.get("refit_fit_end") != 12579
            or receipt.get("refit_scaler_sha") != task["refit_scaler_sha256"]):
        raise ValueError("Fresh completed fit or its frozen recipe/data/scaler differs: " + run_id)
    meta, score, cost = manifest["metadata"], manifest["score"], manifest["cost_record"]
    if (manifest.get("run_id") != run_id or manifest.get("split") != "V_select"
            or manifest.get("cohort") != "jena:primary:V_select:stride48"
            or manifest.get("data_sha") != task["data_sha256"] or manifest.get("scaler_sha") != task["refit_scaler_sha256"]
            or meta.get("family") != condition["family"] or meta.get("fold") != "primary"
            or meta.get("artifact_sha256") != receipt["artifact_sha256"] or meta.get("reused_prediction_cache")
            or meta.get("canonical_scaler_fit_end") != 12579
            or any(score.get(key) != value for key, value in (("run_id", run_id), ("split", "V_select"), ("n_origins", 43), ("num_channels", 21), ("horizon", 48)))
            or any(cost.get(key) != value for key, value in (("run_id", run_id), ("batch", 43), ("latent_k", 6), ("actual_F_streams", 258), ("expected_F_streams", 258)))):
        raise ValueError("Actual fresh full V43 bundle binding differs: " + run_id)
    if (set(manifest["arrays"]) != set(SHAPES)
            or any(manifest["arrays"][key]["shape"] != shape for key, shape in SHAPES.items())
            or any(reference.get(key) != manifest.get(key) for key in IDENTITY)
            or reference.get("arrays") != {key: manifest["arrays"][key] for key in SHAPES if key != "pred_normalized"}
            or reference.get("expected_F_streams") != 258):
        raise ValueError("Full V43 reference, array axes or stream contract differs: " + run_id)
    for key in ("artifact_cache_relative_path",):
        if not runner.relative(private, receipt[key]).is_file():
            raise ValueError("Selected artifact is absent: " + run_id)
    if not runner.relative(private, manifest["cache_relative_path"]).is_file():
        raise ValueError("Standard V archive is absent: " + run_id)
    return dict(run_id=run_id, family=condition["family"], receipt=receipt, manifest=manifest,
                metadata_pins={path.relative_to(public).as_posix(): runner.sha(path) for path in paths.values()}), []


def metadata(old_public, old_cache, asset, plan_sha=None):
    runner = load_runner()
    plan = runner.metadata(old_public, old_cache, asset, plan_sha)
    public, private = Path(old_public) / runner.NAME, Path(old_cache) / runner.NAME
    alias_ready = registry_alias(runner, plan, public)
    bindings, missing = [], []
    for condition in plan["conditions"]:
        binding, absent = fit_binding(runner, plan, public, private, condition)
        missing.extend(absent)
        if binding is not None:
            bindings.append(binding)
    return dict(plan=plan, bindings=bindings, missing=missing, alias_ready=alias_ready,
                ready=alias_ready and len(bindings) == 4)


def cost_spec(canonical, binding, plan, runner):
    run_id, receipt, manifest = (binding[key] for key in ("run_id", "receipt", "manifest"))
    expected = dict(run_id=run_id, dataset="jena", family=binding["family"], fold="primary", latent_k=6,
                    canonical_fit_end=12579, foundation_sha=plan["foundation_tensor_sha256"], model_size="small",
                    artifact_cache_relative_path=receipt["artifact_cache_relative_path"], artifact_sha256=receipt["artifact_sha256"],
                    expected_prediction_model_sha=manifest["model_sha"],
                    fit_receipt_public_relative_path="runs/"+run_id+"/BASE_FIT_RECEIPT.json")
    manifest_path = "predictions/"+run_id+"/V_select/PREDICTION_MANIFEST.json"
    standard = dict(model_sha=manifest["model_sha"], archive_sha256=manifest["archive_sha256"],
                    manifest_public_relative_path=manifest_path, manifest_sha256=binding["metadata_pins"][manifest_path])
    if (any(canonical.get(key) != value for key, value in expected.items())
            or canonical.get("standard_V_binding") != standard
            or canonical.get("fit_receipt_sha256") != binding["metadata_pins"][expected["fit_receipt_public_relative_path"]]
            or "accuracy_run_id" in canonical or "row_microbatch" in canonical):
        raise ValueError("Original canonical route differs from selected fit/V binding")
    return dict(canonical, accuracy_run_id=run_id, row_microbatch=128)


def worker_arguments(spec_relative, output_relative, batch, model_relative):
    return ["{PYTHON}", "-X", "utf8", "-u", "-B", "{ORIGINAL_PUBLIC_ROOT}/measure_costs.py", "--worker",
            "--route-spec", "{POST_T_CACHE_ROOT}/"+spec_relative, "--public-root", "{POST_T_PUBLIC_ROOT}",
            "--cache-root", "{POST_T_CACHE_ROOT}", "--model", "{ASSET_ROOT}/"+model_relative,
            "--device", "cpu", "--batch", str(batch), "--warmups", "10", "--timed-trials", "20",
            "--stage-trials", "20", "--output", "{POST_T_PUBLIC_ROOT}/"+output_relative]


class ScienceRuntime:
    def __init__(self, old_public):
        sys.path.insert(0, str(old_public))
        import torch
        import data_contract
        import fit_base
        import route_catalog
        torch.set_num_threads(2)
        if torch.cuda.is_initialized():
            raise ValueError("CPU-only route preparation required")
        self.data, self.fit, self.catalog = data_contract, fit_base, route_catalog

    def task(self, runner, plan, cache):
        task = self.data.load_task("jena", cache_dir=cache)
        runner.validate_task(task, plan, self.data, self.fit)
        return task

    def route(self, run_id, task, private, public):
        return self.catalog.build_route(run_id, task, private, public)


def execute(old_public, old_cache, asset, plan_sha, output, *, _runtime=None):
    if not plan_sha:
        raise ValueError("Explicit immutable post-T plan SHA required")
    state = metadata(old_public, old_cache, asset, plan_sha)
    runner, plan = load_runner(), state["plan"]
    public, private = old_public / runner.NAME, old_cache / runner.NAME
    if not state["ready"]:
        raise ValueError("Wait for all four complete fit/full-V receipts and the exact registry alias; no alias is created here")
    if (not output.is_relative_to(public) or output == public or output.exists()):
        raise FileExistsError("New preparation output under the post-T public root required")
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or any(key.startswith("INFORMS_GPU_") for key in os.environ)
            or any(os.environ.get(key) != "2" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
            or any(os.environ.get(key) != "1" for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))
            or Path(os.environ.get("INFORMS_ASSET_ROOT", "")).resolve() != asset
            or Path(os.environ.get("INFORMS_CACHE_ROOT", "")).resolve() != private):
        raise ValueError("Offline CPU-only environment and exact separate cache roots required")
    if time.time() >= plan["deadline_unix"]:
        raise ValueError("Original campaign wall allowance exhausted")
    used = sum(path.stat().st_size for root in (old_public, old_cache) for path in root.rglob("*") if path.is_file())
    if used+64*1024**2 > plan["storage_limit_bytes"]:
        raise ValueError("Combined campaign storage allowance exhausted")
    token = str(time.time_ns())+"_"+uuid.uuid4().hex
    specs_root = private / "cost_specs" / token
    costs_root = public / "costs" / token
    if specs_root.exists() or costs_root.exists():
        raise FileExistsError("Fresh cost specification and receipt paths required")
    helper_sha = runner.sha(__file__)
    runtime = _runtime if _runtime is not None else ScienceRuntime(old_public)
    try:
        task = runtime.task(runner, plan, old_cache)
        specifications = []
        for binding in state["bindings"]:
            canonical, size = runtime.route(binding["run_id"], task, private, public)
            if size != "small":
                raise ValueError("Original selected Bolt-small route required")
            specifications.append((binding, cost_spec(canonical, binding, plan, runner)))
    finally:
        runner.metadata(old_public, old_cache, asset, plan_sha)
    after = metadata(old_public, old_cache, asset, plan_sha)
    if (not after["ready"] or after["bindings"] != state["bindings"] or runner.sha(__file__) != helper_sha):
        raise ValueError("Preparation source, selected fits or V bindings changed")
    output.mkdir(parents=True, exist_ok=False)
    specs_root.mkdir(parents=True, exist_ok=False)
    jobs = []
    for batch in (1, 8):
        for binding, specification in specifications:
            label = binding["family"]+"_B"+str(batch)
            spec_path = specs_root / (label+".json")
            runner.new(spec_path, specification)
            spec_relative = spec_path.relative_to(private).as_posix()
            receipt_relative = (costs_root / (label+".json")).relative_to(public).as_posix()
            jobs.append(dict(job_id=label, run_id=binding["run_id"], accuracy_run_id=binding["run_id"],
                family=binding["family"], batch=batch, device="cpu", L=512, H=48, C=21, K=6,
                row_microbatch=128, checkpoint_inputs=False, warmups=10, timed_trials=20, stage_trials=20,
                required_receipt_status="COMPLETE_PROTOCOL", expected_F_streams_per_trial=batch*6,
                F_physical_calls="Read actual original worker trial receipts; no fixed call count assumed",
                spec_cache_relative_path=spec_relative, spec_sha256=runner.sha(spec_path),
                artifact_cache_relative_path=specification["artifact_cache_relative_path"], artifact_sha256=specification["artifact_sha256"],
                foundation_tensor_sha256=specification["foundation_sha"], standard_V_binding=specification["standard_V_binding"],
                fit_and_V_metadata_pins=binding["metadata_pins"], output_public_relative_path=receipt_relative,
                arguments=worker_arguments(spec_relative, receipt_relative, batch, plan["model_asset_relative_path"])))
    runner.metadata(old_public, old_cache, asset, plan_sha)
    if metadata(old_public, old_cache, asset, plan_sha)["bindings"] != state["bindings"]:
        raise ValueError("Fit/V metadata changed while specifications were written; preserve partial preparation")
    result = dict(status="PREPARED_NOT_EXECUTED_CPU_COST_WORKERS", post_T_plan_sha256=plan_sha,
        preparation_helper_sha256=helper_sha, original_source39_and_immutable_pins_preserved=True,
        registry_alias_sha256=plan["postT_registry_sha256"], CPU_threads=2, device="cpu", jobs=jobs,
        worker_count=8, workers_are_fresh_sequential_processes=True, maximum_concurrent_cost_workers=1,
        actual_subprocesses_started=0, actual_cost_trials=0, actual_training=0, new_T_target_or_scoring=False,
        requires_all_four_fit_processes_terminated_before_cost_workers=True,
        ROOT_must_verify_no_concurrent_training_or_other_cost_worker=True,
        external_CPU_process_tree_supervisor_required=True,
        worker_has_no_builtin_CPU_deadline_or_RSS_guard=True,
        resource_limits=dict(deadline_unix=plan["deadline_unix"], max_RSS_bytes=plan["max_RSS_bytes"],
                             combined_storage_bytes=plan["storage_limit_bytes"], GPU_usage_charge_expected=0),
        preparation_reads="Original whole safeFIT Task and hash-bound local selected adapter bundles; no foundation construction, inference, training or timing",
        cost_scope="Fresh-process single-route physical-input CPU forecasting; original Route/foundation load time excludes Python/data startup and does not establish cold disk cache",
        memory_scope="Original worker CPU resident weights and after-close RSS; active_adapter_MiB is GPU-only, CPU peak RSS needs external monitoring",
        parameter_count_scope="Use selected hash-bound fit receipts or original networks.parameter_count on the live Route; original cost worker does not emit parameter counts",
        F_calls_scope="Trial F_calls is actual row-call telemetry; standard V bundle F_calls remains its original bookkeeping field",
        created_unix=time.time())
    runner.new(output / "COST_EXECUTION_PLAN.JSON", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for root in ("public", "cache", "asset"):
        parser.add_argument("--"+root+"-root", required=True)
    parser.add_argument("--plan-sha")
    parser.add_argument("--output")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    public, cache, asset = (Path(value).resolve() for value in (args.public_root, args.cache_root, args.asset_root))
    if args.execute:
        if not args.plan_sha or not args.output:
            parser.error("--execute requires --plan-sha and a new --output")
        result = execute(public, cache, asset, args.plan_sha, Path(args.output).resolve())
        print(json.dumps(dict(status=result["status"], worker_count=result["worker_count"], actual_subprocesses_started=0)))
    else:
        state = metadata(public, cache, asset, args.plan_sha)
        print(json.dumps(dict(status="METADATA_READY_FOR_EXPLICIT_PREPARATION" if state["ready"] else "METADATA_WAITING_FOR_FOUR_FITS_V_AND_ALIAS",
            complete_fit_V_bundles=len(state["bindings"]), exact_registry_alias_present=state["alias_ready"],
            missing_metadata_files=state["missing"], numpy_imported="numpy" in sys.modules, torch_imported="torch" in sys.modules,
            actual_model_array_or_cost_execution=0, files_written=0)))


if __name__ == "__main__":
    main()
