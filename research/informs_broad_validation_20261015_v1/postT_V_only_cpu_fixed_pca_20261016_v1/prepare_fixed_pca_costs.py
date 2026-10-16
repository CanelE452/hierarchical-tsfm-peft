"""Prepare one completed fixed-PCA route's B1/B8 CPU cost specifications only.

Default: stdlib metadata. Execute: original safeFIT Task and SHA-bound local
adapter loads, no foundation construction, forecasting, training, or cost timer.
Actual workers require separate quiet sequential CPU process-tree supervision.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
import uuid

HERE = Path(__file__).resolve().parent
PLAN_SHA = "b4207edec63331cf95b24909c24c692091251dd8e29316f8867b9f66c65a0930"
RUNNER_SHA = "d73d866a6d03061d00fb7ca015a8f3b41e273b73011b22f0c1cd0fbbfb073dd9"
VERIFIER_SHA = "44ad54b2085b503caf83d31932ff6654b92e30c1d5f1d02ba4bb2b87609a0c9f"
FIT_SOURCES = ("fit_base.py", "training.py", "networks.py", "data_contract.py", "evaluation.py", "gpu_runtime.py")
IDENTITY = ("run_id", "split", "cohort", "source_sha", "config_sha", "data_sha", "model_sha", "scaler_sha")


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def load_modules():
    modules = []
    for name, digest in (("fixed_pca_runner.py", RUNNER_SHA), ("verify_fixed_pca_outputs.py", VERIFIER_SHA)):
        path = HERE / name
        if sha(path) != digest:
            raise ValueError("Reviewed fixed-PCA metadata dependency changed")
        spec = importlib.util.spec_from_file_location("pca_cost_" + path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


def alias_ready(runner, plan, post):
    source = runner.relative(post, plan["postT_registry_relative_path"])
    if sha(source) != plan["postT_registry_sha256"]:
        raise ValueError("Separate scientific registration changed")
    alias = post / "EXPERIMENT_REGISTRY.csv"
    if not alias.exists():
        return False
    if not alias.is_file() or alias.read_bytes() != source.read_bytes():
        raise ValueError("Existing portable-route alias differs from exact six-row registration")
    return True


def route_binding(runner, verifier, plan, public, cache, condition):
    post, private = public / runner.NAME, cache / runner.NAME
    run_id, dataset, k = condition["run_id"], condition["spec"]["dataset"], condition["spec"]["latent_k"]
    run, folder = post / "runs" / run_id, post / "predictions" / run_id / "V_select"
    needed = [run / name for name in ("BASE_FIT_RECEIPT.json", "SOURCE_CONTRACT.json", "search/RECEIPT.json", "refit/RECEIPT.json")]
    needed += [folder / name for name in ("PREDICTION_MANIFEST.json", "REFERENCE_CONTRACT.json", "SCORES.csv")]
    missing = [path.relative_to(post).as_posix() for path in needed if not path.is_file()]
    if missing:
        return None, missing
    values, score, paths, pins = verifier.prepare_inputs(runner, plan, public, cache, run_id, condition)
    receipt, source, manifest, reference = (values[key] for key in ("fit_receipt", "source_contract", "manifest", "reference"))
    task = plan["task_contracts"][dataset]
    if (source.get("files") != {name: plan["source39"][name] for name in FIT_SOURCES}
            or receipt.get("source_sha") != source.get("source_sha") or receipt.get("epochs") != [.25, 1., 3.]
            or receipt.get("search_fit_end") != task["primary_split_ranges"]["A_fit"][1]
            or receipt.get("refit_fit_end") != task["primary_split_ranges"]["A_stop"][1]
            or receipt.get("refit_scaler_sha") != task["refit_scaler_sha256"]
            or manifest["metadata"].get("reused_prediction_cache")):
        raise ValueError("Exact fixed source, registered schedule, A50/A60 scaler or fresh V contract differs")
    if (any(reference.get(key) != manifest.get(key) for key in IDENTITY)
            or reference.get("arrays") != {key:value for key,value in manifest["arrays"].items() if key != "pred_normalized"}
            or reference.get("expected_F_streams") != k*task["legal_origin_counts_primary"]["V_select"]):
        raise ValueError("Saved full-V reference and array metadata differ")
    missing = ["private/"+name for name in (receipt["artifact_cache_relative_path"], manifest["cache_relative_path"])
               if not runner.relative(private, name).is_file()]
    if missing:
        return None, missing
    receipt_relative = paths["fit_receipt"].relative_to(post).as_posix()
    manifest_relative = paths["manifest"].relative_to(post).as_posix()
    return dict(run_id=run_id, dataset=dataset, K=k, H=task["H"], C=task["shape"][1],
        V_origins=task["legal_origin_counts_primary"]["V_select"], canonical_fit_end=task["primary_split_ranges"]["A_stop"][1],
        receipt=receipt, manifest=manifest, receipt_relative=receipt_relative, receipt_sha256=pins[paths["fit_receipt"].relative_to(public).as_posix()],
        manifest_relative=manifest_relative, manifest_sha256=pins[paths["manifest"].relative_to(public).as_posix()],
        metadata_pins=pins, fixed_zero_updates_exposures_and_trainable_parameters_verified=True), []


def metadata(public, cache, asset, run_id, plan_sha=None):
    public, cache, asset = map(Path, (public, cache, asset))
    if plan_sha is not None and plan_sha != PLAN_SHA:
        raise ValueError("Only the pinned fixed-PCA plan is supported")
    runner, verifier = load_modules()
    plan = runner.metadata(public, cache, asset, PLAN_SHA)
    condition = verifier.choose(plan, run_id, "rescore")
    alias = alias_ready(runner, plan, public / runner.NAME)
    try:
        binding, missing = route_binding(runner, verifier, plan, public, cache, condition)
        error_class = None
    except (OSError, ValueError, KeyError, TypeError) as error:
        binding, missing, error_class = None, [], type(error).__name__
    return dict(plan=plan, condition=condition, binding=binding, missing=missing, route_error_class=error_class,
        alias_ready=alias, ready=alias and binding is not None, only_requested_run_id_checked=run_id,
        other_five_routes_are_not_a_readiness_gate=True)


def cost_spec(canonical, binding, plan):
    receipt, manifest = binding["receipt"], binding["manifest"]
    expected = dict(run_id=binding["run_id"], dataset=binding["dataset"], family="PCA_FIXED_BOLT", fold="primary", latent_k=binding["K"],
        canonical_fit_end=binding["canonical_fit_end"], foundation_sha=plan["foundation_tensor_sha256"], model_size="small",
        artifact_cache_relative_path=receipt["artifact_cache_relative_path"], artifact_sha256=receipt["artifact_sha256"],
        expected_prediction_model_sha=manifest["model_sha"], fit_receipt_public_relative_path=binding["receipt_relative"],
        fit_receipt_sha256=binding["receipt_sha256"])
    standard = dict(model_sha=manifest["model_sha"], archive_sha256=manifest["archive_sha256"],
        manifest_public_relative_path=binding["manifest_relative"], manifest_sha256=binding["manifest_sha256"])
    if (any(canonical.get(key) != value for key,value in expected.items()) or canonical.get("standard_V_binding") != standard
            or "accuracy_run_id" in canonical or "row_microbatch" in canonical):
        raise ValueError("Original canonical fixed-PCA deployment route differs from selected task/artifact/V")
    return dict(canonical, accuracy_run_id=binding["run_id"], row_microbatch=128)


def worker_arguments(spec_relative, output_relative, batch, model_relative):
    return ["{PYTHON}", "-X", "utf8", "-u", "-B", "{ORIGINAL_PUBLIC_ROOT}/measure_costs.py", "--worker",
        "--route-spec", "{POST_T_CACHE_ROOT}/"+spec_relative, "--public-root", "{POST_T_PUBLIC_ROOT}",
        "--cache-root", "{POST_T_CACHE_ROOT}", "--model", "{ASSET_ROOT}/"+model_relative,
        "--device", "cpu", "--batch", str(batch), "--warmups", "10", "--timed-trials", "20", "--stage-trials", "20",
        "--output", "{POST_T_PUBLIC_ROOT}/"+output_relative]


def preserve_failure(private, cache, run_id, phase, error):
    folder = private / "failures"
    folder.mkdir(exist_ok=True)
    path = folder / ("cost_preparation_"+run_id+"_"+phase+"_"+str(time.time_ns())+".json")
    new(path, dict(error_class=type(error).__name__, error_message=str(error), traceback=traceback.format_exc()))
    return dict(phase=phase, error_class=type(error).__name__, private_failure_cache_relative_path=path.relative_to(cache).as_posix())


class ScienceRuntime:
    def __init__(self, public):
        sys.path.insert(0, str(public))
        import torch
        import data_contract
        import fit_base
        import route_catalog
        torch.set_num_threads(2)
        if torch.cuda.is_initialized():
            raise ValueError("CPU-only preparation required")
        self.data, self.fit, self.catalog = data_contract, fit_base, route_catalog

    def task(self, runner, plan, dataset, cache):
        task = self.data.load_task(dataset, cache_dir=cache)
        runner.validate_task(task, plan, dataset, self.data, self.fit)
        return task

    def route(self, run_id, task, private, public):
        return self.catalog.build_route(run_id, task, private, public)


def execute(public, cache, asset, plan_sha, run_id, output, *, deadline=None, _runtime=None):
    if not run_id or plan_sha != PLAN_SHA:
        raise ValueError("One explicit registered ID and immutable plan SHA required")
    state = metadata(public, cache, asset, run_id, plan_sha)
    runner, _ = load_modules()
    plan = state["plan"]
    post, private = public / runner.NAME, cache / runner.NAME
    if not state["ready"]:
        raise ValueError("Requested fixed route is not ready or exact registry alias absent; no all-six gate or alias creation")
    if output == post or not output.is_relative_to(post) or output.exists():
        raise FileExistsError("A fresh output under the separate fixed-PCA root is required")
    runner.require_environment(public, cache, asset)
    deadline = plan["deadline_unix"] if deadline is None else deadline
    if not time.time() < deadline <= plan["deadline_unix"]:
        raise ValueError("Live deadline within the fixed campaign limit required")
    used = sum(path.stat().st_size for root in (public, cache) for path in root.rglob("*") if path.is_file())
    if used+64*1024**2 > plan["storage_limit_bytes"]:
        raise ValueError("Combined 30GiB storage admission failed")
    if not private.exists():
        private.mkdir(parents=True, exist_ok=False)
    claim_root = private / "cost_preparation_claims"
    claim_root.mkdir(exist_ok=True)
    new(claim_root / (run_id+".json"), dict(run_id=run_id, plan_sha256=plan_sha, helper_sha256=sha(__file__), claimed_unix=time.time()))
    output.mkdir(parents=True, exist_ok=False)
    source_sha = sha(__file__)
    new(output / "STARTED.json", dict(run_id=run_id, helper_sha256=source_sha, plan_sha256=plan_sha,
        actual_pid=os.getpid(), actual_cost_workers_started=0, actual_cost_trials=0))
    import psutil
    process = psutil.Process()
    stop = threading.Event()
    def resource_guard():
        while not stop.wait(.25):
            rss = process.memory_info().rss
            if rss > plan["max_RSS_bytes"] or time.time() >= deadline:
                new(output / "RESOURCE_STOP.json", dict(status="RESOURCE_STOP", actual_pid=process.pid, rss_bytes=rss, deadline_unix=deadline))
                os._exit(98)
    watch = threading.Thread(target=resource_guard, daemon=True)
    watch.start()
    failure, postflight_error, jobs, preserved, failures = None, None, [], False, []
    task_call_attempted, route_call_attempted = False, False
    prepared_route = False
    token = str(time.time_ns())+"_"+uuid.uuid4().hex
    specs_root = private / "cost_specs" / token
    try:
        runtime = _runtime if _runtime is not None else ScienceRuntime(public)
        binding = state["binding"]
        task_call_attempted = True
        task = runtime.task(runner, plan, binding["dataset"], cache)
        route_call_attempted = True
        canonical, size = runtime.route(run_id, task, private, post)
        if size != "small":
            raise ValueError("Pinned Bolt-small canonical route required")
        specification = cost_spec(canonical, binding, plan)
        after = metadata(public, cache, asset, run_id, plan_sha)
        if not after["ready"] or after["binding"] != binding or sha(__file__) != source_sha:
            raise ValueError("Fixed route source or selected input binding changed")
        specs_root.mkdir(parents=True, exist_ok=False)
        for batch in (1, 8):
            label = "PCA_FIXED_BOLT_k%d_B%d" % (binding["K"], batch)
            spec_path = specs_root / (label+".json")
            new(spec_path, specification)
            spec_relative = spec_path.relative_to(private).as_posix()
            receipt_relative = "costs/"+token+"/"+label+".json"
            jobs.append(dict(job_id=label, run_id=run_id, accuracy_run_id=run_id, dataset=binding["dataset"], family="PCA_FIXED_BOLT",
                batch=batch, device="cpu", L=512, H=binding["H"], C=binding["C"], K=binding["K"], V_origins=binding["V_origins"],
                row_microbatch=128, checkpoint_inputs=False, warmups=10, timed_trials=20, stage_trials=20,
                expected_F_streams_per_trial=batch*binding["K"], required_receipt_status="COMPLETE_PROTOCOL",
                spec_cache_relative_path=spec_relative, spec_sha256=sha(spec_path), artifact_sha256=specification["artifact_sha256"],
                artifact_cache_relative_path=specification["artifact_cache_relative_path"], foundation_tensor_sha256=plan["foundation_tensor_sha256"],
                standard_V_binding=specification["standard_V_binding"], fit_and_V_metadata_pins=binding["metadata_pins"],
                output_public_relative_path=receipt_relative,
                arguments=worker_arguments(spec_relative, receipt_relative, batch, plan["model_asset_relative_path"])))
        prepared_route = True
    except Exception as error:
        failure = type(error).__name__
        failures.append(preserve_failure(private, cache, run_id, "preparation", error))
    finally:
        stop.set()
        watch.join(timeout=2)
        try:
            after = metadata(public, cache, asset, run_id, plan_sha)
            preserved = after["ready"] and after["binding"] == state["binding"] and sha(__file__) == source_sha
            if not preserved:
                raise ValueError("Input bindings changed during cost preparation")
        except Exception as error:
            postflight_error = type(error).__name__
            failures.append(preserve_failure(private, cache, run_id, "postflight", error))
        success = prepared_route and failure is None and postflight_error is None and preserved
        result = dict(status="PREPARED_NOT_EXECUTED_CPU_COST_WORKERS" if success else "FAILED_PRESERVED", run_id=run_id,
            post_T_plan_sha256=plan_sha, preparation_helper_sha256=source_sha, verifier_sha256=VERIFIER_SHA, runner_sha256=RUNNER_SHA,
            original_source39_registry_lock_and_selected_inputs_preserved=preserved,
            requested_routes_prepared=1 if success else 0, worker_count=len(jobs), jobs=jobs, error_class=failure, postflight_error_class=postflight_error,
            private_failures=failures,
            actual_cost_workers_started=0, actual_subprocesses_started=0, actual_cost_trials=0, actual_training=0, actual_forecasting=0,
            actual_foundation_models_constructed=0, original_safeFIT_task_load_call_attempted=task_call_attempted,
            original_route_builder_call_attempted=route_call_attempted,
            original_route_artifact_load_contract="Hash-bound campaign-owned local bundle; original route_catalog torch.load weights_only=False,map_location=cpu",
            fixed_zero_optimizer_updates_training_exposures_trainable_parameters_verified=True,
            BASE_literal_AdamW_is_not_evidence_of_actual_training=True,
            other_five_routes_are_not_a_readiness_gate=True, new_T_target_or_scoring=False, GPU_owner_or_lease_created=False,
            deadline_unix=deadline, max_RSS_bytes=plan["max_RSS_bytes"], CPU_threads=2,
            actual_pid=process.pid, actual_create_time=process.create_time(),
            source_bindings=dict(source39=plan["source39"], immutable_public=plan["immutable_public"], selected_metadata=state["binding"]["metadata_pins"]),
            external_CPU_process_tree_supervision_required=True, maximum_concurrent_cost_workers=1,
            cost_scope="Fresh CPU process per single route; no balanced rotation or cold-disk-cache claim",
            archive_deserialization_scope="Original permitted safeFIT loader may physically deserialize T bytes; no T target/evaluation requested")
        new(output / "COST_EXECUTION_PLAN.JSON", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("public", "cache", "asset"):
        parser.add_argument("--"+name+"-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--plan-sha")
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--output")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    public, cache, asset = (Path(value).resolve() for value in (args.public_root, args.cache_root, args.asset_root))
    if args.execute:
        if not args.plan_sha or not args.output:
            parser.error("--execute requires --plan-sha and a new --output")
        result = execute(public, cache, asset, args.plan_sha, args.run_id, Path(args.output).resolve(), deadline=args.deadline)
        print(json.dumps(dict(status=result["status"], worker_count=result["worker_count"], actual_cost_trials=0)))
        raise SystemExit(0 if result["status"] == "PREPARED_NOT_EXECUTED_CPU_COST_WORKERS" else 1)
    state = metadata(public, cache, asset, args.run_id, args.plan_sha)
    print(json.dumps(dict(status="METADATA_READY_FOR_ONE_ROUTE_PREPARATION" if state["ready"] else "METADATA_REQUESTED_ROUTE_NOT_READY",
        run_id=args.run_id, missing_metadata_files=state["missing"], route_error_class=state["route_error_class"], exact_registry_alias_present=state["alias_ready"],
        no_all_six_gate=True, numpy_imported="numpy" in sys.modules, torch_imported="torch" in sys.modules,
        actual_preparation_execution=0, actual_cost_trials=0, files_written=0)))


if __name__ == "__main__":
    main()
