"""Prepare one of ten sealed CPU routes' B1/B8 jobs; never launch a worker.

No arguments means no metadata read, write or scientific import. Explicit
preparation reuses the original Task/catalog without constructing a foundation.
"""
import argparse
import contextlib
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
import uuid

NAME = "postT_V_only_cpu_cost_completion_20261016_v1"
KINDS = {
    "required": dict(namespace="postT_V_only_cpu_required_minima_20261016_v1",
        verifier="verify_required_minima_outputs.py",
        verifier_sha="c48cfc2bb738798b8e316fdc86de6d160f954ed634ed0831c5427afdcb3616a5",
        plan_sha="04acc4aabffd02a7266f16fee5c6a3884313fac6ca4b5b46ef0db45eb2fe5152",
        ids=tuple("postT_cpu_beijing_" + block + "_k" + str(k) + "_s92601_required_v1"
            for k in (6, 11) for block in ("E1_B2", "E3_LAG_CAUSAL", "E3_LAG_BIDIR"))),
    "family": dict(namespace="postT_V_only_cpu_family_minima_20261016_v1",
        verifier="verify_family_minima_outputs.py",
        verifier_sha="907488e3bf92e3ac13e90f0fc30c4dac09e241e0b51f7ba7c0b274e654dc45fa",
        plan_sha="e34033935916dcaded0885ccde8e1aa084f5cc0df994024d0827510d6db53b10",
        ids=("postT_cpu_jena_E1_B4_k6_s92601_family_v1",
             "postT_cpu_jena_E1_ADAPTS_BOLT_PORT_k6_s92601_family_v1",
             "postT_cpu_jena_E1_LORA_R8_k0_s92601_family_v1",
             "postT_cpu_beijing_E1_B4_k11_s92601_family_v1")),
}
REPLAY_SOURCES = ("route_catalog.py", "route_predictions.py", "deployment.py", "data_contract.py", "evaluation.py",
                  "training.py", "networks.py", "gpu_runtime.py", "ridge_baseline.py", "combination.py", "fit_base.py")
VALIDATOR = "postT_V_only_cpu_fixed_pca_20261016_v1/fixed_pca_runner.py"
VALIDATOR_SHA = "d73d866a6d03061d00fb7ca015a8f3b41e273b73011b22f0c1cd0fbbfb073dd9"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inside(root, relative):
    root, relative = Path(root).absolute(), Path(relative)
    path = root / relative
    if relative.is_absolute() or ".." in relative.parts or any(item.is_symlink() for item in [path] + list(path.parents)):
        raise ValueError("Explicit nonsymlink root-relative path required")
    path, root = path.resolve(), root.resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("Requested path escapes its explicit root")
    return path


def read(path):
    path = Path(path)
    if path.stat().st_size > 8 * 1024**2:
        raise ValueError("Small JSON metadata required")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def load_verifier(kind, public):
    if kind not in KINDS:
        raise ValueError("Only the sealed required6 and family4 kinds are supported")
    item = KINDS[kind]
    path = inside(public, item["namespace"] + "/" + item["verifier"])
    if sha(path) != item["verifier_sha"]:
        raise ValueError("Frozen independent verifier source changed")
    spec = importlib.util.spec_from_file_location("completion_cost_" + kind + "_verifier", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def metadata(public, cache, asset, kind, run_id, plan_sha):
    if kind not in KINDS or plan_sha != KINDS[kind]["plan_sha"] or run_id not in KINDS[kind]["ids"]:
        raise ValueError("One exact sealed kind/ID/PLAN SHA is required")
    verifier = load_verifier(kind, public)
    runner = verifier.load_runner()
    plan = verifier.metadata(runner, public, cache, asset, plan_sha)
    if (set(row["run_id"] for row in plan["conditions"]) != set(KINDS[kind]["ids"])
            or len(plan["source39"]) != 39 or plan["CPU_threads"] != 2
            or plan["max_RSS_bytes"] != 3 * 1024**3 or plan["storage_limit_bytes"] != 30 * 1024**3
            or plan["admission_reserve_bytes"] != 256 * 1024**2 or plan["deadline_unix"] != 1792222550):
        raise ValueError("Original ten-condition scope or CPU/RSS/storage/deadline contract differs")
    condition = verifier.choose(runner, plan, run_id, "replay")
    post = public / runner.NAME
    alias = inside(post, "EXPERIMENT_REGISTRY.csv")
    if not alias.is_file() or sha(alias) != plan["postT_registry_sha256"]:
        raise ValueError("Exact existing route registry alias is required; this helper creates none")
    return verifier, runner, plan, condition


def binding_record(condition, plan, values, paths, post, kind):
    dataset = condition["dataset"] if kind == "family" else "beijing"
    task = plan["task_contracts"][dataset] if kind == "family" else plan["task_contract"]
    return dict(run_id=condition["run_id"], dataset=dataset, family=condition["family"],
        width=condition.get("effective_F_width", condition["latent_k"]),
        registered_family=condition.get("registered_family", condition["family"]),
        registered_latent_k=condition.get("registered_latent_k", condition["latent_k"]), task_contract=task,
        base=values["fit_receipt"], manifest=values["manifest"],
        receipt_relative=paths["fit_receipt"].relative_to(post).as_posix(), receipt_sha=sha(paths["fit_receipt"]),
        manifest_relative=paths["manifest"].relative_to(post).as_posix(), manifest_sha=sha(paths["manifest"]))


def cost_spec(canonical, size, binding, plan):
    b, manifest = binding["base"], binding["manifest"]
    expected = dict(run_id=binding["run_id"], dataset=binding["dataset"], family=binding["family"], fold="primary",
        latent_k=binding["width"], canonical_fit_end=binding["task_contract"]["primary_split_ranges"]["A_stop"][1],
        foundation_sha=plan["foundation_tensor_sha256"], model_size="small", artifact_cache_relative_path=b["artifact_cache_relative_path"],
        artifact_sha256=b["artifact_sha256"], expected_prediction_model_sha=manifest["model_sha"],
        fit_receipt_public_relative_path=binding["receipt_relative"], fit_receipt_sha256=binding["receipt_sha"],
        standard_V_binding=dict(model_sha=manifest["model_sha"], archive_sha256=manifest["archive_sha256"],
            manifest_public_relative_path=binding["manifest_relative"], manifest_sha256=binding["manifest_sha"]))
    if size != "small" or canonical != expected:
        raise ValueError("Original canonical route differs from this selected fit/full-V/scaler/foundation binding")
    return dict(canonical, accuracy_run_id=binding["run_id"], row_microbatch=128)


def check_files(public, cache, bindings):
    for label, files in bindings.items():
        if label not in {"public", "cache"}:
            raise ValueError("Unknown original input root")
        root = public if label == "public" else cache
        for relative, digest in files.items():
            if sha(inside(root, relative)) != digest:
                raise ValueError("Captured source, fit, full-V or proof bytes changed")


def validate_proofs(verifier, runner, plan, kind, condition, values, paths, bindings,
                    public, cache, rescore_path, rescore_sha, replay_path, replay_sha):
    captured = {label: dict(files) for label, files in bindings.items()}
    post, private = public / runner.NAME, cache / runner.NAME
    channels = len((plan["task_contracts"][condition["dataset"]] if kind == "family" else plan["task_contract"])["columns"])
    width = condition.get("effective_F_width", condition["latent_k"])
    for mode, path, digest in (("rescore", rescore_path, rescore_sha), ("replay", replay_path, replay_sha)):
        path = inside(post, Path(path).relative_to(post))
        if path.name != "RESULT.json" or not re.fullmatch("[0-9a-f]{64}", digest or "") or sha(path) != digest:
            raise ValueError("Explicit original per-ID verification RESULT path and SHA required")
        result, input_path = read(path), path.with_name("INPUT_BINDINGS.json")
        inputs = read(input_path)
        expected = dict(status="PASS", mode=mode, run_id=condition["run_id"], plan_sha256=KINDS[kind]["plan_sha"],
            verifier_sha256=KINDS[kind]["verifier_sha"], requested_device="cpu", CPU_threads=2,
            actual_Torch_CPU_threads=2, cuda_initialized=False, full_V_origin_count=43, horizon=48, num_channels=channels,
            original_source_registry_selection_T_and_captured_inputs_preserved=True,
            new_T_target_or_scoring=False, post_T_development=True, training_rerun=False)
        if (any(result.get(key) != value for key, value in expected.items())
                or inputs.get("files_sha256_by_original_root") != bindings
                or inputs.get("selected_artifact_sha256") != values["fit_receipt"]["artifact_sha256"]
                or inputs.get("full_three_epoch_search_verified") is not True):
            raise ValueError("Actual per-ID verifier proof is not bound to current selected fit/full-V/source inputs")
        scientific = result.get("scientific_result") or {}
        if mode == "rescore":
            if (result.get("saved_full_V_rescoring") is not True or any(scientific.get(key) != value for key, value in
                    dict(status="PASS", n_origins=43, horizon=48, num_channels=channels, checkpoint_replay=False, atol=1e-9, rtol=1e-7).items())):
                raise ValueError("Original independent full-V43 FP64 proof required")
        else:
            if (scientific.get("executed_now") is not True or any(result.get(key) != value for key, value in dict(actual_fresh_CPU_replay_verified=True,
                    actual_checkpoint_forward_in_this_command=True, replay_batch_size=6, replay_origin_count=6, F_row_microbatch=128).items())):
                raise ValueError("An actual fresh CPU6 replay is required; old/cache-hit receipt device is insufficient")
            verifier.validate_replay(scientific, condition["run_id"], width)
            extra = verifier.validate_replay_bindings(runner, scientific, values, paths, private, post)
            replay_bindings_path = path.with_name("REPLAY_BINDINGS.json")
            replay_bindings = read(replay_bindings_path)
            expected_files = {label: dict(files) for label, files in bindings.items()}
            for label, root, files in (("public", public, list(extra[:2]) + [post / "EXPERIMENT_REGISTRY.csv"]),
                                        ("cache", cache, extra[2:])):
                expected_files[label].update({p.relative_to(root).as_posix(): sha(p) for p in files})
                captured[label].update(expected_files[label])
            if (replay_bindings.get("files_sha256_by_original_root") != expected_files
                    or any(replay_bindings.get(key) != value for key, value in dict(executed_now=True,
                        requested_device="cpu", CPU_threads=2, replay_batch_size=6, original_F_row_microbatch=128).items())):
                raise ValueError("Actual CPU6 replay binding capture differs")
            captured["public"][replay_bindings_path.relative_to(public).as_posix()] = sha(replay_bindings_path)
        for item in (path, input_path):
            captured["public"][item.relative_to(public).as_posix()] = sha(item)
    check_files(public, cache, captured)
    return captured


def validate_route_replay(runner, plan, canonical, reference):
    source = runner.fingerprint({name: plan["source39"][name] for name in REPLAY_SOURCES})
    if reference.get("route_sha") != runner.fingerprint(canonical) or reference.get("source_sha") != source:
        raise ValueError("CPU6 replay canonical route or original deployment-source fingerprint differs")


def worker_arguments(spec_relative, output_relative, batch, model_relative):
    if batch not in (1, 8):
        raise ValueError("Only original B1/B8 conditions are supported")
    return ["{PYTHON}", "-X", "utf8", "-u", "-B", "{ORIGINAL_PUBLIC_ROOT}/measure_costs.py", "--worker",
        "--route-spec", "{ORIGINAL_CACHE_ROOT}/" + spec_relative,
        "--public-root", "{POST_T_PUBLIC_ROOT}", "--cache-root", "{POST_T_CACHE_ROOT}",
        "--model", "{ASSET_ROOT}/" + model_relative, "--device", "cpu", "--batch", str(batch),
        "--warmups", "10", "--timed-trials", "20", "--stage-trials", "20",
        "--output", "{ORIGINAL_PUBLIC_ROOT}/" + output_relative]


class ScienceRuntime:
    def __init__(self, public):
        sys.path.insert(0, str(public))
        import torch
        import data_contract
        import fit_base
        import route_catalog
        for api in (data_contract, fit_base, route_catalog):
            if Path(api.__file__).resolve() != public / (api.__name__ + ".py"):
                raise ValueError("Original pinned scientific module location required")
        torch.set_num_threads(2)
        if torch.get_num_threads() != 2 or torch.cuda.is_initialized():
            raise ValueError("Actual CPU2 and no CUDA initialization required")
        self.data, self.fit, self.catalog = data_contract, fit_base, route_catalog

    def task(self, runner, plan, cache, public, dataset, contract):
        path = inside(public, VALIDATOR)
        if sha(path) != VALIDATOR_SHA:
            raise ValueError("Frozen safeFIT task/scaler validator changed")
        spec = importlib.util.spec_from_file_location("completion_cost_task_validator", path)
        validator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(validator)
        task = self.data.load_task(dataset, cache_dir=cache)
        evidence = validator.validate_task(task, dict(task_contracts={dataset: contract}), dataset, self.data, self.fit)
        return task, evidence

    def route(self, run_id, task, private, post):
        return self.catalog.build_route(run_id, task, private, post)


@contextlib.contextmanager
def limits(plan, deadline, output):
    import psutil
    process, stopped = psutil.Process(), threading.Event()
    def check():
        rss = process.memory_info().rss
        if rss > plan["max_RSS_bytes"] or time.time() >= deadline:
            raise RuntimeError("CPU preparation RSS or deadline allowance exhausted")
    check()
    def guard():
        while not stopped.wait(.25):
            try:
                check()
            except RuntimeError:
                new(output / "RESOURCE_STOP.json", dict(status="RESOURCE_STOP", deadline_unix=deadline,
                    rss_bytes=process.memory_info().rss, actual_cost_workers_started=0))
                os._exit(98)
    watcher = threading.Thread(target=guard, daemon=True)
    watcher.start()
    try:
        yield
        check()
    finally:
        stopped.set()
        watcher.join(timeout=1)


def execute(public, cache, asset, kind, run_id, plan_sha, rescore_path, rescore_sha,
            replay_path, replay_sha, deadline, output, *, _runtime=None):
    verifier, runner, plan, condition = metadata(public, cache, asset, kind, run_id, plan_sha)
    post, private = public / runner.NAME, cache / runner.NAME
    output = inside(public, Path(output).relative_to(public))
    if not output.is_relative_to(public / NAME) or output == public / NAME or output.exists():
        raise FileExistsError("Fresh output under the separate cost-completion namespace required")
    if deadline is None or not math.isfinite(deadline) or not time.time() < deadline <= plan["deadline_unix"]:
        raise ValueError("Explicit live deadline inside the original campaign required")
    runner.environment(cache, asset)
    used = sum(path.stat().st_size for root in (public, cache) for path in root.rglob("*") if path.is_file())
    reserve = 256 * 1024**2
    if used + reserve > plan["storage_limit_bytes"] or any(shutil.disk_usage(root).free < reserve for root in (public, cache)):
        raise ValueError("Combined original storage cap and 256MiB reserve required")
    source_sha = sha(__file__)
    output.mkdir(parents=True, exist_ok=False)
    new(output / "STARTED.json", dict(kind=kind, run_id=run_id, plan_sha256=plan_sha,
        preparation_helper_sha256=source_sha, actual_pid=os.getpid(), actual_cost_workers_started=0,
        deadline_unix=deadline, requested_device="cpu", CPU_threads=2, started_unix=time.time()))
    jobs, captured = [], dict(public={}, cache={})
    error_class = post_error = None
    prepared = preserved = task_attempted = route_attempted = False
    try:
        with limits(plan, deadline, output):
            values, _, paths, original_bindings = verifier.prepare_inputs(runner, plan, public, cache, run_id, condition)
            captured = validate_proofs(verifier, runner, plan, kind, condition, values, paths, original_bindings,
                public, cache, rescore_path, rescore_sha, replay_path, replay_sha)
            b = binding_record(condition, plan, values, paths, post, kind)
            runtime = _runtime if _runtime is not None else ScienceRuntime(public)
            task_attempted = True
            task, task_evidence = runtime.task(runner, plan, cache, public, b["dataset"], b["task_contract"])
            new(output / "TASK_BINDING.json", task_evidence)
            route_attempted = True
            canonical, size = runtime.route(run_id, task, private, post)
            specification = cost_spec(canonical, size, b, plan)
            replay_reference = read(inside(post, "route_replays/" + run_id + "/PRE_INFERENCE_REFERENCE.json"))
            validate_route_replay(runner, plan, canonical, replay_reference)
            new(output / "INPUT_BINDINGS.json", dict(files_sha256_by_original_root=captured,
                canonical_route=canonical, registered_family=b["registered_family"], actual_family=b["family"],
                registered_latent_k=b["registered_latent_k"], effective_F_width=b["width"],
                metadata_proof_scope="Exact hash-bound prior full-V FP64 and actual CPU6 replay; this preparation performs neither",
                other_nine_conditions_are_not_a_gate=True))
            metadata(public, cache, asset, kind, run_id, plan_sha)
            check_files(public, cache, captured)
            if time.time() >= deadline:
                raise RuntimeError("Deadline reached before specification writes")
            token = kind + "_" + str(time.time_ns()) + "_" + uuid.uuid4().hex
            specs = inside(cache, NAME + "/cpu_cost_specs/" + token)
            specs.mkdir(parents=True, exist_ok=False)
            for batch in (1, 8):
                path = specs / ("B" + str(batch) + ".json")
                new(path, specification)
                spec_relative = path.relative_to(cache).as_posix()
                receipt_relative = NAME + "/costs/" + token + "/B" + str(batch) + ".json"
                jobs.append(dict(job_id=kind + "_" + run_id + "_B" + str(batch), run_id=run_id, accuracy_run_id=run_id,
                    kind=kind, dataset=b["dataset"], family=b["family"], registered_family=b["registered_family"],
                    batch=batch, device="cpu", model_size="small", L=512, H=48, C=len(b["task_contract"]["columns"]),
                    K=b["width"], registered_latent_k=b["registered_latent_k"], expected_F_streams_per_trial=batch * b["width"],
                    row_microbatch=128, warmups=10, timed_trials=20, stage_trials=20,
                    required_receipt_status="COMPLETE_PROTOCOL", spec_original_cache_relative_path=spec_relative,
                    spec_sha256=sha(path), output_original_public_relative_path=receipt_relative,
                    worker_public_relative_root=runner.NAME, worker_cache_relative_root=runner.NAME,
                    artifact_sha256=b["base"]["artifact_sha256"], accuracy_manifest_sha256=b["manifest_sha"],
                    rescore_RESULT_sha256=rescore_sha, actual_CPU6_replay_RESULT_sha256=replay_sha,
                    lora_parity_and_exact_frozen_restore_must_be_confirmed_from_actual_worker=b["family"] == "LORA",
                    arguments=worker_arguments(spec_relative, receipt_relative, batch, plan["model_asset_relative_path"])))
            prepared = True
    except Exception as error:
        error_class = type(error).__name__
    finally:
        try:
            metadata(public, cache, asset, kind, run_id, plan_sha)
            check_files(public, cache, captured)
            if sha(__file__) != source_sha:
                raise ValueError("Preparation source changed during execution")
            if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
                raise ValueError("Unexpected CUDA initialization")
            preserved = True
        except Exception as error:
            post_error = type(error).__name__
        success = prepared and preserved and error_class is None and post_error is None and time.time() < deadline
        result = dict(status="PREPARED_NOT_EXECUTED_CPU_COST_WORKERS" if success else "FAILED_PRESERVED",
            kind=kind, run_id=run_id, plan_sha256=plan_sha, preparation_helper_sha256=source_sha,
            frozen_verifier_sha256=KINDS[kind]["verifier_sha"], jobs=jobs, worker_count=len(jobs),
            original_source39_registry_scores_cost_selection_T_and_captured_inputs_preserved=preserved,
            source_bindings=dict(source39=plan["source39"], immutable_public=plan.get("immutable_public", {})),
            error_class=error_class, postflight_error_class=post_error,
            actual_Task_load_attempted=task_attempted, actual_canonical_catalog_call_attempted=route_attempted,
            actual_subprocesses_started=0, actual_cost_workers_started=0, actual_cost_trials=0,
            actual_training=0, actual_forecasting=0, actual_foundation_models_constructed=0,
            new_T_target_or_scoring=False, maximum_concurrent_cost_workers=1, external_CPU_tree_guard_required=True,
            original_cost_worker_has_no_CPU_RSS_or_deadline_guard=True,
            resource_limits=dict(CPU_threads=2, max_RSS_bytes=plan["max_RSS_bytes"], deadline_unix=deadline,
                storage_limit_bytes=plan["storage_limit_bytes"], admission_reserve_bytes=reserve),
            requires_ROOT_live_training_inference_and_other_cost_workers_closed_before_cost=True,
            helper_launches_or_kills_any_worker=False, other_nine_conditions_are_not_a_gate=True,
            completed_ten_condition_cost_protocol_claimed=False, expected_workers_if_all_ten_separately_prepared=20,
            original_whole_physical_pipeline_and_counters_unchanged=True, training_peak_measurement_included=False,
            fresh_single_route_cost_is_not_balanced_shared_process_cost=True,
            memory_scope="Original CPU weight accounting/after-close RSS; external process-tree monitor supplies CPU peak RSS",
            preparation_scope="Original safeFIT Task and hash-bound selected adapter catalog only; no F construction/forward/timer/child",
            archive_scope="Original safeFIT may deserialize old T bytes; no new T target, evaluation or selection API",
            finished_unix=time.time())
        new(output / "COST_EXECUTION_PLAN.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--inspect", action="store_true")
    for name in ("public-root", "cache-root", "asset-root", "kind", "run-id", "plan-sha",
                 "rescore-result", "rescore-result-sha", "replay-result", "replay-result-sha", "output"):
        parser.add_argument("--" + name)
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args(argv)
    if not args.execute and not args.inspect:
        print(json.dumps(dict(status="NOT_EXECUTED_NO_INPUTS_READ", registered_conditions=10,
            metadata_files_read=0, files_written=0, actual_Task_array_model_or_child_operations=0,
            numpy_imported="numpy" in sys.modules, torch_imported="torch" in sys.modules, psutil_imported="psutil" in sys.modules)))
        return 0
    required = ("public_root", "cache_root", "asset_root", "kind", "run_id", "plan_sha")
    if args.execute:
        required += ("rescore_result", "rescore_result_sha", "replay_result", "replay_result_sha", "deadline", "output")
    if any(getattr(args, key) is None for key in required):
        parser.error("Explicit roots/kind/ID/SHA and execution proof/deadline/fresh output required")
    public, cache, asset = (Path(value).absolute() for value in (args.public_root, args.cache_root, args.asset_root))
    if args.execute:
        result = execute(public, cache, asset, args.kind, args.run_id, args.plan_sha,
            Path(args.rescore_result).absolute(), args.rescore_result_sha, Path(args.replay_result).absolute(), args.replay_result_sha,
            args.deadline, Path(args.output).absolute())
        print(json.dumps({key: result[key] for key in ("status", "worker_count", "actual_cost_trials", "actual_subprocesses_started")}))
        return 0 if result["status"] == "PREPARED_NOT_EXECUTED_CPU_COST_WORKERS" else 1
    _, runner, _, condition = metadata(public, cache, asset, args.kind, args.run_id, args.plan_sha)
    print(json.dumps(dict(status="SEALED_METADATA_ONLY_NOT_COST_READY", run_id=condition["run_id"], namespace=runner.NAME,
        actual_Task_array_model_or_child_operations=0, files_written=0, complete_fit_full_V_or_CPU_replay_proven_by_this_command=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
