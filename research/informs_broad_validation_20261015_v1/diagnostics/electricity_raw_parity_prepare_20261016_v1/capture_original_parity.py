"""Preserve original Electricity Raw B1/B8 parity returns; metadata only by default."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PLAN_REL = Path("diagnostics") / HERE.name / "PLAN.json"


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


def inside(root, relative):
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError("Pinned paths must be root-relative")
    target = (Path(root).resolve() / value).resolve()
    if not target.is_relative_to(Path(root).resolve()):
        raise ValueError("Pinned path leaves its root")
    return target


def metadata(public, cache, asset, expected_plan_sha=None):
    public, cache, asset = map(Path, (public, cache, asset))
    plan_path = public / PLAN_REL
    if expected_plan_sha is not None and sha(plan_path) != expected_plan_sha:
        raise ValueError("Plan SHA changed")
    plan = read(plan_path)
    if (plan["helper_sha256"] != sha(__file__) or len(plan["source39"]) != 39
            or (plan["dataset"], plan["L"], plan["H"], plan["C"]) != ("electricity", 512, 48, 321)
            or plan["cost_cumulative_ceiling_seconds"] != 43200
            or plan["candidates"] != ["full", 32, 8, 4] or plan["atol"] != 1e-5 or plan["rtol"] != 1e-4
            or [row["batch"] for row in plan["conditions"]] != [1, 8]):
        raise ValueError("Helper, scientific or resource contract changed")
    if not {"EXPERIMENT_REGISTRY.csv", "SELECTION_LOCK.json", "DATA_CONTRACT.json"}.issubset(plan["immutable_public"]):
        raise ValueError("Registry, selection lock and data contract pins are required")
    for pins in (plan["source39"], plan["immutable_public"]):
        for name, digest in pins.items():
            if sha(inside(public, name)) != digest:
                raise ValueError("Frozen public input changed")
    if sha(inside(asset, plan["external_instance_norm_source_relative_path"])) != plan["external_instance_norm_source_sha256"]:
        raise ValueError("Instance normalization source changed")
    if not inside(asset, plan["model_asset_relative_path"]).is_dir():
        raise ValueError("Pinned local foundation asset is absent")
    canonical = plan["canonical_accuracy_manifest"]
    if sha(inside(public, canonical["relative_path"])) != canonical["sha256"]:
        raise ValueError("Canonical V manifest changed")
    manifest = read(inside(public, canonical["relative_path"]))
    accuracy_id = "primary_electricity_E1_RAW_BOLT_SMALL_k0_s92601"
    if (manifest["run_id"] != accuracy_id or manifest["split"] != "V_select"
            or manifest["data_sha"] != canonical["data_sha256"] or manifest["scaler_sha"] != canonical["scaler_sha"]):
        raise ValueError("Canonical V identity changed")
    for condition in plan["conditions"]:
        if sha(inside(cache, condition["spec_cache_relative_path"])) != condition["spec_sha256"]:
            raise ValueError("Original condition spec changed")
        spec = read(inside(cache, condition["spec_cache_relative_path"]))
        binding = spec["standard_V_binding"]
        if (spec["run_id"] != accuracy_id + "__rowsfull" or spec["accuracy_run_id"] != accuracy_id
                or (spec["dataset"], spec["family"], spec["fold"], spec["latent_k"], spec["model_size"]) != ("electricity", "RAW", "primary", 321, "small")
                or spec.get("cost_origins") is not None or spec["row_microbatch"] != "full"
                or spec["foundation_sha"] != plan["foundation_tensor_sha256"]
                or spec["expected_prediction_model_sha"] != manifest["model_sha"]
                or binding != dict(manifest_public_relative_path=canonical["relative_path"], manifest_sha256=canonical["sha256"],
                                   model_sha=manifest["model_sha"], archive_sha256=manifest["archive_sha256"])
                or len(condition["original_failures"]) != 4):
            raise ValueError("Original Raw accuracy route changed")
        for failure in condition["original_failures"]:
            execution = inside(public, failure["execution_public_relative_path"])
            if (sha(execution) != failure["execution_sha256"]
                    or sha(inside(cache, failure["private_log_cache_relative_path"])) != failure["private_log_sha256"]):
                raise ValueError("Original failure receipt or private log changed")
            recorded = read(execution)
            if recorded["returncode"] == 0 or recorded["success_receipt_present"]:
                raise ValueError("Original failed condition is no longer a failed condition")
    return plan


def capture(function, route, physical, observed, lease, path):
    if Path(path).exists():
        raise FileExistsError("Saved parity return already exists")
    result = function(route, physical, observed, lease=lease)
    new(path, result)
    return result


def execute_child(public, cache, asset, output, expected_plan_sha):
    if not expected_plan_sha:
        raise ValueError("Execution requires an explicit plan SHA")
    plan = metadata(public, cache, asset, expected_plan_sha)
    if not output.is_relative_to(public / "diagnostics"):
        raise ValueError("New public diagnostics output required")
    output.mkdir(parents=True, exist_ok=False)
    new(output / "RUN_CLAIM.json", dict(plan_sha256=expected_plan_sha, helper_sha256=sha(__file__),
        started_unix=time.time(), original_parity_calls_planned=2, training=0, new_T_scoring=0))
    sys.path.insert(0, str(public))
    import measure_costs as original
    import numpy as np
    import torch
    lease = original.worker_lease_check(cache, "cuda")
    if lease is None:
        raise ValueError("An authenticated existing GPU owner child is required")
    torch.set_num_threads(2)
    task = original.load_task("electricity", cache_dir=cache)
    if (task["L"], task["H"], len(task["columns"])) != (512, 48, 321):
        raise ValueError("Original Electricity task shape changed")
    candidates = original.origins(task["split_ranges"]["V_select"], task["L"], task["H"], task["H"])
    if not len(candidates):
        raise ValueError("Original V history cohort is empty")
    canonical = plan["canonical_accuracy_manifest"]
    canonical_manifest = read(public / canonical["relative_path"])
    rows = []
    for condition in plan["conditions"]:
        batch = condition["batch"]
        row = dict(batch=batch, status="NOT_STARTED", original_raw_row_parity_calls_actual=0)
        route = None
        try:
            lease.check()
            if lease.lease.remaining < 30:
                row.update(status="DEFERRED_RESOURCE_REMAINDER_LT30_SECONDS")
                continue
            spec = read(cache / condition["spec_cache_relative_path"])
            oo = np.resize(candidates[:batch], batch).tolist()
            physical = np.stack([task["physical"][o-task["L"]:o] for o in oo])
            observed = np.stack([task["observed"][o-task["L"]:o] for o in oo])
            row["started_unix"] = time.time()
            route = original.Route(spec, task, cache, asset / plan["model_asset_relative_path"], "cuda")
            binding = original.accuracy_binding(spec, route, task, cache, public)
            if (route.family != "RAW" or binding["status"] != "BOUND_EXACT_ROUTE" or binding["artifact_bindings"] != {}
                    or binding["foundation_state_sha256"] != plan["foundation_tensor_sha256"]
                    or binding["data_sha256"] != canonical["data_sha256"] or binding["scaler_sha"] != canonical["scaler_sha"]
                    or binding["accuracy_prediction_model_sha"] != canonical_manifest["model_sha"]
                    or binding["accuracy_manifest_sha256"] != canonical["sha256"]):
                raise ValueError("Actual Raw model, data, scaler or artifact binding changed")
            new(output / ("INPUT_BINDING_B%d.json" % batch), dict(plan_sha256=expected_plan_sha, helper_sha256=sha(__file__),
                accuracy_binding=binding, origins=list(map(int, oo)), batch=batch, L=512, H=48, C=321,
                physical_input_sha256=hashlib.sha256(physical.tobytes()).hexdigest(),
                observation_mask_sha256=hashlib.sha256(observed.tobytes()).hexdigest(),
                forecast_target_arrays_constructed_or_scored=False, parity_forward_inputs_are_V_past_histories=True,
                whole_asset_file_hash_or_load_is_not_row_isolated=True, original_worker_cohort_construction_unchanged=True,
                raw_arrays_published=False))
            lease.check()
            row["original_raw_row_parity_calls_actual"] = 1
            target = output / ("RAW_PARITY_B%d.json" % batch)
            parity = capture(original.raw_row_parity, route, physical, observed, lease, target)
            row.update(status="ORIGINAL_PARITY_RETURN_PRESERVED", original_gate_passed=parity["successful_variants_parity"],
                       parity_return_relative_path=target.relative_to(public).as_posix(), parity_return_sha256=sha(target))
        except Exception as error:
            row.update(status="FAILED_PRESERVED", failure_class=type(error).__name__)
        finally:
            if route is not None:
                try:
                    route.close()
                    row["backbone_exact_restore_at_close"] = True
                except Exception as error:
                    row.update(status="FAILED_PRESERVED", close_failure_class=type(error).__name__)
            row["finished_unix"] = time.time()
            rows.append(row)
            new(output / ("BATCH_B%d_RESULT.json" % batch), row)
    post_failure = None
    try:
        metadata(public, cache, asset, expected_plan_sha)
    except Exception as error:
        post_failure = type(error).__name__
    complete = len(rows) == 2 and all(row["status"] == "ORIGINAL_PARITY_RETURN_PRESERVED" for row in rows) and post_failure is None
    new(output / "RESULT.json", dict(status="DIAGNOSTIC_COMPLETE_ORIGINAL_PARITY_RETURNS_PRESERVED" if complete else "DIAGNOSTIC_PARTIAL_OR_FAILED",
        batches=rows, plan_sha256=expected_plan_sha, helper_sha256=sha(__file__), postflight_failure_class=post_failure,
        source39_registry_selection_lock_and_original_failures_preserved=post_failure is None,
        parity_defaults_unchanged=True, original_raw_row_parity_calls_actual=sum(row["original_raw_row_parity_calls_actual"] for row in rows),
        warmups=0, timed_trials=0, stage_trials=0, cost_protocol_completed=False, old_failure_receipts_replaced=False,
        raw_forecast_or_target_publication=False, training=0, new_T_or_protected_TEST_scoring=0, finished_unix=time.time()))
    return 0 if complete else 1


def wait_owned_child(child, max_wait, process_api=None):
    if process_api is None:
        import psutil as process_api
    def exact_process(row):
        try:
            process = process_api.Process(row["pid"])
            return process if abs(process.create_time()-row["create_time"]) < .001 else None
        except process_api.NoSuchProcess:
            return None
    root_binding = None
    if child.poll() is None:
        try:
            process = process_api.Process(child.pid)
            root_binding = dict(pid=process.pid, create_time=process.create_time())
        except process_api.NoSuchProcess:
            pass
        if child.poll() is not None:
            root_binding = None
    result = dict(child_wait_timeout_seconds=max_wait, child_timeout=False,
                  child_terminate_requested=False, child_kill_requested=False,
                  child_root_binding=root_binding, child_tree_snapshot=[], remaining_owned_processes=[],
                  child_tree_signal_failures=[])
    teardown_started = None
    try:
        code = child.wait(timeout=max_wait)
    except subprocess.TimeoutExpired:
        result["child_timeout"] = True
        teardown_started = time.monotonic()
        root = exact_process(root_binding) if root_binding else None
        snapshot = []
        if root is not None:
            try:
                snapshot = [root_binding] + [dict(pid=p.pid, create_time=p.create_time()) for p in root.children(recursive=True)]
            except process_api.NoSuchProcess:
                snapshot = [root_binding]
        result["child_tree_snapshot"] = snapshot
        waiting = []
        for row in reversed(snapshot):
            process = exact_process(row)
            if process is None:
                continue
            result["child_terminate_requested"] = True
            try:
                process.terminate()
            except process_api.NoSuchProcess:
                continue
            except Exception as error:
                result["child_tree_signal_failures"].append(dict(**row, action="terminate", failure_class=type(error).__name__))
            waiting.append(process)
        _, alive = process_api.wait_procs(waiting, timeout=10)
        killing = []
        alive_pids = {p.pid for p in alive}
        for row in reversed(snapshot):
            if row["pid"] not in alive_pids:
                continue
            process = exact_process(row)
            if process is None:
                continue
            result["child_kill_requested"] = True
            try:
                process.kill()
            except process_api.NoSuchProcess:
                continue
            except Exception as error:
                result["child_tree_signal_failures"].append(dict(**row, action="kill", failure_class=type(error).__name__))
            killing.append(process)
        if killing:
            process_api.wait_procs(killing, timeout=5)
        result["remaining_owned_processes"] = [row for row in snapshot if exact_process(row) is not None]
        code = child.poll()
    result.update(actual_exitcode=code, child_join_confirmed=code is not None,
                  child_tree_join_confirmed=code is not None and not result["remaining_owned_processes"],
                  child_teardown_seconds=time.monotonic()-teardown_started if teardown_started is not None else 0.)
    return result


def run_with_owner(lease, asset_root, cache_root, public_root, expected_plan_sha):
    if (lease is None or getattr(lease, "purpose", None) != "audit" or not getattr(lease, "_entered", False)
            or getattr(getattr(lease, "lock", None), "file", None) is None or getattr(lease, "pid", None) != os.getpid()
            or getattr(lease, "total_gpu_seconds", None) != 43200 or getattr(lease, "reserve_seconds", None) != 0):
        raise ValueError("Same-PID existing ROOT audit43200/reserve0 owner required; no nested owner")
    if not expected_plan_sha:
        raise ValueError("Execution requires an explicit plan SHA")
    lease.check()
    if lease.remaining < 30 or lease.deadline > 1792223150:
        raise ValueError("Insufficient remainder or campaign wall exceeded")
    public, cache, asset = (Path(value).resolve() for value in (public_root, cache_root, asset_root))
    metadata(public, cache, asset, expected_plan_sha)
    stamp = str(time.time_ns())
    output = public / "diagnostics" / ("electricity_raw_parity_actual_" + stamp)
    private = cache / "final_execution" / ("electricity_raw_parity_actual_" + stamp)
    private.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, **lease.child_environment(), INFORMS_ASSET_ROOT=str(asset), INFORMS_CACHE_ROOT=str(cache),
               CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    command = [sys.executable, "-X", "utf8", "-u", "-B", str(Path(__file__).resolve()), "--public-root", str(public),
               "--cache-root", str(cache), "--asset-root", str(asset), "--output", str(output), "--plan-sha", expected_plan_sha, "--execute"]
    with (private / "CHILD.log").open("xb") as stream:
        lease.check()
        remaining = lease.remaining
        if remaining < 30:
            raise ValueError("Insufficient remainder immediately before child spawn")
        max_wait = min(120, max(0, remaining-25))
        child = subprocess.Popen(command, cwd=public, env=env, stdout=stream, stderr=subprocess.STDOUT)
        lifecycle = wait_owned_child(child, max_wait)
    result = dict(**lifecycle, child_launcher_pid=child.pid, public_output_relative_path=output.relative_to(public).as_posix(),
                  private_execution_cache_relative_path=private.relative_to(cache).as_posix(),
                  private_log_sha256=sha(private / "CHILD.log"), new_or_nested_GPU_owner_created=False,
                  original_cost_failure_replaced=False)
    new(private / "CHILD_EXECUTION.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("public", "cache", "asset"):
        parser.add_argument("--" + name + "-root", default=os.environ.get("INFORMS_" + name.upper() + "_ROOT"))
    parser.add_argument("--output")
    parser.add_argument("--plan-sha")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not all((args.public_root, args.cache_root, args.asset_root)):
        parser.error("public/cache/asset root required")
    public, cache, asset = (Path(value).resolve() for value in (args.public_root, args.cache_root, args.asset_root))
    if args.execute:
        if not args.output or not args.plan_sha:
            parser.error("Execution requires new --output and --plan-sha")
        raise SystemExit(execute_child(public, cache, asset, Path(args.output).resolve(), args.plan_sha))
    metadata(public, cache, asset, args.plan_sha)
    print(json.dumps(dict(status="PREPARATION_ONLY", plan_sha256=sha(public / PLAN_REL), helper_sha256=sha(__file__),
        numpy_imported="numpy" in sys.modules, torch_imported="torch" in sys.modules,
        actual_GPU_parity_calls=0, model_or_array_reads=0)), flush=True)


if __name__ == "__main__":
    main()
