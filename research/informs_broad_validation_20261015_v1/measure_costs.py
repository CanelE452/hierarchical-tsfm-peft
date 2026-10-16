"""Whole-pipeline cost measurement; GPU workers require a live owner lease."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import psutil
import torch

from data_contract import load_task, origins, sha256_file
from deployment import FoundationPool, Route
from resources import BudgetExceeded, validate_gpu_child
import route_catalog
import fit_base
from training import fingerprint


class CostLease:
    def __init__(self, lease):
        self.lease = lease

    def check(self):
        self.lease.check()
        if self.lease.remaining < 15:
            raise BudgetExceeded("Reserve 15 seconds for a submitted CUDA operation and orderly teardown")
        return self
from combination import scaler_fingerprint


def accuracy_binding(spec, route, task, cache, public):
    """Bind a measured condition to a concrete, immutable accuracy route."""
    condition = spec["run_id"]
    accuracy = spec.get("accuracy_run_id")
    if not accuracy:
        return dict(status="UNBOUND_NO_DECLARED_ACCURACY",condition_id=condition,accuracy_run_id=None)
    canonical, size = route_catalog.build_route(accuracy,task,cache,public)
    ignored = {"accuracy_run_id","row_microbatch","cost_origins"}
    scientific = {key:value for key,value in spec.items() if key not in ignored}
    scientific["run_id"] = accuracy
    expected = {key:value for key,value in canonical.items() if key not in ignored}
    if scientific != expected:
        raise ValueError("Cost condition changes the exact accuracy route beyond declared measurement settings")
    candidates = [Path(public)/"predictions"/accuracy/"V_select/PREDICTION_MANIFEST.json",
                  Path(public)/"ordered_predictions"/accuracy/"PREDICTION_MANIFEST.json",
                  Path(public)/"route_predictions"/accuracy/"V_select/PREDICTION_MANIFEST.json"]
    path = next((path for path in candidates if path.is_file()),None)
    if path is None: raise ValueError("Declared cost accuracy route lacks an actual V manifest")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    scaler = scaler_fingerprint(route.mean,route.std)
    if manifest["run_id"] != accuracy or manifest["split"] != "V_select" or manifest["data_sha"] != task["data_sha256"] or manifest["scaler_sha"] != scaler:
        raise ValueError("Cost condition accuracy data/scaler/identity differs")
    foundations = {}
    def visit(value,prefix):
        found = {key:value[key] for key in ("foundation_sha","foundation_asset","foundation_revision","model_size") if key in value}
        if found: foundations[prefix] = found
        for key in ("g_route","b_route"):
            if key in value: visit(value[key],prefix+"."+key)
    visit(canonical,"route")
    uses_F = route_catalog.expected_F_streams(canonical,1,len(task["columns"])) > 0
    actual_F = fit_base._model_sha(route.pool.foundation) if uses_F else "NO_FOUNDATION_USED"
    return dict(status="BOUND_EXACT_ROUTE",condition_id=condition,accuracy_run_id=accuracy,
                cost_route_spec=dict(spec),cost_route_spec_sha256=fingerprint(spec),
                accuracy_route_spec=canonical,accuracy_route_spec_sha256=fingerprint(canonical),
                artifact_bindings=route_catalog.artifact_bindings(canonical),foundation_bindings=foundations,
                foundation_state_sha256=actual_F,model_size=size,data_sha256=task["data_sha256"],scaler_sha=scaler,
                accuracy_manifest_public_relative_path=path.relative_to(public).as_posix(),accuracy_manifest_sha256=sha256_file(path),
                accuracy_prediction_model_sha=manifest["model_sha"],accuracy_prediction_source_sha=manifest["source_sha"])


def _summaries_with_accuracy(rows, bindings):
    summaries = summarize(rows)
    for row in summaries:
        row.update(condition_id=row["run_id"],accuracy_run_id=bindings[row["run_id"]].get("accuracy_run_id"))
    return summaries


def worker_lease_check(cache, device):
    if str(device).startswith("cuda"):
        return CostLease(validate_gpu_child(cache)).check()
    return None


def raw_row_parity(route, physical, observed, candidates=("full", 32, 8, 4), atol=1e-5, rtol=1e-4, lease=None):
    if route.family != "RAW":
        return {"status": "NOT_RAW"}
    if route.device.startswith("cuda") and lease is None:
        raise RuntimeError("GPU parity requires an active campaign owner/child lease")
    previous = route.spec.get("row_microbatch", "full")
    reference, reference_label, rows = None, None, []
    try:
        for candidate in candidates:
            if lease is not None: lease.check()
            route.spec["row_microbatch"] = candidate
            try:
                predicted = route.predict_physical(physical, observed)
                if reference is None:
                    reference, reference_label = predicted.copy(), candidate
                passed = bool(np.allclose(reference, predicted, atol=atol, rtol=rtol))
                rows.append({"row_microbatch": candidate, "status": "PASS" if passed else "FAIL",
                             "max_abs": float(np.abs(reference - predicted).max()),
                             "prediction_sha256": hashlib.sha256(predicted.tobytes()).hexdigest(),
                             "F_logical_streams": route.last_receipt["F_logical_streams"]})
            except torch.cuda.OutOfMemoryError:
                rows.append({"row_microbatch": candidate, "status": "BLOCKED_CUDA_OOM"})
                if str(route.device).startswith("cuda"): torch.cuda.empty_cache()
            finally:
                route.deactivate()
    finally:
        route.spec["row_microbatch"] = previous
    return {"status": "CHECKED", "reference_row_microbatch": reference_label, "variants": rows,
            "all_passed": bool(rows and all(row["status"] == "PASS" for row in rows)),
            "successful_variants_parity": bool(reference is not None and not any(row["status"] == "FAIL" for row in rows)),
            "atol": atol, "rtol": rtol, "scope": "Actual physical-output inference, not cached forecasts"}


def measure_balanced(routes, physical, observed, *, warmups=10, timed_trials=20, stage_trials=0, lease=None):
    if not routes or min(warmups, timed_trials, stage_trials) < 0 or timed_trials < 1:
        raise ValueError("Routes and positive timed trial count required")
    if any(route.device.startswith("cuda") for route in routes) and lease is None:
        raise RuntimeError("GPU timing requires an active campaign owner/child lease")
    rows = []
    for phase, repeats, instrumented in (("warmup", warmups, False), ("timed", timed_trials, False), ("stage_probe", stage_trials, True)):
        if phase == "timed" and any(route.device.startswith("cuda") for route in routes):
            torch.cuda.reset_peak_memory_stats()
        for repeat in range(repeats):
            for position in range(len(routes)):
                route = routes[(repeat + position) % len(routes)]
                if lease is not None: lease.check()
                route.activate()
                try:
                    prediction = route.predict_physical(physical, observed, measure_stages=instrumented)
                    row = {"run_id": route.run_id, "phase": phase, "trial": repeat, "order_position": position,
                           "batch_size": len(physical), "whole_seconds": route.last_receipt["whole_seconds"],
                           "F_logical_streams": route.last_receipt["F_logical_streams"],
                           "F_physical_streams": route.last_receipt["F_physical_streams"], "F_calls": route.last_receipt["F_calls"],
                           "prediction_sha256": hashlib.sha256(prediction.tobytes()).hexdigest(),
                           "row_microbatch": route.spec.get("row_microbatch", "full" if route.family == "RAW" else 128),
                           "stage_instrumentation": instrumented}
                    row.update({"stage_" + key + "_seconds": value for key, value in route.last_receipt["stage_seconds"].items()})
                    rows.append(row)
                finally:
                    if len(routes) > 1: route.deactivate()
    return rows


def summarize(rows):
    result = []
    for run_id in sorted({row["run_id"] for row in rows}):
        for phase in ("timed", "stage_probe"):
            group = [row for row in rows if row["run_id"] == run_id and row["phase"] == phase]
            if not group: continue
            seconds = [row["whole_seconds"] for row in group]
            result.append({"run_id": run_id, "phase": phase, "batch_size": group[0]["batch_size"], "trials": len(group),
                           "median_seconds": statistics.median(seconds), "mean_seconds": statistics.mean(seconds),
                           "min_seconds": min(seconds), "max_seconds": max(seconds),
                           "p90_seconds": float(np.quantile(seconds, .9)),
                           "stage_instrumentation": phase == "stage_probe", "stage_sum_equals_whole_claimed": False})
    return result


def _json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def _write_trials(path, rows):
    with gzip.open(path, "xt", encoding="utf-8", newline="") as stream:
        fields = sorted(set().union(*(row.keys() for row in rows)))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def worker(args):
    cache = Path(args.cache_root)
    lease = worker_lease_check(cache, args.device)
    torch.set_num_threads(2)
    spec = json.loads(Path(args.route_spec).read_text(encoding="utf-8"))
    task = load_task(spec["dataset"], cache_dir=cache)
    oo = spec.get("cost_origins")
    if oo is None:
        candidates = origins(task["split_ranges"]["V_select"], task["L"], task["H"], task["H"])
        if not len(candidates): raise ValueError("No registered V cost history")
        oo = np.resize(candidates[:args.batch], args.batch).tolist()
    if len(oo) != args.batch or any(o < task["L"] or o > task["split_ranges"]["V_select"][1] for o in oo):
        raise ValueError("Fixed cost history cohort is outside pre-T V observations")
    physical = np.stack([task["physical"][o - task["L"]:o] for o in oo])
    observed = np.stack([task["observed"][o - task["L"]:o] for o in oo])
    started = time.time()
    route = Route(spec, task, cache, args.model, args.device)
    if args.device.startswith("cuda"): torch.cuda.reset_peak_memory_stats()
    parity = {"status": "NOT_REQUIRED"}
    try:
        binding = accuracy_binding(spec,route,task,cache,Path(getattr(args,"public_root",None) or Path(__file__).resolve().parent))
        if route.family == "RAW":
            parity = raw_row_parity(route, physical, observed, lease=lease)
            if not parity["successful_variants_parity"]:
                raise AssertionError("Actual Raw row variants disagree or all failed")
        elif route.family == "LORA":
            if lease is not None: lease.check()
            parity = route.check_lora_parity(physical, observed)
        if args.device.startswith("cuda"): torch.cuda.empty_cache()
        rows = measure_balanced([route], physical, observed, warmups=args.warmups, timed_trials=args.timed_trials,
                                stage_trials=args.stage_trials, lease=lease)
        memory = route.memory_state()
        if route.pool.instances_created and route.family in {"DIRECT_G", "DIRECT_B2", "DIRECT_B4", "CURRENT", "SEASONAL", "RAW_RIDGE"}:
            raise AssertionError("Fresh direct-only process instantiated a foundation model")
        loading = {"route_loading_seconds": route.load_seconds, "foundation_loading_seconds": route.pool.load_seconds,
                   "foundation_instances_created": route.pool.instances_created}
    finally:
        route.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    trials = output.with_suffix(".trials.csv.gz")
    _write_trials(trials, rows)
    if trials.stat().st_size > 20 * 1024**2: raise ValueError("Public trial file exceeds 20MiB contract")
    receipt = {"status": "COMPLETE_PROTOCOL" if args.warmups >= 10 and args.timed_trials >= 20 and args.stage_trials >= 20 else "SMOKE_OR_PARTIAL_COST_PROTOCOL",
               "run_id": route.run_id, "dataset": spec["dataset"], "family": route.family, "batch_size": args.batch,
               "condition_id": route.run_id, "accuracy_run_id": binding.get("accuracy_run_id"),
               "accuracy_bindings": {route.run_id:binding},
               "scope": "fresh_process_single_route", "device": args.device, "dtype": "float64" if route.family == "RAW_RIDGE" else "float32",
               "physical_output_dtype": "float64", "inverse_transform_dtype": "float64",
               "data_sha256": task["data_sha256"], "scaler_sha": scaler_fingerprint(route.mean, route.std),
               "L": task["L"], "H": task["H"], "C": len(task["columns"]), "latent_k": route.latent_k,
               "row_microbatch": spec.get("row_microbatch", "full" if route.family == "RAW" else 128),
               "lora_rank": route.bundle["spec"].get("lora_rank") if route.family == "LORA" else None,
               "physical_input_sha256": hashlib.sha256(physical.tobytes()).hexdigest(), "observation_mask_sha256": hashlib.sha256(observed.tobytes()).hexdigest(),
               "GPU_device_name": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
               "torch_version": str(torch.__version__), "numpy_version": np.__version__,
               "F_snapshot_revision": Path(args.model).name if args.model else None,
               "origins": list(map(int, oo)), "repeated_histories_for_batch_cost_only": len(set(oo)) != len(oo),
               "warmups": args.warmups, "timed_trials": args.timed_trials, "stage_trials": args.stage_trials,
               "loading": loading, "memory": memory, "RSS_MiB_after_close": psutil.Process().memory_info().rss / 1024**2,
               "parity": parity, "summaries": _summaries_with_accuracy(rows,{route.run_id:binding}), "trials_file": trials.name, "trials_sha256": sha256_file(trials),
               "artifact_sha256": route.artifact_sha256, "source_sha256": {name: sha256_file(Path(__file__).with_name(name)) for name in ("deployment.py", "measure_costs.py")},
               "started_unix": started, "completed_unix": time.time(), "backbone_exact_restore_checked_at_close": loading["foundation_instances_created"] > 0,
               "backbone_close_status": "PASS_EXACT_HASH" if loading["foundation_instances_created"] else "NO_FOUNDATION_INSTANTIATED",
               "training_peak_included": False, "cached_forecast_reads": 0, "stage_probe_is_intrusive_separate_scope": True,
               "GPU_owner_child_lease_validated": lease is not None}
    _json_new(output, receipt)
    return receipt


def balanced_driver(args):
    cache = Path(args.cache_root)
    lease = worker_lease_check(cache, args.device)
    torch.set_num_threads(2)
    specs = json.loads(Path(args.route_spec).read_text(encoding="utf-8"))
    if not isinstance(specs, list) or not specs or len({s["dataset"] for s in specs}) != 1:
        raise ValueError("Balanced mode needs a nonempty JSON list from one task")
    if len({s["run_id"] for s in specs}) != len(specs):
        raise ValueError("Cost condition IDs must be unique, including Raw row variants")
    dataset = specs[0]["dataset"]
    task = load_task(dataset, cache_dir=cache)
    candidates = origins(task["split_ranges"]["V_select"], task["L"], task["H"], task["H"])
    if not len(candidates): raise ValueError("No registered V cost history")
    oo = np.asarray(specs[0].get("cost_origins", np.resize(candidates[:args.batch], args.batch)), np.int64)
    if oo.shape != (args.batch,) or any(o < task["L"] or o > task["split_ranges"]["V_select"][1] for o in oo):
        raise ValueError("Balanced histories must remain in pre-T V observations")
    for spec in specs:
        if "cost_origins" in spec and not np.array_equal(spec["cost_origins"], oo):
            raise ValueError("Balanced routes must use identical physical histories")
    physical = np.stack([task["physical"][o - task["L"]:o] for o in oo])
    observed = np.stack([task["observed"][o - task["L"]:o] for o in oo])
    pool, routes, parities, bindings = FoundationPool(args.model, args.device), [], {}, {}
    started = time.time()
    try:
        for spec in specs:
            if lease is not None: lease.check()
            routes.append(Route(spec, task, cache, args.model, args.device, pool))
            bindings[spec["run_id"]] = accuracy_binding(spec,routes[-1],task,cache,Path(getattr(args,"public_root",None) or Path(__file__).resolve().parent))
        raw_checked = False
        for route in routes:
            if route.family == "RAW" and not raw_checked:
                parities[route.run_id] = raw_row_parity(route, physical, observed, lease=lease)
                if not parities[route.run_id]["successful_variants_parity"]: raise AssertionError("Raw row parity failed")
                raw_checked = True
            elif route.family == "LORA":
                if lease is not None: lease.check()
                parities[route.run_id] = route.check_lora_parity(physical, observed)
                route.deactivate()
        if args.device.startswith("cuda"): torch.cuda.empty_cache()
        rows = measure_balanced(routes, physical, observed, warmups=args.warmups, timed_trials=args.timed_trials,
                                stage_trials=args.stage_trials, lease=lease)
        memory = {route.run_id: route.memory_state() for route in routes}
        loading = {"routes_seconds": {route.run_id: route.load_seconds for route in routes},
                   "foundation_loading_seconds": pool.load_seconds, "foundation_instances_created": pool.instances_created}
    finally:
        for route in reversed(routes): route.close()
        pool.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    trials = output.with_suffix(".trials.csv.gz")
    _write_trials(trials, rows)
    if trials.stat().st_size > 20 * 1024**2: raise ValueError("Public trial file exceeds 20MiB")
    result = {"status": "COMPLETE_PROTOCOL" if args.warmups >= 10 and args.timed_trials >= 20 and args.stage_trials >= 20 else "SMOKE_OR_PARTIAL_COST_PROTOCOL",
              "scope": "general_shared_F_balanced", "dataset": dataset, "run_ids": [route.run_id for route in routes],
              "accuracy_bindings": bindings,
              "batch_size": args.batch, "device": args.device, "dtype": "float32" if all(route.family != "RAW_RIDGE" for route in routes) else "mixed_native_CPU_FP64_and_FP32_do_not_claim_matched_dtype",
              "physical_output_dtype": "float64", "inverse_transform_dtype": "float64",
              "data_sha256": task["data_sha256"], "L": task["L"], "H": task["H"], "C": len(task["columns"]),
              "physical_input_sha256": hashlib.sha256(physical.tobytes()).hexdigest(), "observation_mask_sha256": hashlib.sha256(observed.tobytes()).hexdigest(),
              "GPU_device_name": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
              "torch_version": str(torch.__version__), "numpy_version": np.__version__, "F_snapshot_revision": Path(args.model).name if args.model else None,
              "origins": oo.tolist(), "warmups_per_route": args.warmups, "timed_trials_per_route": args.timed_trials,
              "stage_trials_per_route": args.stage_trials, "summaries": _summaries_with_accuracy(rows,bindings), "loading": loading,
              "memory": memory, "memory_scope": "Shared process contains all routes; route weights count own needed modules, process peak includes every resident route; use fresh workers for exclusive memory",
              "parity": parities, "trials_file": trials.name, "trials_sha256": sha256_file(trials),
              "source_sha256": {name: sha256_file(Path(__file__).with_name(name)) for name in ("deployment.py", "measure_costs.py")},
              "started_unix": started, "completed_unix": time.time(), "GPU_owner_child_lease_validated": lease is not None,
              "backbone_exact_restore_checked_at_close": loading["foundation_instances_created"] > 0,
              "backbone_close_status": "PASS_EXACT_HASH" if loading["foundation_instances_created"] else "NO_FOUNDATION_INSTANTIATED",
              "cached_forecast_reads": 0, "training_peak_included": False}
    _json_new(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--worker", action="store_true")
    mode.add_argument("--balanced", action="store_true")
    parser.add_argument("--route-spec", required=True)
    parser.add_argument("--public-root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--cache-root", default=os.environ.get("INFORMS_CACHE_ROOT"), required=os.environ.get("INFORMS_CACHE_ROOT") is None)
    parser.add_argument("--model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, choices=(1, 8), default=1)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--timed-trials", type=int, default=20)
    parser.add_argument("--stage-trials", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    receipt = balanced_driver(args) if args.balanced else worker(args)
    print(json.dumps({key: receipt[key] for key in ("status", "batch_size", "trials_sha256")}))


if __name__ == "__main__": main()
