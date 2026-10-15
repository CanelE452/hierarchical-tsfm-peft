"""Serial whole-route timing with one shared foundation model and fresh outputs.

The caller owns model/data construction and must stop training before using this
module. No dataset loader, checkpoint loader, or subprocess launcher lives here.
Stage timing is supplied by predict_physical(instrument=True) in the same trial.
"""

from collections import defaultdict
from datetime import datetime, timezone
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import psutil
import torch


STAGES = ("preprocessing", "PE", "F", "DG", "return")


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (Path, torch.device, torch.dtype)):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _json_dump(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, default=_json_default)


def _module(value):
    if isinstance(value, torch.nn.Module):
        return value
    if isinstance(getattr(value, "model", None), torch.nn.Module):
        return value.model
    return None


def _storages(value):
    module = _module(value)
    if module is None:
        return {}
    result = {}
    for tensor in (*module.parameters(), *module.buffers()):
        if tensor.numel():
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr())
            result[key] = storage.nbytes()
    return result


def _device(bolt, requested):
    if requested is not None:
        return torch.device(requested)
    module = _module(bolt)
    if module is not None:
        for tensor in (*module.parameters(), *module.buffers()):
            return tensor.device
    return torch.device(getattr(bolt, "device", "cpu"))


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory(device):
    if device.type != "cuda":
        return {"allocated_bytes": None, "reserved_bytes": None}
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }


def _active_bytes(route, resident_storages):
    declared = getattr(route, "active_parameter_bytes", None)
    if declared is not None:
        return int(declared), "route_declared_active_parameter_bytes"
    storages = _storages(route)
    if storages:
        return sum(size for key, size in storages.items() if key not in resident_storages), "unique_parameter_and_buffer_storage_excluding_F"
    return None, "unavailable; route must provide an nn.Module/model or active_parameter_bytes"


def _input(input_provider, dataset, batch):
    value = input_provider(dataset, batch)
    if isinstance(value, dict):
        physical, observed, mean, std = (value[key] for key in ("physical", "observed", "mean", "std"))
    else:
        physical, observed, mean, std = value
    physical = np.asarray(physical)
    observed = np.asarray(observed)
    mean = np.asarray(mean)
    std = np.asarray(std)
    if physical.ndim != 3 or physical.shape[0] != batch:
        raise ValueError("physical must have shape [requested batch, past length, channels]")
    if observed.shape != physical.shape or observed.dtype != np.bool_:
        raise ValueError("observed must be bool with the same shape as physical")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or not (std > 0).all():
        raise ValueError("mean/std must be finite and std positive")
    return physical, observed, mean, std


def _stages(route):
    supplied = getattr(route, "last_stages", {})
    values = {}
    for stage in STAGES:
        value = supplied.get(f"{stage}_seconds", supplied.get(stage))
        if value is None:
            values[f"{stage}_seconds"] = None
        else:
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid caller-supplied {stage} timer: {value}")
            values[f"{stage}_seconds"] = value
    return values


def _statistics(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return {"count": 0, "median": None, "mean": None, "population_variance": None, "min": None, "max": None}
    return {
        "count": len(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "population_variance": statistics.pvariance(values),
        "min": min(values),
        "max": max(values),
    }


def _trial(row, route, inputs, device, phase, round_index, position, batch, resident_storages, process):
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    before = _memory(device)
    rss_before = int(process.memory_info().rss)
    active, active_scope = _active_bytes(route, resident_storages)
    result = {
        "route_id": row["_route_id"],
        "dataset": row["dataset"],
        "method": row.get("method", row.get("variant", row["_route_id"])),
        "batch": batch,
        "phase": phase,
        "round": round_index,
        "position": position,
        "status": "VALID",
        "device": str(device),
        "whole_seconds": None,
        "F_resident_parameter_buffer_bytes": sum(resident_storages.values()),
        "active_parameter_buffer_bytes": active,
        "active_bytes_scope": active_scope,
        "allocated_before_bytes": before["allocated_bytes"],
        "reserved_before_bytes": before["reserved_bytes"],
        "rss_before_bytes": rss_before,
        "error_type": "",
        "error_message": "",
        "route_metadata_json": _json_dump({key: value for key, value in row.items() if key != "_route_id"}),
    }
    start = time.perf_counter()
    try:
        with torch.inference_mode():
            prediction = route.predict_physical(*inputs, instrument=True)
        _sync(device)
        result["whole_seconds"] = time.perf_counter() - start
        if not isinstance(prediction, np.ndarray) or prediction.ndim != 3:
            raise TypeError("predict_physical must return physical CPU np.ndarray[B,H,C]")
        if prediction.shape[0] != batch or prediction.shape[2] != inputs[0].shape[2]:
            raise ValueError("Prediction batch/channels differ from physical inputs")
        expected_horizon = row.get("horizon")
        if expected_horizon is not None and prediction.shape[1] != int(expected_horizon):
            raise ValueError("Prediction horizon differs from declared route horizon")
        if not np.isfinite(prediction).all():
            raise FloatingPointError("Prediction contains nonfinite values")
        result.update(_stages(route))
        result["output_dtype"] = str(prediction.dtype)
        result["output_shape_json"] = _json_dump(list(prediction.shape))
        complete = all(result[f"{stage}_seconds"] is not None for stage in STAGES)
        result["stage_coverage_complete"] = complete
        result["stage_sum_seconds"] = sum(result[f"{stage}_seconds"] or 0 for stage in STAGES)
        result["whole_minus_stage_sum_seconds"] = result["whole_seconds"] - result["stage_sum_seconds"]
        result["stage_sum_exceeds_whole"] = result["whole_minus_stage_sum_seconds"] < -max(0.001, result["whole_seconds"] * 0.05)
    except Exception as error:
        _sync(device)
        result["whole_seconds"] = time.perf_counter() - start
        result.update(status="ERROR", error_type=type(error).__name__, error_message=str(error))
    after = _memory(device)
    memory_info = process.memory_info()
    result.update(
        allocated_after_bytes=after["allocated_bytes"],
        reserved_after_bytes=after["reserved_bytes"],
        rss_after_bytes=int(memory_info.rss),
        process_lifetime_peak_rss_bytes=int(memory_info.peak_wset) if hasattr(memory_info, "peak_wset") else None,
    )
    if device.type == "cuda":
        peak = int(torch.cuda.max_memory_allocated(device))
        result.update(
            peak_allocated_bytes=peak,
            peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
            peak_allocated_minus_F_resident_bytes=peak - sum(resident_storages.values()),
            peak_allocated_minus_pretrial_allocated_bytes=peak - before["allocated_bytes"],
        )
    else:
        result.update(peak_allocated_bytes=None, peak_reserved_bytes=None,
                      peak_allocated_minus_F_resident_bytes=None,
                      peak_allocated_minus_pretrial_allocated_bytes=None)
    return result


def measure_routes(routes, factory, input_provider, bolt, output_folder,
                   batch_sizes=(1, 8), warmup=5, repeats=10, exclusive=False, device=None):
    """Measure serial routes; input_provider returns physical, observed, mean, std.

    A fresh output folder is mandatory. Each batch size gets five warmups/route,
    then enough rotated complete rounds for every candidate to occupy every order
    position equally. Loading/unloading and input construction are outside timers.
    Set exclusive=True only when the caller created a separate fresh process for
    this one route. This function does not launch or inspect training processes.
    """
    if not routes:
        raise ValueError("At least one route is required")
    if warmup < 0 or repeats < 1 or any(int(batch) < 1 for batch in batch_sizes):
        raise ValueError("Invalid warmup/repeats/batch size")
    if exclusive and len(routes) != 1:
        raise ValueError("Exclusive measurements require one route per process")
    output = Path(output_folder)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, source in enumerate(routes):
        row = dict(source)
        if "dataset" not in row:
            raise ValueError("Each route metadata row requires dataset")
        row["_route_id"] = str(row.get("route_id", row.get("id", f"route_{index:03d}")))
        rows.append(row)
    if len({row["_route_id"] for row in rows}) != len(rows):
        raise ValueError("route_id must be unique within a measurement")
    selected_device = _device(bolt, device)
    resident = _storages(bolt)
    process = psutil.Process()
    rounds = int(math.ceil(repeats / len(rows)) * len(rows))
    inputs = {(row["dataset"], int(batch)): None for row in rows for batch in batch_sizes}
    for key in inputs:
        inputs[key] = _input(input_provider, *key)
    fields = [
        "route_id", "dataset", "method", "batch", "phase", "round", "position", "status", "device", "whole_seconds",
        *(f"{stage}_seconds" for stage in STAGES), "stage_coverage_complete", "stage_sum_seconds", "whole_minus_stage_sum_seconds", "stage_sum_exceeds_whole",
        "F_resident_parameter_buffer_bytes", "active_parameter_buffer_bytes", "active_bytes_scope", "allocated_before_bytes", "reserved_before_bytes",
        "allocated_after_bytes", "reserved_after_bytes", "peak_allocated_bytes", "peak_reserved_bytes", "peak_allocated_minus_F_resident_bytes",
        "peak_allocated_minus_pretrial_allocated_bytes", "rss_before_bytes", "rss_after_bytes", "process_lifetime_peak_rss_bytes",
        "output_dtype", "output_shape_json", "error_type", "error_message", "route_metadata_json",
    ]
    all_trials = []
    started = datetime.now(timezone.utc).isoformat()
    with (output / "RAW_COST_TRIALS.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        def execute(row, batch, phase, round_index, position, repetitions=1):
            route = None
            try:
                route = factory({key: value for key, value in row.items() if key != "_route_id"})
                for repetition in range(repetitions):
                    trial = _trial(row, route, inputs[(row["dataset"], batch)], selected_device,
                                   phase, round_index if repetitions == 1 else repetition,
                                   position, batch, resident, process)
                    writer.writerow(trial)
                    handle.flush()
                    all_trials.append(trial)
            except Exception as error:
                trial = {"route_id": row["_route_id"], "dataset": row["dataset"], "method": row.get("method", row["_route_id"]),
                         "batch": batch, "phase": phase, "round": round_index, "position": position,
                         "status": "ERROR", "device": str(selected_device), "error_type": type(error).__name__, "error_message": str(error),
                         "route_metadata_json": _json_dump({key: value for key, value in row.items() if key != "_route_id"})}
                writer.writerow(trial)
                handle.flush()
                all_trials.append(trial)
            finally:
                try:
                    close = getattr(route, "close", None)
                    if route is not None and callable(close):
                        close()
                except Exception as error:
                    cleanup_trial = {"route_id": row["_route_id"], "dataset": row["dataset"], "method": row.get("method", row["_route_id"]),
                                     "batch": batch, "phase": "cleanup", "round": round_index, "position": position,
                                     "status": "ERROR", "device": str(selected_device), "error_type": type(error).__name__,
                                     "error_message": f"route.close failed: {error}",
                                     "route_metadata_json": _json_dump({key: value for key, value in row.items() if key != "_route_id"})}
                    writer.writerow(cleanup_trial)
                    handle.flush()
                    all_trials.append(cleanup_trial)
                close = None
                del route
                gc.collect()
                if selected_device.type == "cuda":
                    torch.cuda.empty_cache()

        for batch in map(int, batch_sizes):
            if warmup:
                for position, row in enumerate(rows):
                    execute(row, batch, "warmup", -1, position, repetitions=warmup)
            for round_index in range(rounds):
                offset = round_index % len(rows)
                ordered = rows[offset:] + rows[:offset]
                for position, row in enumerate(ordered):
                    execute(row, batch, "measure", round_index, position)

    grouped = defaultdict(list)
    for trial in all_trials:
        if trial["phase"] == "measure":
            grouped[(trial["route_id"], trial["batch"])].append(trial)
    summaries = []
    statistic_fields = ["whole_seconds", *(f"{stage}_seconds" for stage in STAGES), "stage_sum_seconds", "whole_minus_stage_sum_seconds",
                        "peak_allocated_bytes", "peak_reserved_bytes", "peak_allocated_minus_F_resident_bytes",
                        "peak_allocated_minus_pretrial_allocated_bytes", "rss_before_bytes", "rss_after_bytes", "process_lifetime_peak_rss_bytes"]
    for (route_id, batch), trials in grouped.items():
        valid = [trial for trial in trials if trial["status"] == "VALID"]
        summaries.append({
            "route_id": route_id, "dataset": trials[0]["dataset"], "method": trials[0]["method"], "batch": batch,
            "status": "COMPLETE_VALID" if len(valid) == rounds else "INVALID_OR_INCOMPLETE",
            "measured_trials": len(trials), "valid_trials": len(valid), "errors": len(trials) - len(valid),
            "exclusive_process_claimed_by_caller": bool(exclusive),
            "F_resident_parameter_buffer_bytes": sum(resident.values()),
            "active_parameter_buffer_bytes": valid[0].get("active_parameter_buffer_bytes") if valid else None,
            "all_stage_timers_complete": bool(valid) and all(trial["stage_coverage_complete"] for trial in valid),
            "stage_sum_exceeds_whole_trials": sum(bool(trial.get("stage_sum_exceeds_whole")) for trial in valid),
            "statistics": {field: _statistics([trial.get(field) for trial in valid]) for field in statistic_fields},
        })
    metadata = {
        "status": "COMPLETE_VALID" if all(row["status"] == "COMPLETE_VALID" for row in summaries) and not any(trial["status"] == "ERROR" for trial in all_trials) else "INVALID_OR_INCOMPLETE",
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(), "pid": process.pid,
        "python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__, "device": str(selected_device),
        "cuda_runtime": torch.version.cuda, "torch_cpu_threads": torch.get_num_threads(),
        "gpu_name": torch.cuda.get_device_name(selected_device) if selected_device.type == "cuda" else None,
        "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(selected_device).total_memory) if selected_device.type == "cuda" else None,
        "warmups_per_route_batch": warmup, "requested_measured_repeats": repeats,
        "measured_rounds": rounds, "candidate_count": len(rows), "batch_sizes": list(map(int, batch_sizes)),
        "order": "round r rotates candidate list left by r modulo q; rounds rounded up to a multiple of q",
        "active_candidates": 1, "F_instances_supplied_by_caller": 1 if sum(resident.values()) > 0 else 0,
        "factory_load_unload_in_whole_timer": False, "physical_preprocessing_CPU_return_in_whole_timer": True,
        "stage_timers": "caller last_stages, same instrumented trial; missing stages stay null",
        "exclusive_process_claimed_by_caller": bool(exclusive),
        "exclusive_process_not_verified_by_this_module": True,
        "training_stopped_and_no_competing_GPU_work": "caller responsibility; this module does not terminate or inspect jobs",
        "memory_interpretation": "allocated-minus-resident is an accounting difference including active parameters, buffers and temporary allocations; not a causal activation-only measurement",
        "rss_interpretation": "before/after resident set; Windows peak_wset is process-lifetime high-water mark, not a resettable per-trial peak",
        "F_resident_parameter_buffer_bytes": sum(resident.values()), "routes": [{key: value for key, value in row.items() if key != "_route_id"} for row in rows],
        "input_contracts": [{"dataset": key[0], "batch": key[1], "physical_shape": list(value[0].shape),
                             "physical_dtype": str(value[0].dtype), "past_observed_values": int(value[1].sum()),
                             "physical_observed_mean_std_sha256": [hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest() for array in value]}
                            for key, value in inputs.items()],
        "raw_trial_count_including_warmups_errors": len(all_trials), "error_count": sum(trial["status"] == "ERROR" for trial in all_trials),
        "cleanup_failure_count": sum(trial["phase"] == "cleanup" and trial["status"] == "ERROR" for trial in all_trials),
    }
    for filename, payload in (("COST_SUMMARY.json", summaries), ("COST_RUN_METADATA.json", metadata)):
        with (output / filename).open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False, default=_json_default)
            handle.write("\n")
    return {"output_folder": str(output), "metadata": metadata, "summary": summaries}


def measure_one(row, factory, input_provider, bolt, output_folder, *, exclusive=True, **kwargs):
    """Entry for ROOT's separate-process route CLI; no subprocess is spawned here."""
    return measure_routes([row], factory, input_provider, bolt, output_folder, exclusive=exclusive, **kwargs)
