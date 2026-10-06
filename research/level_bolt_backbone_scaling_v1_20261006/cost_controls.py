"""Same-campaign Bolt allocation inference costs; no fitting or TEST inputs."""

import argparse
from collections import defaultdict
import csv
import gc
import hashlib
import json
import os
import subprocess
import time
import traceback

import numpy as np

from runtime import (CACHE, DATASETS, HERE, SCOPE, Job, artifact, check_seal,
                     configure, digest, environment, read_json, save_json)


PASSES, WARMUP, BLOCKS, N_ORIGINS = 10, 10, 3, 24
ARMS = ("BOLT_SMALL_LEVEL", "BOLT_SMALL_F0", "BOLT_MINI_F0", "BOLT_TINY_F0")
MODEL_ARMS = {"BOLT_MINI_F0": "MINI", "BOLT_TINY_F0": "TINY"}
EXPECTED_INSTANCES_PER_DATASET = 5
EXPECTED_CUDA_ROWS = 108


def require_static_apis():
    import baselines
    import bolt_models
    import runtime
    required_runtime = ("HERE", "CACHE", "DATASETS", "SCOPE", "Job", "artifact",
                        "configure", "environment", "read_json", "save_json",
                        "digest", "check_seal", "SEEDS")
    missing = [name for name in required_runtime if not hasattr(runtime, name)]
    missing += [name for name in ("baseline_rows", "inputs", "load_baseline", "load_data")
                if not hasattr(baselines, name)]
    wrapper = getattr(bolt_models, "BoltForecast", None)
    if wrapper is None or not all(hasattr(wrapper, name) for name in ("__call__", "receipt", "module")):
        missing.append("bolt_models.BoltForecast callable/module/receipt")
    if missing:
        raise RuntimeError("Missing required isolated campaign APIs: " + ", ".join(missing))


def root_path(path):
    path = os.fspath(path)
    p = os.path.abspath(path)
    if os.path.isabs(path):
        return path
    return os.path.join(HERE.parents[1], path)


def cleanup(device):
    import torch
    gc.collect()
    if device == "cuda":
        torch.cuda.synchronize()
        if hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
            torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def process_memory():
    try:
        import psutil
        info = psutil.Process().memory_info()
        return {name: int(getattr(info, name)) for name in ("rss", "vms", "peak_wset")
                if hasattr(info, name)}
    except (ImportError, OSError):
        return None


def telemetry():
    fields = "index,uuid,temperature.gpu,clocks.current.sm,utilization.gpu,power.draw,pstate"
    result = {}
    for name, query in (("gpu", ["--query-gpu=" + fields, "--format=csv,noheader,nounits"]),
                        ("processes", ["--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader"])):
        try:
            run = subprocess.run(["nvidia-smi", *query], capture_output=True,
                                 text=True, timeout=15)
            result[name] = {"status": run.returncode, "rows": run.stdout.strip().splitlines(),
                            "error": run.stderr.strip() or None}
        except (OSError, subprocess.TimeoutExpired) as exc:
            result[name] = {"status": None, "rows": None, "error": repr(exc)}
    return {"utc": time.time(), "fields": fields, **result,
            "unavailable_values": "N/A is unavailable, never zero"}


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def require_preflight_pass():
    path = HERE / "preflight_checks.json"
    if not path.exists():
        raise FileNotFoundError("Preflight checks must pass before cost measurement")
    preflight = read_json(path)
    if preflight.get("status") != "pass":
        raise RuntimeError("Preflight status is not pass")
    if preflight.get("test_scoring_performed") not in (False, None):
        raise RuntimeError("Preflight must not score TEST data")
    if preflight.get("fits", 0) != 0 or preflight.get("optimizer_updates", 0) != 0:
        raise RuntimeError("Preflight must not fit or update models")
    return preflight


def prepare(dataset):
    import torch
    from baselines import load_data
    data = load_data(dataset, include_test=False)
    all_origins = np.asarray(data["val_origins"], dtype=np.int64)
    if len(all_origins) < N_ORIGINS or np.any(np.diff(all_origins) <= 0):
        raise ValueError("Cost requires the first 24 chronological VAL origins")
    origins = all_origins[:N_ORIGINS]
    inputs = torch.from_numpy(np.stack([data["x"][o - 512:o] for o in origins]).astype(np.float32))
    if tuple(inputs.shape) != (N_ORIGINS, 512, data["_C"]):
        raise ValueError("Bound cost input axes changed")
    return {"inputs": inputs, "origins": origins, "C": int(data["_C"]), "K": int(data["_K"]),
            "input_sha256": array_hash(inputs.numpy()), "origin_sha256": array_hash(origins),
            "trainval": artifact(data["_path"])}


def configs(method, device):
    if device != "cuda":
        return [{"batch": b, "execution": "full", "chunk_rows": None} for b in (1, 4)]
    if method not in ARMS:
        raise ValueError("Unapproved cost method: " + method)
    return [{"batch": b, "execution": "full", "chunk_rows": None} for b in (1, 4)]


def sentinel_config():
    return {"batch": 1, "execution": "full", "chunk_rows": None}


def rotated(values, block):
    values = list(values)
    if block == 1:
        return list(values[1:]) + values[:1]
    if block == 2:
        return list(reversed(values))
    return values


def instances_by_dataset():
    from baselines import baseline_rows
    result = {}
    for dataset in DATASETS:
        selected = []
        for row in baseline_rows(dataset):
            if row.get("arm") not in ("BOLT_SMALL_LEVEL", "BOLT_SMALL_F0"):
                raise ValueError("Baseline reuse row lacks a fixed Small arm mapping")
            item = dict(row)
            item["original_method"] = item["method"]
            item["method"] = item["arm"]
            item["source_kind"] = "baseline"
            selected.append(item)
        counts = {method: sum(row["method"] == method for row in selected)
                  for method in ("BOLT_SMALL_LEVEL", "BOLT_SMALL_F0")}
        if counts != {"BOLT_SMALL_LEVEL": 2, "BOLT_SMALL_F0": 1}:
            raise ValueError("The prescribed selected Small baseline instances are incomplete")
        for method, key in MODEL_ARMS.items():
            selected.append({"id": dataset + "_" + method, "dataset": dataset,
                             "method": method, "original_method": "F0", "seed": None,
                             "model_key": key, "source_kind": "bolt"})
        if len(selected) != EXPECTED_INSTANCES_PER_DATASET:
            raise ValueError("Every dataset must expose two LEVEL seeds, Small F0, Mini F0, Tiny F0")
        result[dataset] = selected
    return result


def planned_grid(instances, prepared, device):
    rows = []
    for block in range(BLOCKS):
        for dataset in rotated(list(DATASETS), block):
            items = rotated(instances[dataset], block)
            f0 = next(row for row in instances[dataset] if row["method"] == "BOLT_SMALL_F0")
            ordered = []
            if device == "cuda":
                ordered.append((f0, "pre", [sentinel_config()]))
            ordered.extend((row, None, configs(row["method"], device)) for row in items)
            if device == "cuda":
                ordered.append((f0, "post", [sentinel_config()]))
            for item, sentinel, grid in ordered:
                for config in grid:
                    row = {name: item.get(name) for name in ("id", "dataset", "method",
                                                             "original_method", "seed", "source_kind", "model_key")}
                    row.update(device=device, block=block, sentinel=sentinel, **config)
                    row["key"] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:24]
                    row["planned_sequence"] = len(rows)
                    rows.append(row)
    expected = EXPECTED_CUDA_ROWS if device == "cuda" else BLOCKS * len(DATASETS) * EXPECTED_INSTANCES_PER_DATASET * 2
    if len(rows) != expected or len({row["key"] for row in rows}) != expected:
        raise ValueError("The prespecified cost grid changed")
    return rows


def planned_grid_current_limits(device="cuda"):
    prepared = {dataset: prepare(dataset) for dataset in DATASETS}
    instances = instances_by_dataset()
    rows = planned_grid(instances, prepared, device)
    return {"device": device, "expected_rows": len(rows), "counts": call_counts(rows),
            "input_hashes": {dataset: prepared[dataset]["input_sha256"] for dataset in DATASETS},
            "origin_hashes": {dataset: prepared[dataset]["origin_sha256"] for dataset in DATASETS},
            "limits": read_json(HERE / "ledger.json").get("limits")}


def call_counts(rows):
    values = []
    for row in rows:
        batch = row["batch"]
        timed = PASSES * (N_ORIGINS // batch)
        values.append({"key": row["key"], "dataset": row["dataset"], "method": row["method"],
                       "id": row["id"], "batch": batch, "sentinel": row["sentinel"],
                       "warmup_wrapper_calls": WARMUP, "timed_wrapper_calls": timed,
                       "total_wrapper_calls": timed + WARMUP,
                       "timed_origin_units": PASSES * N_ORIGINS,
                       "warmup_origin_units": WARMUP * batch})
    return {"rows": values, "measurement_rows": len(rows),
            "sentinel_rows": sum(row["sentinel"] is not None for row in rows),
            "total_wrapper_calls": sum(row["total_wrapper_calls"] for row in values),
            "timed_origin_units": sum(row["timed_origin_units"] for row in values),
            "warmup_origin_units": sum(row["warmup_origin_units"] for row in values),
            "not_included": ["preflight", "model_load", "prediction/scoring", "hashing/logging"],
            "interpretation": "Wrapper calls and origin units are not independent statistical samples or model.forward calls"}


def state_bytes(module):
    parameters = sum(p.numel() * p.element_size() for p in module.parameters())
    buffers = sum(p.numel() * p.element_size() for p in module.buffers())
    return {"deployed_parameters": sum(p.numel() for p in module.parameters()),
            "parameter_bytes": parameters, "buffer_bytes": buffers,
            "deployment_tensor_bytes": parameters + buffers,
            "inference_trainable_parameters": sum(p.numel() for p in module.parameters() if p.requires_grad)}


def module_state_snapshot(wrapper):
    from bolt_models import state_hash
    module = wrapper.module
    return {"sha256": state_hash(module),
            "eval_mode": not module.training,
            "trainable_parameters": sum(p.numel() for p in module.parameters() if p.requires_grad),
            "optimizer_present": False,
            "hash_scope": "parameters_and_buffers"}


def require_frozen_state(snapshot, label):
    if not snapshot["eval_mode"] or snapshot["trainable_parameters"] != 0 or snapshot["optimizer_present"]:
        raise RuntimeError("STOP_DEBUG: inference state is not frozen at " + label)


def state_preservation(before, after):
    result = {"before_sha256": before["sha256"],
              "after_sha256": after["sha256"],
              "unchanged": before["sha256"] == after["sha256"],
              "eval_mode": after["eval_mode"],
              "trainable_parameters": after["trainable_parameters"],
              "optimizer_present": False,
              "hash_scope": "parameters_and_buffers"}
    if (not result["unchanged"] or not result["eval_mode"]
            or result["trainable_parameters"] != 0 or result["optimizer_present"]):
        raise RuntimeError("STOP_DEBUG: inference state changed during measured cost row")
    return result


def model_info_for_key(key):
    models = read_json(HERE / "model_manifest.json")["models"]
    info = models[key]
    if info["id"] != ("amazon/chronos-bolt-mini" if key == "MINI" else "amazon/chronos-bolt-tiny"):
        raise ValueError("Model key does not match the official Mini/Tiny binding")
    for receipt in info.get("files", {}).values():
        artifact(receipt["path"], receipt["sha256"])
    return info


def load_deployment(item, device):
    cleanup(device)
    if item["source_kind"] == "bolt":
        from bolt_models import BoltForecast
        model_info = model_info_for_key(item["model_key"])
        checkpoint = model_info.get("checkpoint_dir", model_info.get("snapshot"))
        wrapper = BoltForecast(root_path(checkpoint), device=device)
        receipt = wrapper.receipt(include_file_hashes=False, include_state_hash=False)
        info = {"id": item["id"], "method": item["method"], "dataset": item["dataset"],
                "seed": item["seed"], "model_key": item["model_key"],
                "model_id": model_info["id"], "revision": model_info["revision"],
                "checkpoint_dir": checkpoint, **state_bytes(wrapper.module),
                "original_fitted_parameter_count": 0,
                "inference_trainable_parameter_count": 0,
                "deployment_receipt": receipt}
    else:
        from baselines import load_baseline
        baseline_item = dict(item)
        baseline_item["method"] = item["original_method"]
        wrapper = load_baseline(baseline_item, device=device, deploy=True)
        receipt = wrapper.receipt() if callable(wrapper.receipt) else wrapper.receipt
        info = {"id": item["id"], "method": item["method"], "original_method": item["original_method"],
                "dataset": item["dataset"], "seed": item["seed"], **state_bytes(wrapper.module),
                "original_fitted_parameter_count": receipt.get("original_fitted_parameter_count"),
                "inference_trainable_parameter_count": receipt.get("inference_trainable_parameter_count"),
                "deployment_receipt": receipt}
    wrapper.module.eval()
    wrapper.module.requires_grad_(False)
    cleanup(device)
    loaded_state = module_state_snapshot(wrapper)
    require_frozen_state(loaded_state, "model_load")
    info["loaded_state"] = loaded_state
    return wrapper, info


def forward_cpu_output(wrapper, cpu_input):
    import torch
    result = wrapper(cpu_input)
    if not isinstance(result, torch.Tensor):
        result = torch.from_numpy(np.asarray(result))
    result = result.float().cpu().contiguous()
    if tuple(result.shape) != (len(cpu_input), 48, cpu_input.shape[2]):
        raise ValueError("Cost output must contain every original channel and H48")
    return result


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    return {"median": float(np.median(values)), "q25": float(np.quantile(values, .25)),
            "q75": float(np.quantile(values, .75)), "minimum": float(values.min()),
            "maximum": float(values.max())}


def measure(wrapper, prepared, row, device, job):
    import torch
    groups = list(prepared["inputs"].split(row["batch"]))
    if hasattr(wrapper, "set_chunk"):
        wrapper.set_chunk(row["chunk_rows"])
    elif row["chunk_rows"] is not None:
        raise ValueError("Native Bolt F0 cannot be channel-chunked")
    cleanup(device)
    resident = {"allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved()} if device == "cuda" else None
    for index in range(WARMUP):
        job.check_limits()
        forward_cpu_output(wrapper, groups[index % len(groups)])
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    times = []
    for repeat in range(PASSES):
        job.heartbeat({"phase": "cost_pass", "key": row["key"], "repeat": repeat})
        if device == "cuda":
            torch.cuda.synchronize()
        tick = time.perf_counter()
        for group in groups:
            forward_cpu_output(wrapper, group)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - tick
        times.append(elapsed)
        job.heartbeat({"phase": "cost_pass_complete", "key": row["key"], "repeat": repeat})
    memory = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "final_allocated_bytes": torch.cuda.memory_allocated()} if device == "cuda" else {
                  "peak_allocated_bytes": None, "peak_reserved_bytes": None,
                  "final_allocated_bytes": None}
    ms_per_origin = [value * 1000 / N_ORIGINS for value in times]
    origins_per_second = [N_ORIGINS / value for value in times]
    return {"seconds_per_24_origins": times,
            "raw_milliseconds_per_origin": ms_per_origin,
            "raw_origins_per_second": origins_per_second,
            "milliseconds_per_origin": describe(ms_per_origin),
            "origins_per_second": describe(origins_per_second),
            "resident_after_model_load_and_cleanup": resident,
            "process_cpu_memory": process_memory(),
            "warmup_wrapper_calls": WARMUP,
            "timed_wrapper_calls": PASSES * len(groups),
            "passes": PASSES, **memory}


def binding(prepared):
    check_seal()
    require_preflight_pass()
    names = ("input_contract.json", "model_manifest.json", "preflight_checks.json",
             "evaluation_seal.json", "runtime.py", "baselines.py", "bolt_models.py",
             "cost_controls.py")
    return {"files": {name: digest(HERE / name) for name in names},
            "inputs": {dataset: {key: value[key] for key in ("C", "K", "origin_sha256",
                                                             "input_sha256", "trainval")}
                       | {"origins": value["origins"].tolist()} for dataset, value in prepared.items()},
            "precision": "float32", "tf32": False, "cpu_threads": 4,
            "passes": PASSES, "warmup_calls": WARMUP, "blocks": BLOCKS,
            "timing_scope": "Standardized CPU input through full online inference to all original-channel H48 contiguous CPU output",
            "excluded_from_latency": ["model_load", "disk", "hashing", "scoring", "ledger", "telemetry"],
            "test_used_for_cost_selection": False, "additional_fitting": 0,
            "scope": SCOPE}


def persist_grid(rows, bound, device):
    path = HERE / "cost_binding.json"
    if path.exists() and read_json(path) != bound:
        raise ValueError("Cost sources changed; prior rows must remain intact")
    if not path.exists():
        save_json(path, bound)
    value = {"binding_sha256": digest(path), "device": device,
             "rows": rows, "counts": call_counts(rows)}
    grid_path = HERE / f"cost_{device}_grid.json"
    if grid_path.exists() and read_json(grid_path) != value:
        raise ValueError("The fixed cost grid changed")
    if not grid_path.exists():
        save_json(grid_path, value)
    return value["binding_sha256"]


def percentile_or_none(values, q):
    values = [v for v in values if v is not None]
    return float(np.quantile(values, q)) if values else None


def median_or_none(values):
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def summarize(rows, device):
    grouped = defaultdict(list)
    for row in rows:
        if row["sentinel"] is None:
            grouped[(row["dataset"], row["id"], row["method"], row["seed"],
                     row["batch"], row["execution"], row["chunk_rows"])].append(row)
    instances = []
    for group in grouped.values():
        if len(group) != BLOCKS or {row["block"] for row in group} != set(range(BLOCKS)):
            raise ValueError("Every reported configuration requires all three block records")
        first = group[0]
        complete = [row for row in group if row["status"] == "complete"]
        oom = [row for row in group if row["status"] == "oom"]
        ms = [row["milliseconds_per_origin"]["median"] if row["status"] == "complete" else None
              for row in sorted(group, key=lambda r: r["block"])]
        speed = [row["origins_per_second"]["median"] if row["status"] == "complete" else None
                 for row in sorted(group, key=lambda r: r["block"])]
        allocated = [row.get("peak_allocated_bytes") if row["status"] == "complete" else None
                     for row in sorted(group, key=lambda r: r["block"])]
        reserved = [row.get("peak_reserved_bytes") if row["status"] == "complete" else None
                    for row in sorted(group, key=lambda r: r["block"])]
        model_info = next((row.get("model") for row in group if row.get("model")), first.get("model"))
        resident_alloc = [row.get("resident_after_model_load_and_cleanup", {}).get("allocated_bytes")
                          if row.get("resident_after_model_load_and_cleanup") else None for row in group]
        resident_reserved = [row.get("resident_after_model_load_and_cleanup", {}).get("reserved_bytes")
                             if row.get("resident_after_model_load_and_cleanup") else None for row in group]
        status = "complete" if len(complete) == BLOCKS else "oom" if oom else "partial"
        instances.append({name: first[name] for name in ("dataset", "id", "method", "original_method",
                                                         "seed", "batch", "execution", "chunk_rows")}
            | {"device": device, "status": status,
               "complete_blocks": len(complete), "oom_blocks": len(oom),
               "median_block_median_ms": median_or_none(ms),
               "median_block_median_origins_s": median_or_none(speed),
               "block_ms": ms, "block_origins_s": speed,
               "block_median_ms_range": [min(v for v in ms if v is not None),
                                         max(v for v in ms if v is not None)] if any(v is not None for v in ms) else None,
               "block_median_origins_s_range": [min(v for v in speed if v is not None),
                                                max(v for v in speed if v is not None)] if any(v is not None for v in speed) else None,
               "peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved,
               "peak_allocated_range_bytes": [min(v for v in allocated if v is not None),
                                              max(v for v in allocated if v is not None)] if any(v is not None for v in allocated) else None,
               "peak_reserved_range_bytes": [min(v for v in reserved if v is not None),
                                             max(v for v in reserved if v is not None)] if any(v is not None for v in reserved) else None,
               "resident_allocated_bytes_range": [min(v for v in resident_alloc if v is not None),
                                                  max(v for v in resident_alloc if v is not None)] if any(v is not None for v in resident_alloc) else None,
               "resident_reserved_bytes_range": [min(v for v in resident_reserved if v is not None),
                                                 max(v for v in resident_reserved if v is not None)] if any(v is not None for v in resident_reserved) else None,
               "model": model_info})
    by_method = defaultdict(list)
    for row in instances:
        by_method[(row["dataset"], row["method"], row["batch"], row["execution"], row["chunk_rows"])].append(row)
    groups = []
    for key, items in by_method.items():
        expected = 2 if key[1] == "BOLT_SMALL_LEVEL" else 1
        if len(items) != expected:
            raise ValueError("A prescribed selected seed/model is missing from cost: " + str(key))
        complete_items = [row for row in items if row["status"] == "complete"]
        allocated = [v for row in items for v in row["peak_allocated_bytes"] if v is not None]
        reserved = [v for row in items for v in row["peak_reserved_bytes"] if v is not None]
        deployed = [row["model"]["deployed_parameters"] for row in items if row.get("model")]
        parameter_bytes = [row["model"]["parameter_bytes"] for row in items if row.get("model")]
        buffer_bytes = [row["model"]["buffer_bytes"] for row in items if row.get("model")]
        tensor_bytes = [row["model"]["deployment_tensor_bytes"] for row in items if row.get("model")]
        fitted = [row["model"].get("original_fitted_parameter_count") for row in items
                  if row.get("model") and row["model"].get("original_fitted_parameter_count") is not None]
        trainable = [row["model"].get("inference_trainable_parameter_count") for row in items
                     if row.get("model") and row["model"].get("inference_trainable_parameter_count") is not None]
        resident_alloc = [v for row in items for v in (row.get("resident_allocated_bytes_range") or [])]
        resident_reserved = [v for row in items for v in (row.get("resident_reserved_bytes_range") or [])]
        groups.append({"dataset": key[0], "method": key[1], "batch": key[2],
            "execution": key[3], "chunk_rows": key[4], "device": device,
            "ids": [row["id"] for row in items],
            "status": "complete" if len(complete_items) == len(items) else
                      "oom" if any(row["oom_blocks"] for row in items) else "partial",
            "mean_seed_median_ms": float(np.mean([row["median_block_median_ms"] for row in complete_items])) if complete_items else None,
            "mean_seed_median_origins_s": float(np.mean([row["median_block_median_origins_s"] for row in complete_items])) if complete_items else None,
            "mean_seed_median_peak_allocated_bytes": float(np.mean([np.median([v for v in row["peak_allocated_bytes"] if v is not None])
                                                                    for row in complete_items])) if complete_items and allocated else None,
            "mean_seed_median_peak_reserved_bytes": float(np.mean([np.median([v for v in row["peak_reserved_bytes"] if v is not None])
                                                                   for row in complete_items])) if complete_items and reserved else None,
            "block_median_ms_range": [min(v for row in items for v in row["block_ms"] if v is not None),
                                      max(v for row in items for v in row["block_ms"] if v is not None)] if any(v is not None for row in items for v in row["block_ms"]) else None,
            "block_median_origins_s_range": [min(v for row in items for v in row["block_origins_s"] if v is not None),
                                             max(v for row in items for v in row["block_origins_s"] if v is not None)] if any(v is not None for row in items for v in row["block_origins_s"]) else None,
            "allocated_range_bytes": [min(allocated), max(allocated)] if allocated else None,
            "reserved_range_bytes": [min(reserved), max(reserved)] if reserved else None,
            "resident_allocated_bytes_range": [min(resident_alloc), max(resident_alloc)] if resident_alloc else None,
            "resident_reserved_bytes_range": [min(resident_reserved), max(resident_reserved)] if resident_reserved else None,
            "deployed_parameters": deployed[0] if deployed and len(set(deployed)) == 1 else None,
            "deployed_parameters_range": [min(deployed), max(deployed)] if deployed else None,
            "parameter_bytes": parameter_bytes[0] if parameter_bytes and len(set(parameter_bytes)) == 1 else None,
            "parameter_bytes_range": [min(parameter_bytes), max(parameter_bytes)] if parameter_bytes else None,
            "buffer_bytes": buffer_bytes[0] if buffer_bytes and len(set(buffer_bytes)) == 1 else None,
            "buffer_bytes_range": [min(buffer_bytes), max(buffer_bytes)] if buffer_bytes else None,
            "deployment_tensor_bytes": tensor_bytes[0] if tensor_bytes and len(set(tensor_bytes)) == 1 else None,
            "deployment_tensor_bytes_range": [min(tensor_bytes), max(tensor_bytes)] if tensor_bytes else None,
            "original_fitted_parameter_count": fitted[0] if fitted and len(set(fitted)) == 1 else None,
            "original_fitted_parameter_count_range": [min(fitted), max(fitted)] if fitted else None,
            "inference_trainable_parameter_count": trainable[0] if trainable and len(set(trainable)) == 1 else None,
            "inference_trainable_parameter_count_range": [min(trainable), max(trainable)] if trainable else None,
            "deployment": [row["model"] for row in items]})
    sentinels, drift_ranges = [], []
    if device == "cuda":
        for dataset in DATASETS:
            drift_values = []
            for block in range(BLOCKS):
                pair = {row["sentinel"]: row for row in rows if row["dataset"] == dataset
                        and row["block"] == block and row["batch"] == 1 and row["sentinel"] is not None}
                if set(pair) != {"pre", "post"}:
                    raise ValueError("Missing Small F0 pre/post sentinel")
                if pair["pre"]["status"] == pair["post"]["status"] == "complete":
                    pre = pair["pre"]["milliseconds_per_origin"]["median"]
                    post = pair["post"]["milliseconds_per_origin"]["median"]
                    change = post / pre - 1
                    drift_values.append(change)
                else:
                    pre = post = change = None
                sentinels.append({"dataset": dataset, "block": block, "batch": 1,
                                  "pre_status": pair["pre"]["status"],
                                  "post_status": pair["post"]["status"],
                                  "pre_ms": pre, "post_ms": post,
                                  "post_relative_change": change,
                                  "pre_row_key": pair["pre"]["key"],
                                  "post_row_key": pair["post"]["key"]})
            drift_ranges.append({"dataset": dataset,
                                 "relative_change_range": [min(drift_values), max(drift_values)] if drift_values else None,
                                 "complete_pairs": len(drift_values),
                                 "interpretation": "Observed sentinel drift range, not a confidence interval"})
    return {"status": "complete", "device": device, "instances": instances,
            "groups": groups, "sentinels": sentinels,
            "sentinel_drift_ranges": drift_ranges,
            "session_ids": sorted({row["job_id"] for row in rows}),
            "aggregation": "Median of three block medians per instance; arithmetic mean of selected LEVEL seed medians",
            "uncertainty": "All raw blocks/ranges retained; no independent-sample confidence claim"}


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty cost CSV")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, allow_nan=False)
                             if isinstance(v, (dict, list)) else "" if v is None else v
                             for k, v in row.items()})


def write_cost_comparison(summary, rows, device):
    output = []
    for row in summary["groups"]:
        output.append({"row_type": "summary", **row})
    for row in rows:
        flat = {k: v for k, v in row.items() if k not in ("seconds_per_24_origins",
                                                          "raw_milliseconds_per_origin",
                                                          "raw_origins_per_second",
                                                          "telemetry_before", "telemetry_after")}
        if row["status"] == "complete":
            flat.update(block_median_ms=row["milliseconds_per_origin"]["median"],
                        block_median_origins_s=row["origins_per_second"]["median"],
                        deployed_parameters=row["model"]["deployed_parameters"],
                        parameter_bytes=row["model"]["parameter_bytes"],
                        buffer_bytes=row["model"]["buffer_bytes"],
                        deployment_tensor_bytes=row["model"]["deployment_tensor_bytes"],
                        original_fitted_parameter_count=row["model"].get("original_fitted_parameter_count"),
                        inference_trainable_parameter_count=row["model"].get("inference_trainable_parameter_count"))
        output.append({"row_type": "block", **flat})
    write_csv(HERE / "cost_comparison.csv", output)
    summary["cost_comparison_csv"] = artifact(HERE / "cost_comparison.csv")
    return output


def cuda_oom(exc):
    name = type(exc).__name__.lower()
    text = repr(exc).lower()
    return "outofmemory" in name or "out of memory" in text or "cuda error: out of memory" in text


def oom_result(config, binding_sha, job, exc, info=None, state_preservation_value=None):
    return {**config, "status": "oom", "binding_sha256": binding_sha,
            "job_id": job.id, "pid": os.getpid(), "finished_utc": time.time(),
            "error": repr(exc), "traceback": traceback.format_exc(),
            "model": info, "seconds_per_24_origins": [],
            "raw_milliseconds_per_origin": [], "raw_origins_per_second": [],
            "milliseconds_per_origin": None, "origins_per_second": None,
            "resident_after_model_load_and_cleanup": None,
            "process_cpu_memory": process_memory(),
            "peak_allocated_bytes": None, "peak_reserved_bytes": None,
            "final_allocated_bytes": None, "warmup_wrapper_calls": WARMUP,
            "timed_wrapper_calls": 0, "passes": PASSES,
            "state_preservation": state_preservation_value,
            "oom_is_result": True}


def run_cost(device, job, preplanned=None):
    check_seal()
    require_preflight_pass()
    instances = instances_by_dataset()
    prepared = {dataset: prepare(dataset) for dataset in DATASETS}
    grid = planned_grid(instances, prepared, device)
    if preplanned and preplanned["expected_rows"] != len(grid):
        raise ValueError("Pre-job planned grid differs from executable grid")
    binding_sha = persist_grid(grid, binding(prepared), device)
    final_path = HERE / f"cost_{device}_rows.json"
    if final_path.exists():
        raise FileExistsError("This cost phase already completed; do not silently remeasure")
    folder = HERE / "cost_rows" / device
    folder.mkdir(parents=True, exist_ok=True)
    results, wrapper, loaded_key, info = [], None, None, None
    try:
        for config in grid:
            job.check_limits()
            path = folder / (config["key"] + ".json")
            if path.exists():
                previous = read_json(path)
                if previous["binding_sha256"] != binding_sha or previous["status"] not in ("complete", "oom"):
                    raise ValueError("A prior row has changed source bindings or is invalid")
                if any(previous.get(key) != value for key, value in config.items()):
                    raise ValueError("An existing cost row differs from its fixed identity")
                if previous["status"] == "complete":
                    preserved = previous.get("state_preservation")
                    if (not preserved or not preserved.get("unchanged")
                            or preserved.get("trainable_parameters") != 0
                            or not preserved.get("eval_mode")):
                        raise ValueError("A prior complete row lacks frozen state preservation evidence")
                results.append(previous)
                continue
            failures = list(folder.glob(config["key"] + ".failure_*.json"))
            if len(failures) >= 2:
                raise RuntimeError("The single technical retry for this cost condition is exhausted")
            item = next(row for row in instances[config["dataset"]] if row["id"] == config["id"])
            instance_key = (config["dataset"], config["id"], config["block"], config["sentinel"])
            row_state_before = None
            try:
                if instance_key != loaded_key:
                    if wrapper is not None:
                        del wrapper
                        wrapper = None
                    cleanup(device)
                    job.heartbeat({"phase": "cost_model_load", "instance": instance_key,
                                   "completed_rows": len(results), "expected_rows": len(grid)})
                    wrapper, info = load_deployment(item, device)
                    loaded_key = instance_key
                job.heartbeat({"phase": "cost_measure", **config,
                               "completed_rows": len(results), "expected_rows": len(grid)})
                row_state_before = module_state_snapshot(wrapper)
                require_frozen_state(row_state_before, "row_start")
                before = telemetry() if device == "cuda" else None
                values = measure(wrapper, prepared[config["dataset"]], config, device, job)
                row_state_after = module_state_snapshot(wrapper)
                preserved = state_preservation(row_state_before, row_state_after)
                row = {**config, **values, "status": "complete",
                       "binding_sha256": binding_sha, "job_id": job.id,
                       "pid": os.getpid(), "finished_utc": time.time(),
                       "model": info, "telemetry_before": before,
                       "state_preservation": preserved,
                       "telemetry_after": telemetry() if device == "cuda" else None}
                save_json(path, row)
                results.append(row)
            except BaseException as exc:
                if device == "cuda" and cuda_oom(exc):
                    preserved = None
                    if row_state_before is not None and wrapper is not None:
                        try:
                            row_state_after = module_state_snapshot(wrapper)
                            preserved = state_preservation(row_state_before, row_state_after)
                        except Exception as state_exc:
                            if "STOP_DEBUG" in repr(state_exc):
                                raise
                            preserved = {"before_sha256": row_state_before["sha256"],
                                         "after_sha256": None,
                                         "unchanged": None,
                                         "eval_mode": None,
                                         "trainable_parameters": None,
                                         "optimizer_present": False,
                                         "hash_scope": "parameters_and_buffers",
                                         "unavailable_after_oom": repr(state_exc)}
                    row = oom_result(config, binding_sha, job, exc, info, preserved)
                    save_json(path, row)
                    results.append(row)
                    if wrapper is not None:
                        del wrapper
                        wrapper = None
                    loaded_key = None
                    cleanup(device)
                else:
                    save_json(folder / (config["key"] + f".failure_{job.id}.json"),
                              {"status": "failed", "config": config,
                               "binding_sha256": binding_sha, "job_id": job.id,
                               "error": repr(exc), "traceback": traceback.format_exc(),
                               "utc": time.time(), "completed_rows": len(results)})
                    raise
            save_json(HERE / f"cost_{device}_rows_partial.json",
                      {"status": "partial", "rows": results,
                       "completed_rows": len(results), "expected_rows": len(grid),
                       "binding_sha256": binding_sha,
                       "unfinished_keys": [row["key"] for row in grid[len(results):]]})
        output = {"status": "complete", "rows": results, "expected_rows": len(grid),
                  "binding_sha256": binding_sha, "environment": environment(),
                  "counts": call_counts(grid), "session_ids": sorted({row["job_id"] for row in results}),
                  "scope": "Same campaign; standardized CPU input to full online H48 CPU output; VAL only; no fitting",
                  "oom_rows": sum(row["status"] == "oom" for row in results)}
        save_json(final_path, output)
        summary = summarize(results, device)
        write_cost_comparison(summary, results, device)
        save_json(HERE / f"cost_{device}_summary.json", summary)
        return output
    finally:
        if wrapper is not None:
            del wrapper
        cleanup(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--job")
    args = parser.parse_args()
    require_static_apis()
    configure()
    preplanned = planned_grid_current_limits(args.device)
    category = "gpu" if args.device == "cuda" else "cpu"
    job_label = args.job or f"same_campaign_{args.device}_cost"
    with Job(category, job_label, reserve_s=0,
             metadata={"purpose": "same-campaign deployment cost",
                       "device": args.device, "fit_count": 0,
                       "planned_rows": preplanned["expected_rows"],
                       "passes": PASSES, "warmup_calls": WARMUP,
                       "observer": "Synchronous outside each timed pass",
                       "preplanned": preplanned}) as job:
        job._stop.set()
        job._thread.join(timeout=12)
        if job._thread.is_alive():
            raise RuntimeError("Background observer did not stop before timing")
        import torch
        with torch.inference_mode():
            run_cost(args.device, job, preplanned=preplanned)


if __name__ == "__main__":
    main()
