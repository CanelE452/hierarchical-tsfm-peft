"""Fresh-process measurements of the complete, online physical forecast path."""
from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path
import time

import numpy as np
import psutil
import torch

from . import common as c
from . import data as d


MIB = 1024 ** 2


def group_indices(contract, batch, origin_count):
    if batch not in (1, 8):
        raise ValueError("Registered cost batch must be 1 or 8")
    starts = contract["cost"]["fixed_group_indices"][str(batch)]
    if (len(starts) != 3 or any(not isinstance(start, int) or isinstance(start, bool)
                              or start < 0 or start + batch > origin_count for start in starts)):
        raise ValueError("Registered cost group falls outside the fixed VAL origin set")
    return [np.arange(start, start + batch, dtype=np.int64) for start in starts]


def timing_summary(trials):
    timed = [row for row in trials if row["stage"] == "timed"]
    values = np.asarray([row["whole_batch_ms"] for row in timed], dtype=np.float64)
    if not len(timed) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Timing summary requires positive finite timed repetitions")
    if len({row["batch"] for row in timed}) != 1:
        raise ValueError("Timing summary cannot mix physical batch sizes")
    q25, median, q75, p95 = np.quantile(values, [.25, .5, .75, .95], method="linear")
    return {"p50_ms": float(median), "q25_ms": float(q25), "q75_ms": float(q75),
            "iqr_ms": float(q75 - q25), "p95_ms": float(p95), "repetitions": len(timed),
            "origins_per_second_at_median": float(1000 * timed[0]["batch"] / median),
            "peak_allocated_MiB": max(row["peak_allocated_MiB"] for row in timed),
            "peak_reserved_MiB": max(row["peak_reserved_MiB"] for row in timed),
            "RSS_MiB": max(row["RSS_MiB"] for row in timed),
            "quantile_method": "NumPy linear empirical quantiles",
            "scope": "20 repetitions cycling three fixed origin groups; not a service-tail latency estimate"}


def resident_inventory(modules):
    """Count each resident Parameter/buffer object once across all loaded modules."""
    zero = {"parameters": 0, "parameter_bytes": 0, "buffer_bytes": 0}
    by_device = {name: zero.copy() for name in ("cpu", "cuda")}
    by_module = {}
    seen_parameters, seen_buffers, devices = set(), set(), set()
    skipped = {"parameter_objects": 0, "buffer_objects": 0}
    for name, model in modules.items():
        counts = {device: zero.copy() for device in ("cpu", "cuda")}
        for kind, tensors, seen in (("parameter", model.parameters(), seen_parameters),
                                    ("buffer", model.buffers(), seen_buffers)):
            for tensor in tensors:
                if id(tensor) in seen:
                    skipped[kind + "_objects"] += 1
                    continue
                seen.add(id(tensor))
                device = tensor.device.type
                devices.add(str(tensor.device))
                counts.setdefault(device, zero.copy())
                by_device.setdefault(device, zero.copy())
                size_bytes = tensor.numel() * tensor.element_size()
                if kind == "parameter":
                    counts[device]["parameters"] += tensor.numel()
                    by_device[device]["parameters"] += tensor.numel()
                counts[device][kind + "_bytes"] += size_bytes
                by_device[device][kind + "_bytes"] += size_bytes
        by_module[name] = counts
    return {"by_device": by_device, "by_module": by_module, "actual_devices": sorted(devices),
            "total_unique_parameters": sum(row["parameters"] for row in by_device.values()),
            "total_unique_parameter_bytes": sum(row["parameter_bytes"] for row in by_device.values()),
            "total_unique_buffer_bytes": sum(row["buffer_bytes"] for row in by_device.values()),
            "aliases_skipped": skipped,
            "counting_rule": "unique Parameter object id across modules; unique buffer object id separately; actual numel*element_size bytes",
            "scope": "all loaded model weights, including unused TS2 online CPU copy; F counted once; latent compression does not compress resident weights"}


def _online_physical(ctx, route, foundation, physical, mask):
    from .execution import route_forward

    data = ctx["data"]
    normalized = d.fill_scale(physical, mask, data["mean"], data["std"])
    with torch.no_grad():
        x = torch.as_tensor(normalized, dtype=torch.float32, device="cuda")
        normalized_forecast = route_forward(route, foundation, x).cpu().numpy()
    return (normalized_forecast.astype(np.float64) * np.asarray(data["std"], dtype=np.float64)
            + np.asarray(data["mean"], dtype=np.float64))


def _trial(stage, iteration, group, ctx, route, foundation, reference, process):
    from .execution import parity

    torch.cuda.synchronize()
    if stage == "timed":
        torch.cuda.reset_peak_memory_stats()
    calls_before, streams_before = foundation.calls, foundation.streams
    context_start = len(foundation.context_records)
    started = time.perf_counter()
    physical_prediction = _online_physical(ctx, route, foundation, group["physical"], group["mask"])
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000
    allocated = torch.cuda.max_memory_allocated() / MIB
    reserved = torch.cuda.max_memory_reserved() / MIB
    rss = process.memory_info().rss / MIB
    if not np.isfinite(elapsed_ms) or elapsed_ms <= 0:
        raise ValueError("Full-path timing must be positive and finite")
    # The timed path returns physical FP64 forecasts. Re-enter the unchanged
    # normalized score coordinate outside timing for comparison to sealed output.
    normalized_prediction = ((physical_prediction - np.asarray(ctx["data"]["mean"], dtype=np.float64))
                             / np.asarray(ctx["data"]["std"], dtype=np.float64))
    prediction_parity = parity(normalized_prediction, reference[group["indices"]], atol=1e-5, rtol=1e-4)
    contexts = foundation.context_records[context_start:]
    batch = len(group["indices"])
    return {"stage": stage, "iteration": iteration, "group_index": iteration % 3,
            "groupstart": int(group["indices"][0]), "origins_sha256": c.array_sha(group["origins"]),
            "batch": batch, "whole_batch_ms": elapsed_ms, "origins_per_second": 1000 * batch / elapsed_ms,
            "peak_allocated_MiB": allocated, "peak_reserved_MiB": reserved, "RSS_MiB": rss,
            "memory_peak_scope": "one timed repetition including all resident models" if stage == "timed" else "cumulative warmup phase",
            "F_calls": foundation.calls - calls_before, "F_streams": foundation.streams - streams_before,
            "requested_context_length": [row["requested_context_length"] for row in contexts],
            "actual_context_length": [row["actual_context_length"] for row in contexts],
            "F_context_records": contexts, "TS2cache_used": False,
            "prediction_parity": prediction_parity}


def _modules(route, foundation):
    modules = {"F": foundation.model}
    if "gt_route" in route:
        modules["GT_adapter"] = route["gt_route"]
    if "ts2" in route:
        modules.update(TS2_SWA=route["ts2"].net, TS2_online_CPU=route["ts2"]._net)
    if "decoder" in route:
        modules["TS2_decoder"] = route["decoder"]
    return modules


def _provenance(spec, ctx, foundation):
    from .execution import source_identity

    checkpoints = {"F_tensor_state": c.state_sha(foundation.model)}
    sources = {"new_campaign": source_identity(),
               "F_installed_model_implementation": c.file_sha(Path(inspect.getfile(type(foundation.model))))}
    mode, seed = spec["mode"], str(spec["seed"])
    if mode != "raw":
        parents = ctx["parents"][seed]
        if mode in ("gt", "pre", "rand", "ensemble"):
            checkpoints["GT_full_calibrated_parent"] = parents["GT"]["checkpoint_sha256"]
            sources["GT_parent"] = parents["GT"]["source_pins"]
        if mode in ("ts2", "pre", "rand", "ensemble"):
            ts2_reference = "TS2_architecture_parent_reference" if mode == "rand" else "TS2_pretrained_SWA_parent"
            checkpoints[ts2_reference] = parents["TS2"]["checkpoint_sha256"]
            sources["TS2_parent"] = parents["TS2"]["source_pins"]
        if mode in ("ts2", "ensemble"):
            checkpoints["TS2_forecast_decoder_B"] = parents["TS2"]["decoder_checkpoint_sha256"]
    if spec["phase"] == "B":
        updates = spec["updates"]
        checkpoint = c.CACHE / "training" / spec["run_id"] / f"checkpoint_{updates:06d}.pt"
        checkpoints["adapted_final_checkpoint"] = c.file_sha(checkpoint)
    return checkpoints, sources


def _reference(spec, data):
    path = c.CACHE / "predictions" / (spec["run_id"] + ".npz")
    receipt = c.read_json(c.REPORT / "receipts" / (spec["run_id"] + ".json"))
    if (receipt["status"] != "COMPLETE" or receipt["run_id"] != spec["run_id"]
            or receipt["data_contract_sha256"] != data["contract_hash"]
            or c.file_sha(path) != receipt["artifact_sha256"]):
        raise ValueError("Cost reference must be the same sealed run/data prediction artifact")
    with np.load(path, allow_pickle=False) as archive:
        prediction = archive["prediction"].copy()
        for key, expected in (("origins", data["val_origins"]), ("mean", data["mean"]), ("std", data["std"])):
            if not np.array_equal(archive[key], expected):
                raise ValueError("Cost reference origins/normalization differ from the sealed run")
    if (prediction.shape != (len(data["val_origins"]), 48, 21)
            or c.array_sha(prediction) != receipt["prediction_sha256"]):
        raise ValueError("Cost reference prediction SHA/schema changed")
    return prediction, receipt


def measure(run_id, batch):
    """Measure exactly one candidate/batch in this caller-owned fresh process."""
    from . import execution as e

    started_unix = time.time()
    started = time.perf_counter()
    specs = [row for row in e.candidate_specs() if row["run_id"] == run_id]
    if len(specs) != 1 or batch not in (1, 8):
        raise ValueError("Cost worker requires an exact registered candidate and batch 1 or 8")
    spec = dict(specs[0])
    if spec["phase"] == "B":
        spec["updates"] = c.read_json(c.REPORT / "UPDATE_DECISION.json")["updates"]
        if spec["updates"] not in (100, 300):
            raise ValueError("Adapted cost worker requires a registered 100/300 update decision")
    output = c.REPORT / "cost_receipts" / f"{run_id}_B{batch}.json"
    if output.exists():
        raise FileExistsError("Preserve existing cost receipt; it cannot be overwritten")
    contract = c.read_json(c.REPORT / "EXPERIMENT_CONTRACT.json")
    if (contract["cost"]["warmups"] != 10 or contract["cost"]["repetitions"] != 20
            or not contract["cost"]["groups_cycle"] or not contract["cost"]["full_online_TS2"]):
        raise ValueError("Cost worker must use the registered 10 warmups/20 online repetitions")
    context_started = time.perf_counter()
    ctx = e.context()
    data = ctx["data"]
    reference, prediction_receipt = _reference(spec, data)
    context_seconds = time.perf_counter() - context_started
    e.cuda_setup()
    load_started = time.perf_counter()
    foundation = e.foundation(ctx)
    route = e.build_route(spec, ctx, foundation, adapted=spec["phase"] == "B", online=True)
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_started
    if foundation.row_microbatch != 128:
        raise ValueError("Cost F row microbatch must remain fixed at 128")
    process = psutil.Process(os.getpid())
    process_metadata = {"pid": process.pid, "process_create_time_unix": process.create_time()}
    loaded_baseline = {"allocated_MiB": torch.cuda.memory_allocated() / MIB,
                       "reserved_MiB": torch.cuda.memory_reserved() / MIB,
                       "RSS_MiB": process.memory_info().rss / MIB,
                       "loading_peak_allocated_MiB": torch.cuda.max_memory_allocated() / MIB,
                       "loading_peak_reserved_MiB": torch.cuda.max_memory_reserved() / MIB}
    checkpoints, sources = _provenance(spec, ctx, foundation)
    modules = _modules(route, foundation)
    inventory = resident_inventory(modules)
    before_states = {name: c.state_sha(model) for name, model in modules.items()}
    if "TS2_SWA" in before_states:
        checkpoints["TS2_actual_inference_tensor_state"] = before_states["TS2_SWA"]
    slice_started = time.perf_counter()
    groups = []
    for indices in group_indices(contract, batch, len(data["val_origins"])):
        origins = data["val_origins"][indices]
        physical, mask = d.physical_windows(data, origins, spec["L"])
        groups.append({"indices": indices, "origins": origins, "physical": physical, "mask": mask})
    window_slicing_seconds = time.perf_counter() - slice_started
    trials = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for stage, repetitions in (("warmup", 10), ("timed", 20)):
        for iteration in range(repetitions):
            row = _trial(stage, iteration, groups[iteration % 3], ctx, route, foundation, reference, process)
            row.update(run_id=run_id, seed=spec["seed"], phase=spec["phase"], L=spec["L"],
                       updates=spec["updates"],
                       **process_metadata, checkpoint_sha256=checkpoints, loaded_source_sha256=sources)
            trials.append(row)
            c.progress("cost", run_id=run_id, batch=batch, measurement_stage=stage,
                       repetition=iteration + 1, total=repetitions)
    after_states = {name: c.state_sha(model) for name, model in modules.items()}
    gradients_absent = {name: all(parameter.grad is None for parameter in model.parameters())
                        for name, model in modules.items()}
    if before_states != after_states or not all(gradients_absent.values()):
        raise ValueError("A loaded model changed or accumulated gradients during cost measurement")
    f_invariant = foundation.verify_unchanged()
    if not all(f_invariant.values()):
        raise ValueError("Foundation invariant failed during cost measurement")
    summary = timing_summary(trials)
    summary.update(run_id=run_id, seed=spec["seed"], phase=spec["phase"], mode=spec["mode"],
                   arm=spec["arm"], L=spec["L"], updates=spec["updates"], batch=batch, warmups=10, TS2cache_used=False)
    summary.update(resident_parameters=inventory["by_device"]["cuda"]["parameters"],
                   CUDA_resident_parameters=inventory["by_device"]["cuda"]["parameters"],
                   CPU_resident_parameters=inventory["by_device"]["cpu"]["parameters"],
                   system_resident_parameters=inventory["total_unique_parameters"],
                   CUDA_resident_parameter_bytes=inventory["by_device"]["cuda"]["parameter_bytes"],
                   CPU_resident_parameter_bytes=inventory["by_device"]["cpu"]["parameter_bytes"],
                   CUDA_resident_buffer_bytes=inventory["by_device"]["cuda"]["buffer_bytes"],
                   CPU_resident_buffer_bytes=inventory["by_device"]["cpu"]["buffer_bytes"],
                   resident_parameter_scope="resident_parameters is CUDA only; CPU/system counts are separate",
                   resident_inventory=inventory)
    receipt = {"status": "COMPLETE", "run_id": run_id, "spec": spec, "batch": batch,
               **process_metadata, "started_unix": started_unix, "ended_unix": time.time(),
               "wall_seconds": time.perf_counter() - started,
               "device": torch.cuda.get_device_name(), "precision": "FP32, TF32 disabled, no AMP",
               "F_row_microbatch": 128, "TS2cache_used": False,
               "TS2_present": "ts2" in route,
               "TS2_pretrained_weights_used": spec["mode"] in ("ts2", "ensemble", "pre"),
               "boundary": "physical history+mask -> local fill/TRAIN scale -> H2D -> encoders/bridge/F/native D -> D2H -> physical FP64 inverse scale",
               "excluded_from_timing": ["model loading", "physical window slicing", "reference file I/O", "parity checks", "state hashing"],
               "context_and_reference_loading_seconds": context_seconds,
               "model_loading_seconds": model_load_seconds, "window_slicing_seconds": window_slicing_seconds,
               "loaded_baseline": loaded_baseline,
               "resident_inventory": inventory,
               "warmup_peaks": {key: max(row[key] for row in trials if row["stage"] == "warmup")
                                for key in ("peak_allocated_MiB", "peak_reserved_MiB", "RSS_MiB")},
               "memory_scope": "allocated/reserved peaks include all currently resident models; RSS is a sampled process value; no process-GPU hard-cap certification",
               "checkpoint_sha256": checkpoints, "loaded_source_sha256": sources,
               "states_before_sha256": before_states, "states_after_sha256": after_states,
               "reference_prediction_sha256": prediction_receipt["prediction_sha256"],
               "reference_artifact_sha256": prediction_receipt["artifact_sha256"],
               "validation": {"status": "PASS", "prediction_parity_trials": len(trials),
                              "max_abs_prediction_difference": max(row["prediction_parity"]["max_abs"] for row in trials),
                              "F_unchanged": f_invariant["unchanged"], "TS2_unchanged": True,
                              "all_loaded_states_unchanged": True, "parameter_gradients_absent": gradients_absent},
               "summary": summary, "trials": trials}
    c.seal_json(output, receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batch", type=int, choices=(1, 8), required=True)
    args = parser.parse_args(argv)
    receipt = measure(args.run_id, args.batch)
    print(c.json_bytes({"status": receipt["status"], "summary": receipt["summary"],
                        "validation": receipt["validation"]}).decode(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
