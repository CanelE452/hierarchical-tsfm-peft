"""Cost probes for the matched learned E/D campaign.

The file is source-only until the STOP_RESOURCE gate is lifted.  It defines the
fixed training/inference grids and the receipt schema, but it creates no
placeholder result JSON.  Execution phases are intentionally explicit.
"""

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

from runtime_v18 import (
    CACHE,
    DATASETS,
    HERE,
    Job,
    artifact,
    array_hash,
    check_selection_seal,
    configure,
    environment,
    inputs,
    load_data,
    load_original,
    original_levels,
    read_json,
    resolve,
    save_json,
    state_hash,
    targets,
)


METHOD = "MATCHED_LEARNED_ED"
LEVEL = "ORIGINAL_LEVEL"
BLOCKS = 3
WARMUP = 10
REPETITIONS = 10
TRAIN_BATCH = 4
INFER_ORIGINS = 24
INFER_BATCHES = (1, 4)
EXPECTED_TRAINING_ROWS = 36
EXPECTED_INFERENCE_ROWS = 72


def schema_contract():
    return {
        "status": "source_only_unexecuted",
        "blocked_reason": "STOP_RESOURCE before source seal / TRAIN / TEST / cost",
        "training_cost_probe": {
            "path": "research/level_matched_learned_ed_v18_20261006/training_cost_probe.json",
            "rows": EXPECTED_TRAINING_ROWS,
            "grid": "3 datasets × (2 selected ED seeds + 2 original LEVEL seeds) × 3 rotated blocks",
            "row_keys": [
                "row_type",
                "dataset",
                "method",
                "id",
                "seed",
                "block",
                "batch",
                "status",
                "raw_milliseconds_per_origin",
                "milliseconds_per_origin",
                "origins_per_second",
                "state_preservation",
                "model",
                "memory",
            ],
        },
        "inference_cost_rows": {
            "path": "research/level_matched_learned_ed_v18_20261006/inference_cost_rows.json",
            "rows": EXPECTED_INFERENCE_ROWS,
            "grid": "3 datasets × (2 selected ED seeds + 2 original LEVEL seeds) × B1/B4 × 3 rotated blocks",
            "row_keys": [
                "row_type",
                "dataset",
                "method",
                "id",
                "seed",
                "block",
                "batch",
                "status",
                "raw_milliseconds_per_origin",
                "milliseconds_per_origin",
                "origins_per_second",
                "state_preservation",
                "model",
                "memory",
            ],
        },
        "summary_csv": {
            "path": "research/level_matched_learned_ed_v18_20261006/cost_v18_comparison.csv",
            "row_types": ["summary", "block"],
            "aggregation": "Median of three block medians per instance; no CI",
        },
    }


def digest_row(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:24]


def rotated(values, block):
    values = list(values)
    if block == 1:
        return values[1:] + values[:1]
    if block == 2:
        return list(reversed(values))
    return values


def selected_ed_members():
    selection = read_json(HERE / "selection.json")
    members = {}
    for dataset in DATASETS:
        unit = selection["units"][dataset]
        rows = []
        for index, checkpoint in enumerate(unit["checkpoints"]):
            artifact(checkpoint["path"], checkpoint["sha256"])
            rows.append(
                {
                    "dataset": dataset,
                    "method": METHOD,
                    "id": unit["run_ids"][index],
                    "seed": unit["seeds"][index],
                    "lr": unit["lr"],
                    "checkpoint": checkpoint,
                    "selected_epoch": unit["selected_epochs"][index],
                    "source_kind": "selected_ed",
                }
            )
        members[dataset] = rows
    return members


def all_members():
    ed = selected_ed_members()
    members = {}
    for dataset in DATASETS:
        level_rows = []
        for row in original_levels(dataset):
            item = dict(row)
            item.update(method=LEVEL, original_method="LEVEL", source_kind="original_level")
            level_rows.append(item)
        if len(level_rows) != 2 or len(ed[dataset]) != 2:
            raise RuntimeError("Cost requires exactly two selected ED and two original LEVEL members per dataset")
        members[dataset] = ed[dataset] + level_rows
    return members


def planned_training_grid():
    members = all_members()
    rows = []
    for block in range(BLOCKS):
        for dataset in rotated(DATASETS, block):
            for member in rotated(members[dataset], block):
                row = {
                    "row_type": "training_block",
                    "dataset": dataset,
                    "method": member["method"],
                    "id": member["id"],
                    "seed": member.get("seed"),
                    "source_kind": member["source_kind"],
                    "block": block,
                    "batch": TRAIN_BATCH,
                    "warmup": WARMUP,
                    "repetitions": REPETITIONS,
                    "optimizer_updates": 0,
                    "scope": "standardized CPU X/target/mask -> GPU forward + observed-mask MSE + backward; no optimizer.step",
                }
                row["key"] = digest_row(row)
                row["planned_sequence"] = len(rows)
                rows.append(row)
    if len(rows) != EXPECTED_TRAINING_ROWS or len({row["key"] for row in rows}) != EXPECTED_TRAINING_ROWS:
        raise RuntimeError("Training cost grid changed")
    return rows


def planned_inference_grid():
    members = all_members()
    rows = []
    for block in range(BLOCKS):
        for dataset in rotated(DATASETS, block):
            for member in rotated(members[dataset], block):
                for batch in INFER_BATCHES:
                    row = {
                        "row_type": "inference_block",
                        "dataset": dataset,
                        "method": member["method"],
                        "id": member["id"],
                        "seed": member.get("seed"),
                        "source_kind": member["source_kind"],
                        "block": block,
                        "batch": batch,
                        "warmup": WARMUP,
                        "passes": REPETITIONS,
                        "scope": "standardized CPU input -> online model path -> contiguous CPU [B,48,C]",
                    }
                    row["key"] = digest_row(row)
                    row["planned_sequence"] = len(rows)
                    rows.append(row)
    if len(rows) != EXPECTED_INFERENCE_ROWS or len({row["key"] for row in rows}) != EXPECTED_INFERENCE_ROWS:
        raise RuntimeError("Inference cost grid changed")
    return rows


def prepare_training(dataset):
    data = load_data(dataset, include_test=False)
    origins = np.asarray(data["train_origins"], dtype=np.int64)[:TRAIN_BATCH]
    if len(origins) != TRAIN_BATCH or np.any(np.diff(origins) <= 0):
        raise RuntimeError("Training probe requires first four chronological TRAIN origins")
    x_cpu = inputs(data, origins)
    target, mask = targets(data, origins)
    return {
        "origins": origins,
        "inputs": x_cpu,
        "target": target.astype(np.float32),
        "mask": mask.astype(bool),
        "C": int(data["_C"]),
        "K": int(data["_K"]),
        "trainval": artifact(data["_path"]),
        "input_sha256": array_hash(x_cpu.numpy()),
        "target_sha256": array_hash(target.astype(np.float32)),
        "mask_sha256": array_hash(mask.astype(bool)),
        "origin_sha256": array_hash(origins),
    }


def prepare_inference(dataset):
    data = load_data(dataset, include_test=False)
    origins = np.asarray(data["val_origins"], dtype=np.int64)[:INFER_ORIGINS]
    if len(origins) != INFER_ORIGINS or np.any(np.diff(origins) <= 0):
        raise RuntimeError("Inference cost requires first 24 chronological VAL origins")
    x_cpu = inputs(data, origins)
    return {
        "origins": origins,
        "inputs": x_cpu,
        "C": int(data["_C"]),
        "K": int(data["_K"]),
        "trainval": artifact(data["_path"]),
        "input_sha256": array_hash(x_cpu.numpy()),
        "origin_sha256": array_hash(origins),
    }


def cleanup(device):
    import torch

    gc.collect()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def process_memory():
    try:
        import psutil

        info = psutil.Process().memory_info()
        return {name: int(getattr(info, name)) for name in ("rss", "vms", "peak_wset") if hasattr(info, name)}
    except (ImportError, OSError):
        return None


def telemetry():
    fields = "index,uuid,temperature.gpu,clocks.current.sm,utilization.gpu,power.draw,pstate"
    result = {}
    for name, query in (
        ("gpu", ["--query-gpu=" + fields, "--format=csv,noheader,nounits"]),
        ("processes", ["--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader"]),
    ):
        try:
            run = subprocess.run(["nvidia-smi", *query], capture_output=True, text=True, timeout=15)
            result[name] = {"status": run.returncode, "rows": run.stdout.strip().splitlines(), "error": run.stderr.strip() or None}
        except (OSError, subprocess.TimeoutExpired) as exc:
            result[name] = {"status": None, "rows": None, "error": repr(exc)}
    return {"utc": time.time(), "fields": fields, **result, "unavailable_values": "N/A is unavailable, never zero"}


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def model_bytes(model):
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    parameter_bytes = sum(p.numel() * p.element_size() for p in parameters)
    buffer_bytes = sum(b.numel() * b.element_size() for b in buffers)
    return {
        "deployed_parameters": sum(p.numel() for p in parameters),
        "parameter_bytes": parameter_bytes,
        "buffer_bytes": buffer_bytes,
        "deployment_tensor_bytes": parameter_bytes + buffer_bytes,
        "trainable_parameters": sum(p.numel() for p in parameters if p.requires_grad),
    }


def state_snapshot(model):
    return {
        "sha256": state_hash(model),
        "eval_mode": not model.training,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "optimizer_present": False,
        "hash_scope": "parameters_and_buffers_state_dict_only",
    }


def state_preservation(before, after, expect_trainable):
    result = {
        "before_sha256": before["sha256"],
        "after_sha256": after["sha256"],
        "unchanged": before["sha256"] == after["sha256"],
        "eval_mode": after["eval_mode"],
        "trainable_parameters": after["trainable_parameters"],
        "expected_trainable_parameters": expect_trainable,
        "optimizer_present": False,
        "hash_scope": "parameters_and_buffers_state_dict_only",
    }
    if not result["unchanged"] or result["trainable_parameters"] != expect_trainable:
        raise RuntimeError("STOP_DEBUG: cost row model state changed or trainable set drifted")
    return result


def load_member(member, device, training):
    if member["method"] == METHOD:
        from model_v18 import restore_model

        model = restore_model(member["checkpoint"]["path"], device=device, expected_sha256=member["checkpoint"]["sha256"])
        if training:
            model.train()
        else:
            model.eval()
            model.requires_grad_(False)
        receipt = model.receipt(include_state_hash=False)
        fitted = 2 * receipt["channels"] * receipt["latent"]
        return model, {**model_bytes(model), "original_fitted_parameter_count": fitted, "inference_trainable_parameter_count": 0 if not training else fitted, "deployment_receipt": receipt}
    original_row = dict(member)
    original_row["method"] = original_row.get("original_method", "LEVEL")
    model = load_original(original_row, device=device, deploy=not training)
    if training:
        for name, param in model.named_parameters():
            param.requires_grad_("residual" in name.lower() or name.lower().startswith("g") or ".g" in name.lower())
        model.train()
        if hasattr(model, "backbone"):
            model.backbone.eval()
    else:
        model.eval()
        model.requires_grad_(False)
    return model, {**model_bytes(model), "original_fitted_parameter_count": member.get("original_fitted_parameter_count"), "inference_trainable_parameter_count": 0 if not training else None}


def backward_level_batch(model, prepared):
    import torch
    from model_v18 import compute_loss

    device = next(model.parameters()).device
    x = prepared["inputs"].to(device)
    y = torch.as_tensor(prepared["target"], device=device)
    mask = torch.as_tensor(prepared["mask"], device=device)
    prediction = model(x)
    loss = compute_loss(prediction, y, mask)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("STOP_DEBUG: nonfinite LEVEL training-probe loss")
    loss.backward()
    return float(loss.detach())


def backward_ed_batch(model, prepared):
    from train_v18 import backward_effective_batch

    return backward_effective_batch(model, prepared["inputs"], prepared["target"], prepared["mask"], microbatch=4)[0]


def zero_grad(model):
    for param in model.parameters():
        param.grad = None


def training_step(model, method, prepared):
    zero_grad(model)
    return backward_ed_batch(model, prepared) if method == METHOD else backward_level_batch(model, prepared)


def forward_cpu_output(model, cpu_input):
    import torch

    result = model(cpu_input.to(next(model.parameters()).device))
    if not isinstance(result, torch.Tensor):
        result = torch.from_numpy(np.asarray(result))
    result = result.float().detach().cpu().contiguous()
    if tuple(result.shape) != (len(cpu_input), 48, cpu_input.shape[2]):
        raise RuntimeError("Cost output must contain every original channel and H48")
    return result


def measure_training(model, method, prepared, device, job):
    import torch

    cleanup(device)
    resident = {"allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved()} if device == "cuda" else None
    for _ in range(WARMUP):
        job.check_limits()
        training_step(model, method, prepared)
        zero_grad(model)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    times = []
    for repeat in range(REPETITIONS):
        job.heartbeat({"phase": "training_cost", "repeat": repeat})
        if device == "cuda":
            torch.cuda.synchronize()
        tick = time.perf_counter()
        training_step(model, method, prepared)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - tick)
        zero_grad(model)
    ms = [value * 1000 / TRAIN_BATCH for value in times]
    speed = [TRAIN_BATCH / value for value in times]
    return {
        "seconds_per_effective_batch": times,
        "raw_milliseconds_per_origin": ms,
        "raw_origins_per_second": speed,
        "milliseconds_per_origin": describe(ms),
        "origins_per_second": describe(speed),
        "resident_after_model_load_and_cleanup": resident,
        "process_cpu_memory": process_memory(),
        "warmup_calls": WARMUP,
        "passes": REPETITIONS,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None,
        "final_allocated_bytes": torch.cuda.memory_allocated() if device == "cuda" else None,
    }


def measure_inference(model, prepared, batch, device, job):
    import torch

    groups = list(prepared["inputs"].split(batch))
    cleanup(device)
    resident = {"allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved()} if device == "cuda" else None
    with torch.inference_mode():
        for index in range(WARMUP):
            job.check_limits()
            forward_cpu_output(model, groups[index % len(groups)])
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        times = []
        for repeat in range(REPETITIONS):
            job.heartbeat({"phase": "inference_cost", "repeat": repeat})
            if device == "cuda":
                torch.cuda.synchronize()
            tick = time.perf_counter()
            for group in groups:
                forward_cpu_output(model, group)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - tick)
    ms = [value * 1000 / INFER_ORIGINS for value in times]
    speed = [INFER_ORIGINS / value for value in times]
    return {
        "seconds_per_24_origins": times,
        "raw_milliseconds_per_origin": ms,
        "raw_origins_per_second": speed,
        "milliseconds_per_origin": describe(ms),
        "origins_per_second": describe(speed),
        "resident_after_model_load_and_cleanup": resident,
        "process_cpu_memory": process_memory(),
        "warmup_wrapper_calls": WARMUP,
        "timed_wrapper_calls": REPETITIONS * len(groups),
        "passes": REPETITIONS,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None,
        "final_allocated_bytes": torch.cuda.memory_allocated() if device == "cuda" else None,
    }


def run_rows(phase, rows, prepared, device, job):
    import torch

    members = all_members()
    folder = HERE / "cost_rows_v18" / phase
    folder.mkdir(parents=True, exist_ok=True)
    output = []
    for config in rows:
        row_path = folder / (config["key"] + ".json")
        if row_path.exists():
            output.append(read_json(row_path))
            continue
        member = next(row for row in members[config["dataset"]] if row["id"] == config["id"])
        try:
            model, info = load_member(member, device, training=phase == "training")
            expected_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            before = state_snapshot(model)
            before_telemetry = telemetry() if device == "cuda" else None
            if phase == "training":
                values = measure_training(model, config["method"], prepared[config["dataset"]], device, job)
            else:
                values = measure_inference(model, prepared[config["dataset"]], config["batch"], device, job)
            after = state_snapshot(model)
            record = {
                **config,
                **values,
                "status": "complete",
                "finished_utc": time.time(),
                "job_id": job.id,
                "pid": os.getpid(),
                "model": info,
                "state_preservation": state_preservation(before, after, expected_trainable),
                "telemetry_before": before_telemetry,
                "telemetry_after": telemetry() if device == "cuda" else None,
            }
            save_json(row_path, record)
            output.append(record)
            del model
            cleanup(device)
        except torch.cuda.OutOfMemoryError as exc:
            record = {**config, "status": "oom", "finished_utc": time.time(), "job_id": job.id, "pid": os.getpid(), "error": repr(exc), "traceback": traceback.format_exc(), "oom_is_result": True}
            save_json(row_path, record)
            output.append(record)
            cleanup(device)
        save_json(HERE / f"{phase}_cost_partial.json", {"status": "partial", "rows": output, "completed_rows": len(output), "expected_rows": len(rows)})
    return output


def summarize(rows, phase):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["method"], row["id"], row.get("seed"), row.get("batch", TRAIN_BATCH))].append(row)
    instances = []
    for key, group in grouped.items():
        medians = [row["milliseconds_per_origin"]["median"] for row in group if row["status"] == "complete"]
        speed = [row["origins_per_second"]["median"] for row in group if row["status"] == "complete"]
        first = group[0]
        instances.append(
            {
                "dataset": key[0],
                "method": key[1],
                "id": key[2],
                "seed": key[3],
                "batch": key[4],
                "phase": phase,
                "status": "complete" if len(medians) == BLOCKS else "oom" if any(row["status"] == "oom" for row in group) else "partial",
                "complete_blocks": len(medians),
                "median_block_median_ms": float(np.median(medians)) if medians else None,
                "median_block_median_origins_s": float(np.median(speed)) if speed else None,
                "block_median_ms_range": [min(medians), max(medians)] if medians else None,
                "block_median_origins_s_range": [min(speed), max(speed)] if speed else None,
                "model": first.get("model"),
            }
        )
    return {"status": "complete", "phase": phase, "instances": instances, "aggregation": "Median of three block medians per instance; no CI", "uncertainty": "Raw blocks retained as fixed repeated measurements"}


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with resolve(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False) if isinstance(value, (dict, list)) else "" if value is None else value for key, value in row.items()})


def run_phase(phase, device, job):
    check_selection_seal()
    if phase == "training":
        rows = planned_training_grid()
        prepared = {dataset: prepare_training(dataset) for dataset in DATASETS}
        output_path = HERE / "training_cost_probe.json"
    else:
        rows = planned_inference_grid()
        prepared = {dataset: prepare_inference(dataset) for dataset in DATASETS}
        output_path = HERE / "inference_cost_rows.json"
    if output_path.exists():
        raise FileExistsError("Cost phase already completed; preserve fixed rows")
    result_rows = run_rows(phase, rows, prepared, device, job)
    result = {
        "schema": f"matched_learned_ed_v18_{phase}_cost_1",
        "status": "complete",
        "phase": phase,
        "device": device,
        "environment": environment(),
        "rows": result_rows,
        "expected_rows": len(rows),
        "input_bindings": {dataset: {key: prepared[dataset][key] for key in prepared[dataset] if key.endswith("sha256") or key == "trainval"} for dataset in DATASETS},
        "scope": "Same selected campaign; no optimizer step; no TEST for cost",
    }
    save_json(output_path, result)
    summary = summarize(result_rows, phase)
    save_json(HERE / f"{phase}_cost_summary.json", summary)
    comparison_rows = [{"row_type": "summary", **row} for row in summary["instances"]]
    comparison_rows.extend({"row_type": "block", **row} for row in result_rows)
    write_csv(HERE / f"{phase}_cost_comparison.csv", comparison_rows)
    return result


def verify_all():
    result = {"status": "pass", "available": {}}
    for name, expected in (("training_cost_probe.json", EXPECTED_TRAINING_ROWS), ("inference_cost_rows.json", EXPECTED_INFERENCE_ROWS)):
        path = HERE / name
        if not path.exists():
            result["available"][name] = {"status": "missing_unexecuted", "expected_rows": expected}
            continue
        doc = read_json(path)
        if doc.get("status") != "complete" or len(doc.get("rows", [])) != expected:
            raise RuntimeError("Cost artifact row count/status changed: " + name)
        result["available"][name] = {"status": "complete", "expected_rows": expected, "artifact": artifact(path)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("schema", "training", "inference", "verify"), required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.phase == "schema":
        print(json.dumps(schema_contract(), ensure_ascii=False, indent=2))
        return
    configure()
    if args.phase == "verify":
        result = verify_all()
    else:
        with Job("gpu" if args.device == "cuda" else "cpu", f"v18_{args.phase}_cost", metadata={"phase": args.phase, "device": args.device}) as job:
            result = {"status": run_phase(args.phase, args.device, job)["status"]}
    print(json.dumps({"phase": args.phase, "result": result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
