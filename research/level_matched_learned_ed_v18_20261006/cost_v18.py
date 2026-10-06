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
    inventory = [
        {"kind": kind, "name": name, "shape": list(value.shape), "dtype": str(value.dtype),
         "numel": value.numel(), "element_bytes": value.element_size(),
         "requires_grad": bool(value.requires_grad)}
        for kind, values in (("parameter", model.named_parameters()), ("buffer", model.named_buffers()))
        for name, value in values
    ]
    parameters = [row for row in inventory if row["kind"] == "parameter"]
    parameter_bytes = sum(row["numel"] * row["element_bytes"] for row in parameters)
    buffer_bytes = sum(row["numel"] * row["element_bytes"] for row in inventory if row["kind"] == "buffer")
    return {
        "deployed_parameters": sum(row["numel"] for row in parameters),
        "parameter_bytes": parameter_bytes,
        "buffer_bytes": buffer_bytes,
        "deployment_tensor_bytes": parameter_bytes + buffer_bytes,
        "trainable_parameters": sum(row["numel"] for row in parameters if row["requires_grad"]),
        "tensor_inventory": inventory,
    }


def state_snapshot(model):
    return {
        "sha256": state_hash(model),
        "eval_mode": not model.training,
        "backbone_eval_mode": not model.backbone.training,
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
        "backbone_eval_mode": after["backbone_eval_mode"],
        "trainable_parameters": after["trainable_parameters"],
        "expected_trainable_parameters": expect_trainable,
        "optimizer_present": False,
        "hash_scope": "parameters_and_buffers_state_dict_only",
    }
    if not result["unchanged"] or result["trainable_parameters"] != expect_trainable or not result["backbone_eval_mode"]:
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
        extra = {"deployment_receipt": receipt}
    else:
        original_row = dict(member)
        original_row["method"] = original_row.get("original_method", "LEVEL")
        model = load_original(original_row, device=device, deploy=not training)
        residual_names = {"residual." + name for name, _ in model.residual.named_parameters()}
        fitted = sum(param.numel() for param in model.residual.parameters())
        if fitted != 17920:
            raise RuntimeError("Literal LEVEL residual parameter count changed")
        for name, param in model.named_parameters():
            param.requires_grad_(training and name in residual_names)
        model.train(training)
        model.backbone.eval()
        extra = {"historically_fitted_parameter_names": sorted(residual_names)}
    info = model_bytes(model)
    if info["trainable_parameters"] != (fitted if training else 0):
        raise RuntimeError("Cost probe trainable set differs from the selected architecture")
    return model, {**info, **extra, "original_fitted_parameter_count": fitted,
                   "training_trainable_parameter_count": fitted,
                   "probe_trainable_parameter_count": info["trainable_parameters"],
                   "inference_trainable_parameter_count": 0}


def training_step(model, method, prepared):
    import torch
    from model_v18 import compute_loss

    device = next(model.parameters()).device
    x = prepared["inputs"].to(device)
    y = torch.as_tensor(prepared["target"], device=device)
    mask = torch.as_tensor(prepared["mask"], device=device)
    prediction = model(x)
    loss = compute_loss(prediction, y, mask)
    loss.backward()
    return loss.detach()


def zero_grad(model):
    for param in model.parameters():
        param.grad = None


def gradient_diagnostics(model, loss):
    import torch

    names = [name for name, param in model.named_parameters() if param.requires_grad]
    gradients = {name: {"present": param.grad is not None,
                        "finite": bool(torch.isfinite(param.grad).all()) if param.grad is not None else False,
                        "nonzero": bool(torch.count_nonzero(param.grad)) if param.grad is not None else False}
                 for name, param in model.named_parameters() if param.requires_grad}
    frozen_clean = all(param.grad is None for param in model.parameters() if not param.requires_grad)
    result = {"trainable_parameter_names": names, "gradients": gradients,
              "frozen_parameters_grad_none": frozen_clean, "loss_finite": bool(torch.isfinite(loss)),
              "scope": "First warmup only; outside timed forward/loss/backward region"}
    if not result["loss_finite"] or not frozen_clean or not all(row["present"] and row["finite"] for row in gradients.values()):
        raise RuntimeError("STOP_DEBUG: nonfinite loss/gradients or frozen parameter gradient")
    return result


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
    gradients = None
    for index in range(WARMUP):
        job.check_limits()
        zero_grad(model)
        loss = training_step(model, method, prepared)
        if index == 0:
            gradients = gradient_diagnostics(model, loss)
        elif not bool(torch.isfinite(loss)):
            raise RuntimeError("STOP_DEBUG: nonfinite training-probe loss")
        zero_grad(model)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    times = []
    for repeat in range(REPETITIONS):
        job.heartbeat({"phase": "training_cost", "repeat": repeat})
        zero_grad(model)
        if device == "cuda":
            torch.cuda.synchronize()
        tick = time.perf_counter()
        loss = training_step(model, method, prepared)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - tick)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("STOP_DEBUG: nonfinite training-probe loss")
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
        "gradient_diagnostics": gradients,
        "timed_region": "CPU input/target/mask to device, one forward, identical observed-mask channel-macro MSE, backward; diagnostics and zero_grad outside timing",
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


def input_binding(prepared):
    return {key: value for key, value in prepared.items() if key.endswith("sha256") or key == "trainval"}


def source_bindings(member):
    result = {name: artifact(HERE / filename) for name, filename in
              (("cost_source", "cost_v18.py"), ("source_seal", "source_seal.json"),
               ("selection_seal", "selection_seal.json"), ("selection", "selection.json"),
               ("parent_manifest", "parent_reuse_manifest.json"))}
    checkpoint = member["checkpoint"]
    result["checkpoint"] = artifact(checkpoint["path"], checkpoint["sha256"])
    if member["method"] == LEVEL:
        ref = member["restore_module_artifact"]
        result["restore_source"] = artifact(ref["path"], ref["sha256"])
    else:
        result["restore_source"] = artifact(HERE / "model_v18.py")
    source_seal = read_json(HERE / "source_seal.json")
    selection_seal = read_json(HERE / "selection_seal.json")
    required = [(result["cost_source"], source_seal), (result["parent_manifest"], source_seal),
                (result["selection"], selection_seal), (result["source_seal"], selection_seal)]
    if member["method"] == METHOD:
        required.extend(((result["restore_source"], source_seal), (result["checkpoint"], selection_seal)))
    for ref, seal in required:
        matches = [row for row in seal["artifacts"] if resolve(row["path"]).resolve() == resolve(ref["path"]).resolve()]
        if len(matches) != 1 or matches[0] != ref:
            raise RuntimeError("Cost checkpoint/source is not bound literally by the campaign seals")
    if (selection_seal.get("fit_count") != 12 or selection_seal.get("test_predictions_at_seal") != 0
            or selection_seal.get("prediction_ensemble") is not False):
        raise RuntimeError("Cost requires all twelve TRAIN/VAL fits jointly sealed before TEST")
    return result


def row_counts(rows, expected):
    return {"attempted_rows": len(rows), "successful_rows": sum(row["status"] == "complete" for row in rows),
            "oom_rows": sum(row["status"] == "oom" for row in rows),
            "failed_rows": sum(row["status"] == "failed" for row in rows), "expected_rows": expected}


def verify_model_receipt(info, config, prepared):
    inventory = info["tensor_inventory"]
    if not inventory or len({(row["kind"], row["name"]) for row in inventory}) != len(inventory):
        raise RuntimeError("Cost tensor inventory missing or duplicated")
    dtype_bytes = {"torch.float32": 4, "torch.float64": 8, "torch.float16": 2, "torch.bfloat16": 2,
                   "torch.int64": 8, "torch.int32": 4, "torch.int16": 2, "torch.int8": 1,
                   "torch.uint8": 1, "torch.bool": 1}
    for item in inventory:
        if (item["kind"] not in ("parameter", "buffer") or item["numel"] != int(np.prod(item["shape"]))
                or item["element_bytes"] != dtype_bytes.get(item["dtype"])):
            raise RuntimeError("Cost tensor shape/dtype/byte enumeration changed")
    parameters = {row["name"]: row for row in inventory if row["kind"] == "parameter"}
    fitted_names = {"encoder.weight", "decoder.weight"} if config["method"] == METHOD else {"residual.down.weight", "residual.up.weight"}
    fitted = 2 * prepared["C"] * prepared["K"] if config["method"] == METHOD else 17920
    training = config["row_type"] == "training_block"
    expected_names = fitted_names if training else set()
    actual_names = {name for name, row in parameters.items() if row["requires_grad"]}
    expected_shapes = {"encoder.weight": [prepared["K"], prepared["C"]], "decoder.weight": [prepared["C"], prepared["K"]]}
    if config["method"] == LEVEL:
        expected_shapes.update({"residual.down.weight": [32, 512], "residual.up.weight": [48, 32]})
    if any(parameters.get(name, {}).get("shape") != shape for name, shape in expected_shapes.items()):
        raise RuntimeError("Cost adapter tensor shapes differ from the selected model")
    if {name for name in parameters if not name.startswith("backbone.")} != set(expected_shapes):
        raise RuntimeError("Cost model contains an unapproved extra parameter")
    if actual_names != expected_names or any(row["dtype"] != "torch.float32" for row in parameters.values()):
        raise RuntimeError("Cost trainable names or parameter precision changed")
    backbone_count = sum(row["numel"] for name, row in parameters.items() if name.startswith("backbone."))
    if backbone_count != 47718016 or sum(parameters[name]["numel"] for name in fitted_names) != fitted:
        raise RuntimeError("Cost frozen Bolt-small/fitted parameter enumeration changed")
    values = {"deployed_parameters": sum(row["numel"] for row in parameters.values()),
              "parameter_bytes": sum(row["numel"] * row["element_bytes"] for row in parameters.values()),
              "buffer_bytes": sum(row["numel"] * row["element_bytes"] for row in inventory if row["kind"] == "buffer"),
              "trainable_parameters": fitted if training else 0,
              "original_fitted_parameter_count": fitted, "training_trainable_parameter_count": fitted,
              "probe_trainable_parameter_count": fitted if training else 0, "inference_trainable_parameter_count": 0}
    values["deployment_tensor_bytes"] = values["parameter_bytes"] + values["buffer_bytes"]
    if any(info.get(key) != value for key, value in values.items()):
        raise RuntimeError("Cost parameter/tensor byte totals changed")
    return expected_names


def verify_row(row, config, prepared, bindings, device):
    if any(row.get(key) != value for key, value in config.items()):
        raise RuntimeError("Cached cost row does not match its planned grid identity")
    if row.get("device") != device or row.get("input_binding") != input_binding(prepared) or row.get("source_bindings") != bindings:
        raise RuntimeError("Cost input/target/mask/origin/checkpoint/source binding changed")
    if row.get("status") not in ("complete", "oom", "failed"):
        raise RuntimeError("Cost row is not a preserved terminal attempt")
    if row["status"] != "complete":
        if not row.get("error") or not row.get("traceback") or (row["status"] == "oom" and row.get("oom_is_result") is not True):
            raise RuntimeError("Failed cost row lacks its actual failure receipt")
        return
    training = config["row_type"] == "training_block"
    expected_names = verify_model_receipt(row["model"], config, prepared)
    state = row["state_preservation"]
    expected_trainable = row["model"]["probe_trainable_parameter_count"]
    if (not state.get("unchanged") or state["before_sha256"] != state["after_sha256"]
            or len(state["before_sha256"]) != 64 or not state.get("backbone_eval_mode")
            or state.get("eval_mode") != (not training) or state.get("optimizer_present") is not False
            or state.get("trainable_parameters") != expected_trainable
            or state.get("expected_trainable_parameters") != expected_trainable):
        raise RuntimeError("Cost state/mode/trainable/no-optimizer evidence changed")
    if row.get("optimizer_updates", 0) != 0 or row.get("passes") != REPETITIONS:
        raise RuntimeError("Cost optimizer updates or repetition count changed")
    seconds = np.asarray(row["seconds_per_effective_batch" if training else "seconds_per_24_origins"], dtype=np.float64)
    if seconds.shape != (REPETITIONS,) or not np.isfinite(seconds).all() or np.any(seconds <= 0):
        raise RuntimeError("Cost requires ten positive finite raw timings")
    origins = TRAIN_BATCH if training else INFER_ORIGINS
    for raw_key, summary_key, values in (("raw_milliseconds_per_origin", "milliseconds_per_origin", seconds * 1000 / origins),
                                          ("raw_origins_per_second", "origins_per_second", origins / seconds)):
        raw = np.asarray(row[raw_key], dtype=np.float64)
        if raw.shape != (REPETITIONS,) or not np.allclose(raw, values, rtol=1e-12, atol=1e-12):
            raise RuntimeError("Cost raw timing unit conversion changed")
        expected = describe(values)
        if set(row[summary_key]) != set(expected) or any(not np.isclose(row[summary_key][key], value, rtol=1e-12, atol=1e-12) for key, value in expected.items()):
            raise RuntimeError("Cost raw timing median/range changed")
    if training:
        gradients = row["gradient_diagnostics"]
        if (row.get("warmup_calls") != WARMUP or set(gradients["trainable_parameter_names"]) != expected_names
                or set(gradients["gradients"]) != expected_names or not gradients["frozen_parameters_grad_none"]
                or not gradients["loss_finite"]
                or not all(item["present"] and item["finite"] for item in gradients["gradients"].values())):
            raise RuntimeError("Cost warmup/gradient isolation evidence changed")
    elif row.get("warmup_wrapper_calls") != WARMUP or row.get("timed_wrapper_calls") != REPETITIONS * (INFER_ORIGINS // config["batch"]):
        raise RuntimeError("Cost inference warmup/output call count changed")
    if device == "cuda":
        resident = row["resident_after_model_load_and_cleanup"]
        numbers = [resident["allocated_bytes"], resident["reserved_bytes"], row["peak_allocated_bytes"],
                   row["peak_reserved_bytes"], row["final_allocated_bytes"]]
        if (any(not isinstance(value, int) or value < 0 for value in numbers)
                or resident["allocated_bytes"] < row["model"]["deployment_tensor_bytes"]
                or resident["reserved_bytes"] < resident["allocated_bytes"]
                or row["peak_allocated_bytes"] < max(resident["allocated_bytes"], row["final_allocated_bytes"])
                or row["peak_reserved_bytes"] < max(resident["reserved_bytes"], row["peak_allocated_bytes"])):
            raise RuntimeError("Cost resident/peak allocated/reserved bytes are inconsistent")
    elif any(row.get(key) is not None for key in ("resident_after_model_load_and_cleanup", "peak_allocated_bytes", "peak_reserved_bytes", "final_allocated_bytes")):
        raise RuntimeError("CPU cost receipt cannot claim CUDA memory measurements")


def run_rows(phase, rows, prepared, device, job):
    import torch

    members = all_members()
    folder = HERE / "cost_rows_v18" / phase
    folder.mkdir(parents=True, exist_ok=True)
    output = []
    for config in rows:
        row_path = folder / (config["key"] + ".json")
        member = next(row for row in members[config["dataset"]] if row["id"] == config["id"])
        binding = source_bindings(member)
        if row_path.exists():
            record = read_json(row_path)
            verify_row(record, config, prepared[config["dataset"]], binding, device)
            if record["status"] == "failed":
                raise RuntimeError("Preserved failed cost row requires inspection; no automatic retry: " + str(row_path))
            output.append(record)
            continue
        model = None
        failure = None
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
                "device": device,
                "input_binding": input_binding(prepared[config["dataset"]]),
                "source_bindings": binding,
                "model": info,
                "state_preservation": state_preservation(before, after, expected_trainable),
                "telemetry_before": before_telemetry,
                "telemetry_after": telemetry() if device == "cuda" else None,
            }
            verify_row(record, config, prepared[config["dataset"]], binding, device)
            save_json(row_path, record)
            output.append(record)
        except torch.cuda.OutOfMemoryError as exc:
            record = {**config, "device": device, "input_binding": input_binding(prepared[config["dataset"]]),
                      "source_bindings": binding, "status": "oom", "finished_utc": time.time(), "job_id": job.id,
                      "pid": os.getpid(), "error": repr(exc), "traceback": traceback.format_exc(), "oom_is_result": True}
            save_json(row_path, record)
            output.append(record)
        except Exception as exc:
            failure = repr(exc)
            record = {**config, "device": device, "input_binding": input_binding(prepared[config["dataset"]]),
                      "source_bindings": binding, "status": "failed", "finished_utc": time.time(), "job_id": job.id,
                      "pid": os.getpid(), "error": failure, "traceback": traceback.format_exc()}
            save_json(row_path, record)
            output.append(record)
        finally:
            model = None
            cleanup(device)
        save_json(HERE / f"{phase}_cost_partial.json", {"status": "partial", "rows": output, **row_counts(output, len(rows))})
        if failure is not None:
            raise RuntimeError("Preserved failed cost attempt: " + failure)
    return output


def summarize(rows, phase):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["method"], row["id"], row.get("seed"), row.get("batch", TRAIN_BATCH))].append(row)
    instances = []
    for key, group in grouped.items():
        medians = [row["milliseconds_per_origin"]["median"] for row in group if row["status"] == "complete"]
        speed = [row["origins_per_second"]["median"] for row in group if row["status"] == "complete"]
        first = next((row for row in group if row["status"] == "complete"), group[0])
        instances.append(
            {
                "dataset": key[0],
                "method": key[1],
                "id": key[2],
                "seed": key[3],
                "batch": key[4],
                "phase": phase,
                "status": "complete" if len(medians) == BLOCKS else "oom" if any(row["status"] == "oom" for row in group) else "failed" if any(row["status"] == "failed" for row in group) else "partial",
                "attempted_blocks": len(group),
                "complete_blocks": len(medians),
                "median_block_median_ms": float(np.median(medians)) if medians else None,
                "median_block_median_origins_s": float(np.median(speed)) if speed else None,
                "block_median_ms_range": [min(medians), max(medians)] if medians else None,
                "block_median_origins_s_range": [min(speed), max(speed)] if speed else None,
                "model": first.get("model"),
            }
        )
    expected = EXPECTED_TRAINING_ROWS if phase == "training" else EXPECTED_INFERENCE_ROWS
    counts = row_counts(rows, expected)
    return {"status": "complete" if counts["successful_rows"] == expected else "incomplete",
            "phase": phase, **counts, "instances": instances,
            "aggregation": "Median of three block medians per instance; no CI",
            "uncertainty": "Raw blocks retained as fixed repeated measurements; incomplete instances are not full-grid cost results"}


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
        "status": "complete" if all(row["status"] == "complete" for row in result_rows) else "incomplete",
        "phase": phase,
        "device": device,
        "environment": environment(),
        "rows": result_rows,
        **row_counts(result_rows, len(rows)),
        "input_bindings": {dataset: input_binding(prepared[dataset]) for dataset in DATASETS},
        "row_artifacts": [artifact(HERE / "cost_rows_v18" / phase / (row["key"] + ".json")) for row in result_rows],
        "scope": "Same selected campaign; no optimizer step; no TEST for cost",
    }
    save_json(output_path, result)
    summary = summarize(result_rows, phase)
    save_json(HERE / f"{phase}_cost_summary.json", summary)
    comparison_rows = [{"row_type": "summary", **row} for row in summary["instances"]]
    comparison_rows.extend({"row_type": "block", **row} for row in result_rows)
    write_csv(HERE / f"{phase}_cost_comparison.csv", comparison_rows)
    return result


def verify_document(doc, phase, planned, prepared, bindings):
    expected = EXPECTED_TRAINING_ROWS if phase == "training" else EXPECTED_INFERENCE_ROWS
    rows = doc.get("rows", [])
    if (doc.get("phase") != phase or doc.get("device") not in ("cpu", "cuda") or len(rows) != expected
            or len(planned) != expected or [row.get("key") for row in rows] != [row["key"] for row in planned]
            or doc.get("input_bindings") != {dataset: input_binding(prepared[dataset]) for dataset in DATASETS}):
        raise RuntimeError("Cost artifact planned grid/input bindings changed")
    for row, config in zip(rows, planned):
        verify_row(row, config, prepared[config["dataset"]], bindings[(config["dataset"], config["id"])], doc["device"])
    counts = row_counts(rows, expected)
    status = "complete" if counts["successful_rows"] == expected else "incomplete"
    if doc.get("status") != status or any(doc.get(key) != value for key, value in counts.items()):
        raise RuntimeError("Cost attempted/successful/failure totals or status changed")
    return {"status": status, **counts}


def verify_all():
    result = {"status": "incomplete", "available": {}}
    names = (("training_cost_probe.json", "training", EXPECTED_TRAINING_ROWS),
             ("inference_cost_rows.json", "inference", EXPECTED_INFERENCE_ROWS))
    present = [name for name, _, _ in names if (HERE / name).exists()]
    if present:
        check_selection_seal()
        members = all_members()
        bindings = {(dataset, row["id"]): source_bindings(row) for dataset in DATASETS for row in members[dataset]}
    for name, phase, expected in names:
        path = HERE / name
        if not path.exists():
            result["available"][name] = {"status": "missing_unexecuted", "expected_rows": expected}
            continue
        doc = read_json(path)
        planned = planned_training_grid() if phase == "training" else planned_inference_grid()
        prepare = prepare_training if phase == "training" else prepare_inference
        prepared = {dataset: prepare(dataset) for dataset in DATASETS}
        status = verify_document(doc, phase, planned, prepared, bindings)
        if len(doc.get("row_artifacts", [])) != expected:
            raise RuntimeError("Cost row artifact receipts missing")
        for row, ref in zip(doc["rows"], doc["row_artifacts"]):
            expected_path = HERE / "cost_rows_v18" / phase / (row["key"] + ".json")
            if resolve(ref["path"]).resolve() != expected_path.resolve() or artifact(ref["path"], ref["sha256"]) != ref or read_json(expected_path) != row:
                raise RuntimeError("Cost row file differs from its full phase artifact")
        if read_json(HERE / f"{phase}_cost_summary.json") != summarize(doc["rows"], phase):
            raise RuntimeError("Cost summary aggregation/status differs from raw rows")
        result["available"][name] = {**status, "artifact": artifact(path)}
    if all(row["status"] == "complete" for row in result["available"].values()):
        result["status"] = "pass"
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
    if args.phase == "verify" and result["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
