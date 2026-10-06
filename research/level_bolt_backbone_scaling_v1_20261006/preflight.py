"""VAL-only checks for the fixed same-family Chronos-Bolt allocation comparison."""

import gc
import hashlib
import json
from pathlib import Path
import re
import time
import traceback

import numpy as np
import torch

from baselines import baseline_rows, inputs, load_baseline, load_data
from bolt_models import BoltForecast, state_hash
from runtime import (CACHE, DATASETS, HERE, LIMITS, Job, artifact, budget_snapshot,
                     configure, read_json, save_json)


REPLAY = {"atol": 1e-6, "rtol": 0.0}
BATCH = {"atol": 1e-5, "rtol": 1e-4}


def compare(actual, expected, tolerance, label):
    actual = torch.as_tensor(actual).detach().cpu().float()
    expected = torch.as_tensor(expected).detach().cpu().float()
    if actual.shape != expected.shape or not actual.numel():
        raise ValueError(f"{label}: invalid shape {actual.shape} versus {expected.shape}")
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError(label + ": nonfinite value")
    error = (actual.double() - expected.double()).abs()
    allowed = tolerance["atol"] + tolerance["rtol"] * expected.double().abs()
    violations = error > allowed
    positive = expected.double().abs() > 0
    relative = error[positive] / expected.double().abs()[positive]
    result = {"label": label, "shape": list(actual.shape), **tolerance,
              "max_absolute_error": float(error.max()),
              "max_relative_error_nonzero_reference": float(relative.max()) if relative.numel() else None,
              "violating_elements": int(violations.sum()), "elements": error.numel(),
              "status": "failed" if violations.any() else "pass"}
    if violations.any():
        failure = RuntimeError(label + ": parity failed")
        failure.check_result = result
        raise failure
    return result


def cleanup():
    gc.collect()
    torch.cuda.synchronize()
    if hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
        torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def assert_frozen(module):
    if any(child.training for child in module.modules()) or any(p.requires_grad for p in module.parameters()):
        raise ValueError("Inference modules must stay in eval mode with requires_grad=False")


def assert_output(output, batch, channels):
    if (tuple(output.shape) != (batch, 48, channels) or output.dtype != torch.float32
            or output.device.type != "cpu" or not output.is_contiguous()
            or output.requires_grad or not torch.isfinite(output).all()):
        raise ValueError("Output shape/dtype/device/contiguity/finiteness/frozen contract failed")


def fixed_inputs(dataset):
    data = load_data(dataset, include_test=False)
    unit = read_json(HERE / "input_contract.json")["units"][dataset]
    origins = np.asarray(data["val_origins"], dtype=np.int64)
    declared = unit["origins"]["val"]
    if (len(origins) < 24 or len(origins) != declared["count"] or np.any(np.diff(origins) <= 0)
            or not np.array_equal(origins, np.asarray(declared["values"], dtype=np.int64))):
        raise ValueError("Canonical chronological VAL origin list changed")
    if (unit["l"], unit["h"]) != (512, 48):
        raise ValueError("The fixed L512/H48 data contract changed")
    bounds = unit["bounds"]["val"]
    if any(int(o) < bounds[0] or int(o) + 48 > bounds[1] for o in origins[:24]):
        raise ValueError("VAL targets are outside the declared half-open split")
    four = origins[:4]
    x = inputs(data, four)
    if tuple(x.shape) != (4, 512, unit["c"]) or x.dtype != torch.float32 or not torch.isfinite(x).all():
        raise ValueError("Canonical TRAIN-standardized/imputed past input is invalid")
    return data, four, x


def verify_model_binding(key, manifest):
    if manifest["id"] != "amazon/chronos-bolt-" + key.lower() or not re.fullmatch("[0-9a-f]{40}", manifest["revision"]):
        raise ValueError("Official model ID or immutable revision binding is missing")
    declared = manifest.get("files")
    if isinstance(declared, list):
        declared = {Path(entry["path"]).name: entry for entry in declared}
    if not isinstance(declared, dict) or not declared:
        raise ValueError("Pinned model config/weight file hashes are missing")
    checkpoint = Path(manifest["checkpoint_dir"]).resolve(strict=True)
    required = {"config.json", *(p.name for p in checkpoint.glob("*.safetensors"))}
    if len(required) < 2 or not required.issubset(declared):
        raise ValueError("The sealed files do not bind config and every native weight file")
    checked = []
    for name, entry in declared.items():
        if Path(name).name != name:
            raise ValueError("Model manifest file names must be local checkpoint basenames")
        source = artifact(entry["path"], entry["sha256"])
        loaded = artifact(checkpoint / name, entry["sha256"])
        checked.append({"name": name, "source": source, "loaded_checkpoint_file": loaded})
    return checked


def native_quantile_probe(wrapper, x):
    records, selected = [], []

    def before(module, args, kwargs):
        if args or set(kwargs) != {"context"}:
            raise ValueError("Native Bolt must receive only context; no future arrays or target")
        context = kwargs["context"].detach().cpu()
        expected = x.transpose(1, 2).reshape(x.shape[0] * x.shape[2], 512)
        if not torch.equal(context, expected):
            raise ValueError("Actual native context lost origin/channel row order")
        records.append({"context_shape": list(context.shape), "context_dtype": str(context.dtype),
                        "row_order_bit_equal": True, "actual_argument_names": sorted(kwargs),
                        "target_provided": False, "future_covariates_provided": False})

    def after(module, args, output):
        prediction = output.quantile_preds
        records[-1].update(native_quantile_shape=list(prediction.shape), native_dtype=str(prediction.dtype),
                           quantile_index=wrapper.median_index,
                           actual_quantile=float(module.quantiles[wrapper.median_index].detach().cpu()))
        selected.append(prediction[:, wrapper.median_index, :48].detach().cpu().clone())

    handles = [wrapper.module.register_forward_pre_hook(before, with_kwargs=True),
               wrapper.module.register_forward_hook(after)]
    try:
        output = wrapper(x)
    finally:
        for handle in handles:
            handle.remove()
    if len(records) != 1 or records[0]["actual_quantile"] != 0.5:
        raise ValueError("q=0.5 or one complete native B*C forward is not established")
    native = selected[0].reshape(x.shape[0], x.shape[2], 48).transpose(1, 2).contiguous()
    quantile_check = compare(output, native, REPLAY, "actual native q=0.5 selection and channel restoration")
    return output, {"status": "pass", "records": records, "selection_parity": quantile_check}


def timed_calls(wrapper, x, job):
    seconds = []
    for _ in range(3):
        job.check_limits()
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = wrapper(x)
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - started)
        assert_output(output, len(x), x.shape[2])
    return {"call_seconds": seconds, "median_s": float(np.median(seconds)), "max_s": max(seconds),
            "calls": 3, "batch_origins": len(x),
            "scope": "VAL execution-time estimate; standardized CPU input through full H48 CPU output",
            "scientific_cost_measurement": False}


def stored_val(row, origins):
    entry = row.get("val_prediction")
    if entry is None:
        return None
    artifact(entry["path"], entry["sha256"])
    with np.load(entry["path"], allow_pickle=False) as archive:
        values, available = archive[entry["key"]], archive[entry["origins_key"]]
        indices = []
        for origin in origins:
            matches = np.flatnonzero(available == origin)
            if len(matches) != 1:
                raise ValueError("Stored VAL prediction must bind each canonical origin exactly once")
            indices.append(int(matches[0]))
        return values[indices].copy()


@torch.inference_mode()
def original_forward(wrapper, x):
    device_x = x.to(wrapper.device)
    value = (wrapper.module.components(device_x)[0] if wrapper.row.get("output_component") == "main"
             else wrapper.module(device_x))
    return value.contiguous().cpu()


def baseline_probe(report, dataset, row, origins, x, job):
    job.heartbeat({"phase": "baseline_val_restore", "dataset": dataset, "id": row["id"]})
    started = time.perf_counter()
    wrapper = load_baseline(row, device="cuda")
    unit = {"dataset": dataset, "id": row["id"], "method": row["method"], "arm": row["arm"],
            "seed": row["seed"], "origins": origins.tolist(), "checks": [],
            "load_seconds": time.perf_counter() - started, "receipt": wrapper.receipt()}
    report["baselines"].append(unit)
    assert_frozen(wrapper.module)
    unit["state_sha256_before"] = state_hash(wrapper.module)
    output = wrapper(x)
    assert_output(output, 4, x.shape[2])
    unit["checks"].append(compare(output, original_forward(wrapper, x), BATCH,
                                  "original selected-model forward versus CPU-output wrapper"))
    unit["checks"].append(compare(wrapper(x), output, REPLAY, "selected baseline same-path replay"))
    saved = stored_val(row, origins)
    unit["saved_val_parity"] = (compare(output, saved, BATCH, "restored baseline versus stored VAL")
                                if saved is not None else
                                {"status": "unavailable", "reason": row.get("val_prediction_missing_reason"),
                                 "substitution_used": False,
                                 "active_original_wrapper_parity": "pass"})
    unit["timings"] = {str(b): timed_calls(wrapper, x[:b], job) for b in (1, 4)}
    for batch in (1, 4):
        report["timings"].append({"dataset": dataset, "arm": row["arm"], "id": row["id"],
                                  "batch": batch, **unit["timings"][str(batch)]})
    unit["state_sha256_after"] = state_hash(wrapper.module)
    assert_frozen(wrapper.module)
    if unit["state_sha256_before"] != unit["state_sha256_after"]:
        raise ValueError("Baseline parameters/buffers changed during inference")
    unit["frozen_state"] = {"status": "pass", "eval_mode": True, "requires_grad": False}
    return wrapper, output, unit


def small_anchor_checks(report, prepared, models, job):
    binding = verify_model_binding("SMALL", models["SMALL"])
    job.heartbeat({"phase": "small_anchor_load"})
    started = time.perf_counter()
    wrapper = BoltForecast(models["SMALL"]["checkpoint_dir"])
    before = state_hash(wrapper.module)
    record = {"model": "SMALL", "id": models["SMALL"]["id"], "revision": models["SMALL"]["revision"],
              "load_seconds": time.perf_counter() - started, "receipt": wrapper.receipt(),
              "verified_checkpoint_files": binding, "state_sha256_before": before, "units": []}
    report["bolt_models"].append(record)
    for dataset in DATASETS:
        data, origins, x = prepared[dataset]
        rows = [row for row in baseline_rows(dataset) if row["method"] == "F0"]
        if len(rows) != 1 or rows[0]["arm"] != "BOLT_SMALL_F0":
            raise ValueError("Exactly one original F0 anchor is required for every dataset")
        original, expected, baseline = baseline_probe(report, dataset, rows[0], origins, x, job)
        if state_hash(original.module.backbone) != before:
            raise ValueError("Original F0 backbone tensors differ from the pinned SMALL checkpoint")
        unit = {"dataset": dataset, "origins": origins.tolist(), "baseline_id": rows[0]["id"],
                "original_instance_norm_forward": original.module.backbone.instance_norm.forward.__func__.__name__,
                "generalized_instance_norm_forward": wrapper.module.instance_norm.forward.__func__.__name__,
                "normalization_treatment": "native inference; original Peacock patch changes the zero-variance backward route only",
                "checks": []}
        record["units"].append(unit)
        output, unit["native_quantile"] = native_quantile_probe(wrapper, x)
        assert_output(output, 4, data["_C"])
        unit["checks"].append(compare(output, expected, BATCH, "generalized SMALL versus original F0 wrapper"))
        unit["checks"].append(compare(wrapper(x[:1]), original(x[:1]), BATCH,
                                      "generalized SMALL versus original F0 wrapper B1"))
        unit["checks"].append(compare(wrapper(x), output, REPLAY, "generalized SMALL same-path replay"))
        unit["status"] = "pass"
        del original
        cleanup()
    record["state_sha256_after"] = state_hash(wrapper.module)
    assert_frozen(wrapper.module)
    if before != record["state_sha256_after"]:
        raise ValueError("Generalized SMALL state changed during anchor checks")
    record["frozen_state"] = {"status": "pass", "eval_mode": True, "requires_grad": False}
    record["status"] = "pass"
    del wrapper
    cleanup()
    report["small_anchor_all_datasets_passed"] = True


def mini_tiny_checks(report, prepared, models, job):
    if report.get("small_anchor_all_datasets_passed") is not True:
        raise ValueError("SMALL anchor parity must pass before Mini/Tiny execution")
    for key in ("MINI", "TINY"):
        binding = verify_model_binding(key, models[key])
        job.heartbeat({"phase": "bolt_load", "model": key})
        started = time.perf_counter()
        wrapper = BoltForecast(models[key]["checkpoint_dir"])
        before = state_hash(wrapper.module)
        record = {"model": key, "id": models[key]["id"], "revision": models[key]["revision"],
                  "load_seconds": time.perf_counter() - started, "receipt": wrapper.receipt(),
                  "verified_checkpoint_files": binding, "state_sha256_before": before, "units": []}
        report["bolt_models"].append(record)
        replays = {}
        for dataset in DATASETS:
            print(f"preflight {key} {dataset} VAL4", flush=True)
            job.heartbeat({"phase": "bolt_val_isolation", "model": key, "dataset": dataset})
            data, origins, x = prepared[dataset]
            unit = {"dataset": dataset, "origins": origins.tolist(), "columns": data["columns"].tolist(),
                    "input_shape": list(x.shape), "input_finite": True, "checks": []}
            record["units"].append(unit)
            assert_frozen(wrapper.module)
            output, unit["native_quantile"] = native_quantile_probe(wrapper, x)
            assert_output(output, 4, data["_C"])
            unit["output_contract"] = {"status": "pass", "shape": list(output.shape),
                                       "dtype": str(output.dtype), "device": "cpu", "finite": True,
                                       "contiguous": True, "requires_grad": False}
            solo = torch.cat([wrapper(x[i:i + 1]) for i in range(4)])
            unit["checks"].append(compare(output, solo, BATCH, "all four solo origins versus B4"))
            unit["checks"].append(compare(wrapper(x), output, REPLAY, "same pinned-checkpoint input replay"))
            permutation = torch.tensor([2, 0, 3, 1])
            unit["checks"].append(compare(wrapper(x[permutation])[torch.argsort(permutation)], output, BATCH,
                                          "origin permutation restored"))
            channel_permutation = torch.arange(x.shape[2] - 1, -1, -1)
            unit["checks"].append(compare(wrapper(x[:, :, channel_permutation])[:, :, torch.argsort(channel_permutation)],
                                          output, BATCH, "channel permutation restored"))
            wave = torch.sin(torch.arange(512, dtype=torch.float32) * 0.071) + 0.25
            other_origin = x.clone()
            other_origin[3] += wave[:, None]
            unit["checks"].append(compare(wrapper(other_origin)[:3], output[:3], BATCH,
                                          "changed origin 3 cannot affect origins 0/1/2"))
            other_channel = x.clone()
            other_channel[0, :, 1] += wave
            unaffected = torch.arange(x.shape[2]) != 1
            unit["checks"].append(compare(wrapper(other_channel)[0, :, unaffected], output[0, :, unaffected], BATCH,
                                          "changed channel 1 cannot affect other channels at origin 0"))
            changed_future = dict(data)
            changed_future["x"] = data["x"].copy()
            origin = int(origins[0])
            changed_future["x"][origin:origin + 48] += np.float32(100.0)
            future_past = inputs(changed_future, origins[:1])
            if not torch.equal(future_past, x[:1]):
                raise ValueError("Changing actual future rows altered the constructed past")
            unit["checks"].append(compare(wrapper(future_past), solo[:1], REPLAY,
                                          "actual future-array change leaves the prediction unchanged"))
            unit["future_barrier"] = {"status": "pass", "actual_future_rows_modified": 48,
                                      "past_input_bit_identical": True, "future_inputs_provided": False,
                                      "scope": "origin 0; future rows may be past for later origins"}
            unit["timings"] = {str(b): timed_calls(wrapper, x[:b], job) for b in (1, 4)}
            for batch in (1, 4):
                report["timings"].append({"dataset": dataset, "arm": "BOLT_" + key + "_F0", "id": key,
                                          "batch": batch, **unit["timings"][str(batch)]})
            assert_frozen(wrapper.module)
            unit["status"] = "pass"
            replays[dataset] = output
        record["state_sha256_after"] = state_hash(wrapper.module)
        if before != record["state_sha256_after"]:
            raise ValueError(key + " frozen parameters/buffers changed during VAL checks")
        record["frozen_state"] = {"status": "pass", "eval_mode": True, "requires_grad": False}
        del wrapper
        cleanup()
        job.heartbeat({"phase": "pinned_reload", "model": key})
        started = time.perf_counter()
        restored = BoltForecast(models[key]["checkpoint_dir"])
        record["reload_seconds"] = time.perf_counter() - started
        record["reload_checks"] = []
        record["reload_state_sha256_before"] = state_hash(restored.module)
        if record["reload_state_sha256_before"] != before:
            raise ValueError("Pinned reload state differs from original model state")
        for dataset in DATASETS:
            record["reload_checks"].append(compare(restored(prepared[dataset][2]), replays[dataset], REPLAY,
                                                     "same pinned checkpoint reload " + dataset))
        record["reload_state_sha256_after"] = state_hash(restored.module)
        assert_frozen(restored.module)
        if record["reload_state_sha256_after"] != before:
            raise ValueError("Reloaded frozen state changed during replay")
        record["status"] = "pass"
        del restored
        cleanup()


def level_baseline_checks(report, prepared, job):
    for dataset in DATASETS:
        _, origins, x = prepared[dataset]
        rows = [row for row in baseline_rows(dataset) if row["method"] == "LEVEL"]
        if len(rows) != 2 or any(row["arm"] != "BOLT_SMALL_LEVEL" for row in rows):
            raise ValueError("Both original selected LEVEL seeds are required")
        for row in rows:
            wrapper, _, _ = baseline_probe(report, dataset, row, origins, x, job)
            del wrapper
            cleanup()


def estimate_campaign(report):
    contract = read_json(HERE / "input_contract.json")["units"]
    cost_seconds, accuracy_seconds, cost_calls = 0.0, 0.0, 0
    for timing in report["timings"]:
        batch = timing["batch"]
        passes = 24 // batch
        calls = 3 * (10 + 10 * passes)
        cost_calls += calls
        cost_seconds += calls * timing["max_s"]
        if timing["arm"] in ("BOLT_MINI_F0", "BOLT_TINY_F0") and batch == 4:
            unit = contract[timing["dataset"]]
            origins = unit["origins"]["test_a"]["count"] + unit["origins"]["test_b"]["count"]
            accuracy_seconds += int(np.ceil(origins / 4)) * timing["max_s"]
    sentinel_seconds = sum(6 * (10 + 10 * 24) * row["max_s"] for row in report["timings"]
                           if row["arm"] == "BOLT_SMALL_F0" and row["batch"] == 1)
    loads = [row["load_seconds"] for row in report["baselines"] + report["bolt_models"]]
    loading_reserve = 108 * max(loads)
    remaining = LIMITS["gpu_seconds"] - budget_snapshot()["gpu_seconds"]
    conservative = cost_seconds + sentinel_seconds + accuracy_seconds + loading_reserve
    return {"status": "estimated_unverified_campaign_runtime", "planned_primary_cost_wrapper_calls": cost_calls,
            "cost_forward_seconds_max_val4_probe": cost_seconds,
            "sentinel_forward_seconds_upper_reserve": sentinel_seconds,
            "new_accuracy_forward_seconds_max_val4_probe": accuracy_seconds,
            "model_load_seconds_upper_reserve": loading_reserve,
            "model_load_reserve_basis": "108 loads times slowest measured constructor; primary90 and sentinel18 rows",
            "conservative_gpu_forward_and_load_estimate_seconds": conservative,
            "gpu_remaining_seconds_at_estimate": remaining,
            "estimated_within_remaining_gpu_budget": conservative < remaining,
            "does_not_guarantee_runtime": True,
            "test_access": "origin counts from sealed metadata only; no TEST array, prediction or score access",
            "excluded_unknown_overhead": ["serialization", "scoring", "telemetry/cleanup", "failure/retry"],
            "oom_policy": "Primary B4 OOM remains recorded; no replacement B2 grid"}


def main():
    configure()
    ledger = read_json(HERE / "ledger.json")
    attempt = 1 + sum(row["category"] == "gpu" and row["label"] == "preflight" for row in ledger["jobs"])
    output = HERE / f"preflight_attempt{attempt:02d}.json"
    if output.exists():
        raise RuntimeError("Preflight attempt output already exists")
    report = {"schema": "level_bolt_scaling_preflight_v1", "status": "running", "attempt": attempt,
              "scope": "first four existing chronological VAL origins per dataset; TRAIN/VAL only; no new fit",
              "tolerances": {"same_path": REPLAY, "batch_merge": BATCH},
              "artifact_scope": {"write_directories": [str(HERE), str(CACHE)],
                                 "old_campaigns": "read-only", "test_arrays_opened": False},
              "code": artifact(__file__), "bolt_models_code": artifact(HERE / "bolt_models.py"),
              "bindings": {name: artifact(HERE / name) for name in
                           ("input_contract.json", "reuse_manifest.json", "model_manifest.json")},
              "bolt_models": [], "baselines": [], "timings": [],
              "small_anchor_all_datasets_passed": False,
              "conditionals": {"small_anchor_failure": "STOP_DEBUG before Mini/Tiny TEST",
                               "quantile_or_revision_or_order_or_future_or_frozen_failure": "STOP_DEBUG",
                               "stored_VAL_absent": "explicit unavailable; active original-wrapper parity still required",
                               "B4_OOM": "STOP_RESOURCE in preflight; retain failure and do not change primary batch",
                               "budget_estimate_exceeds_remaining": "STOP_RESOURCE; estimate is not an authorization"}}
    try:
        with Job("gpu", "preflight", reserve_s=600,
                 metadata={"fit_count": 0, "optimizer_updates": 0, "uses_test": False, "attempt": attempt}) as job:
            save_json(output, report)
            prepared = {dataset: fixed_inputs(dataset) for dataset in DATASETS}
            report["input_units"] = [{"dataset": dataset, "origins": origins.tolist(),
                                      "shape": list(x.shape), "past_sha256": hashlib.sha256(x.numpy().tobytes()).hexdigest(),
                                      "trainval_artifact": read_json(HERE / "input_contract.json")["units"][dataset]["trainval"]}
                                     for dataset, (_, origins, x) in prepared.items()]
            models = read_json(HERE / "model_manifest.json")["models"]
            small_anchor_checks(report, prepared, models, job)
            save_json(output, report)
            mini_tiny_checks(report, prepared, models, job)
            save_json(output, report)
            level_baseline_checks(report, prepared, job)
            report["campaign_runtime_estimate"] = estimate_campaign(report)
            if not report["campaign_runtime_estimate"]["estimated_within_remaining_gpu_budget"]:
                raise RuntimeError("STOP_RESOURCE: estimated remaining campaign work exceeds the GPU budget")
            if len(report["baselines"]) != 9 or sum(len(row["units"]) for row in report["bolt_models"]) != 9:
                raise ValueError("Required preflight units are missing")
            report.update(status="pass", fits=0, optimizer_updates=0, test_scoring_performed=False,
                          baseline_instances_checked=9, mini_tiny_dataset_units_checked=6)
            job.heartbeat({"phase": "preflight_complete"})
    except BaseException as exc:
        report["status"] = "failed"
        report["stop_class"] = "STOP_RESOURCE" if isinstance(exc, torch.cuda.OutOfMemoryError) or "STOP_RESOURCE" in str(exc) else "STOP_DEBUG"
        report["error"], report["traceback"] = repr(exc), traceback.format_exc()
        if hasattr(exc, "check_result"):
            report["failed_check"] = exc.check_result
        save_json(output, report)
        save_json(HERE / "preflight_checks.json", report)
        raise
    save_json(output, report)
    save_json(HERE / "preflight_checks.json", report)
    print(json.dumps({"status": report["status"], "small_anchor_all_datasets_passed": True,
                      "mini_tiny_dataset_units": 6, "baseline_instances": 9,
                      "campaign_runtime_estimate": report["campaign_runtime_estimate"]}), flush=True)


if __name__ == "__main__":
    main()
