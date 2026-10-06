"""Zero-update VAL checks before the fixed Chronos-2 TEST evaluation."""

import gc
import hashlib
import json
import time
import traceback

import numpy as np
import torch

from baselines import (BaselineForecast, baseline_rows, inputs, load_baseline,
                       load_data, source_module)
from c2_models import C2Forecast
from runtime import (DATASETS, HERE, LIMITS, Job, artifact, budget_snapshot,
                     configure, digest, read_json, save_json)


REPLAY = {"atol": 1e-6, "rtol": 0.0}
BATCH = {"atol": 1e-5, "rtol": 1e-4}


def compare(actual, expected, tolerance, label):
    actual = actual.detach().cpu().float()
    expected = torch.as_tensor(expected).detach().cpu().float()
    if actual.shape != expected.shape:
        raise ValueError(f"{label}: shape mismatch {actual.shape} vs {expected.shape}")
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError(label + ": nonfinite value")
    error = (actual.double() - expected.double()).abs()
    allowed = tolerance["atol"] + tolerance["rtol"] * expected.double().abs()
    violations = error > allowed
    positive = expected.double().abs() > 0
    relative = error[positive] / expected.double().abs()[positive]
    value = {"label": label, "shape": list(actual.shape), **tolerance,
             "max_absolute_error": float(error.max()),
             "max_relative_error_nonzero_reference": float(relative.max()) if relative.numel() else None,
             "violating_elements": int(violations.sum()), "elements": error.numel(),
             "status": "pass" if not violations.any() else "failed"}
    if violations.any():
        failure = RuntimeError(label + ": parity failed")
        failure.check_result = value
        raise failure
    return value


def state_hash(module):
    value = hashlib.sha256()
    for kind, tensors in (("parameter", module.named_parameters()), ("buffer", module.named_buffers())):
        for name, tensor in tensors:
            array = tensor.detach().cpu().contiguous().numpy()
            value.update(json.dumps([kind, name, str(array.dtype), list(array.shape)]).encode())
            value.update(array.tobytes())
    return value.hexdigest()


def cleanup():
    gc.collect()
    torch.cuda.synchronize()
    if hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
        torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def timed_calls(wrapper, x, job):
    values = []
    for repeat in range(3):
        job.check_limits()
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = wrapper(x)
        torch.cuda.synchronize()
        values.append(time.perf_counter() - started)
        if output.requires_grad or not torch.isfinite(output).all():
            raise ValueError("Inference must return finite no-grad outputs")
    return {"call_seconds": values, "median_s": float(np.median(values)),
            "max_s": max(values), "batch_origins": len(x), "calls": 3,
            "scope": "Small VAL execution-time probe, not a scientific accuracy result"}


def group_check(records, relation, batch, channels):
    expected = np.repeat(np.arange(batch), channels) if relation == "MV" else np.arange(batch * channels)
    equality = expected[:, None] == expected[None, :]
    if len(records) != 1:
        raise ValueError("Requested B*C rows must preserve the complete task batch in one model call")
    record = records[0]
    for name in ("group_ids", "encoder_group_ids"):
        if not np.array_equal(np.asarray(record[name]), expected):
            raise ValueError("Actual " + name + " do not match origin/channel task groups")
    for name in ("actual_allowed_any_time", "actual_allowed_last_time"):
        if not np.array_equal(np.asarray(record[name], dtype=bool), equality):
            raise ValueError("Actual group attention mask permits incorrect information sharing")
    if record["context_shape"] != [batch * channels, 512]:
        raise ValueError("Actual model context has changed axes")
    if record["future_target_provided"] or record["known_future_values"] != 0:
        raise ValueError("Future target or finite future covariate reached the model")
    return {"status": "pass", "relation": relation, "batch": batch, "channels": channels,
            "allowed_row_pairs": int(equality.sum()),
            "cross_origin_allowed_pairs": 0,
            "within_origin_information": "all channel pairs" if relation == "MV" else "each channel independent",
            "actual_model_records": records}


def fixed_inputs(dataset):
    data = load_data(dataset, include_test=False)
    origins = np.asarray(data["val_origins"], dtype=np.int64)
    if len(origins) < 24 or np.any(np.diff(origins) <= 0):
        raise ValueError("Expected at least 24 strictly chronological VAL origins")
    four = origins[:4]
    x = inputs(data, four)
    unit = read_json(HERE / "input_contract.json")["units"][dataset]
    bounds = unit["bounds"]["val"]
    if any(int(o) < bounds[0] or int(o) + 48 > bounds[1] for o in four):
        raise ValueError("VAL targets are outside the declared half-open split")
    if tuple(x.shape) != (4, 512, unit["c"]) or not torch.isfinite(x).all():
        raise ValueError("Canonical standardized/imputed past input is invalid")
    return data, four, x


def c2_checks(report, prepared, job):
    models = read_json(HERE / "model_manifest.json")["models"]
    for key in ("C2", "C2_SMALL"):
        job.heartbeat({"phase": "c2_load", "model": key})
        load_started = time.perf_counter()
        wrapper = C2Forecast(models[key]["snapshot"], relation="MV", device="cuda")
        load_s = time.perf_counter() - load_started
        before = state_hash(wrapper.module)
        model_report = {"model": key, "revision": models[key]["revision"],
                        "receipt": wrapper.receipt(), "load_seconds": load_s,
                        "state_sha256_before": before, "units": []}
        report["c2_models"].append(model_report)
        replay_values = {}
        for dataset in DATASETS:
            data, origins, x = prepared[dataset]
            for relation in ("MV", "UNI"):
                print(f"preflight {key} {relation} {dataset}", flush=True)
                job.heartbeat({"phase": "c2_group_and_isolation", "model": key,
                               "relation": relation, "dataset": dataset})
                wrapper.relation = relation
                unit = {"dataset": dataset, "relation": relation,
                        "origins": origins.tolist(), "columns": data["columns"].tolist(), "checks": []}
                model_report["units"].append(unit)
                output, records = wrapper.capture_group_probe(x)
                unit["groups"] = group_check(records, relation, len(x), x.shape[2])
                if output.shape != (4, 48, x.shape[2]) or output.requires_grad:
                    raise ValueError("C2 output axes/gradient contract failed")
                if wrapper.module.training or any(p.requires_grad for p in wrapper.module.parameters()):
                    raise ValueError("C2 weights must remain frozen in eval mode")
                solo = wrapper(x[:1])
                unit["checks"].append(compare(output[:1], solo, BATCH, "solo versus grouped batch"))
                unit["checks"].append(compare(wrapper(x), output, REPLAY, "same-path repeat"))
                permutation = torch.tensor([2, 0, 3, 1])
                unit["checks"].append(compare(wrapper(x[permutation])[torch.argsort(permutation)],
                                              output, BATCH, "origin permutation restored"))
                other_origin = x.clone()
                wave = torch.sin(torch.arange(512, dtype=torch.float32) * 0.071) + 0.25
                other_origin[3] += wave[:, None]
                unit["checks"].append(compare(wrapper(other_origin)[:1], output[:1], BATCH,
                                              "changed origin 3 cannot affect origin 0"))
                other_channels = x.clone()
                other_channels[0, :, 1:] += wave[:, None]
                changed = wrapper(other_channels)
                if relation == "UNI":
                    unit["checks"].append(compare(changed[0, :, :1], output[0, :, :1], BATCH,
                                                  "UNI channel 0 ignores other channel histories"))
                else:
                    unit["within_origin_other_channel_effect"] = {
                        "channel_zero_max_absolute_change": float((changed[0, :, 0] - output[0, :, 0]).abs().max()),
                        "interpretation": "Auxiliary evidence only; a large response is not a pass requirement"}
                changed_future = dict(data)
                changed_future["x"] = data["x"].copy()
                origin = int(origins[0])
                changed_future["x"][origin:origin + 48] += np.float32(100.0)
                past_after_future_change = inputs(changed_future, origins[:1])
                if not torch.equal(past_after_future_change, x[:1]):
                    raise ValueError("Changing the actual future array altered the constructed past")
                unit["checks"].append(compare(wrapper(past_after_future_change), solo, REPLAY,
                                              "actual future-array change cannot enter past-only prediction"))
                unit["future_barrier"] = {"future_rows_modified": 48, "past_input_bit_equal": True,
                                           "target_or_covariate_input_added": False}
                unit["timings"] = {str(b): timed_calls(wrapper, x[:b], job) for b in (1, 4)}
                report["timings"].extend(
                    {"dataset": dataset, "id": key, "method": key + "_" + relation,
                     "batch": b, "chunk_rows": None, **unit["timings"][str(b)]} for b in (1, 4))
                if dataset == DATASETS[0]:
                    replay_values[relation] = output.clone()
        after = state_hash(wrapper.module)
        model_report["state_sha256_after"] = after
        if before != after:
            raise ValueError("Frozen Chronos-2 parameters/buffers changed during checks")
        del wrapper
        cleanup()
        reload_started = time.perf_counter()
        restored = C2Forecast(models[key]["snapshot"], relation="MV", device="cuda")
        model_report["reload_seconds"] = time.perf_counter() - reload_started
        model_report["reload_checks"] = []
        for relation in ("MV", "UNI"):
            restored.relation = relation
            model_report["reload_checks"].append(
                compare(restored(prepared[DATASETS[0]][2]), replay_values[relation], REPLAY,
                        "pinned checkpoint reload " + relation))
        if state_hash(restored.module) != before:
            raise ValueError("Reloaded Chronos-2 state differs from pinned checkpoint")
        del restored
        cleanup()


@torch.inference_mode()
def original_forward(wrapper, x):
    gpu_x = x.to("cuda")
    value = wrapper.module.components(gpu_x)[0] if wrapper.row.get("output_component") == "main" else wrapper.module(gpu_x)
    return value.contiguous().cpu()


def stored_val(row, origins):
    entry = row.get("val_prediction")
    if entry is None:
        return None
    artifact(entry["path"], entry["sha256"])
    with np.load(entry["path"], allow_pickle=False) as archive:
        values, available = archive[entry["key"]], archive[entry["origins_key"]]
        index = []
        for origin in origins:
            matches = np.flatnonzero(available == origin)
            if len(matches) != 1:
                raise ValueError("Stored VAL prediction does not contain the canonical origin exactly once")
            index.append(int(matches[0]))
        return values[index].copy()


def baseline_checks(report, prepared, job):
    for dataset in DATASETS:
        data, origins, x = prepared[dataset]
        channels, latent = data["_C"], data["_K"]
        for row in baseline_rows(dataset):
            print(f"preflight baseline {dataset} {row['id']}", flush=True)
            job.heartbeat({"phase": "baseline_restore_and_parity", "dataset": dataset, "id": row["id"]})
            started = time.perf_counter()
            native = load_baseline(row, device="cuda", deploy=False)
            unit = {"dataset": dataset, "id": row["id"], "method": row["method"],
                    "seed": row["seed"], "load_seconds": time.perf_counter() - started,
                    "origins": origins.tolist(), "checks": [], "timings": []}
            report["baselines"].append(unit)
            before = state_hash(native.module)
            output = native(x)
            unit["checks"].append(compare(output, original_forward(native, x), BATCH,
                                          "original selected-model forward versus CPU-output wrapper"))
            unit["checks"].append(compare(native(x), output, REPLAY, "selected-model same-path replay"))
            saved = stored_val(row, origins)
            if saved is None:
                unit["saved_val_parity"] = {"status": "unavailable", "reason": row["val_prediction_missing_reason"],
                                             "substitution_used": False}
            else:
                unit["saved_val_parity"] = compare(output, saved, BATCH, "restored selected checkpoint versus stored VAL")
            if before != state_hash(native.module):
                raise ValueError("Selected baseline state changed during replay")
            if row.get("deploy_function"):
                source = source_module(row)
                started = time.perf_counter()
                deployed_module = getattr(source, row["deploy_function"])(native.module)
                unit["deployment_copy_seconds"] = time.perf_counter() - started
                deploy = BaselineForecast(deployed_module, row, "cuda")
                unit["checks"].append(compare(deploy(x), output, BATCH, "unmerged versus deployment-copy output"))
                del native, deployed_module
                cleanup()
            else:
                deploy = native
                del native
            deployed_state = state_hash(deploy.module)
            grids = [(1, channels), (1, latent), (4, 4 * channels), (4, 4 * latent), (4, latent)] \
                if row["method"] in ("F0", "MSE_LORA") else [(1, None), (4, None)]
            for batch, chunk in grids:
                deploy.set_chunk(None)
                full = deploy(x[:batch])
                deploy.set_chunk(chunk)
                chunked = deploy(x[:batch])
                unit["checks"].append(compare(chunked, full, BATCH, f"full versus row-chunk B{batch}/M{chunk}"))
                timing = {"dataset": dataset, "id": row["id"], "method": row["method"],
                          "batch": batch, "chunk_rows": chunk, **timed_calls(deploy, x[:batch], job)}
                unit["timings"].append(timing)
                report["timings"].append(timing)
            unit["deployment_receipt"] = deploy.receipt()
            unit["state_sha256_before"] = deployed_state
            unit["state_sha256_after"] = state_hash(deploy.module)
            if deployed_state != unit["state_sha256_after"]:
                raise ValueError("Baseline deployment state changed during checks")
            del deploy
            cleanup()


def estimate_campaign(report):
    from cost_controls import call_counts, instances_by_dataset, planned_grid, prepare
    prepared = {dataset: prepare(dataset) for dataset in DATASETS}
    grid = planned_grid(instances_by_dataset(), prepared, "cuda")
    counts = call_counts(grid)
    by_timing = {(r["dataset"], r["id"], r["method"], r["batch"], r["chunk_rows"]): r for r in report["timings"]}
    cost_estimate, cost_max_estimate = 0.0, 0.0
    for row, calls in zip(grid, counts["rows"]):
        timing_id = "C2" if row["method"] == "C2_MV" else "C2_SMALL" if row["method"] == "C2_SMALL_MV" else row["id"]
        chunk = None if row["method"] not in ("F0", "MSE_LORA") else row["chunk_rows"]
        probe = by_timing[(row["dataset"], timing_id, row["method"], row["batch"], chunk)]
        cost_estimate += calls["total_wrapper_calls"] * probe["median_s"]
        cost_max_estimate += calls["total_wrapper_calls"] * probe["max_s"]
    accuracy_estimate = 0.0
    for dataset in DATASETS:
        contract = read_json(HERE / "input_contract.json")["units"][dataset]
        origins = contract["origins"]["test_a"]["count"] + contract["origins"]["test_b"]["count"]
        for key in ("C2", "C2_SMALL"):
            for relation in ("MV", "UNI"):
                probe = next(r for r in report["timings"] if r["dataset"] == dataset
                             and r["id"] == key and r["method"] == key + "_" + relation and r["batch"] == 4)
                accuracy_estimate += int(np.ceil(origins / 4)) * probe["max_s"]
    loads = [u["load_seconds"] for u in report["baselines"]] + [u["load_seconds"] for u in report["c2_models"]]
    measured_load_upper = max(loads)
    loading_reserve = 3 * 3 * (9 + 2) * measured_load_upper
    estimate = {"status": "estimated_unverified_campaign_runtime", "cost_measurement_rows": len(grid),
                "planned_wrapper_calls": counts["total_wrapper_calls"],
                "cost_forward_seconds_median_probe": cost_estimate,
                "cost_forward_seconds_max_probe": cost_max_estimate,
                "accuracy_forward_seconds_max_probe": accuracy_estimate,
                "model_reload_reserve_seconds": loading_reserve,
                "reload_reserve_basis": "99 potential loads times slowest measured baseline/C2 constructor; reserve, not measured campaign work",
                "gpu_remaining_seconds_at_estimate": LIMITS["gpu_seconds"] - budget_snapshot()["gpu_seconds"],
                "does_not_guarantee_runtime": True,
                "excluded_unknown_overhead": ["scoring/serialization", "diagnostic failure/retry", "cost telemetry/cleanup"]}
    estimate["conservative_gpu_forward_and_load_estimate_seconds"] = cost_max_estimate + accuracy_estimate + loading_reserve
    estimate["estimated_within_remaining_gpu_budget"] = \
        estimate["conservative_gpu_forward_and_load_estimate_seconds"] < estimate["gpu_remaining_seconds_at_estimate"]
    return estimate


def main():
    configure()
    old = read_json(HERE / "ledger.json")
    attempt = 1 + sum(j["category"] == "gpu" and j["label"] == "preflight" for j in old["jobs"])
    output = HERE / f"preflight_attempt{attempt:02d}.json"
    if output.exists():
        raise RuntimeError("Preflight attempt output already exists")
    report = {"scope": "TRAIN/VAL checks only; no TEST scoring, fitting, or optimizer updates",
              "status": "running", "attempt": attempt, "tolerances": {"same_path": REPLAY, "batch_merge": BATCH},
              "code_sha256": digest(__file__), "c2_models": [], "baselines": [], "timings": []}
    try:
        with Job("gpu", "preflight", reserve_s=1200, metadata={"fit_count": 0, "optimizer_updates": 0,
                                                              "uses_test": False, "attempt": attempt}) as job:
            prepared = {dataset: fixed_inputs(dataset) for dataset in DATASETS}
            c2_checks(report, prepared, job)
            baseline_checks(report, prepared, job)
            report["campaign_runtime_estimate"] = estimate_campaign(report)
            report["status"] = "pass"
            report["fits"] = report["optimizer_updates"] = 0
            report["test_scoring_performed"] = False
            report["all_baseline_instances_checked"] = len(report["baselines"])
            job.heartbeat({"phase": "preflight_complete"})
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
        if hasattr(exc, "check_result"):
            report["failed_check"] = exc.check_result
        save_json(output, report)
        save_json(HERE / "preflight_checks.json", report)
        raise
    save_json(output, report)
    save_json(HERE / "preflight_checks.json", report)
    print(json.dumps({"status": report["status"], "baseline_instances": len(report["baselines"]),
                      "campaign_runtime_estimate": report["campaign_runtime_estimate"]}), flush=True)


if __name__ == "__main__":
    main()
