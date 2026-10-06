"""Fixed Chronos-Bolt backbone-scaling predictions and paired evaluation."""

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np

from baselines import baseline_rows, inputs, load_data, targets
from runtime import (CACHE, DATASETS, HERE, SCOPE, Job, artifact, check_seal,
                     configure, digest, read_json, save_json)


NEW_METHODS = (("BOLT_MINI_F0", "MINI"), ("BOLT_TINY_F0", "TINY"))
PRIMARY_ARMS = ("BOLT_SMALL_LEVEL", "BOLT_SMALL_F0", "BOLT_MINI_F0", "BOLT_TINY_F0")
PERIODS = ("test_a", "test_b", "combined")
METRICS = ("mse", "mae", "signed_mean_error")
ERROR_KEYS = {"mse": "squared", "mae": "absolute", "signed_mean_error": "signed"}
EXPECTED_GROUP_COUNTS = {
    "BOLT_SMALL_LEVEL": 2,
    "BOLT_SMALL_F0": 1,
    "BOLT_MINI_F0": 1,
    "BOLT_TINY_F0": 1,
}
PAIRED_COMPARISONS = (
    ("BOLT_SMALL_LEVEL", "BOLT_MINI_F0", "primary"),
    ("BOLT_SMALL_LEVEL", "BOLT_TINY_F0", "primary"),
    ("BOLT_SMALL_LEVEL", "BOLT_SMALL_F0", "auxiliary"),
    ("BOLT_SMALL_F0", "BOLT_MINI_F0", "auxiliary"),
    ("BOLT_SMALL_F0", "BOLT_TINY_F0", "auxiliary"),
    ("BOLT_MINI_F0", "BOLT_TINY_F0", "auxiliary"),
)


def require_static_apis():
    import baselines
    import bolt_models
    import runtime
    required_runtime = ("HERE", "CACHE", "DATASETS", "SCOPE", "Job", "artifact",
                        "configure", "environment", "read_json", "save_json",
                        "digest", "check_seal", "SEEDS")
    missing = [name for name in required_runtime if not hasattr(runtime, name)]
    missing += [name for name in ("baseline_rows", "inputs", "load_data", "targets")
                if not hasattr(baselines, name)]
    wrapper = getattr(bolt_models, "BoltForecast", None)
    if wrapper is None or not all(hasattr(wrapper, name) for name in ("__call__", "receipt", "module")):
        missing.append("bolt_models.BoltForecast callable/module/receipt")
    if missing:
        raise RuntimeError("Missing required isolated campaign APIs: " + ", ".join(missing))


def root_path(path):
    path = Path(path)
    return path if path.is_absolute() else HERE.parents[1] / path


def origin_contract(data):
    a, b = data["test_a_origins"], data["test_b_origins"]
    if len(a) == 0 or len(b) == 0 or a[-1] >= b[0]:
        raise ValueError("The fixed TEST periods must be nonempty and ordered")
    origins = np.concatenate((a, b)).astype(np.int64)
    indices = {"test_a": np.arange(len(a)),
               "test_b": np.arange(len(a), len(origins)),
               "combined": np.arange(len(origins))}
    return origins, indices


def read_prediction(receipt, origins, shape):
    artifact(receipt["path"], receipt["sha256"])
    with np.load(root_path(receipt["path"]), allow_pickle=False) as z:
        value = z[receipt.get("key", "prediction")]
        saved_origins = z[receipt.get("origins_key", "origins")]
    if not np.array_equal(saved_origins, origins):
        raise ValueError("Prediction origin order differs: " + receipt["path"])
    if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError("Prediction shape, dtype or finiteness differs: " + receipt["path"])
    return value


def prediction_receipt_path(dataset, method):
    return HERE / "prediction_receipts" / f"{dataset}_{method}.json"


def require_preflight_pass():
    path = HERE / "preflight_checks.json"
    if not path.exists():
        raise FileNotFoundError("Preflight checks must pass before TEST prediction")
    preflight = read_json(path)
    if preflight.get("status") != "pass":
        raise RuntimeError("Preflight status is not pass")
    if preflight.get("test_scoring_performed") not in (False, None):
        raise RuntimeError("Preflight must not score TEST data")
    if preflight.get("fits", 0) != 0 or preflight.get("optimizer_updates", 0) != 0:
        raise RuntimeError("Preflight must not fit or update models")
    return preflight


def model_checkpoint(record):
    path = record.get("checkpoint_dir", record.get("snapshot"))
    if path is None:
        raise ValueError("Model manifest must provide checkpoint_dir or snapshot")
    return root_path(path)


def verify_model_manifest(method, model_key):
    models = read_json(HERE / "model_manifest.json")["models"]
    model = models[model_key]
    if model.get("id") not in ("amazon/chronos-bolt-mini", "amazon/chronos-bolt-tiny"):
        raise ValueError(method + " is not bound to the official requested model")
    if model_key == "MINI" and model["id"] != "amazon/chronos-bolt-mini":
        raise ValueError("BOLT_MINI_F0 is not bound to amazon/chronos-bolt-mini")
    if model_key == "TINY" and model["id"] != "amazon/chronos-bolt-tiny":
        raise ValueError("BOLT_TINY_F0 is not bound to amazon/chronos-bolt-tiny")
    if not model.get("revision") or model["revision"] == "main":
        raise ValueError("Model revision must be a pinned revision SHA, not main")
    for receipt in model.get("files", {}).values():
        artifact(receipt["path"], receipt["sha256"])
    return model


def verify_new_receipt(record, dataset, method, origins, shape, seal_hash):
    if (record["dataset"] != dataset or record["method"] != method
            or record["seal_sha256"] != seal_hash or record["status"] != "complete"):
        raise ValueError("Completed prediction belongs to another fixed campaign")
    model_key = dict(NEW_METHODS)[method]
    source = verify_model_manifest(method, model_key)
    if (record["model_key"] != model_key or record["model_id"] != source["id"]
            or record["revision"] != source["revision"]):
        raise ValueError("Completed prediction uses another model binding")
    if record["checkpoint_dir"] != str(Path(source.get("checkpoint_dir", source.get("snapshot")))):
        raise ValueError("Completed prediction checkpoint_dir differs from model_manifest")
    artifact(record["model_manifest"]["path"], record["model_manifest"]["sha256"])
    artifact(record["preflight"]["path"], record["preflight"]["sha256"])
    read_prediction(record["prediction"], origins, shape)
    if record["prediction"]["shape"] != list(shape) or record["prediction"]["dtype"] != "float32":
        raise ValueError("Prediction receipt does not seal shape/dtype")
    if not record["model_receipt"]["eval_mode"] or record["model_receipt"]["trainable_parameter_count"] != 0:
        raise ValueError("New Bolt deployment must be frozen and eval-only")
    if record["origin_shape_fp32_finite"] != {"origin_count": len(origins),
                                             "shape": list(shape),
                                             "dtype": "float32",
                                             "finite": True}:
        raise ValueError("Origin/shape/FP32/finite receipt does not match")
    return record


def predict_all(job):
    import torch
    from bolt_models import BoltForecast
    check_seal()
    preflight = require_preflight_pass()
    seal_hash = digest(HERE / "evaluation_seal.json")
    manifest_path = HERE / "prediction_manifest.json"
    manifest = {"schema": "level_bolt_backbone_scaling_predictions_v1",
                "scope": SCOPE, "seal_sha256": seal_hash, "status": "predicting",
                "preflight": artifact(HERE / "preflight_checks.json"),
                "preflight_status": preflight["status"], "units": {},
                "state_preservation": {}, "fixed_method_order": [m[0] for m in NEW_METHODS],
                "fixed_dataset_order": list(DATASETS), "prediction_batch_origins": 4,
                "additional_fit_count": 0}
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if existing["status"] != "complete" or existing["seal_sha256"] != seal_hash:
            raise ValueError("Prediction manifest is not the completed fixed phase")
        for receipt in existing["state_preservation"].values():
            artifact(receipt["path"], receipt["sha256"])
        for dataset in DATASETS:
            data = load_data(dataset, include_test=True)
            origins, _ = origin_contract(data)
            shape = (len(origins), 48, data["_C"])
            for method, _ in NEW_METHODS:
                verify_new_receipt(existing["units"][dataset][method], dataset, method,
                                   origins, shape, seal_hash)
        return existing
    folder = CACHE / "predictions01"
    folder.mkdir(parents=True, exist_ok=True)
    for method, model_key in NEW_METHODS:
        incomplete = []
        for dataset in DATASETS:
            data = load_data(dataset, include_test=True)
            origins, _ = origin_contract(data)
            shape = (len(origins), 48, data["_C"])
            record_path = prediction_receipt_path(dataset, method)
            if record_path.exists():
                record = verify_new_receipt(read_json(record_path), dataset, method,
                                            origins, shape, seal_hash)
                manifest["units"].setdefault(dataset, {})[method] = record
            else:
                incomplete.append(dataset)
        if not incomplete:
            state_path = HERE / "prediction_receipts" / f"{method}_state_preservation.json"
            state = read_json(state_path)
            if (not state["unchanged"] or state["seal_sha256"] != seal_hash
                    or state["before"]["state_sha256"] != state["after"]["state_sha256"]):
                raise ValueError("Completed method lacks valid state preservation")
            manifest["state_preservation"][method] = artifact(state_path)
            continue
        model_info = verify_model_manifest(method, model_key)
        forecast = BoltForecast(model_checkpoint(model_info), device="cuda")
        model_before = forecast.receipt(include_file_hashes=True, include_state_hash=True)
        if not model_before["eval_mode"] or model_before["trainable_parameter_count"] != 0:
            raise ValueError("New checkpoint must be eval-only with no trainable parameters")
        for dataset in DATASETS:
            previous = manifest["units"].get(dataset, {}).get(method)
            if previous and previous["model_receipt"]["state_sha256"] != model_before["state_sha256"]:
                raise ValueError("Resumed model differs from completed prediction units")
        for dataset in incomplete:
            job.check_limits()
            data = load_data(dataset, include_test=True)
            origins, _ = origin_contract(data)
            path = folder / f"{dataset}_{method}.npz"
            if path.exists():
                raise FileExistsError("Preserve orphan prediction bytes: " + str(path))
            started = time.perf_counter()
            arrays = []
            for first in range(0, len(origins), 4):
                job.check_limits()
                batch_origins = origins[first:first + 4]
                cpu_x = inputs(data, batch_origins)
                output = forecast(cpu_x)
                if tuple(output.shape) != (len(batch_origins), 48, data["_C"]):
                    raise ValueError("New Bolt forecast did not return [B,48,C]")
                arrays.append(output.numpy().copy())
                if first % 80 == 0:
                    job.heartbeat(f"{method}/{dataset}: {min(first + 4, len(origins))}/{len(origins)} origins")
            prediction = np.concatenate(arrays)
            if prediction.dtype != np.float32 or not np.isfinite(prediction).all():
                raise ValueError("New prediction is not finite FP32")
            np.savez_compressed(path, prediction=prediction, origins=origins)
            record = {"schema": "level_bolt_backbone_prediction_unit_v1", "status": "complete",
                      "dataset": dataset, "method": method, "seed": None, "model_key": model_key,
                      "model_id": model_info["id"], "revision": model_info["revision"],
                      "checkpoint_dir": str(Path(model_info.get("checkpoint_dir", model_info.get("snapshot")))),
                      "model_manifest": artifact(HERE / "model_manifest.json"),
                      "preflight": artifact(HERE / "preflight_checks.json"),
                      "seal_sha256": seal_hash,
                      "prediction": {**artifact(path), "key": "prediction", "origins_key": "origins",
                                     "shape": list(prediction.shape), "dtype": "float32"},
                      "data": artifact(data["_path"]),
                      "origin_values_sha256": hashlib.sha256(origins.tobytes()).hexdigest(),
                      "origin_shape_fp32_finite": {"origin_count": len(origins),
                                                   "shape": list(prediction.shape),
                                                   "dtype": "float32", "finite": True},
                      "point": "Native Chronos-Bolt q=0.5 median, first H48, TRAIN-standardized original coordinates",
                      "inputs": "Only X[o-512:o]; each original channel is an independent univariate task",
                      "reshape": "[B,512,C] -> [B,C,512] -> [B*C,512] -> [B,48,C]",
                      "additional_fit_count": 0, "elapsed_s": time.perf_counter() - started,
                      "model_receipt": model_before, "scope": SCOPE}
            verify_new_receipt(record, dataset, method, origins, prediction.shape, seal_hash)
            save_json(prediction_receipt_path(dataset, method), record)
            manifest["units"].setdefault(dataset, {})[method] = record
            save_json(HERE / "prediction_manifest_partial.json", manifest)
            job.heartbeat(f"{method}/{dataset}: complete")
            del data, arrays, prediction, cpu_x, output
        model_after = forecast.receipt(include_file_hashes=True, include_state_hash=True)
        if model_after["state_sha256"] != model_before["state_sha256"]:
            raise ValueError("Chronos-Bolt state changed during fixed zero-shot prediction")
        state_path = HERE / "prediction_receipts" / f"{method}_state_preservation.json"
        save_json(state_path, {"method": method, "seal_sha256": seal_hash,
                               "before": model_before, "after": model_after, "unchanged": True})
        manifest["state_preservation"][method] = artifact(state_path)
        del forecast
        gc.collect()
        torch.cuda.empty_cache()
    if any(set(manifest["units"].get(d, {})) != {m[0] for m in NEW_METHODS} for d in DATASETS):
        raise ValueError("Every fixed Mini/Tiny dataset must finish before scoring")
    manifest.update(status="complete", evaluation_units=6,
                    full_channel_prediction_origins=sum(
                        row["prediction"]["shape"][0] for unit in manifest["units"].values()
                        for row in unit.values()))
    save_json(manifest_path, manifest)
    return manifest


def error_terms(prediction, target, mask):
    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError("Prediction, observed targets and mask axes must agree")
    if not np.isfinite(prediction).all() or not np.isfinite(target[mask]).all():
        raise ValueError("Nonfinite prediction or observed target")
    error = np.where(mask, prediction.astype(np.float64) - target.astype(np.float64), 0.)
    return {"squared": np.square(error).sum(axis=1),
            "absolute": np.abs(error).sum(axis=1),
            "signed": error.sum(axis=1),
            "count": mask.sum(axis=1).astype(np.float64)}


def score_terms(terms, index):
    count = terms["count"][index].sum(axis=0)
    if np.any(count <= 0):
        raise ValueError("Every fixed channel needs observed targets in each reported period")
    sums = {key: terms[key][index].sum(axis=0) for key in ("squared", "absolute", "signed")}
    result = {metric: float(np.mean(sums[key] / count)) for metric, key in ERROR_KEYS.items()}
    result.update(channel_mse=(sums["squared"] / count).tolist(),
                  channel_mae=(sums["absolute"] / count).tolist(),
                  channel_signed_mean_error=(sums["signed"] / count).tolist(),
                  squared_error_sums=sums["squared"].tolist(),
                  absolute_error_sums=sums["absolute"].tolist(),
                  signed_error_sums=sums["signed"].tolist(),
                  target_counts=count.astype(np.int64).tolist(),
                  target_count=int(count.sum()), origins=len(index))
    return result


def mean_seed_terms(members):
    if any(not np.array_equal(members[0]["count"], t["count"]) for t in members[1:]):
        raise ValueError("Selected seed target denominators differ")
    return {key: members[0]["count"].copy() if key == "count"
            else np.mean(np.stack([t[key] for t in members]), axis=0)
            for key in ("squared", "absolute", "signed", "count")}


def collect_unit(dataset, prediction_manifest, job):
    data = load_data(dataset, include_test=True)
    origins, indices = origin_contract(data)
    target, mask = targets(data, origins)
    mask = mask.astype(bool)
    rows, terms, replay = [], {}, []
    for old in baseline_rows(dataset):
        if old.get("arm") not in ("BOLT_SMALL_LEVEL", "BOLT_SMALL_F0"):
            raise ValueError("Baseline row lacks the fixed Small arm mapping")
        row = {**old, "original_method": old["method"], "method": old["arm"],
               "source_inner_method": old["method"], "reused": True}
        prediction = read_prediction(row["test_prediction"], origins, target.shape)
        terms[row["id"]] = error_terms(prediction, target, mask)
        row["periods"] = {p: score_terms(terms[row["id"]], i) for p, i in indices.items()}
        for period in PERIODS:
            for metric in ("mse", "mae"):
                expected = row["expected_period_scores"][period][metric]
                actual = row["periods"][period][metric]
                tolerance = max(1e-10, abs(expected) * 1e-9)
                item = {"dataset": dataset, "id": row["id"], "method": row["method"],
                        "original_method": row["original_method"], "period": period,
                        "metric": metric, "expected": expected, "actual": actual,
                        "absolute_difference": abs(actual - expected), "tolerance": tolerance,
                        "pass": abs(actual - expected) <= tolerance,
                        "source": row["source_evaluation"],
                        "source_pointer": row["source_pointer"]}
                replay.append(item)
                if not item["pass"]:
                    save_json(HERE / "old_score_replay_failure.json",
                              {"failed": item, "checks": replay})
                    raise ValueError("Original score reproduction failed: "
                                     + row["id"] + "/" + period + "/" + metric)
        rows.append(row)
        job.check_limits()
    seal_hash = digest(HERE / "evaluation_seal.json")
    for method, _ in NEW_METHODS:
        record = prediction_manifest["units"][dataset][method]
        verify_new_receipt(record, dataset, method, origins, target.shape, seal_hash)
        identifier = dataset + "_" + method
        prediction = read_prediction(record["prediction"], origins, target.shape)
        terms[identifier] = error_terms(prediction, target, mask)
        rows.append({**record, "id": identifier, "original_method": "F0",
                     "source_inner_method": "BoltForecast", "reused": False,
                     "periods": {p: score_terms(terms[identifier], i) for p, i in indices.items()}})
        job.check_limits()
    groups = {}
    for method in PRIMARY_ARMS:
        members = [row for row in rows if row["method"] == method]
        if len(members) != EXPECTED_GROUP_COUNTS[method]:
            raise ValueError("A fixed comparison group has the wrong selected instance count: " + method)
        current_terms = mean_seed_terms([terms[row["id"]] for row in members])
        groups[method] = {"ids": [row["id"] for row in members],
                          "seeds": [row["seed"] for row in members],
                          "aggregation": "Mean individual seed losses, not averaged predictions; zero-shot counted once",
                          "periods": {p: score_terms(current_terms, i) for p, i in indices.items()},
                          "terms": current_terms}
    return data, origins, indices, rows, terms, groups, replay


def paired_intervals(groups, indices, unit_contract, job):
    config = unit_contract["bootstrap"]
    draws, block, seed, salt = (config[k] for k in ("draws", "block_origins", "seed", "seed_sequence_salt"))
    weights = {}
    total = len(indices["combined"])
    for period_index, period in enumerate(PERIODS[:2]):
        local_index = indices[period]
        n = len(local_index)
        if n < block:
            raise ValueError("The fixed period is shorter than the recorded bootstrap block")
        rng = np.random.default_rng(np.random.SeedSequence([seed, salt, period_index]))
        matrix = np.zeros((draws, total), dtype=np.int32)
        for draw in range(draws):
            starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
            local = np.concatenate([np.arange(start, start + block) for start in starts])[:n]
            matrix[draw] = np.bincount(local_index[local], minlength=total)
        weights[period] = matrix
    weights["combined"] = weights["test_a"] + weights["test_b"]
    sampled = {}
    for period, matrix in weights.items():
        sampled[period] = {}
        for method, group in groups.items():
            job.check_limits()
            t = group["terms"]
            denominator = matrix @ t["count"]
            if np.any(denominator <= 0):
                raise ValueError("A bootstrap draw has an unobserved fixed channel")
            sampled[period][method] = {metric: np.mean((matrix @ t[key]) / denominator, axis=1)
                                      for metric, key in (("mse", "squared"), ("mae", "absolute"))}
    result = []
    for left, right, role in PAIRED_COMPARISONS:
        for period in PERIODS:
            for metric in ("mse", "mae"):
                a = groups[left]["periods"][period][metric]
                b = groups[right]["periods"][period][metric]
                a_draw = sampled[period][left][metric]
                b_draw = sampled[period][right][metric]
                absolute_ci = np.quantile(a_draw - b_draw, [.025, .975]).tolist()
                relative_available = b != 0 and bool(np.all(b_draw != 0))
                relative_ci = np.quantile(100 * (a_draw / b_draw - 1), [.025, .975]).tolist() if relative_available else None
                result.append({"left": left, "right": right, "comparison_role": role,
                    "period": period, "metric": metric, "left_value": a, "right_value": b,
                    "absolute_difference": a - b,
                    "relative_percent": 100 * (a / b - 1) if b != 0 else None,
                    "denominator": b, "absolute_ci95": absolute_ci,
                    "relative_ci95_percent": relative_ci,
                    "relative_ci_unavailable_reason": None if relative_available else "Zero reference point/draw denominator",
                    "negative_favors": left, "positive_favors": right,
                    "origins": len(indices[period]), "bootstrap_draws": draws,
                    "block_origins": block, "rng_seed": seed, "rng_salt": salt,
                    "period_stratified": True, "fixed_seeds_conditional": True,
                    "not_equivalence_or_noninferiority": True})
    return result, {p: hashlib.sha256(m.tobytes()).hexdigest() for p, m in weights.items()}


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot publish an empty comparison table")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, allow_nan=False) if isinstance(v, (dict, list))
                             else "" if v is None else v for k, v in row.items()})


def score_all(job):
    check_seal()
    require_preflight_pass()
    destination = HERE / "evaluation_summary.json"
    if destination.exists():
        raise FileExistsError("Scoring is complete; use verify rather than silently replacing the evaluation")
    manifest = read_json(HERE / "prediction_manifest.json")
    if manifest["status"] != "complete" or manifest["evaluation_units"] != 6:
        raise ValueError("All six Mini/Tiny prediction units must finish before joint scoring")
    contracts = read_json(HERE / "input_contract.json")["units"]
    result = {"schema": "level_bolt_backbone_scaling_evaluation_v1",
              "status": "scoring", "scope": SCOPE,
              "prediction_manifest": artifact(HERE / "prediction_manifest.json"),
              "seal": artifact(HERE / "evaluation_seal.json"),
              "preflight": artifact(HERE / "preflight_checks.json"),
              "units": {}, "arms": list(PRIMARY_ARMS),
              "primary_metric": "Masked channel-macro MSE in existing TRAIN-standardized coordinates",
              "aggregation": "Pool period channel sums/counts, channel mean, then mean selected-seed losses",
              "uncertainty_scope": "Conditional on fixed checkpoints/seeds and already-exposed TEST data; model selection uncertainty is excluded"}
    scalar_rows, channel_rows, origin_rows, comparisons, replay_checks = [], [], [], [], []
    for dataset in DATASETS:
        data, origins, indices, rows, terms, groups, replay = collect_unit(dataset, manifest, job)
        replay_checks += replay
        intervals, weight_hashes = paired_intervals(groups, indices, contracts[dataset], job)
        comparisons += [{"dataset": dataset, **item} for item in intervals]
        for row in rows:
            for period, score in row["periods"].items():
                scalar_rows.append({"dataset": dataset, "method": row["method"],
                    "original_method": row.get("original_method"), "id": row["id"],
                    "seed": row["seed"], "aggregation": "individual", "period": period,
                    **{k: score[k] for k in METRICS + ("target_count", "origins", "target_counts")},
                    "reused": row["reused"]})
                for channel, name in enumerate(data["columns"].tolist()):
                    channel_rows.append({"dataset": dataset, "method": row["method"],
                        "original_method": row.get("original_method"), "id": row["id"],
                        "seed": row["seed"], "period": period, "channel_index": channel,
                        "channel": name, "mse": score["channel_mse"][channel],
                        "mae": score["channel_mae"][channel],
                        "signed_mean_error": score["channel_signed_mean_error"][channel],
                        "squared_error_sum": score["squared_error_sums"][channel],
                        "absolute_error_sum": score["absolute_error_sums"][channel],
                        "signed_error_sum": score["signed_error_sums"][channel],
                        "target_count": score["target_counts"][channel]})
            current = terms[row["id"]]
            for index, origin in enumerate(origins):
                count = current["count"][index]
                valid = count > 0
                origin_rows.append({"dataset": dataset, "method": row["method"],
                    "original_method": row.get("original_method"), "id": row["id"],
                    "seed": row["seed"], "origin": int(origin),
                    "period": "test_a" if index < len(indices["test_a"]) else "test_b",
                    "mse_observed_channel_mean": float(np.mean(current["squared"][index][valid] / count[valid])) if valid.any() else None,
                    "mae_observed_channel_mean": float(np.mean(current["absolute"][index][valid] / count[valid])) if valid.any() else None,
                    "signed_mean_error_observed_channel_mean": float(np.mean(current["signed"][index][valid] / count[valid])) if valid.any() else None,
                    "observed_channels": int(valid.sum()), "target_count": int(count.sum()),
                    "scope": "Origin-local observed-channel mean; does not replace pooled main metric"})
        for method, group in groups.items():
            for period, score in group["periods"].items():
                scalar_rows.append({"dataset": dataset, "method": method, "original_method": None,
                    "id": ";".join(group["ids"]),
                    "seed": "mean_selected_seeds" if len(group["ids"]) == 2 else "single_model",
                    "aggregation": group["aggregation"], "period": period,
                    **{k: score[k] for k in METRICS + ("target_count", "origins", "target_counts")},
                    "reused": method in ("BOLT_SMALL_LEVEL", "BOLT_SMALL_F0")})
                for channel, name in enumerate(data["columns"].tolist()):
                    channel_rows.append({"dataset": dataset, "method": method,
                        "original_method": None, "id": ";".join(group["ids"]),
                        "seed": "mean_selected_seeds" if len(group["ids"]) == 2 else "single_model",
                        "period": period, "channel_index": channel, "channel": name,
                        "mse": score["channel_mse"][channel], "mae": score["channel_mae"][channel],
                        "signed_mean_error": score["channel_signed_mean_error"][channel],
                        "squared_error_sum": score["squared_error_sums"][channel],
                        "absolute_error_sum": score["absolute_error_sums"][channel],
                        "signed_error_sum": score["signed_error_sums"][channel],
                        "target_count": score["target_counts"][channel]})
        result["units"][dataset] = {"models": rows,
            "methods": {method: {key: value for key, value in group.items() if key != "terms"}
                        for method, group in groups.items()},
            "data": artifact(data["_path"]), "channels": data["columns"].tolist(),
            "origin_values_sha256": hashlib.sha256(origins.tobytes()).hexdigest(),
            "target_mask_values_sha256": hashlib.sha256(np.stack([data["finite"][o:o+48] for o in origins]).tobytes()).hexdigest(),
            "origin_counts": {p: len(i) for p, i in indices.items()},
            "bootstrap": contracts[dataset]["bootstrap"],
            "bootstrap_weight_sha256": weight_hashes,
            "paired_comparisons": intervals}
        save_json(HERE / "evaluation_summary_partial.json", result)
        job.heartbeat(dataset + ": four-arm accuracy and paired intervals complete")
        del data, origins, indices, rows, terms, groups
    tables = [("accuracy_comparison.csv", scalar_rows), ("accuracy_channels.csv", channel_rows),
              ("accuracy_origins.csv", origin_rows), ("paired_comparisons.csv", comparisons)]
    for name, rows in tables:
        write_csv(HERE / name, rows)
    save_json(HERE / "old_score_replay.json",
              {"pass": all(r["pass"] for r in replay_checks), "checks": replay_checks,
               "new_scoring": "Original Small LEVEL/F0 saved arrays rescored without model inference"})
    result.update(status="complete", tables={name: artifact(HERE / name) for name, _ in tables},
                  old_score_replay=artifact(HERE / "old_score_replay.json"))
    save_json(destination, result)
    return result


def verify_saved_outputs(job):
    check_seal()
    result = read_json(HERE / "evaluation_summary.json")
    manifest = read_json(HERE / "prediction_manifest.json")
    if result["status"] != "complete" or manifest["status"] != "complete":
        raise ValueError("Only completed fixed outputs can be verified")
    for receipt in result["tables"].values():
        artifact(receipt["path"], receipt["sha256"])
    checks = []
    for dataset in DATASETS:
        _, _, _, rows, _, groups, _ = collect_unit(dataset, manifest, job)
        for row in rows:
            saved = next(r for r in result["units"][dataset]["models"] if r["id"] == row["id"])
            for period in PERIODS:
                for metric in METRICS:
                    expected, actual = saved["periods"][period][metric], row["periods"][period][metric]
                    checks.append({"dataset": dataset, "id": row["id"], "period": period,
                                   "metric": metric, "expected": expected, "actual": actual,
                                   "pass": abs(actual - expected) <= 1e-11})
        for method, group in groups.items():
            for period in PERIODS:
                for metric in METRICS:
                    expected = result["units"][dataset]["methods"][method]["periods"][period][metric]
                    actual = group["periods"][period][metric]
                    checks.append({"dataset": dataset, "method": method, "period": period,
                                   "metric": metric, "expected": expected, "actual": actual,
                                   "pass": abs(actual - expected) <= 1e-11})
        job.heartbeat(dataset + ": saved-array score verification complete")
    with (HERE / "accuracy_comparison.csv").open(encoding="utf-8", newline="") as stream:
        table = list(csv.DictReader(stream))
    for row in table:
        dataset, method, period = row["dataset"], row["method"], row["period"]
        unit = result["units"][dataset]
        if row["aggregation"] == "individual":
            reference = next(r for r in unit["models"] if r["id"] == row["id"])["periods"][period]
        else:
            reference = unit["methods"][method]["periods"][period]
        for metric in METRICS:
            checks.append({"dataset": dataset, "id": row["id"], "table": "accuracy_comparison.csv",
                           "period": period, "metric": metric,
                           "pass": abs(float(row[metric]) - reference[metric]) <= 1e-11})
    verification = {"schema": "level_bolt_backbone_accuracy_verification_v1",
                    "pass": all(check["pass"] for check in checks), "checks": checks,
                    "zero_fit": True, "model_inference": False,
                    "independent_reproduction": False,
                    "scope": "Execution-team saved-array/aggregate/CSV consistency check",
                    "evaluation": artifact(HERE / "evaluation_summary.json")}
    save_json(HERE / "accuracy_verification.json", verification)
    if not verification["pass"]:
        raise ValueError("Saved-array consistency verification failed")
    return verification


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("predict", "score", "verify"))
    parser.add_argument("--retry-reason")
    args = parser.parse_args()
    require_static_apis()
    configure()
    category = "gpu" if args.phase == "predict" else "cpu"
    attempt_dir = HERE / "evaluation_attempts"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    previous = sorted(attempt_dir.glob(args.phase + "_[0-9][0-9].json"))
    if previous and read_json(previous[-1])["status"] == "failed":
        if not args.retry_reason:
            raise ValueError("A failed phase needs an explicit corrected-cause retry reason")
        if sum(read_json(p)["status"] == "failed" for p in previous) >= 2:
            raise RuntimeError("Two failed phase attempts are preserved; no automatic further retry")
    number = len(previous) + 1
    attempt_path = attempt_dir / f"{args.phase}_{number:02d}.json"
    started = time.perf_counter()
    try:
        with Job(category, "fixed_" + args.phase,
                 metadata={"additional_fit_count": 0, "scope": SCOPE}) as job:
            function = {"predict": predict_all, "score": score_all,
                        "verify": verify_saved_outputs}[args.phase]
            function(job)
        save_json(attempt_path, {"phase": args.phase, "status": "complete",
                  "elapsed_s": time.perf_counter() - started,
                  "additional_fit_count": 0, "retry_reason": args.retry_reason})
    except BaseException:
        partials = []
        for name in ("prediction_manifest_partial.json", "evaluation_summary_partial.json"):
            source = HERE / name
            if source.exists():
                snapshot = attempt_dir / (attempt_path.stem + "_" + name)
                shutil.copyfile(source, snapshot)
                partials.append({"source": artifact(source), "snapshot": artifact(snapshot)})
        save_json(attempt_path, {"phase": args.phase, "status": "failed",
                  "elapsed_s": time.perf_counter() - started,
                  "traceback": traceback.format_exc(), "additional_fit_count": 0,
                  "retry_reason": args.retry_reason, "preserved_partial_records": partials})
        raise


if __name__ == "__main__":
    main()
