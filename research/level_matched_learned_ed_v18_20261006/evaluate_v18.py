"""Sealed TEST prediction and paired evaluation for the matched learned E/D arm.

This file is source-only until the campaign gate allows execution.  It does not
create placeholder receipts.  The executable phases are intentionally separated:
``predict`` creates only the six selected MATCHED_LEARNED_ED TEST forecasts,
``score`` reads sealed forecasts and old prediction bytes, and ``verify`` checks
already materialized artifacts.
"""

import argparse
import csv
import gc
import hashlib
import json
import time
import traceback
from pathlib import Path

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
    dataset_contract,
    digest,
    finish_test,
    inputs,
    load_data,
    read_json,
    register_test,
    resolve,
    save_json,
    targets,
)


METHOD = "MATCHED_LEARNED_ED"
PERIODS = ("test_a", "test_b", "combined")
ERROR_KEYS = {"mse": "squared", "mae": "absolute", "signed_mean_error": "signed"}
METRICS = tuple(ERROR_KEYS)
PRIMARY_AND_SECONDARY_RIGHT = ("ORIGINAL_LEVEL", "COMPRESS", "F0", "MSE_LORA")
REUSED_METHOD_ORDER = (
    "ORIGINAL_LEVEL",
    "COMPRESS",
    "F0",
    "MSE_LORA",
    "DIRECT_NLINEAR",
    "MATCHED_RAW",
    "BOLT_MINI_F0",
    "BOLT_TINY_F0",
    "C2_MV",
    "C2_SMALL_MV",
    "C2_UNI",
    "C2_SMALL_UNI",
)


def schema_contract():
    """Return the intended output contract without touching model/data bytes."""
    return {
        "status": "source_only_unexecuted",
        "blocked_reason": "STOP_RESOURCE before source seal / TRAIN / TEST",
        "prediction_manifest": {
            "path": "research/level_matched_learned_ed_v18_20261006/prediction_manifest.json",
            "rows": "6 selected ED prediction units after execution only",
            "unit_keys": [
                "dataset",
                "method",
                "id",
                "seed",
                "lr",
                "selected_epoch",
                "checkpoint",
                "prediction",
                "model_receipt",
                "state_preservation",
                "selection_seal_sha256",
            ],
        },
        "evaluation_json": {
            "path": "research/level_matched_learned_ed_v18_20261006/evaluation.json",
            "top_keys": [
                "schema",
                "status",
                "method",
                "prediction_manifest",
                "selection",
                "selection_seal",
                "units",
                "old_score_replay",
                "paired_bootstrap_weight_hashes",
                "row_counts",
            ],
        },
        "accuracy_models_csv": {
            "keys": [
                "dataset",
                "aggregation",
                "method",
                "id",
                "seed",
                "period",
                "source_kind",
                "reused",
                "mse",
                "mae",
                "signed_mean_error",
                "target_count",
                "origins",
            ],
            "planned_rows_if_all_reuse_available": 279,
        },
        "accuracy_channels_csv": {
            "keys": [
                "dataset",
                "aggregation",
                "method",
                "id",
                "seed",
                "period",
                "channel",
                "mse",
                "mae",
                "signed_mean_error",
                "target_count",
            ],
            "planned_rows_if_all_reuse_available": "sum over dataset channels of 3 periods × (individual members + method groups)",
        },
        "paired_comparisons_csv": {
            "keys": [
                "dataset",
                "left",
                "right",
                "comparison_role",
                "period",
                "metric",
                "left_value",
                "right_value",
                "absolute_difference",
                "relative_percent",
                "absolute_ci95",
                "relative_ci95_percent",
                "bootstrap_draws",
                "block_origins",
                "rng_seed",
                "rng_salt",
            ],
            "planned_rows": len(DATASETS) * len(PRIMARY_AND_SECONDARY_RIGHT) * len(PERIODS) * 2,
        },
    }


def origin_contract(data):
    a = np.asarray(data["test_a_origins"], dtype=np.int64)
    b = np.asarray(data["test_b_origins"], dtype=np.int64)
    if len(a) == 0 or len(b) == 0 or int(a[-1]) >= int(b[0]):
        raise ValueError("The fixed TEST periods must be nonempty and ordered")
    origins = np.concatenate((a, b)).astype(np.int64)
    return origins, {
        "test_a": np.arange(len(a)),
        "test_b": np.arange(len(a), len(origins)),
        "combined": np.arange(len(origins)),
    }


def read_prediction(receipt, origins, shape):
    artifact(receipt["path"], receipt.get("sha256"))
    with np.load(resolve(receipt["path"]), allow_pickle=False) as archive:
        key = receipt.get("key") or receipt.get("prediction_key") or "prediction"
        value = archive[key]
        saved_origins = archive[receipt.get("origins_key", "origins")]
    if not np.array_equal(saved_origins.astype(np.int64), origins.astype(np.int64)):
        raise ValueError("Prediction origin order differs: " + str(receipt["path"]))
    if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError("Prediction shape, dtype or finiteness differs: " + str(receipt["path"]))
    return value


def selected_units():
    selection = read_json(HERE / "selection.json")
    if selection.get("prediction_ensemble"):
        raise RuntimeError("The selected ED campaign must not ensemble predictions")
    units = {}
    for dataset in DATASETS:
        unit = selection["units"][dataset]
        if unit.get("seeds") != [92601, 92602] or len(unit.get("run_ids", [])) != 2:
            raise RuntimeError("Selection unit does not expose exactly two selected ED seeds: " + dataset)
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
                    "best_val_prediction": unit["best_val_prediction"][index],
                    "mean_best_val_mse": unit["mean_best_val_mse"],
                    "lr_selection": unit["lr_selection"],
                }
            )
        units[dataset] = rows
    return units


def prediction_receipt_path(dataset, run_id):
    return HERE / "prediction_receipts" / f"{dataset}_{run_id}.json"


def verify_new_receipt(record, dataset, row, origins, shape, selection_sha256):
    if record.get("status") != "complete" or record.get("dataset") != dataset:
        raise ValueError("Completed prediction receipt belongs to another dataset")
    if record.get("method") != METHOD or record.get("id") != row["id"] or record.get("seed") != row["seed"]:
        raise ValueError("Completed prediction receipt belongs to another selected ED unit")
    if record.get("selection_seal_sha256") != selection_sha256:
        raise ValueError("Completed prediction receipt is bound to another selection seal")
    read_prediction(record["prediction"], origins, shape)
    state = record.get("state_preservation", {})
    if not state.get("unchanged") or state.get("before_sha256") != state.get("after_sha256"):
        raise ValueError("Completed prediction lacks state-preservation evidence")
    return record


def predict_all(job, device):
    import torch
    from model_v18 import restore_model

    check_selection_seal()
    selection_sha256 = digest(HERE / "selection_seal.json")
    selection = selected_units()
    manifest_path = HERE / "prediction_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("status") != "complete" or manifest.get("selection_seal_sha256") != selection_sha256:
            raise RuntimeError("Existing prediction manifest is not the sealed completed v18 phase")
        for dataset in DATASETS:
            data = load_data(dataset, include_test=True)
            origins, _ = origin_contract(data)
            shape = (len(origins), 48, int(data["_C"]))
            for row in selection[dataset]:
                verify_new_receipt(manifest["units"][dataset][row["id"]], dataset, row, origins, shape, selection_sha256)
        return manifest

    folder = CACHE / "predictions01"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "matched_learned_ed_v18_prediction_manifest_1",
        "status": "predicting",
        "method": METHOD,
        "selection_seal": artifact(HERE / "selection_seal.json"),
        "selection_seal_sha256": selection_sha256,
        "prediction_ensemble": False,
        "prediction_batch_origins": 4,
        "units": {dataset: {} for dataset in DATASETS},
    }
    for dataset in DATASETS:
        data = load_data(dataset, include_test=True)
        origins, _ = origin_contract(data)
        shape = (len(origins), 48, int(data["_C"]))
        for row in selection[dataset]:
            receipt_path = prediction_receipt_path(dataset, row["id"])
            if receipt_path.exists():
                manifest["units"][dataset][row["id"]] = verify_new_receipt(
                    read_json(receipt_path), dataset, row, origins, shape, selection_sha256
                )
                continue
            npz_path = folder / f"{row['id']}.npz"
            if npz_path.exists():
                raise FileExistsError("Preserve orphan prediction bytes: " + str(npz_path))
            test_id = dataset + "::" + row["id"]
            register_test(test_id, {"dataset": dataset, "id": row["id"], "seed": row["seed"], "method": METHOD})
            started = time.perf_counter()
            try:
                model = restore_model(row["checkpoint"]["path"], device=device, expected_sha256=row["checkpoint"]["sha256"])
                before = model.receipt(include_state_hash=True)
                arrays = []
                model.eval()
                with torch.inference_mode():
                    for first in range(0, len(origins), 4):
                        job.check_limits()
                        batch_origins = origins[first:first + 4]
                        x = inputs(data, batch_origins).to(next(model.parameters()).device)
                        value = model(x).detach().cpu().contiguous().numpy()
                        if value.shape != (len(batch_origins), 48, int(data["_C"])) or value.dtype != np.float32 or not np.isfinite(value).all():
                            raise RuntimeError("STOP_DEBUG: ED TEST prediction is not finite FP32 H48C")
                        arrays.append(value.copy())
                prediction = np.concatenate(arrays, axis=0)
                np.savez_compressed(npz_path, prediction=prediction, origins=origins)
                after = model.receipt(include_state_hash=True)
                if before["state_sha256"] != after["state_sha256"]:
                    raise RuntimeError("STOP_DEBUG: selected ED state changed during TEST prediction")
                record = {
                    "schema": "matched_learned_ed_v18_prediction_unit_1",
                    "status": "complete",
                    "dataset": dataset,
                    "method": METHOD,
                    "id": row["id"],
                    "seed": row["seed"],
                    "lr": row["lr"],
                    "selected_epoch": row["selected_epoch"],
                    "checkpoint": row["checkpoint"],
                    "selection_seal_sha256": selection_sha256,
                    "prediction": {**artifact(npz_path), "key": "prediction", "origins_key": "origins", "shape": list(prediction.shape), "dtype": "float32"},
                    "data": artifact(data["_path"]),
                    "origin_values_sha256": array_hash(origins),
                    "origin_shape_fp32_finite": True,
                    "model_receipt": before,
                    "state_preservation": {
                        "before_sha256": before["state_sha256"],
                        "after_sha256": after["state_sha256"],
                        "unchanged": True,
                        "eval_mode": after["eval_mode"],
                        "backbone_eval_mode": after["backbone_eval_mode"],
                    },
                    "elapsed_s": time.perf_counter() - started,
                    "additional_fit_count": 0,
                    "inputs": "Only X[o-512:o]; no TEST used for selection; no prediction ensemble",
                }
                verify_new_receipt(record, dataset, row, origins, shape, selection_sha256)
                save_json(receipt_path, record)
                manifest["units"][dataset][row["id"]] = record
                save_json(HERE / "prediction_manifest_partial.json", manifest)
                finish_test(test_id, {"prediction": artifact(npz_path), "elapsed_s": record["elapsed_s"]})
                del model, arrays, prediction
                gc.collect()
                torch.cuda.empty_cache()
            except BaseException:
                finish_test(test_id, {"error": traceback.format_exc()}, status="failed")
                raise
    if sum(len(unit) for unit in manifest["units"].values()) != 6:
        raise RuntimeError("Exactly six selected ED TEST predictions are required")
    manifest.update(
        status="complete",
        evaluation_units=6,
        full_channel_prediction_origins=sum(row["prediction"]["shape"][0] for unit in manifest["units"].values() for row in unit.values()),
        completed_utc=time.time(),
    )
    save_json(manifest_path, manifest)
    return manifest


def error_terms(prediction, target, mask):
    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError("Prediction, observed targets and mask axes must agree")
    if not np.isfinite(prediction).all() or not np.isfinite(target[mask]).all():
        raise ValueError("Nonfinite prediction or observed target")
    error = np.where(mask, prediction.astype(np.float64) - target.astype(np.float64), 0.0)
    return {
        "squared": np.square(error).sum(axis=1),
        "absolute": np.abs(error).sum(axis=1),
        "signed": error.sum(axis=1),
        "count": mask.sum(axis=1).astype(np.float64),
    }


def score_terms(terms, index):
    count = terms["count"][index].sum(axis=0)
    if np.any(count <= 0):
        raise ValueError("Every fixed channel needs observed targets in each reported period")
    sums = {key: terms[key][index].sum(axis=0) for key in ("squared", "absolute", "signed")}
    result = {metric: float(np.mean(sums[key] / count)) for metric, key in ERROR_KEYS.items()}
    result.update(
        channel_mse=(sums["squared"] / count).tolist(),
        channel_mae=(sums["absolute"] / count).tolist(),
        channel_signed_mean_error=(sums["signed"] / count).tolist(),
        squared_error_sums=sums["squared"].tolist(),
        absolute_error_sums=sums["absolute"].tolist(),
        signed_error_sums=sums["signed"].tolist(),
        target_counts=count.astype(np.int64).tolist(),
        target_count=int(count.sum()),
        origins=len(index),
    )
    return result


def mean_seed_terms(members):
    if any(not np.array_equal(members[0]["count"], row["count"]) for row in members[1:]):
        raise ValueError("Selected seed target denominators differ")
    return {
        key: members[0]["count"].copy() if key == "count" else np.mean(np.stack([row[key] for row in members]), axis=0)
        for key in ("squared", "absolute", "signed", "count")
    }


def reused_rows(dataset):
    parent = read_json(HERE / "parent_reuse_manifest.json")["units"][dataset]["levels"]
    rows = []
    for row in parent:
        item = dict(row)
        item.update(method="ORIGINAL_LEVEL", original_method="LEVEL", source_kind="parent_reuse_manifest")
        rows.append(item)
    c2_reuse = read_json("research/level_chronos2_controls_v1/reuse_manifest.json")["units"][dataset]["rows"]
    for row in c2_reuse:
        if row["method"] not in {"B", "F0", "MSE_LORA", "DIRECT_NLINEAR"}:
            continue
        item = dict(row)
        item["original_method"] = row["method"]
        item["method"] = "COMPRESS" if row["method"] == "B" else row["method"]
        item["source_kind"] = "chronos2_reuse_manifest"
        rows.append(item)
    rows.extend(raw_reuse_rows(dataset))
    for path, source_kind in (
        ("research/level_bolt_backbone_scaling_v1_20261006/prediction_manifest.json", "bolt_minitiny_prediction_manifest"),
        ("research/level_chronos2_controls_v1/prediction_manifest.json", "chronos2_prediction_manifest"),
    ):
        manifest = read_json(path)
        for method, record in manifest["units"][dataset].items():
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "original_method": method,
                    "id": dataset + "_" + method,
                    "seed": None,
                    "test_prediction": record["prediction"],
                    "source_evaluation": artifact(path),
                    "source_pointer": f"/units/{dataset}/{method}",
                    "source_kind": source_kind,
                }
            )
    return rows


def raw_reuse_rows(dataset):
    if dataset == "peacock_education":
        doc = read_json("research/tsfm_peft_peacock_matched_raw_v17_20261002/prediction_manifest.json")
        return [
            {
                "dataset": dataset,
                "method": "MATCHED_RAW",
                "original_method": "matched_raw",
                "id": model["id"],
                "seed": model["seed"],
                "test_prediction": {**model["prediction"], "key": model.get("prediction_key", "prediction"), "origins_key": "origins"},
                "source_evaluation": artifact("research/tsfm_peft_peacock_matched_raw_v17_20261002/prediction_manifest.json"),
                "source_pointer": f"/models/{model['id']}",
                "source_kind": "matched_raw_v17_prediction_manifest",
            }
            for model in doc.get("models", [])
            if model.get("role") == "matched_raw"
        ]
    eval_path = {
        "robin": "research/tsfm_peft_level_confirmation_v6_20260928/evaluation01.json",
        "jena": "research/tsfm_peft_practical_controls_v7_20260928/jena_eval01.json",
    }.get(dataset)
    if eval_path is None:
        return []
    doc = read_json(eval_path)
    rows = []
    for key, model in doc.get("models", {}).items():
        spec = model.get("spec", {})
        role = model.get("role") or spec.get("family") or key
        if "raw" not in str(role).lower() and "raw" not in key.lower():
            continue
        rows.append(
            {
                "dataset": dataset,
                "method": "MATCHED_RAW",
                "original_method": role,
                "id": spec.get("id", key),
                "seed": spec.get("seed"),
                "test_prediction": {**model["prediction"], "key": "prediction", "origins_key": "origins"},
                "expected_period_scores": {period: {metric: model["scores"][period][metric] for metric in ("mse", "mae")} for period in PERIODS},
                "source_evaluation": artifact(eval_path),
                "source_pointer": f"/models/{key}",
                "source_kind": "matched_raw_old_evaluation",
            }
        )
    return rows


def paired_intervals(groups, indices, unit_contract, job):
    config = unit_contract["bootstrap"]
    draws, block, seed, salt = (config[key] for key in ("draws", "block_origins", "seed", "seed_sequence_salt"))
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
            terms = group["terms"]
            denominator = matrix @ terms["count"]
            if np.any(denominator <= 0):
                raise ValueError("A bootstrap draw has an unobserved fixed channel")
            sampled[period][method] = {
                metric: np.mean((matrix @ terms[key]) / denominator, axis=1)
                for metric, key in (("mse", "squared"), ("mae", "absolute"))
            }
    rows = []
    for right in PRIMARY_AND_SECONDARY_RIGHT:
        for period in PERIODS:
            for metric in ("mse", "mae"):
                left_value = groups[METHOD]["periods"][period][metric]
                right_value = groups[right]["periods"][period][metric]
                left_draw, right_draw = sampled[period][METHOD][metric], sampled[period][right][metric]
                relative_available = right_value != 0 and bool(np.all(right_draw != 0))
                rows.append(
                    {
                        "dataset": unit_contract["dataset"],
                        "left": METHOD,
                        "right": right,
                        "comparison_role": "primary" if right == "ORIGINAL_LEVEL" else "secondary",
                        "period": period,
                        "metric": metric,
                        "left_value": left_value,
                        "right_value": right_value,
                        "absolute_difference": left_value - right_value,
                        "relative_percent": 100 * (left_value / right_value - 1) if right_value != 0 else None,
                        "absolute_ci95": np.quantile(left_draw - right_draw, [0.025, 0.975]).tolist(),
                        "relative_ci95_percent": np.quantile(100 * (left_draw / right_draw - 1), [0.025, 0.975]).tolist() if relative_available else None,
                        "relative_ci_unavailable_reason": None if relative_available else "Zero reference point/draw denominator",
                        "negative_favors": METHOD,
                        "positive_favors": right,
                        "origins": len(indices[period]),
                        "bootstrap_draws": draws,
                        "block_origins": block,
                        "rng_seed": seed,
                        "rng_salt": salt,
                        "period_stratified": True,
                        "fixed_seeds_conditional": True,
                        "not_equivalence_or_noninferiority": True,
                    }
                )
    return rows, {period: hashlib.sha256(matrix.tobytes()).hexdigest() for period, matrix in weights.items()}


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot publish an empty table")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with resolve(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False) if isinstance(value, (dict, list)) else "" if value is None else value for key, value in row.items()})


def score_all(job):
    check_selection_seal()
    if (HERE / "evaluation.json").exists():
        raise FileExistsError("Evaluation is already complete; use verify")
    manifest = read_json(HERE / "prediction_manifest.json")
    if manifest.get("status") != "complete" or manifest.get("evaluation_units") != 6:
        raise RuntimeError("All six selected ED TEST predictions must finish before scoring")
    model_rows, channel_rows, paired_rows, replay = [], [], [], []
    evaluation = {
        "schema": "matched_learned_ed_v18_evaluation_1",
        "status": "scoring",
        "method": METHOD,
        "prediction_manifest": artifact(HERE / "prediction_manifest.json"),
        "selection": artifact(HERE / "selection.json"),
        "selection_seal": artifact(HERE / "selection_seal.json"),
        "prediction_ensemble": False,
        "units": {},
        "old_score_replay": replay,
        "paired_bootstrap_weight_hashes": {},
    }
    selected = selected_units()
    selection_sha256 = digest(HERE / "selection_seal.json")
    for dataset in DATASETS:
        data = load_data(dataset, include_test=True)
        origins, indices = origin_contract(data)
        target, mask = targets(data, origins)
        mask = mask.astype(bool)
        terms, rows = {}, []
        for row in reused_rows(dataset):
            prediction = read_prediction(row["test_prediction"], origins, target.shape)
            terms[row["id"]] = error_terms(prediction, target, mask)
            row = {**row, "reused": True, "periods": {period: score_terms(terms[row["id"]], index) for period, index in indices.items()}}
            if row.get("expected_period_scores"):
                for period in PERIODS:
                    for metric in ("mse", "mae"):
                        expected = row["expected_period_scores"][period][metric]
                        actual = row["periods"][period][metric]
                        tolerance = max(1e-10, abs(expected) * 1e-9)
                        check = {"dataset": dataset, "id": row["id"], "period": period, "metric": metric, "expected": expected, "actual": actual, "absolute_difference": abs(actual - expected), "tolerance": tolerance, "pass": abs(actual - expected) <= tolerance, "source": row.get("source_evaluation"), "source_pointer": row.get("source_pointer")}
                        replay.append(check)
                        if not check["pass"]:
                            save_json(HERE / "old_score_replay_failure.json", {"failed": check, "checks": replay})
                            raise RuntimeError("Original score reproduction failed: " + row["id"])
            rows.append(row)
        for row in selected[dataset]:
            record = verify_new_receipt(manifest["units"][dataset][row["id"]], dataset, row, origins, target.shape, selection_sha256)
            prediction = read_prediction(record["prediction"], origins, target.shape)
            terms[row["id"]] = error_terms(prediction, target, mask)
            rows.append({**record, "test_prediction": record["prediction"], "reused": False, "source_kind": "matched_learned_ed_v18_new_prediction", "periods": {period: score_terms(terms[row["id"]], index) for period, index in indices.items()}})
        groups = {}
        for method in (METHOD, *REUSED_METHOD_ORDER):
            members = [row for row in rows if row["method"] == method]
            if not members:
                continue
            merged = mean_seed_terms([terms[row["id"]] for row in members])
            groups[method] = {"ids": [row["id"] for row in members], "seeds": [row.get("seed") for row in members], "periods": {period: score_terms(merged, index) for period, index in indices.items()}, "terms": merged, "aggregation": "Mean individual seed losses; single zero-shot counted once"}
        missing = sorted(set((METHOD, *PRIMARY_AND_SECONDARY_RIGHT)).difference(groups))
        if missing:
            raise RuntimeError("Required evaluation groups missing for " + dataset + ": " + ",".join(missing))
        pairs, weight_hashes = paired_intervals(groups, indices, dataset_contract(dataset), job)
        paired_rows.extend(pairs)
        evaluation["paired_bootstrap_weight_hashes"][dataset] = weight_hashes
        for row in rows:
            append_score_rows(model_rows, channel_rows, dataset, "individual", row["method"], row["id"], row.get("seed"), row.get("source_kind"), row.get("reused"), row["periods"])
        for method, group in groups.items():
            append_score_rows(model_rows, channel_rows, dataset, "method", method, "+".join(group["ids"]), json.dumps(group["seeds"]), "new_or_reused_group", method != METHOD, group["periods"])
        evaluation["units"][dataset] = {
            "origins_sha256": array_hash(origins),
            "period_origin_counts": {period: int(len(indices[period])) for period in PERIODS},
            "groups": {method: {key: value for key, value in group.items() if key != "terms"} for method, group in groups.items()},
        }
    write_csv(HERE / "accuracy_models.csv", model_rows)
    write_csv(HERE / "accuracy_channels.csv", channel_rows)
    write_csv(HERE / "paired_comparisons.csv", paired_rows)
    save_json(HERE / "old_score_replay.json", {"status": "pass", "checks": replay})
    evaluation.update(
        status="complete",
        completed_utc=time.time(),
        row_counts={"accuracy_models": len(model_rows), "accuracy_channels": len(channel_rows), "paired_comparisons": len(paired_rows), "old_score_replay": len(replay)},
        accuracy_models_csv=artifact(HERE / "accuracy_models.csv"),
        accuracy_channels_csv=artifact(HERE / "accuracy_channels.csv"),
        paired_comparisons_csv=artifact(HERE / "paired_comparisons.csv"),
        old_score_replay_json=artifact(HERE / "old_score_replay.json"),
    )
    save_json(HERE / "evaluation.json", evaluation)
    return evaluation


def append_score_rows(model_rows, channel_rows, dataset, aggregation, method, row_id, seed, source_kind, reused, periods):
    for period in PERIODS:
        scores = periods[period]
        model_rows.append(
            {
                "dataset": dataset,
                "aggregation": aggregation,
                "method": method,
                "id": row_id,
                "seed": seed,
                "period": period,
                "source_kind": source_kind,
                "reused": reused,
                **{metric: scores[metric] for metric in METRICS},
                "target_count": scores["target_count"],
                "origins": scores["origins"],
            }
        )
        for channel, value in enumerate(scores["channel_mse"]):
            channel_rows.append(
                {
                    "dataset": dataset,
                    "aggregation": aggregation,
                    "method": method,
                    "id": row_id,
                    "seed": seed,
                    "period": period,
                    "channel": channel,
                    "mse": value,
                    "mae": scores["channel_mae"][channel],
                    "signed_mean_error": scores["channel_signed_mean_error"][channel],
                    "target_count": scores["target_counts"][channel],
                }
            )


def verify_all():
    check_selection_seal()
    manifest = read_json(HERE / "prediction_manifest.json")
    evaluation = read_json(HERE / "evaluation.json")
    if manifest.get("status") != "complete" or manifest.get("evaluation_units") != 6:
        raise RuntimeError("Prediction manifest incomplete")
    if evaluation.get("status") != "complete" or evaluation.get("method") != METHOD:
        raise RuntimeError("Evaluation incomplete")
    for key in ("accuracy_models_csv", "accuracy_channels_csv", "paired_comparisons_csv", "old_score_replay_json"):
        artifact(evaluation[key]["path"], evaluation[key]["sha256"])
    expected_pairs = len(DATASETS) * len(PRIMARY_AND_SECONDARY_RIGHT) * len(PERIODS) * 2
    if evaluation["row_counts"]["paired_comparisons"] != expected_pairs:
        raise RuntimeError("Primary/secondary paired comparison count changed")
    return {"status": "pass", "prediction_units": 6, "row_counts": evaluation["row_counts"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("schema", "predict", "score", "verify"), required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.phase == "schema":
        print(json.dumps(schema_contract(), ensure_ascii=False, indent=2))
        return
    configure()
    if args.phase == "predict":
        with Job("gpu" if args.device == "cuda" else "cpu", "v18_selected_ed_test_predict", metadata={"phase": "predict", "device": args.device}) as job:
            result = {"status": predict_all(job, args.device)["status"]}
    elif args.phase == "score":
        with Job("cpu", "v18_paired_scoring", metadata={"phase": "score", "new_test_predictions": 0}) as job:
            result = {"status": score_all(job)["status"]}
    else:
        result = verify_all()
    print(json.dumps({"phase": args.phase, "result": result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
