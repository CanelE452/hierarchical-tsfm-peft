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
    verify_artifact(receipt)
    with np.load(resolve(receipt["path"]), allow_pickle=False) as archive:
        key = receipt.get("key") or receipt.get("prediction_key") or "prediction"
        value = archive[key]
        saved_origins = archive[receipt.get("origins_key", "origins")]
    if saved_origins.dtype != np.int64 or not np.array_equal(saved_origins, origins):
        raise ValueError("Prediction origin order differs: " + str(receipt["path"]))
    if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError("Prediction shape, dtype or finiteness differs: " + str(receipt["path"]))
    return value


def verify_artifact(receipt, expected=None):
    actual = artifact(receipt["path"], receipt["sha256"])
    if receipt.get("bytes") != actual["bytes"]:
        raise ValueError("Artifact byte count differs: " + receipt["path"])
    if expected is not None:
        wanted = artifact(expected["path"], expected["sha256"])
        if wanted != actual or expected.get("bytes") != wanted["bytes"]:
            raise ValueError("Artifact differs from its immutable reference: " + receipt["path"])
    return actual


def seal_reference(path, seal):
    references = [row for row in seal["artifacts"] if resolve(row["path"]).resolve() == resolve(path).resolve()]
    if len(references) != 1:
        raise ValueError("Exactly one joint-seal reference is required: " + str(path))
    return verify_artifact(references[0])


def verified_seals():
    seal = check_selection_seal()
    for document in (read_json(HERE / "source_seal.json"), seal):
        for reference in document["artifacts"]:
            verify_artifact(reference)
    if (seal.get("fit_count") != 12 or seal.get("test_predictions_at_seal") != 0
            or seal.get("prediction_ensemble") is not False):
        raise ValueError("Joint seal must follow twelve fits and precede every new TEST")
    seal_reference(HERE / "selection.json", seal)
    seal_reference(HERE / "source_seal.json", seal)
    return seal


def selected_units():
    selection = read_json(HERE / "selection.json")
    if selection.get("prediction_ensemble") is not False or selection.get("test_used_for_selection") is not False:
        raise RuntimeError("The selected ED campaign must not ensemble predictions")
    if set(selection.get("units", {})) != set(DATASETS):
        raise RuntimeError("Joint selection must contain exactly the three fixed datasets")
    units = {}
    for dataset in DATASETS:
        unit = selection["units"][dataset]
        if unit.get("seeds") != [92601, 92602] or len(unit.get("run_ids", [])) != 2:
            raise RuntimeError("Selection unit does not expose exactly two selected ED seeds: " + dataset)
        if unit.get("lr") not in (1e-3, 1e-4) or any(len(unit.get(key, [])) != 2 for key in ("checkpoints", "selected_epochs", "best_val_prediction")):
            raise RuntimeError("Selected LR or two-seed artifact grid differs: " + dataset)
        rows = []
        for index, checkpoint in enumerate(unit["checkpoints"]):
            verify_artifact(checkpoint)
            verify_artifact(unit["best_val_prediction"][index])
            epoch = unit["selected_epochs"][index]
            if type(epoch) is not int or not 0 <= epoch <= 120:
                raise RuntimeError("Selected epoch is outside the prescribed grid")
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
    ids = [row["id"] for rows in units.values() for row in rows]
    if len(ids) != 6 or len(set(ids)) != 6:
        raise RuntimeError("Joint selection contains duplicate prediction run IDs")
    seal = verified_seals()
    if sorted(seal.get("selected_run_ids", [])) != sorted(ids):
        raise RuntimeError("Selected run IDs differ from the joint seal")
    return units


def prediction_receipt_path(dataset, run_id):
    return HERE / "prediction_receipts" / f"{dataset}_{run_id}.json"


def verify_new_receipt(record, dataset, row, origins, shape, selection_sha256, *, allow_running_test=False):
    if record.get("status") != "complete" or record.get("dataset") != dataset:
        raise ValueError("Completed prediction receipt belongs to another dataset")
    if record.get("method") != METHOD or record.get("id") != row["id"] or record.get("seed") != row["seed"]:
        raise ValueError("Completed prediction receipt belongs to another selected ED unit")
    if selection_sha256 != digest(HERE / "selection_seal.json") or record.get("selection_seal_sha256") != selection_sha256:
        raise ValueError("Completed prediction receipt is bound to another selection seal")
    if record.get("lr") != row["lr"] or record.get("selected_epoch") != row["selected_epoch"]:
        raise ValueError("Prediction receipt differs from the selected LR or epoch")
    verify_artifact(record["checkpoint"], row["checkpoint"])
    verify_artifact(record["evaluation_source"], artifact(HERE / "evaluate_v18.py"))
    verify_artifact(record["source_seal"], artifact(HERE / "source_seal.json"))
    contract = dataset_contract(dataset)
    verify_artifact(record["data"], contract["test"])
    canonical = np.concatenate([np.asarray(contract["origins"][period]["values"], dtype=np.int64) for period in PERIODS[:2]])
    for period in PERIODS[:2]:
        if array_hash(np.asarray(contract["origins"][period]["values"], dtype=np.int64)) != contract["origins"][period]["values_sha256"]:
            raise ValueError("Canonical TEST origin values differ from their sealed hashes")
    if not np.array_equal(origins, canonical) or tuple(shape) != (len(canonical), contract["h"], contract["c"]):
        raise ValueError("Prediction TEST origins or axes differ from the parent data contract")
    if record.get("origin_values_sha256") != array_hash(canonical) or record.get("origin_shape_fp32_finite") is not True:
        raise ValueError("Prediction origin receipt differs from the canonical contract")
    prediction = read_prediction(record["prediction"], canonical, tuple(shape))
    if record["prediction"].get("shape") != list(prediction.shape) or record["prediction"].get("dtype") != "float32":
        raise ValueError("Prediction metadata differs from actual NPZ bytes")
    seal = verified_seals()
    result_path = HERE / "runs" / row["id"] / "result.json"
    seal_reference(result_path, seal)
    result = read_json(result_path)
    if any(result.get(key) != value for key, value in {"status": "complete", "id": row["id"], "dataset": dataset,
            "seed": row["seed"], "lr": row["lr"], "selected_epoch": row["selected_epoch"], "test_accessed": False}.items()):
        raise ValueError("Prediction selected fit result differs from joint selection")
    verify_artifact(result["checkpoint"], row["checkpoint"])
    initial_reference = read_json(HERE / "preflight.json")["initial_models"]
    verify_artifact(initial_reference)
    initial = read_json(initial_reference["path"])["units"][dataset]
    model = record.get("model_receipt", {})
    for key, value in result["model_receipt"].items():
        if model.get(key) != value:
            raise ValueError("Prediction model differs from selected fit receipt: " + key)
    for key, value in initial["model_receipt"].items():
        if key not in {"state_sha256", "adapter_tensor_sha256"} and model.get(key) != value:
            raise ValueError("Prediction model differs from immutable model contract: " + key)
    if (model.get("trainable_parameter_names") != ["encoder.weight", "decoder.weight"]
            or model.get("trainable_parameter_count") != 2 * contract["c"] * contract["k"]
            or model.get("precision") != "float32" or model.get("all_parameters_fp32") is not True
            or model.get("eval_mode") is not True or model.get("backbone_eval_mode") is not True):
        raise ValueError("Prediction model must be FP32 eval with E/D-only trainable weights")
    frozen = result["frozen_backbone"]
    if (frozen.get("unchanged") is not True or frozen.get("before_sha256") != frozen.get("after_sha256")
            or model.get("backbone_state_sha256") != frozen["before_sha256"]
            or model.get("backbone_state_sha256") != initial["backbone_state_sha256"]
            or model.get("adapter_tensor_sha256") != result["adapter_selected_sha256"]
            or model.get("basis_values_sha256") != result["basis_values_sha256"]):
        raise ValueError("Prediction state differs from selected E/D, frozen backbone or PCA")
    state = record.get("state_preservation", {})
    if (state.get("unchanged") is not True or state.get("before_sha256") != state.get("after_sha256")
            or state.get("before_sha256") != model.get("state_sha256")
            or not isinstance(model.get("state_sha256"), str) or len(model["state_sha256"]) != 64
            or state.get("eval_mode") is not True or state.get("backbone_eval_mode") is not True
            or record.get("additional_fit_count") != 0):
        raise ValueError("Completed prediction lacks state-preservation evidence")
    test_id = dataset + "::" + row["id"]
    entries = [entry for entry in read_json(HERE / "ledger.json")["test_predictions"] if entry.get("id") == test_id]
    if len(entries) != 1:
        raise ValueError("Exactly one matching TEST ledger entry is required")
    entry = entries[0]
    metadata = {"dataset": dataset, "prediction_run_id": row["id"], "seed": row["seed"], "method": METHOD,
                "lr": row["lr"], "selected_epoch": row["selected_epoch"], "selection_seal_sha256": selection_sha256}
    if any(entry.get(key) != value for key, value in metadata.items()):
        raise ValueError("TEST ledger metadata differs from selected prediction")
    verify_artifact(entry["checkpoint"], row["checkpoint"])
    if allow_running_test:
        if entry.get("status") != "running":
            raise ValueError("Only the current newly generated TEST may be running before receipt save")
    else:
        if entry.get("status") != "complete":
            raise ValueError("A failed, running or missing TEST cannot be reused as complete")
        verify_artifact(entry["prediction"], record["prediction"])
        receipt_path = prediction_receipt_path(dataset, row["id"])
        verify_artifact(entry["receipt"], artifact(receipt_path))
        if read_json(receipt_path) != record:
            raise ValueError("Prediction sidecar differs from completed ledger/manifest receipt")
    return record


def verify_prediction_manifest(manifest, selection):
    if (manifest.get("status") != "complete" or manifest.get("method") != METHOD
            or manifest.get("evaluation_units") != 6 or manifest.get("prediction_ensemble") is not False
            or manifest.get("prediction_batch_origins") != 4 or set(manifest.get("units", {})) != set(DATASETS)):
        raise RuntimeError("Prediction manifest must contain exactly six sealed E/D units")
    selection_sha256 = digest(HERE / "selection_seal.json")
    if manifest.get("selection_seal_sha256") != selection_sha256:
        raise RuntimeError("Prediction manifest belongs to another joint selection seal")
    for key, path in (("selection_seal", HERE / "selection_seal.json"), ("source_seal", HERE / "source_seal.json"),
                      ("evaluation_source", HERE / "evaluate_v18.py")):
        verify_artifact(manifest[key], artifact(path))
    expected_test_ids = set()
    origin_count = 0
    for dataset in DATASETS:
        expected_ids = {row["id"] for row in selection[dataset]}
        if set(manifest["units"][dataset]) != expected_ids:
            raise RuntimeError("Prediction manifest has extra, missing or swapped units: " + dataset)
        data = load_data(dataset, include_test=True)
        origins, _ = origin_contract(data)
        shape = (len(origins), 48, int(data["_C"]))
        for row in selection[dataset]:
            verify_new_receipt(manifest["units"][dataset][row["id"]], dataset, row, origins, shape, selection_sha256)
            expected_test_ids.add(dataset + "::" + row["id"])
            origin_count += len(origins)
    ledger_entries = read_json(HERE / "ledger.json")["test_predictions"]
    if len(ledger_entries) != 6 or {entry.get("id") for entry in ledger_entries} != expected_test_ids:
        raise RuntimeError("TEST ledger differs from the exact six-member selected prediction grid")
    if manifest.get("full_channel_prediction_origins") != origin_count:
        raise RuntimeError("Prediction manifest origin count differs from actual data units")
    return manifest


def predict_all(job, device):
    import torch
    from model_v18 import restore_model

    check_selection_seal()
    selection_sha256 = digest(HERE / "selection_seal.json")
    selection = selected_units()
    manifest_path = HERE / "prediction_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        return verify_prediction_manifest(manifest, selection)

    folder = CACHE / "predictions01"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "matched_learned_ed_v18_prediction_manifest_1",
        "status": "predicting",
        "method": METHOD,
        "selection_seal": artifact(HERE / "selection_seal.json"),
        "selection_seal_sha256": selection_sha256,
        "source_seal": artifact(HERE / "source_seal.json"),
        "evaluation_source": artifact(HERE / "evaluate_v18.py"),
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
            register_test(test_id, {"dataset": dataset, "prediction_run_id": row["id"], "seed": row["seed"],
                                   "method": METHOD, "lr": row["lr"], "selected_epoch": row["selected_epoch"],
                                   "checkpoint": row["checkpoint"], "selection_seal_sha256": selection_sha256})
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
                    "source_seal": artifact(HERE / "source_seal.json"),
                    "evaluation_source": artifact(HERE / "evaluate_v18.py"),
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
                verify_new_receipt(record, dataset, row, origins, shape, selection_sha256, allow_running_test=True)
                save_json(receipt_path, record)
                manifest["units"][dataset][row["id"]] = record
                save_json(HERE / "prediction_manifest_partial.json", manifest)
                finish_test(test_id, {"prediction": artifact(npz_path), "receipt": artifact(receipt_path), "elapsed_s": record["elapsed_s"]})
                verify_new_receipt(record, dataset, row, origins, shape, selection_sha256)
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
    verify_prediction_manifest(manifest, selection)
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
    phase0 = read_json(HERE / "phase0_audit.json")
    verify_artifact(phase0["parent_reuse_manifest"]["artifact"], artifact(HERE / "parent_reuse_manifest.json"))
    parent = read_json(HERE / "parent_reuse_manifest.json")["units"][dataset]["levels"]
    rows = []
    for row in parent:
        item = dict(row)
        item.update(method="ORIGINAL_LEVEL", original_method="LEVEL", source_kind="parent_reuse_manifest")
        rows.append(item)
    c2_reuse = read_pinned_reuse("research/level_chronos2_controls_v1/reuse_manifest.json")["units"][dataset]["rows"]
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
        manifest = read_pinned_reuse(path)
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
    for row in rows:
        verify_artifact(row["source_evaluation"])
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Reused prediction row IDs must be unique within a dataset")
    return rows


def read_pinned_reuse(path):
    references = read_json(HERE / "phase0_audit.json")["manifest_artifacts"]
    matches = [row for row in references if resolve(row["path"]).resolve() == resolve(path).resolve()]
    if len(matches) != 1:
        raise ValueError("Exactly one phase0 manifest pin is required: " + str(path))
    verify_artifact(matches[0])
    return read_json(path)


def raw_reuse_rows(dataset):
    if dataset == "peacock_education":
        doc = read_pinned_reuse("research/tsfm_peft_peacock_matched_raw_v17_20261002/prediction_manifest.json")
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
    parent = read_json(HERE / "parent_reuse_manifest.json")["units"][dataset]["levels"]
    references = [row["source_evaluation"] for row in parent if resolve(row["source_evaluation"]["path"]).resolve() == resolve(eval_path).resolve()]
    if not references or any(row != references[0] for row in references):
        raise ValueError("Parent LEVEL does not pin the reused RAW evaluation")
    verify_artifact(references[0])
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
    verified_seals()
    if (HERE / "evaluation.json").exists():
        raise FileExistsError("Evaluation is already complete; use verify")
    manifest = read_json(HERE / "prediction_manifest.json")
    selected = selected_units()
    verify_prediction_manifest(manifest, selected)
    model_rows, channel_rows, paired_rows, replay = [], [], [], []
    evaluation = {
        "schema": "matched_learned_ed_v18_evaluation_1",
        "status": "scoring",
        "method": METHOD,
        "prediction_manifest": artifact(HERE / "prediction_manifest.json"),
        "selection": artifact(HERE / "selection.json"),
        "selection_seal": artifact(HERE / "selection_seal.json"),
        "source_seal": artifact(HERE / "source_seal.json"),
        "evaluation_source": artifact(HERE / "evaluate_v18.py"),
        "phase0_audit": artifact(HERE / "phase0_audit.json"),
        "prediction_ensemble": False,
        "units": {},
        "old_score_replay": replay,
        "paired_bootstrap_weight_hashes": {},
    }
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
            if row["id"] in terms:
                raise ValueError("New and reused prediction IDs must be distinct")
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
    verified_seals()
    selected = selected_units()
    manifest = read_json(HERE / "prediction_manifest.json")
    evaluation = read_json(HERE / "evaluation.json")
    verify_prediction_manifest(manifest, selected)
    if (evaluation.get("status") != "complete" or evaluation.get("method") != METHOD
            or evaluation.get("prediction_ensemble") is not False or set(evaluation.get("units", {})) != set(DATASETS)):
        raise RuntimeError("Evaluation incomplete")
    for key, path in (("prediction_manifest", HERE / "prediction_manifest.json"), ("selection", HERE / "selection.json"),
                      ("selection_seal", HERE / "selection_seal.json"), ("source_seal", HERE / "source_seal.json"),
                      ("evaluation_source", HERE / "evaluate_v18.py"), ("phase0_audit", HERE / "phase0_audit.json")):
        verify_artifact(evaluation[key], artifact(path))
    for key in ("accuracy_models_csv", "accuracy_channels_csv", "paired_comparisons_csv", "old_score_replay_json"):
        verify_artifact(evaluation[key], artifact(HERE / {
            "accuracy_models_csv": "accuracy_models.csv", "accuracy_channels_csv": "accuracy_channels.csv",
            "paired_comparisons_csv": "paired_comparisons.csv", "old_score_replay_json": "old_score_replay.json"}[key]))
    with (HERE / "accuracy_models.csv").open(encoding="utf-8", newline="") as handle:
        model_rows = list(csv.DictReader(handle))
    with (HERE / "accuracy_channels.csv").open(encoding="utf-8", newline="") as handle:
        channel_rows = list(csv.DictReader(handle))
    with (HERE / "paired_comparisons.csv").open(encoding="utf-8", newline="") as handle:
        pairs = list(csv.DictReader(handle))
    model_grid, channel_grid, expected_replay, expected_metadata = set(), set(), {}, {}
    model_lookup = {}
    for table_row in model_rows:
        key = tuple(table_row[name] for name in ("dataset", "aggregation", "method", "id", "period"))
        if key in model_lookup:
            raise RuntimeError("Accuracy model CSV contains duplicate grid members")
        model_lookup[key] = table_row
    for dataset in DATASETS:
        contract = dataset_contract(dataset)
        data = load_data(dataset, include_test=True)
        origins, indices = origin_contract(data)
        unit = evaluation["units"][dataset]
        if (unit.get("origins_sha256") != array_hash(origins)
                or unit.get("period_origin_counts") != {period: len(index) for period, index in indices.items()}):
            raise RuntimeError("Evaluation origin receipt differs from canonical TEST periods")
        members = reused_rows(dataset) + [manifest["units"][dataset][row["id"]] for row in selected[dataset]]
        if len({row["id"] for row in members}) != len(members):
            raise RuntimeError("Evaluation individual IDs overlap")
        expected_methods = {row["method"] for row in members}
        if set(unit.get("groups", {})) != expected_methods:
            raise RuntimeError("Evaluation method groups differ from actual selected/reused members")
        grouped_members = {}
        for row in members:
            grouped_members.setdefault(row["method"], []).append(row)
            for period in PERIODS:
                key = (dataset, "individual", row["method"], row["id"], period)
                model_grid.add(key)
                expected_metadata[key] = {
                    "seed": "" if row.get("seed") is None else str(row["seed"]),
                    "source_kind": "matched_learned_ed_v18_new_prediction" if row["method"] == METHOD else row["source_kind"],
                    "reused": str(row["method"] != METHOD),
                }
                channel_grid.update((*key, str(channel)) for channel in range(contract["c"]))
                if row.get("expected_period_scores"):
                    for metric in ("mse", "mae"):
                        expected_replay[(dataset, row["id"], period, metric)] = row
        for method, group_members in grouped_members.items():
            group = unit["groups"][method]
            ids = [row["id"] for row in group_members]
            if group.get("ids") != ids or group.get("seeds") != [row.get("seed") for row in group_members]:
                raise RuntimeError("Evaluation aggregation members differ from prediction grid")
            for period in PERIODS:
                key = (dataset, "method", method, "+".join(ids), period)
                model_grid.add(key)
                expected_metadata[key] = {"seed": json.dumps(group["seeds"]), "source_kind": "new_or_reused_group", "reused": str(method != METHOD)}
                channel_grid.update((*key, str(channel)) for channel in range(contract["c"]))
                actual = model_lookup.get(key, {})
                scores = group["periods"][period]
                if any(float(actual.get(metric, "nan")) != scores[metric] for metric in METRICS):
                    raise RuntimeError("Accuracy CSV differs from evaluation method scores")
        hashes = evaluation["paired_bootstrap_weight_hashes"][dataset]
        if set(hashes) != set(PERIODS) or any(not isinstance(value, str) or len(value) != 64 for value in hashes.values()):
            raise RuntimeError("Paired bootstrap weight hash receipt is incomplete")
    if set(model_lookup) != model_grid or len(model_rows) != len(model_grid):
        raise RuntimeError("Accuracy model CSV has extra or missing grid members")
    actual_channels = [tuple(row[name] for name in ("dataset", "aggregation", "method", "id", "period", "channel")) for row in channel_rows]
    if len(actual_channels) != len(channel_grid) or set(actual_channels) != channel_grid:
        raise RuntimeError("Accuracy channel CSV has extra, missing or duplicate grid members")
    channel_lookup = dict(zip(actual_channels, channel_rows))
    for key, model_row in model_lookup.items():
        if any(model_row.get(name) != value for name, value in expected_metadata[key].items()):
            raise RuntimeError("Accuracy CSV seed or new/reused source metadata differs")
        channel_count = dataset_contract(key[0])["c"]
        rows = [channel_lookup[(*key, str(channel))] for channel in range(channel_count)]
        if any(row.get("seed") != expected_metadata[key]["seed"] for row in rows):
            raise RuntimeError("Channel CSV seed differs from selected/reused members")
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in rows])
            if not np.isfinite(values).all() or not np.isclose(float(values.mean()), float(model_row[metric]), rtol=1e-12, atol=1e-12):
                raise RuntimeError("Channel-macro scores differ from model CSV")
        counts = [int(row["target_count"]) for row in rows]
        if min(counts) <= 0 or sum(counts) != int(model_row["target_count"]):
            raise RuntimeError("Channel target counts differ from model CSV")
        if int(model_row["origins"]) != evaluation["units"][key[0]]["period_origin_counts"][key[4]]:
            raise RuntimeError("Model CSV period origin count differs")
    pair_grid = {(dataset, METHOD, right, period, metric) for dataset in DATASETS
                 for right in PRIMARY_AND_SECONDARY_RIGHT for period in PERIODS for metric in ("mse", "mae")}
    actual_pairs = [tuple(row[name] for name in ("dataset", "left", "right", "period", "metric")) for row in pairs]
    if len(actual_pairs) != len(pair_grid) or set(actual_pairs) != pair_grid:
        raise RuntimeError("Paired CSV has extra, missing or duplicate comparison members")
    for pair in pairs:
        dataset, period, metric = pair["dataset"], pair["period"], pair["metric"]
        left = evaluation["units"][dataset]["groups"][METHOD]["periods"][period][metric]
        right = evaluation["units"][dataset]["groups"][pair["right"]]["periods"][period][metric]
        config = dataset_contract(dataset)["bootstrap"]
        if (float(pair["left_value"]) != left or float(pair["right_value"]) != right
                or float(pair["absolute_difference"]) != left - right
                or any(int(pair[name]) != config[key] for name, key in (("bootstrap_draws", "draws"),
                       ("block_origins", "block_origins"), ("rng_seed", "seed"), ("rng_salt", "seed_sequence_salt")))):
            raise RuntimeError("Paired CSV point values or bootstrap recipe differ")
        interval = json.loads(pair["absolute_ci95"])
        if len(interval) != 2 or not np.isfinite(interval).all() or interval[0] > interval[1]:
            raise RuntimeError("Paired CSV confidence interval is invalid")
    replay = read_json(HERE / "old_score_replay.json")
    checks = replay.get("checks", [])
    replay_keys = [(row["dataset"], row["id"], row["period"], row["metric"]) for row in checks]
    if (replay.get("status") != "pass" or checks != evaluation.get("old_score_replay")
            or len(replay_keys) != len(expected_replay) or set(replay_keys) != set(expected_replay)):
        raise RuntimeError("Old-score replay receipt has extra, missing or duplicate checks")
    for check, key in zip(checks, replay_keys):
        row = expected_replay[key]
        expected = row["expected_period_scores"][key[2]][key[3]]
        actual = float(model_lookup[(key[0], "individual", row["method"], key[1], key[2])][key[3]])
        tolerance = max(1e-10, abs(expected) * 1e-9)
        if (check.get("pass") is not True or check.get("expected") != expected or check.get("actual") != actual
                or check.get("tolerance") != tolerance or check.get("absolute_difference") != abs(actual - expected)
                or abs(actual - expected) > tolerance or check.get("source") != row.get("source_evaluation")
                or check.get("source_pointer") != row.get("source_pointer")):
            raise RuntimeError("Old-score replay does not reproduce its pinned original score")
    expected_pairs = len(DATASETS) * len(PRIMARY_AND_SECONDARY_RIGHT) * len(PERIODS) * 2
    actual_counts = {"accuracy_models": len(model_rows), "accuracy_channels": len(channel_rows),
                     "paired_comparisons": len(pairs), "old_score_replay": len(checks)}
    if evaluation.get("row_counts") != actual_counts or len(pairs) != expected_pairs:
        raise RuntimeError("Evaluation row counts differ from actual CSV/replay artifacts")
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
