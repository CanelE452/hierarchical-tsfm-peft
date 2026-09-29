from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from runtime_v8 import (
    CACHE,
    HERE,
    ROOT,
    Job,
    digest,
    environment_receipt,
    load_data,
    save_json,
    source_receipt,
)

CONTEXT = 512
HORIZON = 48
PERIODS = ("test_a", "test_b")
SEEDS = (92601, 92602)
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 9262026
V7 = ROOT / "research/tsfm_peft_practical_controls_v7_20260928"
V7_EVALUATIONS = {
    "robin": V7 / "robin_eval01.json",
    "jena": V7 / "jena_eval01.json",
}
SPACE_ROLES = {"pq_split", "raw64", "level_res", "level_raw", "lora", "direct_nlinear", "f0"}


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    return resolve(path).resolve().relative_to(ROOT.resolve()).as_posix()


def receipt(path: str | Path) -> dict[str, str]:
    resolved = resolve(path)
    return {"path": str(resolved), "sha256": digest(resolved)}


def check_receipt(item: dict[str, Any]) -> Path:
    path = resolve(item["path"])
    actual = digest(path)
    if actual != item["sha256"]:
        raise ValueError(f"Sealed input changed: {path} expected {item['sha256']} got {actual}")
    return path


def load_module(path: Path, name: str):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def validate_name(name: str) -> None:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    if not name or any(char not in allowed for char in name):
        raise ValueError("Use a unique alphanumeric/underscore/hyphen name")


def configure_numeric_runtime() -> None:
    try:
        import torch
    except Exception:
        return
    torch.set_num_threads(4)
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False


def protocol() -> dict[str, Any]:
    return read_json(HERE / "protocol.json")


def dataset_config(dataset: str) -> dict[str, Any]:
    cfg = protocol()["datasets"][dataset]
    if int(cfg["channels"]) not in (17, 21):
        raise ValueError("Only approved v8 datasets are allowed")
    return cfg


def normal_role(value: str) -> str:
    aliases = {
        "pq": "pq_split",
        "pqsplit": "pq_split",
        "raw_64": "raw64",
        "raw": "level_raw",
        "raw32": "level_raw",
        "level_tsfm_raw": "level_raw",
        "res": "old_res",
        "tsfm_res": "old_res",
        "level_tsfm_res": "level_res",
        "tsfm_level_res": "level_res",
        "linear_res": "level_linear_res",
        "direct": "direct_nlinear",
        "nlinear": "direct_nlinear",
        "direct_linear": "direct_nlinear",
    }
    return aliases.get(str(value), str(value))


def validate_selection_test_policy(selection: dict[str, Any]) -> dict[str, Any]:
    if selection.get("test_used_for_selection") is False:
        return {"status": "explicit_false", "test_used_for_selection": False}
    if "test_used_for_selection" in selection:
        raise ValueError("Selection metadata says TEST may have been used for selection")
    blocked: list[str] = []

    def walk(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                key_text = str(key).lower()
                if key_text.startswith("test") or key_text in {"test_score", "test_mse", "test_mae", "test_metric", "test_prediction"}:
                    blocked.append(f"{path}.{key}" if path else str(key))
                walk(value, f"{path}.{key}" if path else str(key))
        elif isinstance(obj, list):
            for idx, value in enumerate(obj):
                walk(value, f"{path}[{idx}]")

    walk({key: value for key, value in selection.items() if key != "test_used_for_selection"}, "")
    if blocked:
        raise ValueError(f"Selection omits test_used_for_selection=false and contains TEST-like keys: {blocked[:10]}")
    if "selected" not in selection or "all_neural_runs" not in selection:
        raise ValueError("V8 selection must be select-only selected/all_neural_runs schema")
    return {
        "status": "missing_but_select_only_schema_without_test_keys",
        "test_used_for_selection": False,
        "note": "Accepted because the selection file has the approved select-only schema and no TEST-like keys.",
    }


def _rows_from_selection(selection: dict[str, Any], dataset: str) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    run_map = {row["id"]: dict(row) for row in selection.get("all_neural_runs", [])}
    rows: list[dict[str, Any]] = []
    roles: dict[str, list[str]] = {}
    for family, item in selection.get("selected", {}).items():
        role = normal_role(family)
        if role not in {"pq_split", "raw64"}:
            raise ValueError(f"V8 selection contains an unapproved new family: {family}")
        ids = list(item.get("run_ids", []))
        if len(ids) != len(SEEDS):
            raise ValueError(f"V8 selected family {family} must contain exactly two seed runs")
        roles[role] = ids
        for identifier in ids:
            if identifier not in run_map:
                raise ValueError(f"selected.{family} references missing all_neural_runs id {identifier}")
            row = dict(run_map[identifier])
            row.setdefault("role", role)
            row.setdefault("kind", "checkpoint")
            spec = dict(row.get("spec") or {})
            spec.setdefault("family", role)
            spec.setdefault("dataset", dataset)
            row["spec"] = spec
            rows.append(row)
    return rows, roles


def load_selection(path: Path, dataset: str) -> dict[str, Any]:
    selection = read_json(path)
    if selection.get("dataset", dataset) != dataset:
        raise ValueError(f"Selection dataset mismatch: {selection.get('dataset')} vs {dataset}")
    selection["_selection_test_policy"] = validate_selection_test_policy(selection)
    rows, roles = _rows_from_selection(selection, dataset)
    selection["models"] = rows
    selection["roles"] = roles
    return selection


def model_entry(row: dict[str, Any], dataset: str) -> dict[str, Any]:
    item = dict(row)
    spec = dict(item.get("spec") or {})
    spec["family"] = normal_role(str(spec.get("family") or item.get("family") or item.get("role")))
    spec.setdefault("dataset", dataset)
    if spec["dataset"] != dataset:
        raise ValueError(f"Model {item.get('id')} belongs to {spec['dataset']} not {dataset}")
    item["spec"] = spec
    item["id"] = item.get("id") or spec.get("id")
    if not item["id"]:
        raise ValueError("Every selected model needs an id")
    item["role"] = normal_role(str(item.get("role") or spec["family"]))
    item["kind"] = item.get("kind") or ("stored_prediction" if "prediction" in item else "checkpoint")
    if item["kind"] == "checkpoint":
        if "checkpoint" not in item or "checkpoint_sha256" not in item:
            raise ValueError(f"Checkpoint model missing receipt: {item['id']}")
        check_receipt({"path": item["checkpoint"], "sha256": item["checkpoint_sha256"]})
        if "result_path" in item and "result_sha256" in item:
            check_receipt({"path": item["result_path"], "sha256": item["result_sha256"]})
        if "best_val_prediction" in item:
            check_receipt(item["best_val_prediction"])
        if "best_val_mse" in item and not math.isfinite(float(item["best_val_mse"])):
            raise ValueError(f"Nonfinite VAL loss: {item['id']}")
    elif item["kind"] == "stored_prediction":
        check_receipt(item["prediction"])
    else:
        raise ValueError(f"Unknown model kind {item['kind']} for {item['id']}")
    return item


def collect_selection_models(selection: dict[str, Any], dataset: str) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    models = [model_entry(row, dataset) for row in selection["models"]]
    if len({row["id"] for row in models}) != len(models):
        raise ValueError("Model ids must be unique")
    roles: dict[str, list[str]] = {}
    for row in models:
        roles.setdefault(row["role"], []).append(row["id"])
    for role in ("pq_split", "raw64"):
        if role not in roles or len(roles[role]) != len(SEEDS):
            raise ValueError(f"V8 seal requires exactly two selected {role} seed checkpoints")
    return models, roles


def collect_v7_predictions(dataset: str) -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, str]]:
    source_path = V7_EVALUATIONS[dataset]
    evaluation = read_json(source_path)
    if evaluation.get("status") != "complete" or evaluation.get("dataset") != dataset:
        raise ValueError(f"{dataset} v7 evaluation must remain complete for prediction reuse")
    source = {"path": rel(source_path), "sha256": digest(source_path)}
    models: list[dict[str, Any]] = []
    for identifier, row in evaluation["models"].items():
        pred = row["prediction"]
        check_receipt(pred)
        spec = dict(row.get("spec") or {})
        spec["family"] = normal_role(str(row.get("role") or spec.get("family") or identifier))
        spec["dataset"] = dataset
        models.append(
            {
                "id": identifier,
                "role": normal_role(str(row.get("role") or spec["family"])),
                "kind": "stored_prediction",
                "spec": spec,
                "prediction": pred,
                "source_evaluation": source,
                "original_v7_kind": row.get("kind"),
                "scores_from_v7_not_reused_for_v8_scoring": True,
            }
        )
    roles = {normal_role(key): list(ids) for key, ids in evaluation["roles"].items()}
    return models, roles, source


def default_comparisons(roles: dict[str, list[str]], dataset: str) -> list[dict[str, Any]]:
    pairs = [
        ("pq_split_vs_raw64", "pq_split", "raw64", "independent P/Q correction versus parameter-matched shared RAW64"),
        ("pq_split_vs_level_res", "pq_split", "level_res", "P/Q correction beyond v7 LEVEL residual PEFT"),
        ("pq_split_vs_level_raw", "pq_split", "level_raw", "P/Q correction versus v7 matched RAW32"),
        ("pq_split_vs_direct_nlinear", "pq_split", "direct_nlinear", "P/Q correction versus direct NLinear baseline"),
        ("pq_split_vs_f0", "pq_split", "f0", "remaining accuracy gap to uncompressed frozen TSFM"),
        ("pq_split_vs_lora", "pq_split", "lora", "remaining accuracy gap to uncompressed LoRA"),
        ("pq_split_vs_old_res", "pq_split", "old_res", "P/Q correction beyond pre-LEVEL residual PEFT"),
        ("raw64_vs_level_raw", "raw64", "level_raw", "parameter increase in RAW correction over v7 RAW32"),
        ("raw64_vs_level_res", "raw64", "level_res", "shared RAW64 versus v7 LEVEL residual PEFT"),
        ("raw64_vs_direct_nlinear", "raw64", "direct_nlinear", "shared RAW64 versus direct NLinear baseline"),
        ("raw64_vs_f0", "raw64", "f0", "RAW64 remaining accuracy gap to uncompressed frozen TSFM"),
        ("raw64_vs_lora", "raw64", "lora", "RAW64 remaining accuracy gap to uncompressed LoRA"),
    ]
    output = []
    for name, left, right, question in pairs:
        if left in roles and right in roles:
            output.append(
                {
                    "name": f"{dataset}_{name}",
                    "left": roles[left],
                    "right": roles[right],
                    "primary": name == "pq_split_vs_raw64",
                    "question": question,
                }
            )
    return output


def require_both_dataset_selections() -> dict[str, dict[str, str]]:
    receipts: dict[str, dict[str, str]] = {}
    for dataset in ("robin", "jena"):
        path = HERE / f"selected_{dataset}.json"
        if not path.exists():
            raise FileNotFoundError(f"Both selected_robin.json and selected_jena.json are required before sealing TEST evaluation; missing {path.name}")
        receipts[dataset] = receipt(path)
    return receipts


def seal_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    if args.selection is None:
        raise ValueError("--selection is required for --seal")
    dataset = args.dataset
    cfg = dataset_config(dataset)
    destination = HERE / f"evaluation_seal_{dataset}.json"
    exposure = HERE / f"test_exposure_{dataset}.json"
    if destination.exists() or exposure.exists():
        raise FileExistsError("Never overwrite a dataset selection seal or retroactive exposure record")
    selection_receipts = require_both_dataset_selections()
    data_contract = HERE / f"data_contract_{dataset}.json"
    check_receipt({"path": data_contract, "sha256": digest(data_contract)})
    selection = load_selection(args.selection, dataset)
    models, roles = collect_selection_models(selection, dataset)
    v7_models, v7_roles, v7_source = collect_v7_predictions(dataset)
    known_ids = {row["id"] for row in models}
    models = [row for row in v7_models if row["id"] not in known_ids] + models
    for role, ids in v7_roles.items():
        roles.setdefault(role, [])
        roles[role] = [*ids, *[identifier for identifier in roles[role] if identifier not in ids]]
    required_reuse = {"level_res", "level_raw", "direct_nlinear", "lora", "f0"}
    missing = sorted(required_reuse - set(roles))
    if missing:
        raise ValueError(f"{dataset} v8 seal missing required reused roles: {missing}")
    channels = int(cfg["channels"])
    sealed = {
        "schema": "tsfm_peft_subspace_correction_v8_evaluation_seal_v1",
        "status": "sealed",
        "dataset": dataset,
        "sealed_utc": time.time(),
        "selection": receipt(args.selection),
        "all_selection_files_required_before_test": selection_receipts,
        "selection_test_policy": selection.get("_selection_test_policy"),
        "data_contract": receipt(data_contract),
        "protocol": receipt(HERE / "protocol.json"),
        "plan": receipt(HERE / "PLAN.md"),
        "approval": receipt(HERE / "APPROVAL.txt"),
        "v7_reuse": v7_source,
        "channels": channels,
        "splits": cfg["splits"],
        "expected_origins": cfg["expected_origins"],
        "evaluation_stride": int(cfg["evaluation_stride"]),
        "interval_minutes": int(cfg["interval_minutes"]),
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "block_length_origins": int(cfg["bootstrap_block_origins"]),
        },
        "metric": "TRAIN-standardized raw-space observed-channel macro MSE/MAE; seed losses averaged, not ensembled.",
        "space_metric": "P/Q complete-target-row decomposition using TRAIN PCA basis; no target zero filling before projection.",
        "models": models,
        "roles": roles,
        "comparisons": default_comparisons(roles, dataset),
        "test_used_for_selection": False,
        "test_selection_prohibited": True,
    }
    save_json(destination, sealed)
    return {"seal": rel(destination), "sha256": digest(destination), "models": len(models), "roles": sorted(roles)}


def load_seal(dataset: str) -> dict[str, Any]:
    path = HERE / f"evaluation_seal_{dataset}.json"
    sealed = read_json(path)
    if sealed.get("status") != "sealed" or sealed.get("dataset") != dataset:
        raise ValueError("Evaluation requires a completed dataset-specific seal")
    for key in ("selection", "data_contract", "protocol", "plan", "approval", "v7_reuse"):
        check_receipt(sealed[key])
    for dataset_name, selection in sealed.get("all_selection_files_required_before_test", {}).items():
        check_receipt(selection)
        if dataset_name not in ("robin", "jena"):
            raise ValueError("Unexpected selection guard dataset")
    for row in sealed["models"]:
        if row["kind"] == "checkpoint":
            check_receipt({"path": row["checkpoint"], "sha256": row["checkpoint_sha256"]})
        elif row["kind"] == "stored_prediction":
            check_receipt(row["prediction"])
        else:
            raise ValueError(f"Unknown model kind {row['kind']}")
    return sealed


def split_pair(sealed: dict[str, Any], period: str) -> tuple[np.datetime64, np.datetime64]:
    pair = sealed["splits"][period]
    if isinstance(pair, dict):
        start, end = pair["start"], pair["end"]
    else:
        start, end = pair
    return np.datetime64(start, "ns"), np.datetime64(end, "ns")


def validate_origins(sealed: dict[str, Any], data: dict[str, Any]) -> dict[str, np.ndarray]:
    dataset = sealed["dataset"]
    cfg = dataset_config(dataset)
    times = np.asarray(data["times"]).astype("datetime64[ns]")
    origins_by_period = {period: np.asarray(data[f"{period}_origins"], dtype=np.int64) for period in PERIODS}
    for period, origins in origins_by_period.items():
        expected = int(cfg["expected_origins"][period])
        stride = int(cfg["evaluation_stride"])
        if len(origins) != expected or len(np.unique(origins)) != expected:
            raise ValueError(f"{dataset}/{period}: origin count mismatch")
        if len(origins) > 1 and not np.all(np.diff(origins) == stride):
            raise ValueError(f"{dataset}/{period}: origin stride mismatch")
        if np.any(origins < CONTEXT) or np.any(origins + HORIZON > len(times)):
            raise ValueError(f"{dataset}/{period}: context/target outside archive")
        start, end = split_pair(sealed, period)
        if np.any(times[origins] < start) or np.any(times[origins + HORIZON - 1] >= end):
            raise ValueError(f"{dataset}/{period}: target crosses split bounds")
    return origins_by_period


def load_prediction_receipt(row: dict[str, Any], origins: np.ndarray) -> tuple[np.ndarray, dict[str, str]]:
    pred_path = check_receipt(row["prediction"])
    with np.load(pred_path, allow_pickle=False) as archive:
        if not np.array_equal(archive["origins"], origins):
            raise ValueError(f"Stored prediction origins differ for {row['id']}")
        return archive["prediction"].astype(np.float32), row["prediction"]


def checkpoint_predict(row: dict[str, Any], inputs_all: np.ndarray, batch_size: int, job: Job) -> np.ndarray:
    import torch

    module = load_module(HERE / "model_v8.py", "_v8_evaluation_model")
    model = module.restore_model(resolve(row["checkpoint"]), device="cuda").eval()
    outputs = []
    with torch.inference_mode():
        for offset in range(0, len(inputs_all), batch_size):
            job.check_limits()
            inputs = torch.as_tensor(inputs_all[offset : offset + batch_size], dtype=torch.float32, device="cuda")
            outputs.append(model(inputs).float().cpu().numpy())
            del inputs
    prediction = np.concatenate(outputs, axis=0).astype(np.float32)
    del outputs, model
    gc.collect()
    torch.cuda.synchronize()
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    if torch.cuda.memory_allocated():
        raise RuntimeError("Evaluation retained a prior model or output on CUDA")
    return prediction


def error_terms(prediction: np.ndarray, target: np.ndarray, observed: np.ndarray) -> dict[str, np.ndarray]:
    if prediction.shape != target.shape or observed.shape != target.shape:
        raise ValueError(f"Prediction/target/mask shapes differ: {prediction.shape}, {target.shape}, {observed.shape}")
    if not np.isfinite(prediction).all() or not np.isfinite(target[observed]).all():
        raise ValueError("Predictions and observed targets must be finite")
    delta = np.where(observed, prediction.astype(np.float64) - target.astype(np.float64), 0.0)
    return {
        "squared": (delta * delta).sum(axis=1),
        "absolute": np.abs(delta).sum(axis=1),
        "signed": delta.sum(axis=1),
        "count": observed.sum(axis=1),
    }


def scores_from_terms(terms: dict[str, np.ndarray], index: np.ndarray | None = None) -> dict[str, Any]:
    if index is None:
        index = np.arange(len(terms["count"]), dtype=np.int64)
    counts = terms["count"][index].sum(axis=0)
    squared = terms["squared"][index].sum(axis=0)
    absolute = terms["absolute"][index].sum(axis=0)
    signed = terms["signed"][index].sum(axis=0)
    valid = counts > 0
    channel_mse = [float(squared[c] / counts[c]) if valid[c] else None for c in range(len(counts))]
    channel_mae = [float(absolute[c] / counts[c]) if valid[c] else None for c in range(len(counts))]
    channel_bias = [float(signed[c] / counts[c]) if valid[c] else None for c in range(len(counts))]
    return {
        "status": "available" if valid.all() else "unavailable_missing_channel_targets",
        "mse": float(np.mean(channel_mse)) if valid.all() else None,
        "mae": float(np.mean(channel_mae)) if valid.all() else None,
        "channel_mse": channel_mse,
        "channel_mae": channel_mae,
        "channel_mean_signed_error": channel_bias,
        "channel_target_count": counts.astype(int).tolist(),
        "channel_squared_error_sum": squared.tolist(),
        "channel_absolute_error_sum": absolute.tolist(),
        "channel_signed_error_sum": signed.tolist(),
        "missing_channel_indices": np.flatnonzero(~valid).tolist(),
    }


def space_terms(prediction: np.ndarray, target: np.ndarray, observed: np.ndarray, basis: np.ndarray) -> dict[str, np.ndarray]:
    channels, latent = basis.shape
    if prediction.shape != target.shape or observed.shape != target.shape:
        raise ValueError("Prediction/target/mask axes mismatch for P/Q decomposition")
    if prediction.shape[2] != channels:
        raise ValueError("PCA basis channel count differs from prediction shape")
    complete = observed.all(axis=2)
    rows = np.argwhere(complete)
    projector = basis.astype(np.float64) @ basis.astype(np.float64).T
    q_projector = np.eye(channels, dtype=np.float64) - projector
    p_squared = np.zeros((prediction.shape[0], channels), dtype=np.float64)
    q_squared = np.zeros_like(p_squared)
    p_signed = np.zeros_like(p_squared)
    q_signed = np.zeros_like(p_squared)
    p_absolute = np.zeros_like(p_squared)
    q_absolute = np.zeros_like(p_squared)
    if len(rows):
        values = prediction[complete].astype(np.float64) - target[complete].astype(np.float64)
        retained = values @ projector
        residual = values @ q_projector
        origin_rows = rows[:, 0]
        np.add.at(p_squared, origin_rows, retained * retained)
        np.add.at(q_squared, origin_rows, residual * residual)
        np.add.at(p_signed, origin_rows, retained)
        np.add.at(q_signed, origin_rows, residual)
        np.add.at(p_absolute, origin_rows, np.abs(retained))
        np.add.at(q_absolute, origin_rows, np.abs(residual))
    counts = np.repeat(complete.sum(axis=1)[:, None], channels, axis=1)
    return {
        "p_squared": p_squared,
        "q_squared": q_squared,
        "p_absolute": p_absolute,
        "q_absolute": q_absolute,
        "p_signed": p_signed,
        "q_signed": q_signed,
        "count": counts,
        "complete_target_rows": complete.sum(axis=1),
        "basis_rank": np.asarray([latent], dtype=np.int64),
    }


def scores_from_space_terms(terms: dict[str, np.ndarray], index: np.ndarray | None = None) -> dict[str, Any]:
    if index is None:
        index = np.arange(len(terms["count"]), dtype=np.int64)
    counts = terms["count"][index].sum(axis=0)
    valid = counts > 0

    def score(prefix: str) -> tuple[list[float | None], list[float | None], list[float | None], float | None, float | None]:
        squared = terms[f"{prefix}_squared"][index].sum(axis=0)
        absolute = terms[f"{prefix}_absolute"][index].sum(axis=0)
        signed = terms[f"{prefix}_signed"][index].sum(axis=0)
        channel_mse = [float(squared[c] / counts[c]) if valid[c] else None for c in range(len(counts))]
        channel_mae = [float(absolute[c] / counts[c]) if valid[c] else None for c in range(len(counts))]
        channel_bias = [float(signed[c] / counts[c]) if valid[c] else None for c in range(len(counts))]
        return channel_mse, channel_mae, channel_bias, (float(np.mean(channel_mse)) if valid.all() else None), (float(np.mean(channel_mae)) if valid.all() else None)

    p_mse, p_mae, p_bias, p_macro_mse, p_macro_mae = score("p")
    q_mse, q_mae, q_bias, q_macro_mse, q_macro_mae = score("q")
    return {
        "status": "available" if valid.all() else "unavailable_missing_complete_target_rows",
        "scope": "Only origin-horizon rows with complete observed targets across all channels are projected.",
        "complete_target_rows": int(terms["complete_target_rows"][index].sum()),
        "channel_complete_row_count": counts.astype(int).tolist(),
        "p_retained_space_mse": p_macro_mse,
        "p_retained_space_mae": p_macro_mae,
        "q_residual_space_mse": q_macro_mse,
        "q_residual_space_mae": q_macro_mae,
        "channel_p_mse": p_mse,
        "channel_q_mse": q_mse,
        "channel_p_mae": p_mae,
        "channel_q_mae": q_mae,
        "channel_p_mean_signed_error": p_bias,
        "channel_q_mean_signed_error": q_bias,
    }


def pair_ids(ids: list[str], models: dict[str, dict[str, Any]]) -> list[str]:
    if len(ids) == 1:
        return ids * len(SEEDS)
    by_seed = {models[identifier]["spec"].get("seed"): identifier for identifier in ids}
    if set(by_seed) != set(SEEDS):
        raise ValueError(f"Expected paired seeds {SEEDS} or deterministic singleton; got {ids}")
    return [by_seed[seed] for seed in SEEDS]


def compare_terms(
    comparison: dict[str, Any],
    terms: dict[str, dict[str, np.ndarray]],
    models: dict[str, dict[str, Any]],
    period_indices: dict[str, np.ndarray],
    sealed: dict[str, Any],
    job: Job,
) -> dict[str, Any]:
    left_ids = pair_ids(list(comparison["left"]), models)
    right_ids = pair_ids(list(comparison["right"]), models)
    count = terms[left_ids[0]]["count"]
    for identifier in left_ids + right_ids:
        if not np.array_equal(terms[identifier]["count"], count):
            raise ValueError("All paired comparisons must use the same observed target counts")
    left = np.mean([terms[identifier]["squared"] for identifier in left_ids], axis=0)
    right = np.mean([terms[identifier]["squared"] for identifier in right_ids], axis=0)
    block = int(sealed["bootstrap"]["block_length_origins"])
    draws = int(sealed["bootstrap"]["draws"])
    rng_seed = int(sealed["bootstrap"]["seed"])

    def values(index: np.ndarray) -> dict[str, float] | None:
        denominator = count[index].sum(axis=0)
        if np.any(denominator == 0):
            return None
        left_mse = float(np.mean(left[index].sum(axis=0) / denominator))
        reference_mse = float(np.mean(right[index].sum(axis=0) / denominator))
        return {
            "left_mse": left_mse,
            "reference_mse": reference_mse,
            "delta_mse": left_mse - reference_mse,
            "relative_to_reference_pct": 100 * (left_mse - reference_mse) / reference_mse if reference_mse > 0 else None,
        }

    output = dict(comparison, direction="left minus reference; negative favors left", left_seed_ids=left_ids, right_seed_ids=right_ids, periods={})
    for label in (*PERIODS, "combined"):
        strata = [period_indices[label]] if label != "combined" else [period_indices[period] for period in PERIODS]
        full_index = np.concatenate(strata)
        full = values(full_index)
        seed_values = []
        for seed, left_id, right_id in zip(SEEDS, left_ids, right_ids):
            left_score = scores_from_terms(terms[left_id], full_index)["mse"]
            right_score = scores_from_terms(terms[right_id], full_index)["mse"]
            seed_values.append(
                {
                    "seed": seed,
                    "left_id": left_id,
                    "right_id": right_id,
                    "left_mse": left_score,
                    "reference_mse": right_score,
                    "delta_mse": left_score - right_score if left_score is not None and right_score is not None else None,
                    "relative_to_reference_pct": 100 * (left_score - right_score) / right_score if left_score is not None and right_score is not None and right_score > 0 else None,
                }
            )
        salt = int.from_bytes(hashlib.sha256(f"{comparison['name']}::{label}".encode("utf-8")).digest()[:4], "little")
        rng = np.random.default_rng(np.random.SeedSequence([rng_seed, salt]))
        samples = []
        unavailable = 0
        for draw in range(draws):
            if draw % 250 == 0:
                job.check_limits()
            selected = []
            for stratum in strata:
                n = len(stratum)
                if n < block:
                    unavailable += 1
                    continue
                starts = rng.integers(0, n - block + 1, int(np.ceil(n / block)))
                local = np.concatenate([np.arange(start, start + block) for start in starts])[:n]
                selected.append(stratum[local])
            if len(selected) != len(strata):
                continue
            value = values(np.concatenate(selected))
            if value is None or value["relative_to_reference_pct"] is None:
                unavailable += 1
            else:
                samples.append(value)
        interval = None
        if full is not None and samples and unavailable == 0:
            interval = {
                key: np.quantile([row[key] for row in samples], [0.025, 0.975]).tolist()
                for key in ("delta_mse", "relative_to_reference_pct")
            }
        output["periods"][label] = {
            "full": full,
            "paired_seed_deltas": seed_values,
            "conditional_block95": interval,
            "draws": draws,
            "unavailable_draws": unavailable,
            "block_length_origins": block,
            "strata": [label] if label != "combined" else list(PERIODS),
            "interval_status": "available" if interval is not None else "unavailable_without_dropping_channels",
        }
    output["interval_scope"] = "Non-circular fixed-length origin blocks within each period, common resamples across channels/models; trained seeds fixed. Does not include training or selection uncertainty."
    return output


def evaluate_sealed(args: argparse.Namespace, job: Job) -> dict[str, Any]:
    dataset = args.dataset
    sealed = load_seal(dataset)
    exposure_path = HERE / f"test_exposure_{dataset}.json"
    seal_receipt = receipt(HERE / f"evaluation_seal_{dataset}.json")
    result_path = HERE / f"{args.name}.json"
    partial_path = HERE / f"{args.name}_partial.json"
    prediction_dir = CACHE / f"{args.name}_predictions"
    if exposure_path.exists():
        raise FileExistsError("This script does not overwrite exposed v8 TEST results; use a documented correction script if needed")
    if result_path.exists() or partial_path.exists() or prediction_dir.exists():
        raise FileExistsError("Preserve existing evaluation attempt; use a new name")
    save_json(
        exposure_path,
        {
            "schema": "tsfm_peft_subspace_correction_v8_test_exposure_v1",
            "dataset": dataset,
            "status": "exposed",
            "first_exposed_utc": time.time(),
            "seal_sha256": seal_receipt["sha256"],
            "seal": seal_receipt,
            "evaluation_name": args.name,
            "command": sys.argv,
            "scope": "Written after output path collision checks and before the first protected TEST archive is opened for this dataset and seal.",
        },
    )
    data = load_data(dataset, include_test=True)
    origins_by_period = validate_origins(sealed, data)
    origins = np.concatenate([origins_by_period[period] for period in PERIODS])
    period_indices = {
        "test_a": np.arange(len(origins_by_period["test_a"]), dtype=np.int64),
        "test_b": np.arange(len(origins_by_period["test_a"]), len(origins), dtype=np.int64),
    }
    inputs = np.stack([data["x"][origin - CONTEXT : origin] for origin in origins]).astype(np.float32)
    target = np.stack([data["x"][origin : origin + HORIZON] for origin in origins]).astype(np.float32)
    observed = np.stack([data["finite"][origin : origin + HORIZON] for origin in origins]).astype(bool)
    basis = np.asarray(data["basis"], dtype=np.float64)
    if inputs.shape[1:] != (CONTEXT, int(sealed["channels"])) or target.shape[1:] != (HORIZON, int(sealed["channels"])):
        raise ValueError("Sealed TEST tensor shape differs from approved contract")
    prediction_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "tsfm_peft_subspace_correction_v8_test_evaluation_v1",
        "status": "running",
        "name": args.name,
        "dataset": dataset,
        "seal": seal_receipt,
        "first_exposure": receipt(exposure_path),
        "source": source_receipt(),
        "environment": environment_receipt(),
        "channels": np.asarray(data["columns"]).astype(str).tolist(),
        "origins": {period: origins_by_period[period].astype(int).tolist() for period in PERIODS},
        "roles": sealed["roles"],
        "metric": sealed["metric"],
        "space_metric": sealed["space_metric"],
        "models": {},
        "role_scores": {},
        "role_space_scores": {},
        "comparisons": [],
    }
    save_json(partial_path, result)
    models = {row["id"]: row for row in sealed["models"]}
    terms: dict[str, dict[str, np.ndarray]] = {}
    spaces: dict[str, dict[str, np.ndarray]] = {}
    for identifier, row in models.items():
        job.check_limits()
        reused = False
        if row["kind"] == "stored_prediction":
            prediction, prediction_receipt = load_prediction_receipt(row, origins)
            reused = True
        elif row["kind"] == "checkpoint":
            prediction = checkpoint_predict(row, inputs, int(protocol()["eval_batch_origins"]), job)
            prediction_path = prediction_dir / f"{identifier}.npz"
            np.savez_compressed(prediction_path, prediction=prediction, origins=origins)
            prediction_receipt = receipt(prediction_path)
        else:
            raise ValueError(f"Unknown model kind {row['kind']} for {identifier}")
        terms[identifier] = error_terms(prediction, target, observed)
        scores = {period: scores_from_terms(terms[identifier], period_indices[period]) for period in PERIODS}
        scores["combined"] = scores_from_terms(terms[identifier])
        model_output = {
            "spec": row["spec"],
            "kind": row["kind"],
            "role": row["role"],
            "prediction": prediction_receipt,
            "reused_v7_prediction": reused,
            "scores": scores,
        }
        if row["role"] in SPACE_ROLES:
            spaces[identifier] = space_terms(prediction, target, observed, basis)
            space_scores = {period: scores_from_space_terms(spaces[identifier], period_indices[period]) for period in PERIODS}
            space_scores["combined"] = scores_from_space_terms(spaces[identifier])
            model_output["space_scores"] = space_scores
        result["models"][identifier] = model_output
        del prediction
        save_json(partial_path, result)
        job.heartbeat(f"{dataset}: evaluated {identifier}; {len(result['models'])}/{len(models)}")
    for role, ids in sealed["roles"].items():
        known = [identifier for identifier in ids if identifier in result["models"]]
        if not known:
            continue
        result["role_scores"][role] = {}
        for period in (*PERIODS, "combined"):
            scores = [result["models"][identifier]["scores"][period] for identifier in known]
            available = all(score["mse"] is not None for score in scores)
            result["role_scores"][role][period] = {
                "run_ids": known,
                "count": len(known),
                "mse": float(np.mean([score["mse"] for score in scores])) if available else None,
                "mae": float(np.mean([score["mae"] for score in scores])) if available else None,
                "scope": "Mean of individual model channel-macro losses; not an ensemble.",
            }
        if all(identifier in spaces for identifier in known):
            result["role_space_scores"][role] = {}
            for period in (*PERIODS, "combined"):
                scores = [result["models"][identifier]["space_scores"][period] for identifier in known]
                available = all(score["p_retained_space_mse"] is not None and score["q_residual_space_mse"] is not None for score in scores)
                result["role_space_scores"][role][period] = {
                    "run_ids": known,
                    "count": len(known),
                    "p_retained_space_mse": float(np.mean([score["p_retained_space_mse"] for score in scores])) if available else None,
                    "q_residual_space_mse": float(np.mean([score["q_residual_space_mse"] for score in scores])) if available else None,
                    "p_retained_space_mae": float(np.mean([score["p_retained_space_mae"] for score in scores])) if available else None,
                    "q_residual_space_mae": float(np.mean([score["q_residual_space_mae"] for score in scores])) if available else None,
                    "complete_target_rows": int(np.mean([score["complete_target_rows"] for score in scores])),
                    "scope": "Mean of individual model complete-row P/Q losses; not an ensemble.",
                }
    for comparison in sealed["comparisons"]:
        if all(identifier in terms for identifier in comparison["left"] + comparison["right"]):
            result["comparisons"].append(compare_terms(comparison, terms, models, period_indices, sealed, job))
            job.heartbeat(f"{dataset}: compared {comparison['name']}")
    result.update(
        status="complete",
        completed_utc=time.time(),
        combined_definition="Within each seed/channel, sum TEST-A/B errors and observed counts before channel mean; then average individual seed losses. Deterministic references broadcast for seed-paired deltas.",
        test_reselection=False,
        new_test_model_predictions=sum(not item["reused_v7_prediction"] for item in result["models"].values()),
    )
    save_json(result_path, result)
    return {"result": rel(result_path), "models": len(result["models"]), "comparisons": len(result["comparisons"])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Seal and evaluate approved v8 P/Q correction comparisons.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--seal", action="store_true")
    action.add_argument("--evaluate", action="store_true")
    parser.add_argument("--dataset", choices=("robin", "jena"), required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--reserve-seconds", type=float, required=True)
    args = parser.parse_args()
    validate_name(args.name)
    configure_numeric_runtime()
    attempt_suffix = "seal_attempt" if args.seal else "evaluate_attempt"
    attempt_path = HERE / f"{args.name}_{attempt_suffix}.json"
    if attempt_path.exists():
        raise FileExistsError("Preserve existing attempt file; use a new name")
    attempt = {
        "status": "running",
        "started_utc": time.time(),
        "command": sys.argv,
        "dataset": args.dataset,
        "source": source_receipt(),
        "environment": environment_receipt(),
    }
    save_json(attempt_path, attempt)
    try:
        if args.seal:
            with Job(f"v8_seal_{args.dataset}_{args.name}", category="cpu_analysis", reserve_seconds=args.reserve_seconds, metadata={"dataset": args.dataset, "name": args.name, "action": "seal"}):
                result = seal_evaluation(args)
        else:
            with Job(f"v8_evaluate_{args.dataset}_{args.name}", category="gpu", reserve_seconds=args.reserve_seconds, metadata={"dataset": args.dataset, "name": args.name, "action": "evaluate"}) as job:
                result = evaluate_sealed(args, job)
        attempt.update(status="complete", completed_utc=time.time(), result=result)
        save_json(attempt_path, attempt)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as exc:
        attempt.update(status="failed", failed_utc=time.time(), error=repr(exc))
        save_json(attempt_path, attempt)
        raise


if __name__ == "__main__":
    main()
