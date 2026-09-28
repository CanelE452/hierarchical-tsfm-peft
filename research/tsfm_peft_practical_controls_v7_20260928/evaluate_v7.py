
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

from runtime_v7 import (
    CACHE,
    HERE,
    ROOT,
    V6,
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
V6_EVALUATION = V6 / "evaluation01.json"


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
    if not name or any(c not in allowed for c in name):
        raise ValueError("Use a unique alphanumeric/underscore/hyphen name")


def configure_numeric_runtime() -> None:
    # Keep the approved inference/evaluation numeric contract local to this process.
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
        raise ValueError("Only approved v7 datasets are allowed")
    return cfg


def normal_role(value: str) -> str:
    aliases = {
        "res": "old_res",
        "tsfm_res": "old_res",
        "level_tsfm_res": "level_res",
        "tsfm_level_res": "level_res",
        "level_tsfm_raw": "level_raw",
        "raw": "level_raw",
        "linear_res": "level_linear_res",
        "direct": "direct_nlinear",
        "nlinear": "direct_nlinear",
        "direct_linear": "direct_nlinear",
    }
    return aliases.get(value, value)


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
    compatible = "selected" in selection and "all_neural_runs" in selection
    if not compatible and not any(key in selection for key in ("models", "all_models", "all_selected", "selected_models")):
        raise ValueError("Selection omits test_used_for_selection=false and is not a recognized select-only schema")
    return {
        "status": "missing_but_select_only_schema_without_test_keys",
        "test_used_for_selection": False,
        "note": "Accepted only because the selection file has a recognized select-only schema and no TEST-like keys; add explicit test_used_for_selection=false in future receipts.",
    }


def _rows_from_v6_style_selection(selection: dict[str, Any], dataset: str) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    run_map = {row["id"]: dict(row) for row in selection.get("all_neural_runs", [])}
    rows: list[dict[str, Any]] = []
    roles: dict[str, list[str]] = {}
    for family, item in selection.get("selected", {}).items():
        role = normal_role(family)
        ids = list(item.get("run_ids", []))
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
    for family, row in selection.get("deterministic_models", {}).items():
        role = normal_role(family)
        item = dict(row)
        item.setdefault("role", role)
        item.setdefault("kind", "checkpoint")
        item.setdefault("id", item.get("id") or f"{dataset}_{role}")
        spec = dict(item.get("spec") or {})
        spec.setdefault("family", role)
        spec.setdefault("dataset", dataset)
        spec.setdefault("seed", None)
        item["spec"] = spec
        rows.append(item)
        roles.setdefault(role, []).append(item["id"])
    return rows, roles


def load_selection(path: Path, dataset: str) -> dict[str, Any]:
    selection = read_json(path)
    if selection.get("dataset", dataset) != dataset:
        raise ValueError(f"Selection dataset mismatch: {selection.get('dataset')} vs {dataset}")
    selection["_selection_test_policy"] = validate_selection_test_policy(selection)
    if "models" not in selection:
        for key in ("all_models", "all_selected", "selected_models"):
            if key in selection:
                selection["models"] = selection[key]
                break
    if "models" not in selection and "selected" in selection:
        rows, roles = _rows_from_v6_style_selection(selection, dataset)
        selection["models"] = rows
        merged_roles = {normal_role(key): list(ids) for key, ids in selection.get("roles", {}).items()}
        for key, ids in roles.items():
            merged_roles.setdefault(key, ids)
        selection["roles"] = merged_roles
    if "models" not in selection:
        raise ValueError("Selection file must contain models/all_models/all_selected/selected_models or selected/all_neural_runs")
    return selection


def model_entry(row: dict[str, Any], dataset: str) -> dict[str, Any]:
    item = dict(row)
    spec = dict(item.get("spec") or {})
    if "family" not in spec:
        spec["family"] = item.get("family") or item.get("role")
    spec["family"] = normal_role(str(spec["family"]))
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
    elif item["kind"] == "nlinear_weights":
        check_receipt(item["weights"])
        if "best_val_prediction" in item:
            check_receipt(item["best_val_prediction"])
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
    if "roles" in selection:
        normalized = {normal_role(key): list(ids) for key, ids in selection["roles"].items()}
        known = {row["id"] for row in models}
        for key, ids in normalized.items():
            if not set(ids).issubset(known):
                raise ValueError(f"Selection role {key} references unknown ids")
        roles.update(normalized)
    return models, roles


def collect_v6_robin_predictions() -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, Any]]:
    v6 = read_json(V6_EVALUATION)
    if v6.get("status") != "complete":
        raise ValueError("v6 evaluation must remain complete for Robin reuse")
    models: list[dict[str, Any]] = []
    for identifier, row in v6["models"].items():
        pred = row["prediction"]
        check_receipt(pred)
        spec = dict(row.get("spec") or {})
        spec["family"] = normal_role(str(spec.get("family") or identifier))
        spec["dataset"] = "robin"
        models.append({
            "id": identifier,
            "role": normal_role(spec["family"]),
            "kind": "stored_prediction",
            "spec": spec,
            "prediction": pred,
            "source_evaluation": {"path": rel(V6_EVALUATION), "sha256": digest(V6_EVALUATION)},
            "scores_from_v6_not_reused_for_v7_scoring": True,
        })
    roles = {normal_role(key): list(ids) for key, ids in v6["roles"].items()}
    return models, roles, {"path": rel(V6_EVALUATION), "sha256": digest(V6_EVALUATION)}


def default_comparisons(roles: dict[str, list[str]], dataset: str) -> list[dict[str, Any]]:
    primary_name = "level_res_vs_direct_nlinear" if dataset == "robin" else "level_res_vs_old_res"
    pairs = [
        ("level_res_vs_old_res", "level_res", "old_res", "LEVEL modification over previous residual PEFT"),
        ("level_res_vs_level_raw", "level_res", "level_raw", "residual input contribution under matched level restoration"),
        ("level_res_vs_level_only", "level_res", "level_only", "learned correction beyond level-only reference"),
        ("level_res_vs_compress", "level_res", "compress", "LEVEL correction beyond compressed frozen TSFM"),
        ("level_res_vs_f0", "level_res", "f0", "remaining accuracy gap to uncompressed frozen TSFM"),
        ("level_res_vs_lora", "level_res", "lora", "remaining accuracy gap to uncompressed LoRA"),
        ("level_res_vs_direct_nlinear", "level_res", "direct_nlinear", "compressed TSFM residual design versus direct NLinear"),
        ("direct_nlinear_vs_f0", "direct_nlinear", "f0", "direct NLinear versus uncompressed frozen TSFM"),
        ("direct_nlinear_vs_lora", "direct_nlinear", "lora", "direct NLinear versus uncompressed LoRA"),
    ]
    output = []
    for name, left, right, question in pairs:
        if left in roles and right in roles:
            output.append({"name": f"{dataset}_{name}", "left": roles[left], "right": roles[right], "primary": name == primary_name, "question": question})
    return output


def seal_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    dataset = args.dataset
    cfg = dataset_config(dataset)
    destination = HERE / f"evaluation_seal_{dataset}.json"
    exposure = HERE / f"test_exposure_{dataset}.json"
    if destination.exists() or exposure.exists():
        raise FileExistsError("Never overwrite a dataset selection seal or retroactive exposure record")
    data_contract = HERE / f"data_contract_{dataset}.json"
    check_receipt({"path": data_contract, "sha256": digest(data_contract)})
    selection = load_selection(args.selection, dataset)
    models, roles = collect_selection_models(selection, dataset)
    reused = None
    if dataset == "robin" and args.include_v6_robin:
        v6_models, v6_roles, reused = collect_v6_robin_predictions()
        known_ids = {row["id"] for row in models}
        add = [row for row in v6_models if row["id"] not in known_ids]
        models = add + models
        for role, ids in v6_roles.items():
            roles.setdefault(role, [])
            roles[role] = [*ids, *[i for i in roles[role] if i not in ids]]
    if dataset == "robin" and not args.include_v6_robin:
        raise ValueError("Robin v7 seal must include the approved reused v6 models; pass --include-v6-robin")
    if dataset == "robin" and "direct_nlinear" not in roles:
        raise ValueError("Robin v7 seal must include selected direct_nlinear models")
    if dataset == "robin":
        required_robin_roles = {"level_res", "old_res", "level_raw", "level_only", "compress", "f0", "lora", "direct_nlinear"}
        missing_robin = sorted(required_robin_roles - set(roles))
        if missing_robin:
            raise ValueError(f"Robin seal missing approved reused/direct roles: {missing_robin}")
    if dataset == "jena":
        required = {"level_res", "old_res", "level_raw", "lora", "direct_nlinear", "f0", "compress", "level_only"}
        missing = sorted(required - set(roles))
        if missing:
            raise ValueError(f"Jena seal missing approved roles: {missing}")
    channels = int(cfg["channels"])
    sealed = {
        "schema": "tsfm_peft_practical_controls_v7_evaluation_seal_v1",
        "status": "sealed",
        "dataset": dataset,
        "sealed_utc": time.time(),
        "selection": receipt(args.selection),
        "selection_test_policy": selection.get("_selection_test_policy"),
        "data_contract": receipt(data_contract),
        "protocol": receipt(HERE / "protocol.json"),
        "plan": receipt(HERE / "PLAN.md"),
        "approval": receipt(HERE / "APPROVAL.txt"),
        "v6_reuse": reused,
        "channels": channels,
        "splits": cfg["splits"],
        "expected_origins": cfg["expected_origins"],
        "evaluation_stride": int(cfg["evaluation_stride"]),
        "interval_minutes": int(cfg["interval_minutes"]),
        "bootstrap": {"draws": int(protocol()["bootstrap"]["draws"]), "seed": int(protocol()["bootstrap"]["seed"]), "block_length_origins": int(cfg["bootstrap_block_origins"])},
        "metric": "TRAIN-standardized raw-space observed-channel macro MSE/MAE; seed losses averaged, not ensembled.",
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
    for key in ("selection", "data_contract", "protocol", "plan", "approval"):
        check_receipt(sealed[key])
    for row in sealed["models"]:
        if row["kind"] == "checkpoint":
            check_receipt({"path": row["checkpoint"], "sha256": row["checkpoint_sha256"]})
        elif row["kind"] == "stored_prediction":
            check_receipt(row["prediction"])
        elif row["kind"] == "nlinear_weights":
            check_receipt(row["weights"])
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


def nlinear_predict(inputs: np.ndarray, row: dict[str, Any]) -> np.ndarray:
    with np.load(check_receipt(row["weights"]), allow_pickle=False) as archive:
        weight = archive["weight"].astype(np.float64)
        bias = archive["bias"].astype(np.float64)
    if weight.shape != (CONTEXT, HORIZON) or bias.shape != (HORIZON,):
        raise ValueError(f"Direct NLinear weight shape mismatch for {row['id']}")
    last = inputs[:, -1:, :].astype(np.float64)
    centered = inputs.astype(np.float64) - last
    prediction = np.einsum("blc,lh->bhc", centered, weight) + bias[None, :, None] + last
    return prediction.astype(np.float32)


def checkpoint_predict(row: dict[str, Any], inputs_all: np.ndarray, batch_size: int, job: Job) -> np.ndarray:
    import torch

    module = load_module(HERE / "model_v7.py", "_v7_evaluation_model")
    model = module.restore_model(resolve(row["checkpoint"]), device="cuda").eval()
    outputs = []
    with torch.inference_mode():
        for offset in range(0, len(inputs_all), batch_size):
            job.check_limits()
            inputs = torch.as_tensor(inputs_all[offset: offset + batch_size], dtype=torch.float32, device="cuda")
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
    return {"squared": (delta * delta).sum(axis=1), "absolute": np.abs(delta).sum(axis=1), "signed": delta.sum(axis=1), "count": observed.sum(axis=1)}


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


def pair_ids(ids: list[str], models: dict[str, dict[str, Any]]) -> list[str]:
    if len(ids) == 1:
        return ids * len(SEEDS)
    by_seed = {models[i]["spec"].get("seed"): i for i in ids}
    if set(by_seed) != set(SEEDS):
        raise ValueError(f"Expected paired seeds {SEEDS} or deterministic singleton; got {ids}")
    return [by_seed[seed] for seed in SEEDS]


def compare_terms(comparison: dict[str, Any], terms: dict[str, dict[str, np.ndarray]], models: dict[str, dict[str, Any]], period_indices: dict[str, np.ndarray], sealed: dict[str, Any], job: Job) -> dict[str, Any]:
    left_ids = pair_ids(list(comparison["left"]), models)
    right_ids = pair_ids(list(comparison["right"]), models)
    count = terms[left_ids[0]]["count"]
    for identifier in left_ids + right_ids:
        if not np.array_equal(terms[identifier]["count"], count):
            raise ValueError("All paired comparisons must use the same observed target counts")
    left = np.mean([terms[i]["squared"] for i in left_ids], axis=0)
    right = np.mean([terms[i]["squared"] for i in right_ids], axis=0)
    block = int(sealed["bootstrap"]["block_length_origins"])
    draws = int(sealed["bootstrap"]["draws"])
    rng_seed = int(sealed["bootstrap"]["seed"])

    def values(index: np.ndarray) -> dict[str, float] | None:
        denominator = count[index].sum(axis=0)
        if np.any(denominator == 0):
            return None
        a = float(np.mean(left[index].sum(axis=0) / denominator))
        b = float(np.mean(right[index].sum(axis=0) / denominator))
        return {"left_mse": a, "reference_mse": b, "delta_mse": a - b, "relative_to_reference_pct": 100 * (a - b) / b if b > 0 else None}

    output = dict(comparison, direction="left minus reference; negative favors left", left_seed_ids=left_ids, right_seed_ids=right_ids, periods={})
    for label in (*PERIODS, "combined"):
        strata = [period_indices[label]] if label != "combined" else [period_indices[p] for p in PERIODS]
        full_index = np.concatenate(strata)
        full = values(full_index)
        seed_values = []
        for seed, left_id, right_id in zip(SEEDS, left_ids, right_ids):
            a = scores_from_terms(terms[left_id], full_index)["mse"]
            b = scores_from_terms(terms[right_id], full_index)["mse"]
            seed_values.append({"seed": seed, "left_id": left_id, "right_id": right_id, "left_mse": a, "reference_mse": b, "delta_mse": a - b if a is not None and b is not None else None, "relative_to_reference_pct": 100 * (a - b) / b if a is not None and b is not None and b > 0 else None})
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
            interval = {key: np.quantile([row[key] for row in samples], [0.025, 0.975]).tolist() for key in ("delta_mse", "relative_to_reference_pct")}
        output["periods"][label] = {"full": full, "paired_seed_deltas": seed_values, "conditional_block95": interval, "draws": draws, "unavailable_draws": unavailable, "block_length_origins": block, "strata": [label] if label != "combined" else list(PERIODS), "interval_status": "available" if interval is not None else "unavailable_without_dropping_channels"}
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
        raise FileExistsError("This script does not overwrite exposed v7 TEST results; use a documented correction script if needed")
    if result_path.exists() or partial_path.exists() or prediction_dir.exists():
        raise FileExistsError("Preserve existing evaluation attempt; use a new name")
    save_json(exposure_path, {"schema": "tsfm_peft_practical_controls_v7_test_exposure_v1", "dataset": dataset, "status": "exposed", "first_exposed_utc": time.time(), "seal_sha256": seal_receipt["sha256"], "seal": seal_receipt, "evaluation_name": args.name, "command": sys.argv, "scope": "Written after output path collision checks and before the first protected TEST archive is opened for this dataset and seal."})
    data = load_data(dataset, include_test=True)
    origins_by_period = validate_origins(sealed, data)
    origins = np.concatenate([origins_by_period[p] for p in PERIODS])
    period_indices = {"test_a": np.arange(len(origins_by_period["test_a"]), dtype=np.int64), "test_b": np.arange(len(origins_by_period["test_a"]), len(origins), dtype=np.int64)}
    inputs = np.stack([data["x"][o - CONTEXT:o] for o in origins]).astype(np.float32)
    target = np.stack([data["x"][o:o + HORIZON] for o in origins]).astype(np.float32)
    observed = np.stack([data["finite"][o:o + HORIZON] for o in origins]).astype(bool)
    if inputs.shape[1:] != (CONTEXT, int(sealed["channels"])) or target.shape[1:] != (HORIZON, int(sealed["channels"])):
        raise ValueError("Sealed TEST tensor shape differs from approved contract")
    prediction_dir.mkdir(parents=True, exist_ok=False)
    result = {"schema": "tsfm_peft_practical_controls_v7_test_evaluation_v1", "status": "running", "name": args.name, "dataset": dataset, "seal": seal_receipt, "first_exposure": receipt(exposure_path), "source": source_receipt(), "environment": environment_receipt(), "channels": np.asarray(data["columns"]).astype(str).tolist(), "origins": {p: origins_by_period[p].astype(int).tolist() for p in PERIODS}, "roles": sealed["roles"], "metric": sealed["metric"], "models": {}, "role_scores": {}, "comparisons": []}
    save_json(partial_path, result)
    models = {row["id"]: row for row in sealed["models"]}
    terms: dict[str, dict[str, np.ndarray]] = {}
    for identifier, row in models.items():
        job.check_limits()
        reused = False
        if row["kind"] == "stored_prediction":
            prediction, prediction_receipt = load_prediction_receipt(row, origins)
            reused = True
        elif row["kind"] == "nlinear_weights":
            prediction = nlinear_predict(inputs, row)
            prediction_path = prediction_dir / f"{identifier}.npz"
            np.savez_compressed(prediction_path, prediction=prediction, origins=origins)
            prediction_receipt = receipt(prediction_path)
        else:
            prediction = checkpoint_predict(row, inputs, int(protocol()["eval_batch_origins"]), job)
            prediction_path = prediction_dir / f"{identifier}.npz"
            np.savez_compressed(prediction_path, prediction=prediction, origins=origins)
            prediction_receipt = receipt(prediction_path)
        terms[identifier] = error_terms(prediction, target, observed)
        scores = {period: scores_from_terms(terms[identifier], period_indices[period]) for period in PERIODS}
        scores["combined"] = scores_from_terms(terms[identifier])
        result["models"][identifier] = {"spec": row["spec"], "kind": row["kind"], "role": row["role"], "prediction": prediction_receipt, "reused_v6_prediction": reused, "scores": scores}
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
            result["role_scores"][role][period] = {"run_ids": known, "count": len(known), "mse": float(np.mean([score["mse"] for score in scores])) if available else None, "mae": float(np.mean([score["mae"] for score in scores])) if available else None, "scope": "Mean of individual model channel-macro losses; not an ensemble."}
    for comparison in sealed["comparisons"]:
        if all(identifier in terms for identifier in comparison["left"] + comparison["right"]):
            result["comparisons"].append(compare_terms(comparison, terms, models, period_indices, sealed, job))
            job.heartbeat(f"{dataset}: compared {comparison['name']}")
    result.update(status="complete", completed_utc=time.time(), combined_definition="Within each seed/channel, sum TEST-A/B errors and observed counts before channel mean; then average individual seed losses. Deterministic references broadcast for seed-paired deltas.", test_reselection=False, new_test_model_predictions=sum(not item["reused_v6_prediction"] for item in result["models"].values()))
    save_json(result_path, result)
    return {"result": rel(result_path), "models": len(result["models"]), "comparisons": len(result["comparisons"])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Seal and evaluate approved v7 fixed comparisons.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--seal", action="store_true")
    action.add_argument("--evaluate", action="store_true")
    parser.add_argument("--dataset", choices=("robin", "jena"), required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--reserve-seconds", type=float, required=True)
    parser.add_argument("--include-v6-robin", action="store_true")
    args = parser.parse_args()
    validate_name(args.name)
    configure_numeric_runtime()
    if args.dataset != "robin" and args.include_v6_robin:
        raise ValueError("--include-v6-robin is only valid for Robin")
    attempt_suffix = "seal_attempt" if args.seal else "evaluate_attempt"
    attempt_path = HERE / f"{args.name}_{attempt_suffix}.json"
    if attempt_path.exists():
        raise FileExistsError("Preserve existing attempt file; use a new name")
    attempt = {"status": "running", "started_utc": time.time(), "command": sys.argv, "dataset": args.dataset, "source": source_receipt(), "environment": environment_receipt()}
    save_json(attempt_path, attempt)
    try:
        if args.seal:
            with Job(f"v7_seal_{args.dataset}_{args.name}", category="cpu_analysis", reserve_seconds=args.reserve_seconds, metadata={"dataset": args.dataset, "name": args.name, "action": "seal"}):
                result = seal_evaluation(args)
        else:
            with Job(f"v7_evaluate_{args.dataset}_{args.name}", category="gpu", reserve_seconds=args.reserve_seconds, metadata={"dataset": args.dataset, "name": args.name, "action": "evaluate"}) as job:
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
