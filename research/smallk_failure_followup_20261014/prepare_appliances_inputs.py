"""Appliances-only input bridge for loss warm-start follow-up runs.

Importing this module is source-only.  Actual array/checkpoint reads happen only
through ``--execute`` under ``FollowupLimit``.  The bridge reconstructs the old
Appliances normalized histories from the materialized UCI contract and verifies
them against already exposed RAW/LEARNED prediction artifacts before writing a
fresh follow-up cache bundle.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
import traceback
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FOLLOWUP_CACHE = ROOT / ".cache" / "smallk_failure_followup_20261014"
CACHE_ROOT = FOLLOWUP_CACHE / "appliances_input_bridge"
DEFAULT_OUTPUT = CACHE_ROOT / "phase_v2"

DATASET = "appliances"
SPLITS = ("fit", "tune", "eval")
TRAINING_SPLITS = ("fit", "tune")
EVAL_SPLIT = "eval"
SEEDS = (92601, 92602)
L = 512
H = 48
C = 26
K = 6
CHECKPOINT_UPDATES = (0, 100, 200, 400, 600)
LOSS_CANDIDATES = ("MSE", "MAE", "HUBER_DELTA1")


@dataclass(frozen=True)
class FileAsset:
    path: str
    sha256: str | None = None
    role: str = ""

    @property
    def resolved(self) -> Path:
        value = Path(self.path)
        return value if value.is_absolute() else ROOT / value


SOURCE_ASSETS = {
    "new_run": FileAsset(
        "research/smallk_predictive_alignment_20261013/new_run.py",
        "5e4e7a3f95ee97b13bc0e14d52b693030d02b94d964cbb09d46a6d42e62ae2ee",
        "old_new_dataset_runner",
    ),
    "engine": FileAsset(
        "research/smallk_predictive_alignment_20261013/engine.py",
        "05f2cf63b3f7f6a01bf722c51771455be9c73c21ab301d7eb950b3dd58bab817",
        "old_engine",
    ),
    "controls": FileAsset(
        "research/smallk_predictive_alignment_20261013/controls.py",
        "324c75a65a001ac0cae30610abb148499718cbad838a69e0012df4a85bd83393",
        "old_controls",
    ),
    "newdata": FileAsset(
        "research/smallk_predictive_alignment_20261013/newdata.py",
        "ccef24b64c9beef0e6e31f9bc49dbad4e8e6ad57a6ce7c99973ccbce691de3bd",
        "old_appliances_loader",
    ),
    "alignment": FileAsset(
        "research/smallk_predictive_alignment_20261013/alignment.py",
        "84d5be3e310251987fb4f7403399b9a7b9d37f3a54410e85c68613bfeeac4537",
        "old_alignment_q_aux_source",
    ),
    "method_lock": FileAsset(
        "research/smallk_predictive_alignment_20261013/METHOD_LOCK.json",
        "38fde1bf35b785684944c785a9b7bec7b12cec0f2608fc0c202de04d0632cc82",
        "old_method_lock",
    ),
    "run_new_result": FileAsset(
        "research/smallk_predictive_alignment_20261013/run_new_v1/RESULT.json",
        None,
        "old_run_terminal_manifest_partial_failed_paths_allowed",
    ),
    "eval_first_open": FileAsset(
        "research/smallk_predictive_alignment_20261013/run_new_v1/appliances_EVAL_FIRST_OPEN.json",
        None,
        "already_exposed_eval_open_receipt",
    ),
}

RAW_JSONS = {
    "dev": FileAsset("research/smallk_predictive_alignment_20261013/run_new_v1/appliances/RAW_CONTROLS.json", role="raw_controls_dev"),
    "eval": FileAsset("research/smallk_predictive_alignment_20261013/run_new_v1/appliances/EVAL_RAW_CONTROLS.json", role="raw_controls_eval"),
}

LEARNED_CASE_JSONS = {
    seed: FileAsset(
        f"research/smallk_predictive_alignment_20261013/run_new_v1/appliances/"
        f"S3_ANCHOR_NONLINEAR_D_LEARNED_REGIME_w0.1_s{seed}.json",
        role=f"selected_learned_regime_case_seed_{seed}",
    )
    for seed in SEEDS
}

LEARNED_CONTROL_JSONS = {
    seed: {
        "dev": FileAsset(
            f"research/smallk_predictive_alignment_20261013/run_new_v1/appliances/"
            f"S3_ANCHOR_NONLINEAR_D_LEARNED_REGIME_w0.1_s{seed}_CONTROLS.json",
            role=f"learned_controls_dev_seed_{seed}",
        ),
        "eval": FileAsset(
            f"research/smallk_predictive_alignment_20261013/run_new_v1/appliances/"
            f"EVAL_S3_ANCHOR_NONLINEAR_D_LEARNED_REGIME_w0.1_s{seed}_CONTROLS.json",
            role=f"learned_controls_eval_seed_{seed}",
        ),
    }
    for seed in SEEDS
}

EXPECTED_COUNTS = {"fit": 256, "tune": 64, "eval": 82}
EXPECTED_RAW_ORIGIN_SHA = {
    "fit": "68712f8dac18bfe80b8136ec69272d5dab00e49254e077fcb4818bc083d4911b",
    "tune": "26bb0e09e78106f57b391bd1b05a174ad4e260377786d6444c97c48272506a86",
    "eval": "edc1cc3b24d2a2511d6b72dc6f54dc9cc9186d6df5f2253205ddc86fc2de62a3",
}

METHOD_PLAN = {
    "dataset": DATASET,
    "C": C,
    "K": K,
    "architecture": "ANCHOR_NONLINEAR_D",
    "source_aux_method": "LEARNED_REGIME",
    "source_aux_weight": 0.1,
    "warm_start_from_same_old_selected_learned_codec": True,
    "seeds": list(SEEDS),
    "loss_candidates": list(LOSS_CANDIDATES),
    "huber_delta": 1.0,
    "updates": 600,
    "batch_size": 4,
    "checkpoints": list(CHECKPOINT_UPDATES),
    "selection_primary_metric": "TUNE normalized observed channel-macro MSE against actual Y",
    "selection_includes_u0": True,
    "selection_includes_u600": True,
    "selection_tie_rule": "earliest exact tie",
    "fit_loss_channel_balance": "normalized observed channel macro; no pooling by raw unit scale",
    "frozen_components": ["original_F", "old_codec_architecture_state_before_new_loss_update"],
    "future_alignment_auxiliary": "disabled_for_all_new_loss_candidates",
    "q_aux_state": "old q checkpoint tensors are preserved as provenance only; not loaded into deployment or new loss training",
    "eval_policy": "EVAL predictions/targets already exposed; no open_evaluation call in this bridge",
}


class ApplianceBridgeError(RuntimeError):
    pass


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def rel(path: str | Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_json_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(jsonable(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def contains_protected_test_path(path: str | Path) -> bool:
    value = Path(path)
    return any(part.lower() == "test" for part in value.parts) or value.name.lower() in {"test.npz", "jena_test.npz"}


def guard_no_test_path(path: str | Path, *, role: str = "") -> None:
    if contains_protected_test_path(path):
        raise PermissionError(f"PROTECTED_TEST_PATH_DENIED:{role}:{path}")


def resolve_asset(asset: FileAsset) -> Path:
    path = asset.resolved
    guard_no_test_path(path, role=asset.role)
    return path


def ensure_file_hash(asset: FileAsset) -> dict[str, Any]:
    path = resolve_asset(asset)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_path(path)
    if asset.sha256 is not None and actual != asset.sha256:
        raise ApplianceBridgeError(f"ASSET_SHA_MISMATCH:{asset.path}:{actual}!={asset.sha256}")
    return {"path": rel(path), "sha256": actual, "role": asset.role, "bytes": path.stat().st_size}


def read_json_asset(asset: FileAsset) -> tuple[dict[str, Any], dict[str, Any]]:
    record = ensure_file_hash(asset)
    data = json.loads(resolve_asset(asset).read_text(encoding="utf-8-sig"))
    return data, record


def import_source(asset: FileAsset, name: str):
    ensure_file_hash(asset)
    path = resolve_asset(asset)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def checked_npz(record: Mapping[str, Any], *, role: str, max_uncompressed: int = 1024 * 1024**2) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = Path(record["path"])
    if not path.is_absolute():
        path = ROOT / path
    guard_no_test_path(path, role=role)
    path = path.resolve()
    before = sha256_path(path)
    expected = record.get("sha256")
    if expected and before != expected:
        raise ApplianceBridgeError(f"NPZ_SHA_MISMATCH:{role}:{before}!={expected}")
    with zipfile.ZipFile(path) as archive:
        uncompressed = sum(item.file_size for item in archive.infolist())
    if uncompressed > max_uncompressed:
        raise ApplianceBridgeError(f"NPZ_UNCOMPRESSED_LIMIT:{role}:{uncompressed}")
    with np.load(path, allow_pickle=False) as opened:
        arrays = {key: opened[key].copy() for key in opened.files}
    after = sha256_path(path)
    if after != before:
        raise ApplianceBridgeError(f"NPZ_CHANGED_DURING_READ:{role}")
    return arrays, {"path": rel(path), "sha256": before, "bytes": path.stat().st_size, "uncompressed_bytes": uncompressed, "role": role}


def arrays_equal_exact(left: np.ndarray, right: np.ndarray) -> bool:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    return bool(np.array_equal(left, right, equal_nan=True))


def require_exact(left: np.ndarray, right: np.ndarray, label: str) -> None:
    if not arrays_equal_exact(left, right):
        diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
        max_abs = float(np.nanmax(np.abs(diff))) if diff.size else 0.0
        raise ApplianceBridgeError(f"ARRAY_EXACT_MISMATCH:{label}:max_abs={max_abs}")


def old_window_preprocess(physical: np.ndarray, observed: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    physical = np.asarray(physical, dtype=np.float32)
    observed = np.asarray(observed, dtype=bool)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    filled = np.empty_like(physical, dtype=np.float32)
    last = np.broadcast_to(mean, (len(physical), len(mean))).copy()
    for t in range(physical.shape[1]):
        last = np.where(observed[:, t], physical[:, t], last)
        filled[:, t] = last
    return ((filled - mean) / std).astype(np.float32)


def source_method_record(case: Mapping[str, Any], method: str) -> dict[str, Any]:
    matches = [entry for entry in case.get("methods", []) if entry.get("method") == method]
    if len(matches) != 1:
        raise ApplianceBridgeError(f"METHOD_RECORD_NOT_UNIQUE:{case.get('label')}:{method}:{len(matches)}")
    return matches[0]


def load_training_with_original_loader() -> tuple[dict[str, Any], dict[str, Any]]:
    newdata = import_source(SOURCE_ASSETS["newdata"], "appliance_bridge_newdata")
    method_lock_path = resolve_asset(SOURCE_ASSETS["method_lock"])
    data = newdata.load_training(DATASET, method_lock_path, allow_download=False)
    data["origins"]["eval"] = None
    contract_registry = ROOT / ".cache" / "smallk_predictive_alignment_20261013" / "contract" / f"{DATASET}_{SOURCE_ASSETS['method_lock'].sha256}.json"
    guard_no_test_path(contract_registry, role="contract_registry")
    registry = json.loads(contract_registry.read_text(encoding="utf-8-sig"))
    contract_path = Path(registry["contract_path"])
    guard_no_test_path(contract_path, role="contract_json")
    contract = json.loads(contract_path.read_text(encoding="utf-8-sig"))
    return data, {
        "newdata_source": ensure_file_hash(SOURCE_ASSETS["newdata"]),
        "method_lock": ensure_file_hash(SOURCE_ASSETS["method_lock"]),
        "contract_registry": {"path": rel(contract_registry), "sha256": sha256_path(contract_registry)},
        "contract": {"path": rel(contract_path), "sha256": sha256_path(contract_path), "payloads": contract["payloads"], "origins": contract["origins"]},
    }


def reconstruct_training_split(data: Mapping[str, Any], split: str, mean: np.ndarray, std: np.ndarray) -> dict[str, np.ndarray]:
    origins = data["origins"][split]
    physical = np.stack([data["physical"][o - L : o] for o in origins]).astype(np.float32)
    history_observed = np.stack([data["mask"][o - L : o] for o in origins]).astype(bool)
    history = old_window_preprocess(physical, history_observed, mean, std)
    target = np.stack([data["x"][o : o + H] for o in origins]).astype(np.float32)
    observed = np.stack([data["mask"][o : o + H] for o in origins]).astype(bool)
    return {
        "history": history,
        "history_physical": physical,
        "history_observed": history_observed,
        "target": target,
        "observed": observed,
        "origins": origins.astype(np.int64),
        "raw_last": history[:, -1].astype(np.float32),
    }


def load_eval_past(contract_record: Mapping[str, Any], mean: np.ndarray, std: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    eval_input = contract_record["contract"]["payloads"]["eval_past_input"]
    arrays, source = checked_npz(eval_input, role="eval_past_input_payload")
    for key in ("mean", "std"):
        require_exact(arrays[key].astype(np.float32), mean if key == "mean" else std, f"eval_input:{key}")
    rebuilt = old_window_preprocess(arrays["physical_history"].astype(np.float32), arrays["history_mask"].astype(bool), mean, std)
    require_exact(rebuilt, arrays["history"].astype(np.float32), "eval_history_reconstructed_from_physical")
    return (
        {
            "history": arrays["history"].astype(np.float32),
            "history_physical": arrays["physical_history"].astype(np.float32),
            "history_observed": arrays["history_mask"].astype(bool),
            "origins": arrays["origins"].astype(np.int64),
            "raw_last": arrays["history"][:, -1].astype(np.float32),
        },
        source,
    )


def canonical_prediction_assets() -> tuple[dict[str, Any], dict[str, Any]]:
    raw_dev, raw_dev_json = read_json_asset(RAW_JSONS["dev"])
    raw_eval, raw_eval_json = read_json_asset(RAW_JSONS["eval"])
    raw_bolt = source_method_record(raw_eval, "RAW_BOLT")
    raw_plus_r = source_method_record(raw_eval, "RAW_BOLT_RAW_R")
    learned = {}
    source_cases = {}
    json_records: list[dict[str, Any]] = [raw_dev_json, raw_eval_json]
    for seed in SEEDS:
        case, case_json = read_json_asset(LEARNED_CASE_JSONS[seed])
        dev_control, dev_control_json = read_json_asset(LEARNED_CONTROL_JSONS[seed]["dev"])
        eval_control, eval_control_json = read_json_asset(LEARNED_CONTROL_JSONS[seed]["eval"])
        if case.get("label") != f"S3_ANCHOR_NONLINEAR_D_LEARNED_REGIME_w0.1_s{seed}":
            raise ApplianceBridgeError(f"UNEXPECTED_CASE_LABEL:{seed}:{case.get('label')}")
        if case.get("dataset") != DATASET or case.get("architecture") != "ANCHOR_NONLINEAR_D" or case.get("K") != K:
            raise ApplianceBridgeError(f"UNEXPECTED_CASE_METADATA:{seed}")
        if case.get("aux_method") != "LEARNED_REGIME" or float(case.get("aux_weight")) != 0.1:
            raise ApplianceBridgeError(f"UNEXPECTED_AUX_METHOD:{seed}")
        if eval_control.get("source_checkpoint") != case["training"]["selected_checkpoint"]:
            raise ApplianceBridgeError(f"EVAL_CONTROL_SOURCE_CHECKPOINT_MISMATCH:{seed}")
        if dev_control.get("source_checkpoint") != case["training"]["selected_checkpoint"]:
            raise ApplianceBridgeError(f"DEV_CONTROL_SOURCE_CHECKPOINT_MISMATCH:{seed}")
        source_cases[str(seed)] = {"case": case, "json": case_json, "dev_control_json": dev_control_json, "eval_control_json": eval_control_json}
        learned[str(seed)] = {
            "bolt": source_method_record(eval_control, "BOLT"),
            "bolt_raw_r": source_method_record(eval_control, "BOLT_RAW_R"),
            "source_checkpoint": case["training"]["selected_checkpoint"],
            "source_case_label": case["label"],
            "selected_update": case["training"]["selected_update"],
            "selected_codec_hash": case["training"]["selected_codec_hash"],
            "q_registered_parameters": case["training"].get("q_registered_parameters"),
            "q_active_loss_gradient": case["training"].get("q_active_loss_gradient"),
        }
        json_records.extend([case_json, dev_control_json, eval_control_json])
    return {
        "raw_bolt": raw_bolt,
        "raw_plus_r": raw_plus_r,
        "learned": learned,
        "source_cases": source_cases,
    }, {"json_assets": json_records}


def validate_reference_arrays(reference: Mapping[str, np.ndarray], candidate: Mapping[str, np.ndarray], *, label: str, splits: Iterable[str] = SPLITS) -> None:
    for key in ("mean", "std"):
        require_exact(reference[key].astype(np.float32), candidate[key].astype(np.float32), f"{label}:{key}")
    for split in splits:
        for field in ("target", "observed", "origins"):
            require_exact(reference[f"{split}_{field}"], candidate[f"{split}_{field}"], f"{label}:{split}:{field}")


def load_reference_predictions(asset_records: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    raw_arrays, raw_source = checked_npz(asset_records["raw_bolt"]["prediction"], role="raw_bolt_eval_prediction")
    raw_r_arrays, raw_r_source = checked_npz(asset_records["raw_plus_r"]["prediction"], role="raw_plus_r_eval_prediction")
    validate_reference_arrays(raw_arrays, raw_r_arrays, label="RAW+R_vs_RAW_reference")
    learned_arrays: dict[str, dict[str, np.ndarray]] = {}
    learned_sources: dict[str, Any] = {}
    for seed, seed_record in asset_records["learned"].items():
        arrays, source = checked_npz(seed_record["bolt"]["prediction"], role=f"learned_bolt_eval_prediction_{seed}")
        validate_reference_arrays(raw_arrays, arrays, label=f"LEARNED_BOLT_vs_RAW_reference_{seed}")
        learned_arrays[seed] = arrays
        learned_sources[seed] = source
    return {
        "raw": raw_arrays,
        "raw_plus_r": raw_r_arrays,
        "learned": learned_arrays,
    }, {
        "raw_bolt": raw_source,
        "raw_plus_r": raw_r_source,
        "learned": learned_sources,
    }


def build_bundle() -> tuple[dict[str, Any], dict[str, Any]]:
    data, loader_record = load_training_with_original_loader()
    asset_records, json_record = canonical_prediction_assets()
    predictions, prediction_sources = load_reference_predictions(asset_records)
    raw = predictions["raw"]
    mean = raw["mean"].astype(np.float32)
    std = raw["std"].astype(np.float32)
    if mean.shape != (C,) or std.shape != (C,) or np.any(std <= 0):
        raise ApplianceBridgeError("BAD_STATS_SHAPE_OR_VALUES")
    require_exact(data["mean"].astype(np.float32), mean, "loader_fit_mean_vs_raw")
    require_exact(data["std"].astype(np.float32), std, "loader_fit_std_vs_raw")
    splits: dict[str, dict[str, np.ndarray]] = {}
    for split in TRAINING_SPLITS:
        splits[split] = reconstruct_training_split(data, split, mean, std)
    eval_payload, eval_source = load_eval_past(loader_record, mean, std)
    splits["eval"] = eval_payload
    for split in SPLITS:
        if len(splits[split]["origins"]) != EXPECTED_COUNTS[split]:
            raise ApplianceBridgeError(f"COUNT_MISMATCH:{split}:{len(splits[split]['origins'])}!={EXPECTED_COUNTS[split]}")
        if array_sha256(splits[split]["origins"]) != EXPECTED_RAW_ORIGIN_SHA[split]:
            raise ApplianceBridgeError(f"ORIGIN_SHA_MISMATCH:{split}")
        require_exact(splits[split]["target"] if split != "eval" else raw[f"{split}_target"], raw[f"{split}_target"], f"{split}:target_raw")
        require_exact(splits[split]["observed"] if split != "eval" else raw[f"{split}_observed"], raw[f"{split}_observed"], f"{split}:observed_raw")
        require_exact(splits[split]["origins"], raw[f"{split}_origins"], f"{split}:origins_raw")
        if split == "eval":
            splits[split]["target"] = raw[f"{split}_target"].astype(np.float32)
            splits[split]["observed"] = raw[f"{split}_observed"].astype(bool)
        for seed, arrays in predictions["learned"].items():
            require_exact(splits[split]["target"], arrays[f"{split}_target"], f"{split}:target_learned_{seed}")
            require_exact(splits[split]["observed"], arrays[f"{split}_observed"], f"{split}:observed_learned_{seed}")
            require_exact(splits[split]["origins"], arrays[f"{split}_origins"], f"{split}:origins_learned_{seed}")
    bundle = {
        "dataset": DATASET,
        "splits": splits,
        "mean": mean,
        "std": std,
        "columns": data["columns"],
        "interval_seconds": data["interval_seconds"],
        "asset_records": asset_records,
        "reference_predictions": predictions,
    }
    provenance = {
        "loader": loader_record,
        "json_assets": json_record["json_assets"],
        "prediction_sources": prediction_sources,
        "eval_past_input": eval_source,
        "source_assets": {name: ensure_file_hash(asset) for name, asset in SOURCE_ASSETS.items()},
        "method_plan": METHOD_PLAN,
    }
    return bundle, provenance


def tensor_state_hash(state: Any) -> str | None:
    if state is None:
        return None
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if hasattr(value, "detach"):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
        digest.update(key.encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def inspect_checkpoint(record: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(record["path"])
    if not path.is_absolute():
        path = ROOT / path
    guard_no_test_path(path, role="selected_checkpoint")
    actual = sha256_path(path)
    if actual != record["sha256"]:
        raise ApplianceBridgeError(f"CHECKPOINT_SHA_MISMATCH:{path}:{actual}!={record['sha256']}")
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    codec = payload.get("codec")
    if not isinstance(codec, dict):
        raise ApplianceBridgeError("CHECKPOINT_CODEC_STATE_MISSING")
    q_state = payload.get("q")
    return {
        "path": rel(path),
        "sha256": actual,
        "bytes": path.stat().st_size,
        "payload_update": int(payload.get("update")),
        "payload_seed": int(payload.get("seed")),
        "payload_architecture": payload.get("architecture"),
        "payload_method": payload.get("method"),
        "payload_weight": float(payload.get("weight")),
        "codec_state_keys": sorted(codec),
        "codec_state_sha256": tensor_state_hash(codec),
        "fixed_basis_sha256": tensor_state_hash({"fixed_basis": codec["fixed_basis"]}),
        "q_state_present": q_state is not None,
        "q_state_sha256": tensor_state_hash(q_state) if q_state is not None else None,
        "q_aux_deployment_use": "preserved_not_used",
    }


def checkpoint_metadata(asset_records: Mapping[str, Any]) -> dict[str, Any]:
    records = {}
    for seed, seed_record in asset_records["learned"].items():
        cp = seed_record["source_checkpoint"]
        meta = inspect_checkpoint(cp)
        if meta["payload_seed"] != int(seed) or meta["payload_update"] != seed_record["selected_update"]:
            raise ApplianceBridgeError(f"CHECKPOINT_PAYLOAD_METADATA_MISMATCH:{seed}")
        if meta["payload_architecture"] != "ANCHOR_NONLINEAR_D" or meta["payload_method"] != "LEARNED_REGIME" or meta["payload_weight"] != 0.1:
            raise ApplianceBridgeError(f"CHECKPOINT_PAYLOAD_METHOD_MISMATCH:{seed}")
        records[seed] = {**seed_record, "checkpoint_cpu_inspection": meta}
    return records


def split_arrays_for_save(bundle: Mapping[str, Any], split: str) -> dict[str, np.ndarray]:
    values = bundle["splits"][split]
    refs = bundle["reference_predictions"]
    arrays = {
        "history": values["history"].astype(np.float32),
        "history_physical": values["history_physical"].astype(np.float32),
        "history_observed": values["history_observed"].astype(bool),
        "target": values["target"].astype(np.float32),
        "observed": values["observed"].astype(bool),
        "origins": values["origins"].astype(np.int64),
        "raw_last": values["raw_last"].astype(np.float32),
        "raw_forecast": refs["raw"][f"{split}_prediction"].astype(np.float32),
        "raw_plus_r_forecast": refs["raw_plus_r"][f"{split}_prediction"].astype(np.float32),
        "mean": bundle["mean"].astype(np.float32),
        "std": bundle["std"].astype(np.float32),
    }
    for seed, learned in refs["learned"].items():
        arrays[f"learned_{seed}_forecast"] = learned[f"{split}_prediction"].astype(np.float32)
    return arrays


def array_manifest(path: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        "path": rel(path),
        "sha256": sha256_path(path),
        "arrays": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": array_sha256(value)}
            for key, value in arrays.items()
        },
    }


def save_split_npz(output: Path, bundle: Mapping[str, Any], split: str) -> dict[str, Any]:
    arrays = split_arrays_for_save(bundle, split)
    path = output / f"appliances_{split}_inputs.npz"
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    return {"split": split, "dataset": DATASET, **array_manifest(path, arrays)}


def save_sources(output: Path) -> dict[str, Any]:
    folder = output / "sources"
    folder.mkdir()
    copied = []
    for source in (Path(__file__).resolve(), HERE / "followup_runtime.py"):
        target = folder / source.name
        with source.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        copied.append({"path": rel(source), "sha256": sha256_path(source), "snapshot_path": rel(target), "snapshot_sha256": sha256_path(target)})
    write_json_once(folder / "MANIFEST.json", {"files": copied})
    return {"path": rel(folder / "MANIFEST.json"), "sha256": sha256_path(folder / "MANIFEST.json"), "files": copied}


def write_report(path: Path, result: Mapping[str, Any]) -> None:
    lines = [
        "# Appliances loss warm-start input bridge 준비",
        "",
        f"- 상태: `{result['status']}` / resource terminal `{result.get('resource_terminal_status')}`",
        f"- 데이터셋: `{DATASET}`, C={C}, K={K}, splits fit/tune/eval={EXPECTED_COUNTS}",
        "- [확인] 새 GPU 학습, TEST read, old campaign restart, `open_evaluation()` 호출은 수행하지 않았다.",
        "- [확인] FIT/TUNE history는 기존 newdata loader의 materialized contract에서 복원했고, EVAL history는 이미 저장된 eval past input payload에서 읽어 physical history와 재전처리를 byte-exact 비교했다.",
        "- [확인] target/mask/origins/mean/std는 이미 노출된 RAW/LEARNED prediction NPZ와 exact 비교했다.",
        "- [확인] 두 seed의 old selected `S3_ANCHOR_NONLINEAR_D_LEARNED_REGIME_w0.1` checkpoint를 CPU로 해시 확인했다.",
        "",
        "## Method lock",
        "",
        "- 새 후보 loss: MSE, MAE, Huber delta=1.",
        "- selection primary는 계속 TUNE normalized observed channel-macro MSE이며, MAE/Huber는 학습 loss 비교일 뿐 TUNE-MAE constraint selection으로 바꾸지 않는다.",
        "- 각 seed/loss는 old selected LEARNED codec에서 warm start하고 u0/100/200/400/600 checkpoint를 actual Y TUNE MSE로 고른다.",
        "- q auxiliary/readout state는 provenance로만 보존하며 deployment와 새 loss 학습에는 사용하지 않는다.",
        "",
        "## Output API",
        "",
        "- `load_prepared_bundle(result_path)` returns `manifest`, `splits`, `mean`, `std`, `columns`, `checkpoint_cases`, `raw_references`.",
        "- split arrays include normalized `history`, `target`, `observed`, `origins`, `mean`, `std`, plus `history_physical`, `history_observed`, `raw_last`, RAW/RAW+R/LEARNED reference forecasts.",
        "- `load_codec_checkpoint_state(checkpoint_record)` verifies checkpoint SHA and returns CPU codec/q state dictionaries for the runner.",
    ]
    with path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def execute(output: Path) -> dict[str, Any]:
    from research.smallk_failure_followup_20261014 import followup_runtime as fr

    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {
        "status": "RUNNING",
        "schema": "smallk_appliances_loss_input_bridge_v1",
        "dataset": DATASET,
        "gpu_used": False,
        "protected_TEST_access": False,
        "new_gpu_training": False,
        "open_evaluation_called": False,
        "old_campaign_restart": False,
        "output": rel(output),
        "method_plan": METHOD_PLAN,
    }
    limit = None
    body_started = None
    try:
        try:
            limit = fr.FollowupLimit("appliances_input_bridge_v1", False, followup_roots=(HERE, FOLLOWUP_CACHE, output))
        except Exception as error:
            result.update(status="FAILED_PRESERVED", failure={"type": type(error).__name__, "message": str(error)})
            result["resource_terminal"] = fr.LAST_CONSTRUCTOR_TERMINAL or {"terminal_status": "FAILED_PRESERVED", "terminal_error": result["failure"]}
            result["resource_terminal_status"] = result["resource_terminal"].get("terminal_status")
            raise
        body_started = time.perf_counter()
        bundle, provenance = build_bundle()
        limit.check()
        checkpoint_cases = checkpoint_metadata(bundle["asset_records"])
        split_records = [save_split_npz(output, bundle, split) for split in SPLITS]
        source_manifest = save_sources(output)
        result.update(
            body_wall_seconds=time.perf_counter() - body_started,
            C=C,
            K=K,
            split_counts={split: int(len(bundle["splits"][split]["origins"])) for split in SPLITS},
            columns=list(bundle["columns"]),
            mean={"shape": list(bundle["mean"].shape), "dtype": str(bundle["mean"].dtype), "sha256": array_sha256(bundle["mean"])},
            std={"shape": list(bundle["std"].shape), "dtype": str(bundle["std"].dtype), "sha256": array_sha256(bundle["std"])},
            artifacts=split_records,
            checkpoint_cases=checkpoint_cases,
            raw_references={
                "RAW_BOLT": bundle["asset_records"]["raw_bolt"]["prediction"],
                "RAW_BOLT_RAW_R": bundle["asset_records"]["raw_plus_r"]["prediction"],
                "LEARNED_BOLT": {seed: record["bolt"]["prediction"] for seed, record in bundle["asset_records"]["learned"].items()},
            },
            provenance_details={**provenance, "source_snapshot": source_manifest},
        )
        write_json_once(output / "PROVENANCE.json", result["provenance_details"])
        result["provenance"] = {"path": rel(output / "PROVENANCE.json"), "sha256": sha256_path(output / "PROVENANCE.json")}
        result["status"] = "COMPLETE"
    except Exception as error:
        result["body_wall_seconds"] = None if body_started is None else time.perf_counter() - body_started
        result.update(
            status="FAILED_PRESERVED",
            failure={"type": type(error).__name__, "message": str(error), "traceback_tail": traceback.format_exc().splitlines()[-8:]},
        )
    finally:
        if limit is not None:
            receipt = limit.close(result["status"], None if result["status"] == "COMPLETE" else result.get("failure"))
            result["resource_terminal"] = receipt
            result["resource_terminal_status"] = receipt.get("terminal_status") or receipt.get("status")
            result["resource_terminal_error"] = receipt.get("terminal_error")
            if result["status"] == "COMPLETE" and result["resource_terminal_status"] != "COMPLETE":
                result["status"] = result["resource_terminal_status"]
                result["failure"] = {"type": "FollowupLimitTerminalStatus", "message": str(result["resource_terminal_error"])}
        result["torch_imported"] = "torch" in sys.modules
        result["torch_import_scope"] = "torch is used only for CPU checkpoint inspection and the required FollowupLimit/Bi runtime import; no CUDA execution is performed."
        write_json_once(output / "FOLLOWUP_RUNTIME_RECEIPT.json", result.get("resource_terminal"))
        result["runtime_receipt"] = {"path": rel(output / "FOLLOWUP_RUNTIME_RECEIPT.json"), "sha256": sha256_path(output / "FOLLOWUP_RUNTIME_RECEIPT.json")}
        write_report(output / "METHOD_LOCK_APPLIANCES_LOSS_v1.md", result)
        result["method_lock_report"] = {"path": rel(output / "METHOD_LOCK_APPLIANCES_LOSS_v1.md"), "sha256": sha256_path(output / "METHOD_LOCK_APPLIANCES_LOSS_v1.md")}
        write_json_once(output / "RESULT.json", result)
    return result


def load_prepared_bundle(result_path: str | Path = DEFAULT_OUTPUT / "RESULT.json") -> dict[str, Any]:
    result_path = Path(result_path)
    if not result_path.is_absolute():
        result_path = ROOT / result_path
    guard_no_test_path(result_path, role="prepared_result")
    manifest = json.loads(result_path.read_text(encoding="utf-8-sig"))
    if manifest.get("status") != "COMPLETE":
        raise ApplianceBridgeError(f"PREPARED_BUNDLE_NOT_COMPLETE:{manifest.get('status')}")
    splits = {}
    for record in manifest["artifacts"]:
        arrays, source = checked_npz(record, role=f"prepared_split_{record['split']}")
        expected_arrays = record["arrays"]
        for key, meta in expected_arrays.items():
            value = arrays[key]
            if list(value.shape) != meta["shape"] or str(value.dtype) != meta["dtype"] or array_sha256(value) != meta["sha256"]:
                raise ApplianceBridgeError(f"PREPARED_ARRAY_CONTRACT_MISMATCH:{record['split']}:{key}")
        arrays["_source"] = source
        splits[record["split"]] = arrays
    return {
        "manifest": manifest,
        "splits": splits,
        "mean": splits["fit"]["mean"],
        "std": splits["fit"]["std"],
        "columns": manifest["columns"],
        "checkpoint_cases": manifest["checkpoint_cases"],
        "raw_references": manifest["raw_references"],
        "method_plan": manifest["method_plan"],
    }


def load_codec_checkpoint_state(checkpoint_record: Mapping[str, Any], *, torch_module: Any | None = None) -> dict[str, Any]:
    path = Path(checkpoint_record["path"])
    if not path.is_absolute():
        path = ROOT / path
    guard_no_test_path(path, role="codec_checkpoint_state")
    actual = sha256_path(path)
    if actual != checkpoint_record["sha256"]:
        raise ApplianceBridgeError(f"CHECKPOINT_SHA_MISMATCH:{actual}!={checkpoint_record['sha256']}")
    if torch_module is None:
        import torch as torch_module
    payload = torch_module.load(path, map_location="cpu", weights_only=True)
    return {
        "codec": payload["codec"],
        "q": payload.get("q"),
        "update": payload.get("update"),
        "seed": payload.get("seed"),
        "architecture": payload.get("architecture"),
        "method": payload.get("method"),
        "weight": payload.get("weight"),
        "q_aux_deployment_use": "preserved_not_used",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = execute(args.output)
    print(json.dumps({"status": result["status"], "output": rel(args.output.resolve()), "terminal": result.get("resource_terminal_status")}, ensure_ascii=False))
    return 0 if result["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
