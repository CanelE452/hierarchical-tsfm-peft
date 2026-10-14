"""Old-condition input bridge for the small-K follow-up probes.

Importing this module is preparation-only: it does not import the old engine,
open old NPZ/checkpoint/model assets, or touch GPU state.  Actual scientific
reads are only reachable through ``--execute-after-bi`` after a caller-provided
Bi audit/report/manifest phase barrier.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / ".cache" / HERE.name / "input_bridge"
L = 512
H = 48
SEEDS = (92601, 92602)


@dataclass(frozen=True)
class FileAsset:
    path: str
    sha256: str | None = None
    role: str = ""

    @property
    def resolved(self) -> Path:
        value = Path(self.path)
        return value if value.is_absolute() else ROOT / value


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    output_key: str
    splits: tuple[str, ...]
    eval_split: str
    channels: int
    counts: Mapping[str, int]
    source_role: str


ROBIN = DatasetSpec(
    key="robin",
    output_key="robin",
    splits=("fit", "tune", "val"),
    eval_split="val",
    channels=17,
    counts={"fit": 256, "tune": 64, "val": 30},
    source_role="reused_development",
)

BEIJING = DatasetSpec(
    key="beijing_aotizhongxin",
    output_key="beijing_aotizhongxin",
    splits=("fit", "tune", "eval"),
    eval_split="eval",
    channels=11,
    counts={"fit": 256, "tune": 64, "eval": 146},
    source_role="exposed_evaluation",
)

DATASETS = {ROBIN.key: ROBIN, BEIJING.key: BEIJING}

SOURCE_ASSETS = {
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
    "alignment": FileAsset(
        "research/smallk_predictive_alignment_20261013/alignment.py",
        "84d5be3e310251987fb4f7403399b9a7b9d37f3a54410e85c68613bfeeac4537",
        "old_alignment",
    ),
    "newdata": FileAsset(
        "research/smallk_predictive_alignment_20261013/newdata.py",
        "ccef24b64c9beef0e6e31f9bc49dbad4e8e6ad57a6ce7c99973ccbce691de3bd",
        "newdata_base",
    ),
    "newdata_repair": FileAsset(
        "research/smallk_predictive_alignment_20261013/newdata_repair.py",
        "8cc75ffc4c86329939ae5155192afa03494f7bfff70fcf1ab2946c8c13f83900",
        "beijing_repair",
    ),
    "experiment": FileAsset(
        "research/forecast_generalization_study_20261013/experiment.py",
        "b2dd127caa3bfec96556b6de56eb803960c58fb92fcbfea211c6fb06ea13a9d3",
        "old_ex_preprocess",
    ),
    "common": FileAsset(
        "research/past_only_repr_diagnosis_20261012/common.py",
        "1bb0dc92a37d3bb3af525c2e72b62ab52430237ac4653086da6bf361a8c353c2",
        "old_common_load_data_original_import",
    ),
    "common_archive": FileAsset(
        "research/smallk_predictive_alignment_20261013/run_dev_v1/sources/common.py",
        "1bb0dc92a37d3bb3af525c2e72b62ab52430237ac4653086da6bf361a8c353c2",
        "old_common_load_data_archived_snapshot_same_sha_not_imported",
    ),
    "bi_runtime_declared": FileAsset(
        "research/bihash_future_prediction_20261013/runtime.py",
        "c8852f0811ce76f6ea70b3f97e014b3794e2250a94bf6f0cc02d17977b0a7d1a",
        "shared_bi_runtime_limit_safe_import_actual_only",
    ),
}

PREDICTION_ASSETS = {
    "robin": {
        "raw": FileAsset(
            ".cache/smallk_predictive_alignment_20261013/run_dev_v1/robin/RAW_BOLT.npz",
            "f936ae6aa84d6b3024cfbe0de24e79c86487c2a7ec217a8e55e52b4935eb4851",
            "canonical_raw_prediction",
        ),
        "main_92601": FileAsset(
            ".cache/smallk_predictive_alignment_20261013/run_dev_v1/robin/S1_ANCHOR_NONLINEAR_D_MAIN_w0_s92601/BOLT.npz",
            "a40312e145e17352653a5bda229bb75f3aaee027e1b4cbe8dae10297dd502778",
            "main_prediction",
        ),
        "main_92602": FileAsset(
            ".cache/smallk_predictive_alignment_20261013/run_dev_v1/robin/S1_ANCHOR_NONLINEAR_D_MAIN_w0_s92602/BOLT.npz",
            "bfd8ae6ad23647048f0eb6fc0b0bb74268ce9c2c418821258db250177aa25377",
            "main_prediction",
        ),
        "native_92601": FileAsset(
            ".cache/smallk_predictive_alignment_20261013/run_new_v1/robin/S1_ANCHOR_NONLINEAR_D_MAIN_w0_s92601_CONTROLS/NATIVE.npz",
            "9c4a831ddf1b1caafd0dfacb388845e7724dcbc8b3f6bda33326f3e0cdeafa5e",
            "native_control_prediction",
        ),
        "native_92602": FileAsset(
            ".cache/smallk_predictive_alignment_20261013/run_new_v1/robin/S1_ANCHOR_NONLINEAR_D_MAIN_w0_s92602_CONTROLS/NATIVE.npz",
            "e1aeded615e35fec1f380e0af934575d28a3e763892b6b090ed2ae5060b6c7bf",
            "native_control_prediction",
        ),
    },
    "beijing_aotizhongxin": {
        "raw": FileAsset(
            ".cache/smallk_predictive_alignment_20261013/run_beijing_repair_v1/beijing_aotizhongxin/EVAL/RAW_CONTROLS/RAW_BOLT.npz",
            "2a557027349fc8763995ade7b8150e14e776f6bd67beb73d53cd0e84d4d7bcd8",
            "canonical_raw_prediction",
        ),
        "main_92601": FileAsset(
            ".cache/smallk_predictive_alignment_20261013/run_beijing_repair_v1/beijing_aotizhongxin/EVAL/S3_ANCHOR_NONLINEAR_D_MAIN_w0_s92601/BOLT.npz",
            "5c9985ade70b792e117b4d2273f18b82e937e2f0fcb393e3cdc174fc0a9eb2b9",
            "main_prediction",
        ),
        "main_92602": FileAsset(
            ".cache/smallk_predictive_alignment_20261013/run_beijing_repair_v1/beijing_aotizhongxin/EVAL/S3_ANCHOR_NONLINEAR_D_MAIN_w0_s92602/BOLT.npz",
            "8fd04590facced935425be634ee2f6db4284a6696bf46cf00e9a816eb797ed75",
            "main_prediction",
        ),
        "native_92601": FileAsset(
            ".cache/smallk_predictive_alignment_20261013/run_beijing_repair_v1/beijing_aotizhongxin/EVAL/S3_ANCHOR_NONLINEAR_D_MAIN_w0_s92601_CONTROLS/NATIVE.npz",
            "0101e8e31d20e3885c91007d06296be0032a45ef492fd401a862e9b5756e8465",
            "native_control_prediction",
        ),
        "native_92602": FileAsset(
            ".cache/smallk_predictive_alignment_20261013/run_beijing_repair_v1/beijing_aotizhongxin/EVAL/S3_ANCHOR_NONLINEAR_D_MAIN_w0_s92602_CONTROLS/NATIVE.npz",
            "06e8c17bb32205b75c3db1dac3d6b94f63ea42c2fa6b5eb0de4fe4643d8f237a",
            "native_control_prediction",
        ),
    },
}

CHECKPOINT_ASSETS = {
    "robin_92601": FileAsset(
        ".cache/smallk_predictive_alignment_20261013/run_dev_v1/robin/S1_ANCHOR_NONLINEAR_D_MAIN_w0_s92601/u100.pt",
        "9fee31847c8546a9ffcd0237271d3e8b65b45f646004e75a3be7f07167630946",
        "main_checkpoint_file_hash_only",
    ),
    "robin_92602": FileAsset(
        ".cache/smallk_predictive_alignment_20261013/run_dev_v1/robin/S1_ANCHOR_NONLINEAR_D_MAIN_w0_s92602/u100.pt",
        "f32d2fbd3fbaad83c455a5fe21cd5201682c066a1dacfea870c2a02380f2280c",
        "main_checkpoint_file_hash_only",
    ),
    "beijing_92601": FileAsset(
        ".cache/smallk_predictive_alignment_20261013/run_beijing_repair_v1/beijing_aotizhongxin/S3_ANCHOR_NONLINEAR_D_MAIN_w0_s92601/u400.pt",
        "c585c780cd49224676546d58ef87fc79ae518acaa1f658c5b33bb9c416c9000d",
        "main_checkpoint_file_hash_only",
    ),
    "beijing_92602": FileAsset(
        ".cache/smallk_predictive_alignment_20261013/run_beijing_repair_v1/beijing_aotizhongxin/S3_ANCHOR_NONLINEAR_D_MAIN_w0_s92602/u600.pt",
        "398a0d4a2985f98feaadd0716615a1a39a47e53fee95022a70bba7209bf6d74d",
        "main_checkpoint_file_hash_only",
    ),
}

ROBIN_CONTRACT = FileAsset(
    "research/smallk_predictive_alignment_20261013/run_dev_v1/robin_contract.json",
    None,
    "legacy_robin_contract_json",
)
BEIJING_METHOD_LOCK = FileAsset(
    "research/smallk_predictive_alignment_20261013/METHOD_LOCK.json",
    "38fde1bf35b785684944c785a9b7bec7b12cec0f2608fc0c202de04d0632cc82",
    "method_lock",
)
BEIJING_REPAIR_CACHE = ROOT / ".cache" / "smallk_predictive_alignment_20261013" / "beijing_repair"


class PhaseBarrierError(RuntimeError):
    pass


class InputBridgeError(RuntimeError):
    pass


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def arrays_equal_exact(left: np.ndarray, right: np.ndarray) -> bool:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if np.issubdtype(left.dtype, np.floating):
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json_once(path: str | Path, value: Mapping[str, Any]) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(jsonable(dict(value)), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def ensure_file_hash(asset: FileAsset) -> dict[str, Any]:
    path = asset.resolved
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_path(path)
    if asset.sha256 is not None and actual != asset.sha256:
        raise InputBridgeError(f"DECLARED_SHA_MISMATCH:{asset.path}:{actual}!={asset.sha256}")
    return {
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "role": asset.role,
        "sha256": actual,
        "declared_sha256": asset.sha256,
        "bytes": path.stat().st_size,
    }


def old_window_preprocess(physical: np.ndarray, observed: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Byte-for-byte equivalent math to the old ex.preprocess for window batches."""
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


def target_from_physical_observed(physical: np.ndarray, observed: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    safe = np.where(observed, physical, mean).astype(np.float32)
    return ((safe - mean) / std).astype(np.float32)


def validate_stored_stats(mean: np.ndarray, std: np.ndarray, channels: int) -> None:
    if mean.shape != (channels,) or std.shape != (channels,):
        raise InputBridgeError("STORED_STATS_SHAPE")
    if mean.dtype != np.float32 or std.dtype != np.float32:
        raise InputBridgeError("STORED_STATS_DTYPE")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise InputBridgeError("STORED_STATS_NONFINITE_OR_NONPOSITIVE")


def validate_split_payload(split: str, payload: Mapping[str, np.ndarray], channels: int) -> None:
    history = payload["history"]
    history_physical = payload["history_physical"]
    physical_past = payload.get("physical_past", history_physical)
    raw = payload["raw_forecast"]
    target = payload["target"]
    observed = payload["observed"]
    history_observed = payload["history_observed"]
    physical_past_mask = payload["physical_past_mask"]
    origins = payload["origins"]
    raw_last = payload["raw_last"]
    if history.ndim != 3 or history.shape[1:] != (L, channels):
        raise InputBridgeError(f"HISTORY_SHAPE:{split}")
    if history_physical.shape != history.shape or physical_past.shape != history.shape:
        raise InputBridgeError(f"HISTORY_PHYSICAL_SHAPE:{split}")
    if raw.ndim != 3 or raw.shape[1:] != (H, channels) or target.shape != raw.shape or observed.shape != raw.shape:
        raise InputBridgeError(f"FORECAST_TARGET_MASK_SHAPE:{split}")
    if history_observed.shape != history.shape or raw_last.shape != (len(history), channels):
        raise InputBridgeError(f"HISTORY_MASK_LAST_SHAPE:{split}")
    if physical_past_mask.shape != history.shape:
        raise InputBridgeError(f"PHYSICAL_PAST_MASK_SHAPE:{split}")
    if origins.shape != (len(history),) or not len(origins):
        raise InputBridgeError(f"ORIGIN_SHAPE:{split}")
    if (
        history.dtype != np.float32
        or history_physical.dtype != np.float32
        or physical_past.dtype != np.float32
        or raw.dtype != np.float32
        or target.dtype != np.float32
        or raw_last.dtype != np.float32
    ):
        raise InputBridgeError(f"FLOAT32_REQUIRED:{split}")
    if observed.dtype != np.bool_ or history_observed.dtype != np.bool_ or physical_past_mask.dtype != np.bool_:
        raise InputBridgeError(f"BOOL_MASK_REQUIRED:{split}")
    if origins.dtype != np.int64:
        raise InputBridgeError(f"INT64_ORIGINS_REQUIRED:{split}")
    if not np.array_equal(physical_past_mask, history_observed):
        raise InputBridgeError(f"PHYSICAL_PAST_MASK_HISTORY_MASK_MISMATCH:{split}")
    if not np.isfinite(history).all() or not np.isfinite(raw).all() or not np.isfinite(target[observed]).all():
        raise InputBridgeError(f"NONFINITE_OBSERVED_VALUES:{split}")
    if not arrays_equal_exact(history_physical, physical_past):
        raise InputBridgeError(f"HISTORY_PHYSICAL_ALIAS_MISMATCH:{split}")
    if not np.isfinite(history_physical[physical_past_mask]).all():
        raise InputBridgeError(f"NONFINITE_OBSERVED_HISTORY_PHYSICAL:{split}")
    if not np.array_equal(raw_last, history[:, -1]):
        raise InputBridgeError(f"RAW_LAST_MISMATCH:{split}")


def validate_dataset_bundle(spec: DatasetSpec, bundle: Mapping[str, Any]) -> None:
    validate_stored_stats(bundle["mean"], bundle["std"], spec.channels)
    if tuple(bundle["splits"]) != spec.splits:
        raise InputBridgeError(f"SPLIT_ORDER:{spec.key}")
    for split in spec.splits:
        validate_split_payload(split, bundle["split_payloads"][split], spec.channels)
        expected = spec.counts[split]
        if len(bundle["split_payloads"][split]["origins"]) != expected:
            raise InputBridgeError(f"SPLIT_COUNT:{spec.key}:{split}")


def load_prediction_bundle(asset: FileAsset, spec: DatasetSpec) -> dict[str, Any]:
    path = asset.resolved
    ensure_file_hash(asset)
    with np.load(path, allow_pickle=False) as z:
        mean = np.array(z["mean"], copy=True)
        std = np.array(z["std"], copy=True)
        split_payloads: dict[str, dict[str, np.ndarray]] = {}
        for split in spec.splits:
            split_payloads[split] = {
                "prediction": np.array(z[f"{split}_prediction"], copy=True),
                "target": np.array(z[f"{split}_target"], copy=True),
                "observed": np.array(z[f"{split}_observed"], copy=True),
                "origins": np.array(z[f"{split}_origins"], copy=True),
            }
    validate_stored_stats(mean, std, spec.channels)
    for split, payload in split_payloads.items():
        pred, target, observed, origins = (payload[key] for key in ("prediction", "target", "observed", "origins"))
        if pred.dtype != np.float32 or target.dtype != np.float32 or observed.dtype != np.bool_ or origins.dtype != np.int64:
            raise InputBridgeError(f"PREDICTION_BUNDLE_DTYPE:{asset.path}:{split}")
        if pred.shape != target.shape or pred.shape != observed.shape or pred.shape[1:] != (H, spec.channels):
            raise InputBridgeError(f"PREDICTION_BUNDLE_SHAPE:{asset.path}:{split}")
        if origins.shape != (pred.shape[0],):
            raise InputBridgeError(f"PREDICTION_BUNDLE_ORIGINS:{asset.path}:{split}")
    return {"asset": asset, "mean": mean, "std": std, "split_payloads": split_payloads}


def compare_prediction_reference(
    spec: DatasetSpec,
    raw: Mapping[str, Any],
    asset: FileAsset,
) -> dict[str, Any]:
    candidate = load_prediction_bundle(asset, spec)
    checks = []
    for field in ("mean", "std"):
        if not np.array_equal(raw[field], candidate[field]):
            raise InputBridgeError(f"REFERENCE_{field.upper()}_MISMATCH:{spec.key}:{asset.path}")
        checks.append({"field": field, "sha256": array_sha256(candidate[field])})
    for split in spec.splits:
        for raw_name, candidate_name in (("target", "target"), ("observed", "observed"), ("origins", "origins")):
            left = raw["split_payloads"][split][raw_name]
            right = candidate["split_payloads"][split][candidate_name]
            if not arrays_equal_exact(left, right):
                raise InputBridgeError(f"REFERENCE_{raw_name.upper()}_MISMATCH:{spec.key}:{split}:{asset.path}")
            checks.append({"split": split, "field": raw_name, "sha256": array_sha256(right)})
    return {"asset": ensure_file_hash(asset), "checks": checks}


def build_split_from_physical(
    *,
    split: str,
    origins: np.ndarray,
    physical: np.ndarray,
    mask: np.ndarray,
    raw_prediction_payload: Mapping[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    preprocess: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray] = old_window_preprocess,
    compare_observed_target_to_source: bool = True,
) -> dict[str, np.ndarray]:
    origins = np.asarray(origins, dtype=np.int64)
    physical_windows = np.stack([physical[o - L : o] for o in origins]).astype(np.float32)
    history_observed = np.stack([mask[o - L : o] for o in origins]).astype(bool)
    history = preprocess(physical_windows, history_observed, mean, std)
    future_mask = np.stack([mask[o : o + H] for o in origins]).astype(bool)
    raw_forecast = raw_prediction_payload["prediction"]
    target = raw_prediction_payload["target"]
    observed = raw_prediction_payload["observed"]
    raw_origins = raw_prediction_payload["origins"]
    if not np.array_equal(origins, raw_origins):
        raise InputBridgeError(f"RAW_ORIGINS_MISMATCH:{split}")
    if not np.array_equal(observed, future_mask):
        raise InputBridgeError(f"RAW_TARGET_MASK_MISMATCH:{split}")
    if compare_observed_target_to_source:
        expected = target_from_physical_observed(np.stack([physical[o : o + H] for o in origins]), future_mask, mean, std)
        if not np.allclose(target[future_mask], expected[future_mask], rtol=0, atol=2e-6):
            raise InputBridgeError(f"RAW_TARGET_OBSERVED_VALUE_MISMATCH:{split}")
    payload = {
        "history": history.astype(np.float32, copy=False),
        "history_physical": physical_windows,
        "physical_past": physical_windows,
        "raw_forecast": raw_forecast.astype(np.float32, copy=False),
        "target": target.astype(np.float32, copy=False),
        "observed": observed.astype(bool, copy=False),
        "history_observed": history_observed,
        "physical_past_mask": history_observed.copy(),
        "origins": origins,
        "raw_last": history[:, -1].copy(),
    }
    validate_split_payload(split, payload, mean.shape[0])
    return payload


def build_split_from_history(
    *,
    split: str,
    history: np.ndarray,
    history_observed: np.ndarray,
    physical_past: np.ndarray,
    origins: np.ndarray,
    raw_prediction_payload: Mapping[str, np.ndarray],
    channels: int,
) -> dict[str, np.ndarray]:
    history = np.asarray(history, dtype=np.float32)
    history_observed = np.asarray(history_observed, dtype=bool)
    physical_past = np.asarray(physical_past, dtype=np.float32)
    origins = np.asarray(origins, dtype=np.int64)
    if not np.array_equal(origins, raw_prediction_payload["origins"]):
        raise InputBridgeError(f"EVAL_HISTORY_RAW_ORIGINS_MISMATCH:{split}")
    payload = {
        "history": history,
        "history_physical": physical_past,
        "physical_past": physical_past,
        "raw_forecast": raw_prediction_payload["prediction"].astype(np.float32, copy=False),
        "target": raw_prediction_payload["target"].astype(np.float32, copy=False),
        "observed": raw_prediction_payload["observed"].astype(bool, copy=False),
        "history_observed": history_observed,
        "physical_past_mask": history_observed.copy(),
        "origins": origins,
        "raw_last": history[:, -1].copy(),
    }
    validate_split_payload(split, payload, channels)
    return payload


def verify_reference_assets(spec: DatasetSpec, raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = []
    for name, asset in PREDICTION_ASSETS[spec.key].items():
        if name == "raw":
            continue
        records.append({"name": name, **compare_prediction_reference(spec, raw, asset)})
    return records


def import_module_from_path(name: str, path: Path, *, extra_sys_path: tuple[Path, ...] = ()) -> Any:
    original = list(sys.path)
    try:
        for item in reversed(extra_sys_path):
            sys.path.insert(0, str(item))
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = original


def load_old_common_and_experiment() -> tuple[Any, Any]:
    common = import_module_from_path("common", SOURCE_ASSETS["common"].resolved)
    experiment = import_module_from_path(
        "old_experiment_for_input_bridge",
        SOURCE_ASSETS["experiment"].resolved,
        extra_sys_path=(SOURCE_ASSETS["common"].resolved.parent,),
    )
    return common, experiment


def reconstruct_robin_inputs(
    *,
    raw: Mapping[str, Any],
    common_module: Any | None = None,
    experiment_module: Any | None = None,
) -> dict[str, Any]:
    spec = ROBIN
    common_module, experiment_module = (common_module, experiment_module) if common_module and experiment_module else load_old_common_and_experiment()
    data = common_module.load_data("robin", sizes=(256, 64, None))
    mean = raw["mean"]
    std = raw["std"]
    validate_stored_stats(mean, std, spec.channels)
    common_mean = np.asarray(data["mean"], dtype=np.float32).copy()
    common_std = np.asarray(data["std"], dtype=np.float32).copy()
    split_payloads = {}
    for split in spec.splits:
        if not np.array_equal(np.asarray(data["origins"][split], dtype=np.int64), raw["split_payloads"][split]["origins"]):
            raise InputBridgeError(f"ROBIN_CONTRACT_RAW_ORIGINS_MISMATCH:{split}")
        split_payloads[split] = build_split_from_physical(
            split=split,
            origins=raw["split_payloads"][split]["origins"],
            physical=np.asarray(data["physical"], dtype=np.float32),
            mask=np.asarray(data["mask"], dtype=bool),
            raw_prediction_payload=raw["split_payloads"][split],
            mean=mean,
            std=std,
            preprocess=experiment_module.preprocess,
        )
    bundle = {
        "dataset": spec.key,
        "splits": spec.splits,
        "mean": mean,
        "std": std,
        "columns": np.asarray(data["columns"]),
        "split_payloads": split_payloads,
        "common_load_mean": common_mean,
        "common_load_std": common_std,
        "lineage": {
            "source": data.get("source"),
            "source_hash": data.get("source_hash"),
            "normalization": "canonical RAW stored mean/std override; common.load_data stats retained for audit only",
            "history_preprocess": "old experiment.preprocess per-origin window reset to stored FIT mean",
        },
    }
    validate_dataset_bundle(spec, bundle)
    return bundle


def import_newdata_repair() -> Any:
    source_dir = ROOT / "research" / "smallk_predictive_alignment_20261013"
    return import_module_from_path("newdata_repair_for_input_bridge", source_dir / "newdata_repair.py", extra_sys_path=(source_dir,))


def ensure_existing_beijing_repair_payloads(repair_module: Any, contract: Mapping[str, Any]) -> dict[str, Any]:
    input_path = Path(contract["payloads"]["eval_past_input"]["path"])
    manifest_path = input_path.with_name("repair_eval_past_input_manifest.json")
    physical_payload_path = input_path.with_name("repair_eval_past_physical_payload.npz")
    if not input_path.is_file() or not manifest_path.is_file() or not physical_payload_path.is_file():
        raise InputBridgeError("BEIJING_REPAIR_PHYSICAL_PAYLOAD_NOT_PREEXISTING")
    declared_input_sha = contract["payloads"]["eval_past_input"].get("sha256")
    if declared_input_sha is not None and sha256_path(input_path) != declared_input_sha:
        raise InputBridgeError("BEIJING_EVAL_INPUT_PAYLOAD_HASH_MISMATCH")
    manifest = read_json(manifest_path)
    manifest_payload = manifest.get("payload")
    if manifest_payload is not None and Path(manifest_payload).resolve() != physical_payload_path.resolve():
        raise InputBridgeError("BEIJING_REPAIR_PHYSICAL_PAYLOAD_PATH_MISMATCH")
    if manifest.get("no_eval_future_target_payload_opened") is False:
        raise InputBridgeError("BEIJING_REPAIR_MANIFEST_EVAL_TARGET_OPENED")
    actual = repair_module.base.sha256_path(physical_payload_path)
    if actual != manifest.get("history_physical_sha256"):
        raise InputBridgeError("BEIJING_REPAIR_PHYSICAL_PAYLOAD_HASH_MISMATCH")
    return {
        "manifest": {
            "path": str(manifest_path.relative_to(ROOT)) if manifest_path.is_relative_to(ROOT) else str(manifest_path),
            "sha256": sha256_path(manifest_path),
        },
        "physical_payload": {
            "path": str(physical_payload_path.relative_to(ROOT)) if physical_payload_path.is_relative_to(ROOT) else str(physical_payload_path),
            "sha256": actual,
        },
    }


def reconstruct_beijing_inputs(
    *,
    raw: Mapping[str, Any],
    repair_module: Any | None = None,
    experiment_module: Any | None = None,
) -> dict[str, Any]:
    spec = BEIJING
    if repair_module is None:
        repair_module = import_newdata_repair()
    if experiment_module is None:
        _common, experiment_module = load_old_common_and_experiment()
    method_lock = BEIJING_METHOD_LOCK.resolved
    contract = repair_module.get_contract(spec.key, method_lock, cache_root=BEIJING_REPAIR_CACHE)
    repair_lineage = ensure_existing_beijing_repair_payloads(repair_module, contract)
    training = repair_module.load_training(spec.key, method_lock, cache_root=BEIJING_REPAIR_CACHE)
    evaluation = repair_module.load_evaluation_inputs(spec.key, method_lock, cache_root=BEIJING_REPAIR_CACHE)
    mean = raw["mean"]
    std = raw["std"]
    for source_name, source in (("training", training), ("evaluation", evaluation)):
        if not np.array_equal(np.asarray(source["mean"], dtype=np.float32), mean):
            raise InputBridgeError(f"BEIJING_{source_name.upper()}_MEAN_DIFFERS_FROM_RAW")
        if not np.array_equal(np.asarray(source["std"], dtype=np.float32), std):
            raise InputBridgeError(f"BEIJING_{source_name.upper()}_STD_DIFFERS_FROM_RAW")
    split_payloads = {}
    for split in ("fit", "tune"):
        split_payloads[split] = build_split_from_physical(
            split=split,
            origins=raw["split_payloads"][split]["origins"],
            physical=np.asarray(training["physical"], dtype=np.float32),
            mask=np.asarray(training["mask"], dtype=bool),
            raw_prediction_payload=raw["split_payloads"][split],
            mean=mean,
            std=std,
            preprocess=experiment_module.preprocess,
        )
    split_payloads["eval"] = build_split_from_history(
        split="eval",
        history=evaluation["history"],
        history_observed=evaluation["history_observed"],
        physical_past=evaluation["history_physical"],
        origins=evaluation["origins"],
        raw_prediction_payload=raw["split_payloads"]["eval"],
        channels=spec.channels,
    )
    bundle = {
        "dataset": spec.key,
        "splits": spec.splits,
        "mean": mean,
        "std": std,
        "columns": np.asarray(training["columns"]),
        "split_payloads": split_payloads,
        "common_load_mean": np.asarray(training["mean"], dtype=np.float32).copy(),
        "common_load_std": np.asarray(training["std"], dtype=np.float32).copy(),
        "lineage": {
            "contract_sha256": contract["contract_sha256"],
            "fit_tune_payload_sha256": training["fit_tune_payload_sha256"],
            "eval_input_payload_sha256": evaluation["eval_input_payload_sha256"],
            "repair_history_physical_payload_sha256": evaluation["repair_history_physical_payload_sha256"],
            "repair_payload_guard": repair_lineage,
            "eval_target_source": "canonical exposed RAW prediction NPZ only; open_evaluation not called",
            "history_preprocess": "FIT/TUNE old experiment.preprocess; EVAL stored repaired normalized history",
        },
    }
    validate_dataset_bundle(spec, bundle)
    return bundle


def bundle_array_records(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {"field": "mean", "shape": list(bundle["mean"].shape), "dtype": str(bundle["mean"].dtype), "sha256": array_sha256(bundle["mean"])},
        {"field": "std", "shape": list(bundle["std"].shape), "dtype": str(bundle["std"].dtype), "sha256": array_sha256(bundle["std"])},
        {
            "field": "common_load_mean",
            "shape": list(bundle["common_load_mean"].shape),
            "dtype": str(bundle["common_load_mean"].dtype),
            "sha256": array_sha256(bundle["common_load_mean"]),
        },
        {
            "field": "common_load_std",
            "shape": list(bundle["common_load_std"].shape),
            "dtype": str(bundle["common_load_std"].dtype),
            "sha256": array_sha256(bundle["common_load_std"]),
        },
    ]
    for split, payload in bundle["split_payloads"].items():
        for field, value in payload.items():
            rows.append({"split": split, "field": field, "shape": list(value.shape), "dtype": str(value.dtype), "sha256": array_sha256(value)})
    return rows


def save_split_npz(output: Path, dataset: str, split: str, bundle: Mapping[str, Any]) -> dict[str, Any]:
    payload = bundle["split_payloads"][split]
    columns = np.asarray(bundle["columns"])
    arrays = {
        "history": payload["history"],
        "history_physical": payload["history_physical"],
        "physical_past": payload["physical_past"],
        "raw_forecast": payload["raw_forecast"],
        "target": payload["target"],
        "observed": payload["observed"],
        "history_observed": payload["history_observed"],
        "physical_past_mask": payload["physical_past_mask"],
        "origins": payload["origins"],
        "raw_last": payload["raw_last"],
        "mean": bundle["mean"],
        "std": bundle["std"],
        "columns": columns,
        "common_load_mean": bundle["common_load_mean"],
        "common_load_std": bundle["common_load_std"],
    }
    path = output / f"{dataset}_{split}_input.npz"
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    return {
        "dataset": dataset,
        "split": split,
        "path": str(path.relative_to(ROOT)) if path.resolve().is_relative_to(ROOT) else str(path),
        "sha256": sha256_path(path),
        "bytes": path.stat().st_size,
        "arrays": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": array_sha256(value)}
            for key, value in arrays.items()
        },
    }


def load_followup_limit_class() -> type:
    module = import_module_from_path("followup_runtime_for_input_bridge", HERE / "followup_runtime.py")
    return module.FollowupLimit


def protected_test_paths() -> list[Path]:
    paths = []
    for rel in (
        ".cache/tsfm_peft_practical_controls_v7_20260928/data/jena_test.npz",
        ".cache/tsfm_peft_level_confirmation_v6_20260928/data/test.npz",
    ):
        paths.append((ROOT / rel).resolve())
    return paths


def install_protected_test_guard() -> list[str]:
    protected = set(protected_test_paths())

    def audit(event: str, args: tuple[Any, ...]) -> None:
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if path in protected or path.name.lower() in {"test.npz", "jena_test.npz"}:
                raise PermissionError("PROTECTED_TEST_ACCESS_DENIED")

    sys.addaudithook(audit)
    return [str(path) for path in sorted(protected)]


def collect_status_tokens(value: Any, key_hint: str = "") -> list[str]:
    tokens: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            hint = f"{key_hint}.{key}" if key_hint else str(key)
            if str(key).lower() in {"status", "overall_status", "verdict", "result", "state"}:
                tokens.append(str(item).upper())
            tokens.extend(collect_status_tokens(item, hint))
    elif isinstance(value, list):
        for item in value:
            tokens.extend(collect_status_tokens(item, key_hint))
    return tokens


def _manifest_file_sha(manifest: Mapping[str, Any], manifest_path: Path, target_path: Path) -> str:
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise PhaseBarrierError("BI_PHASE_BARRIER_MANIFEST_FILES_REQUIRED")
    candidates = [target_path.name]
    try:
        candidates.append(target_path.resolve().relative_to(manifest_path.resolve().parent).as_posix())
    except ValueError:
        pass
    try:
        candidates.append(target_path.resolve().relative_to(ROOT).as_posix())
    except ValueError:
        pass
    for key in candidates:
        value = files.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping) and isinstance(value.get("sha256"), str):
            return str(value["sha256"])
    raise PhaseBarrierError(f"BI_PHASE_BARRIER_MANIFEST_FILE_MISSING:{target_path.name}")


def verify_phase_barrier(audit_path: Path, report_path: Path, manifest_path: Path) -> dict[str, Any]:
    for path in (audit_path, report_path, manifest_path):
        if not path.is_file():
            raise PhaseBarrierError(f"BI_PHASE_BARRIER_MISSING:{path}")
    audit = read_json(audit_path)
    manifest = read_json(manifest_path)
    if audit.get("whole_experiment_complete") is not True:
        raise PhaseBarrierError("BI_PHASE_BARRIER_WHOLE_EXPERIMENT_INCOMPLETE")
    if int(audit.get("failed_count", 0)) != 0:
        raise PhaseBarrierError("BI_PHASE_BARRIER_FAILED_COUNT_NONZERO")
    status = str(audit.get("status", "")).upper()
    if "PASS" not in status:
        raise PhaseBarrierError(f"BI_PHASE_BARRIER_AUDIT_STATUS_NOT_PASS:{status}")
    checks = audit.get("checks", [])
    if isinstance(checks, list) and any(item.get("passed") is not True for item in checks if isinstance(item, Mapping)):
        raise PhaseBarrierError("BI_PHASE_BARRIER_AUDIT_CHECK_FAILED")
    manifest_status = str(manifest.get("status", "")).upper()
    if manifest_status != "GENERATED_FROM_ACTUAL_TERMINAL_ARTIFACTS":
        raise PhaseBarrierError(f"BI_PHASE_BARRIER_MANIFEST_STATUS:{manifest_status}")
    audit_sha = sha256_path(audit_path)
    report_sha = sha256_path(report_path)
    manifest_audit_sha = _manifest_file_sha(manifest, manifest_path, audit_path)
    manifest_report_sha = _manifest_file_sha(manifest, manifest_path, report_path)
    if manifest_audit_sha != audit_sha:
        raise PhaseBarrierError(f"BI_PHASE_BARRIER_MANIFEST_AUDIT_SHA_MISMATCH:{manifest_audit_sha}!={audit_sha}")
    if manifest_report_sha != report_sha:
        raise PhaseBarrierError(f"BI_PHASE_BARRIER_MANIFEST_REPORT_SHA_MISMATCH:{manifest_report_sha}!={report_sha}")
    return {
        "audit": {"path": str(audit_path), "sha256": audit_sha},
        "report": {"path": str(report_path), "sha256": report_sha},
        "manifest": {"path": str(manifest_path), "sha256": sha256_path(manifest_path)},
        "manifest_audit_sha256": manifest_audit_sha,
        "manifest_report_sha256": manifest_report_sha,
        "audit_status": audit.get("status"),
        "whole_experiment_complete": True,
        "failed_count": 0,
    }


def verify_declared_assets_for_actual() -> dict[str, Any]:
    records = {
        "source": {name: ensure_file_hash(asset) for name, asset in SOURCE_ASSETS.items() if asset.sha256 is not None},
        "predictions": {
            dataset: {name: ensure_file_hash(asset) for name, asset in assets.items()}
            for dataset, assets in PREDICTION_ASSETS.items()
        },
        "checkpoints_file_hash_only": {name: ensure_file_hash(asset) for name, asset in CHECKPOINT_ASSETS.items()},
        "method_lock": ensure_file_hash(BEIJING_METHOD_LOCK),
        "model_snapshot_state_hash": "declared in ASSET_MAP_PREPARATION_KO; not torch-loaded by this input bridge",
    }
    return records


def execute_after_bi(output: Path, barrier: Mapping[str, Any]) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if not output.is_relative_to(CACHE):
        raise InputBridgeError("OUTPUT_MUST_BE_FRESH_FOLLOWUP_CACHE")
    output.mkdir(parents=True, exist_ok=False)
    limit = None
    limit_class = None
    result: dict[str, Any] = {
        "status": "RUNNING",
        "actual_status": {
            "old_npz_opened": False,
            "old_checkpoint_file_hash_done": False,
            "old_checkpoint_torch_load_done": False,
            "model_loaded": False,
            "gpu_executed": False,
            "first_eval_open_called": False,
            "protected_test_opened": False,
        },
        "phase_barrier": dict(barrier),
        "datasets": {},
        "artifacts": [],
    }
    try:
        limit_class = load_followup_limit_class()
        limit = limit_class("input_bridge", False, followup_roots=(HERE, CACHE, output))
        result["resource_checks"] = [limit.check()]
        result["protected_test_guarded"] = install_protected_test_guard()
        result["declared_asset_hashes"] = verify_declared_assets_for_actual()
        result["actual_status"]["old_checkpoint_file_hash_done"] = True
        bundles = {}
        for spec in (ROBIN, BEIJING):
            raw = load_prediction_bundle(PREDICTION_ASSETS[spec.key]["raw"], spec)
            result["actual_status"]["old_npz_opened"] = True
            references = verify_reference_assets(spec, raw)
            bundle = reconstruct_robin_inputs(raw=raw) if spec.key == "robin" else reconstruct_beijing_inputs(raw=raw)
            bundles[spec.key] = bundle
            artifacts = [save_split_npz(output, spec.output_key, split, bundle) for split in spec.splits]
            result["artifacts"].extend(artifacts)
            result["datasets"][spec.key] = {
                "split_roles": {
                    split: (
                        "training"
                        if split == "fit"
                        else "model_selection"
                        if split == "tune"
                        else spec.source_role
                    )
                    for split in spec.splits
                },
                "counts": {split: int(len(bundle["split_payloads"][split]["origins"])) for split in spec.splits},
                "channels": spec.channels,
                "array_sha256": bundle_array_records(bundle),
                "lineage": bundle["lineage"],
                "reference_equivalence": references,
                "storage_scale": "normalized",
                "normalization_origin": "canonical RAW stored mean/std; not refit by bridge",
                "strict_k_only_compression": False,
            }
            result["resource_checks"].append(limit.check())
        result["status"] = "COMPLETE"
        result["actual_status"]["RESULT_written"] = True
        return result
    except Exception as error:
        result.update(status="FAILED_PRESERVED", failure={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        status = result.get("status", "FAILED_PRESERVED")
        if limit is None:
            runtime_module = sys.modules.get(getattr(limit_class, "__module__", "")) if limit_class is not None else None
            constructor_terminal = getattr(runtime_module, "LAST_CONSTRUCTOR_TERMINAL", None) if runtime_module is not None else None
            result["resource_terminal"] = constructor_terminal or {
                "terminal_status": "FAILED_PRESERVED",
                "requested_status": status,
                "constructor_failure_scope": (
                    "FollowupLimit was not assigned; rt.Limit construction may have failed before shared receipt ownership"
                ),
                "failure": jsonable(result.get("failure")),
            }
        else:
            result["resource_terminal"] = limit.close(status, None if status == "COMPLETE" else jsonable(result.get("failure")))
        terminal_status = result["resource_terminal"].get("terminal_status") or result["resource_terminal"].get("status")
        result["resource_terminal_status"] = terminal_status
        if result.get("status") == "COMPLETE" and terminal_status and terminal_status != "COMPLETE":
            result.update(
                status=terminal_status,
                failure={
                    "type": "FollowupLimitTerminalStatus",
                    "message": f"resource terminal returned {terminal_status}",
                },
            )
        with (output / "FOLLOWUP_RUNTIME_RECEIPT.json").open("x", encoding="utf-8") as handle:
            json.dump(jsonable(result["resource_terminal"]), handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        with (output / "RESULT.json").open("x", encoding="utf-8") as handle:
            json.dump(jsonable(result), handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")


def describe() -> dict[str, Any]:
    return {
        "status": "PREPARATION_ONLY",
        "actual_status": {
            "old_npz_opened": False,
            "old_checkpoint_read_or_loaded": False,
            "model_loaded": False,
            "target_opened_by_open_evaluation": False,
            "gpu_executed": False,
            "actual_hash_done": False,
        },
        "domain": "time-series forecasting old-condition input reconstruction",
        "downstream_decision": "D1/D2 GPU probes should compare old small-K conditions against exact RAW/MAIN/NATIVE input contracts after Bi completion.",
        "leakage_controls": [
            "phase barrier before any old NPZ/checkpoint hash/read",
            "protected TEST audit hook",
            "no newdata_repair.open_evaluation call",
            "Beijing EVAL target/mask only from already exposed canonical RAW prediction NPZ",
            "canonical RAW stored mean/std reused exactly; no bridge refit",
        ],
        "entrypoint": (
            "prepare_inputs.py --execute-after-bi --bi-audit <completed_audit.json> "
            "--bi-report <completed_report.md> --bi-manifest <completed_manifest.json> "
            "--output-dir .cache/smallk_failure_followup_20261014/input_bridge/<fresh_name>"
        ),
        "runtime_api": "followup_runtime.FollowupLimit(tag, gpu)",
        "outputs": "Per-dataset split NPZ files plus RESULT.json and FOLLOWUP_RUNTIME_RECEIPT.json in a fresh follow-up cache directory.",
        "split_npz_schema": {
            "history": "normalized float32[N,512,C]",
            "history_physical": "raw physical float32[N,512,C] before old causal preprocessing; NaN where missing",
            "physical_past": "alias of history_physical for physical-past consumers",
            "history_observed": "bool[N,512,C]",
            "physical_past_mask": "alias mask matching history_observed",
            "raw_forecast": "normalized canonical RAW prediction float32[N,48,C]",
            "target": "normalized canonical target float32[N,48,C], masked entries preserved exactly",
            "observed": "bool[N,48,C]",
            "origins": "int64[N]",
            "raw_last": "history[:, -1]",
            "mean/std": "canonical RAW stored float32[C]",
        },
        "datasets": {
            name: {
                "splits": spec.splits,
                "counts": spec.counts,
                "channels": spec.channels,
                "evaluation_role": spec.source_role,
            }
            for name, spec in DATASETS.items()
        },
    }


def default_output_dir() -> Path:
    return CACHE / f"old_condition_inputs_{os.getpid()}_{time.time_ns()}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--execute-after-bi", action="store_true")
    parser.add_argument("--bi-audit")
    parser.add_argument("--bi-report")
    parser.add_argument("--bi-manifest")
    parser.add_argument("--output-dir")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.describe:
        print(json.dumps(describe(), ensure_ascii=False, indent=2))
        return 0
    if not (args.bi_audit and args.bi_report and args.bi_manifest):
        raise PhaseBarrierError("BI_PHASE_BARRIER_REFS_REQUIRED")
    barrier = verify_phase_barrier(Path(args.bi_audit), Path(args.bi_report), Path(args.bi_manifest))
    output = Path(args.output_dir) if args.output_dir else default_output_dir()
    result = execute_after_bi(output, barrier)
    print(json.dumps({"status": result["status"], "output": str(output.resolve())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
