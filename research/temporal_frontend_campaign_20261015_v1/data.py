"""Exact-allowlist FIT assets and past-only windows for this campaign.

Importing this module and ``describe_assets`` never open data arrays. The root
runner calls ``load_dataset`` explicitly and records its returned receipt.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
L, H = 512, 48

ASSETS = {
    "beijing": {
        "path": ".cache/smallk_predictive_alignment_20261013/beijing_repair/raw_dataset/beijing_aotizhongxin/b04da438b2f331ac/fit_tune_payload.npz",
        "sha256": "78063851dd3e65512e3e3ced22210839697553f2d652fc0be9513736af598277",
        "bytes": 1524625,
        "contract": ".cache/smallk_predictive_alignment_20261013/beijing_repair/raw_dataset/beijing_aotizhongxin/b04da438b2f331ac/contract.json",
        "contract_sha256": "9116703c22b3dea317cb1ff750fd45d05c5d6e4a226946a41334e0794484849f",
        "permitted_bounds": (0, 21038),
        "original_tune_bounds": (21038, 28051),
        "channels": 11,
        "interval_seconds": 3600,
        "period_from_existing_metadata": 24,
        "small_k": 2,
        "kind": "physical",
        "ancestry": "Stored physical FP32 observations and original mask, reparsed Aotizhongxin station; new statistics are fit on this fold's A only.",
    },
    "jena": {
        "path": ".cache/tsfm_peft_practical_controls_v7_20260928/data/jena_trainval.npz",
        "sha256": "29cfae52d7cd7128e551ef85da11138ed4d3f7b3ccf27af359c4fc1dedf4a933",
        "bytes": 1561064,
        "contract": "research/level_matched_learned_ed_v18_20261006/data_contract_jena.json",
        "contract_sha256": "678bd4cbd63265290c0eeed8a78b09545a05219b666fd3b818682ce385906820",
        "permitted_bounds": (0, 20965),
        "original_tune_bounds": (20965, 26207),
        "channels": 21,
        "interval_seconds": 600,
        "period_from_existing_metadata": 144,
        "small_k": 5,
        "kind": "legacy_normalized",
        "ancestry": "FP32 inverse x*legacy_std+legacy_mean of safe TRAIN/VAL archive, originally normalized on TRAIN[0,26207); rounding and whole-TRAIN normalization ancestry remain. This is not lossless raw A-only recovery.",
    },
    "robin": {
        "path": ".cache/tsfm_peft_level_confirmation_v6_20260928/data/trainval.npz",
        "sha256": "04523b70fa2df7b37514ab0f12e0cb25048c35fd120a11d4c6ab56e0ae464e14",
        "bytes": 111474,
        "contract": "research/level_matched_learned_ed_v18_20261006/data_contract_robin.json",
        "contract_sha256": "b7c02be4f69617e1c60dd874ef6337facde1a4165f8cbd30233aefb0601a85b2",
        "permitted_bounds": (528, 2121),
        "original_tune_bounds": (2121, 2520),
        "channels": 17,
        "interval_seconds": 3600,
        "period_from_existing_metadata": 24,
        "small_k": 4,
        "kind": "legacy_normalized",
        "ancestry": "FP32 inverse x*legacy_std+legacy_mean of safe TRAIN/VAL archive, originally normalized on TRAIN[528,2520); rounding and whole-TRAIN normalization ancestry remain. This is not lossless raw A-only recovery.",
    },
}

FOLD_FRACTIONS = {"primary": (0.6, 0.8), "secondary": (0.5, 0.7)}
_GUARD_INSTALLED = False


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe_assets() -> dict:
    """Report source metadata and pinned hashes, without reading archive bytes."""
    result = {}
    for name, asset in ASSETS.items():
        path = ROOT / asset["path"]
        contract = ROOT / asset["contract"]
        result[name] = {
            **asset,
            "absolute_path": str(path.resolve()),
            "exists": path.is_file(),
            "actual_bytes": path.stat().st_size if path.is_file() else None,
            "contract_exists": contract.is_file(),
            "data_hash_status": "PINNED_METADATA_NOT_REHASHED_BY_DESCRIBE",
            "algorithmically_held_out_this_run": "C scores deferred until root selection lock",
            "historically_unseen": False,
        }
    return result


def install_data_access_guard() -> None:
    """Reject data-file reads outside exact source allowlist/new campaign paths.

    The audit hook remains active for the process. It covers Python file opens;
    it is not an OS sandbox and does not prove absence of native-library reads.
    """
    global _GUARD_INSTALLED
    if _GUARD_INSTALLED:
        return
    source_allowlist = {(ROOT / spec["path"]).resolve() for spec in ASSETS.values()}
    new_paths = (HERE.resolve(), (ROOT / ".cache" / HERE.name).resolve())
    data_suffixes = {".npz", ".npy", ".csv", ".zip", ".parquet", ".arrow", ".feather"}

    def audit(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.suffix.lower() not in data_suffixes:
            return
        if path in source_allowlist or any(path.is_relative_to(parent) for parent in new_paths):
            return
        raise PermissionError(f"Campaign data allowlist rejected: {path}")

    sys.addaudithook(audit)
    _GUARD_INSTALLED = True


def _read_contract(spec: dict) -> dict:
    path = ROOT / spec["contract"]
    if sha256_file(path) != spec["contract_sha256"]:
        raise ValueError(f"Pinned contract changed: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_dataset(name: str) -> dict:
    """Load only the permitted FIT slice from one pinned safe archive.

    NPZ members decompress in full before slicing. Consequently the returned
    scope is FIT-only, but physical deserialization is not row-isolated: the
    safe archive also contains old TUNE/VAL rows. Those rows never reach this
    return value, new statistics, training, or selection.
    """
    if name == "beijing_aotizhongxin":
        name = "beijing"
    spec = ASSETS[name]
    contract = _read_contract(spec)
    install_data_access_guard()
    path = (ROOT / spec["path"]).resolve()
    if not path.is_file() or path.stat().st_size != spec["bytes"]:
        raise FileNotFoundError(f"Pinned safe asset absent/size changed: {path}")
    actual_sha = sha256_file(path)
    if actual_sha != spec["sha256"]:
        raise ValueError(f"Pinned safe asset hash changed: {path}")
    lo, hi = spec["permitted_bounds"]
    with np.load(path, allow_pickle=False) as archive:
        mask_key = "mask" if spec["kind"] == "physical" else "finite"
        observed = archive[mask_key][lo:hi].astype(bool, copy=True)
        if spec["kind"] == "physical":
            physical = archive["physical"][lo:hi].astype(np.float32, copy=True)
        else:
            # Slice before inverse conversion, avoiding computation on old TUNE/VAL.
            x = archive["x"][lo:hi].astype(np.float32, copy=True)
            legacy_mean = archive["mean"].astype(np.float32, copy=True)
            legacy_std = archive["std"].astype(np.float32, copy=True)
            physical = (x * legacy_std + legacy_mean).astype(np.float32)
        times = archive["times"][lo:hi].copy()
        columns = archive["columns"].tolist()
    if physical.shape != (hi - lo, spec["channels"]) or observed.shape != physical.shape:
        raise ValueError("Safe asset does not match pinned FIT shape")
    if len(times) != len(physical) or len(columns) != spec["channels"]:
        raise ValueError("Timestamp/channel metadata shape mismatch")
    if not np.isfinite(physical[observed]).all():
        raise ValueError("Original observed mask marks a non-finite physical target")
    physical[~observed] = np.nan
    if times.dtype.kind in {"U", "S"}:
        times = times.astype("datetime64[ns]").astype(np.int64)
    else:
        times = times.astype(np.int64)
    if len(times) > 1 and not np.all(np.diff(times) == spec["interval_seconds"] * 1_000_000_000):
        raise ValueError("Safe asset timestamps differ from the pinned uniform grid")
    expected_columns = contract.get("columns", contract.get("source", {}).get("parse", {}).get("columns"))
    if expected_columns is not None and columns != expected_columns:
        raise ValueError("Pinned channel order differs from archive")
    if spec["kind"] == "physical":
        if tuple(contract["bounds"]["fit"]) != (lo, hi):
            raise ValueError("Pinned Beijing FIT boundary differs from contract")
    else:
        original_train = contract["bounds"]["train"]
        original_fit_hi = original_train[0] + int(0.8 * (original_train[1] - original_train[0]))
        if (original_train[0], original_fit_hi) != (lo, hi):
            raise ValueError("Legacy original FIT calculation differs from pinned boundary")
    return {
        "dataset": name,
        "physical": physical,
        "observed": observed,
        "times": times,
        "indices": np.arange(lo, hi, dtype=np.int64),
        "columns": columns,
        "interval_seconds": spec["interval_seconds"],
        "period_from_existing_metadata": spec["period_from_existing_metadata"],
        "small_k": spec["small_k"],
        "permitted_bounds": (lo, hi),
        "original_tune_bounds": spec["original_tune_bounds"],
        "ancestry": spec["ancestry"],
        "source": str(path),
        "source_sha256": actual_sha,
        "contract_sha256": spec["contract_sha256"],
        "historically_unseen": False,
        "archive_deserialization_scope": "Whole required NPZ members (including old TUNE/VAL) before immediate permitted-FIT slicing; not physically row-isolated",
        "statistical_and_training_scope": "Returned permitted FIT only; fold-specific A statistics freshly fitted",
    }


def _even_stride_with_endpoint(lo: int, hi: int, stride: int) -> np.ndarray:
    if lo > hi:
        return np.empty(0, dtype=np.int64)
    values = np.arange(lo, hi + 1, stride, dtype=np.int64)
    return values if values[-1] == hi else np.append(values, np.int64(hi))


def make_fold(data_or_rows, fold="primary", length=L, horizon=H, eval_stride=H) -> dict:
    """Predeclared 60/20/20 or 50/20/30 splits, with target-disjoint origins.

    Origins/bounds use local rows in the returned FIT arrays. Absolute metadata
    retains the source archive offset (528 for Robin). B/C histories may use
    already observed earlier split rows; all future targets remain in split.
    """
    if isinstance(data_or_rows, dict):
        rows = len(data_or_rows["physical"])
        offset = int(data_or_rows.get("permitted_bounds", (0, rows))[0])
    else:
        rows, offset = int(data_or_rows), 0
    if min(length, horizon, eval_stride) < 1 or rows < 1:
        raise ValueError("Positive rows, length, horizon, and stride required")
    af, bf = FOLD_FRACTIONS[fold]
    a_end, b_end = int(rows * af), int(rows * bf)
    bounds = {"A": (0, a_end), "B": (a_end, b_end), "C": (b_end, rows)}
    origins = {}
    for split, (start, stop) in bounds.items():
        lower, upper = max(start, length), stop - horizon
        origins[split] = (
            np.arange(lower, upper + 1, dtype=np.int64)
            if split == "A"
            else _even_stride_with_endpoint(lower, upper, eval_stride)
        )
    receipts = {}
    for previous, following in (("A", "B"), ("B", "C")):
        before, after = origins[previous], origins[following]
        valid = not len(before) or not len(after) or int(before[-1] + horizon - 1) < int(after[0])
        receipts[f"{previous}_target_before_{following}_origin"] = bool(valid)
        if not valid:
            raise AssertionError("Target-purge boundary violated")
    return {
        "fold": fold,
        "fractions": (af, bf - af, 1 - bf),
        "bounds": bounds,
        "absolute_bounds": {key: (lo + offset, hi + offset) for key, (lo, hi) in bounds.items()},
        "origins": origins,
        "absolute_origins": {key: value + offset for key, value in origins.items()},
        "length": int(length),
        "horizon": int(horizon),
        "eval_stride": int(eval_stride),
        "purge_receipt": receipts,
        "algorithmically_held_out_this_run": "C not used for fit/checkpoint/design selection; enforced by root controller selection lock",
        "historically_unseen": False,
        "origin_rules": "A every legal origin; B/C H-stride plus each split's last legal origin (last pair may overlap); original FIT-local past only",
    }


def common_origins(fold: dict, lengths=(512, 1024, 2048)) -> dict:
    """Intersection of legal histories, leaving the original fixed target cohort."""
    maximum = max(int(value) for value in lengths)
    return {split: values[values >= maximum].copy() for split, values in fold["origins"].items()}


def fit_statistics(data: dict, bounds: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Physical true-observed population moments on the explicit A row range."""
    lo, hi = bounds
    physical = np.asarray(data["physical"])[lo:hi]
    observed = np.asarray(data["observed"], dtype=bool)[lo:hi]
    if lo < 0 or hi > len(data["physical"]) or lo >= hi:
        raise ValueError("Statistics bounds outside permitted FIT")
    count = observed.sum(axis=0)
    if np.any(count == 0) or not np.isfinite(physical[observed]).all():
        raise ValueError("A has no finite true observation in a channel")
    values = np.where(observed, physical, 0).astype(np.float64)
    mean64 = values.sum(axis=0) / count
    centered = np.where(observed, physical.astype(np.float64) - mean64, 0)
    std64 = np.sqrt((centered * centered).sum(axis=0) / count)
    mean = mean64.astype(np.float32)
    std = np.where(std64 > 1e-6, std64, 1).astype(np.float32)
    return mean, std, count


def causal_forward_fill(physical: np.ndarray, observed: np.ndarray, initial: np.ndarray) -> np.ndarray:
    """Fill from prior observed rows, resetting initial state for each window."""
    physical = np.asarray(physical, dtype=np.float32)
    observed = np.asarray(observed, dtype=bool)
    if physical.shape != observed.shape or physical.ndim != 2:
        raise ValueError("Physical/mask must be matching [time, channel] arrays")
    if not np.isfinite(physical[observed]).all() or not np.isfinite(initial).all():
        raise ValueError("Non-finite true observation or initial state")
    last = np.asarray(initial, dtype=np.float32).copy()
    result = np.empty_like(physical)
    for row in range(len(physical)):
        last[observed[row]] = physical[row, observed[row]]
        result[row] = last
    return result


def history_window(data: dict, origin: int, mean: np.ndarray, std: np.ndarray, length=L):
    if origin - length < 0 or origin > len(data["physical"]):
        raise ValueError("History outside permitted FIT or beyond its origin")
    mask = data["observed"][origin - length:origin].copy()
    physical = data["physical"][origin - length:origin]
    filled = causal_forward_fill(physical, mask, mean)
    return ((filled - mean) / std).astype(np.float32), mask


def target_window(data: dict, origin: int, mean: np.ndarray, std: np.ndarray, horizon=H):
    if origin < 0 or origin + horizon > len(data["physical"]):
        raise ValueError("Target outside permitted FIT")
    mask = data["observed"][origin:origin + horizon].copy()
    physical = data["physical"][origin:origin + horizon]
    # Numeric placeholders are excluded by the original mask in every loss/metric.
    safe = np.where(mask, physical, mean)
    return ((safe - mean) / std).astype(np.float32), mask


def origin_exposure(data: dict, origins: np.ndarray, length=L, horizon=H) -> dict:
    """Count actual unique row exposure and target repetitions, independently."""
    origins = np.asarray(origins, dtype=np.int64)
    rows = len(data["physical"])
    input_diff = np.zeros(rows + 1, dtype=np.int64)
    target_diff = np.zeros(rows + 1, dtype=np.int64)
    if len(origins):
        if np.any(origins < length) or np.any(origins + horizon > rows):
            raise ValueError("Exposure origins outside permitted FIT")
        np.add.at(input_diff, origins - length, 1)
        np.add.at(input_diff, origins, -1)
        np.add.at(target_diff, origins, 1)
        np.add.at(target_diff, origins + horizon, -1)
    inputs, targets = input_diff.cumsum()[:-1], target_diff.cumsum()[:-1]
    observed = data["observed"]
    return {
        "origins": int(len(origins)),
        "unique_input_times": int((inputs > 0).sum()),
        "unique_target_times": int((targets > 0).sum()),
        "target_max_repetition": int(targets.max(initial=0)),
        "target_observed_with_repetition": (targets[:, None] * observed).sum(axis=0).tolist(),
        "target_observed_unique": observed[targets > 0].sum(axis=0).tolist(),
        "input_row_minmax": None if not len(origins) else [int(origins.min() - length), int(origins.max() - 1)],
        "target_row_minmax": None if not len(origins) else [int(origins.min()), int(origins.max() + horizon - 1)],
    }
