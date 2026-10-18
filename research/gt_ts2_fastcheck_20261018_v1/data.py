"""Read-only Jena TRAIN/VAL data and origin-local causal preprocessing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


H = 48
TRAIN_END = 26207
VAL_END = 35135
EPS = 1e-6
ARCHIVE = ".cache/tsfm_peft_practical_controls_v7_20260928/data/jena_trainval.npz"
ARCHIVE_SHA256 = "29cfae52d7cd7128e551ef85da11138ed4d3f7b3ccf27af359c4fc1dedf4a933"
LEGACY_CONTRACT_SHA256 = "742f7189e27439d282bb4ccc0f7a72c3088146e393f8e12c7e1b72af16fd5d8e"
HISTORICAL_RAW_PHYSICAL_SHA256 = "0c4932c7a35a01fc955db8d9d6add8b575baaac0ce791dc9ef42eb0b116be209"
COLUMNS = (
    "p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)", "rh (%)",
    "VPmax (mbar)", "VPact (mbar)", "VPdef (mbar)", "sh (g/kg)",
    "H2OC (mmol/mol)", "rho (g/m**3)", "wv (m/s)", "max. wv (m/s)",
    "wd (deg)", "rain (mm)", "raining (s)", "SWDR (W/m²)",
    "PAR (µmol/m²/s)", "max. PAR (µmol/m²/s)", "Tlog (degC)", "CO2 (ppm)",
)
LEGACY_HASHES = {
    "legacy_x": "1681c7d0aa3deb7ea784921e34176440fe6bff8c71f2810e0a4707e551a539d3",
    "mask": "43354f82269ee9d72680b6fe5ec06ab7e331762eef9cdbc631ae99007932423f",
    "mean": "0b2d6ed2ea90ef8d326f06a572cc07fea91f9a46a4f386d72002b6c16820692f",
    "std": "e1875d31b7ac944bcc6e0c72a292b898ede117834b19cd8ae5930c7a7f450037",
    "val_origins": "6493583a54ec2a350be4834e88642b893b182985cfa6b85b7eb52393d6f8f09b",
    "graph": "ffcb0e1bf71229db00a97303f167642d63dd15177fa60ea6bedcf36cadd5fd23",
}


def array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _json_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fill_scale(physical: np.ndarray, mask: np.ndarray, mean: np.ndarray,
               std: np.ndarray) -> np.ndarray:
    """Fill each [L,C] history independently, then apply fixed TRAIN scaling.

    An optional leading batch axis is supported. Unobserved values are ignored;
    each channel starts from its TRAIN mean, including all-missing channels.
    """
    physical = np.asarray(physical, dtype=np.float32)
    mask = np.asarray(mask)
    mean, std = np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)
    if physical.ndim not in (2, 3) or any(size == 0 for size in physical.shape):
        raise ValueError("Physical histories must have nonempty [L,C] or [B,L,C] shape")
    if mask.shape != physical.shape or mask.dtype != np.bool_:
        raise ValueError("Observation mask must be bool and match physical histories")
    channels = physical.shape[-1]
    if mean.shape != (channels,) or std.shape != (channels,):
        raise ValueError("TRAIN mean/std must have one entry per ordered channel")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("TRAIN mean/std must be finite with positive standard deviations")
    if not np.isfinite(physical[mask]).all():
        raise ValueError("Observed physical values must be finite")
    batch = physical[None] if physical.ndim == 2 else physical
    observed = mask[None] if mask.ndim == 2 else mask
    last = np.broadcast_to(mean, (batch.shape[0], channels)).copy()
    filled = np.empty_like(batch)
    for time in range(batch.shape[1]):
        last = np.where(observed[:, time], batch[:, time], last)
        filled[:, time] = last
    normalized = ((filled - mean) / std).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("Scaled histories contain nonfinite values")
    return normalized[0] if physical.ndim == 2 else normalized


def _observed_graph(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    xt = x[:TRAIN_END].astype(np.float64)
    mt = mask[:TRAIN_END]
    channels = xt.shape[1]
    graph = np.zeros((channels, channels), dtype=np.float64)
    for i in range(channels):
        for j in range(i + 1, channels):
            both = mt[:, i] & mt[:, j]
            if both.sum() < 3:
                continue
            xi, xj = xt[both, i], xt[both, j]
            if np.std(xi) <= EPS or np.std(xj) <= EPS:
                continue
            corr = float(np.corrcoef(xi, xj)[0, 1])
            if np.isfinite(corr):
                graph[i, j] = graph[j, i] = abs(corr)
    np.fill_diagonal(graph, 0.)
    spectral_norm = float(np.linalg.norm(graph, ord=2))
    return (graph / spectral_norm if spectral_norm > EPS else graph).astype(np.float32)


def _legacy_contract_hash(data: dict[str, Any]) -> str:
    return _json_hash({
        "dataset": "jena", "x": array_hash(data["legacy_x"]), "mask": array_hash(data["mask"]),
        "mean": array_hash(data["mean"]), "std": array_hash(data["std"]),
        "train_bounds": (0, TRAIN_END), "val_bounds": (TRAIN_END, VAL_END),
        "train_window_starts": array_hash(np.arange(0, TRAIN_END - 512 + 1, 128, dtype=np.int64)),
        "train_forecast_origins": array_hash(np.arange(512, TRAIN_END - H + 1, 128, dtype=np.int64)),
        "val_origins": array_hash(data["val_origins"]), "graph": array_hash(data["graph"]),
        "l": 512, "h": H, "rec_stride": 128, "forecast_stride": 128,
    })


def load_data(asset_root: str | Path) -> dict[str, Any]:
    """Load the explicitly selected TRAIN/VAL archive without old loader writes.

    This minimal port preserves the original FP32 reconstruction and statistics
    operations byte for byte. Only the single allowlisted TRAIN/VAL file is
    opened; protected TEST archives and old artifact-writing helpers are unused.
    """
    archive_path = Path(asset_root) / ARCHIVE
    if _file_hash(archive_path) != ARCHIVE_SHA256:
        raise ValueError("Jena TRAIN/VAL archive SHA256 differs from the sealed parent")
    with np.load(archive_path, allow_pickle=False) as archive:
        if any("test" in key.lower() for key in archive.files):
            raise ValueError("TRAIN/VAL archive unexpectedly includes TEST arrays")
        columns = archive["columns"].tolist()
        x_v18 = archive["x"].astype(np.float32)
        mask = archive["finite"].astype(bool)
        v18_mean, v18_std = archive["mean"].astype(np.float32), archive["std"].astype(np.float32)
        val_origins = archive["val_origins"].astype(np.int64)
        times = archive["times"].astype(np.int64)
        if columns != list(COLUMNS) or archive["basis"].shape != (21, 6):
            raise ValueError("Original Jena channel/basis contract changed")
        if int(archive["train_start"]) != 0 or int(archive["train_end"]) != TRAIN_END:
            raise ValueError("Original Jena TRAIN boundary changed")
    if x_v18.shape != (VAL_END, 21) or mask.shape != x_v18.shape or times.shape != (VAL_END,):
        raise ValueError("Original Jena TRAIN/VAL shape changed")
    if not np.array_equal(val_origins, np.arange(TRAIN_END, VAL_END - H + 1, 24, dtype=np.int64)):
        raise ValueError("Original Jena VAL origin grid changed")
    physical_v18 = x_v18 * v18_std.reshape(1, -1) + v18_mean.reshape(1, -1)
    physical = np.where(mask, physical_v18, np.nan).astype(np.float32)
    train_values = np.where(mask[:TRAIN_END], physical[:TRAIN_END], np.nan)
    mean = np.nanmean(train_values, axis=0).astype(np.float32)
    std = np.nanstd(train_values, axis=0).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > EPS), std, 1.).astype(np.float32)
    legacy_x = fill_scale(physical, mask, mean, std)
    train_origins = np.arange(1024, TRAIN_END - H + 1, 128, dtype=np.int64)
    diagnostic_indices = np.linspace(0, len(train_origins) - 1, 16, dtype=np.int64)
    data = {
        "dataset": "jena", "raw_physical": physical, "mask": mask, "mean": mean, "std": std,
        "legacy_x": legacy_x, "graph": _observed_graph(legacy_x, mask), "columns": columns,
        "times": times, "train_end": TRAIN_END, "val_end": VAL_END,
        "train_bounds": (0, TRAIN_END), "val_bounds": (TRAIN_END, VAL_END),
        "train_origins": train_origins, "val_origins": val_origins,
        "diagnostic_origins": train_origins[diagnostic_indices], "h": H,
    }
    for key, expected in LEGACY_HASHES.items():
        if array_hash(data[key]) != expected:
            raise ValueError(f"Legacy Jena preprocessing fingerprint differs: {key}")
    data["legacy_contract_hash"] = _legacy_contract_hash(data)
    if data["legacy_contract_hash"] != LEGACY_CONTRACT_SHA256:
        raise ValueError("Legacy Jena data contract hash differs")
    data["contract_hash"] = public_contract(data)["contract_hash"]
    return data


def _window_origins(data: dict[str, Any], origins: Any, length: int) -> np.ndarray:
    if not isinstance(length, (int, np.integer)) or isinstance(length, (bool, np.bool_)) or length <= 0:
        raise ValueError("History length must be a positive integer")
    origins = np.asarray(origins)
    if origins.ndim != 1 or (origins.size and origins.dtype.kind not in "iu"):
        raise ValueError("Forecast origins must be a one-dimensional integer array")
    origins = origins.astype(np.int64)
    physical, mask = data["raw_physical"], data["mask"]
    if physical.ndim != 2 or mask.shape != physical.shape or mask.dtype != np.bool_:
        raise ValueError("Physical/mask data shape or observation dtype differs")
    end = int(data["val_end"])
    if end > len(physical) or np.any(origins < length) or np.any(origins > end - H):
        raise ValueError("Forecast history or target falls outside the available TRAIN/VAL timeline")
    return origins


def physical_windows(data: dict[str, Any], origins: Any, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Return only the exact [origin-length, origin) physical histories and masks."""
    origins = _window_origins(data, origins, length)
    channels = data["raw_physical"].shape[1]
    if not origins.size:
        return np.empty((0, length, channels), np.float32), np.empty((0, length, channels), bool)
    physical = np.stack([data["raw_physical"][o - length:o] for o in origins]).astype(np.float32)
    mask = np.stack([data["mask"][o - length:o] for o in origins]).astype(bool)
    return physical, mask


def windows(data: dict[str, Any], origins: Any, length: int,
            legacy: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized history, original-coordinate target, and target mask.

    ``legacy=True`` is solely for historical A0 reproduction. New histories
    always restart fill from TRAIN means at their own left boundary. Targets
    retain the parent's finite normalized values and original observation mask.
    """
    origins = _window_origins(data, origins, length)
    source = data["legacy_x"]
    if source.shape != data["raw_physical"].shape:
        raise ValueError("Legacy target coordinate shape differs")
    channels = source.shape[1]
    if not origins.size:
        return (np.empty((0, length, channels), np.float32), np.empty((0, H, channels), np.float32),
                np.empty((0, H, channels), bool))
    if legacy:
        history = np.stack([source[o - length:o] for o in origins]).astype(np.float32)
    else:
        physical, mask = physical_windows(data, origins, length)
        history = fill_scale(physical, mask, data["mean"], data["std"])
    target = np.stack([source[o:o + H] for o in origins]).astype(np.float32)
    target_mask = np.stack([data["mask"][o:o + H] for o in origins]).astype(bool)
    if not np.isfinite(history).all() or not np.isfinite(target[target_mask]).all():
        raise ValueError("History or observed target contains nonfinite values")
    return history, target, target_mask


def channel_macro_loss(pred, target, mask, global_counts=None):
    """Equal-channel observed MSE, optionally one effective-batch contribution.

    With ``global_counts=full_mask.sum((0, 1))``, sum microbatch losses before
    the optimizer step. A microbatch may lack a channel when the effective
    batch has observations for it. Do not divide contributions by microbatches.
    """
    import torch

    if pred.ndim != 3 or pred.shape != target.shape or mask.shape != pred.shape:
        raise ValueError("Prediction, target and mask must share [B,H,C] shape")
    if mask.dtype != torch.bool or not pred.is_floating_point() or not target.is_floating_point():
        raise ValueError("Predictions/targets must be floating point and mask must be bool")
    if any(size == 0 for size in pred.shape):
        raise ValueError("Loss requires a nonempty effective batch")
    if not torch.isfinite(pred).all() or not torch.isfinite(target[mask]).all():
        raise ValueError("Forecast or observed target is nonfinite")
    local_counts = mask.sum((0, 1)).to(dtype=pred.dtype)
    counts = local_counts if global_counts is None else torch.as_tensor(global_counts, device=pred.device,
                                                                       dtype=pred.dtype).detach()
    if (counts.shape != (pred.shape[-1],) or not torch.isfinite(counts).all()
            or torch.any(counts <= 0) or torch.any(counts < local_counts)):
        raise ValueError("Each channel needs valid full-batch observed counts covering this microbatch")
    error = torch.where(mask, pred - target, torch.zeros((), device=pred.device, dtype=pred.dtype))
    return (error.square().sum((0, 1)) / counts).mean()


def public_contract(data: dict[str, Any]) -> dict[str, Any]:
    """Return reproducible fingerprints and scopes without local absolute paths."""
    contract = {
        "schema": "gt_ts2_fastcheck_window_data_v1", "dataset": "jena", "horizon": H,
        "history_lengths": [512, 1024, 2048], "shape": list(data["raw_physical"].shape),
        "columns": list(data["columns"]), "train_bounds": [0, int(data["train_end"])],
        "val_bounds": [int(data["train_end"]), int(data["val_end"])],
        "normalization_fit_scope": "TRAIN observed values only",
        "graph_fit_scope": "TRAIN observed pairwise rows only",
        "input_preprocessing": "Each history starts at TRAIN mean, forward-fills within that window, then TRAIN-scales",
        "target_preprocessing": "Original parent normalized targets and observation masks",
        "val_history_scope": "TRAIN and earlier VAL history allowed; origin+48 <= VAL end",
        "train_origin_rule": "1024..26112 inclusive, stride=128; history and target within TRAIN",
        "diagnostic_origin_rule": "16 linearly spaced TRAIN-pool indices, integer truncation; fixed before errors",
        "interval_seconds": 600, "protected_test_accessed": False,
        "archive": {"path": ARCHIVE, "sha256": ARCHIVE_SHA256},
        "legacy_contract_hash": data["legacy_contract_hash"],
        "array_hashes": {key: array_hash(data[key]) for key in (*LEGACY_HASHES, "raw_physical")},
        "historical_raw_physical_sha256": HISTORICAL_RAW_PHYSICAL_SHA256,
        "raw_physical_fingerprint_note": "Historical DATA_CHECKS raw physical receipt differs from the current unchanged legacy loader; the six legacy model-coordinate fingerprints and legacy contract are checked strictly",
        "origin_hashes": {key: array_hash(data[key + "_origins"]) for key in ("train", "val", "diagnostic")},
        "origin_counts": {key: len(data[key + "_origins"]) for key in ("train", "val", "diagnostic")},
    }
    contract["contract_hash"] = _json_hash(contract)
    return contract
