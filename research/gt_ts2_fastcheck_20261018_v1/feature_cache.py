"""Immutable, origin-local PRE/RAND TS2 features with TRAIN-only statistics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .common import array_sha, file_sha, state_sha
from .data import windows
from .model import ForecastRoute
from .parents import export_official, export_tensor, load_ts2


LENGTH = 1024
BATCH = 4


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _source_pins():
    directory = Path(__file__).resolve().parent
    return {name: file_sha(directory / name) for name in
            ("feature_cache.py", "data.py", "model.py", "parents.py", "common.py")}


def _read_features(path, record, origins):
    if file_sha(path) != record["file_sha256"]:
        raise ValueError("Feature cache file SHA mismatch")
    with np.load(path, allow_pickle=False) as archive:
        features = archive["features"].copy()
        saved_origins = archive["origins"].copy()
    if (features.shape != (len(origins), LENGTH // 4, 320) or features.dtype != np.float32
            or not np.isfinite(features).all() or not np.array_equal(saved_origins, origins)
            or array_sha(features) != record["features_sha256"]):
        raise ValueError("Feature cache array schema, origin or SHA mismatch")
    return features


def _result(directory, receipt, features):
    train = features["train"]
    mean = train.astype(np.float64).mean(axis=(0, 1)).astype(np.float32)
    std64 = train.astype(np.float64).std(axis=(0, 1), ddof=0)
    std = np.where(std64 == 0, 1., std64).astype(np.float32)
    if (not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0)
            or array_sha(mean) != receipt["statistics"]["mean_sha256"]
            or array_sha(std) != receipt["statistics"]["std_sha256"]):
        raise ValueError("Feature cache TRAIN statistics SHA/schema mismatch")
    return {"features_paths": {split: directory / f"{split}_features.npz" for split in ("train", "val")},
            "mean": mean, "std": std, "state_sha": receipt["encoder_state_sha256"], "receipt": receipt}


def build_feature_cache(binding, data, kind, seed, output_dir, callback=None, device="cuda"):
    """Export full histories then pool blocks of four; cache unstandardized R4.

    The caller owns GPU scheduling. Existing files are immutable and reusable
    only when their complete identity and bytes still verify.
    """
    if kind not in ("PRE", "RAND"):
        raise ValueError("Feature kind must be PRE or RAND")
    if int(binding["seed"]) != int(seed):
        raise ValueError("Feature seed must match its canonical parent seed")
    directory = Path(output_dir).resolve()
    if directory.exists() and any(directory.iterdir()) and not (directory / "receipt.json").is_file():
        raise FileExistsError("Preserve partial feature cache; choose a fresh attempt directory")
    started = time.perf_counter()
    origins = {split: np.asarray(data[f"{split}_origins"], dtype=np.int64) for split in ("train", "val")}
    if any(not len(values) for values in origins.values()):
        raise ValueError("Feature statistics/export need nonempty TRAIN and VAL origins")
    if (np.any(origins["train"] - LENGTH < 0)
            or np.any(origins["train"] + 48 > int(data["train_end"]))):
        raise ValueError("TRAIN feature histories and targets must stay inside TRAIN")
    history = {}
    for split in ("train", "val"):
        history[split] = windows(data, origins[split], LENGTH, legacy=False)[0]
    encoder = load_ts2(binding, device=device, random=(kind == "RAND"))
    initial_sha = state_sha(encoder.net)
    identity = {
        "schema": "gt_ts2_pooled_features_v1", "kind": kind, "seed": int(seed),
        "length": LENGTH, "pooling": "mean_nonoverlapping_four_timestamps",
        "storage": "unstandardized_FP32_R4", "encoder_state_sha256": initial_sha,
        "parent_id": binding["id"], "parent_checkpoint_sha256": binding["checkpoint_sha256"],
        "data_contract_hash": data["contract_hash"], "preprocessing": "origin_local_TRAIN_mean_fill_then_TRAIN_scale",
        "input_sha256": {split: array_sha(history[split]) for split in ("train", "val")},
        "origins_sha256": {split: array_sha(origins[split]) for split in ("train", "val")},
        "origin_counts": {split: len(origins[split]) for split in ("train", "val")},
        "source_pins": _source_pins(), "parent_source_pins": binding.get("source_pins", {}),
    }
    identity_sha = _json_hash(identity)
    receipt_path = directory / "receipt.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (receipt.get("status") != "COMPLETE" or receipt.get("identity") != identity
                or receipt.get("identity_sha256") != identity_sha):
            raise ValueError("Existing feature cache identity differs; preserve and use a new attempt")
        if set(path.name for path in directory.iterdir()) != {"receipt.json", "train_features.npz", "val_features.npz"}:
            raise ValueError("Existing feature cache contains unexpected artifacts")
        features = {split: _read_features(directory / f"{split}_features.npz", receipt["files"][split], origins[split])
                    for split in ("train", "val")}
        if state_sha(encoder.net) != initial_sha:
            raise ValueError("Frozen TS2 encoder changed while validating feature cache")
        return _result(directory, receipt, features)
    directory.mkdir(parents=True, exist_ok=True)
    features = {}
    for split in ("train", "val"):
        pooled = []
        for start in range(0, len(history[split]), BATCH):
            x = torch.from_numpy(history[split][start:start + BATCH])
            representation = export_tensor(encoder, x)
            pooled.append(ForecastRoute.pool_features(representation).cpu().numpy())
            if callback is not None:
                callback(f"features_{kind}_s{seed}_{split}", min(start + BATCH, len(history[split])), len(history[split]))
        features[split] = np.concatenate(pooled).astype(np.float32)
        if not np.isfinite(features[split]).all():
            raise ValueError("Frozen TS2 pooled features contain NaN/Inf")
    check_count = min(BATCH, len(history["train"]))
    fixture = history["train"][:check_count]
    online = export_tensor(encoder, torch.from_numpy(fixture))
    official = export_official(encoder, fixture)
    online_numpy = online.cpu().numpy()
    np.testing.assert_allclose(online_numpy, official, atol=1e-5, rtol=1e-4)
    repeated_pool = ForecastRoute.pool_features(online).cpu().numpy()
    cached_pool = features["train"][:check_count]
    if array_sha(repeated_pool) != array_sha(cached_pool):
        raise ValueError("Feature cache does not exactly reproduce its online export")
    singles = torch.cat([ForecastRoute.pool_features(export_tensor(encoder, torch.from_numpy(fixture[i:i + 1])))
                         for i in range(check_count)]).cpu().numpy()
    np.testing.assert_allclose(singles, cached_pool, atol=1e-5, rtol=1e-4)
    final_sha = state_sha(encoder.net)
    if final_sha != initial_sha or any(p.grad is not None for n in (encoder.net, encoder._net) for p in n.parameters()):
        raise ValueError("Frozen TS2 encoder state or gradients changed during export")
    mean = features["train"].astype(np.float64).mean(axis=(0, 1)).astype(np.float32)
    std64 = features["train"].astype(np.float64).std(axis=(0, 1), ddof=0)
    std = np.where(std64 == 0, 1., std64).astype(np.float32)
    files = {}
    for split in ("train", "val"):
        path = directory / f"{split}_features.npz"
        arrays = {"features": features[split], "origins": origins[split]}
        if split == "train":
            arrays.update(mean=mean, std=std)
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        # Read back the exact bytes that training will consume.
        files[split] = {"reference": f"cache://{kind}_s{seed}_L{LENGTH}/{path.name}",
                        "file_sha256": file_sha(path), "features_sha256": array_sha(features[split]),
                        "origin_sha256": array_sha(origins[split]), "bytes": path.stat().st_size,
                        "shape": list(features[split].shape), "dtype": "float32"}
        _read_features(path, files[split], origins[split])
    receipt = {
        "status": "COMPLETE", "identity": identity, "identity_sha256": identity_sha,
        "encoder_state_sha256": initial_sha, "encoder_state_after_sha256": final_sha,
        "encoder_unchanged": True, "encoder_fitted": False, "files": files,
        "device": str(device), "batch_size": BATCH, "wall_seconds": time.perf_counter() - started,
        "statistics": {"fit_scope": "TRAIN pooled features only, all origins and ticks",
                       "sample_count": int(len(features["train"]) * (LENGTH // 4)),
                       "ddof": 0, "accumulation_dtype": "float64", "stored_dtype": "float32",
                       "exact_constant_dimensions": int(np.sum(std64 == 0)),
                       "constant_std_rule": "exact std==0 becomes 1; no threshold",
                       "mean_sha256": array_sha(mean), "std_sha256": array_sha(std)},
        "parity": {"fixture": "first four legal TRAIN origins, real full L1024 histories",
                   "fixture_origin_count": check_count, "atol": 1e-5, "rtol": 1e-4,
                   "tensor_official_max_abs": float(np.max(np.abs(online_numpy - official))),
                   "cache_online_exact": True, "cache_online_sha256": array_sha(cached_pool),
                   "batch1_batch4_max_abs": float(np.max(np.abs(singles - cached_pool)))},
    }
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return _result(directory, receipt, features)
