"""Independent v3 verifier for final report adoption.

This source is a runnable verifier for the root agent after all intended jobs
finish.  It performs no model training or inference.  It reads saved JSON/NPZ
artifacts, recomputes metrics in NumPy, checks hashes and selection rules, and
writes only to a caller-provided fresh output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"

sys.path.insert(0, str(HERE))
from common import Budget, LAMBDAS, access_guard  # noqa: E402
from synthetic import CHANGED_LAGS, CHANGED_SOURCES, INITIAL_LAGS, INITIAL_SOURCES, LENGTH  # noqa: E402


STRICT_TOLERANCES = {
    "float32_physical_roundtrip": 5e-4,
    "float64_physical_roundtrip": 1e-8,
    "selection_metric_order": 1e-12,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ahash(array: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def value_at(obj: Any, *keys: str) -> Any:
    cur = obj
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def check_status(ok: bool | None) -> str:
    if ok is True:
        return "PASS"
    if ok is False:
        return "FAIL"
    return "UNCHECKED"


def combine_statuses(*values: Any) -> str:
    statuses: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            status = value.get("status")
            if status in {"PASS", "FAIL", "UNCHECKED"}:
                statuses.append(status)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
        elif value in {"PASS", "FAIL", "UNCHECKED"}:
            statuses.append(str(value))

    for value in values:
        walk(value)
    if any(s == "FAIL" for s in statuses):
        return "FAIL"
    if any(s == "UNCHECKED" for s in statuses) or not statuses:
        return "UNCHECKED"
    return "PASS"


def verify_file_hash(path_value: Any, expected: Any) -> dict[str, Any]:
    if not path_value or not expected:
        return {"status": "UNCHECKED", "path": str(path_value or ""), "expected_sha256": expected}
    path = Path(str(path_value))
    if not path.is_absolute():
        root_path = ROOT / path
        path = root_path if root_path.exists() else HERE / path
    if not path.exists() or not path.is_file():
        return {"status": "FAIL", "path": str(path), "expected_sha256": expected, "exists": False}
    actual = sha256(path)
    return {"status": check_status(actual == expected), "path": str(path), "expected_sha256": expected,
            "actual_sha256": actual, "exists": True}


def metric_tolerance(calc: dict[str, Any]) -> float:
    dtype = str(calc.get("physical_prediction_dtype") or "")
    if "float32" in dtype:
        return STRICT_TOLERANCES["float32_physical_roundtrip"]
    return STRICT_TOLERANCES["float64_physical_roundtrip"]


def flatten_metric_diffs(calc: Any, reported: Any, prefix: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(calc, dict) and isinstance(reported, dict):
        for key, cval in calc.items():
            if key in {"mask_hash", "target_hash", "prediction_hash", "origin_hash", "prediction_dtype", "target_dtype",
                       "physical_prediction_dtype", "restriction"}:
                continue
            name = f"{prefix}.{key}" if prefix else str(key)
            if key not in reported:
                required_metric_key = key in {"mse", "mae", "per_channel_mse", "per_channel_mae", "observed_counts", "horizon_metrics", "normalized"}
                empty_optional_horizon = key == "horizon_metrics" and not cval
                if required_metric_key and not empty_optional_horizon:
                    rows.append({"path": name, "status": "FAIL", "reason": "reported metric key missing"})
                continue
            rows.extend(flatten_metric_diffs(cval, reported[key], name))
    elif isinstance(calc, (list, tuple)) and isinstance(reported, (list, tuple)):
        if len(calc) != len(reported):
            rows.append({"path": prefix, "status": "FAIL", "reason": f"length mismatch {len(calc)} != {len(reported)}"})
        for i, (cval, rval) in enumerate(zip(calc, reported)):
            rows.extend(flatten_metric_diffs(cval, rval, f"{prefix}[{i}]"))
    else:
        cnum = finite(calc)
        rnum = finite(reported)
        if cnum is not None and rnum is not None:
            rows.append({"path": prefix, "diff": abs(cnum - rnum), "status": "PENDING"})
    return rows


def metric_check(calc: dict[str, Any], reported: dict[str, Any], *, require_normalized: bool = True) -> dict[str, Any]:
    rows = flatten_metric_diffs(calc, reported)
    if require_normalized and "normalized" not in reported:
        rows.append({"path": "normalized", "status": "FAIL", "reason": "reported normalized companion missing"})
    if finite(calc.get("mse")) is not None and finite(calc.get("mae")) is not None:
        sanity_ok = float(calc["mae"]) ** 2 <= float(calc["mse"]) + 1e-9
        rows.append({"path": "MAE^2<=MSE", "status": check_status(sanity_ok),
                     "diff": 0.0 if sanity_ok else float(calc["mae"]) ** 2 - float(calc["mse"])})
    scalar_diffs = [float(row.get("diff", 0.0)) for row in rows if row.get("status") == "PENDING"]
    if scalar_diffs:
        diff = max(scalar_diffs)
    else:
        try:
            diff = metric_diff(calc, reported)
        except Exception:
            diff = float("inf")
    tol = metric_tolerance(calc)
    statuses = []
    for row in rows:
        if row.get("status") == "PENDING":
            row["tolerance"] = tol
            row["status"] = check_status(float(row.get("diff", 0.0)) <= tol)
        statuses.append(row["status"])
    return {"diff": diff, "tolerance": tol, "status": combine_statuses(statuses), "component_checks": rows[:2000]}


def broadcast_scale(scale: np.ndarray, arr: np.ndarray, name: str) -> np.ndarray:
    scale = np.asarray(scale)
    if scale.ndim == 0:
        return scale
    if arr.ndim == 3:
        if scale.ndim == 1:
            return scale[None, None, :]
        if scale.ndim == 2:
            if scale.shape[0] != arr.shape[0] or scale.shape[1] != arr.shape[2]:
                raise ValueError(f"{name} shape {scale.shape} cannot broadcast to {arr.shape}")
            return scale[:, None, :]
    if arr.ndim == 2 and scale.ndim == 1:
        return scale[None, :]
    try:
        np.broadcast_shapes(scale.shape, arr.shape)
    except ValueError as exc:
        raise ValueError(f"{name} shape {scale.shape} cannot broadcast to {arr.shape}") from exc
    return scale


def score(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    mask = mask.astype(bool)
    if pred.shape != target.shape or mask.shape != target.shape:
        raise ValueError(f"shape mismatch pred={pred.shape} target={target.shape} mask={mask.shape}")
    if not np.isfinite(pred).all():
        raise ValueError("prediction contains non-finite values")
    if not np.isfinite(target[mask]).all():
        raise ValueError("observed target contains non-finite values")
    err = np.where(mask, pred.astype(np.float64) - target.astype(np.float64), 0.0)
    counts = mask.sum(axis=(0, 1)).astype(np.float64)
    if np.any(counts <= 0):
        raise ValueError("empty observed channel")
    per_mse = (err * err).sum(axis=(0, 1)) / counts
    per_mae = np.abs(err).sum(axis=(0, 1)) / counts
    horizons: dict[str, Any] = {}
    if pred.ndim == 3 and pred.shape[1] == 48:
        for name, lo, hi in (("1_8", 0, 8), ("9_24", 8, 24), ("25_48", 24, 48)):
            horizons[name] = score(pred[:, lo:hi], target[:, lo:hi], mask[:, lo:hi])
    return {
        "mse": float(per_mse.mean()),
        "mae": float(per_mae.mean()),
        "per_channel_mse": per_mse.tolist(),
        "per_channel_mae": per_mae.tolist(),
        "observed_counts": counts.astype(np.int64).tolist(),
        "observed_total": int(mask.sum()),
        "mask_hash": ahash(mask),
        "target_hash": ahash(target),
        "horizon_metrics": horizons,
    }


def metric_diff(a: dict[str, Any], b: dict[str, Any]) -> float:
    return max(abs(float(a["mse"]) - float(b.get("mse"))), abs(float(a["mae"]) - float(b.get("mae"))))


def physical_npz_score(npz_path: Path, mean: np.ndarray | None = None,
                       std: np.ndarray | None = None) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=False) as z:
        pred = z["prediction"]
        target = z["target"]
        mask = z["mask"].astype(bool)
        if mean is None:
            mean = z["mean"] if "mean" in z.files else np.zeros(pred.shape[-1], dtype=pred.dtype)
        if std is None:
            std = z["std"] if "std" in z.files else np.ones(pred.shape[-1], dtype=pred.dtype)
        mean = broadcast_scale(np.asarray(mean), pred, "mean")
        std = broadcast_scale(np.asarray(std), pred, "std")
        # Let NumPy preserve the natural physical dtype: float32 predictions stay
        # float32 when mean/std are float32; ridge float64 stays float64.
        physical_pred = pred * std + mean
        physical_target = target * std + mean
        out = score(physical_pred, physical_target, mask)
        out["normalized"] = score(pred, target, mask)
        out["prediction_dtype"] = str(pred.dtype)
        out["target_dtype"] = str(target.dtype)
        out["physical_prediction_dtype"] = str(np.asarray(physical_pred).dtype)
        out["prediction_hash"] = ahash(pred)
        out["origin_hash"] = ahash(z["origins"]) if "origins" in z.files else (ahash(z["pairs"]) if "pairs" in z.files else None)
        return out


def _score_arrays_no_hash(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    scored = score(pred, target, mask)
    return {k: v for k, v in scored.items() if k not in {"mask_hash", "target_hash", "horizon_metrics"}}


def slice_metrics_from_npz(npz_path: Path) -> dict[str, Any]:
    slices = {
        "posthoc_0_127": (0, 128),
        "posthoc_128_255": (128, 256),
        "posthoc_256_383": (256, 384),
        "posthoc_384_511": (384, 512),
    }
    with np.load(npz_path, allow_pickle=False) as z:
        pred = z["prediction"]
        target = z["target"]
        mask = z["mask"].astype(bool)
        mean = broadcast_scale(z["mean"] if "mean" in z.files else np.zeros(pred.shape[-1], dtype=pred.dtype), pred, "mean")
        std = broadcast_scale(z["std"] if "std" in z.files else np.ones(pred.shape[-1], dtype=pred.dtype), pred, "std")
        physical_pred = pred * std + mean
        physical_target = target * std + mean
        out = {}
        length = pred.shape[1] if pred.ndim == 3 else 0
        for name, (lo, hi) in slices.items():
            if hi > length:
                continue
            out[name] = {
                "range": [lo, hi - 1],
                "physical": _score_arrays_no_hash(physical_pred[:, lo:hi], physical_target[:, lo:hi], mask[:, lo:hi]),
                "normalized": _score_arrays_no_hash(pred[:, lo:hi], target[:, lo:hi], mask[:, lo:hi]),
                "observed_total": int(mask[:, lo:hi].sum()),
                "post_hoc_not_prespecified": True,
            }
        return out


def reconstruction_npz_audit(npz_path: Path, reported: dict[str, Any]) -> dict[str, Any]:
    if not npz_path.exists():
        return {"status": "UNCHECKED", "path": str(npz_path), "reason": "reconstruction NPZ missing"}
    calc = physical_npz_score(npz_path)
    mcheck = metric_check(calc, reported or {}, require_normalized=False)
    return {
        "status": mcheck["status"],
        "path": str(npz_path),
        "sha256": sha256(npz_path),
        "reported_metric_max_abs_diff": mcheck["diff"],
        "metric_tolerance": mcheck["tolerance"],
        "metric_component_checks": mcheck.get("component_checks", []),
        "calculated": {k: calc[k] for k in ("mse", "mae", "normalized", "observed_total", "mask_hash", "target_hash", "origin_hash") if k in calc},
        "post_hoc_slice_metrics": slice_metrics_from_npz(npz_path),
        "slice_note": "post-hoc diagnostic quartiles of the 512-step reconstruction window; not a prespecified success gate",
    }


def per_origin_npz_scores(npz_path: Path) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=False) as z:
        pred = z["prediction"]
        target = z["target"]
        mask = z["mask"].astype(bool)
        mean = broadcast_scale(z["mean"] if "mean" in z.files else np.zeros(pred.shape[-1], dtype=pred.dtype), pred, "mean")
        std = broadcast_scale(z["std"] if "std" in z.files else np.ones(pred.shape[-1], dtype=pred.dtype), pred, "std")
        physical_pred = pred * std + mean
        physical_target = target * std + mean
        origins = z["origins"].astype(np.int64).tolist() if "origins" in z.files else list(range(pred.shape[0]))

    def per_sample(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> dict[str, Any]:
        err = np.where(m, a.astype(np.float64) - b.astype(np.float64), 0.0)
        counts = m.sum(axis=1).astype(np.float64)
        valid = counts > 0
        per_ch_mse = np.divide((err * err).sum(axis=1), counts, out=np.full_like(counts, np.nan, dtype=np.float64), where=valid)
        per_ch_mae = np.divide(np.abs(err).sum(axis=1), counts, out=np.full_like(counts, np.nan, dtype=np.float64), where=valid)
        return {
            "mse": np.nanmean(per_ch_mse, axis=1).astype(float).tolist(),
            "mae": np.nanmean(per_ch_mae, axis=1).astype(float).tolist(),
        }

    raw = per_sample(physical_pred, physical_target, mask)
    normalized = per_sample(pred, target, mask)
    return {
        "path": str(npz_path),
        "sha256": sha256(npz_path),
        "origins": origins,
        "origin_count": len(origins),
        "observed_count_per_origin": mask.sum(axis=(1, 2)).astype(np.int64).tolist(),
        "raw_mse": raw["mse"],
        "raw_mae": raw["mae"],
        "normalized_mse": normalized["mse"],
        "normalized_mae": normalized["mae"],
        "paired_origin_companion_only": True,
    }


def e5_probe_npz_audit(child: dict[str, Any], dataset_source: Path) -> dict[str, Any]:
    probe = child.get("reconstruction_probe") or {}
    npz_path = Path(probe.get("prediction_path", ""))
    if not npz_path.exists():
        return {"status": "UNCHECKED", "path": str(npz_path), "reason": "RECONSTRUCTION_PROBE_VAL.npz missing"}
    if not dataset_source.exists():
        return {"status": "UNCHECKED", "path": str(npz_path), "reason": "dataset source missing", "dataset_source": str(dataset_source)}
    with np.load(npz_path, allow_pickle=False) as z, np.load(dataset_source, allow_pickle=False) as dz:
        pred = z["prediction"]
        mask = z["mask"].astype(bool)
        origins = z["origins"].astype(np.int64)
        x = dz["x"]
        finite_mask = dz["finite"].astype(bool)
        source_mean = dz["mean"]
        source_std = dz["std"]
        target_norm = np.stack([x[int(o) - 512:int(o)] for o in origins])
        target_mask = np.stack([finite_mask[int(o) - 512:int(o)] for o in origins]) & mask
        target_phys = target_norm * source_std + source_mean
        fit_mean = np.asarray(child["fit_normalization_mean"], dtype=np.float32)
        fit_std = np.asarray(child["fit_normalization_std"], dtype=np.float32)
        physical_pred = pred * fit_std + fit_mean
        calc = score(physical_pred, target_phys, target_mask)
        calc["normalized"] = score(pred, target_norm, target_mask)
        calc["prediction_dtype"] = str(pred.dtype)
        calc["physical_prediction_dtype"] = str(np.asarray(physical_pred).dtype)
        calc["prediction_hash"] = ahash(pred)
        calc["origin_hash"] = ahash(origins)
    mcheck = metric_check(calc, probe.get("val", {}), require_normalized=False)
    return {
        "status": mcheck["status"],
        "path": str(npz_path),
        "sha256": sha256(npz_path),
        "dataset_source": str(dataset_source),
        "dataset_source_sha256": sha256(dataset_source),
        "reported_metric_max_abs_diff": mcheck["diff"],
        "metric_tolerance": mcheck["tolerance"],
        "metric_component_checks": mcheck.get("component_checks", []),
        "calculated": {k: calc[k] for k in ("mse", "mae", "normalized", "observed_total", "mask_hash", "target_hash", "origin_hash") if k in calc},
    }


def data_source_path(dataset: str) -> Path:
    assets = read_json(HERE / "ASSETS_AND_PREFLIGHT.json")
    source = value_at(assets, "data", dataset, "source")
    return Path(str(source or ""))


def restricted_full_observed_npz_score(npz_path: Path) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=False) as z:
        pred = z["prediction"]
        target = z["target"]
        mask = z["mask"].astype(bool)
        mean = broadcast_scale(z["mean"] if "mean" in z.files else np.zeros(pred.shape[-1], dtype=pred.dtype), pred, "mean")
        std = broadcast_scale(z["std"] if "std" in z.files else np.ones(pred.shape[-1], dtype=pred.dtype), pred, "std")
        full_observed = np.broadcast_to(mask.all(axis=2, keepdims=True), mask.shape)
        physical_pred = pred * std + mean
        physical_target = target * std + mean
        out = score(physical_pred, physical_target, full_observed)
        out["restriction"] = "all raw channels observed at timestamp; diagnostic only"
        out["prediction_dtype"] = str(pred.dtype)
        out["physical_prediction_dtype"] = str(np.asarray(physical_pred).dtype)
        return out


def latent_prediction_score(latent_path: Path, reference_path: Path | None = None) -> dict[str, Any]:
    with np.load(latent_path, allow_pickle=False) as z:
        pred = z["prediction"]
        mask = z["mask"].astype(bool)
        if "reference" in z.files:
            ref = z["reference"]
        elif reference_path is not None:
            with np.load(reference_path, allow_pickle=False) as rz:
                ref = rz["reference"]
        else:
            raise ValueError(f"No latent reference available for {latent_path}")
        out = score(pred, ref, mask)
        out["prediction_dtype"] = str(pred.dtype)
        out["reference_dtype"] = str(ref.dtype)
        out["physical_prediction_dtype"] = str(np.asarray(pred).dtype)
        out["prediction_hash"] = ahash(pred)
        out["reference_hash"] = ahash(ref)
        out["mask_hash"] = ahash(mask)
        return out


def decompose_same_target(prediction_path: Path, reference_path: Path) -> dict[str, Any]:
    """Descriptive same-target identity: P-y = (P-REF) + (REF-y)."""
    with np.load(prediction_path, allow_pickle=False) as pz, np.load(reference_path, allow_pickle=False) as rz:
        pmean = broadcast_scale(pz["mean"], pz["prediction"], "prediction_mean")
        pstd = broadcast_scale(pz["std"], pz["prediction"], "prediction_std")
        rmean = broadcast_scale(rz["mean"], rz["prediction"], "reference_mean")
        rstd = broadcast_scale(rz["std"], rz["prediction"], "reference_std")
        pred = pz["prediction"] * pstd + pmean
        ref = rz["prediction"] * rstd + rmean
        target = pz["target"] * pstd + pmean
        mask = pz["mask"].astype(bool)
        same = {
            "target_hash_equal": ahash(pz["target"]) == ahash(rz["target"]),
            "mask_hash_equal": ahash(mask) == ahash(rz["mask"].astype(bool)),
            "origin_hash_equal": ahash(pz["origins"]) == ahash(rz["origins"]) if "origins" in pz.files and "origins" in rz.files else None,
        }

    def part(a: np.ndarray, b: np.ndarray, y: np.ndarray, m: np.ndarray) -> dict[str, Any]:
        q = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
        r = np.asarray(b, dtype=np.float64) - np.asarray(y, dtype=np.float64)
        m = m.astype(bool)
        counts = m.sum(axis=(0, 1)).astype(np.float64)
        q2 = (np.where(m, q, 0.0) ** 2).sum(axis=(0, 1)) / counts
        r2 = (np.where(m, r, 0.0) ** 2).sum(axis=(0, 1)) / counts
        cross = 2.0 * (np.where(m, q * r, 0.0)).sum(axis=(0, 1)) / counts
        total = q2 + r2 + cross
        direct = np.asarray(score(a, y, m)["per_channel_mse"])
        residual = np.asarray(total) - direct
        return {
            "q2_mean": float(np.mean(q2)),
            "r2_mean": float(np.mean(r2)),
            "cross_2qr_mean": float(np.mean(cross)),
            "sum_mean": float(np.mean(total)),
            "direct_mse_mean": float(np.mean(direct)),
            "identity_max_abs_residual": float(np.max(np.abs(residual))),
            "per_channel_q2": q2.tolist(),
            "per_channel_r2": r2.tolist(),
            "per_channel_cross_2qr": cross.tolist(),
        }

    horizons = {}
    for name, lo, hi in (("1_8", 0, 8), ("9_24", 8, 24), ("25_48", 24, 48)):
        horizons[name] = part(pred[:, lo:hi], ref[:, lo:hi], target[:, lo:hi], mask[:, lo:hi])
    return {
        "same_target_checks": same,
        "whole": part(pred, ref, target, mask),
        "horizons": horizons,
        "interpretation": (
            "Descriptive identity on identical future samples. q is prediction-reference deviation; "
            "r is reference-to-target residual shared by representation/decoder. The cross-term can cancel, "
            "REF is not a lower bound, and r is not decoder-only."
        ),
    }


def argmin_update(curve: list[dict[str, Any]], key: str) -> int | None:
    rows = [(finite(row.get(key)), row.get("update")) for row in curve]
    rows = [(v, u) for v, u in rows if v is not None and u is not None]
    if not rows:
        return None
    return int(min(rows, key=lambda x: x[0])[1])


def verify_e34() -> dict[str, Any]:
    cases = []
    max_diff = 0.0
    statuses: list[str] = []
    for path in sorted((HERE / "e34").glob("*.json")):
        data = read_json(path)
        if data.get("experiment_id") != "E3_E4":
            continue
        case = {
            "path": str(path),
            "sha256": sha256(path),
            "dataset": data.get("dataset"),
            "dim": data.get("dim"),
            "seed": data.get("seed"),
            "status": data.get("status"),
            "device": data.get("device"),
            "scope": data.get("scope"),
            "origin_hashes": data.get("origin_hashes"),
            "source_pin_check": verify_source_pins(data.get("source_hashes") or data.get("source_pins")),
            "checkpoint_hash_ok": None,
            "decoder_selection_checks": [],
            "reference_isolation_ok": None,
            "methods": [],
        }
        statuses.append(case["source_pin_check"]["status"])
        ckpt = data.get("checkpoint_record") or {}
        ckpt_path = Path(ckpt.get("checkpoint", ""))
        if ckpt_path.exists() and ckpt.get("checkpoint_sha256"):
            case["checkpoint_hash_ok"] = sha256(ckpt_path) == ckpt["checkpoint_sha256"]
        case["checkpoint_hash_status"] = check_status(case["checkpoint_hash_ok"])
        statuses.append(case["checkpoint_hash_status"])
        mask_hashes, target_hashes, origin_hashes = set(), set(), set()
        method_npz_paths: dict[str, Path] = {}
        for section in ("E3", "E4"):
            for item in data.get(section, []):
                npz_path = Path(item.get("prediction_path", ""))
                method_npz_paths[str(item.get("method"))] = npz_path
                pred_current_sha = sha256(npz_path) if npz_path.exists() else None
                pred_expected_sha = item.get("prediction_hash")
                calc = physical_npz_score(npz_path)
                mcheck = metric_check(calc, item.get("metrics", {}))
                diff = float(mcheck["diff"])
                max_diff = max(max_diff, diff)
                statuses.append(str(mcheck["status"]))
                mask_hashes.add(calc["mask_hash"])
                target_hashes.add(calc["target_hash"])
                if calc.get("origin_hash"):
                    origin_hashes.add(calc["origin_hash"])
                method = {
                    "section": section,
                    "method": item.get("method"),
                    "prediction_path": str(npz_path),
                    "prediction_sha256_expected": pred_expected_sha,
                    "prediction_sha256_current": pred_current_sha,
                    "prediction_file_hash_ok": pred_current_sha == pred_expected_sha if pred_current_sha and pred_expected_sha else False,
                    "prediction_file_hash_status": check_status(pred_current_sha == pred_expected_sha if pred_current_sha and pred_expected_sha else False),
                    "reported_metric_max_abs_diff": diff,
                    "metric_tolerance": mcheck["tolerance"],
                    "metric_status": mcheck["status"],
                    "metric_component_checks": mcheck.get("component_checks", []),
                    "calculated": {k: calc[k] for k in ("mse", "mae", "normalized", "prediction_dtype", "physical_prediction_dtype", "mask_hash", "target_hash", "origin_hash")},
                }
                try:
                    method["originwise_scores"] = per_origin_npz_scores(npz_path)
                except Exception as exc:
                    method["originwise_scores"] = {"status": "FAIL", "error": str(exc)}
                    statuses.append("FAIL")
                if section == "E3":
                    rec_audit = reconstruction_npz_audit(npz_path.parent / "RECONSTRUCTION.npz", item.get("decoder", {}).get("reconstruction", {}))
                    method["reconstruction_npz_verification"] = rec_audit
                    statuses.append(rec_audit.get("status", "UNCHECKED"))
                if section == "E4" and item.get("latent_metrics"):
                    latent_name = {
                        "BOLT_LATENT": "BOLT_LATENT_PREDICTION.npz",
                        "LATENT_SELF_RIDGE": "LATENT_SELF_RIDGE_LATENT_PREDICTION.npz",
                        "LATENT_JOINT_RIDGE": "LATENT_JOINT_RIDGE_LATENT_PREDICTION.npz",
                        "LATENT_PERSISTENCE": "LATENT_PERSISTENCE_LATENT.npz",
                    }.get(str(item.get("method")))
                    if latent_name:
                        reference_latent = npz_path.parent / "BOLT_LATENT_PREDICTION.npz"
                        latent_calc = latent_prediction_score(npz_path.parent / latent_name, reference_latent)
                        latent_check = metric_check(latent_calc, item.get("latent_metrics", {}), require_normalized=False)
                        method["latent_metric_recalculation"] = {
                            "latent_path": str(npz_path.parent / latent_name),
                            "latent_sha256_current": sha256(npz_path.parent / latent_name) if (npz_path.parent / latent_name).exists() else None,
                            "reference_latent_path": str(reference_latent),
                            "reference_latent_sha256_current": sha256(reference_latent) if reference_latent.exists() else None,
                            "metric_status": latent_check["status"],
                            "reported_metric_max_abs_diff": latent_check["diff"],
                            "metric_tolerance": latent_check["tolerance"],
                            "calculated": {k: latent_calc[k] for k in ("mse", "mae", "per_channel_mse", "per_channel_mae", "horizon_metrics", "mask_hash", "reference_hash")},
                        }
                        statuses.append(str(latent_check["status"]))
                    restricted = restricted_full_observed_npz_score(npz_path)
                    method["restricted_full_observed_raw"] = {
                        "mse": restricted["mse"],
                        "mae": restricted["mae"],
                        "mask_hash": restricted["mask_hash"],
                        "observed_total": restricted["observed_total"],
                        "restriction": restricted["restriction"],
                    }
                statuses.append(method["prediction_file_hash_status"])
                if item.get("method") == "REFERENCE_LATENT_DIAGNOSTIC":
                    method["reference_is_diagnostic_not_bound"] = True
                    method["deployable"] = item.get("DEPLOYABLE")
                    case["reference_isolation_ok"] = item.get("DEPLOYABLE") is False
                    statuses.append(check_status(case["reference_isolation_ok"]))
                if item.get("decoder", {}).get("curve"):
                    curve = item["decoder"]["curve"]
                    expected = argmin_update(curve, "tune_rec_mse")
                    observed = item["decoder"].get("selected_update")
                    selection_status = check_status(observed == expected if expected is not None else None)
                    statuses.append(selection_status)
                    case["decoder_selection_checks"].append({
                        "method": item.get("method"),
                        "observed_selected_update": observed,
                        "argmin_tune_rec_mse_update": expected,
                        "argmin_ok": observed == expected,
                        "status": selection_status,
                        "selection_boundary": item["decoder"].get("selection_boundary"),
                    })
                case["methods"].append(method)
        case["same_mask_all_methods"] = len(mask_hashes) == 1
        case["same_target_all_methods"] = len(target_hashes) == 1
        case["same_origins_all_methods"] = len(origin_hashes) == 1
        case["same_mask_status"] = check_status(case["same_mask_all_methods"])
        case["same_target_status"] = check_status(case["same_target_all_methods"])
        case["same_origins_status"] = check_status(case["same_origins_all_methods"])
        statuses.extend([case["same_mask_status"], case["same_target_status"], case["same_origins_status"]])
        ref_path = method_npz_paths.get("REFERENCE_LATENT_DIAGNOSTIC")
        pred_path = method_npz_paths.get("BOLT_LATENT") or method_npz_paths.get("PREFIX_BOLT_REC_D")
        if ref_path and pred_path and ref_path.exists() and pred_path.exists():
            case["same_target_descriptive_decomposition"] = decompose_same_target(pred_path, ref_path)
        rank_rows = []
        for method in case["methods"]:
            latent = None
            for section in ("E3", "E4"):
                for item in data.get(section, []):
                    if item.get("method") == method.get("method") and isinstance(item.get("latent_metrics"), dict):
                        latent = item["latent_metrics"].get("mse")
            if latent is not None:
                rank_rows.append({
                    "method": method.get("method"),
                    "latent_mse": latent,
                    "reported_raw_mse": method.get("calculated", {}).get("mse"),
                    "restricted_full_observed_raw_mse": value_at(method, "restricted_full_observed_raw", "mse"),
                })
        if rank_rows:
            latent_rank = [r["method"] for r in sorted(rank_rows, key=lambda r: finite(r.get("latent_mse")) if finite(r.get("latent_mse")) is not None else float("inf"))]
            reported_rank = [r["method"] for r in sorted(rank_rows, key=lambda r: finite(r.get("reported_raw_mse")) if finite(r.get("reported_raw_mse")) is not None else float("inf"))]
            restricted_rank = [r["method"] for r in sorted(rank_rows, key=lambda r: finite(r.get("restricted_full_observed_raw_mse")) if finite(r.get("restricted_full_observed_raw_mse")) is not None else float("inf"))]
            case["latent_raw_ranking_reversal"] = {
                "rows": rank_rows,
                "latent_rank": latent_rank,
                "reported_raw_rank": reported_rank,
                "restricted_full_observed_raw_rank": restricted_rank,
                "reported_reversal": bool(latent_rank and reported_rank and latent_rank[0] != reported_rank[0]),
                "restricted_reversal": bool(latent_rank and restricted_rank and latent_rank[0] != restricted_rank[0]),
                "interpretation": "Same-D diagnostic; mask-aligned raw companion reduces population mismatch but does not assign D-only causal responsibility.",
            }
        case["verification_status"] = combine_statuses(case)
        cases.append(case)
    return {"status": combine_statuses(statuses), "case_count": len(cases), "max_reported_metric_diff": max_diff, "cases": cases}


def verify_e5() -> dict[str, Any]:
    runs = []
    max_diff = 0.0
    statuses: list[str] = []
    for path in sorted(RESULTS.glob("E5*/RESULT.json")):
        data = read_json(path)
        if data.get("experiment") != "E5" or "conditions" not in data:
            continue
        run = {
            "path": str(path),
            "sha256": sha256(path),
            "status": data.get("status"),
            "dataset": data.get("dataset"),
            "plan_sha256": data.get("plan_sha256"),
            "gate_sha256": data.get("gate_sha256"),
            "plan_hash_check": verify_file_hash(data.get("plan_path"), data.get("plan_sha256")),
            "gate_hash_check": verify_file_hash(data.get("gate_path"), data.get("gate_sha256")),
            "budget_ledger_hash_check": verify_file_hash(data.get("budget_ledger"), data.get("budget_ledger_sha256")),
            "source_pin_check": verify_source_pins(data.get("source_pins")),
            "conditions": [],
        }
        statuses.extend([run["plan_hash_check"]["status"], run["gate_hash_check"]["status"],
                         run["budget_ledger_hash_check"]["status"], run["source_pin_check"]["status"]])
        dataset_source = data_source_path(str(data.get("dataset", "")))
        for cond in data.get("conditions", []):
            child_path = Path(cond.get("path", ""))
            child_sha_current = sha256(child_path) if child_path.exists() else None
            child_sha_expected = cond.get("sha256")
            child = read_json(child_path)
            mean = np.asarray(child["fit_normalization_mean"], dtype=np.float32)
            std = np.asarray(child["fit_normalization_std"], dtype=np.float32)
            calc = physical_npz_score(Path(child["prediction_path"]), mean=mean, std=std)
            mcheck = metric_check(calc, child["val"])
            diff = float(mcheck["diff"])
            max_diff = max(max_diff, diff)
            probe = child.get("reconstruction_probe", {})
            curve = probe.get("curves", [])
            observed_probe = probe.get("selected_update")
            expected_probe = argmin_update(
                [{"update": row.get("update"), "mse": row.get("tune_observed_reconstruction", {}).get("mse")} for row in curve],
                "mse",
            )
            tune_scores = child.get("direct_ridge", {}).get("tune_scores", [])
            selected_lambda = child.get("direct_ridge", {}).get("lambda")
            lambda_argmin = None
            if isinstance(tune_scores, list) and tune_scores:
                if all(not isinstance(item, dict) for item in tune_scores):
                    scores = [finite(item) for item in tune_scores]
                    pairs = [(m, LAMBDAS[i]) for i, m in enumerate(scores[:len(LAMBDAS)]) if m is not None]
                else:
                    pairs = [
                        (finite(item.get("mse") or item.get("physical_mse") or item.get("tune_mse")), item.get("lambda"))
                        for item in tune_scores if isinstance(item, dict)
                    ]
                    pairs = [(m, l) for m, l in pairs if m is not None]
                if pairs:
                    lambda_argmin = min(pairs, key=lambda x: x[0])[1]
            child_hash_ok = child_sha_current == child_sha_expected
            prediction_path = Path(child["prediction_path"])
            prediction_sha_current = sha256(prediction_path) if prediction_path.exists() else None
            prediction_sha_expected = child.get("prediction_sha256")
            pred_hash_ok = prediction_sha_current == prediction_sha_expected
            probe_audit = e5_probe_npz_audit(child, dataset_source)
            probe_status = check_status(observed_probe == expected_probe if expected_probe is not None else None)
            lambda_status = check_status(selected_lambda == lambda_argmin if lambda_argmin is not None else None)
            condition_status = combine_statuses([
                check_status(child_hash_ok),
                check_status(pred_hash_ok),
                str(mcheck["status"]),
                probe_status,
                lambda_status,
                probe_audit.get("status", "UNCHECKED"),
            ])
            statuses.append(condition_status)
            run["conditions"].append({
                "condition": cond.get("condition"),
                "seed": cond.get("seed"),
                "child_path": str(child_path),
                "child_sha256_expected": child_sha_expected,
                "child_sha256_current": child_sha_current,
                "child_hash_ok": child_hash_ok,
                "child_hash_status": check_status(child_hash_ok),
                "prediction_path": str(prediction_path),
                "prediction_sha256_expected": prediction_sha_expected,
                "prediction_sha256_current": prediction_sha_current,
                "prediction_path": str(prediction_path),
                "prediction_sha256_expected": prediction_sha_expected,
                "prediction_sha256_current": prediction_sha_current,
                "prediction_hash_ok": pred_hash_ok,
                "prediction_hash_status": check_status(pred_hash_ok),
                "reported_metric_max_abs_diff": diff,
                "metric_tolerance": mcheck["tolerance"],
                "metric_status": mcheck["status"],
                "metric_component_checks": mcheck.get("component_checks", []),
                "calculated": {k: calc[k] for k in ("mse", "mae", "normalized", "prediction_dtype", "physical_prediction_dtype", "mask_hash", "target_hash", "origin_hash")},
                "probe_selected_update": observed_probe,
                "probe_argmin_update": expected_probe,
                "probe_argmin_ok": observed_probe == expected_probe,
                "probe_argmin_status": probe_status,
                "selected_lambda": selected_lambda,
                "lambda_argmin": lambda_argmin,
                "lambda_argmin_ok": None if lambda_argmin is None else selected_lambda == lambda_argmin,
                "lambda_argmin_status": lambda_status,
                "feature_scaler_scope": child.get("direct_ridge", {}).get("feature_scaler_scope"),
                "reconstruction_probe_npz_verification": probe_audit,
                "verification_status": condition_status,
            })
        run["verification_status"] = combine_statuses(
            run["plan_hash_check"],
            run["gate_hash_check"],
            run["budget_ledger_hash_check"],
            run["source_pin_check"],
            run["conditions"],
        )
        runs.append(run)
    return {"status": combine_statuses(statuses), "run_count": len(runs), "max_reported_metric_diff": max_diff, "runs": runs}


def bolt_parity_status(parity: Any) -> dict[str, Any]:
    if not isinstance(parity, dict):
        return {"status": "UNCHECKED", "reason": "Bolt_parity field absent or non-dict"}
    max_abs_values: list[float] = []
    frozen_flags: list[bool] = []

    def walk(obj: Any, key_path: str = "") -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                walk(value, f"{key_path}.{key}" if key_path else str(key))
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                walk(value, f"{key_path}[{i}]")
        else:
            lower = key_path.lower()
            if "max_abs" in lower or "maximum_absolute" in lower:
                x = finite(obj)
                if x is not None and not isinstance(obj, bool):
                    max_abs_values.append(abs(x))
            if "weights_frozen" in lower or "frozen_weight" in lower or lower.endswith("frozen"):
                if isinstance(obj, bool):
                    frozen_flags.append(obj)

    walk(parity)
    if not max_abs_values and not frozen_flags:
        return {"status": "UNCHECKED", "reason": "Bolt_parity has no max_abs / weights_frozen metadata", "raw": parity}
    max_abs = max(max_abs_values) if max_abs_values else 0.0
    ok = max_abs == 0.0 and all(frozen_flags or [True])
    return {"status": check_status(ok), "max_abs_numeric_value": max_abs,
            "weights_frozen_flags": frozen_flags, "raw": parity,
            "metadata_excluded": "boolean True and parameter counts are not treated as numeric parity errors"}


def verify_e5_bolt() -> dict[str, Any]:
    runs = []
    statuses: list[str] = []
    max_diff = 0.0
    result_paths = set(RESULTS.glob("E5_BOLT*/RESULT.json")) | set(RESULTS.glob("*BOLT*/RESULT.json")) | set(RESULTS.glob("*bolt*/RESULT.json"))
    for path in sorted(result_paths):
        data = read_json(path)
        if data.get("experiment") != "E5_BOLT":
            continue
        run = {
            "path": str(path),
            "sha256": sha256(path),
            "status": data.get("status"),
            "dataset": data.get("dataset", "jena"),
            "plan_hash_check": verify_file_hash(data.get("plan_path"), data.get("plan_sha256")),
            "gate_hash_check": verify_file_hash(data.get("gate_path"), data.get("gate_sha256")),
            "budget_ledger_hash_check": verify_file_hash(data.get("budget_ledger"), data.get("budget_ledger_sha256")),
            "source_pin_check": verify_source_pins(data.get("source_pins")),
            "bolt_parity_check": bolt_parity_status(data.get("Bolt_parity")),
            "conditions": [],
            "cost_only_not_science": "COST_ONLY" in str(data.get("status", "")),
        }
        if run["cost_only_not_science"]:
            run["verification_status"] = "COST_ONLY_PROFILE_NOT_SCIENCE"
            runs.append(run)
            continue
        run_statuses = [run["plan_hash_check"]["status"], run["gate_hash_check"]["status"],
                        run["budget_ledger_hash_check"]["status"], run["source_pin_check"]["status"],
                        run["bolt_parity_check"]["status"]]
        plan_path = Path(str(data.get("plan_path", "")))
        if data.get("plan_path") and not plan_path.is_absolute():
            root_path = ROOT / plan_path
            plan_path = root_path if root_path.exists() else HERE / plan_path
        plan = read_json(plan_path) if data.get("plan_path") and plan_path.exists() else {}
        if not run.get("dataset"):
            run["dataset"] = plan.get("dataset", "jena")
        binding_map = {(str(b.get("condition")), str(b.get("seed"))): b for b in plan.get("bindings", [])}
        for cond in data.get("conditions", []) or data.get("cases", []):
            condition = str(cond.get("condition", ""))
            seed = str(cond.get("seed", ""))
            binding = binding_map.get((condition, seed), {})
            child_path = Path(cond.get("path", ""))
            child_sha_current = sha256(child_path) if child_path.exists() else None
            child_sha_expected = cond.get("sha256")
            child = read_json(child_path) if child_path.exists() else {}
            original_case = read_json(Path(binding["case_path"])) if binding.get("case_path") and Path(binding["case_path"]).exists() else {}
            cond_statuses = [check_status(child_sha_current == child_sha_expected)]
            if binding.get("asset_pins"):
                asset_check = verify_source_pins(binding.get("asset_pins"))
                cond_statuses.append(asset_check["status"])
            else:
                asset_check = {"status": "UNCHECKED", "reason": "binding asset_pins absent"}
                cond_statuses.append("UNCHECKED")
            target_hashes, mask_hashes, origin_hashes = set(), set(), set()
            methods = []
            for method in cond.get("methods", []) or child.get("methods", []):
                if not isinstance(method, dict):
                    continue
                npz_path = Path(method.get("prediction_path", ""))
                mean = std = None
                if method.get("method") == "DIRECT_RAW_RIDGE_REUSED_SAME_ORIGINS":
                    with np.load(npz_path, allow_pickle=False) as z:
                        has_scale = "mean" in z.files and "std" in z.files
                    if not has_scale:
                        mean = np.asarray(original_case.get("fit_normalization_mean"), dtype=np.float32)
                        std = np.asarray(original_case.get("fit_normalization_std"), dtype=np.float32)
                calc = physical_npz_score(npz_path, mean=mean, std=std)
                mcheck = metric_check(calc, method.get("val", {}))
                diff = float(mcheck["diff"])
                max_diff = max(max_diff, diff)
                prediction_sha_current = sha256(npz_path) if npz_path.exists() else None
                prediction_sha_expected = method.get("prediction_sha256")
                pred_hash_ok = prediction_sha_current == prediction_sha_expected
                cond_statuses.extend([check_status(pred_hash_ok), str(mcheck["status"])])
                target_hashes.add(calc["target_hash"])
                mask_hashes.add(calc["mask_hash"])
                if calc.get("origin_hash"):
                    origin_hashes.add(calc["origin_hash"])
                methods.append({
                    "method": method.get("method"),
                    "prediction_path": str(npz_path),
                    "prediction_sha256_expected": prediction_sha_expected,
                    "prediction_sha256_current": prediction_sha_current,
                    "prediction_hash_ok": pred_hash_ok,
                    "prediction_hash_status": check_status(pred_hash_ok),
                    "reported_metric_max_abs_diff": diff,
                    "metric_tolerance": mcheck["tolerance"],
                    "metric_status": mcheck["status"],
                    "metric_component_checks": mcheck.get("component_checks", []),
                    "calculated": {k: calc[k] for k in ("mse", "mae", "normalized", "prediction_dtype", "physical_prediction_dtype", "mask_hash", "target_hash", "origin_hash")},
                })
            same_target = len(target_hashes) <= 1 and len(target_hashes) > 0
            same_mask = len(mask_hashes) <= 1 and len(mask_hashes) > 0
            same_origin = len(origin_hashes) <= 1 and len(origin_hashes) > 0
            cond_statuses.extend([check_status(same_target), check_status(same_mask), check_status(same_origin)])
            frozen_declared = child.get("optimizer_updates") == 0 and bool(child.get("E_hash_before_after")) and bool(child.get("D_hash_before_after"))
            cond_statuses.append(check_status(frozen_declared))
            condition_status = combine_statuses(cond_statuses)
            run_statuses.append(condition_status)
            run["conditions"].append({
                "condition": condition,
                "seed": seed,
                "child_path": str(child_path),
                "child_sha256_expected": child_sha_expected,
                "child_sha256_current": child_sha_current,
                "asset_pin_check": asset_check,
                "same_target_all_methods": same_target,
                "same_mask_all_methods": same_mask,
                "same_origins_all_methods": same_origin,
                "frozen_E_D_declared": frozen_declared,
                "frozen_E_D_note": "e5_bolt_v2 source checks before/after hashes; result stores the checked hash values, not a two-element tuple",
                "methods": methods,
                "verification_status": condition_status,
            })
        run["verification_status"] = combine_statuses(run_statuses)
        statuses.append(run["verification_status"])
        runs.append(run)
    return {"status": combine_statuses(statuses), "run_count": len(runs), "max_reported_metric_diff": max_diff, "runs": runs}


def verify_source_pins(pins: dict[str, Any] | None) -> dict[str, Any]:
    if not pins:
        return {"status": "UNCHECKED", "reason": "no source_pins field"}
    snapshots = list((HERE / "source_snapshots").glob("*.py"))
    records = []
    statuses: list[str] = []
    for raw_path, expected in sorted(pins.items()):
        path = Path(raw_path)
        if not path.is_absolute():
            root_path = ROOT / path
            path = root_path if root_path.exists() else HERE / path
        current_sha = sha256(path) if path.exists() and path.is_file() else None
        current_ok = current_sha == expected
        snapshot_match = None
        if not current_ok:
            for snap in snapshots:
                if sha256(snap) == expected:
                    snapshot_match = str(snap)
                    break
        status = check_status(bool(current_ok or snapshot_match))
        statuses.append(status)
        records.append({
            "path": str(path),
            "expected_sha256": expected,
            "current_exists": path.exists(),
            "current_sha256": current_sha,
            "current_match": current_ok,
            "historical_snapshot_match": snapshot_match,
            "status": status,
        })
    return {"status": combine_statuses(statuses), "records": records}


def graph_npz_trace_audit(result_path: Path, data: dict[str, Any]) -> dict[str, Any]:
    """Audit saved graph trace arrays without running graph/model code."""
    result_dir = result_path.parent
    preprocess_dir = result_dir
    prepared_stage_check: dict[str, Any] | None = None
    if not (preprocess_dir / "GRAPH_PREPROCESS_fit.json").exists():
        completed_stage = result_dir / "COMPLETED_CPU_STAGE.json"
        prepared_path_value = None
        prepared_sha_value = None
        if completed_stage.exists():
            stage = read_json(completed_stage)
            prepared_path_value = stage.get("prepared_path")
            prepared_sha_value = stage.get("prepared_sha256")
        if not prepared_path_value and data.get("plan_path"):
            plan_path = Path(str(data["plan_path"]))
            if not plan_path.is_absolute():
                root_path = ROOT / plan_path
                plan_path = root_path if root_path.exists() else HERE / plan_path
            plan = read_json(plan_path)
            stage = plan.get("completed_CPU_stage") or {}
            prepared_path_value = stage.get("path")
            prepared_sha_value = stage.get("sha256")
        if prepared_path_value:
            prepared_path = Path(str(prepared_path_value))
            if not prepared_path.is_absolute():
                root_path = ROOT / prepared_path
                prepared_path = root_path if root_path.exists() else HERE / prepared_path
            prepared_stage_check = verify_file_hash(prepared_path, prepared_sha_value)
            if prepared_path.exists():
                proof = read_json(prepared_path)
                ledger_check = verify_file_hash(proof.get("CPU_preparation_ledger"), proof.get("CPU_preparation_ledger_sha256"))
                prepared_stage_check["CPU_preparation_ledger_check"] = ledger_check
                prepared_stage_check["source_pin_check"] = verify_source_pins(proof.get("source_pins"))
                preprocess_dir = prepared_path.parent
    dataset = str(data.get("dataset", ""))
    split_audits = {}
    statuses: list[str] = []
    if prepared_stage_check:
        statuses.append(prepared_stage_check["status"])
        statuses.append(value_at(prepared_stage_check, "CPU_preparation_ledger_check", "status") or "UNCHECKED")
        statuses.append(value_at(prepared_stage_check, "source_pin_check", "status") or "UNCHECKED")
    for split, graph in (data.get("graph_arrays") or {}).items():
        npz_path = Path((graph or {}).get("path", ""))
        preprocess_path = preprocess_dir / f"GRAPH_PREPROCESS_{split}.json"
        split_statuses: list[str] = []
        audit: dict[str, Any] = {
            "graph_npz_path": str(npz_path),
            "preprocess_path": str(preprocess_path),
            "prepared_stage_check": prepared_stage_check,
            "file_hash_status": "UNCHECKED",
            "array_hash_status": "UNCHECKED",
            "cutoff_status_by_mode": {},
            "known_edge_trace_recovery": None,
        }
        if not npz_path.exists() or not preprocess_path.exists():
            audit["status"] = "FAIL"
            audit["reason"] = "graph NPZ or GRAPH_PREPROCESS json missing"
            statuses.append("FAIL")
            split_audits[split] = audit
            continue
        file_hash_ok = sha256(npz_path) == (graph or {}).get("sha256")
        audit["file_hash_status"] = check_status(file_hash_ok)
        split_statuses.append(audit["file_hash_status"])
        preprocess = read_json(preprocess_path)
        with np.load(npz_path, allow_pickle=False) as z:
            array_hash_checks = {}
            missing_hash_keys = []
            for key, expected_hash in ((graph or {}).get("array_hashes") or {}).items():
                if key in z.files:
                    array_hash_checks[key] = ahash(z[key]) == expected_hash
                else:
                    missing_hash_keys.append(key)
            audit["array_hash_status"] = check_status(bool(array_hash_checks) and all(array_hash_checks.values()))
            if missing_hash_keys:
                audit["array_hash_status"] = "FAIL"
                audit["missing_expected_array_hash_keys"] = missing_hash_keys
            audit["array_hash_checked_count"] = len(array_hash_checks)
            split_statuses.append(audit["array_hash_status"])
            pairs = z["record_origin_pairs"].astype(np.int64)
            record_ids = pairs[:, 0]
            origins = pairs[:, 1]
            input_end = np.asarray(preprocess.get("input_end_inclusive", origins - 1), dtype=np.int64)
            support_start = np.asarray(preprocess.get("graph_support_start", origins - 512), dtype=np.int64)
            fixed_exclusive = np.asarray(preprocess.get("fixed_cutoffs_exclusive", []), dtype=np.int64)
            for mode in ("FIXED_LAG_GRAPH", "PAST_UPDATED_LAG_GRAPH", "WRONG_EDGE_GRAPH"):
                ckey = f"{mode}_trace_effective_cutoff_inclusive"
                ukey = f"{mode}_trace_update_indices"
                if ckey not in z.files or ukey not in z.files:
                    mode_status = "UNCHECKED"
                    audit["cutoff_status_by_mode"][mode] = {"status": mode_status, "reason": "trace arrays missing"}
                    split_statuses.append(mode_status)
                    continue
                cutoffs = z[ckey].astype(np.int64)
                updates = z[ukey].astype(np.int64)
                if mode == "PAST_UPDATED_LAG_GRAPH":
                    allowed = support_start[:, None] + updates
                    valid = (cutoffs < 0) | ((cutoffs <= allowed) & (cutoffs <= input_end[:, None]))
                    audit["cutoff_status_by_mode"][mode] = {
                        "status": check_status(bool(np.all(valid))),
                        "rule": "UPDATED cutoff must be held (-1) or <= support_start + local_update and <= input_end",
                        "max_violation": int(np.max(np.where(valid, 0, cutoffs - np.minimum(allowed, input_end[:, None])))) if valid.size else 0,
                    }
                else:
                    expected = fixed_exclusive[np.maximum(record_ids, 0)] - 1 if fixed_exclusive.size else np.full(len(record_ids), -1)
                    exact = cutoffs == expected[:, None]
                    if split == "fit":
                        nofuture = np.ones_like(exact, dtype=bool)
                        nofuture_note = "FIT fixed artifact may use full FIT cutoff even for earlier FIT origins; not treated as deploy input leak."
                    else:
                        nofuture = cutoffs <= input_end[:, None]
                        nofuture_note = "TUNE/VAL fixed artifact cutoff must be before each origin input end."
                    ok = np.all(exact) and np.all(nofuture)
                    audit["cutoff_status_by_mode"][mode] = {
                        "status": check_status(bool(ok)),
                        "rule": "FIXED/WRONG cutoff must equal frozen FIT cutoff-1; TUNE/VAL additionally before input end",
                        "exact_fixed_cutoff": bool(np.all(exact)),
                        "nofuture_relative_to_input": bool(np.all(nofuture)),
                        "nofuture_note": nofuture_note,
                    }
                split_statuses.append(audit["cutoff_status_by_mode"][mode]["status"])
            if dataset.startswith("synthetic") and "FIXED_LAG_GRAPH_trace_source" in z.files and "FIXED_LAG_GRAPH_trace_lag" in z.files:
                source = z["FIXED_LAG_GRAPH_trace_source"].astype(np.int64)
                lag = z["FIXED_LAG_GRAPH_trace_lag"].astype(np.int64)
                tune_end = int(0.8 * LENGTH)
                change = tune_end + (LENGTH - tune_end) // 2 if dataset.endswith("changed") else None
                hits = 0
                possible = 0
                for i, origin in enumerate(origins):
                    changed = change is not None and origin - 1 >= change
                    sources = CHANGED_SOURCES if changed else INITIAL_SOURCES
                    lags = CHANGED_LAGS if changed else INITIAL_LAGS
                    event_idx = -1
                    for j, (src, ell) in enumerate(zip(sources, lags)):
                        receiver = 3 + j
                        possible += 1
                        hits += int(np.any((source[i, event_idx, receiver] == src) & (lag[i, event_idx, receiver] == ell)))
                frac = float(hits / possible) if possible else None
                audit["known_edge_trace_recovery"] = {
                    "source": "synthetic.py metadata constants; FIXED_LAG_GRAPH final recorded event per VAL/FIT/TUNE row",
                    "hits": hits,
                    "possible": possible,
                    "fraction": frac,
                    "not_causal_proof": True,
                }
        audit["status"] = combine_statuses(split_statuses)
        statuses.append(audit["status"])
        split_audits[split] = audit
    return {"status": combine_statuses(statuses), "splits": split_audits}


def verify_e6() -> dict[str, Any]:
    runs = []
    failures = []
    statuses: list[str] = []
    for path in sorted(RESULTS.glob("E6*/FAILURE.json")):
        data = read_json(path)
        failures.append({"path": str(path), "sha256": sha256(path), "status": data.get("status"), "failure_reason": data.get("failure_reason")})
    for path in sorted(RESULTS.glob("E6*/RESULT.json")):
        data = read_json(path)
        if data.get("experiment") != "E6":
            continue
        run = {
            "path": str(path),
            "sha256": sha256(path),
            "dataset": data.get("dataset"),
            "status": data.get("status"),
            "compute_device": data.get("compute_device"),
            "plan_sha256": data.get("plan_sha256"),
            "gate_sha256": data.get("gate_sha256"),
            "plan_hash_check": verify_file_hash(data.get("plan_path"), data.get("plan_sha256")),
            "gate_hash_check": verify_file_hash(data.get("gate_path"), data.get("gate_sha256")),
            "graph_preprocessing_ledger_hash_check": verify_file_hash(data.get("graph_preprocessing_ledger"), data.get("graph_preprocessing_ledger_sha256")),
            "neural_budget_ledger_hash_check": verify_file_hash(data.get("neural_budget_ledger"), data.get("neural_budget_ledger_sha256")),
            "graph_cutoff_trace_hashes_present": {},
            "source_pin_check": verify_source_pins(data.get("source_pins")),
            "graph_arrays": data.get("graph_arrays"),
            "graph_trace_audit": None,
            "true_edge_recovery": None,
            "conditions": [],
            "matched_baselines": {},
        }
        statuses.extend([run["plan_hash_check"]["status"], run["gate_hash_check"]["status"],
                         run["graph_preprocessing_ledger_hash_check"]["status"],
                         run["neural_budget_ledger_hash_check"]["status"], run["source_pin_check"]["status"]])
        for split, graph in (data.get("graph_arrays") or {}).items():
            hashes = (graph or {}).get("array_hashes", {})
            run["graph_cutoff_trace_hashes_present"][split] = {
                "effective_cutoff": any("trace_effective_cutoff" in key for key in hashes),
                "observed_fit_samples": any("trace_observed_fit_samples" in key for key in hashes),
                "source": any("trace_source" in key for key in hashes),
                "lag": any("trace_lag" in key for key in hashes),
            }
        run["graph_trace_audit"] = graph_npz_trace_audit(path, data)
        statuses.append(run["graph_trace_audit"]["status"])
        if str(data.get("dataset", "")).startswith("synthetic"):
            # This is an audit of known-edge trace availability, not a causal proof.
            run["true_edge_recovery"] = {
                "available": bool(value_at(run["graph_trace_audit"], "splits", "val", "known_edge_trace_recovery")),
                "from_npz_trace": value_at(run["graph_trace_audit"], "splits", "val", "known_edge_trace_recovery"),
                "claim": "exact source/lag trace audit against synthetic metadata; not causal proof or real-data evidence",
                "not_causal_proof": True,
            }
        for name, baseline in (data.get("matched_baselines") or {}).items():
            metric = baseline.get("val", {})
            arrays = baseline.get("arrays") or {}
            baseline_statuses: list[str] = []
            calc = None
            mcheck = None
            if arrays.get("path"):
                arr_path = Path(arrays["path"])
                arr_hash_ok = arr_path.exists() and sha256(arr_path) == arrays.get("sha256")
                baseline_statuses.append(check_status(arr_hash_ok))
                if arr_path.exists():
                    with np.load(arr_path, allow_pickle=False) as z:
                        missing = [key for key in (arrays.get("array_hashes") or {}) if key not in z.files]
                        hash_ok = all(ahash(z[key]) == expected for key, expected in (arrays.get("array_hashes") or {}).items() if key in z.files) and not missing
                    baseline_statuses.append(check_status(hash_ok))
                    calc = physical_npz_score(arr_path)
                    mcheck = metric_check(calc, metric)
                    baseline_statuses.append(str(mcheck["status"]))
            else:
                baseline_statuses.append("UNCHECKED")
            run["matched_baselines"][name] = {
                "mse": metric.get("mse"),
                "mae": metric.get("mae"),
                "normalized": metric.get("normalized"),
                "arrays": arrays,
                "array_sha256_expected": arrays.get("sha256"),
                "array_sha256_current": sha256(Path(arrays["path"])) if arrays.get("path") and Path(arrays["path"]).exists() else None,
                "calculated": None if calc is None else {k: calc[k] for k in ("mse", "mae", "normalized", "mask_hash", "target_hash", "origin_hash")},
                "metric_status": None if mcheck is None else mcheck["status"],
                "reported_metric_max_abs_diff": None if mcheck is None else mcheck["diff"],
                "verification_status": combine_statuses(baseline_statuses),
                "note": "pooled raw baseline; SELF neural condition still sees all raw channels in history",
            }
            statuses.append(run["matched_baselines"][name]["verification_status"])
        for cond in data.get("conditions", []):
            child_path = Path(cond.get("path", ""))
            child_sha_current = sha256(child_path) if child_path.exists() else None
            child_sha_expected = cond.get("sha256")
            child = read_json(child_path)
            arrays = child.get("arrays", {})
            prediction_path = Path(arrays.get("path") or child.get("prediction_path", ""))
            calc = None
            diff = None
            condition_statuses: list[str] = []
            child_hash_ok = child_sha_current == child_sha_expected
            condition_statuses.append(check_status(child_hash_ok))
            pred_hash_ok = None
            prediction_sha_current = sha256(prediction_path) if prediction_path.exists() else None
            prediction_sha_expected = arrays.get("sha256") or child.get("prediction_sha256")
            if prediction_path.exists():
                pred_hash_ok = prediction_sha_current == prediction_sha_expected
                condition_statuses.append(check_status(pred_hash_ok))
            if prediction_path.exists():
                try:
                    calc = physical_npz_score(prediction_path)
                    mcheck = metric_check(calc, child["val"])
                    diff = float(mcheck["diff"])
                    condition_statuses.append(str(mcheck["status"]))
                    metric_status = mcheck["status"]
                    metric_tolerance = mcheck["tolerance"]
                except Exception as exc:
                    calc = {"verification_error": str(exc)}
                    condition_statuses.append("FAIL")
                    metric_status = "FAIL"
                    metric_tolerance = None
            else:
                condition_statuses.append("FAIL")
                metric_status = "FAIL"
                metric_tolerance = None
            no_history_ok = child.get("no_history_retroactive_graph") is True
            future_held_ok = child.get("future_graph_held") is True
            if cond.get("mode") != "SELF_ONLY":
                condition_statuses.append(check_status(no_history_ok))
                condition_statuses.append(check_status(future_held_ok))
            condition_status = combine_statuses(condition_statuses)
            statuses.append(condition_status)
            run["conditions"].append({
                "mode": cond.get("mode"),
                "seed": cond.get("seed"),
                "child_path": str(child_path),
                "child_sha256_expected": child_sha_expected,
                "child_sha256_current": child_sha_current,
                "child_hash_ok": child_hash_ok,
                "child_hash_status": check_status(child_hash_ok),
                "prediction_hash_ok": pred_hash_ok,
                "prediction_hash_status": check_status(pred_hash_ok),
                "input_support": child.get("input_support"),
                "latent_timestamp_rule": child.get("latent_timestamp_rule"),
                "graph_fit_cutoff": child.get("graph_fit_cutoff"),
                "no_history_retroactive_graph": child.get("no_history_retroactive_graph"),
                "future_graph_held": child.get("future_graph_held"),
                "graph_diagnostics": child.get("graph_diagnostics"),
                "reported_metric_max_abs_diff": diff,
                "metric_tolerance": metric_tolerance,
                "metric_status": metric_status,
                "metric_component_checks": mcheck.get("component_checks", []) if isinstance(mcheck, dict) else [],
                "calculated": calc,
                "self_neural_not_no_other_channels": cond.get("mode") == "SELF_ONLY",
                "verification_status": condition_status,
            })
        run["verification_status"] = combine_statuses(statuses[-len(run.get("conditions", [])):], run["plan_hash_check"], run["gate_hash_check"], run["graph_preprocessing_ledger_hash_check"], run["neural_budget_ledger_hash_check"], run["source_pin_check"], run.get("graph_trace_audit"), list(run.get("matched_baselines", {}).values()))
        statuses.append(run["verification_status"])
        runs.append(run)
    return {"status": combine_statuses(statuses), "run_count": len(runs), "runs": runs, "failures": failures}



def timing_block_status(block: Any) -> dict[str, Any]:
    if not isinstance(block, dict):
        return {"status": "UNCHECKED", "reason": "timing block missing"}
    samples = block.get("samples_seconds") or []
    timed = finite(block.get("timed"))
    checks = [
        check_status(bool(block.get("device"))),
        check_status(block.get("physical_inverse_and_CPU_output_included") is True),
        check_status(block.get("model_loading_disk_and_guards_excluded") is True),
        check_status(isinstance(samples, list) and len(samples) == int(timed or -1)),
    ]
    for key in ("p50_seconds", "p95_seconds", "seconds_per_window_p50"):
        value = finite(block.get(key))
        checks.append(check_status(value is not None and value >= 0.0))
    for key in ("baseline_allocated_vram", "peak_allocated_vram", "peak_reserved_vram"):
        value = finite(block.get(key))
        if value is not None:
            checks.append(check_status(value >= 0.0))
    return {
        "status": combine_statuses(checks),
        "batch": block.get("batch"),
        "device": block.get("device"),
        "p50_seconds": block.get("p50_seconds"),
        "p95_seconds": block.get("p95_seconds"),
        "seconds_per_window_p50": block.get("seconds_per_window_p50"),
        "physical_inverse_and_CPU_output_included": block.get("physical_inverse_and_CPU_output_included"),
        "model_loading_disk_and_guards_excluded": block.get("model_loading_disk_and_guards_excluded"),
        "sample_count": len(samples) if isinstance(samples, list) else None,
        "timed": block.get("timed"),
        "peak_allocated_vram": block.get("peak_allocated_vram"),
        "peak_reserved_vram": block.get("peak_reserved_vram"),
        "baseline_allocated_vram": block.get("baseline_allocated_vram"),
        "vram_peak_scope": block.get("vram_peak_scope"),
    }


def cost_parity_status(parity: Any) -> dict[str, Any]:
    if not isinstance(parity, dict):
        return {"status": "UNCHECKED", "reason": "row parity missing"}
    normalized = finite(parity.get("normalized_max_abs"))
    physical = finite(parity.get("physical_max_abs"))
    atol = finite(parity.get("normalized_atol")) or 1e-4
    normalized_ok = normalized is not None and abs(normalized) <= atol
    physical_finite_nonnegative = physical is not None and physical >= 0.0
    return {
        "status": check_status(normalized_ok and physical_finite_nonnegative),
        "normalized_acceptance_only": True,
        "physical_delta_descriptive_only": True,
        "physical_finite_nonnegative_status": check_status(physical_finite_nonnegative),
        "raw": parity,
    }


def cost_causal_status(flags: Any, required: bool) -> dict[str, Any]:
    if not required:
        return {"status": "PASS", "required": False, "reason": "not required for this cost part"}
    if not isinstance(flags, dict):
        return {"status": "UNCHECKED", "required": True, "reason": "causal flags missing"}
    status = flags.get("status")
    causal_expected = flags.get("causal_expected")
    max_delta = finite(flags.get("maximum_anchor_latent_delta"))
    atol = finite(flags.get("atol")) or 1e-5
    encoder_unchanged = flags.get("encoder_unchanged")
    if encoder_unchanged is None and flags.get("encoder_state_before") and flags.get("encoder_state_after"):
        encoder_unchanged = flags.get("encoder_state_before") == flags.get("encoder_state_after")
    if causal_expected is False and status == "FULL_CONTEXT_REVISION_DIAGNOSTIC":
        ok = True if encoder_unchanged is None else bool(encoder_unchanged)
        return {"status": check_status(ok), "required": True, "causal_expected": False,
                "diagnostic_not_causal_failure": True, "raw": flags}
    if causal_expected is True:
        ok = status == "PASS" and max_delta is not None and max_delta <= atol and (encoder_unchanged is True or encoder_unchanged is None)
        return {"status": check_status(ok), "required": True, "causal_expected": True,
                "max_delta": max_delta, "atol": atol, "encoder_unchanged": encoder_unchanged, "raw": flags}
    return {"status": "UNCHECKED", "required": True, "reason": "causal_expected schema missing", "raw": flags}


def cost_diagnostic_row_status(row: dict[str, Any]) -> dict[str, Any]:
    parity = cost_parity_status(row.get("parity"))
    finite_checks = []
    for key in ("normalized_prediction_max_abs_vs_same_precision_original_microbatch8", "latent_max_abs_vs_same_precision_original_microbatch8"):
        if key in row:
            x = finite(row.get(key))
            finite_checks.append(check_status(x is not None and x >= 0.0))
    for key, value in (row.get("cross_precision_deltas") or {}).items():
        x = finite(value)
        finite_checks.append(check_status(x is not None and x >= 0.0))
    unchanged = row.get("encoder_unchanged")
    if unchanged is None and row.get("encoder_state_before") and row.get("encoder_state_after"):
        unchanged = row.get("encoder_state_before") == row.get("encoder_state_after")
    finite_checks.append(check_status(unchanged is True))
    finite_checks.append(check_status(row.get("scores_computed") is False))
    return {"status": combine_statuses(finite_checks), "parity_precision_status": parity.get("status"),
            "parity_check": parity, "diagnostic_finite_checks": finite_checks}


def top_parity_flags_status(value: Any) -> dict[str, Any]:
    if isinstance(value, list):
        ok = len(value) == 0
        count = len(value)
    else:
        numeric = finite(value)
        ok = numeric == 0.0 if numeric is not None else None
        count = numeric
    return {"status": check_status(ok), "parity_flags": value, "flag_count": count}


def verify_cost_closure() -> dict[str, Any]:
    runs = []
    statuses: list[str] = []
    expected_rows = {"e2_cpu": 16, "e4_gpu": 72, "e5_gpu": 32, "e5_precision_gpu": 6}
    for path in sorted(RESULTS.glob("COST_CLOSURE*/RESULT.json")):
        data = read_json(path)
        if data.get("experiment") != "COST_CLOSURE":
            continue
        part = str(data.get("part") or "")
        budget_check = verify_file_hash(data.get("budget_ledger"), data.get("budget_ledger_sha256"))
        budget_closed = None
        if budget_check.get("status") == "PASS" and budget_check.get("path"):
            try:
                budget_data = read_json(Path(str(budget_check["path"])))
                budget_closed = budget_data.get("status") in {"COMPLETE", "FAILED", "BUDGET_HELD", "FAILED_PRESERVED"}
            except Exception:
                budget_closed = False
        dynamic_expected_rows = expected_rows.get(part)
        if part == "e5_gpu" and data.get("reused_Jena_cost"):
            dynamic_expected_rows = 16
        run = {
            "path": str(path),
            "sha256": sha256(path),
            "part": part,
            "status": data.get("status"),
            "plan_hash_check": verify_file_hash(data.get("plan_path"), data.get("plan_sha256")),
            "gate_hash_check": verify_file_hash(data.get("gate_path"), data.get("gate_sha256")),
            "budget_ledger_hash_check": budget_check,
            "budget_ledger_closed_status": check_status(budget_closed),
            "source_pin_check": verify_source_pins(data.get("pins") or data.get("source_pins")),
            "result_status_check": check_status(data.get("status") == "COMPLETE_COST_ONLY"),
            "no_science_update_check": check_status(data.get("science_updates") == 0 and data.get("scores_computed") is False),
            "top_parity_flags_check": top_parity_flags_status(data.get("parity_flags")),
            "row_count": len(data.get("rows", []) or []),
            "expected_row_count": dynamic_expected_rows,
            "row_count_status": check_status(len(data.get("rows", []) or []) == dynamic_expected_rows if dynamic_expected_rows is not None else None),
            "timer_contract": data.get("timer_contract"),
            "cached_head_contract": data.get("cached_head_contract"),
            "runtime_comparison_limit": data.get("runtime_comparison_limit"),
            "reused_BOLT_cost": data.get("reused_BOLT_cost"),
            "reused_Jena_cost": data.get("reused_Jena_cost"),
            "precision_diagnostic_proof": data.get("precision_diagnostic_proof"),
            "measured_conditions_subset": data.get("measured_conditions_subset"),
            "rows": [],
        }
        statuses.extend([
            run["plan_hash_check"]["status"], run["gate_hash_check"]["status"],
            run["budget_ledger_hash_check"]["status"], run["budget_ledger_closed_status"],
            run["source_pin_check"]["status"], run["result_status_check"],
            run["no_science_update_check"], run["row_count_status"],
        ])
        for row in data.get("rows", []) or []:
            parity = cost_parity_status(row.get("parity"))
            measurement = str(row.get("measurement") or "")
            if part == "e5_precision_gpu":
                diag = cost_diagnostic_row_status(row)
                e2e = {"status": "PASS", "reason": "precision diagnostic row has no timing block by schema"}
                cached = {"status": "PASS", "reason": "precision diagnostic row has no cached timing block by schema"}
                causal = {"status": "PASS", "required": False, "reason": "precision diagnostic, not causality timing"}
                row_status = diag["status"]
            elif measurement == "CACHED_PREFIX_HEAD_ONLY":
                e2e = {"status": "PASS", "reason": "end_to_end not part of cached-prefix-head measurement schema"}
                cached = timing_block_status(row.get("cached_head_only"))
                causal = cost_causal_status(row.get("trained_encoder_history_causality") or row.get("causalBflags"), required=False)
                row_status = combine_statuses(cached, causal)
            elif measurement == "REPRESENTATIVE_FULL_PREFIX_NATIVE":
                e2e = timing_block_status(row.get("end_to_end"))
                cached = {"status": "PASS", "reason": "cached_head_only not part of representative full-prefix measurement schema"}
                causal = cost_causal_status(row.get("trained_encoder_history_causality") or row.get("causalBflags"), required=False)
                row_status = combine_statuses(e2e, causal)
            else:
                e2e = timing_block_status(row.get("end_to_end"))
                cached = timing_block_status(row.get("cached_head_only")) if row.get("cached_head_only") is not None else {"status": "UNCHECKED", "reason": "cached_head_only missing"}
                causal = cost_causal_status(row.get("trained_encoder_history_causality") or row.get("causalBflags"), required=part == "e5_gpu")
                row_status = combine_statuses(e2e, cached, causal)
            parity_precision_status = parity.get("status", "UNCHECKED")
            statuses.append(row_status)
            run["rows"].append({
                "experiment": row.get("experiment"),
                "dataset": row.get("dataset"),
                "method": row.get("method"),
                "condition": row.get("condition"),
                "seed": row.get("seed"),
                "measurement": row.get("measurement"),
                "batch": row.get("batch"),
                "cudnn_allow_tf32": row.get("cudnn_allow_tf32"),
                "cuda_matmul_allow_tf32": row.get("cuda_matmul_allow_tf32"),
                "origin_hash": row.get("origin_hash"),
                "unique_inputs": row.get("unique_inputs"),
                "repeated_inputs": row.get("repeated_inputs"),
                "binding": row.get("binding"),
                "parity_check": parity,
                "parity_precision_status": parity_precision_status,
                "end_to_end_timing_check": e2e,
                "cached_head_timing_check": cached,
                "causalBflags_check": causal,
                "precision_diagnostic_check": cost_diagnostic_row_status(row) if part == "e5_precision_gpu" else None,
                "original_export_full_origin_replay": row.get("original_export_full_origin_replay"),
                "inference_precision": row.get("inference_precision"),
                "actual_forward_microbatches": row.get("actual_forward_microbatches"),
                "original_run_device": row.get("original_run_device"),
                "verification_status": row_status,
            })
        run["precision_parity_status"] = combine_statuses(run["top_parity_flags_check"], [row.get("parity_check") for row in run["rows"]])
        run["verification_status"] = combine_statuses(run["result_status_check"], run["no_science_update_check"], run["plan_hash_check"], run["gate_hash_check"], run["budget_ledger_hash_check"], run["budget_ledger_closed_status"], run["source_pin_check"], run["row_count_status"], [row.get("verification_status") for row in run["rows"]])
        statuses.append(run["verification_status"])
        runs.append(run)
    return {"status": combine_statuses(statuses), "run_count": len(runs), "runs": runs, "cost_only_not_science": True}

def run(args: argparse.Namespace) -> int:
    if args.dry_run or not args.run:
        preview = {
            "mode": "dry_run",
            "would_verify": ["E3/E4 NPZ metrics", "E5 NPZ metrics and selection", "E5_BOLT attachment bundles", "E6 graph/result audits", "COST_CLOSURE timing/parity/causalB audits"],
            "would_write": "FINAL_VERIFY_v3.json inside --out-dir",
        }
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return 0
    out_dir = Path(args.out_dir).resolve()
    if not out_dir.exists() or not out_dir.is_dir():
        raise FileNotFoundError("--out-dir must be an existing fresh v3 report directory")
    if not str(out_dir).startswith(str(HERE.resolve())):
        raise ValueError("--out-dir must stay inside the campaign directory")
    out_path = out_dir / "FINAL_VERIFY_v3.json"
    if out_path.exists():
        raise FileExistsError(out_path)
    start = time.perf_counter()
    with Budget(phase="final_verify_v3", gpu=False, label="FINAL_VERIFY_v3 read-only NumPy verification"):
        protected = access_guard()
        payload = {
            "generated_unix": time.time(),
            "scope": "read-only saved artifact verification; no model inference or TEST access",
            "protected_test_paths_registered": protected,
            "E3_E4": verify_e34(),
            "E5": verify_e5(),
            "E5_BOLT": verify_e5_bolt(),
            "E6": verify_e6(),
            "COST_CLOSURE": verify_cost_closure(),
            "notes": [
                "physical dtype follows saved prediction/mean/std arithmetic before float64 error accumulation",
                "reference latent diagnostic is not a deployable bound",
                "graph true-edge recovery checks are trace audits and not causal proof",
            ],
        }
        payload["overall_status"] = combine_statuses(payload["E3_E4"], payload["E5"], payload["E5_BOLT"], payload["E6"], payload["COST_CLOSURE"])
        payload["wall_seconds_before_write"] = time.perf_counter() - start
        if payload["wall_seconds_before_write"] > args.max_seconds:
            raise TimeoutError(f"verification exceeded {args.max_seconds}s")
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"verify": str(out_path), "sha256": sha256(out_path)}, ensure_ascii=False, indent=2))
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent final v3 verifier.")
    parser.add_argument("--run", action="store_true", help="Write FINAL_VERIFY_v3.json to --out-dir.")
    parser.add_argument("--dry-run", action="store_true", help="Preview verification scope without writing.")
    parser.add_argument("--out-dir", default="", help="Existing final_report_v3 output directory.")
    parser.add_argument("--max-seconds", type=float, default=120.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args(sys.argv[1:])))
