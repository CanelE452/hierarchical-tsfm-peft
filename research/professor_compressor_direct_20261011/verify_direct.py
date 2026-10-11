"""Independent CPU audit for professor compressor direct experiment artifacts.

The verifier is intentionally read-only for science artifacts.  It writes one
new timestamped receipt under ``checkreceipts/`` and does not launch models or
open TEST archives.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / ".cache" / HERE.name
RECEIPT_DIR = HERE / "checkreceipts"
SOURCE_SNAPSHOT_DIR = HERE / "source_snapshots"

PLANNED_DATASETS = ("robin", "jena")
PLANNED_SEEDS = (92601, 92602)
GT_VARIANTS = (False, True)
TS2_DIMS = (16, 320)
FORECAST_ARMS = (
    "RAW_F",
    "MEAN_BLOCK_F",
    "TRANSFER_F",
    "CALIBRATED_F",
    "LATENT_PERSIST_TRANSFER",
    "LATENT_PERSIST_CALIBRATED",
    "TS2_NATIVE_RIDGE",
)


def rel(path: Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return str(path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(jsonable(v) for v in value)
    return value


def write_receipt(payload: dict[str, Any]) -> Path:
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = RECEIPT_DIR / f"direct_verification_{time.time_ns()}.json"
    path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def observed_metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray, std: np.ndarray | None = None) -> dict[str, Any]:
    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError(f"shape mismatch: prediction={prediction.shape}, target={target.shape}, mask={mask.shape}")
    mask = mask.astype(bool)
    if not np.isfinite(prediction).all():
        raise ValueError("prediction contains non-finite values")
    if not np.isfinite(target[mask]).all():
        raise ValueError("observed target contains non-finite values")
    axes = tuple(range(prediction.ndim - 1))
    count = mask.sum(axis=axes)
    valid = count > 0
    error = np.where(mask, prediction.astype(np.float64) - target.astype(np.float64), 0.0)
    squared_sum = np.square(error).sum(axis=axes)
    absolute_sum = np.abs(error).sum(axis=axes)
    channel_mse = np.divide(squared_sum, count, out=np.full(count.shape, np.nan, dtype=np.float64), where=valid)
    channel_mae = np.divide(absolute_sum, count, out=np.full(count.shape, np.nan, dtype=np.float64), where=valid)
    result = {
        "mse": float(np.nanmean(channel_mse)) if valid.any() else None,
        "mae": float(np.nanmean(channel_mae)) if valid.any() else None,
        "channel_mse": [float(v) if np.isfinite(v) else None for v in channel_mse],
        "channel_mae": [float(v) if np.isfinite(v) else None for v in channel_mae],
        "observed_counts": count.astype(np.int64).tolist(),
        "observed_count_total": int(count.sum()),
        "unobserved_channels": np.flatnonzero(~valid).astype(np.int64).tolist(),
    }
    if std is not None:
        result["physical_channel_mae"] = [
            float(a * b) if a is not None and np.isfinite(a) else None
            for a, b in zip(result["channel_mae"], std.astype(np.float64))
        ]
    if prediction.ndim == 3:
        result["origin_count"] = int(prediction.shape[0])
        result["horizon_mse"] = []
        result["horizon_mae"] = []
        for step in range(prediction.shape[1]):
            partial = observed_metrics(prediction[:, step], target[:, step], mask[:, step])
            result["horizon_mse"].append(partial["mse"])
            result["horizon_mae"].append(partial["mae"])
    return result


def close(a: Any, b: Any, atol: float = 1e-8, rtol: float = 1e-8) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return bool(np.isclose(float(a), float(b), atol=atol, rtol=rtol))


def list_active_processes() -> list[dict[str, Any]]:
    script_names = (
        "runner_gt.py",
        "runner_ts2.py",
        "runner_baselines.py",
        "dispatch_direct.py",
        "analyze_direct.py",
        "memory_retry.py",
    )
    if os.name != "nt":
        try:
            result = subprocess.run(["ps", "-eo", "pid,ppid,command"], capture_output=True, text=True, check=False)
            rows = []
            for line in result.stdout.splitlines():
                lower = line.lower()
                if "python" in lower and any(name in line for name in script_names) and "verify_direct.py" not in line:
                    rows.append({"line": line})
            return rows
        except Exception as exc:
            return [{"error": repr(exc)}]
    script = (
        "Get-CimInstance Win32_Process | "
        "Select-Object ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True, check=False)
        text = result.stdout.strip()
        if not text:
            return []
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            parsed = [parsed]
        rows = []
        for item in parsed:
            command = item.get("CommandLine") or ""
            name = (item.get("Name") or "").lower()
            if "python" in name and any(script_name in command for script_name in script_names) and "verify_direct.py" not in command:
                rows.append(item)
        return rows
    except Exception as exc:
        return [{"error": repr(exc)}]


def source_pin_check(result: dict[str, Any]) -> list[dict[str, Any]]:
    checks = []
    for path_text, expected in sorted(result.get("source_pins", {}).items()):
        path = ROOT / path_text
        actual = digest(path) if path.exists() else None
        preserved_snapshot = None
        if expected and actual != expected and SOURCE_SNAPSHOT_DIR.exists():
            for candidate in sorted(SOURCE_SNAPSHOT_DIR.rglob("*")):
                if candidate.is_file() and digest(candidate) == expected:
                    preserved_snapshot = rel(candidate)
                    break
        checks.append({
            "path": path_text,
            "exists": path.exists(),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "match": bool(actual == expected),
            "preserved_snapshot_path": preserved_snapshot,
            "provenance_preserved": bool(actual == expected or preserved_snapshot),
        })
    return checks


def load_npz(path_text: str) -> dict[str, np.ndarray]:
    path = ROOT / path_text
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def artifact_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def compare_metric_dict(measured: dict[str, Any], reported: dict[str, Any], arm: str) -> dict[str, Any]:
    scalar = {
        "mse": close(measured.get("mse"), reported.get("mse")),
        "mae": close(measured.get("mae"), reported.get("mae")),
    }
    vector = {}
    for key in ("channel_mse", "channel_mae", "horizon_mse", "horizon_mae", "physical_channel_mae"):
        if key in reported and key in measured:
            left = np.asarray([np.nan if v is None else v for v in measured[key]], dtype=np.float64)
            right = np.asarray([np.nan if v is None else v for v in reported[key]], dtype=np.float64)
            vector[key] = bool(left.shape == right.shape and np.allclose(left, right, equal_nan=True, atol=1e-8, rtol=1e-8))
    return {
        "arm": arm,
        "scalar_match": scalar,
        "vector_match": vector,
        "match": all(scalar.values()) and all(vector.values()),
        "measured_mse": measured.get("mse"),
        "reported_mse": reported.get("mse"),
        "measured_mae": measured.get("mae"),
        "reported_mae": reported.get("mae"),
    }


def state_diff(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor], prefix: str | None = None) -> dict[str, Any]:
    keys = sorted(set(a) & set(b))
    if prefix is not None:
        keys = [key for key in keys if key.startswith(prefix)]
    changed = []
    max_abs = 0.0
    for key in keys:
        if a[key].shape != b[key].shape:
            changed.append(key)
            continue
        diff = float((a[key].detach().cpu() - b[key].detach().cpu()).abs().max())
        max_abs = max(max_abs, diff)
        if diff != 0.0:
            changed.append(key)
    return {
        "keys_checked": len(keys),
        "changed_keys": changed,
        "changed": bool(changed),
        "max_abs": max_abs,
    }


def inspect_gt_checkpoints(result: dict[str, Any]) -> dict[str, Any]:
    training = result.get("training", {})
    calibration = result.get("calibration", {})
    rec_path = ROOT / training.get("checkpoint_path", "")
    cal_path = ROOT / calibration.get("checkpoint_path", "")
    out = {"available": rec_path.exists() and cal_path.exists(), "reconstruction_checkpoint": rel(rec_path), "calibration_checkpoint": rel(cal_path)}
    if not out["available"]:
        return out
    rec = torch.load(rec_path, map_location="cpu", weights_only=False)
    cal = torch.load(cal_path, map_location="cpu", weights_only=False)
    rec_state = rec["state_dict"]
    cal_state = cal["state_dict"]
    out.update({
        "reconstruction_epoch": rec.get("epoch"),
        "reconstruction_updates": rec.get("actual_updates"),
        "calibration_epoch": cal.get("epoch"),
        "calibration_updates": cal.get("actual_updates"),
        "encoder_diff": state_diff(rec_state, cal_state, "encoder."),
        "product_diff": {
            "scalar_product": state_diff(rec_state, cal_state, "product.s"),
            "graph": state_diff(rec_state, cal_state, "product.graph"),
        },
        "decoder_diff": state_diff(rec_state, cal_state, "decoder."),
        "same_native_decoder_family": result.get("shape", {}).get("latent") == [128, result.get("shape", {}).get("input", [None, None])[1], 2],
        "reported_E_and_shared_product_unchanged": calibration.get("E_and_shared_product_unchanged"),
        "reported_native_D_trainable_parameters": calibration.get("native_D_trainable_parameters"),
    })
    return out


def validate_prediction_npz(result_path: Path, result: dict[str, Any], data: dict[str, Any] | None, baseline_arrays: dict[str, np.ndarray] | None) -> dict[str, Any]:
    pred_path = ROOT / result["prediction_path"]
    arrays = load_npz(result["prediction_path"])
    target = arrays["targets"]
    mask = arrays["mask"].astype(bool)
    std = arrays["std"]
    out = {
        "path": rel(pred_path),
        "sha256": digest(pred_path),
        "sha256_match": digest(pred_path) == result.get("prediction_sha256"),
        "keys": sorted(arrays),
        "shape": list(target.shape),
        "mask_observed": int(mask.sum()),
        "targets_finite_on_mask": bool(np.isfinite(target[mask]).all()),
        "predictions_all_finite": {},
        "metric_checks": [],
        "origin_checks": {},
        "same_targets_as_raw_baseline": None,
    }
    if data is not None:
        expected_target, expected_mask = make_windows(data, data["val_origins"], future=True, masked=True)
        out["origin_checks"] = {
            "origins_match_contract": bool(np.array_equal(arrays["origins"], data["val_origins"])),
            "targets_match_contract": bool(np.array_equal(target, expected_target)),
            "mask_match_contract": bool(np.array_equal(mask, expected_mask)),
            "mean_match_contract": bool(np.array_equal(arrays["mean"], data["mean"])),
            "std_match_contract": bool(np.array_equal(std, data["std"])),
        }
    if baseline_arrays is not None:
        out["same_targets_as_raw_baseline"] = bool(
            np.array_equal(target, baseline_arrays["targets"])
            and np.array_equal(mask, baseline_arrays["mask"])
            and np.array_equal(arrays["origins"], baseline_arrays["origins"])
        )
    for arm, reported in sorted(result.get("forecast", {}).items()):
        if arm not in arrays:
            out["metric_checks"].append({"arm": arm, "match": False, "error": "missing prediction array"})
            continue
        prediction = arrays[arm]
        out["predictions_all_finite"][arm] = bool(np.isfinite(prediction).all())
        measured = observed_metrics(prediction, target, mask, std)
        out["metric_checks"].append(compare_metric_dict(measured, reported, arm))
    if "TRANSFER_F" in arrays and "CALIBRATED_F" in arrays:
        delta = np.asarray(arrays["TRANSFER_F"], dtype=np.float64) - np.asarray(arrays["CALIBRATED_F"], dtype=np.float64)
        out["transfer_calibrated_array_equal"] = bool(np.array_equal(arrays["TRANSFER_F"], arrays["CALIBRATED_F"]))
        out["transfer_calibrated_max_abs"] = float(np.max(np.abs(delta))) if delta.size else 0.0
    out["all_metric_checks_pass"] = bool(out["metric_checks"] and all(row.get("match") for row in out["metric_checks"]))
    return out


def validate_diagnostic_npz(result: dict[str, Any], data: dict[str, Any] | None) -> dict[str, Any]:
    if "diagnostic_path" not in result:
        return {"available": False}
    path = ROOT / result["diagnostic_path"]
    if not path.exists():
        return {"available": False, "path": rel(path)}
    arrays = load_npz(result["diagnostic_path"])
    out = {
        "available": True,
        "path": rel(path),
        "sha256": digest(path),
        "sha256_match": digest(path) == result.get("diagnostic_sha256"),
        "keys": sorted(arrays),
        "all_numeric_arrays_finite": True,
    }
    for key, value in arrays.items():
        if np.issubdtype(value.dtype, np.number):
            out["all_numeric_arrays_finite"] = out["all_numeric_arrays_finite"] and bool(np.isfinite(value).all())
    if data is not None and {"indices", "past", "past_mask"}.issubset(arrays):
        origins = data["val_origins"][arrays["indices"].astype(np.int64)]
        expected_past = make_windows(data, origins)
        expected_past_mask = make_windows(data, origins, masked=True)[1]
        out["past_matches_contract"] = bool(np.array_equal(arrays["past"], expected_past))
        out["past_mask_matches_contract"] = bool(np.array_equal(arrays["past_mask"], expected_past_mask))
    return out


def validate_baseline(result_path: Path, result: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    arrays_check = validate_prediction_npz(result_path, result, data, None)
    cost = validate_cost(result)
    return {
        "result_path": rel(result_path),
        "result_sha256": digest(result_path),
        "status": result.get("status"),
        "job": result.get("job"),
        "data_contract_hash_match": result.get("data_contract_hash") == data["contract_hash"],
        "prediction": arrays_check,
        "cost": cost,
        "source_pins": source_pin_check(result),
        "F_guards": {
            "pipeline_parity": result.get("F_pipeline_parity"),
            "guard": result.get("F_guard"),
        },
    }


def validate_cost(result: dict[str, Any]) -> dict[str, Any]:
    rows = result.get("cost", [])
    checks = []
    for row in rows:
        checks.append({
            "method": row.get("method"),
            "batch": row.get("batch"),
            "warmup_is_3": row.get("warmup") == 3,
            "timed_is_10": row.get("timed") == 10,
            "seconds_count_is_10": len(row.get("seconds", [])) == 10,
            "parity_within_5e_4": row.get("prediction_parity_max_abs") is not None and row.get("prediction_parity_max_abs") <= 5e-4,
            "has_peak_cuda_allocated": row.get("peak_cuda_allocated_bytes") is not None,
            "has_peak_cuda_reserved": row.get("peak_cuda_reserved_bytes") is not None,
        })
    family = result.get("job", {}).get("family")
    if family == "BASELINES":
        cost_methods = ["RAW_F", "MEAN_BLOCK_F"]
    elif family == "GT":
        cost_methods = ["TRANSFER_F", "CALIBRATED_F"]
    elif family == "TS2":
        cost_methods = ["TRANSFER_F", "CALIBRATED_F", "TS2_NATIVE_RIDGE"]
    else:
        cost_methods = sorted({row.get("method") for row in rows if row.get("method")})
    required = {(method, batch) for method in cost_methods for batch in (1, 32)}
    actual = {(row.get("method"), row.get("batch")) for row in rows}
    forecast_arms = set(result.get("forecast", {}))
    missing = required - actual
    return {
        "rows": len(rows),
        "checks": checks,
        "required_method_batch_pairs": sorted([list(x) for x in required], key=str),
        "actual_method_batch_pairs": sorted([list(x) for x in actual], key=str),
        "missing_method_batch_pairs": sorted([list(x) for x in missing], key=str),
        "forecast_arms_without_cost": sorted(forecast_arms - set(cost_methods)),
        "coverage_pass": not missing if required else bool(rows),
        "all_rows_pass": all(
            row["warmup_is_3"] and row["timed_is_10"] and row["seconds_count_is_10"]
            and row["parity_within_5e_4"] and row["has_peak_cuda_allocated"]
            for row in checks
        ) if checks else False,
    }


def validate_training_curves(result_path: Path, result: dict[str, Any]) -> dict[str, Any]:
    job = result.get("job", {})
    out: dict[str, Any] = {"job_id": job.get("id"), "family": job.get("family")}
    if job.get("family") == "GT":
        if job.get("attempt"):
            rec_curve = result_path.parent / "attempts" / job["attempt"] / "reconstruction_curve.json"
            cal_curve = result_path.parent / "attempts" / job["attempt"] / "calibration_curve.json"
        else:
            rec_curve = result_path.parent / "reconstruction_curve.json"
            cal_curve = result_path.parent / "calibration_curve.json"
        rec_rows = read_json(rec_curve) if rec_curve.exists() else []
        cal_rows = read_json(cal_curve) if cal_curve.exists() else []
        out.update({
            "reconstruction_curve": rel(rec_curve),
            "reconstruction_curve_exists": rec_curve.exists(),
            "reconstruction_rows": len(rec_rows),
            "reported_reconstruction_epochs": result.get("training", {}).get("epochs"),
            "reported_reconstruction_updates": result.get("training", {}).get("actual_optimizer_updates"),
            "last_reconstruction_updates": rec_rows[-1].get("actual_updates") if rec_rows else None,
            "calibration_curve": rel(cal_curve),
            "calibration_curve_exists": cal_curve.exists(),
            "calibration_rows": len(cal_rows),
            "reported_calibration_epochs": result.get("calibration", {}).get("epochs"),
            "reported_calibration_updates": result.get("calibration", {}).get("actual_optimizer_updates"),
            "last_calibration_updates": cal_rows[-1].get("actual_updates") if cal_rows else None,
        })
        out["counts_match"] = (
            out["reconstruction_rows"] == out["reported_reconstruction_epochs"]
            and out["last_reconstruction_updates"] == out["reported_reconstruction_updates"]
            and out["calibration_rows"] == out["reported_calibration_epochs"]
            and out["last_calibration_updates"] == out["reported_calibration_updates"]
        )
    elif job.get("family") == "TS2":
        training = result.get("training", {})
        calibration = result.get("calibration", {})
        encoder_curve = artifact_path(training.get("encoder", {}).get("iteration_curve"))
        reconstruction_curve = artifact_path(training.get("decoder_reconstruction", {}).get("learning_curve"))
        calibration_curve = artifact_path(calibration.get("learning_curve"))
        encoder_rows = read_json(encoder_curve) if encoder_curve and encoder_curve.exists() else []
        reconstruction_rows = read_json(reconstruction_curve) if reconstruction_curve and reconstruction_curve.exists() else []
        calibration_rows = read_json(calibration_curve) if calibration_curve and calibration_curve.exists() else []
        out.update({
            "encoder_curve": rel(encoder_curve) if encoder_curve else None,
            "encoder_curve_exists": bool(encoder_curve and encoder_curve.exists()),
            "encoder_rows": len(encoder_rows),
            "reported_encoder_iterations": training.get("encoder", {}).get("configuration", {}).get("n_iters"),
            "last_encoder_iteration": encoder_rows[-1].get("iteration") if encoder_rows else None,
            "reconstruction_curve": rel(reconstruction_curve) if reconstruction_curve else None,
            "reconstruction_curve_exists": bool(reconstruction_curve and reconstruction_curve.exists()),
            "reconstruction_rows": len(reconstruction_rows),
            "reported_reconstruction_epochs": training.get("decoder_reconstruction", {}).get("epochs"),
            "calibration_curve": rel(calibration_curve) if calibration_curve else None,
            "calibration_curve_exists": bool(calibration_curve and calibration_curve.exists()),
            "calibration_rows": len(calibration_rows),
            "reported_calibration_epochs": calibration.get("epochs"),
        })
        out["counts_match"] = (
            out["encoder_rows"] == out["reported_encoder_iterations"]
            and out["last_encoder_iteration"] == out["reported_encoder_iterations"]
            and out["reconstruction_rows"] == out["reported_reconstruction_epochs"]
            and out["calibration_rows"] == out["reported_calibration_epochs"]
        )
    return out


def validate_ts2_protocol(result: dict[str, Any], expected_job: dict[str, Any] | None = None) -> dict[str, Any]:
    job = result.get("job", {})
    expected_batch = (expected_job or {}).get("encoder_batch", 8)
    training = result.get("training", {})
    calibration = result.get("calibration", {})
    encoder = training.get("encoder", {})
    config = encoder.get("configuration", {})
    train_shape = encoder.get("TRAIN_shape") or []
    original_instances = int(train_shape[0]) if len(train_shape) >= 1 else None
    train_length = int(train_shape[1]) if len(train_shape) >= 2 else None
    max_train_length = config.get("max_train_length")
    if original_instances is not None and train_length is not None and max_train_length:
        sections = train_length // int(max_train_length)
        post_split_instances = sections if sections >= 2 else original_instances
        effective_loader_batch = min(int(config.get("batch_size") or 0), post_split_instances)
        cropped_train_length = min(train_length, int(max_train_length))
    else:
        sections = None
        post_split_instances = None
        effective_loader_batch = None
        cropped_train_length = None
    native = training.get("native_forecasting_head", {})
    foundation = training.get("foundation_frozen_verification", {})
    calibration_epoch0_considered = (
        calibration.get("epoch0_evaluated") is True
        or calibration.get("initial_state_evaluated") is True
        or "epoch0_val_macro_mse" in calibration
        or "initial_val_macro_mse" in calibration
        or "initial_state_val_macro_mse" in calibration
        or calibration.get("best_epoch") == 0
    )
    checks = {
        "output_dim_matches_job": config.get("output_dims") == job.get("dim"),
        "hidden_dims_64": config.get("hidden_dims") == 64,
        "depth_10": config.get("depth") == 10,
        "max_train_length_3000": config.get("max_train_length") == 3000,
        "batch_size_matches_protocol": config.get("batch_size") == expected_batch,
        "source_revision_pinned": config.get("source_revision") == "b0088e14a99706c05451316dc6db8d3da9351163",
        "save_load_export_close": training.get("SWA_save_load_actual_max_abs_difference") is not None
        and training.get("SWA_save_load_actual_max_abs_difference") <= 1e-5,
        "future_perturbation_unchanged": training.get("actual_future_perturbation_unchanged") is True,
        "encoder_frozen_during_decoder_training": training.get("encoder_frozen_during_all_decoder_training") is True,
        "foundation_unchanged": foundation.get("unchanged") is True,
        "foundation_gradients_absent": foundation.get("parameter_gradients_absent") is True,
        "foundation_no_backward": foundation.get("F_backward_calls") == 0,
        "original_reconstruction_decoder_preserved": calibration.get("original_decoder_preserved") is True,
        "calibration_epoch0_or_initial_state_considered": calibration_epoch0_considered,
        "encoder_and_F_frozen_during_calibration": calibration.get("encoder_and_F_frozen") is True,
        "native_ridge_math_verified": native.get("ridge_math_verification", {}).get("status") == "PASS",
    }
    return {
        "configuration": {
            "output_dims": config.get("output_dims"),
            "hidden_dims": config.get("hidden_dims"),
            "depth": config.get("depth"),
            "max_train_length": config.get("max_train_length"),
            "batch_size": config.get("batch_size"),
            "expected_batch_size": expected_batch,
            "source_revision": config.get("source_revision"),
        },
        "official_sampling_geometry": {
            "TRAIN_shape": train_shape,
            "max_train_length": max_train_length,
            "sections_by_official_split_rule": sections,
            "post_split_instances": post_split_instances,
            "configured_batch_size": config.get("batch_size"),
            "effective_loader_batch": effective_loader_batch,
            "cropped_train_length_upper_bound": cropped_train_length,
            "instance_contrastive_loss_zero_by_source": effective_loader_batch == 1,
            "temporal_contrastive_loss_available_initially": bool(cropped_train_length and cropped_train_length > 1),
            "source_rule": "official losses.py returns 0 for instance_contrastive_loss when B == 1; temporal loss still applies when T > 1",
        },
        "checks": checks,
        "calibration_epoch0_evidence": {
            "epoch0_evaluated": calibration.get("epoch0_evaluated"),
            "initial_state_evaluated": calibration.get("initial_state_evaluated"),
            "has_epoch0_val_macro_mse": "epoch0_val_macro_mse" in calibration,
            "has_initial_val_macro_mse": "initial_val_macro_mse" in calibration,
            "has_initial_state_val_macro_mse": "initial_state_val_macro_mse" in calibration,
            "best_epoch": calibration.get("best_epoch"),
        },
        "pass": all(checks.values()),
    }


def validate_memory_retry_environment(job: dict[str, Any]) -> dict[str, Any]:
    if not job.get("memory_retry"):
        return {"required": False}
    expected_batch = job.get("encoder_batch", 8)
    expected_settings = {
        "PYTORCH_ALLOC_CONF": "backend:native,garbage_collection_threshold:0.6",
        "TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT": "128",
    }
    path = HERE / "execution_environment" / f"{job['id']}.json"
    out: dict[str, Any] = {
        "required": True,
        "path": rel(path),
        "exists": path.exists(),
        "job_id": job.get("id"),
    }
    if not path.exists():
        out["pass"] = False
        out["reason"] = "missing_execution_environment_receipt"
        return out
    receipt = read_json(path)
    command = receipt.get("command", [])
    runner_path = HERE / "runner_ts2.py"
    wrapper_path = HERE / "memory_retry.py"
    actual_settings = receipt.get("operational_environment", {})
    runner_sha = digest(runner_path) if runner_path.exists() else None
    wrapper_sha = digest(wrapper_path) if wrapper_path.exists() else None
    checks = {
        "job_id_matches": receipt.get("job_id") == job.get("id"),
        "settings_before_torch_import": receipt.get("settings_before_torch_import") is True,
        "allocator_settings_match": actual_settings == expected_settings,
        "cuda_allocator_backend_native": receipt.get("cuda_allocator_backend") == "native",
        "encoder_batch_matches_protocol": receipt.get("encoder_batch") == expected_batch,
        "geometry_or_loss_unchanged": receipt.get("official_geometry_or_loss_changed") is False,
        "command_uses_memory_retry": any(str(item).endswith("memory_retry.py") for item in command),
        "command_passes_dataset": "--dataset" in command and command[command.index("--dataset") + 1] == job.get("dataset"),
        "command_passes_dim": "--dim" in command and command[command.index("--dim") + 1] == str(job.get("dim")),
        "command_passes_seed": "--seed" in command and command[command.index("--seed") + 1] == str(job.get("seed")),
        "command_passes_retry_tag": "--retry-tag" in command and command[command.index("--retry-tag") + 1] == job.get("retry_tag"),
        "command_passes_encoder_batch": "--encoder-batch" in command and command[command.index("--encoder-batch") + 1] == str(expected_batch),
        "runner_sha256_matches_current": receipt.get("runner_sha256") == runner_sha,
        "wrapper_sha256_matches_current": receipt.get("wrapper_sha256") == wrapper_sha,
    }
    out.update({
        "status": receipt.get("status"),
        "started_unix": receipt.get("started_unix"),
        "ended_unix": receipt.get("ended_unix"),
        "result_exists": receipt.get("result_exists"),
        "expected_encoder_batch": expected_batch,
        "batch_adjustment_report": {
            "default_batch": 8,
            "actual_batch": expected_batch,
            "geometry_hidden_depth_max_train_length_iterations_changed": False,
            "negative_pool_and_section_exposure_changed": bool(expected_batch != 8),
            "note": (
                "Batch 4 changes the official TS2Vec mini-batch composition, "
                "including simultaneous section exposure and in-batch negative pairs; "
                "hidden64/depth10/max_train_length3000/iteration budget remain the same."
                if expected_batch != 8
                else "Default batch 8; no batch-size exposure adjustment."
            ),
        },
        "operational_environment": actual_settings,
        "command": command,
        "runner_sha256": receipt.get("runner_sha256"),
        "wrapper_sha256": receipt.get("wrapper_sha256"),
        "checks": checks,
        "pass": all(checks.values()),
    })
    return out


def gt_calibration_decoder_policy(audit: dict[str, Any]) -> dict[str, Any]:
    """Validate GT D behavior under epoch-0-aware calibration selection.

    If epoch 0 wins, the calibrated checkpoint is intentionally the transfer
    decoder.  In that case D must be unchanged and stored TRANSFER/CALIBRATED
    predictions must be exactly identical.  If a positive epoch wins, D must
    change from the reconstruction checkpoint.
    """
    selection = audit.get("calibration_selection", {})
    selected_epoch = selection.get("selected_epoch")
    checkpoint = audit.get("gt_checkpoint_comparison", {})
    decoder_changed = checkpoint.get("decoder_diff", {}).get("changed")
    prediction = audit.get("prediction", {})
    transfer_equal = prediction.get("transfer_calibrated_array_equal")
    max_abs = prediction.get("transfer_calibrated_max_abs")
    if selected_epoch == 0:
        passed = decoder_changed is False and transfer_equal is True
        reason = "epoch0_selected_requires_unchanged_decoder_and_identical_predictions"
    elif isinstance(selected_epoch, (int, float)) and selected_epoch > 0:
        passed = decoder_changed is True
        reason = "positive_epoch_selected_requires_changed_decoder"
    else:
        passed = False
        reason = "missing_selected_epoch"
    return {
        "selected_epoch": selected_epoch,
        "decoder_changed": decoder_changed,
        "transfer_calibrated_array_equal": transfer_equal,
        "transfer_calibrated_max_abs": max_abs,
        "pass": passed,
        "reason": reason,
    }


def ts2_calibration_decoder_policy(audit: dict[str, Any]) -> dict[str, Any]:
    """Validate TS2 decoder-selection semantics with the same epoch-0 rule."""
    calibration = audit.get("ts2_calibration", {})
    best_epoch = calibration.get("best_epoch")
    changed = calibration.get("initial_to_selected_changed")
    prediction = audit.get("prediction", {})
    transfer_equal = prediction.get("transfer_calibrated_array_equal")
    max_abs = prediction.get("transfer_calibrated_max_abs")
    if best_epoch == 0:
        passed = changed is False and transfer_equal is True
        reason = "epoch0_selected_requires_unchanged_decoder_and_identical_predictions"
    elif isinstance(best_epoch, (int, float)) and best_epoch > 0:
        passed = changed is True
        reason = "positive_epoch_selected_requires_changed_decoder"
    else:
        passed = False
        reason = "missing_best_epoch"
    return {
        "best_epoch": best_epoch,
        "initial_to_selected_changed": changed,
        "transfer_calibrated_array_equal": transfer_equal,
        "transfer_calibrated_max_abs": max_abs,
        "pass": passed,
        "reason": reason,
    }


def static_code_inspection() -> dict[str, Any]:
    runner_gt = (HERE / "runner_gt.py").read_text(encoding="utf-8")
    runner_ts2 = (HERE / "runner_ts2.py").read_text(encoding="utf-8")
    models_gt = (HERE / "models_gt.py").read_text(encoding="utf-8")
    data_io = (HERE / "data_io.py").read_text(encoding="utf-8")
    foundation = (HERE / "foundation.py").read_text(encoding="utf-8")
    ts2_adapter = (HERE / "ts2_adapter.py").read_text(encoding="utf-8")
    return {
        "data_io_include_test_false_literal": "include_test=False" in data_io,
        "data_io_no_include_test_true_literal": "include_test=True" not in data_io,
        "runner_gt_future_guard_present": "altered[origin:]" in runner_gt and "torch.equal(a, b)" in runner_gt,
        "runner_gt_native_decoder_after_F_present": "decode_future" in runner_gt and "foundation.forecast" in runner_gt,
        "runner_gt_calibration_freezes_model_then_enables_decoder_parameters": (
            "model.requires_grad_(False)" in runner_gt and "for parameter in model.decoder_parameters()" in runner_gt
        ),
        "runner_gt_replacement_d8_or_level_disabled": (
            ("'replacement_D8_or_LEVEL_G': False" in runner_gt or '"replacement_D8_or_LEVEL_G": False' in runner_gt)
            and "residual_G" not in runner_gt
            and "PCA" not in runner_gt
        ),
        "models_gt_time_down_and_up_present": "temporal_downsample" in models_gt and "temporal_upsample" in models_gt,
        "models_gt_R4_present": "TIME_REDUCTION = 4" in models_gt,
        "models_gt_native_decoder_parameters_present": "def decoder_parameters" in models_gt and "return self.decoder.parameters()" in models_gt,
        "foundation_no_grad_forecast": "@torch.no_grad()" in foundation and "def forecast" in foundation,
        "foundation_verify_unchanged_reports_grad_absent": "parameter_gradients_absent" in foundation and "F_backward_calls" in foundation,
        "runner_ts2_official_hidden_depth_dims": "hidden_dims': 64" in runner_ts2 and "'depth': 10" in runner_ts2 and "output_dims" in ts2_adapter,
        "runner_ts2_save_load_check_present": "Actual trained encoder save/load export mismatch" in runner_ts2,
        "runner_ts2_same_decoder_prototype_check_present": "Calibration changed original reconstruction decoder" in runner_ts2,
        "runner_ts2_future_guard_present": "Origin-after observations alter actual E/F/D inference" in runner_ts2,
    }


def validate_ts2_vendor() -> dict[str, Any]:
    source_root = ROOT / "research" / "level_k_tradeoff_20261011" / "vendor" / "ts2vec"
    prov_path = source_root / "provenance.json"
    if not prov_path.exists():
        return {"available": False}
    receipt = read_json(prov_path)
    checks = []
    for item in receipt.get("files", []):
        path = source_root / item["path"]
        actual = digest(path) if path.exists() else None
        checks.append({
            "path": rel(path),
            "expected_sha256": item.get("sha256"),
            "actual_sha256": actual,
            "match": actual == item.get("sha256"),
            "exists": path.exists(),
        })
    return {
        "available": True,
        "declared_revision": receipt.get("revision") or receipt.get("commit"),
        "unmodified_upstream_files": receipt.get("unmodified_upstream_files"),
        "file_checks": checks,
        "all_file_checks_pass": all(row["match"] for row in checks),
    }


def build_expected_coverage() -> list[dict[str, Any]]:
    protocol_path = HERE / "protocol.json"
    if protocol_path.exists():
        protocol = read_json(protocol_path)
        jobs = protocol.get("jobs", [])
        if jobs:
            return [dict(job) for job in jobs]
    jobs = []
    for dataset in PLANNED_DATASETS:
        for seed in PLANNED_SEEDS:
            jobs.append({"id": f"{dataset}_GT_full_s{seed}", "dataset": dataset, "seed": seed, "family": "GT", "no_spatial": False})
            jobs.append({"id": f"{dataset}_GT_no_spatial_s{seed}", "dataset": dataset, "seed": seed, "family": "GT", "no_spatial": True})
            for dim in TS2_DIMS:
                jobs.append({"id": f"{dataset}_TS2_d{dim}_s{seed}", "dataset": dataset, "seed": seed, "family": "TS2", "dim": dim})
    return jobs


def load_data_contracts() -> tuple[dict[str, Any], dict[str, Any]]:
    sys.path.insert(0, str(HERE))
    from data_io import load_data as direct_load_data, make_windows as direct_make_windows
    globals()["make_windows"] = direct_make_windows
    data = {dataset: direct_load_data(dataset) for dataset in PLANNED_DATASETS}
    data_checks = read_json(HERE / "DATA_CHECKS.json")
    return data, data_checks


def inspect_partial_ts2_caches() -> list[dict[str, Any]]:
    rows = []
    for folder in sorted((CACHE / "runs").glob("*_TS2_*")):
        result = HERE / "runs" / folder.name / "result.json"
        failure = HERE / "runs" / folder.name / "failure.json"
        curve = folder / "encoder_curve.json"
        encoder_updates = None
        if curve.exists():
            try:
                rows_json = read_json(curve)
                encoder_updates = rows_json[-1].get("iteration") if rows_json else 0
            except Exception:
                encoder_updates = "unreadable"
        rows.append({
            "id": folder.name,
            "cache_path": rel(folder),
            "result_exists": result.exists(),
            "failure_exists": failure.exists(),
            "encoder_curve_exists": curve.exists(),
            "encoder_last_iteration": encoder_updates,
            "cache_files": sorted(p.name for p in folder.iterdir() if p.is_file()),
            "status": "complete" if result.exists() else ("failed" if failure.exists() else "partial_or_running"),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="Return non-zero unless all planned jobs are complete and valid.")
    args = parser.parse_args()

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    data: dict[str, Any] = {}
    data_checks: dict[str, Any] = {}
    try:
        data, data_checks = load_data_contracts()
    except Exception as exc:
        errors.append({"stage": "load_data_contracts", "error": repr(exc)})

    receipt: dict[str, Any] = {
        "created_unix": time.time(),
        "scope": "CPU-only independent audit of saved direct-study artifacts; no model training; no TEST archive access",
        "active_processes": list_active_processes(),
        "data_contracts": {},
        "static_code_inspection": static_code_inspection(),
        "ts2_vendor": validate_ts2_vendor(),
        "baseline_audits": [],
        "run_audits": [],
        "coverage": {},
        "partial_ts2_caches": inspect_partial_ts2_caches(),
        "warnings": warnings,
        "errors": errors,
    }

    for dataset, item in data.items():
        check = data_checks.get("datasets", {}).get(dataset, {})
        receipt["data_contracts"][dataset] = {
            "contract_hash": item.get("contract_hash"),
            "x_shape": list(item["x"].shape),
            "mask_shape": list(item["mask"].shape),
            "train_window_count": int(len(item["train_window_starts"])),
            "train_forecast_origin_count": int(len(item["train_forecast_origins"])),
            "val_origin_count": int(len(item["val_origins"])),
            "data_checks_overall_pass": data_checks.get("overall_pass"),
            "no_test_read_contract": check.get("no_test_read_contract"),
            "quality_metrics": check.get("quality_metrics"),
        }

    baseline_arrays: dict[str, dict[str, np.ndarray]] = {}
    for path in sorted((HERE / "baselines").glob("*/result.json")):
        try:
            result = read_json(path)
            dataset = result.get("job", {}).get("dataset")
            audit = validate_baseline(path, result, data[dataset])
            receipt["baseline_audits"].append(audit)
            baseline_arrays[dataset] = load_npz(result["prediction_path"])
        except Exception as exc:
            errors.append({"stage": "baseline", "path": rel(path), "error": repr(exc)})

    for path in sorted((HERE / "runs").glob("*/result.json")):
        try:
            result = read_json(path)
            job = result.get("job", {})
            dataset = job.get("dataset")
            item = data.get(dataset)
            pred_check = validate_prediction_npz(path, result, item, baseline_arrays.get(dataset))
            diag_check = validate_diagnostic_npz(result, item)
            audit = {
                "result_path": rel(path),
                "result_sha256": digest(path),
                "status": result.get("status"),
                "job": job,
                "canonical_id": job.get("canonical_id") or job.get("id"),
                "is_retry_for_canonical": bool(job.get("canonical_id")),
                "data_contract_hash_match": bool(item is not None and result.get("data_contract_hash") == item.get("contract_hash")),
                "prediction": pred_check,
                "diagnostic": diag_check,
                "cost": validate_cost(result),
                "source_pins": source_pin_check(result),
                "training_curves": validate_training_curves(path, result),
                "source_scope": result.get("scope"),
                "calibration_selection": {
                    "selected_epoch": result.get("calibration", {}).get("selected_epoch"),
                    "best_epoch": result.get("calibration", {}).get("best_epoch"),
                    "selected_optimizer_updates": result.get("calibration", {}).get("selected_optimizer_updates"),
                },
            }
            if job.get("family") == "GT":
                audit["gt_checkpoint_comparison"] = inspect_gt_checkpoints(result)
                audit["gt_calibration_decoder_policy"] = gt_calibration_decoder_policy(audit)
                guards_path = path.parent / "INFERENCE_GUARDS.json"
                audit["future_guard_receipt"] = {
                    "path": rel(guards_path),
                    "exists": guards_path.exists(),
                    "value": read_json(guards_path) if guards_path.exists() else None,
                }
            if job.get("family") == "TS2":
                audit["ts2_training"] = result.get("training", {})
                audit["ts2_calibration"] = result.get("calibration", {})
                audit["ts2_protocol_checks"] = validate_ts2_protocol(result)
                audit["ts2_calibration_decoder_policy"] = ts2_calibration_decoder_policy(audit)
            receipt["run_audits"].append(audit)
        except Exception as exc:
            errors.append({"stage": "run", "path": rel(path), "error": repr(exc)})

    expected = build_expected_coverage()
    expected_by_id = {job["id"]: job for job in expected}
    for audit in receipt["run_audits"]:
        if audit.get("job", {}).get("family") == "TS2":
            audit["ts2_protocol_checks"] = validate_ts2_protocol(
                audit.get("raw_result", {}) if "raw_result" in audit else read_json(ROOT / audit["result_path"]),
                expected_by_id.get(audit.get("job", {}).get("id")),
            )
    complete_run_audits = [audit for audit in receipt["run_audits"] if audit.get("status") == "complete"]
    matching_by_expected_id: dict[str, list[dict[str, Any]]] = {job["id"]: [] for job in expected}
    for audit in complete_run_audits:
        physical_id = audit.get("job", {}).get("id")
        if physical_id in matching_by_expected_id:
            matching_by_expected_id[physical_id].append(audit)

    def audit_sort_key(audit: dict[str, Any]) -> tuple[int, float]:
        path = ROOT / audit.get("result_path", "")
        try:
            mtime = path.stat().st_mtime
        except Exception:
            mtime = 0.0
        return (1 if audit.get("is_retry_for_canonical") else 0, mtime)

    selected_by_expected_id: dict[str, dict[str, Any]] = {}
    for expected_id, candidates in matching_by_expected_id.items():
        if candidates:
            selected_by_expected_id[expected_id] = sorted(candidates, key=audit_sort_key, reverse=True)[0]
    selected_run_audits = list(selected_by_expected_id.values())
    selected_object_ids = {id(audit) for audit in selected_run_audits}
    extra_completed_run_audits = [audit for audit in complete_run_audits if id(audit) not in selected_object_ids]

    coverage_rows = []
    for job in expected:
        result_path = HERE / "runs" / job["id"] / "result.json"
        failure_path = HERE / "runs" / job["id"] / "failure.json"
        cache_path = CACHE / "runs" / job["id"]
        environment_check = validate_memory_retry_environment(job)
        matches = matching_by_expected_id[job["id"]]
        selected = selected_by_expected_id.get(job["id"])
        coverage_rows.append({
            "job": job,
            "status": "complete" if selected else ("failed" if failure_path.exists() else ("partial_or_running" if cache_path.exists() else "not_started")),
            "result_path": rel(result_path) if result_path.exists() else None,
            "failure_path": rel(failure_path) if failure_path.exists() else None,
            "cache_path": rel(cache_path) if cache_path.exists() else None,
            "selected_result_path": selected.get("result_path") if selected else None,
            "matching_completed_result_paths": [audit.get("result_path") for audit in matches],
            "memory_retry_environment": environment_check,
        })
    baseline_expected = {dataset: HERE / "baselines" / dataset / "result.json" for dataset in PLANNED_DATASETS}
    receipt["coverage"] = {
        "planned_representation_fits": len(expected),
        "completed_representation_fits": len(selected_by_expected_id),
        "actual_completed_run_results": len(complete_run_audits),
        "extra_completed_run_results": [
            {
                "id": audit.get("job", {}).get("id"),
                "canonical_id": audit.get("canonical_id"),
                "result_path": audit.get("result_path"),
            }
            for audit in extra_completed_run_audits
        ],
        "coverage_rows": coverage_rows,
        "baseline_expected": {dataset: {"result_path": rel(path), "exists": path.exists()} for dataset, path in baseline_expected.items()},
        "baseline_completed": sorted(dataset for dataset, path in baseline_expected.items() if path.exists()),
        "complete_all_planned": len(selected_by_expected_id) == len(expected) and all(path.exists() for path in baseline_expected.values()),
    }

    summary_audits = receipt["baseline_audits"] + selected_run_audits
    all_current_source_pins_ok = all(
        all(row.get("match") for row in audit.get("source_pins", []))
        for audit in summary_audits
    )
    all_source_provenance_ok = all(
        all(row.get("provenance_preserved") for row in audit.get("source_pins", []))
        for audit in summary_audits
    )
    all_metric_ok = all(audit.get("prediction", {}).get("all_metric_checks_pass") for audit in summary_audits)
    all_cost_ok = all(audit.get("cost", {}).get("coverage_pass") and audit.get("cost", {}).get("all_rows_pass") for audit in summary_audits)
    all_diagnostics_ok = all(
        (
            audit.get("diagnostic", {}).get("available") is not True
            or (
                audit.get("diagnostic", {}).get("sha256_match") is True
                and audit.get("diagnostic", {}).get("all_numeric_arrays_finite") is True
                and audit.get("diagnostic", {}).get("past_matches_contract", True) is True
                and audit.get("diagnostic", {}).get("past_mask_matches_contract", True) is True
            )
        )
        for audit in selected_run_audits
    )
    all_gt_checkpoints_ok = all(
        (
            audit.get("job", {}).get("family") != "GT"
            or (
                audit.get("gt_checkpoint_comparison", {}).get("available") is True
                and audit.get("gt_checkpoint_comparison", {}).get("encoder_diff", {}).get("changed") is False
                and audit.get("gt_checkpoint_comparison", {}).get("product_diff", {}).get("scalar_product", {}).get("changed") is False
                and audit.get("gt_checkpoint_comparison", {}).get("product_diff", {}).get("graph", {}).get("changed") is False
                and audit.get("gt_calibration_decoder_policy", {}).get("pass") is True
                and audit.get("future_guard_receipt", {}).get("value", {}).get("future_target_never_passed") is True
                and audit.get("future_guard_receipt", {}).get("value", {}).get("past_only_repeat_bitwise") is True
                and audit.get("future_guard_receipt", {}).get("value", {}).get("replacement_D8_or_LEVEL_G") is False
                and audit.get("future_guard_receipt", {}).get("value", {}).get("unchanged") is True
                and audit.get("future_guard_receipt", {}).get("value", {}).get("parameter_gradients_absent") is True
                and audit.get("future_guard_receipt", {}).get("value", {}).get("F_backward_calls") == 0
            )
        )
        for audit in selected_run_audits
    )
    all_ts2_protocol_ok = all(
        audit.get("job", {}).get("family") != "TS2"
        or (
            audit.get("ts2_protocol_checks", {}).get("pass") is True
            and audit.get("ts2_calibration_decoder_policy", {}).get("pass") is True
        )
        for audit in selected_run_audits
    )
    selected_has_ts2 = any(audit.get("job", {}).get("family") == "TS2" for audit in selected_run_audits)
    ts2_vendor_ok = (receipt["ts2_vendor"].get("all_file_checks_pass") is True) if selected_has_ts2 else True
    all_started_memory_retry_environments_ok = all(
        row.get("memory_retry_environment", {}).get("required") is not True
        or not row.get("memory_retry_environment", {}).get("exists")
        or row.get("memory_retry_environment", {}).get("pass") is True
        for row in receipt["coverage"]["coverage_rows"]
    )
    static_ok = all(bool(value) for value in receipt["static_code_inspection"].values())
    receipt["summary"] = {
        "data_contracts_loaded": bool(data) and not any(e["stage"] == "load_data_contracts" for e in errors),
        "all_saved_prediction_metrics_recomputed": all_metric_ok,
        "all_current_source_pins_match": all_current_source_pins_ok,
        "all_source_pins_or_snapshots_preserved": all_source_provenance_ok,
        "all_completed_cost_rows_valid": all_cost_ok,
        "all_completed_diagnostics_valid": all_diagnostics_ok,
        "all_selected_gt_checkpoint_guards_pass": all_gt_checkpoints_ok,
        "all_selected_ts2_protocol_checks_pass": all_ts2_protocol_ok,
        "ts2_vendor_hashes_match": ts2_vendor_ok,
        "all_started_memory_retry_environments_valid": all_started_memory_retry_environments_ok,
        "static_code_checks_pass": static_ok,
        "overall_saved_completed_artifacts_pass": (
            bool(data)
            and all_metric_ok
            and all_source_provenance_ok
            and all_cost_ok
            and all_diagnostics_ok
            and all_gt_checkpoints_ok
            and all_ts2_protocol_ok
            and ts2_vendor_ok
            and all_started_memory_retry_environments_ok
            and static_ok
            and not errors
        ),
        "campaign_complete": receipt["coverage"]["complete_all_planned"],
        "result_scope": "partial" if not receipt["coverage"]["complete_all_planned"] else "all planned artifacts present",
    }

    receipt_path = write_receipt(receipt)
    print(json.dumps({
        "receipt": rel(receipt_path),
        "overall_saved_completed_artifacts_pass": receipt["summary"]["overall_saved_completed_artifacts_pass"],
        "campaign_complete": receipt["summary"]["campaign_complete"],
        "completed_representation_fits": receipt["coverage"]["completed_representation_fits"],
        "planned_representation_fits": receipt["coverage"]["planned_representation_fits"],
        "baseline_completed": receipt["coverage"]["baseline_completed"],
        "active_processes": len(receipt["active_processes"]),
        "errors": len(errors),
    }, ensure_ascii=False, indent=2))
    if args.strict and not receipt["summary"]["campaign_complete"]:
        return 2
    return 0 if receipt["summary"]["overall_saved_completed_artifacts_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
