"""Independently verify saved Robin readout fits without solving or calling F."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / ".cache" / HERE.name


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.warnings: list[str] = []

    def check(self, name: str, passed: bool, **details: Any) -> None:
        self.checks.append({"name": name, "passed": bool(passed), **details})

    def close(self, name: str, actual: Any, expected: Any, atol: float = 2e-10, rtol: float = 2e-10) -> None:
        a = np.asarray(actual)
        b = np.asarray(expected)
        passed = a.shape == b.shape and np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False)
        difference = float(np.max(np.abs(a.astype(float) - b.astype(float)))) if a.shape == b.shape and a.size else None
        self.check(name, passed, max_abs_difference=difference, atol=atol, rtol=rtol)

    def equal_array(self, name: str, actual: np.ndarray, expected: np.ndarray) -> None:
        self.check(name, actual.dtype == expected.dtype and np.array_equal(actual, expected),
                   actual_shape=list(actual.shape), expected_shape=list(expected.shape),
                   actual_dtype=str(actual.dtype), expected_dtype=str(expected.dtype))

    def metric(self, name: str, recorded: dict, recomputed: dict) -> None:
        for key in recomputed:
            self.check(f"{name}.{key}.present", key in recorded)
            if key in recorded:
                self.close(f"{name}.{key}", recorded[key], recomputed[key])

    def summary(self) -> dict:
        failures = [item["name"] for item in self.checks if not item["passed"]]
        return {"status": "PASS" if not failures else "FAIL", "checks_count": len(self.checks),
                "failures_count": len(failures), "failures": failures, "warnings": self.warnings,
                "scope": "Saved arrays, metrics, scales, normal equations and choices only; no fit, F, GPU or TEST execution",
                "checks": self.checks}


def metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict:
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("Prediction, target and mask must have equal shape")
    observed = mask.astype(bool)
    if not np.isfinite(prediction).all() or not np.isfinite(target[observed]).all():
        raise ValueError("Nonfinite prediction or observed target")
    counts = observed.sum(axis=(0, 1))
    valid = counts > 0
    error = np.where(observed, prediction.astype(np.float64) - target.astype(np.float64), 0.0)
    mse = np.sum(error * error, axis=(0, 1)) / np.maximum(counts, 1)
    mae = np.sum(np.abs(error), axis=(0, 1)) / np.maximum(counts, 1)
    signed = np.sum(error, axis=(0, 1)) / np.maximum(counts, 1)
    return {"mse": float(np.mean(mse[valid])), "mae": float(np.mean(mae[valid])),
            "signed_mean_error": float(np.mean(signed[valid])), "channel_mse": mse.tolist(),
            "channel_mae": mae.tolist(), "target_counts": counts.tolist(),
            "target_count": int(counts.sum()), "origins": len(prediction)}


def horizon_metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict:
    return {"all": metrics(prediction, target, mask),
            "h1_8": metrics(prediction[:, :8], target[:, :8], mask[:, :8]),
            "h9_48": metrics(prediction[:, 8:48], target[:, 8:48], mask[:, 8:48])}


def features(history: np.ndarray, forecast: np.ndarray, arm: dict) -> np.ndarray:
    lag = int(arm["lag"])
    if lag == 0:
        return forecast.astype(np.float64)
    context = np.concatenate((history[:, -lag:], forecast), axis=1)
    horizon = forecast.shape[1]
    blocks = [context[:, lag - shift:lag - shift + horizon].astype(np.float64)
              for shift in range(lag + 1)]
    if arm["support"] == "lag_minus_current":
        return np.concatenate([block - blocks[0] for block in blocks[1:]], axis=2)
    return np.concatenate(blocks, axis=2)


def support(arm: dict, dimension: int, channel: int, channels: int) -> np.ndarray:
    if arm["support"] == "diagonal":
        return np.arange(channel, dimension, channels)
    return np.arange(dimension)


def prediction_from_coeff(feature: np.ndarray, coefficient: np.ndarray, raw: np.ndarray, arm: dict) -> np.ndarray:
    branch = np.einsum("btd,dc->btc", feature.astype(np.float64), coefficient.astype(np.float64)).astype(np.float32)
    return (raw + branch).astype(np.float32) if arm["prior"] == "identity" else branch


def rms_scale(feature: np.ndarray, mask: np.ndarray) -> np.ndarray:
    row_mask = np.any(mask.astype(bool), axis=2)
    return np.maximum(np.sqrt(np.mean(feature[row_mask] ** 2, axis=0)), 1e-12)


def check_normal_equations(audit: Audit, label: str, arm: dict, feature: np.ndarray,
                           target: np.ndarray, mask: np.ndarray, raw: np.ndarray,
                           coefficients: np.ndarray, scale: np.ndarray, lambdas: np.ndarray) -> None:
    audit.close(f"{label}.train_only_rms", scale, rms_scale(feature, mask), atol=5e-13, rtol=5e-13)
    expected_dimension = feature.shape[-1]
    expected_shape = (len(lambdas), expected_dimension, target.shape[-1])
    audit.check(f"{label}.coefficient_shape", coefficients.shape == expected_shape,
                actual=list(coefficients.shape), expected=list(expected_shape))
    audit.check(f"{label}.coefficient_fp64", coefficients.dtype == np.float64)
    audit.check(f"{label}.scale_fp64", scale.dtype == np.float64)
    if coefficients.shape != expected_shape:
        return
    response = target.astype(np.float64)
    if arm["prior"] == "identity":
        response = response - raw.astype(np.float64)
    channels = target.shape[-1]
    for channel in range(channels):
        indices = support(arm, expected_dimension, channel, channels)
        outside = np.ones(expected_dimension, dtype=bool)
        outside[indices] = False
        if outside.any():
            audit.check(f"{label}.channel{channel}.diagonal_support", np.count_nonzero(coefficients[:, outside, channel]) == 0)
        rows = mask[:, :, channel].astype(bool)
        x = feature[rows][:, indices] / scale[indices]
        y = response[:, :, channel][rows]
        gram = x.T @ x
        rhs = x.T @ y
        for index, lam in enumerate(lambdas):
            beta = coefficients[index, indices, channel] * scale[indices]
            left = gram @ beta + float(lam) * beta
            residual = np.max(np.abs(left - rhs))
            denominator = np.linalg.norm(gram, ord=np.inf) * np.linalg.norm(beta, ord=np.inf)
            denominator += abs(float(lam)) * np.linalg.norm(beta, ord=np.inf) + np.linalg.norm(rhs, ord=np.inf)
            relative = float(residual / max(denominator, 1e-30))
            audit.check(f"{label}.channel{channel}.lambda{index}.normal_equation", relative < 2e-10,
                        relative_residual=relative, observations=int(rows.sum()), lambda_value=float(lam))


def effective_taps(coefficient: np.ndarray, arm: dict, channels: int) -> np.ndarray:
    if arm["support"] == "lag_minus_current":
        lagged = coefficient.reshape(int(arm["lag"]), channels, channels)
        taps = np.concatenate((-lagged.sum(axis=0, keepdims=True), lagged), axis=0)
    else:
        taps = coefficient.reshape(int(arm["lag"]) + 1, channels, channels).copy()
    if arm["prior"] == "identity":
        taps[0] += np.eye(channels)
    return taps


def check_feature_causality(audit: Audit, arm: dict, history: np.ndarray, forecast: np.ndarray) -> None:
    original = features(history[:2], forecast[:2], arm)
    changed = forecast[:2].copy()
    changed[:, 12:] += np.float32(19.0)
    perturbed = features(history[:2], changed, arm)
    audit.equal_array(f"{arm['id']}.features_no_future_lookahead", original[:, :12], perturbed[:, :12])
    if arm["support"] == "lag_minus_current":
        shift = np.arange(history.shape[-1], dtype=np.float64)[None, None, :] / 16.0
        shifted = features(history[:2].astype(np.float64) + shift, forecast[:2].astype(np.float64) + shift, arm)
        audit.close(f"{arm['id']}.dc_feature_invariance", original, shifted, atol=2e-14, rtol=0)


def check_horizon_metric(audit: Audit, label: str, recorded: dict, recomputed: dict) -> None:
    for group, scores in recomputed.items():
        audit.check(f"{label}.{group}.present", group in recorded)
        if group in recorded:
            audit.metric(f"{label}.{group}", recorded[group], scores)


def verify_preservation(audit: Audit, include_full: bool) -> None:
    seal_path = HERE / "source_seal.json"
    audit.check("source_seal.exists", seal_path.is_file())
    if seal_path.is_file():
        seal = read_json(seal_path)
        for relative, receipt in seal["files"].items():
            expected = receipt if isinstance(receipt, str) else receipt["sha256"]
            path = resolve(relative)
            audit.check(f"source_seal.{relative}", path.is_file() and sha_file(path) == expected)
    preservation_path = HERE / "preservation.before.json"
    audit.check("preservation_before.exists", preservation_path.is_file())
    if not preservation_path.is_file():
        return
    preservation = read_json(preservation_path)
    if include_full:
        for relative, receipt in preservation["files"].items():
            path = resolve(relative)
            audit.check(f"preserved.{relative}", path.is_file() and path.stat().st_size == receipt["bytes"]
                        and sha_file(path) == receipt["sha256"])
    history = preservation.get("history")
    if history:
        path = resolve(history["path"])
        prefix = path.read_bytes()[:int(history["bytes"])] if path.is_file() else b""
        audit.check("preserved.history_prefix", len(prefix) == int(history["bytes"])
                    and hashlib.sha256(prefix).hexdigest() == history["sha256"])
    actual_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    audit.check("preserved.git_head", actual_head == preservation["git_head"], actual=actual_head)


def check_receipt(audit: Audit, label: str, receipt: dict) -> Path:
    path = resolve(receipt["path"])
    audit.check(f"{label}.exists", path.is_file())
    if not path.is_file():
        raise FileNotFoundError(path)
    audit.check(f"{label}.sha256", sha_file(path) == receipt["sha256"])
    if "bytes" in receipt:
        audit.check(f"{label}.bytes", path.stat().st_size == int(receipt["bytes"]))
    return path


def check_fit_details(audit: Audit, label: str, details: dict, feature: np.ndarray,
                      mask: np.ndarray, arm: dict) -> None:
    row_mask = mask.astype(bool).any(axis=2)
    dimension, channels = feature.shape[-1], mask.shape[-1]
    parameter_count = sum(len(support(arm, dimension, channel, channels)) for channel in range(channels))
    unbounded_rms = np.sqrt(np.mean(feature[row_mask] ** 2, axis=0))
    expected = {"scale_rows": int(row_mask.sum()),
                "scale_row_mask_sha256": sha_array(row_mask.astype(np.uint8)),
                "near_zero_features": int((unbounded_rms <= 1e-12).sum()),
                "support_parameters": parameter_count, "feature_dimension": dimension,
                "target_observations": mask.sum(axis=(0, 1)).tolist()}
    for key, value in expected.items():
        audit.check(f"{label}.{key}", details.get(key) == value)


def verify_study(audit: Audit, result: dict) -> None:
    protocol = read_json(HERE / "protocol.json")
    audit.check("result.complete", result["status"] == "complete")
    audit.check("result.robin_only", result["dataset"] == "robin")
    audit.check("result.protocol_literal", result["protocol"] == protocol)
    check_receipt(audit, "protocol_receipt", result["protocol_receipt"])
    lambdas = np.asarray(protocol["ridge_lambdas"], dtype=np.float64)
    audit.check("protocol.fixed_lambdas", np.array_equal(lambdas, np.asarray([1e-6, 1e-4, .01, 1, 10, 100, 1000, 10000, 100000, 1000000], dtype=np.float64)))
    expected_counters = {"candidate_fits": 230, "channel_solves": 3910, "reused_full_train_candidates": 10,
                         "new_backbone_calls": 0, "new_test_predictions": 0, "gpu_jobs": 0}
    for key, value in expected_counters.items():
        audit.check(f"counters.{key}", result["counters"][key] == value)
    audit.check("wall_budget", 0 <= result["elapsed_seconds"] <= protocol["limits"]["wall_seconds"])
    input_path = check_receipt(audit, "shared_inputs", result["shared_inputs"])
    data = load_npz(input_path)
    provenance = result["provenance"]
    for key in ("parent_result", "legacy_predictions", "raw_predictions", "original_trainval_archive"):
        check_receipt(audit, f"provenance.{key}", provenance[key])
    for index, receipt in enumerate(provenance["runtime_source_files"]):
        check_receipt(audit, f"provenance.runtime_source{index}", receipt)
    contract_path = ROOT / "research/level_matched_learned_ed_v18_20261006/data_contract_robin.json"
    contract = read_json(contract_path)
    audit.check("data_contract.literal", provenance["data_contract"] == contract)
    original_path = resolve(contract["trainval"]["path"])
    audit.check("original_trainval.contract_sha", sha_file(original_path) == contract["trainval"]["sha256"])
    audit.check("original_trainval.provenance_path", original_path.resolve() == resolve(provenance["original_trainval_archive"]["path"]).resolve())
    original = load_npz(original_path)
    audit.check("original_trainval.shape", original["x"].shape == (3312, 17))
    for key, receipt in contract["trainval_array_receipts"].items():
        audit.check(f"original_trainval.{key}.array_sha", sha_array(original[key]) == receipt["sha256"])
        audit.check(f"original_trainval.{key}.shape", list(original[key].shape) == receipt["shape"])
        audit.check(f"original_trainval.{key}.dtype", str(original[key].dtype) == receipt["dtype"])
    legacy = load_npz(resolve(provenance["legacy_predictions"]["path"]))
    raw = load_npz(resolve(provenance["raw_predictions"]["path"]))
    parent = read_json(resolve(provenance["parent_result"]["path"]))
    parent_legacy = parent["datasets"]["robin"]["decoder_contracts"]["robin_RAW_C_D8"]["contracts"]["FORECAST_RIDGE_D8"]
    expected_train_origins = np.sort(np.random.default_rng(92603).choice(original["train_origins"], size=512, replace=False)).astype(np.int64)
    audit.equal_array("TRAIN512.literal_sampling_seed92603", data["train_origins"], expected_train_origins)
    audit.equal_array("TRAIN512.legacy_origins", data["train_origins"], legacy["train_origins"])
    audit.equal_array("VAL30.original_origins", data["val_origins"], original["val_origins"])
    audit.equal_array("VAL30.legacy_origins", data["val_origins"], legacy["origins"])
    audit.equal_array("TRAIN.raw_forecast_parent", data["train_forecast"], legacy["train_latent_forecast"])
    audit.equal_array("VAL.raw_forecast_parent", data["val_forecast"], legacy["latent_forecast"])
    audit.equal_array("VAL.raw_forecast_F0", data["val_forecast"], raw["prediction_tsfm"])
    audit.equal_array("channel_order", data["channel_names"], original["columns"])
    for split in ("train", "val"):
        origins = data[f"{split}_origins"]
        targets = np.stack([original["x"][int(origin):int(origin) + 48] for origin in origins]).astype(np.float32)
        masks = np.stack([original["finite"][int(origin):int(origin) + 48] for origin in origins]).astype(bool)
        history = np.stack([original["x"][int(origin) - 8:int(origin)] for origin in origins]).astype(np.float32)
        audit.equal_array(f"{split}.target_original", data[f"{split}_target"], targets)
        audit.equal_array(f"{split}.mask_original", data[f"{split}_mask"], masks)
        audit.equal_array(f"{split}.history_original", data[f"{split}_history"], history)
        bounds = np.asarray(contract["bounds"][split], dtype=np.int64)
        audit.equal_array(f"{split}.bounds_original", data[f"{split}_bounds"], bounds)
        audit.check(f"{split}.target_in_split", bool(np.all((origins >= bounds[0]) & (origins + 48 <= bounds[1]))))
        audit.check(f"{split}.input_context_available", bool(np.all(origins - 512 >= 0)))
        audit.check(f"{split}.forecast_fp32", data[f"{split}_forecast"].dtype == np.float32)
        check_horizon_metric(audit, f"RAW_F.{split}", result["arms"]["RAW_F"][split],
                             horizon_metrics(data[f"{split}_forecast"], targets, masks))
    raw_result = result["arms"]["RAW_F"]
    audit.check("RAW_F.parameters0", raw_result["capacity"]["trainable_parameters"] == 0)
    check_horizon_metric(audit, "RAW_F.parent_metrics", result["arms"]["RAW_F"]["val"], parent["datasets"]["robin"]["raw_f0"]["future_tsfm"])
    cuts = protocol["cv"]["timestamp_cuts"]
    expected_folds = []
    train_origins = data["train_origins"]
    for number in range(1, 4):
        fit = np.flatnonzero((train_origins >= cuts[0]) & (train_origins + 48 <= cuts[number])).astype(np.int64)
        evaluate = np.flatnonzero((train_origins >= cuts[number]) & (train_origins + 48 <= cuts[number + 1])).astype(np.int64)
        expected_folds.append((fit, evaluate))
        fold = result["folds"][number - 1]
        target_gap = int(train_origins[evaluate].min() - train_origins[fit].max() - 48)
        expected_fold = {"fold": number, "fit_target_bounds": [int(train_origins[fit].min()), int(train_origins[fit].max() + 48)],
                         "evaluation_target_bounds": [int(train_origins[evaluate].min()), int(train_origins[evaluate].max() + 48)],
                         "cut": cuts[number], "evaluation_end": cuts[number + 1], "fit_origins": len(fit),
                         "evaluation_origins": len(evaluate), "target_gap": target_gap}
        audit.check(f"CV.fold{number}.literal_contract", fold == expected_fold)
        audit.check(f"CV.fold{number}.purged_targets", target_gap >= 0 and not np.intersect1d(fit, evaluate).size)
        audit.check(f"CV.fold{number}.counts", (len(fit), len(evaluate)) == ((114, 119), (243, 125), (380, 123))[number - 1])
    expected_arms = {arm["id"]: arm for arm in protocol["arms"]}
    audit.check("arms.literal_ids", set(result["arms"]) == set(expected_arms))
    checked_candidates = 0
    for arm_id, arm in expected_arms.items():
        if arm_id == "RAW_F":
            continue
        recorded = result["arms"][arm_id]
        expected_spec = {"lag": arm["lag"], "residual": arm["prior"] == "identity",
                         "diagonal": arm["support"] == "diagonal", "dc_difference": arm["support"] == "lag_minus_current"}
        audit.check(f"{arm_id}.spec", recorded["spec"] == expected_spec)
        audit.check(f"{arm_id}.capacity", recorded["capacity"]["trainable_parameters"] == arm["parameters"])
        audit.check(f"{arm_id}.identity_skip", recorded["capacity"]["identity_skip_fixed_parameters"] == (17 if arm["prior"] == "identity" else 0))
        audit.check(f"{arm_id}.reuse_flag", recorded["full_train_coefficients_reused"] == (arm_id == "ZERO_FULL_D8"))
        family = load_npz(check_receipt(audit, f"{arm_id}.full_archive", recorded["full_train_archive"]))
        audit.equal_array(f"{arm_id}.lambdas", family["lambdas"], lambdas)
        audit.equal_array(f"{arm_id}.train_origins", family["train_origins"], data["train_origins"])
        audit.equal_array(f"{arm_id}.val_origins", family["val_origins"], data["val_origins"])
        train_feature = features(data["train_history"], data["train_forecast"], arm)
        val_feature = features(data["val_history"], data["val_forecast"], arm)
        check_feature_causality(audit, arm, data["train_history"], data["train_forecast"])
        expected_support = np.zeros((train_feature.shape[-1], 17), dtype=bool)
        for channel in range(17):
            expected_support[support(arm, train_feature.shape[-1], channel, 17), channel] = True
        audit.equal_array(f"{arm_id}.support", family["support"], expected_support)
        check_fit_details(audit, f"{arm_id}.full_details", recorded["full_train_fit_details"], train_feature, data["train_mask"], arm)
        check_normal_equations(audit, f"{arm_id}.full", arm, train_feature, data["train_target"], data["train_mask"],
                               data["train_forecast"], family["coefficients"], family["feature_scale"], lambdas)
        checked_candidates += len(lambdas)
        recomputed_val_scores = []
        audit.check(f"{arm_id}.choices_count", len(recorded["choices"]) == 10)
        for index, coefficient in enumerate(family["coefficients"]):
            choice = recorded["choices"][index]
            audit.check(f"{arm_id}.choice{index}.grid_position", choice["index"] == index and choice["lambda"] == float(lambdas[index]))
            for split, feature in (("train", train_feature), ("val", val_feature)):
                prediction = prediction_from_coeff(feature, coefficient, data[f"{split}_forecast"], arm)
                audit.equal_array(f"{arm_id}.choice{index}.{split}.prediction", family[f"predictions_{split}"][index], prediction)
                scores = horizon_metrics(prediction, data[f"{split}_target"], data[f"{split}_mask"])
                check_horizon_metric(audit, f"{arm_id}.choice{index}.{split}", choice[split], scores)
                if split == "val":
                    recomputed_val_scores.append(scores["all"]["mse"])
            if arm["support"] == "lag_minus_current":
                audit.close(f"{arm_id}.choice{index}.dc_identity", effective_taps(coefficient, arm, 17).sum(axis=0), np.eye(17), atol=2e-11, rtol=0)
        if arm_id == "ZERO_FULL_D8":
            audit.equal_array("legacy_grid.coefficients_literal", family["coefficients"], legacy["grid_coefficients"])
            audit.equal_array("legacy_grid.scale_literal", family["feature_scale"], legacy["feature_scale"])
            old_index = int(parent_legacy["selected_index"])
            audit.equal_array("legacy_grid.selected_prediction_literal", family["predictions_val"][old_index], legacy["prediction_tsfm"])
            audit.check("legacy_grid.selected_index_parity", int(np.argmin(recomputed_val_scores)) == old_index)
            for index, choice in enumerate(recorded["choices"]):
                check_horizon_metric(audit, f"legacy_grid.choice{index}.parent_metrics", choice["val"], parent_legacy["choices"][index]["metrics"])
        cv_grid_scores = []
        audit.check(f"{arm_id}.cv_folds3", len(recorded["cv"]["folds"]) == 3)
        for number, (fit, evaluate) in enumerate(expected_folds, start=1):
            fold_recorded = recorded["cv"]["folds"][number - 1]
            label = f"{arm_id}.fold{number}"
            audit.check(f"{label}.number", fold_recorded["fold"] == number)
            saved = load_npz(check_receipt(audit, f"{label}.archive", fold_recorded["archive"]))
            audit.equal_array(f"{label}.fit_indices", saved["fit_indices"], fit)
            audit.equal_array(f"{label}.evaluation_indices", saved["evaluation_indices"], evaluate)
            audit.equal_array(f"{label}.fit_origins", saved["fit_origins"], train_origins[fit])
            audit.equal_array(f"{label}.evaluation_origins", saved["evaluation_origins"], train_origins[evaluate])
            audit.equal_array(f"{label}.lambdas", saved["lambdas"], lambdas)
            check_fit_details(audit, f"{label}.fit_details", fold_recorded["fit_details"], train_feature[fit], data["train_mask"][fit], arm)
            check_normal_equations(audit, label, arm, train_feature[fit], data["train_target"][fit], data["train_mask"][fit],
                                   data["train_forecast"][fit], saved["coefficients"], saved["feature_scale"], lambdas)
            checked_candidates += len(lambdas)
            fold_scores = []
            for index, coefficient in enumerate(saved["coefficients"]):
                choice = fold_recorded["choices"][index]
                audit.check(f"{label}.choice{index}.grid_position", choice["index"] == index and choice["lambda"] == float(lambdas[index]))
                prediction = prediction_from_coeff(train_feature[evaluate], coefficient, data["train_forecast"][evaluate], arm)
                audit.equal_array(f"{label}.choice{index}.prediction", saved["predictions_evaluation"][index], prediction)
                scores = horizon_metrics(prediction, data["train_target"][evaluate], data["train_mask"][evaluate])
                check_horizon_metric(audit, f"{label}.choice{index}.metrics", choice["metrics"], scores)
                fold_scores.append(scores["all"]["mse"])
            cv_grid_scores.append(fold_scores)
        mean_cv = np.mean(cv_grid_scores, axis=0)
        audit.close(f"{arm_id}.cv_equal_fold_mean", recorded["cv"]["equal_fold_mean_mse"], mean_cv)
        cv_index = int(np.argmin(mean_cv))
        audit.check(f"{arm_id}.cv_index", recorded["cv"]["selected_index"] == cv_index)
        audit.check(f"{arm_id}.cv_lambda", recorded["cv"]["selected_lambda"] == float(lambdas[cv_index]))
        for selection, index in (("VAL_EXPLORATORY", int(np.argmin(recomputed_val_scores))), ("TRAIN_ROLLING_CV", cv_index)):
            selected = recorded["selections"][selection]
            audit.check(f"{arm_id}.{selection}.earliest_argmin", selected["index"] == index)
            audit.check(f"{arm_id}.{selection}.lambda", selected["lambda"] == float(lambdas[index]))
            audit.check(f"{arm_id}.{selection}.train_scores_literal", selected["train"] == recorded["choices"][index]["train"])
            audit.check(f"{arm_id}.{selection}.val_scores_literal", selected["val"] == recorded["choices"][index]["val"])
            if index in (0, len(lambdas) - 1):
                audit.warnings.append(f"{arm_id}/{selection}: selected lambda {lambdas[index]:g} is a grid boundary; finite-grid optimum only")
    audit.check("arrays.candidates240", checked_candidates == 240, actual=checked_candidates)
    audit.check("arrays.fresh230_reused10", checked_candidates - 10 == result["counters"]["candidate_fits"])
    audit.check("arrays.channel_solves3910", (checked_candidates - 10) * 17 == result["counters"]["channel_solves"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=HERE / "result.json")
    parser.add_argument("--output", type=Path, default=HERE / "verification.json")
    parser.add_argument("--full-preservation", action="store_true")
    args = parser.parse_args()
    if not args.result.is_file():
        raise FileNotFoundError("Run the scientific study before this saved-artifact verifier")
    audit = Audit()
    verify_study(audit, read_json(args.result))
    verify_preservation(audit, args.full_preservation)
    output = args.output.resolve()
    if not (output.is_relative_to(HERE) or output.is_relative_to(CACHE)):
        raise ValueError("Verifier output must remain inside this study or its cache")
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = audit.summary()
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("status", "checks_count", "failures_count", "failures", "warnings")}, ensure_ascii=False))
    raise SystemExit(0 if summary["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
