"""CPU-only readout comparison using sealed Robin TRAIN/VAL forecast traces."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / ".cache" / HERE.name
PARENT = ROOT / "research/level_k_tradeoff_20261011"
LAMBDAS = [1e-6, 1e-4, .01, 1., 10., 100., 1000., 10000., 100000., 1000000.]
CV_CUTS = [528, 1026, 1524, 2022, 2520]
SPECS = {
    "ZERO_FULL_D8": {"lag": 8, "residual": False, "diagonal": False, "dc_difference": False},
    "RES_FULL_D8": {"lag": 8, "residual": True, "diagonal": False, "dc_difference": False},
    "ZERO_DIAG_D8": {"lag": 8, "residual": False, "diagonal": True, "dc_difference": False},
    "RES_DIAG_D8": {"lag": 8, "residual": True, "diagonal": True, "dc_difference": False},
    "RES_FULL_D0": {"lag": 0, "residual": True, "diagonal": False, "dc_difference": False},
    "RES_DC_DIFF_D8": {"lag": 8, "residual": True, "diagonal": False, "dc_difference": True},
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def relpath(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def jsonable(value):
    if isinstance(value, Path):
        return relpath(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, value) -> None:
    path = path.resolve()
    if not (path.is_relative_to(HERE) or path.is_relative_to(CACHE)):
        raise PermissionError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def receipt(path: Path) -> dict:
    return {"path": relpath(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def save_archive(path: Path, **arrays) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite experiment archive: {path}")
    np.savez_compressed(path, **arrays)
    return {**receipt(path), "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": array_sha256(value)} for name, value in arrays.items()}}


def verify_file_map(path: Path) -> dict:
    manifest = read_json(path)
    files = manifest["files"]
    for name, expected in files.items():
        expected_hash = expected if isinstance(expected, str) else expected["sha256"]
        if sha256(resolve(name)) != expected_hash:
            raise RuntimeError(f"sealed/preserved file changed: {name}")
    return {"manifest": receipt(path), "files_verified": len(files)}


def import_parent():
    module_name = "_robin_readout_parent_sweep"
    if module_name not in sys.modules:
        path = PARENT / "sweep.py"
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name]


def future_features(history: np.ndarray, forecast: np.ndarray, lag: int) -> np.ndarray:
    """Use history up to origin-1 and already saved predictions up to horizon h."""
    if history.shape[0] != forecast.shape[0] or history.shape[2] != forecast.shape[2]:
        raise ValueError("history and forecast batch/channel mismatch")
    if history.shape[1] < lag:
        raise ValueError("insufficient historical tail")
    full = np.concatenate([history[:, -lag:] if lag else history[:, :0], forecast], axis=1)
    return np.concatenate([full[:, lag - ell:lag - ell + forecast.shape[1]] for ell in range(lag + 1)], axis=2).astype(np.float64)


def features_for(history: np.ndarray, forecast: np.ndarray, spec: dict) -> np.ndarray:
    features = future_features(history, forecast, int(spec["lag"]))
    if spec["dc_difference"]:
        channels = forecast.shape[2]
        current = features[:, :, :channels]
        features = np.concatenate([features[:, :, ell * channels:(ell + 1) * channels] - current for ell in range(1, spec["lag"] + 1)], axis=2)
    return features


def support_matrix(dim: int, channels: int, spec: dict) -> np.ndarray:
    support = np.ones((dim, channels), dtype=bool)
    if spec["diagonal"]:
        support[:] = False
        for channel in range(channels):
            support[channel::channels, channel] = True
    return support


def predict(features: np.ndarray, coeff: np.ndarray, raw_forecast: np.ndarray, residual: bool) -> np.ndarray:
    branch = np.einsum("btd,dc->btc", features.astype(np.float64), coeff.astype(np.float64)).astype(np.float32)
    return (branch + raw_forecast.astype(np.float32)).astype(np.float32) if residual else branch


def metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict:
    if pred.shape != target.shape or pred.shape != mask.shape or pred.ndim != 3:
        raise ValueError("prediction, target, mask must share [origins,horizon,channels]")
    observed = mask.astype(bool)
    if not np.isfinite(pred).all() or not np.isfinite(target[observed]).all():
        raise ValueError("nonfinite predictions or observed targets")
    error = np.where(observed, pred.astype(np.float64) - target.astype(np.float64), 0.)
    counts = observed.sum(axis=(0, 1))
    valid = counts > 0
    if not valid.any():
        raise ValueError("no observed targets")
    channel_mse = np.square(error).sum(axis=(0, 1)) / np.maximum(counts, 1)
    channel_mae = np.abs(error).sum(axis=(0, 1)) / np.maximum(counts, 1)
    return {"mse": float(channel_mse[valid].mean()), "mae": float(channel_mae[valid].mean()),
            "signed_mean_error": float((error.sum(axis=(0, 1)) / np.maximum(counts, 1))[valid].mean()),
            "channel_mse": channel_mse.tolist(), "channel_mae": channel_mae.tolist(),
            "target_counts": counts.tolist(), "target_count": int(counts.sum()), "origins": len(pred)}


def horizon_metrics(pred, target, mask, groups) -> dict:
    output = {"all": metrics(pred, target, mask)}
    for start, end in groups:
        output[f"h{start}_{end}"] = metrics(pred[:, start - 1:end], target[:, start - 1:end], mask[:, start - 1:end])
    return output


def fit_grid(features: np.ndarray, target: np.ndarray, mask: np.ndarray, raw_forecast: np.ndarray,
             spec: dict, lambdas: list[float], counters: dict | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit squared-error ridge with TRAIN-only RMS and channel-observation masks."""
    row_mask = mask.any(axis=2)
    flat = features.reshape(-1, features.shape[-1]).astype(np.float64)
    if not row_mask.any():
        raise ValueError("no feature scaling rows")
    rms = np.sqrt(np.mean(np.square(flat[row_mask.reshape(-1)]), axis=0))
    scale = np.maximum(rms, 1e-12)
    scaled = features.astype(np.float64) / scale
    rhs_target = target.astype(np.float64) - raw_forecast.astype(np.float64) if spec["residual"] else target.astype(np.float64)
    channels, dim = target.shape[2], features.shape[2]
    support = support_matrix(dim, channels, spec)
    statistics = []
    for channel in range(channels):
        active = np.flatnonzero(support[:, channel])
        observed = mask[:, :, channel].astype(bool)
        if not observed.any():
            raise ValueError(f"no observations for channel {channel}")
        x = scaled[observed][:, active]
        y = rhs_target[:, :, channel][observed]
        statistics.append((active, x.T @ x, x.T @ y))
    coefficients = []
    for lam in lambdas:
        coeff = np.zeros((dim, channels), dtype=np.float64)
        for channel, (active, xtx, xty) in enumerate(statistics):
            coeff[active, channel] = np.linalg.solve(xtx + float(lam) * np.eye(len(active)), xty) / scale[active]
            if counters is not None:
                counters["channel_solves"] += 1
        coefficients.append(coeff)
        if counters is not None:
            counters["candidate_fits"] += 1
    details = {"scale_rows": int(row_mask.sum()), "scale_row_mask_sha256": array_sha256(row_mask.astype(np.uint8)),
               "near_zero_features": int((rms <= 1e-12).sum()), "support_parameters": int(support.sum()),
               "feature_dimension": dim, "target_observations": mask.sum(axis=(0, 1)).tolist()}
    return np.stack(coefficients), scale, details


def chronological_folds(origins: np.ndarray, cuts: list[int], horizon: int = 48) -> list[dict]:
    folds = []
    for index in range(1, len(cuts) - 1):
        fit = np.flatnonzero((origins >= cuts[0]) & (origins + horizon <= cuts[index]))
        evaluate = np.flatnonzero((origins >= cuts[index]) & (origins + horizon <= cuts[index + 1]))
        if not len(fit) or not len(evaluate):
            raise ValueError("empty chronological fold")
        if origins[fit].max() + horizon > origins[evaluate].min():
            raise RuntimeError("fit/evaluation targets overlap")
        folds.append({"fold": index, "fit_indices": fit, "evaluation_indices": evaluate,
                      "fit_target_bounds": [int(origins[fit].min()), int(origins[fit].max() + horizon)],
                      "evaluation_target_bounds": [int(origins[evaluate].min()), int(origins[evaluate].max() + horizon)],
                      "cut": int(cuts[index]), "evaluation_end": int(cuts[index + 1]),
                      "fit_origins": len(fit), "evaluation_origins": len(evaluate),
                      "target_gap": int(origins[evaluate].min() - origins[fit].max() - horizon)})
    return folds


def validate_protocol(protocol: dict) -> tuple[list[float], list[int], list[tuple[int, int]]]:
    lambdas = [float(value) for value in protocol["ridge_lambdas"]]
    cuts = [int(value) for value in protocol["cv"]["timestamp_cuts"]]
    groups = [tuple(value) for value in protocol["horizon_groups"]]
    if lambdas != LAMBDAS or cuts != CV_CUTS or groups != [(1, 8), (9, 48)]:
        raise RuntimeError("fixed lambda grid, chronological cuts or horizon groups changed")
    expected_arms = [{"id": "RAW_F", "prior": "identity", "support": "identity", "lag": 0, "parameters": 0}]
    for name, spec in SPECS.items():
        dimension = 17 * (spec["lag"] if spec["dc_difference"] else spec["lag"] + 1)
        expected_arms.append({"id": name, "prior": "identity" if spec["residual"] else "zero",
                              "support": "lag_minus_current" if spec["dc_difference"] else ("diagonal" if spec["diagonal"] else "full"),
                              "lag": spec["lag"], "parameters": dimension if spec["diagonal"] else dimension * 17})
    actual_arms = [{key: arm[key] for key in expected_arms[0]} for arm in protocol["arms"]]
    if actual_arms != expected_arms:
        raise RuntimeError("protocol arm order, prior, support, lag or parameter budget changed")
    if [arm["id"] for arm in protocol["arms"] if arm.get("reuse_full_train_grid", False)] != ["ZERO_FULL_D8"]:
        raise RuntimeError("full TRAIN grid reuse differs from implemented plan")
    expected_limits = {"fresh_full_train_grid_candidates": 50, "fresh_cv_grid_candidates": 180,
                       "fresh_ridge_grid_candidates": 230, "fresh_channel_solves": 3910,
                       "reused_grid_candidates": 10, "new_f_calls": 0, "new_test_calls": 0}
    if any(protocol["limits"][key] != value for key, value in expected_limits.items()):
        raise RuntimeError("protocol fit/call budget differs from implemented plan")
    expected_shape = {"dataset": "robin", "context": 512, "horizon": 48, "channels": 17,
                      "train_origins": 512, "val_origins": 30, "decoder_sampling_seed": 92603}
    if any(protocol[key] != value for key, value in expected_shape.items()):
        raise RuntimeError("fixed data/forecast contract changed")
    if protocol["cv"]["folds"] != 3 or protocol["cv"]["train_bounds"] != [528, 2520]:
        raise RuntimeError("CV contract changed")
    if protocol["cv"]["expected_fit_counts"] != [114, 243, 380] or protocol["cv"]["expected_heldout_counts"] != [119, 125, 123]:
        raise RuntimeError("CV count contract changed")
    if any(protocol["fit"][key] for key in ("intercept", "centering", "identity_fallback_candidate")):
        raise RuntimeError("intercept, centering or identity fallback is outside this plan")
    return lambdas, cuts, groups


def load_inputs() -> tuple[dict, dict, dict]:
    parent_result_path = PARENT / "result.json"
    parent_result = read_json(parent_result_path)
    if parent_result["status"] != "complete":
        raise RuntimeError("parent is incomplete")
    dataset = parent_result["datasets"]["robin"]
    legacy = dataset["decoder_contracts"]["robin_RAW_C_D8"]["contracts"]["FORECAST_RIDGE_D8"]
    legacy_path = resolve(legacy["prediction_npz"])
    raw_path = resolve(dataset["raw_f0"]["prediction_npz"])
    if sha256(legacy_path) != legacy["prediction_sha256"] or sha256(raw_path) != dataset["raw_f0"]["prediction_sha256"]:
        raise RuntimeError("parent prediction bytes changed")
    sweep = import_parent()
    import torch
    torch.set_num_threads(1)
    bundle = sweep.load_original_bundle("robin")
    with np.load(legacy_path, allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    with np.load(raw_path, allow_pickle=False) as saved:
        raw_val = saved["prediction_tsfm"].astype(np.float32)
    if not np.array_equal(raw_val, arrays["latent_forecast"]):
        raise RuntimeError("identity forecast differs from saved RAW_F0")
    origins = arrays["train_origins"].astype(np.int64)
    if len(origins) != 512 or len(np.unique(origins)) != 512 or not np.array_equal(np.sort(origins), origins):
        raise RuntimeError("decoder TRAIN origins changed")
    if not np.array_equal(arrays["origins"], bundle.val_origins) or len(bundle.val_origins) != 30:
        raise RuntimeError("VAL origins changed")
    if not np.isin(origins, bundle.train_origins).all():
        raise RuntimeError("saved TRAIN origins outside original sample population")
    train_y, train_mask = sweep.parent_runner().old_pilot().future_targets(bundle, origins)
    val_y, val_mask = sweep.parent_runner().old_pilot().future_targets(bundle, bundle.val_origins)
    data = {"train_origins": origins, "val_origins": bundle.val_origins.astype(np.int64),
            "train_history": arrays["train_latent_history"].astype(np.float32),
            "val_history": arrays["val_latent_history_tail"].astype(np.float32),
            "train_forecast": arrays["train_latent_forecast"].astype(np.float32), "val_forecast": raw_val,
            "train_target": train_y, "val_target": val_y, "train_mask": train_mask, "val_mask": val_mask,
            "channel_names": np.asarray(bundle.columns), "train_bounds": np.asarray(bundle.train_bounds, dtype=np.int64),
            "val_bounds": np.asarray(bundle.val_bounds, dtype=np.int64)}
    provenance = {"parent_result": receipt(parent_result_path), "legacy_predictions": receipt(legacy_path),
                  "raw_predictions": receipt(raw_path), "original_trainval_archive": receipt(resolve(bundle.source_path)),
                  "identity_val_bitwise_matches_parent": True,
                  "data_contract": bundle.contract, "loaded_trainval_shape": list(bundle.x.shape),
                  "runtime_source_files": [receipt(ROOT / path) for path in (
                      "research/level_k_tradeoff_20261011/sweep.py", "research/level_encoder_contract_20261010/runner.py",
                      "research/level_encoder_contract_20261010/encoders.py", "research/level_lag_compressor_pilot_20261010/pilot.py",
                      "research/level_matched_learned_ed_v18_20261006/runtime_v18.py")]}
    return data, arrays, {"legacy": legacy, "provenance": provenance}


def score_grid(coefficients, train_features, val_features, data, spec, groups):
    predictions_train, predictions_val, choices = [], [], []
    for index, coeff in enumerate(coefficients):
        train_pred = predict(train_features, coeff, data["train_forecast"], spec["residual"])
        val_pred = predict(val_features, coeff, data["val_forecast"], spec["residual"])
        predictions_train.append(train_pred)
        predictions_val.append(val_pred)
        choices.append({"index": index, "lambda": LAMBDAS[index],
                        "train": horizon_metrics(train_pred, data["train_target"], data["train_mask"], groups),
                        "val": horizon_metrics(val_pred, data["val_target"], data["val_mask"], groups)})
    return np.stack(predictions_train), np.stack(predictions_val), choices


def run() -> dict:
    started = time.perf_counter()
    cpu_started = time.process_time()
    result_path = HERE / "result.json"
    if result_path.exists() or (HERE / "execution.started.json").exists():
        raise FileExistsError("study already started; scientific outputs will not be overwritten")
    protocol = read_json(HERE / "protocol.json")
    lambdas, cuts, groups = validate_protocol(protocol)
    seal_before = verify_file_map(HERE / "source_seal.json")
    preservation_before = verify_file_map(HERE / "preservation.before.json")
    write_json(HERE / "execution.started.json", {"pid": os.getpid(), "started_unix": time.time(),
               "source_seal": seal_before, "preservation": preservation_before})
    counters = {"candidate_fits": 0, "channel_solves": 0, "reused_full_train_candidates": 10,
                "new_backbone_calls": 0, "new_test_predictions": 0, "gpu_jobs": 0}
    with threadpool_limits(limits=1):
        data, legacy_arrays, parent = load_inputs()
        shared = save_archive(CACHE / "inputs.npz", **data)
        folds = chronological_folds(data["train_origins"], cuts)
        expected_counts = [[114, 119], [243, 125], [380, 123]]
        if [[fold["fit_origins"], fold["evaluation_origins"]] for fold in folds] != expected_counts:
            raise RuntimeError("chronological fold counts changed")
        result = {"schema_version": 1, "status": "running", "study": HERE.name, "dataset": "robin",
                  "protocol": protocol, "protocol_receipt": receipt(HERE / "protocol.json"),
                  "provenance": parent["provenance"], "shared_inputs": shared,
                  "precision": "features/statistics/coefficients FP64; readout branch FP32; residual branch added to saved FP32 RAW_F",
                  "cv_scope": "conditional decoder CV; source normalization and cached backbone inputs use original full TRAIN statistics",
                  "selection_scope": "VAL_EXPLORATORY reuses previously inspected VAL; TRAIN_ROLLING_CV selects lambda without VAL targets; no identity fallback",
                  "arms": {"RAW_F": {"spec": {"identity": True}, "capacity": {"trainable_parameters": 0},
                           "train": horizon_metrics(data["train_forecast"], data["train_target"], data["train_mask"], groups),
                           "val": horizon_metrics(data["val_forecast"], data["val_target"], data["val_mask"], groups)}},
                  "folds": [{key: value for key, value in fold.items() if key not in ("fit_indices", "evaluation_indices")} for fold in folds]}
        for arm, spec in SPECS.items():
            arm_started = time.perf_counter()
            train_features = features_for(data["train_history"], data["train_forecast"], spec)
            val_features = features_for(data["val_history"], data["val_forecast"], spec)
            if arm == "ZERO_FULL_D8":
                if not np.array_equal(legacy_arrays["grid_lambdas"], np.asarray(lambdas, dtype=np.float64)):
                    raise RuntimeError("legacy lambda grid changed")
                coeffs = legacy_arrays["grid_coefficients"].astype(np.float64)
                scale = legacy_arrays["feature_scale"].astype(np.float64)
                details = {"support_parameters": int(np.prod(coeffs.shape[1:])), "feature_dimension": len(scale),
                           "scale_rows": int(data["train_mask"].any(axis=2).sum()),
                           "scale_row_mask_sha256": array_sha256(data["train_mask"].any(axis=2).astype(np.uint8)),
                           "near_zero_features": int((scale <= 1e-12).sum()),
                           "target_observations": data["train_mask"].sum(axis=(0, 1)).tolist()}
                full_fit_seconds = 0.
            else:
                fit_started = time.perf_counter()
                coeffs, scale, details = fit_grid(train_features, data["train_target"], data["train_mask"], data["train_forecast"], spec, lambdas, counters)
                full_fit_seconds = time.perf_counter() - fit_started
            train_predictions, val_predictions, choices = score_grid(coeffs, train_features, val_features, data, spec, groups)
            val_index = int(np.argmin([choice["val"]["all"]["mse"] for choice in choices]))
            legacy_reproduction = None
            if arm == "ZERO_FULL_D8":
                old_index = int(parent["legacy"]["selected_index"])
                prediction_exact = np.array_equal(val_predictions[old_index], legacy_arrays["prediction_tsfm"])
                max_metric_difference = max(abs(choices[index]["val"]["all"][metric] - parent["legacy"]["choices"][index]["metrics"]["all"][metric])
                                            for index in range(len(lambdas)) for metric in ("mse", "mae"))
                if not prediction_exact or val_index != old_index or max_metric_difference > 1e-14:
                    raise RuntimeError("legacy decoder reproduction failed")
                legacy_reproduction = {"selected_prediction_bitwise_exact": prediction_exact, "selected_index_equal": True,
                                       "maximum_mse_mae_difference": max_metric_difference}
            full_archive = save_archive(CACHE / "families" / f"{arm}.npz", coefficients=coeffs, feature_scale=scale,
                           support=support_matrix(train_features.shape[2], data["train_forecast"].shape[2], spec),
                           lambdas=np.asarray(lambdas, dtype=np.float64), predictions_train=train_predictions,
                           predictions_val=val_predictions, train_origins=data["train_origins"], val_origins=data["val_origins"])
            cv_folds, cv_grid_scores = [], []
            for fold in folds:
                fit_indices, eval_indices = fold["fit_indices"], fold["evaluation_indices"]
                cv_started = time.perf_counter()
                cv_coeffs, cv_scale, cv_details = fit_grid(train_features[fit_indices], data["train_target"][fit_indices],
                           data["train_mask"][fit_indices], data["train_forecast"][fit_indices], spec, lambdas, counters)
                eval_predictions = np.stack([predict(train_features[eval_indices], coefficient, data["train_forecast"][eval_indices], spec["residual"]) for coefficient in cv_coeffs])
                cv_choices = [{"index": index, "lambda": lambdas[index], "metrics": horizon_metrics(prediction, data["train_target"][eval_indices], data["train_mask"][eval_indices], groups)} for index, prediction in enumerate(eval_predictions)]
                cv_grid_scores.append([choice["metrics"]["all"]["mse"] for choice in cv_choices])
                cv_archive = save_archive(CACHE / "cv" / f"{arm}_fold{fold['fold']}.npz", coefficients=cv_coeffs,
                             feature_scale=cv_scale, predictions_evaluation=eval_predictions,
                             fit_indices=fit_indices.astype(np.int64), evaluation_indices=eval_indices.astype(np.int64),
                             fit_origins=data["train_origins"][fit_indices], evaluation_origins=data["train_origins"][eval_indices],
                             lambdas=np.asarray(lambdas, dtype=np.float64))
                cv_folds.append({"fold": fold["fold"], "fit_details": cv_details, "choices": cv_choices,
                                 "archive": cv_archive, "fit_and_evaluation_seconds": time.perf_counter() - cv_started})
                write_json(HERE / "progress.json", {"arm": arm, "completed_cv_fold": fold["fold"], "counters": counters,
                           "elapsed_seconds": time.perf_counter() - started, "pid": os.getpid()})
            mean_cv = np.asarray(cv_grid_scores, dtype=np.float64).mean(axis=0)
            cv_index = int(np.argmin(mean_cv))
            capacity = {"trainable_parameters": details["support_parameters"], "feature_dimension": details["feature_dimension"],
                        "cross_channel": not spec["diagonal"], "lag": spec["lag"], "identity_skip_fixed_parameters": 17 if spec["residual"] else 0}
            selections = {}
            for selection, index in (("VAL_EXPLORATORY", val_index), ("TRAIN_ROLLING_CV", cv_index)):
                selections[selection] = {"index": index, "lambda": lambdas[index], "train": choices[index]["train"], "val": choices[index]["val"]}
            result["arms"][arm] = {"spec": spec, "capacity": capacity, "full_train_fit_details": details,
                                   "full_train_fit_seconds": full_fit_seconds, "full_train_archive": full_archive,
                                   "full_train_coefficients_reused": arm == "ZERO_FULL_D8", "legacy_reproduction": legacy_reproduction,
                                   "choices": choices, "selections": selections,
                                   "cv": {"folds": cv_folds, "equal_fold_mean_mse": mean_cv.tolist(), "selected_index": cv_index,
                                          "selected_lambda": lambdas[cv_index], "tie_rule": "first/lower grid index"},
                                   "arm_total_seconds": time.perf_counter() - arm_started}
            print(json.dumps({"arm": arm, "VAL_lambda": lambdas[val_index], "CV_lambda": lambdas[cv_index],
                              "VAL_mse": choices[val_index]["val"]["all"]["mse"], "CV_selected_VAL_mse": choices[cv_index]["val"]["all"]["mse"],
                              "candidate_fits": counters["candidate_fits"]}), flush=True)
        if counters["candidate_fits"] != protocol["limits"]["fresh_ridge_grid_candidates"] or counters["channel_solves"] != protocol["limits"]["fresh_channel_solves"]:
            raise RuntimeError(f"fit counts differ: {counters}")
        result["integrity"] = {"source_seal_before": seal_before, "source_seal_after": verify_file_map(HERE / "source_seal.json"),
                               "preservation_before": preservation_before, "preservation_after": verify_file_map(HERE / "preservation.before.json")}
        result.update(status="complete", counters=counters, elapsed_seconds=time.perf_counter() - started,
                      process_cpu_seconds=time.process_time() - cpu_started)
        write_json(result_path, result)
        write_json(HERE / "progress.json", {"status": "complete", "counters": counters, "elapsed_seconds": result["elapsed_seconds"], "result": receipt(result_path)})
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="run only after root seals protocol and preservation manifests")
    args = parser.parse_args()
    if not args.execute:
        parser.error("--execute is required; fixture tests do not run scientific fits")
    try:
        result = run()
        print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"], "counters": result["counters"]}), flush=True)
    except Exception as error:
        if (HERE / "execution.started.json").exists() and not (HERE / "result.json").exists():
            write_json(HERE / "failure.json", {"status": "failed", "type": type(error).__name__, "message": str(error), "time_unix": time.time()})
        raise


if __name__ == "__main__":
    main()
