"""Independent equations for the PCA/ForeCA fixed E/D follow-up.

Importing this module opens no dataset, model, checkpoint or producer result.
Actual file reads require a caller-supplied live authorization check. Neither
the projection producer nor native trainer is imported. Saved arithmetic and
provenance checks are not an independent holdout or a replay of Bolt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

np = sys.modules.get("numpy")
CHECKPOINTS = (0, 100, 200, 400, 600)
SEEDS = (92601, 92602)
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = HERE.parents[2]
DATASETS = {"robin": (17, 4, "val", 30, "first_robin_v1"),
            "beijing_aotizhongxin": (11, 2, "eval", 146, "first_beijing_v1")}
COST_ROUTES = ("RAW_BOLT", "PCA_BOLT", "FORECA_BOLT", "PCA_NATIVE", "FORECA_NATIVE")


class AuditFailure(ValueError):
    pass


def initialize_numpy():
    global np
    import numpy as loaded_numpy
    np = loaded_numpy


def require(condition, message):
    if not condition:
        raise AuditFailure(message)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def exact(actual, expected, message):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape and actual.dtype == expected.dtype
            and np.array_equal(actual, expected, equal_nan=True), "EXACT_MISMATCH:" + message)


def close(actual, expected, message, *, atol=1e-5, rtol=2e-6):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape and np.isfinite(actual).all()
            and np.isfinite(expected).all()
            and np.allclose(actual, expected, atol=atol, rtol=rtol), "NUMERIC_MISMATCH:" + message)
    return float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))) if actual.size else 0.0


def scores(prediction, target, observed, fit_std):
    """Loop over observed channels: macro averages differ from pooled errors."""
    prediction, target, observed = (np.asarray(value) for value in (prediction, target, observed))
    require(prediction.shape == target.shape and target.ndim == 3
            and observed.shape == target.shape and observed.dtype == np.bool_, "SCORE_SHAPE_MASK")
    channels = target.shape[-1]
    std = np.asarray(fit_std, np.float64)
    require(std.shape == (channels,) and np.isfinite(std).all() and np.all(std > 0), "SCORE_FIT_STD")
    per_mse, per_mae, counts = [], [], []
    for channel in range(channels):
        rows = observed[:, :, channel]
        predicted = prediction[:, :, channel][rows].astype(np.float64)
        actual = target[:, :, channel][rows].astype(np.float64)
        require(np.isfinite(predicted).all() and np.isfinite(actual).all(), "NONFINITE_OBSERVED_ERROR")
        count = int(rows.sum())
        counts.append(count)
        error = predicted - actual
        per_mse.append(float(np.dot(error, error) / count) if count else 0.0)
        per_mae.append(float(np.abs(error).sum() / count) if count else 0.0)
    active = np.array(counts) > 0
    require(active.any(), "NO_OBSERVED_CHANNEL")
    mse, mae = np.array(per_mse), np.array(per_mae)
    common = {"observed_counts": counts, "active_channels": int(active.sum())}
    output = {}
    for units, squared_scale, absolute_scale in (
            ("normalized", np.ones(channels), np.ones(channels)),
            ("physical_per_channel_diagnostic", std * std, std)):
        output[units] = {"mse": float((mse * squared_scale)[active].mean()),
                         "mae": float((mae * absolute_scale)[active].mean()),
                         "per_channel_mse": (mse * squared_scale).tolist(),
                         "per_channel_mae": (mae * absolute_scale).tolist(), **common}
    return output


def compare_scores(recorded, recalculated, label):
    checks = 0
    for units in ("normalized", "physical_per_channel_diagnostic"):
        for field in ("mse", "mae", "per_channel_mse", "per_channel_mae"):
            actual = recorded[units][field]
            if field.startswith("per_channel"):
                require(len(actual) == len(recalculated[units][field]), "PER_CHANNEL_SCORE_SHAPE:" + label)
                actual = [0.0 if value is None and not recalculated[units]["observed_counts"][index] else value
                          for index, value in enumerate(actual)]
            close(actual, recalculated[units][field], label + ":" + units + ":" + field,
                  atol=1e-10, rtol=1e-10)
            checks += 1
        for field in ("observed_counts", "active_channels"):
            require(recorded[units][field] == recalculated[units][field], "MASK_COUNT_MISMATCH:" + label)
            checks += 1
    return checks


def verify_origins(actual, locked):
    origins = np.asarray(actual)
    require(origins.dtype == np.int64 and origins.ndim == 1 and origins.size > 0
            and np.all(np.diff(origins) > 0), "ORDERED_INT64_ORIGINS_REQUIRED")
    if isinstance(locked, list):
        exact(origins, np.asarray(locked, np.int64), "locked_origins")
    else:
        require(isinstance(locked, dict) and set(locked) == {"count", "first", "last", "sha256"},
                "ORIGIN_LOCK_SCHEMA")
        require(len(origins) == locked["count"] and int(origins[0]) == locked["first"]
                and int(origins[-1]) == locked["last"]
                and hashlib.sha256(np.ascontiguousarray(origins).tobytes()).hexdigest() == locked["sha256"],
                "ORIGIN_LOCK_DIGEST")
    return len(origins)


def check_raw_split(values, *, channels, windows, locked_origins=None):
    history, target, observed = (np.asarray(values[key]) for key in ("history", "target", "observed"))
    require(history.dtype == np.float32 and history.shape == (windows, 512, channels)
            and np.isfinite(history).all(), "FINITE_FP32_FIXED_PAST_SPLIT")
    require(target.dtype == np.float32 and target.shape == (windows, 48, channels)
            and observed.dtype == np.bool_ and observed.shape == target.shape
            and np.isfinite(target[observed]).all(), "FP32_FUTURE_OBSERVED_MASK_SPLIT")
    require(values["history_observed"].dtype == np.bool_
            and values["history_observed"].shape == history.shape, "ORIGINAL_PAST_OBSERVABILITY_MASK")
    exact(values["raw_last"], history[:, -1], "past_current_C_anchor")
    require(values["origins"].dtype == np.int64 and values["origins"].shape == (windows,)
            and np.all(np.diff(values["origins"]) > 0), "FIXED_SPLIT_ORIGINS")
    if locked_origins is not None:
        verify_origins(values["origins"], locked_origins)
    for key in ("mean", "std"):
        require(values[key].dtype == np.float32 and values[key].shape == (channels,)
                and np.isfinite(values[key]).all(), "FIT_ONLY_RAW_SCALER:" + key)
    require(np.all(values["std"] > 0), "POSITIVE_FIT_ONLY_RAW_STD")
    return {"windows": windows, "channels": channels, "first_origin": int(values["origins"][0]),
            "last_origin": int(values["origins"][-1]), "observed_counts": observed.sum(axis=(0, 1)).tolist()}


def causal_physical_preprocess(physical, observed, mean, std):
    physical, observed, mean, std = (np.asarray(value) for value in (physical, observed, mean, std))
    require(physical.dtype == np.float32 and physical.ndim == 3 and physical.shape[1] == 512
            and observed.dtype == np.bool_ and physical.shape == observed.shape, "PHYSICAL_FP32_N512C_MASK")
    channels = physical.shape[-1]
    require(mean.dtype == std.dtype == np.float32 and mean.shape == std.shape == (channels,)
            and np.isfinite(mean).all() and np.isfinite(std).all() and np.all(std > 0)
            and np.isfinite(physical[observed]).all(), "PHYSICAL_FIT_ONLY_SCALER")
    filled = np.empty_like(physical)
    for window in range(len(physical)):
        last = mean.copy()
        for position in range(512):
            for channel in range(channels):
                if observed[window, position, channel]:
                    last[channel] = physical[window, position, channel]
            filled[window, position] = last
    return np.ascontiguousarray((filled - mean) / std)


def check_frozen_arrays(before, after, label="E_D"):
    require(set(before) == set(after) and bool(before), "FROZEN_STATE_KEYS:" + label)
    hashes = {}
    for key in before:
        exact(before[key], after[key], label + ":" + key)
        hashes[key] = array_sha(before[key])
    return hashes


def encode(history, mean, encoder):
    history, mean, encoder = (np.asarray(value) for value in (history, mean, encoder))
    require(history.ndim == 3 and history.shape[-1] == len(mean)
            and mean.ndim == 1 and encoder.ndim == 2 and encoder.shape[0] == len(mean)
            and encoder.shape[1] > 0 and np.isfinite(history).all()
            and np.isfinite(mean).all() and np.isfinite(encoder).all(), "ENCODER_SHAPE_FINITE")
    require(history.dtype == np.float32, "FP32_DEPLOYMENT_HISTORY_REQUIRED")
    return np.ascontiguousarray((np.ascontiguousarray(history) - mean.astype(np.float32))
                                @ encoder.astype(np.float32))


def anchor_prediction(history, raw_last, latent_forecast, mean, encoder, decoder):
    """The affine D intercept cancels; the same original current-C anchor remains."""
    history, raw_last, latent_forecast, decoder = (
        np.asarray(value) for value in (history, raw_last, latent_forecast, decoder))
    require(history.dtype == np.float32 and raw_last.dtype == np.float32
            and raw_last.shape == (len(history), history.shape[-1]), "FP32_CURRENT_C_ANCHOR")
    exact(raw_last, history[:, -1], "original_current_C_anchor")
    z_last = encode(history, mean, encoder)[:, -1:]
    require(latent_forecast.ndim == 3 and latent_forecast.shape[0] == len(history)
            and latent_forecast.shape[-1] == z_last.shape[-1]
            and decoder.shape == (z_last.shape[-1], history.shape[-1])
            and np.isfinite(latent_forecast).all() and np.isfinite(decoder).all(), "DECODER_FORECAST_SHAPE_FINITE")
    require(latent_forecast.dtype == np.float32, "FP32_LATENT_FORECAST_REQUIRED")
    innovation = (latent_forecast - z_last) @ decoder.astype(np.float32)
    return np.ascontiguousarray(raw_last[:, None] + innovation)


def native_latent_forecast(latent_history, weights, bias):
    latent, weights, bias = (np.asarray(value) for value in (latent_history, weights, bias))
    require(latent.ndim == 3 and weights.ndim == 3 and bias.shape == weights.shape[:2]
            and weights.shape[0] == latent.shape[-1] and weights.shape[2] == latent.shape[1]
            and all(value.dtype == np.float32 and np.isfinite(value).all()
                    for value in (latent, weights, bias)), "FINITE_FP32_NATIVE_KHL_STATE")
    last = latent[:, -1:]
    difference = latent - last
    future = np.empty((len(latent), weights.shape[1], latent.shape[-1]), np.float32)
    for component in range(latent.shape[-1]):
        innovation = difference[:, :, component] @ weights[component].T
        future[:, :, component] = last[:, :, component] + (innovation + bias[component])
    return future


def torch_state_sha(saved_arrays):
    """Recreate the producer's declared tensor signature without importing Torch."""
    digest = hashlib.sha256()
    for key, value in sorted(saved_arrays.items()):
        array = np.ascontiguousarray(value)
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def spectral_diagnostics(latent_history):
    """Independent declared Hann/one-sided/reflected-seven-bin entropy arithmetic."""
    values = np.asarray(latent_history, np.float64)
    require(values.ndim == 3 and values.shape[1] == 512 and np.isfinite(values).all(), "SPECTRAL_N512K_REQUIRED")
    centered = values - values.reshape(-1, values.shape[-1]).mean(axis=0)
    transform = np.fft.rfft(centered * np.hanning(512)[None, :, None], axis=1)
    power = np.mean(transform.real ** 2 + transform.imag ** 2, axis=0)
    power[1:-1] *= 2.0
    power /= 512 * np.square(np.hanning(512)).sum()
    reflected = np.pad(power, ((3, 3), (0, 0)), mode="reflect")
    smoothed = np.empty_like(power)
    for frequency in range(power.shape[0]):
        smoothed[frequency] = reflected[frequency:frequency + 7].mean(axis=0)
    total = smoothed.sum(axis=0)
    require(np.isfinite(total).all() and np.all(total > 0), "COLLAPSED_SPECTRAL_POWER")
    probabilities = smoothed / total
    entropy = np.zeros(values.shape[-1], np.float64)
    for component in range(values.shape[-1]):
        positive = probabilities[:, component] > 0
        entropy[component] = -np.dot(probabilities[positive, component],
                                    np.log(probabilities[positive, component]))
    return {"power": power, "smoothed_power": smoothed, "probabilities": probabilities,
            "entropy": entropy, "normalized_entropy": entropy / np.log(power.shape[0]),
            "forecastability": 1.0 - entropy / np.log(power.shape[0])}


def check_projection_state(method, fit_history, mean, encoder, decoder, *,
                           fit_observed=None, decoder_bias=None, rank_cutoff=1e-8):
    """FIT past only covariance, valid rank, E constraints and D normal equations."""
    history = np.asarray(fit_history)
    require(method in ("PCA", "FORECA") and history.ndim == 3 and history.dtype == np.float32
            and np.isfinite(history).all() and history.size > 0, "APPROVED_FINITE_FIT_PAST")
    raw = history.reshape(-1, history.shape[-1]).astype(np.float64)
    expected_mean = raw.mean(axis=0)
    close(mean, expected_mean, "FIT_past_mean", atol=1e-10, rtol=1e-10)
    centered = raw - expected_mean
    covariance = centered.T @ centered / len(raw)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    require(eigenvalues[0] > 0, "ZERO_FIT_COVARIANCE")
    effective_rank = int(np.count_nonzero(eigenvalues > rank_cutoff * eigenvalues[0]))
    encoder, decoder = np.asarray(encoder, np.float64), np.asarray(decoder, np.float64)
    components = encoder.shape[1] if encoder.ndim == 2 else 0
    require(encoder.shape == (history.shape[-1], components) and components > 0
            and decoder.shape == (components, history.shape[-1])
            and np.isfinite(encoder).all() and np.isfinite(decoder).all(), "FINITE_E_D_SHAPES")
    require(effective_rank >= components, "EFFECTIVE_RANK_BELOW_K")
    latent_covariance = encoder.T @ covariance @ encoder
    close(latent_covariance, np.eye(components), method + "_unit_variance_orthogonal_E", atol=1e-7, rtol=0)
    if method == "PCA":
        close(covariance @ encoder, encoder * eigenvalues[:components][None, :],
              "PCA_top_K_covariance", atol=1e-8, rtol=1e-8)
    require(np.linalg.eigvalsh(latent_covariance)[0] > 0, "COLLAPSED_LATENT_COMPONENT")
    latent = centered @ encoder
    require(fit_observed is not None and decoder_bias is not None, "MASKED_AFFINE_FIT_D_CONTRACT_REQUIRED")
    observed = np.asarray(fit_observed)
    bias = np.asarray(decoder_bias, np.float64)
    require(observed.dtype == np.bool_ and observed.shape == history.shape and bias.shape == mean.shape
            and np.isfinite(bias).all(), "MASKED_AFFINE_FIT_D_SHAPES")
    residual = latent @ decoder + bias - raw
    normal_maximum = 0.0
    flat_mask = observed.reshape(raw.shape)
    for channel in range(history.shape[-1]):
        rows = flat_mask[:, channel]
        require(rows.any(), "DECODER_CHANNEL_WITHOUT_OBSERVED_FIT_PAST")
        equations = np.r_[latent[rows].T @ residual[rows, channel] / rows.sum(),
                          residual[rows, channel].mean()]
        close(equations, np.zeros_like(equations), "FIT_past_D_normal_equations", atol=1e-8, rtol=0)
        normal_maximum = max(normal_maximum, float(np.abs(equations).max()))
    return {"effective_rank": effective_rank, "components": components,
            "FIT_rows_with_overlap": len(raw), "rank_cutoff": rank_cutoff,
            "latent_covariance": latent_covariance.tolist(),
            "decoder_normal_equation_max_abs": normal_maximum,
            "independent_samples_claimed": False}


def check_projection_metadata(metadata, state, fit_history, fit_observed):
    require(metadata["route"] in ("PCA", "FORECA") and metadata["channels"] == fit_history.shape[-1]
            and metadata["K"] == state["encoder"].shape[-1] and metadata["length"] == 512,
            "PROJECTION_METADATA_DIMENSIONS")
    require(metadata["fit_windows"] == len(fit_history)
            and metadata["fit_positions_with_overlap"] == len(fit_history) * 512,
            "PROJECTION_FIT_WINDOW_POSITION_COUNTS")
    require(metadata["history_sha256"] == array_sha(fit_history)
            and metadata["history_observed_sha256"] == array_sha(fit_observed), "FIT_PAST_ARRAY_SHA_BINDING")
    require(metadata["decoder_observed_counts"] == fit_observed.sum(axis=(0, 1)).tolist(),
            "DECODER_ORIGINAL_FIT_OBSERVED_COUNTS")
    require(metadata["actual_future_target_used_in_projection_or_decoder_fit"] is False
            and metadata["strict_K_only_compression"] is False
            and metadata["encoder_fit_dtype"] == metadata["serialized_state_dtype"] == "float64"
            and metadata["deployment_dtype"] == "float32", "PROJECTION_FIT_INPUT_PRECISION_SCOPE")
    require(metadata["eigenvalue_relative_cut"] == 1e-8, "PREDECLARED_RELATIVE_RANK_CUTOFF")
    if "state_array_sha256" in metadata:
        for key in ("center", "encoder", "decoder", "decoder_bias"):
            require(metadata["state_array_sha256"][key] == array_sha(state[key]), "PROJECTION_STATE_ARRAY_SHA:" + key)
    report = check_projection_state(metadata["route"], fit_history, state["center"], state["encoder"], state["decoder"],
                                    fit_observed=fit_observed, decoder_bias=state["decoder_bias"])
    require(metadata["whitening_rank"] == report["effective_rank"], "DECLARED_WHITENING_RANK")
    close(metadata["latent_fit_covariance"], report["latent_covariance"], "declared_FIT_latent_covariance",
          atol=1e-10, rtol=1e-10)
    # Spectral diagnostics use fitted FP64 coordinates, not FP32 deployment values.
    centered = fit_history.astype(np.float64) - state["center"]
    latent64 = centered @ state["encoder"]
    spectral = spectral_diagnostics(latent64)
    search = metadata["foreca_search"]
    if metadata["route"] == "PCA":
        require(search == [], "PCA_HAS_NO_FORECA_OPTIMIZER_START_SELECTION")
    else:
        require(len(search) == metadata["K"] and metadata["foreca_max_iterations_per_start"] == 200,
                "FORECA_K_TWO_START_PREDECLARED_SEARCH")
        for component, entry in enumerate(search):
            require(entry["component"] == component and len(entry["starts"]) == 2,
                    "FORECA_COMPONENT_TWO_START_SCHEMA")
            values = [float(start["final_entropy"]) for start in entry["starts"]]
            require(all(np.isfinite(value) and 0 <= value <= 1 for value in values), "FORECA_FINITE_ENTROPY")
            selected = 0 if values[0] <= values[1] else 1
            require(entry["selected_start"] == selected, "FORECA_FIT_ENTROPY_EARLIEST_MINIMUM")
            close(values[selected], spectral["normalized_entropy"][component], "FORECA_saved_component_entropy",
                  atol=1e-10, rtol=1e-10)
            for start in entry["starts"]:
                require(0 <= start["accepted_steps"] <= start["iterations"] <= 200
                        and float(start["final_entropy"]) <= float(start["initial_entropy"]) + 1e-12,
                        "FORECA_ITERATION_OR_DESCENT_CONTRACT")
    return {**report, "FIT_spectral_normalized_entropy": spectral["normalized_entropy"].tolist(),
            "FIT_spectral_forecastability": spectral["forecastability"].tolist(),
            "optimizer_trajectory_independently_replayed": False}


def select_checkpoint(checkpoints):
    require(tuple(row["update"] for row in checkpoints) == CHECKPOINTS, "PREDECLARED_CHECKPOINT_GRID_REQUIRED")
    best = 0
    for index, row in enumerate(checkpoints):
        value = float(row["tune_mse"])
        require(np.isfinite(value) and value >= 0, "FINITE_NONNEGATIVE_TUNE_MSE")
        if value < float(checkpoints[best]["tune_mse"]):
            best = index
    return {"selected_index": best, "selected_update": checkpoints[best]["update"],
            "selected_tune_mse": float(checkpoints[best]["tune_mse"])}


def oracle_mask(observed):
    observed = np.asarray(observed)
    require(observed.ndim == 3 and observed.dtype == np.bool_, "ORACLE_BOOL_NHC_MASK")
    return np.repeat(observed.all(axis=-1, keepdims=True), observed.shape[-1], axis=-1)


def check_oracle_contract(recorded_mask, observed, recorded_role):
    require(recorded_role == "diagnostic_only_not_deployment_or_selection", "ORACLE_DIAGNOSTIC_ROLE_REQUIRED")
    expected = oracle_mask(observed)
    exact(recorded_mask, expected, "oracle_same_complete_target_positions")
    require(expected.any(), "NO_COMPLETE_TARGET_POSITION_ORACLE_INFEASIBLE")
    return {"complete_target_positions": int(expected[:, :, 0].sum()),
            "channels": observed.shape[-1], "eligible_for_adoption": False}


def check_frozen_model_records(before, after, config_before, config_after):
    require(isinstance(before, dict) and bool(before) and before == after, "FROZEN_F_STATE_SIGNATURE_CHANGED")
    require(isinstance(config_before, dict) and bool(config_before) and config_before == config_after,
            "FROZEN_F_CONFIG_SIGNATURE_CHANGED")
    return {"state_signature_before_after_equal": True, "config_signature_before_after_equal": True,
            "actual_model_forward_independently_replayed": False}


def audit_recorded_times(row, raw_row):
    seconds, raw_seconds = row["seconds"], raw_row["seconds"]
    require(len(seconds) == len(raw_seconds) == 6
            and all(np.isfinite(value) and value > 0 for value in seconds + raw_seconds), "SIX_POSITIVE_RECORDED_TIMES")
    batch = row["batch"]
    require(batch in (1, 8) and batch == raw_row["batch"], "MATCHED_BATCH_COST_REQUIRED")
    ratios = [a / b for a, b in zip(seconds, raw_seconds)]
    expected = {"median_seconds": float(statistics.median(seconds)),
                "median_seconds_per_window": float(statistics.median(seconds)) / batch,
                "paired_RAW_median_ratio": float(statistics.median(ratios))}
    for key, value in expected.items():
        close(row[key], value, "cost:" + key, atol=1e-12, rtol=1e-12)
    require(row["paired_RAW_seconds"] == raw_seconds, "PAIRED_RAW_RECORDED_TIMES")
    close(row["paired_RAW_ratios"], ratios, "paired_RAW_ratios", atol=1e-12, rtol=1e-12)
    require(row["repeats"] == 6 and row["warmup_each_route"] == 2, "PREDECLARED_COST_REPEATS_WARMUP")
    return {**expected, "samples_checked": 6, "independently_remeasured_clock_or_memory": False}


def verified_arrays(record, check):
    """Use only after the root runner has explicitly authorized these saved assets."""
    check()
    path = Path(record["path"])
    require(path.is_absolute(), "ABSOLUTE_CANONICAL_ARTIFACT_REQUIRED")
    require(path.suffix == ".npz" and file_sha(path) == record["sha256"], "NPZ_FILE_SHA_MISMATCH")
    require(isinstance(record.get("arrays"), dict) and bool(record["arrays"]), "NPZ_ARRAY_MANIFEST_REQUIRED")
    check()
    with np.load(path, allow_pickle=False) as opened:
        values = {key: opened[key].copy() for key in opened.files}
    require(set(values) == set(record["arrays"]), "NPZ_ARRAY_KEYS_MISMATCH")
    for key, value in values.items():
        expected = record["arrays"][key]
        require(str(value.dtype) == expected["dtype"] and list(value.shape) == expected["shape"]
                and hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() == expected["sha256"],
                "NPZ_ARRAY_CONTENT_MISMATCH:" + key)
    check()
    return values


def verified_json(record, check):
    check()
    path = Path(record["path"])
    require(path.is_absolute() and file_sha(path) == record["sha256"], "JSON_FILE_SHA_MISMATCH")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    check()
    return value


def bind_canonical_npz(checked_record, canonical_record):
    checked = Path(checked_record["path"]).resolve()
    canonical = Path(canonical_record["path"])
    canonical = canonical.resolve() if canonical.is_absolute() else (ROOT / canonical).resolve()
    require(checked == canonical and checked_record["sha256"] == canonical_record["sha256"]
            and isinstance(canonical_record.get("arrays"), dict) and bool(canonical_record["arrays"]),
            "CANONICAL_PATH_FILE_SHA_ARRAY_MANIFEST_BINDING")
    return {**canonical_record, "path": str(canonical)}


def compare_normalized(recorded, recalculated, label):
    for key in ("mse", "mae", "per_channel_mse", "per_channel_mae"):
        actual = recorded[key]
        if key.startswith("per_channel"):
            require(len(actual) == len(recalculated[key]), "PER_CHANNEL_SCORE_SHAPE:" + label)
            actual = [0.0 if value is None and not recalculated["observed_counts"][index] else value
                      for index, value in enumerate(actual)]
        close(actual, recalculated[key], label + ":" + key, atol=1e-10, rtol=1e-10)
    for key in ("observed_counts", "active_channels"):
        require(recorded[key] == recalculated[key], "MASK_COUNT_MISMATCH:" + label)
    return 6


def compare_optional_scores(recorded, prediction, values, label, mask=None):
    observed = values["observed"] if mask is None else mask
    if not observed.any():
        require(recorded == {"status": "NO_OBSERVED_TARGETS", "normalized": None,
                              "physical_per_channel_diagnostic": None}, "EMPTY_TARGET_SCORE_SCOPE:" + label)
        return 0
    return compare_scores(recorded, scores(prediction, values["target"], observed, values["std"]), label)


def forecast_from_latents(raw_last, history_latents, future_latents, decoder):
    require(raw_last.dtype == history_latents.dtype == future_latents.dtype == np.float32
            and raw_last.ndim == 2 and history_latents.ndim == future_latents.ndim == 3
            and len(raw_last) == len(history_latents) == len(future_latents)
            and history_latents.shape[-1] == future_latents.shape[-1] == decoder.shape[0]
            and raw_last.shape[-1] == decoder.shape[1], "FIXED_FP32_LATENT_ANCHOR_D_SHAPES")
    return raw_last[:, None] + (future_latents - history_latents[:, -1:]) @ decoder.astype(np.float32)


def native_state_certificate(state_arrays, projection, metadata=None):
    require(set(state_arrays) == {"projection.center", "projection.encoder", "projection.decoder",
                                 "projection.decoder_bias", "head.weight", "head.bias"}, "EXACT_NATIVE_STATE_KEYS")
    frozen = {}
    for key in ("center", "encoder", "decoder", "decoder_bias"):
        expected = np.asarray(projection[key], np.float32)
        actual = state_arrays["projection." + key]
        exact(actual, expected, "frozen_native_projection:" + key)
        frozen[key] = actual
    k, c = projection["encoder"].shape[1], len(projection["center"])
    require(state_arrays["head.weight"].shape == (k, 48, 512)
            and state_arrays["head.bias"].shape == (k, 48)
            and all(value.dtype == np.float32 and np.isfinite(value).all() for value in state_arrays.values()),
            "NATIVE_FP32_K48_512_WEIGHT_BIAS_STATE")
    if metadata is not None:
        require(metadata["frozen_projection_sha256"] == torch_state_sha(frozen), "NATIVE_FROZEN_BUFFER_DIGEST")
        require(metadata["parameters"] == k * 48 * 513 and metadata["K"] == k
                and metadata["L"] == 512 and metadata["H"] == 48
                and metadata["encoder_decoder_frozen"] is True and metadata["E_D_optimized_for_native"] is False,
                "NATIVE_FIXED_ARCHITECTURE_E_D_SCOPE")
    return {"model_state_sha256": torch_state_sha(state_arrays), "frozen_projection_sha256": torch_state_sha(frozen),
            "channels": c, "K": k}


def audit_condition(condition_json, folder, bridge_npzs, projection_npzs=None, *, check=lambda: None):
    """Audit one completed route using the runner's exact registered schema."""
    condition = condition_json
    require(Path(condition["state_artifact"]["path"]).resolve().parent == Path(folder).resolve(),
            "CURRENT_RUN_PROJECTION_ARTIFACT_LOCATION")
    require(condition["status"] == "COMPLETE" and condition["bolt"]["status"] == "COMPLETE",
            "COMPLETED_PROJECTION_AND_BOLT_ROUTE_REQUIRED")
    state = projection_npzs if projection_npzs is not None else verified_arrays(condition["state_artifact"], check)
    require(set(state) == {"center", "encoder", "decoder", "decoder_bias", "metadata_json"}, "PROJECTION_NPZ_KEYS")
    for key in ("center", "encoder", "decoder", "decoder_bias"):
        require(state[key].dtype == np.float64, "SERIALIZED_FP64_PROJECTION_STATE:" + key)
    embedded = json.loads(str(state["metadata_json"].item()))
    require(embedded == {key: value for key, value in condition["fit_metadata"].items() if key != "state_array_sha256"},
            "PROJECTION_EMBEDDED_METADATA_BINDING")
    projection = check_projection_metadata(condition["fit_metadata"], state,
                                           bridge_npzs["fit"]["history"], bridge_npzs["fit"]["history_observed"])
    report = {"projection": projection, "bolt": {}, "native": {}, "oracle": {},
              "score_field_checks": 0, "saved_prediction_arrays_checked": 0,
              "F_forward_independently_replayed": False}
    bolt_predictions, native_predictions = {}, {}
    for split, values in bridge_npzs.items():
        check()
        saved = verified_arrays(condition["bolt"]["artifacts"][split], check)
        require(set(saved) == {"prediction", "latent_history", "latent_future", "origins"}, "BOLT_SAVED_ARRAY_KEYS")
        exact(saved["origins"], values["origins"], "bolt_origins:" + split)
        latent = encode(values["history"], state["center"], state["encoder"])
        encoded_error = close(saved["latent_history"], latent, "bolt_encoded_past:" + split)
        require(saved["latent_future"].shape == (len(values["history"]), 48, state["encoder"].shape[1])
                and saved["latent_future"].dtype == np.float32 and np.isfinite(saved["latent_future"]).all(),
                "FP32_N48K_ACTUAL_BOLT_FUTURE")
        anchored = forecast_from_latents(values["raw_last"], saved["latent_history"], saved["latent_future"], state["decoder"])
        anchored_error = close(saved["prediction"], anchored, "bolt_current_C_D_anchor:" + split)
        report["score_field_checks"] += compare_optional_scores(condition["bolt"]["scores"][split], saved["prediction"],
                                                                values, "bolt:" + split)
        report["saved_prediction_arrays_checked"] += 1
        report["bolt"][split] = {"encoder_max_abs": encoded_error, "anchored_decoder_max_abs": anchored_error}
        bolt_predictions[split] = saved["prediction"]
    require(set(condition["native"]) == {str(seed) for seed in SEEDS}, "EXACT_TWO_REGISTERED_NATIVE_SEEDS")
    for seed in SEEDS:
        check()
        entry, tune = condition["native"][str(seed)], bridge_npzs["tune"]
        metadata = entry["metadata"]
        require(entry["status"] == "COMPLETE" and metadata["seed"] == seed and metadata["updates"] == 600
                and metadata["batch_size"] == 4 and metadata["checkpoint_updates"] == list(CHECKPOINTS),
                "NATIVE_SEED_UPDATES_BATCH_CHECKPOINT_CONTRACT")
        require(metadata["optimizer"] == "AdamW" and metadata["learning_rate"] == .001
                and metadata["weight_decay"] == 0 and metadata["gradient_clip"] == 1.0
                and metadata["initial_weights_and_biases_zero"] is True
                and metadata["seed_changes_initial_weights"] is False
                and metadata["future_encoding_in_forward"] is False and metadata["new_backbone_training"] is False,
                "REGISTERED_NATIVE_OPTIMIZER_FORWARD_SCOPE")
        order = np.random.default_rng(seed).integers(0, len(bridge_npzs["fit"]["history"]), size=(600, 4))
        require(metadata["order_sha256"] == hashlib.sha256(order.tobytes()).hexdigest(), "NATIVE_FIT_ONLY_SEEDED_MINIBATCH_ORDER")
        require(len(entry["learning_curve"]) == 600, "ACTUAL_600_UPDATE_CURVE_REQUIRED")
        for update, item in enumerate(entry["learning_curve"], 1):
            require(item["update"] == update and np.isfinite(item["loss"]) and item["loss"] >= 0
                    and np.isfinite(item["gradient_norm_before_clip"]) and item["gradient_norm_before_clip"] >= 0,
                    "FINITE_ACTUAL_NATIVE_LEARNING_CURVE")
        checkpoints = entry["checkpoints"]
        require(tuple(item["update"] for item in checkpoints) == CHECKPOINTS, "EXACT_NATIVE_FIVE_CHECKPOINTS")
        checkpoint_rows, checkpoint_arrays, checkpoint_details = [], {}, []
        for checkpoint in checkpoints:
            check()
            arrays = verified_arrays(checkpoint["state_artifact"], check)
            certificate = native_state_certificate(arrays, state, metadata)
            require(certificate["model_state_sha256"] == checkpoint["state_sha256"], "NATIVE_CHECKPOINT_STATE_DIGEST")
            saved = verified_arrays(checkpoint["tune_prediction_artifact"], check)
            require(set(saved) == {"prediction", "origins"}, "CHECKPOINT_TUNE_SAVED_ARRAY_KEYS")
            exact(saved["origins"], tune["origins"], "native_checkpoint_TUNE_origins")
            latent = encode(tune["history"], state["center"], state["encoder"])
            future = native_latent_forecast(latent, arrays["head.weight"], arrays["head.bias"])
            prediction = forecast_from_latents(tune["raw_last"], latent, future, state["decoder"])
            difference = close(saved["prediction"], prediction, "native_checkpoint_forward_u" + str(checkpoint["update"]))
            recalculated = scores(saved["prediction"], tune["target"], tune["observed"], tune["std"])["normalized"]
            report["score_field_checks"] += compare_normalized(checkpoint["tune"], recalculated, "native_checkpoint_TUNE")
            report["saved_prediction_arrays_checked"] += 1
            checkpoint_rows.append({"update": checkpoint["update"], "tune_mse": recalculated["mse"]})
            checkpoint_arrays[checkpoint["update"]] = arrays
            checkpoint_details.append({"update": checkpoint["update"], "forward_max_abs": difference,
                                       "recalculated_tune_mse": recalculated["mse"]})
            if checkpoint["update"] == 0:
                require(np.count_nonzero(arrays["head.weight"]) == np.count_nonzero(arrays["head.bias"]) == 0,
                        "NATIVE_CP0_ZERO_WEIGHTS_BIASES")
                close(saved["prediction"], np.broadcast_to(tune["raw_last"][:, None], tune["target"].shape),
                      "native_CP0_original_C_persistence")
                require(metadata["initial_model_sha256"] == certificate["model_state_sha256"], "NATIVE_INITIAL_MODEL_DIGEST")
        chosen = select_checkpoint(checkpoint_rows)
        winner = checkpoints[chosen["selected_index"]]
        require(metadata["selected_update"] == chosen["selected_update"]
                and metadata["selected_state_sha256"] == winner["state_sha256"]
                and metadata["selected_tune"] == winner["tune"], "TUNE_EARLIEST_NATIVE_SELECTION_CERTIFICATE")
        selected_state = checkpoint_arrays[chosen["selected_update"]]
        split_details, native_predictions[str(seed)] = {}, {}
        for split, values in bridge_npzs.items():
            saved = verified_arrays(entry["artifacts"][split], check)
            require(set(saved) == {"prediction", "latent_history", "latent_future", "origins"}, "NATIVE_SAVED_ARRAY_KEYS")
            exact(saved["origins"], values["origins"], "native_selected_origins:" + split)
            latent_error = close(saved["latent_history"], encode(values["history"], state["center"], state["encoder"]),
                                 "native_frozen_E:" + split)
            future_error = close(saved["latent_future"], native_latent_forecast(saved["latent_history"],
                                 selected_state["head.weight"], selected_state["head.bias"]), "native_selected_temporal_head:" + split)
            anchored = forecast_from_latents(values["raw_last"], saved["latent_history"], saved["latent_future"], state["decoder"])
            prediction_error = close(saved["prediction"], anchored, "native_same_D_current_C_anchor:" + split)
            report["score_field_checks"] += compare_optional_scores(entry["scores"][split], saved["prediction"], values,
                                                                    "native_selected:" + str(seed) + ":" + split)
            report["saved_prediction_arrays_checked"] += 1
            split_details[split] = {"encoder_max_abs": latent_error, "latent_future_max_abs": future_error,
                                   "anchored_decoder_max_abs": prediction_error}
            native_predictions[str(seed)][split] = saved["prediction"]
        report["native"][str(seed)] = {"selection": chosen, "checkpoints": checkpoint_details, "splits": split_details,
                                       "optimizer_updates_independently_replayed": False}
    for split, values in bridge_npzs.items():
        check()
        entry = condition["oracle"][split]
        require(entry["role"] == "diagnostic_only" and entry["selection_used"] is False,
                "ORACLE_NEVER_DEPLOYMENT_OR_SELECTION")
        saved = verified_arrays(entry["artifact"], check)
        require(set(saved) == {"prediction", "latent_future", "evaluation_mask", "origins"}, "ORACLE_SAVED_ARRAY_KEYS")
        exact(saved["origins"], values["origins"], "oracle_origins:" + split)
        mask = oracle_mask(values["observed"])
        exact(saved["evaluation_mask"], mask, "oracle_complete_positions_mask:" + split)
        count = int(mask[:, :, 0].sum())
        require(entry["complete_target_positions"] == count
                and entry["total_target_positions"] == len(values["history"]) * 48, "ORACLE_COMPLETE_POSITION_COUNTS")
        safe_y = np.where(values["observed"], values["target"], np.float32(0.))
        expected_future = encode(safe_y, state["center"], state["encoder"])
        future_error = close(saved["latent_future"], expected_future, "oracle_only_true_future_diagnostic:" + split)
        expected = anchor_prediction(values["history"], values["raw_last"], saved["latent_future"],
                                     state["center"], state["encoder"], state["decoder"])
        prediction_error = close(saved["prediction"], expected, "oracle_same_D_current_C_anchor:" + split)
        comparisons = (("scores", saved["prediction"]), ("raw_scores_same_mask", values["raw_forecast"]),
                       ("bolt_scores_same_mask", bolt_predictions[split]))
        for key, prediction in comparisons:
            report["score_field_checks"] += compare_optional_scores(entry[key], prediction, values,
                                                                    "oracle_mask_comparison:" + key, mask)
        require(set(entry["native_scores_same_mask"]) == {str(seed) for seed in SEEDS}, "ORACLE_TWO_NATIVE_SAME_MASK")
        for seed in SEEDS:
            report["score_field_checks"] += compare_optional_scores(entry["native_scores_same_mask"][str(seed)],
                native_predictions[str(seed)][split], values, "oracle_mask_native:" + str(seed), mask)
        report["saved_prediction_arrays_checked"] += 1
        report["oracle"][split] = {"complete_target_positions": count, "latent_future_max_abs": future_error,
                                    "anchored_decoder_max_abs": prediction_error,
                                    "deployment_or_selection_use": False,
                                    "strict_lower_bound_for_fixed_decoder_or_all_decoders": False}
    perturbation = condition["future_label_perturbation"]
    require(perturbation["target_and_future_mask_changed"] is True and perturbation["bolt_exact_equal"] is True
            and perturbation["native_exact_equal"] == {str(seed): True for seed in SEEDS}, "RECORDED_ACTUAL_FUTURE_LABEL_FIXTURE")
    report["future_label_perturbation"] = {"recorded_scope": perturbation["scope"],
        "producer_record_and_executed_source_checked": True, "independently_replayed_in_audit": False,
        "whole_raw_future_replay_or_general_leakage_proof": False}
    return report, {"BOLT": bolt_predictions, "NATIVE": native_predictions}


def compare_parity_record(recorded, actual, reference, label, *, atol=1e-5, rtol=2e-6):
    maximum = close(actual, reference, label, atol=atol, rtol=rtol)
    error = actual.astype(np.float64) - reference.astype(np.float64)
    rms = float(np.sqrt(np.mean(error * error)))
    require(recorded["pass"] is True and recorded["atol"] == atol and recorded["rtol"] == rtol,
            "REGISTERED_PARITY_TOLERANCE:" + label)
    close(recorded["max_abs"], maximum, "parity_max_abs:" + label, atol=1e-12, rtol=1e-12)
    close(recorded["rmse"], rms, "parity_rmse:" + label, atol=1e-12, rtol=1e-12)
    return {"max_abs": maximum, "rmse": rms}


def audit_cost_dataset(cost, physical, development, scientific_predictions, check, *, sealed_unix):
    require(cost["status"] == "COMPLETE" and cost["routes"] == list(COST_ROUTES)
            and cost["expected_cells_this_dataset"] == 10 and cost["expected_cells_two_datasets"] == 20
            and len(cost["rows"]) == 10 and cost["warmup_each_route"] == 2 and cost["repeats"] == 6,
            "COMPLETE_MATCHED_TEN_COST_CELLS")
    require(cost["new_training_updates"] == 0 and cost["encoder_decoder_refit_in_cost"] is False
            and cost["protected_TEST_accessed"] is False and cost["independent_holdout"] is False,
            "COST_NO_REFIT_OR_NEW_HOLDOUT")
    normalized_history = causal_physical_preprocess(physical["history_physical"], physical["history_observed"],
                                                   development["mean"], development["std"])
    close(normalized_history, development["history"], "cost_physical_causal_preprocess")
    exact(physical["history_observed"], development["history_observed"], "cost_original_past_mask")
    if "origins" in physical:
        exact(physical["origins"], development["origins"], "cost_physical_bridge_origins")
    for key, actual in (("physical_past", physical["history_physical"]),
                        ("history_observed", physical["history_observed"]), ("FIT_mean", development["mean"]),
                        ("FIT_std", development["std"]), ("origins", development["origins"])):
        declared = cost["input_array_metadata"][key]
        require(declared == {"shape": list(actual.shape), "dtype": str(actual.dtype),
                              "sha256": hashlib.sha256(np.ascontiguousarray(actual).tobytes()).hexdigest()},
                "COST_PHYSICAL_PAST_FIT_SCALER_INPUT_CONTENT:" + key)
    require(cost["order_seed"] == 20261015, "PREDECLARED_COST_ORDER_SEED")
    generator = np.random.default_rng(20261015)
    order = generator.permutation(5)
    expected_schedule = [[COST_ROUTES[index] for index in np.roll(order, -repeat)] for repeat in range(5)]
    expected_schedule.append([COST_ROUTES[index] for index in generator.permutation(5)])
    require(cost["schedule"] == expected_schedule, "ACTUAL_FIVE_ROTATIONS_PLUS_EXTRA_PERMUTATION")
    table = {(row["route"], row["batch"]): row for row in cost["rows"]}
    require(len(table) == 10 and set(table) == {(route, batch) for route in COST_ROUTES for batch in (1, 8)},
            "EXACT_FIVE_ROUTE_TWO_BATCH_COST_MATRIX")
    source = cost["source_lineage"]
    require(file_sha(HERE / "whole_cost_v1.py") == source["whole_cost_source_sha256"]
            and file_sha(source["copied_preprocess_source"]) == source["copied_preprocess_source_sha256"]
            and source["legacy_scientific_runtime_imported"] is False, "WHOLE_COST_SOURCE_LINEAGE")
    details = []
    for (route, batch), row in table.items():
        check()
        require(row["status"] == "COMPLETE" and row["dataset"] == cost["dataset"], "COST_ROW_IDENTITY")
        arithmetic = audit_recorded_times(row, table[("RAW_BOLT", batch)])
        require(row["positions"] == [order.index(route) for order in expected_schedule]
                and sorted(row["positions"][:5]) == list(range(5)), "FIRST_FIVE_COST_POSITIONS_BALANCED")
        require(row["origins"] == development["origins"][:batch].tolist(), "SAME_FIRST_B_COST_ORIGINS")
        reference = verified_arrays(row["reference"], check)
        timed = verified_arrays(row["timed_outputs"], check)
        require(set(reference) == {"normalized", "physical", "scientific_normalized", "origins"}
                and set(timed) == {"normalized", "physical", "block", "position", "origins"}, "COST_OUTPUT_ARRAY_KEYS")
        exact(reference["origins"], development["origins"][:batch], "cost_reference_origins")
        exact(timed["origins"], development["origins"][:batch], "cost_timed_origins")
        exact(timed["block"], np.arange(6, dtype=np.int64), "cost_repeat_block")
        exact(timed["position"], np.asarray(row["positions"], np.int64), "cost_position_indices")
        require(reference["normalized"].shape == reference["physical"].shape == (batch, 48, len(development["mean"]))
                and timed["normalized"].shape == timed["physical"].shape == (6, batch, 48, len(development["mean"]))
                and all(value.dtype == np.float32 for value in (reference["normalized"], reference["physical"],
                                                                timed["normalized"], timed["physical"])),
                "FP32_REGISTERED_REFERENCE_TIMED_OUTPUT_SHAPES")
        exact(reference["scientific_normalized"], scientific_predictions[route][:batch], "cost_canonical_scientific_prediction")
        close(reference["normalized"], reference["scientific_normalized"], "cost_reference_scientific_parity")
        exact(reference["physical"], (reference["normalized"] * development["std"] + development["mean"]).astype(np.float32),
              "cost_reference_actual_CPU_physical_inverse")
        scientific_physical = (reference["scientific_normalized"] * development["std"] + development["mean"]).astype(np.float32)
        require(len(row["samples"]) == len(row["parity"]) == len(row["memory"]) == 6, "SIX_ACTUAL_COST_SAMPLES")
        for repeat, sample in enumerate(row["samples"]):
            check()
            require(sample["repeat"] == sample["block"] == repeat and sample["route"] == route
                    and sample["batch"] == batch and sample["dataset"] == cost["dataset"]
                    and sample["position"] == row["positions"][repeat], "SAME_PAIRED_COST_SAMPLE_IDENTITY")
            require(sample["perf_end_ns"] > sample["perf_start_ns"]
                    and sample["unix_end_ns"] >= sample["unix_start_ns"] > sealed_unix * 1e9,
                    "POSITIVE_COST_TIMER_AFTER_SELECTION_SEAL")
            close(sample["seconds"], (sample["perf_end_ns"] - sample["perf_start_ns"]) / 1e9,
                  "cost_recorded_perf_clock_arithmetic", atol=1e-12, rtol=1e-12)
            require(sample["seconds"] == row["seconds"][repeat]
                    and sample["parity"] == row["parity"][repeat]
                    and sample["memory"] == row["memory"][repeat], "COST_ROW_SAMPLE_BINDING")
            require(all(isinstance(value, int) and value >= 0 for value in sample["memory"].values()),
                    "NONNEGATIVE_RECORDED_SHARED_MEMORY")
            sample_path = Path(sample["sample_artifact"]["path"]).with_suffix(".json")
            literal = verified_json({"path": str(sample_path), "sha256": file_sha(sample_path)}, check)
            require(literal == sample, "COST_LITERAL_SAMPLE_JSON_BINDING")
            saved = verified_arrays(sample["sample_artifact"], check)
            require(set(saved) == {"normalized", "physical", "origins"}, "COST_SAMPLE_SAVED_ARRAY_KEYS")
            exact(saved["origins"], development["origins"][:batch], "cost_sample_origins")
            exact(saved["normalized"], timed["normalized"][repeat], "cost_sample_stack_normalized")
            exact(saved["physical"], timed["physical"][repeat], "cost_sample_stack_physical")
            exact(saved["physical"], (saved["normalized"] * development["std"] + development["mean"]).astype(np.float32),
                  "cost_sample_actual_CPU_physical_inverse")
            for key, actual, expected, atol, rtol in (
                    ("normalized", saved["normalized"], reference["normalized"], 1e-5, 2e-6),
                    ("physical", saved["physical"], reference["physical"], 2e-3, 2e-5),
                    ("scientific_normalized", saved["normalized"], reference["scientific_normalized"], 1e-5, 2e-6),
                    ("scientific_physical", saved["physical"], scientific_physical, 2e-3, 2e-5)):
                compare_parity_record(sample["parity"][key], actual, expected, "cost:" + route + ":" + key,
                                      atol=atol, rtol=rtol)
        require(row["memory_counter_reset_inside_module"] is False and row["stage_decomposition"] is False
                and row["optimized_native_implementation_claim"] is False, "RECORDED_WHOLE_SHARED_MEMORY_COST_SCOPE")
        if route.endswith("_NATIVE"):
            require(row["selected_metadata"]["native_seed"] == 92601 and row["selected_metadata"]["head_device"] == "cpu"
                    and row["selected_metadata"]["E_D_frozen"] is True, "REGISTERED_COST_NATIVE_SEED_FROZEN_CPU_ROUTE")
        details.append({"route": route, "batch": batch, **arithmetic})
    return {"rows": details, "cells_checked": len(details), "recorded_timer_samples_checked": len(details) * 6,
            "clock_memory_or_F_forward_independently_remeasured": False,
            "first_five_rounds_balanced_sixth_extra": True,
            "same_first_B_origins_repeated_not_independent_draws": True}


def incomplete_producer_report(result):
    require(result.get("status") != "COMPLETE", "INCOMPLETE_STATUS_REQUIRED")
    return {"status": "NOT_AUDITED_INCOMPLETE_PRODUCER", "producer_status": result.get("status"),
            "producer_failure": result.get("failure"), "scientific_assets_opened": False,
            "full_scope_completion_or_PASS_claimed": False,
            "reported_completed_projection_fits_not_independently_audited": result.get("projection_fits", 0),
            "reported_completed_native_fits_not_independently_audited": result.get("native_training_fits", 0),
            "resume": "Preserve failed producer; repair only its failed path and use a new run and audit output tag."}


def audit_result(path, meter):
    check = meter.check
    check()
    path = Path(path).resolve()
    require(path.parent.parent == HERE / "runs" and path.name == "RESULT.json", "EXPLICIT_CURRENT_FOLLOWUP_RUN_RESULT")
    record = {"path": str(path), "sha256": file_sha(path)}
    result = verified_json(record, check)
    if result.get("status") != "COMPLETE":
        return {**incomplete_producer_report(result), "producer": record}
    require(result["scientific_execution"] is True and result["protected_TEST_accessed"] is False
            and result["independent_holdout"] is False and result["new_backbone_training"] is False
            and result["native_training_fits"] == 8 and result["projection_fits"] == 4
            and result["hard_elapsed_or_cumulative_resource_caps"] is None, "COMPLETED_REGISTERED_FOUR_EIGHT_RUN_SCOPE")
    require(set(result["datasets"]) == set(result["cost"]) == set(DATASETS), "EXACT_TWO_FOLLOWUP_DATASETS")
    for name, expected_path in (("authorization", HERE / "AUTHORIZATION_SCOPE_v1.json"),
                                ("continuation_manifest", HERE / "CONTINUATION_MANIFEST_v1.json")):
        expected_record = result[name]
        require(Path(expected_record["path"]).resolve() == expected_path
                and expected_record["sha256"] == file_sha(expected_path), "SAME_ACTUAL_CONTINUATION_AUTH_MANIFEST:" + name)
    source_checks = 0
    for entry in result["source_records"] + result["source_snapshots"]:
        check()
        require(file_sha(entry["path"]) == entry["sha256"], "PRODUCER_SOURCE_FILE_BYTES_CHANGED")
        source_checks += 1
        if "snapshot_path" in entry:
            require(file_sha(entry["snapshot_path"]) == entry["sha256"], "EXECUTED_SOURCE_SNAPSHOT_CHANGED")
            source_checks += 1
    require(set(result["F_before"]) == {"old_algorithm", "bi_algorithm"}
            and result["F_before"] == result["F_after"], "DUAL_F_STATE_SIGNATURE_BEFORE_AFTER")
    for name in ("model_binary", "model_config"):
        before, after = result[name + "_before"], result[name + "_after"]
        require(Path(before["path"]).resolve() == Path(after["path"]).resolve()
                and before["sha256"] == after["sha256"] == file_sha(after["path"]), "ACTUAL_F_FILE_BYTES_UNCHANGED:" + name)
        source_checks += 1
    behavior = result["runtime_behavior_contract"]
    require(behavior["evaluation"] is True and behavior["all_backbone_parameters_frozen"] is True
            and behavior["context_length"] == 2048 and behavior["prediction_length"] == 64
            and behavior["input_patch_size"] == behavior["input_patch_stride"] == 16
            and behavior["median_index"] == 4 and behavior["returned_horizon"] == 48
            and behavior["quantiles"] == [.1, .2, .3, .4, .5, .6, .7, .8, .9]
            and behavior["pipeline_predict_not_used"] is True, "EXACT_RECORDED_FROZEN_BOLT_CONFIG")
    require(behavior["normalizer_before"]["class"] == "InstanceNorm"
            and behavior["normalizer_before"]["eps"] == behavior["normalizer_after"]["eps"]
            and behavior["normalizer_before"]["use_arcsinh"] == behavior["normalizer_after"]["use_arcsinh"]
            and set(behavior["normalizer_fixture_forward_parity"]) == {"standardized", "location", "scale"}
            and all(value["exact_equal_including_nan"] is True for value in behavior["normalizer_fixture_forward_parity"].values()),
            "RECORDED_ACTUAL_NORMALIZER_FORWARD_PARITY")
    seal = verified_json(result["selection_lock"], check)
    require(seal["status"] == "SEALED_BEFORE_NEW_DEV_AND_ORACLE" and set(seal["datasets"]) == set(DATASETS),
            "SELECTION_LOCK_BEFORE_DEVELOPMENT_OR_ORACLE")
    receipt = verified_json(result["resource_terminal"], check)
    require(receipt["status"] == "COMPLETE" and receipt["elapsed_or_cumulative_budget_caps"] is None
            and receipt["started_unix"] <= seal["sealed_unix"] <= receipt["ended_unix"], "COMPLETED_ACTUAL_PRODUCER_RESOURCE_RECEIPT")
    report = {"status": "PASS", "producer": record, "datasets": {}, "cost": {},
              "F_frozen": {"dual_state_signatures_equal": True, "actual_binary_config_bytes_rehashed": True,
                           "normalizer_forward_producer_fixture_record_checked": True,
                           "F_forward_or_model_tensor_hash_independently_replayed": False},
              "totals": {"projection_states_checked": 0, "native_fits_checked": 0,
                         "checkpoint_states_checked": 0, "score_field_checks": 0,
                         "saved_prediction_arrays_checked": 0, "cost_cells_checked": 0, "recorded_timer_samples_checked": 0},
              "source_file_checks": source_checks, "protected_TEST_accessed": False,
              "new_training_or_GPU_execution": False, "independent_generalization_claimed": False,
              "oracle_is_fixed_E_Y_diagnostic_not_optimal_latent_or_strict_error_floor": True}
    for dataset, (channels, k, development, development_count, parent_name) in DATASETS.items():
        check()
        row = result["datasets"][dataset]
        require(row["status"] == "COMPLETE" and row["K"] == k
                and set(row["inputs"]) == {"fit", "tune", development}
                and set(row["projections"]) == {"PCA", "FORECA"}, "REGISTERED_DATASET_K_SPLITS_PROJECTIONS")
        parent_path = BASE / "runs" / parent_name / "RESULT.json"
        parent_record = {"path": str(parent_path), "sha256": file_sha(parent_path)}
        parent = verified_json(parent_record, check)
        require(parent["status"] == "COMPLETE" and parent["dataset"] == dataset, "COMPLETED_ORIGINAL_INPUT_PRODUCER")
        bridge_record = parent["bridge_provenance"]["input_bridge"]
        bridge = verified_json(bridge_record, check)
        require(bridge["status"] == "COMPLETE" and isinstance(bridge["protected_test_guarded"], list)
                and bool(bridge["protected_test_guarded"]),
                "COMPLETE_ORIGINAL_CANONICAL_INPUT_BRIDGE")
        raw, splits = {}, {}
        for split, count in (("fit", 256), ("tune", 64), (development, development_count)):
            require(row["inputs"][split] == parent["inputs"][split], "EXACT_ORIGINAL_INPUT_RECORD_REUSE:" + split)
            raw[split] = verified_arrays(row["inputs"][split], check)
            splits[split] = check_raw_split(raw[split], channels=channels, windows=count)
            require(row["data_contract"][split]["windows"] == count
                    and row["data_contract"][split]["raw_channels"] == channels
                    and row["data_contract"][split]["latent_channels"] == k, "DECLARED_ORIGINAL_SPLIT_DIMENSION_CONTRACT")
            for key in ("mean", "std"):
                exact(raw[split][key], raw["fit"][key], "same_FIT_only_raw_scaler:" + split + ":" + key)
            require(raw[split]["raw_forecast"].shape == raw[split]["target"].shape
                    and raw[split]["raw_forecast"].dtype == np.float32 and np.isfinite(raw[split]["raw_forecast"]).all(),
                    "ORIGINAL_FP32_RAW_BOLT_SAVED_PREDICTION")
            report["totals"]["score_field_checks"] += compare_optional_scores(row["raw_scores"][split], raw[split]["raw_forecast"],
                                                                              raw[split], "raw_reference:" + dataset + ":" + split)
        for split in ("fit", "tune"):
            parity = row["raw_parity"][split]
            require(parity["pass"] is True and parity["atol"] == 1e-5 and parity["rtol"] == 2e-6
                    and np.isfinite(parity["max_abs"]) and parity["max_abs"] >= 0
                    and np.isfinite(parity["rmse"]) and parity["rmse"] >= 0, "RECORDED_ACTUAL_RAW_PARITY")
        dataset_report = {"original_input_producer": parent_record, "original_canonical_input_bridge": bridge_record,
                          "splits": splits, "projections": {},
                          "raw_model_parity_fixture_independently_replayed": False}
        science = {"RAW_BOLT": raw[development]["raw_forecast"]}
        for route in ("PCA", "FORECA"):
            condition_report, predictions = audit_condition(row["projections"][route], path.parent, raw, check=check)
            dataset_report["projections"][route] = condition_report
            for seed in SEEDS:
                require(seal["datasets"][dataset][route][str(seed)] == condition_report["native"][str(seed)]["selection"]["selected_update"],
                        "SELECTED_NATIVE_MODEL_MATCHES_PRE_DEVELOPMENT_LOCK")
            science[route + "_BOLT"] = predictions["BOLT"][development]
            science[route + "_NATIVE"] = predictions["NATIVE"]["92601"][development]
            report["totals"]["projection_states_checked"] += 1
            report["totals"]["native_fits_checked"] += 2
            report["totals"]["checkpoint_states_checked"] += 10
            for key in ("score_field_checks", "saved_prediction_arrays_checked"):
                report["totals"][key] += condition_report[key]
        physical_candidates = [entry for entry in bridge["artifacts"]
                               if entry["dataset"] == dataset and entry["split"] == development]
        require(len(physical_candidates) == 1, "EXACT_ORIGINAL_DEVELOPMENT_PHYSICAL_BRIDGE_ARTIFACT")
        canonical_physical = bind_canonical_npz(row["physical_bridge"], physical_candidates[0])
        dataset_report["canonical_physical_bridge"] = canonical_physical
        physical = verified_arrays(canonical_physical, check)
        cost = audit_cost_dataset(result["cost"][dataset], physical, raw[development], science, check, sealed_unix=seal["sealed_unix"])
        report["cost"][dataset] = cost
        report["totals"]["cost_cells_checked"] += cost["cells_checked"]
        report["totals"]["recorded_timer_samples_checked"] += cost["recorded_timer_samples_checked"]
        report["datasets"][dataset] = dataset_report
    require(report["totals"]["projection_states_checked"] == 4 and report["totals"]["native_fits_checked"] == 8
            and report["totals"]["checkpoint_states_checked"] == 40
            and report["totals"]["cost_cells_checked"] == 20 and report["totals"]["recorded_timer_samples_checked"] == 120,
            "COMPLETE_REGISTERED_AUDIT_SCOPE_COUNTS")
    require(file_sha(path) == record["sha256"], "PRODUCER_RESULT_CHANGED_DURING_AUDIT")
    return report


def execute(result_path, output_path):
    from continuation_runtime_v1 import Meter, write_new
    output_path = Path(output_path).resolve()
    require(not output_path.exists(), "NEW_UNIQUE_AUDIT_OUTPUT_REQUIRED")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meter = Meter("audit_projection_" + output_path.stem, gpu=False)
    source = {"path": str(Path(__file__).resolve()), "sha256": file_sha(__file__)}
    report, error = None, None
    try:
        meter.protected_check()
        initialize_numpy()
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2):
            report = audit_result(result_path, meter)
        meter.protected_check()
        require(file_sha(__file__) == source["sha256"], "INDEPENDENT_AUDIT_SOURCE_CHANGED_DURING_EXECUTION")
    except Exception as failure:
        error = {"type": type(failure).__name__, "message": str(failure)}
        report = {"status": "FAILED_PRESERVED", "failure": error, "full_scope_PASS_claimed": False,
                  "result_path": str(Path(result_path).resolve())}
    finally:
        report["audit_source"] = source
        report["resource_terminal"] = meter.close(report["status"], error)
        write_new(output_path, report)
    print(json.dumps({"status": report["status"], "output": str(output_path), "totals": report.get("totals"),
                      "failure": error}, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output", required=True)
    raise SystemExit(execute(**{key + "_path": value for key, value in vars(parser.parse_args()).items()}))
