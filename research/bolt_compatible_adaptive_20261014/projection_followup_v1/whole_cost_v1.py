"""Whole deployment timers; caller supplies checked past arrays and frozen predictors."""
from pathlib import Path
import hashlib
import json
import time

import numpy as np


ROUTES = ("RAW_BOLT", "PCA_BOLT", "FORECA_BOLT", "PCA_NATIVE", "FORECA_NATIVE")
ORDER_SEED = 20261015
WARMUP, REPEATS = 2, 6
NORM_ATOL, NORM_RTOL = 1e-5, 2e-6
PHYSICAL_ATOL, PHYSICAL_RTOL = 2e-3, 2e-5
HERE = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def array_record(value):
    array = np.asarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json_once(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(_jsonable(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def save_outputs(path, arrays):
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    return {"path": str(Path(path).resolve()), "bytes": Path(path).stat().st_size,
            "sha256": sha(path), "arrays": {key: array_record(value) for key, value in arrays.items()}}


def balanced_schedule(seed=ORDER_SEED):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ROUTES))
    schedule = [[ROUTES[index] for index in np.roll(order, -repeat)]
                for repeat in range(len(ROUTES))]
    schedule.append([ROUTES[index] for index in rng.permutation(len(ROUTES))])
    return schedule


def fill_vectorized(physical, observed, mean):
    steps = np.arange(1, physical.shape[1] + 1, dtype=np.int32)[None, :, None]
    last_indices = np.maximum.accumulate(np.where(observed, steps, 0), axis=1)
    initial = np.broadcast_to(mean, (len(physical), 1, len(mean)))
    values = np.concatenate((initial, physical), axis=1)
    return np.take_along_axis(values, last_indices, axis=1).astype(np.float32)


def preprocess(physical, observed, mean, std):
    return ((fill_vectorized(physical, observed, mean) - mean) / std).astype(np.float32)


def parity(actual, reference, atol=NORM_ATOL, rtol=NORM_RTOL):
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.shape != reference.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        return {"pass": False, "reason": "shape_or_nonfinite", "atol": atol, "rtol": rtol}
    difference = actual.astype(np.float64) - reference.astype(np.float64)
    return {"pass": bool(np.all(np.abs(difference) <= atol + rtol * np.abs(reference))),
            "max_abs": float(np.max(np.abs(difference))),
            "rmse": float(np.sqrt(np.mean(difference * difference))), "atol": atol, "rtol": rtol}


def validate_inputs(physical, observed, mean, std, origins, predictors, references, metadata):
    if (physical.dtype != np.float32 or physical.ndim != 3 or physical.shape[1] != 512
            or len(physical) < 8 or physical.shape[2] < 1 or observed.dtype != np.bool_
            or observed.shape != physical.shape or not np.isfinite(physical[observed]).all()):
        raise ValueError("OBSERVED_FINITE_FP32_N512C_PAST_AND_BOOL_MASK_REQUIRED")
    c = physical.shape[-1]
    if (mean.dtype != np.float32 or std.dtype != np.float32 or mean.shape != (c,) or std.shape != (c,)
            or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0)
            or origins.dtype != np.int64 or origins.shape != (len(physical),)):
        raise ValueError("FINITE_FP32_FIT_STATS_AND_INT64_ORIGINS_REQUIRED")
    if set(predictors) != set(ROUTES) or set(references) != set(ROUTES) or not set(ROUTES).issubset(metadata):
        raise ValueError("EXACT_FIVE_COST_ROUTES_AND_METADATA_REQUIRED")
    for route in ROUTES:
        source = np.asarray(references[route])
        if (not callable(predictors[route]) or source.dtype != np.float32
                or source.shape != (len(physical), 48, c) or not np.isfinite(source).all()):
            raise ValueError("FP32_N48C_SAVED_PREDICTION_AND_CALLABLE_REQUIRED:" + route)


def whole_predict(predictor, physical, observed, mean, std):
    normalized_history = preprocess(physical, observed, mean, std)
    if not np.isfinite(normalized_history).all():
        raise ValueError("NONFINITE_NORMALIZED_PAST")
    normalized = np.asarray(predictor(normalized_history))
    if normalized.dtype != np.float32 or normalized.shape != (len(physical), 48, physical.shape[-1]):
        raise ValueError("NORMALIZED_FP32_N48C_DEPLOYMENT_OUTPUT_REQUIRED")
    if not np.isfinite(normalized).all():
        raise ValueError("NONFINITE_DEPLOYMENT_PREDICTION")
    physical_prediction = (normalized * std + mean).astype(np.float32, copy=False).copy()
    if not np.isfinite(physical_prediction).all():
        raise ValueError("NONFINITE_PHYSICAL_CPU_RETURN")
    return normalized, physical_prediction


def measure_cost(dataset, physical, history_observed, mean, std, origins, predictors,
                 saved_normalized_predictions, metadata, outputdirectory, check,
                 synchronize, memory_snapshot):
    """Measure ten cells for one dataset; data/model setup and guards belong to caller."""
    physical = np.asarray(physical)
    observed = np.asarray(history_observed)
    mean, std, origins = np.asarray(mean), np.asarray(std), np.asarray(origins)
    validate_inputs(physical, observed, mean, std, origins, predictors, saved_normalized_predictions, metadata)
    output = Path(outputdirectory)
    output.mkdir(parents=True, exist_ok=False)
    schedule = balanced_schedule()
    result = {"dataset": dataset, "status": "RUNNING", "rows": [], "warmup_each_route": WARMUP,
              "repeats": REPEATS, "routes": list(ROUTES), "order_seed": ORDER_SEED, "schedule": schedule,
              "schedule_scope": "First five circular rotations balance every route over all five positions; sixth seeded permutation is extra, so all six are not perfectly balanced",
              "sampling_scope": "Same first B1/B8 past origins repeated six times and shared across all five routes",
              "expected_cells_this_dataset": 10, "expected_cells_two_datasets": 20,
              "new_training_updates": 0, "encoder_decoder_refit_in_cost": False,
              "protected_TEST_accessed": False, "independent_holdout": False,
              "input_array_metadata": {name: array_record(value) for name, value in
                  (("physical_past", physical), ("history_observed", observed), ("FIT_mean", mean),
                   ("FIT_std", std), ("origins", origins))},
              "source_metadata": metadata,
              "source_lineage": {"whole_cost_source_sha256": sha(Path(__file__)),
                  "copied_preprocess_source": str(HERE.parent / "cost_readout_v1.py"),
                  "copied_preprocess_source_sha256": sha(HERE.parent / "cost_readout_v1.py"),
                  "copied_functions": ["fill_vectorized", "preprocess"],
                  "legacy_scientific_runtime_imported": False}}
    try:
        for batch in (1, 8):
            check()
            folder = output / f"B{batch}"
            folder.mkdir(exist_ok=False)
            values, masks = physical[:batch], observed[:batch]
            chosen_origins = origins[:batch].copy()
            references = {}
            for route in ROUTES:
                check()
                normalized, physical_value = whole_predict(predictors[route], values, masks, mean, std)
                synchronize()
                source = np.asarray(saved_normalized_predictions[route])[:batch]
                scientific_physical = (source * std + mean).astype(np.float32)
                checks = {"normalized": parity(normalized, source),
                          "physical": parity(physical_value, scientific_physical, PHYSICAL_ATOL, PHYSICAL_RTOL)}
                reference_record = save_outputs(folder / (route + "_REFERENCE.npz"), {
                    "normalized": normalized, "physical": physical_value,
                    "scientific_normalized": source, "origins": chosen_origins})
                if not all(value["pass"] for value in checks.values()):
                    raise ValueError("SCIENTIFIC_REFERENCE_PARITY_FAILED:" + route)
                references[route] = {"normalized": normalized.copy(), "physical": physical_value.copy(),
                                     "source": source, "scientific_physical": scientific_physical,
                                     "reference_artifact": reference_record, "scientific_parity": checks}
            for order in schedule[:WARMUP]:
                for route in order:
                    check()
                    whole_predict(predictors[route], values, masks, mean, std)
                    synchronize()
            samples = {route: [] for route in ROUTES}
            for repeat, order in enumerate(schedule):
                for position, route in enumerate(order):
                    check()
                    synchronize()
                    unix_start_ns = time.time_ns()
                    perf_start_ns = time.perf_counter_ns()
                    normalized, physical_value = whole_predict(predictors[route], values, masks, mean, std)
                    synchronize()
                    perf_end_ns = time.perf_counter_ns()
                    unix_end_ns = time.time_ns()
                    seconds = (perf_end_ns - perf_start_ns) / 1e9
                    memory = _jsonable(memory_snapshot())
                    reference = references[route]
                    checks = {"normalized": parity(normalized, reference["normalized"]),
                              "physical": parity(physical_value, reference["physical"], PHYSICAL_ATOL, PHYSICAL_RTOL),
                              "scientific_normalized": parity(normalized, reference["source"]),
                              "scientific_physical": parity(physical_value, reference["scientific_physical"], PHYSICAL_ATOL, PHYSICAL_RTOL)}
                    artifact = save_outputs(folder / f"{route}_r{repeat}.npz", {
                        "normalized": normalized, "physical": physical_value, "origins": chosen_origins})
                    sample = {"dataset": dataset, "route": route, "batch": batch, "repeat": repeat,
                              "block": repeat, "position": position, "seconds": seconds,
                              "perf_start_ns": perf_start_ns, "perf_end_ns": perf_end_ns,
                              "unix_start_ns": unix_start_ns, "unix_end_ns": unix_end_ns,
                              "parity": checks, "memory": memory, "sample_artifact": artifact}
                    write_json_once(folder / f"{route}_r{repeat}.json", sample)
                    samples[route].append({**sample, "normalized": normalized.copy(), "physical": physical_value.copy()})
                    if not all(value["pass"] for value in checks.values()):
                        raise ValueError("TIMED_OUTPUT_PARITY_FAILED:" + route)
                    check()
            raw_seconds = [sample["seconds"] for sample in samples["RAW_BOLT"]]
            for route in ROUTES:
                records = samples[route]
                seconds = [record["seconds"] for record in records]
                ratios = [value / raw if raw > 0 else None for value, raw in zip(seconds, raw_seconds)]
                artifact = save_outputs(folder / (route + "_TIMED.npz"), {
                    "normalized": np.stack([record["normalized"] for record in records]),
                    "physical": np.stack([record["physical"] for record in records]),
                    "block": np.arange(REPEATS, dtype=np.int64),
                    "position": np.array([record["position"] for record in records], dtype=np.int64),
                    "origins": chosen_origins})
                result["rows"].append({"dataset": dataset, "route": route, "status": "COMPLETE", "batch": batch,
                    "warmup_each_route": WARMUP, "repeats": len(records), "seconds": seconds,
                    "median_seconds": float(np.median(seconds)),
                    "median_seconds_per_window": float(np.median(seconds)) / batch,
                    "paired_RAW_seconds": raw_seconds, "paired_RAW_ratios": ratios,
                    "paired_RAW_median_ratio": float(np.median(ratios)) if all(value is not None for value in ratios) else None,
                    "positions": [record["position"] for record in records], "origins": chosen_origins.tolist(),
                    "parity": [record["parity"] for record in records], "memory": [record["memory"] for record in records],
                    "selected_metadata": metadata[route], "samples": [{key: value for key, value in record.items()
                         if key not in ("normalized", "physical")} for record in records],
                    "timed_outputs": artifact, "reference": references[route]["reference_artifact"],
                    "whole_includes": "Actual causal CPU fill, FIT normalization, callback including H2D/F/D2H where used, CPU inverse, materialized physical CPU return, final synchronization",
                    "whole_excludes": "Data/model/state load, scientific references, warmup, output save, hashes, guards and memory snapshots outside timer",
                    "memory_scope": "Caller-provided shared resident frozen F and projection/native objects; total process/counter observations, not isolated method allocation",
                    "memory_counter_reset_inside_module": False,
                    "balanced_scope": result["schedule_scope"], "stage_decomposition": False,
                    "optimized_native_implementation_claim": False})
        result["status"] = "COMPLETE"
    except Exception as failure:
        result.update(status="FAILED_PRESERVED", failure={"type": type(failure).__name__, "message": str(failure)})
        raise
    finally:
        write_json_once(output / "RESULT.json", result)
    return result
