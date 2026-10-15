"""Ten-route fixed-FIT-D whole costs; source import never loads scientific assets."""
from pathlib import Path
import hashlib
import json
import time

import numpy as np


ROUTES = ("RAW_BOLT", "PCA_BOLT", "FIT_TARGET_BOLT", "RAW_SUBSET_BOLT",
          "PCA_NATIVE_s92601", "PCA_NATIVE_s92602", "FIT_TARGET_NATIVE_s92601", "FIT_TARGET_NATIVE_s92602",
          "RAW_SUBSET_NATIVE_s92601", "RAW_SUBSET_NATIVE_s92602")
ORDER_SEED = 20261015
WARMUP, REPEATS = 2, 10
NORM_ATOL, NORM_RTOL = 1e-5, 2e-6
PHYSICAL_ATOL, PHYSICAL_RTOL = 2e-3, 2e-5
HERE = Path(__file__).resolve().parent
COPIED_TIMER_SHA = "6779646de2a809c625a2b503b60ac03a64ae72826b15b8efad1d6d2213ae6cc2"


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def array_record(value):
    array = np.asarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}


def _jsonable(value):
    if isinstance(value, dict): return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)): return [_jsonable(child) for child in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    return value


def write_json_once(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(_jsonable(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def save_outputs(path, arrays):
    with Path(path).open("xb") as handle: np.savez_compressed(handle, **arrays)
    return {"path": str(Path(path).resolve()), "bytes": Path(path).stat().st_size, "sha256": sha(path),
            "arrays": {key: array_record(value) for key, value in arrays.items()}}


def balanced_schedule(seed=ORDER_SEED):
    order = np.random.default_rng(seed).permutation(len(ROUTES))
    return [[ROUTES[index] for index in np.roll(order, -repeat)] for repeat in range(REPEATS)]


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
            "max_abs": float(np.max(np.abs(difference))), "rmse": float(np.sqrt(np.mean(difference * difference))),
            "atol": atol, "rtol": rtol}


def validate_inputs(physical, observed, mean, std, origins, predictors, references, metadata, unavailable):
    if (physical.dtype != np.float32 or physical.ndim != 3 or physical.shape[1] != 512 or len(physical) < 8
            or physical.shape[2] < 1 or observed.dtype != np.bool_ or observed.shape != physical.shape
            or not np.isfinite(physical[observed]).all()):
        raise ValueError("OBSERVED_FINITE_FP32_N512C_PAST_AND_BOOL_MASK_REQUIRED")
    c = physical.shape[-1]
    if (mean.dtype != np.float32 or std.dtype != np.float32 or mean.shape != (c,) or std.shape != (c,)
            or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0)
            or origins.dtype != np.int64 or origins.shape != (len(physical),)):
        raise ValueError("FINITE_FP32_FIT_STATS_AND_INT64_ORIGINS_REQUIRED")
    if (not set(unavailable) <= set(ROUTES) or set(predictors) != set(ROUTES)-set(unavailable)
            or set(references) != set(predictors) or not set(ROUTES) <= set(metadata)):
        raise ValueError("DECLARED_TEN_ROUTES_WITH_EXPLICIT_UNAVAILABLE_PATHS_REQUIRED")
    for route in predictors:
        source = np.asarray(references[route])
        if (not callable(predictors[route]) or source.dtype != np.float32
                or source.shape != (len(physical), 48, c) or not np.isfinite(source).all()):
            raise ValueError("FP32_N48C_SAVED_PREDICTION_AND_CALLABLE_REQUIRED:"+route)


def whole_predict(predictor, physical, observed, mean, std):
    normalized_history = preprocess(physical, observed, mean, std)
    if not np.isfinite(normalized_history).all(): raise ValueError("NONFINITE_NORMALIZED_PAST")
    normalized = np.asarray(predictor(normalized_history, observed))
    if normalized.dtype != np.float32 or normalized.shape != (len(physical), 48, physical.shape[-1]):
        raise ValueError("NORMALIZED_FP32_N48C_DEPLOYMENT_OUTPUT_REQUIRED")
    if not np.isfinite(normalized).all(): raise ValueError("NONFINITE_DEPLOYMENT_PREDICTION")
    physical_prediction = (normalized * std + mean).astype(np.float32, copy=False).copy()
    if not np.isfinite(physical_prediction).all(): raise ValueError("NONFINITE_PHYSICAL_CPU_RETURN")
    return normalized, physical_prediction


def measure_cost(dataset, physical, history_observed, mean, std, origins, predictors,
                 saved_normalized_predictions, metadata, outputdirectory, check, synchronize,
                 memory_snapshot, unavailable_routes=None):
    physical, observed = np.asarray(physical), np.asarray(history_observed)
    mean, std, origins = np.asarray(mean), np.asarray(std), np.asarray(origins)
    unavailable = dict(unavailable_routes or {})
    validate_inputs(physical, observed, mean, std, origins, predictors, saved_normalized_predictions, metadata, unavailable)
    output = Path(outputdirectory); output.mkdir(parents=True, exist_ok=False)
    schedule = balanced_schedule()
    result = {"dataset": dataset, "status": "RUNNING", "routes": list(ROUTES), "rows": [], "failed_paths": [],
        "recorded_timer_samples": 0, "attempted_timer_calls": 0, "completed_cost_cells": 0,
        "warmup_each_route": WARMUP, "repeats": REPEATS, "order_seed": ORDER_SEED, "schedule": schedule,
        "schedule_scope": "Ten circular rotations; each declared route once at each of ten positions; failed slots remain recorded",
        "sampling_scope": "Same first B1/B8 origins repeated ten times; not independent evaluation draws",
        "expected_cells_this_dataset": 20, "expected_cells_two_datasets": 40,
        "new_training_updates": 0, "encoder_refit_in_cost": False, "per_origin_decoder_fit": False,
        "fixed_FIT_decoder_loaded_before_timer": True, "fit_input_binding_hashes_inside_timer": False,
        "protected_TEST_accessed": False, "independent_holdout": False,
        "input_array_metadata": {name: array_record(value) for name, value in (("physical_past", physical),
            ("history_observed", observed), ("FIT_mean", mean), ("FIT_std", std), ("origins", origins))},
        "source_metadata": metadata, "source_lineage": {"whole_cost_source_sha256": sha(__file__),
            "copied_timer_source": str(HERE/"whole_matched_shrunk_native_cost_v1.py"), "copied_timer_source_sha256": COPIED_TIMER_SHA,
            "copied_preprocess_source": str(HERE.parent/"cost_readout_v1.py"),
            "copied_preprocess_source_sha256": sha(HERE.parent/"cost_readout_v1.py"),
            "copied_functions": ["fill_vectorized", "preprocess", "parity"], "legacy_scientific_runtime_imported": False}}
    def failed(route, batch, stage, error, **timing):
        entry = {"route": route, "batch": batch, "stage": stage, "status": "FAILED_PATH_PRESERVED",
            "failure": {"type": type(error).__name__, "message": str(error)}, **timing}
        result["failed_paths"].append(entry); return entry
    try:
        for batch in (1, 8):
            check(); folder = output/f"B{batch}"; folder.mkdir(exist_ok=False)
            values, masks, chosen_origins = physical[:batch], observed[:batch], origins[:batch].copy()
            references, broken, samples = {}, {}, {route: [] for route in ROUTES}
            for route in ROUTES:
                check()
                if route in unavailable:
                    broken[route] = failed(route, batch, "PREEXISTING_UNAVAILABLE_ROUTE", ValueError(str(unavailable[route])))
                    continue
                try:
                    normalized, physical_value = whole_predict(predictors[route], values, masks, mean, std); synchronize()
                    source = saved_normalized_predictions[route][:batch]
                    scientific_physical = (source*std+mean).astype(np.float32)
                    checks = {"normalized": parity(normalized, source),
                        "physical": parity(physical_value, scientific_physical, PHYSICAL_ATOL, PHYSICAL_RTOL)}
                    artifact = save_outputs(folder/(route+"_REFERENCE.npz"), {"normalized": normalized, "physical": physical_value,
                        "scientific_normalized": source, "origins": chosen_origins})
                    if not all(item["pass"] for item in checks.values()): raise ValueError("SCIENTIFIC_REFERENCE_PARITY_FAILED:"+route)
                    references[route] = {"normalized": normalized.copy(), "physical": physical_value.copy(), "source": source,
                        "scientific_physical": scientific_physical, "reference_artifact": artifact}
                except Exception as error: broken[route] = failed(route, batch, "MATCHING_BATCH_REFERENCE", error)
            for order in schedule[:WARMUP]:
                for route in order:
                    check()
                    if route in broken: continue
                    try: whole_predict(predictors[route], values, masks, mean, std); synchronize()
                    except Exception as error: broken[route] = failed(route, batch, "WARMUP", error)
            for repeat, order in enumerate(schedule):
                for position, route in enumerate(order):
                    check()
                    if route in broken: continue
                    synchronize(); result["attempted_timer_calls"] += 1
                    unix_start_ns, perf_start_ns = time.time_ns(), time.perf_counter_ns()
                    try:
                        normalized, physical_value = whole_predict(predictors[route], values, masks, mean, std)
                        synchronize(); perf_end_ns, unix_end_ns = time.perf_counter_ns(), time.time_ns()
                    except Exception as error:
                        broken[route] = failed(route, batch, "TIMED_CALLBACK", error, repeat=repeat, position=position,
                            unix_start_ns=unix_start_ns, perf_start_ns=perf_start_ns,
                            perf_end_ns=time.perf_counter_ns(), unix_end_ns=time.time_ns())
                        write_json_once(folder/f"{route}_r{repeat}_FAILED.json", broken[route]); continue
                    seconds = (perf_end_ns-perf_start_ns)/1e9
                    memory = _jsonable(memory_snapshot()); reference = references[route]
                    checks = {"normalized": parity(normalized, reference["normalized"]),
                        "physical": parity(physical_value, reference["physical"], PHYSICAL_ATOL, PHYSICAL_RTOL),
                        "scientific_normalized": parity(normalized, reference["source"]),
                        "scientific_physical": parity(physical_value, reference["scientific_physical"], PHYSICAL_ATOL, PHYSICAL_RTOL)}
                    artifact = save_outputs(folder/f"{route}_r{repeat}.npz", {"normalized": normalized, "physical": physical_value, "origins": chosen_origins})
                    sample = {"dataset": dataset, "route": route, "batch": batch, "repeat": repeat, "block": repeat,
                        "position": position, "seconds": seconds, "perf_start_ns": perf_start_ns, "perf_end_ns": perf_end_ns,
                        "unix_start_ns": unix_start_ns, "unix_end_ns": unix_end_ns, "parity": checks, "memory": memory, "sample_artifact": artifact}
                    write_json_once(folder/f"{route}_r{repeat}.json", sample); result["recorded_timer_samples"] += 1
                    samples[route].append({**sample, "normalized": normalized.copy(), "physical": physical_value.copy()})
                    if not all(item["pass"] for item in checks.values()): broken[route] = failed(route, batch, "TIMED_OUTPUT_PARITY", ValueError(route), repeat=repeat)
                    check()
            raw = {sample["repeat"]: sample["seconds"] for sample in samples["RAW_BOLT"]}
            for route in ROUTES:
                records = samples[route]; seconds = [item["seconds"] for item in records]
                ratios = [item["seconds"]/raw[item["repeat"]] if raw.get(item["repeat"], 0)>0 else None for item in records]
                artifact = save_outputs(folder/(route+"_TIMED.npz"), {"normalized": np.stack([item["normalized"] for item in records]),
                    "physical": np.stack([item["physical"] for item in records]), "block": np.array([item["repeat"] for item in records], np.int64),
                    "position": np.array([item["position"] for item in records], np.int64), "origins": chosen_origins}) if records else None
                complete = len(records)==REPEATS and route not in broken
                result["rows"].append({"dataset": dataset, "route": route, "batch": batch, "status": "COMPLETE" if complete else "PARTIAL_OR_FAILED_PATH_PRESERVED",
                    "warmup_each_route": WARMUP, "repeats": len(records), "seconds": seconds,
                    "median_seconds": float(np.median(seconds)) if seconds else None,
                    "median_seconds_per_window": float(np.median(seconds))/batch if seconds else None,
                    "paired_RAW_seconds": [raw.get(item["repeat"]) for item in records], "paired_RAW_ratios": ratios,
                    "paired_RAW_median_ratio": float(np.median(ratios)) if ratios and all(value is not None for value in ratios) else None,
                    "positions": [item["position"] for item in records], "origins": chosen_origins.tolist(),
                    "parity": [item["parity"] for item in records], "memory": [item["memory"] for item in records],
                    "selected_metadata": metadata[route], "samples": [{key: value for key, value in item.items() if key not in ("normalized", "physical")} for item in records],
                    "timed_outputs": artifact, "reference": references.get(route, {}).get("reference_artifact"),
                    "failure": broken.get(route), "declared_route_deleted": False,
                    "whole_includes": "Physical causal CPU fill/FIT normalization, actual backend E/F or selected CPU2 head, frozen global D32/currentC anchor, materialized physical CPU return and final synchronization",
                    "whole_excludes": "Data/model/fixed E-D-head setup, FIT subset/global D fitting and native training, references/warmup/save, all source/file/state metadata hashes, guards/memory snapshots outside timer",
                    "memory_scope": "Shared process with all fixed E/D/native heads and frozen F resident; no isolated allocation or bandwidth claim",
                    "memory_counter_reset_inside_module": False, "per_origin_decoder_fit": False, "stage_decomposition": False,
                    "balanced_scope": result["schedule_scope"], "native_head_loaded_or_timed": "_NATIVE_s" in route})
                result["completed_cost_cells"] += int(complete)
        result["status"] = "COMPLETE" if result["completed_cost_cells"]==20 and result["recorded_timer_samples"]==200 and not result["failed_paths"] else "PARTIAL_COMPLETED"
    except Exception as error:
        result.update(status="FAILED_PRESERVED", failure={"type": type(error).__name__, "message": str(error)}); raise
    finally: write_json_once(output/"RESULT.json", result)
    return result
