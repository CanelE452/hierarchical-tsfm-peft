"""Diagnostic-only fixed-D innovation least-squares geometry; never a deployable oracle."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATASETS = {"robin": (17, 4, "val"), "beijing_aotizhongxin": (11, 2, "eval")}
ROUTES = ("PCA", "FORECA")
PINV_RCOND = 1e-15
LS_RTOL = 1e-10


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def score_complete(prediction, target, complete, std):
    predicted = np.asarray(prediction, dtype=np.float64)[complete]
    actual = np.asarray(target, dtype=np.float64)[complete]
    if not len(actual) or not np.isfinite(predicted).all() or not np.isfinite(actual).all():
        raise ValueError("FINITE_COMPLETE_TARGET_POSITIONS_REQUIRED")
    error = predicted - actual
    mse = np.mean(error * error, axis=0)
    mae = np.mean(np.abs(error), axis=0)
    scale = np.asarray(std, dtype=np.float64)
    if scale.shape != (error.shape[1],) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("FINITE_POSITIVE_FIT_STD_REQUIRED")
    common = {"observed_counts": [len(actual)] * error.shape[1], "active_channels": error.shape[1]}
    return {"normalized": {"mse": float(mse.mean()), "mae": float(mae.mean()),
                           "per_channel_mse": mse.tolist(), "per_channel_mae": mae.tolist(), **common},
            "physical_per_channel_diagnostic": {"mse": float((mse * scale * scale).mean()),
                "mae": float((mae * scale).mean()), "per_channel_mse": (mse * scale * scale).tolist(),
                "per_channel_mae": (mae * scale).tolist(), **common}}


def fixed_decoder_geometry(encoder, decoder):
    e = np.asarray(encoder, dtype=np.float32).astype(np.float64)
    d = np.asarray(decoder, dtype=np.float32).astype(np.float64)
    if (e.ndim != 2 or d.shape != (e.shape[1], e.shape[0])
            or not np.isfinite(e).all() or not np.isfinite(d).all()):
        raise ValueError("FINITE_CK_ENCODER_KC_DECODER_REQUIRED")
    singular = np.linalg.svd(d, compute_uv=False)
    cutoff = float(singular[0] * PINV_RCOND) if len(singular) else 0.
    rank = int(np.sum(singular > cutoff))
    inverse = np.linalg.pinv(d, rcond=PINV_RCOND)
    orthogonal = inverse @ d
    ed, de = e @ d, d @ e
    return {"decoder_deployment_float32_as_float64": d, "encoder_deployment_float32_as_float64": e,
            "decoder_pseudoinverse": inverse, "orthogonal_output_projector": orthogonal,
            "encoder_decoder_operator": ed, "decoder_encoder_operator": de}, {
        "channels": e.shape[0], "K": e.shape[1], "rank_D": rank,
        "singularvalues_D": singular.tolist(), "pinv_rcond": PINV_RCOND, "svd_rank_cutoff": cutoff,
        "rank_D_equals_K": rank == e.shape[1],
        "E_D_symmetry_max_abs": float(np.max(np.abs(ed - ed.T))),
        "E_D_idempotence_max_abs": float(np.max(np.abs(ed @ ed - ed))),
        "D_E_minus_I_max_abs": float(np.max(np.abs(de - np.eye(e.shape[1])))),
        "orthogonal_projector_symmetry_max_abs": float(np.max(np.abs(orthogonal - orthogonal.T))),
        "orthogonal_projector_idempotence_max_abs": float(np.max(np.abs(orthogonal @ orthogonal - orthogonal))),
        "geometry_dtype": "Fixed deployed E/D FP32 arrays interpreted in FP64 arithmetic",
        "numerical_rank_scope": "SVD pseudoinverse cutoff 1e-15; numerical retained output span, not an exact symbolic rank proof"}


def diagnose_span(target, observed, raw_last, encoder, decoder, predictions,
                  fixed_future_latents, fixed_last_latents, std):
    """Use actual complete future labels only to diagnose one fixed output subspace."""
    target, observed, last = np.asarray(target), np.asarray(observed), np.asarray(raw_last)
    if (target.ndim != 3 or observed.dtype != np.bool_ or observed.shape != target.shape
            or last.shape != (target.shape[0], target.shape[2]) or not np.isfinite(last).all()):
        raise ValueError("TARGET_OBSERVED_CURRENT_C_SHAPES_REQUIRED")
    complete = np.all(observed, axis=2)
    if not complete.any():
        raise ValueError("NO_ALL_C_OBSERVED_TARGET_POSITIONS")
    if not np.isfinite(target[complete]).all():
        raise ValueError("NONFINITE_COMPLETE_TARGET")
    arrays, geometry = fixed_decoder_geometry(encoder, decoder)
    d = arrays["decoder_deployment_float32_as_float64"]
    if d.shape[1] != target.shape[2]:
        raise ValueError("FIXED_D_TARGET_CHANNEL_MISMATCH")
    actual = target.astype(np.float64)
    last64 = last.astype(np.float64)
    repeated_last = np.broadcast_to(last64[:, None], target.shape)
    innovation = actual[complete] - repeated_last[complete]
    coordinates = innovation @ arrays["decoder_pseudoinverse"]
    projected = coordinates @ d
    residual = innovation - projected
    normal_residual = coordinates @ (d @ d.T) - innovation @ d.T
    orthogonal_residual = residual @ d.T
    normal_scale = 1 + np.linalg.norm(innovation) * np.linalg.norm(d)
    normal_tolerance = LS_RTOL * normal_scale
    if (np.max(np.abs(normal_residual)) > normal_tolerance
            or np.max(np.abs(orthogonal_residual)) > normal_tolerance):
        raise ValueError("FIXED_D_LEAST_SQUARES_NORMAL_EQUATION_FAILED")
    prediction = np.full(actual.shape, np.nan, dtype=np.float64)
    prediction[complete] = repeated_last[complete] + projected
    bound_scores = score_complete(prediction, actual, complete, std)
    best_latents = np.full((*target.shape[:2], d.shape[0]), np.nan, dtype=np.float64)
    best_latents[complete] = coordinates
    arrays.update(target=target.copy(), observed=observed.copy(), raw_last=last.copy(),
                  complete_target_mask=complete, least_squares_latent_innovation=best_latents,
                  fixed_D_bound_prediction=prediction, complete_innovation=innovation,
                  complete_projection_residual=residual, normal_equation_residual=normal_residual,
                  output_residual_orthogonality=orthogonal_residual)
    compared = {}
    for method, saved in predictions.items():
        saved = np.asarray(saved)
        if saved.shape != target.shape or not np.isfinite(saved[complete]).all():
            raise ValueError("FINITE_SAME_MASK_COMPARISON_PREDICTION_REQUIRED:" + method)
        entry = {"saved_prediction_scores_same_mask": score_complete(saved, actual, complete, std),
                 "uses_fixed_D_span": method in fixed_future_latents}
        arrays[method + "_saved_prediction"] = saved.copy()
        if method in fixed_future_latents:
            future = np.asarray(fixed_future_latents[method])
            previous = np.asarray(fixed_last_latents[method])
            if previous.ndim == 2:
                previous = previous[:, None]
            if future.shape != (*target.shape[:2], d.shape[0]) or previous.shape != (len(target), 1, d.shape[0]):
                raise ValueError("MATCHED_FIXED_D_LATENT_SHAPES_REQUIRED:" + method)
            delta = future.astype(np.float64) - previous.astype(np.float64)
            real_prediction = last64[:, None] + delta @ d
            if not np.isfinite(real_prediction[complete]).all():
                raise ValueError("NONFINITE_FIXED_D_REAL_ARITHMETIC_PREDICTION:" + method)
            real_scores = score_complete(real_prediction, actual, complete, std)
            real_mse = real_scores["normalized"]["mse"]
            actual_mse = entry["saved_prediction_scores_same_mask"]["normalized"]["mse"]
            tolerance = LS_RTOL * max(1., abs(real_mse), abs(bound_scores["normalized"]["mse"]))
            allowance = abs(actual_mse - real_mse)
            passed = bound_scores["normalized"]["mse"] <= real_mse + tolerance
            if not passed:
                raise ValueError("BOUND_ABOVE_SAVED_LATENT_FIXED_D_REAL_MSE:" + method)
            if bound_scores["normalized"]["mse"] > actual_mse + allowance + tolerance:
                raise ValueError("DECLARED_FP32_ROUNDING_ALLOWANCE_INCONSISTENT:" + method)
            entry.update(real_arithmetic_scores_same_mask=real_scores,
                         bound_le_real_arithmetic_mse=passed,
                         bound_le_saved_FP32_mse_plus_measured_allowance=True,
                         measured_saved_FP32_vs_real_MSE_allowance=allowance,
                         numerical_MSE_tolerance=tolerance,
                         saved_FP32_vs_real_prediction_max_abs=float(np.max(np.abs(saved[complete].astype(np.float64) - real_prediction[complete]))))
            arrays[method + "_latent_future"] = future.copy()
            arrays[method + "_latent_last"] = previous.copy()
            arrays[method + "_real_arithmetic_prediction"] = real_prediction
        compared[method] = entry
    return {"role": "DIAGNOSTIC_ONLY_FIXED_D_SUBSPACE", "deployment_used": False,
            "checkpoint_or_branch_selection_used": False, "actual_future_labels_used": True,
            "scope": "Numerical real-arithmetic least-squares minimum for this fixed linear deployed D and current-C anchor, matching all-C observed positions; excludes FP32 rounding, nonlinear D, and unrestricted predictors",
            "complete_target_positions": int(complete.sum()), "total_target_positions": int(complete.size),
            "fixed_D_bound_scores_same_mask": bound_scores, "geometry": geometry,
            "independent_linear_checks": {"normal_equation_max_abs": float(np.max(np.abs(normal_residual))),
                "residual_orthogonality_max_abs": float(np.max(np.abs(orthogonal_residual))),
                "declared_absolute_tolerance": normal_tolerance,
                "independent_lstsq_max_abs": float(np.max(np.abs(coordinates - np.linalg.lstsq(d.T, innovation.T, rcond=PINV_RCOND)[0].T))),
                "normal_equation_pass": True, "orthogonality_pass": True},
            "comparisons": compared,
            "RAW_bound_inequality_applied": False}, arrays


def checked_payload(record, meter, records):
    path = Path(record["path"])
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_relative_to(ROOT) or path.suffix.lower() != ".npz" or path.name.lower() in ("test.npz", "jena_test.npz"):
        raise ValueError("APPROVED_NON_TEST_LOCAL_NPZ_RECORD_REQUIRED")
    meter.check("hash-check diagnostic input " + path.name)
    if sha(path) != record["sha256"]:
        raise ValueError("EXACT_DIAGNOSTIC_INPUT_SHA_REQUIRED:" + str(path))
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    for name, value in arrays.items():
        if name not in record["arrays"]:
            raise ValueError("PAYLOAD_MEMBER_ABSENT_FROM_DECLARED_MANIFEST:" + name)
        expected = record["arrays"][name]
        if list(value.shape) != expected["shape"] or str(value.dtype) != expected["dtype"] or array_sha(value) != expected["sha256"]:
            raise ValueError("DIAGNOSTIC_INPUT_ARRAY_RECORD_MISMATCH:" + name)
    if set(arrays) != set(record["arrays"]):
        raise ValueError("EXACT_DECLARED_NPZ_MEMBERS_REQUIRED")
    records[str(path)] = {"path": str(path), "sha256": record["sha256"]}
    return arrays


def save_npz(path, arrays):
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    return {"path": str(Path(path).resolve()), "sha256": sha(path), "bytes": Path(path).stat().st_size,
            "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": array_sha(value)}
                       for name, value in arrays.items()}}


def verify_audit_binding(result_path, audit_path):
    result_path, audit_path = Path(result_path).resolve(), Path(audit_path).resolve()
    result, audit = read_json(result_path), read_json(audit_path)
    producer = audit.get("producer", {})
    if (result.get("status") != "COMPLETE" or audit.get("status") != "PASS"
            or Path(producer.get("path", "")).resolve() != result_path or producer.get("sha256") != sha(result_path)
            or result.get("protected_TEST_accessed") is not False or result.get("independent_holdout") is not False):
        raise ValueError("COMPLETE_PRODUCER_EXACT_SHA_BOUND_PASS_AUDIT_REQUIRED")
    if set(result["datasets"]) != set(DATASETS):
        raise ValueError("EXACT_APPROVED_TWO_DATASETS_REQUIRED")
    return result, {"path": str(result_path), "sha256": sha(result_path)}, {"path": str(audit_path), "sha256": sha(audit_path)}


def execute(result_path, audit_path, output_path):
    from continuation_runtime_v1 import Meter, write_new
    result, result_record, audit_record = verify_audit_binding(result_path, audit_path)
    output = Path(output_path).resolve()
    if not output.is_relative_to(HERE) or output.exists():
        raise ValueError("FRESH_DIAGNOSTIC_OUTPUT_WITHIN_FOLLOWUP_REQUIRED")
    output.mkdir(parents=True, exist_ok=False)
    meter = Meter("fixed_decoder_span_" + output.name, gpu=False)
    source = {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)}
    records = {}
    report = {"status": "RUNNING", "producer": result_record, "independent_projection_audit": audit_record,
              "source": source, "role": "DIAGNOSTIC_ONLY_FIXED_D_SUBSPACE", "datasets": {},
              "protected_TEST_accessed": False, "independent_holdout": False,
              "model_or_checkpoint_loaded": False, "new_fit_or_F_forward": False,
              "checkpoint_or_method_or_branch_selection_changed": False,
              "original_producer_selected_forecasts_changed": False}
    error = None
    try:
        meter.protected_check()
        old = ROOT / "research/smallk_failure_followup_20261014"
        if str(old) not in sys.path:
            sys.path.insert(0, str(old))
        from prepare_inputs import install_protected_test_guard
        report["protected_guard_paths"] = install_protected_test_guard()
        lock = result["selection_lock"]
        lock_path = Path(lock["path"]).resolve()
        if sha(lock_path) != lock["sha256"]:
            raise ValueError("EXACT_UNCHANGED_SELECTION_LOCK_REQUIRED")
        records[str(lock_path)] = lock
        for dataset, (channels, k, development) in DATASETS.items():
            row = result["datasets"][dataset]
            if row["K"] != k or set(row["inputs"]) != {"fit", "tune", development} or set(row["projections"]) != set(ROUTES):
                raise ValueError("APPROVED_DATASET_K_ROUTE_SPLITS_REQUIRED")
            inputs = {split: checked_payload(record, meter, records) for split, record in row["inputs"].items()}
            report["datasets"][dataset] = {"channels": channels, "K": k, "development_split": development, "projections": {}}
            for route in ROUTES:
                condition = row["projections"][route]
                state = checked_payload(condition["state_artifact"], meter, records)
                diagnostics = {}
                report["datasets"][dataset]["projections"][route] = diagnostics
                for split, values in inputs.items():
                    meter.check(dataset + " " + route + " " + split)
                    bolt = checked_payload(condition["bolt"]["artifacts"][split], meter, records)
                    oracle = checked_payload(condition["oracle"][split]["artifact"], meter, records)
                    expected_mask = np.broadcast_to(np.all(values["observed"], axis=2, keepdims=True), values["observed"].shape)
                    if not np.array_equal(oracle["evaluation_mask"], expected_mask):
                        raise ValueError("EXACT_ORACLE_COMPLETE_POSITION_MASK_REQUIRED")
                    predictions = {"RAW_BOLT": values["raw_forecast"], "BOLT": bolt["prediction"], "E_Y": oracle["prediction"]}
                    futures = {"BOLT": bolt["latent_future"], "E_Y": oracle["latent_future"]}
                    last = {"BOLT": bolt["latent_history"][:, -1], "E_Y": bolt["latent_history"][:, -1]}
                    if set(condition["native"]) != {"92601", "92602"}:
                        raise ValueError("EXACT_TWO_NATIVE_SEEDS_REQUIRED")
                    origin_records = [bolt["origins"], oracle["origins"]]
                    for seed, entry in condition["native"].items():
                        saved = checked_payload(entry["artifacts"][split], meter, records)
                        name = "NATIVE_s" + seed
                        predictions[name], futures[name], last[name] = saved["prediction"], saved["latent_future"], saved["latent_history"][:, -1]
                        origin_records.append(saved["origins"])
                    if not all(np.array_equal(origins, values["origins"]) for origins in origin_records):
                        raise ValueError("EXACT_SAME_DIAGNOSTIC_ORIGINS_REQUIRED")
                    summary, arrays = diagnose_span(values["target"], values["observed"], values["raw_last"],
                        state["encoder"], state["decoder"], predictions, futures, last, values["std"])
                    arrays["origins"] = values["origins"].copy()
                    arrays["FIT_std"] = values["std"].copy()
                    artifact = save_npz(output / f"{dataset}_{route}_{split}_SPAN.npz", arrays)
                    diagnostics[split] = {**summary, "artifact": artifact,
                                          "source_input": row["inputs"][split], "source_state": condition["state_artifact"]}
        for record in list(records.values()) + [result_record, audit_record, source]:
            if sha(record["path"]) != record["sha256"]:
                raise ValueError("DIAGNOSTIC_SOURCE_OR_SELECTION_CHANGED_DURING_EXECUTION")
        meter.protected_check()
        report.update(status="COMPLETE", source_records=list(records.values()),
                      source_before_after_SHA_pass=True, diagnostic_cells=12,
                      original_selected_predictions_and_locks_unchanged=True,
                      future_information_scope="Actual all-C observed Y used only in fixed-D mathematical diagnosis; no deployment, fitting or selection")
    except Exception as failure:
        error = {"type": type(failure).__name__, "message": str(failure)}
        report.update(status="FAILED_PRESERVED", failure=error, full_scope_complete_claimed=False)
    finally:
        report["resource_terminal"] = meter.close(report["status"], error)
        write_new(output / "RESULT.json", report)
    print(json.dumps({"status": report["status"], "output": str(output), "diagnostic_cells": report.get("diagnostic_cells"), "failure": error}), flush=True)
    return 0 if report["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    raise SystemExit(execute(arguments.result, arguments.audit, arguments.output))
