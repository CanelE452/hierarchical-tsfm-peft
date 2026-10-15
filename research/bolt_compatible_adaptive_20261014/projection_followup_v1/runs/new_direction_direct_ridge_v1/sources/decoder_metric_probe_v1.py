"""Label-free decoder intervention on frozen audited E/F/native cached latents."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

import fixed_decoder_span_v1 as span


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DECODERS = ("ORIGINAL_PAST_LS", "ENCODER_PINV")
METHODS = ("BOLT", "NATIVE_s92601", "NATIVE_s92602", "E_Y")
RCOND = 1e-15


def decoder_from_encoder(encoder):
    e = np.ascontiguousarray(encoder, dtype=np.float32).astype(np.float64)
    if e.ndim != 2 or not np.isfinite(e).all() or not e.shape[1]:
        raise ValueError("FINITE_CK_ENCODER_REQUIRED")
    singular = np.linalg.svd(e, compute_uv=False)
    cutoff = singular[0] * RCOND
    rank = int(np.sum(singular > cutoff))
    if rank != e.shape[1]:
        raise ValueError("FULL_COLUMN_RANK_ENCODER_REQUIRED")
    d64 = np.linalg.pinv(e, rcond=RCOND)
    d32 = np.ascontiguousarray(d64, dtype=np.float32)
    if not np.isfinite(d32).all():
        raise ValueError("FINITE_DEPLOYMENT_PINVERSE_REQUIRED")
    return d32, {"construction": "pinv(original deployed E32 interpreted as FP64,rcond=1e-15) cast to FP32",
                 "future_labels_used_to_construct_D": False, "hyperparameter_or_seed_selection": False,
                 "rank_E": rank, "singularvalues_E": singular.tolist(), "pinv_rcond": RCOND,
                 "decoder_FP64_before_cast_sha256": span.array_sha(d64),
                 "decoder_FP32_sha256": span.array_sha(d32)}


def cached_forecast(raw_last, future, latent_last, decoder, backend):
    raw, future, last, d = (np.ascontiguousarray(value, dtype=np.float32)
                            for value in (raw_last, future, latent_last, decoder))
    if raw.ndim == 2:
        raw = raw[:, None]
    if last.ndim == 2:
        last = last[:, None]
    if (future.ndim != 3 or last.shape != (len(future), 1, future.shape[2])
            or d.ndim != 2 or d.shape[0] != future.shape[2]
            or raw.shape != (len(future), 1, d.shape[1])
            or not all(np.isfinite(value).all() for value in (raw, future, last, d))):
        raise ValueError("EXACT_FINITE_CACHED_FORECAST_SHAPES_REQUIRED")
    if backend == "numpy_FP32_full_N":
        prediction = raw + (future - last) @ d
    elif backend == "torch_CPU_FP32_threads2_chunk32":
        import torch
        torch.set_num_threads(2)
        pieces = []
        with torch.no_grad():
            coefficient = torch.from_numpy(d)
            for start in range(0, len(future), 32):
                values = torch.from_numpy(future[start:start + 32])
                previous = torch.from_numpy(last[start:start + 32])
                anchor = torch.from_numpy(raw[start:start + 32])
                pieces.append((anchor + (values - previous) @ coefficient).numpy().copy())
        prediction = np.concatenate(pieces)
    else:
        raise ValueError("REGISTERED_CACHED_NUMPY_OR_NATIVE32_BACKEND_REQUIRED")
    if not np.isfinite(prediction).all():
        raise ValueError("NONFINITE_DECODER_INTERVENTION_OUTPUT")
    return np.ascontiguousarray(prediction, dtype=np.float32)


def masked_scores(prediction, target, observed, std):
    prediction, target, observed = np.asarray(prediction), np.asarray(target), np.asarray(observed)
    if prediction.shape != target.shape or observed.shape != target.shape or observed.dtype != np.bool_:
        raise ValueError("MATCHED_SCORE_SHAPE_BOOL_MASK_REQUIRED")
    counts, mse, mae = [], [], []
    for channel in range(target.shape[2]):
        valid = observed[:, :, channel]
        error = prediction[:, :, channel][valid].astype(np.float64) - target[:, :, channel][valid].astype(np.float64)
        if not np.isfinite(error).all():
            raise ValueError("FINITE_OBSERVED_SCORE_ERRORS_REQUIRED")
        counts.append(int(valid.sum()))
        mse.append(float(np.mean(error * error)) if len(error) else 0.)
        mae.append(float(np.mean(np.abs(error))) if len(error) else 0.)
    active = np.array(counts) > 0
    if not active.any():
        raise ValueError("SCORE_MASK_HAS_NO_OBSERVED_TARGETS")
    squared, absolute, scale = np.array(mse), np.array(mae), np.asarray(std, dtype=np.float64)
    common = {"observed_counts": counts, "active_channels": int(active.sum())}
    return {"normalized": {"mse": float(squared[active].mean()), "mae": float(absolute[active].mean()),
                           "per_channel_mse": mse, "per_channel_mae": mae, **common},
            "physical_per_channel_diagnostic": {"mse": float((squared[active] * scale[active] ** 2).mean()),
                "mae": float((absolute[active] * scale[active]).mean()),
                "per_channel_mse": (squared * scale ** 2).tolist(), "per_channel_mae": (absolute * scale).tolist(), **common}}


def gate(result_path, projection_audit_path, span_result_path, span_audit_path):
    producer, producer_record, projection_audit = span.verify_audit_binding(result_path, projection_audit_path)
    fixed_path, audit_path = Path(span_result_path).resolve(), Path(span_audit_path).resolve()
    fixed, audit = span.read_json(fixed_path), span.read_json(audit_path)
    if (fixed.get("status") != "COMPLETE" or audit.get("status") != "PASS"
            or audit.get("diagnostic") != {"path": str(fixed_path), "sha256": span.sha(fixed_path)}
            or audit.get("producer") != producer_record or fixed.get("producer") != producer_record
            or fixed.get("independent_projection_audit") != projection_audit):
        raise ValueError("EXACT_PRIOR_PROJECTION_AND_SPAN_PASS_BINDINGS_REQUIRED")
    return producer, {"producer": producer_record, "projection_audit": projection_audit,
                      "span_diagnostic": {"path": str(fixed_path), "sha256": span.sha(fixed_path)},
                      "span_audit": {"path": str(audit_path), "sha256": span.sha(audit_path)}}


def evaluate_case(values, state, cached):
    new_decoder, inverse_metadata = decoder_from_encoder(state["encoder"])
    decoders = {"ORIGINAL_PAST_LS": np.ascontiguousarray(state["decoder"], dtype=np.float32),
                "ENCODER_PINV": new_decoder}
    encoder = np.ascontiguousarray(state["encoder"], dtype=np.float32)
    complete = np.all(values["observed"], axis=2)
    complete_mask = np.broadcast_to(complete[:, :, None], values["observed"].shape).copy()
    if not complete.any():
        raise ValueError("SAME_COMPLETE_DIAGNOSTIC_MASK_EMPTY")
    raw_last = values["raw_last"]
    outputs = {"target": values["target"].copy(), "observed": values["observed"].copy(),
               "complete_target_mask": complete, "raw_last": raw_last.copy(), "origins": values["origins"].copy(),
               "FIT_mean": values["mean"].copy(), "FIT_std": values["std"].copy(),
               "encoder_deployed_FP32": encoder, "RAW_prediction": values["raw_forecast"].copy()}
    summary = {"status": "COMPLETE", "complete_target_positions": int(complete.sum()),
               "total_target_positions": int(complete.size), "alternative_decoder_construction": inverse_metadata,
               "RAW_scores_all_observed": masked_scores(values["raw_forecast"], values["target"], values["observed"], values["std"]),
               "RAW_scores_complete_mask": masked_scores(values["raw_forecast"], values["target"], complete_mask, values["std"]),
               "decoders": {}}
    futures, lasts = {}, {}
    for method, entry in cached.items():
        if not np.array_equal(entry["origins"], values["origins"]):
            raise ValueError("EXACT_CACHED_SOURCE_ORIGINS_REQUIRED:" + method)
        futures[method], lasts[method] = entry["latent_future"], entry["latent_last"]
        outputs[method + "_cached_latent_future"] = entry["latent_future"].copy()
        outputs[method + "_cached_latent_last"] = entry["latent_last"].copy()
    for name, d in decoders.items():
        geometry_arrays, geometry = span.fixed_decoder_geometry(encoder, d)
        forecasts = {}
        current = {"geometry": geometry, "forecasts": {}, "native_heads_optimized_for_this_D": name == "ORIGINAL_PAST_LS"}
        outputs[name + "_decoder_deployed_FP32"] = d.copy()
        outputs[name + "_orthogonal_output_projector"] = geometry_arrays["orthogonal_output_projector"]
        for method in METHODS:
            backend = "torch_CPU_FP32_threads2_chunk32" if method.startswith("NATIVE") else "numpy_FP32_full_N"
            entry = cached[method]
            prediction = cached_forecast(raw_last, entry["latent_future"], entry["latent_last"], d, backend)
            exact = prediction.dtype == entry["prediction"].dtype and prediction.shape == entry["prediction"].shape and prediction.tobytes() == np.ascontiguousarray(entry["prediction"]).tobytes()
            if name == "ORIGINAL_PAST_LS" and not exact:
                raise ValueError("ORIGINAL_REGISTERED_CACHE_BYTE_REPLAY_FAILED:" + method)
            constant = np.repeat(entry["latent_last"][:, None] if entry["latent_last"].ndim == 2 else entry["latent_last"], 48, axis=1)
            anchored = cached_forecast(raw_last, constant, entry["latent_last"], d, backend)
            if not np.array_equal(anchored, np.repeat(raw_last[:, None], 48, axis=1)):
                raise ValueError("ZERO_DELTA_CURRENT_C_ANCHOR_INVARIANT_FAILED")
            forecasts[method] = prediction
            outputs[name + "_" + method + "_prediction"] = prediction
            current["forecasts"][method] = {"backend": backend, "diagnostic_only": method == "E_Y",
                "original_saved_prediction_byte_exact": exact if name == "ORIGINAL_PAST_LS" else None,
                "zero_delta_current_C_anchor_elementwise_exact": True,
                "scores_all_observed": masked_scores(prediction, values["target"], values["observed"], values["std"]),
                "scores_complete_mask": masked_scores(prediction, values["target"], complete_mask, values["std"])}
        diagnostic, _ = span.diagnose_span(values["target"], values["observed"], raw_last, encoder, d,
                    {"RAW_BOLT": values["raw_forecast"], **forecasts}, futures, lasts, values["std"])
        current["fixed_D_real_arithmetic_diagnostic_same_complete_mask"] = {
            "scores": diagnostic["fixed_D_bound_scores_same_mask"],
            "independent_linear_checks": diagnostic["independent_linear_checks"],
            "role": diagnostic["role"], "scope": diagnostic["scope"]}
        past = cached["BOLT"]["latent_history"]
        bias = np.ascontiguousarray(state["decoder_bias"], dtype=np.float32) if name == "ORIGINAL_PAST_LS" else np.ascontiguousarray(state["center"], dtype=np.float32)
        reconstructed = np.ascontiguousarray(past @ d + bias, dtype=np.float32)
        current["past_reconstruction_observed_only"] = masked_scores(reconstructed, values["history"], values["history_observed"], values["std"])
        current["past_reconstruction_bias_rule"] = "original observed-past LS intercept" if name == "ORIGINAL_PAST_LS" else "original FIT encoder center; no fit"
        summary["decoders"][name] = current
    summary["intervention_interpretation"] = "D mapping alone changed on identical cached E/F/head trajectories; output span, scale and decoder metric may all change, so this does not isolate obliqueness as the sole mechanism. Native heads remain optimized for old D."
    summary["new_decoder_whole_deployment_cost_measured"] = False
    summary["new_decoder_deployment_cost_status"] = "PENDING_NOT_MEASURED: only actual cached-latent decoder CPU inference, no new E/F/head generation or complete deployment benchmark"
    return summary, outputs


def execute(result_path, projection_audit_path, span_result_path, span_audit_path, output_path):
    from continuation_runtime_v1 import Meter, write_new
    producer, bindings = gate(result_path, projection_audit_path, span_result_path, span_audit_path)
    output = Path(output_path).resolve()
    if not output.is_relative_to(HERE) or output.exists():
        raise ValueError("FRESH_DECODER_INTERVENTION_DIRECTORY_REQUIRED")
    output.mkdir(parents=True, exist_ok=False)
    meter = Meter("decoder_metric_" + output.name, gpu=False)
    source = {"path": str(Path(__file__).resolve()), "sha256": span.sha(__file__)}
    helper = {"path": str(HERE / "fixed_decoder_span_v1.py"), "sha256": span.sha(HERE / "fixed_decoder_span_v1.py")}
    records = {}
    report = {"status": "RUNNING", **bindings, "source": source, "checked_math_helper": helper,
              "datasets": {}, "failed_paths": [], "completed_cells": 0,
              "new_encoder_F_or_native_head_forward": False, "new_training": False,
              "future_labels_used_to_fit_alternative_D": False, "method_checkpoint_branch_selection_used": False,
              "protected_TEST_accessed": False, "independent_holdout": False,
              "new_decoder_whole_deployment_cost_measured": False}
    error = None
    try:
        meter.protected_check()
        old = ROOT / "research/smallk_failure_followup_20261014"
        if str(old) not in sys.path:
            sys.path.insert(0, str(old))
        from prepare_inputs import install_protected_test_guard
        report["protected_guard_paths"] = install_protected_test_guard()
        lock = producer["selection_lock"]
        if span.sha(lock["path"]) != lock["sha256"]:
            raise ValueError("UNCHANGED_SEALED_SELECTION_LOCK_REQUIRED")
        records[lock["path"]] = lock
        for dataset, (_, k, development) in span.DATASETS.items():
            data = producer["datasets"][dataset]
            if data["K"] != k or set(data["inputs"]) != {"fit", "tune", development}:
                raise ValueError("APPROVED_DATASET_K_SPLIT_CONTRACT_REQUIRED")
            report["datasets"][dataset] = {"K": k, "development_split": development, "projections": {}}
            for route in span.ROUTES:
                condition = data["projections"][route]
                try:
                    state = span.checked_payload(condition["state_artifact"], meter, records)
                    decoder_from_encoder(state["encoder"])
                except ValueError as failure:
                    row = {"dataset": dataset, "route": route, "stage": "decoder_rank", "status": "FAILED_PATH_PRESERVED", "error": str(failure)}
                    report["failed_paths"].append(row)
                    report["datasets"][dataset]["projections"][route] = {"status": "FAILED_PATH_PRESERVED", "failure": row}
                    continue
                route_report = {"status": "COMPLETE", "source_state": condition["state_artifact"], "splits": {}}
                report["datasets"][dataset]["projections"][route] = route_report
                for split, record in data["inputs"].items():
                    meter.check(dataset + " " + route + " " + split + " cached decoder intervention")
                    try:
                        values = span.checked_payload(record, meter, records)
                        bolt = span.checked_payload(condition["bolt"]["artifacts"][split], meter, records)
                        oracle = span.checked_payload(condition["oracle"][split]["artifact"], meter, records)
                        complete = np.broadcast_to(np.all(values["observed"], axis=2, keepdims=True), values["observed"].shape)
                        if not np.array_equal(oracle["evaluation_mask"], complete):
                            raise ValueError("EXACT_ORIGINAL_ORACLE_COMPLETE_MASK_REQUIRED")
                        cached = {"BOLT": {**bolt, "latent_last": bolt["latent_history"][:, -1]},
                                  "E_Y": {**oracle, "latent_last": bolt["latent_history"][:, -1]}}
                        sources = {"BOLT": condition["bolt"]["artifacts"][split], "E_Y": condition["oracle"][split]["artifact"]}
                        for seed in ("92601", "92602"):
                            native_record = condition["native"][seed]["artifacts"][split]
                            saved = span.checked_payload(native_record, meter, records)
                            cached["NATIVE_s" + seed] = {**saved, "latent_last": saved["latent_history"][:, -1]}
                            sources["NATIVE_s" + seed] = native_record
                        summary, arrays = evaluate_case(values, state, cached)
                        summary["source_input"] = record
                        summary["cached_future_sources"] = sources
                        summary["artifact"] = span.save_npz(output / f"{dataset}_{route}_{split}_DECODER_METRIC.npz", arrays)
                        route_report["splits"][split] = summary
                        report["completed_cells"] += 1
                    except ValueError as failure:
                        failed = {"dataset": dataset, "route": route, "split": split,
                                  "status": "FAILED_PATH_PRESERVED", "error": str(failure)}
                        route_report["splits"][split] = failed
                        route_report["status"] = "PARTIAL_FAILED_PATHS"
                        report["failed_paths"].append(failed)
        for record in list(records.values()) + list(bindings.values()) + [source, helper]:
            if span.sha(record["path"]) != record["sha256"]:
                raise ValueError("ORIGINAL_CACHE_SOURCE_OR_LOCK_CHANGED_DURING_DECODER_INTERVENTION")
        meter.protected_check()
        report.update(status="COMPLETE" if report["completed_cells"] == 12 and not report["failed_paths"] else "PARTIAL_FAILED_PATHS",
                      source_records=list(records.values()), source_before_after_SHA_pass=True,
                      original_selected_forecasts_and_locks_unchanged=True,
                      decoder_CPU_inference_forecasts=report["completed_cells"] * 8,
                      supervised_E_D_or_backbone_or_head_updates=0,
                      comparison_scope="Same source E,F,cached future/last, K,stats,origins,current-C anchor and old selected heads; label-free deterministic D intervention. Whole new-D deployment costs not measured.")
    except Exception as failure:
        error = {"type": type(failure).__name__, "message": str(failure)}
        report.update(status="FAILED_PRESERVED", failure=error, full_scope_complete_claimed=False)
    finally:
        report["resource_terminal"] = meter.close(report["status"], error)
        write_new(output / "RESULT.json", report)
    print(json.dumps({"status": report["status"], "output": str(output), "completed_cells": report["completed_cells"], "failed_paths": len(report["failed_paths"]), "failure": error}), flush=True)
    return 0 if report["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--span-result", required=True)
    parser.add_argument("--span-audit", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    raise SystemExit(execute(arguments.result, arguments.audit, arguments.span_result, arguments.span_audit, arguments.output))
