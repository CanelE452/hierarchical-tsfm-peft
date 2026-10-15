"""Independent decoder-only cached-forecast audit; no model or head is called."""
from pathlib import Path
import argparse
import json
import sys

import audit_projection_results_v1 as prior
import audit_fixed_decoder_span_v1 as span_audit

np = sys.modules.get("numpy")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATASETS = {"robin": (17, 4, "val"), "beijing_aotizhongxin": (11, 2, "eval")}
DECODERS = ("ORIGINAL_PAST_LS", "ENCODER_PINV")
DEPLOY_METHODS = ("BOLT", "NATIVE_s92601", "NATIVE_s92602")


def initialize_numpy():
    global np
    span_audit.initialize_numpy()
    np = sys.modules["numpy"]


def require(condition, label):
    prior.require(condition, label)


def close(actual, expected, label, *, atol=1e-10, rtol=1e-10):
    return prior.close(actual, expected, label, atol=atol, rtol=rtol)


def byte_exact(actual, expected, label):
    require(actual.dtype == expected.dtype and actual.shape == expected.shape
            and prior.array_sha(actual) == prior.array_sha(expected), "BYTE_EXACT_MISMATCH:" + label)


def independent_encoder_inverse(encoder):
    e = np.asarray(encoder, np.float32).astype(np.float64)
    require(e.ndim == 2 and e.shape[0] >= e.shape[1] > 0 and np.isfinite(e).all(), "FINITE_CK_DEPLOYED_E32")
    decoder, _, rank, singular = np.linalg.lstsq(e, np.eye(e.shape[0]), rcond=1e-15)
    projector = e @ decoder
    close(projector, projector.T, "encoder_inverse_real_projector_symmetry")
    close(projector @ projector, projector, "encoder_inverse_real_projector_idempotence")
    if rank == e.shape[1]:
        close(decoder @ e, np.eye(e.shape[1]), "encoder_inverse_real_left_identity")
    return {"E32_as64": e, "D_inverse64": decoder, "D_inverse32": np.ascontiguousarray(decoder, dtype=np.float32),
            "E_rank_lstsq": int(rank), "singularvalues_E_lstsq": singular, "real_projector": projector}


def geometry(encoder, decoder):
    e, d = (np.asarray(value, np.float32).astype(np.float64) for value in (encoder, decoder))
    require(e.ndim == 2 and d.shape == (e.shape[1], e.shape[0]), "MATCHED_CK_KC_GEOMETRY")
    ed, de = e @ d, d @ e
    return {"DE_minus_I_max_abs": float(np.abs(de - np.eye(e.shape[1])).max()),
            "ED_symmetry_max_abs": float(np.abs(ed - ed.T).max()),
            "ED_idempotence_max_abs": float(np.abs(ed @ ed - ed).max())}


def cached_forecast(raw_last, latent_future, latent_last, decoder, *, backend):
    require(raw_last.dtype == latent_future.dtype == latent_last.dtype == decoder.dtype == np.float32
            and raw_last.ndim == 2 and latent_future.ndim == 3
            and latent_last.shape in ((len(raw_last), latent_future.shape[-1]), (len(raw_last), 1, latent_future.shape[-1]))
            and len(raw_last) == len(latent_future) and decoder.shape == (latent_future.shape[-1], raw_last.shape[-1])
            and all(np.isfinite(value).all() for value in (raw_last, latent_future, latent_last, decoder)),
            "SAME_FINITE_FP32_CURRENT_C_FUTURE_K_LAST_K_FIXED_D")
    previous = latent_last[:, None] if latent_last.ndim == 2 else latent_last
    if backend == "numpy_full_N":
        return np.ascontiguousarray(raw_last[:, None] + (latent_future - previous) @ decoder)
    require(backend == "torch_CPU2_chunk32", "REGISTERED_DECODER_REPLAY_BACKEND")
    import torch
    torch.set_num_threads(2)
    weight = torch.from_numpy(np.ascontiguousarray(decoder))
    pieces = []
    with torch.no_grad():
        for start in range(0, len(raw_last), 32):
            last = torch.from_numpy(np.ascontiguousarray(raw_last[start:start + 32]))[:, None]
            future = torch.from_numpy(np.ascontiguousarray(latent_future[start:start + 32]))
            zlast = torch.from_numpy(np.ascontiguousarray(previous[start:start + 32]))
            pieces.append((last + (future - zlast) @ weight).numpy())
    return np.ascontiguousarray(np.concatenate(pieces))


def audit_cell(entry, arrays, values, state, cached):
    require(entry["status"] == "COMPLETE" and set(entry["decoders"]) == set(DECODERS), "COMPLETE_REGISTERED_TWO_DECODER_CELL")
    for key in ("target", "observed", "raw_last", "origins"):
        byte_exact(arrays[key], values[key], "same_original_input:" + key)
    byte_exact(arrays["FIT_mean"], values["mean"], "same_original_FIT_mean")
    byte_exact(arrays["FIT_std"], values["std"], "same_original_FIT_std")
    byte_exact(arrays["RAW_prediction"], values["raw_forecast"], "same_original_RAW_prediction")
    e32 = np.ascontiguousarray(state["encoder"], np.float32)
    byte_exact(arrays["encoder_deployed_FP32"], e32, "fixed_source_E32")
    inverse = independent_encoder_inverse(e32)
    require(inverse["E_rank_lstsq"] == e32.shape[1], "ENCODER_FULL_COLUMN_RANK_FOR_REGISTERED_ALTERNATIVE")
    byte_exact(arrays["ORIGINAL_PAST_LS_decoder_deployed_FP32"], np.ascontiguousarray(state["decoder"], np.float32), "original_D32")
    byte_exact(arrays["ENCODER_PINV_decoder_deployed_FP32"], inverse["D_inverse32"], "independent_lstsq_alternative_D32")
    construction = entry["alternative_decoder_construction"]
    require(construction["future_labels_used_to_construct_D"] is False
            and construction["hyperparameter_or_seed_selection"] is False and construction["pinv_rcond"] == 1e-15
            and construction["rank_E"] == inverse["E_rank_lstsq"], "LABEL_FREE_FIXED_E_INVERSE_CONSTRUCTION")
    close(construction["singularvalues_E"], inverse["singularvalues_E_lstsq"], "independent_encoder_lstsq_spectrum")
    import hashlib
    require(construction["decoder_FP32_sha256"] == hashlib.sha256(inverse["D_inverse32"].tobytes()).hexdigest(),
            "EXACT_NEW_D32_BYTE_DIGEST")
    require(isinstance(construction["decoder_FP64_before_cast_sha256"], str)
            and len(construction["decoder_FP64_before_cast_sha256"]) == 64, "DECLARED_UNSAVED_CONSTRUCTION_HASH_FORMAT")
    complete = values["observed"].all(axis=-1)
    prior.exact(arrays["complete_target_mask"], complete, "same_complete_C_mask")
    require(entry["complete_target_positions"] == int(complete.sum())
            and entry["total_target_positions"] == int(complete.size), "COMPLETE_C_MASK_POSITION_COUNTS")
    checks = prior.compare_scores(entry["RAW_scores_all_observed"], prior.scores(values["raw_forecast"], values["target"],
                                  values["observed"], values["std"]), "RAW_all_original_observations")
    checks += prior.compare_scores(entry["RAW_scores_complete_mask"], span_audit.complete_scores(values["raw_forecast"],
                                   values["target"], complete, values["std"]), "RAW_complete_C_only")
    methods = (*DEPLOY_METHODS, "E_Y")
    for method in methods:
        source = cached[method]
        byte_exact(arrays[method + "_cached_latent_future"], source["latent_future"], "unchanged_cached_future:" + method)
        byte_exact(arrays[method + "_cached_latent_last"], source["latent_last"], "unchanged_cached_last:" + method)
    results = {}
    for name in DECODERS:
        current = entry["decoders"][name]
        d32 = arrays[name + "_decoder_deployed_FP32"]
        require(current["native_heads_optimized_for_this_D"] == (name == "ORIGINAL_PAST_LS"),
                "REGISTERED_NATIVE_HEAD_TRAINING_DECODER_IDENTITY")
        expected_geometry, geometric_values = span_audit.independent_geometry(e32, d32)
        close(arrays[name + "_orthogonal_output_projector"], expected_geometry["orthogonal_output_projector"], "independent_D_rowspace_projector")
        for key, value in geometric_values.items():
            if isinstance(value, (bool, int)):
                require(current["geometry"][key] == value, "SAME_D_GEOMETRY_DISCRETE:" + key)
            else:
                close(current["geometry"][key], value, "same_D_geometry:" + key)
        require(set(current["forecasts"]) == set(methods), "EXACT_THREE_DEPLOYMENT_AND_ONE_DIAGNOSTIC_FORECAST")
        details = {}
        for method in methods:
            forecast = current["forecasts"][method]
            source = cached[method]
            registered_backend = "torch_CPU_FP32_threads2_chunk32" if method.startswith("NATIVE") else "numpy_FP32_full_N"
            backend = "torch_CPU2_chunk32" if method.startswith("NATIVE") else "numpy_full_N"
            require(forecast["backend"] == registered_backend and forecast["diagnostic_only"] == (method == "E_Y"),
                    "FROZEN_REGISTERED_BACKEND_AND_EY_DIAGNOSTIC_SCOPE")
            expected = cached_forecast(values["raw_last"], source["latent_future"], source["latent_last"], d32, backend=backend)
            prediction = arrays[name + "_" + method + "_prediction"]
            byte_exact(prediction, expected, "actual_CPU_D_only_replay:" + name + ":" + method)
            if name == "ORIGINAL_PAST_LS":
                byte_exact(prediction, source["prediction"], "ORIGINAL_v2_prediction:" + method)
                require(forecast["original_saved_prediction_byte_exact"] is True, "RECORDED_ORIGINAL_BYTE_PARITY")
            else:
                require(forecast["original_saved_prediction_byte_exact"] is None, "ALTERNATIVE_NOT_LABELLED_ORIGINAL_PARITY")
            previous = source["latent_last"][:, None] if source["latent_last"].ndim == 2 else source["latent_last"]
            zero_future = np.ascontiguousarray(np.broadcast_to(previous, source["latent_future"].shape))
            anchor = cached_forecast(values["raw_last"], zero_future, source["latent_last"], d32, backend=backend)
            prior.exact(anchor, np.broadcast_to(values["raw_last"][:, None], values["target"].shape), "zero_delta_original_current_C_anchor")
            require(forecast["zero_delta_current_C_anchor_elementwise_exact"] is True, "ZERO_DELTA_ANCHOR_INVARIANT")
            full_scores = prior.scores(prediction, values["target"], values["observed"], values["std"])
            same_mask = span_audit.complete_scores(prediction, values["target"], complete, values["std"])
            checks += prior.compare_scores(forecast["scores_all_observed"], full_scores, "all_original_observations:" + method)
            checks += prior.compare_scores(forecast["scores_complete_mask"], same_mask, "same_complete_C_mask:" + method)
            details[method] = {"normalized_MSE_all_observed": full_scores["normalized"]["mse"],
                               "normalized_MAE_all_observed": full_scores["normalized"]["mae"],
                               "normalized_MSE_complete_C": same_mask["normalized"]["mse"],
                               "diagnostic_only": method == "E_Y", "byte_exact_CPU_replay": True}
        fixed = current["fixed_D_real_arithmetic_diagnostic_same_complete_mask"]
        require(fixed["role"] == "DIAGNOSTIC_ONLY_FIXED_D_SUBSPACE", "NEW_D_FUTURE_SPAN_ONLY_DIAGNOSTIC")
        solution = span_audit.independent_span(values["target"], values["observed"], values["raw_last"], d32, values["std"])
        checks += prior.compare_scores(fixed["scores"], solution["scores"], "independent_fixed_D_real_minimum:" + name)
        certificate = fixed["independent_linear_checks"]
        close(certificate["declared_absolute_tolerance"], solution["normal_tolerance"], "registered_D_minimum_normal_tolerance")
        require(certificate["normal_equation_pass"] is True and certificate["orthogonality_pass"] is True
                and certificate["normal_equation_max_abs"] <= solution["normal_tolerance"]
                and certificate["residual_orthogonality_max_abs"] <= solution["normal_tolerance"]
                and certificate["independent_lstsq_max_abs"] <= 1e-10, "REGISTERED_NEW_D_MINIMUM_LINEAR_CERTIFICATE")
        bias = np.ascontiguousarray(state["decoder_bias"] if name == "ORIGINAL_PAST_LS" else state["center"], np.float32)
        reconstructed = cached["BOLT"]["latent_history"] @ d32 + bias
        past_scores = prior.scores(reconstructed, values["history"], values["history_observed"], values["std"])
        checks += prior.compare_scores(current["past_reconstruction_observed_only"], past_scores, "observed_original_past_reconstruction:" + name)
        require(current["past_reconstruction_bias_rule"] == ("original observed-past LS intercept" if name == "ORIGINAL_PAST_LS"
                else "original FIT encoder center; no fit"), "NO_NEW_FUTURE_OR_PAST_RECONSTRUCTION_BIAS_FIT")
        results[name] = {"forecasts": details, "normalized_MSE_fixed_D_real_minimum": solution["scores"]["normalized"]["mse"],
                         "heads_were_trained_under_this_D": name == "ORIGINAL_PAST_LS",
                         "head_global_optimality_claimed": False}
    require(entry["new_decoder_whole_deployment_cost_measured"] is False, "CPU_CACHED_D_PROBE_IS_NOT_WHOLE_DEPLOYMENT_COST")
    return {"complete_target_positions": int(complete.sum()), "score_field_checks": checks, "decoders": results,
            "D32_from_independent_lstsq_byte_exact": True, "E_Y_partial_missing_all_observed_score_is_not_deployment_or_primary_oracle": True,
            "changing_D_also_changes_output_span_scale_and_amplification_not_sole_obliqueness_proof": True,
            "new_whole_deployment_cost_in_this_probe_measured": False,
            "unsaved_FP64_constructor_hash_not_independently_rehashed": True}


def audit_diagnostic(diagnostic_path, result_path, projection_audit_path, span_path, span_audit_path, meter):
    check = meter.check
    records = {name: {"path": str(Path(path).resolve()), "sha256": prior.file_sha(path)}
               for name, path in (("diagnostic", diagnostic_path), ("producer", result_path),
                                  ("projection_audit", projection_audit_path), ("span_diagnostic", span_path),
                                  ("span_audit", span_audit_path))}
    data = {name: prior.verified_json(record, check) for name, record in records.items()}
    diagnostic, producer, projection, span, fixed_audit = (data[name] for name in records)
    if diagnostic["status"] != "COMPLETE":
        return {"status": "NOT_AUDITED_INCOMPLETE_PROBE", **records, "failed_paths": diagnostic.get("failed_paths"),
                "reported_completed_cells_not_independently_audited": diagnostic.get("completed_cells"),
                "registered_full_scope_PASS_claimed": False}
    require(producer["status"] == span["status"] == "COMPLETE" and projection["status"] == fixed_audit["status"] == "PASS",
            "COMPLETE_ORIGINAL_PRODUCER_SPAN_AND_INDEPENDENT_PASS_AUDITS")
    require(diagnostic["producer"] == projection["producer"] == span["producer"] == fixed_audit["producer"] == records["producer"]
            and diagnostic["projection_audit"] == span["independent_projection_audit"] == records["projection_audit"]
            and diagnostic["span_diagnostic"] == fixed_audit["diagnostic"] == records["span_diagnostic"]
            and diagnostic["span_audit"] == records["span_audit"], "EXACT_FIVE_PROBE_LINEAGE_BINDINGS")
    require(diagnostic["completed_cells"] == 12 and diagnostic["decoder_CPU_inference_forecasts"] == 96
            and diagnostic["failed_paths"] == [] and diagnostic["new_encoder_F_or_native_head_forward"] is False
            and diagnostic["new_training"] is False and diagnostic["future_labels_used_to_fit_alternative_D"] is False
            and diagnostic["method_checkpoint_branch_selection_used"] is False and diagnostic["protected_TEST_accessed"] is False
            and diagnostic["independent_holdout"] is False and diagnostic["new_decoder_whole_deployment_cost_measured"] is False
            and diagnostic["supervised_E_D_or_backbone_or_head_updates"] == 0
            and diagnostic["source_before_after_SHA_pass"] is True and diagnostic["original_selected_forecasts_and_locks_unchanged"] is True,
            "EXACT_LABEL_FREE_CACHED_CPU_D_ONLY_PROBE_SCOPE")
    require(set(diagnostic["datasets"]) == set(producer["datasets"]) == set(DATASETS), "EXACT_TWO_PROBE_DATASETS")
    source_records = diagnostic["source_records"] + [diagnostic["source"], diagnostic["checked_math_helper"]]
    for record in source_records:
        check()
        require(prior.file_sha(record["path"]) == record["sha256"], "PROBE_SOURCE_OLD_CACHE_OR_LOCK_FILE_CHANGED")
    require(any(Path(record["path"]).resolve() == Path(producer["selection_lock"]["path"]).resolve()
                and record["sha256"] == producer["selection_lock"]["sha256"] for record in source_records), "UNCHANGED_ORIGINAL_SELECTION_LOCK")
    seal = prior.verified_json(producer["selection_lock"], check)
    require(seal["status"] == "SEALED_BEFORE_NEW_DEV_AND_ORACLE", "ORIGINAL_PRE_DEVELOPMENT_NATIVE_SELECTION_SEAL")
    report = {"status": "PASS", **records, "cells": [], "source_file_checks": len(source_records),
              "totals": {"base_cells_checked": 0, "decoder_cells_checked": 0, "cached_forecast_arrays_checked": 0,
                         "cached_deployment_forecast_arrays_checked": 0, "cached_E_Y_diagnostic_arrays_checked": 0,
                         "original_predictions_byte_exact_checked": 0, "score_field_checks": 0},
              "original_encoder_F_native_head_forward_or_new_training_or_new_whole_timer_replay": False,
              "protected_TEST_accessed": False, "alternative_D_future_label_or_method_checkpoint_selection_used": False,
              "original_v2_current_anchor_E_D_F_stats_origins_cached_trajectories_preserved": True,
              "new_heads_trained_under_alternative_D_or_globally_optimal_claimed": False}
    for dataset, (_, k, development) in DATASETS.items():
        original, row = producer["datasets"][dataset], diagnostic["datasets"][dataset]
        require(row["K"] == k and row["development_split"] == development
                and set(row["projections"]) == {"PCA", "FORECA"}, "REGISTERED_PROBE_K_DEVELOPMENT_AND_ROUTES")
        raw = {split: prior.verified_arrays(record, check) for split, record in original["inputs"].items()}
        require(set(raw) == {"fit", "tune", development}, "REGISTERED_THREE_PROBE_SPLITS")
        for route in ("PCA", "FORECA"):
            condition = original["projections"][route]
            candidate = row["projections"][route]
            require(candidate["status"] == "COMPLETE" and candidate["source_state"] == condition["state_artifact"]
                    and set(candidate["splits"]) == set(raw), "COMPLETE_ROUTE_FIXED_E_AND_ALL_SPLITS")
            state = prior.verified_arrays(condition["state_artifact"], check)
            for split, values in raw.items():
                check()
                entry = candidate["splits"][split]
                require(entry["source_input"] == original["inputs"][split], "EXACT_OLD_PROBE_INPUT_RECORD")
                bolt = prior.verified_arrays(condition["bolt"]["artifacts"][split], check)
                ey = prior.verified_arrays(condition["oracle"][split]["artifact"], check)
                cached = {"BOLT": {**bolt, "latent_last": bolt["latent_history"][:, -1]},
                          "E_Y": {**ey, "latent_last": bolt["latent_history"][:, -1]}}
                source_bindings = {"BOLT": condition["bolt"]["artifacts"][split], "E_Y": condition["oracle"][split]["artifact"]}
                prior.exact(ey["evaluation_mask"], prior.oracle_mask(values["observed"]), "REGISTERED_EY_COMPLETE_C_MASK")
                for seed in ("92601", "92602"):
                    native = condition["native"][seed]
                    require(seal["datasets"][dataset][route][seed] == native["metadata"]["selected_update"], "SAME_OLD_SELECTED_HEAD")
                    saved = prior.verified_arrays(native["artifacts"][split], check)
                    name = "NATIVE_s" + seed
                    cached[name] = {**saved, "latent_last": saved["latent_history"][:, -1]}
                    source_bindings[name] = native["artifacts"][split]
                require(entry["cached_future_sources"] == source_bindings, "EXACT_AUDITED_CACHED_FUTURE_SOURCE_RECORDS")
                for method, saved in cached.items():
                    byte_exact(saved["origins"], values["origins"], "same_cached_origin:" + method)
                arrays = prior.verified_arrays(entry["artifact"], check)
                cell = audit_cell(entry, arrays, values, state, cached)
                report["cells"].append({"dataset": dataset, "route": route, "split": split, **cell})
                report["totals"]["base_cells_checked"] += 1
                report["totals"]["decoder_cells_checked"] += 2
                report["totals"]["cached_forecast_arrays_checked"] += 8
                report["totals"]["cached_deployment_forecast_arrays_checked"] += 6
                report["totals"]["cached_E_Y_diagnostic_arrays_checked"] += 2
                report["totals"]["original_predictions_byte_exact_checked"] += 4
                report["totals"]["score_field_checks"] += cell["score_field_checks"]
    require(report["totals"]["base_cells_checked"] == 12 and report["totals"]["decoder_cells_checked"] == 24
            and report["totals"]["cached_forecast_arrays_checked"] == 96
            and report["totals"]["original_predictions_byte_exact_checked"] == 48, "FULL_REGISTERED_D_PROBE_AUDIT_COUNTS")
    for record in list(records.values()) + source_records:
        require(prior.file_sha(record["path"]) == record["sha256"], "AUDITED_PROBE_ORIGINAL_SOURCES_CHANGED_DURING_CHECK")
    return report


def execute(diagnostic, result, projection_audit, span_diagnostic, span_audit_path, output):
    from continuation_runtime_v1 import Meter, write_new
    output_path = Path(output).resolve()
    require(not output_path.exists(), "FRESH_UNIQUE_DECODER_METRIC_AUDIT_OUTPUT")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meter = Meter("audit_decoder_metric_" + output_path.stem, gpu=False)
    source = {"path": str(Path(__file__).resolve()), "sha256": prior.file_sha(__file__)}
    report, failure = None, None
    try:
        meter.protected_check()
        initialize_numpy()
        sys.path.insert(0, str(ROOT / "research/smallk_failure_followup_20261014"))
        from prepare_inputs import install_protected_test_guard
        guarded_paths = install_protected_test_guard()
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2):
            report = audit_diagnostic(diagnostic, result, projection_audit, span_diagnostic, span_audit_path, meter)
        report["protected_guard_paths"] = guarded_paths
        meter.protected_check()
        require(prior.file_sha(__file__) == source["sha256"], "DECODER_METRIC_AUDITOR_SOURCE_CHANGED")
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        report = {"status": "FAILED_PRESERVED", "failure": failure, "full_scope_PASS_claimed": False}
    finally:
        report["audit_source"] = source
        report["independent_helper_sources"] = [{"path": str(Path(module.__file__).resolve()), "sha256": prior.file_sha(module.__file__)}
                                                for module in (prior, span_audit)]
        report["resource_terminal"] = meter.close(report["status"], failure)
        write_new(output_path, report)
    print(json.dumps({"status": report["status"], "output": str(output_path), "totals": report.get("totals"),
                      "failure": failure}, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--projection-audit", required=True)
    parser.add_argument("--span-diagnostic", required=True)
    parser.add_argument("--span-audit", dest="span_audit_path", required=True)
    parser.add_argument("--output", required=True)
    raise SystemExit(execute(**vars(parser.parse_args())))
