"""Independent saved-array fixed-D span audit, using direct least squares.

This imports no diagnostic producer, encoder, decoder trainer, forecaster or
model loader. The only minimum checked is normalized MSE in the numerical
real-arithmetic output span of the fixed deployed linear D on complete-C
target positions. Physical MSE and MAE describe that solution only.
"""
from pathlib import Path
import argparse
import json
import sys

import audit_projection_results_v1 as prior

np = sys.modules.get("numpy")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATASETS = {"robin": (17, 4, "val"), "beijing_aotizhongxin": (11, 2, "eval")}
METHODS = ("RAW_BOLT", "BOLT", "E_Y", "NATIVE_s92601", "NATIVE_s92602")
FIXED_METHODS = METHODS[1:]
RTOL = 1e-10


def initialize_numpy():
    global np
    prior.initialize_numpy()
    np = sys.modules["numpy"]


def require(condition, label):
    prior.require(condition, label)


def close(actual, expected, label, *, atol=1e-10, rtol=1e-10):
    return prior.close(actual, expected, label, atol=atol, rtol=rtol)


def complete_scores(prediction, target, complete, std):
    require(complete.dtype == np.bool_ and complete.shape == target.shape[:2] and complete.any(),
            "NONEMPTY_BOOL_COMPLETE_TARGET_POSITIONS")
    mask = np.repeat(complete[:, :, None], target.shape[-1], axis=-1)
    return prior.scores(prediction, target, mask, std)


def independent_geometry(encoder, decoder):
    e, d = (np.asarray(value, np.float32).astype(np.float64) for value in (encoder, decoder))
    require(e.ndim == 2 and d.shape == (e.shape[1], e.shape[0])
            and np.isfinite(e).all() and np.isfinite(d).all(), "FINITE_DEPLOYED_E32_D32_GEOMETRY")
    # Direct least squares to each C coordinate, independently of producer pinv.
    inverse = np.linalg.lstsq(d.T, np.eye(d.shape[1]), rcond=1e-15)[0].T
    projector = inverse @ d
    singular = np.linalg.svd(d, compute_uv=False)
    cutoff = float(singular[0] * 1e-15)
    rank = int((singular > cutoff).sum())
    ed, de = e @ d, d @ e
    arrays = {"decoder_deployment_float32_as_float64": d, "encoder_deployment_float32_as_float64": e,
              "decoder_pseudoinverse": inverse, "orthogonal_output_projector": projector,
              "encoder_decoder_operator": ed, "decoder_encoder_operator": de}
    geometry = {"channels": e.shape[0], "K": e.shape[1], "rank_D": rank, "singularvalues_D": singular.tolist(),
                "pinv_rcond": 1e-15, "svd_rank_cutoff": cutoff, "rank_D_equals_K": rank == e.shape[1],
                "E_D_symmetry_max_abs": float(np.abs(ed - ed.T).max()),
                "E_D_idempotence_max_abs": float(np.abs(ed @ ed - ed).max()),
                "D_E_minus_I_max_abs": float(np.abs(de - np.eye(e.shape[1])).max()),
                "orthogonal_projector_symmetry_max_abs": float(np.abs(projector - projector.T).max()),
                "orthogonal_projector_idempotence_max_abs": float(np.abs(projector @ projector - projector).max())}
    close(projector, projector.T, "orthogonal_projector_symmetry")
    close(projector @ projector, projector, "orthogonal_projector_idempotence")
    return arrays, geometry


def independent_span(target, observed, raw_last, decoder, std):
    require(target.ndim == 3 and observed.shape == target.shape and observed.dtype == np.bool_
            and raw_last.shape == (len(target), target.shape[-1]), "SAME_TARGET_MASK_CURRENT_C_ANCHOR")
    complete = observed.all(axis=-1)
    require(complete.any() and np.isfinite(target[complete]).all() and np.isfinite(raw_last).all(),
            "FINITE_NONEMPTY_COMPLETE_TARGET_AND_CURRENT_C")
    d = np.asarray(decoder, np.float32).astype(np.float64)
    require(d.ndim == 2 and d.shape[1] == target.shape[-1] and np.isfinite(d).all(), "FIXED_D32_C_TARGET_DIMENSION")
    last = np.broadcast_to(raw_last.astype(np.float64)[:, None], target.shape)
    innovation = target.astype(np.float64)[complete] - last[complete]
    coefficients, _, numerical_rank, _ = np.linalg.lstsq(d.T, innovation.T, rcond=1e-15)
    coordinates = coefficients.T
    projected = coordinates @ d
    residual = innovation - projected
    normal = coordinates @ (d @ d.T) - innovation @ d.T
    orthogonality = residual @ d.T
    tolerance = 1e-10 * (1 + np.linalg.norm(innovation) * np.linalg.norm(d))
    require(np.abs(normal).max() <= tolerance and np.abs(orthogonality).max() <= tolerance,
            "DIRECT_LSTSQ_NORMAL_EQUATION_OR_ORTHOGONALITY")
    prediction = np.full(target.shape, np.nan, np.float64)
    prediction[complete] = last[complete] + projected
    latent = np.full((*target.shape[:2], d.shape[0]), np.nan, np.float64)
    latent[complete] = coordinates
    return {"complete": complete, "innovation": innovation, "coordinates": coordinates,
            "latent_innovation": latent, "prediction": prediction, "residual": residual,
            "normal": normal, "orthogonality": orthogonality, "normal_tolerance": float(tolerance),
            "rank_lstsq": int(numerical_rank), "scores": complete_scores(prediction, target, complete, std)}


def real_trace_and_rounding(saved_prediction, latent_future, latent_last, raw_last, decoder, target, complete, std):
    previous = np.asarray(latent_last)
    if previous.ndim == 2:
        previous = previous[:, None]
    d = np.asarray(decoder, np.float32).astype(np.float64)
    require(latent_future.shape == (*target.shape[:2], d.shape[0])
            and previous.shape == (len(target), 1, d.shape[0])
            and saved_prediction.shape == target.shape, "MATCHED_FUTURE_LAST_K_AND_CURRENT_C_TRACE")
    real = raw_last.astype(np.float64)[:, None] + (latent_future.astype(np.float64) - previous.astype(np.float64)) @ d
    actual_scores = complete_scores(saved_prediction, target, complete, std)
    real_scores = complete_scores(real, target, complete, std)
    actual_mse, real_mse = (value["normalized"]["mse"] for value in (actual_scores, real_scores))
    return {"prediction": real, "saved_scores": actual_scores, "real_scores": real_scores,
            "measured_MSE_allowance": abs(actual_mse - real_mse),
            "prediction_max_abs": float(np.abs(saved_prediction[complete].astype(np.float64) - real[complete]).max())}


def audit_cell(entry, arrays, values, state, predictions, futures, last_latents):
    require(entry["role"] == "DIAGNOSTIC_ONLY_FIXED_D_SUBSPACE" and entry["deployment_used"] is False
            and entry["checkpoint_or_branch_selection_used"] is False and entry["actual_future_labels_used"] is True
            and entry["RAW_bound_inequality_applied"] is False, "DIAGNOSTIC_ROLE_FUTURE_ONLY_NO_SELECTION_RAW_BOUND")
    for key in ("target", "observed", "raw_last", "origins"):
        prior.exact(arrays[key], values[key], "span_exact_original_input:" + key)
    prior.exact(arrays["FIT_std"], values["std"], "span_same_original_FIT_std")
    expected, geometry = independent_geometry(state["encoder"], state["decoder"])
    for key, expected_array in expected.items():
        close(arrays[key], expected_array, "fixed_D_geometric_state:" + key)
    for key, expected_value in geometry.items():
        if isinstance(expected_value, (bool, int)):
            require(entry["geometry"][key] == expected_value, "DECLARED_D_GEOMETRY_DISCRETE:" + key)
        else:
            close(entry["geometry"][key], expected_value, "declared_D_geometry:" + key)
    solution = independent_span(values["target"], values["observed"], values["raw_last"], state["decoder"], values["std"])
    complete = solution["complete"]
    prior.exact(arrays["complete_target_mask"], complete, "all_C_observed_complete_mask")
    require(entry["complete_target_positions"] == int(complete.sum())
            and entry["total_target_positions"] == int(complete.size), "SAME_COMPLETE_TARGET_POSITION_COUNTS")
    require(np.isnan(arrays["least_squares_latent_innovation"][~complete]).all()
            and np.isnan(arrays["fixed_D_bound_prediction"][~complete]).all(), "UNOBSERVED_COMPLETE_POSITION_NOT_SCORED_OR_IMPUTED")
    require(arrays["least_squares_latent_innovation"].dtype == arrays["fixed_D_bound_prediction"].dtype == np.float64,
            "REAL_ARITHMETIC_FP64_MINIMUM_SAVED_DTYPE")
    comparisons = ((arrays["least_squares_latent_innovation"][complete], solution["coordinates"], "optimal_latent_innovation"),
                   (arrays["fixed_D_bound_prediction"][complete], solution["prediction"][complete], "fixed_D_optimal_prediction"),
                   (arrays["complete_innovation"], solution["innovation"], "target_minus_same_current_C"),
                   (arrays["complete_projection_residual"], solution["residual"], "orthogonal_output_residual"),
                   (arrays["normal_equation_residual"], solution["normal"], "normal_equation_residual"),
                   (arrays["output_residual_orthogonality"], solution["orthogonality"], "output_residual_orthogonality"))
    errors = {label: close(actual, wanted, label) for actual, wanted, label in comparisons}
    checks = prior.compare_scores(entry["fixed_D_bound_scores_same_mask"], solution["scores"], "fixed_D_bound_solution")
    internal = entry["independent_linear_checks"]
    close(internal["declared_absolute_tolerance"], solution["normal_tolerance"], "normal_equation_registered_tolerance")
    require(internal["normal_equation_pass"] is True and internal["orthogonality_pass"] is True,
            "PRODUCER_INTERNAL_LINEAR_CHECKS_RECOMPUTED")
    close(internal["normal_equation_max_abs"], float(np.abs(arrays["normal_equation_residual"]).max()),
          "recorded_normal_equation_maximum")
    close(internal["residual_orthogonality_max_abs"], float(np.abs(arrays["output_residual_orthogonality"]).max()),
          "recorded_output_orthogonality_maximum")
    require(internal["independent_lstsq_max_abs"] <= 1e-10, "PRODUCER_LSTSQ_INTERNAL_COORDINATE_CHECK")
    bound = solution["scores"]["normalized"]["mse"]
    details = {}
    require(set(entry["comparisons"]) == set(METHODS), "EXACT_FIVE_SAME_MASK_METHOD_COMPARISONS")
    for method in METHODS:
        compared = entry["comparisons"][method]
        prior.exact(arrays[method + "_saved_prediction"], predictions[method], "exact_audited_original_prediction:" + method)
        checks += prior.compare_scores(compared["saved_prediction_scores_same_mask"],
                                       complete_scores(predictions[method], values["target"], complete, values["std"]),
                                       "same_complete_mask:" + method)
        require(compared["uses_fixed_D_span"] == (method in FIXED_METHODS), "FIXED_D_BOUND_METHOD_SCOPE:" + method)
        if method == "RAW_BOLT":
            require("bound_le_real_arithmetic_mse" not in compared, "RAW_IS_OUTSIDE_FIXED_D_BOUND")
            details[method] = {"normalized_mse_same_complete_mask": compared["saved_prediction_scores_same_mask"]["normalized"]["mse"],
                               "fixed_D_bound_applied": False}
            continue
        prior.exact(arrays[method + "_latent_future"], futures[method], "original_audited_latent_future:" + method)
        normalized_last = last_latents[method]
        if normalized_last.ndim == 2:
            normalized_last = normalized_last[:, None]
        prior.exact(arrays[method + "_latent_last"], normalized_last, "original_audited_latent_last:" + method)
        trace = real_trace_and_rounding(predictions[method], futures[method], normalized_last,
                                       values["raw_last"], state["decoder"], values["target"], complete, values["std"])
        close(arrays[method + "_real_arithmetic_prediction"], trace["prediction"], "fixed_D_real_trace:" + method)
        checks += prior.compare_scores(compared["real_arithmetic_scores_same_mask"], trace["real_scores"], "fixed_D_real_trace_score:" + method)
        real_mse = trace["real_scores"]["normalized"]["mse"]
        tolerance = 1e-10 * max(1., abs(real_mse), abs(bound))
        require(bound <= real_mse + tolerance, "INDEPENDENT_FIXED_D_MINIMUM_ABOVE_REAL_TRACE:" + method)
        saved_mse = trace["saved_scores"]["normalized"]["mse"]
        require(bound <= saved_mse + trace["measured_MSE_allowance"] + tolerance,
                "FIXED_D_SAVED_FP32_BOUND_WITH_MEASURED_ROUNDING:" + method)
        close(compared["measured_saved_FP32_vs_real_MSE_allowance"], trace["measured_MSE_allowance"], "measured_FP32_rounding_MSE:" + method)
        close(compared["numerical_MSE_tolerance"], tolerance, "numerical_MSE_tolerance:" + method)
        close(compared["saved_FP32_vs_real_prediction_max_abs"], trace["prediction_max_abs"], "measured_FP32_rounding_prediction:" + method)
        require(compared["bound_le_real_arithmetic_mse"] is True
                and compared["bound_le_saved_FP32_mse_plus_measured_allowance"] is True,
                "RECORDED_FIXED_D_BOUND_COMPARISON")
        details[method] = {"real_arithmetic_mse": real_mse, "saved_FP32_mse": saved_mse,
                           "gap_above_fixed_D_minimum_real": real_mse - bound,
                           "measured_FP32_rounding_MSE_allowance": trace["measured_MSE_allowance"]}
    return {"complete_target_positions": int(complete.sum()), "channels": values["target"].shape[-1],
            "direct_lstsq_rank_D": solution["rank_lstsq"], "normalized_MSE_fixed_D_real_minimum": bound,
            "score_field_checks": checks, "saved_array_max_abs_differences": errors, "comparisons": details,
            "normalized_MSE_minimum_scope": "this fixed deployed linear D numerical output span; complete-C positions; real arithmetic",
            "physical_MSE_or_MAE_minimum_claimed": False, "unrestricted_predictor_or_other_decoder_minimum_claimed": False,
            "actual_F_model_training_or_timer_independently_replayed": False}


def audit_diagnostic(diagnostic_path, result_path, projection_audit_path, meter):
    check = meter.check
    records = {name: {"path": str(Path(path).resolve()), "sha256": prior.file_sha(path)}
               for name, path in (("diagnostic", diagnostic_path), ("producer", result_path),
                                  ("projection_audit", projection_audit_path))}
    diagnostic, result, projection_audit = (prior.verified_json(records[name], check)
                                           for name in ("diagnostic", "producer", "projection_audit"))
    require(diagnostic["status"] == result["status"] == "COMPLETE" and projection_audit["status"] == "PASS",
            "COMPLETED_DIAGNOSTIC_AND_PRODUCER_WITH_PASSED_PROJECTION_AUDIT")
    require(diagnostic["producer"] == records["producer"]
            and diagnostic["independent_projection_audit"] == records["projection_audit"]
            and projection_audit["producer"] == records["producer"], "EXACT_DIAGNOSTIC_PRODUCER_PROJECTION_AUDIT_BINDING")
    require(diagnostic["role"] == "DIAGNOSTIC_ONLY_FIXED_D_SUBSPACE"
            and diagnostic["protected_TEST_accessed"] is False and diagnostic["independent_holdout"] is False
            and diagnostic["model_or_checkpoint_loaded"] is False and diagnostic["new_fit_or_F_forward"] is False
            and diagnostic["checkpoint_or_method_or_branch_selection_changed"] is False
            and diagnostic["original_producer_selected_forecasts_changed"] is False
            and diagnostic["source_before_after_SHA_pass"] is True
            and diagnostic["original_selected_predictions_and_locks_unchanged"] is True
            and diagnostic["diagnostic_cells"] == 12, "EXACT_DIAGNOSTIC_SCOPE_NO_DEPLOYMENT_OR_SELECTION")
    require(set(diagnostic["datasets"]) == set(result["datasets"]) == set(DATASETS), "EXACT_TWO_DIAGNOSTIC_DATASETS")
    source_records = diagnostic["source_records"] + [diagnostic["source"]]
    for record in source_records:
        check()
        require(prior.file_sha(record["path"]) == record["sha256"], "DIAGNOSTIC_SOURCE_OR_SELECTION_FILE_CHANGED")
    require(any(Path(record["path"]).resolve() == Path(result["selection_lock"]["path"]).resolve()
                and record["sha256"] == result["selection_lock"]["sha256"] for record in source_records),
            "ORIGINAL_SELECTION_LOCK_EXACTLY_PRESERVED")
    seal = prior.verified_json(result["selection_lock"], check)
    require(seal["status"] == "SEALED_BEFORE_NEW_DEV_AND_ORACLE", "ORIGINAL_NATIVE_SELECTION_LOCK_STATE")
    report = {"status": "PASS", **records, "cells": [],
              "totals": {"diagnostic_cells_checked": 0, "fixed_D_method_inequalities_checked": 0,
                         "score_field_checks": 0, "bound_RAW_inequalities_checked": 0},
              "source_file_checks": len(source_records), "protected_TEST_accessed": False,
              "new_fit_F_forward_model_loading_training_or_timer_replay": False,
              "selection_changed": False, "future_labels_used_only_for_registered_diagnostic": True,
              "minimum_claim_only_normalized_MSE_fixed_linear_D_real_span_on_complete_C_positions": True,
              "physical_MSE_or_MAE_minimum_claimed": False, "other_decoder_or_unrestricted_predictor_bound_claimed": False}
    for dataset, (channels, k, development) in DATASETS.items():
        original, row = result["datasets"][dataset], diagnostic["datasets"][dataset]
        require(row["channels"] == channels and row["K"] == k and row["development_split"] == development
                and set(row["projections"]) == {"PCA", "FORECA"}, "REGISTERED_DIAGNOSTIC_DATASET_K_ROUTES")
        raw = {split: prior.verified_arrays(record, check) for split, record in original["inputs"].items()}
        require(set(raw) == {"fit", "tune", development}, "REGISTERED_DIAGNOSTIC_SPLITS")
        for route in ("PCA", "FORECA"):
            condition = original["projections"][route]
            state = prior.verified_arrays(condition["state_artifact"], check)
            require(set(row["projections"][route]) == set(raw), "REGISTERED_THREE_FIXED_D_CELLS")
            for split, values in raw.items():
                check()
                entry = row["projections"][route][split]
                require(entry["source_input"] == original["inputs"][split]
                        and entry["source_state"] == condition["state_artifact"], "EXACT_AUDITED_FIXED_D_INPUT_STATE_RECORD")
                arrays = prior.verified_arrays(entry["artifact"], check)
                bolt = prior.verified_arrays(condition["bolt"]["artifacts"][split], check)
                ey = prior.verified_arrays(condition["oracle"][split]["artifact"], check)
                predictions = {"RAW_BOLT": values["raw_forecast"], "BOLT": bolt["prediction"], "E_Y": ey["prediction"]}
                futures = {"BOLT": bolt["latent_future"], "E_Y": ey["latent_future"]}
                latents = {"BOLT": bolt["latent_history"][:, -1], "E_Y": bolt["latent_history"][:, -1]}
                for artifact in (bolt, ey):
                    prior.exact(artifact["origins"], values["origins"], "original_projection_oracle_origin")
                prior.exact(ey["evaluation_mask"], prior.oracle_mask(values["observed"]), "same_complete_C_oracle_mask")
                for seed in ("92601", "92602"):
                    native = condition["native"][seed]
                    require(seal["datasets"][dataset][route][seed] == native["metadata"]["selected_update"],
                            "REGISTERED_FROZEN_NATIVE_SELECTED_CHECKPOINT")
                    saved = prior.verified_arrays(native["artifacts"][split], check)
                    prior.exact(saved["origins"], values["origins"], "original_native_origin")
                    name = "NATIVE_s" + seed
                    predictions[name], futures[name], latents[name] = (saved["prediction"], saved["latent_future"],
                                                                     saved["latent_history"][:, -1])
                cell = audit_cell(entry, arrays, values, state, predictions, futures, latents)
                report["cells"].append({"dataset": dataset, "route": route, "split": split, **cell})
                report["totals"]["diagnostic_cells_checked"] += 1
                report["totals"]["fixed_D_method_inequalities_checked"] += 4
                report["totals"]["score_field_checks"] += cell["score_field_checks"]
    require(report["totals"]["diagnostic_cells_checked"] == 12
            and report["totals"]["fixed_D_method_inequalities_checked"] == 48
            and report["totals"]["bound_RAW_inequalities_checked"] == 0, "FULL_REGISTERED_FIXED_D_AUDIT_COUNTS")
    for record in list(records.values()) + source_records:
        require(prior.file_sha(record["path"]) == record["sha256"], "AUDITED_DIAGNOSTIC_SOURCE_CHANGED_DURING_EXECUTION")
    return report


def execute(diagnostic, result, projection_audit, output):
    from continuation_runtime_v1 import Meter, write_new
    output_path = Path(output).resolve()
    require(not output_path.exists(), "FRESH_INDEPENDENT_SPAN_AUDIT_OUTPUT")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meter = Meter("audit_fixed_D_span_" + output_path.stem, gpu=False)
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
            report = audit_diagnostic(diagnostic, result, projection_audit, meter)
        report["protected_guard_paths"] = guarded_paths
        meter.protected_check()
        require(prior.file_sha(__file__) == source["sha256"], "FIXED_D_AUDITOR_SOURCE_CHANGED")
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        report = {"status": "FAILED_PRESERVED", "failure": failure, "full_scope_PASS_claimed": False}
    finally:
        report["audit_source"] = source
        report["independent_metric_IO_source"] = {"path": str(Path(prior.__file__).resolve()), "sha256": prior.file_sha(prior.__file__)}
        report["resource_terminal"] = meter.close(report["status"], failure)
        write_new(output_path, report)
    print(json.dumps({"status": report["status"], "output": str(output_path), "totals": report.get("totals"),
                      "failure": failure}, ensure_ascii=False), flush=True)
    return 0 if failure is None else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--projection-audit", required=True)
    parser.add_argument("--output", required=True)
    raise SystemExit(execute(**vars(parser.parse_args())))
