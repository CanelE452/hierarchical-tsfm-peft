"""Future-informed rank-K innovation geometry, diagnostic only and never a model."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np

import fixed_decoder_span_v1 as span


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATASETS = span.DATASETS
RTOL = 1e-10


def rank_oracle(target, observed, raw_last, k, std):
    target, observed, raw_last = np.asarray(target), np.asarray(observed), np.asarray(raw_last)
    if (target.ndim != 3 or observed.dtype != np.bool_ or observed.shape != target.shape
            or raw_last.shape != (len(target), target.shape[2]) or not np.isfinite(raw_last).all()
            or isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer))
            or not 1 <= k <= target.shape[2]):
        raise ValueError("EXACT_COMPLETE_MASK_CURRENT_C_AND_INTEGER_K_REQUIRED")
    complete = observed.all(axis=2)
    if not complete.any() or not np.isfinite(target[complete]).all():
        raise ValueError("FINITE_COMPLETE_FUTURE_TARGET_POSITIONS_REQUIRED")
    anchor = np.broadcast_to(raw_last.astype(np.float64)[:, None], target.shape)
    innovation = target.astype(np.float64)[complete] - anchor[complete]
    left, singular, right = np.linalg.svd(innovation, full_matrices=False)
    active_k = min(int(k), len(singular))
    reconstructed = (left[:, :active_k] * singular[:active_k]) @ right[:active_k]
    residual = innovation - reconstructed
    squared_singular = singular * singular
    total_energy = float(np.sum(innovation * innovation))
    retained = float(squared_singular[:active_k].sum())
    discarded = float(squared_singular[active_k:].sum())
    residual_energy = float(np.sum(residual * residual))
    gram = innovation.T @ innovation
    gram_eigenvalues = np.linalg.eigvalsh(gram)[::-1]
    gram_negative_tolerance = 100 * np.finfo(np.float64).eps * max(1., total_energy)
    if np.min(gram_eigenvalues) < -gram_negative_tolerance:
        raise ValueError("INNOVATION_GRAM_NOT_NUMERICALLY_PSD")
    clipped_eigenvalues = np.maximum(gram_eigenvalues, 0.)
    gram_tail = float(clipped_eigenvalues[int(k):].sum())
    tolerance = RTOL * max(1., total_energy)
    if (abs(total_energy - float(squared_singular.sum())) > tolerance
            or abs(total_energy - retained - discarded) > tolerance
            or abs(discarded - residual_energy) > tolerance
            or abs(discarded - gram_tail) > tolerance
            or abs(total_energy - float(np.trace(gram))) > tolerance):
        raise ValueError("SVD_GRAM_DISCARDED_ENERGY_CERTIFICATE_FAILED")
    projector = right[:active_k].T @ right[:active_k]
    perpendicular = residual @ right[:active_k].T
    if np.max(np.abs(perpendicular)) > tolerance:
        raise ValueError("RANK_K_RESIDUAL_NOT_ORTHOGONAL_TO_RETAINED_DIRECTIONS")
    full_prediction = np.full(target.shape, np.nan, np.float64)
    full_prediction[complete] = anchor[complete] + reconstructed
    scores = span.score_complete(full_prediction, target, complete, std)
    denominator = int(len(innovation) * target.shape[2])
    oracle_mse = discarded / denominator
    if abs(scores["normalized"]["mse"] - oracle_mse) > tolerance / denominator:
        raise ValueError("RECONSTRUCTED_SCORE_DISAGREES_WITH_TAIL_ENERGY_MINIMUM")
    arrays = {"target": target.copy(), "observed": observed.copy(), "raw_last": raw_last.copy(),
              "complete_target_mask": complete, "complete_innovation": innovation,
              "rank_K_innovation_reconstruction": reconstructed, "rank_K_prediction": full_prediction,
              "rank_K_projection_residual": residual, "singularvalues": singular,
              "squared_singularvalues": squared_singular, "retained_right_singular_vectors": right[:active_k],
              "retained_left_singular_vectors": left[:, :active_k], "innovation_gram": gram,
              "gram_diagonal_energy": np.diag(gram).copy(), "gram_eigenvalues": gram_eigenvalues,
              "orthogonal_rank_K_projector": projector, "retained_residual_orthogonality": perpendicular}
    summary = {"role": "STRONGLY_FUTURE_INFORMED_RANK_K_DIAGNOSTIC_ONLY", "K": int(k),
               "channels": target.shape[2], "complete_target_positions": len(innovation),
               "total_target_positions": int(complete.size), "future_matrix_shape": list(innovation.shape),
               "directions_fitted_to_known_future_of_this_split": True,
               "one_shared_output_space_across_splits_assumed": False,
               "centered_or_future_intercept_used": False,
               "deployment_used": False, "method_checkpoint_branch_selection_used": False,
               "independent_holdout": False, "new_deployable_training": False,
               "minimum_normalized_MSE_from_discarded_energy": oracle_mse,
               "rank_K_oracle_scores_same_mask": scores,
               "numerical_rank_future_matrix": int(np.sum(singular > singular[0] * max(innovation.shape) * np.finfo(np.float64).eps)) if singular[0] > 0 else 0,
               "energy_certificate": {"total_innovation_energy": total_energy,
                    "retained_energy": retained, "discarded_energy": discarded,
                    "reconstruction_residual_energy": residual_energy, "SVD_squared_singularvalues_sum": float(squared_singular.sum()),
                    "Gram_trace": float(np.trace(gram)), "Gram_tail_eigenvalue_sum": gram_tail,
                    "negative_Gram_eigenvalue_tolerance": gram_negative_tolerance,
                    "absolute_energy_tolerance": tolerance,
                    "normalized_MSE_denominator_M_times_C": denominator,
                    "residual_orthogonality_max_abs": float(np.max(np.abs(perpendicular))),
                    "checks_pass": True},
               "scope": "Numerical real-arithmetic minimum over arbitrary global rank-at-most-K linear innovation output spaces for this already-known all-C observed future matrix and fixed current-C anchor. This oracle learns directions separately from each split's future labels; it is unavailable at deployment, excludes FP32 rounding and nonlinear/readout alternatives, and does not prove past input-information sufficiency or generalization. MAE/physical scores are values of the normalized-MSE minimizer, not separate minima."}
    return summary, arrays


def compare_fixed_span(oracle, fixed_diagnostics):
    minimum = oracle["minimum_normalized_MSE_from_discarded_energy"]
    comparisons = {}
    for route, fixed in fixed_diagnostics.items():
        if fixed["complete_target_positions"] != oracle["complete_target_positions"] or fixed["geometry"]["K"] != oracle["K"]:
            raise ValueError("MATCHED_COMPLETE_MASK_AND_K_REQUIRED")
        floor = fixed["fixed_D_bound_scores_same_mask"]["normalized"]["mse"]
        tolerance = RTOL * max(1., abs(floor), abs(minimum))
        if minimum > floor + tolerance:
            raise ValueError("RANK_K_ORACLE_ABOVE_AUDITED_FIXED_D_FLOOR")
        real_methods = {}
        for method, row in fixed["comparisons"].items():
            if not row["uses_fixed_D_span"]:
                continue
            mse = row["real_arithmetic_scores_same_mask"]["normalized"]["mse"]
            if minimum > mse + RTOL * max(1., abs(mse), abs(minimum)):
                raise ValueError("RANK_K_ORACLE_ABOVE_SAVED_FIXED_D_REAL_PREDICTION")
            real_methods[method] = {"real_arithmetic_MSE": mse, "rank_K_oracle_le_real_arithmetic_MSE": True}
        raw = fixed["comparisons"]["RAW_BOLT"]["saved_prediction_scores_same_mask"]["normalized"]["mse"]
        comparisons[route] = {"fixed_D_floor_MSE_same_mask": floor,
                              "fixed_D_floor_minus_rank_K_oracle": floor - minimum,
                              "RAW_MSE_same_mask": raw, "RAW_minus_rank_K_oracle": raw - minimum,
                              "numerical_comparison_tolerance": tolerance,
                              "rank_K_oracle_le_fixed_D_floor": True, "saved_fixed_D_real_predictions": real_methods}
    return comparisons


def gate_bindings(result_path, audit_path, span_result_path, span_audit_path):
    result, producer_record, projection_audit_record = span.verify_audit_binding(result_path, audit_path)
    span_result_path, span_audit_path = Path(span_result_path).resolve(), Path(span_audit_path).resolve()
    fixed, audit = span.read_json(span_result_path), span.read_json(span_audit_path)
    bound = audit.get("diagnostic", {})
    if (fixed.get("status") != "COMPLETE" or audit.get("status") != "PASS"
            or Path(bound.get("path", "")).resolve() != span_result_path or bound.get("sha256") != span.sha(span_result_path)
            or fixed.get("producer") != producer_record or audit.get("producer") != producer_record
            or fixed.get("independent_projection_audit") != projection_audit_record
            or fixed.get("diagnostic_cells") != 12 or fixed.get("protected_TEST_accessed") is not False
            or fixed.get("checkpoint_or_method_or_branch_selection_changed") is not False):
        raise ValueError("EXACT_SHA_BOUND_PASS_FIXED_SPAN_AUDIT_REQUIRED")
    return result, fixed, {"producer": producer_record, "projection_audit": projection_audit_record,
                           "fixed_span": {"path": str(span_result_path), "sha256": span.sha(span_result_path)},
                           "fixed_span_audit": {"path": str(span_audit_path), "sha256": span.sha(span_audit_path)}}


def execute(result_path, audit_path, span_result_path, span_audit_path, output_path):
    from continuation_runtime_v1 import Meter, write_new
    producer, fixed, bindings = gate_bindings(result_path, audit_path, span_result_path, span_audit_path)
    output = Path(output_path).resolve()
    if not output.is_relative_to(HERE) or output.exists():
        raise ValueError("NEW_EXCLUSIVE_RANK_ORACLE_OUTPUT_WITHIN_FOLLOWUP_REQUIRED")
    output.mkdir(parents=True, exist_ok=False)
    meter = Meter("oracle_rank_" + output.name, gpu=False)
    source = {"path": str(Path(__file__).resolve()), "sha256": span.sha(__file__)}
    helper = {"path": str(HERE / "fixed_decoder_span_v1.py"), "sha256": span.sha(HERE / "fixed_decoder_span_v1.py")}
    records = {}
    report = {"status": "RUNNING", **bindings, "source": source, "checked_input_helper": helper, "datasets": {},
              "role": "STRONGLY_FUTURE_INFORMED_RANK_K_DIAGNOSTIC_ONLY",
              "protected_TEST_accessed": False, "independent_holdout": False,
              "model_or_checkpoint_loaded": False, "new_backbone_forward_or_training": False,
              "deployment_or_method_checkpoint_branch_selection_used": False,
              "oracle_target_matrix_factorizations": 0}
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
            raise ValueError("ORIGINAL_SELECTION_LOCK_CHANGED")
        records[lock["path"]] = lock
        for dataset, (channels, k, development) in DATASETS.items():
            row = producer["datasets"][dataset]
            if row["K"] != k or set(row["inputs"]) != {"fit", "tune", development}:
                raise ValueError("EXACT_ORIGINAL_INPUT_SCOPE_REQUIRED")
            report["datasets"][dataset] = {"channels": channels, "K": k, "development_split": development, "splits": {}}
            for split, input_record in row["inputs"].items():
                meter.check(dataset + " " + split + " known-future rank oracle")
                values = span.checked_payload(input_record, meter, records)
                summary, arrays = rank_oracle(values["target"], values["observed"], values["raw_last"], k, values["std"])
                fixed_rows = {route: fixed["datasets"][dataset]["projections"][route][split] for route in span.ROUTES}
                for entry in fixed_rows.values():
                    if entry["source_input"] != input_record:
                        raise ValueError("RANK_ORACLE_AND_AUDITED_FIXED_D_INPUT_RECORD_MISMATCH")
                summary["fixed_D_and_RAW_comparisons_same_mask"] = compare_fixed_span(summary, fixed_rows)
                arrays["origins"] = values["origins"].copy()
                arrays["FIT_std"] = values["std"].copy()
                arrays["RAW_prediction"] = values["raw_forecast"].copy()
                summary["RAW_scores_same_mask"] = span.score_complete(values["raw_forecast"], values["target"], arrays["complete_target_mask"], values["std"])
                for entry in fixed_rows.values():
                    raw_mse = entry["comparisons"]["RAW_BOLT"]["saved_prediction_scores_same_mask"]["normalized"]["mse"]
                    if not np.isclose(raw_mse, summary["RAW_scores_same_mask"]["normalized"]["mse"], atol=1e-12, rtol=1e-12):
                        raise ValueError("EXACT_RECOMPUTED_RAW_SAME_MASK_SCORE_MISMATCH")
                summary["artifact"] = span.save_npz(output / f"{dataset}_{split}_RANK_ORACLE.npz", arrays)
                summary["source_input"] = input_record
                report["datasets"][dataset]["splits"][split] = summary
                report["oracle_target_matrix_factorizations"] += 1
        for record in list(records.values()) + list(bindings.values()) + [source, helper]:
            if span.sha(record["path"]) != record["sha256"]:
                raise ValueError("RANK_DIAGNOSTIC_ORIGINAL_SOURCE_OR_LOCK_CHANGED")
        meter.protected_check()
        report.update(status="COMPLETE", source_before_after_SHA_pass=True, source_records=list(records.values()),
                      original_selected_forecasts_and_selection_lock_unchanged=True,
                      oracle_scope="Each split's already-known future matrix determines its own best global rank-K output directions; strong unavailable-at-deployment mathematical oracle, never a newly evaluated predictive method")
    except Exception as failure:
        error = {"type": type(failure).__name__, "message": str(failure)}
        report.update(status="FAILED_PRESERVED", failure=error, full_scope_complete_claimed=False)
    finally:
        report["resource_terminal"] = meter.close(report["status"], error)
        write_new(output / "RESULT.json", report)
    print(json.dumps({"status": report["status"], "output": str(output), "oracle_target_matrix_factorizations": report["oracle_target_matrix_factorizations"], "failure": error}), flush=True)
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
