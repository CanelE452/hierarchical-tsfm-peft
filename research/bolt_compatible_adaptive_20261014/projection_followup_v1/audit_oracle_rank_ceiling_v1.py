"""Independent Gram-eigenvalue audit of the strongly future-informed rank-K oracle.

No SVD or diagnostic producer helper is called. Known future labels determine
each split's mathematical oracle directions. This is neither a deployable
model nor evidence of past input-information sufficiency or generalization.
"""
from pathlib import Path
import argparse
import json
import sys

import audit_projection_results_v1 as prior
import audit_fixed_decoder_span_v1 as fixed_audit

np = sys.modules.get("numpy")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATASETS = {"robin": (17, 4, "val"), "beijing_aotizhongxin": (11, 2, "eval")}


def initialize_numpy():
    global np
    fixed_audit.initialize_numpy()
    np = sys.modules["numpy"]


def require(condition, label):
    prior.require(condition, label)


def close(actual, expected, label, *, atol=1e-10, rtol=1e-10):
    return prior.close(actual, expected, label, atol=atol, rtol=rtol)


def gram_oracle(target, observed, raw_last, k, std):
    require(target.ndim == 3 and observed.dtype == np.bool_ and observed.shape == target.shape
            and raw_last.shape == (len(target), target.shape[-1]) and np.isfinite(raw_last).all()
            and isinstance(k, (int, np.integer)) and not isinstance(k, (bool, np.bool_))
            and 1 <= k <= target.shape[-1], "SAME_CURRENT_C_COMPLETE_MASK_INTEGER_K")
    complete = observed.all(axis=-1)
    require(complete.any() and np.isfinite(target[complete]).all(), "FINITE_NONEMPTY_COMPLETE_FUTURE_MATRIX")
    last = np.broadcast_to(raw_last.astype(np.float64)[:, None], target.shape)
    innovation = target.astype(np.float64)[complete] - last[complete]
    gram = innovation.T @ innovation
    eigenvalues, vectors = np.linalg.eigh(gram)
    eigenvalues, vectors = eigenvalues[::-1], vectors[:, ::-1]
    total = float(np.sum(innovation * innovation))
    negative_tolerance = 100 * np.finfo(np.float64).eps * max(1., total)
    require(float(eigenvalues.min()) >= -negative_tolerance, "GRAM_NUMERICALLY_PSD")
    nonnegative = np.maximum(eigenvalues, 0.)
    active_k = min(k, min(innovation.shape))
    selected = vectors[:, :active_k]
    reconstruction = (innovation @ selected) @ selected.T
    residual = innovation - reconstruction
    discarded, retained = float(nonnegative[k:].sum()), float(nonnegative[:k].sum())
    tolerance = 1e-10 * max(1., total)
    require(abs(total - float(np.trace(gram))) <= tolerance
            and abs(total - retained - discarded) <= tolerance
            and abs(discarded - float(np.sum(residual * residual))) <= tolerance,
            "INDEPENDENT_GRAM_TRACE_TAIL_AND_RECONSTRUCTION_ENERGY")
    gap = float(eigenvalues[active_k - 1] - eigenvalues[active_k]) if active_k < len(eigenvalues) else float("inf")
    unique_boundary = active_k == len(eigenvalues) or gap > tolerance
    full_prediction = np.full(target.shape, np.nan, np.float64)
    full_prediction[complete] = last[complete] + reconstruction
    return {"complete": complete, "innovation": innovation, "gram": gram, "eigenvalues": eigenvalues,
            "nonnegative_eigenvalues": nonnegative, "active_k": active_k, "projector": selected @ selected.T,
            "reconstruction": reconstruction, "residual": residual, "prediction": full_prediction,
            "total_energy": total, "discarded_energy": discarded, "retained_energy": retained,
            "energy_tolerance": tolerance, "negative_tolerance": negative_tolerance,
            "normalized_mse_minimum": discarded / innovation.size,
            "boundary_spectral_gap": gap, "unique_retained_boundary": unique_boundary,
            "scores": fixed_audit.complete_scores(full_prediction, target, complete, std)}


def audit_cell(entry, arrays, values, k, fixed_rows):
    require(entry["role"] == "STRONGLY_FUTURE_INFORMED_RANK_K_DIAGNOSTIC_ONLY" and entry["K"] == k
            and entry["directions_fitted_to_known_future_of_this_split"] is True
            and entry["one_shared_output_space_across_splits_assumed"] is False
            and entry["centered_or_future_intercept_used"] is False
            and entry["deployment_used"] is False and entry["method_checkpoint_branch_selection_used"] is False
            and entry["independent_holdout"] is False and entry["new_deployable_training"] is False,
            "STRONGLY_FUTURE_INFORMED_SPLIT_SPECIFIC_UNCENTERED_DIAGNOSTIC_SCOPE")
    for key in ("target", "observed", "raw_last", "origins"):
        prior.exact(arrays[key], values[key], "exact_registered_rank_oracle_input:" + key)
    prior.exact(arrays["FIT_std"], values["std"], "same_FIT_std")
    prior.exact(arrays["RAW_prediction"], values["raw_forecast"], "same_audited_RAW_prediction")
    solution = gram_oracle(values["target"], values["observed"], values["raw_last"], k, values["std"])
    complete, innovation = solution["complete"], solution["innovation"]
    prior.exact(arrays["complete_target_mask"], complete, "rank_oracle_complete_C_positions")
    close(arrays["complete_innovation"], innovation, "uncentered_Y64_minus_current_C64")
    close(arrays["innovation_gram"], solution["gram"], "independent_innovation_Gram")
    close(arrays["gram_diagonal_energy"], np.diag(solution["gram"]), "Gram_diagonal_channel_energy")
    energy_tolerance = solution["energy_tolerance"]
    close(arrays["gram_eigenvalues"], solution["eigenvalues"], "independent_Gram_eigenvalues",
          atol=energy_tolerance, rtol=1e-10)
    singular, squared = arrays["singularvalues"], arrays["squared_singularvalues"]
    require(singular.ndim == 1 and len(singular) == min(innovation.shape) and np.isfinite(singular).all()
            and np.all(singular >= 0) and np.all(np.diff(singular) <= 0), "DECLARED_ORDERED_NONNEGATIVE_SINGULAR_VALUES")
    close(squared, singular * singular, "squared_singular_values", atol=energy_tolerance, rtol=1e-10)
    close(squared, solution["nonnegative_eigenvalues"][:len(singular)], "SVD_energy_vs_independent_Gram_energy",
          atol=energy_tolerance, rtol=1e-10)
    right, left = arrays["retained_right_singular_vectors"], arrays["retained_left_singular_vectors"]
    active_k = solution["active_k"]
    require(right.shape == (active_k, innovation.shape[1]) and left.shape == (len(innovation), active_k),
            "RETAINED_ORACLE_FACTOR_DIMENSIONS")
    close(right @ right.T, np.eye(active_k), "retained_right_orthonormal")
    close(left.T @ left, np.eye(active_k), "retained_left_orthonormal")
    projector = right.T @ right
    close(arrays["orthogonal_rank_K_projector"], projector, "saved_retained_direction_projector")
    close(projector, projector.T, "rank_K_projector_symmetry")
    close(projector @ projector, projector, "rank_K_projector_idempotence")
    reconstruction = arrays["rank_K_innovation_reconstruction"]
    close(reconstruction, innovation @ projector, "reconstruction_from_saved_rank_K_output_space")
    close(reconstruction, (left * singular[:active_k]) @ right, "declared_retained_factor_reconstruction")
    residual = innovation - reconstruction
    close(arrays["rank_K_projection_residual"], residual, "rank_K_saved_projection_residual")
    close(arrays["retained_residual_orthogonality"], residual @ right.T, "retained_residual_orthogonality")
    require(float(np.abs(residual @ right.T).max()) <= energy_tolerance, "RANK_K_RESIDUAL_ORTHOGONALITY")
    if solution["unique_retained_boundary"]:
        close(reconstruction, solution["reconstruction"], "independent_eigen_rank_K_reconstruction")
        close(projector, solution["projector"], "unique_boundary_rank_K_projector")
    else:
        require(abs(float(np.sum(residual * residual)) - solution["discarded_energy"]) <= energy_tolerance,
                "DEGENERATE_BOUNDARY_OPTIMAL_ENERGY_NO_VECTOR_IDENTITY_CLAIM")
    expected_prediction = np.broadcast_to(values["raw_last"].astype(np.float64)[:, None], values["target"].shape)[complete] + reconstruction
    require(arrays["rank_K_prediction"].dtype == np.float64
            and np.isnan(arrays["rank_K_prediction"][~complete]).all(), "REAL_ORACLE_PREDICTION_ONLY_COMPLETE_TARGETS")
    close(arrays["rank_K_prediction"][complete], expected_prediction, "same_current_C_anchor_rank_K_prediction")
    oracle_scores = fixed_audit.complete_scores(arrays["rank_K_prediction"], values["target"], complete, values["std"])
    checks = prior.compare_scores(entry["rank_K_oracle_scores_same_mask"], oracle_scores, "rank_K_oracle_saved_scores")
    raw_scores = fixed_audit.complete_scores(values["raw_forecast"], values["target"], complete, values["std"])
    checks += prior.compare_scores(entry["RAW_scores_same_mask"], raw_scores, "rank_K_RAW_same_complete_mask")
    minimum = solution["normalized_mse_minimum"]
    close(entry["minimum_normalized_MSE_from_discarded_energy"], minimum, "independent_tail_MSE_minimum")
    close(oracle_scores["normalized"]["mse"], minimum, "reconstructed_MSE_matches_tail_energy")
    require(entry["complete_target_positions"] == int(complete.sum())
            and entry["total_target_positions"] == int(complete.size)
            and entry["future_matrix_shape"] == list(innovation.shape), "SAME_COMPLETE_MASK_AND_M_C_DENOMINATOR")
    certificate = entry["energy_certificate"]
    expected = {"total_innovation_energy": solution["total_energy"], "retained_energy": solution["retained_energy"],
                "discarded_energy": solution["discarded_energy"], "reconstruction_residual_energy": float(np.sum(residual * residual)),
                "SVD_squared_singularvalues_sum": float(squared.sum()), "Gram_trace": float(np.trace(solution["gram"])),
                "Gram_tail_eigenvalue_sum": solution["discarded_energy"],
                "negative_Gram_eigenvalue_tolerance": solution["negative_tolerance"],
                "absolute_energy_tolerance": energy_tolerance,
                "residual_orthogonality_max_abs": float(np.abs(residual @ right.T).max())}
    for key, value in expected.items():
        parameter = key in ("negative_Gram_eigenvalue_tolerance", "absolute_energy_tolerance")
        close(certificate[key], value, "independent_energy_certificate:" + key,
              atol=1e-20 if parameter else energy_tolerance, rtol=1e-10)
    require(certificate["checks_pass"] is True
            and certificate["normalized_MSE_denominator_M_times_C"] == innovation.size, "REAL_MSE_CERTIFICATE_M_C_DENOMINATOR")
    reported_rank = int(np.count_nonzero(singular > singular[0] * max(innovation.shape) * np.finfo(np.float64).eps)) if singular[0] > 0 else 0
    require(entry["numerical_rank_future_matrix"] == reported_rank, "NUMERICAL_RANK_DECLARATION_FROM_SAVED_SPECTRUM")
    require(set(entry["fixed_D_and_RAW_comparisons_same_mask"]) == {"PCA", "FORECA"}, "EXACT_REGISTERED_FIXED_D_COMPARISONS")
    compared = {}
    for route, fixed in fixed_rows.items():
        require(fixed["complete_target_positions"] == int(complete.sum()) and fixed["geometry"]["K"] == k,
                "RANK_K_FIXED_D_SAME_COMPLETE_MASK_AND_K")
        recorded = entry["fixed_D_and_RAW_comparisons_same_mask"][route]
        floor = fixed["fixed_D_bound_scores_same_mask"]["normalized"]["mse"]
        tolerance = 1e-10 * max(1., abs(floor), abs(minimum))
        require(minimum <= floor + tolerance and recorded["rank_K_oracle_le_fixed_D_floor"] is True,
                "GLOBAL_RANK_K_CONTAINS_THIS_FIXED_D_OUTPUT_CLASS")
        for key, value in (("fixed_D_floor_MSE_same_mask", floor), ("fixed_D_floor_minus_rank_K_oracle", floor - minimum),
                           ("RAW_MSE_same_mask", raw_scores["normalized"]["mse"]),
                           ("RAW_minus_rank_K_oracle", raw_scores["normalized"]["mse"] - minimum),
                           ("numerical_comparison_tolerance", tolerance)):
            close(recorded[key], value, "same_mask_fixed_D_comparison:" + route + ":" + key)
        require(set(recorded["saved_fixed_D_real_predictions"]) == set(fixed_audit.FIXED_METHODS),
                "ONLY_REGISTERED_FIXED_D_REAL_PREDICTIONS_HAVE_INEQUALITY")
        for method in fixed_audit.FIXED_METHODS:
            real_mse = fixed["comparisons"][method]["real_arithmetic_scores_same_mask"]["normalized"]["mse"]
            close(recorded["saved_fixed_D_real_predictions"][method]["real_arithmetic_MSE"], real_mse,
                  "bound_comparison_registered_real_prediction:" + method)
            require(minimum <= real_mse + 1e-10 * max(1., abs(real_mse), abs(minimum))
                    and recorded["saved_fixed_D_real_predictions"][method]["rank_K_oracle_le_real_arithmetic_MSE"] is True,
                    "RANK_K_BOUND_FOR_REGISTERED_FIXED_D_REAL_TRACE:" + method)
        compared[route] = {"fixed_D_real_floor": floor, "gap_above_known_future_rank_K_minimum": floor - minimum}
    return {"complete_target_positions": int(complete.sum()), "channels": innovation.shape[1], "K": k,
            "normalized_MSE_known_future_rank_K_real_minimum": minimum, "RAW_normalized_MSE_same_complete_mask": raw_scores["normalized"]["mse"],
            "independent_Gram_eigenvalues": solution["eigenvalues"].tolist(),
            "clipped_nonnegative_Gram_eigenvalues": solution["nonnegative_eigenvalues"].tolist(),
            "total_innovation_energy": solution["total_energy"], "retained_energy": solution["retained_energy"],
            "discarded_energy": solution["discarded_energy"],
            "Gram_negative_eigenvalue_roundoff_tolerance": solution["negative_tolerance"],
            "fixed_D_comparisons": compared, "score_field_checks": checks,
            "Gram_boundary_unique_with_registered_energy_tolerance": solution["unique_retained_boundary"],
            "SVD_or_producer_math_imported_or_replayed": False,
            "physical_MSE_MAE_or_input_information_or_nonlinear_limit_claimed": False,
            "RAW_prediction_bound_inequality_applied": False}


def audit_diagnostic(diagnostic_path, result_path, projection_audit_path, span_path, span_audit_path, meter):
    check = meter.check
    records = {name: {"path": str(Path(path).resolve()), "sha256": prior.file_sha(path)}
               for name, path in (("diagnostic", diagnostic_path), ("producer", result_path),
                                 ("projection_audit", projection_audit_path), ("span_diagnostic", span_path),
                                 ("span_audit", span_audit_path))}
    data = {name: prior.verified_json(record, check) for name, record in records.items()}
    diagnostic, producer, projection, span, span_audit = (data[name] for name in records)
    require(diagnostic["status"] == producer["status"] == span["status"] == "COMPLETE"
            and projection["status"] == span_audit["status"] == "PASS", "TERMINAL_ORACLE_PRODUCER_SPAN_AND_BOTH_PASS_AUDITS")
    require(diagnostic["producer"] == projection["producer"] == span["producer"] == span_audit["producer"] == records["producer"]
            and diagnostic["projection_audit"] == span["independent_projection_audit"] == records["projection_audit"]
            and diagnostic["fixed_span"] == span_audit["diagnostic"] == records["span_diagnostic"]
            and diagnostic["fixed_span_audit"] == records["span_audit"], "EXACT_FIVE_ARTIFACT_SHA_BINDINGS_BEFORE_NUMERICAL_ACCESS")
    require(diagnostic["role"] == "STRONGLY_FUTURE_INFORMED_RANK_K_DIAGNOSTIC_ONLY"
            and diagnostic["protected_TEST_accessed"] is False and diagnostic["independent_holdout"] is False
            and diagnostic["model_or_checkpoint_loaded"] is False and diagnostic["new_backbone_forward_or_training"] is False
            and diagnostic["deployment_or_method_checkpoint_branch_selection_used"] is False
            and diagnostic["oracle_target_matrix_factorizations"] == 6
            and diagnostic["source_before_after_SHA_pass"] is True
            and diagnostic["original_selected_forecasts_and_selection_lock_unchanged"] is True,
            "EXACT_RANK_ORACLE_FUTURE_INFORMED_DIAGNOSTIC_SCOPE")
    require(set(diagnostic["datasets"]) == set(producer["datasets"]) == set(DATASETS), "EXACT_TWO_RANK_ORACLE_DATASETS")
    source_records = diagnostic["source_records"] + [diagnostic["source"], diagnostic["checked_input_helper"]]
    for record in source_records:
        check()
        require(prior.file_sha(record["path"]) == record["sha256"], "REGISTERED_ORACLE_SOURCE_OR_INPUT_FILE_CHANGED")
    require(any(Path(record["path"]).resolve() == Path(producer["selection_lock"]["path"]).resolve()
                and record["sha256"] == producer["selection_lock"]["sha256"] for record in source_records),
            "REGISTERED_UNCHANGED_ORIGINAL_SELECTION_LOCK")
    forecast_records = []
    for original in producer["datasets"].values():
        for condition in original["projections"].values():
            for record in condition["bolt"]["artifacts"].values():
                forecast_records.append(record)
            for native in condition["native"].values():
                forecast_records.extend(native["artifacts"].values())
    require(len(forecast_records) == 36, "EXACT_ORIGINAL_36_SELECTED_FORECAST_ARTIFACTS")
    for record in forecast_records:
        require(prior.file_sha(record["path"]) == record["sha256"], "ORIGINAL_SELECTED_FORECAST_FILE_BYTES_CHANGED")
    report = {"status": "PASS", **records, "cells": [],
              "totals": {"dataset_split_cells_checked": 0, "fixed_D_floor_inequalities_checked": 0,
                         "fixed_D_real_method_inequalities_checked": 0, "bound_RAW_inequalities_checked": 0,
                         "score_field_checks": 0, "selected_forecast_file_hashes_checked": len(forecast_records)},
              "source_file_checks": len(source_records), "protected_TEST_accessed": False,
              "new_training_F_forward_model_or_checkpoint_loading_or_timer_replay": False,
              "physical_MSE_MAE_nonlinear_limit_input_information_or_generalization_claimed": False,
              "known_future_directions_fitted_separately_each_split_and_never_deployable": True}
    for dataset, (channels, k, development) in DATASETS.items():
        original, row = producer["datasets"][dataset], diagnostic["datasets"][dataset]
        require(row["channels"] == channels and row["K"] == k and row["development_split"] == development
                and set(row["splits"]) == set(original["inputs"]) == {"fit", "tune", development},
                "REGISTERED_RANK_K_DATASET_AND_SPLITS")
        for split, entry in row["splits"].items():
            check()
            require(entry["source_input"] == original["inputs"][split], "EXACT_ORIGINAL_RANK_ORACLE_INPUT_RECORD")
            values = prior.verified_arrays(original["inputs"][split], check)
            arrays = prior.verified_arrays(entry["artifact"], check)
            fixed_rows = {route: span["datasets"][dataset]["projections"][route][split] for route in ("PCA", "FORECA")}
            for fixed in fixed_rows.values():
                require(fixed["source_input"] == original["inputs"][split], "SAME_AUDITED_INPUT_FOR_FIXED_D_ORACLE_COMPARISON")
            cell = audit_cell(entry, arrays, values, k, fixed_rows)
            report["cells"].append({"dataset": dataset, "split": split, **cell})
            report["totals"]["dataset_split_cells_checked"] += 1
            report["totals"]["fixed_D_floor_inequalities_checked"] += 2
            report["totals"]["fixed_D_real_method_inequalities_checked"] += 8
            report["totals"]["score_field_checks"] += cell["score_field_checks"]
    require(report["totals"]["dataset_split_cells_checked"] == 6
            and report["totals"]["fixed_D_floor_inequalities_checked"] == 12
            and report["totals"]["fixed_D_real_method_inequalities_checked"] == 48
            and report["totals"]["bound_RAW_inequalities_checked"] == 0, "COMPLETE_RANK_ORACLE_AUDIT_COUNTS")
    for record in list(records.values()) + source_records + forecast_records:
        require(prior.file_sha(record["path"]) == record["sha256"], "RANK_AUDIT_SOURCE_ORIGINAL_FORECAST_CHANGED_DURING_EXECUTION")
    return report


def execute(diagnostic, result, projection_audit, span_diagnostic, span_audit, output):
    from continuation_runtime_v1 import Meter, write_new
    output_path = Path(output).resolve()
    require(not output_path.exists(), "FRESH_INDEPENDENT_RANK_ORACLE_AUDIT_OUTPUT")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meter = Meter("audit_rank_oracle_" + output_path.stem, gpu=False)
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
            report = audit_diagnostic(diagnostic, result, projection_audit, span_diagnostic, span_audit, meter)
        report["protected_guard_paths"] = guarded_paths
        meter.protected_check()
        require(prior.file_sha(__file__) == source["sha256"], "RANK_ORACLE_AUDITOR_SOURCE_CHANGED")
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        report = {"status": "FAILED_PRESERVED", "failure": failure, "full_scope_PASS_claimed": False}
    finally:
        report["audit_source"] = source
        report["independent_helper_sources"] = [{"path": str(Path(module.__file__).resolve()), "sha256": prior.file_sha(module.__file__)}
                                                for module in (prior, fixed_audit)]
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
    parser.add_argument("--span-diagnostic", required=True)
    parser.add_argument("--span-audit", required=True)
    parser.add_argument("--output", required=True)
    raise SystemExit(execute(**vars(parser.parse_args())))
