"""Reproduce preserved Gram score guard from FIT past only, with direct-LS evidence."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import math
import sys

import audit_projection_results_v1 as prior

HERE, ROOT = prior.HERE, prior.ROOT
RCOND, GRAM_CUTOFF = 1e-4, 1e-8
OLD_RESULT_SHA = "4240cedea826e052b227ef2fba4e16ff52cdcd6db69281844485eca6c0b0bd6b"
OLD_MATH_SHA = "2583b221d1d271a38a3070455d4a2a72c561c101d66ac5b9707822f33ed8909b"
DATASETS = {"robin": (17, 4), "beijing_aotizhongxin": (11, 2)}
np = sys.modules.get("numpy")
require = prior.require


def initialize_numpy():
    global np
    prior.initialize_numpy(); np = sys.modules["numpy"]


def raw_binding(value):
    return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}


def channel_statistics(history, observed):
    require(history.dtype == np.float32 and history.ndim == 3 and len(history) > 0 and history.shape[1] == 512 and np.isfinite(history).all() and observed.dtype == np.bool_ and observed.shape == history.shape, "FINITE_FIT_PAST_AND_OBSERVED_MASK_ONLY")
    flat, mask = history.reshape(-1, history.shape[-1]).astype(np.float64), observed.reshape(-1, history.shape[-1])
    stats = []
    for channel in range(flat.shape[-1]):
        values = flat[mask[:, channel]]
        if not len(values): stats.append(None); continue
        mean = values.mean(axis=0); centered = values - mean; response = centered[:, channel]
        gram = centered.T @ centered / len(values); gram = (gram + gram.T) / 2
        stats.append({"count": len(values), "gram": gram, "cross": centered.T @ response / len(values), "response_variance": float(np.mean(response * response))})
    return flat, mask, stats


def inspect_response(flat, mask, stats, channel, indices):
    active = stats[channel]
    if active is None: return {"observed_count": 0, "skipped_unobserved_response": True}
    gram = active["gram"][np.ix_(indices, indices)]; q = active["cross"][indices]
    eigenvalues, vectors = np.linalg.eigh(gram)
    scale = max(float(eigenvalues[-1]), 0.)
    psd_tolerance = 128 * np.finfo(np.float64).eps * max(1., float(np.linalg.norm(gram, 2)))
    require(eigenvalues[0] >= -psd_tolerance, "PRESERVED_GRAM_PSD_GUARD_FAILED_BEFORE_NEGATIVE_SCORE")
    retained = eigenvalues > scale * GRAM_CUTOFF
    if scale == 0: retained[:] = False
    coefficient = np.zeros(len(indices), np.float64)
    if retained.any():
        basis = vectors[:, retained]
        coefficient = basis @ ((basis.T @ q) / eigenvalues[retained])
    variance = float(active["response_variance"])
    cross_term, quadratic = float(coefficient @ q), float(coefficient @ gram @ coefficient)
    error = float(variance - 2 * coefficient @ q + coefficient @ gram @ coefficient)
    tolerance = 512 * np.finfo(np.float64).eps * max(1., abs(variance), abs(cross_term), abs(quadratic))
    values = flat[mask[:, channel]]
    features, target = values[:, indices], values[:, channel]
    centered, response = features - features.mean(axis=0), target - target.mean()
    direct, _, rank, singular = np.linalg.lstsq(centered, response, rcond=RCOND)
    direct_residual, gram_residual = centered @ direct - response, centered @ coefficient - response
    direct_mse, gram_residual_mse = float(np.mean(direct_residual * direct_residual)), float(np.mean(gram_residual * gram_residual))
    require(math.isfinite(direct_mse) and direct_mse >= 0 and math.isfinite(gram_residual_mse) and gram_residual_mse >= 0, "DIRECT_SQUARED_RESIDUALS_FINITE_NONNEGATIVE")
    return {"response_original_index": channel, "candidate_indices": list(indices), "observed_count": int(active["count"]),
        "response_centered_variance": variance, "Gram_matrix": gram.tolist(), "cross_q": q.tolist(), "Gram_eigenvalues": eigenvalues.tolist(),
        "Gram_coefficient": coefficient.tolist(), "coefficient_dot_q": cross_term, "coefficient_Gram_coefficient": quadratic,
        "subtractive_Gram_error": error, "original_negative_score_tolerance": float(tolerance), "negative_score_guard_failed": bool(error < -tolerance),
        "negative_error_divided_by_original_tolerance": float(-error / tolerance) if error < 0 else 0.,
        "Gram_retained_rank": int(retained.sum()), "Gram_retained_condition": float(np.sqrt(scale / eigenvalues[retained][0])) if retained.any() else None,
        "direct_centered_lstsq_coefficients": direct.tolist(), "direct_centered_lstsq_rank": int(rank), "direct_centered_design_singularvalues": singular.tolist(),
        "direct_centered_design_retained_condition": float(singular[0] / singular[rank - 1]) if rank else None,
        "direct_centered_lstsq_residual_MSE": direct_mse, "same_Gram_coefficients_direct_residual_MSE": gram_residual_mse,
        "same_Gram_coefficient_residual_MSE_minus_subtractive_score": gram_residual_mse - error,
        "Gram_vs_direct_coefficient_max_abs": float(np.max(np.abs(coefficient - direct))),
        "direct_residual_max_abs": float(np.max(np.abs(direct_residual))), "Gram_residual_max_abs": float(np.max(np.abs(gram_residual))),
        "cutoff_singular": RCOND, "cutoff_Gram_eigen": GRAM_CUTOFF, "original_tolerance_relaxed": False,
        "future_labels_forecasts_or_TEST_used": False, "forecast_method_or_training_failure_inferred": False}


def first_failure(history, observed, k):
    flat, mask, stats = channel_statistics(history, observed)
    c = history.shape[-1]
    require(type(k) is int and 1 <= k <= c and any(value is not None for value in stats), "REGISTERED_K_AND_ACTIVE_FIT_RESPONSES")
    order, completed, inspected = [], [], 0
    for stage in range(1, k + 1):
        candidates = []
        for index in range(c):
            if index in order: continue
            indices = sorted([*order, index]); errors = []
            for channel in range(c):
                value = inspect_response(flat, mask, stats, channel, indices)
                if value.get("skipped_unobserved_response"): continue
                inspected += 1
                if value["negative_score_guard_failed"]:
                    return {"status": "REPRODUCED_ORIGINAL_NEGATIVE_GRAM_SCORE_GUARD", "failure_stage": stage, "candidate_original_index": index,
                        "already_selected_original_indices": list(order), "completed_stages": completed, "response_evaluations_until_first_failure": inspected,
                        "first_failure": value, "original_greedy_attempt_stopped_here": True, "new_subset_selected": False, "new_F_native_training_or_timing_executed": False}
                errors.append(max(0., value["subtractive_Gram_error"]))
            candidates.append({"candidate_original_index": index, "macro_normalized_MSE": float(np.mean(errors))})
        chosen = min(candidates, key=lambda row: (row["macro_normalized_MSE"], row["candidate_original_index"]))
        order.append(chosen["candidate_original_index"])
        completed.append({"stage": stage, "selected_original_index": order[-1], "all_original_numeric_candidate_scores": candidates})
    return {"status": "ORIGINAL_NEGATIVE_GUARD_NOT_REPRODUCED", "completed_stages": completed, "original_selection_order_replayed": order, "response_evaluations_until_first_failure": inspected, "new_subset_selected": False, "new_F_native_training_or_timing_executed": False}


def read_fit_past_only(record, check):
    path = Path(record["path"]).resolve()
    require(path.is_relative_to(ROOT) and path.suffix.lower() == ".npz" and path.name.lower() not in ("test.npz", "jena_test.npz") and prior.file_sha(path) == record["sha256"], "EXACT_APPROVED_FIT_NPZ_ONLY")
    check()
    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name].copy() for name in ("history", "history_observed")}
    for name, value in values.items(): require(record["arrays"][name] == raw_binding(value), "FIT_PAST_ARRAY_SHA_SHAPE_DTYPE:" + name)
    return values


def source_contract():
    path = HERE / "raw_subset_projection_v1.py"
    require(prior.file_sha(path) == OLD_MATH_SHA, "PRESERVED_ORIGINAL_EXECUTED_SUBSET_MATH_SOURCE")
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    score = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "score_subset")
    text = ast.unparse(score)
    require("stats.response_variance[channel] - 2 * coefficient @ q + coefficient @ gram @ coefficient" in text and "512 * np.finfo(np.float64).eps" in text and "error < -tolerance" in text, "PRESERVED_SUBTRACTIVE_SCORE_AND_ORIGINAL_GUARD")
    return {"path": str(path), "sha256": OLD_MATH_SHA}


def verify_recorded_Gram_case(row):
    gram, q, coefficient = row["Gram"], row["cross"], row["beta_Gram"]
    require(row["candidate_index"] == row["response_channel"] and len(gram) == len(q) == len(coefficient) == 1 and len(gram[0]) == 1 and row["observed_rows"] > 1, "FIRST_STAGE_SELF_RESPONSE_RECONSTRUCTION_CASE")
    variance = row["variance"]
    cross, quadratic = coefficient[0] * q[0], coefficient[0] * gram[0][0] * coefficient[0]
    error = variance - 2 * cross + quadratic
    tolerance = 512 * sys.float_info.epsilon * max(1., abs(variance), abs(cross), abs(quadratic))
    for actual, expected in ((cross, row["cross_contribution"]), (quadratic, row["quadratic_contribution"]), (error, row["Gram_MSE"]), (tolerance, row["original_negative_tolerance"])):
        require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=2e-15), "RECORDED_FIRST_NEGATIVE_SCALAR_ARITHMETIC")
    require(error < -tolerance and row["negative_over_tolerance"] > 1 and math.isclose(-error / tolerance, row["negative_over_tolerance"], rel_tol=1e-12), "NEGATIVE_SCORE_EXCEEDS_UNCHANGED_ORIGINAL_GUARD")
    direct_mse = row["direct_centered_residual_MSE"]
    require(math.isfinite(direct_mse) and direct_mse >= 0 and row["direct_rank"] == 1 and len(row["direct_singular_values"]) == 1 and row["direct_singular_values"][0] > 0 and len(row["direct_lstsq_beta"]) == 1 and math.isfinite(row["direct_residual_max_abs"]) and row["direct_residual_max_abs"] >= 0 and direct_mse <= row["direct_residual_max_abs"] ** 2, "RECORDED_DIRECT_LS_SELF_CHANNEL_RESIDUAL_CERTIFICATE_SCOPE")
    return {"candidate_original_index": row["candidate_index"], "response_original_index": row["response_channel"], "observed_rows": row["observed_rows"], "recomputed_subtractive_Gram_MSE": error, "unchanged_original_guard_tolerance": tolerance, "negative_to_tolerance_ratio": -error / tolerance, "recorded_direct_residual_MSE": direct_mse, "recorded_direct_rank": 1, "direct_LS_reexecuted_by_this_metadata_check": False, "original_tolerance_relaxed": False, "model_prediction_quality_failure_inferred": False}


def verify_existing_receipt(receipt_path, output_path):
    from continuation_runtime_v1 import Meter, write_new
    output = Path(output_path).resolve(); require(not output.exists(), "FRESH_RECORDED_ARITHMETIC_VERIFICATION_OUTPUT")
    meter = Meter("raw_subset_recorded_Gram_arithmetic_" + output.stem, gpu=False); failure = None
    code = {"path": str(Path(__file__).resolve()), "sha256": prior.file_sha(__file__)}
    try:
        meter.protected_check()
        path = Path(receipt_path).resolve(); record = {"path": str(path), "sha256": prior.file_sha(path)}
        receipt = prior.verified_json(record, meter.check)
        require(receipt["status"] == "CONFIRMED_FIRST_STAGE_GRAM_CANCELLATION_VS_DIRECT_RESIDUAL" and receipt["opened_member_names"] == ["history", "history_observed"] and receipt["FIT_future_targets_read"] is receipt["F_head_forward_training"] is receipt["TUNE_DEV_read"] is receipt["protected_TEST_accessed"] is False, "RECORDED_ROOT_FIT_PAST_ONLY_DIAGNOSTIC_SCOPE")
        require(receipt["old_source"]["sha256"] == OLD_MATH_SHA and receipt["partial_result"]["sha256"] == OLD_RESULT_SHA, "SAME_PRESERVED_EXECUTED_SOURCE_AND_PARTIAL_RESULT")
        original = source_contract()
        partial = prior.verified_json(receipt["partial_result"], meter.check)
        source = prior.verified_json(partial["source_producer"], meter.check)
        require(partial["status"] == "PARTIAL_COMPLETED" and partial["subset_selection_fits"] == partial["actual_subset_F_split_inferences"] == 0 and partial["F_before"] == partial["F_after"], "SAME_ZERO_SUBSET_F_PARTIAL_ATTEMPT")
        result = {}
        for dataset in DATASETS:
            entry = receipt["datasets"][dataset]
            require(entry["FIT_input"] == source["datasets"][dataset]["inputs"]["fit"], "RECORDED_EXACT_ORIGINAL_FIT_PAST_FILE_BINDING")
            result[dataset] = verify_recorded_Gram_case(entry["first_Gram_failure"])
        terminal = prior.verified_json(receipt["resource_terminal"], meter.check)
        require(terminal["status"] == receipt["status"] and terminal["gpu_process"] is False, "ACTUAL_ROOT_CPU_DIAGNOSTIC_RESOURCE_RECORD")
        require(prior.file_sha(path) == record["sha256"] and prior.file_sha(__file__) == code["sha256"], "RECORDED_DIAGNOSTIC_AND_NEW_SOURCE_PRESERVED")
        meter.protected_check()
        value = {"status": "INDEPENDENT_RECORDED_GRAM_ARITHMETIC_VERIFIED_FIT_NOT_REOPENED", "actual_FIT_diagnostic": record, "old_source": original, "partial_result": receipt["partial_result"], "datasets": result, "NPZ_or_FIT_past_arrays_reopened": False, "direct_LS_or_Gram_from_data_reexecuted": False, "future_arrays_model_training_or_TEST_used": False, "unchanged_original_tolerance_verified": True, "method_prediction_failure_inferred": False}
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}; value = {"status": "FAILED_RECORDED_ARITHMETIC_VERIFICATION_PRESERVED", "failure": failure}
    finally:
        value["diagnostic_source"] = code; value["resource_terminal"] = meter.close(value["status"], failure); write_new(output, value)
    print(json.dumps({"status": value["status"], "output": str(output), "failure": failure}), flush=True)
    return 0 if value["status"] == "INDEPENDENT_RECORDED_GRAM_ARITHMETIC_VERIFIED_FIT_NOT_REOPENED" else 2


def execute(result_path, output_path):
    from continuation_runtime_v1 import Meter, write_new
    output = Path(output_path).resolve(); require(not output.exists(), "FRESH_FAILURE_DIAGNOSTIC_OUTPUT")
    meter = Meter("raw_subset_Gram_failure_" + output.stem, gpu=False); failure = None
    code = {"path": str(Path(__file__).resolve()), "sha256": prior.file_sha(__file__)}
    try:
        meter.protected_check(); path = Path(result_path).resolve(); record = {"path": str(path), "sha256": prior.file_sha(path)}
        result = prior.verified_json(record, meter.check)
        require(record["sha256"] == OLD_RESULT_SHA and result["status"] == "PARTIAL_COMPLETED" and result["subset_selection_fits"] == result["actual_subset_F_split_inferences"] == 0 and result["F_before"] == result["F_after"] and result["protected_TEST_accessed"] is False, "PRESERVED_ACTUAL_SUBSET_ARITHMETIC_FAILURE")
        original_math = source_contract(); source = prior.verified_json(result["source_producer"], meter.check)
        require(source["status"] == "COMPLETE" and result["source_producer"]["sha256"] == "5d213da750bc6eb3b43dcb9d25fa08c3313f4c2662148cc2bad45adcc2cf6f98", "EXACT_ORIGINAL_ACCEPTED_FIT_SOURCE")
        initialize_numpy(); sys.path.insert(0, str(ROOT / "research/smallk_failure_followup_20261014"))
        from prepare_inputs import install_protected_test_guard
        install_protected_test_guard()
        from threadpoolctl import threadpool_limits
        datasets = {}
        with threadpool_limits(limits=2):
            for dataset, (channels, k) in DATASETS.items():
                source_record = source["datasets"][dataset]["inputs"]["fit"]
                values = read_fit_past_only(source_record, meter.check)
                require(values["history"].shape == (256, 512, channels), "ORIGINAL_FIT_PAST_WINDOW_SHAPE")
                diagnostic = first_failure(values["history"], values["history_observed"], k)
                require(diagnostic["status"] == "REPRODUCED_ORIGINAL_NEGATIVE_GRAM_SCORE_GUARD", "ACTUAL_ORIGINAL_FIRST_NEGATIVE_GUARD_REPRODUCTION")
                diagnostic["FIT_source"] = source_record; datasets[dataset] = diagnostic
                require(prior.file_sha(source_record["path"]) == source_record["sha256"], "ORIGINAL_FIT_NPZ_PRESERVED")
        require(prior.file_sha(path) == record["sha256"] and prior.file_sha(__file__) == code["sha256"], "ORIGINAL_PARTIAL_RESULT_AND_DIAGNOSTIC_SOURCE_PRESERVED")
        meter.protected_check()
        report = {"status": "CONFIRMED_PRESERVED_NUMERIC_SCORE_CONTRACT_FAILURE", "partial_result": record, "original_math_source": original_math, "source_producer": result["source_producer"], "source_audit": result["source_audit"], "datasets": datasets, "opened_arrays": ["FIT.history", "FIT.history_observed"], "future_target_or_raw_F_forecast_arrays_opened": False, "new_F_native_training_or_timing_executed": False, "protected_TEST_accessed": False, "numeric_tolerance_relaxed": False, "method_prediction_quality_failure_inferred": False, "new_subset_or_winner_adopted": False}
        json.dumps(report, allow_nan=False)
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}; report = {"status": "FAILED_DIAGNOSTIC_PRESERVED", "failure": failure}
    finally:
        report["diagnostic_source"] = code; report["resource_terminal"] = meter.close(report["status"], failure); write_new(output, report)
    print(json.dumps({"status": report["status"], "output": str(output), "failure": failure}), flush=True)
    return 0 if report["status"] == "CONFIRMED_PRESERVED_NUMERIC_SCORE_CONTRACT_FAILURE" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True); mode.add_argument("--result"); mode.add_argument("--root-receipt")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raise SystemExit(verify_existing_receipt(args.root_receipt, args.output) if args.root_receipt else execute(args.result, args.output))
