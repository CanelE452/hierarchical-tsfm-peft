"""Locked-D[n] output geometry and E(Y) diagnostics; no forecast/model fitting."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import audit_projection_results_v1 as prior
import audit_local_past_decoder_probe_v1 as local_audit

np = sys.modules.get("numpy")
HERE, ROOT = prior.HERE, prior.ROOT
DATASETS = local_audit.DATASETS
RCOND = 1e-15
require, exact = prior.require, local_audit.exact


def initialize_numpy():
    global np
    local_audit.initialize_numpy()
    np = sys.modules["numpy"]


def selected_real_space(values, decoder32):
    target, current, observed = values["target"], values["raw_last"], values["observed"]
    require(target.dtype == current.dtype == decoder32.dtype == np.float32 and target.ndim == 3 and target.shape[1] == 48
            and observed.dtype == np.bool_ and observed.shape == target.shape and current.shape == (len(target), target.shape[-1])
            and decoder32.ndim == 3 and decoder32.shape[0] == len(target) and decoder32.shape[-1] == target.shape[-1]
            and 1 <= decoder32.shape[1] <= target.shape[-1]
            and np.isfinite(current).all() and np.isfinite(decoder32).all() and np.isfinite(target[observed]).all(), "FINITE_LOCKED_ORIGIN_SPECIFIC_D32_AND_ORIGINAL_COMPLETE_C_TARGET_CONTRACT")
    complete = observed.all(axis=-1)
    require(complete.any(), "NO_COMPLETE_C_TARGET_POSITIONS_FOR_SELECTED_OUTPUT_SPACE")
    n, h, c = target.shape
    k = decoder32.shape[1]
    delta = np.zeros((n, h, k), np.float64)
    prediction = np.broadcast_to(current.astype(np.float64)[:, None], target.shape).copy()
    projectors = np.zeros((n, c, c), np.float64)
    spectra = np.zeros((n, min(k, c)), np.float64)
    ranks, certificates, energy = [], [], 0.
    for window in range(n):
        d = decoder32[window].astype(np.float64)
        u, singular, vh = np.linalg.svd(d, full_matrices=False)
        keep = singular > singular[0] * RCOND if len(singular) else np.zeros(0, bool)
        rank = int(keep.sum())
        p = vh[keep].T @ vh[keep] if rank else np.zeros((c, c), np.float64)
        require(np.max(np.abs(p - p.T)) <= 1e-10 and np.max(np.abs(p @ p - p)) <= 1e-10
                and abs(float(np.trace(p)) - rank) <= 1e-10, "SELECTED_D_NUMERICAL_ROW_PROJECTOR_SYMMETRY_IDEMPOTENCE_RANK")
        projectors[window], spectra[window] = p, singular
        ranks.append(rank)
        valid = complete[window]
        if not valid.any():
            certificates.append({"window": window, "complete_positions": 0, "rank_D": rank})
            continue
        v = target[window, valid].astype(np.float64) - current[window].astype(np.float64)
        svd_coordinates = (v @ vh[keep].T / singular[keep]) @ u[:, keep].T if rank else np.zeros((len(v), k), np.float64)
        best = svd_coordinates @ d
        independent_coordinates, _, ls_rank, _ = np.linalg.lstsq(d.T, v.T, rcond=RCOND)
        independent_projection = independent_coordinates.T @ d
        prior.close(best, v @ p, "selected_SVD_coordinate_and_projector_projection", atol=1e-10, rtol=1e-10)
        prior.close(best, independent_projection, "independent_direct_LS_selected_D_projection", atol=1e-10, rtol=1e-10)
        require(int(ls_rank) == rank, "INDEPENDENT_LS_AND_SVD_SELECTED_D_NUMERICAL_RANK")
        residual = v - best
        normal = residual @ d.T
        tolerance = float(1e-10 * (1 + np.linalg.norm(v) * np.linalg.norm(d)))
        require(np.max(np.abs(normal)) <= tolerance, "SELECTED_D_FUTURE_DIAGNOSTIC_NORMAL_RESIDUAL_CERTIFICATE")
        retained_energy, discarded_energy = float(np.square(best).sum()), float(np.square(residual).sum())
        prior.close(float(np.square(v).sum()), retained_energy + discarded_energy, "selected_D_projection_energy_decomposition", atol=1e-10, rtol=1e-10)
        delta[window, valid], prediction[window, valid] = svd_coordinates, current[window].astype(np.float64) + best
        energy += discarded_energy
        certificates.append({"window": window, "complete_positions": int(valid.sum()), "rank_D": rank,
            "normal_residual_max_abs": float(np.abs(normal).max()), "normal_tolerance": tolerance,
            "retained_energy": retained_energy, "discarded_energy": discarded_energy})
    mask = np.broadcast_to(complete[:, :, None], target.shape)
    score = prior.scores(prediction, target, mask, values["std"])
    mse = energy / (int(complete.sum()) * c)
    prior.close(score["normalized"]["mse"], mse, "selected_D_complete_macro_MSE_discarded_energy_denominator", atol=1e-10, rtol=1e-10)
    return {"complete": complete, "prediction": prediction, "optimal_latent_delta_real64": delta,
            "projectors_real64": projectors, "singularvalues": spectra, "rank_D_per_origin": ranks, "certificates": certificates,
            "scores": score, "minimum_normalized_MSE_real_numeric_rowspace": mse}


def encode_Y(target32, center, encoder, backend):
    require(target32.dtype == np.float32 and target32.ndim == 3 and target32.shape[1] == 48 and np.isfinite(target32).all(), "FINITE_FP32_NORMALIZED_TARGET_FOR_NONDEPLOYABLE_POINTWISE_EY")
    if backend == "BOLT_NUMPY":
        return prior.encode(target32, center, encoder)
    require(backend == "NATIVE_TORCH32", "EXACT_REGISTERED_EY_NUMERIC_BACKEND")
    import torch
    torch.set_num_threads(2)
    center_tensor = torch.from_numpy(np.ascontiguousarray(center.astype(np.float32)))
    encoder_tensor = torch.from_numpy(np.ascontiguousarray(encoder.astype(np.float32)))
    chunks = []
    with torch.no_grad():
        for start in range(0, len(target32), 32):
            values = torch.from_numpy(np.ascontiguousarray(target32[start:start + 32]))
            chunks.append(((values - center_tensor) @ encoder_tensor).numpy().copy())
    return np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)


def score_real_and_FP32(actual32, current32, future32, last32, decoder32, values, space, label):
    require(actual32.dtype == np.float32, "SAVED_ACTUAL_FP32_FORECAST_REQUIRED:" + label)
    exact(actual32, local_audit.local_forecast(current32, future32, last32, decoder32), "locked_D_actual_same_cached_latent_forecast:" + label)
    real = current32.astype(np.float64)[:, None] + np.einsum("nhk,nkc->nhc", future32.astype(np.float64) - last32.astype(np.float64)[:, None], decoder32.astype(np.float64), optimize=False)
    complete_mask = np.broadcast_to(space["complete"][:, :, None], values["target"].shape)
    actual_score = prior.scores(actual32, values["target"], complete_mask, values["std"])
    real_score = prior.scores(real, values["target"], complete_mask, values["std"])
    actual_mse, real_mse = actual_score["normalized"]["mse"], real_score["normalized"]["mse"]
    rounding = abs(actual_mse - real_mse)
    minimum = space["minimum_normalized_MSE_real_numeric_rowspace"]
    require(real_mse + 1e-10 >= minimum and actual_mse + rounding + 1e-10 >= minimum, "SELECTED_D_REAL_ROWSPACE_ONLY_BOUND_WITH_MEASURED_FP32_ROUNDING:" + label)
    return {"scores_complete_C": actual_score, "real_trace_scores_complete_C": real_score,
            "measured_FP32_rounding_normalized_MSE_allowance": rounding,
            "real_trace_minus_minimum_normalized_MSE": real_mse - minimum,
            "actual_FP32_minus_minimum_normalized_MSE": actual_mse - minimum}, real


def save_npz(path, arrays):
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    return {"path": str(Path(path).resolve()), "sha256": prior.file_sha(path), "bytes": Path(path).stat().st_size,
            "arrays": {key: local_audit.raw_binding(value) for key, value in arrays.items()}}


def audit_result(result_path, audit_path, output, meter):
    check = meter.check
    record = {"path": str(Path(result_path).resolve()), "sha256": prior.file_sha(result_path)}
    audit_record = {"path": str(Path(audit_path).resolve()), "sha256": prior.file_sha(audit_path)}
    result, accepted = prior.verified_json(record, check), prior.verified_json(audit_record, check)
    require(result["status"] == "COMPLETE" and accepted["status"] == "PASS" and accepted["diagnostic"] == record, "COMPLETE_SHRINKAGE_AND_EXACT_INDEPENDENT_PASS_REQUIRED")
    require(record["sha256"] == "aa553512965bf9253fae1983424e2fcb4f2916cc95ec12e9ed671ae325dbd532"
            and audit_record["sha256"] == "145999ea8bc7068eba08f08039287a33097cdb4e9ec8dab002f4ebc7630fdda9", "THIS_FIXED_SHRINKAGE_RESULT_AND_PASS_ONLY_NO_ASSET_ASSUMPTION")
    code = accepted["audit_source"]
    require(Path(code["path"]).name == "audit_local_decoder_shrinkage_v1.py" and prior.file_sha(code["path"]) == code["sha256"], "IMMUTABLE_REGISTERED_SHRINKAGE_INDEPENDENT_AUDITOR")
    counts = {"dataset_split_cells_checked": 6, "RAW_control_cells_checked": 6, "reused_control_method_pairs_checked": 36, "reused_control_prediction_arrays_byte_preserved": 72,
        "positive_rho_backend_states_checked": 88, "FIT_TUNE_positive_rho_prediction_arrays_checked": 120, "selected_DEV_prediction_arrays_checked": 12,
        "positive_rho_forecast_arrays_checked": 132, "window_fits_checked": 13504, "channel_slots_checked": 187664,
        "observed_channel_fits_checked": 187664, "positive_lambda_solver_calls_checked": 187664, "zero_trace_prior_fallbacks_checked": 0,
        "no_observation_prior_fallbacks_checked": 0, "native_seed_D_reuses_checked": 6752, "selection_locks_checked": 4, "score_field_checks": 6960,
        "protected_prior_file_hashes_checked": 916}
    require(all(accepted["totals"].get(key) == value for key, value in counts.items()) and result["protected_TEST_accessed"] is accepted["protected_TEST_accessed"] is False
            and result["future_labels_used_for_decoder_fit"] is False and result["development_positive_rho_sweep_executed"] is False
            and result["original_source_cache_model_and_selection_SHA_unchanged"] is True, "FULL_ACCEPTED_FIXED_SELECTION_SHRINKAGE_SCIENCE_SCOPE")
    producer = prior.verified_json(result["producer"], check)
    require(producer["status"] == "COMPLETE" and accepted["producer"] == result["producer"], "EXACT_ORIGINAL_INPUT_E_AND_F_CACHE_PRODUCER")
    sources = [{"path": str(HERE / name), "sha256": prior.file_sha(HERE / name)} for name in ("audit_selected_shrinkage_output_space_v1.py", "PLAN_SELECTED_SHRINKAGE_OUTPUT_SPACE_KO_v1.md",
        "audit_projection_results_v1.py", "audit_local_past_decoder_probe_v1.py", "audit_decoder_metric_probe_v1.py", "native_head_v1.py", "native_batch_contract_v2.py")]
    guarded = [record, audit_record, code, result["producer"], *sources, *result["source_records"], *result["input_records"], *result["protected_prior_records"]]
    locks = {}
    for dataset in DATASETS:
        for route in ("PCA_PINV", "FIT_TARGET_PINV"):
            binding = result["selection_locks"][dataset][route]
            lock = prior.verified_json(binding, check)
            require(lock["status"] == "LOCKED" and lock["selected_candidate"] in ("RHO_0.001", "RHO_0.01", "RHO_0.1", "RHO_1", "RHO_10")
                    and lock["native_specific_selection"] is False and lock["development_or_oracle_scores_used_for_selection"] is False
                    and lock["native_uses_same_Bolt_selected_candidate"] is True, "EXPECTED_SELECTED_POSITIVE_RHO_ONLY_DIAGNOSTIC_NO_NEW_BRANCH_CONSTRUCTION")
            require(accepted["selection_locks"][dataset][route]["record"] == binding
                    and accepted["selection_locks"][dataset][route]["selected_candidate"] == lock["selected_candidate"], "AUDITED_SELECTION_LOCK_IDENTITY")
            locks[(dataset, route)] = (binding, lock)
            guarded.append(binding)
    for item in guarded:
        check()
        require(prior.file_sha(item["path"]) == item["sha256"], "UNCHANGED_SELECTED_SPACE_SOURCE_INPUT_STATE_CACHE_AND_LOCK")
    report = {"status": "RUNNING", "shrinkage_diagnostic": record, "shrinkage_audit": audit_record,
        "producer": result["producer"], "fit_target_audit": result["fit_target_audit"], "source_records": sources,
        "selection_locks": result["selection_locks"], "datasets": {}, "totals": {"selected_output_space_cells_checked": 0,
            "selected_DEV_backend_cells_checked": 0, "E_Y_diagnostic_predictions_checked": 0, "actual_predictions_checked": 0,
            "real_output_space_inequalities_checked": 0, "RAW_bound_inequalities_checked": 0, "score_field_checks": 0,
            "selected_D_origin_matrices_checked": 0, "protected_prior_file_hashes_checked": len(result["protected_prior_records"])},
        "role": "DIAGNOSTIC_ONLY_LOCKED_PAST_FITTED_SELECTED_D_PER_ORIGIN_REAL_OUTPUT_SPACE",
        "future_labels_used_for_diagnostic_coordinates_and_EY": True, "future_labels_used_to_fit_or_modify_D": False,
        "new_rho_or_branch_or_checkpoint_selection": False, "new_training_or_F_head_forward": False,
        "pointwise_EY_CPU_arithmetic_only": True, "protected_TEST_accessed": False, "new_GPU_execution": False,
        "minimum_claim_only_normalized_MSE_in_registered_real_D32_numeric_rowspace": True,
        "strict_lower_bound_over_all_FP32_outputs_claimed": False, "MAE_or_physical_MSE_minimum_claimed": False,
        "old_static_or_global_rankK_floor_reused": False, "forecastability_or_independent_generalization_proven": False}
    for dataset, (channels, k, development, dev_count) in DATASETS.items():
        original = producer["datasets"][dataset]
        row = {"channels": channels, "K": k, "splits": {}}
        report["datasets"][dataset] = row
        for split, count in (("fit", 256), ("tune", 64), (development, dev_count)):
            values = prior.verified_arrays(original["inputs"][split], check)
            prior.check_raw_split(values, channels=channels, windows=count)
            complete_mask = prior.oracle_mask(values["observed"])
            require(complete_mask.any(), "ACTUAL_SELECTED_SPACE_SPLIT_HAS_NO_COMPLETE_C_FUTURE_DIAGNOSTIC")
            raw_complete = prior.scores(values["raw_forecast"], values["target"], complete_mask, values["std"])
            raw_full = prior.scores(values["raw_forecast"], values["target"], values["observed"], values["std"])
            cell = {"RAW_scores_complete_C": raw_complete, "RAW_scores_all_observed": raw_full,
                    "complete_target_positions": int(values["observed"].all(axis=-1).sum()), "routes": {}}
            row["splits"][split] = cell
            for route in ("PCA_PINV", "FIT_TARGET_PINV"):
                lock_record, lock = locks[(dataset, route)]
                label = lock["selected_candidate"]
                selected = result["datasets"][dataset]["splits"][split]["routes"][route]["positive_rho_candidates"][label]
                require(selected["rho"] == lock["selected_rho"] and set(selected["backends"]) == {"BOLT_NUMPY", "NATIVE_TORCH32"}, "EXACT_SELECTED_RHO_SAVED_FIT_TUNE_DEV_BACKENDS")
                state_encoder = prior.verified_arrays(original["projections"][route]["state_artifact"], check)
                out_route = {"selected_candidate": label, "selected_rho": lock["selected_rho"], "selection_lock": lock_record, "backends": {}}
                cell["routes"][route] = out_route
                for backend, entry in selected["backends"].items():
                    check()
                    require(entry["status"] == "COMPLETE" and entry["state_metadata"]["rho"] == lock["selected_rho"]
                            and entry["state_metadata"]["future_labels_used_in_decoder_fit"] is False, "LOCKED_PAST_ONLY_SELECTED_STATE_BEFORE_ANY_FUTURE_DIAGNOSTIC")
                    state = prior.verified_arrays(entry["state_artifact"], check)
                    predictions = prior.verified_arrays(entry["artifact"], check)
                    require(json.loads(str(state["metadata_json"].item())) == entry["state_metadata"], "SELECTED_STATE_EMBEDDED_METADATA_BINDING")
                    d32 = state["decoder_deployed_FP32"]
                    require(d32.dtype == np.float32 and d32.shape == (count, k, channels), "SAVED_SELECTED_D32_PER_ORIGIN_DIMENSIONS")
                    for key in ("origins", "target", "observed", "raw_last"):
                        exact(predictions[key], values[key], "selected_space_same_original_" + key)
                    space = selected_real_space(values, d32)
                    # E is pointwise. Excluded incomplete positions receive a neutral placeholder only for finite CPU arithmetic.
                    future_teacher_input = np.where(complete_mask, values["target"], np.float32(0)).astype(np.float32)
                    future_y = encode_Y(future_teacher_input, state_encoder["center"], state_encoder["encoder"], backend)
                    last = predictions["latent_history"][:, -1]
                    ey32 = local_audit.local_forecast(values["raw_last"], future_y, last, d32)
                    ey, ey_real = score_real_and_FP32(ey32, values["raw_last"], future_y, last, d32, values, space, "E_Y_NONDEPLOYABLE")
                    output_arrays = {"target": values["target"].copy(), "observed": values["observed"].copy(), "raw_last": values["raw_last"].copy(),
                        "origins": values["origins"].copy(), "FIT_std": values["std"].copy(), "complete_target_positions_mask": space["complete"],
                        "selected_decoder_FP32": d32.copy(), "minimum_prediction_real64": space["prediction"], "optimal_latent_delta_real64": space["optimal_latent_delta_real64"],
                        "selected_row_projector_real64": space["projectors_real64"], "selected_D_singularvalues": space["singularvalues"],
                        "E_Y_latent_future_FP32": future_y, "E_Y_prediction_FP32": ey32, "E_Y_prediction_real64": ey_real, "cached_latent_last_FP32": last.copy(),
                        "RAW_prediction_FP32": values["raw_forecast"].copy()}
                    method_results = {}
                    expected_methods = {"BOLT"} if backend == "BOLT_NUMPY" else {"NATIVE_s92601", "NATIVE_s92602"}
                    require(set(entry["methods"]) == set(entry["cached_method_sources"]) == expected_methods, "EXACT_SELECTED_SPACE_BOLT_OR_TWO_NATIVE_METHODS")
                    for method, source in entry["cached_method_sources"].items():
                        cached = prior.verified_arrays(source, check)
                        exact(cached["origins"], values["origins"], "selected_space_actual_cached_origins:" + method)
                        exact(cached["latent_history"], predictions["latent_history"], "selected_space_native_same_past_before_D_reuse:" + method)
                        exact(predictions[method + "_cached_future"], cached["latent_future"], "selected_space_same_original_F_or_head_future:" + method)
                        actual32 = predictions[method + "_prediction"]
                        scored, real = score_real_and_FP32(actual32, values["raw_last"], cached["latent_future"], last, d32, values, space, method)
                        checks = prior.compare_scores(entry["methods"][method]["scores_complete_mask"], scored["scores_complete_C"], "selected_space_original_actual_completeC_score:" + method)
                        full = prior.scores(actual32, values["target"], values["observed"], values["std"])
                        checks += prior.compare_scores(entry["methods"][method]["scores_all_observed"], full, "selected_space_original_actual_fullscore:" + method)
                        method_results[method] = {**scored, "scores_all_observed": full}
                        output_arrays[method + "_actual_prediction_FP32"], output_arrays[method + "_real_trace_prediction_FP64"] = actual32.copy(), real
                        report["totals"]["score_field_checks"] += checks
                        report["totals"]["actual_predictions_checked"] += 1
                        report["totals"]["real_output_space_inequalities_checked"] += 1
                    artifact = save_npz(output / (dataset + "_" + route + "_" + split + "_" + backend + "_SELECTED_SPACE.npz"), output_arrays)
                    out_route["backends"][backend] = {"status": "PASS", "artifact": artifact, "source_input": original["inputs"][split],
                        "source_selected_state": entry["state_artifact"], "source_selected_forecasts": entry["artifact"], "source_fixed_encoder": original["projections"][route]["state_artifact"],
                        "cached_method_sources": entry["cached_method_sources"], "selected_rho": lock["selected_rho"], "selection_lock": lock_record,
                        "minimum_normalized_MSE_real_numeric_rowspace": space["minimum_normalized_MSE_real_numeric_rowspace"],
                        "minimum_solution_scores_complete_C": space["scores"], "minimum_only_normalized_MSE": True,
                        "rank_D_per_origin": space["rank_D_per_origin"], "certificates": space["certificates"], "numeric_rowspace_relative_singular_cutoff": RCOND,
                        "E_Y_diagnostic": {"role": "NONDEPLOYABLE_KNOWN_FUTURE_TARGET_DIAGNOSTIC_NOT_STRICT_MINIMUM", "numeric_encoder_backend": backend,
                            "incomplete_positions_zero_placeholder_only_never_scored": True, **ey},
                        "actual_methods": method_results, "RAW_scores_same_complete_C": raw_complete, "RAW_bound_inequality_applied": False,
                        "original_static_or_global_rank_K_lower_bound_reused": False, "future_target_used_for_new_D_or_rho_selection": False}
                    report["totals"]["selected_output_space_cells_checked"] += 1
                    report["totals"]["selected_DEV_backend_cells_checked"] += int(split == development)
                    report["totals"]["E_Y_diagnostic_predictions_checked"] += 1
                    report["totals"]["real_output_space_inequalities_checked"] += 1
                    report["totals"]["selected_D_origin_matrices_checked"] += count
    require(all(report["totals"][key] == value for key, value in {"selected_output_space_cells_checked": 24, "selected_DEV_backend_cells_checked": 8,
        "E_Y_diagnostic_predictions_checked": 24, "actual_predictions_checked": 36, "real_output_space_inequalities_checked": 60,
        "RAW_bound_inequalities_checked": 0, "score_field_checks": 864, "selected_D_origin_matrices_checked": 3264}.items()), "FULL_SELECTED_SHRINKAGE_OUTPUT_SPACE_DIAGNOSTIC_COVERAGE")
    for item in guarded:
        check()
        require(prior.file_sha(item["path"]) == item["sha256"], "ORIGINAL_SELECTED_SHRINKAGE_STATE_SOURCE_PREDICTION_AND_LOCK_UNCHANGED_AFTER_DIAGNOSTIC")
    report["status"] = "PASS"
    report["original_source_cache_selection_SHA_unchanged"] = True
    return report


def execute(result_path, audit_path, output_path):
    from continuation_runtime_v1 import Meter, write_new
    output = Path(output_path).resolve()
    require(output.is_relative_to(HERE) and not output.exists(), "FRESH_SELECTED_SHRINKAGE_OUTPUT_SPACE_DIRECTORY")
    output.mkdir(parents=True, exist_ok=False)
    meter = Meter("selected_shrinkage_space_" + output.name, gpu=False)
    source = {"path": str(Path(__file__).resolve()), "sha256": prior.file_sha(__file__)}
    failure = None
    try:
        meter.protected_check()
        initialize_numpy()
        sys.path.insert(0, str(ROOT / "research/smallk_failure_followup_20261014"))
        from prepare_inputs import install_protected_test_guard
        install_protected_test_guard()
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2):
            report = audit_result(result_path, audit_path, output, meter)
        meter.protected_check()
        require(prior.file_sha(__file__) == source["sha256"], "SELECTED_SPACE_DIAGNOSTIC_SOURCE_UNCHANGED")
        json.dumps(report, allow_nan=False)
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        report = {"status": "FAILED_PRESERVED", "failure": failure, "full_scope_PASS_claimed": False}
    finally:
        report["audit_source"] = source
        report["resource_terminal"] = meter.close(report["status"], failure)
        write_new(output / "RESULT.json", report)
    print(json.dumps({"status": report["status"], "output": str(output / "RESULT.json"), "totals": report.get("totals"), "failure": failure}, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raise SystemExit(execute(args.result, args.audit, args.output))
