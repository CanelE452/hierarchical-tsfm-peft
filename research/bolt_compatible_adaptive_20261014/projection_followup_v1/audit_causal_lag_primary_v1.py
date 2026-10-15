"""Independent primary cached predictions; joint-SVD diagnostics remain unresolved."""
from pathlib import Path
import argparse
import json
import sys

import audit_causal_lag_objective_v1 as base

HERE, ROOT = base.HERE, base.ROOT
DATASETS, SPACES, CONDITIONS, PARENT_SHA = base.DATASETS, base.SPACES, base.CONDITIONS, base.PARENT_SHA
prior, raw_audit, require, exact = base.prior, base.raw_audit, base.require, base.exact
load_gates, check_state, source_contract, score_checked = base.load_gates, base.check_state, base.source_contract, base.score_checked
anchored_forecast = base.anchored_forecast
np = sys.modules.get("numpy")
SUCCESS = "PASS_PRIMARY_PREDICTIONS_WITH_JOINT_DIAGNOSTICS_UNRESOLVED"


def initialize_numpy():
    global np
    base.initialize_numpy(); np = sys.modules["numpy"]


def audit_split(entry, arrays, values, latent, future, encoder, decoder, lags):
    require(entry["status"] in ("COMPLETE", "COMPLETE_PREDICTION_WITH_DIAGNOSTIC_FAILURES") and entry["windows"] == len(latent), "PRIMARY_CACHED_PREDICTION_EXISTS")
    complete = values["observed"].all(axis=2); mask = prior.oracle_mask(values["observed"])
    require(entry["complete_target_positions"] == int(complete.sum()) and entry["target_positions"] == complete.size and entry["EY_incomplete_C_horizons_filled_with_currentC"] == int((~complete).sum()), "EXACT_PRIMARY_MASK_AND_EY_FILL_CONTRACT")
    sources = {"target": values["target"], "observed": values["observed"], "currentC": values["raw_last"], "origins": values["origins"], "FIT_mean": values["mean"], "FIT_std": values["std"], "complete_horizons": complete, "latent_history": latent, "actual_F_future": future}
    flat = np.ascontiguousarray(np.broadcast_to(latent[:, -1:], future.shape))
    ey_inputs = np.where(complete[..., None], values["target"], values["raw_last"][:, None]).astype(np.float32)
    ey = raw_audit.encode(ey_inputs, encoder, "BOLT_NUMPY")
    sources.update(flat_future=flat, EY_future=ey)
    for name, value in sources.items(): exact(arrays[name], value, "primary_same_source:" + name)
    predictions = {"ACTUAL_F": future, "FLAT": flat, "EY": ey}
    output = {"methods": {}, "current_C_persistence": {}, "RAW_BOLT": {}, "joint_diagnostics_independently_accepted": False, "joint_unresolved_reason": "Numerical rank/retained-space sensitivity is outside this primary prediction audit"}
    score_fields = 0
    for method, latent_future in predictions.items():
        predicted = anchored_forecast(latent, latent_future, values["raw_last"], decoder, lags)
        real = anchored_forecast(latent, latent_future, values["raw_last"], decoder, lags, real=True)
        exact(arrays[method + "_prediction_FP32"], predicted, "primary_independent_FP32_equations:" + method)
        prior.close(arrays[method + "_real_trace_FP64"], real, "primary_independent_real_trace_equations:" + method, atol=1e-10, rtol=1e-10)
        source = entry["diagnostics"]["methods"][method]; checked = output["methods"][method] = {}
        for key, p, observed in (("scores_all_observed", predicted, values["observed"]), ("scores_complete_mask", predicted, mask), ("real_trace_scores_complete_mask", real, mask)):
            checked[key], count = score_checked(source[key], p, values, observed, "primary:" + method + ":" + key); score_fields += count
    require(entry["scores_all_observed"] == entry["diagnostics"]["methods"]["ACTUAL_F"]["scores_all_observed"] and entry["scores_complete_mask"] == entry["diagnostics"]["methods"]["ACTUAL_F"]["scores_complete_mask"], "ACTUAL_F_PRIMARY_TOP_SCORE_ALIAS")
    for method, p in (("current_C_persistence", np.broadcast_to(values["raw_last"][:, None], values["target"].shape)), ("RAW_BOLT", values["raw_forecast"])):
        for key, observed in (("scores_all_observed", values["observed"]), ("scores_complete_mask", mask)):
            output[method][key], count = score_checked(entry[method][key], p, values, observed, "primary:" + method + ":" + key); score_fields += count
    require(entry["RAW_BOLT"]["bound_inequality_applied"] is False, "NO_ORACLE_BOUND_ON_RAW")
    perturbation = entry["cached_future_label_perturbation"]
    require(perturbation["status"] == "PASS" and perturbation["windows"] == min(2, len(latent)) and perturbation["target_and_observed_mask_perturbed"] is perturbation["prediction_byte_exact"] is perturbation["fit_state_held_fixed"] is True and perturbation["E_F_forward_replayed"] is False, "FIXED_CACHE_PERTURBATION_SCOPE_ONLY")
    altered = {**values, "target": np.full_like(values["target"], np.float32(1e6)), "observed": np.zeros_like(values["observed"])}
    n = perturbation["windows"]
    before = anchored_forecast(latent[:n], future[:n], values["raw_last"][:n], decoder, lags)
    after = anchored_forecast(latent[:n], future[:n], altered["raw_last"][:n], decoder, lags)
    exact(before, after, "primary_no_future_labels_in_fixed_cached_forward")
    return output, score_fields


def audit_result(result_path, meter, *, expected_windows=None):
    check = meter.check
    binding, result, parents = load_gates(result_path, check)
    if not parents: return {"status": "NOT_AUDITED_INCOMPLETE_PRODUCER", "diagnostic": binding, "producer_status": result.get("status"), "scientific_assets_opened": False}
    producer_complete = result["status"] == "COMPLETE"
    source = parents["raw_subset_diagnostic"]
    files = [binding, *[result[key] for key in PARENT_SHA], *result["source_records"], *result["source_snapshots"], *result["protected_prior_records"], *result["input_records"], result["plan"]]
    for record in files: check(); require(prior.file_sha(record["path"]) == record["sha256"], "SEALED_SOURCE_CACHE_MODEL_LOCK_AND_PLAN_SHA")
    originals = {Path(record["path"]).name: record for record in result["source_records"]}
    for record in result["source_snapshots"]: require(record["sha256"] == originals[Path(record["path"]).name]["sha256"], "EXECUTED_SOURCE_SNAPSHOT_EXACT_BYTES")
    src_contract = source_contract(result["source_records"])
    registry = prior.verified_json(result["fit_registry"], check); files.append(result["fit_registry"])
    keys = {(data, space, condition) for data in DATASETS for space in SPACES for condition in CONDITIONS}
    locks = {}
    require(registry["status"] == "ALL_24_RESOLVED_BEFORE_TUNE_DEV" and registry["TUNE_DEV_NPZ_opened_before_seal"] is False and result["all_24_locks_before_TUNE_DEV_NPZ"] is True, "ALL_FIXED_D_LOCKS_AND_REGISTRY_BEFORE_TUNE_DEV")
    for row in registry["conditions"]:
        key = row["dataset"], row["space"], row["condition"]
        require(key not in locks and row["fit_lock"] == result["datasets"][key[0]]["spaces"][key[1]]["conditions"][key[2]]["fit_lock"], "UNIQUE_EXACT_FIT_LOCK_BINDING")
        lock = prior.verified_json(row["fit_lock"], check); files.append(row["fit_lock"]); locks[key] = lock
        require(lock["status"] in ("LOCKED_FIT_ONLY", "FAILED_PATH_PRESERVED") and lock["sealed_unix"] <= registry["sealed_unix"], "EACH_FIT_PATH_RESOLVED_BEFORE_REGISTRY_SEAL")
    require(set(locks) == keys and result["resolved_fit_lock_files"] == len(keys), "ALL_NOMINAL_TWENTYFOUR_FIT_PATHS_RESOLVED")
    sealed = [i for i, event in enumerate(result["execution_events"]) if event["event"] == "ALL_24_FIT_LOCKS_SEALED"]
    require(len(sealed) == 1 and result["execution_events"][sealed[0]]["registry"] == result["fit_registry"], "EXACT_ONE_REGISTRY_EVENT")
    events = [event for event in result["execution_events"] if event["event"] == "CONTEXT_OPENED"]
    require(all(i > sealed[0] and event["all_24_locks_already_sealed"] is True for i, event in enumerate(result["execution_events"]) if event["event"] == "CONTEXT_OPENED" and event["split"] != "fit"), "EVERY_TUNE_DEV_CONTEXT_OPEN_AFTER_ALL_FIT_LOCKS")
    totals = {"decoder_states_checked": 0, "fit_lock_files_checked": len(locks), "FIT_source_contexts_checked": 0, "cached_Bolt_source_contexts_checked": 0, "cached_D_split_prediction_arrays_checked": 0, "independent_FP32_method_prediction_arrays_checked": 0, "real_trace_prediction_arrays_checked": 0, "joint_output_diagnostic_cells_checked": 0, "EY_prediction_arrays_checked": 0, "flat_prediction_arrays_checked": 0, "retained_trace_inequalities_checked": 0, "original_real_trace_inequalities_evaluated": 0, "FP32_with_allowance_inequalities_evaluated": 0, "RAW_bound_inequalities_checked": 0, "cached_future_label_perturbation_checks": 0, "independent_fit_response_columns_checked": 0, "score_field_checks": 0, "protected_prior_file_hashes_checked": len(result["protected_prior_records"])}
    report = {"status": "PASS_PRIMARY_PREDICTIONS_WITH_JOINT_DIAGNOSTICS_UNRESOLVED" if producer_complete else "PARTIAL_PRIMARY_AUDIT_NO_FULL_PASS", "diagnostic": binding, **{key: result[key] for key in PARENT_SHA}, "plan": result["plan"], "fit_registry": result["fit_registry"], "source_contract": src_contract, "totals": totals, "datasets": {}, "producer_failed_paths": result["failed_paths"], "audit_failed_paths": [], "partial_conditions_not_accepted": [], "protected_TEST_accessed": False, "new_F_native_or_encoder_training": False, "actual_F_forward_independently_replayed": False, "TUNE_DEV_or_oracle_used_for_new_fit_selection": False, "all_real_latent_forecasts_bound_claimed": False, "strict_all_FP32_bound_claimed": False, "MAE_or_physical_MSE_minimum_claimed": False, "strict_K_only_compression": False, "independent_generalization_claimed": False, "whole_deployment_cost_measured": False}
    for dataset, (c, k, dev, dev_n) in DATASETS.items():
        row = result["datasets"][dataset]; parent = source["datasets"][dataset]
        require(row["channels"] == c and row["K"] == k and row["development_split"] == dev, "FIXED_C_K_AND_EXPOSED_DEVELOPMENT_SPLIT")
        out = report["datasets"][dataset] = {"spaces": {}}
        for space in SPACES:
            entry, original = row["spaces"][space], parent["spaces"][space]
            contexts = {}; output = out["spaces"][space] = {"conditions": {}}
            require(entry["E_source"] == original["E_state_artifact"] and entry["FIT_input_source"] == original["inputs"]["fit"] and entry["FIT_Bolt_cache"] == original["Bolt"]["fit"]["artifact"], "FIXED_E_FIT_INPUT_AND_ACTUAL_F_CACHE")
            encoder = prior.verified_arrays(entry["E_source"], check)
            for split, count in (("fit", 256), ("tune", 64), (dev, dev_n)):
                if expected_windows is not None: count = expected_windows[split]
                values = prior.verified_arrays(original["inputs"][split], check); prior.check_raw_split(values, channels=c, windows=count)
                cached = prior.verified_arrays(original["Bolt"][split]["artifact"], check)
                latent = raw_audit.encode(values["history"], encoder, "BOLT_NUMPY")
                exact(latent, values["latent_history_BOLT_NUMPY"], "fixed_numpy_E_input_cache_lineage")
                exact(latent, cached["latent_history"], "actual_F_cache_encoded_past")
                exact(values["origins"], cached["origins"], "actual_F_cache_origin_lineage")
                require(cached["latent_future"].dtype == np.float32 and cached["latent_future"].shape == (count, 48, k) and np.isfinite(cached["latent_future"]).all(), "FINITE_ACTUAL_SAVED_K_STREAM_F_FORECAST")
                contexts[split] = (values, latent, cached["latent_future"])
                if split != "fit":
                    for name in ("mean", "std"): exact(values[name], contexts["fit"][0][name], "fixed_FIT_scaler:" + name)
                totals["cached_Bolt_source_contexts_checked"] += 1
                totals["FIT_source_contexts_checked"] += int(split == "fit")
            for condition, (lags, objective) in CONDITIONS.items():
                saved = entry["conditions"][condition]; lock = locks[(dataset, space, condition)]
                if saved["status"] != "LOCKED_FIT_ONLY":
                    require(not producer_complete, "COMPLETE_PRODUCER_WITH_FAILED_FIT")
                    report["partial_conditions_not_accepted"].append({"dataset": dataset, "space": space, "condition": condition, "metadata": saved}); continue
                try:
                    require({key: value for key, value in saved.items() if key not in ("splits", "fit_lock")} == {key: value for key, value in lock.items() if key != "sealed_unix"}, "FIT_LOCK_EXACT_STATE_METADATA_AND_OBJECTIVE")
                    require(saved["FIT_future_targets_used"] == (objective == "FORECAST") and saved["TUNE_DEV_used_for_fit_or_selection"] is saved["oracle_used_for_fit_or_selection"] is False, "FIT_ONLY_SUPERVISION_SEPARATE_FROM_INFERENCE")
                    state = prior.verified_arrays(saved["state_artifact"], check)
                    fitted = check_state(state, saved["fit_metadata"], *contexts["fit"], lags, objective)
                    checked = output["conditions"][condition] = {"state": saved["state_artifact"], "fit": fitted, "splits": {}}
                    totals["decoder_states_checked"] += 1; totals["independent_fit_response_columns_checked"] += c
                    for split, (values, latent, future) in contexts.items():
                        summary = saved["splits"][split]
                        if summary["status"] != "COMPLETE":
                            require(not producer_complete, "COMPLETE_PRODUCER_WITH_FAILED_SPLIT"); report["partial_conditions_not_accepted"].append(summary); continue
                        require(summary["input_source"] == original["inputs"][split] and summary["cached_Bolt_source"] == original["Bolt"][split]["artifact"] and summary["state_source"] == saved["state_artifact"] and summary["fit_lock"] == saved["fit_lock"], "CACHED_SPLIT_INPUT_F_STATE_AND_FIT_LOCK_BINDINGS")
                        arrays = prior.verified_arrays(summary["artifact"], check)
                        checked["splits"][split], count = audit_split(summary, arrays, values, latent, future, encoder, state["decoder_FP32"], lags)
                        totals["score_field_checks"] += count; totals["cached_D_split_prediction_arrays_checked"] += 1
                        for key in ("EY_prediction_arrays_checked", "flat_prediction_arrays_checked", "cached_future_label_perturbation_checks"): totals[key] += 1
                        for key in ("independent_FP32_method_prediction_arrays_checked", "real_trace_prediction_arrays_checked"): totals[key] += 3

                except Exception as error:
                    report["audit_failed_paths"].append({"dataset": dataset, "space": space, "condition": condition, "failure": {"type": type(error).__name__, "message": str(error)}})
    receipt = prior.verified_json(result["resource_terminal"], check)
    require(receipt["status"] == result["status"] and receipt["elapsed_or_cumulative_budget_caps"] is None and result["source_cache_model_selection_SHA_unchanged"] is True, "ACTUAL_CPU_EXECUTION_RECEIPT_AND_OLD_ASSET_PRESERVATION")
    for record in files: check(); require(prior.file_sha(record["path"]) == record["sha256"], "POST_AUDIT_OLD_SOURCE_MODEL_CACHE_AND_LOCKS_UNCHANGED")
    if report["audit_failed_paths"]: report["status"] = "FAILED_PARTIAL_AUDIT_PRESERVED_NO_FULL_PASS"
    elif producer_complete:
        nominal_states = len(keys); nominal_splits = nominal_states * 3
        require(totals["decoder_states_checked"] == result["decoder_fits_completed"] == nominal_states and totals["cached_D_split_prediction_arrays_checked"] == result["cached_D_predictions_completed"] == nominal_splits and totals["cached_Bolt_source_contexts_checked"] == result["actual_F_cache_splits_reused"] == len(DATASETS) * len(SPACES) * 3 and not result["failed_paths"] and not report["partial_conditions_not_accepted"], "FULL_REGISTERED_CAUSAL_LAG_COHORT_COUNTS")
        require(totals["EY_prediction_arrays_checked"] == result["EY_diagnostics_completed"] and totals["flat_prediction_arrays_checked"] == result["flat_diagnostics_completed"], "ACTUAL_EY_AND_FLAT_PREDICTION_COUNTS")
    return report
