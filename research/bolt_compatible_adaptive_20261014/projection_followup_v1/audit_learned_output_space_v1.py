"""Saved selected-state output geometry audit; no producer, F or learning replay."""
from pathlib import Path
import argparse
import json
import sys

import audit_causal_lag_objective_v2 as math_audit
import audit_linear_e_adaptation_v1 as learned

prior, require, HERE = math_audit.prior, math_audit.require, math_audit.HERE
DATASETS, SEEDS = learned.DATASETS, learned.SEEDS
LAGS = {"L0": (0,), "L7": (0, 1, 2, 4, 8, 16, 32)}
RESULT_SHA = "cf042cc764ff36cbb877211777721783efadbaab6ba8741268eaccba26cbb34b"
SOURCE_SHA = "08f5806c80b825c9510eaaad9eaf85c9fb0e9633668e3d33be0317162e50f795"
MATH_SHA = "022e9506fe0dbaa593f069e48972cb878a6a2f5cb2f0d062feb0cdde5cbeae30"
PARENTS = {"stop_gradient_diagnostic": "e27989fb23716be8bbf507442d44033884b28522d641a010d18b426a0eb6d10a", "stop_gradient_audit": "64f19658ac8bc8cee1d6fa3713113a7146ba282ebb8ad2b9eb49498d91e6418f", "full_gradient_parent": learned.TRAIN_SHA, "full_gradient_parent_audit": "9796b4ed53f74186bc2513db56fe321c2d450cdd9bffbb7c5f341766e73ee36d"}
np = sys.modules.get("numpy")


def initialize_numpy():
    global np
    math_audit.initialize_numpy(); learned.initialize_numpy(); np = sys.modules["numpy"]


def audit_cell(row, arrays, saved, state, raw, lags):
    learned.replay(saved, state, raw, lags); past, future = saved["latent_history"], saved["latent_future"]
    complete = raw["observed"].all(2); mask = prior.oracle_mask(raw["observed"]); n, h, c = raw["target"].shape; k = past.shape[-1]
    require(row["complete_target_positions"] == int(complete.sum()) and row["target_positions"] == complete.size and row["joint_unknown_future_coordinates"] == h*k and row["joint_output_coordinates_before_mask"] == h*c and row["lag_slope_parameters_per_C"] == len(lags)*k and row["lags"] == list(lags), "SAME_COMPLETE_C_MASK_ALL_HK_UNKNOWNS")
    for key in ("RAW_bound_applied", "new_F_forward", "all_real_latent_or_all_FP32_lower_bound_claimed", "old_initial_D_minimum_reused", "diagnostic_future_labels_used_for_new_fit_or_selection"): require(row[key] is False, "DIAGNOSTIC_SCOPE:" + key)
    require(row["minimum_only_normalized_MSE"] is row["selected_E_D_previously_supervised_by_FIT_future_targets"] is True and row["EY_incomplete_C_horizons_filled_with_current_C"] == int((~complete).sum()) and row["EY_fill_can_affect_later_L7_complete_outputs"] == (len(lags)>1) and row["EY_encoding_backend"].startswith("NumPy FP32 GEMM"), "EY_BACKEND_FILL_AND_SUPERVISED_PARENT_SCOPE")
    ey_input = np.where(complete[..., None], raw["target"], raw["raw_last"][:, None]).astype(np.float32)
    futures = {"ACTUAL_F": future, "EY": (ey_input-state["center"])@state["E"], "FLAT": np.ascontiguousarray(np.broadcast_to(past[:, -1:], future.shape))}
    persistence = np.ascontiguousarray(np.broadcast_to(raw["raw_last"][:, None], raw["target"].shape))
    context = {**{key: raw[key] for key in ("target", "observed", "raw_last", "origins", "mean", "std")}, "latent_history": past, "EY_input_C_FP32": ey_input, "complete_horizons": complete, "RAW_BOLT_prediction_FP32": raw["raw_forecast"], "current_C_persistence_FP32": persistence}
    for key, value in context.items(): learned.exact(arrays[key], value, "saved_selected_context:" + key)
    expected_keys = set(context) | {name+suffix for name in futures for suffix in ("_future_FP32", "_prediction_FP32", "_real_trace_FP64")}
    minimum = math_audit.joint_minimum(raw, past, state["D"], lags); fields = 0
    if complete.any():
        expected_keys |= {"joint_A_FP64", "joint_known_offset_FP64", "joint_minimum_prediction_FP64", "joint_best_future_FP64"} | {name+suffix for name in futures for suffix in ("_retained_trace_FP64", "_discarded_output_FP64")}
        require(row["status"] == "COMPLETE" and row["independent_reduced_LS_vs_SVD_verified"] is True, "RECORDED_COMPLETE_NEW_SELECTED_D_SPACE")
        meta = row["joint_metadata"]; require(meta["rcond"] == 1e-15 and meta["A_shape"] == [h*c, h*k] and meta["all_HK_unknowns_retained_before_output_mask"] is True and meta["C_times_complete_positions_denominator"] == int(complete.sum())*c, "REGISTERED_NUMERIC_RETAINED_SPACE_AND_DENOMINATOR")
        ranks = [0]*n
        for certificate in minimum["certificates"]:
            for index in certificate["window_indices"]: ranks[index] = certificate["numeric_rank"]
        require(meta["numeric_rank_per_origin"] == ranks and meta["complete_target_positions_per_origin"] == complete.sum(1).tolist(), "SAME_MASK_GROUP_RANKS_AND_COMPLETE_COUNTS")
        for key in ("MAE_or_physical_MSE_minimum_claimed", "strict_all_FP32_output_lower_bound_claimed", "strict_untruncated_real_rowspace_minimum_claimed", "all_real_latent_forecasts_satisfy_registered_bound_claimed"): require(meta[key] is False, "NOT_UNIVERSAL_LOWER_BOUND:" + key)
        prior.close(row["minimum_normalized_MSE"], minimum["minimum_normalized_MSE_real_joint_D32"], "new_selected_D_numeric_minimum", atol=1e-10, rtol=1e-10)
        for key, value in (("joint_A_FP64", minimum["joint_map"]), ("joint_known_offset_FP64", minimum["known_offset"]), ("joint_minimum_prediction_FP64", minimum["minimum_solution_prediction"]), ("joint_best_future_FP64", minimum["best_absolute_future_latent"])):
            require(arrays[key].dtype == np.float64, "SAVED_FP64_JOINT_GEOMETRY"); prior.close(arrays[key], value, "independent_joint_geometry:" + key, atol=1e-10, rtol=1e-10)
        fields += prior.compare_scores(row["minimum_solution_scores_complete_C"], minimum["minimum_solution_scores_complete_C"], "minimum_solution_not_MAE_or_physical_minimum")
    else: require(row["status"] == "NO_COMPLETE_C_TARGET_POSITIONS" and row["minimum_normalized_MSE"] is None and row["independent_reduced_LS_vs_SVD_verified"] is False, "NO_FABRICATED_COMPLETE_C_MINIMUM")
    require(set(arrays) == expected_keys and set(row["methods"]) == set(futures), "EXACT_CONTEXT_METHOD_AND_JOINT_ARTIFACT_MEMBERS")
    output = {"minimum_normalized_MSE": row["minimum_normalized_MSE"], "RAW_scores_complete_C": row["RAW_BOLT_scores_complete_mask"], "complete_target_positions": int(complete.sum()), "methods": {}}
    for name, f in futures.items():
        learned.exact(arrays[name+"_future_FP32"], f, "same_selected_method_future:" + name)
        predicted = math_audit.anchored_forecast(past, f, raw["raw_last"], state["D"], lags)
        if name == "ACTUAL_F": learned.exact(arrays[name+"_prediction_FP32"], saved["prediction"], "actual_F_saved_prediction_bytes"); prior.close(arrays[name+"_prediction_FP32"], predicted, "Torch_saved_vs_NumPy_replay")
        else: learned.exact(arrays[name+"_prediction_FP32"], predicted, "independent_NumPy_diagnostic_prediction:" + name)
        p = arrays[name+"_prediction_FP32"]; real = math_audit.anchored_forecast(past, f, raw["raw_last"], state["D"], lags, real=True)
        prior.close(arrays[name+"_real_trace_FP64"], real, "independent_real_trace:" + name, atol=1e-10, rtol=1e-10)
        recorded = row["methods"][name]
        require(recorded["role"] == {"ACTUAL_F": "ACTUAL_DEPLOYMENT_SAVED_FUTURE", "EY": "DIAGNOSTIC_FUTURE_Y_NOT_DEPLOYABLE", "FLAT": "PAST_ONLY_LATENT_PERSISTENCE"}[name], "METHOD_FUTURE_INFORMATION_ROLE")
        for key, used_mask in (("scores_all_observed", raw["observed"]), ("scores_complete_mask", mask)):
            _, count = math_audit.score_checked(recorded[key], p, raw, used_mask, "method:"+name+":"+key); fields += count
        if complete.any():
            trace = math_audit.diagnostic_trace(p, f, raw, past, state["D"], lags, minimum)
            for key, value in trace.items():
                if key.startswith("_"): continue
                if isinstance(value, dict): fields += prior.compare_scores(recorded[key], value, "retained_or_real_score:"+key)
                elif isinstance(value, bool): require(recorded[key] == value, "RETAINED_VS_REAL_FP32_BOOLEAN:"+key)
                else: prior.close(recorded[key], value, "numeric_trace_metadata:"+key, atol=1e-10, rtol=1e-10)
            require(np.isnan(arrays[name+"_retained_trace_FP64"][~mask]).all(), "UNOBSERVED_RETAINED_OUTPUT_ROWS_EXPLICIT_NAN")
            prior.close(arrays[name+"_retained_trace_FP64"][mask], trace["_retained_prediction_FP64"][mask], "independent_retained_output", atol=1e-10, rtol=1e-10)
            prior.close(arrays[name+"_discarded_output_FP64"], trace["_discarded_output_FP64"], "independent_discarded_numeric_output", atol=1e-10, rtol=1e-10)
        output["methods"][name] = {key: value for key, value in recorded.items() if key != "scores_all_observed"}
    for key, p, used_mask in (("RAW_BOLT_scores_all_observed", raw["raw_forecast"], raw["observed"]), ("RAW_BOLT_scores_complete_mask", raw["raw_forecast"], mask), ("current_C_persistence_scores_complete_mask", persistence, mask)):
        _, count = math_audit.score_checked(row[key], p, raw, used_mask, key); fields += count
    output["latent_flat_equals_current_C_persistence_bytes"] = arrays["FLAT_prediction_FP32"].tobytes() == persistence.tobytes()
    return output, fields


def metadata(path, check):
    require(isinstance(RESULT_SHA, str) and len(RESULT_SHA) == 64, "ACTUAL_LEARNED_OUTPUT_RESULT_NOT_YET_REGISTERED")
    binding = learned.file_record(path); require(binding["sha256"] == RESULT_SHA, "EXACT_NEW_SELECTED_OUTPUT_DIAGNOSTIC")
    result = prior.verified_json(binding, check); require(result["status"] in ("COMPLETE", "PARTIAL_COMPLETED"), "COMPLETED_OR_PRESERVED_PARTIAL_DIAGNOSTIC")
    for key, sha in PARENTS.items(): require(result[key]["sha256"] == sha, "EXACT_ACCEPTED_SELECTED_PARENT:"+key)
    return binding, result


def audit_result(path, meter):
    check = meter.check; binding, result = metadata(path, check); parents = {key: prior.verified_json(result[key], check) for key in PARENTS}
    for audit_key, source in (("stop_gradient_audit", "stop_gradient_diagnostic"), ("full_gradient_parent_audit", "full_gradient_parent")):
        require(parents[audit_key]["status"] == "PASS" and parents[audit_key]["diagnostic"] == result[source] and parents[source]["status"] == "COMPLETE", "PREVIOUS_TRAINING_REUSED_ONLY_AFTER_INDEPENDENT_PASS")
    require(all(result[key] == parents["full_gradient_parent"][key] == parents["stop_gradient_diagnostic"][key] for key in ("causal_lag_diagnostic", "causal_lag_audit")) and result["selection_registry"] == parents["stop_gradient_diagnostic"]["selection_registry"] and result["full_gradient_selection_registry"] == parents["full_gradient_parent"]["selection_registry"], "SAME_RAW_INPUT_AND_PREVIOUS_SELECTION_REGISTRIES")
    for key in ("new_E_D_fit", "CP_or_branch_reselection", "oracle_used_for_fit_or_selection", "protected_TEST_accessed", "independent_holdout", "strict_K_only", "new_F_state_hash_computed", "whole_deployment_cost_measured"): require(result[key] is False, "ZERO_NEW_LEARNING_OR_SELECTION_SCOPE:"+key)
    require(result["new_F_calls"] == result["new_training_updates"] == result["new_model_objects_loaded"] == 0, "NO_NEW_F_MODEL_OR_UPDATES")
    guarded = [binding, *[result[key] for key in PARENTS], result["plan"], *result["source_records"]]; actual_sources = {Path(r["path"]).name: r for r in result["source_records"]}
    require(actual_sources["run_learned_output_space_diagnostic_v1.py"]["sha256"] == SOURCE_SHA and prior.file_sha(math_audit.__file__) == MATH_SHA, "IMMUTABLE_DIAGNOSTIC_AND_INDEPENDENT_JOINT_HELPERS")
    for item in guarded:
        require(prior.file_sha(item["path"]) == item["sha256"], "SOURCE_OR_PARENT_SHA")
        if "snapshot_path" in item: require(prior.file_sha(item["snapshot_path"]) == item["sha256"], "EXECUTED_SOURCE_SNAPSHOT_SHA")
    science = prior.verified_json(result["causal_lag_diagnostic"], check); cache = {}
    def load(item):
        key = item["path"], item["sha256"]
        if key not in cache: cache[key] = prior.verified_arrays(item, check); guarded.append(item)
        return cache[key]
    keys = ("stop_gradient_diagnostic", "stop_gradient_audit", "full_gradient_parent", "full_gradient_parent_audit", "causal_lag_diagnostic", "causal_lag_audit", "selection_registry", "full_gradient_selection_registry", "plan")
    totals = {key: 0 for key in ("selected_states_checked", "diagnostic_cells_checked", "saved_actual_F_predictions_checked", "EY_predictions_checked", "latent_persistence_predictions_checked", "current_C_persistence_predictions_checked", "joint_maps_and_offsets_checked", "complete_C_numeric_minima_checked", "retained_trace_inequalities_checked", "original_real_trace_inequalities_evaluated", "FP32_with_measured_allowance_inequalities_evaluated", "RAW_bound_inequalities_checked", "score_fields_checked")}
    report = {"status": "PASS" if result["status"] == "COMPLETE" else "PARTIAL_AUDITED_NO_FULL_PASS", "diagnostic": binding, **{key: result[key] for key in keys}, "datasets": {}, "partial_paths": [], "totals": totals, "protected_TEST_accessed": False, "independent_F_or_training_replayed": False, "producer_functions_called_by_this_auditor": False, "scope": "Independent saved-state E/lag/D, selected lineage, complete-C scores, all-HK joint map and registered numeric retained-space LS/SVD/trace checks using immutable prior independent helpers. Previously accepted training audits reused; no new F/Jacobian/optimizer replay, independent holdout, MAE/physical minimum, or universal FP32/untruncated-real lower bound."}
    for data, (c, k, dev, ndev) in DATASETS.items():
        report["datasets"][data] = {"conditions": {}}
        owners = [("FULL_"+lag+"_s"+str(seed), parents["full_gradient_parent"]["datasets"][data]["conditions"][data+"_"+lag+"_E_LEARN_s"+str(seed)], lags) for lag, lags in LAGS.items() for seed in SEEDS] + [("STOP_L0_s"+str(seed), parents["stop_gradient_diagnostic"]["datasets"][data]["conditions"][str(seed)], LAGS["L0"]) for seed in SEEDS]
        require(set(result["datasets"][data]["conditions"]) == {name for name, _, _ in owners}, "EXACT_TWELVE_SELECTED_CONDITIONS")
        for name, chosen, lags in owners:
            check(); row = result["datasets"][data]["conditions"][name]
            if row["status"] != "COMPLETE": report["partial_paths"].append({"dataset": data, "condition": name, "status": row["status"]})
            if row["status"] not in ("COMPLETE", "PARTIAL_COMPLETED"): continue
            require(row["selected_state"] == chosen["selected_state"] and row["selected_update"] == chosen["selected_update"] and row["selection_lock"] == chosen["selection_lock"] and row["seed"] == chosen["seed"], "UNCHANGED_SELECTED_STATE_CP_AND_LOCK")
            state = load(chosen["selected_state"]); require(learned.state_hash(state) == chosen["selected_state_sha256"], "SELECTED_TENSOR_STATE_BYTES_HASH")
            selected_cp = next(cp for cp in chosen["checkpoints"] if cp["update"] == chosen["selected_update"]); require(selected_cp["state"] == chosen["selected_state"], "PREVIOUS_SELECTED_CP_STATE")
            out = report["datasets"][data]["conditions"][name] = {"selected_update": chosen["selected_update"], "selected_state": chosen["selected_state"], "splits": {}}; totals["selected_states_checked"] += 1
            for split, reference, n in (("tune", selected_cp, 64), (dev, chosen["DEV"], ndev)):
                cell = row["splits"][split]
                if cell["status"] not in ("COMPLETE", "NO_COMPLETE_C_TARGET_POSITIONS"): report["partial_paths"].append({"dataset": data, "condition": name, "split": split, "status": cell["status"]}); continue
                raw_source = science["datasets"][data]["spaces"]["PCA"]["conditions"]["L0_FORECAST"]["splits"][split]["input_source"]
                require(cell["input_source"] == raw_source and cell["selected_forecast_source"] == reference["artifact"], "EXACT_RAW_MASK_AND_SELECTED_FORECAST_LINEAGE")
                raw, saved = load(raw_source), load(reference["artifact"]); prior.check_raw_split(raw, channels=c, windows=n)
                out["splits"][split], fields = audit_cell(cell, load(cell["artifact"]), saved, state, raw, lags); totals["score_fields_checked"] += fields
                totals["diagnostic_cells_checked"] += 1
                for key in ("saved_actual_F_predictions_checked", "EY_predictions_checked", "latent_persistence_predictions_checked", "current_C_persistence_predictions_checked"): totals[key] += 1
                if raw["observed"].all(2).any():
                    totals["joint_maps_and_offsets_checked"] += 1; totals["complete_C_numeric_minima_checked"] += 1
                    for key in ("retained_trace_inequalities_checked", "original_real_trace_inequalities_evaluated", "FP32_with_measured_allowance_inequalities_evaluated"): totals[key] += 3
    for item in guarded: check(); require(prior.file_sha(item["path"]) == item["sha256"], "POST_AUDIT_SOURCE_INPUT_OR_SELECTED_BYTES_UNCHANGED")
    totals["protected_source_and_input_file_hashes_checked"] = len(guarded)
    if result["status"] == "COMPLETE": require(totals["selected_states_checked"] == result["selected_states_diagnosed"] == 12 and totals["diagnostic_cells_checked"] == result["completed_diagnostic_cells"] == result["expected_diagnostic_cells"] == 24 and not result["failed_paths"] and not report["partial_paths"], "TWELVE_STATES_TWENTY_FOUR_COMPLETE_CELLS")
    return report


def execute(path, output):
    from continuation_runtime_v1 import Meter, write_new
    output = Path(output).resolve(); require(not output.exists(), "FRESH_LEARNED_OUTPUT_AUDIT_OUTPUT")
    meter = Meter("audit_learned_output_"+output.stem, gpu=False); source = learned.file_record(__file__); failure = None
    try:
        meter.protected_check(); metadata(path, meter.check); initialize_numpy(); sys.path.insert(0, str(learned.ROOT/"research/smallk_failure_followup_20261014"))
        from prepare_inputs import install_protected_test_guard
        from threadpoolctl import threadpool_limits
        install_protected_test_guard()
        with threadpool_limits(limits=2): report = audit_result(path, meter)
        meter.protected_check(); require(prior.file_sha(__file__) == source["sha256"], "AUDIT_SOURCE_UNCHANGED"); json.dumps(report, allow_nan=False)
    except Exception as error: failure = {"type": type(error).__name__, "message": str(error)}; report = {"status": "FAILED_PRESERVED", "failure": failure}
    finally: report["audit_source"] = source; report["resource"] = meter.close(report["status"], failure); write_new(output, report)
    print(json.dumps({"status": report["status"], "totals": report.get("totals"), "failure": failure}), flush=True); return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--result", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); raise SystemExit(execute(args.result, args.output))
