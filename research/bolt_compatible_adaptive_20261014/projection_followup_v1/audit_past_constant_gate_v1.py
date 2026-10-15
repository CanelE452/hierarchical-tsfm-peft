"""Independent exact-flat past output gate equations; no producer/F imports."""
from pathlib import Path
import argparse
import json
import sys

import audit_projection_results_v1 as prior
import audit_decoder_metric_probe_v1 as decoder_audit

np = sys.modules.get("numpy")
HERE, ROOT = prior.HERE, prior.ROOT
require, exact = prior.require, decoder_audit.byte_exact
DATASETS = {"robin": (17, 4, "val", 30), "beijing_aotizhongxin": (11, 2, "eval", 146)}
METHODS = ("RAW_BOLT", "PCA_PINV_BOLT", "PCA_PINV_NATIVE_s92601", "PCA_PINV_NATIVE_s92602",
           "FIT_TARGET_PINV_BOLT", "FIT_TARGET_PINV_NATIVE_s92601", "FIT_TARGET_PINV_NATIVE_s92602")


def initialize_numpy():
    global np
    decoder_audit.initialize_numpy()
    np = sys.modules["numpy"]


def force_current_from_past(history, history_observed):
    require(history.dtype == np.float32 and history.ndim == 3 and history.shape[1] == 512 and len(history) > 0 and history.shape[-1] > 0
            and history_observed.dtype == np.bool_ and history_observed.shape == history.shape
            and np.isfinite(history).all(), "FINITE_FP32_512_PAST_AND_ORIGINAL_OBSERVED_MASK")
    force = np.zeros((len(history), history.shape[-1]), bool)
    counts = np.zeros(force.shape, np.int64)
    for window in range(len(history)):
        for channel in range(history.shape[-1]):
            counts[window, channel] = int(history_observed[window, :, channel].sum())
            if counts[window, channel] < 2:
                continue
            constant = True
            current = history[window, -1, channel]
            for position in range(512):
                if history[window, position, channel] != current:
                    constant = False
                    break
            force[window, channel] = constant
    return force, counts


def independently_gate_prediction(prediction, current, force_current):
    require(prediction.dtype == current.dtype == np.float32 and prediction.ndim == 3
            and prediction.shape[1] == 48 and current.shape == force_current.shape == (len(prediction), prediction.shape[-1])
            and force_current.dtype == np.bool_ and np.isfinite(prediction).all() and np.isfinite(current).all(),
            "FINITE_FP32_PREDICTION_CURRENT_AND_BOOL_PAST_GATE")
    gated = np.ascontiguousarray(prediction).copy()
    for window in range(len(gated)):
        for channel in range(gated.shape[-1]):
            if force_current[window, channel]:
                gated[window, :, channel] = current[window, channel]
    unchanged = np.broadcast_to(~force_current[:, None], prediction.shape)
    fixed = np.broadcast_to(force_current[:, None], prediction.shape)
    exact(gated[unchanged], prediction[unchanged], "unchanged_forecast_coordinates")
    exact(gated[fixed], np.broadcast_to(current[:, None], prediction.shape)[fixed], "forced_current_C_coordinates")
    return gated


def future_diagnostic_counts(target, observed, current, force_current):
    require(target.dtype == current.dtype == np.float32 and target.ndim == 3 and target.shape[1] == 48
            and observed.dtype == np.bool_ and observed.shape == target.shape
            and current.shape == force_current.shape == (len(target), target.shape[-1]) and force_current.dtype == np.bool_
            and np.isfinite(target[observed]).all(), "OBSERVED_FINITE_FUTURE_DIAGNOSTIC_ONLY")
    by_channel = []
    for channel in range(target.shape[-1]):
        windows = np.flatnonzero(force_current[:, channel])
        observed_count = changed_count = eligible_windows = changed_windows = all_observed_windows = 0
        for window in windows:
            rows = observed[window, :, channel]
            count = int(rows.sum())
            changes = int(np.count_nonzero(target[window, :, channel][rows] != current[window, channel]))
            observed_count += count
            changed_count += changes
            eligible_windows += int(count > 0)
            changed_windows += int(changes > 0)
            all_observed_windows += int(count == 48)
        by_channel.append({"gated_window_channel_pairs": len(windows), "gated_future_observed_positions": observed_count,
                           "gated_future_changed_positions": changed_count,
                           "gated_future_equal_current_positions": observed_count - changed_count,
                           "gated_windows_with_observed_future": eligible_windows,
                           "gated_windows_with_any_future_change": changed_windows,
                           "gated_windows_with_all_48_future_observed": all_observed_windows})
    return {"role": "DIAGNOSTIC_ONLY_NOT_GATE_INPUT", "per_channel": by_channel,
            "totals": {key: int(sum(channel[key] for channel in by_channel)) for key in by_channel[0]},
            "constant_past_guarantees_constant_future_claimed": False}


def gated_score_pair(original, gated, values, force):
    exact(gated, independently_gate_prediction(original, values["raw_last"], force), "independent_gate_output")
    full_before = prior.scores(original, values["target"], values["observed"], values["std"])
    full_after = prior.scores(gated, values["target"], values["observed"], values["std"])
    complete_mask = prior.oracle_mask(values["observed"])
    if complete_mask.any():
        complete_before = prior.scores(original, values["target"], complete_mask, values["std"])
        complete_after = prior.scores(gated, values["target"], complete_mask, values["std"])
    else:
        complete_before = complete_after = None
    changes = {scope: {unit: {field: after[unit][field] - before[unit][field] for field in ("mse", "mae")}
                       for unit in ("normalized", "physical_per_channel_diagnostic")}
               for scope, before, after in (("all_observed", full_before, full_after),
                                            ("complete_C", complete_before, complete_after)) if before is not None}
    return {"all_observed": {"original": full_before, "gated": full_after},
            "complete_C": {"original": complete_before, "gated": complete_after}, "gated_minus_original": changes,
            "fixed_static_D_error_floor_applied_to_gated_forecast": False}


def audit_case(summary, arrays, values, predictions):
    force, counts = force_current_from_past(values["history"], values["history_observed"])
    flat = np.ones(force.shape, bool)
    for window in range(len(flat)):
        for channel in range(flat.shape[1]):
            flat[window, channel] = all(values["history"][window, position, channel] == values["raw_last"][window, channel]
                                        for position in range(512))
    expected_keys = {"keep_forecast", "exact_flat", "true_observed_counts", "gate_eligible", "target", "observed", "raw_last", "origins",
                     "FIT_std", "FIT_mean", "complete_target_positions_mask"}
    expected_keys |= {method + suffix for method in METHODS for suffix in ("_ungated_prediction", "_gated_prediction")}
    require(set(arrays) == expected_keys and set(summary["methods"]) == set(predictions) == set(METHODS), "EXACT_SEVEN_METHODS_AND_SAVED_GATE_OUTPUT_KEYS")
    for key in ("target", "observed", "raw_last", "origins"):
        exact(arrays[key], values[key], "gate_source_input:" + key)
    exact(arrays["FIT_std"], values["std"], "gate_original_FIT_std")
    exact(arrays["FIT_mean"], values["mean"], "gate_original_FIT_mean")
    exact(arrays["keep_forecast"], ~force, "independent_keep_forecast")
    exact(arrays["gate_eligible"], force, "independent_gate_eligible")
    exact(arrays["exact_flat"], flat, "independent_past_exact_flat")
    exact(arrays["true_observed_counts"], counts, "original_past_true_observed_counts")
    complete = values["observed"].all(axis=-1)
    exact(arrays["complete_target_positions_mask"], complete, "gate_same_complete_C_positions")
    require(summary["status"] == "COMPLETE" and summary["windows"] == len(force) and summary["channels"] == force.shape[1]
            and summary["exact_flat_windows_per_channel"] == flat.sum(axis=0).tolist()
            and summary["gate_eligible_windows_per_channel"] == force.sum(axis=0).tolist()
            and summary["complete_target_positions"] == int(complete.sum())
            and summary["total_target_positions"] == int(complete.size), "GATE_PAST_COUNTS_DIMENSIONS_AND_COMPLETE_POSITION_COUNTS")
    diagnostic = future_diagnostic_counts(values["target"], values["observed"], values["raw_last"], force)
    for recorded, independent in (
            ("eligible_windows_with_any_observed_future_per_channel", "gated_windows_with_observed_future"),
            ("eligible_windows_with_observed_future_change_per_channel", "gated_windows_with_any_future_change"),
            ("eligible_observed_future_positions_per_channel", "gated_future_observed_positions"),
            ("eligible_observed_future_changed_positions_per_channel", "gated_future_changed_positions")):
        require(summary[recorded] == [channel[independent] for channel in diagnostic["per_channel"]], "GATE_FUTURE_DIAGNOSTIC_COUNTS_ONLY:" + recorded)
    methods, checks = {}, 0
    for method in METHODS:
        original, gated = arrays[method + "_ungated_prediction"], arrays[method + "_gated_prediction"]
        exact(original, predictions[method], "gate_accepted_ungated_prediction:" + method)
        pair = gated_score_pair(original, gated, values, force)
        entry = summary["methods"][method]
        for recorded, actual in (("ungated_scores_all_observed", pair["all_observed"]["original"]),
                                ("gated_scores_all_observed", pair["all_observed"]["gated"]),
                                ("ungated_scores_complete_mask", pair["complete_C"]["original"]),
                                ("gated_scores_complete_mask", pair["complete_C"]["gated"])):
            if actual is None:
                require(entry[recorded]["status"] == "NO_OBSERVED_TARGETS", "GATE_NO_COMPLETE_C_SCORE_WHEN_EMPTY")
            else:
                checks += prior.compare_scores(entry[recorded], actual, "past_constant_gate:" + method + ":" + recorded)
        require(entry["unchanged_coordinates_byte_exact"] is entry["gated_coordinates_current_anchor_byte_exact"] is True
                and entry["native_head_frozen_intervention"] == ("NATIVE" in method)
                and entry["optimized_gated_native_head_claimed"] is False, "GATE_BYTE_PARITY_AND_FIXED_HEAD_INTERVENTION_SCOPE")
        changed = np.ascontiguousarray(original).view(np.uint32) != np.ascontiguousarray(gated).view(np.uint32)
        require(entry["changed_prediction_coordinates_per_channel"] == changed.sum(axis=(0, 1)).tolist(), "GATE_CHANGED_PREDICTION_BYTE_COUNTS")
        methods[method] = pair
    return {"windows": len(force), "channels": force.shape[1], "gate_eligible_windows_per_channel": force.sum(axis=0).tolist(),
            "exact_flat_windows_per_channel": flat.sum(axis=0).tolist(), "complete_target_positions": int(complete.sum()),
            "future_change_diagnostic": diagnostic, "methods": methods, "score_field_checks": checks,
            "gate_uses_past_only": True, "future_flatness_guaranteed": False,
            "original_static_D_floor_applied_to_gated_predictions": False}


def audit_result(gate_path, meter):
    check = meter.check
    path = Path(gate_path).resolve()
    record = {"path": str(path), "sha256": prior.file_sha(path)}
    gate = prior.verified_json(record, check)
    if gate.get("status") != "COMPLETE":
        return {"status": "NOT_AUDITED_INCOMPLETE_PRODUCER", "diagnostic": record,
                "producer_status": gate.get("status"), "scientific_assets_opened": False}
    require(gate["completed_dataset_split_cells"] == 6 and gate["completed_method_pairs"] == 42 and gate["failed_paths"] == []
            and gate["cached_CPU_output_gate_inferences"] == 42 and gate["E_F_native_head_forward_runs"] == 0
            and gate["new_training"] is gate["new_encoder_forecaster_or_head_forward"] is gate["protected_TEST_accessed"] is False
            and gate["whole_deployment_cost_measured"] is gate["independent_holdout"] is gate["strict_K_only_compression"] is False
            and gate["same_decoder_side_information_as_ungated"] is gate["static_original_fixed_D_floor_reused_as_gated_lower_bound"] is False
            and gate["method_selection_used"] is gate["future_labels_used_for_gate"] is False
            and gate["past_C_condition_metadata_added"] is True, "COMPLETE_CACHED_CPU_GATE_NO_FIT_F_FORWARD_COST_OR_SELECTION")
    producer = prior.verified_json(gate["producer"], check)
    fit_audit = prior.verified_json(gate["independent_FIT_target_audit"], check)
    require(producer["status"] == "COMPLETE" and fit_audit["status"] == "PASS" and fit_audit["producer"] == gate["producer"]
            and Path(fit_audit["audit_source"]["path"]).name == "audit_fit_target_comparison_v2.py"
            and prior.file_sha(fit_audit["audit_source"]["path"]) == fit_audit["audit_source"]["sha256"], "EXACT_ACCEPTED_FIT_TARGET_V2_PASS_BINDING")
    require(all(fit_audit["totals"].get(key) == expected for key, expected in {
        "projection_states_checked": 4, "native_fits_checked": 8, "checkpoint_states_checked": 40,
        "score_field_checks": 1464, "saved_prediction_arrays_checked": 88, "cost_cells_checked": 20,
        "recorded_timer_samples_checked": 120, "target_projection_fits_checked": 2,
        "reused_PCA_controls_checked": 2, "output_subspace_diagnostic_cells_checked": 12}.items()), "FULL_FIT_TARGET_REFERENCE_AUDIT_COUNTS")
    scope = prior.verified_json(gate["scope"], check)
    require(scope["source_producer"] == gate["producer"] and scope["source_audit"] == gate["independent_FIT_target_audit"]
            and scope["source_audit_code"] == fit_audit["audit_source"]
            and scope["decoder_input_interface"] == "K streams + current C + past-constant flags (C bits)"
            and scope["status"] == "CONTINUING_GOAL_SCOPE_JUDGMENT" and scope["individual_exact_human_method_approval_claimed"] is False
            and scope["past_C_condition_metadata_added"] is True and scope["same_decoder_side_information_as_ungated"] is False
            and scope["representation_learning_improvement_claimed"] is scope["threshold_tuning_or_fit"] is False
            and scope["whole_deployment_cost_measured"] is scope["original_static_fixed_D_floor_is_gated_lower_bound"] is False,
            "FRESH_SCOPE_EXTRA_C_PAST_METADATA_NOT_REPRESENTATION_IMPROVEMENT")
    guarded = [record, gate["producer"], gate["independent_FIT_target_audit"], gate["scope"], fit_audit["audit_source"]]
    guarded += gate["source_records"] + gate["input_records"] + gate["protected_prior_records"]
    for entry in guarded:
        check()
        require(prior.file_sha(entry["path"]) == entry["sha256"], "UNCHANGED_GATE_SOURCE_ACCEPTED_INPUT_PREDICTION_OR_SELECTION_FILE")
    resource = prior.verified_json(gate["resource_terminal"], check)
    require(resource["status"] == "COMPLETE" and resource["elapsed_or_cumulative_budget_caps"] is None, "ACTUAL_COMPLETED_CPU_GATE_RECEIPT")
    require(set(gate["datasets"]) == set(DATASETS), "EXACT_TWO_CACHED_GATE_DATASETS")
    report = {"status": "PASS", "diagnostic": record, "producer": gate["producer"], "fit_target_audit": gate["independent_FIT_target_audit"],
              "scope": gate["scope"], "datasets": {}, "totals": {"dataset_split_cells_checked": 0, "method_pairs_checked": 0,
                  "ungated_prediction_arrays_checked": 0, "gated_prediction_arrays_checked": 0, "score_field_checks": 0,
                  "protected_prior_file_hashes_checked": len(gate["protected_prior_records"])},
              "protected_TEST_accessed": False, "new_training_or_GPU_execution": False, "independent_generalization_claimed": False,
              "whole_deployment_cost_measured": False, "new_F_or_head_forward_executed": False,
              "past_C_condition_metadata_added": True, "same_decoder_side_information_as_ungated": False,
              "representation_learning_improvement_proven": False, "static_D_lower_bound_for_gated_forecasts_claimed": False}
    for dataset, (channels, k, development, development_count) in DATASETS.items():
        row, original = gate["datasets"][dataset], producer["datasets"][dataset]
        require(row["channels"] == channels and row["K"] == original["K"] == k
                and set(row["splits"]) == {"fit", "tune", development}, "EXACT_GATE_CHANNELS_K_SPLITS")
        report["datasets"][dataset] = {"K": k, "channels": channels, "splits": {}}
        for split, count in (("fit", 256), ("tune", 64), (development, development_count)):
            entry = row["splits"][split]
            require(entry["input_source"] == original["inputs"][split], "EXACT_GATE_ORIGINAL_SPLIT_RECORD")
            values = prior.verified_arrays(entry["input_source"], check)
            prior.check_raw_split(values, channels=channels, windows=count)
            methods, sources = {"RAW_BOLT": values["raw_forecast"]}, {"RAW_BOLT": original["inputs"][split]}
            for route in ("PCA_PINV", "FIT_TARGET_PINV"):
                condition = original["projections"][route]
                for method, source in [(route + "_BOLT", condition["bolt"]["artifacts"][split]),
                                      *[(route + "_NATIVE_s" + str(seed), condition["native"][str(seed)]["artifacts"][split]) for seed in (92601, 92602)]]:
                    cached = prior.verified_arrays(source, check)
                    exact(cached["origins"], values["origins"], "original_cached_gate_method_origins:" + method)
                    methods[method], sources[method] = cached["prediction"], source
            require(entry["cached_prediction_sources"] == sources, "EXACT_SEVEN_GATE_CACHED_PREDICTION_RECORDS")
            arrays = prior.verified_arrays(entry["artifact"], check)
            cell = audit_case(entry, arrays, values, methods)
            report["datasets"][dataset]["splits"][split] = cell
            report["totals"]["dataset_split_cells_checked"] += 1
            report["totals"]["method_pairs_checked"] += 7
            report["totals"]["ungated_prediction_arrays_checked"] += 7
            report["totals"]["gated_prediction_arrays_checked"] += 7
            report["totals"]["score_field_checks"] += cell["score_field_checks"]
    require(report["totals"]["dataset_split_cells_checked"] == 6 and report["totals"]["method_pairs_checked"] == 42
            and report["totals"]["score_field_checks"] == 2016, "FULL_REGISTERED_GATE_AUDIT_COUNTS")
    require(gate["source_cache_selection_SHA_unchanged"] is True, "PRODUCER_RECORDED_ORIGINAL_SOURCE_CACHE_SELECTION_PRESERVATION")
    for entry in guarded:
        check()
        require(prior.file_sha(entry["path"]) == entry["sha256"], "GATE_AUDITED_SOURCE_CACHE_OR_RESULT_CHANGED_DURING_EXECUTION")
    return report


def execute(gate_path, output_path):
    from continuation_runtime_v1 import Meter, write_new
    output = Path(output_path).resolve()
    require(not output.exists(), "FRESH_UNIQUE_INDEPENDENT_GATE_AUDIT_OUTPUT")
    output.parent.mkdir(parents=True, exist_ok=True)
    meter = Meter("audit_past_constant_" + output.stem, gpu=False)
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
            report = audit_result(gate_path, meter)
        meter.protected_check()
        require(prior.file_sha(__file__) == source["sha256"], "INDEPENDENT_GATE_AUDIT_SOURCE_UNCHANGED")
        # Metadata serialization is checked before closing a PASS receipt.
        json.dumps(report, allow_nan=False)
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        report = {"status": "FAILED_PRESERVED", "failure": failure, "full_scope_PASS_claimed": False}
    finally:
        report["audit_source"] = source
        report["resource_terminal"] = meter.close(report["status"], failure)
        write_new(output, report)
    print(json.dumps({"status": report["status"], "output": str(output), "totals": report.get("totals"), "failure": failure}, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    raise SystemExit(execute(arguments.gate, arguments.output))
