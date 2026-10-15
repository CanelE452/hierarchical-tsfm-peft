"""Independent recorded causal-lag whole costs; no F/GPU/timer remeasurement."""
from pathlib import Path
import argparse
import ast
import json
import math
import statistics
import sys

import audit_projection_results_v1 as prior
import audit_raw_subset_cost_v2 as static_audit
import audit_causal_lag_objective_v2 as lag_audit

HERE, BASE, ROOT = prior.HERE, prior.BASE, prior.ROOT
DATASETS, SPACES, CONDITIONS = lag_audit.DATASETS, lag_audit.SPACES, lag_audit.CONDITIONS
ROUTES = ("RAW_BOLT", *(space + "_" + condition + "_BOLT" for space in SPACES for condition in CONDITIONS))
PARENT_KEYS = ("causal_lag_diagnostic", "causal_lag_audit", "raw_subset_diagnostic", "raw_subset_audit", "parent_report", "parent_report_manifest", "parent_report_audit", "plan", "fit_registry")
LAG_SHA = "4168cd05962e479130a722919310f4c192c1bab03d9b78cb7983754f2e502c17"
LAG_AUDIT_SHA = "d395a82e76a31e38520db1dd4d56642963ca4e362f358a6d86480049d402a31d"
SCIENCE_COUNTS = {"decoder_states_checked": 24, "fit_lock_files_checked": 24, "FIT_source_contexts_checked": 6,
    "cached_Bolt_source_contexts_checked": 18, "cached_D_split_prediction_arrays_checked": 72,
    "independent_FP32_method_prediction_arrays_checked": 216, "real_trace_prediction_arrays_checked": 216,
    "joint_output_diagnostic_cells_checked": 72, "EY_prediction_arrays_checked": 72, "flat_prediction_arrays_checked": 72,
    "retained_trace_inequalities_checked": 216, "original_real_trace_inequalities_evaluated": 216,
    "FP32_with_allowance_inequalities_evaluated": 216, "RAW_bound_inequalities_checked": 0,
    "cached_future_label_perturbation_checks": 72, "independent_fit_response_columns_checked": 336, "score_field_checks": 14688}
require, exact = prior.require, lag_audit.exact
array_record, parity_checked, json_record = static_audit.array_record, static_audit.parity_checked, static_audit.json_record
np = sys.modules.get("numpy")


def initialize_numpy():
    global np
    static_audit.initialize_numpy(); lag_audit.initialize_numpy(); np = sys.modules["numpy"]


def independent_schedule():
    initial = np.random.default_rng(20261015).permutation(len(ROUTES))
    return [[ROUTES[index] for index in np.roll(initial, -repeat)] for repeat in range(len(ROUTES))]


def recorded_times(row, raw):
    samples = row["samples"]; require(len(samples) == row["repeats"] and row["warmup_each_route"] == 2, "ACTUAL_SAMPLE_COUNT_AND_WARMUP")
    require([sample["repeat"] for sample in samples] == sorted({sample["repeat"] for sample in samples}) and all(0 <= sample["repeat"] < len(ROUTES) for sample in samples), "UNIQUE_REGISTERED_REPEAT_NUMBERS")
    seconds = [(sample["perf_end_ns"] - sample["perf_start_ns"]) / 1e9 for sample in samples]
    require(all(value > 0 and math.isfinite(value) for value in seconds) and row["seconds"] == seconds, "POSITIVE_ACTUAL_NS_ELAPSED_TIMES")
    raw_by_repeat = {sample["repeat"]: (sample["perf_end_ns"] - sample["perf_start_ns"]) / 1e9 for sample in raw["samples"]}
    raw_seconds = [raw_by_repeat.get(sample["repeat"]) for sample in samples]
    ratios = [value / base if base is not None and base > 0 else None for value, base in zip(seconds, raw_seconds)]
    median = statistics.median(seconds) if seconds else None
    paired = statistics.median(ratios) if ratios and all(value is not None for value in ratios) else None
    require(row["paired_RAW_seconds"] == raw_seconds and row["paired_RAW_ratios"] == ratios, "SAME_REPEAT_PAIRED_RAW_TIMES_AND_RATIOS")
    for field, value in (("median_seconds", median), ("median_seconds_per_window", median / row["batch"] if median is not None else None), ("paired_RAW_median_ratio", paired)):
        if value is None: require(row[field] is None, "EMPTY_OR_MISSING_RAW_SUMMARY_NONE")
        else: prior.close(row[field], value, "fixed_cost:" + field, atol=1e-12, rtol=1e-12)
    complete = row["status"] == "COMPLETE"
    if complete: require(len(samples) == len(ROUTES) and row["failure"] is None, "COMPLETE_THIRTEEN_SAMPLE_ROW_WITHOUT_FAILURE")
    raw_median = statistics.median([value for value in raw_by_repeat.values()]) if raw_by_repeat else None
    return {"median_seconds": median, "median_seconds_per_window": median / row["batch"] if median else None, "paired_RAW_median_ratio": paired, "ratio_of_RAW_medians": median / raw_median if median is not None and raw_median else None, "ratio_of_RAW_medians_uses_complete_raw_cohort": raw["status"] == "COMPLETE", "samples_checked": len(samples), "independent_timer_remeasurement": False}


def memory_checked(memory):
    fields = {"RSS_bytes", "allocated_VRAM_bytes", "reserved_VRAM_bytes", "max_allocated_VRAM_bytes", "max_reserved_VRAM_bytes"}
    require(set(memory) == fields | {"scope"} and all(type(memory[field]) is int and memory[field] >= 0 for field in fields) and memory["scope"] == "shared resident frozen F, six E and all fixed D; no method-exclusive allocation", "SHARED_PROCESS_MEMORY_SCOPE_AND_COUNTER_TYPES")
    require(memory["reserved_VRAM_bytes"] >= memory["allocated_VRAM_bytes"] and memory["max_reserved_VRAM_bytes"] >= memory["max_allocated_VRAM_bytes"] >= memory["allocated_VRAM_bytes"] and memory["max_reserved_VRAM_bytes"] >= memory["reserved_VRAM_bytes"], "CURRENT_AND_UNRESET_PEAK_MEMORY_RELATIONSHIPS")


def source_contract(helper_text, runner_text, original_text):
    def functions(text): return {node.name: node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)}
    helper, runner, original = map(functions, (helper_text, runner_text, original_text))
    for name in ("fill_vectorized", "preprocess", "parity", "whole_predict"):
        require(ast.dump(helper[name], include_attributes=False) == ast.dump(original[name], include_attributes=False), "UNCHANGED_PHYSICAL_PREPROCESS_TOLERANCE_CPU_RETURN:" + name)
    names = {"ROUTES", "ORDER_SEED", "WARMUP", "REPEATS", "NORM_ATOL", "NORM_RTOL", "PHYSICAL_ATOL", "PHYSICAL_RTOL"}; constants = {}
    for node in ast.parse(helper_text).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1: continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in names: constants[target.id] = ast.literal_eval(node.value)
        elif isinstance(target, ast.Tuple) and all(isinstance(item, ast.Name) and item.id in names for item in target.elts): constants.update(zip((item.id for item in target.elts), ast.literal_eval(node.value)))
    require(constants == {"ROUTES": ROUTES, "ORDER_SEED": 20261015, "WARMUP": 2, "REPEATS": 13, "NORM_ATOL": 1e-5, "NORM_RTOL": 2e-6, "PHYSICAL_ATOL": 2e-3, "PHYSICAL_RTOL": 2e-5}, "THIRTEEN_FULL_ROTATIONS_EXACT_REGISTERED_CONSTANTS")
    require(ast.dump(helper["balanced_schedule"], include_attributes=False) == ast.dump(original["balanced_schedule"], include_attributes=False), "UNCHANGED_LATIN_SCHEDULE_ALGORITHM")
    timers = [node for node in ast.walk(helper["measure_cost"]) if isinstance(node, ast.Try) and len(node.body) == 3 and "perf_end_ns, unix_end_ns = (time.perf_counter_ns(), time.time_ns())" in ast.unparse(node)]
    require(len(timers) == 1 and [ast.unparse(node) for node in timers[0].body] == ["normalized, physical_value = whole_predict(predictors[route], values, masks, mean, std)", "synchronize()", "perf_end_ns, unix_end_ns = (time.perf_counter_ns(), time.time_ns())"], "WHOLE_CALLBACK_AND_SYNC_INSIDE_TIMER")
    measure = ast.unparse(helper["measure_cost"])
    require("unix_start_ns, perf_start_ns = (time.time_ns(), time.perf_counter_ns())" in measure and "seconds = (perf_end_ns - perf_start_ns) / 1000000000.0" in measure and "memory = _jsonable(memory_snapshot())" in measure, "ACTUAL_NS_TIMERS_AND_OUTSIDE_MEMORY_SNAPSHOT")
    callbacks = [node for node in runner["bolt_predictor"].body if isinstance(node, ast.FunctionDef)]
    require(len(callbacks) == 1 and [arg.arg for arg in callbacks[0].args.args] == ["history", "history_observed"] and [ast.unparse(node) for node in callbacks[0].body] == ["z = encode_numpy(state, indices, history)", "future = frozen_F(z)", "return predict(z, future, history[:, -1], decoder, lags)"], "ACTUAL_E_THEN_F_THEN_FIXED_CAUSAL_LAG_DECODER_CALLBACK")
    require("fit_reconstruction(" not in runner_text and "fit_forecast(" not in runner_text and "native_predictor" not in runner_text, "NO_D_FIT_NATIVE_LOAD_OR_CACHED_F_RETURN_IN_COST")
    return {"physical_preprocessing_actual_E_F_lag_features_fixed_D_current_C_CPU_inverse_inside_timer": True, "all_source_file_state_metadata_hashes_outside_timer": True, "per_origin_decoder_fit_in_cost": False, "thirteen_complete_order_rotations": True, "independent_F_or_timers_replayed": False}


def load_parent_metadata(cost, check):
    return {key: lag_audit.verified_metadata(cost[key], check) for key in PARENT_KEYS}


def metadata_gates(cost_path, check):
    binding, cost = json_record(cost_path, check)
    if cost.get("status") not in ("COMPLETE", "PARTIAL_COMPLETED"): return binding, cost, {}
    require(cost["routes"] == list(ROUTES) and cost["expected_cost_cells"] == 52 and cost["expected_timer_samples"] == 676 and cost["new_training_updates"] == cost["new_native_head_training"] == 0, "REGISTERED_THIRTEEN_ROUTES_52_CELLS_676_SAMPLES")
    for key in ("new_backbone_training", "native_heads_loaded_or_timed", "per_origin_decoder_fit", "cached_future_returned_instead_of_actual_F_inside_timer", "strict_K_only_compression", "protected_TEST_accessed", "independent_holdout", "new_D_or_lag_objective_selection"):
        require(cost[key] is False, "NO_NEW_FIT_SELECTION_TEST_OR_CACHED_F_CALLBACK:" + key)
    require(cost["fixed_FIT_decoder_loaded_before_timer"] is cost["all_source_file_state_metadata_hashes_outside_timer"] is True and cost["hard_elapsed_or_cumulative_resource_caps"] is None, "FIXED_D_AND_METADATA_HASHES_OUTSIDE_TIMER")
    parents = load_parent_metadata(cost, check); science, audit = parents["causal_lag_diagnostic"], parents["causal_lag_audit"]
    require(cost["causal_lag_diagnostic"]["sha256"] == LAG_SHA and cost["causal_lag_audit"]["sha256"] == LAG_AUDIT_SHA and science["status"] == "COMPLETE" and audit["status"] == "PASS" and all(audit["totals"].get(key) == value for key, value in SCIENCE_COUNTS.items()), "EXACT_FULL_SCIENCE_AND_INDEPENDENT_V2_PASS_BEFORE_ARRAYS")
    static_audit.old_cost_audit.assert_binding(audit["diagnostic"], cost["causal_lag_diagnostic"], "accepted causal-lag science")
    static_audit.old_cost_audit.assert_binding(audit["audit_source"], {"path": str(HERE / "audit_causal_lag_objective_v2.py"), "sha256": "022e9506fe0dbaa593f069e48972cb878a6a2f5cb2f0d062feb0cdde5cbeae30"}, "immutable science auditor v2")
    runner = static_audit.literal_constants(HERE / "run_causal_lag_cost_v1.py")
    require(runner["SCIENCE_RESULT_SHA"] == LAG_SHA and runner["SCIENCE_AUDIT_SHA"] == LAG_AUDIT_SHA and runner["SCIENCE_AUDIT_COUNTS"] == SCIENCE_COUNTS, "ACTUAL_COST_GATE_PINS_EXACT_FULL_SCIENCE_PASS")
    require(prior.file_sha(HERE / "causal_lag_decoder_v1.py") == runner["PURE_SHA"] == "f9846499e2cb9c8a322b25cb1e2c328a5eb3ff059d674a20ba172258c207ee6b", "UNCHANGED_ACCEPTED_PURE_DECODER")
    for key in PARENT_KEYS[2:]:
        static_audit.old_cost_audit.assert_binding(cost[key], science[key], "science parent " + key)
        static_audit.old_cost_audit.assert_binding(audit[key], science[key], "audit parent " + key)
    registry = parents["fit_registry"]
    require(registry["status"] == "ALL_24_RESOLVED_BEFORE_TUNE_DEV" and len(registry["conditions"]) == 24 and registry["failed_fit_paths"] == 0 and registry["TUNE_DEV_NPZ_opened_before_seal"] is False, "ALL_24_FIT_ONLY_LOCKS_SEALED")
    keys = set()
    for row in registry["conditions"]:
        key = row["dataset"], row["space"], row["condition"]
        require(key not in keys and row["fit_lock"] == cost["fit_locks"][key[0]][key[1]][key[2]], "UNIQUE_EXACT_COST_FIT_LOCK")
        keys.add(key); lock = prior.verified_json(row["fit_lock"], check); entry = science["datasets"][key[0]]["spaces"][key[1]]["conditions"][key[2]]
        lags, objective = CONDITIONS[key[2]]
        require(lock["status"] == "LOCKED_FIT_ONLY" and (lock["dataset"], lock["space"], lock["condition"]) == key and lock["TUNE_DEV_used_for_fit_or_selection"] is lock["oracle_used_for_fit_or_selection"] is False and lock["lags"] == entry["lags"] == list(lags) and lock["objective"] == entry["objective"] == objective and lock["state_artifact"] == entry["state_artifact"] and row["fit_lock"] == entry["fit_lock"], "FIXED_FIT_ONLY_LAG_D_EXACT_STATE_AND_OBJECTIVE")
    require(keys == {(data, space, condition) for data in DATASETS for space in SPACES for condition in CONDITIONS}, "ALL_24_UNCHANGED_D_CONDITIONS")
    return binding, cost, parents


def physical_precheck(entry, arrays, physical, raw, fixed, cached, encoder, decoders, predictions):
    prefix = prior.causal_physical_preprocess(physical["history_physical"][:8], physical["history_observed"][:8], raw["mean"], raw["std"])
    z = lag_audit.raw_audit.encode(prefix, encoder, "BOLT_NUMPY")
    expected_keys = {"preprocessed_history", "history_observed", "origins", "latent_history", "actual_F_latent_future"} | {condition + "_" + suffix for condition in decoders for suffix in ("prediction_FP32", "decoder_FP32", "lags")}
    require(set(arrays) == expected_keys and entry["origins"] == raw["origins"][:8].tolist() and entry["actual_F_invoked"] is True and entry["per_origin_decoder_fit"] is False, "PHYSICAL_FIRST8_EXACT_ARRAY_CONTRACT")
    for key, value in (("preprocessed_history", prefix), ("history_observed", physical["history_observed"][:8]), ("origins", raw["origins"][:8]), ("latent_history", z)): exact(arrays[key], value, "physical_lag:" + key)
    decisions = {"normalized_history": prefix.tobytes() == raw["history"][:8].tobytes(), "history_observed": physical["history_observed"][:8].tobytes() == raw["history_observed"][:8].tobytes(), "origins": raw["origins"][:8].tobytes() == fixed["origins"][:8].tobytes()}
    require(entry["exact_decisions"] == decisions and all(decisions.values()), "EXACT_PHYSICAL_CAUSAL_MASK_ORIGIN_DECISIONS")
    comparisons = {"history": (prefix, raw["history"][:8]), "currentC": (prefix[:, -1], raw["raw_last"][:8]), "latent_history": (z, fixed["latent_history_BOLT_NUMPY"][:8]), "latent_future": (arrays["actual_F_latent_future"], cached["latent_future"][:8])}
    require(set(entry["checks"]) == set(comparisons) and set(entry["prediction_checks"]) == set(decoders), "PHYSICAL_PREFLIGHT_ALL_FOUR_D_CHECKS")
    for key, (actual, expected) in comparisons.items(): parity_checked(entry["checks"][key], actual, expected)
    for condition, state in decoders.items():
        d, lags = state["decoder_FP32"], state["lags"].tolist()
        exact(arrays[condition + "_decoder_FP32"], d, "physical_frozen_D")
        exact(arrays[condition + "_lags"], state["lags"], "physical_frozen_lags")
        p = lag_audit.anchored_forecast(z, arrays["actual_F_latent_future"], prefix[:, -1], d, lags)
        exact(arrays[condition + "_prediction_FP32"], p, "independent_lag_features_anchor_D32")
        parity_checked(entry["prediction_checks"][condition]["normalized"], p, predictions[condition][:8])
        parity_checked(entry["prediction_checks"][condition]["physical"], (p * raw["std"] + raw["mean"]).astype(np.float32), (predictions[condition][:8] * raw["std"] + raw["mean"]).astype(np.float32), physical=True)
    require(entry["status"] == "PASS" and entry["input_backend_pass"] is True, "PHYSICAL_PREFLIGHT_RECORDED_PASS")
    return {"first8_windows_checked": 8, "fixed_D_predictions_checked": len(decoders), "independent_F_calls": 0, "independent_lag_feature_D32_predictions_replayed": True}


def audit_cost_dataset(cost, physical, dev, predictions, metadata, check, sealed_unix):
    require(cost["routes"] == list(ROUTES) and len(cost["rows"]) == 2 * len(ROUTES) and cost["expected_cells_this_dataset"] == 2 * len(ROUTES) and cost["expected_cells_two_datasets"] == 4 * len(ROUTES) and cost["warmup_each_route"] == 2 and cost["repeats"] == len(ROUTES), "THIRTEEN_DECLARED_ROUTES_TWO_BATCHES_ALL_SLOTS_RETAINED")
    require(cost["order_seed"] == 20261015 and cost["schedule"] == independent_schedule(), "EXACT_THIRTEEN_LATIN_ROTATIONS")
    require(cost["new_training_updates"] == 0 and cost["encoder_refit_in_cost"] is cost["per_origin_decoder_fit"] is cost["fit_input_binding_hashes_inside_timer"] is cost["protected_TEST_accessed"] is cost["independent_holdout"] is False and cost["fixed_FIT_decoder_loaded_before_timer"] is True, "FIXED_FIT_D_NO_FIT_OR_METADATA_HASHES_IN_TIMED_CALLBACK")
    exact(prior.causal_physical_preprocess(physical["history_physical"], physical["history_observed"], dev["mean"], dev["std"]), dev["history"], "all_physical_past_exact_original_normalized_input")
    for key, value in (("physical_past", physical["history_physical"]), ("history_observed", physical["history_observed"]), ("FIT_mean", dev["mean"]), ("FIT_std", dev["std"]), ("origins", dev["origins"])): require(cost["input_array_metadata"][key] == array_record(value), "COST_INPUT_ARRAY_SHA:" + key)
    table = {(row["route"], row["batch"]): row for row in cost["rows"]}
    require(len(table) == 2 * len(ROUTES) and set(table) == {(route, batch) for route in ROUTES for batch in (1, 8)}, "NO_DUPLICATE_OR_MISSING_DECLARED_COST_CELL")
    lineage = cost["source_lineage"]
    require(lineage["whole_cost_source_sha256"] == prior.file_sha(HERE / "whole_causal_lag_cost_v1.py") and Path(lineage["copied_timer_source"]).resolve() == HERE / "whole_raw_subset_cost_v1.py" and lineage["copied_timer_source_sha256"] == prior.file_sha(HERE / "whole_raw_subset_cost_v1.py") and Path(lineage["copied_preprocess_source"]).resolve() == BASE / "cost_readout_v1.py" and lineage["copied_preprocess_source_sha256"] == prior.file_sha(BASE / "cost_readout_v1.py") and lineage["copied_functions"] == ["fill_vectorized", "preprocess", "parity"] and lineage["legacy_scientific_runtime_imported"] is False, "UNCHANGED_PHYSICAL_TIMER_SOURCE_LINEAGE")
    schedule, details, chronological, completed, samples_total = independent_schedule(), [], [], 0, 0
    require(set(cost["source_metadata"]) == set(ROUTES), "ALL_THIRTEEN_ROUTE_METADATA_SLOTS_PRESERVED")
    for (route, batch), row in table.items():
        check(); accepted = row["status"] == "COMPLETE"
        require(row["dataset"] == cost["dataset"] and row["declared_route_deleted"] is row["memory_counter_reset_inside_module"] is row["per_origin_decoder_fit"] is row["stage_decomposition"] is False and row["native_head_loaded_or_timed"] == ("_NATIVE_s" in route), "WHOLE_COST_SCOPE_AND_SAVED_PARTIAL_ROUTE_IDENTITY")
        if route in metadata: require(row["selected_metadata"] == cost["source_metadata"][route] == metadata[route], "EXACT_COST_ROUTE_REFERENCE_AND_SELECTED_CP_METADATA")
        else: require(not accepted, "COMPLETE_ROW_MISSING_SCIENTIFIC_ROUTE_METADATA")
        require(row["origins"] == dev["origins"][:batch].tolist(), "SAME_FIRST_B_ORIGINS")
        arithmetic = recorded_times(row, table[("RAW_BOLT", batch)])
        repeats = [sample["repeat"] for sample in row["samples"]]
        require(row["positions"] == [schedule[repeat].index(route) for repeat in repeats] and len(row["samples"]) == len(row["parity"]) == len(row["memory"]), "THIRTEEN_ROTATION_POSITION_AND_SAMPLE_LIST_IDENTITY")
        if not row["samples"]:
            require(not accepted and row["timed_outputs"] is None, "EMPTY_FAILED_PATH_WITHOUT_FABRICATED_TIMERS")
            details.append({"route": route, "batch": batch, "status": "PARTIAL_NO_RECORDED_SAMPLES", **arithmetic}); continue
        require(route in predictions and row["reference"] is not None, "SAVED_TIMER_REQUIRES_BOUND_SCIENTIFIC_FORECAST")
        refs, timed = (prior.verified_arrays(row[key], check) for key in ("reference", "timed_outputs"))
        require(set(refs) == {"normalized", "physical", "scientific_normalized", "origins"} and set(timed) == {"normalized", "physical", "block", "position", "origins"}, "EXACT_SAVED_REFERENCE_AND_STACK_KEYS")
        shape = (batch, 48, len(dev["mean"]))
        require(all(refs[key].shape == shape and refs[key].dtype == np.float32 for key in ("normalized", "physical", "scientific_normalized")) and all(timed[key].shape == (len(repeats),) + shape and timed[key].dtype == np.float32 for key in ("normalized", "physical")), "ACTUAL_FP32_REFERENCE_AND_TIMER_STACK_DIMENSIONS")
        for arrays in (refs, timed): exact(arrays["origins"], dev["origins"][:batch], "timer_saved_origins")
        exact(timed["block"], np.asarray(repeats, np.int64), "timer_recorded_repeat_stack")
        exact(timed["position"], np.asarray(row["positions"], np.int64), "timer_recorded_position_stack")
        exact(refs["scientific_normalized"], predictions[route][:batch], "exact_selected_scientific_output_reference")
        prior.close(refs["normalized"], refs["scientific_normalized"], "same_B_batch_vs_canonical32_saved_prediction", atol=1e-5, rtol=2e-6)
        exact(refs["physical"], (refs["normalized"] * dev["std"] + dev["mean"]).astype(np.float32), "reference_materialized_CPU_physical_inverse")
        physical_science = (predictions[route][:batch] * dev["std"] + dev["mean"]).astype(np.float32)
        errors = []
        for index, sample in enumerate(row["samples"]):
            require(sample["dataset"] == cost["dataset"] and sample["route"] == route and sample["batch"] == batch and sample["repeat"] == sample["block"] == repeats[index] and sample["position"] == row["positions"][index], "EXACT_RECORDED_TIMER_SAMPLE_IDENTITY")
            require(sample["perf_end_ns"] > sample["perf_start_ns"] and sample["unix_end_ns"] >= sample["unix_start_ns"] > sealed_unix * 1e9 and sample["seconds"] == row["seconds"][index] and sample["parity"] == row["parity"][index] and sample["memory"] == row["memory"][index], "POSITIVE_RECORDED_TIMERS_AFTER_SELECTION_AND_ROW_BINDINGS")
            prior.close(sample["seconds"], (sample["perf_end_ns"] - sample["perf_start_ns"]) / 1e9, "elapsed_ns_seconds", atol=1e-12, rtol=1e-12)
            memory_checked(sample["memory"]); chronological.append((batch, sample["repeat"], sample["position"], sample))
            _, literal = json_record(Path(sample["sample_artifact"]["path"]).with_suffix(".json"), check)
            require(literal == sample, "EXACT_TIMER_SAMPLE_LITERAL_JSON")
            saved = prior.verified_arrays(sample["sample_artifact"], check)
            require(set(saved) == {"normalized", "physical", "origins"}, "THREE_MATERIALIZED_TIMER_ARRAYS")
            exact(saved["origins"], dev["origins"][:batch], "sample_origins")
            for key in ("normalized", "physical"): exact(saved[key], timed[key][index], "saved_timer_stack_member:" + key)
            exact(saved["physical"], (saved["normalized"] * dev["std"] + dev["mean"]).astype(np.float32), "timed_physical_CPU_inverse")
            errors.append({key: parity_checked(sample["parity"][key], actual, expected, physical=physical_units, must_pass=accepted) for key, actual, expected, physical_units in (("normalized", saved["normalized"], refs["normalized"], False), ("physical", saved["physical"], refs["physical"], True), ("scientific_normalized", saved["normalized"], refs["scientific_normalized"], False), ("scientific_physical", saved["physical"], physical_science, True))})
        memory_summary = {key: {"min": min(entry[key] for entry in row["memory"]), "max": max(entry[key] for entry in row["memory"])} for key in row["memory"][0] if key != "scope"}
        details.append({"route": route, "batch": batch, "status": "ACCEPTED_COMPLETE" if accepted else "PARTIAL_RECORDED_NOT_ACCEPTED", **arithmetic, "recorded_shared_memory_range_bytes": memory_summary, "independent_saved_output_parity": errors, "all_recorded_samples_retained": True})
        completed += int(accepted); samples_total += len(row["samples"])
    chronological.sort(key=lambda item: item[:3])
    for before, after in zip(chronological, chronological[1:]):
        require(after[3]["perf_start_ns"] >= before[3]["perf_end_ns"], "SERIAL_TIMER_SAMPLES_NONOVERLAPPING")
        for key in ("max_allocated_VRAM_bytes", "max_reserved_VRAM_bytes"): require(after[3]["memory"][key] >= before[3]["memory"][key], "UNRESET_CUMULATIVE_SHARED_VRAM_PEAKS")
    require(cost["completed_cost_cells"] == completed and cost["recorded_timer_samples"] == samples_total and cost["attempted_timer_calls"] >= samples_total, "ACTUAL_COMPLETE_CELLS_AND_ALL_SAVED_PARTIAL_SAMPLES")
    if cost["status"] == "COMPLETE": require(completed == 2 * len(ROUTES) and samples_total == cost["attempted_timer_calls"] == 2 * len(ROUTES) ** 2 and cost["failed_paths"] == [], "FULL_THIRTEEN_ROUTE_COST_DATASET_NOT_PARTIAL")
    return {"rows": details, "cells_checked": completed, "recorded_timer_samples_checked": samples_total, "declared_cells_checked": 2 * len(ROUTES), "actual_timers_or_F_remeasured": False, "all_thirteen_positions_balanced_for_complete_routes": True, "exclusive_per_method_memory_claimed": False, "outliers_or_samples_removed": False, "same_first_B_origins_repeated_not_independent_draws": True}


def route_metadata(scientific_row, subset_row, dev, locks):
    output = {"RAW_BOLT": {"scope": "actual full-C frozen F", "source": subset_row["raw_input_sources"][dev]}}
    for space in SPACES:
        for condition, (lags, objective) in CONDITIONS.items():
            entry = scientific_row["spaces"][space]["conditions"][condition]
            output[space + "_" + condition + "_BOLT"] = {"space": space, "condition": condition, "lags": list(lags), "objective": objective, "E_state": subset_row["spaces"][space]["E_state_artifact"], "D_state": entry["state_artifact"], "fit_lock": locks[space][condition], "reference": entry["splits"][dev]["artifact"], "reference_prediction_array": "ACTUAL_F_prediction_FP32", "actual_F_forward": True, "cached_F_return_inside_timer": False, "per_origin_fit": False, "current_C_anchor": True, "known_K_past_buffer": True, "strict_K_only": False}
    return output


def audit_result(cost_path, meter):
    check = meter.check; binding, cost, parents = metadata_gates(cost_path, check)
    if not parents: return {"status": "NOT_AUDITED_INCOMPLETE_PRODUCER", "cost": binding, "producer_status": cost.get("status"), "scientific_assets_opened": False}
    producer_complete = cost["status"] == "COMPLETE"
    science, subset = parents["causal_lag_diagnostic"], parents["raw_subset_diagnostic"]
    original = prior.verified_json(subset["source_producer"], check)
    paths = [HERE / name for name in ("whole_causal_lag_cost_v1.py", "run_causal_lag_cost_v1.py", "whole_raw_subset_cost_v1.py")]
    source = source_contract(*(path.read_text(encoding="utf-8-sig") for path in paths))
    guarded = [binding, *[cost[key] for key in PARENT_KEYS], *cost["source_records"], *cost["source_snapshots"], *cost["protected_prior_records"], *cost["input_records"], cost["cost_plan"], parents["causal_lag_audit"]["audit_source"], subset["source_producer"]]
    for record in guarded:
        check(); require(prior.file_sha(record["path"]) == record["sha256"], "PRE_AUDIT_SCIENCE_COST_SOURCE_AND_CACHE_SHA")
        if "snapshot_path" in record: require(prior.file_sha(record["snapshot_path"]) == record["sha256"], "EXACT_EXECUTED_SOURCE_SNAPSHOT_BYTES")
    require(cost["F_before"] == cost["F_after"] == subset["F_before"] == subset["F_after"] == original["F_before"] and set(cost["F_before"]) == {"old_algorithm", "bi_algorithm"} and science["frozen_F_parent_signatures"] == {"before": subset["F_before"], "after": subset["F_after"]}, "SAME_DUAL_F_SIGNATURE_AND_SCIENCE_CACHE_LINEAGE")
    for name in ("model_binary", "model_config"):
        static_audit.old_cost_audit.assert_binding(cost[name + "_before"], cost[name + "_after"], "frozen before/after " + name)
        static_audit.old_cost_audit.assert_binding(cost[name + "_before"], original[name + "_before"], "original accepted " + name)
        guarded.append(cost[name + "_before"])
    for key in ("evaluation", "all_backbone_parameters_frozen", "context_length", "prediction_length", "input_patch_size", "input_patch_stride", "median_index", "returned_horizon", "quantiles", "pipeline_predict_not_used", "normalizer_before", "normalizer_after", "normalizer_fixture_forward_parity"):
        require(cost["runtime_behavior_contract"][key] == original["runtime_behavior_contract"][key], "SAME_FROZEN_BOLT_RUNTIME:" + key)
    receipt = prior.verified_json(cost["resource_terminal"], check)
    require(receipt["status"] == cost["status"] and receipt["elapsed_or_cumulative_budget_caps"] is None, "ACTUAL_COST_RESOURCE_RECEIPT_UNCAPPED_AUTHORIZATION")
    registry = parents["fit_registry"]; sealed = registry["sealed_unix"]
    totals = {key: 0 for key in ("cost_cells_checked", "recorded_timer_samples_checked", "physical_backend_prechecks_checked", "physical_first8_windows_checked", "fixed_D_first8_prediction_checks", "encoder_state_hashes_checked", "fixed_decoder_state_hashes_checked", "scientific_prediction_file_hashes_checked", "cached_Bolt_file_hashes_checked", "canonical_physical_bridges_checked")}
    totals.update(fit_lock_files_checked=24, protected_prior_file_hashes_checked=len(cost["protected_prior_records"]))
    report = {"status": "PASS" if producer_complete else "PARTIAL_AUDITED_NO_FULL_PASS", "cost": binding, **{key: cost[key] for key in PARENT_KEYS}, "fit_locks": cost["fit_locks"], "source_contract": source, "totals": totals, "datasets": {}, "cost_details": {}, "producer_failed_paths": cost["failed_paths"], "partial_cost_paths_not_accepted": [], "protected_TEST_accessed": False, "new_training_or_F_or_GPU_execution": False, "F_or_actual_timers_independently_replayed": False, "independent_CPU_lag_feature_and_fixed_D_equations_replayed": True, "per_origin_decoder_fit_in_cost": False, "per_method_exclusive_RAM_or_VRAM_claimed": False, "strict_K_only_compression": False, "independent_generalization_claimed": False, "outliers_or_samples_removed": False, "new_D_or_lag_objective_selection_performed": False}
    hashes = {"E": {}, "D_states": {}}
    for dataset, (channels, k, dev, count) in DATASETS.items():
        check(); row, srow, urow = cost["datasets"][dataset], science["datasets"][dataset], subset["datasets"][dataset]
        if "canonical_physical_bridge" not in row:
            require(not producer_complete, "COMPLETE_COST_MISSING_PHYSICAL_INPUTS")
            report["partial_cost_paths_not_accepted"].append({"dataset": dataset, "metadata": row}); continue
        raw = prior.verified_arrays(urow["raw_input_sources"][dev], check); prior.check_raw_split(raw, channels=channels, windows=count)
        proof = row["canonical_physical_bridge"]
        bridge_row = {"physical_bridge_canonical": proof["canonical_record"], "canonical_physical_bridge_binding": proof}
        physical = static_audit.canonical_bridge(bridge_row, urow, original, dataset, dev, check)
        for name in ("origins", "history_observed", "mean", "std"): exact(physical[name], raw[name], "physical_same_contract:" + name)
        totals["canonical_physical_bridges_checked"] += 1
        predictions, prechecks = {"RAW_BOLT": raw["raw_forecast"]}, {}
        for space in SPACES:
            parent, prepared = urow["spaces"][space], row["spaces"].get(space)
            if prepared is None:
                require(not producer_complete, "COMPLETE_COST_MISSING_E_PREPARATION")
                report["partial_cost_paths_not_accepted"].append({"dataset": dataset, "space": space}); continue
            e = prior.verified_arrays(parent["E_state_artifact"], check)
            indices = e["selected_indices"].tolist() if space == "RAW_SUBSET" else None
            require(prepared["E_state"] == parent["E_state_artifact"] and prepared["indices"] == indices and set(prepared["available_conditions"]) <= set(CONDITIONS), "EXACT_FROZEN_E_AND_SUBSET_INDICES")
            key = str((dataset, space)); state = {"center": e["center"], "encoder": e["encoder"], "decoder": np.zeros((k, channels), np.float64), "decoder_bias": np.zeros(channels, np.float64)}
            hashes["E"][key] = {name: array_record(value)["sha256"] for name, value in state.items()}; totals["encoder_state_hashes_checked"] += 1
            fixed = prior.verified_arrays(parent["inputs"][dev], check); cached = prior.verified_arrays(parent["Bolt"][dev]["artifact"], check)
            for name in ("history", "history_observed", "raw_last", "origins"): exact(fixed[name], raw[name], "same_cached_E_context:" + name)
            exact(cached["origins"], raw["origins"], "same_old_F_cache_origins")
            decoders, by_condition = {}, {}
            for condition, (lags, objective) in CONDITIONS.items():
                entry = srow["spaces"][space]["conditions"][condition]; record = cost["fit_locks"][dataset][space][condition]
                lock = prior.verified_json(record, check); guarded.append(record); sealed = max(sealed, lock["sealed_unix"])
                if condition not in prepared["available_conditions"]:
                    require(not producer_complete, "COMPLETE_COST_WITH_UNAVAILABLE_D")
                    report["partial_cost_paths_not_accepted"].append({"dataset": dataset, "space": space, "condition": condition}); continue
                d = prior.verified_arrays(entry["state_artifact"], check)
                require(set(d) == {"decoder_FP64", "decoder_FP32", "bias_FP64", "bias_FP32", "lags"} and d["decoder_FP32"].dtype == np.float32 and d["decoder_FP32"].shape == (len(lags) * k, channels) and np.isfinite(d["decoder_FP32"]).all() and d["lags"].dtype == np.int64 and d["lags"].tolist() == list(lags), "LOCKED_FIXED_LAG_D_STATE_ARRAYS")
                hashes["D_states"][str((dataset, space, condition))] = {name: array_record(value)["sha256"] for name, value in d.items()}; totals["fixed_decoder_state_hashes_checked"] += 1
                saved = prior.verified_arrays(entry["splits"][dev]["artifact"], check)
                exact(saved["origins"], raw["origins"], "same_fixed_D_science_prediction_origins")
                decoders[condition], by_condition[condition] = d, saved["ACTUAL_F_prediction_FP32"]
                predictions[space + "_" + condition + "_BOLT"] = by_condition[condition]
            pre = row["physical_prechecks"][space]
            prechecks[space] = physical_precheck(pre, prior.verified_arrays(pre["artifact"], check), physical, raw, fixed, cached, e, decoders, by_condition)
            totals["physical_backend_prechecks_checked"] += 1; totals["physical_first8_windows_checked"] += 8; totals["fixed_D_first8_prediction_checks"] += len(decoders)
        metadata = route_metadata(srow, urow, dev, cost["fit_locks"][dataset])
        if dataset in cost["cost"]:
            details = audit_cost_dataset(cost["cost"][dataset], physical, raw, predictions, metadata, check, sealed)
            _, literal = json_record(Path(binding["path"]).parent / ("cost_" + dataset) / "RESULT.json", check)
            require(literal == cost["cost"][dataset], "EXACT_SAVED_COST_DATASET_JSON")
            report["cost_details"][dataset] = details; totals["cost_cells_checked"] += details["cells_checked"]; totals["recorded_timer_samples_checked"] += details["recorded_timer_samples_checked"]
        else: require(not producer_complete, "COMPLETE_COST_MISSING_DATASET_TIMERS")
        report["datasets"][dataset] = {"physical_prechecks": prechecks, "canonical_physical_bridge": proof}
    for dataset, row in science["datasets"].items():
        for space in row["spaces"].values():
            for condition in space["conditions"].values():
                for split in condition["splits"].values(): guarded.append(split["artifact"]); totals["scientific_prediction_file_hashes_checked"] += 1
    for row in subset["datasets"].values():
        for space in row["spaces"].values():
            for entry in space["Bolt"].values(): guarded.append(entry["artifact"]); totals["cached_Bolt_file_hashes_checked"] += 1
    for record in guarded: check(); require(prior.file_sha(record["path"]) == record["sha256"], "POST_AUDIT_ORIGINAL_SOURCE_CACHE_LOCK_AND_STATE_BYTES")
    require(hashes == cost["fixed_states_before"] == cost["fixed_states_after"], "INDEPENDENT_SIX_E_AND_24_FIXED_LAG_D_HASHES")
    if producer_complete:
        expected = {"cost_cells_checked": 52, "recorded_timer_samples_checked": 676, "physical_backend_prechecks_checked": 6, "physical_first8_windows_checked": 48, "fixed_D_first8_prediction_checks": 24, "fit_lock_files_checked": 24, "encoder_state_hashes_checked": 6, "fixed_decoder_state_hashes_checked": 24, "scientific_prediction_file_hashes_checked": 72, "cached_Bolt_file_hashes_checked": 18, "canonical_physical_bridges_checked": 2}
        require(all(totals[key] == number for key, number in expected.items()) and cost["completed_cost_cells"] == 52 and cost["recorded_timer_samples"] == cost["completed_cell_timer_samples"] == 676 and cost["failed_paths"] == [] and report["partial_cost_paths_not_accepted"] == [], "COMPLETE_CAUSAL_LAG_COST_REGISTERED_COVERAGE")
    return report


def execute(cost_path, output_path):
    from continuation_runtime_v1 import Meter, write_new
    output = Path(output_path).resolve(); require(not output.exists(), "FRESH_CAUSAL_LAG_COST_AUDIT_OUTPUT")
    output.parent.mkdir(parents=True, exist_ok=True); meter = Meter("audit_causal_lag_cost_" + output.stem, gpu=False)
    source = {"path": str(Path(__file__).resolve()), "sha256": prior.file_sha(__file__)}; failure = None
    try:
        meter.protected_check(); _, _, parents = metadata_gates(cost_path, meter.check)
        if not parents: report = {"status": "NOT_AUDITED_INCOMPLETE_PRODUCER", "scientific_assets_opened": False}
        else:
            initialize_numpy(); sys.path.insert(0, str(ROOT / "research/smallk_failure_followup_20261014"))
            from prepare_inputs import install_protected_test_guard
            install_protected_test_guard()
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=2): report = audit_result(cost_path, meter)
        meter.protected_check(); require(prior.file_sha(__file__) == source["sha256"], "INDEPENDENT_AUDITOR_SOURCE_UNCHANGED"); json.dumps(report, allow_nan=False)
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}; report = {"status": "FAILED_PRESERVED", "failure": failure, "full_scope_PASS_claimed": False}
    finally:
        report["audit_source"] = source; report["resource_terminal"] = meter.close(report["status"], failure); write_new(output, report)
    print(json.dumps({"status": report["status"], "totals": report.get("totals"), "failure": failure}), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--cost", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); raise SystemExit(execute(args.cost, args.output))
