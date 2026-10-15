"""Independent selected-shrinkage physical/ridge/timer audit; never replays frozen F."""
from pathlib import Path
import argparse
import ast
import math
import json
import sys

import audit_projection_results_v1 as prior
import audit_decoder_metric_cost_v1 as timed_audit
import audit_local_decoder_shrinkage_v1 as shrink_audit

np = sys.modules.get("numpy")
HERE, BASE, ROOT = prior.HERE, prior.BASE, prior.ROOT
ROUTES = ("RAW_BOLT", "PCA_PINV_BOLT", "FIT_TARGET_PINV_BOLT", "PCA_SHRUNK_BOLT", "FIT_TARGET_SHRUNK_BOLT")
DATASETS = prior.DATASETS
require, exact = prior.require, shrink_audit.exact
array_record, assert_binding = timed_audit.array_record, timed_audit.assert_binding
recorded_timer = timed_audit.recorded_timer
SHRINK_SHA = "aa553512965bf9253fae1983424e2fcb4f2916cc95ec12e9ed671ae325dbd532"
SHRINK_AUDIT_SHA = "145999ea8bc7068eba08f08039287a33097cdb4e9ec8dab002f4ebc7630fdda9"


def initialize_numpy():
    global np
    timed_audit.initialize_numpy()
    shrink_audit.initialize_numpy()
    np = sys.modules["numpy"]


def independent_schedule():
    generator = np.random.default_rng(20261015)
    first = generator.permutation(5)
    rows = [[ROUTES[index] for index in np.roll(first, -repeat)] for repeat in range(5)]
    rows.append([ROUTES[index] for index in generator.permutation(5)])
    return rows


def source_contract(original_text, helper_text, runner_text, math_text):
    original, helper, runner, math_source = (ast.parse(text) for text in (original_text, helper_text, runner_text, math_text))
    def functions(tree):
        return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    previous, current, callbacks, math_functions = (functions(tree) for tree in (original, helper, runner, math_source))
    for name in ("sha", "array_record", "_jsonable", "write_json_once", "save_outputs", "fill_vectorized", "preprocess", "parity", "balanced_schedule"):
        require(ast.dump(previous[name], include_attributes=False) == ast.dump(current[name], include_attributes=False), "UNCHANGED_WHOLE_TIMING_PREPROCESS_OR_SCHEDULE:" + name)
    def timed_sequence(function):
        found = []
        for node in ast.walk(function):
            if not isinstance(node, ast.For):
                continue
            names = [statement.targets[0].id if isinstance(statement, ast.Assign) and len(statement.targets) == 1
                     and isinstance(statement.targets[0], ast.Name) else None for statement in node.body]
            if "perf_start_ns" in names and "perf_end_ns" in names:
                start, end = names.index("perf_start_ns"), names.index("perf_end_ns")
                found.append([ast.dump(statement, include_attributes=False) for statement in node.body[start:end + 1]])
        require(len(found) == 1, "ONE_ACTUAL_WHOLE_PERF_TIMER_SEQUENCE")
        return found[0]
    require(timed_sequence(previous["measure_cost"]) == timed_sequence(current["measure_cost"]), "WHOLE_PREDICT_PREPROCESS_AND_FINAL_SYNC_INSIDE_TIMER")
    constants = {}
    expected_constant_names = {"ROUTES", "ORDER_SEED", "WARMUP", "REPEATS", "NORM_ATOL", "NORM_RTOL", "PHYSICAL_ATOL", "PHYSICAL_RTOL"}
    for node in helper.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in expected_constant_names:
            constants[target.id] = ast.literal_eval(node.value)
        elif isinstance(target, ast.Tuple) and all(isinstance(value, ast.Name) and value.id in expected_constant_names for value in target.elts):
            constants.update(zip((value.id for value in target.elts), ast.literal_eval(node.value)))
    require(constants == {"ROUTES": ROUTES, "ORDER_SEED": 20261015, "WARMUP": 2, "REPEATS": 6,
            "NORM_ATOL": 1e-5, "NORM_RTOL": 2e-6, "PHYSICAL_ATOL": 2e-3, "PHYSICAL_RTOL": 2e-5}, "EXACT_REGISTERED_FIVE_ROUTE_SEED_AND_UNCHANGED_TOLERANCES")
    calls = [node for node in ast.walk(current["whole_predict"]) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "predictor"]
    require(len(calls) == 1 and [ast.unparse(arg) for arg in calls[0].args] == ["normalized_history", "observed"], "TRUE_PAST_OBSERVATION_MASK_INSIDE_TIMED_CALLBACK")
    nested = [node for node in callbacks["shrunk_predictor"].body if isinstance(node, ast.FunctionDef)]
    require(len(nested) == 1 and [arg.arg for arg in nested[0].args.args] == ["history", "history_observed"], "PAST_ONLY_LOCAL_CALLBACK_SIGNATURE")
    local_calls = [node for node in ast.walk(nested[0]) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "fit_shrunk_local_decoder"]
    require(len(local_calls) == 1 and not local_calls[0].keywords and [ast.unparse(arg) for arg in local_calls[0].args] == ["history", "history_observed", "z", "prior32", "rho"], "LOCAL_LS_EXECUTED_ON_EACH_TIMED_CALL_WITH_NO_CACHED_D_OR_FUTURE_LABEL")
    require([ast.unparse(node) for node in nested[0].body] == ["z = state.encode(history)", "fitted = fit_shrunk_local_decoder(history, history_observed, z, prior32, rho)", "return fitted.forecast(history[:, -1], frozen_F(z), z[:, -1])"], "ACTUAL_ENCODE_LOCAL_FIT_FROZEN_F_DYNAMIC_DECODE_CALLBACK_ORDER")
    fit = math_functions["fit_shrunk_local_decoder"]
    require([arg.arg for arg in fit.args.args] == ["history", "observed", "latent_history", "prior_decoder", "rho"], "PAST_ONLY_LOCAL_SOLVER_API")
    fit_text = ast.unparse(fit)
    require("np.linalg.solve(matrix, linear - gram @ d[window, :, channel])" in fit_text
            and "np.linalg.svd(matrix.astype(np.float64), compute_uv=False)" in fit_text
            and "hashlib.sha256(value.tobytes())" in fit_text and "return LocalPastDecoder(" in fit_text
            and "'actual_lambda'" in fit_text and "'ridge_matrix_condition'" in fit_text,
            "FULL_REGISTERED_RIDGE_IMPLEMENTED_DIAGNOSTICS_INPUT_BINDING_HASHES_INSIDE_CALLBACK")
    for text in (helper_text, runner_text):
        require("fit input binding hashes" in text or "fit_input_binding_hashes_inside_timer" in text,
                "EXPLICIT_FIT_INPUT_HASHES_INCLUDED_IN_TIMER")
    require("source/file/state metadata hashes generated outside timed callback" in helper_text,
            "ONLY_HASHES_GENERATED_OUTSIDE_CALLBACK_EXCLUDED")
    return {"original_perf_counter_whole_call_and_final_sync": True, "unchanged_IO_preprocess_schedule_functions": 9,
            "physical_preprocessing_and_materialized_CPU_return_inside_timer": True, "explicit_true_past_mask": True,
            "selected_ridge_and_registered_full_diagnostics_input_hashes_inside_each_callback": True, "cached_D_in_timed_pipeline": False,
            "independent_F_or_timer_replay_performed": False}


def validate_route_metadata(route, actual, expected):
    require(actual == expected and actual["shared_resident_frozen_F"] is True
            and actual["native_head_loaded_or_timed"] is False, "EXACT_SELECTED_SHRINKAGE_ROUTE_METADATA:" + route)
    if route == "RAW_BOLT":
        require(actual["local_fit"] is False and actual["reference_prediction_array"] == "raw_forecast", "ORIGINAL_RAW_BOLT_REFERENCE")
        return
    shrunk = route.endswith("SHRUNK_BOLT")
    require(actual["local_fit"] == shrunk and actual["local_fit_uses_current_known_past_and_true_observed_mask"] == shrunk
            and actual["local_ridge_and_implemented_diagnostics_inside_timer"] == shrunk
            and actual["fit_input_binding_hashes_inside_timer"] == shrunk
            and actual["cached_local_D_used_for_timing"] is actual["new_E_F_or_head_training"] is actual["strict_K_only_compression"] is actual["new_rho_selection_in_cost"] is False
            and actual["past_C_condition_metadata_added"] == shrunk and actual["same_decoder_side_information_as_static"] == (not shrunk)
            and actual["reference_prediction_array"] == ("BOLT_prediction" if shrunk else "BOLT_static_prediction"),
            "SELECTED_RIDGE_INSIDE_CALLBACK_STATIC_CONTROL_OR_CHANGED_C_INFORMATION")


def independent_physical_precheck(entry, arrays, physical, dev, state, cached, science_state, science_metadata, expected_sources, rho):
    require(entry["status"] == "PASS" and all(entry[key] == value for key, value in expected_sources.items())
            and entry["rho"] == rho and entry["lambda_rule"] == "rho*trace(observed_centered_Z.T@Z/n)/K"
            and entry["decision_acceptance_by_output_tolerance_only"] is False and entry["coefficient_tolerance_loosened"] is False
            and entry["history_observed_byte_exact"] is True and entry["origins"] == dev["origins"][:8].tolist(), "EXACT_LOCKED_RIDGE_PHYSICAL_PREFLIGHT_SOURCES_AND_RULE")
    state_keys = ("decoder_fit_FP64", "decoder_deployed_FP32", "decoder_bias_fit_FP64", "decoder_bias_FP32", "metadata_json")
    require(set(arrays) == {*state_keys, "preprocessed_history", "latent_history", "history_observed", "origins", "prior_decoder_FP32"}, "EXACT_SELECTED_RIDGE_PHYSICAL_PREFLIGHT_ARRAY_SET")
    prefix = prior.causal_physical_preprocess(physical["history_physical"][:8], physical["history_observed"][:8], dev["mean"], dev["std"])
    latent = prior.encode(prefix, state["center"], state["encoder"])
    prior32 = np.ascontiguousarray(state["decoder"].astype(np.float32))
    for key, actual in (("preprocessed_history", prefix), ("latent_history", latent), ("history_observed", dev["history_observed"][:8]), ("origins", dev["origins"][:8]), ("prior_decoder_FP32", prior32)):
        exact(arrays[key], actual, "physical_selected_ridge_saved:" + key)
    require(entry["deployed_static_prior"] == array_record(prior32), "EXACT_PRIOR_DEPLOYMENT_RAWBYTE_BINDING")
    metadata = entry["fitted_shrunk_metadata"]
    require(science_metadata["rho"] == rho and science_metadata["lambda_rule"] == metadata["lambda_rule"]
            and metadata["slope_singular_cutoff_used"] is science_metadata["slope_singular_cutoff_used"] is False, "SAME_ORIGINAL_LOCKED_RHO_AND_TRACE_RULE")
    comparisons = [("normalized_history", prefix, dev["history"][:8]), ("normalized_current_C", prefix[:, -1], dev["raw_last"][:8]),
        ("encoded_past", latent, cached["latent_history"][:8]), ("decoder_FP32", arrays["decoder_deployed_FP32"], science_state["decoder_deployed_FP32"][:8]),
        ("decoder_bias_FP32", arrays["decoder_bias_FP32"], science_state["decoder_bias_FP32"][:8])]
    comparisons += [(key, np.asarray(metadata[key]), np.asarray(science_metadata[key])[:8]) for key in
        ("actual_lambda", "local_covariance_trace", "ridge_matrix_condition", "real_stationarity_residual_norm")]
    require(set(entry["checks"]) == {key for key, _, _ in comparisons}, "EXACT_NINE_PHYSICAL_RIDGE_NUMERIC_CHECKS")
    for key, actual, expected in comparisons:
        prior.compare_parity_record(entry["checks"][key], actual, expected, "physical_shrunk:" + key)
    decisions = ("true_observed_counts", "zero_trace_prior_slope_retained", "no_observation_prior_with_current_intercept", "past_metric_valid", "ridge_matrix_condition_valid")
    require(set(entry["exact_decisions"]) == {*decisions, "positive_lambda_solver_decision"}, "EXACT_SIX_RIDGE_PREFLIGHT_DECISIONS")
    for key in decisions:
        exact(np.asarray(metadata[key]), np.asarray(science_metadata[key])[:8], "physical_shrunk_exact_decision:" + key)
        require(entry["exact_decisions"][key] is True, "NO_OUTPUT_TOLERANCE_ONLY_RIDGE_DECISION_ACCEPTANCE")
    exact(np.asarray(metadata["actual_lambda"]) > 0, np.asarray(science_metadata["actual_lambda"])[:8] > 0, "physical_shrunk_positive_lambda_solver_decision")
    require(entry["exact_decisions"]["positive_lambda_solver_decision"] is True, "EXACT_POSITIVE_LAMBDA_DECISION_REQUIRED")
    known = {"history": prefix, "history_observed": physical["history_observed"][:8]}
    verified = shrink_audit.check_state({"state_metadata": metadata}, {key: arrays[key] for key in state_keys}, known, latent, prior32, rho)
    return {"physical_first8_independent_augmented_LS_ridge_recomputed": True, "known_past_only": True,
        "actual_original_prior_cast32": array_record(prior32), "rho": rho, "state_checks": verified,
        "six_decisions_verified_exact": list(entry["exact_decisions"]), "nine_numeric_parity_records_checked": 9,
        "output_tolerance_only_decision_acceptance_used": False,
        "decoder_FP32_max_abs_vs_CPU_science": float(np.abs(arrays["decoder_deployed_FP32"].astype(np.float64) - science_state["decoder_deployed_FP32"][:8].astype(np.float64)).max())}


def audit_cost_dataset(cost, physical, dev, predictions, metadata, check, *, sealed_unix):
    require(cost["status"] == "COMPLETE" and cost["routes"] == list(ROUTES)
            and cost["expected_cells_this_dataset"] == 10 and cost["expected_cells_two_datasets"] == 20
            and len(cost["rows"]) == 10 and cost["warmup_each_route"] == 2 and cost["repeats"] == 6,
            "COMPLETE_FIVE_BOLT_TWO_BATCH_COST_MATRIX")
    require(cost["new_training_updates"] == 0 and cost["encoder_refit_in_cost"] is False
            and cost["local_decoder_fit_inside_timed_callback"] is cost["implemented_local_fit_diagnostics_included"] is True
            and cost["fit_input_binding_hashes_inside_timer"] is True
            and cost["protected_TEST_accessed"] is cost["independent_holdout"] is False, "COST_NO_REFIT_OR_HOLDOUT")
    require(cost["recorded_timer_samples"] == cost["attempted_timer_calls"] == 60,
            "COMPLETE_ACTUAL_LOCAL_COST_SAVED_AND_ATTEMPTED_SAMPLE_COUNTS")
    normalized = prior.causal_physical_preprocess(physical["history_physical"], physical["history_observed"], dev["mean"], dev["std"])
    prior.close(normalized, dev["history"], "independent_causal_FIT_physical_preprocessing")
    exact(physical["history_observed"], dev["history_observed"], "original_past_observed_mask")
    if "origins" in physical:
        exact(physical["origins"], dev["origins"], "original_physical_bridge_origins")
    for key, value in (("physical_past", physical["history_physical"]), ("history_observed", physical["history_observed"]),
                       ("FIT_mean", dev["mean"]), ("FIT_std", dev["std"]), ("origins", dev["origins"])):
        require(cost["input_array_metadata"][key] == array_record(value), "EXACT_COST_INPUT_CONTENT:" + key)
    schedule = independent_schedule()
    require(cost["order_seed"] == 20261015 and cost["schedule"] == schedule, "PREDECLARED_FIVE_BALANCED_PLUS_EXTRA_SCHEDULE")
    require(set(cost["source_metadata"]) == set(ROUTES), "EXACT_FIVE_SCIENTIFIC_SOURCE_ROUTES")
    table = {(row["route"], row["batch"]): row for row in cost["rows"]}
    require(len(table) == 10 and set(table) == {(route, batch) for route in ROUTES for batch in (1, 8)}, "NO_MISSING_OR_DUPLICATE_COST_CELL")
    lineage = cost["source_lineage"]
    require(lineage["whole_cost_source_sha256"] == prior.file_sha(HERE / "whole_selected_shrinkage_cost_v1.py")
            and Path(lineage["copied_preprocess_source"]).resolve() == BASE / "cost_readout_v1.py"
            and lineage["copied_preprocess_source_sha256"] == prior.file_sha(BASE / "cost_readout_v1.py")
            and lineage["copied_functions"] == ["fill_vectorized", "preprocess"]
            and lineage["legacy_scientific_runtime_imported"] is False, "UNCHANGED_SOURCE_TIMING_PREPROCESS_LINEAGE")
    details = []
    chronological = []
    for (route, batch), row in table.items():
        check()
        require(row["status"] == "COMPLETE" and row["dataset"] == cost["dataset"], "COMPLETE_BOUND_COST_ROW")
        validate_route_metadata(route, cost["source_metadata"][route], metadata[route])
        validate_route_metadata(route, row["selected_metadata"], metadata[route])
        arithmetic = prior.audit_recorded_times(row, table[("RAW_BOLT", batch)])
        require(row["positions"] == [block.index(route) for block in schedule]
                and sorted(row["positions"][:5]) == list(range(5)), "FIVE_POSITION_BALANCE_SIXTH_EXTRA")
        require(row["origins"] == dev["origins"][:batch].tolist(), "EXACT_SAME_FIRST_B_ORIGINS")
        refs = prior.verified_arrays(row["reference"], check)
        timed = prior.verified_arrays(row["timed_outputs"], check)
        require(set(refs) == {"normalized", "physical", "scientific_normalized", "origins"}
                and set(timed) == {"normalized", "physical", "block", "position", "origins"}, "EXACT_COST_SAVED_KEYS")
        for arrays in (refs, timed):
            exact(arrays["origins"], dev["origins"][:batch], "cost_saved_first_B_origins")
        exact(timed["block"], np.arange(6, dtype=np.int64), "cost_repeat_numbers")
        exact(timed["position"], np.asarray(row["positions"], np.int64), "cost_actual_positions")
        shape = (batch, 48, len(dev["mean"]))
        require(all(refs[key].shape == shape and refs[key].dtype == np.float32 for key in ("normalized", "physical", "scientific_normalized"))
                and all(timed[key].shape == (6,) + shape and timed[key].dtype == np.float32 for key in ("normalized", "physical")),
                "SAVED_FP32_TIMED_AND_REFERENCE_DIMENSIONS")
        exact(refs["scientific_normalized"], predictions[route][:batch], "cost_accepted_probe_scientific_prediction")
        prior.close(refs["normalized"], refs["scientific_normalized"], "cost_reference_scientific_parity")
        exact(refs["physical"], (refs["normalized"] * dev["std"] + dev["mean"]).astype(np.float32), "cost_reference_actual_CPU_inverse")
        science_physical = (refs["scientific_normalized"] * dev["std"] + dev["mean"]).astype(np.float32)
        require(len(row["samples"]) == len(row["parity"]) == len(row["memory"]) == 6, "SIX_RECORDED_SAMPLES_PER_CELL")
        errors = []
        for repeat, sample in enumerate(row["samples"]):
            require(sample["repeat"] == sample["block"] == repeat and sample["route"] == route
                    and sample["batch"] == batch and sample["dataset"] == cost["dataset"]
                    and sample["position"] == row["positions"][repeat], "EXACT_ACTUAL_TIMER_SAMPLE_IDENTITY")
            recorded_timer(sample, row, repeat, sealed_unix)
            chronological.append((batch, repeat, sample["position"], sample))
            sample_path = Path(sample["sample_artifact"]["path"]).with_suffix(".json")
            literal = prior.verified_json({"path": str(sample_path), "sha256": prior.file_sha(sample_path)}, check)
            require(literal == sample, "EXACT_LITERAL_TIMER_JSON_BINDING")
            saved = prior.verified_arrays(sample["sample_artifact"], check)
            require(set(saved) == {"normalized", "physical", "origins"}, "EXACT_TIMED_SAMPLE_KEYS")
            exact(saved["origins"], dev["origins"][:batch], "cost_sample_original_origins")
            exact(saved["normalized"], timed["normalized"][repeat], "cost_sample_normalized_stack")
            exact(saved["physical"], timed["physical"][repeat], "cost_sample_physical_stack")
            exact(saved["physical"], (saved["normalized"] * dev["std"] + dev["mean"]).astype(np.float32), "cost_sample_actual_CPU_inverse")
            error = {}
            for key, actual, expected, atol, rtol in (
                    ("normalized", saved["normalized"], refs["normalized"], 1e-5, 2e-6),
                    ("physical", saved["physical"], refs["physical"], 2e-3, 2e-5),
                    ("scientific_normalized", saved["normalized"], refs["scientific_normalized"], 1e-5, 2e-6),
                    ("scientific_physical", saved["physical"], science_physical, 2e-3, 2e-5)):
                error[key] = prior.compare_parity_record(sample["parity"][key], actual, expected, "newD_cost:" + route + ":" + key, atol=atol, rtol=rtol)
            errors.append(error)
        require(row["memory_counter_reset_inside_module"] is row["stage_decomposition"] is row["native_head_loaded_or_timed"] is False,
                "WHOLE_RESIDENT_SHARED_MEMORY_NO_STAGE_OR_NATIVE_CLAIM")
        memory_summary = {key: {"min": min(value[key] for value in row["memory"]), "max": max(value[key] for value in row["memory"])}
                          for key in row["memory"][0]}
        details.append({"route": route, "batch": batch, **arithmetic, "recorded_shared_memory_range_bytes": memory_summary,
                        "independent_saved_output_parity": errors})
    chronological.sort(key=lambda value: value[:3])
    for previous, current in zip(chronological, chronological[1:]):
        require(current[3]["perf_start_ns"] >= previous[3]["perf_end_ns"], "SERIAL_RECORDED_TIMERS_NONOVERLAPPING")
        for key in ("peak_allocated_vram_bytes", "peak_reserved_vram_bytes"):
            require(current[3]["memory"][key] >= previous[3]["memory"][key], "UNRESET_CUMULATIVE_SHARED_VRAM_PEAKS")
    return {"rows": details, "cells_checked": 10, "recorded_timer_samples_checked": 60,
            "clock_memory_or_F_forward_independently_remeasured": False,
            "first_five_rounds_balanced_sixth_extra": True, "same_first_B_origins_repeated_not_independent_draws": True,
            "per_method_exclusive_memory_claimed": False}


def metadata_gates(cost_path, check):
    """No NumPy, NPZ, model, F or timing work before this completed metadata gate."""
    path = Path(cost_path).resolve()
    cost_record = {"path": str(path), "sha256": prior.file_sha(path)}
    cost = prior.verified_json(cost_record, check)
    if cost.get("status") not in ("COMPLETE", "PARTIAL_COMPLETED"):
        return cost_record, cost, {}, {}
    require(cost["routes"] == list(ROUTES) and cost["expected_cost_cells"] == 20
            and cost["new_training_updates"] == cost["projection_fits"] == 0,
            "REGISTERED_SELECTED_SHRINKAGE_WHOLE_MATRIX")
    for key in ("protected_TEST_accessed", "independent_holdout", "new_backbone_training", "new_native_head_forward",
        "native_deployment_cost_measured", "future_labels_used_for_shrunk_fit", "new_rho_or_method_selection_in_cost",
        "same_decoder_side_information_as_static", "strict_K_only_compression", "static_D_or_global_rankK_oracle_is_local_lower_bound",
        "local_coefficients_cached_inside_timed_pipeline", "native_heads_loaded_or_timed"):
        require(cost[key] is False, "SAME_PAST_ONLY_FROZEN_F_NO_NEW_SELECTION_COST_SCOPE:" + key)
    for key in ("inference_time_local_decoder_fits", "uses_preexisting_TUNE_Bolt_selection", "TUNE_future_labels_used_for_earlier_rho_selection",
        "past_C_condition_metadata_added", "local_ridge_and_implemented_diagnostics_inside_timer", "fit_input_binding_hashes_inside_timer",
        "source_file_state_metadata_hashes_outside_timed_callback"):
        require(cost[key] is True, "ACTUAL_RIDGE_AND_INPUT_HASH_BOUNDARIES:" + key)
    parents = {key: prior.verified_json(cost[key], check) for key in ("producer", "fit_target_audit", "shrinkage_diagnostic", "shrinkage_audit")}
    producer, fit_pass, shrink, shrink_pass = (parents[key] for key in ("producer", "fit_target_audit", "shrinkage_diagnostic", "shrinkage_audit"))
    require(producer["status"] == shrink["status"] == "COMPLETE" and fit_pass["status"] == shrink_pass["status"] == "PASS", "COMPLETE_EXACT_SCIENCE_AND_SHRINKAGE_PASSES_BEFORE_ARRAYS")
    require(cost["shrinkage_diagnostic"]["sha256"] == SHRINK_SHA and cost["shrinkage_audit"]["sha256"] == SHRINK_AUDIT_SHA, "EXACT_SEALED_SHRINKAGE_AND_INDEPENDENT_AUDIT")
    assert_binding(fit_pass["producer"], cost["producer"], "FIT producer")
    assert_binding(shrink_pass["diagnostic"], cost["shrinkage_diagnostic"], "shrinkage audit diagnostic")
    for key in ("producer", "fit_target_audit"):
        assert_binding(shrink[key], cost[key], "shrinkage cost " + key)
    for key in ("local_decoder_diagnostic", "local_decoder_audit", "producer", "fit_target_audit", "past_constant_diagnostic", "past_constant_audit", "past_constant_whole_cost", "past_constant_whole_cost_audit"):
        assert_binding(shrink_pass[key], shrink[key], "shrinkage sealed parent " + key)
        prior.verified_json(shrink[key], check)
    counts = {"dataset_split_cells_checked": 6, "RAW_control_cells_checked": 6, "positive_rho_backend_states_checked": 88,
        "window_fits_checked": 13504, "channel_slots_checked": 187664, "positive_lambda_solver_calls_checked": 187664,
        "zero_trace_prior_fallbacks_checked": 0, "no_observation_prior_fallbacks_checked": 0, "native_seed_D_reuses_checked": 6752,
        "FIT_TUNE_positive_rho_prediction_arrays_checked": 120, "selected_DEV_prediction_arrays_checked": 12,
        "reused_control_method_pairs_checked": 36, "reused_control_prediction_arrays_byte_preserved": 72,
        "selection_locks_checked": 4, "score_field_checks": 6960, "protected_prior_file_hashes_checked": 916}
    require(all(shrink_pass["totals"].get(key) == value for key, value in counts.items()), "FULL_SEALED_SHRINKAGE_AUDIT_COVERAGE")
    assert_binding(shrink_pass["audit_source"], {"path": str(HERE / "audit_local_decoder_shrinkage_v1.py"), "sha256": prior.file_sha(HERE / "audit_local_decoder_shrinkage_v1.py")}, "immutable independent shrinkage auditor")
    require(shrink["protected_TEST_accessed"] is shrink_pass["protected_TEST_accessed"] is False
            and shrink["future_labels_used_for_decoder_fit"] is shrink["native_specific_selection"] is shrink["development_positive_rho_sweep_executed"] is False
            and shrink["all_selection_lock_files_saved_before_any_DEV_candidate_compute"] is True,
            "KNOWN_PAST_FIT_AND_FOUR_PREDEV_BOLT_SELECTION_LOCKS")
    require(cost["selection_locks"] == shrink["selection_locks"] and set(cost["selection_locks"]) == set(DATASETS), "FOUR_ORIGINAL_SELECTION_LOCK_BINDINGS")
    registered = {}
    for dataset in DATASETS:
        require(set(cost["selection_locks"][dataset]) == {"PCA_PINV", "FIT_TARGET_PINV"}, "TWO_PROJECTION_LOCKS_PER_DATASET")
        registered[dataset] = {}
        for projection, item in cost["selection_locks"][dataset].items():
            lock = prior.verified_json(item, check)
            proof = shrink_pass["selection_locks"][dataset][projection]
            require(proof["record"] == item and proof["selected_candidate"] == lock["selected_candidate"]
                    and proof["selected_rho"] == lock["selected_rho"] and proof["recomputed_winner_equal_registered_winner"] is True,
                    "INDEPENDENTLY_AUDITED_ORIGINAL_SELECTED_RULE")
            label, rho, scores = lock["selected_candidate"], lock["selected_rho"], lock["TUNE_Bolt_normalized_MSE_by_candidate"]
            mapping = dict(zip(shrink_audit.RHO_LABELS, shrink_audit.RHOS))
            require(label in mapping and rho == mapping[label] and lock["status"] == "LOCKED"
                    and lock["selection_predictor"] == "BOLT_NUMPY" and lock["selection_split"] == "TUNE"
                    and lock["selection_metric"] == "normalized_macro_channel_MSE" and lock["native_specific_selection"] is False
                    and lock["development_or_oracle_scores_used_for_selection"] is False and lock["MAE_used_for_selection"] is False,
                    "SELECTED_POSITIVE_BOLT_TUNE_ONLY_RULE")
            require(set(scores) == set(shrink_audit.CANDIDATE_ORDER) and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in scores.values()), "SEVEN_FINITE_LOCKED_TUNE_SCORES")
            winner = shrink_audit.CANDIDATE_ORDER[0]
            for candidate in shrink_audit.CANDIDATE_ORDER[1:]:
                if scores[candidate] < scores[winner]: winner = candidate
            require(winner == label and lock["selected_TUNE_Bolt_normalized_MSE"] == scores[winner], "AUDIT_LOCK_WINNER_WITHOUT_NEW_METHOD_SELECTION")
            registered[dataset][projection] = {"record": item, "selected_candidate": label, "rho": float(rho)}
    require(cost["selected_rule_metadata"] == registered, "EXACT_LOCKED_RULES_USED_IN_COST")
    return cost_record, cost, parents, registered


def expected_route_metadata(original, development, selected_cell, selected_rules):
    metadata = {"RAW_BOLT": {"source_inputs": original["inputs"][development], "reference_prediction_array": "raw_forecast",
        "shared_resident_frozen_F": True, "local_fit": False, "native_head_loaded_or_timed": False}}
    for projection, registered in selected_rules.items():
        label, rho = registered["selected_candidate"], registered["rho"]
        cell = selected_cell["routes"][projection]
        backend, control = cell["positive_rho_candidates"][label]["backends"]["BOLT_NUMPY"], cell["control_backends"]["BOLT_NUMPY"]
        for route, shrunk in ((projection + "_BOLT", False), (projection.replace("_PINV", "_SHRUNK") + "_BOLT", True)):
            metadata[route] = {"source_state": original["projections"][projection]["state_artifact"],
                "selected_CPU_reference": backend["artifact"] if shrunk else control["artifact"],
                "reference_prediction_array": "BOLT_prediction" if shrunk else "BOLT_static_prediction", "local_fit": shrunk,
                "local_fit_uses_current_known_past_and_true_observed_mask": shrunk,
                "local_ridge_and_implemented_diagnostics_inside_timer": shrunk, "fit_input_binding_hashes_inside_timer": shrunk,
                "cached_local_D_used_for_timing": False, "rho": rho if shrunk else None, "selection_lock": registered["record"],
                "selected_candidate": label, "new_rho_selection_in_cost": False, "shared_resident_frozen_F": True,
                "new_E_F_or_head_training": False, "native_head_loaded_or_timed": False, "past_C_condition_metadata_added": shrunk,
                "same_decoder_side_information_as_static": not shrunk, "strict_K_only_compression": False}
    return metadata




def audit_result(cost_path, meter):
    check = meter.check
    path = Path(cost_path).resolve()
    record, cost, parents, registered = metadata_gates(path, check)
    if not parents:
        return {"status": "NOT_AUDITED_INCOMPLETE_PRODUCER", "cost": record, "producer_status": cost.get("status"), "scientific_assets_opened": False}
    complete = cost["status"] == "COMPLETE"
    producer, fit_pass, shrink, shrink_pass = (parents[key] for key in ("producer", "fit_target_audit", "shrinkage_diagnostic", "shrinkage_audit"))
    original_helper, helper, runner, math_source = (HERE / name for name in ("whole_local_past_decoder_cost_v1.py", "whole_selected_shrinkage_cost_v1.py", "run_selected_shrinkage_cost_v1.py", "local_decoder_shrinkage_v1.py"))
    source_checks = source_contract(*(item.read_text(encoding="utf-8-sig") for item in (original_helper, helper, runner, math_source)))
    assert_binding(cost["timing_helper_copy"]["original"], {"path": str(original_helper), "sha256": prior.file_sha(original_helper)}, "original_timing_helper")
    guarded = [record, *[cost[key] for key in parents], shrink_pass["audit_source"], fit_pass["audit_source"],
               *cost["source_records"], *cost["protected_prior_records"], *cost["source_snapshots"], *producer["source_snapshots"]]
    for item in guarded:
        check()
        require(prior.file_sha(item["path"]) == item["sha256"], "UNCHANGED_EXECUTED_LOCAL_COST_SOURCE_INPUT_CACHE_MODEL_CP_OR_SELECTION")
        if "snapshot_path" in item:
            require(prior.file_sha(item["snapshot_path"]) == item["sha256"], "EXACT_EXECUTED_LOCAL_COST_SOURCE_SNAPSHOT")
    local_math = [item for item in shrink["source_records"] if Path(item["path"]).name == "local_decoder_shrinkage_v1.py"]
    require(len(local_math) == 1 and local_math[0]["sha256"] == prior.file_sha(math_source), "SAME_REGISTERED_FULL_LOCAL_SOLVER_AND_DIAGNOSTICS_AS_CPU_SCIENCE")
    for key, expected in (("authorization", HERE / "AUTHORIZATION_SCOPE_v1.json"), ("continuation_manifest", HERE / "CONTINUATION_MANIFEST_v1.json")):
        assert_binding(cost[key], {"path": str(expected), "sha256": prior.file_sha(expected)}, key)
    require(cost["F_before"] == cost["F_after"] == producer["F_before"] == producer["F_after"] and set(cost["F_before"]) == {"old_algorithm", "bi_algorithm"}, "RECORDED_DUAL_FROZEN_BOLT_STATE_IDENTITY")
    for key in ("model_binary", "model_config"):
        assert_binding(cost[key + "_before"], cost[key + "_after"], key + "_before_after")
        assert_binding(cost[key + "_before"], producer[key + "_before"], key + "_accepted_original")
    for key in ("evaluation", "all_backbone_parameters_frozen", "context_length", "prediction_length", "input_patch_size", "input_patch_stride", "median_index", "returned_horizon", "quantiles", "pipeline_predict_not_used", "normalizer_before", "normalizer_after", "normalizer_fixture_forward_parity"):
        require(cost["runtime_behavior_contract"][key] == producer["runtime_behavior_contract"][key], "UNCHANGED_ACCEPTED_BOLT_BEHAVIOR:" + key)
    seal = prior.verified_json(producer["selection_lock"], check)
    require(seal["status"] == "SEALED_BEFORE_NEW_DEV_AND_ORACLE", "ORIGINAL_CHECKPOINT_SELECTION_LOCK_BEFORE_COST")
    guarded.append(producer["selection_lock"])
    receipt = prior.verified_json(cost["resource_terminal"], check)
    require(receipt["status"] == cost["status"] and receipt["elapsed_or_cumulative_budget_caps"] is None, "ACTUAL_SELECTED_SHRINKAGE_WHOLE_COST_TERMINAL_RECEIPT")
    require(set(cost["datasets"]) == set(DATASETS) and set(cost["cost"]).issubset(DATASETS), "REGISTERED_LOCAL_WHOLE_COST_DATASET_PATHS")
    report = {"status": "PASS" if complete else "PARTIAL_AUDITED_NO_FULL_PASS", "cost": record,
        **{key: cost[key] for key in parents}, "source_contract": source_checks, "datasets": {}, "cost_details": {}, "failed_paths": cost["failed_paths"],
        "totals": {"cost_cells_checked": 0, "recorded_timer_samples_checked": 0, "physical_shrunk_prechecks_checked": 0, "selected_rule_lock_files_checked": 4,
                   "physical_shrunk_first8_window_fits_checked": 0, "selected_forecast_file_hashes_checked": 0,
                   "canonical_physical_bridges_checked": 0, "protected_prior_file_hashes_checked": len(cost["protected_prior_records"])},
        "protected_TEST_accessed": False, "new_training_or_GPU_execution": False, "F_forward_or_timers_independently_replayed": False,
        "all_five_callbacks_use_same_frozen_Bolt": True, "native_heads_timed": False, "actual_selected_ridge_full_diagnostics_and_input_hashes_inside_timer": True,
        "exclusive_per_method_RAM_or_VRAM_claimed": False, "independent_generalization_claimed": False,
        "past_C_condition_metadata_added": True, "same_decoder_side_information_as_static": False, "strict_K_only_compression": False,
        "partial_cost_paths_not_accepted": [], "selection_locks": cost["selection_locks"], "selected_rule_metadata": registered,
        "new_rho_or_native_specific_selection_performed": False}
    for dataset, (channels, k, development, count, parent_tag) in DATASETS.items():
        row = cost["datasets"][dataset]
        if row["status"] != "COMPLETE":
            require(not complete, "COMPLETE_LOCAL_WHOLE_COST_HAS_FAILED_DATASET_PATH")
            report["partial_cost_paths_not_accepted"].append({"dataset": dataset, "metadata": row,
                "preserved_sample_count": cost.get("cost", {}).get(dataset, {}).get("recorded_timer_samples", 0)})
            continue
        original, cpu_cell = producer["datasets"][dataset], shrink["datasets"][dataset]["splits"][development]
        require(row["K"] == k and row["channels"] == channels and row["development_split"] == development
                and row["inputs"] == {key: original["inputs"][key] for key in ("fit", development)}
                and row["selected_CPU_cell_source"] == cpu_cell["input_source"], "EXACT_ACCEPTED_LOCAL_COST_K_C_DEVELOPMENT_INPUTS")
        fit, dev = (prior.verified_arrays(row["inputs"][key], check) for key in ("fit", development))
        prior.check_raw_split(dev, channels=channels, windows=count)
        for key in ("mean", "std"):
            exact(dev[key], fit[key], "same_FIT_only_Local_whole_scaler:" + key)
        parent_path = BASE / "runs" / parent_tag / "RESULT.json"
        parent = prior.verified_json({"path": str(parent_path), "sha256": prior.file_sha(parent_path)}, check)
        bridge = prior.verified_json(parent["bridge_provenance"]["input_bridge"], check)
        matches = [item for item in bridge["artifacts"] if item["dataset"] == dataset and item["split"] == development]
        require(len(matches) == 1 and row["physical_bridge"] == original["physical_bridge"], "ONE_EXACT_ORIGINAL_LOCAL_PHYSICAL_BRIDGE")
        canonical = prior.bind_canonical_npz(row["physical_bridge"], matches[0])
        physical = prior.verified_arrays(canonical, check)
        exact(physical["origins"], dev["origins"], "whole_Local_physical_original_origins")
        exact(physical["history_observed"], dev["history_observed"], "whole_Local_original_true_past_mask")
        predictions, prechecks = {"RAW_BOLT": dev["raw_forecast"]}, {}
        require(set(row["physical_shrunk_prechecks"]) == {"PCA_PINV", "FIT_TARGET_PINV"}, "BOTH_LOCAL_PHYSICAL_PREFLIGHTS_PRESENT")
        for projection in ("PCA_PINV", "FIT_TARGET_PINV"):
            condition, cell = original["projections"][projection], cpu_cell["routes"][projection]
            rule = registered[dataset][projection]
            label, rho = rule["selected_candidate"], rule["rho"]
            require(cell["selected_candidate"] == label and cell["selection_lock"] == rule["record"]
                    and cell["unselected_positive_rho_DEV_candidates_computed"] is False, "SAME_LOCK_SELECTED_DEVELOPMENT_CONDITION")
            backend = cell["positive_rho_candidates"][label]["backends"]["BOLT_NUMPY"]
            control_backend = cell["control_backends"]["BOLT_NUMPY"]
            state = prior.verified_arrays(condition["state_artifact"], check)
            cached = prior.verified_arrays(condition["bolt"]["artifacts"][development], check)
            science_state, references = (prior.verified_arrays(backend[key], check) for key in ("state_artifact", "artifact"))
            for key in ("origins", "raw_last", "target", "observed"):
                exact(references[key], dev[key], "CPU_science_Local_whole_same_" + key)
            exact(references["latent_history"], cached["latent_history"], "CPU_science_same_NumPy_E_history")
            exact(references["BOLT_cached_future"], cached["latent_future"], "CPU_science_same_cached_F_future")
            controls = prior.verified_arrays(control_backend["artifact"], check)
            exact(controls["BOLT_static_prediction"], cached["prediction"], "CPU_science_original_BOLT_bytes")
            expected_sources = {"source_shrunk_state": backend["state_artifact"], "source_shrunk_forecast": backend["artifact"],
                                "selection_lock": rule["record"], "selected_candidate": label,
                                "source_state": condition["state_artifact"], "source_BOLT_cache": condition["bolt"]["artifacts"][development]}
            entry = row["physical_shrunk_prechecks"][projection]
            arrays = prior.verified_arrays(entry["artifact"], check)
            prechecks[projection] = independent_physical_precheck(entry, arrays, physical, dev, state, cached, science_state, backend["state_metadata"], expected_sources, rho)
            literal_path = path.parent / (dataset + "_" + projection + "_PHYSICAL_SHRUNK_PRECHECK.json")
            require(prior.verified_json({"path": str(literal_path), "sha256": prior.file_sha(literal_path)}, check) == entry, "EXACT_LITERAL_LOCAL_PHYSICAL_PREFLIGHT_JSON")
            predictions[projection + "_BOLT"] = controls["BOLT_static_prediction"]
            predictions[projection.replace("_PINV", "_SHRUNK") + "_BOLT"] = references["BOLT_prediction"]
            report["totals"]["physical_shrunk_prechecks_checked"] += 1
            report["totals"]["physical_shrunk_first8_window_fits_checked"] += 8
        metadata = expected_route_metadata(original, development, cpu_cell, registered[dataset])
        require(row["selected_metadata"] == metadata, "EXACT_FIVE_LOCAL_BOLT_COST_SELECTED_REFERENCE_METADATA")
        details = audit_cost_dataset(cost["cost"][dataset], physical, dev, predictions, metadata, check, sealed_unix=seal["sealed_unix"])
        literal_path = path.parent / ("cost_" + dataset) / "RESULT.json"
        require(prior.verified_json({"path": str(literal_path), "sha256": prior.file_sha(literal_path)}, check) == cost["cost"][dataset], "EXACT_LITERAL_LOCAL_WHOLE_COST_RESULT")
        report["cost_details"][dataset] = details
        report["datasets"][dataset] = {"original_physical_bridge": canonical, "channels": channels, "K": k, "development_split": development, "physical_shrunk_prechecks": prechecks}
        report["totals"]["cost_cells_checked"] += details["cells_checked"]
        report["totals"]["recorded_timer_samples_checked"] += details["recorded_timer_samples_checked"]
        report["totals"]["canonical_physical_bridges_checked"] += 1
    for original in producer["datasets"].values():
        for condition in original["projections"].values():
            for group in (condition["bolt"], *condition["native"].values()):
                for item in group["artifacts"].values():
                    guarded.append(item)
                    report["totals"]["selected_forecast_file_hashes_checked"] += 1
    require(cost["source_cache_selection_SHA_unchanged"] is cost["original_selected_forecasts_and_locks_unchanged"] is True, "RECORDED_UNCHANGED_ORIGINAL_LOCAL_SOURCE_CACHE_SELECTION_LOCK")
    for item in guarded:
        check()
        require(prior.file_sha(item["path"]) == item["sha256"], "LOCAL_COST_AUDITED_INPUT_SOURCE_CHECKPOINT_CACHE_LOCK_HASH_UNCHANGED")
    if complete:
        require(not cost["failed_paths"] and cost["completed_cost_cells"] == 20 and cost["recorded_timer_samples"] == cost["completed_cell_timer_samples"] == 120
                and all(report["totals"][key] == value for key, value in {"cost_cells_checked": 20, "recorded_timer_samples_checked": 120,
                    "physical_shrunk_prechecks_checked": 4, "physical_shrunk_first8_window_fits_checked": 32,
                    "selected_forecast_file_hashes_checked": 36, "canonical_physical_bridges_checked": 2}.items()), "FULL_ACTUAL_LOCAL_WHOLE_COST_AND_PHYSICAL_PREFLIGHT_AUDIT_COUNTS")
    return report


def execute(cost_path, output_path):
    from continuation_runtime_v1 import Meter, write_new
    output = Path(output_path).resolve()
    require(not output.exists(), "FRESH_SELECTED_SHRINKAGE_COST_AUDIT_FILE")
    output.parent.mkdir(parents=True, exist_ok=True)
    meter = Meter("audit_selected_shrinkage_cost_" + output.stem, gpu=False)
    source = {"path": str(Path(__file__).resolve()), "sha256": prior.file_sha(__file__)}
    failure = None
    try:
        meter.protected_check()
        _, value, parents, _ = metadata_gates(cost_path, meter.check)
        if not parents:
            report = {"status": "NOT_AUDITED_INCOMPLETE_PRODUCER", "producer_status": value.get("status"), "scientific_assets_opened": False}
        else:
            initialize_numpy()
            sys.path.insert(0, str(ROOT / "research/smallk_failure_followup_20261014"))
            from prepare_inputs import install_protected_test_guard
            install_protected_test_guard()
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=2):
                report = audit_result(cost_path, meter)
        meter.protected_check()
        require(prior.file_sha(__file__) == source["sha256"], "INDEPENDENT_SELECTED_COST_AUDIT_SOURCE_UNCHANGED")
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
    parser.add_argument("--cost", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    raise SystemExit(execute(arguments.cost, arguments.output))


