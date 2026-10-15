"""Independent physical/ridge/CPU-head and recorded whole-cost audit, with no F replay."""
from pathlib import Path
import argparse
import ast
import json
import math
import statistics
import sys

import audit_projection_results_v1 as prior
import audit_decoder_metric_cost_v1 as timed_audit
import audit_local_decoder_shrinkage_v1 as shrink_audit
import audit_matched_shrunk_native_v1 as matched_audit

HERE, BASE, ROOT = prior.HERE, prior.BASE, prior.ROOT
DATASETS = prior.DATASETS
SEEDS = (92601, 92602)
ROUTES = ("RAW_BOLT", "PCA_SHRUNK_BOLT", "FIT_TARGET_SHRUNK_BOLT", "PCA_SHRUNK_NATIVE_s92601",
          "PCA_SHRUNK_NATIVE_s92602", "FIT_TARGET_SHRUNK_NATIVE_s92601", "FIT_TARGET_SHRUNK_NATIVE_s92602")
PARENT_KEYS = ("matched_native_diagnostic", "matched_native_audit", "shrinkage_diagnostic", "shrinkage_audit", "producer", "fit_target_audit")
MATCHED_SHA = "e13587c86a9f28b3dccc5bf445cb85cd855a8a20a0ce1d9caad7dd57b93ba77a"
MATCHED_AUDIT_SHA = "0eb57a67b2ee3c25a7a5309efbdb62e55f756fb0c9552bd251b50f2ea6505077"
MATCHED_COUNTS = {"native_fits_checked": 8, "training_updates_logged_checked": 4800, "checkpoint_states_checked": 40,
    "checkpoint_TUNE_prediction_arrays_checked": 40, "selected_FIT_TUNE_DEV_prediction_arrays_checked": 24,
    "source_backend_contexts_checked": 12, "preserved_control_prediction_arrays_checked": 48,
    "persistence_baseline_contexts_computed": 12, "unique_persistence_split_cells_computed": 6,
    "selection_locks_checked": 8, "resolved_registry_conditions_checked": 8, "future_label_perturbation_scope_checks": 8,
    "score_field_checks": 2064, "protected_prior_file_hashes_checked": 1129}
require, exact = prior.require, shrink_audit.exact
assert_binding, array_record, recorded_timer = timed_audit.assert_binding, timed_audit.array_record, timed_audit.recorded_timer
np = sys.modules.get("numpy")


def initialize_numpy():
    global np
    timed_audit.initialize_numpy(); shrink_audit.initialize_numpy(); matched_audit.initialize_numpy()
    np = sys.modules["numpy"]


def independent_schedule():
    permutation = np.random.default_rng(20261015).permutation(7)
    return [[ROUTES[index] for index in np.roll(permutation, -repeat)] for repeat in range(7)]


def recorded_times(row, raw):
    seconds, baseline = row["seconds"], raw["seconds"]
    require(len(seconds) == len(baseline) == 7 and all(math.isfinite(value) and value > 0 for value in seconds + baseline), "SEVEN_FINITE_POSITIVE_TIMES")
    require(row["batch"] == raw["batch"] and row["batch"] in (1, 8) and row["repeats"] == 7 and row["warmup_each_route"] == 2, "REGISTERED_SEVEN_REPEATS_AND_TWO_WARMUPS")
    ratios = [value / base for value, base in zip(seconds, baseline)]
    values = {"median_seconds": statistics.median(seconds), "median_seconds_per_window": statistics.median(seconds) / row["batch"], "paired_RAW_median_ratio": statistics.median(ratios)}
    for name, value in values.items(): prior.close(row[name], value, "matched_cost:" + name, atol=1e-12, rtol=1e-12)
    require(row["paired_RAW_seconds"] == baseline, "EXACT_PAIRED_RAW_TIMES")
    prior.close(row["paired_RAW_ratios"], ratios, "matched_cost_paired_ratios", atol=1e-12, rtol=1e-12)
    return {**values, "samples_checked": 7, "independently_remeasured_clock_or_memory": False}


def functions(text):
    return {node.name: node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)}


def timed_sequence(node):
    found = []
    for loop in ast.walk(node):
        if not isinstance(loop, ast.For): continue
        names = [item.targets[0].id if isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name) else None for item in loop.body]
        if "perf_start_ns" in names and "perf_end_ns" in names:
            found.append([ast.dump(item, include_attributes=False) for item in loop.body[names.index("perf_start_ns"):names.index("perf_end_ns") + 1]])
    require(len(found) == 1, "ONE_RECORDED_WHOLE_PERF_TIMER")
    return found[0]


def source_contract(original_text, helper_text, runner_text, math_text, first_runner_text):
    previous, helper, callbacks, solver = map(functions, (original_text, helper_text, runner_text, math_text))
    for name in ("sha", "array_record", "_jsonable", "write_json_once", "save_outputs", "fill_vectorized", "preprocess", "parity", "whole_predict"):
        require(ast.dump(previous[name], include_attributes=False) == ast.dump(helper[name], include_attributes=False), "UNCHANGED_TIMED_IO_AND_PREPROCESS:" + name)
    require(timed_sequence(previous["measure_cost"]) == timed_sequence(helper["measure_cost"]), "PREPROCESS_CALLBACK_MATERIALIZED_CPU_RETURN_FINAL_SYNC_INSIDE_TIMER")
    constants = {}
    names = {"ROUTES", "ORDER_SEED", "WARMUP", "REPEATS", "NORM_ATOL", "NORM_RTOL", "PHYSICAL_ATOL", "PHYSICAL_RTOL"}
    for node in ast.parse(helper_text).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1: continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in names: constants[target.id] = ast.literal_eval(node.value)
        elif isinstance(target, ast.Tuple) and all(isinstance(item, ast.Name) and item.id in names for item in target.elts):
            constants.update(zip((item.id for item in target.elts), ast.literal_eval(node.value)))
    require(constants == {"ROUTES": ROUTES, "ORDER_SEED": 20261015, "WARMUP": 2, "REPEATS": 7, "NORM_ATOL": 1e-5, "NORM_RTOL": 2e-6, "PHYSICAL_ATOL": 2e-3, "PHYSICAL_RTOL": 2e-5}, "EXACT_SEVEN_ROUTE_ORDER_SEED_REPEAT_TOLERANCE_CONTRACT")
    require(ast.unparse(helper["balanced_schedule"]) == "def balanced_schedule(seed=ORDER_SEED):\n    rng = np.random.default_rng(seed)\n    order = rng.permutation(len(ROUTES))\n    schedule = [[ROUTES[index] for index in np.roll(order, -repeat)] for repeat in range(len(ROUTES))]\n    return schedule", "FULL_SEVEN_CIRCULAR_LATIN_SCHEDULE_SOURCE")
    expected = {"bolt_predictor": ["z = state.encode(history)", "fitted = fit_shrunk_local_decoder(history, history_observed, z, prior, rho)", "return fitted.forecast(history[:, -1], frozen_F(z), z[:, -1])"],
                "native_predictor": ["z = native_latents(projection, history)", "fitted = fit_shrunk_local_decoder(history, history_observed, z, prior32, rho)", "return predict_cached_latents(model, z, history[:, -1], fitted.decoder32, batch_size=32)"]}
    first_callbacks = functions(first_runner_text)
    for name, body in expected.items():
        inner = [node for node in callbacks[name].body if isinstance(node, ast.FunctionDef)]
        require(len(inner) == 1 and [arg.arg for arg in inner[0].args.args] == ["history", "history_observed"] and [ast.unparse(node) for node in inner[0].body] == body, "FRESH_PAST_RIDGE_EACH_CALLBACK_NO_SHARED_OR_CACHED_D:" + name)
        require(ast.dump(callbacks[name], include_attributes=False) == ast.dump(first_callbacks[name], include_attributes=False), "V2_BRIDGE_REPAIR_DID_NOT_CHANGE_CALLBACK:" + name)
    require(ast.dump(callbacks["native_latents"], include_attributes=False) == ast.dump(first_callbacks["native_latents"], include_attributes=False), "CANONICAL_TORCH_CPU_PAST_ENCODING_UNCHANGED")
    fit = ast.unparse(solver["fit_shrunk_local_decoder"])
    require("np.linalg.solve(matrix, linear - gram @ d[window, :, channel])" in fit and "hashlib.sha256(value.tobytes())" in fit and "np.linalg.svd(matrix.astype(np.float64), compute_uv=False)" in fit and "'real_stationarity_residual_norm'" in fit, "RIDGE_FULL_IMPLEMENTED_DIAGNOSTICS_AND_INPUT_HASHES_IN_EVERY_CALLBACK")
    require("source/file/state metadata hashes generated outside timed callback" in helper_text, "STATE_FILE_METADATA_SAVE_OUTSIDE_TIMER")
    return {"unchanged_registered_tolerance": True, "canonical_seven_round_schedule": True, "no_timed_decoder_cache_or_seed_sharing": True, "full_ridge_diagnostics_and_fit_input_hashes_inside_each_callback": True, "source_file_state_metadata_hashes_outside_timer": True, "independent_F_or_timer_replay": False}


def native_encode(past, state):
    import torch
    torch.set_num_threads(2)
    center, encoder = (torch.from_numpy(np.ascontiguousarray(state[name].astype(np.float32))) for name in ("center", "encoder"))
    pieces = []
    with torch.no_grad():
        for start in range(0, len(past), 32):
            value = torch.from_numpy(np.ascontiguousarray(past[start:start + 32]))
            pieces.append(((value - center) @ encoder).numpy().copy())
    return np.ascontiguousarray(np.concatenate(pieces), dtype=np.float32)


def independent_physical_precheck(entry, arrays, physical, dev, state, science_state, scientific, science_metadata, expected_sources, rho, backend, head_states, predictions):
    require(entry["status"] == "PASS" and entry["backend"] == backend and entry["rho"] == rho and entry["lambda_rule"] == "rho*trace(observed_centered_Z.T@Z/n)/K" and all(entry[key] == value for key, value in expected_sources.items()), "EXACT_MATCHED_PHYSICAL_PREFLIGHT_SOURCES")
    require(entry["history_observed_byte_exact"] is True and entry["decision_acceptance_by_output_tolerance_only"] is entry["coefficient_tolerance_loosened"] is False and entry["origins"] == dev["origins"][:8].tolist(), "EXACT_PREFLIGHT_DECISIONS_MASK_ORIGINS_NOT_TOLERANCE_ONLY")
    keys = ("decoder_fit_FP64", "decoder_deployed_FP32", "decoder_bias_fit_FP64", "decoder_bias_FP32", "metadata_json")
    expected_keys = {*keys, "preprocessed_history", "latent_history", "history_observed", "origins", "prior_decoder_FP32"}
    if backend == "NATIVE_TORCH32": expected_keys.update("MATCHED_NATIVE_s" + str(seed) + "_prediction" for seed in SEEDS)
    require(set(arrays) == expected_keys, "EXACT_PHYSICAL_PREFLIGHT_ARRAY_KEYS")
    prefix = prior.causal_physical_preprocess(physical["history_physical"][:8], physical["history_observed"][:8], dev["mean"], dev["std"])
    z = prior.encode(prefix, state["center"], state["encoder"]) if backend == "BOLT_NUMPY" else native_encode(prefix, state)
    prior32 = np.ascontiguousarray(state["decoder"].astype(np.float32))
    for key, value in (("preprocessed_history", prefix), ("latent_history", z), ("prior_decoder_FP32", prior32), ("history_observed", dev["history_observed"][:8]), ("origins", dev["origins"][:8])): exact(arrays[key], value, "physical_matched:" + key)
    metadata = entry["fitted_shrunk_metadata"]
    require(metadata["rho"] == science_metadata["rho"] == rho and metadata["slope_singular_cutoff_used"] is science_metadata["slope_singular_cutoff_used"] is False and metadata["fit_input_bindings"]["prior_decoder"] == array_record(prior32), "ORIGINAL_PRIOR_RHO_NO_CUTOFF_BINDINGS")
    comparisons = [("normalized_history", prefix, dev["history"][:8]), ("normalized_current_C", prefix[:, -1], dev["raw_last"][:8]), ("encoded_past", z, scientific["latent_history"][:8]), ("decoder_FP32", arrays["decoder_deployed_FP32"], science_state["decoder_deployed_FP32"][:8]), ("decoder_bias_FP32", arrays["decoder_bias_FP32"], science_state["decoder_bias_FP32"][:8])]
    comparisons += [(key, np.asarray(metadata[key]), np.asarray(science_metadata[key])[:8]) for key in ("actual_lambda", "local_covariance_trace", "ridge_matrix_condition", "real_stationarity_residual_norm")]
    require(set(entry["checks"]) == {key for key, _, _ in comparisons}, "NINE_REGISTERED_PHYSICAL_CHECKS")
    for key, actual, expected in comparisons: prior.compare_parity_record(entry["checks"][key], actual, expected, "matched_physical:" + key)
    decisions = ("true_observed_counts", "zero_trace_prior_slope_retained", "no_observation_prior_with_current_intercept", "past_metric_valid", "ridge_matrix_condition_valid")
    require(set(entry["exact_decisions"]) == {*decisions, "positive_lambda_solver_decision"}, "SIX_EXACT_PHYSICAL_SOLVER_DECISIONS")
    for key in decisions:
        exact(np.asarray(metadata[key]), np.asarray(science_metadata[key])[:8], "physical_matched_decision:" + key)
        require(entry["exact_decisions"][key] is True, "REGISTERED_EXACT_DECISION_PASS:" + key)
    exact(np.asarray(metadata["actual_lambda"]) > 0, np.asarray(science_metadata["actual_lambda"])[:8] > 0, "same_positive_lambda_decision")
    require(entry["exact_decisions"]["positive_lambda_solver_decision"] is True, "POSITIVE_LAMBDA_EXACT")
    checked = shrink_audit.check_state({"state_metadata": metadata}, {key: arrays[key] for key in keys}, {"history": prefix, "history_observed": physical["history_observed"][:8]}, z, prior32, rho)
    expected_heads = {str(seed) for seed in SEEDS} if backend == "NATIVE_TORCH32" else set()
    require(set(entry["matched_native_head_prediction_checks"]) == expected_heads, "EXACT_SELECTED_NATIVE_PREFLIGHT_HEADS")
    for seed in expected_heads:
        prediction, _ = matched_audit.independent_head_forward({"latent_history": z, "raw_last": prefix[:, -1], "decoder": arrays["decoder_deployed_FP32"]}, head_states[seed])
        exact(prediction, arrays["MATCHED_NATIVE_s" + seed + "_prediction"], "physical_independent_matched_head_bytes:" + seed)
        prior.compare_parity_record(entry["matched_native_head_prediction_checks"][seed], prediction, predictions[seed][:8], "physical_matched_head_scientific:" + seed)
    return {"state_checks": checked, "backend": backend, "rho": rho, "six_exact_decisions_checked": True, "nine_numeric_checks_checked": True, "native_first8_prediction_checks": len(expected_heads), "independent_F_forward_performed": False}


def validate_route_metadata(route, actual, expected):
    require(actual == expected, "EXACT_SEVEN_ROUTE_SELECTED_METADATA:" + route)
    native = "_NATIVE_s" in route
    require(actual["native_head_loaded_or_timed"] == native and actual["local_fit"] == (route != "RAW_BOLT"), "NATIVE_VS_BOLT_ROUTE_ROLE")
    if native:
        require(actual["head_device"] == "cpu" and actual["head_threads"] == 2 and actual["canonical_prediction_batch_size"] == 32 and actual["F_forward_in_this_route"] is False and actual["native_seed_D_cache_shared_inside_timer"] is False, "NATIVE_ACTUAL_CPU2_CANONICAL32_NO_F_OR_D_SHARING")
    else: require(actual["shared_resident_frozen_F"] is True, "FROZEN_BOLT_ROUTE")
    if route != "RAW_BOLT": require(actual["cached_local_D_used_for_timing"] is False and actual["fit_input_binding_hashes_inside_timer"] is True, "FRESH_LOCAL_RIDGE_INSIDE_TIMER")


def audit_cost_dataset(cost, physical, dev, predictions, metadata, check, *, sealed_unix):
    require(cost["status"] == "COMPLETE" and cost["routes"] == list(ROUTES)
            and cost["expected_cells_this_dataset"] == 14 and cost["expected_cells_two_datasets"] == 28
            and len(cost["rows"]) == 14 and cost["warmup_each_route"] == 2 and cost["repeats"] == 7,
            "COMPLETE_SEVEN_BOLT_NATIVE_TWO_BATCH_COST_MATRIX")
    require(cost["new_training_updates"] == 0 and cost["encoder_refit_in_cost"] is False
            and cost["local_decoder_fit_inside_timed_callback"] is cost["implemented_local_fit_diagnostics_included"] is True
            and cost["fit_input_binding_hashes_inside_timer"] is True
            and cost["protected_TEST_accessed"] is cost["independent_holdout"] is False, "COST_NO_REFIT_OR_HOLDOUT")
    require(cost["recorded_timer_samples"] == cost["attempted_timer_calls"] == 98,
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
    require(cost["order_seed"] == 20261015 and cost["schedule"] == schedule, "PREDECLARED_SEVEN_FULL_LATIN_SCHEDULE")
    require(set(cost["source_metadata"]) == set(ROUTES), "EXACT_SEVEN_SCIENTIFIC_SOURCE_ROUTES")
    table = {(row["route"], row["batch"]): row for row in cost["rows"]}
    require(len(table) == 14 and set(table) == {(route, batch) for route in ROUTES for batch in (1, 8)}, "NO_MISSING_OR_DUPLICATE_COST_CELL")
    lineage = cost["source_lineage"]
    require(lineage["whole_cost_source_sha256"] == prior.file_sha(HERE / "whole_matched_shrunk_native_cost_v1.py")
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
        arithmetic = recorded_times(row, table[("RAW_BOLT", batch)])
        require(row["positions"] == [block.index(route) for block in schedule]
                and sorted(row["positions"]) == list(range(7)), "SEVEN_POSITION_FULL_BALANCE")
        require(row["origins"] == dev["origins"][:batch].tolist(), "EXACT_SAME_FIRST_B_ORIGINS")
        refs = prior.verified_arrays(row["reference"], check)
        timed = prior.verified_arrays(row["timed_outputs"], check)
        require(set(refs) == {"normalized", "physical", "scientific_normalized", "origins"}
                and set(timed) == {"normalized", "physical", "block", "position", "origins"}, "EXACT_COST_SAVED_KEYS")
        for arrays in (refs, timed):
            exact(arrays["origins"], dev["origins"][:batch], "cost_saved_first_B_origins")
        exact(timed["block"], np.arange(7, dtype=np.int64), "cost_repeat_numbers")
        exact(timed["position"], np.asarray(row["positions"], np.int64), "cost_actual_positions")
        shape = (batch, 48, len(dev["mean"]))
        require(all(refs[key].shape == shape and refs[key].dtype == np.float32 for key in ("normalized", "physical", "scientific_normalized"))
                and all(timed[key].shape == (7,) + shape and timed[key].dtype == np.float32 for key in ("normalized", "physical")),
                "SAVED_FP32_TIMED_AND_REFERENCE_DIMENSIONS")
        exact(refs["scientific_normalized"], predictions[route][:batch], "cost_accepted_probe_scientific_prediction")
        prior.close(refs["normalized"], refs["scientific_normalized"], "cost_reference_scientific_parity")
        exact(refs["physical"], (refs["normalized"] * dev["std"] + dev["mean"]).astype(np.float32), "cost_reference_actual_CPU_inverse")
        science_physical = (refs["scientific_normalized"] * dev["std"] + dev["mean"]).astype(np.float32)
        require(len(row["samples"]) == len(row["parity"]) == len(row["memory"]) == 7, "SEVEN_RECORDED_SAMPLES_PER_CELL")
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
        require(row["memory_counter_reset_inside_module"] is row["stage_decomposition"] is False and row["native_head_loaded_or_timed"] == ("_NATIVE_s" in route),
                "WHOLE_RESIDENT_SHARED_MEMORY_NO_EXCLUSIVE_STAGE_CLAIM")
        memory_summary = {key: {"min": min(value[key] for value in row["memory"]), "max": max(value[key] for value in row["memory"])}
                          for key in row["memory"][0]}
        details.append({"route": route, "batch": batch, **arithmetic, "recorded_shared_memory_range_bytes": memory_summary,
                        "independent_saved_output_parity": errors})
    chronological.sort(key=lambda value: value[:3])
    for previous, current in zip(chronological, chronological[1:]):
        require(current[3]["perf_start_ns"] >= previous[3]["perf_end_ns"], "SERIAL_RECORDED_TIMERS_NONOVERLAPPING")
        for key in ("peak_allocated_vram_bytes", "peak_reserved_vram_bytes"):
            require(current[3]["memory"][key] >= previous[3]["memory"][key], "UNRESET_CUMULATIVE_SHARED_VRAM_PEAKS")
    return {"rows": details, "cells_checked": 14, "recorded_timer_samples_checked": 98,
            "clock_memory_or_F_forward_independently_remeasured": False,
            "all_seven_rounds_balanced": True, "same_first_B_origins_repeated_not_independent_draws": True,
            "per_method_exclusive_memory_claimed": False}


def metadata_gates(cost_path, check):
    path = Path(cost_path).resolve()
    record = {"path": str(path), "sha256": prior.file_sha(path)}
    cost = prior.verified_json(record, check)
    if cost.get("status") not in ("COMPLETE", "PARTIAL_COMPLETED"): return record, cost, {}
    require(cost["routes"] == list(ROUTES) and cost["expected_cost_cells"] == 28 and cost["new_training_updates"] == 0, "REGISTERED_SEVEN_ROUTE_COST_COHORT")
    for key in ("protected_TEST_accessed", "independent_holdout", "new_backbone_training", "new_encoder_fit", "new_rho_head_seed_or_method_selection_in_cost", "local_coefficients_cached_inside_timed_pipeline", "native_seed_D_cache_shared_inside_timer", "same_Bolt_native_lambda_or_D_bytes_claimed", "strict_K_only_compression", "static_D_or_global_rankK_oracle_is_local_lower_bound"):
        require(cost[key] is False, "UNCHANGED_PAST_ONLY_SELECTION_AND_CACHE_SCOPE:" + key)
    for key in ("local_ridge_and_implemented_diagnostics_inside_timer", "fit_input_binding_hashes_inside_timer", "source_file_state_metadata_hashes_outside_timed_callback", "past_C_condition_metadata_added"):
        require(cost[key] is True, "ACTUAL_FRESH_RIDGE_TIMER_BOUNDARY:" + key)
    parents = {key: prior.verified_json(cost[key], check) for key in PARENT_KEYS}
    for diagnostic, audit in (("matched_native_diagnostic", "matched_native_audit"), ("shrinkage_diagnostic", "shrinkage_audit"), ("producer", "fit_target_audit")):
        require(parents[diagnostic]["status"] == "COMPLETE" and parents[audit]["status"] == "PASS", "COMPLETED_DIAGNOSTIC_AND_EXACT_PASS:" + diagnostic)
        assert_binding(parents[audit]["producer" if diagnostic == "producer" else "diagnostic"], cost[diagnostic], "cost parent " + diagnostic)
    matched, accepted = parents["matched_native_diagnostic"], parents["matched_native_audit"]
    require(cost["matched_native_diagnostic"]["sha256"] == MATCHED_SHA and cost["matched_native_audit"]["sha256"] == MATCHED_AUDIT_SHA and all(accepted["totals"].get(key) == value for key, value in MATCHED_COUNTS.items()), "EXACT_FULL_MATCHED_NATIVE_INDEPENDENT_PASS")
    assert_binding(accepted["audit_source"], {"path": str(HERE / "audit_matched_shrunk_native_v1.py"), "sha256": prior.file_sha(HERE / "audit_matched_shrunk_native_v1.py")}, "immutable matched auditor")
    for key in ("shrinkage_diagnostic", "shrinkage_audit", "producer", "fit_target_audit"):
        assert_binding(matched[key], cost[key], "unchanged matched parent " + key)
        assert_binding(accepted[key], cost[key], "matched audit parent " + key)
    require(cost["selection_locks"] == matched["condition_selection_locks"] and cost["all_native_selection_registry"] == matched["selection_lock"] and cost["old_Bolt_rho_selection_locks"] == parents["shrinkage_diagnostic"]["selection_locks"], "UNCHANGED_EIGHT_CP_AND_FOUR_RHO_LOCK_BINDINGS")
    registry = prior.verified_json(matched["selection_lock"], check)
    require(registry["complete_fit_tune_conditions"] == 8, "ALL_EIGHT_MATCHED_LOCKS_RESOLVED")
    for dataset in DATASETS:
        for route in ("PCA_PINV", "FIT_TARGET_PINV"):
            old = prior.verified_json(cost["old_Bolt_rho_selection_locks"][dataset][route], check)
            condition = matched["datasets"][dataset]["routes"][route]
            require(condition["rho"] == old["selected_rho"] and condition["selected_candidate"] == old["selected_candidate"], "ORIGINAL_SHARED_BOLT_TUNE_RHO")
            for seed in SEEDS:
                binding = cost["selection_locks"][dataset][route][str(seed)]
                lock = prior.verified_json(binding, check)
                head = condition["native"][str(seed)]
                require(lock["status"] == "LOCKED" and lock["selected_update"] == head["metadata"]["selected_update"] and lock["selected_state_artifact"] == next(item for item in head["checkpoints"] if item["update"] == lock["selected_update"])["state_artifact"], "SAME_ACCEPTED_SELECTED_HEAD_STATE_AND_CP")
    if cost["status"] == "COMPLETE": require(cost["native_deployment_cost_measured"] is cost["native_heads_loaded_or_timed"] is True and cost["loaded_native_heads_count"] == 8, "ACTUAL_NATIVE_CPU_WHOLE_COST_MEASURED")
    if "prior_failed_cost_attempt" in cost:
        failure = prior.verified_json(cost["prior_failed_cost_attempt"], check)
        require(cost["prior_failed_cost_attempt"]["sha256"] == "6940e169e2acaa3c27b08c1c80e1e8ae2ddcef944a100af331629165fff09ea9" and failure["status"] == "PARTIAL_COMPLETED" and failure["completed_cost_cells"] == failure["recorded_timer_samples"] == 0 and len(failure["failed_paths"]) == 2, "PRESERVED_ZERO_SAMPLE_V1_SCHEMA_FAILURE")
        for row in failure["failed_paths"]: require(row["failure"] == {"type": "KeyError", "message": "'arrays'"}, "EXACT_ARRAYLESS_V1_FAILURE")
        prior.verified_json(cost["input_schema_failure_identity"], check)
    return record, cost, parents


def expected_route_metadata(original, development, matched_condition, shrink_cell, locks, lock_records):
    output = {"RAW_BOLT": {"source_inputs": original["inputs"][development], "reference_prediction_array": "raw_forecast", "shared_resident_frozen_F": True, "native_head_loaded_or_timed": False, "local_fit": False}}
    for projection in ("PCA_PINV", "FIT_TARGET_PINV"):
        condition = matched_condition["routes"][projection]
        label, rho = condition["selected_candidate"], condition["rho"]
        backend = shrink_cell["routes"][projection]["positive_rho_candidates"][label]["backends"]["BOLT_NUMPY"]
        short = projection.replace("_PINV", "_SHRUNK")
        output[short + "_BOLT"] = {"source_fixed_E": original["projections"][projection]["state_artifact"], "selected_CPU_reference": backend["artifact"], "reference_prediction_array": "BOLT_prediction", "backend": "BOLT_NUMPY", "rho": rho, "old_Bolt_rho_selection_lock": condition["old_Bolt_rho_selection_lock"], "native_head_loaded_or_timed": False, "shared_resident_frozen_F": True, "local_fit": True, "cached_local_D_used_for_timing": False, "fit_input_binding_hashes_inside_timer": True}
        for seed in SEEDS:
            lock = locks[(projection, str(seed))]
            output[short + "_NATIVE_s" + str(seed)] = {"source_fixed_E": original["projections"][projection]["state_artifact"], "selected_CPU_reference": condition["native"][str(seed)]["artifacts"][development], "reference_prediction_array": "prediction", "selected_head_state": lock["selected_state_artifact"], "selected_head_lock": lock_records[projection][str(seed)], "selected_update": lock["selected_update"], "native_seed": seed, "backend": "NATIVE_TORCH32", "rho": rho, "head_device": "cpu", "head_threads": 2, "canonical_prediction_batch_size": 32, "native_head_loaded_or_timed": True, "F_forward_in_this_route": False, "local_fit": True, "cached_local_D_used_for_timing": False, "native_seed_D_cache_shared_inside_timer": False, "fit_input_binding_hashes_inside_timer": True, "old_Bolt_rho_selection_lock": condition["old_Bolt_rho_selection_lock"]}
    return output


def verify_canonical_bridge(cost, dataset, original, development, check):
    parent_binding = cost["canonical_physical_bridge_bindings"][dataset]["original_first_parent"]
    assert_binding(parent_binding, original["source_parent_results"][dataset], "physical original first parent")
    parent = prior.verified_json(parent_binding, check)
    bridge = prior.verified_json(parent["bridge_provenance"]["input_bridge"], check)
    require(parent["status"] == bridge["status"] == "COMPLETE" and parent["protected_TEST_accessed"] is bridge["actual_status"]["protected_test_opened"] is False, "NO_TEST_COMPLETE_ORIGINAL_PHYSICAL_BRIDGE_CHAIN")
    matches = [item for item in bridge["artifacts"] if item["dataset"] == dataset and item["split"] == development]
    require(len(matches) == 1, "ONE_CANONICAL_PHYSICAL_BRIDGE")
    row = cost["datasets"][dataset]
    alias = original["datasets"][dataset]["physical_bridge"]
    canonical = prior.bind_canonical_npz(alias, matches[0])
    binding = cost["canonical_physical_bridge_bindings"][dataset]
    require(binding == row["physical_bridge_binding"] and row["physical_bridge"] == alias and row["physical_bridge_canonical"] == binding["canonical_record"] and binding["original_alias"] == alias, "SAME_CANONICAL_AND_ARRAYLESS_ALIAS_FILE_AND_SHA")
    assert_binding(binding["canonical_record"], canonical, "canonical physical rebound")
    require(binding["canonical_record"]["arrays"] == matches[0]["arrays"] and binding["same_physical_file_path_and_SHA"] is binding["complete_array_manifest_preserved"] is True and binding["loader_file_SHA_member_shape_dtype_SHA_checks_relaxed"] is binding["numeric_tolerance_changed"] is False, "NO_LOADER_MANIFEST_OR_NUMERIC_RELAXATION")
    for name in ("history", "history_observed", "target", "observed", "raw_last", "origins", "mean", "std"):
        require(canonical["arrays"][name] == original["datasets"][dataset]["inputs"][development]["arrays"][name], "PHYSICAL_SCIENCE_SHARED_ARRAY_METADATA:" + name)
    return prior.verified_arrays(canonical, check), canonical


def audit_result(cost_path, meter):
    check = meter.check
    record, cost, parents = metadata_gates(cost_path, check)
    if not parents: return {"status": "NOT_AUDITED_INCOMPLETE_PRODUCER", "cost": record, "producer_status": cost.get("status"), "scientific_assets_opened": False}
    complete = cost["status"] == "COMPLETE"
    matched, shrink, original = (parents[name] for name in ("matched_native_diagnostic", "shrinkage_diagnostic", "producer"))
    files = [HERE / name for name in ("whole_selected_shrinkage_cost_v1.py", "whole_matched_shrunk_native_cost_v1.py", "run_matched_shrunk_native_cost_v2.py", "local_decoder_shrinkage_v1.py", "run_matched_shrunk_native_cost_v1.py")]
    source = source_contract(*(file.read_text(encoding="utf-8-sig") for file in files))
    assert_binding(cost["timing_helper_copy"]["original"], {"path": str(files[0]), "sha256": prior.file_sha(files[0])}, "whole original helper")
    guarded = [record, *[cost[key] for key in PARENT_KEYS], *cost["source_records"], *cost["source_snapshots"], *cost["protected_prior_records"]]
    for name in ("authorization", "continuation_manifest", "repair_addendum", "input_schema_failure_identity", "prior_failed_cost_attempt"):
        guarded.append(cost[name])
    for item in guarded:
        check(); require(prior.file_sha(item["path"]) == item["sha256"], "EXACT_SOURCE_INPUT_HEAD_MODEL_LOCK_FILE_SHA")
        if "snapshot_path" in item: require(prior.file_sha(item["snapshot_path"]) == item["sha256"], "EXACT_EXECUTED_SOURCE_SNAPSHOT")
    require(cost["F_before"] == cost["F_after"] == original["F_before"] == original["F_after"] and set(cost["F_before"]) == {"old_algorithm", "bi_algorithm"}, "FROZEN_DUAL_BOLT_STATE_SIGNATURES")
    for key in ("model_binary", "model_config"):
        assert_binding(cost[key + "_before"], cost[key + "_after"], "unchanged " + key)
        assert_binding(cost[key + "_before"], original[key + "_before"], "original " + key)
    for key in ("evaluation", "all_backbone_parameters_frozen", "context_length", "prediction_length", "input_patch_size", "input_patch_stride", "median_index", "returned_horizon", "quantiles", "pipeline_predict_not_used", "normalizer_before", "normalizer_after", "normalizer_fixture_forward_parity"):
        require(cost["runtime_behavior_contract"][key] == original["runtime_behavior_contract"][key], "FROZEN_BOLT_RUNTIME:" + key)
    seal = prior.verified_json(original["selection_lock"], check)
    guarded.append(original["selection_lock"])
    receipt = prior.verified_json(cost["resource_terminal"], check)
    require(receipt["status"] == cost["status"] and receipt["elapsed_or_cumulative_budget_caps"] is None, "ACTUAL_COST_RESOURCE_RECEIPT")
    report = {"status": "PASS" if complete else "PARTIAL_AUDITED_NO_FULL_PASS", "cost": record, **{key: cost[key] for key in PARENT_KEYS}, "selection_locks": cost["selection_locks"], "old_Bolt_rho_selection_locks": cost["old_Bolt_rho_selection_locks"], "prior_failed_cost_attempt": cost["prior_failed_cost_attempt"], "source_contract": source, "datasets": {}, "cost_details": {}, "failed_paths": cost["failed_paths"], "partial_cost_paths_not_accepted": [],
        "totals": {"cost_cells_checked": 0, "recorded_timer_samples_checked": 0, "physical_backend_prechecks_checked": 0, "physical_first8_window_fits_checked": 0, "native_first8_prediction_checks": 0, "native_selection_locks_checked": 8, "old_Bolt_rho_lock_files_checked": 4, "native_and_E_state_hashes_checked": 0, "selected_forecast_file_hashes_checked": 0, "canonical_physical_bridges_checked": 0, "protected_prior_file_hashes_checked": len(cost["protected_prior_records"])},
        "protected_TEST_accessed": False, "new_training_or_GPU_execution": False, "F_forward_or_timers_independently_replayed": False, "independent_cpu_native_head_equations_replayed": True, "exclusive_per_method_RAM_or_VRAM_claimed": False, "independent_generalization_claimed": False, "native_whole_cost_measured": complete, "past_C_condition_metadata_added": True, "strict_K_only_compression": False, "new_rho_CP_seed_or_method_selection_performed": False}
    state_hashes = {}
    for dataset, (channels, k, development, count, _) in DATASETS.items():
        row = cost["datasets"][dataset]
        if row["status"] != "COMPLETE":
            require(not complete, "COMPLETE_COST_MISSING_DATASET")
            report["partial_cost_paths_not_accepted"].append({"dataset": dataset, "metadata": row, "preserved_sample_count": cost.get("cost", {}).get(dataset, {}).get("recorded_timer_samples", 0)})
            continue
        science, conditions = original["datasets"][dataset], matched["datasets"][dataset]
        require(row["channels"] == channels and row["K"] == k and row["development_split"] == development and row["inputs"] == {name: science["inputs"][name] for name in ("fit", development)}, "SAME_ACCEPTED_C_K_SPLITS")
        fit, dev = (prior.verified_arrays(row["inputs"][name], check) for name in ("fit", development))
        prior.check_raw_split(dev, channels=channels, windows=count)
        for name in ("mean", "std"): exact(fit[name], dev[name], "FIT_only_scaler:" + name)
        physical, canonical = verify_canonical_bridge(cost, dataset, original, development, check)
        exact(physical["origins"], dev["origins"], "physical_origins"); exact(physical["history_observed"], dev["history_observed"], "physical_true_past_mask")
        predictions, prechecks, locks = {"RAW_BOLT": dev["raw_forecast"]}, {}, {}
        shrink_cell = shrink["datasets"][dataset]["splits"][development]
        require(set(row["physical_backend_prechecks"]) == {"PCA_PINV", "FIT_TARGET_PINV"}, "BOTH_FIXED_PROJECTION_PREFLIGHTS")
        for projection in ("PCA_PINV", "FIT_TARGET_PINV"):
            condition = conditions["routes"][projection]
            prior_state = prior.verified_arrays(science["projections"][projection]["state_artifact"], check)
            state_hashes[dataset + ":" + projection + ":E"] = prior.torch_state_sha({name: prior_state[name].astype(np.float32) for name in ("center", "encoder", "decoder", "decoder_bias")})
            head_states, native_predictions = {}, {}
            for seed in SEEDS:
                lock = prior.verified_json(cost["selection_locks"][dataset][projection][str(seed)], check)
                locks[(projection, str(seed))] = lock
                state = prior.verified_arrays(lock["selected_state_artifact"], check)
                checkpoint = next(item for item in condition["native"][str(seed)]["checkpoints"] if item["update"] == lock["selected_update"])
                require(checkpoint["state_artifact"] == lock["selected_state_artifact"] and prior.torch_state_sha(state) == checkpoint["state_sha256"], "SELECTED_NATIVE_HEAD_CP_HASH")
                head_states[str(seed)] = state
                state_hashes[dataset + ":" + projection + ":s" + str(seed)] = prior.torch_state_sha(state)
                predicted = prior.verified_arrays(condition["native"][str(seed)]["artifacts"][development], check)
                exact(predicted["origins"], dev["origins"], "selected_native_reference_origins")
                native_predictions[str(seed)] = predicted["prediction"]
                predictions[projection.replace("_PINV", "_SHRUNK") + "_NATIVE_s" + str(seed)] = predicted["prediction"]
            label, rho = condition["selected_candidate"], condition["rho"]
            backends = shrink_cell["routes"][projection]["positive_rho_candidates"][label]["backends"]
            prechecks[projection] = {}
            require(set(row["physical_backend_prechecks"][projection]) == {"BOLT_NUMPY", "NATIVE_TORCH32"}, "BOTH_NUMPY_TORCH_PHYSICAL_BACKENDS")
            for backend in ("BOLT_NUMPY", "NATIVE_TORCH32"):
                selected = backends[backend]
                selected_state, selected_forecasts = (prior.verified_arrays(selected[name], check) for name in ("state_artifact", "artifact"))
                entry = row["physical_backend_prechecks"][projection][backend]
                expected_sources = {"source_selected_state": selected["state_artifact"], "source_selected_forecasts": selected["artifact"], "source_fixed_E": science["projections"][projection]["state_artifact"], "old_Bolt_rho_selection_lock": condition["old_Bolt_rho_selection_lock"]}
                prechecks[projection][backend] = independent_physical_precheck(entry, prior.verified_arrays(entry["artifact"], check), physical, dev, prior_state, selected_state, selected_forecasts, selected["state_metadata"], expected_sources, rho, backend, head_states, native_predictions)
                literal_path = Path(entry["artifact"]["path"]).with_suffix(".json")
                require(prior.verified_json({"path": str(literal_path), "sha256": prior.file_sha(literal_path)}, check) == entry, "EXACT_LITERAL_PHYSICAL_PREFLIGHT_JSON")
                report["totals"]["physical_backend_prechecks_checked"] += 1
                report["totals"]["physical_first8_window_fits_checked"] += 8
                report["totals"]["native_first8_prediction_checks"] += prechecks[projection][backend]["native_first8_prediction_checks"]
                if backend == "BOLT_NUMPY": predictions[projection.replace("_PINV", "_SHRUNK") + "_BOLT"] = selected_forecasts["BOLT_prediction"]
        metadata = expected_route_metadata(science, development, conditions, shrink_cell, locks, cost["selection_locks"][dataset])
        require(row["selected_metadata"] == metadata, "EXACT_SEVEN_PHYSICAL_WHOLE_COST_REFERENCES")
        details = audit_cost_dataset(cost["cost"][dataset], physical, dev, predictions, metadata, check, sealed_unix=seal["sealed_unix"])
        literal_path = Path(cost_path).parent / ("cost_" + dataset) / "RESULT.json"
        require(prior.verified_json({"path": str(literal_path), "sha256": prior.file_sha(literal_path)}, check) == cost["cost"][dataset], "EXACT_LITERAL_WHOLE_COST_RESULT")
        report["datasets"][dataset] = {"physical_backend_prechecks": prechecks, "canonical_physical_bridge": canonical}
        report["cost_details"][dataset] = details
        report["totals"]["cost_cells_checked"] += details["cells_checked"]
        report["totals"]["recorded_timer_samples_checked"] += details["recorded_timer_samples_checked"]
        report["totals"]["canonical_physical_bridges_checked"] += 1
    for dataset in original["datasets"].values():
        for condition in dataset["projections"].values():
            for method in (condition["bolt"], *condition["native"].values()):
                for artifact in method["artifacts"].values(): guarded.append(artifact); report["totals"]["selected_forecast_file_hashes_checked"] += 1
    for dataset in matched["datasets"].values():
        for condition in dataset["routes"].values():
            for head in condition["native"].values():
                for artifact in head["artifacts"].values(): guarded.append(artifact); report["totals"]["selected_forecast_file_hashes_checked"] += 1
    require(cost["source_cache_selection_SHA_unchanged"] is cost["original_selected_forecasts_and_locks_unchanged"] is True, "OLD_SCIENCE_FORECAST_CACHE_LOCKS_PRESERVED")
    for item in guarded: check(); require(prior.file_sha(item["path"]) == item["sha256"], "POST_AUDIT_SEALED_FILE_PRESERVED")
    if complete:
        require(state_hashes == cost["native_and_E_state_hashes_before"] == cost["native_and_E_state_hashes_after"] and len(state_hashes) == 12, "INDEPENDENT_FOUR_E_EIGHT_HEAD_HASH_IDENTITIES")
        report["totals"]["native_and_E_state_hashes_checked"] = 12
        expected = {"cost_cells_checked": 28, "recorded_timer_samples_checked": 196, "physical_backend_prechecks_checked": 8, "physical_first8_window_fits_checked": 64, "native_first8_prediction_checks": 8, "native_and_E_state_hashes_checked": 12, "selected_forecast_file_hashes_checked": 60, "canonical_physical_bridges_checked": 2}
        require(not cost["failed_paths"] and cost["completed_cost_cells"] == 28 and cost["recorded_timer_samples"] == cost["completed_cell_timer_samples"] == 196 and all(report["totals"][key] == value for key, value in expected.items()), "FULL_REGISTERED_MATCHED_NATIVE_COST_AUDIT_COUNTS")
    return report


def execute(cost_path, output_path):
    from continuation_runtime_v1 import Meter, write_new
    output = Path(output_path).resolve()
    require(not output.exists(), "FRESH_MATCHED_COST_AUDIT_OUTPUT")
    output.parent.mkdir(parents=True, exist_ok=True)
    meter = Meter("audit_matched_shrunk_native_cost_" + output.stem, gpu=False)
    source = {"path": str(Path(__file__).resolve()), "sha256": prior.file_sha(__file__)}
    failure = None
    try:
        meter.protected_check()
        _, _, parents = metadata_gates(cost_path, meter.check)
        if not parents: report = {"status": "NOT_AUDITED_INCOMPLETE_PRODUCER", "scientific_assets_opened": False}
        else:
            initialize_numpy()
            sys.path.insert(0, str(ROOT / "research/smallk_failure_followup_20261014"))
            from prepare_inputs import install_protected_test_guard
            install_protected_test_guard()
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=2): report = audit_result(cost_path, meter)
        meter.protected_check()
        require(prior.file_sha(__file__) == source["sha256"], "INDEPENDENT_AUDIT_SOURCE_UNCHANGED")
        json.dumps(report, allow_nan=False)
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        report = {"status": "FAILED_PRESERVED", "failure": failure, "full_scope_PASS_claimed": False}
    finally:
        report["audit_source"] = source
        report["resource_terminal"] = meter.close(report["status"], failure)
        write_new(output, report)
    print(json.dumps({"status": report["status"], "output": str(output), "totals": report.get("totals"), "failure": failure}), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cost", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raise SystemExit(execute(args.cost, args.output))

