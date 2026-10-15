"""Independent fixed-D whole-cost samples/physical/head audit; no F or timer replay."""
from pathlib import Path
import argparse
import ast
import json
import math
import statistics
import sys

import audit_projection_results_v1 as prior
import audit_decoder_metric_cost_v1 as old_cost_audit
import audit_raw_subset_comparison_v3 as science_audit

HERE, BASE, ROOT = prior.HERE, prior.BASE, prior.ROOT
DATASETS, SPACES, SEEDS = science_audit.DATASETS, tuple(science_audit.SPACES), science_audit.SEEDS
ROUTES = ("RAW_BOLT", "PCA_BOLT", "FIT_TARGET_BOLT", "RAW_SUBSET_BOLT", "PCA_NATIVE_s92601", "PCA_NATIVE_s92602", "FIT_TARGET_NATIVE_s92601", "FIT_TARGET_NATIVE_s92602", "RAW_SUBSET_NATIVE_s92601", "RAW_SUBSET_NATIVE_s92602")
require, exact = prior.require, science_audit.exact
array_record = old_cost_audit.array_record
np = sys.modules.get("numpy")
PARENT_KEYS = ("raw_subset_diagnostic", "raw_subset_audit", "source_producer", "source_audit", "prior_partial", "Gram_failure_diagnostic", "numeric_repair_addendum")
SCIENCE_SHA = "2b271ba9aa6ae359033444b282ddc1becc5d1f60ef063451171e9c93f70db4ff"
SCIENCE_COUNTS = {"subset_selection_fits_checked": 2, "greedy_candidate_scores_checked": 83, "global_decoder_states_checked": 12,
    "source_encoder_states_checked": 6, "source_split_contexts_checked": 18, "native_fits_checked": 12, "training_updates_logged_checked": 7200,
    "checkpoint_states_checked": 60, "checkpoint_TUNE_prediction_arrays_checked": 60, "selected_native_split_prediction_arrays_checked": 36,
    "Bolt_split_prediction_arrays_checked": 18, "actual_subset_F_split_provenance_checks": 6, "actual_subset_F_model_batch_calls_registered": 103,
    "reused_mixed_F_cache_lineages_checked": 12, "resolved_native_selection_locks_checked": 12, "cached_future_label_perturbation_checks": 12,
    "new_fixed_D_output_space_cells_computed": 36, "new_E_Y_diagnostic_predictions_computed": 36, "new_fixed_D_real_method_inequalities_checked": 90,
    "RAW_bound_inequalities_checked": 0, "current_C_persistence_split_baselines_computed": 6}


def initialize_numpy():
    global np
    old_cost_audit.initialize_numpy(); science_audit.initialize_numpy(); np = sys.modules["numpy"]


def independent_schedule():
    first = np.random.default_rng(20261015).permutation(10)
    return [[ROUTES[index] for index in np.roll(first, -repeat)] for repeat in range(10)]


def parity_checked(recorded, actual, expected, *, physical=False, must_pass=True):
    atol, rtol = (2e-3, 2e-5) if physical else (1e-5, 2e-6)
    require(actual.shape == expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all(), "FINITE_MATCHED_PARITY_ARRAYS")
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    value = {"pass": bool(np.all(np.abs(difference) <= atol + rtol * np.abs(expected))), "max_abs": float(np.abs(difference).max()), "rmse": float(np.sqrt(np.mean(difference * difference))), "atol": atol, "rtol": rtol}
    require(recorded["pass"] == value["pass"] and recorded["atol"] == atol and recorded["rtol"] == rtol, "REGISTERED_PARITY_BOOLEAN_AND_UNCHANGED_TOLERANCES")
    for field in ("max_abs", "rmse"): prior.close(recorded[field], value[field], "recorded_cost_parity:" + field, atol=1e-12, rtol=1e-12)
    require(not must_pass or value["pass"], "COMPLETE_ROW_CANNOT_HAVE_FAILED_OUTPUT_PARITY")
    return value


def recorded_times(row, raw):
    samples = row["samples"]; require(len(samples) == row["repeats"] and row["warmup_each_route"] == 2, "ACTUAL_SAMPLE_COUNT_AND_WARMUP")
    require([sample["repeat"] for sample in samples] == sorted({sample["repeat"] for sample in samples}) and all(0 <= sample["repeat"] < 10 for sample in samples), "UNIQUE_REGISTERED_REPEAT_NUMBERS")
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
    if complete: require(len(samples) == 10 and row["failure"] is None, "COMPLETE_TEN_SAMPLE_ROW_WITHOUT_FAILURE")
    raw_median = statistics.median([value for value in raw_by_repeat.values()]) if raw_by_repeat else None
    return {"median_seconds": median, "median_seconds_per_window": median / row["batch"] if median else None, "paired_RAW_median_ratio": paired, "ratio_of_RAW_medians": median / raw_median if median is not None and raw_median else None, "ratio_of_RAW_medians_uses_complete_raw_cohort": raw["status"] == "COMPLETE", "samples_checked": len(samples), "independent_timer_remeasurement": False}


def memory_checked(memory):
    fields = {"RSS_bytes", "allocated_VRAM_bytes", "reserved_VRAM_bytes", "max_allocated_VRAM_bytes", "max_reserved_VRAM_bytes"}
    require(set(memory) == fields | {"scope"} and all(type(memory[field]) is int and memory[field] >= 0 for field in fields) and memory["scope"] == "shared resident F/E/D/all native heads", "SHARED_PROCESS_MEMORY_SCOPE_AND_COUNTER_TYPES")
    require(memory["reserved_VRAM_bytes"] >= memory["allocated_VRAM_bytes"] and memory["max_reserved_VRAM_bytes"] >= memory["max_allocated_VRAM_bytes"] >= memory["allocated_VRAM_bytes"] and memory["max_reserved_VRAM_bytes"] >= memory["reserved_VRAM_bytes"], "CURRENT_AND_UNRESET_PEAK_MEMORY_RELATIONSHIPS")


def source_contract(helper_text, runner_text, original_text):
    def functions(text): return {node.name: node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)}
    helper, runner, original = map(functions, (helper_text, runner_text, original_text))
    for name in ("fill_vectorized", "preprocess", "parity"):
        require(ast.dump(helper[name], include_attributes=False) == ast.dump(original[name], include_attributes=False), "UNCHANGED_CANONICAL_PHYSICAL_PREPROCESS_OR_TOLERANCE:" + name)
    constants = {}
    names = {"ROUTES", "ORDER_SEED", "WARMUP", "REPEATS", "NORM_ATOL", "NORM_RTOL", "PHYSICAL_ATOL", "PHYSICAL_RTOL"}
    for node in ast.parse(helper_text).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1: continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in names: constants[target.id] = ast.literal_eval(node.value)
        elif isinstance(target, ast.Tuple) and all(isinstance(item, ast.Name) and item.id in names for item in target.elts): constants.update(zip((item.id for item in target.elts), ast.literal_eval(node.value)))
    require(constants == {"ROUTES": ROUTES, "ORDER_SEED": 20261015, "WARMUP": 2, "REPEATS": 10, "NORM_ATOL": 1e-5, "NORM_RTOL": 2e-6, "PHYSICAL_ATOL": 2e-3, "PHYSICAL_RTOL": 2e-5}, "REGISTERED_TEN_FULL_ROTATIONS_NO_TOLERANCE_CHANGE")
    require(ast.unparse(helper["balanced_schedule"]) == "def balanced_schedule(seed=ORDER_SEED):\n    order = np.random.default_rng(seed).permutation(len(ROUTES))\n    return [[ROUTES[index] for index in np.roll(order, -repeat)] for repeat in range(REPEATS)]", "TEN_ROTATION_ORDER_SOURCE")
    whole = ast.unparse(helper["whole_predict"])
    require("preprocess(physical, observed, mean, std)" in whole and "predictor(normalized_history, observed)" in whole and "(normalized * std + mean).astype(np.float32, copy=False).copy()" in whole, "PHYSICAL_PREPROCESS_THROUGH_MATERIALIZED_CPU_RETURN")
    timers = []
    for node in ast.walk(helper["measure_cost"]):
        if isinstance(node, ast.Try) and "perf_end_ns, unix_end_ns = (time.perf_counter_ns(), time.time_ns())" in ast.unparse(node):
            if len(node.body) == 3: timers.append(node)
    require(len(timers) == 1 and [ast.unparse(node) for node in timers[0].body] == ["normalized, physical_value = whole_predict(predictors[route], values, masks, mean, std)", "synchronize()", "perf_end_ns, unix_end_ns = (time.perf_counter_ns(), time.time_ns())"], "ACTUAL_WHOLE_PREDICT_AND_FINAL_SYNC_BEFORE_PERF_END")
    measure = ast.unparse(helper["measure_cost"])
    require("unix_start_ns, perf_start_ns = (time.time_ns(), time.perf_counter_ns())" in measure and "seconds = (perf_end_ns - perf_start_ns) / 1000000000.0" in measure and "memory = _jsonable(memory_snapshot())" in measure, "SOURCE_TIMER_START_END_ARITHMETIC_AND_OUTSIDE_MEMORY_SNAPSHOT")
    expected = {"bolt_predictor": ["z = encode_numpy(state, indices, history)", "return history[:, -1, None, :] + (frozen_F(z) - z[:, -1, None, :]) @ decoder"], "native_predictor": ["z = encode_native(projection, indices, history)", "return predict_components(model, z, history[:, -1], batch_size=32)[0]"]}
    for name, body in expected.items():
        nested = [node for node in runner[name].body if isinstance(node, ast.FunctionDef)]
        require(len(nested) == 1 and [arg.arg for arg in nested[0].args.args] == ["history", "history_observed"] and [ast.unparse(node) for node in nested[0].body] == body, "FIXED_STATIC_D_ACTUAL_E_F_OR_CPU_HEAD_CALLBACK:" + name)
    require("fit_global_observed_decoder" not in runner_text and "fit_shrunk_local_decoder" not in runner_text and "fit_local_past_decoder" not in runner_text, "NO_PER_ORIGIN_OR_NEW_GLOBAL_FIT_IN_COST")
    return {"physical_preprocessing_model_forward_fixed_D_and_CPU_inverse_inside_timer": True, "all_source_file_state_metadata_hashes_outside_timer": True, "per_origin_decoder_fit_in_cost": False, "ten_complete_order_rotations": True, "independent_F_or_timers_replayed": False}


def physical_precheck(entry, arrays, physical, raw, fixed, e, d, backend, head_states, predictions, Bolt_future):
    require(entry["backend"] == backend and entry["origins"] == raw["origins"][:8].tolist() and entry["per_origin_decoder_fit"] is False and entry["fixed_D_byte_source"] == array_record(d), "PHYSICAL_PREFLIGHT_ORIGINS_STATIC_D_AND_BACKEND")
    prefix = prior.causal_physical_preprocess(physical["history_physical"][:8], physical["history_observed"][:8], raw["mean"], raw["std"])
    z = science_audit.encode(prefix, e, backend)
    base_keys = {"preprocessed_history", "latent_history", "history_observed", "origins", "fixed_decoder_FP32"}
    expected_keys = base_keys | ({"prediction", "latent_future"} if backend == "BOLT_NUMPY" else {"native_s" + seed + "_" + suffix for seed in head_states for suffix in ("prediction", "latent_future")})
    require(set(arrays) == expected_keys, "EXACT_PHYSICAL_PREFLIGHT_ARRAY_KEYS")
    for key, value in (("preprocessed_history", prefix), ("latent_history", z), ("history_observed", physical["history_observed"][:8]), ("origins", raw["origins"][:8]), ("fixed_decoder_FP32", d)): exact(arrays[key], value, "physical_fixed_D:" + key)
    expected_decisions = {"history_observed": physical["history_observed"][:8].tobytes() == raw["history_observed"][:8].tobytes(), "origins": raw["origins"][:8].tobytes() == fixed["origins"][:8].tobytes(), "normalized_history": prefix.tobytes() == raw["history"][:8].tobytes()}
    require(entry["exact_decisions"] == expected_decisions, "PHYSICAL_EXACT_DECISIONS_INDEPENDENT_BYTE_CHECKS")
    comparisons = {"history": (prefix, raw["history"][:8]), "current_C": (prefix[:, -1], raw["raw_last"][:8]), "latent_history": (z, fixed["latent_history_" + backend][:8]), "latent_last": (z[:, -1], fixed["latent_history_" + backend][:8, -1])}
    for key, (actual, expected) in comparisons.items(): parity_checked(entry["checks"][key], actual, expected)
    require(entry["input_backend_pass"] == (all(expected_decisions.values()) and all(entry["checks"][key]["pass"] for key in comparisons)), "EXACT_INPUT_BACKEND_ACCEPTANCE")
    independent_predictions = 0
    if backend == "BOLT_NUMPY":
        require(set(entry["prediction_checks"]) == {"BOLT"}, "ONE_BOLT_PREFLIGHT_PREDICTION")
        parity_checked(entry["checks"]["latent_future"], arrays["latent_future"], Bolt_future[:8])
        p = science_audit.fixed_forecast(prefix[:, -1], arrays["latent_future"], z[:, -1], d, backend)
        exact(p, arrays["prediction"], "preflight_saved_F_future_fixed_D_anchor")
        parity_checked(entry["prediction_checks"]["BOLT"], p, predictions["BOLT"][:8])
        require(entry["checks"]["prediction"] == entry["prediction_checks"]["BOLT"], "BOLT_PREFLIGHT_PARITY_DUPLICATE_BINDING")
    else:
        require(set(entry["prediction_checks"]) == set(head_states), "ALL_SELECTED_NATIVE_PREFLIGHT_HEADS")
        for seed, state in head_states.items():
            exact(state["decoder"], d, "native_physical_head_fixed_D")
            p, f = science_audit.independent_native({"latent_history": z, "raw_last": prefix[:, -1]}, state)
            exact(p, arrays["native_s" + seed + "_prediction"], "independent_native_physical_prediction")
            exact(f, arrays["native_s" + seed + "_latent_future"], "independent_native_physical_future")
            parity_checked(entry["prediction_checks"][seed], p, predictions[seed][:8])
            require(entry["prediction_checks"][seed] == entry["checks"]["native_s" + seed], "NATIVE_PREFLIGHT_PARITY_DUPLICATE_BINDING")
            independent_predictions += 1
    require(entry["status"] == "PASS" and entry["input_backend_pass"] is True and all(entry["checks"][key]["pass"] for key in entry["checks"]), "COMPLETE_PHYSICAL_PREFLIGHT_PASS")
    return {"first8_windows_checked": 8, "backend": backend, "fixed_D_byte_exact": True, "independent_native_prediction_checks": independent_predictions, "F_future_reinference_by_auditor": False}


def json_record(path, check):
    path = Path(path)
    path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    record = {"path": str(path), "sha256": prior.file_sha(path)}
    return record, prior.verified_json(record, check)


def literal_constants(path):
    output = {}
    for node in ast.parse(Path(path).read_text(encoding="utf-8-sig")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try: output[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError): pass
    return output


def load_parent_metadata(cost, check):
    parents = {key: prior.verified_json(cost[key], check) for key in PARENT_KEYS if key != "numeric_repair_addendum"}
    record = cost["numeric_repair_addendum"]; path = Path(record["path"])
    check(); require(path.is_absolute() and path.suffix.lower() == ".md" and path.is_file() and prior.file_sha(path) == record["sha256"], "EXACT_MARKDOWN_ADDENDUM_TEXT_SHA_NO_JSON_PARSE")
    if "bytes" in record: require(path.stat().st_size == record["bytes"], "EXACT_MARKDOWN_ADDENDUM_BYTES")
    return parents


def metadata_gates(cost_path, check):
    binding, cost = json_record(cost_path, check)
    if cost.get("status") not in ("COMPLETE", "PARTIAL_COMPLETED"): return binding, cost, {}
    require(cost["routes"] == list(ROUTES) and cost["expected_cost_cells"] == 40 and cost["expected_timer_samples"] == 400 and cost["new_training_updates"] == 0, "REGISTERED_TEN_ROUTE_FORTY_CELL_COST")
    for key in ("new_backbone_training", "new_encoder_subset_decoder_or_CP_selection", "per_origin_decoder_fit", "protected_TEST_accessed", "independent_holdout", "strict_K_only_compression"):
        require(cost[key] is False, "NO_NEW_SELECTION_TRAINING_OR_TEST:" + key)
    require(cost["fixed_FIT_decoder_loaded_before_timer"] is cost["all_source_file_state_metadata_hashes_outside_timer"] is True and cost["hard_elapsed_or_cumulative_resource_caps"] is None, "FIXED_FIT_D_AND_OUTSIDE_METADATA_TIMER_BOUNDARY")
    parents = load_parent_metadata(cost, check)
    science, audit = parents["raw_subset_diagnostic"], parents["raw_subset_audit"]
    require(cost["raw_subset_diagnostic"]["sha256"] == SCIENCE_SHA and science["status"] == "COMPLETE" and audit["status"] == "PASS" and all(audit["totals"].get(key) == value for key, value in SCIENCE_COUNTS.items()), "COMPLETE_SCIENCE_EXACT_INDEPENDENT_COVERAGE_BEFORE_ARRAYS")
    old_cost_audit.assert_binding(audit["diagnostic"], cost["raw_subset_diagnostic"], "accepted raw subset science")
    old_cost_audit.assert_binding(audit["audit_source"], {"path": str(HERE / "audit_raw_subset_comparison_v3.py"), "sha256": "0782c7d8d66bae5bd124c5c778eadde08786e2da2246882233eb4489b3410b20"}, "immutable independently verified science source v3")
    runner = literal_constants(HERE / "run_raw_subset_cost_v1.py")
    require(runner["SCIENCE_RESULT_SHA"] == SCIENCE_SHA and runner["SCIENCE_AUDIT_SHA"] == cost["raw_subset_audit"]["sha256"] and runner["SCIENCE_AUDITOR"] == "audit_raw_subset_comparison_v3.py" and all(runner["SCIENCE_AUDIT_COUNTS"].get(key) == value for key, value in SCIENCE_COUNTS.items()), "ACTUAL_COST_IMPORT_GATE_PINS_EXACT_FULL_SCIENCE_PASS")
    for name, digest in runner["SCIENCE_SOURCE_SHA"].items(): require(prior.file_sha(HERE / name) == digest, "FROZEN_SCIENCE_SOURCE:" + name)
    for key in PARENT_KEYS[2:]:
        old_cost_audit.assert_binding(cost[key], science[key], "unchanged science parent " + key)
    require(parents["source_producer"]["status"] == "COMPLETE" and parents["source_audit"]["status"] == "PASS" and parents["prior_partial"]["status"] == "PARTIAL_COMPLETED", "ORIGINAL_FIT_PASS_AND_PRESERVED_V1_PARTIAL")
    identity = audit["mixed_reexecution_identity"]
    require(identity["status"] == "PASS" and identity["total_physical_native_fit_attempts_across_v1_v2"] == 20 and identity["distinct_registered_native_conditions"] == 12 and identity["new_independent_conditions_from_reexecution"] == 0 and identity["actual_old_and_new_NPZ_logical_bytes_checked"] is True, "ACCEPTED_MIXED_REPEAT_NOT_NEW_INDEPENDENT_CONDITIONS")
    registry = prior.verified_json(cost["selection_registry"], check)
    require(cost["selection_registry"] == science["selection_registry"] and registry["status"] == "ALL_TWELVE_RESOLVED_BEFORE_DEV" and registry["failed_paths_at_seal"] == [], "EXACT_TWELVE_ORIGINAL_SUCCESSFUL_SELECTION_LOCKS")
    expected_keys = {(dataset, space, str(seed)) for dataset in DATASETS for space in SPACES for seed in SEEDS}
    keys = set()
    for row in registry["conditions"]:
        key = (row["dataset"], row["space"], row["seed"])
        require(key not in keys and row["lock"] == cost["native_selection_locks"][key[0]][key[1]][key[2]], "UNIQUE_UNCHANGED_COST_NATIVE_CP_LOCK")
        keys.add(key); lock = prior.verified_json(row["lock"], check)
        head = science["datasets"][key[0]]["spaces"][key[1]]["native"][key[2]]
        require(lock["status"] == "LOCKED" and lock["DEV_used_for_CP_selection"] is False and lock["selected_update"] == head["metadata"]["selected_update"] and lock["selected_state_artifact"] == next(cp for cp in head["checkpoints"] if cp["update"] == lock["selected_update"])["state_artifact"], "ORIGINAL_TUNE_ONLY_SELECTED_CP_EXACT_STATE")
    require(keys == expected_keys and len(registry["conditions"]) == 12, "ALL_TWELVE_UNCHANGED_CP_LOCKS")
    for dataset in DATASETS:
        require(cost["subset_selection_locks"][dataset] == science["datasets"][dataset]["subset_selection_lock"], "FIXED_FIT_PAST_SUBSET_LOCK")
        subset = prior.verified_json(cost["subset_selection_locks"][dataset], check)
        require(subset["status"] == "LOCKED_FIT_PAST_ONLY", "NO_SUBSET_RESELECTION_IN_COST")
    if cost["status"] == "COMPLETE": require(cost["native_deployment_cost_measured"] is True and cost["loaded_native_heads_count"] == 12 and cost["native_head_device"] == "cpu" and cost["native_threads"] == 2 and cost["canonical_native_prediction_batch_size"] == 32, "TWELVE_SELECTED_CPU2_HEADS_TIMED_CANONICAL32")
    return binding, cost, parents


def route_metadata(science_row, dev, locks, lock_records):
    output = {"RAW_BOLT": {"source_input": science_row["raw_input_sources"][dev], "reference_prediction_array": "raw_forecast", "full_C_F_called": True, "per_origin_decoder_fit": False}}
    for space in SPACES:
        condition = science_row["spaces"][space]
        indices = None if space != "RAW_SUBSET" else locks["subset"]["selected_indices"]
        common = {"space": space, "E_state": condition["E_state_artifact"], "reference_prediction_array": "prediction", "per_origin_decoder_fit": False}
        output[space + "_BOLT"] = {**common, "backend": "BOLT_NUMPY", "global_D_state": condition["backends"]["BOLT_NUMPY"]["state_artifact"], "selected_CPU_reference": condition["Bolt"][dev]["artifact"], "subset_selection_lock": science_row["subset_selection_lock"] if indices is not None else None, "selected_indices": indices, "actual_K_channel_F_inference": True, "full_C_slice_used_as_inference": False}
        for seed in SEEDS:
            lock = locks[(space, str(seed))]
            output[space + "_NATIVE_s" + str(seed)] = {**common, "backend": "NATIVE_TORCH32", "global_D_state": condition["backends"]["NATIVE_TORCH32"]["state_artifact"], "selected_CPU_reference": condition["native"][str(seed)]["artifacts"][dev], "selected_head_lock": lock_records[space][str(seed)], "selected_head_state": lock["selected_state_artifact"], "native_seed": seed, "selected_update": lock["selected_update"], "head_device": "cpu", "head_threads": 2, "canonical_prediction_batch_size": 32, "F_forward_in_this_route": False}
    return output


def canonical_bridge(row, scientific_row, original, dataset, dev, check):
    require(row["physical_bridge_canonical"] == scientific_row["physical_bridge_canonical"] and row["canonical_physical_bridge_binding"] == scientific_row["canonical_physical_bridge_binding"], "UNCHANGED_FULL_PHYSICAL_MANIFEST_AND_ALIAS_PROVENANCE")
    proof = row["canonical_physical_bridge_binding"]
    old_cost_audit.assert_binding(proof["original_first_parent"], original["source_parent_results"][dataset], "original physical parent")
    parent = prior.verified_json(proof["original_first_parent"], check)
    old_cost_audit.assert_binding(proof["canonical_input_bridge"], parent["bridge_provenance"]["input_bridge"], "canonical input bridge")
    bridge = prior.verified_json(proof["canonical_input_bridge"], check)
    require(parent["status"] == bridge["status"] == "COMPLETE" and parent["protected_TEST_accessed"] is bridge["actual_status"]["protected_test_opened"] is False, "COMPLETE_PHYSICAL_INPUT_CHAIN_NO_TEST")
    matches = [record for record in bridge["artifacts"] if record["dataset"] == dataset and record["split"] == dev]
    require(len(matches) == 1 and proof["original_alias"] == original["datasets"][dataset]["physical_bridge"], "ONE_SAME_ORIGINAL_PHYSICAL_ALIAS")
    canonical = prior.bind_canonical_npz(proof["original_alias"], matches[0])
    require(canonical == proof["canonical_record"] == row["physical_bridge_canonical"] and proof["same_physical_file_path_and_SHA"] is proof["complete_array_manifest_preserved"] is True and proof["loader_file_SHA_member_shape_dtype_SHA_checks_relaxed"] is proof["numeric_tolerance_changed"] is False, "EXACT_FULL_ARRAY_MANIFEST_NO_LOADER_OR_TOLERANCE_RELAXATION")
    return prior.verified_arrays(canonical, check)


def audit_cost_dataset(cost, physical, dev, predictions, metadata, check, sealed_unix):
    require(cost["routes"] == list(ROUTES) and len(cost["rows"]) == 20 and cost["expected_cells_this_dataset"] == 20 and cost["expected_cells_two_datasets"] == 40 and cost["warmup_each_route"] == 2 and cost["repeats"] == 10, "TEN_DECLARED_ROUTES_TWO_BATCHES_ALL_SLOTS_RETAINED")
    require(cost["order_seed"] == 20261015 and cost["schedule"] == independent_schedule(), "EXACT_TEN_LATIN_ROTATIONS")
    require(cost["new_training_updates"] == 0 and cost["encoder_refit_in_cost"] is cost["per_origin_decoder_fit"] is cost["fit_input_binding_hashes_inside_timer"] is cost["protected_TEST_accessed"] is cost["independent_holdout"] is False and cost["fixed_FIT_decoder_loaded_before_timer"] is True, "FIXED_FIT_D_NO_FIT_OR_METADATA_HASHES_IN_TIMED_CALLBACK")
    exact(prior.causal_physical_preprocess(physical["history_physical"], physical["history_observed"], dev["mean"], dev["std"]), dev["history"], "all_physical_past_exact_original_normalized_input")
    for key, value in (("physical_past", physical["history_physical"]), ("history_observed", physical["history_observed"]), ("FIT_mean", dev["mean"]), ("FIT_std", dev["std"]), ("origins", dev["origins"])): require(cost["input_array_metadata"][key] == array_record(value), "COST_INPUT_ARRAY_SHA:" + key)
    table = {(row["route"], row["batch"]): row for row in cost["rows"]}
    require(len(table) == 20 and set(table) == {(route, batch) for route in ROUTES for batch in (1, 8)}, "NO_DUPLICATE_OR_MISSING_DECLARED_COST_CELL")
    lineage = cost["source_lineage"]
    require(lineage["whole_cost_source_sha256"] == prior.file_sha(HERE / "whole_raw_subset_cost_v1.py") and Path(lineage["copied_timer_source"]).resolve() == HERE / "whole_matched_shrunk_native_cost_v1.py" and lineage["copied_timer_source_sha256"] == prior.file_sha(HERE / "whole_matched_shrunk_native_cost_v1.py") and Path(lineage["copied_preprocess_source"]).resolve() == BASE / "cost_readout_v1.py" and lineage["copied_preprocess_source_sha256"] == prior.file_sha(BASE / "cost_readout_v1.py") and lineage["copied_functions"] == ["fill_vectorized", "preprocess", "parity"] and lineage["legacy_scientific_runtime_imported"] is False, "UNCHANGED_PHYSICAL_TIMER_SOURCE_LINEAGE")
    schedule, details, chronological, completed, samples_total = independent_schedule(), [], [], 0, 0
    require(set(cost["source_metadata"]) == set(ROUTES), "ALL_TEN_ROUTE_METADATA_SLOTS_PRESERVED")
    for (route, batch), row in table.items():
        check(); accepted = row["status"] == "COMPLETE"
        require(row["dataset"] == cost["dataset"] and row["declared_route_deleted"] is row["memory_counter_reset_inside_module"] is row["per_origin_decoder_fit"] is row["stage_decomposition"] is False and row["native_head_loaded_or_timed"] == ("_NATIVE_s" in route), "WHOLE_COST_SCOPE_AND_SAVED_PARTIAL_ROUTE_IDENTITY")
        if route in metadata: require(row["selected_metadata"] == cost["source_metadata"][route] == metadata[route], "EXACT_COST_ROUTE_REFERENCE_AND_SELECTED_CP_METADATA")
        else: require(not accepted, "COMPLETE_ROW_MISSING_SCIENTIFIC_ROUTE_METADATA")
        require(row["origins"] == dev["origins"][:batch].tolist(), "SAME_FIRST_B_ORIGINS")
        arithmetic = recorded_times(row, table[("RAW_BOLT", batch)])
        repeats = [sample["repeat"] for sample in row["samples"]]
        require(row["positions"] == [schedule[repeat].index(route) for repeat in repeats] and len(row["samples"]) == len(row["parity"]) == len(row["memory"]), "TEN_ROTATION_POSITION_AND_SAMPLE_LIST_IDENTITY")
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
    if cost["status"] == "COMPLETE": require(completed == 20 and samples_total == cost["attempted_timer_calls"] == 200 and cost["failed_paths"] == [], "FULL_TEN_ROUTE_COST_DATASET_NOT_PARTIAL")
    return {"rows": details, "cells_checked": completed, "recorded_timer_samples_checked": samples_total, "declared_cells_checked": 20, "actual_timers_or_F_remeasured": False, "all_ten_positions_balanced_for_complete_routes": True, "exclusive_per_method_memory_claimed": False, "outliers_or_samples_removed": False, "same_first_B_origins_repeated_not_independent_draws": True}


def audit_result(cost_path, meter):
    check = meter.check
    binding, cost, parents = metadata_gates(cost_path, check)
    if not parents: return {"status": "NOT_AUDITED_INCOMPLETE_PRODUCER", "cost": binding, "producer_status": cost.get("status"), "scientific_assets_opened": False}
    producer_complete = cost["status"] == "COMPLETE"
    science, original = parents["raw_subset_diagnostic"], parents["source_producer"]
    paths = [HERE / name for name in ("whole_raw_subset_cost_v1.py", "run_raw_subset_cost_v1.py", "whole_matched_shrunk_native_cost_v1.py")]
    source = source_contract(*(path.read_text(encoding="utf-8-sig") for path in paths))
    guarded = [binding, *[cost[key] for key in PARENT_KEYS], *cost["source_records"], *cost["source_snapshots"], *cost["protected_prior_records"], *cost["input_records"], cost["cost_plan"], cost["authorization"], cost["continuation_manifest"], parents["raw_subset_audit"]["audit_source"], cost["selection_registry"]]
    for record in guarded:
        check(); require(prior.file_sha(record["path"]) == record["sha256"], "PRE_AUDIT_SCIENCE_COST_SOURCE_MODEL_CACHE_AND_LOCK_SHA")
        if "snapshot_path" in record: require(prior.file_sha(record["snapshot_path"]) == record["sha256"], "ACTUAL_EXECUTED_COST_SOURCE_SNAPSHOT_BYTES")
    require(cost["F_before"] == cost["F_after"] == science["F_before"] == science["F_after"] == original["F_before"] and set(cost["F_before"]) == {"old_algorithm", "bi_algorithm"}, "SAME_FROZEN_BOLT_DUAL_SIGNATURES")
    for name in ("model_binary", "model_config"):
        old_cost_audit.assert_binding(cost[name + "_before"], cost[name + "_after"], "frozen before/after " + name)
        old_cost_audit.assert_binding(cost[name + "_before"], science[name + "_before"], "same science " + name)
    for key in ("evaluation", "all_backbone_parameters_frozen", "context_length", "prediction_length", "input_patch_size", "input_patch_stride", "median_index", "returned_horizon", "quantiles", "pipeline_predict_not_used", "normalizer_before", "normalizer_after", "normalizer_fixture_forward_parity"):
        require(cost["runtime_behavior_contract"][key] == science["runtime_behavior_contract"][key], "SAME_FROZEN_BOLT_RUNTIME:" + key)
    receipt = prior.verified_json(cost["resource_terminal"], check)
    require(receipt["status"] == cost["status"] and receipt["elapsed_or_cumulative_budget_caps"] is None, "ACTUAL_COST_RESOURCE_RECEIPT")
    original_lock = prior.verified_json(original["selection_lock"], check); guarded.append(original["selection_lock"])
    selected_lock_seals = [prior.verified_json(record, check)["sealed_unix"] for spaces in cost["native_selection_locks"].values() for seeds in spaces.values() for record in seeds.values()]
    latest_selection_seal = max(original_lock["sealed_unix"], *selected_lock_seals)
    totals = {"cost_cells_checked": 0, "recorded_timer_samples_checked": 0, "physical_backend_prechecks_checked": 0, "physical_first8_windows_checked": 0, "native_first8_prediction_checks": 0, "native_selection_locks_checked": 12, "subset_selection_locks_checked": 2, "encoder_state_hashes_checked": 0, "native_projection_state_hashes_checked": 0, "global_decoder_state_hashes_checked": 0, "native_head_state_hashes_checked": 0, "selected_forecast_file_hashes_checked": 0, "canonical_physical_bridges_checked": 0, "protected_prior_file_hashes_checked": len(cost["protected_prior_records"])}
    report = {"status": "PASS" if producer_complete else "PARTIAL_AUDITED_NO_FULL_PASS", "cost": binding, **{key: cost[key] for key in PARENT_KEYS}, "native_selection_locks": cost["native_selection_locks"], "subset_selection_locks": cost["subset_selection_locks"], "selection_registry": cost["selection_registry"], "source_contract": source, "totals": totals, "datasets": {}, "cost_details": {}, "producer_failed_paths": cost["failed_paths"], "partial_cost_paths_not_accepted": [], "protected_TEST_accessed": False, "new_training_or_F_or_GPU_execution": False, "F_or_actual_timers_independently_replayed": False, "independent_CPU_selected_head_equations_replayed": True, "per_origin_decoder_fit_in_cost": False, "per_method_exclusive_RAM_or_VRAM_claimed": False, "strict_K_only_compression": False, "independent_generalization_claimed": False, "outliers_or_samples_removed": False, "new_method_or_checkpoint_selection_performed": False}
    hashes = {"E": {}, "projections": {}, "heads": {}, "global_decoders": {}}
    for dataset, (channels, k, dev, count) in DATASETS.items():
        check(); row, srow = cost["datasets"][dataset], science["datasets"][dataset]
        if "raw_input_source" not in row:
            require(not producer_complete, "COMPLETE_COST_MISSING_DATASET_INPUTS")
            report["partial_cost_paths_not_accepted"].append({"dataset": dataset, "metadata": row}); continue
        require(row["raw_input_source"] == srow["raw_input_sources"][dev], "EXACT_SCIENTIFIC_DEVELOPMENT_INPUT_REFERENCE")
        raw = prior.verified_arrays(row["raw_input_source"], check); prior.check_raw_split(raw, channels=channels, windows=count)
        physical = canonical_bridge(row, srow, original, dataset, dev, check)
        for name in ("origins", "history_observed", "mean", "std"): exact(physical[name], raw[name], "physical_cost_same_source:" + name)
        totals["canonical_physical_bridges_checked"] += 1
        predictions, locks, prechecks = {"RAW_BOLT": raw["raw_forecast"]}, {}, {}
        locks["subset"] = prior.verified_json(cost["subset_selection_locks"][dataset], check); guarded.append(cost["subset_selection_locks"][dataset])
        for space in SPACES:
            condition = srow["spaces"][space]; prepared = row["spaces"][space]
            if prepared["status"] != "COMPLETE_PREPARED":
                require(not producer_complete, "COMPLETE_COST_WITH_PARTIAL_SPACE_PREPARATION")
                report["partial_cost_paths_not_accepted"].append({"dataset": dataset, "space": space, "metadata": prepared}); continue
            e = prior.verified_arrays(condition["E_state_artifact"], check)
            require(prepared["E_state"] == condition["E_state_artifact"] and prepared["global_D_states"] == {backend: entry["state_artifact"] for backend, entry in condition["backends"].items()} and prepared["indices"] == (locks["subset"]["selected_indices"] if space == "RAW_SUBSET" else None), "SAME_FIXED_E_D_INDICES_AS_SCIENTIFIC_COHORT")
            if space == "RAW_SUBSET": state = {"center": e["center"], "encoder": e["encoder"], "decoder": np.zeros((k, channels), np.float64), "decoder_bias": np.zeros(channels, np.float64)}
            else: state = prior.verified_arrays(condition["original_E_state"], check)
            key = str((dataset, space))
            hashes["E"][key] = {name: array_record(state[name])["sha256"] for name in ("center", "encoder", "decoder", "decoder_bias")}
            hashes["projections"][key] = prior.torch_state_sha({name: state[name].astype(np.float32) for name in ("center", "encoder", "decoder", "decoder_bias")})
            require(hashes["projections"][key] == condition["fixed_E_hashes_before"]["native_projection_state"], "INDEPENDENT_NATIVE_PROJECTION_STATE_SCIENCE_BINDING")
            totals["encoder_state_hashes_checked"] += 1; totals["native_projection_state_hashes_checked"] += 1
            fixed = prior.verified_arrays(condition["inputs"][dev], check)
            for name in ("history", "history_observed", "origins", "raw_last"): exact(fixed[name], raw[name], "fixed_space_same_past_context:" + name)
            head_states, native_predictions = {}, {}
            for seed in SEEDS:
                record = cost["native_selection_locks"][dataset][space][str(seed)]; lock = prior.verified_json(record, check); guarded.append(record); locks[(space, str(seed))] = lock
                state = prior.verified_arrays(lock["selected_state_artifact"], check); head_states[str(seed)] = state
                checkpoint = next(cp for cp in condition["native"][str(seed)]["checkpoints"] if cp["update"] == lock["selected_update"])
                digest = prior.torch_state_sha(state)
                require(digest == checkpoint["state_sha256"] == prepared["model_states_before"][str(seed)] and set(state) == {"decoder", "head.weight", "head.bias"}, "INDEPENDENT_SELECTED_NATIVE_GLOBAL_D_HEAD_STATE_HASH")
                hashes["heads"][str((dataset, space, str(seed)))] = digest; totals["native_head_state_hashes_checked"] += 1
                saved = prior.verified_arrays(condition["native"][str(seed)]["artifacts"][dev], check)
                exact(saved["origins"], raw["origins"], "selected_native_cost_reference_origins")
                predictions[space + "_NATIVE_s" + str(seed)] = native_predictions[str(seed)] = saved["prediction"]
            bolt = prior.verified_arrays(condition["Bolt"][dev]["artifact"], check); predictions[space + "_BOLT"] = bolt["prediction"]
            exact(bolt["origins"], raw["origins"], "selected_Bolt_cost_reference_origins")
            prechecks[space] = {}
            for backend in ("BOLT_NUMPY", "NATIVE_TORCH32"):
                d = prior.verified_arrays(condition["backends"][backend]["state_artifact"], check)["decoder_FP32"]
                require(d.dtype == np.float32 and d.shape == (k, channels) and np.isfinite(d).all(), "STATIC_GLOBAL_FIT_DECODER_FP32")
                hashes["global_decoders"][str((dataset, space, backend))] = array_record(d)["sha256"]; totals["global_decoder_state_hashes_checked"] += 1
                entry = row["physical_backend_prechecks"][space][backend]
                prechecks[space][backend] = physical_precheck(entry, prior.verified_arrays(entry["artifact"], check), physical, raw, fixed, e, d, backend, head_states, {"BOLT": bolt["prediction"]} if backend == "BOLT_NUMPY" else native_predictions, bolt["latent_future"])
                totals["physical_backend_prechecks_checked"] += 1; totals["physical_first8_windows_checked"] += 8
                totals["native_first8_prediction_checks"] += prechecks[space][backend]["independent_native_prediction_checks"]
        metadata = route_metadata(srow, dev, locks, cost["native_selection_locks"][dataset]) if all((space, str(seed)) in locks for space in SPACES for seed in SEEDS) else {}
        if dataset in cost["cost"]:
            details = audit_cost_dataset(cost["cost"][dataset], physical, raw, predictions, metadata, check, latest_selection_seal)
            _, literal = json_record(Path(binding["path"]).parent / ("cost_" + dataset) / "RESULT.json", check)
            require(literal == cost["cost"][dataset], "EXACT_SAVED_WHOLE_COST_RESULT_JSON")
            report["cost_details"][dataset] = details; totals["cost_cells_checked"] += details["cells_checked"]; totals["recorded_timer_samples_checked"] += details["recorded_timer_samples_checked"]
        else: require(not producer_complete, "COMPLETE_COST_MISSING_DATASET_TIMERS")
        report["datasets"][dataset] = {"physical_backend_prechecks": prechecks, "physical_bridge_canonical": row["physical_bridge_canonical"]}
    for row in science["datasets"].values():
        for space in row["spaces"].values():
            for split in space["Bolt"].values(): guarded.append(split["artifact"]); totals["selected_forecast_file_hashes_checked"] += 1
            for head in space["native"].values():
                for artifact in head["artifacts"].values(): guarded.append(artifact); totals["selected_forecast_file_hashes_checked"] += 1
    for record in guarded: check(); require(prior.file_sha(record["path"]) == record["sha256"], "POST_AUDIT_OLD_SCIENCE_CACHE_HEAD_LOCK_SOURCE_BYTES_UNCHANGED")
    require(hashes == cost["fixed_states_before"] == cost["fixed_states_after"], "INDEPENDENT_SIX_E_SIX_PROJECTIONS_TWELVE_D_TWELVE_HEAD_HASHES")
    if producer_complete:
        expected = {"cost_cells_checked": 40, "recorded_timer_samples_checked": 400, "physical_backend_prechecks_checked": 12, "physical_first8_windows_checked": 96, "native_first8_prediction_checks": 12, "native_selection_locks_checked": 12, "subset_selection_locks_checked": 2, "encoder_state_hashes_checked": 6, "native_projection_state_hashes_checked": 6, "global_decoder_state_hashes_checked": 12, "native_head_state_hashes_checked": 12, "selected_forecast_file_hashes_checked": 54, "canonical_physical_bridges_checked": 2}
        require(all(totals[key] == number for key, number in expected.items()) and cost["completed_cost_cells"] == 40 and cost["recorded_timer_samples"] == cost["completed_cell_timer_samples"] == 400 and cost["failed_paths"] == [] and report["partial_cost_paths_not_accepted"] == [], "COMPLETE_FIXED_SUBSET_WHOLE_COST_AUDIT_REGISTERED_COVERAGE")
    return report


def execute(cost_path, output_path):
    from continuation_runtime_v1 import Meter, write_new
    output = Path(output_path).resolve(); require(not output.exists(), "FRESH_RAW_SUBSET_COST_AUDIT_OUTPUT")
    output.parent.mkdir(parents=True, exist_ok=True)
    meter = Meter("audit_raw_subset_cost_" + output.stem, gpu=False)
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
        meter.protected_check(); require(prior.file_sha(__file__) == source["sha256"], "INDEPENDENT_COST_AUDITOR_SOURCE_UNCHANGED"); json.dumps(report, allow_nan=False)
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}; report = {"status": "FAILED_PRESERVED", "failure": failure, "full_scope_PASS_claimed": False}
    finally:
        report["audit_source"] = source; report["resource_terminal"] = meter.close(report["status"], failure); write_new(output, report)
    print(json.dumps({"status": report["status"], "totals": report.get("totals"), "failure": failure}), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--cost", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); raise SystemExit(execute(args.cost, args.output))
