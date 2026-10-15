"""Recorded TUNE whole-cost audit; no F, GPU, optimizer or timer remeasurement."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import math
import statistics

import audit_projection_results_v1 as prior
import audit_linear_e_adaptation_v1 as linear
import audit_raw_subset_cost_v2 as shared

HERE, ROOT = prior.HERE, prior.ROOT
DATASETS, LAGS, SEEDS = linear.DATASETS, linear.LAGS, linear.SEEDS
ROUTES = ("RAW_BOLT", "L0_E_FIXED_s92601", "L0_E_FIXED_s92602", "L0_E_LEARN_s92601", "L0_E_LEARN_s92602", "L7_E_FIXED_s92601", "L7_E_FIXED_s92602", "L7_E_LEARN_s92601", "L7_E_LEARN_s92602")
COST_SHA = "5038bc1cd6c52fadf4b9397b3eabb9f1bcf3085c9901d954222d43835c05a97d"
TRAIN_SHA = "3f0cbd9176f2e7882778452e537564aab7445e1b88c1534fd48b1bce7ef639a4"
AUDIT_SHA = "9796b4ed53f74186bc2513db56fe321c2d450cdd9bffbb7c5f341766e73ee36d"
require, exact, parity = prior.require, linear.exact, shared.parity_checked
np = None


def initialize_numpy():
    global np
    linear.initialize_numpy(); shared.initialize_numpy(); np = linear.np


def schedule():
    order = np.random.default_rng(20261015).permutation(9)
    return [[ROUTES[index] for index in np.roll(order, -repeat)] for repeat in range(9)]


def bound(record, check):
    check(); require(Path(record["path"]).is_absolute() and prior.file_sha(record["path"]) == record["sha256"], "EXACT_ABSOLUTE_FILE_BINDING")
    if "bytes" in record: require(Path(record["path"]).stat().st_size == record["bytes"], "EXACT_FILE_BYTES")


def source_contract(helper, runner, original):
    def functions(text): return {n.name: n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
    h, r, old = map(functions, (helper, runner, original))
    for name in ("fill_vectorized", "preprocess", "parity", "whole_predict", "balanced_schedule"): require(ast.dump(h[name]) == ast.dump(old[name]), "UNCHANGED_CANONICAL_TIMER_FUNCTION:" + name)
    constants = {}
    for node in ast.parse(helper).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1: continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            try: constants[target.id] = ast.literal_eval(node.value)
            except (ValueError, TypeError): pass
        elif isinstance(target, ast.Tuple): constants.update(zip((v.id for v in target.elts), ast.literal_eval(node.value)))
    require(constants["ROUTES"] == ROUTES and constants["ORDER_SEED"] == 20261015 and constants["NORM_ATOL"] == 1e-5 and constants["NORM_RTOL"] == 2e-6 and constants["PHYSICAL_ATOL"] == 2e-3 and constants["PHYSICAL_RTOL"] == 2e-5 and "WARMUP, REPEATS = 2, 9" in helper, "NINE_ROUTE_CONSTANTS_AND_UNCHANGED_TOLERANCES")
    callback = next(n for n in r["adapter_predictor"].body if isinstance(n, ast.FunctionDef))
    require("model(tensor(history), bolt).detach().cpu().numpy().copy()" in ast.unparse(callback) and "torch.no_grad()" in ast.unparse(callback), "ACTUAL_TORCH_E_F_LAG_D_AND_CPU_RETURN_CALLBACK")
    timer = [n for n in ast.walk(h["measure_cost"]) if isinstance(n, ast.Try) and len(n.body) == 3 and "perf_end_ns, unix_end_ns = (time.perf_counter_ns(), time.time_ns())" in ast.unparse(n)]
    require(len(timer) == 1 and [ast.unparse(n) for n in timer[0].body] == ["normalized, physical_value = whole_predict(predictors[route], values, masks, mean, std)", "synchronize()", "perf_end_ns, unix_end_ns = (time.perf_counter_ns(), time.time_ns())"], "WHOLE_ACTUAL_CALLBACK_AND_SYNC_INSIDE_NS_BOUNDARY")
    require("canonical_tune_bridge(original, data)" in runner and 'raw_input_sources"]["tune"]' in runner and 'reference_split": "tune"' in runner and 'selected["DEV"]' not in runner, "REGISTERED_TUNE_NOT_DEV_INPUT_AND_SELECTED_CP_REFERENCE")
    return {"actual_E_F_lag_D_physical_CPU_return_inside_timer": True, "source_file_state_hashes_and_fit_outside_timer": True, "nine_full_rotations": True, "TUNE_first_B_origins": True, "independent_F_or_timer_replay": False}


def precheck(entry, arrays, physical, raw, state, reference, lags):
    small = {key: raw[key] if key in ("mean", "std") else raw[key][:8] for key in ("history", "target", "observed", "raw_last", "origins", "mean", "std", "history_observed")}
    normalized = prior.causal_physical_preprocess(physical["history_physical"][:8], physical["history_observed"][:8], raw["mean"], raw["std"])
    exact(arrays["normalized_history"], normalized, "physical_causal_normalization")
    exact(arrays["history_observed"], small["history_observed"], "physical_observation_mask"); exact(arrays["origins"], small["origins"], "physical_origins")
    decisions = {"physical_raw_origins": physical["origins"][:8].tobytes() == small["origins"].tobytes(), "selected_raw_origins": reference["origins"][:8].tobytes() == small["origins"].tobytes(), "physical_raw_history_observed": physical["history_observed"][:8].tobytes() == small["history_observed"].tobytes(), "normalized_history_byte_exact": normalized.tobytes() == small["history"].tobytes()}
    require(entry["exact_lineage"] == decisions and all(decisions.values()) and entry["input_backend_pass"] is entry["prediction_backend_pass"] is True and entry["intermediate_parity_required_for_acceptance"] is False, "EXACT_PHYSICAL_LINEAGE_AND_PREDICTION_PARITY_ACCEPTANCE")
    parity(entry["input_checks"]["history"], normalized, small["history"]); parity(entry["input_checks"]["currentC"], normalized[:, -1], small["raw_last"])
    saved = {**small, **{key: arrays[key] for key in ("prediction", "latent_history", "latent_future", "lag_features")}}
    linear.replay(saved, state, small, lags)
    parity(entry["prediction_checks"]["normalized"], arrays["prediction"], reference["prediction"][:8])
    parity(entry["prediction_checks"]["physical"], (arrays["prediction"] * raw["std"] + raw["mean"]).astype(np.float32), (reference["prediction"][:8] * raw["std"] + raw["mean"]).astype(np.float32), physical=True)
    for key in ("latent_history", "latent_future", "lag_features"): parity(entry["intermediate_checks"][key], arrays[key], reference[key][:8], must_pass=False)
    return {"windows_checked": 8, "independent_NumPy_E_lag_D_replay": True, "independent_F_calls": 0, "intermediate_parity_is_separate_from_registered_output_acceptance": True}


def check_dataset(cost, physical, raw, predictions, metadata, check, seal):
    require(cost["routes"] == list(ROUTES) and cost["schedule"] == schedule() and cost["order_seed"] == 20261015 and cost["warmup_each_route"] == 2 and cost["repeats"] == 9 and cost["expected_cells_this_dataset"] == 18 and cost["expected_cells_two_datasets"] == 36, "NINE_FULL_ROTATIONS_AND_ALL_DECLARED_ROWS")
    require(cost["new_training_updates"] == 0 and cost["encoder_refit_in_cost"] is cost["per_origin_decoder_fit"] is cost["fit_input_binding_hashes_inside_timer"] is cost["protected_TEST_accessed"] is cost["independent_holdout"] is False and cost["fixed_FIT_decoder_loaded_before_timer"] is True, "NO_FIT_OR_HASHES_INSIDE_TIMER")
    lineage = cost["source_lineage"]
    require(lineage["whole_cost_source_sha256"] == prior.file_sha(HERE / "whole_linear_e_cost_v1.py") and Path(lineage["copied_timer_source"]).resolve() == HERE / "whole_causal_lag_cost_v1.py" and lineage["copied_timer_source_sha256"] == prior.file_sha(HERE / "whole_causal_lag_cost_v1.py") and lineage["copied_preprocess_source_sha256"] == prior.file_sha(HERE.parent / "cost_readout_v1.py") and lineage["copied_functions"] == ["fill_vectorized", "preprocess", "parity"] and lineage["legacy_scientific_runtime_imported"] is False, "CANONICAL_TIMER_PREPROCESS_SOURCE_LINEAGE")
    exact(prior.causal_physical_preprocess(physical["history_physical"], physical["history_observed"], raw["mean"], raw["std"]), raw["history"], "all_TUNE_physical_causal_normalization")
    for key, value in (("physical_past", physical["history_physical"]), ("history_observed", physical["history_observed"]), ("FIT_mean", raw["mean"]), ("FIT_std", raw["std"]), ("origins", raw["origins"])): require(cost["input_array_metadata"][key] == shared.array_record(value), "EXACT_COST_INPUT_ARRAY_SHA")
    rows = {(r["route"], r["batch"]): r for r in cost["rows"]}; require(set(rows) == {(r, b) for r in ROUTES for b in (1, 8)} and len(cost["rows"]) == 18, "NO_MISSING_DUPLICATE_OR_DROPPED_COST_ROWS")
    totals, chronology = {"complete_cells": 0, "saved_samples": 0}, []; details = []
    for (route, batch), row in rows.items():
        raw_row = rows["RAW_BOLT", batch]; base = {s["repeat"]: (s["perf_end_ns"] - s["perf_start_ns"]) / 1e9 for s in raw_row["samples"]}
        times = [(s["perf_end_ns"] - s["perf_start_ns"]) / 1e9 for s in row["samples"]]; ratios = [t / base[s["repeat"]] if s["repeat"] in base else None for t, s in zip(times, row["samples"])]
        require(row["seconds"] == times and row["paired_RAW_ratios"] == ratios and row["paired_RAW_seconds"] == [base.get(s["repeat"]) for s in row["samples"]] and row["repeats"] == len(times), "NS_TIMES_AND_SAME_REPEAT_PAIRED_RATIOS")
        require([s["repeat"] for s in row["samples"]] == sorted({s["repeat"] for s in row["samples"]}) and all(0 <= s["repeat"] < 9 for s in row["samples"]), "UNIQUE_REGISTERED_REPEAT_NUMBERS")
        complete = row["status"] == "COMPLETE"
        require(row["selected_metadata"] == cost["source_metadata"][route] == metadata[route] and row["origins"] == raw["origins"][:batch].tolist() and row["warmup_each_route"] == 2 and row["native_head_loaded_or_timed"] is row["per_origin_decoder_fit"] is row["declared_route_deleted"] is row["memory_counter_reset_inside_module"] is row["stage_decomposition"] is False, "EXACT_ROUTE_SELECTED_STATE_AND_TIMED_SCOPE")
        if not times: require(not complete and row["timed_outputs"] is None, "EMPTY_FAILED_PATH_PRESERVED"); continue
        median = statistics.median(times); paired = statistics.median(ratios) if all(r is not None for r in ratios) else None
        require(row["median_seconds"] == median and row["median_seconds_per_window"] == median / batch and row["paired_RAW_median_ratio"] == paired, "INDEPENDENT_MEDIANS_AND_PER_WINDOW_ARITHMETIC")
        if complete: require(len(times) == 9 and row["failure"] is None, "COMPLETE_ROW_ALL_NINE_SAMPLES")
        refs, stack = (prior.verified_arrays(row[key], check) for key in ("reference", "timed_outputs"))
        require(all(refs[key].dtype == np.float32 and refs[key].shape == (batch, 48, len(raw["mean"])) for key in ("normalized", "physical", "scientific_normalized")) and all(stack[key].dtype == np.float32 and stack[key].shape == (len(times), batch, 48, len(raw["mean"])) for key in ("normalized", "physical")), "FP32_REFERENCE_AND_ALL_TIMER_OUTPUT_SHAPES")
        for arrays in (refs, stack): exact(arrays["origins"], raw["origins"][:batch], "reference_stack_origins")
        exact(refs["scientific_normalized"], predictions[route][:batch], "exact_selected_TUNE_reference")
        exact(refs["physical"], (refs["normalized"] * raw["std"] + raw["mean"]).astype(np.float32), "reference_CPU_physical_inverse")
        prior.close(refs["normalized"], predictions[route][:batch], "reference_B_batch_vs_selected_canonical32")
        require(len(row["samples"]) == len(row["parity"]) == len(row["memory"]), "ALL_SAVED_SAMPLE_LIST_LENGTHS")
        for i, sample in enumerate(row["samples"]):
            require(sample["dataset"] == cost["dataset"] and sample["route"] == route and sample["batch"] == batch and sample["repeat"] == sample["block"] and sample["position"] == row["positions"][i] == schedule()[sample["repeat"]].index(route) and sample["seconds"] == times[i] > 0 and sample["unix_end_ns"] >= sample["unix_start_ns"] > seal * 1e9, "SAMPLE_IDENTITY_NS_AFTER_LOCKS_AND_ROTATION")
            require(sample["parity"] == row["parity"][i] and sample["memory"] == row["memory"][i], "ROW_SAMPLE_PARITY_MEMORY_BINDING")
            saved = prior.verified_arrays(sample["sample_artifact"], check)
            literal = json.loads(Path(sample["sample_artifact"]["path"]).with_suffix(".json").read_text(encoding="utf-8")); require(literal == sample, "EXACT_SAVED_SAMPLE_JSON")
            for key in ("normalized", "physical"): exact(saved[key], stack[key][i], "saved_sample_stack_member")
            exact(saved["origins"], raw["origins"][:batch], "same_first_B_TUNE_origins"); exact(saved["physical"], (saved["normalized"] * raw["std"] + raw["mean"]).astype(np.float32), "timed_CPU_physical_inverse")
            for key, expected, physical_units in (("normalized", refs["normalized"], False), ("physical", refs["physical"], True), ("scientific_normalized", predictions[route][:batch], False), ("scientific_physical", (predictions[route][:batch] * raw["std"] + raw["mean"]).astype(np.float32), True)): parity(sample["parity"][key], saved["physical" if physical_units else "normalized"], expected, physical=physical_units, must_pass=complete)
            memory = sample["memory"]; require(memory["scope"] == "Shared resident frozen F and all available selected adapters; no exclusive per-method allocation" and all(type(v) is int and v >= 0 for k, v in memory.items() if k != "scope") and memory["max_reserved_VRAM_bytes"] >= memory["reserved_VRAM_bytes"] >= memory["allocated_VRAM_bytes"] and memory["max_reserved_VRAM_bytes"] >= memory["max_allocated_VRAM_bytes"] >= memory["allocated_VRAM_bytes"], "SHARED_PROCESS_MEMORY_COUNTER_SCOPE")
            chronology.append((batch, sample["repeat"], sample["position"], sample))
        exact(stack["block"], np.asarray([s["repeat"] for s in row["samples"]], np.int64), "timer_stack_repeats"); exact(stack["position"], np.asarray(row["positions"], np.int64), "timer_stack_positions")
        totals["complete_cells"] += int(complete); totals["saved_samples"] += len(times)
        details.append({"route": route, "batch": batch, "samples": len(times), "median_seconds": median, "median_seconds_per_window": median / batch, "ratio_of_RAW_medians": median / statistics.median(base.values()), "median_paired_RAW_ratio": paired, "status": row["status"]})
    chronology.sort(key=lambda r: r[:3])
    for a, b in zip(chronology, chronology[1:]): require(a[3]["perf_end_ns"] <= b[3]["perf_start_ns"] and all(a[3]["memory"][k] <= b[3]["memory"][k] for k in ("max_allocated_VRAM_bytes", "max_reserved_VRAM_bytes")), "SERIAL_NS_AND_UNRESET_SHARED_MEMORY_PEAKS")
    require(cost["completed_cost_cells"] == totals["complete_cells"] and cost["recorded_timer_samples"] == totals["saved_samples"] and cost["attempted_timer_calls"] >= totals["saved_samples"], "ACTUAL_RECORDED_COUNTS")
    if cost["status"] == "COMPLETE": require(totals == {"complete_cells": 18, "saved_samples": 162} and not cost["failed_paths"], "COMPLETE_DATASET_18_162")
    return {**totals, "rows": details, "independent_F_or_timer_replayed": False}


def audit_result(path, meter):
    require(isinstance(COST_SHA, str) and len(COST_SHA) == 64, "ACTUAL_COST_RESULT_NOT_YET_REGISTERED")
    check = meter.check; binding = linear.file_record(path); require(binding["sha256"] == COST_SHA, "EXACT_ACTUAL_COST_RESULT_SHA")
    cost = prior.verified_json(binding, check); require(cost["status"] in ("COMPLETE", "PARTIAL_COMPLETED") and cost["cost_split"] == "tune" and cost["expected_cost_cells"] == 36 and cost["expected_timer_samples"] == 324, "COMPLETED_TUNE_COST_SCOPE")
    training = prior.verified_json(cost["linear_e_diagnostic"], check); accepted = prior.verified_json(cost["linear_e_audit"], check)
    require(cost["linear_e_diagnostic"]["sha256"] == TRAIN_SHA and cost["linear_e_audit"]["sha256"] == AUDIT_SHA and training["status"] == "COMPLETE" and accepted["status"] == "PASS" and accepted["diagnostic"] == cost["linear_e_diagnostic"], "EXACT_COMPLETE_ADAPTATION_AND_PASS")
    require(all(accepted["totals"][key] == value for key, value in {"adapter_fits_checked": 16, "checkpoint_states_checked": 80, "checkpoint_TUNE_predictions_checked": 80, "selected_FIT_DEV_predictions_checked": 32, "CP0_FIT_DEV_baselines_checked": 8, "logged_updates_checked": 9600, "probe_cases_checked": 4, "selection_locks_checked": 16}.items()), "ACCEPTED_FULL_SCIENCE_COVERAGE")
    for key in ("probe_result", "causal_lag_diagnostic", "causal_lag_audit", "selection_registry"): require(cost[key] == accepted[key] == training[key], "UNCHANGED_SCIENTIFIC_PARENT:" + key)
    science = prior.verified_json(cost["causal_lag_diagnostic"], check); subset = prior.verified_json(science["raw_subset_diagnostic"], check); original = prior.verified_json(subset["source_producer"], check)
    source = source_contract(*[(HERE / name).read_text(encoding="utf-8-sig") for name in ("whole_linear_e_cost_v1.py", "run_linear_e_cost_v1.py", "whole_causal_lag_cost_v1.py")])
    require(cost["new_training_updates"] == 0 and cost["selected_E_D_center_loaded_outside_timer"] is cost["all_source_file_state_metadata_hashes_outside_timer"] is True, "SELECTED_STATE_READY_OUTSIDE_TIMER")
    for key in ("new_backbone_training", "new_encoder_decoder_fit", "new_CP_or_seed_or_lag_selection", "native_predictor_loaded_or_timed", "per_origin_decoder_fit", "cached_future_returned_instead_of_actual_F_inside_timer", "protected_TEST_accessed", "independent_holdout", "strict_K_only"): require(cost[key] is False, "ACTUAL_FIXED_MODEL_COST_SCOPE:" + key)
    guarded = [binding, *[cost[key] for key in ("linear_e_diagnostic", "linear_e_audit", "probe_result", "causal_lag_diagnostic", "causal_lag_audit", "selection_registry", "cost_plan")], *cost["source_snapshots"], *cost["source_records"], *cost["input_records"], *cost["protected_prior_records"]]
    for item in guarded:
        bound(item, check)
        if "snapshot_path" in item: require(prior.file_sha(item["snapshot_path"]) == item["sha256"], "EXACT_EXECUTED_COST_SOURCE_SNAPSHOT")
    require(cost["F_before"] == cost["F_after"] == training["F_before"] == training["F_after"] == original["F_before"] and set(cost["F_before"]) == {"old_algorithm", "bi_algorithm"}, "UNCHANGED_ORIGINAL_F_DUAL_SIGNATURE")
    for name in ("model_binary", "model_config"):
        for suffix in ("_before", "_after"): require(cost[name + suffix]["path"] == training[name + "_before"]["path"] and cost[name + suffix]["sha256"] == training[name + "_before"]["sha256"], "SAME_FROZEN_MODEL_FILE"); bound(cost[name + suffix], check)
    for key in training["runtime_behavior_contract"]:
        require(cost["runtime_behavior_contract"][key] == training["runtime_behavior_contract"][key], "SAME_F_RUNTIME_NORMALIZATION_CONTRACT:" + key)
    registry = prior.verified_json(cost["selection_registry"], check); require(len(registry["conditions"]) == 16 and registry["status"] == "ALL_16_RESOLVED_BEFORE_DEV", "SAME_16_TUNE_SELECTIONS")
    locks = {}; seal = registry["sealed_unix"]
    for item in registry["conditions"]:
        key = item["dataset"], item["lag"], item["learn_E"], item["seed"]; condition = key[0] + "_" + key[1] + ("_E_LEARN_s" if key[2] else "_E_FIXED_s") + str(key[3])
        require(key not in locks and cost["selection_locks"][condition] == item["selection_lock"], "UNIQUE_EXACT_COST_SELECTION_LOCK")
        lock = prior.verified_json(item["selection_lock"], check); require(lock["status"] == "COMPLETE" and lock["selected_state"] == training["datasets"][key[0]]["conditions"][condition]["selected_state"], "LOCKED_SELECTED_STATE_NOT_RESELECTED"); locks[key] = item["selection_lock"]; guarded.append(item["selection_lock"]); seal = max(seal, lock["sealed_unix"])
    report = {"status": "PASS" if cost["status"] == "COMPLETE" else "PARTIAL_AUDITED_NO_FULL_PASS", "cost": binding, **{key: cost[key] for key in ("linear_e_diagnostic", "linear_e_audit", "probe_result", "causal_lag_diagnostic", "causal_lag_audit", "selection_registry", "selection_locks")}, "source_contract": source, "totals": {"cost_cells_checked": 0, "recorded_timer_samples_checked": 0, "selected_state_hashes_checked": 0, "selection_locks_checked": len(locks), "physical_prechecks_checked": 0, "physical_first8_windows_checked": 0, "canonical_TUNE_bridges_checked": 0}, "datasets": {}, "protected_TEST_accessed": False, "independent_F_or_timer_replayed": False, "per_method_exclusive_memory_claimed": False, "scope": "Recorded whole TUNE cost and saved actual-F outputs; independent NumPy E/lag/D, physical normalization, parity, ns schedules and two ratios. No new F, timer, optimizer or independent generalization test."}
    hashes = {}
    for data, (c, k, _, _) in DATASETS.items():
        row = cost["datasets"][data]; raw = prior.verified_arrays(subset["datasets"][data]["raw_input_sources"]["tune"], check); prior.check_raw_split(raw, channels=c, windows=64)
        proof = row["canonical_physical_bridge"]; require(proof["split"] == "tune" and proof["dataset"] == data and proof["original_first_parent"] == original["source_parent_results"][data], "CANONICAL_TUNE_NOT_DEV_ALIAS")
        parent = prior.verified_json(proof["original_first_parent"], check); require(proof["canonical_input_bridge"] == parent["bridge_provenance"]["input_bridge"], "FULL_ORIGINAL_INPUT_BRIDGE_BINDING")
        bridge = prior.verified_json(proof["canonical_input_bridge"], check); candidates = [a for a in bridge["artifacts"] if a["dataset"] == data and a["split"] == "tune"]
        require(len(candidates) == 1 and proof["canonical_record"] == {**candidates[0], "path": str(Path(candidates[0]["path"]).resolve())} and proof["normalized_raw_input"] == original["datasets"][data]["inputs"]["tune"] and proof["complete_array_manifest_preserved"] is True and proof["inverse_normalization_used"] is proof["DEV_physical_alias_used"] is proof["loader_file_SHA_member_shape_dtype_SHA_checks_relaxed"] is proof["numeric_tolerance_changed"] is False, "EXACT_EXISTING_FULL_TUNE_PHYSICAL_RECORD_NO_RECONSTRUCTION_OR_RELAXATION")
        physical = prior.verified_arrays(proof["canonical_record"], check)
        for field in ("history_observed", "origins", "mean", "std"): exact(physical[field], raw[field], "same_TUNE_bridge_" + field)
        report["totals"]["canonical_TUNE_bridges_checked"] += 1
        predictions = {"RAW_BOLT": raw["raw_forecast"]}; metadata = {"RAW_BOLT": {"source": subset["datasets"][data]["raw_input_sources"]["tune"], "scope": "actual full-C frozen Bolt on registered TUNE origins"}}; checks = {}
        for key, lock_record in locks.items():
            if key[0] != data: continue
            _, lag, learn, seed = key; route = lag + ("_E_LEARN_s" if learn else "_E_FIXED_s") + str(seed); condition = data + "_" + route; selected = training["datasets"][data]["conditions"][condition]
            state = prior.verified_arrays(selected["selected_state"], check); require(linear.state_hash(state) == selected["selected_state_sha256"], "EXACT_SELECTED_E_D_CENTER_TENSOR_HASH")
            hashes[str(key)] = {"state_sha256": linear.state_hash(state), "arrays": {name: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() for name, value in state.items()}}
            cp = [p for p in selected["checkpoints"] if p["update"] == selected["selected_update"]]; require(len(cp) == 1 and cp[0]["state"] == selected["selected_state"] and cp[0]["tune"] == selected["selected_tune"], "SAME_SELECTED_TUNE_CP_REFERENCE")
            reference = prior.verified_arrays(cp[0]["artifact"], check); predictions[route] = reference["prediction"]
            metadata[route] = {"training_condition": condition, "lag": lag, "lags": list(LAGS[lag]), "learn_E": learn, "seed": seed, "selected_update": selected["selected_update"], "selected_state": selected["selected_state"], "selection_lock": lock_record, "reference": cp[0]["artifact"], "reference_split": "tune", "reference_prediction_array": "prediction", "actual_F_forward": True, "current_C_anchor": True, "known_K_past_buffer": True, "strict_K_only": False, "per_origin_fit": False}
            entry = row["physical_prechecks"][route]; checks[route] = precheck(entry, prior.verified_arrays(entry["artifact"], check), physical, raw, state, reference, LAGS[lag])
            report["totals"]["selected_state_hashes_checked"] += 1; report["totals"]["physical_prechecks_checked"] += 1; report["totals"]["physical_first8_windows_checked"] += 8
        details = check_dataset(cost["cost"][data], physical, raw, predictions, metadata, check, seal)
        literal = json.loads((Path(binding["path"]).parent / ("cost_" + data) / "RESULT.json").read_text(encoding="utf-8")); require(literal == cost["cost"][data], "SAVED_DATASET_COST_JSON_IDENTITY")
        report["datasets"][data] = {"prechecks": checks, "cost": details}; report["totals"]["cost_cells_checked"] += details["complete_cells"]; report["totals"]["recorded_timer_samples_checked"] += details["saved_samples"]
    require(hashes == cost["selected_states_before"] == cost["selected_states_after"], "ALL_16_SELECTED_STATE_ARRAY_AND_TENSOR_HASHES_BEFORE_AFTER")
    for item in guarded: bound(item, check)
    if cost["status"] == "COMPLETE": require(report["totals"] == {"cost_cells_checked": 36, "recorded_timer_samples_checked": 324, "selected_state_hashes_checked": 16, "selection_locks_checked": 16, "physical_prechecks_checked": 16, "physical_first8_windows_checked": 128, "canonical_TUNE_bridges_checked": 2} and cost["selected_adapters_loaded"] == 16 and cost["completed_cost_cells"] == 36 and cost["recorded_timer_samples"] == 324 and not cost["failed_paths"], "FULL_CURRENT_36_324_TUNE_COST_COVERAGE")
    return report


def execute(path, output):
    from continuation_runtime_v1 import Meter, write_new
    output = Path(output).resolve(); require(not output.exists(), "FRESH_COST_AUDIT_OUTPUT")
    meter = Meter("audit_linear_e_cost_" + output.stem, gpu=False); failure = None; source = linear.file_record(__file__)
    try:
        require(isinstance(COST_SHA, str) and len(COST_SHA) == 64, "ACTUAL_COST_RESULT_NOT_YET_REGISTERED"); meter.protected_check(); initialize_numpy()
        import sys
        sys.path.insert(0, str(ROOT / "research/smallk_failure_followup_20261014"))
        from prepare_inputs import install_protected_test_guard
        install_protected_test_guard()
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2): report = audit_result(path, meter)
        meter.protected_check(); require(prior.file_sha(__file__) == source["sha256"], "AUDIT_SOURCE_UNCHANGED"); json.dumps(report, allow_nan=False)
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}; report = {"status": "FAILED_PRESERVED", "failure": failure}
    finally:
        report["audit_source"] = source; report["resource_terminal"] = meter.close(report["status"], failure); write_new(output, report)
    print(json.dumps({"status": report["status"], "totals": report.get("totals"), "failure": failure}), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--cost", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); raise SystemExit(execute(args.cost, args.output))
