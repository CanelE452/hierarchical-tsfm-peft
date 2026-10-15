"""Independent six-route saved whole-cost/gate audit; no F or timer replay."""
from pathlib import Path
import argparse
import ast
import json
import sys

import audit_projection_results_v1 as prior
import audit_decoder_metric_cost_v1 as cost_audit
import audit_past_constant_gate_v1 as gate_audit

np = sys.modules.get("numpy")
HERE, BASE = prior.HERE, prior.BASE
ROUTES = ("RAW_BOLT", "RAW_BOLT_GATED", "PCA_PINV_BOLT", "PCA_PINV_BOLT_GATED",
          "FIT_TARGET_PINV_BOLT", "FIT_TARGET_PINV_BOLT_GATED")
DATASETS = prior.DATASETS
require, exact = prior.require, gate_audit.exact
array_record, assert_binding = cost_audit.array_record, cost_audit.assert_binding
recorded_timer = cost_audit.recorded_timer


def initialize_numpy():
    global np
    cost_audit.initialize_numpy()
    gate_audit.initialize_numpy()
    np = sys.modules["numpy"]


def independent_schedule():
    order = np.random.default_rng(20261015).permutation(6)
    return [[ROUTES[index] for index in np.roll(order, -repeat)] for repeat in range(6)]


def source_contract(original_text, helper_text, runner_text):
    original, helper, runner = (ast.parse(text) for text in (original_text, helper_text, runner_text))
    def functions(tree):
        return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    old_functions, new_functions, calls = functions(original), functions(helper), functions(runner)
    for name in ("sha", "array_record", "_jsonable", "write_json_once", "save_outputs", "fill_vectorized", "preprocess", "parity"):
        require(ast.dump(old_functions[name], include_attributes=False) == ast.dump(new_functions[name], include_attributes=False),
                "UNCHANGED_TIMING_SAVED_IO_PREPROCESS_FUNCTION:" + name)
    def timed_sequence(function):
        sequences = []
        for node in ast.walk(function):
            if not isinstance(node, ast.For):
                continue
            names = [statement.targets[0].id if isinstance(statement, ast.Assign)
                     and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name) else None for statement in node.body]
            if "perf_start_ns" in names and "perf_end_ns" in names:
                start, end = names.index("perf_start_ns"), names.index("perf_end_ns")
                sequences.append([ast.dump(statement, include_attributes=False) for statement in node.body[start:end + 1]])
        require(len(sequences) == 1, "ONE_AUTHENTIC_PERF_COUNTER_TIMED_SEQUENCE")
        return sequences[0]
    require(timed_sequence(old_functions["measure_cost"]) == timed_sequence(new_functions["measure_cost"]),
            "ORIGINAL_WHOLE_CALL_AND_FINAL_SYNC_INSIDE_PERF_TIMER")
    whole_calls = [node for node in ast.walk(new_functions["whole_predict"]) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Name) and node.func.id == "predictor"]
    require(len(whole_calls) == 1 and [ast.unparse(arg) for arg in whole_calls[0].args] == ["normalized_history", "observed"],
            "ORIGINAL_TRUE_OBSERVED_MASK_EXPLICIT_IN_TIMED_CALLBACK")
    gated = [node for node in calls["gated_predictor"].body if isinstance(node, ast.FunctionDef)]
    require(len(gated) == 1 and [arg.arg for arg in gated[0].args.args] == ["history", "history_observed"], "GATED_CALLBACK_PAST_ONLY_SIGNATURE")
    invoked = [node for node in ast.walk(gated[0]) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    names = [node.func.id for node in invoked]
    require(names == ["ungated", "past_constant_gate", "apply_output_gate"], "GATE_COMPUTED_INSIDE_EACH_TIMED_PREDICTOR_CALL")
    require([ast.unparse(arg) for arg in invoked[1].args] == ["history", "history_observed"], "GATE_NEVER_TAKES_TARGET_OR_PRECOMPUTED_FLAGS")
    return {"unchanged_saved_IO_preprocess_functions": 8, "original_perf_timed_call_and_sync": True,
            "explicit_true_past_mask_to_callback": True, "gate_computation_inside_each_timed_callback": True}


def validate_route_metadata(route, actual, expected):
    require(actual == expected, "EXACT_SIX_ROUTE_SAVED_CPU_GATE_REFERENCE_METADATA:" + route)
    gated = route.endswith("_GATED")
    base = route[:-6] if gated else route
    require(base in ("RAW_BOLT", "PCA_PINV_BOLT", "FIT_TARGET_PINV_BOLT") and actual["base_method"] == base
            and actual["gated"] == gated and actual["reference_prediction_array"] == base + ("_gated_prediction" if gated else "_ungated_prediction")
            and actual["shared_resident_frozen_F"] is True and actual["native_head_loaded_or_timed"] is False
            and actual["history_observed_explicitly_passed"] is True and actual["past_C_metadata_added"] == gated
            and actual["same_decoder_side_information_as_ungated"] == (not gated) and actual["new_fit"] is False,
            "SIX_AUTHENTIC_F_ONLY_ROUTES_AND_ADDITIONAL_C_PAST_GATE_METADATA")


def independent_physical_gate_precheck(entry, physical, dev, refs, physical_record, reference_record):
    prefix = prior.causal_physical_preprocess(physical["history_physical"][:8], physical["history_observed"][:8], dev["mean"], dev["std"])
    force, counts = gate_audit.force_current_from_past(prefix, physical["history_observed"][:8])
    original_force, original_counts = gate_audit.force_current_from_past(dev["history"][:8], dev["history_observed"][:8])
    flat = np.ones(force.shape, bool)
    for window in range(8):
        for channel in range(force.shape[1]):
            flat[window, channel] = all(prefix[window, position, channel] == prefix[window, -1, channel] for position in range(512))
    controls = {"keep_forecast": ~force, "exact_flat": flat, "true_observed_counts": counts}
    require(entry["status"] == "PASS" and entry["physical_source"] == physical_record and entry["CPU_gate_source"] == reference_record
            and entry["origins"] == dev["origins"][:8].tolist()
            and entry["all_three_gate_arrays_byte_exact"] is True and entry["tolerance_parity_used_to_accept_gate_decision"] is False,
            "EXACT_PHYSICAL_PRE_TIMER_GATE_IDENTITY_SCOPE")
    exact(force, original_force, "physical_vs_original_gate_eligible_decision_no_tolerance")
    exact(counts, original_counts, "physical_vs_original_true_past_observed_counts")
    exact(physical["history_observed"][:8], dev["history_observed"][:8], "original_physical_first_B_past_mask")
    require(entry["history_observed"] == array_record(physical["history_observed"][:8])
            and entry["preprocessed_history"] == array_record(prefix), "INDEPENDENT_PRE_TIMER_PHYSICAL_PREPROCESS_BYTE_RECORD")
    for key, value in controls.items():
        exact(value, refs[key][:8], "physical_vs_accepted_CPU_gate_array:" + key)
        require(entry["actual_physical_gate_arrays"][key] == array_record(value)
                and entry["CPU_reference_gate_arrays"][key] == array_record(refs[key][:8]), "EXACT_PHYSICAL_GATE_ARRAY_MANIFEST:" + key)
    prior.compare_parity_record(entry["normalized_history_parity"], prefix, dev["history"][:8], "physical_gate_precheck_history")
    prior.compare_parity_record(entry["normalized_current_C_parity"], prefix[:, -1], refs["raw_last"][:8], "physical_gate_precheck_current_C")
    return {"physical_normalized_prefix": prefix, "force_current": force,
            "metadata": {"first_B1_B8_gate_arrays_byte_exact": True, "gate_decision_tolerance_used": False,
                         "physical_gate_eligible_window_channel_pairs_first8": int(force.sum())}}


def compare_timed_conditional_gate(cost, physical_prefix, force, check):
    table = {(row["route"], row["batch"]): row for row in cost["rows"]}
    errors = []
    for batch in (1, 8):
        for base in ("RAW_BOLT", "PCA_PINV_BOLT", "FIT_TARGET_PINV_BOLT"):
            original = prior.verified_arrays(table[(base, batch)]["timed_outputs"], check)["normalized"]
            gated = prior.verified_arrays(table[(base + "_GATED", batch)]["timed_outputs"], check)["normalized"]
            fixed = np.broadcast_to(force[:batch, None], gated.shape[1:])
            anchor = np.broadcast_to(physical_prefix[:batch, -1:, :], gated.shape[1:])
            maxima = []
            for repeat in range(6):
                expected = gate_audit.independently_gate_prediction(original[repeat], physical_prefix[:batch, -1], force[:batch])
                maxima.append(prior.close(gated[repeat], expected, "same_block_conditional_output_from_separate_F_calls"))
                exact(gated[repeat][fixed], anchor[fixed], "actual_timed_forced_current_C_bytes")
            errors.append({"base_method": base, "batch": batch, "per_repeat_predicted_gate_max_abs": maxima,
                           "forced_current_coordinates_byte_exact": True,
                           "unmodified_coordinates_between_separate_F_calls_numeric_parity_only": True})
    return errors


def audit_cost_dataset(cost, physical, dev, predictions, metadata, check, *, sealed_unix):
    require(cost["status"] == "COMPLETE" and cost["routes"] == list(ROUTES)
            and cost["expected_cells_this_dataset"] == 12 and cost["expected_cells_two_datasets"] == 24
            and len(cost["rows"]) == 12 and cost["warmup_each_route"] == 2 and cost["repeats"] == 6,
            "COMPLETE_FIVE_BOLT_TWO_BATCH_COST_MATRIX")
    require(cost["new_training_updates"] == 0 and cost["encoder_decoder_refit_in_cost"] is False
            and cost["protected_TEST_accessed"] is cost["independent_holdout"] is False, "COST_NO_REFIT_OR_HOLDOUT")
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
    require(len(table) == 12 and set(table) == {(route, batch) for route in ROUTES for batch in (1, 8)}, "NO_MISSING_OR_DUPLICATE_COST_CELL")
    lineage = cost["source_lineage"]
    require(lineage["whole_cost_source_sha256"] == prior.file_sha(HERE / "whole_past_constant_gate_cost_v1.py")
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
                and sorted(row["positions"]) == list(range(6)), "FIVE_POSITION_BALANCE_SIXTH_EXTRA")
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
        require(row["memory_counter_reset_inside_module"] is row["stage_decomposition"] is row["optimized_native_implementation_claim"] is False,
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
    return {"rows": details, "cells_checked": 12, "recorded_timer_samples_checked": 72,
            "clock_memory_or_F_forward_independently_remeasured": False,
            "all_six_rounds_balanced": True, "same_first_B_origins_repeated_not_independent_draws": True,
            "per_method_exclusive_memory_claimed": False}


def audit_result(cost_path, meter):
    check = meter.check
    path = Path(cost_path).resolve()
    record = {"path": str(path), "sha256": prior.file_sha(path)}
    cost = prior.verified_json(record, check)
    if cost.get("status") != "COMPLETE":
        return {"status": "NOT_AUDITED_INCOMPLETE_PRODUCER", "cost": record, "scientific_assets_opened": False}
    require(cost["routes"] == list(ROUTES) and cost["expected_cost_cells"] == 24 and cost["recorded_timer_samples"] == 144
            and cost["protected_TEST_accessed"] is cost["independent_holdout"] is cost["new_backbone_training"] is False
            and cost["new_training_updates"] == cost["projection_fits"] == 0
            and cost["new_native_head_forward"] is cost["native_deployment_cost_measured"] is cost["method_selection_used"] is False
            and cost["past_C_condition_metadata_added"] is cost["gated_RAW_baseline_included"] is True
            and cost["same_decoder_side_information_as_ungated"] is cost["strict_K_only_compression"] is cost["static_original_fixed_D_floor_is_gated_lower_bound"] is False
            and cost["hard_elapsed_or_cumulative_resource_caps"] is None, "COMPLETE_REGISTERED_SIX_F_ONLY_WHOLE_COST_SCOPE")
    gates = {key: prior.verified_json(cost[key], check) for key in ("producer", "fit_target_audit", "gate_diagnostic", "gate_audit")}
    producer, fit_audit, diagnostic, gate_pass = (gates[key] for key in ("producer", "fit_target_audit", "gate_diagnostic", "gate_audit"))
    require(producer["status"] == diagnostic["status"] == "COMPLETE" and fit_audit["status"] == gate_pass["status"] == "PASS", "COMPLETE_FIT_CPU_PROBE_AND_EXACT_PASS_GATES")
    for value, key, expected in ((fit_audit, "producer", cost["producer"]), (diagnostic, "producer", cost["producer"]),
                               (diagnostic, "independent_FIT_target_audit", cost["fit_target_audit"]),
                               (gate_pass, "diagnostic", cost["gate_diagnostic"]), (gate_pass, "producer", cost["producer"]),
                               (gate_pass, "fit_target_audit", cost["fit_target_audit"])):
        assert_binding(value[key], expected, "accepted_cost_gate." + key)
    require(all(fit_audit["totals"].get(key) == expected for key, expected in {
        "projection_states_checked": 4, "native_fits_checked": 8, "checkpoint_states_checked": 40, "score_field_checks": 1464,
        "saved_prediction_arrays_checked": 88, "cost_cells_checked": 20, "recorded_timer_samples_checked": 120,
        "target_projection_fits_checked": 2, "reused_PCA_controls_checked": 2, "output_subspace_diagnostic_cells_checked": 12}.items())
        and all(gate_pass["totals"].get(key) == expected for key, expected in {"dataset_split_cells_checked": 6,
            "method_pairs_checked": 42, "ungated_prediction_arrays_checked": 42, "gated_prediction_arrays_checked": 42, "score_field_checks": 2016}.items()), "FULL_ACCEPTED_FIT_AND_CPU_GATE_AUDIT_COUNTS")
    require(diagnostic["new_training"] is diagnostic["new_encoder_forecaster_or_head_forward"] is diagnostic["future_labels_used_for_gate"] is False
            and diagnostic["method_selection_used"] is diagnostic["static_original_fixed_D_floor_reused_as_gated_lower_bound"] is False,
            "ACCEPTED_PAST_ONLY_CPU_OUTPUT_PROBE_CONTRACT")
    helper_record = {"path": str(HERE / "whole_cost_v1.py"), "sha256": prior.file_sha(HERE / "whole_cost_v1.py")}
    assert_binding(cost["timing_helper_copy"]["original"], helper_record, "original_accepted_whole_timer")
    source_checks = source_contract((HERE / "whole_cost_v1.py").read_text(encoding="utf-8-sig"),
        (HERE / "whole_past_constant_gate_cost_v1.py").read_text(encoding="utf-8-sig"),
        (HERE / "run_past_constant_gate_cost_v1.py").read_text(encoding="utf-8-sig"))
    guarded = [record] + [cost[key] for key in gates]
    guarded += cost["source_records"] + cost["source_snapshots"] + producer["protected_prior_records"] + diagnostic["protected_prior_records"]
    guarded += diagnostic["source_records"] + [diagnostic["scope"], producer["selection_lock"]]
    for entry in guarded:
        check()
        require(prior.file_sha(entry["path"]) == entry["sha256"], "UNCHANGED_SIX_ROUTE_COST_SOURCE_ASSET_OR_SELECTION_BINDING")
        if "snapshot_path" in entry:
            require(prior.file_sha(entry["snapshot_path"]) == entry["sha256"], "ACTUAL_COST_EXECUTED_SOURCE_SNAPSHOT_BYTES")
    for key, expected in (("authorization", HERE / "AUTHORIZATION_SCOPE_v1.json"), ("continuation_manifest", HERE / "CONTINUATION_MANIFEST_v1.json")):
        assert_binding(cost[key], {"path": str(expected), "sha256": prior.file_sha(expected)}, key)
    require(cost["F_before"] == cost["F_after"] == producer["F_before"] == producer["F_after"]
            and set(cost["F_before"]) == {"old_algorithm", "bi_algorithm"}, "EXACT_RECORDED_FROZEN_F_DUAL_SIGNATURES")
    for key in ("model_binary", "model_config"):
        assert_binding(cost[key + "_before"], cost[key + "_after"], key + "_actual_before_after")
        assert_binding(cost[key + "_before"], producer[key + "_before"], key + "_accepted_FIT_source")
    behavior = cost["runtime_behavior_contract"]
    for key in ("evaluation", "all_backbone_parameters_frozen", "context_length", "prediction_length", "input_patch_size", "input_patch_stride",
                "median_index", "returned_horizon", "quantiles", "pipeline_predict_not_used", "normalizer_before", "normalizer_after", "normalizer_fixture_forward_parity"):
        require(behavior[key] == producer["runtime_behavior_contract"][key], "SAME_FROZEN_BOLT_RUNTIME_CONFIG:" + key)
    resource = prior.verified_json(cost["resource_terminal"], check)
    cpu_resource = prior.verified_json(diagnostic["resource_terminal"], check)
    lock = prior.verified_json(producer["selection_lock"], check)
    require(resource["status"] == cpu_resource["status"] == "COMPLETE" and resource["elapsed_or_cumulative_budget_caps"] is None
            and lock["status"] == "SEALED_BEFORE_NEW_DEV_AND_ORACLE", "COMPLETED_FROZEN_SELECTION_AND_CPU_BEFORE_WHOLE_COST")
    earliest_timer = max(lock["sealed_unix"], cpu_resource["ended_unix"], resource["started_unix"])
    require(set(cost["datasets"]) == set(cost["cost"]) == set(DATASETS), "EXACT_TWO_WHOLE_GATE_COST_DATASETS")
    report = {"status": "PASS", "cost": record, **{key: cost[key] for key in gates}, "datasets": {}, "cost_details": {},
              "source_contract_checks": source_checks, "totals": {"cost_cells_checked": 0, "recorded_timer_samples_checked": 0,
                  "physical_gate_prechecks_checked": 0, "paired_gate_timed_output_arrays_checked": 0,
                  "original_selected_forecast_hashes_checked": 0},
              "protected_TEST_accessed": False, "new_training_or_GPU_execution": False,
              "actual_F_or_timers_independently_remeasured": False, "exclusive_method_RAM_or_VRAM_claimed": False,
              "native_deployment_cost_measured": False, "all_six_routes_use_same_frozen_F": True,
              "past_C_condition_metadata_added": True, "same_decoder_side_information_as_ungated": False,
              "representation_learning_improvement_claimed": False, "static_D_lower_bound_for_gated_output_claimed": False}
    for dataset, (channels, k, dev, count, parent_tag) in DATASETS.items():
        original, row = producer["datasets"][dataset], cost["datasets"][dataset]
        case = diagnostic["datasets"][dataset]["splits"][dev]
        require(row["status"] == "COMPLETE" and row["K"] == k and row["development_split"] == dev
                and row["inputs"] == {key: original["inputs"][key] for key in ("fit", dev)}
                and row["CPU_gate_reference"] == case["artifact"] and row["physical_bridge"] == original["physical_bridge"], "EXACT_ORIGINAL_COST_SPLIT_GATE_PHYSICAL_REFERENCES")
        fit, values = (prior.verified_arrays(row["inputs"][key], check) for key in ("fit", dev))
        prior.check_raw_split(values, channels=channels, windows=count)
        for key in ("mean", "std"):
            exact(fit[key], values[key], "same_original_FIT_only_cost_" + key)
        parent_path = BASE / "runs" / parent_tag / "RESULT.json"
        parent = prior.verified_json({"path": str(parent_path), "sha256": prior.file_sha(parent_path)}, check)
        bridge = prior.verified_json(parent["bridge_provenance"]["input_bridge"], check)
        candidates = [entry for entry in bridge["artifacts"] if entry["dataset"] == dataset and entry["split"] == dev]
        require(len(candidates) == 1, "ONE_CANONICAL_PHYSICAL_COST_BRIDGE")
        canonical = prior.bind_canonical_npz(row["physical_bridge"], candidates[0])
        physical, refs = prior.verified_arrays(canonical, check), prior.verified_arrays(case["artifact"], check)
        for key in ("origins", "raw_last"):
            exact(refs[key], values[key], "CPU_probe_original_cost_" + key)
        exact(refs["FIT_mean"], fit["mean"], "CPU_probe_original_cost_FIT_mean")
        exact(refs["FIT_std"], fit["std"], "CPU_probe_original_cost_FIT_std")
        original_force, original_counts = gate_audit.force_current_from_past(values["history"], values["history_observed"])
        exact(refs["keep_forecast"], ~original_force, "accepted_CPU_gate_keep_forecast")
        exact(refs["true_observed_counts"], original_counts, "accepted_CPU_gate_observation_counts")
        require(row["gate_keep_metadata"] == array_record(refs["keep_forecast"]), "FULL_CPU_GATE_KEEP_FLAGS_CONTENT_BINDING")
        precheck = independent_physical_gate_precheck(row["physical_gate_precheck"], physical, values, refs,
                                                    row["physical_bridge"], case["artifact"])
        predictions, metadata = {}, {}
        for route in ROUTES:
            gated = route.endswith("_GATED")
            base = route[:-6] if gated else route
            key = base + ("_gated_prediction" if gated else "_ungated_prediction")
            predictions[route] = refs[key]
            metadata[route] = {"base_method": base, "gated": gated, "shared_resident_frozen_F": True,
                "native_head_loaded_or_timed": False, "saved_reference": case["artifact"], "reference_prediction_array": key,
                "history_observed_explicitly_passed": True, "past_C_metadata_added": gated,
                "same_decoder_side_information_as_ungated": not gated, "gate_rule": cost["gate_contract"] if gated else None, "new_fit": False}
            if base != "RAW_BOLT":
                metadata[route]["source_state"] = original["projections"][base[:-5]]["state_artifact"]
            validate_route_metadata(route, row["selected_metadata"][route], metadata[route])
        exact(refs["RAW_BOLT_ungated_prediction"], values["raw_forecast"], "accepted_original_RAW_whole_cost_reference")
        for projection in ("PCA_PINV", "FIT_TARGET_PINV"):
            cached = prior.verified_arrays(original["projections"][projection]["bolt"]["artifacts"][dev], check)
            exact(refs[projection + "_BOLT_ungated_prediction"], cached["prediction"], "accepted_frozen_projection_whole_cost_reference")
        details = audit_cost_dataset(cost["cost"][dataset], physical, values, predictions, metadata, check, sealed_unix=earliest_timer)
        details["paired_timed_gate_output_checks"] = compare_timed_conditional_gate(cost["cost"][dataset],
            precheck["physical_normalized_prefix"], precheck["force_current"], check)
        literal_path = path.parent / ("cost_" + dataset) / "RESULT.json"
        literal = prior.verified_json({"path": str(literal_path), "sha256": prior.file_sha(literal_path)}, check)
        require(literal == cost["cost"][dataset], "LITERAL_DATASET_TWELVE_CELL_RESULT_BINDING")
        report["datasets"][dataset] = {"channels": channels, "K": k, "development_split": dev,
                                     "canonical_physical_bridge": canonical, "physical_gate_precheck": precheck["metadata"]}
        report["cost_details"][dataset] = details
        report["totals"]["cost_cells_checked"] += 12
        report["totals"]["recorded_timer_samples_checked"] += 72
        report["totals"]["physical_gate_prechecks_checked"] += 1
        report["totals"]["paired_gate_timed_output_arrays_checked"] += 36
    for dataset in producer["datasets"].values():
        for condition in dataset["projections"].values():
            for entry in condition["bolt"]["artifacts"].values():
                guarded.append(entry)
                report["totals"]["original_selected_forecast_hashes_checked"] += 1
            for native in condition["native"].values():
                for entry in native["artifacts"].values():
                    guarded.append(entry)
                    report["totals"]["original_selected_forecast_hashes_checked"] += 1
    require(cost["source_cache_selection_SHA_unchanged"] is cost["original_selected_forecasts_and_locks_unchanged"] is True,
            "RECORDED_ORIGINAL_SOURCE_SELECTED_CACHE_AND_LOCK_PRESERVATION")
    for entry in guarded:
        check()
        require(prior.file_sha(entry["path"]) == entry["sha256"], "ORIGINAL_COST_CACHE_FROZEN_STATE_SOURCE_OR_LOCK_CHANGED_DURING_AUDIT")
    require(report["totals"]["cost_cells_checked"] == 24 and report["totals"]["recorded_timer_samples_checked"] == 144,
            "FULL_REGISTERED_TWENTY_FOUR_COST_ONE_FORTY_FOUR_TIMER_COVERAGE")
    return report


def execute(cost_path, output_path):
    from continuation_runtime_v1 import Meter, write_new
    output = Path(output_path).resolve()
    require(not output.exists(), "FRESH_UNIQUE_PAST_CONSTANT_COST_AUDIT_OUTPUT")
    output.parent.mkdir(parents=True, exist_ok=True)
    meter = Meter("audit_past_constant_cost_" + output.stem, gpu=False)
    source = {"path": str(Path(__file__).resolve()), "sha256": prior.file_sha(__file__)}
    failure = None
    try:
        meter.protected_check()
        initialize_numpy()
        sys.path.insert(0, str(prior.ROOT / "research/smallk_failure_followup_20261014"))
        from prepare_inputs import install_protected_test_guard
        install_protected_test_guard()
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2):
            report = audit_result(cost_path, meter)
        meter.protected_check()
        require(source["sha256"] == prior.file_sha(__file__), "INDEPENDENT_WHOLE_GATE_COST_AUDIT_SOURCE_UNCHANGED")
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

