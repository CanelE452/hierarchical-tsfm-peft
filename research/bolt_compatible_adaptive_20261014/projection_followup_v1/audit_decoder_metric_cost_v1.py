"""Independent saved-output and recorded whole-cost audit; no F/timer replay."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import sys

import audit_projection_results_v1 as prior
import audit_decoder_metric_probe_v1 as decoder_audit

np = sys.modules.get("numpy")
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROUTES = ("RAW_BOLT", "PCA_PAST_LS_BOLT", "FORECA_PAST_LS_BOLT",
          "PCA_ENCODER_PINV_BOLT", "FORECA_ENCODER_PINV_BOLT")
DATASETS = prior.DATASETS
require = prior.require
exact = decoder_audit.byte_exact


def initialize_numpy():
    global np
    decoder_audit.initialize_numpy()
    np = sys.modules["numpy"]


def array_record(value):
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}


def assert_binding(record, expected, label):
    require(Path(record["path"]).resolve() == Path(expected["path"]).resolve()
            and record["sha256"] == expected["sha256"] == prior.file_sha(expected["path"]),
            "EXACT_BOUND_RECORD:" + label)


def timing_copy_contract(original_text, current_text):
    original, current = ast.parse(original_text), ast.parse(current_text)
    for tree in (original, current):
        assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "ROUTES" for target in node.targets)]
        require(len(assignments) == 1, "ONE_LITERAL_TIMING_ROUTE_DECLARATION")
        if tree is current:
            require(ast.literal_eval(assignments[0].value) == ROUTES, "FIVE_AUTHENTIC_BOLT_TIMING_ROUTES")
        assignments[0].value = ast.Constant(value="SAME_ROUTE_SLOT")
    require(ast.dump(original, include_attributes=False) == ast.dump(current, include_attributes=False),
            "TIMING_HELPER_ONLY_ROUTE_NAMES_CHANGED")


def independent_schedule():
    generator = np.random.default_rng(20261015)
    initial = generator.permutation(5)
    schedule = [[ROUTES[index] for index in np.roll(initial, -repeat)] for repeat in range(5)]
    schedule.append([ROUTES[index] for index in generator.permutation(5)])
    return schedule


def recorded_timer(sample, row, repeat, sealed_unix):
    require(sample["perf_end_ns"] > sample["perf_start_ns"]
            and sample["unix_end_ns"] >= sample["unix_start_ns"] > sealed_unix * 1e9,
            "POSITIVE_RECORDED_TIMER_AFTER_ORIGINAL_SELECTION_LOCK")
    prior.close(sample["seconds"], (sample["perf_end_ns"] - sample["perf_start_ns"]) / 1e9,
                "recorded_perf_counter_arithmetic", atol=1e-12, rtol=1e-12)
    require(sample["seconds"] == row["seconds"][repeat]
            and sample["parity"] == row["parity"][repeat]
            and sample["memory"] == row["memory"][repeat], "SAMPLE_ROW_LITERAL_VALUES_BINDING")
    memory = sample["memory"]
    require(set(memory) == {"resident_rss_bytes", "peak_allocated_vram_bytes", "peak_reserved_vram_bytes"}
            and all(type(value) is int and value >= 0 for value in memory.values())
            and memory["peak_reserved_vram_bytes"] >= memory["peak_allocated_vram_bytes"],
            "NONNEGATIVE_SHARED_RESIDENT_MEMORY_COUNTERS")


def validate_route_metadata(route, actual, expected):
    require(actual == expected, "EXACT_COST_SAVED_SCIENTIFIC_ROUTE_METADATA:" + route)
    require(actual["shared_resident_model"] is True, "SHARED_RESIDENT_FROZEN_BOLT:" + route)
    if route == "RAW_BOLT":
        require(actual["source"] == "original frozen Bolt" and actual["prediction_array"] == "raw_forecast",
                "AUTHENTIC_RAW_BOLT_SAVED_PREDICTION_SOURCE")
        return
    projection = "PCA" if route.startswith("PCA_") else "FORECA"
    variant = "ENCODER_PINV" if "ENCODER_PINV" in route else "ORIGINAL_PAST_LS"
    require(actual["projection_route"] == projection and actual["decoder_variant"] == variant
            and actual["reference_prediction_array"] == variant + "_BOLT_prediction"
            and actual["deployed_decoder_array"] == variant + "_decoder_deployed_FP32"
            and actual["E_F_frozen"] is actual["D_frozen"] is True
            and actual["encoder_decoder_refit_in_cost"] is False, "FROZEN_AUTHENTIC_DECODER_BOLT_CALLBACK_METADATA")


def audit_cost_dataset(cost, physical, dev, predictions, metadata, check, *, sealed_unix):
    require(cost["status"] == "COMPLETE" and cost["routes"] == list(ROUTES)
            and cost["expected_cells_this_dataset"] == 10 and cost["expected_cells_two_datasets"] == 20
            and len(cost["rows"]) == 10 and cost["warmup_each_route"] == 2 and cost["repeats"] == 6,
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
    require(len(table) == 10 and set(table) == {(route, batch) for route in ROUTES for batch in (1, 8)}, "NO_MISSING_OR_DUPLICATE_COST_CELL")
    lineage = cost["source_lineage"]
    require(lineage["whole_cost_source_sha256"] == prior.file_sha(HERE / "whole_decoder_metric_cost_v1.py")
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
    return {"rows": details, "cells_checked": 10, "recorded_timer_samples_checked": 60,
            "clock_memory_or_F_forward_independently_remeasured": False,
            "first_five_rounds_balanced_sixth_extra": True, "same_first_B_origins_repeated_not_independent_draws": True,
            "per_method_exclusive_memory_claimed": False}


def audit_result(cost_path, meter):
    check = meter.check
    path = Path(cost_path).resolve()
    record = {"path": str(path), "sha256": prior.file_sha(path)}
    cost = prior.verified_json(record, check)
    if cost.get("status") != "COMPLETE":
        return {"status": "NOT_AUDITED_INCOMPLETE_PRODUCER", "cost": record, "scientific_assets_opened": False}
    require(cost["scientific_execution"] is True and cost["protected_TEST_accessed"] is cost["independent_holdout"] is False
            and cost["new_backbone_training"] is False and cost["new_training_updates"] == cost["projection_fits"] == cost["alternative_decoder_refits"] == 0
            and cost["method_checkpoint_branch_selection_used"] is False and cost["hard_elapsed_or_cumulative_resource_caps"] is None
            and cost["routes"] == list(ROUTES) and cost["expected_cost_cells"] == 20 and cost["recorded_timer_samples"] == 120,
            "COMPLETE_REGISTERED_NEW_D_WHOLE_COST_SCOPE")
    gates = {key: prior.verified_json(cost[key], check) for key in ("producer", "projection_audit", "decoder_diagnostic", "decoder_audit")}
    producer, projection_pass, diagnostic, decoder_pass = (gates[key] for key in ("producer", "projection_audit", "decoder_diagnostic", "decoder_audit"))
    require(producer["status"] == diagnostic["status"] == "COMPLETE" and projection_pass["status"] == decoder_pass["status"] == "PASS",
            "EXACT_PRIOR_PROJECTION_AND_DECODER_PASS_GATES")
    for key, expected in (("producer", cost["producer"]), ("projection_audit", cost["projection_audit"]), ("diagnostic", cost["decoder_diagnostic"])):
        assert_binding(decoder_pass[key], expected, "decoder_audit." + key)
    assert_binding(projection_pass["producer"], cost["producer"], "projection_audit.producer")
    for key in ("producer", "projection_audit"):
        assert_binding(diagnostic[key], cost[key], "decoder_diagnostic." + key)
    for key in ("span_diagnostic", "span_audit"):
        assert_binding(diagnostic[key], decoder_pass[key], "diagnostic_audit_span." + key)
    require(all(projection_pass["totals"].get(key) == expected for key, expected in {"projection_states_checked": 4, "native_fits_checked": 8,
                "checkpoint_states_checked": 40, "cost_cells_checked": 20, "recorded_timer_samples_checked": 120}.items())
            and all(decoder_pass["totals"].get(key) == expected for key, expected in {"base_cells_checked": 12, "decoder_cells_checked": 24,
                "cached_forecast_arrays_checked": 96, "original_predictions_byte_exact_checked": 48}.items()), "COMPLETE_PRIOR_REGISTERED_AUDIT_COUNTS")
    require(diagnostic["new_training"] is diagnostic["future_labels_used_to_fit_alternative_D"] is diagnostic["method_checkpoint_branch_selection_used"] is False,
            "LABEL_FREE_NEW_D_NO_CHECKPOINT_OR_BRANCH_SELECTION")
    timing_copy_contract((HERE / "whole_cost_v1.py").read_text(encoding="utf-8-sig"),
                         (HERE / "whole_decoder_metric_cost_v1.py").read_text(encoding="utf-8-sig"))
    assert_binding(cost["timing_helper_copy"]["original"], {"path": str(HERE / "whole_cost_v1.py"), "sha256": prior.file_sha(HERE / "whole_cost_v1.py")}, "original_timing_helper")
    source_checks = 0
    guarded = [cost[key] for key in ("producer", "projection_audit", "decoder_diagnostic", "decoder_audit")]
    for entry in cost["source_records"] + cost["source_snapshots"] + producer["source_snapshots"]:
        require(entry["sha256"] == prior.file_sha(entry["path"]), "EXACT_EXECUTED_COST_SOURCE_OR_LOADED_ASSET")
        guarded.append(entry)
        source_checks += 1
        if "snapshot_path" in entry:
            require(prior.file_sha(entry["snapshot_path"]) == entry["sha256"], "COST_EXECUTED_SOURCE_SNAPSHOT_BYTES")
            source_checks += 1
    for key, expected in (("authorization", HERE / "AUTHORIZATION_SCOPE_v1.json"), ("continuation_manifest", HERE / "CONTINUATION_MANIFEST_v1.json")):
        assert_binding(cost[key], {"path": str(expected), "sha256": prior.file_sha(expected)}, key)
    require(cost["F_before"] == cost["F_after"] == producer["F_before"] == producer["F_after"]
            and set(cost["F_before"]) == {"old_algorithm", "bi_algorithm"}, "RECORDED_DUAL_FROZEN_F_STATE_HASH_IDENTITY")
    for key in ("model_binary", "model_config"):
        assert_binding(cost[key + "_before"], cost[key + "_after"], key + "_before_after")
        assert_binding(cost[key + "_before"], producer[key + "_before"], key + "_original")
    behavior = cost["runtime_behavior_contract"]
    for key in ("evaluation", "all_backbone_parameters_frozen", "context_length", "prediction_length", "input_patch_size", "input_patch_stride",
                "median_index", "returned_horizon", "quantiles", "pipeline_predict_not_used", "normalizer_before", "normalizer_after", "normalizer_fixture_forward_parity"):
        require(behavior[key] == producer["runtime_behavior_contract"][key], "EXACT_ORIGINAL_BOLT_BEHAVIOR:" + key)
    seal = prior.verified_json(producer["selection_lock"], check)
    require(seal["status"] == "SEALED_BEFORE_NEW_DEV_AND_ORACLE", "ORIGINAL_SEALED_CHECKPOINT_SELECTION_LOCK")
    guarded.append(producer["selection_lock"])
    resource = prior.verified_json(cost["resource_terminal"], check)
    require(resource["status"] == "COMPLETE" and resource["elapsed_or_cumulative_budget_caps"] is None, "COMPLETE_ACTUAL_COST_RESOURCE_RECEIPT")
    require(set(cost["datasets"]) == set(cost["cost"]) == set(DATASETS), "EXACT_TWO_COST_DATASETS")
    report = {"status": "PASS", "cost": record, **{key: cost[key] for key in gates}, "datasets": {}, "cost_details": {},
              "totals": {"cost_cells_checked": 0, "recorded_timer_samples_checked": 0, "selected_forecast_file_hashes_checked": 0,
                         "canonical_physical_bridges_checked": 0}, "source_file_checks": source_checks,
              "protected_TEST_accessed": False, "new_training_or_GPU_execution": False,
              "F_forward_or_timers_independently_replayed": False, "independent_generalization_claimed": False,
              "exclusive_per_method_RAM_or_VRAM_claimed": False, "all_five_callbacks_use_same_frozen_Bolt": True}
    for dataset, (channels, k, development, count, parent_tag) in DATASETS.items():
        original = producer["datasets"][dataset]
        row = cost["datasets"][dataset]
        require(row["status"] == "COMPLETE" and row["K"] == k and row["development_split"] == development
                and row["inputs"] == {key: original["inputs"][key] for key in ("fit", development)}, "SAME_ORIGINAL_COST_INPUTS_AND_SMALL_K")
        fit, dev = (prior.verified_arrays(row["inputs"][key], check) for key in ("fit", development))
        prior.check_raw_split(dev, channels=channels, windows=count)
        for key in ("mean", "std"):
            exact(dev[key], fit[key], "same_FIT_only_cost_scaler:" + key)
        parent_path = BASE / "runs" / parent_tag / "RESULT.json"
        parent = prior.verified_json({"path": str(parent_path), "sha256": prior.file_sha(parent_path)}, check)
        bridge = prior.verified_json(parent["bridge_provenance"]["input_bridge"], check)
        candidates = [entry for entry in bridge["artifacts"] if entry["dataset"] == dataset and entry["split"] == development]
        require(len(candidates) == 1 and row["physical_bridge"] == original["physical_bridge"], "ONE_ORIGINAL_DEVELOPMENT_PHYSICAL_BRIDGE")
        canonical = prior.bind_canonical_npz(row["physical_bridge"], candidates[0])
        physical = prior.verified_arrays(canonical, check)
        predictions = {"RAW_BOLT": dev["raw_forecast"]}
        metadata = {"RAW_BOLT": {"source": "original frozen Bolt", "shared_resident_model": True,
                    "source_input": original["inputs"][development], "prediction_array": "raw_forecast"}}
        for projection in ("PCA", "FORECA"):
            state_record = original["projections"][projection]["state_artifact"]
            probe_record = diagnostic["datasets"][dataset]["projections"][projection]["splits"][development]["artifact"]
            state, refs = prior.verified_arrays(state_record, check), prior.verified_arrays(probe_record, check)
            for key in ("origins", "raw_last"):
                exact(refs[key], dev[key], "newD_cost_original_" + key)
            exact(refs["encoder_deployed_FP32"], state["encoder"].astype(np.float32), "newD_cost_fixed_E32")
            exact(refs["ORIGINAL_PAST_LS_decoder_deployed_FP32"], state["decoder"].astype(np.float32), "newD_cost_original_D32")
            exact(refs["ENCODER_PINV_decoder_deployed_FP32"], decoder_audit.independent_encoder_inverse(refs["encoder_deployed_FP32"])["D_inverse32"], "newD_cost_independent_inverse_D32")
            for variant, short in (("ORIGINAL_PAST_LS", "PAST_LS"), ("ENCODER_PINV", "ENCODER_PINV")):
                route = projection + "_" + short + "_BOLT"
                dkey, pkey = variant + "_decoder_deployed_FP32", variant + "_BOLT_prediction"
                predictions[route] = refs[pkey]
                metadata[route] = {"projection_route": projection, "decoder_variant": variant, "E_F_frozen": True, "D_frozen": True,
                    "encoder_decoder_refit_in_cost": False, "shared_resident_model": True, "source_state": state_record,
                    "saved_reference": probe_record, "reference_prediction_array": pkey, "deployed_decoder_array": dkey,
                    "decoder_array_metadata": array_record(refs[dkey]), "encoder_array_metadata": array_record(refs["encoder_deployed_FP32"]),
                    "center_array_metadata": array_record(state["center"].astype(np.float32))}
                validate_route_metadata(route, row["decoders"][route], metadata[route])
        require(set(row["decoders"]) == set(ROUTES) - {"RAW_BOLT"}, "EXACT_FOUR_COST_DECODER_CALLBACKS")
        details = audit_cost_dataset(cost["cost"][dataset], physical, dev, predictions, metadata, check, sealed_unix=seal["sealed_unix"])
        literal_path = path.parent / ("cost_" + dataset) / "RESULT.json"
        literal = prior.verified_json({"path": str(literal_path), "sha256": prior.file_sha(literal_path)}, check)
        require(literal == cost["cost"][dataset], "LITERAL_DATASET_COST_RESULT_BINDING")
        report["cost_details"][dataset] = details
        report["datasets"][dataset] = {"original_physical_bridge": canonical, "channels": channels, "K": k, "development_split": development}
        report["totals"]["cost_cells_checked"] += details["cells_checked"]
        report["totals"]["recorded_timer_samples_checked"] += details["recorded_timer_samples_checked"]
        report["totals"]["canonical_physical_bridges_checked"] += 1
    for original in producer["datasets"].values():
        for condition in original["projections"].values():
            for entry in condition["bolt"]["artifacts"].values():
                guarded.append(entry)
                report["totals"]["selected_forecast_file_hashes_checked"] += 1
            for native in condition["native"].values():
                for entry in native["artifacts"].values():
                    guarded.append(entry)
                    report["totals"]["selected_forecast_file_hashes_checked"] += 1
    require(cost["source_before_after_SHA_pass"] is cost["original_selected_forecasts_and_locks_unchanged"] is True,
            "RECORDED_COST_SOURCE_SELECTION_CACHE_PRESERVATION")
    for entry in guarded:
        check()
        require(prior.file_sha(entry["path"]) == entry["sha256"], "ORIGINAL_SELECTED_CACHE_SOURCE_AND_LOCK_CURRENT_SHA_UNCHANGED")
    require(prior.file_sha(path) == record["sha256"] and report["totals"]["cost_cells_checked"] == 20
            and report["totals"]["recorded_timer_samples_checked"] == 120, "COMPLETE_UNCHANGED_REGISTERED_NEW_D_COST_AUDIT")
    return report


def execute(cost_path, output_path):
    from continuation_runtime_v1 import Meter, write_new
    output_path = Path(output_path).resolve()
    require(not output_path.exists(), "NEW_UNIQUE_INDEPENDENT_AUDIT_OUTPUT")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meter = Meter("audit_decoder_cost_" + output_path.stem, gpu=False)
    source = {"path": str(Path(__file__).resolve()), "sha256": prior.file_sha(__file__)}
    error = None
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
        require(source["sha256"] == prior.file_sha(__file__), "AUDITOR_SOURCE_UNCHANGED_DURING_EXECUTION")
    except Exception as failure:
        error = {"type": type(failure).__name__, "message": str(failure)}
        report = {"status": "FAILED_PRESERVED", "failure": error, "full_scope_PASS_claimed": False}
    finally:
        report["audit_source"] = source
        report["resource_terminal"] = meter.close(report["status"], error)
        write_new(output_path, report)
    print(json.dumps({"status": report["status"], "output": str(output_path), "totals": report.get("totals"), "failure": error}, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cost", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    raise SystemExit(execute(arguments.cost, arguments.output))
