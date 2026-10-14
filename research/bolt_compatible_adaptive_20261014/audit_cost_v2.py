"""Independent saved deployment outputs and recorded cost arithmetic audit.

The fresh guard is created before NumPy or scientific files are opened. This
source never imports the cost producer, reruns Bolt, refits a model, reruns a
timer, or treats recorded memory samples as an independent measurement.
"""
from pathlib import Path
import hashlib
import json
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
np = sys.modules.get("numpy")
ROUTES = ("PERSIST", "MEAN", "SELF33", "JOINT33", "SELF512", "RAW_BOLT")
NORM_ATOL, NORM_RTOL = 1e-5, 2e-6
PHYSICAL_ATOL, PHYSICAL_RTOL = 2e-3, 2e-5


class AuditFailure(ValueError):
    pass


def require(condition, label):
    if not condition:
        raise AuditFailure(label)


def initialize_numpy():
    global np
    import numpy as loaded_numpy
    np = loaded_numpy


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def exact(actual, expected, label):
    require(actual.dtype == expected.dtype and actual.shape == expected.shape
            and np.array_equal(actual, expected, equal_nan=True), "EXACT_ARRAY_MISMATCH:" + label)


def bind_canonical_npz(file_record, canonical, label):
    """Checked-file lineage records do not carry the original NPZ array map."""
    require(resolve(file_record["path"]).resolve() == resolve(canonical["path"]).resolve()
            and file_record["sha256"] == canonical["sha256"]
            and isinstance(canonical.get("arrays"), dict) and bool(canonical["arrays"]),
            "CANONICAL_NPZ_PATH_SHA_ARRAY_BINDING_MISMATCH:" + label)
    return canonical


def scalar_preprocess(physical, observed, mean, std):
    """Independent chronological loops, with a FIT-mean reset in every window."""
    require(physical.ndim == 3 and physical.dtype == np.float32 and observed.dtype == np.bool_
            and physical.shape == observed.shape and physical.shape[1] == 512,
            "ACTUAL_FP32_PHYSICAL_N512C_BOOL_MASK_REQUIRED")
    require(mean.dtype == np.float32 and std.dtype == np.float32 and mean.shape == (physical.shape[2],)
            and std.shape == mean.shape and np.isfinite(mean).all() and np.isfinite(std).all()
            and (std > 0).all() and np.isfinite(physical[observed]).all(), "FINITE_FIT_ONLY_RAW_STATISTICS_REQUIRED")
    filled = np.empty_like(physical)
    for window in range(len(physical)):
        for channel in range(physical.shape[2]):
            last = mean[channel]
            for time_index in range(512):
                if observed[window, time_index, channel]:
                    last = physical[window, time_index, channel]
                filled[window, time_index, channel] = last
    return ((filled - mean) / std).astype(np.float32)


def parity(actual, reference, *, atol=NORM_ATOL, rtol=NORM_RTOL):
    require(actual.shape == reference.shape and np.isfinite(actual).all()
            and np.isfinite(reference).all(), "PARITY_SHAPE_OR_FINITENESS_FAILURE")
    error = actual.astype(np.float64) - reference.astype(np.float64)
    return {"pass": bool(np.all(np.abs(error) <= atol + rtol * np.abs(reference))),
            "max_abs": float(np.abs(error).max()), "rmse": float(np.sqrt(np.square(error).mean())),
            "atol": atol, "rtol": rtol}


def compare_parity(recorded, actual, label):
    require(recorded["pass"] == actual["pass"] and actual["pass"], "RECORDED_PARITY_FAILURE:" + label)
    for name in ("atol", "rtol", "max_abs", "rmse"):
        require(abs(float(recorded[name]) - actual[name]) <= 1e-12 * max(1.0, abs(actual[name])),
                "PARITY_ARITHMETIC_MISMATCH:" + label + ":" + name)


def audit_recorded_times(row, raw_row):
    seconds = row["seconds"]
    raw_seconds = raw_row["seconds"]
    require(len(seconds) == len(raw_seconds) == 6 and all(value > 0 and np.isfinite(value) for value in seconds + raw_seconds),
            "SIX_FINITE_POSITIVE_RECORDED_TIMES_REQUIRED")
    median = float(statistics.median(seconds))
    ratios = [value / baseline for value, baseline in zip(seconds, raw_seconds)]
    expected = {"median_seconds": median, "median_seconds_per_window": median / row["batch"],
                "paired_RAW_median_ratio": float(statistics.median(ratios))}
    for name, value in expected.items():
        require(abs(row[name] - value) <= 1e-12 * max(1.0, abs(value)), "RECORDED_COST_ARITHMETIC_MISMATCH:" + name)
    require(row["paired_RAW_seconds"] == raw_seconds and np.array_equal(row["paired_RAW_ratios"], ratios),
            "PAIRED_RAW_BLOCK_RATIOS_MISMATCH")
    require(row["repeats"] == 6 and row["warmup_each_route"] == 2
            and sorted(row["positions"]) == list(range(6)), "SIX_POSITION_BALANCE_REQUIRED")
    return {**expected, "positions_checked": row["positions"], "ratios_checked": ratios,
            "clock_readings_independently_remeasured": False}


def audit_cost_result(path, limit):
    check = limit.shared.checkfast
    from audit_readout_v2 import verified_arrays, verified_json, fit_mean_innovation
    check()
    path = resolve(path)
    record = {"path": str(path), "sha256": file_sha(path)}
    result = verified_json(record, check)
    require(result["status"] == "COMPLETE" and len(result["rows"]) == result["planned_rows"] == 24,
            "COMPLETE_24_CELL_COST_PRODUCER_REQUIRED")
    require(result["actual_GPU_execution"] is True and result["new_training_updates"] == 0
            and result["coefficient_refit_in_cost"] is False and result["protected_TEST_accessed"] is False
            and result["independent_holdout"] is False, "COST_SCOPE_DECLARATION_MISMATCH")
    require(result["permission_gate"] == limit.authorization_record and result["run_manifest"] == limit.run_manifest_record,
            "SAME_APPROVED_NEW_CAMPAIGN_REQUIRED")
    scientific_audit = verified_json(result["independent_scientific_audit"], check)
    require(scientific_audit["status"] == "PASS" and scientific_audit["totals"]["pool_candidates_checked"] == 54,
            "INDEPENDENT_FIRST_54_POOL_AUDIT_REQUIRED")
    producers, producer_records = {}, []
    for source in result["producer_records"]:
        value = verified_json(source, check)
        dataset = value["dataset"]
        require(dataset in ("robin", "beijing_aotizhongxin") and dataset not in producers and value["status"] == "COMPLETE",
                "TWO_UNIQUE_COMPLETED_SCIENTIFIC_PRODUCERS_REQUIRED")
        bindings = [entry for entry in scientific_audit["producer_records"] if entry["dataset"] == dataset]
        require(len(bindings) == 1 and bindings[0]["sha256"] == source["sha256"], "SCIENTIFIC_AUDIT_EXACT_PRODUCER_SHA_REQUIRED")
        producers[dataset] = value
        producer_records.append({"dataset": dataset, "path": source["path"], "sha256": source["sha256"]})
    require(set(producers) == {"robin", "beijing_aotizhongxin"}, "TWO_PREDECLARED_DATASETS_REQUIRED")
    source_checks = 0
    for source in result["source_records"] + result["source_snapshots"]:
        check()
        require(file_sha(resolve(source["path"])) == source["sha256"], "COST_DECLARED_SOURCE_BYTES_CHANGED")
        source_checks += 1
        if "snapshot_path" in source:
            require(file_sha(resolve(source["snapshot_path"])) == source["sha256"], "COST_SOURCE_SNAPSHOT_CHANGED")
            source_checks += 1
    require(result["F_before"] == result["F_after"] and result["source_before_after_SHA_pass"] is True
            and result["model_binary_before_after_SHA_pass"] is True, "FROZEN_MODEL_BEFORE_AFTER_BINDING_MISMATCH")
    for name in ("model_binary", "model_config"):
        before, after = result[name + "_before"], result[name + "_after"]
        require(before["sha256"] == after["sha256"] == file_sha(resolve(after["path"])), "MODEL_BINARY_CONFIG_BYTES_CHANGED:" + name)
    behavior = result["runtime_behavior_contract"]
    require(behavior["evaluation"] is True and behavior["all_backbone_parameters_frozen"] is True
            and behavior["context_length"] == 2048 and behavior["prediction_length"] == 64
            and behavior["input_patch_size"] == behavior["input_patch_stride"] == 16
            and behavior["median_index"] == 4 and behavior["returned_horizon"] == 48
            and behavior["quantiles"] == [.1, .2, .3, .4, .5, .6, .7, .8, .9]
            and behavior["pipeline_predict_not_used"] is True, "RECORDED_BOLT_WRAPPER_CONTRACT_MISMATCH")
    require(all(entry["exact_equal_including_nan"] is True for entry in behavior["normalizer_fixture_forward_parity"].values())
            and set(behavior["normalizer_fixture_forward_parity"]) == {"standardized", "location", "scale"},
            "RECORDED_NORMALIZER_FORWARD_CONTRACT_FAILURE")
    for key in ("eps", "use_arcsinh"):
        require(behavior["normalizer_before"][key] == behavior["normalizer_after"][key], "NORMALIZER_PARAMETER_MISMATCH")
    rows_by_key = {(row["dataset"], row["batch"], row["route"]): row for row in result["rows"]}
    require(len(rows_by_key) == 24 and set(rows_by_key) == {(dataset, batch, route)
            for dataset in producers for batch in (1, 8) for route in ROUTES}, "EXACT_24_COST_KEYS_REQUIRED")
    raw_inputs, scientific_outputs = {}, {}
    for dataset, producer in producers.items():
        limit.checkfull()
        split = "val" if dataset == "robin" else "eval"
        binding = result["dataset_input_sources"][dataset]
        require(binding["development_split"] == split and binding["inverse_normalized_proxy_used"] is False
                and binding["scientific_input"]["sha256"] == producer["inputs"][split]["sha256"], "ACTUAL_PHYSICAL_BRIDGE_AND_SCIENTIFIC_SPLIT_REQUIRED")
        original_bridge = verified_json(producer["bridge_provenance"]["input_bridge"], check)
        bridge_records = [value for value in original_bridge["artifacts"]
                          if value.get("dataset") == dataset and value.get("split") == split]
        require(len(bridge_records) == 1, "ONE_ORIGINAL_PHYSICAL_BRIDGE_ARRAY_RECORD_REQUIRED")
        physical_record = bind_canonical_npz(binding["physical_bridge"], bridge_records[0], "physical_bridge:" + dataset)
        scientific_record = bind_canonical_npz(binding["scientific_input"], producer["inputs"][split], "scientific_input:" + dataset)
        physical = verified_arrays(physical_record, check)
        raw = verified_arrays(scientific_record, check)
        for key, array in raw.items():
            exact(physical[key], array, "physical_scientific_raw_contract:" + dataset + ":" + key)
        reconstructed = scalar_preprocess(physical["history_physical"][:8], physical["history_observed"][:8], raw["mean"], raw["std"])
        exact(reconstructed, raw["history"][:8], "independent_actual_physical_FF_and_normalization:" + dataset)
        raw_inputs[dataset] = raw
        controls = verified_arrays(producer["shared_mean_artifact"], check)
        fit_raw = verified_arrays(producer["inputs"]["fit"], check)
        expected_mean, expected_count = fit_mean_innovation(fit_raw["target"], fit_raw["raw_last"], fit_raw["observed"])
        exact(controls["innovation"], expected_mean, "shared_mean_for_cost:" + dataset)
        exact(controls["observed_counts"], expected_count, "shared_mean_observability_for_cost:" + dataset)
        scientific_outputs[dataset, "PERSIST"] = np.repeat(raw["raw_last"][:, None], 48, axis=1)
        scientific_outputs[dataset, "MEAN"] = raw["raw_last"][:, None] + controls["innovation"]
        scientific_outputs[dataset, "RAW_BOLT"] = raw["raw_forecast"]
        for route in ROUTES[2:5]:
            scientific_outputs[dataset, route] = verified_arrays(producer["families"][route]["selected_artifacts"][split], check)["prediction"]
    conditions = []
    total_repeats = 0
    for dataset in producers:
        for batch in (1, 8):
            raw = raw_inputs[dataset]
            for block in range(6):
                positions = [rows_by_key[dataset, batch, route]["positions"][block] for route in ROUTES]
                require(sorted(positions) == list(range(6)), "EVERY_BLOCK_HAS_EACH_ROUTE_ONCE")
            raw_row = rows_by_key[dataset, batch, "RAW_BOLT"]
            for route in ROUTES:
                check()
                row = rows_by_key[dataset, batch, route]
                require(row["status"] == "COMPLETE" and len(row["parity"]) == len(row["memory"]) == len(row["sample_artifacts"]) == 6
                        and row["stage_decomposition"] is False and row["optimized_native_implementation_claim"] is False,
                        "COMPLETE_REAL_WHOLE_COST_ROW_REQUIRED")
                require(all(word in row["whole_includes"] for word in ("causal", "FIT normalization", "CPU inverse", "physical return", "synchronization"))
                        and all(word in row["whole_excludes"] for word in ("load", "disk", "hashes", "guards")), "WHOLE_COST_BOUNDARY_DECLARATION_MISMATCH")
                timing = audit_recorded_times(row, raw_row)
                recorded = verified_arrays(row["timed_outputs"], check)
                reference = verified_arrays(row["reference"], check)
                channels = raw["history"].shape[2]
                require(recorded["normalized"].dtype == recorded["physical"].dtype == np.float32
                        and recorded["normalized"].shape == recorded["physical"].shape == (6, batch, 48, channels),
                        "SIX_ACTUAL_FP32_TIMED_RETURN_ARRAYS_REQUIRED")
                exact(recorded["origins"], raw["origins"][:batch], "timed_same_origins")
                exact(reference["origins"], raw["origins"][:batch], "reference_same_origins")
                exact(recorded["block"], np.arange(6, dtype=np.int64), "timed_blocks")
                exact(recorded["position"], np.asarray(row["positions"], np.int64), "timed_balanced_positions")
                scientific = scientific_outputs[dataset, route][:batch]
                norm_reference, physical_reference = reference[route + "_normalized"], reference[route + "_physical"]
                require(parity(norm_reference, scientific)["pass"], "MATCHED_REFERENCE_AND_SCIENTIFIC_RETURN_PARITY_FAILURE")
                exact(physical_reference, (norm_reference * raw["std"] + raw["mean"]).astype(np.float32), "REFERENCE_CPU_FP32_INVERSE")
                metadata = row["selected_metadata"]
                if route in ROUTES[2:5]:
                    family = producers[dataset]["families"][route]
                    require(metadata["selected_kind"] == family["selected_kind"] and metadata["selected_index"] == family["selected_index"]
                            and metadata["selected_alpha"] == family["selected_alpha"], "SELECTED_COST_MODEL_BINDING_MISMATCH")
                    if metadata["selected_kind"] == "RIDGE":
                        require(metadata["selected_model_sha256"] == family["selected_model_sha256"], "SELECTED_COST_MODEL_SHA_MISMATCH")
                for block in range(6):
                    check()
                    normalized, physical = recorded["normalized"][block], recorded["physical"][block]
                    exact(physical, (normalized * raw["std"] + raw["mean"]).astype(np.float32), "TIMED_CPU_FP32_INVERSE")
                    expected = {"normalized": parity(normalized, norm_reference),
                                "physical": parity(physical, physical_reference, atol=PHYSICAL_ATOL, rtol=PHYSICAL_RTOL),
                                "scientific_normalized": parity(normalized, scientific)}
                    for name, value in expected.items():
                        compare_parity(row["parity"][block][name], value, route + ":" + str(block) + ":" + name)
                    sample = verified_arrays(row["sample_artifacts"][block], check)
                    exact(sample["normalized"], normalized, "SINGLE_SAMPLE_VS_STACKED_NORMALIZED")
                    exact(sample["physical"], physical, "SINGLE_SAMPLE_VS_STACKED_PHYSICAL")
                    exact(sample["origins"], raw["origins"][:batch], "SINGLE_SAMPLE_ORIGINS")
                    memory = row["memory"][block]
                    require(0 <= memory["peak_allocated_vram_bytes"] <= memory["peak_reserved_vram_bytes"] <= limit.limits["reserved_vram_bytes"]
                            and 0 < memory["resident_rss_bytes"] <= limit.limits["process_rss_bytes"], "RECORDED_MEMORY_CAP_OR_ACCOUNTING_MISMATCH")
                    total_repeats += 1
                conditions.append({"dataset": dataset, "batch": batch, "route": route, "status": "PASS", **timing,
                                   "saved_repeat_outputs_checked": 6, "actual_same_scientific_origins": True,
                                   "CPU_FP32_inverse_exact": True, "selected_metadata": metadata})
    require(total_repeats == 144, "EXACT_144_SAVED_TIMED_REPEAT_CHECKS_REQUIRED")
    limit.checkfull()
    return {"status": "PASS", "cost_producer_record": record, "producer_records": producer_records,
            "scientific_audit_record": result["independent_scientific_audit"], "rows_checked": 24,
            "timed_repeats_checked": total_repeats, "source_file_bindings_checked": source_checks,
            "source_before_after_SHA_pass": True, "conditions": conditions,
            "scope": {"saved_array_and_recorded_timing_arithmetic": True, "actual_GPU_trial_replayed": False,
                      "clock_readings_independently_remeasured": False, "memory_sampling_independently_replayed": False,
                      "independent_hardware_trials": False, "frozen_model_forward_replayed": False,
                      "preprocessing_reconstructed_from_actual_physical_history": True, "protected_TEST_accessed": False}}


def main():
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--input-result", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"scientific_execution": False, "scope": "saved cost outputs and timing arithmetic"}))
        return 0
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "2")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    sys.path[:0] = [str(HERE), str(ROOT / "research/smallk_failure_followup_20261014"), str(ROOT)]
    from adaptive_runtime_v1 import ContinuationLimit
    limit = ContinuationLimit(args.tag, gpu=False)
    output = resolve(args.output)
    report = {"status": "RUNNING", "scientific_execution": True}
    error = None
    try:
        require(output.is_relative_to(HERE) and not output.exists(), "NEW_CAMPAIGN_AUDIT_OUTPUT_ONLY")
        initialize_numpy()
        import audit_readout_v2
        audit_readout_v2.initialize_numpy()
        import prepare_inputs
        prepare_inputs.install_protected_test_guard()
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2):
            report.update(audit_cost_result(args.input_result, limit))
        report["permission_gate"] = limit.authorization_record
        report["run_manifest"] = limit.run_manifest_record
    except Exception as failure:
        import traceback
        error = {"type": type(failure).__name__, "message": str(failure), "traceback": traceback.format_exc()}
        report.update(status="FAIL", failure=error)
    finally:
        terminal = limit.close("COMPLETE" if report["status"] == "PASS" else "FAILED_PRESERVED", error)
        report["resource_terminal"] = terminal
        if terminal["terminal_status"] != "COMPLETE":
            report["status"] = "FAIL"
        if output.is_relative_to(HERE) and not output.exists():
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n")
        print(json.dumps({"audit": str(output), "status": report["status"], "rows_checked": report.get("rows_checked"), "failure": error}), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
