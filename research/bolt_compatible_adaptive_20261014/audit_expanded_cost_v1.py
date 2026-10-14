"""Independent saved-output/recorded-time audit of the four dense CPU cells."""
from pathlib import Path
import argparse
import json
import os
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
np = sys.modules.get("numpy")


def verify_two_positions(positions):
    if len(positions) != 6 or sorted(positions) != [0, 0, 0, 1, 1, 1]:
        raise ValueError("TWO_POSITIONS_THREE_REPEATS_EACH_REQUIRED")
    return True


def audit_result(path, limit):
    import audit_readout_v2 as readout
    import audit_cost_v2 as cost
    check = limit.shared.checkfast
    producer_record = {"path": str(readout.resolve(path)), "sha256": readout.file_sha(readout.resolve(path))}
    result = readout.verified_json(producer_record, check)
    readout.require(result["status"] == "COMPLETE" and result["phase"] == "MATCHED_FIT512_ONE_DIAGNOSTIC_CPU_COST"
                    and result["planned_rows"] == len(result["rows"]) == 4, "COMPLETE_FOUR_DENSE_CPU_COST_CELLS_REQUIRED")
    readout.require(result["permission_gate"] == limit.authorization_record and result["run_manifest"] == limit.run_manifest_record,
                    "SAME_APPROVED_NEW_CAMPAIGN_REQUIRED")
    readout.require(result["GPU_execution"] is False and result["new_training_updates"] == 0
                    and result["protected_TEST_accessed"] is False and result["independent_holdout"] is False
                    and result["no_fresh_RAW_baseline"] is True and result["cross_trial_speed_superiority_claim"] is False
                    and result["no_inverse_normalized_input_proxy"] is True, "DENSE_CPU_ONLY_NO_CROSS_TRIAL_CLAIM_REQUIRED")
    scientific = readout.verified_json(result["producer_record"], check)
    audit = readout.verified_json(result["independent_scientific_audit"], check)
    readout.require(audit["status"] == "PASS" and audit["pool_candidates_checked"] == 18
                    and audit["producer_record"]["sha256"] == result["producer_record"]["sha256"]
                    and scientific["status"] == "COMPLETE" and scientific["phase"] == "MATCHED_FIT512_ONE_DIAGNOSTIC"
                    and scientific["dataset"] == "robin", "EXACT_DENSE18_SCIENTIFIC_AUDIT_PRODUCER_BINDING_REQUIRED")
    bridge = readout.verified_json(scientific["bridge_provenance"]["input_bridge"], check)
    canonical = [record for record in bridge["artifacts"] if record.get("dataset") == "robin" and record.get("split") == "val"]
    readout.require(len(canonical) == 1, "ONE_ORIGINAL_PHYSICAL_VAL_MANIFEST_REQUIRED")
    physical_record = cost.bind_canonical_npz(result["physical_input_source"], canonical[0], "dense_ROBIN_VAL_physical")
    physical = readout.verified_arrays(physical_record, check)
    raw = readout.verified_arrays(scientific["inputs"]["val"], check)
    for key, value in raw.items():
        readout.exact(physical[key], value, "dense_physical_scientific_inputs:" + key)
    independent_history = cost.scalar_preprocess(physical["history_physical"][:8], physical["history_observed"][:8], raw["mean"], raw["std"])
    readout.exact(independent_history, raw["history"][:8], "dense_actual_physical_scalar_FF_normalization")
    source_checks = 0
    for record in result["source_records"] + result["source_snapshots"]:
        check()
        readout.require(readout.file_sha(readout.resolve(record["path"])) == record["sha256"], "DENSE_COST_SOURCE_BYTES_CHANGED")
        source_checks += 1
        if "snapshot_path" in record:
            readout.require(readout.file_sha(readout.resolve(record["snapshot_path"])) == record["sha256"], "DENSE_COST_SNAPSHOT_BYTES_CHANGED")
            source_checks += 1
    readout.require(result["source_before_after_SHA_pass"] is True, "DENSE_COST_SOURCE_BEFORE_AFTER_ASSERTION_REQUIRED")
    rows = {(row["batch"], row["route"]): row for row in result["rows"]}
    readout.require(len(rows) == 4 and set(rows) == {(batch, route) for batch in (1, 8) for route in ("SELF33", "JOINT33")},
                    "EXACT_FOUR_DENSE_CPU_KEYS_REQUIRED")
    outputs = {}
    for route in ("SELF33", "JOINT33"):
        selected = readout.verified_arrays(scientific["families"][route]["selected_artifacts"]["val"], check)
        for key in ("target", "observed", "origins", "raw_last", "mean", "std"):
            readout.exact(selected[key], raw[key], "dense_selected_cost_same_contract:" + route + ":" + key)
        outputs[route] = selected["prediction"]
    conditions = []
    repeats_checked = 0
    for batch in (1, 8):
        for block in range(6):
            readout.require(sorted([rows[batch, route]["positions"][block] for route in ("SELF33", "JOINT33")]) == [0, 1],
                            "DENSE_EACH_BLOCK_TWO_DIFFERENT_POSITIONS_REQUIRED")
        for route in ("SELF33", "JOINT33"):
            row = rows[batch, route]
            verify_two_positions(row["positions"])
            readout.require(row["status"] == "COMPLETE" and row["dataset"] == "robin" and row["repeats"] == 6
                            and row["warmup_each_route"] == 2 and row["GPU_executed"] is False and row["GPU_resident_model"] is False
                            and row["paired_RAW_ratio"] is None and row["blocks"] == list(range(6)), "DENSE_CPU_COST_ROW_CONTRACT_REQUIRED")
            readout.require(len(row["seconds"]) == 6 and all(value > 0 and np.isfinite(value) for value in row["seconds"]),
                            "SIX_FINITE_POSITIVE_DENSE_RECORDED_TIMES_REQUIRED")
            median = float(statistics.median(row["seconds"]))
            readout.close(row["median_seconds"], median, "dense_recorded_time_median", rtol=1e-12, atol=1e-12)
            readout.close(row["median_seconds_per_window"], median / batch, "dense_time_batch_scaling", rtol=1e-12, atol=1e-12)
            metadata = result["metadata"][route]
            family = scientific["families"][route]
            readout.require(metadata["FIT_rows"] == 512 and metadata["selected_kind"] == family["selected_kind"]
                            and metadata["selected_alpha"] == family["selected_alpha"], "DENSE_SELECTED_MODEL_METADATA_MISMATCH")
            if metadata["selected_kind"] == "RIDGE":
                readout.require(metadata["selected_model_sha256"] == family["selected_model_sha256"], "DENSE_SELECTED_MODEL_SHA_MISMATCH")
            timed = readout.verified_arrays(row["timed_outputs"], check)
            readout.require(timed["normalized"].dtype == timed["physical"].dtype == np.float32
                            and timed["normalized"].shape == timed["physical"].shape == (6, batch, 48, 17), "SIX_DENSE_FP32_RETURN_ARRAYS_REQUIRED")
            readout.exact(timed["origins"], raw["origins"][:batch], "dense_timed_same_origins")
            readout.exact(timed["block"], np.arange(6, dtype=np.int64), "dense_timed_blocks")
            readout.exact(timed["position"], np.asarray(row["positions"], np.int64), "dense_timed_positions")
            sampled_rss = []
            for block in range(6):
                check()
                normalized, physical_return = timed["normalized"][block], timed["physical"][block]
                expected_parity = cost.parity(normalized, outputs[route][:batch])
                cost.compare_parity(row["normalized_parity"][block], expected_parity, "dense_scientific_match:" + route)
                readout.exact(physical_return, (normalized * raw["std"] + raw["mean"]).astype(np.float32), "dense_CPU_FP32_inverse")
                sample_record = row["sample_artifacts"][block]
                sample = readout.verified_arrays(sample_record, check)
                readout.exact(sample["normalized"], normalized, "dense_sample_vs_stacked_normalized")
                readout.exact(sample["physical"], physical_return, "dense_sample_vs_stacked_physical")
                readout.exact(sample["origins"], raw["origins"][:batch], "dense_sample_origins")
                timing_path = readout.resolve(sample_record["path"]).with_suffix(".json")
                item = json.loads(timing_path.read_text(encoding="utf-8-sig"))
                readout.require(item["route"] == route and item["batch"] == batch and item["block"] == block
                                and item["position"] == row["positions"][block] and item["seconds"] == row["seconds"][block]
                                and item["sample_artifact"] == sample_record, "DENSE_PER_SAMPLE_TIME_METADATA_MISMATCH")
                readout.require(0 < item["resident_rss_bytes"] <= limit.limits["process_rss_bytes"], "DENSE_RECORDED_RSS_CAP_FAILURE")
                sampled_rss.append(item["resident_rss_bytes"])
                repeats_checked += 1
            readout.require(max(sampled_rss) == row["sampled_peak_rss_bytes"], "DENSE_RECORDED_RSS_PEAK_ARITHMETIC_MISMATCH")
            conditions.append({"status": "PASS", "dataset": "robin", "route": route, "batch": batch,
                               "median_seconds": median, "median_seconds_per_window": median / batch,
                               "saved_repeat_outputs_checked": 6, "CPU_FP32_inverse_exact": True,
                               "two_positions_three_times_each": True, "fresh_RAW_baseline": False})
    readout.require(repeats_checked == 24, "EXACT_24_DENSE_SAVED_SAMPLES_REQUIRED")
    limit.checkfull()
    return {"status": "PASS", "cost_producer_record": producer_record, "producer_record": result["producer_record"],
            "scientific_audit_record": result["independent_scientific_audit"], "rows_checked": 4,
            "timed_repeats_checked": repeats_checked, "source_before_after_SHA_pass": True,
            "source_file_bindings_checked": source_checks, "conditions": conditions,
            "scope": {"saved_array_and_recorded_timing_arithmetic": True, "actual_GPU_trial_replayed": False,
                      "clock_readings_independently_remeasured": False, "memory_sampling_independently_replayed": False,
                      "independent_hardware_trials": False, "frozen_model_forward_replayed": False,
                      "fresh_RAW_comparator_present": False, "cross_trial_speed_superiority_claim": False,
                      "protected_TEST_accessed": False}}


def main():
    global np
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--input-result", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not args.execute:
        return 0
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "2")
    sys.path[:0] = [str(HERE), str(ROOT / "research/smallk_failure_followup_20261014"), str(ROOT)]
    from adaptive_runtime_v1 import ContinuationLimit
    limit = ContinuationLimit(args.tag, gpu=False)
    output = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    report = {"status": "RUNNING", "scientific_execution": True}
    error = None
    try:
        import numpy as np
        import audit_readout_v2
        import audit_cost_v2
        audit_readout_v2.initialize_numpy()
        audit_cost_v2.initialize_numpy()
        audit_readout_v2.require(output.is_relative_to(HERE) and not output.exists(), "NEW_CAMPAIGN_AUDIT_OUTPUT_ONLY")
        import prepare_inputs
        prepare_inputs.install_protected_test_guard()
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2):
            report.update(audit_result(args.input_result, limit))
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
