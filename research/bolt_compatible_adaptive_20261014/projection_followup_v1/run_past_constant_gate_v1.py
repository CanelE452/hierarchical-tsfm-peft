"""CPU cached-output gate diagnostic; no encoder, forecaster, head or fit rerun."""
from pathlib import Path
import argparse
import hashlib
import json
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATASETS = {"robin": (17, 4, "val", 30), "beijing_aotizhongxin": (11, 2, "eval", 146)}
ROUTES = ("PCA_PINV", "FIT_TARGET_PINV")
METHODS = ("RAW_BOLT", "PCA_PINV_BOLT", "PCA_PINV_NATIVE_s92601", "PCA_PINV_NATIVE_s92602",
           "FIT_TARGET_PINV_BOLT", "FIT_TARGET_PINV_NATIVE_s92601", "FIT_TARGET_PINV_NATIVE_s92602")


def record(path):
    path = Path(path).resolve()
    with path.open("rb") as handle:
        return {"path": str(path), "sha256": hashlib.file_digest(handle, "sha256").hexdigest()}


def accepted_gate(result_path, audit_path):
    source_path, audit_path = Path(result_path).resolve(), Path(audit_path).resolve()
    source = json.loads(source_path.read_text(encoding="utf-8-sig"))
    audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))
    if source.get("status") != "COMPLETE" or audit.get("status") != "PASS":
        raise ValueError("COMPLETE_FIT_TARGET_AND_V2_PASS_REQUIRED")
    binding = audit.get("producer", {})
    path = Path(binding.get("path", ""))
    path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    source_record = record(source_path)
    if path != source_path or binding.get("sha256") != source_record["sha256"]:
        raise ValueError("EXACT_FIT_TARGET_V2_AUDIT_PRODUCER_BINDING_REQUIRED")
    audit_source = audit.get("audit_source", {})
    audit_code = Path(audit_source.get("path", ""))
    if (audit_code.name != "audit_fit_target_comparison_v2.py"
            or not audit_code.is_file() or record(audit_code)["sha256"] != audit_source.get("sha256")):
        raise ValueError("EXACT_INDEPENDENT_V2_AUDIT_SOURCE_REQUIRED")
    expected = {"projection_states_checked": 4, "native_fits_checked": 8,
                "checkpoint_states_checked": 40, "score_field_checks": 1464,
                "saved_prediction_arrays_checked": 88, "cost_cells_checked": 20,
                "recorded_timer_samples_checked": 120, "target_projection_fits_checked": 2,
                "reused_PCA_controls_checked": 2, "output_subspace_diagnostic_cells_checked": 12}
    if any(audit.get("totals", {}).get(key) != value for key, value in expected.items()):
        raise ValueError("FULL_ACCEPTED_FIT_TARGET_V2_AUDIT_COVERAGE_REQUIRED")
    if (source.get("protected_TEST_accessed") is not False or audit.get("protected_TEST_accessed") is not False
            or source.get("new_backbone_training") is not False
            or source.get("native_training_fits") != 8 or source.get("projection_fits") != 2
            or source.get("reused_projection_states") != 2 or source.get("projection_conditions") != 4
            or source.get("failed_paths")):
        raise ValueError("EXACT_ACCEPTED_NO_TEST_FROZEN_FIT_TARGET_SCOPE_REQUIRED")
    return source, source_record, record(audit_path), audit_source


def evaluate_cached_case(values, methods):
    import numpy as np
    from past_constant_gate_v1 import past_constant_gate, apply_output_gate
    from decoder_metric_probe_v1 import masked_scores
    history, past_observed = values["history"], values["history_observed"]
    target, observed, current = values["target"], values["observed"], values["raw_last"]
    if (target.dtype != np.float32 or target.shape != (len(history), 48, history.shape[-1])
            or observed.dtype != np.bool_ or observed.shape != target.shape
            or not np.isfinite(target[observed]).all() or not np.array_equal(current, history[:, -1])):
        raise ValueError("EXACT_CACHED_TARGET_MASK_CURRENT_ANCHOR_CONTRACT_REQUIRED")
    gate = past_constant_gate(history, past_observed)
    eligible = ~gate["keep_forecast"]
    complete = np.broadcast_to(observed.all(axis=2, keepdims=True), observed.shape)
    eligible_future = np.broadcast_to(eligible[:, None], observed.shape)
    changed_future = observed & eligible_future & (target != current[:, None])
    eligible_future_observed = eligible_future & observed
    summary = {"status": "COMPLETE", "windows": len(history), "channels": history.shape[-1],
        "exact_flat_windows_per_channel": gate["exact_flat"].sum(axis=0).tolist(),
        "gate_eligible_windows_per_channel": eligible.sum(axis=0).tolist(),
        "eligible_windows_with_any_observed_future_per_channel": (eligible & observed.any(axis=1)).sum(axis=0).tolist(),
        "eligible_windows_with_observed_future_change_per_channel": changed_future.any(axis=1).sum(axis=0).tolist(),
        "eligible_observed_future_positions_per_channel": eligible_future_observed.sum(axis=(0, 1)).tolist(),
        "eligible_observed_future_changed_positions_per_channel": changed_future.sum(axis=(0, 1)).tolist(),
        "future_change_diagnostic_scope": "Stored normalized FP32 target vs current FP32 at observed future positions; diagnosis only; normalization/rounding can hide physical changes",
        "complete_target_positions": int(observed.all(axis=2).sum()),
        "total_target_positions": int(observed.shape[0] * observed.shape[1]), "methods": {}}
    arrays = {**gate, "gate_eligible": eligible, "target": target.copy(), "observed": observed.copy(),
              "raw_last": current.copy(), "origins": values["origins"].copy(), "FIT_std": values["std"].copy(),
              "FIT_mean": values["mean"].copy(), "complete_target_positions_mask": observed.all(axis=2)}
    def scores(prediction, mask):
        if not mask.any():
            return {"status": "NO_OBSERVED_TARGETS", "normalized": None, "physical_per_channel_diagnostic": None}
        return masked_scores(prediction, target, mask, values["std"])
    for name, prediction in methods.items():
        gated = apply_output_gate(prediction, current, gate["keep_forecast"])
        keep = np.broadcast_to(gate["keep_forecast"][:, None], prediction.shape)
        old_bits, new_bits = prediction.view(np.uint32), gated.view(np.uint32)
        anchor_bits = np.broadcast_to(current[:, None], prediction.shape).view(np.uint32)
        if not (np.array_equal(new_bits[keep], old_bits[keep]) and np.array_equal(new_bits[~keep], anchor_bits[~keep])):
            raise ValueError("GATE_UNCHANGED_COORDINATES_OR_CURRENT_ANCHOR_BIT_PARITY_FAILED")
        arrays[name + "_ungated_prediction"] = prediction.copy()
        arrays[name + "_gated_prediction"] = gated
        summary["methods"][name] = {
            "ungated_scores_all_observed": scores(prediction, observed),
            "gated_scores_all_observed": scores(gated, observed),
            "ungated_scores_complete_mask": scores(prediction, complete),
            "gated_scores_complete_mask": scores(gated, complete),
            "unchanged_coordinates_byte_exact": True, "gated_coordinates_current_anchor_byte_exact": True,
            "changed_prediction_coordinates_per_channel": (old_bits != new_bits).sum(axis=(0, 1)).tolist(),
            "native_head_frozen_intervention": "NATIVE" in name,
            "optimized_gated_native_head_claimed": False}
    return summary, arrays


def execute(result_path, audit_path, output_path):
    source, source_record, audit_record, audit_source = accepted_gate(result_path, audit_path)
    output = Path(output_path).resolve()
    if not output.is_relative_to(HERE) or output.exists():
        raise ValueError("FRESH_PAST_CONSTANT_OUTPUT_DIRECTORY_REQUIRED")
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import numpy as np
    import fixed_decoder_span_v1 as span
    from continuation_runtime_v1 import Meter, write_new
    from run_fit_target_comparison_v1 import prior_file_records
    output.mkdir(parents=True, exist_ok=False)
    meter = Meter("past_constant_" + output.name, gpu=False)
    tracked = [record(Path(__file__)), record(HERE / "past_constant_gate_v1.py"),
               record(HERE / "PLAN_PAST_CONSTANT_GATE_KO_v1.md"),
               record(HERE / "decoder_metric_probe_v1.py"), record(HERE / "fixed_decoder_span_v1.py"),
               record(HERE / "run_fit_target_comparison_v1.py")]
    scope = {"status": "CONTINUING_GOAL_SCOPE_JUDGMENT", "individual_exact_human_method_approval_claimed": False,
        "purpose": "Known-past exact-constant conditional output diagnostic on the accepted FT cached forecasts",
        "source_producer": source_record, "source_audit": audit_record, "source_audit_code": audit_source,
        "plan": tracked[2], "decoder_input_interface": "K streams + current C + past-constant flags (C bits)",
        "past_C_condition_metadata_added": True, "same_decoder_side_information_as_ungated": False,
        "strict_K_only_compression": False, "representation_learning_improvement_claimed": False,
        "threshold_tuning_or_fit": False, "protected_TEST_accessed": False,
        "new_encoder_forecaster_or_head_forward": False, "new_training": False,
        "whole_deployment_cost_measured": False, "original_static_fixed_D_floor_is_gated_lower_bound": False}
    write_new(output / "SCOPE.json", scope)
    report = {"status": "RUNNING", "producer": source_record, "independent_FIT_target_audit": audit_record,
        "scope": record(output / "SCOPE.json"), "source_records": tracked, "datasets": {},
        "completed_dataset_split_cells": 0, "completed_method_pairs": 0, "failed_paths": [],
        "new_training": False, "new_encoder_forecaster_or_head_forward": False, "protected_TEST_accessed": False,
        "whole_deployment_cost_measured": False, "independent_holdout": False,
        "strict_K_only_compression": False, "past_C_condition_metadata_added": True,
        "same_decoder_side_information_as_ungated": False,
        "static_original_fixed_D_floor_reused_as_gated_lower_bound": False}
    records, error = {}, None
    try:
        meter.protected_check()
        sys.path.insert(0, str(ROOT / "research/smallk_failure_followup_20261014"))
        from prepare_inputs import install_protected_test_guard
        report["protected_guard_paths"] = install_protected_test_guard()
        protected = prior_file_records(source, source_record, audit_record, audit_source)
        if span.sha(source["selection_lock"]["path"]) != source["selection_lock"]["sha256"]:
            raise ValueError("UNCHANGED_SOURCE_SELECTION_LOCK_REQUIRED")
        for dataset, (channels, k, development, dev_count) in DATASETS.items():
            parent = source["datasets"][dataset]
            if parent["K"] != k or set(parent["inputs"]) != {"fit", "tune", development}:
                raise ValueError("REGISTERED_DATASET_K_AND_SPLITS_REQUIRED")
            report["datasets"][dataset] = {"K": k, "channels": channels, "splits": {}}
            for split, count in (("fit", 256), ("tune", 64), (development, dev_count)):
                try:
                    values = span.checked_payload(parent["inputs"][split], meter, records)
                    if values["history"].shape != (count, 512, channels):
                        raise ValueError("REGISTERED_FIT_TUNE_EXPOSED_DEV_WINDOW_COUNTS_REQUIRED")
                    methods, sources = {"RAW_BOLT": values["raw_forecast"]}, {"RAW_BOLT": parent["inputs"][split]}
                    for route in ROUTES:
                        condition = parent["projections"][route]
                        for method, file_record in [(route + "_BOLT", condition["bolt"]["artifacts"][split]),
                            *[(route + "_NATIVE_s" + str(seed), condition["native"][str(seed)]["artifacts"][split])
                              for seed in (92601, 92602)]]:
                            cached = span.checked_payload(file_record, meter, records)
                            if not np.array_equal(cached["origins"], values["origins"]):
                                raise ValueError("EXACT_CACHED_SOURCE_ORIGINS_REQUIRED:" + method)
                            methods[method], sources[method] = cached["prediction"], file_record
                    if set(methods) != set(METHODS):
                        raise ValueError("EXACT_RAW_PLUS_SIX_FIXED_PROJECTION_METHODS_REQUIRED")
                    summary, arrays = evaluate_cached_case(values, methods)
                    summary["input_source"] = parent["inputs"][split]
                    summary["cached_prediction_sources"] = sources
                    summary["artifact"] = span.save_npz(output / f"{dataset}_{split}_PAST_CONSTANT_GATE.npz", arrays)
                    report["datasets"][dataset]["splits"][split] = summary
                    report["completed_dataset_split_cells"] += 1
                    report["completed_method_pairs"] += len(methods)
                except Exception as failure:
                    row = {"dataset": dataset, "split": split, "status": "FAILED_PATH_PRESERVED",
                           "failure": {"type": type(failure).__name__, "message": str(failure)}}
                    report["datasets"][dataset]["splits"][split] = row
                    report["failed_paths"].append(row)
        for binding in protected + tracked + list(records.values()):
            if record(binding["path"])["sha256"] != binding["sha256"]:
                raise ValueError("SOURCE_CACHE_MODEL_OR_SELECTION_CHANGED_DURING_GATE_DIAGNOSTIC")
        meter.protected_check()
        report.update(status="COMPLETE" if report["completed_method_pairs"] == 42 and not report["failed_paths"] else "PARTIAL_COMPLETED",
            protected_prior_records=protected, input_records=list(records.values()), source_cache_selection_SHA_unchanged=True,
            cached_CPU_output_gate_inferences=report["completed_method_pairs"],
            E_F_native_head_forward_runs=0,
            method_selection_used=False, future_labels_used_for_gate=False,
            cost_status="NOT_MEASURED: cached CPU gate only; future whole deployment cost must include C-past flags",
            scope_limit="Exploratory conditional output control after exposed development results; no future-flatness guarantee or representation improvement evidence")
    except Exception as failure:
        error = {"type": type(failure).__name__, "message": str(failure)}
        report.update(status="FAILED_PRESERVED", failure=error, full_scope_complete_claimed=False)
    finally:
        report["resource_terminal"] = meter.close(report["status"], error)
        write_new(output / "RESULT.json", report)
    print(json.dumps({"status": report["status"], "output": str(output / "RESULT.json"),
        "cells": report["completed_dataset_split_cells"], "method_pairs": report["completed_method_pairs"], "failure": error}), flush=True)
    return 0 if report["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    raise SystemExit(execute(arguments.result, arguments.audit, arguments.output))
