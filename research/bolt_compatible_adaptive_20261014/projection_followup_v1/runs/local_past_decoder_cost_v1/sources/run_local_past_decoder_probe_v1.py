"""Past-only origin-specific decoder diagnostic on immutable accepted caches."""
from pathlib import Path
import argparse
import hashlib
import json
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATASETS = {"robin": (17, 4, "val", 30), "beijing_aotizhongxin": (11, 2, "eval", 146)}
ROUTES, SEEDS = ("PCA_PINV", "FIT_TARGET_PINV"), (92601, 92602)


def record(path):
    path = Path(path).resolve()
    with path.open("rb") as handle:
        return {"path": str(path), "sha256": hashlib.file_digest(handle, "sha256").hexdigest()}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_gates(producer_path, fit_audit_path, gate_path, gate_audit_path, cost_path, cost_audit_path):
    paths = (producer_path, fit_audit_path, gate_path, gate_audit_path, cost_path, cost_audit_path)
    producer, fit_pass, gate, gate_pass, cost, cost_pass = (read(path) for path in paths)
    if any(value.get("status") != status for value, status in zip(
            (producer, fit_pass, gate, gate_pass, cost, cost_pass), ("COMPLETE", "PASS", "COMPLETE", "PASS", "COMPLETE", "PASS"))):
        raise ValueError("COMPLETE_FIT_CONSTANT_AND_WHOLE_COST_WITH_EXACT_PASSES_REQUIRED")
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from run_past_constant_gate_v1 import accepted_gate
    from run_past_constant_gate_cost_v1 import load_gates as prior_gates
    accepted_gate(producer_path, fit_audit_path)
    prior_gates(producer_path, fit_audit_path, gate_path, gate_audit_path)
    bindings = {name: record(path) for name, path in zip(
        ("producer", "fit_target_audit", "past_constant_diagnostic", "past_constant_audit",
         "past_constant_whole_cost", "past_constant_whole_cost_audit"), paths)}
    expected_refs = {"producer": bindings["producer"], "fit_target_audit": bindings["fit_target_audit"],
                     "gate_diagnostic": bindings["past_constant_diagnostic"], "gate_audit": bindings["past_constant_audit"]}
    if (any(cost.get(key) != value or cost_pass.get(key) != value for key, value in expected_refs.items())
            or cost_pass.get("cost") != bindings["past_constant_whole_cost"]):
        raise ValueError("EXACT_BASELINE_WHOLE_COST_AND_ALL_PREDECESSOR_BINDINGS_REQUIRED")
    code = cost_pass.get("audit_source", {})
    code_path = Path(code.get("path", ""))
    if (code_path.name != "audit_past_constant_gate_cost_v2.py" or not code_path.is_file()
            or record(code_path)["sha256"] != code.get("sha256")):
        raise ValueError("EXACT_WHOLE_COST_V2_AUDIT_CODE_REQUIRED")
    expected = {"cost_cells_checked": 24, "recorded_timer_samples_checked": 144,
                "physical_gate_prechecks_checked": 2, "paired_gate_timed_output_arrays_checked": 72,
                "original_selected_forecast_hashes_checked": 36}
    if any(cost_pass.get("totals", {}).get(key) != value for key, value in expected.items()):
        raise ValueError("FULL_BASELINE_WHOLE_COST_V2_COVERAGE_REQUIRED")
    if (cost.get("expected_cost_cells") != 24 or cost.get("recorded_timer_samples") != 144
            or cost.get("new_training_updates") != 0 or cost.get("projection_fits") != 0
            or cost.get("protected_TEST_accessed") is not False or cost_pass.get("protected_TEST_accessed") is not False
            or cost.get("new_backbone_training") is not False
            or cost.get("source_cache_selection_SHA_unchanged") is not True
            or cost.get("original_selected_forecasts_and_locks_unchanged") is not True):
        raise ValueError("UNCHANGED_COMPLETE_NO_TEST_BASELINE_WHOLE_COST_SCOPE_REQUIRED")
    return producer, gate, cost, bindings


def byte_equal(a, b):
    import numpy as np
    return a.dtype == b.dtype and a.shape == b.shape and np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()


def shared_native_past(first, second):
    if not byte_equal(first["latent_history"], second["latent_history"]) or not byte_equal(first["origins"], second["origins"]):
        raise ValueError("NATIVE_SEED_CACHED_FIXED_E_PAST_NOT_BYTE_IDENTICAL")
    return first["latent_history"]


def score_pair(values, original, local, static_replay):
    import numpy as np
    from decoder_metric_probe_v1 import masked_scores
    target, observed, scale = values["target"], values["observed"], values["std"]
    complete = np.broadcast_to(observed.all(axis=2, keepdims=True), observed.shape)
    def scores(prediction, mask):
        if not mask.any():
            return {"status": "NO_OBSERVED_TARGETS", "normalized": None, "physical_per_channel_diagnostic": None}
        return masked_scores(prediction, target, mask, scale)
    difference = static_replay.astype(np.float64) - original.astype(np.float64)
    return {"static_scores_all_observed": scores(original, observed), "local_scores_all_observed": scores(local, observed),
            "static_scores_complete_mask": scores(original, complete), "local_scores_complete_mask": scores(local, complete),
            "static_D_einsum_arithmetic_replay": {"max_abs_vs_original": float(np.abs(difference).max()),
                "MSE_vs_original": float(np.mean(difference ** 2)),
                "byte_exact": byte_equal(static_replay, original),
                "scores_all_observed": scores(static_replay, observed),
                "scope": "Same original D and cached future/last with new FP32 einsum order; numeric-rounding diagnostic only"}}


def backend_predictions(local_state, state_arrays, values, cached):
    import numpy as np
    from local_past_decoder_v1 import LocalPastDecoder
    n = len(values["history"])
    original_D = np.broadcast_to(state_arrays["decoder"].astype(np.float32)[None], local_state.decoder32.shape).copy()
    original_bias = np.broadcast_to(state_arrays["decoder_bias"].astype(np.float32)[None], (n, values["history"].shape[2])).copy()
    control = LocalPastDecoder(original_D.astype(np.float64), original_D, original_bias.astype(np.float64), original_bias, {})
    predictions = {}
    for name, arrays in cached.items():
        if not np.array_equal(arrays["origins"], values["origins"]):
            raise ValueError("EXACT_CACHED_FUTURE_AND_LOCAL_INPUT_ORIGINS_REQUIRED")
        future, last = arrays["latent_future"], arrays["latent_history"][:, -1]
        local = local_state.forecast(values["raw_last"], future, last)
        replay = control.forecast(values["raw_last"], future, last)
        predictions[name] = {"original": arrays["prediction"], "local": local, "static_replay": replay}
    return predictions


def isolated_method_predictions(local_state, state_arrays, values, cached):
    predictions, failures = {}, {}
    for name, arrays in cached.items():
        try:
            predictions.update(backend_predictions(local_state, state_arrays, values, {name: arrays}))
        except Exception as error:
            failures[name] = {"status": "FAILED_PATH_PRESERVED",
                              "failure": {"type": type(error).__name__, "message": str(error)}}
    return predictions, failures


def execute(producer_path, fit_audit_path, gate_path, gate_audit_path, cost_path, cost_audit_path, output_path):
    producer, gate, cost, bindings = load_gates(producer_path, fit_audit_path, gate_path, gate_audit_path, cost_path, cost_audit_path)
    output = Path(output_path).resolve()
    if not output.is_relative_to(HERE) or output.exists():
        raise ValueError("FRESH_LOCAL_DECODER_PROBE_DIRECTORY_REQUIRED")
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import numpy as np
    import fixed_decoder_span_v1 as io
    from local_past_decoder_v1 import fit_local_past_decoder
    from decoder_metric_probe_v1 import masked_scores
    from continuation_runtime_v1 import Meter, write_new
    from run_fit_target_comparison_v1 import prior_file_records
    tracked = [record(Path(__file__)), record(HERE / "local_past_decoder_v1.py"),
               record(HERE / "PLAN_LOCAL_PAST_DECODER_KO_v1.md"), record(HERE / "decoder_metric_probe_v1.py"),
               record(HERE / "fixed_decoder_span_v1.py"), record(HERE / "run_fit_target_comparison_v1.py"),
               record(HERE / "run_past_constant_gate_v1.py"), record(HERE / "run_past_constant_gate_cost_v1.py")]
    output.mkdir(parents=True, exist_ok=False)
    meter = Meter("local_past_decoder_" + output.name, gpu=False)
    report = {"status": "RUNNING", **bindings, "source_records": tracked, "datasets": {}, "failed_paths": [],
        "completed_method_pairs": 0, "completed_backend_batches": 0,
        "attempted_backend_batches": 0, "completed_window_matrix_fits": 0,
        "completed_channel_solver_calls": 0, "completed_no_observation_fallbacks": 0,
        "native_seed_D_reuses": 0, "new_E_F_native_head_forward_runs": 0,
        "new_E_F_or_native_head_training": False, "new_backbone_training": False,
        "inference_time_past_decoder_refitting": True, "fit_uses_TUNE_DEV_known_past": True,
        "future_labels_used_for_local_fit": False, "method_checkpoint_threshold_selection_used": False,
        "protected_TEST_accessed": False, "new_GPU_execution": False, "whole_local_deployment_cost_measured": False,
        "same_decoder_side_information_as_static": False, "past_C_condition_metadata_added": True,
        "strict_K_only_compression": False, "static_D_or_global_rankK_oracle_reused_as_local_lower_bound": False,
        "independent_holdout": False}
    records, failure = {}, None
    try:
        meter.protected_check()
        sys.path.insert(0, str(ROOT / "research/smallk_failure_followup_20261014"))
        from prepare_inputs import install_protected_test_guard
        report["protected_guard_paths"] = install_protected_test_guard()
        guarded = prior_file_records(producer, gate, cost, *bindings.values(), *tracked)
        for parent in (producer, cost):
            for item in parent.get("source_snapshots", []):
                snapshot = record(item["snapshot_path"])
                if snapshot["sha256"] != item["sha256"]:
                    raise ValueError("ORIGINAL_EXECUTED_SOURCE_SNAPSHOT_CHANGED")
                guarded.append(snapshot)
        scope = {"status": "CONTINUING_GOAL_CAUSAL_DIAGNOSTIC_JUDGMENT", "individual_exact_human_approval_claimed": False,
            "bindings": bindings, "plan": tracked[2], "solver": "Observed-centered slope lstsq1e-4 with analytic intercept",
            "decoder_input_interface": "K future streams + current C + N-specific KxC coefficients from known C past512/mask",
            "fit_uses_TUNE_DEV_past_at_origin": True, "future_labels_used_for_fit": False,
            "threshold_or_lambda_search": False, "noobservation_only_fallback": "Dcol0,currentC",
            "inference_time_past_decoder_refitting": True, "new_E_F_or_head_training": False,
            "strict_K_only_compression": False, "same_decoder_side_information_as_static": False,
            "whole_local_deployment_cost_measured": False, "static_and_global_rankK_oracle_lower_bounds_applicable": False,
            "independent_generalization_proven": False}
        write_new(output / "SCOPE.json", scope)
        report["scope"] = record(output / "SCOPE.json")
        for dataset, (channels, k, development, dev_count) in DATASETS.items():
            parent = producer["datasets"][dataset]
            row = {"K": k, "channels": channels, "splits": {}}
            report["datasets"][dataset] = row
            for split, count in (("fit", 256), ("tune", 64), (development, dev_count)):
                try:
                    values = io.checked_payload(parent["inputs"][split], meter, records)
                    if values["history"].shape != (count, 512, channels) or not np.array_equal(values["raw_last"], values["history"][:, -1]):
                        raise ValueError("EXACT_REGISTERED_INPUT_COUNT_CURRENT_C_REQUIRED")
                    complete = np.broadcast_to(values["observed"].all(axis=2, keepdims=True), values["observed"].shape)
                    cell = {"status": "RUNNING", "input_source": parent["inputs"][split], "routes": {},
                            "RAW_scores_all_observed": masked_scores(values["raw_forecast"], values["target"], values["observed"], values["std"]),
                            "RAW_scores_complete_mask": masked_scores(values["raw_forecast"], values["target"], complete, values["std"])}
                    row["splits"][split] = cell
                except Exception as error:
                    failed = {"dataset": dataset, "split": split, "stage": "input", "status": "FAILED_PATH_PRESERVED",
                              "failure": {"type": type(error).__name__, "message": str(error)}}
                    row["splits"][split] = failed
                    report["failed_paths"].append(failed)
                    continue
                for route in ROUTES:
                    condition = parent["projections"][route]
                    route_row = {"backends": {}}
                    cell["routes"][route] = route_row
                    for backend in ("BOLT_NUMPY", "NATIVE_TORCH32"):
                        report["attempted_backend_batches"] += 1
                        try:
                            state_arrays = io.checked_payload(condition["state_artifact"], meter, records)
                            if backend == "BOLT_NUMPY":
                                sources = {"BOLT": condition["bolt"]["artifacts"][split]}
                            else:
                                sources = {"NATIVE_s" + str(seed): condition["native"][str(seed)]["artifacts"][split] for seed in SEEDS}
                            cached = {name: io.checked_payload(item, meter, records) for name, item in sources.items()}
                            for arrays in cached.values():
                                if not np.array_equal(arrays["origins"], values["origins"]):
                                    raise ValueError("REGISTERED_ORIGINS_FOR_LOCAL_PAST_AND_FUTURE_REQUIRED")
                            z = cached["BOLT"]["latent_history"] if backend == "BOLT_NUMPY" else shared_native_past(
                                cached["NATIVE_s92601"], cached["NATIVE_s92602"])
                            if z.shape != (count, 512, k):
                                raise ValueError("EXACT_REGISTERED_FIXED_E_CACHED_PAST_REQUIRED")
                            local_state = fit_local_past_decoder(values["history"], values["history_observed"], z)
                            report["completed_window_matrix_fits"] += count
                            observed_count = np.asarray(local_state.diagnostics["true_observed_counts"])
                            report["completed_channel_solver_calls"] += int((observed_count > 0).sum())
                            report["completed_no_observation_fallbacks"] += int((observed_count == 0).sum())
                            if backend == "NATIVE_TORCH32":
                                report["native_seed_D_reuses"] += count
                            prefix = dataset + "_" + route + "_" + split + "_" + backend
                            coefficient = io.save_npz(output / (prefix + "_LOCAL_STATE.npz"), local_state.to_npz())
                            backend_row = {"status": "RUNNING", "local_state_artifact": coefficient,
                                "local_fit_metadata": local_state.metadata(), "original_state_source": condition["state_artifact"],
                                "cached_method_sources": sources, "native_cached_past_byte_exact_and_D_shared": backend == "NATIVE_TORCH32",
                                "methods": {}}
                            route_row["backends"][backend] = backend_row
                            past = local_state.reconstruct(z)
                            backend_row["past_reconstruction_deployment_scores_observed_only"] = (masked_scores(
                                past, values["history"], values["history_observed"], values["std"])
                                if values["history_observed"].any() else {"status": "NO_OBSERVED_PAST_TARGETS",
                                    "normalized": None, "physical_per_channel_diagnostic": None})
                            predictions, method_failures = isolated_method_predictions(local_state, state_arrays, values, cached)
                            for name, item in method_failures.items():
                                backend_row["methods"][name] = item
                                report["failed_paths"].append({"dataset": dataset, "route": route, "split": split,
                                    "backend": backend, "method": name, **item})
                            arrays = {"latent_history": z.copy(), "past_prediction_FP32": past, "origins": values["origins"].copy(),
                                      "target": values["target"].copy(), "observed": values["observed"].copy(), "raw_last": values["raw_last"].copy()}
                            for name, item in predictions.items():
                                backend_row["methods"][name] = score_pair(values, item["original"], item["local"], item["static_replay"])
                                backend_row["methods"][name].update(original_selected_prediction_byte_preserved=True,
                                    native_head_frozen_intervention_not_reoptimized=backend == "NATIVE_TORCH32")
                                arrays[name + "_static_prediction"] = item["original"].copy()
                                arrays[name + "_local_prediction"] = item["local"]
                                arrays[name + "_static_einsum_replay"] = item["static_replay"]
                                arrays[name + "_cached_future"] = cached[name]["latent_future"].copy()
                            backend_row["artifact"] = io.save_npz(output / (prefix + "_LOCAL_FORECASTS.npz"), arrays)
                            backend_row["status"] = "PARTIAL_COMPLETED" if method_failures else "COMPLETE"
                            if not method_failures:
                                report["completed_backend_batches"] += 1
                            report["completed_method_pairs"] += len(predictions)
                        except Exception as error:
                            failed = {"dataset": dataset, "route": route, "split": split, "backend": backend,
                                "status": "FAILED_PATH_PRESERVED", "failure": {"type": type(error).__name__, "message": str(error)}}
                            route_row["backends"].setdefault(backend, {}).update(failed)
                            report["failed_paths"].append(failed)
                cell["status"] = "COMPLETE" if all(item.get("status") == "COMPLETE"
                    for route in cell["routes"].values() for item in route["backends"].values()) else "PARTIAL_COMPLETED"
        for item in guarded + tracked + list(records.values()):
            if record(item["path"])["sha256"] != item["sha256"]:
                raise ValueError("ORIGINAL_SOURCE_CACHE_MODEL_OR_SELECTION_CHANGED")
        meter.protected_check()
        report.update(status="COMPLETE" if report["completed_method_pairs"] == 36 and not report["failed_paths"] else "PARTIAL_COMPLETED",
            protected_prior_records=guarded, input_records=list(records.values()), original_source_cache_selection_SHA_unchanged=True,
            forecast_finite_for_completed_paths=True, expected_method_pairs=36, maximum_window_matrix_fits=3264,
            maximum_channel_slots=44304, cost_status="NOT_MEASURED_LOCAL: current stage cached CPU decoder fit/reconstruction/forecast only",
            failure_count_semantics="Completed fit counts exclude failed solver batches; attempted batch count preserves failed paths without pretending every scheduled channel completed")
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        report.update(status="FAILED_PRESERVED", failure=failure, full_scope_complete_claimed=False)
    finally:
        report["resource_terminal"] = meter.close(report["status"], failure)
        write_new(output / "RESULT.json", report)
    print(json.dumps({"status": report["status"], "output": str(output / "RESULT.json"),
        "pairs": report["completed_method_pairs"], "window_fits": report["completed_window_matrix_fits"], "failure": failure}), flush=True)
    return 0 if report["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("producer", "fit-audit", "gate-diagnostic", "gate-audit", "cost", "cost-audit", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    raise SystemExit(execute(args.producer, args.fit_audit, args.gate_diagnostic, args.gate_audit, args.cost, args.cost_audit, args.output))
