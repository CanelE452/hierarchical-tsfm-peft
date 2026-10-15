"""Gated CPU 2x2 decoder comparison using immutable actual Bolt caches only."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
ROOT, OLD = HERE.parents[2], HERE.parents[2] / "research/smallk_failure_followup_20261014"
DATASETS = {"robin": (17, 4, "val", 30), "beijing_aotizhongxin": (11, 2, "eval", 146)}
SPLIT_COUNTS = {"fit": 256, "tune": 64}
SPACES = ("PCA", "FIT_TARGET", "RAW_SUBSET")
CONDITIONS = {"L0_RECON": ((0,), "RECON"), "L7_RECON": ((0, 1, 2, 4, 8, 16, 32), "RECON"),
              "L0_FORECAST": ((0,), "FORECAST"), "L7_FORECAST": ((0, 1, 2, 4, 8, 16, 32), "FORECAST")}
SCIENCE_SHA = "2b271ba9aa6ae359033444b282ddc1becc5d1f60ef063451171e9c93f70db4ff"
SCIENCE_AUDIT_SHA = "e177652ac34414bccfbfc564da820fca0bde5a462922ceb0edaf04c47081e95d"
REPORT_SHA = "49130355afe1d89e43d130b416a536c6f683f36a76ab0d4e844afe11ec1a652b"
REPORT_MANIFEST_SHA = "a6d8a837a543712402224d742fcc7c5e08b9eba1333968f6e53baa56a8c45e36"
REPORT_AUDIT_SHA = "a184351bcbc503c5308cd3cb74232fb470136f7cdcc8711cc3ac20972c86cb00"
SCIENCE_AUDITOR_SHA = "0782c7d8d66bae5bd124c5c778eadde08786e2da2246882233eb4489b3410b20"
REPORT_AUDITOR_SHA = "acf9752f59cd2402780254cdc5b11f16605a6ff4faeab65a28747a3ffdc3ac03"
SCIENCE_COUNTS = {"subset_selection_fits_checked": 2, "greedy_candidate_scores_checked": 83,
    "global_decoder_states_checked": 12, "source_encoder_states_checked": 6, "source_split_contexts_checked": 18,
    "native_fits_checked": 12, "training_updates_logged_checked": 7200, "checkpoint_states_checked": 60,
    "checkpoint_TUNE_prediction_arrays_checked": 60, "selected_native_split_prediction_arrays_checked": 36,
    "Bolt_split_prediction_arrays_checked": 18, "actual_subset_F_split_provenance_checks": 6,
    "actual_subset_F_model_batch_calls_registered": 103, "reused_mixed_F_cache_lineages_checked": 12,
    "resolved_native_selection_locks_checked": 12, "cached_future_label_perturbation_checks": 12,
    "new_fixed_D_output_space_cells_computed": 36, "new_E_Y_diagnostic_predictions_computed": 36,
    "new_fixed_D_real_method_inequalities_checked": 90, "RAW_bound_inequalities_checked": 0,
    "current_C_persistence_split_baselines_computed": 6, "score_field_checks": 1800,
    "protected_prior_file_hashes_checked": 778}


def require(value, message):
    if not value: raise ValueError(message)


def resolve(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def record(path):
    path = resolve(path)
    with path.open("rb") as handle:
        return {"path": str(path), "sha256": hashlib.file_digest(handle, "sha256").hexdigest()}


def bound(binding, expected=None):
    path = resolve(binding["path"])
    require(path.is_relative_to(ROOT) and path.name.lower() not in ("test.npz", "jena_test.npz"), "LOCAL_NON_TEST_BINDING_REQUIRED")
    require(record(path)["sha256"] == binding["sha256"], "EXACT_IMMUTABLE_FILE_SHA_REQUIRED:" + path.name)
    if "bytes" in binding: require(path.stat().st_size == binding["bytes"], "EXACT_OPTIONAL_FILE_BYTES_REQUIRED")
    if expected is not None:
        require(path == resolve(expected["path"]) and binding["sha256"] == expected["sha256"], "EXACT_PARENT_BINDING_REQUIRED")
    return path


def read(path):
    path = resolve(path)
    require(path.suffix.lower() == ".json", "JSON_ONLY_GATE_READ_REQUIRED")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_gates(science_path, audit_path, report_audit_path):
    source, audit, report_audit = read(science_path), read(audit_path), read(report_audit_path)
    source_record, audit_record, report_audit_record = record(science_path), record(audit_path), record(report_audit_path)
    require(source_record["sha256"] == SCIENCE_SHA and audit_record["sha256"] == SCIENCE_AUDIT_SHA
        and source.get("status") == "COMPLETE" and audit.get("status") == "PASS", "EXACT_COMPLETE_SCIENCE_AND_PASS_REQUIRED_BEFORE_ARRAY_IMPORT")
    bound(audit["diagnostic"], source_record)
    require(all(audit.get("totals", {}).get(key) == value for key, value in SCIENCE_COUNTS.items()), "FULL_ACCEPTED_SCIENCE_AUDIT_COVERAGE")
    require(source.get("failed_paths") == [] and source.get("saved_Bolt_split_predictions") == 18
        and source.get("F_before") == source.get("F_after") and source.get("all_twelve_locks_before_any_DEV_input_or_prediction") is True,
        "COMPLETE_UNCHANGED_F_ACCEPTED_SOURCE")
    for value in (source, audit, report_audit):
        require(value.get("protected_TEST_accessed") is False, "NO_PROTECTED_TEST_SOURCE_SCOPE")
    require(source.get("independent_holdout") is False and source.get("new_backbone_training") is False, "EXPOSED_DEV_FROZEN_BACKBONE_SCOPE")
    code = audit["audit_source"]
    require(bound(code).name == "audit_raw_subset_comparison_v3.py" and code["sha256"] == SCIENCE_AUDITOR_SHA, "CANONICAL_SCIENCE_AUDITOR_SOURCE")
    require(report_audit_record["sha256"] == REPORT_AUDIT_SHA and report_audit.get("status") == "PASS", "EXACT_PARENT_REPORT_PASS_REQUIRED")
    require(report_audit["files_checked"] == 32 and report_audit["total_CSV_primitive_fields_checked"] == 171848
        and report_audit["local_links_checked"] == 47 and report_audit["PNG_count"] == report_audit["SVG_count"] == 6,
        "FULL_PARENT_REPORT_VERIFICATION_REQUIRED")
    require(bound(report_audit["report"]).name == "REPORT_KO.md" and report_audit["report"]["sha256"] == REPORT_SHA
        and report_audit["manifest"]["sha256"] == REPORT_MANIFEST_SHA, "EXACT_ACCEPTED_REPORT_AND_MANIFEST")
    verifier = report_audit["audit_source"]
    require(bound(verifier).name == "audit_raw_subset_report_v1.py" and verifier["sha256"] == REPORT_AUDITOR_SHA, "CANONICAL_REPORT_AUDITOR_SOURCE")
    for key, expected in (("science", source_record), ("audit", audit_record)):
        bound(report_audit["inputs"][key], expected)
    manifest_path = bound(report_audit["manifest"])
    manifest = read(manifest_path)
    require(manifest.get("status") == "COMPLETE" and len(manifest["files"]) == 32, "COMPLETE_PARENT_REPORT_MANIFEST")
    report_files = []
    for entry in manifest["files"]:
        path = (manifest_path.parent / entry["path"]).resolve()
        require(path.is_relative_to(manifest_path.parent), "PARENT_REPORT_FILE_CONTAINMENT")
        bound({**entry, "path": str(path)})
        report_files.append({**entry, "path": str(path)})
    return source, {"raw_subset_diagnostic": source_record, "raw_subset_audit": audit_record,
        "parent_report": report_audit["report"], "parent_report_manifest": report_audit["manifest"],
        "parent_report_audit": report_audit_record, "science_audit_source": code, "report_audit_source": verifier,
        "parent_report_file_records": report_files}


def fresh_output(tag):
    require(isinstance(tag, str) and bool(tag) and Path(tag).name == tag and tag not in (".", ".."), "SINGLE_NEW_RUN_TAG_REQUIRED")
    output = (HERE / "runs" / tag).resolve()
    require(output.parent == (HERE / "runs").resolve() and not output.exists(), "FRESH_OUTPUT_DIRECTORY_REQUIRED")
    return output


def byte_equal(first, second):
    return first.dtype == second.dtype and first.shape == second.shape and first.tobytes() == second.tobytes()


def failure_row(error, dataset, space, condition, stage, split=None):
    row = {"status": "FAILED_PATH_PRESERVED", "dataset": dataset, "space": space, "condition": condition,
           "stage": stage, "failure": {"type": type(error).__name__, "message": str(error)}}
    if split is not None: row["split"] = split
    return row


def fit_all_conditions(report, output, load_context, fit_decoder, save_state, write_new, check):
    states, locks = {}, {}
    for dataset, (channels, k, development, _) in DATASETS.items():
        row = report["datasets"][dataset] = {"channels": channels, "K": k, "development_split": development, "spaces": {}}
        for space in SPACES:
            parent = row["spaces"][space] = {"conditions": {}}
            try:
                context, error = load_context(dataset, space, "fit"), None
                parent["FIT_input_source"], parent["E_source"] = context["input_source"], context["E_source"]
                parent["FIT_Bolt_cache"] = context["cached_Bolt_source"]
            except Exception as caught:
                context, error = None, caught
            for condition, (lags, objective) in CONDITIONS.items():
                key, prefix = (dataset, space, condition), dataset + "_" + space + "_" + condition
                try:
                    if error is not None: raise ValueError("FIT_CONTEXT_UNAVAILABLE:" + str(error))
                    check({"stage": "FIT_D", "dataset": dataset, "space": space, "condition": condition})
                    started = time.perf_counter()
                    fitted = fit_decoder(context, lags, objective)
                    artifact = save_state(output / (prefix + "_STATE.npz"), {name: fitted[name] for name in
                        ("decoder_FP64", "decoder_FP32", "bias_FP64", "bias_FP32", "lags")})
                    entry = {"status": "LOCKED_FIT_ONLY", "dataset": dataset, "space": space, "condition": condition,
                        "lags": list(lags), "objective": objective, "state_artifact": artifact, "fit_metadata": fitted["metadata"],
                        "fit_wall_seconds": time.perf_counter() - started, "FIT_input_source": context["input_source"],
                        "FIT_Bolt_cache": context["cached_Bolt_source"], "FIT_future_targets_used": objective == "FORECAST",
                        "TUNE_DEV_used_for_fit_or_selection": False, "oracle_used_for_fit_or_selection": False}
                    states[key] = fitted
                    report["decoder_fits_completed"] += 1
                except Exception as caught:
                    entry = failure_row(caught, dataset, space, condition, "FIT_D")
                    report["failed_paths"].append(entry)
                lock_path = output / (prefix + "_FIT_LOCK.json")
                write_new(lock_path, {**entry, "sealed_unix": time.time()})
                entry["fit_lock"] = record(lock_path)
                locks[key], parent["conditions"][condition] = entry["fit_lock"], {**entry, "splits": {}}
    expected = {(data, space, condition) for data in DATASETS for space in SPACES for condition in CONDITIONS}
    require(set(locks) == expected, "ALL_24_FIT_PATHS_MUST_RESOLVE_BEFORE_TUNE_DEV")
    registry = {"status": "ALL_24_RESOLVED_BEFORE_TUNE_DEV", "sealed_unix": time.time(),
        "conditions": [{"dataset": key[0], "space": key[1], "condition": key[2], "fit_lock": value} for key, value in locks.items()],
        "failed_fit_paths": len(locks) - len(states), "TUNE_DEV_NPZ_opened_before_seal": False}
    registry_path = output / "FIT_REGISTRY.json"
    write_new(registry_path, registry)
    report["fit_registry"], report["resolved_fit_lock_files"] = record(registry_path), len(locks)
    report["all_24_locks_before_TUNE_DEV_NPZ"] = True
    return states


def evaluate_all_conditions(report, states, load_context, evaluate_case, save_arrays, output, check):
    require(report.get("all_24_locks_before_TUNE_DEV_NPZ") is True and report.get("resolved_fit_lock_files") == 24,
        "ALL_24_FIT_LOCKS_REQUIRED_BEFORE_EVALUATION")
    bound(report["fit_registry"])
    for dataset, (_, _, development, _) in DATASETS.items():
        for space in SPACES:
            for split in ("fit", "tune", development):
                try: context, error = load_context(dataset, space, split), None
                except Exception as caught: context, error = None, caught
                for condition in CONDITIONS:
                    key = dataset, space, condition
                    entry = report["datasets"][dataset]["spaces"][space]["conditions"][condition]
                    if key not in states:
                        entry["splits"][split] = {"status": "SKIPPED_FAILED_FIT", "reason": "See immutable FIT lock"}
                        continue
                    try:
                        if error is not None: raise ValueError("EVALUATION_CONTEXT_UNAVAILABLE:" + str(error))
                        bound(entry["fit_lock"]); bound(entry["state_artifact"])
                        check({"stage": "CACHED_D_INFERENCE", "dataset": dataset, "space": space, "condition": condition, "split": split})
                        summary, arrays = evaluate_case(context, states[key])
                        summary.update(input_source=context["input_source"], cached_Bolt_source=context["cached_Bolt_source"],
                            state_source=entry["state_artifact"], fit_lock=entry["fit_lock"])
                        summary["artifact"] = save_arrays(output / ("_".join((*key, split)) + "_PREDICTION_DIAGNOSTIC.npz"), arrays)
                        entry["splits"][split] = summary
                        report["cached_D_predictions_completed"] += 1
                        report["joint_output_diagnostics_completed"] += int(summary["diagnostics"]["joint_floor"]["status"] == "COMPLETE")
                        report["EY_diagnostics_completed"] += int(summary["diagnostics"]["methods"].get("EY", {}).get("status") == "COMPLETE")
                        report["flat_diagnostics_completed"] += int(summary["diagnostics"]["methods"].get("FLAT", {}).get("status") == "COMPLETE")
                        for diagnostic_failure in summary["diagnostic_failures"]:
                            report["failed_paths"].append({**diagnostic_failure, "dataset": dataset, "space": space, "condition": condition, "split": split})
                    except Exception as caught:
                        failed = failure_row(caught, dataset, space, condition, "CACHED_PREDICTION_OR_DIAGNOSTIC", split)
                        entry["splits"][split] = failed; report["failed_paths"].append(failed)


def retained_output_traces(mapping, complete, traces):
    import numpy as np
    A, offset = mapping["A"], mapping["offset"]
    n, H, c = offset.shape
    selected = {name: np.full_like(offset, np.nan, dtype=np.float64) for name in traces}
    discarded = {name: np.zeros_like(offset, dtype=np.float64) for name in traces}
    patterns = {}
    for origin in range(n): patterns.setdefault(complete[origin].tobytes(), []).append(origin)
    for indices in patterns.values():
        rows = np.repeat(complete[indices[0]], c)
        if not rows.any(): continue
        U, singular, _ = np.linalg.svd(A[rows], full_matrices=False)
        keep = singular > singular[0] * 1e-15 if len(singular) and singular[0] > 0 else np.zeros(len(singular), bool)
        retained_U = U[:, keep]
        base = offset.reshape(n, -1)[np.ix_(indices, rows)]
        for name, trace in traces.items():
            component = trace.reshape(n, -1)[np.ix_(indices, rows)] - base
            projected = (component @ retained_U) @ retained_U.T
            selected[name].reshape(n, -1)[np.ix_(indices, rows)] = base + projected
            discarded[name].reshape(n, -1)[np.ix_(indices, rows)] = component - projected
    return selected, discarded


def evaluate_case(context, fitted):
    import numpy as np
    import causal_lag_decoder_v1 as math
    from decoder_metric_probe_v1 import masked_scores
    values, z, future = context["values"], context["z"], context["future"]
    target, observed, current = values["target"], values["observed"], values["raw_last"]
    complete = observed.all(axis=2)
    complete_mask = np.broadcast_to(complete[:, :, None], observed.shape)
    lags, d = tuple(int(value) for value in fitted["lags"]), fitted["decoder_FP32"]
    flat = np.ascontiguousarray(np.broadcast_to(z[:, -1:], future.shape))
    diagnostic_Y = np.ascontiguousarray(np.where(complete[:, :, None], target, current[:, None]), dtype=np.float32)
    arrays = {"target": target, "observed": observed, "currentC": current, "origins": values["origins"],
              "FIT_mean": values["mean"], "FIT_std": values["std"], "complete_horizons": complete,
              "latent_history": z, "actual_F_future": future, "flat_future": flat}
    def scores(prediction, mask):
        return masked_scores(prediction, target, mask, values["std"]) if mask.any() else {
            "status": "NO_OBSERVED_TARGETS", "normalized": None, "physical_per_channel_diagnostic": None}
    summary = {"status": "COMPLETE", "windows": len(z), "complete_target_positions": int(complete.sum()),
        "target_positions": int(complete.size), "EY_incomplete_C_horizons_filled_with_currentC": int((~complete).sum()),
        "EY_target_role": "Diagnosis only; incomplete-C future vectors replaced entirely by currentC before fixed E",
        "diagnostics": {"methods": {}}, "diagnostic_failures": []}
    predicted = math.predict(z, future, current, d, lags)
    arrays["ACTUAL_F_prediction_FP32"] = predicted
    actual = {"status": "COMPLETE", "scores_all_observed": scores(predicted, observed),
        "scores_complete_mask": scores(predicted, complete_mask)}
    summary["diagnostics"]["methods"]["ACTUAL_F"] = actual
    summary.update(scores_all_observed=actual["scores_all_observed"], scores_complete_mask=actual["scores_complete_mask"])
    traces, latents = {}, {"ACTUAL_F": future, "FLAT": flat}
    try:
        latents["EY"] = context["encode"](diagnostic_Y)
        arrays["EY_future"] = latents["EY"]
    except Exception as error:
        failed = {"status": "FAILED_PATH_PRESERVED", "stage": "EY_ENCODER_DIAGNOSTIC", "failure": {"type": type(error).__name__, "message": str(error)}}
        summary["diagnostics"]["methods"]["EY"] = failed; summary["diagnostic_failures"].append(failed)
    for method, latent in latents.items():
        try:
            if method != "ACTUAL_F":
                predicted = math.predict(z, latent, current, d, lags)
                arrays[method + "_prediction_FP32"] = predicted
                summary["diagnostics"]["methods"][method] = {"status": "COMPLETE", "scores_all_observed": scores(predicted, observed),
                    "scores_complete_mask": scores(predicted, complete_mask)}
            real = math.real_trace(z, latent, current, d, lags)
            traces[method] = real; arrays[method + "_real_trace_FP64"] = real
            summary["diagnostics"]["methods"][method]["real_trace_scores_complete_mask"] = scores(real, complete_mask)
        except Exception as error:
            failed = {"status": "FAILED_PATH_PRESERVED", "stage": method + "_REAL_TRACE_DIAGNOSTIC", "failure": {"type": type(error).__name__, "message": str(error)}}
            summary["diagnostic_failures"].append(failed)
            if method != "ACTUAL_F": summary["diagnostics"]["methods"][method] = failed
    probe_n = min(2, len(z))
    changed_values = {"target": target[:probe_n].copy(), "observed": observed[:probe_n].copy(), "raw_last": current[:probe_n].copy()}
    changed_values["target"][:] = np.float32(1e6)
    changed_values["observed"][:] = False
    before = math.predict(z[:probe_n], future[:probe_n], current[:probe_n], d, lags)
    after = math.predict(z[:probe_n], future[:probe_n], changed_values["raw_last"], d, lags)
    require(byte_equal(before, after), "FIXED_CACHED_D_FUTURE_LABEL_PERTURBATION_PARITY")
    summary["cached_future_label_perturbation"] = {"status": "PASS", "windows": probe_n,
        "target_and_observed_mask_perturbed": True, "prediction_byte_exact": True,
        "fit_state_held_fixed": True, "E_F_forward_replayed": False,
        "scope": "Only fixed cached past/future/current/D prediction; excludes FIT objective and EY diagnosis, which legitimately use targets"}
    summary["current_C_persistence"] = {"scores_all_observed": scores(np.broadcast_to(current[:, None], target.shape), observed),
        "scores_complete_mask": scores(np.broadcast_to(current[:, None], target.shape), complete_mask)}
    summary["RAW_BOLT"] = {"scores_all_observed": scores(values["raw_forecast"], observed),
        "scores_complete_mask": scores(values["raw_forecast"], complete_mask), "bound_inequality_applied": False}
    if complete.any():
        try:
            minimum = math.masked_joint_minimum(z, current, d, lags, target, observed)
            mapping = math.joint_real_map(z, current, d, lags, H=target.shape[1])
            retained_traces, discarded = retained_output_traces(mapping, complete, traces)
            arrays.update({"joint_minimum_prediction_FP64": minimum["prediction_FP64"],
                "joint_optimal_future_FP64": minimum["optimal_future_FP64"], "joint_A_FP64": mapping["A"], "joint_known_offset_FP64": mapping["offset"]})
            floor = float(minimum["minimum_normalized_MSE"])
            summary["diagnostics"]["joint_floor"] = {"status": "COMPLETE", "minimum_normalized_MSE": floor,
                "minimum_solution_scores_complete_mask": scores(minimum["prediction_FP64"], complete_mask),
                "metadata": minimum["metadata"], "all_HK_future_variables_retained": True,
                "complete_C_output_rows_only": True, "RAW_bound_inequality_applied": False,
                "all_real_latent_forecasts_satisfy_registered_bound_claimed": False,
                "scope": "Numeric retained joint affine space, real arithmetic, this complete-C mask; only normalized MSE minimum"}
            for method, trace in traces.items():
                entry = summary["diagnostics"]["methods"][method]
                real_mse = entry["real_trace_scores_complete_mask"]["normalized"]["mse"]
                fp32_mse = entry["scores_complete_mask"]["normalized"]["mse"]
                retained_scores = scores(retained_traces[method], complete_mask)
                retained_mse = retained_scores["normalized"]["mse"]
                tolerance = 1e-10 * max(1., abs(floor), abs(real_mse), abs(retained_mse))
                allowance = abs(real_mse - fp32_mse)
                require(floor <= retained_mse + tolerance, "JOINT_RETAINED_SPACE_MINIMUM_ABOVE_RETAINED_TRACE:" + method)
                entry.update(real_trace_minus_minimum_normalized_MSE=real_mse - floor,
                    measured_FP32_rounding_normalized_MSE_allowance=allowance, numerical_MSE_tolerance=tolerance,
                    real_trace_bound_inequality_pass=bool(floor <= real_mse + tolerance),
                    FP32_bound_with_measured_allowance_pass=bool(floor <= fp32_mse + allowance + tolerance),
                    retained_output_real_trace_scores_complete_mask=retained_scores,
                    retained_output_trace_bound_inequality_pass=True,
                    discarded_numeric_map_output_squared_norm=float(np.sum(discarded[method][complete]**2)),
                    discarded_numeric_map_output_max_abs=float(np.max(np.abs(discarded[method][complete]))))
                arrays[method + "_retained_joint_trace_FP64"] = retained_traces[method]
                arrays[method + "_discarded_joint_component_FP64"] = discarded[method]
        except Exception as error:
            failed = {"status": "FAILED_PATH_PRESERVED", "stage": "JOINT_OUTPUT_SPACE_DIAGNOSTIC", "failure": {"type": type(error).__name__, "message": str(error)}}
            summary["diagnostics"]["joint_floor"] = failed; summary["diagnostic_failures"].append(failed)
    else:
        summary["diagnostics"]["joint_floor"] = {"status": "SKIPPED_NO_COMPLETE_C_TARGETS", "reason": "No eligible output rows; no minimum claimed"}
    if summary["diagnostic_failures"]: summary["status"] = "COMPLETE_PREDICTION_WITH_DIAGNOSTIC_FAILURES"
    return summary, arrays


def execute(tag, science_path, audit_path, report_audit_path):
    output = fresh_output(tag)
    source, bindings = load_gates(science_path, audit_path, report_audit_path)
    sys.path[:0] = [str(HERE), str(OLD), str(ROOT)]
    import numpy as np
    from threadpoolctl import threadpool_limits
    import causal_lag_decoder_v1 as math
    import fixed_decoder_span_v1 as io
    from projection_math_v1 import ProjectionState
    from run_raw_subset_comparison_v2 import encode_numpy
    from run_fit_target_comparison_v1 import prior_file_records
    from prepare_inputs import install_protected_test_guard
    import continuation_runtime_v1 as runtime
    output.mkdir(parents=True, exist_ok=False)
    meter = runtime.Meter("causal_lag_" + tag, gpu=False)
    report = {"status": "RUNNING", **bindings, "datasets": {}, "failed_paths": [], "execution_events": [],
        "decoder_fits_completed": 0, "resolved_fit_lock_files": 0, "cached_D_predictions_completed": 0,
        "joint_output_diagnostics_completed": 0, "EY_diagnostics_completed": 0, "flat_diagnostics_completed": 0,
        "actual_F_cache_splits_reused": 0, "all_24_locks_before_TUNE_DEV_NPZ": False,
        "new_F_calls": 0, "new_backbone_training": False, "new_native_head_training": 0,
        "protected_TEST_accessed": False, "independent_holdout": False, "oracle_used_for_fit_or_selection": False,
        "TUNE_DEV_used_for_fit_or_selection": False, "strict_K_only_compression": False,
        "new_representation_learning_claimed": False, "whole_deployment_cost_measured": False,
        "cost_status": "NOT_MEASURED: cached CPU D inference and diagnostics; whole deployment is a separate stage",
        "individual_exact_human_method_approval_claimed": False,
        "frozen_F_parent_signatures": {"before": source["F_before"], "after": source["F_after"]},
        "new_F_object_or_state_hash_computed": False, "source_records": [], "source_snapshots": []}
    checked, contexts, encoders, failure = {}, {}, {}, None
    try:
        meter.protected_check()
        report["protected_guard_paths"] = install_protected_test_guard()
        protected = prior_file_records({}, *bindings.values(), read(bindings["parent_report_manifest"]["path"])["inputs"])
        for filename in (Path(__file__).name, "causal_lag_decoder_v1.py", "PLAN_CAUSAL_LAG_OBJECTIVE_KO_v1.md",
                "continuation_runtime_v1.py", "fixed_decoder_span_v1.py", "decoder_metric_probe_v1.py", "projection_math_v1.py",
                "run_raw_subset_comparison_v2.py", "raw_subset_projection_v2.py", "raw_subset_projection_v1.py", "run_fit_target_comparison_v1.py"):
            binding = record(HERE / filename); report["source_records"].append(binding)
        report["source_records"].append(record(OLD / "prepare_inputs.py"))
        snapshots = output / "sources"; snapshots.mkdir()
        for binding in report["source_records"]:
            destination = snapshots / Path(binding["path"]).name
            shutil.copyfile(binding["path"], destination)
            require(record(destination)["sha256"] == binding["sha256"], "EXACT_SOURCE_SNAPSHOT_REQUIRED")
            report["source_snapshots"].append(record(destination))
        report["plan"] = record(HERE / "PLAN_CAUSAL_LAG_OBJECTIVE_KO_v1.md")
        report["local_HEAD_at_start"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        report["local_git_status_at_start"] = subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
        def load_context(dataset, space, split):
            key = dataset, space, split
            if key in contexts: return contexts[key]
            require(split == "fit" or report["all_24_locks_before_TUNE_DEV_NPZ"], "NO_TUNE_DEV_ARRAY_IO_BEFORE_ALL_24_FIT_LOCKS")
            c, k, dev, dev_count = DATASETS[dataset]
            parent = source["datasets"][dataset]["spaces"][space]
            e_key = dataset, space
            if e_key not in encoders:
                e = io.checked_payload(parent["E_state_artifact"], meter, checked)
                if space == "RAW_SUBSET":
                    indices = e["selected_indices"].astype(np.int64).tolist()
                    require(len(indices) == k and len(set(indices)) == k and all(0 <= value < c for value in indices), "FIXED_SUBSET_INDICES")
                    projection = ProjectionState(e["center"], e["encoder"], np.zeros((k, c)), np.zeros(c), {})
                else:
                    indices = None
                    projection = ProjectionState.from_npz(io.checked_payload(parent["original_E_state"], meter, checked))
                    require(byte_equal(e["center"], projection.center) and byte_equal(e["encoder"], projection.encoder), "EXACT_ORIGINAL_MIXED_E")
                encoders[e_key] = (projection, indices)
            projection, indices = encoders[e_key]
            values = io.checked_payload(parent["inputs"][split], meter, checked)
            cached = io.checked_payload(parent["Bolt"][split]["artifact"], meter, checked)
            count = SPLIT_COUNTS.get(split, dev_count)
            require(split in ("fit", "tune", dev) and values["history"].dtype == np.float32 and values["history"].shape == (count, 512, c)
                and values["history_observed"].dtype == np.bool_ and values["history_observed"].shape == values["history"].shape,
                "REGISTERED_PAST_WINDOW_AND_TRUE_OBSERVED_CONTRACT")
            require(values["target"].dtype == np.float32 and values["target"].shape == (count, 48, c)
                and values["observed"].dtype == np.bool_ and values["observed"].shape == values["target"].shape
                and np.isfinite(values["target"][values["observed"]]).all() and byte_equal(values["raw_last"], values["history"][:, -1]),
                "REGISTERED_ACTUAL_TARGET_MASK_CURRENT_C")
            require(values["mean"].dtype == values["std"].dtype == np.float32 and values["mean"].shape == values["std"].shape == (c,)
                and np.isfinite(values["mean"]).all() and np.isfinite(values["std"]).all() and (values["std"] > 0).all(), "EXACT_FINITE_FIT_SCALER")
            require(values["origins"].dtype == np.int64 and values["origins"].shape == (count,)
                and byte_equal(values["origins"], cached["origins"]), "EXACT_CACHED_BOLT_ORIGINS")
            z = encode_numpy(projection, indices, values["history"])
            require(byte_equal(z, values["latent_history_BOLT_NUMPY"]) and byte_equal(z, cached["latent_history"]), "EXACT_NUMPY_E_F_HISTORY_LINEAGE")
            require(cached["latent_future"].dtype == np.float32 and cached["latent_future"].shape == (count, 48, k)
                and np.isfinite(cached["latent_future"]).all(), "ACTUAL_FINITE_FROZEN_K_STREAM_FUTURE_CACHE")
            context = {"values": values, "z": z, "future": cached["latent_future"],
                "encode": lambda x, p=projection, i=indices: encode_numpy(p, i, x),
                "input_source": parent["inputs"][split], "cached_Bolt_source": parent["Bolt"][split]["artifact"], "E_source": parent["E_state_artifact"]}
            contexts[key] = context; report["actual_F_cache_splits_reused"] += 1
            report["execution_events"].append({"event": "CONTEXT_OPENED", "dataset": dataset, "space": space, "split": split,
                "all_24_locks_already_sealed": report["all_24_locks_before_TUNE_DEV_NPZ"]})
            return context
        def fit_decoder(context, lags, objective):
            values = context["values"]
            if objective == "RECON":
                return math.fit_reconstruction(values["history"], values["history_observed"], context["z"], lags)
            return math.fit_forecast(context["z"], context["future"], values["raw_last"], values["target"], values["observed"], lags)
        with threadpool_limits(limits=2):
            states = fit_all_conditions(report, output, load_context, fit_decoder, io.save_npz, runtime.write_new, meter.check)
            report["execution_events"].append({"event": "ALL_24_FIT_LOCKS_SEALED", "registry": report["fit_registry"]})
            protected.extend(prior_file_records(source))
            evaluate_all_conditions(report, states, load_context, evaluate_case, io.save_npz, output, meter.check)
        for binding in protected + report["source_records"] + report["source_snapshots"] + list(checked.values()): bound(binding)
        bound(report["fit_registry"])
        for row in report["datasets"].values():
            for parent in row["spaces"].values():
                for entry in parent["conditions"].values():
                    bound(entry["fit_lock"])
                    if entry["status"] == "LOCKED_FIT_ONLY": bound(entry["state_artifact"])
        meter.protected_check()
        report.update(protected_prior_records=protected, input_records=list(checked.values()), source_cache_model_selection_SHA_unchanged=True)
        full = (report["decoder_fits_completed"] == 24 and report["cached_D_predictions_completed"] == 72
            and report["joint_output_diagnostics_completed"] == report["EY_diagnostics_completed"] == report["flat_diagnostics_completed"] == 72
            and report["actual_F_cache_splits_reused"] == 18 and not report["failed_paths"])
        report["status"] = "COMPLETE" if full else "PARTIAL_COMPLETED"
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        report.update(status="FAILED_PRESERVED", failure=failure, full_scope_complete_claimed=False)
    finally:
        report["resource_terminal"] = meter.close(report["status"], failure)
        runtime.write_new(output / "RESULT.json", report)
    print(json.dumps({"status": report["status"], "result": str(output / "RESULT.json"), "D_fits": report["decoder_fits_completed"],
        "predictions": report["cached_D_predictions_completed"], "joint_diagnostics": report["joint_output_diagnostics_completed"], "failure": failure}), flush=True)
    return 0 if report["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--science", default=str(HERE / "runs/raw_subset_comparison_v2/RESULT.json"))
    parser.add_argument("--audit", default=str(HERE / "audits/RAW_SUBSET_COMPARISON_AUDIT_v3.json"))
    parser.add_argument("--report-audit", default=str(HERE / "audits/RAW_SUBSET_REPORT_AUDIT_v1.json"))
    args = parser.parse_args()
    raise SystemExit(execute(args.tag, args.science, args.audit, args.report_audit))
