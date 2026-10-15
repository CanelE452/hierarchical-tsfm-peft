"""JSON-only integrated selected-shrinkage and freshly matched native report."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATASETS = {"robin": (17, 4, "val"), "beijing_aotizhongxin": (11, 2, "eval")}
PROJECTIONS = ("PCA_PINV", "FIT_TARGET_PINV")
SEEDS = ("92601", "92602")
BACKENDS = {"BOLT_NUMPY": ("BOLT",), "NATIVE_TORCH32": ("NATIVE_s92601", "NATIVE_s92602")}
SCALES = ("normalized", "physical_per_channel_diagnostic")
CANDIDATES = ("STATIC", "LOCAL", "RHO_0.001", "RHO_0.01", "RHO_0.1", "RHO_1", "RHO_10")
SELECTED_ROUTES = ("RAW_BOLT", "PCA_PINV_BOLT", "FIT_TARGET_PINV_BOLT", "PCA_SHRUNK_BOLT", "FIT_TARGET_SHRUNK_BOLT")
MATCHED_ROUTES = ("RAW_BOLT", "PCA_SHRUNK_BOLT", "FIT_TARGET_SHRUNK_BOLT", "PCA_SHRUNK_NATIVE_s92601", "PCA_SHRUNK_NATIVE_s92602",
                  "FIT_TARGET_SHRUNK_NATIVE_s92601", "FIT_TARGET_SHRUNK_NATIVE_s92602")
INPUT_NAMES = ("shrinkage", "shrinkage_audit", "space", "selected_cost", "selected_cost_audit", "matched", "matched_audit", "matched_cost", "matched_cost_audit",
               "repeat_matched_cost", "repeat_matched_cost_audit")
FIXED_SHA = {"shrinkage": "aa553512965bf9253fae1983424e2fcb4f2916cc95ec12e9ed671ae325dbd532",
    "shrinkage_audit": "145999ea8bc7068eba08f08039287a33097cdb4e9ec8dab002f4ebc7630fdda9",
    "space": "6093d1b2cd7b61c1549aa51ac234476404b13a24edc492a42764ccac0c67deff",
    "selected_cost_audit": "43e5986d250a707cdb95958e8887184f46794204f10e4d2767acdc7ab83424c6",
    "matched": "e13587c86a9f28b3dccc5bf445cb85cd855a8a20a0ce1d9caad7dd57b93ba77a",
    "matched_audit": "0eb57a67b2ee3c25a7a5309efbdb62e55f756fb0c9552bd251b50f2ea6505077",
    "matched_cost": "a67df4dacf9fdf354dd8e16cfeebb22017f8d1dcf3551e467b170fefa025457e",
    "matched_cost_audit": "cc2b9e432a354b62d362ede8f8fb681e4a9c5d4fbfd65587190c6dabb48bc31d",
    "repeat_matched_cost": "e1b0db88225100fa05b7d74178801dc0c4eb77c61e03c8f0c9b64fd91489eed8"}
FAILED_COST_SHA = "6940e169e2acaa3c27b08c1c80e1e8ae2ddcef944a100af331629165fff09ea9"
AUDIT_INVOCATION_IDENTITY_SHA = "07ee9e805073d2cf5deed6489b9555b78ddb80c69533269702bc5c0c5436500b"
REPEAT_PLAN_SHA = "73f0beb075295ad3a5be57d779b83dda4b55d60ac1f81a6d586c66f5477893d1"
MATCHED_COST_COUNTS = {"cost_cells_checked": 28, "recorded_timer_samples_checked": 196,
    "physical_backend_prechecks_checked": 8, "physical_first8_window_fits_checked": 64,
    "native_first8_prediction_checks": 8, "native_selection_locks_checked": 8,
    "old_Bolt_rho_lock_files_checked": 4, "native_and_E_state_hashes_checked": 12,
    "selected_forecast_file_hashes_checked": 60, "canonical_physical_bridges_checked": 2}


def require(condition, label):
    if not condition: raise ValueError(label)


def sha(path):
    with Path(path).open("rb") as handle: return hashlib.file_digest(handle, "sha256").hexdigest()


def record(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def read(path):
    require(Path(path).suffix.lower() == ".json", "JSON_METADATA_ONLY_INPUT")
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def bound(item, expected, label):
    path = resolve(item["path"])
    require(path.is_file() and resolve(expected["path"]) == path and item["sha256"] == expected["sha256"] == sha(path), "EXACT_BINDING:" + label)
    for value in (item, expected):
        if "bytes" in value: require(value["bytes"] == path.stat().st_size, "OPTIONAL_BOUND_BYTES:" + label)


def require_counts(data, expected, label):
    require(all(data.get("totals", {}).get(key) == value for key, value in expected.items()), "COMPLETE_INDEPENDENT_COUNTS:" + label)


def load_verified(paths, failed_cost_path):
    require(set(paths) == set(INPUT_NAMES), "ALL_ELEVEN_STAGE_METADATA_INPUTS_REQUIRED")
    values, bindings = {name: read(path) for name, path in paths.items()}, {name: record(path) for name, path in paths.items()}
    for name, value in values.items():
        require(value.get("status") == ("PASS" if name.endswith("audit") or name == "space" else "COMPLETE"), "COMPLETE_OR_PASS_REQUIRED:" + name)
        require(value.get("protected_TEST_accessed") is False, "NO_PROTECTED_TEST:" + name)
        if name in FIXED_SHA: require(bindings[name]["sha256"] == FIXED_SHA[name], "REGISTERED_ACTUAL_STAGE_SHA:" + name)
    shrink, audit, space, oldcost, oldpass, matched, nativepass, newcost, newpass = (values[name] for name in INPUT_NAMES[:9])
    repeatcost, repeatpass = values["repeat_matched_cost"], values["repeat_matched_cost_audit"]
    for item, expected, label in ((audit["diagnostic"], bindings["shrinkage"], "shrinkage PASS"),
        (space["shrinkage_diagnostic"], bindings["shrinkage"], "space source"), (space["shrinkage_audit"], bindings["shrinkage_audit"], "space accepted PASS"),
        (oldpass["cost"], bindings["selected_cost"], "F-only wholecost PASS"),
        (matched["shrinkage_diagnostic"], bindings["shrinkage"], "matched source"),
        (matched["shrinkage_audit"], bindings["shrinkage_audit"], "matched accepted PASS"),
        (nativepass["diagnostic"], bindings["matched"], "matched native PASS"),
        (newpass["cost"], bindings["matched_cost"], "new wholecost PASS"),
        (newcost["matched_native_diagnostic"], bindings["matched"], "new cost matched"),
        (newcost["matched_native_audit"], bindings["matched_audit"], "new cost matched PASS"),
        (newpass["matched_native_diagnostic"], bindings["matched"], "new cost PASS matched"),
        (newpass["matched_native_audit"], bindings["matched_audit"], "new cost PASS matched PASS"),
        (repeatpass["cost"], bindings["repeat_matched_cost"], "repeat wholecost PASS")):
        bound(item, expected, label)
    for value in (repeatcost, repeatpass):
        bound(value["matched_native_diagnostic"], bindings["matched"], "repeat same matched native")
        bound(value["matched_native_audit"], bindings["matched_audit"], "repeat same matched native PASS")
    for value in (audit, space, oldcost, oldpass, matched, nativepass, newcost, newpass, repeatcost, repeatpass):
        for key in ("producer", "fit_target_audit"): bound(value[key], shrink[key], "same original FIT " + key)
    for value in (oldcost, oldpass, nativepass, newcost, newpass, repeatcost, repeatpass):
        bound(value["shrinkage_diagnostic"], bindings["shrinkage"], "same shrinkage source")
        bound(value["shrinkage_audit"], bindings["shrinkage_audit"], "same shrinkage PASS")
    for name, code in (("shrinkage_audit", "audit_local_decoder_shrinkage_v1.py"), ("space", "audit_selected_shrinkage_output_space_v1.py"),
                       ("selected_cost_audit", "audit_selected_shrinkage_cost_v1.py"), ("matched_audit", "audit_matched_shrunk_native_v1.py"),
                       ("matched_cost_audit", "audit_matched_shrunk_native_cost_v1.py"),
                       ("repeat_matched_cost_audit", "audit_matched_shrunk_native_cost_v1.py")):
        item = values[name]["audit_source"]
        require(resolve(item["path"]).name == code and item["sha256"] == sha(resolve(item["path"])), "CANONICAL_AUDIT_SOURCE:" + name)
    require_counts(audit, {"positive_rho_backend_states_checked": 88, "window_fits_checked": 13504, "selection_locks_checked": 4,
        "FIT_TUNE_positive_rho_prediction_arrays_checked": 120, "selected_DEV_prediction_arrays_checked": 12, "score_field_checks": 6960}, "shrinkage")
    require_counts(space, {"selected_output_space_cells_checked": 24, "E_Y_diagnostic_predictions_checked": 24,
        "actual_predictions_checked": 36, "real_output_space_inequalities_checked": 60, "RAW_bound_inequalities_checked": 0,
        "score_field_checks": 864, "selected_D_origin_matrices_checked": 3264}, "selected D outputspace")
    require_counts(oldpass, {"cost_cells_checked": 20, "recorded_timer_samples_checked": 120,
        "physical_shrunk_prechecks_checked": 4, "physical_shrunk_first8_window_fits_checked": 32, "selected_rule_lock_files_checked": 4}, "selected F-only costs")
    require_counts(newpass, MATCHED_COST_COUNTS, "new matched whole costs")
    require_counts(repeatpass, MATCHED_COST_COUNTS, "repeat matched whole costs")
    require_counts(nativepass, {"native_fits_checked": 8, "training_updates_logged_checked": 4800,
        "checkpoint_states_checked": 40, "checkpoint_TUNE_prediction_arrays_checked": 40,
        "selected_FIT_TUNE_DEV_prediction_arrays_checked": 24, "source_backend_contexts_checked": 12,
        "preserved_control_prediction_arrays_checked": 48, "persistence_baseline_contexts_computed": 12,
        "unique_persistence_split_cells_computed": 6, "selection_locks_checked": 8,
        "resolved_registry_conditions_checked": 8, "future_label_perturbation_scope_checks": 8,
        "score_field_checks": 2064}, "new native and current-C persistence")
    require(shrink["future_labels_used_for_decoder_fit"] is False and shrink["TUNE_future_labels_used_for_normal_hyperparameter_selection"] is True
            and shrink["development_positive_rho_sweep_executed"] is False and shrink["native_specific_selection"] is False, "KNOWN_PAST_FIT_NORMAL_TUNE_SELECTION")
    require(space["minimum_claim_only_normalized_MSE_in_registered_real_D32_numeric_rowspace"] is True
            and space["strict_lower_bound_over_all_FP32_outputs_claimed"] is False and space["MAE_or_physical_MSE_minimum_claimed"] is False
            and space["future_labels_used_to_fit_or_modify_D"] is False, "KNOWN_FUTURE_DIAGNOSTIC_SCOPE")
    expected = {"native_training_fits": 8, "completed_training_updates": 4800, "checkpoint_states_saved": 40,
        "completed_fit_tune_predictions": 16, "completed_DEV_predictions": 8, "completed_selection_locks": 8,
        "completed_source_backend_cells": 12, "future_label_perturbation_checks": 8}
    require(all(matched[key] == value for key, value in expected.items()) and not matched["failed_paths"], "FULL_FRESH_MATCHED_NATIVE_COHORT")
    require(matched["rho_selection_repeated"] is matched["rho_selected_by_native"] is matched["new_encoder_decoder_fit"] is matched["new_backbone_training"] is False
            and matched["all_eight_condition_locks_saved_before_any_new_DEV_prediction"] is True
            and matched["all_conditions_resolved_registry_saved_before_any_DEV_input_load"] is True, "EIGHT_LOCKS_BEFORE_DEV_AND_ORIGINAL_RHO")
    for cost, routes, cells, samples in ((oldcost, SELECTED_ROUTES, 20, 120), (newcost, MATCHED_ROUTES, 28, 196), (repeatcost, MATCHED_ROUTES, 28, 196)):
        require(tuple(cost["routes"]) == routes and cost["completed_cost_cells"] == cells
                and cost["recorded_timer_samples"] == cost["completed_cell_timer_samples"] == samples and not cost["failed_paths"], "COMPLETE_DISTINCT_COST_COHORT")
        require(cost["F_before"] == cost["F_after"] and cost["new_training_updates"] == 0 and cost["local_ridge_and_implemented_diagnostics_inside_timer"] is True
                and cost["fit_input_binding_hashes_inside_timer"] is True and cost["local_coefficients_cached_inside_timed_pipeline"] is False,
                "ACTUAL_FROZEN_WHOLE_TIMER_WITH_RIDGE_AND_INPUT_HASHES")
    require(oldcost["native_heads_loaded_or_timed"] is False and newcost["native_deployment_cost_measured"] is repeatcost["native_deployment_cost_measured"] is True,
            "F_ONLY_VS_NEW_NATIVE_DEPLOYMENT_COST_SCOPE")
    require(repeatcost["F_before"] == newcost["F_before"]
            and [{key: item[key] for key in ("path", "sha256")} for item in repeatcost["source_snapshots"]]
            == [{key: item[key] for key in ("path", "sha256")} for item in newcost["source_snapshots"]]
            and resolve(bindings["repeat_matched_cost"]["path"]) != resolve(bindings["matched_cost"]["path"]),
            "SAME_FROZEN_F_SOURCE_SEPARATE_ACTUAL_COST_COHORTS")
    for value in (shrink, matched): require(value["independent_holdout"] is False, "EXPOSED_DEV_NOT_INDEPENDENT_HOLDOUT")
    for dataset in DATASETS:
        for projection in PROJECTIONS:
            rho_record = shrink["selection_locks"][dataset][projection]
            bound(rho_record, record(resolve(rho_record["path"])), "preserved Bolt TUNE rho lock")
            for value in (space, newcost, newpass, repeatcost, repeatpass):
                rho_map = value["selection_locks"] if value is space else value["old_Bolt_rho_selection_locks"]
                bound(rho_map[dataset][projection], rho_record, "same four Bolt rho locks")
            for seed in SEEDS:
                cp_record = matched["condition_selection_locks"][dataset][projection][seed]
                bound(cp_record, record(resolve(cp_record["path"])), "preserved native TUNE checkpoint lock")
                for value in (nativepass, newcost, newpass, repeatcost, repeatpass):
                    lock_map = value["condition_selection_locks"] if value is nativepass else value["selection_locks"]
                    bound(lock_map[dataset][projection][seed], cp_record, "same eight native CP locks")
    failed = read(failed_cost_path)
    require(sha(failed_cost_path) == FAILED_COST_SHA
            and failed["status"] == "PARTIAL_COMPLETED" and failed["completed_cost_cells"] == failed["recorded_timer_samples"] == 0
            and failed["protected_TEST_accessed"] is False, "PRESERVED_ZERO_SAMPLE_IO_SCHEMA_FAILURE_V1")
    values["failed_matched_cost_v1"], bindings["failed_matched_cost_v1"] = failed, record(failed_cost_path)
    for value in (newcost, newpass, repeatcost, repeatpass): bound(value["prior_failed_cost_attempt"], bindings["failed_matched_cost_v1"], "preserved prior IO failure")
    identity_record, repair_record = newcost["input_schema_failure_identity"], newcost["repair_addendum"]
    bound(identity_record, record(resolve(identity_record["path"])), "verified IO failure identity")
    bound(repair_record, record(resolve(repair_record["path"])), "fresh physical bridge binding addendum")
    identity = read(resolve(identity_record["path"]))
    require(identity["status"] == "CONFIRMED_PRESERVED_IMPLEMENTATION_IO_CONTRACT_FAILURE"
            and identity["actual_timer_samples"] == 0 and identity["method_or_training_failure_inferred"] is False,
            "CONFIRMED_PRESERVED_ZERO_MEASUREMENT_SCHEMA_FAILURE")
    for dataset in DATASETS:
        bridge = newcost["canonical_physical_bridge_bindings"][dataset]
        require(resolve(bridge["canonical_record"]["path"]) == resolve(bridge["original_alias"]["path"])
            and bridge["canonical_record"]["sha256"] == bridge["original_alias"]["sha256"],
            "AUDITED_SAME_PHYSICAL_BRIDGE_METADATA_ONLY")
        require(bridge["same_physical_file_path_and_SHA"] is bridge["complete_array_manifest_preserved"] is True
            and bridge["loader_file_SHA_member_shape_dtype_SHA_checks_relaxed"] is bridge["numeric_tolerance_changed"] is False,
            "STRICT_CANONICAL_PHYSICAL_BRIDGE_BINDING")
    values["failed_cost_identity"] = identity
    bindings["failed_cost_identity"], bindings["physical_bridge_repair_addendum"] = identity_record, repair_record
    for key in ("input_schema_failure_identity", "repair_addendum"):
        bound(repeatcost[key], newcost[key], "repeat same preserved physical repair " + key)
    return values, bindings


def metric_rows(base, scores, **extra):
    return [{**base, "metric_scale": scale, **scores[scale], **extra} for scale in SCALES]


def summary(values):
    require(len(values) > 0 and all(type(v) in (int, float) and math.isfinite(v) for v in values), "FINITE_NONEMPTY_METADATA_SUMMARY")
    return {"min": min(values), "median": statistics.median(values), "max": max(values), "mean": statistics.mean(values)}


def collect_tables(values):
    shrink, space, matched = (values[name] for name in ("shrinkage", "space", "matched"))
    out = {key: [] for key in ("SCORES", "RHO_CANDIDATES", "TUNE_SELECTION", "RHO_LOCKS", "SELECTED_D_BACKENDS", "SELECTED_D_WINDOWS",
        "SELECTED_PAST_RECONSTRUCTION", "CURRENT_C_PERSISTENCE", "CHECKPOINTS", "LEARNING_CURVE", "NATIVE_TRAINING", "NATIVE_LOCKS", "OUTPUT_DIAGNOSTIC",
        "OUTPUT_GEOMETRY", "SELECTED_F_ONLY_COST", "SELECTED_F_ONLY_COST_SAMPLES", "MATCHED_WHOLE_COST", "MATCHED_WHOLE_COST_SAMPLES",
        "REPEAT_MATCHED_WHOLE_COST", "REPEAT_MATCHED_WHOLE_COST_SAMPLES", "REGISTERED_ADOPTION", "COST_FAILURE_PROVENANCE")}
    for dataset, (c, k, dev) in DATASETS.items():
        for projection in PROJECTIONS:
            lock_record = shrink["selection_locks"][dataset][projection]; lock = read(resolve(lock_record["path"]))
            label, rho = lock["selected_candidate"], lock["selected_rho"]
            out["RHO_LOCKS"].append({"dataset": dataset, "projection": projection, "C": c, "K": k, "rho": rho,
                "selected_candidate": label, "lock_record": lock_record, "lock": lock, "DEV_used": False, "native_specific_selection": False})
            tune = shrink["datasets"][dataset]["splits"]["tune"]["routes"][projection]
            all_tune = {**tune["controls"], **tune["positive_rho_candidates"]}
            for candidate in CANDIDATES:
                score = all_tune[candidate]["methods"]["BOLT"]["scores_all_observed"]["normalized"]
                out["TUNE_SELECTION"].append({"dataset": dataset, "projection": projection, "candidate": candidate,
                    "selected": candidate == label, "normalized_MSE": score["mse"], "normalized_MAE": score["mae"],
                    "rho": all_tune[candidate].get("rho"), "selection_metric": "normalized_MSE", "MAE_used_for_selection": False})
            for split in ("fit", "tune", dev):
                cell = shrink["datasets"][dataset]["splits"][split]
                route = cell["routes"][projection]
                if projection == PROJECTIONS[0]:
                    for mask, suffix in (("all_original_observed", "all_observed"), ("all_C_complete", "complete_mask")):
                        out["SCORES"] += metric_rows({"dataset": dataset, "projection": "NA", "split": split, "model": "RAW_BOLT", "variant": "RAW", "mask": mask}, cell["RAW"]["scores_" + suffix])
                selected = route["positive_rho_candidates"][label]
                for candidate, condition in (("STATIC", route["controls"]["STATIC"]), ("LOCAL", route["controls"]["LOCAL"]), (label, selected)):
                    for method, score in condition["methods"].items():
                        for mask, suffix in (("all_original_observed", "all_observed"), ("all_C_complete", "complete_mask")):
                            out["SCORES"] += metric_rows({"dataset": dataset, "projection": projection, "split": split, "model": method,
                                "variant": candidate, "mask": mask}, score["scores_" + suffix],
                                native_head_frozen_intervention_not_reoptimized=method.startswith("NATIVE"))
                if split in ("fit", "tune"):
                    for candidate, condition in route["positive_rho_candidates"].items():
                        for method, score in condition["methods"].items():
                            for mask, suffix in (("all_original_observed", "all_observed"), ("all_C_complete", "complete_mask")):
                                out["RHO_CANDIDATES"] += metric_rows({"dataset": dataset, "projection": projection, "split": split,
                                    "candidate": candidate, "rho": condition["rho"], "model": method, "mask": mask}, score["scores_" + suffix])
                for backend in BACKENDS:
                    branch = selected["backends"][backend]; meta = branch["state_metadata"]
                    geo = space["datasets"][dataset]["splits"][split]["routes"][projection]["backends"][backend]
                    base = {"dataset": dataset, "projection": projection, "split": split, "backend": backend, "rho": rho}
                    valid_conditions = [v for condition, flags in zip(meta["ridge_matrix_condition"], meta["ridge_matrix_condition_valid"]) for v, flag in zip(condition, flags) if flag]
                    out["SELECTED_D_BACKENDS"].append({**base, "windows": meta["windows"], "C": c, "K": k,
                        "max_abs_D": summary(meta["max_abs_decoder_per_window"]), "operator_norm": summary(meta["decoder_operator_norm_per_window"]),
                        "Frobenius_norm": summary(meta["decoder_Frobenius_norm_per_window"]),
                        "valid_ridge_condition": summary(valid_conditions) if valid_conditions else None,
                        "zero_trace_prior_count": sum(sum(row) for row in meta["zero_trace_prior_slope_retained"]),
                        "no_observation_prior_count": sum(sum(row) for row in meta["no_observation_prior_with_current_intercept"]),
                        "state_record": branch["state_artifact"], "mask_scope": "true observed past only", "absolute_gain_bound_claimed": False})
                    out["SELECTED_PAST_RECONSTRUCTION"] += metric_rows(base, branch["past_reconstruction_scores_observed_only"],
                        future_forecast_task=False, reconstruction_is_in_sample_known_past=True)
                    for window in range(meta["windows"]):
                        out["SELECTED_D_WINDOWS"].append({**base, "window_index_zero_based": window,
                            "max_abs_D": meta["max_abs_decoder_per_window"][window], "operator_norm": meta["decoder_operator_norm_per_window"][window],
                            "Frobenius_norm": meta["decoder_Frobenius_norm_per_window"][window], "output_numeric_rank": geo["rank_D_per_origin"][window],
                            "observed_count_per_C": meta["true_observed_counts"][window], "lambda_summary": summary(meta["actual_lambda"][window]),
                            "covariance_trace_summary": summary(meta["local_covariance_trace"][window]), "origin_values_not_invented": True})
                    out["OUTPUT_GEOMETRY"].append({**base, "minimum_normalized_MSE": geo["minimum_normalized_MSE_real_numeric_rowspace"],
                        "numeric_relative_singular_cutoff": geo["numeric_rowspace_relative_singular_cutoff"],
                        "complete_target_positions": space["datasets"][dataset]["splits"][split]["complete_target_positions"],
                        "per_origin_rank": geo["rank_D_per_origin"], "per_origin_certificates": geo["certificates"], "RAW_bound_applied": False})
                    diagnostics = {"RAW_BOLT": geo["RAW_scores_same_complete_C"], "SELECTED_D_MINIMUM": geo["minimum_solution_scores_complete_C"],
                        "E_Y": geo["E_Y_diagnostic"]["scores_complete_C"], **{name: entry["scores_complete_C"] for name, entry in geo["actual_methods"].items()}}
                    if backend == "NATIVE_TORCH32":
                        for seed in SEEDS: diagnostics["MATCHED_NATIVE_s" + seed] = matched["datasets"][dataset]["routes"][projection]["native"][seed]["scores"][split]["scores_complete_mask"]
                    for model, scores in diagnostics.items():
                        minimum = geo["minimum_normalized_MSE_real_numeric_rowspace"]
                        out["OUTPUT_DIAGNOSTIC"] += metric_rows({**base, "model": model, "mask": "all_C_complete"}, scores,
                            minimum_only_normalized_MSE=model == "SELECTED_D_MINIMUM", MAE_is_minimum=False, physical_MSE_is_minimum=False,
                            actual_normalized_MSE_minus_known_real_minimum=scores["normalized"]["mse"] - minimum,
                            known_future_diagnostic_not_deployable=model in ("SELECTED_D_MINIMUM", "E_Y"), RAW_bound_applied=False,
                            new_matched_real_trace_inequality_verified_in_report=False)
                native_route = matched["datasets"][dataset]["routes"][projection]
                persistence = values["matched_audit"]["datasets"][dataset]["routes"][projection]["inputs"][split]["persistence_current_C_baseline"]
                for mask, suffix in (("all_original_observed", "all_observed"), ("all_C_complete", "complete_mask")):
                    out["CURRENT_C_PERSISTENCE"] += metric_rows({"dataset": dataset, "projection": projection, "split": split,
                        "model": "CURRENT_C_PERSISTENCE", "mask": mask}, persistence["scores_" + suffix],
                        unique_dataset_split_cells=6, repeated_projection_contexts=12, independent_statistical_draws_claimed=False,
                        Raw_Bolt_identical_claimed=False, whole_deployment_cost_measured=False)
                for seed in SEEDS:
                    for mask, suffix in (("all_original_observed", "all_observed"), ("all_C_complete", "complete_mask")):
                        out["SCORES"] += metric_rows({"dataset": dataset, "projection": projection, "split": split,
                            "model": "MATCHED_NATIVE_s" + seed, "variant": label, "mask": mask}, native_route["native"][seed]["scores"][split]["scores_" + suffix],
                            native_head_frozen_intervention_not_reoptimized=False)
            for seed in SEEDS:
                native = matched["datasets"][dataset]["routes"][projection]["native"][seed]; meta = native["metadata"]
                base = {"dataset": dataset, "projection": projection, "seed": seed, "rho": rho}
                checkpoints = native["checkpoints"]
                require([point["update"] for point in checkpoints] == [0, 100, 200, 400, 600] and len(native["learning_curve"]) == 600, "FULL_CP_AND_CURVE_COHORT")
                chosen = min(checkpoints, key=lambda point: (point["tune"]["mse"], point["update"]))
                require(chosen["update"] == meta["selected_update"] and meta["updates"] == 600
                        and meta["old_head_weights_reused_or_warm_started"] is False and meta["decoder_condition_selected_by_native"] is False,
                        "FRESH_HEAD_WITH_TUNE_ONLY_CP_AND_FIXED_RHO")
                for point in checkpoints: out["CHECKPOINTS"].append({**base, "update": point["update"], "selected": point["update"] == chosen["update"],
                    **point["tune"], "state_sha256": point["state_sha256"], "state_record": point["state_artifact"], "prediction_record": point["tune_prediction_artifact"]})
                for point in native["learning_curve"]: out["LEARNING_CURVE"].append({**base, **point, "loss_scope": "minibatch actual-C masked macro MSE, not full FIT loss"})
                out["NATIVE_TRAINING"].append({**base, "metadata": meta, "fresh_train_wall_seconds": meta["train_wall_seconds"], "parameters": meta["parameters"],
                    "selected_update": meta["selected_update"], "zero_init_seed_changes_batch_order_only": True})
                item = matched["condition_selection_locks"][dataset][projection][seed]
                out["NATIVE_LOCKS"].append({**base, "lock_record": item, "lock": read(resolve(item["path"])), "DEV_used_for_CP_selection": False})
        for name, table, expected_routes, repeats in (("selected_cost", "SELECTED_F_ONLY_COST", SELECTED_ROUTES, 6),
            ("matched_cost", "MATCHED_WHOLE_COST", MATCHED_ROUTES, 7), ("repeat_matched_cost", "REPEAT_MATCHED_WHOLE_COST", MATCHED_ROUTES, 7)):
            rows = values[name]["cost"][dataset]["rows"]
            require(len(rows) == 2 * len(expected_routes) and {(r["route"], r["batch"]) for r in rows} == {(route, b) for route in expected_routes for b in (1, 8)}, "FULL_DISTINCT_COST_ROWS")
            for row in rows:
                raw = next(r for r in rows if r["route"] == "RAW_BOLT" and r["batch"] == row["batch"])
                seconds = [(s["perf_end_ns"] - s["perf_start_ns"]) / 1e9 for s in row["samples"]]
                raw_seconds = [(s["perf_end_ns"] - s["perf_start_ns"]) / 1e9 for s in raw["samples"]]
                ratio = [a / b for a, b in zip(seconds, raw_seconds)]
                require(len(seconds) == repeats and row["seconds"] == seconds and row["median_seconds"] == statistics.median(seconds)
                        and row["paired_RAW_ratios"] == ratio and row["paired_RAW_median_ratio"] == statistics.median(ratio), "SAVED_TIMER_SOURCE_ARITHMETIC")
                out[table].append({"dataset": dataset, "route": row["route"], "batch": row["batch"], "repeats": repeats,
                    "median_seconds": statistics.median(seconds), "median_seconds_per_window": statistics.median(seconds) / row["batch"],
                    "ratio_of_RAW_medians": statistics.median(seconds) / statistics.median(raw_seconds), "median_paired_RAW_ratio": statistics.median(ratio),
                    "seconds": seconds, "origins": row["origins"], "shared_process_memory": row["memory"], "includes": row["whole_includes"], "excludes": row["whole_excludes"],
                    "selected_metadata": row["selected_metadata"], "reference": row["reference"], "timed_outputs": row["timed_outputs"], "cohort": name})
                for index, sample in enumerate(row["samples"]): out[table + "_SAMPLES"].append({"dataset": dataset, "route": row["route"], "batch": row["batch"],
                    "repeat": sample["repeat"], "seconds": seconds[index], "paired_RAW_ratio": ratio[index], "sample": sample})
        for route in MATCHED_ROUTES[1:]:
            projection = "PCA_PINV" if route.startswith("PCA") else "FIT_TARGET_PINV"
            if "_NATIVE_s" in route:
                seed = route.rsplit("s", 1)[1]; score = matched["datasets"][dataset]["routes"][projection]["native"][seed]["scores"]["tune"]["scores_all_observed"]["normalized"]
            else:
                label = read(resolve(shrink["selection_locks"][dataset][projection]["path"]))["selected_candidate"]
                score = shrink["datasets"][dataset]["splits"]["tune"]["routes"][projection]["positive_rho_candidates"][label]["methods"]["BOLT"]["scores_all_observed"]["normalized"]
            raw = shrink["datasets"][dataset]["splits"]["tune"]["RAW"]["scores_all_observed"]["normalized"]
            for table in ("MATCHED_WHOLE_COST", "REPEAT_MATCHED_WHOLE_COST"):
                timing = next(r for r in out[table] if r["dataset"] == dataset and r["route"] == route and r["batch"] == 8)
                mse, mae, latency = score["mse"] / raw["mse"], score["mae"] / raw["mae"], timing["ratio_of_RAW_medians"]
                out["REGISTERED_ADOPTION"].append({"dataset": dataset, "route": route, "cost_cohort": table,
                    "TUNE_MSE_ratio_to_RAW": mse, "TUNE_MAE_ratio_to_RAW": mae, "B8_ratio_of_RAW_medians": latency,
                    "all_conditions_pass": mse <= 1.01 and mae <= 1.01 and latency <= .9,
                    "no_new_rho_seed_checkpoint_selection_from_cost_or_DEV": True})
    require(len(out["SCORES"]) == 552 and len(out["RHO_CANDIDATES"]) == 480 and len(out["TUNE_SELECTION"]) == 28
            and len(out["SELECTED_D_WINDOWS"]) == 3264 and len(out["CURRENT_C_PERSISTENCE"]) == 48
            and len(out["CHECKPOINTS"]) == 40 and len(out["LEARNING_CURVE"]) == 4800
            and len(out["SELECTED_F_ONLY_COST"]) == 20 and len(out["MATCHED_WHOLE_COST"]) == len(out["REPEAT_MATCHED_WHOLE_COST"]) == 28,
            "COMPLETE_TABLE_COHORT_COUNTS")
    failed = values["failed_matched_cost_v1"]
    out["COST_FAILURE_PROVENANCE"].append({"stage": "matched_whole_cost_v1", "status": failed["status"],
        "completed_cells": failed["completed_cost_cells"], "recorded_samples": failed["recorded_timer_samples"],
        "failed_paths": failed["failed_paths"], "source_snapshots": failed["source_snapshots"], "resource_terminal": failed["resource_terminal"],
        "performance_failure_claimed": False, "same_encoder_decoder_rho_checkpoints_and_tolerances_preserved": True,
        "repair_scope": "fresh v2 canonical physical-bridge array metadata binding; prior v1 remains immutable"})
    return out


def choose(rows, **where):
    found = [row for row in rows if all(row[key] == value for key, value in where.items())]
    require(len(found) == 1, "UNIQUE_PLOT_OR_TEXT_SCALAR:" + repr(where))
    return found[0]


def figures(tables, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    folder = Path(output) / "figures"; folder.mkdir(exist_ok=False)
    saved = []
    colors = ("#666666", "#0072B2", "#56B4E9", "#D55E00", "#E69F00", "#009E73", "#CC79A7")
    def save(fig, name):
        for suffix in ("png", "svg"):
            path = folder / (name + "." + suffix)
            fig.savefig(path, dpi=165 if suffix == "png" else None, bbox_inches="tight"); saved.append(path)
        plt.close(fig)
    def bars(ax, values, labels, palette=colors):
        artists = ax.bar(range(len(values)), values, color=palette[:len(values)])
        ax.bar_label(artists, labels=[f"{value:.4g}" for value in values], padding=3, fontsize=8)
        ax.set_xticks(range(len(values)), labels, rotation=18, ha="right")
        ax.grid(axis="y", alpha=.2); ax.set_ylim(0, max(values) * 1.25 if max(values) else 1)
        ax.set_ylabel("Normalized channel-macro MSE")
    def actual(dataset, projection, split, model, variant):
        return choose(tables["SCORES"], dataset=dataset, projection=projection, split=split, model=model,
            variant=variant, mask="all_original_observed", metric_scale="normalized")["mse"]
    def selected(dataset, projection):
        return choose(tables["RHO_LOCKS"], dataset=dataset, projection=projection)["selected_candidate"]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8), layout="constrained")
    for di, dataset in enumerate(DATASETS):
        for pi, projection in enumerate(PROJECTIONS):
            ax = axes[di, pi]
            rows = [choose(tables["TUNE_SELECTION"], dataset=dataset, projection=projection, candidate=label) for label in CANDIDATES]
            ax.plot(range(7), [row["normalized_MSE"] for row in rows], marker="o", color="#0072B2")
            winner = next(index for index, row in enumerate(rows) if row["selected"])
            ax.scatter([winner], [rows[winner]["normalized_MSE"]], color="#D55E00", s=90, label="Locked by Bolt TUNE")
            ax.set_xticks(range(7), ("Static", "Unregularized", ".001", ".01", ".1", "1", "10"), rotation=15)
            ax.set_title(dataset + " / " + projection); ax.set_ylabel("Full-observed normalized MSE"); ax.set_xlabel("Predeclared candidate / positive rho")
            ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle("Four Bolt-only TUNE locks; all seven candidates retained, no DEV/native selection", fontsize=12)
    save(fig, "01_TUNE_rho_selection")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), layout="constrained")
    for di, (dataset, (_, _, dev)) in enumerate(DATASETS.items()):
        for si, split in enumerate(("tune", dev)):
            values = [actual(dataset, "NA", split, "RAW_BOLT", "RAW")]
            labels = ["Raw"]
            for projection, short in (("PCA_PINV", "PCA"), ("FIT_TARGET_PINV", "Target")):
                for variant, name in (("STATIC", "static"), ("LOCAL", "unreg"), (selected(dataset, projection), "selected")):
                    values.append(actual(dataset, projection, split, "BOLT", variant)); labels.append(short + "/" + name)
            bars(axes[di, si], values, labels)
            axes[di, si].set_title(dataset + " / " + split.upper())
    fig.suptitle("Actual Bolt forecasts: static, unregularized and selected past-fitted D", fontsize=12)
    save(fig, "02_actual_Bolt_decoder_comparison")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), layout="constrained")
    for di, (dataset, (_, _, dev)) in enumerate(DATASETS.items()):
        for pi, projection in enumerate(PROJECTIONS):
            ax = axes[di, pi]; label = selected(dataset, projection)
            persistence = choose(tables["CURRENT_C_PERSISTENCE"], dataset=dataset, projection=projection, split=dev,
                mask="all_original_observed", metric_scale="normalized")["mse"]
            values = [actual(dataset, "NA", dev, "RAW_BOLT", "RAW"), persistence, actual(dataset, projection, dev, "BOLT", label)]
            labels = ["Raw", "Current C", "Bolt"]
            for seed in SEEDS:
                values += [actual(dataset, projection, dev, "NATIVE_s" + seed, label), actual(dataset, projection, dev, "MATCHED_NATIVE_s" + seed, label)]
                labels += ["Frozen/" + seed, "Matched/" + seed]
            bars(ax, values, labels)
            ax.set_title(dataset + " / " + projection + " / exposed DEV")
    fig.suptitle("Old head D-intervention versus eight freshly matched heads; limited affine native family", fontsize=12)
    save(fig, "03_frozen_and_matched_native")
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.5), layout="constrained")
    for di, (dataset, (_, _, dev)) in enumerate(DATASETS.items()):
        for pi, projection in enumerate(PROJECTIONS):
            ax = axes[di, pi]
            query = (("BOLT_NUMPY", "SELECTED_D_MINIMUM"), ("NATIVE_TORCH32", "SELECTED_D_MINIMUM"),
                ("BOLT_NUMPY", "E_Y"), ("BOLT_NUMPY", "BOLT"), ("NATIVE_TORCH32", "MATCHED_NATIVE_s92601"), ("NATIVE_TORCH32", "MATCHED_NATIVE_s92602"))
            points = [choose(tables["OUTPUT_DIAGNOSTIC"], dataset=dataset, projection=projection, split=dev,
                backend=backend, model=model, metric_scale="normalized")["mse"] for backend, model in query]
            bars(ax, points, ("D/F floor", "D/N floor", "E(Y)/F", "Actual Bolt", "Matched/01", "Matched/02"))
            raw = choose(tables["OUTPUT_DIAGNOSTIC"], dataset=dataset, projection=projection, split=dev, backend="BOLT_NUMPY", model="RAW_BOLT", metric_scale="normalized")["mse"]
            ax.axhline(raw, linestyle="--", color="#666666", label="Raw same complete-C mask"); ax.legend(fontsize=7)
            ax.set_title(dataset + " / " + projection + " / complete-C DEV")
    fig.suptitle("Known-future diagnostic space versus actual forecasts; E(Y) is not the minimum", fontsize=12)
    save(fig, "04_selected_output_space_and_forecast_gap")
    fig, axes = plt.subplots(2, 3, figsize=(21, 9.5), layout="constrained")
    for di, dataset in enumerate(DATASETS):
        for ci, (table, title, routes) in enumerate((("SELECTED_F_ONLY_COST", "Earlier F-only cohort, six repeats", SELECTED_ROUTES),
            ("MATCHED_WHOLE_COST", "Primary matched cohort, seven repeats", MATCHED_ROUTES),
            ("REPEAT_MATCHED_WHOLE_COST", "Separate repeated matched cohort, seven repeats", MATCHED_ROUTES))):
            ax = axes[di, ci]
            rows = [choose(tables[table], dataset=dataset, route=route, batch=8) for route in routes]
            values = [row["median_seconds_per_window"] * 1000 for row in rows]
            artists = ax.bar(range(len(rows)), values, color=colors[:len(rows)], alpha=.8)
            ax.bar_label(artists, labels=[f"{value:.3f}\nRmed {row['ratio_of_RAW_medians']:.3f}\nRpaired {row['median_paired_RAW_ratio']:.3f}"
                for value, row in zip(values, rows)], padding=3, fontsize=7)
            for index, row in enumerate(rows):
                n = len(row["seconds"])
                ax.scatter([index + (.3 * j / max(1, n - 1) - .15) for j in range(n)], [t * 1000 / 8 for t in row["seconds"]], color="black", s=10)
            ax.set_xticks(range(len(rows)), [row["route"].replace("FIT_TARGET", "T").replace("PCA", "P").replace("SHRUNK", "rho").replace("_NATIVE_s", "/N").replace("_BOLT", "/F") for row in rows], rotation=25, ha="right")
            ax.set_ylim(0, max(max(row["seconds"]) * 1000 / 8 for row in rows) * 1.42)
            ax.set_ylabel("Whole ms/window, B8"); ax.set_title(dataset + " / " + title); ax.grid(axis="y", alpha=.2)
    fig.suptitle("Three preserved cohorts: absolute Raw times and both relative statistics; no favorable-run selection", fontsize=12)
    save(fig, "05_separate_actual_whole_cost_cohorts")
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8), layout="constrained")
    for di, dataset in enumerate(DATASETS):
        for pi, projection in enumerate(PROJECTIONS):
            ax = axes[di, pi]
            for seed, color in zip(SEEDS, ("#0072B2", "#D55E00")):
                points = [row for row in tables["CHECKPOINTS"] if row["dataset"] == dataset and row["projection"] == projection and row["seed"] == seed]
                ax.plot([row["update"] for row in points], [row["mse"] for row in points], marker="o", color=color, label="Seed " + seed)
                chosen = next(row for row in points if row["selected"])
                ax.scatter([chosen["update"]], [chosen["mse"]], marker="*", s=150, color=color)
            ax.set_title(dataset + " / " + projection); ax.set_xlabel("Head updates; stars are TUNE-selected CP")
            ax.set_ylabel("Full TUNE normalized channel-macro MSE"); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle("Forty checkpoint evaluations, eight locks before DEV; all 4,800 minibatch records in CSV", fontsize=12)
    save(fig, "06_native_TUNE_checkpoints")
    return saved


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows: writer.writerow({key: json.dumps(value, ensure_ascii=False, separators=(",", ":")) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def rel(path, folder):
    return Path(os.path.relpath(Path(path).resolve(), Path(folder).resolve())).as_posix()


def report_text(values, tables, linked, output):
    lines = ["# 선택한 과거 디코더와 재학습한 작은 native head: 무엇이 남았는가", "",
        "실험의 답은 실행 가능한 작은 K 디코더와 미래 출력 공간의 여지가 남아 있다는 것이다. 이것과 실제 잠재 미래 예측의 성공은 다르다. 아래는 TUNE로 선택한 shrinkage, 고정 head에 D만 바꾼 대조, 새 D에 맞춰 실제 학습한 head와 세 비용 cohort를 연결한 기록이다.", "",
        "## 먼저 확인한 답과 원수치", ""]
    for dataset, (_, _, dev) in DATASETS.items():
        for projection in PROJECTIONS:
            label = choose(tables["RHO_LOCKS"], dataset=dataset, projection=projection)["selected_candidate"]
            for split in ("tune", dev):
                def mse(model, variant=label, space=projection):
                    return choose(tables["SCORES"], dataset=dataset, projection=space, split=split, model=model, variant=variant,
                        mask="all_original_observed", metric_scale="normalized")["mse"]
                lines.append(f"[확인] {dataset}/{projection}/{split.upper()} 정규화 MSE: Raw {mse('RAW_BOLT','RAW','NA'):.9f}; static Bolt {mse('BOLT','STATIC'):.9f}; unregularized Bolt {mse('BOLT','LOCAL'):.9f}; selected Bolt {mse('BOLT'):.9f}; old frozen native01/02 {mse('NATIVE_s92601'):.9f}/{mse('NATIVE_s92602'):.9f}; freshly matched native01/02 {mse('MATCHED_NATIVE_s92601'):.9f}/{mse('MATCHED_NATIVE_s92602'):.9f}.")
                persistence = choose(tables["CURRENT_C_PERSISTENCE"], dataset=dataset, projection=projection, split=split,
                    mask="all_original_observed", metric_scale="normalized")["mse"]
                lines.append(f"[확인] 같은 mask의 current-C persistence MSE는 {persistence:.9f}, fresh native01/02의 persistence 대비 비율은 {mse('MATCHED_NATIVE_s92601')/persistence:.6f}/{mse('MATCHED_NATIVE_s92602')/persistence:.6f}다. Raw와 persistence는 서로 다른 기준이다.")
    lines += ["", "[판단] 선택된 ridge는 이 기록의 Bolt TUNE 목적을 안정화하는 대조다. ρ를 Bolt TUNE로 고른 결과이므로 native 전용으로 최적화한 조건이 아니다. gain·출력 방향·과거 적합 오차가 함께 달라진다. 개선을 gain 하나의 인과 원인으로 확정하지 않는다. 새 head가 persistence를 개선한 결과는 과거 표현에서 예측 가능한 정보가 일부 사용됐다는 제한된 근거다. 따라서 표현이 아무 쓸모없거나 native가 전혀 학습하지 못했다고 말하지 않는다. Raw보다 나쁜 결과와 이 학습 효과를 함께 보고한다. 같은 actual-Y 목적과 D를 사용했지만 이 제한된 affine512→48 head·600 update 조건에서 전 문제가 해결됐다고 볼 수 없다. 고정 Bolt가 이 head군보다 나은 사례는 Bolt의 필요성이나 모든 다른 predictor보다 우월함을 증명하지 않는다.", "",
        "## 과거 정보·선정·미래 진단의 계약", "",
        "E는 원 PCA/FIT-target 공간으로 고정했다. FIT-target E는 이전 단계에서 실제 FIT 미래 innovation의 감독을 사용했고 배포 E 입력은 과거만 본다. origin별 D는 알려진 원 C 과거512와 true observation mask를 쓰는 centered ridge이며 λ=ρ·trace(G)/K다. 미래 target으로 D를 fitting하지 않았지만 TUNE 미래 정답은 정상 ρ 선택과 native checkpoint 선택에 사용했다. 모든 네 Bolt ρ lock, 새 native 여덟 lock과 전체 registry를 보존했으며 DEV에 앞서 잠갔다. DEV는 이미 노출된 탐색적 구간이고 독립 holdout 일반화 검증이 아니다.", "",
        "decoder 정보는 K future streams + current C + 원 C의 known past에서 계산한 per-origin K×C coefficients다. strict K-only 압축이나 static D와 동일한 decoder-side information 비교가 아니다. Bolt NumPy와 native Torch32에는 같은 ρ·λ 규칙을 썼지만 backend별 표현과 D·λ 바이트가 같다는 주장은 하지 않는다. static/local 실패, 모든 양수ρ의 FIT/TUNE 결과, native 두 seed의 나쁜 결과를 CSV에 남겼다. current-C persistence는6개 같은 split을 두 E context에 반복 기재한12contexts이고12독립 draw가 아니다. 그 baseline의 whole deployment 비용은 측정하지 않았다. 검사 통과를 예측 개선으로 바꾸지 않는다.", "",
        "![All seven TUNE candidates](figures/01_TUNE_rho_selection.png)", "",
        "![Actual Bolt comparisons](figures/02_actual_Bolt_decoder_comparison.png)", "",
        "## 디코더 조건을 맞춘 native 재학습", "",
        "새 head8개는 원 selected native D[n]과 cached 과거 E를 고정하여 actual-C 관측 macro 정규화 MSE로 600 update씩 학습했다. 이전 static head의 weights를 재사용하지 않았다. 0/100/200/400/600 update의 full TUNE score40개에서 정확히 같은 값이면 이른 checkpoint를 선택했다. seed는 초기 zero weights를 바꾸는 대신 minibatch 순서를 바꾼다. 4,800개 minibatch loss·gradient 기록과 training cost·parameter 수·lock을 보존한다. minibatch loss를 full FIT loss라고 하지 않는다. 원 고정 head의 D 개입과 새 matched 학습은 구분한다. 더 큰/다른 native나 native 전용ρ·architecture·training budget 탐색은 하지 않았다.", "",
        "![Frozen versus matched native](figures/03_frozen_and_matched_native.png)", "",
        "![TUNE checkpoint selection](figures/06_native_TUNE_checkpoints.png)", "",
        "## 작은 K 출력 공간과 실제 잠재 예측 사이", "",
        "선정된 D32[n]을 실수로 해석한 origin별 rowspace에 실제 미래 Y−currentC를 투영한 최소값은 미래를 아는 진단이다. 같은 complete-C mask에서만 Raw·E(Y)·actual Bolt·old/fresh native와 비교한다. 이 수학적 최소는 정규화 MSE에 한정하며 MAE·물리 단위 점수는 최소해의 값일 뿐 각각의 최소값이 아니다. E(Y)는 strict minimum이 아니다. Raw에 공간 부등식을 적용하지 않는다. 이전 static/global rank-K floor를 현재 D[n]의 하한으로 사용하지 않는다. FP32 전체 출력의 strict lower bound나 과거 정보의 충분성을 증명한 것으로 말하지 않는다.", ""]
    for dataset, (_, _, dev) in DATASETS.items():
        for projection in PROJECTIONS:
            def val(backend, model): return choose(tables["OUTPUT_DIAGNOSTIC"], dataset=dataset, projection=projection, split=dev, backend=backend, model=model, metric_scale="normalized")["mse"]
            lines.append(f"[확인] {dataset}/{projection}/complete-C {dev.upper()}: D/F real minimum {val('BOLT_NUMPY','SELECTED_D_MINIMUM'):.9f}, E(Y)/F {val('BOLT_NUMPY','E_Y'):.9f}, actual Bolt {val('BOLT_NUMPY','BOLT'):.9f}; D/native real minimum {val('NATIVE_TORCH32','SELECTED_D_MINIMUM'):.9f}, fresh native01/02 {val('NATIVE_TORCH32','MATCHED_NATIVE_s92601'):.9f}/{val('NATIVE_TORCH32','MATCHED_NATIVE_s92602'):.9f}.")
    lines += ["", "[판단] 낮은 공간 최소값과 큰 actual error가 함께 있으면 실제 잠재 미래 예측 및 D와의 결합에 문제가 남는다. 과거 reconstruction이 낮아도 미래 forecast 적합성을 보장하지 않는다. 이 증거는 입력 표현·future predictor·decoder를 분리해 볼 근거이며, 대조학습·동적 그래프가 꼭 필요하다거나 표현학습 자체가 불가능함을 확정하는 근거가 아니다. 새 matched score와 기존 real minimum의 차이는 같은 mask의 수치 차이로 보고하며 보고서가 새 real-trace 부등식을 추가 검산했다고 하지 않는다.", "",
        "![Known output space and actual gap](figures/04_selected_output_space_and_forecast_gap.png)", "",
        "## 세 실제 비용 cohort의 범위", "",
        "앞선 F-only cohort는5경로×2데이터×B1/B8=20셀·각6회=120표본이고 native head를 측정하지 않았다. 첫 matched cohort와 별도 process의 동일 프로토콜 반복은 각각7경로로28셀·각7회=196표본이며 새 selected head도 실제 실행한다. 세 cohort의 Raw 절대 시간·각 cohort Raw 중앙값의 비율·짝지은 비율을 모두 보존한다. 빠른 cohort만 고르거나 첫 성공 결과를 반복으로 대체하지 않는다. 서로 다른 시행의 절대 latency를 이어 가속 효과라고 주장하지 않는다. 첫 F-only cohort는 최초5회 rotation만 균형이고 여섯 번째는 추가 permutation이다. 두 matched cohort의 각7회 rotation은 완전히 위치 균형이다. 두 process의14 timing sample을 독립 데이터14개 또는 새로운 방법14개로 세지 않는다.", "",
        "타이머는 physical causal fill 전부터 FIT normalize → 실제 backend E → 매 callback known-past shrinkage D fitting과 구현된 진단·fit input binding hashes → frozen F 또는 selected head → D/currentC anchor → materialized physical CPU return과 최종 synchronization까지 포함한다. D 또는 native seed의 D를 cache해 타이머 밖으로 빼지 않았다. source/file/state metadata hashes generated outside timed callback·load·참조·warmup·저장·guard·memory snapshot은 측정 밖이다. 이 구현의 진단/hash overhead를 제거한 production 최소 비용이 아니다. shared resident F/head/E가 있는 process RSS/peak VRAM이며 방법별 독점 메모리 감소를 의미하지 않는다. 다른 CPU 활동이 전혀 없었다는 보장은 없고6/7회 같은 origin 반복은 독립 성능표본이 아니다.", "",
        "[확인] 새 matched cost v1은 physical_bridge alias에 array metadata가 빠진 I/O 계약 오류로 두 데이터 모두 KeyError 'arrays'를 보존했고0셀·0측정이었다. 이 실패를 모델 성능 실패로 세지 않는다. 원 v1 source/result/resource receipt를 유지한 채 fresh v2에서 정확한 canonical bridge의 array metadata를 다시 연결했다. encoder·decoder·ρ·head CP·공차·timer helper를 변경하거나 학습을 재실행한 repair가 아니다. 보고서에는 실제 v2 COMPLETE와 독립 PASS 이후의 비용만 확정 표로 넣고 v1 실패는 별도 provenance CSV/링크로 남긴다.", "",
        "[확인] 첫 비용 감사 v1의 실패는 상대 literal JSON 경로를 절대 경로 필수 helper에 넘긴 호출 오류였다. 실행된 auditor source·체크섬·예측·공차를 바꾸지 않고 동일 파일의 절대 경로로 fresh audit v2가 PASS했다. 이 호출 복구는 whole cost 재측정이나 모델 학습이 아니며 원 실패 감사와 identity를 보존한다. 별도 비용 반복은 앞선 Raw 시간 차이를 확인하려는 관측 검증이고, 원인 해결이나 빠른 시행 선택이 아니다.", "",
        "![Separate actual whole costs](figures/05_separate_actual_whole_cost_cohorts.png)", "", "```text", "Dataset / New route / B8    whole ms/window    ratio of Raw medians    paired median ratio"]
    for table in ("SELECTED_F_ONLY_COST", "MATCHED_WHOLE_COST", "REPEAT_MATCHED_WHOLE_COST"):
        lines.append(table)
        for row in tables[table]:
            if row["batch"] == 8: lines.append(f"{row['dataset']}/{row['route']} {row['median_seconds_per_window']*1000:.6f} {row['ratio_of_RAW_medians']:.6f} {row['median_paired_RAW_ratio']:.6f}")
    lines += ["```", ""]
    for table in ("MATCHED_WHOLE_COST", "REPEAT_MATCHED_WHOLE_COST"):
        checks = [row for row in tables["REGISTERED_ADOPTION"] if row["cost_cohort"] == table]
        passes = sum(row["all_conditions_pass"] for row in checks)
        lines.append(f"{table}: 동일 Raw TUNE MSE·MAE≤1.01배와 해당 cohort B8 ratio of medians≤.90을 함께 요구한 검산은 {passes}/{len(checks)}개다.")
    lines += ["paired median ratio는 다른 기술통계이고 이 기준에 혼용하지 않는다. 두 측정을 새 방법24개로 세거나 기준 확인으로 새 모델·seed·ρ를 재선택하지 않았다.", ""]
    for dataset in DATASETS:
        first = choose(tables["MATCHED_WHOLE_COST"], dataset=dataset, route="RAW_BOLT", batch=8)
        repeated = choose(tables["REPEAT_MATCHED_WHOLE_COST"], dataset=dataset, route="RAW_BOLT", batch=8)
        earlier = choose(tables["SELECTED_F_ONLY_COST"], dataset=dataset, route="RAW_BOLT", batch=8)
        lines.append(f"[확인] {dataset}/Raw B8 절대 ms/window: 앞선 F-only {earlier['median_seconds_per_window']*1000:.9f}; 첫 matched {first['median_seconds_per_window']*1000:.9f}; 별도 반복 {repeated['median_seconds_per_window']*1000:.9f}. 시행/프로세스 사이 변동 원인은 미확정이며 이를 제거한 speedup을 증명하지 않는다.")
    lines += ["",
        "## 확인한 것과 남은 판단", "",
        "[확인] causal past-only D fitting, 네 Bolt TUNE locks, 새 D에 matched actual-Y native8fit, 새로운 같은-mask 미래 공간 진단, 실제 세 비용 cohort와 독립 감사를 수행한 범위의 기록이다. 보호된 TEST·새 backbone/encoder/decoder 감독 학습·기존 작업 종료나 중단 학습 재개·자동 Git 게시는 하지 않았다. 본 생성기는 JSON metadata만 읽으며 NPZ·checkpoint/model을 열거나 F/head/fit/timer를 실행하지 않는다.", "",
        "[판단] selected shrinkage가 TUNE 불안정성을 줄이는 결과와 representation/latent future forecast 문제가 모두 해결되는 것은 별개다. 학습 예산·head family·ρ 선정이 제한됐고 이미 본 DEV에 적응해 다음 진단을 고른 탐색적 작업이다. 정규화 MSE의 채택 기준과 비용을 보존한 broad 연구 목적의 달성, 독립 일반화, Bolt 필요성, ExoTimer/contrastive/dynamic graph 효과는 아직 입증하지 않았다. 이전 실패를 덮지 않고 이 보고서의 다음 목적도 그 불확실성을 분리하는 것이다.", "", "## 원수치와 실제 파일", ""]
    for name, path in linked.items(): lines.append(f"- [{name}]({rel(path, output)}) — SHA256 `{sha(path)}`")
    for name, rows in tables.items(): lines.append(f"- [{name}: {len(rows)} rows](tables/{name}.csv)")
    return "\n".join(lines) + "\n"


def generate(paths, failed_cost_path, prior_reports, output, repeat_plan_path=None, audit_invocation_failure_path=None):
    values, bindings = load_verified(paths, failed_cost_path)
    require(repeat_plan_path is not None and audit_invocation_failure_path is not None, "COST_REPEAT_PLAN_AND_AUDIT_FAILURE_PROVENANCE_REQUIRED")
    require(Path(repeat_plan_path).suffix == ".md" and sha(repeat_plan_path) == REPEAT_PLAN_SHA, "REGISTERED_COST_REPEAT_PLAN")
    invocation = read(audit_invocation_failure_path)
    require(sha(audit_invocation_failure_path) == AUDIT_INVOCATION_IDENTITY_SHA
        and invocation["status"] == "CONFIRMED_PRESERVED_AUDITOR_INVOCATION_PATH_FAILURE"
        and invocation["scientific_inputs_sources_or_outputs_changed"] is invocation["new_F_native_forward_or_training_or_ridge_fit"] is invocation["new_cost_timing"] is False,
        "PRESERVED_AUDITOR_RELATIVE_PATH_INVOCATION_FAILURE")
    bound(invocation["audit_source"], values["matched_cost_audit"]["audit_source"], "same executed auditor invocation source")
    bound(invocation["failed_audit"], record(resolve(invocation["failed_audit"]["path"])), "preserved failed audit v1")
    bindings["cost_repeat_plan"], bindings["audit_invocation_failure"] = record(repeat_plan_path), record(audit_invocation_failure_path)
    require(all(Path(path).is_file() and Path(path).suffix == ".md" for path in prior_reports), "EXISTING_PRIOR_REPORTS_ONLY")
    output = Path(output).resolve(); require(not output.exists(), "FRESH_INTEGRATED_REPORT_DIRECTORY")
    tables = collect_tables(values)
    output.mkdir(parents=True, exist_ok=False); (output / "tables").mkdir()
    for name, rows in tables.items(): write_csv(output / "tables" / (name + ".csv"), rows)
    images = figures(tables, output)
    linked = {**paths, "preserved matched cost v1 IO failure": failed_cost_path,
        "confirmed v1 input schema failure": resolve(bindings["failed_cost_identity"]["path"]),
        "fresh v2 physical bridge binding repair": resolve(bindings["physical_bridge_repair_addendum"]["path"]),
        "same protocol repeat plan": repeat_plan_path,
        "preserved relative path auditor invocation failure": audit_invocation_failure_path,
        "preserved failed cost audit v1": resolve(invocation["failed_audit"]["path"]),
        **{"prior report " + str(index + 1): path for index, path in enumerate(prior_reports)}, "report source": Path(__file__)}
    (output / "REPORT_KO.md").write_text(report_text(values, tables, linked, output), encoding="utf-8")
    manifest = {"status": "COMPLETE", "source": record(__file__), "inputs": bindings, "prior_reports": [record(path) for path in prior_reports],
        "source_bindings": {name: record(path) for name, path in linked.items()}, "metadata_only": True, "NPZ_or_model_opened": False,
        "F_head_fit_or_timer_executed_by_report_generator": False, "protected_TEST_accessed": False, "same_decoder_side_information_claimed": False,
        "strict_K_only_compression": False, "independent_generalization_claimed": False, "Bolt_necessity_proven": False,
        "new_rho_seed_checkpoint_or_method_selection": False, "MAE_or_physical_MSE_minimum_claimed": False,
        "strict_FP32_universal_output_lower_bound_claimed": False, "RAW_bound_applied": False, "automatic_commit_push": False,
        "selected_F_only_cost_cells": 20, "selected_F_only_cost_samples": 120, "new_matched_whole_cost_cells": 28, "new_matched_whole_cost_samples": 196,
        "repeat_matched_whole_cost_cells": 28, "repeat_matched_whole_cost_samples": 196, "favorable_cost_cohort_selected": False,
        "CSV_rows": {name: len(rows) for name, rows in tables.items()}, "figures": [rel(path, output) for path in images],
        "files": [{"path": rel(path, output), "bytes": path.stat().st_size, "sha256": sha(path)} for path in sorted(output.rglob("*")) if path.is_file()]}
    with (output / "MANIFEST.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in INPUT_NAMES: parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--failed-matched-cost-v1", type=Path, required=True)
    parser.add_argument("--repeat-plan", type=Path, required=True)
    parser.add_argument("--cost-audit-relative-path-failure", type=Path, required=True)
    parser.add_argument("--prior-report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = generate({name: getattr(args, name) for name in INPUT_NAMES}, args.failed_matched_cost_v1, args.prior_report, args.output,
        args.repeat_plan, args.cost_audit_relative_path_failure)
    print(json.dumps({"status": receipt["status"], "CSV_rows": receipt["CSV_rows"]}))
