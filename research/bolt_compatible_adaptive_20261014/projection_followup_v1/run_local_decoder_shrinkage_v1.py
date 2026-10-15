"""TUNE-Bolt-selected observed-past ridge D; frozen accepted latent caches only."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATASETS = {"robin": (17, 4, "val", 30), "beijing_aotizhongxin": (11, 2, "eval", 146)}
ROUTES, SEEDS = ("PCA_PINV", "FIT_TARGET_PINV"), (92601, 92602)
RHOS = (.001, .01, .1, 1., 10.)
RHO_LABELS = ("RHO_0.001", "RHO_0.01", "RHO_0.1", "RHO_1", "RHO_10")
CANDIDATE_ORDER = ("STATIC", "LOCAL", *RHO_LABELS)
LOCAL_SHA = "86f894c72ea091c3a7d2b068a901fa59ec9229868acd6437a53c8709203084dc"
LOCAL_AUDIT_SHA = "96f37ff913ec4e8169b145a6bd67a3880b9e99f632f9d1fbbd96850183963bde"
PRIOR_SHAS = {"producer": "5d213da750bc6eb3b43dcb9d25fa08c3313f4c2662148cc2bad45adcc2cf6f98",
    "fit_target_audit": "14399051dc1614f785f4548daa139d92c8769e59fcbee9f38a98126914aedab6",
    "past_constant_diagnostic": "2c69a82644c1ee94bf743cedd640bdc07535dce13032f48f144dd642ffb5b917",
    "past_constant_audit": "4538d235d81280e7ddad2b632f8861ed37a25ffd841e5f704d3bb1a9cd9ca1a8",
    "past_constant_whole_cost": "37ccbf82b9e1df426aa28fd354eb04093325e9df570eccf6bdfaa5891c4a1537",
    "past_constant_whole_cost_audit": "88df2ac3e4b51b0618179a665baadd9b373b87f2b3773eca2114aa038c6da597"}
LOCAL_AUDIT_COUNTS = {"dataset_split_cells_checked": 6, "RAW_control_cells_checked": 6,
    "backend_states_checked": 24, "completed_backend_batches_checked": 24,
    "method_pairs_checked": 36, "window_matrix_fits_checked": 3264,
    "channel_slots_checked": 44304, "observed_channel_solver_calls_checked": 44304,
    "no_observation_fallbacks_checked": 0, "native_seed_D_reuses_checked": 1632,
    "static_prediction_arrays_byte_preserved": 36, "local_prediction_arrays_checked": 36,
    "static_einsum_control_arrays_checked": 36, "cached_future_arrays_byte_preserved": 36,
    "score_field_checks": 2592, "new_per_origin_output_diagnostic_backend_cells_checked": 24,
    "new_per_origin_output_inequalities_checked": 36, "RAW_bound_inequalities_checked": 0,
    "protected_prior_file_hashes_checked": 859}


def record(path):
    path = Path(path).resolve()
    with path.open("rb") as handle:
        return {"path": str(path), "sha256": hashlib.file_digest(handle, "sha256").hexdigest()}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_gates(local_path, local_audit_path):
    """Only JSON and source hashes are read before science/cache imports."""
    local, audit = read(local_path), read(local_audit_path)
    if local.get("status") != "COMPLETE" or audit.get("status") != "PASS":
        raise ValueError("COMPLETE_LOCAL_D_AND_INDEPENDENT_PASS_REQUIRED_BEFORE_SCIENCE")
    bindings = {"local_decoder_diagnostic": record(local_path), "local_decoder_audit": record(local_audit_path)}
    if (bindings["local_decoder_diagnostic"]["sha256"] != LOCAL_SHA
            or bindings["local_decoder_audit"]["sha256"] != LOCAL_AUDIT_SHA
            or audit.get("diagnostic") != bindings["local_decoder_diagnostic"]):
        raise ValueError("EXACT_IMMUTABLE_LOCAL_D_AND_PASS_SHA_REQUIRED")
    if any(audit.get("totals", {}).get(key) != value for key, value in LOCAL_AUDIT_COUNTS.items()):
        raise ValueError("FULL_LOCAL_D_INDEPENDENT_AUDIT_COVERAGE_REQUIRED")
    expected = {"completed_method_pairs": 36, "completed_backend_batches": 24,
        "completed_window_matrix_fits": 3264, "completed_channel_solver_calls": 44304,
        "completed_no_observation_fallbacks": 0, "native_seed_D_reuses": 1632,
        "new_E_F_native_head_forward_runs": 0, "protected_TEST_accessed": False,
        "new_E_F_or_native_head_training": False, "new_backbone_training": False,
        "new_GPU_execution": False, "original_source_cache_selection_SHA_unchanged": True,
        "future_labels_used_for_local_fit": False, "method_checkpoint_threshold_selection_used": False,
        "whole_local_deployment_cost_measured": False, "independent_holdout": False}
    if any(local.get(key) != value for key, value in expected.items()) or local.get("failed_paths") != []:
        raise ValueError("UNCHANGED_COMPLETE_NO_TEST_LOCAL_D_SCOPE_REQUIRED")
    if (audit.get("protected_TEST_accessed") is not False or audit.get("new_F_E_head_forward_or_training") is not False
            or audit.get("old_static_or_global_rankK_lower_bound_reused") is not False):
        raise ValueError("NO_TEST_NO_FORWARD_NO_INVALID_BOUND_AUDIT_REQUIRED")
    code = audit.get("audit_source", {})
    code_path = Path(code.get("path", ""))
    if (code_path.name != "audit_local_past_decoder_probe_v1.py" or not code_path.is_file()
            or record(code_path) != code):
        raise ValueError("EXACT_LOCAL_D_INDEPENDENT_AUDIT_SOURCE_REQUIRED")
    parents = {}
    for index, (name, expected_sha) in enumerate(PRIOR_SHAS.items()):
        item = local.get(name, {})
        if (item.get("sha256") != expected_sha or record(item.get("path", "")) != item
                or audit.get(name) != item):
            raise ValueError("EXACT_IMMUTABLE_PARENT_JSON_BINDING_REQUIRED:" + name)
        value = read(item["path"])
        if value.get("status") != ("COMPLETE" if index % 2 == 0 else "PASS") or value.get("protected_TEST_accessed") is not False:
            raise ValueError("COMPLETE_PASS_NO_TEST_PARENT_REQUIRED:" + name)
        bindings[name], parents[name] = item, value
    return local, parents["producer"], bindings


def byte_equal(first, second):
    import numpy as np
    return first.dtype == second.dtype and first.shape == second.shape and np.ascontiguousarray(first).tobytes() == np.ascontiguousarray(second).tobytes()


def shared_native_past(first, second):
    if not byte_equal(first["latent_history"], second["latent_history"]) or not byte_equal(first["origins"], second["origins"]):
        raise ValueError("NATIVE_SEED_CACHED_FIXED_E_PAST_NOT_BYTE_IDENTICAL")
    return first["latent_history"]


def select_bolt_only(candidates):
    """Every candidate must have a finite Bolt TUNE primary score; native is ignored."""
    scores = {}
    for name in CANDIDATE_ORDER:
        method = candidates.get(name, {}).get("methods", {}).get("BOLT", {})
        value = method.get("scores_all_observed", {}).get("normalized", {}).get("mse")
        if (method.get("status") != "COMPLETE" or not isinstance(value, (int, float))
                or isinstance(value, bool) or not math.isfinite(value) or value < 0):
            raise ValueError("COMPLETE_FINITE_ALL_SEVEN_BOLT_TUNE_SCORES_REQUIRED:" + name)
        scores[name] = float(value)
    selected = min(CANDIDATE_ORDER, key=lambda name: (scores[name], CANDIDATE_ORDER.index(name)))
    return {"status": "LOCKED", "selected_candidate": selected,
        "selected_rho": dict(zip(RHO_LABELS, RHOS)).get(selected), "selected_TUNE_Bolt_normalized_MSE": scores[selected],
        "TUNE_Bolt_normalized_MSE_by_candidate": scores, "candidate_tie_order": list(CANDIDATE_ORDER),
        "tie_rule": "Exact scalar equality; first in predeclared STATIC,LOCAL,.001,.01,.1,1,10 order",
        "selection_predictor": "BOLT_NUMPY", "selection_split": "TUNE", "selection_mask": "all_true_observed",
        "selection_metric": "normalized_macro_channel_MSE", "MAE_used_for_selection": False,
        "native_specific_selection": False, "native_uses_same_Bolt_selected_candidate": True,
        "development_or_oracle_scores_used_for_selection": False}


def locked_candidate_for_development(selection_records, key):
    expected = {(dataset, route) for dataset in DATASETS for route in ROUTES}
    if set(selection_records) != expected:
        raise ValueError("ALL_FOUR_SELECTION_LOCK_FILES_REQUIRED_BEFORE_ANY_DEV_CANDIDATE")
    for binding in selection_records.values():
        if record(binding["path"]) != binding:
            raise ValueError("SELECTION_LOCK_CHANGED_BEFORE_DEV")
    lock = read(selection_records[key]["path"])
    if lock.get("status") != "LOCKED":
        return None
    candidate = lock.get("selected_candidate")
    if candidate not in CANDIDATE_ORDER or lock.get("native_specific_selection") is not False:
        raise ValueError("VALID_SHARED_BOLT_ONLY_SELECTION_LOCK_REQUIRED")
    return candidate


def score_prediction(prediction, values):
    import numpy as np
    from decoder_metric_probe_v1 import masked_scores
    observed = values["observed"]
    complete = np.broadcast_to(observed.all(axis=2, keepdims=True), observed.shape)
    def score(mask):
        return masked_scores(prediction, values["target"], mask, values["std"]) if mask.any() else {
            "status": "NO_OBSERVED_TARGETS", "normalized": None, "physical_per_channel_diagnostic": None}
    return {"status": "COMPLETE", "scores_all_observed": score(observed), "scores_complete_mask": score(complete)}


def cached_backend(values, condition, local_backend, backend, io, meter, records):
    import numpy as np
    prior = io.checked_payload(condition["state_artifact"], meter, records)
    sources = ({"BOLT": condition["bolt"]["artifacts"][values["split"]]} if backend == "BOLT_NUMPY" else {
        "NATIVE_s" + str(seed): condition["native"][str(seed)]["artifacts"][values["split"]] for seed in SEEDS})
    cached = {name: io.checked_payload(item, meter, records) for name, item in sources.items()}
    original = io.checked_payload(local_backend["artifact"], meter, records)
    for name in ("origins", "target", "observed", "raw_last"):
        if not byte_equal(original[name], values[name]):
            raise ValueError("LOCAL_D_CONTROL_INPUT_BYTES_MISMATCH:" + name)
    z = cached["BOLT"]["latent_history"] if backend == "BOLT_NUMPY" else shared_native_past(
        cached["NATIVE_s92601"], cached["NATIVE_s92602"])
    if not byte_equal(original["latent_history"], z):
        raise ValueError("LOCAL_D_AND_FROZEN_CACHE_LATENT_PAST_BYTES_REQUIRED")
    for name, arrays in cached.items():
        if (not byte_equal(arrays["origins"], values["origins"])
                or not byte_equal(original[name + "_static_prediction"], arrays["prediction"])
                or not byte_equal(original[name + "_cached_future"], arrays["latent_future"])):
            raise ValueError("ORIGINAL_STATIC_PREDICTION_FUTURE_OR_ORIGINS_NOT_BYTE_PRESERVED:" + name)
    n, _, c = values["history"].shape
    k = condition["K"] if "K" in condition else prior["decoder"].shape[0]
    stored_prior = prior["decoder"]
    if (z.shape != (n, 512, k) or stored_prior.dtype not in (np.dtype("float32"), np.dtype("float64"))
            or stored_prior.shape != (k, c) or not np.isfinite(stored_prior).all()):
        raise ValueError("MATCHED_FIXED_E_AND_STATIC_PRIOR_DECODER_SHAPES_REQUIRED")
    with np.errstate(over="ignore", invalid="ignore"):
        prior32 = np.ascontiguousarray(stored_prior.astype(np.float32))
    if not np.isfinite(prior32).all():
        raise ValueError("FINITE_FP32_DEPLOYED_STATIC_PRIOR_CAST_REQUIRED")
    cast_binding = {"stored_prior_dtype": str(stored_prior.dtype), "deployment_cast": "ascontiguousarray(stored decoder.astype(float32)); original serialized prior preserved",
        "deployed_prior": {"shape": list(prior32.shape), "dtype": str(prior32.dtype),
            "sha256": hashlib.sha256(prior32.tobytes()).hexdigest()}}
    return {"prior": prior, "prior_decoder32": prior32, "static_prior_deployment_binding": cast_binding,
        "latent_history": z, "cached": cached, "controls": original,
        "cached_sources": sources, "original_state_source": condition["state_artifact"],
        "local_state_source": local_backend["local_state_artifact"], "local_forecast_source": local_backend["artifact"]}


def candidate_predictions(state, values, cached):
    import numpy as np
    predictions, failures = {}, {}
    for name, arrays in cached.items():
        try:
            if not byte_equal(arrays["origins"], values["origins"]):
                raise ValueError("MATCHED_METHOD_ORIGINS_REQUIRED")
            predictions[name] = state.forecast(values["raw_last"], arrays["latent_future"], arrays["latent_history"][:, -1])
        except Exception as error:
            failures[name] = {"status": "FAILED_PATH_PRESERVED", "failure": {
                "type": type(error).__name__, "message": str(error)}}
    return predictions, failures


def execute(local_path, local_audit_path, output_path):
    local, producer, bindings = load_gates(local_path, local_audit_path)
    output = Path(output_path).resolve()
    if not output.is_relative_to(HERE) or output.exists():
        raise ValueError("FRESH_SHRINKAGE_DIAGNOSTIC_DIRECTORY_REQUIRED")
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import numpy as np
    import fixed_decoder_span_v1 as io
    from local_decoder_shrinkage_v1 import fit_shrunk_local_decoder
    from decoder_metric_probe_v1 import masked_scores
    from continuation_runtime_v1 import Meter, write_new
    from run_fit_target_comparison_v1 import prior_file_records
    sources = [record(HERE / name) for name in (Path(__file__).name, "local_decoder_shrinkage_v1.py",
        "PLAN_LOCAL_DECODER_SHRINKAGE_KO_v1.md", "local_past_decoder_v1.py", "decoder_metric_probe_v1.py",
        "fixed_decoder_span_v1.py", "run_fit_target_comparison_v1.py", "continuation_runtime_v1.py")]
    output.mkdir(parents=True, exist_ok=False)
    meter = Meter("local_decoder_shrinkage_" + output.name, gpu=False)
    report = {"status": "RUNNING", **bindings, "source_records": sources, "datasets": {}, "failed_paths": [],
        "rho_candidates_predeclared": list(RHOS), "candidate_tie_order": list(CANDIDATE_ORDER), "execution_events": [],
        "attempted_positive_rho_backend_batches": 0, "completed_positive_rho_backend_batches": 0,
        "completed_window_matrix_fits": 0, "completed_channel_slots": 0,
        "completed_observed_channel_fits": 0, "completed_ridge_solver_calls": 0,
        "completed_zero_trace_prior_slope_fallbacks": 0, "completed_no_observation_fallbacks": 0,
        "native_seed_D_reuses": 0, "completed_FIT_TUNE_positive_rho_method_predictions": 0,
        "completed_selected_DEV_method_predictions": 0, "completed_reused_control_method_pairs": 0,
        "completed_RAW_control_cells": 0, "completed_selection_locks": 0,
        "new_E_F_native_head_forward_runs": 0, "new_E_F_or_native_head_training": False,
        "new_backbone_training": False, "new_GPU_execution": False, "protected_TEST_accessed": False,
        "future_labels_used_for_decoder_fit": False, "TUNE_future_labels_used_for_normal_hyperparameter_selection": True,
        "native_specific_selection": False, "native_uses_same_Bolt_selected_candidate": True,
        "all_selection_lock_files_saved_before_any_DEV_candidate_compute": False,
        "development_positive_rho_sweep_executed": False, "whole_selected_deployment_cost_measured": False,
        "fit_uses_TUNE_DEV_known_past_at_origin": True, "same_decoder_side_information_as_static": False,
        "strict_K_only_compression": False, "old_static_or_global_rankK_floor_reused": False,
        "independent_holdout": False}
    records, failure, guarded, selection_records = {}, None, [], {}

    def event(kind, **details):
        report["execution_events"].append({"sequence": len(report["execution_events"]), "kind": kind, **details})
        meter.check({"stage": kind, **details})

    def failed(error, **details):
        value = {"status": "FAILED_PATH_PRESERVED", **details,
            "failure": {"type": type(error).__name__, "message": str(error)}}
        report["failed_paths"].append(value)
        return value

    def split_input(dataset, split, count, c):
        values = io.checked_payload(producer["datasets"][dataset]["inputs"][split], meter, records)
        if (values["history"].shape != (count, 512, c)
                or not byte_equal(values["raw_last"], values["history"][:, -1])):
            raise ValueError("EXACT_REGISTERED_PAST_COUNT_AND_CURRENT_C_REQUIRED")
        values["split"] = split
        return values

    def prepare_split(dataset, split, count, c):
        values = split_input(dataset, split, count, c)
        cell = {"status": "RUNNING", "input_source": producer["datasets"][dataset]["inputs"][split],
            "RAW": score_prediction(values["raw_forecast"], values), "routes": {}}
        report["datasets"][dataset]["splits"][split] = cell
        report["completed_RAW_control_cells"] += 1
        return values, cell

    def prepare_route(dataset, route, split, values, cell):
        route_row = {"controls": {name: {"methods": {}} for name in ("STATIC", "LOCAL")},
            "positive_rho_candidates": {}, "control_backends": {}}
        cell["routes"][route] = route_row
        contexts = {}
        for backend in ("BOLT_NUMPY", "NATIVE_TORCH32"):
            try:
                old = local["datasets"][dataset]["splits"][split]["routes"][route]["backends"][backend]
                context = cached_backend(values, producer["datasets"][dataset]["projections"][route], old,
                    backend, io, meter, records)
                contexts[backend] = context
                preserved = {name: array.copy() for name, array in context["controls"].items()}
                prefix = dataset + "_" + route + "_" + split + "_" + backend
                control_record = io.save_npz(output / (prefix + "_REUSED_CONTROLS.npz"), preserved)
                route_row["control_backends"][backend] = {"status": "COMPLETE", "artifact": control_record,
                    "original_state_source": context["original_state_source"], "local_state_source": context["local_state_source"],
                    "local_forecast_source": context["local_forecast_source"], "cached_method_sources": context["cached_sources"],
                    "static_prior_deployment_binding": context["static_prior_deployment_binding"],
                    "native_past_byte_exact_before_D_reuse": backend == "NATIVE_TORCH32", "state_refit": False}
                for name in context["cached"]:
                    for control, suffix in (("STATIC", "static"), ("LOCAL", "local")):
                        scores = score_prediction(preserved[name + "_" + suffix + "_prediction"], values)
                        scores.update(artifact=control_record, prediction_member=name + "_" + suffix + "_prediction",
                            original_cached_prediction_byte_preserved=True, native_head_frozen_intervention_not_reoptimized=backend == "NATIVE_TORCH32")
                        route_row["controls"][control]["methods"][name] = scores
                    route_row["controls"]["STATIC"]["methods"][name]["static_einsum_replay_scores"] = score_prediction(
                        preserved[name + "_static_einsum_replay"], values)
                    report["completed_reused_control_method_pairs"] += 1
            except Exception as error:
                route_row["control_backends"][backend] = failed(error, dataset=dataset, route=route, split=split,
                    backend=backend, stage="load_and_preserve_controls")
        return route_row, contexts

    def positive_rho(dataset, route, split, values, contexts, label, rho, development=False):
        entry = {"rho": rho, "backends": {}, "methods": {}}
        for backend, context in contexts.items():
            report["attempted_positive_rho_backend_batches"] += 1
            event("DEV_SELECTED_RHO_FIT" if development else "FIT_TUNE_RHO_FIT", dataset=dataset,
                route=route, split=split, backend=backend, candidate=label)
            try:
                state = fit_shrunk_local_decoder(values["history"], values["history_observed"],
                    context["latent_history"], context["prior_decoder32"], rho)
                counts = np.asarray(state.diagnostics["true_observed_counts"])
                actual_lambda = np.asarray(state.diagnostics["actual_lambda"])
                report["completed_window_matrix_fits"] += len(values["history"])
                report["completed_channel_slots"] += int(counts.size)
                report["completed_observed_channel_fits"] += int((counts > 0).sum())
                report["completed_ridge_solver_calls"] += int((actual_lambda > 0).sum())
                report["completed_zero_trace_prior_slope_fallbacks"] += int(np.asarray(state.diagnostics["zero_trace_prior_slope_retained"]).sum())
                report["completed_no_observation_fallbacks"] += int((counts == 0).sum())
                if backend == "NATIVE_TORCH32":
                    report["native_seed_D_reuses"] += len(values["history"])
                prefix = dataset + "_" + route + "_" + split + "_" + backend + "_" + label
                state_artifact = io.save_npz(output / (prefix + "_STATE.npz"), state.to_npz())
                past = state.reconstruct(context["latent_history"])
                backend_row = {"status": "RUNNING", "state_artifact": state_artifact, "state_metadata": state.metadata(),
                    "cached_method_sources": context["cached_sources"], "static_prior_source": context["original_state_source"],
                    "static_prior_deployment_binding": context["static_prior_deployment_binding"],
                    "native_past_byte_exact_before_D_reuse": backend == "NATIVE_TORCH32", "methods": {},
                    "past_reconstruction_scores_observed_only": masked_scores(past, values["history"], values["history_observed"], values["std"])
                        if values["history_observed"].any() else {"status": "NO_OBSERVED_PAST_TARGETS", "normalized": None}}
                entry["backends"][backend] = backend_row
                predictions, errors = candidate_predictions(state, values, context["cached"])
                arrays = {"past_prediction_FP32": past, "latent_history": context["latent_history"].copy(),
                    "origins": values["origins"].copy(), "target": values["target"].copy(),
                    "observed": values["observed"].copy(), "raw_last": values["raw_last"].copy()}
                for name, prediction in predictions.items():
                    arrays[name + "_prediction"] = prediction
                    arrays[name + "_cached_future"] = context["cached"][name]["latent_future"].copy()
                artifact = io.save_npz(output / (prefix + "_FORECASTS.npz"), arrays)
                backend_row["artifact"] = artifact
                for name, prediction in predictions.items():
                    scores = score_prediction(prediction, values)
                    scores.update(artifact=artifact, prediction_member=name + "_prediction", new_latent_forward_or_training=False,
                        native_head_frozen_intervention_not_reoptimized=backend == "NATIVE_TORCH32")
                    backend_row["methods"][name] = entry["methods"][name] = scores
                    key = "completed_selected_DEV_method_predictions" if development else "completed_FIT_TUNE_positive_rho_method_predictions"
                    report[key] += 1
                for name, item in errors.items():
                    backend_row["methods"][name] = entry["methods"][name] = item
                    report["failed_paths"].append({"dataset": dataset, "route": route, "split": split,
                        "backend": backend, "candidate": label, "method": name, **item})
                backend_row["status"] = "PARTIAL_COMPLETED" if errors else "COMPLETE"
                if not errors:
                    report["completed_positive_rho_backend_batches"] += 1
            except Exception as error:
                entry["backends"].setdefault(backend, {}).update(failed(error, dataset=dataset, route=route,
                    split=split, backend=backend, candidate=label, stage="positive_rho_fit_or_forecast"))
        return entry

    try:
        meter.protected_check()
        sys.path.insert(0, str(ROOT / "research/smallk_failure_followup_20261014"))
        from prepare_inputs import install_protected_test_guard
        report["protected_guard_paths"] = install_protected_test_guard()
        guarded = prior_file_records(local, read(local_audit_path), producer, *bindings.values(), *sources)
        scope = {"status": "CONTINUING_GOAL_PAST_DECODER_STABILITY_JUDGMENT", "individual_exact_human_approval_claimed": False,
            "bindings": bindings, "plan": sources[2], "rho_candidates_predeclared": list(RHOS),
            "objective": "mean observed-centered past squared residual + lambda||D-Dstatic||^2; lambda=rho*trace(G)/K",
            "selection": "Four dataset-by-Espace locks; all seven Bolt TUNE full-observed normalized macro-MSE candidates; shared selected condition for both frozen native seeds",
            "decoder_fit_uses_TUNE_DEV_known_past": True, "decoder_fit_uses_future_labels": False,
            "TUNE_future_labels_used_for_normal_hyperparameter_selection": True,
            "DEV_positive_rho_sweep": False, "all_four_lock_files_before_any_DEV_candidate_compute": True,
            "static_and_original_local_controls_reused_byte_exact": True,
            "native_head_frozen_intervention_not_reoptimized": True, "strict_K_only_compression": False,
            "same_decoder_side_information_as_static": False, "whole_selected_deployment_cost_measured": False,
            "independent_generalization_proven": False, "new_E_F_head_forward_training_GPU_TEST": False}
        write_new(output / "SCOPE.json", scope)
        report["scope"] = record(output / "SCOPE.json")
        for dataset, (channels, k, development, dev_count) in DATASETS.items():
            report["datasets"][dataset] = {"K": k, "channels": channels, "development_split": development, "splits": {}}
            for split, count in (("fit", 256), ("tune", 64)):
                event("FIT_TUNE_SPLIT_BEGIN", dataset=dataset, split=split)
                try:
                    values, cell = prepare_split(dataset, split, count, channels)
                except Exception as error:
                    report["datasets"][dataset]["splits"][split] = failed(error, dataset=dataset, split=split, stage="input")
                    continue
                for route in ROUTES:
                    route_row, contexts = prepare_route(dataset, route, split, values, cell)
                    for label, rho in zip(RHO_LABELS, RHOS):
                        route_row["positive_rho_candidates"][label] = positive_rho(dataset, route, split, values, contexts, label, rho)
                    route_row["status"] = "COMPLETE" if len(contexts) == 2 and all(
                        len(item["methods"]) == 3 and all(value.get("status") == "COMPLETE" for value in item["methods"].values())
                        for item in route_row["positive_rho_candidates"].values()) else "PARTIAL_COMPLETED"
                cell["status"] = "COMPLETE" if all(row["status"] == "COMPLETE" for row in cell["routes"].values()) else "PARTIAL_COMPLETED"
        for dataset in DATASETS:
            for route in ROUTES:
                try:
                    tune = report["datasets"][dataset]["splits"]["tune"]["routes"][route]
                    candidates = {**tune["controls"], **tune["positive_rho_candidates"]}
                    lock = select_bolt_only(candidates)
                    lock.update(dataset=dataset, route=route, local_decoder_diagnostic=bindings["local_decoder_diagnostic"],
                        local_decoder_audit=bindings["local_decoder_audit"],
                        TUNE_input_source=producer["datasets"][dataset]["inputs"]["tune"],
                        TUNE_Bolt_candidate_prediction_records={name: {
                            "artifact": candidates[name]["methods"]["BOLT"]["artifact"],
                            "prediction_member": candidates[name]["methods"]["BOLT"]["prediction_member"]} for name in CANDIDATE_ORDER})
                    report["completed_selection_locks"] += 1
                except Exception as error:
                    lock = failed(error, dataset=dataset, route=route, stage="Bolt_TUNE_selection")
                    lock.update(status="SKIPPED_MISSING_COMPLETE_TUNE_CANDIDATE_SET", native_specific_selection=False,
                        development_dependent_candidate_compute_allowed=False)
                path = output / (dataset + "_" + route + "_SELECTION_LOCK.json")
                write_new(path, lock)
                selection_records[(dataset, route)] = record(path)
                event("SELECTION_LOCK_SAVED", dataset=dataset, route=route, status=lock["status"], binding=selection_records[(dataset, route)])
        report["selection_locks"] = {dataset: {route: selection_records[(dataset, route)] for route in ROUTES} for dataset in DATASETS}
        report["all_selection_lock_files_saved_before_any_DEV_candidate_compute"] = True
        event("ALL_FOUR_SELECTION_LOCK_FILES_SAVED", files=len(selection_records))
        for dataset, (channels, k, development, dev_count) in DATASETS.items():
            event("DEV_SPLIT_BEGIN", dataset=dataset, split=development)
            try:
                values, cell = prepare_split(dataset, development, dev_count, channels)
            except Exception as error:
                report["datasets"][dataset]["splits"][development] = failed(error, dataset=dataset, split=development, stage="input")
                continue
            for route in ROUTES:
                candidate = locked_candidate_for_development(selection_records, (dataset, route))
                route_row, contexts = prepare_route(dataset, route, development, values, cell)
                route_row.update(selection_lock=selection_records[(dataset, route)], selected_candidate=candidate,
                    unselected_positive_rho_DEV_candidates_computed=False)
                if candidate is None:
                    route_row.update(status="SKIPPED_DEPENDENT_DEV_NO_SELECTION", selected_methods={})
                    continue
                if candidate in ("STATIC", "LOCAL"):
                    route_row["selected_methods"] = route_row["controls"][candidate]["methods"]
                    route_row["selected_positive_rho_backend_fits"] = 0
                    report["completed_selected_DEV_method_predictions"] += len(route_row["selected_methods"])
                    event("DEV_SELECTED_CACHED_CONTROL", dataset=dataset, route=route, candidate=candidate)
                else:
                    entry = positive_rho(dataset, route, development, values, contexts, candidate,
                        dict(zip(RHO_LABELS, RHOS))[candidate], development=True)
                    route_row["positive_rho_candidates"][candidate] = entry
                    route_row["selected_methods"] = entry["methods"]
                route_row["status"] = "COMPLETE" if len(route_row["selected_methods"]) == 3 and all(
                    item.get("status") == "COMPLETE" for item in route_row["selected_methods"].values()) else "PARTIAL_COMPLETED"
            cell["status"] = "COMPLETE" if all(row["status"] == "COMPLETE" for row in cell["routes"].values()) else "PARTIAL_COMPLETED"
        for item in guarded + sources + list(records.values()) + list(selection_records.values()):
            if record(item["path"])["sha256"] != item["sha256"]:
                raise ValueError("ORIGINAL_SOURCE_CACHE_MODEL_SELECTION_OR_NEW_SELECTION_CHANGED")
        meter.protected_check()
        complete = (report["completed_FIT_TUNE_positive_rho_method_predictions"] == 120
            and report["completed_selected_DEV_method_predictions"] == 12 and report["completed_reused_control_method_pairs"] == 36
            and report["completed_RAW_control_cells"] == 6 and report["completed_selection_locks"] == 4 and not report["failed_paths"])
        report.update(status="COMPLETE" if complete else "PARTIAL_COMPLETED", protected_prior_records=guarded,
            input_records=list(records.values()), original_source_cache_model_and_selection_SHA_unchanged=True,
            expected_FIT_TUNE_positive_rho_method_predictions=120, expected_selected_DEV_method_predictions=12,
            expected_reused_control_method_pairs=36, expected_selection_locks=4,
            maximum_FIT_TUNE_window_matrix_fits=12800, maximum_selected_DEV_window_matrix_fits=704,
            maximum_total_window_matrix_fits=13504, maximum_channel_slots=187664, maximum_native_seed_D_reuses=6752,
            solver_count_semantics="Completed decoder fit batches; trace>0 np.linalg.solve counted separately from zero-trace and no-observed prior fallbacks",
            cost_status="NOT_MEASURED: current stage uses cached CPU past-decoder fit/reconstruction/forecast; selected actual F whole cost is a separate step",
            full_research_resolution_claimed=False)
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        report.update(status="FAILED_PRESERVED", failure=failure, full_scope_complete_claimed=False)
    finally:
        report["resource_terminal"] = meter.close(report["status"], failure)
        write_new(output / "RESULT.json", report)
    print(json.dumps({"status": report["status"], "output": str(output / "RESULT.json"),
        "selection_locks": report["completed_selection_locks"], "window_fits": report["completed_window_matrix_fits"],
        "FIT_TUNE_positive_predictions": report["completed_FIT_TUNE_positive_rho_method_predictions"],
        "selected_DEV_predictions": report["completed_selected_DEV_method_predictions"], "failure": failure}), flush=True)
    return 0 if report["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("local-diagnostic", "local-audit", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    raise SystemExit(execute(args.local_diagnostic, args.local_audit, args.output))
