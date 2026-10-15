"""Independent CPU32 head equations and matched frozen-payload/CP audit."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import sys

import audit_projection_results_v1 as prior
import audit_local_past_decoder_probe_v1 as local_audit

np = sys.modules.get("numpy")
HERE, ROOT = prior.HERE, prior.ROOT
DATASETS = local_audit.DATASETS
SEEDS, CHECKPOINTS = (92601, 92602), (0, 100, 200, 400, 600)
require, exact = prior.require, local_audit.exact
raw_binding = local_audit.raw_binding


def initialize_numpy():
    global np
    local_audit.initialize_numpy()
    np = sys.modules["numpy"]


def independent_head_forward(payload, state):
    z, current, decoder = (payload[key] for key in ("latent_history", "raw_last", "decoder"))
    require(set(state) == {"head.weight", "head.bias"} and z.dtype == current.dtype == decoder.dtype == np.float32
            and z.ndim == 3 and z.shape[1] == 512 and len(z) > 0 and current.ndim == 2 and len(current) == len(z)
            and decoder.shape == (len(z), z.shape[-1], current.shape[-1])
            and state["head.weight"].shape == (z.shape[-1], 48, 512) and state["head.bias"].shape == (z.shape[-1], 48)
            and all(value.dtype == np.float32 and np.isfinite(value).all() for value in (*state.values(), z, current, decoder)), "MATCHED_FINITE_FP32_ZERO_OR_TRAINED_HEAD_AND_FROZEN_PAST_D_CONTRACT")
    import torch
    torch.set_num_threads(2)
    weights = torch.from_numpy(np.ascontiguousarray(state["head.weight"]))
    bias = torch.from_numpy(np.ascontiguousarray(state["head.bias"]))
    predictions, futures = [], []
    with torch.no_grad():
        for start in range(0, len(z), 32):
            cached = torch.from_numpy(np.ascontiguousarray(z[start:start + 32]))
            raw = torch.from_numpy(np.ascontiguousarray(current[start:start + 32]))
            d = torch.from_numpy(np.ascontiguousarray(decoder[start:start + 32]))
            last = cached[:, -1:]
            innovation = torch.einsum("nlk,khl->nhk", cached - last, weights)
            future = last + (innovation + bias.transpose(0, 1).unsqueeze(0))
            prediction = raw[:, None] + torch.einsum("nhk,nkc->nhc", future - last, d)
            predictions.append(prediction.numpy().copy()); futures.append(future.numpy().copy())
    return tuple(np.ascontiguousarray(np.concatenate(chunks)) for chunks in (predictions, futures))


def normalized_CP_score(recorded, prediction, payload, std, label):
    score = prior.scores(prediction, payload["target"], payload["observed"], std)["normalized"]
    return score, prior.compare_normalized(recorded, score, label)


def selected_checkpoint(rows):
    require(len(rows) == 5 and [row["update"] for row in rows] == list(CHECKPOINTS)
            and all(np.isfinite(row["mse"]) and row["mse"] >= 0 for row in rows), "ALL_FIVE_FINITE_CHECKPOINT_PRIMARY_SCORES")
    chosen = rows[0]
    for row in rows[1:]:
        if row["mse"] < chosen["mse"]:
            chosen = row
    return chosen["update"]


def score_full_complete(metadata, prediction, values, label):
    output, checks = {}, 0
    for field, mask in (("scores_all_observed", values["observed"]), ("scores_complete_mask", prior.oracle_mask(values["observed"]))):
        score, count = local_audit.score_checked(metadata[field], prediction, values["target"], mask, values["std"], label + ":" + field)
        output[field], checks = score, checks + count
    return output, checks


def check_registry_events(events, registry, locks):
    require([event["sequence"] for event in events] == list(range(len(events))), "CONTIGUOUS_MATCHED_TRAINING_EXECUTION_EVENTS")
    expected = {(data, route, str(seed)) for data in DATASETS for route in ("PCA_PINV", "FIT_TARGET_PINV") for seed in SEEDS}
    require(set(locks) == expected and registry["status"] == "ALL_EIGHT_CONDITIONS_RESOLVED_BEFORE_ANY_NEW_DEV_INPUT_OR_PREDICTION"
            and registry["expected_conditions"] == len(registry["conditions"]) == 8, "ALL_EIGHT_MATCHED_CONDITION_LOCKS_AND_REGISTRY")
    require({(entry["dataset"], entry["route"], entry["seed"]) for entry in registry["conditions"]} == expected, "NO_DUPLICATE_OR_MISSING_REGISTRY_CONDITION")
    for entry in registry["conditions"]:
        key = (entry["dataset"], entry["route"], entry["seed"])
        require(entry["lock"] == locks[key] and entry["status"] in ("LOCKED", "FAILED_PATH_PRESERVED"), "REGISTRY_EXACT_LOCK_OR_PRESERVED_FAILURE_BINDING")
    saved = [event for event in events if event["kind"] == "CONDITION_SELECTION_LOCK_SAVED"]
    sealed = [event for event in events if event["kind"] == "ALL_EIGHT_CONDITION_LOCKS_AND_REGISTRY_SAVED"]
    require(len(saved) == 8 and {(event["dataset"], event["route"], event["seed"]) for event in saved} == expected
            and len(sealed) == 1 and max(event["sequence"] for event in saved) < sealed[0]["sequence"], "ALL_EIGHT_LOCKS_SAVED_BEFORE_REGISTRY_GATE")
    for event in events:
        if event["kind"] in ("DEV_INPUT_LOAD_AFTER_ALL_LOCKS", "DEV_NATIVE_PREDICTION_AFTER_ALL_LOCKS"):
            require(event["sequence"] > sealed[0]["sequence"], "NO_NEW_DEV_INPUT_OR_PREDICTION_BEFORE_ALL_EIGHT_RESOLVED_LOCKS")
    return {"all_eight_resolved_locks_before_any_new_DEV_input_or_prediction": True, "events_checked": len(events)}


def source_contract(records):
    names = {Path(entry["path"]).name: entry for entry in records}
    require("native_shrunk_decoder_head_v1.py" in names and "run_matched_shrunk_native_v1.py" in names, "EXACT_MATCHED_SOURCE_FILES_PRESENT")
    text = Path(names["native_shrunk_decoder_head_v1.py"]["path"]).read_text(encoding="utf-8-sig")
    tree = ast.parse(text)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NativeShrunkForecaster")
    components = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "predict_components")
    require([arg.arg for arg in components.args.args] == ["self", "latent_history", "raw_last", "decoder"], "MATCHED_FORWARD_HAS_NO_FUTURE_TARGET_OR_MASK_PARAMETER")
    require("value.detach()" in ast.unparse(components), "FROZEN_PAST_CURRENT_AND_DECODER_DETACHED_BEFORE_FORWARD")
    trainer = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train_native_shrunk")
    require("optimizer = torch.optim.AdamW(model.head.parameters(), lr=0.001, weight_decay=0.0)" in ast.unparse(trainer), "ONLY_FRESH_HEAD_PARAMETERS_IN_REGISTERED_OPTIMIZER")
    return {"three_past_only_forward_inputs": True, "head_only_optimizer_source_checked": True,
            "new_learning_trajectory_independently_replayed": False, "independent_saved_Torch_CPU32_equations": True}


def audit_context(entry, original, shrink_route, split, values, check):
    label, rho = shrink_route["selected_candidate"], shrink_route["rho"]
    selected = original["positive_rho_candidates"][label]
    native = selected["backends"]["NATIVE_TORCH32"]
    bolt = selected["backends"]["BOLT_NUMPY"]
    require(entry["status"] == "COMPLETE" and entry["rho"] == rho and entry["selected_candidate"] == label and entry["backend"] == "NATIVE_TORCH32"
            and entry["selected_native_state_source"] == native["state_artifact"] and entry["selected_native_forecasts_source"] == native["artifact"]
            and entry["selected_Bolt_forecasts_source"] == bolt["artifact"] and entry["original_native_cached_sources"] == native["cached_method_sources"]
            and entry["seed_cached_past_byte_identical_and_D_shared"] is entry["control_predictions_copied_byte_exact"] is True and entry["D_refit"] is False,
            "EXACT_MATCHED_CONTEXT_ALREADY_SELECTED_NATIVE_BACKEND_D_AND_BOLT_RHO")
    state, cached, bolt_arrays, archived = (prior.verified_arrays(record, check) for record in (entry["selected_native_state_source"], entry["selected_native_forecasts_source"], entry["selected_Bolt_forecasts_source"], entry["fixed_inputs_and_controls_artifact"]))
    expected_keys = {"latent_history", "decoder", "raw_last", "target", "observed", "origins", "mean", "std", "RAW_BOLT_prediction", "SELECTED_BOLT_prediction", "OLD_FROZEN_NATIVE_s92601_prediction", "OLD_FROZEN_NATIVE_s92602_prediction"}
    require(set(archived) == expected_keys, "EXACT_ARCHIVED_FROZEN_INPUTS_AND_FOUR_CONTROLS")
    for key in ("raw_last", "target", "observed", "origins", "mean", "std"):
        exact(archived[key], values[key], "matched_same_original_raw_payload:" + key)
    exact(archived["decoder"], state["decoder_deployed_FP32"], "matched_same_audited_native_selected_D32")
    exact(archived["latent_history"], cached["latent_history"], "matched_same_selected_native_cached_past")
    for source in native["cached_method_sources"].values():
        old = prior.verified_arrays(source, check)
        exact(old["latent_history"], archived["latent_history"], "matched_native_two_seed_same_past_before_shared_D")
        exact(old["origins"], archived["origins"], "matched_native_original_cache_origins")
    controls = {"RAW_BOLT": values["raw_forecast"], "SELECTED_BOLT": bolt_arrays["BOLT_prediction"],
                **{"OLD_FROZEN_NATIVE_s" + str(seed): cached["NATIVE_s" + str(seed) + "_prediction"] for seed in SEEDS}}
    require(set(entry["controls"]) == set(controls), "EXACT_RAW_SELECTED_BOLT_TWO_OLD_NATIVE_CONTROLS")
    scores, checks = {}, 0
    for name, prediction in controls.items():
        member = name + "_prediction"
        exact(archived[member], prediction, "matched_preserved_control_forecast_bytes:" + name)
        require(entry["controls"][name]["prediction_member"] == member, "MATCHED_CONTROL_SCORE_ARRAY_BINDING")
        score, count = score_full_complete(entry["controls"][name], prediction, values, "matched_control:" + name)
        scores[name], checks = score, checks + count
    bindings = native["state_metadata"]["fit_input_bindings"]
    require(bindings["history"] == raw_binding(values["history"]) and bindings["history_observed"] == raw_binding(values["history_observed"])
            and bindings["latent_history"] == raw_binding(archived["latent_history"]), "SELECTED_D_FIT_BOUND_TO_SAME_KNOWN_PAST_NATIVE_BACKEND")
    return {"payload": {key: archived[key] for key in ("latent_history", "decoder", "raw_last", "target", "observed", "origins")},
            "std": values["std"], "values": values, "scores": scores,
            "persistence_current_C_baseline": {"role": "PAST_ONLY_CURRENT_C_REPEAT48_NO_MODEL_FORECAST",
                "scores_all_observed": prior.scores(np.broadcast_to(values["raw_last"][:, None], values["target"].shape), values["target"], values["observed"], values["std"]),
                "scores_complete_mask": prior.scores(np.broadcast_to(values["raw_last"][:, None], values["target"].shape), values["target"], prior.oracle_mask(values["observed"]), values["std"]) if prior.oracle_mask(values["observed"]).any() else None,
                "whole_deployment_cost_measured": False, "Raw_Bolt_identical_claimed": False},
            "score_field_checks": checks, "archived_record": entry["fixed_inputs_and_controls_artifact"]}


def audit_native(entry, contexts, lock, seed, channels, k, development, check):
    metadata = entry["metadata"]
    require(entry["status"] == "COMPLETE" and metadata["seed"] == seed and metadata["updates"] == 600 and metadata["batch_size"] == 4
            and metadata["K"] == k and metadata["C"] == channels and metadata["L"] == 512 and metadata["H"] == 48
            and metadata["parameters"] == k * 48 * 513 and metadata["optimizer"] == "AdamW" and metadata["learning_rate"] == .001
            and metadata["weight_decay"] == 0 and metadata["gradient_clip"] == 1 and metadata["checkpoint_updates"] == list(CHECKPOINTS)
            and metadata["canonical_checkpoint_prediction_batch_size"] == 32 and metadata["checkpoint_callback_prediction_is_exact_scored_array"] is True
            and metadata["initial_weights_and_biases_zero"] is True and metadata["seed_changes_initial_weights"] is False
            and metadata["decoder_condition_selected_by_native"] is metadata["rho_selection_inside_trainer"] is metadata["encoder_decoder_backbone_optimized"] is metadata["fixed_per_window_D_beta_requires_grad"] is False
            and metadata["future_encoding_in_forward"] is metadata["new_backbone_training"] is metadata["DEV_loaded_by_trainer"] is metadata["old_head_weights_reused_or_warm_started"] is False,
            "EIGHT_FRESH_ZERO_HEAD_REGISTERED_OPTIMIZER_MATCHED_DECODER_SCOPE")
    frozen = {split: prior.torch_state_sha({name: context["payload"][name] for name in ("latent_history", "decoder", "raw_last")}) for split, context in contexts.items() if split in ("fit", "tune")}
    require(metadata["frozen_payload_before"] == metadata["frozen_payload_after"] == frozen, "FROZEN_NATIVE_PAST_CURRENT_D_BYTE_DIGEST_BEFORE_AFTER")
    require(metadata["payload_array_bindings"] == {split: {name: raw_binding(value) for name, value in context["payload"].items()} for split, context in contexts.items() if split in ("fit", "tune")}, "EXACT_FIT_TUNE_HEAD_SUPERVISION_AND_FROZEN_PAYLOAD_BINDINGS")
    order = np.random.default_rng(seed).integers(0, len(contexts["fit"]["payload"]["latent_history"]), size=(600, 4))
    require(metadata["order_sha256"] == hashlib.sha256(order.tobytes()).hexdigest(), "MATCHED_SAME_FIT_ONLY_SEEDED_MINIBATCH_ORDER")
    require(len(entry["learning_curve"]) == 600, "ACTUAL_MATCHED_600_UPDATE_CURVE")
    for update, row in enumerate(entry["learning_curve"], 1):
        require(row["update"] == update and np.isfinite(row["loss"]) and row["loss"] >= 0
                and np.isfinite(row["gradient_norm_before_clip"]) and row["gradient_norm_before_clip"] >= 0, "FINITE_ACTUAL_MATCHED_TRAINING_UPDATE_LOG")
    require([row["update"] for row in entry["checkpoints"]] == list(CHECKPOINTS), "EXACT_FIVE_MATCHED_CHECKPOINT_RECORDS")
    tune, checkpoint_states, checkpoint_predictions, rows, details, checks = contexts["tune"], {}, {}, [], [], 0
    for cp in entry["checkpoints"]:
        require(cp["canonical_prediction_batch_size"] == 32, "MATCHED_CANONICAL_CP_PREDICTION_BATCH32")
        state = prior.verified_arrays(cp["state_artifact"], check)
        require(set(state) == {"head.weight", "head.bias"} and prior.torch_state_sha(state) == cp["state_sha256"], "MATCHED_HEAD_ONLY_CHECKPOINT_STATE_DIGEST")
        saved = prior.verified_arrays(cp["tune_prediction_artifact"], check)
        require(set(saved) == {"prediction", "origins"}, "EXACT_MATCHED_CP_TUNE_PREDICTION_KEYS")
        exact(saved["origins"], tune["payload"]["origins"], "matched_CP_original_TUNE_origins")
        predicted, _ = independent_head_forward(tune["payload"], state)
        exact(saved["prediction"], predicted, "independent_Torch32_CP_exact_scored_array_u" + str(cp["update"]))
        score, count = normalized_CP_score(cp["tune"], saved["prediction"], tune["payload"], tune["std"], "matched_CP_TUNE_actualY")
        checks += count
        rows.append({"update": cp["update"], "mse": score["mse"]})
        checkpoint_states[cp["update"]], checkpoint_predictions[cp["update"]] = state, saved["prediction"]
        details.append({"update": cp["update"], "normalized_TUNE": score, "state_sha256": cp["state_sha256"], "independent_forward_byte_exact": True})
        if cp["update"] == 0:
            require(np.count_nonzero(state["head.weight"]) == np.count_nonzero(state["head.bias"]) == 0 and metadata["initial_model_sha256"] == prior.torch_state_sha(state), "MATCHED_FRESH_CP0_ZERO_MODEL_STATE")
            require(np.array_equal(saved["prediction"], np.broadcast_to(tune["payload"]["raw_last"][:, None], saved["prediction"].shape)), "MATCHED_CP0_NUMERIC_CURRENT_C_ANCHOR")
            prior.close(metadata["CP0_anchor_max_abs_difference_FP32"], 0., "matched_zero_CP_current_anchor", atol=1e-10, rtol=1e-10)
    selected = selected_checkpoint(rows)
    literal_selected = selected_checkpoint([{"update": cp["update"], "mse": cp["tune"]["mse"]} for cp in entry["checkpoints"]])
    chosen_cp = entry["checkpoints"][list(CHECKPOINTS).index(selected)]
    require(metadata["selected_update"] == selected == literal_selected == lock["selected_update"]
            and metadata["selected_state_sha256"] == chosen_cp["state_sha256"] and metadata["selected_tune"] == chosen_cp["tune"]
            and lock["selected_tune"] == chosen_cp["tune"] and lock["selected_state_artifact"] == chosen_cp["state_artifact"]
            and lock["selected_CP_TUNE_prediction_artifact"] == chosen_cp["tune_prediction_artifact"], "EXACT_RECOMPUTED_NORMALIZED_MSE_EARLIEST_TIE_SELECTED_CP_LOCK")
    _, count = normalized_CP_score(metadata["selected_tune"], checkpoint_predictions[selected], tune["payload"], tune["std"], "matched_metadata_selected_TUNE")
    checks += count
    _, count = normalized_CP_score(lock["selected_tune"], checkpoint_predictions[selected], tune["payload"], tune["std"], "matched_lock_selected_TUNE")
    checks += count
    predictions = {}
    require(set(entry["artifacts"]) == set(entry["scores"]) == {"fit", "tune", development}, "ALL_THREE_SELECTED_MATCHED_FORECAST_SPLITS")
    for split, context in contexts.items():
        saved = prior.verified_arrays(entry["artifacts"][split], check)
        require(set(saved) == {"prediction", "latent_future", "origins"}, "EXACT_SELECTED_MATCHED_HEAD_FORECAST_KEYS")
        exact(saved["origins"], context["payload"]["origins"], "matched_selected_same_split_origins")
        prediction, future = independent_head_forward(context["payload"], checkpoint_states[selected])
        exact(saved["prediction"], prediction, "matched_selected_CPU32_actual_C_forecast:" + split)
        exact(saved["latent_future"], future, "matched_selected_CPU32_latent_future:" + split)
        if split == "tune":
            exact(saved["prediction"], checkpoint_predictions[selected], "matched_selected_saved_TUNE_is_CP_exact_scored_array")
        score, count = score_full_complete(entry["scores"][split], prediction, context["values"], "matched_selected:" + split)
        checks += count
        predictions[split] = score
    perturb = entry["future_label_perturbation"]
    require(perturb["exact_prediction_bytes_equal"] is perturb["target_and_future_mask_changed"] is True and perturb["windows"] == 2
            and perturb["new_encoder_decoder_fit_or_F_forward"] is False and perturb["scope"] == "Fixed audited cached past Z, currentC and D[n]; does not replay upstream raw future or refit E/D", "SCOPED_CACHED_HEAD_FUTURE_LABEL_PERTURBATION_NOT_UPSTREAM_LEAK_PROOF")
    tiny = {name: value[:2].copy() for name, value in contexts[development]["payload"].items()}
    modified = {name: value.copy() for name, value in tiny.items()}; modified["target"][:] = np.float32(1e6); modified["observed"][:] = False
    first, _ = independent_head_forward(tiny, checkpoint_states[selected]); second, _ = independent_head_forward(modified, checkpoint_states[selected])
    exact(first, second, "independent_fixed_cached_head_future_label_perturbation")
    return {"selected_update": selected, "checkpoints": details, "scores": predictions, "score_field_checks": checks,
            "training_updates_logged_checked": 600, "checkpoint_states_checked": 5, "checkpoint_TUNE_prediction_arrays_checked": 5,
            "selected_FIT_TUNE_DEV_prediction_arrays_checked": 3, "future_label_perturbation_scope_checks": 1,
            "frozen_payload_before_after_SHA_checked": True, "independent_learning_trajectory_replayed": False,
            "whole_matched_native_cost_measured": False}


def audit_result(result_path, meter):
    check = meter.check
    path = Path(result_path).resolve()
    record = {"path": str(path), "sha256": prior.file_sha(path)}
    result = prior.verified_json(record, check)
    if result.get("status") != "COMPLETE":
        return {"status": "NOT_AUDITED_INCOMPLETE_PRODUCER", "diagnostic": record, "producer_status": result.get("status"),
            "preserved_failed_paths": result.get("failed_paths"), "completeness": result.get("completeness"),
            "scientific_assets_opened": False, "full_scope_PASS_claimed": False}
    require(result["protected_TEST_accessed"] is result["independent_holdout"] is result["new_backbone_training"] is result["new_E_F_forward_or_training"] is result["new_encoder_decoder_fit"] is result["new_GPU_execution"] is False
            and result["rho_selection_repeated"] is result["rho_selected_by_native"] is result["same_Bolt_native_lambda_or_D_bytes_claimed"] is False
            and result["native_original_static_head_weights_reused"] is result["native_head_frozen_intervention_not_reoptimized"] is result["whole_matched_native_deployment_cost_measured"] is False
            and result["strict_K_only_compression"] is result["old_static_or_global_rankK_floor_reused"] is result["full_research_resolution_claimed"] is result["individual_exact_human_method_approval_claimed"] is False
            and result["shared_Bolt_TUNE_selected_rho"] is result["native_backend_cached_D_required"] is result["past_C_decoder_coefficients_added_to_information"] is result["continuation_scope_is_documented_judgment"] is True
            and result["all_eight_condition_locks_saved_before_any_new_DEV_prediction"] is result["all_conditions_resolved_registry_saved_before_any_DEV_input_load"] is True
            and result["hard_elapsed_or_cumulative_resource_caps"] is None and result["failed_paths"] == [] and result["cost_status"].startswith("NOT_MEASURED;"), "COMPLETE_FRESH_HEAD_TRAINING_WITH_FROZEN_SELECTED_PAST_ONLY_D_SCOPE")
    expected = {"native_training_fits": 8, "completed_training_updates": 4800, "checkpoint_states_saved": 40,
        "completed_fit_tune_predictions": 16, "completed_DEV_predictions": 8, "completed_selection_locks": 8,
        "resolved_condition_lock_files": 8, "completed_source_backend_cells": 12, "future_label_perturbation_checks": 8}
    require(result["native_training_attempts"] == 8 and result["completeness"]["expected"] == result["completeness"]["actual"] == expected
            and all(result[key] == value for key,value in expected.items()) and result["full_scope_complete_claimed"] is True, "ACTUAL_EIGHT_MATCHED_NATIVE_FITS_AND_FULL_PRODUCER_COUNTERS")
    shrink = prior.verified_json(result["shrinkage_diagnostic"], check)
    shrink_pass = prior.verified_json(result["shrinkage_audit"], check)
    producer = prior.verified_json(result["producer"], check)
    fit_pass = prior.verified_json(result["fit_target_audit"], check)
    require(shrink["status"] == producer["status"] == "COMPLETE" and shrink_pass["status"] == fit_pass["status"] == "PASS"
            and shrink_pass["diagnostic"] == result["shrinkage_diagnostic"] and shrink_pass["producer"] == shrink["producer"] == result["producer"]
            and shrink_pass["fit_target_audit"] == shrink["fit_target_audit"] == result["fit_target_audit"] and fit_pass["producer"] == result["producer"], "EXACT_COMPLETE_SHRINKAGE_AND_ORIGINAL_INPUT_PASS_BINDINGS")
    require(result["shrinkage_diagnostic"]["sha256"] == "aa553512965bf9253fae1983424e2fcb4f2916cc95ec12e9ed671ae325dbd532"
            and result["shrinkage_audit"]["sha256"] == "145999ea8bc7068eba08f08039287a33097cdb4e9ec8dab002f4ebc7630fdda9", "EXACT_IMMUTABLE_BOLT_SELECTED_RHO_BASE_SCIENCE")
    require(all(shrink_pass["totals"].get(key) == value for key,value in {"positive_rho_backend_states_checked":88,"window_fits_checked":13504,"channel_slots_checked":187664,
        "FIT_TUNE_positive_rho_prediction_arrays_checked":120,"selected_DEV_prediction_arrays_checked":12,"selection_locks_checked":4,"score_field_checks":6960}.items()), "FULL_ACCEPTED_SHRUNK_DECODER_SCOPE")
    for audit in (shrink_pass, fit_pass):
        require(prior.file_sha(audit["audit_source"]["path"]) == audit["audit_source"]["sha256"], "IMMUTABLE_PREDECESSOR_INDEPENDENT_AUDIT_SOURCE")
    scope = prior.verified_json(result["scope"], check)
    bind_names=("shrinkage_diagnostic","shrinkage_audit","producer","fit_target_audit")
    require(scope["bindings"] == {key:result[key] for key in bind_names} and scope["status"] == "CONTINUING_GOAL_MATCHED_PAST_DECODER_NATIVE_TRAINING_JUDGMENT"
            and scope["individual_exact_human_method_approval_claimed"] is False and scope["new_native_fits"] == 8 and scope["updates_each"] == 600
            and scope["CP_each"] == list(CHECKPOINTS) and scope["seeds"] == list(SEEDS) and scope["new_backbone_or_encoder_decoder_training"] is False
            and scope["decoder_and_rho_selected_already_by_Bolt_TUNE"] is scope["NativeTorch32_backend_specific_selected_D_and_cached_past"] is scope["all_eight_resolved_locks_before_DEV"] is True
            and scope["native_specific_rho_selection"] is scope["strict_K_only_interface"] is scope["whole_matched_native_cost_measured"] is scope["independent_holdout"] is scope["old_static_or_global_rankK_floor_reused"] is scope["F_necessity_proven"] is False, "MATCHED_HEAD_CONTINUATION_JUDGMENT_NOT_BACKBONE_OR_RHO_REFIT")
    guarded=[record,result["scope"],*[result[key] for key in bind_names],shrink_pass["audit_source"],fit_pass["audit_source"],*result["source_records"],*result["input_records"],*result["protected_prior_records"]]
    for item in guarded:
        check();require(prior.file_sha(item["path"]) == item["sha256"], "ORIGINAL_MATCHED_INPUT_SOURCE_CP_MODEL_CACHE_OR_SELECTION_HASH_UNCHANGED")
    for item in result["source_snapshots"]:
        require(prior.file_sha(item["snapshot_path"]) == item["sha256"], "EXECUTED_MATCHED_SOURCE_SNAPSHOT_BYTES")
        guarded.append({"path":item["snapshot_path"],"sha256":item["sha256"]})
    source_check=source_contract(result["source_records"])
    require(scope["plan"] in result["source_records"], "MATCHED_SCOPE_PLAN_SOURCE_BINDING")
    registry=prior.verified_json(result["selection_lock"],check)
    require(registry["complete_fit_tune_conditions"] == 8 and registry["failed_paths_at_seal"] == [], "COMPLETE_ALL_EIGHT_SEALED_MATCHED_MODELS_REGISTRY")
    locks={(data,route,str(seed)):result["condition_selection_locks"][data][route][str(seed)] for data in DATASETS for route in ("PCA_PINV","FIT_TARGET_PINV") for seed in SEEDS}
    event_check=check_registry_events(result["execution_events"],registry,locks)
    guarded.append(result["selection_lock"])
    totals={"native_fits_checked":0,"training_updates_logged_checked":0,"checkpoint_states_checked":0,"checkpoint_TUNE_prediction_arrays_checked":0,
        "selected_FIT_TUNE_DEV_prediction_arrays_checked":0,"source_backend_contexts_checked":0,"preserved_control_prediction_arrays_checked":0,
        "persistence_baseline_contexts_computed":0,"unique_persistence_split_cells_computed":0,"selection_locks_checked":0,"resolved_registry_conditions_checked":8,
        "future_label_perturbation_scope_checks":0,"score_field_checks":0,"protected_prior_file_hashes_checked":len(result["protected_prior_records"])}
    report={"status":"PASS","diagnostic":record,**{key:result[key] for key in bind_names},"scope":result["scope"],"selection_lock":result["selection_lock"],
        "condition_selection_locks":result["condition_selection_locks"],"source_contract":source_check,"registry_event_checks":event_check,"datasets":{},"totals":totals,
        "protected_TEST_accessed":False,"new_learning_trajectory_or_GPU_or_F_forward_executed":False,"independent_saved_head_CPU32_arithmetic_replayed":True,
        "new_encoder_or_decoder_fit_executed":False,"rho_reselected_by_native":False,"whole_matched_native_cost_measured":False,
        "original_static_head_warm_start":False,"same_Bolt_native_D_or_lambda_bytes_claimed":False,"past_C_coefficients_added_to_decoder_information":True,
        "independent_generalization_or_F_necessity_proven":False,"posthoc_CP0_or_DEV_score_based_winner_change":False}
    unique_baselines=set()
    require(set(result["datasets"]) == set(DATASETS), "EXACT_TWO_MATCHED_NATIVE_DATASETS")
    for data,(channels,k,development,dev_count) in DATASETS.items():
        row=result["datasets"][data]; original=producer["datasets"][data]
        require(row["channels"] == channels and row["K"] == k and row["development_split"] == development and set(row["routes"]) == {"PCA_PINV","FIT_TARGET_PINV"}, "EXACT_MATCHED_DATASET_K_C_ROUTES")
        out={"channels":channels,"K":k,"development_split":development,"routes":{}};report["datasets"][data]=out
        for route in ("PCA_PINV","FIT_TARGET_PINV"):
            condition=row["routes"][route]
            old_rho_lock_record=shrink["selection_locks"][data][route]
            old_rho_lock=prior.verified_json(old_rho_lock_record,check)
            require(condition["old_Bolt_rho_selection_lock"] == old_rho_lock_record and condition["rho"] == old_rho_lock["selected_rho"]
                    and condition["selected_candidate"] == old_rho_lock["selected_candidate"], "ORIGINAL_BOLT_TUNE_RHO_LOCK_NEVER_RESELECTED_BY_NATIVE")
            guarded.append(old_rho_lock_record)
            output_condition={"rho":condition["rho"],"selected_candidate":condition["selected_candidate"],"inputs":{},"native":{}};out["routes"][route]=output_condition
            contexts={}
            require(set(condition["inputs"]) == {"fit","tune",development}, "ALL_THREE_MATCHED_NATIVE_SOURCE_CONTEXTS")
            for split,count in (("fit",256),("tune",64),(development,dev_count)):
                entry=condition["inputs"][split]
                require(entry["input_source"] == original["inputs"][split], "MATCHED_EXACT_ORIGINAL_SPLIT_SOURCE")
                values=prior.verified_arrays(entry["input_source"],check);prior.check_raw_split(values,channels=channels,windows=count)
                context=audit_context(entry,shrink["datasets"][data]["splits"][split]["routes"][route],condition,split,values,check)
                contexts[split]=context
                output_condition["inputs"][split]={"controls":context["scores"],"persistence_current_C_baseline":context["persistence_current_C_baseline"],
                    "source_input":entry["input_source"],"fixed_input_and_controls":context["archived_record"],"selected_native_state_source":entry["selected_native_state_source"]}
                totals["source_backend_contexts_checked"]+=1;totals["preserved_control_prediction_arrays_checked"]+=4;totals["score_field_checks"]+=context["score_field_checks"]
                totals["persistence_baseline_contexts_computed"]+=1;unique_baselines.add((data,split))
            require(set(condition["native"]) == {str(seed) for seed in SEEDS}, "EXACT_TWO_FRESH_NATIVE_SEEDS")
            for seed in SEEDS:
                check();key=(data,route,str(seed));lock=prior.verified_json(locks[key],check);guarded.append(locks[key])
                require(lock["status"] == "LOCKED" and (lock["dataset"],lock["route"],lock["seed"]) == key
                        and lock["rho"] == condition["rho"] and lock["old_Bolt_rho_selection_lock"] == old_rho_lock_record
                        and lock["rho_selected_by_native"] is lock["DEV_predictions_computed_before_lock"] is False
                        and lock["sealed_unix"] <= registry["sealed_unix"], "MATCHED_CONDITION_CP_LOCK_BEFORE_REGISTRY_AND_DEV_NO_RHO_SELECTION")
                audited=audit_native(condition["native"][str(seed)],contexts,lock,seed,channels,k,development,check)
                output_condition["native"][str(seed)]=audited
                totals["native_fits_checked"]+=1;totals["selection_locks_checked"]+=1;totals["score_field_checks"]+=audited["score_field_checks"]
                for name in ("training_updates_logged_checked","checkpoint_states_checked","checkpoint_TUNE_prediction_arrays_checked","selected_FIT_TUNE_DEV_prediction_arrays_checked","future_label_perturbation_scope_checks"):
                    totals[name]+=audited[name]
    totals["unique_persistence_split_cells_computed"]=len(unique_baselines)
    require(all(totals[key] == value for key,value in {"native_fits_checked":8,"training_updates_logged_checked":4800,"checkpoint_states_checked":40,"checkpoint_TUNE_prediction_arrays_checked":40,
        "selected_FIT_TUNE_DEV_prediction_arrays_checked":24,"source_backend_contexts_checked":12,"preserved_control_prediction_arrays_checked":48,
        "persistence_baseline_contexts_computed":12,"unique_persistence_split_cells_computed":6,"selection_locks_checked":8,"resolved_registry_conditions_checked":8,
        "future_label_perturbation_scope_checks":8,"score_field_checks":2064}.items()), "FULL_FRESH_MATCHED_NATIVE_STATE_PREDICTION_LOCK_BASELINE_AUDIT_COUNTS")
    require(result["original_source_cache_model_and_selection_SHA_unchanged"] is True, "MATCHED_PRODUCER_DECLARED_ORIGINAL_SOURCE_CACHE_SELECTION_UNCHANGED")
    for item in guarded:
        check();require(prior.file_sha(item["path"]) == item["sha256"], "MATCHED_AUDITED_SOURCE_OLD_MODEL_INPUT_D_CACHE_AND_LOCK_UNCHANGED_AFTER_EXECUTION")
    receipt=prior.verified_json(result["resource_terminal"],check)
    require(receipt["status"] == "COMPLETE" and receipt["elapsed_or_cumulative_budget_caps"] is None, "ACTUAL_COMPLETE_MATCHED_HEAD_TRAINING_RECEIPT")
    return report


def execute(result_path,output_path):
    from continuation_runtime_v1 import Meter,write_new
    output=Path(output_path).resolve();require(not output.exists(),"FRESH_INDEPENDENT_MATCHED_NATIVE_AUDIT_FILE");output.parent.mkdir(parents=True,exist_ok=True)
    meter=Meter("audit_matched_native_"+output.stem,gpu=False)
    source={"path":str(Path(__file__).resolve()),"sha256":prior.file_sha(__file__)};failure=None
    try:
        meter.protected_check();initialize_numpy()
        sys.path.insert(0,str(ROOT/"research/smallk_failure_followup_20261014"))
        from prepare_inputs import install_protected_test_guard
        install_protected_test_guard()
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2):report=audit_result(result_path,meter)
        meter.protected_check();require(prior.file_sha(__file__)==source["sha256"],"INDEPENDENT_MATCHED_AUDIT_SOURCE_UNCHANGED");json.dumps(report,allow_nan=False)
    except Exception as error:
        failure={"type":type(error).__name__,"message":str(error)};report={"status":"FAILED_PRESERVED","failure":failure,"full_scope_PASS_claimed":False}
    finally:
        report["audit_source"]=source;report["resource_terminal"]=meter.close(report["status"],failure);write_new(output,report)
    print(json.dumps({"status":report["status"],"output":str(output),"totals":report.get("totals"),"failure":failure},ensure_ascii=False),flush=True)
    return 0 if report["status"]=="PASS" else 2


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--result",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args();raise SystemExit(execute(args.result,args.output))
