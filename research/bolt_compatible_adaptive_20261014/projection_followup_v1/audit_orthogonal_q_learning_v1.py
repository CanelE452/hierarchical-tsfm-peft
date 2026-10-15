"""Saved Q-learning arithmetic/selection audit; no F, VJP or optimizer replay."""
from pathlib import Path
import argparse
import hashlib
import json
import math

import audit_orthogonal_basis_controls_v1 as common

base, prior, require = common.base, common.prior, common.require
HERE, ROOT = common.HERE, common.ROOT
STATES, SEEDS, CPS, LAGS = common.STATES, (92601, 92602), (0, 100, 200, 400, 600), common.LAGS
RESULT_SHA = "e1067bce2ef20381477c5345d5dd5566528a4340b69ba0b19bbcdd717d141fcc"
CONTROLLER_SHA = "b5d8e7f6649ef5ee6c23eb6ad92e49b4d1f00a59bec6fd9811fe5a724b460e41"
MODULE_SHA = "ac515b9cb064e80d89c30a252f7bed4186c81eceb0c1c9d6db1fdbe5cab41491"
PARENT_SHA = {**common.PARENT_SHA, "fixed_controls_diagnostic": common.RESULT_SHA,
    "fixed_controls_audit": "156e7513b200215f574b511828428c3e11c5b8a79ab2f79c85448a0eb87b4977",
    "fixed_controls_auditor_source": "4d16612c65b9b1ee3e33a95639b87992b73ce6aa3991dea4c922f58ae17d9f43"}
REFS = tuple(k for k in common.REFS if k != "control_registry") + ("fixed_controls_diagnostic", "fixed_controls_audit", "fixed_controls_auditor_source", "fixed_control_registry", "selection_registry")
FLAGS = {"trainable_parameter_names": ["theta"], "theta_parameters": 6, "E_requires_grad": False, "D_requires_grad": False,
    "E_grad_is_None": True, "D_grad_is_None": True, "center_requires_grad": False, "F_requires_grad": False,
    "F_grad_is_None": True, "gradient_clipping_parameter_scope": "THETA_ONLY", "learned_zero_theta_bypass": False}
RECIPE = {"optimizer": "AdamW", "learning_rate": .001, "weight_decay": 0., "batch_size": 4, "clip_norm": 1., "updates_per_condition": 600, "checkpoint_updates": list(CPS), "theta_parameters": 6}
np = None


def initialize_numpy():
    global np
    common.initialize_numpy(); np = common.np


def flags(row):
    require({key: row[key] for key in FLAGS} == FLAGS, "ONLY_THETA_TRAINABLE_F_GRAD_NONE_RECORDED")


def independent_Q(theta):
    require(theta.dtype == np.float32 and theta.shape == (6,) and np.isfinite(theta).all(), "FINITE_SIX_FP32_THETA")
    skew = np.zeros((4, 4), np.float64); skew[np.triu_indices(4, 1)] = theta.astype(np.float64); skew -= skew.T
    values, vectors = np.linalg.eigh(1j*skew)
    exponential = (vectors*np.exp(-1j*values)[None, :])@vectors.conj().T
    require(np.abs(exponential.imag).max() < 1e-12, "REAL_SKEW_EXPONENTIAL")
    return exponential.real


def check_basis(theta, q, recorded):
    require(q.dtype == np.float32 and q.shape == (4, 4) and np.isfinite(q).all(), "FINITE_FP32_4X4_Q")
    prior.close(q, independent_Q(theta), "independent_Hermitian_spectral_exp_skew_theta", atol=2e-6, rtol=2e-6)
    q64 = q.astype(np.float64); orthogonal = float(np.abs(q64.T@q64-np.eye(4)).max()); determinant = float(np.linalg.det(q64))
    require(orthogonal <= 1e-5 and abs(determinant-1.) <= 1e-5, "VALID_LEARNED_SO4")
    require(recorded["finite"] is recorded["old_past_and_decoder_unchanged"] is True and recorded["identity_bypass"] is recorded["input_information_added"] is False, "LEARNED_Q_NO_BYPASS_NO_ADDED_INFORMATION")
    prior.close([recorded[k] for k in ("orthogonality_max_abs", "determinant", "theta_L2")], [orthogonal, determinant, float(np.linalg.norm(theta.astype(np.float64)))], "basis_diagnostics", atol=1e-12, rtol=1e-12)
    return {"theta_FP32": theta.tolist(), "Q_FP32": q.tolist(), "basis_diagnostics": recorded, "independent_Q_exponential_atol": 2e-6, "independent_Q_exponential_rtol": 2e-6}


def selection_score(prediction, target, observed):
    require(prediction.shape == target.shape == observed.shape and observed.dtype == np.bool_ and np.isfinite(prediction).all() and np.isfinite(target[observed]).all(), "FINITE_CANONICAL_SELECTION_SCORE")
    counts = observed.sum((0, 1)); active = counts > 0; require(active.any(), "OBSERVED_SELECTION_CHANNEL")
    error = np.where(observed, prediction.astype(np.float64)-target.astype(np.float64), 0.)
    mse = np.square(error).sum((0, 1))/np.maximum(counts, 1); mae = np.abs(error).sum((0, 1))/np.maximum(counts, 1)
    return {"mse": float(mse[active].mean()), "mae": float(mae[active].mean()), "observed_counts": counts.tolist(), "active_channels": int(active.sum()),
        "per_channel_mse": [float(v) if a else None for v, a in zip(mse, active)], "per_channel_mae": [float(v) if a else None for v, a in zip(mae, active)]}


def check_cell(row, arrays, independent, raw, cached, frozen, theta, q, *, cp0=False):
    n, length, c = raw["history"].shape; h = raw["target"].shape[1]
    shapes = {"prediction": (n,h,c), "old_past": (n,length,4), "rotated_past": (n,length,4), "F_rotated_future": (n,h,4), "unrotated_future": (n,h,4), "transport_future": (n,h,4), "transport_prediction": (n,h,c)}
    context = ("target", "observed", "raw_last", "origins", "mean", "std")
    require(set(arrays) == set(shapes)|set(context)|{"theta_FP32", "Q_FP32"} and set(independent) == {"independent_NumPy_future", "independent_NumPy_prediction"}, "EXACT_Q_FORWARD_ARTIFACT_MEMBERS")
    for key, shape in shapes.items(): require(arrays[key].dtype == np.float32 and arrays[key].shape == shape and np.isfinite(arrays[key]).all(), "FINITE_FP32_SAVED_COMPONENT:"+key)
    for key in context: base.exact(arrays[key], raw[key], "original_context_"+key)
    base.exact(arrays["theta_FP32"], theta, "output_theta_matches_CP"); base.exact(arrays["Q_FP32"], q, "output_Q_matches_CP")
    future = np.ascontiguousarray(arrays["F_rotated_future"]@q.T, dtype=np.float32)
    predicted = raw["raw_last"][:,None]+base.delta_features(arrays["old_past"], future, LAGS)@frozen["D"]
    transported = (cached["latent_future"]@q)@q.T
    transport_C = raw["raw_last"][:,None]+base.delta_features(cached["latent_history"], transported, LAGS)@frozen["D"]
    base.exact(independent["independent_NumPy_future"], future, "saved_NumPy_inverse_Q_future")
    prior.close(independent["independent_NumPy_prediction"], predicted, "saved_NumPy_original_lag_D", atol=1e-5, rtol=2e-6)
    checks = {"old_past_vs_cached": (arrays["old_past"],cached["latent_history"]), "NumPy_fresh_E": (arrays["old_past"],(raw["history"]-frozen["center"])@frozen["E"]), "NumPy_rotated_past": (arrays["rotated_past"],arrays["old_past"]@q), "NumPy_inverse_Q_future": (arrays["unrotated_future"],future), "NumPy_original_lag_D": (arrays["prediction"],predicted), "analytic_transport_future": (arrays["transport_future"],cached["latent_future"]), "analytic_transport_C": (arrays["transport_prediction"],cached["prediction"])}
    if cp0: checks.update(CP0_future_vs_cached=(arrays["unrotated_future"],cached["latent_future"]), CP0_C_vs_cached=(arrays["prediction"],cached["prediction"]))
    require(set(row["checks"]) == set(checks), "ALL_FROZEN_DEPLOYMENT_PARITY_CHECKS")
    for key, pair in checks.items(): common.check_parity(row["checks"][key], *pair, key)
    prior.close(arrays["transport_future"], transported, "independent_analytic_transport", atol=1e-5, rtol=2e-6)
    prior.close(arrays["transport_prediction"], transport_C, "independent_transport_old_D", atol=1e-5, rtol=2e-6)
    require(row["status"] == "COMPLETE" and row["CP0_parity_required"] == cp0 and row["learned_zero_theta_bypass"] is row["future_truth_is_forward_input"] is False and row["FIT_future_is_training_response_only"] is True and row["canonical_batch_size"] == 32, "CANONICAL_SAVED_FORWARD_SCOPE")
    complete = np.broadcast_to(raw["observed"].all(2,keepdims=True),raw["observed"].shape); fields = 0
    for name, prediction, recorded in (("actual",arrays["prediction"],{"all_observed":row["scores_all_observed"],"complete_C":row["scores_complete_mask"]}), ("baseline_Bolt",cached["prediction"],row["baseline_Bolt_scores"]), ("RAW_BOLT",raw["raw_forecast"],row["RAW_BOLT_scores"])):
        for mask_name, mask in (("all_observed",raw["observed"]),("complete_C",complete)):
            if mask.any(): fields += prior.compare_scores(recorded[mask_name],prior.scores(prediction,raw["target"],mask,raw["std"]),name+mask_name)
            else: require(recorded[mask_name] == {"status":"NO_COMPLETE_C"}, "NO_COMPLETE_MASK_NOT_ZERO_ERROR")
    require(row["complete_target_positions"] == int(complete[:,:,0].sum()), "COMPLETE_TARGET_POSITION_COUNT")
    for key, actual, baseline in (("latent_commutator",arrays["unrotated_future"],cached["latent_future"]),("C_commutator",arrays["prediction"],cached["prediction"])):
        computed = common.stats(actual,baseline); require(set(row[key]) == set(computed), "COMMUTATOR_FIELDS")
        for field in computed: prior.close(row[key][field],computed[field],key+field,atol=1e-10,rtol=1e-10)
    return {"score_fields_checked":fields, "scores_all_observed":row["scores_all_observed"], "scores_complete_mask":row["scores_complete_mask"], "baseline_Bolt_scores":row["baseline_Bolt_scores"], "RAW_BOLT_scores":row["RAW_BOLT_scores"], "latent_commutator":row["latent_commutator"], "C_commutator":row["C_commutator"], "CP0_parity_required":cp0, "F_forward_independently_replayed":False}


def check_logs(row, logs, fit, parent_order):
    require(len(logs) == 600 and row["training_updates_completed"] == 600 and row["post_optimizer_frozen_flag_checks"] == 600 and row["frozen_state_checkpoint_byte_checks"] == 5 and row["frozen_bytes_checked_at_CP_and_final_only"] is True and row["independent_per_step_backward_optimizer_or_frozen_bytes_replayed"] is False, "EXACT_RECORDED_UPDATE_AND_FROZEN_CHECK_COUNTS")
    for update, event in enumerate(logs,1):
        require(event["stage"] == "TRAINING_UPDATE" and event["update"] == update, "ORDERED_SIX_HUNDRED_LOGS"); flags(event)
        gradient = np.asarray(event["theta_gradient_before_clip"],np.float32); theta = np.asarray(event["theta_after_optimizer_step"],np.float32)
        require(gradient.shape == theta.shape == (6,) and np.isfinite(gradient).all() and np.isfinite(theta).all() and math.isfinite(event["loss"]) and event["loss"] >= 0., "FINITE_LOGGED_THETA_GRADIENT_LOSS")
        norm = event["theta_gradient_norm_before_clip"]; require(math.isfinite(norm) and norm >= 0. and event["global_gradient_norm_before_clip"] == norm and event["clipped"] == (norm > 1.), "THETA_ONLY_LOGGED_CLIP_DECISION")
        prior.close(norm,np.linalg.norm(gradient.astype(np.float64)),"logged_theta_gradient_norm",atol=1e-8,rtol=2e-6)
    prior.close(row["global_clipping_fraction"],sum(v["clipped"] for v in logs)/600,"clipping_fraction",atol=1e-12,rtol=0.)
    require(row["FIT_channel_observed_counts"] == fit["observed"].sum((0,1)).tolist() and row["loss"] == "N_fit/B/C_active * sum_c SSE_batch_c/N_FIT_c; fixed entire FIT observed counts", "GLOBAL_FIT_MACRO_DENOMINATORS")
    order = np.random.default_rng(row["seed"]).integers(0,len(fit["history"]),size=(600,4))
    require(row["order_sha256"] == parent_order == hashlib.sha256(order.tobytes()).hexdigest(), "SAME_PARENT_MINIBATCH_ORDER")
    flags(row["final_frozen_status"]); return order


def check_probe(recorded, arrays, order, first_log, *, length=512, horizon=48):
    shapes = {"theta_gradient_full":(6,),"theta_gradient_via_F_input":(6,),"theta_gradient_via_inverse_Q":(6,),"theta_gradient_closure_residual":(6,),"rotated_F_input_gradient":(4,length,4),"unrotated_future_gradient":(4,horizon,4),"theta_FP32":(6,),"Q_FP32":(4,4)}
    require(set(arrays) == set(shapes)|{"FIT_window_indices"}, "EXACT_FIRST_BATCH_GRADIENT_MEMBERS")
    for key, shape in shapes.items(): require(arrays[key].dtype == np.float32 and arrays[key].shape == shape and np.isfinite(arrays[key]).all(), "FINITE_FIRST_BATCH_GRADIENT_ARRAY:"+key)
    base.exact(arrays["FIT_window_indices"],order[0],"first_training_batch_order"); base.exact(arrays["theta_FP32"],np.zeros(6,np.float32),"zero_initial_theta"); base.exact(arrays["Q_FP32"],np.eye(4,dtype=np.float32),"first_Q_identity")
    full, through, inverse = (arrays[key] for key in ("theta_gradient_full","theta_gradient_via_F_input","theta_gradient_via_inverse_Q"))
    base.exact(arrays["theta_gradient_closure_residual"],full-through-inverse,"FP32_saved_gradient_closure_residual")
    require(np.allclose(full,through+inverse,atol=1e-6,rtol=1e-4) and recorded["gradient_closure_atol"] == 1e-6 and recorded["gradient_closure_rtol"] == 1e-4 and recorded["status"] == "PASS" and recorded["finite"] is recorded["gradient_closure_pass"] is True, "SAVED_GRADIENT_CLOSURE_FIXED_TOLERANCES")
    for name,key in (("full_theta_gradient_norm","theta_gradient_full"),("theta_F_input_chain_norm","theta_gradient_via_F_input"),("theta_inverse_Q_contribution_norm","theta_gradient_via_inverse_Q"),("rotated_F_input_gradient_norm","rotated_F_input_gradient"),("closure_residual_norm","theta_gradient_closure_residual")):
        prior.close(recorded[name],np.linalg.norm(arrays[key].astype(np.float64)),name,atol=1e-10,rtol=1e-10)
    prior.close(first_log["theta_gradient_before_clip"],full,"first_full_gradient_matches_training_backward",atol=1e-7,rtol=2e-6)
    require(recorded["F_input_path_nonzero_observed"] == bool(np.linalg.norm(through.astype(np.float64)) > 0) and recorded["nonzero_total_theta_gradient_alone_proves_F_input_path"] is False and recorded["zero_gradient_does_not_prove_missing_autograd_path"] is recorded["probe_reuses_first_training_forward"] is True and recorded["optimizer_updates_in_probe"] == 0, "GRADIENT_PROBE_LIMITED_SCOPE")
    return {**{k:v for k,v in recorded.items() if k != "artifact"}, "artifact":recorded["artifact"], **{key:arrays[key].tolist() for key in shapes if key.startswith("theta_gradient")}, "independent_F_VJP_or_full_backward_replayed":False}


def check_registry(result, registry, load):
    require(result["all_6_locks_before_DEV"] is True and result["resolved_selection_locks"] == 6 and registry["status"] == "ALL_SIX_RESOLVED_BEFORE_DEV" and registry["DEV_opened_before_seal"] is False and len(registry["conditions"]) == 6, "ALL_SIX_RESOLVED_BEFORE_DEV")
    locks = {}
    for item in registry["conditions"]:
        key = item["state"],item["seed"]; require(key in {(s,t) for s in STATES for t in SEEDS} and key not in locks and item["status"] == "COMPLETE", "SIX_UNIQUE_COMPLETE_LEARNING_CONDITIONS")
        row = result["states"][key[0]]["learning"][str(key[1])]; common.same(item["selection_lock"],row["selection_lock"]); lock = load(item["selection_lock"])
        require(lock["DEV_opened_before_lock"] is False and lock["sealed_unix"] <= registry["sealed_unix"] and {k:v for k,v in lock.items() if k not in ("DEV_opened_before_lock","sealed_unix")} == {k:v for k,v in row.items() if k not in ("DEV","selection_lock")}, "EXACT_IMMUTABLE_PRE_DEV_CONDITION_LOCK")
        locks[key] = lock
    return locks


def check_selection(row, checkpoints):
    require([v["update"] for v in checkpoints] == list(CPS) and row["checkpoint_count"] == 5, "ALL_FIVE_CHECKPOINTS_PRESERVED")
    selected = min(checkpoints,key=lambda v:(v["tune"]["mse"],v["update"]))
    require(row["selected_update"] == selected["update"] and row["selected_tune"] == selected["tune"], "OWN_TUNE_EXACT_EARLIER_CP_SELECTION")
    common.same(row["selected_state"],selected["state"])
    require(row["selected_basis_diagnostics"] == selected["basis_diagnostics"] and row["TUNE"] == {k:v for k,v in selected.items() if k not in ("state","basis_diagnostics","frozen_status","E_D_center_exact_initial_bytes")}, "SELECTED_TUNE_REUSES_IMMUTABLE_CP_NO_REFIT")
    return selected


def audit_result(path, meter):
    require(isinstance(RESULT_SHA,str) and len(RESULT_SHA) == 64, "ACTUAL_Q_LEARNING_RESULT_NOT_REGISTERED")
    check = meter.check; binding = base.file_record(path); require(binding["sha256"] == RESULT_SHA, "EXACT_ACTUAL_Q_LEARNING_RESULT")
    result = prior.verified_json(binding,check)
    require(result["status"] == "COMPLETE" and not result["failed_paths"] and set(result["states"]) == set(STATES), "ALL_SIX_Q_LEARNING_COMPLETE")
    for key,digest in PARENT_SHA.items(): require(result[key]["sha256"] == digest,"EXACT_ACCEPTED_PARENT:"+key)
    require(result["controller_source"]["sha256"] == CONTROLLER_SHA and result["module_source"]["sha256"] == MODULE_SHA and result["training_recipe"] == RECIPE and result["selection_rule"] == "own canonical32 normalized TUNE macro MSE, exact ties earliest CP", "EXACT_EXECUTED_Q_LEARNING_SOURCE_RECIPE_SELECTION")
    for key in ("new_backbone_training","whole_deployment_cost_measured","new_output_space_computed","protected_TEST_accessed","independent_holdout","strict_K_only"): require(result[key] is False,"Q_LEARNING_SCOPE:"+key)
    expected = {"Q_training_fits":6,"training_updates_completed":3600,"checkpoint_states_saved":30,"DEV_evaluations_completed":6,"new_F_forward_calls":3714,"new_F_forward_attempts":3714,"new_F_model_objects_loaded":1,"new_E_D_training_updates":0,"new_optimizer_updates":3600,"new_Q_learning_updates":3600,"checkpoint_artifact_hashes_rechecked":90}
    require(all(result[k] == v for k,v in expected.items()),"EXACT_Q_LEARNING_EXECUTION_COUNTS")
    guarded = [binding,*[result[k] for k in REFS],*result["source_records"],*result["input_records"],*result["protected_prior_records"]]
    for item in guarded:
        common.bound(item,check)
        if "snapshot_path" in item: require(prior.file_sha(item["snapshot_path"]) == item["sha256"],"EXECUTED_SOURCE_SNAPSHOT_BYTES")
    parent = prior.verified_json(result["latent_calibration_diagnostic"],check); native = prior.verified_json(result["matched_l7_native_diagnostic"],check); controls = prior.verified_json(result["fixed_controls_diagnostic"],check); control_audit = prior.verified_json(result["fixed_controls_audit"],check); full = prior.verified_json(result["full_gradient_parent"],check)
    require(control_audit["status"] == "PASS","ACCEPTED_FIXED_Q_AUDIT"); common.same(control_audit["diagnostic"],result["fixed_controls_diagnostic"]); common.same(control_audit["audit_source"],result["fixed_controls_auditor_source"]); common.same(result["fixed_control_registry"],controls["control_registry"])
    for key in common.REFS:
        if key not in ("controller_source","module_source","plan","control_registry"): common.same(result[key],controls[key])
    require(result["F_before"] == result["F_after"] == controls["F_before"] == controls["F_after"] == parent["F_parent_signatures_reused"]["before"] == parent["F_parent_signatures_reused"]["after"] and result["runtime_behavior_contract"] == controls["runtime_behavior_contract"] == full["runtime_behavior_contract"],"F_DUAL_HASH_AND_MEDIAN_RUNTIME_UNCHANGED")
    for key in ("model_binary","model_config"):
        common.same(result[key+"_before"],result[key+"_after"]); common.same(result[key+"_before"],controls[key+"_before"]); common.bound(result[key+"_before"],check); common.bound(result[key+"_after"],check)
    registry = prior.verified_json(result["selection_registry"],check); locks = check_registry(result,registry,lambda item:prior.verified_json(item,check))
    from audit_fixed_l7_latent_calibration_v1 import receipt_check
    guarded += receipt_check(result["resource_terminal"],"COMPLETE",check)
    receipt = result["resource_terminal"]; require(receipt["error"] is None and receipt["started_unix"] <= registry["sealed_unix"] <= receipt["ended_unix"],"SELECTION_REGISTRY_WITHIN_COMPLETE_PROCESS_RECEIPT")
    output = Path(binding["path"]).parent; cache = {}; produced_paths = set()
    def load(item):
        key = item["path"],item["sha256"]
        if key not in cache: cache[key] = prior.verified_arrays(item,check)
        return cache[key]
    def produced(item,name,*,arrays=True):
        target = Path(item["path"]).resolve(); require(target == output/name and target not in produced_paths,"UNIQUE_REGISTERED_PRODUCER_ARTIFACT_PATH"); produced_paths.add(target)
        if arrays: return load(item)
        common.bound(item,check); return [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    totals = {"Q_training_conditions_checked":0,"training_update_logs_checked":0,"checkpoint_states_checked":0,"checkpoint_TUNE_prediction_evaluations_checked":0,"selected_FIT_prediction_evaluations_checked":0,"selected_TUNE_entries_reused_checked":0,"selected_DEV_prediction_evaluations_checked":0,"first_batch_gradient_closures_checked":0,"selection_locks_checked":len(locks),"produced_NPZ_artifacts_checked":0,"actual_F_calls_registered":0,"score_fields_checked":0,"commutator_summaries_checked":0,"frozen_E_D_center_sources_checked":0,"same_parent_orders_checked":0,"protected_prior_file_hashes_checked":len(result["protected_prior_records"])}
    report = {"status":"PASS","diagnostic":binding,**{k:result[k] for k in REFS},"totals":totals,"states":{},"partial_paths":[],"protected_TEST_accessed":False,"independent_holdout":False,"independent_F_GPU_optimizer_or_timer_replayed":False,"independent_F_input_VJP_replayed":False,"DEV_access_events_independently_replayed":False,"scope":"Same agent prepared producer and auditor; ROOT reviews both. Independent NumPy skew exponential, saved inverse-Q/old lag-specific-origin decoder/transport, scores, gradients closure, recorded logs/order and immutable selection/source/receipt checks; shared accepted audit helpers. No F output, VJP, optimizer step, per-step loss, or full training replay. Sealed records plus frozen source control flow support before-DEV order; no independent access-event replay. 600 updates is a finite budget, not optimizer exhaustion; DEV is adaptive, TEST untouched, no new output geometry or generalization proof."}
    for state in STATES:
        owner = result["states"][state]; source = native["states"][state]["frozen_state_source"]; common.same(source,owner["frozen_state_source"]); common.same(source,controls["states"][state]["frozen_state_source"]); frozen = load(source)
        require(set(frozen) == {"E","D","center"} and frozen["E"].shape == (17,4) and frozen["D"].shape == (28,17) and frozen["center"].shape == (17,) and all(v.dtype == np.float32 and np.isfinite(v).all() for v in frozen.values()),"ORIGINAL_FROZEN_C17_K4_L7")
        require(owner["frozen_state_value_sha256"] == base.state_hash(frozen) and owner["E_previously_supervised_by_FIT_future"] == (state != "RECON_CP0") and owner["D_previously_supervised_by_FIT_future"] is False and set(owner["learning"]) == {str(v) for v in SEEDS},"EXACT_FROZEN_HISTORY_AND_TWO_ORDER_SEEDS")
        totals["frozen_E_D_center_sources_checked"] += 1; report["states"][state] = {"frozen_state_source":source,"learning":{}}
        contexts = {}
        def context(entry,split):
            common.same(entry["input_source"],parent["states"][state]["families"]["SCALE_ONLY"][{"fit":"FIT","tune":"TUNE","val":"DEV"}[split]]["input_source"]); common.same(entry["cached_Bolt_source"],native["states"][state]["Bolt"][split]["artifact"])
            if split not in contexts:
                raw,cached = load(entry["input_source"]),load(entry["cached_Bolt_source"]); prior.check_raw_split(raw,channels=17,windows={"fit":256,"tune":64,"val":30}[split])
                for key in ("target","observed","raw_last","origins","mean","std"): base.exact(cached[key],raw[key],"baseline_original_context_"+key)
                require(cached["latent_history"].shape == (len(raw["history"]),512,4) and cached["latent_future"].shape == (len(raw["history"]),48,4) and cached["prediction"].shape == raw["target"].shape and all(cached[k].dtype == np.float32 and np.isfinite(cached[k]).all() for k in ("latent_history","latent_future","prediction")),"FINITE_MATCHED_BASELINE_CACHE")
                contexts[split] = raw,cached
            return contexts[split]
        for seed in SEEDS:
            row = owner["learning"][str(seed)]; label = state+"_Q_LEARN_s"+str(seed); require(row["status"] == "COMPLETE" and row["state_label"] == state and row["seed"] == seed and row["initial_theta_is_zero"] is True,"COMPLETE_FRESH_ZERO_THETA_CONDITION"); common.same(row["frozen_state_source"],source)
            fit,_ = context(row["FIT"],"fit"); logs = produced(row["updates_log"],label+"_UPDATES.jsonl",arrays=False); order = check_logs(row,logs,fit,native["states"][state]["native"][str(seed)]["order_sha256"])
            totals["training_update_logs_checked"] += len(logs); totals["same_parent_orders_checked"] += 1
            verified = {"selected_update":row["selected_update"],"selected_tune":row["selected_tune"],"checkpoints":[],"order_sha256":row["order_sha256"],"global_clipping_fraction":row["global_clipping_fraction"],"training_updates_completed":600}; states = {}
            for cp in row["checkpoints"]:
                update = cp["update"]; saved = produced(cp["state"],label+"_CP"+str(update)+"_STATE.npz"); require(set(saved) == {"E","D","center","fixed_Q","theta"},"EXACT_CP_STATE_MEMBERS")
                for key in frozen: base.exact(saved[key],frozen[key],"CP_frozen_"+key)
                base.exact(saved["fixed_Q"],np.eye(4,dtype=np.float32),"unused_fixed_Q_identity"); flags(cp["frozen_status"])
                require(cp["E_D_center_exact_initial_bytes"] is True and cp["canonical_score_link_atol"] == cp["canonical_score_link_rtol"] == 1e-12,"FROZEN_CP_AND_REGISTERED_SCORE_LINK")
                if update == 0: base.exact(saved["theta"],np.zeros(6,np.float32),"CP0_theta_zero"); require(base.state_hash(saved) == row["initial_state_sha256"],"INITIAL_STATE_HASH")
                else: base.exact(saved["theta"],np.asarray(logs[update-1]["theta_after_optimizer_step"],np.float32),"CP_theta_matches_recorded_optimizer_state")
                raw,cached = context(cp,"tune"); arrays = produced(cp["artifact"],label+"_CP"+str(update)+"_TUNE.npz"); independent = produced(cp["independent_artifact"],label+"_CP"+str(update)+"_TUNE_INDEPENDENT.npz")
                basis = check_basis(saved["theta"],arrays["Q_FP32"],cp["basis_diagnostics"]); v = check_cell(cp,arrays,independent,raw,cached,frozen,saved["theta"],arrays["Q_FP32"],cp0=update == 0)
                require(cp["tune"] == selection_score(arrays["prediction"],raw["target"],raw["observed"]),"ORIGINAL_REDUCTION_EXACT_CP_SELECTION_SCORE")
                for key in ("mse","mae"): prior.close(cp["tune"][key],cp["scores_all_observed"]["normalized"][key],"score_reduction_link",atol=1e-12,rtol=1e-12)
                states[update] = saved,arrays["Q_FP32"]; verified["checkpoints"].append({"update":update,"state":cp["state"],"tune":cp["tune"],**basis,**v})
                totals["checkpoint_states_checked"] += 1; totals["checkpoint_TUNE_prediction_evaluations_checked"] += 1; totals["produced_NPZ_artifacts_checked"] += 3; totals["score_fields_checked"] += v["score_fields_checked"]; totals["commutator_summaries_checked"] += 2; totals["actual_F_calls_registered"] += (len(raw["history"])+31)//32
            selected = check_selection(row,row["checkpoints"]); saved,q = states[selected["update"]]; require(base.state_hash(saved) == row["selected_state_sha256"],"SELECTED_STATE_VALUE_HASH")
            totals["selected_TUNE_entries_reused_checked"] += 1
            probe_arrays = produced(row["first_batch_gradient_probe"]["artifact"],label+"_FIRST_BATCH_GRADIENTS.npz"); verified["first_batch_gradient_probe"] = check_probe(row["first_batch_gradient_probe"],probe_arrays,order,logs[0]); totals["first_batch_gradient_closures_checked"] += 1; totals["produced_NPZ_artifacts_checked"] += 1
            verified.update(theta_FP32=saved["theta"].tolist(),Q_FP32=q.tolist(),selected_basis_diagnostics=row["selected_basis_diagnostics"],selected_state=row["selected_state"],selection_lock=row["selection_lock"],TUNE=next(v for v in verified["checkpoints"] if v["update"] == selected["update"]))
            for name,split in (("FIT","fit"),("DEV","val")):
                entry = row[name]; raw,cached = context(entry,split); arrays = produced(entry["artifact"],label+"_SELECTED_"+name+".npz"); independent = produced(entry["independent_artifact"],label+"_SELECTED_"+name+"_INDEPENDENT.npz")
                v = check_cell(entry,arrays,independent,raw,cached,frozen,saved["theta"],q,cp0=selected["update"] == 0); verified[name] = v
                totals["selected_"+name+"_prediction_evaluations_checked"] += 1; totals["produced_NPZ_artifacts_checked"] += 2; totals["score_fields_checked"] += v["score_fields_checked"]; totals["commutator_summaries_checked"] += 2; totals["actual_F_calls_registered"] += (len(raw["history"])+31)//32
            totals["actual_F_calls_registered"] += len(logs); totals["Q_training_conditions_checked"] += 1; report["states"][state]["learning"][str(seed)] = verified
    require(totals["Q_training_conditions_checked"] == 6 and totals["training_update_logs_checked"] == 3600 and totals["checkpoint_states_checked"] == totals["checkpoint_TUNE_prediction_evaluations_checked"] == 30 and totals["selected_FIT_prediction_evaluations_checked"] == totals["selected_TUNE_entries_reused_checked"] == totals["selected_DEV_prediction_evaluations_checked"] == totals["first_batch_gradient_closures_checked"] == totals["same_parent_orders_checked"] == 6 and totals["produced_NPZ_artifacts_checked"] == 120 and len(produced_paths) == 126 and totals["actual_F_calls_registered"] == 3714,"COMPLETE_Q_LEARNING_AUDIT_COVERAGE")
    for item in guarded: common.bound(item,check)
    return report


def execute(path, output):
    from continuation_runtime_v1 import Meter,write_new
    output = Path(output).resolve(); require(not output.exists(),"FRESH_Q_LEARNING_AUDIT_OUTPUT"); meter = Meter("audit_orthogonal_Q_learning_"+output.stem,gpu=False); failure = None; source = base.file_record(__file__)
    try:
        meter.protected_check(); initialize_numpy()
        import sys
        sys.path.insert(0,str(ROOT/"research/smallk_failure_followup_20261014"))
        from prepare_inputs import install_protected_test_guard
        from threadpoolctl import threadpool_limits
        install_protected_test_guard()
        with threadpool_limits(limits=2): report = audit_result(path,meter)
        meter.protected_check(); require(prior.file_sha(__file__) == source["sha256"],"Q_LEARNING_AUDITOR_SOURCE_UNCHANGED"); json.dumps(report,allow_nan=False)
    except Exception as error: failure = {"type":type(error).__name__,"message":str(error)}; report = {"status":"FAILED_PRESERVED","failure":failure}
    finally: report["audit_source"] = source; report["resource_terminal"] = meter.close(report["status"],failure); write_new(output,report)
    print(json.dumps({"status":report["status"],"totals":report.get("totals"),"failure":failure}),flush=True); return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--result",required=True); parser.add_argument("--output",required=True)
    args = parser.parse_args(); raise SystemExit(execute(args.result,args.output))
