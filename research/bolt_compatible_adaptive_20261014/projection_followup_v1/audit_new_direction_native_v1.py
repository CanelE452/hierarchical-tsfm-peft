"""Saved native head/lag-D/selection audit for eight fixed new-direction states; no training replay."""
from pathlib import Path
import argparse
import json
import math

import audit_matched_l7_native_v1 as native
import audit_orthogonal_basis_controls_v1 as common

base,prior,require = native.base,native.prior,native.require
HERE,ROOT,LAGS,SEEDS,CPS = native.HERE,base.ROOT,native.LAGS,native.SEEDS,base.CHECKPOINTS
DATASETS = {"robin":(17,4,"val",30),"beijing_aotizhongxin":(11,2,"eval",146)}
STATES = tuple((data,kind+"_K"+str(k)) for data,(_,k0,_,_) in DATASETS.items() for kind in ("PCA","CROSS_MOMENT") for k in (k0,2*k0))
RESULT_SHA = "786edbd75a35d2c5a760a7b4f5af5a1389ee1bc4bc93594d7b359a29ecfd83ed"
CONTROLLER_SHA = "4aaf63c4ae078daed71f4b11540d4ade8ee62ffce6a8345bd031fe162b9062e4"
NATIVE_SHA = "cfd2438be26dd7cec0bfd53552ef17f545cb7da0f7716d4185ee557ab249a534"
PARENT_SHA = {**common.PARENT_SHA,"predictive_cross_diagnostic":"99bffd5831143ca35947099fef9567f89dd209d83451661decee0682b2b8e13c","predictive_cross_audit":"90c68a9057909fb5b6c23284e5a45523cdf126da178a5ba0d9d9e96a50dede30","predictive_cross_auditor_source":"6577d15a200728dc585bf2c8e93ccff6baabd1ba916384dfb2cce7aa0014a731"}
REFS = tuple(key for key in common.REFS if key not in ("controller_source","module_source","plan","control_registry"))+("parent_metadata_reader","predictive_cross_diagnostic","predictive_cross_audit","predictive_cross_auditor_source","predictive_cross_state_registry","selection_registry","controller_source","native_source","plan","parent_controller_source","reused_matched_native_source")
RECIPE = {"optimizer":"AdamW","learning_rate":.001,"weight_decay":0.,"batch_size":4,"clip_norm":1.,"updates_per_condition":600,"checkpoint_updates":list(CPS),"head":"stream-specific affine512_to_48_from_history_minus_last_plus_last","head_zero_initialized":True}
np = None


def initialize_numpy():
    global np
    native.initialize_numpy(); common.initialize_numpy(); np = native.np


def metadata(path, check):
    require(isinstance(RESULT_SHA,str) and len(RESULT_SHA) == 64,"ACTUAL_NEW_DIRECTION_NATIVE_RESULT_NOT_YET_REGISTERED")
    binding = base.file_record(path); require(binding["sha256"] == RESULT_SHA,"EXACT_NEW_DIRECTION_NATIVE_RESULT")
    result = prior.verified_json(binding,check)
    require(result["status"] == "COMPLETE" and not result["failed_paths"] and result["controller_source"]["sha256"] == CONTROLLER_SHA and result["native_source"]["sha256"] == NATIVE_SHA,"COMPLETE_EXACT_EXECUTED_NATIVE_CONTROLLER_AND_HEAD")
    require(set(result["datasets"]) == set(DATASETS) and all(set(result["datasets"][data]["conditions"]) == {name for d,name in STATES if d == data} for data in DATASETS),"EXACT_TWO_DATASETS_EIGHT_FROZEN_STATES")
    require(result["reused_matched_native_source"]["sha256"] == native.CONTROLLER_SHA,"EXACT_REUSED_MATCHED_NATIVE_SOURCE")
    for key,digest in PARENT_SHA.items(): require(result[key]["sha256"] == digest,"EXACT_ACCEPTED_PARENT:"+key)
    return binding,result


def check_recorded_parity(recorded, actual, expected, label, *, required=True):
    computed = common.parity(actual,expected)
    require(set(recorded) == set(computed) and recorded["pass"] == computed["pass"] and recorded["atol"] == 1e-5 and recorded["rtol"] == 2e-6,"REGISTERED_PARITY_FLAGS:"+label)
    prior.close([recorded["max_abs"],recorded["rmse"]],[computed["max_abs"],computed["rmse"]],label,atol=1e-10,rtol=1e-10)
    if required: require(computed["pass"],"REQUIRED_DEPLOYMENT_PARITY:"+label)
    return computed


def selection_score(prediction, target, observed):
    require(prediction.shape == target.shape == observed.shape and observed.dtype == np.bool_ and np.isfinite(prediction).all() and np.isfinite(target[observed]).all(),"FINITE_OWN_TUNE_SELECTION_ARRAYS")
    counts = observed.sum((0,1)); active = counts > 0; require(active.any(),"OBSERVED_SELECTION_CHANNEL")
    error = np.where(observed,prediction.astype(np.float64)-target.astype(np.float64),0.)
    mse = np.square(error).sum((0,1))/np.maximum(counts,1); mae = np.abs(error).sum((0,1))/np.maximum(counts,1)
    return {"mse":float(mse[active].mean()),"mae":float(mae[active].mean()),"observed_counts":counts.tolist(),"active_channels":int(active.sum()),"per_channel_mse":[float(v) if a else None for v,a in zip(mse,active)],"per_channel_mae":[float(v) if a else None for v,a in zip(mae,active)]}


def check_cell(row, arrays, independent, raw, cached, state):
    n,length,c = raw["history"].shape; k = state["E"].shape[1]; h = raw["target"].shape[1]
    shapes = {"prediction":(n,h,c),"latent_history":(n,length,k),"latent_future":(n,h,k),"lag_features":(n,h,len(LAGS)*k)}
    context = ("target","observed","raw_last","origins","mean","std")
    require(set(arrays) == set(shapes)|set(context) and set(independent) == {"independent_NumPy_prediction","independent_NumPy_lag_features"},"EXACT_NATIVE_FORECAST_AND_INDEPENDENT_MEMBERS")
    for key,shape in shapes.items(): require(arrays[key].dtype == np.float32 and arrays[key].shape == shape and np.isfinite(arrays[key]).all(),"FINITE_ACTUAL_NATIVE_COMPONENT:"+key)
    for key in context: base.exact(arrays[key],raw[key],"original_native_context_"+key)
    base.exact(arrays["latent_history"],cached["latent_history"],"same_Torch32_cached_E_history")
    future = native.head_future(state,arrays["latent_history"]); head_parity = common.parity(arrays["latent_future"],future); require(head_parity["pass"],"INDEPENDENT_NUMPY_NATIVE_AFFINE_FUTURE")
    features = base.delta_features(arrays["latent_history"],arrays["latent_future"],LAGS); prediction = raw["raw_last"][:,None]+features@state["D"]
    base.exact(independent["independent_NumPy_lag_features"],features,"independent_saved_old_lag_features"); prior.close(independent["independent_NumPy_prediction"],prediction,"independent_saved_old_lag_C",atol=1e-5,rtol=2e-6)
    require(set(row["checks"]) == {"cached_Torch_latent_history_byte_exact","independent_NumPy_old_lag_C","independent_NumPy_lag_features"} and row["checks"]["cached_Torch_latent_history_byte_exact"] is True,"ALL_NATIVE_SAVED_DEPLOYMENT_CHECKS")
    check_recorded_parity(row["checks"]["independent_NumPy_old_lag_C"],arrays["prediction"],prediction,"original_lag_D")
    check_recorded_parity(row["checks"]["independent_NumPy_lag_features"],arrays["lag_features"],features,"original_lag_features")
    fresh = check_recorded_parity(row["NumPy_fresh_E_parity"],arrays["latent_history"],(raw["history"]-state["center"])@state["E"],"fresh_E_auxiliary",required=False)
    require(row["status"] == "COMPLETE" and row["canonical_batch_size"] == 32 and row["future_truth_is_forward_input"] is row["new_F_forward"] is row["NumPy_fresh_E_parity_is_primary_gate"] is False and row["same_D_and_current_C_anchor"] is True,"HEAD_ONLY_SAVED_FORWARD_SCOPE")
    complete = np.broadcast_to(raw["observed"].all(2,keepdims=True),raw["observed"].shape); fields = base.check_score(row,arrays,raw)
    for name,p in (("baseline_Bolt_scores",cached["prediction"]),("RAW_BOLT_scores",raw["raw_forecast"])):
        for mask_name,mask in (("all_observed",raw["observed"]),("complete_C",complete)):
            if mask.any(): fields += prior.compare_scores(row[name][mask_name],prior.scores(p,raw["target"],mask,raw["std"]),name+mask_name)
            else: require(row[name][mask_name] == {"status":"NO_COMPLETE_C"},"NO_COMPLETE_C_EXPLICIT")
    require(row["complete_target_positions"] == int(complete[:,:,0].sum()),"EXACT_COMPLETE_TARGET_COUNT")
    return {"artifact":row["artifact"],"independent_artifact":row["independent_artifact"],"scores_all_observed":row["scores_all_observed"],"scores_complete_mask":row["scores_complete_mask"],"baseline_Bolt_scores":row["baseline_Bolt_scores"],"RAW_BOLT_scores":row["RAW_BOLT_scores"],"score_fields_checked":fields,"independent_NumPy_head_future_parity":head_parity,"NumPy_fresh_E_auxiliary_parity":fresh,"new_F_forward_replayed":False}


def check_probe(row, arrays, cp0, cached, raw, frozen):
    expected = {"native_"+key for key in ("prediction","latent_history","latent_future","lag_features")}|{"cached_Torch_latent_history","flat_future","NumPy_flat_prediction","origins"}
    require(set(arrays) == expected and row["status"] == "PASS" and row["optimizer_updates"] == 0 and row["latent_persistence_is_current_C_persistence_claimed"] is False,"EXACT_ZERO_HEAD_CP0_PROBE_SCOPE")
    require(set(row["checks"]) == {"cached_Torch_latent_history_byte_exact","latent_future_is_last_byte_exact","NumPy_lag_prediction"} and row["checks"]["cached_Torch_latent_history_byte_exact"] is row["checks"]["latent_future_is_last_byte_exact"] is True,"EXACT_CP0_PROBE_CHECKS")
    for key in ("prediction","latent_history","latent_future","lag_features"): base.exact(arrays["native_"+key],cp0[key],"CP0_probe_and_immutable_CP0_"+key)
    base.exact(arrays["origins"],raw["origins"],"probe_origins"); base.exact(arrays["cached_Torch_latent_history"],cached["latent_history"],"probe_cached_history")
    flat = np.ascontiguousarray(np.broadcast_to(cached["latent_history"][:,-1:],cp0["latent_future"].shape)); base.exact(arrays["flat_future"],flat,"probe_flat_future"); base.exact(cp0["latent_future"],flat,"CP0_zero_head_latent_persistence")
    prediction = raw["raw_last"][:,None]+base.delta_features(cached["latent_history"],flat,LAGS)@frozen["D"]
    prior.close(arrays["NumPy_flat_prediction"],prediction,"probe_independent_flat_old_D",atol=1e-5,rtol=2e-6)
    check_recorded_parity(row["checks"]["NumPy_lag_prediction"],cp0["prediction"],prediction,"CP0_flat_old_D")
    return {"status":"PASS","artifact":row["artifact"],"latent_persistence_is_current_C_persistence_claimed":False,"optimizer_updates":0}


def check_registry(result, registry, load):
    require(result["all_16_locks_before_DEV"] is True and result["resolved_selection_locks"] == 16 and registry["status"] == "ALL_SIXTEEN_RESOLVED_BEFORE_DEV" and registry["DEV_opened_before_seal"] is False and len(registry["conditions"]) == 16,"ALL_SIXTEEN_LOCKS_BEFORE_DEV")
    locks = {}; allowed = {(data,name,seed) for data,name in STATES for seed in SEEDS}
    for item in registry["conditions"]:
        key = item["dataset"],item["condition"],item["seed"]; require(key in allowed and key not in locks and item["status"] == "COMPLETE","SIXTEEN_UNIQUE_COMPLETE_CONDITIONS")
        row = result["datasets"][key[0]]["conditions"][key[1]]["native"][str(key[2])]; common.same(item["selection_lock"],row["selection_lock"]); lock = load(item["selection_lock"])
        require(lock["DEV_opened_before_lock"] is False and lock["sealed_unix"] <= registry["sealed_unix"] and {k:v for k,v in lock.items() if k not in ("DEV_opened_before_lock","sealed_unix")} == {k:v for k,v in row.items() if k not in ("DEV","selection_lock")},"EXACT_PRE_DEV_IMMUTABLE_CONDITION_LOCK")
        locks[key] = lock
    return locks


def check_selection(row, states):
    require([cp["update"] for cp in row["checkpoints"]] == list(CPS) and row["checkpoint_count"] == 5 and row["CP0_TUNE"] == row["checkpoints"][0],"ALL_FIVE_CP_AND_CP0_PRESERVED")
    selected = min(row["checkpoints"],key=lambda cp:(cp["tune"]["mse"],cp["update"])); update = selected["update"]
    require(row["selected_update"] == update and row["selected_tune"] == selected["tune"] and row["selected_state_sha256"] == base.state_hash(states[update]) and row["selected_head_state_sha256"] == selected["head_state_sha256"],"OWN_TUNE_EXACT_EARLIER_CP_HEAD_SELECTION")
    common.same(row["selected_state"],selected["state"])
    require(row["TUNE"] == {k:v for k,v in selected.items() if k not in ("state","frozen_status","head_state_sha256","E_D_center_exact_initial_bytes")},"SELECTED_TUNE_REUSED_WITHOUT_REFIT")
    zero = states[0]
    require(row["initial_full_state_sha256"] == base.state_hash(zero) and row["initial_head_state_sha256"] == base.state_hash({k.split(".",1)[1]:zero[k] for k in ("head.weight","head.bias")}) and not np.count_nonzero(zero["head.weight"]) and not np.count_nonzero(zero["head.bias"]),"FRESH_ZERO_HEAD_INITIAL_STATE")
    require(row["training_head_values_changed"] == any(not np.array_equal(states[600][k],zero[k]) for k in ("head.weight","head.bias")) and row["selected_head_nonzero"] == any(np.count_nonzero(states[update][k]) > 0 for k in ("head.weight","head.bias")),"TRAINED_VERSUS_SELECTED_HEAD_VALUES")
    return selected


def check_logs(row, fit, parent_order):
    count = native.check_logs(row,fit); require(row["loss"] == "N_fit/B/C_active * sum_c SSE_batch_c/N_fit_c; fixed entire-FIT counts" and row["order_sha256"] == parent_order,"SAME_PARENT_SEED_ORDER_AND_GLOBAL_FIT_LOSS")
    return count


def audit_result(path, meter):
    check = meter.check; binding,result = metadata(path,check)
    for key in ("new_F_forward_calls","new_F_model_objects_loaded","new_E_training_updates","new_D_training_updates"): require(result[key] == 0,"NO_NEW_F_E_D_TRAINING:"+key)
    for key in ("new_F_state_hash_computed","new_backbone_training","whole_deployment_cost_measured","new_output_space_computed","protected_TEST_accessed","independent_holdout","strict_K_only"): require(result[key] is False,"NEW_NATIVE_SCOPE:"+key)
    require(result["training_recipe"] == RECIPE and result["selection_rule"] == "own canonical32 normalized TUNE macro MSE; exact ties earlier CP" and result["gradient_clipping_parameter_scope"] == "HEAD_ONLY","FROZEN_NATIVE_RECIPE_AND_SELECTION")
    expected = {"native_training_fits":16,"training_updates_completed":9600,"checkpoint_states_saved":80,"DEV_evaluations_completed":16,"cached_actual_F_contexts_reused":24,"cached_actual_F_forward_calls_registered_in_parent":104,"new_optimizer_updates":9600,"new_native_head_updates":9600,"checkpoint_artifact_hashes_rechecked":240}
    require(all(result[k] == v for k,v in expected.items()),"EXACT_NATIVE_EXECUTION_COUNTS")
    guarded = [binding,*[result[k] for k in REFS],*result["source_records"],*result["input_records"],*result["protected_prior_records"]]
    for item in guarded:
        common.bound(item,check)
        if "snapshot_path" in item: require(prior.file_sha(item["snapshot_path"]) == item["sha256"],"EXECUTED_SOURCE_SNAPSHOT_BYTES")
    parent = prior.verified_json(result["predictive_cross_diagnostic"],check); accepted = prior.verified_json(result["predictive_cross_audit"],check); previous = prior.verified_json(result["matched_l7_native_diagnostic"],check); science = prior.verified_json(result["causal_lag_diagnostic"],check)
    require(parent["status"] == "COMPLETE" and accepted["status"] == "PASS" and not accepted["partial_paths"] and not parent["failed_paths"],"ACCEPTED_PARENT_EIGHT_STATE_F_CACHE_AUDIT")
    common.same(accepted["diagnostic"],result["predictive_cross_diagnostic"]); common.same(accepted["audit_source"],result["predictive_cross_auditor_source"]); common.same(result["predictive_cross_state_registry"],parent["state_registry"]); common.same(result["parent_controller_source"],parent["controller_source"])
    require(result["F_parent_signatures_reused"] == {"before":parent["F_before"],"after":parent["F_after"]} and parent["F_before"] == parent["F_after"] and result["runtime_behavior_contract_reused"] == parent["runtime_behavior_contract"],"UNCHANGED_REUSED_F_HASH_AND_MEDIAN_CONTRACT")
    for key in REFS:
        if key in parent and key not in ("controller_source","native_source","plan","selection_registry"): common.same(result[key],parent[key])
    registry = prior.verified_json(result["selection_registry"],check); locks = check_registry(result,registry,lambda item:prior.verified_json(item,check))
    from audit_fixed_l7_latent_calibration_v1 import receipt_check
    guarded += receipt_check(result["resource_terminal"],"COMPLETE",check)
    receipt = result["resource_terminal"]; require(receipt["error"] is None and receipt["started_unix"] <= registry["sealed_unix"] <= receipt["ended_unix"],"REGISTRY_WITHIN_COMPLETE_NATIVE_RECEIPT")
    output = Path(binding["path"]).parent; cache = {}; produced_paths = set()
    def load(item):
        key = item["path"],item["sha256"]
        if key not in cache: cache[key] = prior.verified_arrays(item,check)
        return cache[key]
    def produced(item, name):
        target = Path(item["path"]).resolve(); require(target == output/name and target not in produced_paths,"UNIQUE_REGISTERED_NATIVE_PRODUCER_ARTIFACT"); produced_paths.add(target); return load(item)
    names = ("native_fits_checked","logged_updates_checked","checkpoint_states_checked","checkpoint_TUNE_predictions_checked","selected_FIT_predictions_checked","selected_TUNE_entries_reused_checked","selected_DEV_predictions_checked","CP0_probe_cases_checked","cached_actual_F_context_lineages_checked","independent_NumPy_head_and_old_D_cells_checked","frozen_CP_member_byte_comparisons_checked","native_cached_Torch_history_byte_comparisons_checked","training_score_fields_checked","baseline_score_fields_checked","produced_NPZ_artifacts_checked","same_parent_seed_orders_checked","frozen_E_D_center_states_checked")
    totals = {key:0 for key in names}; totals.update(selection_locks_checked=len(locks),parent_actual_F_calls_registered=parent["new_F_forward_calls"],protected_prior_file_hashes_checked=len(result["protected_prior_records"]),new_output_space_diagnostics_computed=0)
    report = {"status":"PASS","diagnostic":binding,**{k:result[k] for k in REFS},"totals":totals,"datasets":{},"partial_paths":[],"protected_TEST_accessed":False,"independent_holdout":False,"new_output_space_computed":False,"new_whole_deployment_cost_measured":False,"independent_F_GPU_backward_optimizer_or_per_step_loss_replayed":False,"DEV_access_events_independently_replayed":False,"scope":"Independent NumPy saved per-stream affine head and old lag-specific-origin decoder, exact frozen state/cache contexts, observed-channel macro metrics, checkpoint selection, recorded update flags/global FIT counts/order, source/receipt/locks. Reuses accepted auditor helpers; same agent prepared producer and auditor, ROOT reviews both. No F, gradient, optimizer or per-step loss replay, new output-space computation or cost. Before-DEV ordering uses sealed records and frozen source flow, not filemtime/access-event replay. Six hundred updates is a finite budget; affine-head results do not establish impossibility of all native predictors. DEV remains adaptive and TEST untouched."}
    contexts = {}
    def context(data, split, entry):
        source = science["datasets"][data]["spaces"]["PCA"]["conditions"]["L7_RECON"]["splits"][split]["input_source"]; common.same(entry["input_source"],source)
        if (data,split) not in contexts:
            raw = load(source); c,_,dev,devn = DATASETS[data]; prior.check_raw_split(raw,channels=c,windows={"fit":256,"tune":64}.get(split,devn)); contexts[data,split] = raw
        return contexts[data,split]
    for data,name in STATES:
        owner = result["datasets"][data]["conditions"][name]; original = parent["datasets"][data]["conditions"][name]; c,_,dev,_ = DATASETS[data]; k = original["K"]; source = original["state_artifact"]; common.same(owner["frozen_state_source"],source); frozen = load(source)
        require(set(frozen) == {"E","D","center"} and frozen["E"].shape == (c,k) and frozen["D"].shape == (7*k,c) and frozen["center"].shape == (c,) and all(v.dtype == np.float32 and np.isfinite(v).all() for v in frozen.values()),"FINITE_ACCEPTED_E_D_CENTER")
        require(owner["kind"] == original["kind"] and owner["K"] == k and owner["C"] == c and owner["E_FIT_future_supervised"] == original["E_FIT_future_supervised"] and owner["D_FIT_future_supervised"] is False and owner["frozen_state_value_sha256"] == base.state_hash(frozen) and set(owner["native"]) == {str(v) for v in SEEDS} and set(owner["Bolt"]) == {"fit","tune",dev},"EXACT_FROZEN_HISTORY_AND_BOTH_NATIVE_SEEDS")
        verified_owner = {"kind":owner["kind"],"K":k,"C":c,"frozen_state_source":source,"native":{},"Bolt":{}}; report["datasets"].setdefault(data,{"conditions":{}})["conditions"][name] = verified_owner; totals["frozen_E_D_center_states_checked"] += 1; cached = {}
        for split in ("fit","tune",dev):
            entry = owner["Bolt"][split]; accepted_entry = original["splits"][split]; raw = context(data,split,entry); common.same(entry["artifact"],accepted_entry["artifact"]); common.same(entry["state_source"],source)
            require(entry["status"] == "COMPLETE" and entry["new_F_forward"] is False and entry["original_actual_F_cache_reused"] is entry["parent_actual_F_forward_calls_not_reexecuted"] is True and entry["methods"] == accepted_entry["methods"],"EXACT_PARENT_F_CACHE_REUSE_WITHOUT_FORWARD")
            values = load(entry["artifact"]); base.exact(values["origins"],raw["origins"],"F_cache_original_origins")
            require(values["latent_history"].shape == (len(raw["history"]),512,k) and values["latent_future"].shape == (len(raw["history"]),48,k) and values["prediction"].shape == raw["target"].shape and all(values[key].dtype == np.float32 and np.isfinite(values[key]).all() for key in ("latent_history","latent_future","prediction")),"FINITE_REGISTERED_PARENT_F_CACHE")
            prior.close(values["prediction"],raw["raw_last"][:,None]+base.delta_features(values["latent_history"],values["latent_future"],LAGS)@frozen["D"],"parent_F_saved_old_D",atol=1e-5,rtol=2e-6)
            complete = np.broadcast_to(raw["observed"].all(2,keepdims=True),raw["observed"].shape)
            for method,prediction in (("ACTUAL_F",values["prediction"]),("RAW_BOLT",raw["raw_forecast"])):
                for field,mask in (("scores_all_observed",raw["observed"]),("scores_complete_mask",complete)):
                    if mask.any(): totals["baseline_score_fields_checked"] += prior.compare_scores(entry["methods"][method][field],prior.scores(prediction,raw["target"],mask,raw["std"]),"parent_F_Raw_"+method+field)
                    else: require(entry["methods"][method][field] == {"status":"NO_VALID_TARGETS"},"NO_PARENT_COMPLETE_C_EXPLICIT")
            cached[split] = values; verified_owner["Bolt"][split] = {"artifact":entry["artifact"],"input_source":entry["input_source"],"methods":entry["methods"]}; totals["cached_actual_F_context_lineages_checked"] += 1
        for seed in SEEDS:
            row = owner["native"][str(seed)]; label = data+"_"+name+"_NATIVE_s"+str(seed); require(row["status"] == "COMPLETE" and row["dataset"] == data and row["condition"] == name and row["seed"] == seed and row["initial_frozen_status"] == native.FLAGS and row["head_parameters"] == k*48*513,"EXACT_NATIVE_IDENTITY_RECIPE_AND_HEAD_CAPACITY"); common.same(row["frozen_state_source"],source)
            states,outputs = {},{}; verified = {"checkpoints":[],"selected_update":row["selected_update"],"selected_state":row["selected_state"],"selected_state_sha256":row["selected_state_sha256"],"selected_head_state_sha256":row["selected_head_state_sha256"],"selected_tune":row["selected_tune"],"selection_lock":row["selection_lock"],"order_sha256":row["order_sha256"],"global_clipping_fraction":row["global_clipping_fraction"]}
            for cp in row["checkpoints"]:
                update = cp["update"]; state = produced(cp["state"],label+"_CP"+str(update)+"_STATE.npz"); native.frozen_members(state,frozen)
                require(cp["frozen_status"] == native.FLAGS and cp["E_D_center_exact_initial_bytes"] is True and cp["canonical_score_link_atol"] == cp["canonical_score_link_rtol"] == 1e-12 and cp["head_state_sha256"] == base.state_hash({key.split(".",1)[1]:state[key] for key in ("head.weight","head.bias")}),"CP_FROZEN_FLAGS_HEAD_HASH_AND_SCORE_LINK")
                raw = context(data,"tune",cp); common.same(cp["cached_Bolt_source"],original["splits"]["tune"]["artifact"]); common.same(cp["state_source"],source); values = produced(cp["artifact"],label+"_CP"+str(update)+"_TUNE.npz"); independent = produced(cp["independent_artifact"],label+"_CP"+str(update)+"_TUNE_INDEPENDENT.npz")
                v = check_cell(cp,values,independent,raw,cached["tune"],state); require(cp["tune"] == selection_score(values["prediction"],raw["target"],raw["observed"]),"ORIGINAL_NATIVE_REDUCTION_EXACT_CP_SELECTION_SCORE")
                for key in ("mse","mae"): prior.close(cp["tune"][key],cp["scores_all_observed"]["normalized"][key],"CP_score_reduction_link",atol=1e-12,rtol=1e-12)
                states[update],outputs[update] = state,values; verified["checkpoints"].append({"update":update,"state":cp["state"],"tune":cp["tune"],"head_state_sha256":cp["head_state_sha256"],**v}); totals["checkpoint_states_checked"] += 1; totals["checkpoint_TUNE_predictions_checked"] += 1; totals["frozen_CP_member_byte_comparisons_checked"] += 3; totals["produced_NPZ_artifacts_checked"] += 3; totals["independent_NumPy_head_and_old_D_cells_checked"] += 1; totals["native_cached_Torch_history_byte_comparisons_checked"] += 1; totals["training_score_fields_checked"] += v["score_fields_checked"]
            selected = check_selection(row,states); selected_state = states[selected["update"]]; verified["TUNE"] = next(cp for cp in verified["checkpoints"] if cp["update"] == selected["update"]); totals["selected_TUNE_entries_reused_checked"] += 1
            common.bound(row["updates_log"],check); require(Path(row["updates_log"]["path"]).resolve() == output/(label+"_UPDATES.jsonl"),"REGISTERED_UNIQUE_NATIVE_LOG_PATH"); guarded.append(row["updates_log"])
            totals["logged_updates_checked"] += check_logs(row,context(data,"fit",row["FIT"]),previous["states"]["RECON_CP0"]["native"][str(seed)]["order_sha256"]); totals["same_parent_seed_orders_checked"] += 1
            probe = row["probe"]; common.same(probe["actual_Bolt_component_source"],original["splits"]["tune"]["artifact"]); arrays = produced(probe["artifact"],label+"_CP0_PARITY.npz"); verified["probe"] = check_probe(probe,arrays,outputs[0],cached["tune"],context(data,"tune",row["checkpoints"][0]),frozen); totals["CP0_probe_cases_checked"] += 1; totals["produced_NPZ_artifacts_checked"] += 1
            for field,split in (("FIT","fit"),("DEV",dev)):
                entry = row[field]; common.same(entry["cached_Bolt_source"],original["splits"][split]["artifact"]); common.same(entry["state_source"],source); raw = context(data,split,entry); arrays = produced(entry["artifact"],label+"_SELECTED_"+field+".npz"); independent = produced(entry["independent_artifact"],label+"_SELECTED_"+field+"_INDEPENDENT.npz"); v = check_cell(entry,arrays,independent,raw,cached[split],selected_state); verified[field] = v
                totals["selected_"+field+"_predictions_checked"] += 1; totals["produced_NPZ_artifacts_checked"] += 2; totals["independent_NumPy_head_and_old_D_cells_checked"] += 1; totals["native_cached_Torch_history_byte_comparisons_checked"] += 1; totals["training_score_fields_checked"] += v["score_fields_checked"]
            verified_owner["native"][str(seed)] = verified; totals["native_fits_checked"] += 1
    require(totals["native_fits_checked"] == 16 and totals["logged_updates_checked"] == 9600 and totals["checkpoint_states_checked"] == totals["checkpoint_TUNE_predictions_checked"] == 80 and totals["selected_FIT_predictions_checked"] == totals["selected_TUNE_entries_reused_checked"] == totals["selected_DEV_predictions_checked"] == totals["CP0_probe_cases_checked"] == totals["same_parent_seed_orders_checked"] == 16 and totals["cached_actual_F_context_lineages_checked"] == 24 and totals["independent_NumPy_head_and_old_D_cells_checked"] == totals["native_cached_Torch_history_byte_comparisons_checked"] == 112 and totals["frozen_E_D_center_states_checked"] == 8 and totals["produced_NPZ_artifacts_checked"] == len(produced_paths) == 320,"COMPLETE_SIXTEEN_NATIVE_AUDIT_COVERAGE")
    for item in guarded: common.bound(item,check)
    for path,digest in cache: check(); require(prior.file_sha(path) == digest,"ALL_LOADED_NATIVE_ARRAY_BYTES_UNCHANGED")
    return report


def execute(path, output):
    from continuation_runtime_v1 import Meter,write_new
    output = Path(output).resolve(); require(not output.exists(),"FRESH_NEW_NATIVE_AUDIT_OUTPUT"); meter = Meter("audit_new_direction_native_"+output.stem,gpu=False); source = base.file_record(__file__); failure = None
    try:
        meter.protected_check(); metadata(path,meter.check); initialize_numpy()
        import sys
        sys.path.insert(0,str(ROOT/"research/smallk_failure_followup_20261014"))
        from prepare_inputs import install_protected_test_guard
        from threadpoolctl import threadpool_limits
        install_protected_test_guard()
        with threadpool_limits(limits=2): report = audit_result(path,meter)
        meter.protected_check(); require(prior.file_sha(__file__) == source["sha256"],"NATIVE_AUDITOR_SOURCE_UNCHANGED"); json.dumps(report,allow_nan=False)
    except Exception as error: failure = {"type":type(error).__name__,"message":str(error)}; report = {"status":"FAILED_PRESERVED","failure":failure}
    finally: report["audit_source"] = source; report["resource_terminal"] = meter.close(report["status"],failure); write_new(output,report)
    print(json.dumps({"status":report["status"],"totals":report.get("totals"),"failure":failure}),flush=True); return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--result",required=True); parser.add_argument("--output",required=True)
    args = parser.parse_args(); raise SystemExit(execute(args.result,args.output))
