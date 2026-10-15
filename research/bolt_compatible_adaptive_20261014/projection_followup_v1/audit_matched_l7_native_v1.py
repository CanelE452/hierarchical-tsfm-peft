"""Saved native affine/E/lag/D audit plus separately new geometry; no F or optimizer replay."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys

import audit_linear_e_adaptation_v1 as base
import audit_recon_init_sensitivity_v2 as geometry

HERE, prior, require = base.HERE, base.prior, base.require
STATES = ("RECON_CP0", "E_ONLY_s92601", "E_ONLY_s92602")
SEEDS, LAGS = (92601, 92602), (0, 1, 2, 4, 8, 16, 32)
CONDITIONS = {(state, seed) for state in STATES for seed in SEEDS}
RESULT_SHA = "4a6f2d3601c27bf000659aa99a586a2f04d72371766f7558cc78cc16028caa2e"
CONTROLLER_SHA = "8eafb31307ade30f66d32d18a2c6d62da18a156b0c307cede8ee4bfe72567d43"
PURE_SHA = "cfd2438be26dd7cec0bfd53552ef17f545cb7da0f7716d4185ee557ab249a534"
PARENT_SHA = {"E_only_diagnostic": "4b1559216ea99937ca8e7e0f6cdac9885abc29479a6050d82336dcfa0305bb2b", "E_only_audit": "f79324dc9452192601ac16b483e4b19c296f3c252f5814062c7c05f7d4f9e85b"}
FLAGS = {"trainable_parameter_names": ["head.weight", "head.bias"], "E_requires_grad": False, "D_requires_grad": False, "center_requires_grad": False, "E_grad_is_None": True, "D_grad_is_None": True}
np = sys.modules.get("numpy")


def initialize_numpy():
    global np
    base.initialize_numpy(); np = sys.modules["numpy"]


def same_binding(a, b):
    require(Path(a["path"]).resolve() == Path(b["path"]).resolve() and a["sha256"] == b["sha256"], "SAME_PATH_SHA_OPTIONAL_BYTES_ALLOWED")


def frozen_members(state, initial):
    for key in ("E", "D", "center"): base.exact(state[key], initial[key], "frozen_"+key)


def head_future(state, past):
    k, length = past.shape[2], past.shape[1]
    require(set(state) == {"E", "D", "center", "head.weight", "head.bias"} and state["head.weight"].shape == (k,48,length) and state["head.bias"].shape == (k,48) and all(v.dtype == np.float32 and np.isfinite(v).all() for v in state.values()), "FINITE_SAVED_NATIVE_HEAD_SHAPES")
    innovation = np.einsum("nlk,khl->nhk", past-past[:,-1:], state["head.weight"])
    return past[:,-1:] + (innovation + state["head.bias"].T[None])


def replay(saved, state, raw, cached):
    base.exact(saved["latent_history"], cached["latent_history"], "same_canonical_Torch_E_cached_history")
    prior.close(saved["latent_future"], head_future(state, saved["latent_history"]), "independent_NumPy_native_affine_future")
    return base.replay(saved, {key:state[key] for key in ("E","D","center")}, raw, LAGS)


def check_logs(row, fit):
    path = Path(row["updates_log"]["path"]); require(prior.file_sha(path) == row["updates_log"]["sha256"], "NATIVE_LOG_SHA")
    logs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    require([p["update"] for p in logs] == list(range(1,601)) and row["training_updates_completed"] == 600, "SIX_HUNDRED_HEAD_UPDATES")
    for item in logs:
        require(item["stage"] == "TRAINING_UPDATE" and all(type(item[k]) in (int,float) and math.isfinite(item[k]) and item[k] >= 0 for k in ("loss","head_weight_gradient_norm_before_clip","head_bias_gradient_norm_before_clip","global_gradient_norm_before_clip")) and item["clipped"] == (item["global_gradient_norm_before_clip"] > 1.) and item["gradient_clipping_parameter_scope"] == "HEAD_ONLY" and all(item[k] == v for k,v in FLAGS.items()), "FINITE_HEAD_ONLY_LOGGED_FLAGS_NORMS_AND_CLIPPING")
    require(row["FIT_channel_observed_counts"] == fit["observed"].sum((0,1)).tolist() and row["post_optimizer_frozen_flag_checks"] == 600 and row["frozen_state_checkpoint_byte_checks"] == 5 and row["final_frozen_status"] == FLAGS and row["frozen_bytes_checked_at_CP_and_final_only"] is True and row["independent_per_update_frozen_bytes_replayed"] is False and row["gradient_clipping_parameter_scope"] == "HEAD_ONLY", "FULL_FIT_COUNTS_AND_RECORDED_FROZEN_SCOPE")
    prior.close(row["global_clipping_fraction"], sum(p["clipped"] for p in logs)/600, "native_clipping_fraction", atol=1e-10, rtol=0.)
    order = np.random.default_rng(row["seed"]).integers(0,len(fit["history"]),size=(600,4))
    require(row["order_sha256"] == hashlib.sha256(order.tobytes()).hexdigest(), "EXACT_NATIVE_SEED_ORDER")
    return len(logs)


def new_space(saved, state, raw, state_name, head_update, e_update, destination):
    import run_learned_output_space_diagnostic_v1 as calculator
    import fixed_decoder_span_v1 as io
    require(prior.file_sha(calculator.__file__) == geometry.SPACE_SOURCE_SHA and not Path(destination).exists(), "FROZEN_SPACE_CALCULATOR_FRESH_OUTPUT")
    calculator.initialize_numpy(); cell, arrays = calculator.diagnose_cell(saved, {key:state[key] for key in ("E","D","center")}, raw, LAGS)
    cell["methods"]["ACTUAL_NATIVE"] = cell["methods"].pop("ACTUAL_F"); cell["methods"]["ACTUAL_NATIVE"]["role"] = "ACTUAL_DEPLOYMENT_TRAINED_NATIVE_HEAD_SAVED_FUTURE"
    arrays = {key.replace("ACTUAL_F_","ACTUAL_NATIVE_",1) if key.startswith("ACTUAL_F_") else key:value for key,value in arrays.items()}
    cell.pop("selected_E_D_previously_supervised_by_FIT_future_targets")
    cell.update(calculation_role="NEW_OUTPUT_SPACE_DIAGNOSTIC_COMPUTED_DURING_THIS_AUDIT", original_training_verification_replaced_by_this_calculation=False, independent_external_replay_of_new_diagnostic_claimed=False, numerical_paths="Existing frozen pure SVD and independent exact-zero-reduced LS/full SVD, not a third external replay", calculator_source=base.file_record(calculator.__file__), selected_native_head_update=head_update, frozen_E_source_selected_update=e_update, selected_E_previously_supervised_by_FIT_future_targets=e_update>0, selected_D_previously_supervised_by_FIT_future_targets=False, selected_native_head_previously_supervised_by_FIT_future_targets=head_update>0, native_head_training_procedure_used_FIT_future_targets=True, frozen_state_group=state_name, duplicate_EY_flat_minimum_across_head_seeds_are_not_independent_datasets=True, artifact=io.save_npz(destination,arrays))
    return cell


def metadata(path, check):
    require(isinstance(RESULT_SHA,str) and len(RESULT_SHA)==64,"ACTUAL_MATCHED_L7_NATIVE_RESULT_NOT_YET_REGISTERED")
    binding=base.file_record(path); require(binding["sha256"]==RESULT_SHA,"EXACT_MATCHED_L7_NATIVE_RESULT")
    result=prior.verified_json(binding,check)
    require(result["status"] in ("COMPLETE","PARTIAL_COMPLETED") and result["controller_source"]["sha256"]==CONTROLLER_SHA and result["native_source"]["sha256"]==PURE_SHA and all(result[k]["sha256"]==v for k,v in PARENT_SHA.items()),"EXACT_EXECUTED_NATIVE_AND_ACCEPTED_PARENT")
    return binding,result


def audit_result(path,meter,directory):
    check=meter.check; binding,result=metadata(path,check); parents={key:prior.verified_json(result[key],check) for key in ("E_only_diagnostic","E_only_audit","joint_recon_diagnostic","full_gradient_parent","causal_lag_diagnostic")}
    e_only,joint,full,science=(parents[k] for k in ("E_only_diagnostic","joint_recon_diagnostic","full_gradient_parent","causal_lag_diagnostic")); accepted=parents["E_only_audit"]
    require(all(parents[k]["status"]==("PASS" if k=="E_only_audit" else "COMPLETE") for k in parents) and accepted["training_verification_status"]=="PASS" and accepted["new_space_diagnostic_status"]=="COMPLETE","ACCEPTED_PARENT_TRAIN_AND_SPACE")
    same_binding(accepted["diagnostic"],result["E_only_diagnostic"])
    for key in ("joint_recon_diagnostic","full_gradient_parent","causal_lag_diagnostic"): same_binding(result[key],e_only[key])
    for key in ("new_F_forward_calls","new_F_model_objects_loaded","new_E_training_updates","new_D_training_updates","probe_optimizer_updates"): require(result[key]==0,"NO_NEW_F_OR_E_D_TRAINING:"+key)
    for key in ("new_F_state_hash_computed","new_backbone_training","whole_deployment_cost_measured","new_output_space_diagnostics_computed_by_training_producer","protected_TEST_accessed","independent_holdout","strict_K_only","NumPy_context_parity_is_primary_gate"): require(result[key] is False,"NATIVE_SCOPE:"+key)
    require(result["F_parent_signatures_reused"]=={"before":e_only["F_before"],"after":e_only["F_after"]} and e_only["F_before"]==e_only["F_after"] and result["gradient_clipping_parameter_scope"]=="HEAD_ONLY","F_SIGNATURE_REUSED_HEAD_ONLY_SCOPE")
    guarded=[binding,result["plan"],*result["source_records"],*result["input_records"],*result["protected_prior_records"]]
    for item in guarded:
        require(prior.file_sha(item["path"])==item["sha256"],"SOURCE_PARENT_AND_INPUT_SHA")
        if "snapshot_path" in item: require(prior.file_sha(item["snapshot_path"])==item["sha256"],"EXECUTED_SNAPSHOT_SHA")
    registry=prior.verified_json(result["selection_registry"],check); guarded.append(result["selection_registry"])
    require(registry["status"]=="ALL_SIX_RESOLVED_BEFORE_DEV" and registry["DEV_opened_before_seal"] is False and len(registry["conditions"])==result["resolved_selection_locks"]==6 and result["all_6_locks_before_DEV"] is True,"ALL_SIX_LOCKS_BEFORE_DEV")
    locks={}
    for item in registry["conditions"]:
        key=item["state"],item["seed"]; require(key in CONDITIONS and key not in locks,"UNIQUE_SIX_CONDITION_LOCKS")
        lock=prior.verified_json(item["selection_lock"],check); require(lock["DEV_opened_before_lock"] is False and lock["sealed_unix"]<=registry["sealed_unix"],"LOCK_BEFORE_REGISTRY"); locks[key]=item,lock; guarded.append(item["selection_lock"])
    cache={}
    def load(item):
        key=item["path"],item["sha256"]
        if key not in cache: cache[key]=prior.verified_arrays(item,check); guarded.append(item)
        return cache[key]
    recon=science["datasets"]["robin"]["spaces"]["PCA"]["conditions"]["L7_RECON"]
    raw={split:load(recon["splits"][split]["input_source"]) for split in ("fit","tune","val")}
    for split,n in (("fit",256),("tune",64),("val",30)): prior.check_raw_split(raw[split],channels=17,windows=n)
    total_names=("native_fits_checked","checkpoint_states_checked","checkpoint_TUNE_predictions_checked","selected_FIT_DEV_predictions_checked","logged_updates_checked","recorded_post_optimizer_frozen_flag_checks","CP0_probe_cases_checked","CP0_latent_persistence_byte_checks","frozen_state_CP_member_byte_comparisons_checked","matched_Bolt_decodes_checked","E0_new_RECON_C_decodes_checked","selected_E_only_Bolt_C_byte_checks","cached_Bolt_source_contexts_checked","native_cached_latent_history_byte_comparisons_checked","training_score_fields_checked","new_space_diagnostic_cells_computed","new_space_LS_SVD_crosschecks_computed")
    totals={key:0 for key in total_names}; totals["selection_locks_checked"]=6
    refs=("E_only_diagnostic","E_only_audit","joint_recon_diagnostic","joint_recon_audit","full_gradient_parent","full_gradient_parent_audit","causal_lag_diagnostic","causal_lag_audit","parent_report","parent_report_manifest","parent_report_audit","selection_registry","plan")
    report={"status":"PASS" if result["status"]=="COMPLETE" else "PARTIAL_AUDITED_NO_FULL_PASS","training_verification_status":"PASS" if result["status"]=="COMPLETE" else "PARTIAL","new_space_diagnostic_status":"RUNNING","diagnostic":binding,**{k:result[k] for k in refs},"totals":totals,"states":{},"partial_paths":[],"new_output_space_diagnostics":{},"new_output_space_diagnostic_failures":[],"optional_NumPy_context_checks":[],"optional_NumPy_context_parity_failures_recorded":result["optional_NumPy_context_parity_failures"],"protected_TEST_accessed":False,"independent_F_backward_optimizer_or_per_update_frozen_bytes_replayed":False,"new_whole_deployment_cost_measured":False,"unique_frozen_E_D_spaces":3,"unique_dataset_splits_in_new_geometry":2,"scope":"Independent NumPy latent-head affine and E/lag/D, saved score/CP/order/locks/frozen state bytes; source/recorded step flags are not optimizer or per-step loss/bytes replay. New 12 geometry cells repeat only 3 fixed E/D and 2 known splits: two frozen numeric paths, not a third external replay or independent datasets; no F inference."}
    Path(directory).mkdir(parents=True,exist_ok=False)
    frozen_states,actual_caches,cache_sources={},{},{}
    for state_name in STATES:
        owner=result["states"][state_name]; expected_source=joint["datasets"]["robin"]["conditions"]["92601"]["CP0_TUNE"]["state"] if state_name=="RECON_CP0" else e_only["datasets"]["robin"]["conditions"][state_name[-5:]]["selected_state"]
        same_binding(owner["frozen_state_source"],expected_source); frozen_states[state_name]=load(expected_source); state=frozen_states[state_name]
        require(base.state_hash(state)==owner["frozen_state_value_sha256"],"FROZEN_SOURCE_VALUE_HASH")
        if state_name=="RECON_CP0":
            original=load(full["datasets"]["robin"]["conditions"]["robin_L7_E_LEARN_s92601"]["CP0_TUNE"]["state"]); d=load(recon["state_artifact"])
            for name in ("E","center"): base.exact(state[name],original[name],"original_PCA_"+name)
            base.exact(state["D"],d["decoder_FP32"],"original_RECON_D"); base.exact(d["lags"],np.asarray(LAGS,np.int64),"original_RECON_lags")
        else:
            for name in ("D","center"): base.exact(state[name],frozen_states["RECON_CP0"][name],"all_states_same_"+name)
        require(owner["E_previously_supervised_by_FIT_future"]==(state_name!="RECON_CP0") and owner["D_previously_supervised_by_FIT_future"] is False,"FROZEN_E_D_SUPERVISION_HISTORY")
        report["states"][state_name]={"frozen_state_source":expected_source,"native":{},"Bolt":{}}; report["new_output_space_diagnostics"][state_name]={}
    def audit_baseline(state_name,split):
        owner=result["states"][state_name]; state=frozen_states[state_name]
        if state_name=="RECON_CP0": source=joint["datasets"]["robin"]["conditions"]["92601"]["CP0_TUNE"]["artifact"] if split=="tune" else full["CP0_LS_and_RAW_baselines"]["robin"]["L7"][split]["CP0_LS"]["artifact"]
        else:
            parent=e_only["datasets"]["robin"]["conditions"][state_name[-5:]]; source=next(p["artifact"] for p in parent["checkpoints"] if p["update"]==parent["selected_update"]) if split=="tune" else parent["FIT" if split=="fit" else "DEV"]["artifact"]
        cache_sources[state_name,split]=source; cached=actual_caches[state_name,split]=load(source); entry=owner["Bolt"][split]; same_binding(entry["actual_F_component_source"],source); saved=load(entry["artifact"])
        for key in ("latent_history","latent_future"): base.exact(saved[key],cached[key],"matched_Bolt_parent_"+key)
        for key in ("target","observed","raw_last","origins","mean","std"):
            base.exact(saved[key],raw[split][key],"matched_Bolt_input_"+key); base.exact(cached[key],raw[split][key],"original_Bolt_cache_input_"+key)
        base.replay(saved,state,raw[split],LAGS); totals["training_score_fields_checked"]+=base.check_score(entry,saved,raw[split]); totals["matched_Bolt_decodes_checked"]+=1; totals["cached_Bolt_source_contexts_checked"]+=1
        require(entry["new_F_forward"] is entry["source_C_prediction_reused"] is False and entry["fresh_C_decode_only"] is True and entry["source_C_used_FORECAST_D"]==(state_name=="RECON_CP0" and split!="tune") and Path(entry["artifact"]["path"]).stat().st_mtime>=registry["sealed_unix"],"NEW_C_DECODE_NOT_NEW_F_ALL_BASELINES_AFTER_LOCKS")
        if state_name!="RECON_CP0" or split=="tune": base.exact(saved["prediction"],cached["prediction"],"preserved_current_RECON_C_prediction")
        if state_name=="RECON_CP0":
            totals["E0_new_RECON_C_decodes_checked"]+=1
            try:
                legacy=load(entry["NumPy_context_reference"]); close=np.allclose(saved["prediction"],legacy["ACTUAL_F_prediction_FP32"],atol=1e-5,rtol=2e-6); recorded=entry["NumPy_RECON_context_parity"]; require(recorded.get("pass")==bool(close),"RECORDED_OPTIONAL_NUMPY_CONTEXT_RESULT")
                report["optional_NumPy_context_checks"].append({"split":split,"registered_tolerance_pass":bool(close),"primary_gate":False,"recorded":recorded})
            except Exception as error: report["optional_NumPy_context_checks"].append({"split":split,"status":"OPTIONAL_CONTEXT_NOT_REVERIFIED","primary_gate":False,"type":type(error).__name__,"message":str(error)})
        else: totals["selected_E_only_Bolt_C_byte_checks"]+=1
        report["states"][state_name]["Bolt"][split]={"artifact":entry["artifact"],"scores_all_observed":entry["scores_all_observed"],"scores_complete_mask":entry["scores_complete_mask"],"actual_F_component_source":source}
    for state_name in STATES:
        for split in ("fit","tune","val"):
            try: audit_baseline(state_name,split)
            except Exception as error: report["partial_paths"].append({"state":state_name,"split":split,"stage":"MATCHED_BOLT_BASELINE_AUDIT","type":type(error).__name__,"message":str(error)})
    def audit_condition(state_name,seed):
        owner=result["states"][state_name]; row=owner["native"][str(seed)]; item,lock=locks[state_name,seed]; same_binding(row["selection_lock"],item["selection_lock"])
        require(row["status"]==item["status"]==lock["status"],"UNCHANGED_CONDITION_LOCK_STATUS")
        if row["status"]!="COMPLETE": report["partial_paths"].append({"state":state_name,"seed":seed,"status":row["status"]}); return
        require(row["state"]==state_name and row["seed"]==seed and row["checkpoints"]==lock["checkpoints"] and row["initial_frozen_status"]==FLAGS,"SAME_HEAD_RECIPE_AND_LOCK")
        same_binding(row["frozen_state_source"],owner["frozen_state_source"])
        initial=frozen_states[state_name]; states,saved={},{ }; require([cp["update"] for cp in row["checkpoints"]]==list(base.CHECKPOINTS),"FIVE_NATIVE_CP")
        for cp in row["checkpoints"]:
            state,values=load(cp["state"]),load(cp["artifact"]); frozen_members(state,initial); require(cp["frozen_status"]==FLAGS,"CP_FROZEN_FLAGS")
            replay(values,state,raw["tune"],actual_caches[state_name,"tune"]); score=prior.scores(values["prediction"],raw["tune"]["target"],raw["tune"]["observed"],raw["tune"]["std"]); base.unit_score(cp["tune"],score["normalized"])
            require(cp["head_state_sha256"]==base.state_hash({key.split(".",1)[1]:state[key] for key in ("head.weight","head.bias")}),"CP_HEAD_VALUE_HASH"); prior.close(cp["TUNE_latent_variance"],np.var(values["latent_history"].astype(np.float64),axis=(0,1)),"native_TUNE_latent_variance",atol=1e-10,rtol=0.)
            totals["training_score_fields_checked"]+=6+base.check_score(cp,values,raw["tune"]); totals["checkpoint_states_checked"]+=1; totals["checkpoint_TUNE_predictions_checked"]+=1; totals["native_cached_latent_history_byte_comparisons_checked"]+=1; totals["frozen_state_CP_member_byte_comparisons_checked"]+=3; states[cp["update"]],saved[cp["update"]]=state,values
        zero=states[0]; require(not np.count_nonzero(zero["head.weight"]) and not np.count_nonzero(zero["head.bias"]) and row["initial_full_state_sha256"]==base.state_hash(zero) and row["initial_head_state_sha256"]==base.state_hash({key.split(".",1)[1]:zero[key] for key in ("head.weight","head.bias")}),"ZERO_FRESH_HEAD_INITIAL_STATE")
        flat=np.ascontiguousarray(np.broadcast_to(saved[0]["latent_history"][:,-1:],saved[0]["latent_future"].shape)); base.exact(saved[0]["latent_future"],flat,"CP0_zero_head_future_persistence"); totals["CP0_latent_persistence_byte_checks"]+=1
        probe=row["probe"]; arrays=load(probe["artifact"]); require(probe["status"]=="PASS" and probe["optimizer_updates"]==0 and probe["checks"]["cached_Torch_latent_history_byte_exact"] is probe["checks"]["latent_future_is_last_byte_exact"] is probe["checks"]["NumPy_lag_prediction"]["pass"] is True and probe["latent_persistence_is_current_C_persistence_claimed"] is False,"CP0_PROBE_MAIN_PARITY")
        same_binding(probe["actual_Bolt_component_source"],cache_sources[state_name,"tune"])
        for key in ("prediction","latent_history","latent_future","lag_features"): base.exact(arrays["native_"+key],saved[0][key],"CP0_probe_"+key)
        base.exact(arrays["origins"],raw["tune"]["origins"],"CP0_origins"); base.exact(arrays["flat_future"],flat,"CP0_flat_reference"); base.exact(arrays["cached_Torch_latent_history"],actual_caches[state_name,"tune"]["latent_history"],"CP0_parent_cache"); prior.close(arrays["NumPy_flat_prediction"],raw["tune"]["raw_last"][:,None]+base.delta_features(arrays["cached_Torch_latent_history"],flat,LAGS)@initial["D"],"CP0_independent_numpy_prediction"); totals["CP0_probe_cases_checked"]+=1
        selected=min(row["checkpoints"],key=lambda p:(p["tune"]["mse"],p["update"])); update=selected["update"]
        require(row["selected_update"]==lock["selected_update"]==update and row["selected_state"]==lock["selected_state"]==selected["state"] and row["selected_state_sha256"]==base.state_hash(states[update]) and row["selected_head_state_sha256"]==selected["head_state_sha256"] and row["selected_tune"]==selected["tune"] and row["CP0_TUNE"]==row["checkpoints"][0],"TUNE_ONLY_EARLIEST_NATIVE_CP")
        require(row["training_head_values_changed"]==any(not np.array_equal(states[600][k],zero[k]) for k in ("head.weight","head.bias")) and row["selected_head_nonzero"]==any(np.count_nonzero(states[update][k])>0 for k in ("head.weight","head.bias")),"TRAINED_VERSUS_SELECTED_HEAD_CHANGE")
        totals["logged_updates_checked"]+=check_logs(row,raw["fit"]); totals["recorded_post_optimizer_frozen_flag_checks"]+=row["post_optimizer_frozen_flag_checks"]; require(row["order_sha256"]==e_only["datasets"]["robin"]["conditions"][str(seed)]["order_sha256"]==lock["order_sha256"],"PAIRED_EXISTING_SEED_ORDER"); totals["native_fits_checked"]+=1
        values={"tune":saved[update]}
        for label,split in (("FIT","fit"),("DEV","val")):
            values[split]=load(row[label]["artifact"]); replay(values[split],states[update],raw[split],actual_caches[state_name,split]); totals["training_score_fields_checked"]+=base.check_score(row[label],values[split],raw[split]); totals["selected_FIT_DEV_predictions_checked"]+=1; totals["native_cached_latent_history_byte_comparisons_checked"]+=1
            if split=="val": require(Path(row[label]["artifact"]["path"]).stat().st_mtime>=registry["sealed_unix"],"SELECTED_NATIVE_DEV_AFTER_ALL_LOCKS")
        report["states"][state_name]["native"][str(seed)]={"selected_update":update,"selected_state":row["selected_state"],"selected_TUNE":row["selected_tune"],"order_sha256":row["order_sha256"],"scores":{split:{"all_observed":(selected if split=="tune" else row["FIT" if split=="fit" else "DEV"])["scores_all_observed"],"complete_C":(selected if split=="tune" else row["FIT" if split=="fit" else "DEV"])["scores_complete_mask"]} for split in ("fit","tune","val")}}
        report["new_output_space_diagnostics"][state_name][str(seed)]={}; e_update=0 if state_name=="RECON_CP0" else e_only["datasets"]["robin"]["conditions"][state_name[-5:]]["selected_update"]
        for split in ("tune","val"):
            try:
                cell=new_space(values[split],states[update],raw[split],state_name,update,e_update,Path(directory)/(state_name+"_s"+str(seed)+"_"+split+".npz")); cell.update(selected_state=row["selected_state"],selected_forecast_source=selected["artifact"] if split=="tune" else row["DEV"]["artifact"],input_source=recon["splits"][split]["input_source"]); report["new_output_space_diagnostics"][state_name][str(seed)][split]=cell; totals["new_space_diagnostic_cells_computed"]+=1; totals["new_space_LS_SVD_crosschecks_computed"]+=int(cell["independent_reduced_LS_vs_SVD_verified"])
            except Exception as error: report["new_output_space_diagnostic_failures"].append({"state":state_name,"seed":seed,"split":split,"type":type(error).__name__,"message":str(error)})
    for state_name in STATES:
        for seed in SEEDS:
            try: audit_condition(state_name,seed)
            except Exception as error: report["partial_paths"].append({"state":state_name,"seed":seed,"stage":"SAVED_TRAINING_AUDIT","type":type(error).__name__,"message":str(error)})
    for item in guarded: check(); require(prior.file_sha(item["path"])==item["sha256"],"OLD_SOURCE_AND_ARRAY_BYTES_UNCHANGED")
    totals["protected_source_and_input_file_hashes_checked"]=len(guarded)
    if report["partial_paths"]: report.update(status="PARTIAL_AUDITED_NO_FULL_PASS",training_verification_status="PARTIAL")
    if result["status"]=="COMPLETE" and not report["partial_paths"]:
        require(totals["native_fits_checked"]==result["native_training_fits"]==6 and totals["logged_updates_checked"]==result["training_updates_completed"]==3600 and totals["checkpoint_states_checked"]==result["checkpoint_states_saved"]==30 and totals["selected_FIT_DEV_predictions_checked"]==12 and totals["matched_Bolt_decodes_checked"]==result["matched_Bolt_C_decodes"]==9 and totals["E0_new_RECON_C_decodes_checked"]==result["E0_new_matched_RECON_C_decodes"]==3 and totals["selected_E_only_Bolt_C_byte_checks"]==result["selected_E_only_Bolt_C_byte_checks"]==6 and totals["cached_Bolt_source_contexts_checked"]==result["cached_Bolt_contexts_reused"]==9 and not result["failed_paths"],"SIX_FITS_THIRTY_CP_3600_UPDATES_NINE_BOLT_BASELINES")
    report["new_space_diagnostic_status"]="COMPLETE" if totals["new_space_diagnostic_cells_computed"]==totals["new_space_LS_SVD_crosschecks_computed"]==12 and not report["new_output_space_diagnostic_failures"] else "PARTIAL_COMPLETED"
    if report["status"]=="PASS" and report["new_space_diagnostic_status"]!="COMPLETE": report["status"]="PARTIAL_AUDITED_DIAGNOSTICS_INCOMPLETE"
    return report


def execute(path,output):
    from continuation_runtime_v1 import Meter,write_new
    output=Path(output).resolve(); require(not output.exists(),"FRESH_MATCHED_NATIVE_AUDIT_OUTPUT"); meter=Meter("audit_matched_L7_native_"+output.stem,gpu=False); source=base.file_record(__file__); failure=None
    try:
        meter.protected_check(); metadata(path,meter.check); initialize_numpy(); sys.path.insert(0,str(base.ROOT/"research/smallk_failure_followup_20261014"))
        from prepare_inputs import install_protected_test_guard
        from threadpoolctl import threadpool_limits
        install_protected_test_guard()
        with threadpool_limits(limits=2): report=audit_result(path,meter,output.with_name(output.stem+"_new_space_arrays"))
        meter.protected_check(); require(prior.file_sha(__file__)==source["sha256"],"AUDIT_SOURCE_UNCHANGED"); json.dumps(report,allow_nan=False)
    except Exception as error: failure={"type":type(error).__name__,"message":str(error)}; report={"status":"FAILED_PRESERVED","failure":failure}
    finally: report["audit_source"]=source; report["resource"]=meter.close(report["status"],failure); write_new(output,report)
    print(json.dumps({"status":report["status"],"totals":report.get("totals"),"failure":failure}),flush=True); return 0 if report["status"]=="PASS" else 2


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--result",required=True); parser.add_argument("--output",required=True); args=parser.parse_args(); raise SystemExit(execute(args.result,args.output))
