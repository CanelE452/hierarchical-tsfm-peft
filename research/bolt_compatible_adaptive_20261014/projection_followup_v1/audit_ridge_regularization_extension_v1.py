"""Saved adaptive ridge-extension audit; selected new fits only, no F or training replay."""
from pathlib import Path
import argparse
import hashlib
import json
import math

import audit_new_direction_direct_ridge_v1 as original_audit
import audit_orthogonal_basis_controls_v1 as common

base,prior,require = original_audit.base,original_audit.prior,original_audit.require
HERE,ROOT,DATASETS = original_audit.HERE,original_audit.ROOT,original_audit.DATASETS
OLD_ALPHAS,NEW_ALPHAS = original_audit.ALPHAS,(100.,1000.,10000.)
MENU = tuple("ALPHA_"+format(a,"g") for a in (*OLD_ALPHAS,*NEW_ALPHAS))+("INTERCEPT_ONLY","CURRENT_C_ANCHOR")
CONDITIONS = tuple((data,route) for data,(_,k,_,_) in DATASETS.items() for route in ("RAW_HISTORY512",*(kind+"_K"+str(width)+"_HISTORY512" for kind in ("PCA","CROSS_MOMENT") for width in (k,2*k))))
RESULT_SHA = "8fcd2e19fcfdf9f9a0b261553cf8179de129b813a3fef92039c86de415c689b3"
CONTROLLER_SHA = "310f56d668d328e6ab5b49c00a25c527c7068379e4df701beee0fcb17a035dff"
MODULE_SHA = "49c08461bd7f114511bd9bf3cdcfa8e5fbf8b89152edde4435c6944767d8ba46"
ORIGINAL_SHA = "b8d7eae62ce5bb52b3d46ffc543f916f8b81857028098370a5b2f8398510f1ed"
ORIGINAL_AUDIT_SHA = "8bfeb29cc6836d2ff5528c30ee13a4300b79cc4c250a452b9c5c81fdfc9bac90"
ORIGINAL_AUDITOR_SHA = "d8b6e8b8ed6862af69e78ecc63758b13fd03378862aa6c6c6644052fa36d85d4"
KEYS = ("coefficient_FP64","intercept_FP64","coefficient_FP32","intercept_FP32")
np = None


def initialize_numpy():
    global np
    original_audit.initialize_numpy(); common.initialize_numpy(); np = original_audit.np


def selected_positive_fit(x,y,mask,alpha,summary,info,state):
    require(x.dtype == np.float32 and y.dtype == np.float64 and mask.dtype == np.bool_ and y.shape == mask.shape and len(x) == len(y) and alpha in NEW_ALPHAS and np.isfinite(x).all() and np.isfinite(y[mask]).all(),"FINITE_SELECTED_NEW_POSITIVE_RIDGE_INPUT")
    require(set(state) == set(KEYS) and all(state["coefficient_"+suffix].shape == (x.shape[1],y.shape[1]) and state["intercept_"+suffix].shape == (y.shape[1],) and state["coefficient_"+suffix].dtype == state["intercept_"+suffix].dtype == dtype and np.isfinite(state["coefficient_"+suffix]).all() and np.isfinite(state["intercept_"+suffix]).all() for suffix,dtype in (("FP64",np.float64),("FP32",np.float32))),"FINITE_SELECTED_FUSED_COEFFICIENT_SHAPES_AND_DTYPES")
    for key in ("coefficient","intercept"): base.exact(state[key+"_FP32"],state[key+"_FP64"].astype(np.float32),"selected_fused_cast_"+key)
    x64=x.astype(np.float64); mean,rms,constant=original_audit.scale(x64)
    for key,wanted in (("global_feature_mean_FP64",mean),("global_feature_scale_FP64",rms)):
        require(info[key].dtype == np.float64,"FP64_GLOBAL_FIT_SCALING"); prior.close(info[key],wanted,"extension_"+key,atol=1e-12,rtol=1e-12)
    base.exact(info["global_constant_features"],constant,"extension_constant_features")
    groups={}
    for j in range(y.shape[1]): groups.setdefault(mask[:,j].tobytes(),[]).append(j)
    require(summary["normalization"] == "GLOBAL_FIT" and summary["alpha_grid"] == list(NEW_ALPHAS) and summary["design_singular_rcond"] == 1e-8 and summary["positive_alpha_uses_all_singular_directions"] is summary["coefficient_fused_to_original_feature_units"] is True and summary["intercept_penalized"] is summary["unobserved_response_placeholders_used"] is summary["TUNE_DEV_used_for_fit"] is False and summary["group_count"] == len(summary["groups"]) == len(groups),"NEW_ALPHA_ONLY_GLOBAL_SCALING_INTERCEPT_AND_MASK_CONTRACT")
    expected_c=np.zeros_like(state["coefficient_FP64"]); expected_b=np.zeros_like(state["intercept_FP64"]); diagnostics=[]
    for index,(columns,recorded) in enumerate(zip(groups.values(),summary["groups"])):
        valid=mask[:,columns[0]]; n=int(valid.sum()); require(n >= 2 and recorded["response_columns"] == columns and recorded["observed_count"] == n,"TRUE_OBSERVED_EXTENSION_RESPONSE_GROUP")
        names=recorded["array_members"]; base.exact(info["group_"+str(index)+"_response_columns"],np.asarray(columns,np.int64),"group_response_columns"); base.exact(info[names["observed_mask"]],valid,"group_observed_mask")
        local_mean=x64[valid].mean(0)
        for key,wanted in (("feature_mean_FP64",local_mean),("feature_scale_FP64",rms)):
            require(info[names[key]].dtype == np.float64,"FP64_GROUP_CENTER_AND_GLOBAL_RMS"); prior.close(info[names[key]],wanted,"group_"+key,atol=1e-12,rtol=1e-12)
        base.exact(info[names["constant_features"]],constant,"group_global_constant_columns")
        a=(x64[valid]-local_mean)/rms; a[:,constant]=0; b=y[np.ix_(valid,columns)]; response_mean=b.mean(0); centered=b-response_mean
        standardized=a.T@np.linalg.solve(a@a.T+n*alpha*np.eye(n),centered)
        coefficient=standardized/rms[:,None]; coefficient[constant]=0; expected_c[:,columns]=coefficient; expected_b[columns]=response_mean-local_mean@coefficient
        singular=info[names["singularvalues"]]; require(singular.dtype == np.float64 and singular.shape == (min(a.shape),) and np.isfinite(singular).all() and (singular >= 0).all() and (np.diff(singular) <= 0).all(),"RECORDED_EXTENSION_SPECTRUM_SCHEMA")
        rank=int((singular > singular[0]*1e-8).sum()) if singular.size and singular[0] > 0 else 0; require(recorded["minimum_norm_retained_rank"] == rank,"RECORDED_RCOND_RANK_INTERNAL_CONSISTENCY")
        residual=a@standardized-centered; normal=a.T@residual+n*alpha*standardized
        diagnostics.append({"response_columns":columns,"observed_count":n,"solver":"independent_dual_n_by_n_positive_ridge","normal_equation_max_abs":float(np.abs(normal).max(initial=0.)),"recorded_rank":rank,"spectrum_independently_recomputed":False})
    prior.close(state["coefficient_FP64"],expected_c,"selected_extension_coefficient64",atol=1e-8,rtol=1e-7); prior.close(state["intercept_FP64"],expected_b,"selected_extension_intercept64",atol=1e-8,rtol=1e-7)
    return {"mask_groups":diagnostics,"selected_new_fit_only_independently_solved":True,"unselected_coefficients_independently_refitted":False,"alpha":alpha}


def zero_controls(x,y,mask):
    require(x.dtype == np.float32 and x.ndim == 2 and y.dtype == np.float64 and mask.dtype == np.bool_ and y.shape == mask.shape and len(x) == len(y) and np.isfinite(y[mask]).all() and (mask.sum(0) >= 2).all(),"TRUE_OBSERVED_FIT_INNOVATIONS_FOR_ZERO_CONTROLS")
    mean=np.array([y[mask[:,j],j].mean() for j in range(y.shape[1])],np.float64)
    return {"INTERCEPT_ONLY":mean,"CURRENT_C_ANCHOR":np.zeros_like(mean)}


def zero_records(feature_dimension,intercept):
    require(type(feature_dimension) is int and feature_dimension > 0 and intercept.dtype == np.float64 and intercept.ndim == 1 and np.isfinite(intercept).all(),"FINITE_ZERO_CONTROL_HASH_INPUT")
    shape=(feature_dimension,len(intercept)); records={}; digest=hashlib.sha256(); block=b"\0"*(1024*1024)
    for name in sorted(KEYS):
        dtype=np.dtype(np.float32 if name.endswith("FP32") else np.float64); dimensions=shape if name.startswith("coefficient") else intercept.shape; array_digest=hashlib.sha256()
        for part in (name.encode(),str(dtype).encode(),str(dimensions).encode()): digest.update(part)
        if name.startswith("coefficient"):
            remaining=math.prod(dimensions)*dtype.itemsize
            while remaining:
                chunk=block[:min(len(block),remaining)]; array_digest.update(chunk); digest.update(chunk); remaining-=len(chunk)
        else:
            payload=np.ascontiguousarray(intercept.astype(dtype)).tobytes(); array_digest.update(payload); digest.update(payload)
        records[name]={"shape":list(dimensions),"dtype":str(dtype),"sha256":array_digest.hexdigest()}
    return records,digest.hexdigest()


def zero_prediction(anchor,intercept):
    require(anchor.dtype == np.float32 and anchor.ndim == 2 and intercept.shape == (48*anchor.shape[1],),"ZERO_CONTROL_H48_CURRENT_C_SHAPE")
    return (np.zeros((len(anchor),len(intercept)),np.float32)+intercept.astype(np.float32)).reshape(-1,48,anchor.shape[1])+anchor[:,None]


def check_menu(row,old):
    require([p["candidate_id"] for p in row["candidates"]] == list(MENU),"TWELVE_EXACT_ORDERED_EXTENSION_CANDIDATES")
    for index,(candidate,accepted) in enumerate(zip(row["candidates"][:7],old["candidates"])):
        extra={"candidate_id":MENU[index],"kind":"RIDGE","new_in_this_extension":False,"original_candidate_index":index,"original_prediction_artifact":accepted["prediction_artifact"],"new_FIT_TUNE_inference":False}
        require(candidate == {**accepted,**extra},"IMMUTABLE_ACCEPTED_ORIGINAL_CANDIDATE")
    for index,candidate in enumerate(row["candidates"][7:]):
        kind="RIDGE" if index < 3 else MENU[7+index]; alpha=NEW_ALPHAS[index] if index < 3 else None
        require(candidate["kind"] == kind and candidate["alpha"] == alpha and candidate["new_in_this_extension"] is candidate["new_FIT_TUNE_inference"] is True,"EXACT_NEW_ALPHA_AND_ZERO_CONTROL_IDENTITIES")
    require(all(p["status"] == "COMPLETE" and math.isfinite(p["TUNE_MSE"]) and p["TUNE_MSE"] == p["TUNE"]["scores_all_observed"]["normalized"]["mse"] for p in row["candidates"]),"COMPLETE_OWN_ACTUAL_FP32_TUNE_MENU")
    chosen=min(row["candidates"],key=lambda p:(p["TUNE_MSE"],MENU.index(p["candidate_id"])))
    require(row["selected_candidate"] == chosen["candidate_id"] and row["selected_kind"] == chosen["kind"] and row["selected_alpha"] == chosen["alpha"] and row["selected_TUNE_MSE"] == chosen["TUNE_MSE"] and row["FIT"] == chosen["FIT"] and row["TUNE"] == chosen["TUNE"] and row["selected_original_candidate"] == (not chosen["new_in_this_extension"]),"OWN_TUNE_ONLY_EXACT_MENU_ORDER_AND_REUSE")
    original_audit.same(row["selected_FIT_TUNE_prediction_artifact"],chosen["prediction_artifact"])
    require(row["selected_coefficient_value_sha256"] == chosen["coefficient_value_sha256"] and row["selected_new_alpha_upper_boundary"] == (chosen["alpha"] == NEW_ALPHAS[-1]) and row["zero_slope_selected"] == (chosen["kind"] != "RIDGE") and row["history_features_needed_to_predict"] == (chosen["kind"] == "RIDGE") and row["encoder_needed_to_predict"] == (chosen["kind"] == "RIDGE" and row["route"] != "RAW_HISTORY512") and row["F_needed_to_predict"] is False,"ZERO_SLOPE_FEATURE_REQUIREMENTS_NOT_COMPRESSION_BENEFIT")
    if row["selected_original_candidate"]:
        require(chosen["alpha"] == old["selected_alpha"] and chosen["coefficient_value_sha256"] == old["selected_coefficient_value_sha256"],"RESELECTED_OLD_CANDIDATE_IS_ORIGINAL_WINNER"); original_audit.same(row["selected_coefficients"],old["selected_coefficients"])
    return chosen


def metadata(path,check):
    require(isinstance(RESULT_SHA,str) and len(RESULT_SHA) == 64,"ACTUAL_RIDGE_EXTENSION_RESULT_NOT_REGISTERED"); binding=base.file_record(path); require(binding["sha256"] == RESULT_SHA,"EXACT_ACTUAL_RIDGE_EXTENSION_RESULT"); result=prior.verified_json(binding,check)
    require(result["status"] == "COMPLETE" and not result["failed_paths"] and result["controller_source"]["sha256"] == CONTROLLER_SHA and result["module_source"]["sha256"] == MODULE_SHA and result["original_ridge_diagnostic"]["sha256"] == ORIGINAL_SHA and result["original_ridge_audit"]["sha256"] == ORIGINAL_AUDIT_SHA and result["candidate_order"] == list(MENU),"COMPLETE_FROZEN_EXTENSION_AND_ACCEPTED_V1")
    expected={"extension_fit_attempts":10,"extension_fits_completed":10,"new_positive_alpha_candidates_completed":30,"new_zero_slope_candidates_completed":20,"new_FIT_TUNE_prediction_evaluations_completed":100,"reused_original_candidates":70,"reused_original_FIT_TUNE_prediction_evaluations":140,"new_selected_DEV_prediction_evaluations":5,"reused_original_DEV_prediction_evaluations":5,"resolved_selection_locks":10,"new_F_forward_calls":0,"new_F_model_objects_loaded":0,"new_native_training_updates":0,"new_optimizer_updates":0}
    require(all(result[k] == v for k,v in expected.items()),"EXACT_EXTENSION_NEW_AND_REUSED_EXECUTION_COUNTS")
    for key in ("new_E_D_fit","new_backbone_training","whole_deployment_cost_measured","original_v1_whole_cost_reassigned_to_extension","protected_TEST_accessed","independent_holdout","strict_K_only"): require(result[key] is False,"ADAPTIVE_EXTENSION_SCOPE:"+key)
    require(set(result["datasets"]) == set(DATASETS) and all(set(result["datasets"][data]["routes"]) == {route for d,route in CONDITIONS if d == data} for data in DATASETS),"EXACT_TEN_HISTORY_PATHS_NO_NEW_CURRENT_C_FIT")
    return binding,result


def check_registry(result,registry,load):
    require(registry["status"] == "ALL_TEN_RESOLVED_BEFORE_DEV" and registry["DEV_opened_before_seal"] is False and result["all_10_locks_before_DEV"] is True and len(registry["conditions"]) == result["resolved_selection_locks"] == 10,"ALL_TEN_EXTENSION_LOCKS")
    locks={}
    for item in registry["conditions"]:
        key=item["dataset"],item["route"]; require(key in CONDITIONS and key not in locks and item["status"] == "COMPLETE","TEN_UNIQUE_COMPLETE_EXTENSION_LOCKS"); row=result["datasets"][key[0]]["routes"][key[1]]; original_audit.same(row["selection_lock"],item["selection_lock"]); lock=load(item["selection_lock"])
        require(lock["status"] == row["status"] == "COMPLETE" and lock["dataset"] == key[0] and lock["route"] == key[1] and lock["DEV_opened_before_lock"] is False and lock["sealed_unix"] <= registry["sealed_unix"] and all(lock[k] == row[k] for k in ("selected_candidate","selected_kind","selected_alpha","selected_TUNE_MSE","selected_coefficients","selected_coefficient_value_sha256","selected_original_candidate")),"EXACT_PRE_DEV_SELECTION_LOCK"); locks[key]=lock
    return locks


def check_reused_DEV(entry,original):
    require(entry == {**original,"original_DEV_prediction_reused":True,"new_DEV_inference":False,"original_DEV_source":original["artifact"]},"EXACT_ACCEPTED_ORIGINAL_DEV_REUSE")


def audit_result(path,meter):
    check=meter.check; binding,result=metadata(path,check); old=prior.verified_json(result["original_ridge_diagnostic"],check); accepted=prior.verified_json(result["original_ridge_audit"],check); cross=prior.verified_json(result["cross_moment_diagnostic"],check); science=prior.verified_json(result["causal_lag_diagnostic"],check)
    require(old["status"] == "COMPLETE" and accepted["status"] == "PASS" and not accepted.get("partial_paths",[]) and accepted["audit_source"]["sha256"] == ORIGINAL_AUDITOR_SHA and result["F_parent_signatures_reused"] == old["F_parent_signatures_reused"],"ACCEPTED_V1_AUDIT_AND_REUSED_F_METADATA")
    original_audit.same(accepted["diagnostic"],result["original_ridge_diagnostic"]); original_audit.same(result["original_ridge_selection_registry"],old["selection_registry"]); original_audit.same(accepted["selection_registry"],old["selection_registry"])
    for key in ("original_controller_source","original_module_source"): original_audit.same(result[key],old[key.removeprefix("original_")])
    for key in ("cross_moment_diagnostic","cross_moment_audit","cross_state_registry","cross_auditor_source","causal_lag_diagnostic"): original_audit.same(result[key],old[key])
    guarded=[binding,result["original_ridge_diagnostic"],result["original_ridge_audit"],accepted["audit_source"],result["controller_source"],result["module_source"],result["plan"],*result["source_records"],*result["input_records"],*result["protected_prior_records"],*original_audit.receipt_check(result["resource_terminal"],check)]
    for item in guarded:
        common.bound(item,check)
        if "snapshot_path" in item: require(prior.file_sha(item["snapshot_path"]) == item["sha256"],"EXECUTED_EXTENSION_SOURCE_SNAPSHOT")
    registry=prior.verified_json(result["selection_registry"],check); require(registry["status"] == "ALL_TEN_RESOLVED_BEFORE_DEV" and registry["DEV_opened_before_seal"] is False and result["all_10_locks_before_DEV"] is True and len(registry["conditions"]) == 10 and result["resource_terminal"]["started_unix"] <= registry["sealed_unix"] <= result["resource_terminal"]["ended_unix"],"ALL_TEN_LOCKS_WITHIN_COMPLETE_CPU_RECEIPT")
    locks=check_registry(result,registry,lambda item:prior.verified_json(item,check)); guarded += [item["selection_lock"] for item in registry["conditions"]]
    output=Path(binding["path"]).parent; loaded=[]; produced_paths=set(); contexts={}
    def load(item): loaded.append(item); return prior.verified_arrays(item,check)
    def produced(item,name):
        target=Path(item["path"]).resolve(); require(target == output/name and target not in produced_paths,"UNIQUE_NEW_EXTENSION_ARTIFACT_PATH"); produced_paths.add(target); return load(item)
    def features(data,route,split,sources):
        record=science["datasets"][data]["spaces"]["PCA"]["conditions"]["L7_RECON"]["splits"][split]["input_source"]; original_audit.same(sources["input_source"],record)
        if (data,split) not in contexts:
            raw=load(record); c,_,dev,dn=DATASETS[data]; prior.check_raw_split(raw,channels=c,windows={"fit":256,"tune":64,dev:dn}[split]); contexts[data,split]=raw
        raw=contexts[data,split]; latent=None
        if route != "RAW_HISTORY512":
            owner=cross["datasets"][data]["conditions"][route.removesuffix("_HISTORY512")]; original_audit.same(sources["encoded_past_source"],owner["splits"][split]["artifact"]); original_audit.same(sources["frozen_E_D_source"],owner["state_artifact"]); values=load(sources["encoded_past_source"]); base.exact(values["origins"],raw["origins"],"original_encoded_past_origins"); latent=values["latent_history"]
            require(latent.shape == (len(raw["history"]),512,owner["K"]),"EXACT_FROZEN_ENCODED_PAST_K")
        else: require(sources["encoded_past_source"] is sources["frozen_E_D_source"] is None,"RAW_HISTORY_NEEDS_NO_ENCODER")
        return (*original_audit.feature_response(raw,route,latent),raw)
    totals={key:0 for key in ("extension_conditions_checked","selected_new_positive_fits_independently_solved","selected_original_fits_reused_without_refit","selected_zero_slope_conditions","new_positive_candidates_checked","new_zero_slope_candidates_checked","new_FIT_TUNE_prediction_evaluations_checked","original_candidates_immutable_checked","original_FIT_TUNE_evaluations_reused","new_selected_DEV_predictions_checked","original_DEV_predictions_reused_checked","selected_FP32_arithmetic_predictions_checked","zero_control_FP32_arithmetic_predictions_checked","mask_groups_in_selected_new_fits_checked","score_fields_checked","produced_NPZ_artifacts_checked","RAW_CURRENT_C_references_preserved")}; totals["selection_locks_checked"]=len(locks)
    report={"status":"PASS","diagnostic":binding,"original_ridge_diagnostic":result["original_ridge_diagnostic"],"original_ridge_audit":result["original_ridge_audit"],"selection_registry":result["selection_registry"],"controller_source":result["controller_source"],"module_source":result["module_source"],"plan":result["plan"],"totals":totals,"datasets":{},"partial_paths":[],"protected_TEST_accessed":False,"independent_holdout":False,"whole_cost_measured_or_v1_cost_reassigned":False,"unselected_positive_coefficient_values_unavailable_and_not_refitted":True,"F_GPU_optimizer_or_training_replayed":False,"DEV_access_events_independently_replayed":False,"scope":"Independent dual refit of selected NEW positive ridge only; selected old coefficients rely on exact accepted v1 audit bindings. All selected fused FP32 arithmetic, new saved scores, immutable original menus and true-observed FIT innovation mean/current-C controls are checked. Unselected new positive coefficient values are unavailable: hashes/schema and saved predictions, not every fit objective, are checked. Sealed records plus immutable source support ten locks before DEV, not access-event replay. Adaptive follow-up after exposed DEV, no independent generalization; changed coefficients have no new whole-cost measurement and never replace v1 cost."}
    for data in DATASETS:
        require(result["RAW_CURRENT_C_reference_only"][data] == old["datasets"][data]["routes"]["RAW_CURRENT_C"],"UNCHANGED_RAW_CURRENT_C_REFERENCE_ONLY"); totals["RAW_CURRENT_C_references_preserved"]+=1
    for data,route in CONDITIONS:
        row=result["datasets"][data]["routes"][route]; original=old["datasets"][data]["routes"][route]; require(row["route"] == route and row["status"] == "COMPLETE","COMPLETE_HISTORY_EXTENSION_CONDITION"); original_audit.same(row["original_selection_lock"],original["selection_lock"]); chosen=check_menu(row,original)
        fitx,fita,y,mask,fit=features(data,route,"fit",row["FIT_sources"]); tunex,tunea,_,_,tune=features(data,route,"tune",row["TUNE_sources"])
        require(row["FIT_feature_binding"] == original["FIT_feature_binding"] == original_audit.array_record(fitx) and row["FIT_response_mask_binding"] == original["FIT_response_mask_binding"] == original_audit.array_record(mask) and row["feature_dimension"] == fitx.shape[1] and row["response_dimension"] == y.shape[1],"SAME_ORIGINAL_FEATURES_TRUE_OBSERVED_RESPONSES_AND_DIMENSIONS")
        controls=zero_controls(fitx,y,mask); state=load(row["selected_coefficients"]); records,digest=original_audit.coefficient_records(state); require(records == chosen["coefficient_array_records"] and digest == row["selected_coefficient_value_sha256"],"EXACT_SELECTED_COEFFICIENT_RECORDS_AND_VALUES")
        info=produced(row["fit_artifact"],data+"_"+route+"_EXTENSION_FIT_INFO.npz"); totals["produced_NPZ_artifacts_checked"]+=1; fit_diagnostic=None
        if row["selected_original_candidate"]:
            fit_diagnostic={"accepted_v1_fit_diagnostic":accepted["datasets"][data][route]["fit_diagnostic"],"selected_old_objective_not_refitted":True}; totals["selected_original_fits_reused_without_refit"]+=1
        else:
            target=Path(row["selected_coefficients"]["path"]).resolve(); require(target == output/(data+"_"+route+"_SELECTED_COEFFICIENTS.npz") and target not in produced_paths,"NEW_SELECTED_COEFFICIENTS_REGISTERED_PATH"); produced_paths.add(target); totals["produced_NPZ_artifacts_checked"]+=1
            if chosen["kind"] == "RIDGE": fit_diagnostic=selected_positive_fit(fitx,y,mask,chosen["alpha"],row["fit_metadata"],info,state); totals["selected_new_positive_fits_independently_solved"]+=1; totals["mask_groups_in_selected_new_fits_checked"]+=len(fit_diagnostic["mask_groups"])
            else: totals["selected_zero_slope_conditions"]+=1
        selected_saved=None; zero_metadata={}
        for index,candidate in enumerate(row["candidates"]):
            if index < 7:
                guarded.append(candidate["prediction_artifact"]); totals["original_candidates_immutable_checked"]+=1; totals["original_FIT_TUNE_evaluations_reused"]+=2
                if candidate is not chosen: continue
                saved=load(candidate["prediction_artifact"])
            else:
                saved=produced(candidate["prediction_artifact"],data+"_"+route+"_"+candidate["candidate_id"]+".npz"); totals["produced_NPZ_artifacts_checked"]+=1
                require(set(candidate["coefficient_array_records"]) == set(KEYS) and len(candidate["coefficient_value_sha256"]) == 64,"PRESERVED_NEW_CANDIDATE_COEFFICIENT_HASH_SCHEMA")
                for name,record in candidate["coefficient_array_records"].items(): require(record["shape"] == ([fitx.shape[1],y.shape[1]] if name.startswith("coefficient") else [y.shape[1]]) and record["dtype"] == ("float32" if name.endswith("FP32") else "float64") and len(record["sha256"]) == 64,"NEW_COEFFICIENT_ARRAY_SHAPES_DTYPES_HASHES")
                if candidate["kind"] != "RIDGE":
                    mean=controls[candidate["kind"]]; expected_records,expected_digest=zero_records(fitx.shape[1],mean); require(candidate["coefficient_array_records"] == expected_records and candidate["coefficient_value_sha256"] == expected_digest,"INDEPENDENT_ZERO_SLOPE_COEFFICIENT_AND_INTERCEPT_HASHES"); zero_metadata[candidate["kind"]]={"intercept_FP64":mean.tolist(),"intercept_FP32":mean.astype(np.float32).tolist(),"coefficient_value_sha256":expected_digest}; totals["new_zero_slope_candidates_checked"]+=1
                else: totals["new_positive_candidates_checked"]+=1
            require(set(saved) == {"FIT_prediction_FP32","TUNE_prediction_FP32","FIT_origins","TUNE_origins"},"EXACT_CANDIDATE_FIT_TUNE_PREDICTION_MEMBERS")
            for label,x,anchor,raw in (("FIT",fitx,fita,fit),("TUNE",tunex,tunea,tune)):
                base.exact(saved[label+"_origins"],raw["origins"],label+"_original_origins"); prediction=saved[label+"_prediction_FP32"]
                if index >= 7: totals["score_fields_checked"]+=original_audit.score_check(candidate[label],prediction,raw); totals["new_FIT_TUNE_prediction_evaluations_checked"]+=int(index >= 7)
                if candidate["kind"] != "RIDGE": base.exact(prediction,zero_prediction(anchor,controls[candidate["kind"]]),"true_observed_FIT_zero_control_"+label); totals["zero_control_FP32_arithmetic_predictions_checked"]+=1
                if candidate is chosen: base.exact(prediction,original_audit.deployed_prediction(x,anchor,state),"selected_actual_fused_FP32_"+label); totals["selected_FP32_arithmetic_predictions_checked"]+=1
            if candidate is chosen: selected_saved=saved
        require(selected_saved is not None,"SELECTED_FIT_TUNE_ARCHIVE_CHECKED")
        entry=row["DEV"]; dev=DATASETS[data][2]
        if row["selected_original_candidate"]:
            check_reused_DEV(entry,original["DEV"]); saved=load(entry["artifact"]); totals["original_DEV_predictions_reused_checked"]+=1
        else:
            require(entry["original_DEV_prediction_reused"] is False and entry["new_DEV_inference"] is True,"NEW_SELECTED_EXTENSION_DEV_FORWARD"); saved=produced(entry["artifact"],data+"_"+route+"_SELECTED_DEV.npz"); totals["produced_NPZ_artifacts_checked"]+=1; totals["new_selected_DEV_predictions_checked"]+=1
        original_audit.same(entry["selected_coefficients"],row["selected_coefficients"]); x,anchor,_,_,raw=features(data,route,dev,entry); require(set(saved) == {"prediction_FP32","origins"},"EXACT_SELECTED_DEV_ARCHIVE_MEMBERS"); base.exact(saved["origins"],raw["origins"],"selected_DEV_origins"); base.exact(saved["prediction_FP32"],original_audit.deployed_prediction(x,anchor,state),"selected_extension_or_old_DEV_actual_FP32"); totals["score_fields_checked"]+=original_audit.score_check(entry,saved["prediction_FP32"],raw); totals["selected_FP32_arithmetic_predictions_checked"]+=1
        report["datasets"].setdefault(data,{})[route]={"selected_candidate":row["selected_candidate"],"selected_alpha":row["selected_alpha"],"selected_kind":row["selected_kind"],"selected_original_candidate":row["selected_original_candidate"],"selected_coefficients":row["selected_coefficients"],"selection_lock":row["selection_lock"],"selected_fit_diagnostic":fit_diagnostic,"FIT":row["FIT"],"TUNE":row["TUNE"],"DEV":entry,"zero_controls":zero_metadata,"candidates":[{k:p[k] for k in ("candidate_id","kind","alpha","TUNE_MSE","new_in_this_extension","prediction_artifact","coefficient_value_sha256")} for p in row["candidates"]]}; totals["extension_conditions_checked"]+=1
        del state,info,fitx,tunex,y,mask,x,selected_saved,saved
    require([totals[k] for k in ("extension_conditions_checked","selected_new_positive_fits_independently_solved","selected_original_fits_reused_without_refit","selected_zero_slope_conditions","new_positive_candidates_checked","new_zero_slope_candidates_checked","new_FIT_TUNE_prediction_evaluations_checked","original_candidates_immutable_checked","original_FIT_TUNE_evaluations_reused","new_selected_DEV_predictions_checked","original_DEV_predictions_reused_checked","selected_FP32_arithmetic_predictions_checked","zero_control_FP32_arithmetic_predictions_checked","produced_NPZ_artifacts_checked","RAW_CURRENT_C_references_preserved")]==[10,5,5,0,30,20,100,70,140,5,5,30,40,70,2] and len(produced_paths) == 70,"EXACT_EXTENSION_SELECTED_AND_REUSED_COVERAGE")
    for item in guarded+loaded+[result["selection_registry"]]: common.bound(item,check)
    totals["protected_source_input_result_hashes_checked"]=len({(str(Path(item["path"]).resolve()),item["sha256"]) for item in guarded+loaded+[result["selection_registry"]]})
    return report


def execute(path,output):
    from continuation_runtime_v1 import Meter,write_new
    output=Path(output).resolve(); require(not output.exists(),"FRESH_RIDGE_EXTENSION_AUDIT_OUTPUT"); meter=Meter("audit_ridge_regularization_extension_"+output.stem,gpu=False); source=base.file_record(__file__); failure=None
    try:
        metadata(path,meter.check); initialize_numpy()
        import sys
        sys.path.insert(0,str(ROOT/"research/smallk_failure_followup_20261014"))
        from prepare_inputs import install_protected_test_guard
        from threadpoolctl import threadpool_limits
        install_protected_test_guard(); meter.protected_check()
        with threadpool_limits(limits=2): report=audit_result(path,meter)
        meter.protected_check(); require(prior.file_sha(__file__) == source["sha256"],"EXTENSION_AUDITOR_SOURCE_UNCHANGED"); json.dumps(report,allow_nan=False)
    except Exception as error: failure={"type":type(error).__name__,"message":str(error)}; report={"status":"FAILED_PRESERVED","failure":failure}
    finally: report["audit_source"]=source; report["resource_terminal"]=meter.close(report["status"],failure); write_new(output,report)
    print(json.dumps({"status":report["status"],"totals":report.get("totals"),"failure":failure}),flush=True); return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--result",required=True); parser.add_argument("--output",required=True); args=parser.parse_args(); raise SystemExit(execute(args.result,args.output))
