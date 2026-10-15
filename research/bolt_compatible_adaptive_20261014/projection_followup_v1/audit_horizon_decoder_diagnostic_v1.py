"""Independent saved horizon-delta equations; actual result gate remains closed during SOURCE preparation."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys

import audit_linear_e_adaptation_v1 as base
import audit_new_direction_direct_ridge_v1 as ridge

prior,HERE,ROOT,require=base.prior,base.HERE,base.ROOT,base.require
ALPHAS=(0.,1e-4,.001,.01,.1,1.,10.,100.,1000.,10000.)
ARMS=("STATIC_DELTA","RAMP_DELTA")
LAGS=(0,1,2,4,8,16,32)
DATASETS={"robin":(17,4,"val",30),"beijing_aotizhongxin":(11,2,"eval",146)}
RESULT_SHA="5f158382640dac74ec63cc3d1005b385c898821d3c032d51b543f2c9b27d7dac"
CONTROLLER_SHA="151da5b0b4cc4cacdb9d7a547d12e7837dcdcb5ee4c5047a1d6e3993afe525c6"
CROSS_RESULT_SHA="99bffd5831143ca35947099fef9567f89dd209d83451661decee0682b2b8e13c"
CROSS_AUDIT_SHA="90c68a9057909fb5b6c23284e5a45523cdf126da178a5ba0d9d9e96a50dede30"
MODULE_SHA="406252e940ae676e7a78f69e067f5b3fa70456bdb1ea27d57169b8964ceabe0e"
np=sys.modules.get("numpy")


def initialize_numpy():
    global np
    base.initialize_numpy();np=sys.modules["numpy"]
    ridge.initialize_numpy()


def old_lag_features(history,future):
    require(history.dtype==future.dtype==np.float32 and history.ndim==future.ndim==3 and history.shape[0]==future.shape[0] and history.shape[2]==future.shape[2] and history.shape[1]>max(LAGS) and np.isfinite(history).all() and np.isfinite(future).all(),"FINITE_SAME_K_PAST_FUTURE_REQUIRED")
    n,length,k=history.shape;h=future.shape[1];result=np.empty((n,h,len(LAGS)*k),np.float32)
    for index in range(h):
        for j,lag in enumerate(LAGS):
            relative=index+1-lag
            value=future[:,relative-1] if relative>0 else history[:,length-1+relative]
            result[:,index,j*k:(j+1)*k]=value-history[:,-1-lag]
    return result


def gate_and_features(phi,arm):
    require(phi.dtype==np.float32 and phi.ndim==3 and phi.shape[1]>1 and np.isfinite(phi).all() and arm in ARMS,"REGISTERED_FINITE_FP32_HORIZON_FEATURES")
    h=phi.shape[1];g=np.ones(h,np.float32) if arm==ARMS[0] else np.array([np.float32(i)/np.float32(h-1) for i in range(h)],dtype=np.float32)
    return g,np.ascontiguousarray(phi*g[None,:,None],dtype=np.float32)


def actual_F_scale(phi):
    require(phi.dtype==np.float32 and phi.ndim==3 and phi.size and np.isfinite(phi).all(),"FINITE_UNGATED_ACTUAL_F_FIT_FEATURES")
    rms=np.linalg.norm(phi.astype(np.float64),axis=(0,1))/math.sqrt(phi.shape[0]*phi.shape[1]);constant=rms==0
    return np.where(constant,1.,rms),constant


def observed_residual(target,baseline,observed):
    require(target.dtype==baseline.dtype==np.float32 and target.shape==baseline.shape==observed.shape and observed.dtype==np.bool_ and np.isfinite(baseline).all() and np.isfinite(target[observed]).all(),"FINITE_OBSERVED_Y_AND_ORIGINAL_BASELINE32")
    response=np.zeros(target.shape,np.float64);response[observed]=target[observed].astype(np.float64)-baseline[observed].astype(np.float64)
    return response


def selected_fit(features,response,observed,scale,constant,alpha):
    require(features.dtype==np.float32 and features.ndim==3 and response.dtype==np.float64 and response.shape==observed.shape and response.shape[:2]==features.shape[:2] and observed.dtype==np.bool_ and scale.dtype==np.float64 and scale.shape==(features.shape[2],) and (scale>0).all() and constant.dtype==np.bool_ and constant.shape==scale.shape and alpha in ALPHAS and np.isfinite(features).all() and np.isfinite(response[observed]).all(),"REGISTERED_SELECTED_NONCENTERED_OBSERVED_RIDGE")
    n,h,p=features.shape;c=response.shape[2];design=features.astype(np.float64).reshape(n*h,p)/scale;design[:,constant]=0.;y=response.reshape(n*h,c);mask=observed.reshape(n*h,c);coefficient=np.zeros((p,c),np.float64);groups={};diagnostics=[]
    for channel in range(c):groups.setdefault(mask[:,channel].tobytes(),[]).append(channel)
    for columns in groups.values():
        valid=mask[:,columns[0]];count=int(valid.sum());a=design[valid];b=y[np.ix_(valid,columns)]
        if count:
            if alpha==0:
                weight,_,rank,singular=np.linalg.lstsq(a,b,rcond=1e-8);solver="direct_lstsq_design_singular_rcond1e-8"
            else:
                weight=np.linalg.solve(a.T@a+(count*alpha)*np.eye(p),a.T@b);singular=np.linalg.svd(a,compute_uv=False);rank=int((singular>singular[0]*1e-8).sum()) if singular.size and singular[0]>0 else 0;solver="independent_primal_positive_ridge_normal_solve"
            value=weight/scale[:,None];value[constant]=0.;coefficient[:,columns]=value;residual=a@weight-b;gradient=a.T@residual+(count*alpha)*weight
            diagnostics.append({"response_columns":columns,"observed_count":count,"rank":int(rank),"singularvalues":singular.tolist(),"solver":solver,"observed_residual_MSE":float(np.mean(residual**2)),"normal_equation_max_abs":float(np.abs(gradient).max(initial=0.)),"fallback":None})
        else:diagnostics.append({"response_columns":columns,"observed_count":0,"rank":0,"singularvalues":[],"solver":None,"fallback":"NO_OBSERVED_RESPONSE_ZERO_CORRECTION"})
    require(np.isfinite(coefficient).all(),"FINITE_INDEPENDENT_SELECTED_COEFFICIENT64")
    return coefficient,{"groups":diagnostics,"feature_centering":False,"intercept_fitted":False,"alpha":alpha,"positive_alpha_singular_cutoff_applied":False,"coefficient_fused_to_original_feature_units":True}


def deployed_prediction(baseline,ungated_features,coefficient,arm,*,zero=False):
    if zero:return baseline.copy()
    _,features=gate_and_features(ungated_features,arm);pieces=[]
    for start in range(0,len(features),32):
        block=features[start:start+32];correction=(block.reshape(-1,block.shape[2])@coefficient).reshape(len(block),block.shape[1],baseline.shape[2]);pieces.append(baseline[start:start+32]+correction)
    return np.ascontiguousarray(np.concatenate(pieces),dtype=np.float32)


def oracle_dependency_mask(observed):
    require(observed.dtype==np.bool_ and observed.ndim==3,"OBSERVED_FUTURE_C_BOOL_MASK")
    complete=observed.all(2);valid=complete.copy();h=observed.shape[1]
    for index in range(h):
        for lag in LAGS:
            previous=index-lag
            if previous>=0:valid[:,index]&=complete[:,previous]
    return valid


def select_candidate(candidates):
    menu=tuple("ALPHA_"+format(a,"g") for a in ALPHAS)+("ZERO_DELTA",)
    require(len(candidates)==len(menu) and [p["id"] for p in candidates]==list(menu) and all(p["status"]=="COMPLETE" and math.isfinite(p["TUNE_MSE"]) for p in candidates),"COMPLETE_REGISTERED_ELEVEN_CANDIDATE_MENU")
    return min(candidates,key=lambda row:(row["TUNE_MSE"],menu.index(row["id"])))


def array_record(value):
    return {"shape":list(value.shape),"dtype":str(value.dtype),"sha256":hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}


def check_parity(recorded,actual,expected,*,required=True):
    difference=actual.astype(np.float64)-expected.astype(np.float64);maximum=float(np.max(np.abs(difference),initial=0.));rmse=float(np.sqrt(np.mean(difference**2)));passed=bool(np.allclose(actual,expected,atol=1e-5,rtol=2e-6))
    require(recorded["pass"]==passed and recorded["atol"]==1e-5 and recorded["rtol"]==2e-6,"REGISTERED_ACTUAL_PARITY_SCOPE");prior.close([recorded["max_abs"],recorded["rmse"]],[maximum,rmse],"parity_numeric",atol=1e-10,rtol=1e-10)
    if required:require(passed,"REQUIRED_PARITY_FAILED")
    return passed


def score_check(entry,prediction,raw):
    require(entry["status"]=="COMPLETE" and prediction.dtype==np.float32 and prediction.shape==raw["target"].shape and np.isfinite(prediction).all(),"FINITE_FP32_SCORED_C_FORECAST")
    count=prior.compare_scores(entry["scores_all_observed"],prior.scores(prediction,raw["target"],raw["observed"],raw["std"]),"all_observed")
    complete=np.broadcast_to(raw["observed"].all(2,keepdims=True),raw["observed"].shape)
    if complete.any():count+=prior.compare_scores(entry["scores_complete_mask"],prior.scores(prediction,raw["target"],complete,raw["std"]),"complete_C")
    else:require(entry["scores_complete_mask"]=={"status":"NO_COMPLETE_C"},"EMPTY_COMPLETE_C_REPORTED")
    return count


def normalized_check(recorded,recalculated):
    for key in ("mse","mae","per_channel_mse","per_channel_mae"):
        values=recorded[key]
        if key.startswith("per_channel"):values=[0. if value is None and recalculated["observed_counts"][j]==0 else value for j,value in enumerate(values)]
        prior.close(values,recalculated[key],"temporal_"+key,atol=1e-10,rtol=1e-10)
    for key in ("observed_counts","active_channels"):require(recorded[key]==recalculated[key],"TEMPORAL_OBSERVED_CHANNEL_COUNT")
    return 6


def temporal_check(recorded,prediction,raw):
    require(len(recorded["per_horizon"])==48 and set(recorded["registered_intervals"])=={"1_16","17_32","33_48"} and recorded["horizon_averages_not_equal_full_observed_channel_macro"] is True,"REGISTERED_HORIZON_AND_INTERVAL_SCOPE");count=0
    for label,start,stop in [*((h,h,h+1) for h in range(48)),("1_16",0,16),("17_32",16,32),("33_48",32,48)]:
        row=recorded["per_horizon"][label] if type(label) is int else recorded["registered_intervals"][label];mask=raw["observed"][:,start:stop]
        if mask.any():count+=normalized_check(row,prior.scores(prediction[:,start:stop],raw["target"][:,start:stop],mask,raw["std"])["normalized"])
        else:require(row=={"status":"NO_OBSERVED_TARGETS"},"EMPTY_HORIZON_OBSERVATIONS_EXPLICIT")
    return count


def latent_check(recorded,streams,raw,state):
    complete=raw["observed"].all(2);count=int(complete.sum());require(recorded["complete_C_horizons"]==count and recorded["target_horizon_mask_only_for_latent_point_error"] is True and recorded["incomplete_C_vectors_encoded"] is False and recorded["C_lag_output_EY_or_floor_status"]=="NOT_MEASURED_IN_THIS_STAGE","LATENT_POINT_DIAGNOSTIC_COMPLETE_C_MASK_ONLY")
    truth=np.zeros((*raw["target"].shape[:2],state["E"].shape[1]),np.float32);truth[complete]=(raw["target"][complete]-state["center"])@state["E"]
    for mode in ("F","FLAT"):
        row=recorded[mode]
        if not count:require(row=={"status":"NO_COMPLETE_C"},"EMPTY_LATENT_DIAGNOSTIC_MASK");continue
        error=streams[mode]["latent_future"].astype(np.float64)-truth.astype(np.float64);valid=error[complete]
        for key,value in (("mse",float(np.mean(valid**2))),("mae",float(np.mean(np.abs(valid)))) ,("per_latent_mse",np.mean(valid**2,axis=0))):prior.close(row[key],value,"latent_"+mode+key,atol=1e-10,rtol=1e-10)
        require(row["per_horizon_complete_C_counts"]==complete.sum(0).tolist() and len(row["per_horizon_mse"])==48,"SAME_LATENT_HORIZON_COUNTS")
        for h in range(48):
            if complete[:,h].any():prior.close(row["per_horizon_mse"][h],float(np.mean(error[:,h][complete[:,h]]**2)),"latent_horizon_mse",atol=1e-10,rtol=1e-10)
            else:require(row["per_horizon_mse"][h] is None,"EMPTY_LATENT_HORIZON_NONE")
    return 2


def fit_info_check(row,info,features,response,observed,scale,constant,coefficient):
    summary=row["fit_metadata"];require(summary["alpha_grid"]==list(ALPHAS) and summary["intercept_fitted"] is summary["feature_centering"] is summary["TUNE_DEV_used_for_fit"] is summary["unobserved_response_placeholders_used"] is False and summary["positive_alpha_uses_all_singular_directions"] is True and summary["coefficient_fused_to_original_feature_units"] is True and summary["design_singular_rcond"]==1e-8,"SAME_NONCENTERED_NO_INTERCEPT_OBJECTIVE_AND_GRID")
    prior.close(info["scale_FP64"],scale,"shared_actual_F_FIT_RMS",atol=1e-12,rtol=1e-12);base.exact(info["constant_features"],constant,"shared_exact_zero_FIT_feature_columns")
    y=response.reshape(-1,response.shape[2]);mask=observed.reshape(-1,observed.shape[2]);groups={}
    for channel in range(y.shape[1]):groups.setdefault(mask[:,channel].tobytes(),[]).append(channel)
    require(summary["group_count"]==len(summary["groups"])==len(groups),"EXACT_OBSERVED_RESPONSE_GROUPS")
    for index,(columns,recorded) in enumerate(zip(groups.values(),summary["groups"])):
        valid=mask[:,columns[0]];count=int(valid.sum());names=recorded["array_members"];require(recorded["response_columns"]==columns and recorded["observed_count"]==count and recorded["fit_executed"]==bool(count),"OBSERVED_CHANNEL_GROUP_AND_SVD_EXECUTION");base.exact(info["group_"+str(index)+"_response_columns"],np.asarray(columns,np.int64),"group_response_columns");base.exact(info[names["observed_mask"]],valid,"group_observed_rows")
        a=features.astype(np.float64).reshape(-1,features.shape[2])[valid]/scale;a[:,constant]=0.;singular=np.linalg.svd(a,compute_uv=False) if count else np.array([],np.float64);rank=int((singular>singular[0]*1e-8).sum()) if singular.size and singular[0]>0 else 0
        prior.close(info[names["singularvalues"]],singular,"independent_singular_spectrum",atol=1e-9,rtol=1e-7);require(recorded["retained_rank"]==rank and recorded["fallback"]==("NO_OBSERVED_RESPONSE_ZERO_CORRECTION" if not count else None),"SVD_RANK_AND_UNOBSERVED_FALLBACK")
    if row["selected_kind"]=="ZERO_DELTA":base.exact(coefficient["coefficient_FP64"],np.zeros_like(coefficient["coefficient_FP64"]),"ZERO_selected_slope64");diagnostic={"selected_ZERO_no_fit_claimed":True}
    else:
        wanted,diagnostic=selected_fit(features,response,observed,scale,constant,row["selected_alpha"]);prior.close(coefficient["coefficient_FP64"],wanted,"independent_selected_delta64",atol=1e-8,rtol=1e-7)
    require(set(coefficient)=={"coefficient_FP64","coefficient_FP32"} and coefficient["coefficient_FP64"].dtype==np.float64 and coefficient["coefficient_FP32"].dtype==np.float32,"EXACT_SLOPE_ONLY_SELECTED_STATE");base.exact(coefficient["coefficient_FP32"],coefficient["coefficient_FP64"].astype(np.float32),"selected_delta_fused_FP32_cast")
    return len(groups),diagnostic


def audit_result(path,meter):
    require(all(isinstance(value,str) and len(value)==64 for value in (RESULT_SHA,CONTROLLER_SHA)),"ACTUAL_HORIZON_RESULT_NOT_YET_REGISTERED")
    check=meter.check;binding=base.file_record(path);require(binding["sha256"]==RESULT_SHA,"EXACT_ACTUAL_HORIZON_RESULT");result=prior.verified_json(binding,check)
    require(result["status"]=="COMPLETE" and not result["failed_paths"] and result["controller_source"]["sha256"]==CONTROLLER_SHA and result["module_source"]["sha256"]==MODULE_SHA and result["cross_moment_diagnostic"]["sha256"]==CROSS_RESULT_SHA and result["cross_moment_audit"]["sha256"]==CROSS_AUDIT_SHA,"COMPLETE_FROZEN_HORIZON_SOURCE_AND_PARENT")
    for key,value in (("fit_attempts",24),("fits_completed",24),("alpha_candidates_completed",240),("ZERO_candidates_completed",24),("candidate_FIT_TUNE_evaluations_completed",528),("selected_DEV_evaluations_completed",24),("resolved_selection_locks",24),("source_F_registered_calls_reused_metadata",104),("new_F_forward_calls",0),("new_F_model_objects_loaded",0),("new_optimizer_updates",0),("new_E_D_training_updates",0)):require(result[key]==value,"EXACT_EXECUTED_COUNT:"+key)
    require(all(result[k] is False for k in ("new_backbone_training","whole_deployment_cost_measured","protected_TEST_accessed","independent_holdout","strict_K_only","per_origin_fit","full_horizon_independent_decoders_fitted","same_capacity_PCA_and_RAW_claimed")) and result["all_24_locks_before_DEV"] is result["actual_F_ungated_FIT_RMS_shared_by_four_arms"] is True and result["output_space_minimum_status"]=="NOT_MEASURED_IN_THIS_STAGE","DECLARED_LIMITED_DECODER_DIAGNOSTIC_SCOPE")
    cross=prior.verified_json(result["cross_moment_diagnostic"],check);accepted=prior.verified_json(result["cross_moment_audit"],check);science=prior.verified_json(result["causal_lag_diagnostic"],check);ridge.same(accepted["diagnostic"],result["cross_moment_diagnostic"])
    require(cross["status"]=="COMPLETE" and accepted["status"]=="PASS" and result["F_parent_signatures_reused"]=={"before":cross["F_before"],"after":cross["F_after"]} and cross["F_before"]==cross["F_after"],"ACCEPTED_ACTUAL_F_CACHE_SIGNATURES_NOT_REPLAYED")
    guarded=[binding,result["cross_moment_diagnostic"],result["cross_moment_audit"],result["causal_lag_diagnostic"],result["plan"],result["controller_source"],result["module_source"],*result["source_records"],*result["input_records"],*result["protected_prior_records"],*ridge.receipt_check(result["resource_terminal"],check)]
    for item in guarded:
        require(prior.file_sha(item["path"])==item["sha256"],"ORIGINAL_SOURCE_INPUT_RESULT_HASH_PRESERVATION")
        if "snapshot_path" in item:require(prior.file_sha(item["snapshot_path"])==item["sha256"],"EXECUTED_SOURCE_SNAPSHOT_BYTES")
    registry=prior.verified_json(result["selection_registry"],check);require(registry["status"]=="ALL_24_RESOLVED_BEFORE_DEV" and registry["DEV_opened_before_seal"] is False and len(registry["conditions"])==24,"ALL_TWENTY_FOUR_TUNE_LOCKS")
    expected={(data,space,mode,arm) for data,(_,k,_,_) in DATASETS.items() for space in ("RAW_C","PCA_K"+str(k),"PCA_K"+str(2*k)) for mode in ("F","FLAT") for arm in ARMS};seen=set()
    totals={key:0 for key in ("decoder_fit_paths_checked","observed_mask_groups_checked","alpha_candidates_checked","alpha_zero_candidates_checked","positive_alpha_candidates_checked","ZERO_candidates_checked","FIT_TUNE_prediction_evaluations_checked","selected_DEV_prediction_evaluations_checked","selected_coefficient_states_checked","selected_fits_independently_solved","selected_ZERO_states_checked","selected_FP32_arithmetic_predictions_checked","selection_locks_checked","source_contexts_checked","shared_RMS_spaces_checked","latent_point_diagnostic_modes_checked","score_fields_checked","temporal_score_fields_checked")}
    report={"status":"PASS","diagnostic":binding,**{key:result[key] for key in ("cross_moment_diagnostic","cross_moment_audit","causal_lag_diagnostic","selection_registry","controller_source","module_source","plan")},"totals":totals,"datasets":{},"partial_paths":[],"selected_fits_only_independently_solved":True,"unselected_coefficient_values_unavailable_and_not_refitted":True,"F_GPU_optimizer_or_cost_replayed":False,"independent_OS_DEV_access_events_verified":False,"joint_output_space_minimum_computed":False,"C_output_EY_computed":False,"protected_TEST_accessed":False,"independent_holdout":False,"scope":"Independent saved latent point diagnostics, source lag buffers, shared actual-F uncentered RMS, selected per-C observed ridge and FP32 corrections, all candidate scores, horizon/interval errors and sealed TUNE selection. No F/GPU or new method fit; old joint floors are not reused. DEV remains adaptive. Raw original F/currentC baselines are exact source copies; Raw identity-D recombination is an auxiliary FP32 check only."}
    raw_cache={};contexts={};loaded=[]
    def load(record):loaded.append(record);return prior.verified_arrays(record,check)
    def context(data,space,split):
        key=data,space,split
        if key in contexts:return contexts[key]
        c,k,dev,dn=DATASETS[data];metadata=result["datasets"][data]["spaces"][space]["contexts"][split];source=science["datasets"][data]["spaces"]["PCA"]["conditions"]["L7_RECON"]["splits"][split]["input_source"];ridge.same(metadata["input_source"],source)
        if (data,split) not in raw_cache:raw_cache[data,split]=load(source);prior.check_raw_split(raw_cache[data,split],channels=c,windows={"fit":256,"tune":64,dev:dn}[split])
        raw=raw_cache[data,split];cache=None
        if space=="RAW_C":
            eye=np.eye(c,dtype=np.float32);state={"E":eye,"center":np.zeros(c,np.float32),"D":np.concatenate([eye,*[np.zeros_like(eye) for _ in LAGS[1:]]])};z=raw["history"];future=raw["raw_forecast"];base_F=raw["raw_forecast"]
            require(metadata["cached_latent_source"] is metadata["frozen_state_source"] is None and metadata["RAW_identity_E_lag0_identity_D"] is True,"RAW_IDENTITY_STATE_NOT_BORROWED_FROM_PCA")
        else:
            owner=cross["datasets"][data]["conditions"][space];entry=owner["splits"][split];ridge.same(metadata["cached_latent_source"],entry["artifact"]);ridge.same(metadata["frozen_state_source"],owner["state_artifact"]);ridge.same(entry["input_source"],source);cache=load(entry["artifact"]);state=load(owner["state_artifact"]);base.exact(cache["origins"],raw["origins"],"original_actual_F_cache_origins");z=cache["latent_history"];future=cache["latent_future"];base_F=cache["prediction"];require(metadata["RAW_identity_E_lag0_identity_D"] is False,"FIXED_PCA_NOT_RAW_IDENTITY")
        require(metadata["K"]==state["E"].shape[1],"SAME_FIXED_K");streams={}
        for mode in ("F","FLAT"):
            f=future if mode=="F" else np.broadcast_to(z[:,-1:],future.shape).copy();baseline=base_F if mode=="F" else np.broadcast_to(raw["raw_last"][:,None],raw["target"].shape).copy() if space=="RAW_C" else cache["FLAT_prediction"];phi=old_lag_features(z,f);stream={"features":phi,"baseline":baseline,"latent_history":z,"latent_future":f};streams[mode]=stream
            require(metadata["array_bindings"][mode]=={name:array_record(value) for name,value in stream.items()},"EXACT_FROZEN_STREAM_BASELINE_AND_LAG_ARRAY_BINDINGS")
            checks=metadata["checks"][mode];require(checks["same_old_decoder"]["required"]==(space!="RAW_C") and checks["same_old_decoder"]["parity_not_required"]==(space=="RAW_C"),"RAW_RECOMBINATION_AUXILIARY_PCA_DECODER_REQUIRED");check_parity(checks["original_encode"],z,(raw["history"]-state["center"])@state["E"]);recombined=raw["raw_last"][:,None]+phi@state["D"];check_parity(checks["same_old_decoder"],baseline,recombined,required=space!="RAW_C")
            if space!="RAW_C" and mode=="F":check_parity(checks["original_lag_features"],phi,cache["lag_features"])
            totals["score_fields_checked"]+=score_check(metadata["baseline_scores"][mode],baseline,raw);totals["temporal_score_fields_checked"]+=temporal_check(metadata["baseline_temporal_diagnostics"][mode],baseline,raw)
        totals["score_fields_checked"]+=score_check(metadata["current_C_persistence_scores"],np.broadcast_to(raw["raw_last"][:,None],raw["target"].shape).copy(),raw);totals["latent_point_diagnostic_modes_checked"]+=latent_check(metadata["latent_error_diagnostics"],streams,raw,state);totals["source_contexts_checked"]+=1;contexts[key]=(raw,streams,metadata);return contexts[key]
    for item in registry["conditions"]:
        data,space,mode,arm=(item[key] for key in ("dataset","space","mode","arm"));key=data,space,mode,arm;require(key in expected and key not in seen and item["status"]=="COMPLETE","UNIQUE_REGISTERED_DECODER_CONDITION");seen.add(key);name=mode+"_"+arm;row=result["datasets"][data]["spaces"][space]["conditions"][name]
        require(row["status"]=="COMPLETE" and row["mode"]==mode and row["arm"]==arm and row["space"]==space,"COMPLETE_CORRECT_ARM");ridge.same(item["selection_lock"],row["selection_lock"]);lock=prior.verified_json(item["selection_lock"],check);ridge.same(lock["selected_coefficients"],row["selected_coefficients"])
        require(lock["status"]=="COMPLETE" and lock["DEV_opened_before_lock"] is False and lock["sealed_unix"]<=registry["sealed_unix"] and all(lock[field]==row[field] for field in ("selected_candidate","selected_kind","selected_alpha","selected_TUNE_MSE","selected_coefficient_value_sha256")),"EXACT_TUNE_SELECTION_LOCK")
        fit,fitstreams,fitmeta=context(data,space,"fit");tune,tunestreams,tunemeta=context(data,space,"tune");require(row["FIT_sources"]==fitmeta and row["TUNE_sources"]==tunemeta and row["K"]==fitmeta["K"],"SAME_FIT_TUNE_CONTEXT_LINEAGE");scale,constant=actual_F_scale(fitstreams["F"]["features"]);response=observed_residual(fit["target"],fitstreams[mode]["baseline"],fit["observed"]);_,features=gate_and_features(fitstreams[mode]["features"],arm);state=load(row["selected_coefficients"]);records,digest=ridge.coefficient_records(state)
        require(digest==row["selected_coefficient_value_sha256"] and row["additional_slope_parameters"]==state["coefficient_FP64"].size==7*fitmeta["K"]*fit["target"].shape[2],"SAME_SELECTED_DELTA_AND_ADDITIONAL_SLOPE_COUNT");group_count,fitdiagnostic=fit_info_check(row,load(row["fit_artifact"]),features,response,fit["observed"],scale,constant,state);totals["observed_mask_groups_checked"]+=group_count;totals["selected_coefficient_states_checked"]+=1;totals["selected_ZERO_states_checked" if row["selected_kind"]=="ZERO_DELTA" else "selected_fits_independently_solved"]+=1
        chosen=select_candidate(row["candidates"]);require(chosen["id"]==row["selected_candidate"] and chosen["kind"]==row["selected_kind"] and chosen["alpha"]==row["selected_alpha"] and chosen["TUNE_MSE"]==row["selected_TUNE_MSE"] and chosen["coefficient_value_sha256"]==digest and chosen["coefficient_array_records"]==records,"OWN_ACTUAL_FP32_TUNE_WINNER_AND_COEFFICIENT_HASH");selected_predictions=None
        for index,candidate in enumerate(row["candidates"]):
            is_zero=index==len(ALPHAS);require(candidate["kind"]==("ZERO_DELTA" if is_zero else "ALPHA") and candidate["alpha"]==(None if is_zero else ALPHAS[index]) and candidate["TUNE_MSE"]==candidate["TUNE"]["scores_all_observed"]["normalized"]["mse"],"REGISTERED_CANDIDATE_KIND_ALPHA_AND_TUNE_SCORE")
            saved=load(candidate["prediction_artifact"])
            for label,raw,stream in (("FIT",fit,fitstreams[mode]),("TUNE",tune,tunestreams[mode])):
                prediction=saved[label+"_prediction_FP32"];base.exact(saved[label+"_origins"],raw["origins"],"candidate_origins");totals["score_fields_checked"]+=score_check(candidate[label],prediction,raw);totals["FIT_TUNE_prediction_evaluations_checked"]+=1
                if is_zero:base.exact(prediction,stream["baseline"],"ZERO_original_baseline_bytes");require(candidate["parity_checks"][label]=={"pass":True,"max_abs":0.,"ZERO_original_baseline_byte_exact":True},"ZERO_BYPASS_SCOPE")
                elif candidate is chosen:
                    wanted=deployed_prediction(stream["baseline"],stream["features"],state["coefficient_FP32"],arm);base.exact(prediction,wanted,"selected_separate_FP32_correction");_,gated=gate_and_features(stream["features"],arm);check_parity(candidate["parity_checks"][label],prediction,stream["baseline"]+gated@state["coefficient_FP32"])
                else:require(candidate["parity_checks"][label]["pass"] is True,"RECORDED_UNSELECTED_FORWARD_PARITY_NOT_INDEPENDENTLY_REPLAYED")
            if candidate is chosen:selected_predictions=saved;totals["selected_FP32_arithmetic_predictions_checked"]+=2
            totals["ZERO_candidates_checked" if is_zero else "alpha_candidates_checked"]+=1
            if not is_zero:totals["alpha_zero_candidates_checked" if candidate["alpha"]==0 else "positive_alpha_candidates_checked"]+=1
        ridge.same(row["selected_FIT_TUNE_prediction_artifact"],chosen["prediction_artifact"]);require(row["FIT"]==chosen["FIT"] and row["TUNE"]==chosen["TUNE"] and row["selected_FIT_TUNE_reused_without_new_inference"] is True,"SELECTED_FIT_TUNE_ENTRIES_REUSED_WITHOUT_NEW_FORWARD")
        for label,raw in (("FIT",fit),("TUNE",tune)):totals["temporal_score_fields_checked"]+=temporal_check(row["selected_temporal_diagnostics"][label],selected_predictions[label+"_prediction_FP32"],raw)
        raw,streams,devmeta=context(data,space,DATASETS[data][2]);dev=row["DEV"];require(dev["sources"]==devmeta,"SAME_SELECTED_DEV_CONTEXT");ridge.same(dev["selected_coefficients"],row["selected_coefficients"]);saved=load(dev["artifact"]);base.exact(saved["origins"],raw["origins"],"selected_DEV_origins");prediction=deployed_prediction(streams[mode]["baseline"],streams[mode]["features"],state["coefficient_FP32"],arm,zero=row["selected_kind"]=="ZERO_DELTA");base.exact(saved["prediction_FP32"],prediction,"selected_DEV_actual_FP32_or_ZERO");totals["score_fields_checked"]+=score_check(dev,prediction,raw);totals["temporal_score_fields_checked"]+=temporal_check(dev["temporal_diagnostics"],prediction,raw);totals["selected_DEV_prediction_evaluations_checked"]+=1;totals["selected_FP32_arithmetic_predictions_checked"]+=1;totals["decoder_fit_paths_checked"]+=1;totals["selection_locks_checked"]+=1
        report["datasets"].setdefault(data,{"spaces":{}})["spaces"].setdefault(space,{"conditions":{}})["conditions"][name]={"selected_candidate":row["selected_candidate"],"selected_kind":row["selected_kind"],"selected_alpha":row["selected_alpha"],"selected_TUNE_MSE":row["selected_TUNE_MSE"],"selected_coefficients":row["selected_coefficients"],"selected_upper_grid_boundary":row["selected_alpha"]==ALPHAS[-1],"additional_slope_parameters":row["additional_slope_parameters"],"independent_fit":fitdiagnostic,"FIT":row["FIT"],"TUNE":row["TUNE"],"DEV":dev}
    require(seen==expected and totals["decoder_fit_paths_checked"]==24 and totals["alpha_candidates_checked"]==240 and totals["alpha_zero_candidates_checked"]==24 and totals["positive_alpha_candidates_checked"]==216 and totals["ZERO_candidates_checked"]==24 and totals["FIT_TUNE_prediction_evaluations_checked"]==528 and totals["selected_DEV_prediction_evaluations_checked"]==24 and totals["source_contexts_checked"]==18,"FULL_REGISTERED_HORIZON_DIAGNOSTIC_COVERAGE")
    for data,(_,k,_,_) in DATASETS.items():
        for space in ("RAW_C","PCA_K"+str(k),"PCA_K"+str(2*k)):
            owner=result["datasets"][data]["spaces"][space];require(set(owner["conditions"])=={mode+"_"+arm for mode in ("F","FLAT") for arm in ARMS},"EXACT_FOUR_ARMS_WITHOUT_HIDDEN_WINNER");scales=[load(owner["conditions"][name]["fit_artifact"])["scale_FP64"] for name in owner["conditions"]]
            for other in scales[1:]:base.exact(other,scales[0],"all_four_arms_same_stored_actual_F_RMS")
            totals["shared_RMS_spaces_checked"]+=1;report["datasets"][data]["spaces"][space]["contexts"]={split:value[2] for (d,s,split),value in contexts.items() if d==data and s==space}
    for item in [*guarded,*loaded,result["selection_registry"],*[row["selection_lock"] for data in result["datasets"].values() for space in data["spaces"].values() for row in space["conditions"].values()]]:require(prior.file_sha(item["path"])==item["sha256"],"FINAL_SOURCE_AND_SAVED_ARRAYS_UNCHANGED")
    totals["protected_source_input_and_result_hashes_checked"]=len({(str(Path(item["path"]).resolve()),item["sha256"]) for item in [*guarded,*loaded]});return report


def execute(path,output):
    from continuation_runtime_v1 import Meter,write_new
    output=Path(output).resolve();require(not output.exists(),"FRESH_HORIZON_AUDIT_OUTPUT");meter=Meter("audit_horizon_decoder_"+output.stem,gpu=False);failure=None;source=base.file_record(__file__)
    try:
        require(all(isinstance(value,str) and len(value)==64 for value in (RESULT_SHA,CONTROLLER_SHA)),"ACTUAL_HORIZON_RESULT_NOT_YET_REGISTERED");meter.protected_check();initialize_numpy();sys.path.insert(0,str(ROOT/"research/smallk_failure_followup_20261014"))
        from prepare_inputs import install_protected_test_guard
        from threadpoolctl import threadpool_limits
        install_protected_test_guard()
        with threadpool_limits(limits=2):report=audit_result(path,meter)
        meter.protected_check();require(prior.file_sha(__file__)==source["sha256"],"AUDITOR_SOURCE_UNCHANGED");json.dumps(report,allow_nan=False)
    except Exception as error:failure={"type":type(error).__name__,"message":str(error)};report={"status":"FAILED_PRESERVED","failure":failure}
    finally:report["audit_source"]=source;report["resource_terminal"]=meter.close(report["status"],failure);write_new(output,report)
    print(json.dumps({"status":report["status"],"totals":report.get("totals"),"failure":failure}),flush=True);return 0 if report["status"]=="PASS" else 2


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--result",required=True);parser.add_argument("--output",required=True);args=parser.parse_args();raise SystemExit(execute(args.result,args.output))
