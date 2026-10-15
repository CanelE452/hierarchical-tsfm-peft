"""Selected-fit independent ridge audit; unselected coefficients and F are not replayed."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys

import audit_linear_e_adaptation_v1 as base

HERE,ROOT,prior,require=base.HERE,base.ROOT,base.prior,base.require
DATASETS={"robin":(17,4,"val",30),"beijing_aotizhongxin":(11,2,"eval",146)}
ALPHAS=(0.,1e-4,.001,.01,.1,1.,10.)
RESULT_SHA="b8d7eae62ce5bb52b3d46ffc543f916f8b81857028098370a5b2f8398510f1ed"
CONTROLLER_SHA="7dd19e1504f0ff4b323c52412fac492f2b5c17c23e1ba0d898691a672f5ac88e"
MODULE_SHA="967fb95eef55f9597a73a01efc0118d8c7e936586adb724f7c5a449334ab3717"
CROSS_SHA="99bffd5831143ca35947099fef9567f89dd209d83451661decee0682b2b8e13c"
CROSS_AUDIT_SHA="90c68a9057909fb5b6c23284e5a45523cdf126da178a5ba0d9d9e96a50dede30"
np=sys.modules.get("numpy")


def initialize_numpy():
    global np
    base.initialize_numpy();np=sys.modules["numpy"]


def same(a,b):require(Path(a["path"]).resolve()==Path(b["path"]).resolve() and a["sha256"]==b["sha256"],"EXACT_PATH_SHA_BINDING")


def array_record(a):return {"shape":list(a.shape),"dtype":str(a.dtype),"sha256":hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()}


def coefficient_records(state):
    digest=hashlib.sha256();records={}
    for name,value in sorted(state.items()):
        value=np.ascontiguousarray(value);records[name]=array_record(value)
        for part in (name.encode(),str(value.dtype).encode(),str(value.shape).encode(),value.tobytes()):digest.update(part)
    return records,digest.hexdigest()


def feature_response(raw,route,latent=None):
    require(raw["history"].dtype==raw["target"].dtype==raw["raw_last"].dtype==np.float32 and raw["observed"].dtype==np.bool_,"FP32_SOURCE_BOOL_OBSERVATIONS")
    base.exact(raw["history"][:,-1],raw["raw_last"],"original_current_C")
    if route=="RAW_CURRENT_C":
        x=raw["raw_last"].copy();anchor=np.zeros_like(x);mask=np.broadcast_to(raw["observed"].all(2,keepdims=True),raw["observed"].shape).copy()
    else:
        history=raw["history"] if route=="RAW_HISTORY512" else latent
        require(history is not None and history.dtype==np.float32 and history.ndim==3 and history.shape[:2]==raw["history"].shape[:2] and np.isfinite(history).all(),"ONLY_FINITE_FP32_PAST_FEATURES")
        x=np.ascontiguousarray((history-history[:,-1:]).reshape(len(history),-1));anchor=raw["raw_last"].copy();mask=raw["observed"].copy()
    y=np.zeros(raw["target"].shape,np.float64)
    endpoints=raw["target"][mask].astype(np.float64)
    if route=="RAW_CURRENT_C":y[mask]=endpoints
    else:y[mask]=endpoints-np.broadcast_to(anchor.astype(np.float64)[:,None],y.shape)[mask]
    require(np.isfinite(x).all() and np.isfinite(y[mask]).all(),"FINITE_TRUE_OBSERVED_FIT_RESPONSE")
    return x,anchor,y.reshape(len(y),-1),mask.reshape(len(mask),-1)


def scale(x):
    mean=x.mean(0);rms=np.sqrt(np.mean((x-mean)**2,axis=0));constant=rms==0
    return mean,np.where(constant,1.,rms),constant


def selected_fit(x,y,mask,alpha,normalization,summary,info,state):
    require(x.dtype==np.float32 and y.dtype==np.float64 and mask.dtype==np.bool_ and y.shape==mask.shape and len(x)==len(y) and alpha in ALPHAS and normalization in ("GLOBAL_FIT","PER_RESPONSE_MASK"),"REGISTERED_SELECTED_FIT_CONTRACT")
    require(set(state)=={"coefficient_FP64","intercept_FP64","coefficient_FP32","intercept_FP32"} and state["coefficient_FP64"].shape==(x.shape[1],y.shape[1]) and state["intercept_FP64"].shape==(y.shape[1],),"SELECTED_FUSED_COEFFICIENT_SHAPES")
    for suffix,dtype in (("FP64",np.float64),("FP32",np.float32)):
        require(all(state[k+suffix].dtype==dtype and np.isfinite(state[k+suffix]).all() for k in ("coefficient_","intercept_")),"FINITE_SELECTED_COEFFICIENT_PRECISIONS")
    for key in ("coefficient","intercept"):base.exact(state[key+"_FP32"],state[key+"_FP64"].astype(np.float32),"fused_cast_"+key)
    x64=x.astype(np.float64);global_mean,global_rms,global_constant=scale(x64)
    for key,wanted in (("global_feature_mean_FP64",global_mean),("global_feature_scale_FP64",global_rms)):
        require(info[key].dtype==np.float64,"GLOBAL_SCALING_FP64");prior.close(info[key],wanted,"independent_"+key,atol=1e-12,rtol=1e-12)
    base.exact(info["global_constant_features"],global_constant,"independent_global_constant")
    groups={}
    for column in range(y.shape[1]):groups.setdefault(mask[:,column].tobytes(),[]).append(column)
    require(len(groups)==summary["group_count"]==len(summary["groups"]) and summary["normalization"]==normalization and summary["alpha_grid"]==list(ALPHAS) and summary["design_singular_rcond"]==1e-8 and summary["positive_alpha_uses_all_singular_directions"] is True and summary["intercept_penalized"] is summary["unobserved_response_placeholders_used"] is summary["TUNE_DEV_used_for_fit"] is False and summary["coefficient_fused_to_original_feature_units"] is True,"FROZEN_SCALING_OBJECTIVE_AND_MASK_GROUPS")
    expected_c=np.zeros_like(state["coefficient_FP64"]);expected_b=np.zeros_like(state["intercept_FP64"]);diagnostics=[]
    for index,(columns,recorded) in enumerate(zip(groups.values(),summary["groups"])):
        valid=mask[:,columns[0]];n=int(valid.sum());require(n>=2 and recorded["response_columns"]==columns and recorded["observed_count"]==n,"EXACT_TRUE_OBSERVED_RESPONSE_GROUP")
        names=recorded["array_members"];base.exact(info["group_"+str(index)+"_response_columns"],np.asarray(columns,dtype=np.int64),"group_columns")
        base.exact(info[names["observed_mask"]],valid,"group_true_observed_rows")
        local_mean,local_rms,local_constant=scale(x64[valid]);rms,constant=(global_rms,global_constant) if normalization=="GLOBAL_FIT" else (local_rms,local_constant)
        for key,wanted in (("feature_mean_FP64",local_mean),("feature_scale_FP64",rms)):prior.close(info[names[key]],wanted,"independent_group_"+key,atol=1e-12,rtol=1e-12)
        base.exact(info[names["constant_features"]],constant,"group_constant_features")
        a=(x64[valid]-local_mean)/rms;a[:,constant]=0;b=y[np.ix_(valid,columns)];mean_y=b.mean(0);centered=b-mean_y
        if alpha==0:
            standardized,_,rank,singular=np.linalg.lstsq(a,centered,rcond=1e-8)
            require(recorded["minimum_norm_retained_rank"]==int(rank),"INDEPENDENT_ZERO_ALPHA_RETAINED_RANK");prior.close(info[names["singularvalues"]],singular,"independent_zero_alpha_singularvalues",atol=1e-9,rtol=1e-7)
        else:
            standardized=a.T@np.linalg.solve(a@a.T+(n*alpha)*np.eye(n),centered)
            singular=info[names["singularvalues"]];require(singular.dtype==np.float64 and singular.shape==(min(a.shape),) and np.isfinite(singular).all() and (singular>=0).all() and (np.diff(singular)<=0).all(),"RECORDED_SINGULAR_SPECTRUM_SCHEMA")
            require(recorded["minimum_norm_retained_rank"]==int((singular>singular[0]*1e-8).sum()) if singular.size and singular[0]>0 else recorded["minimum_norm_retained_rank"]==0,"RECORDED_CUTOFF_RANK_INTERNAL_CONSISTENCY")
        coefficient=standardized/rms[:,None];coefficient[constant]=0;expected_c[:,columns]=coefficient;expected_b[columns]=mean_y-local_mean@coefficient
        residual=a@standardized-centered;normal=a.T@residual+(n*alpha)*standardized
        diagnostics.append({"response_columns":columns,"observed_count":n,"solver":"direct_lstsq_rcond1e-8" if alpha==0 else "dual_n_by_n_positive_ridge_solve","retained_rank_recorded":recorded["minimum_norm_retained_rank"],"spectrum_independently_recomputed":alpha==0,"normal_equation_max_abs":float(np.abs(normal).max(initial=0.)),"observed_centered_residual_MSE":float(np.mean(residual**2))})
    prior.close(state["coefficient_FP64"],expected_c,"independent_selected_coefficient64",atol=1e-8,rtol=1e-7);prior.close(state["intercept_FP64"],expected_b,"independent_selected_intercept64",atol=1e-8,rtol=1e-7)
    return {"mask_groups":diagnostics,"normalization":normalization,"independent_selected_fit_only":True,"unselected_coefficients_independently_refitted":False,"coefficient_max_abs":float(np.abs(state["coefficient_FP64"]).max(initial=0.))}


def deployed_prediction(x,anchor,state):
    pieces=[]
    for start in range(0,len(x),32):pieces.append((x[start:start+32]@state["coefficient_FP32"]+state["intercept_FP32"]).reshape(-1,48,anchor.shape[1])+anchor[start:start+32,None])
    return np.ascontiguousarray(np.concatenate(pieces),dtype=np.float32)


def score_check(entry,prediction,raw):
    require(entry["status"]=="COMPLETE" and prediction.dtype==np.float32 and prediction.shape==raw["target"].shape and np.isfinite(prediction).all(),"FINITE_ACTUAL_FP32_SAVED_PREDICTION")
    count=prior.compare_scores(entry["scores_all_observed"],prior.scores(prediction,raw["target"],raw["observed"],raw["std"]),"actual_all_observed")
    complete=np.broadcast_to(raw["observed"].all(2,keepdims=True),raw["observed"].shape)
    if complete.any():count+=prior.compare_scores(entry["scores_complete_mask"],prior.scores(prediction,raw["target"],complete,raw["std"]),"actual_complete_C")
    else:require(entry["scores_complete_mask"]=={"status":"NO_COMPLETE_C"},"EMPTY_COMPLETE_C_IS_EXPLICIT")
    return count


def winner(candidates):
    require(len(candidates)==7 and [p["alpha"] for p in candidates]==list(ALPHAS) and all(p["status"]=="COMPLETE" and math.isfinite(p["TUNE_MSE"]) and p["TUNE_MSE"]==p["TUNE"]["scores_all_observed"]["normalized"]["mse"] for p in candidates),"SEVEN_OWN_ACTUAL_TUNE_CANDIDATES")
    return min(candidates,key=lambda p:(p["TUNE_MSE"],ALPHAS.index(p["alpha"])))


def metadata(path,check):
    require(isinstance(RESULT_SHA,str) and len(RESULT_SHA)==64,"ACTUAL_RIDGE_RESULT_NOT_REGISTERED");binding=base.file_record(path);require(binding["sha256"]==RESULT_SHA,"EXACT_ACTUAL_RIDGE_RESULT");result=prior.verified_json(binding,check)
    require(result["status"]=="COMPLETE" and not result["failed_paths"] and result["controller_source"]["sha256"]==CONTROLLER_SHA and result["module_source"]["sha256"]==MODULE_SHA,"COMPLETE_FROZEN_RIDGE_PRODUCER")
    require(result["cross_moment_diagnostic"]["sha256"]==CROSS_SHA and result["cross_moment_audit"]["sha256"]==CROSS_AUDIT_SHA,"ACCEPTED_CROSS_RESULT_AND_AUDIT")
    for key in ("new_F_forward_calls","new_F_model_objects_loaded","new_native_training_updates","new_optimizer_updates"):require(result[key]==0,"NO_NEW_F_OR_TRAINING:"+key)
    for key in ("new_backbone_training","old_ridge_coefficients_or_predictions_reused","whole_deployment_cost_measured","same_D_native_or_capacity_match_claimed","protected_TEST_accessed","independent_holdout","strict_K_only"):require(result[key] is False,"DIRECT_RIDGE_SCOPE:"+key)
    require(result["D_bypassed_by_direct_C_readouts"] is True and result["CPU_FP32_canonical_batch_size"]==32,"DIRECT_C_READOUT_BOUNDARY")
    for key,count in {"ridge_fit_attempts":12,"ridge_fits_completed":12,"alpha_candidates_completed":84,"FIT_TUNE_prediction_evaluations_completed":168,"selected_DEV_evaluations_completed":12,"resolved_selection_locks":12}.items():require(result[key]==count,"EXACT_COMPLETED_COUNT:"+key)
    return binding,result


def receipt_check(receipt,check):
    require(all(type(receipt.get(k)) in (int,float) and math.isfinite(receipt[k]) and receipt[k]>=0 for k in ("process_cpu_seconds","wall_seconds")),"CANONICAL_CPU_AND_WALL")
    binding={key:receipt[key] for key in ("path","sha256")};recorded=prior.verified_json(binding,check)
    require(recorded=={k:v for k,v in receipt.items() if k not in ("path","sha256")} and receipt["status"]=="COMPLETE" and receipt["gpu_process"] is False and receipt["original_budget_records_unchanged"] is True and receipt["elapsed_or_cumulative_budget_caps"] is None,"EXACT_CPU_ONLY_UNCAPPED_RECEIPT")
    authorization=prior.verified_json(receipt["authorization"],check);manifest=prior.verified_json(receipt["continuation_manifest"],check)
    require(authorization["status"]=="APPROVED_HUMAN_UNCAPPED_CONTINUATION" and authorization["latest_human_override"]=="상한내수행하지 말고 쭉해줘" and manifest["authorization_sha256"]==receipt["authorization"]["sha256"] and manifest["runtime_sha256"]==prior.file_sha(HERE/"continuation_runtime_v1.py"),"AUTHORIZATION_RUNTIME_MANIFEST_CHAIN")
    return [binding,receipt["authorization"],receipt["continuation_manifest"]]


def audit_result(path,meter):
    check=meter.check;binding,result=metadata(path,check);cross=prior.verified_json(result["cross_moment_diagnostic"],check);cross_audit=prior.verified_json(result["cross_moment_audit"],check);science=prior.verified_json(result["causal_lag_diagnostic"],check)
    same(cross_audit["diagnostic"],result["cross_moment_diagnostic"]);require(cross["status"]=="COMPLETE" and cross_audit["status"]=="PASS" and result["F_parent_signatures_reused"]=={"before":cross["F_before"],"after":cross["F_after"]} and cross["F_before"]==cross["F_after"],"ACCEPTED_PARENT_ARITHMETIC_AND_F_SIGNATURE_METADATA")
    guarded=[binding,result["cross_moment_diagnostic"],result["cross_moment_audit"],result["causal_lag_diagnostic"],result["plan"],*result["source_records"],*result["input_records"],*result["protected_prior_records"],*receipt_check(result["resource_terminal"],check)]
    for item in guarded:
        check();require(prior.file_sha(item["path"])==item["sha256"],"ORIGINAL_SOURCE_INPUT_RESULT_SHA")
        if "snapshot_path" in item:require(prior.file_sha(item["snapshot_path"])==item["sha256"],"EXECUTED_SOURCE_SNAPSHOT_SHA")
    registry=prior.verified_json(result["selection_registry"],check);require(registry["status"]=="ALL_TWELVE_RESOLVED_BEFORE_DEV" and registry["DEV_opened_before_seal"] is False and result["all_12_locks_before_DEV"] is True and len(registry["conditions"])==12 and result["resource_terminal"]["started_unix"]<=registry["sealed_unix"]<=result["resource_terminal"]["ended_unix"],"TWELVE_LOCK_REGISTRY_IN_SOURCE_RECEIPT")
    totals={key:0 for key in ("selected_ridge_fits_checked","mask_groups_checked","alpha_candidates_checked","FIT_TUNE_prediction_evaluations_checked","selected_DEV_prediction_evaluations_checked","selected_coefficient_FP32_arithmetic_predictions_checked","selection_locks_checked","score_fields_checked","selected_upper_grid_boundary_conditions")}
    report={"status":"PASS","diagnostic":binding,"predictive_cross_diagnostic":result["cross_moment_diagnostic"],"predictive_cross_audit":result["cross_moment_audit"],"selection_registry":result["selection_registry"],"controller_source":result["controller_source"],"module_source":result["module_source"],"plan":result["plan"],"totals":totals,"datasets":{},"selected_fits_only_independently_solved":True,"unselected_coefficient_values_unavailable_and_not_refitted":True,"F_GPU_optimizer_or_training_replayed":False,"independent_OS_DEV_access_events_verified":False,"whole_cost_measured_in_this_audit":False,"independent_holdout":False,"protected_TEST_accessed":False}
    contexts={};loaded=[]
    def load(record):loaded.append(record);return prior.verified_arrays(record,check)
    def context(data,split):
        if (data,split) not in contexts:
            source=science["datasets"][data]["spaces"]["PCA"]["conditions"]["L7_RECON"]["splits"][split]["input_source"];raw=load(source);c,_,dev,dn=DATASETS[data];prior.check_raw_split(raw,channels=c,windows={"fit":256,"tune":64,dev:dn}[split]);contexts[data,split]=(raw,source)
        return contexts[data,split]
    def features(data,route,split,sources):
        raw,record=context(data,split);same(sources["input_source"],record);latent=None
        require(sources["future_latent_or_oracle_used_as_features"] is False,"PAST_ONLY_FEATURE_DECLARATION")
        if route not in ("RAW_CURRENT_C","RAW_HISTORY512"):
            name=route.removesuffix("_HISTORY512");owner=cross["datasets"][data]["conditions"][name];entry=owner["splits"][split];same(entry["input_source"],record);same(sources["encoded_past_source"],entry["artifact"]);same(sources["frozen_E_D_source"],owner["state_artifact"]);saved=load(entry["artifact"]);base.exact(saved["origins"],raw["origins"],"accepted_encoded_past_origins");latent=saved["latent_history"]
            require(latent.shape==(len(raw["history"]),512,owner["K"]),"EXACT_SOURCE_K_PAST_SHAPE")
        else:require(sources["encoded_past_source"] is sources["frozen_E_D_source"] is None,"RAW_READOUT_NO_ENCODER_SOURCE")
        return (*feature_response(raw,route,latent),raw)
    seen=set()
    for item in registry["conditions"]:
        data,route=item["dataset"],item["route"];c,k,dev,_=DATASETS[data];allowed=("RAW_CURRENT_C","RAW_HISTORY512",*(kind+"_K"+str(width)+"_HISTORY512" for kind in ("PCA","CROSS_MOMENT") for width in (k,2*k)))
        require(route in allowed and (data,route) not in seen,"UNIQUE_TWELVE_REGISTERED_READOUTS");seen.add((data,route));row=result["datasets"][data]["routes"][route];same(item["selection_lock"],row["selection_lock"]);lock=prior.verified_json(row["selection_lock"],check)
        require(row["status"]==lock["status"]==item["status"]=="COMPLETE" and lock["dataset"]==data and lock["route"]==row["route"]==route and lock["DEV_opened_before_lock"] is False and lock["sealed_unix"]<=registry["sealed_unix"],"LOCKED_COMPLETE_CONDITION_BEFORE_DEV");guarded.append(row["selection_lock"])
        totals["selection_locks_checked"]+=1;chosen=winner(row["candidates"])
        require(row["selected_alpha"]==lock["selected_alpha"]==chosen["alpha"] and row["selected_TUNE_MSE"]==lock["selected_TUNE_MSE"]==chosen["TUNE_MSE"] and row["FIT"]==chosen["FIT"] and row["TUNE"]==chosen["TUNE"] and row["selected_FIT_TUNE_reused_without_new_inference"] is True,"OWN_ALPHA_TUNE_ONLY_WINNER_AND_REUSE")
        same(row["selected_coefficients"],lock["selected_coefficients"]);same(row["selected_FIT_TUNE_prediction_artifact"],chosen["prediction_artifact"]);require(row["selected_coefficient_value_sha256"]==lock["selected_coefficient_value_sha256"]==chosen["coefficient_value_sha256"],"LOCKED_SELECTED_COEFFICIENT_VALUES")
        state=load(row["selected_coefficients"]);records,digest=coefficient_records(state);require(records==chosen["coefficient_array_records"] and digest==chosen["coefficient_value_sha256"],"ARCHIVED_SELECTED_COEFFICIENT_HASHES")
        fitx,fita,y,mask,fit=features(data,route,"fit",row["FIT_sources"]);tunex,tunea,_,_,tune=features(data,route,"tune",row["TUNE_sources"])
        require(row["FIT_feature_binding"]==array_record(fitx) and row["FIT_response_mask_binding"]==array_record(mask) and row["feature_dimension"]==fitx.shape[1] and row["response_dimension"]==48*c and row["normalization"]==("PER_RESPONSE_MASK" if route=="RAW_CURRENT_C" else "GLOBAL_FIT") and row["feature_subtraction_FP32"]==(route!="RAW_CURRENT_C") and row["future_label_placeholders_excluded_by_mask"] is True,"EXACT_FEATURE_MASK_NORMALIZATION_AND_RESPONSE_DIMENSIONS")
        fit_diagnostic=selected_fit(fitx,y,mask,chosen["alpha"],row["normalization"],row["fit_metadata"],load(row["fit_artifact"]),state);totals["selected_ridge_fits_checked"]+=1;totals["mask_groups_checked"]+=len(fit_diagnostic["mask_groups"])
        for candidate in row["candidates"]:
            saved=load(candidate["prediction_artifact"])
            require(set(saved)=={"FIT_prediction_FP32","TUNE_prediction_FP32","FIT_origins","TUNE_origins"},"CANDIDATE_PREDICTION_ARCHIVE_EXACT_KEYS")
            for label,raw,x,anchor in (("FIT",fit,fitx,fita),("TUNE",tune,tunex,tunea)):
                base.exact(saved[label+"_origins"],raw["origins"],"candidate_"+label+"_origins");prediction=saved[label+"_prediction_FP32"];totals["score_fields_checked"]+=score_check(candidate[label],prediction,raw);totals["FIT_TUNE_prediction_evaluations_checked"]+=1
                if candidate is chosen:base.exact(prediction,deployed_prediction(x,anchor,state),"selected_fused_CPU32_"+label);totals["selected_coefficient_FP32_arithmetic_predictions_checked"]+=int(candidate is chosen)
            require(len(candidate["coefficient_value_sha256"])==64 and set(candidate["coefficient_array_records"])==set(state),"PRESERVED_CANDIDATE_COEFFICIENT_HASH_SCHEMA")
            for name,record in candidate["coefficient_array_records"].items():require(record["shape"]==list(state[name].shape) and record["dtype"]==str(state[name].dtype) and len(record["sha256"])==64,"PRESERVED_UNSELECTED_COEFFICIENT_RECORD")
            totals["alpha_candidates_checked"]+=1
        entry=row["DEV"];same(entry["selected_coefficients"],row["selected_coefficients"]);x,anchor,_,_,raw=features(data,route,dev,entry);saved=load(entry["artifact"]);base.exact(saved["origins"],raw["origins"],"selected_DEV_origins");base.exact(saved["prediction_FP32"],deployed_prediction(x,anchor,state),"selected_fused_CPU32_DEV");totals["score_fields_checked"]+=score_check(entry,saved["prediction_FP32"],raw);totals["selected_DEV_prediction_evaluations_checked"]+=1;totals["selected_coefficient_FP32_arithmetic_predictions_checked"]+=1
        require(Path(entry["artifact"]["path"]).stat().st_mtime>=registry["sealed_unix"],"SAVED_DEV_ARTIFACT_AFTER_ALL_TWELVE_LOCKS");totals["selected_upper_grid_boundary_conditions"]+=int(chosen["alpha"]==ALPHAS[-1])
        report["datasets"].setdefault(data,{})[route]={"selected_alpha":chosen["alpha"],"selected_TUNE_MSE":chosen["TUNE_MSE"],"selected_upper_grid_boundary":chosen["alpha"]==ALPHAS[-1],"fit_diagnostic":fit_diagnostic,"selected_coefficients":row["selected_coefficients"],"FIT":row["FIT"],"TUNE":row["TUNE"],"DEV":entry,"candidates":[{"alpha":p["alpha"],"TUNE_MSE":p["TUNE_MSE"],"coefficient_value_sha256":p["coefficient_value_sha256"],"prediction_artifact":p["prediction_artifact"]} for p in row["candidates"]]}
        del state,fitx,tunex,y,mask,x,saved
    require(len(seen)==12 and [totals[key] for key in ("selected_ridge_fits_checked","alpha_candidates_checked","FIT_TUNE_prediction_evaluations_checked","selected_DEV_prediction_evaluations_checked","selected_coefficient_FP32_arithmetic_predictions_checked","selection_locks_checked")]==[12,84,168,12,36,12],"EXACT_TWELVE_SELECTED_FITS_AND_180_STORED_PREDICTIONS")
    for item in guarded+loaded+[result["selection_registry"]]:check();require(prior.file_sha(item["path"])==item["sha256"],"PRESERVED_SOURCE_INPUT_RESULT_BYTES")
    totals["protected_source_input_and_result_hashes_checked"]=len({(str(Path(item["path"]).resolve()),item["sha256"]) for item in guarded+loaded+[result["selection_registry"]]})
    return report


def execute(path,output):
    from continuation_runtime_v1 import Meter,write_new
    output=Path(output).resolve();require(not output.exists(),"FRESH_DIRECT_RIDGE_AUDIT_OUTPUT");meter=Meter("audit_direct_ridge_"+output.stem,gpu=False);source=base.file_record(__file__);failure=None
    try:
        metadata(path,meter.check);initialize_numpy();sys.path.insert(0,str(ROOT/"research/smallk_failure_followup_20261014"))
        from prepare_inputs import install_protected_test_guard
        from threadpoolctl import threadpool_limits
        install_protected_test_guard();meter.protected_check()
        with threadpool_limits(limits=2):report=audit_result(path,meter)
        meter.protected_check();require(prior.file_sha(__file__)==source["sha256"],"AUDITOR_SOURCE_UNCHANGED");json.dumps(report,allow_nan=False)
    except Exception as error:failure={"type":type(error).__name__,"message":str(error)};report={"status":"FAILED_PRESERVED","failure":failure}
    finally:report["audit_source"]=source;report["resource_terminal"]=meter.close(report["status"],failure);write_new(output,report)
    print(json.dumps({"status":report["status"],"totals":report.get("totals"),"failure":failure}),flush=True);return 0 if report["status"]=="PASS" else 2


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--result",required=True);parser.add_argument("--output",required=True);args=parser.parse_args();raise SystemExit(execute(args.result,args.output))
