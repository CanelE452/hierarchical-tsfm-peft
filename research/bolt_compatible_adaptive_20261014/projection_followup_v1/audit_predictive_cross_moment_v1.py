"""Independent saved FIT cross-moment / reconstruction-D / point-forecast audit; no F replay."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import audit_linear_e_adaptation_v1 as base
from audit_fixed_l7_latent_calibration_v1 import receipt_check

HERE,ROOT,prior,require=base.HERE,base.ROOT,base.prior,base.require
DATASETS={"robin":(17,4,"val",30),"beijing_aotizhongxin":(11,2,"eval",146)}
KINDS=("PCA","CROSS_MOMENT");LAGS=(0,1,2,4,8,16,32)
RESULT_SHA="99bffd5831143ca35947099fef9567f89dd209d83451661decee0682b2b8e13c"
CONTROLLER_SHA="e5b971277673118454d30f13819a7013c9c0b0ef1c6e682c18d5fab971ef1f50"
MODULE_SHA="c1cec75137a8df2f1d5250829514791a3cc1d3747508e3bd62bf6ab0cf02bffe"
PARENT_SHA={"latent_calibration_diagnostic":"88ddbbf7287ad89b7bf81e170b3e05ddfc89bead700e2fefa50bb0dd4e85f576","latent_calibration_audit":"b31b742d3947a205dbaf01cd84239b3cde95e1ea9baf3331a3c08c48a0aa5ba0","latent_calibration_cost":"d8db7445a7a5a750361149cde9a3ad653d4926e871d72c387fe8cbd285aca17b","latent_calibration_cost_audit":"4a54f5b738beef15bc02a17687624afb8c4493f5df7463aeb7b8e4e35b619c40","parent_report":"e8f1036dcc27ee2a460b6d87a0f9d1d03368067dab5cb3abd32ec2ec8e368c4f","parent_report_manifest":"e5a17e7f5131ed611b87beca2d49ec4a05e093c2eaf1d0f42d0343b5f0350198","parent_report_audit":"2e98180c7e987e2be807eb00914d9da8876690490daa2b3b56c2f9ce39603dfc"}
np=sys.modules.get("numpy")


def initialize_numpy():
    global np
    base.initialize_numpy();np=sys.modules["numpy"]


def same(a,b):require(Path(a["path"]).resolve()==Path(b["path"]).resolve() and a["sha256"]==b["sha256"],"EXACT_PARENT_PATH_SHA")


def array_record(a):return {"shape":list(a.shape),"dtype":str(a.dtype),"sha256":hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()}


def canon(a):
    a=a.copy()
    for j in range(a.shape[1]):
        if a[np.argmax(np.abs(a[:,j])),j]<0:a[:,j]*=-1
    return a


def independent_direction(raw,kind,k):
    x=raw["history"].reshape(-1,raw["history"].shape[-1]).astype(np.float64);center=x.mean(0);a=x-center;c=x.shape[1]
    covariance=np.empty((c,c),np.float64)
    for i in range(c):
        for j in range(c):covariance[i,j]=np.sum(a[:,i]*a[:,j])/len(a)
    values,vectors=np.linalg.eigh(covariance);retained=np.flatnonzero(values>values[-1]*1e-8)[::-1];require(len(retained)>=k,"INDEPENDENT_HISTORY_RANK_BELOW_K")
    w=canon(vectors[:,retained])/np.sqrt(values[retained])[None];complete=raw["observed"].all(2);counts=complete.sum(0);require((counts>=2).all(),"INDEPENDENT_TWO_COMPLETE_RESPONSES_PER_HORIZON")
    current=raw["history"][:,-1].astype(np.float64);cross=np.empty((48,c,c),np.float64)
    for h in range(48):
        left=current[complete[:,h]];right=raw["target"][complete[:,h],h].astype(np.float64);left=left-left.mean(0);right=right-right.mean(0)
        for i in range(c):
            for j in range(c):cross[h,i,j]=np.sum(left[:,i]*right[:,j])/len(left)
    moment=np.concatenate([w.T@cross[h] for h in range(48)],axis=1)/np.sqrt(48.);u,s,_=np.linalg.svd(moment,full_matrices=False);rank=int((s>s[0]*1e-8).sum()) if s[0]>0 else 0
    require(kind in KINDS and (kind!="CROSS_MOMENT" or rank>=k),"INDEPENDENT_REGISTERED_KIND_AND_MOMENT_RANK")
    e=canon(w@u[:,:k]) if kind=="CROSS_MOMENT" else canon(w[:,:k]);latent_cov=e.T@covariance@e
    prior.close(latent_cov,np.eye(k),"independent_whitened_FIT_covariance",atol=1e-7,rtol=0.)
    return {"center":center,"E":e,"cross":cross,"whitening":w,"singular":s,"whitening_rank":len(retained),"moment_rank":rank,"counts":counts,"latent_covariance":latent_cov}


def independent_decoder(raw,z):
    features=np.concatenate([z[:,32-lag:512-lag] for lag in LAGS],axis=2).reshape(-1,7*z.shape[2]).astype(np.float64);y=raw["history"][:,32:].reshape(-1,raw["history"].shape[2]).astype(np.float64);mask=raw["history_observed"][:,32:].reshape(y.shape)
    d=np.zeros((features.shape[1],y.shape[1]),np.float64);bias=np.zeros(y.shape[1]);counts=mask.sum(0);ranks=[];singular=[];residual=[]
    for channel,count in enumerate(counts):
        if not count:ranks.append(0);singular.append([]);residual.append(None);continue
        a,b=features[mask[:,channel]],y[mask[:,channel],channel];am,bm=a.mean(0),b.mean();slope,_,rank,s=np.linalg.lstsq(a-am,b-bm,rcond=1e-4)
        d[:,channel]=slope;bias[channel]=bm-am@slope;ranks.append(int(rank));singular.append(s.tolist());residual.append(float(np.mean((a@slope+bias[channel]-b)**2)))
    return d,bias,{"counts":counts.tolist(),"rank":ranks,"singularvalues":singular,"residual":residual,"latent_binding":array_record(z)}


def valid_oracle_mask(observed):
    complete=observed.all(2);valid=np.ones_like(complete)
    for h in range(observed.shape[1]):
        for lag in LAGS:
            if h>=lag:valid[:,h] &= complete[:,h-lag]
    return valid


def check_timestamps(recorded,raw):
    past=raw["origins"][:,None]+np.arange(-512,0);future=raw["origins"][:,None]+np.arange(48);complete=raw["observed"].all(2)
    for label,values in (("history",past),("future",future),("complete_C_response",future[complete])):
        timestamps,multiplicity=np.unique(values,return_counts=True);expected={"occurrences":values.size,"unique_timestamps":timestamps.size,"maximum_multiplicity":int(multiplicity.max(initial=0)),"timestamp_min":int(timestamps.min()) if timestamps.size else None,"timestamp_max":int(timestamps.max()) if timestamps.size else None}
        require(recorded[label]==expected,"INDEPENDENT_TIMESTAMP_OCCURRENCES:"+label)
    require(recorded["origins_unique"]==np.unique(raw["origins"]).size and recorded["complete_HC_origins"]==int(complete.all(1).sum()) and recorded["overlapping_occurrences_not_independent_samples"] is recorded["FIT_boundary_contract_inherited_from_exact_accepted_source"] is True and recorded["timestamp_deduplication_or_reweighting"] is recorded["new_raw_data_split_boundary_reconstruction"] is False,"TIMESTAMP_SCOPE_AND_COMPLETE_HC")
    prior.close(recorded["source_current_filled_fraction"],(~raw["history_observed"][:,-1]).mean(),"current_fill_fraction",atol=1e-12,rtol=0.);prior.close(recorded["source_history_filled_fraction"],(~raw["history_observed"]).mean(),"history_fill_fraction",atol=1e-12,rtol=0.)


def check_cell(row,saved,independent,raw,state):
    n,c=raw["target"].shape[0],raw["target"].shape[2];k=state["E"].shape[1];future=saved["latent_future"]
    require(set(state)=={"E","D","center"} and state["E"].shape==(c,k) and state["D"].shape==(7*k,c) and state["center"].shape==(c,) and all(a.dtype==np.float32 and np.isfinite(a).all() for a in state.values()),"FROZEN_FP32_STATE_SHAPES")
    require(saved["latent_history"].shape==(n,512,k) and saved["prediction"].shape==(n,48,c) and saved["lag_features"].shape==(n,48,7*k) and all(saved[name].dtype==np.float32 and np.isfinite(saved[name]).all() for name in ("latent_history","prediction","lag_features")),"FINITE_FP32_SAVED_FORWARD_COMPONENTS")
    require(future.dtype==np.float32 and future.shape==(n,48,k) and np.isfinite(future).all(),"FINITE_STORED_ACTUAL_F_NOT_REPLAYED")
    z=(raw["history"]-state["center"])@state["E"];prior.close(saved["latent_history"],z,"independent_FP32_E")
    features=base.delta_features(saved["latent_history"],future,LAGS);base.exact(saved["lag_features"],features,"independent_lag_features")
    expected=raw["raw_last"][:,None]+features@state["D"];prior.close(saved["prediction"],expected,"independent_actual_C")
    complete=raw["observed"].all(2);safe=np.broadcast_to(raw["raw_last"][:,None],raw["target"].shape).copy();safe[complete]=raw["target"][complete]
    ey=(safe-state["center"])@state["E"];eyC=raw["raw_last"][:,None]+base.delta_features(saved["latent_history"],ey,LAGS)@state["D"]
    flat=np.repeat(saved["latent_history"][:,-1:],48,axis=1);flatC=raw["raw_last"][:,None]+base.delta_features(saved["latent_history"],flat,LAGS)@state["D"]
    for name,calculated in (("EY_future",ey),("EY_prediction",eyC),("FLAT_future",flat),("FLAT_prediction",flatC)):prior.close(saved[name],calculated,"independent_"+name)
    valid=valid_oracle_mask(raw["observed"]);base.exact(saved["oracle_valid_horizon_mask"],valid,"oracle_all_consumed_future_references")
    for name,calculated in (("independent_prediction",expected),("independent_EY_future",ey),("independent_EY_prediction",eyC),("independent_FLAT_prediction",flatC),("oracle_valid_horizon_mask",valid)):
        if calculated.dtype==np.bool_:base.exact(independent[name],calculated,"saved_"+name)
        else:prior.close(independent[name],calculated,"saved_"+name)
    base.exact(saved["origins"],raw["origins"],"source_origins");common=np.broadcast_to(valid[:,:,None],raw["observed"].shape);cmask=np.broadcast_to(complete[:,:,None],raw["observed"].shape)
    predictions={"ACTUAL_F":saved["prediction"],"RAW_BOLT":raw["raw_forecast"],"FLAT":saved["FLAT_prediction"],"CURRENT_C":np.broadcast_to(raw["raw_last"][:,None],raw["target"].shape),"E_Y":saved["EY_prediction"]};fields=0
    for method,prediction in predictions.items():
        entry=row["methods"][method];masks=(("scores_oracle_common_mask",common),) if method=="E_Y" else (("scores_all_observed",raw["observed"]),("scores_complete_mask",cmask),("scores_oracle_common_mask",common))
        for label,mask in masks:
            if mask.any():fields+=prior.compare_scores(entry[label],prior.scores(prediction,raw["target"],mask,raw["std"]),method+"_"+label)
            else:require(entry[label]=={"status":"NO_VALID_TARGETS"},"EMPTY_MASK_NO_FAKE_SCORES")
    require(row["status"]=="COMPLETE" and all(p["pass"] for p in row["checks"].values()) and row["oracle_valid_horizons"]==int(valid.sum()) and row["complete_C_horizons"]==int(complete.sum()) and row["total_horizons"]==complete.size,"ORACLE_COVERAGE_AND_RECORDED_CHECKS")
    require(row["oracle_valid_requires_all_consumed_future_references_complete_C"] is row["incomplete_future_horizon_vectors_filled_with_current_C"] is True and row["oracle_is_deployment_or_predictability_proof"] is row["flat_latent_is_current_C_persistence"] is row["all_FP32_or_all_real_lower_bound_claimed"] is False and row["output_space_minimum_status"]=="NOT_MEASURED_IN_THIS_STAGE" and row["methods"]["E_Y"]["non_oracle_mask_scores_not_reported"] is True,"EXACT_DIAGNOSTIC_SCOPE")
    return {"score_fields_checked":fields,"methods":row["methods"],"oracle_valid_horizons":int(valid.sum()),"complete_C_horizons":int(complete.sum()),"total_horizons":int(complete.size),"actual_F_independently_reinferred":False,"legacy_PCA_parity_is_not_a_success_gate":True}


def audit_result(path,meter):
    require(isinstance(RESULT_SHA,str) and len(RESULT_SHA)==64,"ACTUAL_CROSS_MOMENT_RESULT_NOT_REGISTERED")
    check=meter.check;binding=base.file_record(path);require(binding["sha256"]==RESULT_SHA,"EXACT_CROSS_MOMENT_RESULT");result=prior.verified_json(binding,check)
    require(result["status"] in ("COMPLETE","PARTIAL_COMPLETED") and result["controller_source"]["sha256"]==CONTROLLER_SHA and result["module_source"]["sha256"]==MODULE_SHA,"ACTUAL_STATUS_AND_EXECUTED_SOURCES")
    for key,digest in PARENT_SHA.items():require(result[key]["sha256"]==digest,"ACCEPTED_CALIBRATION_PARENT:"+key)
    for key in ("new_optimizer_updates","new_Raw_F_inferences"):require(result[key]==0,"NO_NEW_OPTIMIZER_OR_RAW_F")
    for key in ("new_backbone_training","native_predictors_or_raw_ridge_fitted","whole_deployment_cost_measured","protected_TEST_accessed","independent_holdout","strict_K_only","K_or_direction_selected_using_DEV"):require(result[key] is False,"SOURCE_SCOPE:"+key)
    require(result["following_native_ridge_and_cost_stage_still_required"] is True and result["output_space_minimum_status"]=="NOT_MEASURED","FOLLOWING_METHOD_STAGES_UNRESOLVED")
    guarded=[binding,*[result[k] for k in PARENT_SHA],*result["source_records"],*result["input_records"],*result["protected_prior_records"]]
    for item in guarded:
        check();require(prior.file_sha(item["path"])==item["sha256"],"ORIGINAL_FILE_SHA")
        if "snapshot_path" in item:require(prior.file_sha(item["snapshot_path"])==item["sha256"],"EXECUTED_SOURCE_SNAPSHOT")
    science=prior.verified_json(result["causal_lag_diagnostic"],check);parent=prior.verified_json(result["latent_calibration_diagnostic"],check);full=prior.verified_json(result["full_gradient_parent"],check)
    require(result["F_before"]==result["F_after"]==parent["F_parent_signatures_reused"]["before"]==parent["F_parent_signatures_reused"]["after"] and result["runtime_behavior_contract"]==full["runtime_behavior_contract"],"UNCHANGED_F_SIGNATURE_AND_RUNTIME")
    for key in ("model_binary","model_config"):same(result[key+"_before"],result[key+"_after"]);require(prior.file_sha(result[key+"_before"]["path"])==result[key+"_before"]["sha256"],"LOCAL_MODEL_FILE_SHA")
    registry=prior.verified_json(result["state_registry"],check);require(registry["status"]=="ALL_EIGHT_RESOLVED_BEFORE_DEV" and registry["DEV_opened_before_seal"] is registry["state_selection_performed"] is False and len(registry["conditions"])==result["resolved_state_locks"]==8 and result["all_8_locks_before_DEV"] is True,"EIGHT_STATE_REGISTRY_BEFORE_DEV")
    receipt_check(result["resource_terminal"],result["status"],check);require(result["resource_terminal"]["started_unix"]<=registry["sealed_unix"]<=result["resource_terminal"]["ended_unix"],"REGISTRY_INSIDE_SOURCE_RECEIPT")
    cache={}
    def load(item):
        key=item["path"],item["sha256"]
        if key not in cache:cache[key]=prior.verified_arrays(item,check)
        return cache[key]
    totals={k:0 for k in ("states_checked","direction_fits_checked","decoder_fits_checked","decoder_response_columns_checked","FIT_TUNE_cells_checked","DEV_cells_checked","independent_saved_F_arithmetic_cells_checked","oracle_dependency_masks_checked","score_fields_checked","state_locks_checked","legacy_PCA_nonrequired_diagnostics_preserved")};report={"status":"PASS" if result["status"]=="COMPLETE" else "PARTIAL_AUDITED_NO_FULL_PASS","diagnostic":binding,**{k:result[k] for k in PARENT_SHA},"causal_lag_diagnostic":result["causal_lag_diagnostic"],"state_registry":result["state_registry"],"totals":totals,"datasets":{},"partial_paths":[],"protected_TEST_accessed":False,"independent_holdout":False,"F_GPU_optimizer_native_ridge_or_cost_replayed":False,"independent_OS_DEV_access_events_replayed":False,"new_output_space_minimum_computed":False}
    keys=set();timestamp_checked=set()
    for item in registry["conditions"]:
        data,name=item["dataset"],item["condition"];c,basek,dev,devn=DATASETS[data];kind,k=name.rsplit("_K",1);k=int(k);require(kind in KINDS and k in (basek,2*basek) and (data,name) not in keys,"EXACT_EIGHT_REGISTERED_CONDITIONS");keys.add((data,name))
        row=result["datasets"][data]["conditions"][name];same(item["state_lock"],row["state_lock"]);lock=prior.verified_json(row["state_lock"],check);require(lock["status"]==row["status"]==item["status"] and lock["DEV_opened_before_lock"] is lock["state_selection_performed"] is False and lock["sealed_unix"]<=registry["sealed_unix"],"SAME_RESOLVED_STATE_LOCK");totals["state_locks_checked"]+=1
        if row["status"]!="COMPLETE":report["partial_paths"].append({"dataset":data,"condition":name,"status":row["status"]});continue
        same(lock["state_artifact"],row["state_artifact"]);same(lock["direction_artifact"],row["direction_artifact"]);same(lock["decoder_artifact"],row["decoder_artifact"]);require(lock["FIT_TUNE_statuses"]=={"fit":"COMPLETE","tune":"COMPLETE"} and row["kind"]==kind and row["K"]==k and row["C"]==c and row["E_FIT_future_supervised"]==(kind=="CROSS_MOMENT") and row["D_FIT_future_supervised"] is False,"LOCKED_FIXED_CONDITION_AND_SUPERVISION_HISTORY")
        fit_source=science["datasets"][data]["spaces"]["PCA"]["conditions"]["L7_RECON"]["splits"]["fit"]["input_source"];same(row["FIT_input_source"],fit_source);fit=load(fit_source);prior.check_raw_split(fit,channels=c,windows=256)
        direction=load(row["direction_artifact"]);state=load(row["state_artifact"]);decoder=load(row["decoder_artifact"]);calculated=independent_direction(fit,kind,k)
        require(decoder["decoder_FP64"].dtype==decoder["bias_FP64"].dtype==np.float64 and decoder["decoder_FP32"].dtype==decoder["bias_FP32"].dtype==np.float32,"DECODER_FIT64_DEPLOY32_STATE");base.exact(decoder["lags"],np.asarray(LAGS,dtype=np.int64),"registered_decoder_lags")
        for key,wanted in (("center_FP64",calculated["center"]),("encoder_FP64",calculated["E"]),("cross_moments_FP64",calculated["cross"]),("whitening_FP64",calculated["whitening"]),("singularvalues",calculated["singular"])):prior.close(direction[key],wanted,"independent_"+key,atol=1e-9,rtol=1e-7)
        for key,wanted in (("center",calculated["center"].astype(np.float32)),("E",calculated["E"].astype(np.float32))):prior.close(direction[key],wanted,"independent_deployed_"+key);base.exact(state[key],direction[key],"frozen_"+key)
        base.exact(state["D"],decoder["decoder_FP32"],"refitted_D_state");meta=row["direction_fit_metadata"];require(meta["whitening_rank"]==calculated["whitening_rank"] and meta["moment_rank"]==calculated["moment_rank"] and meta["horizon_complete_C_origins"]==calculated["counts"].tolist() and meta["eigenvalue_relative_cut"]==meta["moment_relative_cut"]==1e-8 and meta["TUNE_DEV_used"] is meta["unobserved_targets_used"] is meta["exact_masked_RRR"] is False,"DIRECTION_RANK_MASK_AND_NO_NEW_ESTIMAND")
        fit_z=load(row["splits"]["fit"]["artifact"])["latent_history"];d,b,dm=independent_decoder(fit,fit_z)
        prior.close(decoder["decoder_FP64"],d,"independent_RECON_D64",atol=1e-8,rtol=1e-7);prior.close(decoder["bias_FP64"],b,"independent_RECON_bias64",atol=1e-8,rtol=1e-7)
        for source,key in ((d,"decoder_FP32"),(b,"bias_FP32")):prior.close(decoder[key],source.astype(np.float32),"independent_"+key)
        dmeta=row["decoder_fit_metadata"];require(dmeta["rank"]==dm["rank"] and dmeta["true_observed_counts"]==dm["counts"] and dmeta["singular_relative_cutoff"]==1e-4 and dmeta["common_history_slice"]==[32,512] and dmeta["future_FIT_Y_used_as_supervision"] is False and dmeta["input_bindings"]["latent_history"]==dm["latent_binding"],"RECON_TRUE_OBSERVED_FIT_Z_RANK_AND_CUTOFF")
        for recorded,expected in zip(dmeta["singularvalues"],dm["singularvalues"]):prior.close(recorded,expected,"independent_RECON_singular",atol=1e-9,rtol=1e-7)
        for count,recorded,expected in zip(dm["counts"],dmeta["direct_residual_MSE"],dm["residual"]):
            if count:prior.close(recorded,expected,"independent_RECON_residual",atol=1e-10,rtol=1e-7)
            else:require(recorded is expected is None,"NO_OBSERVED_RESPONSE_NO_FAKE_RESIDUAL")
        verified={"kind":kind,"K":k,"C":c,"state_artifact":row["state_artifact"],"direction_fit_metadata":meta,"decoder_fit_metadata":dmeta,"splits":{}}
        for split,n in (("fit",256),("tune",64),(dev,devn)):
            entry=row["splits"][split];source=science["datasets"][data]["spaces"]["PCA"]["conditions"]["L7_RECON"]["splits"][split]["input_source"];same(entry["input_source"],source);same(entry["state_source"],row["state_artifact"]);raw=load(source);prior.check_raw_split(raw,channels=c,windows=n)
            if (data,split) not in timestamp_checked:check_timestamps(result["datasets"][data]["timestamp_occurrences"][split],raw);timestamp_checked.add((data,split))
            v=check_cell(entry,load(entry["artifact"]),load(entry["independent_artifact"]),raw,state);verified["splits"][split]=v;totals["score_fields_checked"]+=v["score_fields_checked"];totals["DEV_cells_checked" if split==dev else "FIT_TUNE_cells_checked"]+=1;totals["independent_saved_F_arithmetic_cells_checked"]+=1;totals["oracle_dependency_masks_checked"]+=1
            if "legacy_PCA_diagnostic" in entry:verified["splits"][split]["legacy_PCA_nonrequired_diagnostic"]=entry["legacy_PCA_diagnostic"];totals["legacy_PCA_nonrequired_diagnostics_preserved"]+=1
        report["datasets"].setdefault(data,{"conditions":{}})["conditions"][name]=verified;totals["states_checked"]+=1;totals["direction_fits_checked"]+=1;totals["decoder_fits_checked"]+=1;totals["decoder_response_columns_checked"]+=c
    if result["status"]=="COMPLETE":require(len(keys)==totals["states_checked"]==totals["direction_fits_checked"]==totals["decoder_fits_checked"]==result["direction_fits_completed"]==result["decoder_fits_completed"]==8 and totals["FIT_TUNE_cells_checked"]==result["FIT_TUNE_evaluations_completed"]==16 and totals["DEV_cells_checked"]==result["DEV_evaluations_completed"]==8 and result["new_F_forward_calls"]==result["new_F_forward_attempts"]==104 and result["new_F_model_objects_loaded"]==1 and not result["failed_paths"],"COMPLETE_EIGHT_STATE_24_ACTUAL_F_CONTEXTS")
    for item in guarded:check();require(prior.file_sha(item["path"])==item["sha256"],"PROTECTED_SOURCE_INPUT_RESULT_BYTES_UNCHANGED")
    for (path,digest) in cache:check();require(prior.file_sha(path)==digest,"ALL_LOADED_ARRAY_FILES_UNCHANGED")
    totals["protected_prior_file_hashes_checked"]=len(result["protected_prior_records"]);report["scope"]="Independent FIT center/covariance/eigenspace/whitening/horizon-complete current-future moments/direction SVD and actual saved FIT Torch32 z→observed centered RECON LS; stored future E/lag-D/score/oracle masks/source/receipt locks. No F/GPU/backward/optimizer/timer replay, independent OS access-event proof, output-space minimum, new native/ridge/cost or independent generalization."
    totals["unique_dataset_split_timestamp_contexts_checked"]=len(timestamp_checked)
    return report


def execute(path,output):
    from continuation_runtime_v1 import Meter,write_new
    output=Path(output).resolve();require(not output.exists(),"FRESH_CROSS_MOMENT_AUDIT_REQUIRED");meter=Meter("audit_predictive_cross_moment_"+output.stem,gpu=False);failure=None;source=base.file_record(__file__)
    try:
        meter.protected_check();initialize_numpy();sys.path.insert(0,str(ROOT/"research/smallk_failure_followup_20261014"))
        from prepare_inputs import install_protected_test_guard
        from threadpoolctl import threadpool_limits
        install_protected_test_guard()
        with threadpool_limits(limits=2):report=audit_result(path,meter)
        require(prior.file_sha(__file__)==source["sha256"],"AUDITOR_SOURCE_UNCHANGED");meter.protected_check();json.dumps(report,allow_nan=False)
    except Exception as error:failure={"type":type(error).__name__,"message":str(error)};report={"status":"FAILED_PRESERVED","failure":failure}
    finally:report["audit_source"]=source;report["resource_terminal"]=meter.close(report["status"],failure);write_new(output,report)
    print(json.dumps({"status":report["status"],"totals":report.get("totals"),"failure":failure}),flush=True);return 0 if report["status"]=="PASS" else 2


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--result",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args();raise SystemExit(execute(args.result,args.output))
