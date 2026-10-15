"""Independent saved readout audit, conditional on already-fitted full-FIT upstream."""
from pathlib import Path
import argparse
import json
import math
import sys

import audit_horizon_decoder_diagnostic_v1 as old

base,prior,ridge,HERE,ROOT,require=old.base,old.prior,old.ridge,old.HERE,old.ROOT,old.require
RESULT_SHA="4ad5ee73f789a3b21eab8d0b732b04bde2622e5380e915a07f226869a2a6becf"
CONTROLLER_SHA="777c147fa77f63b180470918b902cfde1d07d205d87caa8739cc35e3ec892734"
MODULE_SHA="406252e940ae676e7a78f69e067f5b3fa70456bdb1ea27d57169b8964ceabe0e"
HORIZON_SHA="5f158382640dac74ec63cc3d1005b385c898821d3c032d51b543f2c9b27d7dac"
HORIZON_AUDIT_SHA="0f65053536403a3b7b8f53fca3b9f8274fb1a406eaf6dfdcb73bb0467fffe698"
CROSS_SHA="99bffd5831143ca35947099fef9567f89dd209d83451661decee0682b2b8e13c"
CROSS_AUDIT_SHA="90c68a9057909fb5b6c23284e5a45523cdf126da178a5ba0d9d9e96a50dede30"
ROUTES=("PCA_K8","RAW_C")
np=sys.modules.get("numpy")


def initialize_numpy():
    global np
    old.initialize_numpy();np=sys.modules["numpy"]


def partition_reference(origins,*,a_count=128,b_count=64,horizon=48):
    require(origins.dtype==np.int64 and origins.ndim==1 and len(origins)>a_count+b_count and a_count>0 and b_count>0 and horizon>0,"REGISTERED_ORIGIN_FIRST_FUTURE_PARTITION")
    order=np.argsort(origins,kind="stable");ordered=origins[order];require(np.all(np.diff(ordered)>0),"UNIQUE_SORTED_ACTUAL_ORIGINS")
    ac,bc,ci=order[:a_count],order[a_count:a_count+b_count],order[a_count+b_count:]
    ai=ac[origins[ac]+horizon-1<origins[bc[0]]];bi=bc[origins[bc]+horizon-1<origins[ci[0]]]
    require(len(ai)>0 and len(bi)>0 and len(ci)>0,"NONEMPTY_PURGED_A_B_C")
    return {"ordered_source_indices":order,"A_indices":ai,"B_indices":bi,"C_indices":ci,"A_candidate_indices":ac,"B_candidate_indices":bc,"A_purged_indices":ac[~np.isin(ac,ai)],"B_purged_indices":bc[~np.isin(bc,bi)],"source_origins":origins.copy()}


def timestamp_reference(origins,observed):
    require(origins.dtype==np.int64 and observed.dtype==np.bool_ and observed.ndim==3 and len(origins)==len(observed) and len(origins)>0,"MATCHED_COHORT_TIMESTAMPS_AND_MASK")
    response=origins[:,None]+np.arange(observed.shape[1],dtype=np.int64)[None];unique,multiplicity=np.unique(response,return_counts=True)
    return {"origin_count":len(origins),"first_origin":int(origins.min()),"last_origin":int(origins.max()),"first_response_timestamp":int(unique[0]),"last_response_timestamp":int(unique[-1]),"response_timestamp_occurrences":int(response.size),"unique_response_timestamps":len(unique),"maximum_response_multiplicity":int(multiplicity.max()),"observed_target_counts_per_channel":observed.sum((0,1)).tolist()}


def endpoint_reference(baseline,prediction,target,observed):
    require(baseline.dtype==prediction.dtype==target.dtype==np.float32 and observed.dtype==np.bool_ and baseline.shape==prediction.shape==target.shape==observed.shape and baseline.ndim==3 and np.isfinite(baseline).all() and np.isfinite(prediction).all() and np.isfinite(target[observed]).all(),"FINITE_ACTUAL_FP32_ENDPOINTS_WITH_TRUE_OBSERVATIONS")
    channels=[]
    for c in range(target.shape[-1]):
        mask=observed[:,:,c];count=int(mask.sum());value={"channel":c,"observed_count":count}
        if count:
            b=baseline[:,:,c][mask].astype(np.float64);p=prediction[:,:,c][mask].astype(np.float64);y=target[:,:,c][mask].astype(np.float64);delta=p-b;e=b-y
            value.update(baseline_MSE=float(np.dot(e,e)/count),selected_MSE=float(np.dot(p-y,p-y)/count),interaction_term=float(2*np.dot(e,delta)/count),correction_energy=float(np.dot(delta,delta)/count))
        channels.append(value)
    active=[v for v in channels if v["observed_count"]];result={"observed_counts":[v["observed_count"] for v in channels],"active_channels":len(active),"per_channel":channels}
    if active:
        result.update({key:float(np.mean([v[key] for v in active])) for key in ("baseline_MSE","selected_MSE","interaction_term","correction_energy")});result["delta_MSE"]=result["selected_MSE"]-result["baseline_MSE"];result["closure_residual"]=result["delta_MSE"]-result["interaction_term"]-result["correction_energy"]
    return result


def endpoint_check(recorded,baseline,prediction,raw):
    require(recorded["actual_return_FP32_endpoints_interpreted_as_FP64"] is True and recorded["horizon_macros_averaged_to_reconstruct_full"] is recorded["unobserved_targets_used"] is False and recorded["prediction_baseline_byte_equal"]==(baseline.tobytes()==prediction.tobytes()),"OBSERVED_ENDPOINT_DIAGNOSTIC_SCOPE")
    groups=[("full",None,0,48),*(("per_horizon",h,h,h+1) for h in range(48)),*(("registered_intervals",name,start,stop) for name,start,stop in (("1_16",0,16),("17_32",16,32),("33_48",32,48)))];checks=0
    for kind,key,start,stop in groups:
        row=recorded[kind] if key is None else recorded[kind][key];wanted=endpoint_reference(baseline[:,start:stop],prediction[:,start:stop],raw["target"][:,start:stop],raw["observed"][:,start:stop]);require(row["observed_counts"]==wanted["observed_counts"] and row["active_channels"]==wanted["active_channels"],"ERROR_TERMS_OBSERVED_DENOMINATORS")
        for v,w in [(row,wanted),*zip(row["per_channel"],wanted["per_channel"])]:
            if w.get("observed_count",w.get("active_channels",0)):
                for field in ("baseline_MSE","selected_MSE","interaction_term","correction_energy"):prior.close(v[field],w[field],"endpoint_"+field,atol=1e-12,rtol=1e-12)
                delta=w["selected_MSE"]-w["baseline_MSE"];closure=delta-w["interaction_term"]-w["correction_energy"];prior.close(v["delta_MSE"],delta,"endpoint_delta",atol=1e-12,rtol=1e-12);prior.close(v["direct_squared_error_change"],delta,"endpoint_direct_change",atol=1e-12,rtol=1e-12);tolerance=1e-12+1e-12*max(abs(v[k]) for k in ("baseline_MSE","selected_MSE","interaction_term","correction_energy","direct_squared_error_change"));require(v["closure_pass"] is True and v["closure_tolerance"]==tolerance and abs(closure)<=tolerance,"REGISTERED_ENDPOINT_CLOSURE_NO_TOLERANCE_RELAXATION")
            else:require(v["status"]=="NO_OBSERVED_TARGETS" and v["baseline_MSE"] is v["selected_MSE"] is None,"NO_MISSING_LABEL_AS_ZERO_SCORE")
        checks+=1
    return checks


def select_reference(candidates):
    menu=["ALPHA_"+format(a,"g") for a in old.ALPHAS]+["ZERO_DELTA"]
    require([v["id"] for v in candidates]==menu and all(v["status"]=="COMPLETE" and math.isfinite(v["B_MSE"]) for v in candidates),"ALL_ELEVEN_CANDIDATES_FROM_A_SELECTED_BY_B")
    return min(candidates,key=lambda v:(v["B_MSE"],menu.index(v["id"])))


def metadata_gate(path,check):
    require(all(isinstance(v,str) and len(v)==64 for v in (RESULT_SHA,CONTROLLER_SHA)),"ACTUAL_PURGED_RESULT_NOT_YET_REGISTERED")
    binding=base.file_record(path);require(binding["sha256"]==RESULT_SHA,"EXACT_ACTUAL_PURGED_RESULT");value=prior.verified_json(binding,check);require(value["status"]=="COMPLETE" and not value["failed_paths"],"COMPLETE_BOTH_ROUTES_NO_FAILED_PATHS")
    return value,binding


def timestamp_check(recorded,origins,observed):
    value=timestamp_reference(origins,observed);mapping={"origin_count":"origin_count","origins_min":"first_origin","origins_max":"last_origin","first_target_timestamp":"first_response_timestamp","last_target_timestamp":"last_response_timestamp","horizon_slots":"response_timestamp_occurrences","unique_target_timestamps":"unique_response_timestamps","max_target_multiplicity":"maximum_response_multiplicity","observed_counts_per_channel":"observed_target_counts_per_channel"}
    require(all(recorded[key]==value[wanted] for key,wanted in mapping.items()) and recorded["independent_repeat_count"] is None,"COHORT_ORIGIN_RESPONSE_AND_OBSERVED_COUNTS")
    time=origins[:,None]+np.arange(48,dtype=np.int64)[None];_,counts=np.unique(time,return_counts=True);multiplicity,frequency=np.unique(counts,return_counts=True)
    histogram=[{"multiplicity":int(m),"unique_timestamp_count":int(n)} for m,n in zip(multiplicity,frequency)];require(recorded["target_multiplicity_histogram"]==histogram and recorded["unique_observed_target_timestamps_per_channel"]==[int(np.unique(time[observed[:,:,c]]).size) for c in range(observed.shape[2])],"REPEATED_RESPONSE_TIMESTAMPS_NOT_INDEPENDENT_SAMPLES")


def audit_result(path,meter):
    result,binding=metadata_gate(path,meter.check);check=meter.check
    require(result["controller_source"]["sha256"]==CONTROLLER_SHA and result["module_source"]["sha256"]==MODULE_SHA and prior.file_sha(old.__file__)=="accb6fb775d9b12ca6e39c0be7e062545b98d429089ee32e0ac46ed65edae613","EXACT_PRODUCER_AND_FROZEN_INDEPENDENT_MATH_HELPER")
    pins={"horizon_diagnostic":HORIZON_SHA,"horizon_audit":HORIZON_AUDIT_SHA,"cross_moment_diagnostic":CROSS_SHA,"cross_moment_audit":CROSS_AUDIT_SHA};parents={}
    for key,digest in pins.items():require(result[key]["sha256"]==digest,"EXACT_PARENT:"+key);parents[key]=prior.verified_json(result[key],check);require(parents[key]["status"]==("PASS" if key.endswith("audit") else "COMPLETE"),"ACCEPTED_PARENT_STATUS")
    ridge.same(parents["horizon_audit"]["diagnostic"],result["horizon_diagnostic"]);ridge.same(parents["cross_moment_audit"]["diagnostic"],result["cross_moment_diagnostic"])
    require(all(result[k] is True for k in ("conditional_readout_diagnostic","fixed_full_FIT_upstream","upstream_used_A_outside_data","C_targets_loaded_with_full_FIT_archive","C_baseline_and_features_available_from_fixed_full_FIT_context_before_locks","C_mask_and_timestamp_facts_computed_before_locks","protected_and_loaded_inputs_unchanged_after_run","both_locks_before_C")) and all(result[k] is False for k in ("independent_holdout","end_to_end_temporal_validation","strict_K_only","new_backbone_training","whole_deployment_cost_measured","protected_TEST_accessed","TUNE_DEV_arrays_deserialized","TUNE_DEV_scored","same_capacity_PCA_and_RAW_claimed","NEW_SELECTED_C_predictions_or_C_scores_before_locks","independent_OS_C_access_events_verified")) and result["new_coefficient_cost_status"]=="NOT_MEASURED","CONDITIONAL_FULL_FIT_UPSTREAM_NO_NEW_HOLDOUT_OR_COST")
    require(all(result[k]==0 for k in ("new_F_forward_calls","new_model_objects_loaded","new_optimizer_updates","new_E_D_training_updates","new_DEV_winners_selected")),"NO_NEW_F_MODEL_OR_OPTIMIZER")
    expected={"fits_attempted":2,"fits_completed":2,"alpha_candidates_completed":20,"ZERO_candidates_completed":2,"candidate_A_B_evaluations_completed":44,"selected_C_evaluations_completed":2,"resolved_selection_locks":2};require(result["totals"]==expected and result["planned_counts"]=={k:v for k,v in expected.items() if k!="fits_attempted"} and set(result["routes"])==set(ROUTES),"EXACT_TWO_FITS_22_CANDIDATES_44_AB_TWO_C")
    guarded=[binding,*[result[k] for k in pins],result["plan"],result["controller_source"],result["module_source"],result["error_terms_module_source"],*result["source_records"],*result["input_records"],*result["protected_prior_records"],*ridge.receipt_check(result["resource_terminal"],check)];loaded=[]
    def load(item):loaded.append(item);return prior.verified_arrays(item,check)
    for item in guarded:
        require(prior.file_sha(item["path"])==item["sha256"],"SOURCE_INPUT_AND_PARENT_BYTES")
        if "snapshot_path" in item:require(prior.file_sha(item["snapshot_path"])==item["sha256"],"EXACT_EXECUTED_SOURCE_SNAPSHOT")
    registry=prior.verified_json(result["selection_registry"],check);require(registry["status"]=="BOTH_ROUTES_RESOLVED_BEFORE_C" and registry["C_outputs_or_scores_created_before_seal"] is False and len(registry["conditions"])==2,"TWO_B_LOCKS_SAVED_BEFORE_NEW_SELECTED_C")
    horizon,cross=parents["horizon_diagnostic"],parents["cross_moment_diagnostic"];source=horizon["datasets"]["robin"]["spaces"]["PCA_K8"]["contexts"]["fit"]["input_source"];raw=load(source);prior.check_raw_split(raw,channels=17,windows=256);parts=partition_reference(raw["origins"]);saved=load(result["partition_artifact"]);require(set(saved)==set(parts),"EXACT_PARTITION_ARRAY_MEMBERS")
    for key,value in parts.items():base.exact(saved[key],value,"fixed_partition_"+key)
    purge=result["partition"]["purge"];require(purge["A_candidate_count"]==128 and purge["B_candidate_count"]==purge["C_count"]==64 and purge["A_removed_count"]==len(parts["A_purged_indices"]) and purge["B_removed_count"]==len(parts["B_purged_indices"]) and purge["first_B_origin"]==int(raw["origins"][parts["B_candidate_indices"][0]]) and purge["first_C_origin"]==int(raw["origins"][parts["C_indices"][0]]) and purge["A_predicate"]=="origin+47 < first_B_origin" and purge["B_predicate"]=="origin+47 < first_C_origin" and purge["origin_means_first_future_timestamp"] is purge["history_overlap_permitted"] is True and purge["complete_L_plus_H_separation_claimed"] is False and purge["original_input_order_sorted"]==bool(np.array_equal(parts["ordered_source_indices"],np.arange(256))),"ACTUAL_FIRST_FUTURE_TIMESTAMP_STRICT_PURGE")
    for label in ("A","B","C"):
        indices=parts[label+"_indices"];timestamp_check(result["partition"]["cohorts"][label],raw["origins"][indices],raw["observed"][indices])
    counts={k:0 for k in ("readout_fit_paths_checked","alpha_candidates_checked","ZERO_candidates_checked","A_B_prediction_evaluations_checked","selected_C_prediction_evaluations_checked","coefficient_artifacts_checked","FP32_arithmetic_predictions_checked","selected_fits_independently_solved","selected_ZERO_states_checked","observed_mask_groups_checked","selection_locks_checked","score_fields_checked","temporal_score_fields_checked","endpoint_diagnostic_cells_checked")};report={"status":"PASS","diagnostic":binding,**{k:result[k] for k in (*pins,"controller_source","module_source","error_terms_module_source","plan","selection_registry")},"partition":result["partition"],"routes":{},"totals":counts,"partial_paths":[],"conditional_readout_diagnostic":True,"fixed_full_FIT_upstream":True,"C_targets_loaded_with_full_FIT_archive":True,"fixed_C_baseline_and_features_already_available_before_locks":True,"new_selected_C_outputs_or_scores_before_locks":False,"independent_OS_C_access_events_verified":False,"TUNE_DEV_arrays_deserialized":False,"F_GPU_optimizer_or_cost_replayed":False,"independent_holdout":False,"end_to_end_temporal_validation":False,"new_coefficient_cost_measured":False,"protected_TEST_accessed":False}
    seen=set()
    for entry in registry["conditions"]:
        space=entry["space"];require(space in ROUTES and space not in seen and entry["status"]=="COMPLETE","UNIQUE_COMPLETE_TWO_ROUTE_REGISTRY");seen.add(space);row=result["routes"][space];metadata=row["source_context"];ridge.same(metadata["input_source"],source);ridge.same(entry["selection_lock"],row["selection_lock"]);lock=prior.verified_json(row["selection_lock"],check)
        require(lock["status"]==row["status"]=="COMPLETE" and lock["C_outputs_or_scores_created_before_lock"] is False and lock["C_targets_in_same_FIT_archive_already_loaded"] is True and lock["sealed_unix"]<=registry["sealed_unix"] and all(lock[k]==row[k] for k in ("selected_candidate","selected_kind","selected_alpha","selected_B_MSE","selected_coefficient_value_sha256")),"SAVED_B_ONLY_SELECTION_LOCK_SCOPE")
        ridge.same(lock["selected_coefficients"],row["selected_coefficients"]);state=cache=None
        if space=="RAW_C":require(metadata["cached_latent_source"] is metadata["frozen_state_source"] is None,"RAW_FIXED_IDENTITY_STREAM");z,future,baseline=raw["history"],raw["raw_forecast"],raw["raw_forecast"];E=np.eye(17,dtype=np.float32);center=np.zeros(17,np.float32);D=np.concatenate([E,*[np.zeros_like(E) for _ in old.LAGS[1:]]])
        else:
            owner=cross["datasets"]["robin"]["conditions"][space];ridge.same(metadata["cached_latent_source"],owner["splits"]["fit"]["artifact"]);ridge.same(metadata["frozen_state_source"],owner["state_artifact"]);ridge.same(owner["splits"]["fit"]["input_source"],source);cache=load(metadata["cached_latent_source"]);state=load(metadata["frozen_state_source"]);base.exact(cache["origins"],raw["origins"],"same_original_FIT_cache_origins");z,future,baseline=cache["latent_history"],cache["latent_future"],cache["prediction"];E,center,D=state["E"],state["center"],state["D"]
        require(row["space"]==space and row["mode"]=="F" and row["arm"]=="STATIC_DELTA" and row["K"]==z.shape[2] and row["additional_slope_parameters"]==7*z.shape[2]*17,"FIXED_F_ONLY_STATIC_DIMENSIONS");phi=old.old_lag_features(z,future);stream={"features":phi,"baseline":baseline,"latent_history":z,"latent_future":future};require(metadata["array_bindings"]=={k:old.array_record(v) for k,v in stream.items()},"EXACT_FULL_FIT_FROZEN_STREAM_ARRAY_BINDINGS")
        checks=metadata["stream_checks"];old.check_parity(checks["original_encode"],z,(raw["history"]-center)@E);require(checks["same_old_decoder"]["required"]==(space!="RAW_C") and checks["same_old_decoder"]["parity_not_required"]==(space=="RAW_C"),"RAW_REASSOCIATION_IS_ONLY_AUXILIARY");old.check_parity(checks["same_old_decoder"],baseline,raw["raw_last"][:,None]+phi@D,required=space!="RAW_C")
        if cache is not None:old.check_parity(checks["original_lag_features"],phi,cache["lag_features"])
        ai=parts["A_indices"];scale,constant=old.actual_F_scale(phi[ai]);response=old.observed_residual(raw["target"][ai],baseline[ai],raw["observed"][ai]);coefficient=load(row["selected_coefficients"]);records,digest=ridge.coefficient_records(coefficient);require(digest==row["selected_coefficient_value_sha256"],"SELECTED_A_FITTED_COEFFICIENT_HASH");group_count,fitdiagnostic=old.fit_info_check(row,load(row["fit_artifact"]),phi[ai],response,raw["observed"][ai],scale,constant,coefficient);counts["observed_mask_groups_checked"]+=group_count;counts["selected_ZERO_states_checked" if row["selected_kind"]=="ZERO_DELTA" else "selected_fits_independently_solved"]+=1
        chosen=select_reference(row["candidates"]);require(chosen["id"]==row["selected_candidate"] and chosen["kind"]==row["selected_kind"] and chosen["alpha"]==row["selected_alpha"] and chosen["B_MSE"]==row["selected_B_MSE"] and chosen["coefficient_value_sha256"]==digest and chosen["coefficient_array_records"]==records,"OWN_B_ONLY_WINNER_AND_SELECTED_VALUES");selected_predictions=None
        for j,candidate in enumerate(row["candidates"]):
            zero=j==len(old.ALPHAS);require(candidate["kind"]==("ZERO_DELTA" if zero else "ALPHA") and candidate["alpha"]==(None if zero else old.ALPHAS[j]) and candidate["B_MSE"]==candidate["B"]["scores_all_observed"]["normalized"]["mse"],"EXACT_PREDECLARED_ALPHA_AND_B_SCORE")
            weights=load(candidate["coefficient_artifact"]);require(set(weights)=={"coefficient_FP64","coefficient_FP32"} and weights["coefficient_FP64"].shape==(phi.shape[2],17) and weights["coefficient_FP64"].dtype==np.float64 and weights["coefficient_FP32"].dtype==np.float32 and all(np.isfinite(v).all() for v in weights.values()),"FINITE_SLOPE_ONLY_CANDIDATE_STATE");base.exact(weights["coefficient_FP32"],weights["coefficient_FP64"].astype(np.float32),"candidate_fused_cast");require(ridge.coefficient_records(weights)==(candidate["coefficient_array_records"],candidate["coefficient_value_sha256"]),"ALL_SAVED_CANDIDATE_COEFFICIENT_HASHES")
            if zero:require(all(not np.any(v) for v in weights.values()),"ZERO_SLOPE_ARRAYS")
            saved=load(candidate["prediction_artifact"])
            for label in ("A","B"):
                indices=parts[label+"_indices"];cohort={**raw,**{k:raw[k][indices] for k in ("target","observed","origins")}};prediction=old.deployed_prediction(baseline[indices],phi[indices],weights["coefficient_FP32"],"STATIC_DELTA",zero=zero);base.exact(saved[label+"_origins"],cohort["origins"],"candidate_fixed_cohort_origins");base.exact(saved[label+"_prediction_FP32"],prediction,"all_candidate_deployed_FP32_equations");counts["score_fields_checked"]+=old.score_check(candidate[label],prediction,cohort);counts["A_B_prediction_evaluations_checked"]+=1;counts["FP32_arithmetic_predictions_checked"]+=1
                if zero:require(candidate["parity_checks"][label]=={"pass":True,"max_abs":0.,"ZERO_original_baseline_byte_exact":True},"ZERO_BYTE_COPY_METADATA")
                else:old.check_parity(candidate["parity_checks"][label],prediction,baseline[indices]+phi[indices]@weights["coefficient_FP32"])
            if candidate is chosen:selected_predictions=saved;ridge.same(candidate["coefficient_artifact"],row["selected_coefficients"])
            counts["ZERO_candidates_checked" if zero else "alpha_candidates_checked"]+=1;counts["coefficient_artifacts_checked"]+=1
        ridge.same(row["selected_A_B_prediction_artifact"],chosen["prediction_artifact"]);require(row["selected_A_B_reused_without_new_inference"] is True,"SELECTED_A_B_PREDICTION_REUSE")
        for label in ("A","B","C"):
            indices=parts[label+"_indices"];cohort={**raw,**{k:raw[k][indices] for k in ("target","observed","origins")}};actual=row[label]
            if label=="C":saved=load(actual["artifact"]);base.exact(saved["origins"],cohort["origins"],"selected_C_origins");prediction=old.deployed_prediction(baseline[indices],phi[indices],coefficient["coefficient_FP32"],"STATIC_DELTA",zero=row["selected_kind"]=="ZERO_DELTA");base.exact(saved["prediction_FP32"],prediction,"selected_C_frozen_A_B_coefficient");counts["selected_C_prediction_evaluations_checked"]+=1;counts["FP32_arithmetic_predictions_checked"]+=1
            else:prediction=selected_predictions[label+"_prediction_FP32"]
            counts["score_fields_checked"]+=old.score_check(actual,prediction,cohort)+old.score_check(row["baseline_scores"][label],baseline[indices],cohort);counts["temporal_score_fields_checked"]+=old.temporal_check(actual["temporal_diagnostics"],prediction,cohort);counts["endpoint_diagnostic_cells_checked"]+=endpoint_check(actual["error_terms"],baseline[indices],prediction,cohort)
        report["routes"][space]={"selected_candidate":row["selected_candidate"],"selected_kind":row["selected_kind"],"selected_alpha":row["selected_alpha"],"selected_B_MSE":row["selected_B_MSE"],"selected_coefficients":row["selected_coefficients"],"selection_lock":row["selection_lock"],"independent_selected_fit":fitdiagnostic,"A":row["A"],"B":row["B"],"C":row["C"],"baseline_scores":row["baseline_scores"]};counts["readout_fit_paths_checked"]+=1;counts["selection_locks_checked"]+=1
    require(seen==set(ROUTES) and counts["readout_fit_paths_checked"]==2 and counts["alpha_candidates_checked"]==20 and counts["ZERO_candidates_checked"]==2 and counts["A_B_prediction_evaluations_checked"]==44 and counts["selected_C_prediction_evaluations_checked"]==2,"FULL_TWO_ROUTE_REGISTERED_COVERAGE")
    for label in ("B","C"):
        a,b=(result["routes"][s] for s in ROUTES);mse=lambda v:v["scores_all_observed"]["normalized"]["mse"];expected={"PCA_selected_minus_ZERO":mse(a[label])-mse(a["baseline_scores"][label]),"Raw_selected_minus_ZERO":mse(b[label])-mse(b["baseline_scores"][label]),"PCA_selected_minus_Raw_selected":mse(a[label])-mse(b[label])};require(result["comparisons"][label]==expected,"UNCHANGED_REGISTERED_B_C_COMPARISONS")
    report["comparisons"]=result["comparisons"]
    for item in [*guarded,*loaded,result["selection_registry"],*[result["routes"][s]["selection_lock"] for s in ROUTES]]:require(prior.file_sha(item["path"])==item["sha256"],"FINAL_SOURCE_INPUT_STATE_PREDICTION_PRESERVATION")
    counts["protected_source_input_and_result_hashes_checked"]=len({(str(Path(v["path"]).resolve()),v["sha256"]) for v in [*guarded,*loaded]});return report


def execute(path,output):
    from continuation_runtime_v1 import Meter,write_new
    output=Path(output).resolve();require(not output.exists(),"FRESH_PURGED_AUDIT_OUTPUT");meter=Meter("audit_purged_readout_"+output.stem,gpu=False);failure=None;source=base.file_record(__file__)
    try:
        metadata_gate(path,meter.check);meter.protected_check();initialize_numpy();sys.path.insert(0,str(ROOT/"research/smallk_failure_followup_20261014"))
        from prepare_inputs import install_protected_test_guard
        from threadpoolctl import threadpool_limits
        install_protected_test_guard()
        with threadpool_limits(limits=2):report=audit_result(path,meter)
        meter.protected_check();require(prior.file_sha(__file__)==source["sha256"],"AUDITOR_SOURCE_IMMUTABLE");json.dumps(report,allow_nan=False)
    except Exception as error:failure={"type":type(error).__name__,"message":str(error)};report={"status":"FAILED_PRESERVED","failure":failure}
    finally:report["audit_source"]=source;report["resource_terminal"]=meter.close(report["status"],failure);write_new(output,report)
    print(json.dumps({"status":report["status"],"totals":report.get("totals"),"failure":failure}),flush=True);return 0 if report["status"]=="PASS" else 2


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--result",required=True);parser.add_argument("--output",required=True);args=parser.parse_args();raise SystemExit(execute(args.result,args.output))
