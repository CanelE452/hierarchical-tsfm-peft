"""Independent saved full-quantile point summaries; no F, GPU or timing replay."""
from pathlib import Path
import argparse
import json

import audit_orthogonal_basis_controls_v1 as common

HERE,ROOT,base,prior,require=common.HERE,common.ROOT,common.base,common.prior,common.require
same,bound,check_parity=common.same,common.bound,common.check_parity
STATES=("RECON_CP0","E_ONLY_s92601","E_ONLY_s92602")
ROUTES=("RAW_C",)+STATES
SUMMARIES=("MEDIAN","NINE_MEAN_PROXY","TAIL_CLAMPED_PROXY")
LAGS=(0,1,2,4,8,16,32)
RESULT_SHA="70d09544210602d3566d2d1f2405d06087499c496d6691dbade6dea64762d780"
CONTROLLER_SHA="57678bc90025a60b8d3ae50cc0bf5cac8cc5829cb108bc72168a4560770fb7b1"
MODULE_SHA="8af5ee93375a9ab7e2e937682cfbabad6e31e8a038bf6f6cca9c993ca3b87a28"
PARENT_READER_SHA="1c0676b75b2baaa1d937bade5e1893d06ba46c01f7c73d48953144e9d09d6daa"
REFS=tuple(common.PARENT_SHA)+("matched_l7_native_diagnostic","matched_l7_native_audit","E_only_diagnostic","E_only_audit","joint_recon_diagnostic","joint_recon_audit","causal_lag_diagnostic","causal_lag_audit","full_gradient_parent","calibration_report_metadata_reader","parent_metadata_reader","controller_source","module_source","plan","selection_registry")
np=None


def initialize_numpy():
    global np
    common.initialize_numpy();np=common.np


def summary(full,name):
    require(name in SUMMARIES and full.dtype==np.float32 and full.ndim==4 and full.shape[-1]==9 and np.isfinite(full).all(),"REGISTERED_N_H_K_NINE_FP32")
    if name=="MEDIAN":return full[...,4]
    weights=np.full(9,np.float32(1/9),dtype=np.float32) if name=="NINE_MEAN_PROXY" else np.array([.15,.1,.1,.1,.1,.1,.1,.1,.15],np.float32)
    return np.add.reduce(np.multiply(full,weights),axis=-1,dtype=np.float32)


def crossing(full):
    difference=full[...,1:].astype(np.float64)-full[...,:-1].astype(np.float64);bad=difference<0
    return {"shape":list(full.shape),"dtype":str(full.dtype),"quantile_levels":[i/10 for i in range(1,10)],
        "adjacent_pairs":int(bad.size),"crossing_pairs":int(bad.sum()),"crossing_fraction":float(bad.mean()),
        "maximum_adjacent_crossing":float(np.maximum(-difference,0).max(initial=0)),
        "crossing_pairs_by_level_pair":[int(bad[...,q].sum()) for q in range(8)],
        "crossing_pairs_by_horizon":[int(bad[:,h].sum()) for h in range(bad.shape[1])],
        "minimum":float(full.min()),"maximum":float(full.max()),"quantiles_sorted_after_inference":False,
        "finite_nine_quantiles_identify_conditional_mean":False,"decoded_C_quantiles_or_CRPS_claimed":False}


def check_diagnostic(recorded,full):
    expected=crossing(full);require(set(recorded)==set(expected),"ALL_CROSSING_AND_SCOPE_FIELDS")
    for key,value in expected.items():
        if type(value) is float:prior.close(recorded[key],value,"quantile_diagnostic_"+key,atol=1e-10,rtol=1e-10)
        else:require(recorded[key]==value,"EXACT_QUANTILE_DIAGNOSTIC:"+key)


def check_cell(row,full_arrays,candidates,independent,raw,cached,frozen,route):
    n,c=raw["history"].shape[0],raw["history"].shape[2];raw_route=route=="RAW_C";k=c if raw_route else 4
    require(set(full_arrays)=={"latent_history","full_quantiles_FP32","origins"} and set(candidates)==set(SUMMARIES) and set(independent)=={s+suffix for s in SUMMARIES for suffix in ("_NumPy_future","_NumPy_prediction")},"EXACT_FULL_CANDIDATE_AND_INDEPENDENT_MEMBERS")
    full,past=full_arrays["full_quantiles_FP32"],full_arrays["latent_history"]
    require(full.dtype==past.dtype==np.float32 and full.shape==(n,64,k,9) and past.shape==(n,512,k) and np.isfinite(full).all() and np.isfinite(past).all(),"FULL_N64K9_NOT_MEDIAN_ONLY_CACHE")
    base.exact(full_arrays["origins"],raw["origins"],"full_quantile_origins")
    expected_past=raw["history"] if raw_route else (raw["history"]-frozen["center"])@frozen["E"]
    median=candidates["MEDIAN"];baseline_future=raw["raw_forecast"] if raw_route else cached["latent_future"];baseline_C=raw["raw_forecast"] if raw_route else cached["prediction"]
    checks={"fresh_past_vs_original":(past,raw["history"] if raw_route else cached["latent_history"]),"independent_NumPy_encode":(past,expected_past),"median_future_vs_accepted_cache":(median["latent_future_FP32"],baseline_future),"median_C_vs_accepted_cache":(median["prediction_FP32"],baseline_C)}
    require(set(row["checks"])==set(checks) and set(row["summaries"])==set(SUMMARIES),"COMPLETE_THREE_SUMMARY_AND_CACHE_CHECK_MENU")
    for key,(actual,expected) in checks.items():check_parity(row["checks"][key],actual,expected,key)
    require(row["status"]=="COMPLETE" and row["canonical_batch_size"]==32 and row["stored_horizon"]==64 and row["evaluated_horizon"]==48 and row["actual_full_F_calls"]==(n+31)//32,"FULL_CONTEXT_CANONICAL_PROFILE_AND_RECORDED_CALLS")
    for flag in ("future_labels_used_for_forward","extra_manual_normalization","pipeline_mean_used","true_conditional_mean_or_C_quantile_claimed"):require(row[flag] is False,"POINT_SUMMARY_SCOPE:"+flag)
    complete=np.broadcast_to(raw["observed"].all(2,keepdims=True),raw["observed"].shape);fields=0;verified={}
    for name in SUMMARIES:
        values=candidates[name];entry=row["summaries"][name]
        require(set(values)=={"prediction_FP32","latent_future_FP32","origins"} and values["prediction_FP32"].shape==(n,48,c) and values["latent_future_FP32"].shape==(n,48,k) and all(values[key].dtype==np.float32 and np.isfinite(values[key]).all() for key in ("prediction_FP32","latent_future_FP32")),"FINITE_48_STEP_SUMMARY_AND_C_ARRAYS")
        base.exact(values["origins"],raw["origins"],"candidate_origins")
        future=summary(full[:,:48],name);prediction=future if raw_route else raw["raw_last"][:,None]+base.delta_features(past,future,LAGS)@frozen["D"]
        base.exact(independent[name+"_NumPy_future"],future,"saved_independent_weighted_summary")
        prior.close(independent[name+"_NumPy_prediction"],prediction,"saved_independent_lag_D")
        require(entry["status"]=="COMPLETE" and set(entry["checks"])=={"NumPy_FP32_summary","NumPy_old_lag_D_C"},"COMPLETE_OWN_SUMMARY_CHECKS")
        check_parity(entry["checks"]["NumPy_FP32_summary"],values["latent_future_FP32"],future,"own_FP32_summary")
        check_parity(entry["checks"]["NumPy_old_lag_D_C"],values["prediction_FP32"],prediction,"own_C_decode")
        for key,mask in (("scores_all_observed",raw["observed"]),("scores_complete_mask",complete)):
            if mask.any():fields+=prior.compare_scores(entry[key],prior.scores(values["prediction_FP32"],raw["target"],mask,raw["std"]),name+key)
            else:require(entry[key]=={"status":"NO_COMPLETE_C"},"EMPTY_COMPLETE_MASK_NOT_ZERO_ERROR")
        verified[name]={"scores_all_observed":entry["scores_all_observed"],"scores_complete_mask":entry["scores_complete_mask"]}
    check_diagnostic(row["full64_quantile_diagnostics"],full);check_diagnostic(row["evaluated48_quantile_diagnostics"],full[:,:48])
    require(row["complete_target_positions"]==int(complete[:,:,0].sum()),"EXACT_COMPLETE_C_POSITIONS")
    return {"summaries":verified,"score_fields_checked":fields,"full64_quantile_diagnostics":crossing(full),"evaluated48_quantile_diagnostics":crossing(full[:,:48]),"median_C_byte_equal_to_accepted_cache":median["prediction_FP32"].tobytes()==baseline_C.tobytes(),"median_future_byte_equal_to_accepted_cache":median["latent_future_FP32"].tobytes()==baseline_future.tobytes(),"F_inference_independently_replayed":False}


def check_selection(result,registry,read_lock):
    require(result["all_4_locks_before_DEV"] is True and result["resolved_selection_locks"]==4 and registry["status"]=="ALL_FOUR_RESOLVED_BEFORE_DEV" and registry["DEV_opened_before_seal"] is False and len(registry["conditions"])==4,"FOUR_OWN_TUNE_LOCKS_BEFORE_DEV")
    locks={}
    for item in registry["conditions"]:
        route=item["route"];require(route in ROUTES and route not in locks and item["status"]=="COMPLETE","FOUR_UNIQUE_RAW_AND_COMPRESSED_ROUTES")
        row=result["routes"][route];same(item["selection_lock"],row["selection_lock"]);lock=read_lock(item["selection_lock"])
        require(row["status"]==lock["status"]=="COMPLETE" and lock["route"]==route and lock["DEV_opened_before_lock"] is False and lock["sealed_unix"]<=registry["sealed_unix"],"EXACT_SEALED_ROUTE_SELECTION")
        menu={s:row["splits"]["tune"]["summaries"][s]["scores_all_observed"]["normalized"]["mse"] for s in SUMMARIES}
        require(all(np.isfinite(v) for v in menu.values()) and lock["TUNE_menu"]==menu,"SAME_THREE_ACTUAL_TUNE_MSE_MENU_INCLUDING_RAW")
        winner=min(SUMMARIES,key=lambda s:(menu[s],SUMMARIES.index(s)))
        require(row["selected_summary"]==lock["selected_summary"]==winner and row["selected_TUNE_MSE"]==lock["selected_TUNE_MSE"]==menu[winner],"OWN_TUNE_SELECTION_EXACT_TIE_MEDIAN_THEN_MENU_ORDER")
        for summary_name,entry in row["splits"]["val"]["summaries"].items():require(entry["selected"]==(summary_name==winner) and entry["nonselected_DEV_diagnostic_only"]==(summary_name!=winner),"NO_DEV_WINNER_RESELECTION")
        locks[route]={"selected_summary":winner,"selected_TUNE_MSE":menu[winner],"TUNE_menu":menu,"selection_lock":item["selection_lock"]}
    return locks


def audit_result(path,meter):
    require(all(isinstance(v,str) and len(v)==64 for v in (RESULT_SHA,CONTROLLER_SHA)),"ACTUAL_QUANTILE_RESULT_AND_SOURCE_NOT_REGISTERED")
    check=meter.check;binding=base.file_record(path);require(binding["sha256"]==RESULT_SHA,"EXACT_QUANTILE_ACTUAL_RESULT")
    result=prior.verified_json(binding,check);require(result["status"]=="COMPLETE" and not result["failed_paths"] and set(result["routes"])==set(ROUTES),"FULL_RAW_AND_THREE_E_D_SUMMARY_EXECUTION")
    require(result["controller_source"]["sha256"]==CONTROLLER_SHA and result["module_source"]["sha256"]==MODULE_SHA and result["parent_metadata_reader"]["sha256"]==PARENT_READER_SHA,"EXACT_EXECUTED_QUANTILE_SOURCES")
    for key,value in common.PARENT_SHA.items():require(result[key]["sha256"]==value,"EXACT_ACCEPTED_PARENT:"+key)
    for key in ("new_optimizer_updates","new_E_D_Q_learning_updates"):require(result[key]==0,"NO_ADAPTER_OR_SUMMARY_TRAINING")
    for flag in ("new_backbone_training","native_head_weights_loaded","extra_manual_normalization","pipeline_mean_used","true_conditional_mean_or_decoded_C_quantile_claimed","protected_TEST_accessed","independent_holdout","strict_K_only","whole_deployment_cost_measured"):require(result[flag] is False,"SUMMARY_SCOPE:"+flag)
    require(result["Q_identity"] is True and result["all_three_DEV_summaries_are_not_twelve_independent_datasets"] is True and result["FIT_TUNE_summary_evaluations_completed"]==24 and result["DEV_diagnostic_summary_evaluations_completed"]==12 and result["selected_DEV_evaluations_completed"]==result["selected_routes_completed"]==4 and result["full_quantile_contexts_completed"]==12 and result["new_F_forward_calls"]==result["new_F_forward_attempts"]==44 and result["new_F_model_objects_loaded"]==1,"EXACT_EXECUTION_COUNTS_NOT_PERFORMANCE_SUCCESS")
    guarded=[binding,*[result[k] for k in REFS],*result["source_records"],*result["input_records"],*result["protected_prior_records"]]
    for item in guarded:
        bound(item,check)
        if "snapshot_path" in item:require(prior.file_sha(item["snapshot_path"])==item["sha256"],"EXECUTED_SNAPSHOT_UNCHANGED")
    parent=prior.verified_json(result["latent_calibration_diagnostic"],check);native=prior.verified_json(result["matched_l7_native_diagnostic"],check);cost=prior.verified_json(result["latent_calibration_cost"],check);full=prior.verified_json(result["full_gradient_parent"],check)
    require(result["F_before"]==result["F_after"]==parent["F_parent_signatures_reused"]["before"]==parent["F_parent_signatures_reused"]["after"] and result["runtime_behavior_contract"]==full["runtime_behavior_contract"],"SAME_F_DUAL_HASH_AND_INTERNAL_NORMALIZATION")
    for key in ("model_binary","model_config"):
        same(result[key+"_before"],cost[key+"_before"]);same(result[key+"_after"],cost[key+"_after"]);bound(result[key+"_before"],check);bound(result[key+"_after"],check)
    registry=prior.verified_json(result["selection_registry"],check);selected=check_selection(result,registry,lambda item:prior.verified_json(item,check))
    receipt=result["resource_terminal"];saved=prior.verified_json(receipt,check)
    require(saved=={k:v for k,v in receipt.items() if k not in ("path","sha256")} and receipt["status"]=="COMPLETE" and receipt["gpu_process"] is True and receipt["error"] is None and receipt["started_unix"]<=registry["sealed_unix"]<=receipt["ended_unix"] and receipt["wall_seconds"]>=0 and receipt["process_cpu_seconds"]>=0,"COMPLETE_PROCESS_RECEIPT_NOT_DEPLOYMENT_BENCHMARK")
    for key in ("authorization","continuation_manifest"):bound(receipt[key],check)
    cache={};produced_paths=set();output=Path(binding["path"]).parent
    def load(item):
        key=item["path"],item["sha256"]
        if key not in cache:cache[key]=prior.verified_arrays(item,check)
        return cache[key]
    def produced(item,name):
        path=Path(item["path"]).resolve();require(path==output/name and path not in produced_paths,"UNIQUE_NEW_QUANTILE_ARTIFACT_REGISTERED_PATH")
        produced_paths.add(path);return load(item)
    totals={"full_quantile_contexts_checked":0,"summary_prediction_cells_checked":0,"selection_locks_checked":4,"recorded_full_F_calls_checked":0,"produced_NPZ_artifacts_checked":0,"quantile_diagnostic_records_checked":0,"score_fields_checked":0,"frozen_E_D_center_sources_checked":0,"median_cache_parity_contexts_checked":0,"protected_prior_file_hashes_checked":len(result["protected_prior_records"])}
    report={"status":"PASS","diagnostic":binding,**{key:result[key] for key in REFS},"totals":totals,"routes":{},"selected_summaries":selected,"partial_paths":[],"protected_TEST_accessed":False,"independent_holdout":False,"independent_F_GPU_or_timer_replayed":False,"DEV_access_events_independently_replayed":False,"scope":"Independent stored nine-quantile FP32 weighted point sums, median-cache anchoring, original lag decoder and observed/complete masks, crossing and own-TUNE same-menu Raw selection. Full F outputs are producer evidence, not an F replay; sealed records and source guards support pre-DEV ordering, not independent access-event replay. Proxies are not identified conditional means or decoded C quantiles; process receipt is not deployment latency."}
    for route in ROUTES:
        owner=result["routes"][route];require(set(owner["splits"])=={"fit","tune","val"} and owner["Q_identity"] is True and owner["E_previously_supervised_by_FIT_future"]==route.startswith("E_ONLY") and owner["D_previously_supervised_by_FIT_future"] is False,"FIXED_STATE_SUPERVISION_SCOPE")
        frozen=None
        if route!="RAW_C":
            same(owner["frozen_state_source"],native["states"][route]["frozen_state_source"]);same(owner["frozen_state_source"],parent["states"][route]["frozen_state_source"]);frozen=load(owner["frozen_state_source"])
            require(set(frozen)=={"E","D","center"} and frozen["E"].shape==(17,4) and frozen["D"].shape==(28,17) and frozen["center"].shape==(17,) and all(a.dtype==np.float32 and np.isfinite(a).all() for a in frozen.values()),"ORIGINAL_FROZEN_E_D_CENTER")
            totals["frozen_E_D_center_sources_checked"]+=1
        else:require(owner["frozen_state_source"] is None,"RAW_HAS_NO_COMPRESSION_OR_DECODER_STATE")
        report["routes"][route]={"selected_summary":selected[route]["selected_summary"],"splits":{}}
        for split,n in (("fit",256),("tune",64),("val",30)):
            row=owner["splits"][split];state=STATES[0] if route=="RAW_C" else route;p=parent["states"][state]["families"]["SCALE_ONLY"][{"fit":"FIT","tune":"TUNE","val":"DEV"}[split]]
            same(row["input_source"],p["input_source"]);raw=load(row["input_source"]);prior.check_raw_split(raw,channels=17,windows=n)
            if route=="RAW_C":
                b=row["accepted_median_baseline_source"];same(b["artifact"],row["input_source"]);require(b["member"]=="raw_forecast" and b["member_manifest"]==row["input_source"]["arrays"]["raw_forecast"],"RAW_MEDIAN_EXACT_SOURCE_MEMBER")
                cached={"prediction":raw["raw_forecast"],"latent_future":raw["raw_forecast"]}
            else:
                source=native["states"][route]["Bolt"][split]["artifact"];same(row["accepted_median_baseline_source"],source);same(source,p["cached_Bolt_source"]);cached=load(source)
                for key in ("target","observed","raw_last","origins","mean","std"):base.exact(cached[key],raw[key],"accepted_median_raw_binding_"+key)
            f=produced(row["artifact"],route+"_"+split+"_FULL_QUANTILES.npz");candidates={}
            for name in SUMMARIES:
                same(row["candidate_artifacts"][name],row["summaries"][name]["artifact"]);same(row["summaries"][name]["full_quantile_artifact"],row["artifact"])
                candidates[name]=produced(row["candidate_artifacts"][name],route+"_"+name+"_"+split+".npz")
            independent=produced(row["independent_artifact"],route+"_"+split+"_INDEPENDENT.npz")
            verified=check_cell(row,f,candidates,independent,raw,cached,frozen,route);report["routes"][route]["splits"][split]=verified
            totals["full_quantile_contexts_checked"]+=1;totals["summary_prediction_cells_checked"]+=3;totals["recorded_full_F_calls_checked"]+=row["actual_full_F_calls"];totals["produced_NPZ_artifacts_checked"]+=5;totals["quantile_diagnostic_records_checked"]+=2;totals["score_fields_checked"]+=verified["score_fields_checked"];totals["median_cache_parity_contexts_checked"]+=1
    require(totals["full_quantile_contexts_checked"]==12 and totals["summary_prediction_cells_checked"]==36 and totals["produced_NPZ_artifacts_checked"]==len(produced_paths)==60 and totals["recorded_full_F_calls_checked"]==44,"FULL_QUANTILE_AUDIT_COVERAGE")
    for item in guarded:bound(item,check)
    return report


def execute(path,output):
    from continuation_runtime_v1 import Meter,write_new
    output=Path(output).resolve();require(not output.exists(),"FRESH_POINT_SUMMARY_AUDIT_OUTPUT");meter=Meter("audit_quantile_point_summary_"+output.stem,gpu=False);failure=None;source=base.file_record(__file__)
    try:
        require(all(isinstance(v,str) and len(v)==64 for v in (RESULT_SHA,CONTROLLER_SHA)),"ACTUAL_QUANTILE_RESULT_AND_SOURCE_NOT_REGISTERED");meter.protected_check();initialize_numpy()
        import sys
        sys.path.insert(0,str(ROOT/"research/smallk_failure_followup_20261014"))
        from prepare_inputs import install_protected_test_guard
        from threadpoolctl import threadpool_limits
        install_protected_test_guard()
        with threadpool_limits(limits=2):report=audit_result(path,meter)
        meter.protected_check();require(prior.file_sha(__file__)==source["sha256"],"QUANTILE_AUDITOR_SOURCE_UNCHANGED");json.dumps(report,allow_nan=False)
    except Exception as error:failure={"type":type(error).__name__,"message":str(error)};report={"status":"FAILED_PRESERVED","failure":failure}
    finally:report["audit_source"]=source;report["resource_terminal"]=meter.close(report["status"],failure);write_new(output,report)
    print(json.dumps({"status":report["status"],"totals":report.get("totals"),"failure":failure}),flush=True);return 0 if report["status"]=="PASS" else 2


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--result",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args();raise SystemExit(execute(args.result,args.output))
