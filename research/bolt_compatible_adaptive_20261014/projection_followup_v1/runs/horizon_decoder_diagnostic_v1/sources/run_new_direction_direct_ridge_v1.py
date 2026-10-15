"""Twelve masked direct-C ridge paths; cached past features, no F/native model or optimizer."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import gc
import hashlib
import json
import shutil
import sys
import time

import run_linear_e_adaptation_v1 as old

HERE,ROOT,BASE,OLD,require=old.HERE,old.ROOT,old.BASE,old.OLD,old.require
DATASETS={"robin":(17,4,"val",30),"beijing_aotizhongxin":(11,2,"eval",146)}
ALPHAS=(0.,1e-4,.001,.01,.1,1.,10.)
CROSS_RESULT_SHA="99bffd5831143ca35947099fef9567f89dd209d83451661decee0682b2b8e13c"
CROSS_AUDIT_SHA="90c68a9057909fb5b6c23284e5a45523cdf126da178a5ba0d9d9e96a50dede30"
CROSS_AUDITOR_SOURCE={"path":str(HERE/"audit_predictive_cross_moment_v1.py"),"sha256":"6577d15a200728dc585bf2c8e93ccff6baabd1ba916384dfb2cce7aa0014a731"}
MODULE_SHA="967fb95eef55f9597a73a01efc0118d8c7e936586adb724f7c5a449334ab3717"
CROSS_COUNTS={"states_checked":8,"direction_fits_checked":8,"decoder_fits_checked":8,"decoder_response_columns_checked":112,"FIT_TUNE_cells_checked":16,"DEV_cells_checked":8,"independent_saved_F_arithmetic_cells_checked":24,"oracle_dependency_masks_checked":24,"score_fields_checked":3744,"state_locks_checked":8,"legacy_PCA_nonrequired_diagnostics_preserved":6,"protected_prior_file_hashes_checked":1928,"unique_dataset_split_timestamp_contexts_checked":6}
CONDITIONS=tuple((data,name) for data,(_,k,_,_) in DATASETS.items() for name in ("RAW_CURRENT_C","RAW_HISTORY512",*(kind+"_K"+str(width)+"_HISTORY512" for kind in ("PCA","CROSS_MOMENT") for width in (k,2*k))))


def load_gates(args):
    require(all(isinstance(v,str) and len(v)==64 for v in (CROSS_RESULT_SHA,CROSS_AUDIT_SHA,MODULE_SHA)) and bool(CROSS_COUNTS),"ACTUAL_CROSS_AUDIT_NOT_REGISTERED")
    refs={"cross_moment_diagnostic":old.record(args.cross_result),"cross_moment_audit":old.record(args.cross_audit)}
    require(refs["cross_moment_diagnostic"]["sha256"]==CROSS_RESULT_SHA and refs["cross_moment_audit"]["sha256"]==CROSS_AUDIT_SHA,"EXACT_ACCEPTED_CROSS_RESULT_AUDIT")
    cross,audit=old.read(args.cross_result),old.read(args.cross_audit);require(cross["status"]=="COMPLETE" and audit["status"]=="PASS" and not cross["failed_paths"] and cross["protected_TEST_accessed"] is audit["protected_TEST_accessed"] is False and all(audit["totals"].get(k)==v for k,v in CROSS_COUNTS.items()),"FULL_EIGHT_STATE_ACCEPTANCE")
    old.bound(audit["diagnostic"],refs["cross_moment_diagnostic"]);old.bound(audit["audit_source"],CROSS_AUDITOR_SOURCE);old.bound(audit["state_registry"],cross["state_registry"])
    old.bound({"path":str(HERE/"masked_direct_ridge_v1.py"),"sha256":MODULE_SHA});old.bound(cross["controller_source"]);old.bound(cross["module_source"])
    from run_orthogonal_basis_controls_v1 import load_gates as parent
    parentargs=SimpleNamespace(**{argument:cross[key]["path"] for argument,key in (("calibration_result","latent_calibration_diagnostic"),("calibration_audit","latent_calibration_audit"),("calibration_cost","latent_calibration_cost"),("calibration_cost_audit","latent_calibration_cost_audit"),("report","parent_report"),("report_manifest","parent_report_manifest"),("report_audit","parent_report_audit"))})
    calibration,native,science,full,parentrefs=parent(parentargs)
    for key,value in parentrefs.items():
        if key in cross:old.bound(value,cross[key])
    refs.update(parentrefs,cross_state_registry=cross["state_registry"],cross_auditor_source=audit["audit_source"])
    return cross,science,refs


def feature_payload(raw,route,latent_history=None):
    import numpy as np
    require(raw["history"].dtype==raw["raw_last"].dtype==np.float32 and old.byte_equal(raw["history"][:,-1],raw["raw_last"]),"ORIGINAL_FP32_PAST_CURRENT")
    n=len(raw["history"])
    if route=="RAW_CURRENT_C":return raw["raw_last"].copy(),np.zeros_like(raw["raw_last"])
    history=raw["history"] if route=="RAW_HISTORY512" else latent_history
    require(history is not None and history.dtype==np.float32 and history.ndim==3 and history.shape[:2]==(n,512) and np.isfinite(history).all(),"FINITE_FP32_PAST_FEATURE_STREAMS")
    return np.ascontiguousarray((history-history[:,-1:]).reshape(n,-1),dtype=np.float32),raw["raw_last"].copy()


def response_payload(raw,route):
    import numpy as np
    observed=raw["observed"] if route!="RAW_CURRENT_C" else np.broadcast_to(raw["observed"].all(2,keepdims=True),raw["observed"].shape)
    response=np.zeros(raw["target"].shape,np.float64)
    if route=="RAW_CURRENT_C":response[observed]=raw["target"][observed].astype(np.float64)
    else:
        anchor=np.broadcast_to(raw["raw_last"].astype(np.float64)[:,None],response.shape);response[observed]=raw["target"][observed].astype(np.float64)-anchor[observed]
    return response.reshape(len(response),-1),observed.reshape(len(response),-1).copy()


def predict(features,state,anchor,check=lambda value:None,*,batch_size=32):
    import numpy as np
    import masked_direct_ridge_v1 as module
    require(anchor.dtype==np.float32 and anchor.ndim==2 and len(anchor)==len(features) and state["coefficient_FP32"].shape[1]==48*anchor.shape[1],"SAME_FP32_CURRENT_C_AND_H48_OUTPUT")
    pieces=[]
    for start in range(0,len(features),batch_size):
        p=module.predict(features[start:start+batch_size],state).reshape(-1,48,anchor.shape[1])+anchor[start:start+batch_size,None]
        require(p.dtype==np.float32 and np.isfinite(p).all(),"FINITE_ACTUAL_FP32_DIRECT_C");pieces.append(p);check({"stage":"ACTUAL_CPU_FP32_FUSED_DIRECT_RIDGE","start":start,"batch_size":batch_size,"new_F_forward":False})
    return np.ascontiguousarray(np.concatenate(pieces),dtype=np.float32)


def coefficient_records(state):
    import numpy as np
    arrays={key:value for key,value in state.items() if key!="alpha"};digest=hashlib.sha256();records={}
    for key,value in sorted(arrays.items()):
        value=np.ascontiguousarray(value);records[key]={"shape":list(value.shape),"dtype":str(value.dtype),"sha256":hashlib.sha256(value.tobytes()).hexdigest()}
        for part in (key.encode(),str(value.dtype).encode(),str(value.shape).encode(),value.tobytes()):digest.update(part)
    return records,digest.hexdigest()


def fit_info(metadata):
    import numpy as np
    arrays={key:metadata[key] for key in ("global_feature_mean_FP64","global_feature_scale_FP64","global_constant_features")};summary={key:value for key,value in metadata.items() if key not in (*arrays,"groups")};summary["groups"]=[]
    for index,group in enumerate(metadata["groups"]):
        prefix="group_"+str(index)+"_";names={}
        for key in ("observed_mask","feature_mean_FP64","feature_scale_FP64","constant_features","singularvalues"):
            arrays[prefix+key]=group[key];names[key]=prefix+key
        arrays[prefix+"response_columns"]=np.asarray(group["columns"],dtype=np.int64)
        summary["groups"].append({"response_columns":group["columns"],"observed_count":group["observed_count"],"minimum_norm_retained_rank":group["minimum_norm_retained_rank"],"array_members":names})
    summary.update(group_count=len(summary["groups"]),GLOBAL_FIT_mean_is_recorded_but_subset_centering_eliminates_unpenalized_intercept=True)
    return summary,arrays


def select_alpha(candidates):
    import math
    valid=[p for p in candidates if p["status"]=="COMPLETE" and math.isfinite(p.get("TUNE_MSE",float("nan")))]
    require(bool(valid),"FINITE_COMPLETE_RIDGE_CANDIDATE_REQUIRED");return min(valid,key=lambda p:(p["TUNE_MSE"],ALPHAS.index(p["alpha"])))


def resolve_paths(result,run,seal,registry):
    resolved={};locks=[]
    for data,route in CONDITIONS:
        row={}
        try:row=run(data,route);require(row["status"]=="COMPLETE","SEVEN_COMPLETE_ALPHA_CANDIDATES_REQUIRED")
        except Exception as error:row=result["datasets"][data]["routes"].get(route,row);row.update(status="FAILED_PATH_PRESERVED",failure={"type":type(error).__name__,"message":str(error)})
        row["selection_lock"]=seal(data,route,row);result["datasets"][data]["routes"][route]=row;locks.append({"dataset":data,"route":route,"status":row["status"],"selection_lock":row["selection_lock"]})
        if row["status"]=="COMPLETE":resolved[data,route]=row
        else:result["failed_paths"].append({"dataset":data,"route":route,"stage":"FIT_TUNE_AND_ALPHA_SELECTION","failure":row.get("failure")})
        gc.collect()
    result["selection_registry"]=registry(locks);result.update(resolved_selection_locks=len(locks),all_12_locks_before_DEV=True);return resolved


def infer_development(result,resolved,infer):
    require(result["all_12_locks_before_DEV"] and result["resolved_selection_locks"]==12,"ALL_TWELVE_LOCKS_BEFORE_ANY_DEV")
    for (data,route),row in resolved.items():
        try:row["DEV"]=infer(data,route,row);require(row["DEV"]["status"]=="COMPLETE","SELECTED_DEV_FINITE_PREDICTION_REQUIRED");result["selected_DEV_evaluations_completed"]+=1
        except Exception as error:result["failed_paths"].append({"dataset":data,"route":route,"stage":"SELECTED_DEV","failure":{"type":type(error).__name__,"message":str(error)}})
        gc.collect()


def execute(args):
    cross,science,refs=load_gates(args);output=old.fresh_output(args.tag);sys.path[:0]=[str(HERE),str(BASE),str(OLD),str(ROOT)]
    import numpy as np
    from threadpoolctl import threadpool_limits
    import masked_direct_ridge_v1 as module
    import fixed_decoder_span_v1 as io
    import continuation_runtime_v1 as rt
    from decoder_metric_probe_v1 import masked_scores
    from run_fit_target_comparison_v1 import prior_file_records
    from prepare_inputs import install_protected_test_guard
    require(old.record(module.__file__)["sha256"]==MODULE_SHA and tuple(module.ALPHAS)==ALPHAS,"FROZEN_MASKED_DIRECT_RIDGE_MODULE_AND_GRID")
    output.mkdir(parents=True,exist_ok=False);meter=rt.Meter("new_direction_direct_ridge_"+args.tag,gpu=False);failure=None
    result={"status":"RUNNING",**refs,"datasets":{d:{"routes":{}} for d in DATASETS},"failed_paths":[],"source_records":[],"ridge_fit_attempts":0,"ridge_fits_completed":0,"alpha_candidates_completed":0,"FIT_TUNE_prediction_evaluations_completed":0,"selected_DEV_evaluations_completed":0,"resolved_selection_locks":0,"all_12_locks_before_DEV":False,"new_F_forward_calls":0,"new_F_model_objects_loaded":0,"new_native_training_updates":0,"new_optimizer_updates":0,"new_backbone_training":False,"old_ridge_coefficients_or_predictions_reused":False,"whole_deployment_cost_measured":False,"D_bypassed_by_direct_C_readouts":True,"same_D_native_or_capacity_match_claimed":False,"CPU_FP32_canonical_batch_size":32,"protected_TEST_accessed":False,"independent_holdout":False,"strict_K_only":False,"resource_scope":"CPU FIT group thin SVD/seven-alpha fitting, actual fused FP32 forecasts, hash/file IO and diagnostics; no F/model/GPU/native or whole-cost timing"}
    checked,contexts={},{};protected=[]
    try:
        install_protected_test_guard();meter.protected_check();protected=prior_file_records(cross,science,refs)
        folder=output/"sources";folder.mkdir()
        for path in (Path(__file__),HERE/"masked_direct_ridge_v1.py",HERE/"PLAN_NEW_DIRECTION_NATIVE_BASELINES_KO_v1.md",HERE/"run_orthogonal_basis_controls_v1.py",HERE/"fixed_decoder_span_v1.py",HERE/"decoder_metric_probe_v1.py",HERE/"continuation_runtime_v1.py",OLD/"prepare_inputs.py"):
            destination=folder/path.name;shutil.copyfile(path,destination);result["source_records"].append({**old.record(path),"snapshot_path":str(destination)})
        result.update(controller_source=old.record(__file__),module_source=old.record(module.__file__),plan=old.record(HERE/"PLAN_NEW_DIRECTION_NATIVE_BASELINES_KO_v1.md"),F_parent_signatures_reused={"before":cross["F_before"],"after":cross["F_after"]})
        def load(item):return io.checked_payload({**item,"path":str(old.bound(item))},meter,checked)
        def context(data,split):
            require(split in ("fit","tune") or result["all_12_locks_before_DEV"],"TWELVE_RIDGE_LOCKS_BEFORE_DEV_ARRAY_IO")
            if (data,split) not in contexts:
                source=science["datasets"][data]["spaces"]["PCA"]["conditions"]["L7_RECON"]["splits"][split]["input_source"];raw=load(source);c,_,dev,devn=DATASETS[data];n={"fit":256,"tune":64}.get(split,devn)
                require(split in ("fit","tune",dev) and raw["history"].shape==(n,512,c) and raw["target"].shape==raw["observed"].shape==(n,48,c) and raw["history"].dtype==raw["target"].dtype==np.float32 and raw["observed"].dtype==np.bool_ and np.isfinite(raw["history"]).all() and np.isfinite(raw["target"][raw["observed"]]).all() and old.byte_equal(raw["raw_last"],raw["history"][:,-1]),"ORIGINAL_COHORT_PAST_CURRENT_OBSERVED_Y")
                contexts[data,split]=(raw,source)
            return contexts[data,split]
        def features(data,route,split):
            raw,input_source=context(data,split);latent=None;source=None;state_source=None
            if route not in ("RAW_CURRENT_C","RAW_HISTORY512"):
                name=route.removesuffix("_HISTORY512");owner=cross["datasets"][data]["conditions"][name];entry=owner["splits"][split];old.bound(entry["input_source"],input_source);source=entry["artifact"];state_source=owner["state_artifact"];arrays=load(source);require(old.byte_equal(arrays["origins"],raw["origins"]),"EXACT_ENCODED_PAST_ORIGINS");latent=arrays["latent_history"]
                require(latent.shape==(len(raw["history"]),512,owner["K"]) and latent.dtype==np.float32,"SOURCE_CANONICAL_TORCH32_Z_ONLY")
            x,anchor=feature_payload(raw,route,latent);return x,anchor,raw,{"input_source":input_source,"encoded_past_source":source,"frozen_E_D_source":state_source,"future_latent_or_oracle_used_as_features":False}
        def scores(prediction,raw):
            complete=np.broadcast_to(raw["observed"].all(2,keepdims=True),raw["observed"].shape)
            return {"status":"COMPLETE","scores_all_observed":masked_scores(prediction,raw["target"],raw["observed"],raw["std"]),"scores_complete_mask":masked_scores(prediction,raw["target"],complete,raw["std"]) if complete.any() else {"status":"NO_COMPLETE_C"}}
        def run(data,route):
            fitx,fita,fit,fitbindings=features(data,route,"fit");tunex,tunea,tune,tunebindings=features(data,route,"tune");response,observed=response_payload(fit,route);normalization="PER_RESPONSE_MASK" if route=="RAW_CURRENT_C" else "GLOBAL_FIT";started=time.perf_counter();result["ridge_fit_attempts"]+=1
            states,metadata=module.fit_path(fitx,response,observed,normalization=normalization);elapsed=time.perf_counter()-started;summary,info_arrays=fit_info(metadata)
            row={"status":"RUNNING","route":route,"feature_dimension":fitx.shape[1],"response_dimension":response.shape[1],"normalization":normalization,"FIT_feature_binding":{"shape":list(fitx.shape),"dtype":str(fitx.dtype),"sha256":hashlib.sha256(fitx.tobytes()).hexdigest()},"FIT_response_mask_binding":{"shape":list(observed.shape),"dtype":str(observed.dtype),"sha256":hashlib.sha256(observed.tobytes()).hexdigest()},"fit_metadata":summary,"fit_wall_seconds":elapsed,"FIT_sources":fitbindings,"TUNE_sources":tunebindings,"candidates":[],"response":"future level" if route=="RAW_CURRENT_C" else "future FP64 innovation from FP32 Y/current endpoints","global_FIT_RMS_with_mask_subset_intercept_centering":normalization=="GLOBAL_FIT","future_label_placeholders_excluded_by_mask":True,"feature_subtraction_FP32":route!="RAW_CURRENT_C","coefficient_deployment":"feature32 @ fused_coefficient32 + fused_intercept32 + anchor32; CPU canonical32"}
            result["datasets"][data]["routes"][route]=row;row["fit_artifact"]=io.save_npz(output/(data+"_"+route+"_FIT_INFO.npz"),info_arrays);result["ridge_fits_completed"]+=1;meter.check({"stage":"RIDGE_ALL_ALPHA_FIT_COMPLETE","dataset":data,"route":route,"groups":summary["group_count"]})
            for state in states:
                alpha=state["alpha"];candidate={"alpha":alpha,"status":"FAILED_CANDIDATE_PRESERVED"};candidate["coefficient_array_records"],candidate["coefficient_value_sha256"]=coefficient_records(state)
                try:
                    p,t=predict(fitx,state,fita,meter.check),predict(tunex,state,tunea,meter.check);candidate.update(status="COMPLETE",FIT=scores(p,fit),TUNE=scores(t,tune));candidate["TUNE_MSE"]=candidate["TUNE"]["scores_all_observed"]["normalized"]["mse"]
                    artifact=io.save_npz(output/(data+"_"+route+"_ALPHA_"+format(alpha,"g")+".npz"),{"FIT_prediction_FP32":p,"TUNE_prediction_FP32":t,"FIT_origins":fit["origins"],"TUNE_origins":tune["origins"]});candidate["prediction_artifact"]=artifact;result["alpha_candidates_completed"]+=1;result["FIT_TUNE_prediction_evaluations_completed"]+=2
                except Exception as error:candidate["failure"]={"type":type(error).__name__,"message":str(error)}
                row["candidates"].append(candidate)
            winner=select_alpha(row["candidates"]);index=ALPHAS.index(winner["alpha"]);selected=states[index];selected_artifact=io.save_npz(output/(data+"_"+route+"_SELECTED_COEFFICIENTS.npz"),{k:v for k,v in selected.items() if k!="alpha"})
            row.update(selected_alpha=winner["alpha"],selected_TUNE_MSE=winner["TUNE_MSE"],selected_coefficients=selected_artifact,selected_coefficient_value_sha256=winner["coefficient_value_sha256"],FIT=winner["FIT"],TUNE=winner["TUNE"],selected_FIT_TUNE_prediction_artifact=winner["prediction_artifact"],selection_rule="Own actual FP32 TUNE observed macro MSE; exact tie registered alpha order",selected_FIT_TUNE_reused_without_new_inference=True,status="COMPLETE" if all(p["status"]=="COMPLETE" for p in row["candidates"]) else "PARTIAL_CANDIDATE_MENU_PRESERVED")
            del states,metadata,info_arrays,fitx,tunex,response,observed,p,t;gc.collect();meter.check({"stage":"RIDGE_ROUTE_LARGE_ARRAYS_RELEASED","dataset":data,"route":route});return row
        def seal(data,route,row):
            path=output/(data+"_"+route+"_SELECTION_LOCK.json");rt.write_new(path,{"status":row["status"],"dataset":data,"route":route,"selected_alpha":row.get("selected_alpha"),"selected_TUNE_MSE":row.get("selected_TUNE_MSE"),"selected_coefficients":row.get("selected_coefficients"),"selected_coefficient_value_sha256":row.get("selected_coefficient_value_sha256"),"DEV_opened_before_lock":False,"sealed_unix":time.time()});return old.record(path)
        def registry(locks):
            path=output/"RIDGE_SELECTION_REGISTRY.json";rt.write_new(path,{"status":"ALL_TWELVE_RESOLVED_BEFORE_DEV","conditions":locks,"DEV_opened_before_seal":False,"sealed_unix":time.time()});return old.record(path)
        def development(data,route,row):
            old.bound(result["selection_registry"]);old.bound(row["selection_lock"]);state=load(row["selected_coefficients"]);records,digest=coefficient_records(state);require(digest==row["selected_coefficient_value_sha256"],"SELECTED_COEFFICIENT_VALUES_UNCHANGED");x,anchor,raw,bindings=features(data,route,DATASETS[data][2]);p=predict(x,state,anchor,meter.check);entry=scores(p,raw);entry.update(artifact=io.save_npz(output/(data+"_"+route+"_SELECTED_DEV.npz"),{"prediction_FP32":p,"origins":raw["origins"]}),**bindings,selected_coefficients=row["selected_coefficients"]);return entry
        with threadpool_limits(limits=2):resolved=resolve_paths(result,run,seal,registry);infer_development(result,resolved,development)
        for item in protected+list(checked.values())+result["source_records"]:old.bound(item)
        for item in result["source_records"]:require(old.record(item["snapshot_path"])["sha256"]==item["sha256"],"EXECUTED_SOURCE_SNAPSHOT_UNCHANGED")
        result.update(input_records=list(checked.values()),protected_prior_records=protected,status="COMPLETE" if len(resolved)==12 and result["ridge_fits_completed"]==12 and result["alpha_candidates_completed"]==84 and result["FIT_TUNE_prediction_evaluations_completed"]==168 and result["selected_DEV_evaluations_completed"]==12 and not result["failed_paths"] else "PARTIAL_COMPLETED");meter.protected_check();json.dumps(result,allow_nan=False)
    except Exception as error:failure={"type":type(error).__name__,"message":str(error)};result.update(status="FAILED_PRESERVED",failure=failure)
    finally:result["resource_terminal"]=meter.close(result["status"],failure);rt.write_new(output/"RESULT.json",result)
    print(json.dumps({"status":result["status"],"result":str(output/"RESULT.json")}),flush=True);return 0 if result["status"]=="COMPLETE" else 2


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("tag","cross-result","cross-audit"):parser.add_argument("--"+key,required=True)
    raise SystemExit(execute(parser.parse_args()))
