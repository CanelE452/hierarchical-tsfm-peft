"""Adaptive upper-grid extension for ten history ridge paths; original v1 remains intact."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import gc
import hashlib
import json
import shutil
import sys
import time

import run_new_direction_direct_ridge_v1 as parent

old,HERE,ROOT,BASE,OLD,require=parent.old,parent.HERE,parent.ROOT,parent.BASE,parent.OLD,parent.require
CONDITIONS=tuple((data,route) for data,route in parent.CONDITIONS if route!="RAW_CURRENT_C")
NEW_ALPHAS=(100.,1000.,10000.)
COEFFICIENT_KEYS=("coefficient_FP64","intercept_FP64","coefficient_FP32","intercept_FP32")
MENU=tuple("ALPHA_"+format(a,"g") for a in (*parent.ALPHAS,*NEW_ALPHAS))+("INTERCEPT_ONLY","CURRENT_C_ANCHOR")
ORIGINAL_RESULT_SHA="b8d7eae62ce5bb52b3d46ffc543f916f8b81857028098370a5b2f8398510f1ed"
ORIGINAL_CONTROLLER_SHA="7dd19e1504f0ff4b323c52412fac492f2b5c17c23e1ba0d898691a672f5ac88e"
ORIGINAL_MODULE_SHA="967fb95eef55f9597a73a01efc0118d8c7e936586adb724f7c5a449334ab3717"
ORIGINAL_AUDIT_SHA="8bfeb29cc6836d2ff5528c30ee13a4300b79cc4c250a452b9c5c81fdfc9bac90"
ORIGINAL_AUDITOR_SOURCE={"path":str(HERE/"audit_new_direction_direct_ridge_v1.py"),"sha256":"d8b6e8b8ed6862af69e78ecc63758b13fd03378862aa6c6c6644052fa36d85d4"}
ORIGINAL_AUDIT_COUNTS={"selected_ridge_fits_checked":12,"mask_groups_checked":1319,"alpha_candidates_checked":84,"FIT_TUNE_prediction_evaluations_checked":168,"selected_DEV_prediction_evaluations_checked":12,"selected_coefficient_FP32_arithmetic_predictions_checked":36,"selection_locks_checked":12,"score_fields_checked":4320,"selected_upper_grid_boundary_conditions":10,"protected_source_input_and_result_hashes_checked":2155}
MODULE_SHA="49c08461bd7f114511bd9bf3cdcfa8e5fbf8b89152edde4435c6944767d8ba46"


def load_gates(result_path,audit_path):
    require(all(isinstance(v,str) and len(v)==64 for v in (ORIGINAL_RESULT_SHA,ORIGINAL_AUDIT_SHA,MODULE_SHA)) and isinstance(ORIGINAL_AUDITOR_SOURCE,dict) and bool(ORIGINAL_AUDIT_COUNTS),"ACTUAL_ORIGINAL_RIDGE_PASS_NOT_YET_REGISTERED")
    refs={"original_ridge_diagnostic":old.record(result_path),"original_ridge_audit":old.record(audit_path)}
    require(refs["original_ridge_diagnostic"]["sha256"]==ORIGINAL_RESULT_SHA and refs["original_ridge_audit"]["sha256"]==ORIGINAL_AUDIT_SHA,"EXACT_ORIGINAL_RIDGE_AND_PASS")
    original,audit=old.read(result_path),old.read(audit_path)
    require(original["status"]=="COMPLETE" and audit["status"]=="PASS" and original["protected_TEST_accessed"] is audit["protected_TEST_accessed"] is False and not original["failed_paths"] and not audit.get("partial_paths",[]) and all(type(audit["totals"].get(k)) is int and audit["totals"][k]==v for k,v in ORIGINAL_AUDIT_COUNTS.items()),"COMPLETE_ORIGINAL_RIDGE_AND_INDEPENDENT_COVERAGE")
    old.bound(audit["diagnostic"],refs["original_ridge_diagnostic"]);old.bound(audit["audit_source"],ORIGINAL_AUDITOR_SOURCE);old.bound(audit["selection_registry"],original["selection_registry"])
    old.bound(original["controller_source"],{"path":parent.__file__,"sha256":ORIGINAL_CONTROLLER_SHA});old.bound(original["module_source"],{"path":str(HERE/"masked_direct_ridge_v1.py"),"sha256":ORIGINAL_MODULE_SHA});old.bound({"path":str(HERE/"masked_direct_ridge_extension_v1.py"),"sha256":MODULE_SHA})
    require(original["ridge_fits_completed"]==original["resolved_selection_locks"]==12 and original["alpha_candidates_completed"]==84 and original["FIT_TUNE_prediction_evaluations_completed"]==168 and original["selected_DEV_evaluations_completed"]==12 and original["all_12_locks_before_DEV"] is True,"ALL_ORIGINAL_TWELVE_PATHS_AND_SELECTIONS")
    cross,science,parentrefs=parent.load_gates(SimpleNamespace(cross_result=original["cross_moment_diagnostic"]["path"],cross_audit=original["cross_moment_audit"]["path"]))
    for key in ("cross_moment_diagnostic","cross_moment_audit","cross_state_registry","cross_auditor_source"):old.bound(original[key],parentrefs[key])
    for data,route in CONDITIONS:
        row=original["datasets"][data]["routes"][route]
        require(row["status"]=="COMPLETE" and row["selected_alpha"]==10. and [p["alpha"] for p in row["candidates"]]==list(parent.ALPHAS) and all(p["status"]=="COMPLETE" for p in row["candidates"]),"TEN_HISTORY_PATHS_ORIGINAL_UPPER_GRID_SELECTION")
        lock=old.read(old.bound(row["selection_lock"]));require(lock["status"]=="COMPLETE" and lock["DEV_opened_before_lock"] is False and lock["selected_alpha"]==row["selected_alpha"],"ORIGINAL_TUNE_SELECTION_LOCK");old.bound(lock["selected_coefficients"],row["selected_coefficients"])
    return original,audit,cross,science,{**parentrefs,**refs,"original_ridge_selection_registry":original["selection_registry"]}


def zero_slope_states(features,response,observed):
    import numpy as np
    require(features.dtype==np.float32 and features.ndim==2 and response.dtype==np.float64 and response.ndim==2 and observed.dtype==np.bool_ and observed.shape==response.shape and len(features)==len(response) and np.isfinite(response[observed]).all(),"FINITE_OBSERVED_FIT_INNOVATIONS_FOR_ZERO_SLOPE_CONTROLS")
    counts=observed.sum(0);require((counts>=2).all(),"TWO_TRUE_OBSERVED_FIT_RESPONSES_PER_ZERO_CONTROL_COLUMN")
    intercept=np.array([response[observed[:,j],j].mean() for j in range(response.shape[1])],dtype=np.float64)
    result=[]
    for value in (intercept,np.zeros_like(intercept)):
        coefficient=np.zeros((features.shape[1],response.shape[1]),np.float64)
        result.append({"alpha":None,"coefficient_FP64":coefficient,"intercept_FP64":value.copy(),"coefficient_FP32":coefficient.astype(np.float32),"intercept_FP32":value.astype(np.float32)})
    return result


def original_menu(row):
    candidates=[]
    for index,item in enumerate(row["candidates"]):
        require(item["status"]=="COMPLETE" and item["alpha"]==parent.ALPHAS[index],"UNCHANGED_ORIGINAL_SEVEN_ALPHA_MENU")
        candidates.append({**item,"candidate_id":MENU[index],"kind":"RIDGE","new_in_this_extension":False,"original_candidate_index":index,"original_prediction_artifact":item["prediction_artifact"],"new_FIT_TUNE_inference":False})
    return candidates


def select_candidate(candidates):
    import math
    require(len({p["candidate_id"] for p in candidates})==len(candidates) and all(p["candidate_id"] in MENU for p in candidates),"UNIQUE_REGISTERED_TWELVE_CANDIDATE_IDENTIFIERS")
    valid=[p for p in candidates if p["status"]=="COMPLETE" and math.isfinite(p.get("TUNE_MSE",float("nan")))]
    require(bool(valid),"COMPLETE_FINITE_TUNE_CANDIDATE_REQUIRED")
    return min(valid,key=lambda p:(p["TUNE_MSE"],MENU.index(p["candidate_id"])))


def resolve_paths(result,run,seal,registry):
    resolved={};locks=[]
    for data,route in CONDITIONS:
        row={}
        try:row=run(data,route);require(row["status"]=="COMPLETE","FULL_TWELVE_CANDIDATE_MENU_REQUIRED")
        except Exception as error:row=result["datasets"][data]["routes"].get(route,row);row.update(status="FAILED_PATH_PRESERVED",failure={"type":type(error).__name__,"message":str(error)})
        row["selection_lock"]=seal(data,route,row);result["datasets"][data]["routes"][route]=row;locks.append({"dataset":data,"route":route,"status":row["status"],"selection_lock":row["selection_lock"]})
        if row["status"]=="COMPLETE":resolved[data,route]=row
        else:result["failed_paths"].append({"dataset":data,"route":route,"stage":"FIT_TUNE_MENU_AND_SELECTION","failure":row.get("failure")})
        gc.collect()
    result["selection_registry"]=registry(locks);result.update(resolved_selection_locks=len(locks),all_10_locks_before_DEV=True);return resolved


def infer_development(result,resolved,infer):
    require(result["all_10_locks_before_DEV"] and result["resolved_selection_locks"]==10,"ALL_TEN_EXTENSION_LOCKS_BEFORE_DEV")
    for (data,route),row in resolved.items():
        try:
            row["DEV"]=infer(data,route,row);require(row["DEV"]["status"]=="COMPLETE","COMPLETE_SELECTED_DEV_OR_EXACT_ORIGINAL_REUSE")
            key="reused_original_DEV_prediction_evaluations" if row["selected_original_candidate"] else "new_selected_DEV_prediction_evaluations";result[key]+=1
        except Exception as error:result["failed_paths"].append({"dataset":data,"route":route,"stage":"SELECTED_DEV","failure":{"type":type(error).__name__,"message":str(error)}})
        gc.collect()


def execute(args):
    original,audit,cross,science,refs=load_gates(args.ridge_result,args.ridge_audit);output=old.fresh_output(args.tag);sys.path[:0]=[str(HERE),str(BASE),str(OLD),str(ROOT)]
    import numpy as np
    from threadpoolctl import threadpool_limits
    import masked_direct_ridge_extension_v1 as module
    import fixed_decoder_span_v1 as io
    import continuation_runtime_v1 as rt
    from decoder_metric_probe_v1 import masked_scores
    from run_fit_target_comparison_v1 import prior_file_records
    from prepare_inputs import install_protected_test_guard
    require(old.record(module.__file__)["sha256"]==MODULE_SHA and tuple(module.ALPHAS)==NEW_ALPHAS,"FROZEN_THREE_ALPHA_EXTENSION_MODULE")
    output.mkdir(parents=True,exist_ok=False);meter=rt.Meter("ridge_regularization_extension_"+args.tag,gpu=False);failure=None
    result={"status":"RUNNING",**refs,"datasets":{d:{"routes":{}} for d in parent.DATASETS},"failed_paths":[],"source_records":[],"extension_fit_attempts":0,"extension_fits_completed":0,"new_positive_alpha_candidates_completed":0,"new_zero_slope_candidates_completed":0,"new_FIT_TUNE_prediction_evaluations_completed":0,"reused_original_candidates":0,"reused_original_FIT_TUNE_prediction_evaluations":0,"new_selected_DEV_prediction_evaluations":0,"reused_original_DEV_prediction_evaluations":0,"resolved_selection_locks":0,"all_10_locks_before_DEV":False,"new_F_forward_calls":0,"new_F_model_objects_loaded":0,"new_native_training_updates":0,"new_E_D_fit":False,"new_optimizer_updates":0,"new_backbone_training":False,"whole_deployment_cost_measured":False,"original_v1_whole_cost_reassigned_to_extension":False,"cost_needed_for_changed_selected_models":"Not measured here; do not silently substitute into v1 76-cell cost","protected_TEST_accessed":False,"independent_holdout":False,"strict_K_only":False,"adaptive_scope":"TUNE upper-grid boundary follow-up after exposed DEV; no independent generalization claim","candidate_order":list(MENU),"zero_slope_scope":"INTERCEPT_ONLY and CURRENT_C_ANCHOR require no history/E/F to predict; not compression benefit","RAW_CURRENT_C_reference_only":{d:original["datasets"][d]["routes"]["RAW_CURRENT_C"] for d in parent.DATASETS}}
    checked,contexts={},{};protected=[]
    try:
        install_protected_test_guard();meter.protected_check();protected=prior_file_records(original,audit,cross,science,refs)
        folder=output/"sources";folder.mkdir()
        for path in (Path(__file__),Path(module.__file__),Path(parent.__file__),HERE/"masked_direct_ridge_v1.py",HERE/"PLAN_RIDGE_REGULARIZATION_EXTENSION_KO_v1.md",HERE/"fixed_decoder_span_v1.py",HERE/"decoder_metric_probe_v1.py",HERE/"continuation_runtime_v1.py",OLD/"prepare_inputs.py"):
            destination=folder/path.name;shutil.copyfile(path,destination);result["source_records"].append({**old.record(path),"snapshot_path":str(destination)})
        result.update(controller_source=old.record(__file__),module_source=old.record(module.__file__),original_controller_source=original["controller_source"],original_module_source=original["module_source"],plan=old.record(HERE/"PLAN_RIDGE_REGULARIZATION_EXTENSION_KO_v1.md"),F_parent_signatures_reused=original["F_parent_signatures_reused"])
        def load(item):return io.checked_payload({**item,"path":str(old.bound(item))},meter,checked)
        def features(data,route,split):
            require(split in ("fit","tune") or result["all_10_locks_before_DEV"],"TEN_EXTENSION_LOCKS_BEFORE_DEV_ARRAY_IO")
            if (data,split) not in contexts:
                source=science["datasets"][data]["spaces"]["PCA"]["conditions"]["L7_RECON"]["splits"][split]["input_source"];raw=load(source);c,_,dev,devn=parent.DATASETS[data];n={"fit":256,"tune":64}.get(split,devn)
                require(split in ("fit","tune",dev) and raw["history"].shape==(n,512,c) and raw["target"].shape==raw["observed"].shape==(n,48,c) and raw["history"].dtype==raw["target"].dtype==np.float32 and raw["observed"].dtype==np.bool_ and np.isfinite(raw["target"][raw["observed"]]).all() and old.byte_equal(raw["raw_last"],raw["history"][:,-1]),"EXACT_ORIGINAL_HISTORY_CURRENT_AND_OBSERVED_Y")
                contexts[data,split]=raw,source
            raw,source=contexts[data,split];owner=original["datasets"][data]["routes"][route];bindings=owner["FIT_sources"] if split=="fit" else owner["TUNE_sources"] if split=="tune" else owner["DEV"];old.bound(bindings["input_source"],source);z=None
            if route!="RAW_HISTORY512":
                name=route.removesuffix("_HISTORY512");accepted=cross["datasets"][data]["conditions"][name];old.bound(bindings["frozen_E_D_source"],accepted["state_artifact"]);old.bound(bindings["encoded_past_source"],accepted["splits"][split]["artifact"]);cache=load(bindings["encoded_past_source"]);require(old.byte_equal(cache["origins"],raw["origins"]),"ORIGINAL_ENCODED_PAST_ORIGINS");z=cache["latent_history"]
            x,anchor=parent.feature_payload(raw,route,z);return x,anchor,raw,{key:bindings.get(key) for key in ("input_source","encoded_past_source","frozen_E_D_source")}
        def scores(p,raw):
            complete=np.broadcast_to(raw["observed"].all(2,keepdims=True),raw["observed"].shape)
            return {"status":"COMPLETE","scores_all_observed":masked_scores(p,raw["target"],raw["observed"],raw["std"]),"scores_complete_mask":masked_scores(p,raw["target"],complete,raw["std"]) if complete.any() else {"status":"NO_COMPLETE_C"}}
        def run(data,route):
            prior=original["datasets"][data]["routes"][route];fitx,fita,fit,fitbindings=features(data,route,"fit");tunex,tunea,tune,tunebindings=features(data,route,"tune");response,observed=parent.response_payload(fit,route)
            row={"status":"RUNNING","route":route,"candidates":original_menu(prior),"FIT_sources":fitbindings,"TUNE_sources":tunebindings,"original_selection_lock":prior["selection_lock"],"feature_dimension":fitx.shape[1],"response_dimension":response.shape[1],"FIT_feature_binding":{"shape":list(fitx.shape),"dtype":str(fitx.dtype),"sha256":hashlib.sha256(fitx.tobytes()).hexdigest()},"FIT_response_mask_binding":{"shape":list(observed.shape),"dtype":str(observed.dtype),"sha256":hashlib.sha256(observed.tobytes()).hexdigest()}}
            result["datasets"][data]["routes"][route]=row;result["reused_original_candidates"]+=7;result["reused_original_FIT_TUNE_prediction_evaluations"]+=14
            require(row["FIT_feature_binding"]==prior["FIT_feature_binding"] and row["FIT_response_mask_binding"]==prior["FIT_response_mask_binding"],"SAME_ORIGINAL_FEATURE_AND_TRUE_OBSERVED_RESPONSE_MASK")
            result["extension_fit_attempts"]+=1;started=time.perf_counter();states,metadata=module.fit_path(fitx,response,observed,normalization="GLOBAL_FIT");row["fit_wall_seconds"]=time.perf_counter()-started;result["extension_fits_completed"]+=1
            summary,info=parent.fit_info(metadata);row.update(fit_metadata=summary,fit_artifact=io.save_npz(output/(data+"_"+route+"_EXTENSION_FIT_INFO.npz"),info));states+=zero_slope_states(fitx,response,observed)
            for index,state in enumerate(states):
                identifier=MENU[7+index];kind="RIDGE" if index<3 else identifier;candidate={"candidate_id":identifier,"kind":kind,"alpha":state["alpha"],"new_in_this_extension":True,"new_FIT_TUNE_inference":True,"status":"FAILED_CANDIDATE_PRESERVED"};candidate["coefficient_array_records"],candidate["coefficient_value_sha256"]=parent.coefficient_records(state)
                try:
                    p,t=parent.predict(fitx,state,fita,meter.check),parent.predict(tunex,state,tunea,meter.check);candidate.update(status="COMPLETE",FIT=scores(p,fit),TUNE=scores(t,tune));candidate["TUNE_MSE"]=candidate["TUNE"]["scores_all_observed"]["normalized"]["mse"];candidate["prediction_artifact"]=io.save_npz(output/(data+"_"+route+"_"+identifier+".npz"),{"FIT_prediction_FP32":p,"TUNE_prediction_FP32":t,"FIT_origins":fit["origins"],"TUNE_origins":tune["origins"]});result["new_FIT_TUNE_prediction_evaluations_completed"]+=2;result["new_positive_alpha_candidates_completed" if index<3 else "new_zero_slope_candidates_completed"]+=1
                except Exception as error:candidate["failure"]={"type":type(error).__name__,"message":str(error)}
                row["candidates"].append(candidate)
            winner=select_candidate(row["candidates"]);is_old=not winner["new_in_this_extension"]
            if is_old:
                require(winner["alpha"]==prior["selected_alpha"] and winner["coefficient_value_sha256"]==prior["selected_coefficient_value_sha256"],"RESELECTED_OLD_MENU_MUST_BE_ORIGINAL_WINNER");coefficient_source=prior["selected_coefficients"];old.bound(coefficient_source)
            else:coefficient_source=io.save_npz(output/(data+"_"+route+"_SELECTED_COEFFICIENTS.npz"),{key:states[MENU.index(winner["candidate_id"])-7][key] for key in COEFFICIENT_KEYS})
            row.update(selected_candidate=winner["candidate_id"],selected_kind=winner["kind"],selected_alpha=winner["alpha"],selected_TUNE_MSE=winner["TUNE_MSE"],selected_coefficients=coefficient_source,selected_coefficient_value_sha256=winner["coefficient_value_sha256"],selected_original_candidate=is_old,selected_new_alpha_upper_boundary=winner["alpha"]==NEW_ALPHAS[-1],selected_FIT_TUNE_prediction_artifact=winner["prediction_artifact"],FIT=winner["FIT"],TUNE=winner["TUNE"],zero_slope_selected=winner["kind"]!="RIDGE",history_features_needed_to_predict=winner["kind"]=="RIDGE",encoder_needed_to_predict=winner["kind"]=="RIDGE" and route!="RAW_HISTORY512",F_needed_to_predict=False,status="COMPLETE" if len(row["candidates"])==12 and all(p["status"]=="COMPLETE" for p in row["candidates"]) else "PARTIAL_CANDIDATE_MENU_PRESERVED")
            del states,metadata,info,fitx,tunex,response,observed;return row
        def seal(data,route,row):
            path=output/(data+"_"+route+"_SELECTION_LOCK.json");rt.write_new(path,{"status":row["status"],"dataset":data,"route":route,**{key:row.get(key) for key in ("selected_candidate","selected_kind","selected_alpha","selected_TUNE_MSE","selected_coefficients","selected_coefficient_value_sha256","selected_original_candidate")},"DEV_opened_before_lock":False,"sealed_unix":time.time()});return old.record(path)
        def registry(locks):
            path=output/"EXTENSION_SELECTION_REGISTRY.json";rt.write_new(path,{"status":"ALL_TEN_RESOLVED_BEFORE_DEV","conditions":locks,"DEV_opened_before_seal":False,"sealed_unix":time.time()});return old.record(path)
        def development(data,route,row):
            old.bound(result["selection_registry"]);old.bound(row["selection_lock"]);prior=original["datasets"][data]["routes"][route]
            if row["selected_original_candidate"]:
                old.bound(row["selected_coefficients"],prior["selected_coefficients"]);old.bound(prior["DEV"]["artifact"]);return {**prior["DEV"],"original_DEV_prediction_reused":True,"new_DEV_inference":False,"original_DEV_source":prior["DEV"]["artifact"]}
            state=load(row["selected_coefficients"]);_,digest=parent.coefficient_records(state);require(digest==row["selected_coefficient_value_sha256"],"SELECTED_EXTENSION_COEFFICIENT_VALUES_IMMUTABLE");x,anchor,raw,bindings=features(data,route,parent.DATASETS[data][2]);p=parent.predict(x,state,anchor,meter.check);entry=scores(p,raw);entry.update(artifact=io.save_npz(output/(data+"_"+route+"_SELECTED_DEV.npz"),{"prediction_FP32":p,"origins":raw["origins"]}),**bindings,selected_coefficients=row["selected_coefficients"],original_DEV_prediction_reused=False,new_DEV_inference=True);return entry
        with threadpool_limits(limits=2):resolved=resolve_paths(result,run,seal,registry);infer_development(result,resolved,development)
        for item in protected+list(checked.values())+result["source_records"]:old.bound(item)
        for item in result["source_records"]:require(old.record(item["snapshot_path"])["sha256"]==item["sha256"],"EXECUTED_SOURCE_SNAPSHOTS_UNCHANGED")
        result.update(input_records=list(checked.values()),protected_prior_records=protected,status="COMPLETE" if len(resolved)==10 and result["extension_fits_completed"]==10 and result["new_positive_alpha_candidates_completed"]==30 and result["new_zero_slope_candidates_completed"]==20 and result["new_FIT_TUNE_prediction_evaluations_completed"]==100 and result["reused_original_candidates"]==70 and result["new_selected_DEV_prediction_evaluations"]+result["reused_original_DEV_prediction_evaluations"]==10 and not result["failed_paths"] else "PARTIAL_COMPLETED");meter.protected_check();json.dumps(result,allow_nan=False)
    except Exception as error:failure={"type":type(error).__name__,"message":str(error)};result.update(status="FAILED_PRESERVED",failure=failure)
    finally:result["resource_terminal"]=meter.close(result["status"],failure);rt.write_new(output/"RESULT.json",result)
    print(json.dumps({"status":result["status"],"result":str(output/"RESULT.json")}),flush=True);return 0 if result["status"]=="COMPLETE" else 2


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("tag","ridge-result","ridge-audit"):parser.add_argument("--"+key,required=True)
    raise SystemExit(execute(parser.parse_args()))
