"""Decompose saved selected hD forecast changes; no fit, F, model or GPU."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import shutil
import sys

import audit_new_direction_direct_ridge_v1 as common
import run_linear_e_adaptation_v1 as old

base,prior,HERE,ROOT,require=common.base,common.prior,common.HERE,common.ROOT,common.require
DATASETS={"robin":(17,4,"val",30),"beijing_aotizhongxin":(11,2,"eval",146)}
SPLIT_N={"fit":256,"tune":64}
HORIZON_RESULT_SHA="5f158382640dac74ec63cc3d1005b385c898821d3c032d51b543f2c9b27d7dac"
HORIZON_AUDIT_SHA="0f65053536403a3b7b8f53fca3b9f8274fb1a406eaf6dfdcb73bb0467fffe698"
HORIZON_CONTROLLER_SHA="151da5b0b4cc4cacdb9d7a547d12e7837dcdcb5ee4c5047a1d6e3993afe525c6"
HORIZON_AUDITOR_SHA="accb6fb775d9b12ca6e39c0be7e062545b98d429089ee32e0ac46ed65edae613"
MODULE_SHA="806a66415834394d1b5efb07bc95fdbd7d103e6167c15abaf7be03f64026a187"
SCORE_ATOL,SCORE_RTOL=1e-12,1e-12


def load_gates(args):
    require(all(isinstance(value,str) and len(value)==64 for value in (HORIZON_RESULT_SHA,HORIZON_AUDIT_SHA,HORIZON_CONTROLLER_SHA,HORIZON_AUDITOR_SHA,MODULE_SHA)),"ERROR_TERMS_INPUT_GATES_NOT_REGISTERED")
    refs={"horizon_diagnostic":old.record(args.horizon_result),"horizon_audit":old.record(args.horizon_audit)}
    require(refs["horizon_diagnostic"]["sha256"]==HORIZON_RESULT_SHA and refs["horizon_audit"]["sha256"]==HORIZON_AUDIT_SHA,"EXACT_ACCEPTED_HORIZON_RESULT_AND_AUDIT")
    result=prior.verified_json(refs["horizon_diagnostic"],lambda:None);audit=prior.verified_json(refs["horizon_audit"],lambda:None);common.same(audit["diagnostic"],refs["horizon_diagnostic"])
    require(result["status"]=="COMPLETE" and not result["failed_paths"] and audit["status"]=="PASS" and not audit["partial_paths"] and result["controller_source"]["sha256"]==HORIZON_CONTROLLER_SHA and audit["audit_source"]["sha256"]==HORIZON_AUDITOR_SHA,"COMPLETE_ACCEPTED_HORIZON_SOURCES")
    for key,count in (("fits_completed",24),("candidate_FIT_TUNE_evaluations_completed",528),("selected_DEV_evaluations_completed",24),("resolved_selection_locks",24),("new_F_forward_calls",0)):
        require(result[key]==count,"HORIZON_COMPLETED_COUNT:"+key)
    require(result["all_24_locks_before_DEV"] is True and result["protected_TEST_accessed"] is False and audit["protected_TEST_accessed"] is False and audit["totals"]["decoder_fit_paths_checked"]==24 and audit["totals"]["selected_FP32_arithmetic_predictions_checked"]==72,"ACCEPTED_SAVED_ARITHMETIC_AND_LOCK_SCOPE")
    for key in ("controller_source","module_source","plan","selection_registry","cross_moment_diagnostic","cross_moment_audit","causal_lag_diagnostic"):common.same(audit[key],result[key]);old.bound(result[key])
    for record in (result["controller_source"],audit["audit_source"],{"path":str(HERE/"horizon_correction_error_terms_v1.py"),"sha256":MODULE_SHA}):old.bound(record)
    return result,audit,refs


def array_record(value):
    import numpy as np
    return {"shape":list(value.shape),"dtype":str(value.dtype),"sha256":hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}


def score_link(cell,source,field):
    if cell["status"]=="NO_OBSERVED_TARGETS":
        require(source.get("status")=="NO_OBSERVED_TARGETS","SOURCE_NO_OBSERVED_SCORE");return {"status":"NO_OBSERVED_TARGETS","recorded_MSE":None,"MSE_discrepancy":None,"max_per_channel_MSE_abs_discrepancy":None}
    require(source["observed_counts"]==cell["observed_counts"] and source["active_channels"]==cell["active_channels"],"SOURCE_SCORE_COUNTS")
    require(math.isclose(cell[field],source["mse"],abs_tol=SCORE_ATOL,rel_tol=SCORE_RTOL),"SOURCE_SCORE_MSE")
    discrepancies=[]
    for j,row in enumerate(cell["per_channel"]):
        if row["observed_count"]:
            require(math.isclose(row[field],source["per_channel_mse"][j],abs_tol=SCORE_ATOL,rel_tol=SCORE_RTOL),"SOURCE_SCORE_PER_CHANNEL_MSE");discrepancies.append(abs(row[field]-source["per_channel_mse"][j]))
    return {"status":"COMPLETE","recorded_MSE":source["mse"],"MSE_discrepancy":cell[field]-source["mse"],"max_per_channel_MSE_abs_discrepancy":max(discrepancies),"atol":SCORE_ATOL,"rtol":SCORE_RTOL}


def analyze(result,load,read):
    import numpy as np
    import horizon_correction_error_terms_v1 as module
    registry=read(result["selection_registry"]);require(registry["status"]=="ALL_24_RESOLVED_BEFORE_DEV" and registry["DEV_opened_before_seal"] is False and len(registry["conditions"])==24,"ORIGINAL_24_LOCK_REGISTRY")
    expected={(data,space,mode,arm) for data,(_,k,_,_) in DATASETS.items() for space in ("RAW_C","PCA_K"+str(k),"PCA_K"+str(2*k)) for mode in ("F","FLAT") for arm in ("STATIC_DELTA","RAMP_DELTA")};seen=set();raw_cache={};contexts={}
    totals={key:0 for key in ("conditions_checked","full_cells_completed","horizon_cells_completed","interval_cells_completed","per_channel_cells_completed","source_contexts_checked","selection_locks_preserved","ZERO_split_cells_checked","observed_scalar_score_links_checked")}
    report={"status":"COMPLETE","datasets":{},"totals":totals,"new_fits":0,"new_F_forward_calls":0,"new_model_objects_loaded":0,"new_optimizer_updates":0,"new_selection_decisions":0,"new_DEV_winners_selected":0,"whole_deployment_cost_measured":False,"protected_TEST_accessed":False,"independent_holdout":False,"scope":"Diagnostic recalculation of saved, previously selected FP32 endpoints and observed normalized-C errors. Source arithmetic and existing helpers are reused; this is not an external audit, new selection, optimizer/F replay or a unique causal explanation.","metric":"mean across active channels of each channel's own true-observed mean","correction":"prediction_FP32 cast to FP64 minus original baseline_FP32 cast to FP64","unobserved_channels_and_intervals":"null, never zero-filled","intervals":list(module.INTERVALS),"maximum_closure_abs":0.}
    def context(data,space,split):
        key=data,space,split
        if key in contexts:return contexts[key]
        metadata=result["datasets"][data]["spaces"][space]["contexts"][split];source=metadata["input_source"];c,k,dev,dn=DATASETS[data]
        if (data,split) not in raw_cache:
            raw=load(source);n=SPLIT_N.get(split,dn);require(raw["target"].shape==raw["observed"].shape==(n,48,c) and raw["target"].dtype==np.float32 and raw["observed"].dtype==np.bool_ and raw["origins"].shape==(n,) and raw["std"].shape==(c,) and np.isfinite(raw["std"]).all() and (raw["std"]>0).all(),"ORIGINAL_TRUE_OBSERVED_TARGET_ORIGINS_AND_SCALER");raw_cache[data,split]=(raw,source)
        raw,raw_source=raw_cache[data,split];common.same(source,raw_source);cache=None
        if space!="RAW_C":
            require(metadata["cached_latent_source"] is not None and metadata["frozen_state_source"] is not None,"SAME_PCA_STATE_AND_CACHE_REQUIRED");cache=load(metadata["cached_latent_source"]);base.exact(cache["origins"],raw["origins"],"fixed_PCA_source_origins")
        else:require(metadata["cached_latent_source"] is metadata["frozen_state_source"] is None,"RAW_ORIGINAL_BASELINE_SOURCE")
        baselines={}
        for mode in ("F","FLAT"):
            baseline=(raw["raw_forecast"] if mode=="F" else np.broadcast_to(raw["raw_last"][:,None],raw["target"].shape).copy()) if cache is None else cache["prediction" if mode=="F" else "FLAT_prediction"]
            require(array_record(baseline)==metadata["array_bindings"][mode]["baseline"],"EXACT_PARENT_BASELINE_ARRAY_BINDING");baselines[mode]=baseline
        contexts[key]=(raw,baselines,metadata);totals["source_contexts_checked"]+=1;return contexts[key]
    for item in registry["conditions"]:
        data,space,mode,arm=(item[key] for key in ("dataset","space","mode","arm"));key=data,space,mode,arm;require(key in expected and key not in seen and item["status"]=="COMPLETE","UNIQUE_ORIGINAL_CONDITION");seen.add(key);name=mode+"_"+arm;row=result["datasets"][data]["spaces"][space]["conditions"][name];lock=read(row["selection_lock"]);common.same(item["selection_lock"],row["selection_lock"])
        require(row["status"]==lock["status"]=="COMPLETE" and lock["DEV_opened_before_lock"] is False and lock["sealed_unix"]<=registry["sealed_unix"] and all(lock[field]==row[field] for field in ("selected_candidate","selected_kind","selected_alpha","selected_TUNE_MSE","selected_coefficients","selected_coefficient_value_sha256")),"ORIGINAL_SELECTED_LOCK_AND_COEFFICIENT_BINDINGS")
        require(all(lock[field]==value for field,value in zip(("dataset","space","mode","arm"),key)) and (row["space"],row["mode"],row["arm"])==(space,mode,arm),"ORIGINAL_SELECTED_CONDITION_LABELS");saved=load(row["selected_FIT_TUNE_prediction_artifact"])
        output={field:row[field] for field in ("selected_candidate","selected_kind","selected_alpha","selected_TUNE_MSE","selected_coefficients","selected_coefficient_value_sha256","selection_lock","K")}
        for label,split in (("FIT","fit"),("TUNE","tune"),("DEV",DATASETS[data][2])):
            raw,baselines,metadata=context(data,space,split);baseline=baselines[mode];entry=row[label]
            if label=="DEV":
                require(entry["sources"]==metadata,"ORIGINAL_DEV_CONTEXT");common.same(entry["selected_coefficients"],row["selected_coefficients"]);record=entry["artifact"];values=load(record);prediction=values["prediction_FP32"];origins=values["origins"];member="prediction_FP32";selected_temporal=entry["temporal_diagnostics"]
            else:
                require(row[label+"_sources"]==metadata,"ORIGINAL_FIT_TUNE_CONTEXT");record=row["selected_FIT_TUNE_prediction_artifact"];member=label+"_prediction_FP32";prediction=saved[member];origins=saved[label+"_origins"];selected_temporal=row["selected_temporal_diagnostics"][label]
            base.exact(origins,raw["origins"],"selected_"+label+"_origins");cell=module.diagnose(baseline,prediction,raw["target"],raw["observed"],require_zero=row["selected_kind"]=="ZERO_DELTA");baseline_temporal=metadata["baseline_temporal_diagnostics"][mode]
            pairs=[(cell["full"],metadata["baseline_scores"][mode]["scores_all_observed"]["normalized"],entry["scores_all_observed"]["normalized"])]+[(a,b,c) for a,b,c in zip(cell["per_horizon"],baseline_temporal["per_horizon"],selected_temporal["per_horizon"])]+[(cell["registered_intervals"][key],baseline_temporal["registered_intervals"][key],selected_temporal["registered_intervals"][key]) for key,_,_ in module.INTERVALS]
            require(len(selected_temporal["per_horizon"])==len(baseline_temporal["per_horizon"])==48,"ALL_ORIGINAL_H48_SOURCE_SCORES")
            for value,base_score,new_score in pairs:
                value["source_score_links"]={"baseline":score_link(value,base_score,"baseline_MSE"),"selected":score_link(value,new_score,"selected_MSE")};totals["observed_scalar_score_links_checked"]+=2*(1+value["active_channels"]) if value["status"]=="COMPLETE" else 0;totals["per_channel_cells_completed"]+=len(value["per_channel"])
                for record_value in [value,*value["per_channel"]]:
                    if record_value["status"]=="COMPLETE":report["maximum_closure_abs"]=max(report["maximum_closure_abs"],*(abs(record_value[field]) for field in ("closure_residual","direct_closure_residual","delta_minus_direct_change")))
            cell["source_bindings"]={"input_source":metadata["input_source"],"cached_latent_source":metadata["cached_latent_source"],"frozen_state_source":metadata["frozen_state_source"],"selected_prediction_artifact":record,"selected_prediction_member":member,"original_baseline_array":array_record(baseline),"target_mask_origins_scaler_arrays":{key:array_record(raw[key]) for key in ("target","observed","origins","mean","std")},"original_temporal_score_sources_reused":True};output[label]=cell;totals["full_cells_completed"]+=1;totals["horizon_cells_completed"]+=48;totals["interval_cells_completed"]+=3;totals["ZERO_split_cells_checked"]+=int(row["selected_kind"]=="ZERO_DELTA")
        tune,dev=output["TUNE"]["full"],output["DEV"]["full"];output["TUNE_DEV_observation"]={"TUNE_delta_negative_DEV_delta_positive":tune["delta_MSE"]<0<dev["delta_MSE"],"DEV_minus_TUNE_interaction_term":dev["interaction_term"]-tune["interaction_term"],"DEV_minus_TUNE_correction_energy":dev["correction_energy"]-tune["correction_energy"],"DEV_minus_TUNE_delta_MSE":dev["delta_MSE"]-tune["delta_MSE"],"same_split_masks_used_within_each_decomposition":True,"causal_distribution_shift_or_overfit_proven":False}
        report["datasets"].setdefault(data,{"spaces":{}})["spaces"].setdefault(space,{"conditions":{}})["conditions"][name]=output;totals["conditions_checked"]+=1;totals["selection_locks_preserved"]+=1
    require(seen==expected and totals["full_cells_completed"]==72 and totals["horizon_cells_completed"]==3456 and totals["interval_cells_completed"]==216 and totals["source_contexts_checked"]==18,"FULL_REGISTERED_ERROR_TERMS_COVERAGE");return report


def receipt_records(receipt,status,check):
    require(receipt["status"]==status and receipt["gpu_process"] is False and all(type(receipt.get(key)) in (int,float) and math.isfinite(receipt[key]) and receipt[key]>=0 for key in ("process_cpu_seconds","wall_seconds")),"CANONICAL_PARENT_CPU_RECEIPT")
    binding={key:receipt[key] for key in ("path","sha256")};recorded=prior.verified_json(binding,check);require(recorded=={key:value for key,value in receipt.items() if key not in ("path","sha256")},"EXACT_PARENT_RECEIPT_FILE")
    authorization=prior.verified_json(receipt["authorization"],check);manifest=prior.verified_json(receipt["continuation_manifest"],check)
    require(authorization["status"]=="APPROVED_HUMAN_UNCAPPED_CONTINUATION" and manifest["authorization_sha256"]==receipt["authorization"]["sha256"] and manifest["runtime_sha256"]==prior.file_sha(HERE/"continuation_runtime_v1.py"),"PARENT_RECEIPT_AUTHORIZATION_AND_RUNTIME_CHAIN");return [binding,receipt["authorization"],receipt["continuation_manifest"]]


def execute(args):
    result,audit,refs=load_gates(args);output=old.fresh_output(args.tag)
    import horizon_correction_error_terms_v1 as module
    import continuation_runtime_v1 as rt
    common.initialize_numpy();sys.path.insert(0,str(ROOT/"research/smallk_failure_followup_20261014"))
    from prepare_inputs import install_protected_test_guard
    output.mkdir(parents=True,exist_ok=False);meter=rt.Meter("horizon_error_terms_"+args.tag,gpu=False);failure=None;loaded=[];documents=[];protected=[];sources=[];report={"status":"RUNNING",**refs}
    try:
        install_protected_test_guard();meter.protected_check();protected=[*refs.values(),audit["audit_source"],*result["source_records"],*result["input_records"],*result["protected_prior_records"],*receipt_records(result["resource_terminal"],"COMPLETE",meter.check),*receipt_records(audit["resource_terminal"],"PASS",meter.check)]
        for key in ("controller_source","module_source","plan","selection_registry","cross_moment_diagnostic","cross_moment_audit","causal_lag_diagnostic"):protected.append(result[key])
        for record in protected:
            old.bound(record)
            if "snapshot_path" in record:require(old.record(record["snapshot_path"])["sha256"]==record["sha256"],"PARENT_SOURCE_SNAPSHOT_UNCHANGED")
        folder=output/"sources";folder.mkdir()
        for path in (Path(__file__),Path(module.__file__),HERE/"PLAN_HORIZON_CORRECTION_ERROR_TERMS_KO_v1.md",HERE/"audit_new_direction_direct_ridge_v1.py",HERE/"audit_linear_e_adaptation_v1.py",HERE/"audit_projection_results_v1.py",HERE/"run_linear_e_adaptation_v1.py",HERE/"continuation_runtime_v1.py"):
            destination=folder/path.name;shutil.copyfile(path,destination);sources.append({**old.record(path),"snapshot_path":str(destination)})
        def load(record):old.bound(record);loaded.append(record);return prior.verified_arrays(record,meter.check)
        def read(record):old.bound(record);documents.append(record);return prior.verified_json(record,meter.check)
        report={**analyze(result,load,read),**refs,"cross_moment_diagnostic":result["cross_moment_diagnostic"],"cross_moment_audit":result["cross_moment_audit"],"original_selection_registry":result["selection_registry"],"controller_source":old.record(__file__),"module_source":old.record(module.__file__),"plan":old.record(HERE/"PLAN_HORIZON_CORRECTION_ERROR_TERMS_KO_v1.md"),"parent_resource_receipts":{"horizon_diagnostic":result["resource_terminal"],"horizon_audit":audit["resource_terminal"]}}
        for row in report["datasets"].values():
            for space in row["spaces"].values():
                for condition in space["conditions"].values():protected.append(condition["selected_coefficients"])
        for record in [*protected,*loaded,*documents,*sources]:old.bound(record)
        for record in sources:require(old.record(record["snapshot_path"])["sha256"]==record["sha256"],"NEW_SOURCE_SNAPSHOT_UNCHANGED")
        report.update(input_records=list({(record["path"],record["sha256"]):record for record in loaded}.values()),metadata_records=documents,source_records=sources,protected_prior_records=protected);report["totals"]["protected_source_input_result_hashes_checked"]=len({(record["path"],record["sha256"]) for record in [*protected,*loaded,*documents,*sources]});meter.protected_check();json.dumps(report,allow_nan=False)
    except Exception as error:failure={"type":type(error).__name__,"message":str(error)};report.update(status="FAILED_PRESERVED",failure=failure,source_records=sources,input_records=loaded,metadata_records=documents,protected_prior_records=protected)
    finally:report["resource_terminal"]=meter.close(report["status"],failure);rt.write_new(output/"RESULT.json",report)
    print(json.dumps({"status":report["status"],"result":str(output/"RESULT.json"),"totals":report.get("totals"),"failure":failure}),flush=True);return 0 if report["status"]=="COMPLETE" else 2


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("tag","horizon-result","horizon-audit"):parser.add_argument("--"+name,required=True)
    raise SystemExit(execute(parser.parse_args()))
