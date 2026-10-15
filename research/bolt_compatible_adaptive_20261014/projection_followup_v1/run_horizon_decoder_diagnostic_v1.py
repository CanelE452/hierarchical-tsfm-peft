"""Cached-F equal-slope static/ramp residual decoder diagnostic; no model or GPU."""
from pathlib import Path
import argparse
import gc
import json
import math
import shutil
import sys
import time

import run_new_direction_direct_ridge_v1 as parent

old,HERE,ROOT,BASE,OLD,require=parent.old,parent.HERE,parent.ROOT,parent.BASE,parent.OLD,parent.require
DATASETS=parent.DATASETS
LAGS=(0,1,2,4,8,16,32)
ALPHAS=(0.,1e-4,.001,.01,.1,1.,10.,100.,1000.,10000.)
ARMS=("STATIC_DELTA","RAMP_DELTA");MODES=("F","FLAT")
MODULE_SHA="406252e940ae676e7a78f69e067f5b3fa70456bdb1ea27d57169b8964ceabe0e"
PARENT_READER_SHA="7dd19e1504f0ff4b323c52412fac492f2b5c17c23e1ba0d898691a672f5ac88e"
CONDITIONS=tuple((data,space,mode,arm) for data,(_,k,_,_) in DATASETS.items() for space in ("RAW_C","PCA_K"+str(k),"PCA_K"+str(2*k)) for mode in MODES for arm in ARMS)


def load_gates(args):
    require(all(isinstance(s,str) and len(s)==64 for s in (MODULE_SHA,PARENT_READER_SHA)),"HORIZON_SOURCE_GATES_NOT_REGISTERED")
    old.bound({"path":str(HERE/"horizon_decoder_delta_v1.py"),"sha256":MODULE_SHA});old.bound({"path":str(HERE/"run_new_direction_direct_ridge_v1.py"),"sha256":PARENT_READER_SHA})
    cross,science,refs=parent.load_gates(args);refs["parent_metadata_reader"]=old.record(parent.__file__);return cross,science,refs


def stream_payload(history,current,raw_forecast,space,mode,cache=None,state=None):
    import numpy as np
    from causal_lag_decoder_v1 import lagged_delta_features,predict
    require(history.dtype==current.dtype==raw_forecast.dtype==np.float32 and history.shape[1]==512 and raw_forecast.shape==(len(history),48,history.shape[2]) and mode in MODES,"FINITE_FIXED_H48_FP32_SOURCE_STREAM")
    require(old.byte_equal(current,history[:,-1]) and all(np.isfinite(v).all() for v in (history,current,raw_forecast)),"PAST_CURRENT_AND_RAW_FORECAST_SOURCE")
    if space=="RAW_C":
        z=history;future=raw_forecast if mode=="F" else np.broadcast_to(current[:,None],raw_forecast.shape).copy();baseline=raw_forecast.copy() if mode=="F" else np.broadcast_to(current[:,None],raw_forecast.shape).copy()
        e=np.eye(history.shape[2],dtype=np.float32);d=np.concatenate([e,*[np.zeros_like(e) for _ in LAGS[1:]]]);state={"E":e,"D":d,"center":np.zeros(history.shape[2],np.float32)}
    else:
        require(cache is not None and state is not None and space.startswith("PCA_K"),"FIXED_PCA_CACHE_STATE_REQUIRED");z=cache["latent_history"]
        require(z.shape[:2]==history.shape[:2] and z.dtype==np.float32 and z.shape[2]==state["E"].shape[1] and state["D"].shape==(7*z.shape[2],history.shape[2]),"SAME_PCA_K_AND_OLD_L7_D")
        future=cache["latent_future"] if mode=="F" else np.broadcast_to(z[:,-1:],cache["latent_future"].shape).copy();baseline=cache["prediction"] if mode=="F" else cache["FLAT_prediction"]
    features=lagged_delta_features(z,future,LAGS);checks={"original_encode":old.parity(z,(history-state["center"])@state["E"]),"same_old_decoder":{**old.parity(baseline,predict(z,future,current,state["D"],LAGS)),"required":space!="RAW_C","parity_not_required":space=="RAW_C"}}
    if space=="RAW_C":require(old.byte_equal(baseline,raw_forecast if mode=="F" else np.broadcast_to(current[:,None],raw_forecast.shape).copy()),"RAW_BASELINE_ORIGINAL_BYTES_PRESERVED")
    if space!="RAW_C" and mode=="F":checks["original_lag_features"]=old.parity(features,cache["lag_features"])
    require(all(item["pass"] for item in checks.values() if item.get("required",True)),"ORIGINAL_FIXED_BASELINE_AND_LAG_PARITY")
    return {"features":features,"baseline":baseline,"latent_history":z,"latent_future":future,"checks":checks}


def fit_response(target,observed,baseline):
    import numpy as np
    require(target.dtype==baseline.dtype==np.float32 and observed.dtype==np.bool_ and target.shape==observed.shape==baseline.shape and np.isfinite(target[observed]).all(),"OBSERVED_FP32_TARGET_AND_BASELINE")
    response=np.zeros(target.shape,np.float64);response[observed]=target[observed].astype(np.float64)-baseline[observed].astype(np.float64);return response


def fit_info(metadata):
    import numpy as np
    arrays={k:metadata[k] for k in ("scale_FP64","constant_features")};summary={k:v for k,v in metadata.items() if k not in (*arrays,"groups")};summary["groups"]=[]
    for index,group in enumerate(metadata["groups"]):
        names={key:"group_"+str(index)+"_"+key for key in ("observed_mask","singularvalues")}
        for key,name in names.items():arrays[name]=group[key]
        arrays["group_"+str(index)+"_response_columns"]=np.asarray(group["response_columns"],dtype=np.int64);summary["groups"].append({**{k:v for k,v in group.items() if k not in names},"array_members":names})
    summary["group_count"]=len(summary["groups"]);return summary,arrays


def predict_candidate(baseline,features,state,arm,kind):
    import numpy as np
    import horizon_decoder_delta_v1 as module
    require(kind in ("ALPHA","ZERO_DELTA"),"REGISTERED_CORRECTION_CANDIDATE")
    if kind=="ZERO_DELTA":return baseline.copy(),{"pass":True,"max_abs":0.,"ZERO_original_baseline_byte_exact":True}
    prediction=module.predict(baseline,features,state,arm);gate=np.ones(48,np.float32) if arm=="STATIC_DELTA" else np.arange(48,dtype=np.float32)/np.float32(47)
    expected=baseline+(features*gate[None,:,None])@state["coefficient_FP32"];check=old.parity(prediction,expected);require(check["pass"],"ACTUAL_FP32_CORRECTION_NUMPY_PARITY");return prediction,check


def select_candidate(candidates):
    order=["ALPHA_"+format(a,"g") for a in ALPHAS]+["ZERO_DELTA"];valid=[p for p in candidates if p["status"]=="COMPLETE" and math.isfinite(p.get("TUNE_MSE",float("nan")))]
    require(bool(valid),"FINITE_TUNE_CANDIDATE_REQUIRED");return min(valid,key=lambda p:(p["TUNE_MSE"],order.index(p["id"])))


def latent_errors(streams,target,observed,state):
    import numpy as np
    complete=observed.all(2);truth=np.zeros((*target.shape[:2],state["E"].shape[1]),np.float32)
    truth[complete]=(target[complete]-state["center"])@state["E"];count=int(complete.sum());result={"complete_C_horizons":count,"target_horizon_mask_only_for_latent_point_error":True,"incomplete_C_vectors_encoded":False,"C_lag_output_EY_or_floor_status":"NOT_MEASURED_IN_THIS_STAGE","unit_scope":"fixed encoded normalized-C coordinates, not comparable to C macro MSE"}
    for mode in MODES:
        if not count:result[mode]={"status":"NO_COMPLETE_C"};continue
        error=streams[mode]["latent_future"].astype(np.float64)-truth.astype(np.float64);values=error[complete];per_h=[]
        for h in range(48):per_h.append(float(np.mean(error[:,h][complete[:,h]]**2)) if complete[:,h].any() else None)
        result[mode]={"mse":float(np.mean(values**2)),"mae":float(np.mean(np.abs(values))),"per_latent_mse":np.mean(values**2,axis=0).tolist(),"per_horizon_mse":per_h,"per_horizon_complete_C_counts":complete.sum(0).tolist()}
    return result


def temporal_scores(prediction,raw):
    from decoder_metric_probe_v1 import masked_scores
    def score(start,stop):
        mask=raw["observed"][:,start:stop]
        return masked_scores(prediction[:,start:stop],raw["target"][:,start:stop],mask,raw["std"])["normalized"] if mask.any() else {"status":"NO_OBSERVED_TARGETS"}
    return {"per_horizon":[score(h,h+1) for h in range(48)],"registered_intervals":{"1_16":score(0,16),"17_32":score(16,32),"33_48":score(32,48)},"horizon_averages_not_equal_full_observed_channel_macro":True}


def resolve_paths(result,run,seal,registry):
    resolved={};locks=[]
    for data,space,mode,arm in CONDITIONS:
        name=mode+"_"+arm;row={}
        try:row=run(data,space,mode,arm);require(row["status"]=="COMPLETE","ALL_ELEVEN_CANDIDATES_REQUIRED")
        except Exception as error:row=result["datasets"][data]["spaces"][space]["conditions"].get(name,row);row.update(status="FAILED_PATH_PRESERVED",failure={"type":type(error).__name__,"message":str(error)})
        row["selection_lock"]=seal(data,space,mode,arm,row);result["datasets"][data]["spaces"][space]["conditions"][name]=row;locks.append({"dataset":data,"space":space,"mode":mode,"arm":arm,"status":row["status"],"selection_lock":row["selection_lock"]})
        if row["status"]=="COMPLETE":resolved[data,space,mode,arm]=row
        else:result["failed_paths"].append({"dataset":data,"space":space,"condition":name,"stage":"FIT_TUNE_AND_SELECTION","failure":row.get("failure")})
        gc.collect()
    result["selection_registry"]=registry(locks);result.update(resolved_selection_locks=len(locks),all_24_locks_before_DEV=True);return resolved


def infer_development(result,resolved,infer):
    require(result["all_24_locks_before_DEV"] and result["resolved_selection_locks"]==24,"ALL_24_LOCKS_BEFORE_DEV")
    for (data,space,mode,arm),row in resolved.items():
        try:row["DEV"]=infer(data,space,mode,arm,row);require(row["DEV"]["status"]=="COMPLETE","FINITE_SELECTED_DEV_REQUIRED");result["selected_DEV_evaluations_completed"]+=1
        except Exception as error:result["failed_paths"].append({"dataset":data,"space":space,"condition":mode+"_"+arm,"stage":"SELECTED_DEV","failure":{"type":type(error).__name__,"message":str(error)}})


def execute(args):
    cross,science,refs=load_gates(args);output=old.fresh_output(args.tag);sys.path[:0]=[str(HERE),str(BASE),str(OLD),str(ROOT)]
    import numpy as np
    import horizon_decoder_delta_v1 as module
    import fixed_decoder_span_v1 as io
    import continuation_runtime_v1 as rt
    from decoder_metric_probe_v1 import masked_scores
    from run_fit_target_comparison_v1 import prior_file_records
    from prepare_inputs import install_protected_test_guard
    from threadpoolctl import threadpool_limits
    require(old.record(module.__file__)["sha256"]==MODULE_SHA and tuple(module.ALPHAS)==ALPHAS and tuple(module.ARMS)==ARMS,"FROZEN_DECODER_DELTA_MODULE_GRID_ARMS")
    output.mkdir(parents=True,exist_ok=False);meter=rt.Meter("horizon_decoder_"+args.tag,gpu=False);failure=None
    result={"status":"RUNNING",**refs,"datasets":{data:{"spaces":{space:{"conditions":{},"contexts":{}} for space in ("RAW_C","PCA_K"+str(k),"PCA_K"+str(2*k))}} for data,(_,k,_,_) in DATASETS.items()},"failed_paths":[],"source_records":[],"fit_attempts":0,"fits_completed":0,"alpha_candidates_completed":0,"ZERO_candidates_completed":0,"candidate_FIT_TUNE_evaluations_completed":0,"selected_DEV_evaluations_completed":0,"resolved_selection_locks":0,"all_24_locks_before_DEV":False,"new_F_forward_calls":0,"new_F_model_objects_loaded":0,"new_optimizer_updates":0,"new_E_D_training_updates":0,"new_backbone_training":False,"whole_deployment_cost_measured":False,"protected_TEST_accessed":False,"independent_holdout":False,"strict_K_only":False,"per_origin_fit":False,"full_horizon_independent_decoders_fitted":False,"same_capacity_PCA_and_RAW_claimed":False,"actual_F_ungated_FIT_RMS_shared_by_four_arms":True,"output_space_minimum_status":"NOT_MEASURED_IN_THIS_STAGE","source_F_registered_calls_reused_metadata":104,"F_parent_signatures_reused":{"before":cross["F_before"],"after":cross["F_after"]},"resource_scope":"CPU cached-source delta fits/all candidates/selected forecasts/diagnostics/hash/save; no live F/GPU or whole-cost timing"}
    checked,raw_cache,contexts,scales={}, {}, {}, {};protected=[]
    try:
        install_protected_test_guard();meter.protected_check();protected=prior_file_records(cross,science,refs);folder=output/"sources";folder.mkdir()
        for path in (Path(__file__),HERE/"horizon_decoder_delta_v1.py",HERE/"PLAN_HORIZON_DECODER_DIAGNOSTIC_KO_v1.md",HERE/"run_new_direction_direct_ridge_v1.py",HERE/"causal_lag_decoder_v1.py",HERE/"fixed_decoder_span_v1.py",HERE/"decoder_metric_probe_v1.py",HERE/"continuation_runtime_v1.py",OLD/"prepare_inputs.py"):
            destination=folder/path.name;shutil.copyfile(path,destination);result["source_records"].append({**old.record(path),"snapshot_path":str(destination)})
        result.update(controller_source=old.record(__file__),module_source=old.record(module.__file__),plan=old.record(HERE/"PLAN_HORIZON_DECODER_DIAGNOSTIC_KO_v1.md"))
        def load(record):return io.checked_payload({**record,"path":str(old.bound(record))},meter,checked)
        def scores(prediction,raw):
            complete=np.broadcast_to(raw["observed"].all(2,keepdims=True),raw["observed"].shape)
            return {"status":"COMPLETE","scores_all_observed":masked_scores(prediction,raw["target"],raw["observed"],raw["std"]),"scores_complete_mask":masked_scores(prediction,raw["target"],complete,raw["std"]) if complete.any() else {"status":"NO_COMPLETE_C"}}
        def context(data,space,split):
            require(split in ("fit","tune") or result["all_24_locks_before_DEV"],"ALL_24_LOCKS_BEFORE_DEV_ARRAY_IO")
            if (data,space,split) in contexts:return contexts[data,space,split]
            if (data,split) not in raw_cache:
                source=science["datasets"][data]["spaces"]["PCA"]["conditions"]["L7_RECON"]["splits"][split]["input_source"];raw=load(source);c,_,dev,dn=DATASETS[data];n={"fit":256,"tune":64,dev:dn}[split]
                require(raw["history"].shape==(n,512,c) and raw["target"].shape==raw["observed"].shape==(n,48,c) and raw["history"].dtype==raw["target"].dtype==np.float32 and raw["observed"].dtype==np.bool_ and np.isfinite(raw["target"][raw["observed"]]).all(),"ORIGINAL_DATASET_COHORT_AND_TRUE_OBSERVATIONS");raw_cache[data,split]=(raw,source)
            raw,source=raw_cache[data,split];saved=state=None;cached_source=state_source=None
            if space!="RAW_C":
                owner=cross["datasets"][data]["conditions"][space];entry=owner["splits"][split];old.bound(entry["input_source"],source);cached_source=entry["artifact"];state_source=owner["state_artifact"];saved=load(cached_source);state=load(state_source);require(old.byte_equal(saved["origins"],raw["origins"]),"EXACT_PCA_CACHE_ORIGINS")
            streams={mode:stream_payload(raw["history"],raw["raw_last"],raw["raw_forecast"],space,mode,saved,state) for mode in MODES}
            if state is None:
                c=raw["history"].shape[2];eye=np.eye(c,dtype=np.float32);state={"E":eye,"D":np.concatenate([eye,*[np.zeros_like(eye) for _ in LAGS[1:]]]),"center":np.zeros(c,np.float32)}
            metadata={"input_source":source,"cached_latent_source":cached_source,"frozen_state_source":state_source,"RAW_identity_E_lag0_identity_D":space=="RAW_C","K":state["E"].shape[1],"checks":{mode:streams[mode]["checks"] for mode in MODES},"latent_error_diagnostics":latent_errors(streams,raw["target"],raw["observed"],state),"baseline_scores":{mode:scores(streams[mode]["baseline"],raw) for mode in MODES},"current_C_persistence_scores":scores(np.broadcast_to(raw["raw_last"][:,None],raw["target"].shape),raw)}
            metadata["array_bindings"]={mode:{key:{"shape":list(streams[mode][key].shape),"dtype":str(streams[mode][key].dtype),"sha256":io.array_sha(streams[mode][key])} for key in ("features","baseline","latent_history","latent_future")} for mode in MODES};metadata["baseline_temporal_diagnostics"]={mode:temporal_scores(streams[mode]["baseline"],raw) for mode in MODES};result["datasets"][data]["spaces"][space]["contexts"][split]=metadata;contexts[data,space,split]=(raw,streams,metadata);return contexts[data,space,split]
        def run(data,space,mode,arm):
            raw,fitstreams,fit_sources=context(data,space,"fit");tune,tunestreams,tune_sources=context(data,space,"tune");f,t=fitstreams[mode],tunestreams[mode]
            if (data,space) not in scales:scales[data,space]=module.feature_scale(fitstreams["F"]["features"])
            scale,constant=scales[data,space];response=fit_response(raw["target"],raw["observed"],f["baseline"]);started=time.perf_counter();result["fit_attempts"]+=1;states,metadata=module.fit_path(module.gated_features(f["features"],arm),response,raw["observed"],scale,constant);fit_wall=time.perf_counter()-started;summary,arrays=fit_info(metadata);name=mode+"_"+arm;prefix=data+"_"+space+"_"+name
            row={"status":"RUNNING","space":space,"mode":mode,"arm":arm,"K":fit_sources["K"],"additional_slope_parameters":states[0]["coefficient_FP64"].size,"fit_metadata":summary,"FIT_sources":fit_sources,"TUNE_sources":tune_sources,"shared_RMS_source":"actual F ungated FIT lag-delta features for same dataset/space","fit_wall_seconds":fit_wall,"candidates":[]};result["datasets"][data]["spaces"][space]["conditions"][name]=row;row["fit_artifact"]=io.save_npz(output/(prefix+"_FIT_INFO.npz"),arrays);result["fits_completed"]+=1;meter.check({"stage":"FIT_STATIC_OR_RAMP_CORRECTION","dataset":data,"space":space,"condition":name})
            zero={"alpha":None,"coefficient_FP64":np.zeros_like(states[0]["coefficient_FP64"]),"coefficient_FP32":np.zeros_like(states[0]["coefficient_FP32"])}
            for state in [*states,zero]:
                kind="ZERO_DELTA" if state["alpha"] is None else "ALPHA";identifier=kind if kind=="ZERO_DELTA" else "ALPHA_"+format(state["alpha"],"g");candidate={"id":identifier,"kind":kind,"alpha":state["alpha"],"status":"FAILED_CANDIDATE_PRESERVED"};candidate["coefficient_array_records"],candidate["coefficient_value_sha256"]=parent.coefficient_records(state)
                try:
                    p,pc=predict_candidate(f["baseline"],f["features"],state,arm,kind);q,qc=predict_candidate(t["baseline"],t["features"],state,arm,kind);candidate.update(status="COMPLETE",FIT=scores(p,raw),TUNE=scores(q,tune),parity_checks={"FIT":pc,"TUNE":qc});candidate["TUNE_MSE"]=candidate["TUNE"]["scores_all_observed"]["normalized"]["mse"];candidate["prediction_artifact"]=io.save_npz(output/(prefix+"_"+identifier+".npz"),{"FIT_prediction_FP32":p,"TUNE_prediction_FP32":q,"FIT_origins":raw["origins"],"TUNE_origins":tune["origins"]});result["alpha_candidates_completed" if kind=="ALPHA" else "ZERO_candidates_completed"]+=1;result["candidate_FIT_TUNE_evaluations_completed"]+=2
                except Exception as error:candidate["failure"]={"type":type(error).__name__,"message":str(error)}
                row["candidates"].append(candidate)
            winner=select_candidate(row["candidates"]);selected=zero if winner["kind"]=="ZERO_DELTA" else states[ALPHAS.index(winner["alpha"])];artifact=io.save_npz(output/(prefix+"_SELECTED_COEFFICIENTS.npz"),{k:v for k,v in selected.items() if k!="alpha"})
            selected_predictions=load(winner["prediction_artifact"]);row.update(status="COMPLETE" if all(p["status"]=="COMPLETE" for p in row["candidates"]) else "PARTIAL_CANDIDATE_MENU_PRESERVED",selected_candidate=winner["id"],selected_kind=winner["kind"],selected_alpha=winner["alpha"],selected_TUNE_MSE=winner["TUNE_MSE"],selected_coefficients=artifact,selected_coefficient_value_sha256=winner["coefficient_value_sha256"],FIT=winner["FIT"],TUNE=winner["TUNE"],selected_FIT_TUNE_prediction_artifact=winner["prediction_artifact"],selected_FIT_TUNE_reused_without_new_inference=True,selected_temporal_diagnostics={"FIT":temporal_scores(selected_predictions["FIT_prediction_FP32"],raw),"TUNE":temporal_scores(selected_predictions["TUNE_prediction_FP32"],tune)},selection_rule="Own actual FP32 TUNE all-observed macro MSE; alpha-grid order then ZERO exact tie");return row
        def seal(data,space,mode,arm,row):
            path=output/(data+"_"+space+"_"+mode+"_"+arm+"_SELECTION_LOCK.json");rt.write_new(path,{"status":row["status"],"dataset":data,"space":space,"mode":mode,"arm":arm,"selected_candidate":row.get("selected_candidate"),"selected_kind":row.get("selected_kind"),"selected_alpha":row.get("selected_alpha"),"selected_TUNE_MSE":row.get("selected_TUNE_MSE"),"selected_coefficients":row.get("selected_coefficients"),"selected_coefficient_value_sha256":row.get("selected_coefficient_value_sha256"),"DEV_opened_before_lock":False,"sealed_unix":time.time()});return old.record(path)
        def registry(locks):
            path=output/"HORIZON_SELECTION_REGISTRY.json";rt.write_new(path,{"status":"ALL_24_RESOLVED_BEFORE_DEV","conditions":locks,"DEV_opened_before_seal":False,"sealed_unix":time.time()});return old.record(path)
        def development(data,space,mode,arm,row):
            old.bound(result["selection_registry"]);old.bound(row["selection_lock"]);state=load(row["selected_coefficients"]);require(parent.coefficient_records(state)[1]==row["selected_coefficient_value_sha256"],"SELECTED_COEFFICIENT_VALUES_UNCHANGED");raw,streams,metadata=context(data,space,DATASETS[data][2]);p,check=predict_candidate(streams[mode]["baseline"],streams[mode]["features"],state,arm,row["selected_kind"]);entry=scores(p,raw);entry.update(artifact=io.save_npz(output/(data+"_"+space+"_"+mode+"_"+arm+"_SELECTED_DEV.npz"),{"prediction_FP32":p,"origins":raw["origins"]}),sources=metadata,parity_check=check,selected_coefficients=row["selected_coefficients"],temporal_diagnostics=temporal_scores(p,raw));return entry
        with threadpool_limits(limits=2):resolved=resolve_paths(result,run,seal,registry);infer_development(result,resolved,development)
        for item in protected+list(checked.values())+result["source_records"]:old.bound(item)
        for item in result["source_records"]:require(old.record(item["snapshot_path"])["sha256"]==item["sha256"],"EXECUTED_SOURCES_UNCHANGED")
        result.update(input_records=list(checked.values()),protected_prior_records=protected,status="COMPLETE" if len(resolved)==24 and result["fits_completed"]==24 and result["alpha_candidates_completed"]==240 and result["ZERO_candidates_completed"]==24 and result["candidate_FIT_TUNE_evaluations_completed"]==528 and result["selected_DEV_evaluations_completed"]==24 and not result["failed_paths"] else "PARTIAL_COMPLETED");meter.protected_check();json.dumps(result,allow_nan=False)
    except Exception as error:failure={"type":type(error).__name__,"message":str(error)};result.update(status="FAILED_PRESERVED",failure=failure)
    finally:result["resource_terminal"]=meter.close(result["status"],failure);rt.write_new(output/"RESULT.json",result)
    print(json.dumps({"status":result["status"],"result":str(output/"RESULT.json")}),flush=True);return 0 if result["status"]=="COMPLETE" else 2


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("tag","cross-result","cross-audit"):parser.add_argument("--"+key,required=True)
    raise SystemExit(execute(parser.parse_args()))
