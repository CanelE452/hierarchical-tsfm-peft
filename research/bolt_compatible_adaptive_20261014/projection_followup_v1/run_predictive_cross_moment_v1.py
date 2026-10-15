"""Eight FIT-only spatial directions / same refitted L7 D / actual frozen Bolt controls."""
from pathlib import Path
import argparse
import json
import shutil
import sys
import time

import run_linear_e_adaptation_v1 as old

HERE,ROOT,BASE,OLD,require=old.HERE,old.ROOT,old.BASE,old.OLD,old.require
DATASETS={"robin":(17,4,"val",30),"beijing_aotizhongxin":(11,2,"eval",146)}
KINDS=("PCA","CROSS_MOMENT")
CONDITIONS=tuple((data,kind,k) for data,(_,base,_,_) in DATASETS.items() for kind in KINDS for k in (base,2*base))
LAGS=(0,1,2,4,8,16,32)
MODULE_SHA="c1cec75137a8df2f1d5250829514791a3cc1d3747508e3bd62bf6ab0cf02bffe"
PARENT_READER_SHA="1c0676b75b2baaa1d937bade5e1893d06ba46c01f7c73d48953144e9d09d6daa"


def load_gates(args):
    require(all(isinstance(v,str) and len(v)==64 for v in (MODULE_SHA,PARENT_READER_SHA)),"CROSS_MOMENT_SOURCE_GATE_NOT_REGISTERED")
    old.bound({"path":str(HERE/"predictive_cross_moment_v1.py"),"sha256":MODULE_SHA});old.bound({"path":str(HERE/"run_orthogonal_basis_controls_v1.py"),"sha256":PARENT_READER_SHA})
    from run_orthogonal_basis_controls_v1 import load_gates as parent
    values=parent(args);values[-1].update(parent_metadata_reader=old.record(HERE/"run_orthogonal_basis_controls_v1.py"));return values


def encode_torch(model,history,*,batch_size=32):
    import numpy as np
    import torch
    with torch.no_grad():pieces=[model.encode(torch.as_tensor(history[i:i+batch_size],dtype=torch.float32,device=model.E.device)).cpu().numpy().copy() for i in range(0,len(history),batch_size)]
    z=np.ascontiguousarray(np.concatenate(pieces),dtype=np.float32);require(np.isfinite(z).all(),"FINITE_ACTUAL_FP32_ENCODER");return z


def fit_state(raw,kind,k,*,device="cuda"):
    import numpy as np
    import predictive_cross_moment_v1 as module
    from causal_lag_decoder_v1 import fit_reconstruction
    from linear_e_adaptation_v1 import LinearLagAdapter
    started=time.perf_counter();direction=module.fit_directions(raw["history"],raw["history_observed"],raw["target"],raw["observed"],k,kind=kind);direction_wall=time.perf_counter()-started
    zero=np.zeros((len(LAGS)*k,raw["history"].shape[2]),dtype=np.float32)
    encoder=LinearLagAdapter(direction["center"],direction["E"],zero,LAGS,learn_E=False).to(device).eval().requires_grad_(False)
    z=encode_torch(encoder,raw["history"]);started=time.perf_counter();fitted=fit_reconstruction(raw["history"],raw["history_observed"],z,LAGS);decoder_wall=time.perf_counter()-started
    state={"center":direction["center"],"E":direction["E"],"D":fitted["decoder_FP32"]}
    return direction,fitted,state,z,{"direction_fit_wall_seconds":direction_wall,"decoder_fit_wall_seconds":decoder_wall}


def oracle_dependency_mask(observed,lags=LAGS):
    import numpy as np
    require(observed.dtype==np.bool_ and observed.ndim==3 and len(lags)>0 and lags[0]==0,"OBSERVED_NHC_AND_LAG_ZERO_REQUIRED")
    complete=observed.all(2);valid=complete.copy();h=observed.shape[1]
    for lag in lags:
        require(type(lag) is int and lag>=0,"NONNEGATIVE_INTEGER_LAGS")
        if lag<h:valid[:,lag:] &= complete[:,:h-lag]
    return valid


def timestamp_metadata(raw):
    import numpy as np
    origins=raw["origins"];past=origins[:,None]+np.arange(-512,0);future=origins[:,None]+np.arange(48);complete=raw["observed"].all(2)
    def counts(values):
        unique,multiplicity=np.unique(values,return_counts=True)
        return {"occurrences":int(values.size),"unique_timestamps":int(unique.size),"maximum_multiplicity":int(multiplicity.max(initial=0)),"timestamp_min":int(unique.min()) if unique.size else None,"timestamp_max":int(unique.max()) if unique.size else None}
    return {"history":counts(past),"future":counts(future),"complete_C_response":counts(future[complete]),
        "origins_unique":int(np.unique(origins).size),"complete_HC_origins":int(complete.all(1).sum()),
        "source_current_filled_fraction":float((~raw["history_observed"][:,-1]).mean()),"source_history_filled_fraction":float((~raw["history_observed"]).mean()),
        "overlapping_occurrences_not_independent_samples":True,"timestamp_deduplication_or_reweighting":False,
        "FIT_boundary_contract_inherited_from_exact_accepted_source":True,"new_raw_data_split_boundary_reconstruction":False}


def evaluate_arrays(outputs,raw,state,oracle_future,oracle_prediction,flat_prediction):
    import numpy as np
    from causal_lag_decoder_v1 import predict,lagged_delta_features
    from decoder_metric_probe_v1 import masked_scores
    z,f,p=(outputs[k] for k in ("latent_history","latent_future","prediction"));expected_z=(raw["history"]-state["center"])@state["E"]
    independent=predict(z,f,raw["raw_last"],state["D"],LAGS);features=lagged_delta_features(z,f,LAGS)
    complete=raw["observed"].all(2);safe=np.where(complete[:,:,None],raw["target"],raw["raw_last"][:,None]).astype(np.float32)
    independent_EY=(safe-state["center"])@state["E"];independent_EY_C=predict(z,independent_EY,raw["raw_last"],state["D"],LAGS)
    flat=np.broadcast_to(z[:,-1:],f.shape).copy();independent_flat=predict(z,flat,raw["raw_last"],state["D"],LAGS)
    checks={"independent_NumPy_encode":old.parity(z,expected_z),"independent_NumPy_old_lag_C":old.parity(p,independent),
        "independent_NumPy_lag_features":old.parity(outputs["lag_features"],features),"independent_NumPy_EY_encode":old.parity(oracle_future,independent_EY),
        "independent_NumPy_EY_lag_C":old.parity(oracle_prediction,independent_EY_C),"independent_NumPy_flat_lag_C":old.parity(flat_prediction,independent_flat)}
    oracle_valid=oracle_dependency_mask(raw["observed"]);oracle_mask=np.broadcast_to(oracle_valid[:,:,None],raw["observed"].shape);complete_mask=np.broadcast_to(complete[:,:,None],raw["observed"].shape)
    predictions={"ACTUAL_F":p,"RAW_BOLT":raw["raw_forecast"],"FLAT":flat_prediction,"CURRENT_C":np.broadcast_to(raw["raw_last"][:,None],p.shape)}
    def score(a,mask):return masked_scores(a,raw["target"],mask,raw["std"]) if mask.any() else {"status":"NO_VALID_TARGETS"}
    methods={name:{"scores_all_observed":score(a,raw["observed"]),"scores_complete_mask":score(a,complete_mask),"scores_oracle_common_mask":score(a,oracle_mask)} for name,a in predictions.items()}
    methods["E_Y"]={"scores_oracle_common_mask":score(oracle_prediction,oracle_mask),"non_oracle_mask_scores_not_reported":True,"deployment_method":False}
    row={"status":"COMPLETE" if all(v["pass"] for v in checks.values()) else "FAILED_PARITY_PRESERVED","checks":checks,"methods":methods,
        "oracle_valid_horizons":int(oracle_valid.sum()),"complete_C_horizons":int(complete.sum()),"total_horizons":int(complete.size),
        "oracle_valid_requires_all_consumed_future_references_complete_C":True,"incomplete_future_horizon_vectors_filled_with_current_C":True,
        "oracle_is_deployment_or_predictability_proof":False,"flat_latent_is_current_C_persistence":False,
        "output_space_minimum_status":"NOT_MEASURED_IN_THIS_STAGE","all_FP32_or_all_real_lower_bound_claimed":False}
    independent_arrays={"independent_prediction":independent,"independent_EY_future":independent_EY,"independent_EY_prediction":independent_EY_C,"independent_FLAT_prediction":independent_flat,"oracle_valid_horizon_mask":oracle_valid}
    return row,independent_arrays


def resolve_states(result,run,seal,registry):
    resolved={};locks=[]
    for data,kind,k in CONDITIONS:
        name=kind+"_K"+str(k);row={}
        try:row=run(data,kind,k);require(row["status"]=="COMPLETE","FIT_STATE_AND_BOTH_FIT_TUNE_CHECKS_REQUIRED")
        except Exception as error:row=result["datasets"][data]["conditions"].get(name,row);row.update(status="FAILED_PATH_PRESERVED",failure={"type":type(error).__name__,"message":str(error)})
        row["state_lock"]=seal(data,name,row);result["datasets"][data]["conditions"][name]=row;locks.append({"dataset":data,"condition":name,"status":row["status"],"state_lock":row["state_lock"]})
        if row["status"]=="COMPLETE":resolved[data,name]=row
        else:result["failed_paths"].append({"dataset":data,"condition":name,"stage":"FIT_STATE_FIT_TUNE","failure":row.get("failure")})
    result["state_registry"]=registry(locks);result.update(resolved_state_locks=len(locks),all_8_locks_before_DEV=True);return resolved


def infer_development(result,resolved,infer):
    require(result["all_8_locks_before_DEV"] and result["resolved_state_locks"]==8,"ALL_EIGHT_STATE_LOCKS_BEFORE_DEV")
    for (data,name),row in resolved.items():
        try:
            split=DATASETS[data][2];row["splits"][split]=infer(data,name,row);require(row["splits"][split]["status"]=="COMPLETE","DEV_INDEPENDENT_CHECKS_REQUIRED");result["DEV_evaluations_completed"]+=1
        except Exception as error:result["failed_paths"].append({"dataset":data,"condition":name,"stage":"DEV","failure":{"type":type(error).__name__,"message":str(error)}})


def execute(args):
    calibration,native,science,full,refs=load_gates(args);output=old.fresh_output(args.tag);sys.path[:0]=[str(HERE),str(BASE),str(OLD),str(ROOT)]
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    import predictive_cross_moment_v1 as module
    import fixed_decoder_span_v1 as io
    import continuation_runtime_v1 as rt
    import run_condition_probes as probes
    import cost_readout_v1 as runtime
    from linear_e_adaptation_v1 import LinearLagAdapter
    from run_fit_target_comparison_v1 import prior_file_records
    from prepare_inputs import install_protected_test_guard
    require(old.record(module.__file__)["sha256"]==MODULE_SHA,"FROZEN_DIRECTION_SOURCE")
    output.mkdir(parents=True,exist_ok=False);meter=rt.Meter("predictive_cross_moment_"+args.tag,gpu=True);failure=None
    result={"status":"RUNNING",**refs,"datasets":{data:{"conditions":{}} for data in DATASETS},"failed_paths":[],"source_records":[],
        "state_fit_attempts":0,"direction_fits_completed":0,"decoder_fits_completed":0,"FIT_TUNE_evaluations_completed":0,"DEV_evaluations_completed":0,"resolved_state_locks":0,"all_8_locks_before_DEV":False,
        "new_F_forward_calls":0,"new_F_forward_attempts":0,"new_F_model_objects_loaded":0,"new_Raw_F_inferences":0,"new_optimizer_updates":0,"new_backbone_training":False,
        "native_predictors_or_raw_ridge_fitted":False,"whole_deployment_cost_measured":False,"following_native_ridge_and_cost_stage_still_required":True,
        "protected_TEST_accessed":False,"independent_holdout":False,"strict_K_only":False,"K_or_direction_selected_using_DEV":False,"output_space_minimum_status":"NOT_MEASURED"}
    checked,contexts,models,states,initial,rows={},{},{},{},{},{};protected=[]
    try:
        install_protected_test_guard();meter.protected_check();protected=prior_file_records(calibration,native,science,full,refs)
        folder=output/"sources";folder.mkdir()
        for path in (Path(__file__),HERE/"predictive_cross_moment_v1.py",HERE/"PLAN_PREDICTIVE_CROSS_MOMENT_KO_v1.md",HERE/"PREDICTABLE_PROJECTION_DATA_CONTRACT_KO_v1.md",HERE/"run_orthogonal_basis_controls_v1.py",HERE/"linear_e_adaptation_v1.py",HERE/"causal_lag_decoder_v1.py",HERE/"decoder_metric_probe_v1.py",HERE/"fixed_decoder_span_v1.py",HERE/"continuation_runtime_v1.py",OLD/"run_condition_probes.py",OLD/"prepare_inputs.py",BASE/"cost_readout_v1.py"):
            destination=folder/path.name;shutil.copyfile(path,destination);result["source_records"].append({**old.record(path),"snapshot_path":str(destination)})
        result.update(controller_source=old.record(__file__),module_source=old.record(module.__file__),plan=old.record(HERE/"PLAN_PREDICTIVE_CROSS_MOMENT_KO_v1.md"),data_contract=old.record(HERE/"PREDICTABLE_PROJECTION_DATA_CONTRACT_KO_v1.md"))
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        subset=old.read(old.bound(science["raw_subset_diagnostic"]));original=old.read(old.bound(subset["source_producer"]))
        engine=probes.import_old_engine();weight,config=(probes.verified_file(original[key]) for key in ("model_binary_before","model_config_before"));bolt=engine.ex.previous.FrozenBolt();result["new_F_model_objects_loaded"]=1
        result.update(F_before=probes.model_hashes(bolt),model_binary_before=weight,model_config_before=config)
        require(result["F_before"]==original["F_before"]==original["F_after"]==calibration["F_parent_signatures_reused"]["before"]==calibration["F_parent_signatures_reused"]["after"] and all(not p.requires_grad and p.grad is None for p in bolt.model.parameters()),"ORIGINAL_FROZEN_F_DUAL_HASH")
        result["runtime_behavior_contract"]=runtime.bolt_runtime_contract(engine,bolt,meter.check);require(result["runtime_behavior_contract"]==full["runtime_behavior_contract"],"UNCHANGED_F_INTERNAL_NORMALIZATION")
        def load(item):return io.checked_payload({**item,"path":str(old.bound(item))},meter,checked)
        def context(data,split):
            require(split in ("fit","tune") or result["all_8_locks_before_DEV"],"EIGHT_LOCKS_BEFORE_DEV_ARRAY_IO")
            key=data,split
            if key not in contexts:
                source=science["datasets"][data]["spaces"]["PCA"]["conditions"]["L7_RECON"]["splits"][split]["input_source"];raw=load(source);c,_,dev,devn=DATASETS[data];n={"fit":256,"tune":64}.get(split,devn)
                require(split in ("fit","tune",dev) and raw["history"].dtype==raw["target"].dtype==raw["raw_forecast"].dtype==np.float32 and raw["history"].shape==(n,512,c) and raw["target"].shape==raw["observed"].shape==raw["raw_forecast"].shape==(n,48,c) and raw["history_observed"].dtype==raw["observed"].dtype==np.bool_ and raw["history_observed"].shape==raw["history"].shape and np.isfinite(raw["history"]).all() and np.isfinite(raw["target"][raw["observed"]]).all() and np.isfinite(raw["raw_forecast"]).all() and old.byte_equal(raw["raw_last"],raw["history"][:,-1]),"SAME_FINITE_ORIGINAL_DATASET_COHORT")
                require(raw["origins"].dtype==np.int64 and raw["origins"].shape==(n,) and raw["mean"].dtype==raw["std"].dtype==np.float32 and raw["mean"].shape==raw["std"].shape==(c,) and np.isfinite(raw["mean"]).all() and np.isfinite(raw["std"]).all() and (raw["std"]>0).all(),"ORIGINAL_ORIGINS_AND_FINITE_FIT_SCALER")
                contexts[key]=(raw,source);result["datasets"][data].setdefault("timestamp_occurrences",{})[split]=timestamp_metadata(raw)
            return contexts[key]
        def actual_F(z):
            result["new_F_forward_attempts"]+=1;future=bolt(z);result["new_F_forward_calls"]+=1;return future
        def evaluate(data,name,split):
            raw,input_source=context(data,split);key=data,name;model=models[key];outputs=old.predict_components(model,actual_F,raw["history"],meter.check)
            complete=raw["observed"].all(2);safe=np.where(complete[:,:,None],raw["target"],raw["raw_last"][:,None]).astype(np.float32)
            ey=encode_torch(model,safe);flat=np.broadcast_to(outputs["latent_history"][:,-1:],outputs["latent_future"].shape).copy();eypieces=[];flatpieces=[]
            with torch.no_grad():
                for start in range(0,len(safe),32):
                    current=torch.as_tensor(raw["raw_last"][start:start+32],device=model.E.device);past=torch.as_tensor(outputs["latent_history"][start:start+32],device=model.E.device)
                    eypieces.append(model.decode(current,past,torch.as_tensor(ey[start:start+32],device=model.E.device)).cpu().numpy().copy());flatpieces.append(model.decode(current,past,torch.as_tensor(flat[start:start+32],device=model.E.device)).cpu().numpy().copy())
            eyp,flatp=(np.ascontiguousarray(np.concatenate(v),dtype=np.float32) for v in (eypieces,flatpieces))
            artifact=io.save_npz(output/(data+"_"+name+"_"+split+".npz"),{**outputs,"EY_future":ey,"EY_prediction":eyp,"FLAT_future":flat,"FLAT_prediction":flatp,"oracle_valid_horizon_mask":oracle_dependency_mask(raw["observed"]),"origins":raw["origins"]})
            row={"status":"FAILED_DIAGNOSTIC_PRESERVED","artifact":artifact,"input_source":input_source,"state_source":rows[key]["state_artifact"],"new_actual_F_inference":True,"Raw_baseline_reused_from_original_input_member":"raw_forecast"}
            try:
                diagnostic,independent=evaluate_arrays(outputs,raw,states[key],ey,eyp,flatp);row.update(diagnostic);row["independent_artifact"]=io.save_npz(output/(data+"_"+name+"_"+split+"_INDEPENDENT.npz"),independent)
            except Exception as error:row["failure"]={"type":type(error).__name__,"message":str(error)}
            if name=="PCA_K"+str(DATASETS[data][1]):
                previous=science["datasets"][data]["spaces"]["PCA"]["conditions"]["L7_RECON"]["splits"][split];legacy=load(previous["artifact"])
                row["legacy_PCA_diagnostic"]={"source":previous["artifact"],"old_NumPy_history_vs_new_Torch32":old.parity(outputs["latent_history"],raw["latent_history_BOLT_NUMPY"]),"old_prediction_vs_refitted_new_prediction":old.parity(outputs["prediction"],legacy["ACTUAL_F_prediction_FP32"]),"parity_required_for_new_comparison_success":False,"different_FP32_backend_D_refit_or_sign_can_change_prediction":True}
            if split in ("fit","tune") and row["status"]=="COMPLETE":result["FIT_TUNE_evaluations_completed"]+=1
            return row
        def run(data,kind,k):
            name=kind+"_K"+str(k);key=data,name;raw,source=context(data,"fit");result["state_fit_attempts"]+=1;direction,fitted,state,z,fit_times=fit_state(raw,kind,k)
            row={"status":"RUNNING","kind":kind,"K":k,"C":DATASETS[data][0],"FIT_input_source":source,"direction_fit_metadata":direction["metadata"],**fit_times,"E_FIT_future_supervised":kind=="CROSS_MOMENT","D_FIT_future_supervised":False,"splits":{}}
            singular=direction["singularvalues"];row["moment_singular_gap_at_K"]={"sigma_K":float(singular[k-1]),"sigma_Kplus1":float(singular[k]) if k<len(singular) else None,"gap":float(singular[k-1]-singular[k]) if k<len(singular) else None,"ratio":float(singular[k-1]/singular[k]) if k<len(singular) and singular[k]>0 else None,"tied_or_near_tied_directions_not_unique":True}
            result["datasets"][data]["conditions"][name]=row
            row["direction_artifact"]=io.save_npz(output/(data+"_"+name+"_DIRECTION.npz"),{a:b for a,b in direction.items() if a!="metadata"});result["direction_fits_completed"]+=1
            row["decoder_fit_metadata"]=fitted["metadata"];row["decoder_artifact"]=io.save_npz(output/(data+"_"+name+"_DECODER.npz"),{a:b for a,b in fitted.items() if a!="metadata"});result["decoder_fits_completed"]+=1
            row["state_artifact"]=io.save_npz(output/(data+"_"+name+"_STATE.npz"),state)
            model=LinearLagAdapter(state["center"],state["E"],state["D"],LAGS,learn_E=False).to("cuda").eval().requires_grad_(False);require(all(old.byte_equal(old.state_arrays(model)[a],b) for a,b in state.items()),"EXACT_NEW_FROZEN_E_D_CENTER")
            states[key]=state;models[key]=model;initial[key]=old.state_arrays(model);rows[key]=row
            if kind=="PCA" and k==DATASETS[data][1]:
                previous=science["datasets"][data]["spaces"]["PCA"];e=load(previous["E_source"]);oldD=load(previous["conditions"]["L7_RECON"]["state_artifact"])
                row["legacy_PCA_state_diagnostic"]={"E_source":previous["E_source"],"D_source":previous["conditions"]["L7_RECON"]["state_artifact"],"encoder_FP32":old.parity(state["E"],e["encoder"].astype(np.float32)),"center_FP32":old.parity(state["center"],e["center"].astype(np.float32)),"refit_decoder_FP32":old.parity(state["D"],oldD["decoder_FP32"]),"equality_or_identical_D_rank_not_required":True}
            for split in ("fit","tune"):
                try:row["splits"][split]=evaluate(data,name,split)
                except Exception as error:row["splits"][split]={"status":"FAILED_PATH_PRESERVED","failure":{"type":type(error).__name__,"message":str(error)}}
            row["status"]="COMPLETE" if all(v["status"]=="COMPLETE" for v in row["splits"].values()) else "FAILED_PATH_PRESERVED";return row
        def seal(data,name,row):
            path=output/(data+"_"+name+"_STATE_LOCK.json");rt.write_new(path,{"status":row["status"],"dataset":data,"condition":name,"state_artifact":row.get("state_artifact"),"direction_artifact":row.get("direction_artifact"),"decoder_artifact":row.get("decoder_artifact"),"FIT_TUNE_statuses":{a:b["status"] for a,b in row.get("splits",{}).items()},"DEV_opened_before_lock":False,"state_selection_performed":False,"sealed_unix":time.time()});return old.record(path)
        def registry(locks):
            path=output/"STATE_REGISTRY.json";rt.write_new(path,{"status":"ALL_EIGHT_RESOLVED_BEFORE_DEV","conditions":locks,"DEV_opened_before_seal":False,"state_selection_performed":False,"sealed_unix":time.time()});return old.record(path)
        def development(data,name,row):old.bound(result["state_registry"]);old.bound(row["state_lock"]);old.bound(row["state_artifact"]);return evaluate(data,name,DATASETS[data][2])
        with threadpool_limits(limits=2):resolved=resolve_states(result,run,seal,registry);infer_development(result,resolved,development)
        for key,model in models.items():require(all(old.byte_equal(v,old.state_arrays(model)[a]) for a,v in initial[key].items()) and all(not p.requires_grad and p.grad is None for p in model.parameters()),"FINAL_FROZEN_E_D_CENTER_BYTES_GRAD_NONE")
        result["F_after"]=probes.model_hashes(bolt);result["model_binary_after"],result["model_config_after"]=probes.verified_file(weight),probes.verified_file(config);require(result["F_before"]==result["F_after"] and all(not p.requires_grad and p.grad is None for p in bolt.model.parameters()),"FROZEN_F_UNCHANGED")
        for item in protected+list(checked.values())+result["source_records"]:old.bound(item)
        for item in result["source_records"]:require(old.record(item["snapshot_path"])["sha256"]==item["sha256"],"SOURCE_SNAPSHOTS_UNCHANGED")
        result.update(input_records=list(checked.values()),protected_prior_records=protected,status="COMPLETE" if len(resolved)==8 and result["direction_fits_completed"]==result["decoder_fits_completed"]==8 and result["FIT_TUNE_evaluations_completed"]==16 and result["DEV_evaluations_completed"]==8 and not result["failed_paths"] else "PARTIAL_COMPLETED");meter.protected_check();json.dumps(result,allow_nan=False)
    except Exception as error:failure={"type":type(error).__name__,"message":str(error)};result.update(status="FAILED_PRESERVED",failure=failure)
    finally:result["resource_terminal"]=meter.close(result["status"],failure);rt.write_new(output/"RESULT.json",result)
    print(json.dumps({"status":result["status"],"result":str(output/"RESULT.json")}),flush=True);return 0 if result["status"]=="COMPLETE" else 2


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("tag","calibration-result","calibration-audit","calibration-cost","calibration-cost-audit","report","report-manifest","report-audit"):parser.add_argument("--"+key,required=True)
    raise SystemExit(execute(parser.parse_args()))
