"""Fresh full-nine-quantile F inference and fixed point-summary selection; no training or cost timing."""
from pathlib import Path
import argparse
import json
import shutil
import sys
import time

import run_linear_e_adaptation_v1 as old

HERE,ROOT,BASE,OLD,require=old.HERE,old.ROOT,old.BASE,old.OLD,old.require
STATES=("RECON_CP0","E_ONLY_s92601","E_ONLY_s92602")
ROUTES=("RAW_C",)+STATES
SUMMARIES=("MEDIAN","NINE_MEAN_PROXY","TAIL_CLAMPED_PROXY")
LAGS=(0,1,2,4,8,16,32)
MODULE_SHA="8af5ee93375a9ab7e2e937682cfbabad6e31e8a038bf6f6cca9c993ca3b87a28"
PARENT_READER_SHA="1c0676b75b2baaa1d937bade5e1893d06ba46c01f7c73d48953144e9d09d6daa"


def load_gates(args):
    require(all(isinstance(v,str) and len(v)==64 for v in (MODULE_SHA,PARENT_READER_SHA)),"POINT_SUMMARY_SOURCE_GATE_NOT_REGISTERED")
    old.bound({"path":str(HERE/"quantile_point_summary_v1.py"),"sha256":MODULE_SHA})
    old.bound({"path":str(HERE/"run_orthogonal_basis_controls_v1.py"),"sha256":PARENT_READER_SHA})
    from run_orthogonal_basis_controls_v1 import load_gates as accepted_parent
    values=accepted_parent(args);values[-1].update(parent_metadata_reader=old.record(HERE/"run_orthogonal_basis_controls_v1.py"))
    return values


def infer_full_quantiles(model,bolt,history,check,*,batch_size=32):
    import numpy as np
    import torch
    from quantile_point_summary_v1 import summarize_torch
    device=next(bolt.model.parameters()).device;pieces={key:[] for key in ("latent_history","full_quantiles_FP32",*(s+"_future" for s in SUMMARIES),*(s+"_prediction" for s in SUMMARIES))};calls=0
    with torch.no_grad():
        for start in range(0,len(history),batch_size):
            x=torch.as_tensor(history[start:start+batch_size],dtype=torch.float32,device=device)
            past=x if model is None else model.encode(x);n,length,k=past.shape
            rows=past.transpose(1,2).reshape(n*k,length)
            full=bolt.model(context=rows).quantile_preds;calls+=1
            require(full.dtype==torch.float32 and tuple(full.shape)==(n*k,9,64) and torch.isfinite(full).all(),"ACTUAL_MODEL_NK_NINE_64_FINITE_FP32")
            full=full.reshape(n,k,9,64).permute(0,3,1,2).contiguous()
            values={"latent_history":past,"full_quantiles_FP32":full}
            for summary in SUMMARIES:
                future=summarize_torch(full[:,:48],summary);prediction=future if model is None else model.decode(x[:,-1],past,future)
                require(torch.isfinite(prediction).all(),"FINITE_ACTUAL_POINT_C_PREDICTION")
                values.update({summary+"_future":future,summary+"_prediction":prediction})
            for key,value in values.items():pieces[key].append(value.detach().cpu().numpy().copy())
            check({"stage":"ACTUAL_FULL_NINE_QUANTILE_CANONICAL32","start":start,"rows":n*k,"new_F_forward":True,"new_optimizer_update":False})
    return {key:np.ascontiguousarray(np.concatenate(value),dtype=np.float32) for key,value in pieces.items()},calls


def evaluate_arrays(outputs,raw,cached,frozen,route):
    import numpy as np
    from quantile_point_summary_v1 import summarize_numpy,quantile_diagnostics
    from causal_lag_decoder_v1 import predict
    from decoder_metric_probe_v1 import masked_scores
    full=outputs["full_quantiles_FP32"];past=outputs["latent_history"];n=len(raw["history"]);raw_route=route=="RAW_C";k=raw["history"].shape[2] if raw_route else 4
    require(full.dtype==past.dtype==np.float32 and full.shape==(n,64,k,9) and past.shape==(n,512,k) and np.isfinite(full).all() and np.isfinite(past).all(),"FINITE_FULL_QUANTILE_AND_ORIGINAL_HISTORY_LAYOUT")
    expected_past=raw["history"] if raw_route else (raw["history"]-frozen["center"])@frozen["E"]
    baseline_future=raw["raw_forecast"] if raw_route else cached["latent_future"];baseline_C=raw["raw_forecast"] if raw_route else cached["prediction"]
    common={"fresh_past_vs_original":old.parity(past,raw["history"] if raw_route else cached["latent_history"]),"independent_NumPy_encode":old.parity(past,expected_past),
        "median_future_vs_accepted_cache":old.parity(outputs["MEDIAN_future"],baseline_future),"median_C_vs_accepted_cache":old.parity(outputs["MEDIAN_prediction"],baseline_C)}
    complete=np.broadcast_to(raw["observed"].all(2,keepdims=True),raw["observed"].shape)
    summaries={};independent={}
    for summary in SUMMARIES:
        future=summarize_numpy(full[:,:48],summary);prediction=future if raw_route else predict(past,future,raw["raw_last"],frozen["D"],LAGS)
        checks={"NumPy_FP32_summary":old.parity(outputs[summary+"_future"],future),"NumPy_old_lag_D_C":old.parity(outputs[summary+"_prediction"],prediction)}
        summaries[summary]={"status":"COMPLETE" if all(p["pass"] for p in checks.values()) else "FAILED_PARITY_PRESERVED","checks":checks,
            "scores_all_observed":masked_scores(outputs[summary+"_prediction"],raw["target"],raw["observed"],raw["std"]),
            "scores_complete_mask":masked_scores(outputs[summary+"_prediction"],raw["target"],complete,raw["std"]) if complete.any() else {"status":"NO_COMPLETE_C"}}
        independent.update({summary+"_NumPy_future":future,summary+"_NumPy_prediction":prediction})
    return {"status":"COMPLETE" if all(p["pass"] for p in common.values()) and all(p["status"]=="COMPLETE" for p in summaries.values()) else "FAILED_PARITY_PRESERVED",
        "checks":common,"summaries":summaries,"full64_quantile_diagnostics":quantile_diagnostics(full),"evaluated48_quantile_diagnostics":quantile_diagnostics(full[:,:48]),
        "complete_target_positions":int(complete[:,:,0].sum()),"future_labels_used_for_forward":False,"extra_manual_normalization":False,
        "pipeline_mean_used":False,"true_conditional_mean_or_C_quantile_claimed":False},independent


def resolve_routes(result,run,seal,registry):
    from quantile_point_summary_v1 import select_summary
    resolved={};locks=[]
    for route in ROUTES:
        row={}
        try:
            row=run(route)
            require(row["status"]=="COMPLETE","FULL_THREE_SUMMARY_MENU_AND_MEDIAN_CACHE_PARITY_REQUIRED")
            candidates=[{"summary":s,"status":"COMPLETE","TUNE_MSE":row["splits"]["tune"]["summaries"][s]["scores_all_observed"]["normalized"]["mse"]} for s in SUMMARIES]
            winner=select_summary(candidates);row.update(selected_summary=winner["summary"],selected_TUNE_MSE=winner["TUNE_MSE"],selection_rule="Own actual TUNE observed-channel-macro normalized MSE; exact tie MEDIAN,NINE_MEAN_PROXY,TAIL_CLAMPED_PROXY")
        except Exception as error:
            row.update(status="FAILED_PATH_PRESERVED",failure={"type":type(error).__name__,"message":str(error)})
        row["selection_lock"]=seal(route,row);result["routes"][route]=row;locks.append({"route":route,"status":row["status"],"selection_lock":row["selection_lock"]})
        if row["status"]=="COMPLETE":resolved[route]=row
        else:result["failed_paths"].append({"route":route,"stage":"FIT_TUNE_MENU_AND_SELECTION","failure":row.get("failure")})
    result["selection_registry"]=registry(locks);result.update(resolved_selection_locks=len(locks),all_4_locks_before_DEV=True)
    return resolved


def infer_development(result,resolved,infer):
    require(result["all_4_locks_before_DEV"] and result["resolved_selection_locks"]==4,"ALL_FOUR_LOCKS_BEFORE_ANY_DEV")
    for route,row in resolved.items():
        try:
            row["splits"]["val"]=infer(route,row);require(row["splits"]["val"]["status"]=="COMPLETE","ALL_THREE_DEV_DIAGNOSTIC_SUMMARIES_PASS")
            for summary,entry in row["splits"]["val"]["summaries"].items():entry.update(selected=summary==row["selected_summary"],nonselected_DEV_diagnostic_only=summary!=row["selected_summary"])
            result["DEV_diagnostic_summary_evaluations_completed"]+=3;result["selected_DEV_evaluations_completed"]+=1
        except Exception as error:result["failed_paths"].append({"route":route,"stage":"DEV_SUMMARY_DIAGNOSTIC","failure":{"type":type(error).__name__,"message":str(error)}})


def execute(args):
    calibration,native,science,full,refs=load_gates(args);output=old.fresh_output(args.tag);sys.path[:0]=[str(HERE),str(BASE),str(OLD),str(ROOT)]
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    import quantile_point_summary_v1 as module
    import fixed_decoder_span_v1 as io
    import continuation_runtime_v1 as rt
    import run_condition_probes as probes
    import cost_readout_v1 as runtime
    from linear_e_adaptation_v1 import LinearLagAdapter
    from prepare_inputs import install_protected_test_guard
    from run_fit_target_comparison_v1 import prior_file_records
    require(old.record(module.__file__)["sha256"]==MODULE_SHA,"FROZEN_POINT_SUMMARY_SOURCE")
    output.mkdir(parents=True,exist_ok=False);meter=rt.Meter("quantile_point_summary_"+args.tag,gpu=True);failure=None
    result={"status":"RUNNING",**refs,"routes":{},"failed_paths":[],"source_records":[],"FIT_TUNE_summary_evaluations_completed":0,
        "DEV_diagnostic_summary_evaluations_completed":0,"selected_DEV_evaluations_completed":0,"resolved_selection_locks":0,"all_4_locks_before_DEV":False,
        "full_quantile_contexts_completed":0,"new_F_forward_calls":0,"new_F_forward_attempts":0,"new_F_model_objects_loaded":0,"new_optimizer_updates":0,
        "new_E_D_Q_learning_updates":0,"new_backbone_training":False,"native_head_weights_loaded":False,"Q_identity":True,"extra_manual_normalization":False,
        "pipeline_mean_used":False,"true_conditional_mean_or_decoded_C_quantile_claimed":False,"protected_TEST_accessed":False,"independent_holdout":False,
        "strict_K_only":False,"whole_deployment_cost_measured":False,"all_three_DEV_summaries_are_not_twelve_independent_datasets":True}
    checked,contexts,caches,states,models,initial={},{},{},{},{},{};protected=[]
    try:
        install_protected_test_guard();meter.protected_check();protected=prior_file_records(calibration,native,science,full,refs)
        folder=output/"sources";folder.mkdir()
        for path in (Path(__file__),HERE/"quantile_point_summary_v1.py",HERE/"PLAN_QUANTILE_POINT_SUMMARY_KO_v1.md",HERE/"run_orthogonal_basis_controls_v1.py",HERE/"linear_e_adaptation_v1.py",HERE/"causal_lag_decoder_v1.py",HERE/"decoder_metric_probe_v1.py",HERE/"fixed_decoder_span_v1.py",HERE/"continuation_runtime_v1.py",OLD/"run_condition_probes.py",OLD/"prepare_inputs.py",BASE/"cost_readout_v1.py"):
            destination=folder/path.name;shutil.copyfile(path,destination);result["source_records"].append({**old.record(path),"snapshot_path":str(destination)})
        result.update(controller_source=old.record(__file__),module_source=old.record(module.__file__),plan=old.record(HERE/"PLAN_QUANTILE_POINT_SUMMARY_KO_v1.md"))
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        subset=old.read(old.bound(science["raw_subset_diagnostic"]));original=old.read(old.bound(subset["source_producer"]))
        engine=probes.import_old_engine();weight,config=(probes.verified_file(original[key]) for key in ("model_binary_before","model_config_before"));bolt=engine.ex.previous.FrozenBolt();result["new_F_model_objects_loaded"]=1
        result.update(F_before=probes.model_hashes(bolt),model_binary_before=weight,model_config_before=config)
        require(result["F_before"]==original["F_before"]==original["F_after"]==calibration["F_parent_signatures_reused"]["before"]==calibration["F_parent_signatures_reused"]["after"] and all(not p.requires_grad and p.grad is None for p in bolt.model.parameters()),"EXACT_FROZEN_F_DUAL_SIGNATURE")
        result["runtime_behavior_contract"]=runtime.bolt_runtime_contract(engine,bolt,meter.check);require(result["runtime_behavior_contract"]==full["runtime_behavior_contract"],"UNCHANGED_INTERNAL_NORMALIZATION_RUNTIME")
        require(tuple(bolt.model.chronos_config.quantiles)==module.QUANTILES and bolt.model.chronos_config.prediction_length==64,"ORIGINAL_NINE_LEVELS_AND_64_HORIZON")
        def load(item):return io.checked_payload({**item,"path":str(old.bound(item))},meter,checked)
        def context(route,split):
            require(split in ("fit","tune") or result["all_4_locks_before_DEV"],"FOUR_SUMMARY_LOCKS_BEFORE_DEV_ARRAY_IO")
            state=STATES[0] if route=="RAW_C" else route;source=calibration["states"][state]["families"]["SCALE_ONLY"][{"fit":"FIT","tune":"TUNE","val":"DEV"}[split]]["input_source"]
            if split not in contexts:
                raw=load(source);n={"fit":256,"tune":64,"val":30}[split]
                require(raw["history"].dtype==raw["target"].dtype==raw["raw_forecast"].dtype==np.float32 and raw["history"].shape==(n,512,17) and raw["target"].shape==raw["observed"].shape==raw["raw_forecast"].shape==(n,48,17) and raw["observed"].dtype==np.bool_ and np.isfinite(raw["history"]).all() and np.isfinite(raw["raw_forecast"]).all() and np.isfinite(raw["target"][raw["observed"]]).all() and old.byte_equal(raw["raw_last"],raw["history"][:,-1]),"ORIGINAL_ROBIN_MASK_SCALER_ORIGINS_AND_RAW_BASELINE")
                contexts[split]=(raw,source)
            old.bound(source,contexts[split][1]);return contexts[split]
        def model_for(route):
            if route=="RAW_C":return None
            if route not in models:
                source=native["states"][route]["frozen_state_source"];old.bound(source,calibration["states"][route]["frozen_state_source"]);v=load(source)
                require(set(v)=={"E","D","center"} and v["E"].shape==(17,4) and v["D"].shape==(28,17) and v["center"].shape==(17,) and all(a.dtype==np.float32 and np.isfinite(a).all() for a in v.values()),"EXACT_THREE_FROZEN_E_D_CENTER_STATES")
                model=LinearLagAdapter(v["center"],v["E"],v["D"],LAGS,learn_E=False).to("cuda").eval().requires_grad_(False)
                require(all(old.byte_equal(model.state_dict()[k].detach().cpu().numpy(),v[k]) for k in v),"FROZEN_SOURCE_STATE_BYTES")
                models[route]=model;states[route]=v;initial[route]=old.state_arrays(model)
            return models[route]
        def cache(route,split):
            raw,source=context(route,split)
            if route=="RAW_C":return {"prediction":raw["raw_forecast"],"latent_future":raw["raw_forecast"]}, {"artifact":source,"member":"raw_forecast","member_manifest":source["arrays"]["raw_forecast"]}
            if (route,split) not in caches:
                source=native["states"][route]["Bolt"][split]["artifact"];value=load(source)
                for name in ("target","observed","raw_last","origins","mean","std"):require(old.byte_equal(raw[name],value[name]),"SAME_ACCEPTED_BOLT_CONTEXT:"+name)
                caches[route,split]=(value,source)
            return caches[route,split]
        def evaluate(route,split):
            raw,input_source=context(route,split);model=model_for(route);cached,baseline_source=cache(route,split)
            class CountedModel:
                def parameters(self):return bolt.model.parameters()
                def __call__(self,**kwargs):
                    result["new_F_forward_attempts"]+=1;v=bolt.model(**kwargs);result["new_F_forward_calls"]+=1;return v
            class DirectBolt:model=CountedModel()
            outputs,calls=infer_full_quantiles(model,DirectBolt(),raw["history"],meter.check)
            artifact=io.save_npz(output/(route+"_"+split+"_FULL_QUANTILES.npz"),{key:outputs[key] for key in ("latent_history","full_quantiles_FP32")}|{"origins":raw["origins"]})
            row={"status":"FAILED_DIAGNOSTIC_PRESERVED","artifact":artifact,"input_source":input_source,"accepted_median_baseline_source":baseline_source,"actual_full_F_calls":calls,"canonical_batch_size":32,"stored_horizon":64,"evaluated_horizon":48}
            result["full_quantile_contexts_completed"]+=1
            for summary in SUMMARIES:
                row.setdefault("candidate_artifacts",{})[summary]=io.save_npz(output/(route+"_"+summary+"_"+split+".npz"),{"prediction_FP32":outputs[summary+"_prediction"],"latent_future_FP32":outputs[summary+"_future"],"origins":raw["origins"]})
            try:
                diagnostic,independent=evaluate_arrays(outputs,raw,cached,states.get(route),route);row.update(diagnostic)
                row["independent_artifact"]=io.save_npz(output/(route+"_"+split+"_INDEPENDENT.npz"),independent)
                for summary,entry in row["summaries"].items():entry.update(artifact=row["candidate_artifacts"][summary],full_quantile_artifact=artifact)
                if split in ("fit","tune"):result["FIT_TUNE_summary_evaluations_completed"]+=sum(v["status"]=="COMPLETE" for v in row["summaries"].values())
            except Exception as error:row["failure"]={"type":type(error).__name__,"message":str(error)}
            return row
        def run(route):
            model_for(route);row={"status":"RUNNING","splits":{},"Q_identity":True,"E_previously_supervised_by_FIT_future":route.startswith("E_ONLY"),"D_previously_supervised_by_FIT_future":False,"frozen_state_source":native["states"][route]["frozen_state_source"] if route!="RAW_C" else None}
            for split in ("fit","tune"):
                try:row["splits"][split]=evaluate(route,split)
                except Exception as error:row["splits"][split]={"status":"FAILED_PATH_PRESERVED","failure":{"type":type(error).__name__,"message":str(error)}}
            row["status"]="COMPLETE" if all(v["status"]=="COMPLETE" for v in row["splits"].values()) else "FAILED_PATH_PRESERVED";return row
        def seal(route,row):
            path=output/(route+"_SUMMARY_SELECTION_LOCK.json");rt.write_new(path,{"status":row["status"],"route":route,"selected_summary":row.get("selected_summary"),"selected_TUNE_MSE":row.get("selected_TUNE_MSE"),"TUNE_menu":{s:v["scores_all_observed"]["normalized"]["mse"] for s,v in row.get("splits",{}).get("tune",{}).get("summaries",{}).items()},"DEV_opened_before_lock":False,"sealed_unix":time.time()});return old.record(path)
        def registry(rows):
            path=output/"SUMMARY_SELECTION_REGISTRY.json";rt.write_new(path,{"status":"ALL_FOUR_RESOLVED_BEFORE_DEV","conditions":rows,"DEV_opened_before_seal":False,"sealed_unix":time.time()});return old.record(path)
        def development(route,row):old.bound(result["selection_registry"]);old.bound(row["selection_lock"]);return evaluate(route,"val")
        with threadpool_limits(limits=2):resolved=resolve_routes(result,run,seal,registry);infer_development(result,resolved,development)
        result["selected_routes_completed"]=len(resolved)
        for route,model in models.items():require(all(old.byte_equal(v,old.state_arrays(model)[k]) for k,v in initial[route].items()) and all(not p.requires_grad and p.grad is None for p in model.parameters()),"FINAL_E_D_CENTER_BYTES_AND_GRAD_NONE")
        result["F_after"]=probes.model_hashes(bolt);result["model_binary_after"],result["model_config_after"]=probes.verified_file(weight),probes.verified_file(config)
        require(result["F_before"]==result["F_after"] and all(not p.requires_grad and p.grad is None for p in bolt.model.parameters()),"FINAL_F_FROZEN_HASH_AND_GRAD_NONE")
        for item in protected+list(checked.values())+result["source_records"]:old.bound(item)
        for item in result["source_records"]:require(old.record(item["snapshot_path"])["sha256"]==item["sha256"],"ORIGINAL_EXECUTED_SOURCE_BYTES")
        result.update(input_records=list(checked.values()),protected_prior_records=protected,status="COMPLETE" if len(resolved)==4 and result["FIT_TUNE_summary_evaluations_completed"]==24 and result["DEV_diagnostic_summary_evaluations_completed"]==12 and result["selected_DEV_evaluations_completed"]==4 and result["full_quantile_contexts_completed"]==12 and not result["failed_paths"] else "PARTIAL_COMPLETED");meter.protected_check();json.dumps(result,allow_nan=False)
    except Exception as error:failure={"type":type(error).__name__,"message":str(error)};result.update(status="FAILED_PRESERVED",failure=failure)
    finally:result["resource_terminal"]=meter.close(result["status"],failure);rt.write_new(output/"RESULT.json",result)
    print(json.dumps({"status":result["status"],"result":str(output/"RESULT.json")}),flush=True);return 0 if result["status"]=="COMPLETE" else 2


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("tag","calibration-result","calibration-audit","calibration-cost","calibration-cost-audit","report","report-manifest","report-audit"):parser.add_argument("--"+key,required=True)
    raise SystemExit(execute(parser.parse_args()))
