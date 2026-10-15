"""Selected static/ramp whole cost: live frozen F or past-only FLAT, no cache return."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import shutil
import sys

import run_linear_e_cost_v1 as old

HERE,ROOT,BASE,OLD=old.HERE,old.ROOT,old.BASE,old.OLD
require,record,read,bound,byte_equal=old.require,old.record,old.read,old.bound,old.byte_equal
DATASETS={"robin":(17,4),"beijing_aotizhongxin":(11,2)}
LAGS=(0,1,2,4,8,16,32)
SCIENCE_SHA="5f158382640dac74ec63cc3d1005b385c898821d3c032d51b543f2c9b27d7dac"
AUDIT_SHA="0f65053536403a3b7b8f53fca3b9f8274fb1a406eaf6dfdcb73bb0467fffe698"
AUDITOR_SOURCE={"path":str(HERE/"audit_horizon_decoder_diagnostic_v1.py"),"sha256":"accb6fb775d9b12ca6e39c0be7e062545b98d429089ee32e0ac46ed65edae613"}
AUDIT_COUNTS={"decoder_fit_paths_checked":24,"observed_mask_groups_checked":96,"alpha_candidates_checked":240,"alpha_zero_candidates_checked":24,"positive_alpha_candidates_checked":216,"ZERO_candidates_checked":24,"FIT_TUNE_prediction_evaluations_checked":528,"selected_DEV_prediction_evaluations_checked":24,"selected_coefficient_states_checked":24,"selected_fits_independently_solved":22,"selected_ZERO_states_checked":2,"selected_FP32_arithmetic_predictions_checked":72,"selection_locks_checked":24,"source_contexts_checked":18,"shared_RMS_spaces_checked":6,"latent_point_diagnostic_modes_checked":36,"score_fields_checked":14544,"temporal_score_fields_checked":33048,"protected_source_input_and_result_hashes_checked":2359}
SCIENCE_CONTROLLER_SHA="151da5b0b4cc4cacdb9d7a547d12e7837dcdcb5ee4c5047a1d6e3993afe525c6"
MODULE_SHA="406252e940ae676e7a78f69e067f5b3fa70456bdb1ea27d57169b8964ceabe0e"


def load_gates(science_path,audit_path):
    require(all(isinstance(v,str) and len(v)==64 for v in (SCIENCE_SHA,AUDIT_SHA)) and isinstance(AUDITOR_SOURCE,dict) and bool(AUDIT_COUNTS),"ACTUAL_HORIZON_SCIENCE_AUDIT_PASS_NOT_REGISTERED")
    refs={"horizon_decoder_diagnostic":record(science_path),"horizon_decoder_audit":record(audit_path)};require([p["sha256"] for p in refs.values()]==[SCIENCE_SHA,AUDIT_SHA],"EXACT_HORIZON_SCIENCE_AND_PASS")
    source,audit=read(science_path),read(audit_path)
    require(source["status"]=="COMPLETE" and audit["status"]=="PASS" and not source["failed_paths"] and not audit.get("partial_paths",[]) and all(type(audit["totals"].get(k)) is int and audit["totals"][k]==v for k,v in AUDIT_COUNTS.items()),"COMPLETE_HORIZON_SCIENCE_AND_EXACT_AUDIT_COVERAGE")
    bound(audit["diagnostic"],refs["horizon_decoder_diagnostic"]);bound(audit["audit_source"],AUDITOR_SOURCE);bound(audit["selection_registry"],source["selection_registry"])
    require(source["controller_source"]["sha256"]==SCIENCE_CONTROLLER_SHA and source["module_source"]["sha256"]==MODULE_SHA,"EXECUTED_SCIENCE_SOURCES");bound(source["controller_source"]);bound(source["module_source"])
    for field,count in {"fits_completed":24,"alpha_candidates_completed":240,"ZERO_candidates_completed":24,"candidate_FIT_TUNE_evaluations_completed":528,"selected_DEV_evaluations_completed":24,"resolved_selection_locks":24}.items():require(source[field]==count,"EXACT_SCIENCE_COUNT:"+field)
    require(source["all_24_locks_before_DEV"] is True and source["new_F_forward_calls"]==source["new_F_model_objects_loaded"]==source["new_optimizer_updates"]==0 and source["protected_TEST_accessed"] is audit["protected_TEST_accessed"] is source["whole_deployment_cost_measured"] is False,"ACCEPTED_SOURCE_SCOPE")
    import run_horizon_decoder_diagnostic_v1 as scientific
    cross,science,parentrefs=scientific.load_gates(SimpleNamespace(cross_result=source["cross_moment_diagnostic"]["path"],cross_audit=source["cross_moment_audit"]["path"]))
    for key,item in parentrefs.items():bound(item,source[key])
    subset=read(bound(science["raw_subset_diagnostic"]));original=read(bound(subset["source_producer"]));full=read(bound(source["full_gradient_parent"]))
    require(source["F_parent_signatures_reused"]=={"before":cross["F_before"],"after":cross["F_after"]} and cross["F_before"]==cross["F_after"]==original["F_before"]==original["F_after"],"ONE_FROZEN_PARENT_F")
    registry=read(bound(source["selection_registry"]));require(registry["status"]=="ALL_24_RESOLVED_BEFORE_DEV" and registry["DEV_opened_before_seal"] is False and len(registry["conditions"])==24,"TWENTY_FOUR_LOCK_REGISTRY");locks={}
    for item in registry["conditions"]:
        data,space,mode,arm=(item[k] for k in ("dataset","space","mode","arm"));key=data+"/"+space+"_"+mode+"_"+arm;require((data,space,mode,arm) in scientific.CONDITIONS and key not in locks and item["status"]=="COMPLETE","EXACT_UNIQUE_SELECTION_KEY")
        row=source["datasets"][data]["spaces"][space]["conditions"][mode+"_"+arm];lock=read(bound(item["selection_lock"],row["selection_lock"]))
        require(lock["status"]==row["status"]=="COMPLETE" and lock["DEV_opened_before_lock"] is False and all(lock[k]==row[k] for k in ("selected_candidate","selected_kind","selected_alpha","selected_TUNE_MSE","selected_coefficient_value_sha256")),"EXACT_TUNE_ONLY_CORRECTION_LOCK");bound(lock["selected_coefficients"],row["selected_coefficients"]);locks[key]=item["selection_lock"]
    refs.update(parentrefs,selection_registry=source["selection_registry"],cost_plan=record(HERE/"PLAN_HORIZON_DECODER_WHOLE_COST_KO_v1.md"));return source,cross,science,subset,original,full,refs,locks


def forecast_components(history,space,mode,variant,model,bolt,tensor,coefficient=None,selected_kind=None,*,capture=False):
    import numpy as np
    import torch
    import horizon_decoder_delta_v1 as module
    from causal_lag_decoder_v1 import lagged_delta_features
    require(mode in ("F","FLAT") and variant in ("BASELINE","STATIC_DELTA","RAMP_DELTA"),"REGISTERED_COST_MODE_AND_VARIANT")
    active=variant!="BASELINE" and selected_kind!="ZERO_DELTA";need_latents=active or capture;z=f=features=None
    with torch.no_grad():
        if space=="RAW_C":
            future=np.ascontiguousarray(bolt(tensor(history)).detach().cpu().numpy().copy(),dtype=np.float32) if mode=="F" else np.broadcast_to(history[:,-1:],(len(history),48,history.shape[2])).copy()
            baseline=future.copy()
            if need_latents:z=history;f=future;features=lagged_delta_features(z,f,LAGS)
        else:
            x=tensor(history);past=model.encode(x);future=bolt(past) if mode=="F" else past[:,-1:].expand(-1,48,-1)
            baseline=np.ascontiguousarray(model.decode(x[:,-1],past,future).detach().cpu().numpy().copy(),dtype=np.float32)
            if need_latents:
                z=np.ascontiguousarray(past.detach().cpu().numpy().copy(),dtype=np.float32);f=np.ascontiguousarray(future.detach().cpu().numpy().copy(),dtype=np.float32);features=lagged_delta_features(z,f,LAGS)
    prediction=module.predict(baseline,features,coefficient,variant) if active else baseline.copy()
    require(prediction.dtype==np.float32 and prediction.shape==(len(history),48,history.shape[2]) and np.isfinite(prediction).all(),"FINITE_ACTUAL_B1_B8_CORRECTION_OUTPUT")
    return {"prediction":prediction,"baseline":baseline,"features":features,"latent_history":z,"latent_future":f} if capture else prediction


def predictor(space,mode,variant,model,bolt,tensor,coefficient=None,selected_kind=None):
    def callback(history,history_observed):return forecast_components(history,space,mode,variant,model,bolt,tensor,coefficient,selected_kind)
    return callback


def physical_precheck(physical,raw,reference,space,mode,variant,model,bolt,tensor,coefficient,selected_kind,source_stream):
    import numpy as np
    import whole_horizon_decoder_cost_v1 as wc
    history=wc.preprocess(physical["history_physical"][:8],physical["history_observed"][:8],raw["mean"],raw["std"])
    actual=forecast_components(history,space,mode,variant,model,bolt,tensor,coefficient,selected_kind,capture=True);expected=reference[:8]
    exact={"physical_raw_origins":byte_equal(physical["origins"][:8],raw["origins"][:8]),"physical_raw_mask":byte_equal(physical["history_observed"][:8],raw["history_observed"][:8]),"normalized_history_byte_exact":byte_equal(history,raw["history"][:8])}
    checks={"normalized":wc.parity(actual["prediction"],expected),"physical":wc.parity((actual["prediction"]*raw["std"]+raw["mean"]).astype(np.float32),(expected*raw["std"]+raw["mean"]).astype(np.float32),wc.PHYSICAL_ATOL,wc.PHYSICAL_RTOL)}
    intermediate={key:wc.parity(actual[key],source_stream[key][:8]) for key in ("baseline","features","latent_history","latent_future")}
    actual.update(normalized_history=history,history_observed=physical["history_observed"][:8].copy(),origins=raw["origins"][:8].copy(),scientific_normalized=expected.copy())
    return {"exact_lineage":exact,"input_backend_pass":all(exact.values()),"prediction_backend_pass":all(p["pass"] for p in checks.values()),"prediction_checks":checks,"intermediate_checks":intermediate,"intermediate_parity_required_for_acceptance":False,"capture_diagnostics_and_extra_host_transfers_outside_timer":True,"scope":"Physical first8 and selected canonical32 normalized/physical parity; live F for F, no F for FLAT; exact source/state/coef/origin binding"},actual


def execute(args):
    source,cross,science,subset,original,full,refs,locks=load_gates(args.science_result,args.science_audit);output=old.fresh_output(args.tag);sys.path[:0]=[str(HERE),str(BASE),str(OLD),str(ROOT)]
    import numpy as np
    import torch
    import psutil
    from threadpoolctl import threadpool_limits
    import continuation_runtime_v1 as rt
    import fixed_decoder_span_v1 as io
    import whole_horizon_decoder_cost_v1 as wc
    import run_condition_probes as probes
    import cost_readout_v1 as runtime
    import run_horizon_decoder_diagnostic_v1 as scientific
    from linear_e_adaptation_v1 import LinearLagAdapter
    from prepare_inputs import install_protected_test_guard
    from run_fit_target_comparison_v1 import prior_file_records
    require(wc.REPEATS==5 and wc.WARMUP==2 and all(len(wc.routes_for(d))==18 for d in DATASETS),"THIRTY_SIX_ROUTES_SEVENTY_TWO_CELLS_360_SAMPLES")
    output.mkdir(parents=True,exist_ok=False);meter=rt.Meter("horizon_decoder_cost_"+args.tag,gpu=True);failure=None
    result={"status":"RUNNING",**refs,"selection_locks":locks,"cost_split":"tune","routes":{d:list(wc.routes_for(d)) for d in DATASETS},"cost":{},"datasets":{},"source_snapshots":[],"failed_paths":[],"new_training_updates":0,"new_encoder_decoder_fit":False,"new_alpha_arm_or_K_selection":False,"new_backbone_training":False,"cached_future_returned_instead_of_actual_F_inside_timer":False,"selected_state_and_coefficients_setup_outside_timer":True,"all_source_file_state_and_coefficient_hashes_outside_timer":True,"correction_backend":"CPU FP32 canonical outerbatch up to32 fused slope; active PCA correction materializes CUDA past/future to CPU and computes CPU delta features; host transfer timed","baseline_backend":"PCA CUDA E and original model.decode; live CUDA frozen F for F / no F for FLAT; RawF direct model output, RawFLAT currentC; B1/B8 no padding","ZERO_selected_skips_coefficient_matmul_and_extra_latent_host_transfer":True,"timed_internal_shape_finite_guards":True,"external_meter_and_hash_guards_outside_timer":True,"protected_TEST_accessed":False,"independent_holdout":False,"strict_K_only":False,"cost_origins":"Original physical first B1/B8 TUNE; no DEV timing or inverse-normalized synthetic physical input","capacity_match_RAW_and_PCA_claimed":False}
    checked,models,coefficients,prepared={}, {}, {}, {};protected=[]
    def load(item):return io.checked_payload({**item,"path":str(bound(item))},meter,checked)
    def fail(error,**where):
        item={"status":"FAILED_PATH_PRESERVED",**where,"failure":{"type":type(error).__name__,"message":str(error)}};result["failed_paths"].append(item);return item
    try:
        install_protected_test_guard();meter.protected_check();protected=prior_file_records(source,cross,science,subset,original,full,refs);folder=output/"sources";folder.mkdir()
        for path in (Path(__file__),HERE/"whole_horizon_decoder_cost_v1.py",HERE/"whole_f_input_gradient_cost_v1.py",HERE/"run_linear_e_cost_v1.py",HERE/"run_horizon_decoder_diagnostic_v1.py",HERE/"horizon_decoder_delta_v1.py",HERE/"causal_lag_decoder_v1.py",HERE/"linear_e_adaptation_v1.py",HERE/"continuation_runtime_v1.py",HERE/"fixed_decoder_span_v1.py",HERE/"PLAN_HORIZON_DECODER_WHOLE_COST_KO_v1.md",OLD/"prepare_inputs.py",OLD/"run_condition_probes.py",BASE/"cost_readout_v1.py"):
            destination=folder/path.name;shutil.copyfile(path,destination);result["source_snapshots"].append({**record(path),"snapshot_path":str(destination)})
        result.update(controller_source=record(__file__),whole_cost_source=record(wc.__file__),science_controller_source=source["controller_source"],delta_module_source=source["module_source"])
        engine=probes.import_old_engine();weight,config=(probes.verified_file(original[k]) for k in ("model_binary_before","model_config_before"));bolt=engine.ex.previous.FrozenBolt();result.update(F_before=probes.model_hashes(bolt),model_binary_before=weight,model_config_before=config)
        require(result["F_before"]==cross["F_before"]==cross["F_after"] and all(not p.requires_grad and p.grad is None for p in bolt.model.parameters()),"ACTUAL_FROZEN_F_DUAL_SIGNATURE")
        result["runtime_behavior_contract"]=runtime.bolt_runtime_contract(engine,bolt,meter.check);require(result["runtime_behavior_contract"]==cross["runtime_behavior_contract"]==full["runtime_behavior_contract"],"SAME_MEDIAN_F_RUNTIME")
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        with threadpool_limits(limits=2):
            for data,(c,_) in DATASETS.items():
                try:
                    raw_source=science["datasets"][data]["spaces"]["PCA"]["conditions"]["L7_RECON"]["splits"]["tune"]["input_source"];raw=load(raw_source);proof=old.canonical_tune_bridge(original,data);physical=load(proof["canonical_record"]);require(raw["history"].shape==(64,512,c),"ORIGINAL_TUNE64_COHORT")
                    for key in ("history","history_observed","target","observed","raw_last","origins","mean","std","raw_forecast"):require(byte_equal(raw[key],physical[key]),"FULL_CANONICAL_PHYSICAL_RAW_BINDING:"+key)
                    p=prepared[data]={"raw":raw,"physical":physical,"predictors":{},"references":{},"metadata":{},"unavailable":{}};result["datasets"][data]={"canonical_physical_bridge":proof,"raw_TUNE_source":raw_source,"physical_prechecks":{}}
                except Exception as error:result["datasets"][data]=fail(error,dataset=data,stage="COMMON_PHYSICAL_ASSET");continue
                def unavailable(route,error,stage):
                    item=fail(error,dataset=data,route=route,stage=stage);p["predictors"].pop(route,None);p["references"].pop(route,None);p["unavailable"][route]=item;p["metadata"].setdefault(route,{"unavailable":item})
                for space in wc.spaces_for(data):
                    model=cached=frozen=None;state_record=cache_record=None
                    try:
                        owner=source["datasets"][data]["spaces"][space];context=owner["contexts"]["tune"];bound(context["input_source"],raw_source)
                        if space!="RAW_C":
                            accepted=cross["datasets"][data]["conditions"][space];state_record=accepted["state_artifact"];cache_record=accepted["splits"]["tune"]["artifact"];bound(context["frozen_state_source"],state_record);bound(context["cached_latent_source"],cache_record);frozen=load(state_record);cached=load(cache_record);require(byte_equal(cached["origins"],raw["origins"]),"SAME_PCA_TUNE_ORIGINS")
                            model=LinearLagAdapter(frozen["center"],frozen["E"],frozen["D"],LAGS,learn_E=False).to("cuda").eval().requires_grad_(False);require(all(byte_equal(model.state_dict()[k].detach().cpu().numpy(),frozen[k]) for k in frozen) and all(not p.requires_grad and p.grad is None for p in model.parameters()),"UNCHANGED_FROZEN_E_D_CENTER");models[data+"/"+space]=model
                        streams={mode:scientific.stream_payload(raw["history"],raw["raw_last"],raw["raw_forecast"],space,mode,cached,frozen) for mode in ("F","FLAT")}
                    except Exception as error:
                        for mode in ("F","FLAT"):
                            for variant in ("BASELINE","STATIC_DELTA","RAMP_DELTA"):unavailable(space+"_"+mode+"_"+variant,error,"FROZEN_SPACE_SOURCE")
                        continue
                    for mode in ("F","FLAT"):
                        for variant in ("BASELINE","STATIC_DELTA","RAMP_DELTA"):
                            route=space+"_"+mode+"_"+variant
                            try:
                                row=None;coefficient=kind=None;selection={};reference=streams[mode]["baseline"];reference_record=raw_source if space=="RAW_C" else cache_record;reference_member=("raw_forecast" if mode=="F" else "currentC_broadcast") if space=="RAW_C" else ("prediction" if mode=="F" else "FLAT_prediction")
                                if variant!="BASELINE":
                                    row=owner["conditions"][mode+"_"+variant];coefficient=load(row["selected_coefficients"]);require(set(coefficient)=={"coefficient_FP64","coefficient_FP32"} and old.byte_equal(coefficient["coefficient_FP32"],coefficient["coefficient_FP64"].astype(np.float32)) and scientific.parent.coefficient_records(coefficient)[1]==row["selected_coefficient_value_sha256"],"EXACT_SELECTED_FUSED_CORRECTION_STATE")
                                    coefficients[data+"/"+route]=coefficient;kind=row["selected_kind"];reference_record=row["selected_FIT_TUNE_prediction_artifact"];saved=load(reference_record);require(byte_equal(saved["TUNE_origins"],raw["origins"]),"SAME_SELECTED_TUNE_ORIGINS");reference=saved["TUNE_prediction_FP32"];reference_member="TUNE_prediction_FP32";selection={"selected_coefficients":row["selected_coefficients"],"selected_candidate":row["selected_candidate"],"selected_kind":kind,"selected_alpha":row["selected_alpha"],"coefficient_value_sha256":row["selected_coefficient_value_sha256"],"selection_lock":locks[data+"/"+route]}
                                p["predictors"][route]=predictor(space,mode,variant,model,bolt,engine.tensor,coefficient,kind);p["references"][route]=reference;p["metadata"][route]={"space":space,"mode":mode,"variant":variant,"reference":reference_record,"reference_prediction_array":reference_member,"reference_split":"tune","frozen_state_source":state_record,"lags":list(LAGS),"actual_F_forward":mode=="F","no_F_for_FLAT":mode=="FLAT","ZERO_selected_baseline_shortcut":kind=="ZERO_DELTA","matched_baseline_route":wc.matched_baseline(route),"raw_F_baseline_reassociated":False,"capacity_match_RAW_and_PCA_claimed":False,**selection}
                                check,arrays=physical_precheck(physical,raw,reference,space,mode,variant,model,bolt,engine.tensor,coefficient,kind,streams[mode]);check["artifact"]=io.save_npz(output/(data+"_"+route+"_PHYSICAL_FIRST8.npz"),arrays);result["datasets"][data]["physical_prechecks"][route]=check;require(check["input_backend_pass"] and check["prediction_backend_pass"],"REGISTERED_PHYSICAL_INPUT_AND_SELECTED_PREDICTION_PARITY")
                            except Exception as error:unavailable(route,error,"SELECTED_CORRECTION_OR_PHYSICAL_PRECHECK")
            result["frozen_E_D_before"]=old.adapter_hashes(models);result["selected_coefficients_before"]={key:scientific.parent.coefficient_records(v)[1] for key,v in coefficients.items()};process=psutil.Process()
            def memory():return {"RSS_bytes":process.memory_info().rss,"allocated_VRAM_bytes":torch.cuda.memory_allocated(),"reserved_VRAM_bytes":torch.cuda.memory_reserved(),"max_allocated_VRAM_bytes":torch.cuda.max_memory_allocated(),"max_reserved_VRAM_bytes":torch.cuda.max_memory_reserved(),"scope":"Shared frozen F/four original PCA E-D states/24 selected CPU correction coefficient states; no exclusive method allocation"}
            for data,p in prepared.items():
                try:result["cost"][data]=wc.measure_cost(data,p["physical"]["history_physical"],p["physical"]["history_observed"],p["raw"]["mean"],p["raw"]["std"],p["raw"]["origins"],p["predictors"],p["references"],p["metadata"],output/("cost_"+data),meter.check,torch.cuda.synchronize,memory,p["unavailable"]);result["failed_paths"] += [{"dataset":data,**v} for v in result["cost"][data]["failed_paths"]]
                except Exception as error:
                    fail(error,dataset=data,stage="WHOLE_COST");partial=output/("cost_"+data)/"RESULT.json"
                    if partial.exists():result["cost"][data]=read(partial)
        result["F_after"]=probes.model_hashes(bolt);result["frozen_E_D_after"]=old.adapter_hashes(models);result["selected_coefficients_after"]={key:scientific.parent.coefficient_records(v)[1] for key,v in coefficients.items()};require(result["F_before"]==result["F_after"] and all(result[k+"_before"]==result[k+"_after"] for k in ("frozen_E_D","selected_coefficients")),"F_E_D_CENTER_AND_COEFFICIENTS_IMMUTABLE")
        result["model_binary_after"],result["model_config_after"]=probes.verified_file(weight),probes.verified_file(config)
        for item in protected+list(checked.values())+list(locks.values()):bound(item)
        for item in result["source_snapshots"]:require(record(item["path"])["sha256"]==record(item["snapshot_path"])["sha256"]==item["sha256"],"EXECUTED_SOURCES_UNCHANGED")
        rows=[row for cost in result["cost"].values() for row in cost["rows"]];complete=sum(row["status"]=="COMPLETE" for row in rows);samples=sum(len(row["samples"]) for row in rows);prechecks=sum(v.get("input_backend_pass",False) and v.get("prediction_backend_pass",False) for d in result["datasets"].values() for v in d.get("physical_prechecks",{}).values())
        result.update(completed_cost_cells=complete,recorded_timer_samples=samples,expected_cost_cells=72,expected_timer_samples=360,frozen_PCA_models_loaded=len(models),selected_correction_states_loaded=len(coefficients),expected_physical_prechecks=36,expected_physical_first8_windows=288,physical_prechecks_completed=prechecks,input_records=list(checked.values()),protected_prior_records=protected,status="COMPLETE" if complete==72 and samples==360 and prechecks==36 and not result["failed_paths"] else "PARTIAL_COMPLETED");meter.protected_check();json.dumps(result,allow_nan=False)
    except Exception as error:failure={"type":type(error).__name__,"message":str(error)};result.update(status="FAILED_PRESERVED",failure=failure)
    finally:result["resource_terminal"]=meter.close(result["status"],failure);rt.write_new(output/"RESULT.json",result)
    print(json.dumps({"status":result["status"],"result":str(output/"RESULT.json")}),flush=True);return 0 if result["status"]=="COMPLETE" else 2


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("tag","science-result","science-audit"):parser.add_argument("--"+key,required=True)
    raise SystemExit(execute(parser.parse_args()))
