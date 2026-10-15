"""New TUNE whole-cost cohort: frozen F/native/D and direct-C fused ridge, no cache return."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import shutil
import sys

import run_linear_e_cost_v1 as old
from run_linear_e_cost_v1 import canonical_tune_bridge,adapter_predictor,physical_precheck

HERE,ROOT,BASE,OLD=old.HERE,old.ROOT,old.BASE,old.OLD
require,record,read,bound,byte_equal=old.require,old.record,old.read,old.bound,old.byte_equal
DATASETS={"robin":(17,4),"beijing_aotizhongxin":(11,2)}
LAGS=(0,1,2,4,8,16,32)
NATIVE_RESULT_SHA="786edbd75a35d2c5a760a7b4f5af5a1389ee1bc4bc93594d7b359a29ecfd83ed"
NATIVE_AUDIT_SHA="51e438d7544bb5c52d2a6ea3ddd2da9c38773b2aa42286cf3a77c67d405d7bf6"
RIDGE_RESULT_SHA="b8d7eae62ce5bb52b3d46ffc543f916f8b81857028098370a5b2f8398510f1ed"
RIDGE_AUDIT_SHA="8bfeb29cc6836d2ff5528c30ee13a4300b79cc4c250a452b9c5c81fdfc9bac90"
NATIVE_AUDITOR_SOURCE={"path":str(HERE/"audit_new_direction_native_v1.py"),"sha256":"6654e9dc8680c827ca9626aa6dd97cfe9d199a4c71017b239b7d6cd8e4f78fb5"}
RIDGE_AUDITOR_SOURCE={"path":str(HERE/"audit_new_direction_direct_ridge_v1.py"),"sha256":"d8b6e8b8ed6862af69e78ecc63758b13fd03378862aa6c6c6644052fa36d85d4"}
NATIVE_AUDIT_COUNTS={"native_fits_checked":16,"logged_updates_checked":9600,"checkpoint_states_checked":80,"checkpoint_TUNE_predictions_checked":80,"selected_FIT_predictions_checked":16,"selected_TUNE_entries_reused_checked":16,"selected_DEV_predictions_checked":16,"CP0_probe_cases_checked":16,"cached_actual_F_context_lineages_checked":24,"independent_NumPy_head_and_old_D_cells_checked":112,"frozen_CP_member_byte_comparisons_checked":240,"native_cached_Torch_history_byte_comparisons_checked":112,"training_score_fields_checked":8064,"baseline_score_fields_checked":1152,"produced_NPZ_artifacts_checked":320,"same_parent_seed_orders_checked":16,"frozen_E_D_center_states_checked":8,"selection_locks_checked":16,"parent_actual_F_calls_registered":104,"protected_prior_file_hashes_checked":2017,"new_output_space_diagnostics_computed":0}
RIDGE_AUDIT_COUNTS={"selected_ridge_fits_checked":12,"mask_groups_checked":1319,"alpha_candidates_checked":84,"FIT_TUNE_prediction_evaluations_checked":168,"selected_DEV_prediction_evaluations_checked":12,"selected_coefficient_FP32_arithmetic_predictions_checked":36,"selection_locks_checked":12,"score_fields_checked":4320,"selected_upper_grid_boundary_conditions":10,"protected_source_input_and_result_hashes_checked":2155}
NATIVE_CONTROLLER_SHA="4aaf63c4ae078daed71f4b11540d4ade8ee62ffce6a8345bd031fe162b9062e4"
RIDGE_CONTROLLER_SHA="7dd19e1504f0ff4b323c52412fac492f2b5c17c23e1ba0d898691a672f5ac88e"
RIDGE_MODULE_SHA="967fb95eef55f9597a73a01efc0118d8c7e936586adb724f7c5a449334ab3717"


def load_gates(native_path,native_audit_path,ridge_path,ridge_audit_path):
    digests=(NATIVE_RESULT_SHA,NATIVE_AUDIT_SHA,RIDGE_RESULT_SHA,RIDGE_AUDIT_SHA)
    require(all(isinstance(v,str) and len(v)==64 for v in digests) and isinstance(NATIVE_AUDITOR_SOURCE,dict) and isinstance(RIDGE_AUDITOR_SOURCE,dict) and bool(NATIVE_AUDIT_COUNTS) and bool(RIDGE_AUDIT_COUNTS),"ACTUAL_NATIVE_AND_RIDGE_INDEPENDENT_PASS_NOT_YET_REGISTERED")
    refs={key:record(path) for key,path in (("new_direction_native_diagnostic",native_path),("new_direction_native_audit",native_audit_path),("new_direction_ridge_diagnostic",ridge_path),("new_direction_ridge_audit",ridge_audit_path))}
    require(tuple(v["sha256"] for v in refs.values())==digests,"EXACT_NEW_NATIVE_RIDGE_RESULTS_AND_PASSES")
    native,na,ridge,ra=(read(path) for path in (native_path,native_audit_path,ridge_path,ridge_audit_path))
    for producer,audit,key,source,counts in ((native,na,"new_direction_native_diagnostic",NATIVE_AUDITOR_SOURCE,NATIVE_AUDIT_COUNTS),(ridge,ra,"new_direction_ridge_diagnostic",RIDGE_AUDITOR_SOURCE,RIDGE_AUDIT_COUNTS)):
        require(producer["status"]=="COMPLETE" and audit["status"]=="PASS" and producer["protected_TEST_accessed"] is audit["protected_TEST_accessed"] is False and not producer["failed_paths"] and not audit.get("partial_paths",[]) and all(type(audit["totals"].get(k)) is int and audit["totals"][k]==v for k,v in counts.items()),"COMPLETE_NEW_SCIENCE_AND_EXACT_AUDIT_COVERAGE")
        bound(audit["diagnostic"],refs[key]);bound(audit["audit_source"],source);bound(producer["controller_source"]);bound(audit["selection_registry"],producer["selection_registry"])
    require(native["controller_source"]["sha256"]==NATIVE_CONTROLLER_SHA and ridge["controller_source"]["sha256"]==RIDGE_CONTROLLER_SHA and ridge["module_source"]["sha256"]==RIDGE_MODULE_SHA,"EXACT_EXECUTED_NEW_NATIVE_RIDGE_SOURCES")
    bound(ridge["module_source"])
    require(native["native_training_fits"]==native["resolved_selection_locks"]==16 and native["training_updates_completed"]==9600 and native["checkpoint_states_saved"]==80 and native["DEV_evaluations_completed"]==16 and native["all_16_locks_before_DEV"] is True and native["cached_actual_F_contexts_reused"]==24,"SIXTEEN_SELECTED_NEW_NATIVE_HEADS")
    require(ridge["ridge_fits_completed"]==ridge["resolved_selection_locks"]==12 and ridge["alpha_candidates_completed"]==84 and ridge["FIT_TUNE_prediction_evaluations_completed"]==168 and ridge["selected_DEV_evaluations_completed"]==12 and ridge["all_12_locks_before_DEV"] is True,"TWELVE_SELECTED_SEVEN_ALPHA_DIRECT_RIDGES")
    for item in (native,ridge):
        require(item["new_F_forward_calls"]==item["new_F_model_objects_loaded"]==0 and item["new_backbone_training"] is False and item["whole_deployment_cost_measured"] is False,"SCIENCE_CACHE_REUSE_NOT_PREVIOUS_COST")
    import run_new_direction_native_v1 as ns
    import run_new_direction_direct_ridge_v1 as rs
    cross=read(bound(native["predictive_cross_diagnostic"]));bound(native["predictive_cross_diagnostic"],ridge["cross_moment_diagnostic"]);bound(native["predictive_cross_audit"],ridge["cross_moment_audit"])
    args=SimpleNamespace(cross_result=native["predictive_cross_diagnostic"]["path"],cross_audit=native["predictive_cross_audit"]["path"],**{argument:cross[key]["path"] for argument,key in (("calibration_result","latent_calibration_diagnostic"),("calibration_audit","latent_calibration_audit"),("calibration_cost","latent_calibration_cost"),("calibration_cost_audit","latent_calibration_cost_audit"),("report","parent_report"),("report_manifest","parent_report_manifest"),("report_audit","parent_report_audit"))})
    cross,calibration,previous,science,full,parentrefs=ns.load_gates(args);ridge_cross,ridge_science,_=rs.load_gates(args)
    require(cross==ridge_cross and science==ridge_science,"SAME_AUDITED_CROSS_STATES_AND_INPUTS_FOR_NATIVE_RIDGE")
    subset=read(bound(science["raw_subset_diagnostic"]));original=read(bound(subset["source_producer"]))
    require(native["F_parent_signatures_reused"]==ridge["F_parent_signatures_reused"]=={"before":cross["F_before"],"after":cross["F_after"]} and cross["F_before"]==cross["F_after"]==original["F_before"]==original["F_after"],"ONE_FROZEN_F_DUAL_SIGNATURE")
    refs.update(predictive_cross_diagnostic=native["predictive_cross_diagnostic"],predictive_cross_audit=native["predictive_cross_audit"],cross_state_registry=cross["state_registry"],native_selection_registry=native["selection_registry"],ridge_selection_registry=ridge["selection_registry"],cost_plan=record(HERE/"PLAN_NEW_DIRECTION_NATIVE_BASELINES_KO_v1.md"))
    locks={"native":{},"ridge":{}}
    for family,producer,status,total in (("native",native,"ALL_SIXTEEN_RESOLVED_BEFORE_DEV",16),("ridge",ridge,"ALL_TWELVE_RESOLVED_BEFORE_DEV",12)):
        registry=read(bound(producer["selection_registry"]));require(registry["status"]==status and registry["DEV_opened_before_seal"] is False and len(registry["conditions"])==total,"EXACT_SELECTION_REGISTRY_BEFORE_TIMING")
        for item in registry["conditions"]:
            data=item["dataset"];require(data in DATASETS and item["status"]=="COMPLETE","COMPLETE_REGISTERED_DATASET_LOCK")
            if family=="native":
                name,seed=item["condition"],item["seed"];row=producer["datasets"][data]["conditions"][name]["native"][str(seed)];key=data+"/"+name+"_NATIVE_s"+str(seed)
            else:row=producer["datasets"][data]["routes"][item["route"]];key=data+"/"+item["route"]
            require(key not in locks[family],"UNIQUE_SELECTION_LOCKS");lock=read(bound(item["selection_lock"],row["selection_lock"]));require(lock["status"]=="COMPLETE" and lock["DEV_opened_before_lock"] is False,"SEALED_TUNE_ONLY_SELECTED_LOCK")
            field="selected_state" if family=="native" else "selected_coefficients";bound(lock[field],row[field]);require(lock["selected_update" if family=="native" else "selected_alpha"]==row["selected_update" if family=="native" else "selected_alpha"],"EXACT_SELECTED_CP_OR_ALPHA");locks[family][key]=item["selection_lock"]
    return native,ridge,cross,science,subset,original,full,refs,locks


def ridge_route(route):
    if route in ("RAW_CURRENT_C_RIDGE","RAW_HISTORY512_RIDGE"):return route.removesuffix("_RIDGE")
    require(route.endswith("_RIDGE_HISTORY512"),"REGISTERED_RIDGE_COST_ROUTE");return route.removesuffix("_RIDGE_HISTORY512")+"_HISTORY512"


def ridge_components(history,route,state,encoder=None):
    import run_new_direction_direct_ridge_v1 as source
    from run_predictive_cross_moment_v1 import encode_torch
    z=encode_torch(encoder,history) if encoder is not None else None
    feature,anchor=source.feature_payload({"history":history,"raw_last":history[:,-1].copy()},ridge_route(route),z)
    prediction=source.predict(feature,state,anchor)
    return {"prediction":prediction,"features":feature,"anchor":anchor,**({"latent_history":z} if z is not None else {})}


def ridge_predictor(route,state,encoder=None):
    def callback(history,history_observed):return ridge_components(history,route,state,encoder)["prediction"]
    return callback


def ridge_physical_precheck(physical,raw,reference,state,route,encoder=None,encoded_reference=None):
    import numpy as np
    import whole_new_direction_cost_v1 as wc
    import run_new_direction_direct_ridge_v1 as source
    history=wc.preprocess(physical["history_physical"][:8],physical["history_observed"][:8],raw["mean"],raw["std"])
    arrays=ridge_components(history,route,state,encoder);expected=reference["TUNE_prediction_FP32"][:8]
    feature,anchor=source.feature_payload({"history":raw["history"][:8],"raw_last":raw["raw_last"][:8]},ridge_route(route),None if encoded_reference is None else encoded_reference["latent_history"][:8])
    exact={"physical_raw_origins":byte_equal(physical["origins"][:8],raw["origins"][:8]),"selected_raw_origins":byte_equal(reference["TUNE_origins"][:8],raw["origins"][:8]),"physical_raw_mask":byte_equal(physical["history_observed"][:8],raw["history_observed"][:8]),"normalized_history_byte_exact":byte_equal(history,raw["history"][:8]),"anchor_byte_exact":byte_equal(arrays["anchor"],anchor)}
    checks={"normalized":wc.parity(arrays["prediction"],expected),"physical":wc.parity((arrays["prediction"]*raw["std"]+raw["mean"]).astype(np.float32),(expected*raw["std"]+raw["mean"]).astype(np.float32),wc.PHYSICAL_ATOL,wc.PHYSICAL_RTOL)}
    intermediate={"features":wc.parity(arrays["features"],feature)}
    if encoder is not None:intermediate["latent_history"]=wc.parity(arrays["latent_history"],encoded_reference["latent_history"][:8])
    arrays.update(normalized_history=history,history_observed=physical["history_observed"][:8].copy(),origins=raw["origins"][:8].copy(),scientific_normalized=expected.copy(),scientific_features=feature,scientific_anchor=anchor)
    return {"exact_lineage":exact,"input_backend_pass":all(exact.values()),"prediction_backend_pass":all(v["pass"] for v in checks.values()),"prediction_checks":checks,"intermediate_checks":intermediate,"intermediate_parity_required_for_acceptance":False,"scope":"Physical first8; same fused FP32 CPU direct readout; encoded CUDA E and host transfer; D bypassed; no future target/mask passed to prediction"},arrays


def execute(args):
    native,ridge,cross,science,subset,original,full,refs,locks=load_gates(args.native_result,args.native_audit,args.ridge_result,args.ridge_audit)
    output=old.fresh_output(args.tag);sys.path[:0]=[str(HERE),str(BASE),str(OLD),str(ROOT)]
    import numpy as np
    import torch
    import psutil
    from threadpoolctl import threadpool_limits
    import continuation_runtime_v1 as rt
    import fixed_decoder_span_v1 as io
    import whole_new_direction_cost_v1 as wc
    import run_condition_probes as probes
    import cost_readout_v1 as runtime
    import run_new_direction_direct_ridge_v1 as ridge_source
    from linear_e_adaptation_v1 import LinearLagAdapter
    from lagged_native_head_v1 import LaggedNativeForecaster
    from native_head_v1 import state_hash
    from prepare_inputs import install_protected_test_guard
    from run_fit_target_comparison_v1 import prior_file_records
    require(set(wc.DATASETS)==set(DATASETS) and wc.REPEATS==5 and wc.WARMUP==2 and all(len(wc.routes_for(data))==19 for data in DATASETS),"THIRTY_EIGHT_ROUTES_SEVENTY_SIX_CELLS_380_TIMERS")
    output.mkdir(parents=True,exist_ok=False);meter=rt.Meter("new_direction_cost_"+args.tag,gpu=True);failure=None
    result={"status":"RUNNING",**refs,"selection_locks":locks,"cost_split":"tune","routes":{d:list(wc.routes_for(d)) for d in DATASETS},"cost":{},"datasets":{},"source_snapshots":[],"failed_paths":[],"new_training_updates":0,"new_encoder_decoder_fit":False,"new_alpha_or_seed_selection":False,"new_backbone_training":False,"cached_future_returned_instead_of_actual_F_inside_timer":False,"selected_state_and_coefficients_setup_outside_timer":True,"all_source_file_state_and_coefficient_hashes_outside_timer":True,"encoded_ridge_backend":"actual CUDA E canonical batch up to32 -> CPU features/fused FP32 ridge; transfer timed; D bypassed","native_and_Bolt_backend":"CUDA FP32 B1/B8, no padding to32; not best-backend claim","timed_internal_shape_finite_guards":True,"external_meter_and_hash_guards_outside_timer":True,"protected_TEST_accessed":False,"independent_holdout":False,"strict_K_only":False,"cost_origins":"Original physical first B1/B8 TUNE origins; no DEV timing or inverse-normalized synthetic physical input","encoded_ridge_matched_Bolt_denominator_scope":"Same E state only; ridge bypasses D and is not a same-D or capacity-matched native"}
    checked,protected,models,coefficients,prepared={},[],{},{},{}
    def load(item):return io.checked_payload({**item,"path":str(bound(item))},meter,checked)
    def fail(error,**where):
        item={"status":"FAILED_PATH_PRESERVED",**where,"failure":{"type":type(error).__name__,"message":str(error)}};result["failed_paths"].append(item);return item
    try:
        install_protected_test_guard();meter.protected_check();protected=prior_file_records(native,ridge,cross,science,subset,original,full,refs)
        folder=output/"sources";folder.mkdir()
        paths=(Path(__file__),HERE/"whole_new_direction_cost_v1.py",HERE/"whole_f_input_gradient_cost_v1.py",HERE/"run_linear_e_cost_v1.py",HERE/"run_new_direction_native_v1.py",HERE/"run_new_direction_direct_ridge_v1.py",HERE/"masked_direct_ridge_v1.py",HERE/"run_predictive_cross_moment_v1.py",HERE/"linear_e_adaptation_v1.py",HERE/"lagged_native_head_v1.py",HERE/"native_head_v1.py",HERE/"continuation_runtime_v1.py",HERE/"fixed_decoder_span_v1.py",HERE/"PLAN_NEW_DIRECTION_NATIVE_BASELINES_KO_v1.md",OLD/"prepare_inputs.py",OLD/"run_condition_probes.py",BASE/"cost_readout_v1.py")
        for path in paths:
            destination=folder/path.name;shutil.copyfile(path,destination);result["source_snapshots"].append({**record(path),"snapshot_path":str(destination)})
        result.update(controller_source=record(__file__),whole_cost_source=record(wc.__file__),ridge_controller_source=record(ridge_source.__file__))
        engine=probes.import_old_engine();weight,config=(probes.verified_file(original[k]) for k in ("model_binary_before","model_config_before"));bolt=engine.ex.previous.FrozenBolt()
        result.update(F_before=probes.model_hashes(bolt),model_binary_before=weight,model_config_before=config)
        require(result["F_before"]==cross["F_before"]==cross["F_after"] and all(not p.requires_grad and p.grad is None for p in bolt.model.parameters()),"ONE_ACTUAL_FROZEN_F_DUAL_SIGNATURE")
        result["runtime_behavior_contract"]=runtime.bolt_runtime_contract(engine,bolt,meter.check);require(result["runtime_behavior_contract"]==cross["runtime_behavior_contract"]==full["runtime_behavior_contract"],"SAME_MEDIAN_F_RUNTIME")
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        def raw_callback(history,history_observed):
            with torch.no_grad():return np.ascontiguousarray(bolt(engine.tensor(history)).detach().cpu().numpy().copy(),dtype=np.float32)
        with threadpool_limits(limits=2):
            for data,(c,_) in DATASETS.items():
                try:
                    raw_source=science["datasets"][data]["spaces"]["PCA"]["conditions"]["L7_RECON"]["splits"]["tune"]["input_source"];raw=load(raw_source);proof=canonical_tune_bridge(original,data);physical=load(proof["canonical_record"])
                    require(raw["history"].shape==(64,512,c),"ORIGINAL_TUNE64_COHORT")
                    for key in ("history","history_observed","target","observed","raw_last","origins","mean","std","raw_forecast"):require(byte_equal(raw[key],physical[key]),"FULL_CANONICAL_PHYSICAL_RAW_BINDING:"+key)
                    prepared[data]={"raw":raw,"physical":physical,"predictors":{"RAW_BOLT":raw_callback},"references":{"RAW_BOLT":raw["raw_forecast"]},"metadata":{"RAW_BOLT":{"reference":raw_source,"reference_prediction_array":"raw_forecast","reference_split":"tune","actual_F_forward":True,"matched_Bolt_route":"RAW_BOLT"}},"unavailable":{}}
                    result["datasets"][data]={"canonical_physical_bridge":proof,"raw_TUNE_source":raw_source,"physical_prechecks":{}}
                except Exception as error:result["datasets"][data]=fail(error,dataset=data,stage="COMMON_PHYSICAL_ASSET");continue
                p=prepared[data]
                def unavailable(route,error,stage):
                    item=fail(error,dataset=data,route=route,stage=stage);p["predictors"].pop(route,None);p["references"].pop(route,None);p["unavailable"][route]=item;p["metadata"].setdefault(route,{"unavailable":item})
                encoders={}
                for name in wc.state_names(data):
                    for kind,seed in (("BOLT",None),("NATIVE",92601),("NATIVE",92602)):
                        route=name+"_"+kind+("_s"+str(seed) if seed else "")
                        try:
                            owner=native["datasets"][data]["conditions"][name];accepted=cross["datasets"][data]["conditions"][name];bound(owner["frozen_state_source"],accepted["state_artifact"]);frozen=load(accepted["state_artifact"])
                            require(set(frozen)=={"E","D","center"},"EXACT_FROZEN_STATE_MEMBERS")
                            if kind=="NATIVE":
                                selected=owner["native"][str(seed)];saved=load(selected["selected_state"]);require(set(saved)=={"E","D","center","head.weight","head.bias"},"EXACT_SELECTED_NATIVE_STATE_MEMBERS")
                                model=LaggedNativeForecaster(saved["center"],saved["E"],saved["D"],LAGS).to("cuda");model.load_state_dict({k:torch.as_tensor(v,device="cuda") for k,v in saved.items()});model.eval().requires_grad_(False)
                                require(state_hash(model.state_dict())==selected["selected_state_sha256"] and state_hash(model.head.state_dict())==selected["selected_head_state_sha256"],"EXACT_NATIVE_SELECTED_STATE_AND_HEAD_HASH")
                                ref_record=old.selected_TUNE_checkpoint(selected)["artifact"];predictor=None;selection={"selected_state":selected["selected_state"],"selected_update":selected["selected_update"],"seed":seed,"selection_lock":locks["native"][data+"/"+route]}
                            else:
                                model=LinearLagAdapter(frozen["center"],frozen["E"],frozen["D"],LAGS,learn_E=False).to("cuda").eval().requires_grad_(False)
                                require(state_hash(model.state_dict())==owner["frozen_state_value_sha256"],"EXACT_FROZEN_BOLT_STATE_HASH");encoders[name]=model
                                ref_record=accepted["splits"]["tune"]["artifact"];predictor=bolt;selection={"frozen_state_source":accepted["state_artifact"]};bound(accepted["splits"]["tune"]["input_source"],raw_source);bound(owner["Bolt"]["tune"]["artifact"],ref_record)
                            require(all(not v.requires_grad and v.grad is None for v in model.parameters()) and all(byte_equal(model.state_dict()[k].detach().cpu().numpy(),frozen[k]) for k in frozen),"INFERENCE_FROZEN_E_D_CENTER_HEAD_SAME_SCIENCE")
                            models[data+"/"+route]=model;reference=load(ref_record);require(byte_equal(reference["origins"],raw["origins"]),"SAME_SELECTED_TUNE_ORIGINS")
                            for key in ("target","observed","raw_last","mean","std"):
                                if key in reference:require(byte_equal(reference[key],raw[key]),"SAME_SELECTED_TUNE_CONTEXT:"+key)
                            p["predictors"][route]=adapter_predictor(model,predictor,engine.tensor);p["references"][route]=reference["prediction"];p["metadata"][route]={"kind":kind,"reference":ref_record,"reference_prediction_array":"prediction","reference_split":"tune","frozen_state_source":accepted["state_artifact"],"matched_Bolt_route":name+"_BOLT","lags":list(LAGS),"actual_F_forward":kind=="BOLT","actual_native_head_forward":kind=="NATIVE",**selection}
                            check,arrays=physical_precheck(physical,raw,reference,model,predictor,engine.tensor);check["artifact"]=io.save_npz(output/(data+"_"+route+"_PHYSICAL_FIRST8.npz"),arrays);result["datasets"][data]["physical_prechecks"][route]=check
                            require(check["input_backend_pass"] and check["prediction_backend_pass"],"REGISTERED_PHYSICAL_AND_SELECTED_PREDICTION_PARITY")
                        except Exception as error:unavailable(route,error,"F_OR_NATIVE_STATE_OR_PHYSICAL_PRECHECK")
                for route in (r for r in wc.routes_for(data) if "RIDGE" in r):
                    try:
                        name=ridge_route(route);row=ridge["datasets"][data]["routes"][name];bound(row["TUNE_sources"]["input_source"],raw_source);state=load(row["selected_coefficients"])
                        require(set(state)=={"coefficient_FP64","intercept_FP64","coefficient_FP32","intercept_FP32"},"EXACT_FUSED_RIDGE_COEFFICIENT_MEMBERS")
                        records,digest=ridge_source.coefficient_records(state);require(digest==row["selected_coefficient_value_sha256"],"EXACT_SELECTED_RIDGE_COEFFICIENT_VALUES")
                        encoder=None;encoded=None
                        if not name.startswith("RAW_"):
                            state_name=name.removesuffix("_HISTORY512");encoder=encoders[state_name];accepted=cross["datasets"][data]["conditions"][state_name];bound(row["TUNE_sources"]["frozen_E_D_source"],accepted["state_artifact"]);bound(row["TUNE_sources"]["encoded_past_source"],accepted["splits"]["tune"]["artifact"]);encoded=load(row["TUNE_sources"]["encoded_past_source"])
                            require(byte_equal(encoded["origins"],raw["origins"]),"SAME_ENCODED_PAST_ORIGINS")
                        coefficients[data+"/"+route]=state;ref_record=row["selected_FIT_TUNE_prediction_artifact"];reference=load(ref_record);require(byte_equal(reference["TUNE_origins"],raw["origins"]),"SAME_RIDGE_SELECTED_TUNE_ORIGINS")
                        p["predictors"][route]=ridge_predictor(route,state,encoder);p["references"][route]=reference["TUNE_prediction_FP32"];p["metadata"][route]={"kind":"DIRECT_C_RIDGE","reference":ref_record,"reference_prediction_array":"TUNE_prediction_FP32","reference_split":"tune","selected_coefficients":row["selected_coefficients"],"coefficient_value_sha256":digest,"coefficient_arrays":records,"selected_alpha":row["selected_alpha"],"selection_lock":locks["ridge"][data+"/"+name],"input_sources":row["TUNE_sources"],"matched_Bolt_route":wc.matched_bolt(route),"same_D_or_capacity_match_claimed":False,"D_bypassed":True,"actual_F_forward":False,"actual_native_head_forward":False,"encoded_GPU_E_and_host_transfer_timed":encoder is not None}
                        check,arrays=ridge_physical_precheck(physical,raw,reference,state,route,encoder,encoded);check["artifact"]=io.save_npz(output/(data+"_"+route+"_PHYSICAL_FIRST8.npz"),arrays);result["datasets"][data]["physical_prechecks"][route]=check
                        require(check["input_backend_pass"] and check["prediction_backend_pass"],"REGISTERED_FUSED_RIDGE_PHYSICAL_PREDICTION_PARITY")
                    except Exception as error:unavailable(route,error,"RIDGE_STATE_OR_PHYSICAL_PRECHECK")
            result["selected_states_before"]=old.adapter_hashes(models);result["native_heads_before"]={key:state_hash(model.head.state_dict()) for key,model in models.items() if "_NATIVE_" in key};result["ridge_coefficients_before"]={key:ridge_source.coefficient_records(value)[1] for key,value in coefficients.items()};process=psutil.Process()
            def memory():return {"RSS_bytes":process.memory_info().rss,"allocated_VRAM_bytes":torch.cuda.memory_allocated(),"reserved_VRAM_bytes":torch.cuda.memory_reserved(),"max_allocated_VRAM_bytes":torch.cuda.max_memory_allocated(),"max_reserved_VRAM_bytes":torch.cuda.max_memory_reserved(),"scope":"Shared frozen F, all available24 adapters/16 heads and12 CPU ridge coefficient states; no exclusive allocation or best-backend claim"}
            for data,p in prepared.items():
                try:
                    result["cost"][data]=wc.measure_cost(data,p["physical"]["history_physical"],p["physical"]["history_observed"],p["raw"]["mean"],p["raw"]["std"],p["raw"]["origins"],p["predictors"],p["references"],p["metadata"],output/("cost_"+data),meter.check,torch.cuda.synchronize,memory,p["unavailable"]);result["failed_paths"] += [{"dataset":data,**v} for v in result["cost"][data]["failed_paths"]]
                except Exception as error:
                    fail(error,dataset=data,stage="WHOLE_COST");partial=output/("cost_"+data)/"RESULT.json"
                    if partial.exists():result["cost"][data]=read(partial)
        result["F_after"]=probes.model_hashes(bolt);result["selected_states_after"]=old.adapter_hashes(models);result["native_heads_after"]={key:state_hash(model.head.state_dict()) for key,model in models.items() if "_NATIVE_" in key};result["ridge_coefficients_after"]={key:ridge_source.coefficient_records(value)[1] for key,value in coefficients.items()}
        require(result["F_before"]==result["F_after"] and all(result[key+"_before"]==result[key+"_after"] for key in ("selected_states","native_heads","ridge_coefficients")),"F_E_D_CENTER_HEAD_AND_RIDGE_COEFFICIENTS_IMMUTABLE")
        result["model_binary_after"],result["model_config_after"]=probes.verified_file(weight),probes.verified_file(config)
        for item in protected+list(checked.values())+[v for family in locks.values() for v in family.values()]:bound(item)
        for item in result["source_snapshots"]:require(record(item["path"])["sha256"]==record(item["snapshot_path"])["sha256"]==item["sha256"],"EXECUTED_SOURCE_SNAPSHOTS_UNCHANGED")
        rows=[row for cost in result["cost"].values() for row in cost["rows"]];complete=sum(row["status"]=="COMPLETE" for row in rows);samples=sum(len(row["samples"]) for row in rows)
        result.update(completed_cost_cells=complete,recorded_timer_samples=samples,expected_cost_cells=76,expected_timer_samples=380,selected_adapters_loaded=len(models),selected_native_heads_loaded=len(result["native_heads_after"]),selected_ridge_states_loaded=len(coefficients),expected_physical_prechecks=36,expected_physical_first8_windows=288,physical_prechecks_completed=sum(v.get("input_backend_pass",False) and v.get("prediction_backend_pass",False) for d in result["datasets"].values() for v in d.get("physical_prechecks",{}).values()),input_records=list(checked.values()),protected_prior_records=protected,status="COMPLETE" if complete==76 and samples==380 and not result["failed_paths"] else "PARTIAL_COMPLETED");meter.protected_check();json.dumps(result,allow_nan=False)
    except Exception as error:failure={"type":type(error).__name__,"message":str(error)};result.update(status="FAILED_PRESERVED",failure=failure)
    finally:result["resource_terminal"]=meter.close(result["status"],failure);rt.write_new(output/"RESULT.json",result)
    print(json.dumps({"status":result["status"],"result":str(output/"RESULT.json")}),flush=True);return 0 if result["status"]=="COMPLETE" else 2


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("tag","native-result","native-audit","ridge-result","ridge-audit"):parser.add_argument("--"+key,required=True)
    raise SystemExit(execute(parser.parse_args()))
