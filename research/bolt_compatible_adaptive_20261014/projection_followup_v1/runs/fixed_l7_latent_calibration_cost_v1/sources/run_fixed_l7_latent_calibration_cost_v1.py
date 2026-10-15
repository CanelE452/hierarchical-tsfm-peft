"""Actual fixed-q TUNE whole cost; closed until exact science COMPLETE and independent PASS."""
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
STATES=("RECON_CP0","E_ONLY_s92601","E_ONLY_s92602")
FAMILIES=("SCALE_ONLY","SCALE_BIAS")
LAGS=(0,1,2,4,8,16,32)
ROUTES=("RAW_BOLT",)+tuple(r for state in STATES for r in (state+"_BOLT",state+"_SCALE_ONLY",state+"_SCALE_BIAS"))
Q_RESULT_SHA="88ddbbf7287ad89b7bf81e170b3e05ddfc89bead700e2fefa50bb0dd4e85f576"
Q_AUDIT_SHA="b31b742d3947a205dbaf01cd84239b3cde95e1ea9baf3331a3c08c48a0aa5ba0"
Q_CONTROLLER_SHA="2d81492385dc8f2c889993f9c5d4f02b32d1917a8e9e20f1b53c88778173ee04"
Q_MODULE_SHA="088b9800724b07ced0c42cfa2f514077e430d2c3a7f291dea6f45d1f444cb523"
Q_AUDITOR_SOURCE={"path":str(HERE/"audit_fixed_l7_latent_calibration_v1.py"),"sha256":"208816a2656800afa190a9f713abe3e51c4f8ec88cfc1e6116bd4e4788b2464d"}
AUDIT_COUNTS={
    "fit_conditions_checked": 6,
    "actual_SVD_fits_checked": 6,
    "ridge_candidates_checked": 42,
    "identity_candidates_checked": 6,
    "FIT_TUNE_prediction_evaluations_checked": 96,
    "selected_DEV_prediction_evaluations_checked": 6,
    "score_fields_checked": 3672,
    "selection_locks_checked": 6,
    "protected_source_input_and_result_hashes_checked": 226
}


def load_gates(result_path,audit_path):
    require(all(isinstance(v,str) and len(v)==64 for v in (Q_RESULT_SHA,Q_AUDIT_SHA,Q_CONTROLLER_SHA,Q_MODULE_SHA)) and isinstance(Q_AUDITOR_SOURCE,dict) and bool(AUDIT_COUNTS),"ACTUAL_FIXED_Q_COMPLETE_AND_PASS_NOT_YET_REGISTERED")
    refs={"latent_calibration_diagnostic":record(result_path),"latent_calibration_audit":record(audit_path)}
    require(refs["latent_calibration_diagnostic"]["sha256"]==Q_RESULT_SHA and refs["latent_calibration_audit"]["sha256"]==Q_AUDIT_SHA,"EXACT_FIXED_Q_RESULT_AND_PASS")
    calibration,audit=read(result_path),read(audit_path)
    require(calibration["status"]=="COMPLETE" and audit["status"]=="PASS" and calibration["protected_TEST_accessed"] is audit["protected_TEST_accessed"] is False and not calibration["failed_paths"] and not calibration["failed_candidate_paths"] and all(audit["totals"].get(k)==v for k,v in AUDIT_COUNTS.items()),"FULL_FIXED_Q_SCIENCE_NO_TEST")
    bound(audit["diagnostic"],refs["latent_calibration_diagnostic"]); bound(audit["audit_source"],Q_AUDITOR_SOURCE)
    require(calibration["controller_source"]["sha256"]==Q_CONTROLLER_SHA and calibration["module_source"]["sha256"]==Q_MODULE_SHA,"EXACT_EXECUTED_FIXED_Q_SOURCES")
    bound(calibration["controller_source"]); bound(calibration["module_source"])
    require(calibration["fit_conditions_completed"]==calibration["selected_conditions_completed"]==calibration["resolved_selection_locks"]==calibration["selected_DEV_evaluations_completed"]==6 and calibration["ridge_candidates_generated"]==42 and calibration["identity_comparisons_generated"]==6 and calibration["candidate_FIT_TUNE_evaluations_completed"]==96 and calibration["all_6_locks_before_DEV"] is True and calibration["cached_Bolt_contexts_reused"]==calibration["cached_baseline_redecode_checks"]==9,"EXACT_SIX_FIXED_Q_FAMILIES_AND_SELECTION")
    for key in ("new_F_forward_calls","new_F_model_objects_loaded","new_optimizer_updates","new_E_D_training_updates"): require(calibration[key]==0,"ACCEPTED_FIXED_Q_SCOPE:"+key)
    require(calibration["native_head_weights_loaded"] is calibration["new_backbone_training"] is False,"NO_NATIVE_HEAD_OR_BACKBONE_TRAINING")
    for key in ("matched_l7_native_diagnostic","matched_l7_native_audit","matched_l7_native_cost","matched_l7_native_cost_audit","native_selection_registry","native_bundle_metadata_reader","parent_report_auditor_source","parent_report","parent_report_manifest","parent_report_audit","E_only_diagnostic","E_only_audit","joint_recon_diagnostic","joint_recon_audit","causal_lag_diagnostic","causal_lag_audit","selection_registry","plan"):
        bound(calibration[key]); refs[key]=calibration[key]
    from run_fixed_l7_latent_calibration_v1 import load_gates as accepted_parent
    args=SimpleNamespace(**{key:calibration[value]["path"] for key,value in (("report","parent_report"),("report_manifest","parent_report_manifest"),("report_audit","parent_report_audit"),("native_result","matched_l7_native_diagnostic"),("native_audit","matched_l7_native_audit"),("native_cost","matched_l7_native_cost"),("native_cost_audit","matched_l7_native_cost_audit"))})
    native,science,parent_refs=accepted_parent(args)
    for key,value in parent_refs.items(): bound(refs[key],value)
    subset=read(bound(science["raw_subset_diagnostic"])); original=read(bound(subset["source_producer"]))
    require(calibration["F_parent_signatures_reused"]==native["F_parent_signatures_reused"] and calibration["F_parent_signatures_reused"]["before"]==calibration["F_parent_signatures_reused"]["after"]==original["F_before"]==original["F_after"],"SAME_FROZEN_F_DECLARED_SIGNATURES")
    registry=read(bound(calibration["selection_registry"])); require(registry["status"]=="ALL_SIX_RESOLVED_BEFORE_DEV" and registry["DEV_opened_before_seal"] is False and len(registry["conditions"])==6,"ALL_SIX_Q_LOCKS")
    locks={}
    for item in registry["conditions"]:
        key=item["state"],item["family"]; require(key in {(s,f) for s in STATES for f in FAMILIES} and key not in locks and item["status"]=="COMPLETE","SIX_UNIQUE_Q_CONDITIONS")
        row=calibration["states"][key[0]]["families"][key[1]]; lock=read(bound(item["selection_lock"],row["selection_lock"]))
        require(lock["status"]=="COMPLETE" and lock["DEV_opened_before_lock"] is False and all(lock[k]==row[k] for k in ("selected_candidate","selected_kind","selected_lambda","selected_TUNE_MSE","selected_coefficient_value_sha256")),"EXACT_FIXED_Q_TUNE_SELECTION_LOCK")
        bound(lock["selected_coefficients"],row["selected_coefficients"]); locks[key[0]+"_"+key[1]]=item["selection_lock"]
    return calibration,audit,native,science,subset,original,refs,locks


def calibration_hashes(models):
    from native_head_v1 import state_hash
    return {route:{"coefficients_state_sha256":state_hash({"a":model.a,"b":model.b}),"config":model.calibration_config()} for route,model in models.items() if "_SCALE_" in route}


def compose_reference(baseline,calibrated):
    from causal_lag_decoder_v1 import lagged_delta_features
    require(byte_equal(baseline["origins"],calibrated["origins"]),"COMPOSED_Q_REFERENCE_EXACT_ORIGINS")
    return {**baseline,"prediction":calibrated["prediction_FP32"],"latent_future":calibrated["latent_future_calibrated_FP32"],"lag_features":lagged_delta_features(baseline["latent_history"],calibrated["latent_future_calibrated_FP32"],LAGS)}


def execute(tag,result_path,audit_path):
    calibration,audit,native,science,subset,original,refs,locks=load_gates(result_path,audit_path)
    output=old.fresh_output(tag); sys.path[:0]=[str(HERE),str(BASE),str(OLD),str(ROOT)]
    import numpy as np
    import torch
    import psutil
    from threadpoolctl import threadpool_limits
    import continuation_runtime_v1 as rt
    import fixed_decoder_span_v1 as io
    import whole_fixed_l7_latent_calibration_cost_v1 as wc
    import run_condition_probes as probes
    import cost_readout_v1 as runtime
    from native_head_v1 import state_hash
    from linear_e_adaptation_v1 import LinearLagAdapter
    from calibrated_l7_forecaster_v1 import CalibratedL7Forecaster
    from prepare_inputs import install_protected_test_guard
    from run_fit_target_comparison_v1 import prior_file_records
    require(tuple(wc.ROUTES)==ROUTES and wc.REPEATS==5 and wc.WARMUP==2,"TEN_ROUTE_FIVE_ROTATION_Q_COST")
    output.mkdir(parents=True,exist_ok=False); meter=rt.Meter("fixed_L7_latent_calibration_cost_"+tag,gpu=True)
    result={"status":"RUNNING",**refs,"cost_plan":calibration["plan"],"routes":list(ROUTES),"selection_locks":locks,"cost_split":"tune","cost":{},"datasets":{},"source_snapshots":[],"failed_paths":[],"new_training_updates":0,"new_encoder_decoder_or_q_fit":False,"new_CP_family_or_lambda_selection":False,"new_backbone_training":False,"native_predictor_loaded_or_timed":False,"latent_calibration_loaded_or_timed":True,"cached_future_returned_instead_of_actual_F_inside_timer":False,"identity_returns_cached_scientific_baseline_inside_timer":False,"selected_E_D_q_setup_outside_timer":True,"all_source_file_state_and_q_metadata_hashes_outside_timer":True,"internal_finite_shape_guards_inside_timer":True,"backend":"CUDA FP32 actual E/F/q/D; B1/B8; no best-backend claim","protected_TEST_accessed":False,"independent_holdout":False,"strict_K_only":False,"cost_origins":"Original physical first B1/B8 TUNE origins; no DEV timing or synthesized physical input"}
    checked,protected,models,failure={},[],{},None
    def load(item): return io.checked_payload({**item,"path":str(bound(item))},meter,checked)
    def fail(error,**where):
        item={"status":"FAILED_PATH_PRESERVED",**where,"failure":{"type":type(error).__name__,"message":str(error)}}; result["failed_paths"].append(item); return item
    try:
        install_protected_test_guard(); meter.protected_check(); protected=prior_file_records(calibration,audit,native,science,subset,original,*refs.values())
        folder=output/"sources"; folder.mkdir()
        for path in (Path(__file__),HERE/"whole_fixed_l7_latent_calibration_cost_v1.py",HERE/"whole_f_input_gradient_cost_v1.py",HERE/"calibrated_l7_forecaster_v1.py",HERE/"fixed_l7_latent_calibration_v1.py",HERE/"run_fixed_l7_latent_calibration_v1.py",HERE/"run_linear_e_cost_v1.py",HERE/"run_linear_e_adaptation_v1.py",HERE/"linear_e_adaptation_v1.py",HERE/"causal_lag_decoder_v1.py",HERE/"continuation_runtime_v1.py",HERE/"fixed_decoder_span_v1.py",HERE/"PLAN_FIXED_L7_LATENT_CALIBRATION_KO_v1.md",OLD/"prepare_inputs.py",OLD/"run_condition_probes.py",BASE/"cost_readout_v1.py"):
            destination=folder/path.name; shutil.copyfile(path,destination); result["source_snapshots"].append({**record(path),"snapshot_path":str(destination)})
        engine=probes.import_old_engine(); weight,config=(probes.verified_file(original[k]) for k in ("model_binary_before","model_config_before")); bolt=engine.ex.previous.FrozenBolt()
        result.update(F_before=probes.model_hashes(bolt),model_binary_before=weight,model_config_before=config)
        require(result["F_before"]==calibration["F_parent_signatures_reused"]["before"],"EXACT_ACTUAL_FROZEN_F_DUAL_SIGNATURE"); result["runtime_behavior_contract"]=runtime.bolt_runtime_contract(engine,bolt,meter.check)
        torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
        raw_source=science["datasets"]["robin"]["spaces"]["PCA"]["conditions"]["L7_RECON"]["splits"]["tune"]["input_source"]; raw=load(raw_source); proof=canonical_tune_bridge(original,"robin"); physical=load(proof["canonical_record"])
        require(raw["history"].shape==(64,512,17),"ORIGINAL_ROBIN_TUNE64_PHYSICAL_COHORT")
        for key in ("history","history_observed","target","observed","raw_last","origins","mean","std","raw_forecast"): require(byte_equal(raw[key],physical[key]),"FULL_CANONICAL_PHYSICAL_RAW_ARRAY_BINDING:"+key)
        result["datasets"]["robin"]={"canonical_physical_bridge":proof,"raw_TUNE_source":raw_source,"physical_prechecks":{}}
        def raw_callback(history,history_observed):
            with torch.no_grad(): return np.ascontiguousarray(bolt(engine.tensor(history)).detach().cpu().numpy().copy(),dtype=np.float32)
        predictors={"RAW_BOLT":raw_callback}; references={"RAW_BOLT":raw["raw_forecast"]}; metadata={"RAW_BOLT":{"reference":raw_source,"reference_prediction_array":"raw_forecast","reference_split":"tune","actual_F_forward":True,"matched_Bolt_route":"RAW_BOLT"}}; unavailable={}
        with threadpool_limits(limits=2):
            for state in STATES:
                for family in (None,*FAMILIES):
                    route=state+"_"+(family or "BOLT")
                    try:
                        source=calibration["states"][state]["frozen_state_source"]; bound(source,native["states"][state]["frozen_state_source"]); frozen=load(source); require(set(frozen)=={"E","D","center"},"EXACT_FROZEN_ADAPTER_MEMBERS")
                        baseline_source=native["states"][state]["Bolt"]["tune"]["artifact"]; baseline=load(baseline_source)
                        if family:
                            row=calibration["states"][state]["families"][family]; coefficients=load(row["selected_coefficients"])
                            require(set(coefficients)=={"a_FP64","b_FP64","a_FP32","b_FP32","coefficients_FP64","coefficients_FP32"} and state_hash({k:torch.as_tensor(v) for k,v in coefficients.items()})==row["selected_coefficient_value_sha256"],"EXACT_SELECTED_FIXED_Q_COEFFICIENTS")
                            model=CalibratedL7Forecaster(frozen["center"],frozen["E"],frozen["D"],coefficients["a_FP32"],coefficients["b_FP32"],family=family,identity=row["selected_kind"]=="IDENTITY").to("cuda").eval().requires_grad_(False)
                            reference_row=row["TUNE"]; bound(reference_row["input_source"],raw_source); bound(reference_row["cached_Bolt_source"],baseline_source); selected=load(reference_row["artifact"]); reference=compose_reference(baseline,selected)
                            reference_record=io.save_npz(output/(route+"_COMPOSED_TUNE_REFERENCE.npz"),reference)
                            selection={"selected_candidate":row["selected_candidate"],"selected_kind":row["selected_kind"],"selected_lambda":row["selected_lambda"],"selected_coefficients":row["selected_coefficients"],"selected_coefficient_value_sha256":row["selected_coefficient_value_sha256"],"selection_lock":locks[route],"selected_q_prediction":reference_row["artifact"],"matched_Bolt_source":baseline_source,"reference_component_composition":"Accepted matched Bolt history and selected q future, original NumPy lag features outside timer; selected C prediction preserved"}
                        else:
                            model=LinearLagAdapter(frozen["center"],frozen["E"],frozen["D"],LAGS,learn_E=False).to("cuda").eval().requires_grad_(False); reference,reference_record=baseline,baseline_source; selection={"matched_Bolt_source":baseline_source}
                        require(all(not p.requires_grad and p.grad is None for p in model.parameters()) and all(byte_equal(model.state_dict()[k].detach().cpu().numpy(),frozen[k]) for k in frozen),"ALL_INFERENCE_FROZEN_AND_ORIGINAL_E_D_CENTER")
                        models[route]=model
                        for key in ("target","observed","raw_last","origins","mean","std"): require(byte_equal(reference[key],raw[key]),"SAME_SELECTED_TUNE_CONTEXT:"+key)
                        predictors[route]=adapter_predictor(model,bolt,engine.tensor); references[route]=reference["prediction"]
                        metadata[route]={"frozen_state_label":state,"frozen_state_source":source,"family":family,"lags":list(LAGS),"reference":reference_record,"reference_prediction_array":"prediction","reference_split":"tune","actual_F_forward":True,"latent_calibration_applied":bool(family),"matched_Bolt_route":state+"_BOLT",**selection}
                        check,arrays=physical_precheck(physical,raw,reference,model,bolt,engine.tensor); check["artifact"]=io.save_npz(output/(route+"_PHYSICAL_FIRST8.npz"),arrays); result["datasets"]["robin"]["physical_prechecks"][route]=check
                        require(check["input_backend_pass"] and check["prediction_backend_pass"],"REGISTERED_FIXED_Q_PHYSICAL_NORMALIZED_AND_PHYSICAL_PREDICTION_PARITY")
                    except Exception as error:
                        item=fail(error,route=route,stage="SELECTED_Q_STATE_OR_PHYSICAL_PRECHECK"); predictors.pop(route,None); references.pop(route,None); unavailable[route]=item; metadata.setdefault(route,{"unavailable":item})
            result["selected_states_before"]=old.adapter_hashes(models); result["selected_calibrations_before"]=calibration_hashes(models); process=psutil.Process()
            def memory(): return {"RSS_bytes":process.memory_info().rss,"allocated_VRAM_bytes":torch.cuda.memory_allocated(),"reserved_VRAM_bytes":torch.cuda.memory_reserved(),"max_allocated_VRAM_bytes":torch.cuda.max_memory_allocated(),"max_reserved_VRAM_bytes":torch.cuda.max_memory_reserved(),"scope":"Shared resident frozen F, up to six q adapters and three Bolt adapters; no isolated per-route allocation"}
            try: result["cost"]["robin"]=wc.measure_cost("robin",physical["history_physical"],physical["history_observed"],raw["mean"],raw["std"],raw["origins"],predictors,references,metadata,output/"cost_robin",meter.check,torch.cuda.synchronize,memory,unavailable)
            except Exception as error:
                fail(error,stage="WHOLE_COST"); partial=output/"cost_robin/RESULT.json"
                if partial.exists(): result["cost"]["robin"]=read(partial)
        result["F_after"]=probes.model_hashes(bolt); result["selected_states_after"]=old.adapter_hashes(models); result["selected_calibrations_after"]=calibration_hashes(models)
        require(result["F_before"]==result["F_after"] and result["selected_states_before"]==result["selected_states_after"] and result["selected_calibrations_before"]==result["selected_calibrations_after"],"F_E_D_CENTER_Q_COEFFICIENT_AND_CONFIG_UNCHANGED")
        result["model_binary_after"],result["model_config_after"]=probes.verified_file(weight),probes.verified_file(config)
        for item in protected+list(checked.values())+list(locks.values()): bound(item)
        for item in result["source_snapshots"]: require(record(item["path"])["sha256"]==record(item["snapshot_path"])["sha256"]==item["sha256"],"EXECUTED_SOURCE_SNAPSHOTS_UNCHANGED")
        rows=result["cost"].get("robin",{}).get("rows",[]); result["failed_paths"]+=result["cost"].get("robin",{}).get("failed_paths",[]); completed=sum(r["status"]=="COMPLETE" for r in rows); samples=sum(len(r["samples"]) for r in rows)
        result.update(completed_cost_cells=completed,recorded_timer_samples=samples,expected_cost_cells=20,expected_timer_samples=100,selected_adapters_loaded=len(models),selected_calibrations_loaded=len(result["selected_calibrations_after"]),input_records=list(checked.values()),protected_prior_records=protected,status="COMPLETE" if completed==20 and samples==100 and not result["failed_paths"] else "PARTIAL_COMPLETED"); meter.protected_check()
    except Exception as error: failure={"type":type(error).__name__,"message":str(error)}; result.update(status="FAILED_PRESERVED",failure=failure)
    finally: result["resource_terminal"]=meter.close(result["status"],failure); rt.write_new(output/"RESULT.json",result)
    print(json.dumps({"status":result["status"],"result":str(output/"RESULT.json")}),flush=True); return 0 if result["status"]=="COMPLETE" else 2


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("tag","calibration-result","calibration-audit"): parser.add_argument("--"+key,required=True)
    args=parser.parse_args(); raise SystemExit(execute(args.tag,args.calibration_result,args.calibration_audit))
