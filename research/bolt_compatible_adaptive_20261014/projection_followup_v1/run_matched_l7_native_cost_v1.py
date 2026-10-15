"""Actual same-CUDA TUNE whole cost; closed until exact six-head independent PASS."""
from pathlib import Path
import argparse
import json
import shutil
import sys

import run_linear_e_cost_v1 as old
from run_linear_e_cost_v1 import canonical_tune_bridge,adapter_predictor,physical_precheck

HERE,ROOT,BASE,OLD=old.HERE,old.ROOT,old.BASE,old.OLD
require,record,read,bound,byte_equal=old.require,old.record,old.read,old.bound,old.byte_equal
STATES=("RECON_CP0","E_ONLY_s92601","E_ONLY_s92602")
SEEDS=(92601,92602)
LAGS=(0,1,2,4,8,16,32)
ROUTES=("RAW_BOLT",)+tuple(r for s in STATES for r in (s+"_BOLT",s+"_NATIVE_s92601",s+"_NATIVE_s92602"))
NATIVE_RESULT_SHA="4a6f2d3601c27bf000659aa99a586a2f04d72371766f7558cc78cc16028caa2e"
NATIVE_AUDIT_SHA="429296ac7346189953813292291584d0c5ceb0a60028b4c9b955a2a429a1db87"
NATIVE_AUDITOR_SOURCE={"path":str(HERE/"audit_matched_l7_native_v1.py"),"sha256":"4930f1525fbcff3fad80d0358a017f02cbb22a97796a2841c2137b80e8a6b059"}
AUDIT_COUNTS={
    "native_fits_checked": 6,
    "checkpoint_states_checked": 30,
    "checkpoint_TUNE_predictions_checked": 30,
    "selected_FIT_DEV_predictions_checked": 12,
    "logged_updates_checked": 3600,
    "recorded_post_optimizer_frozen_flag_checks": 3600,
    "CP0_probe_cases_checked": 6,
    "CP0_latent_persistence_byte_checks": 6,
    "frozen_state_CP_member_byte_comparisons_checked": 90,
    "matched_Bolt_decodes_checked": 9,
    "E0_new_RECON_C_decodes_checked": 3,
    "selected_E_only_Bolt_C_byte_checks": 6,
    "cached_Bolt_source_contexts_checked": 9,
    "native_cached_latent_history_byte_comparisons_checked": 42,
    "training_score_fields_checked": 1404,
    "new_space_diagnostic_cells_computed": 12,
    "new_space_LS_SVD_crosschecks_computed": 12,
    "selection_locks_checked": 6,
    "protected_source_and_input_file_hashes_checked": 1784
}
NATIVE_CONTROLLER_SHA="8eafb31307ade30f66d32d18a2c6d62da18a156b0c307cede8ee4bfe72567d43"
HEAD_SOURCE_SHA="cfd2438be26dd7cec0bfd53552ef17f545cb7da0f7716d4185ee557ab249a534"


def load_gates(result_path,audit_path):
    require(isinstance(NATIVE_RESULT_SHA,str) and len(NATIVE_RESULT_SHA)==64 and isinstance(NATIVE_AUDIT_SHA,str) and len(NATIVE_AUDIT_SHA)==64 and isinstance(NATIVE_AUDITOR_SOURCE,dict) and bool(AUDIT_COUNTS),"ACTUAL_MATCHED_L7_NATIVE_PASS_NOT_YET_REGISTERED")
    refs={"matched_l7_native_diagnostic":record(result_path),"matched_l7_native_audit":record(audit_path)}
    require(refs["matched_l7_native_diagnostic"]["sha256"]==NATIVE_RESULT_SHA and refs["matched_l7_native_audit"]["sha256"]==NATIVE_AUDIT_SHA,"EXACT_MATCHED_NATIVE_RESULT_AND_PASS")
    training,audit=read(result_path),read(audit_path)
    require(training["status"]=="COMPLETE" and audit["status"]=="PASS" and training["protected_TEST_accessed"] is audit["protected_TEST_accessed"] is False and not training["failed_paths"] and all(audit["totals"].get(k)==v for k,v in AUDIT_COUNTS.items()),"FULL_MATCHED_NATIVE_COVERAGE_AND_NO_TEST")
    bound(audit["diagnostic"],refs["matched_l7_native_diagnostic"]); bound(audit["audit_source"],NATIVE_AUDITOR_SOURCE)
    require(training["controller_source"]["sha256"]==NATIVE_CONTROLLER_SHA and training["native_source"]["sha256"]==HEAD_SOURCE_SHA and training["native_training_fits"]==training["resolved_selection_locks"]==6 and training["training_updates_completed"]==3600 and training["checkpoint_states_saved"]==30 and training["all_6_locks_before_DEV"] is True and training["matched_Bolt_C_decodes"]==training["cached_Bolt_contexts_reused"]==9 and training["E0_new_matched_RECON_C_decodes"]==3 and training["selected_E_only_Bolt_C_byte_checks"]==6,"EXACT_SIX_HEAD_TRAINING_AND_NINE_MATCHED_BOLT_CONTEXTS")
    for key in ("new_F_forward_calls","new_F_model_objects_loaded","new_E_training_updates","new_D_training_updates","probe_optimizer_updates"): require(training[key]==0,"NATIVE_TRAINING_SCOPE:"+key)
    bound(training["controller_source"]); bound(training["native_source"])
    for key in ("E_only_diagnostic","E_only_audit","joint_recon_diagnostic","joint_recon_audit","full_gradient_parent","full_gradient_parent_audit","causal_lag_diagnostic","causal_lag_audit","parent_report","parent_report_manifest","parent_report_audit","selection_registry","plan"):
        bound(training[key]); bound(audit[key],training[key]); refs[key]=training[key]
    _,_,science,subset,original,full_refs,_=old.load_gates(training["full_gradient_parent"]["path"],training["full_gradient_parent_audit"]["path"])
    bound(training["causal_lag_diagnostic"],full_refs["causal_lag_diagnostic"])
    require(training["F_parent_signatures_reused"]["before"]==training["F_parent_signatures_reused"]["after"]==original["F_before"]==original["F_after"],"SAME_DECLARED_ACTUAL_FROZEN_F")
    registry=read(bound(training["selection_registry"])); require(registry["status"]=="ALL_SIX_RESOLVED_BEFORE_DEV" and registry["DEV_opened_before_seal"] is False and len(registry["conditions"])==6,"ALL_SIX_SELECTION_LOCKS")
    locks={}
    for item in registry["conditions"]:
        key=item["state"],item["seed"]; require(key in {(s,seed) for s in STATES for seed in SEEDS} and key not in locks and item["status"]=="COMPLETE","UNIQUE_SIX_NATIVE_LOCKS")
        row=training["states"][key[0]]["native"][str(key[1])]; lock=read(bound(item["selection_lock"],row["selection_lock"]))
        require(lock["status"]=="COMPLETE" and lock["DEV_opened_before_lock"] is False and lock["selected_update"]==row["selected_update"],"EXACT_NATIVE_TUNE_SELECTION_LOCK"); bound(lock["selected_state"],row["selected_state"]); locks[key[0]+"_NATIVE_s"+str(key[1])]=item["selection_lock"]
    return training,audit,science,subset,original,refs,locks


def native_head_hashes(models):
    from native_head_v1 import state_hash
    return {route:state_hash(model.head.state_dict()) for route,model in models.items() if "_NATIVE_" in route}


def execute(tag,result_path,audit_path):
    training,audit,science,subset,original,refs,locks=load_gates(result_path,audit_path)
    output=old.fresh_output(tag); sys.path[:0]=[str(HERE),str(BASE),str(OLD),str(ROOT)]
    import numpy as np
    import torch
    import psutil
    from threadpoolctl import threadpool_limits
    import continuation_runtime_v1 as rt
    import fixed_decoder_span_v1 as io
    import whole_matched_l7_native_cost_v1 as wc
    import run_condition_probes as probes
    import cost_readout_v1 as runtime
    from native_head_v1 import state_hash
    from linear_e_adaptation_v1 import LinearLagAdapter
    from lagged_native_head_v1 import LaggedNativeForecaster
    from prepare_inputs import install_protected_test_guard
    from run_fit_target_comparison_v1 import prior_file_records
    require(tuple(wc.ROUTES)==ROUTES and wc.REPEATS==5 and wc.WARMUP==2,"TEN_ROUTE_FIVE_ROTATION_COST_CONTRACT")
    output.mkdir(parents=True,exist_ok=False); meter=rt.Meter("matched_L7_native_cost_"+tag,gpu=True)
    result={"status":"RUNNING",**refs,"cost_plan":training["plan"],"routes":list(ROUTES),"selection_locks":locks,"cost_split":"tune","cost":{},"datasets":{},"source_snapshots":[],"failed_paths":[],"new_training_updates":0,"new_encoder_decoder_fit":False,"new_CP_seed_or_lag_selection":False,"new_backbone_training":False,"native_predictor_loaded_or_timed":True,"cached_future_returned_instead_of_actual_F_inside_timer":False,"selected_E_D_head_setup_outside_timer":True,"all_source_file_state_and_head_metadata_hashes_outside_timer":True,"native_and_Bolt_backend":"CUDA FP32; canonical cost batches B1/B8; not best CPU/native backend","protected_TEST_accessed":False,"independent_holdout":False,"strict_K_only":False,"cost_origins":"Original physical first B1/B8 TUNE origins; no DEV timing or reconstructed physical input"}
    checked,protected,models,failure={ },[],{},None
    def load(item): return io.checked_payload({**item,"path":str(bound(item))},meter,checked)
    def fail(error,**where):
        value={"status":"FAILED_PATH_PRESERVED",**where,"failure":{"type":type(error).__name__,"message":str(error)}}; result["failed_paths"].append(value); return value
    try:
        install_protected_test_guard(); meter.protected_check(); protected=prior_file_records(training,audit,science,subset,original,*refs.values())
        folder=output/"sources"; folder.mkdir()
        for path in (Path(__file__),HERE/"whole_matched_l7_native_cost_v1.py",HERE/"whole_f_input_gradient_cost_v1.py",HERE/"run_linear_e_cost_v1.py",HERE/"run_linear_e_adaptation_v1.py",HERE/"linear_e_adaptation_v1.py",HERE/"lagged_native_head_v1.py",HERE/"native_head_v1.py",HERE/"continuation_runtime_v1.py",HERE/"fixed_decoder_span_v1.py",HERE/"PLAN_MATCHED_L7_NATIVE_KO_v1.md",OLD/"prepare_inputs.py",OLD/"run_condition_probes.py",BASE/"cost_readout_v1.py"):
            destination=folder/path.name; shutil.copyfile(path,destination); result["source_snapshots"].append({**record(path),"snapshot_path":str(destination)})
        engine=probes.import_old_engine(); weight,config=(probes.verified_file(original[k]) for k in ("model_binary_before","model_config_before"))
        bolt=engine.ex.previous.FrozenBolt(); result.update(F_before=probes.model_hashes(bolt),model_binary_before=weight,model_config_before=config)
        require(result["F_before"]==training["F_parent_signatures_reused"]["before"],"EXACT_ACTUAL_FROZEN_F_DUAL_SIGNATURE"); result["runtime_behavior_contract"]=runtime.bolt_runtime_contract(engine,bolt,meter.check)
        torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
        raw_source=science["datasets"]["robin"]["spaces"]["PCA"]["conditions"]["L7_RECON"]["splits"]["tune"]["input_source"]; raw=load(raw_source)
        proof=canonical_tune_bridge(original,"robin"); physical=load(proof["canonical_record"])
        require(raw["history"].shape==(64,512,17),"ORIGINAL_ROBIN_TUNE64_PHYSICAL_COHORT")
        for key in ("history","history_observed","target","observed","raw_last","origins","mean","std","raw_forecast"): require(byte_equal(raw[key],physical[key]),"FULL_CANONICAL_PHYSICAL_RAW_ARRAY_BINDING:"+key)
        result["datasets"]["robin"]={"canonical_physical_bridge":proof,"raw_TUNE_source":raw_source,"physical_prechecks":{}}
        def raw_callback(history,history_observed):
            with torch.no_grad(): return np.ascontiguousarray(bolt(engine.tensor(history)).detach().cpu().numpy().copy(),dtype=np.float32)
        predictors={"RAW_BOLT":raw_callback}; references={"RAW_BOLT":raw["raw_forecast"]}; metadata={"RAW_BOLT":{"reference":raw_source,"reference_prediction_array":"raw_forecast","reference_split":"tune","actual_F_forward":True,"matched_Bolt_route":"RAW_BOLT"}}; unavailable={}
        with threadpool_limits(limits=2):
            for state in STATES:
                for kind,seed in (("BOLT",None),("NATIVE",92601),("NATIVE",92602)):
                    route=state+"_"+kind+("_s"+str(seed) if seed else "")
                    try:
                        owner=training["states"][state]; frozen=load(owner["frozen_state_source"])
                        if kind=="NATIVE":
                            row=owner["native"][str(seed)]; saved=load(row["selected_state"])
                            require(set(saved)=={"E","D","center","head.weight","head.bias"},"EXACT_NATIVE_HEAD_AND_FROZEN_STATE_MEMBERS")
                            model=LaggedNativeForecaster(saved["center"],saved["E"],saved["D"],LAGS).to("cuda"); model.load_state_dict({k:torch.as_tensor(v,device="cuda") for k,v in saved.items()}); model.eval().requires_grad_(False)
                            require(state_hash(model.state_dict())==row["selected_state_sha256"] and state_hash(model.head.state_dict())==row["selected_head_state_sha256"],"EXACT_SELECTED_NATIVE_FULL_STATE_AND_HEAD_HASH")
                            reference_record=old.selected_TUNE_checkpoint(row)["artifact"]; predictor=None; selection={"head_order_seed":seed,"selected_update":row["selected_update"],"selected_state":row["selected_state"],"selected_head_state_sha256":row["selected_head_state_sha256"],"selection_lock":locks[route]}
                        else:
                            require(set(frozen)=={"E","D","center"},"EXACT_FROZEN_BOLT_ADAPTER_MEMBERS")
                            model=LinearLagAdapter(frozen["center"],frozen["E"],frozen["D"],LAGS,learn_E=False).to("cuda").eval().requires_grad_(False)
                            require(state_hash(model.state_dict())==owner["frozen_state_value_sha256"],"EXACT_FROZEN_BOLT_ADAPTER_HASH")
                            reference_record=owner["Bolt"]["tune"]["artifact"]; predictor=bolt; selection={"frozen_state_source":owner["frozen_state_source"]}
                        require(all(not p.requires_grad and p.grad is None for p in model.parameters()) and all(byte_equal(model.state_dict()[k].detach().cpu().numpy(),frozen[k]) for k in frozen),"INFERENCE_ALL_FROZEN_AND_SAME_MATCHING_E_D_CENTER")
                        models[route]=model; reference=load(reference_record)
                        for key in ("target","observed","raw_last","origins","mean","std"): require(byte_equal(reference[key],raw[key]),"SAME_SELECTED_TUNE_INPUT_AND_TARGET_MASK:"+key)
                        predictors[route]=adapter_predictor(model,predictor,engine.tensor); references[route]=reference["prediction"]
                        metadata[route]={"kind":kind,"frozen_state_label":state,"frozen_state_source":owner["frozen_state_source"],"lags":list(LAGS),"reference":reference_record,"reference_prediction_array":"prediction","reference_split":"tune","actual_F_forward":kind=="BOLT","actual_native_head_forward":kind=="NATIVE","matched_Bolt_route":state+"_BOLT",**selection}
                        check,arrays=physical_precheck(physical,raw,reference,model,predictor,engine.tensor); check["artifact"]=io.save_npz(output/(route+"_PHYSICAL_FIRST8.npz"),arrays); result["datasets"]["robin"]["physical_prechecks"][route]=check
                        require(check["input_backend_pass"] and check["prediction_backend_pass"],"REGISTERED_SELECTED_PHYSICAL_NORMALIZED_AND_PHYSICAL_PREDICTION_PARITY")
                    except Exception as error:
                        item=fail(error,route=route,stage="SELECTED_STATE_OR_PHYSICAL_PRECHECK"); predictors.pop(route,None); references.pop(route,None); unavailable[route]=item; metadata.setdefault(route,{"unavailable":item})
            result["selected_states_before"]=old.adapter_hashes(models); result["native_heads_before"]=native_head_hashes(models); process=psutil.Process()
            def memory(): return {"RSS_bytes":process.memory_info().rss,"allocated_VRAM_bytes":torch.cuda.memory_allocated(),"reserved_VRAM_bytes":torch.cuda.memory_reserved(),"max_allocated_VRAM_bytes":torch.cuda.max_memory_allocated(),"max_reserved_VRAM_bytes":torch.cuda.max_memory_reserved(),"scope":"Shared resident frozen F, up to six heads and three Bolt adapters; no isolated per-route allocation"}
            try: result["cost"]["robin"]=wc.measure_cost("robin",physical["history_physical"],physical["history_observed"],raw["mean"],raw["std"],raw["origins"],predictors,references,metadata,output/"cost_robin",meter.check,torch.cuda.synchronize,memory,unavailable)
            except Exception as error:
                fail(error,stage="WHOLE_COST"); partial=output/"cost_robin/RESULT.json"
                if partial.exists(): result["cost"]["robin"]=read(partial)
        result["F_after"]=probes.model_hashes(bolt); result["selected_states_after"]=old.adapter_hashes(models); result["native_heads_after"]=native_head_hashes(models)
        require(result["F_before"]==result["F_after"] and result["selected_states_before"]==result["selected_states_after"] and result["native_heads_before"]==result["native_heads_after"],"F_E_D_CENTER_AND_NATIVE_HEADS_UNCHANGED")
        result["model_binary_after"],result["model_config_after"]=probes.verified_file(weight),probes.verified_file(config)
        for item in protected+list(checked.values())+list(locks.values()): bound(item)
        for item in result["source_snapshots"]: require(record(item["path"])["sha256"]==record(item["snapshot_path"])["sha256"]==item["sha256"],"EXECUTED_SOURCE_SNAPSHOTS_UNCHANGED")
        rows=result["cost"].get("robin",{}).get("rows",[]); result["failed_paths"]+=result["cost"].get("robin",{}).get("failed_paths",[])
        completed=sum(r["status"]=="COMPLETE" for r in rows); samples=sum(len(r["samples"]) for r in rows)
        result.update(completed_cost_cells=completed,recorded_timer_samples=samples,expected_cost_cells=20,expected_timer_samples=100,selected_adapters_loaded=len(models),selected_native_heads_loaded=len(result["native_heads_after"]),input_records=list(checked.values()),protected_prior_records=protected,status="COMPLETE" if completed==20 and samples==100 and not result["failed_paths"] else "PARTIAL_COMPLETED"); meter.protected_check()
    except Exception as error: failure={"type":type(error).__name__,"message":str(error)}; result.update(status="FAILED_PRESERVED",failure=failure)
    finally: result["resource_terminal"]=meter.close(result["status"],failure); rt.write_new(output/"RESULT.json",result)
    print(json.dumps({"status":result["status"],"result":str(output/"RESULT.json")}),flush=True); return 0 if result["status"]=="COMPLETE" else 2


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("tag","native-result","native-audit"): parser.add_argument("--"+key,required=True)
    args=parser.parse_args(); raise SystemExit(execute(args.tag,args.native_result,args.native_audit))
