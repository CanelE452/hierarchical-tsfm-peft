"""Six fresh head-only L7 native fits; exact frozen E/D and reused actual Bolt latents."""
from pathlib import Path
import argparse
import json
import shutil
import sys
import time

import run_linear_e_adaptation_v1 as old

HERE,ROOT,BASE,OLD,require=old.HERE,old.ROOT,old.BASE,old.OLD,old.require
STATES=("RECON_CP0","E_ONLY_s92601","E_ONLY_s92602")
SEEDS=(92601,92602)
LAGS=(0,1,2,4,8,16,32)
CONDITIONS=tuple((state,seed) for state in STATES for seed in SEEDS)
FIXED_SHA={"train":"4b1559216ea99937ca8e7e0f6cdac9885abc29479a6050d82336dcfa0305bb2b","audit":"f79324dc9452192601ac16b483e4b19c296f3c252f5814062c7c05f7d4f9e85b","report":"8c38be2568a1b06c2272a4fc231e1fdf63e41ff4e16373a22381f400c4cac29d","manifest":"ef507cdf9dbda5dccb0165ce2be4428b0fc847732f38b0cfb9abd850ae380f6d","report_audit":"b9f6873eecaad4bc5cd2cae0a81be4d5f7dfc41a0343d13df0cecb5351a36e21"}
CONTROLLER_SHA="1c5f5a1e8f878b15705fc8f808af35ab8bc0ea9cdbc21375b7ad789a5dc45ff1"
AUDITOR_SHA="adaae4d3bc0767192ac8a98a4cacf2f087800558fe72d8b391cafeb72a5c2fba"
AUDIT_COUNTS={"E_only_fits_checked":2,"checkpoint_states_checked":10,"checkpoint_TUNE_predictions_checked":10,"selected_FIT_DEV_predictions_checked":4,"logged_updates_checked":1200,"recorded_post_optimizer_D_flag_checks":1200,"CP0_probe_cases_checked":2,"CP0_component_byte_comparisons_checked":16,"fixed_D_center_CP_byte_comparisons_checked":20,"reused_joint_conditions_checked":2,"training_score_fields_checked":396,"new_space_diagnostic_cells_computed":4,"new_space_LS_SVD_crosschecks_computed":4,"selection_locks_checked":2,"protected_source_and_input_file_hashes_checked":53}


def frozen_source(e_only,joint,state):
    return joint["datasets"]["robin"]["conditions"]["92601"]["CP0_TUNE"]["state"] if state=="RECON_CP0" else e_only["datasets"]["robin"]["conditions"][state[-5:]]["selected_state"]


def cache_source(e_only,joint,full,state,split):
    if state=="RECON_CP0":
        return joint["datasets"]["robin"]["conditions"]["92601"]["CP0_TUNE"]["artifact"] if split=="tune" else full["CP0_LS_and_RAW_baselines"]["robin"]["L7"][split]["CP0_LS"]["artifact"]
    row=e_only["datasets"]["robin"]["conditions"][state[-5:]]
    if split=="tune": return next(cp["artifact"] for cp in row["checkpoints"] if cp["update"]==row["selected_update"])
    return row["FIT" if split=="fit" else "DEV"]["artifact"]


def load_gates(train_path,audit_path,report_path,manifest_path,report_audit_path):
    require(all(isinstance(v,str) and len(v)==64 for v in FIXED_SHA.values()),"CURRENT_E_ONLY_REPORT_NOT_YET_REGISTERED")
    paths=dict(zip(FIXED_SHA,(train_path,audit_path,report_path,manifest_path,report_audit_path))); records={k:old.record(p) for k,p in paths.items()}
    require(all(records[k]["sha256"]==FIXED_SHA[k] for k in paths),"EXACT_CURRENT_E_ONLY_REPORT_BUNDLE")
    train,audit,manifest,report_audit=(old.read(paths[k]) for k in ("train","audit","manifest","report_audit"))
    require(train["status"]==manifest["status"]=="COMPLETE" and audit["status"]==report_audit["status"]=="PASS" and audit["training_verification_status"]=="PASS" and audit["new_space_diagnostic_status"]=="COMPLETE" and not audit["partial_paths"] and not audit["new_output_space_diagnostic_failures"] and audit["protected_TEST_accessed"] is report_audit["protected_TEST_accessed"] is manifest["protected_TEST_accessed"] is False,"E_ONLY_TRAIN_NEW_SPACES_AND_REPORT_COMPLETE_PASS")
    for actual,wanted in ((audit["diagnostic"],records["train"]),(report_audit["report"],records["report"]),(report_audit["manifest"],records["manifest"]),(manifest["inputs"]["train"],records["train"]),(manifest["inputs"]["audit"],records["audit"])): old.bound(actual,wanted)
    require(train["controller_source"]["sha256"]==CONTROLLER_SHA and audit["audit_source"]["sha256"]==AUDITOR_SHA and all(audit["totals"].get(k)==v for k,v in AUDIT_COUNTS.items()),"EXACT_ACCEPTED_E_ONLY_SOURCE_AND_COVERAGE")
    old.bound(train["controller_source"]); old.bound(audit["audit_source"])
    for item in manifest["files"]:
        path=(old.resolve(manifest_path).parent/item["path"]).resolve(); require(path.is_relative_to(old.resolve(manifest_path).parent),"PARENT_REPORT_FILE_CONTAINMENT"); old.bound({**item,"path":str(path)})
    parents={}
    for key in ("joint_recon_diagnostic","joint_recon_audit","full_gradient_parent","full_gradient_parent_audit","causal_lag_diagnostic","causal_lag_audit"):
        old.bound(train[key]); old.bound(audit[key],train[key]); parents[key]=old.read(train[key]["path"])
        require(parents[key]["status"]==("PASS" if "audit" in key else "COMPLETE"),"ACCEPTED_PARENT_COMPLETE_PASS")
    require(train["F_before"]==train["F_after"]==parents["joint_recon_diagnostic"]["F_before"] and train["fixed_D_E_only_training_fits"]==2 and train["all_2_locks_before_DEV"] is True and not train["failed_paths"] and train["protected_TEST_accessed"] is False,"UNCHANGED_ACCEPTED_E_ONLY_CONDITIONS")
    refs={"E_only_diagnostic":records["train"],"E_only_audit":records["audit"],"parent_report":records["report"],"parent_report_manifest":records["manifest"],"parent_report_audit":records["report_audit"],"E_only_selection_registry":train["selection_registry"],**{k:train[k] for k in parents}}
    return train,parents["joint_recon_diagnostic"],parents["full_gradient_parent"],parents["causal_lag_diagnostic"],parents,refs


def cp0_probe(outputs,cache,state,raw):
    import numpy as np
    from causal_lag_decoder_v1 import predict
    flat=np.ascontiguousarray(np.broadcast_to(cache["latent_history"][:,-1:],outputs["latent_future"].shape))
    reference=predict(cache["latent_history"],flat,raw["raw_last"],state["D"],LAGS)
    checks={"cached_Torch_latent_history_byte_exact":old.byte_equal(outputs["latent_history"],cache["latent_history"]),"latent_future_is_last_byte_exact":old.byte_equal(outputs["latent_future"],flat),"NumPy_lag_prediction":old.parity(outputs["prediction"],reference)}
    passed=checks["cached_Torch_latent_history_byte_exact"] and checks["latent_future_is_last_byte_exact"] and checks["NumPy_lag_prediction"]["pass"]
    return {"status":"PASS" if passed else "FAILED_PARITY_PRESERVED","optimizer_updates":0,"checks":checks,"latent_persistence_is_current_C_persistence_claimed":False}, {**{"native_"+k:v for k,v in outputs.items()},"cached_Torch_latent_history":cache["latent_history"],"flat_future":flat,"NumPy_flat_prediction":reference}


def decode_cached_Bolt(model,cache,raw,check):
    import numpy as np
    import torch
    parts={k:[] for k in ("prediction","latent_history","latent_future","lag_features")}
    with torch.no_grad():
        for start in range(0,len(raw["raw_last"]),32):
            z,f,current=(torch.as_tensor(a[start:start+32],dtype=torch.float32,device=model.E.device) for a in (cache["latent_history"],cache["latent_future"],raw["raw_last"]))
            p=model.decode(current,z,f); features=model.lag_features(z,f)
            for key,value in zip(parts,(p,z,f,features)): parts[key].append(value.detach().cpu().numpy().copy())
            check({"stage":"CACHED_ACTUAL_F_LATENT_RECON_DECODE","start":start,"new_F_forward":False})
    return {k:np.ascontiguousarray(np.concatenate(v),dtype=np.float32) for k,v in parts.items()}


def resolve_all(result,run,seal,registry):
    resolved,locks={},[]
    for state,seed in CONDITIONS:
        try: row=run(state,seed)
        except Exception as error: row={"status":"FAILED_PATH_PRESERVED","failure":{"type":type(error).__name__,"message":str(error)}}
        binding=seal(state,seed,row); row["selection_lock"]=binding; result["states"][state]["native"][str(seed)]=row
        locks.append({"state":state,"seed":seed,"status":row["status"],"selection_lock":binding}); result["training_updates_completed"]+=row.get("training_updates_completed",0); result["checkpoint_states_saved"]+=row.get("checkpoint_count",0)
        if row["status"]=="COMPLETE": resolved[(state,seed)]=row; result["native_training_fits"]+=1
        else: result["failed_paths"].append({"state":state,"seed":seed,**row})
    result["selection_registry"]=registry(locks); result.update(resolved_selection_locks=len(locks),all_6_locks_before_DEV=True); return resolved


def execute(args):
    e_only,joint,full,science,parents,refs=load_gates(args.e_only_result,args.e_only_audit,args.report,args.report_manifest,args.report_audit)
    output=old.fresh_output(args.tag); sys.path[:0]=[str(HERE),str(BASE),str(OLD),str(ROOT)]
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    import fixed_decoder_span_v1 as io
    import continuation_runtime_v1 as rt
    import lagged_native_head_v1 as native
    from native_head_v1 import state_hash
    from linear_e_adaptation_v1 import LinearLagAdapter
    from decoder_metric_probe_v1 import masked_scores
    from run_fit_target_comparison_v1 import prior_file_records
    from prepare_inputs import install_protected_test_guard
    output.mkdir(parents=True,exist_ok=False); meter=rt.Meter("matched_L7_native_"+args.tag,gpu=True)
    result={"status":"RUNNING",**refs,"states":{s:{"native":{},"Bolt":{}} for s in STATES},"failed_paths":[],"source_records":[],"native_training_fits":0,"training_updates_completed":0,"checkpoint_states_saved":0,"resolved_selection_locks":0,"probe_optimizer_updates":0,"all_6_locks_before_DEV":False,"cached_Bolt_contexts_reused":0,"matched_Bolt_C_decodes":0,"E0_new_matched_RECON_C_decodes":0,"selected_E_only_Bolt_C_byte_checks":0,"new_F_forward_calls":0,"new_F_model_objects_loaded":0,"new_F_state_hash_computed":False,"new_backbone_training":False,"new_E_training_updates":0,"new_D_training_updates":0,"whole_deployment_cost_measured":False,"new_output_space_diagnostics_computed_by_training_producer":False,"independent_holdout":False,"strict_K_only":False,"protected_TEST_accessed":False,"encoder_backend":"CUDA FP32; canonical32 prediction; native training B4","gradient_clipping_parameter_scope":"HEAD_ONLY","F_parent_signatures_reused":{"before":e_only["F_before"],"after":e_only["F_after"]},"NumPy_context_parity_is_primary_gate":False,"optional_NumPy_context_parity_checks":0,"optional_NumPy_context_parity_failures":[]}
    checked,contexts,states,caches,models,baseline_models={},{},{},{},{},{}; failure=None
    try:
        install_protected_test_guard(); meter.protected_check(); protected=prior_file_records(e_only,joint,full,science,*parents.values(),*refs.values())
        folder=output/"sources"; folder.mkdir()
        for path in (Path(__file__),HERE/"lagged_native_head_v1.py",HERE/"PLAN_MATCHED_L7_NATIVE_KO_v1.md",HERE/"run_linear_e_adaptation_v1.py",HERE/"linear_e_adaptation_v1.py",HERE/"native_head_v1.py",HERE/"causal_lag_decoder_v1.py",HERE/"fixed_decoder_span_v1.py",HERE/"continuation_runtime_v1.py",OLD/"prepare_inputs.py"):
            destination=folder/path.name; shutil.copyfile(path,destination); result["source_records"].append({**old.record(path),"snapshot_path":str(destination)})
        result.update(controller_source=old.record(__file__),native_source=old.record(HERE/"lagged_native_head_v1.py"),plan=old.record(HERE/"PLAN_MATCHED_L7_NATIVE_KO_v1.md"))
        torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
        recon=science["datasets"]["robin"]["spaces"]["PCA"]["conditions"]["L7_RECON"]
        def load(item): return io.checked_payload({**item,"path":str(old.bound(item))},meter,checked)
        def context(split):
            require(split in ("fit","tune") or result["all_6_locks_before_DEV"],"SIX_LOCK_REGISTRY_BEFORE_ANY_DEV_ARRAY_OR_DECODE")
            if split not in contexts:
                raw=load(recon["splits"][split]["input_source"]); n={"fit":256,"tune":64,"val":30}[split]
                require(raw["history"].dtype==raw["target"].dtype==np.float32 and raw["history"].shape==(n,512,17) and raw["target"].shape==raw["observed"].shape==(n,48,17) and raw["observed"].dtype==np.bool_ and np.isfinite(raw["history"]).all() and np.isfinite(raw["target"][raw["observed"]]).all() and old.byte_equal(raw["raw_last"],raw["history"][:,-1]),"EXACT_ORIGINAL_ROBIN_PAST_ACTUAL_Y_CURRENT_C"); contexts[split]=raw
            return contexts[split]
        def frozen(state):
            if state not in states:
                source=frozen_source(e_only,joint,state); value=load(source)
                require(set(value)=={"E","D","center"} and value["E"].shape==(17,4) and value["D"].shape==(28,17) and value["center"].shape==(17,) and all(v.dtype==np.float32 and np.isfinite(v).all() for v in value.values()),"FINITE_ORIGINAL_FROZEN_L7_STATE")
                if state=="RECON_CP0":
                    initial_ref=full["datasets"]["robin"]["conditions"]["robin_L7_E_LEARN_s92601"]["CP0_TUNE"]["state"]; initial=load(initial_ref); original_D=load(recon["state_artifact"])
                    require(all(old.byte_equal(value[k],initial[k]) for k in ("E","center")) and old.byte_equal(value["D"],original_D["decoder_FP32"]) and old.byte_equal(original_D["lags"],np.asarray(LAGS,np.int64)),"EXACT_ORIGINAL_PCA_E_CENTER_AND_RECON_D")
                    result["states"][state]["original_PCA_E_center_CP0_source"]=initial_ref; result["states"][state]["original_RECON_D_source"]=recon["state_artifact"]
                else: require(all(old.byte_equal(value[k],frozen("RECON_CP0")[k]) for k in ("D","center")),"ALL_THREE_SAME_RECON_D_AND_CENTER")
                result["states"][state].update(frozen_state_source=source,frozen_state_value_sha256=state_hash({k:torch.as_tensor(v) for k,v in value.items()}),E_previously_supervised_by_FIT_future=state!="RECON_CP0",D_previously_supervised_by_FIT_future=False); states[state]=value
            return states[state]
        def cache(state,split):
            key=state,split
            if key not in caches:
                raw=context(split); source=cache_source(e_only,joint,full,state,split); value=load(source)
                for k in ("target","observed","raw_last","origins","mean","std"): require(old.byte_equal(raw[k],value[k]),"EXACT_CACHED_ACTUAL_F_INPUT_BINDING:"+k)
                require(value["latent_history"].dtype==value["latent_future"].dtype==np.float32 and value["latent_history"].shape==(len(raw["history"]),512,4) and value["latent_future"].shape==(len(raw["history"]),48,4) and np.isfinite(value["latent_history"]).all() and np.isfinite(value["latent_future"]).all(),"FINITE_CANONICAL_TORCH_ACTUAL_F_CACHE")
                caches[key]=value; result["cached_Bolt_contexts_reused"]+=1
            return caches[key]
        def new_model(state):
            source=frozen(state); model=native.LaggedNativeForecaster(source["center"],source["E"],source["D"],LAGS).to("cuda").eval(); native.frozen_status(model)
            require(all(old.byte_equal(v.detach().cpu().numpy(),source[k]) for k,v in model.state_dict().items() if k in source),"EXACT_SOURCE_FROZEN_E_D_CENTER_BYTES"); return model
        def save_forecast(path,outputs,raw):
            mask=np.broadcast_to(raw["observed"].all(2,keepdims=True),raw["observed"].shape)
            return {"artifact":io.save_npz(path,{**outputs,**{k:raw[k] for k in ("target","observed","raw_last","origins","mean","std")}}),"scores_all_observed":masked_scores(outputs["prediction"],raw["target"],raw["observed"],raw["std"]),"scores_complete_mask":masked_scores(outputs["prediction"],raw["target"],mask,raw["std"]) if mask.any() else {"status":"NO_COMPLETE_C"}}
        def Bolt_baseline(state,split):
            owner=result["states"][state]
            if split not in owner["Bolt"]:
                raw=context(split); source=cache_source(e_only,joint,full,state,split); cached=cache(state,split)
                if state not in baseline_models:
                    value=frozen(state); baseline_models[state]=LinearLagAdapter(value["center"],value["E"],value["D"],LAGS,learn_E=False).to("cuda").eval().requires_grad_(False)
                outputs=decode_cached_Bolt(baseline_models[state],cached,raw,meter.check)
                if state!="RECON_CP0" or split=="tune": require(old.byte_equal(outputs["prediction"],cached["prediction"]),"ORIGINAL_RECON_CACHED_BOLT_C_PREDICTION_BYTES")
                value={**save_forecast(output/(state+"_BOLT_"+split+".npz"),outputs,raw),"actual_F_component_source":source,"new_F_forward":False,"fresh_C_decode_only":True,"source_C_prediction_reused":False,"source_C_used_FORECAST_D":state=="RECON_CP0" and split!="tune"}
                if state=="RECON_CP0":
                    value["NumPy_context_reference"]=recon["splits"][split]["artifact"]; result["optional_NumPy_context_parity_checks"]+=1
                    try:
                        legacy=load(value["NumPy_context_reference"]); value["NumPy_RECON_context_parity"]=old.parity(outputs["prediction"],legacy["ACTUAL_F_prediction_FP32"])
                        if not value["NumPy_RECON_context_parity"]["pass"]: result["optional_NumPy_context_parity_failures"].append({"state":state,"split":split,"reason":"Original NumPy E/F context differs from matched CUDA E/cache path beyond registered tolerance; not a matched-path gate",**value["NumPy_RECON_context_parity"]})
                    except Exception as error:
                        value["NumPy_RECON_context_parity"]={"status":"FAILED_CONTEXT_PRESERVED","type":type(error).__name__,"message":str(error)}; result["optional_NumPy_context_parity_failures"].append({"state":state,"split":split,**value["NumPy_RECON_context_parity"]})
                    result["E0_new_matched_RECON_C_decodes"]+=1
                else: result["selected_E_only_Bolt_C_byte_checks"]+=1
                owner["Bolt"][split]=value; result["matched_Bolt_C_decodes"]+=1
            return owner["Bolt"][split]
        def run(state,seed):
            label=state+"_NATIVE_s"+str(seed); row={"status":"FAILED_PATH_PRESERVED","state":state,"seed":seed,"training_updates_completed":0,"checkpoint_count":0}; checkpoints=[]
            try:
                fit,tune=context("fit"),context("tune"); cached=cache(state,"tune"); source=frozen(state); model=new_model(state)
                row.update(frozen_state_source=frozen_source(e_only,joint,state),initial_full_state_sha256=state_hash(model.state_dict()),initial_head_state_sha256=state_hash(model.head.state_dict()),initial_frozen_status=native.frozen_status(model))
                outputs=old.predict_components(model,None,tune["history"],meter.check); probe,arrays=cp0_probe(outputs,cached,source,tune)
                probe.update(artifact=io.save_npz(output/(label+"_CP0_PARITY.npz"),{**arrays,"origins":tune["origins"]}),actual_Bolt_component_source=cache_source(e_only,joint,full,state,"tune")); row["probe"]=probe; require(probe["status"]=="PASS","NATIVE_CP0_PARITY_FAILED_ARRAYS_PRESERVED")
                def checkpoint(update,torch_state,score,values):
                    require(old.byte_equal(values["latent_history"],cached["latent_history"]),"EVERY_CP_SAME_FROZEN_TORCH_E_HISTORY_BYTES")
                    item={"update":update,"tune":score,"state":io.save_npz(output/(label+"_CP"+str(update)+"_STATE.npz"),{k:v.numpy().copy() for k,v in torch_state.items()}),"frozen_status":native.frozen_status(model),"head_state_sha256":state_hash(model.head.state_dict()),"TUNE_latent_variance":np.var(values["latent_history"].astype(np.float64),axis=(0,1)).tolist(),**save_forecast(output/(label+"_CP"+str(update)+"_TUNE.npz"),values,tune)}
                    checkpoints.append(item); row["checkpoint_count"]+=1; return item
                path=output/(label+"_UPDATES.jsonl")
                with path.open("x",encoding="utf-8") as log:
                    def check(event=None):
                        meter.check(event)
                        if event and event.get("stage")=="OPTIMIZER_STEP_EXECUTED": row["training_updates_completed"]=event["update"]
                        if event and event.get("stage")=="TRAINING_UPDATE": log.write(json.dumps(event,allow_nan=False)+"\n"); log.flush(); row["training_updates_completed"]=event["update"]
                    trained=native.train_head(model,fit,tune,seed,check,checkpoint)
                row.update({k:v for k,v in trained.items() if k!="model"}); require(row["order_sha256"]==e_only["datasets"]["robin"]["conditions"][str(seed)]["order_sha256"],"SAME_HEAD_SEED_600_BY_4_ORDER_AS_PREVIOUS_FIT")
                fitted=old.predict_components(model,None,fit["history"],meter.check); require(old.byte_equal(fitted["latent_history"],cache(state,"fit")["latent_history"]),"SELECTED_FIT_SAME_FROZEN_TORCH_E_HISTORY_BYTES")
                row.update(status="COMPLETE",checkpoints=checkpoints,CP0_TUNE=checkpoints[0],updates_log=old.record(path),selected_state=next(p["state"] for p in checkpoints if p["update"]==trained["selected_update"]),selected_head_state_sha256=state_hash(model.head.state_dict()),FIT=save_forecast(output/(label+"_SELECTED_FIT.npz"),fitted,fit)); models[(state,seed)]=model
            except Exception as error:
                row.update(status="FAILED_PATH_PRESERVED",failure={"type":type(error).__name__,"message":str(error)},checkpoints=checkpoints)
                if (output/(label+"_UPDATES.jsonl")).exists(): row["updates_log"]=old.record(output/(label+"_UPDATES.jsonl"))
            return row
        def seal(state,seed,row):
            path=output/(state+"_s"+str(seed)+"_SELECTION_LOCK.json"); rt.write_new(path,{**row,"DEV_opened_before_lock":False,"sealed_unix":time.time()}); return old.record(path)
        def registry(rows):
            path=output/"SELECTION_REGISTRY.json"; rt.write_new(path,{"status":"ALL_SIX_RESOLVED_BEFORE_DEV","conditions":rows,"DEV_opened_before_seal":False,"sealed_unix":time.time()}); return old.record(path)
        with threadpool_limits(limits=2):
            resolved=resolve_all(result,run,seal,registry)
            for state in STATES:
                for split in ("fit","tune","val"):
                    try: Bolt_baseline(state,split)
                    except Exception as error: result["failed_paths"].append({"state":state,"split":split,"stage":"BOLT_CACHED_COMPONENT_DECODE","type":type(error).__name__,"message":str(error)})
            for (state,seed),row in resolved.items():
                try:
                    old.bound(row["selection_lock"]); old.bound(result["selection_registry"]); raw=context("val"); values=old.predict_components(models[(state,seed)],None,raw["history"],meter.check)
                    require(old.byte_equal(values["latent_history"],cache(state,"val")["latent_history"]),"SELECTED_DEV_SAME_FROZEN_TORCH_E_HISTORY_BYTES"); row["DEV"]=save_forecast(output/(state+"_NATIVE_s"+str(seed)+"_SELECTED_DEV.npz"),values,raw)
                except Exception as error: row["DEV_failure"]={"type":type(error).__name__,"message":str(error)}; result["failed_paths"].append({"state":state,"seed":seed,"stage":"DEV",**row["DEV_failure"]})
        for key,model in models.items():
            source=states[key[0]]; require(all(old.byte_equal(model.state_dict()[k].detach().cpu().numpy(),source[k]) for k in source),"FINAL_FROZEN_E_D_CENTER_BYTES"); native.frozen_status(model)
        for item in protected+result["source_records"]+list(checked.values()): old.bound(item)
        for item in result["source_records"]: require(old.record(item["snapshot_path"])["sha256"]==item["sha256"],"SOURCE_SNAPSHOT_UNCHANGED")
        result.update(input_records=list(checked.values()),protected_prior_records=protected,status="COMPLETE" if result["native_training_fits"]==6 and result["training_updates_completed"]==3600 and result["checkpoint_states_saved"]==30 and result["matched_Bolt_C_decodes"]==9 and not result["failed_paths"] else "PARTIAL_COMPLETED"); meter.protected_check()
    except Exception as error: failure={"type":type(error).__name__,"message":str(error)}; result.update(status="FAILED_PRESERVED",failure=failure)
    finally: result["resource_terminal"]=meter.close(result["status"],failure); rt.write_new(output/"RESULT.json",result)
    print(json.dumps({"status":result["status"],"result":str(output/"RESULT.json")}),flush=True); return 0 if result["status"]=="COMPLETE" else 2


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("tag","e-only-result","e-only-audit","report","report-manifest","report-audit"): parser.add_argument("--"+key,required=True)
    raise SystemExit(execute(parser.parse_args()))
