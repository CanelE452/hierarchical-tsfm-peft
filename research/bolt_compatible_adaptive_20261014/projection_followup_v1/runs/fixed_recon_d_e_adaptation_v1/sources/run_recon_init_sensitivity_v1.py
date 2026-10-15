"""Robin L7: two fresh full-gradient fits from the existing RECON D initialization."""
from pathlib import Path
import argparse
import json
import shutil
import sys
import time

import run_linear_e_adaptation_v1 as old
from run_linear_e_adaptation_v1 import train_condition, predict_components

HERE, ROOT, BASE, OLD = old.HERE, old.ROOT, old.BASE, old.OLD
SEEDS, LAGS = (92601,92602), (0,1,2,4,8,16,32)
REPORT_SHA = "a057dc5f202c146156044f7e4aea5987c93ea2c0b44e5ea3161b19aab0ea2e0c"
MANIFEST_SHA = "d231904818058a0c8cc3b7ed8e62b6264ab8f6ce28fa7c6f8ae4ed7ed068ee5e"
REPORT_AUDIT_SHA = "d4f8657dd39e6f8370b2da729efb5cb2b0fd8a2db465d79829736398f05d42bd"
SPACE_SHA = "cf042cc764ff36cbb877211777721783efadbaab6ba8741268eaccba26cbb34b"
SPACE_AUDIT_SHA = "1b0dbacb711becce70ac877825cd53b84069918ff62ac79dc9b9b5d8ac410738"
require = old.require


def initial_state(encoder, recon_decoder, forecast_initial):
    import numpy as np
    state={"E":encoder["encoder"].astype(np.float32),"center":encoder["center"].astype(np.float32),"D":recon_decoder["decoder_FP32"].copy()}
    c,k=state["E"].shape
    require(state["D"].dtype==np.float32 and state["D"].shape==(len(LAGS)*k,c) and state["center"].shape==(c,) and all(np.isfinite(x).all() for x in state.values()) and old.byte_equal(recon_decoder["lags"],np.asarray(LAGS,np.int64)),"ORIGINAL_FINITE_FP32_RECON_L7_D")
    require(all(old.byte_equal(state[key],forecast_initial[key]) for key in ("E","center")),"SAME_ORIGINAL_PCA_E_CENTER_AS_FORECAST_INITIALIZATION")
    return state


def cp0_parity(actual, reference):
    from causal_lag_decoder_v1 import lagged_delta_features
    expected={"prediction":reference["ACTUAL_F_prediction_FP32"],"latent_history":reference["latent_history"],"latent_future":reference["actual_F_future"],"lag_features":lagged_delta_features(reference["latent_history"],reference["actual_F_future"],LAGS)}
    checks={name:old.parity(actual[name],value) for name,value in expected.items()}
    return {"status":"PASS" if all(v["pass"] for v in checks.values()) else "FAILED_PARITY_PRESERVED","optimizer_updates":0,"checks":checks,"scope":"Actual canonical32 Torch CP0 versus prior fixed-PCA L7_RECON cached NumPy E/D and actual F; original registered tolerances, no numeric repair"}, {**{"Torch_"+k:v for k,v in actual.items()},**{"previous_RECON_"+k:v for k,v in expected.items()}}


def resolve_two(report, run, seal, registry):
    resolved,rows={},[]
    for seed in SEEDS:
        try: row=run(seed)
        except Exception as error: row={"status":"FAILED_PATH_PRESERVED","failure":{"type":type(error).__name__,"message":str(error)}}
        lock=seal(seed,row); row["selection_lock"]=lock; report["datasets"]["robin"]["conditions"][str(seed)]=row
        rows.append({"dataset":"robin","seed":seed,"status":row["status"],"selection_lock":lock})
        report["training_updates_completed"]+=row.get("training_updates_completed",0); report["checkpoint_states_saved"]+=row.get("checkpoint_count",0)
        if row["status"]=="COMPLETE": resolved[seed]=row; report["recon_initialization_training_fits"]+=1
        else: report["failed_paths"].append({"dataset":"robin","seed":seed,**row})
    report["selection_registry"]=registry(rows); report.update(resolved_selection_locks=len(rows),all_2_locks_before_DEV=True)
    return resolved


def load_gates(training,audit,report,manifest,report_audit):
    require(all(isinstance(x,str) and len(x)==64 for x in (REPORT_SHA,MANIFEST_SHA,REPORT_AUDIT_SHA)),"CURRENT_OUTPUT_SPACE_REPORT_NOT_REGISTERED")
    refs={key:old.record(path) for key,path in (("parent_report",report),("parent_report_manifest",manifest),("parent_report_audit",report_audit))}
    require([x["sha256"] for x in refs.values()]==[REPORT_SHA,MANIFEST_SHA,REPORT_AUDIT_SHA],"EXACT_COMPLETED_CURRENT_REPORT_BUNDLE")
    checked,files=old.read(report_audit),old.read(manifest)
    require(checked["status"]=="PASS" and files["status"]=="COMPLETE","CURRENT_REPORT_COMPLETE_AND_INDEPENDENT_PASS")
    old.bound(checked["report"],refs["parent_report"]); old.bound(checked["manifest"],refs["parent_report_manifest"])
    for item in files["files"]:
        path=(old.resolve(manifest).parent/item["path"]).resolve(); require(path.is_relative_to(old.resolve(manifest).parent),"PARENT_REPORT_FILE_CONTAINMENT"); old.bound({**item,"path":str(path)})
    space_ref,space_audit_ref=(files["inputs"][key] for key in ("diagnostic","audit"))
    require(space_ref["sha256"]==SPACE_SHA and space_audit_ref["sha256"]==SPACE_AUDIT_SHA,"EXACT_LEARNED_SPACE_DIAGNOSTIC_AND_PASS")
    space,space_audit=old.read(old.bound(space_ref)),old.read(old.bound(space_audit_ref))
    require(space["status"]=="COMPLETE" and space_audit["status"]=="PASS" and space_audit["totals"]["selected_states_checked"]==12 and space_audit["totals"]["diagnostic_cells_checked"]==24,"TWELVE_STATE_TWENTY_FOUR_CELL_DIAGNOSIS_COMPLETED")
    old.bound(space_audit["diagnostic"],space_ref); old.bound(space_audit["audit_source"])
    from run_linear_e_cost_v1 import load_gates as full_gates
    full,full_audit,science,subset,original,_,_=full_gates(training,audit)
    old.bound(space["full_gradient_parent"],old.record(training)); old.bound(space["full_gradient_parent_audit"],old.record(audit))
    refs.update(full_gradient_parent=old.record(training),full_gradient_parent_audit=old.record(audit),learned_output_space_diagnostic=space_ref,learned_output_space_audit=space_audit_ref,causal_lag_diagnostic=full["causal_lag_diagnostic"],causal_lag_audit=full["causal_lag_audit"])
    return full,full_audit,science,subset,original,space,space_audit,refs


def execute(args):
    full,full_audit,science,subset,original,space,space_audit,refs=load_gates(args.training_result,args.training_audit,args.report,args.report_manifest,args.report_audit)
    output=old.fresh_output(args.tag); sys.path[:0]=[str(HERE),str(BASE),str(OLD),str(ROOT)]
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    import continuation_runtime_v1 as rt
    import fixed_decoder_span_v1 as io
    import run_condition_probes as probes
    import cost_readout_v1 as runtime
    from linear_e_adaptation_v1 import LinearLagAdapter,parameter_diagnostics
    from native_head_v1 import state_hash
    from decoder_metric_probe_v1 import masked_scores
    from run_fit_target_comparison_v1 import prior_file_records
    from prepare_inputs import install_protected_test_guard
    output.mkdir(parents=True,exist_ok=False); meter=rt.Meter("RECON_init_"+args.tag,gpu=True)
    result={"status":"RUNNING",**refs,"datasets":{"robin":{"conditions":{}}},"failed_paths":[],"source_records":[],"recon_initialization_training_fits":0,"training_updates_completed":0,"checkpoint_states_saved":0,"probe_optimizer_updates":0,"forecast_initialization_conditions_reused":2,"new_forecast_initialization_fits":0,"new_backbone_training":False,"new_D_initialization_fit":False,"protected_TEST_accessed":False,"independent_holdout":False,"strict_K_only":False,"whole_deployment_cost_measured":False,"all_2_locks_before_DEV":False,"scope":"Adaptive initialization sensitivity: same prediction objective and training recipe, prior FIT RECON D versus reused FORECAST-LS D initialization; not reconstruction-versus-forecast objective training"}
    checked,contexts,models,failure={},{},{},None
    try:
        install_protected_test_guard(); meter.protected_check(); protected=prior_file_records(full,full_audit,science,subset,original,space,space_audit,*refs.values())
        folder=output/"sources"; folder.mkdir()
        for path in (Path(__file__),HERE/"PLAN_RECON_INIT_SENSITIVITY_KO_v1.md",HERE/"run_linear_e_adaptation_v1.py",HERE/"linear_e_adaptation_v1.py",HERE/"run_linear_e_cost_v1.py",HERE/"native_head_v1.py",HERE/"fixed_decoder_span_v1.py",HERE/"continuation_runtime_v1.py",OLD/"prepare_inputs.py",OLD/"run_condition_probes.py",BASE/"cost_readout_v1.py"):
            destination=folder/path.name; shutil.copyfile(path,destination); result["source_records"].append({**old.record(path),"snapshot_path":str(destination)})
        result.update(controller_source=old.record(__file__),plan=old.record(HERE/"PLAN_RECON_INIT_SENSITIVITY_KO_v1.md"))
        engine=probes.import_old_engine(); bolt=engine.ex.previous.FrozenBolt()
        result.update(F_before=probes.model_hashes(bolt),model_binary_before=probes.verified_file(original["model_binary_before"]),model_config_before=probes.verified_file(original["model_config_before"]))
        require(result["F_before"]==full["F_before"]==full["F_after"],"EXACT_ORIGINAL_FROZEN_F")
        result["runtime_behavior_contract"]=runtime.bolt_runtime_contract(engine,bolt,meter.check)
        torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
        owner=science["datasets"]["robin"]["spaces"]["PCA"]; recon=owner["conditions"]["L7_RECON"]
        def load(item): return io.checked_payload({**item,"path":str(old.bound(item))},meter,checked)
        def context(split):
            require(split in ("fit","tune") or result["all_2_locks_before_DEV"],"TWO_LOCK_REGISTRY_BEFORE_ANY_DEV_ARRAY")
            if split not in contexts:
                source=recon["splits"][split]["input_source"]; values=load(source); n={"fit":256,"tune":64,"val":30}[split]
                require(values["history"].shape==(n,512,17) and values["target"].shape==values["observed"].shape==(n,48,17) and np.isfinite(values["history"]).all() and np.isfinite(values["target"][values["observed"]]).all() and old.byte_equal(values["raw_last"],values["history"][:,-1]),"SAME_ORIGINAL_ROBIN_SPLIT_AND_CURRENT_C")
                contexts[split]=values
            return contexts[split]
        def save_forecast(path,outputs,values):
            mask=np.broadcast_to(values["observed"].all(2,keepdims=True),values["observed"].shape)
            return {"artifact":io.save_npz(path,{**outputs,**{k:values[k] for k in ("target","observed","raw_last","origins","mean","std")}}),"scores_all_observed":masked_scores(outputs["prediction"],values["target"],values["observed"],values["std"]),"scores_complete_mask":masked_scores(outputs["prediction"],values["target"],mask,values["std"]) if mask.any() else {"status":"NO_COMPLETE_C"}}
        def run(seed):
            label="robin_L7_RECON_INIT_s"+str(seed); row={"status":"FAILED_PATH_PRESERVED","dataset":"robin","seed":seed,"lag":"L7","learn_E":True,"training_updates_completed":0,"checkpoint_count":0}; checkpoints=[]
            try:
                fit,tune=context("fit"),context("tune"); previous=full["datasets"]["robin"]["conditions"]["robin_L7_E_LEARN_s"+str(seed)]
                e,d,forecast_initial=load(owner["E_source"]),load(recon["state_artifact"]),load(previous["CP0_TUNE"]["state"])
                state=initial_state(e,d,forecast_initial); require(state["E"].shape==(17,4),"ORIGINAL_ROBIN_PCA_K4")
                model=LinearLagAdapter(state["center"],state["E"],state["D"],LAGS,learn_E=True).to("cuda").eval()
                row.update(initial_state_sha256=state_hash(model.state_dict()),initial_sources={"PCA_E":owner["E_source"],"RECON_D":recon["state_artifact"],"FORECAST_LS_CP0":previous["CP0_TUNE"]["state"]},full_gradient_reference={"condition":"robin_L7_E_LEARN_s"+str(seed),"selected_update":previous["selected_update"],"selected_state":previous["selected_state"],"CP0_TUNE":previous["CP0_TUNE"]})
                baseline=recon["splits"]["tune"]; old_values=load(baseline["artifact"])
                for before,key in (("target","target"),("observed","observed"),("currentC","raw_last"),("origins","origins"),("FIT_mean","mean"),("FIT_std","std")): require(old.byte_equal(old_values[before],tune[key]),"SAME_CP0_RECON_CANONICAL_INPUT:"+key)
                before=state_hash(model.state_dict()); outputs=predict_components(model,bolt,tune["history"],meter.check); probe,arrays=cp0_parity(outputs,old_values)
                require(state_hash(model.state_dict())==before and all(p.grad is None for p in model.parameters()),"CP0_PROBE_NO_UPDATE_OR_GRAD_MUTATION")
                probe.update(artifact=io.save_npz(output/(label+"_CP0_RECON_PARITY.npz"),{**arrays,"origins":tune["origins"]}),reference=baseline["artifact"],initial_state_sha256=before)
                row["probe"]=probe; require(probe["status"]=="PASS","ORIGINAL_RECON_CP0_PARITY_FAILED_ARRAYS_PRESERVED")
                def checkpoint(update,torch_state,score,values):
                    item={"update":update,"tune":score,"state":io.save_npz(output/(label+"_CP"+str(update)+"_STATE.npz"),{k:v.numpy().copy() for k,v in torch_state.items()}),"parameter_diagnostics":parameter_diagnostics(model),"TUNE_latent_variance":np.var(values["latent_history"].astype(np.float64),axis=(0,1)).tolist(),**save_forecast(output/(label+"_CP"+str(update)+"_TUNE.npz"),values,tune)}
                    checkpoints.append(item); row["checkpoint_count"]+=1; return item
                log_path=output/(label+"_UPDATES.jsonl")
                with log_path.open("x",encoding="utf-8") as log:
                    def check(event=None):
                        meter.check(event)
                        if event and event.get("stage")=="OPTIMIZER_STEP_EXECUTED": row["training_updates_completed"]=event["update"]
                        if event and event.get("stage")=="TRAINING_UPDATE": log.write(json.dumps(event,allow_nan=False)+"\n"); log.flush(); row["training_updates_completed"]=event["update"]
                    trained=train_condition(model,bolt,fit,tune,seed,check,checkpoint)
                row.update({k:v for k,v in trained.items() if k not in ("model","checkpoints","learning_curve")})
                require(row["order_sha256"]==previous["order_sha256"],"SAME_SEED_600_BY_4_ORDER_AS_FORECAST_INITIALIZATION")
                row.update(status="COMPLETE",checkpoints=checkpoints,updates_log=old.record(log_path),selected_state=next(x["state"] for x in checkpoints if x["update"]==trained["selected_update"]),CP0_TUNE=checkpoints[0],FIT=save_forecast(output/(label+"_SELECTED_FIT.npz"),predict_components(model,bolt,fit["history"],meter.check),fit)); models[seed]=model
            except Exception as error:
                row.update(status="FAILED_PATH_PRESERVED",failure={"type":type(error).__name__,"message":str(error)},checkpoints=checkpoints)
                if (output/(label+"_UPDATES.jsonl")).exists(): row["updates_log"]=old.record(output/(label+"_UPDATES.jsonl"))
            return row
        def seal(seed,row):
            path=output/("robin_s"+str(seed)+"_SELECTION_LOCK.json"); rt.write_new(path,{**row,"DEV_opened_before_lock":False,"sealed_unix":time.time()}); return old.record(path)
        def registry(rows):
            path=output/"SELECTION_REGISTRY.json"; rt.write_new(path,{"status":"ALL_TWO_RESOLVED_BEFORE_DEV","conditions":rows,"DEV_opened_before_seal":False,"sealed_unix":time.time()}); return old.record(path)
        with threadpool_limits(limits=2):
            resolved=resolve_two(result,run,seal,registry)
            for seed,row in resolved.items():
                try:
                    old.bound(row["selection_lock"]); old.bound(result["selection_registry"]); values=context("val")
                    row["DEV"]=save_forecast(output/("robin_L7_RECON_INIT_s"+str(seed)+"_SELECTED_DEV.npz"),predict_components(models[seed],bolt,values["history"],meter.check),values)
                except Exception as error: row["DEV_failure"]={"type":type(error).__name__,"message":str(error)}; result["failed_paths"].append({"seed":seed,"stage":"DEV",**row["DEV_failure"]})
        result["F_after"]=probes.model_hashes(bolt); require(result["F_before"]==result["F_after"] and all(not p.requires_grad and p.grad is None for p in bolt.model.parameters()),"F_FROZEN_AND_GRAD_NONE_AFTER_TWO_FITS")
        for item in protected+result["source_records"]+list(checked.values())+[result["model_binary_before"],result["model_config_before"]]: old.bound(item)
        for item in result["source_records"]: require(old.record(item["snapshot_path"])["sha256"]==item["sha256"],"SOURCE_SNAPSHOT_UNCHANGED")
        result.update(input_records=list(checked.values()),protected_prior_records=protected,status="COMPLETE" if result["recon_initialization_training_fits"]==2 and not result["failed_paths"] else "PARTIAL_COMPLETED"); meter.protected_check()
    except Exception as error:
        failure={"type":type(error).__name__,"message":str(error)}; result.update(status="FAILED_PRESERVED",failure=failure)
    finally:
        result["resource_terminal"]=meter.close(result["status"],failure); rt.write_new(output/"RESULT.json",result)
    print(json.dumps({"status":result["status"],"result":str(output/"RESULT.json")}),flush=True)
    return 0 if result["status"]=="COMPLETE" else 2


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("tag","training-result","training-audit","report","report-manifest","report-audit"): parser.add_argument("--"+key,required=True)
    raise SystemExit(execute(parser.parse_args()))
