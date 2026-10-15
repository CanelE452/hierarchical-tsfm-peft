"""Gated fixed-Q actual frozen-F controls; no optimization or Q selection."""
from pathlib import Path
import argparse
import json
import shutil
import sys
import time

import run_linear_e_adaptation_v1 as old

HERE,ROOT,BASE,OLD,require=old.HERE,old.ROOT,old.BASE,old.OLD,old.require
STATES=("RECON_CP0","E_ONLY_s92601","E_ONLY_s92602")
CONTROLS=("IDENTITY","PERMUTATION","SIGN_FLIP","DENSE_SO_s92711","DENSE_SO_s92712")
CONDITIONS=tuple((state,control) for state in STATES for control in CONTROLS)
LAGS=(0,1,2,4,8,16,32)
REPORT_SHA="e8f1036dcc27ee2a460b6d87a0f9d1d03368067dab5cb3abd32ec2ec8e368c4f"
MANIFEST_SHA="e5a17e7f5131ed611b87beca2d49ec4a05e093c2eaf1d0f42d0343b5f0350198"
REQUIRE_REPORT_AUDIT_SHA="2e98180c7e987e2be807eb00914d9da8876690490daa2b3b56c2f9ce39603dfc"
REPORT_SOURCE_SHA="3b91764dc6864dd09f1f68a07bd94f11094a002f7f508fc2f97d388e147ce1ad"
MODULE_SHA="ac515b9cb064e80d89c30a252f7bed4186c81eceb0c1c9d6db1fdbe5cab41491"


def load_gates(args):
    require(all(isinstance(v,str) and len(v)==64 for v in (REPORT_SHA,MANIFEST_SHA,REQUIRE_REPORT_AUDIT_SHA,MODULE_SHA)),"ACCEPTED_CALIBRATION_REPORT_AUDIT_NOT_YET_REGISTERED")
    refs={key:old.record(getattr(args,arg)) for key,arg in (("parent_report","report"),("parent_report_manifest","report_manifest"),("parent_report_audit","report_audit"))}
    require([refs[k]["sha256"] for k in refs]==[REPORT_SHA,MANIFEST_SHA,REQUIRE_REPORT_AUDIT_SHA],"EXACT_ACCEPTED_CALIBRATION_REPORT")
    manifest,audit=old.read(args.report_manifest),old.read(args.report_audit)
    require(manifest["status"]=="COMPLETE" and audit["status"]=="PASS" and audit["files_checked"]==23
        and audit["PNG_count"]==audit["SVG_count"]==4 and manifest["protected_TEST_accessed"] is audit["protected_TEST_accessed"] is False,"FULL_CALIBRATION_REPORT_ACCEPTANCE")
    old.bound(audit["report"],refs["parent_report"]);old.bound(audit["manifest"],refs["parent_report_manifest"])
    require(manifest["source"]["sha256"]==REPORT_SOURCE_SHA,"EXACT_CALIBRATION_REPORT_READER")
    old.bound(manifest["source"]);old.bound({"path":str(HERE/"orthogonal_latent_basis_v1.py"),"sha256":MODULE_SHA})
    for item in manifest["files"]:
        path=(old.resolve(args.report_manifest).parent/item["path"]).resolve()
        require(path.is_relative_to(old.resolve(args.report_manifest).parent),"PARENT_REPORT_CONTAINMENT");old.bound({**item,"path":str(path)})
    from make_fixed_l7_latent_calibration_report_v1 import load_verified
    paths={key:getattr(args,arg) for key,arg in (("train","calibration_result"),("audit","calibration_audit"),("cost","calibration_cost"),("cost_audit","calibration_cost_audit"))}
    values,bindings=load_verified(paths)
    for key in paths:old.bound(audit["inputs"][key],bindings[key]);old.bound(manifest["inputs"][key],bindings[key])
    refs.update({"latent_calibration_"+{"train":"diagnostic","audit":"audit","cost":"cost","cost_audit":"cost_audit"}[key]:bindings[key] for key in paths})
    for key in ("matched_l7_native_diagnostic","matched_l7_native_audit","E_only_diagnostic","E_only_audit","joint_recon_diagnostic","joint_recon_audit","causal_lag_diagnostic","causal_lag_audit"):refs[key]=values["train"][key];old.bound(refs[key])
    native=values["native"];old.bound(native["full_gradient_parent"]);full=old.read(native["full_gradient_parent"]["path"])
    refs.update(full_gradient_parent=native["full_gradient_parent"],calibration_report_metadata_reader=manifest["source"])
    return values["train"],native,values["science"],full,refs


def resolve_all(result,run,seal,registry):
    resolved,locks={},[]
    for state,control in CONDITIONS:
        try:row=run(state,control)
        except Exception as error:row={"status":"FAILED_PATH_PRESERVED","failure":{"type":type(error).__name__,"message":str(error)}}
        row["control_lock"]=seal(state,control,row);result["states"][state]["controls"][control]=row
        locks.append({"state":state,"control":control,"status":row["status"],"control_lock":row["control_lock"]})
        if row["status"]=="COMPLETE":resolved[state,control]=row
        else:result["failed_paths"].append({"state":state,"control":control,"stage":"FIT_TUNE_CONTROL_RESOLUTION","failure":row.get("failure"),"split_statuses":{k:v.get("status") for k,v in row.get("splits",{}).items()}})
    result["control_registry"]=registry(locks);result.update(resolved_control_locks=len(locks),all_15_locks_before_DEV=True)
    return resolved


def infer_development(result,resolved,infer):
    require(result["all_15_locks_before_DEV"] and result["resolved_control_locks"]==15,"ALL_FIFTEEN_LOCKS_BEFORE_ANY_DEV")
    for (state,control),row in resolved.items():
        try:
            row["splits"]["val"]=infer(state,control,row)
            require(row["splits"]["val"]["status"]=="COMPLETE","DEV_CONTROL_CHECKS_UNRESOLVED")
            result["DEV_evaluations_completed"]+=1
        except Exception as error:result["failed_paths"].append({"state":state,"control":control,"stage":"DEV_CONTROL","failure":{"type":type(error).__name__,"message":str(error)}})


def infer_control(model,bolt,history,cached,check,*,batch_size=32):
    import numpy as np
    import torch
    from orthogonal_latent_basis_v1 import analytic_transport
    keys=("prediction","old_past","rotated_past","F_rotated_future","unrotated_future","transport_future","transport_prediction")
    pieces={key:[] for key in keys};calls=0
    with torch.no_grad():
        for start in range(0,len(history),batch_size):
            x=torch.as_tensor(history[start:start+batch_size],dtype=torch.float32,device=model.E.device);capture={}
            def actual_F(rotated_past):
                nonlocal calls
                capture["rotated_past"]=rotated_past;capture["F_rotated_future"]=bolt(rotated_past);calls+=1
                return capture["F_rotated_future"]
            prediction,past,future=model(x,actual_F,return_components=True)
            original_past,original_future=(torch.as_tensor(cached[key][start:start+batch_size],dtype=torch.float32,device=model.E.device) for key in ("latent_history","latent_future"))
            transported=analytic_transport(original_future,model.basis())
            transported_C=model.decode(x[:,-1],original_past,transported)
            values=(prediction,past,capture["rotated_past"],capture["F_rotated_future"],future,transported,transported_C)
            for key,value in zip(keys,values):pieces[key].append(value.detach().cpu().numpy().copy())
            check({"stage":"ACTUAL_CANONICAL32_E_Q_F_QT_OLD_LAG_D","start":start,"new_F_forward":True})
    return {key:np.ascontiguousarray(np.concatenate(value),dtype=np.float32) for key,value in pieces.items()},calls


def difference(actual,reference):
    import numpy as np
    value=actual.astype(np.float64)-reference.astype(np.float64)
    return {"max_abs":float(np.abs(value).max(initial=0)),"RMS":float(np.sqrt(np.mean(value*value))),
        "by_horizon_max_abs":np.max(np.abs(value),axis=(0,2)).tolist(),"by_horizon_RMS":np.sqrt(np.mean(value*value,axis=(0,2))).tolist()}


def evaluate_arrays(outputs,raw,cached,frozen,Q,control):
    import numpy as np
    from causal_lag_decoder_v1 import predict
    from decoder_metric_probe_v1 import masked_scores
    required=("prediction","old_past","rotated_past","F_rotated_future","unrotated_future","transport_future","transport_prediction")
    require(set(outputs)==set(required) and all(v.dtype==np.float32 and np.isfinite(v).all() for v in outputs.values()),"FINITE_FP32_ACTUAL_CONTROL_ARRAYS")
    identity=control=="IDENTITY";old_past=outputs["old_past"]
    independent_future=outputs["F_rotated_future"] if identity else outputs["F_rotated_future"]@Q.T
    independent_C=predict(old_past,independent_future,raw["raw_last"],frozen["D"],LAGS)
    independent_transport=(cached["latent_future"]@Q)@Q.T
    checks={"old_past_vs_cached":old.parity(old_past,cached["latent_history"]),
        "NumPy_fresh_encode":old.parity(old_past,(raw["history"]-frozen["center"])@frozen["E"]),
        "NumPy_rotated_past":old.parity(outputs["rotated_past"],old_past if identity else old_past@Q),
        "NumPy_inverse_rotated_F_future":old.parity(outputs["unrotated_future"],independent_future),
        "NumPy_old_lag_D_actual_C":old.parity(outputs["prediction"],independent_C),
        "NumPy_analytic_future_transport":old.parity(outputs["transport_future"],independent_transport),
        "analytic_future_transport_vs_original":old.parity(outputs["transport_future"],cached["latent_future"]),
        "analytic_transport_old_C_vs_baseline":old.parity(outputs["transport_prediction"],cached["prediction"])}
    if control in ("IDENTITY","PERMUTATION"):
        checks["null_F_future_vs_original"]=old.parity(outputs["unrotated_future"],cached["latent_future"])
        checks["null_C_vs_original"]=old.parity(outputs["prediction"],cached["prediction"])
    complete=np.broadcast_to(raw["observed"].all(2,keepdims=True),raw["observed"].shape)
    def scores(prediction):
        return {"all_observed":masked_scores(prediction,raw["target"],raw["observed"],raw["std"]),
            "complete_C":masked_scores(prediction,raw["target"],complete,raw["std"]) if complete.any() else {"status":"NO_COMPLETE_C"}}
    passed=all(check["pass"] for check in checks.values())
    row={"status":"COMPLETE" if passed else "FAILED_PARITY_PRESERVED","checks":checks,
        "null_control_required":control in ("IDENTITY","PERMUTATION"),"sign_or_dense_equivariance_required":False,
        "scores":scores(outputs["prediction"]),"baseline_Bolt_scores":scores(cached["prediction"]),
        "RAW_BOLT_scores":scores(raw["raw_forecast"]),"transport_scores":scores(outputs["transport_prediction"]),
        "latent_commutator":difference(outputs["unrotated_future"],cached["latent_future"]),
        "C_commutator":difference(outputs["prediction"],cached["prediction"]),
        "transport_future_gap":difference(outputs["transport_future"],cached["latent_future"]),
        "transport_C_gap":difference(outputs["transport_prediction"],cached["prediction"]),
        "complete_target_positions":int(complete[:,:,0].sum()),"observed_target_values":int(raw["observed"].sum()),
        "future_labels_used_for_forward_or_Q_choice":False,"performance_improvement_required_for_execution_success":False}
    return row,{"independent_NumPy_future":independent_future,"independent_NumPy_prediction":independent_C}


def execute(args):
    calibration,native,science,full,refs=load_gates(args);output=old.fresh_output(args.tag);sys.path[:0]=[str(HERE),str(BASE),str(OLD),str(ROOT)]
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    import orthogonal_latent_basis_v1 as module
    import fixed_decoder_span_v1 as io
    import continuation_runtime_v1 as rt
    import run_condition_probes as probes
    import cost_readout_v1 as runtime
    from prepare_inputs import install_protected_test_guard
    from run_fit_target_comparison_v1 import prior_file_records
    require(old.record(module.__file__)["sha256"]==MODULE_SHA,"FROZEN_ORTHOGONAL_PURE_SOURCE")
    output.mkdir(parents=True,exist_ok=False);meter=rt.Meter("orthogonal_basis_controls_"+args.tag,gpu=True);failure=None
    result={"status":"RUNNING",**refs,"states":{s:{"controls":{}} for s in STATES},"failed_paths":[],"source_records":[],
        "FIT_TUNE_evaluations_completed":0,"DEV_evaluations_completed":0,"resolved_control_locks":0,"all_15_locks_before_DEV":False,
        "new_F_forward_calls":0,"new_F_forward_attempts":0,"new_F_model_objects_loaded":0,"new_optimizer_updates":0,"new_Q_learning_updates":0,
        "new_E_D_training_updates":0,"new_backbone_training":False,"Q_selection_performed":False,"all_fixed_controls_reported":True,
        "protected_TEST_accessed":False,"independent_holdout":False,"strict_K_only":False,"whole_deployment_cost_measured":False,
        "encoder_and_F_backend":"CUDA FP32 canonical32 actual E/Q/F/QT/original L7 D; no cache returned in place of F",
        "control_resolution_is_selection":False,"performance_success_claimed_from_parity":False}
    checked,contexts,caches,states,models,initial={},{},{},{},{},{};protected=[]
    try:
        install_protected_test_guard();meter.protected_check();protected=prior_file_records(calibration,native,science,full,refs)
        folder=output/"sources";folder.mkdir()
        for path in (Path(__file__),HERE/"orthogonal_latent_basis_v1.py",HERE/"linear_e_adaptation_v1.py",HERE/"causal_lag_decoder_v1.py",HERE/"decoder_metric_probe_v1.py",HERE/"fixed_decoder_span_v1.py",HERE/"continuation_runtime_v1.py",HERE/"PLAN_METHODOLOGY_COMPLETION_KO_v1.md",HERE/"make_fixed_l7_latent_calibration_report_v1.py",OLD/"run_condition_probes.py",OLD/"prepare_inputs.py",BASE/"cost_readout_v1.py"):
            destination=folder/path.name;shutil.copyfile(path,destination);result["source_records"].append({**old.record(path),"snapshot_path":str(destination)})
        result.update(controller_source=old.record(__file__),module_source=old.record(module.__file__),plan=old.record(HERE/"PLAN_METHODOLOGY_COMPLETION_KO_v1.md"))
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        subset=old.read(old.bound(science["raw_subset_diagnostic"]));original=old.read(old.bound(subset["source_producer"]))
        engine=probes.import_old_engine();weight,config=(probes.verified_file(original[key]) for key in ("model_binary_before","model_config_before"));bolt=engine.ex.previous.FrozenBolt();result["new_F_model_objects_loaded"]=1
        result.update(F_before=probes.model_hashes(bolt),model_binary_before=weight,model_config_before=config)
        require(result["F_before"]==original["F_before"]==original["F_after"]==calibration["F_parent_signatures_reused"]["before"]==calibration["F_parent_signatures_reused"]["after"] and all(not p.requires_grad and p.grad is None for p in bolt.model.parameters()),"EXACT_FROZEN_F_DUAL_SIGNATURE_AND_NO_GRADIENT")
        result["runtime_behavior_contract"]=runtime.bolt_runtime_contract(engine,bolt,meter.check);require(result["runtime_behavior_contract"]==full["runtime_behavior_contract"],"UNCHANGED_F_RUNTIME_CONTRACT")
        controls=module.fixed_orthogonal_controls();require(tuple(controls)==CONTROLS,"EXACT_FIVE_PREDECLARED_CONTROLS")
        def actual_F(z):
            result["new_F_forward_attempts"]+=1;future=bolt(z);result["new_F_forward_calls"]+=1;return future
        def load(item):return io.checked_payload({**item,"path":str(old.bound(item))},meter,checked)
        def context(state,split):
            require(split in ("fit","tune") or result["all_15_locks_before_DEV"],"FIFTEEN_CONTROL_LOCKS_BEFORE_DEV_ARRAY_IO")
            row=calibration["states"][state]["families"]["SCALE_ONLY"][{"fit":"FIT","tune":"TUNE","val":"DEV"}[split]];source=row["input_source"]
            if split not in contexts:
                raw=load(source);n={"fit":256,"tune":64,"val":30}[split]
                require(raw["history"].dtype==raw["target"].dtype==np.float32 and raw["history"].shape==(n,512,17) and raw["target"].shape==raw["observed"].shape==(n,48,17) and raw["observed"].dtype==np.bool_ and np.isfinite(raw["history"]).all() and np.isfinite(raw["target"][raw["observed"]]).all() and old.byte_equal(raw["raw_last"],raw["history"][:,-1]),"SAME_ROBIN_CAUSAL_INPUT_OBSERVED_ACTUAL_Y")
                contexts[split]=(raw,source)
            old.bound(source,contexts[split][1]);return contexts[split]
        def cache(state,split):
            if (state,split) not in caches:
                raw,_=context(state,split);source=native["states"][state]["Bolt"][split]["artifact"]
                qrow=calibration["states"][state]["families"]["SCALE_ONLY"][{"fit":"FIT","tune":"TUNE","val":"DEV"}[split]];old.bound(qrow["cached_Bolt_source"],source);value=load(source)
                for key in ("target","observed","raw_last","origins","mean","std"):require(old.byte_equal(value[key],raw[key]),"EXACT_MATCHED_BOLT_CONTEXT:"+key)
                require(value["latent_history"].shape==(len(raw["history"]),512,4) and value["latent_future"].shape==(len(raw["history"]),48,4) and value["prediction"].shape==raw["target"].shape and all(value[k].dtype==np.float32 and np.isfinite(value[k]).all() for k in ("latent_history","latent_future","prediction")),"FINITE_MATCHED_BOLT_BASELINE")
                caches[state,split]=value
            return caches[state,split]
        def model_for(state,control):
            if state not in states:
                source=native["states"][state]["frozen_state_source"];old.bound(source,calibration["states"][state]["frozen_state_source"]);v=load(source)
                require(set(v)=={"E","D","center"} and v["E"].shape==(17,4) and v["D"].shape==(28,17) and v["center"].shape==(17,) and all(a.dtype==np.float32 and np.isfinite(a).all() for a in v.values()),"ORIGINAL_E_D_CENTER_ONLY")
                states[state]=v;result["states"][state]["frozen_state_source"]=source
            if (state,control) not in models:
                v=states[state];model=module.OrthogonalLatentBasisAdapter(v["center"],v["E"],v["D"],LAGS,controls[control]["Q"],learn_Q=False,identity=control=="IDENTITY").to("cuda").eval().requires_grad_(False);model.frozen_status()
                require(all(old.byte_equal(model.state_dict()[k].detach().cpu().numpy(),v[k]) for k in v),"EXACT_E_D_CENTER_BYTES_WITH_FIXED_Q")
                models[state,control]=model;initial[state,control]=old.state_arrays(model)
            return models[state,control]
        def evaluate(state,control,split):
            raw,input_source=context(state,split);cached=cache(state,split);model=model_for(state,control)
            outputs,calls=infer_control(model,actual_F,raw["history"],cached,meter.check)
            artifact=io.save_npz(output/(state+"_"+control+"_"+split+".npz"),{**outputs,"origins":raw["origins"]})
            try:
                row,independent=evaluate_arrays(outputs,raw,cached,states[state],model.basis().detach().cpu().numpy(),control)
                row["independent_artifact"]=io.save_npz(output/(state+"_"+control+"_"+split+"_INDEPENDENT.npz"),independent)
            except Exception as error:row={"status":"FAILED_DIAGNOSTIC_PRESERVED","failure":{"type":type(error).__name__,"message":str(error)}}
            row.update(artifact=artifact,input_source=input_source,cached_Bolt_source=native["states"][state]["Bolt"][split]["artifact"],actual_F_calls=calls,canonical_batch_size=32)
            if split in ("fit","tune") and row["status"]=="COMPLETE":result["FIT_TUNE_evaluations_completed"]+=1
            return row
        def run(state,control):
            model=model_for(state,control);q=controls[control];qrecord=io.save_npz(output/(state+"_"+control+"_Q.npz"),{"Q_FP64":q["Q"],"Q_FP32":model.basis().detach().cpu().numpy().copy()})
            row={"status":"RUNNING","Q_artifact":qrecord,"Q_metadata":q["metadata"],"Q_before":model.basis_diagnostics(),"frozen_status":model.frozen_status(),"splits":{},"Q_learned_or_selected":False}
            for split in ("fit","tune"):
                try:row["splits"][split]=evaluate(state,control,split)
                except Exception as error:row["splits"][split]={"status":"FAILED_PATH_PRESERVED","failure":{"type":type(error).__name__,"message":str(error)}}
            row["status"]="COMPLETE" if all(v["status"]=="COMPLETE" for v in row["splits"].values()) else "FAILED_PATH_PRESERVED"
            return row
        def seal(state,control,row):
            path=output/(state+"_"+control+"_CONTROL_LOCK.json");rt.write_new(path,{"status":row["status"],"state":state,"control":control,"Q_artifact":row.get("Q_artifact"),"FIT_TUNE_statuses":{k:v["status"] for k,v in row.get("splits",{}).items()},"DEV_opened_before_lock":False,"Q_selection_performed":False,"sealed_unix":time.time()});return old.record(path)
        def registry(rows):
            path=output/"CONTROL_REGISTRY.json";rt.write_new(path,{"status":"ALL_FIFTEEN_RESOLVED_BEFORE_DEV","conditions":rows,"DEV_opened_before_seal":False,"Q_selection_performed":False,"sealed_unix":time.time()});return old.record(path)
        def development(state,control,row):old.bound(result["control_registry"]);old.bound(row["control_lock"]);old.bound(row["Q_artifact"]);return evaluate(state,control,"val")
        with threadpool_limits(limits=2):resolved=resolve_all(result,run,seal,registry);infer_development(result,resolved,development)
        result["controls_completed"]=len(resolved)
        for key,model in models.items():
            require(all(old.byte_equal(v,old.state_arrays(model)[name]) for name,v in initial[key].items()),"FINAL_ORIGINAL_E_D_CENTER_Q_ALL_BYTES_UNCHANGED");model.frozen_status()
            result["states"][key[0]]["controls"][key[1]]["Q_after"]=model.basis_diagnostics()
        result["F_after"]=probes.model_hashes(bolt);result["model_binary_after"],result["model_config_after"]=probes.verified_file(weight),probes.verified_file(config)
        require(result["F_before"]==result["F_after"] and all(not p.requires_grad and p.grad is None for p in bolt.model.parameters()),"FINAL_F_FROZEN_NO_GRADIENT")
        for item in protected+list(checked.values())+result["source_records"]:old.bound(item)
        for item in result["source_records"]:require(old.record(item["snapshot_path"])["sha256"]==item["sha256"],"EXECUTED_SNAPSHOT_BYTES_UNCHANGED")
        result.update(input_records=list(checked.values()),protected_prior_records=protected,status="COMPLETE" if len(resolved)==15 and result["FIT_TUNE_evaluations_completed"]==30 and result["DEV_evaluations_completed"]==15 and not result["failed_paths"] else "PARTIAL_COMPLETED");meter.protected_check();json.dumps(result,allow_nan=False)
    except Exception as error:failure={"type":type(error).__name__,"message":str(error)};result.update(status="FAILED_PRESERVED",failure=failure)
    finally:result["resource_terminal"]=meter.close(result["status"],failure);rt.write_new(output/"RESULT.json",result)
    print(json.dumps({"status":result["status"],"result":str(output/"RESULT.json")}),flush=True);return 0 if result["status"]=="COMPLETE" else 2


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("tag","calibration-result","calibration-audit","calibration-cost","calibration-cost-audit","report","report-manifest","report-audit"):parser.add_argument("--"+key,required=True)
    raise SystemExit(execute(parser.parse_args()))
