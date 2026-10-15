"""FIT-only latent calibration from cached F; actual CUDA q/decode, no F/head/model loading."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import time

import run_linear_e_adaptation_v1 as old

HERE,ROOT,BASE,OLD,require=old.HERE,old.ROOT,old.BASE,old.OLD,old.require
STATES=("RECON_CP0","E_ONLY_s92601","E_ONLY_s92602")
FAMILIES=("SCALE_ONLY","SCALE_BIAS")
CONDITIONS=tuple((state,family) for state in STATES for family in FAMILIES)
LAGS=(0,1,2,4,8,16,32)
REPORT_SHA="2adaf2ac2104b658e36ec10060e58604ce5b699a17f95d4b6ea1b2ee35b70c62"
MANIFEST_SHA="e4ec0a98b24019b37bdd414b9569c322825a3df805bdd084e5d1d843b47e272d"
REPORT_AUDIT_SHA="91da4c088a2328229526f90c83e3b73ad8ba065504036dfb6b32065dc8b51e24"
READER_SHA="b401543919ada3d1633df67d909a7ad4152bae5567d479ea6b440ee30033bc20"
MODULE_SHA="088b9800724b07ced0c42cfa2f514077e430d2c3a7f291dea6f45d1f444cb523"


def load_gates(args):
    require(all(isinstance(v,str) and len(v)==64 for v in (REPORT_SHA,MANIFEST_SHA,REPORT_AUDIT_SHA,MODULE_SHA)),"ACTUAL_NATIVE_BUNDLE_AND_CALIBRATION_MODULE_NOT_YET_REGISTERED")
    paths={"parent_report":args.report,"parent_report_manifest":args.report_manifest,"parent_report_audit":args.report_audit}; refs={k:old.record(p) for k,p in paths.items()}
    require([refs[k]["sha256"] for k in paths]==[REPORT_SHA,MANIFEST_SHA,REPORT_AUDIT_SHA],"EXACT_ACCEPTED_NATIVE_REPORT_BUNDLE_BEFORE_SCIENTIFIC_IMPORT")
    audit,manifest=old.read(args.report_audit),old.read(args.report_manifest)
    require(audit["status"]=="PASS" and manifest["status"]=="COMPLETE" and audit["files_checked"]==22 and audit["PNG_count"]==audit["SVG_count"]==4 and audit["local_links_checked"]==41 and audit["protected_TEST_accessed"] is manifest["protected_TEST_accessed"] is False,"FULL_NATIVE_REPORT_ACCEPTANCE")
    old.bound(audit["report"],refs["parent_report"]); old.bound(audit["manifest"],refs["parent_report_manifest"])
    require(manifest["source"]["sha256"]==READER_SHA,"FROZEN_NATIVE_REPORT_METADATA_READER")
    old.bound(manifest["source"]); old.bound({"path":str(HERE/"fixed_l7_latent_calibration_v1.py"),"sha256":MODULE_SHA})
    for item in manifest["files"]:
        path=(old.resolve(args.report_manifest).parent/item["path"]).resolve(); require(path.is_relative_to(old.resolve(args.report_manifest).parent),"PARENT_REPORT_FILE_CONTAINMENT"); old.bound({**item,"path":str(path)})
    from make_matched_l7_native_report_v2 import load_verified
    paths=dict(zip(("train","audit","cost","cost_audit"),(args.native_result,args.native_audit,args.native_cost,args.native_cost_audit)))
    values,bindings=load_verified(paths)
    for key in paths: old.bound(audit["inputs"][key],bindings[key]); refs["matched_l7_native_"+{"train":"diagnostic","audit":"audit","cost":"cost","cost_audit":"cost_audit"}[key]]=bindings[key]
    native=values["train"]
    refs.update(native_selection_registry=native["selection_registry"],native_bundle_metadata_reader=manifest["source"],parent_report_auditor_source=audit["audit_source"])
    for key in ("E_only_diagnostic","E_only_audit","joint_recon_diagnostic","joint_recon_audit","causal_lag_diagnostic","causal_lag_audit"): refs[key]=native[key]
    return native,values["science"],refs


def resolve_families(result,run,seal,registry):
    locks=[]; resolved={}
    for state,family in CONDITIONS:
        try: row=run(state,family)
        except Exception as error: row={"status":"FAILED_PATH_PRESERVED","failure":{"type":type(error).__name__,"message":str(error)}}
        lock=seal(state,family,row); row["selection_lock"]=lock; result["states"][state]["families"][family]=row; locks.append({"state":state,"family":family,"status":row["status"],"selection_lock":lock})
        if row["status"]=="COMPLETE": resolved[state,family]=row
        else: result["failed_paths"].append({"state":state,"family":family,"stage":"FAMILY_FIT_AND_SELECTION","failure":row.get("failure")})
    result["selection_registry"]=registry(locks); result.update(resolved_selection_locks=len(locks),all_6_locks_before_DEV=True); return resolved


def infer_development(result,resolved,infer):
    require(result["all_6_locks_before_DEV"] and result["resolved_selection_locks"]==6,"ALL_SIX_LOCKS_BEFORE_ANY_DEV")
    for (state,family),row in resolved.items():
        try:
            row["DEV"]=infer(state,family,row); require(row["DEV"]["status"]=="COMPLETE","SELECTED_DEV_PARITY_MUST_PASS"); result["selected_DEV_evaluations_completed"]+=1
        except Exception as error:
            row["DEV_failure"]={"type":type(error).__name__,"message":str(error)}; result["failed_paths"].append({"state":state,"family":family,"stage":"SELECTED_DEV",**row["DEV_failure"]})


def coefficient_arrays(candidate):
    return {key:value for key,value in candidate.items() if key in ("a_FP64","b_FP64","a_FP32","b_FP32","coefficients_FP64","coefficients_FP32") and value is not None}


def value_hash(arrays):
    digest=hashlib.sha256()
    for key,value in sorted(arrays.items()):
        for part in (key.encode(),str(value.dtype).encode(),str(value.shape).encode(),value.tobytes()): digest.update(part)
    return digest.hexdigest()


def predict_candidate(model,raw,cached,candidate,module,check,*,batch_size=32):
    import numpy as np
    import torch
    predictions=[]; futures=[]; identity_checks=0
    with torch.no_grad():
        for start in range(0,len(raw["raw_last"]),batch_size):
            current,past,future,baseline=(torch.as_tensor(v[start:start+batch_size],dtype=torch.float32,device=model.E.device) for v in (raw["raw_last"],cached["latent_history"],cached["latent_future"],cached["prediction"]))
            prediction,adjusted=module.predict_calibrated(model,current,past,future,candidate,baseline)
            if candidate["kind"]=="IDENTITY": require(prediction is baseline and adjusted is future,"IDENTITY_RETURNS_ORIGINAL_BASELINE_AND_FUTURE_OBJECTS"); identity_checks+=1
            predictions.append(prediction.cpu().numpy().copy()); futures.append(adjusted.cpu().numpy().copy()); check({"stage":"CALIBRATION_CANONICAL_CUDA_PREDICTION","start":start,"candidate":candidate["id"],"new_F_forward":False})
    prediction,future=(np.ascontiguousarray(np.concatenate(values),dtype=np.float32) for values in (predictions,futures)); require(np.isfinite(prediction).all() and np.isfinite(future).all(),"FINITE_ACTUAL_FP32_CALIBRATED_OUTPUTS")
    if candidate["kind"]=="IDENTITY": require(old.byte_equal(prediction,cached["prediction"]) and old.byte_equal(future,cached["latent_future"]),"IDENTITY_BASELINE_AND_FUTURE_BYTES")
    return prediction,future,{"identity_object_checks":identity_checks,"identity_byte_exact":candidate["kind"]=="IDENTITY","batch_size":batch_size}


def independent_deployment(prediction,future,cached,raw,decoder,candidate):
    from causal_lag_decoder_v1 import predict
    original=cached["latent_future"]
    expected=original if candidate["kind"]=="IDENTITY" else original+candidate["a_FP32"]*(original-cached["latent_history"][:,-1:])+candidate["b_FP32"]
    expected_C=predict(cached["latent_history"],expected,raw["raw_last"],decoder,LAGS)
    checks={"latent_future":old.parity(future,expected),"C_prediction":old.parity(prediction,expected_C)}
    return checks,{"independent_NumPy_future_FP32":expected,"independent_NumPy_prediction_FP32":expected_C}


def execute(args):
    native,science,refs=load_gates(args); output=old.fresh_output(args.tag); sys.path[:0]=[str(HERE),str(BASE),str(OLD),str(ROOT)]
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    import fixed_l7_latent_calibration_v1 as module
    import fixed_decoder_span_v1 as io
    import continuation_runtime_v1 as rt
    from linear_e_adaptation_v1 import LinearLagAdapter
    from decoder_metric_probe_v1 import masked_scores
    from run_fit_target_comparison_v1 import prior_file_records
    from prepare_inputs import install_protected_test_guard
    require(old.record(module.__file__)["sha256"]==MODULE_SHA,"EXACT_CALIBRATION_PURE_SOURCE")
    output.mkdir(parents=True,exist_ok=False); meter=rt.Meter("fixed_L7_latent_calibration_"+args.tag,gpu=True); failure=None
    result={"status":"RUNNING",**refs,"states":{state:{"families":{}} for state in STATES},"failed_paths":[],"failed_candidate_paths":[],"source_records":[],"fit_conditions_completed":0,"SVD_fits_completed":0,"ridge_candidates_generated":0,"identity_comparisons_generated":0,"candidate_FIT_TUNE_evaluations_completed":0,"selected_DEV_evaluations_completed":0,"resolved_selection_locks":0,"all_6_locks_before_DEV":False,"cached_Bolt_contexts_reused":0,"cached_baseline_redecode_checks":0,"new_F_forward_calls":0,"new_F_model_objects_loaded":0,"native_head_weights_loaded":False,"new_optimizer_updates":0,"new_E_D_training_updates":0,"new_backbone_training":False,"new_F_state_hash_computed":False,"F_parent_signatures_reused":native["F_parent_signatures_reused"],"whole_deployment_cost_measured":False,"independent_holdout":False,"strict_K_only":False,"protected_TEST_accessed":False,"resource_scope":"GPU process receipt includes CPU FP64 design/SVD/ridge and actual CUDA FP32 calibration/current fixed lag decode; no F or head load/forward","selected_FIT_TUNE_predictions_reused_without_new_inference":True}
    checked,contexts,caches,models,initial_states,objects={},{},{},{},{},{}; protected=[]
    try:
        install_protected_test_guard(); meter.protected_check(); protected=prior_file_records(refs)
        folder=output/"sources"; folder.mkdir()
        for path in (Path(__file__),HERE/"fixed_l7_latent_calibration_v1.py",HERE/"PLAN_FIXED_L7_LATENT_CALIBRATION_KO_v1.md",HERE/"make_matched_l7_native_report_v2.py",HERE/"make_linear_e_adaptation_report_v2.py",HERE/"run_linear_e_adaptation_v1.py",HERE/"linear_e_adaptation_v1.py",HERE/"causal_lag_decoder_v1.py",HERE/"fixed_decoder_span_v1.py",HERE/"decoder_metric_probe_v1.py",HERE/"continuation_runtime_v1.py",OLD/"prepare_inputs.py"):
            destination=folder/path.name; shutil.copyfile(path,destination); result["source_records"].append({**old.record(path),"snapshot_path":str(destination)})
        result.update(controller_source=old.record(__file__),module_source=old.record(module.__file__),plan=old.record(HERE/"PLAN_FIXED_L7_LATENT_CALIBRATION_KO_v1.md"))
        torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
        recon=science["datasets"]["robin"]["spaces"]["PCA"]["conditions"]["L7_RECON"]
        def load(item): return io.checked_payload({**item,"path":str(old.bound(item))},meter,checked)
        def context(split):
            require(split in ("fit","tune") or result["all_6_locks_before_DEV"],"SIX_FAMILY_LOCKS_BEFORE_DEV_ARRAY_IO")
            if split not in contexts:
                raw=load(recon["splits"][split]["input_source"]); n={"fit":256,"tune":64,"val":30}[split]
                require(raw["history"].dtype==raw["target"].dtype==np.float32 and raw["history"].shape==(n,512,17) and raw["target"].shape==raw["observed"].shape==(n,48,17) and raw["observed"].dtype==np.bool_ and np.isfinite(raw["history"]).all() and np.isfinite(raw["target"][raw["observed"]]).all() and old.byte_equal(raw["raw_last"],raw["history"][:,-1]),"SAME_OBSERVED_ROBIN_CAUSAL_INPUT_AND_ACTUAL_Y"); contexts[split]=raw
            return contexts[split]
        def model_for(state):
            if state not in models:
                source=native["states"][state]["frozen_state_source"]; values=load(source); require(set(values)=={"E","D","center"} and values["E"].shape==(17,4) and values["D"].shape==(28,17) and values["center"].shape==(17,) and all(v.dtype==np.float32 and np.isfinite(v).all() for v in values.values()),"ORIGINAL_THREE_FROZEN_E_D_STATES_ONLY_NO_HEAD_WEIGHTS")
                model=LinearLagAdapter(values["center"],values["E"],values["D"],LAGS,learn_E=False).to("cuda").eval().requires_grad_(False)
                require(all(old.byte_equal(v.detach().cpu().numpy(),values[k]) for k,v in model.state_dict().items()) and all(not p.requires_grad and p.grad is None for p in model.parameters()),"EXACT_SOURCE_FROZEN_E_D_CENTER_NO_GRADIENT")
                models[state]=model; initial_states[state]=old.state_arrays(model); result["states"][state].update(frozen_state_source=source,E_previously_supervised_by_FIT_future=state!="RECON_CP0",D_previously_supervised_by_FIT_future=False)
            return models[state]
        def cache(state,split):
            key=state,split
            if key not in caches:
                raw=context(split); source=native["states"][state]["Bolt"][split]["artifact"]; values=load(source); model=model_for(state)
                for name in ("target","observed","raw_last","origins","mean","std"): require(old.byte_equal(raw[name],values[name]),"EXACT_ACCEPTED_BOLT_CONTEXT:"+name)
                require(values["latent_history"].dtype==values["latent_future"].dtype==values["prediction"].dtype==np.float32 and values["latent_history"].shape==(len(raw["raw_last"]),512,4) and values["latent_future"].shape==(len(raw["raw_last"]),48,4) and values["prediction"].shape==raw["target"].shape and all(np.isfinite(values[k]).all() for k in ("latent_history","latent_future","prediction")),"FINITE_MATCHED_BOLT_CACHE")
                outputs=[]
                with torch.no_grad():
                    for start in range(0,len(raw["raw_last"]),32):
                        current,past,future=(torch.as_tensor(v[start:start+32],dtype=torch.float32,device=model.E.device) for v in (raw["raw_last"],values["latent_history"],values["latent_future"])); outputs.append(model.decode(current,past,future).cpu().numpy().copy()); meter.check({"stage":"CACHED_BASELINE_REDECODE","state":state,"split":split,"start":start})
                require(old.byte_equal(np.ascontiguousarray(np.concatenate(outputs)),values["prediction"]),"CANONICAL_CURRENT_D_BASELINE_REDECODE_BYTES")
                caches[key]=values; result["cached_Bolt_contexts_reused"]+=1; result["cached_baseline_redecode_checks"]+=1
            return caches[key]
        def evaluate(state,family,candidate,split,label):
            raw=context(split); cached=cache(state,split); model=model_for(state); before=value_hash(coefficient_arrays(candidate)); prediction,future,identity=predict_candidate(model,raw,cached,candidate,module,meter.check)
            parity,independent_arrays=independent_deployment(prediction,future,cached,raw,initial_states[state]["D"],candidate)
            phi=module.design_columns(cached["latent_history"],cached["latent_future"],initial_states[state]["D"],LAGS,family); surrogate=cached["prediction"].astype(np.float64)+phi@candidate["coefficients_FP64"]; diagnostic=module.surrogate_diagnostics(prediction,cached["prediction"],phi,candidate["coefficients_FP64"])
            require(value_hash(coefficient_arrays(candidate))==before,"CALIBRATION_COEFFICIENT_VALUES_UNCHANGED_BY_DEPLOYMENT"); complete=np.broadcast_to(raw["observed"].all(2,keepdims=True),raw["observed"].shape)
            arrays={"prediction_FP32":prediction,"latent_future_calibrated_FP32":future,"surrogate_prediction_FP64":surrogate,"actual_minus_surrogate_FP64":prediction.astype(np.float64)-surrogate,"origins":raw["origins"],**independent_arrays}
            row={"status":"COMPLETE" if all(p["pass"] for p in parity.values()) else "FAILED_DEPLOYMENT_PARITY_PRESERVED","artifact":io.save_npz(output/(label+"_"+split+".npz"),arrays),"scores_all_observed":masked_scores(prediction,raw["target"],raw["observed"],raw["std"]),"scores_complete_mask":masked_scores(prediction,raw["target"],complete,raw["std"]) if complete.any() else {"status":"NO_COMPLETE_C"},"surrogate_scores_all_observed":masked_scores(surrogate,raw["target"],raw["observed"],raw["std"]),"surrogate_diagnostics":diagnostic,"independent_FP32_deployment_parity":parity,"FP64_surrogate_is_not_deployment_parity_target":True,"identity_checks":identity,"coefficient_value_sha256":before,"input_source":recon["splits"][split]["input_source"],"cached_Bolt_source":native["states"][state]["Bolt"][split]["artifact"],"target_future_is_loss_or_evaluation_only":True}
            return row
        def run_complete(state,family,row):
            fit,tune=context("fit"),context("tune"); cached=cache(state,"fit"); model_for(state)
            fitted=module.fit_calibration(cached["latent_history"],cached["latent_future"],initial_states[state]["D"],cached["prediction"],fit["target"],fit["observed"],LAGS,family)
            result["fit_conditions_completed"]+=1; result["SVD_fits_completed"]+=fitted["metadata"]["SVD_fits"]; result["ridge_candidates_generated"]+=len(fitted["candidates"]); result["identity_comparisons_generated"]+=1
            row.update(fit_metadata=fitted["metadata"],fit_artifact=io.save_npz(output/(state+"_"+family+"_FIT_SOLVER.npz"),{k:fitted[k] for k in ("column_scales_FP64","active_columns","singularvalues_FP64")}),FIT_input_source=recon["splits"]["fit"]["input_source"],FIT_cached_Bolt_source=native["states"][state]["Bolt"]["fit"]["artifact"],frozen_state_source=native["states"][state]["frozen_state_source"])
            for candidate in (fitted["identity"],*fitted["candidates"]):
                label=state+"_"+family+"_"+candidate["id"]; item={k:candidate[k] for k in ("id","kind","family","lambda","status","metadata")}; item["coefficient_artifact"]=io.save_npz(output/(label+"_COEFFICIENTS.npz"),coefficient_arrays(candidate)); item["coefficient_value_sha256"]=value_hash(coefficient_arrays(candidate)); row["candidates"].append(item); objects[state,family,candidate["id"]]=candidate
                if candidate["status"]!="COMPLETE": result["failed_candidate_paths"].append({"state":state,"family":family,"candidate":candidate["id"],"status":candidate["status"],"coefficient_artifact":item["coefficient_artifact"]}); continue
                try:
                    for split in ("fit","tune"):
                        item[split.upper()]=evaluate(state,family,candidate,split,label); require(item[split.upper()]["status"]=="COMPLETE","ACTUAL_FP32_NUMPY_DEPLOYMENT_PARITY_MUST_PASS"); result["candidate_FIT_TUNE_evaluations_completed"]+=1
                    item["TUNE_MSE"]=item["TUNE"]["scores_all_observed"]["normalized"]["mse"]
                except Exception as error:
                    item.update(status="FAILED_CANDIDATE_PRESERVED",failure={"type":type(error).__name__,"message":str(error)}); result["failed_candidate_paths"].append({"state":state,"family":family,"candidate":candidate["id"],**item["failure"]})
            selected=module.select_candidate(row["candidates"]); row.update(status="COMPLETE",selected_candidate=selected["id"],selected_kind=selected["kind"],selected_lambda=selected["lambda"],selected_TUNE_MSE=selected["TUNE_MSE"],selected_coefficients=selected["coefficient_artifact"],selected_coefficient_value_sha256=selected["coefficient_value_sha256"],FIT=selected["FIT"],TUNE=selected["TUNE"],selection_rule="Actual FP32 own-TUNE observed macro MSE, exact tie identity then larger lambda")
            return row
        def run(state,family):
            row={"status":"FAILED_PATH_PRESERVED","state":state,"family":family,"candidates":[],"new_F_forward":False}
            try: return run_complete(state,family,row)
            except Exception as error: row.update(status="FAILED_PATH_PRESERVED",failure={"type":type(error).__name__,"message":str(error)}); return row
        def seal(state,family,row):
            path=output/(state+"_"+family+"_SELECTION_LOCK.json"); json.dumps(row,allow_nan=False); rt.write_new(path,{**row,"DEV_opened_before_lock":False,"sealed_unix":time.time()}); return old.record(path)
        def registry(rows):
            path=output/"SELECTION_REGISTRY.json"; rt.write_new(path,{"status":"ALL_SIX_RESOLVED_BEFORE_DEV","conditions":rows,"DEV_opened_before_seal":False,"sealed_unix":time.time()}); return old.record(path)
        def infer(state,family,row):
            old.bound(row["selection_lock"]); old.bound(result["selection_registry"]); old.bound(row["selected_coefficients"]); candidate=objects[state,family,row["selected_candidate"]]; require(value_hash(coefficient_arrays(candidate))==row["selected_coefficient_value_sha256"],"SELECTED_FIT_COEFFICIENTS_BEFORE_DEV")
            return evaluate(state,family,candidate,"val",state+"_"+family+"_SELECTED")
        with threadpool_limits(limits=2): resolved=resolve_families(result,run,seal,registry); infer_development(result,resolved,infer)
        result["selected_conditions_completed"]=len(resolved)
        for state,model in models.items():
            require(all(old.byte_equal(value.detach().cpu().numpy(),initial_states[state][key]) for key,value in model.state_dict().items()) and all(not p.requires_grad and p.grad is None for p in model.parameters()),"FINAL_E_D_CENTER_BYTES_FROZEN_NO_GRADIENT")
        for item in protected+result["source_records"]+list(checked.values()): old.bound(item)
        for item in result["source_records"]: require(old.record(item["snapshot_path"])["sha256"]==item["sha256"],"EXECUTED_SOURCES_UNCHANGED")
        result.update(input_records=list(checked.values()),protected_prior_records=protected,status="COMPLETE" if len(resolved)==6 and result["fit_conditions_completed"]==6 and result["ridge_candidates_generated"]==42 and result["identity_comparisons_generated"]==6 and result["candidate_FIT_TUNE_evaluations_completed"]==96 and result["selected_DEV_evaluations_completed"]==6 and not result["failed_paths"] and not result["failed_candidate_paths"] else "PARTIAL_COMPLETED"); meter.protected_check(); json.dumps(result,allow_nan=False)
    except Exception as error: failure={"type":type(error).__name__,"message":str(error)}; result.update(status="FAILED_PRESERVED",failure=failure)
    finally: result["resource_terminal"]=meter.close(result["status"],failure); rt.write_new(output/"RESULT.json",result)
    print(json.dumps({"status":result["status"],"result":str(output/"RESULT.json")}),flush=True); return 0 if result["status"]=="COMPLETE" else 2


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("tag","native-result","native-audit","native-cost","native-cost-audit","report","report-manifest","report-audit"): parser.add_argument("--"+key,required=True)
    raise SystemExit(execute(parser.parse_args()))
