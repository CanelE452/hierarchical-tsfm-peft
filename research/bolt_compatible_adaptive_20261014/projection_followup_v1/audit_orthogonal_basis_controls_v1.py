"""Independent saved fixed-Q arithmetic/score/lineage audit; no F or GPU replay."""
from pathlib import Path
import argparse
import json

import audit_linear_e_adaptation_v1 as base

HERE,ROOT,prior,require=base.HERE,base.ROOT,base.prior,base.require
STATES=("RECON_CP0","E_ONLY_s92601","E_ONLY_s92602")
CONTROLS=("IDENTITY","PERMUTATION","SIGN_FLIP","DENSE_SO_s92711","DENSE_SO_s92712")
LAGS=(0,1,2,4,8,16,32)
RESULT_SHA="67661b1468384aed58d57c8f3c84e3a04d8f6e806cbfb4dbdb92d0f6c34cf42b"
CONTROLLER_SHA="1c0676b75b2baaa1d937bade5e1893d06ba46c01f7c73d48953144e9d09d6daa"
MODULE_SHA="ac515b9cb064e80d89c30a252f7bed4186c81eceb0c1c9d6db1fdbe5cab41491"
PARENT_SHA={"parent_report":"e8f1036dcc27ee2a460b6d87a0f9d1d03368067dab5cb3abd32ec2ec8e368c4f",
    "parent_report_manifest":"e5a17e7f5131ed611b87beca2d49ec4a05e093c2eaf1d0f42d0343b5f0350198",
    "parent_report_audit":"2e98180c7e987e2be807eb00914d9da8876690490daa2b3b56c2f9ce39603dfc",
    "latent_calibration_diagnostic":"88ddbbf7287ad89b7bf81e170b3e05ddfc89bead700e2fefa50bb0dd4e85f576",
    "latent_calibration_audit":"b31b742d3947a205dbaf01cd84239b3cde95e1ea9baf3331a3c08c48a0aa5ba0",
    "latent_calibration_cost":"d8db7445a7a5a750361149cde9a3ad653d4926e871d72c387fe8cbd285aca17b",
    "latent_calibration_cost_audit":"4a54f5b738beef15bc02a17687624afb8c4493f5df7463aeb7b8e4e35b619c40"}
REFS=tuple(PARENT_SHA)+("matched_l7_native_diagnostic","matched_l7_native_audit","E_only_diagnostic","E_only_audit","joint_recon_diagnostic","joint_recon_audit","causal_lag_diagnostic","causal_lag_audit","full_gradient_parent","calibration_report_metadata_reader","controller_source","module_source","plan","control_registry")
np=None


def initialize_numpy():
    global np
    base.initialize_numpy();np=base.np


def same(a,b):require(Path(a["path"]).resolve()==Path(b["path"]).resolve() and a["sha256"]==b["sha256"],"SAME_PATH_SHA_OPTIONAL_BYTES")


def bound(item,check):
    check();require(Path(item["path"]).is_absolute() and prior.file_sha(item["path"])==item["sha256"],"EXACT_ABSOLUTE_SOURCE_FILE_SHA")
    if "bytes" in item:require(Path(item["path"]).stat().st_size==item["bytes"],"EXACT_FILE_SIZE")


def parity(actual,expected):
    require(actual.dtype==expected.dtype==np.float32 and actual.shape==expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all(),"FINITE_MATCHED_FP32_PARITY")
    delta=actual.astype(np.float64)-expected.astype(np.float64)
    return {"max_abs":float(np.abs(delta).max(initial=0)),"rmse":float(np.sqrt(np.mean(delta*delta))),"pass":bool(np.allclose(actual,expected,atol=1e-5,rtol=2e-6)),"atol":1e-5,"rtol":2e-6}


def check_parity(recorded,actual,expected,label):
    computed=parity(actual,expected)
    require(set(recorded)==set(computed) and recorded["pass"]==computed["pass"] and recorded["atol"]==1e-5 and recorded["rtol"]==2e-6,"EXACT_PARITY_FLAGS_AND_REGISTERED_TOLERANCES:"+label)
    prior.close([recorded["max_abs"],recorded["rmse"]],[computed["max_abs"],computed["rmse"]],label,atol=1e-10,rtol=1e-10)
    require(computed["pass"],"REQUIRED_FIXED_Q_PARITY:"+label)


def registered_control(control):
    if control=="IDENTITY":q=np.eye(4);meta={"kind":control,"seed":None,"determinant_FP64":1.,"identity_bypass":True}
    elif control=="PERMUTATION":q=np.eye(4)[:,[1,0,3,2]];meta={"kind":control,"seed":None,"determinant_FP64":1.,"identity_bypass":False}
    elif control=="SIGN_FLIP":q=np.diag([-1.,1.,1.,1.]);meta={"kind":control,"seed":None,"determinant_FP64":-1.,"identity_bypass":False}
    else:
        require(control in CONTROLS,"REGISTERED_DENSE_SEED_ONLY");seed=int(control[-5:]);q,r=np.linalg.qr(np.random.default_rng(seed).normal(size=(4,4)))
        q*=np.where(np.diag(r)<0,-1.,1.)[None,:];determinant=float(np.linalg.det(q));flipped=determinant<0
        if flipped:q[:,-1]*=-1
        meta={"kind":"DENSE_SO","seed":seed,"source":"NumPy default_rng Gaussian 4x4 QR; make diag(R) positive, then flip last Q column iff det<0","canonical_QR_determinant_FP64":determinant,"SO_last_column_flipped":flipped,"determinant_FP64":float(np.linalg.det(q)),"identity_bypass":False}
    return q,meta


def check_Q(arrays,row,control):
    expected,meta=registered_control(control);require(set(arrays)=={"Q_FP64","Q_FP32"},"EXACT_Q_ARTIFACT_MEMBERS")
    base.exact(arrays["Q_FP64"],expected,"registered_Q_FP64");base.exact(arrays["Q_FP32"],expected.astype(np.float32),"registered_Q_cast_FP32");require(row["Q_metadata"]==meta,"CANONICAL_QR_AND_SO_SIGN_PROVENANCE")
    q=arrays["Q_FP32"].astype(np.float64);orthogonal=float(np.abs(q.T@q-np.eye(4)).max())
    prior.close(expected.T@expected,np.eye(4),"FP64_orthogonality",atol=1e-10,rtol=1e-9)
    require(orthogonal<=1e-5,"FP32_REGISTERED_Q_ORTHOGONALITY")
    for key in ("Q_before","Q_after"):
        d=row[key];require(d["finite"] and d["identity_bypass"]==(control=="IDENTITY") and d["theta_L2"] is None and d["old_past_and_decoder_unchanged"] is True and d["input_information_added"] is False,"FIXED_Q_DIAGNOSTIC_SCOPE")
        prior.close([d["orthogonality_max_abs"],d["determinant"]],[orthogonal,float(np.linalg.det(q))],"Q_diagnostics",atol=1e-12,rtol=1e-12)
    require(row["Q_before"]==row["Q_after"] and row["Q_learned_or_selected"] is False,"Q_FIXED_BEFORE_AFTER")
    expected_flags={"E_trainable":False,"D_trainable":False,"center_is_buffer":True,"theta_trainable":False,"trainable_parameters":0,"fixed_identity_arithmetic_bypass":control=="IDENTITY","learned_zero_theta_bypass":False,"F_freezing_checked_by_controller":True,"future_observed_mask_and_loss_owned_by_controller":True}
    require(row["frozen_status"]==expected_flags,"NO_E_D_Q_TRAINABLE_PARAMETERS_RECORDED")
    return {"Q_FP64":arrays["Q_FP64"].tolist(),"Q_FP32":arrays["Q_FP32"].tolist(),"Q_metadata":meta,"orthogonality_FP32_as_FP64_max_abs":orthogonal}


def stats(actual,expected):
    difference=actual.astype(np.float64)-expected.astype(np.float64)
    return {"max_abs":float(np.abs(difference).max(initial=0)),"RMS":float(np.sqrt(np.mean(difference*difference))),
        "by_horizon_max_abs":[float(np.abs(difference[:,h]).max()) for h in range(difference.shape[1])],
        "by_horizon_RMS":[float(np.sqrt(np.mean(difference[:,h]**2))) for h in range(difference.shape[1])]}


def check_cell(row,arrays,independent,raw,cached,frozen,q,control):
    n,c=raw["history"].shape[0],raw["history"].shape[2]
    shapes={"prediction":(n,48,c),"old_past":(n,512,4),"rotated_past":(n,512,4),"F_rotated_future":(n,48,4),"unrotated_future":(n,48,4),"transport_future":(n,48,4),"transport_prediction":(n,48,c)}
    require(set(arrays)==set(shapes)|{"origins"} and set(independent)=={"independent_NumPy_future","independent_NumPy_prediction"},"EXACT_SAVED_FORWARD_AND_INDEPENDENT_MEMBERS")
    for key,shape in shapes.items():require(arrays[key].dtype==np.float32 and arrays[key].shape==shape and np.isfinite(arrays[key]).all(),"FINITE_SAVED_ARRAY:"+key)
    base.exact(arrays["origins"],raw["origins"],"split_origins")
    past=arrays["old_past"];future=arrays["F_rotated_future"] if control=="IDENTITY" else arrays["F_rotated_future"]@q.T
    actual_C=raw["raw_last"][:,None]+base.delta_features(past,future,LAGS)@frozen["D"]
    transported=(cached["latent_future"]@q)@q.T
    transport_C=raw["raw_last"][:,None]+base.delta_features(cached["latent_history"],transported,LAGS)@frozen["D"]
    base.exact(independent["independent_NumPy_future"],future,"saved_NumPy_inverse_future")
    prior.close(independent["independent_NumPy_prediction"],actual_C,"saved_NumPy_old_D_C",atol=1e-5,rtol=2e-6)
    checks={"old_past_vs_cached":(past,cached["latent_history"]),"NumPy_fresh_encode":(past,(raw["history"]-frozen["center"])@frozen["E"]),
        "NumPy_rotated_past":(arrays["rotated_past"],past if control=="IDENTITY" else past@q),"NumPy_inverse_rotated_F_future":(arrays["unrotated_future"],future),
        "NumPy_old_lag_D_actual_C":(arrays["prediction"],actual_C),"NumPy_analytic_future_transport":(arrays["transport_future"],transported),
        "analytic_future_transport_vs_original":(arrays["transport_future"],cached["latent_future"]),"analytic_transport_old_C_vs_baseline":(arrays["transport_prediction"],cached["prediction"])}
    null=control in ("IDENTITY","PERMUTATION")
    if null:checks.update(null_F_future_vs_original=(arrays["unrotated_future"],cached["latent_future"]),null_C_vs_original=(arrays["prediction"],cached["prediction"]))
    require(set(row["checks"])==set(checks),"ALL_REGISTERED_CHECK_FIELDS")
    for key,(actual,expected) in checks.items():check_parity(row["checks"][key],actual,expected,key)
    prior.close(arrays["transport_prediction"],transport_C,"independent_transport_C_old_past_old_D")
    require(row["status"]=="COMPLETE" and row["null_control_required"]==null and row["sign_or_dense_equivariance_required"] is row["future_labels_used_for_forward_or_Q_choice"] is row["performance_improvement_required_for_execution_success"] is False,"EXECUTION_NOT_PERFORMANCE_OR_SIGN_EQUIVARIANCE")
    complete=np.broadcast_to(raw["observed"].all(2,keepdims=True),raw["observed"].shape);score_fields=0
    for name,prediction in (("scores",arrays["prediction"]),("baseline_Bolt_scores",cached["prediction"]),("RAW_BOLT_scores",raw["raw_forecast"]),("transport_scores",arrays["transport_prediction"])):
        for mask_name,mask in (("all_observed",raw["observed"]),("complete_C",complete)):
            if mask.any():score_fields+=prior.compare_scores(row[name][mask_name],prior.scores(prediction,raw["target"],mask,raw["std"]),name+mask_name)
            else:require(row[name][mask_name]=={"status":"NO_COMPLETE_C"},"NO_COMPLETE_MASK_NOT_ZERO_SCORE")
    require(row["complete_target_positions"]==int(complete[:,:,0].sum()) and row["observed_target_values"]==int(raw["observed"].sum()),"EXACT_COMPLETE_AND_OBSERVED_COUNTS")
    summaries={}
    for name,actual,expected in (("latent_commutator",arrays["unrotated_future"],cached["latent_future"]),("C_commutator",arrays["prediction"],cached["prediction"]),("transport_future_gap",arrays["transport_future"],cached["latent_future"]),("transport_C_gap",arrays["transport_prediction"],cached["prediction"])):
        summaries[name]=stats(actual,expected);require(set(row[name])==set(summaries[name]),"COMMUTATOR_FIELDS")
        for key in summaries[name]:prior.close(row[name][key],summaries[name][key],name+key,atol=1e-10,rtol=1e-10)
    require(row["canonical_batch_size"]==32 and row["actual_F_calls"]==(n+31)//32,"REGISTERED_CANONICAL_BATCH_AND_RECORDED_F_CALLS")
    return {"score_fields_checked":score_fields,"null_control":null,"null_C_future_byte_equalities":sum(a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes() for a,b in ((arrays["prediction"],cached["prediction"]),(arrays["unrotated_future"],cached["latent_future"]))) if null else 0,
        "scores":row["scores"],"baseline_Bolt_scores":row["baseline_Bolt_scores"],"RAW_BOLT_scores":row["RAW_BOLT_scores"],**summaries,"complete_target_positions":row["complete_target_positions"],"F_inference_independently_replayed":False}


def check_registry(result,registry,load):
    require(result["all_15_locks_before_DEV"] is True and result["resolved_control_locks"]==15 and registry["status"]=="ALL_FIFTEEN_RESOLVED_BEFORE_DEV" and registry["DEV_opened_before_seal"] is registry["Q_selection_performed"] is False and len(registry["conditions"])==15,"FIFTEEN_RESOLVED_FIXED_CONTROLS_BEFORE_DEV")
    locks={}
    for item in registry["conditions"]:
        key=item["state"],item["control"];require(key in {(s,q) for s in STATES for q in CONTROLS} and key not in locks and item["status"]=="COMPLETE","FIFTEEN_UNIQUE_COMPLETE_CONTROLS")
        row=result["states"][key[0]]["controls"][key[1]];same(item["control_lock"],row["control_lock"]);lock=load(item["control_lock"])
        require(lock["status"]=="COMPLETE" and (lock["state"],lock["control"])==key and lock["DEV_opened_before_lock"] is lock["Q_selection_performed"] is False and lock["FIT_TUNE_statuses"]=={"fit":"COMPLETE","tune":"COMPLETE"} and lock["sealed_unix"]<=registry["sealed_unix"],"EXACT_FIXED_Q_NO_SELECTION_LOCK")
        same(lock["Q_artifact"],row["Q_artifact"]);locks[key]=lock
    return locks


def audit_result(path,meter):
    require(isinstance(RESULT_SHA,str) and len(RESULT_SHA)==64,"ACTUAL_FIXED_Q_RESULT_NOT_REGISTERED")
    check=meter.check;binding=base.file_record(path);require(binding["sha256"]==RESULT_SHA,"EXACT_ACTUAL_FIXED_Q_RESULT")
    result=prior.verified_json(binding,check)
    require(result["status"]=="COMPLETE" and not result["failed_paths"] and set(result["states"])==set(STATES),"COMPLETE_FIXED_CONTROLS_REQUIRED_FOR_FULL_PASS")
    for key,digest in PARENT_SHA.items():require(result[key]["sha256"]==digest,"EXACT_ACCEPTED_PARENT:"+key)
    require(result["controller_source"]["sha256"]==CONTROLLER_SHA and result["module_source"]["sha256"]==MODULE_SHA,"EXACT_EXECUTED_FIXED_Q_SOURCE")
    for key in ("new_optimizer_updates","new_Q_learning_updates","new_E_D_training_updates"):require(result[key]==0,"NO_Q_E_D_OPTIMIZER_UPDATES")
    for key in ("new_backbone_training","Q_selection_performed","protected_TEST_accessed","independent_holdout","strict_K_only","whole_deployment_cost_measured","control_resolution_is_selection","performance_success_claimed_from_parity"):require(result[key] is False,"FIXED_Q_SCOPE:"+key)
    require(result["all_fixed_controls_reported"] is True and result["controls_completed"]==15 and result["FIT_TUNE_evaluations_completed"]==30 and result["DEV_evaluations_completed"]==15 and result["new_F_forward_calls"]==result["new_F_forward_attempts"]==165 and result["new_F_model_objects_loaded"]==1,"EXACT_FIXED_Q_EXECUTION_COUNTS")
    guarded=[binding,*[result[k] for k in REFS],*result["source_records"],*result["input_records"],*result["protected_prior_records"]]
    for item in guarded:
        bound(item,check)
        if "snapshot_path" in item:require(prior.file_sha(item["snapshot_path"])==item["sha256"],"EXECUTED_SOURCE_SNAPSHOT_BYTES")
    parent=prior.verified_json(result["latent_calibration_diagnostic"],check);native=prior.verified_json(result["matched_l7_native_diagnostic"],check);full=prior.verified_json(result["full_gradient_parent"],check)
    for key in ("matched_l7_native_diagnostic","matched_l7_native_audit","E_only_diagnostic","E_only_audit","joint_recon_diagnostic","joint_recon_audit","causal_lag_diagnostic","causal_lag_audit"):same(result[key],parent[key])
    require(result["F_before"]==result["F_after"]==parent["F_parent_signatures_reused"]["before"]==parent["F_parent_signatures_reused"]["after"] and result["runtime_behavior_contract"]==full["runtime_behavior_contract"],"F_DUAL_SIGNATURE_AND_RUNTIME_UNCHANGED")
    for key in ("model_binary","model_config"):
        same(result[key+"_before"],result[key+"_after"]);bound(result[key+"_before"],check);bound(result[key+"_after"],check)
    registry=prior.verified_json(result["control_registry"],check);locks=check_registry(result,registry,lambda item:prior.verified_json(item,check))
    receipt=result["resource_terminal"];saved_receipt=prior.verified_json(receipt,check)
    require(saved_receipt=={k:v for k,v in receipt.items() if k not in ("path","sha256")} and receipt["status"]=="COMPLETE" and receipt["gpu_process"] is True and receipt["error"] is None and receipt["started_unix"]<=registry["sealed_unix"]<=receipt["ended_unix"] and receipt["process_cpu_seconds"]>=0 and receipt["wall_seconds"]>=0,"EXACT_COMPLETE_PROCESS_RECEIPT")
    bound(receipt["authorization"],check);bound(receipt["continuation_manifest"],check)
    cache={};produced_paths=set();output=Path(binding["path"]).parent
    def load(item):
        key=item["path"],item["sha256"]
        if key not in cache:cache[key]=prior.verified_arrays(item,check)
        return cache[key]
    def produced(item,name):
        path=Path(item["path"]).resolve();require(path==output/name and path not in produced_paths,"UNIQUE_NEW_ARTIFACT_AT_REGISTERED_PRODUCER_PATH")
        produced_paths.add(path);return load(item)
    totals={"fixed_control_conditions_checked":0,"FIT_TUNE_prediction_evaluations_checked":0,"DEV_prediction_evaluations_checked":0,"control_locks_checked":len(locks),"produced_NPZ_artifacts_checked":0,"actual_F_calls_registered":0,"null_control_split_cells_checked":0,"null_C_future_byte_equalities_observed":0,"analytic_transports_checked":0,"independent_forward_arithmetic_cells_checked":0,"score_fields_checked":0,"commutator_summaries_checked":0,"frozen_E_D_center_sources_checked":0,"protected_prior_file_hashes_checked":len(result["protected_prior_records"])}
    report={"status":"PASS","diagnostic":binding,**{k:result[k] for k in REFS},"totals":totals,"states":{},"partial_paths":[],"protected_TEST_accessed":False,"independent_holdout":False,"independent_F_GPU_training_or_timer_replayed":False,"DEV_access_events_independently_replayed":False,"scope":"Independent saved NumPy E/Q/inverse/causal-lag-D/analytic transport, all observed and complete-C metrics, fixed controls/locks and source/receipt preservation. Before-DEV gate uses sealed records and immutable source control flow, not filemtime or independent access-event replay. F output itself is stored producer evidence, not an independent F replay. Parity is not performance/generalization."}
    for state in STATES:
        source=native["states"][state]["frozen_state_source"];same(result["states"][state]["frozen_state_source"],source);same(source,parent["states"][state]["frozen_state_source"]);frozen=load(source)
        require(set(frozen)=={"E","D","center"} and frozen["E"].shape==(17,4) and frozen["D"].shape==(28,17) and frozen["center"].shape==(17,) and all(v.dtype==np.float32 and np.isfinite(v).all() for v in frozen.values()),"FINITE_ORIGINAL_FROZEN_STATE")
        totals["frozen_E_D_center_sources_checked"]+=1;report["states"][state]={"frozen_state_source":source,"controls":{}}
        owner=result["states"][state];require(set(owner["controls"])==set(CONTROLS),"ALL_FIVE_CONTROLS_PRESERVED")
        for control in CONTROLS:
            row=owner["controls"][control];require(row["status"]=="COMPLETE" and set(row["splits"])=={"fit","tune","val"},"ALL_THREE_CONTROL_SPLITS_COMPLETE")
            Q=produced(row["Q_artifact"],state+"_"+control+"_Q.npz");verified=check_Q(Q,row,control);verified["splits"]={};totals["produced_NPZ_artifacts_checked"]+=1;totals["fixed_control_conditions_checked"]+=1
            for split,n in (("fit",256),("tune",64),("val",30)):
                entry=row["splits"][split];p=parent["states"][state]["families"]["SCALE_ONLY"][{"fit":"FIT","tune":"TUNE","val":"DEV"}[split]]
                same(entry["input_source"],p["input_source"]);same(entry["cached_Bolt_source"],p["cached_Bolt_source"]);same(entry["cached_Bolt_source"],native["states"][state]["Bolt"][split]["artifact"])
                raw,cached=load(entry["input_source"]),load(entry["cached_Bolt_source"]);prior.check_raw_split(raw,channels=17,windows=n)
                for key in ("target","observed","raw_last","origins","mean","std"):base.exact(cached[key],raw[key],"accepted_Bolt_raw_context_"+key)
                arrays=produced(entry["artifact"],state+"_"+control+"_"+split+".npz");independent=produced(entry["independent_artifact"],state+"_"+control+"_"+split+"_INDEPENDENT.npz");v=check_cell(entry,arrays,independent,raw,cached,frozen,Q["Q_FP32"],control)
                verified["splits"][split]=v;totals["score_fields_checked"]+=v["score_fields_checked"];totals["null_control_split_cells_checked"]+=int(v["null_control"]);totals["null_C_future_byte_equalities_observed"]+=v["null_C_future_byte_equalities"]
                for key in ("analytic_transports_checked","independent_forward_arithmetic_cells_checked"):totals[key]+=1
                totals["produced_NPZ_artifacts_checked"]+=2;totals["actual_F_calls_registered"]+=entry["actual_F_calls"];totals["commutator_summaries_checked"]+=4;totals["DEV_prediction_evaluations_checked" if split=="val" else "FIT_TUNE_prediction_evaluations_checked"]+=1
            report["states"][state]["controls"][control]=verified
    require(totals["fixed_control_conditions_checked"]==15 and totals["produced_NPZ_artifacts_checked"]==len(produced_paths)==105 and totals["actual_F_calls_registered"]==165 and totals["FIT_TUNE_prediction_evaluations_checked"]==30 and totals["DEV_prediction_evaluations_checked"]==15,"COMPLETE_ACTUAL_FIXED_Q_AUDIT_COVERAGE")
    for item in guarded:bound(item,check)
    return report


def execute(path,output):
    from continuation_runtime_v1 import Meter,write_new
    output=Path(output).resolve();require(not output.exists(),"FRESH_FIXED_Q_AUDIT_OUTPUT");meter=Meter("audit_orthogonal_basis_controls_"+output.stem,gpu=False);failure=None;source=base.file_record(__file__)
    try:
        meter.protected_check();initialize_numpy()
        import sys
        sys.path.insert(0,str(ROOT/"research/smallk_failure_followup_20261014"))
        from prepare_inputs import install_protected_test_guard
        from threadpoolctl import threadpool_limits
        install_protected_test_guard()
        with threadpool_limits(limits=2):report=audit_result(path,meter)
        meter.protected_check();require(prior.file_sha(__file__)==source["sha256"],"FIXED_Q_AUDITOR_SOURCE_UNCHANGED");json.dumps(report,allow_nan=False)
    except Exception as error:failure={"type":type(error).__name__,"message":str(error)};report={"status":"FAILED_PRESERVED","failure":failure}
    finally:report["audit_source"]=source;report["resource_terminal"]=meter.close(report["status"],failure);write_new(output,report)
    print(json.dumps({"status":report["status"],"totals":report.get("totals"),"failure":failure}),flush=True);return 0 if report["status"]=="PASS" else 2


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--result",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args();raise SystemExit(execute(args.result,args.output))
