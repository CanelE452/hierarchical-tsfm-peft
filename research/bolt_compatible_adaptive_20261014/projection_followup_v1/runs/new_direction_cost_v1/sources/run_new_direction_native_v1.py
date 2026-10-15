"""Sixteen head-only native fits on the eight audited frozen PCA/cross-moment E/Ds."""
from pathlib import Path
import argparse
import json
import math
import shutil
import sys
import time

import run_predictive_cross_moment_v1 as cross
import run_matched_l7_native_v1 as matched

old, HERE, ROOT, BASE, OLD, require = cross.old, cross.HERE, cross.ROOT, cross.BASE, cross.OLD, cross.require
DATASETS, LAGS, SEEDS = cross.DATASETS, cross.LAGS, (92601, 92602)
STATES = tuple((data,kind+"_K"+str(k)) for data,kind,k in cross.CONDITIONS)
CONDITIONS = tuple((data,name,seed) for data,name in STATES for seed in SEEDS)
CROSS_RESULT_SHA = "99bffd5831143ca35947099fef9567f89dd209d83451661decee0682b2b8e13c"
CROSS_AUDIT_SHA = "90c68a9057909fb5b6c23284e5a45523cdf126da178a5ba0d9d9e96a50dede30"
CROSS_SOURCE_SHA = "e5b971277673118454d30f13819a7013c9c0b0ef1c6e682c18d5fab971ef1f50"
CROSS_AUDITOR_SHA = "6577d15a200728dc585bf2c8e93ccff6baabd1ba916384dfb2cce7aa0014a731"
REUSED_SOURCES = {"run_matched_l7_native_v1.py":"8eafb31307ade30f66d32d18a2c6d62da18a156b0c307cede8ee4bfe72567d43",
    "lagged_native_head_v1.py":"cfd2438be26dd7cec0bfd53552ef17f545cb7da0f7716d4185ee557ab249a534",
    "native_head_v1.py":"ad128f6713021dea58191609d3e6b1075c2cdfafe5d1b930d94d91c4eac436c7",
    "linear_e_adaptation_v1.py":"17c8ac7d20b9f3dd759a3b3fb2e019e027469e9c8a057eb1176da98b9c319931",
    "run_linear_e_adaptation_v1.py":"e268528d77d17b0c8f5a300ff8d667ce5ddd25b1f61754c3895df87929fcd1f3"}
CROSS_AUDIT_COUNTS = {"states_checked":8,"direction_fits_checked":8,"decoder_fits_checked":8,"decoder_response_columns_checked":112,"FIT_TUNE_cells_checked":16,"DEV_cells_checked":8,"independent_saved_F_arithmetic_cells_checked":24,"oracle_dependency_masks_checked":24,"state_locks_checked":8,"unique_dataset_split_timestamp_contexts_checked":6}
RECIPE = {"optimizer":"AdamW","learning_rate":.001,"weight_decay":0.,"batch_size":4,"clip_norm":1.,"updates_per_condition":600,"checkpoint_updates":[0,100,200,400,600],"head":"stream-specific affine512_to_48_from_history_minus_last_plus_last","head_zero_initialized":True}


def load_gates(args):
    require(all(isinstance(v,str) and len(v) == 64 for v in (CROSS_RESULT_SHA,CROSS_AUDIT_SHA,CROSS_SOURCE_SHA,CROSS_AUDITOR_SHA)),"NEW_DIRECTION_CROSS_AUDIT_NOT_YET_REGISTERED")
    old.bound({"path":cross.__file__,"sha256":CROSS_SOURCE_SHA})
    for name,digest in REUSED_SOURCES.items(): old.bound({"path":str(HERE/name),"sha256":digest})
    calibration,previous_native,science,full,refs = cross.load_gates(args)
    binding,audit_binding = old.record(args.cross_result),old.record(args.cross_audit)
    require(binding["sha256"] == CROSS_RESULT_SHA and audit_binding["sha256"] == CROSS_AUDIT_SHA,"EXACT_ACCEPTED_NEW_DIRECTION_RESULT_AND_AUDIT")
    result,audit = old.read(binding["path"]),old.read(audit_binding["path"])
    require(result["status"] == "COMPLETE" and audit["status"] == "PASS" and result["protected_TEST_accessed"] is audit["protected_TEST_accessed"] is False and not result["failed_paths"] and not audit["partial_paths"],"COMPLETE_INDEPENDENTLY_AUDITED_EIGHT_STATES")
    old.bound(audit["diagnostic"],binding); old.bound(result["controller_source"],{"path":cross.__file__,"sha256":CROSS_SOURCE_SHA}); old.bound(audit["audit_source"])
    require(audit["audit_source"]["sha256"] == CROSS_AUDITOR_SHA and all(type(audit["totals"].get(k)) is int and audit["totals"][k] == v for k,v in CROSS_AUDIT_COUNTS.items()),"EXACT_CROSS_AUDITOR_SOURCE_AND_COVERAGE")
    for key,item in refs.items(): old.bound(result[key],item)
    old.bound(result["module_source"],{"path":str(HERE/"predictive_cross_moment_v1.py"),"sha256":cross.MODULE_SHA}); old.bound(result["state_registry"],audit["state_registry"])
    require(result["direction_fits_completed"] == result["decoder_fits_completed"] == result["resolved_state_locks"] == 8 and result["FIT_TUNE_evaluations_completed"] == 16 and result["DEV_evaluations_completed"] == 8 and result["all_8_locks_before_DEV"] is True and result["new_F_forward_calls"] == result["new_F_forward_attempts"] == 104 and result["new_F_model_objects_loaded"] == 1,"ACCEPTED_CROSS_EIGHT_FROZEN_STATES_104_ACTUAL_F_CALLS")
    require(result["F_before"] == result["F_after"] == calibration["F_parent_signatures_reused"]["before"] == calibration["F_parent_signatures_reused"]["after"] and result["runtime_behavior_contract"] == full["runtime_behavior_contract"],"SAME_ACCEPTED_F_AND_MEDIAN_RUNTIME")
    require(set(result["datasets"]) == set(DATASETS),"EXACT_TWO_PARENT_DATASETS")
    for data in DATASETS: require(set(result["datasets"][data]["conditions"]) == {name for d,name in STATES if d == data},"EXACT_FOUR_REGISTERED_STATES_PER_DATASET")
    for data,name in STATES:
        row = result["datasets"][data]["conditions"][name]; c,basek,dev,_ = DATASETS[data]; kind,k = name.rsplit("_K",1)
        require(row["status"] == "COMPLETE" and row["kind"] == kind and row["K"] == int(k) and row["C"] == c and row["E_FIT_future_supervised"] == (kind == "CROSS_MOMENT") and row["D_FIT_future_supervised"] is False and set(row["splits"]) == {"fit","tune",dev},"SAME_FROZEN_STATE_AND_SUPERVISION_HISTORY")
        old.bound(row["state_artifact"],audit["datasets"][data]["conditions"][name]["state_artifact"])
        for split in ("fit","tune",dev): require(row["splits"][split]["status"] == "COMPLETE","ALL_PARENT_F_CACHES_ACCEPTED")
    refs.update(predictive_cross_diagnostic=binding,predictive_cross_audit=audit_binding,predictive_cross_auditor_source=audit["audit_source"],predictive_cross_state_registry=result["state_registry"])
    return result,calibration,previous_native,science,full,refs


def evaluate_arrays(outputs, raw, frozen, cached):
    import numpy as np
    from causal_lag_decoder_v1 import predict,lagged_delta_features
    from decoder_metric_probe_v1 import masked_scores
    n,length,c = raw["history"].shape; h = raw["target"].shape[1]; k = frozen["E"].shape[1]
    shapes = {"prediction":(n,h,c),"latent_history":(n,length,k),"latent_future":(n,h,k),"lag_features":(n,h,len(LAGS)*k)}
    require(set(outputs) == set(shapes) and all(outputs[key].dtype == np.float32 and outputs[key].shape == shape and np.isfinite(outputs[key]).all() for key,shape in shapes.items()),"FINITE_NATIVE_ACTUAL_FP32_COMPONENTS")
    history_same = old.byte_equal(outputs["latent_history"],cached["latent_history"])
    prediction = predict(outputs["latent_history"],outputs["latent_future"],raw["raw_last"],frozen["D"],LAGS); features = lagged_delta_features(outputs["latent_history"],outputs["latent_future"],LAGS)
    checks = {"cached_Torch_latent_history_byte_exact":history_same,"independent_NumPy_old_lag_C":old.parity(outputs["prediction"],prediction),"independent_NumPy_lag_features":old.parity(outputs["lag_features"],features)}
    complete = np.broadcast_to(raw["observed"].all(2,keepdims=True),raw["observed"].shape)
    def scores(value): return {"all_observed":masked_scores(value,raw["target"],raw["observed"],raw["std"]),"complete_C":masked_scores(value,raw["target"],complete,raw["std"]) if complete.any() else {"status":"NO_COMPLETE_C"}}
    actual = scores(outputs["prediction"])
    row = {"status":"COMPLETE" if history_same and all(checks[key]["pass"] for key in checks if key != "cached_Torch_latent_history_byte_exact") else "FAILED_PARITY_PRESERVED","checks":checks,"scores_all_observed":actual["all_observed"],"scores_complete_mask":actual["complete_C"],"baseline_Bolt_scores":scores(cached["prediction"]),"RAW_BOLT_scores":scores(raw["raw_forecast"]),"complete_target_positions":int(complete[:,:,0].sum()),"canonical_batch_size":32,"future_truth_is_forward_input":False,"new_F_forward":False,"same_D_and_current_C_anchor":True,"NumPy_fresh_E_parity":old.parity(outputs["latent_history"],(raw["history"]-frozen["center"])@frozen["E"]),"NumPy_fresh_E_parity_is_primary_gate":False}
    return row,{"independent_NumPy_prediction":prediction,"independent_NumPy_lag_features":features}


def check_score_link(score, entry):
    require(all(math.isclose(score[key],entry["scores_all_observed"]["normalized"][key],abs_tol=1e-12,rel_tol=1e-12) for key in ("mse","mae")),"OWN_CANONICAL_TUNE_SCORE_LINK")


def resolve_all(result, run, seal, registry):
    resolved,locks = {},[]
    for data,name,seed in CONDITIONS:
        owner = result["datasets"][data]["conditions"][name]["native"]
        try: row = run(data,name,seed)
        except Exception as error:
            row = owner.get(str(seed),{}); row.update(status="FAILED_PATH_PRESERVED",failure={"type":type(error).__name__,"message":str(error)})
        row["selection_lock"] = seal(data,name,seed,row); owner[str(seed)] = row
        locks.append({"dataset":data,"condition":name,"seed":seed,"status":row["status"],"selection_lock":row["selection_lock"]})
        result["training_updates_completed"] += row.get("training_updates_completed",0); result["checkpoint_states_saved"] += row.get("checkpoint_count",0)
        if row["status"] == "COMPLETE": resolved[data,name,seed] = row; result["native_training_fits"] += 1
        else: result["failed_paths"].append({"dataset":data,"condition":name,"seed":seed,"stage":"FIT_TUNE","failure":row.get("failure"),"training_updates_completed":row.get("training_updates_completed",0)})
    result["selection_registry"] = registry(locks); result.update(resolved_selection_locks=len(locks),all_16_locks_before_DEV=True)
    return resolved


def infer_development(result, resolved, infer):
    require(result["all_16_locks_before_DEV"] and result["resolved_selection_locks"] == 16,"ALL_SIXTEEN_LOCKS_BEFORE_ANY_DEV")
    for (data,name,seed),row in resolved.items():
        try:
            row["DEV"] = infer(data,name,seed,row); require(row["DEV"]["status"] == "COMPLETE","SELECTED_DEV_INDEPENDENT_PARITY"); result["DEV_evaluations_completed"] += 1
        except Exception as error: result["failed_paths"].append({"dataset":data,"condition":name,"seed":seed,"stage":"DEV","failure":{"type":type(error).__name__,"message":str(error)}})


def execute(args):
    parent,calibration,previous_native,science,full,refs = load_gates(args); output = old.fresh_output(args.tag); sys.path[:0] = [str(HERE),str(BASE),str(OLD),str(ROOT)]
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    import fixed_decoder_span_v1 as io
    import continuation_runtime_v1 as rt
    import lagged_native_head_v1 as native
    from native_head_v1 import state_hash
    from run_fit_target_comparison_v1 import prior_file_records
    from prepare_inputs import install_protected_test_guard
    output.mkdir(parents=True,exist_ok=False); meter = rt.Meter("new_direction_native_"+args.tag,gpu=True); failure = None
    result = {"status":"RUNNING",**refs,"datasets":{d:{"conditions":{name:{"native":{},"Bolt":{}} for data,name in STATES if data == d}} for d in DATASETS},"failed_paths":[],"source_records":[],"native_training_fits":0,"training_updates_completed":0,"checkpoint_states_saved":0,"resolved_selection_locks":0,"all_16_locks_before_DEV":False,"DEV_evaluations_completed":0,"cached_actual_F_contexts_reused":0,"cached_actual_F_forward_calls_registered_in_parent":104,"new_F_forward_calls":0,"new_F_model_objects_loaded":0,"new_F_state_hash_computed":False,"new_backbone_training":False,"new_E_training_updates":0,"new_D_training_updates":0,"whole_deployment_cost_measured":False,"new_output_space_computed":False,"protected_TEST_accessed":False,"independent_holdout":False,"strict_K_only":False,"training_recipe":RECIPE,"selection_rule":"own canonical32 normalized TUNE macro MSE; exact ties earlier CP","seed_scope":"same zero head; minibatch order only; two seeds are not independent datasets","gradient_clipping_parameter_scope":"HEAD_ONLY","F_parent_signatures_reused":{"before":parent["F_before"],"after":parent["F_after"]},"runtime_behavior_contract_reused":parent["runtime_behavior_contract"],"native_family_scope":"stream-specific affine temporal head; no claim about all native predictors, channel-mixing heads or optimization exhaustion"}
    checked,contexts,states,caches,models,selected_states = {},{},{},{},{},{}; protected = []
    try:
        install_protected_test_guard(); meter.protected_check(); protected = prior_file_records(parent,calibration,previous_native,science,full,refs)
        folder = output/"sources"; folder.mkdir()
        for path in (Path(__file__),HERE/"PLAN_NEW_DIRECTION_NATIVE_BASELINES_KO_v1.md",Path(cross.__file__),HERE/"audit_predictive_cross_moment_v1.py",*[HERE/name for name in REUSED_SOURCES],HERE/"causal_lag_decoder_v1.py",HERE/"decoder_metric_probe_v1.py",HERE/"fixed_decoder_span_v1.py",HERE/"continuation_runtime_v1.py",OLD/"prepare_inputs.py"):
            destination = folder/path.name; shutil.copyfile(path,destination); result["source_records"].append({**old.record(path),"snapshot_path":str(destination)})
        result.update(controller_source=old.record(__file__),native_source=old.record(native.__file__),plan=old.record(HERE/"PLAN_NEW_DIRECTION_NATIVE_BASELINES_KO_v1.md"),parent_controller_source=parent["controller_source"],reused_matched_native_source=old.record(matched.__file__))
        torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
        def load(item): return io.checked_payload({**item,"path":str(old.bound(item))},meter,checked)
        def context(data,split):
            require(split in ("fit","tune") or result["all_16_locks_before_DEV"],"SIXTEEN_LOCKS_BEFORE_DEV_ARRAY_IO")
            key = data,split
            if key not in contexts:
                c,_,dev,devn = DATASETS[data]; require(split in ("fit","tune",dev),"REGISTERED_DATASET_SPLIT_ONLY"); n = {"fit":256,"tune":64}.get(split,devn)
                source = science["datasets"][data]["spaces"]["PCA"]["conditions"]["L7_RECON"]["splits"][split]["input_source"]; raw = load(source)
                require(raw["history"].dtype == raw["target"].dtype == raw["raw_forecast"].dtype == np.float32 and raw["history"].shape == (n,512,c) and raw["target"].shape == raw["observed"].shape == raw["raw_forecast"].shape == (n,48,c) and raw["observed"].dtype == raw["history_observed"].dtype == np.bool_ and raw["history_observed"].shape == raw["history"].shape and np.isfinite(raw["history"]).all() and np.isfinite(raw["target"][raw["observed"]]).all() and np.isfinite(raw["raw_forecast"]).all() and old.byte_equal(raw["raw_last"],raw["history"][:,-1]),"EXACT_ORIGINAL_COHORT_ACTUAL_Y_CURRENT_C")
                require(raw["origins"].dtype == np.int64 and raw["origins"].shape == (n,) and raw["mean"].dtype == raw["std"].dtype == np.float32 and raw["mean"].shape == raw["std"].shape == (c,) and np.isfinite(raw["mean"]).all() and np.isfinite(raw["std"]).all() and (raw["std"] > 0).all(),"ORIGINAL_ORIGINS_AND_FIT_ONLY_SCALER")
                contexts[key] = raw,source
            return contexts[key]
        def frozen(data,name):
            key = data,name
            if key not in states:
                accepted = parent["datasets"][data]["conditions"][name]; value = load(accepted["state_artifact"]); c = DATASETS[data][0]; k = accepted["K"]
                require(set(value) == {"E","D","center"} and value["E"].shape == (c,k) and value["D"].shape == (7*k,c) and value["center"].shape == (c,) and all(v.dtype == np.float32 and np.isfinite(v).all() for v in value.values()),"EXACT_FROZEN_NEW_DIRECTION_E_D_CENTER")
                result["datasets"][data]["conditions"][name].update(kind=accepted["kind"],K=k,C=c,frozen_state_source=accepted["state_artifact"],frozen_state_value_sha256=state_hash({key:torch.as_tensor(v) for key,v in value.items()}),E_FIT_future_supervised=accepted["E_FIT_future_supervised"],D_FIT_future_supervised=False)
                states[key] = value
            return states[key]
        def cache(data,name,split):
            key = data,name,split
            if key not in caches:
                raw,source = context(data,split); accepted = parent["datasets"][data]["conditions"][name]; entry = accepted["splits"][split]; old.bound(entry["input_source"],source); old.bound(entry["state_source"],accepted["state_artifact"]); value = load(entry["artifact"]); k = accepted["K"]; n = len(raw["history"])
                require(old.byte_equal(value["origins"],raw["origins"]) and value["latent_history"].shape == (n,512,k) and value["latent_future"].shape == (n,48,k) and value["prediction"].shape == raw["target"].shape and all(value[v].dtype == np.float32 and np.isfinite(value[v]).all() for v in ("latent_history","latent_future","prediction")),"SAME_FROZEN_STATE_AND_ORIGINAL_CONTEXT_ACTUAL_F_CACHE")
                caches[key] = value; result["cached_actual_F_contexts_reused"] += 1
                result["datasets"][data]["conditions"][name]["Bolt"][split] = {"status":"COMPLETE","artifact":entry["artifact"],"input_source":source,"state_source":accepted["state_artifact"],"methods":entry["methods"],"new_F_forward":False,"original_actual_F_cache_reused":True,"parent_actual_F_forward_calls_not_reexecuted":True}
            return caches[key]
        def save_evaluation(label, values, data, name, split):
            raw,source = context(data,split); cached = cache(data,name,split)
            artifact = io.save_npz(output/(label+".npz"),{**values,**{key:raw[key] for key in ("target","observed","raw_last","origins","mean","std")}})
            try:
                entry,independent = evaluate_arrays(values,raw,frozen(data,name),cached); entry["independent_artifact"] = io.save_npz(output/(label+"_INDEPENDENT.npz"),independent)
            except Exception as error: entry = {"status":"FAILED_DIAGNOSTIC_PRESERVED","failure":{"type":type(error).__name__,"message":str(error)}}
            entry.update(artifact=artifact,input_source=source,cached_Bolt_source=parent["datasets"][data]["conditions"][name]["splits"][split]["artifact"],state_source=parent["datasets"][data]["conditions"][name]["state_artifact"])
            return entry
        def run(data,name,seed):
            label = data+"_"+name+"_NATIVE_s"+str(seed); row = {"status":"FAILED_PATH_PRESERVED","dataset":data,"condition":name,"seed":seed,"training_updates_completed":0,"checkpoint_count":0}; result["datasets"][data]["conditions"][name]["native"][str(seed)] = row; checkpoints = []; path = output/(label+"_UPDATES.jsonl")
            try:
                fit,_ = context(data,"fit"); tune,_ = context(data,"tune"); cached = cache(data,name,"tune"); source = frozen(data,name)
                model = native.LaggedNativeForecaster(source["center"],source["E"],source["D"],LAGS).to("cuda").eval(); native.frozen_status(model)
                require(all(old.byte_equal(old.state_arrays(model)[key],value) for key,value in source.items()),"NATIVE_INITIAL_EXACT_FROZEN_STATE_BYTES")
                row.update(frozen_state_source=parent["datasets"][data]["conditions"][name]["state_artifact"],initial_full_state_sha256=state_hash(model.state_dict()),initial_head_state_sha256=state_hash(model.head.state_dict()),initial_frozen_status=native.frozen_status(model),head_parameters=sum(p.numel() for p in model.head.parameters()))
                values = old.predict_components(model,None,tune["history"],meter.check); probe,arrays = matched.cp0_probe(values,cached,source,tune)
                probe.update(artifact=io.save_npz(output/(label+"_CP0_PARITY.npz"),{**arrays,"origins":tune["origins"]}),actual_Bolt_component_source=parent["datasets"][data]["conditions"][name]["splits"]["tune"]["artifact"]); row["probe"] = probe; require(probe["status"] == "PASS","NATIVE_CP0_PARITY_ARRAYS_PRESERVED")
                def checkpoint(update,tensors,score,values):
                    digest = state_hash(tensors); require(all(old.byte_equal(tensors[key].numpy(),value) for key,value in source.items()),"EVERY_CP_FROZEN_E_D_CENTER_BYTES")
                    item = {"update":update,"tune":score,"state":io.save_npz(output/(label+"_CP"+str(update)+"_STATE.npz"),{key:value.numpy().copy() for key,value in tensors.items()}),"frozen_status":native.frozen_status(model),"head_state_sha256":state_hash(model.head.state_dict()),"E_D_center_exact_initial_bytes":True,"canonical_score_link_atol":1e-12,"canonical_score_link_rtol":1e-12,**save_evaluation(label+"_CP"+str(update)+"_TUNE",values,data,name,"tune")}
                    checkpoints.append(item); row["checkpoint_count"] += 1
                    require(state_hash(tensors) == digest and item["status"] == "COMPLETE","IMMUTABLE_CP_STATE_AND_VALID_NUMPY_OLD_D_DEPLOYMENT"); check_score_link(score,item); return item
                with path.open("x",encoding="utf-8") as log:
                    def check(event=None):
                        meter.check(event)
                        if event and event.get("stage") == "OPTIMIZER_STEP_EXECUTED": row["training_updates_completed"] = event["update"]
                        if event and event.get("stage") == "TRAINING_UPDATE": log.write(json.dumps(event,allow_nan=False)+"\n"); log.flush(); row["training_updates_completed"] = event["update"]
                    trained = native.train_head(model,fit,tune,seed,check,checkpoint)
                row.update({key:value for key,value in trained.items() if key != "model"}); require(row["order_sha256"] == previous_native["states"]["RECON_CP0"]["native"][str(seed)]["order_sha256"],"SAME_PARENT_256_WINDOW_600_BY_FOUR_ORDER"); selected = next(v for v in checkpoints if v["update"] == trained["selected_update"])
                values = old.predict_components(model,None,fit["history"],meter.check); fitted = save_evaluation(label+"_SELECTED_FIT",values,data,name,"fit")
                row.update(status="COMPLETE" if fitted["status"] == "COMPLETE" else "FAILED_PATH_PRESERVED",checkpoints=checkpoints,CP0_TUNE=checkpoints[0],updates_log=old.record(path),selected_state=selected["state"],selected_head_state_sha256=state_hash(model.head.state_dict()),FIT=fitted,TUNE={key:value for key,value in selected.items() if key not in ("state","frozen_status","head_state_sha256","E_D_center_exact_initial_bytes")})
                models[data,name,seed] = model; selected_states[data,name,seed] = old.state_arrays(model)
            except Exception as error:
                row.update(status="FAILED_PATH_PRESERVED",failure={"type":type(error).__name__,"message":str(error)},checkpoints=checkpoints)
                if path.exists(): row["updates_log"] = old.record(path)
            return row
        def seal(data,name,seed,row):
            path = output/(data+"_"+name+"_s"+str(seed)+"_SELECTION_LOCK.json"); rt.write_new(path,{**row,"DEV_opened_before_lock":False,"sealed_unix":time.time()}); return old.record(path)
        def registry(rows):
            path = output/"SELECTION_REGISTRY.json"; rt.write_new(path,{"status":"ALL_SIXTEEN_RESOLVED_BEFORE_DEV","conditions":rows,"DEV_opened_before_seal":False,"sealed_unix":time.time()}); return old.record(path)
        def development(data,name,seed,row):
            old.bound(row["selection_lock"]); old.bound(result["selection_registry"]); old.bound(row["selected_state"]); raw,_ = context(data,DATASETS[data][2]); values = old.predict_components(models[data,name,seed],None,raw["history"],meter.check)
            return save_evaluation(data+"_"+name+"_NATIVE_s"+str(seed)+"_SELECTED_DEV",values,data,name,DATASETS[data][2])
        with threadpool_limits(limits=2):
            resolved = resolve_all(result,run,seal,registry)
            for data,name in STATES:
                for split in ("fit","tune",DATASETS[data][2]):
                    try: cache(data,name,split)
                    except Exception as error: result["failed_paths"].append({"dataset":data,"condition":name,"split":split,"stage":"ACTUAL_F_CACHE_REUSE","failure":{"type":type(error).__name__,"message":str(error)}})
            infer_development(result,resolved,development)
        for key,model in models.items():
            native.frozen_status(model); require(all(old.byte_equal(value,old.state_arrays(model)[name]) for name,value in selected_states[key].items()),"SELECTED_FULL_STATE_IMMUTABLE_DURING_FINAL_INFERENCE")
            require(all(old.byte_equal(value,old.state_arrays(model)[name]) for name,value in states[key[:2]].items()),"FINAL_FROZEN_E_D_CENTER_BYTES")
        result["checkpoint_artifact_hashes_rechecked"] = 0
        for data,name,seed in CONDITIONS:
            row = result["datasets"][data]["conditions"][name]["native"][str(seed)]; old.bound(row["selection_lock"])
            for cp in row.get("checkpoints",[]):
                for field in ("state","artifact","independent_artifact"):
                    if field in cp: old.bound(cp[field]); result["checkpoint_artifact_hashes_rechecked"] += 1
            for key in ("FIT","DEV","probe"):
                for field in ("artifact","independent_artifact"):
                    if field in row.get(key,{}): old.bound(row[key][field])
        old.bound(result["selection_registry"])
        for item in protected+list(checked.values())+result["source_records"]: old.bound(item)
        for item in result["source_records"]: require(old.record(item["snapshot_path"])["sha256"] == item["sha256"],"EXECUTED_SOURCE_SNAPSHOT_UNCHANGED")
        result.update(input_records=list(checked.values()),protected_prior_records=protected,new_optimizer_updates=result["training_updates_completed"],new_native_head_updates=result["training_updates_completed"],status="COMPLETE" if result["native_training_fits"] == 16 and result["training_updates_completed"] == 9600 and result["checkpoint_states_saved"] == 80 and result["DEV_evaluations_completed"] == 16 and result["cached_actual_F_contexts_reused"] == 24 and not result["failed_paths"] else "PARTIAL_COMPLETED"); meter.protected_check(); json.dumps(result,allow_nan=False)
    except Exception as error: failure = {"type":type(error).__name__,"message":str(error)}; result.update(status="FAILED_PRESERVED",failure=failure)
    finally: result["resource_terminal"] = meter.close(result["status"],failure); rt.write_new(output/"RESULT.json",result)
    print(json.dumps({"status":result["status"],"result":str(output/"RESULT.json")}),flush=True); return 0 if result["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("tag","cross-result","cross-audit","calibration-result","calibration-audit","calibration-cost","calibration-cost-audit","report","report-manifest","report-audit"): parser.add_argument("--"+key,required=True)
    raise SystemExit(execute(parser.parse_args()))
