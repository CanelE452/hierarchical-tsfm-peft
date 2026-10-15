"""Independent saved 76-cell TUNE cost audit; no F/GPU/fit/timer replay."""
from pathlib import Path
import argparse
import ast
import json
import statistics

import audit_matched_l7_native_cost_v1 as previous
import audit_new_direction_direct_ridge_v1 as ridge_audit

prior,linear,shared=previous.prior,previous.linear,previous.shared
HERE,ROOT=previous.HERE,previous.ROOT
require,exact,parity,bound,same=previous.require,previous.exact,previous.parity,previous.bound,previous.same_record
DATASETS={"robin":(17,4),"beijing_aotizhongxin":(11,2)}
LAGS=(0,1,2,4,8,16,32)
COST_SHA="8a1876aeb1490a53a4872c294a9ec2e1a4f296a1a32ff8787b084b0b98d6a1fb"
RUNNER_SHA="4c721b0b87d9252fa55e90b68e03a35e9b18e15f61dfc9d1b14e4289058f5bfa"
HELPER_SHA="49a094ddd8b40d79932ef6c0dc87ce37548abd8159c0f2b1cae8cb0b617a1ae9"
PARENT_TIMER_SHA="bd59a61ba2c7a14f48c2cf7fe3ed9fa5f366e98eca61019e75672137493ba9e4"
PINS={"new_direction_native_diagnostic":"786edbd75a35d2c5a760a7b4f5af5a1389ee1bc4bc93594d7b359a29ecfd83ed","new_direction_native_audit":"51e438d7544bb5c52d2a6ea3ddd2da9c38773b2aa42286cf3a77c67d405d7bf6","new_direction_ridge_diagnostic":"b8d7eae62ce5bb52b3d46ffc543f916f8b81857028098370a5b2f8398510f1ed","new_direction_ridge_audit":"8bfeb29cc6836d2ff5528c30ee13a4300b79cc4c250a452b9c5c81fdfc9bac90","predictive_cross_diagnostic":"99bffd5831143ca35947099fef9567f89dd209d83451661decee0682b2b8e13c","predictive_cross_audit":"90c68a9057909fb5b6c23284e5a45523cdf126da178a5ba0d9d9e96a50dede30"}
MEMORY_SCOPE="Shared frozen F, all available24 adapters/16 heads and12 CPU ridge coefficient states; no exclusive allocation or best-backend claim"
np=None


def initialize_numpy():
    global np
    previous.initialize_numpy();ridge_audit.initialize_numpy();np=previous.np


def state_names(data):
    k=DATASETS[data][1];return tuple(kind+"_K"+str(width) for kind in ("PCA","CROSS_MOMENT") for width in (k,2*k))


def routes(data):
    names=state_names(data)
    return ("RAW_BOLT",)+tuple(route for name in names for route in (name+"_BOLT",name+"_NATIVE_s92601",name+"_NATIVE_s92602"))+("RAW_CURRENT_C_RIDGE","RAW_HISTORY512_RIDGE")+tuple(name+"_RIDGE_HISTORY512" for name in names)


def matched_bolt(route):
    if route.startswith("RAW_"):return "RAW_BOLT"
    for marker in ("_NATIVE_","_RIDGE_"):
        if marker in route:return route.split(marker)[0]+"_BOLT"
    return route


def schedule(data):
    names=routes(data);order=np.random.default_rng(20261015).permutation(19)
    return [[names[index] for index in np.roll(order,-repeat)] for repeat in range(5)]


def source_contract(helper,runner,parent):
    def functions(text):return {n.name:n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
    h,r,p=map(functions,(helper,runner,parent))
    def timed(function):
        candidates=[n for n in ast.walk(function) if isinstance(n,ast.Try) and len(n.body)==3 and "perf_end_ns, unix_end_ns = (time.perf_counter_ns(), time.time_ns())" in ast.unparse(n)]
        require(len(candidates)==1,"ONE_CANONICAL_TIMED_TRY");return [ast.dump(n) for n in candidates[0].body]
    require(timed(h["measure_cost"])==timed(p["measure_cost"]),"UNCHANGED_WHOLE_PREDICT_AND_FINAL_SYNC_NS_BOUNDARY")
    imports=[n for n in ast.parse(helper).body if isinstance(n,ast.ImportFrom) and n.module=="whole_f_input_gradient_cost_v1"]
    require(len(imports)==1 and {"preprocess","whole_predict","fill_vectorized","parity","NORM_ATOL","NORM_RTOL","PHYSICAL_ATOL","PHYSICAL_RTOL"}<={n.name for n in imports[0].names},"CANONICAL_PREPROCESS_AND_FROZEN_TOLERANCES_REUSED")
    require("ORDER_SEED, WARMUP, REPEATS = (20261015, 2, 5)" in ast.unparse(ast.parse(helper)),"FIVE_OF_NINETEEN_ROTATIONS_TWO_WARMUPS")
    execution=ast.unparse(r["execute"]);ridge=ast.unparse(r["ridge_components"])
    require("canonical_tune_bridge(original, data)" in execution and "adapter_predictor(model, predictor, engine.tensor)" in execution and "predictor = None" in execution and "physical_precheck(physical, raw, reference, model, predictor, engine.tensor)" in execution,"ACTUAL_F_OR_SELECTED_NATIVE_AND_ORIGINAL_PHYSICAL_COHORT")
    require("encode_torch(encoder, history)" in ridge and "source.feature_payload" in ridge and "source.predict(feature, state, anchor)" in ridge and "meter.check" not in ridge,"ENCODED_CUDA_E_HOST_TRANSFER_AND_CPU_RIDGE_INSIDE_CALLBACK")
    require("np.isfinite" in ast.unparse(p["whole_predict"]) and "frozen_status(" not in execution,"TIMED_INTERNAL_GUARDS_AND_NO_TRAIN_ONLY_GUARD")
    calls=[ast.unparse(n.func) for n in ast.walk(ast.parse(runner)) if isinstance(n,ast.Call)]
    require(not any(name.endswith(("fit_path","train_head","backward","optimizer.step")) for name in calls),"NO_TRAINING_OR_REFIT_IN_COST_SOURCE")
    return {"warmup_each_route":2,"timed_repeats":5,"five_of_nineteen_positions_only":True,"complete_nineteen_position_balance_claimed":False,"actual_F_or_head_or_fused_ridge_no_cached_return":True,"encoded_ridge_CUDA_E_host_transfer_inside_timer":True,"ridge_D_bypassed":True,"internal_shape_finite_guards_inside_timer":True,"external_meter_file_hash_and_memory_outside_timer":True,"untimed_references_and_prechecks_in_addition_to_two_warmups":True,"F_or_timer_replayed":False}


def check_dataset(cost,physical,raw,predictions,check,seal):
    data=cost["dataset"];names=routes(data);ordered=schedule(data)
    require(cost["routes"]==list(names) and cost["schedule"]==ordered and cost["warmup_each_route"]==2 and cost["repeats"]==5 and cost["order_seed"]==20261015 and cost["expected_cells_this_dataset"]==38 and cost["expected_datasets"]==2 and cost["expected_timer_samples_this_dataset"]==190 and cost["expected_total_cells"]==76 and cost["expected_timer_samples"]==380,"REGISTERED_TWO_DATASET_19_ROUTE_COST")
    lineage=cost["source_lineage"];require(lineage["whole_cost_source_sha256"]==HELPER_SHA and lineage["copied_timer_source_sha256"]==PARENT_TIMER_SHA and Path(lineage["copied_timer_source"]).resolve()==HERE/"whole_f_input_gradient_cost_v1.py" and lineage["copied_preprocess_source_sha256"]==prior.file_sha(HERE.parent/"cost_readout_v1.py") and lineage["imported_functions"]==["fill_vectorized","preprocess","parity","whole_predict"] and lineage["legacy_scientific_runtime_imported"] is False,"IMMUTABLE_TIMER_PREPROCESS_SOURCE_LINEAGE")
    require(cost["new_training_updates"]==0 and cost["encoder_refit_in_cost"] is cost["per_origin_decoder_fit"] is cost["fit_input_binding_hashes_inside_timer"] is cost["protected_TEST_accessed"] is cost["independent_holdout"] is False and cost["fixed_FIT_decoder_loaded_before_timer"] is True,"FIXED_SELECTED_STATES_NO_FITTING")
    exact(prior.causal_physical_preprocess(physical["history_physical"],physical["history_observed"],raw["mean"],raw["std"]),raw["history"],"all_TUNE_physical_causal_fill_FIT_normalization")
    for key,value in (("physical_past",physical["history_physical"]),("history_observed",physical["history_observed"]),("FIT_mean",raw["mean"]),("FIT_std",raw["std"]),("origins",raw["origins"])):require(cost["input_array_metadata"][key]==shared.array_record(value),"EXACT_INPUT_ARRAY_SHA:"+key)
    rows={(r["route"],r["batch"]):r for r in cost["rows"]};require(len(cost["rows"])==38 and set(rows)=={(name,b) for name in names for b in (1,8)},"ALL_38_UNIQUE_ROWS")
    details,chronology=[],[]
    for (route,batch),row in rows.items():
        require(row["status"]=="COMPLETE" and row["failure"] is None and row["repeats"]==len(row["samples"])==5 and row["warmup_each_route"]==2,"FIVE_SAVED_TIMERS_COMPLETE_ROW")
        times=[(s["perf_end_ns"]-s["perf_start_ns"])/1e9 for s in row["samples"]];require(row["seconds"]==times and all(t>0 for t in times) and [s["repeat"] for s in row["samples"]]==list(range(5)),"POSITIVE_INTEGER_NS_DURATIONS_AND_REPEATS")
        require(row["selected_metadata"]==cost["source_metadata"][route] and row["origins"]==raw["origins"][:batch].tolist() and row["native_head_loaded_or_timed"]==("_NATIVE_" in route) and row["ridge_loaded_or_timed"]==row["direct_C_readout_bypasses_D"]==("RIDGE" in route),"EXACT_SELECTED_ROUTE_AND_DIRECT_C_SCOPE")
        require(all(row[k] is False for k in ("declared_route_deleted","per_origin_decoder_fit","stage_decomposition","memory_counter_reset_inside_module")),"NO_DROPPED_PATH_OR_HIDDEN_COST_SCOPE")
        median=statistics.median(times);require(row["median_seconds"]==median and row["median_seconds_per_window"]==median/batch,"INDEPENDENT_NS_MEDIANS")
        ratios={}
        for label,denominator in (("RAW","RAW_BOLT"),("matched_Bolt",matched_bolt(route))):
            base={s["repeat"]:(s["perf_end_ns"]-s["perf_start_ns"])/1e9 for s in rows[denominator,batch]["samples"]};paired_seconds=[base[s["repeat"]] for s in row["samples"]];paired=[t/b for t,b in zip(times,paired_seconds)];ratio=median/statistics.median(base.values())
            require(row["paired_"+label+"_seconds"]==paired_seconds and row["paired_"+label+"_ratios"]==paired and row["paired_"+label+"_median_ratio"]==statistics.median(paired) and row["ratio_of_"+label+"_medians"]==ratio,"INDEPENDENT_RAW_OR_SAME_E_BOLT_TWO_RATIOS:"+label);ratios[label]={"denominator_route":denominator,"ratio_of_medians":ratio,"median_paired_ratio":statistics.median(paired)}
        require(row["matched_Bolt_route"]==matched_bolt(route),"MATCHED_BOLT_DENOMINATOR")
        reference,stack=(prior.verified_arrays(row[k],check) for k in ("reference","timed_outputs"));c=len(raw["mean"])
        require(all(reference[k].dtype==np.float32 and reference[k].shape==(batch,48,c) and np.isfinite(reference[k]).all() for k in ("normalized","physical","scientific_normalized")) and all(stack[k].dtype==np.float32 and stack[k].shape==(5,batch,48,c) for k in ("normalized","physical")),"FINITE_SAVED_FP32_OUTPUT_SHAPES")
        exact(reference["scientific_normalized"],predictions[route][:batch],"exact_selected_TUNE_reference");prior.close(reference["normalized"],predictions[route][:batch],"B1_B8_vs_canonical32_registered_parity");exact(reference["physical"],(reference["normalized"]*raw["std"]+raw["mean"]).astype(np.float32),"reference_physical_inverse")
        for value in (reference,stack):exact(value["origins"],raw["origins"][:batch],"reference_stack_origins")
        require(len(row["parity"])==len(row["memory"])==len(row["positions"])==5,"ALL_RECORDED_SAMPLE_LISTS")
        for i,sample in enumerate(row["samples"]):
            require(sample["dataset"]==data and sample["route"]==route and sample["batch"]==batch and sample["repeat"]==sample["block"]==i and sample["position"]==row["positions"][i]==ordered[i].index(route) and sample["seconds"]==times[i] and sample["unix_end_ns"]>=sample["unix_start_ns"]>seal*1e9,"TIMER_IDENTITY_SCHEDULE_AND_POST_SELECTION_SEAL")
            saved=prior.verified_arrays(sample["sample_artifact"],check);literal=json.loads(Path(sample["sample_artifact"]["path"]).with_suffix(".json").read_text(encoding="utf-8"));require(literal==sample and sample["parity"]==row["parity"][i] and sample["memory"]==row["memory"][i],"SAMPLE_JSON_ROW_AND_STACK_IDENTITY")
            for key in ("normalized","physical"):exact(saved[key],stack[key][i],"sample_stack_"+key)
            exact(saved["origins"],raw["origins"][:batch],"timed_first_B_origins");exact(saved["physical"],(saved["normalized"]*raw["std"]+raw["mean"]).astype(np.float32),"timed_physical_inverse")
            for key,expected,physical_units in (("normalized",reference["normalized"],False),("physical",reference["physical"],True),("scientific_normalized",predictions[route][:batch],False),("scientific_physical",(predictions[route][:batch]*raw["std"]+raw["mean"]).astype(np.float32),True)):parity(sample["parity"][key],saved["physical" if physical_units else "normalized"],expected,physical=physical_units,must_pass=True)
            memory=sample["memory"];require(memory["scope"]==MEMORY_SCOPE and all(type(v) is int and v>=0 for k,v in memory.items() if k!="scope") and memory["max_reserved_VRAM_bytes"]>=memory["reserved_VRAM_bytes"]>=memory["allocated_VRAM_bytes"] and memory["max_reserved_VRAM_bytes"]>=memory["max_allocated_VRAM_bytes"]>=memory["allocated_VRAM_bytes"],"SHARED_RESIDENT_MEMORY_NO_EXCLUSIVE_CLAIM");chronology.append((batch,i,sample["position"],sample))
        exact(stack["block"],np.arange(5,dtype=np.int64),"stack_repeat_ids");exact(stack["position"],np.asarray(row["positions"],np.int64),"stack_position_ids");details.append({"route":route,"batch":batch,"samples":5,"median_seconds":median,"median_seconds_per_window":median/batch,**ratios})
    chronology.sort(key=lambda value:value[:3])
    for a,b in zip(chronology,chronology[1:]):require(a[3]["perf_end_ns"]<=b[3]["perf_start_ns"] and all(a[3]["memory"][key]<=b[3]["memory"][key] for key in ("max_allocated_VRAM_bytes","max_reserved_VRAM_bytes")),"SERIAL_TIMERS_AND_NONRESET_SHARED_PEAKS")
    require(cost["status"]=="COMPLETE" and cost["completed_cost_cells"]==38 and cost["recorded_timer_samples"]==cost["attempted_timer_calls"]==190 and not cost["failed_paths"],"FULL_38_CELLS_190_SAMPLES")
    return {"complete_cells":38,"saved_samples":190,"rows":details,"F_or_timer_replayed":False}


def ridge_precheck(entry,arrays,physical,raw,state,reference,route,frozen=None,encoded=None):
    normalized=prior.causal_physical_preprocess(physical["history_physical"][:8],physical["history_observed"][:8],raw["mean"],raw["std"]);exact(arrays["normalized_history"],normalized,"ridge_physical_fill_normalization");exact(arrays["history_observed"],raw["history_observed"][:8],"ridge_history_mask");exact(arrays["origins"],raw["origins"][:8],"ridge_first8_origins")
    if route=="RAW_CURRENT_C_RIDGE":feature=normalized[:,-1].copy();anchor=np.zeros_like(feature);reference_feature=raw["raw_last"][:8].copy()
    else:
        past=normalized if frozen is None else arrays["latent_history"];feature=np.ascontiguousarray((past-past[:,-1:]).reshape(8,-1));anchor=normalized[:,-1].copy();source_past=raw["history"][:8] if encoded is None else encoded["latent_history"][:8];reference_feature=np.ascontiguousarray((source_past-source_past[:,-1:]).reshape(8,-1))
    exact(arrays["features"],feature,"ridge_actual_fp32_past_features");exact(arrays["anchor"],anchor,"ridge_actual_anchor");exact(arrays["scientific_features"],reference_feature,"ridge_canonical_features");exact(arrays["scientific_anchor"],anchor,"ridge_scientific_anchor");exact(arrays["scientific_normalized"],reference["TUNE_prediction_FP32"][:8],"ridge_scientific_prediction_bytes")
    expected=(feature@state["coefficient_FP32"]+state["intercept_FP32"]).reshape(8,48,len(raw["mean"]))+anchor[:,None];exact(arrays["prediction"],expected,"ridge_independent_FP32_coefficient_arithmetic")
    decisions={"physical_raw_origins":physical["origins"][:8].tobytes()==raw["origins"][:8].tobytes(),"selected_raw_origins":reference["TUNE_origins"][:8].tobytes()==raw["origins"][:8].tobytes(),"physical_raw_mask":physical["history_observed"][:8].tobytes()==raw["history_observed"][:8].tobytes(),"normalized_history_byte_exact":normalized.tobytes()==raw["history"][:8].tobytes(),"anchor_byte_exact":arrays["anchor"].tobytes()==anchor.tobytes()}
    require(entry["exact_lineage"]==decisions and all(decisions.values()) and entry["input_backend_pass"] is entry["prediction_backend_pass"] is True and entry["intermediate_parity_required_for_acceptance"] is False,"RIDGE_INPUT_AND_PREDICTION_ACCEPTANCE")
    parity(entry["prediction_checks"]["normalized"],arrays["prediction"],reference["TUNE_prediction_FP32"][:8]);parity(entry["prediction_checks"]["physical"],(arrays["prediction"]*raw["std"]+raw["mean"]).astype(np.float32),(reference["TUNE_prediction_FP32"][:8]*raw["std"]+raw["mean"]).astype(np.float32),physical=True);parity(entry["intermediate_checks"]["features"],feature,reference_feature,must_pass=False)
    if frozen is not None:
        prior.close(arrays["latent_history"],(normalized-frozen["center"])@frozen["E"],"ridge_independent_NumPy_E");parity(entry["intermediate_checks"]["latent_history"],arrays["latent_history"],encoded["latent_history"][:8],must_pass=False)
    return {"windows_checked":8,"independent_fused_FP32_arithmetic":True,"encoded_E_checked":frozen is not None,"D_bypassed":True,"F_or_timer_replayed":False}


def audit_result(path,meter):
    require(isinstance(COST_SHA,str) and len(COST_SHA)==64,"ACTUAL_COST_NOT_YET_REGISTERED");check=meter.check;binding=linear.file_record(path);require(binding["sha256"]==COST_SHA,"EXACT_ACTUAL_COST_RESULT")
    cost=prior.verified_json(binding,check);require(cost["status"]=="COMPLETE" and cost["cost_split"]=="tune" and not cost["failed_paths"] and set(cost["datasets"])==set(cost["cost"])==set(DATASETS) and cost["routes"]=={d:list(routes(d)) for d in DATASETS},"COMPLETE_REGISTERED_NEW_COST_REQUIRED_FOR_PASS")
    parents={}
    for key,digest in PINS.items():require(cost[key]["sha256"]==digest,"EXACT_ACCEPTED_PARENT:"+key);parents[key]=prior.verified_json(cost[key],check)
    native,ridge,cross=(parents[k] for k in ("new_direction_native_diagnostic","new_direction_ridge_diagnostic","predictive_cross_diagnostic"))
    for key in ("native","ridge"):
        accepted=parents["new_direction_"+key+"_audit"];producer=parents["new_direction_"+key+"_diagnostic"];require(accepted["status"]=="PASS" and producer["status"]=="COMPLETE" and accepted["protected_TEST_accessed"] is producer["protected_TEST_accessed"] is False,"ACCEPTED_COMPLETE_NEW_SCIENCE");same(accepted["diagnostic"],cost["new_direction_"+key+"_diagnostic"])
    same(native["predictive_cross_diagnostic"],ridge["cross_moment_diagnostic"]);same(cost["predictive_cross_diagnostic"],native["predictive_cross_diagnostic"]);same(cost["cross_state_registry"],cross["state_registry"]);same(cost["native_selection_registry"],native["selection_registry"]);same(cost["ridge_selection_registry"],ridge["selection_registry"]);same(cost["cost_plan"],native["plan"]);same(cost["cost_plan"],ridge["plan"])
    science=prior.verified_json(native["causal_lag_diagnostic"],check);subset=prior.verified_json(science["raw_subset_diagnostic"],check);original=prior.verified_json(subset["source_producer"],check)
    require(prior.file_sha(HERE/"run_new_direction_cost_v1.py")==RUNNER_SHA and prior.file_sha(HERE/"whole_new_direction_cost_v1.py")==HELPER_SHA and prior.file_sha(HERE/"whole_f_input_gradient_cost_v1.py")==PARENT_TIMER_SHA,"FROZEN_ACTUAL_COST_SOURCES")
    source=source_contract(*[(HERE/name).read_text(encoding="utf-8") for name in ("whole_new_direction_cost_v1.py","run_new_direction_cost_v1.py","whole_f_input_gradient_cost_v1.py")]);receipt=cost["resource_terminal"];saved_receipt=prior.verified_json(receipt,check)
    require(saved_receipt=={k:v for k,v in receipt.items() if k not in ("path","sha256","bytes")} and receipt["status"]=="COMPLETE" and receipt["gpu_process"] is True and receipt["error"] is None and receipt["ended_unix"]>=receipt["started_unix"] and receipt["process_cpu_seconds"]>=0 and receipt["wall_seconds"]>=0,"EXACT_TERMINAL_PROCESS_RECEIPT")
    for key in ("authorization","continuation_manifest"):bound(receipt[key],check)
    require(cost["new_training_updates"]==0 and all(cost[k] is False for k in ("new_encoder_decoder_fit","new_alpha_or_seed_selection","new_backbone_training","cached_future_returned_instead_of_actual_F_inside_timer","protected_TEST_accessed","independent_holdout","strict_K_only")) and all(cost[k] is True for k in ("selected_state_and_coefficients_setup_outside_timer","all_source_file_state_and_coefficient_hashes_outside_timer","timed_internal_shape_finite_guards","external_meter_and_hash_guards_outside_timer")),"FIXED_INFERENCE_NO_FIT_OR_HIDDEN_SCOPE")
    require(cost["F_before"]==cost["F_after"]==cross["F_before"]==cross["F_after"]==original["F_before"]==original["F_after"] and cost["runtime_behavior_contract"]==cross["runtime_behavior_contract"],"FROZEN_DUAL_F_SIGNATURE_AND_RUNTIME")
    for key in ("model_binary","model_config"):
        for suffix in ("_before","_after"):same(cost[key+suffix],original[key+"_before"]);bound(cost[key+suffix],check)
    guarded=[binding,*[cost[k] for k in PINS],cost["cross_state_registry"],cost["native_selection_registry"],cost["ridge_selection_registry"],cost["cost_plan"],*cost["source_snapshots"],*cost["input_records"],*cost["protected_prior_records"]]
    for item in guarded:
        bound(item,check)
        if "snapshot_path" in item:require(prior.file_sha(item["snapshot_path"])==item["sha256"],"SOURCE_SNAPSHOT_EXACT_BYTES")
    seal=0.;lock_counts={"native":0,"ridge":0}
    for family,producer,status,total in (("native",native,"ALL_SIXTEEN_RESOLVED_BEFORE_DEV",16),("ridge",ridge,"ALL_TWELVE_RESOLVED_BEFORE_DEV",12)):
        registry=prior.verified_json(cost[family+"_selection_registry"],check);require(registry["status"]==status and registry["DEV_opened_before_seal"] is False and len(registry["conditions"])==total,"REGISTERED_ALL_TUNE_SELECTION_LOCKS");seal=max(seal,registry["sealed_unix"]);seen=set()
        for item in registry["conditions"]:
            data=item["dataset"]
            if family=="native":row=producer["datasets"][data]["conditions"][item["condition"]]["native"][str(item["seed"])];key=data+"/"+item["condition"]+"_NATIVE_s"+str(item["seed"])
            else:row=producer["datasets"][data]["routes"][item["route"]];key=data+"/"+item["route"]
            require(key not in seen and item["status"]=="COMPLETE","UNIQUE_COMPLETE_SELECTED_LOCK");seen.add(key);same(item["selection_lock"],row["selection_lock"]);same(item["selection_lock"],cost["selection_locks"][family][key]);lock=prior.verified_json(item["selection_lock"],check);field="selected_state" if family=="native" else "selected_coefficients";same(lock[field],row[field]);require(lock["status"]=="COMPLETE" and lock["DEV_opened_before_lock"] is False and lock["selected_update" if family=="native" else "selected_alpha"]==row["selected_update" if family=="native" else "selected_alpha"],"SAME_CP_ALPHA_NO_RESELECTION");seal=max(seal,lock["sealed_unix"]);guarded.append(item["selection_lock"]);lock_counts[family]+=1
        require(seen==set(cost["selection_locks"][family]),"EXACT_TWENTY_EIGHT_COST_LOCKS")
    hashes,heads,coefficients,datasets={},{},{},{};precheck_count=0
    for data,(c,_) in DATASETS.items():
        record=cost["datasets"][data];raw_source=science["datasets"][data]["spaces"]["PCA"]["conditions"]["L7_RECON"]["splits"]["tune"]["input_source"];same(record["raw_TUNE_source"],raw_source);raw=prior.verified_arrays(raw_source,check);prior.check_raw_split(raw,channels=c,windows=64);proof=record["canonical_physical_bridge"]
        same(proof["original_first_parent"],original["source_parent_results"][data]);parent=prior.verified_json(proof["original_first_parent"],check);same(proof["canonical_input_bridge"],parent["bridge_provenance"]["input_bridge"]);bridge=prior.verified_json(proof["canonical_input_bridge"],check);matches=[r for r in bridge["artifacts"] if r["dataset"]==data and r["split"]=="tune"];require(len(matches)==1,"UNIQUE_ORIGINAL_PHYSICAL_TUNE_MANIFEST");canonical=matches[0];path=Path(canonical["path"]);path=path.resolve() if path.is_absolute() else (ROOT/path).resolve();require(proof["canonical_record"]=={**canonical,"path":str(path)} and proof["split"]=="tune" and proof["dataset"]==data,"EXACT_CANONICAL_PHYSICAL_PAYLOAD")
        same(proof["normalized_raw_input"],original["datasets"][data]["inputs"]["tune"]);require(proof["complete_array_manifest_preserved"] is True and all(proof[k] is False for k in ("inverse_normalization_used","DEV_physical_alias_used","loader_file_SHA_member_shape_dtype_SHA_checks_relaxed","numeric_tolerance_changed")),"PHYSICAL_INPUT_NOT_SYNTHESIZED_OR_RELAXED")
        physical=prior.verified_arrays(proof["canonical_record"],check)
        for key in ("history","history_observed","target","observed","raw_last","origins","mean","std","raw_forecast"):exact(physical[key],raw[key],"full_canonical_raw_"+key)
        metadata=cost["cost"][data]["source_metadata"];predictions={"RAW_BOLT":raw["raw_forecast"]};prechecks={};require(set(record["physical_prechecks"])==set(routes(data)[1:]),"EXACT_EIGHTEEN_FIRST8_PRECHECKS")
        same(metadata["RAW_BOLT"]["reference"],raw_source);require(metadata["RAW_BOLT"]["reference_prediction_array"]=="raw_forecast" and metadata["RAW_BOLT"]["actual_F_forward"] is True,"ORIGINAL_RAW_F_REFERENCE")
        for name in state_names(data):
            owner=native["datasets"][data]["conditions"][name];accepted=cross["datasets"][data]["conditions"][name];same(owner["frozen_state_source"],accepted["state_artifact"]);frozen=prior.verified_arrays(accepted["state_artifact"],check);require(set(frozen)=={"E","D","center"} and linear.state_hash(frozen)==owner["frozen_state_value_sha256"],"SAME_GLOBAL_FROZEN_STATE")
            for kind,seed in (("BOLT",None),("NATIVE",92601),("NATIVE",92602)):
                route=name+"_"+kind+("_s"+str(seed) if seed else "");key=data+"/"+route;m=metadata[route]
                if seed:
                    row=owner["native"][str(seed)];state=prior.verified_arrays(row["selected_state"],check)
                    for member in frozen:exact(state[member],frozen[member],"selected_frozen_"+member)
                    require(set(state)=={"E","D","center","head.weight","head.bias"} and linear.state_hash(state)==row["selected_state_sha256"],"SELECTED_NATIVE_FULL_STATE_HASH");heads[key]=linear.state_hash({k[5:]:v for k,v in state.items() if k.startswith("head.")});require(heads[key]==row["selected_head_state_sha256"],"SELECTED_NATIVE_HEAD_HASH");cp=[p for p in row["checkpoints"] if p["update"]==row["selected_update"]];require(len(cp)==1 and cp[0]["tune"]==row["selected_tune"],"EXACT_SELECTED_TUNE_CP");same(cp[0]["state"],row["selected_state"]);reference_source=cp[0]["artifact"];same(m["selection_lock"],row["selection_lock"]);same(m["selected_state"],row["selected_state"]);require(m["selected_update"]==row["selected_update"] and m["seed"]==seed,"SAME_SELECTED_NATIVE_CP_AND_ORDER_SEED")
                else:state=frozen;reference_source=accepted["splits"]["tune"]["artifact"];same(owner["Bolt"]["tune"]["artifact"],reference_source)
                hashes[key]=previous.state_signature(state);reference=prior.verified_arrays(reference_source,check);same(m["reference"],reference_source);same(m["frozen_state_source"],accepted["state_artifact"]);require(m["kind"]==kind and m["reference_prediction_array"]=="prediction" and m["reference_split"]=="tune" and m["actual_F_forward"]==(kind=="BOLT") and m["actual_native_head_forward"]==(kind=="NATIVE") and m["matched_Bolt_route"]==name+"_BOLT","EXACT_ACTUAL_CALLBACK_METADATA");exact(reference["origins"],raw["origins"],"selected_reference_origins");predictions[route]=reference["prediction"]
                entry=record["physical_prechecks"][route];arrays=prior.verified_arrays(entry["artifact"],check);prechecks[route]=previous.legacy.precheck(entry,arrays,physical,raw,frozen,reference,LAGS)
                if seed:previous.check_native_head(arrays,state)
                precheck_count+=1
        for route in (name for name in routes(data) if "RIDGE" in name):
            name=route.removesuffix("_RIDGE") if route.startswith("RAW_") else route.removesuffix("_RIDGE_HISTORY512")+"_HISTORY512";row=ridge["datasets"][data]["routes"][name];state=prior.verified_arrays(row["selected_coefficients"],check);records,digest=ridge_audit.coefficient_records(state);key=data+"/"+route;coefficients[key]=digest;require(digest==row["selected_coefficient_value_sha256"],"EXACT_SELECTED_RIDGE_COEFFICIENTS");m=metadata[route];same(m["selected_coefficients"],row["selected_coefficients"]);same(m["selection_lock"],row["selection_lock"]);same(m["reference"],row["selected_FIT_TUNE_prediction_artifact"]);same(row["TUNE_sources"]["input_source"],raw_source);require(m["selected_alpha"]==row["selected_alpha"] and m["coefficient_value_sha256"]==digest and m["coefficient_arrays"]==records and m["D_bypassed"] is True and m["actual_F_forward"] is m["actual_native_head_forward"] is m["same_D_or_capacity_match_claimed"] is False and m["input_sources"]==row["TUNE_sources"] and m["reference_prediction_array"]=="TUNE_prediction_FP32" and m["reference_split"]=="tune","SAME_ORIGINAL_V1_RIDGE_SELECTION_AND_D_FREE_SCOPE")
            frozen=encoded=None
            if not name.startswith("RAW_"):
                state_name=name.removesuffix("_HISTORY512");accepted=cross["datasets"][data]["conditions"][state_name];same(row["TUNE_sources"]["encoded_past_source"],accepted["splits"]["tune"]["artifact"]);same(row["TUNE_sources"]["frozen_E_D_source"],accepted["state_artifact"]);encoded=prior.verified_arrays(row["TUNE_sources"]["encoded_past_source"],check);frozen=prior.verified_arrays(accepted["state_artifact"],check);exact(encoded["origins"],raw["origins"],"ridge_encoded_origins")
            require(m["encoded_GPU_E_and_host_transfer_timed"]==(frozen is not None),"ENCODED_HOST_TRANSFER_SCOPE");reference=prior.verified_arrays(row["selected_FIT_TUNE_prediction_artifact"],check);exact(reference["TUNE_origins"],raw["origins"],"ridge_selected_TUNE_origins");predictions[route]=reference["TUNE_prediction_FP32"];entry=record["physical_prechecks"][route];arrays=prior.verified_arrays(entry["artifact"],check);prechecks[route]=ridge_precheck(entry,arrays,physical,raw,state,reference,route,frozen,encoded);precheck_count+=1
        datasets[data]={"prechecks":prechecks,"cost":check_dataset(cost["cost"][data],physical,raw,predictions,check,seal)};require(json.loads((Path(binding["path"]).parent/("cost_"+data)/"RESULT.json").read_text(encoding="utf-8"))==cost["cost"][data],"SAVED_DATASET_COST_RESULT")
    require(hashes==cost["selected_states_before"]==cost["selected_states_after"] and heads==cost["native_heads_before"]==cost["native_heads_after"] and coefficients==cost["ridge_coefficients_before"]==cost["ridge_coefficients_after"],"ALL_SELECTED_FULL_STATES_HEADS_AND_COEFFICIENTS_UNCHANGED")
    require(cost["completed_cost_cells"]==cost["expected_cost_cells"]==76 and cost["recorded_timer_samples"]==cost["expected_timer_samples"]==380 and cost["selected_adapters_loaded"]==24 and cost["selected_native_heads_loaded"]==16 and cost["selected_ridge_states_loaded"]==12 and cost["physical_prechecks_completed"]==cost["expected_physical_prechecks"]==precheck_count==36 and cost["expected_physical_first8_windows"]==288,"EXACT_ACTUAL_76_380_24_16_12_36_COUNTS")
    for data in DATASETS:
        for row in cost["cost"][data]["rows"]:
            for sample in row["samples"]:require(receipt["started_unix"]*1e9<=sample["unix_start_ns"]<=sample["unix_end_ns"]<=receipt["ended_unix"]*1e9,"TIMERS_WITHIN_TERMINAL_PROCESS_RECEIPT")
    for item in guarded:bound(item,check)
    return {"status":"PASS","cost":binding,**{k:cost[k] for k in (*PINS,"cross_state_registry","native_selection_registry","ridge_selection_registry","cost_plan")},"selection_locks":cost["selection_locks"],"source_contract":source,"totals":{"cost_cells_checked":76,"recorded_timer_samples_checked":380,"selected_full_state_hashes_checked":24,"selected_native_head_hashes_checked":16,"selected_ridge_coefficient_hashes_checked":12,"native_selection_locks_checked":lock_counts["native"],"ridge_selection_locks_checked":lock_counts["ridge"],"physical_prechecks_checked":36,"physical_first8_windows_checked":288,"independent_native_head_equations_checked":16,"independent_ridge_FP32_predictions_checked":12,"canonical_TUNE_bridges_checked":2,"protected_prior_file_hashes_checked":len(cost["protected_prior_records"])},"datasets":datasets,"protected_TEST_accessed":False,"independent_F_or_timer_replayed":False,"per_method_exclusive_memory_claimed":False,"scope":"Saved first8 NumPy E/head/lag-D or fused FP32 direct-C ridge arithmetic, exact physical inputs, source/selection/state preservation, all recorded ns and both cost ratios. No F/GPU/optimizer/fit/timer replay; ridge is D-free and not capacity matched; five of nineteen positions only; adaptive DEV remains exposed."}


def execute(path,output):
    from continuation_runtime_v1 import Meter,write_new
    output=Path(output).resolve();require(not output.exists(),"FRESH_COST_AUDIT_OUTPUT");meter=Meter("audit_new_direction_cost_"+output.stem,gpu=False);failure=None;source=linear.file_record(__file__)
    try:
        require(isinstance(COST_SHA,str) and len(COST_SHA)==64,"ACTUAL_COST_NOT_YET_REGISTERED");meter.protected_check();initialize_numpy()
        import sys
        sys.path.insert(0,str(ROOT/"research/smallk_failure_followup_20261014"))
        from prepare_inputs import install_protected_test_guard
        from threadpoolctl import threadpool_limits
        install_protected_test_guard()
        with threadpool_limits(limits=2):report=audit_result(path,meter)
        meter.protected_check();require(prior.file_sha(__file__)==source["sha256"],"AUDITOR_SOURCE_UNCHANGED");json.dumps(report,allow_nan=False)
    except Exception as error:failure={"type":type(error).__name__,"message":str(error)};report={"status":"FAILED_PRESERVED","failure":failure}
    finally:report["audit_source"]=source;report["resource_terminal"]=meter.close(report["status"],failure);write_new(output,report)
    print(json.dumps({"status":report["status"],"totals":report.get("totals"),"failure":failure}),flush=True);return 0 if report["status"]=="PASS" else 2


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--cost",required=True);parser.add_argument("--output",required=True);args=parser.parse_args();raise SystemExit(execute(args.cost,args.output))
