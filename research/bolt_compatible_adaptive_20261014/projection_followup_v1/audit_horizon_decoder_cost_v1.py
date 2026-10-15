"""Independent saved 72-cell horizon correction cost audit; no F/GPU/fit/timer replay."""
from pathlib import Path
import argparse
import ast
import json
import statistics

import audit_new_direction_cost_v1 as previous
import audit_horizon_decoder_diagnostic_v1 as diagnostic
from audit_fixed_l7_latent_calibration_v1 import receipt_check

prior,linear,shared=previous.prior,previous.linear,previous.shared
HERE,ROOT=previous.HERE,previous.ROOT
require,exact,parity,bound,same=previous.require,previous.exact,previous.parity,previous.bound,previous.same
DATASETS={"robin":(17,4),"beijing_aotizhongxin":(11,2)}
LAGS=(0,1,2,4,8,16,32)
ALLOWED_COST_SHA=("1bbdacf6326b7111828a4f73cb54009580ba6fb021cc200e8a0a7b60cb6cbb59","71391a967f264e578f469ed2b0f1eceb86830954e34ebe86d7d51999f094f713")
RUNNER_SHA="11f0a97e8ad711d35126a4528c34984ddc60174b756813d6d8236c69e3777e5a"
HELPER_SHA="dc6f1f5a227cf31c8ed58a6d3ac789f5a8bbc7eda311a48d40b5acfbf0a9606a"
PARENT_TIMER_SHA="bd59a61ba2c7a14f48c2cf7fe3ed9fa5f366e98eca61019e75672137493ba9e4"
PINS={"horizon_decoder_diagnostic":"5f158382640dac74ec63cc3d1005b385c898821d3c032d51b543f2c9b27d7dac","horizon_decoder_audit":"0f65053536403a3b7b8f53fca3b9f8274fb1a406eaf6dfdcb73bb0467fffe698","cross_moment_diagnostic":"99bffd5831143ca35947099fef9567f89dd209d83451661decee0682b2b8e13c","cross_moment_audit":"90c68a9057909fb5b6c23284e5a45523cdf126da178a5ba0d9d9e96a50dede30"}
MEMORY_SCOPE="Shared frozen F/four original PCA E-D states/24 selected CPU correction coefficient states; no exclusive method allocation"
np=None


def initialize_numpy():
    global np
    previous.initialize_numpy();diagnostic.initialize_numpy();np=previous.np


def spaces(data):
    k=DATASETS[data][1];return ("RAW_C","PCA_K"+str(k),"PCA_K"+str(2*k))


def routes(data):return tuple(s+"_"+m+"_"+v for s in spaces(data) for m in ("F","FLAT") for v in ("BASELINE","STATIC_DELTA","RAMP_DELTA"))


def matched_baseline(route):
    for suffix in ("_STATIC_DELTA","_RAMP_DELTA"):
        if route.endswith(suffix):return route.removesuffix(suffix)+"_BASELINE"
    return route


def schedule(data):
    names=routes(data);order=np.random.default_rng(20261015).permutation(18)
    return [[names[index] for index in np.roll(order,-repeat)] for repeat in range(5)]


def source_contract(helper,runner,parent):
    def functions(text):return {n.name:n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
    h,r,p=map(functions,(helper,runner,parent))
    def timer(function):
        nodes=[n for n in ast.walk(function) if isinstance(n,ast.Try) and len(n.body)==3 and "perf_end_ns, unix_end_ns = (time.perf_counter_ns(), time.time_ns())" in ast.unparse(n)]
        require(len(nodes)==1,"ONE_CANONICAL_TIMED_TRY");return [ast.dump(n) for n in nodes[0].body]
    require(timer(h["measure_cost"])==timer(p["measure_cost"]),"UNCHANGED_WHOLE_CALLBACK_FINAL_SYNC_NS_BOUNDARY")
    imports=[n for n in ast.parse(helper).body if isinstance(n,ast.ImportFrom) and n.module=="whole_f_input_gradient_cost_v1"]
    require(len(imports)==1 and {"preprocess","whole_predict","fill_vectorized","parity","NORM_ATOL","NORM_RTOL","PHYSICAL_ATOL","PHYSICAL_RTOL"}<={n.name for n in imports[0].names} and "ORDER_SEED, WARMUP, REPEATS = (20261015, 2, 5)" in ast.unparse(ast.parse(helper)),"IMMUTABLE_PREPROCESS_TOLERANCES_AND_FIVE_ROTATIONS")
    forward=ast.unparse(r["forecast_components"]);callback=ast.unparse(r["predictor"]);execution=ast.unparse(r["execute"])
    require("torch.no_grad()" in forward and "bolt(tensor(history))" in forward and "bolt(past) if mode == 'F' else" in forward and "if mode == 'F' else np.broadcast_to" in forward and "baseline = future.copy()" in forward and "model.encode(x)" in forward and "model.decode(x[:, -1], past, future)" in forward,"SOURCE_LIVE_F_OR_PAST_ONLY_FLAT_AND_ORIGINAL_DECODER")
    require("active = variant != 'BASELINE' and selected_kind != 'ZERO_DELTA'" in forward and "need_latents = active or capture" in forward and "if need_latents:" in forward and "module.predict(baseline, features, coefficient, variant) if active else baseline.copy()" in forward and "lagged_delta_features(z, f, LAGS)" in forward,"SOURCE_ZERO_FAST_PATH_AND_ACTIVE_CPU_FEATURES")
    require("capture=True" not in callback and "old.canonical_tune_bridge(original, data)" in execution and "capture=True" in ast.unparse(r["physical_precheck"]) and "np.isfinite" in ast.unparse(p["whole_predict"]),"PREFLIGHT_CAPTURE_OUTSIDE_TIMED_CALLBACK_AND_INTERNAL_GUARDS_TIMED")
    calls=[ast.unparse(n.func) for n in ast.walk(ast.parse(runner)) if isinstance(n,ast.Call)];require(not any(name.endswith(("fit_path","backward","optimizer.step","train_head")) for name in calls),"NO_COST_TRAINING_OR_REFIT")
    return {"warmup_each_route":2,"timed_repeats":5,"five_of_eighteen_positions_only":True,"complete_position_balance_claimed":False,"source_live_F_control_flow_verified":True,"source_FLAT_no_F_and_ZERO_no_extra_CPU_matmul_transfer_paths_verified":True,"internal_shape_finite_guards_inside_timer":True,"external_meter_hash_save_memory_and_precheck_capture_outside_timer":True,"runtime_F_event_or_timer_replayed":False}


def check_dataset(cost,physical,raw,predictions,check,seal):
    data=cost["dataset"];names=routes(data);ordered=schedule(data)
    require(cost["routes"]==list(names) and cost["schedule"]==ordered and cost["warmup_each_route"]==2 and cost["repeats"]==5 and cost["order_seed"]==20261015 and cost["expected_cells_this_dataset"]==36 and cost["expected_datasets"]==2 and cost["expected_timer_samples_this_dataset"]==180 and cost["expected_total_cells"]==72 and cost["expected_timer_samples"]==360,"REGISTERED_TWO_DATASET_18_ROUTE_COST")
    lineage=cost["source_lineage"];require(lineage["whole_cost_source_sha256"]==HELPER_SHA and lineage["copied_timer_source_sha256"]==PARENT_TIMER_SHA and Path(lineage["copied_timer_source"]).resolve()==HERE/"whole_f_input_gradient_cost_v1.py" and lineage["copied_preprocess_source_sha256"]==prior.file_sha(HERE.parent/"cost_readout_v1.py") and lineage["imported_functions"]==["fill_vectorized","preprocess","parity","whole_predict"] and lineage["legacy_scientific_runtime_imported"] is False,"IMMUTABLE_TIMER_PREPROCESS_SOURCE_LINEAGE")
    require(cost["new_training_updates"]==0 and cost["encoder_refit_in_cost"] is cost["per_origin_decoder_fit"] is cost["fit_input_binding_hashes_inside_timer"] is cost["protected_TEST_accessed"] is cost["independent_holdout"] is False and cost["fixed_FIT_decoder_loaded_before_timer"] is True,"FIXED_SELECTED_STATES_NO_FITTING")
    exact(prior.causal_physical_preprocess(physical["history_physical"],physical["history_observed"],raw["mean"],raw["std"]),raw["history"],"all_TUNE_physical_causal_fill_FIT_normalization")
    for key,value in (("physical_past",physical["history_physical"]),("history_observed",physical["history_observed"]),("FIT_mean",raw["mean"]),("FIT_std",raw["std"]),("origins",raw["origins"])):require(cost["input_array_metadata"][key]==shared.array_record(value),"EXACT_INPUT_ARRAY_SHA:"+key)
    rows={(r["route"],r["batch"]):r for r in cost["rows"]};require(len(cost["rows"])==36 and set(rows)=={(name,b) for name in names for b in (1,8)},"ALL_36_UNIQUE_ROWS")
    details,chronology=[],[]
    for (route,batch),row in rows.items():
        require(row["status"]=="COMPLETE" and row["failure"] is None and row["repeats"]==len(row["samples"])==5 and row["warmup_each_route"]==2,"FIVE_SAVED_TIMERS_COMPLETE_ROW")
        times=[(s["perf_end_ns"]-s["perf_start_ns"])/1e9 for s in row["samples"]];require(row["seconds"]==times and all(t>0 for t in times) and [s["repeat"] for s in row["samples"]]==list(range(5)),"POSITIVE_INTEGER_NS_DURATIONS_AND_REPEATS")
        require(row["selected_metadata"]==cost["source_metadata"][route] and row["origins"]==raw["origins"][:batch].tolist() and row["native_head_loaded_or_timed"] is False and row["selected_delta_loaded_or_timed"]==(not route.endswith("_BASELINE")) and row["F_skipped_for_FLAT"]==("_FLAT_" in route),"EXACT_SELECTED_ROUTE_F_FLAT_AND_DELTA_SCOPE")
        require(all(row[k] is False for k in ("declared_route_deleted","per_origin_decoder_fit","stage_decomposition","memory_counter_reset_inside_module")),"NO_DROPPED_PATH_OR_HIDDEN_COST_SCOPE")
        median=statistics.median(times);require(row["median_seconds"]==median and row["median_seconds_per_window"]==median/batch,"INDEPENDENT_NS_MEDIANS")
        ratios={}
        for label,denominator in (("RAW","RAW_C_F_BASELINE"),("matched_baseline",matched_baseline(route))):
            base={s["repeat"]:(s["perf_end_ns"]-s["perf_start_ns"])/1e9 for s in rows[denominator,batch]["samples"]};paired_seconds=[base[s["repeat"]] for s in row["samples"]];paired=[t/b for t,b in zip(times,paired_seconds)];ratio=median/statistics.median(base.values())
            require(row["paired_"+label+"_seconds"]==paired_seconds and row["paired_"+label+"_ratios"]==paired and row["paired_"+label+"_median_ratio"]==statistics.median(paired) and row["ratio_of_"+label+"_medians"]==ratio,"INDEPENDENT_RAW_OR_SAME_SPACE_MODE_BASELINE_TWO_RATIOS:"+label);ratios[label]={"denominator_route":denominator,"ratio_of_medians":ratio,"median_paired_ratio":statistics.median(paired)}
        require(row["matched_baseline_route"]==matched_baseline(route),"MATCHED_BASELINE_DENOMINATOR")
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
    require(cost["status"]=="COMPLETE" and cost["completed_cost_cells"]==36 and cost["recorded_timer_samples"]==cost["attempted_timer_calls"]==180 and not cost["failed_paths"],"FULL_36_CELLS_180_SAMPLES")
    return {"complete_cells":36,"saved_samples":180,"rows":details,"F_or_timer_replayed":False}


def precheck(entry,arrays,physical,raw,reference,canonical,frozen,space,mode,variant,coefficient,kind):
    normalized=prior.causal_physical_preprocess(physical["history_physical"][:8],physical["history_observed"][:8],raw["mean"],raw["std"]);exact(arrays["normalized_history"],normalized,"physical_causal_fill_FIT_scale");exact(arrays["history_observed"],raw["history_observed"][:8],"physical_past_mask");exact(arrays["origins"],raw["origins"][:8],"physical_first8_origins");exact(arrays["scientific_normalized"],reference[:8],"exact_scientific_selected_reference")
    z,f=arrays["latent_history"],arrays["latent_future"]
    if space=="RAW_C":
        exact(z,normalized,"Raw_direct_past_stream");base=f if mode=="F" else np.broadcast_to(normalized[:,-1:],(8,48,normalized.shape[2])).copy();exact(arrays["baseline"],base,"Raw_original_F_or_current_C_baseline_no_reassociation")
    else:
        prior.close(z,(normalized-frozen["center"])@frozen["E"],"independent_NumPy_E_first8");base=normalized[:,-1,None]+diagnostic.old_lag_features(z,f)@frozen["D"];prior.close(arrays["baseline"],base,"independent_original_PCA_D_first8")
    if mode=="FLAT":exact(f,np.broadcast_to(z[:,-1:],f.shape).copy(),"FLAT_is_known_latent_last_no_F")
    phi=diagnostic.old_lag_features(z,f);exact(arrays["features"],phi,"independent_lag_specific_delta_features")
    no_correction=variant=="BASELINE" or kind=="ZERO_DELTA"
    wanted=arrays["baseline"].copy() if no_correction else diagnostic.deployed_prediction(arrays["baseline"],phi,coefficient["coefficient_FP32"],variant)
    exact(arrays["prediction"],wanted,"independent_first8_selected_CPU_FP32_correction")
    decisions={"physical_raw_origins":physical["origins"][:8].tobytes()==raw["origins"][:8].tobytes(),"physical_raw_mask":physical["history_observed"][:8].tobytes()==raw["history_observed"][:8].tobytes(),"normalized_history_byte_exact":normalized.tobytes()==raw["history"][:8].tobytes()}
    require(entry["exact_lineage"]==decisions and all(decisions.values()) and entry["input_backend_pass"] is entry["prediction_backend_pass"] is entry["capture_diagnostics_and_extra_host_transfers_outside_timer"] is True and entry["intermediate_parity_required_for_acceptance"] is False,"EXACT_PHYSICAL_SOURCE_AND_ACCEPTANCE_FLAGS")
    parity(entry["prediction_checks"]["normalized"],arrays["prediction"],reference[:8],must_pass=True);parity(entry["prediction_checks"]["physical"],(arrays["prediction"]*raw["std"]+raw["mean"]).astype(np.float32),(reference[:8]*raw["std"]+raw["mean"]).astype(np.float32),physical=True,must_pass=True)
    for key in ("baseline","features","latent_history","latent_future"):parity(entry["intermediate_checks"][key],arrays[key],canonical[key][:8],must_pass=False)
    return {"space":space,"mode":mode,"variant":variant,"first8_B8_component_arithmetic_checked":True,"ZERO_or_baseline_exact_copy":no_correction,"saved_F_latents_used_not_reinferred":mode=="F","all_timed_B1_B8_latent_components_replayed":False}


def canonical_bridge(proof,original,data,raw,check):
    same(proof["original_first_parent"],original["source_parent_results"][data]);parent=prior.verified_json(proof["original_first_parent"],check);same(proof["canonical_input_bridge"],parent["bridge_provenance"]["input_bridge"]);bridge=prior.verified_json(proof["canonical_input_bridge"],check);matches=[r for r in bridge["artifacts"] if r["dataset"]==data and r["split"]=="tune"];require(len(matches)==1,"ONE_ORIGINAL_TUNE_PHYSICAL_MANIFEST");canonical=matches[0];path=Path(canonical["path"]);path=path.resolve() if path.is_absolute() else (ROOT/path).resolve()
    require(proof["canonical_record"]=={**canonical,"path":str(path)} and proof["split"]=="tune" and proof["dataset"]==data and proof["complete_array_manifest_preserved"] is True and all(proof[k] is False for k in ("inverse_normalization_used","DEV_physical_alias_used","loader_file_SHA_member_shape_dtype_SHA_checks_relaxed","numeric_tolerance_changed")),"ORIGINAL_CANONICAL_PHYSICAL_PAYLOAD_NO_SYNTHESIS")
    same(proof["normalized_raw_input"],original["datasets"][data]["inputs"]["tune"]);physical=prior.verified_arrays(proof["canonical_record"],check)
    for key in ("history","history_observed","target","observed","raw_last","origins","mean","std","raw_forecast"):exact(physical[key],raw[key],"canonical_raw_"+key)
    return physical


def metadata(path,check):
    require(isinstance(ALLOWED_COST_SHA,tuple) and len(ALLOWED_COST_SHA)==len(set(ALLOWED_COST_SHA))==2 and all(isinstance(v,str) and len(v)==64 for v in ALLOWED_COST_SHA),"ACTUAL_HORIZON_COST_NOT_REGISTERED");binding=linear.file_record(path);require(binding["sha256"] in ALLOWED_COST_SHA,"EXACT_REGISTERED_PRIMARY_OR_REPEAT_HORIZON_COST");cost=prior.verified_json(binding,check)
    require(cost["status"]=="COMPLETE" and cost["cost_split"]=="tune" and not cost["failed_paths"] and cost["routes"]=={d:list(routes(d)) for d in DATASETS} and set(cost["datasets"])==set(cost["cost"])==set(DATASETS),"FULL_TWO_DATASET_72_COST_REQUIRED_FOR_PASS")
    for key,digest in PINS.items():require(cost[key]["sha256"]==digest,"EXACT_ACCEPTED_PARENT:"+key)
    return binding,cost


def audit_result(path,meter):
    check=meter.check;binding,cost=metadata(path,check);parents={k:prior.verified_json(cost[k],check) for k in PINS};source,cross=parents["horizon_decoder_diagnostic"],parents["cross_moment_diagnostic"]
    require(source["status"]==cross["status"]=="COMPLETE" and parents["horizon_decoder_audit"]["status"]==parents["cross_moment_audit"]["status"]=="PASS" and all(not p.get("protected_TEST_accessed",False) for p in parents.values()),"ACCEPTED_SCIENCE_AND_CROSS_PASSES");same(parents["horizon_decoder_audit"]["diagnostic"],cost["horizon_decoder_diagnostic"]);same(parents["cross_moment_audit"]["diagnostic"],cost["cross_moment_diagnostic"]);same(cost["selection_registry"],source["selection_registry"])
    science=prior.verified_json(cost["causal_lag_diagnostic"],check);subset=prior.verified_json(science["raw_subset_diagnostic"],check);original=prior.verified_json(subset["source_producer"],check)
    for name,digest in (("run_horizon_decoder_cost_v1.py",RUNNER_SHA),("whole_horizon_decoder_cost_v1.py",HELPER_SHA),("whole_f_input_gradient_cost_v1.py",PARENT_TIMER_SHA)):require(prior.file_sha(HERE/name)==digest,"FROZEN_EXECUTED_COST_SOURCE")
    contract=source_contract(*[(HERE/name).read_text(encoding="utf-8") for name in ("whole_horizon_decoder_cost_v1.py","run_horizon_decoder_cost_v1.py","whole_f_input_gradient_cost_v1.py")]);receipt=cost["resource_terminal"];receipts=receipt_check(receipt,"COMPLETE",check)
    require(cost["new_training_updates"]==0 and all(cost[k] is False for k in ("new_encoder_decoder_fit","new_alpha_arm_or_K_selection","new_backbone_training","cached_future_returned_instead_of_actual_F_inside_timer","protected_TEST_accessed","independent_holdout","strict_K_only","capacity_match_RAW_and_PCA_claimed")) and all(cost[k] is True for k in ("selected_state_and_coefficients_setup_outside_timer","all_source_file_state_and_coefficient_hashes_outside_timer","ZERO_selected_skips_coefficient_matmul_and_extra_latent_host_transfer","timed_internal_shape_finite_guards","external_meter_and_hash_guards_outside_timer")),"REGISTERED_FIXED_INFERENCE_AND_TIMER_SCOPE")
    require(cost["F_before"]==cost["F_after"]==cross["F_before"]==cross["F_after"]==original["F_before"]==original["F_after"] and cost["runtime_behavior_contract"]==cross["runtime_behavior_contract"],"FROZEN_F_DUAL_SIGNATURE_AND_RUNTIME")
    for key in ("model_binary","model_config"):
        for suffix in ("_before","_after"):same(cost[key+suffix],original[key+"_before"]);bound(cost[key+suffix],check)
    guarded=[binding,*[cost[k] for k in PINS],cost["selection_registry"],cost["cost_plan"],*cost["source_snapshots"],*cost["input_records"],*cost["protected_prior_records"],*receipts]
    for item in guarded:
        bound(item,check)
        if "snapshot_path" in item:require(prior.file_sha(item["snapshot_path"])==item["sha256"],"EXECUTED_SOURCE_SNAPSHOT_BYTES")
    registry=prior.verified_json(cost["selection_registry"],check);require(registry["status"]=="ALL_24_RESOLVED_BEFORE_DEV" and registry["DEV_opened_before_seal"] is False and len(registry["conditions"])==24,"ALL_24_ORIGINAL_SELECTION_LOCKS");seal=registry["sealed_unix"];seen=set()
    for item in registry["conditions"]:
        data,space,mode,arm=(item[k] for k in ("dataset","space","mode","arm"));key=data+"/"+space+"_"+mode+"_"+arm;require(key not in seen and item["status"]=="COMPLETE","UNIQUE_COMPLETE_REGISTERED_LOCK");seen.add(key);row=source["datasets"][data]["spaces"][space]["conditions"][mode+"_"+arm];same(item["selection_lock"],row["selection_lock"]);same(item["selection_lock"],cost["selection_locks"][key]);lock=prior.verified_json(item["selection_lock"],check);same(lock["selected_coefficients"],row["selected_coefficients"]);require(lock["status"]=="COMPLETE" and lock["DEV_opened_before_lock"] is False and all(lock[k]==row[k] for k in ("selected_kind","selected_candidate","selected_alpha","selected_TUNE_MSE","selected_coefficient_value_sha256")),"ORIGINAL_TUNE_WINNER_NO_RESELECTION");seal=max(seal,lock["sealed_unix"]);guarded.append(item["selection_lock"])
    require(seen==set(cost["selection_locks"]) and len(seen)==24,"EXACT_24_COST_SELECTION_LOCKS")
    models,coefficients,datasets={}, {}, {};prechecks=0;zero_count=0
    for data,(c,_) in DATASETS.items():
        record=cost["datasets"][data];raw_source=science["datasets"][data]["spaces"]["PCA"]["conditions"]["L7_RECON"]["splits"]["tune"]["input_source"];same(record["raw_TUNE_source"],raw_source);raw=prior.verified_arrays(raw_source,check);prior.check_raw_split(raw,channels=c,windows=64);physical=canonical_bridge(record["canonical_physical_bridge"],original,data,raw,check);require(set(record["physical_prechecks"])==set(routes(data)),"ALL_18_FIRST8_PREFLIGHTS_PER_DATASET");predictions={};details={};metadata=cost["cost"][data]["source_metadata"]
        for space in spaces(data):
            owner=source["datasets"][data]["spaces"][space];context=owner["contexts"]["tune"];same(context["input_source"],raw_source);frozen=cached=None;state_record=cache_record=None
            if space=="RAW_C":z=raw["history"];F=raw["raw_forecast"];bases={"F":F,"FLAT":np.broadcast_to(raw["raw_last"][:,None],raw["target"].shape).copy()}
            else:
                accepted=cross["datasets"][data]["conditions"][space];state_record=accepted["state_artifact"];cache_record=accepted["splits"]["tune"]["artifact"];same(context["frozen_state_source"],state_record);same(context["cached_latent_source"],cache_record);frozen=prior.verified_arrays(state_record,check);cached=prior.verified_arrays(cache_record,check);require(set(frozen)=={"E","D","center"},"ORIGINAL_PCA_STATE_MEMBERS");models[data+"/"+space]=previous.previous.state_signature(frozen);exact(cached["origins"],raw["origins"],"original_cached_TUNE_origins");z,F=cached["latent_history"],cached["latent_future"];bases={"F":cached["prediction"],"FLAT":cached["FLAT_prediction"]}
            for mode in ("F","FLAT"):
                future=F if mode=="F" else np.broadcast_to(z[:,-1:],F.shape).copy();canonical={"baseline":bases[mode],"latent_history":z,"latent_future":future,"features":diagnostic.old_lag_features(z,future)}
                for variant in ("BASELINE","STATIC_DELTA","RAMP_DELTA"):
                    route=space+"_"+mode+"_"+variant;m=metadata[route];row=state=kind=None;reference=bases[mode];reference_record=raw_source if space=="RAW_C" else cache_record;member=("raw_forecast" if mode=="F" else "currentC_broadcast") if space=="RAW_C" else ("prediction" if mode=="F" else "FLAT_prediction")
                    if variant!="BASELINE":
                        row=owner["conditions"][mode+"_"+variant];state=prior.verified_arrays(row["selected_coefficients"],check);require(set(state)=={"coefficient_FP64","coefficient_FP32"} and all(np.isfinite(v).all() for v in state.values()),"SELECTED_FINITE_DELTA_STATE");exact(state["coefficient_FP32"],state["coefficient_FP64"].astype(np.float32),"selected_delta_cast");digest=diagnostic.ridge.coefficient_records(state)[1];require(digest==row["selected_coefficient_value_sha256"]==m["coefficient_value_sha256"],"EXACT_SELECTED_DELTA_VALUES");coefficients[data+"/"+route]=digest;kind=row["selected_kind"];same(m["selection_lock"],row["selection_lock"]);same(m["selected_coefficients"],row["selected_coefficients"]);require(all(m[k]==row[k] for k in ("selected_candidate","selected_kind","selected_alpha")),"SAME_SELECTED_ALPHA_AND_KIND");reference_record=row["selected_FIT_TUNE_prediction_artifact"];saved=prior.verified_arrays(reference_record,check);exact(saved["TUNE_origins"],raw["origins"],"selected_reference_origins");reference=saved["TUNE_prediction_FP32"];member="TUNE_prediction_FP32";zero_count+=int(kind=="ZERO_DELTA")
                        if kind=="ZERO_DELTA":require(all(not np.any(v) for v in state.values()),"ZERO_SELECTED_COEFFICIENT_ARRAYS")
                    same(m["reference"],reference_record)
                    if state_record is None:require(m["frozen_state_source"] is None,"RAW_NO_PCA_STATE")
                    else:same(m["frozen_state_source"],state_record)
                    require(m["space"]==space and m["mode"]==mode and m["variant"]==variant and m["lags"]==list(LAGS) and m["reference_split"]=="tune" and m["reference_prediction_array"]==member and m["actual_F_forward"]==(mode=="F") and m["no_F_for_FLAT"]==(mode=="FLAT") and m["ZERO_selected_baseline_shortcut"]==(kind=="ZERO_DELTA") and m["raw_F_baseline_reassociated"] is m["capacity_match_RAW_and_PCA_claimed"] is False and m["matched_baseline_route"]==matched_baseline(route),"MATCHED_REFERENCE_FORWARD_AND_SHORTCUT_METADATA")
                    predictions[route]=reference;entry=record["physical_prechecks"][route];arrays=prior.verified_arrays(entry["artifact"],check);details[route]=precheck(entry,arrays,physical,raw,reference,canonical,frozen,space,mode,variant,state,kind);prechecks+=1
        datasets[data]={"prechecks":details,"cost":check_dataset(cost["cost"][data],physical,raw,predictions,check,seal)};require(json.loads((Path(binding["path"]).parent/("cost_"+data)/"RESULT.json").read_text(encoding="utf-8"))==cost["cost"][data],"SAVED_DATASET_TERMINAL_COST_RESULT")
    require(models==cost["frozen_E_D_before"]==cost["frozen_E_D_after"] and coefficients==cost["selected_coefficients_before"]==cost["selected_coefficients_after"] and len(models)==cost["frozen_PCA_models_loaded"]==4 and len(coefficients)==cost["selected_correction_states_loaded"]==24,"ALL_ORIGINAL_PCA_AND_SELECTED_COEFFICIENT_HASHES_UNCHANGED")
    require(cost["completed_cost_cells"]==cost["expected_cost_cells"]==72 and cost["recorded_timer_samples"]==cost["expected_timer_samples"]==360 and cost["physical_prechecks_completed"]==cost["expected_physical_prechecks"]==prechecks==36 and cost["expected_physical_first8_windows"]==288,"ALL_72_COST_CELLS_360_SAMPLES_36_PREFLIGHTS")
    for data in DATASETS:
        for row in cost["cost"][data]["rows"]:
            for sample in row["samples"]:require(receipt["started_unix"]*1e9<=sample["unix_start_ns"]<=sample["unix_end_ns"]<=receipt["ended_unix"]*1e9,"ALL_TIMERS_WITHIN_ACTUAL_PROCESS_RECEIPT")
    for item in guarded:bound(item,check)
    return {"status":"PASS","cost":binding,**{k:cost[k] for k in (*PINS,"selection_registry","cost_plan")},"selection_locks":cost["selection_locks"],"source_contract":contract,"datasets":datasets,"totals":{"cost_cells_checked":72,"recorded_timer_samples_checked":360,"frozen_PCA_state_hashes_checked":4,"selected_delta_coefficient_hashes_checked":24,"selection_locks_checked":24,"physical_prechecks_checked":36,"physical_first8_windows_checked":288,"independent_first8_B8_component_prediction_equations_checked":36,"ZERO_selected_fastpath_prechecks_checked":zero_count,"canonical_TUNE_bridges_checked":2,"protected_prior_file_hashes_checked":len(cost["protected_prior_records"])},"protected_TEST_accessed":False,"independent_F_or_timer_replayed":False,"all_timed_sample_latent_components_available_or_replayed":False,"per_method_exclusive_memory_claimed":False,"scope":"Saved physical-first8 B8 NumPy E/lag-D/selected CPU corrections and original Raw baseline; all B1/B8 saved references/output parity/ns/ratios and state hashes. Timed samples do not save every latent component. No F/GPU/fit/timer replay; same first origins/five of eighteen positions; adaptive DEV remains exposed; current cohort only."}


def execute(path,output):
    from continuation_runtime_v1 import Meter,write_new
    output=Path(output).resolve();require(not output.exists(),"FRESH_HORIZON_COST_AUDIT_OUTPUT");meter=Meter("audit_horizon_cost_"+output.stem,gpu=False);failure=None;source=linear.file_record(__file__)
    try:
        metadata(path,meter.check);meter.protected_check();initialize_numpy()
        import sys
        sys.path.insert(0,str(ROOT/"research/smallk_failure_followup_20261014"))
        from prepare_inputs import install_protected_test_guard
        from threadpoolctl import threadpool_limits
        install_protected_test_guard()
        with threadpool_limits(limits=2):report=audit_result(path,meter)
        meter.protected_check();require(prior.file_sha(__file__)==source["sha256"],"AUDIT_SOURCE_IMMUTABLE");json.dumps(report,allow_nan=False)
    except Exception as error:failure={"type":type(error).__name__,"message":str(error)};report={"status":"FAILED_PRESERVED","failure":failure}
    finally:report["audit_source"]=source;report["resource_terminal"]=meter.close(report["status"],failure);write_new(output,report)
    print(json.dumps({"status":report["status"],"totals":report.get("totals"),"failure":failure}),flush=True);return 0 if report["status"]=="PASS" else 2


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--cost",required=True);parser.add_argument("--output",required=True);args=parser.parse_args();raise SystemExit(execute(args.cost,args.output))


