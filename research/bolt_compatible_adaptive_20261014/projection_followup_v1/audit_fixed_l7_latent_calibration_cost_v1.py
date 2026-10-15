"""Independent saved fixed-q TUNE cost audit; no F, GPU, fit or timer replay."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import statistics

import audit_matched_l7_native_cost_v1 as legacy

prior,linear,shared=legacy.prior,legacy.linear,legacy.shared
HERE,ROOT=legacy.HERE,legacy.ROOT
require,exact,parity,bound,same_record,state_signature=legacy.require,legacy.exact,legacy.parity,legacy.bound,legacy.same_record,legacy.state_signature
STATES=("RECON_CP0","E_ONLY_s92601","E_ONLY_s92602")
FAMILIES=("SCALE_ONLY","SCALE_BIAS")
ROUTES=("RAW_BOLT",)+tuple(r for state in STATES for r in (state+"_BOLT",state+"_SCALE_ONLY",state+"_SCALE_BIAS"))
LAGS=(0,1,2,4,8,16,32)
COST_SHA="d8db7445a7a5a750361149cde9a3ad653d4926e871d72c387fe8cbd285aca17b"
Q_SHA="88ddbbf7287ad89b7bf81e170b3e05ddfc89bead700e2fefa50bb0dd4e85f576"
Q_AUDIT_SHA="b31b742d3947a205dbaf01cd84239b3cde95e1ea9baf3331a3c08c48a0aa5ba0"
Q_AUDITOR_SOURCE={"path":str(HERE/"audit_fixed_l7_latent_calibration_v1.py"),"sha256":"208816a2656800afa190a9f713abe3e51c4f8ec88cfc1e6116bd4e4788b2464d"}
Q_AUDIT_COUNTS={
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
RUNNER_SHA="47303f8c326a5cad0134de8537b22c547dddcf62edd4617c431a8a0a295f606a"
HELPER_SHA="e24967a4cea70834df74d9df8552e02cf0287bad69111bb246ac9c006d55b27c"
WRAPPER_SHA="7f0fa92762e03302140bed21eef47cb6df0f1da7a3c6a7007d79f5d34e94e781"
MODULE_SHA="088b9800724b07ced0c42cfa2f514077e430d2c3a7f291dea6f45d1f444cb523"
PARENT_TIMER_SHA="bd59a61ba2c7a14f48c2cf7fe3ed9fa5f366e98eca61019e75672137493ba9e4"
MEMORY_SCOPE="Shared resident frozen F, up to six q adapters and three Bolt adapters; no isolated per-route allocation"
PARENTS=("latent_calibration_diagnostic","latent_calibration_audit","matched_l7_native_diagnostic","matched_l7_native_audit","matched_l7_native_cost","matched_l7_native_cost_audit","native_selection_registry","native_bundle_metadata_reader","parent_report_auditor_source","parent_report","parent_report_manifest","parent_report_audit","E_only_diagnostic","E_only_audit","joint_recon_diagnostic","joint_recon_audit","causal_lag_diagnostic","causal_lag_audit","selection_registry","plan")
np=None


def initialize_numpy():
    global np
    legacy.initialize_numpy(); np=legacy.np


def matched_bolt(route):
    return route.split("_SCALE_")[0]+"_BOLT" if "_SCALE_" in route else route


def schedule():
    order=np.random.default_rng(20261015).permutation(10)
    return [[ROUTES[index] for index in np.roll(order,-repeat)] for repeat in range(5)]


def source_contract(helper,runner,parent,wrapper):
    def functions(text): return {n.name:n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
    h,p=functions(helper),functions(parent)
    def timed(function):
        values=[node for node in ast.walk(function) if isinstance(node,ast.Try) and len(node.body)==3 and "perf_end_ns, unix_end_ns = (time.perf_counter_ns(), time.time_ns())" in ast.unparse(node)]
        require(len(values)==1,"EXACT_SINGLE_TIMED_CALLBACK"); return [ast.dump(node) for node in values[0].body]
    require(timed(h["measure_cost"])==timed(p["measure_cost"]),"CANONICAL_WHOLE_CALLBACK_AND_SYNC_BOUNDARY_UNCHANGED")
    imports=[node for node in ast.parse(helper).body if isinstance(node,ast.ImportFrom) and node.module=="whole_f_input_gradient_cost_v1"]
    require(len(imports)==1 and {"preprocess","whole_predict","parity","fill_vectorized","NORM_ATOL","NORM_RTOL","PHYSICAL_ATOL","PHYSICAL_RTOL"}<={n.name for n in imports[0].names},"IMMUTABLE_IMPORTED_TIMER_AND_TOLERANCES")
    require("ORDER_SEED, WARMUP, REPEATS = (20261015, 2, 5)" in ast.unparse(ast.parse(helper)),"TEN_ROUTES_FIVE_ROTATIONS_WARM_TWO")
    text=ast.unparse(functions(runner)["execute"])
    require("canonical_tune_bridge(original, 'robin')" in text and "adapter_predictor(model, bolt, engine.tensor)" in text and "physical_precheck(physical, raw, reference, model, bolt, engine.tensor)" in text and "compose_reference(baseline, selected)" in text,"ACTUAL_F_SELECTED_Q_AND_CANONICAL_TUNE_REFERENCE")
    class_node=next(n for n in ast.parse(wrapper).body if isinstance(n,ast.ClassDef)); forward=next(n for n in class_node.body if isinstance(n,ast.FunctionDef) and n.name=="forward")
    body=ast.unparse(forward)
    require("original_future = bolt(zpast)" in body and "future = original_future if self.identity else apply_calibration(zpast, original_future, self.a, self.b)" in body and "prediction = self.decode(x[:, -1], zpast, future)" in body and "predict_calibrated" not in body,"IDENTITY_AND_RIDGE_CALL_F_THEN_Q_AND_ORIGINAL_D_NOT_CACHED_BASELINE")
    require("np.isfinite" in ast.unparse(p["whole_predict"]),"INTERNAL_FINITE_GUARDS_REMAIN_TIMED")
    return {"five_of_ten_circular_positions_only":True,"complete_ten_position_balance_claimed":False,"warmup_each_route":2,"separate_untimed_reference_and_physical_prechecks":True,"actual_E_F_q_lag_D_physical_CPU_return_inside_timer":True,"identity_cost_runs_actual_F_and_D":True,"internal_shape_finite_guards_inside_timer":True,"external_meter_source_file_state_and_q_hashes_outside_timer":True,"F_or_timer_replayed":False}


def check_composed_reference(compound,baseline,calibrated,coefficients,kind,decoder):
    from causal_lag_decoder_v1 import lagged_delta_features,predict
    require(set(compound)==set(baseline) and kind in ("IDENTITY","RIDGE"),"UNCHANGED_BASELINE_REFERENCE_MEMBERS_AND_REGISTERED_KIND")
    for key in set(baseline)-{"prediction","latent_future","lag_features"}: exact(compound[key],baseline[key],"composed_unchanged_"+key)
    exact(compound["origins"],calibrated["origins"],"composed_selected_q_origins"); exact(compound["prediction"],calibrated["prediction_FP32"],"selected_q_C_prediction_bytes_preserved"); exact(compound["latent_future"],calibrated["latent_future_calibrated_FP32"],"selected_q_future_bytes_preserved")
    exact(compound["lag_features"],lagged_delta_features(baseline["latent_history"],compound["latent_future"],LAGS),"composed_original_numpy_lag_features")
    original=baseline["latent_future"]; a,b=coefficients["a_FP32"],coefficients["b_FP32"]
    require(a.dtype==b.dtype==np.float32 and a.shape==b.shape==(original.shape[2],) and np.isfinite(a).all() and np.isfinite(b).all(),"FINITE_SELECTED_FP32_PER_K_COEFFICIENTS")
    if kind=="IDENTITY":
        require(np.all(a==0) and np.all(b==0),"IDENTITY_EXACT_ZERO_COEFFICIENTS"); expected=original
        exact(calibrated["prediction_FP32"],baseline["prediction"],"science_identity_C_cache_bytes"); exact(calibrated["latent_future_calibrated_FP32"],original,"science_identity_future_bytes")
    else: expected=original+a*(original-baseline["latent_history"][:,-1:])+b
    prior.close(calibrated["latent_future_calibrated_FP32"],expected,"independent_numpy_canonical_q_future")
    expected_C=predict(baseline["latent_history"],expected,baseline["raw_last"],decoder,LAGS); prior.close(calibrated["prediction_FP32"],expected_C,"independent_numpy_canonical_q_lag_D_C")
    return {"selected_C_prediction_bytes_preserved":True,"matched_baseline_history_and_selected_q_future_bound":True,"independent_NumPy_q_and_lag_D_from_canonical_F_reference":True,"science_identity_future_and_C_byte_checks":kind=="IDENTITY","cost_identity_source_calls_actual_F_and_D":True,"actual_precheck_original_F_future_independently_replayed":False}


def check_dataset(cost,physical,raw,predictions,metadata,check,seal):
    require(cost["routes"]==list(ROUTES) and cost["schedule"]==schedule() and cost["order_seed"]==20261015 and cost["warmup_each_route"]==2 and cost["repeats"]==5 and cost["expected_cells_this_dataset"]==20 and cost["expected_datasets"]==1 and cost["expected_timer_samples"]==100,"TEN_REGISTERED_ROUTES_FIVE_ROTATIONS")
    require(cost["new_training_updates"]==0 and cost["encoder_refit_in_cost"] is cost["per_origin_decoder_fit"] is cost["fit_input_binding_hashes_inside_timer"] is cost["protected_TEST_accessed"] is cost["independent_holdout"] is False and cost["fixed_FIT_decoder_loaded_before_timer"] is True,"FIXED_STATES_NO_FITTING")
    lineage=cost["source_lineage"]
    require(lineage["whole_cost_source_sha256"]==HELPER_SHA and lineage["copied_timer_source_sha256"]==PARENT_TIMER_SHA and Path(lineage["copied_timer_source"]).resolve()==HERE/"whole_f_input_gradient_cost_v1.py" and lineage["copied_preprocess_source_sha256"]==prior.file_sha(HERE.parent/"cost_readout_v1.py") and lineage["imported_functions"]==["fill_vectorized","preprocess","parity","whole_predict"] and lineage["legacy_scientific_runtime_imported"] is False,"EXACT_TIMER_AND_PREPROCESS_LINEAGE")
    exact(prior.causal_physical_preprocess(physical["history_physical"],physical["history_observed"],raw["mean"],raw["std"]),raw["history"],"all_TUNE_original_physical_normalization")
    for key,value in (("physical_past",physical["history_physical"]),("history_observed",physical["history_observed"]),("FIT_mean",raw["mean"]),("FIT_std",raw["std"]),("origins",raw["origins"])): require(cost["input_array_metadata"][key]==shared.array_record(value),"EXACT_COST_INPUT_ARRAY_SHA")
    rows={(r["route"],r["batch"]):r for r in cost["rows"]}; require(set(rows)=={(r,b) for r in ROUTES for b in (1,8)} and len(cost["rows"])==20,"ALL_TWENTY_UNIQUE_ROWS_PRESERVED")
    details,chronology=[],[]
    for (route,batch),row in rows.items():
        require(row["status"]=="COMPLETE" and row["failure"] is None and row["repeats"]==5 and len(row["samples"])==5,"COMPLETE_ROW_HAS_FIVE_TIMERS")
        times=[(s["perf_end_ns"]-s["perf_start_ns"])/1e9 for s in row["samples"]]
        require(row["seconds"]==times and [s["repeat"] for s in row["samples"]]==list(range(5)),"EXACT_NS_DURATION_AND_UNIQUE_REPEATS")
        require(row["selected_metadata"]==cost["source_metadata"][route]==metadata[route] and row["origins"]==raw["origins"][:batch].tolist() and row["warmup_each_route"]==2 and row["latent_calibration_loaded_or_timed"]==("_SCALE_" in route),"EXACT_ROUTE_REFERENCE_AND_NATIVE_SCOPE")
        for flag in ("declared_route_deleted","per_origin_decoder_fit","stage_decomposition","memory_counter_reset_inside_module"): require(row[flag] is False,"NO_HIDDEN_COST_SCOPE:"+flag)
        median=statistics.median(times); require(row["median_seconds"]==median and row["median_seconds_per_window"]==median/batch,"INDEPENDENT_NS_MEDIAN_AND_PER_WINDOW")
        pair={}
        for label,denominator in (("RAW","RAW_BOLT"),("matched_Bolt",matched_bolt(route))):
            base={(s["repeat"]):(s["perf_end_ns"]-s["perf_start_ns"])/1e9 for s in rows[denominator,batch]["samples"]}
            seconds=[base[s["repeat"]] for s in row["samples"]]; ratios=[t/b for t,b in zip(times,seconds)]
            paired=statistics.median(ratios); ratio=median/statistics.median(base.values())
            require(row["paired_"+label+"_seconds"]==seconds and row["paired_"+label+"_ratios"]==ratios and row["paired_"+label+"_median_ratio"]==paired and row["ratio_of_"+label+"_medians"]==ratio,"SAME_REPEAT_AND_MEDIAN_DENOMINATOR:"+label)
            pair[label]={"route":denominator,"median_paired_ratio":paired,"ratio_of_medians":ratio}
        require(row["matched_Bolt_route"]==matched_bolt(route),"MATCHING_FROZEN_E_D_BOLT_DENOMINATOR")
        refs,stack=(prior.verified_arrays(row[k],check) for k in ("reference","timed_outputs")); c=len(raw["mean"])
        require(all(refs[k].dtype==np.float32 and refs[k].shape==(batch,48,c) for k in ("normalized","physical","scientific_normalized")) and all(stack[k].dtype==np.float32 and stack[k].shape==(5,batch,48,c) for k in ("normalized","physical")),"ALL_FINITE_FP32_COST_OUTPUT_SHAPES")
        for arrays in (refs,stack): exact(arrays["origins"],raw["origins"][:batch],"reference_stack_origins")
        exact(refs["scientific_normalized"],predictions[route][:batch],"selected_TUNE_reference"); exact(refs["physical"],(refs["normalized"]*raw["std"]+raw["mean"]).astype(np.float32),"physical_reference_inverse")
        prior.close(refs["normalized"],predictions[route][:batch],"B1_B8_selected_canonical32_reference")
        require(len(row["parity"])==len(row["memory"])==len(row["positions"])==5,"ALL_TIMER_LIST_LENGTHS")
        for i,sample in enumerate(row["samples"]):
            require(sample["dataset"]==cost["dataset"] and sample["route"]==route and sample["batch"]==batch and sample["repeat"]==sample["block"]==i and sample["position"]==row["positions"][i]==schedule()[i].index(route) and sample["seconds"]==times[i]>0 and sample["unix_end_ns"]>=sample["unix_start_ns"]>seal*1e9,"TIMER_IDENTITY_SCHEDULE_AND_POST_LOCK_NS")
            require(sample["parity"]==row["parity"][i] and sample["memory"]==row["memory"][i],"SAMPLE_AND_ROW_PARITY_MEMORY_IDENTITY")
            saved=prior.verified_arrays(sample["sample_artifact"],check)
            literal=json.loads(Path(sample["sample_artifact"]["path"]).with_suffix(".json").read_text(encoding="utf-8")); require(literal==sample,"SAVED_TIMER_JSON_IDENTITY")
            for key in ("normalized","physical"): exact(saved[key],stack[key][i],"sample_stack_"+key)
            exact(saved["origins"],raw["origins"][:batch],"same_first_B_TUNE_origins"); exact(saved["physical"],(saved["normalized"]*raw["std"]+raw["mean"]).astype(np.float32),"timed_physical_CPU_inverse")
            for key,expected,physical_units in (("normalized",refs["normalized"],False),("physical",refs["physical"],True),("scientific_normalized",predictions[route][:batch],False),("scientific_physical",(predictions[route][:batch]*raw["std"]+raw["mean"]).astype(np.float32),True)): parity(sample["parity"][key],saved["physical" if physical_units else "normalized"],expected,physical=physical_units,must_pass=True)
            memory=sample["memory"]; require(memory["scope"]==MEMORY_SCOPE and all(type(v) is int and v>=0 for k,v in memory.items() if k!="scope") and memory["max_reserved_VRAM_bytes"]>=memory["reserved_VRAM_bytes"]>=memory["allocated_VRAM_bytes"] and memory["max_reserved_VRAM_bytes"]>=memory["max_allocated_VRAM_bytes"]>=memory["allocated_VRAM_bytes"],"SHARED_RESIDENT_MEMORY_NO_EXCLUSIVE_CLAIM")
            chronology.append((batch,i,sample["position"],sample))
        exact(stack["block"],np.arange(5,dtype=np.int64),"stack_repeats"); exact(stack["position"],np.asarray(row["positions"],np.int64),"stack_positions")
        details.append({"route":route,"batch":batch,"samples":5,"median_seconds":median,"median_seconds_per_window":median/batch,"RAW":pair["RAW"],"matched_Bolt":pair["matched_Bolt"]})
    chronology.sort(key=lambda x:x[:3])
    for a,b in zip(chronology,chronology[1:]): require(a[3]["perf_end_ns"]<=b[3]["perf_start_ns"] and all(a[3]["memory"][key]<=b[3]["memory"][key] for key in ("max_allocated_VRAM_bytes","max_reserved_VRAM_bytes")),"SERIAL_TIMER_NS_AND_UNRESET_SHARED_PEAKS")
    require(cost["status"]=="COMPLETE" and cost["completed_cost_cells"]==20 and cost["recorded_timer_samples"]==cost["attempted_timer_calls"]==100 and not cost["failed_paths"],"COMPLETE_TWENTY_CELLS_ONE_HUNDRED_SAMPLES")
    return {"complete_cells":20,"saved_samples":100,"rows":details,"F_or_timer_replayed":False}


def audit_result(path,meter):
    require(all(isinstance(v,str) and len(v)==64 for v in (COST_SHA,Q_AUDIT_SHA,RUNNER_SHA)) and isinstance(Q_AUDITOR_SOURCE,dict) and bool(Q_AUDIT_COUNTS),"ACTUAL_FIXED_Q_COST_AND_SCIENCE_PASS_NOT_YET_REGISTERED")
    check=meter.check; binding=linear.file_record(path); require(binding["sha256"]==COST_SHA,"EXACT_ACTUAL_Q_COST_RESULT")
    cost=prior.verified_json(binding,check); require(cost["status"]=="COMPLETE" and cost["cost_split"]=="tune" and cost["routes"]==list(ROUTES) and set(cost["datasets"])==set(cost["cost"])=={"robin"} and not cost["failed_paths"],"FULL_TWENTY_CELL_Q_COST_REQUIRED_FOR_PASS")
    receipt=cost["resource_terminal"]; saved_receipt=prior.verified_json(receipt,check)
    require(saved_receipt=={k:v for k,v in receipt.items() if k not in ("path","sha256")} and receipt["status"]=="COMPLETE" and receipt["error"] is None and receipt["gpu_process"] is True and receipt["ended_unix"]>=receipt["started_unix"] and receipt["wall_seconds"]>=0 and receipt["process_cpu_seconds"]>=0,"SAVED_COMPLETE_GPU_PROCESS_RECEIPT")
    for key in ("authorization","continuation_manifest"): bound(receipt[key],check)
    calibration=prior.verified_json(cost["latent_calibration_diagnostic"],check); accepted=prior.verified_json(cost["latent_calibration_audit"],check)
    require(cost["latent_calibration_diagnostic"]["sha256"]==Q_SHA and cost["latent_calibration_audit"]["sha256"]==Q_AUDIT_SHA and calibration["status"]=="COMPLETE" and accepted["status"]=="PASS" and all(accepted["totals"].get(k)==v for k,v in Q_AUDIT_COUNTS.items()),"EXACT_COMPLETE_Q_SCIENCE_AND_INDEPENDENT_PASS")
    same_record(accepted["diagnostic"],cost["latent_calibration_diagnostic"]); same_record(accepted["audit_source"],Q_AUDITOR_SOURCE); bound(Q_AUDITOR_SOURCE,check)
    for key in PARENTS[2:]: same_record(cost[key],calibration[key])
    same_record(cost["cost_plan"],calibration["plan"])
    native=prior.verified_json(cost["matched_l7_native_diagnostic"],check); science=prior.verified_json(cost["causal_lag_diagnostic"],check); subset=prior.verified_json(science["raw_subset_diagnostic"],check); original=prior.verified_json(subset["source_producer"],check)
    full=prior.verified_json(native["full_gradient_parent"],check); require(cost["runtime_behavior_contract"]==full["runtime_behavior_contract"],"UNCHANGED_F_RUNTIME_NORMALIZATION_CONTRACT")
    for name,digest in (("run_fixed_l7_latent_calibration_cost_v1.py",RUNNER_SHA),("whole_fixed_l7_latent_calibration_cost_v1.py",HELPER_SHA),("calibrated_l7_forecaster_v1.py",WRAPPER_SHA),("fixed_l7_latent_calibration_v1.py",MODULE_SHA),("whole_f_input_gradient_cost_v1.py",PARENT_TIMER_SHA)): require(prior.file_sha(HERE/name)==digest,"EXACT_FROZEN_Q_COST_SOURCE:"+name)
    source=source_contract(*[(HERE/name).read_text(encoding="utf-8") for name in ("whole_fixed_l7_latent_calibration_cost_v1.py","run_fixed_l7_latent_calibration_cost_v1.py","whole_f_input_gradient_cost_v1.py","calibrated_l7_forecaster_v1.py")])
    require(cost["new_training_updates"]==0 and cost["latent_calibration_loaded_or_timed"] is cost["selected_E_D_q_setup_outside_timer"] is cost["all_source_file_state_and_q_metadata_hashes_outside_timer"] is cost["internal_finite_shape_guards_inside_timer"] is True,"FIXED_Q_INFERENCE_SETUP_AND_GUARD_SCOPE")
    for key in ("new_encoder_decoder_or_q_fit","new_CP_family_or_lambda_selection","new_backbone_training","native_predictor_loaded_or_timed","cached_future_returned_instead_of_actual_F_inside_timer","identity_returns_cached_scientific_baseline_inside_timer","protected_TEST_accessed","independent_holdout","strict_K_only"): require(cost[key] is False,"ACTUAL_Q_COST_SCOPE:"+key)
    guarded=[binding,*[cost[k] for k in PARENTS],cost["cost_plan"],*cost["source_snapshots"],*cost["input_records"],*cost["protected_prior_records"]]
    for item in guarded:
        bound(item,check)
        if "snapshot_path" in item: require(prior.file_sha(item["snapshot_path"])==item["sha256"],"EXECUTED_Q_COST_SOURCE_SNAPSHOT_IDENTITY")
    require(cost["F_before"]==cost["F_after"]==calibration["F_parent_signatures_reused"]["before"]==calibration["F_parent_signatures_reused"]["after"]==original["F_before"]==original["F_after"] and set(cost["F_before"])=={"old_algorithm","bi_algorithm"},"FROZEN_ORIGINAL_F_DUAL_SIGNATURE")
    for key in ("model_binary","model_config"):
        for suffix in ("_before","_after"): same_record(cost[key+suffix],original[key+"_before"]); bound(cost[key+suffix],check)
    registry=prior.verified_json(cost["selection_registry"],check); require(registry["status"]=="ALL_SIX_RESOLVED_BEFORE_DEV" and registry["DEV_opened_before_seal"] is False and len(registry["conditions"])==6,"SIX_Q_SELECTION_LOCKS_BEFORE_DEV")
    seal=registry["sealed_unix"]; locks={}
    for item in registry["conditions"]:
        route=item["state"]+"_"+item["family"]; require(route in ROUTES and route not in locks and item["status"]=="COMPLETE","EXACT_SIX_UNIQUE_Q_FAMILIES")
        chosen=calibration["states"][item["state"]]["families"][item["family"]]; same_record(item["selection_lock"],chosen["selection_lock"]); same_record(item["selection_lock"],cost["selection_locks"][route]); lock=prior.verified_json(item["selection_lock"],check)
        require(lock["status"]=="COMPLETE" and lock["DEV_opened_before_lock"] is False and all(lock[k]==chosen[k] for k in ("selected_candidate","selected_kind","selected_lambda","selected_TUNE_MSE","selected_coefficient_value_sha256")),"UNCHANGED_Q_TUNE_SELECTION_NO_RESELECTION"); same_record(lock["selected_coefficients"],chosen["selected_coefficients"])
        locks[route]=item["selection_lock"]; seal=max(seal,lock["sealed_unix"]); guarded.append(item["selection_lock"])
    require(set(locks)==set(cost["selection_locks"]),"EXACT_SIX_SELECTED_Q_LOCKS")
    data=cost["datasets"]["robin"]; require(set(data["physical_prechecks"])==set(ROUTES[1:]),"EXACT_NINE_Q_OR_MATCHED_BOLT_PRECHECKS")
    raw_source=science["datasets"]["robin"]["spaces"]["PCA"]["conditions"]["L7_RECON"]["splits"]["tune"]["input_source"]; same_record(data["raw_TUNE_source"],raw_source)
    raw=prior.verified_arrays(raw_source,check); prior.check_raw_split(raw,channels=17,windows=64)
    proof=data["canonical_physical_bridge"]; require(proof["dataset"]=="robin" and proof["split"]=="tune","EXACT_CANONICAL_TUNE_ONLY")
    same_record(proof["original_first_parent"],original["source_parent_results"]["robin"]); parent=prior.verified_json(proof["original_first_parent"],check); same_record(proof["canonical_input_bridge"],parent["bridge_provenance"]["input_bridge"])
    bridge=prior.verified_json(proof["canonical_input_bridge"],check); matches=[r for r in bridge["artifacts"] if r["dataset"]=="robin" and r["split"]=="tune"]; require(len(matches)==1 and proof["canonical_record"]=={**matches[0],"path":str(Path(matches[0]["path"]).resolve())},"EXACT_FULL_CANONICAL_TUNE_PHYSICAL_MANIFEST")
    same_record(proof["normalized_raw_input"],original["datasets"]["robin"]["inputs"]["tune"])
    require(proof["complete_array_manifest_preserved"] is True and all(proof[k] is False for k in ("inverse_normalization_used","DEV_physical_alias_used","loader_file_SHA_member_shape_dtype_SHA_checks_relaxed","numeric_tolerance_changed")),"NO_SYNTHESIZED_PHYSICAL_OR_LOADER_RELAXATION")
    physical=prior.verified_arrays(proof["canonical_record"],check)
    for key in ("history","history_observed","target","observed","raw_last","origins","mean","std","raw_forecast"): exact(physical[key],raw[key],"full_canonical_payload_"+key)
    predictions={"RAW_BOLT":raw["raw_forecast"]}; metadata={"RAW_BOLT":{"reference":raw_source,"reference_prediction_array":"raw_forecast","reference_split":"tune","actual_F_forward":True,"matched_Bolt_route":"RAW_BOLT"}}; hashes,configs,prechecks,compositions={},{},{},{}
    for state in STATES:
        source_state=calibration["states"][state]["frozen_state_source"]; same_record(source_state,native["states"][state]["frozen_state_source"]); frozen=prior.verified_arrays(source_state,check); require(set(frozen)=={"E","D","center"} and linear.state_hash(frozen)==native["states"][state]["frozen_state_value_sha256"],"ORIGINAL_MATCHED_E_D_CENTER_VALUE_HASH")
        baseline_source=native["states"][state]["Bolt"]["tune"]["artifact"]; baseline=prior.verified_arrays(baseline_source,check)
        for family in (None,*FAMILIES):
            route=state+"_"+(family or "BOLT")
            if family:
                chosen=calibration["states"][state]["families"][family]; coefficients=prior.verified_arrays(chosen["selected_coefficients"],check)
                require(set(coefficients)=={"a_FP64","b_FP64","a_FP32","b_FP32","coefficients_FP64","coefficients_FP32"} and linear.state_hash(coefficients)==chosen["selected_coefficient_value_sha256"],"EXACT_SELECTED_FIT_Q_COEFFICIENT_ARRAYS")
                require(chosen["selected_kind"] in ("IDENTITY","RIDGE") and (family!="SCALE_ONLY" or np.all(coefficients["b_FP32"]==0)),"REGISTERED_Q_FAMILY_AND_IDENTITY_CONFIG")
                selected=chosen["TUNE"]; same_record(selected["input_source"],raw_source); same_record(selected["cached_Bolt_source"],baseline_source); calibrated=prior.verified_arrays(selected["artifact"],check)
                record=cost["cost"]["robin"]["source_metadata"][route]["reference"]; require(Path(record["path"]).resolve()==Path(binding["path"]).parent/(route+"_COMPOSED_TUNE_REFERENCE.npz"),"FRESH_COST_REFERENCE_ARTIFACT_PATH")
                reference=prior.verified_arrays(record,check); compositions[route]=check_composed_reference(reference,baseline,calibrated,coefficients,chosen["selected_kind"],frozen["D"])
                combined={**frozen,"a":coefficients["a_FP32"],"b":coefficients["b_FP32"]}; configs[route]={"coefficients_state_sha256":linear.state_hash({"a":coefficients["a_FP32"],"b":coefficients["b_FP32"]}),"config":{"family":family,"identity":chosen["selected_kind"]=="IDENTITY","lags":list(LAGS),"K":frozen["E"].shape[1],"cached_baseline_returned":False,"identity_actual_F_then_current_D_decode":True}}
                selection={"selected_candidate":chosen["selected_candidate"],"selected_kind":chosen["selected_kind"],"selected_lambda":chosen["selected_lambda"],"selected_coefficients":chosen["selected_coefficients"],"selected_coefficient_value_sha256":chosen["selected_coefficient_value_sha256"],"selection_lock":locks[route],"selected_q_prediction":selected["artifact"],"matched_Bolt_source":baseline_source,"reference_component_composition":"Accepted matched Bolt history and selected q future, original NumPy lag features outside timer; selected C prediction preserved"}
            else: reference,record,combined,selection=baseline,baseline_source,frozen,{"matched_Bolt_source":baseline_source}
            hashes[route]=state_signature(combined)
            for key in ("target","observed","raw_last","origins","mean","std"): exact(reference[key],raw[key],"selected_TUNE_context_"+key)
            predictions[route]=reference["prediction"]; metadata[route]={"frozen_state_label":state,"frozen_state_source":source_state,"family":family,"lags":list(LAGS),"reference":record,"reference_prediction_array":"prediction","reference_split":"tune","actual_F_forward":True,"latent_calibration_applied":bool(family),"matched_Bolt_route":state+"_BOLT",**selection}
            entry=data["physical_prechecks"][route]; arrays=prior.verified_arrays(entry["artifact"],check); prechecks[route]=legacy.legacy.precheck(entry,arrays,physical,raw,frozen,reference,LAGS)
    require(hashes==cost["selected_states_before"]==cost["selected_states_after"] and configs==cost["selected_calibrations_before"]==cost["selected_calibrations_after"],"ALL_NINE_STATES_AND_SIX_COEFFICIENT_CONFIG_HASHES_BEFORE_AFTER")
    details=check_dataset(cost["cost"]["robin"],physical,raw,predictions,metadata,check,seal)
    require(json.loads((Path(binding["path"]).parent/"cost_robin/RESULT.json").read_text(encoding="utf-8"))==cost["cost"]["robin"],"SAVED_DATASET_COST_RESULT_IDENTITY")
    require(cost["completed_cost_cells"]==cost["expected_cost_cells"]==20 and cost["recorded_timer_samples"]==cost["expected_timer_samples"]==100 and cost["selected_adapters_loaded"]==9 and cost["selected_calibrations_loaded"]==6,"COMPLETE_TOP_LEVEL_Q_COST_COUNTS")
    for row in cost["cost"]["robin"]["rows"]:
        for sample in row["samples"]: require(receipt["started_unix"]*1e9<=sample["unix_start_ns"]<=sample["unix_end_ns"]<=receipt["ended_unix"]*1e9,"TIMERS_WITHIN_COST_PROCESS_RECEIPT")
    for item in guarded: bound(item,check)
    return {"status":"PASS","cost":binding,**{key:cost[key] for key in PARENTS},"selection_locks":cost["selection_locks"],"source_contract":source,"totals":{"cost_cells_checked":20,"recorded_timer_samples_checked":100,"selected_full_state_hashes_checked":9,"selected_q_coefficient_config_hashes_checked":6,"selection_locks_checked":6,"physical_prechecks_checked":9,"physical_first8_windows_checked":72,"composed_q_references_checked":6,"independent_canonical_q_and_lag_D_checks":6,"canonical_TUNE_bridges_checked":1,"protected_prior_file_hashes_checked":len(cost["protected_prior_records"])},"datasets":{"robin":{"prechecks":prechecks,"composed_references":compositions,"cost":details}},"protected_TEST_accessed":False,"independent_F_or_timer_replayed":False,"per_method_exclusive_memory_claimed":False,"scope":"Saved TUNE cost outputs, immutable selected coefficients/config and source provenance; NumPy canonical-reference q/lag-D plus actual-precheck E/lag-D arithmetic. Actual precheck original F future is not independently rerun; its batch intermediates remain separate from registered normalized/physical C acceptance. No F/GPU/fit/optimizer/timer replay."}


def execute(path,output):
    from continuation_runtime_v1 import Meter,write_new
    output=Path(output).resolve(); require(not output.exists(),"FRESH_Q_COST_AUDIT_OUTPUT"); meter=Meter("audit_fixed_l7_latent_calibration_cost_"+output.stem,gpu=False); failure=None; source=linear.file_record(__file__)
    try:
        require(all(isinstance(v,str) and len(v)==64 for v in (COST_SHA,Q_AUDIT_SHA,RUNNER_SHA)) and isinstance(Q_AUDITOR_SOURCE,dict) and bool(Q_AUDIT_COUNTS),"ACTUAL_FIXED_Q_COST_AND_SCIENCE_PASS_NOT_YET_REGISTERED"); meter.protected_check(); initialize_numpy()
        import sys
        sys.path.insert(0,str(ROOT/"research/smallk_failure_followup_20261014"))
        from prepare_inputs import install_protected_test_guard
        from threadpoolctl import threadpool_limits
        install_protected_test_guard()
        with threadpool_limits(limits=2): report=audit_result(path,meter)
        meter.protected_check(); require(prior.file_sha(__file__)==source["sha256"],"Q_COST_AUDITOR_SOURCE_UNCHANGED"); json.dumps(report,allow_nan=False)
    except Exception as error: failure={"type":type(error).__name__,"message":str(error)}; report={"status":"FAILED_PRESERVED","failure":failure}
    finally: report["audit_source"]=source; report["resource_terminal"]=meter.close(report["status"],failure); write_new(output,report)
    print(json.dumps({"status":report["status"],"totals":report.get("totals"),"failure":failure}),flush=True); return 0 if report["status"]=="PASS" else 2


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--cost",required=True); parser.add_argument("--output",required=True)
    args=parser.parse_args(); raise SystemExit(execute(args.cost,args.output))
