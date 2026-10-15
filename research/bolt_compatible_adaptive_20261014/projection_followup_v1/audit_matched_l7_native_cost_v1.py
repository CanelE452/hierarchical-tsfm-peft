"""Independent saved TUNE cost audit; no F, GPU, optimizer or timer replay."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import statistics

import audit_f_input_gradient_cost_v1 as legacy

prior,linear,shared=legacy.prior,legacy.linear,legacy.shared
HERE,ROOT=legacy.HERE,legacy.ROOT
require,exact,parity,bound,same_record=legacy.require,legacy.exact,legacy.parity,legacy.bound,legacy.same_record
STATES=("RECON_CP0","E_ONLY_s92601","E_ONLY_s92602")
ROUTES=("RAW_BOLT",)+tuple(r for state in STATES for r in (state+"_BOLT",state+"_NATIVE_s92601",state+"_NATIVE_s92602"))
LAGS=(0,1,2,4,8,16,32)
COST_SHA="6fce6c6a462cb0b45f6cc3932f94c615ad02b17892c6aade642bc2ad39a6b51c"
NATIVE_SHA="4a6f2d3601c27bf000659aa99a586a2f04d72371766f7558cc78cc16028caa2e"
NATIVE_AUDIT_SHA="429296ac7346189953813292291584d0c5ceb0a60028b4c9b955a2a429a1db87"
RUNNER_SHA="679b42eddfcc1eef40c3d0e03074bfa5740e6d7fcb050a3ef3ede6a041981183"
HELPER_SHA="3487ac7c18dd25cbfe60bc49db0f20587f620a0a37197b9f7aaa66640dab89ed"
PARENT_TIMER_SHA="bd59a61ba2c7a14f48c2cf7fe3ed9fa5f366e98eca61019e75672137493ba9e4"
MEMORY_SCOPE="Shared resident frozen F, up to six heads and three Bolt adapters; no isolated per-route allocation"
PARENTS=("matched_l7_native_diagnostic","matched_l7_native_audit","E_only_diagnostic","E_only_audit","joint_recon_diagnostic","joint_recon_audit","full_gradient_parent","full_gradient_parent_audit","causal_lag_diagnostic","causal_lag_audit","parent_report","parent_report_manifest","parent_report_audit","selection_registry","plan")
np=None


def initialize_numpy():
    global np
    legacy.initialize_numpy(); np=legacy.np


def matched_bolt(route):
    return route.split("_NATIVE_")[0]+"_BOLT" if "_NATIVE_" in route else route


def schedule():
    order=np.random.default_rng(20261015).permutation(10)
    return [[ROUTES[index] for index in np.roll(order,-repeat)] for repeat in range(5)]


def source_contract(helper,runner,parent):
    def functions(text): return {n.name:n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
    h,p=functions(helper),functions(parent)
    def timed(function):
        values=[node for node in ast.walk(function) if isinstance(node,ast.Try) and len(node.body)==3 and "perf_end_ns, unix_end_ns = (time.perf_counter_ns(), time.time_ns())" in ast.unparse(node)]
        require(len(values)==1,"EXACT_SINGLE_TIMED_CALLBACK"); return [ast.dump(node) for node in values[0].body]
    require(timed(h["measure_cost"])==timed(p["measure_cost"]),"CANONICAL_WHOLE_CALLBACK_AND_SYNC_BOUNDARY_UNCHANGED")
    imports=[node for node in ast.parse(helper).body if isinstance(node,ast.ImportFrom) and node.module=="whole_f_input_gradient_cost_v1"]
    require(len(imports)==1 and {"preprocess","whole_predict","parity","fill_vectorized","NORM_ATOL","NORM_RTOL","PHYSICAL_ATOL","PHYSICAL_RTOL"} <= {n.name for n in imports[0].names},"IMMUTABLE_IMPORTED_TIMER_AND_TOLERANCES")
    require("ORDER_SEED, WARMUP, REPEATS = (20261015, 2, 5)" in ast.unparse(ast.parse(helper)) and "range(REPEATS)" in ast.unparse(h["balanced_schedule"]),"TEN_ROUTES_FIVE_ROTATIONS_WARM_TWO")
    r=functions(runner); text=ast.unparse(r["execute"])
    require('canonical_tune_bridge(original, \'robin\')' in text and "predictor = None" in text and "adapter_predictor(model, predictor, engine.tensor)" in text and "physical_precheck(physical, raw, reference, model, predictor, engine.tensor)" in text and "frozen_status(" not in text,"ACTUAL_SELECTED_NATIVE_OR_F_CALLBACK_AND_TUNE_INPUT")
    require('"tune"' in runner and "old.selected_TUNE_checkpoint(row)" in runner and 'owner["Bolt"]["tune"]["artifact"]' in runner,"EXACT_SELECTED_TUNE_REFERENCES")
    original=(HERE/"run_linear_e_cost_v1.py").read_text(encoding="utf-8"); callback=functions(original)["adapter_predictor"]
    require("model(tensor(history), bolt).detach().cpu().numpy().copy()" in ast.unparse(callback),"ACTUAL_FORWARD_CPU_MATERIALIZATION_NOT_CACHED_FUTURE")
    require("np.isfinite" in ast.unparse(p["whole_predict"]) and 'raise ValueError' in ast.unparse(p["whole_predict"]),"INTERNAL_FINITE_SHAPE_GUARDS_REMAIN_INSIDE_TIMER")
    return {"five_of_ten_circular_positions_only":True,"complete_ten_position_balance_claimed":False,"warmup_each_route":2,"separate_untimed_reference_and_physical_prechecks":True,"actual_E_F_or_head_lag_D_physical_CPU_return_inside_timer":True,"internal_shape_finite_guards_inside_timer":True,"external_meter_source_file_state_hashes_outside_timer":True,"F_or_timer_replayed":False}


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
        require(row["selected_metadata"]==cost["source_metadata"][route]==metadata[route] and row["origins"]==raw["origins"][:batch].tolist() and row["warmup_each_route"]==2 and row["native_head_loaded_or_timed"]==("_NATIVE_" in route),"EXACT_ROUTE_REFERENCE_AND_NATIVE_SCOPE")
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


def state_signature(state):
    return {"state_sha256":linear.state_hash(state),"arrays":{name:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for name,v in state.items()}}


def check_native_head(arrays,state):
    z=arrays["latent_history"]; weight,bias=state["head.weight"],state["head.bias"]
    require(z.dtype==weight.dtype==bias.dtype==np.float32 and weight.shape==(z.shape[2],48,z.shape[1]) and bias.shape==(z.shape[2],48) and np.isfinite(weight).all() and np.isfinite(bias).all(),"FINITE_INDEPENDENT_AFFINE_HEAD_SHAPES")
    future=z[:,-1:]+(np.einsum("nlk,khl->nhk",z-z[:,-1:],weight)+bias.T[None])
    prior.close(arrays["latent_future"],future,"independent_NumPy_native_affine_head")


def audit_result(path,meter):
    require(isinstance(COST_SHA,str) and len(COST_SHA)==64,"ACTUAL_COST_RESULT_NOT_YET_REGISTERED")
    check=meter.check; binding=linear.file_record(path); require(binding["sha256"]==COST_SHA,"EXACT_ACTUAL_COST_RESULT")
    cost=prior.verified_json(binding,check); require(cost["status"]=="COMPLETE" and cost["cost_split"]=="tune" and cost["routes"]==list(ROUTES) and set(cost["datasets"])==set(cost["cost"])=={"robin"} and not cost["failed_paths"],"FULL_COST_REQUIRED_FOR_PASS")
    receipt=cost["resource_terminal"]; recorded_receipt=prior.verified_json(receipt,check)
    require(recorded_receipt=={k:v for k,v in receipt.items() if k not in ("path","sha256")} and receipt["status"]=="COMPLETE" and receipt["error"] is None and receipt["gpu_process"] is True and receipt["ended_unix"]>=receipt["started_unix"] and receipt["wall_seconds"]>=0 and receipt["process_cpu_seconds"]>=0,"SAVED_COMPLETE_GPU_PROCESS_RECEIPT")
    for key in ("authorization","continuation_manifest"): bound(receipt[key],check)
    training=prior.verified_json(cost["matched_l7_native_diagnostic"],check); accepted=prior.verified_json(cost["matched_l7_native_audit"],check)
    require(cost["matched_l7_native_diagnostic"]["sha256"]==NATIVE_SHA and cost["matched_l7_native_audit"]["sha256"]==NATIVE_AUDIT_SHA and training["status"]=="COMPLETE" and accepted["status"]==accepted["training_verification_status"]=="PASS" and accepted["new_space_diagnostic_status"]=="COMPLETE","EXACT_ACCEPTED_SIX_HEAD_TRAINING")
    same_record(accepted["diagnostic"],cost["matched_l7_native_diagnostic"])
    for key in PARENTS[2:]: same_record(cost[key],training[key]); same_record(cost[key],accepted[key])
    same_record(cost["cost_plan"],training["plan"])
    full=prior.verified_json(cost["full_gradient_parent"],check); require(cost["runtime_behavior_contract"]==full["runtime_behavior_contract"],"UNCHANGED_ACTUAL_F_RUNTIME_NORMALIZATION_CONTRACT")
    science=prior.verified_json(cost["causal_lag_diagnostic"],check); subset=prior.verified_json(science["raw_subset_diagnostic"],check); original=prior.verified_json(subset["source_producer"],check)
    require(prior.file_sha(HERE/"run_matched_l7_native_cost_v1.py")==RUNNER_SHA and prior.file_sha(HERE/"whole_matched_l7_native_cost_v1.py")==HELPER_SHA and prior.file_sha(HERE/"whole_f_input_gradient_cost_v1.py")==PARENT_TIMER_SHA,"EXACT_FROZEN_COST_SOURCES")
    source=source_contract(*[(HERE/name).read_text(encoding="utf-8") for name in ("whole_matched_l7_native_cost_v1.py","run_matched_l7_native_cost_v1.py","whole_f_input_gradient_cost_v1.py")])
    require(cost["new_training_updates"]==0 and cost["native_predictor_loaded_or_timed"] is cost["selected_E_D_head_setup_outside_timer"] is cost["all_source_file_state_and_head_metadata_hashes_outside_timer"] is True,"FIXED_INFERENCE_SETUP_NO_TRAINING")
    for key in ("new_encoder_decoder_fit","new_CP_seed_or_lag_selection","new_backbone_training","cached_future_returned_instead_of_actual_F_inside_timer","protected_TEST_accessed","independent_holdout","strict_K_only"): require(cost[key] is False,"COST_SCOPE:"+key)
    guarded=[binding,*[cost[k] for k in PARENTS],cost["cost_plan"],*cost["source_snapshots"],*cost["input_records"],*cost["protected_prior_records"]]
    for item in guarded:
        bound(item,check)
        if "snapshot_path" in item: require(prior.file_sha(item["snapshot_path"])==item["sha256"],"EXECUTED_SOURCE_SNAPSHOT_IDENTITY")
    require(cost["F_before"]==cost["F_after"]==training["F_parent_signatures_reused"]["before"]==training["F_parent_signatures_reused"]["after"]==original["F_before"]==original["F_after"] and set(cost["F_before"])=={"old_algorithm","bi_algorithm"},"FROZEN_ORIGINAL_F_DUAL_SIGNATURE")
    for key in ("model_binary","model_config"):
        for suffix in ("_before","_after"): same_record(cost[key+suffix],original[key+"_before"]); bound(cost[key+suffix],check)
    registry=prior.verified_json(cost["selection_registry"],check); require(registry["status"]=="ALL_SIX_RESOLVED_BEFORE_DEV" and registry["DEV_opened_before_seal"] is False and len(registry["conditions"])==6,"SIX_NATIVE_TUNE_LOCKS_BEFORE_DEV")
    seal=registry["sealed_unix"]; locks={}
    for item in registry["conditions"]:
        route=item["state"]+"_NATIVE_s"+str(item["seed"]); require(route in ROUTES and route not in locks and item["status"]=="COMPLETE","SIX_UNIQUE_REGISTERED_HEADS")
        chosen=training["states"][item["state"]]["native"][str(item["seed"])]; same_record(item["selection_lock"],chosen["selection_lock"]); same_record(item["selection_lock"],cost["selection_locks"][route]); lock=prior.verified_json(item["selection_lock"],check)
        same_record(lock["selected_state"],chosen["selected_state"]); require(lock["status"]=="COMPLETE" and lock["selected_update"]==chosen["selected_update"] and lock["DEV_opened_before_lock"] is False,"NO_NEW_SELECTION")
        locks[route]=item["selection_lock"]; seal=max(seal,lock["sealed_unix"]); guarded.append(item["selection_lock"])
    require(set(locks)==set(cost["selection_locks"]),"EXACT_SIX_COST_SELECTION_LOCKS")
    data=cost["datasets"]["robin"]; require(set(data["physical_prechecks"])==set(ROUTES[1:]),"EXACT_NINE_PHYSICAL_BACKEND_PRECHECKS")
    raw_source=science["datasets"]["robin"]["spaces"]["PCA"]["conditions"]["L7_RECON"]["splits"]["tune"]["input_source"]; same_record(data["raw_TUNE_source"],raw_source)
    raw=prior.verified_arrays(raw_source,check); prior.check_raw_split(raw,channels=17,windows=64)
    proof=data["canonical_physical_bridge"]; require(proof["dataset"]=="robin" and proof["split"]=="tune","CANONICAL_TUNE_ONLY")
    same_record(proof["original_first_parent"],original["source_parent_results"]["robin"]); parent=prior.verified_json(proof["original_first_parent"],check); same_record(proof["canonical_input_bridge"],parent["bridge_provenance"]["input_bridge"])
    bridge=prior.verified_json(proof["canonical_input_bridge"],check); matches=[r for r in bridge["artifacts"] if r["dataset"]=="robin" and r["split"]=="tune"]; require(len(matches)==1 and proof["canonical_record"]=={**matches[0],"path":str(Path(matches[0]["path"]).resolve())},"EXACT_FULL_CANONICAL_PHYSICAL_MANIFEST")
    same_record(proof["normalized_raw_input"],original["datasets"]["robin"]["inputs"]["tune"])
    require(proof["complete_array_manifest_preserved"] is True and all(proof[k] is False for k in ("inverse_normalization_used","DEV_physical_alias_used","loader_file_SHA_member_shape_dtype_SHA_checks_relaxed","numeric_tolerance_changed")),"NO_SYNTHESIZED_PHYSICAL_OR_LOADER_RELAXATION")
    physical=prior.verified_arrays(proof["canonical_record"],check)
    for key in ("history","history_observed","target","observed","raw_last","origins","mean","std","raw_forecast"): exact(physical[key],raw[key],"full_canonical_payload_"+key)
    predictions={"RAW_BOLT":raw["raw_forecast"]}; metadata={"RAW_BOLT":{"reference":raw_source,"reference_prediction_array":"raw_forecast","reference_split":"tune","actual_F_forward":True,"matched_Bolt_route":"RAW_BOLT"}}; hashes,heads,prechecks={},{},{}
    for state in STATES:
        owner=training["states"][state]; frozen=prior.verified_arrays(owner["frozen_state_source"],check); require(set(frozen)=={"E","D","center"} and linear.state_hash(frozen)==owner["frozen_state_value_sha256"],"EXACT_MATCHED_FROZEN_E_D_CENTER")
        for kind,seed in (("BOLT",None),("NATIVE",92601),("NATIVE",92602)):
            route=state+"_"+kind+("_s"+str(seed) if seed else "")
            if seed:
                chosen=owner["native"][str(seed)]; selected=prior.verified_arrays(chosen["selected_state"],check); require(set(selected)=={"E","D","center","head.weight","head.bias"},"COMPLETE_NATIVE_INFERENCE_STATE")
                for key in frozen: exact(selected[key],frozen[key],"native_frozen_"+key)
                require(linear.state_hash(selected)==chosen["selected_state_sha256"],"SELECTED_NATIVE_FULL_STATE_HASH"); heads[route]=linear.state_hash({k[5:]:v for k,v in selected.items() if k.startswith("head.")}); require(heads[route]==chosen["selected_head_state_sha256"],"SELECTED_NATIVE_HEAD_HASH")
                cps=[r for r in chosen["checkpoints"] if r["update"]==chosen["selected_update"]]; require(len(cps)==1 and cps[0]["tune"]==chosen["selected_tune"],"SELECTED_TUNE_CHECKPOINT"); same_record(cps[0]["state"],chosen["selected_state"]); reference_record=cps[0]["artifact"]
                selection={"head_order_seed":seed,"selected_update":chosen["selected_update"],"selected_state":chosen["selected_state"],"selected_head_state_sha256":chosen["selected_head_state_sha256"],"selection_lock":locks[route]}
            else: selected=frozen; reference_record=owner["Bolt"]["tune"]["artifact"]; selection={"frozen_state_source":owner["frozen_state_source"]}
            hashes[route]=state_signature(selected); reference=prior.verified_arrays(reference_record,check)
            for key in ("target","observed","raw_last","origins","mean","std"): exact(reference[key],raw[key],"selected_TUNE_input_"+key)
            predictions[route]=reference["prediction"]; metadata[route]={"kind":kind,"frozen_state_label":state,"frozen_state_source":owner["frozen_state_source"],"lags":list(LAGS),"reference":reference_record,"reference_prediction_array":"prediction","reference_split":"tune","actual_F_forward":kind=="BOLT","actual_native_head_forward":kind=="NATIVE","matched_Bolt_route":state+"_BOLT",**selection}
            entry=data["physical_prechecks"][route]; arrays=prior.verified_arrays(entry["artifact"],check); prechecks[route]=legacy.precheck(entry,arrays,physical,raw,frozen,reference,LAGS)
            if seed: check_native_head(arrays,selected); prechecks[route]["independent_native_head_equation_checked"]=True
    require(hashes==cost["selected_states_before"]==cost["selected_states_after"] and heads==cost["native_heads_before"]==cost["native_heads_after"],"ALL_NINE_FULL_STATES_AND_SIX_HEADS_BEFORE_AFTER")
    details=check_dataset(cost["cost"]["robin"],physical,raw,predictions,metadata,check,seal)
    require(json.loads((Path(binding["path"]).parent/"cost_robin/RESULT.json").read_text(encoding="utf-8"))==cost["cost"]["robin"],"SAVED_DATASET_RESULT_IDENTITY")
    require(cost["completed_cost_cells"]==cost["expected_cost_cells"]==20 and cost["recorded_timer_samples"]==cost["expected_timer_samples"]==100 and cost["selected_adapters_loaded"]==9 and cost["selected_native_heads_loaded"]==6,"FULL_TOP_LEVEL_COMPLETION_COUNTS")
    for row in cost["cost"]["robin"]["rows"]:
        for sample in row["samples"]: require(receipt["started_unix"]*1e9<=sample["unix_start_ns"]<=sample["unix_end_ns"]<=receipt["ended_unix"]*1e9,"TIMERS_WITHIN_PROCESS_RECEIPT")
    for item in guarded: bound(item,check)
    return {"status":"PASS","cost":binding,**{k:cost[k] for k in PARENTS},"selection_locks":cost["selection_locks"],"source_contract":source,"totals":{"cost_cells_checked":20,"recorded_timer_samples_checked":100,"selected_full_state_hashes_checked":9,"selected_native_head_hashes_checked":6,"selection_locks_checked":6,"physical_prechecks_checked":9,"physical_first8_windows_checked":72,"native_first8_head_equations_checked":6,"canonical_TUNE_bridges_checked":1,"protected_prior_file_hashes_checked":len(cost["protected_prior_records"])},"datasets":{"robin":{"prechecks":prechecks,"cost":details}},"protected_TEST_accessed":False,"independent_F_or_timer_replayed":False,"per_method_exclusive_memory_claimed":False,"scope":"Saved exact TUNE cost outputs and selected states; independent NumPy E/affine-head/lag/D checks, ns medians and both Raw/matching-Bolt denominators. F/GPU/backward/optimizer/timer are not rerun; batch intermediate parity remains separate from registered output acceptance."}


def execute(path,output):
    from continuation_runtime_v1 import Meter,write_new
    output=Path(output).resolve(); require(not output.exists(),"FRESH_COST_AUDIT_OUTPUT"); meter=Meter("audit_matched_l7_native_cost_"+output.stem,gpu=False); failure=None; source=linear.file_record(__file__)
    try:
        require(isinstance(COST_SHA,str) and len(COST_SHA)==64,"ACTUAL_COST_RESULT_NOT_YET_REGISTERED"); meter.protected_check(); initialize_numpy()
        import sys
        sys.path.insert(0,str(ROOT/"research/smallk_failure_followup_20261014"))
        from prepare_inputs import install_protected_test_guard
        from threadpoolctl import threadpool_limits
        install_protected_test_guard()
        with threadpool_limits(limits=2): report=audit_result(path,meter)
        meter.protected_check(); require(prior.file_sha(__file__)==source["sha256"],"AUDITOR_SOURCE_UNCHANGED"); json.dumps(report,allow_nan=False)
    except Exception as error: failure={"type":type(error).__name__,"message":str(error)}; report={"status":"FAILED_PRESERVED","failure":failure}
    finally: report["audit_source"]=source; report["resource_terminal"]=meter.close(report["status"],failure); write_new(output,report)
    print(json.dumps({"status":report["status"],"totals":report.get("totals"),"failure":failure}),flush=True); return 0 if report["status"]=="PASS" else 2


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--cost",required=True); parser.add_argument("--output",required=True)
    args=parser.parse_args(); raise SystemExit(execute(args.cost,args.output))
