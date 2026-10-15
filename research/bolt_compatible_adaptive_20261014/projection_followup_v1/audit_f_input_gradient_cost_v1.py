"""Audit saved five-route TUNE cost; no F, backward, optimizer or latency replay."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import statistics

import audit_linear_e_cost_v1 as legacy

prior,linear,shared=legacy.prior,legacy.linear,legacy.shared
HERE,ROOT,DATASETS=legacy.HERE,legacy.ROOT,legacy.DATASETS
ROUTES=("RAW_BOLT","FULL_s92601","FULL_s92602","STOP_s92601","STOP_s92602")
COST_SHA="354cbeb47e298c1f313c08c52a61abeb19020396dfa1980e4c12b41647e2c9de"
STOP_SHA="e27989fb23716be8bbf507442d44033884b28522d641a010d18b426a0eb6d10a"
STOP_AUDIT_SHA="64f19658ac8bc8cee1d6fa3713113a7146ba282ebb8ad2b9eb49498d91e6418f"
STOP_AUDIT_COUNTS={"stop_gradient_fits_checked":4,"checkpoint_states_checked":20,"checkpoint_TUNE_predictions_checked":20,"selected_FIT_DEV_predictions_checked":8,
    "logged_updates_checked":2400,"probe_cases_checked":4,"reused_full_gradient_conditions_checked":4,"CP0_full_stop_parent_component_byte_comparisons_checked":32,
    "saved_gradient_vector_fields_checked":32,"analytic_gradient_comparisons_checked":28,"score_fields_checked":792,"selection_locks_checked":4,"protected_prior_file_hashes_checked":1485}
FULL_SHA="3f0cbd9176f2e7882778452e537564aab7445e1b88c1534fd48b1bce7ef639a4"
FULL_AUDIT_SHA="9796b4ed53f74186bc2513db56fe321c2d450cdd9bffbb7c5f341766e73ee36d"
require,exact,parity,bound,precheck=legacy.require,legacy.exact,legacy.parity,legacy.bound,legacy.precheck
np=None


def initialize_numpy():
    global np
    legacy.initialize_numpy(); np=legacy.np


def schedule():
    order=np.random.default_rng(20261015).permutation(5)
    return [[ROUTES[index] for index in np.roll(order,-repeat)] for repeat in range(5)]


def same_record(first,second):
    require(Path(first["path"]).resolve()==Path(second["path"]).resolve() and first["sha256"]==second["sha256"],"SAME_EXACT_PATH_SHA_BINDING")
    if "bytes" in first and "bytes" in second: require(first["bytes"]==second["bytes"],"SAME_OPTIONAL_FILE_BYTES")


def source_contract(helper,runner,original):
    def functions(text): return {n.name:n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
    h,old=functions(helper),functions(original)
    for name in ("fill_vectorized","preprocess","parity","whole_predict","balanced_schedule"):
        require(ast.dump(h[name])==ast.dump(old[name]),"UNCHANGED_CANONICAL_TIMER_FUNCTION:"+name)
    require('ROUTES = ("RAW_BOLT", "FULL_s92601", "FULL_s92602", "STOP_s92601", "STOP_s92602")' in helper
        and "WARMUP, REPEATS = 2, 5" in helper and "ORDER_SEED = 20261015" in helper
        and "NORM_ATOL, NORM_RTOL = 1e-5, 2e-6" in helper and "PHYSICAL_ATOL, PHYSICAL_RTOL = 2e-3, 2e-5" in helper,"FIVE_ROUTE_SCHEDULE_AND_REGISTERED_TOLERANCES")
    require("physical_precheck, adapter_predictor = old.physical_precheck, old.adapter_predictor" in runner
        and "adapter_predictor(model, bolt, engine.tensor)" in runner and "old.canonical_tune_bridge(original, data)" in runner
        and 'raw_input_sources"]["tune"]' in runner and 'reference_split": "tune"' in runner and 'selected["DEV"]' not in runner,"IMMUTABLE_ACTUAL_CALLBACK_AND_TUNE_REFERENCE")
    callback=functions((HERE/"run_linear_e_cost_v1.py").read_text(encoding="utf-8"))["adapter_predictor"]
    require("model(tensor(history), bolt).detach().cpu().numpy().copy()" in ast.unparse(callback) and "torch.no_grad()" in ast.unparse(callback),"ACTUAL_E_F_D_CPU_RETURN_NOT_CACHED_FUTURE")
    timer=[n for n in ast.walk(h["measure_cost"]) if isinstance(n,ast.Try) and len(n.body)==3 and "perf_end_ns, unix_end_ns = (time.perf_counter_ns(), time.time_ns())" in ast.unparse(n)]
    require(len(timer)==1 and [ast.unparse(n) for n in timer[0].body]==["normalized, physical_value = whole_predict(predictors[route], values, masks, mean, std)","synchronize()","perf_end_ns, unix_end_ns = (time.perf_counter_ns(), time.time_ns())"],"WHOLE_CALLBACK_AND_SYNC_INSIDE_NS_BOUNDARY")
    require(prior.file_sha(HERE/"audit_linear_e_cost_v1.py")=="26374f7f3531863450661781fd9b11bb4e019afc5394765dc4c135142272909a","IMMUTABLE_REUSED_PRECHECK_AND_AUDIT_LINEAGE")
    return {"TUNE_first_B_origins":True,"five_full_rotations":True,"actual_E_F_D_physical_CPU_return_inside_timer":True,"independent_F_or_timer_replay":False,"precheck_reused_unchanged_from_linear_E_cost":True}


def check_dataset(cost, physical, raw, predictions, metadata, check, seal):
    require(cost["routes"] == list(ROUTES) and cost["schedule"] == schedule() and cost["order_seed"] == 20261015 and cost["warmup_each_route"] == 2 and cost["repeats"] == 5 and cost["expected_cells_this_dataset"] == 10 and cost["expected_cells_two_datasets"] == 20, "FIVE_FULL_ROTATIONS_AND_ALL_DECLARED_ROWS")
    require(cost["new_training_updates"] == 0 and cost["encoder_refit_in_cost"] is cost["per_origin_decoder_fit"] is cost["fit_input_binding_hashes_inside_timer"] is cost["protected_TEST_accessed"] is cost["independent_holdout"] is False and cost["fixed_FIT_decoder_loaded_before_timer"] is True, "NO_FIT_OR_HASHES_INSIDE_TIMER")
    lineage = cost["source_lineage"]
    require(lineage["whole_cost_source_sha256"] == prior.file_sha(HERE / "whole_f_input_gradient_cost_v1.py") and Path(lineage["copied_timer_source"]).resolve() == HERE / "whole_linear_e_cost_v1.py" and lineage["copied_timer_source_sha256"] == prior.file_sha(HERE / "whole_linear_e_cost_v1.py") and lineage["copied_preprocess_source_sha256"] == prior.file_sha(HERE.parent / "cost_readout_v1.py") and lineage["copied_functions"] == ["fill_vectorized", "preprocess", "parity"] and lineage["legacy_scientific_runtime_imported"] is False, "CANONICAL_TIMER_PREPROCESS_SOURCE_LINEAGE")
    exact(prior.causal_physical_preprocess(physical["history_physical"], physical["history_observed"], raw["mean"], raw["std"]), raw["history"], "all_TUNE_physical_causal_normalization")
    for key, value in (("physical_past", physical["history_physical"]), ("history_observed", physical["history_observed"]), ("FIT_mean", raw["mean"]), ("FIT_std", raw["std"]), ("origins", raw["origins"])): require(cost["input_array_metadata"][key] == shared.array_record(value), "EXACT_COST_INPUT_ARRAY_SHA")
    rows = {(r["route"], r["batch"]): r for r in cost["rows"]}; require(set(rows) == {(r, b) for r in ROUTES for b in (1, 8)} and len(cost["rows"]) == 10, "NO_MISSING_DUPLICATE_OR_DROPPED_COST_ROWS")
    totals, chronology = {"complete_cells": 0, "saved_samples": 0}, []; details = []
    for (route, batch), row in rows.items():
        raw_row = rows["RAW_BOLT", batch]; base = {s["repeat"]: (s["perf_end_ns"] - s["perf_start_ns"]) / 1e9 for s in raw_row["samples"]}
        times = [(s["perf_end_ns"] - s["perf_start_ns"]) / 1e9 for s in row["samples"]]; ratios = [t / base[s["repeat"]] if s["repeat"] in base else None for t, s in zip(times, row["samples"])]
        require(row["seconds"] == times and row["paired_RAW_ratios"] == ratios and row["paired_RAW_seconds"] == [base.get(s["repeat"]) for s in row["samples"]] and row["repeats"] == len(times), "NS_TIMES_AND_SAME_REPEAT_PAIRED_RATIOS")
        require([s["repeat"] for s in row["samples"]] == sorted({s["repeat"] for s in row["samples"]}) and all(0 <= s["repeat"] < 5 for s in row["samples"]), "UNIQUE_REGISTERED_REPEAT_NUMBERS")
        complete = row["status"] == "COMPLETE"
        require(row["selected_metadata"] == cost["source_metadata"][route] == metadata[route] and row["origins"] == raw["origins"][:batch].tolist() and row["warmup_each_route"] == 2 and row["native_head_loaded_or_timed"] is row["per_origin_decoder_fit"] is row["declared_route_deleted"] is row["memory_counter_reset_inside_module"] is row["stage_decomposition"] is False, "EXACT_ROUTE_SELECTED_STATE_AND_TIMED_SCOPE")
        if not times: require(not complete and row["timed_outputs"] is None, "EMPTY_FAILED_PATH_PRESERVED"); continue
        median = statistics.median(times); paired = statistics.median(ratios) if all(r is not None for r in ratios) else None
        require(row["median_seconds"] == median and row["median_seconds_per_window"] == median / batch and row["paired_RAW_median_ratio"] == paired, "INDEPENDENT_MEDIANS_AND_PER_WINDOW_ARITHMETIC")
        if complete: require(len(times) == 5 and row["failure"] is None, "COMPLETE_ROW_ALL_FIVE_SAMPLES")
        refs, stack = (prior.verified_arrays(row[key], check) for key in ("reference", "timed_outputs"))
        require(all(refs[key].dtype == np.float32 and refs[key].shape == (batch, 48, len(raw["mean"])) for key in ("normalized", "physical", "scientific_normalized")) and all(stack[key].dtype == np.float32 and stack[key].shape == (len(times), batch, 48, len(raw["mean"])) for key in ("normalized", "physical")), "FP32_REFERENCE_AND_ALL_TIMER_OUTPUT_SHAPES")
        for arrays in (refs, stack): exact(arrays["origins"], raw["origins"][:batch], "reference_stack_origins")
        exact(refs["scientific_normalized"], predictions[route][:batch], "exact_selected_TUNE_reference")
        exact(refs["physical"], (refs["normalized"] * raw["std"] + raw["mean"]).astype(np.float32), "reference_CPU_physical_inverse")
        prior.close(refs["normalized"], predictions[route][:batch], "reference_B_batch_vs_selected_canonical32")
        require(len(row["samples"]) == len(row["parity"]) == len(row["memory"]), "ALL_SAVED_SAMPLE_LIST_LENGTHS")
        for i, sample in enumerate(row["samples"]):
            require(sample["dataset"] == cost["dataset"] and sample["route"] == route and sample["batch"] == batch and sample["repeat"] == sample["block"] and sample["position"] == row["positions"][i] == schedule()[sample["repeat"]].index(route) and sample["seconds"] == times[i] > 0 and sample["unix_end_ns"] >= sample["unix_start_ns"] > seal * 1e9, "SAMPLE_IDENTITY_NS_AFTER_LOCKS_AND_ROTATION")
            require(sample["parity"] == row["parity"][i] and sample["memory"] == row["memory"][i], "ROW_SAMPLE_PARITY_MEMORY_BINDING")
            saved = prior.verified_arrays(sample["sample_artifact"], check)
            literal = json.loads(Path(sample["sample_artifact"]["path"]).with_suffix(".json").read_text(encoding="utf-8")); require(literal == sample, "EXACT_SAVED_SAMPLE_JSON")
            for key in ("normalized", "physical"): exact(saved[key], stack[key][i], "saved_sample_stack_member")
            exact(saved["origins"], raw["origins"][:batch], "same_first_B_TUNE_origins"); exact(saved["physical"], (saved["normalized"] * raw["std"] + raw["mean"]).astype(np.float32), "timed_CPU_physical_inverse")
            for key, expected, physical_units in (("normalized", refs["normalized"], False), ("physical", refs["physical"], True), ("scientific_normalized", predictions[route][:batch], False), ("scientific_physical", (predictions[route][:batch] * raw["std"] + raw["mean"]).astype(np.float32), True)): parity(sample["parity"][key], saved["physical" if physical_units else "normalized"], expected, physical=physical_units, must_pass=complete)
            memory = sample["memory"]; require(memory["scope"] == "Shared resident frozen F and all available selected adapters; no exclusive per-method allocation" and all(type(v) is int and v >= 0 for k, v in memory.items() if k != "scope") and memory["max_reserved_VRAM_bytes"] >= memory["reserved_VRAM_bytes"] >= memory["allocated_VRAM_bytes"] and memory["max_reserved_VRAM_bytes"] >= memory["max_allocated_VRAM_bytes"] >= memory["allocated_VRAM_bytes"], "SHARED_PROCESS_MEMORY_COUNTER_SCOPE")
            chronology.append((batch, sample["repeat"], sample["position"], sample))
        exact(stack["block"], np.asarray([s["repeat"] for s in row["samples"]], np.int64), "timer_stack_repeats"); exact(stack["position"], np.asarray(row["positions"], np.int64), "timer_stack_positions")
        totals["complete_cells"] += int(complete); totals["saved_samples"] += len(times)
        details.append({"route": route, "batch": batch, "samples": len(times), "median_seconds": median, "median_seconds_per_window": median / batch, "ratio_of_RAW_medians": median / statistics.median(base.values()), "median_paired_RAW_ratio": paired, "status": row["status"]})
    chronology.sort(key=lambda r: r[:3])
    for a, b in zip(chronology, chronology[1:]): require(a[3]["perf_end_ns"] <= b[3]["perf_start_ns"] and all(a[3]["memory"][k] <= b[3]["memory"][k] for k in ("max_allocated_VRAM_bytes", "max_reserved_VRAM_bytes")), "SERIAL_NS_AND_UNRESET_SHARED_MEMORY_PEAKS")
    require(cost["completed_cost_cells"] == totals["complete_cells"] and cost["recorded_timer_samples"] == totals["saved_samples"] and cost["attempted_timer_calls"] >= totals["saved_samples"], "ACTUAL_RECORDED_COUNTS")
    if cost["status"] == "COMPLETE": require(totals == {"complete_cells": 10, "saved_samples": 50} and not cost["failed_paths"], "COMPLETE_DATASET_10_50")
    return {**totals, "rows": details, "independent_F_or_timer_replayed": False}


def audit_result(path,meter):
    require(isinstance(COST_SHA,str) and len(COST_SHA)==64 and isinstance(STOP_AUDIT_SHA,str) and len(STOP_AUDIT_SHA)==64 and bool(STOP_AUDIT_COUNTS),"ACTUAL_COST_RESULT_NOT_YET_REGISTERED")
    check=meter.check; binding=linear.file_record(path); require(binding["sha256"]==COST_SHA,"EXACT_ACTUAL_COST_RESULT_SHA")
    cost=prior.verified_json(binding,check)
    require(cost["status"] in ("COMPLETE","PARTIAL_COMPLETED") and cost["cost_split"]=="tune" and cost["expected_cost_cells"]==20 and cost["expected_timer_samples"]==100,"COMPLETED_TUNE_COST_SCOPE")
    stopped=prior.verified_json(cost["stop_gradient_diagnostic"],check); accepted=prior.verified_json(cost["stop_gradient_audit"],check)
    full=prior.verified_json(cost["full_gradient_parent"],check); full_audit=prior.verified_json(cost["full_gradient_parent_audit"],check)
    require([cost[k]["sha256"] for k in ("stop_gradient_diagnostic","stop_gradient_audit","full_gradient_parent","full_gradient_parent_audit")]==[STOP_SHA,STOP_AUDIT_SHA,FULL_SHA,FULL_AUDIT_SHA]
        and stopped["status"]==full["status"]=="COMPLETE" and accepted["status"]==full_audit["status"]=="PASS","EXACT_COMPLETE_STOP_AND_REUSED_FULL_PARENTS")
    same_record(accepted["diagnostic"],cost["stop_gradient_diagnostic"]); same_record(full_audit["diagnostic"],cost["full_gradient_parent"])
    require(all(accepted["totals"].get(k)==v for k,v in STOP_AUDIT_COUNTS.items()),"FULL_INDEPENDENT_STOP_GRADIENT_COVERAGE")
    for key in ("full_gradient_parent","full_gradient_parent_audit","parent_report","parent_report_manifest","parent_report_audit","causal_lag_diagnostic","causal_lag_audit","selection_registry"):
        same_record(cost[key],stopped[key]); same_record(cost[key],accepted[key])
    same_record(cost["full_gradient_selection_registry"],full["selection_registry"])
    science=prior.verified_json(cost["causal_lag_diagnostic"],check); subset=prior.verified_json(science["raw_subset_diagnostic"],check); original=prior.verified_json(subset["source_producer"],check)
    source=source_contract(*[(HERE/name).read_text(encoding="utf-8-sig") for name in ("whole_f_input_gradient_cost_v1.py","run_f_input_gradient_cost_v1.py","whole_linear_e_cost_v1.py")])
    require(cost["new_training_updates"]==0 and cost["selected_E_D_center_loaded_outside_timer"] is cost["all_source_file_state_metadata_hashes_outside_timer"] is cost["training_detach_does_not_change_deployment_formula"] is True,"FIXED_SELECTED_STATES_OUTSIDE_TIMER")
    for key in ("new_backbone_training","new_encoder_decoder_fit","new_CP_or_seed_or_lag_selection","native_predictor_loaded_or_timed","per_origin_decoder_fit","cached_future_returned_instead_of_actual_F_inside_timer","protected_TEST_accessed","independent_holdout","strict_K_only"): require(cost[key] is False,"FIXED_COST_SCOPE:"+key)
    guarded=[binding,*[cost[k] for k in ("stop_gradient_diagnostic","stop_gradient_audit","full_gradient_parent","full_gradient_parent_audit","selection_registry","full_gradient_selection_registry","cost_plan")],*cost["source_snapshots"],*cost["source_records"],*cost["input_records"],*cost["protected_prior_records"]]
    for item in guarded:
        bound(item,check)
        if "snapshot_path" in item: require(prior.file_sha(item["snapshot_path"])==item["sha256"],"SOURCE_SNAPSHOT_UNCHANGED")
    require(cost["F_before"]==cost["F_after"]==stopped["F_before"]==stopped["F_after"]==full["F_before"]==full["F_after"]==original["F_before"] and set(cost["F_before"])=={"old_algorithm","bi_algorithm"},"UNCHANGED_ORIGINAL_F_DUAL_SIGNATURE")
    for key in ("model_binary","model_config"):
        for suffix in ("_before","_after"): same_record(cost[key+suffix],stopped[key+"_before"]); bound(cost[key+suffix],check)
    require(cost["runtime_behavior_contract"]==stopped["runtime_behavior_contract"]==full["runtime_behavior_contract"],"SAME_F_RUNTIME_NORMALIZATION_CONTRACT")
    registries={mode:prior.verified_json(cost[key],check) for mode,key in (("FULL","full_gradient_selection_registry"),("STOP","selection_registry"))}
    require(registries["FULL"]["status"]=="ALL_16_RESOLVED_BEFORE_DEV" and len(registries["FULL"]["conditions"])==16 and registries["STOP"]["status"]=="ALL_FOUR_RESOLVED_BEFORE_DEV" and len(registries["STOP"]["conditions"])==4,"PRESERVED_FULL16_AND_NEW_STOP4_SELECTION_REGISTRIES")
    rows={}; seal=max(v["sealed_unix"] for v in registries.values())
    for data in DATASETS:
        for mode,parent in (("FULL",full),("STOP",stopped)):
            for seed in (92601,92602):
                route=mode+"_s"+str(seed); chosen=parent["datasets"][data]["conditions"][data+"_L0_E_LEARN_s"+str(seed) if mode=="FULL" else str(seed)]
                matches=[r for r in registries[mode]["conditions"] if r["dataset"]==data and r["seed"]==seed and (mode=="STOP" or (r["lag"]=="L0" and r["learn_E"] is True))]
                require(len(matches)==1,"EXACT_MATCHED_FULL_OR_STOP_LOCK"); ref=matches[0]["selection_lock"]; same_record(ref,chosen["selection_lock"]); same_record(ref,cost["selection_locks"][data+"_"+route])
                lock=prior.verified_json(ref,check); same_record(lock["selected_state"],chosen["selected_state"])
                require(lock["status"]=="COMPLETE" and lock["selected_update"]==chosen["selected_update"] and lock["DEV_opened_before_lock"] is False,"TUNE_LOCK_NO_RESELECTION")
                rows[data,route]=(chosen,ref,"FULL_REUSED" if mode=="FULL" else "STOP_NEW"); guarded.append(ref); seal=max(seal,lock["sealed_unix"])
    require(set(cost["selection_locks"])=={data+"_"+route for data,route in rows},"EXACT_EIGHT_COST_MODEL_LOCKS")
    report={"status":"PASS" if cost["status"]=="COMPLETE" else "PARTIAL_AUDITED_NO_FULL_PASS","cost":binding,
        **{k:cost[k] for k in ("stop_gradient_diagnostic","stop_gradient_audit","full_gradient_parent","full_gradient_parent_audit","parent_report","parent_report_manifest","parent_report_audit","causal_lag_diagnostic","causal_lag_audit","selection_registry","full_gradient_selection_registry","selection_locks")},
        "source_contract":source,"totals":{"cost_cells_checked":0,"recorded_timer_samples_checked":0,"selected_state_hashes_checked":0,"selection_locks_checked":8,"physical_prechecks_checked":0,"physical_first8_windows_checked":0,"canonical_TUNE_bridges_checked":0},
        "datasets":{},"protected_TEST_accessed":False,"independent_F_or_timer_replayed":False,"per_method_exclusive_memory_claimed":False,
        "scope":"Saved TUNE whole-cost outputs; immutable original NumPy E/lag/D and physical-precheck arithmetic; same-reference ns schedules/two ratios. F/backward/optimizer/timer are not rerun."}
    hashes={}
    for data,(c,k,_,_) in DATASETS.items():
        row=cost["datasets"][data]; raw=prior.verified_arrays(subset["datasets"][data]["raw_input_sources"]["tune"],check); prior.check_raw_split(raw,channels=c,windows=64)
        proof=row["canonical_physical_bridge"]; require(proof["split"]=="tune" and proof["dataset"]==data and proof["original_first_parent"]==original["source_parent_results"][data],"CANONICAL_TUNE_NOT_DEV_ALIAS")
        parent=prior.verified_json(proof["original_first_parent"],check); require(proof["canonical_input_bridge"]==parent["bridge_provenance"]["input_bridge"],"ORIGINAL_INPUT_BRIDGE_BINDING")
        bridge=prior.verified_json(proof["canonical_input_bridge"],check); candidates=[a for a in bridge["artifacts"] if a["dataset"]==data and a["split"]=="tune"]
        require(len(candidates)==1 and proof["canonical_record"]=={**candidates[0],"path":str(Path(candidates[0]["path"]).resolve())} and proof["normalized_raw_input"]==original["datasets"][data]["inputs"]["tune"] and proof["complete_array_manifest_preserved"] is True
            and proof["inverse_normalization_used"] is proof["DEV_physical_alias_used"] is proof["loader_file_SHA_member_shape_dtype_SHA_checks_relaxed"] is proof["numeric_tolerance_changed"] is False,"FULL_EXISTING_TUNE_PHYSICAL_RECORD_NO_RELAXATION")
        physical=prior.verified_arrays(proof["canonical_record"],check)
        for field in ("history_observed","origins","mean","std"): exact(physical[field],raw[field],"same_TUNE_bridge_"+field)
        report["totals"]["canonical_TUNE_bridges_checked"]+=1; predictions={"RAW_BOLT":raw["raw_forecast"]}; metadata={"RAW_BOLT":{"source":subset["datasets"][data]["raw_input_sources"]["tune"],"scope":"same-cohort actual full-C frozen Bolt"}}; checks={}
        for route in ROUTES[1:]:
            chosen,lock,origin=rows[data,route]; state=prior.verified_arrays(chosen["selected_state"],check); digest=linear.state_hash(state)
            require(digest==chosen["selected_state_sha256"],"SELECTED_E_D_CENTER_STATE_HASH"); hashes[str((data,route))]={"state_sha256":digest,"arrays":{name:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for name,v in state.items()}}
            cps=[p for p in chosen["checkpoints"] if p["update"]==chosen["selected_update"]]; require(len(cps)==1 and cps[0]["state"]==chosen["selected_state"] and cps[0]["tune"]==chosen["selected_tune"],"SAME_SELECTED_TUNE_CP_REFERENCE")
            reference=prior.verified_arrays(cps[0]["artifact"],check); predictions[route]=reference["prediction"]
            metadata[route]={"origin":origin,"seed":chosen["seed"],"lag":"L0","selected_update":chosen["selected_update"],"selected_state":chosen["selected_state"],"selection_lock":lock,"reference":cps[0]["artifact"],"reference_split":"tune","reference_prediction_array":"prediction","actual_F_forward":True,"current_C_anchor":True,"strict_K_only":False}
            entry=row["physical_prechecks"][route]; checks[route]=precheck(entry,prior.verified_arrays(entry["artifact"],check),physical,raw,state,reference,(0,))
            report["totals"]["selected_state_hashes_checked"]+=1; report["totals"]["physical_prechecks_checked"]+=1; report["totals"]["physical_first8_windows_checked"]+=8
        details=check_dataset(cost["cost"][data],physical,raw,predictions,metadata,check,seal)
        require(json.loads((Path(binding["path"]).parent/("cost_"+data)/"RESULT.json").read_text(encoding="utf-8"))==cost["cost"][data],"SAVED_DATASET_COST_JSON_IDENTITY")
        report["datasets"][data]={"prechecks":checks,"cost":details}; report["totals"]["cost_cells_checked"]+=details["complete_cells"]; report["totals"]["recorded_timer_samples_checked"]+=details["saved_samples"]
    require(hashes==cost["selected_states_before"]==cost["selected_states_after"],"ALL_EIGHT_SELECTED_STATE_HASHES_BEFORE_AFTER")
    for item in guarded: bound(item,check)
    if cost["status"]=="COMPLETE": require(report["totals"]=={"cost_cells_checked":20,"recorded_timer_samples_checked":100,"selected_state_hashes_checked":8,"selection_locks_checked":8,"physical_prechecks_checked":8,"physical_first8_windows_checked":64,"canonical_TUNE_bridges_checked":2} and cost["selected_adapters_loaded"]==8 and cost["completed_cost_cells"]==20 and cost["recorded_timer_samples"]==100 and not cost["failed_paths"],"FULL_CURRENT_20_100_TUNE_COST_COVERAGE")
    return report


def execute(path,output):
    from continuation_runtime_v1 import Meter,write_new
    output=Path(output).resolve(); require(not output.exists(),"FRESH_COST_AUDIT_OUTPUT")
    meter=Meter("audit_f_input_gradient_cost_"+output.stem,gpu=False); failure=None; source=linear.file_record(__file__)
    try:
        require(isinstance(COST_SHA,str) and len(COST_SHA)==64,"ACTUAL_COST_RESULT_NOT_YET_REGISTERED"); meter.protected_check(); initialize_numpy()
        import sys
        sys.path.insert(0,str(ROOT/"research/smallk_failure_followup_20261014"))
        from prepare_inputs import install_protected_test_guard
        from threadpoolctl import threadpool_limits
        install_protected_test_guard()
        with threadpool_limits(limits=2): report=audit_result(path,meter)
        meter.protected_check(); require(prior.file_sha(__file__)==source["sha256"],"AUDIT_SOURCE_UNCHANGED"); json.dumps(report,allow_nan=False)
    except Exception as error: failure={"type":type(error).__name__,"message":str(error)}; report={"status":"FAILED_PRESERVED","failure":failure}
    finally:
        report["audit_source"]=source; report["resource_terminal"]=meter.close(report["status"],failure); write_new(output,report)
    print(json.dumps({"status":report["status"],"totals":report.get("totals"),"failure":failure}),flush=True); return 0 if report["status"]=="PASS" else 2


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--cost",required=True); parser.add_argument("--output",required=True)
    args=parser.parse_args(); raise SystemExit(execute(args.cost,args.output))
