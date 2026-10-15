"""JSON-metadata-only PCA/cross-moment, native, direct ridge and whole-cost comparison."""
from pathlib import Path
import argparse
import collections
import json
import math
import statistics

from make_linear_e_adaptation_report_v2 import require,resolve,sha,record,read,bound,same_opaque,choose,rel,write_csv

HERE = Path(__file__).resolve().parent
DATASETS = {"robin":(17,4,"val"),"beijing_aotizhongxin":(11,2,"eval")}
SEEDS,ALPHAS,SCALES = (92601,92602),(0.,1e-4,.001,.01,.1,1.,10.),("normalized","physical_per_channel_diagnostic")
TABLES = ("SCORES","DIRECTION_EFFECTS","NATIVE_CHECKPOINTS","RIDGE_CANDIDATES","SELECTIONS","ORACLE_CONTEXT","SOLVERS","WHOLE_COST","WHOLE_COST_SAMPLES","ACCURACY_COST_REFERENCE","RESOURCES","FAILURES")
FIXED_SHA = {"cross":"99bffd5831143ca35947099fef9567f89dd209d83451661decee0682b2b8e13c","cross_audit":"90c68a9057909fb5b6c23284e5a45523cdf126da178a5ba0d9d9e96a50dede30","native":"786edbd75a35d2c5a760a7b4f5af5a1389ee1bc4bc93594d7b359a29ecfd83ed","native_audit":"51e438d7544bb5c52d2a6ea3ddd2da9c38773b2aa42286cf3a77c67d405d7bf6","ridge":"b8d7eae62ce5bb52b3d46ffc543f916f8b81857028098370a5b2f8398510f1ed","ridge_audit":"8bfeb29cc6836d2ff5528c30ee13a4300b79cc4c250a452b9c5c81fdfc9bac90","cost":"8a1876aeb1490a53a4872c294a9ec2e1a4f296a1a32ff8787b084b0b98d6a1fb","cost_audit":"e2f3ad6f9ad3f6cd3f1344aacfc7829c7818e23842e1d60061b14cc6833f3491"}
AUDIT_SOURCES = {"cross_audit":{"path":str(HERE/"audit_predictive_cross_moment_v1.py"),"sha256":"6577d15a200728dc585bf2c8e93ccff6baabd1ba916384dfb2cce7aa0014a731"},"native_audit":{"path":str(HERE/"audit_new_direction_native_v1.py"),"sha256":"6654e9dc8680c827ca9626aa6dd97cfe9d199a4c71017b239b7d6cd8e4f78fb5"},"ridge_audit":{"path":str(HERE/"audit_new_direction_direct_ridge_v1.py"),"sha256":"d8b6e8b8ed6862af69e78ecc63758b13fd03378862aa6c6c6644052fa36d85d4"},"cost_audit":{"path":str(HERE/"audit_new_direction_cost_v1.py"),"sha256":"6d5abc6d85a64d618bd2871b237455294bfc7c666aa8a7caedf88701290e334f"}}
COUNTS = {"cross_audit":{"states_checked":8,"direction_fits_checked":8,"decoder_fits_checked":8,"decoder_response_columns_checked":112,"FIT_TUNE_cells_checked":16,"DEV_cells_checked":8,"independent_saved_F_arithmetic_cells_checked":24,"oracle_dependency_masks_checked":24,"state_locks_checked":8},"native_audit":{"native_fits_checked":16,"logged_updates_checked":9600,"checkpoint_states_checked":80,"checkpoint_TUNE_predictions_checked":80,"selected_FIT_predictions_checked":16,"selected_TUNE_entries_reused_checked":16,"selected_DEV_predictions_checked":16,"CP0_probe_cases_checked":16,"cached_actual_F_context_lineages_checked":24,"independent_NumPy_head_and_old_D_cells_checked":112,"produced_NPZ_artifacts_checked":320,"training_score_fields_checked":8064,"baseline_score_fields_checked":1152,"protected_prior_file_hashes_checked":2017},"ridge_audit":{"selected_ridge_fits_checked":12,"mask_groups_checked":1319,"alpha_candidates_checked":84,"FIT_TUNE_prediction_evaluations_checked":168,"selected_DEV_prediction_evaluations_checked":12,"selected_coefficient_FP32_arithmetic_predictions_checked":36,"selection_locks_checked":12,"score_fields_checked":4320,"selected_upper_grid_boundary_conditions":10,"protected_source_input_and_result_hashes_checked":2155},"cost_audit":{"cost_cells_checked":76,"recorded_timer_samples_checked":380,"selected_full_state_hashes_checked":24,"selected_native_head_hashes_checked":16,"selected_ridge_coefficient_hashes_checked":12,"native_selection_locks_checked":16,"ridge_selection_locks_checked":12,"physical_prechecks_checked":36,"physical_first8_windows_checked":288,"independent_native_head_equations_checked":16,"independent_ridge_FP32_predictions_checked":12,"canonical_TUNE_bridges_checked":2,"protected_prior_file_hashes_checked":2513}}
PRODUCER_SOURCES = {"cross":{"path":str(HERE/"run_predictive_cross_moment_v1.py"),"sha256":"e5b971277673118454d30f13819a7013c9c0b0ef1c6e682c18d5fab971ef1f50"},"native":{"path":str(HERE/"run_new_direction_native_v1.py"),"sha256":"4aaf63c4ae078daed71f4b11540d4ade8ee62ffce6a8345bd031fe162b9062e4"},"ridge":{"path":str(HERE/"run_new_direction_direct_ridge_v1.py"),"sha256":"7dd19e1504f0ff4b323c52412fac492f2b5c17c23e1ba0d898691a672f5ac88e"},"cost":{"path":str(HERE/"run_new_direction_cost_v1.py"),"sha256":"4c721b0b87d9252fa55e90b68e03a35e9b18e15f61dfc9d1b14e4289058f5bfa"}}
WHOLE_COST_SOURCE = {"path":str(HERE/"whole_new_direction_cost_v1.py"),"sha256":"49a094ddd8b40d79932ef6c0dc87ce37548abd8159c0f2b1cae8cb0b617a1ae9"}
UTILITY = {"path":str(HERE/"make_linear_e_adaptation_report_v2.py"),"sha256":"32e2460ea4724144067f264ab1ceb0332cd04c5fce2296838024c119c9572466"}


def state_names(data):
    k = DATASETS[data][1]; return tuple(kind+"_K"+str(width) for kind in ("PCA","CROSS_MOMENT") for width in (k,2*k))


def ridge_routes(data): return ("RAW_CURRENT_C","RAW_HISTORY512")+tuple(name+"_HISTORY512" for name in state_names(data))


def routes(data): return ("RAW_BOLT",)+tuple(route for name in state_names(data) for route in (name+"_BOLT",name+"_NATIVE_s92601",name+"_NATIVE_s92602"))+("RAW_CURRENT_C_RIDGE","RAW_HISTORY512_RIDGE")+tuple(name+"_RIDGE_HISTORY512" for name in state_names(data))


def matched_bolt(route):
    if route.startswith("RAW_"): return "RAW_BOLT"
    for marker in ("_NATIVE_","_RIDGE_"):
        if marker in route: return route.split(marker)[0]+"_BOLT"
    return route


def ridge_cost_route(route): return route+"_RIDGE" if route.startswith("RAW_") else route.removesuffix("_HISTORY512")+"_RIDGE_HISTORY512"


def load_verified(paths):
    require(set(paths) == set(FIXED_SHA) and all(isinstance(v,str) and len(v) == 64 for v in FIXED_SHA.values()) and all(isinstance(v,dict) for v in AUDIT_SOURCES.values()) and all(isinstance(v,dict) and v for v in COUNTS.values()) and all(isinstance(v,dict) for v in PRODUCER_SOURCES.values()),"NEW_SCIENCE_AND_WHOLE_COST_INDEPENDENT_PASSES_NOT_YET_REGISTERED")
    bound(UTILITY); values = {key:read(path) for key,path in paths.items()}; bindings = {key:record(path) for key,path in paths.items()}
    for key,value in values.items():
        require(bindings[key]["sha256"] == FIXED_SHA[key] and value["status"] == ("PASS" if key.endswith("audit") else "COMPLETE") and value["protected_TEST_accessed"] is False,"EXACT_COMPLETE_PASS_METADATA:"+key)
        receipt=value["resource_terminal"]; require(all(type(receipt[field]) in (int,float) and math.isfinite(receipt[field]) and receipt[field] >= 0 for field in ("process_cpu_seconds","wall_seconds")) and receipt["status"] == value["status"] and receipt["error"] is None,"CANONICAL_COMPLETE_RESOURCE_RECEIPT:"+key)
        require(read(bound(receipt)) == {k:v for k,v in receipt.items() if k not in ("path","sha256")},"EXACT_RESOURCE_FILE_AND_EMBEDDED_CONTENT:"+key)
    for study in ("cross","native","ridge","cost"):
        audit = values[study+"_audit"]; field = "cost" if study == "cost" else "diagnostic"
        bound(audit[field],bindings[study]); bound(audit["audit_source"],AUDIT_SOURCES[study+"_audit"]); bound(values[study]["controller_source"],PRODUCER_SOURCES[study]); require(not values[study]["failed_paths"] and not audit.get("partial_paths",[]) and all(type(audit["totals"].get(k)) is int and audit["totals"][k] == v for k,v in COUNTS[study+"_audit"].items()),"FULL_REGISTERED_AUDIT_COVERAGE:"+study)
        bindings[study+"_auditor_source"] = audit["audit_source"]; bindings[study+"_controller_source"] = values[study]["controller_source"]
    cross,native,ridge,cost = (values[key] for key in ("cross","native","ridge","cost"))
    for value,first,second in ((native,"predictive_cross_diagnostic","predictive_cross_audit"),(ridge,"cross_moment_diagnostic","cross_moment_audit"),(cost,"predictive_cross_diagnostic","predictive_cross_audit")):
        bound(value[first],bindings["cross"]); bound(value[second],bindings["cross_audit"])
    for key,stage in (("new_direction_native_diagnostic","native"),("new_direction_native_audit","native_audit"),("new_direction_ridge_diagnostic","ridge"),("new_direction_ridge_audit","ridge_audit")):
        for value in (cost,values["cost_audit"]): bound(value[key],bindings[stage])
    for study,field in (("native","native_selection_registry"),("ridge","ridge_selection_registry")):
        bound(cost[field],values[study]["selection_registry"]); bound(values[study+"_audit"]["selection_registry"],values[study]["selection_registry"])
    require(cross["direction_fits_completed"] == cross["decoder_fits_completed"] == 8 and cross["FIT_TUNE_evaluations_completed"] == 16 and cross["DEV_evaluations_completed"] == 8 and cross["all_8_locks_before_DEV"] is True and cross["new_F_forward_calls"] == 104,"EIGHT_NEW_FIXED_SPACES_AND_ACTUAL_F_CACHES")
    require(native["native_training_fits"] == native["resolved_selection_locks"] == 16 and native["training_updates_completed"] == 9600 and native["checkpoint_states_saved"] == 80 and native["DEV_evaluations_completed"] == 16 and native["all_16_locks_before_DEV"] is True,"SIXTEEN_HEADS_EIGHTY_CP_AND_LOCKED_DEV")
    require(ridge["ridge_fits_completed"] == ridge["resolved_selection_locks"] == 12 and ridge["alpha_candidates_completed"] == 84 and ridge["FIT_TUNE_prediction_evaluations_completed"] == 168 and ridge["selected_DEV_evaluations_completed"] == 12 and ridge["all_12_locks_before_DEV"] is True and ridge["D_bypassed_by_direct_C_readouts"] is True and ridge["same_D_native_or_capacity_match_claimed"] is False,"TWELVE_DIRECT_C_RIDGES_FIXED_SEVEN_ALPHA_MENU")
    for value in (native,ridge): require(value["new_F_forward_calls"] == value["new_F_model_objects_loaded"] == 0 and value["whole_deployment_cost_measured"] is False and value["F_parent_signatures_reused"] == {"before":cross["F_before"],"after":cross["F_after"]},"SCIENCE_CACHE_REUSE_IS_NOT_COST_REPLAY")
    require(cross["F_before"] == cross["F_after"] == cost["F_before"] == cost["F_after"] and cost["runtime_behavior_contract"] == cross["runtime_behavior_contract"],"SAME_FROZEN_MEDIAN_F_ALL_STAGES")
    require(cost["completed_cost_cells"] == cost["expected_cost_cells"] == 76 and cost["recorded_timer_samples"] == cost["expected_timer_samples"] == 380 and cost["physical_prechecks_completed"] == 36 and cost["new_training_updates"] == 0 and cost["new_encoder_decoder_fit"] is cost["new_alpha_or_seed_selection"] is cost["cached_future_returned_instead_of_actual_F_inside_timer"] is False,"FULL_LIVE_WHOLE_COST_NOT_CACHED_RETURN")
    for key in ("selected_states","native_heads","ridge_coefficients"): require(cost[key+"_before"] == cost[key+"_after"],"FROZEN_COST_STATES_AND_COEFFICIENTS")
    for study in ("cross","native","ridge"):
        for key in ("controller_source","plan"):
            bound(values[study][key]); bindings[study+"_"+key] = values[study][key]
    for key in ("whole_cost_source","cost_plan"):
        bound(cost[key]); bindings[key] = cost[key]
    bound(cost["whole_cost_source"],WHOLE_COST_SOURCE)
    return values,bindings


def native_selected(row):
    require([cp["update"] for cp in row["checkpoints"]] == [0,100,200,400,600],"FIVE_PREDECLARED_NATIVE_CP")
    selected = min(row["checkpoints"],key=lambda cp:(cp["tune"]["mse"],cp["update"]))
    require(row["selected_update"] == selected["update"] and row["selected_tune"] == selected["tune"],"OWN_TUNE_EXACT_EARLIER_NATIVE_CP"); same_opaque(row["selected_state"],selected["state"]); return selected


def ridge_selected(row):
    require([p["alpha"] for p in row["candidates"]] == list(ALPHAS),"EXACT_V1_ALPHA_GRID")
    valid = [p for p in row["candidates"] if p["status"] == "COMPLETE" and math.isfinite(p["TUNE_MSE"])]; require(bool(valid),"VALID_RIDGE_CANDIDATE")
    winner = min(valid,key=lambda p:(p["TUNE_MSE"],ALPHAS.index(p["alpha"])))
    require(row["selected_alpha"] == winner["alpha"] and row["selected_TUNE_MSE"] == winner["TUNE_MSE"] and row["FIT"] == winner["FIT"] and row["TUNE"] == winner["TUNE"],"OWN_ACTUAL_FP32_TUNE_ALPHA_AND_REUSE"); same_opaque(row["selected_FIT_TUNE_prediction_artifact"],winner["prediction_artifact"]); return winner


def checked_cost_rows(cost):
    rows = []; samples = []
    for data in DATASETS:
        group = cost["cost"][data]; require(tuple(group["routes"]) == routes(data) and len(group["rows"]) == 38,"ALL_DECLARED_COST_ROUTES_PRESERVED")
        index = {(row["route"],row["batch"]):row for row in group["rows"]}; require(len(index) == 38,"UNIQUE_COST_CELLS")
        seconds = {}
        for (route,batch),row in index.items():
            require(route in routes(data) and batch in (1,8) and row["declared_route_deleted"] is False,"REGISTERED_ROUTE_AND_BATCH")
            measured = {}; order = []
            for sample in row["samples"]:
                require(sample["dataset"] == data and sample["route"] == route and sample["batch"] == batch and type(sample["repeat"]) is int and sample["repeat"] not in measured and sample["block"] == sample["repeat"] and all(type(sample[key]) is int for key in ("perf_start_ns","perf_end_ns","unix_start_ns","unix_end_ns")) and sample["perf_end_ns"] > sample["perf_start_ns"],"INTEGER_TIMER_PRIMITIVES_AND_UNIQUE_REPEAT")
                value = (sample["perf_end_ns"]-sample["perf_start_ns"])/1e9; require(value == sample["seconds"] and group["schedule"][sample["repeat"]][sample["position"]] == route,"SAMPLE_INTEGER_NS_SECONDS_AND_SCHEDULE")
                measured[sample["repeat"]] = value; order.append(value)
            require(row["seconds"] == order,"ROW_SECONDS_FROM_TIMER_PRIMITIVES")
            if row["status"] == "COMPLETE": require(set(measured) == set(range(5)) and all(p["pass"] for sample in row["samples"] for p in sample["parity"].values()),"FIVE_COMPLETE_PARITY_CHECKED_TIMERS")
            seconds[route,batch] = measured
        for batch in (1,8):
            for route in routes(data):
                row = index[route,batch]; actual = seconds[route,batch]; matched = matched_bolt(route); raw = seconds["RAW_BOLT",batch]; peer = seconds[matched,batch]; keys = [p["repeat"] for p in row["samples"]]; values = [actual[k] for k in keys]
                raw_ratios = [actual[k]/raw[k] if raw.get(k,0) > 0 else None for k in keys]; peer_ratios = [actual[k]/peer[k] if peer.get(k,0) > 0 else None for k in keys]
                med = statistics.median(values) if values else None
                derived = {"median_seconds":med,"median_seconds_per_window":None if med is None else med/batch,"paired_RAW_median_ratio":statistics.median(raw_ratios) if raw_ratios and None not in raw_ratios else None,"ratio_of_RAW_medians":med/statistics.median(raw.values()) if med is not None and raw else None,"paired_matched_Bolt_median_ratio":statistics.median(peer_ratios) if peer_ratios and None not in peer_ratios else None,"ratio_of_matched_Bolt_medians":med/statistics.median(peer.values()) if med is not None and peer else None}
                require(row["matched_Bolt_route"] == matched and row["paired_RAW_ratios"] == raw_ratios and row["paired_matched_Bolt_ratios"] == peer_ratios and row["paired_RAW_seconds"] == [raw.get(k) for k in keys] and row["paired_matched_Bolt_seconds"] == [peer.get(k) for k in keys],"BOTH_EXPLICIT_PAIRED_DENOMINATORS")
                require(row["origins"] == index["RAW_BOLT",batch]["origins"] and len(row["origins"]) == batch and row["positions"] == [p["position"] for p in row["samples"]],"SAME_PHYSICAL_ORIGINS_AND_TIMER_POSITIONS")
                for key,value in derived.items(): require((row[key] is None and value is None) or (value is not None and math.isclose(row[key],value,abs_tol=1e-12,rel_tol=1e-12)),"RECOMPUTED_COST_AGGREGATE:"+key)
                rows.append({"dataset":data,"route":route,"batch":batch,"status":row["status"],**derived,"matched_Bolt_route":matched,"same_D_or_capacity_match":False if "RIDGE" in route else None,"seconds":values,"paired_RAW_ratios":raw_ratios,"paired_matched_Bolt_ratios":peer_ratios,**{key:row[key] for key in ("positions","origins","memory","selected_metadata","timed_outputs","reference","failure","whole_includes","whole_excludes","memory_scope")}})
                for sample,rr,mr in zip(row["samples"],raw_ratios,peer_ratios): samples.append({**sample,"seconds_per_window":sample["seconds"]/batch,"paired_RAW_ratio":rr,"paired_matched_Bolt_ratio":mr})
    return rows,samples


def collect_tables(values):
    cross,native,ridge = (values[k] for k in ("cross","native","ridge")); tables = {key:[] for key in TABLES}
    def add_scores(data,route,split,entry,*,source=None):
        for mask,key in (("all_observed","scores_all_observed"),("complete_C","scores_complete_mask")):
            score = entry[key]
            for scale in SCALES: tables["SCORES"].append({"dataset":data,"route":route,"split":split,"mask":mask,"scale":scale,"status":"COMPLETE" if scale in score else score["status"],**(score[scale] if scale in score else {}),"source":source,"Raw_point_summary":"MEDIAN_0.5"})
    for data,(_,basek,dev) in DATASETS.items():
        for split in ("fit","tune",dev):
            first = cross["datasets"][data]["conditions"][state_names(data)[0]]["splits"][split]; raw = first["methods"]["RAW_BOLT"]; add_scores(data,"RAW_BOLT",split,raw,source=first["input_source"])
            for name in state_names(data):
                owner = cross["datasets"][data]["conditions"][name]; entry = owner["splits"][split]; require(entry["methods"]["RAW_BOLT"] == raw,"IDENTICAL_ORIGINAL_MEDIAN_RAW_CONTEXT")
                add_scores(data,name+"_BOLT",split,entry["methods"]["ACTUAL_F"],source=entry["artifact"])
                for method in ("ACTUAL_F","RAW_BOLT","E_Y"):
                    score = entry["methods"][method]["scores_oracle_common_mask"]; tables["ORACLE_CONTEXT"].append({"dataset":data,"condition":name,"split":split,"method":method,"mask":"oracle_dependency_common_complete_C_only","scores":score,"oracle_valid_horizons":entry["oracle_valid_horizons"],"complete_C_horizons":entry["complete_C_horizons"],"artifact":entry["artifact"],"deployment_method":method != "E_Y","future_truth_used_by_E_Y_only":method == "E_Y"})
                for seed in SEEDS:
                    row = native["datasets"][data]["conditions"][name]["native"][str(seed)]; cp = native_selected(row); selected = {"fit":row["FIT"],"tune":row["TUNE"],dev:row["DEV"]}[split]; add_scores(data,name+"_NATIVE_s"+str(seed),split,selected,source=selected["artifact"])
                    if split == "tune":
                        tables["SELECTIONS"].append({"dataset":data,"family":"NATIVE","route":name+"_NATIVE_s"+str(seed),"condition":name,"seed":seed,"selected_update":cp["update"],"selected_TUNE_MSE":cp["tune"]["mse"],"selected_state":row["selected_state"],"selection_lock":row["selection_lock"],"updates_log":row["updates_log"]})
                        for candidate in row["checkpoints"]: tables["NATIVE_CHECKPOINTS"].append({"dataset":data,"condition":name,"seed":seed,"update":candidate["update"],"selected":candidate["update"] == cp["update"],"tune":candidate["tune"],"checkpoint_state":candidate["state"],"artifact":candidate["artifact"],"head_state_sha256":candidate["head_state_sha256"],"checks":candidate["checks"]})
        for route in ridge_routes(data):
            row = ridge["datasets"][data]["routes"][route]; chosen = ridge_selected(row); cost_route = ridge_cost_route(route)
            for split,entry in (("fit",row["FIT"]),("tune",row["TUNE"]),(dev,row["DEV"])): add_scores(data,cost_route,split,entry,source=entry.get("artifact",row["selected_FIT_TUNE_prediction_artifact"]))
            tables["SELECTIONS"].append({"dataset":data,"family":"DIRECT_C_RIDGE","route":cost_route,"condition":None if route.startswith("RAW_") else route.removesuffix("_HISTORY512"),"selected_alpha":chosen["alpha"],"selected_upper_grid_boundary":chosen["alpha"] == ALPHAS[-1],"selected_TUNE_MSE":chosen["TUNE_MSE"],"selected_coefficients":row["selected_coefficients"],"selection_lock":row["selection_lock"],"D_bypassed":True,"same_D_or_capacity_matched":False})
            for candidate in row["candidates"]: tables["RIDGE_CANDIDATES"].append({"dataset":data,"route":cost_route,"alpha":candidate["alpha"],"status":candidate["status"],"selected":candidate["alpha"] == chosen["alpha"],"FIT":candidate.get("FIT"),"TUNE":candidate.get("TUNE"),"TUNE_MSE":candidate.get("TUNE_MSE"),"prediction_artifact":candidate.get("prediction_artifact"),"coefficient_value_sha256":candidate["coefficient_value_sha256"],"coefficient_array_records":candidate["coefficient_array_records"],"failure":candidate.get("failure")})
            tables["SOLVERS"].append({"dataset":data,"route":cost_route,"feature_dimension":row["feature_dimension"],"response_dimension":row["response_dimension"],"normalization":row["normalization"],"response":row["response"],"fit_metadata":row["fit_metadata"],"fit_artifact":row["fit_artifact"],"independent_selected_fit_diagnostic":values["ridge_audit"]["datasets"][data][route]["fit_diagnostic"],"FIT_sources":row["FIT_sources"],"coefficient_deployment":row["coefficient_deployment"]})
        for split in ("fit","tune",dev):
            for suffix in ("_BOLT","_NATIVE_s92601","_NATIVE_s92602","_RIDGE_HISTORY512"):
                for contrast,pairs in (("CROSS_minus_PCA",[("PCA_K"+str(k),"CROSS_MOMENT_K"+str(k)) for k in (basek,2*basek)]),("doubleK_minus_baseK",[(kind+"_K"+str(basek),kind+"_K"+str(2*basek)) for kind in ("PCA","CROSS_MOMENT")])):
                    for first,second in pairs:
                        a,b = (choose(tables["SCORES"],dataset=data,route=name+suffix,split=split,mask="all_observed",scale="normalized")["mse"] for name in (first,second))
                        tables["DIRECTION_EFFECTS"].append({"dataset":data,"split":split,"contrast":contrast,"predictor":suffix.removeprefix("_"),"first_route":first+suffix,"second_route":second+suffix,"first_MSE":a,"second_MSE":b,"second_minus_first_MSE":b-a,"second_to_first_MSE_ratio":b/a if a else None,"DEV_does_not_select_direction_or_K":True})
    tables["WHOLE_COST"],tables["WHOLE_COST_SAMPLES"] = checked_cost_rows(values["cost"])
    for cost in tables["WHOLE_COST"]:
        data,route = cost["dataset"],cost["route"]; dev = DATASETS[data][2]; score = choose(tables["SCORES"],dataset=data,route=route,split=dev,mask="all_observed",scale="normalized"); raw = choose(tables["SCORES"],dataset=data,route="RAW_BOLT",split=dev,mask="all_observed",scale="normalized")
        tables["ACCURACY_COST_REFERENCE"].append({"dataset":data,"route":route,"batch":cost["batch"],"cost_status":cost["status"],"DEV_MSE":score["mse"],"median_RAW_DEV_MSE":raw["mse"],"DEV_MSE_to_median_RAW":score["mse"]/raw["mse"] if raw["mse"] else None,"paired_RAW_median_latency_ratio":cost["paired_RAW_median_ratio"],"median_seconds_per_window":cost["median_seconds_per_window"],"accuracy_split":dev,"timing_split":"first_B_TUNE_physical","DEV_selected_or_timed":False})
    tables["RESOURCES"] = [{"stage":key,**values[key]["resource_terminal"]} for key in FIXED_SHA]
    for stage in ("cross","native","ridge","cost"):
        for failure in values[stage]["failed_paths"]: tables["FAILURES"].append({"stage":stage,**failure})
    for row in tables["WHOLE_COST"]:
        if row["status"] != "COMPLETE": tables["FAILURES"].append({"stage":"cost_cell","dataset":row["dataset"],"route":row["route"],"batch":row["batch"],"status":row["status"],"failure":row["failure"]})
    require([len(tables[key]) for key in TABLES[:-1]] == [456,96,80,84,28,72,12,76,len(tables["WHOLE_COST_SAMPLES"]),76,8] and len(tables["WHOLE_COST_SAMPLES"]) <= 380 and (values["cost"]["status"] != "COMPLETE" or len(tables["WHOLE_COST_SAMPLES"]) == 380),"COMPLETE_NEW_DIRECTION_REPORT_TABLE_COVERAGE")
    return tables


def short(route): return route.replace("CROSS_MOMENT","X").replace("PCA","P").replace("_NATIVE_s92601"," N1").replace("_NATIVE_s92602"," N2").replace("_RIDGE_HISTORY512"," EH-ridge").replace("_BOLT"," F").replace("RAW_CURRENT_C_RIDGE","Raw C-ridge").replace("RAW_HISTORY512_RIDGE","Raw H-ridge")


def figures(tables, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    folder = output/"figures"; folder.mkdir(); files=[]; colors=("#0072B2","#D55E00","#009E73","#CC79A7")
    def save(fig,name,bottom=.10):
        fig.tight_layout(rect=(0,bottom,1,.94))
        for ext in ("png","svg"):
            path=folder/(name+"."+ext); fig.savefig(path,dpi=160,bbox_inches="tight"); files.append(path)
        plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(15,10))
    for ri,data in enumerate(DATASETS):
        for ax,split in zip(axes[ri],("tune",DATASETS[data][2])):
            for j,(suffix,label) in enumerate((("_BOLT","Frozen F"),("_NATIVE_s92601","Native order1"),("_NATIVE_s92602","Native order2"),("_RIDGE_HISTORY512","Encoded-history ridge; bypass D"))):
                ys=[choose(tables["SCORES"],dataset=data,route=name+suffix,split=split,mask="all_observed",scale="normalized")["mse"] for name in state_names(data)]; ax.plot(range(4),ys,"o-",color=colors[j],label=label)
            for route,style,label in (("RAW_BOLT",":","Raw median F"),("RAW_CURRENT_C_RIDGE","--","Raw current ridge"),("RAW_HISTORY512_RIDGE","-.","Raw history ridge")):
                value=choose(tables["SCORES"],dataset=data,route=route,split=split,mask="all_observed",scale="normalized")["mse"]; ax.axhline(value,color="black",linestyle=style,label=label)
            ax.set_xticks(range(4),[name.replace("CROSS_MOMENT","CROSS") for name in state_names(data)],rotation=12); ax.set_yscale("log"); ax.set_title(data+" / "+("TUNE" if split=="tune" else "exposed DEV")); ax.set_ylabel("All-observed normalized MSE (log)"); ax.grid(alpha=.18)
    fig.suptitle("Direction and K with two predictor controls: same-D native vs D-bypassing ridge"); fig.legend(*axes[0,0].get_legend_handles_labels(),loc="lower center",ncol=3,fontsize=9); save(fig,"01_directions_and_predictors",.13)
    fig,axes=plt.subplots(2,2,figsize=(15,11))
    for ri,data in enumerate(DATASETS):
        ax=axes[ri,0]
        for j,route in enumerate(ridge_routes(data)):
            rows=[choose(tables["RIDGE_CANDIDATES"],dataset=data,route=ridge_cost_route(route),alpha=alpha) for alpha in ALPHAS]; ys=[r["TUNE_MSE"] for r in rows]; ax.plot(range(7),ys,"o-",label=short(ridge_cost_route(route)),linewidth=1.1); selected=next(i for i,r in enumerate(rows) if r["selected"]); ax.scatter(selected,ys[selected],facecolors="none",edgecolors="black",s=75,zorder=5)
        ax.set_xticks(range(7),["0","1e-4",".001",".01",".1","1","10"]); ax.set_yscale("log"); ax.set_title(data+" / direct-ridge own TUNE"); ax.set_xlabel("v1 alpha menu; circled selection"); ax.set_ylabel("All-observed MSE (log)"); ax.grid(alpha=.18); ax.legend(fontsize=7,loc="best")
        ax=axes[ri,1]; positions=np.arange(8)
        for j,(method,label) in enumerate((("RAW_BOLT","Raw median F"),("ACTUAL_F","Compressed F"),("E_Y","Future-label oracle E(Y)"))):
            rows=[choose(tables["ORACLE_CONTEXT"],dataset=data,condition=name,split=split,method=method) for name in state_names(data) for split in ("tune",DATASETS[data][2])]; ax.plot(positions,[r["scores"]["normalized"]["mse"] for r in rows],"o-",color=colors[j],label=label)
        ax.set_xticks(positions,[name.replace("CROSS_MOMENT","X").replace("PCA","P")+"\n"+split for name in state_names(data) for split in ("TUNE","DEV")],rotation=15); ax.set_yscale("log"); ax.set_title(data+" / conservative oracle dependency mask only"); ax.set_ylabel("Same oracle-common-mask MSE (log)"); ax.grid(alpha=.18); ax.legend(fontsize=8)
    fig.suptitle("Finite ridge regularization menu and separate same-mask oracle context"); save(fig,"02_ridge_menu_and_oracle",.07)
    fig,axes=plt.subplots(2,2,figsize=(16,15))
    for ri,data in enumerate(DATASETS):
        for ax,batch in zip(axes[ri],(1,8)):
            rows=[choose(tables["WHOLE_COST"],dataset=data,route=route,batch=batch) for route in routes(data)]
            for index,row in enumerate(rows):
                color="black" if row["route"] == "RAW_BOLT" else colors[3] if "RIDGE" in row["route"] else colors[1] if "NATIVE" in row["route"] else colors[0]
                if row["median_seconds_per_window"] is not None: ax.barh(index,row["median_seconds_per_window"]*1000,color=color,alpha=.6); ax.scatter([v/batch*1000 for v in row["seconds"]],[index]*len(row["seconds"]),color=color,s=10,zorder=3)
                if row["status"] != "COMPLETE": ax.text(0,index,row["status"],fontsize=7,color="red")
            ax.set_yticks(range(19),[short(route) for route in routes(data)],fontsize=8); ax.invert_yaxis(); ax.set_title(data+" / B"+str(batch)); ax.set_xlabel("Whole ms/window; bar=median, dots=5 repeated TUNE samples"); ax.grid(axis="x",alpha=.18)
    fig.suptitle("Actual physical CPU input to physical CPU return: 76 cells / 380 samples"); save(fig,"03_whole_cost",.04)
    fig,axes=plt.subplots(2,2,figsize=(15,10))
    for ri,data in enumerate(DATASETS):
        for ax,batch in zip(axes[ri],(1,8)):
            for row in [r for r in tables["ACCURACY_COST_REFERENCE"] if r["dataset"] == data and r["batch"] == batch]:
                route=row["route"]; color="black" if route == "RAW_BOLT" else colors[3] if "RIDGE" in route else colors[1] if "NATIVE" in route else colors[0]; marker="D" if "RIDGE" in route else "s" if "NATIVE" in route else "o"
                if row["cost_status"] != "COMPLETE": continue
                x,y=row["paired_RAW_median_latency_ratio"],row["DEV_MSE_to_median_RAW"]; ax.scatter(x,y,color=color,marker=marker,s=45,alpha=.8)
                if "NATIVE" not in route and ("RIDGE" not in route or route.startswith("RAW_")): ax.annotate(short(route),(x,y),xytext=(4,5),textcoords="offset points",fontsize=7)
            ax.axvline(1,color="gray",linestyle=":"); ax.axhline(1,color="gray",linestyle=":"); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(data+" / B"+str(batch)); ax.set_xlabel("Median paired whole latency / Raw median F (log)"); ax.set_ylabel("Exposed DEV MSE / Raw median F (log)"); ax.grid(alpha=.18)
    handles=[Line2D([],[],color=color,marker=marker,linestyle="",label=label) for color,marker,label in (("black","o","Raw median F"),(colors[0],"o","Frozen E/F/D"),(colors[1],"s","Selected native heads"),(colors[3],"D","Selected direct-C ridge"))]
    fig.legend(handles=handles,loc="lower center",ncol=4); fig.suptitle("Accuracy-cost context: DEV accuracy and TUNE cohort timing; no DEV winner selection"); save(fig,"04_accuracy_cost",.10)
    return files


def make_text(values,tables,bindings,output):
    def primary(data,route,split): return choose(tables["SCORES"],dataset=data,route=route,split=split,mask="all_observed",scale="normalized")["mse"]
    native=[r for r in tables["SELECTIONS"] if r["family"] == "NATIVE"]; ridge=[r for r in tables["SELECTIONS"] if r["family"] == "DIRECT_C_RIDGE"]; histogram=collections.Counter(r["selected_update"] for r in native); endpoint=sum(r["selected_upper_grid_boundary"] for r in ridge)
    lines=["# 새 방향·차원과 예측기·직접 회귀의 비교","",f"[확인] 두 dataset의 PCA/CROSS 두 방향과 기본K/2K로8개 E/D를 고정했고 같은 공간에서 native16개를600회씩 학습했다. 이 보고서는 원 median Raw/F 기준만 사용한다. native CP 선정은 "+", ".join(f"CP{k} {v}/16" for k,v in sorted(histogram.items()))+f"이다. direct ridge12개 중 α 목록 최댓값10을 고른 조건은 {endpoint}/12다. 후보 범위가 제한된 결과이며, α>10의 후속 실험이나 다른 계수를 이번 v1 비용값으로 대체하지 않는다.",""]
    for data,(_,basek,dev) in DATASETS.items():
        raw=primary(data,"RAW_BOLT",dev); fbetter=sum(primary(data,name+"_BOLT",dev)<raw for name in state_names(data)); nbetter=sum(primary(data,r["route"],dev)<primary(data,r["condition"]+"_BOLT",dev) for r in native if r["dataset"] == data)
        lines.append(f"- {data}: 원 median Raw DEV MSE {raw:.12g}; 같은 Raw보다 낮은 frozen F {fbetter}/4, 같은 E/D의 F보다 낮은 native {nbetter}/8. 이 DEV 비교로 방향·K·seed를 다시 선정하지 않았다.")
        for name in state_names(data):
            route=name+"_BOLT"; t,d=primary(data,route,"tune"),primary(data,route,dev); n1,n2=(primary(data,name+"_NATIVE_s"+str(seed),dev) for seed in SEEDS); er=primary(data,name+"_RIDGE_HISTORY512",dev)
            lines.append(f"- {data} {name}: F TUNE {t:.12g}, DEV {d:.12g}; native DEV {n1:.12g}/{n2:.12g}; encoded-history 직접 C ridge DEV {er:.12g}. 마지막 경로는 D를 우회한다.")
    lines += ["","![방향·차원·예측기](figures/01_directions_and_predictors.png)","","**질문과 공정한 대조.** FIT 과거 covariance whitening의 PCA 방향과 horizon별 complete-C current/future cross-moment 방향을 같은 K/2K에서 비교했다. E와 과거32~511의 관측 reconstruction LS로 얻은 causal L7 D는 각 조건 안에서 고정했다. native는 동일 E/D/center·512과거·48미래·현재 C anchor·lag [0,1,2,4,8,16,32]를 쓰는 stream별 affine temporal head다. zero head와 seed92601/92602의 자료 순서, AdamW lr.001/WD0/B4/clip1/600update 및 CP0/100/200/400/600을 유지한다. 두 seed는 독립 dataset이 아니며 600회는 유한 예산이다. 이 head의 성적을 모든 native 최적화나 채널 혼합 예측기의 불가능성으로 확대하지 않는다.","","**D를 우회하는 대조의 의미.** RAW_CURRENT_C는 현재 C로 미래 level, RAW_HISTORY512는 vec(history-currentC)로 미래 innovation, ENCODED_HISTORY512는 같은 actual Torch32 z의 vec(z-z_last)로 미래 innovation을 직접 회귀한다. intercept는 penalty에서 제외하며 원 currentC를 뒤에 더한다. Raw current는 horizon별 complete-C FIT 응답과 subset mean/RMS, history 경로는 전체 FIT mean/RMS와 각(h,c) 관측 subset을 사용한다. horizon별 관측 분모가 다르며 미래 dummy를 응답에 넣지 않는다. D를 우회한 encoded ridge가 좋으면 과거 표현에 예측 정보가 남는 단서지만, latent future 예측·frozen F 성공이나 same-D/capacity match의 증명은 아니다. 원 C의 current anchor와 과거 buffer도 입력 정보다.","","ridge의 α=[0,1e-4,.001,.01,.1,1,10]과 FIT 표준화·비벌점 intercept를 고정했다. alpha0의 normalized design minimum-norm cutoff1e-8과 positive ridge의 모든 singular 방향을 구분했다. 실제 배포는 fused coefficient/intercept를 FP32로 저장한 feature32@coef32+intercept32+anchor32이며 own actual FP32 TUNE macro MSE로 선정한다. native 동률은 이른 CP, ridge 동률은 사전 α 순서다. 원16개 및12개 선정 잠금 뒤 DEV를 열었고 DEV는 새 선정에 쓰지 않았다. selected ridge12개만 독립 목적 해를 재계산했고, 미선정72계수는 보존 해시와168개 FIT/TUNE 저장 예측 점수를 검산한 범위다.","","![유한 alpha 메뉴와 공통 mask oracle](figures/02_ridge_menu_and_oracle.png)","","**관측 지표와 oracle.** 주 지표는 각 split 자체 true-observed channel count를 분모로 한 normalized channel-macro MSE다. complete-C 지표와 물리 단위 channel 진단도 CSV에 따로 보존했다. E(Y)는 미래 정답을 쓰며 모든 소비 미래 참조가 complete-C인 보수적 oracle dependency mask에서만 Raw/F와 함께 제시했다. 특히 Beijing의 oracle MSE를 all-observed MSE와 직접 비교하지 않는다. E(Y)는 배포 예측기가 아니며 이 단계에서 새 output-space minimum을 계산하지 않았다.",""]
    for data in DATASETS:
        for route in ("RAW_CURRENT_C_RIDGE","RAW_HISTORY512_RIDGE"):
            row=choose(tables["SELECTIONS"],dataset=data,route=route); lines.append(f"- {data} {route}: selected α {row['selected_alpha']:g}; TUNE {primary(data,route,'tune'):.12g}, DEV {primary(data,route,DATASETS[data][2]):.12g}.")
    complete=sum(r["status"] == "COMPLETE" for r in tables["WHOLE_COST"])
    lines += ["","**새 전체 배포 비용.** 과학 학습은 기존 F24context(원104forward 기록)를 재사용했지만 이 비용은 실제 frozen F 또는 선정 native/ridge를 실행한다. 원 physical CPU TUNE 첫 B1/B8에서 causal fill/FIT scaler→E/actual F 또는 native/old D, 혹은 direct ridge→physical CPU 반환과 sync까지 측정했다. encoded ridge의 실제 CUDA E와 CPU 특징/계수 계산 및 전송도 timer 안이다. 로드·학습·선정·reference·warmup·save·외부 hash/meter·memory snapshot은 밖이고 내부 shape/finite 검사와 sync는 안에 남는다.","",f"19routes×2datasets×B1/B8의 {complete}/76셀과 {len(tables['WHOLE_COST_SAMPLES'])}/380개 정수 ns 표본을 보존했다. 각 warm2/실측5이며 19위치 중5회전은 완전 균형이 아니다. 각 셀의5회는 같은 첫B origin 반복이어서 독립 표본 CI를 만들지 않았다. 공유 resident F/24adapter/16head/12CPU ridge 상태의 RSS/VRAM은 독점 메모리가 아니다. query 또는 stream 수의 비를 latency 비로 대체하지 않는다.","","![전체 비용과 모든 다섯 표본](figures/03_whole_cost.png)","","Raw와 같은 E의 Bolt 각각에 대해 repeat를 맞춘 median paired ratio와 ratio of medians를 정수 timer 원수치에서 다시 계산해 분리했다. encoded ridge의 matching Bolt 분모는 같은 E일 뿐 D/capacity가 일치한다는 뜻이 아니다. B1/B8는 실제 해당 크기로 실행했고 canonical32 과학 출력과의 first8/물리 reference를 검산했다. 서로 다른 cohort의 시간이나 후속 ridge 계수로 이번 결과를 바꾸지 않았다. 실패 route는 status와 FAILURES에서 보존하며 완전 보고 생성 gate는 과학 및 비용 COMPLETE/독립 PASS를 요구한다.","","![정확도와 비용의 문맥](figures/04_accuracy_cost.png)","","[해석 범위] 네 번째 그림은 적응적으로 노출된 DEV 정확도와 TUNE cohort 전체 시간을 연결한 문맥이다. DEV를 비용 측정하거나 새 winner를 고른 그림이 아니다. 기존 median Raw 기준을 quantile-summary 선정 Raw로 대체하지 않았고 이번 결과를 SOTA·새 정보 보존 증명·F 필요성/제거 가능성의 확정으로 포장하지 않는다. 같은 자료의 seed·K·방향을 독립 dataset으로 세지 않으며 보호 TEST 접근0, 독립 holdout0, strict K-only0이다. 원 E/D의 과거 supervised 이력과 현재 head/ridge FIT 응답 사용을 구분한다. F/optimizer 전체를 재현한 독립 실험이 아니라 저장 산술·로그·원 source/receipt 검산과 새 실제 timer 기록의 근거다.","","본문·CSV·그림은 JSON metadata만 읽었다. NPZ/model/F/GPU/훈련을 이 보고 생성기에서 열거나 실행하지 않는다. RESOURCES는 cross/native/ridge/cost 및 각 감사의 process CPU·process wall·GPU process 범위를 원 필드 그대로 보존하며 학습 wall을 배포 latency로 쓰지 않는다.",""]
    lines += [f"- [{key}.csv](tables/{key}.csv): {len(rows)}행" for key,rows in tables.items()]
    lines += [""]+[f"- [{key}]({rel(item['path'],output)}): `{item['sha256']}`" for key,item in bindings.items()]
    return "\n".join(lines)+"\n"


def generate(paths,destination):
    output=resolve(destination); require(output.parent == (HERE/"reports").resolve() and not output.exists(),"FRESH_LOCAL_REPORT_OUTPUT")
    values,bindings=load_verified(paths); tables=collect_tables(values); source=record(__file__); bindings["report_source"]=source; output.mkdir(parents=True,exist_ok=False); (output/"tables").mkdir()
    for key,rows in tables.items(): write_csv(output/"tables"/(key+".csv"),rows,output)
    images=figures(tables,output); (output/"REPORT_KO.md").write_text(make_text(values,tables,bindings,output),encoding="utf-8",newline="\n"); require(record(__file__) == source,"REPORT_SOURCE_UNCHANGED")
    for item in bindings.values(): bound(item)
    manifest={"status":"COMPLETE","source":source,"inputs":bindings,"files":[{"path":path.relative_to(output).as_posix(),"sha256":sha(path),"bytes":path.stat().st_size} for path in sorted(output.rglob("*")) if path.is_file()],"table_row_counts":{key:len(rows) for key,rows in tables.items()},"PNG_count":4,"SVG_count":4,"scientific_NPZ_or_model_read":False,"new_fit_or_training_during_generation":0,"new_output_space_computed":False,"old_median_RAW_only":True,"v1_ridge_alpha_upper_bound":10.,"actual_cost_has_live_F":True,"protected_TEST_accessed":False,"independent_holdout":False,"strict_K_only":False}
    with (output/"MANIFEST.json").open("x",encoding="utf-8") as handle: json.dump(manifest,handle,ensure_ascii=False,indent=2,allow_nan=False); handle.write("\n")
    print(json.dumps({"status":"COMPLETE","manifest":str(output/"MANIFEST.json"),"images":len(images)}),flush=True); return manifest


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in (*FIXED_SHA,"output"): parser.add_argument("--"+key.replace("_","-"),dest=key,required=True)
    args=vars(parser.parse_args()); destination=args.pop("output"); generate(args,destination)
