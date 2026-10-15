"""JSON-only Korean report of matched static/ramp corrections and new whole cost."""
from pathlib import Path
import argparse
import json
import math
import statistics

from make_linear_e_adaptation_report_v2 import require,resolve,sha,record,read,bound,same_opaque,choose,rel,write_csv

HERE=Path(__file__).resolve().parent
DATASETS={"robin":(17,4,"val"),"beijing_aotizhongxin":(11,2,"eval")}
ARMS=("STATIC_DELTA","RAMP_DELTA");MODES=("F","FLAT")
ALPHAS=(0.,1e-4,.001,.01,.1,1.,10.,100.,1000.,10000.)
TABLES=("SELECTED","CANDIDATES","SCORES","BASELINE_SCORES","TEMPORAL","LATENT","SOLVERS","WHOLE_COST","WHOLE_COST_SAMPLES","WHOLE_COST_REPEAT","WHOLE_COST_REPEAT_SAMPLES","RESOURCES")
FIXED_SHA={"science":"5f158382640dac74ec63cc3d1005b385c898821d3c032d51b543f2c9b27d7dac","science_audit":"0f65053536403a3b7b8f53fca3b9f8274fb1a406eaf6dfdcb73bb0467fffe698","cost":"1bbdacf6326b7111828a4f73cb54009580ba6fb021cc200e8a0a7b60cb6cbb59","cost_audit":"84e32f1f937733990140e7e70c1dcfd389aec807d116e600e3e91a4a52aa2f37","repeat_cost":"71391a967f264e578f469ed2b0f1eceb86830954e34ebe86d7d51999f094f713","repeat_cost_audit":"27d721c6f248caf521bf11dd045af2eb61151629bdb64fbda76750d4b056566c"}
AUDIT_SOURCES={"science_audit":{"path":str(HERE/"audit_horizon_decoder_diagnostic_v1.py"),"sha256":"accb6fb775d9b12ca6e39c0be7e062545b98d429089ee32e0ac46ed65edae613"},"cost_audit":{"path":str(HERE/"audit_horizon_decoder_cost_v2.py"),"sha256":"dada7b486966b3c9374c504aa001a8287c5c45b6b51a218bc2f7cbeae3185a3e"},"repeat_cost_audit":{"path":str(HERE/"audit_horizon_decoder_cost_v2.py"),"sha256":"dada7b486966b3c9374c504aa001a8287c5c45b6b51a218bc2f7cbeae3185a3e"}}
SCIENCE_COUNTS={"decoder_fit_paths_checked":24,"observed_mask_groups_checked":96,"alpha_candidates_checked":240,"alpha_zero_candidates_checked":24,"positive_alpha_candidates_checked":216,"ZERO_candidates_checked":24,"FIT_TUNE_prediction_evaluations_checked":528,"selected_DEV_prediction_evaluations_checked":24,"selected_coefficient_states_checked":24,"selected_fits_independently_solved":22,"selected_ZERO_states_checked":2,"selected_FP32_arithmetic_predictions_checked":72,"selection_locks_checked":24,"source_contexts_checked":18,"shared_RMS_spaces_checked":6,"latent_point_diagnostic_modes_checked":36,"score_fields_checked":14544,"temporal_score_fields_checked":33048,"protected_source_input_and_result_hashes_checked":2359}
COST_COUNTS={"cost_cells_checked":72,"recorded_timer_samples_checked":360,"frozen_PCA_state_hashes_checked":4,"selected_delta_coefficient_hashes_checked":24,"selection_locks_checked":24,"physical_prechecks_checked":36,"physical_first8_windows_checked":288,"independent_first8_B8_component_prediction_equations_checked":36,"ZERO_selected_fastpath_prechecks_checked":2,"canonical_TUNE_bridges_checked":2,"protected_prior_file_hashes_checked":2386}
REPEAT_COST_COUNTS=dict(COST_COUNTS)
PRESERVED_COST_AUDIT_FAILURES={"failed_cost_audit_v1":{"path":str(HERE/"audits/HORIZON_DECODER_COST_AUDIT_v1.json"),"sha256":"ee838121d9da22a9b1706f544978c8dccd514c31501f91fea1b66e039d23d92b"},"failed_repeat_cost_audit_v1":{"path":str(HERE/"audits/HORIZON_DECODER_COST_REPEAT_AUDIT_v1.json"),"sha256":"62f2c6939e7595c70b3daae6e7561dc4f8541b57540e54565be3d7f2de874e59"},"failed_cost_auditor_v1_source":{"path":str(HERE/"audit_horizon_decoder_cost_v1.py"),"sha256":"2169c11b3a423c50fbcff86443286462324c231ba0e11fef3715cf50df6dc4a8"}}
CONTROLLER_SOURCE={"path":str(HERE/"run_horizon_decoder_diagnostic_v1.py"),"sha256":"151da5b0b4cc4cacdb9d7a547d12e7837dcdcb5ee4c5047a1d6e3993afe525c6"}
MODULE_SOURCE={"path":str(HERE/"horizon_decoder_delta_v1.py"),"sha256":"406252e940ae676e7a78f69e067f5b3fa70456bdb1ea27d57169b8964ceabe0e"}
COST_SOURCE={"path":str(HERE/"run_horizon_decoder_cost_v1.py"),"sha256":"11f0a97e8ad711d35126a4528c34984ddc60174b756813d6d8236c69e3777e5a"}
COST_HELPER_SOURCE={"path":str(HERE/"whole_horizon_decoder_cost_v1.py"),"sha256":"dc6f1f5a227cf31c8ed58a6d3ac789f5a8bbc7eda311a48d40b5acfbf0a9606a"}
UTILITY={"path":str(HERE/"make_linear_e_adaptation_report_v2.py"),"sha256":"32e2460ea4724144067f264ab1ceb0332cd04c5fce2296838024c119c9572466"}


def spaces(data):
    k=DATASETS[data][1];return ("RAW_C","PCA_K"+str(k),"PCA_K"+str(2*k))


def routes(data):return tuple(space+"_"+mode+"_"+variant for space in spaces(data) for mode in MODES for variant in ("BASELINE",*ARMS))


def matched_baseline(route):
    for suffix in ("_STATIC_DELTA","_RAMP_DELTA"):
        if route.endswith(suffix):return route.removesuffix(suffix)+"_BASELINE"
    return route


def load_verified(paths):
    require(set(paths)==set(FIXED_SHA) and all(isinstance(v,str) and len(v)==64 for v in FIXED_SHA.values()) and all(isinstance(v,dict) for v in AUDIT_SOURCES.values()) and isinstance(COST_SOURCE,dict) and isinstance(COST_HELPER_SOURCE,dict) and bool(COST_COUNTS) and bool(REPEAT_COST_COUNTS),"ACTUAL_SCIENCE_AND_NEW_WHOLE_COST_PASSES_NOT_YET_REGISTERED")
    bound(UTILITY);values={key:read(path) for key,path in paths.items()};bindings={key:record(path) for key,path in paths.items()}
    for key,value in values.items():
        require(bindings[key]["sha256"]==FIXED_SHA[key] and value["status"]==("PASS" if key.endswith("audit") else "COMPLETE") and value["protected_TEST_accessed"] is False,"EXACT_COMPLETE_OR_PASS_NO_TEST:"+key)
        receipt=value["resource_terminal"];require(receipt["status"]==value["status"] and receipt["error"] is None and all(type(receipt[field]) in (int,float) and math.isfinite(receipt[field]) and receipt[field]>=0 for field in ("process_cpu_seconds","wall_seconds")),"VALID_PROCESS_RESOURCE_RECEIPT")
        require(read(bound(receipt))=={k:v for k,v in receipt.items() if k not in ("path","sha256","bytes")},"EXACT_EMBEDDED_RESOURCE_AND_SAVED_RECEIPT")
    science,audit=values["science"],values["science_audit"]
    for key,counts in (("science_audit",SCIENCE_COUNTS),("cost_audit",COST_COUNTS),("repeat_cost_audit",REPEAT_COST_COUNTS)):
        require(all(type(values[key]["totals"].get(k)) is int and values[key]["totals"][k]==v for k,v in counts.items()) and not values[key].get("partial_paths",[]),"FULL_REGISTERED_INDEPENDENT_COVERAGE:"+key);bound(values[key]["audit_source"],AUDIT_SOURCES[key]);bindings[key+"_source"]=values[key]["audit_source"]
    bound(audit["diagnostic"],bindings["science"]);bound(science["controller_source"],CONTROLLER_SOURCE);bound(science["module_source"],MODULE_SOURCE)
    for stage,audit_stage in (("cost","cost_audit"),("repeat_cost","repeat_cost_audit")):
        cost,ca=values[stage],values[audit_stage];bound(ca["cost"],bindings[stage])
        for value in (cost,ca):bound(value["horizon_decoder_diagnostic"],bindings["science"]);bound(value["horizon_decoder_audit"],bindings["science_audit"]);bound(value["selection_registry"],science["selection_registry"])
        bound(cost["controller_source"],COST_SOURCE);bound(cost["whole_cost_source"],COST_HELPER_SOURCE)
        require(cost["completed_cost_cells"]==cost["expected_cost_cells"]==72 and cost["recorded_timer_samples"]==cost["expected_timer_samples"]==360 and cost["physical_prechecks_completed"]==36 and cost["frozen_PCA_models_loaded"]==4 and cost["selected_correction_states_loaded"]==24 and not cost["failed_paths"],"ALL_NEW_72_COST_CELLS_360_TIMERS:"+stage)
        require(cost["F_before"]==cost["F_after"]==science["F_parent_signatures_reused"]["before"]==science["F_parent_signatures_reused"]["after"] and cost["frozen_E_D_before"]==cost["frozen_E_D_after"] and cost["selected_coefficients_before"]==cost["selected_coefficients_after"],"F_E_D_AND_SELECTED_CORRECTIONS_UNCHANGED:"+stage)
        for key in ("cost_plan","whole_cost_source","controller_source"):bound(cost[key]);bindings[stage+"_"+key]=cost[key]
    require(science["fits_completed"]==science["resolved_selection_locks"]==science["selected_DEV_evaluations_completed"]==24 and science["alpha_candidates_completed"]==240 and science["ZERO_candidates_completed"]==24 and science["candidate_FIT_TUNE_evaluations_completed"]==528 and science["all_24_locks_before_DEV"] is True and not science["failed_paths"],"ALL_TWENTY_FOUR_SCIENCE_PATHS")
    require(science["new_F_forward_calls"]==science["new_F_model_objects_loaded"]==science["new_optimizer_updates"]==science["new_E_D_training_updates"]==0 and science["actual_F_ungated_FIT_RMS_shared_by_four_arms"] is True and science["output_space_minimum_status"]=="NOT_MEASURED_IN_THIS_STAGE" and science["whole_deployment_cost_measured"] is False,"CACHED_F_SCIENCE_SCOPE")
    require(values["cost"]["routes"]==values["repeat_cost"]["routes"] and values["cost"]["selection_locks"]==values["repeat_cost"]["selection_locks"] and values["cost"]["selected_coefficients_before"]==values["repeat_cost"]["selected_coefficients_before"],"SAME_METHODS_AND_SELECTIONS_TWO_PRESERVED_COST_COHORTS")
    for key in ("cross_moment_diagnostic","cross_moment_audit","selection_registry","plan"):
        bound(science[key]);bindings[key]=science[key]
        if key!="plan":bound(audit[key],science[key])
    reason=record(HERE/"COST_REPEAT_REASON_KO_v1.md");bound(reason);bindings["cost_repeat_reason"]=reason
    for key,value in PRESERVED_COST_AUDIT_FAILURES.items():bound(value);bindings[key]=value
    bindings.update(science_controller_source=science["controller_source"],module_source=science["module_source"],report_utility=UTILITY)
    return values,bindings


def score(entry):return entry["scores_all_observed"]["normalized"]["mse"]


def selected(row):
    menu=["ALPHA_"+format(a,"g") for a in ALPHAS]+["ZERO_DELTA"];require([p["id"] for p in row["candidates"]]==menu and all(p["status"]=="COMPLETE" for p in row["candidates"]),"ALL_ELEVEN_PREDECLARED_CANDIDATES")
    winner=min(row["candidates"],key=lambda p:(p["TUNE_MSE"],menu.index(p["id"])));require(winner["id"]==row["selected_candidate"] and winner["TUNE_MSE"]==row["selected_TUNE_MSE"] and winner["FIT"]==row["FIT"] and winner["TUNE"]==row["TUNE"],"OWN_FP32_TUNE_WINNER_AND_REUSED_FIT_TUNE");same_opaque(winner["prediction_artifact"],row["selected_FIT_TUNE_prediction_artifact"]);return winner


def checked_cost_rows(cost):
    result,samples=[],[]
    for data in DATASETS:
        group=cost["cost"][data];require(group["routes"]==list(routes(data)) and len(group["rows"])==36,"ALL_DECLARED_EIGHTEEN_COST_ROUTES")
        index={(row["route"],row["batch"]):row for row in group["rows"]};require(set(index)=={(r,b) for r in routes(data) for b in (1,8)},"UNIQUE_36_DATASET_ROWS");durations={}
        for (route,batch),row in index.items():
            current={}
            for sample in row["samples"]:
                require(type(sample["perf_start_ns"]) is type(sample["perf_end_ns"]) is int and sample["perf_end_ns"]>sample["perf_start_ns"] and sample["repeat"] not in current and group["schedule"][sample["repeat"]][sample["position"]]==route and sample["route"]==route and sample["batch"]==batch and sample["dataset"]==data,"INTEGER_TIMER_IDENTITIES_AND_SERIAL_ROUTE_ORDER")
                seconds=(sample["perf_end_ns"]-sample["perf_start_ns"])/1e9;require(sample["seconds"]==seconds,"EXACT_NS_SECONDS");current[sample["repeat"]]=seconds;samples.append({**sample,"dataset":data})
            require(row["status"]=="COMPLETE" and row["repeats"]==5 and set(current)==set(range(5)) and row["seconds"]==list(current.values()),"COMPLETE_FIVE_TIMER_ROW");durations[route,batch]=current
        for (route,batch),row in index.items():
            seconds=durations[route,batch];median=statistics.median(seconds.values());require(row["median_seconds"]==median and row["median_seconds_per_window"]==median/batch,"NS_MEDIAN_AND_PER_WINDOW");base=matched_baseline(route);require(row["matched_baseline_route"]==base,"MATCHING_SPACE_MODE_BASELINE")
            for label,denominator in (("RAW","RAW_C_F_BASELINE"),("matched_baseline",base)):
                denom=durations[denominator,batch];paired=[seconds[i]/denom[i] for i in range(5)];require(row["paired_"+label+"_seconds"]==[denom[i] for i in range(5)] and row["paired_"+label+"_ratios"]==paired and row["paired_"+label+"_median_ratio"]==statistics.median(paired) and row["ratio_of_"+label+"_medians"]==median/statistics.median(denom.values()),"TWO_COST_RATIOS_RAW_AND_MATCHED_BASELINE")
            result.append({key:value for key,value in row.items() if key!="samples"})
    return result,samples


def collect_tables(values):
    tables={key:[] for key in TABLES};science=values["science"];audit=values["science_audit"]
    for data,(_,_,dev) in DATASETS.items():
        for space in spaces(data):
            owner=science["datasets"][data]["spaces"][space]
            for split in ("fit","tune",dev):
                context=owner["contexts"][split]
                for mode in (*MODES,"CURRENT_C"):
                    entry=context["current_C_persistence_scores"] if mode=="CURRENT_C" else context["baseline_scores"][mode];tables["BASELINE_SCORES"].append({"dataset":data,"space":space,"split":split,"mode":mode,"K":context["K"],"all_observed":entry["scores_all_observed"],"complete_C":entry["scores_complete_mask"],"input_source":context["input_source"]})
                for mode in MODES:tables["LATENT"].append({"dataset":data,"space":space,"split":split,"mode":mode,"K":context["K"],**context["latent_error_diagnostics"][mode],"complete_C_horizons":context["latent_error_diagnostics"]["complete_C_horizons"],"unit_scope":context["latent_error_diagnostics"]["unit_scope"],"point_mask_only":True,"C_output_EY_or_floor_status":"NOT_MEASURED_IN_THIS_STAGE"})
            for mode in MODES:
                for arm in ARMS:
                    name=mode+"_"+arm;row=owner["conditions"][name];winner=selected(row);common={"dataset":data,"space":space,"mode":mode,"arm":arm,"K":row["K"]}
                    summaries={split:{"selected_MSE":score(row["FIT" if split=="fit" else "TUNE" if split=="tune" else "DEV"]),"baseline_MSE":score(owner["contexts"][split]["baseline_scores"][mode]),"raw_F_MSE":score(science["datasets"][data]["spaces"]["RAW_C"]["contexts"][split]["baseline_scores"]["F"])} for split in ("fit","tune",dev)}
                    tables["SELECTED"].append({**common,"selected_candidate":row["selected_candidate"],"selected_kind":row["selected_kind"],"selected_alpha":row["selected_alpha"],"selected_TUNE_MSE":row["selected_TUNE_MSE"],"additional_slope_parameters":row["additional_slope_parameters"],"split_summary":summaries,"selected_coefficients":row["selected_coefficients"],"selection_lock":row["selection_lock"],"upper_alpha_boundary":row["selected_alpha"]==ALPHAS[-1]})
                    for candidate in row["candidates"]:tables["CANDIDATES"].append({**common,**candidate,"selected":candidate["id"]==winner["id"]})
                    for split,label in (("fit","FIT"),("tune","TUNE"),(dev,"DEV")):
                        entry=row[label];tables["SCORES"].append({**common,"split":split,"all_observed":entry["scores_all_observed"],"complete_C":entry["scores_complete_mask"]});temporal=row["selected_temporal_diagnostics"][label] if label!="DEV" else entry["temporal_diagnostics"];baseline=owner["contexts"][split]["baseline_temporal_diagnostics"][mode]
                        for period,index in (("horizon",i) for i in range(48)):
                            actual,base=temporal["per_horizon"][index],baseline["per_horizon"][index];valid="mse" in actual and "mse" in base;tables["TEMPORAL"].append({**common,"split":split,"period":period,"index":index+1,"normalized":actual,"baseline_normalized":base,"MSE_delta":actual["mse"]-base["mse"] if valid else None,"MAE_delta":actual["mae"]-base["mae"] if valid else None,"not_full_macro_average":True})
                        for interval in ("1_16","17_32","33_48"):
                            actual,base=temporal["registered_intervals"][interval],baseline["registered_intervals"][interval];valid="mse" in actual and "mse" in base;tables["TEMPORAL"].append({**common,"split":split,"period":"interval","index":interval,"normalized":actual,"baseline_normalized":base,"MSE_delta":actual["mse"]-base["mse"] if valid else None,"MAE_delta":actual["mae"]-base["mae"] if valid else None,"not_full_macro_average":True})
                    verified=audit["datasets"][data]["spaces"][space]["conditions"][name];tables["SOLVERS"].append({**common,"fit_wall_seconds":row["fit_wall_seconds"],"fit_metadata":row["fit_metadata"],"fit_artifact":row["fit_artifact"],"independent_selected_fit":verified["independent_fit"],"unselected_coefficients_independently_refitted":False})
    tables["WHOLE_COST"],tables["WHOLE_COST_SAMPLES"]=checked_cost_rows(values["cost"])
    tables["WHOLE_COST_REPEAT"],tables["WHOLE_COST_REPEAT_SAMPLES"]=checked_cost_rows(values["repeat_cost"])
    for stage,value in values.items():tables["RESOURCES"].append({"stage":stage,**value["resource_terminal"],"process_receipt_not_per_forecast_latency":True})
    expected={"SELECTED":24,"CANDIDATES":264,"SCORES":72,"BASELINE_SCORES":54,"TEMPORAL":3672,"LATENT":36,"SOLVERS":24,"WHOLE_COST":72,"WHOLE_COST_SAMPLES":360,"WHOLE_COST_REPEAT":72,"WHOLE_COST_REPEAT_SAMPLES":360,"RESOURCES":6};require({k:len(v) for k,v in tables.items()}==expected,"EXACT_REPORT_COHORT_COUNTS");return tables


def figures(tables,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    folder=output/"figures";folder.mkdir();files=[];colors={"F":"#0072B2","FLAT":"#D55E00"};styles={"STATIC_DELTA":"-","RAMP_DELTA":"--"}
    def save(fig,name,bottom=.08):
        fig.tight_layout(rect=(0,bottom,1,.93))
        for suffix in ("png","svg"):
            path=folder/(name+"."+suffix);fig.savefig(path,dpi=180);files.append(path)
        plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(15,9))
    for di,(data,(_,_,dev)) in enumerate(DATASETS.items()):
        for si,split in enumerate(("tune",dev)):
            ax=axes[di,si];names=spaces(data)
            for mode in MODES:
                bases=[choose(tables["BASELINE_SCORES"],dataset=data,space=s,split=split,mode=mode)["all_observed"]["normalized"]["mse"] for s in names];ax.plot(range(3),bases,":",color=colors[mode],marker="x",label=mode+" baseline")
                for arm in ARMS:
                    points=[choose(tables["SCORES"],dataset=data,space=s,split=split,mode=mode,arm=arm)["all_observed"]["normalized"]["mse"] for s in names];ax.plot(range(3),points,styles[arm],color=colors[mode],marker="o" if arm=="STATIC_DELTA" else "s",label=mode+" "+arm)
            ax.set_xticks(range(3),names);ax.set_title(data+" / "+split);ax.set_ylabel("All-observed normalized macro MSE");ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc="lower center",ncol=3,fontsize=9);fig.suptitle("Same fixed source, own TUNE selection; F/FLAT and static/ramp are separate contrasts");save(fig,"01_selected_accuracy",.12)
    fig,axes=plt.subplots(2,3,figsize=(16,8))
    for di,(data,(_,_,dev)) in enumerate(DATASETS.items()):
        for sj,space in enumerate(spaces(data)):
            ax=axes[di,sj]
            for arm,color in (("STATIC_DELTA","#0072B2"),("RAMP_DELTA","#E69F00")):
                rows=[r for r in tables["TEMPORAL"] if r["dataset"]==data and r["space"]==space and r["split"]==dev and r["mode"]=="F" and r["arm"]==arm and r["period"]=="horizon"];ax.plot([r["index"] for r in rows],[r["MSE_delta"] for r in rows],styles[arm],color=color,label=arm)
            ax.axhline(0,color="black",linewidth=.8);ax.axvline(16.5,color="gray",linestyle=":");ax.axvline(32.5,color="gray",linestyle=":");ax.set_title(data+" / "+space);ax.set_xlabel("Future horizon h (1..48)");ax.set_ylabel("MSE minus same F baseline");ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc="lower center",ncol=2);fig.suptitle("Exposed DEV: horizon-specific observed-channel macro error; negative is better; not full-macro averaging");save(fig,"02_horizon_error_delta")
    fig,axes=plt.subplots(2,3,figsize=(16,8))
    for di,(data,(_,_,dev)) in enumerate(DATASETS.items()):
        for sj,space in enumerate(spaces(data)):
            ax=axes[di,sj]
            for mode,offset in (("F",-.15),("FLAT",.15)):
                rows=[choose(tables["LATENT"],dataset=data,space=space,split=split,mode=mode) for split in ("fit","tune",dev)];ax.bar([i+offset for i in range(3)],[r.get("mse",float("nan")) for r in rows],width=.28,color=colors[mode],label=mode)
            ax.set_xticks(range(3),("FIT","TUNE","DEV"));ax.set_title(data+" / "+space);ax.set_ylabel("Latent point MSE");ax.grid(axis="y",alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc="lower center",ncol=2);fig.suptitle("Same complete-C target-horizon mask only; latent units differ across spaces; no C-output EY/oracle floor");save(fig,"03_latent_point_errors")
    fig,axes=plt.subplots(2,2,figsize=(17,10));spec=(("WHOLE_COST","paired_RAW_median_ratio",-.15,"o","#0072B2","Primary / RawF"),("WHOLE_COST_REPEAT","paired_RAW_median_ratio",-.05,"^","#0072B2","Repeat / RawF"),("WHOLE_COST","paired_matched_baseline_median_ratio",.05,"D","#D55E00","Primary / same space-mode"),("WHOLE_COST_REPEAT","paired_matched_baseline_median_ratio",.15,"s","#D55E00","Repeat / same space-mode"))
    for di,data in enumerate(DATASETS):
        for bi,batch in enumerate((1,8)):
            ax=axes[di,bi];rows=[choose(tables["WHOLE_COST"],dataset=data,route=route,batch=batch) for route in routes(data)]
            for table,field,offset,marker,color,label in spec:ax.scatter([i+offset for i in range(18)],[choose(tables[table],dataset=data,route=r["route"],batch=batch)[field] for r in rows],marker=marker,color=color,s=30)
            labels=[r["route"].replace("RAW_C","Raw").replace("PCA_K","P").replace("FLAT","L").replace("BASELINE","B").replace("STATIC_DELTA","S").replace("RAMP_DELTA","R").replace("_","/") for r in rows];ax.set_xticks(range(18),labels,rotation=65,ha="right",fontsize=8);ax.axhline(1,color="black",linestyle=":");ax.set_ylabel("Whole latency ratio");ax.set_title(data+" / B"+str(batch));ax.grid(axis="y",alpha=.2)
    fig.legend(handles=[Line2D([0],[0],marker=s[3],color=s[4],linestyle="",label=s[5]) for s in spec],loc="lower center",ncol=2,fontsize=10);fig.suptitle("Two preserved TUNE cost cohorts: paired medians; ratios of medians retained in separate CSVs; no fastest-run selection");save(fig,"04_whole_cost",.11)
    return files


def make_text(values,tables,bindings,output):
    rows=tables["SELECTED"];paired=[]
    for data,(_,_,dev) in DATASETS.items():
        for space in spaces(data):
            for arm in ARMS:
                f=choose(rows,dataset=data,space=space,mode="F",arm=arm);flat=choose(rows,dataset=data,space=space,mode="FLAT",arm=arm);paired.append(f["split_summary"][dev]["selected_MSE"]<flat["split_summary"][dev]["selected_MSE"])
    ramp=[r for r in rows if r["space"]!="RAW_C" and r["mode"]=="F" and r["arm"]=="RAMP_DELTA"];ramp_worse=sum(r["split_summary"][DATASETS[r["dataset"]][2]]["selected_MSE"]>r["split_summary"][DATASETS[r["dataset"]][2]]["baseline_MSE"] for r in ramp);zeros=sum(r["selected_kind"]=="ZERO_DELTA" for r in rows);latent_pairs=[]
    for data,(_,_,dev) in DATASETS.items():
        for space in spaces(data):
            for split in ("fit","tune",dev):latent_pairs.append(choose(tables["LATENT"],dataset=data,space=space,split=split,mode="F")["mse"]<choose(tables["LATENT"],dataset=data,space=space,split=split,mode="FLAT")["mse"])
    lines=["# 공통·horizon ramp 디코더 보정: 무엇이 개선됐고 무엇이 남았나","",f"[확인] 같은 dataset·입력 공간·보정 arm에서 실제 F 미래를 쓴 선정 예측은 FLAT 미래보다 DEV MSE가 낮은 조건이 {sum(paired)}/{len(paired)}다. 그러나 PCA의 F+RAMP는 원 F baseline보다 DEV가 나빠진 조건이 {ramp_worse}/{len(ramp)}다. 같은 추가 계수 수의 한 번의 고정 ramp가 압축 문제를 해결했다고 볼 수 없다. ZERO_DELTA를 고른 조건은 {zeros}/24이며 실제 실행 성공과 성능 개선을 구분한다.",""]
    for data,(_,_,dev) in DATASETS.items():
        raw=choose(tables["BASELINE_SCORES"],dataset=data,space="RAW_C",mode="F",split=dev)["all_observed"]["normalized"]["mse"];parts=[]
        for arm in ARMS:
            row=choose(rows,dataset=data,space="RAW_C",mode="F",arm=arm);parts.append(arm+f" {row['split_summary'][dev]['selected_MSE']:.12g} ({row['selected_candidate']})")
        lines.append(f"- {data}: 원 Raw F DEV MSE {raw:.12g}; "+"; ".join(parts)+".")
        for space in spaces(data)[1:]:
            static=choose(rows,dataset=data,space=space,mode="F",arm="STATIC_DELTA");ramp_row=choose(rows,dataset=data,space=space,mode="F",arm="RAMP_DELTA");s=static["split_summary"][dev];r=ramp_row["split_summary"][dev];lines.append(f"- {data} {space}: 같은 F baseline DEV {s['baseline_MSE']:.12g}, STATIC {s['selected_MSE']:.12g}, RAMP {r['selected_MSE']:.12g}; TUNE는 {static['selected_TUNE_MSE']:.12g}/{ramp_row['selected_TUNE_MSE']:.12g}다.")
    lines += ["","![선정 예측과 같은 baseline](figures/01_selected_accuracy.png)","","**질문과 방법.** PCA 기본K/2K의 E·center·원 RECON L7 D 및 실제 F cache를 고정했다. 추가 slope ΔD 하나의 크기는 7K×C이고, STATIC은 모든 미래에1, RAMP는 저장 index i=0..47에서 i/47을 곱한다. 첫 미래 h=1의 ramp 보정은0, h=48은1이다. 이것은 origin h=0이나48개 독립 D_h가 아니다. 두 arm의 free slope 수는 같지만 effective rank·shrinkage·함수집합까지 일치하지 않는다. F와 FLAT 네 arm은 같은 실제 F ungated FIT 특징의 uncentered RMS를 공유한다.","","FIT의 true-observed 원 C residual(Y32_as64−원 baseline32_as64)을 C별 SSE/N_c+α||표준화 slope||²로 적합했다. 절편·feature centering·시간 bias는 없다. alpha0는 design singular cutoff1e-8 최소노름, 양수α는 모든 singular 방향을 쓰며 fused ΔD64를FP32로 저장한다. 배포는 원 baseline32+FP32(g·phi)@ΔD32이고 D_old+ΔD를 먼저 합성한 산술로 바꾸지 않았다. ZERO는 원 baseline 배열을 그대로 복사한다. Raw F도 원 raw_forecast byte를 쓰며 currentC+(rawF−currentC)의 재조합은 FP32 cancellation 때문에 보조 진단이다.","","두 자료×Raw/PCA기본K/PCA2K×F/FLAT×STATIC/RAMP의24경로에서 alpha0 포함240후보와 ZERO24, 총264후보·528 FIT/TUNE 비교·선정 DEV24를 실행했다. 선택은 각 arm의 실제 FP32 TUNE all-observed normalized channel-macro MSE이며 정확 동률은 등록α 순서 뒤 ZERO다. 모든24개 성공/실패 잠금 뒤 DEV를 열었다. 과학 단계는 원 F cache를 재사용했고 새 F forward·E/D/native 학습·optimizer update는0이다. 실제 미래 residual로 ΔD를 적합한 감독학습이며, 재구성 목적만 학습했다고 설명하지 않는다.","","**무엇을 분리했나.** L7의 lag별 origin 참조와 알려진 과거 prefix 때문에 latent-last FLAT은 원 current-C persistence와 일반적으로 다르다. F가 FLAT보다 좋다는 관측은 이 보정 class 안의 미래 정보 기여 단서다. 모든 native나 채널 혼합 예측기보다 F가 반드시 필요하다는 증거는 아니다. Raw에도 같은 static/ramp 기회를 줬으며 Raw7C² 대 PCA7KC로 용량과 보존 정보가 다르다. Beijing의 Raw 보정 이득은 압축 고유 정확도 이득으로 세지 않는다.","","![horizon별 원 F 대비 오차](figures/02_horizon_error_delta.png)","","TEMPORAL은48개 h와 사전1–16/17–32/33–48 구간의 true-observed C macro 오차, 같은 mode baseline, signed MSE/MAE 차이를 보존한다. 각 h의 관측 채널 분모가 달라 h별 평균이 전체 all-observed macro와 같지 않다. 전체·complete-C·물리 단위 per-C 진단은 SCORES/BASELINE_SCORES에 구분했다. 시간 구간과 겹치는 창을 독립 dataset 또는 반복으로 세지 않았다.","",f"**표현 공간의 미래 오차.** point latent MSE에서 F가 FLAT보다 낮은 context는 {sum(latent_pairs)}/{len(latent_pairs)}다. 이는 각 공간 고유 단위이며 C normalized MSE와 직접 같은 척도로 비교하지 않는다. 원 C 전체가 관측된 target h만 E(Y)로 변환했고, 미관측 C vector를 안전한 latent 정답으로 취급하지 않았다. 이 단계는 C-output E(Y)나 새 joint output-space minimum을 계산하지 않았다. 옛 고정 D floor를 새 시간 보정 모델의 하한으로 재사용하지 않는다.","","![같은 complete-C point mask의 latent 오차](figures/03_latent_point_errors.png)","","**새 전체 비용.** 다음 수치는 이24개 선정 correction의 별도 cohort다. 원 physical CPU TUNE 첫 B1/B8에서 causal fill/FIT scaler→필요한 실제 CUDA E/F/old D→CPU static/ramp correction 및 필요한 latent 전송/lag 특징→physical CPU 반환·sync를 측정한다. FLAT은 F를 호출하지 않고, ZERO도 새 실제 baseline을 구한 뒤 복사한다. cached F를 반환한 과학 CPU 시간을 F 배포 비용으로 쓰지 않았다. 세팅·계수 적합·reference·추가 precheck·warmup·저장·외부 meter/hash·memory snapshot은 밖이고 내부 finite/shape 검사는 안이다.","",f"18routes×2자료×B1/B8의 {len(tables['WHOLE_COST'])}/72셀, {len(tables['WHOLE_COST_SAMPLES'])}/360 integer-ns 표본을 보존했다. warm2/실측5는18위치 전체 균형이 아닌5회전이며 같은 첫B창을 반복했다. Raw F 및 같은 space/mode baseline 각각에 대한 median paired ratio와 ratio of medians를 구분했다. matching FLAT baseline을 Bolt라고 부르지 않는다. 공유 F/4개 PCA E-D/24개 CPU correction의 RSS·VRAM은 방법별 독점 메모리가 아니다.",""]
    for data in DATASETS:
        for batch in (1,8):
            raw=choose(tables["WHOLE_COST"],dataset=data,route="RAW_C_F_BASELINE",batch=batch);repeat=choose(tables["WHOLE_COST_REPEAT"],dataset=data,route="RAW_C_F_BASELINE",batch=batch);selected_rows=[r for r in tables["WHOLE_COST"] if r["dataset"]==data and r["batch"]==batch and not r["route"].endswith("_BASELINE")];repeat_rows=[r for r in tables["WHOLE_COST_REPEAT"] if r["dataset"]==data and r["batch"]==batch and not r["route"].endswith("_BASELINE")];faster=sum(r["ratio_of_RAW_medians"]<1 for r in selected_rows);repeat_faster=sum(r["ratio_of_RAW_medians"]<1 for r in repeat_rows);lines.append(f"- {data} B{batch}: Raw F 원 실행 {raw['median_seconds_per_window']*1000:.12g}, 반복 {repeat['median_seconds_per_window']*1000:.12g} ms/window; 각각의 Raw F보다 median latency가 낮은 선정 보정 route {faster}/{len(selected_rows)}, {repeat_faster}/{len(repeat_rows)}. 정확도와 독립된 속도 비교다.")
    lines += ["","![원·반복 전체 비용의 paired ratios](figures/04_whole_cost.png)","",f"원 실행과 같은 source/선정 상태의 반복 실행도 별도 {len(tables['WHOLE_COST_REPEAT'])}/72셀·{len(tables['WHOLE_COST_REPEAT_SAMPLES'])}/360표본으로 보존했다. 그림은 두 실행의 paired medians를 비교하며 두 분모의 ratio-of-medians도 각각의 CSV에 그대로 있다. 빠른 실행을 골라 원값을 대체하지 않았고 세 번째 반복이나 새로운 계수 선정을 하지 않았다. 이전76셀 cohort와 Raw 시간이 달랐던 원인은 미확정이며 callback 차이나 방법 개선으로 확정하지 않는다. 다른 cohort의 유리한 Raw 절대시간을 분모로 차용하지 않았다.","","[판단] TUNE 이득이 이미 노출된 DEV로 모두 옮겨지지 않았고, PCA RAMP의 원 F 대비 일관된 해결은 관측되지 않았다. 한 번의 고정 ramp 결과로 full D_h·비선형 D·추가 학습 가능성을 모두 기각하지 않는다. 이번 진단 후 실패를 이유로 spline/bin/고차 함수를 자동 늘리지 않는다. adaptive 설계와 제한된2개 dataset·두 K의 결과이며 독립 일반화/분포 변화 원인 확정이 아니다. 보호 TEST0, strict K-only0이다. current C anchor와 K 과거 buffer를 포함하며, 과거 기반 배포와 FIT 미래 target 감독을 구분한다. 기존24D 결과·원 v1 76셀 비용·history ridge 규제 확장·그 실패 기록은 보존하고 이 묶음으로 대체하지 않는다.","","검산은 저장 산술·선정 계수·마스크·점수·source/SHA/영수증과 실제 비용 기록의 확인이다. F GPU·optimizer를 독립 재현한 실험이나 별도 holdout 검증이 아니다. 이 생성기는 JSON metadata만 읽으며 NPZ·모델·F·GPU·학습을 열거나 실행하지 않는다. RESOURCES의 science/audit/원cost와감사/반복cost와감사 CPU·process wall은 각 범위로 보존하며 학습시간 speedup으로 쓰지 않는다.",""]
    lines += ["원·반복 비용의 첫 v1 감사는 metadata 함수와 지역 변수의 이름 충돌(UnboundLocalError)로 실패했고, 원 source와 두 실패 영수증을 보존했다. 새 v2 감사는 지역 식별자 두 곳만 바꿔 원·반복 모두 PASS했다. 과학 예측·계수·비용 표본·수식·공차는 바꾸지 않았으며, 이 감사 수정 때문에 과학 실행이나 비용 측정을 다시 하지 않았다. 아래 failed_cost_audit_v1·failed_repeat_cost_audit_v1·failed_cost_auditor_v1_source 링크에 원 실패를 남긴다.",""]
    lines += [f"- [{key}.csv](tables/{key}.csv): {len(items)}행" for key,items in tables.items()]
    lines += [""]+[f"- [{key}]({rel(value['path'],output)}): `{value['sha256']}`" for key,value in bindings.items()]
    return "\n".join(lines)+"\n"


def generate(paths,destination):
    output=resolve(destination);require(output.parent==(HERE/"reports").resolve() and not output.exists(),"FRESH_LOCAL_REPORT_OUTPUT");values,bindings=load_verified(paths);tables=collect_tables(values);source=record(__file__);bindings["report_source"]=source;output.mkdir(parents=True,exist_ok=False);(output/"tables").mkdir()
    for key,rows in tables.items():write_csv(output/"tables"/(key+".csv"),rows,output)
    images=figures(tables,output);(output/"REPORT_KO.md").write_text(make_text(values,tables,bindings,output),encoding="utf-8",newline="\n");require(record(__file__)==source,"REPORT_SOURCE_UNCHANGED")
    for value in bindings.values():bound(value)
    manifest={"status":"COMPLETE","source":source,"inputs":bindings,"files":[{"path":path.relative_to(output).as_posix(),"sha256":sha(path),"bytes":path.stat().st_size} for path in sorted(output.rglob("*")) if path.is_file()],"table_row_counts":{key:len(rows) for key,rows in tables.items()},"PNG_count":4,"SVG_count":4,"scientific_NPZ_or_model_read":False,"new_fit_or_training_during_generation":0,"new_output_space_computed":False,"C_output_EY_computed":False,"source_F_cache_is_not_whole_cost":True,"original_v1_76_cell_cost_or_ridge_extension_mixed":False,"two_cost_cohorts_separately_preserved":True,"fastest_cost_cohort_selected":False,"protected_TEST_accessed":False,"independent_holdout":False,"strict_K_only":False}
    with (output/"MANIFEST.json").open("x",encoding="utf-8") as handle:json.dump(manifest,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write("\n")
    print(json.dumps({"status":"COMPLETE","manifest":str(output/"MANIFEST.json"),"images":len(images)}),flush=True);return manifest


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in (*FIXED_SHA,"output"):parser.add_argument("--"+key.replace("_","-"),dest=key,required=True)
    args=vars(parser.parse_args());destination=args.pop("output");generate(args,destination)
