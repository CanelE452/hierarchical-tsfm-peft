"""JSON/JSONL-only matched L7 native/Bolt report; no scientific payload or model reads."""
from pathlib import Path
import argparse
import json
import statistics

from make_linear_e_adaptation_report_v2 import require,resolve,sha,record,read,bound,same_opaque,choose,rel,write_csv

HERE=Path(__file__).resolve().parent
STATES=("RECON_CP0","E_ONLY_s92601","E_ONLY_s92602")
SEEDS=(92601,92602)
ROUTES=("RAW_BOLT",)+tuple(r for s in STATES for r in (s+"_BOLT",s+"_NATIVE_s92601",s+"_NATIVE_s92602"))
SCALES=("normalized","physical_per_channel_diagnostic")
TABLES=("SCORES","PAIRED","CHECKPOINTS","UPDATES","CONDITIONS","PROBES","SPACE_CELLS","SPACE_METHODS","CONTEXT_PARITY","FAILURES","WHOLE_COST","WHOLE_COST_SAMPLES","RESOURCES")
FIXED_SHA={"train":"4a6f2d3601c27bf000659aa99a586a2f04d72371766f7558cc78cc16028caa2e","audit":"429296ac7346189953813292291584d0c5ceb0a60028b4c9b955a2a429a1db87","cost":"6fce6c6a462cb0b45f6cc3932f94c615ad02b17892c6aade642bc2ad39a6b51c","cost_audit":"cd8716119609ad5a3148d7f9c620fb08b938f9a5d36f61ba7f111dc3ca880983"}
UTILITY={"path":str(HERE/"make_linear_e_adaptation_report_v2.py"),"sha256":"32e2460ea4724144067f264ab1ceb0332cd04c5fce2296838024c119c9572466"}
AUDIT_SOURCES={"audit":{"path":str(HERE/"audit_matched_l7_native_v1.py"),"sha256":"4930f1525fbcff3fad80d0358a017f02cbb22a97796a2841c2137b80e8a6b059"},"cost_audit":{"path":str(HERE/"audit_matched_l7_native_cost_v1.py"),"sha256":"dc6c136f9122ec882e6fbf5b24946f847057709f1c613874cfee1b23b51577fe"}}
AUDIT_COUNTS={"native_fits_checked":6,"checkpoint_states_checked":30,"checkpoint_TUNE_predictions_checked":30,"selected_FIT_DEV_predictions_checked":12,"logged_updates_checked":3600,"recorded_post_optimizer_frozen_flag_checks":3600,"CP0_probe_cases_checked":6,"CP0_latent_persistence_byte_checks":6,"frozen_state_CP_member_byte_comparisons_checked":90,"matched_Bolt_decodes_checked":9,"E0_new_RECON_C_decodes_checked":3,"selected_E_only_Bolt_C_byte_checks":6,"cached_Bolt_source_contexts_checked":9,"native_cached_latent_history_byte_comparisons_checked":42,"training_score_fields_checked":1404,"new_space_diagnostic_cells_computed":12,"new_space_LS_SVD_crosschecks_computed":12,"selection_locks_checked":6,"protected_source_and_input_file_hashes_checked":1784}
COST_COUNTS={"cost_cells_checked":20,"recorded_timer_samples_checked":100,"selected_full_state_hashes_checked":9,"selected_native_head_hashes_checked":6,"selection_locks_checked":6,"physical_prechecks_checked":9,"physical_first8_windows_checked":72,"native_first8_head_equations_checked":6,"canonical_TUNE_bridges_checked":1,"protected_prior_file_hashes_checked":1758}


def selected(row):
    require([p["update"] for p in row["checkpoints"]]==[0,100,200,400,600],"FIVE_NATIVE_CP")
    cp=min(row["checkpoints"],key=lambda p:(p["tune"]["mse"],p["update"])); require(cp["update"]==row["selected_update"],"TUNE_ONLY_EARLIEST_CP"); same_opaque(cp["state"],row["selected_state"]); return cp


def load_verified(paths):
    require(set(paths)==set(FIXED_SHA),"FOUR_EXACT_INPUTS")
    require(all(isinstance(v,str) and len(v)==64 for v in FIXED_SHA.values()) and all(isinstance(v,dict) for v in AUDIT_SOURCES.values()) and bool(COST_COUNTS),"ACTUAL_MATCHED_NATIVE_COST_AUDIT_NOT_YET_REGISTERED")
    require(all(Path(p).suffix==".json" for p in paths.values()),"FOUR_JSON_METADATA_INPUTS_ONLY")
    bound(UTILITY); values={k:read(p) for k,p in paths.items()}; bindings={k:record(p) for k,p in paths.items()}
    for key,value in values.items(): require(bindings[key]["sha256"]==FIXED_SHA[key] and value["status"]==("PASS" if "audit" in key else "COMPLETE") and value["protected_TEST_accessed"] is False,"EXACT_COMPLETE_PASS_NO_TEST:"+key)
    train,audit,cost,costaudit=(values[k] for k in FIXED_SHA); bound(audit["diagnostic"],bindings["train"]); bound(costaudit["cost"],bindings["cost"])
    require(audit["training_verification_status"]=="PASS" and audit["new_space_diagnostic_status"]=="COMPLETE" and not audit["partial_paths"] and not audit["new_output_space_diagnostic_failures"],"SAVED_TRAINING_AND_TWELVE_NEW_GEOMETRY_CELLS_COMPLETE")
    for key,counts in (("audit",AUDIT_COUNTS),("cost_audit",COST_COUNTS)):
        require(all(values[key]["totals"].get(k)==v for k,v in counts.items()),"EXACT_INDEPENDENT_COVERAGE"); bound(values[key]["audit_source"],AUDIT_SOURCES[key]); bindings[key+"_source"]=values[key]["audit_source"]
    for value in (cost,costaudit):
        bound(value["matched_l7_native_diagnostic"],bindings["train"]); bound(value["matched_l7_native_audit"],bindings["audit"]); bound(value["selection_registry"],train["selection_registry"])
    require(train["native_training_fits"]==train["resolved_selection_locks"]==6 and train["training_updates_completed"]==3600 and train["checkpoint_states_saved"]==30 and train["all_6_locks_before_DEV"] is True and train["matched_Bolt_C_decodes"]==train["cached_Bolt_contexts_reused"]==9 and train["E0_new_matched_RECON_C_decodes"]==3 and train["selected_E_only_Bolt_C_byte_checks"]==6 and not train["failed_paths"],"SIX_FRESH_HEADS_NINE_REUSED_BOLT_CONTEXTS")
    for key in ("new_F_forward_calls","new_F_model_objects_loaded","new_E_training_updates","new_D_training_updates","probe_optimizer_updates"): require(train[key]==0,"NO_NEW_F_OR_E_D_TRAINING:"+key)
    for key in ("new_backbone_training","whole_deployment_cost_measured","new_output_space_diagnostics_computed_by_training_producer","independent_holdout","strict_K_only"): require(train[key] is False,"EXPOSED_MATCHED_HEAD_SCOPE:"+key)
    require(tuple(cost["routes"])==ROUTES and cost["cost_split"]=="tune" and cost["completed_cost_cells"]==20 and cost["recorded_timer_samples"]==100 and not cost["failed_paths"] and cost["native_predictor_loaded_or_timed"] is True and cost["cached_future_returned_instead_of_actual_F_inside_timer"] is False and cost["new_training_updates"]==0 and cost["F_before"]==cost["F_after"]==train["F_parent_signatures_reused"]["before"]==train["F_parent_signatures_reused"]["after"],"LIVE_TEN_ROUTE_SAME_CUDA_COST")
    require(cost["selected_states_before"]==cost["selected_states_after"] and cost["native_heads_before"]==cost["native_heads_after"] and len(cost["selected_states_after"])==9 and len(cost["native_heads_after"])==6 and all(cost[k] is False for k in ("new_encoder_decoder_fit","new_CP_seed_or_lag_selection","new_backbone_training","independent_holdout","strict_K_only")),"NINE_FROZEN_ADAPTERS_AND_SIX_HEADS_NO_NEW_SELECTION_OR_HOLDOUT")
    registry=read(bound(train["selection_registry"])); require(registry["status"]=="ALL_SIX_RESOLVED_BEFORE_DEV" and len(registry["conditions"])==6 and registry["DEV_opened_before_seal"] is False,"ALL_SIX_LOCKS_BEFORE_DEV")
    for item in registry["conditions"]:
        row=train["states"][item["state"]]["native"][str(item["seed"])]; lock=read(bound(item["selection_lock"],row["selection_lock"])); require(lock["status"]==row["status"]==item["status"]=="COMPLETE" and lock["DEV_opened_before_lock"] is False and lock["selected_update"]==selected(row)["update"],"SUCCESSFUL_TUNE_ONLY_LOCK")
    for key in ("E_only_diagnostic","E_only_audit","joint_recon_diagnostic","joint_recon_audit","full_gradient_parent","full_gradient_parent_audit","causal_lag_diagnostic","causal_lag_audit","parent_report","parent_report_manifest","parent_report_audit","selection_registry","plan"):
        bound(train[key]); bound(audit[key],train[key]); bindings[key]=train[key]
    values["science"]=read(train["causal_lag_diagnostic"]["path"]); require(values["science"]["status"]=="COMPLETE","ACCEPTED_RAW_BASELINE_METADATA")
    bindings.update(cost_plan=cost["cost_plan"],controller_source=train["controller_source"],native_source=train["native_source"],report_utility=UTILITY)
    for item in bindings.values(): bound(item)
    return values,bindings


def collect_tables(values):
    train,audit,cost=(values[k] for k in ("train","audit","cost")); out={k:[] for k in TABLES}
    def scores(base,item):
        for mask,field in (("all_observed","scores_all_observed"),("complete_C","scores_complete_mask")):
            for scale in SCALES: out["SCORES"].append({**base,"mask":mask,"scale":scale,**item[field][scale],"artifact":item.get("artifact")})
    for state in STATES:
        owner=train["states"][state]
        for split in ("fit","tune","val"): scores({"state":state,"predictor":"BOLT","seed":0,"split":split,"new_head_fit":False},owner["Bolt"][split])
        for seed in SEEDS:
            row=owner["native"][str(seed)]; cp=selected(row); base={"state":state,"predictor":"NATIVE","seed":seed,"new_head_fit":True}
            for split,item in (("fit",row["FIT"]),("tune",cp),("val",row["DEV"])): scores({**base,"split":split},item)
            out["CHECKPOINTS"] += [{**base,**p,"selected":p["update"]==cp["update"]} for p in row["checkpoints"]]
            log=bound(row["updates_log"]); require(log.suffix==".jsonl","JSONL_METADATA_ONLY"); events=[json.loads(line,parse_constant=lambda value: (_ for _ in ()).throw(ValueError("NONFINITE_LOG:"+value))) for line in log.read_text(encoding="utf-8").splitlines()]
            require([p["update"] for p in events]==list(range(1,601)) and sum(p["clipped"] for p in events)/600==row["global_clipping_fraction"],"SIX_HUNDRED_UPDATES_AND_CLIP_FRACTION"); out["UPDATES"] += [{**base,**p} for p in events]
            keys=("selected_update","selected_state","selected_head_state_sha256","order_sha256","updates_log","selection_lock","loss","FIT_channel_observed_counts","global_clipping_fraction","post_optimizer_frozen_flag_checks","frozen_state_checkpoint_byte_checks","final_frozen_status","training_head_values_changed","selected_head_nonzero","frozen_bytes_checked_at_CP_and_final_only","independent_per_update_frozen_bytes_replayed","gradient_clipping_parameter_scope")
            out["CONDITIONS"].append({**base,**{k:row[k] for k in keys},"frozen_state_source":owner["frozen_state_source"],"E_previously_supervised_by_FIT_future":owner["E_previously_supervised_by_FIT_future"],"D_previously_supervised_by_FIT_future":owner["D_previously_supervised_by_FIT_future"]}); out["PROBES"].append({**base,**row["probe"]})
            for split in ("tune","val"):
                cell=audit["new_output_space_diagnostics"][state][str(seed)][split]; where={"state":state,"seed":seed,"split":split}; require(cell["status"]=="COMPLETE" and set(cell["methods"])=={"ACTUAL_NATIVE","EY","FLAT"} and cell["all_real_latent_or_all_FP32_lower_bound_claimed"] is False and cell["independent_external_replay_of_new_diagnostic_claimed"] is False,"NATIVE_GEOMETRY_LABEL_AND_NUMERIC_SCOPE")
                keys=("minimum_normalized_MSE","complete_target_positions","target_positions","joint_unknown_future_coordinates","joint_output_coordinates_before_mask","lag_slope_parameters_per_C","joint_metadata","minimum_solution_scores_complete_C","RAW_BOLT_scores_complete_mask","current_C_persistence_scores_complete_mask","selected_E_previously_supervised_by_FIT_future_targets","selected_D_previously_supervised_by_FIT_future_targets","selected_native_head_previously_supervised_by_FIT_future_targets","native_head_training_procedure_used_FIT_future_targets","EY_encoding_backend","EY_incomplete_C_horizons_filled_with_current_C","artifact","selected_state","selected_forecast_source","input_source")
                summary={**where,**{k:cell[k] for k in keys},"computed_during_current_audit":True,"RAW_complete_MSE":cell["RAW_BOLT_scores_complete_mask"]["normalized"]["mse"],"matched_Bolt_complete_MSE":owner["Bolt"][split]["scores_complete_mask"]["normalized"]["mse"]}
                summary.update({label:cell["methods"][method]["scores_complete_mask"]["normalized"]["mse"] for label,method in (("native_complete_MSE","ACTUAL_NATIVE"),("EY_complete_MSE","EY"),("FLAT_complete_MSE","FLAT"))}); out["SPACE_CELLS"].append(summary); out["SPACE_METHODS"] += [{**where,"method":method,**entry} for method,entry in cell["methods"].items()]
    raw=values["science"]["datasets"]["robin"]["spaces"]["PCA"]["conditions"]["L7_RECON"]["splits"]
    for split in ("fit","tune","val"): scores({"state":"RAW_C","predictor":"RAW_BOLT","seed":0,"split":split,"new_head_fit":False},raw[split]["RAW_BOLT"])
    for state in STATES:
        for seed in SEEDS:
            for split in ("fit","tune","val"):
                for mask in ("all_observed","complete_C"):
                    for scale in SCALES:
                        where={"state":state,"split":split,"mask":mask,"scale":scale}; a=choose(out["SCORES"],**where,predictor="NATIVE",seed=seed); b=choose(out["SCORES"],**where,predictor="BOLT",seed=0); r=choose(out["SCORES"],state="RAW_C",split=split,mask=mask,scale=scale,predictor="RAW_BOLT",seed=0)
                        out["PAIRED"].append({**where,"seed":seed,"native_MSE":a["mse"],"Bolt_MSE":b["mse"],"RAW_MSE":r["mse"],"native_minus_Bolt_MSE":a["mse"]-b["mse"],"native_minus_RAW_MSE":a["mse"]-r["mse"],"native_MAE":a["mae"],"Bolt_MAE":b["mae"],"RAW_MAE":r["mae"],"native_minus_Bolt_MAE":a["mae"]-b["mae"]})
    out["CONTEXT_PARITY"]=[dict(p) for p in audit["optional_NumPy_context_checks"]]
    for stage,field in (("train","failed_paths"),("audit","partial_paths"),("audit","new_output_space_diagnostic_failures"),("cost","failed_paths")): out["FAILURES"] += [{"stage":stage,"collection":field,**p} for p in values[stage][field]]
    rows=cost["cost"]["robin"]["rows"]; require(len(rows)==20 and {(r["route"],r["batch"]) for r in rows}=={(route,b) for route in ROUTES for b in (1,8)},"EXACT_TEN_ROUTES_TWO_BATCHES")
    for row in rows:
        raw_row=choose(rows,route="RAW_BOLT",batch=row["batch"]); matched=choose(rows,route=row["matched_Bolt_route"],batch=row["batch"]); sec=row["seconds"]
        require(row["status"]=="COMPLETE" and len(row["samples"])==len(sec)==5 and row["median_seconds"]==statistics.median(sec) and row["median_seconds_per_window"]==statistics.median(sec)/row["batch"],"FIVE_WHOLE_TIMER_SAMPLES")
        for field,denominator in (("RAW",raw_row),("matched_Bolt",matched)):
            ratios=[a/b for a,b in zip(sec,denominator["seconds"])]; require(row["paired_"+field+"_median_ratio"]==statistics.median(ratios) and row["ratio_of_"+field+"_medians"]==statistics.median(sec)/statistics.median(denominator["seconds"]),"DISTINCT_PAIRED_AND_RATIO_OF_MEDIANS")
        out["WHOLE_COST"].append({k:v for k,v in row.items() if k!="samples"})
        for sample in row["samples"]:
            require(type(sample["perf_start_ns"]) is type(sample["perf_end_ns"]) is int and (sample["perf_end_ns"]-sample["perf_start_ns"])/1e9==sample["seconds"],"INTEGER_NS_TIMER_DURATION"); out["WHOLE_COST_SAMPLES"].append(sample)
    out["RESOURCES"]=[{"stage":key,"new_deployment_timing":key=="cost",**values[key]["resource" if key=="audit" else "resource_terminal"]} for key in FIXED_SHA]
    require([len(out[k]) for k in TABLES]==[120,72,30,3600,6,6,12,36,3,0,20,100,4],"FULL_MATCHED_NATIVE_TABLE_COUNTS")
    return out


def figures(tables,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder=output/"figures"; folder.mkdir(); files=[]; labels=("E0: PCA / RECON CP0","E1: selected E / s92601","E2: selected E / s92602")
    def save(fig,name,bottom=.14):
        fig.tight_layout(rect=(0,bottom,1,.91))
        for ext in ("png","svg"):
            path=folder/(name+"."+ext); fig.savefig(path,dpi=160,bbox_inches="tight"); files.append(path)
        plt.close(fig)
    def error(state,predictor,seed,split,mask="all_observed"):
        return choose(tables["SCORES"],state=state,predictor=predictor,seed=seed,split=split,mask=mask,scale="normalized")["mse"]
    fig,axes=plt.subplots(1,2,figsize=(13,5.6))
    for ax,split in zip(axes,("tune","val")):
        for predictor,seed,color,marker,label in (("BOLT",0,"#D55E00","s","Frozen Bolt / matched E,D"),("NATIVE",92601,"#0072B2","o","Native head / order 92601"),("NATIVE",92602,"#56B4E9","^","Native head / order 92602")):
            ax.plot(range(3),[error(s,predictor,seed,split) for s in STATES],marker=marker,color=color,label=label)
        ax.axhline(error("RAW_C","RAW_BOLT",0,split),color="black",linestyle="--",label="Raw-C Bolt"); ax.set_xticks(range(3),labels,rotation=12); ax.set_title("TUNE selection" if split=="tune" else "Exposed DEV"); ax.set_ylabel("All-observed normalized MSE"); ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=2); fig.suptitle("Six fresh independent-affine heads at three frozen E/D states; lower MSE is better"); save(fig,"01_actual_errors")
    fig,axes=plt.subplots(2,1,figsize=(13,8.7),gridspec_kw={"height_ratios":[2,1]}); colors=("#0072B2","#56B4E9","#D55E00"); names=[]; clips=[]
    for si,state in enumerate(STATES):
        for seed,style in zip(SEEDS,("-","--")):
            rows=[r for r in tables["CHECKPOINTS"] if r["state"]==state and r["seed"]==seed]; label="E"+str(si)+" / "+str(seed); names.append(label)
            axes[0].plot([r["update"] for r in rows],[r["tune"]["mse"] for r in rows],marker="o",color=colors[si],linestyle=style,label=label); cp=next(r for r in rows if r["selected"]); axes[0].scatter(cp["update"],cp["tune"]["mse"],s=150,marker="*",color=colors[si],zorder=5)
            clips.append(choose(tables["CONDITIONS"],state=state,seed=seed)["global_clipping_fraction"])
    axes[0].set(xlabel="Optimizer update; stars mark TUNE-selected CP",ylabel="TUNE normalized MSE"); axes[0].grid(alpha=.2); axes[1].bar(range(6),clips,color=[color for color in colors for _ in SEEDS]); axes[1].set_xticks(range(6),names); axes[1].set(ylabel="Clipped update fraction",ylim=(0,1.13)); axes[1].grid(axis="y",alpha=.2)
    for index,value in enumerate(clips): axes[1].text(index,value+.025,f"{value:.4f}",ha="center",va="bottom")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=3); fig.suptitle("3,600 actual updates / 30 CPs; clipping is recorded evidence, not a convergence diagnosis"); save(fig,"02_CP_and_clipping",.12)
    fig,axes=plt.subplots(1,2,figsize=(14,5.8)); names=["E"+str(i)+"/"+str(s)[-2:] for i in range(3) for s in SEEDS]
    for ax,split in zip(axes,("tune","val")):
        rows=[choose(tables["SPACE_CELLS"],state=state,seed=seed,split=split) for state in STATES for seed in SEEDS]
        for key,color,marker,label in (("native_complete_MSE","#0072B2","o","Actual native head"),("matched_Bolt_complete_MSE","#D55E00","s","Matched frozen Bolt"),("EY_complete_MSE","#CC79A7","x","E(Y): known-future diagnostic"),("minimum_normalized_MSE","#009E73","*","Retained numeric minimum")):
            ax.plot(range(6),[r[key] for r in rows],color=color,marker=marker,label=label)
        ax.axhline(rows[0]["RAW_complete_MSE"],color="black",linestyle="--",label="Raw / same complete-C"); ax.set_xticks(range(6),names); ax.set_title("TUNE" if split=="tune" else "Exposed DEV"); ax.set_ylabel("Complete-C normalized MSE"); ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=3); fig.suptitle("12 diagnostic cells reuse 3 frozen E/D states and 2 splits; low minima do not prove predictability"); save(fig,"03_output_space_diagnostics",.16)
    fig,axes=plt.subplots(2,2,figsize=(15,11),gridspec_kw={"width_ratios":[1.2,1]})
    short={"RAW_BOLT":"Raw-C Bolt"}
    for index,state in enumerate(STATES):
        short[state+"_BOLT"]="E"+str(index)+" / Bolt"
        for seed in SEEDS: short[state+"_NATIVE_s"+str(seed)]="E"+str(index)+" / Native "+str(seed)[-2:]
    for ri,batch in enumerate((1,8)):
        ax,ratio_ax=axes[ri]; rows=[choose(tables["WHOLE_COST"],route=route,batch=batch) for route in ROUTES]
        for index,row in enumerate(rows):
            color="black" if row["route"]=="RAW_BOLT" else "#0072B2" if "_NATIVE_" in row["route"] else "#D55E00"; ax.barh(index,row["median_seconds_per_window"]*1000,color=color,alpha=.65); ax.scatter([s/batch*1000 for s in row["seconds"]],[index]*5,color=color,s=16,alpha=.65,zorder=3)
        ax.set_yticks(range(10),[short[r["route"]] for r in rows]); ax.invert_yaxis(); ax.set(xlabel="Whole milliseconds per window; bars=median, points=5 samples",title="Same CUDA backend / B"+str(batch)); ax.grid(axis="x",alpha=.2)
        native=[r for r in rows if "_NATIVE_" in r["route"]]
        for key,color,marker,label in (("paired_RAW_median_ratio","#0072B2","o","Median paired ratio / Raw"),("paired_matched_Bolt_median_ratio","#D55E00","s","Median paired ratio / same-E Bolt")):
            ratio_ax.scatter([r[key] for r in native],range(6),color=color,marker=marker,label=label)
        ratio_ax.axvline(1,color="black",linestyle="--"); ratio_ax.set_yticks(range(6),[short[r["route"]] for r in native]); ratio_ax.invert_yaxis(); ratio_ax.set(xlabel="Ratio below 1 is faster; paired ratios, not ratio of medians",title="B"+str(batch)+" / two explicit denominators"); ratio_ax.grid(axis="x",alpha=.2)
    fig.legend(*axes[0,1].get_legend_handles_labels(),loc="lower center",ncol=2); fig.suptitle("20 whole-pipeline conditions / 100 samples; first-B repeats and shared resident memory"); save(fig,"04_same_CUDA_whole_cost",.08)
    return files


def make_text(values,tables,bindings,output):
    train,audit,cost=(values[k] for k in ("train","audit","cost"))
    def paired(state,seed,split): return choose(tables["PAIRED"],state=state,seed=seed,split=split,mask="all_observed",scale="normalized")
    worse={split:sum(paired(state,seed,split)["native_MSE"]>paired(state,seed,split)["Bolt_MSE"] for state in STATES for seed in SEEDS) for split in ("tune","val")}; raw_better=sum(paired(state,seed,"val")["native_MSE"]<paired(state,seed,"val")["RAW_MSE"] for state in STATES for seed in SEEDS)
    cp100=sum(r["selected_update"]==100 for r in tables["CONDITIONS"]); clips=[r["global_clipping_fraction"] for r in tables["CONDITIONS"]]
    lines=["# 고정 L7 E/D에서 Bolt와 단순 잠재 예측기 비교","",f"[확인] 같은 세 E/D에서 새 native head는 frozen Bolt보다 TUNE {worse['tune']}/6, 기노출 DEV {worse['val']}/6 조건에서 정규화 MSE가 높았다. Raw보다 DEV MSE가 낮은 native는 {raw_better}/6이다. 현재 비교는 Raw gap을 해결하지 못했고, 이 K별 독립 affine512→48 head가 Bolt를 대체한다는 근거도 얻지 못했다.","","세 상태는 원 PCA E/과거 RECON D인 E0, 이전 E-only 선정 E1(seed92601/CP400), E2(seed92602/CP600)다. 같은 D·center·C17/K4·L512/H48·lag [0,1,2,4,8,16,32]·현재 C anchor·FIT256/TUNE64/기노출 VAL30·scaler/origin/mask를 썼다. E1/E2는 FIT 미래 actual-C 감독과 Bolt를 통한 최적화, TUNE 선정 이력이 있고 E0는 이 E 학습이 없다. D는 모두 FIT 과거 재구성으로 고정됐다. native는 각 상태에서 fresh head 두 개를 FIT 미래 actual-C loss로 학습했다. 선정 head 감독 이력과 고정 E/D 감독 이력을 나눠 기록한다.",""]
    for state in STATES:
        for seed in SEEDS:
            row=choose(tables["CONDITIONS"],state=state,seed=seed); t,d=(paired(state,seed,split) for split in ("tune","val"))
            lines.append(f"- {state} / {seed}: CP{row['selected_update']}; TUNE native {t['native_MSE']:.10f}, Bolt {t['Bolt_MSE']:.10f}; DEV native {d['native_MSE']:.10f}, Bolt {d['Bolt_MSE']:.10f}, Raw {d['RAW_MSE']:.10f}.")
    lines += ["","![실제 점수](figures/01_actual_errors.png)","","**무엇을 새로 했는가.** 새 6fit×600=3,600 update·30CP·선정 FIT/DEV12예측이고 E/D/F 학습은0이다. E-only 두 상태를 만들었던 이전 학습을 새 6fit에 합산하거나 새 독립 조건으로 세지 않는다. 두 seed는 zero-init의 다른 minibatch 순서이지 독립 가중치 초기화가 아니다. NativeHead는 K별 독립 affine512→48만 표현하므로 모든 native 예측기의 대표가 아니다. E1/E2는 Bolt로 이미 최적화돼 있어 같은 학습 이력이나 같은 전체 학습 계산량 대조도 아니다.","","head만 AdamW lr.001/WD0/clip1/B4로 학습했고 고정 전체 FIT 채널 관측수 가중 actual-C macro 정규화 MSE를 사용했다. 과거 native의 minibatch 관측수 분모와 같은 손실이라고 쓰지 않는다. E encoding은 학습 B4, CP/최종 평가 CUDA FP32 canonical32다. 같은 seed의600×4 순서를 이전 E-only와 대조했고 CP0/100/200/400/600 전체 TUNE MSE 최소·이른 tie를 사용했다. 모든6개 성공/실패 잠금·registry 뒤 DEV를 읽었다.","",f"[확인] 선정 CP100은 {cp100}/6이고 clipping fraction은 {min(clips):.6f}~{max(clips):.6f}다. 후기 checkpoint의 TUNE 악화와 높은 clipping은 현재 최적화 경로의 관측이며 수렴 완료, 학습 부족, 모든 예측기 불가능성 중 하나를 확정하지 않는다. CP0 잠재 미래는 latent-last와 byte-exact지만 최종 C는 현재 C persistence와 같지 않다. L7의 알려진 과거 prefix뿐 아니라 각 lag에서 빼는 기준 z_past[-1-l]이 서로 다르기 때문이다. prefix가 모두 끝난 뒤에도 C_current + sum_l (z_last - z_past[-1-l]) D_l의 plateau가 남을 수 있다.","","![Checkpoint와 clipping](figures/02_CP_and_clipping.png)","","**비교 입력과 실제 F 계보.** 기존 실제 Torch F context9개를 재사용했고 과학 실행의 새 F forward는0이다. E0 FIT/DEV는 기존 FORECAST-D 실험의 latent_history/future만 가져와 현 RECON D로 새 C decode를 했다. 그 이전 C 예측을 주 비교로 가져오지 않았다. E0 TUNE까지 새 matched RECON C decode3개, E-only C byte 보존6개를 구분한다. 원 NumPy RECON과 새 Torch 문맥 비교는 별도 보존하며 실패를 주 gate로 바꾸거나 공차를 늘리지 않는다.","",f"NumPy 문맥 검사 {train['optional_NumPy_context_parity_checks']}개, 기록된 문맥 실패 {len(train['optional_NumPy_context_parity_failures'])}개다. 새 head의 모든 CP·선정 FIT/DEV의 latent history는 같은 고정 E에 대한 원 Bolt 캐시와 byte-exact로 대조됐다. zero head 미래·독립 NumPy affine 미래·lag D 산술·원수치/마스크 점수를 검산했다. 기록 per-step frozen flag와 CP/final E/D/center bytes 검사를 구분하며 매 update bytes·loss·backward·optimizer를 독립 재현한 검사는 아니다.","","**출력 공간과 남은 예측 격차.** 새 공간12cell은 감사 중 새 계산이며 3개 E/D×2split을 두 head에 반복한 것이므로 12개 독립 자료가 아니다. actual native·같은 E의 Bolt·E(Y)·FLAT·현재 C persistence·minimum을 같은 complete-C로 비교한다. E(Y)/minimum은 미래 정답을 쓰는 진단이며 fit/CP/배포에 들어가지 않았다. 두 동결 SVD/LS 수학 경로 교차이지 세 번째 외부 재현은 아니다.",""]
    tune_spaces=[choose(tables["SPACE_CELLS"],state=state,seed=SEEDS[0],split="tune") for state in STATES]; dev_spaces=[choose(tables["SPACE_CELLS"],state=state,seed=SEEDS[0],split="val") for state in STATES]
    lines += [f"[확인] 같은 complete-C에서 TUNE retained numeric minimum이 Raw보다 높은 고정 E/D는 {sum(r['minimum_normalized_MSE']>r['RAW_complete_MSE'] for r in tune_spaces)}/3이고, DEV minimum이 Raw보다 낮은 고정 E/D는 {sum(r['minimum_normalized_MSE']<r['RAW_complete_MSE'] for r in dev_spaces)}/3이다. 현재 TUNE의 등록 retained 공간 자체가 Raw에 못 미치는 조건과 DEV 공간에 여지가 있는 조건을 구분해야 한다. 이는 모든 실수/FP32 미래의 보편 하한이나 과거 예측 가능성의 증명이 아니다.",""]
    for row in tables["SPACE_CELLS"]:
        lines.append(f"- {row['state']} / {row['seed']} / {row['split']}: native {row['native_complete_MSE']:.10f}, Bolt {row['matched_Bolt_complete_MSE']:.10f}, E(Y) {row['EY_complete_MSE']:.10f}, minimum {row['minimum_normalized_MSE']:.10f}, Raw {row['RAW_complete_MSE']:.10f}; complete {row['complete_target_positions']}/{row['target_positions']}.")
    lines += ["","![출력 공간 진단](figures/03_output_space_diagnostics.png)","","minimum은 D32·known-past offset·complete-C 출력 행·모든 HK 미래 좌표·rcond1e-15 retained numeric space에 조건부인 정규화 MSE 해다. 모든 실수/FP32 latent나 모든 예측기의 보편 하한이 아니며 낮은 minimum이 과거 예측 가능성을 보장하지 않는다. MAE·물리 단위 값은 그 MSE 해의 점수이지 각 지표의 minimum이 아니다. real/FP32 false-bound flags를 그대로 보존한다. E(Y)의 불완전 C horizon은 현재 C로 채우며 L7의 뒤 출력에도 영향을 줄 수 있다. FLAT latent와 현재 C persistence를 같은 방법으로 세지 않는다.","","**실제 전체 비용.** 과학 캐시 재사용과 달리 이번 비용은 실제 frozen Bolt 또는 선정 native head를 실행한다. 같은 CUDA backend에서 원 물리 TUNE 첫B의 causal fill/FIT 정규화→GPU E→actual predictor→Torch lag D/current C anchor→물리 CPU 반환/sync를 포함했다. Robin B1/B8×10경로=20셀, 명시적 warmup2 뒤5순환100표본이다. 명시적 warmup2는 모든 비측정 forward가2회라는 뜻이 아니다. 5순환은 10개 위치의 완전 균형이 아니다. 고정 D는 준비 단계이며 per-origin fit은 없다. setup/학습/선정·reference·외부 meter/hash/guard·warmup·저장·memory snapshot은 timer 밖이고 whole_predict와 LinearLagAdapter 내부 finite/shape 검사·sync는 timer 안에 남아 있다. CPU 최적 native나 가능한 모든 backend의 최솟값을 측정하지 않았다.","","Raw와 같은 E의 Bolt를 각각 분모로 median of paired ratios와 ratio of medians를 따로 제시한다. 같은 첫B 반복·5개 timing 표본·모든 모델 공유 resident 메모리를 독립 자료나 독점 peak로 바꾸지 않는다. 다른 cohort의 절대 timing을 섞거나 빠른 실행만 고르지 않는다. 지연이 낮아져도 정확도 실패를 해결한 것으로 해석하지 않는다.",""]
    for row in tables["WHOLE_COST"]:
        lines.append(f"- B{row['batch']} {row['route']}: {row['median_seconds_per_window']*1000:.6f} ms/window; Raw paired {row['paired_RAW_median_ratio']:.6f}, medians {row['ratio_of_RAW_medians']:.6f}; same-E Bolt paired {row['paired_matched_Bolt_median_ratio']:.6f}, medians {row['ratio_of_matched_Bolt_medians']:.6f}.")
    lines += ["","![실제 전체 비용](figures/04_same_CUDA_whole_cost.png)","","[판단] 이번 근거는 이 고정 E/D와 이 독립 affine head에서 Bolt가 더 낮은 오차였다는 범위다. native가 나빠도 Bolt가 필수이거나 단순 예측기가 모두 불가능하다는 뜻은 아니다. 양쪽 모두 Raw gap이 남는다. 이 head/손실/최적화 조건의 TUNE 곡선과 예측 격차를 함께 보고 다음 원인을 좁혀야 한다. 기존 native·joint-Y/RF3·잠재 미래 회귀를 처음 한 실험처럼 쓰지 않는다. 이미 노출된 DEV와 기존 결과를 보고 고른 적응적 탐색이며 독립 일반화 검증이 아니다. K streams+K 과거 buffer+현재 C anchor여서 strict K-only도 아니다. TEST·설치·큰 모델 파라미터 학습·원 작업 중단·Git은 이번 보고 생성에서 실행하지 않았다.","","FAILURES/CONTEXT_PARITY와 원 scientific/cost JSON은 실패·문맥 기록을 보존한다. RESOURCE는 새 head 학습/독립 저장 결과 검산·새 공간 계산/실제 비용/비용 검산을 구분한다. 이 생성기는 JSON/JSONL metadata만 읽고 과학 NPZ/model/F/학습을 재실행하지 않는다.",""]
    lines += [f"- [{key}.csv](tables/{key}.csv): {len(rows)}행" for key,rows in tables.items()]
    lines += [""]+[f"- [{key}]({rel(item['path'],output)}): `{item['sha256']}`" for key,item in bindings.items()]
    return "\n".join(lines)+"\n"


def generate(paths,destination):
    output=resolve(destination); require(output.parent==(HERE/"reports").resolve() and not output.exists(),"FRESH_LOCAL_REPORT_REQUIRED")
    values,bindings=load_verified(paths); tables=collect_tables(values); source=record(__file__); bindings["report_source"]=source
    output.mkdir(parents=True,exist_ok=False); (output/"tables").mkdir()
    for key,rows in tables.items(): write_csv(output/"tables"/(key+".csv"),rows,output)
    images=figures(tables,output); (output/"REPORT_KO.md").write_text(make_text(values,tables,bindings,output),encoding="utf-8",newline="\n")
    require(record(__file__)==source,"REPORT_SOURCE_UNCHANGED")
    for item in bindings.values(): bound(item)
    manifest={"status":"COMPLETE","source":source,"inputs":bindings,"files":[{"path":p.relative_to(output).as_posix(),"sha256":sha(p),"bytes":p.stat().st_size} for p in sorted(output.rglob("*")) if p.is_file()],"table_row_counts":{k:len(v) for k,v in tables.items()},"PNG_count":4,"SVG_count":4,"scientific_NPZ_or_model_read":False,"new_updates_during_generation":0,"new_head_fits_in_source":6,"new_head_updates_in_source":3600,"new_F_forward_calls_in_scientific_source":0,"cost_uses_live_F_and_native":True,"geometry_cells_reuse_only_three_frozen_E_D_states":True,"independent_holdout":False,"strict_K_only":False,"protected_TEST_accessed":False}
    with (output/"MANIFEST.json").open("x",encoding="utf-8") as handle: json.dump(manifest,handle,ensure_ascii=False,indent=2,allow_nan=False); handle.write("\n")
    print(json.dumps({"status":"COMPLETE","manifest":str(output/"MANIFEST.json"),"images":len(images)}),flush=True); return manifest


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in (*FIXED_SHA,"output"): parser.add_argument("--"+key.replace("_","-"),dest=key,required=True)
    args=vars(parser.parse_args()); output=args.pop("output"); generate(args,output)
