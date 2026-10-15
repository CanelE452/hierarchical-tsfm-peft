"""JSON-only fixed-E/D latent calibration report; no scientific arrays or model execution."""
from pathlib import Path
import argparse
import json
import math
import statistics

from make_linear_e_adaptation_report_v2 import require,resolve,sha,record,read,bound,same_opaque,choose,rel,write_csv

HERE=Path(__file__).resolve().parent
STATES=("RECON_CP0","E_ONLY_s92601","E_ONLY_s92602")
FAMILIES=("SCALE_ONLY","SCALE_BIAS")
LAMBDAS=(0.,1e-6,1e-4,1e-2,.1,1.,10.)
ROUTES=("RAW_BOLT",)+tuple(route for state in STATES for route in (state+"_BOLT",state+"_SCALE_ONLY",state+"_SCALE_BIAS"))
SCALES=("normalized","physical_per_channel_diagnostic")
TABLES=("CANDIDATE_SCORES","SELECTED_SCORES","BASELINES","PAIRED","CANDIDATES","COEFFICIENTS","SOLVERS","DIAGNOSTICS","SELECTION","SPACE_CONTEXT","FAILURES","WHOLE_COST","WHOLE_COST_SAMPLES","RESOURCES")
FIXED_SHA={"train":"88ddbbf7287ad89b7bf81e170b3e05ddfc89bead700e2fefa50bb0dd4e85f576","audit":"b31b742d3947a205dbaf01cd84239b3cde95e1ea9baf3331a3c08c48a0aa5ba0","cost":"d8db7445a7a5a750361149cde9a3ad653d4926e871d72c387fe8cbd285aca17b","cost_audit":"4a54f5b738beef15bc02a17687624afb8c4493f5df7463aeb7b8e4e35b619c40"}
AUDIT_SOURCES={"audit":{"path":str(HERE/"audit_fixed_l7_latent_calibration_v1.py"),"sha256":"208816a2656800afa190a9f713abe3e51c4f8ec88cfc1e6116bd4e4788b2464d"},"cost_audit":{"path":str(HERE/"audit_fixed_l7_latent_calibration_cost_v1.py"),"sha256":"4a78b4b8ecc0e000021941f613360b2e94457a3a2784ff104602be44d98e06d0"}}
AUDIT_COUNTS={"fit_conditions_checked":6,"actual_SVD_fits_checked":6,"ridge_candidates_checked":42,"identity_candidates_checked":6,"FIT_TUNE_prediction_evaluations_checked":96,"selected_DEV_prediction_evaluations_checked":6,"score_fields_checked":3672,"selection_locks_checked":6,"protected_source_input_and_result_hashes_checked":226}
COST_COUNTS={"cost_cells_checked":20,"recorded_timer_samples_checked":100,"selected_full_state_hashes_checked":9,"selected_q_coefficient_config_hashes_checked":6,"selection_locks_checked":6,"physical_prechecks_checked":9,"physical_first8_windows_checked":72,"composed_q_references_checked":6,"independent_canonical_q_and_lag_D_checks":6,"canonical_TUNE_bridges_checked":1,"protected_prior_file_hashes_checked":1923}
UTILITY={"path":str(HERE/"make_linear_e_adaptation_report_v2.py"),"sha256":"32e2460ea4724144067f264ab1ceb0332cd04c5fce2296838024c119c9572466"}


def selected(row):
    candidates=[p for p in row["candidates"] if p["status"]=="COMPLETE" and math.isfinite(p.get("TUNE_MSE",float("nan")))]
    require(bool(candidates),"FINITE_COMPLETE_TUNE_CANDIDATE_REQUIRED")
    winner=min(candidates,key=lambda p:(p["TUNE_MSE"],0 if p["kind"]=="IDENTITY" else 1,-p["lambda"] if p["kind"]=="RIDGE" else 0))
    require(winner["id"]==row["selected_candidate"] and winner["kind"]==row["selected_kind"] and winner["lambda"]==row["selected_lambda"] and winner["TUNE_MSE"]==row["selected_TUNE_MSE"],"OWN_TUNE_IDENTITY_THEN_LARGER_LAMBDA_EXACT_TIE")
    same_opaque(winner["coefficient_artifact"],row["selected_coefficients"]); require(winner["coefficient_value_sha256"]==row["selected_coefficient_value_sha256"],"SELECTED_COEFFICIENT_VALUE_HASH"); return winner


def load_verified(paths):
    require(set(paths)==set(FIXED_SHA),"FOUR_EXACT_STAGE_INPUTS")
    require(all(isinstance(v,str) and len(v)==64 for v in FIXED_SHA.values()) and all(isinstance(v,dict) for v in AUDIT_SOURCES.values()) and bool(AUDIT_COUNTS) and bool(COST_COUNTS),"ACTUAL_CALIBRATION_AND_WHOLE_COST_AUDITS_NOT_YET_REGISTERED")
    require(all(Path(p).suffix==".json" for p in paths.values()),"JSON_METADATA_ONLY"); bound(UTILITY)
    values={key:read(path) for key,path in paths.items()}; bindings={key:record(path) for key,path in paths.items()}
    for key in paths: require(bindings[key]["sha256"]==FIXED_SHA[key] and values[key]["status"]==("PASS" if "audit" in key else "COMPLETE") and values[key]["protected_TEST_accessed"] is False,"EXACT_COMPLETE_PASS_NO_TEST:"+key)
    train,audit,cost,costaudit=(values[k] for k in FIXED_SHA); bound(audit["diagnostic"],bindings["train"]); bound(costaudit["cost"],bindings["cost"])
    for key,counts in (("audit",AUDIT_COUNTS),("cost_audit",COST_COUNTS)):
        require(all(values[key]["totals"].get(k)==v for k,v in counts.items()),"EXACT_INDEPENDENT_AUDIT_COVERAGE"); bound(values[key]["audit_source"],AUDIT_SOURCES[key]); bindings[key+"_source"]=values[key]["audit_source"]
    for value in (cost,costaudit): bound(value["latent_calibration_diagnostic"],bindings["train"]); bound(value["latent_calibration_audit"],bindings["audit"]); bound(value["selection_registry"],train["selection_registry"])
    require(train["fit_conditions_completed"]==6 and 0<=train["SVD_fits_completed"]<=6 and train["ridge_candidates_generated"]==42 and train["identity_comparisons_generated"]==6 and train["candidate_FIT_TUNE_evaluations_completed"]==96 and train["selected_DEV_evaluations_completed"]==train["resolved_selection_locks"]==train["selected_conditions_completed"]==6 and train["all_6_locks_before_DEV"] is True and not train["failed_paths"] and not train["failed_candidate_paths"],"SIX_FITS_MAX_SIX_SVD_FORTY_EIGHT_CANDIDATES_AND_SIX_DEV")
    for key in ("new_F_forward_calls","new_F_model_objects_loaded","new_optimizer_updates","new_E_D_training_updates"): require(train[key]==0,"NO_NEW_F_OR_GRADIENT_TRAINING:"+key)
    for key in ("native_head_weights_loaded","new_backbone_training","new_F_state_hash_computed","whole_deployment_cost_measured","independent_holdout","strict_K_only"): require(train[key] is False,"CALIBRATION_SCOPE:"+key)
    require(tuple(cost["routes"])==ROUTES and cost["cost_split"]=="tune" and cost["completed_cost_cells"]==20 and cost["recorded_timer_samples"]==100 and not cost["failed_paths"] and cost["F_before"]==cost["F_after"]==train["F_parent_signatures_reused"]["before"]==train["F_parent_signatures_reused"]["after"],"LIVE_SAME_COHORT_TEN_ROUTE_WHOLE_COST")
    keys=("parent_report","parent_report_manifest","parent_report_audit","matched_l7_native_diagnostic","matched_l7_native_audit","matched_l7_native_cost","matched_l7_native_cost_audit","native_selection_registry","native_bundle_metadata_reader","parent_report_auditor_source","E_only_diagnostic","E_only_audit","joint_recon_diagnostic","joint_recon_audit","causal_lag_diagnostic","causal_lag_audit")
    for key in keys: bound(train[key]); bindings[key]=train[key]
    values["native"]=read(train["matched_l7_native_diagnostic"]["path"]); values["native_audit"]=read(train["matched_l7_native_audit"]["path"]); values["science"]=read(train["causal_lag_diagnostic"]["path"])
    require(values["native"]["status"]==values["science"]["status"]=="COMPLETE" and values["native_audit"]["status"]=="PASS","ACCEPTED_BASELINE_AND_CONDITIONAL_SPACE_METADATA")
    registry=read(bound(train["selection_registry"])); require(registry["status"]=="ALL_SIX_RESOLVED_BEFORE_DEV" and registry["DEV_opened_before_seal"] is False and len(registry["conditions"])==6,"SIX_LOCKS_BEFORE_ANY_DEV")
    for item in registry["conditions"]:
        row=train["states"][item["state"]]["families"][item["family"]]; lock=read(bound(item["selection_lock"],row["selection_lock"])); require(lock["status"]==row["status"]==item["status"]=="COMPLETE" and lock["DEV_opened_before_lock"] is False and lock["selected_candidate"]==selected(row)["id"],"EXACT_FIT_COEFFICIENT_TUNE_SELECTION_LOCK"); same_opaque(lock["selected_coefficients"],row["selected_coefficients"])
    bindings.update(controller_source=train["controller_source"],module_source=train["module_source"],selection_registry=train["selection_registry"],plan=train["plan"],cost_plan=cost["cost_plan"],report_utility=UTILITY)
    for item in bindings.values(): bound(item)
    return values,bindings


def collect_tables(values):
    train,audit,cost,native=(values[k] for k in ("train","audit","cost","native")); out={key:[] for key in TABLES}
    def scores(table,base,entry):
        for mask,key in (("all_observed","scores_all_observed"),("complete_C","scores_complete_mask")):
            for scale in SCALES: out[table].append({**base,"mask":mask,"scale":scale,**entry[key][scale],"prediction_artifact":entry.get("artifact")})
    def diagnostic(base,entry):
        out["DIAGNOSTICS"].append({**base,**{key:entry[key] for key in ("status","surrogate_diagnostics","independent_FP32_deployment_parity","FP64_surrogate_is_not_deployment_parity_target","identity_checks","coefficient_value_sha256","input_source","cached_Bolt_source")},"prediction_artifact":entry["artifact"]})
    for state in STATES:
        for split in ("fit","tune","val"): scores("BASELINES",{"state_label":state,"predictor":"BOLT","split":split},native["states"][state]["Bolt"][split])
        for family in FAMILIES:
            row=train["states"][state]["families"][family]; winner=selected(row); base={"state_label":state,"family":family}; verified=audit["states"][state][family]
            require([p["kind"] for p in row["candidates"]]==["IDENTITY"]+["RIDGE"]*7 and [p["lambda"] for p in row["candidates"]][1:]==list(LAMBDAS),"ALL_IDENTITY_AND_SEVEN_REGISTERED_LAMBDAS")
            out["SOLVERS"].append({**base,"fit_metadata":row["fit_metadata"],"fit_artifact":row["fit_artifact"],"independently_recomputed_fit":verified["independently_recomputed_fit"],"FIT_solver_rank":verified["FIT_solver_rank"],"FIT_input_source":row["FIT_input_source"],"FIT_cached_Bolt_source":row["FIT_cached_Bolt_source"],"frozen_state_source":row["frozen_state_source"]})
            for candidate in row["candidates"]:
                group={**base,"candidate_id":candidate["id"],"kind":candidate["kind"],"lambda":candidate["lambda"],"selected":candidate["id"]==winner["id"]}; vc=choose(verified["candidates"],id=candidate["id"])
                out["CANDIDATES"].append({**group,**{k:candidate[k] for k in ("status","metadata","coefficient_artifact","coefficient_value_sha256")},"TUNE_MSE":candidate.get("TUNE_MSE"),"FIT_prediction_artifact":candidate.get("FIT",{}).get("artifact"),"TUNE_prediction_artifact":candidate.get("TUNE",{}).get("artifact")})
                out["COEFFICIENTS"].append({**group,**{key:vc[key] for key in ("a_FP64","b_FP64","a_FP32","b_FP32","coefficients_FP64","coefficients_FP32","nonfinite_coefficient_indices")},"coefficient_artifact":candidate["coefficient_artifact"]})
                for label,split in (("FIT","fit"),("TUNE","tune")):
                    if label in candidate: scores("CANDIDATE_SCORES",{**group,"split":split},candidate[label]); diagnostic({**group,"phase":"CANDIDATE","split":split},candidate[label])
            out["SELECTION"].append({**base,**{key:row[key] for key in ("selected_candidate","selected_kind","selected_lambda","selected_TUNE_MSE","selected_coefficients","selected_coefficient_value_sha256","selection_lock","selection_rule")},"E_previously_supervised_by_FIT_future":train["states"][state]["E_previously_supervised_by_FIT_future"],"D_previously_supervised_by_FIT_future":False})
            for label,split in (("FIT","fit"),("TUNE","tune"),("DEV","val")): scores("SELECTED_SCORES",{**base,"candidate_id":winner["id"],"split":split},row[label])
            diagnostic({**base,"candidate_id":winner["id"],"kind":winner["kind"],"lambda":winner["lambda"],"selected":True,"phase":"SELECTED","split":"val"},row["DEV"])
        for split in ("tune","val"):
            cell=values["native_audit"]["new_output_space_diagnostics"][state]["92601"][split]; out["SPACE_CONTEXT"].append({"state_label":state,"split":split,"minimum_normalized_MSE":cell["minimum_normalized_MSE"],"RAW_complete_MSE":cell["RAW_BOLT_scores_complete_mask"]["normalized"]["mse"],"complete_target_positions":cell["complete_target_positions"],"target_positions":cell["target_positions"],"source_audit":train["matched_l7_native_audit"],"new_geometry_computed":False,"universal_all_FP32_bound":False})
    raw=values["science"]["datasets"]["robin"]["spaces"]["PCA"]["conditions"]["L7_RECON"]["splits"]
    for split in ("fit","tune","val"): scores("BASELINES",{"state_label":"RAW_C","predictor":"RAW_BOLT","split":split},raw[split]["RAW_BOLT"])
    for state in STATES:
        for family in FAMILIES:
            for split in ("fit","tune","val"):
                for mask in ("all_observed","complete_C"):
                    for scale in SCALES:
                        where={"state_label":state,"family":family,"split":split,"mask":mask,"scale":scale}; q=choose(out["SELECTED_SCORES"],**where); b=choose(out["BASELINES"],state_label=state,predictor="BOLT",split=split,mask=mask,scale=scale); r=choose(out["BASELINES"],state_label="RAW_C",predictor="RAW_BOLT",split=split,mask=mask,scale=scale)
                        out["PAIRED"].append({**where,"q_MSE":q["mse"],"baseline_MSE":b["mse"],"RAW_MSE":r["mse"],"q_minus_baseline_MSE":q["mse"]-b["mse"],"q_relative_baseline_MSE_change":q["mse"]/b["mse"]-1 if b["mse"] else None,"q_MAE":q["mae"],"baseline_MAE":b["mae"],"RAW_MAE":r["mae"],"q_minus_baseline_MAE":q["mae"]-b["mae"]})
    for stage,field in (("train","failed_paths"),("train","failed_candidate_paths"),("audit","partial_paths"),("cost","failed_paths")): out["FAILURES"] += [{"stage":stage,"collection":field,**item} for item in values[stage][field]]
    rows=cost["cost"]["robin"]["rows"]; require(len(rows)==20 and {(r["route"],r["batch"]) for r in rows}=={(route,b) for route in ROUTES for b in (1,8)},"TEN_ROUTES_TWO_BATCHES")
    for row in rows:
        sec=row["seconds"]; require(row["status"]=="COMPLETE" and len(sec)==len(row["samples"])==5 and row["median_seconds"]==statistics.median(sec) and row["median_seconds_per_window"]==statistics.median(sec)/row["batch"],"FIVE_COMPLETE_TIMER_SAMPLES")
        for field,route in (("RAW","RAW_BOLT"),("matched_Bolt",row["matched_Bolt_route"])):
            denominator=choose(rows,route=route,batch=row["batch"]); require(row["paired_"+field+"_median_ratio"]==statistics.median(a/b for a,b in zip(sec,denominator["seconds"])) and row["ratio_of_"+field+"_medians"]==statistics.median(sec)/statistics.median(denominator["seconds"]),"DISTINCT_PAIRED_AND_MEDIAN_RATIOS")
        out["WHOLE_COST"].append({k:v for k,v in row.items() if k!="samples"})
        for sample in row["samples"]: require(type(sample["perf_start_ns"]) is type(sample["perf_end_ns"]) is int and (sample["perf_end_ns"]-sample["perf_start_ns"])/1e9==sample["seconds"],"INTEGER_NS_TIMING"); out["WHOLE_COST_SAMPLES"].append(sample)
    out["RESOURCES"]=[{"stage":key,**values[key]["resource_terminal"]} for key in FIXED_SHA]
    require([len(out[k]) for k in TABLES]==[384,72,48,72,48,48,6,102,6,6,0,20,100,4],"FULL_SIX_CONDITION_FORTY_EIGHT_CANDIDATE_REPORT_COUNTS")
    return out


def figures(tables,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    folder=output/"figures"; folder.mkdir(); files=[]; groups=[(s,f) for s in STATES for f in FAMILIES]; labels=["E"+str(STATES.index(s))+"/"+("scale" if f=="SCALE_ONLY" else "scale+bias") for s,f in groups]
    def save(fig,name,bottom=.1):
        fig.tight_layout(rect=(0,bottom,1,.91))
        for ext in ("png","svg"):
            path=folder/(name+"."+ext); fig.savefig(path,dpi=160,bbox_inches="tight"); files.append(path)
        plt.close(fig)
    def paired(state,family,split): return choose(tables["PAIRED"],state_label=state,family=family,split=split,mask="all_observed",scale="normalized")
    fig,axes=plt.subplots(2,2,figsize=(14,8.8))
    for column,split in enumerate(("tune","val")):
        rows=[paired(state,family,split) for state,family in groups]; upper,lower=axes[0,column],axes[1,column]
        upper.plot(range(6),[r["q_MSE"] for r in rows],"o-",color="#0072B2",label="Selected calibration"); upper.plot(range(6),[r["baseline_MSE"] for r in rows],"x--",color="#D55E00",label="Matching frozen Bolt"); upper.axhline(rows[0]["RAW_MSE"],color="black",linestyle=":",label="Raw-C Bolt")
        upper.set_title("TUNE selection" if split=="tune" else "Exposed DEV"); upper.set_ylabel("All-observed normalized MSE"); lower.axhline(0,color="black",linewidth=.8); lower.plot(range(6),[r["q_minus_baseline_MSE"] for r in rows],"o-",color="#0072B2"); lower.set_ylabel("Selected minus baseline MSE\nlabels: relative change versus baseline"); lower.ticklabel_format(axis="y",style="sci",scilimits=(0,0)); lower.margins(y=.3)
        for index,row in enumerate(rows): lower.annotate(f"{100*row['q_relative_baseline_MSE_change']:+.4g}%",(index,row["q_minus_baseline_MSE"]),xytext=(0,9 if row["q_minus_baseline_MSE"]>=0 else -16),textcoords="offset points",ha="center",fontsize=9)
        for ax in (upper,lower): ax.set_xticks(range(6),labels,rotation=12); ax.grid(alpha=.2)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="lower center",ncol=3); fig.suptitle("Fixed E/D: selected latent calibration versus its baseline and Raw; signed deltas expose small changes"); save(fig,"01_selected_errors")
    fig,axes=plt.subplots(2,3,figsize=(15,9)); xlabels=("ID","0","1e-6","1e-4",".01",".1","1","10")
    for ax,(state,family) in zip(axes.flat,groups):
        candidates=[p for p in tables["CANDIDATES"] if p["state_label"]==state and p["family"]==family]; fit=[]; tune=[]
        for candidate in candidates:
            for split,out in (("fit",fit),("tune",tune)): out.append(choose(tables["CANDIDATE_SCORES"],state_label=state,family=family,candidate_id=candidate["candidate_id"],split=split,mask="all_observed",scale="normalized")["mse"])
        ax.plot(range(8),fit,"o-",color="#0072B2",label="FIT actual FP32"); ax.plot(range(8),tune,"s-",color="#D55E00",label="TUNE actual FP32"); ax.axhline(tune[0],color="gray",linestyle=":",label="Identity TUNE"); index=next(i for i,p in enumerate(candidates) if p["selected"]); ax.scatter(index,tune[index],marker="*",s=160,color="black",zorder=4,label="TUNE selection")
        solver=choose(tables["SOLVERS"],state_label=state,family=family); diag=[p["surrogate_diagnostics"]["max_abs_actual_FP32_minus_linear_surrogate"] for p in tables["DIAGNOSTICS"] if p["state_label"]==state and p["family"]==family and p["phase"]=="CANDIDATE"]
        ax.set_title("E"+str(STATES.index(state))+" / "+family+" | rank "+str(solver["FIT_solver_rank"])+"\nchosen "+candidates[index]["candidate_id"]+f"; max surrogate gap {max(diag):.2e}",fontsize=10); ax.set_xticks(range(8),xlabels); ax.set_ylabel("Normalized MSE"); ax.set_xlabel("Registered lambda; ID is a separate exact-baseline path"); ax.grid(alpha=.2)
    fig.legend(*axes.flat[0].get_legend_handles_labels(),loc="lower center",ncol=4); fig.suptitle("42 ridge candidates plus 6 identities; FIT-weighted FP64 solver, actual FP32 TUNE selection"); save(fig,"02_lambda_curves",.08)
    fig,axes=plt.subplots(1,2,figsize=(15,6.3),gridspec_kw={"width_ratios":[1.2,1]}); delta=[]
    for state,family in groups:
        q=choose(tables["SELECTED_SCORES"],state_label=state,family=family,split="val",mask="all_observed",scale="normalized"); b=choose(tables["BASELINES"],state_label=state,predictor="BOLT",split="val",mask="all_observed",scale="normalized"); delta.append([a-v for a,v in zip(q["per_channel_mse"],b["per_channel_mse"])])
    limit=max(abs(v) for row in delta for v in row) or 1.; image=axes[0].imshow(delta,cmap="coolwarm",aspect="auto",vmin=-limit,vmax=limit); axes[0].set_yticks(range(6),labels); axes[0].set_xticks(range(len(delta[0])),["C"+str(i) for i in range(len(delta[0]))],rotation=60); axes[0].set_title("DEV selected-minus-baseline per-channel normalized MSE"); fig.colorbar(image,ax=axes[0],shrink=.8,label="Signed MSE difference")
    for gi,(state,family) in enumerate(groups):
        for split,color in (("fit","#0072B2"),("tune","#D55E00")):
            rows=[p for p in tables["DIAGNOSTICS"] if p["state_label"]==state and p["family"]==family and p["phase"]=="CANDIDATE" and p["split"]==split]; ys=[p["surrogate_diagnostics"]["max_abs_actual_FP32_minus_linear_surrogate"] for p in rows]; axes[1].scatter([gi+(j-3.5)*.025 for j in range(len(rows))],ys,color=color,alpha=.75,s=22,label=split.upper() if gi==0 else None)
        selected=[p for p in tables["DIAGNOSTICS"] if p["state_label"]==state and p["family"]==family and p["phase"]=="CANDIDATE" and p["selected"]]; axes[1].scatter([gi]*len(selected),[p["surrogate_diagnostics"]["max_abs_actual_FP32_minus_linear_surrogate"] for p in selected],facecolors="none",edgecolors="black",s=90,label="Selected candidate" if gi==0 else None)
    axes[1].set_yscale("symlog",linthresh=1e-12); axes[1].set_xticks(range(6),labels,rotation=12); axes[1].set_ylabel("max |actual FP32 minus linear FP64 surrogate|"); axes[1].set_title("Measured surrogate discrepancy; identity zeros remain visible"); axes[1].grid(alpha=.2); fig.legend(*axes[1].get_legend_handles_labels(),loc="lower center",ncol=3); fig.suptitle("Channel effects and numerical context: surrogate error is not the deployment parity target"); save(fig,"03_channel_and_surrogate_context",.12)
    fig,axes=plt.subplots(2,2,figsize=(15,11),gridspec_kw={"width_ratios":[1.2,1]}); short={"RAW_BOLT":"Raw-C Bolt"}
    for index,state in enumerate(STATES):
        short[state+"_BOLT"]="E"+str(index)+" / Bolt"
        for family in FAMILIES: short[state+"_"+family]="E"+str(index)+" / "+("scale" if family=="SCALE_ONLY" else "scale+bias")
    for ri,batch in enumerate((1,8)):
        ax,ratios=axes[ri]; rows=[choose(tables["WHOLE_COST"],route=route,batch=batch) for route in ROUTES]
        for index,row in enumerate(rows):
            color="black" if row["route"]=="RAW_BOLT" else "#D55E00" if row["route"].endswith("_BOLT") else "#0072B2"; ax.barh(index,row["median_seconds_per_window"]*1000,color=color,alpha=.65); ax.scatter([s/batch*1000 for s in row["seconds"]],[index]*5,color=color,s=16,zorder=3)
        ax.set_yticks(range(10),[short[r["route"]] for r in rows]); ax.invert_yaxis(); ax.set(xlabel="Whole ms per window; bar=median, dots=5 samples",title="Same actual cohort / B"+str(batch)); ax.grid(axis="x",alpha=.2); calibrated=[r for r in rows if not r["route"].endswith("_BOLT")]
        for key,color,marker,label in (("paired_RAW_median_ratio","#0072B2","o","Median paired ratio / Raw"),("paired_matched_Bolt_median_ratio","#D55E00","s","Median paired ratio / matching Bolt")): ratios.scatter([r[key] for r in calibrated],range(6),color=color,marker=marker,label=label)
        ratios.axvline(1,color="black",linestyle="--"); ratios.set_yticks(range(6),[short[r["route"]] for r in calibrated]); ratios.invert_yaxis(); ratios.set(xlabel="Below 1 is faster; paired ratios, not ratio of medians",title="B"+str(batch)+" / two explicit denominators"); ratios.grid(axis="x",alpha=.2)
    fig.legend(*axes[0,1].get_legend_handles_labels(),loc="lower center",ncol=2); fig.suptitle("Actual frozen F + fixed latent calibration + lag D: 20 conditions / 100 samples"); save(fig,"04_whole_cost",.08)
    return files


def make_text(values,tables,bindings,output):
    train=values["train"]
    def paired(state,family,split): return choose(tables["PAIRED"],state_label=state,family=family,split=split,mask="all_observed",scale="normalized")
    selections=tables["SELECTION"]; identities=sum(p["selected_kind"]=="IDENTITY" for p in selections); tune_better=sum(paired(s,f,"tune")["q_MSE"]<paired(s,f,"tune")["baseline_MSE"] for s in STATES for f in FAMILIES); dev_better=sum(paired(s,f,"val")["q_MSE"]<paired(s,f,"val")["baseline_MSE"] for s in STATES for f in FAMILIES); raw_better=sum(paired(s,f,"val")["q_MSE"]<paired(s,f,"val")["RAW_MSE"] for s in STATES for f in FAMILIES)
    lines=["# 고정 L7 E/D의 잠재 미래 scale·bias 보정","",f"[확인] 6조건 중 identity 선정은 {identities}/6이다. 원 Bolt baseline보다 MSE가 낮은 선정 q는 TUNE {tune_better}/6, 기노출 DEV {dev_better}/6이고 Raw보다 DEV MSE가 낮은 조건은 {raw_better}/6이다. 이 작은 보정이 Raw gap을 해결했다고 볼 수 없다. 작은 TUNE 개선은 단일 노출 자료에서의 점추정이며 독립 일반화나 유의한 개선의 근거가 아니다.",""]
    for row in selections:
        s,f=row["state_label"],row["family"]; t,d=(paired(s,f,split) for split in ("tune","val"))
        lines.append(f"- {s} / {f}: {row['selected_candidate']}; TUNE q {t['q_MSE']:.12f}, baseline {t['baseline_MSE']:.12f}, delta {t['q_minus_baseline_MSE']:+.12f} ({100*t['q_relative_baseline_MSE_change']:+.6f}%); DEV q {d['q_MSE']:.12f}, baseline {d['baseline_MSE']:.12f}, delta {d['q_minus_baseline_MSE']:+.12f} ({100*d['q_relative_baseline_MSE_change']:+.6f}%), Raw {d['RAW_MSE']:.12f}.")
    fit_improved=0
    for state in STATES:
        for family in FAMILIES:
            rows=[p for p in tables["CANDIDATE_SCORES"] if p["state_label"]==state and p["family"]==family and p["split"]=="fit" and p["mask"]=="all_observed" and p["scale"]=="normalized"]
            fit_improved+=min(p["mse"] for p in rows)<choose(rows,candidate_id="IDENTITY")["mse"]
    lines += ["",f"[확인] actual FIT MSE의 후보 최솟값이 identity보다 낮은 조건은 {fit_improved}/6이다. 이 제한된 family가 FIT 잔차 일부를 적합했는지와 TUNE/DEV로 옮겨졌는지를 구분한다. FP64 surrogate 직접 해를 actual FP32 MSE의 정확한 최적해로 부르지 않으며, 이 직접 solver의 결과를 모든 SGD 학습의 수렴 판단으로 확대하지 않는다."]
    lines += ["","![선정 보정과 기준](figures/01_selected_errors.png)","","**질문과 고정 조건.** 원 RECON CP0 E0와 이전 E-only 선정 E1/E2의 같은 E/D/center, Robin C17/K4·L512/H48·lag [0,1,2,4,8,16,32]·FIT256/TUNE64/기노출 VAL30·현재 C anchor·원 scaler/origin/mask를 유지했다. E1/E2의 과거 FIT 미래 감독 이력은 없어지지 않지만 이번 E/D/F 학습·optimizer·F load/forward·native head weight load는0이다. FIT 미래는 정상 회귀 response이고 배포 입력은 과거와 동결 F 예측이다.","","두 family는 u'=u+a_k(u-z_last), 또는 여기에 b_k를 더하는 P4/P8 global 보정이며 origin/horizon에 계수를 공유한다. 별도 C intercept나 새 D는 없다. 미래만 바꾸는 lag 설계 열을 쓰며 알려진 과거 prefix와 lag별 z_past[-1-lag] 기준을 보존했다. 실수 산술에서 a=-1,b=0은 z_last지만 FP32 u-(u-z_last)의 cancellation 때문에 일반 byte-collapse를 보장하지 않는다. FLAT latent도 현재 C persistence가 아니며 prefix가 끝나도 lag별 기준 때문에 plateau가 남을 수 있다. a=b=0 identity는 원 baseline/future 객체를 그대로 반환하는 별도 byte-exact 경로다.","",f"[확인] 실제 {train['fit_conditions_completed']} fit 조건에서 SVD 호출은 {train['SVD_fits_completed']}회이고 ridge42후보+identity6비교를 했다. FIT/TUNE actual FP32 평가는 {train['candidate_FIT_TUNE_evaluations_completed']}개, 선정 DEV는 {train['selected_DEV_evaluations_completed']}개다. 전체 FIT feature가 정확0이면 해당 fit의 SVD0과7개 zero 후보를 보존하는 계약이며 무조건6SVD라고 세지 않는다. 같은 세 상태의 두 family를6개 독립 자료로 세지 않는다.","","**적합과 선정.** 원 FIT 관측만 w=1/(C_active·N_FIT,c)로 가중했고 관측 행의 합은1이다. r=Y-actual cached baseline, 비중심 설계의 FIT RMS s_j와 A=sqrt(w)X/s로 FP64 identity-centered ridge를 적합했다. 정확0 feature 열만 제거하고 원 위치에0을 넣었다. lambda0 상대 cutoff1e-12, 양수 lambda의 sigma/(sigma²+lambda)를 구분하고 계수나 음수 gain을 사후 clip하지 않았다. 기존 공간 minimum의 cutoff1e-15를 바꾸는 작업이 아니다. λ=[0,1e-6,1e-4,.01,.1,1,10]과 identity의 actual CUDA FP32 C 예측으로 TUNE 자체 관측수 macro MSE를 계산했다. 동률이면 identity, 이후 큰 lambda를 우선하며 모든6잠금·registry 뒤 DEV를 열었다. FIT 가중치를 TUNE/DEV 분모로 재사용하지 않는다.","","![λ와 적합 문맥](figures/02_lambda_curves.png)","","**수치 검산의 두 역할.** actual FP32 q+원 lag D는 독립 NumPy FP32 q+lag-specific decode와 고정 atol1e-5/rtol2e-6으로 검산했다. FP64 baseline+Xβ는 이 parity 대상이 아니다. actual과 선형 surrogate의 max/RMS/per-channel 차이는 별도로 측정하고0으로 가정하지 않는다. 계수64→32, FIT RMS/zero columns/rank/singularvalues 및 실제 저장 값을 독립 감사 JSON에서 받아 CSV로 보존했다. 양수 ridge의 augmented weighted least squares 등 독립 solver 검산은 저장 계산을 대상으로 하며 실제 F/GPU/optimizer를 재현한 것이 아니다. rank가 부족하면 계수를 고유한 잠재 의미나 인과효과로 해석하지 않는다.",""]
    diagnostic=tables["DIAGNOSTICS"]; lines += [f"현재 보존된 actual-minus-surrogate max absolute error의 최댓값은 {max(p['surrogate_diagnostics']['max_abs_actual_FP32_minus_linear_surrogate'] for p in diagnostic):.12g}, RMS의 최댓값은 {max(p['surrogate_diagnostics']['RMS_actual_FP32_minus_linear_surrogate'] for p in diagnostic):.12g}다. 독립 FP32 C parity max error의 최댓값은 {max(p['independent_FP32_deployment_parity']['C_prediction']['max_abs'] for p in diagnostic):.12g}다. 세 수치는 서로 다른 검사다.","","![채널 효과와 surrogate 차이](figures/03_channel_and_surrogate_context.png)","","**현재 고정 공간의 해석.** 앞서 검산된 같은 D32/known-past offset/complete-C/rcond1e-15 retained numeric minimum을 문맥으로 재사용하며 새 공간 적합은0이다. 낮은 DEV minimum은 과거 예측 가능성의 증명이 아니고 TUNE와 DEV의 반대 근거를 함께 읽어야 한다. 모든 실수/FP32 forecast의 보편 하한으로 바꾸지 않는다.",""]
    for row in tables["SPACE_CONTEXT"]: lines.append(f"- {row['state_label']} / {row['split']}: retained numeric minimum {row['minimum_normalized_MSE']:.12f}, Raw {row['RAW_complete_MSE']:.12f}; complete {row['complete_target_positions']}/{row['target_positions']}; 이전 감사 문맥 재사용.")
    lines += ["","**실제 전체 비용.** 과학 실행에서는 검산된 F 캐시를 썼지만 새 whole cost는 실제 동결 Bolt를 실행한다. 선정 q6+원 Bolt3+Raw1을 같은 물리 TUNE 첫B B1/B8에서 warm2·seed20261015·5순환으로20셀100표본 측정했다. 5순환은10위치 완전 균형이 아니며 명시적 warm2가 전체 비측정 forward2라는 뜻도 아니다. causal CPU fill/FIT 정규화→GPU E→actual F→선정 q 또는 identity→원 lag D/current C anchor→물리 CPU 반환/sync가 timer 안이다. 내부 finite/shape 검사·sync도 남아 있다. 외부 meter/hash/guard/reference/setup/학습·선정/warm/save/memory snapshot은 밖이다. identity가 선정돼도 cached 과학 출력을 반환해 F 비용을 없애지 않았다.","","Raw와 같은 E의 원 Bolt를 분모로 paired median ratio와 ratio of medians를 나눠 제시한다. 같은 첫B 반복과 공유 resident 메모리를 독립 표본·독점 peak·모든 backend 최솟값으로 바꾸지 않는다. 다른 cohort의 절대 시간이나 빠른 실행만 차용하지 않는다. identity 선정 route는 matching Bolt와 같은 함수이므로 중복 route의 시간 차이를 서로 다른 학습 방법의 속도 승리로 세지 않는다. q가 identity를 제외하면 H×K 연산을 더하고 F/E/D가 계속 필요하므로 무료 보정으로 부르지 않는다.",""]
    for row in tables["WHOLE_COST"]: lines.append(f"- B{row['batch']} {row['route']}: {row['median_seconds_per_window']*1000:.6f} ms/window; Raw paired {row['paired_RAW_median_ratio']:.6f}, medians {row['ratio_of_RAW_medians']:.6f}; matching Bolt paired {row['paired_matched_Bolt_median_ratio']:.6f}, medians {row['ratio_of_matched_Bolt_medians']:.6f}.")
    lines += ["","![실제 전체 비용](figures/04_whole_cost.png)","","[판단] 이번 결과는 이 고정 E/D와 P4/P8 보정·사전 λ 목록 안에서만 해석한다. 실패나 identity 선정을 모든 보정·다른 E/D/K·forecast-compatible 표현의 불가능성으로 확대하지 않는다. 기존 D forecast-LS/D2·MAIN residual·joint-Y/Raw-F teacher 증류는 이미 수행했으므로 첫 calibration·첫 예측 목적 학습·처음 시도할 teacher로 쓰지 않는다. 이미 노출된 DEV와 기존 진단을 보고 고른 적응적 탐색이며 독립 일반화 검증이 아니다. K streams+K 과거 buffer+현재 C anchor여서 strict K-only도 아니다. Bolt 필요성이나 제거 가능성도 이 결과만으로 확정하지 않는다.","","FAILURES는 실제 실패 기록을 보존하며 미실행을 성공으로 바꾸지 않는다. 본문·CSV·그림은 JSON metadata만 읽고 NPZ/model/F/학습/TEST/Git을 실행하지 않았다. source 원수치·선정·캐시 재사용과 실제 비용을 구분하며 RESOURCE는 과학 적합/과학 감사/새 비용/비용 감사를 각각 기록한다.",""]
    lines += [f"- [{key}.csv](tables/{key}.csv): {len(rows)}행" for key,rows in tables.items()]; lines += [""]+[f"- [{key}]({rel(item['path'],output)}): `{item['sha256']}`" for key,item in bindings.items()]
    return "\n".join(lines)+"\n"


def generate(paths,destination):
    output=resolve(destination); require(output.parent==(HERE/"reports").resolve() and not output.exists(),"FRESH_LOCAL_REPORT_REQUIRED")
    values,bindings=load_verified(paths); tables=collect_tables(values); source=record(__file__); bindings["report_source"]=source; output.mkdir(parents=True,exist_ok=False); (output/"tables").mkdir()
    for key,rows in tables.items(): write_csv(output/"tables"/(key+".csv"),rows,output)
    images=figures(tables,output); (output/"REPORT_KO.md").write_text(make_text(values,tables,bindings,output),encoding="utf-8",newline="\n"); require(record(__file__)==source,"REPORT_SOURCE_UNCHANGED")
    for item in bindings.values(): bound(item)
    manifest={"status":"COMPLETE","source":source,"inputs":bindings,"files":[{"path":p.relative_to(output).as_posix(),"sha256":sha(p),"bytes":p.stat().st_size} for p in sorted(output.rglob("*")) if p.is_file()],"table_row_counts":{k:len(v) for k,v in tables.items()},"PNG_count":4,"SVG_count":4,"scientific_NPZ_or_model_read":False,"new_fit_or_training_during_generation":0,"new_geometry_computed":False,"FP64_surrogate_is_not_FP32_deployment":True,"actual_cost_has_live_F":True,"protected_TEST_accessed":False,"independent_holdout":False,"strict_K_only":False}
    with (output/"MANIFEST.json").open("x",encoding="utf-8") as handle: json.dump(manifest,handle,ensure_ascii=False,indent=2,allow_nan=False); handle.write("\n")
    print(json.dumps({"status":"COMPLETE","manifest":str(output/"MANIFEST.json"),"images":len(images)}),flush=True); return manifest


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in (*FIXED_SHA,"output"): parser.add_argument("--"+key.replace("_","-"),dest=key,required=True)
    args=vars(parser.parse_args()); output=args.pop("output"); generate(args,output)
