"""Gated JSON/JSONL-only report of same-initialization linear-E adaptation."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import statistics


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATASETS = {"robin": (17,4,"val"), "beijing_aotizhongxin": (11,2,"eval")}
LAGS, SEEDS = ("L0","L7"), (92601,92602)
ROUTES = ("RAW_BOLT",) + tuple(f"{lag}_{'E_LEARN' if learn else 'E_FIXED'}_s{seed}" for lag in LAGS for learn in (False,True) for seed in SEEDS)
INPUTS = ("probe","train","audit","cost","cost_audit")
FIXED_SHA = {"probe":"00440bf6e09ac31db4666ffc656ec348cc4b5f698b18600c65d0ff99cf0d1b8d",
    "train":"3f0cbd9176f2e7882778452e537564aab7445e1b88c1534fd48b1bce7ef639a4",
    "audit":"9796b4ed53f74186bc2513db56fe321c2d450cdd9bffbb7c5f341766e73ee36d",
    "cost":"5038bc1cd6c52fadf4b9397b3eabb9f1bcf3085c9901d954222d43835c05a97d",
    "cost_audit":"7ddd11e9d201d4dad9e051a99e4fc0d52af7d0441a0ea3fef10c646f87827a8f"}
AUDITOR_SOURCES = {"audit":{"path":str(HERE/"audit_linear_e_adaptation_v1.py"),
    "sha256":"4d31543d2f1e9f9920096e545865c6f9eee88e912961f83310abd4c44458aa89"},
    "cost_audit":{"path":str(HERE/"audit_linear_e_cost_v1.py"),
        "sha256":"26374f7f3531863450661781fd9b11bb4e019afc5394765dc4c135142272909a"}}
SCIENCE_COUNTS = {"adapter_fits_checked":16,"checkpoint_states_checked":80,"checkpoint_TUNE_predictions_checked":80,
    "selected_FIT_DEV_predictions_checked":32,"CP0_FIT_DEV_baselines_checked":8,"logged_updates_checked":9600,
    "score_fields_checked":3744,"probe_cases_checked":4,"selection_locks_checked":16}
COST_COUNTS = {"cost_cells_checked":36,"recorded_timer_samples_checked":324,"selected_state_hashes_checked":16,
    "selection_locks_checked":16,"physical_prechecks_checked":16,"physical_first8_windows_checked":128,"canonical_TUNE_bridges_checked":2}
SCALES = ("normalized","physical_per_channel_diagnostic")
TABLES = ("SCORES","PAIRED_E","CHANNEL_COMPARISONS","CHECKPOINTS","UPDATES","CONDITIONS","PROBES","SELECTION_LOCKS",
    "WHOLE_COST","WHOLE_COST_SAMPLES","ACCURACY_COST_REFERENCE","EXECUTION_RESOURCES","PROVENANCE","PRESERVED_FAILURES")


def require(value,message):
    if not value: raise ValueError(message)


def resolve(path):
    path=Path(path); return path.resolve() if path.is_absolute() else (ROOT/path).resolve()


def sha(path):
    with Path(path).open("rb") as handle: return hashlib.file_digest(handle,"sha256").hexdigest()


def record(path):
    path=resolve(path); return {"path":str(path),"sha256":sha(path),"bytes":path.stat().st_size}


def read(path):
    path=resolve(path); require(path.suffix.lower()==".json","JSON_METADATA_ONLY_INPUT")
    return json.loads(path.read_text(encoding="utf-8-sig"),parse_constant=lambda value: (_ for _ in ()).throw(ValueError("NONFINITE_JSON:"+value)))


def bound(item,expected=None):
    path=resolve(item["path"])
    require(path.suffix.lower() in (".json",".jsonl",".md",".py") and sha(path)==item["sha256"],"EXACT_METADATA_BINDING")
    if "bytes" in item: require(path.stat().st_size==item["bytes"],"EXACT_METADATA_BYTES")
    if expected: require(path==resolve(expected["path"]) and item["sha256"]==expected["sha256"],"EXACT_PARENT_BINDING")
    return path


def same_opaque(first,second):
    require(resolve(first["path"])==resolve(second["path"]) and first["sha256"]==second["sha256"],"EXACT_OPAQUE_ARRAY_REFERENCE")


def load_verified(paths):
    require(set(paths)==set(INPUTS),"FIVE_EXACT_STAGE_INPUTS_REQUIRED")
    require(all(isinstance(FIXED_SHA[key],str) and len(FIXED_SHA[key])==64 for key in INPUTS)
        and all(isinstance(value,dict) for value in AUDITOR_SOURCES.values()) and bool(SCIENCE_COUNTS) and bool(COST_COUNTS),
        "ACTUAL_TRAIN_COST_PASS_NOT_YET_REGISTERED")
    values={key:read(paths[key]) for key in INPUTS}; bindings={key:record(paths[key]) for key in INPUTS}
    for key in INPUTS:
        require(bindings[key]["sha256"]==FIXED_SHA[key] and values[key].get("status")==("PASS" if "audit" in key else "COMPLETE"),"EXACT_ACTUAL_COMPLETE_PASS:"+key)
        require(values[key].get("protected_TEST_accessed") is False,"NO_PROTECTED_TEST:"+key)
    probe,train,audit,cost,cost_audit=(values[key] for key in INPUTS)
    bound(audit["diagnostic"],bindings["train"]); bound(audit["probe_result"],bindings["probe"]); bound(train["probe_result"],bindings["probe"])
    bound(cost_audit["cost"],bindings["cost"])
    for value in (cost,cost_audit):
        bound(value["linear_e_diagnostic"],bindings["train"]); bound(value["linear_e_audit"],bindings["audit"])
        bound(value["probe_result"],bindings["probe"]); bound(value["selection_registry"],train["selection_registry"])
    for key,counts in (("audit",SCIENCE_COUNTS),("cost_audit",COST_COUNTS)):
        require(all(values[key].get("totals",{}).get(name)==count for name,count in counts.items()),"FULL_INDEPENDENT_AUDIT_COVERAGE")
        bound(values[key]["audit_source"],AUDITOR_SOURCES[key]); bindings[key+"_source"]=values[key]["audit_source"]
    require(probe["mode"]=="probe" and probe["probe_optimizer_updates_completed"]==4 and not probe["failed_paths"],"FOUR_ACTUAL_PROBES")
    for data in DATASETS:
        for lag in LAGS:
            row=probe["cases"][data][lag]; gradient=row["gradient_diagnostics"]
            require(row["status"]=="PASS" and row["fixed_learned_CP0_byte_exact"] is True
                and all(item["pass"] for group in ("CP0_NumPy_parity","canonical32_vs_batch4") for item in row[group].values())
                and gradient["finite"] is True and gradient["F_path_available"] is True and gradient["E_F_path_gradient_norm"]>0,"PROBE_PARITY_AND_F_PATH")
    require(train["mode"]=="train" and train["adaptation_training_fits"]==train["resolved_selection_locks"]==16
        and train["training_updates_completed"]==9600 and train["checkpoint_states_saved"]==80
        and train["all_16_locks_before_any_DEV"] is True and not train["failed_paths"],"COMPLETE_SIXTEEN_FITS")
    require(train["new_backbone_training"] is False and train["independent_holdout"] is False and train["strict_K_only"] is False
        and train["same_initialization_seeds_change_order_only"] is True and train["training_contract"]["full_FIT_channel_count_loss"] is True
        and train["training_contract"]["F_parameters_optimized"] is False and train["training_contract"]["F_only_gradient_probe_is_necessity_test"] is False,"EXACT_SUPERVISION_AND_INTERPRETATION_SCOPE")
    for value in (probe,train):
        require(value["F_before"]==value["F_after"] and value["old_assets_sources_F_weights_unchanged"] is True,"F_AND_PRIOR_ASSETS_PRESERVED")
    require(tuple(cost["routes"])==ROUTES and cost["cost_split"]=="tune" and cost["completed_cost_cells"]==36 and cost["recorded_timer_samples"]==324
        and not cost["failed_paths"] and cost["selected_states_before"]==cost["selected_states_after"]
        and cost["new_training_updates"]==0 and cost["new_encoder_decoder_fit"] is False and cost["F_before"]==cost["F_after"]
        and cost["per_origin_decoder_fit"] is False and cost["cached_future_returned_instead_of_actual_F_inside_timer"] is False,"NEW_ACTUAL_NINE_ROUTE_WHOLE_COST")
    registry=read(bound(train["selection_registry"]))
    require(registry["status"]=="ALL_16_RESOLVED_BEFORE_DEV" and registry["DEV_opened_before_seal"] is False and len(registry["conditions"])==16,"ALL_LOCKS_BEFORE_DEV")
    for item in registry["conditions"]:
        name=f"{item['dataset']}_{item['lag']}_{'E_LEARN' if item['learn_E'] else 'E_FIXED'}_s{item['seed']}"
        row=train["datasets"][item["dataset"]]["conditions"][name]
        bound(item["selection_lock"],row["selection_lock"]); lock=read(bound(row["selection_lock"]))
        require(lock["status"]=="COMPLETE" and lock["DEV_opened_before_lock"] is False and lock["selected_update"]==row["selected_update"],"TUNE_LOCK_SELECTED_BEFORE_DEV")
        same_opaque(lock["selected_state"],row["selected_state"])
    for key in ("plan","controller_source","pure_source","selection_registry","causal_lag_diagnostic","causal_lag_audit","parent_report","parent_report_manifest","parent_report_audit"):
        bound(train[key]); bindings[key]=train[key]
    for key in ("cost_plan",):
        bound(cost[key]); bindings[key]=cost[key]
    return values,bindings


def choose(rows,**where):
    found=[row for row in rows if all(row.get(key)==value for key,value in where.items())]
    require(len(found)==1,"UNIQUE_REPORT_ROW:"+str(where)); return found[0]


def collect_tables(values):
    train,cost=values["train"],values["cost"]; tables={key:[] for key in TABLES}
    expected={f"{data}_{lag}_{'E_LEARN' if learn else 'E_FIXED'}_s{seed}" for data in DATASETS for lag in LAGS for learn in (False,True) for seed in SEEDS}
    require({name for item in train["datasets"].values() for name in item["conditions"]}==expected,"EXACT_16_MATCHED_CONDITIONS")
    def scores(base,value):
        for mask,key in (("all_observed","scores_all_observed"),("complete_C","scores_complete_mask")):
            if value[key].get("status")=="NO_COMPLETE_C": continue
            for scale in SCALES: tables["SCORES"].append({**base,"mask":mask,"scale":scale,**value[key][scale]})
    for data,(_,_,dev) in DATASETS.items():
        for lag in LAGS:
            probe=values["probe"]["cases"][data][lag]; tables["PROBES"].append({"dataset":data,"lag":lag,**probe})
            for split,baseline in train["CP0_LS_and_RAW_baselines"][data][lag].items():
                scores({"dataset":data,"lag":lag,"seed":0,"learn_E":False,"method":"CP0_LS","split":split},baseline["CP0_LS"])
                if lag=="L0":
                    for scale in SCALES: tables["SCORES"].append({"dataset":data,"lag":"NA","seed":0,"learn_E":False,"method":"RAW_BOLT","split":split,"mask":"all_observed","scale":scale,**baseline["RAW_scores"][scale]})
        for name,row in train["datasets"][data]["conditions"].items():
            require(row["status"]=="COMPLETE" and row["checkpoint_count"]==5,"COMPLETE_CONDITION_REQUIRED")
            base={"dataset":data,"condition":name,"lag":row["lag"],"learn_E":row["learn_E"],"seed":row["seed"]}
            checkpoints=row["checkpoints"]; require([cp["update"] for cp in checkpoints]==[0,100,200,400,600],"ALL_FIVE_CHECKPOINTS")
            selected=min(checkpoints,key=lambda cp:(cp["tune"]["mse"],cp["update"]))
            require(row["selected_update"]==selected["update"],"EARLIEST_TUNE_MINIMUM_REQUIRED")
            same_opaque(row["selected_state"],selected["state"])
            for cp in checkpoints: tables["CHECKPOINTS"].append({**base,"selected":cp["update"]==row["selected_update"],**cp})
            for split,value in (("fit",row["FIT"]),("tune",selected),(dev,row["DEV"])): scores({**base,"method":"SELECTED_ADAPTER","split":split},value)
            if row["seed"]==SEEDS[0] and not row["learn_E"]:
                scores({"dataset":data,"lag":row["lag"],"seed":0,"learn_E":False,"method":"CP0_LS","split":"tune"},row["CP0_TUNE"])
                if row["lag"]=="L0":
                    for scale in SCALES: tables["SCORES"].append({"dataset":data,"lag":"NA","seed":0,"learn_E":False,"method":"RAW_BOLT","split":"tune","mask":"all_observed","scale":scale,**row["RAW_TUNE_scores"][scale]})
            log=bound(row["updates_log"]); require(log.suffix==".jsonl","JSONL_UPDATES_ONLY")
            updates=[json.loads(line,parse_constant=lambda value: (_ for _ in ()).throw(ValueError("NONFINITE_JSON:"+value))) for line in log.read_text(encoding="utf-8").splitlines()]
            require([item["update"] for item in updates]==list(range(1,601)),"EXACT_UPDATE_SEQUENCE_600")
            require(sum(item["clipped"] for item in updates)/600==row["global_clipping_fraction"],"CLIP_FRACTION_EQUALS_UPDATE_LOG")
            tables["UPDATES"] += [{**base,**item} for item in updates]
            tables["CONDITIONS"].append({**base,**{key:value for key,value in row.items() if key not in ("checkpoints","FIT","DEV","CP0_TUNE","RAW_TUNE_scores")},
                "selected_parameter_diagnostics":selected["parameter_diagnostics"],"selected_TUNE_latent_variance":selected["TUNE_latent_variance"],
                "independent_parameter_change":values["audit"].get("datasets",{}).get(data,{}).get(name,{}).get("parameter_change"),
                "parameter_change_missing_is_unmeasured_not_zero":True})
            tables["SELECTION_LOCKS"].append({**base,"selected_update":row["selected_update"],"selection_lock":row["selection_lock"]})
    for data,(_,_,dev) in DATASETS.items():
        for lag in LAGS:
            for seed in SEEDS:
                for split in ("fit","tune",dev):
                    for mask in ("all_observed","complete_C"):
                        for scale in SCALES:
                            common={"dataset":data,"lag":lag,"seed":seed,"split":split,"mask":mask,"scale":scale,"method":"SELECTED_ADAPTER"}
                            fixed=choose(tables["SCORES"],**common,learn_E=False); learn=choose(tables["SCORES"],**common,learn_E=True)
                            tables["PAIRED_E"].append({**{k:v for k,v in common.items() if k!="method"},"fixed_MSE":fixed["mse"],"learn_MSE":learn["mse"],
                                "MSE_learn_minus_fixed":learn["mse"]-fixed["mse"],"fixed_MAE":fixed["mae"],"learn_MAE":learn["mae"],
                                "MAE_learn_minus_fixed":learn["mae"]-fixed["mae"],"paired_order_seed_only_not_independent_initialization":True})
                            if mask=="all_observed" and scale=="normalized":
                                initial=choose(tables["SCORES"],dataset=data,lag=lag,split=split,method="CP0_LS",mask=mask,scale=scale)
                                raw=choose(tables["SCORES"],dataset=data,split=split,method="RAW_BOLT",mask=mask,scale=scale)
                                require(fixed["observed_counts"]==learn["observed_counts"]==initial["observed_counts"]==raw["observed_counts"],"SAME_PER_CHANNEL_OBSERVED_COUNTS")
                                for channel,count in enumerate(fixed["observed_counts"]):
                                    f,l,p,r=(item["per_channel_mse"][channel] for item in (fixed,learn,initial,raw))
                                    tables["CHANNEL_COMPARISONS"].append({"dataset":data,"lag":lag,"seed":seed,"split":split,"channel_index":channel,"observed_count":count,
                                        "fixed_MSE":f,"learn_MSE":l,"CP0_LS_MSE":p,"RAW_MSE":r,"learn_minus_fixed":l-f if count else None,
                                        "learn_minus_CP0":l-p if count else None,"learn_minus_RAW":l-r if count else None,"horizon_errors_measured_in_metadata":False})
        for row in cost["cost"][data]["rows"]:
            raw=choose(cost["cost"][data]["rows"],route="RAW_BOLT",batch=row["batch"])
            require(len(row["seconds"])==len(row["samples"])==9,"NINE_TIMER_SAMPLES_REQUIRED")
            require(statistics.median(row["seconds"])==row["median_seconds"] and row["median_seconds_per_window"]==row["median_seconds"]/row["batch"],"EXACT_COST_MEDIAN")
            ratios=[a/b for a,b in zip(row["seconds"],raw["seconds"])]; paired=statistics.median(ratios)
            require(ratios==row["paired_RAW_ratios"] and paired==row["paired_RAW_median_ratio"],"EXACT_PAIRED_RAW_RATIO")
            tables["WHOLE_COST"].append({"dataset":data,**row,"ratio_of_RAW_medians":row["median_seconds"]/raw["median_seconds"],"median_paired_RAW_ratio":paired})
            tables["WHOLE_COST_SAMPLES"] += [{"dataset":data,"route":row["route"],"batch":row["batch"],"paired_RAW_ratio":ratios[i],**sample} for i,sample in enumerate(row["samples"])]
        for lag in LAGS:
            for learn in (False,True):
                for seed in SEEDS:
                    trained=choose(tables["SCORES"],dataset=data,lag=lag,learn_E=learn,seed=seed,split="tune",mask="all_observed",scale="normalized",method="SELECTED_ADAPTER")
                    raw=choose(tables["SCORES"],dataset=data,split="tune",method="RAW_BOLT",mask="all_observed",scale="normalized")
                    route=f"{lag}_{'E_LEARN' if learn else 'E_FIXED'}_s{seed}"
                    timing=choose(tables["WHOLE_COST"],dataset=data,route=route,batch=8)
                    tables["ACCURACY_COST_REFERENCE"].append({"dataset":data,"lag":lag,"learn_E":learn,"seed":seed,"route":route,
                        "TUNE_MSE_over_RAW":trained["mse"]/raw["mse"],"TUNE_MAE_over_RAW":trained["mae"]/raw["mae"],
                        "B8_ratio_of_RAW_medians":timing["ratio_of_RAW_medians"],"B8_median_paired_RAW_ratio":timing["median_paired_RAW_ratio"],
                        "prior_bundle_reference_criteria_met":trained["mse"]<=raw["mse"]*1.01 and trained["mae"]<=raw["mae"]*1.01 and timing["ratio_of_RAW_medians"]<=.90,
                        "criterion_scope":"Unchanged prior-bundle reference criteria; cost plan written after training started and first four TUNE conditions exposed; not train preregistration; no DEV selection"})
    for stage in ("probe","train","cost"):
        tables["EXECUTION_RESOURCES"].append({"stage":stage,**values[stage].get("resource",values[stage].get("resource_terminal",{}))})
        tables["PRESERVED_FAILURES"] += [{"stage":stage,**item} for item in values[stage].get("failed_paths",[])]
    tables["PROVENANCE"]=[{"stage":stage,"metadata":{key:value for key,value in values[stage].items() if key not in ("datasets","cases","cost","protected_prior_records","input_records","CP0_LS_and_RAW_baselines")}} for stage in INPUTS]
    require(len(tables["CHECKPOINTS"])==80 and len(tables["UPDATES"])==9600 and len(tables["WHOLE_COST"])==36 and len(tables["WHOLE_COST_SAMPLES"])==324,"FULL_REPORT_COVERAGE")
    return tables


def rel(path,output): return os.path.relpath(resolve(path),output).replace("\\","/")


def portable(value,output):
    if isinstance(value,dict): return {key:rel(child,output) if key in ("path","snapshot_path") and isinstance(child,str) else portable(child,output) for key,child in value.items()}
    if isinstance(value,list): return [portable(child,output) for child in value]
    return value


def write_csv(path,rows,output):
    fields=list(dict.fromkeys(key for row in rows for key in row)) or ["no_failed_paths"]
    with path.open("x",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({key:json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(",",":")) if isinstance(value,(dict,list)) else value for key,value in portable(row,output).items()})


def figures(tables,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    folder=output/"figures"; folder.mkdir(); images=[]
    colors={False:"#0072B2",True:"#D55E00"}
    def save(fig,name):
        fig.tight_layout(rect=(0,0,1,.94))
        for suffix in ("png","svg"):
            path=folder/(name+"."+suffix); fig.savefig(path,dpi=150); images.append(path)
        plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for i,(data,(_,_,dev)) in enumerate(DATASETS.items()):
        for j,split in enumerate(("fit","tune",dev)):
            ax=axes[i,j]
            for li,lag in enumerate(LAGS):
                for learn in (False,True):
                    for si,seed in enumerate(SEEDS):
                        row=choose(tables["SCORES"],dataset=data,lag=lag,learn_E=learn,seed=seed,split=split,mask="all_observed",scale="normalized",method="SELECTED_ADAPTER")
                        ax.scatter(li+(-.13 if not learn else .13),row["mse"],color=colors[learn],marker="o" if si==0 else "x",s=45)
                initial=choose(tables["SCORES"],dataset=data,lag=lag,split=split,method="CP0_LS",mask="all_observed",scale="normalized")
                ax.plot([li-.25,li+.25],[initial["mse"]]*2,color="gray",linestyle=":")
            raw=choose(tables["SCORES"],dataset=data,split=split,method="RAW_BOLT",mask="all_observed",scale="normalized")
            ax.axhline(raw["mse"],color="black",linestyle="--"); ax.set_xticks([0,1],LAGS); ax.set_title(data+" / "+split); ax.set_ylabel("Full-observed normalized macro MSE"); ax.grid(alpha=.2)
    fig.legend(handles=[Line2D([0],[0],marker="o",color=colors[b],label="E learned" if b else "E fixed") for b in (False,True)]+[Line2D([0],[0],color="gray",linestyle=":",label="Same Torch CP0/LS"),Line2D([0],[0],color="black",linestyle="--",label="Raw Bolt")],loc="lower center",ncol=4)
    fig.suptitle("Same PCA / FORECAST-LS initialization; seeds change minibatch order only; exposed DEV is exploratory"); save(fig,"01_accuracy_and_cp0")
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for ax,data in zip(axes,DATASETS):
        rows=[r for r in tables["PAIRED_E"] if r["dataset"]==data and r["mask"]=="all_observed" and r["scale"]=="normalized"]
        for i,row in enumerate(rows): ax.scatter(i,row["MSE_learn_minus_fixed"],color=colors[row["split"]!="tune"],marker="o" if row["seed"]==SEEDS[0] else "x")
        ax.set_xticks(range(len(rows)),[r["lag"]+"/"+r["split"]+"/"+str(r["seed"])[-1] for r in rows],rotation=65,fontsize=7); ax.axhline(0,color="black"); ax.set_title(data); ax.set_ylabel("E learned MSE minus paired E fixed MSE"); ax.grid(alpha=.2)
    fig.suptitle("Paired same-lag/order-seed contrasts; negative is improvement; no significance or independent-holdout claim"); save(fig,"02_paired_e_contrasts")
    fig,axes=plt.subplots(2,2,figsize=(13,7))
    for di,data in enumerate(DATASETS):
        for li,lag in enumerate(LAGS):
            ax=axes[di,li]
            for learn in (False,True):
                for seed in SEEDS:
                    rows=[r for r in tables["CHECKPOINTS"] if r["dataset"]==data and r["lag"]==lag and r["learn_E"]==learn and r["seed"]==seed]
                    ax.plot([r["update"] for r in rows],[r["tune"]["mse"] for r in rows],color=colors[learn],linestyle="-" if seed==SEEDS[0] else "--",alpha=.8)
                    selected=next(r for r in rows if r["selected"]); ax.scatter(selected["update"],selected["tune"]["mse"],color=colors[learn],marker="*")
            ax.set_title(data+" / "+lag); ax.set_xlabel("Update (CP0/100/200/400/600)"); ax.set_ylabel("Entire TUNE normalized MSE"); ax.grid(alpha=.2)
    fig.suptitle("All 80 checkpoints; earliest minimum TUNE selects CP; DEV does not select checkpoint"); save(fig,"03_checkpoint_selection")
    fig,axes=plt.subplots(2,2,figsize=(13,7))
    for di,data in enumerate(DATASETS):
        for li,lag in enumerate(LAGS):
            ax=axes[di,li]
            for learn in (False,True):
                for seed in SEEDS:
                    rows=[r for r in tables["CHECKPOINTS"] if r["dataset"]==data and r["lag"]==lag and r["learn_E"]==learn and r["seed"]==seed]
                    first=rows[0]["parameter_diagnostics"]["E_l2_norm"]
                    ax.plot([r["update"] for r in rows],[r["parameter_diagnostics"]["E_l2_norm"]/first for r in rows],color=colors[learn],linestyle="-" if seed==SEEDS[0] else "--")
            p=choose(tables["PROBES"],dataset=data,lag=lag)["gradient_diagnostics"]
            ax.set_title(data+" / "+lag); ax.set_xlabel("Update"); ax.set_ylabel("E norm / initial E norm"); ax.grid(alpha=.2)
            ax.text(.02,.97,f"Probe E gradient via F={p['E_F_path_gradient_norm']:.4g}\nDirect encoded-past={p['E_direct_past_gradient_norm']:.4g}\nPresence does not prove necessity",transform=ax.transAxes,va="top",fontsize=8)
    fig.suptitle("Geometry is diagnostic: E norm is not E-change distance; singular values, variance and D gain preserved in CSV"); save(fig,"04_geometry_and_gradient_scope")
    fig,axes=plt.subplots(2,2,figsize=(14,8))
    for di,data in enumerate(DATASETS):
        for bi,batch in enumerate((1,8)):
            ax=axes[di,bi]; rows=[choose(tables["WHOLE_COST"],dataset=data,route=route,batch=batch) for route in ROUTES]
            for i,row in enumerate(rows):
                ax.plot([row["ratio_of_RAW_medians"],row["median_paired_RAW_ratio"]],[i,i],color="gray"); ax.scatter(row["ratio_of_RAW_medians"],i,color="#0072B2"); ax.scatter(row["median_paired_RAW_ratio"],i,facecolors="none",edgecolors="#D55E00")
            ax.set_yticks(range(9),[r["route"].replace("_s9260","/s") for r in rows],fontsize=7); ax.axvline(1,color="black",linestyle="--"); ax.set_xlabel("Same-cohort Raw ratio: filled medians / open paired median"); ax.set_title(data+" / B"+str(batch)); ax.grid(alpha=.2)
    fig.suptitle("New actual full-pipeline 36-cell / 324-sample cohort; 9 complete rotations; shared resident memory"); save(fig,"05_whole_pipeline_cost")
    fig,axes=plt.subplots(1,2,figsize=(15,4.5))
    for ax,(data,(channels,_,dev)) in zip(axes,DATASETS.items()):
        matrix=[[choose(tables["CHANNEL_COMPARISONS"],dataset=data,lag=lag,seed=seed,split=dev,channel_index=c)["learn_minus_fixed"] for c in range(channels)] for lag in LAGS for seed in SEEDS]
        maximum=max(abs(value) for row in matrix for value in row if value is not None) or 1.
        image=ax.imshow(matrix,cmap="RdBu_r",vmin=-maximum,vmax=maximum,aspect="auto")
        ax.set_xticks(range(channels),[str(c) for c in range(channels)],fontsize=8); ax.set_yticks(range(4),[lag+"/s"+str(seed)[-1] for lag in LAGS for seed in SEEDS]); ax.set_xlabel("Original C channel index; no supplied channel-name inference"); ax.set_title(data+" / exposed "+dev)
        fig.colorbar(image,ax=ax,label="Normalized per-channel MSE: E learned - E fixed")
    fig.suptitle("All paired DEV channels; red is worse / blue is better; full CP0/Raw deltas and observed counts in CSV"); save(fig,"06_channel_error_deltas")
    return images


def make_text(values,tables,bindings,output):
    paired=[r for r in tables["PAIRED_E"] if r["mask"]=="all_observed" and r["scale"]=="normalized"]
    tune=sum(r["MSE_learn_minus_fixed"]<0 for r in paired if r["split"]=="tune"); dev=sum(r["MSE_learn_minus_fixed"]<0 for r in paired if r["split"] in ("val","eval"))
    better_raw=sum(row["mse"]<choose(tables["SCORES"],dataset=row["dataset"],split=row["split"],method="RAW_BOLT",mask="all_observed",scale="normalized")["mse"]
        for row in tables["SCORES"] if row["method"]=="SELECTED_ADAPTER" and row["split"] in ("val","eval") and row["mask"]=="all_observed" and row["scale"]=="normalized")
    last_cp=sum(row["learn_E"] and row["selected_update"]==600 for row in tables["CONDITIONS"])
    b1_slower=sum(row["route"]!="RAW_BOLT" and row["batch"]==1 and row["ratio_of_RAW_medians"]>1 for row in tables["WHOLE_COST"])
    b8_faster=sum(row["route"]!="RAW_BOLT" and row["batch"]==8 and row["ratio_of_RAW_medians"]<1 for row in tables["WHOLE_COST"])
    lines=["# 같은 초기값에서 E 고정·갱신을 분리한 결과","",f"**E를 갱신한 조건이 같은 lag·minibatch-order seed의 E 고정보다 정규화 MSE를 낮춘 비교는 TUNE {tune}/8, 기노출 DEV {dev}/8이다. 그러나 Raw보다 기노출 DEV MSE가 작은 선정 조건은 {better_raw}/16이다.** 아래 원수치는 초기 CP0/LS 및 Raw와 함께 읽는다. D-only가 초기값에서 나빠진 것만으로 표현 학습의 성공이라고 판단하지 않는다. 실행·검산 완료와 연구 목적 해결은 별개다.","",
        "**F를 통과하는 gradient는 실제 확인했지만 필요성은 미확정이다.** 네 실제 probe는 loss→F future→F input→E 경로와 encoded-past 기준값/알려진 prefix에서 E로 가는 직접 경로를 분리했다. 총 E gradient만으로 F 경로를 증명하지 않았다. E 갱신이 좋아져도 stop-gradient-F나 동일 native 대조가 없어 Bolt gradient 또는 Bolt 자체의 필요성을 입증하지 않는다.","",
        "**정상 FIT 감독과 배포 정보 조건.** FIT 미래 Y는 actual-C 손실 target에만 쓰며 forward는 origin에서 알려진 과거만 받는다. E는 시점별 C→K 선형 행렬, center는 고정 buffer, F는 실제 frozen Bolt q.5 H64의 앞48, D는 origin-reference lag 선형 복원이다. K streams·K 과거 buffer·현재 C anchor를 쓰므로 strict K-only가 아니다. 미래 label 누출이 없다는 좁은 구현 검사와 독립 일반화는 구분한다.","",
        "Robin C17/K4와 Beijing C11/K2, L512/H48, FIT256/TUNE64와 기노출 VAL30/EVAL146, 원 scaler·origin·관측 mask를 유지했다. 초기 PCA와 각 lag의 FORECAST-LS D는 고정 출발점이며 PCA가 최선이라고 선정한 결과가 아니다. L0와 L7은 계수가 7배 달라 동일 용량 비교가 아니다. 같은 초기값의 seed92601/92602는 minibatch 순서만 바꾸므로 독립 초기화나 독립 데이터 반복으로 세지 않는다.","",
        "본 학습은 16 fit × 600 update = 9,600회이며 실제 기술 probe의 추가 4회 update와 별도로 기록했다. AdamW lr.001/weight_decay0, batch4, clip1, CP0/100/200/400/600을 고정했다. 손실은 FIT 전체의 고정 채널별 관측수로 가중한 `N_fit/B/C_active × Σ SSE_batch_c/count_fit_c`다. minibatch별 관측수 macro와 같다고 쓰지 않는다. 80개 CP 중 전체 TUNE 정규화 MSE가 최소인 것을 고르고 exact tie는 이른 CP를 택했다. 16개 잠금·registry가 모두 봉인된 뒤 DEV를 열었다.","","```text","dataset  lag  seed   CP_fixed CP_learn  TUNE_fixed TUNE_learn  DEV_fixed DEV_learn"]
    for data,(_,_,development) in DATASETS.items():
        for lag in LAGS:
            for seed in SEEDS:
                fixed=choose(tables["CONDITIONS"],dataset=data,lag=lag,seed=seed,learn_E=False); learned=choose(tables["CONDITIONS"],dataset=data,lag=lag,seed=seed,learn_E=True)
                t=choose(paired,dataset=data,lag=lag,seed=seed,split="tune"); d=choose(paired,dataset=data,lag=lag,seed=seed,split=development)
                lines.append(f"{('Robin' if data=='robin' else 'Beijing'):<8} {lag:<4} {seed} {fixed['selected_update']:>8} {learned['selected_update']:>8} {t['fixed_MSE']:.9f} {t['learn_MSE']:.9f} {d['fixed_MSE']:.9f} {d['learn_MSE']:.9f}")
        raw=choose(tables["SCORES"],dataset=data,method="RAW_BOLT",split=development,mask="all_observed",scale="normalized")
        lines.append(f"{data} Raw exposed_DEV MSE={raw['mse']:.9f}")
    lines += ["```","","표는 채널별 관측 위치를 사용하는 channel-macro 정규화 MSE다. complete-C 및 물리 단위 점수는 별도 CSV에 보존했다.","",
        "![초기값 및 Raw와 비교](figures/01_accuracy_and_cp0.png)","[SVG](figures/01_accuracy_and_cp0.svg)","","![같은 lag·seed의 E 적응 차이](figures/02_paired_e_contrasts.png)","[SVG](figures/02_paired_e_contrasts.svg)","",
        "![모든 checkpoint와 TUNE 선정](figures/03_checkpoint_selection.png)","[SVG](figures/03_checkpoint_selection.svg)","",
        f"E 갱신 8조건 중 {last_cp}/8은 마지막 CP600을 선택했다. 따라서 600 update의 비교 결과를 수렴이나 충분한 훈련의 증거로 해석하지 않는다. 더 오래 학습할 때 달라질 가능성은 남지만 추가 연장은 아직 미계획·미실행이다. 현재 원수치나 선정 CP를 바꾸지 않는다.","",
        "**학습 안정성과 표현 유용성은 따로 본다.** 전체 update의 손실·E/D gradient·clip 여부와 모든 CP의 E/D singular value·TUNE latent variance·D feature→C gain을 보존했다. E norm 변화는 초기→최종 E 거리와 다르며 gain은 공동 H-horizon 미래 오차의 상한이 아니다. 독립 감사가 저장 상태로 계산한 초기→선정 E/D 거리·E rank는 CONDITIONS에 원 metadata로 보존한다. 없는 진단값을 0으로 추정하지 않는다. 입력 NaN을 mask 뒤에서 제곱해 gradient를 오염시키지 않도록 미관측 target을 차이·제곱 전에 안전하게 처리했다. finite 통과나 nonzero gradient만으로 예측 표현이 적합하다고 결론 내리지 않는다. 감사는 저장된 실제 F 미래와 계수의 산술·점수·선정·로그를 검산했으며 F forward·backward·optimizer와 9,600 손실값을 독립 재실행한 것은 아니다. 네 probe의 VJP 수치는 실제 producer 증거이며 독립 VJP replay로 바꾸어 말하지 않는다.","",
        "![계수와 gradient 경로](figures/04_geometry_and_gradient_scope.png)","[SVG](figures/04_geometry_and_gradient_scope.svg)","",
        f"**압축 경로가 언제나 빠른 것은 아니다.** 같은 cohort Raw와 비교한 ratio-of-medians에서 선정 16조건 중 B1이 더 느린 조건은 {b1_slower}/16, B8이 더 빠른 조건은 {b8_faster}/16이다. batch 크기와 Torch E/lag D 및 CPU↔GPU 처리의 전체 비용을 함께 본다. 이 측정만으로 어느 단계가 차이를 만들었는지 인과 분해하지 않는다.","",
        "**비용은 별도 실제 전체 추론이다.** 새 9경로 × 2자료 × B1/B8 = 36조건, warmup2·9완전순환의 324표본이다. 첫 TUNE origins와 선정 CP의 동일 TUNE 예측을 reference로 쓰며 DEV timing은 하지 않았다. 원 physical causal fill/FIT 정규화부터 선택된 Torch E→실제 frozen GPU F→Torch lag D/currentC anchor→물리 단위 CPU 반환과 최종 sync를 포함한다. 데이터·모델·선정 E/D setup 및 학습·reference·warmup·저장·source/file/state metadata hash·guard·memory snapshot은 타이머 밖이다. 새 E/D 적합·native head 실행은 없다. 같은 첫 B TUNE origins를 반복하므로 324개의 독립 평가 자료가 아니다. 모든 E/D와 F가 resident인 공유 process RSS/VRAM을 방법별 독점 메모리나 장비 독점으로 바꾸어 말하지 않는다.","","```text","dataset  route                   B8_ms/window ratio_medians median_paired"]
    for data in DATASETS:
        for route in ROUTES:
            row=choose(tables["WHOLE_COST"],dataset=data,route=route,batch=8)
            lines.append(f"{('Robin' if data=='robin' else 'Beijing'):<8} {route:<25} {row['median_seconds_per_window']*1000:.6f} {row['ratio_of_RAW_medians']:.6f} {row['median_paired_RAW_ratio']:.6f}")
    lines += ["```","","![같은 cohort의 두 비용 비율](figures/05_whole_pipeline_cost.png)","[SVG](figures/05_whole_pipeline_cost.svg)","",
        "ratio-of-medians와 반복별 paired ratio의 중앙값은 다른 통계라 둘 다 보존했다. ACCURACY_COST_REFERENCE의 ×1.01 MSE/MAE·B8≤.90은 이전 묶음 기준을 그대로 참고 적용한 값이다. 비용 계획은 본 학습 시작 뒤·첫 네 조건 TUNE 노출 뒤에 작성됐으므로 이번 학습의 사전등록 gate라고 부르지 않는다. 조건 선정은 TUNE MSE뿐이며 DEV 최저 조건으로 선택을 바꾸지 않는다.","",
        "**채널별 실패 위치.** 같은 lag·seed의 8개 비교에서 전체 채널의 E 갱신−고정/CP0/Raw 정규화 MSE 차이와 관측수를 CHANNEL_COMPARISONS에 보존했다. 아래 그림의 channel index는 원 순서이며 metadata에 없는 채널명을 추정하지 않았다. horizon별 오차는 metadata에 없어 미측정이다. 한 채널의 큰 악화가 전체 macro 오차에 미친 흔적과 기저표현의 원인을 구분한다.","","![기노출 DEV의 채널별 차이](figures/06_channel_error_deltas.png)","[SVG](figures/06_channel_error_deltas.svg)","",
        "**남은 불확실성.** E 갱신이 개선해도 선형 E/D 공동 적응 안의 결과다. 이득이 없으면 현재 초기 PCA·작은 K·선형 lag·600 update 조건의 한계이며 모든 예측 목적 표현의 불가능성이 아니다. 설계는 이미 노출된 DEV를 본 뒤의 적응적 탐색이고 독립 일반화는 확인하지 않았다. 과거 joint-Y와 RF3도 frozen F를 통한 E/D 학습을 실제 수행했다. 이번을 최초 end-to-end 학습 또는 재구성만 했던 연구를 처음 예측 목적으로 바꾼 시도로 포장하지 않는다. 새 대조는 같은 선형 초기값·origin 기준 D에서 E 적응을 분리한다는 것이다.","",
        "전체 원수치와 재현 근거는 다음 상대 링크에 있다.",""]
    lines += [f"- [{name}.csv](tables/{name}.csv): {len(rows)}행" for name,rows in tables.items()]
    lines += [""]+[f"- [{name}]({rel(item['path'],output)}): `{item['sha256']}`" for name,item in bindings.items()]
    lines += [""]+[f"- [{row['condition']} 원 학습 곡선]({rel(row['updates_log']['path'],output)}): `{row['updates_log']['sha256']}`" for row in tables["CONDITIONS"]]
    lines += ["","보고서 생성기는 JSON·JSONL metadata만 읽었다. 계수·예측 원배열은 독립 감사가 검산한 원 artifact에 두고 NPZ·모델을 다시 열거나 F/학습/타이머를 재실행하지 않았다. 기존 파일·실패 기록은 보존하며 보고서 생성은 Git push나 논문/PPT 완료를 뜻하지 않는다.",""]
    return "\n".join(lines)


def generate(paths,output_path):
    output=resolve(output_path); require(output.parent==(HERE/"reports").resolve() and not output.exists(),"FRESH_LOCAL_REPORT_DIRECTORY_REQUIRED")
    values,bindings=load_verified(paths); tables=collect_tables(values); source=record(__file__); bindings["report_source"]=source
    output.mkdir(parents=True,exist_ok=False); (output/"tables").mkdir()
    for name,rows in tables.items(): write_csv(output/"tables"/(name+".csv"),rows,output)
    images=figures(tables,output)
    with (output/"REPORT_KO.md").open("x",encoding="utf-8",newline="\n") as handle: handle.write(make_text(values,tables,bindings,output))
    require(record(__file__)==source,"REPORT_SOURCE_UNCHANGED")
    for item in bindings.values(): bound(item)
    files=[{"path":path.relative_to(output).as_posix(),"sha256":sha(path),"bytes":path.stat().st_size} for path in sorted(output.rglob("*")) if path.is_file()]
    manifest={"status":"COMPLETE","source":source,"inputs":bindings,"files":files,"table_row_counts":{key:len(rows) for key,rows in tables.items()},
        "figure_png_count":6,"figure_svg_count":6,"scientific_NPZ_or_model_read":False,"new_fit_F_or_training_executed":False,
        "protected_TEST_accessed":False,"independent_generalization_claimed":False,"all_16_conditions_80_checkpoints_9600_updates_preserved":True,
        "CSV_artifact_paths_relative_to_report_directory":True,"gradient_presence_not_necessity":True}
    with (output/"MANIFEST.json").open("x",encoding="utf-8",newline="\n") as handle: json.dump(manifest,handle,ensure_ascii=False,indent=2,allow_nan=False); handle.write("\n")
    print(json.dumps({"status":"COMPLETE","manifest":str(output/"MANIFEST.json"),"tables":len(tables),"files":len(files),"images":len(images)}),flush=True)
    return manifest


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in (*INPUTS,"output"): parser.add_argument("--"+key.replace("_","-"),required=True)
    arguments=vars(parser.parse_args()); output=arguments.pop("output"); generate(arguments,output)
