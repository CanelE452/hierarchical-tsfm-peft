"""JSON/JSONL-only comparison of reused full-gradient and fresh stop-gradient fits."""
from pathlib import Path
import argparse
import json
import statistics

from make_linear_e_adaptation_report_v2 import require,resolve,sha,record,read,bound,same_opaque,choose,rel,write_csv


HERE=Path(__file__).resolve().parent
DATASETS={"robin":(17,4,"val"),"beijing_aotizhongxin":(11,2,"eval")}
SEEDS=(92601,92602)
ROUTES=("RAW_BOLT","FULL_s92601","FULL_s92602","STOP_s92601","STOP_s92602")
INPUTS=("train","audit","cost","cost_audit")
FIXED_SHA={"train":"e27989fb23716be8bbf507442d44033884b28522d641a010d18b426a0eb6d10a",
    "audit":"64f19658ac8bc8cee1d6fa3713113a7146ba282ebb8ad2b9eb49498d91e6418f",
    "cost":"354cbeb47e298c1f313c08c52a61abeb19020396dfa1980e4c12b41647e2c9de",
    "cost_audit":"0a70e0a1be13044846bca8bbb1414794857ee7e6eb6f76f43d679c4070f835dc"}
AUDITOR_SOURCES={"audit":{"path":str(HERE/"audit_f_input_gradient_ablation_v1.py"),"sha256":"512629c4bc02e5dd08d05cb1cad64da5d5036e13ef8b457faf98a2a619d0c232"},
    "cost_audit":{"path":str(HERE/"audit_f_input_gradient_cost_v1.py"),"sha256":"9aaff46507a0e9f35d77f5a44a28d91ae0826d2bd0822a28d341e548220a0db4"}}
AUDIT_COUNTS={"stop_gradient_fits_checked":4,"checkpoint_states_checked":20,"checkpoint_TUNE_predictions_checked":20,"selected_FIT_DEV_predictions_checked":8,
    "logged_updates_checked":2400,"probe_cases_checked":4,"reused_full_gradient_conditions_checked":4,"CP0_full_stop_parent_component_byte_comparisons_checked":32,
    "saved_gradient_vector_fields_checked":32,"analytic_gradient_comparisons_checked":28,"score_fields_checked":792,"selection_locks_checked":4,"protected_prior_file_hashes_checked":1485}
COST_COUNTS={"cost_cells_checked":20,"recorded_timer_samples_checked":100,"selected_state_hashes_checked":8,"selection_locks_checked":8,
    "physical_prechecks_checked":8,"physical_first8_windows_checked":64,"canonical_TUNE_bridges_checked":2}
PARENT_SHA="3f0cbd9176f2e7882778452e537564aab7445e1b88c1534fd48b1bce7ef639a4"
PARENT_AUDIT_SHA="9796b4ed53f74186bc2513db56fe321c2d450cdd9bffbb7c5f341766e73ee36d"
UTILITY={"path":str(HERE/"make_linear_e_adaptation_report_v2.py"),"sha256":"32e2460ea4724144067f264ab1ceb0332cd04c5fce2296838024c119c9572466"}
TABLES=("SCORES","PAIRED","CHECKPOINTS","UPDATES","CLIP_SUMMARY","PROBES","SELECTION","WHOLE_COST","WHOLE_COST_SAMPLES","RESOURCES","PROVENANCE")
SCALES=("normalized","physical_per_channel_diagnostic")


def load_verified(paths):
    require(set(paths)==set(INPUTS),"FOUR_EXACT_STAGE_INPUTS")
    require(all(isinstance(FIXED_SHA[k],str) and len(FIXED_SHA[k])==64 for k in INPUTS) and all(isinstance(v,dict) for v in AUDITOR_SOURCES.values())
        and bool(AUDIT_COUNTS) and bool(COST_COUNTS),"ACTUAL_TRAIN_COST_PASS_NOT_YET_REGISTERED")
    bound(UTILITY); values={k:read(p) for k,p in paths.items()}; bindings={k:record(p) for k,p in paths.items()}
    for key,value in values.items():
        require(bindings[key]["sha256"]==FIXED_SHA[key] and value["status"]==("PASS" if "audit" in key else "COMPLETE") and value["protected_TEST_accessed"] is False,"EXACT_COMPLETE_PASS_NO_TEST:"+key)
    train,audit,cost,costaudit=(values[k] for k in INPUTS)
    bound(audit["diagnostic"],bindings["train"]); bound(costaudit["cost"],bindings["cost"])
    for key,counts in (("audit",AUDIT_COUNTS),("cost_audit",COST_COUNTS)):
        require(all(values[key]["totals"].get(k)==v for k,v in counts.items()),"FULL_INDEPENDENT_AUDIT_COVERAGE")
        bound(values[key]["audit_source"],AUDITOR_SOURCES[key]); bindings[key+"_source"]=values[key]["audit_source"]
    for value in (cost,costaudit):
        bound(value["stop_gradient_diagnostic"],bindings["train"]); bound(value["stop_gradient_audit"],bindings["audit"])
        bound(value["selection_registry"],train["selection_registry"])
    for key,wanted in (("full_gradient_parent",PARENT_SHA),("full_gradient_parent_audit",PARENT_AUDIT_SHA)):
        bound(train[key]); require(train[key]["sha256"]==wanted,"EXACT_REUSED_PARENT")
        values["parent" if key=="full_gradient_parent" else "parent_audit"]=read(train[key]["path"]); bindings[key]=train[key]
        for value in (audit,cost,costaudit): bound(value[key],train[key])
    require(values["parent"]["status"]=="COMPLETE" and values["parent_audit"]["status"]=="PASS","PRESERVED_FULL_GRADIENT_COMPLETE_PASS")
    require(train["stop_gradient_training_fits"]==train["resolved_selection_locks"]==4 and train["training_updates_completed"]==2400 and train["checkpoint_states_saved"]==20
        and train["probe_optimizer_updates"]==train["new_full_gradient_fits"]==0 and train["full_gradient_conditions_reused"]==4 and train["all_4_locks_before_DEV"] is True and not train["failed_paths"],"FOUR_FRESH_FITS_WITH_REUSED_CONTROLS")
    for key in ("new_backbone_training","independent_holdout","strict_K_only","whole_deployment_cost_measured"): require(train[key] is False,"LIMITED_ABLATION_SCOPE")
    require(train["F_before"]==train["F_after"]==values["parent"]["F_before"] and cost["F_before"]==cost["F_after"]==train["F_before"],"SAME_FROZEN_F")
    require(tuple(cost["routes"])==ROUTES and cost["cost_split"]=="tune" and cost["completed_cost_cells"]==20 and cost["recorded_timer_samples"]==100 and not cost["failed_paths"]
        and cost["new_training_updates"]==0 and cost["cached_future_returned_instead_of_actual_F_inside_timer"] is False,"NEW_FIVE_ROUTE_ACTUAL_WHOLE_COST")
    registry=read(bound(train["selection_registry"]))
    require(registry["status"]=="ALL_FOUR_RESOLVED_BEFORE_DEV" and registry["DEV_opened_before_seal"] is False and len(registry["conditions"])==4,"FOUR_RESOLVED_LOCKS_BEFORE_DEV")
    for item in registry["conditions"]:
        row=train["datasets"][item["dataset"]]["conditions"][str(item["seed"])]; lock=read(bound(item["selection_lock"],row["selection_lock"]))
        require(lock["status"]=="COMPLETE" and lock["DEV_opened_before_lock"] is False and lock["selected_update"]==row["selected_update"],"SUCCESSFUL_TUNE_ONLY_SELECTION_LOCK")
        same_opaque(lock["selected_state"],row["selected_state"])
    for key in ("controller_source","plan","parent_report","parent_report_manifest","parent_report_audit","selection_registry"):
        bound(train[key]); bindings[key]=train[key]
    bound(cost["cost_plan"]); bindings["cost_plan"]=cost["cost_plan"]; bindings["report_utility"]=UTILITY
    return values,bindings


def selected(row):
    cps=row["checkpoints"]; require([c["update"] for c in cps]==[0,100,200,400,600],"ALL_FIVE_CP")
    result=min(cps,key=lambda cp:(cp["tune"]["mse"],cp["update"]))
    require(row["selected_update"]==result["update"],"EARLIEST_TUNE_MINIMUM_REQUIRED"); same_opaque(row["selected_state"],result["state"])
    return result


def collect_tables(values):
    train,parent,cost=values["train"],values["parent"],values["cost"]; tables={key:[] for key in TABLES}
    require({(d,int(seed)) for d,item in train["datasets"].items() for seed in item["conditions"]}=={(d,s) for d in DATASETS for s in SEEDS},"EXACT_FOUR_MATCHED_CONDITIONS")
    def add_scores(base,entry):
        for mask,key in (("all_observed","scores_all_observed"),("complete_C","scores_complete_mask")):
            if entry[key].get("status")=="NO_COMPLETE_C": continue
            for scale in SCALES: tables["SCORES"].append({**base,"mask":mask,"scale":scale,**entry[key][scale],"artifact":entry.get("artifact")})
    for data,(_,_,dev) in DATASETS.items():
        for seed in SEEDS:
            full=parent["datasets"][data]["conditions"][f"{data}_L0_E_LEARN_s{seed}"]; stop=train["datasets"][data]["conditions"][str(seed)]
            require(stop["order_sha256"]==full["order_sha256"],"SAME_MINIBATCH_ORDER")
            fixed=parent["datasets"][data]["conditions"][f"{data}_L0_E_FIXED_s{seed}"]
            for method,row in (("FULL_GRADIENT",full),("STOP_GRADIENT",stop),("E_FIXED",fixed)):
                cp=selected(row); base={"dataset":data,"seed":seed,"method":method,"new_execution":method=="STOP_GRADIENT"}
                for split,entry in (("fit",row["FIT"]),("tune",cp),(dev,row["DEV"])): add_scores({**base,"split":split},entry)
                if method=="E_FIXED": continue
                tables["CHECKPOINTS"] += [{**base,**item,"selected":item["update"]==row["selected_update"]} for item in row["checkpoints"]]
                tables["SELECTION"].append({**base,"selected_update":row["selected_update"],"selected_state":row["selected_state"],"selection_lock":row["selection_lock"],"order_sha256":row["order_sha256"],"updates_log":row["updates_log"]})
                log=bound(row["updates_log"]); require(log.suffix==".jsonl","JSONL_UPDATE_LOG_ONLY")
                updates=[json.loads(line,parse_constant=lambda v: (_ for _ in ()).throw(ValueError("NONFINITE_JSON:"+v))) for line in log.read_text(encoding="utf-8").splitlines()]
                require([u["update"] for u in updates]==list(range(1,601)),"EXACT_600_UPDATE_SEQUENCE")
                fraction=sum(u["clipped"] for u in updates)/600; require(fraction==row["global_clipping_fraction"],"CLIP_FRACTION_MATCHES_LOG")
                logs=[]
                for item in updates:
                    scale=min(1.,1./(item["global_gradient_norm_before_clip"]+1e-6))
                    logs.append({**base,**item,"global_clip_scale_from_logged_norm":scale,"D_gradient_norm_after_clip_formula":scale*item["D_gradient_norm_before_clip"],"D_gradient_scaling_is_AdamW_step_norm":False})
                tables["UPDATES"]+=logs
                tables["CLIP_SUMMARY"].append({**base,"clipping_fraction":fraction,"mean_reconstructed_clip_scale":statistics.mean(r["global_clip_scale_from_logged_norm"] for r in logs),
                    "mean_D_preclip_norm":statistics.mean(r["D_gradient_norm_before_clip"] for r in logs),"mean_D_afterclip_formula_norm":statistics.mean(r["D_gradient_norm_after_clip_formula"] for r in logs),
                    "actual_per_step_AdamW_parameter_update_norm_measured":False,"CP0_parameter_diagnostics":row["checkpoints"][0]["parameter_diagnostics"],"selected_parameter_diagnostics":cp["parameter_diagnostics"]})
            p=stop["probe"]; require(p["status"]=="PASS" and p["optimizer_updates"]==0 and p["forward_byte_exact"] is True and p["finite_gradient_vectors"] is True
                and all(item["pass"] for item in p["gradient_checks"].values()),"ZERO_UPDATE_MATCHED_PROBE_PASS_REQUIRED")
            tables["PROBES"].append({"dataset":data,"seed":seed,**p,"CP0_TUNE_byte_checks":stop.get("CP0_TUNE_byte_checks"),"CP0_TUNE_parity_artifact":stop.get("CP0_TUNE_parity_artifact")})
        baseline=parent["CP0_LS_and_RAW_baselines"][data]["L0"]
        for split in ("fit","tune",dev):
            initial=full["CP0_TUNE"] if split=="tune" else baseline[split]["CP0_LS"]
            raw=full["RAW_TUNE_scores"] if split=="tune" else baseline[split]["RAW_scores"]
            add_scores({"dataset":data,"seed":0,"method":"CP0_LS","split":split,"new_execution":False},initial)
            for scale in SCALES: tables["SCORES"].append({"dataset":data,"seed":0,"method":"RAW_BOLT","split":split,"mask":"all_observed","scale":scale,"new_execution":False,**raw[scale]})
        for seed in SEEDS:
            for split in ("fit","tune",dev):
                for mask in ("all_observed","complete_C"):
                    for scale in SCALES:
                        where={"dataset":data,"seed":seed,"split":split,"mask":mask,"scale":scale}
                        full_score=choose(tables["SCORES"],**where,method="FULL_GRADIENT"); stop_score=choose(tables["SCORES"],**where,method="STOP_GRADIENT")
                        tables["PAIRED"].append({**where,"full_MSE":full_score["mse"],"stop_MSE":stop_score["mse"],"MSE_stop_minus_full":stop_score["mse"]-full_score["mse"],
                            "full_MAE":full_score["mae"],"stop_MAE":stop_score["mae"],"MAE_stop_minus_full":stop_score["mae"]-full_score["mae"]})
        for row in cost["cost"][data]["rows"]:
            raw=choose(cost["cost"][data]["rows"],route="RAW_BOLT",batch=row["batch"]); ratios=[a/b for a,b in zip(row["seconds"],raw["seconds"])]
            require(len(row["samples"])==len(row["seconds"])==5 and statistics.median(row["seconds"])==row["median_seconds"] and ratios==row["paired_RAW_ratios"]
                and statistics.median(ratios)==row["paired_RAW_median_ratio"],"FIVE_TIMER_SAMPLES_AND_TWO_RATIOS")
            tables["WHOLE_COST"].append({"dataset":data,**row,"ratio_of_RAW_medians":row["median_seconds"]/raw["median_seconds"],"median_paired_RAW_ratio":statistics.median(ratios)})
            tables["WHOLE_COST_SAMPLES"] += [{"dataset":data,"route":row["route"],"batch":row["batch"],**item,"paired_RAW_ratio":ratios[i]} for i,item in enumerate(row["samples"])]
    tables["RESOURCES"]=[{"stage":key,**values[key].get("resource_terminal",{})} for key in ("train","cost")]
    tables["PROVENANCE"]=[{"stage":key,"status":values[key]["status"],"totals":values[key].get("totals"),"failed_paths":values[key].get("failed_paths",[]),"scope":values[key].get("scope"),"D_step_scope":values[key].get("D_step_scope")} for key in INPUTS]
    require(len(tables["PAIRED"])==48 and len(tables["CHECKPOINTS"])==40 and len(tables["UPDATES"])==4800 and len(tables["WHOLE_COST"])==20,"FULL_MATCHED_REPORT_COVERAGE")
    return tables


def figures(tables,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    folder=output/"figures"; folder.mkdir(); files=[]; colors={"FULL_GRADIENT":"#D55E00","STOP_GRADIENT":"#009E73","E_FIXED":"#0072B2"}
    legend=[Line2D([0],[0],color=color,label=method) for method,color in colors.items()]+[Line2D([0],[0],marker="o" if seed==SEEDS[0] else "x",linestyle="none",color="black",label="Order seed "+str(seed)) for seed in SEEDS]
    def save(fig,name,bottom=.10):
        fig.tight_layout(rect=(0,bottom,1,.93))
        for suffix in ("png","svg"):
            path=folder/(name+"."+suffix); fig.savefig(path,dpi=150); files.append(path)
        plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,7))
    for di,(data,(_,_,dev)) in enumerate(DATASETS.items()):
        for si,split in enumerate(("tune",dev)):
            ax=axes[di,si]
            for mi,method in enumerate(colors):
                for seed in SEEDS:
                    r=choose(tables["SCORES"],dataset=data,seed=seed,method=method,split=split,mask="all_observed",scale="normalized")
                    ax.scatter(mi+(-.08 if seed==SEEDS[0] else .08),r["mse"],color=colors[method],marker="o" if seed==SEEDS[0] else "x")
            for method,style in (("RAW_BOLT","--"),("CP0_LS",":")):
                baseline=choose(tables["SCORES"],dataset=data,method=method,split=split,mask="all_observed",scale="normalized")
                ax.axhline(baseline["mse"],color="black" if method=="RAW_BOLT" else "gray",linestyle=style,label=method)
            ax.set_xticks(range(3),["Full (reused)","Stop (new)","E fixed (reused)"]); ax.set_title(data+" / "+split); ax.set_ylabel("Full-observed normalized macro MSE"); ax.grid(alpha=.2)
    fig.legend(handles=legend+[Line2D([0],[0],color="black",linestyle="--",label="Raw Bolt"),Line2D([0],[0],color="gray",linestyle=":",label="CP0/LS")],loc="lower center",ncol=4)
    fig.suptitle("L0 only: same initialization/order seed; exposed DEV is exploratory; removing a gradient is not removing F values"); save(fig,"01_matched_accuracy")
    fig,axes=plt.subplots(2,2,figsize=(13,7))
    for di,data in enumerate(DATASETS):
        ax=axes[0,di]
        for method in ("FULL_GRADIENT","STOP_GRADIENT"):
            for seed in SEEDS:
                rows=[r for r in tables["CHECKPOINTS"] if r["dataset"]==data and r["method"]==method and r["seed"]==seed]
                ax.plot([r["update"] for r in rows],[r["tune"]["mse"] for r in rows],color=colors[method],linestyle="-" if seed==SEEDS[0] else "--")
                cp=next(r for r in rows if r["selected"]); ax.scatter(cp["update"],cp["tune"]["mse"],color=colors[method],marker="*")
        ax.set_title(data+" / all TUNE checkpoints"); ax.set_xlabel("Update"); ax.set_ylabel("TUNE MSE"); ax.grid(alpha=.2)
        rows=[r for r in tables["CLIP_SUMMARY"] if r["dataset"]==data]; ax=axes[1,di]
        ax.bar(range(4),[r["clipping_fraction"] for r in rows],color=[colors[r["method"]] for r in rows]); ax.set_xticks(range(4),[r["method"].split("_")[0]+"/s"+str(r["seed"])[-1] for r in rows]); ax.set_ylim(0,1); ax.set_ylabel("Fraction of updates globally clipped"); ax.set_title("D scaling interacts with E; not AdamW step norm")
    fig.legend(handles=[Line2D([0],[0],color=colors[m],label=m) for m in ("FULL_GRADIENT","STOP_GRADIENT")]+[Line2D([0],[0],color="black",linestyle="-" if s==SEEDS[0] else "--",label="Order seed "+str(s)) for s in SEEDS],loc="lower center",ncol=4)
    fig.suptitle("20 new + 20 reused checkpoints; 2,400 new + 2,400 reused update logs; no DEV checkpoint selection"); save(fig,"02_checkpoints_and_clipping")
    fig,axes=plt.subplots(2,2,figsize=(13,7))
    for ax,row in zip(axes.flat,tables["PROBES"]):
        keys=("full_E_gradient","F_path_E_gradient","direct_E_gradient","stop_E_gradient","full_D_gradient","stop_D_gradient")
        ax.bar(range(6),[row["gradient_norms"][key] for key in keys],color=["#D55E00","#CC79A7","#0072B2","#009E73","gray","gray"])
        ax.set_xticks(range(6),["E full","E via F","E direct","E stop","D full","D stop"],rotation=20); ax.set_ylabel("Preclip Euclidean gradient norm")
        err=row["gradient_checks"]["stop_E_equals_full_minus_F_path"]["max_abs"]
        ax.set_title(row["dataset"]+" / order seed "+str(row["seed"])+f"\nE identity max-abs={err:.4g}; probe optimizer updates=0",fontsize=10); ax.grid(axis="y",alpha=.2)
    fig.suptitle("Four first-minibatch gradient probes; actual saved vectors and audit scope remain separate from performance necessity"); save(fig,"03_gradient_probes",bottom=.03)
    fig,axes=plt.subplots(2,2,figsize=(13,7))
    for di,data in enumerate(DATASETS):
        for bi,batch in enumerate((1,8)):
            ax=axes[di,bi]; rows=[choose(tables["WHOLE_COST"],dataset=data,route=route,batch=batch) for route in ROUTES]
            for i,row in enumerate(rows):
                ax.plot([row["ratio_of_RAW_medians"],row["median_paired_RAW_ratio"]],[i,i],color="gray"); ax.scatter(row["ratio_of_RAW_medians"],i,color="#0072B2"); ax.scatter(row["median_paired_RAW_ratio"],i,facecolors="none",edgecolors="#D55E00")
            ax.set_yticks(range(5),[r["route"].replace("_s9260","/s") for r in rows],fontsize=8); ax.axvline(1,color="black",linestyle="--"); ax.set_title(data+" / B"+str(batch)); ax.set_xlabel("Raw ratio: filled medians; open paired median"); ax.grid(alpha=.2)
    fig.suptitle("New TUNE first-B cohort: 20 whole-pipeline cells / 100 samples / 5 complete rotations; shared resident memory"); save(fig,"04_whole_pipeline_cost",bottom=.03)
    return files


def make_text(values,tables,bindings,output):
    paired=[r for r in tables["PAIRED"] if r["mask"]=="all_observed" and r["scale"]=="normalized"]
    worse_tune=sum(r["MSE_stop_minus_full"]>0 for r in paired if r["split"]=="tune"); worse_dev=sum(r["MSE_stop_minus_full"]>0 for r in paired if r["split"] in ("val","eval"))
    better_raw=sum(r["mse"]<choose(tables["SCORES"],dataset=r["dataset"],split=r["split"],method="RAW_BOLT",mask="all_observed",scale="normalized")["mse"] for r in tables["SCORES"] if r["method"] in ("FULL_GRADIENT","STOP_GRADIENT") and r["split"] in ("val","eval") and r["mask"]=="all_observed" and r["scale"]=="normalized")
    lines=["# F 입력 gradient를 끊은 L0 대조 결과","",f"**같은 초기값·minibatch 순서에서 stop-gradient가 기존 full-gradient보다 정규화 MSE가 큰 비교는 TUNE {worse_tune}/4, 기노출 DEV {worse_dev}/4다. 두 방식의 8조건 중 Raw보다 DEV MSE가 작은 조건은 {better_raw}/8이다.** 이 결과는 현재 L0·선형 E/D·600 update의 최적화 대조이며 연구 전체가 해결됐다는 뜻이 아니다.","",
        "검증한 질문은 **F 입력으로 돌아가는 gradient가 이 구성의 E 적응에 도움을 주는가**다. F 출력 tensor에만 detach를 적용하고, E가 만든 과거 기준값의 직접 경로와 D 학습을 유지했다. F 예측값은 제거하지 않았다. 매번 갱신된 E의 과거를 실제 frozen Bolt에 넣어 미래를 새로 계산했으며 캐시나 실제 미래 Y로 대체하지 않았다. F 입력 gradient·F 예측값·Bolt/TSFM 자체의 필요성은 서로 다른 질문이다. 이번 결과만으로 native·대조학습·그래프 또는 F 예측값의 필요성을 증명하지 않는다.","",
        "Robin C17/K4와 Beijing C11/K2, L512/H48, FIT256/TUNE64·기노출 VAL30/EVAL146, 초기 PCA/FORECAST-LS E/D·고정 center·current-C anchor·관측 mask·scaler를 유지했다. 정보 인터페이스는 K streams+K past buffer+current-C anchor이므로 strict K-only 압축이 아니다. 새 조건은 L0의 stop-gradient 4 fit뿐이다. full-gradient 4조건과 E 고정·CP0/LS·Raw는 이미 검산한 이전 결과를 재사용하며 새로 학습했다고 세지 않는다. seed92601/92602는 동일 초기값의 minibatch 순서만 바꾸므로 독립 초기화나 독립 표본이 아니다.","",
        "새 학습은 4×600=2,400 update, 20 CP이며 비교용 이전 full-gradient의 2,400 update·20 CP 로그를 별도로 표시한다. 네 실제 probe는 optimizer update 0회다. 같은 seed의 600×4 배치·AdamW.001/WD0/clip1·고정 전체 FIT 관측수 가중 macro MSE·CP0/100/200/400/600을 유지했다. 전체 TUNE MSE 최소와 이른 tie로 CP를 고르고 4개 성공/실패 잠금과 registry를 봉인한 뒤 DEV를 추론했다. 같은 update budget이라도 선정 CP는 다를 수 있는 학습 procedure ablation이며 모든 선정 예측을 CP600끼리의 비교로 해석하지 않는다. FIT 미래 Y는 손실 target으로만 쓰고 배포 forward는 과거만 받는다.","","```text","dataset seed   CP_full CP_stop TUNE_full    TUNE_stop    DEV_full     DEV_stop"]
    for data,(_,_,dev) in DATASETS.items():
        for seed in SEEDS:
            t=choose(paired,dataset=data,seed=seed,split="tune"); d=choose(paired,dataset=data,seed=seed,split=dev)
            fcp=choose(tables["SELECTION"],dataset=data,seed=seed,method="FULL_GRADIENT")["selected_update"]; scp=choose(tables["SELECTION"],dataset=data,seed=seed,method="STOP_GRADIENT")["selected_update"]
            lines.append(f"{('Robin' if data=='robin' else 'Beijing'):<7} {seed} {fcp:>7} {scp:>7} {t['full_MSE']:.9f} {t['stop_MSE']:.9f} {d['full_MSE']:.9f} {d['stop_MSE']:.9f}")
    lines += ["```","","표는 all-observed channel-macro 정규화 MSE다. complete-C/물리 단위 점수·전체 채널 원수치는 CSV에 따로 보존했다.","","![초기값·참고 조건과 실제 비교](figures/01_matched_accuracy.png)","[SVG](figures/01_matched_accuracy.svg)","",
        "**gradient와 최적화의 범위.** 초기 canonical32의 Z/F/lag 특징/예측은 full·stop·이전 CP0와 byte 비교했다. seed별 첫 FIT minibatch의 clipping 전 D gradient 동일성과 `E_stop≈E_full−E_via_F`를 등록 공차 atol1e−5/rtol2e−6으로 검사했고 실제 벡터를 보존했다. 독립 감사는 저장된 F-input VJP 이후의 chain-rule를 NumPy로 검산하며 실제 F/VJP/backward/optimizer를 재실행한 검사는 아니다. 구체적인 scope는 감사 JSON 그대로 읽는다. 초기 D gradient가 같아도 global clipping과 이후 E trajectory는 D 계수 변화도 바꾸므로 F gradient의 단독 인과효과나 순수한 E 방향만 분리한 대조가 아니다. scalar norm에서 재구성한 clip 계수와 CP 계수 변화만 기록했으며 실제 per-step AdamW parameter update norm을 측정한 것은 아니다.","","![checkpoint와 clipping](figures/02_checkpoints_and_clipping.png)","[SVG](figures/02_checkpoints_and_clipping.svg)","","![네 gradient probe](figures/03_gradient_probes.png)","[SVG](figures/03_gradient_probes.svg)","",
        "**비용은 새 실제 배포 cohort다.** Raw+full 두 seed+stop 두 seed의 5경로를 첫 TUNE B1/B8 origins에서 warmup2·5완전순환으로 측정했다. 총20조건·100표본이며 이전9경로 시간이나 빠른 시행을 새 분모로 재사용하지 않았다. physical causal fill/FIT 정규화→GPU E/실제 frozen F/lag D→물리 단위 CPU 반환·sync를 포함하고 model/state setup·hash·학습·warmup·저장·guard/memory snapshot은 타이머 밖이다. F detach는 학습 backward 개입이므로 배포 inference가 저절로 빨라진다고 해석하지 않는다. 실제 두 통계는 아래와 같고 B1/B8 전체 표본·timestamp는 CSV에 있다.","","```text","dataset route                      B8_ms/window ratio_medians median_paired"]
    for data in DATASETS:
        for route in ROUTES:
            row=choose(tables["WHOLE_COST"],dataset=data,route=route,batch=8)
            lines.append(f"{('Robin' if data=='robin' else 'Beijing'):<7} {route:<28} {row['median_seconds_per_window']*1000:.6f} {row['ratio_of_RAW_medians']:.6f} {row['median_paired_RAW_ratio']:.6f}")
    lines += ["```","","![새 동일 cohort 전체 비용](figures/04_whole_pipeline_cost.png)","[SVG](figures/04_whole_pipeline_cost.svg)","",
        "ratio-of-medians와 반복별 paired ratio 중앙값을 분리했다. 같은 첫 origins 반복100개는 독립 데이터100개가 아니다. 모든 선택 E/D와 F가 resident인 공유 RSS/VRAM이며 방법별 독점 메모리나 장비 독점은 주장하지 않는다. 새 학습과 별도 비용의 자원은 RESOURCES에 기록했고 이전16fit 전체 자원을 재사용4조건에 임의 배분하지 않았다. 이전16fit과 새4fit process의 모집단·고정 overhead가 다르므로 두 wall 시간을 직접 나눠 학습 speedup이라고 주장하지 않는다.","",
        "**남는 판단.** stop-gradient가 악화되면 현재 global clipping·AdamW·E/D 상호작용을 포함한 최적화에서 F 경로가 도움이 됐다는 조건부 근거다. 비슷하거나 나아지면 이 구성에서는 해당 backward 없이도 적응 이득이 유지될 가능성이라는 범위다. 어느 경우도 F 미래 예측값을 없애도 된다는 결론은 아니다. 이미 노출된 두 자료와 L0만의 적응적 탐색이며 독립 일반화·모든 lag·충분한 수렴·다른 표현의 한계를 확정하지 않는다. 후속 학습 연장은 이번에 수행하지 않았다. 실행/검산 성공을 Raw 개선이나 연구 해결로 바꾸어 말하지 않는다.","","원수치·실패 상태·재개 근거:",""]
    lines += [f"- [{name}.csv](tables/{name}.csv): {len(rows)}행" for name,rows in tables.items()]
    lines += [""]+[f"- [{name}]({rel(item['path'],output)}): `{item['sha256']}`" for name,item in bindings.items()]
    lines += [""]+[f"- [{r['dataset']}/{r['method']}/s{r['seed']} 원 곡선]({rel(r['updates_log']['path'],output)})" for r in tables["SELECTION"]]
    lines += ["","생성기는 JSON·JSONL만 읽고 NPZ/model·학습·F·타이머를 실행하지 않았다. 기존 결과와 실패 기록을 덮지 않으며 이 보고서 생성이 Git push·논문/PPT 완료를 뜻하지 않는다.",""]
    return "\n".join(lines)


def generate(paths,destination):
    output=resolve(destination); require(output.parent==(HERE/"reports").resolve() and not output.exists(),"FRESH_LOCAL_REPORT_REQUIRED")
    values,bindings=load_verified(paths); tables=collect_tables(values); source=record(__file__); bindings["report_source"]=source
    output.mkdir(parents=True,exist_ok=False); (output/"tables").mkdir()
    for name,rows in tables.items(): write_csv(output/"tables"/(name+".csv"),rows,output)
    images=figures(tables,output); (output/"REPORT_KO.md").write_text(make_text(values,tables,bindings,output),encoding="utf-8",newline="\n")
    require(record(__file__)==source,"REPORT_SOURCE_PRESERVED")
    for item in bindings.values(): bound(item)
    manifest={"status":"COMPLETE","source":source,"inputs":bindings,"table_row_counts":{name:len(rows) for name,rows in tables.items()},
        "files":[{"path":p.relative_to(output).as_posix(),"sha256":sha(p),"bytes":p.stat().st_size} for p in sorted(output.rglob("*")) if p.is_file()],"PNG_count":4,"SVG_count":4,
        "scientific_NPZ_or_model_read":False,"new_scientific_execution":False,"protected_TEST_accessed":False,"new_updates":2400,"reused_full_gradient_updates":2400,
        "probe_optimizer_updates":0,"gradient_intervention_scope":"F-input-gradient intervention only; not F-value/TSFM necessity; limited optimization evidence"}
    with (output/"MANIFEST.json").open("x",encoding="utf-8") as handle: json.dump(manifest,handle,ensure_ascii=False,indent=2,allow_nan=False); handle.write("\n")
    print(json.dumps({"status":"COMPLETE","manifest":str(output/"MANIFEST.json"),"tables":len(tables),"images":len(images)}),flush=True); return manifest


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in (*INPUTS,"output"): parser.add_argument("--"+key.replace("_","-"),required=True)
    args=vars(parser.parse_args()); destination=args.pop("output"); generate(args,destination)
