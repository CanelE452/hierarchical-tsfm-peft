"""JSON/JSONL-only report of two new Robin RECON-initialized fits and reused controls."""
from pathlib import Path
import argparse
import json

from make_linear_e_adaptation_report_v2 import require,resolve,sha,record,read,bound,same_opaque,choose,rel,write_csv


HERE=Path(__file__).resolve().parent
SEEDS=(92601,92602)
FIXED_SHA={"train":"f0fc3f95f1e1a61eff4ffc99b7bb7d62affd34afa064f6e29113bb0c37f52ec8","audit":"b1d28415ade18db8ddc0c987a14e879268413efdd5d1c62ad93e27b24becd52e"}
AUDITOR_SOURCE={"path":str(HERE/"audit_recon_init_sensitivity_v2.py"),"sha256":"55066f6dbc6ce8c6a175fe4ce5c8952fe8eabdde15492fd097b3a8d98c1ecdfd"}
AUDIT_COUNTS={"recon_initialization_fits_checked":2,"checkpoint_states_checked":10,"checkpoint_TUNE_predictions_checked":10,"selected_FIT_DEV_predictions_checked":4,"logged_updates_checked":1200,"CP0_probe_cases_checked":2,"CP0_parity_components_checked":8,"reused_FORECAST_initialization_conditions_checked":2,"training_score_fields_checked":396,"new_space_diagnostic_cells_computed":4,"new_space_LS_SVD_crosschecks_computed":4,"selection_locks_checked":2,"protected_source_and_input_file_hashes_checked":55}
CONTROLLER_SHA="a00300046d2236c46be7d6d532cead122cd178f066a62fd4bbbea76566d2534b"
PARENT_SHA="3f0cbd9176f2e7882778452e537564aab7445e1b88c1534fd48b1bce7ef639a4"
UTILITY={"path":str(HERE/"make_linear_e_adaptation_report_v2.py"),"sha256":"32e2460ea4724144067f264ab1ceb0332cd04c5fce2296838024c119c9572466"}
PRESERVED_AUDIT_FAILURE={"path":str(HERE/"audits/RECON_INIT_SENSITIVITY_AUDIT_v1.json"),"sha256":"9d5cfdcfd58645669f0a7f1ff56dbceb6e33f99a87784246b79523917ed6693a"}
SCALES=("normalized","physical_per_channel_diagnostic")
TABLES=("SCORES","PAIRED","CHECKPOINTS","UPDATES","CONDITIONS","PROBES","SPACE_CELLS","SPACE_METHODS","SVD_CERTIFICATES","RESOURCE")


def load_verified(paths):
    require(set(paths)==set(FIXED_SHA),"TWO_EXACT_STAGE_INPUTS")
    require(all(isinstance(v,str) and len(v)==64 for v in FIXED_SHA.values()) and isinstance(AUDITOR_SOURCE,dict) and bool(AUDIT_COUNTS),"ACTUAL_TRAIN_AUDIT_AND_NEW_SPACE_NOT_YET_REGISTERED")
    bound(UTILITY); values={k:read(p) for k,p in paths.items()}; bindings={k:record(p) for k,p in paths.items()}
    for key in paths: require(bindings[key]["sha256"]==FIXED_SHA[key] and values[key]["status"]==("PASS" if key=="audit" else "COMPLETE") and values[key]["protected_TEST_accessed"] is False,"EXACT_COMPLETE_PASS_NO_TEST:"+key)
    train,audit=values["train"],values["audit"]; bound(audit["diagnostic"],bindings["train"]); bound(audit["audit_source"],AUDITOR_SOURCE)
    require(audit["training_verification_status"]=="PASS" and audit["new_space_diagnostic_status"]=="COMPLETE" and not audit["partial_paths"] and not audit["new_output_space_diagnostic_failures"],"TRAINING_VERIFICATION_AND_FOUR_NEW_DIAGNOSTICS_COMPLETE")
    require(all(audit["totals"].get(k)==v for k,v in AUDIT_COUNTS.items()),"FULL_REGISTERED_AUDIT_COUNTS")
    require(train["controller_source"]["sha256"]==CONTROLLER_SHA and train["full_gradient_parent"]["sha256"]==PARENT_SHA,"EXACT_EXECUTED_CONTROLLER_AND_REUSED_PARENT")
    for key in ("full_gradient_parent","full_gradient_parent_audit","learned_output_space_diagnostic","learned_output_space_audit","causal_lag_diagnostic","causal_lag_audit","selection_registry","plan"):
        bound(train[key]); bound(audit[key],train[key]); bindings[key]=train[key]
    for key,source in (("parent","full_gradient_parent"),("science","causal_lag_diagnostic"),("space","learned_output_space_diagnostic")):
        values[key]=read(train[source]["path"]); require(values[key]["status"]=="COMPLETE","COMPLETE_REUSED_METADATA")
    require(train["F_before"]==train["F_after"]==values["parent"]["F_before"]==values["parent"]["F_after"],"SAME_FROZEN_F")
    require(train["recon_initialization_training_fits"]==train["resolved_selection_locks"]==2 and train["training_updates_completed"]==1200 and train["checkpoint_states_saved"]==10
        and train["forecast_initialization_conditions_reused"]==2 and train["probe_optimizer_updates"]==train["new_forecast_initialization_fits"]==0 and train["all_2_locks_before_DEV"] is True and not train["failed_paths"],"TWO_NEW_FITS_AND_TWO_REUSED_CONTROLS")
    for key in ("new_backbone_training","new_D_initialization_fit","independent_holdout","strict_K_only","whole_deployment_cost_measured"): require(train[key] is False,"INITIALIZATION_ONLY_SCOPE:"+key)
    registry=read(train["selection_registry"]["path"])
    require(registry["status"]=="ALL_TWO_RESOLVED_BEFORE_DEV" and registry["DEV_opened_before_seal"] is False and {(x["dataset"],x["seed"]) for x in registry["conditions"]}=={("robin",s) for s in SEEDS} and len(registry["conditions"])==2,"TWO_RESOLVED_LOCKS_BEFORE_DEV")
    for item in registry["conditions"]:
        row=train["datasets"]["robin"]["conditions"][str(item["seed"])]; lock=read(bound(item["selection_lock"],row["selection_lock"]))
        require(lock["status"]==row["status"]==item["status"]=="COMPLETE" and lock["DEV_opened_before_lock"] is False and lock["selected_update"]==row["selected_update"],"EXACT_NEW_SELECTED_CP_LOCK"); same_opaque(lock["selected_state"],row["selected_state"])
    bound(train["controller_source"]); bound(PRESERVED_AUDIT_FAILURE)
    require(read(PRESERVED_AUDIT_FAILURE["path"])["status"]=="FAILED_PRESERVED","ORIGINAL_METADATA_AUDIT_FAILURE_PRESERVED")
    bindings.update(controller_source=train["controller_source"],audit_source=AUDITOR_SOURCE,report_utility=UTILITY,preserved_failed_audit=PRESERVED_AUDIT_FAILURE)
    return values,bindings


def selected(row):
    cps=row["checkpoints"]; require([p["update"] for p in cps]==[0,100,200,400,600],"FIVE_TUNE_CHECKPOINTS")
    cp=min(cps,key=lambda p:(p["tune"]["mse"],p["update"]))
    require(row["selected_update"]==cp["update"] and row["selected_tune"]==cp["tune"],"TUNE_ONLY_EARLIEST_MINIMUM"); same_opaque(row["selected_state"],cp["state"])
    return cp


def collect_tables(values):
    train,parent,audit=values["train"],values["parent"],values["audit"]; tables={k:[] for k in TABLES}
    require(set(train["datasets"])=={"robin"} and set(train["datasets"]["robin"]["conditions"])=={str(s) for s in SEEDS},"EXACT_TWO_NEW_ROBIN_CONDITIONS")
    def scores(base,entry):
        for mask,key in (("all_observed","scores_all_observed"),("complete_C","scores_complete_mask")):
            for scale in SCALES: tables["SCORES"].append({**base,"mask":mask,"scale":scale,**entry[key][scale],"artifact":entry.get("artifact")})
    for seed in SEEDS:
        new=train["datasets"]["robin"]["conditions"][str(seed)]; old=parent["datasets"]["robin"]["conditions"]["robin_L7_E_LEARN_s"+str(seed)]
        require(new["order_sha256"]==old["order_sha256"],"SAME_SEED_ORDER_FOR_BOTH_INITIALIZATIONS")
        for method,row in (("RECON_INIT_TRAINED",new),("FORECAST_INIT_TRAINED",old)):
            cp=selected(row); base={"seed":seed,"method":method,"new_execution":method=="RECON_INIT_TRAINED"}
            for split,entry in (("fit",row["FIT"]),("tune",cp),("val",row["DEV"])): scores({**base,"split":split},entry)
            tables["CHECKPOINTS"] += [{**base,**p,"selected":p["update"]==cp["update"]} for p in row["checkpoints"]]
            path=bound(row["updates_log"]); require(path.suffix==".jsonl","JSONL_LOG_ONLY")
            logs=[json.loads(line,parse_constant=lambda v: (_ for _ in ()).throw(ValueError("NONFINITE_LOG:"+v))) for line in path.read_text(encoding="utf-8").splitlines()]
            require([p["update"] for p in logs]==list(range(1,601)) and sum(p["clipped"] for p in logs)/600==row["global_clipping_fraction"],"EXACT_600_UPDATE_AND_CLIP_LOG")
            tables["UPDATES"] += [{**base,**p} for p in logs]
            tables["CONDITIONS"].append({**base,"selected_update":cp["update"],"selected_state":row["selected_state"],"selection_lock":row["selection_lock"],"order_sha256":row["order_sha256"],"global_clipping_fraction":row["global_clipping_fraction"],
                "CP0_parameter_diagnostics":row["checkpoints"][0]["parameter_diagnostics"],"selected_parameter_diagnostics":cp["parameter_diagnostics"],"selected_TUNE_latent_variance":cp["TUNE_latent_variance"],"parameter_change":audit["datasets"]["robin"]["conditions"][str(seed)].get("parameter_change") if base["new_execution"] else None,"updates_log":row["updates_log"]})
        require(new["probe"]["status"]=="PASS" and new["probe"]["optimizer_updates"]==0 and all(x["pass"] for x in new["probe"]["checks"].values()),"TWO_ZERO_UPDATE_CP0_PARITY_PROBES")
        tables["PROBES"].append({"seed":seed,**new["probe"]})
    recon=values["science"]["datasets"]["robin"]["spaces"]["PCA"]["conditions"]["L7_RECON"]
    full=parent["datasets"]["robin"]["conditions"]["robin_L7_E_LEARN_s92601"]
    for split in ("fit","tune","val"):
        forecast_cp0=full["CP0_TUNE"] if split=="tune" else parent["CP0_LS_and_RAW_baselines"]["robin"]["L7"][split]["CP0_LS"]
        for method,entry in (("RECON_CP0",recon["splits"][split]),("FORECAST_CP0",forecast_cp0),("RAW_BOLT",recon["splits"][split]["RAW_BOLT"])):
            scores({"seed":0,"method":method,"split":split,"new_execution":False},entry)
    for seed in SEEDS:
        for split in ("fit","tune","val"):
            for mask in ("all_observed","complete_C"):
                for scale in SCALES:
                    where={"seed":seed,"split":split,"mask":mask,"scale":scale}; new=choose(tables["SCORES"],**where,method="RECON_INIT_TRAINED"); old=choose(tables["SCORES"],**where,method="FORECAST_INIT_TRAINED")
                    tables["PAIRED"].append({**where,"RECON_MSE":new["mse"],"FORECAST_MSE":old["mse"],"MSE_RECON_minus_FORECAST":new["mse"]-old["mse"],"RECON_MAE":new["mae"],"FORECAST_MAE":old["mae"],"MAE_RECON_minus_FORECAST":new["mae"]-old["mae"]})
        for initialization in ("RECON","FORECAST"):
            cells=audit["new_output_space_diagnostics"][str(seed)] if initialization=="RECON" else values["space"]["datasets"]["robin"]["conditions"]["FULL_L7_s"+str(seed)]["splits"]
            for split in ("tune","val"):
                cell=cells[split]; base={"initialization":initialization,"seed":seed,"split":split,"computed_during_current_audit":initialization=="RECON"}
                require(cell["status"]=="COMPLETE" and cell["independent_reduced_LS_vs_SVD_verified"] is True and cell["diagnostic_future_labels_used_for_new_fit_or_selection"] is False and cell["all_real_latent_or_all_FP32_lower_bound_claimed"] is False,"REGISTERED_NEW_OR_REUSED_NUMERIC_SPACE_SCOPE")
                tables["SPACE_CELLS"].append({**base,"minimum_normalized_MSE":cell["minimum_normalized_MSE"],"actual_complete_MSE":cell["methods"]["ACTUAL_F"]["scores_complete_mask"]["normalized"]["mse"],"EY_complete_MSE":cell["methods"]["EY"]["scores_complete_mask"]["normalized"]["mse"],"FLAT_complete_MSE":cell["methods"]["FLAT"]["scores_complete_mask"]["normalized"]["mse"],"RAW_complete_MSE":cell["RAW_BOLT_scores_complete_mask"]["normalized"]["mse"],"current_C_persistence_scores_complete_mask":cell["current_C_persistence_scores_complete_mask"],
                    **{k:cell[k] for k in ("complete_target_positions","target_positions","joint_unknown_future_coordinates","joint_output_coordinates_before_mask","joint_metadata","minimum_solution_scores_complete_C","EY_encoding_backend","EY_incomplete_C_horizons_filled_with_current_C","artifact","input_source","selected_forecast_source")}})
                tables["SPACE_METHODS"] += [{**base,"method":method,**row} for method,row in cell["methods"].items()]
                tables["SVD_CERTIFICATES"].append({**base,"certificates":cell["independent_LS_certificates"]})
    tables["RESOURCE"]=[{"stage":"new two-fit training" if key=="train" else "saved-training audit plus NEW output-space calculation","new_deployment_timing":False,**values[key].get("resource_terminal" if key=="train" else "resource",{})} for key in ("train","audit")]
    require(len(tables["SCORES"])==84 and len(tables["UPDATES"])==2400 and len(tables["SPACE_CELLS"])==8,"FULL_TWO_SEED_REPORT_COVERAGE")
    return tables


def figures(tables,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder=output/"figures"; folder.mkdir(); files=[]
    colors={"RECON_INIT_TRAINED":"#0072B2","FORECAST_INIT_TRAINED":"#D55E00"}
    def save(fig,name,bottom=.16):
        fig.tight_layout(rect=(0,bottom,1,.90))
        for ext in ("png","svg"):
            path=folder/(name+"."+ext); fig.savefig(path,dpi=150); files.append(path)
        plt.close(fig)
    def mse(method,seed,split): return choose(tables["SCORES"],method=method,seed=seed,split=split,mask="all_observed",scale="normalized")["mse"]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,split in zip(axes,("tune","val")):
        for method,color in colors.items(): ax.plot(range(2),[mse(method,s,split) for s in SEEDS],"o-",color=color,label=method.replace("_INIT_TRAINED"," init, trained"))
        for method,color,style in (("RECON_CP0","#009E73",":"),("FORECAST_CP0","gray","-."),("RAW_BOLT","black","--")):
            ax.axhline(mse(method,0,split),color=color,linestyle=style,label=method.replace("_"," "))
        ax.set_xticks(range(2),[str(s) for s in SEEDS]); ax.set_title("Robin / "+("TUNE selection" if split=="tune" else "exposed DEV")); ax.set_ylabel("All-observed normalized MSE"); ax.grid(alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels(); fig.legend(handles,labels,loc="lower center",ncol=3)
    fig.suptitle("Initialization sensitivity: two fresh RECON fits versus two reused FORECAST fits; CP0 and Raw references"); save(fig,"01_initialization_errors")
    fig,ax=plt.subplots(figsize=(11,4.8))
    for method,color in colors.items():
        for seed,style in zip(SEEDS,("-","--")):
            rows=[r for r in tables["CHECKPOINTS"] if r["method"]==method and r["seed"]==seed]
            ax.plot([r["update"] for r in rows],[r["tune"]["mse"] for r in rows],marker="o",linestyle=style,color=color,label=method.replace("_INIT_TRAINED"," init")+" / "+str(seed))
            chosen=next(r for r in rows if r["selected"]); ax.scatter(chosen["update"],chosen["tune"]["mse"],marker="*",s=170,color=color,zorder=4)
    ax.set_xlabel("Optimizer update; stars are selected TUNE checkpoints"); ax.set_ylabel("All-observed TUNE normalized MSE"); ax.grid(alpha=.2)
    fig.legend(*ax.get_legend_handles_labels(),loc="lower center",ncol=2); fig.suptitle("Same 600-update budget and seed-specific minibatch order; 1,200 new updates and 1,200 reused"); save(fig,"02_TUNE_checkpoints")
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,split in zip(axes,("tune","val")):
        rows=[choose(tables["SPACE_CELLS"],initialization=init,seed=s,split=split) for init in ("FORECAST","RECON") for s in SEEDS]
        for key,color,marker,label in (("actual_complete_MSE","#0072B2","o","Saved actual F"),("EY_complete_MSE","#D55E00","x","E(Y): future diagnostic"),("minimum_normalized_MSE","#009E73","*","Retained numeric minimum")):
            ax.plot(range(4),[r[key] for r in rows],color=color,marker=marker,label=label)
        ax.axhline(rows[0]["RAW_complete_MSE"],color="black",linestyle="--",label="Raw same complete-C")
        ax.set_xticks(range(4),[r["initialization"]+"/"+str(r["seed"])[-2:] for r in rows],rotation=15); ax.set_title("Robin / "+split); ax.set_ylabel("Complete-C normalized MSE"); ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=2); fig.suptitle("Selected D32 spaces: RECON cells newly calculated during audit; no universal FP32 or past-predictability bound"); save(fig,"03_selected_output_spaces")
    return files


def make_text(values,tables,bindings,output):
    def score(method,seed,split): return choose(tables["SCORES"],method=method,seed=seed,split=split,mask="all_observed",scale="normalized")["mse"]
    initial=score("RECON_CP0",0,"val"); raw=score("RAW_BOLT",0,"val")
    tune_improved=sum(score("RECON_INIT_TRAINED",s,"tune")<score("RECON_CP0",0,"tune") for s in SEEDS)
    dev_worse_initial=sum(score("RECON_INIT_TRAINED",s,"val")>initial for s in SEEDS)
    paired_improved=sum(score("RECON_INIT_TRAINED",s,"val")<score("FORECAST_INIT_TRAINED",s,"val") for s in SEEDS)
    raw_worse=sum(score("RECON_INIT_TRAINED",s,"val")>raw for s in SEEDS)
    dev_spaces=[r for r in tables["SPACE_CELLS"] if r["initialization"]=="RECON" and r["split"]=="val"]
    min_above_raw=sum(r["minimum_normalized_MSE"]>r["RAW_complete_MSE"] for r in dev_spaces)
    lines=["# Robin L7: RECON D 초기화 민감도", "",
        f"[확인] 새 RECON 초기화 학습은 TUNE에서 초기 RECON CP0보다 {tune_improved}/2 조건이 좋아졌지만, 기노출 DEV에서는 {dev_worse_initial}/2 조건이 원 RECON CP0보다 나빴다. 원 FORECAST 초기화의 학습 결과보다는 DEV {paired_improved}/2 조건이 개선됐고 Raw보다 나쁜 조건은 {raw_worse}/2이다. 초기화가 결과에 영향을 주었으나 현재 600 update 예측 목적 E/D 학습으로 원 고정 RECON·Raw 격차가 해결되지는 않았다.","",
        f"All-observed 정규화 MSE: RECON CP0 TUNE {score('RECON_CP0',0,'tune'):.10f}, DEV {initial:.10f}; Raw TUNE {score('RAW_BOLT',0,'tune'):.10f}, DEV {raw:.10f}.","",
        f"[확인] 새 선정 D의 DEV retained numeric minimum도 같은 complete-C Raw보다 {min_above_raw}/2 조건에서 높았다. "+"; ".join(f"seed {r['seed']}: actual {r['actual_complete_MSE']:.10f}, E(Y) {r['EY_complete_MSE']:.10f}, minimum {r['minimum_normalized_MSE']:.10f}, Raw {r['RAW_complete_MSE']:.10f}" for r in dev_spaces)+". 현재 선정 D와 알려진 과거 offset에 조건부인 진단이므로 모든 표현의 불가능성을 뜻하지 않는다. 다음 고정 D 대조에서 E 갱신과 D 공간 이동의 효과를 분리할 필요가 있다.",""]
    for seed in SEEDS:
        new=choose(tables["CONDITIONS"],method="RECON_INIT_TRAINED",seed=seed); old=choose(tables["CONDITIONS"],method="FORECAST_INIT_TRAINED",seed=seed)
        lines.append(f"- seed {seed}: 새 RECON-init CP{new['selected_update']}, TUNE {score('RECON_INIT_TRAINED',seed,'tune'):.10f}, DEV {score('RECON_INIT_TRAINED',seed,'val'):.10f}; 재사용 FORECAST-init CP{old['selected_update']}, TUNE {score('FORECAST_INIT_TRAINED',seed,'tune'):.10f}, DEV {score('FORECAST_INIT_TRAINED',seed,'val'):.10f}.")
    lines += ["", "![초기화별 실제 점수](figures/01_initialization_errors.png)","",
        "**비교 조건.** Robin C17/K4, L512/H48, FIT256/TUNE64/기노출 VAL30, 같은 scaler·origin·mask·현재 C anchor·lag [0,1,2,4,8,16,32]를 유지했다. 초기 PCA E와 center는 원 CP0와 byte-exact이고 기존 FIT RECON D32만 복사했다. 이후 두 초기화 모두 같은 actual-C 예측 손실로 E/D를 갱신한다. 재구성 목적 학습과 예측 목적 학습 전체의 대조가 아니다. E/D는 AdamW lr .001/WD0/clip1/B4/600 update, 전체 FIT 채널 관측수 고정 가중 macro MSE, seed별 동일 minibatch 순서를 쓴다. F는 동결하고 실제 F 입력 gradient를 유지한다. seed는 순서의 차이이며 두 독립 자료나 일반화 반복이 아니다.","",
        "새 2fit의 1,200 update·10CP와 기존 FORECAST-init 2fit의 1,200 update·10CP 재사용을 구분한다. 새 CP0 parity probe 2개는 optimizer update 0이다. CP0/100/200/400/600의 전체 TUNE MSE와 이른 tie로 선정한 뒤 두 성공/실패 lock 및 registry를 봉인하고 DEV를 추론했다. 새 결과로 기존 CP·학습 예산·선정 규칙을 바꾸지 않았다. K streams + K 과거 buffer + 현재 C anchor이므로 strict K-only는 아니다.","",
        "![TUNE checkpoint 곡선](figures/02_TUNE_checkpoints.png)","",
        "**선정된 공간에 남은 문제.** 아래는 같은 complete-C mask의 actual F, E(Y), FLAT 및 retained numeric minimum이다. all-observed 주점수와 마스크를 섞지 않는다. 새 RECON-init의 4cell은 학습 감사 중 새 계산이며, 기존 FORECAST-init 4cell은 이전 실제 진단 재사용이다. 불변 SVD와 exact-zero-column-reduced LS 두 수학 경로를 교차했지만 새 진단을 별도 외부 감사자가 다시 재현했다고 부르지 않는다.",""]
    for row in tables["SPACE_CELLS"]:
        lines.append(f"- {row['initialization']} / {row['seed']} / {row['split']}: actual {row['actual_complete_MSE']:.10f}, E(Y) {row['EY_complete_MSE']:.10f}, minimum {row['minimum_normalized_MSE']:.10f}, Raw {row['RAW_complete_MSE']:.10f}; complete {row['complete_target_positions']}/{row['target_positions']}.")
    lines += ["", "![선정된 출력 공간 진단](figures/03_selected_output_spaces.png)","",
        "minimum은 선정 D32와 known-past offset, 관측 출력 행, rcond 1e-15로 남긴 수치 공간에서 정규화 MSE만 최소화한 값이다. 미래 미지수는 HK이며 H×7K 자유 변수가 아니다. MAE·물리 단위 오차는 그 MSE 해의 값이고 각각의 최소가 아니다. 모든 FP32/실수 예측의 보편 하한이나 과거 정보로 그 미래를 예측할 수 있다는 증명이 아니다. 원 real/FP32 false-bound flags와 버린 성분은 SPACE_METHODS에 그대로 남겼다. E(Y)는 실제 미래를 쓰는 NumPy FP32 진단이며 배포 불가하다. 불완전 C horizon은 현재 C로 채우며 L7의 뒤 horizon에도 영향을 줄 수 있다. FLAT은 잠재 미래 persistence여서 알려진 과거 lag 때문에 현재 C persistence와 같다는 보장이 없다.","",
        "**검산 범위와 다음 판단.** 기존 FIT에서 적합된 초기 D를 다시 fit하지 않았고 새 학습은 E/D 예측 손실 갱신이다. 초기화 민감도는 확인하되 낮은 과거 D minimum을 새 D 가능성으로 재사용하지 않는다. 학습 감사는 저장 E/lag/D 산술·score·CP·로그·order·lock과 원 파일 보존을 확인하며 실제 F forward/backward·optimizer·매 update 학습 손실을 독립 재실행한 것은 아니다. 감사 중 새 계산된 공간과 실제 latent forecast gap을 분리해야 하며, 현재 결과만으로 재구성 표현의 원리적 실패·학습 수렴·Bolt/대조학습/동적 그래프 필요성을 확정하지 않는다. 기노출 DEV에서 원인을 본 뒤 선택한 적응적 탐색이므로 독립 일반화 검증이 아니다. 기존 joint-Y/RF3 등의 예측 목적 E/D 학습도 이미 실행했다.","",
        "새 deployment 비용은 계획대로 미측정이다. 같은 구조라도 다른 cohort의 기존 timing을 새 계수 비용으로 차용하지 않는다. RESOURCE는 새 두 학습 실행과 저장 결과 감사·공간 계산의 CPU/wall이며, 기존 16fit process와 직접 나누어 초기화별 학습 speedup을 주장하지 않는다. 기존 결과·실패·선정·보고서는 보존했고 이번 보고 생성은 JSON/JSONL metadata만 읽으며 NPZ/model/F/훈련/TEST/Git을 실행하지 않는다.","",
        "**보존된 감사 실패.** 감사 v1은 같은 path·SHA의 공간 진단 record 한쪽에 bytes 항목이 추가된 metadata dict를 통째 비교하여 멈췄다. 원 실패와 SOURCE를 보존하고 v2에서 path·SHA 결속으로 고쳤다. 학습·예측·배열·수식·공차를 바꾼 복구가 아니며 v1의 실패를 PASS로 덮지 않았다. 현재 학습 검산 PASS와 새 공간 COMPLETE는 별도 v2 기록이다.",""]
    lines += [f"- [{name}.csv](tables/{name}.csv): {len(rows)}행" for name,rows in tables.items()]
    lines += [""]+[f"- [{name}]({rel(item['path'],output)}): `{item['sha256']}`" for name,item in bindings.items()]
    return "\n".join(lines)+"\n"


def generate(paths,destination):
    output=resolve(destination); require(output.parent==(HERE/"reports").resolve() and not output.exists(),"FRESH_LOCAL_REPORT_REQUIRED")
    values,bindings=load_verified(paths); tables=collect_tables(values); source=record(__file__); bindings["report_source"]=source
    output.mkdir(parents=True,exist_ok=False); (output/"tables").mkdir()
    for name,rows in tables.items(): write_csv(output/"tables"/(name+".csv"),rows,output)
    images=figures(tables,output); (output/"REPORT_KO.md").write_text(make_text(values,tables,bindings,output),encoding="utf-8",newline="\n")
    require(record(__file__)==source,"REPORT_SOURCE_UNCHANGED")
    for item in bindings.values(): bound(item)
    manifest={"status":"COMPLETE","source":source,"inputs":bindings,"files":[{"path":p.relative_to(output).as_posix(),"sha256":sha(p),"bytes":p.stat().st_size} for p in sorted(output.rglob("*")) if p.is_file()],"table_row_counts":{k:len(v) for k,v in tables.items()},"PNG_count":3,"SVG_count":3,"scientific_NPZ_or_model_read":False,"new_training_updates_during_generation":0,"new_whole_deployment_cost_measured":False,"protected_TEST_accessed":False,"new_space_calculation_role":"Computed during training audit, not third external diagnostic replay"}
    with (output/"MANIFEST.json").open("x",encoding="utf-8") as handle: json.dump(manifest,handle,ensure_ascii=False,indent=2,allow_nan=False); handle.write("\n")
    print(json.dumps({"status":"COMPLETE","manifest":str(output/"MANIFEST.json"),"images":len(images)}),flush=True); return manifest


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("train","audit","output"): parser.add_argument("--"+key,required=True)
    args=vars(parser.parse_args()); output=args.pop("output"); generate(args,output)
