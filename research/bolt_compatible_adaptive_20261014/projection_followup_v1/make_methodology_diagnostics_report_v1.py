"""Metadata-only fixed/learned Q and quantile-summary report; no scientific arrays or F."""
from pathlib import Path
import argparse
import json

from make_linear_e_adaptation_report_v2 import require,resolve,sha,record,read,bound,same_opaque,choose,rel,write_csv

HERE=Path(__file__).resolve().parent
STATES=("RECON_CP0","E_ONLY_s92601","E_ONLY_s92602")
SEEDS=(92601,92602)
CONTROLS=("IDENTITY","PERMUTATION","SIGN_FLIP","DENSE_SO_s92711","DENSE_SO_s92712")
SUMMARIES=("MEDIAN","NINE_MEAN_PROXY","TAIL_CLAMPED_PROXY")
TABLES=("FIXED_CONTROLS","QUANTILE_SUMMARIES","Q_CHECKPOINTS","Q_SELECTED","GRADIENTS","RESOURCES")
FIXED_SHA={"controls":"67661b1468384aed58d57c8f3c84e3a04d8f6e806cbfb4dbdb92d0f6c34cf42b",
    "controls_audit":"156e7513b200215f574b511828428c3e11c5b8a79ab2f79c85448a0eb87b4977",
    "quantile":"70d09544210602d3566d2d1f2405d06087499c496d6691dbade6dea64762d780",
    "quantile_audit":"02e6a1df4a6297c71faf97345394e483f43f2d15a59ee2ee16f4776b4385783f",
    "learning":"e1067bce2ef20381477c5345d5dd5566528a4340b69ba0b19bbcdd717d141fcc","learning_audit":"eb0f264c1cf3985481c20e7abe0a49c8a043c0aa036a1f620849fc276bb35bfe"}
AUDIT_SOURCES={"controls_audit":{"path":str(HERE/"audit_orthogonal_basis_controls_v1.py"),"sha256":"4d16612c65b9b1ee3e33a95639b87992b73ce6aa3991dea4c922f58ae17d9f43"},
    "quantile_audit":{"path":str(HERE/"audit_quantile_point_summary_v1.py"),"sha256":"eaaa08be6ca2c9c9db27231de08eeb0157d33546f59a7718f3f6b9d906b94a5e"},
    "learning_audit":{"path":str(HERE/"audit_orthogonal_q_learning_v1.py"),"sha256":"a3730599a881e2963b5bf7205d0792104a21d3bad9ae4daa9e8c3ba466a04d60"}}
COUNTS={"controls_audit":{"fixed_control_conditions_checked":15,"FIT_TUNE_prediction_evaluations_checked":30,"DEV_prediction_evaluations_checked":15,"control_locks_checked":15,"produced_NPZ_artifacts_checked":105,"score_fields_checked":4320},
    "quantile_audit":{"full_quantile_contexts_checked":12,"summary_prediction_cells_checked":36,"selection_locks_checked":4,"recorded_full_F_calls_checked":44,"produced_NPZ_artifacts_checked":60,"score_fields_checked":864},
    "learning_audit":{"Q_training_conditions_checked":6,"training_update_logs_checked":3600,"checkpoint_states_checked":30,"checkpoint_TUNE_prediction_evaluations_checked":30,"selected_FIT_prediction_evaluations_checked":6,"selected_TUNE_entries_reused_checked":6,"selected_DEV_prediction_evaluations_checked":6,"first_batch_gradient_closures_checked":6,"selection_locks_checked":6,"produced_NPZ_artifacts_checked":120,"actual_F_calls_registered":3714,"score_fields_checked":3024,"commutator_summaries_checked":84,"frozen_E_D_center_sources_checked":3,"same_parent_orders_checked":6,"protected_prior_file_hashes_checked":2056}}
UTILITY={"path":str(HERE/"make_linear_e_adaptation_report_v2.py"),"sha256":"32e2460ea4724144067f264ab1ceb0332cd04c5fce2296838024c119c9572466"}


def load_verified(paths):
    require(set(paths)==set(FIXED_SHA) and all(isinstance(v,str) and len(v)==64 for v in FIXED_SHA.values())
        and all(isinstance(v,dict) for v in AUDIT_SOURCES.values()) and all(isinstance(v,dict) and v for v in COUNTS.values()),"THREE_STUDIES_AND_INDEPENDENT_AUDITS_NOT_YET_REGISTERED")
    bound(UTILITY);values={k:read(p) for k,p in paths.items()};bindings={k:record(p) for k,p in paths.items()}
    for key in paths:require(bindings[key]["sha256"]==FIXED_SHA[key] and values[key]["status"]==("PASS" if key.endswith("audit") else "COMPLETE") and values[key]["protected_TEST_accessed"] is False,"EXACT_COMPLETE_PASS_NO_TEST:"+key)
    for study in ("controls","quantile","learning"):
        audit=values[study+"_audit"];bound(audit["diagnostic"],bindings[study]);bound(audit["audit_source"],AUDIT_SOURCES[study+"_audit"])
        require(not audit["partial_paths"] and not values[study]["failed_paths"] and all(audit["totals"].get(k)==v for k,v in COUNTS[study+"_audit"].items()),"FULL_INDEPENDENT_COVERAGE:"+study)
        bindings[study+"_auditor_source"]=audit["audit_source"]
    controls,quantile,learning=(values[k] for k in ("controls","quantile","learning"))
    for value in (quantile,learning):
        for key in ("parent_report","parent_report_manifest","parent_report_audit","latent_calibration_diagnostic","matched_l7_native_diagnostic","matched_l7_native_audit"):bound(value[key],controls[key])
    bound(learning["fixed_controls_diagnostic"],bindings["controls"]);bound(learning["fixed_controls_audit"],bindings["controls_audit"])
    require(controls["F_before"]==controls["F_after"]==quantile["F_before"]==quantile["F_after"]==learning["F_before"]==learning["F_after"],"SAME_FROZEN_BACKBONE_DIFFERENT_INTERVENTIONS")
    require(learning["Q_training_fits"]==6 and learning["training_updates_completed"]==3600 and learning["checkpoint_states_saved"]==30 and learning["DEV_evaluations_completed"]==6 and learning["all_6_locks_before_DEV"] is True,"SIX_FITS_THIRTY_CP_AND_SIX_LOCKED_DEV")
    for study in ("controls","quantile","learning"):
        value=values[study];require(value["whole_deployment_cost_measured"] is value["independent_holdout"] is value["strict_K_only"] is value["new_backbone_training"] is False,"EXPLORATORY_SAME_K_NO_NEW_DEPLOYMENT_COST")
        for key in ("controller_source","module_source","plan"):
            bound(value[key]);bindings[study+"_"+key]=value[key]
        registry_key="control_registry" if study=="controls" else "selection_registry"
        bound(value[registry_key]);bindings[study+"_registry"]=value[registry_key]
    source=controls["matched_l7_native_audit"];bound(source);values["space_context"]=read(source["path"]);bindings["existing_space_audit"]=source
    require(values["space_context"]["status"]=="PASS","EXISTING_CONDITIONAL_SPACE_DIAGNOSTIC_PASS")
    return values,bindings


def selected(row):
    cps=row["checkpoints"];require([p["update"] for p in cps]==[0,100,200,400,600],"ALL_FIVE_PREDECLARED_CHECKPOINTS")
    winner=min(cps,key=lambda p:(p["tune"]["mse"],p["update"]))
    require(winner["update"]==row["selected_update"] and winner["tune"]==row["selected_tune"],"OWN_TUNE_EARLIEST_EXACT_TIE")
    same_opaque(row["selected_state"],winner["state"]);return winner


def collect_tables(values):
    controls,quantile,learning=(values[k] for k in ("controls","quantile","learning"));tables={k:[] for k in TABLES}
    for state in STATES:
        for control in CONTROLS:
            condition=controls["states"][state]["controls"][control]
            for split in ("fit","tune","val"):
                entry=condition["splits"][split]
                tables["FIXED_CONTROLS"].append({"state":state,"control":control,"split":split,
                    **{k:entry[k] for k in ("scores","baseline_Bolt_scores","RAW_BOLT_scores","checks","latent_commutator","C_commutator","transport_future_gap","transport_C_gap","complete_target_positions","artifact","independent_artifact","actual_F_calls","input_source","cached_Bolt_source")},"Q_metadata":condition["Q_metadata"],"Q_artifact":condition["Q_artifact"],"Q_before":condition["Q_before"],"Q_after":condition["Q_after"],"control_lock":condition["control_lock"],"Q_selected":False})
    for route in ("RAW_C",)+STATES:
        condition=quantile["routes"][route]
        for split in ("fit","tune","val"):
            context=condition["splits"][split];median=context["summaries"]["MEDIAN"]
            for summary,entry in context["summaries"].items():
                tables["QUANTILE_SUMMARIES"].append({"route":route,"summary":summary,"split":split,"selected_on_TUNE":summary==condition["selected_summary"],
                    "nonselected_DEV_diagnostic_only":split=="val" and summary!=condition["selected_summary"],**{k:entry[k] for k in ("scores_all_observed","scores_complete_mask","checks","artifact")},
                    "fresh_median_scores_all_observed":median["scores_all_observed"],"fresh_median_scores_complete_mask":median["scores_complete_mask"],
                    "MSE_minus_own_fresh_median":entry["scores_all_observed"]["normalized"]["mse"]-median["scores_all_observed"]["normalized"]["mse"],
                    "full_quantile_artifact":context["artifact"],"full64_quantile_diagnostics":context["full64_quantile_diagnostics"],"evaluated48_quantile_diagnostics":context["evaluated48_quantile_diagnostics"],
                    "median_cache_parity":context["checks"],"selection_lock":condition["selection_lock"],"input_source":context["input_source"]})
    for state in STATES:
        for seed in SEEDS:
            row=learning["states"][state]["learning"][str(seed)];cp=selected(row);base={"state":state,"seed":seed}
            require(row["training_updates_completed"]==row["post_optimizer_frozen_flag_checks"]==600 and row["frozen_state_checkpoint_byte_checks"]==5 and row["final_frozen_status"]["trainable_parameter_names"]==["theta"],"SIX_HUNDRED_THETA_ONLY_UPDATES_AND_FROZEN_CP_CHECKS")
            for checkpoint in row["checkpoints"]:
                require(checkpoint["E_D_center_exact_initial_bytes"] is True,"SAVED_CP_FROZEN_E_D_CENTER")
                tables["Q_CHECKPOINTS"].append({"state_label":state,"seed":seed,"selected":checkpoint["update"]==cp["update"],**checkpoint,"updates_log":row["updates_log"]})
            for split,entry in (("fit",row["FIT"]),("tune",row["TUNE"]),("val",row["DEV"])):
                context=values["space_context"]["new_output_space_diagnostics"][state]["92601"][split] if split!="fit" else None
                tables["Q_SELECTED"].append({**base,"split":split,"selected_update":cp["update"],"selected_state":row["selected_state"],"selected_basis_diagnostics":row["selected_basis_diagnostics"],"selection_lock":row["selection_lock"],
                    **{k:entry[k] for k in ("scores_all_observed","scores_complete_mask","baseline_Bolt_scores","RAW_BOLT_scores","latent_commutator","C_commutator","artifact","input_source","cached_Bolt_source")},
                    "CP0_TUNE_MSE":row["checkpoints"][0]["tune"]["mse"],"existing_space_context":None if context is None else {"minimum_normalized_MSE":context["minimum_normalized_MSE"],"RAW_complete_MSE":context["RAW_BOLT_scores_complete_mask"]["normalized"]["mse"],"complete_target_positions":context["complete_target_positions"],"source":learning["matched_l7_native_audit"],"new_geometry_computed":False,"universal_all_FP32_bound":False}})
            probe=row["first_batch_gradient_probe"];require(probe["status"]=="PASS" and probe["optimizer_updates_in_probe"]==0 and probe["gradient_closure_pass"] is True,"SIX_SAVED_FIRST_BATCH_GRADIENT_DECOMPOSITIONS")
            tables["GRADIENTS"].append({**base,**probe,"global_clipping_fraction":row["global_clipping_fraction"],"order_sha256":row["order_sha256"],"updates_log":row["updates_log"],"post_optimizer_frozen_flag_checks":row["post_optimizer_frozen_flag_checks"],"frozen_bytes_checked_at_CP_and_final_only":row["frozen_bytes_checked_at_CP_and_final_only"],"independent_per_step_backward_optimizer_or_frozen_bytes_replayed":row["independent_per_step_backward_optimizer_or_frozen_bytes_replayed"],"final_frozen_status":row["final_frozen_status"]})
    tables["RESOURCES"]=[{"stage":key,**values[key]["resource_terminal"],"new_whole_deployment_cost_measured":False} for key in FIXED_SHA]
    require([len(tables[k]) for k in TABLES]==[45,36,30,18,6,6],"FULL_THREE_STUDY_SIX_TABLE_COVERAGE")
    return tables


def figures(tables,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    folder=output/"figures";folder.mkdir();files=[];colors=("#0072B2","#D55E00","#009E73")
    def save(fig,name,bottom=.16):
        fig.tight_layout(rect=(0,bottom,1,.91))
        for ext in ("png","svg"):
            path=folder/(name+"."+ext);fig.savefig(path,dpi=160,bbox_inches="tight");files.append(path)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.6))
    for ax,split in zip(axes,("tune","val")):
        for index,state in enumerate(STATES):
            rows=[choose(tables["FIXED_CONTROLS"],state=state,control=q,split=split) for q in CONTROLS]
            ax.plot(range(5),[r["scores"]["all_observed"]["normalized"]["mse"] for r in rows],"o-",color=colors[index],label="E"+str(index))
        raw=rows[0]["RAW_BOLT_scores"]["all_observed"]["normalized"]["mse"];ax.axhline(raw,color="black",linestyle=":",label="Accepted Raw median")
        ax.set_xticks(range(5),["I","Perm","Sign","Dense 92711","Dense 92712"],rotation=15);ax.set_yscale("log");ax.set_ylabel("All-observed normalized MSE (log)");ax.set_title("TUNE" if split=="tune" else "Exposed DEV");ax.grid(alpha=.18)
    fig.suptitle("Fixed orthogonal coordinates: all predeclared controls, no Q selection");handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc="lower center",ncol=4);save(fig,"fig01_fixed_Q_controls")
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.6));routes=("RAW_C",)+STATES;x=np.arange(4)
    for ax,split in zip(axes,("tune","val")):
        for j,summary in enumerate(SUMMARIES[1:]):
            rows=[choose(tables["QUANTILE_SUMMARIES"],route=r,summary=summary,split=split) for r in routes]
            ax.bar(x+(j-.5)*.34,[r["MSE_minus_own_fresh_median"] for r in rows],width=.34,color=colors[j],label=summary)
        ax.axhline(0,color="black",linewidth=.8);ax.set_xticks(x,["Raw-C","E0","E1","E2"]);ax.set_ylabel("MSE minus own fresh median");ax.set_title("TUNE selection" if split=="tune" else "Exposed DEV diagnostic");ax.grid(axis="y",alpha=.18)
    fig.suptitle("Same three-summary menu for Raw and compressed routes; proxies are not identified means");handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc="lower center",ncol=2);save(fig,"fig02_quantile_summary_deltas")
    fig,axes=plt.subplots(1,3,figsize=(14.5,5.3))
    for index,(ax,state) in enumerate(zip(axes,STATES)):
        for seed,style in zip(SEEDS,("-","--")):
            rows=[r for r in tables["Q_CHECKPOINTS"] if r["state_label"]==state and r["seed"]==seed]
            ax.plot([r["update"] for r in rows],[r["tune"]["mse"] for r in rows],style,color=colors[SEEDS.index(seed)],marker="o",label="seed "+str(seed))
            selected_row=next(r for r in rows if r["selected"]);ax.scatter([selected_row["update"]],[selected_row["tune"]["mse"]],s=110,facecolors="none",edgecolors="black",zorder=5)
        ax.set_title("E"+str(index));ax.set_xlabel("Theta-only updates");ax.set_ylabel("Own TUNE normalized MSE");ax.set_xticks([0,100,200,400,600]);ax.grid(alpha=.18)
    fig.suptitle("Q learning: six parameters; circled checkpoint selected on TUNE, DEV excluded");handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc="lower center",ncol=2);save(fig,"fig03_Q_checkpoint_curves")
    fig,axes=plt.subplots(1,2,figsize=(14,6.5));x=np.arange(3)
    for ax,split in zip(axes,("tune","val")):
        initial=[choose(tables["FIXED_CONTROLS"],state=s,control="IDENTITY",split=split) for s in STATES]
        ax.plot(x,[r["scores"]["complete_C"]["normalized"]["mse"] for r in initial],"ko-",label="Original median / I")
        for seed,style in zip(SEEDS,("--","-.",)):
            rows=[choose(tables["Q_SELECTED"],state=s,seed=seed,split=split) for s in STATES]
            ax.plot(x,[r["scores_complete_mask"]["normalized"]["mse"] for r in rows],style,marker="o",label="Selected Q s"+str(seed))
        chosen=[next(r for r in tables["QUANTILE_SUMMARIES"] if r["route"]==s and r["split"]==split and r["selected_on_TUNE"]) for s in STATES]
        ax.plot(x,[r["scores_complete_mask"]["normalized"]["mse"] for r in chosen],"s-",color="#CC79A7",label="Own-TUNE selected point summary")
        rows=[choose(tables["Q_SELECTED"],state=s,seed=92601,split=split) for s in STATES]
        ax.plot(x,[r["existing_space_context"]["minimum_normalized_MSE"] for r in rows],"x:",color="#777777",label="Existing retained-space diagnostic")
        raw=next(r for r in tables["QUANTILE_SUMMARIES"] if r["route"]=="RAW_C" and r["split"]==split and r["selected_on_TUNE"])
        ax.axhline(raw["scores_complete_mask"]["normalized"]["mse"],color="#D55E00",linestyle=":",label="Own-TUNE selected Raw summary")
        ax.set_xticks(x,["E0","E1","E2"]);ax.set_title("TUNE" if split=="tune" else "Exposed DEV");ax.set_ylabel("Same-mask complete-C normalized MSE");ax.grid(alpha=.18)
    fig.suptitle("Different interventions on fixed information and D; no new deployment cost measurement")
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc="lower center",ncol=2,fontsize=9)
    fig.text(.5,.205,"Numeric minima are reused conditional diagnostics, not universal FP32 bounds or deployed predictors.",ha="center",fontsize=9)
    save(fig,"fig04_stage_comparison",bottom=.27)
    return files


def make_text(values,tables,bindings,output):
    def mse(entry,key="scores_all_observed"):return entry[key]["normalized"]["mse"]
    qdev=[r for r in tables["Q_SELECTED"] if r["split"]=="val"];qtune=[r for r in tables["Q_SELECTED"] if r["split"]=="tune"]
    qbetter=sum(mse(r)<r["baseline_Bolt_scores"]["all_observed"]["normalized"]["mse"] for r in qdev)
    qraw=sum(mse(r)<r["RAW_BOLT_scores"]["all_observed"]["normalized"]["mse"] for r in qdev)
    tunebetter=sum(mse(r)<r["CP0_TUNE_MSE"] for r in qtune);last=sum(r["selected_update"]==600 for r in qtune);zero=sum(r["selected_update"]==0 for r in qtune)
    nontrivial=[r for r in tables["FIXED_CONTROLS"] if r["control"] in CONTROLS[2:] and r["split"]=="val"]
    fixedworse=sum(r["scores"]["all_observed"]["normalized"]["mse"]>r["baseline_Bolt_scores"]["all_observed"]["normalized"]["mse"] for r in nontrivial)
    selected_quantile=[r for r in tables["QUANTILE_SUMMARIES"] if r["selected_on_TUNE"] and r["split"]=="val"]
    raw=choose(selected_quantile,route="RAW_C");compressed=[r for r in selected_quantile if r["route"]!="RAW_C"]
    quantileworse=sum(r["MSE_minus_own_fresh_median"]>0 for r in compressed)
    oldraw=choose(tables["FIXED_CONTROLS"],state=STATES[0],control="IDENTITY",split="val")["RAW_BOLT_scores"]["all_observed"]["normalized"]["mse"]
    freshraw=raw["fresh_median_scores_all_observed"]["normalized"]["mse"];rawgain=1-mse(raw)/freshraw
    lines=["# 같은 정보에서의 좌표·점요약 진단","",
        f"고정 sign/dense Q의 DEV {fixedworse}/{len(nontrivial)}개는 원 median 경로보다 나빴다. 좌표를 바꾸면 같은 정보·같은 D에서도 F 출력이 달라지지만, 이것만으로 성능 개선을 뜻하지 않는다.","",
        f"θ 6개만 학습한 Q는 TUNE {tunebetter}/6개, DEV {qbetter}/6개에서 자기 원 baseline보다 낮은 MSE였다. 원 Raw median보다 좋은 DEV 조건은 {qraw}/6개다. 선정 CP0 {zero}/6개는 비갱신 기준이며, 마지막 CP600 선정 {last}/6개이므로 수렴이나 충분한 학습을 단정하지 않는다.","",
        f"같은 세 점요약 메뉴에서 선정한 Raw는 DEV fresh median {freshraw:.12g} → {mse(raw):.12g} (상대 감소 {rawgain*100:.6f}%)였다. 압축 세 경로 중 {quantileworse}/3개는 자기 fresh median보다 DEV가 악화됐다. Raw의 같은 메뉴를 제외하고 압축 이득을 주장하지 않는다.","",
        "## 목적·수식·학습 범위","",
        "`z=(X-center)E`, `u_Q=F(zQ)Qᵀ`, `Yhat=old_lag_decoder(current_C,z,u_Q;D)`를 사용했다. E/D/center/F와 K4, L512/H48, FIT256/TUNE64/기노출 DEV30, scaler·origin·관측 mask는 유지했다. old past와 lag별 origin reference를 회전된 값으로 바꾸지 않았다. 고정 I는 불필요한 Q 연산을 생략하고, 학습 θ=0은 matrix_exp(skew(θ))를 통과하여 gradient를 유지했다.","",
        "Q 학습은 θ6만 AdamW lr.001/weight_decay0, B4, clip1, 600 update×6조건이었다. seed는 같은 초기 θ에서 minibatch order만 바꾼다. 전 FIT 고정 채널 관측수로 정규화한 actual-C macro MSE를 학습하고 CP0/100/200/400/600의 자기 TUNE MSE로 선정했다. 여섯 lock 후 DEV를 평가했다. gradient probe는 첫 학습 forward를 재사용한 0추가-update 검사이며 F입력 경로와 역회전 경로의 저장 gradient 합을 확인한다. 이것은 F 필요성이나 전체 backward/optimizer의 독립 재현이 아니다.","",
        "분위 요약은 같은 frozen F의 실제 9×64 출력을 새로 저장하고 첫48만 사용했다. MEDIAN은 .5, NINE_MEAN_PROXY는 균등9가중합, TAIL_CLAMPED_PROXY는 [.15,.1,.1,.1,.1,.1,.1,.1,.15]다. 두 proxy는 식별된 조건부 평균이 아니다. crossing을 정렬로 없애지 않았으며, decoded C 분위구간·CRPS를 주장하지 않는다. 네 경로 모두 자기 TUNE에서 같은 메뉴를 고르고 네 lock 후 DEV를 평가했다. 비선정 DEV 8행은 진단용이다.","",
        "## 원수치와 남은 문제","",
        f"기존 Raw cache DEV MSE {oldraw:.12g}와 이번 새 full-quantile median {freshraw:.12g}는 작은 수치 차이가 있어 같은 값으로 대체하지 않았다. 고정/학습 Q는 자신의 accepted median baseline, 점요약은 자신의 fresh median과 비교했다.","","```text","상태·seed       Q TUNE / 원 TUNE          Q DEV / 원 DEV             CP"]
    for row in qdev:
        tune=choose(qtune,state=row["state"],seed=row["seed"]);label="E"+str(STATES.index(row["state"]))+"/"+str(row["seed"])
        lines.append(f"{label:<14} {mse(tune):.9f} / {tune['CP0_TUNE_MSE']:.9f}    {mse(row):.9f} / {row['baseline_Bolt_scores']['all_observed']['normalized']['mse']:.9f}   {row['selected_update']}")
    lines += ["```","","```text","경로       선정 요약               fresh median DEV / 선정 DEV"]
    for row in selected_quantile:lines.append(f"{('Raw' if row['route']=='RAW_C' else 'E'+str(STATES.index(row['route']))):<10} {row['summary']:<24} {row['fresh_median_scores_all_observed']['normalized']['mse']:.12g} / {mse(row):.12g}")
    lines += ["```","","기존 matched-L7 감사의 retained numeric minimum을 아래에 재사용했다. complete-C 같은 마스크이며 이번 Q/점요약에서 새 geometry를 계산한 결과가 아니다. 이 값은 알려진 미래를 사용한 해당 수치 공간의 진단으로, 모든 FP32 출력의 보편 하한·과거 예측 가능성·방법 성공을 뜻하지 않는다.","","```text","상태     TUNE minimum / Raw       DEV minimum / Raw"]
    for state in STATES:
        t=choose(qtune,state=state,seed=92601)["existing_space_context"];d=choose(qdev,state=state,seed=92601)["existing_space_context"]
        lines.append(f"E{STATES.index(state)}       {t['minimum_normalized_MSE']:.9f} / {t['RAW_complete_MSE']:.9f}    {d['minimum_normalized_MSE']:.9f} / {d['RAW_complete_MSE']:.9f}")
    lines += ["```","","현재 근거는 같은 정보에서의 좌표 민감도, 제한된 θ 최적화 효과, 점요약의 목표 차이를 분리한다. 작은 K와 복원 공간의 한계 및 실제 latent 미래 예측의 차이는 남아 있다. 새 표현·큰 K·더 긴 최적화·다른 TSFM에서의 결과로 확대하지 않는다. 이전 E/D 공동 예측 학습도 이미 시행됐으므로 재구성 목적만이 유일 원인이었다고 쓰지 않는다.","",
        "## 검증·비용·제한","",
        "실제 학습·추론과 저장값 검산은 각각 구분했다. fixed I/permutation null, analytic transport, NumPy 역회전/old lag D, 관측 채널 macro MSE/MAE·원단위 per-C와 complete-C를 검산했다. F 자체의 독립 재추론은 감사 범위가 아니다. gradient 존재는 현재 절차의 작동 증거이며 TSFM 필요성 증명은 아니다.","",
        "세 진단의 새 whole-deployment 비용은 미측정이다. RESOURCES는 학습/과학 추론/CPU 감사 process 자원이며 기존 q 비용을 새 Q/점요약의 latency로 차용하지 않는다. 배포 입력은 과거뿐이지만 K streams + K past buffer + current C anchor를 사용하므로 strict K-only가 아니다. FIT 미래는 Q 학습 응답으로만 썼다. DEV는 이미 노출된 적응적 탐색이며 독립 일반화·보호 TEST 결과가 아니다. 교수님께는 작은 E0 이득과 Raw 격차, 다른 상태의 악화, 비용 미측정까지 함께 제시해야 한다.","","## 그림과 재현 자료",""]
    for name,caption in (("fig01_fixed_Q_controls","고정 Q 전체 대조"),("fig02_quantile_summary_deltas","동일 Raw 메뉴의 점요약 변화"),("fig03_Q_checkpoint_curves","Q 학습의 TUNE checkpoint"),("fig04_stage_comparison","같은 complete-C 마스크의 단계 대조와 진단 한계")):
        lines += [f"![{caption}](figures/{name}.png)",""]
    for key in TABLES:lines.append(f"- [{key}.csv](tables/{key}.csv): {len(tables[key])}행")
    lines += [""]
    for key,item in bindings.items():lines.append(f"- [{key}]({rel(item['path'],output)}) · SHA256 `{item['sha256']}`")
    return "\n".join(lines)+"\n"


def generate(paths,output_path):
    output=resolve(output_path);require(output.parent==(HERE/"reports").resolve() and not output.exists(),"FRESH_LOCAL_DIAGNOSTIC_REPORT_DIRECTORY")
    values,bindings=load_verified(paths);tables=collect_tables(values);source=record(__file__);bindings["report_source"]=source;bindings["report_utility"]=UTILITY
    output.mkdir(parents=True,exist_ok=False);(output/"tables").mkdir()
    for name,rows in tables.items():write_csv(output/"tables"/(name+".csv"),rows,output)
    images=figures(tables,output);(output/"REPORT_KO.md").write_text(make_text(values,tables,bindings,output),encoding="utf-8")
    require(record(__file__)==source,"REPORT_SOURCE_UNCHANGED")
    for item in bindings.values():bound(item)
    files=[{"path":p.relative_to(output).as_posix(),"sha256":sha(p),"bytes":p.stat().st_size} for p in sorted(output.rglob("*")) if p.is_file()]
    manifest={"status":"COMPLETE","source":source,"inputs":bindings,"files":files,"table_row_counts":{k:len(v) for k,v in tables.items()},"PNG_count":4,"SVG_count":4,
        "scientific_NPZ_or_model_read":False,"F_training_GPU_or_new_geometry_executed":False,"whole_deployment_cost_measured":False,"existing_numeric_space_context_reused":True,"protected_TEST_accessed":False,"independent_holdout":False,"strict_K_only":False}
    with (output/"MANIFEST.json").open("x",encoding="utf-8") as handle:json.dump(manifest,handle,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({"status":"COMPLETE","tables":len(tables),"files":len(files),"figures":len(images),"manifest":str(output/"MANIFEST.json")}),flush=True);return manifest


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in (*FIXED_SHA,"output"):parser.add_argument("--"+key.replace("_","-"),required=True)
    args=vars(parser.parse_args());output=args.pop("output");generate(args,output)
