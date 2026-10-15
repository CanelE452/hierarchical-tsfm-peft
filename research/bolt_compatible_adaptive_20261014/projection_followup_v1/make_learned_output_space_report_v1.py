"""Metadata-only report of selected learned E/D spaces and saved-future diagnostics."""
from pathlib import Path
import argparse
import json

from make_linear_e_adaptation_report_v2 import require,resolve,sha,record,read,bound,choose,rel,write_csv


HERE=Path(__file__).resolve().parent
DATASETS={"robin":(17,4,"val"),"beijing_aotizhongxin":(11,2,"eval")}
CONDITIONS=tuple(f"FULL_{lag}_s{seed}" for lag in ("L0","L7") for seed in (92601,92602))+tuple(f"STOP_L0_s{seed}" for seed in (92601,92602))
FIXED_SHA={"diagnostic":"cf042cc764ff36cbb877211777721783efadbaab6ba8741268eaccba26cbb34b","audit":'1b0dbacb711becce70ac877825cd53b84069918ff62ac79dc9b9b5d8ac410738'}
PRODUCER_SOURCE={'path': 'E:\\CODING\\proj\\hierarchical-tsfm-peft\\research\\bolt_compatible_adaptive_20261014\\projection_followup_v1\\run_learned_output_space_diagnostic_v1.py', 'sha256': '08f5806c80b825c9510eaaad9eaf85c9fb0e9633668e3d33be0317162e50f795'}
AUDITOR_SOURCE={'path': 'E:\\CODING\\proj\\hierarchical-tsfm-peft\\research\\bolt_compatible_adaptive_20261014\\projection_followup_v1\\audit_learned_output_space_v1.py', 'sha256': '0e33409bd54c776c7f4d3df08fbe7969470e5e8cef21922162ddb6e55a62a1a5'}
EXPECTED_COUNTS={'selected_states_checked': 12, 'diagnostic_cells_checked': 24, 'saved_actual_F_predictions_checked': 24, 'EY_predictions_checked': 24, 'latent_persistence_predictions_checked': 24, 'current_C_persistence_predictions_checked': 24, 'joint_maps_and_offsets_checked': 24, 'complete_C_numeric_minima_checked': 24, 'retained_trace_inequalities_checked': 72, 'original_real_trace_inequalities_evaluated': 72, 'FP32_with_measured_allowance_inequalities_evaluated': 72, 'RAW_bound_inequalities_checked': 0, 'score_fields_checked': 5472, 'protected_source_and_input_file_hashes_checked': 78}
UTILITY={"path":str(HERE/"make_linear_e_adaptation_report_v2.py"),"sha256":"32e2460ea4724144067f264ab1ceb0332cd04c5fce2296838024c119c9572466"}
SCALES=("normalized","physical_per_channel_diagnostic")


def load_verified(paths):
    require(set(paths)==set(FIXED_SHA),"TWO_EXACT_DIAGNOSTIC_INPUTS")
    require(all(isinstance(v,str) and len(v)==64 for v in FIXED_SHA.values()) and isinstance(PRODUCER_SOURCE,dict)
        and isinstance(AUDITOR_SOURCE,dict) and bool(EXPECTED_COUNTS),"ACTUAL_DIAGNOSTIC_AUDIT_SOURCE_NOT_YET_REGISTERED")
    bound(UTILITY); values={key:read(path) for key,path in paths.items()}; bindings={key:record(path) for key,path in paths.items()}
    for key in paths: require(bindings[key]["sha256"]==FIXED_SHA[key] and values[key]["protected_TEST_accessed"] is False,"EXACT_ACTUAL_INPUT_NO_TEST")
    diagnostic,audit=values["diagnostic"],values["audit"]
    require((diagnostic["status"],audit["status"]) in (("COMPLETE","PASS"),("PARTIAL_COMPLETED","PARTIAL_AUDITED_NO_FULL_PASS")),"MATCHED_ACTUAL_DIAGNOSTIC_AND_AUDIT_STATUS")
    bound(audit["diagnostic"],bindings["diagnostic"]); bound(audit["audit_source"],AUDITOR_SOURCE)
    require(all(audit["totals"].get(k)==v for k,v in EXPECTED_COUNTS.items()),"EXACT_INDEPENDENT_AUDIT_COVERAGE")
    sources=[r for r in diagnostic["source_records"] if Path(r["path"]).name=="run_learned_output_space_diagnostic_v1.py"]
    require(len(sources)==1,"EXACT_EXECUTED_PRODUCER_SOURCE"); bound(sources[0],PRODUCER_SOURCE)
    for key in ("new_F_calls","new_model_objects_loaded","new_training_updates"): require(diagnostic[key]==0,"NO_NEW_MODEL_F_OR_OPTIMIZER")
    for key in ("new_E_D_fit","CP_or_branch_reselection","oracle_used_for_fit_or_selection","independent_holdout","strict_K_only","new_F_state_hash_computed","whole_deployment_cost_measured"):
        require(diagnostic[key] is False,"DIAGNOSTIC_ONLY_SCOPE:"+key)
    if diagnostic["status"]=="COMPLETE":
        require(diagnostic["completed_diagnostic_cells"]==diagnostic["saved_ACTUAL_F_predictions_diagnosed"]==diagnostic["EY_diagnostics"]==diagnostic["latent_persistence_diagnostics"]==diagnostic["independent_LS_SVD_crosschecks"]==24
            and diagnostic["selected_states_diagnosed"]==12 and not diagnostic["failed_paths"],"FULL_TWELVE_STATE_TWENTY_FOUR_CELL_COVERAGE")
    for key in ("stop_gradient_diagnostic","stop_gradient_audit","full_gradient_parent","full_gradient_parent_audit","causal_lag_diagnostic","causal_lag_audit","selection_registry","full_gradient_selection_registry","plan"):
        bound(diagnostic[key]); bound(audit[key],diagnostic[key]); bindings[key]=diagnostic[key]
    bindings.update(producer_source=PRODUCER_SOURCE,audit_source=AUDITOR_SOURCE,report_utility=UTILITY)
    return values,bindings


def ref(value): return {key:value[key] for key in ("path","sha256","bytes") if key in value}


def collect_tables(values):
    d=values["diagnostic"]; tables={key:[] for key in ("CELLS","SCORES","TRACES","SVD_GROUPS","MATCHED_L0_FULL_STOP","FAILURES","RESOURCE")}
    require(set(d["datasets"])==set(DATASETS) and all(set(d["datasets"][data]["conditions"])==set(CONDITIONS) for data in DATASETS),"EXACT_TWELVE_SELECTED_STATES")
    def scores(base,value):
        if value.get("normalized") is None: return
        for scale in SCALES: tables["SCORES"].append({**base,"scale":scale,**value[scale]})
    for data,(_,_,dev) in DATASETS.items():
        for condition,owner in d["datasets"][data]["conditions"].items():
            for split in ("tune",dev):
                base={"dataset":data,"condition":condition,"split":split,"origin":owner["origin"],"seed":owner["seed"],"selected_update":owner["selected_update"]}
                cell=owner["splits"].get(split,{"status":"NOT_PRODUCED","failure":{"message":"No split artifact"}})
                if cell["status"]!="COMPLETE":
                    tables["FAILURES"].append({**base,"status":cell["status"],"failure":cell.get("failure"),"original_state":ref(owner["selected_state"])}); continue
                require(cell["all_real_latent_or_all_FP32_lower_bound_claimed"] is False and cell["minimum_only_normalized_MSE"] is True
                    and cell["RAW_bound_applied"] is False and cell["diagnostic_future_labels_used_for_new_fit_or_selection"] is False,"REGISTERED_NUMERIC_DIAGNOSTIC_SCOPE")
                ranks=cell["joint_metadata"]["numeric_rank_per_origin"]
                tables["CELLS"].append({**base,"status":cell["status"],"complete_target_positions":cell["complete_target_positions"],"target_positions":cell["target_positions"],
                    "HK_future_unknowns":cell["joint_unknown_future_coordinates"],"HC_outputs_before_mask":cell["joint_output_coordinates_before_mask"],"lag_slope_parameters_per_C":cell["lag_slope_parameters_per_C"],
                    "minimum_normalized_MSE":cell["minimum_normalized_MSE"],"actual_complete_MSE":cell["methods"]["ACTUAL_F"]["scores_complete_mask"]["normalized"]["mse"],
                    "EY_complete_MSE":cell["methods"]["EY"]["scores_complete_mask"]["normalized"]["mse"],"FLAT_complete_MSE":cell["methods"]["FLAT"]["scores_complete_mask"]["normalized"]["mse"],
                    "RAW_complete_MSE":cell["RAW_BOLT_scores_complete_mask"]["normalized"]["mse"],"RAW_all_observed_MSE":cell["RAW_BOLT_scores_all_observed"]["normalized"]["mse"],
                    "current_C_persistence_complete_MSE":cell["current_C_persistence_scores_complete_mask"]["normalized"]["mse"],"rcond":cell["joint_metadata"]["rcond"],
                    "numeric_rank_per_origin":ranks,"numeric_rank_min":min(ranks),"numeric_rank_max":max(ranks),"minimum_space_definition":cell["joint_metadata"]["minimum_space_definition"],
                    "EY_encoding_backend":cell["EY_encoding_backend"],"EY_filled_incomplete_C_horizons":cell["EY_incomplete_C_horizons_filled_with_current_C"],"EY_fill_can_affect_later_L7":cell["EY_fill_can_affect_later_L7_complete_outputs"],
                    "selected_E_D_previously_supervised_by_FIT_future_targets":cell["selected_E_D_previously_supervised_by_FIT_future_targets"],
                    **{key:ref(value) for key,value in (("selected_state",owner["selected_state"]),("selection_lock",owner["selection_lock"]),("input_source",cell["input_source"]),("selected_forecast_source",cell["selected_forecast_source"]),("artifact",cell["artifact"]))}})
                for method,out in cell["methods"].items():
                    for phase,key in (("FP32","scores_complete_mask"),("REAL_TRACE","real_trace_scores_complete_C"),("RETAINED_TRACE","retained_output_real_trace_scores_complete_C")):
                        scores({**base,"method":method,"phase":phase,"mask":"complete_C","role":out["role"]},out[key])
                    if method=="ACTUAL_F": scores({**base,"method":method,"phase":"FP32","mask":"all_observed","role":out["role"]},out["scores_all_observed"])
                    tables["TRACES"].append({**base,"method":method,**{k:v for k,v in out.items() if isinstance(v,(str,int,float,bool)) or v is None}})
                for method,key,mask in (("RAW_BOLT","RAW_BOLT_scores_complete_mask","complete_C"),("RAW_BOLT","RAW_BOLT_scores_all_observed","all_observed"),("CURRENT_C_PERSISTENCE","current_C_persistence_scores_complete_mask","complete_C"),("MINIMUM_SOLUTION","minimum_solution_scores_complete_C","complete_C")):
                    scores({**base,"method":method,"phase":"REAL_MINIMUM_SOLUTION" if method=="MINIMUM_SOLUTION" else "FP32","mask":mask,"role":"Normalized MSE numeric-space solution; MAE/physical are values of this solution, not separate minima" if method=="MINIMUM_SOLUTION" else "same-original-mask baseline"},cell[key])
                for index,certificate in enumerate(cell["independent_LS_certificates"]):
                    rank=certificate["numeric_rank"]; singular=certificate["singularvalues"]
                    tables["SVD_GROUPS"].append({**base,"mask_group":index,"window_indices":certificate["window_indices"],"complete_horizons":certificate["complete_horizons"],"selected_output_rows":certificate["selected_output_rows"],
                        "unknown_future_coordinates_kept":certificate["unknown_future_coordinates_kept"],"numeric_rank":rank,"largest_singular":singular[0] if singular else None,
                        "last_retained_singular":singular[rank-1] if rank else None,"first_discarded_singular":singular[rank] if rank<len(singular) else None,
                        "normal_residual_max_abs":certificate["normal_residual_max_abs"],"normal_tolerance":certificate["normal_tolerance"],"exact_zero_reduced_LS":certificate.get("exact_zero_reduced_LS")})
        for seed in (92601,92602):
            for split in ("tune",dev):
                full=[r for r in tables["CELLS"] if r["dataset"]==data and r["condition"]==f"FULL_L0_s{seed}" and r["split"]==split]
                stop=[r for r in tables["CELLS"] if r["dataset"]==data and r["condition"]==f"STOP_L0_s{seed}" and r["split"]==split]
                tables["MATCHED_L0_FULL_STOP"].append({"dataset":data,"seed":seed,"split":split,"status":"COMPLETE" if full and stop else "UNAVAILABLE_PARTIAL",
                    **{key+"_stop_minus_full":stop[0][key]-full[0][key] if full and stop else None for key in ("minimum_normalized_MSE","actual_complete_MSE","EY_complete_MSE")},
                    "same_learned_E_D_claimed":False})
    require(len(tables["CELLS"])==d["completed_diagnostic_cells"],"ACTUAL_COMPLETED_CELL_COUNTER")
    tables["RESOURCE"]=[{"stage":"saved-output diagnosis; not deployment timing",**d.get("resource_terminal",{})}]
    return tables


def figures(tables,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    folder=output/"figures"; folder.mkdir(); files=[]
    labels=[c.replace("FULL_","F/").replace("STOP_","S/").replace("_s9260","/s") for c in CONDITIONS]
    def save(fig,name,bottom=.08):
        fig.tight_layout(rect=(0,bottom,1,.93))
        for suffix in ("png","svg"):
            path=folder/(name+"."+suffix); fig.savefig(path,dpi=150); files.append(path)
        plt.close(fig)
    def rows(data,split): return {r["condition"]:r for r in tables["CELLS"] if r["dataset"]==data and r["split"]==split}
    def values(found,key): return [found[c][key] if c in found else float("nan") for c in CONDITIONS]
    fig,axes=plt.subplots(2,2,figsize=(14,7))
    for i,(data,(_,_,dev)) in enumerate(DATASETS.items()):
        for j,split in enumerate(("tune",dev)):
            ax=axes[i,j]; found=rows(data,split)
            ax.plot(range(6),values(found,"actual_complete_MSE"),"o-",color="#0072B2",label="Saved actual F")
            ax.plot(range(6),values(found,"minimum_normalized_MSE"),"*-",color="#D55E00",label="Retained numeric minimum")
            if found:
                first=next(iter(found.values())); ax.axhline(first["RAW_complete_MSE"],color="black",linestyle="--",label="Raw same complete-C")
            ax.set_xticks(range(6),labels,rotation=25); ax.set_title(data+" / "+split); ax.set_ylabel("Complete-C normalized MSE"); ax.grid(alpha=.2)
    fig.legend(handles=[Line2D([0],[0],marker="o",color="#0072B2",label="Saved actual F"),Line2D([0],[0],marker="*",color="#D55E00",label="Retained numeric minimum"),Line2D([0],[0],color="black",linestyle="--",label="Raw same complete-C")],loc="lower center",ncol=3)
    fig.suptitle("Selected learned E/D spaces: same target mask; minimum does not establish past predictability or an all-FP32 bound"); save(fig,"01_actual_space_and_raw")
    fig,axes=plt.subplots(2,2,figsize=(14,7))
    for i,(data,(_,_,dev)) in enumerate(DATASETS.items()):
        for j,split in enumerate(("tune",dev)):
            ax=axes[i,j]; found=rows(data,split)
            for key,color,label in (("actual_complete_MSE","#0072B2","Actual F"),("EY_complete_MSE","#D55E00","Future E(Y): diagnostic"),("FLAT_complete_MSE","#009E73","Flat latent: past-only")):
                ax.plot(range(6),[found[c][key]-found[c]["minimum_normalized_MSE"] if c in found else float("nan") for c in CONDITIONS],"o-",color=color,label=label)
            ax.axhline(0,color="black"); ax.set_xticks(range(6),labels,rotation=25); ax.set_title(data+" / "+split); ax.set_ylabel("Complete-C FP32 MSE minus numeric minimum"); ax.grid(alpha=.2)
    fig.legend(handles=[Line2D([0],[0],color=color,label=label) for color,label in (("#0072B2","Actual F"),("#D55E00","E(Y): known-future diagnostic"),("#009E73","Flat latent: past-only"))],loc="lower center",ncol=3)
    fig.suptitle("Same D and mask for each cell; FP32/real trace false inequalities and discarded components are preserved separately"); save(fig,"02_future_diagnostic_gaps")
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for ax,(data,(_,_,dev)) in zip(axes,DATASETS.items()):
        paired=[r for r in tables["MATCHED_L0_FULL_STOP"] if r["dataset"]==data]
        for key,color,label in (("minimum_normalized_MSE_stop_minus_full","#D55E00","Space minimum difference"),("actual_complete_MSE_stop_minus_full","#0072B2","Actual F difference")):
            ax.plot(range(4),[r[key] if r[key] is not None else float("nan") for r in paired],"o-",color=color,label=label)
        ax.axhline(0,color="black"); ax.set_xticks(range(4),[str(r["seed"])+"/"+r["split"] for r in paired],rotation=25); ax.set_title(data); ax.set_ylabel("Stop minus full, complete-C normalized MSE"); ax.grid(alpha=.2)
    fig.legend(handles=[Line2D([0],[0],color=c,label=l) for c,l in (("#D55E00","Numeric space minimum"),("#0072B2","Saved actual F"))],loc="lower center",ncol=2)
    fig.suptitle("Matched L0/order seed after training; selected E/D spaces differ; no fixed-space causal effect claimed"); save(fig,"03_full_stop_space_contrasts",bottom=.12)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for ax,(data,(_,_,dev)) in zip(axes,DATASETS.items()):
        found=rows(data,dev)
        for key,color,label in (("FLAT_complete_MSE","#009E73","Flat latent"),("current_C_persistence_complete_MSE","gray","Current-C persistence")):
            ax.plot(range(6),values(found,key),"o-",color=color,label=label)
        ax.set_xticks(range(6),labels,rotation=25); ax.set_title(data+" / exposed "+dev); ax.set_ylabel("Same complete-C normalized MSE"); ax.grid(alpha=.2)
    fig.legend(handles=[Line2D([0],[0],color=c,label=l) for c,l in (("#009E73","Flat latent"),("gray","Current-C persistence"))],loc="lower center",ncol=2)
    fig.suptitle("Flat future latent can retain known L7 prefix effects; this is not a universal persistence improvement claim"); save(fig,"04_flat_and_current_C",bottom=.12)
    return files


def make_text(values,tables,bindings,output):
    d=values["diagnostic"]; done=len(tables["CELLS"]); failures=len(tables["FAILURES"])
    lines=["# 학습된 E/D의 공간과 실제 잠재 미래 예측 오차","",f"실제 진단 {done}/24, 실패·관측부족 등 미완료 {failures}/24. 원 상태는 `{d['status']}`, 독립 감사는 `{values['audit']['status']}`다. 기존 선정 E/D 12상태의 TUNE·기노출 DEV를 재사용한 원인 진단이며 새 학습·F 추론·새 선정·배포 비용 측정은 모두0이다.",""]
    for data,(_,_,dev) in DATASETS.items():
        tune=[r for r in tables["CELLS"] if r["dataset"]==data and r["split"]=="tune"]
        rows=[r for r in tables["CELLS"] if r["dataset"]==data and r["split"]==dev]
        above=sum(r["minimum_normalized_MSE"]>r["RAW_complete_MSE"] for r in rows)
        if data=="robin":
            tune_above=sum(r["minimum_normalized_MSE"]>r["RAW_complete_MSE"] for r in tune)
            lines += [f"**Robin: 동일 complete-C에서 minimum>Raw가 TUNE {tune_above}/{len(tune)} → 기노출 DEV {above}/{len(rows)}로 달라졌다.** 현재 선정 D32와 known-past offset의 수치 유지 공간이 이 DEV 구간에서는 제약으로 남는다는 조건부 진단이다. 구간별 차이를 확인했을 뿐 분포 shift가 원인이라고 확정하지 않는다.",""]
        else:
            ey_below=sum(r["EY_complete_MSE"]<r["RAW_complete_MSE"] for r in rows); actual_above=sum(r["actual_complete_MSE"]>r["RAW_complete_MSE"] for r in rows)
            lines += [f"**Beijing: 동일 complete-C DEV에서 minimum>Raw {above}/{len(rows)}, E(Y)<Raw {ey_below}/{len(rows)}, actual F>Raw {actual_above}/{len(rows)}다.** 현재 선정 공간에는 여지가 있고 미래 실제 C를 인코딩한 E(Y)도 Raw보다 낮지만 실제 F가 그 좌표에 정렬되지 않는 문제가 남는다는 단서다. 이를 F만의 단독 원인이나 과거로 그 미래 좌표를 예측할 수 있다는 증거로 확대하지 않는다.",""]
        for split in ("tune",dev):
            representative=next((r for r in tables["CELLS"] if r["dataset"]==data and r["condition"]=="FULL_L0_s92601" and r["split"]==split),None)
            if representative:
                r=representative; lines += [f"대표 FULL_L0/seed92601 {split}: complete-C normalized MSE actual F={r['actual_complete_MSE']:.9f}, E(Y)={r['EY_complete_MSE']:.9f}, minimum={r['minimum_normalized_MSE']:.9f}, Raw={r['RAW_complete_MSE']:.9f}.",""]
    lines += ["```text","dataset split condition          minimum     actual_F    E(Y)        Raw_complete Raw_allobs"]
    for row in tables["CELLS"]:
        lines.append(f"{('Robin' if row['dataset']=='robin' else 'Beijing'):<7} {row['split']:<5} {row['condition']:<19} {row['minimum_normalized_MSE']:.9f} {row['actual_complete_MSE']:.9f} {row['EY_complete_MSE']:.9f} {row['RAW_complete_MSE']:.9f} {row['RAW_all_observed_MSE']:.9f}")
    lines += ["```","","**마스크를 섞지 않는다.** 위 minimum·actual F·E(Y)·Raw_complete는 모든 C가 관측된 horizon만 쓰는 complete-C 정규화 MSE다. Raw_allobs와 실제 예측의 all-observed 점수는 채널별 원 관측 위치에서 따로 계산했다. 특히 Beijing의 complete-C Raw와 전체 관측 Raw는 같은 수치가 아니므로 한쪽을 다른 비교의 분모로 쓰지 않는다. 24조건의 Raw는 동일 자료·split의 4개 고유 baseline을 각 상태에서 반복 참조한 값이며 24개의 독립 평가가 아니다.","","![실제 예측·수치 공간·동일 마스크 Raw](figures/01_actual_space_and_raw.png)","[SVG](figures/01_actual_space_and_raw.svg)","",
        "**계산한 공간.** 갱신된 D32와 저장된 실제 latent past를 그대로 써 joint HC×HK 선형 실수 맵과 origin별 known-past offset을 다시 만들었다. lag가 7개여도 미래 자유변수는 HK이며 H×7K가 아니다. 출력의 complete-C 행만 줄이고 관측 없는 horizon의 미래 좌표는 남겼다. rcond1e−15의 retained numeric left-singular 공간 minimum이며 exact0 열만 줄인 LS 후 전체 HK 좌표 복원과 SVD 출력을 교차했다. 이전 초기 PCA D의 minimum을 재사용하지 않았다. SVD_GROUPS는 mask별 rank·singular 경계·normal certificate 요약이며 전체 vector는 원 JSON/NPZ에 있다.","",
        "minimum은 정규화 MSE 목적의 수치 유지 공간 해다. 그 해의 MAE·물리 MSE는 각각의 minimum이 아니며 모든 실수/모든 FP32 예측에 대한 보편 하한도 아니다. 실제 FP32·실수 trace·retained trace를 분리했고 원 real/FP32 부등식 거짓은 TRACES에 그대로 남겼다. retained trace에만 최소 부등식을 필수 검사한다. 낮은 minimum은 미래를 이미 아는 자유 좌표로 얻은 것이므로 과거 예측 가능성·독립 일반화·작은 K의 일반 가능성을 입증하지 않는다.","","![실제·E(Y)·잠재 persistence 진단 gap](figures/02_future_diagnostic_gaps.png)","[SVG](figures/02_future_diagnostic_gaps.svg)","",
        "**E(Y)는 배포 입력이 아니다.** 실제 미래 C를 선정 E/center로 NumPy FP32 인코딩했다. C 일부가 결측인 horizon은 벡터 전체를 current-C로 바꿔 인코딩하며, L7에서는 그 대체가 이후 관측 horizon에도 전파될 수 있다. 이 backend의 반올림은 저장 Torch 인코딩과 다를 수 있다. E(Y)가 좋거나 나쁜 것은 해당 예측 목적 D에서 자연스러운 E(Y) 좌표와 복원이 정렬됐는지의 진단이며 잠재 최적해나 예측 성공 자체가 아니다. 과거 E/D의 FIT 미래 감독은 인정하고 이번 진단에서 새 적합·선정에 미래를 쓰지 않았다는 범위만 주장한다.","",
        "L0의 full·stop 조건은 같은 초기값·minibatch-order seed·600 update budget에서 출발했지만 선정 CP와 학습 후 E/D가 다르다. 출력 공간 차이와 실제 예측 차이를 함께 제시하며 동일 고정 공간의 단독 인과효과로 부르지 않는다. L7은 별도 구조 진단으로 계수 수·buffer·offset이 다르다. K streams+K past buffer+현재 C anchor이므로 strict K-only도 아니다.","","![선정된 full·stop L0 공간 비교](figures/03_full_stop_space_contrasts.png)","[SVG](figures/03_full_stop_space_contrasts.svg)","",
        "**FLAT과 current-C persistence를 구분한다.** FLAT은 미래 latent를 마지막 latent로 반복한 과거 전용 대조다. L7의 알려진 past prefix와 lag별 기준값 때문에 복원 결과가 현재 C 반복과 다를 수 있다. 이는 특정 구조의 반례·현재 조건 비교이며 FLAT의 보편 우수성이나 미래 정답의 복구를 뜻하지 않는다.","","![잠재 persistence와 원 C persistence](figures/04_flat_and_current_C.png)","[SVG](figures/04_flat_and_current_C.svg)","",
        "**검산과 한계.** 저장 actual-F 파일·관측 mask·scaler·origin·선정 상태 SHA를 결속했고 원 실제 점수와 NumPy E/lag/D 재생을 대조했다. LS/SVD 교차 및 별도 감사는 기존 불변 수학 경로를 재사용한 수치 검산이며 새로운 F 실행이나 독립적인 일반화 검증이 아니다. 두 자료의 기노출 구간에서 수행한 적응적 원인 탐색이다. 기존 joint-Y/RF3 등 예측 손실의 E/D 학습을 이미 실행했으므로 이번은 예측 목적 표현의 첫 시도가 아니다. 이 진단으로 새 모델·학습 연장·Bolt/대조학습/그래프 필요성을 결정했다고 쓰지 않는다. 새 timing이 없어 deployment latency나 speedup은 보고하지 않는다. RESOURCE의 CPU/wall은 이 저장 결과 진단 실행 자원이다.",""]
    if tables["FAILURES"]:
        lines += ["**미완료 경로.** 아래 실패는 원 결과대로 보존한다. 원 선정 상태·예측을 바꾸지 않고 해당 진단 원인을 확인한 뒤 새 tag/출력에 재개해야 하며 기존 실패 파일을 덮지 않는다.",""]+[f"- {r['dataset']}/{r['condition']}/{r['split']}: {json.dumps(r['failure'],ensure_ascii=False)}" for r in tables["FAILURES"]]+[""]
    lines += [f"- [{name}.csv](tables/{name}.csv): {len(rows)}행" for name,rows in tables.items()]
    lines += [""]+[f"- [{name}]({rel(item['path'],output)}): `{item['sha256']}`" for name,item in bindings.items()]
    lines += ["","보고서 생성기는 JSON metadata만 읽으며 원 NPZ/model·F·훈련·TEST·Git을 실행하지 않았다. 실행/검산 완료와 연구 해결을 구분한다.",""]
    return "\n".join(lines)


def generate(paths,destination):
    output=resolve(destination); require(output.parent==(HERE/"reports").resolve() and not output.exists(),"FRESH_LOCAL_REPORT_REQUIRED")
    values,bindings=load_verified(paths); tables=collect_tables(values); source=record(__file__); bindings["report_source"]=source
    output.mkdir(parents=True,exist_ok=False); (output/"tables").mkdir()
    for name,rows in tables.items(): write_csv(output/"tables"/(name+".csv"),rows,output)
    images=figures(tables,output); (output/"REPORT_KO.md").write_text(make_text(values,tables,bindings,output),encoding="utf-8",newline="\n")
    require(record(__file__)==source,"REPORT_SOURCE_UNCHANGED")
    for item in bindings.values(): bound(item)
    complete=values["diagnostic"]["status"]=="COMPLETE" and values["audit"]["status"]=="PASS"
    manifest={"status":"COMPLETE" if complete else "PARTIAL_DIAGNOSTIC_REPORTED","diagnostic_status":values["diagnostic"]["status"],"audit_status":values["audit"]["status"],"source":source,"inputs":bindings,
        "files":[{"path":p.relative_to(output).as_posix(),"sha256":sha(p),"bytes":p.stat().st_size} for p in sorted(output.rglob("*")) if p.is_file()],"table_row_counts":{k:len(v) for k,v in tables.items()},
        "PNG_count":4,"SVG_count":4,"scientific_NPZ_or_model_read":False,"new_F_or_training_or_timing":False,"protected_TEST_accessed":False,"all_FP32_bound_or_past_predictability_claimed":False}
    with (output/"MANIFEST.json").open("x",encoding="utf-8") as handle: json.dump(manifest,handle,ensure_ascii=False,indent=2,allow_nan=False); handle.write("\n")
    print(json.dumps({"status":manifest["status"],"manifest":str(output/"MANIFEST.json"),"images":len(images)}),flush=True); return manifest


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("diagnostic","audit","output"): parser.add_argument("--"+key,required=True)
    args=vars(parser.parse_args()); output=args.pop("output"); generate(args,output)
