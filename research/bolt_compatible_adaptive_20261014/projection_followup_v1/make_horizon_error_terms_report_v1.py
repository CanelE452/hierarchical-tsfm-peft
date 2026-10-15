"""JSON-only publication of saved selected-correction error terms."""
from pathlib import Path
import argparse
import json
import math

from make_linear_e_adaptation_report_v2 import require,resolve,record,read,bound,rel,write_csv,sha,same_opaque

HERE=Path(__file__).resolve().parent
RESULT_SHA="605a2dec55f9d7a48bea3df99d177b5c54706d06cf14268de4f8c6fcc32204eb"
CONTROLLER_SHA="d6c93aae038b584d8ca927254c3e1d3fc7a16bf47f79cd97d6286a4239df1172"
MODULE_SHA="806a66415834394d1b5efb07bc95fdbd7d103e6167c15abaf7be03f64026a187"
PARENT_SHA="5f158382640dac74ec63cc3d1005b385c898821d3c032d51b543f2c9b27d7dac"
PARENT_AUDIT_SHA="0f65053536403a3b7b8f53fca3b9f8274fb1a406eaf6dfdcb73bb0467fffe698"
PARENT_AUDITOR_SHA="accb6fb775d9b12ca6e39c0be7e062545b98d429089ee32e0ac46ed65edae613"
DATASETS={"robin":(17,4),"beijing_aotizhongxin":(11,2)}
FIELDS=("baseline_MSE","selected_MSE","delta_MSE","interaction_term","correction_energy","direct_squared_error_change","closure_residual","direct_closure_residual","delta_minus_direct_change","closure_tolerance","closure_pass")
COUNTS={"FULL":72,"HORIZONS":3456,"INTERVALS":216,"PER_CHANNEL":52416,"CONDITIONS":24,"CHANNEL_BALANCE":48,"RESOURCES":3}
FIGURES=("01_F_signed_terms","02_robin_K4_K8_terms","03_horizon_delta","04_channel_balance")


def load_verified(path):
    require(all(isinstance(value,str) and len(value)==64 for value in (RESULT_SHA,CONTROLLER_SHA,MODULE_SHA,PARENT_SHA,PARENT_AUDIT_SHA,PARENT_AUDITOR_SHA)),"ACTUAL_ERROR_TERMS_NOT_REGISTERED")
    reference=record(path);require(reference["sha256"]==RESULT_SHA,"EXACT_ACTUAL_ERROR_TERMS_RESULT");result=read(path)
    require(result["status"]=="COMPLETE" and result["protected_TEST_accessed"] is result["independent_holdout"] is result["whole_deployment_cost_measured"] is False and all(result[key]==0 for key in ("new_fits","new_F_forward_calls","new_model_objects_loaded","new_optimizer_updates","new_selection_decisions","new_DEV_winners_selected")),"COMPLETE_DIAGNOSTIC_NO_NEW_FIT_OR_SELECTION")
    require(result["controller_source"]["sha256"]==CONTROLLER_SHA and result["module_source"]["sha256"]==MODULE_SHA and result["horizon_diagnostic"]["sha256"]==PARENT_SHA and result["horizon_audit"]["sha256"]==PARENT_AUDIT_SHA,"EXACT_DIAGNOSTIC_SOURCES_AND_PARENT")
    references={"diagnostic":reference,**{key:result[key] for key in ("controller_source","module_source","plan","horizon_diagnostic","horizon_audit","cross_moment_diagnostic","cross_moment_audit","original_selection_registry")}}
    for value in references.values():bound(value)
    parent=read(bound(result["horizon_diagnostic"]));audit=read(bound(result["horizon_audit"]));require(parent["status"]=="COMPLETE" and audit["status"]=="PASS","ACCEPTED_PARENT_SCIENCE");bound(audit["diagnostic"],result["horizon_diagnostic"]);require(audit["audit_source"]["sha256"]==PARENT_AUDITOR_SHA,"EXACT_PARENT_AUDITOR");bound(audit["audit_source"]);references["parent_auditor_source"]=audit["audit_source"]
    for data,group in result["datasets"].items():
        for space,owner in group["spaces"].items():
            for name,row in owner["conditions"].items():
                original=parent["datasets"][data]["spaces"][space]["conditions"][name]
                require(all(row[key]==original[key] for key in ("selected_candidate","selected_kind","selected_alpha","selected_TUNE_MSE","selected_coefficient_value_sha256","K")),"UNCHANGED_ORIGINAL_SELECTED_CONDITION")
                for key in ("selected_coefficients","selection_lock"):same_opaque(row[key],original[key])
    resources={"diagnostic":result["resource_terminal"],**result["parent_resource_receipts"]}
    for name,receipt in resources.items():
        require(read(bound(receipt))=={key:value for key,value in receipt.items() if key not in ("path","sha256")} and receipt["status"]==("PASS" if name=="horizon_audit" else "COMPLETE") and receipt["gpu_process"] is False,"EXACT_DIAGNOSTIC_CPU_RECEIPT")
        require(all(type(receipt[key]) in (int,float) and math.isfinite(receipt[key]) and receipt[key]>=0 for key in ("process_cpu_seconds","wall_seconds")),"CANONICAL_CPU_WALL_FIELDS");references[name+"_receipt"]={key:receipt[key] for key in ("path","sha256")}
        authorization=read(bound(receipt["authorization"]));manifest=read(bound(receipt["continuation_manifest"]));require(authorization["status"]=="APPROVED_HUMAN_UNCAPPED_CONTINUATION" and manifest["authorization_sha256"]==receipt["authorization"]["sha256"] and manifest["runtime_sha256"]==sha(HERE/"continuation_runtime_v1.py"),"RESOURCE_AUTHORIZATION_AND_RUNTIME_CHAIN")
    return result,references


def tables(result):
    output={key:[] for key in COUNTS};require(set(result["datasets"])==set(DATASETS),"EXACT_TWO_DATASETS")
    for data,(channels,k) in DATASETS.items():
        spaces=result["datasets"][data]["spaces"];require(set(spaces)=={"RAW_C","PCA_K"+str(k),"PCA_K"+str(2*k)},"EXACT_RAW_AND_TWO_PCA_SPACES")
        for space,owner in spaces.items():
            require(set(owner["conditions"])=={mode+"_"+arm for mode in ("F","FLAT") for arm in ("STATIC_DELTA","RAMP_DELTA")},"EXACT_FOUR_FIXED_CONDITIONS")
            for name,row in owner["conditions"].items():
                mode,arm=name.split("_",1);identity={"dataset":data,"space":space,"condition":name,"mode":mode,"arm":arm,"K":row["K"],"selected_candidate":row["selected_candidate"],"selected_kind":row["selected_kind"],"selected_alpha":row["selected_alpha"],"selected_coefficient_value_sha256":row["selected_coefficient_value_sha256"]}
                for split in ("FIT","TUNE","DEV"):
                    entry=row[split];require(len(entry["per_horizon"])==48 and set(entry["registered_intervals"])=={"1_16","17_32","33_48"},"COMPLETE_H48_AND_THREE_INTERVALS")
                    cells=[("FULL",None,entry["full"])]+[("HORIZONS",h+1,value) for h,value in enumerate(entry["per_horizon"])]+[("INTERVALS",key,entry["registered_intervals"][key]) for key in ("1_16","17_32","33_48")]
                    for scope,index,cell in cells:
                        require(len(cell["per_channel"])==len(cell["observed_counts"])==channels,"EXACT_ORIGINAL_CHANNEL_COUNTS")
                        values={key:cell[key] for key in FIELDS};require(all(value is None or type(value) is bool or (type(value) in (int,float) and math.isfinite(value)) for value in values.values()),"FINITE_OR_NULL_ERROR_TERMS")
                        record_row={**identity,"split":split,"status":cell["status"],"observed_counts":cell["observed_counts"],"observed_values":cell["observed_values"],"active_channels":cell["active_channels"],**values,"source_score_links":cell["source_score_links"]}
                        if scope=="HORIZONS":record_row["horizon"]=index
                        elif scope=="INTERVALS":record_row["interval"]=index
                        else:record_row.update(source_bindings=entry["source_bindings"],prediction_baseline_byte_equal=entry["prediction_baseline_byte_equal"])
                        output[scope].append(record_row)
                        for channel in cell["per_channel"]:output["PER_CHANNEL"].append({**identity,"split":split,"scope":scope,"horizon_or_interval":index,**channel})
                    if split!="FIT":
                        cells=entry["full"]["per_channel"];active=[cell for cell in cells if cell["observed_count"]]
                        output["CHANNEL_BALANCE"].append({**identity,"split":split,"channel_count":channels,"active_channels":len(active),"improved_channels":sum(cell["delta_MSE"]<0 for cell in active),"worse_channels":sum(cell["delta_MSE"]>0 for cell in active),"unchanged_channels":sum(cell["delta_MSE"]==0 for cell in active),"unobserved_channels":channels-len(active),"sign_test_or_independent_CI":False})
                tune,dev=row["TUNE"]["full"],row["DEV"]["full"];observation=tune["delta_MSE"]<0<dev["delta_MSE"];require(observation==row["TUNE_DEV_observation"]["TUNE_delta_negative_DEV_delta_positive"],"RECORDED_TUNE_DEV_SIGN_OBSERVATION")
                output["CONDITIONS"].append({**identity,"selected_coefficients":row["selected_coefficients"],"selection_lock":row["selection_lock"],"TUNE_improved_DEV_worse":observation,"TUNE_interaction":tune["interaction_term"],"TUNE_energy":tune["correction_energy"],"TUNE_delta":tune["delta_MSE"],"DEV_interaction":dev["interaction_term"],"DEV_energy":dev["correction_energy"],"DEV_delta":dev["delta_MSE"],"observation":row["TUNE_DEV_observation"]})
    for stage,value in {"diagnostic":result["resource_terminal"],**result["parent_resource_receipts"]}.items():output["RESOURCES"].append({"stage":stage,"resource_terminal":value})
    require({name:len(rows) for name,rows in output.items()}==COUNTS,"EXACT_ERROR_TERMS_TABLE_COUNTS");return output


def figures(values,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    folder=output/"figures";folder.mkdir()
    def save(fig,name):
        for extension in ("png","svg"):fig.savefig(folder/(name+"."+extension),dpi=160,bbox_inches="tight")
        plt.close(fig)
    def label(row):return row["space"].replace("PCA_","")+" "+("S" if row["arm"]=="STATIC_DELTA" else "R")+" "+row["split"]
    def term_panel(ax,rows,title):
        y=np.arange(len(rows));ax.barh(y-.13,[row["interaction_term"] for row in rows],.25,color="#277da8",label="Interaction 2 E[e0 c]");ax.barh(y+.13,[row["correction_energy"] for row in rows],.25,color="#ed9b40",label="Energy E[c^2]");ax.scatter([row["delta_MSE"] for row in rows],y,color="black",marker="D",s=20,label="Signed delta MSE",zorder=4);ax.axvline(0,color="gray",linewidth=.8);ax.set_yticks(y,[label(row) for row in rows],fontsize=8);ax.invert_yaxis();ax.set_title(title);ax.set_xlabel("All-observed channel-macro normalized MSE terms");ax.grid(axis="x",alpha=.18)
    fig,axes=plt.subplots(1,2,figsize=(17,9))
    for ax,data in zip(axes,DATASETS):
        rows=[row for row in values["FULL"] if row["dataset"]==data and row["mode"]=="F" and row["split"] in ("TUNE","DEV")];term_panel(ax,rows,data+" / fixed F corrections");ax.legend(fontsize=8,loc="best")
    fig.suptitle("Blue + orange = black diamond; negative delta improves its own fixed baseline");fig.tight_layout(rect=(0,0,1,.95));save(fig,FIGURES[0])
    fig,axes=plt.subplots(1,2,figsize=(15,5.8))
    for ax,space in zip(axes,("PCA_K4","PCA_K8")):
        rows=[row for row in values["FULL"] if row["dataset"]=="robin" and row["space"]==space and row["mode"]=="F" and row["split"] in ("TUNE","DEV")];term_panel(ax,rows,"Robin / "+space);ax.legend(fontsize=8)
    fig.suptitle("Fixed selected corrections: inspect interaction sign and energy separately");fig.tight_layout(rect=(0,0,1,.93));save(fig,FIGURES[1])
    fig,axes=plt.subplots(2,2,figsize=(18,13));cmap=plt.get_cmap("RdBu_r").copy();cmap.set_bad("#dedede")
    for ax,(data,mode) in zip(axes.flat,((data,mode) for data in DATASETS for mode in ("F","FLAT"))):
        keys=[row for row in values["FULL"] if row["dataset"]==data and row["mode"]==mode and row["split"] in ("TUNE","DEV")];matrix=[]
        for key in keys:
            rows=[row for row in values["HORIZONS"] if all(row[field]==key[field] for field in ("dataset","condition","space","split"))];matrix.append([np.nan if row["delta_MSE"] is None else row["delta_MSE"] for row in rows])
        matrix=np.asarray(matrix);limit=max(float(np.nanmax(np.abs(matrix))),1e-12);image=ax.imshow(matrix,aspect="auto",cmap=cmap,vmin=-limit,vmax=limit);ax.set_title(data+" / "+mode+" / panel-specific symmetric scale");ax.set_yticks(range(len(keys)),[label(row) for row in keys],fontsize=8);ax.set_xticks((0,15,31,47),(1,16,32,48));ax.set_xlabel("Forecast horizon (each has its own observed channel counts)");fig.colorbar(image,ax=ax,shrink=.8,label="Signed delta MSE")
    fig.suptitle("Blue improves, red worsens its own baseline; gray means no observed targets");fig.tight_layout(rect=(0,0,1,.95));save(fig,FIGURES[2])
    fig,axes=plt.subplots(2,2,figsize=(18,13))
    for ax,(data,mode) in zip(axes.flat,((data,mode) for data in DATASETS for mode in ("F","FLAT"))):
        rows=[row for row in values["CHANNEL_BALANCE"] if row["dataset"]==data and row["mode"]==mode];y=np.arange(len(rows));left=np.zeros(len(rows))
        for field,color,title in (("improved_channels","#277da8","Improved"),("worse_channels","#ed9b40","Worse"),("unchanged_channels","#b5b5b5","Exact unchanged"),("unobserved_channels","#f0f0f0","No observations")):
            width=np.array([row[field] for row in rows]);ax.barh(y,width,left=left,color=color,label=title);left+=width
        ax.set_yticks(y,[label(row) for row in rows],fontsize=8);ax.invert_yaxis();ax.set_title(data+" / "+mode);ax.set_xlabel("Channel counts from full-split observed means");ax.set_xlim(0,DATASETS[data][0]);ax.legend(fontsize=8);ax.grid(axis="x",alpha=.15)
    fig.suptitle("All 24 fixed conditions, two splits each; channels and horizons are not independent replicates");fig.tight_layout(rect=(0,0,1,.95));save(fig,FIGURES[3])


def generate(path,destination):
    output=resolve(destination);require(output.parent==(HERE/"reports").resolve() and not output.exists(),"FRESH_ERROR_TERMS_REPORT_OUTPUT");result,references=load_verified(path);values=tables(result);source=record(__file__);output.mkdir();(output/"tables").mkdir()
    for name,rows in values.items():write_csv(output/("tables/"+name+".csv"),rows,output)
    figures(values,output);reversals=sum(row["TUNE_improved_DEV_worse"] for row in values["CONDITIONS"])
    lines=["# 선정 보정이 바꾼 오차: 상호작용과 보정 에너지","",f"[확인] 이미 선정된24조건 중 {reversals}/24는 TUNE에서 자기 baseline보다 MSE를 낮췄지만 노출 DEV에서는 높였다. 새 fit/F/모델/optimizer/선정/DEV winner는 모두0이며, 원24개 잠금과 선정 계수를 유지했다. 이 문서는 저장 출력의 진단 재계산이다. 외부 독립 감사나 새 방법의 일반화 성공을 뜻하지 않는다.","","실제로 반환된 FP32 baseline b, 선정 예측 p, 관측 정답 y를 각각 FP64로 올리고 e0=b−y, c=p−b로 계산했다. c는 이상적인 feature×계수 대신 반올림이 끝난 실제 출력의 변화량이다. 같은 관측 집합에서 ΔMSE=2 macro E[e0c]+macro E[c²]이며 음수 Δ는 그 조건의 자기 baseline 대비 개선이다. 상호작용이 음수면 기존 오차를 줄이는 방향으로 작용하고, 보정 에너지는 비음수다. 이 두 항을 합한 부호와 크기로 결과를 해석한다.","","각 채널의 true-observed 평균을 먼저 내고 active 채널에 같은 가중치를 준다. 전체·각 h·사전1–16/17–32/33–48 구간은 각각 자기 관측 분모를 쓴다. h별 macro의 단순평균으로 전체 macro를 만들지 않았고 미관측 채널·구간은 NA다. F/FLAT·STATIC/RAMP·Raw/PCA K를 모두 유지한다. Raw는 원 median F 또는 현재C baseline이며 새 분위 요약 Raw로 교체하지 않았다.","","![F 조건의 상호작용과 보정 에너지](figures/01_F_signed_terms.png)",""]
    def example(data,space,arm):
        row=next(row for row in values["CONDITIONS"] if row["dataset"]==data and row["space"]==space and row["mode"]=="F" and row["arm"]==arm)
        observation=""
        if row["TUNE_interaction"]<0<row["DEV_interaction"]:observation=" 상호작용이 TUNE의 음수에서 DEV의 양수로 바뀌었다."
        elif row["DEV_interaction"]<0 and row["DEV_energy"]>-row["DEV_interaction"]:observation=" DEV 상호작용은 음수지만 보정 에너지가 그 절댓값보다 커서 Δ가 양수다."
        elif row["TUNE_delta"]<0 and row["DEV_delta"]<0:observation=" 두 split에서 자기 baseline 대비 개선이 유지됐다."
        return f"{data} {space} {arm}: TUNE 상호작용 {row['TUNE_interaction']:.12g} + 에너지 {row['TUNE_energy']:.12g} = Δ {row['TUNE_delta']:.12g}; DEV {row['DEV_interaction']:.12g} + {row['DEV_energy']:.12g} = Δ {row['DEV_delta']:.12g}."+observation
    lines += [example("robin","PCA_K4","STATIC_DELTA"),"",example("robin","PCA_K8","STATIC_DELTA"),"",example("robin","PCA_K8","RAMP_DELTA"),"",example("beijing_aotizhongxin","RAW_C","RAMP_DELTA"),"","![Robin K4와 K8의 고정 보정](figures/02_robin_K4_K8_terms.png)","","상호작용의 부호와 보정 에너지의 크기를 각각 읽어야 한다. 항의 변화는 관찰된 오차 증가·감소의 분해이며 과적합·공변량 변화·학습량 부족의 유일한 인과 증명이 아니다. 특정 DEV 결과로 새 arm·alpha·공간을 고르지 않는다.","","![h별 signed MSE 변화](figures/03_horizon_delta.png)","","h 그림은 각 h의 원 true-observed mask를 그대로 쓴 자기 baseline 대비 Δ다. 각 패널의 색 범위는 따로 대칭 설정했으므로 서로 다른 패널의 색 강도를 같은 절댓값으로 비교하지 않는다. 채널 그림은 전체 split의 각 채널 MSE 부호를 센 것으로 독립 반복 표본·유의성·신뢰구간이 아니다.","","![전체24조건의 채널별 개선과 악화 수](figures/04_channel_balance.png)","",f"전체72·h3456·구간216·per-C52416 cell을 CSV에 보존했다. 원 진단의 최대 closure 절댓값은 {result['maximum_closure_abs']:.12g}이며 원 점수 연결 공차는 사전 atol=rtol=1e-12다. ZERO는 원 baseline bytes를 유지했다. 자원 표의 세 행은 원 hD fit·원 과학 감사·저장 오차항 진단의 CPU 영수증이다. 이 보고서 그림 생성 시간이나 전체 배포 latency로 쓰지 않는다. 새 비용 측정은 하지 않았고 DEV는 이미 노출된 적응적 분석 자료이며 보호 TEST 접근0이다.",""]
    lines += [f"- [{name} {len(rows)}행](tables/{name}.csv)" for name,rows in values.items()];lines += [""]+[f"- [{name}]({rel(value['path'],output)}): `{value['sha256']}`" for name,value in references.items()]
    (output/"REPORT_KO.md").write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n");require(record(__file__)==source,"REPORT_SOURCE_UNCHANGED")
    for value in references.values():bound(value)
    manifest={"status":"COMPLETE","source":source,"inputs":references,"table_row_counts":COUNTS,"files":[{"path":path.relative_to(output).as_posix(),"sha256":sha(path),"bytes":path.stat().st_size} for path in sorted(output.rglob("*")) if path.is_file()],"PNG_count":4,"SVG_count":4,"scientific_NPZ_or_model_read":False,"new_fit_or_selection":False,"external_independent_audit_claimed":False,"protected_TEST_accessed":False,"independent_holdout":False}
    with (output/"MANIFEST.json").open("x",encoding="utf-8") as handle:json.dump(manifest,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write("\n")
    print(json.dumps({"status":"COMPLETE","manifest":str(output/"MANIFEST.json")}));return manifest


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("result","output"):parser.add_argument("--"+name,required=True)
    args=parser.parse_args();generate(args.result,args.output)
