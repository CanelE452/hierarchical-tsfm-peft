"""JSON-only conditional A/B/C readout report; actual gates stay closed until registered."""
from pathlib import Path
import argparse
import json
import math

from make_linear_e_adaptation_report_v2 import require,resolve,record,read,bound,rel,write_csv,sha

HERE=Path(__file__).resolve().parent
FIXED_SHA={"result":"4ad5ee73f789a3b21eab8d0b732b04bde2622e5380e915a07f226869a2a6becf","audit":"4113a7dd05616f4ee0285afa4714657ec295674b4bbc8098950b1a345a6b9525"}
SOURCE_SHA={"controller_source":"777c147fa77f63b180470918b902cfde1d07d205d87caa8739cc35e3ec892734","module_source":"406252e940ae676e7a78f69e067f5b3fa70456bdb1ea27d57169b8964ceabe0e","audit_source":"245808368224b4af23a57cce542029eb8acc458d80bb9f7511eb9d7961253b8f"}
PARENT_SHA={"horizon_diagnostic":"5f158382640dac74ec63cc3d1005b385c898821d3c032d51b543f2c9b27d7dac","horizon_audit":"0f65053536403a3b7b8f53fca3b9f8274fb1a406eaf6dfdcb73bb0467fffe698","cross_moment_diagnostic":"99bffd5831143ca35947099fef9567f89dd209d83451661decee0682b2b8e13c","cross_moment_audit":"90c68a9057909fb5b6c23284e5a45523cdf126da178a5ba0d9d9e96a50dede30"}
ROUTES=("PCA_K8","RAW_C")
SPLITS=("A","B","C")
ALPHAS=(0.,1e-4,.001,.01,.1,1.,10.,100.,1000.,10000.)
TABLES=("SELECTED","CANDIDATES","SPLITS","ERROR_TERMS","MARGINAL_TERMS","RESOURCES")
TERM_FIELDS=("baseline_MSE","selected_MSE","delta_MSE","interaction_term","correction_energy","closure_residual","closure_pass")
FIGURES=("01_selected_and_ZERO","02_interaction_and_energy","03_origin_purge_context")
AUDIT_COUNTS={"readout_fit_paths_checked":2,"alpha_candidates_checked":20,"ZERO_candidates_checked":2,"A_B_prediction_evaluations_checked":44,"selected_C_prediction_evaluations_checked":2,"coefficient_artifacts_checked":22,"FP32_arithmetic_predictions_checked":46,"selected_fits_independently_solved":2,"selected_ZERO_states_checked":0,"observed_mask_groups_checked":2,"selection_locks_checked":2,"score_fields_checked":1344,"temporal_score_fields_checked":1836,"endpoint_diagnostic_cells_checked":312,"protected_source_input_and_result_hashes_checked":2442}


def load_verified(result_path,audit_path):
    require(all(isinstance(value,str) and len(value)==64 for value in (*FIXED_SHA.values(),*SOURCE_SHA.values())),"ACTUAL_PURGED_RESULT_AUDIT_NOT_REGISTERED")
    references={"result":record(result_path),"audit":record(audit_path)};require(all(references[key]["sha256"]==value for key,value in FIXED_SHA.items()),"EXACT_ACTUAL_PURGED_RESULT_AUDIT");result,audit=read(result_path),read(audit_path)
    require(result["status"] in ("COMPLETE","PARTIAL_COMPLETED") and audit["status"]=="PASS","ACCEPTED_PURGED_RESULT_STATUS");bound(audit["diagnostic"],references["result"])
    require(all(audit["totals"][key]==value for key,value in AUDIT_COUNTS.items()) and not audit["partial_paths"],"FULL_PURGED_SAVED_ARITHMETIC_AUDIT")
    require(result["fixed_full_FIT_upstream"] is result["conditional_readout_diagnostic"] is True and result["independent_holdout"] is result["end_to_end_temporal_validation"] is False,"CONDITIONAL_ALREADY_SEEN_FULL_FIT_SCOPE")
    require(result["C_targets_loaded_with_full_FIT_archive"] is result["C_baseline_and_features_available_from_fixed_full_FIT_context_before_locks"] is result["C_mask_and_timestamp_facts_computed_before_locks"] is True and result["NEW_SELECTED_C_predictions_or_C_scores_before_locks"] is result["independent_OS_C_access_events_verified"] is False,"PRECISE_CONDITIONAL_C_ACCESS_SCOPE")
    require(all(result[key]==0 for key in ("new_F_forward_calls","new_model_objects_loaded","new_optimizer_updates","new_E_D_training_updates")) and result["TUNE_DEV_arrays_deserialized"] is result["TUNE_DEV_scored"] is result["protected_TEST_accessed"] is False,"FIT_ONLY_NEW_READOUT_NO_NEW_F_OR_PROTECTED_SPLIT")
    for key in ("controller_source","module_source"):require(result[key]["sha256"]==SOURCE_SHA[key],"EXACT_PURGED_SOURCE");bound(result[key]);bound(audit[key],result[key]);references[key]=result[key]
    require(result["error_terms_module_source"]["sha256"]=="806a66415834394d1b5efb07bc95fdbd7d103e6167c15abaf7be03f64026a187","EXACT_ERROR_TERMS_MODULE");bound(result["error_terms_module_source"]);bound(audit["error_terms_module_source"],result["error_terms_module_source"]);references["error_terms_module_source"]=result["error_terms_module_source"]
    require(audit["audit_source"]["sha256"]==SOURCE_SHA["audit_source"],"EXACT_PURGED_AUDITOR_SOURCE");bound(audit["audit_source"]);references["auditor_source"]=audit["audit_source"]
    for key,value in PARENT_SHA.items():require(result[key]["sha256"]==value,"EXACT_ACCEPTED_PARENT");bound(result[key]);bound(audit[key],result[key]);references[key]=result[key]
    for key in ("plan","selection_registry"):bound(result[key]);bound(audit[key],result[key]);references[key]=result[key]
    for name,value in (("result",result),("audit",audit)):
        receipt=value["resource_terminal"];require(read(bound(receipt))=={key:value for key,value in receipt.items() if key not in ("path","sha256")} and receipt["gpu_process"] is False,"EXACT_CPU_RESOURCE_RECEIPT")
        require(receipt["status"]==value["status"] and all(type(receipt[key]) in (int,float) and math.isfinite(receipt[key]) and receipt[key]>=0 for key in ("process_cpu_seconds","wall_seconds")),"CANONICAL_CPU_WALL_STATUS");references[name+"_receipt"]={key:receipt[key] for key in ("path","sha256")}
        authorization=read(bound(receipt["authorization"]));manifest=read(bound(receipt["continuation_manifest"]));require(authorization["status"]=="APPROVED_HUMAN_UNCAPPED_CONTINUATION" and manifest["authorization_sha256"]==receipt["authorization"]["sha256"] and manifest["runtime_sha256"]==sha(HERE/"continuation_runtime_v1.py"),"AUTHORIZATION_RUNTIME_CHAIN")
    return result,audit,references


def tables(result,audit):
    values={key:[] for key in TABLES};require(set(result["routes"])==set(ROUTES) and set(result["partition"]["cohorts"])==set(SPLITS),"TWO_FIXED_ROUTES_THREE_COHORTS")
    for split in SPLITS:values["SPLITS"].append({"split":split,**result["partition"]["cohorts"][split],"purge":result["partition"]["purge"]})
    for route in ROUTES:
        row=result["routes"][route];complete=row["status"]=="COMPLETE";summary={"route":route,"status":row["status"],"failure":row.get("failure"),"selected_candidate":row["selected_candidate"] if complete else None,"selected_kind":row["selected_kind"] if complete else None,"selected_alpha":row["selected_alpha"] if complete else None,"selected_B_MSE":row["selected_B_MSE"] if complete else None,"selection_lock":row["selection_lock"] if complete else row.get("selection_lock")}
        summary["recorded_failures"]=[failure for failure in result["failed_paths"] if failure["space"]==route]
        for key in ("K","additional_slope_parameters","source_context","fit_metadata","selected_coefficients","selected_coefficient_value_sha256","selected_A_B_prediction_artifact"):summary[key]=row[key] if complete else row.get(key)
        for split in SPLITS:
            available=complete and split in row and row[split]["status"]=="COMPLETE";summary[split+"_status"]=row[split]["status"] if complete and split in row else "UNAVAILABLE";summary[split+"_selected_MSE"]=row[split]["scores_all_observed"]["normalized"]["mse"] if available else None;summary[split+"_ZERO_MSE"]=row["baseline_scores"][split]["scores_all_observed"]["normalized"]["mse"] if available else None
        values["SELECTED"].append(summary)
        for candidate in row["candidates"]:
            ok=candidate["status"]=="COMPLETE";values["CANDIDATES"].append({"route":route,"candidate":candidate["id"],"kind":candidate["kind"],"alpha":candidate["alpha"],"status":candidate["status"],"selected":complete and candidate["id"]==row["selected_candidate"],"A_MSE":candidate["A"]["scores_all_observed"]["normalized"]["mse"] if ok else None,"B_MSE":candidate["B_MSE"] if ok else None,"failure":None if ok else candidate.get("failure")})
        if not complete:continue
        require(len(row["candidates"])==11,"FULL_REGISTERED_A_FIT_B_SELECTION_MENU")
        for split in SPLITS:
            if split not in row or row[split]["status"]!="COMPLETE":continue
            entry=row[split]["error_terms"];full=entry["full"];values["ERROR_TERMS"].append({"route":route,"split":split,"selected_candidate":row["selected_candidate"],"observed_counts":full["observed_counts"],"active_channels":full["active_channels"],**{key:full[key] for key in TERM_FIELDS}})
            for channel in full["per_channel"]:values["MARGINAL_TERMS"].append({"route":route,"split":split,"scope":"CHANNEL","interval":None,**channel})
            for interval in ("1_16","17_32","33_48"):values["MARGINAL_TERMS"].append({"route":route,"split":split,"scope":"INTERVAL","interval":interval,**entry["registered_intervals"][interval]})
    for name,value in (("result",result),("audit",audit)):values["RESOURCES"].append({"stage":name,"resource_terminal":value["resource_terminal"]})
    return values


def figures(values,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    folder=output/"figures";folder.mkdir()
    def save(fig,name):
        for extension in ("png","svg"):fig.savefig(folder/(name+"."+extension),dpi=160,bbox_inches="tight")
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,5.5))
    for ax,split in zip(axes,SPLITS):
        labels=[];scores=[];colors=[]
        for row in values["SELECTED"]:
            for kind,color in (("ZERO","#a8a8a8"),("selected","#277da8" if row["route"]=="PCA_K8" else "#ed9b40")):
                labels.append(row["route"].replace("RAW_C","Raw F")+"\n"+kind);scores.append(np.nan if row[split+"_"+kind+"_MSE"] is None else row[split+"_"+kind+"_MSE"]);colors.append(color)
        ax.bar(range(4),scores,color=colors);ax.set_xticks(range(4),labels,fontsize=8);ax.set_title(split+" / "+{"A":"readout fit","B":"own selection","C":"fixed later evaluation"}[split]);ax.set_ylabel("All-observed channel-macro normalized MSE");ax.grid(axis="y",alpha=.18)
        for x,value in enumerate(scores):
            if np.isfinite(value):ax.annotate(format(value,".5g"),(x,value),xytext=(0,4),textcoords="offset points",ha="center",fontsize=8)
            else:ax.text(x,0,"Unavailable",ha="center",rotation=90)
    fig.suptitle("Conditional on full-FIT upstream; own ZERO and Raw retain the same cohorts");fig.tight_layout(rect=(0,0,1,.93));save(fig,FIGURES[0])
    fig,axes=plt.subplots(1,3,figsize=(16,5.2))
    for ax,split in zip(axes,SPLITS):
        rows=[row for row in values["ERROR_TERMS"] if row["split"]==split];y=np.arange(len(rows));ax.barh(y-.13,[row["interaction_term"] for row in rows],.25,color="#277da8",label="Interaction 2 E[e0 c]");ax.barh(y+.13,[row["correction_energy"] for row in rows],.25,color="#ed9b40",label="Energy E[c^2]");ax.scatter([row["delta_MSE"] for row in rows],y,color="black",marker="D",label="Delta MSE",zorder=4);ax.set_yticks(y,[row["route"] for row in rows]);ax.axvline(0,color="gray",linewidth=.8);ax.set_title(split);ax.set_xlabel("Signed normalized MSE terms");ax.grid(axis="x",alpha=.18);ax.legend(fontsize=8)
    fig.suptitle("Blue + orange = diamond; actual returned FP32 endpoints interpreted as FP64");fig.tight_layout(rect=(0,0,1,.93));save(fig,FIGURES[1])
    fig,(ax,context)=plt.subplots(2,1,figsize=(15,8),gridspec_kw={"height_ratios":[1,1.6]});colors=("#277da8","#43aa8b","#ed9b40")
    for index,(row,color) in enumerate(zip(values["SPLITS"],colors)):
        ax.barh(index,row["origins_max"]-row["origins_min"]+47,left=row["origins_min"],height=.45,color=color,alpha=.4);ax.plot([row["origins_min"],row["origins_max"]],[index,index],color=color,linewidth=4);ax.text(row["origins_min"],index+.28,row["split"]+" / n="+str(row["origin_count"]),fontsize=9)
    purge=values["SPLITS"][0]["purge"]
    for key,color in (("first_B_origin","#43aa8b"),("first_C_origin","#ed9b40")):ax.axvline(purge[key],color=color,linestyle="--",label=key+"="+str(purge[key]))
    ax.set_yticks([]);ax.invert_yaxis();ax.set_xlabel("Origin index = first response time; pale extent reaches last origin + 47");ax.legend(fontsize=8);ax.grid(axis="x",alpha=.15);context.axis("off")
    lines=["A candidates="+str(purge["A_candidate_count"])+", purged="+str(purge["A_removed_count"])+"; B candidates="+str(purge["B_candidate_count"])+", purged="+str(purge["B_removed_count"])+"; C fixed="+str(purge["C_count"]),"Strict response purge: o + 47 < next cohort's first origin; shared history is allowed."]
    for row in values["SPLITS"]:lines += ["",row["split"]+": timestamp "+str(row["first_target_timestamp"])+" .. "+str(row["last_target_timestamp"]),"origins="+str(row["origin_count"])+" / response slots="+str(row["horizon_slots"])+" / unique times="+str(row["unique_target_timestamps"])+" / max reuse="+str(row["max_target_multiplicity"])]
    context.text(0,1,"\n".join(lines),va="top",fontsize=10,linespacing=1.45);fig.suptitle("Conditional chronology: full-FIT upstream fixed; C labels are loaded with the full FIT archive");fig.tight_layout(rect=(0,0,1,.95));save(fig,FIGURES[2])


def generate(result_path,audit_path,destination):
    output=resolve(destination);require(output.parent==(HERE/"reports").resolve() and not output.exists(),"FRESH_PURGED_REPORT_OUTPUT");result,audit,references=load_verified(result_path,audit_path);values=tables(result,audit);source=record(__file__);output.mkdir();(output/"tables").mkdir()
    for name,rows in values.items():write_csv(output/("tables/"+name+".csv"),rows,output)
    figures(values,output);totals=result["totals"];failed=[row for row in values["SELECTED"] if row["status"]!="COMPLETE" or row["C_selected_MSE"] is None];finished=[row for row in values["SELECTED"] if row["status"]=="COMPLETE" and row["C_selected_MSE"] is not None];reversed_count=sum(row["B_selected_MSE"]<row["B_ZERO_MSE"] and row["C_selected_MSE"]>row["C_ZERO_MSE"] for row in finished)
    lines=["# 고정 upstream을 조건으로 한 보정의 시간 이전 진단","",
        f"[확인] C까지 완료된 {len(finished)}경로 중 {reversed_count}경로는 B에서 자기 ZERO보다 낮은 MSE를 보였지만 C에서는 높았다. 이 조건부 진단에서 B의 이득이 C에 유지됐는지에 답하며, 실패의 유일한 원인이나 모든 readout의 불가능성을 증명하지 않는다.","",
        "[확인] Robin FIT256 안의 앞128·중간64·뒤64 후보를 시간순 A/B/C로 고정하고, A/B에서 다음 구간의 첫 origin 전에 끝나지 않는48-step response를 purge했다. 실제 남은 수·timestamp 경계·반복 시점 수는 아래 그림과 SPLITS 표에 보존한다. A의 actual-F ungated RMS와 계수만 새로 적합하고 B의 자기 점수로 선정했다. 두 B 잠금 뒤에만 새 선정 보정의 C 예측·점수를 계산하며 C로 새 선정을 하지 않는다.","",
        "**full-FIT upstream 고정 조건부 진단이다.** 기존 scaler·PCA E/center·RECON D·median F cache는 A 밖까지 포함한 full FIT에서 이미 정해졌다. 원 FIT 자료도 이전 연구에서 사용했다. 독립 holdout이나 전체 end-to-end temporal validation으로 부르지 않는다. response 시점의 purge는 과거 history 공유를 없애지 않으며 L+H 완전 분리가 아니다.","",
        "원 full-FIT archive를 deserialize할 때 C 정답도 메모리에 로드되고, 고정 upstream의 C baseline·feature 및 C mask·timestamp 요약도 잠금 전에 이미 사용 가능하다. 따라서 C blind·C 접근0·모든 C 출력의 부재를 주장하지 않는다. 두 B 잠금 전 새 선정 보정의 C 예측·C 점수·C를 사용한 선정0은 source 흐름의 계약이며 독립 OS 접근 추적 증거가 아니다. 원 TUNE/DEV/TEST를 이 readout의 적합·선정 자료로 쓰지 않았다.","",
        "A-only uncentered RMS와 절편 없는 STATIC slope ridge를 쓰며, 10개 alpha와 ZERO의 실제 FP32 B 관측 채널-macro MSE로 고른다. 정확한 동률은 등록 alpha 순서 뒤 ZERO다. PCA K8과 Raw K17의 추가 slope 수는 달라 동일 매개변수 용량 대조를 주장하지 않는다. timestamp는 원 정수 index이며 달력 시각으로 추정하지 않는다.","",
        f"실행 metadata: fits {totals['fits_completed']}/{totals['fits_attempted']}, alpha 후보 {totals['alpha_candidates_completed']}, ZERO {totals['ZERO_candidates_completed']}, A/B 평가 {totals['candidate_A_B_evaluations_completed']}, C 평가 {totals['selected_C_evaluations_completed']}, 잠금 {totals['resolved_selection_locks']}. 보고서에 보존한 실패 경로는 {len(failed)}개다. 실패 지표는0으로 채우지 않는다.","","![A/B/C 선정과 자기 ZERO·Raw 비교](figures/01_selected_and_ZERO.png)",""]
    for row in values["SELECTED"]:
        if row["status"]=="COMPLETE" and row["C_selected_MSE"] is not None:lines.append(f"- {row['route']}: {row['selected_candidate']}, B {row['B_selected_MSE']:.12g} / ZERO {row['B_ZERO_MSE']:.12g}, C {row['C_selected_MSE']:.12g} / ZERO {row['C_ZERO_MSE']:.12g}.")
        else:lines.append("- "+row["route"]+": "+row["status"]+", "+json.dumps({"failure":row["failure"],"recorded_failures":row["recorded_failures"]},ensure_ascii=False))
    lines += ["","주 비교는 이미 고정된 B/C의 PCA_selected−PCA_ZERO, Raw_selected−Raw_ZERO, PCA_selected−Raw_selected다. 각각의 자기 ZERO와 같은 구간 Raw를 구분한다. A 점수는 적합 구간의 진단이며 일반화 성능으로 해석하지 않는다.",""]
    for split,comparison in result["comparisons"].items():lines.append("- "+split+": "+", ".join(key+"="+format(value,".12g") for key,value in comparison.items())+".")
    lines += ["","![반환된 보정의 상호작용과 에너지](figures/02_interaction_and_energy.png)","","e0=baseline32_as64−Y32_as64, c=selected32_as64−baseline32_as64로 두고 같은 관측 mask에서 ΔMSE=2 macro E[e0c]+macro E[c²]를 읽는다. 각 채널의 true-observed 평균을 먼저 내고 active 채널을 균등 평균한다. 전체 per-C와 사전1–16/17–32/33–48 구간은 보조 진단으로 남기며 미관측 값은 NA다. 이 산술 분해는 과적합·분포 변화의 유일한 인과 증명이 아니다. 구간·채널·rolling origin·반복 target 시점을 독립 표본으로 세거나 p값·CI를 만들지 않는다.","","![실제 origin 범위·purge와 timestamp 요약](figures/03_origin_purge_context.png)","","양성 관찰도 기존 노출 DEV 실패·Raw gap·독립성 부족을 없애지 않는다. 음성 관찰로 더 복잡한 D를 자동으로 추가하거나 모든 readout의 불가능성을 주장하지 않는다. 새 계수의 비용은 미측정이며 이전 timer를 새 정확도에 대입하지 않는다. 자원 표는 이번 적합·저장·검산의 CPU 영수증이고 전체 배포 latency가 아니다. 선정 계수의 독립 검산과 미선정 전체 학습 재실행·F/GPU 재생을 구분한다.",""]
    lines += [f"- [{name} {len(rows)}행](tables/{name}.csv)" for name,rows in values.items()];lines += [""]+[f"- [{name}]({rel(value['path'],output)}): `{value['sha256']}`" for name,value in references.items()];(output/"REPORT_KO.md").write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n");require(record(__file__)==source,"REPORT_SOURCE_UNCHANGED")
    for value in references.values():bound(value)
    manifest={"status":"COMPLETE","source":source,"inputs":references,"files":[{"path":path.relative_to(output).as_posix(),"sha256":sha(path),"bytes":path.stat().st_size} for path in sorted(output.rglob("*")) if path.is_file()],"table_row_counts":{name:len(rows) for name,rows in values.items()},"PNG_count":3,"SVG_count":3,"scientific_NPZ_or_model_read":False,"independent_holdout":False,"end_to_end_temporal_validation":False,"fixed_full_FIT_upstream":True,"C_targets_loaded_with_full_FIT_archive":True,"C_baseline_and_features_available_before_locks":True,"C_mask_and_timestamp_facts_computed_before_locks":True,"NEW_SELECTED_C_predictions_or_C_scores_before_locks":False,"C_blindness_or_independent_OS_access_audit_claimed":False,"new_coefficient_cost_measured":False,"protected_TEST_accessed":False,"failed_route_count":len(failed),"preserved_failure_records":result["failed_paths"]}
    with (output/"MANIFEST.json").open("x",encoding="utf-8") as handle:json.dump(manifest,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write("\n")
    print(json.dumps({"status":"COMPLETE","manifest":str(output/"MANIFEST.json")}));return manifest


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("result","audit","output"):parser.add_argument("--"+name,required=True)
    args=parser.parse_args();generate(args.result,args.audit,args.output)
