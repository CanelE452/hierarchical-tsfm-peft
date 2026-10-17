"""CreateNew reports from a future remaining-family audit and pinned references.

Default0 I/O. No training, array, statistic, trial or model interpretation.
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys

NAME="priority_valid_family_20261016_v1"
COMMON="priority_valid_minima_20261016_v1"
INPUT_HELPER_SHA="8bb730a55e0f0f924de0616bf917ca4b47e4117503027cfc4656c6ef502750dd"
OUTPUTS=("REMAINING_FAMILY_RESULTS_v1.csv","REMAINING_FAMILY_KO_v1.md","REMAINING_FAMILY_BINDINGS_v1.json")
CONDITIONS=(("jena","postT_cpu_jena_E1_ADAPTS_BOLT_PORT_k6_s92601_family_v1","ADAPTS",6,21,43344,12020,29769,1863),
    ("beijing","postT_cpu_beijing_E1_B4_k11_s92601_family_v1","B4",11,11,20539,12063,29880,1868))
B2_RUN="postT_cpu_jena_E3_B2_k6_s92601_v1"
B2_BASE="postT_V_only_cpu_lag_20261016_v1/runs/"+B2_RUN+"/BASE_FIT_RECEIPT.json"
REFERENCE_PINS={
 NAME+"/INITIAL_FAMILY_RESULTS_v1.csv":"6e66eb183f784d80c0c66ae551c1f0f93a5f74b6df4900b468fdc87bf956178e",
 NAME+"/INITIAL_FAMILY_BINDINGS_v1.json":"51cccca9e739687c64649503f1480a76c569da9c99ae0185996825bbebe4475b",
 NAME+"/INDEPENDENT_INITIAL_FAMILY_RECOVERY_AUDIT_ROOT_v2.json":"fbeb44d0d6d77d5c51e56d72e333d0a80b95ab48d95411f112bd6d9da9853d3c",
 NAME+"/B2_REFERENCE_COSTS_v1.csv":"e87b84a1639322d6f8cee91f6cf2dc7c048e8e0121101002ffbcf82cdde121fa",
 NAME+"/B2_REFERENCE_REPORT_BINDINGS_v1.json":"a58fdf1b85a4ddaa07942edc7ab1d1e0f22992d718f5a1fdfbb749c8d7403749",
 NAME+"/INDEPENDENT_B2_REFERENCE_AUDIT_ROOT_v1.json":"346ae638277afc669facfcc98cc22f7e01820c79cd3bea60486913cdb700dae8",
 B2_BASE:"0100fa0551f4865cfb047716ed5ab12bea3315612f3d11126856bdb6ebed2acf",
 COMMON+"/CORE_CONTRAST_RESULTS_v1.csv":"b32570de93f624da4f47a5c8ab7c9398b10babcd5553089ee8bf98a514425a67",
 COMMON+"/CORE_CONTRAST_BINDINGS_v1.json":"7515623564621f646ee642dd1370c67d49183983142177e296083bfe13a11a8a",
 COMMON+"/INDEPENDENT_CORE_CONTRAST_AUDIT_ROOT_v1.json":"8452da8d8586eff682565438b66aa5ddc29d35f4688d70fea45a8b84c8e53846"}
REFERENCE_SOURCE_PINS={NAME+"/jena_b2_reference.py":"43affba72ae780154527a75e59725bc97e70ad72070543f725b7741cf948aabc",
 NAME+"/build_initial_family_recovery_v2.py":"ca4958e5b1486afb211a84555b19d3750284f7e625403cc12ab0f843089d4267",
 COMMON+"/build_core_contrast.py":"051497ad62f53cbba93c4fb2096ae634a0e93018476bf2df673402518db687a8"}
FIELDS=("dataset","row_scope","role","run_id","family","K","registered_K","effective_F_width","C","L","H","seed","MSE","MAE","accuracy_device","training_device","CPU_threads","V_origins","observed_labels","accuracy_reused","cost_reused","reference_source","decoder_scope","initialization_scope","lr","lr_scope","search_max_epoch","selected_refit_epoch","training_status","search_updates","search_exposures","refit_updates","refit_exposures","search_seconds","refit_seconds","artifact_sha256","manifest_sha256","F_resident_bytes","F_resident_bytes_scope")+tuple(f"B{batch}_"+key for batch in (1,8) for key in ("median_seconds","p90_seconds","memory_scope","guard_sampled_peak_RSS_bytes","native_F_counters","native_validation_proof_SHA_map","physical_output_and_inverse_dtype","model_load_seconds","cold_load_or_merge_in_whole_timer"))


def sha(path):
    result=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):result.update(block)
    return result.hexdigest()


def number(value):
    result=float(value)
    if not math.isfinite(result) or result<0:raise ValueError("Finite nonnegative original value required")
    return result


def digest(value):return isinstance(value,str) and len(value)==64 and all(c in "0123456789abcdef" for c in value)


def timed(cost,batch):
    rows=[r for r in cost["summaries"] if r.get("phase")=="timed" and r.get("batch_size")==batch and r.get("trials")==20]
    if len(rows)!=1:raise ValueError("Actual ordinary whole20 summary required")
    for key in ("median_seconds","p90_seconds"):number(rows[0][key])
    return rows[0]


def validate_audit(value):
    if (value.get("schema")!="INDEPENDENT_REMAINING_FAMILY_AUDIT_v1" or value.get("status")!="PASS" or value.get("error") is not None
            or value.get("input_SHA_before_after_equal") is not True or value.get("source48_preserved") is not True
            or value.get("whole_MASTER_complete") is not False or value.get("independent_generalization_claimed") is not False
            or len(value.get("rows",[]))!=2 or len(value.get("accuracy",[]))!=2 or len(value.get("cost_conditions",[]))!=4):raise ValueError("Actual complete remaining-two audit PASS required; partial needs separate honest ROOT reporting")
    by_id={r["run_id"]:r for r in value["rows"]}
    if set(by_id)!={r[1] for r in CONDITIONS}:raise ValueError("Only the two remaining original registered recipes")
    for dataset,identifier,family,k,c,observed,refit_origins,search_exposures,search_updates in CONDITIONS:
        row=by_id[identifier];fit=row["fit_receipt"];search=row["search_stage"];refit=row["refit_stage"];epoch=fit.get("selected_epoch")
        if (row.get("dataset"),row.get("family"),row.get("latent_k"),row.get("C"))!=(dataset,family,k,c) or epoch not in (0.,.25,1.,3.):raise ValueError("Dataset/width and actual selected epoch must remain distinct")
        want=dict(status="COMPLETE_VALID",run_id=identifier,family=family,seed=92601,epochs=[.25,1.,3.],search_exposures=search_exposures,search_updates=search_updates,
            refit_exposures=math.ceil(epoch*refit_origins),refit_updates=math.ceil(math.ceil(epoch*refit_origins)/16),protected_T_target_operations=0,probe_weights_optimizer_or_sampler_reused=False,
            training_status="UNDERTRAINED_OR_BUDGET_LIMIT" if epoch==3 else "SCREEN_COMPLETE_CONVERGENCE_UNRESOLVED")
        if any(fit.get(key)!=v or row.get(key)!=v for key,v in want.items()):raise ValueError("Preserve exact raw BASE activity/selected length/T/probe provenance")
        for stage,ends,label in ((search,[0.,.25,1.,3.],"search"),(refit,[0.]+([epoch] if epoch else []),"refit")):
            if stage.get("selected_epoch")!=epoch or stage.get("completed_epochs")!=ends or any(stage.get(key)!=fit[label+"_"+key] for key in ("updates","exposures")):raise ValueError("Actual selected search/refit endpoints and activity disagree")
            number(stage["seconds"])
        for key in ("mse","mae"):number(row[key])
        if not all(digest(row.get(key)) for key in ("artifact_sha256","manifest_sha256")):raise ValueError("Actual model/manifest SHA required")
        for batch in (1,8):
            cost=row[f"cost_B{batch}"];raw=cost["receipt"];timed(cost,batch)
            expected=dict(status="COMPLETE_PROTOCOL",device="cpu",CPU_threads=2,batch_size=batch,warmups=10,timed_trials=20,stage_trials=20,actual_trial_rows=50,
                F_counters=dict(streams=50*batch*k,physical_streams=50*batch*k,physical_calls=50),physical_output_and_inverse_dtype="float64",cached_Z_or_forecast_reads=0,
                training_peak_included=False,LoRA_merge_and_model_loading_in_whole_trial_timer=False,base_restored_after_original_Route_close=True)
            if any(raw.get(key)!=v for key,v in expected.items()) or cost["summaries"]!=raw["summaries"] or cost["memory_scope"]!=raw["memory"]:raise ValueError("Native recorded counters/whole physical CPU2 cost fields must be preserved")
            number(cost["guard_sampled_peak_RSS_bytes"])


def new_rows(audit):
    validate_audit(audit);by_id={r["run_id"]:r for r in audit["rows"]};rows=[]
    for dataset,identifier,family,k,c,n,_,_,_ in CONDITIONS:
        original=by_id[identifier];fit=original["fit_receipt"]
        row=dict(dataset=dataset,row_scope="NEW_REMAINING_FAMILY",role="ADAPTS_BOLT_PORT_K6" if family=="ADAPTS" else "B4_K11",run_id=identifier,family=family,K=k,effective_F_width=k,C=c,L=512,H=48,seed=92601,MSE=original["mse"],MAE=original["mae"],accuracy_device="cpu",training_device="cpu",CPU_threads=2,V_origins=43,observed_labels=n,accuracy_reused=False,cost_reused=False,reference_source="ACTUAL_REMAINING_FAMILY_AUDIT",
            decoder_scope="AFFINE_POINTWISE_NO_LAG_OR_CURRENT_ANCHOR" if family=="ADAPTS" else "TRAINED_LAG7_CURRENT_VALUE_ANCHOR",
            initialization_scope="AdaPTSLinearPort nn.Linear defaults; PCA not applied" if family=="ADAPTS" else "K=C identity basis/center0 plus registered B4 nonlinear parameters",
            lr=.001,lr_scope="ORIGINAL_REGISTERED_RECIPE",search_max_epoch=3,selected_refit_epoch=fit["selected_epoch"],training_status=fit["training_status"],search_seconds=original["search_stage"]["seconds"],refit_seconds=original["refit_stage"]["seconds"],artifact_sha256=original["artifact_sha256"],manifest_sha256=original["manifest_sha256"],F_resident_bytes=None,F_resident_bytes_scope="NOT_SEPARATELY_RECORDED")
        for key in ("search_updates","search_exposures","refit_updates","refit_exposures"):row[key]=fit[key]
        for batch in (1,8):
            cost=original[f"cost_B{batch}"];raw=cost["receipt"];summary=timed(cost,batch)
            for key in ("median_seconds","p90_seconds"):row[f"B{batch}_"+key]=summary[key]
            row.update({f"B{batch}_memory_scope":cost["memory_scope"],f"B{batch}_guard_sampled_peak_RSS_bytes":cost["guard_sampled_peak_RSS_bytes"],f"B{batch}_native_F_counters":raw["F_counters"],f"B{batch}_native_validation_proof_SHA_map":raw.get("validation_result_sha256"),f"B{batch}_physical_output_and_inverse_dtype":raw["physical_output_and_inverse_dtype"],f"B{batch}_model_load_seconds":raw.get("model_load_seconds"),f"B{batch}_cold_load_or_merge_in_whole_timer":raw["LoRA_merge_and_model_loading_in_whole_trial_timer"]})
        row["registered_K"]=k;rows.append(row)
    return rows


def base_reference(dataset,role,identifier,family,k,c,n,source):
    return dict(dataset=dataset,row_scope="REUSED_REFERENCE",role=role,run_id=identifier,family=family,K=0 if family=="LORA" else k,registered_K=0 if family=="LORA" else k,effective_F_width=k,C=c,L=512,H=48,seed=92601,accuracy_device="cpu",training_device="cpu",CPU_threads=2,V_origins=43,observed_labels=n,accuracy_reused=True,cost_reused=True,reference_source=source,search_max_epoch=3,F_resident_bytes=None,F_resident_bytes_scope="NOT_SEPARATELY_RECORDED")


def b2_reference(rows,receipt):
    if len(rows)!=4 or {(r.get("batch_size"),r.get("phase")) for r in rows}!={(str(b),p) for b in (1,8) for p in ("timed","stage_probe")}:raise ValueError("Exactly the reused B2 four summary rows required")
    expected=dict(run_id=B2_RUN,dataset="jena",family="B2",seed="92601",C="21",K="6",L="512",H="48",n_origins="43",n_observed="43344",device="cpu",CPU_threads="2",worker_RSS_limit_bytes="3221225472",accuracy_reused="True",new_training="0",trials="20",F_resident_bytes_scope="NOT_SEPARATELY_RECORDED")
    if any(any(r.get(key)!=v for key,v in expected.items()) or r.get("F_resident_bytes") not in (None,"","NA") for r in rows) or receipt.get("run_id")!=B2_RUN or receipt.get("family")!="B2" or receipt.get("selected_epoch") not in (0.,.25,1.,3.):raise ValueError("Exact existing B2 CPU2/RSS3GiB V43 reference and original selected receipt required")
    if len({(r["normalized_MSE"],r["normalized_MAE"]) for r in rows})!=1:raise ValueError("One unchanged reused B2 accuracy required")
    row=base_reference("jena","B2_K6_REFERENCE",B2_RUN,"B2",6,21,43344,"B2_REFERENCE_COSTS_v1.csv")
    row.update(MSE=number(rows[0]["normalized_MSE"]),MAE=number(rows[0]["normalized_MAE"]),decoder_scope="TRAINED_LAG7_CURRENT_VALUE_ANCHOR",initialization_scope="A-fitted PCA K6 original source",lr=None,lr_scope="NOT_COPIED_FROM_REFERENCE_RECEIPT",selected_refit_epoch=receipt["selected_epoch"],training_status=receipt["training_status"])
    for key in ("search_updates","search_exposures","refit_updates","refit_exposures"):row[key]=receipt[key]
    for batch in (1,8):
        entry=next(r for r in rows if r["batch_size"]==str(batch) and r["phase"]=="timed")
        for key in ("median_seconds","p90_seconds"):row[f"B{batch}_"+key]=number(entry[key])
    return row


def other_references(initial,core):
    result=[];expected={"LORA_R8_C21":("postT_cpu_jena_E1_LORA_R8_k0_s92601_family_v1","LORA",21),"B4_K6":("postT_cpu_jena_E1_B4_k6_s92601_family_v1","B4",6)}
    if len(initial)!=2 or {r.get("role") for r in initial}!=set(expected):raise ValueError("Only original two Jena initial-family reference rows")
    for entry in initial:
        identifier,family,k=expected[entry["role"]]
        if (entry.get("run_id")!=identifier or any(entry.get(key)!=v for key,v in dict(accuracy_device="cpu",training_device="cpu",CPU_threads="2",V_origins="43",observed_labels="43344",effective_F_width=str(k),F_resident_bytes_scope="NOT_SEPARATELY_RECORDED").items()) or entry.get("F_resident_bytes") not in (None,"")):raise ValueError("Jena reference cohort/width differs")
        row=base_reference("jena",entry["role"]+"_REFERENCE",identifier,family,k,21,43344,"INITIAL_FAMILY_RESULTS_v1.csv")
        row.update(decoder_scope="ORIGINAL_C21_LORA_ROUTE" if family=="LORA" else "TRAINED_LAG7_CURRENT_VALUE_ANCHOR",initialization_scope="LoRA original r8 modules" if family=="LORA" else "A-fitted PCA K6 plus registered B4 nonlinear parameters",lr=number(entry["lr"]),lr_scope="ORIGINAL_PINNED_REFERENCE_CSV")
        for key in ("MSE","MAE","selected_refit_epoch","search_updates","refit_updates","search_exposures","refit_exposures","search_seconds","refit_seconds"):row[key]=number(entry[key])
        row["training_status"]=entry["training_status"]
        for batch in (1,8):
            for key in ("median_seconds","p90_seconds"):row[f"B{batch}_"+key]=number(entry[f"B{batch}_"+key])
        result.append(row)
    found=[r for r in core if r.get("role")=="B2_K11" and r.get("run_id")=="postT_cpu_beijing_E1_B2_k11_s92601_required_v1"]
    if len(found)!=1:raise ValueError("Exactly already completed Beijing full-C B2 K11 reference required")
    entry=found[0]
    if any(entry.get(key)!=v for key,v in dict(K="11",accuracy_device="cpu",training_device="cpu",CPU_threads="2",V_origins="43",observed_labels="20539",F_resident_bytes_scope="NOT_SEPARATELY_RECORDED").items()) or entry.get("F_resident_bytes") not in (None,""):raise ValueError("Beijing reference cohort/width differs")
    row=base_reference("beijing","B2_K11_REFERENCE",entry["run_id"],"B2",11,11,20539,"../"+COMMON+"/CORE_CONTRAST_RESULTS_v1.csv")
    row.update(decoder_scope="TRAINED_LAG7_CURRENT_VALUE_ANCHOR",initialization_scope=entry["initialization"],lr=None,lr_scope="NOT_COPIED_FROM_REFERENCE_CSV",training_status=entry["training_status"])
    for key in ("MSE","MAE","selected_refit_epoch","search_updates","refit_updates","search_exposures","refit_exposures","search_seconds","refit_seconds"):row[key]=number(entry[key])
    for batch in (1,8):
        for key in ("median_seconds","p90_seconds"):row[f"B{batch}_"+key]=number(entry[f"B{batch}_"+key])
    return [*result,row]


def cell(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False) if isinstance(value,(dict,list)) else value


def compose(audit,references,audit_relative):
    fresh=new_rows(audit);rows=[]
    for dataset in ("jena","beijing"):rows.extend([r for r in references if r["dataset"]==dataset]+[r for r in fresh if r["dataset"]==dataset])
    out=io.StringIO(newline="");writer=csv.DictWriter(out,fieldnames=FIELDS,lineterminator="\n");writer.writeheader();writer.writerows({key:cell(row.get(key)) for key in FIELDS} for row in rows)
    tables=[]
    for dataset,title,n in (("jena","Jena C21",43344),("beijing","Beijing C11",20539)):
        values="\n".join(f"{r['role']:<27} {r['effective_F_width']:2} {float(r['MSE']):.17g} {float(r['MAE']):.17g} {float(r['B1_median_seconds']):.17g} {float(r['B8_median_seconds']):.17g} {r['selected_refit_epoch']} {r['accuracy_reused']}" for r in rows if r["dataset"]==dataset)
        tables.append(f"{title} / V43 / observed labels{n}\n\n```text\nrole                        F-width MSE                 MAE                 B1sec/batch B8sec/batch selected-refit-epoch reused\n{values}\n```\n")
    md=("# 남은 원 방법군: Jena AdaPTS port와 Beijing B4\n\n[확인] 원 recipe 두 조건의 실제 학습·전체 V·FP64 재채점·fresh CPU6·B1/B8 검산이 완료된 감사에서만 이 보고를 만든다. 두 자료의 scaler/mask/origin/관측 라벨은 다르다. **자료 간 MSE 순위**를 만들지 않으며 각 자료 안에서만 원수치를 비교한다.\n\n"
        "[원정밀도 CSV](REMAINING_FAMILY_RESULTS_v1.csv) · [보고 결속](REMAINING_FAMILY_BINDINGS_v1.json) · [실제 독립 감사]("+audit_relative+")\n\n"+"\n".join(tables)+"\n"
        "새 결과는 Jena ADAPTS_BOLT_PORT K6와 Beijing B4 K11 두 행이다. Jena B2 K6/B4 K6/LoRA C21 및 Beijing B2 K11은 pinned 기존 결과 재사용이며 새 학습·추론·비용이나 독립 일반화로 세지 않는다. 참고의 상세 provenance는 [기존 Jena 초기 방법군](INITIAL_FAMILY_KO_v1.md), [B2 정확도/비용 검증](B2_REFERENCE_VALIDATION_KO_v1.md), [Beijing core 보고](../"+COMMON+"/CORE_CONTRAST_KO_v1.md)에 남아 있다.\n\n"
        "모든 행은 L512/H48·seed92601·CPU2·반복 개발 V43의 normalized channel-macro MSE/MAE다. Jena는43344, Beijing은20539 관측 라벨이며 같은 V를 반복 사용한 것이므로 독립 표본 수가 증가하지 않는다. 새 BASE의 actual selected refit0/.25/1/3, update/exposure와 stage wall을 CSV/결속에 그대로 보존한다. 선택을3으로 가정하지 않는다. search는 원 grid0/.25/1/3 endpoint이고 선택3이면 UNDERTRAINED_OR_BUDGET_LIMIT, 더 짧은 선택도 수렴을 입증하지 않는다. 제한 학습·한 seed·반복 개발 V에서 충분 학습/독립 일반화는 미확정이다.\n\n"
        "AdaPTS는 저장소의 LinearAutoEncoder/MultichannelProjector 원리를 affine pointwise encoder/decoder·RevIN=False/patch1로 옮긴 **AdaPTSLinearPort**다. 이 저장소의 macro loss·시간 분할·미분 가능한 frozen Chronos-Bolt를 사용하므로 원 논문의 모든 설정·모델·재학습·수치를 완전 재현한 결과가 아니다. PCA 초기화나 lag7/current anchor가 없는 nn.Linear 경로이며 B2/B4의 학습 lag7 D/current anchor와 달라 encoder 하나의 효과 대조가 아니다. Beijing B4 K=C는 identity basis/center0에서 시작하는 등록 nonlinear 경로이고 B2 full-C도 identity basis이지만 모수·구조는 다르다. Jena B2/B4 K6 PCA 초기화, LoRA r8와effective C21/F 적응·lr1e-4, 새 B4/AdaPTS lr1e-3의 차이를 동등 compute/동일 표현 비교로 바꾸지 않는다.\n\n"
        "비용은 physical history/mask→채움·정규화→실제 K개 F 입력→등록 decoder→원 C 미래·FP64 inverse/CPU 반환의 한 batch 전체다. warm10·ordinary timed20·별도 intrusive stage20을 분리하고 ordinary median/p90를 초/batch로 보존한다. B8은8개 입력 한 batch 전체이며8로 나누지 않는다. cold F pool/Route load·LoRA merge는 warm whole 밖이고 model_load는 원 기록 범위만이다. Route memory dict·guard tree RSS·별도 측정하지 않은 F-only resident NA를 구분한다. native family receipt가 실제 기록한 F counter/proofmap/float64 dtype는 그대로 보존하며 N7의 미기록 NA 계약과 혼동하지 않는다.\n\n"
        "기존 Jena B2의 원 학습/정규화 ancestry와4GiB guard를 새3GiB 학습으로 소급 바꾸지 않는다. 새 B2 reference 비용은 별도 CPU2/RSS3GiB quiet 검증 결과다. 원 초기 POST mixed-slash PARTIAL·recovery 기록과 모든 기존 산출물을 그대로 둔다. 이 보고는 기존 결과 파일·숫자·MD를 고치지 않는다. 새로운 두 조건의 성능이 좋거나 나쁨은 완성 보고 gate가 아니며 partial/검산 실패는 ROOT가 성공·미실행·저장점·재개 조건을 별도 정직하게 보고해야 한다.\n\n"
        "전체 E1/E4/56/MASTER 완료가 아니다. N7 실제 종료·보고/main 및 원 잔여 예산 확인 후 별도 입장한 경로만 보고할 수 있다. 보호 T 새 수치 파싱·평가·선택·기존 중단 실험 재개·예산 확대·새 방법 추가는 하지 않는다. 이 builder의 새 모델/Task/NPZ/CP·통계/trial 재집계·학습/추론/비용·프로세스 제어·Git/network는0이다.\n")
    return {OUTPUTS[0]:out.getvalue(),OUTPUTS[1]:md}


def load_references(inputs):
    for name,want in {**REFERENCE_PINS,**REFERENCE_SOURCE_PINS}.items():inputs.capture("public",name,want)
    specs=((NAME+"/INITIAL_FAMILY_BINDINGS_v1.json","INITIAL_FAMILY_REPORT_BINDINGS_v1","FROM_ACTUAL_COMPLETE_INITIAL_FAMILY_RECOVERY_AUDIT",NAME+"/INDEPENDENT_INITIAL_FAMILY_RECOVERY_AUDIT_ROOT_v2.json",NAME+"/INITIAL_FAMILY_RESULTS_v1.csv"),
        (NAME+"/B2_REFERENCE_REPORT_BINDINGS_v1.json","EXISTING_JENA_B2_REFERENCE_REPORT_BINDINGS_v1","REPORT_FROM_ACTUAL_AUDIT_PASS",NAME+"/INDEPENDENT_B2_REFERENCE_AUDIT_ROOT_v1.json",NAME+"/B2_REFERENCE_COSTS_v1.csv"),
        (COMMON+"/CORE_CONTRAST_BINDINGS_v1.json","BEIJING_CORE_CONTRAST_REPORT_BINDINGS_v1","FROM_ACTUAL_COMPLETE_CORE_AUDIT",COMMON+"/INDEPENDENT_CORE_CONTRAST_AUDIT_ROOT_v1.json",COMMON+"/CORE_CONTRAST_RESULTS_v1.csv"))
    for name,schema,status,audit,csv_name in specs:
        value=inputs.json("public",name)
        if (value.get("schema")!=schema or value.get("status")!=status or value.get("audit_sha256")!=REFERENCE_PINS[audit] or value.get("input_SHA_before_after_equal") is not True or value.get("whole_MASTER_complete") is not False
                or len([r for r in value["outputs"] if r["relative_path"]==csv_name and r["sha256"]==REFERENCE_PINS[csv_name]])!=1):raise ValueError("Exact completed reference output bindings required; no reference re-audit")
    b2=b2_reference(inputs.csv("public",NAME+"/B2_REFERENCE_COSTS_v1.csv"),inputs.json("public",B2_BASE))
    return [b2,*other_references(inputs.csv("public",NAME+"/INITIAL_FAMILY_RESULTS_v1.csv"),inputs.csv("public",COMMON+"/CORE_CONTRAST_RESULTS_v1.csv"))]


def execute(public,cache,audit_path,audit_sha,auditor_sha,builder_sha):
    public,cache=Path(public).resolve(),Path(cache).resolve();scope=public/NAME;audit_path=Path(audit_path).resolve()
    if any((scope/name).exists() for name in OUTPUTS):raise FileExistsError("Existing remaining-family outputs preserved before reading inputs")
    if not audit_path.is_relative_to(scope) or audit_path.suffix!=".json":raise ValueError("Actual public N5 remaining audit required")
    if Path(__file__).resolve()!=scope/"build_remaining_family_report_v2.py" or sha(__file__)!=builder_sha or sha(scope/"audit_remaining_family.py")!=auditor_sha or sha(audit_path)!=audit_sha:raise ValueError("ROOT actual audit and current reviewed producers must be SHA-bound")
    helper_path=public/COMMON/"build_beijing_report.py"
    if sha(helper_path)!=INPUT_HELPER_SHA:raise ValueError("Frozen stdlib input helper changed")
    spec=importlib.util.spec_from_file_location("remaining_report_inputs",helper_path);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);inputs=helper.Inputs(public,cache)
    for name,want in ((NAME+"/build_remaining_family_report_v2.py",builder_sha),(NAME+"/audit_remaining_family.py",auditor_sha),(COMMON+"/build_beijing_report.py",INPUT_HELPER_SHA)):inputs.capture("public",name,want)
    relative=audit_path.relative_to(public).as_posix();audit=inputs.json("public",relative,audit_sha);validate_audit(audit)
    if audit.get("audit_source_sha256")!=auditor_sha:raise ValueError("Actual remaining auditor source differs")
    for row in audit["inputs"]:inputs.capture(row["root"],row["relative_path"],row["sha256"])
    references=load_references(inputs);texts=compose(audit,references,audit_path.relative_to(scope).as_posix())
    if not inputs.preserved():raise ValueError("Captured new audit and reused references changed")
    binding=dict(schema="REMAINING_FAMILY_REPORT_BINDINGS_v1",status="FROM_ACTUAL_COMPLETE_REMAINING_FAMILY_AUDIT",audit_relative_path=relative,audit_sha256=audit_sha,auditor_source_sha256=auditor_sha,builder_source_sha256=builder_sha,
        input_SHA_before_after_equal=True,source48_preserved=True,inputs=[dict(root=root,relative_path=name,sha256=want) for (root,name),want in sorted(inputs.files.items())],outputs=[dict(relative_path=NAME+"/"+name,sha256=hashlib.sha256(text.encode()).hexdigest()) for name,text in texts.items()],
        new_accuracy_conditions=2,new_cost_conditions=4,reused_Jena_reference_rows=3,reused_Beijing_reference_rows=1,references=references,new_training_receipts={r["run_id"]:r["fit_receipt"] for r in audit["rows"]},
        comparison_scope="WITHIN_EACH_DATASET_ONLY",cross_dataset_MSE_ranking=False,reference_audits_rerun=False,whole_MASTER_complete=False,independent_generalization_claimed=False,new_model_array_statistic_trial_training_inference_cost_process_Git_or_network_operations=0)
    texts[OUTPUTS[2]]=json.dumps(binding,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    for name,text in texts.items():
        with (scope/name).open("x",encoding="utf-8",newline="\n") as stream:stream.write(text)
    return dict(status="REMAINING_REPORT_CREATED_FROM_ACTUAL_AUDIT_PASS",files_written=3,new_conditions=2,reused_reference_rows=4)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--execute",action="store_true");parser.add_argument("--root-triggered",action="store_true")
    for name in ("public-root","cache-root","audit-path","audit-sha","auditor-sha","builder-sha"):parser.add_argument("--"+name)
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="PREPARATION_ONLY_NOT_EXECUTED",input_reads=0,files_written=0,numpy_imported="numpy" in sys.modules,torch_imported="torch" in sys.modules,future_actual_audit_SHA=None,future_actual_admission_SHA=None,actual_renderer=0)));return 0
    if not args.root_triggered or any(getattr(args,key) is None for key in ("public_root","cache_root","audit_path","audit_sha","auditor_sha","builder_sha")):parser.error("ROOT actual completed audit and reviewed current producer SHAs with fresh outputs required")
    print(json.dumps(execute(args.public_root,args.cache_root,args.audit_path,args.audit_sha,args.auditor_sha,args.builder_sha)));return 0


if __name__=="__main__":raise SystemExit(main())
