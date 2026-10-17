"""CreateNew multi-K report from exact actual audit PASS and pinned references.

Default0 I/O. No model/NPZ/checkpoint/statistic/trial decoding or experiments.
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

NAME="priority_multik_resume_20261016_v1"
COMMON="priority_valid_minima_20261016_v1"
INPUT_HELPER_SHA="8bb730a55e0f0f924de0616bf917ca4b47e4117503027cfc4656c6ef502750dd"
OUTPUTS=("MULTIK_RESUME_RESULTS_v1.csv","MULTIK_RESUME_KO_v1.md","MULTIK_RESUME_BINDINGS_v1.json")
REFERENCE_PINS={
 "priority_nested_cost_20261016_v1/RESULTS_WITH_COST_v1.csv":"e3d35a0ae6dbac212a4e5b7cfa589f71168ee1b990cbd0916b57e10d8a22a4c9",
 "priority_nested_cost_20261016_v1/REPORT_INPUT_BINDINGS_v1.json":"2590c93f9ae0d4520a03c2e5dbf11818df5dca377ce1f3787e434e7d4ed8d40b",
 "priority_nested_cost_20261016_v1/INDEPENDENT_COST_REVIEW_ROOT_v1.json":"2d00f17d0520d6cd9482d304f4b403b5cf7da613d5faec9a9cd92483ac090c5b",
 "priority_valid_minima_20261016_v1/CORE_CONTRAST_RESULTS_v1.csv":"b32570de93f624da4f47a5c8ab7c9398b10babcd5553089ee8bf98a514425a67",
 "priority_valid_minima_20261016_v1/CORE_CONTRAST_BINDINGS_v1.json":"7515623564621f646ee642dd1370c67d49183983142177e296083bfe13a11a8a",
 "priority_valid_minima_20261016_v1/INDEPENDENT_CORE_CONTRAST_AUDIT_ROOT_v1.json":"8452da8d8586eff682565438b66aa5ddc29d35f4688d70fea45a8b84c8e53846"}
FIELDS=("row_scope","role","run_id","parent_run_id","K","MSE","MAE","accuracy_reused","cost_reused","reference_source","decoder_scope","CPU_threads","V_origins","observed_labels","artifact_sha256","manifest_sha256","training_status","search_restore_call_seconds","historical_saved_search_segment_seconds","refit_seconds_this_resume_call","search_updates","search_exposures","refit_updates","refit_exposures","old_unsaved_interrupted_progress","stage_F_counters","F_resident_bytes","F_resident_bytes_scope")+tuple(f"B{batch}_"+field for batch in (1,8) for field in ("median_seconds","p90_seconds","guard_sampled_peak_RSS_bytes","memory_scope","raw_receipt_total_F_counters","receipt_validation_proof_SHA_map","physical_output_dtype_from_receipt","raw_receipt_field_scope","derived_trial_F_logical_streams","derived_trial_F_physical_streams","derived_trial_F_calls","derived_trial_F_scope"))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1048576),b""):h.update(chunk)
    return h.hexdigest()


def finite(value):
    result=float(value)
    if not math.isfinite(result) or result<0:raise ValueError("Finite nonnegative actual value required")
    return result


def expected_aliases():
    rows={}
    for kind,role in (("shared","ORDERED_MULTI_K"),("max_only","MAXK_PCA_TRAINED")):
        parent=f"postT_cpu_beijing_E4_{role}_k11_s92601_v1"
        for k in (3,6,11):
            tag=role if k==11 else ("SHARED_PREFIX" if kind=="shared" else "POSTHOC_PREFIX")
            rows[f"postT_cpu_beijing_E4_{tag}_k{k}_s92601_v1"]=(parent,kind,k,"shared" if kind=="shared" else ("max_only" if k==11 else "posthoc"))
    return rows


def timed_summary(cost,batch):
    found=[s for s in cost.get("summaries",[]) if s.get("phase")=="timed" and s.get("batch_size")==batch and s.get("trials")==20]
    if len(found)!=1:raise ValueError("Exactly one ordinary timed20 summary per actual cost required")
    for key in ("median_seconds","p90_seconds"):finite(found[0][key])
    return found[0]


def validate_audit(value):
    if (value.get("schema")!="INDEPENDENT_MULTIK_RESUME_AUDIT_v1" or value.get("status")!="PASS" or value.get("error") is not None
            or value.get("input_SHA_before_after_equal") is not True or value.get("source48_preserved") is not True
            or value.get("whole_MASTER_complete") is not False or value.get("independent_generalization_claimed") is not False):raise ValueError("Actual complete original multi-K audit PASS required; partial needs separate honest ROOT reporting")
    expected=expected_aliases();parents=value.get("parents",[]);rows=value.get("rows",[])
    if len(parents)!=2 or {p.get("run_id") for p in parents}!={x[0] for x in expected.values()} or len(rows)!=6 or {r.get("run_id") for r in rows}!=set(expected) or len(value.get("accuracy",[]))!=6 or len(value.get("cost_conditions",[]))!=12:raise ValueError("Two parents/six aliases/twelve costs must remain distinct")
    for parent in parents:
        wanted=dict(status="COMPLETE_VALID",selected_epoch=3,training_status="UNDERTRAINED_OR_BUDGET_LIMIT",search_updates=1868,search_exposures=29880,refit_updates=2262,refit_exposures=36189,search_new_optimizer_updates=0,search_new_F_calls=0,full_V_completed=False,old_unsaved_interrupted_progress="UNKNOWN_NOT_LOGGED")
        if any(parent.get(k)!=v for k,v in wanted.items()):raise ValueError("Original saved search/no-new-updates/refit3/no-V BASE provenance required")
        for key in ("search_seconds_this_resume_call","historical_saved_search_segment_seconds","refit_seconds_this_resume_call"):finite(parent[key])
    for row in rows:
        want=expected[row["run_id"]]
        if (row.get("parent_run_id"),row.get("kind"),row.get("latent_k"),row.get("variant"))!=want:raise ValueError("One exact registered prefix/parent/K path required")
        for key in ("mse","mae"):finite(row[key])
        for key in ("artifact_sha256","manifest_sha256"):
            v=row.get(key)
            if not isinstance(v,str) or len(v)!=64 or any(c not in "0123456789abcdef" for c in v):raise ValueError("Actual selected parent/manifest SHA required")
        for batch in (1,8):
            cost=row.get(f"cost_B{batch}",{});timed_summary(cost,batch)
            if (any(cost.get(key) is not None for key in ("raw_receipt_total_F_counters","receipt_validation_proof_SHA_map","physical_output_dtype_from_receipt")) or cost.get("raw_receipt_field_scope")!="NOT_RECORDED_BY_ORIGINAL_N7_RECEIPT"):raise ValueError("Unlogged native receipt fields must stay null, not inferred totals or dtype")
            derived=cost.get("derived_trial_F_counters",{});k=row["latent_k"]
            if derived!=dict(F_logical_streams=50*batch*k,F_physical_streams=50*batch*k,F_calls=50,scope="INDEPENDENTLY_SUMMED_FROM_50_PUBLIC_TRIAL_ROWS"):raise ValueError("Independent actual50 trial counters differ from native missing-field scope")
            finite(cost["guard_sampled_peak_RSS_bytes"])


def new_rows(audit):
    parents={p["run_id"]:p for p in audit["parents"]};result=[]
    for r in audit["rows"]:
        p=parents[r["parent_run_id"]]
        row=dict(row_scope="NEW_TRAINED_PARENT_PREFIX",role="SHARED_MULTI_K" if r["kind"]=="shared" else "MAX_ONLY_POSTHOC",run_id=r["run_id"],parent_run_id=r["parent_run_id"],K=r["latent_k"],MSE=r["mse"],MAE=r["mae"],accuracy_reused=False,cost_reused=False,reference_source="ACTUAL_CURRENT_MULTIK_AUDIT",decoder_scope="TRAINED_LAG7_CURRENT_VALUE_ANCHOR",CPU_threads=2,V_origins=43,observed_labels=20539,artifact_sha256=r["artifact_sha256"],manifest_sha256=r["manifest_sha256"],training_status=p["training_status"],search_restore_call_seconds=p["search_seconds_this_resume_call"],historical_saved_search_segment_seconds=p["historical_saved_search_segment_seconds"],refit_seconds_this_resume_call=p["refit_seconds_this_resume_call"],old_unsaved_interrupted_progress=p["old_unsaved_interrupted_progress"],stage_F_counters=p.get("stage_F_counters"),F_resident_bytes=None,F_resident_bytes_scope="NOT_SEPARATELY_RECORDED")
        for key in ("search_updates","search_exposures","refit_updates","refit_exposures"):row[key]=p[key]
        for batch in (1,8):
            cost=r[f"cost_B{batch}"];timed=timed_summary(cost,batch);prefix=f"B{batch}_"
            for key in ("median_seconds","p90_seconds"):row[prefix+key]=timed[key]
            for key in ("guard_sampled_peak_RSS_bytes","memory_scope","raw_receipt_total_F_counters","receipt_validation_proof_SHA_map","physical_output_dtype_from_receipt","raw_receipt_field_scope"):row[prefix+key]=cost.get(key)
            for key in ("F_logical_streams","F_physical_streams","F_calls","scope"):row[prefix+"derived_trial_"+key]=cost["derived_trial_F_counters"][key]
        result.append(row)
    return result


def reference_rows(nested,core):
    if len(nested)!=3 or {int(r["K"]) for r in nested}!={3,6,11}:raise ValueError("Exactly the three reused fixed nested-PCA rows required")
    result=[]
    for r in nested:
        k=int(r["K"])
        if (r.get("run_id")!=f"postT_cpu_beijing_E4_NESTED_PCA_k{k}_s92601_v1" or r.get("accuracy_device")!="cpu" or int(r["CPU_threads"])!=2 or int(r["V_origins"])!=43 or int(r["observed_labels"])!=20539 or int(r["C"])!=11 or int(r["L"])!=512 or int(r["H"])!=48):raise ValueError("Same CPU2 Beijing original V43 fixed-PCA reference required")
        row=dict(row_scope="REUSED_FIXED_NESTED_PCA",role="NESTED_PCA_REFERENCE",run_id=r["run_id"],parent_run_id="FIXED_A60_PCA_BASIS",K=k,MSE=finite(r["MSE"]),MAE=finite(r["MAE"]),accuracy_reused=True,cost_reused=True,reference_source="../priority_nested_cost_20261016_v1/RESULTS_WITH_COST_v1.csv",decoder_scope="FIXED_PCA_BASIS_TRANSPOSE_NO_TRAINED_LAG_D",CPU_threads=2,V_origins=43,observed_labels=20539,training_status="NOT_APPLICABLE_FIXED_TRANSFORM",F_resident_bytes=None,F_resident_bytes_scope="NOT_SEPARATELY_RECORDED")
        for batch in (1,8):
            for field in ("median_seconds","p90_seconds"):row[f"B{batch}_"+field]=finite(r[f"B{batch}_"+field])
        result.append(row)
    found=[r for r in core if r.get("role")=="B0_K6" and r.get("run_id")=="postT_cpu_beijing_E4_INDEPENDENT_B0_k6_s92601_v1"]
    if len(found)!=1:raise ValueError("Only the already completed independent B0 K6 reference may be reused")
    r=found[0]
    if int(r["K"])!=6 or r.get("accuracy_device")!="cpu" or r.get("training_device")!="cpu" or int(r["CPU_threads"])!=2 or int(r["V_origins"])!=43 or int(r["observed_labels"])!=20539:raise ValueError("Same CPU2 original V43 independent K6 reference required")
    row=dict(row_scope="REUSED_INDEPENDENT_K6",role="INDEPENDENT_K6_REFERENCE",run_id=r["run_id"],parent_run_id=r["run_id"],K=6,MSE=finite(r["MSE"]),MAE=finite(r["MAE"]),accuracy_reused=True,cost_reused=True,reference_source="../priority_valid_minima_20261016_v1/CORE_CONTRAST_RESULTS_v1.csv",decoder_scope="TRAINED_B0_LAG7_CURRENT_VALUE_ANCHOR",CPU_threads=2,V_origins=43,observed_labels=20539,training_status=r["training_status"],F_resident_bytes=None,F_resident_bytes_scope="NOT_SEPARATELY_RECORDED")
    for batch in (1,8):
        for field in ("median_seconds","p90_seconds"):row[f"B{batch}_"+field]=finite(r[f"B{batch}_"+field])
    return [*result,row]


def cell(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False) if isinstance(value,(dict,list)) else value


def compose(audit,references,audit_relative):
    validate_audit(audit);fresh=new_rows(audit);rows=[*fresh,*references]
    stream=io.StringIO(newline="");writer=csv.DictWriter(stream,fieldnames=FIELDS,lineterminator="\n");writer.writeheader();writer.writerows({key:cell(r.get(key)) for key in FIELDS} for r in rows)
    values="\n".join(f"{r['role']:<25} {r['K']:2} {float(r['MSE']):.17g} {float(r['MAE']):.17g} {float(r['B1_median_seconds']):.17g} {float(r['B8_median_seconds']):.17g} {r['accuracy_reused']}" for r in rows)
    training="\n".join(f"{p['kind']:<8} {p['search_seconds_this_resume_call']:.17g} {p['historical_saved_search_segment_seconds']:.17g} {p['refit_seconds_this_resume_call']:.17g} {p['search_updates']}/{p['refit_updates']} {p['search_exposures']}/{p['refit_exposures']} {p['old_unsaved_interrupted_progress']}" for p in audit["parents"])
    md=f"""# Beijing: 원 shared/max 부모 재개와 실제 여러 K 출력

[확인] 이 보고는 원 shared `[3,6,11]` 및 max-only `[11]` **두 부모**의 마지막 정상 refit CP0 재개와 K3/6/11 **여섯 출력**을 구분한다. 여섯 출력을 새 독립 학습6개로 세지 않는다. 아래는 같은 C11/L512/H48·seed92601·CPU2·반복 개발 V43/20539 관측 라벨의 normalized channel-macro MSE·MAE와 실제 warm whole batch 비용이다. 원 숫자는 다음 CSV에 보존한다.

[원정밀도 CSV](MULTIK_RESUME_RESULTS_v1.csv) · [입력/출력 SHA](MULTIK_RESUME_BINDINGS_v1.json) · [실제 독립 감사]({audit_relative}) · [원 재개 계약](README_PREPARATION_KO.md)

```text
role                       K MSE                 MAE                 B1 median sec/batch B8 median sec/batch accuracy-reused
{values}
```

재사용 참고는 [고정 nested PCA3](../priority_nested_cost_20261016_v1/RESULTS_WITH_COST_KO_v1.md)와 [기존 독립 B0 K6](../priority_valid_minima_20261016_v1/CORE_CONTRAST_KO_v1.md)다. 새 감사의 rows6과 재사용 참고행을 분리하며, 이 builder가 참고의 모델/통계/비용을 다시 실행한 것은 아니다. 고정 PCA는 basis transpose 복원이고 학습 가능한 lag7 D·현재값 anchor와 다르므로 압축 학습만의 효과 대조가 아니다. 독립 K3/K11 새 학습·fullV·비용은 이번 완료에 포함되지 않는다.

첫 질문은 같은 한 부모의 실제 K 입력 전환에서 오차와 전체 비용이 어떻게 달라졌는지다. 둘째는 shared와 max-only 사후 prefix가 이 조건에서 어떻게 달랐는지다. 두 부모는 같은 A-only full-C PCA 초기화지만 K별 감독·노출·F 연산 배분이 다르다. 동일3 epoch·optimizer update가 동일 K별 노출·F 연산·총 학습 시간까지 보장하지 않는다. 총 예산을 맞춘 추가 비교와 K별 독립 모델의 총비용·전환 비용을 이번 표가 완료시키지는 않는다.

shared/max K11은 grid가 있어 PCA 초기화이고 원 independent K11은 grid 없는 identity/center0 초기화다. 해당 K11 간 차이를 공유 학습만의 순수 효과로 단정할 수 없다. K3/6의 PCA E·center는 같은 A-stage 전체 고유분해의 prefix 규칙이지만 최종 학습 가중치·D·용량과 노출은 다르다. 현재 비교는 같은 최종 표현에 predictor만 바꾼 실험이 아니다.

```text
parent   search-restore-call-sec historical-saved-search-segment-sec current-refit-sec search/refit-updates search/refit-exposures unsaved-progress
{training}
```

search restore 호출 시간은 이번 재구축/복원 overhead이며 **과거 search 전체 wall**이 아니다. saved segment 시간은 과거 저장된 segment 기록 범위만 뜻한다. 원 search0/.25/1/3의1868 update·29880 exposure는 재사용 이력이며 새 search update/F call은0이다. 원 refit CP0에서3 epoch의2262 update·36189 exposure를 수행했다. 저장 전 손실 진행량은 UNKNOWN_NOT_LOGGED다. 원 stage별 session/total F counter는 입력 감사의 parent `stage_F_counters` 및 CSV JSON열에 있는 값만 보존하고, 누락값을 추정 합산하지 않는다. sampler·Torch CPU RNG 복원과 Python/NumPy 전역 RNG·중단 없는 bitwise parity는 구분한다.

부모 BASE의 `full_V_completed=false`는 그대로 보존한다. alias별 실제 predict/manifest·전체 저장 V FP64·fresh CPU6(4+2)로 완료를 판단하며 BASE를 true로 고치지 않는다. full V 검산은 저장 예측의 재채점이고 checkpoint 재추론은6 origin이다. 같은 V를 여러 K·방법에서 반복 사용했으며 독립 표본 수를 늘리지 않는다. search 마지막3 epoch 선택 상태는 UNDERTRAINED_OR_BUDGET_LIMIT이며 충분 수렴과 독립 일반화는 미확정이다.

각 B1/B8 비용은 물리 history/mask→채움·정규화→실제 K개 F 입력→lag D/현재값 anchor→원 C 미래·FP64 inverse/CPU 반환의 **한 batch 전체**다. B8을8로 나누거나 K/C를 속도비로 쓰지 않는다. warm10·ordinary timed20·별도 intrusive stage20을 분리하며 CSV에는 ordinary median/p90와 관측 guard tree RSS를 남긴다. stage 합=whole을 주장하지 않는다. 원 F pool cold load와 Route loading은 warm whole 밖이다. prefix K가 작아져도 부모 E/D는 최대 K11 구조이므로 모델 resident memory가 K배 줄었다고 하지 않는다. Route memory dict와 guard RSS는 다른 범위이며 F-only resident bytes는 미측정 NA다.

**원 N7 COST receipt에는 전체 F counter 합계·validation proof SHA 사전·physical output dtype 필드가 없다.** 해당 CSV열은 비워두고 NOT_RECORDED_BY_ORIGINAL_N7_RECEIPT로 표시한다. 실제 argv/proof 결속과 공개50 trial에서 독립 합산한 counter는 별도 근거다. 그 합계를 원 receipt에 기록된 값으로 바꾸지 않는다. 비용 측정과 학습을 겹치지 않았다는 실제 quiet/guard 증명 및 종료는 원 감사 범위로 보존한다.

전체 E4/56/Master 완료가 아니다. Electricity 파일럿·독립 K3/K11·다른 자료/seed·충분 학습·동등 총 예산·저장량/전환 비용 등 원 잔여는 남는다. 고정 nested PCA와 독립 K6 재사용은 새 원리 학습/새 일반화가 아니다. 보호 T 새 수치 파싱·평가·선택·원 큰 큐 자동 재시작·새 모듈 추가를 수행하지 않았다. 부분 종료나 증거 누락이면 이 완성 builder는 거부하며 ROOT는 실패·미실행·저장점·재개 조건을 별도 정직한 부분 보고로 남겨야 한다.
"""
    return {OUTPUTS[0]:stream.getvalue(),OUTPUTS[1]:md}


def load_input_helper(public):
    path=public/COMMON/"build_beijing_report.py"
    if sha(path)!=INPUT_HELPER_SHA:raise ValueError("Frozen stdlib input helper changed")
    spec=importlib.util.spec_from_file_location("multik_report_inputs",path);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);return helper


def load_references(inputs):
    for path,want in REFERENCE_PINS.items():inputs.capture("public",path,want)
    nested_audit=inputs.json("public","priority_nested_cost_20261016_v1/INDEPENDENT_COST_REVIEW_ROOT_v1.json")
    nested_binding=inputs.json("public","priority_nested_cost_20261016_v1/REPORT_INPUT_BINDINGS_v1.json")
    core_audit=inputs.json("public",COMMON+"/INDEPENDENT_CORE_CONTRAST_AUDIT_ROOT_v1.json");core_binding=inputs.json("public",COMMON+"/CORE_CONTRAST_BINDINGS_v1.json")
    if nested_audit.get("schema")!="INDEPENDENT_NESTED_COST_REVIEW_v1" or nested_audit.get("status")!="PASS" or nested_binding.get("input_SHA_before_after_equal") is not True or nested_binding.get("whole_MASTER_complete") is not False:raise ValueError("Pinned fixed nested-PCA actual audit/report bindings required")
    if core_audit.get("schema")!="INDEPENDENT_BEIJING_CORE_CONTRAST_v1" or core_audit.get("status")!="PASS" or core_audit.get("error") is not None or core_binding.get("audit_sha256")!=REFERENCE_PINS[COMMON+"/INDEPENDENT_CORE_CONTRAST_AUDIT_ROOT_v1.json"] or core_binding.get("input_SHA_before_after_equal") is not True:raise ValueError("Pinned existing independent K6 actual audit/report required")
    return reference_rows(inputs.csv("public","priority_nested_cost_20261016_v1/RESULTS_WITH_COST_v1.csv"),inputs.csv("public",COMMON+"/CORE_CONTRAST_RESULTS_v1.csv"))


def execute(public,cache,audit_path,audit_sha,auditor_sha,builder_sha):
    public,cache=Path(public).resolve(),Path(cache).resolve();scope=public/NAME;audit_path=Path(audit_path).resolve()
    if any((scope/name).exists() for name in OUTPUTS):raise FileExistsError("Existing multi-K reports preserved before inputs")
    if not audit_path.is_relative_to(scope) or audit_path.suffix!=".json":raise ValueError("Actual public N7 audit JSON required")
    if Path(__file__).resolve()!=scope/"build_multik_resume_report.py" or sha(__file__)!=builder_sha or sha(scope/"audit_multik_resume.py")!=auditor_sha or sha(audit_path)!=audit_sha:raise ValueError("Explicit actual audit/current producer SHA required")
    helper=load_input_helper(public);inputs=helper.Inputs(public,cache)
    for name,want in (("build_multik_resume_report.py",builder_sha),("audit_multik_resume.py",auditor_sha)):inputs.capture("public",NAME+"/"+name,want)
    inputs.capture("public",COMMON+"/build_beijing_report.py",INPUT_HELPER_SHA)
    audit=inputs.json("public",audit_path.relative_to(public).as_posix(),audit_sha);validate_audit(audit)
    if audit.get("audit_source_sha256")!=auditor_sha:raise ValueError("Actual audit source identity differs")
    for row in audit["inputs"]:inputs.capture(row["root"],row["relative_path"],row["sha256"])
    references=load_references(inputs);texts=compose(audit,references,audit_path.relative_to(scope).as_posix())
    if not inputs.preserved():raise ValueError("Captured report inputs changed")
    binding=dict(schema="MULTIK_RESUME_REPORT_BINDINGS_v1",status="FROM_ACTUAL_MULTIK_AUDIT_PASS_AND_PINNED_REUSED_REFERENCES",audit_relative_path=audit_path.relative_to(public).as_posix(),audit_sha256=audit_sha,auditor_source_sha256=auditor_sha,builder_source_sha256=builder_sha,input_SHA_before_after_equal=True,source48_preserved=True,inputs=[dict(root=root,relative_path=name,sha256=want) for (root,name),want in sorted(inputs.files.items())],outputs=[dict(relative_path=NAME+"/"+name,sha256=hashlib.sha256(text.encode()).hexdigest()) for name,text in texts.items()],resumed_parent_refits=2,new_registered_conditions=0,new_prefix_accuracy_outputs=6,new_cost_conditions=12,reused_fixed_nested_PCA_rows=3,reused_independent_K6_rows=1,reused_reference_pins=REFERENCE_PINS,parents=audit["parents"],references=references,unlogged_native_cost_fields_preserved_null=True,whole_E4_complete=False,whole_MASTER_complete=False,independent_generalization_claimed=False,new_model_NPZ_CP_statistics_trial_training_inference_cost_or_process_operations=0)
    texts[OUTPUTS[2]]=json.dumps(binding,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    for name,text in texts.items():
        with (scope/name).open("x",encoding="utf-8",newline="\n") as stream:stream.write(text)
    return dict(status="MULTIK_REPORT_CREATED_FROM_ACTUAL_AUDIT_PASS",files_written=3,resumed_parent_refits=2,new_registered_conditions=0,new_outputs=6,reused_reference_rows=4)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--execute",action="store_true");p.add_argument("--root-triggered",action="store_true")
    for name in ("public-root","cache-root","audit-path","audit-sha","auditor-sha","builder-sha"):p.add_argument("--"+name)
    v=p.parse_args(argv)
    if not v.execute:
        print(json.dumps(dict(status="PREPARATION_ONLY_NOT_EXECUTED",input_reads=0,files_written=0,scientific_or_child_operations=0,numpy_imported="numpy" in sys.modules,torch_imported="torch" in sys.modules)));return 0
    if not v.root_triggered or any(getattr(v,key) is None for key in ("public_root","cache_root","audit_path","audit_sha","auditor_sha","builder_sha")):p.error("ROOT explicit actual audit/current producer SHAs and three fresh outputs required")
    print(json.dumps(execute(v.public_root,v.cache_root,v.audit_path,v.audit_sha,v.auditor_sha,v.builder_sha)));return 0


if __name__=="__main__":raise SystemExit(main())
