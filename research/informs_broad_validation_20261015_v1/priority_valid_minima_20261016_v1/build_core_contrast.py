"""Build three core-contrast report files only from ROOT-pinned actual audit PASS.

Default0 I/O. No statistical gzip, NPZ/model/Task/CP decoding or experiments.
The existing N4 B2 K6 row is reused with exact original report/manifest SHA.
"""
import argparse
import copy
import csv
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys

NAME="priority_valid_minima_20261016_v1"
OUTPUTS=("CORE_CONTRAST_RESULTS_v1.csv","CORE_CONTRAST_KO_v1.md","CORE_CONTRAST_BINDINGS_v1.json")
FIELDS=("role","run_id","K","MSE","MAE","accuracy_reused","source_of_values","accuracy_device","training_device","CPU_threads","search_max_epoch","selected_refit_epoch","search_updates","refit_updates","search_exposures","refit_exposures","search_seconds","refit_seconds","training_status","initialization","parameter_count_scope","MSE_vs_B2K6_relative","MAE_vs_B2K6_relative","B1_vs_B2K6_relative","B8_vs_B2K6_relative",
    "B1_median_seconds","B1_p90_seconds","B8_median_seconds","B8_p90_seconds","B1_CPU_weights_MiB","B8_CPU_weights_MiB","B1_observed_tree_peak_RSS_bytes","B8_observed_tree_peak_RSS_bytes","F_used","frozen_F_sha256","F_resident_bytes","F_resident_bytes_scope","B1_F_instances","B8_F_instances","B1_F_streams_per_trial","B8_F_streams_per_trial","B1_F_calls_per_trial","B8_F_calls_per_trial","observed_labels","V_origins")


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""):h.update(chunk)
    return h.hexdigest()


def relative_delta(value,reference):return value/reference-1 if reference else None


def validate_audit(value):
    expected=dict(new_accuracy_conditions=2,reused_accuracy_conditions=1,new_public_stats_rows=2838,unique_observed_V_labels=20539,new_trial_rows=200,new_cost_conditions=4,reused_cost_conditions=2,new_summary_groups=8,new_summary_numeric_fields_checked=40,F_logical_streams=7650,F_physical_streams=7650,F_calls=200)
    if (value.get("schema")!="INDEPENDENT_BEIJING_CORE_CONTRAST_v1" or value.get("status")!="PASS" or value.get("error") is not None or value.get("input_SHA_before_after_equal") is not True
            or value.get("whole_MASTER_complete") is not False or value.get("independent_generalization_claimed") is not False or value.get("totals")!=expected):raise ValueError("Exact complete new two-condition numerical audit PASS required")
    rows=value.get("rows",[])
    if len(rows)!=3 or [r.get("role") for r in rows]!=["B0_K6","B2_K6","B2_K11"] or [r.get("K") for r in rows]!=[6,6,11] or [r.get("accuracy_reused") for r in rows]!=[False,True,False]:raise ValueError("Exactly two new rows and one reused B2 row required")
    for row in rows:
        if row.get("observed_labels")!=20539 or row.get("V_origins")!=43 or row.get("CPU_threads")!=2 or row.get("F_used") is not True or row.get("F_resident_bytes") is not None or row.get("F_resident_bytes_scope")!="NOT_SEPARATELY_RECORDED":raise ValueError("Same CPU V cohort and honest F memory scope required")
        if any(not math.isfinite(float(row.get(k,math.nan))) or float(row[k])<0 for k in ("MSE","MAE")):raise ValueError("Finite raw errors required")


def compose(source_rows):
    rows=copy.deepcopy(source_rows);base=next(r for r in rows if r["role"]=="B2_K6")
    for row in rows:
        for out,key in (("MSE_vs_B2K6_relative","MSE"),("MAE_vs_B2K6_relative","MAE"),("B1_vs_B2K6_relative","B1_median_seconds"),("B8_vs_B2K6_relative","B8_median_seconds")):row[out]=relative_delta(row[key],base[key])
    stream=io.StringIO(newline="");writer=csv.DictWriter(stream,fieldnames=FIELDS,lineterminator="\n");writer.writeheader();writer.writerows({k:r.get(k,"") for k in FIELDS} for r in rows)
    b0=next(r for r in rows if r["role"]=="B0_K6");full=next(r for r in rows if r["role"]=="B2_K11")
    errors="\n".join(f"{r['role']:<8} {r['MSE']:.17g} {r['MAE']:.17g} {r['K']} {r['accuracy_reused']}" for r in rows)
    activity="\n".join(f"{r['role']:<8} {r['selected_refit_epoch']} {r['search_updates']}/{r['refit_updates']} {r['search_exposures']}/{r['refit_exposures']} {r['search_seconds']:.17g}/{r['refit_seconds']:.17g} {r['training_status']}" for r in rows)
    costs="\n".join(f"{r['role']:<8} {r['B1_median_seconds']:.17g}/{r['B1_p90_seconds']:.17g} {r['B8_median_seconds']:.17g}/{r['B8_p90_seconds']:.17g} {r['B1_CPU_weights_MiB']:.17g}/{r['B8_CPU_weights_MiB']:.17g} {r['B1_observed_tree_peak_RSS_bytes']}/{r['B8_observed_tree_peak_RSS_bytes']}" for r in rows)
    md=f"""# Beijing: 같은 K의 B2 추가 가치와 full-C 압축 경계

먼저 두 질문에 답한다. 아래 차이는 같은 반복 개발 V43/seed92601/CPU2에서 관찰한 원수치의 조건부 비교다. normalized channel-macro MSE가 주지표이며 MAE·전체 비용도 함께 판단한다. 성능에 따라 지표·gate·학습 길이를 바꾸지 않았다.

1. **B0 K6에 B2 앞단을 더할 근거가 있는가?** B2 K6−B0 K6의 MSE 차이는 `{base['MSE']-b0['MSE']:.17g}`, MAE 차이는 `{base['MAE']-b0['MAE']:.17g}`다. B1 whole median 차이는 `{base['B1_median_seconds']-b0['B1_median_seconds']:.17g}`초, B8은 `{base['B8_median_seconds']-b0['B8_median_seconds']:.17g}`초다. 음수는 해당 오차/비용이 작다는 뜻이다. 이 값과 선택 refit·노출량을 함께 보아 이 개발 조건에서의 추가 가치를 판단하며, 독립적인 일반적 우위를 선언하지 않는다.
2. **B2를 C11에서 K6으로 줄일 때 정확도와 전체 비용이 어떻게 달라지는가?** K6−K11의 MSE 차이는 `{base['MSE']-full['MSE']:.17g}`, MAE 차이는 `{base['MAE']-full['MAE']:.17g}`이고 B1/B8 whole median 차이는 `{base['B1_median_seconds']-full['B1_median_seconds']:.17g}`/`{base['B8_median_seconds']-full['B8_median_seconds']:.17g}`초다. K6 PCA 초기화와 K11 identity 초기화·표현/앞단·파라미터 용량·선택 학습량의 차이가 함께 있으므로 순수 정보 압축 손실이나 encoder만의 효과로 확대하지 않는다.

[원정밀도 CSV](CORE_CONTRAST_RESULTS_v1.csv) · [입력 SHA와 실제 감사 결속](CORE_CONTRAST_BINDINGS_v1.json) · [기존 N4 보고](RESULTS_KO_v1.md)

```text
role     MSE                 MAE                 K reused
{errors}
```

B0도 B2도 **frozen Bolt**를 실제 사용하는 경로다. 여기서는 noBolt를 새로 학습·비교하지 않았고, 같은 표현에 예측기만 교체한 실험도 아니다. 고정 PCA와 동일 D 효과를 분리한 비교로 부르지 않는다. 기존 N4 B2 K6의 전체 V·FP64·CPU6·두 비용은 exact audit/report/manifest SHA로 재사용했다. 새 B0 K6와 B2 K11만 실제 새 fit/fullV/독립채점/replay/4비용 증거를 연결했다.

```text
role     selected-refit search/refit-updates search/refit-exposures search/refit-seconds training-status
{activity}
```

세 조건은 C11/L512/H48, A50[0,10519), A_stop[10519,12622), fresh A60[0,12622), V[14726,16830)의43 origins·20539 observed labels·원 scaler/target/mask/origin/channel 지문을 공유한다. 같은20539 labels를 반복 사용한 것이며 세 개의 독립 검증 표본이 아니다. 검색0/.25/1/3은 같아도 A_stop 선택 refit·실제 update/exposure·구조/모수·소요시간이 다를 수 있다. 한 seed·제한된3 검색·반복 개발 V이며 충분한 수렴과 독립 일반화는 미확정이다.

```text
role     B1 median/p90 seconds  B8 median/p90 seconds  CPU weights MiB B1/B8  observed-tree RSS bytes B1/B8
{costs}
```

비용은 CPU2 물리 history/mask→채움/정규화→P/E/F/D→FP64 inverse/CPU 반환의 실제 한 batch 전체다. B8을8로 나눈 값이 아니고 K/C를 속도비로 쓰지 않는다. 각50trial은 warm10·ordinary timed20·별도 intrusive stage20이며 표는 ordinary whole median/p90다. stage 합=whole을 주장하지 않는다. F instances와 실제 stream/call은 CSV에 따로 기록했다. CPU weights는 F와 adapter의 합이며 F만의 resident bytes는 원 측정에서 분리하지 않아 NA/NOT_SEPARATELY_RECORDED다. CPU resident weights·관측된 tree RSS·GPU peak·로딩·학습 peak를 같은 메모리 범위로 섞지 않는다.

새2조건의 공개 충분통계2838행과 비용200trial/8summary/40지연시간 필드를 원 정책으로 감사했다. 기존 FP64 재채점·freshCPU6와 actual selected artifact/scaler/source 및 강한 observed process-tree closure/quiet 영수증을 결속했으며 이 builder가 다시 모델·NPZ·비용을 실행한 것은 아니다. 허용오차는 원 MSE/MAE atol1e-9/rtol1e-7, cost 요약은 기존 exact 정책이다. 검사 통과는 성능 개선·인과성·충분 수렴·일반화의 증거가 아니다.

**전체 MASTER 미완료**다. 다른 K/자료·C subset·seed·충분 학습·방법군·공유 multi-K 등 원 공백이 남는다. 보호 T를 새로 평가하거나 사용해 선택하지 않았으며 현재 비교의 실패도 그대로 보고한다. 다음 판단은 두 질문의 오차·전체비용·용량/학습량 혼재를 함께 검토하는 것이다. 이 보고에서 주방법을 자동 선택하거나 일반적인 Bolt 필요성을 확정하지 않는다.
"""
    return {OUTPUTS[0]:stream.getvalue(),OUTPUTS[1]:md}


def execute(public,cache,audit_path,audit_sha,builder_sha):
    public,cache=Path(public).resolve(),Path(cache).resolve();scope=public/NAME
    if any((scope/name).exists() for name in OUTPUTS):raise FileExistsError("Existing core report preserved before inputs")
    audit_path=Path(audit_path).resolve()
    if not audit_path.is_relative_to(scope) or audit_path.suffix!=".json":raise ValueError("Explicit new public core audit JSON path required")
    if Path(__file__).resolve()!=scope/"build_core_contrast.py" or sha(__file__)!=builder_sha or sha(audit_path)!=audit_sha:raise ValueError("Exact actual audit/current builder source SHA required")
    source=scope/"audit_core_contrast.py";spec=importlib.util.spec_from_file_location("core_report_audit_contract",source);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    _,b=a.load_helpers(public);inputs=b.Inputs(public,cache)
    inputs.capture("public",NAME+"/build_core_contrast.py",builder_sha);source_sha=sha(source);inputs.capture("public",NAME+"/audit_core_contrast.py",source_sha)
    audit=inputs.json("public",audit_path.relative_to(public).as_posix(),audit_sha);validate_audit(audit)
    if audit["audit_source_sha256"]!=source_sha:raise ValueError("Audit belongs to a different auditor source")
    for row in audit["inputs"]:
        if row["root"] not in ("public","cache"):raise ValueError("Explicit relative input root required")
        inputs.capture(row["root"],row["relative_path"],row["sha256"])
    prior=inputs.csv("public",NAME+"/RESULTS_WITH_COST_v1.csv");baseline=[r for r in prior if r["run_id"]==a.OLD_RUN]
    if len(baseline)!=1:raise ValueError("Exactly one original report B2 row required")
    manifest=inputs.json("public",NAME+"/predictions/"+a.OLD_RUN+"/V_select/PREDICTION_MANIFEST.json",a.OLD_MANIFEST_SHA);b.validate_cohort(manifest)
    reused=next(r for r in audit["rows"] if r["role"]=="B2_K6")
    for key in ("MSE","MAE","B1_median_seconds","B1_p90_seconds","B8_median_seconds","B8_p90_seconds"):
        if float(baseline[0][key])!=reused[key]:raise ValueError("Reused original N4 CSV differs from audited receipt values")
    if reused["MSE"]!=manifest["score"]["mse"] or reused["MAE"]!=manifest["score"]["mae"]:raise ValueError("Reused exact manifest score differs")
    texts=compose(audit["rows"])
    if not inputs.preserved():raise ValueError("Captured inputs changed; no completed report emitted")
    bindings=dict(schema="BEIJING_CORE_CONTRAST_REPORT_BINDINGS_v1",status="FROM_ACTUAL_COMPLETE_CORE_AUDIT",audit_relative_path=audit_path.relative_to(public).as_posix(),audit_sha256=audit_sha,auditor_source_sha256=source_sha,builder_source_sha256=builder_sha,input_SHA_before_after_equal=True,
        inputs=[dict(root=root,relative_path=name,sha256=digest) for (root,name),digest in sorted(inputs.files.items())],outputs=[dict(relative_path=NAME+"/"+name,sha256=hashlib.sha256(text.encode()).hexdigest()) for name,text in texts.items()],new_accuracy_conditions=2,reused_accuracy_conditions=1,new_cost_conditions=4,reused_cost_conditions=2,new_models_NPZ_raw_T_numeric_training_inference_cost_or_process_operations=0,whole_MASTER_complete=False,independent_generalization_claimed=False)
    texts[OUTPUTS[2]]=json.dumps(bindings,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    for name,text in texts.items():
        with (scope/name).open("x",encoding="utf-8",newline="\n") as f:f.write(text)
    return dict(status="REPORT_CREATED_FROM_ACTUAL_COMPLETE_CORE_AUDIT",files_written=3,input_count=len(inputs.files))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--execute",action="store_true");p.add_argument("--root-triggered",action="store_true")
    for flag in ("public-root","cache-root","audit-path","audit-sha","builder-sha"):p.add_argument("--"+flag)
    a=p.parse_args(argv)
    if not a.execute:
        print(json.dumps(dict(status="PREPARATION_ONLY_NOT_EXECUTED",input_reads=0,files_written=0,scientific_or_child_operations=0,numpy_imported="numpy" in sys.modules,torch_imported="torch" in sys.modules)));return 0
    if not a.root_triggered or any(getattr(a,k) is None for k in ("public_root","cache_root","audit_path","audit_sha","builder_sha")):p.error("ROOT trigger/current source/actual PASS audit SHA/roots required")
    try:r=execute(a.public_root,a.cache_root,a.audit_path,a.audit_sha,a.builder_sha)
    except (ValueError,FileNotFoundError,FileExistsError,KeyError):print(json.dumps(dict(status="NOT_REPORTABLE_EVIDENCE_INCOMPLETE_OR_CHANGED")));return 1
    print(json.dumps(r));return 0


if __name__=="__main__":raise SystemExit(main())
