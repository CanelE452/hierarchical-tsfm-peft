"""CreateNew Korean two-family reports only from an actual ROOT-pinned audit PASS.

Default0 I/O; no statistic/trial/model/NPZ/Task parsing or experiment operations.
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
OUTPUTS=("INITIAL_FAMILY_RESULTS_v1.csv","INITIAL_FAMILY_KO_v1.md","INITIAL_FAMILY_BINDINGS_v1.json")
FIELDS=("role","run_id","registered_K","effective_F_width","MSE","MAE","accuracy_device","training_device","CPU_threads","observed_labels","V_origins","search_max_epoch","selected_refit_epoch","search_updates","refit_updates","search_exposures","refit_exposures","search_seconds","refit_seconds","training_status","lr","lora_rank","accuracy_reused","unmerged_F_base_sha256","F_resident_bytes","F_resident_bytes_scope")+tuple(f"B{batch}_"+name for batch in (1,8) for name in ("median_seconds","p90_seconds","CPU_weights_MiB","observed_tree_peak_RSS_bytes","F_instances","F_streams_per_trial","F_calls_per_trial","effective_F_state_sha256","effective_F_state_scope","model_load_seconds","LoRA_merge_and_model_loading_in_whole_trial_timer","CPU_restore_snapshot_bytes_in_Route_memory_state"))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""):h.update(chunk)
    return h.hexdigest()


def validate_audit(v):
    expected=dict(new_accuracy_conditions=2,new_public_stats_rows=5418,unique_observed_V_labels=43344,new_trial_rows=200,new_cost_conditions=4,new_summary_groups=8,new_summary_numeric_fields_checked=40,F_logical_streams=12150,F_physical_streams=12150,F_calls=250)
    if (v.get("schema")!="INDEPENDENT_INITIAL_FAMILY_AUDIT_v1" or v.get("status")!="PASS" or v.get("error") is not None
            or v.get("input_SHA_before_after_equal") is not True or v.get("whole_MASTER_complete") is not False or v.get("independent_generalization_claimed") is not False
            or v.get("baseline_comparison_included") is not False or v.get("totals")!=expected):raise ValueError("Actual complete two-family arithmetic/source audit PASS required")
    rows=v.get("rows",[])
    if len(rows)!=2 or [r.get("role") for r in rows]!=["LORA_R8_C21","B4_K6"] or [r.get("effective_F_width") for r in rows]!=[21,6] or [r.get("registered_K") for r in rows]!=[0,6]:raise ValueError("Exact initial two original family routes required")
    for row in rows:
        if (row.get("observed_labels")!=43344 or row.get("V_origins")!=43 or row.get("CPU_threads")!=2 or row.get("accuracy_device")!="cpu"
                or row.get("F_resident_bytes") is not None or row.get("F_resident_bytes_scope")!="NOT_SEPARATELY_RECORDED"):raise ValueError("Exact cohort/CPU scope and honest memory NA required")
        if any(not math.isfinite(float(row.get(k,math.nan))) or float(row[k])<0 for k in ("MSE","MAE")):raise ValueError("Finite original errors required")
        for batch in (1,8):
            if row.get(f"B{batch}_LoRA_merge_and_model_loading_in_whole_trial_timer") is not False:raise ValueError("Cold load/merge excluded from warm whole timers")
            if row["effective_F_width"]==21 and row.get(f"B{batch}_CPU_restore_snapshot_bytes_in_Route_memory_state") is not False:raise ValueError("LoRA restore snapshot exclusion must remain explicit")


def compose(rows):
    stream=io.StringIO(newline="");writer=csv.DictWriter(stream,fieldnames=FIELDS,lineterminator="\n");writer.writeheader();writer.writerows({k:r.get(k,"") for k in FIELDS} for r in rows)
    errors="\n".join(f"{r['role']:<14} {r['MSE']:.17g} {r['MAE']:.17g} {r['registered_K']}/{r['effective_F_width']}" for r in rows)
    activity="\n".join(f"{r['role']:<14} {r['selected_refit_epoch']} {r['search_updates']}/{r['refit_updates']} {r['search_exposures']}/{r['refit_exposures']} {r['search_seconds']:.17g}/{r['refit_seconds']:.17g} {r['lr']:.17g} {r['training_status']}" for r in rows)
    costs="\n".join(f"{r['role']:<14} {r['B1_median_seconds']:.17g}/{r['B1_p90_seconds']:.17g} {r['B8_median_seconds']:.17g}/{r['B8_p90_seconds']:.17g} {r['B1_CPU_weights_MiB']:.17g}/{r['B8_CPU_weights_MiB']:.17g} {r['B1_observed_tree_peak_RSS_bytes']}/{r['B8_observed_tree_peak_RSS_bytes']}" for r in rows)
    loads="\n".join(f"{r['role']:<14} {r['B1_model_load_seconds']:.17g}/{r['B8_model_load_seconds']:.17g} {r['B1_LoRA_merge_and_model_loading_in_whole_trial_timer']}/{r['B8_LoRA_merge_and_model_loading_in_whole_trial_timer']} {r['B1_CPU_restore_snapshot_bytes_in_Route_memory_state']}/{r['B8_CPU_restore_snapshot_bytes_in_Route_memory_state']}" for r in rows)
    md=f"""# Jena: 초기 LoRA·B4 방법군 최소 비교

이 보고가 답하는 질문은 **등록된 두 다른 적응 원리를 실제로 학습·전체 V 추론·검산·단독 비용까지 수행했는가, 그 원수치는 무엇인가**다. 아래 결과는 같은 Jena C21/L512/H48, seed92601, CPU2, 반복 개발 V43의 normalized channel-macro MSE와 MAE다. 이 두 경로의 완료가 주방법 우위나 전체 연구 해결을 뜻하지 않는다.

[원정밀도 CSV](INITIAL_FAMILY_RESULTS_v1.csv) · [입력·출력 SHA 결속](INITIAL_FAMILY_BINDINGS_v1.json)

```text
role           MSE                 MAE                 registered-K/effective-F-width
{errors}
```

LoRA는 등록 K0/rank8/alpha16이고 실제 F 입력은 원 C21이다. B4는 실제 K6 시계열을 F에 넣는다. 등록 K0를0개 입력으로 해석하거나, LoRA C21과 B4 K6를 같은 입력 구조·같은 모수 수의 격리 실험으로 해석하지 않는다. 원 등록 LoRA lr0.0001과 B4 lr0.001도 보존했다. 같은 방식의 decoder만 비교한 실험이 아니다.

```text
role           selected-refit search/refit-updates search/refit-exposures search/refit-seconds lr training-status
{activity}
```

두 조건은 search A50[0,10482), A_stop[10482,12579), fresh refit A60[0,12579), V[14675,16772)의43 origins·43344 observed labels를 공유한다. 43344는 원 manifest에서 확인한 유효 label 수이며 두 결과를 합쳐 독립 표본 수를 늘리지 않는다. 원 scaler·target·mask·origin·channel 지문을 대사했다. 검색0/.25/1/3은 같아도 선택 refit 길이와 exposure/update가 다를 수 있다. **단일 seed**, 제한 학습, 반복 개발 V이고 충분한 수렴·독립 일반화는 미확정이다. MSE 목적의 실제 학습을 재구성 학습 실패로 바꾸어 설명하지 않는다.

현재 entry의 새 계산은 허용한 과거 prefix만 사용하며 보호 T를 새로 파싱하거나 선택하지 않았다. 그러나 원 Jena **전체 TRAIN 정규화 ancestry**와 과거 quarter/probe의 **full-loader ancestry**는 그대로 남는다. prefix 경계 검사 통과를 원자료 전체의 독립적인 causal 생성·정규화 검증으로 확대하지 않는다. 이 보고를 위해 전처리나 기존 모델을 재학습하지 않았다.

```text
role           B1 median/p90 seconds B8 median/p90 seconds CPU weights MiB B1/B8 observed-tree RSS bytes B1/B8
{costs}
```

각 비용은 물리 history/mask→채움·정규화→해당 적응 경로/F→원 C 출력→FP64 inverse·CPU 반환의 **warm 한 batch 전체**다. B8은8 origin을 담은 한 batch이며8로 나눈 시간이 아니다. ordinary timed20의 median/p90를 표시하고 warm10·별도 intrusive stage20은 구분했다. intrusive stage 합=ordinary whole을 주장하지 않는다. LoRA B8은168 streams·2 F calls/trial, B4 B8은48 streams·1 call/trial이다. 실제 whole cost의200 trial·8 summary·40 지연시간 요약 필드를 원 exact 정책으로 검산했다.

```text
role           Route-model-load-seconds B1/B8 load-and-LoRA-merge-in-whole B1/B8 restore-snapshot-in-Route-weights B1/B8
{loads}
```

**cold load와 LoRA merge는 whole trial timer에 포함되지 않는다**(`LoRA_merge_and_model_loading_in_whole_trial_timer=False`). 표의 Route `model_load_seconds`는 원 기록의 별도 소요시간으로, F 전체 cold startup·서비스 전체 응답시간으로 확대하지 않는다. LoRA는 CPU restore snapshot이 Route weights 집계에 포함되지 않았다(`CPU_restore_snapshot_bytes_in_Route_memory_state=False`). snapshot 용량을 추정해 채우지 않는다. CPU weights는 원 Route의 모듈 집계이고 총 process memory가 아니다. 관측 tree RSS를 별도 보존했고, F만의 resident bytes는 미측정 NA다. F instances=1은 모델 메모리가 K배로 줄었다는 의미가 아니다.

LoRA 비용 중 effective merged F hash와 Route.close 이후 복원한 원 unmerged F hash를 CSV·감사에 따로 보존했다. **LoRA는 내부 저랭크 adapter를 학습하고 이를 merge한 effective F를 사용한다. 원 base weights의 보존·복원 hash 일치가 모든 effective F의 동결을 뜻하지 않으며, B4/B2의 frozen F 대조와 구분한다.** 두 hash가 항상 다르다고 가정하지 않는다. LoRA의 fresh CPU6 replay126 streams/2calls와 추가 merge parity42 streams/2calls는 별도 활동이다. 검산 완료를 새 모델 추론·새 독립 일반화 증거로 부르지 않는다.

기존 [Jena B2 K6 manifest](../postT_V_only_cpu_lag_20261016_v1/predictions/postT_cpu_jena_E3_B2_k6_s92601_v1/V_select/PREDICTION_MANIFEST.json)는 원 fullV outer·STARTED·guard/SPEC·실제 CPU forward 소스까지 추적된 기존 참고다. 해당 manifest의 원 SHA는 `6a885f3c8f298009ae93c731409baff116263a41c9c17ecaff713cf5e9499148`이다. 기존 단독 CPU 비용은 guard4GiB 등 원 resource/quiet 감사 범위가 현재3GiB 계약과 다르므로 여기 새 두 row의 동일 계약 Pareto로 섞지 않았다. 미확인 장치의 Raw/G/PCA도 새 CPU 공정 비교표에 넣지 않았다. B2 대비 추가 가치의 최종 판단에는 기존 결과의 정확한 비용·검산 scope를 따로 대사해야 한다.

**전체 MASTER 미완료**다. 이 초기 묶음은 네 family recipe 중 Jena LoRA·B4 두 조건만 포함하며 Jena ADAPTS와 Beijing B4는 자동 완료로 표시하지 않는다. AdaPTS 원 논문 전체 재현을 뜻하지 않는다. 실제 두 학습·fullV·독립 FP64·CPU6·4비용과 source48 보존·강한 recorded process tree·quiet/nonoverlap가 모두 확인될 때만 이 완성 보고를 생성한다. 부분 종료면 builder가 거절하며 ROOT는 성공·실패·미실행 및 저장점·재개 방법을 별도 정직한 부분 보고로 남길 수 있다. 다음 판단은 이 반복 V의 오차·whole 비용·학습량·구조 혼재를 함께 검토하는 것이며 자동 새 모듈·학습 연장·주방법 변경은 없다.
"""
    return {OUTPUTS[0]:stream.getvalue(),OUTPUTS[1]:md}


def execute(public,cache,audit_path,audit_sha,auditor_sha,builder_sha):
    public,cache=Path(public).resolve(),Path(cache).resolve();scope=public/NAME
    if any((scope/name).exists() for name in OUTPUTS):raise FileExistsError("Existing initial family reports preserved before inputs")
    audit_path=Path(audit_path).resolve();source=scope/"audit_initial_family.py"
    if not audit_path.is_relative_to(scope) or audit_path.suffix!=".json":raise ValueError("Exact actual public N5 audit required")
    if Path(__file__).resolve()!=scope/"build_initial_family.py" or sha(__file__)!=builder_sha or sha(source)!=auditor_sha or sha(audit_path)!=audit_sha:raise ValueError("Exact actual audit and current reviewed producer SHAs required")
    spec=importlib.util.spec_from_file_location("initial_family_builder_contract",source);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a);_,helper=a.load_helpers(public)
    inputs=helper.Inputs(public,cache);inputs.capture("public",NAME+"/build_initial_family.py",builder_sha);inputs.capture("public",NAME+"/audit_initial_family.py",auditor_sha)
    audit=inputs.json("public",audit_path.relative_to(public).as_posix(),audit_sha);validate_audit(audit)
    if audit.get("audit_source_sha256")!=auditor_sha:raise ValueError("Actual audit source differs")
    for row in audit["inputs"]:inputs.capture(row["root"],row["relative_path"],row["sha256"])
    texts=compose(audit["rows"])
    if not inputs.preserved():raise ValueError("Input SHA changed; completed report not emitted")
    bindings=dict(schema="INITIAL_FAMILY_REPORT_BINDINGS_v1",status="FROM_ACTUAL_COMPLETE_INITIAL_FAMILY_AUDIT",audit_relative_path=audit_path.relative_to(public).as_posix(),audit_sha256=audit_sha,auditor_source_sha256=auditor_sha,builder_source_sha256=builder_sha,input_SHA_before_after_equal=True,source48_preserved=True,inputs=[dict(root=root,relative_path=name,sha256=digest) for (root,name),digest in sorted(inputs.files.items())],outputs=[dict(relative_path=NAME+"/"+name,sha256=hashlib.sha256(text.encode()).hexdigest()) for name,text in texts.items()],new_accuracy_conditions=2,new_cost_conditions=4,whole_MASTER_complete=False,independent_generalization_claimed=False,new_models_NPZ_raw_T_numeric_training_inference_cost_or_process_operations=0)
    texts[OUTPUTS[2]]=json.dumps(bindings,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    for name,text in texts.items():
        with (scope/name).open("x",encoding="utf-8",newline="\n") as f:f.write(text)
    return dict(status="REPORT_CREATED_FROM_ACTUAL_COMPLETE_INITIAL_FAMILY_AUDIT",files_written=3,input_count=len(inputs.files))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--execute",action="store_true");p.add_argument("--root-triggered",action="store_true")
    for name in ("public-root","cache-root","audit-path","audit-sha","auditor-sha","builder-sha"):p.add_argument("--"+name)
    v=p.parse_args(argv)
    if not v.execute:
        print(json.dumps(dict(status="PREPARATION_ONLY_NOT_EXECUTED",input_reads=0,files_written=0,scientific_or_child_operations=0,numpy_imported="numpy" in sys.modules,torch_imported="torch" in sys.modules)));return 0
    if not v.root_triggered or any(getattr(v,k) is None for k in ("public_root","cache_root","audit_path","audit_sha","auditor_sha","builder_sha")):p.error("ROOT current source and actual complete audit SHAs required")
    r=execute(v.public_root,v.cache_root,v.audit_path,v.audit_sha,v.auditor_sha,v.builder_sha);print(json.dumps(r));return 0


if __name__=="__main__":raise SystemExit(main())
