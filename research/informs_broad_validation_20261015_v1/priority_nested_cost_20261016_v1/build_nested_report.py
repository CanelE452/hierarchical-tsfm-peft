"""Report existing nested-PCA V accuracy and completed CPU costs; stdlib only."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path


NAME = "priority_nested_cost_20261016_v1"
OLD = "postT_V_only_cpu_multik_20261016_v1"
TOKENS = {3:"464962a4e7",6:"8246ac2ead",11:"a9916278ae"}
OUTPUTS = ("RESULTS_WITH_COST_v1.csv","RESULTS_WITH_COST_KO_v1.md","README_KO.md","REPORT_INPUT_BINDINGS_v1.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Inputs:
    def __init__(self, root):
        self.root,self.files = Path(root).resolve(),{}

    def capture(self, name):
        path = self.root/name
        if Path(name).is_absolute() or ".." in Path(name).parts or not path.resolve().is_relative_to(self.root):
            raise ValueError("Campaign-relative report inputs required")
        digest = sha(path)
        if name in self.files and self.files[name] != digest:
            raise ValueError("Input changed during report construction")
        self.files[name] = digest
        return path

    def json(self, name):
        return json.loads(self.capture(name).read_text(encoding="utf-8-sig"))

    def csv(self, name):
        with self.capture(name).open(encoding="utf-8-sig",newline="") as stream:
            return list(csv.DictReader(stream))

    def preserved(self):
        return all(sha(self.root/name)==digest for name,digest in self.files.items())


def assemble(inputs):
    plan = inputs.json(NAME+"/PLAN.json")
    original = inputs.json(OLD+"/PLAN.json")
    audit = inputs.json(NAME+"/INDEPENDENT_COST_REVIEW_ROOT_v1.json")
    if (plan.get("status") != "SEALED_BY_ROOT" or inputs.files[OLD+"/PLAN.json"] != "2051573b8dee48200eb34e96e37e7af0d197442619ea1ba52a36a80ad4e27aec"
            or plan["original_plan_sha256"] != inputs.files[OLD+"/PLAN.json"] or audit.get("status") != "PASS"
            or audit.get("schema") != "INDEPENDENT_NESTED_COST_REVIEW_v1"):
        raise ValueError("Sealed original/new plans and actual independent cost audit PASS required")
    rows=[]
    for k,token in TOKENS.items():
        rid=f"postT_cpu_beijing_E4_NESTED_PCA_k{k}_s92601_v1"
        folder=OLD+"/ordered_predictions/"+rid
        manifest=inputs.json(folder+"/PREDICTION_MANIFEST.json")
        scores=inputs.csv(folder+"/SCORES.csv")
        proof={}
        for mode in ("rescore","replay"):
            path=OLD+f"/diagnostics/rolling_root_v1_nested_{k}_{mode}_{token}/RESULT.json"
            value=inputs.json(path)
            if (value.get("status") != "PASS" or value.get("run_id") != rid or value.get("error_class") is not None
                    or value.get("postflight_error_class") is not None or value.get("plan_sha256") != inputs.files[OLD+"/PLAN.json"]):
                raise ValueError("Exact completed old per-K rescore/replay proof required")
            proof[mode]=value
        if (len(scores)!=1 or manifest["score"]["n_origins"]!=43 or manifest["score"]["n_observed"]!=20539
                or manifest["score"]["metric_scale"]!="A_normalized_original_channels"
                or manifest["metadata"]["parent_artifact_sha256"]!="daf45933e8df040390c3c7237e892c16134e61ec1c23e8cebec00a64f29faf73"
                or proof["replay"].get("actual_checkpoint_forward_in_this_command") is not True):
            raise ValueError("Old fixed basis/V43/actual CPU6 identity differs")
        for metric in ("mse","mae"):
            if (float(scores[0][metric])!=manifest["score"][metric]
                    or not math.isclose(proof["rescore"]["scientific_result"][metric],manifest["score"][metric],rel_tol=1e-7,abs_tol=1e-9)):
                raise ValueError("Original V score and independent rescore disagree")
        row=dict(dataset="beijing",run_id=rid,K=k,L=512,H=48,C=11,MSE=manifest["score"]["mse"],MAE=manifest["score"]["mae"],
                 V_origins=43,observed_labels=20539,accuracy_scope="REUSED_FULL_V_FP64_CPU6",accuracy_device="cpu",new_optimizer_updates=0,
                 PCA_fit_seconds="NOT_RECORDED",neural_convergence="NOT_APPLICABLE_FIXED_TRANSFORM",CPU_threads=2,row_microbatch=128)
        for batch in (1,8):
            tag=f"nested_k{k}_cost_b{batch}_root_v1";base=NAME+"/diagnostics/"+tag
            outer=inputs.json(base+"/RESULT.json");receipt=inputs.json(base+"/COST_RECEIPT.json");access=inputs.json(base+"/PREFIX_ACCESS_RECEIPT.json")
            inputs.capture(base+"/COST_TRIALS.csv.gz")
            if (outer.get("status")!="COMPLETE_PROTOCOL" or outer.get("error_class") is not None or outer.get("postflight_error_class") is not None
                    or outer.get("original_source39_immutable9_and_selected_inputs_preserved") is not True
                    or outer.get("plan_sha256")!=inputs.files[NAME+"/PLAN.json"] or any(outer["scientific_result"].get(key)!=value for key,value in receipt.items())):
                raise ValueError("Successful final outer cost RESULT and source preservation required")
            if (receipt["device"]!="cpu" or receipt["CPU_threads"]!=2 or receipt["latent_k"]!=k or receipt["batch_size"]!=batch
                    or receipt["accuracy_manifest_sha256"]!=inputs.files[folder+"/PREDICTION_MANIFEST.json"]
                    or receipt["trials_sha256"]!=inputs.files[base+"/COST_TRIALS.csv.gz"] or receipt["actual_trial_rows"]!=50
                    or receipt["actual_training"]!=0 or receipt["cached_Z_or_forecast_reads"]!=0
                    or receipt["expected_F_streams_per_trial"]!=batch*k or receipt["F_calls"]!=50
                    or access["new_T_numeric_values_parsed"]!=0 or access["materialized_rows"]!=16830):
                raise ValueError("Exact physical CPU2 protocol/artifact/prefix binding differs")
            timed=[value for value in receipt["summaries"] if value["phase"]=="timed"]
            if len(timed)!=1 or timed[0]["trials"]!=20:
                raise ValueError("Original20 timed trials required")
            prefix=f"B{batch}_"
            row.update({prefix+"median_seconds":timed[0]["median_seconds"],prefix+"p90_seconds":timed[0]["p90_seconds"],
                        prefix+"model_load_seconds":receipt["model_load_seconds"],prefix+"resident_weights_cpu_MiB":receipt["memory"]["resident_weights_cpu_MiB"],
                        prefix+"observed_peak_RSS_bytes":outer["OS_peak_working_set_bytes"],prefix+"F_streams_per_trial":batch*k,prefix+"F_calls_per_trial":1})
        rows.append(row)
    return rows


def compose(rows):
    stream=io.StringIO(newline="");writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    values="\n".join(f"{r['K']:2} {r['MSE']:.9g} {r['MAE']:.9g} {r['B1_median_seconds']:.9g} {r['B1_p90_seconds']:.9g} {r['B8_median_seconds']:.9g} {r['B8_p90_seconds']:.9g}" for r in rows)
    memory="\n".join(f"{r['K']:2} {r['B1_resident_weights_cpu_MiB']:.9g} {r['B8_resident_weights_cpu_MiB']:.9g} {r['B1_model_load_seconds']:.9g} {r['B8_model_load_seconds']:.9g}" for r in rows)
    report=f"""# Beijing 고정 nested PCA: 기존 V와 새 CPU 비용6개

**[확인] 같은 고정 A60 전체기저의 K3/6/11 기존 V를 재사용하고, 각 K의 B1/B8 물리입력 전체 추론 비용6개를 실제 측정했다. 신규 학습·정확도 재추론은0이며 전체 MASTER 완료가 아니다.** 정확도는 같은 반복 개발 V43/seed92601의 조건부 결과다.

[원정밀도 CSV](RESULTS_WITH_COST_v1.csv) · [독립 비용 감사](INDEPENDENT_COST_REVIEW_ROOT_v1.json) · [실행 계획](PLAN.json) · [원 계획](../{OLD}/PLAN.json) · [입력 결속](REPORT_INPUT_BINDINGS_v1.json)

```text
 K MSE         MAE         B1 median s  B1 p90 s   B8 median s  B8 p90 s
{values}
```

Beijing C11/L512/H48, A50end10519/A60end12622, V[14726,16830)의43원점·20,539관측 정답과 같은 scaler3348bcf1…93fb를 유지했다. 오차는 원 C 채널의 A 정규화 척도다. 기존 전체 V FP64 재점수와 CPU6(4+2) checkpoint 재추론은 정확한 원 영수증으로 재사용했다. 이 보고 단계에서는 새 모델·NPZ 숫자·원자료를 읽거나 비용을 다시 측정하지 않았다.

공통 artifact의 A60 PCA basis11×11과 center를 그대로 쓰며 K는 같은 basis의 앞3/6/11축이다. K11도 회전된 좌표에서 F를 거친 뒤 basis transpose로 복원하므로 Raw 항등 경로와 같지 않다. 학습된 B2 lag D·현재값 anchor와도 복원 구조가 다르다. K 증가의 V 차이를 학습된 시간·채널 압축의 우월성으로 해석하지 않는다.

```text
 K B1 CPU weights MiB  B8 CPU weights MiB  B1 model load s  B8 model load s
{memory}
```

비용은 CPU2/row128, 각10warm·20timed·별도20stage로 총300trial이다. 원 physical history/mask→과거 채움·정규화→고정 projection→동결 Bolt→basis inverse→FP64 역변환·CPU 반환을 측정했고 cached Z/forecast는 쓰지 않았다. 논리·물리 stream은 각 trial의 B×K, B8K11도88stream이어서 모두 실제 F call1회다. stage는 intrusive 별도 범위며 stage 합이 whole과 같다고 하지 않는다. 모든 실제비용 종료·quiet·source48·수치 재집계는 독립 감사의 범위로 연결한다.

모델 적재와 resident weight MiB는 steady-state latency 및 sampled peak RSS와 별개다. K를 줄여도 동결 backbone resident weights가 같은 점을 CSV에서 확인할 수 있다. 표의 적재 시간은 cold disk cache나 전체 Python/data startup 비용을 뜻하지 않는다. PCA 기저 제조 시간은 원 영수증에 별도 기록이 없어 `NOT_RECORDED`다. 새 optimizer0회는 PCA fit 비용0초를 뜻하지 않는다.

이번 숫자 변환은 원 source의0:16830과 작은 channel metadata로 제한됐으며 새 보호 T 수치·target·평가는0이다. 전체 압축파일의 opaque SHA/ZIP I/O와 numeric parse 경계를 구분하고, 옛 CPU6 loader의 역사적 물리 역직렬화 이력을 지우지 않는다. 고정 transform은 신경망 수렴의 판정 대상이 아니며, 이 결과가 원 신경망 학습의 수렴 미확정을 해결하지 않는다.

결론은 한 자료·한 seed에서 고정기저의 비용/오차가 어떻게 달라지는지 확인했다는 것이다. 무조건적인 전체 정확도·메모리·총학습비용 우위나 독립 일반화 근거가 아니다. 다음 기존 Beijing B2/causal/bidir K6 최소3조건은 아직 준비이며 원예산·보호 T 계약 안에서 별도 실행한다. 중단 shared/max·미시작13/core29·M5·다른 자료/K/학습량 요구는 여전히 별도 잔여다.
"""
    readme="# 고정 nested PCA 비용 후속\n\n[한국어 결과](RESULTS_WITH_COST_KO_v1.md) · [원수치](RESULTS_WITH_COST_v1.csv) · [독립 감사](INDEPENDENT_COST_REVIEW_ROOT_v1.json) · [입력 SHA](REPORT_INPUT_BINDINGS_v1.json) · [stdlib builder](build_nested_report.py)\n\n기존 정확도3행+새 CPU 비용6개다. 새 학습0이며 전체 연구는 PARTIAL이다. 기본 builder 호출은 입력읽기·쓰기0, 명시 --execute는 감사 PASS 후 새 보고4개만 만들고 기존 출력을 거부한다.\n\n```powershell\npython -B research/informs_broad_validation_20261015_v1/priority_nested_cost_20261016_v1/build_nested_report.py --execute --public-root research/informs_broad_validation_20261015_v1\n```\n"
    return {"RESULTS_WITH_COST_v1.csv":stream.getvalue(),"RESULTS_WITH_COST_KO_v1.md":report,"README_KO.md":readme}


def execute(root):
    inputs=Inputs(root);scope=inputs.root/NAME
    if any((scope/name).exists() for name in OUTPUTS):
        raise FileExistsError("Existing reports are preserved; fresh outputs required")
    inputs.capture(NAME+"/build_nested_report.py");rows=assemble(inputs);texts=compose(rows)
    if not inputs.preserved():
        raise ValueError("All report inputs must remain unchanged")
    bindings=dict(status="REPORT_FROM_COMPLETED_COSTS_AND_REUSED_V",input_SHA_before_after_equal=True,input_count=len(inputs.files),
                  inputs=[dict(relative_path=name,sha256=digest) for name,digest in sorted(inputs.files.items())],
                  outputs=[dict(relative_path=NAME+"/"+name,sha256=hashlib.sha256(text.encode()).hexdigest()) for name,text in texts.items()],
                  new_training_prediction_cost_NPZ_model_T_numeric_operations=0,accuracy_rows_reused=3,new_cost_conditions_reported=6,
                  original_source48_detail="Reused exact original PLAN and actual independent audit; no duplicate source48 list",whole_MASTER_complete=False)
    texts["REPORT_INPUT_BINDINGS_v1.json"]=json.dumps(bindings,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    for name,text in texts.items():
        with (scope/name).open("x",encoding="utf-8",newline="\n") as stream:stream.write(text)
    return dict(status="REPORT_CREATED",files_written=4,accuracy_rows=3,cost_conditions=6,input_count=len(inputs.files))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--execute",action="store_true");parser.add_argument("--public-root")
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="METADATA_ONLY_NOT_EXECUTED",input_reads=0,files_written=0,scientific_or_child_operations=0)));return 0
    if not args.public_root:parser.error("Explicit --execute --public-root required")
    print(json.dumps(execute(args.public_root)));return 0


if __name__=="__main__":raise SystemExit(main())
