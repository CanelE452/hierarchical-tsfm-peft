"""Build aggregate reports from completed public V receipts; default is no I/O."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path


NAME = "priority_core_followup_20261016_v1"
P1 = "priority_b2_validation_20261016_v1"
LAG = "postT_V_only_cpu_lag_20261016_v1"
PARENT = "diagnostic_pems08_B2_parent_full_CPU_V148_k43_s92601_v1"
OUTPUTS = ("RESULTS_v2.csv", "RESULTS_KO_v2.md", "README_KO.md", "TRAINING_COST_v2.csv", "REPORT_INPUT_BINDINGS_v2.json")
FIELDS = ("dataset", "role", "run_id", "seed", "L", "H", "C", "K", "MSE", "MAE", "selected_refit_epoch",
          "accuracy_device", "accuracy_scope", "B1_median_seconds", "B8_median_seconds", "cost_device", "cost_scope",
          "V_origins", "observed_labels", "search_max_epoch", "refit_updates", "training_status", "notes")
TRAINING_FIELDS = ("dataset", "role", "run_id", "training_device", "search_max_epoch", "selected_refit_epoch",
                   "search_updates", "refit_updates", "search_exposures", "refit_exposures", "search_seconds", "refit_seconds",
                   "fit_and_V_wall_seconds", "full_V_only_wall_seconds", "additional_optimization_only", "parent_training_required", "notes")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Inputs:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.bindings = {}

    def path(self, relative):
        path = self.root / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts or path.is_symlink():
            raise ValueError("Public campaign relative nonsymlink inputs required")
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("Input escapes public campaign")
        return path

    def capture(self, relative):
        path = self.path(relative)
        digest = sha(path)
        if relative in self.bindings and self.bindings[relative] != digest:
            raise ValueError("Report input changed during reading: " + relative)
        self.bindings[relative] = digest
        return path

    def json(self, relative):
        return json.loads(self.capture(relative).read_text(encoding="utf-8-sig"))

    def csv(self, relative):
        with self.capture(relative).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    def preserved(self):
        return all(sha(self.path(name)) == digest for name, digest in self.bindings.items())


def completed(inputs, tag, status=None):
    value = inputs.json(NAME + "/diagnostics/" + tag + "/RESULT.json")
    if value.get("error_class") is not None or value.get("postflight_error_class") is not None:
        raise ValueError("Failed outer RESULT cannot support a report: " + tag)
    if status is not None and value.get("status") != status:
        raise ValueError("Required completed RESULT missing: " + tag)
    if not value.get("status", "").startswith(("COMPLETE", "PASS")):
        raise ValueError("Incomplete RESULT: " + tag)
    return value


def cost(inputs, tag):
    outer = completed(inputs, tag)
    receipt = inputs.json(NAME + "/diagnostics/" + tag + "/COST_RECEIPT.json")
    if (receipt.get("status") not in {"COMPLETE_PROTOCOL","COMPLETE_PARENT_CPU_COST_PROTOCOL"}
            or outer["status"] != receipt["status"] or receipt.get("device") != "cpu"
            or receipt.get("CPU_threads") != 2 or any(outer["scientific_result"].get(key) != value for key, value in receipt.items())):
        raise ValueError("Only exact successful CPU2 cost receipts are used")
    summaries = [row for row in receipt["summaries"] if row["phase"] == "timed"]
    if len(summaries) != 1 or summaries[0]["trials"] != 20:
        raise ValueError("Original20 timed trials required")
    return receipt, summaries[0]["median_seconds"]


def csv_text(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows({key: row.get(key, "") for key in fields} for row in rows)
    return stream.getvalue()


def add_training(inputs, row, run_folder=None, fit_result=None, parent="", additional=False):
    value = {key: row.get(key, "") for key in ("dataset", "role", "run_id", "search_max_epoch", "selected_refit_epoch", "refit_updates", "notes")}
    value.update(training_device="cpu" if row["accuracy_device"] == "cpu" else row["accuracy_device"],
                 parent_training_required=parent, additional_optimization_only=additional)
    if run_folder:
        for stage in ("search", "refit"):
            name = run_folder + "/" + stage + "/RECEIPT.json"
            if inputs.path(name).is_file():
                stage_receipt = inputs.json(name)
                for source, target in (("seconds", "seconds"), ("updates", "updates"), ("exposures", "exposures")):
                    value[stage + "_" + target] = stage_receipt.get(source, "")
    if fit_result is not None:
        value["fit_and_V_wall_seconds"] = fit_result.get("elapsed_wall_seconds", fit_result.get("elapsed_process_wall_seconds", ""))
    return value


def parent_cpu(inputs, inference_name, rescore_name, plan_name):
    plan = inputs.json(plan_name)
    inference = inputs.json(inference_name)
    rescore = inputs.json(rescore_name)
    for result, status in ((inference, "COMPLETE_FULL_CPU_V148"), (rescore, "PASS_FULL_CPU148_INDEPENDENT_FP64_RESCORE")):
        if (result.get("status") != status or result.get("error_class") is not None or result.get("postflight_error_class") is not None
                or result.get("diagnostic_run_id") != PARENT or result.get("full_cpu_plan_sha256") != inputs.bindings[plan_name]):
            raise ValueError("New full CPU148 parent and independent rescore must finish before report generation")
    if plan.get("status") != "SEALED_BY_ROOT":
        raise ValueError("Parent full CPU plan must be sealed")
    folder = NAME + "/parent_full_cpu/" + PARENT + "/V_select"
    manifest = inputs.json(folder + "/PREDICTION_MANIFEST.json")
    score_rows = inputs.csv(folder + "/SCORES.csv")
    science = rescore["scientific_result"]
    if (len(score_rows) != 1 or manifest.get("device") != "cpu" or manifest.get("full_CPU_inference_completed") is not True
            or manifest["score"]["n_origins"] != 148 or manifest["score"]["n_observed"] != 301920
            or inputs.bindings[folder + "/PREDICTION_MANIFEST.json"] != inference["scientific_result"]["full_CPU_manifest_sha256"]
            or science["source_manifest_sha256"] != inputs.bindings[folder + "/PREDICTION_MANIFEST.json"]):
        raise ValueError("Actual same-device full V148 manifest binding differs")
    for metric in ("mse", "mae"):
        if not math.isclose(science[metric], manifest["score"][metric], rel_tol=1e-7, abs_tol=1e-9) or float(score_rows[0][metric]) != manifest["score"][metric]:
            raise ValueError("New CPU148 parent score/rescore mismatch")
    return manifest, inference, rescore


def assemble(inputs, parent_inference, parent_rescore, parent_plan):
    parent, parent_result, parent_check = parent_cpu(inputs, parent_inference, parent_rescore, parent_plan)
    seed_audit = inputs.json(NAME + "/INDEPENDENT_SEED2_REVIEW_ROOT_v1.json")
    pems_audit = inputs.json(NAME + "/INDEPENDENT_PEMS_REVIEW.json")
    parent_audit = inputs.json(NAME + "/INDEPENDENT_PARENT_CPU_REVIEW_ROOT_v1.json")
    if any(not value["status"].startswith("PASS") for value in (seed_audit,pems_audit,parent_audit)):
        raise ValueError("Completed independent seed2 and PeMS reviews required")
    for name in (P1 + "/INDEPENDENT_R5_REVIEW.json", P1 + "/INDEPENDENT_LEGACY_REVIEW.json"):
        if not inputs.json(name)["status"].startswith("PASS"):
            raise ValueError("P1 independent checks must pass")
    rows, training = [], []
    old = inputs.csv(P1 + "/PRIORITY_RESULTS.csv")
    if len(old) != 11:
        raise ValueError("Preserve all eleven P1 comparison rows")
    lag_roles = {"R3_B0", "R4_B2", "AUX_LAG_CAUSAL", "AUX_LAG_BIDIR"}
    for source in old:
        row = dict(source, dataset="jena", seed=92601, L=512, H=48, C=21,
                   accuracy_device="cuda" if source["role"] in {"R0_Raw", "R2_PCA", "R6_LoRA"} else "cpu",
                   accuracy_scope="REUSED_COMPLETE_V43", cost_device="cpu", cost_scope="EXACT_CPU2_RECEIPT_REUSED",
                   training_status="CONVERGENCE_UNRESOLVED")
        if source["role"] == "R0_Raw":
            row.update(cost_scope="FAILED_RAW_ROW_PARITY_NO_TIMED_COST", B1_median_seconds="", B8_median_seconds="")
        if source["role"] in {"R2_PCA", "R6_LoRA", "R0_Raw"}:
            row["notes"] += "; saved CUDA V; CPU6 subset proof does not replace full-V device"
        rows.append(row)
        folder = (LAG if source["role"] in lag_roles else P1 if source["role"] in {"R5a", "R5b"} else "")
        folder = (folder + "/runs/" + source["run_id"]) if folder else "runs/" + source["run_id"]
        fit_result = inputs.json(P1+"/diagnostics/"+source["role"].lower()+"_fit_root_v1/RESULT.json") if source["role"] in {"R5a","R5b"} else None
        item = add_training(inputs,row,folder,fit_result,parent="R4_B2 final P/E/D" if source["role"] == "R5a" else "",
                            additional=source["role"] == "R5a")
        if source["role"] in {"R0_Raw","R2_PCA"}:
            item.update(training_device="NO_NEW_DATASET_OPTIMIZER",search_updates=0,refit_updates=0)
        training.append(item)
    for family, b in (("B0", 0), ("B2", 2)):
        tag = "seed2_jena_b" + str(b)
        fitted = completed(inputs, tag + "_fit_root_v1", "COMPLETE_FIT_AND_FULL_V")
        completed(inputs, tag + "_rescore_root_v1", "PASS_INDEPENDENT_FULL_V_FP64")
        completed(inputs, tag + "_replay_root_v1", "PASS_FRESH_CPU6_REPLAY")
        receipt, score = fitted["scientific_result"]["fit_receipt"], fitted["scientific_result"]["V_score"]
        row = dict(dataset="jena",role=family + "_SEED92602",run_id=receipt["run_id"],seed=92602,L=512,H=48,C=21,K=6,
                   MSE=score["mse"],MAE=score["mae"],selected_refit_epoch=receipt["selected_epoch"],accuracy_device="cpu",
                   accuracy_scope="NEW_FULL_V43_FP64_AND_CPU6",cost_device="cpu",cost_scope="NEW_FRESH_CPU2_PROTOCOL",
                   V_origins=43,observed_labels=43344,search_max_epoch=3,refit_updates=receipt["refit_updates"],
                   training_status=receipt["training_status"],notes="Same repeated V43; new training seed; capacity/epochs differ between B0 and B2")
        for batch in (1,8):
            _,row["B"+str(batch)+"_median_seconds"] = cost(inputs,tag+"_cost_b"+str(batch)+"_root_v1")
        rows.append(row)
        training.append(add_training(inputs,row,NAME+"/runs/"+receipt["run_id"],fitted))
    mapping = inputs.json(NAME + "/PEMS_BASELINE_REUSE.json")
    for baseline in mapping["baselines"] + [mapping["separate_existing_G_reference"]]:
        train = baseline["training"]
        role = "B2_PARENT_GPU_REFERENCE" if baseline["role"] == "B2_PARENT" else baseline["role"]
        row = dict(dataset="pems08",role=role,run_id=baseline["run_id"],seed=92601,L=512,H=12,C=170,K=baseline["effective_F_input_width"],
                   MSE=baseline["MSE"],MAE=baseline["MAE"],selected_refit_epoch=train.get("selected_epoch", ""),
                   accuracy_device="cpu" if baseline["role"] in {"RIDGE","G_CPU_FRESH_THREE_EPOCH_REFERENCE"} else "cuda",
                   accuracy_scope="REUSED_SAVED_V148_REFERENCE",cost_device="",cost_scope="NO_VALID_CPU_COST_NO_CUDA_SUBSTITUTION",
                   V_origins=148,observed_labels=301920,search_max_epoch=max(train.get("epochs",[0])),
                   refit_updates=train.get("refit_updates",0),training_status=train.get("status","NO_NEW_FIT"),
                   notes="Reference only; different training budgets and accuracy devices; no global same-device ranking")
        for old_cost in baseline.get("CPU2_costs",[]):
            bound = old_cost["receipt"]
            inputs.capture(bound["relative_path"])
            if inputs.bindings[bound["relative_path"]] != bound["sha256"]:
                raise ValueError("Old CPU cost receipt binding changed")
            row["B"+str(old_cost["batch_size"])+"_median_seconds"] = old_cost["median_seconds"]
            row.update(cost_device="cpu",cost_scope="REUSED_BOUND_CPU2_SOURCE_THREADS_NOT_RECEIPT_FIELD")
        rows.append(row)
        item = add_training(inputs,row)
        for key in ("search_updates","refit_updates","search_exposures","refit_exposures"):
            item[key] = train.get(key,0 if train.get("optimizer_updates")==0 else "")
        item["training_device"] = train.get("source_fit_device", "cpu" if baseline["role"]=="RIDGE" else "NO_NEW_FIT")
        training.append(item)
    score = parent["score"]
    row = dict(dataset="pems08",role="B2_PARENT_CPU_FULL",run_id=PARENT,seed=92601,L=512,H=12,C=170,K=43,
               MSE=score["mse"],MAE=score["mae"],selected_refit_epoch=1,accuracy_device="cpu",accuracy_scope="NEW_FULL_CPU_V148_AND_INDEPENDENT_FP64",
               cost_device="cpu",cost_scope="EXACT_SAME_PARENT_NEW_CPU2_COST",V_origins=148,observed_labels=301920,search_max_epoch=1,
               refit_updates=637,training_status="LIMITED_SCREEN_ONE_EPOCH_PARENT",
               notes="Same existing parent checkpoint; new CPU148 inference, no new parent training; old CUDA parent stays separate")
    for batch in (1,8):
        _,row["B"+str(batch)+"_median_seconds"] = cost(inputs,"pems_parent_cost_b"+str(batch)+"_root_v1")
    rows.append(row)
    item = add_training(inputs,row,parent="Existing primary B2 __e1 checkpoint; its original training is recorded in GPU reference row")
    item.update(training_device="NO_NEW_FIT_CPU148_INFERENCE_ONLY",search_updates=0,refit_updates=0,search_exposures=0,refit_exposures=0,
                full_V_only_wall_seconds=parent_result["elapsed_process_wall_seconds"])
    training.append(item)
    fitted = completed(inputs,"pems_fit_root_v1","COMPLETE_FIT_AND_FULL_V")
    completed(inputs,"pems_rescore_root_v1","PASS_INDEPENDENT_FULL_V_FP64")
    completed(inputs,"pems_replay_root_v1","PASS_FRESH_CPU6_REPLAY")
    receipt,score = fitted["scientific_result"]["fit_receipt"],fitted["scientific_result"]["V_score"]
    row = dict(dataset="pems08",role="FIXED_B2_NO_BOLT_HEAD",run_id=receipt["run_id"],seed=92601,L=512,H=12,C=170,K=43,
               MSE=score["mse"],MAE=score["mae"],selected_refit_epoch=3,accuracy_device="cpu",accuracy_scope="NEW_FULL_CPU_V148_FP64_CPU6",
               cost_device="cpu",cost_scope="NEW_FRESH_CPU2_PROTOCOL",V_origins=148,observed_labels=301920,search_max_epoch="",
               refit_updates=1911,training_status="FIXED_THREE_EPOCH_HEAD_NO_CONVERGENCE_CLAIM",
               notes="Exact parent final P/E/D fixed; full512x43 linear head adds3A60epochs; no separate joint no-Bolt training")
    for batch in (1,8):
        _,row["B"+str(batch)+"_median_seconds"] = cost(inputs,"pems_cost_b"+str(batch)+"_root_v1")
    rows.append(row)
    item = add_training(inputs,row,fit_result=fitted,parent="B2_PARENT_CPU_FULL: original parent training is additional prerequisite",additional=True)
    item.update(refit_seconds=receipt["refit_summary"]["seconds"],refit_updates=1911,refit_exposures=30570)
    training.append(item)
    return rows,training,parent,parent_result,parent_check


def report(rows,parent,parent_inference,parent_rescore):
    def row(dataset,role):
        return next(value for value in rows if value["dataset"]==dataset and value["role"]==role)
    def number(value):return format(float(value),".9g")
    pairs=[(92601,row("jena","R3_B0"),row("jena","R4_B2")),(92602,row("jena","B0_SEED92602"),row("jena","B2_SEED92602"))]
    p,h=row("pems08","B2_PARENT_CPU_FULL"),row("pems08","FIXED_B2_NO_BOLT_HEAD")
    pair_lines="\n".join(f"{seed}  {number(a['MSE']):>12} {number(b['MSE']):>12} {number(a['MAE']):>12} {number(b['MAE']):>12}  {a['selected_refit_epoch']}/{b['selected_refit_epoch']}" for seed,a,b in pairs)
    values="\n".join(f"{r['dataset']:<7} {r['role']:<34} {number(r['MSE']):>12} {number(r['MAE']):>12} {r['accuracy_device']}" for r in rows)
    costs="\n".join(f"{r['dataset']:<7} {r['role']:<34} {str(r.get('B1_median_seconds','') or 'NOT_MEASURED'):<24} {r.get('B8_median_seconds','') or 'NOT_MEASURED'}" for r in rows if r.get("cost_device")=="cpu")
    mse_change=(float(h["MSE"])/float(p["MSE"])-1)*100
    ratios=[float(p["B"+str(b)+"_median_seconds"])/float(h["B"+str(b)+"_median_seconds"]) for b in (1,8)]
    return f"""# B2 우선 검증 후속: Jena 두 seed와 PeMS 전체 CPU 비교

**[확인] 교수님께 설명할 방법론 후보의 구현은 검증됐고, Jena에서는 두 학습 seed 모두에서 B2의 MSE가 B0보다 낮았다. 다만 강한 직접 예측·적응 기준선 전체를 앞선 방법은 확보하지 못했다.** PeMS의 새 전체 CPU148 부모와 고정 no-Bolt head 비교까지 끝냈으며, 이 문서는 남은 MASTER 전체 완료 보고가 아니다.

1. 구조는 맞는가: 허용 과거512→시간·채널 앞단 P→C에서K로 E→동결 Bolt→lag D+현재값 경로, 실제 K streams·F 입력 역전파·동결·미래 경계를 확인했다. [P0 근거](../{P1}/diagnostics/p0_root_v1/RESULT.json). B0도 F와 D를 통해512과거를 사용한다. 현재값 우회가 있으므로 모든 정보를 오직K로만 전달하는 통신압축은 아니다.
2. B0에 무엇을 더했는가: 아래 두 seed에서 MSE가 낮았고 선택 refit은 B0/B2 모두3/1epoch였다. B2의 추가 파라미터와 학습 구조가 함께 달라 **용량을 맞춘 시간혼합의 인과 효과**로 해석하지 않는다. MAE 이득의 크기도 seed에 따라 다르다.
3. Bolt를 붙일 근거는 있는가: Jena R4는 같은 최종 P/E/D의 전체512K 선형 head R5a와, 동형 구조를 새로 학습한 no-F R5b보다 MSE가 낮다. 그러나 native Ridge·CPU fresh G·제한 학습 LoRA의 V 수치보다 높아 전체 대안 대비 필요성은 확정하지 못했다. LoRA/Raw/PCA의 옛 전체 V 장치는CUDA이며 CPU6 subset과CPU비용을 전체CPU 정확도로 바꾸지 않는다.
4. 전체 비용은 어떤가: PeMS 같은 CPU148에서 부모 MSE {number(p['MSE'])}, 고정 head {number(h['MSE'])}로 head가 {mse_change:.4f}% 높다. CPU B1/B8 부모/head 시간비는 각각 {ratios[0]:.4f}/{ratios[1]:.4f}다. 학습시간·모델 적재·정밀추론은 분리하며 전체 시스템의 무조건적 비용 우위로 확대하지 않는다.
5. 후순위는 무엇인가: 먼저 이 진단을 근거로 정확도·학습·배포비용 중 목표를 좁혀야 한다. 승인된 기존 후속은 원예산·T잠금 안에서만 별도로 실행한다. nested 비용6개/Beijing 최소3조건은 별도 준비이며 기존 미시작13/core29/M5·추가자료·K·학습량 등 전체 잔여를 대체하지 않는다.

[원정밀도 전체 결과](RESULTS_v2.csv) · [학습 비용·학습량](TRAINING_COST_v2.csv) · [입력 SHA와 출력 결속](REPORT_INPUT_BINDINGS_v2.json) · [재현 코드](build_followup_report.py)

![Jena 두 seed와 PeMS CPU 비교](figures/FOLLOWUP_SUMMARY_v3.svg)

## Jena: 두 학습 seed, 같은 개발 V43

```text
seed       B0 MSE       B2 MSE       B0 MAE       B2 MAE  selected refit B0/B2
{pair_lines}
```

L512/H48/C21/K6, A50[0,10482), A_stop[10482,12579), fresh A60[0,12579), V[14675,16772)의 43 origins·43,344 관측 정답과 동일한 scaler/target/mask/channel을 유지했다. 검색은 0/.25/1/3epoch, 선택은 A_stop이며 V를 checkpoint 선택에 쓰지 않았다. seed92602도 B0 search1863/refit2254update, B2 search1863/refit752update다. B0의 마지막 선택 3epoch와 B2의 선택 1epoch는 수렴 검증이 아니다. 두 seed와 반복해서 본 V를 독립시험·유의성 검정으로 부르지 않는다.

R5a는 **R4 최종 P/E/D와 scaler를 고정**하고 전체 Z[512,6]를 공유 Linear(512,48)에 넣어 A60 고정 3epoch 동안 head만 학습했다. R5b는 같은 형태의 P/E/head/D를 fresh로 함께 학습하고 A_stop으로 선택한 별도 가중치다. 부모 표현을 얻는 학습 비용은 R5a head의 추가 비용과 별도다. 이미 미래 MSE 목적을 사용했으므로 현재 실패를 재구성 목표 때문이라고 설명하지 않는다.

## PeMS08: 새 부모 전체 CPU148이 주 비교

새 CPU 부모는 기존 `primary_pems08_E1_B2_k43_s92601__e1`의 동일 checkpoint를 재학습 없이 전체 148 origins에서 추론하고 별도로 FP64 검산했다. H12/C170/K43, A50end8928/A60end10713, V[12499,14284), 301,920 정답과 scaler94e8c71c…69c60을 head와 맞췄다. 실제 원자료의 앞 14284행/feature0만 수치 변환했고 보호 T 수치는 파싱하지 않았다. 옛 GPU 부모 V는 별도 reference 행이다. GPU/CPU 출력 비교 상태는 `{parent['GPU_comparison']['status']}`이며, 새 CPU 예측·독립 재점수의 완료와 구분한다.

부모는 검색 0/.25/1epoch·선택 1epoch의 제한 학습이다. 직접 head는 그 최종 P/E/D를 고정하고 전체 Z[512,43]를 Linear(512,12)에 넣어 head 파라미터 6,156개만 추가 3A60epoch/1911update/30,570 노출로 학습했다. 부모 1epoch와 추가 head 3epoch의 학습량은 같지 않고, PeMS에서 별도의 joint no-Bolt 학습은 수행하지 않았다. 이 결과는 고정 Bolt용 표현/고정 D 아래의 선형 head 진단이며 Bolt의 보편적 필요성이나 표현 학습 방법의 일반 한계를 확정하지 않는다.

## 실제 V 수치와 장치

```text
dataset role                                        MSE          MAE accuracy device
{values}
```

CSV는 원정밀도를 유지한다. CUDA와 CPU 전체 정확도 행을 섞어 동일 장치 전체 순위를 만들지 않는다. 기존 G 1epoch와 CPU fresh G 3epoch, LoRA .25epoch도 서로 다른 학습량이다. native Ridge는 32개 시간 블록 평균+현재값을 쓰는 별도 FP64 닫힌해이며 모든 ridge 대안을 대표하지 않는다. Ridge는 closed-form/no optimizer로 다루며 신경망의 수렴 미확정 판정 대상이 아니다.

## CPU 전체 경로 추론과 학습 비용

```text
dataset role                               CPU B1 seconds/batch     CPU B8 seconds/batch
{costs}
```

원 physical 입력·관측 mask→과거 채움/정규화→P/E→F 또는 직접 head→D→FP64 역변환/CPU 반환을 실제 측정했다. CPU2, B1/B8, 10warm/20timed/별도20stage 조건이다. cached Z/forecast는 쓰지 않았다. stage는 계측이 추가된 별도 범위며 stage 합이 whole과 같다고 하지 않는다. 부모/표현 학습·head 추가 학습·model load·steady-state 추론을 총비용 하나로 더하지 않는다. [학습CSV](TRAINING_COST_v2.csv)의 빈 시간은 미기록이며 0이 아니다. 원 장부에서 다른 장치·동시 실행으로 측정된 학습시간은 인과적 속도 비교가 아니다.

Jena Raw의 원 row-parity 실패는 미해결이라 timed 비용을 채우지 않았다. PeMS Raw/G/LoRA의 옛 CUDA 비용을 CPU 열에 복사하지 않았다. PeMS native Ridge의 기존 CPU 영수증은 thread 필드가 없고 pinned 실행 source의 2threads를 근거로 한다. CPU active_adapter_MiB의 0은 GPU 전용 계수의 0이며 CPU 어댑터 메모리 0을 뜻하지 않는다.

## 검증 범위와 다음 판단

[새seed2독립검산](INDEPENDENT_SEED2_REVIEW_ROOT_v1.json)은 공개 5,418통계행·86,688label과 200costtrial/8summary를 재집계했고 기존 actual CPU6 및 원수치 FP64 검산을 결속했다. [PeMS head와부모CPU6/비용](INDEPENDENT_PEMS_REVIEW.json), [기존baseline](PEMS_BASELINE_REUSE.json), [새부모전체CPU추론]({parent_inference})와 [별도전체FP64검산]({parent_rescore})을 구분한다. [새 부모 전체 CPU의 독립 충분통계 검산](INDEPENDENT_PARENT_CPU_REVIEW_ROOT_v1.json)도 별도 근거다. 단순 검사 PASS를 성능 개선이나 새 일반화로 해석하지 않는다.

확정된 것은 구현 경계와 조건부 비용·오차 차이이다. 미확정은 추가 앞단이 작은 K에서 강한 직접 예측보다 유리한 조건, 충분 학습 후의 안정성, 미노출 구간 일반화와 학술적 신규성이다. 현재 근거로는 B2→K→frozenBolt→lagD+anchor를 **검증 중인 후보**로 설명하고, 전체 우월성을 기여로 내세우지 않는다. 원자료·checkpoint·보호 T를 재선택하거나 G/H·새 graph로 주방법을 전환하지 않는다.
"""


def execute(public_root, inference_name, rescore_name, plan_name):
    inputs = Inputs(public_root)
    scope = inputs.path(NAME)
    if any((scope/name).exists() for name in OUTPUTS):
        raise FileExistsError("Preserve existing v2 reports and README; fresh outputs are required")
    inputs.capture(NAME + "/build_followup_report.py")
    rows,training,parent,_,_ = assemble(inputs,inference_name,rescore_name,plan_name)
    if not inputs.preserved():
        raise ValueError("Report inputs changed before output generation")
    # Markdown links are relative to this report's namespace.
    parent_inference = str(Path(inference_name).relative_to(NAME)).replace("\\","/")
    parent_rescore = str(Path(rescore_name).relative_to(NAME)).replace("\\","/")
    texts = {"RESULTS_v2.csv":csv_text(rows,FIELDS),"TRAINING_COST_v2.csv":csv_text(training,TRAINING_FIELDS),
             "RESULTS_KO_v2.md":report(rows,parent,parent_inference,parent_rescore),"README_KO.md":"# B2 우선 검증 후속 결과\n\n[최종 한국어 결과](RESULTS_KO_v2.md) · [원수치](RESULTS_v2.csv) · [학습량·학습비용](TRAINING_COST_v2.csv) · [입력 결속](REPORT_INPUT_BINDINGS_v2.json)\n\n![결과 요약](figures/FOLLOWUP_SUMMARY_v3.svg)\n\nJena 두 seed와 PeMS 동일 장치 전체 V148을 다룬 개발 결과다. 원 MASTER 전체 완료·독립 일반화·수렴을 주장하지 않는다. 기존 [PeMS v1 snapshot](PEMS_RESULTS_KO_v1.md)과 [P1 보고](../"+P1+"/INFORMS_METHOD_STATUS_KO.md)는 당시 범위를 보존한다.\n\n재현은 stdlib [builder](build_followup_report.py)의 명시 --execute를 사용한다. 기본 호출은 입력읽기·쓰기0이다. 기존 v2 출력이 있으면 덮지 않고 거부하며, 새 부모 전체CPU 추론과 별도FP64 rescore PASS가 없으면 결과를 만들지 않는다. 그림 생성과 게시 검사는 ROOT의 별도 단계다.\n"}
    texts["README_KO.md"] += "\n새 출력이 없는 checkout의 저장소 root에서 실행 예:\n\n```powershell\npython -B research/informs_broad_validation_20261015_v1/priority_core_followup_20261016_v1/build_followup_report.py --execute --public-root research/informs_broad_validation_20261015_v1 --parent-plan priority_core_followup_20261016_v1/PARENT_FULL_CPU_PLAN.json --parent-inference-result priority_core_followup_20261016_v1/diagnostics/pems_parent_full_cpu_inference_root_v1/RESULT.json --parent-rescore-result priority_core_followup_20261016_v1/diagnostics/pems_parent_full_cpu_rescore_root_v1/RESULT.json\n```\n"
    if not inputs.preserved():
        raise ValueError("Input bytes changed while composing report")
    bindings = dict(status="REPORT_BUILT_FROM_COMPLETED_PUBLIC_V_RECEIPTS",input_count=len(inputs.bindings),
                    input_SHA_before_after_equal=True,raw_targets_NPZ_checkpoint_model_reads=0,new_training_inference_cost_process_Git_operations=0,
                    inputs=[dict(relative_path=name,sha256=digest) for name,digest in sorted(inputs.bindings.items())],
                    outputs=[dict(relative_path=NAME+"/"+name,sha256=hashlib.sha256(text.encode()).hexdigest()) for name,text in texts.items()],
                    figure=dict(relative_path=NAME+"/figures/FOLLOWUP_SUMMARY_v3.svg",owner="ROOT",creation_not_claimed_by_builder=True),
                    result_rows=len(rows),Jena_rows=sum(row["dataset"]=="jena" for row in rows),PeMS_rows=sum(row["dataset"]=="pems08" for row in rows),
                    independently_unseen_evaluation=False,original_MASTER_complete=False)
    texts["REPORT_INPUT_BINDINGS_v2.json"] = json.dumps(bindings,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    for name,text in texts.items():
        with (scope/name).open("x",encoding="utf-8",newline="\n") as stream:
            stream.write(text)
    return dict(status="REPORTS_CREATED_FIGURE_AND_PUBLICATION_SEPARATE",rows=len(rows),files_written=len(texts),input_count=len(inputs.bindings))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    for name in ("public-root","parent-inference-result","parent-rescore-result","parent-plan"):
        parser.add_argument("--"+name)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="METADATA_ONLY_NOT_EXECUTED",input_reads=0,files_written=0,scientific_or_child_operations=0)))
        return 0
    if not all((args.public_root,args.parent_inference_result,args.parent_rescore_result,args.parent_plan)):
        parser.error("Explicit --execute public root, relative successful parent inference/rescore RESULT paths and sealed parent PLAN required")
    result = execute(args.public_root,args.parent_inference_result,args.parent_rescore_result,args.parent_plan)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
