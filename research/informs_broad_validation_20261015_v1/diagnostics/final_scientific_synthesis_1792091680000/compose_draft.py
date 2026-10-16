"""Compose a draft and exact figure inputs from existing metadata CSVs only."""
import csv
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):value.update(block)
    return value.hexdigest()


def new(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        if isinstance(value, str):stream.write(value)
        else:json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False);stream.write("\n")


def rows(path):return list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))


def csv_new(path, values):
    fields = sorted(set().union(*(row.keys() for row in values)))
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(values)


def main():
    output=Path(__file__).resolve().parent
    public=output.parents[1]
    inventory=rows(output/"PRIMARY_V_METADATA_INVENTORY.csv")
    indexed={row["run_id"]:row for row in inventory}
    matched=rows(output/"SAME_G_E2_REPORTED.csv")
    keyed={(row["dataset"],row["method"]):row for row in matched}
    matrix=rows(output/"PROFESSOR_BASELINE_MATRIX.csv")
    pins=json.loads((output/"SOURCE_PINS.json").read_text(encoding="utf-8"))
    extra_inputs={}
    def read(path):
        relative=path.relative_to(public).as_posix()
        extra_inputs[relative]=dict(sha256=sha(path),bytes=path.stat().st_size)
        return json.loads(path.read_text(encoding="utf-8-sig"))
    roles=[]
    for dataset,k,suffix in (("beijing",3,"__e1__e3"),("jena",6,"__e1"),("electricity",81,"__e1"),("pems08",43,"__e1")):
        p0=indexed[f"primary_{dataset}_E1_B0_k{k}_s92601{suffix}"]
        p2=indexed[f"primary_{dataset}_E1_B2_k{k}_s92601{suffix}"]
        for method in ("H_B0","H_B2","Hpast","RawH","ALPHA_FIT"):
            row=keyed[dataset,method]
            identity=row["run_id"]
            receipt=read(public/"combinations"/identity/"receipt.json")
            roles.append(dict(dataset=dataset,role=method,run_id=identity,mse=float(row["mse"]),mae=float(row["mae"]),G_run_id=row["G_run_id"],G_model_sha=row["G_model_sha"],
                parent_B0_run_id=p0["run_id"],parent_B2_run_id=p2["run_id"],parent_B0_mse=float(p0["mse"]),parent_B2_mse=float(p2["mse"]),parent_B2_minus_B0=float(p2["mse"])-float(p0["mse"]),
                parent_B0_selected_epoch=p0["selected_epoch"],parent_B2_selected_epoch=p2["selected_epoch"],head_selected_updates=receipt.get("selected_updates"),head_total_updates=receipt["updates"],
                head_selected_state_equals_initial=receipt.get("selected_state_sha")==receipt.get("initial_state_sha") if "selected_state_sha" in receipt else "not_neural_head",
                head_selected_parameter_max_abs_change=receipt.get("selected_parameter_max_abs_change"),alpha_M_fit=receipt.get("alpha_fit",{}).get("alpha_fit"),training_status=receipt["training_status"],
                source_receipt_relative_path=(public/"combinations"/identity/"receipt.json").relative_to(public).as_posix(),source_receipt_sha256=sha(public/"combinations"/identity/"receipt.json"),
                source_mse_csv="diagnostics/raw_deep_g_saved_finalize_1792091230000/MATCHED_G_COMPARISON.csv",source_mse_csv_sha256="43c2970c899301ee20043eae0c598e58a78c55bd0e5d347671e724ed7e46fde8"))
    csv_new(output/"PARENT_AND_HEAD_ROLE_EVIDENCE.csv",roles)
    e4=[]
    for dataset,ks,C in (("beijing",(3,6,11),11),("electricity",(81,161,321),321)):
        for k in ks:
            ids={"independent_B0":f"primary_{dataset}_E1_B0_k{k}_s92601",
                 "shared":f"primary_{dataset}_E4_ORDERED_MULTI_K_k{C}_s92601"+(f"__prefix_k{k}" if k!=C else ""),
                 "posthoc":f"primary_{dataset}_E4_POSTHOC_PREFIX_k{k}_s92601","nested_PCA":f"primary_{dataset}_E4_NESTED_PCA_k{k}_s92601"}
            for method,identity in ids.items():
                item=indexed[identity]
                e4.append(dict(dataset=dataset,K=k,method=method,run_id=identity,mse=float(item["mse"]),mae=float(item["mae"]),selected_epoch=item["selected_epoch"],attempted_epoch_endpoints=item["attempted_epoch_endpoints"],
                    source_manifest_relative_path=item["source_manifest_public_relative_path"],source_manifest_sha256=item["source_manifest_sha256"],parent_or_alias_run_id=item["parent_or_alias_run_id"],
                    training_source_files_match_current=item["training_source_files_match_current"],source_or_initialization_differences_require_caution=True,not_independent_fit_per_prefix=method in {"shared","posthoc","nested_PCA"},
                    actual_switching_cost_status="PENDING_ROOT_FROZEN_PROTOCOL",T_status="NOT_EXECUTED"))
    csv_new(output/"E4_PREDECLARED_K_COMPARISON.csv",e4)
    by_label={(row["table_label"],row["dataset"]):row for row in matrix}
    labels=("RawSmall","RawMini","RawTiny","Ridge_A60","G_initial025","G_GPU_to1","G_GPU_to3","G_CPU_fresh3","G_AplusM025","G_AplusM3","LoRA025","LoRA_GPU3","LoRA_AplusM025","AdaPTS_port025","AdaPTS_port3")
    baseline_table="```text\nCondition           Beijing      Jena         Electricity  PeMS08\n"
    for label in labels:
        baseline_table+=f"{label:20}"+" ".join(f"{float(by_label[label,ds]['mse']):.9f}" if by_label[label,ds].get("mse") else "NA         " for ds in ("beijing","jena","electricity","pems08"))+"\n"
    baseline_table+="```\n"
    same_table="```text\nDataset       G           Hpast       H_B0        H_B2        RawH        alpha_Mfit\n"
    for ds in ("beijing","jena","electricity","pems08"):
        same_table+=f"{ds:12}"+" ".join(f"{float(keyed[ds,m]['mse']):.9f}" for m in ("G","Hpast","H_B0","H_B2","RawH","ALPHA_FIT"))+"\n"
    same_table+="```\n"
    draft="""# 최종 과학 해석 초안 — V 증거와 남은 판단

[확인] 현재 방어 가능한 결론은 **같은 G에서 보조 예측과 H가 일부 V 오차를 줄이지만, 시간 앞단 B2만의 추가 가치와 강한 실제 대안 대비 정확도–비용 이점은 아직 입증되지 않았다**는 것이다. 단일 결과를 모두 실패 또는 모두 해결이라고 묶을 수 없다. 이 초안은 현존 JSON/CSV metadata만 읽었다. 배열·weights·gzip 해제·모델·GPU·T·새 통계 재계산·자동 모델 선택은 0이다.

아래는 A60 정규화 observed 원채널 macro MSE다. MAE·N·H/C·실제 ID·source score 위치·manifest/receipt hash는 [전체 V 원수치](PRIMARY_V_METADATA_INVENTORY.csv), [강한 대조군](STRONG_BASELINES_V.csv), [대표 표](PROFESSOR_BASELINE_MATRIX.csv)에 일대일로 남겼다. 자료 간 평균이나 유의성을 계산하지 않았다. 이것은 최종 T 후보 잠금 또는 모델 순위 결정이 아니다.

## 같은 G를 유지했을 때 달라지는 것

"""+same_table+"""
[확인] H들은 같은 자료의 실제 G artifact/model, seed92601, lr0.001, epoch endpoint0.25/1/3, effective batch16·microbatch4, head 용량과 초기 state를 공유한다. RawH는 원 C의 RawSmall 예측, H_B0/H_B2는 압축 부모 예측, Hpast는 동용량의 과거 대조다. 부모는 A-only로 고정되고 H는 M_cal로 학습하며 V에서 checkpoint를 고른다. Alpha는 M의 analytic 계수이며 neural head와 같은 최적화 과정이 아니다. Alpha0/1 alias를 독립 모델·학습으로 세지 않는다. 모든 차이는 [쌍별 CSV](SAME_G_PAIR_DELTAS.csv), 학습 단계는 [부모/헤드 증거](PARENT_AND_HEAD_ROLE_EVIDENCE.csv)에 있다.

[확인] H_B2−H_B0 MSE는 Beijing +0.009172789, Jena 0, Electricity 0, PeMS +0.001024767이다. 이 **고정 G+H 조건의 원채널 MSE**에서는 B2 추가 개선이 없다. PeMS MAE는 H_B2가 H_B0보다 0.000695267 작으므로 지표별 상충도 남긴다. 주지표를 바꿔 B2가 이겼다고 쓰지 않는다.

[확인] H_B2는 G와 Hpast보다 Beijing·Electricity·PeMS에서 낮지만 Jena에서는 같다. RawH는 H_B2보다 Beijing/Jena/Electricity에서 낮고 PeMS에서는 높다. 단순 alpha_Mfit은 Jena에서 H3종보다 낮다. 따라서 H의 추가 예측 활용이 존재한다는 관찰과 압축·시간 앞단이 필요하다는 주장은 다르다.

### 표현, H 학습과 정보 부족을 구분해야 하는 이유

[확인] Jena 부모 B2의 V MSE는 0.686401862로 B0 0.803309276보다 낮다. 그러나 H_B0/H_B2/Hpast는 모두 selected_updates0·selected state=initial state·선택 parameter 변화0이며 실제 출력도 G와 같았다. RawH는129 update checkpoint, alpha_Mfit은 analytic 계수를 사용한다. 즉 Jena에서 'B2가 아무 도움도 없었다'는 보편 명제는 틀리고, **부모 예측의 차이가 해당 H 선택 결과로 이어지지 않았다**가 정확하다.

[확인] Electricity의 B0/B2 부모는 모두 selected epoch0이고 출력도 같다. 뒤의 H_B0/H_B2도 같은 출력을 냈다. 초기값을 고른 조건으로 시간 P를 충분히 학습한 구조의 한계를 확정할 수 없다. 반대로 실제 개선이 없던 것도 숨기지 않는다. Beijing 부모 B2는 B0보다 낮지 않고, PeMS 부모에서는 B2가 B0보다 조금 낮지만 결합 후 주 MSE 순서는 반대다.

[추정] 경쟁 설명은 (1) 초기값·V checkpoint 선택으로 학습된 변형이 채택되지 않음, (2) G가 이미 설명한 오차와 b-g의 중복, (3) M과 V 시점 차이, (4) H 목적·용량·최적화가 유용한 차이를 살리지 못함이다. 현재 저장 수치만으로 어느 하나를 원인으로 확정하거나 '표현의 정보가 없다/디코더가 원인/대조학습이 필요'라고 결론낼 수 없다. 현재의 예측용 학습 비교 결과를 단순 재구성 학습의 실패로 설명하지 않는다.

## 더 강한 실제 대안이 남기는 반례

"""+baseline_table+"""
NA는 해당 이름의 실제 V가 없는 상태다. epoch3 요청과 selected epoch3은 다르다. 전체 ID·시도 endpoint·선택 epoch·학습구간은 [조건 설명](BASELINE_CONTRACTS_KO.md)과 CSV에 있다.

[확인] Beijing에서 A-only LoRA(심화 시도 후 선택 epoch0.25)는 0.735095062, A+M LoRA0.25는0.731697680이다. 새 H_B0/H_B2/RawH보다 낮다. Beijing E3의 full C=11 bidirectional lag는0.712109361로 강한 반례다. 후자는 무압축 K=C 조건이며, 더 낮은 점수 자체를 경량화 성과라고 부를 수 없다.

[확인] Jena native ridge는0.556646336으로 새 alpha_Mfit0.582534731·H/RawH보다 낮다. Ridge는 32 temporal block 평균+현재값 기준·intercept의 FP64 closed form이며, A_stop alpha 선택 후 A60 전체 합법 origin을 재적합했다. 0.25 epoch 신경망과 학습 노출이 같지 않지만, 실제 사용 대안으로 제외할 수 없다. 새 CPU G3은0.632217057로 GPU G1의0.603786477보다 낮지 않다. CPU backend·새 초기 학습·노출량까지 달라 순수 epoch 효과로 해석하지 않는다.

[확인] Electricity RawSmall0.126620800은 RawH0.139925688·H_B0/H_B2 0.163089911보다 낮다. RawTiny0.140791952와 RawH 차이는 작고, A+M LoRA0.124233516이 더 낮다. 초기 A-only LoRA는 selected epoch0으로 RawSmall과 같다. 새 H로 오래된 약한 G를 이긴 사실만으로 이미 강한 Raw/LoRA 대비 가치를 주장할 수 없다.

[확인] PeMS H_B0 0.087727488·H_B2 0.088752255는 RawTiny0.094063095·CPU G3 0.099388470보다 낮다. 그러나 A-only LoRA0.084724874와 A+M LoRA0.083260132가 더 낮다. H_B0보다 H_B2가 MSE에서 낮지 않고, RawH도 H_B0/H_B2보다 낮지 않다. 정확도만으로 현재 최강 방법을 넘었다는 주장은 지지되지 않는다.

[확인] AdaPTS는 원 논문 전체 재현이 아닌 **Bolt 호환 원리 포트**다. 4자료 초기0.25 결과와 Beijing3 시도를 구분한다. Beijing3은0.833820258이고 last-best UNDERTRAINED 표시를 유지한다. Jena/Electricity/PeMS의3 epoch 포트는 현재 이 표에 없으며 원 논문의 한계라고 일반화하지 않는다.

### 학습·정보량 조건

[확인] A-only neural 부모는 A50 search/A_stop 선택 후 fresh A60 refit이다. A+M G/LoRA는 M까지 직접 학습하고 A60 scaler를 유지한 강한 대조이며 V로 checkpoint를 골랐다. H는 A-only 부모를 고정하고 M에서 별도 head를 학습한다. 둘은 허용 정보의 마지막 시점이 비슷해도 구조·노출·최적화 예산까지 동일한 것은 아니다. Raw는 optimizer0, ridge는 closed form이라 epoch 숫자로 맞추지 않는다.

[확인] CPU G3는 Jena/Electricity/PeMS의 별도 fresh ID이며 GPU의 optimizer/RNG 상태를 옮겨 이어 학습한 것이 아니다. Beijing에는 GPU G의3 epoch 시도가 있고 selected epoch은1이다. Beijing LoRA3도 selected epoch0.25다. `UNDERTRAINED`, `SCREENED_NOT_CONVERGENCE_PROVEN`, 초기 선택과 마지막 best를 그대로 보고한다. 초기 Beijing G의 source contract는 현재 fit 코드와 다르므로 초기→심화 차이를 순수 학습량 효과로 쓰지 않는다.

## E3 지연과 E4 가변 K에서 남는 범위

[확인] [E3/E4 전체 V](E3_E4_V_ALL_AVAILABLE.csv)는34행이다. E3의 Beijing/PeMS quarter·half·C는 실제0.25 screen이고, Beijing quarter3 시도2개를 추가로 분리한다. Beijing quarter bidirectional 0.856169258→0.805591160, causal0.884804018→0.841763779로 해당 시도의 V는 낮아졌다. 그러나 fullC0.25보다 높고,3 epoch의 충분 수렴도 아니다. PeMS K43/85/170 causal은0.107694027/0.101522145/0.097060570, bidirectional은0.108421464/0.102400352/0.096547317이다. 과거 창 안의 lag 변환 대조이지 directed graph 학습이나 인과 식별 증거가 아니다. Jena/Electricity E3 미실행은 0점이 아니라 미실행이다.

[확인] [E4 고정 K 비교](E4_PREDECLARED_K_COMPARISON.csv)는 독립B0·shared·posthoc·nested PCA를 K별로 모두 보존한다. Beijing K3 shared0.878905394/posthoc0.889539548보다 nested PCA0.797925078이 낮고, K6도 PCA0.779456612가 shared0.797958721보다 낮다. K=C에서는 shared0.739527657/posthoc0.737438411이 PCA0.751789975보다 낮다. 모든 K에 공유 학습의 이점이 있다고 말할 수 없다.

[확인] Electricity shared/posthoc는 K81/161/321에서0.211588720/0.175995295/0.138391702로 같고, Kmax 부모의 selected epoch0이다. 학습된 다중 K 순서화 효과가 입증됐다고 말할 수 없다. 초기화·PCA 기저와 독립B0의 차이, per-K 노출·총 update·source 차이도 CSV에 표시한다. Beijing3 shared/max 실행은 BUDGET_LIMIT이고 새 prefix V가 없으므로 초기0.25 prefix 결과를3 epoch 완료로 쓰지 않는다. 전환·저장량·실제 F stream/지연 비용은 ROOT의 진행 중 고정 프로토콜 측정을 기다린다.

## 교수님께 설명할 문제와 현재 기여의 범위

[확인] 현재 질문은 '복원 손실을 줄이는 압축기를 만들었는가'보다 **고정 단변량 TSFM을 다변량 예측에 연결할 때, 제한된 K의 예측이 원 C/간단한 native/내부 적응을 대신할 정확도–비용 근거가 있는가**에 가깝다. 같은 G·동용량 H·같은 부모 예측/정규화 계약을 통해 어느 이득이 P, b 예측, 단순 혼합, 추가 M 학습에서 나왔는지 분리한 것이 현재의 실증적 정리다. 새 학습 방법의 신규성이나 보편적 우월성으로 발표하지 않는다.

[추정] 남는 방법론의 핵심은 'TSFM을 붙일 수 있다'가 아니라, **압축·시간 앞단을 더해야 할 조건을 강한 대안과 비교해 설명하는 것**이다. B2가 같은 G/H에서 B0를 넘지 못한 현재 결과는 P의 필수성을 보류하게 한다. Jena의 ridge 우위는 TSFM 유지의 필요성을 묻고, Electricity의 RawSmall 우위는 채널 압축의 대가를 묻는다. PeMS의 compressed-H 우위는 같은G의 RawH 대비 일부 정확도 가치를 보이지만 LoRA 및 실제 비용 대조가 남는다. 이것들은 실패를 숨긴 결론이 아니라 연구가 답해야 할 경계다.

## 최종 편집을 기다리는 항목

- **T:** 아직 이 초안의 최종 T 결과가 없다. V로 후보·예산·ID를 잠근 뒤 등록 전체 origin·전체 채널로 공식 실행하고 별도 검산해야 한다. V 개선·subset replay·저장 수치 검산을 최종 일반화로 바꾸지 않는다. T 값으로 방법/K/지표를 다시 고르지 않는다.
- **비용:** ROOT의 실제 B1/B8 fresh/shared·K 전환·F 호출·메모리 결과를 기다린다. K/C를 속도비로 쓰거나 cached-parent F0을 배포 F0으로 쓰지 않는다. 비용이 없으므로 RawTiny/G3보다 싸다는 문장을 채우지 않는다.
- **커버리지:** 전체 grid·세 seed·두 번째 cutoff·채널 subset·장기 수렴·미실행 E3 및3 epoch E4 prefix는 완료로 적지 않는다. [미가용 조건](UNAVAILABLE_RELEVANT_CONDITIONS.csv)을 그대로 남긴다.

[확인] 현재 감사 장부의 **457은 저장 식별자 수**다. 405개 V bundle, 원래 존재하는48개 dense-M manifest, 신규 RawH M4의 사후 감사용 파생 계약을 구분한다. '457개의 원본 manifest를 검증했다'고 쓰지 않는다. 이 초안은 이 가운데 주4자료 primary V359행을 metadata로 추출했으며 기존457개를 다시 FP64 채점한 것이 아니다. RawH numeric8 PASS 뒤 원 프로세스 exit1, 별도 postflight readback0의 상태도 원 영수증 그대로다. 세부 감사 범위는 이전 [같은 G 보고](../raw_deep_g_saved_finalize_1792091230000/README_KO.md)를 따른다.
"""
    new(output/"SCIENTIFIC_DRAFT_KO.md",draft)
    contracts="# 대표 대조군의 실제 ID와 학습 계약\n\n[확인] 이 목록은 기존 V metadata의 기술용 인용이다. T 선택·자동 모델 선택이 아니다. NA는 실제 해당 V가 없다는 뜻이다. A60_end와M_cal_end는 각 자료의 원본 row index exclusive endpoint이며, 원시 시간 간격·H/C가 자료별로 달라 평균하지 않는다. 전체 hash와 source score 위치는 CSV에 있다.\n\n```text\n"
    for row in matrix:
        if not row.get("mse"):continue
        contracts+=f"{row['table_label']} | {row['run_id']}\n  V_MSE={row['mse']}; MAE={row['mae']}; endpoints={row['attempted_epoch_endpoints'] or 'optimizer0/closed form'}; selected_epoch={row['selected_epoch'] or 'NA'}; actual_fit_end={row['actual_fit_end'] or 'NA'}; A60_end={row['A60_end']}; M_end={row['M_cal_end']}; selection={row['checkpoint_selection_split'] or 'fixed/no checkpoint'}\n  fit={row['fit_information_scope']}; device={row['fit_device']}; training_status={row['actual_training_status']}; source_files_current={row['training_source_files_match_current'] or 'not applicable/metadata absent'}\n"
    contracts+="```\n\n`optimizer_updates`은 base model에서 search_updates, H/control에서는 해당 fit의 updates, ridge에서는 actual_optimizer_updates를 옮긴 필드다. fresh A60 refit은 `fresh_refit_updates`에 따로 있어 앞 필드를 전체 학습 update라고 읽지 않는다. `attempted_epoch_endpoints`는 실제 receipt의 요청/시도 endpoint이며 선택값은 `selected_epoch`다. 미수렴 판정은 원 receipt를 유지한다.\n"
    new(output/"BASELINE_CONTRACTS_KO.md",contracts)
    for name,expected in extra_inputs.items():
        if sha(public/name)!=expected["sha256"]:raise ValueError("Interpretation receipt changed")
    new(output/"INTERPRETATION_SOURCE_PINS.json",dict(additional_receipts=extra_inputs,existing_extraction_source_pins_sha256=sha(output/"SOURCE_PINS.json"),no_arrays_models_gzip_or_new_statistics=True,automatic_T_selection=False))
    files=[path for path in output.iterdir() if path.is_file() and "attempt1" not in path.name and "attempt2" not in path.name]
    root=public.parents[1]
    new(output/"FILE_MANIFEST.json",dict(files=[dict(repository_relative_path=path.relative_to(root).as_posix(),sha256=sha(path),bytes=path.stat().st_size) for path in sorted(files)],
        excluded_superseded_metadata_helpers=["extract_metadata_attempt1.py","extract_metadata_attempt2.py"],metadata_extraction_actual_exitcode=0,metadata_only_prior_errors_preserved=True))
    files += [output/"FILE_MANIFEST.json",output/"PUBLICATION_ALLOWLIST.txt",output/"PUBLICATION_SCAN.json"]
    new(output/"PUBLICATION_ALLOWLIST.txt","\n".join(sorted(path.relative_to(root).as_posix() for path in files))+"\n")
    sys.path.insert(0,str(public))
    import publication_check
    failures=[]
    for path in files:
        if path.name!="PUBLICATION_SCAN.json":publication_check._scan_text(path.relative_to(root).as_posix(),path.read_text(encoding="utf-8-sig"),failures)
    scan=dict(status="PASS" if not failures else "FAIL",actual_scanned_files=len(files)-1,findings=failures,actual_finding_count=len(failures),scientific_source_or_registry_mutated=False,NPZ_weights_model_gzip_T_GPU_executed=False,
        scope="Owned metadata synthesis text allowlist only; existing scanner; no Git/index/network",scanner_source_sha256=sha(public/"publication_check.py"))
    publication_check._scan_text("PUBLICATION_SCAN.json",json.dumps(scan),failures)
    if failures:raise ValueError("Public text scan failed")
    new(output/"PUBLICATION_SCAN.json",scan)
    print(json.dumps(dict(status="SCIENTIFIC_METADATA_DRAFT_READY",primary_V_rows=359,strong_baseline_rows=50,E3E4_rows=34,source_role_rows=len(roles),allowlist_files=len(files),new_array_scoring=0,scanner_findings=0)),flush=True)


if __name__=="__main__":main()
