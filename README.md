# Hierarchical TSFM PEFT screen

동일 Chronos-Bolt-small + LoRA에서 실제 hierarchy 관계의 추가 예측 가치를 평가합니다.
이 hierarchy screen의 실행 계약은 [MASTER](docs/00_MASTER_CLI_hierarchical_tsfm_peft_new_pc_20260921.txt)입니다. 별도 승인된 채널 압축 잔차 연구는 아래 결과 색인과 해당 PLAN을 따릅니다.
실행 기록은 `results/hier_peft_screen_v1/`에 저장합니다. 논문 PASS를 주장하지 않습니다.

## Data

```python
from datasetsforecast.hierarchical import HierarchicalData
Y, S, tags = HierarchicalData.load("./data", "Labour")
Y2, S2, tags2 = HierarchicalData.load("./data", "TourismLarge")
```

공식 source: https://nixtla-public.s3.amazonaws.com/hierarchical-data/datasets.zip
Raw data, model weights, checkpoints는 Git에 포함하지 않습니다.

## Model and reconciliation

- Model: `amazon/chronos-bolt-small`
- Chronos: https://github.com/amazon-science/chronos-forecasting
- Nixtla: https://github.com/Nixtla/hierarchicalforecast
- Context 48개월, horizon 12개월, CALIBRATION으로만 MinT-shrink 추정.
- VALIDATION으로 LR/checkpoint 선택, TEST로 선택 금지.

## Setup and execution

Python 3.11로 새 `.venv`를 만들고 GPU에 맞는 공식 PyTorch CUDA wheel을 설치합니다.
검증된 `requirements-lock.txt`로 정확한 의존성을 재현합니다.
현재 환경은 Python 3.11.16, PyTorch 2.10.0+cu128 / CUDA 12.8, RTX 4070입니다.
환경의 출처·패키지 wheel 해시는 `ENVIRONMENT.json`에 있습니다.

```text
python -c "from pathlib import Path; Path('.cache').mkdir(exist_ok=True)"
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128 --report .cache/torch-install.json
python -m pip install -r requirements-lock.txt --report .cache/dependencies-install.json
python -m pip install --no-deps -e .
python -m hier_peft.env --verify
python -m hier_peft.data
python -c "from hier_peft.chronos_wrapper import download_model; download_model()"
python experiments/hier_peft_screen_v1/run_preflight.py
python experiments/hier_peft_screen_v1/run_train.py --stage A
python experiments/hier_peft_screen_v1/finalize.py
```

Preflight 미통과 시 학습을 실행하지 않습니다. Stage B는 Labour continuation rule 통과 시에만 허용합니다.
Labour gate가 통과한 경우 finalize 전에 동일 CLI의 `--stage B`를 실행합니다.
이미 main update가 존재하면 자동 재실행을 거부하여 fit/update 예산을 보호합니다.
새 머신에서는 기존 실행 결과를 덮어쓰지 않는 별도 workspace/results가 필요합니다.

## 완료된 Labour screen

`STOP_CURRENT_HIER_ADAPTER_AFTER_LABOUR_SCREEN`: 16 fits / 8,192 main updates를 완료했습니다.
HIER는 SELF·POOL 대비 두 repeat seed에서 일관된 추가 가치를 보이지 않아 TourismLarge 학습은 실행하지 않았습니다.

- [한국어 보고서](results/hier_peft_screen_v1/REPORT_KO.md)
- [최종 결정](results/hier_peft_screen_v1/FINAL_DECISION.md)
- [Seed effects](results/hier_peft_screen_v1/SEED_EFFECTS.csv)
- [Verification](results/hier_peft_screen_v1/VERIFICATION.json)

실행 후 JSON boolean 직렬화 문제만 수정했습니다. 실행 당시 source seal과 후처리 수정 내역을 모두 보존했습니다.

## 승인된 채널 압축 잔차 PEFT 개발 (2026-09-26)

동결 TSFM의 입력 채널 압축에서 남은 재구성 잔차를 작은 시간 예측기로 보완하는 방법을 **조건부 개발 주제**로 선택했습니다. 위 hierarchy screen의 단발 계약과 별도로 승인된 연구입니다. 강한 선형 대조 대비 보호 확증과 논문 전체 확증은 미확보입니다.

- [최종 주제 판단과 전체 비교](research/tsfm_peft_development_20260926/TOPIC_DECISION.md)
- [고정 보호 결과](research/tsfm_peft_development_20260926/protected_evaluation.json) 및 [저장 예측 검산](research/tsfm_peft_development_20260926/protected_verification.json)
- [승인된 계획 원문](research/tsfm_peft_development_20260926/PLAN.md)
- [현재 진행 상태와 다음 판단](research/tsfm_peft_development_20260926/STATUS.md)
- [재현 명령과 검산 범위](research/tsfm_peft_development_20260926/commands.json)
- [최초 16fit 개발 비교](research/tsfm_peft_development_20260926/initial_development.json) 및 [교정된 평가 비용](research/tsfm_peft_development_20260926/initial_cost_recheck_development.json)
- [잔차 예측 rank의 제한적 CPU 진단](research/tsfm_peft_development_20260926/residual_rank_diagnostics.json)
- [학습량 보완 후 개발 비교](research/tsfm_peft_development_20260926/extended_development.json)
- [잔차 용량 수정 후 개발 비교](research/tsfm_peft_development_20260926/revision1_rank32_development.json)
- [압축 차원 증가 대조](research/tsfm_peft_development_20260926/compression16_development.json) 및 [보호 확인 설정 봉인](research/tsfm_peft_development_20260926/final_seal.json)

잔차 rank를 8에서 32로 늘린 수정의 개발 MSE는 0.165850입니다. 동일 용량 원입력 보정보다 8.11%, 강화한 선형 대조보다 7.06% 낮고 두 seed와 두 기간에서 같은 방향입니다. 미압축 TSFM과 LoRA보다는 각각 0.80%, 2.95% 높은 오차입니다. 압축 차원을 두 배로 늘린 대조보다도 20.01% 낮은 오차였습니다. 보호 Bull E1/E2에서는 같은 용량 원입력 보정보다 52.89%/9.02% 낮지만, 강한 선형 대조 대비 1.96%/15.48% 이득의 불확실성 구간은 모두 0을 포함합니다. 특히 E1에서는 미압축 LoRA보다 MSE가 65.29% 높습니다. 개발 근거에 따라 한 방법을 조건부 선택했으며 전이 우위를 확증했다고 부르지 않습니다.

Electricity의 기존 노출 구간은 개발에 사용했고, BDG2 Bull Office는 최종 설정을 봉인·게시한 뒤 고정 평가했습니다. 총 42완료 fit/43attempt, GPU 작업 점유 3.22시간을 사용했습니다. 보호 점수에 맞춘 재튜닝은 없으며 독립 외부 확증·독립 재현을 주장하지 않습니다. 원자료·가중치·예측 배열은 Git에 포함하지 않습니다.

## 채널 압축 잔차 PEFT 후속 개발 v2 (2026-09-27)

별도 승인된 후속 회차에서 E/D 고정·느린 적응, 같은 변경의 RAW 대조, Bull K4→8을 신규20fit로 비교했습니다. Bull E1은 E/D 고정으로 기존 MSE0.267849에서0.210365로21.46% 줄었지만 E2는1.156667에서1.246580으로7.77% 늘었습니다. Electricity에서는 기존 공동학습이 유지됐습니다. 이 회차의 Bull E1/E2는 **노출된 개발 평가**이며 새 확증이 아닙니다. v1 원본·보호 봉인·실행 기록은 보존했습니다.

판정은 **부분 정확도 개선, 논문 주력 확대와 배포 추가 가치는 보류**입니다. Bull FIXED는 대응 RAW 대비 E1 MSE가60.60% 낮지만 LoRA 대비 추론 속도 이점을 확보하지 못했습니다. 비용 블록 변동도 함께 보고합니다. GPU 점유는58.157분이었습니다. 별도 CPU optimizer 검사를 attempt 상한에서 누락한 계상 오류가 있어 승인20회 대비 22회가 실행됐으며, 이 이탈을 최종 보고서에 명시했습니다.

- [v2 방법·결과·비용·다음 연구 판단](research/tsfm_peft_followup_v2_20260926/TOPIC_DECISION.md)
- [전체 개발 비교](research/tsfm_peft_followup_v2_20260926/final_development.json) 및 [20fit 검산](research/tsfm_peft_followup_v2_20260926/final_verification.json)
- [반복 비용 요약](research/tsfm_peft_followup_v2_20260926/cost_summary.json) 및 [최종 예측·선택·비용 검산](research/tsfm_peft_followup_v2_20260926/final_artifact_checks_retry1.json)
- [승인 계획](research/tsfm_peft_followup_v2_20260926/PLAN.md), [진행·변경·예산 기록](research/tsfm_peft_followup_v2_20260926/STATUS.md)

## 잔차 PEFT v3: TSFM 대체 대조와 비용 진단 (2026-09-27)

동일 E/D·G에서 동결 Chronos-Bolt를 문맥 정규화를 포함한 공유 시간 affine 예측기로 대체하는 직접 비교를 신규12fit(첫8+대응 RAW4)로 완료했습니다. LINEAR_RES의 MSE는 TSFM_RES보다 Electricity DEV9.72%, Bull E1 22.31%, E2 9.86% 높았습니다. E2의 조건부 시간블록 구간은0을 포함합니다. RAW 후속도 격차를 닫지 못했습니다. 모든 점수는 노출 개발 결과입니다.

선형 대체의 배포 비용은 크게 작습니다. GPU배포 batch1 중심값은 TSFM_RES 약25.66/25.93ms, LINEAR_RES 약1.33/1.30ms이며 CPU LINEAR는0.57/0.55ms(Electricity/Bull)입니다. GPU상주·CPU 측정의 큰 블록 변동도 남아 있어 원인이 해결됐다고 주장하지 않습니다. **정확도 중심의 TSFM 경로는 조건부 유지, 잔차 PEFT 주력 논문 주제 확보는 보류**입니다. 미압축 F0/LoRA의 더 좋은 점수와 비용 절충을 함께 보고합니다.

실자료12/16attempt, 합성 optimizer2/6세션(각8step), GPU34.902분/180분으로 신규 상한 안에서 종료했습니다. v2의22/20 위반은 그대로 보존했습니다. 새 데이터·새 보호평가·새 PEFT 후보는 추가하지 않았습니다.

- [v3 최종 판단·정확도·비용·그림](research/tsfm_peft_backbone_value_v3_20260927/TOPIC_DECISION.md)
- [직접 대체 비교](research/tsfm_peft_backbone_value_v3_20260927/final_development.json), [대응 LINEAR_RAW 비교](research/tsfm_peft_backbone_value_v3_20260927/linear_raw_development.json)
- [GPU 두 측정 범위](research/tsfm_peft_backbone_value_v3_20260927/cost_gpu_01.json), [CPU 배포](research/tsfm_peft_backbone_value_v3_20260927/cost_cpu_01.json), [구간 수리 후 대표 profile](research/tsfm_peft_backbone_value_v3_20260927/cost_profile_02.json)
- [최종 검산](research/tsfm_peft_backbone_value_v3_20260927/final_checks.json), [승인 계획](research/tsfm_peft_backbone_value_v3_20260927/PLAN.md), [상태·예산·게시 기록](research/tsfm_peft_backbone_value_v3_20260927/STATUS.md)

## 잔차 PEFT v4: Hog 새 집단 고정 확인 (2026-09-27)

BDG2 Hog 사무용 전력32채널/K8에서 사전에 고정한 CURRENT/FIXED_ED·VAL 선택 정책으로 신경망16+CPU ridge3fit를 완료했습니다. 제한된 로컬 이력 조회에서 Hog의 직접 방법선택 흔적을 찾지 못했으나, 백본 사전학습 비중복은 미확정입니다. TEST-A/B를 함께 고정 평가했고 점수에 따른 재튜닝은 없었습니다.

**잔차 입력 설계는 후속 방법으로 유지하며, 현 구성은 GPU 메모리와 정확도 절충을 위한 조건부 선택지입니다.** 선택 RES의 전체 MSE3.220290은 RAW/COMPRESS4.591135보다29.86%, LINEAR_RES3.445439보다6.53% 낮았습니다. 반면 미압축 F0보다17.61%, 병합LoRA보다20.00% 높았습니다. RAW는 정상 학습 후 VAL이step0을 선택해 COMPRESS와 같은 함수이며, 이를 두 독립적인 승리로 세지 않습니다. MAE와 CPU 비용에서는 단순 선형 대안이 더 유리합니다.

RES의 batch4 처리량 중심값357.14원점/초는 LoRA172.52보다 약2.07배지만, 첫 블록에서는 순위가 뒤집혔으므로 안정적인 두 배 속도를 주장하지 않습니다. batch4 peak allocated는225.859MiB로 F0/LoRA보다31.28% 낮았습니다. 백본 저장 크기 자체는 줄지 않습니다. **다음에는 확증 확대보다 알려진 정확도 손해를 줄이는 개발을 우선할 것을 권고합니다.** 한 site 결과가 논문 전체의 신규성·범용 우수성을 확정하지는 않습니다.

실자료19/24attempt, 합성 optimizer1/6세션(총10update), GPU21.533분/240분으로 종료했습니다. 비용 VAL 재생 허용오차 수리1회와 중단된 블록의 원자료를 보존했으며 학습·TEST 수치는 바꾸지 않았습니다. 이전406개 파일 및 v2의22/20회 위반 기록도 보존했습니다.

- [v4 최종 판단·전체/기간/seed·비용·그림](research/tsfm_peft_fresh_group_v4_20260927/TOPIC_DECISION.md)
- [자료 계약](research/tsfm_peft_fresh_group_v4_20260927/data_contract.json), [TEST 전 봉인](research/tsfm_peft_fresh_group_v4_20260927/evaluation_seal.json), [고정 평가](research/tsfm_peft_fresh_group_v4_20260927/v4_test_01.json)
- [GPU 비용](research/tsfm_peft_fresh_group_v4_20260927/v4_cost_gpu_02.json), [CPU 비용](research/tsfm_peft_fresh_group_v4_20260927/v4_cost_cpu_01.json), [제한된 GPU상주 진단](research/tsfm_peft_fresh_group_v4_20260927/v4_cost_diagnostic_01.json)
- [최종 검산](research/tsfm_peft_fresh_group_v4_20260927/final_checks.json), [승인 계획](research/tsfm_peft_fresh_group_v4_20260927/PLAN.md), [상태·예산·게시 기록](research/tsfm_peft_fresh_group_v4_20260927/STATUS.md)

## 잔차 PEFT v5: Hog 정확도 회복 개발 (2026-09-28)

기존 예측 진단으로 선택한 LEVEL(잔차의 마지막 수준 제거·복원)을 matched RAW와 Hog/Bull/Electricity에서 총12fit 비교했습니다. **Hog 개발 구성으로 LEVEL을 조건부 유지합니다.** Hog 전체 MSE는 기존 RES3.220290에서2.666809로17.19% 줄었고, 같은 수준 복원항의 RAW2.949584보다9.59% 낮았습니다. LoRA2.683560 대비 작은 점추정 차이는 우위나 동등성을 확정하지 않으며, Hog B기간·MAE 손해도 남습니다.

Bull은 기존 RES보다 개선했지만 matched RAW의 전체 MSE가 더 낮고, Electricity는 기존 RES보다0.97% 나빠졌습니다. 모든 자료에 LEVEL을 적용하는 권고는 하지 않습니다. 이번 Hog 포함 세 자료는 모두 노출된 개발 평가입니다.

같은 회차 Hog의 LEVEL RES batch4 처리량353.73원점/초와 peak allocated225.86MiB는 병합 LoRA167.33원점/초·328.68MiB와 절충을 보였습니다. 기존 RES와 처리량은 거의 같고, batch1·블록 변동과 전체 배포크기를 함께 보고합니다. 실자료12/24attempt, 합성1/6세션(10update), GPU21.60분/240분으로 종료했습니다. 다음에는 고정 LEVEL의 미사용 집단 확인 여부를 결정할 것을 권고하며 독립 확증·논문 신규성 완료로 세지 않습니다.

- [v5 최종 판단·기간/seed·손익·비용·그림](research/tsfm_peft_accuracy_recovery_v5_20260927/TOPIC_DECISION.md)
- [기존 예측 진단](research/tsfm_peft_accuracy_recovery_v5_20260927/v5_hog_diagnosis_01/diagnosis.json), [최종 비교 JSON](research/tsfm_peft_accuracy_recovery_v5_20260927/final_comparison.json), [CSV](research/tsfm_peft_accuracy_recovery_v5_20260927/final_comparison.csv)
- [같은 회차 비용](research/tsfm_peft_accuracy_recovery_v5_20260927/cost01.json), [검산](research/tsfm_peft_accuracy_recovery_v5_20260927/final_checks.json)
- [승인 계약](research/tsfm_peft_accuracy_recovery_v5_20260927/PLAN.md), [상태·예산·게시](research/tsfm_peft_accuracy_recovery_v5_20260927/STATUS.md), [준비 시간 계상 한계](research/tsfm_peft_accuracy_recovery_v5_20260927/preparation_accounting.json)
