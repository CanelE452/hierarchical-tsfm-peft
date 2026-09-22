# Jena: 예측 변화 보존 대리모델 PEFT — 제한된 Go/No-go

[확인] 이 보고서는 저장된 예측·선택·비용 기록에서 자동 작성됐다. 모든 품질 수치는 원래 Chronos-2에 LoRA를 이식한 뒤의 값이다.

## 1. 예상했던 효과와 이번 판정

예상은 대리모델에서 LoRA를 학습해 원래 Chronos-2로 옮기면 원래 품질을 유지하면서 준비비를 포함한 적응시간이 줄고, 반응 보정이 VALUE 보정보다 이식 후 오차를 더 낮출 것이라는 것이었다. 이번 판정은 `HOLD_NO_AUTO_RESCUE`다 (원판정 `GO_STANDARD_ONLY`; dominating control meets mean criteria but not the per-seed test). 후보 E_RESPONSE_DELTA의 원래 모델 TEST 오차는 F_FULL보다 평균 +0.43% (허용 0.5% 이내)다. 준비비와 128회 선택 절차 전체를 포함한 총시간은 F_FULL 대비 -63.1% 절감이다(음수는 더 오래 걸림). 요구 기준은 20% 절감이다. E_VALUE_DELTA 대비 오차 개선은 -0.02%(기준 0.3%, 두 seed 같은 방향 필요)이고, 후보보다 오차와 총시간이 모두 같거나 낮은 대조군은 F_TOP3, VALUE_DELTA, 품질·시간 목표를 단독으로 충족한 대조군은 없음이다.

## 2. 고정 문제·정보 권한·비교군

- 자료: Jena 2024, 10분 간격 21계열, 과거 1152(8일) → 미래 144(1일). 합법 원점 TRAIN/VAL/TEST = {'train': 565, 'val': 73, 'test': 74}. 이전 감사와 값·scale·원점이 모두 일치했다.
- 원래 모델: amazon/chronos-2 revision 29ec3766…, FP32 master + BF16 autocast, gradient checkpointing.
- 대리모델: encoder Linear 120개를 activation-aware SVD(rank 192/307)로 분해. 텐서 용량 240 MiB (원래 456 MiB).
- 공통: seed [92711, 92712], LoRA rank 8 / alpha 16, LoRA+ (A 1e-5, B 1.6e-4), fit당 128 update, 체크포인트 0/16/32/64/128. 실제 main update 1536회, calibration 32회.
- 선택: 모든 체크포인트를 원래 모델로 이식한 VAL 최소. 대리모델 점수는 선택에 쓰지 않았다.

| 군 | LoRA 모듈 | 학습 파라미터 | 학습 모델 | 배포 adapter (KiB) |
| --- | --- | --- | --- | --- |
| F_FULL (원래 모델 전체 LoRA+) | 97 | 1206912 | 원래 모델 | 4714 |
| F_TOP3 (마지막 3 block + 출력 LoRA+) | 25 | 322176 | 원래 모델 | 1258 |
| E_EMLOC (대리모델 학습 + EMLoC 보정 이식) | 97 | 1206912 | svd | 4714 |
| E_DELTA (출발 예측 보존, 보정 없음) | 97 | 1206912 | svd | 4714 |
| E_VALUE_DELTA (출력 값 보정 16회) | 97 | 1206912 | value | 4714 |
| E_RESPONSE_DELTA (값 + 반응 보정 16회, 후보) | 97 | 1206912 | response | 4714 |

## 3. 원래 TSFM에 이식한 실제 예측 품질

주지표는 원단위 분위수 예측을 TRAIN 표준편차로 나눈 평균 2-pinball이며 낮을수록 좋다. `F_FULL 대비`의 양수는 F_FULL보다 오차가 낮다는 뜻이다.

| 군 | seed 92711 | seed 92712 | 평균 | F_FULL 대비 | nMAE | coverage80 | crossing | 선택 step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F_FULL | 0.1752 | 0.1764 | 0.1758 | — | 0.2512 | 0.795 | 0.0166 | 64/128 |
| F_TOP3 | 0.1762 | 0.1769 | 0.1765 | -0.41% | 0.2517 | 0.810 | 0.0182 | 128/128 |
| E_EMLOC | 0.1763 | 0.1773 | 0.1768 | -0.57% | 0.2528 | 0.783 | 0.0165 | 128/128 |
| E_DELTA | 0.1758 | 0.1773 | 0.1766 | -0.44% | 0.2522 | 0.798 | 0.0157 | 64/128 |
| E_VALUE_DELTA | 0.1757 | 0.1773 | 0.1765 | -0.41% | 0.2520 | 0.805 | 0.0151 | 64/128 |
| E_RESPONSE_DELTA | 0.1760 | 0.1771 | 0.1765 | -0.43% | 0.2519 | 0.802 | 0.0150 | 64/128 |

참조(같은 TEST·같은 지표, 판정에는 쓰지 않음): F0 seed 0 = 0.1798; HEAD_PRIOR_RUN seed 92601 = 0.1780; HEAD_PRIOR_RUN seed 92602 = 0.1777. HEAD_PRIOR_RUN은 이전 실험(다른 seed·다른 실행)의 출력 보정 결과다.

![원래 모델로 이식한 TEST 점수](figures/fig1_transferred_quality.png)

| 비교 군 | 기준 군 | 평균 오차 감소 | 7일 block bootstrap 95% | seed 92711 | seed 92712 |
| --- | --- | --- | --- | --- | --- |
| E_RESPONSE_DELTA | F_FULL | -0.43% | [-0.92, +0.02] | -0.44% | -0.42% |
| E_RESPONSE_DELTA | F_TOP3 | -0.02% | [-0.74, +0.61] | +0.11% | -0.14% |
| E_RESPONSE_DELTA | E_EMLOC | +0.14% | [-0.45, +0.77] | +0.17% | +0.11% |
| E_RESPONSE_DELTA | E_DELTA | +0.01% | [-0.25, +0.30] | -0.10% | +0.13% |
| E_RESPONSE_DELTA | E_VALUE_DELTA | -0.02% | [-0.18, +0.16] | -0.16% | +0.13% |
| F_TOP3 | F_FULL | -0.41% | [-1.12, +0.19] | -0.55% | -0.28% |
| E_EMLOC | F_FULL | -0.57% | [-0.99, -0.12] | -0.61% | -0.53% |
| E_DELTA | F_FULL | -0.44% | [-0.82, -0.07] | -0.33% | -0.55% |
| E_VALUE_DELTA | F_FULL | -0.41% | [-0.79, -0.07] | -0.28% | -0.55% |
| E_EMLOC | E_EMLOC_NAIVE_SAME_STEP | -0.04% | [-0.10, +0.02] | -0.08% | -0.01% |
| F_FULL | F0 | +2.22% | [+0.82, +3.68] | +2.55% | +1.90% |
| F_TOP3 | F0 | +1.82% | [+0.84, +2.75] | +2.01% | +1.63% |
| E_EMLOC | F0 | +1.67% | [+0.23, +3.13] | +1.95% | +1.38% |
| E_DELTA | F0 | +1.79% | [+0.53, +3.11] | +2.22% | +1.36% |
| E_VALUE_DELTA | F0 | +1.82% | [+0.59, +3.05] | +2.28% | +1.36% |
| E_RESPONSE_DELTA | F0 | +1.80% | [+0.63, +2.99] | +2.12% | +1.49% |

## 4. 준비비 포함 총시간과 비용 성분

총시간은 한 사용자가 그 군을 처음부터 적용할 때의 비용이다. 자료 준비, 모델 로드, activation/SVD, probe 교사 예측·gamma 보정, F0/E0 cache, 128 update 전부, 다섯 체크포인트의 이식·보정·원래 모델 VAL, 최종 선택·export를 포함한다. 최선 체크포인트를 미리 안 것처럼 계산하지 않았다. 공유 준비물도 군마다 전체를 청구했다. 진단 전용 계산(대리모델 VAL, 보정 전 EMLoC 이식)은 뺐다.

| 군 | 총시간 평균(s) | seed 92711 | seed 92712 | 준비(s) | 학습·VAL 절차(s) | 이식·선택(s) | 128 update(s) | update/s | step CV | 학습 peak (MiB) | 민감도 총시간(s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F_FULL | 193.5 | 195.0 | 191.9 | 0.2 | 193.3 | 0.0 | 160.8 | 0.80 | 0.05/0.03 | 708 | 193.5 |
| F_TOP3 | 87.4 | 88.7 | 86.1 | 0.2 | 87.2 | 0.0 | 61.6 | 2.08 | 0.07/0.08 | 650 | 87.4 |
| E_EMLOC | 233.5 | 232.2 | 234.8 | 15.7 | 216.6 | 1.2 | 183.1 | 0.70 | 0.04/0.04 | 486 | 233.5 |
| E_DELTA | 303.6 | 300.5 | 306.8 | 85.8 | 217.7 | 0.1 | 184.0 | 0.70 | 0.04/0.03 | 487 | 270.3 |
| E_VALUE_DELTA | 314.0 | 312.9 | 315.2 | 92.9 | 221.1 | 0.1 | 186.9 | 0.68 | 0.03/0.05 | 487 | 279.7 |
| E_RESPONSE_DELTA | 315.6 | 314.0 | 317.3 | 99.4 | 216.1 | 0.1 | 182.9 | 0.70 | 0.04/0.04 | 487 | 282.3 |

민감도 총시간은 보조값이다. 계약 청구액에서 VAL cache(청구하지 않는 대리모델 진단에만 쓰임)와, 학습 schedule이 쓰지 않은 TRAIN 원점 몫의 cache를 뺐다. TRAIN 몫은 원점 수 비례로 추정했다 [추정]. 판정은 계약 청구액으로만 한다. E 군의 fit 안 대리모델 재로드와 이식용 원래 모델 재로드(보수적 이중 청구)는 평균 5.8초다.

![준비비 포함 총시간과 원래 모델 TEST 오차](figures/fig2_total_cost_frontier.png)

![비용 성분](figures/fig3_cost_breakdown.png)

| 군 | 평균 TEST 오차 | 평균 총시간(s) | Pareto 최적 | 지배하는 군 |
| --- | --- | --- | --- | --- |
| F_FULL | 0.1758 | 193.5 | 예 | — |
| F_TOP3 | 0.1765 | 87.4 | 예 | — |
| E_EMLOC | 0.1768 | 233.5 | 아니오 | F_FULL, F_TOP3 |
| E_DELTA | 0.1766 | 303.6 | 아니오 | F_FULL, F_TOP3 |
| E_VALUE_DELTA | 0.1765 | 314.0 | 아니오 | F_FULL |
| E_RESPONSE_DELTA | 0.1765 | 315.6 | 아니오 | F_FULL, F_TOP3, E_VALUE_DELTA |

보조 목표 도달(사후 목표 = 1.005 × F_FULL 두 seed 최선 VAL 평균 = 0.24232; F0 VAL 0.24781, F0 달성 여부 False). 첫 도달 비용은 기록된 체크포인트의 상한이며 학습 종료 규칙이 아니다.

| 군 | seed | 상태 | step | 도달 비용 상한(s) |
| --- | --- | --- | --- | --- |
| F_FULL | 92711 | REACHED_AT_RECORDED_CHECKPOINT | 64 | 108.0 |
| F_TOP3 | 92711 | REACHED_AT_RECORDED_CHECKPOINT | 128 | 88.7 |
| E_EMLOC | 92711 | REACHED_AT_RECORDED_CHECKPOINT | 64 | 135.5 |
| E_DELTA | 92711 | REACHED_AT_RECORDED_CHECKPOINT | 64 | 204.7 |
| E_VALUE_DELTA | 92711 | REACHED_AT_RECORDED_CHECKPOINT | 64 | 214.7 |
| E_RESPONSE_DELTA | 92711 | REACHED_AT_RECORDED_CHECKPOINT | 64 | 218.3 |
| F_FULL | 92712 | REACHED_AT_RECORDED_CHECKPOINT | 128 | 191.9 |
| F_TOP3 | 92712 | NOT_REACHED | — | — |
| E_EMLOC | 92712 | REACHED_AT_RECORDED_CHECKPOINT | 128 | 234.8 |
| E_DELTA | 92712 | REACHED_AT_RECORDED_CHECKPOINT | 128 | 306.7 |
| E_VALUE_DELTA | 92712 | REACHED_AT_RECORDED_CHECKPOINT | 128 | 315.1 |
| E_RESPONSE_DELTA | 92712 | REACHED_AT_RECORDED_CHECKPOINT | 128 | 317.3 |

### EMLoC와 부분 층 LoRA로 충분한가

- F_TOP3: F_FULL 대비 오차 +0.41%, 총시간 +54.8% 절감. 품질 기준 충족, 시간 기준 충족.
- E_EMLOC: F_FULL 대비 오차 +0.57%, 총시간 -20.7% 절감. 품질 기준 미충족, 시간 기준 미충족.
- 사전 판정표: 후보가 GO가 아닐 때 대조군이 F_FULL 대비 품질 0.5% 이내(평균과 두 seed)와 총시간 20% 이상 절감(평균, 두 seed 모두 더 빠름)을 함께 충족하면 `GO_STANDARD_ONLY`다. 충족한 대조군: 없음. 후보를 지배한 대조군: F_TOP3, VALUE_DELTA.
- EMLoC 대조 상태: `LIMITED_BASELINE`. 공식 EMLoC는 attention과 MLP 모두에 LoRA를 두고 보정하지만, 이 계약의 LoRA 지도는 attention q/k/v/o와 출력 head뿐이다. 그래서 압축된 MLP wi/wo의 오차는 EMLoC 보정이 고칠 수 없다(대리모델 3 delta 군은 출발 예측 보존으로 0차 오차를 없앤다).

## 5. 대리모델 점수와 실제 이식 후 점수의 차이

대리모델 쪽 점수는 학습에 쓰인 예측(E_EMLOC는 대리모델 예측, delta 계열은 F0 + 대리모델 변화분)의 VAL 오차다. 실제 점수는 같은 LoRA를 원래 모델에 옮긴 VAL 오차다. 둘의 차이는 진단이며 선택에 쓰지 않았다.

| 군 | seed | 선택 step | 대리모델 쪽 VAL | 이식 후 VAL | 차이(이식−대리) | 예측 차이 RMSE/σ | 보정 전 EMLoC VAL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E_EMLOC | 92711 | 128 | 0.2404 | 0.2416 | 0.0012 | 0.9149 | 0.2418 |
| E_EMLOC | 92712 | 128 | 0.2399 | 0.2413 | 0.0014 | 0.9850 | 0.2413 |
| E_DELTA | 92711 | 64 | 0.2423 | 0.2423 | -0.0001 | 0.2002 | — |
| E_DELTA | 92712 | 128 | 0.2415 | 0.2417 | 0.0002 | 0.1827 | — |
| E_VALUE_DELTA | 92711 | 64 | 0.2421 | 0.2421 | 0.0000 | 0.1849 | — |
| E_VALUE_DELTA | 92712 | 128 | 0.2421 | 0.2422 | 0.0001 | 0.1814 | — |
| E_RESPONSE_DELTA | 92711 | 64 | 0.2428 | 0.2423 | -0.0006 | 0.3095 | — |
| E_RESPONSE_DELTA | 92712 | 128 | 0.2424 | 0.2419 | -0.0005 | 0.2782 | — |

EMLoC 보정의 TEST 효과(같은 step, 보정 전 대비 오차 감소): -0.04% (seed -0.08% / -0.01%).

![대리모델 학습용 예측과 이식 후 예측](figures/fig4_transfer_mismatch.png)

## 6. VALUE 보정 대비 RESPONSE 보정의 추가 효과

보정 대리 지표(낮을수록 원래 모델에 가까움)는 L_value = 출력 오차 / 보정 전 출력 오차, L_response = 반응 오차 / 반응 크기다. 검사용 8원점은 보정에 쓰지 않았고 probe seed도 다르다.

| 대리모델 | L_value (보정 원점) | L_response (보정 원점) | L_value (검사 원점) | L_response (검사 원점) | 반응 크기 MSE/σ² (검사) |
| --- | --- | --- | --- | --- | --- |
| svd | 1.0000 | 0.1017 | 1.0000 | 0.0936 | 2.33e-07 |
| value | 6.0159 | 0.1788 | 2.4271 | 0.1403 | 2.33e-07 |
| response | 6.0933 | 0.2059 | 3.1836 | 0.1268 | 2.33e-07 |

gamma 통계: VALUE 0.0360 (경계 도달 0.0%), RESPONSE 0.0355 (경계 도달 0.0%), 평균 |gamma|.

실제 원래 모델 TEST에서 E_RESPONSE_DELTA의 오차 감소: E_VALUE_DELTA 대비 -0.02% (seed -0.16% / +0.13%, CI [-0.18, +0.16]), E_DELTA 대비 +0.01%. 대리 지표의 개선은 예측 성공의 증거가 아니다.

불일치: 보정 뒤 L_value가 1보다 크다(VALUE 검사 원점 2.427, RESPONSE 3.184). 즉 계약의 고정 recipe(AdamW lr 1e-2, 원점 하나씩 16 update)로 한 gamma 보정이 대리모델 출력을 원래 모델에서 오히려 더 멀어지게 했다. 반응 오차 L_response도 보정 전보다 나빠졌다. 계약에 따라 lr과 update 수는 바꾸지 않았다. 원인으로는 Adam 초기 step이 25,800개 gamma를 모두 약 ±lr씩 움직인 과도 이동을 추정한다 [추정].

1e-4 probe의 FP32 반응 크기(MSE/σ² ≈ 3×10⁻⁹)는 L_response 분모의 ε(1e-8)보다 작다. 그래서 그 probe가 쓰인 보정 step(원점 j mod 4 ∈ {0, 2})에서는 반응 항이 약해진다. ε은 계약 고정값이다.

![VALUE 대비 RESPONSE: 반응 대리 지표와 실제 TEST](figures/fig5_response_ablation.png)

## 7. 판정과 한계

판정: `HOLD_NO_AUTO_RESCUE`

| 사전 기준 | 충족 |
| --- | --- |
| 1) 후보 품질: F_FULL 대비 평균·두 seed 모두 0.5% 이내 손해 | True |
| 2) 총시간: 평균 20% 이상 절감, 두 seed 모두 더 빠름 | False |
| 3) E_VALUE_DELTA보다 평균 0.3% 이상 낮은 오차, 두 seed 같은 방향 | False |
| 4) 더 정확하면서 비용도 낮은 대조군 없음 | False |
| (참고) 표준 대조군의 단독 목표 충족 → GO_STANDARD_ONLY | 없음 |
| 5) 실모델·이식·자료 무결성과 시간 신뢰도 | True |

원판정 `GO_STANDARD_ONLY`: 품질 손해 +0.429%, 총시간 절감 -63.15%, VALUE 대비 -0.016%, 지배 대조군 F_TOP3, VALUE_DELTA. 보정 규칙 적용: dominating control meets mean criteria but not the per-seed test.

[판단] 판정 기준(0.5% / 20% / 0.3%)은 결과 전에 고정한 연구 투자 선별값이다. 통계적 동등성이나 논문 PASS가 아니다. NO_GO여도 모든 대리모델 PEFT가 불가능하다는 뜻이 아니고, 현재 구현의 한계로 남긴다.

- 한 자료(Jena), 한 모델, 두 main seed. 두 seed는 같은 압축·같은 calibration을 공유하므로 calibration 모집단 반복이 아니다.
- 고정 LR·rank·압축비·128 update의 짧은 적응 설정이다. HPO나 수렴 최적 학습률이 아니다.
- 예산 경계 표시(128에서 선택되고 마지막 구간 VAL 개선 0.2% 초과): {'F_FULL_92711': False, 'F_FULL_92712': True, 'F_TOP3_92711': True, 'F_TOP3_92712': True, 'E_EMLOC_92711': False, 'E_EMLOC_92712': True, 'E_DELTA_92711': False, 'E_DELTA_92712': True, 'E_VALUE_DELTA_92711': False, 'E_VALUE_DELTA_92712': True, 'E_RESPONSE_DELTA_92711': False, 'E_RESPONSE_DELTA_92712': True}.
- 시간 신뢰도: step CV 0.2 초과 없음, seed 간 throughput 차이 {'F_FULL': 0.016, 'F_TOP3': 0.024, 'E_EMLOC': 0.015, 'E_DELTA': 0.027, 'E_VALUE_DELTA': 0.003, 'E_RESPONSE_DELTA': 0.02}.
- EMLoC 대조는 Chronos로 옮긴 핵심 수식이다(`LIMITED_BASELINE`). 공식 함수와 FP32 parity를 확인했지만 원논문 VLM 결과의 재현은 아니다. 보정은 FP32로 체크포인트마다 적용했다.
- 반응 보정 forward(교사와 대리모델)는 FP32로 실행했다. BF16에서는 1e-4/1e-3 probe의 반응이 반올림 잡음으로 측정됐기 때문이다(사전검사 기록). 본학습은 계약대로 BF16이다.
- 이 계약의 판정표에서 EMLoC 대조는 항상 LIMITED_BASELINE(MLP 보정 불가)이므로, 후보가 받을 수 있는 최선의 판정은 HOLD_NO_AUTO_RESCUE다(산술 GO도 보류). 산술 원판정은 base_outcome에 따로 남겼다.
- 두 seed 간 차이가 0.3% 기준과 같은 크기일 수 있고(이전 LoRA+ 두 seed TEST 차이 0.445%), gamma는 lr 1e-2 × 16 update로 [−log 2, log 2] 경계에 거의 닿지 않을 것으로 결과 전에 예상했다.
- 이 PC는 Windows 데스크톱 GPU라 다른 앱의 간헐적 GPU 사용이 시간 잡음을 만들 수 있다. 잡음은 재실행으로 숨기지 않았다.
- Jena TEST는 이전 개발에 쓰인 자료다. 독립 확증이 아니다.
- 최종 추론은 모든 군이 원래 모델 + LoRA이므로 추론 가속을 주장하지 않는다.
- 이 비교가 끝난 뒤 추가 진단, 다른 자료(ETTm2 등), 후속 모델을 자동으로 실행하지 않았다.

## 8. 재검산 자료

[자료](DATA_MANIFEST.json) · [모듈 지도](MODULE_MAP.json) · [대리모델](EMULATOR_MANIFEST.json) · [보정](CALIBRATION_MANIFEST.json) · [probe](PROBE_MANIFEST.json) · [사전검사](PREFLIGHT.json) · [사전검사(원래)](PREFLIGHT_BASE.json) · [사전검사(대리)](PREFLIGHT_EMULATOR.json) · [비용 기준](COST_BASIS.json)

[선택 봉인](MODEL_SELECTION.json) · [예측 hash](PREDICTIONS_MANIFEST.json) · [점수](SCORES.csv) · [seed 효과](SEED_EFFECTS.csv) · [자원](RESOURCES.csv) · [비용 성분](COST_COMPONENTS.csv) · [이식 차이](TRANSFER_GAP.csv) · [보정 반응](CALIBRATION_RESPONSE.csv) · [학습곡선](CURVES.csv) · [판정](DECISION.json) · [검산](VERIFICATION.json) · [그림 값](FIGURE_VALUES.csv) · [캡션](CAPTIONS.md)

원자료, 모델·대리모델 가중치, LoRA 체크포인트, 예측 배열은 로컬 `.cache`에만 있다. GitHub의 표만으로 전체를 재생할 수 있다고 주장하지 않는다.
