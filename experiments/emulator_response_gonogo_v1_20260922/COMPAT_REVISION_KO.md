# 봉인 전 구현 revision 기록 (2026-09-22, 모든 학습 전)

계약: `MASTER_CLI.txt`, `config.json`(번들과 바이트 동일). 이 문서는 첫 봉인과 모든 optimizer update 전에 CLI가 바꾼 구현 세부와 그 근거를 적는다. 계약의 자료·비교군·압축비·rank·LR·seed·update 수·판정 기준값은 바꾸지 않았다.

## 1. 실모델 사전 측정 (optimizer update 0회, 결과 폴더 기록 없음)

| 측정 | 값 |
|---|---|
| 압축 대상 / LoRA 대상 / TOP3 대상 | 120 / 97 / 25 모듈 |
| activation 수집 / 120층 SVD / 대리모델 구성 | 1.6초 / 8.5초 / 1.0초 |
| 대리모델 텐서 용량 / 원래 | 52.6% |
| 대리모델과 원래의 loc/scale | 정확히 동일 |
| 초기 LoRA(B=0) 예측 = 기준(원래, 대리, TOP3) | 차이 0.0 |
| EMLoC 공식 함수 vs 이식(scaling 1) | FP32 1.2e-6, BF16 2.6e-2 |
| 공식 함수를 scaling 2에 그대로 쓸 때 active subspace 불일치 | 12% (scaling 명시가 필요한 이유) |
| 원래 모델 forward / fwd+bwd (한 원점) | 0.056초 / 0.313초 |
| 대리모델 fwd+bwd: 번들 층 / exp(γ) 접은 층 | 0.401초 / 0.366초 |
| BF16 probe 반응의 FP32 대비 상대오차 (1e-4 / 1e-3 probe) | 28~59배 / 3.5~5.6배 |
| BF16 probe 반응 중 정확히 0인 원소 | 55~81% |
| FP32 참 반응 크기 MSE/σ² (1e-4 / 1e-3 probe) | 3×10⁻⁹ / 3×10⁻⁷ |

## 2. 바꾼 것

1. calibration 정밀도: 교사와 대리모델의 calibration forward를 모두 FP32로 실행한다(`settings.calibration_precision = fp32`). BF16에서는 반응 목표가 반올림 잡음이다(위 표). 계약 §12(o)의 "정밀도 이상은 TRAIN 전 compat 문서로 기록"에 해당한다. probe 크기는 계약 §6대로 키우지 않았다.
2. 동결 대리모델 층: gamma가 동결된 곳에서는 exp(γ)를 left에 미리 곱한 `FrozenFactorizedLinear`를 쓴다. 같은 분해이고 factor 저장도 유지한다. 번들 층의 BF16→FP32 승격 비효율(약 10%)이 대리모델 군에만 불리하게 작용하는 것을 없앴다. 계약 §9 "기능을 일부러 비활성화하지 않는다"의 취지다. calibration 동안은 번들 `FactorizedLinear`를 그대로 쓴다.
3. 판정표 고정(`decision.py`, 봉인 대상): `GO_STANDARD_ONLY`를 계약 정의대로 "대조군이 품질·시간 목표를 단독으로 충족"으로 판정한다. `TIMING_HOLD`는 속도 근거의 GO에만 적용한다. EMLoC 대조는 `LIMITED_BASELINE`이며 이 상태에서 METHOD GO를 금지한다. 판정 코드를 학습 코드와 함께 봉인하고, 검산은 판정을 별도 코드 경로로 다시 계산한다.
4. EMLoC 대조 제한 명시: 공식 LoRA 대상은 MLP(w1/w2/w3)까지 포함하지만, 이 계약의 LoRA 지도는 attention + 출력 head뿐이다. 그래서 압축된 MLP 오차는 EMLoC 보정 대상이 아니다. 계약의 LoRA 지도는 바꾸지 않았다.
5. cache 비용: 계약대로 TRAIN/VAL 전체를 청구한다. TRAIN과 VAL을 따로 계측하고, 보조 민감도 총시간(VAL cache와 미사용 TRAIN 몫 제외, 비례 추정)을 함께 공개한다.
6. 측정 구간: delta 군의 loc/scale 대조와 finite 검사를 update 측정 구간 밖으로 옮겼다(host 동기화 제거).
7. 사전검사 순서: 장부를 쓰지 않는 모든 검사를 먼저 한다. 여기에는 gamma 저장→재구성 왕복, calibration gradient 경로, BF16/FP32 해상도가 들어간다. smoke update 16회는 각 사전검사 단계의 마지막에만 한다.
8. gamma 저장 키: calibration 중 probe가 층을 감싸 생기는 `.base_layer` 접미사를 원래 모듈 경로로 정규화해 저장한다(수정 전 코드였다면 본학습 시작에서 KeyError).

## 3. 결과 전 예상 (판정에 쓰지 않음)

- 대리모델 update가 원래보다 약 17% 느리고, F0/E0 cache가 각각 약 36초로 투영된다. 그래서 후보가 시간 20% 절감 기준을 넘기는 어렵다고 예상한다. 계약 §4 [한계]가 예상한 경우이며 설계를 바꾸지 않는다.
- 기준 3(VALUE 대비 0.3%)은 두 seed 차이와 같은 크기일 수 있다. gamma는 경계에 거의 닿지 않을 것이다.

## 4. 계획 검증

봉인 전 plan-critic 검토에서 BLOCKER 2개와 MAJOR 5개가 지적됐다.
- 반영: 위 2절의 1~8.
- 반영하지 않음: "시간 기준이 사실상 결정됨"은 계약 §4가 예상한 결과이고 사용자가 설계 변경을 금지했으므로, 계약을 바꾸지 않고 3절에 기록했다.
