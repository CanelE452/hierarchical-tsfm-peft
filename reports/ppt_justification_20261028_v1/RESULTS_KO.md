# PPT 주장 근거 보강 실험 결과

이번 표는 seed 92601–92605의 개발 V 탐색 결과다. 기존 10seed 표와 혼합하지 않는다. 보호 T는 평가하지 않았다.

새 기본 실행 30/30개(필수20, 선택10), 연장 0개 완료. 새 결과가 유리한지에 따라 실행 범위나 판정 기준을 바꾸지 않았다.

```text
Dataset     Method                  n    median MSE      Q1         Q3      params
ETTh1       A5_LAG0                 5  0.281025  0.274169  0.293308    33855
ETTh1       NO_ANCHOR               5  0.806291  0.785138  0.847556    34023
ETTh1       OURS                    5  0.278652  0.277682  0.282602    34023
appliances  NO_ANCHOR               5  0.419277  0.407900  0.430255    37114
appliances  NO_F_LINEAR             5  0.267749  0.262867  0.267831    61738
appliances  OURS                    5  0.282087  0.281950  0.282456    37114
jena        A1                      5  0.768881  0.765829  0.783121     1008
jena        NO_F_LINEAR             5  0.716547  0.709162  0.730922    60341
jena        NO_F_LINEAR_NO_P        5  0.714548  0.703966  0.724058    25632
jena        OURS                    5  0.686402  0.682952  0.703285    35717
```

## 쉽게 읽는 비교 결과

- Jena에서 P를 유지하고 F를 선형 head로 바꿈: 중앙값 MSE +4.39%. 등록된 기술통계 규칙에서는 기준 구성의 오차가 더 낮았다.
- Appliances에서 P를 유지하고 F를 선형 head로 바꿈: 중앙값 MSE -5.08%. 등록된 기술통계 규칙에서는 변형 구성의 오차가 더 낮았다.
- ETTh1에서 마지막 context의 직접 출력 경로만 제거함: 중앙값 MSE +189.35%. 등록된 기술통계 규칙에서는 기준 구성의 오차가 더 낮았다.
- Appliances에서 마지막 context의 직접 출력 경로만 제거함: 중앙값 MSE +48.63%. 등록된 기술통계 규칙에서는 기준 구성의 오차가 더 낮았다.
- ETTh1에서 직접 경로는 유지하고 추가 lag 참조를 제거함: 중앙값 MSE +0.85%. 등록된 기술통계 규칙에서는 차이가 불명확했다. 같거나 효과가 없다는 뜻은 아니다.
- Jena에서 P가 없는 상태로 F를 선형 head로 바꿈: 중앙값 MSE -7.07%. 등록된 기술통계 규칙에서는 변형 구성의 오차가 더 낮았다.
- Jena 선형 head 모델에서 P를 제거함: 중앙값 MSE -0.28%. 등록된 기술통계 규칙에서는 차이가 불명확했다. 같거나 효과가 없다는 뜻은 아니다.
- Jena 고정 F 모델에서 P를 제거함: 중앙값 MSE +12.02%. 등록된 기술통계 규칙에서는 기준 구성의 오차가 더 낮았다.

## 등록 비교와 같은 seed 차이

- F_WITH_P_JENA: NO_F_LINEAR − OURS, 중앙값 MSE +4.39%, REFERENCE_LOWER. 짝 차이 중앙값 +0.030145; reference가 낮은 seed 4/5, variant가 낮은 seed 1/5.
- F_WITH_P_APPLIANCES: NO_F_LINEAR − OURS, 중앙값 MSE -5.08%, VARIANT_LOWER. 짝 차이 중앙값 -0.014201; reference가 낮은 seed 1/5, variant가 낮은 seed 4/5.
- ANCHOR_ETTH1: NO_ANCHOR − OURS, 중앙값 MSE +189.35%, REFERENCE_LOWER. 짝 차이 중앙값 +0.529566; reference가 낮은 seed 5/5, variant가 낮은 seed 0/5.
- ANCHOR_APPLIANCES: NO_ANCHOR − OURS, 중앙값 MSE +48.63%, REFERENCE_LOWER. 짝 차이 중앙값 +0.142551; reference가 낮은 seed 5/5, variant가 낮은 seed 0/5.
- LAG_ETTH1: A5_LAG0 − OURS, 중앙값 MSE +0.85%, UNCLEAR. 짝 차이 중앙값 +0.002373; reference가 낮은 seed 3/5, variant가 낮은 seed 2/5.
- F_WITHOUT_P_JENA: NO_F_LINEAR_NO_P − A1, 중앙값 MSE -7.07%, VARIANT_LOWER. 짝 차이 중앙값 -0.056042; reference가 낮은 seed 0/5, variant가 낮은 seed 5/5.
- P_WITHOUT_F_JENA: NO_F_LINEAR_NO_P − NO_F_LINEAR, 중앙값 MSE -0.28%, UNCLEAR. 짝 차이 중앙값 -0.005195; reference가 낮은 seed 2/5, variant가 낮은 seed 3/5.
- P_WITH_F_JENA: A1 − OURS, 중앙값 MSE +12.02%, REFERENCE_LOWER. 짝 차이 중앙값 +0.085929; reference가 낮은 seed 5/5, variant가 낮은 seed 0/5.

양수는 변형의 오차가 더 크다는 뜻이다. 등록 판정은 두 집단 중앙값 차이를 큰 쪽 IQR과 비교한 기술통계 기준이며 유의성·동등성 검정이 아니다. paired 중앙값도 별도로 공개해 서로 다른 요약을 선택적으로 쓰지 않는다.

## 2×2의 조건부 설명

Jena seed별 [(linear/no-P − linear/P) − (F/no-P − F/P)] 중앙값: -0.086187.
역사적 Full P cap3과 No P cap10의 일정, 선형 대체의 추가 파라미터가 다르므로 인과 상호작용 검정으로 부르지 않는다. 모든 seed별 값은 INTERACTION.json에 있다.

## 각각 무엇을 검증했나

- F 대체는 기존 단일 seed 근거를 확장한다. 선형 head는 512→48이며 추가 학습 파라미터가 24,624개다. F의 중앙값 예측과 선형 head의 결정적 회귀를 비교한다. 결과가 좋더라도 모든 TSFM의 불필요성을 뜻하지 않는다.
- No-anchor는 전처리된 마지막 context 값의 직접 출력 우회만 제거한다. 입력 history와 latent z_last 차분은 남는다. 관측값 전체를 써야 하는지를 시험한 것이 아니다.
- 마지막 context는 원시 마지막 값이 결측이면 forward-fill된 값일 수 있다. 직접 경로의 설계 이유는 현재 수준을 K차원 경로 밖에서 보존하고 learned path가 변화를 모델링하게 하는 데 있다.
- A5 lag0는 anchor를 유지한다. 추가 과거 lag 참조를 제거하는 대조이며 decoder 파라미터 수도 달라진다.
- 역사적 F 1seed, PCA/QR 2seed, 고정 PE reader의 P 반례는 HISTORICAL_EVIDENCE.json과 원 감사 문서에 남겼다. 새 결과로 삭제하지 않는다.

## 실행 범위와 비용

- NO_F_LINEAR/jena: 기본 5/5, 연장 0/5, NOT_REQUIRED; 5seed 주 결과 있음.
- NO_F_LINEAR/appliances: 기본 5/5, 연장 0/5, NOT_REQUIRED; 5seed 주 결과 있음.
- NO_ANCHOR/ETTh1: 기본 5/5, 연장 0/5, NOT_REQUIRED; 5seed 주 결과 있음.
- NO_F_LINEAR_NO_P/jena: 기본 5/5, 연장 0/5, NOT_REQUIRED; 5seed 주 결과 있음.
- A5_LAG0/ETTh1: 기본 5/5, 연장 0/5, NOT_REQUIRED; 5seed 주 결과 있음.
- NO_ANCHOR/appliances: 기본 5/5, 연장 0/5, NOT_REQUIRED; 5seed 주 결과 있음.

```text
Dataset     Method                  device       search_s  refit_s  total_s  add_params
ETTh1       A5_LAG0                cuda           180.76    69.81   257.08          0
ETTh1       NO_ANCHOR              cuda           183.18    74.56   264.58          0
ETTh1       OURS                   not recorded       NA       NA       NA          0
appliances  NO_ANCHOR              cuda           221.18    89.54   323.38          0
appliances  NO_F_LINEAR            cuda            24.62     2.53    33.96      24624
appliances  OURS                   not recorded   212.81    21.63       NA          0
jena        A1                     not recorded   674.51   484.66       NA          0
jena        NO_F_LINEAR            cuda            25.72    10.99    43.30      24624
jena        NO_F_LINEAR_NO_P       cuda            20.05     8.21    36.80      24624
jena        OURS                   not recorded       NA       NA       NA          0
```

시간은 실제 기록의 중앙값이며 NA는 미기록이다. 추론 속도나 모델에 고유한 비용으로 일반화하지 않는다. 구체적 selected epochs/cap/provisional은 SUMMARY.csv에 있다.

## 아직 충분하지 않은 근거

1. E 고정 대 학습의 직접 비교는 이번 범위에 없다. PCA/QR은 E와 decoder 초기화를 함께 바꾼다.
2. C2/C3의 학습 상한 문제와 기존 비교의 잠정 상태는 해소되지 않는다.
3. P의 정보 보존 메커니즘은 성능 대조만으로 증명되지 않는다. 기존 선형 reader 반례도 남는다.
4. n=5 파일럿·반복 확인한 V이므로 일반성, 동등성, 독립 테스트 성능을 확정할 수 없다.
5. 선택 실험이 예산상 빠졌다면 그 질문은 미해결이다. 문구 정리는 실증 해결과 다르다.
