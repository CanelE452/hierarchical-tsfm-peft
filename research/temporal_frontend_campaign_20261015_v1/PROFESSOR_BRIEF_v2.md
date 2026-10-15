# 교수님 상담 설명안: 고정 Bolt용 시간 앞단의 현재 판단

아래는 7장 설명용 Markdown 구성안이다. 원 논문의 새 성능 재현이나 완성된 논문 기여를 주장하는 PPT 파일이 아니다.

## 1. 문제: 고정 단변량 TSFM에 무엇을 적응시키는가

원채널 미래 정확도가 먼저이며, 작은 K와 실제 전체 비용은 그 이득을 유지할 수 있는지 묻는 조건이다. 입력 복원이 잘된다는 이유로 미래 예측 표현이 충분하다고 가정하지 않는다. 과거만으로 P/E를 만들고 실제 원미래 손실로 학습했다.

## 2. 교수님 제안과 표준 대안

Google TSMixer의 시간/채널 혼합, TimeMixer PDM의 분해·다중해상도 혼합을 원격자 과거 앞단으로 변형했다. AdaPTS식 E/D 예측 학습과 SPRINT식 단순 시간 변환은 선행 대안이다. FMM 대신 Bolt를 붙인 변형이며 원 논문 전체 재현이나 이 원리 자체의 신규성 주장은 없다. [원문 대응](LITERATURE_MAPPING.md).

## 3. 단순 시간 변환부터

평활·잔여·분리 예측·endpoint/blockmean·SPRINT-on-Bolt·긴 lookback의 원미래 C 오차를 같은 origin의 Raw와 비교했다. R1 음성 결과로 학습형 경로를 취소하지 않았다. 실제 결과는 [그림1](figures/01_time_transforms_v2.png)과 SCORE 원표에 있다.

## 4. 학습형 P와 C/K

같은 lag7 D와 현재 C anchor에서 B0–B4×C/K×2seed를 실제 학습했다. B로 선택한 작은 K 후보의 원수치: beijing B4/r2 C MSE 0.808416 / jena B2/r5 C MSE 1.106624. [그림2](figures/02_temporal_rank_v2.png)의 두 seed 범위는 신뢰구간이 아니다. C/K는 stream 수·E/D 파라미터 수·복원 공간을 함께 바꾸므로 압축 단독의 인과 효과가 아니다. 같은 C/r의 B3/B4는 파라미터 구조·개수가 같은 별도 P-family 대조다. 최대 update에서도 하강하면 미수렴·미해결로 남긴다.

## 5. 같은 G와 Bolt/native 대조

Raw+G/B0K+G/B2K+G/B4K+G/G-only의 입력·클래스를 맞췄다. 고정 P/E/D에서 실제 원미래에 맞춘 native144계수와 직접 작은 예측기·self ridge·한 LoRA 구성도 비교했다. native/ridge는 모든 합법 A origin을 누적하고 신경망은 같은 A 범위의 minibatch prefix여서 실제 표본 노출·optimizer update까지 같은 비교는 아니다. 좋은 full 결합만 보여주지 않고 G-only 및 Raw+G와의 차이를 함께 읽는다. [그림3](figures/03_matched_residual_v2.png).

## 6. 시간 재적합과 전체 비용

기존 FIT 안에서 사전 고정한 두 경계에 모든 upstream을 다시 적합했다. C는 이번 선택에서 잠갔지만 과거에 미노출인 독립 holdout은 아니다. physical CPU 입력부터 출력 반환까지 batch1/8의 전체 시간·stage·별도 peak 메모리를 측정했다. [학습/시간 그림](figures/04_learning_time_refit_v2.png), [비용 그림](figures/05_error_time_memory_v2.png).

```text
method             Beijing_C_MSE  Jena_C_MSE
Raw                0.655904       1.223530  
B-selected smallK  0.808416       1.106624  
G-only             0.612079       1.017580  
LoRA qv8           0.582379       0.914889  
```

시간 앞단이 r=C에서 도움이 될 수 있지만 작은 K+G의 G-only 대비 추가 가치는 확보하지 못했다. LoRA도 강한 반례이며 현재 결과를 성공 방법론이나 신규성 확정으로 부르지 않는다. 좁힌 질문은 **원채널 예측의 이득을 소수 F streams에서도 단순 직접예측/LoRA보다 유지할 수 있는가?**다. 현재 답은 미확보다. 위 smallK는 C로 다시 고른 모델이 아니라 기존 locked B 선택이며 두 seed/시간 반복의 한계를 유지한다.

## 7. 남는 기여와 보류할 주장

이번 답은 표준 대안 대비 추가 가치가 남는 조건을 확인하는 것이다. 원수치가 나쁘거나 미수렴이면 그 상태를 보존한다. G-only나 native/direct가 충분하면 새 앞단·Bolt의 필요성은 보류한다. 재구성/표현/예측기/디코더 중 원인 하나를 자동 확정하지 않고, matched 비교와 시간 refit의 범위 안에서만 좁힌다. [전체 결과·실패](REPORT_KO_v2.md), [고정/사후 사례](figures/06_forecast_cases_v2.png).
