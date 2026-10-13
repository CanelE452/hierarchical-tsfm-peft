# 평가 기준 정렬과 학습 일반화 재검산

2026-10-13 · 기존 Stage B 자산 재평가 + 같은 FIT/TUNE 조건의 작은 학습 변경

## 결론부터

[확인] 주평가를 TUNE normalized observed channel-macro MSE로 고정해도 기존 PRED checkpoint 선택은 대부분 유지됐다. Jena seed92602만 physical 선택의 update600에서 normalized 선택의 update400으로 바뀌었고, Jena seed92601과 Robin 두 seed는 그대로였다. 하지만 이 선택 정렬 뒤에도 Jena PRED의 VAL normalized MSE는 0.9847-1.0023으로 RAW_BOLT 0.8486보다 높다. 즉 물리 MSE 개선을 주지표 개선으로 바꿔 말할 수 없다.

[확인] 같은 FIT256/TUNE64/VAL origin과 같은 batch order에서 L2, 낮은 learning rate, linear-only를 비교했다. 세 변경 모두 Jena에서는 OLD_PRED보다 나빴고, Robin에서는 normalized MSE를 소폭 낮춘 조건이 있었지만 physical MSE/MAE는 거의 같거나 악화했으며 RAW_BOLT와의 큰 격차는 남았다.

[확인] jena의 이번 주지표 최저는 OLD_PRED seed92602 update400이며 VAL normalized MSE 0.984703, physical MSE 14232.084였다.

[확인] robin의 이번 주지표 최저는 PRED_LR3E4 seed92601 update600이며 VAL normalized MSE 0.594629, physical MSE 246.525였다.

[판단] 현재 병목은 단순히 checkpoint 선택 하나가 아니라, PRED 목적이 일부 큰 단위 채널의 물리 MSE를 낮추는 방향으로 움직이면서 정규화 채널 평균과 MAE를 악화시키는 문제다. 그래서 다음 단계는 새 표현학습보다 loss/selection/decoder를 같은 지표로 묶는 작은 ablation이 먼저다.

## 비교 조건

- 보호 TEST는 열지 않았다. source hash와 origin hash는 `CONTRACTS.json`에 기록했다.

- 정규화와 PCA는 기존처럼 FIT 범위만 사용했다. FIT256/TUNE64/VAL 전체 origin, L512/H48, Jena C21->K5, Robin C17->K4를 유지했다.

- 기존 REC/PRED는 저장된 Stage B checkpoint를 읽어 Bolt로 FIT/TUNE/VAL을 다시 예측했다. 신규 학습은 PRED 계열 세 가지뿐이며, 모두 기존 seed별 batch order를 그대로 썼다.

- 주선택 지표는 TUNE normalized MSE다. physical MSE/MAE와 normalized MAE는 보조 지표다. VAL은 결과 해석용이며 checkpoint 선택에 쓰지 않았다.

## 원수치 요약

아래 표는 주선택 기준인 `tune_normalized_mse`로 고른 PRED 계열만 적었다.

```text
dataset seed method              update  val_norm_mse  val_phys_mse  val_phys_mae
jena    92601 OLD_PRED            600    1.002280      13867.543     38.645
jena    92601 PRED_L2_1E4         600    1.020083      15671.134     43.158
jena    92601 PRED_LINEAR_ONLY    600    1.021288      15676.089     43.159
jena    92601 PRED_LR3E4          600    1.018679      16028.206     43.579
jena    92602 OLD_PRED            400    0.984703      14232.084     38.941
jena    92602 PRED_L2_1E4         400    1.012268      16313.751     43.027
jena    92602 PRED_LINEAR_ONLY    400    1.012710      16318.564     43.036
jena    92602 PRED_LR3E4          600    1.015883      15799.361     43.585
robin   92601 OLD_PRED            200    0.597512      243.308       9.759
robin   92601 PRED_L2_1E4         400    0.596798      244.859       9.841
robin   92601 PRED_LINEAR_ONLY    400    0.596489      243.952       9.807
robin   92601 PRED_LR3E4          600    0.594629      246.525       9.801
robin   92602 OLD_PRED            200    0.602179      245.703       9.779
robin   92602 PRED_L2_1E4         200    0.595343      244.761       9.760
robin   92602 PRED_LINEAR_ONLY    200    0.594719      244.608       9.753
robin   92602 PRED_LR3E4          600    0.599521      246.549       9.809
```

## 그림

![selected_val_normalized_mse.png](selected_val_normalized_mse.png)

![training_generalization_curves.png](training_generalization_curves.png)

![selection_criterion_effect.png](selection_criterion_effect.png)

## Bolt 유지 판단과 연결

이 재검산은 전체 시스템의 순위를 새로 뒤집지 않는다. 이전 결과의 원자료 Bolt/R-only/native+R 대조가 여전히 남아 있고, 이번 추가 결과는 그 전에 주지표를 맞춰야 한다는 경고를 강화한다.

```text
dataset seed method             val_physical_mse  val_physical_mae  val_normalized_mse  val_normalized_mae
jena    NA    RAW_BOLT           13981.151         30.391            0.848632            0.416058
jena    NA    RAW_R_ONLY         10978.293         34.266            0.878199            0.511135
jena    92601 PRED_BOLT          13867.543         38.645            1.002280            0.588501
jena    92601 PRED_BOLT_RAW_R    13058.848         37.560            0.960719            0.565505
jena    92601 PRED_NATIVE_RAW_R  12159.872         36.044            0.871279            0.507685
jena    92602 PRED_BOLT          12988.519         38.911            0.994051            0.597811
jena    92602 PRED_BOLT_RAW_R    12185.094         36.941            0.943587            0.563482
jena    92602 PRED_NATIVE_RAW_R  11702.367         35.612            0.867802            0.508105
robin   NA    RAW_BOLT           151.554           5.939             0.340402            0.313192
robin   NA    RAW_R_ONLY         826.786           16.890            1.142711            0.795741
robin   92601 PRED_BOLT          243.308           9.759             0.597512            0.530728
robin   92601 PRED_BOLT_RAW_R    246.034           9.853             0.600752            0.533185
robin   92601 PRED_NATIVE_RAW_R  307.819           10.895            0.671497            0.571530
robin   92602 PRED_BOLT          245.703           9.779             0.602179            0.530417
robin   92602 PRED_BOLT_RAW_R    249.263           9.894             0.603138            0.533975
robin   92602 PRED_NATIVE_RAW_R  292.330           10.690            0.674133            0.569568
```

## 검산과 한계

- 실행 상태: COMPLETE, wall 401.97s, peak allocated VRAM 306.7 MiB.

- 산출물: `metrics.csv` 300행, `selection_summary.csv` 60행, 그림 3개.

- 검산 통과는 같은 개발 VAL과 저장 자산의 일관성 확인이다. 독립 일반화, 논문급 성능 우위, 또는 Bolt 필요성의 증명으로 쓰지 않는다.

## 다음 판단

1. 주지표를 normalized MSE/MAE로 고정하려면 학습 loss, checkpoint selection, report headline을 모두 이 기준으로 맞춘다.

2. PRED가 Jena에서 physical MSE만 낮추는 현상을 줄이려면 채널별 가중, MAE/Huber 보조항, decoder 고정/재적합을 각각 하나씩 분리해 본다.

3. 이 기준에서도 개선이 없으면 K/lookback/Bolt 대조로 넘어가기 전에 압축 예측 경로의 중심 질문을 '속도를 위해 어느 정도 정규화 오차를 감수할 수 있는가'로 좁히는 편이 맞다.
