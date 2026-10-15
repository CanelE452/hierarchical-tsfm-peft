# 9개 quantile의 점예측 요약과 MSE: 좁은 전략 검토

상태: **SOURCE·수학 검토 / 새 실제 JSON·NPZ·모델·F·GPU·학습·타이머 실행 0**. 기존 실행 SOURCE와 전략 문서는 수정하지 않았다. 아래는 사전 후보 검토이며 개선 결과가 아니다.

**판정:** median만 사용하던 F에서 점요약을 바꾸는 것은, 같은 정보·E/D 공간을 유지하며 MSE와의 정렬을 확인하는 유효한 작은 진단이다. 평균 proxy가 반드시 더 좋거나 현재 실패의 주원인이라는 결론은 아직 없다. Raw도 동일하게 개선할 기회를 줘야 한다.

## 현재 출력이 무엇인가

[확인: SOURCE] 기존 [FrozenBolt](../../forecast_component_study_20261013/component_experiment.py)는 loaded `model(context=rows).quantile_preds`에서 설정의 .5 위치를 골라 H48을 반환한다. 9개 quantile grid가 .1~.9이면 위치는4다. 현재 디스크의 로컬 `chronos_bolt.py`와 [공식 소스](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos_bolt.py) 모두 `predict_quantiles`의 두 번째 반환값 `mean`에 실제로 .5 quantile을 넣는다. API의 변수명만 바꾸거나 그 반환값을 사용해도 평균 실험이 되지 않는다. 현재 디스크/공식 main 관찰이 과거 loaded 소스의 byte 동일성을 증명하는 것은 아니다.

[확인: 전달된 기존 SOURCE audit] 현 연구에서 9개 출력의 세 점요약을 직접 비교한 실제 실행은 없었다. teacher·D2·native·E/D 학습·현재 global q 보정은 이 비교를 대체하지 않는다. 다만 새로운 평균 추정법이나 확률 예측 모델을 발명한 실험으로 소개할 수는 없다.

유한한 2차 적률과 같은 조건 정보 아래, 제곱오차의 이상적 점예측은 conditional mean이고 절대오차는 median이다. 학습된 median이 MSE로 평가됐다는 사실은 손실과 통계량을 점검할 이유이지 코드 오류의 증거는 아니다. 부정확한 quantile에서 만든 평균 proxy가 학습 median보다 MSE가 낮다는 보장도 없다. [Gneiting, JASA2011 원논문](https://arxiv.org/abs/0912.0902).

## 세 후보와 tail 가정

q_j를 동일한 model forward의 j/10 quantile, j=1,...,9라 하자. 모든 후보는 원 latent 단위로 inverse된 `quantile_preds`에서 만든다.

```text
MEDIAN       q_5
Q9_AVERAGE   (q_1+...+q_9)/9
Q9_INTEGRAL  .15*q_1 + .1*(q_2+...+q_8) + .15*q_9
```

[수학 확인] 진짜 quantile 함수 Q와 유한한 E|Z|에 대해서는 E[Z]=∫_0^1 Q(p)dp다. 이는 U~Uniform(0,1)에 대해 Q(U)가 원 분포를 따른다는 변환으로 얻는다. [Duke 확률강의의 quantile 변환](https://www2.stat.duke.edu/courses/Fall18/sta611.01/Lecture/Lecture05.pdf).

Q9_INTEGRAL은 .1~.9 사이를 선형 보간하고 양쪽 tail을 각각 q_1/q_9로 일정하게 연장한 함수의 적분이다. endpoint 가중치는 tail .1+안쪽 trapezoid .05=.15이고, 일곱 내부 가중치는 .1이어서 합은1이다. 실제 tail을 추정한 것이 아니며 true conditional mean이 아니다. Q9_AVERAGE도9개 값의 고정 평균 proxy다. 두 후보 모두 endpoint 범위를 벗어나지 않지만, 비대칭 tail을 가진 실제 분포의 평균은 그 범위 밖일 수 있다. 9개 quantile은 독립 표본9개나 ensemble member9개가 아니다.

quantile crossing은 별도 기록한다. crossing이 있으면 이 격자를 단조 quantile 함수로 해석할 수 없지만, 고정 가중 점요약의 산술 자체는 정의된다. 결과를 보고 sort·clip·tail 외삽·가중치 적합을 추가하면 새로운 후보/선정 예산이다. 사전 세 후보에 조용히 섞지 않는다.

## 선형 L7에서 평균과 median의 차이

과거·현재 C·각 lag 참조가 고정됐을 때 [원 L7 decoder](linear_e_adaptation_v1.py)는 미래 latent vector Z에 대해 b+A Z인 affine 함수다. 따라서 같은 조건 정보 I 아래

\[
E[b+A Z\mid I]=b+A E[Z\mid I].
\]

이는 D의 부호나 latent 간/시점 간 상관과 무관하다. 단, 각 stream의 F는 해당 stream 과거만 조건으로 예측한다. 그 marginal mean들을 모아도 전체 K 과거를 조건으로 한 평균이라는 보장은 없다. 또한 old D는 근사 복원기이므로 이 항등식만으로 실제 Y의 conditional mean을 얻었다고 할 수 없다.

median에는 일반적인 선형 교환법칙이 없다. 예를 들어 (Z1,Z2)가 (0,0),(0,3),(3,0)을 각각1/3로 취하면 각 median은0이지만 Z1+Z2의 median은3이다. 단일 음수 배수의 quantile은 일반적으로 p 대신1−p를 쓰며, .5 하나의 부호 반전은 가능해도 여러 변수 합의 median 성질을 복구하지는 않는다. 같은 τ끼리 D에 통과시킨 C 곡선을 실제 C quantile로 부르거나, 공동 의존구조 없이 C prediction interval/CRPS로 해석하면 안 된다. 이 실험은 점예측 비교다.

가중치 합이1인 두 평균 proxy에서는 실수 대수상 “요약 미래를 한 번 decode”와 “각 quantile-index 곡선을 decode한 뒤 같은 가중 평균”이 같다. known-past prefix/current-C anchor/lag별 `z_past[-1-lag]`는 그대로 유지해야 한다. 이 항등식은 음수 D를 포함한 합성 산술 검사의 좋은 기준이며 FP32 byte 동일성 보장은 아니다. 고정 D 공간은 바뀌지 않으므로 현재 retained numeric TUNE minimum>Raw라는 반대근거도 사라지지 않는다.

## 가장 작은 공정 대조

- 같은 물리 입력·scaler/origin/mask·current-C·정규화·direct model API·canonical batch를 유지한다. Raw-C와 고정 E0/E1/E2 네 경로 각각에 같은 세 후보를 적용한다. E1/E2가 기존 median 경로로 학습됐던 이력은 그대로 명시한다.
- loaded grid/출력 길이를 검증하고 한 forward의 전체 N×channels×9×64 출력을 보존한다. H48 점요약은 그 저장 출력에서 파생한다. median slice가 기존 계약과 맞는지 먼저 검사하며, 새 API나 normalizer로 바꾸지 않는다. 9개 head는 원래 한 forward에서 함께 나오므로9회 F 실행으로 세지 않는다.
- 후보 가중치·FP32 연산 순서와 tie 규칙을 결과 전에 고정한다. median 우선, 나머지는 고정 순서로 동률을 처리할 수 있다. 경로별 TUNE 자체 관측수의 normalized channel-macro MSE로 선정하고, 네 경로의 선택을 모두 잠근 뒤 DEV 점수를 비교한다. 원 median·세 후보 전체의 paired 차이와 selected-vs-selected Raw 비교를 함께 보고 MAE도 보존한다.
- 압축 proxy를 Raw median과만 비교해 성공을 주장하지 않는다. Raw의 선정 proxy도 개선되면 그 새 기준을 적용한다. 새 summary로 좋아져도 loss-alignment 효과이지 압축/표현학습의 고유 기여가 자동으로 성립하지 않는다. 세 후보가 모두 실패하면 즉시 가중치/tail/quantile sweep을 늘리는 대신 이 가설의 우선순위를 낮춘다.

기존 baseline·checkpoint·F 행동 계약은 보존하고 새 출력 tag를 사용한다. 새 fitting이나 backbone update는 필요하지 않지만 실제 새 F 추론·전체 출력 저장과 선택은 별도 실행이다. 배포 추가 비용은 이미 계산되는9개 출력의 H48 reduction이며, 실제 전체 E/F/D 비용을 재기 전에는 무료 또는 측정 가능한 속도 승리라고 부르지 않는다. DEV는 반복 노출된 탐색 자료이고 보호 TEST는 사용하지 않는다. 개선 여부가 아직 없는 단계에서 “실패 원인 발견”이나 새 방법론 완성으로 포장하지 않는다.
