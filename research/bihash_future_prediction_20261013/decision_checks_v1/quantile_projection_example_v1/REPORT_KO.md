# PCA와 분위수 예측의 변환 순서: 합성 반례

[확인] 실제 로컬 FrozenBolt wrapper는 0.5 분위수를 선택한다. 아래는 실제 자료의 실험이 아닌 네 가지 미래 상태의 정확한 확률 계산이다.

상태 (0,0)/(10,0)/(0,10)/(10,10)의 확률은 .15/.20/.20/.45다. 채널별 median은 (10,10)이고, 이 분포의 실제 최상위 PCA 방향은 (1,1)/sqrt(2)다.

RAW median forecast는 rank1 공간으로 정확히 표현된다. 하지만 PCA로 투영한 미래 분포의 median을 선형 복원하면 (5,5)가 된다. 따라서 RAW forecast의 낮은 rank만으로 압축된 과거의 TSFM forecast가 같은 값을 낼 것이라고 가정할 수 없다.

```text
예측                         MSE       MAE
원채널별 median (10,10)      35.00      3.50
잠재 median 복원 (5,5)       25.00      5.00
원채널별 mean (6.5,6.5)      22.75      4.55
```

[판단] 이 반례는 압축된 분위수의 복원이 MSE를 낮추면서 MAE를 높일 수 있음을 보여준다. 실제 Robin/Beijing/Appliances의 원인을 확정하지 않는다. 학습된 decoder가 이를 보정할 가능성도 있으므로 방법의 보편적 한계나 복원 불가능성을 입증한 것은 아니다.

후속 D1에서는 RAW forecast 투영과 투영된 과거의 실제 Bolt forecast를 분리한다. 평균/분위수 목적의 차이는 실제 FIT/TUNE 선택과 동일 baseline 비용을 맞춰 검증해야 한다.

[원수치와 소스 SHA](RESULT.json) · [로컬 구현 근거](../../../forecast_component_study_20261013/component_experiment.py)
