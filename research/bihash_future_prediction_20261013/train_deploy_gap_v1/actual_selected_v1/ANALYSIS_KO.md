# 고정 체크포인트의 실제 학습 코드와 배포 sign 차이

[확인] 네 자료·세 seed·네 방법 48조건에서 원 FrozenBolt와 원 모델 forward로 배포 sign 및 실제 학습 모드를 비교했다. 각 조건은 선택된 동일 가중치와 사전 고정 마지막 8 FIT 창, 8 noise draw를 사용했다. 학습·checkpoint 재선택은 없다.

[확인] 원 E/phi/fusion에는 활성 Dropout·BatchNorm이 없고 FrozenBolt는 eval을 유지한다. 전체 model.train()에서 memory를 재계산한 결과와 cached 배포 memory는 모든 조건에서 예측까지 정확히 같았다. 각 draw 후 weights·buffers·counter를 복구했고 backbone 및 소스16 SHA가 유지됐다. 실제 노이즈, 코드, 예측, target/mask, eligibility, seed/counter를 NPZ에 저장했다.

Primary 온도는 `model.temperature(selected_update)`다. 선택 u100/.875, u200/.75, u400/.5, u600/.25가 실제 존재한다. 이는 선택 가중치에서 다음 학습 모드의 forward를 재는 진단이다. 마지막 과거 optimizer step은 update-1을 사용했으므로 과거 학습 step 재연으로 부르지 않는다. 기존 inference-only tanh 변형은 고정 .25이며 이 진단의 선택-update tanh와 다르다.

```text
자료                         방법                 sign MSE      train MSE     train-sign(%)
jena                         B_GUMBEL               0.41780293    0.42813472    2.473
jena                         B_ST                   0.42231306    0.43679108    3.428
jena                         B_FIXED_GUMBEL         0.43654688    0.45656597    4.586
jena                         B_TANH                 0.38727908    0.37710234   -2.628
robin                        B_GUMBEL               0.08328237    0.08463809    1.628
robin                        B_ST                   0.08273142    0.08413524    1.697
robin                        B_FIXED_GUMBEL         0.08579611    0.08644827    0.760
robin                        B_TANH                 0.08714808    0.08151142   -6.468
uci274_sml2010               B_GUMBEL               0.09557833    0.09630295    0.758
uci274_sml2010               B_ST                   0.09297004    0.09407738    1.191
uci274_sml2010               B_FIXED_GUMBEL         0.10026750    0.10089518    0.626
uci274_sml2010               B_TANH                 0.09826767    0.09648224   -1.817
uci235_household_hourly      B_GUMBEL               0.86763042    0.86963363    0.231
uci235_household_hourly      B_ST                   0.86387260    0.86736525    0.404
uci235_household_hourly      B_FIXED_GUMBEL         0.86103393    0.86782849    0.789
uci235_household_hourly      B_TANH                 0.88054314    0.87952300   -0.116
```

[판단] 실제 Gumbel 코드 전환은 존재한다. B_GUMBEL의 eligible query bit flips는 자료별 평균 약14.6~24.7%다. 그러나 네 자료에서 noisy train-mode의 평균 FIT MSE가 sign 배포보다 .23~2.47% 높았다. 따라서 이번 표본에서 sign 배포가 오차를 악화시킨다고 볼 수 없으며, 기존 전체 성능 실패를 이 차이 하나로 설명할 수 없다.

[판단] B_TANH는 sign이 바뀌지 않아도 코드의 크기와 cosine 각도가 달라져 출력이 달라진다. 실제 tanh 학습 모드보다 sign 배포의 FIT MSE가 네 자료 평균 약 .12~6.47% 높았다. 이는 제한된 FIT mechanism 진단이므로 sign 대신 tanh가 일반화에 더 좋다는 결론이나 EVAL에서의 배포 방식 재선택 근거로 바꾸지 않는다.

두 범주 logits `[I,-I]`에 독립 표준 Gumbel을 넣으면 두 노이즈의 차이는 logistic이고, deterministic sign을 뒤집을 이론 확률은 `sigmoid(-2|I|)`다. 공통 양의 온도는 argmax를 바꾸지 않으므로 낮은 온도가 노이즈 flip을 없애는 것은 아니다. 원 구현의 uniform clamp·FP32 rounding과 경험적 유한 draw 변동은 별도 한계다.

FIT 창은 시간 중첩을 가질 수 있다. 8 draw는 재학습 seed나 독립 검증 표본이 아니다. Legacy TRAIN/DEVVAL와 fresh FIT/TUNE가 shared 압축 컨테이너에 저장된 원 구조 때문에 그 입력 member는 함께 decompress되지만 진단에 조립·채점한 창은 FIT뿐이다. EVAL target·TEST를 열거나 hash하지 않았다.

원수치와 provenance: [RESULT](RESULT.json), [조건별 CSV](CONDITION_METRICS_v1.csv), [자료·방법별 CSV](GROUP_SUMMARY_v1.csv), [CPU 원수치 재검산](NUMERIC_RECHECK_v1.json). 별도 agent의 전체 독립 감사를 대체하지 않는다.
