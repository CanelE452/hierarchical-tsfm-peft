# GTConvAE 긴 관측 과거와 TS2Vec 잠재 결합 빠른 검증

전체 상태: **COMPLETE**. 검증된 예측 33/33개, 최종 추가학습 8/8개, 비용 42/42개.

## 질문 1. 고정한 GT에 긴 과거를 주면 예측이 좋아지는가

[확인] 아래 수치는 같은 VAL 원점·정답·관측 mask·TRAIN 정규화에서 독립 FP64로 다시 계산했다. 음수 변화는 오차 감소다.

- A_GT_L1024_s92601 대 A_GT_L512_s92601: seed 92601: MSE 2.0646974 대 2.0407656, 변화 +1.1727%; MAE 변화 +0.8317%
- A_GT_L2048_s92601 대 A_GT_L512_s92601: seed 92601: MSE 1.9638557 대 2.0407656, 변화 -3.7687%; MAE 변화 -3.7035%
- A_GT_L1024_s92602 대 A_GT_L512_s92602: seed 92602: MSE 0.8033349 대 0.77434712, 변화 +3.7435%; MAE 변화 +3.5798%
- A_GT_L2048_s92602 대 A_GT_L512_s92602: seed 92602: MSE 0.57502585 대 0.77434712, 변화 -25.7406%; MAE 변화 -16.7269%
L1024의 고정 GT MSE가 자기 L512보다 낮은 seed는 0/2개다. 이 결과는 기존 부모에 긴 관측 범위를 준 효과이며 새 reconstruction 사전학습 결과가 아니다.
L2048의 고정 GT MSE가 자기 L512보다 낮은 seed는 2/2개다. 이 결과는 기존 부모에 긴 관측 범위를 준 효과이며 새 reconstruction 사전학습 결과가 아니다.

Raw의 같은 길이 변화와 GT 대 Raw의 같은 길이 비교를 함께 본다. GT가 자기 L512보다 좋아졌다는 사실만으로 실용 기준선을 이겼다고 말하지 않는다.

- A_RAW_L1024 대 A_RAW_L512: MSE 변화 -13.4253%
- A_RAW_L2048 대 A_RAW_L512: MSE 변화 -16.5883%
- A_GT_L512_s92601 대 A_RAW_L512: MSE 변화 +267.7577%
- A_GT_L1024_s92601 대 A_RAW_L1024: MSE 변화 +329.7679%
- A_GT_L2048_s92601 대 A_RAW_L2048: MSE 변화 +324.2787%
- A_GT_L512_s92602 대 A_RAW_L512: MSE 변화 +39.5418%
- A_GT_L1024_s92602 대 A_RAW_L1024: MSE 변화 +67.2146%
- A_GT_L2048_s92602 대 A_RAW_L2048: MSE 변화 +24.2307%
같은 길이 Raw보다 고정 GT의 MSE가 낮은 길이×seed 조합은 0/6개다. 자기 이전 결과의 개선과 Raw 기준선 초과를 구분한다.

## 질문 2. 같은 추가학습 예산에서 TS2Vec 사전학습 표현이 도움이 되는가

사전 고정 예산: FIXED_300_UPDATES / 최종 300 updates. 모든 arm은 같은 seed별 전체 TRAIN 원점 순서를 쓰고 마지막 작은 batch를 유지한다. VAL로 최선 checkpoint를 고르지 않는다.

- ADAPTED_GT_LOOKBACK: seed 92601: MSE 2.0358632 대 2.0169513, 변화 +0.9377%; MAE 변화 +0.7251%
- PRE_VS_GT: seed 92601: MSE 1.8968431 대 2.0358632, 변화 -6.8286%; MAE 변화 -2.9029%
- PRE_VS_RAND: seed 92601: MSE 1.8968431 대 1.8913882, 변화 +0.2884%; MAE 변화 -0.2006%
- ADAPTED_GT_LOOKBACK: seed 92602: MSE 0.79291004 대 0.77122016, 변화 +2.8124%; MAE 변화 +0.8027%
- PRE_VS_GT: seed 92602: MSE 0.68348789 대 0.79291004, 변화 -13.8001%; MAE 변화 -7.2861%
- PRE_VS_RAND: seed 92602: MSE 0.68348789 대 0.69149186, 변화 -1.1575%; MAE 변화 -1.3859%
PRE가 GT1024와 RAND를 두 seed에서 모두 이긴다는 근거는 확보되지 않았다. seed별 차이와 비용을 유지해 해석한다.

최종 Phase B 모델과 같은 L의 Raw 기준선을 별도로 비교한다. CP0 대비 변화는 FINAL_VS_CP0에, 아래 Raw 대비 변화는 ADAPTED_VS_RAW에 기록한다.

- B_GT_FT_L512_s92601 대 A_RAW_L512: seed 92601: MSE 2.0169513 대 0.55492124, 변화 +263.4662%; MAE 변화 +167.3393%
- B_GT_FT_L1024_s92601 대 A_RAW_L1024: seed 92601: MSE 2.0358632 대 0.48042145, 변화 +323.7661%; MAE 변화 +205.7762%
- B_GT_TS_PRE_L1024_s92601 대 A_RAW_L1024: seed 92601: MSE 1.8968431 대 0.48042145, 변화 +294.8290%; MAE 변화 +196.8998%
- B_GT_TS_RAND_L1024_s92601 대 A_RAW_L1024: seed 92601: MSE 1.8913882 대 0.48042145, 변화 +293.6935%; MAE 변화 +197.4967%
- B_GT_FT_L512_s92602 대 A_RAW_L512: seed 92602: MSE 0.77122016 대 0.55492124, 변화 +38.9783%; MAE 변화 +38.0055%
- B_GT_FT_L1024_s92602 대 A_RAW_L1024: seed 92602: MSE 0.79291004 대 0.48042145, 변화 +65.0447%; MAE 변화 +57.9689%
- B_GT_TS_PRE_L1024_s92602 대 A_RAW_L1024: seed 92602: MSE 0.68348789 대 0.48042145, 변화 +42.2684%; MAE 변화 +46.4592%
- B_GT_TS_RAND_L1024_s92602 대 A_RAW_L1024: seed 92602: MSE 0.69149186 대 0.48042145, 변화 +43.9344%; MAE 변화 +48.5175%
GT_FT_L512: 같은 L512 Raw보다 MSE가 낮은 seed 0/2개, MAE가 낮은 seed 0/2개다. MSE 동일/높음 0/2개, MAE 동일/높음 0/2개다.
GT_FT_L1024: 같은 L1024 Raw보다 MSE가 낮은 seed 0/2개, MAE가 낮은 seed 0/2개다. MSE 동일/높음 0/2개, MAE 동일/높음 0/2개다.
GT_TS_PRE_L1024: 같은 L1024 Raw보다 MSE가 낮은 seed 0/2개, MAE가 낮은 seed 0/2개다. MSE 동일/높음 0/2개, MAE 동일/높음 0/2개다.
GT_TS_RAND_L1024: 같은 L1024 Raw보다 MSE가 낮은 seed 0/2개, MAE가 낮은 seed 0/2개다. MSE 동일/높음 0/2개, MAE 동일/높음 0/2개다.

고정 GT의 추론과 예측 목적 추가학습은 서로 다른 질문이다. A0는 과거 loader의 전처리 재현이며 새 창별 전처리 비교에서 제외한다. CP0와 최종 모델을 모두 공개하고, 출력 0.5 평균은 보조 대조이며 잠재 결합 성공을 대신하지 않는다.

두 seed 산술평균도 함께 남긴다. 아래 평균은 반복 두 개의 요약이며 신뢰구간이 아니다.

```text
group                         seeds   MSE_mean        MAE_mean
Frozen_GT_L512                2       1.40755635      0.754172137
Frozen_GT_L1024               2       1.43401615      0.767357024
Frozen_GT_L2048               2       1.26944079      0.693483264
GT_FT_L512                    2       1.39408573      0.757779014
GT_FT_L1024                   2       1.41438664      0.76347371
GT_TS_PRE_L1024               2       1.29016552      0.729911623
GT_TS_RAND_L1024              2       1.29144005      0.734282921
```

출력 평균의 보조 결과:

- OUTPUT_ENSEMBLE_VS_GT: seed 92601: MSE 0.8686071 대 2.0646974, 변화 -57.9305%; MAE 변화 -39.0535%
- OUTPUT_ENSEMBLE_VS_TS2: seed 92601: MSE 0.8686071 대 0.47029773, 변화 +84.6930%; MAE 변화 +60.1058%
- OUTPUT_ENSEMBLE_VS_GT: seed 92602: MSE 0.53491659 대 0.8033349, 변화 -33.4130%; MAE 변화 -21.2413%
- OUTPUT_ENSEMBLE_VS_TS2: seed 92602: MSE 0.53491659 대 0.4670726, 변화 +14.5254%; MAE 변화 +5.8503%

## 고정 TRAIN 진단과 수렴 해석

300 고정 updates 완료는 수렴을 증명하지 않는다. 아래 CP0·100·300점은 같은 고정 TRAIN 진단 원점에서 측정하고 checkpoint 선택이나 조기 종료에 사용하지 않은 기술 통계다. 미니배치 loss와 구분하며, 과거 재구성 점수는 보조 진단이다. 100→300 변화는 U300−U100의 절대 차이이고 음수는 해당 오차 감소다. 기록하지 않은 진단점과 변화는 수치로 채우지 않는다.

```text
run_id                              metric                 CP0             U100            U300            delta_100_to_300
B_GT_FT_L1024_s92601                forecast_mse           0.902195586     0.902640132     0.900958215     -0.00168191737
B_GT_FT_L1024_s92601                forecast_mae           0.624710908     0.629158356     0.633416067     0.0042577108
B_GT_FT_L1024_s92601                reconstruction_mse     0.652406502     0.652106566     0.647684134     -0.004422432
B_GT_FT_L1024_s92601                reconstruction_mae     0.461641268     0.465923571     0.467902944     0.00197937301
B_GT_FT_L1024_s92602                forecast_mse           0.534440582     0.53277921      0.52987795      -0.00290126025
B_GT_FT_L1024_s92602                forecast_mae           0.4322126       0.435044574     0.435330177     0.000285603571
B_GT_FT_L1024_s92602                reconstruction_mse     0.133312693     0.134597428     0.134968846     0.000371418432
B_GT_FT_L1024_s92602                reconstruction_mae     0.177420819     0.183755114     0.186116654     0.00236154027
B_GT_FT_L512_s92601                 forecast_mse           0.895385085     0.896033647     0.896060984     2.73374729e-05
B_GT_FT_L512_s92601                 forecast_mae           0.62214243      0.626925877     0.63146718      0.00454130394
B_GT_FT_L512_s92601                 reconstruction_mse     0.754340981     0.754042273     0.750487027     -0.00355524601
B_GT_FT_L512_s92601                 reconstruction_mae     0.496500916     0.501109517     0.503538799     0.00242928203
B_GT_FT_L512_s92602                 forecast_mse           0.531438363     0.527818795     0.524872137     -0.00294665741
B_GT_FT_L512_s92602                 forecast_mae           0.430345572     0.437425294     0.439125439     0.00170014526
B_GT_FT_L512_s92602                 reconstruction_mse     0.14937633      0.15600257      0.158803144     0.00280057313
B_GT_FT_L512_s92602                 reconstruction_mae     0.184396162     0.204211088     0.211018769     0.00680768108
B_GT_TS_PRE_L1024_s92601            forecast_mse           0.902195586     0.840870219     0.767871218     -0.0729990008
B_GT_TS_PRE_L1024_s92601            forecast_mae           0.624710908     0.585856192     0.55718011      -0.0286760824
B_GT_TS_PRE_L1024_s92601            reconstruction_mse     0.652406502     0.651262915     0.644287425     -0.00697549016
B_GT_TS_PRE_L1024_s92601            reconstruction_mae     0.461641268     0.466063639     0.470405032     0.00434139356
B_GT_TS_PRE_L1024_s92602            forecast_mse           0.534440582     0.432025208     0.377956158     -0.0540690493
B_GT_TS_PRE_L1024_s92602            forecast_mae           0.4322126       0.382408068     0.359152623     -0.0232554453
B_GT_TS_PRE_L1024_s92602            reconstruction_mse     0.133312693     0.13398463      0.135046446     0.00106181559
B_GT_TS_PRE_L1024_s92602            reconstruction_mae     0.177420819     0.180977126     0.183356778     0.0023796517
B_GT_TS_RAND_L1024_s92601           forecast_mse           0.902195586     0.829239588     0.775811236     -0.0534283516
B_GT_TS_RAND_L1024_s92601           forecast_mae           0.624710908     0.57712498      0.559502262     -0.0176227175
B_GT_TS_RAND_L1024_s92601           reconstruction_mse     0.652406502     0.655059662     0.659568777     0.00450911466
B_GT_TS_RAND_L1024_s92601           reconstruction_mae     0.461641268     0.475407699     0.496141288     0.0207335889
B_GT_TS_RAND_L1024_s92602           forecast_mse           0.534440582     0.459501787     0.410909397     -0.0485923904
B_GT_TS_RAND_L1024_s92602           forecast_mae           0.4322126       0.397191048     0.377943151     -0.0192478965
B_GT_TS_RAND_L1024_s92602           reconstruction_mse     0.133312693     0.138535926     0.14201644      0.00348051329
B_GT_TS_RAND_L1024_s92602           reconstruction_mae     0.177420819     0.195942074     0.205116766     0.00917469274
```

## 비용과 자원

마지막 기록 종료까지 wall 3800.426초, 원장에 확정된 GPU 활성 wall 2943.328초. 모델 적재·창 슬라이스·특징 캐시 생성과 timed whole-route 범위를 구분한다.

완료한 최종 adaptation fit 8개와 버린 TRAIN smoke fit 4개를 구분한다. 신규 GT reconstruction/TS2 대규모 사전학습은 0개다. smoke 비용도 GPU 원장에 포함한다.

메모리 수치는 시간–오차 그림과 분리한다. 전체 원정밀도는 [COST.csv](COST.csv), 개별 측정은 [COST_TRIALS.csv](COST_TRIALS.csv)에 있다.

비용의 p50·IQR·p95는 고정 원점 그룹 3개를 순환한 20회 반복의 경험적 요약이다. 서비스 부하나 독립 요청의 tail 분포를 표집한 실험이 아니므로 서비스 tail 지연 보장으로 해석하지 않는다.

```text
run_id                                    B   allocated_MiB   reserved_MiB
A_ENSEMBLE_L1024_s92601                   1         484.546        566.000
A_ENSEMBLE_L1024_s92601                   8         498.933        588.000
A_ENSEMBLE_L1024_s92602                   1         484.546        566.000
A_ENSEMBLE_L1024_s92602                   8         498.933        588.000
A_GT_L1024_s92601                         1         238.873        268.000
A_GT_L1024_s92601                         8         290.892        308.000
A_GT_L1024_s92602                         1         238.873        268.000
A_GT_L1024_s92602                         8         290.892        308.000
A_GT_L2048_s92601                         1         259.999        286.000
A_GT_L2048_s92601                         8         354.213        430.000
A_GT_L2048_s92602                         1         259.999        286.000
A_GT_L2048_s92602                         8         354.213        430.000
A_GT_L512_s92601                          1         228.309        274.000
A_GT_L512_s92601                          8         256.606        266.000
A_GT_L512_s92602                          1         228.309        274.000
A_GT_L512_s92602                          8         256.606        266.000
A_RAW_L1024                               1         258.476        280.000
A_RAW_L1024                               8         481.974        566.000
A_RAW_L2048                               1         301.612        336.000
A_RAW_L2048                               8         736.663        982.000
A_RAW_L512                                1         237.432        260.000
A_RAW_L512                                8         352.629        386.000
A_TS2_L1024_s92601                        1         484.541        562.000
A_TS2_L1024_s92601                        8         498.927        588.000
A_TS2_L1024_s92602                        1         484.541        562.000
A_TS2_L1024_s92602                        8         498.927        588.000
B_GT_FT_L1024_s92601                      1         238.873        268.000
B_GT_FT_L1024_s92601                      8         290.892        308.000
B_GT_FT_L1024_s92602                      1         238.873        268.000
B_GT_FT_L1024_s92602                      8         290.892        308.000
B_GT_FT_L512_s92601                       1         228.309        274.000
B_GT_FT_L512_s92601                       8         256.606        266.000
B_GT_FT_L512_s92602                       1         228.309        274.000
B_GT_FT_L512_s92602                       8         256.606        266.000
B_GT_TS_PRE_L1024_s92601                  1         241.994        270.000
B_GT_TS_PRE_L1024_s92601                  8         297.889        322.000
B_GT_TS_PRE_L1024_s92602                  1         241.994        270.000
B_GT_TS_PRE_L1024_s92602                  8         297.889        322.000
B_GT_TS_RAND_L1024_s92601                 1         241.994        270.000
B_GT_TS_RAND_L1024_s92601                 8         297.889        322.000
B_GT_TS_RAND_L1024_s92602                 1         241.994        270.000
B_GT_TS_RAND_L1024_s92602                 8         297.889        322.000
```

## 주장 범위와 검증 한계

[확인] Jena 한 데이터·두 seed의 반복 개발 VAL 결과다. 기존 부모의 VAL 선택과 과거 TEST 노출 이력을 제거한 독립 평가가 아니다. 이번 보호 TEST의 읽기·해시·요약은 수행하지 않는다. 사전학습 overlap은 UNKNOWN이다. 두 seed의 차이·표준편차를 일반화 신뢰구간으로 해석하지 않는다.

[확인] historical raw_physical SHA 불일치는 미해결 상태로 남긴다. 현재 unchanged loader와 runtime raw 배열 및 기존 6개 모델 좌표 지문·계약의 일치를 별도로 기록한다. 이 차이의 원인을 해결했다고 주장하지 않는다.

[확인] GT 경로는 시간축 L→L/4로 4배 축소하지만 latent feature가 2개이므로 scalar 수는 입력 LC에서 2C(L/4)=LC/2로 2배 압축된다. 원채널 C=21을 유지하며 F 시계열은 2C=42개다. TS2 단독은 320개이며 결합은 42개를 유지하지만 encoder 상주·연산이 추가된다. B2의 C→K 채널 압축이나 더 가벼운 전체 시스템을 입증하지 않는다. 실제 F context가 잘렸다면 요청 길이와 실제 길이를 COST_TRIALS에서 함께 확인한다.

겹치는 원점은 독립 표본이 아니다. origin별 개선율은 기술 통계이며 사후 통과 기준으로 쓰지 않는다. 학습 미니배치 곡선과 고정 TRAIN 진단점을 구분하고, 과거 재구성은 보조 진단으로만 본다. 정상 실행 후 미개선과 검증 실패·미완료를 JOB_STATUS에서 구분한다.

[OBSERVATION_CONTEXT_STATS.csv](OBSERVATION_CONTEXT_STATS.csv)의 fully observed subset은 비교 양쪽의 물리 과거 모든 cell이 관측된 원점을 관측 mask만으로 정한 보조 기술 통계다. Phase A 이후에 기준을 등록한 사후 기술 진단이며, 학습·점수 선택 기준이 아니다. 전체 371개 VAL 원점의 주 결과를 교체하지 않는다.

## 재현 가능한 산출물

- [고정 모델 길이 비교](FIG1_Lookback.png) · [SVG](FIG1_Lookback.svg) · [source CSV](FIG1_Lookback_source.csv)
- [예측 추가학습과 대응 차이](FIG2_Adaptation.png) · [SVG](FIG2_Adaptation.svg) · [source CSV](FIG2_Adaptation_source.csv)
- [전체 지연시간–오차](FIG3_Cost.png) · [SVG](FIG3_Cost.svg) · [source CSV](FIG3_Cost_source.csv)
- [RUNS](RUNS.csv), [COMPARISONS](COMPARISONS.csv), [채널·horizon 충분통계](CHANNEL_HORIZON_STATS.csv), [원점별 효과](ORIGIN_EFFECTS.csv), [학습 곡선](TRAINING_CURVES.csv)
- [방법 변경과 원문](METHOD_DELTA_KO.md), [자원 원장](RESOURCE_LEDGER.json), [작업 상태](JOB_STATUS.csv), [재개 큐](RESUME_QUEUE.md)

```bash
python -m research.gt_ts2_fastcheck_20261018_v1.run report
python -m research.gt_ts2_fastcheck_20261018_v1.run verify
```

공개 CSV·그림은 receipt에 묶인 private prediction에서 다시 계산한다. 원자료·prediction 배열·pretrained weights·로컬 절대경로는 공개하지 않는다. 게시 commit와 원격 bytes 검증은 별도 publication proof를 따른다.
