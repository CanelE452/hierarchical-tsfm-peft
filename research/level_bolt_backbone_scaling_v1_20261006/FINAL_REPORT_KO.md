# Chronos-Bolt backbone allocation 비교

질문: 큰 Bolt-small의 처리 series를 C→K로 줄이고 기존 LEVEL로 보정하는 선택은 더 작은 동일 계열 backbone으로 C개 원채널을 그대로 처리하는 선택보다 정확도–추론비용의 선택 이유가 남는가?

이번 결과는 LEVEL method ablation이 아닌 **same-family practical allocation comparison**입니다. backbone size와 channel allocation, 기존 target-data adaptation 이력이 함께 다릅니다. 모든 세 자료는 이미 TEST가 노출된 후속 비교입니다.

## 비교 공정성

[확인] 아래 값은 봉인된 자료·모델 manifest, 사전 실제 로드 receipt와 이번 비용 receipt에서 읽었습니다. 추론 중 gradient를 사용하는 변수 수는 모든 arm에서 0이며, target-data fit 변수 수는 기존 LEVEL의 G 적합 이력입니다. 이번 회차 새 fit은 0입니다.

Arm names: `BOLT_SMALL_LEVEL`, `BOLT_SMALL_F0`, `BOLT_MINI_F0`, `BOLT_TINY_F0`.

## Robin — C17/K5

| Attribute | Small + LEVEL | Small F0 | Mini F0 | Tiny F0 |
| --- | --- | --- | --- | --- |
| Family | Chronos-Bolt | Chronos-Bolt | Chronos-Bolt | Chronos-Bolt |
| Architecture | ChronosBoltModelForForecasting | ChronosBoltModelForForecasting | ChronosBoltModelForForecasting | ChronosBoltModelForForecasting |
| Pinned model revision | 772f3d25d38aec6d914c8949dab4462e2d46f5d8 | 772f3d25d38aec6d914c8949dab4462e2d46f5d8 | 251268337516a88e253628c43e1d26ec577b376b | a0e552de83495b5c28c14c71c374f3e33280b340 |
| Actual backbone params | 47718016 | 47718016 | 21236096 | 8652672 |
| Actual backbone parameter bytes | 190872064 | 190872064 | 84944384 | 34610688 |
| Total deployed params | 47736106 | 47718016 | 21236096 | 8652672 |
| Deployed parameter bytes | 190944424 | 190872064 | 84944384 | 34610688 |
| Deployed buffer bytes | 36 | 36 | 36 | 36 |
| Total deployed tensor bytes | 190944460 | 190872100 | 84944420 | 34610724 |
| Previously fitted target-data params | 17920 | 0 | 0 | 0 |
| Inference gradient-enabled params | 0 | 0 | 0 | 0 |
| Original input channels C | 17 | 17 | 17 | 17 |
| TSFM rows per origin | 5 | 17 | 17 | 17 |
| TRAIN PCA | Fixed C-to-K E/D | None | None | None |
| Residual G | Previously selected, frozen | None | None | None |
| Target-data fitting history | TRAIN PCA and selected G | None | None | None |
| New fits in this campaign | 0 | 0 | 0 | 0 |
| Context / horizon observations | 512 / 48 | 512 / 48 | 512 / 48 | 512 / 48 |
| Precision / TF32 | FP32 / off | FP32 / off | FP32 / off | FP32 / off |
| Same preprocessing | Yes: existing TRAIN statistics and input imputation | Yes: existing TRAIN statistics and input imputation | Yes: existing TRAIN statistics and input imputation | Yes: existing TRAIN statistics and input imputation |
| Same origins / target mask | Yes: sealed origin order and raw-finite target mask | Yes: sealed origin order and raw-finite target mask | Yes: sealed origin order and raw-finite target mask | Yes: sealed origin order and raw-finite target mask |
| Point forecast | Native q=0.5 -> fixed decode + G | Native q=0.5, first H48 | Native q=0.5, first H48 | Native q=0.5, first H48 |
| Loss aggregation | Mean losses of 2 selected seeds | One deterministic model | One deterministic model | One deterministic model |
| TEST exposure | Previously exposed; post-exposure practical control | Previously exposed; post-exposure practical control | Previously exposed; post-exposure practical control | Previously exposed; post-exposure practical control |
| Pretraining overlap | Unverified | Unverified | Unverified | Unverified |

## Peacock Education — C13/K4

| Attribute | Small + LEVEL | Small F0 | Mini F0 | Tiny F0 |
| --- | --- | --- | --- | --- |
| Family | Chronos-Bolt | Chronos-Bolt | Chronos-Bolt | Chronos-Bolt |
| Architecture | ChronosBoltModelForForecasting | ChronosBoltModelForForecasting | ChronosBoltModelForForecasting | ChronosBoltModelForForecasting |
| Pinned model revision | 772f3d25d38aec6d914c8949dab4462e2d46f5d8 | 772f3d25d38aec6d914c8949dab4462e2d46f5d8 | 251268337516a88e253628c43e1d26ec577b376b | a0e552de83495b5c28c14c71c374f3e33280b340 |
| Actual backbone params | 47718016 | 47718016 | 21236096 | 8652672 |
| Actual backbone parameter bytes | 190872064 | 190872064 | 84944384 | 34610688 |
| Total deployed params | 47736040 | 47718016 | 21236096 | 8652672 |
| Deployed parameter bytes | 190944160 | 190872064 | 84944384 | 34610688 |
| Deployed buffer bytes | 244 | 244 | 36 | 36 |
| Total deployed tensor bytes | 190944404 | 190872308 | 84944420 | 34610724 |
| Previously fitted target-data params | 17920 | 0 | 0 | 0 |
| Inference gradient-enabled params | 0 | 0 | 0 | 0 |
| Original input channels C | 13 | 13 | 13 | 13 |
| TSFM rows per origin | 4 | 13 | 13 | 13 |
| TRAIN PCA | Fixed C-to-K E/D | None | None | None |
| Residual G | Previously selected, frozen | None | None | None |
| Target-data fitting history | TRAIN PCA and selected G | None | None | None |
| New fits in this campaign | 0 | 0 | 0 | 0 |
| Context / horizon observations | 512 / 48 | 512 / 48 | 512 / 48 | 512 / 48 |
| Precision / TF32 | FP32 / off | FP32 / off | FP32 / off | FP32 / off |
| Same preprocessing | Yes: existing TRAIN statistics and input imputation | Yes: existing TRAIN statistics and input imputation | Yes: existing TRAIN statistics and input imputation | Yes: existing TRAIN statistics and input imputation |
| Same origins / target mask | Yes: sealed origin order and raw-finite target mask | Yes: sealed origin order and raw-finite target mask | Yes: sealed origin order and raw-finite target mask | Yes: sealed origin order and raw-finite target mask |
| Point forecast | Native q=0.5 -> fixed decode + G | Native q=0.5, first H48 | Native q=0.5, first H48 | Native q=0.5, first H48 |
| Loss aggregation | Mean losses of 2 selected seeds | One deterministic model | One deterministic model | One deterministic model |
| TEST exposure | Previously exposed; post-exposure practical control | Previously exposed; post-exposure practical control | Previously exposed; post-exposure practical control | Previously exposed; post-exposure practical control |
| Pretraining overlap | Unverified | Unverified | Unverified | Unverified |

## Jena — C21/K6

| Attribute | Small + LEVEL | Small F0 | Mini F0 | Tiny F0 |
| --- | --- | --- | --- | --- |
| Family | Chronos-Bolt | Chronos-Bolt | Chronos-Bolt | Chronos-Bolt |
| Architecture | ChronosBoltModelForForecasting | ChronosBoltModelForForecasting | ChronosBoltModelForForecasting | ChronosBoltModelForForecasting |
| Pinned model revision | 772f3d25d38aec6d914c8949dab4462e2d46f5d8 | 772f3d25d38aec6d914c8949dab4462e2d46f5d8 | 251268337516a88e253628c43e1d26ec577b376b | a0e552de83495b5c28c14c71c374f3e33280b340 |
| Actual backbone params | 47718016 | 47718016 | 21236096 | 8652672 |
| Actual backbone parameter bytes | 190872064 | 190872064 | 84944384 | 34610688 |
| Total deployed params | 47736188 | 47718016 | 21236096 | 8652672 |
| Deployed parameter bytes | 190944752 | 190872064 | 84944384 | 34610688 |
| Deployed buffer bytes | 36 | 36 | 36 | 36 |
| Total deployed tensor bytes | 190944788 | 190872100 | 84944420 | 34610724 |
| Previously fitted target-data params | 17920 | 0 | 0 | 0 |
| Inference gradient-enabled params | 0 | 0 | 0 | 0 |
| Original input channels C | 21 | 21 | 21 | 21 |
| TSFM rows per origin | 6 | 21 | 21 | 21 |
| TRAIN PCA | Fixed C-to-K E/D | None | None | None |
| Residual G | Previously selected, frozen | None | None | None |
| Target-data fitting history | TRAIN PCA and selected G | None | None | None |
| New fits in this campaign | 0 | 0 | 0 | 0 |
| Context / horizon observations | 512 / 48 | 512 / 48 | 512 / 48 | 512 / 48 |
| Precision / TF32 | FP32 / off | FP32 / off | FP32 / off | FP32 / off |
| Same preprocessing | Yes: existing TRAIN statistics and input imputation | Yes: existing TRAIN statistics and input imputation | Yes: existing TRAIN statistics and input imputation | Yes: existing TRAIN statistics and input imputation |
| Same origins / target mask | Yes: sealed origin order and raw-finite target mask | Yes: sealed origin order and raw-finite target mask | Yes: sealed origin order and raw-finite target mask | Yes: sealed origin order and raw-finite target mask |
| Point forecast | Native q=0.5 -> fixed decode + G | Native q=0.5, first H48 | Native q=0.5, first H48 | Native q=0.5, first H48 |
| Loss aggregation | Mean losses of 2 selected seeds | One deterministic model | One deterministic model | One deterministic model |
| TEST exposure | Previously exposed; post-exposure practical control | Previously exposed; post-exposure practical control | Previously exposed; post-exposure practical control | Previously exposed; post-exposure practical control |
| Pretraining overlap | Unverified | Unverified | Unverified | Unverified |

[확인] Mini/Tiny는 native multivariate 모델이 아니라 같은 Chronos-Bolt 계열의 더 작은 backbone입니다. 모든 F0는 `[B,512,C] -> [B*C,512]`로 원채널을 서로 독립적인 단변량 task로 처리합니다. LEVEL만 fixed TRAIN PCA 뒤 K개 task를 처리하고 원채널로 decode한 뒤 G를 더합니다. 부모 backbone과 배포 전체 크기를 혼동하지 않으며, 고정 E/D와 G도 배포 tensor bytes에 포함됩니다.

이 표는 `model_manifest.json`, `input_contract.json`, `preflight_checks.json`, `cost_comparison.csv` 및 `cost_cuda_summary.json`에 연결됩니다. 최종 검산·게시 상태는 `final_checks.json`과 `STATUS.md`에서 별도로 확인합니다.

## 정확도: TEST-A / TEST-B / combined

### Robin

```text
Arm            Period    MSE       MAE       Signed bias  Observed targets  Origins
Small + LEVEL  test_a    0.273677  0.331865  0.029907     32640             40
Small + LEVEL  test_b    0.276712  0.317319  -0.015627    32638             40
Small + LEVEL  combined  0.275196  0.324593  0.007139     65278             80
Small F0       test_a    0.231539  0.283065  0.017943     32640             40
Small F0       test_b    0.226649  0.241079  -0.013022    32638             40
Small F0       combined  0.229095  0.262074  0.002460     65278             80
Mini F0        test_a    0.227795  0.284107  0.003817     32640             40
Mini F0        test_b    0.227475  0.245304  -0.023928    32638             40
Mini F0        combined  0.227636  0.264707  -0.010056    65278             80
Tiny F0        test_a    0.241522  0.289465  0.007015     32640             40
Tiny F0        test_b    0.245783  0.249927  -0.023340    32638             40
Tiny F0        combined  0.243654  0.269697  -0.008163    65278             80
```

### Peacock Education

```text
Arm            Period    MSE       MAE       Signed bias  Observed targets  Origins
Small + LEVEL  test_a    0.234300  0.339193  -0.025786    24958             40
Small + LEVEL  test_b    0.172223  0.290810  -0.001909    24946             40
Small + LEVEL  combined  0.203266  0.315004  -0.013859    49904             80
Small F0       test_a    0.235011  0.324832  -0.033234    24958             40
Small F0       test_b    0.168020  0.275875  -0.011643    24946             40
Small F0       combined  0.201521  0.300358  -0.022447    49904             80
Mini F0        test_a    0.238520  0.328489  -0.038606    24958             40
Mini F0        test_b    0.165583  0.274918  -0.017467    24946             40
Mini F0        combined  0.202057  0.301707  -0.028046    49904             80
Tiny F0        test_a    0.235372  0.328389  -0.039341    24958             40
Tiny F0        test_b    0.170104  0.281162  -0.023480    24946             40
Tiny F0        combined  0.202745  0.304780  -0.031418    49904             80
```

### Jena

```text
Arm            Period    MSE       MAE       Signed bias  Observed targets  Origins
Small + LEVEL  test_a    0.422475  0.381041  -0.016524    367920            365
Small + LEVEL  test_b    0.226243  0.259859  -0.007794    367920            365
Small + LEVEL  combined  0.324359  0.320450  -0.012159    735840            730
Small F0       test_a    0.370756  0.313130  -0.030104    367920            365
Small F0       test_b    0.213844  0.216386  -0.023977    367920            365
Small F0       combined  0.292300  0.264758  -0.027040    735840            730
Mini F0        test_a    0.386689  0.322625  -0.027381    367920            365
Mini F0        test_b    0.217438  0.220224  -0.019576    367920            365
Mini F0        combined  0.302064  0.271424  -0.023478    735840            730
Tiny F0        test_a    0.422848  0.339331  -0.029489    367920            365
Tiny F0        test_b    0.219970  0.222681  -0.019658    367920            365
Tiny F0        combined  0.321409  0.281006  -0.024574    735840            730
```

[확인] Primary는 관측 target의 TRAIN-standardized channel-macro MSE, secondary는 동일 MAE입니다. combined는 A/B별 채널 SSE·절대오차합·count를 먼저 합치고 채널별 비율 뒤 채널 평균을 냈습니다. A/B MSE 단순 평균이나 자료 간 MSE 평균 총점을 만들지 않았습니다. LEVEL은 기존 두 선택 seed의 손실 평균이며 prediction ensemble이 아닙니다. Small/Mini/Tiny F0는 각각 deterministic model 한 번으로 집계했습니다.

## 대응 상대 차이와 조건부 구간

상대 차이는 `100*(LEVEL/alternative - 1)`이며 음수는 LEVEL의 낮은 오차입니다. 기존 paired time-block bootstrap을 재사용한 조건부 95% 구간으로 모델 선택 불확실성은 포함하지 않습니다. A/B 경계를 넘지 않고 method·origin·channel pairing을 유지했습니다. 구간의 0 포함은 차이 불확실이며 동등성·비열등성의 증거가 아닙니다.

### Robin

```text
Alternative  Period    Metric  LEVEL relative %  Conditional 95% %  LEVEL minus alt  Conditional 95%        Difference
Mini F0      test_a    MSE     20.14             [7.33, 41.97]      0.045882         [0.017262, 0.090826]   Excludes zero
Mini F0      test_a    MAE     16.81             [10.74, 26.20]     0.047758         [0.033644, 0.069727]   Excludes zero
Mini F0      test_b    MSE     21.65             [7.14, 52.86]      0.049237         [0.023829, 0.075121]   Excludes zero
Mini F0      test_b    MAE     29.36             [13.00, 53.60]     0.072014         [0.042033, 0.102444]   Excludes zero
Mini F0      combined  MSE     20.89             [10.38, 36.79]     0.047560         [0.028117, 0.073633]   Excludes zero
Mini F0      combined  MAE     22.62             [13.82, 32.99]     0.059886         [0.042245, 0.078462]   Excludes zero
Tiny F0      test_a    MSE     13.31             [-0.23, 42.55]     0.032155         [-0.000544, 0.094451]  Includes zero
Tiny F0      test_a    MAE     14.65             [7.97, 25.45]      0.042400         [0.025031, 0.069816]   Excludes zero
Tiny F0      test_b    MSE     12.58             [-3.90, 50.86]     0.030929         [-0.016518, 0.068508]  Includes zero
Tiny F0      test_b    MAE     26.96             [9.14, 53.81]      0.067392         [0.031197, 0.102223]   Excludes zero
Tiny F0      combined  MSE     12.95             [0.75, 33.00]      0.031542         [0.002387, 0.066559]   Excludes zero
Tiny F0      combined  MAE     20.35             [11.01, 31.51]     0.054896         [0.034205, 0.076392]   Excludes zero
```

### Peacock Education

```text
Alternative  Period    Metric  LEVEL relative %  Conditional 95% %  LEVEL minus alt  Conditional 95%        Difference
Mini F0      test_a    MSE     -1.77             [-5.28, 5.64]      -0.004221        [-0.013625, 0.012945]  Includes zero
Mini F0      test_a    MAE     3.26              [0.57, 6.68]       0.010704         [0.002062, 0.021693]   Excludes zero
Mini F0      test_b    MSE     4.01              [-0.34, 9.92]      0.006640         [-0.000514, 0.015517]  Includes zero
Mini F0      test_b    MAE     5.78              [2.74, 8.08]       0.015892         [0.007419, 0.021814]   Excludes zero
Mini F0      combined  MSE     0.60              [-2.14, 5.82]      0.001209         [-0.004354, 0.011256]  Includes zero
Mini F0      combined  MAE     4.41              [2.28, 6.54]       0.013298         [0.007138, 0.019623]   Excludes zero
Tiny F0      test_a    MSE     -0.46             [-4.58, 6.72]      -0.001072        [-0.011905, 0.015190]  Includes zero
Tiny F0      test_a    MAE     3.29              [0.59, 6.58]       0.010805         [0.002117, 0.021281]   Excludes zero
Tiny F0      test_b    MSE     1.25              [-2.55, 7.92]      0.002119         [-0.004085, 0.012302]  Includes zero
Tiny F0      test_b    MAE     3.43              [0.24, 6.39]       0.009648         [0.000688, 0.017430]   Excludes zero
Tiny F0      combined  MSE     0.26              [-2.50, 5.57]      0.000522         [-0.005283, 0.010737]  Includes zero
Tiny F0      combined  MAE     3.35              [1.33, 5.59]       0.010225         [0.004121, 0.016860]   Excludes zero
```

### Jena

```text
Alternative  Period    Metric  LEVEL relative %  Conditional 95% %  LEVEL minus alt  Conditional 95%        Difference
Mini F0      test_a    MSE     9.25              [5.58, 14.81]      0.035786         [0.022586, 0.049633]   Excludes zero
Mini F0      test_a    MAE     18.11             [15.99, 20.47]     0.058416         [0.051886, 0.067436]   Excludes zero
Mini F0      test_b    MSE     4.05              [0.48, 8.20]       0.008806         [0.001157, 0.017290]   Excludes zero
Mini F0      test_b    MAE     18.00             [15.39, 21.82]     0.039636         [0.033638, 0.046303]   Excludes zero
Mini F0      combined  MSE     7.38              [4.71, 10.88]      0.022296         [0.014719, 0.030352]   Excludes zero
Mini F0      combined  MAE     18.06             [16.44, 20.28]     0.049026         [0.044622, 0.054743]   Excludes zero
Tiny F0      test_a    MSE     -0.09             [-3.78, 4.70]      -0.000373        [-0.019663, 0.015003]  Includes zero
Tiny F0      test_a    MAE     12.29             [10.26, 13.92]     0.041709         [0.035167, 0.047147]   Excludes zero
Tiny F0      test_b    MSE     2.85              [-0.35, 6.84]      0.006274         [-0.000807, 0.014543]  Includes zero
Tiny F0      test_b    MAE     16.70             [14.36, 20.52]     0.037178         [0.031847, 0.043775]   Excludes zero
Tiny F0      combined  MSE     0.92              [-1.88, 4.35]      0.002950         [-0.006957, 0.012197]  Includes zero
Tiny F0      combined  MAE     14.04             [12.54, 15.81]     0.039444         [0.035380, 0.043850]   Excludes zero
```

보조 Small LEVEL/F0, Small F0/Mini, Small F0/Tiny, Mini/Tiny까지 총6개 대응 비교의 A/B/combined·MSE/MAE는 `paired_comparisons.csv`에 보존합니다. block length Robin/Peacock7, Jena42 origins; resamples2000; seed9262026; dataset salt0/0/1과 기존 기간 층화를 유지했습니다. 보고 생성기는 bootstrap을 추가 수행하지 않습니다.

## 같은 회차 비용과 배포 상태

[확인] RTX 4070, FP32, TF32 off, 동일 환경·CPU4 threads에서 첫24 chronological VAL origins를 사용했습니다. 각 shape 예열10, 전체24-origin passes10, 순서회전3 blocks입니다. 측정은 standardized CPU input에서 reshape/transfer, 전체 online inference, H48C contiguous CPU output 복귀까지입니다. LEVEL의 PCA encode·Bolt-small·decode·G를 모두 포함하고, load/disk/hash/score/log는 latency에서 제외했습니다. compile·quantization·임의 B2 대안은 없습니다.

### Robin — B1

```text
Arm            ms/origin  origins/s  Peak alloc MiB  Peak reserved MiB  Block ms range    Block origins/s range  Resident alloc range MiB  Resident reserved range MiB
Small + LEVEL  11.244     88.961     196.628         220.000            [11.048, 11.958]  [83.655, 90.519]       [182.100, 182.100]        [194.000, 194.000]
Small F0       11.517     86.831     209.871         216.000            [10.872, 11.581]  [86.368, 91.987]       [182.031, 182.031]        [194.000, 194.000]
Mini F0        8.714      114.764    104.169         138.000            [8.639, 8.862]    [112.920, 115.755]     [81.010, 81.010]          [132.000, 132.000]
Tiny F0        8.777      113.936    48.962          76.000             [8.325, 9.410]    [106.285, 120.120]     [33.008, 33.008]          [66.000, 66.000]
```

### Robin — B4

```text
Arm            ms/origin  origins/s  Peak alloc MiB  Peak reserved MiB  Block ms range  Block origins/s range  Resident alloc range MiB  Resident reserved range MiB
Small + LEVEL  2.890      346.506    214.489         216.000            [2.767, 2.994]  [334.020, 361.433]     [182.100, 182.100]        [194.000, 194.000]
Small F0       4.056      246.548    265.489         292.000            [4.037, 4.346]  [230.176, 247.684]     [182.031, 182.031]        [194.000, 194.000]
Mini F0        2.635      379.547    143.826         182.000            [2.445, 2.666]  [375.123, 408.974]     [81.010, 81.010]          [132.000, 132.000]
Tiny F0        2.136      468.112    69.438          88.000             [2.052, 2.266]  [441.232, 487.259]     [33.008, 33.008]          [66.000, 66.000]
```

### Peacock Education — B1

```text
Arm            ms/origin  origins/s  Peak alloc MiB  Peak reserved MiB  Block ms range    Block origins/s range  Resident alloc range MiB  Resident reserved range MiB
Small + LEVEL  11.373     87.950     195.521         220.000            [10.970, 12.225]  [81.802, 91.156]       [182.101, 182.101]        [216.000, 216.000]
Small F0       11.538     86.679     205.025         230.000            [11.308, 11.789]  [84.841, 88.430]       [182.031, 182.031]        [216.000, 216.000]
Mini F0        8.647      115.666    100.106         144.000            [8.545, 9.016]    [110.917, 117.041]     [81.010, 81.010]          [132.000, 132.000]
Tiny F0        8.454      118.298    47.356          74.000             [8.318, 8.553]    [117.087, 120.222]     [33.008, 33.008]          [66.000, 66.000]
```

### Peacock Education — B4

```text
Arm            ms/origin  origins/s  Peak alloc MiB  Peak reserved MiB  Block ms range  Block origins/s range  Resident alloc range MiB  Resident reserved range MiB
Small + LEVEL  2.876      347.756    210.021         218.000            [2.790, 3.023]  [330.813, 358.436]     [182.101, 182.101]        [216.000, 216.000]
Small F0       3.571      280.089    248.557         266.000            [3.540, 3.718]  [268.991, 282.482]     [182.031, 182.031]        [216.000, 216.000]
Mini F0        2.398      417.006    130.110         152.000            [2.327, 2.464]  [405.812, 430.100]     [81.010, 81.010]          [132.000, 132.000]
Tiny F0        2.542      393.602    63.397          70.000             [2.415, 2.571]  [388.995, 414.166]     [33.008, 33.008]          [66.000, 66.000]
```

### Jena — B1

```text
Arm            ms/origin  origins/s  Peak alloc MiB  Peak reserved MiB  Block ms range    Block origins/s range  Resident alloc range MiB  Resident reserved range MiB
Small + LEVEL  11.096     90.143     197.710         222.000            [10.873, 11.862]  [84.325, 91.976]       [182.100, 182.100]        [194.000, 194.000]
Small F0       11.207     89.230     214.457         236.000            [11.015, 12.014]  [83.256, 90.788]       [182.031, 182.031]        [194.000, 194.000]
Mini F0        8.503      117.619    107.132         136.000            [8.372, 8.558]    [116.851, 119.450]     [81.010, 81.010]          [132.000, 132.000]
Tiny F0        8.496      117.706    50.568          78.000             [8.474, 8.631]    [115.869, 118.030]     [33.008, 33.008]          [66.000, 66.000]
```

### Jena — B4

```text
Arm            ms/origin  origins/s  Peak alloc MiB  Peak reserved MiB  Block ms range  Block origins/s range  Resident alloc range MiB  Resident reserved range MiB
Small + LEVEL  3.077      325.007    217.904         238.000            [2.882, 3.186]  [313.914, 347.022]     [182.100, 182.100]        [194.000, 194.000]
Small F0       4.698      212.861    281.436         304.000            [4.446, 4.730]  [211.399, 224.924]     [182.031, 182.031]        [194.000, 194.000]
Mini F0        2.731      366.220    155.406         188.000            [2.725, 2.840]  [352.159, 366.916]     [81.010, 81.010]          [132.000, 132.000]
Tiny F0        2.373      421.344    75.862          100.000            [2.115, 2.426]  [412.279, 472.894]     [33.008, 33.008]          [66.000, 66.000]
```

### Mini/Tiny의 LEVEL 대비 비용 차이

아래 상대 차이는 `100*(alternative/LEVEL - 1)`입니다. 처리량의 양수와 latency·bytes의 음수가 alternative에 유리합니다.

```text
Dataset            Alternative  B1 latency %  B4 throughput %  B4 allocated %  B4 reserved %  Deployed tensor bytes %
Robin              Mini F0      -22.49        9.54             -32.95          -15.74         -55.51
Robin              Tiny F0      -21.94        35.09            -67.63          -59.26         -81.87
Peacock Education  Mini F0      -23.96        19.91            -38.05          -30.28         -55.51
Peacock Education  Tiny F0      -25.67        13.18            -69.81          -67.89         -81.87
Jena               Mini F0      -23.37        12.68            -28.68          -21.01         -55.51
Jena               Tiny F0      -23.43        29.64            -65.19          -57.98         -81.87
```

집계는 instance별3 block median의 median을 구한 뒤 LEVEL 선택 seed2개의 값을 평균했습니다. 범위는 모든 선택 seed/block 원값의 min–max이며 신뢰구간이 아닙니다. allocated, reserved, 모델 로드·cleanup 직후 resident, deployed tensor bytes는 다른 양입니다. CPU process memory도 원블록에 보존합니다.

```text
Dataset            Arm            Deploy params  Parameter bytes  Buffer bytes  Total tensor bytes  Prior fit params
Robin              Small + LEVEL  47736106       190944424        36            190944460           17920
Robin              Small F0       47718016       190872064        36            190872100           0
Robin              Mini F0        21236096       84944384         36            84944420            0
Robin              Tiny F0        8652672        34610688         36            34610724            0
Peacock Education  Small + LEVEL  47736040       190944160        244           190944404           17920
Peacock Education  Small F0       47718016       190872064        244           190872308           0
Peacock Education  Mini F0        21236096       84944384         36            84944420            0
Peacock Education  Tiny F0        8652672        34610688         36            34610724            0
Jena               Small + LEVEL  47736188       190944752        36            190944788           17920
Jena               Small F0       47718016       190872064        36            190872100           0
Jena               Mini F0        21236096       84944384         36            84944420            0
Jena               Tiny F0        8652672        34610688         36            34610724            0
```

Mini/Tiny의 작은 실제 배포 가중치는 배포 선택에서 의미가 있습니다. 추가 fit0은 사전학습 비용·가중치 bytes·resident memory0을 뜻하지 않습니다. LEVEL의 이전 G 적합과 이번 inference gradient-enabled params0은 별도로 해석합니다.

### 모든 원블록과 drift

```text
Dataset            Arm            ID                                                        Block  B  Sentinel  ms/origin  origins/s  Peak alloc MiB  Peak reserved MiB  Status
Robin              Small F0       v6_robin_f0                                               0      1  pre       11.491     87.058     209.871         216.000            complete
Robin              Small + LEVEL  v6_robin_level_res_92601                                  0      1  primary   11.835     84.543     196.628         220.000            complete
Robin              Small + LEVEL  v6_robin_level_res_92601                                  0      4  primary   2.790      358.456    214.489         216.000            complete
Robin              Small + LEVEL  v6_robin_level_res_92602                                  0      1  primary   11.958     83.655     196.628         220.000            complete
Robin              Small + LEVEL  v6_robin_level_res_92602                                  0      4  primary   2.994      334.020    214.489         216.000            complete
Robin              Small F0       v6_robin_f0                                               0      1  primary   11.581     86.368     209.871         216.000            complete
Robin              Small F0       v6_robin_f0                                               0      4  primary   4.037      247.684    265.489         292.000            complete
Robin              Mini F0        robin_BOLT_MINI_F0                                        0      1  primary   8.639      115.755    104.169         138.000            complete
Robin              Mini F0        robin_BOLT_MINI_F0                                        0      4  primary   2.445      408.974    143.826         182.000            complete
Robin              Tiny F0        robin_BOLT_TINY_F0                                        0      1  primary   8.325      120.120    48.962          76.000             complete
Robin              Tiny F0        robin_BOLT_TINY_F0                                        0      4  primary   2.266      441.232    69.438          88.000             complete
Robin              Small F0       v6_robin_f0                                               0      1  post      11.429     87.494     209.871         216.000            complete
Peacock Education  Small F0       v12_confirmation_peacock_education_f0                     0      1  pre       11.217     89.149     205.025         230.000            complete
Peacock Education  Small + LEVEL  peacock_education_level_lr1e-3_s92601__retry_heartbeat01  0      1  primary   12.085     82.756     195.521         220.000            complete
Peacock Education  Small + LEVEL  peacock_education_level_lr1e-3_s92601__retry_heartbeat01  0      4  primary   2.922      342.229    210.021         218.000            complete
Peacock Education  Small + LEVEL  peacock_education_level_lr1e-3_s92602                     0      1  primary   11.211     89.198     195.521         220.000            complete
Peacock Education  Small + LEVEL  peacock_education_level_lr1e-3_s92602                     0      4  primary   2.841      351.932    210.021         218.000            complete
Peacock Education  Small F0       v12_confirmation_peacock_education_f0                     0      1  primary   11.538     86.679     205.025         230.000            complete
Peacock Education  Small F0       v12_confirmation_peacock_education_f0                     0      4  primary   3.571      280.089    248.557         266.000            complete
Peacock Education  Mini F0        peacock_education_BOLT_MINI_F0                            0      1  primary   8.647      115.666    100.106         144.000            complete
Peacock Education  Mini F0        peacock_education_BOLT_MINI_F0                            0      4  primary   2.464      405.812    130.110         152.000            complete
Peacock Education  Tiny F0        peacock_education_BOLT_TINY_F0                            0      1  primary   8.318      120.222    47.356          74.000             complete
Peacock Education  Tiny F0        peacock_education_BOLT_TINY_F0                            0      4  primary   2.542      393.602    63.397          70.000             complete
Peacock Education  Small F0       v12_confirmation_peacock_education_f0                     0      1  post      11.657     85.802     205.025         230.000            complete
Jena               Small F0       v7_jena_f0                                                0      1  pre       11.190     89.365     214.457         236.000            complete
Jena               Small + LEVEL  v7_jena_level_res_92601                                   0      1  primary   11.004     90.876     197.710         222.000            complete
Jena               Small + LEVEL  v7_jena_level_res_92601                                   0      4  primary   2.882      347.022    217.904         238.000            complete
Jena               Small + LEVEL  v7_jena_level_res_92602                                   0      1  primary   10.942     91.392     197.710         222.000            complete
Jena               Small + LEVEL  v7_jena_level_res_92602                                   0      4  primary   3.081      324.591    217.904         238.000            complete
Jena               Small F0       v7_jena_f0                                                0      1  primary   11.015     90.788     214.457         236.000            complete
Jena               Small F0       v7_jena_f0                                                0      4  primary   4.698      212.861    281.436         304.000            complete
Jena               Mini F0        jena_BOLT_MINI_F0                                         0      1  primary   8.558      116.851    107.132         136.000            complete
Jena               Mini F0        jena_BOLT_MINI_F0                                         0      4  primary   2.731      366.220    155.406         188.000            complete
Jena               Tiny F0        jena_BOLT_TINY_F0                                         0      1  primary   8.474      118.030    50.568          78.000             complete
Jena               Tiny F0        jena_BOLT_TINY_F0                                         0      4  primary   2.426      412.279    75.862          100.000            complete
Jena               Small F0       v7_jena_f0                                                0      1  post      11.030     90.669     214.457         236.000            complete
Peacock Education  Small F0       v12_confirmation_peacock_education_f0                     1      1  pre       11.029     90.669     205.025         230.000            complete
Peacock Education  Small + LEVEL  peacock_education_level_lr1e-3_s92602                     1      1  primary   10.970     91.156     195.521         220.000            complete
Peacock Education  Small + LEVEL  peacock_education_level_lr1e-3_s92602                     1      4  primary   2.831      353.282    210.021         218.000            complete
Peacock Education  Small F0       v12_confirmation_peacock_education_f0                     1      1  primary   11.308     88.430     205.025         230.000            complete
Peacock Education  Small F0       v12_confirmation_peacock_education_f0                     1      4  primary   3.540      282.482    248.557         266.000            complete
Peacock Education  Mini F0        peacock_education_BOLT_MINI_F0                            1      1  primary   8.545      117.041    100.106         144.000            complete
Peacock Education  Mini F0        peacock_education_BOLT_MINI_F0                            1      4  primary   2.398      417.006    130.110         152.000            complete
Peacock Education  Tiny F0        peacock_education_BOLT_TINY_F0                            1      1  primary   8.553      117.087    47.356          74.000             complete
Peacock Education  Tiny F0        peacock_education_BOLT_TINY_F0                            1      4  primary   2.571      388.995    63.397          70.000             complete
Peacock Education  Small + LEVEL  peacock_education_level_lr1e-3_s92601__retry_heartbeat01  1      1  primary   11.078     90.268     195.521         220.000            complete
Peacock Education  Small + LEVEL  peacock_education_level_lr1e-3_s92601__retry_heartbeat01  1      4  primary   3.023      330.813    210.021         218.000            complete
Peacock Education  Small F0       v12_confirmation_peacock_education_f0                     1      1  post      11.168     89.559     205.025         230.000            complete
Jena               Small F0       v7_jena_f0                                                1      1  pre       11.926     83.848     214.457         236.000            complete
Jena               Small + LEVEL  v7_jena_level_res_92602                                   1      1  primary   11.443     87.388     197.710         222.000            complete
Jena               Small + LEVEL  v7_jena_level_res_92602                                   1      4  primary   2.913      343.293    217.904         238.000            complete
Jena               Small F0       v7_jena_f0                                                1      1  primary   12.014     83.256     214.457         236.000            complete
Jena               Small F0       v7_jena_f0                                                1      4  primary   4.730      211.399    281.436         304.000            complete
Jena               Mini F0        jena_BOLT_MINI_F0                                         1      1  primary   8.372      119.450    107.132         136.000            complete
Jena               Mini F0        jena_BOLT_MINI_F0                                         1      4  primary   2.725      366.916    155.406         188.000            complete
Jena               Tiny F0        jena_BOLT_TINY_F0                                         1      1  primary   8.631      115.869    50.568          78.000             complete
Jena               Tiny F0        jena_BOLT_TINY_F0                                         1      4  primary   2.373      421.344    75.862          100.000            complete
Jena               Small + LEVEL  v7_jena_level_res_92601                                   1      1  primary   11.249     88.894     197.710         222.000            complete
Jena               Small + LEVEL  v7_jena_level_res_92601                                   1      4  primary   3.073      325.423    217.904         238.000            complete
Jena               Small F0       v7_jena_f0                                                1      1  post      11.498     86.988     214.457         236.000            complete
Robin              Small F0       v6_robin_f0                                               1      1  pre       11.498     86.975     209.871         216.000            complete
Robin              Small + LEVEL  v6_robin_level_res_92602                                  1      1  primary   11.090     90.179     196.628         220.000            complete
Robin              Small + LEVEL  v6_robin_level_res_92602                                  1      4  primary   2.978      335.852    214.489         216.000            complete
Robin              Small F0       v6_robin_f0                                               1      1  primary   10.872     91.987     209.871         216.000            complete
Robin              Small F0       v6_robin_f0                                               1      4  primary   4.056      246.548    265.489         292.000            complete
Robin              Mini F0        robin_BOLT_MINI_F0                                        1      1  primary   8.714      114.764    104.169         138.000            complete
Robin              Mini F0        robin_BOLT_MINI_F0                                        1      4  primary   2.635      379.547    143.826         182.000            complete
Robin              Tiny F0        robin_BOLT_TINY_F0                                        1      1  primary   8.777      113.936    48.962          76.000             complete
Robin              Tiny F0        robin_BOLT_TINY_F0                                        1      4  primary   2.136      468.112    69.438          88.000             complete
Robin              Small + LEVEL  v6_robin_level_res_92601                                  1      1  primary   11.245     88.945     196.628         220.000            complete
Robin              Small + LEVEL  v6_robin_level_res_92601                                  1      4  primary   2.767      361.433    214.489         216.000            complete
Robin              Small F0       v6_robin_f0                                               1      1  post      11.279     88.710     209.871         216.000            complete
Jena               Small F0       v7_jena_f0                                                2      1  pre       10.939     91.423     214.457         236.000            complete
Jena               Tiny F0        jena_BOLT_TINY_F0                                         2      1  primary   8.496      117.706    50.568          78.000             complete
Jena               Tiny F0        jena_BOLT_TINY_F0                                         2      4  primary   2.115      472.894    75.862          100.000            complete
Jena               Mini F0        jena_BOLT_MINI_F0                                         2      1  primary   8.503      117.619    107.132         136.000            complete
Jena               Mini F0        jena_BOLT_MINI_F0                                         2      4  primary   2.840      352.159    155.406         188.000            complete
Jena               Small F0       v7_jena_f0                                                2      1  primary   11.207     89.230     214.457         236.000            complete
Jena               Small F0       v7_jena_f0                                                2      4  primary   4.446      224.924    281.436         304.000            complete
Jena               Small + LEVEL  v7_jena_level_res_92602                                   2      1  primary   10.873     91.976     197.710         222.000            complete
Jena               Small + LEVEL  v7_jena_level_res_92602                                   2      4  primary   3.145      317.965    217.904         238.000            complete
Jena               Small + LEVEL  v7_jena_level_res_92601                                   2      1  primary   11.862     84.325     197.710         222.000            complete
Jena               Small + LEVEL  v7_jena_level_res_92601                                   2      4  primary   3.186      313.914    217.904         238.000            complete
Jena               Small F0       v7_jena_f0                                                2      1  post      11.804     84.719     214.457         236.000            complete
Peacock Education  Small F0       v12_confirmation_peacock_education_f0                     2      1  pre       11.057     90.438     205.025         230.000            complete
Peacock Education  Tiny F0        peacock_education_BOLT_TINY_F0                            2      1  primary   8.454      118.298    47.356          74.000             complete
Peacock Education  Tiny F0        peacock_education_BOLT_TINY_F0                            2      4  primary   2.415      414.166    63.397          70.000             complete
Peacock Education  Mini F0        peacock_education_BOLT_MINI_F0                            2      1  primary   9.016      110.917    100.106         144.000            complete
Peacock Education  Mini F0        peacock_education_BOLT_MINI_F0                            2      4  primary   2.327      430.100    130.110         152.000            complete
Peacock Education  Small F0       v12_confirmation_peacock_education_f0                     2      1  primary   11.789     84.841     205.025         230.000            complete
Peacock Education  Small F0       v12_confirmation_peacock_education_f0                     2      4  primary   3.718      268.991    248.557         266.000            complete
Peacock Education  Small + LEVEL  peacock_education_level_lr1e-3_s92602                     2      1  primary   12.225     81.802     195.521         220.000            complete
Peacock Education  Small + LEVEL  peacock_education_level_lr1e-3_s92602                     2      4  primary   2.790      358.436    210.021         218.000            complete
Peacock Education  Small + LEVEL  peacock_education_level_lr1e-3_s92601__retry_heartbeat01  2      1  primary   11.534     86.701     195.521         220.000            complete
Peacock Education  Small + LEVEL  peacock_education_level_lr1e-3_s92601__retry_heartbeat01  2      4  primary   2.799      357.310    210.021         218.000            complete
Peacock Education  Small F0       v12_confirmation_peacock_education_f0                     2      1  post      11.044     90.554     205.025         230.000            complete
Robin              Small F0       v6_robin_f0                                               2      1  pre       11.788     84.837     209.871         216.000            complete
Robin              Tiny F0        robin_BOLT_TINY_F0                                        2      1  primary   9.410      106.285    48.962          76.000             complete
Robin              Tiny F0        robin_BOLT_TINY_F0                                        2      4  primary   2.052      487.259    69.438          88.000             complete
Robin              Mini F0        robin_BOLT_MINI_F0                                        2      1  primary   8.862      112.920    104.169         138.000            complete
Robin              Mini F0        robin_BOLT_MINI_F0                                        2      4  primary   2.666      375.123    143.826         182.000            complete
Robin              Small F0       v6_robin_f0                                               2      1  primary   11.517     86.831     209.871         216.000            complete
Robin              Small F0       v6_robin_f0                                               2      4  primary   4.346      230.176    265.489         292.000            complete
Robin              Small + LEVEL  v6_robin_level_res_92602                                  2      1  primary   11.048     90.519     196.628         220.000            complete
Robin              Small + LEVEL  v6_robin_level_res_92602                                  2      4  primary   2.989      334.557    214.489         216.000            complete
Robin              Small + LEVEL  v6_robin_level_res_92601                                  2      1  primary   11.398     87.742     196.628         220.000            complete
Robin              Small + LEVEL  v6_robin_level_res_92601                                  2      4  primary   2.975      336.218    214.489         216.000            complete
Robin              Small F0       v6_robin_f0                                               2      1  post      12.534     79.847     209.871         216.000            complete
```

```text
Dataset            Block  F0 pre ms  F0 post ms  post/pre-1 %
Robin              0      11.491     11.429      -0.54
Robin              1      11.498     11.279      -1.91
Robin              2      11.788     12.534      6.34
Peacock Education  0      11.217     11.657      3.92
Peacock Education  1      11.029     11.168      1.26
Peacock Education  2      11.057     11.044      -0.12
Jena               0      11.190     11.030      -1.44
Jena               1      11.926     11.498      -3.59
Jena               2      10.939     11.804      7.91
```

이 표와 `cost_cuda_rows.json`/`cost_comparison.csv`는 전체108 block/sentinel과10 pass 원시간, resident allocated/reserved, CPU process memory 및 telemetry를 보존합니다. drift의 원인은 확정하지 않습니다. 이전 회차 timing과 이번 값을 나누어 속도비를 만들지 않았습니다.

## 자료별 Pareto와 Phase 0 판정

### Robin

- **LEVEL vs Mini F0 — Case 1:** `CHANNEL_COMPRESSION_DEPLOYMENT_ADVANTAGE_REJECTED_FOR_TESTED_SETTING`. Mini F0가 MSE·B4 peak allocated·B4 처리량의 세 주축에서 유리합니다. 이 조건의 압축 deployment 선택 주장을 축소합니다.

- **LEVEL vs Tiny F0 — Case 1:** `CHANNEL_COMPRESSION_DEPLOYMENT_ADVANTAGE_REJECTED_FOR_TESTED_SETTING`. Tiny F0가 MSE·B4 peak allocated·B4 처리량의 세 주축에서 유리합니다. 이 조건의 압축 deployment 선택 주장을 축소합니다.

```text
Arm            Point dominators in fixed grid
Small + LEVEL  Mini F0; Tiny F0
Small F0       Mini F0
Mini F0        None (point nondominated)
Tiny F0        None (point nondominated)
```

### Peacock Education

- **LEVEL vs Mini F0 — Case 4:** `NO_SINGLE_DOMINANT_ALLOCATION`. 대응 MSE 구간이 0을 포함하므로 정확도 차이는 불확실합니다. 동등성의 증거가 아닙니다.

- **LEVEL vs Tiny F0 — Case 4:** `NO_SINGLE_DOMINANT_ALLOCATION`. 대응 MSE 구간이 0을 포함하므로 정확도 차이는 불확실합니다. 동등성의 증거가 아닙니다.

```text
Arm            Point dominators in fixed grid
Small + LEVEL  Mini F0; Tiny F0
Small F0       None (point nondominated)
Mini F0        None (point nondominated)
Tiny F0        None (point nondominated)
```

### Jena

- **LEVEL vs Mini F0 — Case 1:** `CHANNEL_COMPRESSION_DEPLOYMENT_ADVANTAGE_REJECTED_FOR_TESTED_SETTING`. Mini F0가 MSE·B4 peak allocated·B4 처리량의 세 주축에서 유리합니다. 이 조건의 압축 deployment 선택 주장을 축소합니다.

- **LEVEL vs Tiny F0 — Case 4:** `NO_SINGLE_DOMINANT_ALLOCATION`. 대응 MSE 구간이 0을 포함하므로 정확도 차이는 불확실합니다. 동등성의 증거가 아닙니다.

```text
Arm            Point dominators in fixed grid
Small + LEVEL  Mini F0; Tiny F0
Small F0       None (point nondominated)
Mini F0        None (point nondominated)
Tiny F0        None (point nondominated)
```

Pareto 표는 MSE·B4 allocated·B4 throughput의 **점추정**만 사용한 기술적 고정 grid입니다. 조건부 구간이0을 포함하거나 비용축이 교차하면 단일 우위를 선언하지 않습니다. reserved·resident·B1·가중치 bytes·MAE를 포함한 모든 지표의 지배, 모든 K의 전선, 범용 우위를 뜻하지 않습니다. 임의 가중 효율 총점·비열등성 margin은 도입하지 않았습니다.

## 가장 강한 반론

- **A — 작은 backbone이 싸지는 것은 당연하다:** 맞습니다. 비용 감소의 신규성 대신 그 대가인 정확도와 실제 전체 배포 비용을 함께 비교하는 질문입니다.

- **B — LEVEL adaptation과 backbone 선택은 method 비교가 아니다:** 맞습니다. 같은 계열의 practical allocation/system comparison으로 명명하며 순수 LEVEL method ablation이나 압축만의 인과 효과로 쓰지 않습니다.

- **C — Mini/Tiny가 이기면 LEVEL 설계가 틀렸는가:** 아닙니다. 같은 Small 안의 COMPRESS→LEVEL, LEVEL/RAW 등 내부 구조 효과는 별도 근거이며 여기서는 deployment 선택 이유만 판정합니다.

- **D — Mini/Tiny에도 LEVEL을 붙이면 되는가:** 별도 후속 연구 질문입니다. 이번 primary 네 arm의 답을 얻는 데 필요하지 않아 자동 실행하지 않습니다.

## LEVEL 주장과 다음 실험

**유지:** 같은 Bolt-small 내 compression-only b 보완과 LEVEL/RAW 등의 기존 구조적 결과는 해당 내부 대조의 범위에 유지합니다. 이번 allocation에서 남는 근거: 내부 구조 대조의 기존 근거와 이번 자료별 조건부 결과.

**축소:** 큰 backbone을 유지하며 채널을 줄이면 작은 원채널 backbone보다 deployment가 유리하다는 주장은 Robin vs Mini F0; Robin vs Tiny F0; Jena vs Mini F0에 맞춰 축소합니다. 한 자료의 결과를 세 자료의 평균 총점이나 범용 우위로 확장하지 않습니다.

**보류:** 3개 주 비교는 차이 불확실·비용 축 교차·측정 범위 중첩 등의 이유로 단일 우위를 보류합니다. 선택 불확실성, 사전학습 중복 여부, 독립 재현과 보호된 외부 확인은 이 회차에서 해결하지 않았습니다.

**다음 실험:** 이번 고정 allocation 질문을 답하기 위해 Mini/Tiny+LEVEL, LoRA, 새 K·G·seed·dataset search는 필요하지 않으며 자동 실행하지 않습니다. 보편적 deployment 주장을 새로 하려면 노출되지 않은 자료·새 장치에서 사전 고정한 별도 확인이 필요합니다. 블록 중첩으로 안정적인 비용 순위를 원하는 경우에는 그 조건만 대상으로 독립 비용 세션을 별도 질문으로 검토할 수 있습니다.

## 근거와 실제 상태

새 fit0과 optimizer update0이며 Mini/Tiny는2모델×3자료의6 canonical TEST inference units입니다. 실제 실행 수·forward call·실패·재시도는 `prediction_manifest.json`, `evaluation_summary.json`, `ledger.json`의 실제 receipt를 기준으로 확인합니다. 기존 LEVEL/F0 prediction은 hash·shape·origin·mask binding 후 재사용했습니다. 검산 완료·commit/push 상태를 이 보고 생성기가 선행 선언하지 않으며 `final_checks.json`과 `STATUS.md`를 확인합니다. 자체검산은 독립 재현 또는 논문 신규성 인증이 아닙니다.

기존 Chronos-2/Chronos-2-Small 비교는 다른 family의 native multivariate system control이며 이번 same-family Bolt allocation과 별도입니다. 원본 PPT/PDF 편집 없이 `PRESENTATION_PATCH.md`에 수정 제안만 남겼습니다. 출력 그림과 CSV hash·사용 행 manifest는 `figures/manifest.json`에 있습니다.
