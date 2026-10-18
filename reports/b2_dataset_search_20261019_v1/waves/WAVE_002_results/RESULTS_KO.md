# 데이터·K 탐색 wave 2

등록한 positive 조건을 만족한 개발 예시는 ETTh1, appliances다.

판정은 DEFAULT128에서 B2 두 seed 모두 Raw보다 MSE가 작고, 평균 MSE가 1% 이상 작으며, 두 seed의 지연과 allocated 피크가 모두 Raw보다 작아야 한다. 이는 Raw 대비 탐색적 판정이다. Direct MLP가 더 좋은 경우도 함께 공개하며 MAE·COMMON_CAP은 별도 축으로 보고한다.

## ETTh1

판정: POSITIVE_EXPLORATORY.

### DEFAULT128

```text
Method        seeds  MSE mean [min,max]         MAE mean [min,max]         B8 ms mean [min,max]       allocated MiB [min,max]    reserved MiB [min,max]
Raw           1      0.283405 [0.283405,0.283405]  0.324030 [0.324030,0.324030]  14.994050 [14.994050,14.994050]  250.995605 [250.995605,250.995605]  262.000000 [262.000000,262.000000]
B2 K=1        2      0.472644 [0.455830,0.489457]  0.491693 [0.480011,0.503376]  11.223675 [11.077700,11.369650]  199.934082 [199.934082,199.934082]  270.000000 [270.000000,270.000000]
B2 K=2        2      0.376260 [0.365062,0.387458]  0.427401 [0.422178,0.432624]  12.084400 [11.084300,13.084500]  208.497559 [208.497559,208.497559]  270.000000 [270.000000,270.000000]
B2 K=4        2      0.277204 [0.276725,0.277682]  0.348309 [0.347895,0.348723]  11.553750 [11.435350,11.672150]  225.593750 [225.593750,225.593750]  262.000000 [262.000000,262.000000]
Direct MLP    2      0.289820 [0.284931,0.294710]  0.363745 [0.359164,0.368327]  0.389575 [0.389400,0.389750]  9.530273 [9.530273,9.530273]  22.000000 [22.000000,22.000000]
```

### COMMON_CAP

```text
Method        seeds  MSE mean [min,max]         MAE mean [min,max]         B8 ms mean [min,max]       allocated MiB [min,max]    reserved MiB [min,max]
Raw           1      0.283405 [0.283405,0.283405]  0.324030 [0.324030,0.324030]  41.010350 [41.010350,41.010350]  209.076660 [209.076660,209.076660]  274.000000 [274.000000,274.000000]
B2 K=1        2      0.472644 [0.455830,0.489457]  0.491693 [0.480011,0.503376]  12.556975 [11.021800,14.092150]  199.934082 [199.934082,199.934082]  270.000000 [270.000000,270.000000]
B2 K=2        2      0.376260 [0.365062,0.387458]  0.427401 [0.422178,0.432624]  11.086250 [11.022500,11.150000]  208.497559 [208.497559,208.497559]  270.000000 [270.000000,270.000000]
B2 K=4        2      0.277204 [0.276725,0.277682]  0.348309 [0.347895,0.348723]  21.698375 [21.694400,21.702350]  208.595703 [208.595703,208.595703]  262.000000 [262.000000,262.000000]
Direct MLP    2      0.289820 [0.284931,0.294710]  0.363745 [0.359164,0.368327]  0.410400 [0.401900,0.418900]  9.530273 [9.530273,9.530273]  22.000000 [22.000000,22.000000]
```

K=1: positive=False, Raw 대비 평균 MSE 감소 -66.77%, MAE 변화 +51.74%, 지연 속도비 1.34배, allocated 변화 -20.34%. Direct MLP의 MSE·지연·allocated 동시 지배: 평균=True, 짝지은 seed={"92601": true, "92602": true}. 조건별 판정={"both_seed_MSE_below_raw": false, "both_seed_default_allocated_below_raw": true, "both_seed_default_latency_below_raw": true, "mean_MSE_gain_at_least_1pct": false}.

K=2: positive=False, Raw 대비 평균 MSE 감소 -32.76%, MAE 변화 +31.90%, 지연 속도비 1.24배, allocated 변화 -16.93%. Direct MLP의 MSE·지연·allocated 동시 지배: 평균=True, 짝지은 seed={"92601": true, "92602": true}. 조건별 판정={"both_seed_MSE_below_raw": false, "both_seed_default_allocated_below_raw": true, "both_seed_default_latency_below_raw": true, "mean_MSE_gain_at_least_1pct": false}.

K=4: positive=True, Raw 대비 평균 MSE 감소 +2.19%, MAE 변화 +7.49%, 지연 속도비 1.30배, allocated 변화 -10.12%. Direct MLP의 MSE·지연·allocated 동시 지배: 평균=False, 짝지은 seed={"92601": false, "92602": false}. 조건별 판정={"both_seed_MSE_below_raw": true, "both_seed_default_allocated_below_raw": true, "both_seed_default_latency_below_raw": true, "mean_MSE_gain_at_least_1pct": true}.

V origin 36개 전체, 관측 목표가 있는 origin 36개, 관측 목표 12096개다. 원점별 관측 여부를 결과에 따라 제거하거나 목표를 보간하지 않았다.

A_fit 구조 진단: {"all_channel_missing_rows": 0, "columns": 7, "constant_columns": [], "missing_fraction_per_channel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "numerical_rank": 7, "observed_counts": [8710, 8710, 8710, 8710, 8710, 8710, 8710], "participation_effective_rank": 3.572062877304564, "rank_definition": "Observed A_fit channel centering/scaling; missing cells set to A_fit mean solely for this structure diagnostic; no interpolation or forecast scoring", "rows": 8710, "scope": "A_fit_only", "selection_information": "TRAIN structure only; no target error or A_stop/M_cal/V_select/T_final value enters this diagnostic", "standardized_energy_fraction_by_k": {"2": 0.6655088005741439, "4": 0.9302931880854464}, "unobserved_columns": []}.

## ETTh2 (safe 17400 cohort)

판정: NEGATIVE_EXPLORATORY.

### DEFAULT128

```text
Method        seeds  MSE mean [min,max]         MAE mean [min,max]         B8 ms mean [min,max]       allocated MiB [min,max]    reserved MiB [min,max]
Raw           1      0.259555 [0.259555,0.259555]  0.312753 [0.312753,0.312753]  16.416450 [16.416450,16.416450]  250.995605 [250.995605,250.995605]  262.000000 [262.000000,262.000000]
B2 K=1        2      0.340582 [0.338782,0.342382]  0.400182 [0.398170,0.402193]  11.295775 [11.223700,11.367850]  199.934082 [199.934082,199.934082]  270.000000 [270.000000,270.000000]
B2 K=2        2      0.292472 [0.282343,0.302601]  0.370946 [0.367349,0.374543]  11.238525 [11.096100,11.380950]  208.497559 [208.497559,208.497559]  270.000000 [270.000000,270.000000]
B2 K=4        2      0.287618 [0.279164,0.296072]  0.369738 [0.361144,0.378333]  12.992450 [12.090550,13.894350]  225.593750 [225.593750,225.593750]  262.000000 [262.000000,262.000000]
Direct MLP    2      0.248751 [0.248425,0.249077]  0.323901 [0.323318,0.324484]  0.387300 [0.383150,0.391450]  9.530273 [9.530273,9.530273]  22.000000 [22.000000,22.000000]
```

### COMMON_CAP

```text
Method        seeds  MSE mean [min,max]         MAE mean [min,max]         B8 ms mean [min,max]       allocated MiB [min,max]    reserved MiB [min,max]
Raw           1      0.259555 [0.259555,0.259555]  0.312753 [0.312753,0.312753]  42.936850 [42.936850,42.936850]  209.076660 [209.076660,209.076660]  274.000000 [274.000000,274.000000]
B2 K=1        2      0.340582 [0.338782,0.342382]  0.400182 [0.398170,0.402193]  11.049925 [10.832500,11.267350]  199.934082 [199.934082,199.934082]  270.000000 [270.000000,270.000000]
B2 K=2        2      0.292472 [0.282343,0.302601]  0.370946 [0.367349,0.374543]  11.781250 [11.333800,12.228700]  208.497559 [208.497559,208.497559]  270.000000 [270.000000,270.000000]
B2 K=4        2      0.287618 [0.279164,0.296072]  0.369738 [0.361144,0.378333]  26.752350 [25.617150,27.887550]  208.595703 [208.595703,208.595703]  262.000000 [262.000000,262.000000]
Direct MLP    2      0.248751 [0.248425,0.249077]  0.323901 [0.323318,0.324484]  0.387300 [0.384550,0.390050]  9.530273 [9.530273,9.530273]  22.000000 [22.000000,22.000000]
```

K=1: positive=False, Raw 대비 평균 MSE 감소 -31.22%, MAE 변화 +27.95%, 지연 속도비 1.45배, allocated 변화 -20.34%. Direct MLP의 MSE·지연·allocated 동시 지배: 평균=True, 짝지은 seed={"92601": true, "92602": true}. 조건별 판정={"both_seed_MSE_below_raw": false, "both_seed_default_allocated_below_raw": true, "both_seed_default_latency_below_raw": true, "mean_MSE_gain_at_least_1pct": false}.

K=2: positive=False, Raw 대비 평균 MSE 감소 -12.68%, MAE 변화 +18.61%, 지연 속도비 1.46배, allocated 변화 -16.93%. Direct MLP의 MSE·지연·allocated 동시 지배: 평균=True, 짝지은 seed={"92601": true, "92602": true}. 조건별 판정={"both_seed_MSE_below_raw": false, "both_seed_default_allocated_below_raw": true, "both_seed_default_latency_below_raw": true, "mean_MSE_gain_at_least_1pct": false}.

K=4: positive=False, Raw 대비 평균 MSE 감소 -10.81%, MAE 변화 +18.22%, 지연 속도비 1.26배, allocated 변화 -10.12%. Direct MLP의 MSE·지연·allocated 동시 지배: 평균=True, 짝지은 seed={"92601": true, "92602": true}. 조건별 판정={"both_seed_MSE_below_raw": false, "both_seed_default_allocated_below_raw": true, "both_seed_default_latency_below_raw": true, "mean_MSE_gain_at_least_1pct": false}.

V origin 36개 전체, 관측 목표가 있는 origin 36개, 관측 목표 12096개다. 원점별 관측 여부를 결과에 따라 제거하거나 목표를 보간하지 않았다.

A_fit 구조 진단: {"all_channel_missing_rows": 0, "columns": 7, "constant_columns": [], "missing_fraction_per_channel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "numerical_rank": 7, "observed_counts": [8700, 8700, 8700, 8700, 8700, 8700, 8700], "participation_effective_rank": 3.3411078425204024, "rank_definition": "Observed A_fit channel centering/scaling; missing cells set to A_fit mean solely for this structure diagnostic; no interpolation or forecast scoring", "rows": 8700, "scope": "A_fit_only", "selection_information": "TRAIN structure only; no target error or A_stop/M_cal/V_select/T_final value enters this diagnostic", "standardized_energy_fraction_by_k": {"2": 0.7239381481060447, "4": 0.9364365071057261}, "unobserved_columns": []}.

## Appliances

판정: POSITIVE_EXPLORATORY.

### DEFAULT128

```text
Method        seeds  MSE mean [min,max]         MAE mean [min,max]         B8 ms mean [min,max]       allocated MiB [min,max]    reserved MiB [min,max]
Raw           1      0.347579 [0.347579,0.347579]  0.333530 [0.333530,0.333530]  40.647600 [40.647600,40.647600]  330.239258 [330.239258,330.239258]  354.000000 [354.000000,354.000000]
B2 K=4        2      0.295668 [0.292607,0.298729]  0.349749 [0.343338,0.356160]  12.874850 [11.908200,13.841500]  225.897461 [225.897461,225.897461]  264.000000 [264.000000,264.000000]
B2 K=7        2      0.292344 [0.289512,0.295175]  0.344820 [0.337089,0.352550]  15.403525 [15.269900,15.537150]  251.543457 [251.543457,251.543457]  268.000000 [268.000000,268.000000]
B2 K=14       2      0.289974 [0.285803,0.294144]  0.345168 [0.341618,0.348719]  23.509250 [23.494250,23.524250]  311.384277 [311.384277,311.384277]  354.000000 [354.000000,354.000000]
Direct MLP    2      0.251567 [0.250582,0.252553]  0.309531 [0.306282,0.312780]  0.510050 [0.509450,0.510650]  10.439453 [10.439453,10.439453]  24.000000 [24.000000,24.000000]
```

### COMMON_CAP

```text
Method        seeds  MSE mean [min,max]         MAE mean [min,max]         B8 ms mean [min,max]       allocated MiB [min,max]    reserved MiB [min,max]
Raw           1      0.347579 [0.347579,0.347579]  0.333530 [0.333530,0.333530]  57.146350 [57.146350,57.146350]  251.835449 [251.835449,251.835449]  262.000000 [262.000000,262.000000]
B2 K=4        2      0.295668 [0.292607,0.298729]  0.349749 [0.343338,0.356160]  12.084125 [12.016650,12.151600]  225.897461 [225.897461,225.897461]  264.000000 [264.000000,264.000000]
B2 K=7        2      0.292344 [0.289512,0.295175]  0.344820 [0.337089,0.352550]  15.310925 [15.267650,15.354200]  251.543457 [251.543457,251.543457]  268.000000 [268.000000,268.000000]
B2 K=14       2      0.289974 [0.285803,0.294144]  0.345168 [0.341618,0.348719]  32.789400 [32.408100,33.170700]  251.890625 [251.890625,251.890625]  274.000000 [274.000000,274.000000]
Direct MLP    2      0.251567 [0.250582,0.252553]  0.309531 [0.306282,0.312780]  0.603400 [0.519200,0.687600]  10.439453 [10.439453,10.439453]  24.000000 [24.000000,24.000000]
```

K=4: positive=True, Raw 대비 평균 MSE 감소 +14.94%, MAE 변화 +4.86%, 지연 속도비 3.16배, allocated 변화 -31.60%. Direct MLP의 MSE·지연·allocated 동시 지배: 평균=True, 짝지은 seed={"92601": true, "92602": true}. 조건별 판정={"both_seed_MSE_below_raw": true, "both_seed_default_allocated_below_raw": true, "both_seed_default_latency_below_raw": true, "mean_MSE_gain_at_least_1pct": true}.

K=7: positive=True, Raw 대비 평균 MSE 감소 +15.89%, MAE 변화 +3.38%, 지연 속도비 2.64배, allocated 변화 -23.83%. Direct MLP의 MSE·지연·allocated 동시 지배: 평균=True, 짝지은 seed={"92601": true, "92602": true}. 조건별 판정={"both_seed_MSE_below_raw": true, "both_seed_default_allocated_below_raw": true, "both_seed_default_latency_below_raw": true, "mean_MSE_gain_at_least_1pct": true}.

K=14: positive=True, Raw 대비 평균 MSE 감소 +16.57%, MAE 변화 +3.49%, 지연 속도비 1.73배, allocated 변화 -5.71%. Direct MLP의 MSE·지연·allocated 동시 지배: 평균=True, 짝지은 seed={"92601": true, "92602": true}. 조건별 판정={"both_seed_MSE_below_raw": true, "both_seed_default_allocated_below_raw": true, "both_seed_default_latency_below_raw": true, "mean_MSE_gain_at_least_1pct": true}.

V origin 41개 전체, 관측 목표가 있는 origin 41개, 관측 목표 51168개다. 원점별 관측 여부를 결과에 따라 제거하거나 목표를 보간하지 않았다.

A_fit 구조 진단: {"all_channel_missing_rows": 0, "columns": 26, "constant_columns": [], "missing_fraction_per_channel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "numerical_rank": 26, "observed_counts": [9867, 9867, 9867, 9867, 9867, 9867, 9867, 9867, 9867, 9867, 9867, 9867, 9867, 9867, 9867, 9867, 9867, 9867, 9867, 9867, 9867, 9867, 9867, 9867, 9867, 9867], "participation_effective_rank": 5.325071090153028, "rank_definition": "Observed A_fit channel centering/scaling; missing cells set to A_fit mean solely for this structure diagnostic; no interpolation or forecast scoring", "rows": 9867, "scope": "A_fit_only", "selection_information": "TRAIN structure only; no target error or A_stop/M_cal/V_select/T_final value enters this diagnostic", "standardized_energy_fraction_by_k": {"13": 0.9510971709760658, "7": 0.8272116777655374}, "unobserved_columns": []}.

## 학습·검산과 해석 범위

```text
run_id  selected_epoch  search_train_origins  training_status
B2_ETTh1_K1_OFF_s92601  3.0  8151  CAP3_STILL_IMPROVING
B2_ETTh1_K1_OFF_s92602  3.0  8151  CAP3_STILL_IMPROVING
B2_ETTh1_K2_OFF_s92601  1.0  8151  CAP3_CONVERGENCE_UNRESOLVED
B2_ETTh1_K2_OFF_s92602  1.0  8151  CAP3_CONVERGENCE_UNRESOLVED
B2_ETTh1_K4_OFF_s92601  0.25  8151  CAP3_CONVERGENCE_UNRESOLVED
B2_ETTh1_K4_OFF_s92602  0.25  8151  CAP3_CONVERGENCE_UNRESOLVED
B2_ETTh2_safe17400_K1_OFF_s92601  3.0  8141  CAP3_STILL_IMPROVING
B2_ETTh2_safe17400_K1_OFF_s92602  3.0  8141  CAP3_STILL_IMPROVING
B2_ETTh2_safe17400_K2_OFF_s92601  0.25  8141  CAP3_CONVERGENCE_UNRESOLVED
B2_ETTh2_safe17400_K2_OFF_s92602  1.0  8141  CAP3_CONVERGENCE_UNRESOLVED
B2_ETTh2_safe17400_K4_OFF_s92601  1.0  8141  CAP3_CONVERGENCE_UNRESOLVED
B2_ETTh2_safe17400_K4_OFF_s92602  3.0  8141  CAP3_STILL_IMPROVING
B2_appliances_K14_OFF_s92601  0.25  9308  CAP3_CONVERGENCE_UNRESOLVED
B2_appliances_K14_OFF_s92602  0.25  9308  CAP3_CONVERGENCE_UNRESOLVED
B2_appliances_K4_OFF_s92601  1.0  9308  CAP3_CONVERGENCE_UNRESOLVED
B2_appliances_K4_OFF_s92602  1.0  9308  CAP3_CONVERGENCE_UNRESOLVED
B2_appliances_K7_OFF_s92601  0.25  9308  CAP3_CONVERGENCE_UNRESOLVED
B2_appliances_K7_OFF_s92602  1.0  9308  CAP3_CONVERGENCE_UNRESOLVED
DirectMLP_ETTh1_s92601  1.0  8151  CAP3_CONVERGENCE_UNRESOLVED
DirectMLP_ETTh1_s92602  3.0  8151  CAP3_STILL_IMPROVING
DirectMLP_ETTh2_safe17400_s92601  3.0  8141  CAP3_STILL_IMPROVING
DirectMLP_ETTh2_safe17400_s92602  1.0  8141  CAP3_CONVERGENCE_UNRESOLVED
DirectMLP_appliances_s92601  3.0  9308  CAP3_STILL_IMPROVING
DirectMLP_appliances_s92602  1.0  9308  CAP3_CONVERGENCE_UNRESOLVED
```

TRAIN은 실제 stride1 전체 origin이며 진단용 stride48 목록 길이가 아니다. 모든 B2 검색을 A_stop에서 strict minimum·earlier tie로 고른 뒤 wave 전체 선택을 봉인하고 fresh A60 refit/V를 실행했다. Direct MLP도 자체 고정 A_stop 선택 후 fresh refit했다. cap3에서 계속 개선한 arm은 수렴한 것으로 해석하지 않는다.

검산한 비용 facet 54개, warmup 540회, timed 1080회다. B2 검색 18개와 fresh refit 18개, Direct MLP search/refit 6쌍을 포함한다. private 전체 V NPZ의 bytes·array 지문·FP64 equal-channel 관측 macro MSE/MAE, raw timing quantile·메모리 최대·F counters, A60 scaler·physical history/target mask·source/model/checkpoint/선택 봉인을 독립 검산했다. physical 출력 byte 배열은 저장되지 않아 그 지문의 재계산까지 주장하지 않는다.

첫 8개 고정 V origin의 physical+mask→causal fill→고정 scaler→실제 모델→inverse→CPU 반환 전체 경로를 잰다. window slicing·load·검산·hash는 시간에서 제외한다. 피크는 CUDA allocated/reserved 텐서 기준이며 프로세스 hard cap이 아니다. 20회 p95는 서비스 tail 보장이 아니고 두 seed 범위는 통계 신뢰구간이 아니다. 각 facet의 (PID, 생성 시각) pair가 모두 달라야 한다. OS가 같은 PID를 다른 생성 시각에 재사용하는 경우는 허용한다.

H48/L512는 같은 스텝 수지만 실시간 길이는 다르다. 과거 Jena H48=8시간; ETTh2 (safe 17400 cohort) H48=48시간; Appliances H48=8시간; ETTh1 H48=48시간. 강제 공통 주파수 리샘플링 없이 데이터 내부 방법 간 조건을 맞췄다. 서로 다른 데이터의 MSE 숫자를 직접 비교하지 않는다.

현재 결과는 순차적으로 후보를 찾는 development V 탐색이다. 이전 다른 방법의 데이터 사용, 관련 원천·주파수 변형의 공유, 기반모델 사전학습 중복 가능성을 배제하지 못한다. 새로운 독립 TEST 성능이나 통계적 유의성을 주장하지 않는다. LoRA·Ridge는 새 wave에서 실행하지 않았다. K·seed·학습 recipe·채널은 V 결과로 바꾸지 않았다.

JENA_HISTORICAL_CONTEXT.csv는 과거 memory_budget_20261018_v1의 별도 실행 B8·confirm 결과다. K3/6/12·Raw·merged LoRA를 문맥으로 보존하며 새 후보 판정에는 쓰지 않는다. 이전 Jena 정규화/데이터 ancestry와 과거 TEST 노출 한계가 유지된다. 과거 CUBLAS 환경의 실제 값은 기록되지 않았고, 새 비용은 부모 환경을 상속한 실제 값을 receipt에 기록했다.

## 준비 정정과 전체 후보 보존

PREPARATION_AMENDMENT_001에 따라 최초 ETTh2 prefix13936 중 16시간은 더 이른 같은 물리 원천 ETTm2 보호 경계와 겹칠 가능성이 있었다. 최초 준비에서 그 수치가 materialize된 사실은 보존·공개한다. 해당 wave는 GPU 학습·예측 점수 계산 전에 superseded되었고, 그 16시간의 새 학습/forecast 사용은 0이다. 원본·진단·WAVE001을 삭제하거나 덮어쓰지 않았다.

교체 ETTh2_safe17400은 원천17420행의 처음17400좌표를 쓰는 custom development cohort이며 공식 ETT split이 아니다. 기존 superseded NPZ를 자르는 대신 원본에서 허용된 prefix13920행만 다시 준비했다. 7개 물리 채널·hourly 주기는 유지했고 cutoff 결정에 예측 점수나 학습 recipe 변경을 사용하지 않았다.

```text
wave dataset status detail
1 ETTh2 SUPERSEDED_BEFORE_TRAINING Original ETTh2 preparation possible shared-source 16-hour overlap; wave never trained
1 appliances SUPERSEDED_BEFORE_TRAINING Original ETTh2 preparation possible shared-source 16-hour overlap; wave never trained
1 ETTh1 SUPERSEDED_BEFORE_TRAINING Original ETTh2 preparation possible shared-source 16-hour overlap; wave never trained
1 uci360_air_quality NOT_READY_UNOBSERVED_CHANNEL [{"role": "A_stop", "unobserved_columns": ["NMHC(GT)"]}]
2 ETTh2_safe17400 NEGATIVE_EXPLORATORY All K and both seeds disclosed; development V, not independent TEST
2 appliances POSITIVE_EXPLORATORY All K and both seeds disclosed; development V, not independent TEST
2 ETTh1 POSITIVE_EXPLORATORY All K and both seeds disclosed; development V, not independent TEST
3 ETTm1 READY_NOT_EXECUTED Conditional prefix preparation only; zero training/forecast experiments counted in this report. GPU activation is allowed only after the complete parent wave has no verified positive. Private numeric prefix was not opened or hashed by analysis.
3 solar_full137 READY_NOT_EXECUTED Conditional prefix preparation only; zero training/forecast experiments counted in this report. GPU activation is allowed only after the complete parent wave has no verified positive. Private numeric prefix was not opened or hashed by analysis.
```

조건부 후속 후보는 CANDIDATE_PLAN_003 등 등록 계획과 준비 JSON의 상태만 읽어 표시했다. ETTm1(C7, K1/2/4)·Solar full137(C137, K18/35/70)은 현재 wave에 포함된 성능 실험으로 세지 않는다. 부모 wave 전체 검산 후 positive가 없을 때만 다음 GPU wave를 활성화하며, positive가 있으면 준비된 후보를 READY_NOT_EXECUTED로 보존한다. 분석기는 이 조건부 후보의 private numeric NPZ를 열거나 hash하지 않았다.

전체 seed 원자료는 RUNS.csv, 평균·범위는 SUMMARY.csv, 모든 후보 상태는 CANDIDATES.csv에 있다.