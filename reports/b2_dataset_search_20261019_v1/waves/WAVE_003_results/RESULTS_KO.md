# ETTm1 데이터·K 탐색 wave 3

판정: POSITIVE_EXPLORATORY. ETTm1 C7의 K1/2/4와 두 seed92601/92602를 모두 실제 실행한 wave다.

등록 조건은 DEFAULT128에서 두 seed의 MSE가 각각 Raw보다 낮고 평균 MSE가 1% 이상 낮으며, 두 seed의 whole B8 중앙 지연과 allocated 피크도 각각 Raw보다 낮아야 한다. MAE 개선이나 Direct MLP 우위까지 뜻하지 않으며 이 두 축도 함께 공개한다.

원래 N69680 중 허용 prefix55744행과 7개 원채널을 유지했다. native 15분 간격이므로 H48=12시간, L512=128시간이다. ETTh1은 같은 transformer station의 hourly 변형으로, 서로 독립적인 새 도메인으로 세지 않는다. ETTh1 H48=48시간·L512=512시간, 과거 Jena H48=8시간과 실제 예측 시간은 다르다. 강제 리샘플링이나 서로 다른 데이터 간 MSE 숫자의 직접 비교는 하지 않는다.

## DEFAULT128

```text
Method       seeds  MSE mean [min,max]           MAE mean [min,max]           B8 ms mean [min,max]         allocated MiB [min,max]      reserved MiB [min,max]
Raw          1      0.278007 [0.278007,0.278007]  0.292479 [0.292479,0.292479]  16.379350 [16.379350,16.379350]  250.995605 [250.995605,250.995605]  266.000000 [266.000000,266.000000]
B2 K=1       2      0.339939 [0.333789,0.346089]  0.404090 [0.400861,0.407320]  12.001075 [11.724700,12.277450]  199.934082 [199.934082,199.934082]  270.000000 [270.000000,270.000000]
B2 K=2       2      0.294847 [0.290531,0.299164]  0.361427 [0.357794,0.365061]  12.309800 [12.198950,12.420650]  208.497559 [208.497559,208.497559]  262.000000 [262.000000,262.000000]
B2 K=4       2      0.237962 [0.237670,0.238254]  0.305800 [0.303062,0.308539]  12.657250 [12.639100,12.675400]  225.593750 [225.593750,225.593750]  264.000000 [264.000000,264.000000]
Direct MLP   2      0.241710 [0.234186,0.249235]  0.310964 [0.308410,0.313518]  0.407025 [0.401900,0.412150]  9.530273 [9.530273,9.530273]  22.000000 [22.000000,22.000000]
```

## COMMON_CAP

```text
Method       seeds  MSE mean [min,max]           MAE mean [min,max]           B8 ms mean [min,max]         allocated MiB [min,max]      reserved MiB [min,max]
Raw          1      0.278007 [0.278007,0.278007]  0.292479 [0.292479,0.292479]  45.809850 [45.809850,45.809850]  209.076660 [209.076660,209.076660]  270.000000 [270.000000,270.000000]
B2 K=1       2      0.339939 [0.333789,0.346089]  0.404090 [0.400861,0.407320]  12.184700 [12.071350,12.298050]  199.934082 [199.934082,199.934082]  270.000000 [270.000000,270.000000]
B2 K=2       2      0.294847 [0.290531,0.299164]  0.361427 [0.357794,0.365061]  12.098050 [12.094700,12.101400]  208.497559 [208.497559,208.497559]  262.000000 [262.000000,262.000000]
B2 K=4       2      0.237962 [0.237670,0.238254]  0.305800 [0.303062,0.308539]  23.899775 [23.448200,24.351350]  208.595703 [208.595703,208.595703]  264.000000 [264.000000,264.000000]
Direct MLP   2      0.241710 [0.234186,0.249235]  0.310964 [0.308410,0.313518]  0.405300 [0.395450,0.415150]  9.530273 [9.530273,9.530273]  22.000000 [22.000000,22.000000]
```

K=1: positive=False; 두 seed MSE 92601: 0.346089 (Raw보다 낮음=False), 92602: 0.333789 (Raw보다 낮음=False). Raw MSE=0.278007. 평균 MSE 감소 -22.28%, MAE 변화 +38.16%, 지연 속도비 1.365배, allocated 변화 -20.34%. 조건별 판정={"both_seed_MSE_below_raw": false, "both_seed_default_allocated_below_raw": true, "both_seed_default_latency_below_raw": true, "mean_MSE_gain_at_least_1pct": false}.

Direct MLP 대비 K=1 평균 MSE 변화 +40.64%, 지연 비 29.485배, allocated 비 20.979배다. Direct MLP의 MSE·지연·allocated 동시 지배는 평균=True, 짝지은 seed={"92601": true, "92602": true}다.

K=2: positive=False; 두 seed MSE 92601: 0.290531 (Raw보다 낮음=False), 92602: 0.299164 (Raw보다 낮음=False). Raw MSE=0.278007. 평균 MSE 감소 -6.06%, MAE 변화 +23.57%, 지연 속도비 1.331배, allocated 변화 -16.93%. 조건별 판정={"both_seed_MSE_below_raw": false, "both_seed_default_allocated_below_raw": true, "both_seed_default_latency_below_raw": true, "mean_MSE_gain_at_least_1pct": false}.

Direct MLP 대비 K=2 평균 MSE 변화 +21.98%, 지연 비 30.243배, allocated 비 21.877배다. Direct MLP의 MSE·지연·allocated 동시 지배는 평균=True, 짝지은 seed={"92601": true, "92602": true}다.

K=4: positive=True; 두 seed MSE 92601: 0.237670 (Raw보다 낮음=True), 92602: 0.238254 (Raw보다 낮음=True). Raw MSE=0.278007. 평균 MSE 감소 +14.40%, MAE 변화 +4.55%, 지연 속도비 1.294배, allocated 변화 -10.12%. 조건별 판정={"both_seed_MSE_below_raw": true, "both_seed_default_allocated_below_raw": true, "both_seed_default_latency_below_raw": true, "mean_MSE_gain_at_least_1pct": true}.

Direct MLP 대비 K=4 평균 MSE 변화 -1.55%, 지연 비 31.097배, allocated 비 23.671배다. Direct MLP의 MSE·지연·allocated 동시 지배는 평균=False, 짝지은 seed={"92601": true, "92602": false}다.

## 학습·검산과 해석 범위

전체 V origin 145개, 관측 목표가 있는 origin 145개, 관측 scalar target 48720개다. 목표를 보간하거나 결과에 따라 원점을 제외하지 않았다.

A_fit 구조 진단: {"all_channel_missing_rows": 0, "columns": 7, "constant_columns": [], "missing_fraction_per_channel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "numerical_rank": 7, "observed_counts": [34840, 34840, 34840, 34840, 34840, 34840, 34840], "participation_effective_rank": 3.55673795882, "rank_definition": "Observed A_fit channel centering/scaling; missing cells set to A_fit mean solely for this structure diagnostic; no interpolation or forecast scoring", "rows": 34840, "scope": "A_fit_only", "selection_information": "TRAIN structure only; no target error or A_stop/M_cal/V_select/T_final value enters this diagnostic", "standardized_energy_fraction_by_k": {"2": 0.6668727966708424, "4": 0.9299864551376296}, "unobserved_columns": []}

```text
run_id  selected_epoch  search_train_origins  training_status
B2_ETTm1_K1_OFF_s92601  3.0  34281  CAP3_STILL_IMPROVING
B2_ETTm1_K1_OFF_s92602  3.0  34281  CAP3_STILL_IMPROVING
B2_ETTm1_K2_OFF_s92601  0.25  34281  CAP3_CONVERGENCE_UNRESOLVED
B2_ETTm1_K2_OFF_s92602  1.0  34281  CAP3_CONVERGENCE_UNRESOLVED
B2_ETTm1_K4_OFF_s92601  0.25  34281  CAP3_CONVERGENCE_UNRESOLVED
B2_ETTm1_K4_OFF_s92602  0.25  34281  CAP3_CONVERGENCE_UNRESOLVED
DirectMLP_ETTm1_s92601  3.0  34281  CAP3_STILL_IMPROVING
DirectMLP_ETTm1_s92602  1.0  34281  CAP3_CONVERGENCE_UNRESOLVED
```

TRAIN은 stride1 전체 origin이다. A_stop의 strict minimum·earlier tie로 고른 전체 B2 선택을 봉인한 뒤 fresh A60 refit/V를 수행했고, Direct MLP는 자체 고정 A_stop 선택 후 fresh refit했다. cap3에서 계속 개선하는 arm을 수렴했다고 주장하지 않는다. RevIN OFF, FP32, L512/H48 및 고정 recipe는 유지했다.

B2 검색 6개·fresh refit 6개, Direct MLP search/refit 2쌍, 비용 18facet, warmup 180회·timed 360회를 검산했다. driver exit0과 train/baselines/costs 완료 journal을 확인한 뒤에만 결과를 읽었다.

각 facet은 별도 (PID, 생성 시각)의 fresh process다. 첫 8개 고정 V origin의 physical+mask→causal fill→고정 scaler→실제 모델→inverse→CPU 반환을 측정했다. slicing/load/hash/검산은 시간에서 제외한다. 전체 V 예측과 두 row 정책의 parity, FP64 관측 equal-channel macro MSE/MAE, A60 scaler·physical history/target mask, raw trial quantile·allocated/reserved 최대, source/checkpoint·선택 봉인을 독립 검산했다. physical 출력 byte 배열이 저장되지 않아 그 hash 재계산은 주장하지 않는다.

두 seed min–max는 신뢰구간이 아니며 timed20회 p95는 서비스 tail 보장이 아니다. CUDA allocated/reserved 피크는 프로세스 hard cap이 아니다. COMMON_CAP=16과 DEFAULT128은 같은 선택 모델·V를 사용하지만 별도 비용 정책이다. MAE와 reserved 메모리도 표에 남겼다.

순차적 development V 후보 탐색이며 독립 TEST나 통계적 유의성 결론이 아니다. 이전 데이터 사용·관련 주파수 원천 공유와 기반모델 사전학습 중복 가능성은 배제하지 못한다. K·seed·채널·recipe는 새 V 결과에 따라 바꾸지 않았다. LoRA/Ridge는 새 wave에서 실행하지 않았다.

JENA_HISTORICAL_CONTEXT.csv는 과거 memory_budget_20261018_v1 B8 confirm의 K3/6/12·Raw·merged LoRA 별도 실행을 역사적 문맥으로 보존한다. 새 ETTm1 판정에는 쓰지 않으며 기존 ancestry/TEST 노출 한계가 유지된다. 과거 실제 CUBLAS 값은 미기록이고 새 비용은 부모 상속값을 receipt에 기록했다.

## 활성화 정정과 역사적 준비 provenance

ACTIVATION_AMENDMENT_WAVE_003은 사용자의 추가 데이터 사례 요청을 새 점수 전에 등록해 ETTm1을 활성화했다. 이 최신 승인은 wave2 positive가 있으면 중단한다는 이전 조건을 이번 wave에 한해 대체한다. wave2 Appliances·ETTh1 positive와 원래 계획은 보존한다.

PREPARATION_AMENDMENT_001의 최초 ETTh2 prefix13936 중 16시간과 같은 물리 원천 ETTm2의 더 이른 보호 경계 겹침 가능성은 역사적 provenance다. 당시 수치 materialize 사실과 superseded WAVE001·원본을 보존했고 그 16시간의 새 학습/forecast 사용은 0이었다. 교체 ETTh2_safe17400의 prefix13920은 원본에서 다시 준비한 custom cohort다. 이 ETTh2/ETTm2 준비 사건은 fresh ETTm1 prefix와 무관하며, 이번 wave에 상속된 후보 이력으로만 공개한다. ETTm1은 N69680/prefix55744와 같은 station의 ETTh1 cutoff를 따로 검증한 native 15분 자료다.

새 보호 T 수치나 전체 raw 파일의 numeric hash는 읽거나 재계산하지 않았다. 소비 prefix의 SHA/array 지문과 과거 metadata의 full-source SHA를 구분한다. 분석은 미실행 Solar numeric NPZ도 열거나 hash하지 않는다.

```text
wave dataset status detail
1 ETTh2 SUPERSEDED_BEFORE_TRAINING Original ETTh2 preparation possible shared-source 16-hour overlap; wave never trained
1 appliances SUPERSEDED_BEFORE_TRAINING Original ETTh2 preparation possible shared-source 16-hour overlap; wave never trained
1 ETTh1 SUPERSEDED_BEFORE_TRAINING Original ETTh2 preparation possible shared-source 16-hour overlap; wave never trained
1 uci360_air_quality NOT_READY_UNOBSERVED_CHANNEL [{"role": "A_stop", "unobserved_columns": ["NMHC(GT)"]}]
2 ETTh2_safe17400 PRESERVED_PRIOR_WAVE All K and both seeds disclosed; development V, not independent TEST
2 appliances PRESERVED_PRIOR_WAVE All K and both seeds disclosed; development V, not independent TEST
2 ETTh1 PRESERVED_PRIOR_POSITIVE_EXPLORATORY Prior wave2 positive preserved by activation amendment; not rerun or counted in wave3.
3 ETTm1 POSITIVE_EXPLORATORY All K and both seeds disclosed; development V, not independent TEST
NEXT_UNREGISTERED solar_full137 READY_NOT_EXECUTED Unexecuted full137-channel Solar; metadata only, numeric prefix not opened or hashed. Additional example found in wave3; retain prepared candidate.
```

Solar full137은 준비된 READY_NOT_EXECUTED로 보존한다. Solar는 이번 wave 성능 실험 수에 포함하지 않는다. 별도 RawContext 캠페인의 실행·예산과도 분리한다.

전체 seed는 RUNS.csv, 평균·범위는 SUMMARY.csv, 후보 이력은 CANDIDATES.csv, 두 비용 정책과 그림은 ETTm1별 파일을 따른다.