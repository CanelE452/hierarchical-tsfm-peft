# 원채널 출력·native head 적응 S1 결과

[확인] S1 계산은 완료되었고 선택·채점·비용·자원 검산은 PASS다. 이 표시는 독립 확증, 내부 적응의 필수성, 새 방법의 신규성을 뜻하지 않는다. 이번 질문은 추가 모듈을 제안하기 전에 표준 출력 보정과 원래 헤드 적응만으로 남은 오차가 얼마나 줄어드는지 확인하는 것이다.

OUTPUT_AFFINE은 F0의 원채널 예측에 C→C affine을 적용하며 채널을 섞는다. NATIVE_HEAD는 동결 encode/Transformer decode 표현을 읽는 원래 비선형 output_patch_embedding 전체를 적응한다. HEAD는 가중치를 공유하는 채널독립 함수다. 두 경로의 차이에는 표현·용량·채널 결합 구조가 함께 들어 있으므로 순수 정보 효과로 해석할 수 없다. HEAD의 2,526,336개 학습 변수는 기존 MSE-LoRA의 294,912개보다 많고, 직접 point loss에 연결되는 계산상 scalars는 1,173,600개다.

[확인] 사전 S0는 모든 canonical VAL에서 초기 OUTPUT/HEAD=F0, online/cache 값·loss·gradient, 결측 effective B4와 micro4/2/1의 분모·gradient, original normalization/inverse, frozen 모든 buffers와 Adam/scheduler/RNG 복원을 검증했다. 이 정상성 검증은 HEAD를 충분히 최적화했다는 보장이 아니다.

[확인] 정확도는 기존 TRAIN 표준화 좌표와 native inverse normalization 뒤의 관측 target/channel-macro 손실이다. 두 seed의 손실을 평균했고 예측 ensemble을 만들지 않았다. TEST-A/B combined는 채널별 error sums와 관측 count를 합친 뒤 ratio와 채널 평균을 계산했다. MSE/MAE는 낮을수록 좋다. 아래 F0 대비 감소율은 100×(F0−후보)/F0이며 양수는 개선, 음수는 악화다.

```text
자료                F0 MSE     OUTPUT MSE  HEAD MSE    OUTPUT 감소%  HEAD 감소%
robin               0.229095   0.275511    0.229719    -20.260        -0.272
peacock_education   0.201521   0.199447    0.201351    +1.029        +0.085
jena                0.292300   0.265080    0.243218    +9.312        +16.792
```

두 기간과 seed의 변동은 다음과 같다. 대괄호는 같은 LR로 선택된 두 seed 손실의 min/max이며 신뢰구간이 아니다.

```text
자료/경로                             A MSE [seed 범위]              B MSE [seed 범위]              combined MAE
robin/OUTPUT_AFFINE                   0.271344 [0.269134, 0.273555]  0.279675 [0.278901, 0.280448]  0.337708
robin/NATIVE_HEAD                     0.236412 [0.234390, 0.238433]  0.223023 [0.222650, 0.223397]  0.264501
peacock_education/OUTPUT_AFFINE       0.229758 [0.229198, 0.230317]  0.169125 [0.167859, 0.170391]  0.306845
peacock_education/NATIVE_HEAD         0.234681 [0.234351, 0.235011]  0.168007 [0.167995, 0.168020]  0.302940
jena/OUTPUT_AFFINE                    0.323612 [0.323487, 0.323737]  0.206547 [0.206540, 0.206553]  0.289956
jena/NATIVE_HEAD                      0.292623 [0.292353, 0.292894]  0.193813 [0.193144, 0.194481]  0.256867
```

[확인] OUTPUT의 combined MAE가 F0보다 높은 자료는 robin, peacock_education, jena다. MSE 감소를 MAE 개선으로 확대하지 않는다. Jena OUTPUT의 combined MAE는 0.289956, F0는 0.264758이다.

같은 원점·채널·seed를 보존한 paired 차이의 combined MSE 조건부 구간이다. 상대변화는 100×(후보−참조)/참조이며 음수는 후보 개선이다.

```text
자료/비교                                      상대변화%     paired block95 상대변화%
robin/OUTPUT_AFFINE__vs__F0                     +20.260       [+7.629, +36.881]
robin/NATIVE_HEAD__vs__F0                       +0.272       [-1.782, +1.843]
robin/NATIVE_HEAD__vs__OUTPUT_AFFINE            -16.621       [-27.168, -6.408]
robin/OUTPUT_AFFINE__vs__MSE_LORA               +26.082       [+8.521, +49.693]
robin/NATIVE_HEAD__vs__MSE_LORA                 +5.126       [-1.318, +12.104]
robin/OUTPUT_AFFINE__vs__LEVEL                  +0.115       [-10.923, +12.997]
robin/NATIVE_HEAD__vs__LEVEL                    -16.525       [-26.095, -8.687]
peacock_education/OUTPUT_AFFINE__vs__F0         -1.029       [-2.483, +1.708]
peacock_education/NATIVE_HEAD__vs__F0           -0.085       [-0.389, +0.807]
peacock_education/NATIVE_HEAD__vs__OUTPUT_AFFINE +0.955       [-1.711, +3.121]
peacock_education/OUTPUT_AFFINE__vs__MSE_LORA   +0.641       [-1.164, +3.288]
peacock_education/NATIVE_HEAD__vs__MSE_LORA     +1.602       [+0.475, +3.235]
peacock_education/OUTPUT_AFFINE__vs__LEVEL      -1.879       [-5.664, -0.173]
peacock_education/NATIVE_HEAD__vs__LEVEL        -0.942       [-6.691, +2.037]
jena/OUTPUT_AFFINE__vs__F0                      -9.312       [-11.612, -7.616]
jena/NATIVE_HEAD__vs__F0                        -16.792       [-20.259, -13.006]
jena/NATIVE_HEAD__vs__OUTPUT_AFFINE             -8.247       [-10.863, -4.951]
jena/OUTPUT_AFFINE__vs__MSE_LORA                +13.821       [+9.785, +17.123]
jena/NATIVE_HEAD__vs__MSE_LORA                  +4.434       [+2.065, +7.272]
jena/OUTPUT_AFFINE__vs__LEVEL                   -18.276       [-20.722, -15.706]
jena/NATIVE_HEAD__vs__LEVEL                     -25.016       [-27.952, -21.145]
```

bootstrap은 2,000회, seed9262026, Robin/Peacock block7·Jena block42, salt0/0/1이다. A/B를 나눈 noncircular contiguous moving blocks에서 ceil(n/block)개의 시작점을 추출하고 n까지 잘랐으며 combined는 두 기간의 occurrence weights를 더했다. 구간은 이미 선택된 두 seed와 노출된 개발 target에 조건부인 시간 재표집 불확실성이다. LR/모델선택 불확실성과 독립 도메인 일반화를 포함하지 않으며 0을 포함해도 동등성은 확인되지 않는다.

기존 저장 결과는 재적합·재선택 없이 참조했다. MSE-LoRA·LEVEL·NLinear·Mini/Tiny·Chronos-2의 같은 canonical target 정확도는 다음과 같다. 기존 LoRA와의 차이는 같은 HEAD 부모에서 내부 적응을 추가한 효과가 아니다.

```text
자료                참조                         combined MSE  combined MAE
robin               LEVEL                        0.275196      0.324593
robin               MSE_LORA                     0.218517      0.258718
robin               DIRECT_NLINEAR               0.327724      0.353410
robin               C2_MV                        0.230150      0.254024
robin               C2_SMALL_MV                  0.224650      0.259768
robin               C2_UNI                       0.232587      0.254876
robin               C2_SMALL_UNI                 0.235434      0.262740
robin               BOLT_MINI_F0                 0.227636      0.264707
robin               BOLT_TINY_F0                 0.243654      0.269697
robin               NATIVE_LORA_QUANTILE         0.223610      0.256347
peacock_education   LEVEL                        0.203266      0.315004
peacock_education   MSE_LORA                     0.198176      0.298839
peacock_education   DIRECT_NLINEAR               0.219884      0.329450
peacock_education   C2_MV                        0.199572      0.295145
peacock_education   C2_SMALL_MV                  0.204274      0.301992
peacock_education   C2_UNI                       0.199978      0.296214
peacock_education   C2_SMALL_UNI                 0.205577      0.303619
peacock_education   BOLT_MINI_F0                 0.202057      0.301707
peacock_education   BOLT_TINY_F0                 0.202745      0.304780
peacock_education   NATIVE_LORA_QUANTILE         0.200123      0.299731
jena                LEVEL                        0.324359      0.320450
jena                MSE_LORA                     0.232892      0.238943
jena                DIRECT_NLINEAR               0.273984      0.286714
jena                C2_MV                        0.253987      0.224457
jena                C2_SMALL_MV                  0.263155      0.230217
jena                C2_UNI                       0.268380      0.237023
jena                C2_SMALL_UNI                 0.271655      0.238883
jena                BOLT_MINI_F0                 0.302064      0.271424
jena                BOLT_TINY_F0                 0.321409      0.281006
jena                NATIVE_LORA_QUANTILE         0.240590      0.226584
```

[확인] 24 fits의 실제 scientific updates 합은 66,432이고, 12개 선택 모델을 TEST에 평가했다. epoch0을 포함한 full-VAL strict minimum과 earliest exact tie를 재검산했고 LR은 두 seed VAL 최소손실 평균으로 선택했다. LR exact tie는 first-listed1e-3이며 scheduler는 epoch1 이후에만 관측했다. 선택 seal은 첫 TEST 접근보다 앞서고 당시 신규 TEST counter는0이었다.

```text
자료/경로                             선택 LR    선택 epoch(seed1/2)  전체 fit epoch0선택/상한도달/OOM
robin/OUTPUT_AFFINE                   1e-03      4/4                2/0/0
robin/NATIVE_HEAD                     1e-04      3/1                2/0/0
peacock_education/OUTPUT_AFFINE       1e-03      4/5                0/0/0
peacock_education/NATIVE_HEAD         1e-04      10/0               3/0/0
jena/OUTPUT_AFFINE                    1e-03      27/20              0/0/0
jena/NATIVE_HEAD                      1e-03      39/39              0/0/0
```

[확인] 모든 fit의 TRAIN batch loss/gradient와 full VAL은 유한했고 frozen parameters 및 모든 buffers의 불변 기록과 현재 선택 checkpoint 해시가 일치했다. epoch0 선택은 7개, 최대epoch 종료는 0개, 기록된 OOM events는 0개다. sampled TRAIN loss가 처음보다 낮아졌으나 마지막 VAL이 선택 최소보다 높은 곡선은 22개다. 이 TRAIN 수치는 epoch별 표본 batch 평균이며 full canonical TRAIN 점수와 같지 않다. 정상 epoch0 선택·정상 악화·상한 종료는 기술 실패나 자원중단과 구분한다.

[확인] 적응비용과 배포비용을 분리했다. 다음 항목은 서로 중복 합산하지 않았고 모든 LR/seed 적합의 VAL·checkpoint I/O를 포함했다. cache 준비의 generation/write와 fit 내부 TRAIN/VAL/checkpoint 시간은 구성 설명용이며 해당 부모 elapsed에 다시 더하지 않았다.

```text
s1_feature_prepare_including_generation_write_restore_hash_seconds 256.363 s
all_24_fit_elapsed_including_train_val_checkpoint_replay_seconds 565.452 s
joint_lr_selection_seconds                                0.352 s
new_12_test_prediction_seconds                            34.493 s
matched_training_and_deployment_probe_elapsed_seconds     1440.327 s
remaining_cache_read_scoring_bootstrap_source_guard_io_overhead_seconds 25.453 s
total_s1_gpu_job_wall_seconds                             2322.440 s
generation_write_included_in_feature_prepare_seconds      244.015 s
fit_train_included_seconds                                473.200 s
fit_val_included_seconds                                  34.421 s
```

전체 S1 GPU job wall은 인터프리터 시작부터의 포괄 비용이다. 남은 차액에는 fit 바깥 cache reads, canonical 자료·참조 재검증, bootstrap/채점, source/checkpoint hash와 guard I/O가 들어 있으므로 모든 비용이 포함되며 특정 항목의 단가로 임의 배정하지 않았다. S0 native calibration과 계측 수정의 시간은 별도 과거 개발비다. 원래 느린 guard를 포함한20-update 측정과 수정 후6개 single-update 진단을 모두 보존했고 single-update를 steady-state20-update 측정으로 바꾸지 않았다. 이 과거 비용으로 S1 속도비를 만들지 않았다.

[확인] S0 native technical updates는 384개다. 별도 CPU 구현 fixture는 toy optimizer40 updates·GPU/모델/자료 로드0·wall 2.795초였고 실제 CPU process time은 미측정이다. 그 fixture를 native calibration이나 scientific update에 섞지 않았다.

[확인] 같은 회차 training probe는 선택 checkpoint의 폐기용 사본에서 B4, warmup10+measured10,3blocks를 측정했다. 108개 행·2,160 technical updates(신규1,440+기존720), Adam moments 생성 뒤 memory와 full optimizer step을 확인했고 선택 파일 pre/post/current hash는 같았다. 아래 시간은 seed별3block median의 평균이며 block range는 CI가 아니다.

```text
자료/경로/path                                  seconds/update
robin/OUTPUT_AFFINE/online                       0.034722
robin/OUTPUT_AFFINE/cached                       0.004160
robin/NATIVE_HEAD/online                         0.036115
robin/NATIVE_HEAD/cached                         0.008620
robin/MSE_LORA/online                            0.114023
robin/LEVEL/online                               0.036227
peacock_education/OUTPUT_AFFINE/online           0.034773
peacock_education/OUTPUT_AFFINE/cached           0.004196
peacock_education/NATIVE_HEAD/online             0.039722
peacock_education/NATIVE_HEAD/cached             0.009464
peacock_education/MSE_LORA/online                0.102846
peacock_education/LEVEL/online                   0.034614
jena/OUTPUT_AFFINE/online                        0.034831
jena/OUTPUT_AFFINE/cached                        0.004346
jena/NATIVE_HEAD/online                          0.036107
jena/NATIVE_HEAD/cached                          0.007530
jena/MSE_LORA/online                             0.104629
jena/LEVEL/online                                0.035153
```

Adam 상태와 peak memory는 같은 회차 실제 행에서 경로별 최댓값을 별도로 보존했다. 아래 allocated/reserved는 Adam warmup 이후 peak이며 weights/gradient/optimizer의 계산상 payload와 다르다.

```text
자료/경로/path                                  trainable   Adam bytes   peak allocated/reserved MiB
robin/OUTPUT_AFFINE/online                             306         2456   274.37/292.00
robin/OUTPUT_AFFINE/cached                             306         2456   200.38/214.00
robin/NATIVE_HEAD/online                           2526336     20210712   294.16/330.00
robin/NATIVE_HEAD/cached                           2526336     20210712   239.74/258.00
robin/MSE_LORA/online                               294912      2359584   567.89/610.00
robin/LEVEL/online                                   17920       143368   223.90/236.00
peacock_education/OUTPUT_AFFINE/online                 182         1464   255.68/406.00
peacock_education/OUTPUT_AFFINE/cached                 182         1464   199.36/406.00
peacock_education/NATIVE_HEAD/online               2526336     20210712   275.58/406.00
peacock_education/NATIVE_HEAD/cached               2526336     20210712   238.46/406.00
peacock_education/MSE_LORA/online                   294912      2359584   486.63/512.00
peacock_education/LEVEL/online                       17920       143368   218.45/238.00
jena/OUTPUT_AFFINE/online                              462         3704   291.10/306.00
jena/OUTPUT_AFFINE/cached                              462         3704   200.40/214.00
jena/NATIVE_HEAD/online                            2526336     20210712   310.02/348.00
jena/NATIVE_HEAD/cached                            2526336     20210712   239.78/258.00
jena/MSE_LORA/online                                294912      2359584   656.88/720.00
jena/LEVEL/online                                    17920       143368   227.47/238.00
```

배포는 cache shortcut 없이 CPU FP32 context→전체 backbone/readout→contiguous CPU[48,C]였다. B1/B4·VAL 첫24원점·warmup10·24원점 pass10·회전3blocks의162행과18쌍 F0 전후 sentinel을 유지했다.

```text
자료/경로/B                                     seconds/origin  origins/s
robin/F0/B1                                      0.027382        36.52
robin/F0/B4                                      0.007361        135.86
robin/OUTPUT_AFFINE/B1                           0.029528        33.87
robin/OUTPUT_AFFINE/B4                           0.007474        133.80
robin/NATIVE_HEAD/B1                             0.029242        34.20
robin/NATIVE_HEAD/B4                             0.007512        133.12
robin/MSE_LORA/B1                                0.027378        36.53
robin/MSE_LORA/B4                                0.007067        141.51
robin/LEVEL/B1                                   0.027933        35.80
robin/LEVEL/B4                                   0.007055        141.75
peacock_education/F0/B1                          0.027315        36.61
peacock_education/F0/B4                          0.006894        145.06
peacock_education/OUTPUT_AFFINE/B1               0.029452        33.95
peacock_education/OUTPUT_AFFINE/B4               0.007250        137.92
peacock_education/NATIVE_HEAD/B1                 0.029056        34.42
peacock_education/NATIVE_HEAD/B4                 0.007364        135.79
peacock_education/MSE_LORA/B1                    0.027343        36.57
peacock_education/MSE_LORA/B4                    0.007042        142.01
peacock_education/LEVEL/B1                       0.027436        36.45
peacock_education/LEVEL/B4                       0.006922        144.47
jena/F0/B1                                       0.026829        37.27
jena/F0/B4                                       0.007102        140.80
jena/OUTPUT_AFFINE/B1                            0.029225        34.22
jena/OUTPUT_AFFINE/B4                            0.007578        131.97
jena/NATIVE_HEAD/B1                              0.028846        34.67
jena/NATIVE_HEAD/B4                              0.007594        131.69
jena/MSE_LORA/B1                                 0.027437        36.45
jena/MSE_LORA/B4                                 0.007219        138.53
jena/LEVEL/B1                                    0.027254        36.69
jena/LEVEL/B4                                    0.007061        141.62
```

F0 sentinel after/before latency ratio 범위는 0.935–1.021다. 사전 drift rejection threshold는 없었으므로 원값을 보존했다. frozen prefix는 cached 적합과 배포에서 GPU에 남았고 특징 캐시는 반복 적응의 비용을 바꾸지만 전체 모델 배포의 압축을 뜻하지 않는다. NLinear/Mini/Tiny/Chronos-2의 과거 비용은 이 회차 배포비 우위에 사용하지 않았다.

[확인] 다음 사용량이 사전 S1 상한 이내였다. 보고 CPU 작업은 중앙 guard에 합산되며 값의 측정 시각은 verification.json에 남겼다.

```text
gpu_job_wall_seconds                        2322.440 /      28800.000 (8.06%)
cpu_only_job_wall_seconds                      1.903 /       1800.000 (0.11%)
cpu_process_seconds                         2093.172 /     117000.000 (1.79%)
operation_elapsed_seconds                   2378.779 /      32400.000 (7.34%)
new_disk_bytes                        1953142481.000 / 3107102908.000 (62.86%)
process_ram_peak_bytes                2843209728.000 / 8589934592.000 (33.10%)
gpu_allocated_peak_bytes               688788992.000 / 8589934592.000 (8.02%)
gpu_reserved_peak_bytes                754974720.000 / 8589934592.000 (8.79%)
technical_updates                           2160.000 /       2160.000 (100.00%)
prefix_calls                               41319.000 /      60000.000 (68.86%)
head_calls                                183803.000 /    1800000.000 (10.21%)
scientific_fits                               24.000 /         24.000 (100.00%)
scientific_updates                         66432.000 /     368640.000 (18.02%)
new_test_predictions                          12.000 /         12.000 (100.00%)
```

가장 강한 반론은 같은 recipe가 native HEAD의 충분한 최적화를 보장하지 않는다는 점이다. OUTPUT의 채널 혼합과 다른 용량도 성능 차이를 설명할 수 있다. 세 자료의 겹치는 시계열 원점을 독립 도메인 수백 개로 세지 않으며 이미 노출된 TEST와 서로 다른 부모·탐색 기회의 기존 LoRA만으로 내부 표현 부족을 증명하지 않는다. Time-PEFT/TRACE의 HeadOnly 및 head+LoRA, AdaPTS decoder/head-first, LogME 전이 지표와 짧은 pilot 탐색이 강한 기존 대안이다. 같은 HEAD 부모 이후 이득과 총 의사결정비용을 줄이는 진단의 차이가 확인되어야 하며 신규성은 미확인이다.

원문·공개 코드의 실제 열람 범위와 미확인 차이는 [선행 중복 지도](prior_art_map.md)에 기록했다.

S2 권고는 **PROPOSE**다. 아래 historical gap은 100×(HEAD−기존 MSE-LoRA)/기존 MSE-LoRA로 양수이면 기존 LoRA 참조가 낮다. 이 권고를 구체화한 시점은 TEST를 본 뒤이므로 사후 탐색적 개발 결정이다. 주 근거는 봉인된 HEAD VAL 변화·검증된 학습 상태와 이번 회차 실제 시간·Adam 메모리다. historical TEST gap은 보조 참조이며 자료 선택 기준이나 matched-parent 추가이득의 측정값은 아니다. 세 자료를 모두 유지한다.

```text
자료                초기 HEAD VAL  선택 HEAD VAL  VAL 감소%  선택 epoch(seed1/2)
robin               0.408587       0.406306       +0.558     3/1
peacock_education   0.153833       0.153523       +0.202     10/0
jena                0.555062       0.423741       +23.659     39/39
```

```text
자료                historical LoRA gap%  HEAD online s/update  LoRA online s/update  joint memory별도peak합 GiB
robin               +5.126                 0.037272              0.147638               0.918
peacock_education   +1.602                 0.044969              0.113093               0.896
jena                +4.434                 0.036871              0.118484               1.043
```

[추정] 제안 범위는 기존 selected HEAD의 dataset/seed별 동일 checkpoint를 부모로 HEAD_CONTINUE와 HEAD_PLUS_LORA를 모든3자료·2seed에서 비교하는12 fits다. 각 자료의 selected HEAD LR 하나를 두 경로에 같이 쓰고 새 LR 선택은 하지 않는다. LoRA 초기수정0·동일 head 초기값·optimizer 재시작 규칙·sample suffix·같은 추가 update checkpoints를 사전 고정한다. 같은 update 수와 같은 시간의 효과는 다르다. 두 LR를 검색하려면24 fits 및 별도 두 배 update 예산으로 승인 범위를 바꿔야 한다.

[추정] 추가 최대120epochs×512origins/B4는 fit당15,360updates,12 fits 합184,320updates다. 이번 회차 online 최느린 block 단가에서 HEAD+LoRA를 HEAD와 LoRA의 합으로 근사하면 update 시간 18967.8초, 매epoch fresh full-VAL(초기 포함121회) 2411.1초, raw 합 21378.9초다. 단순2배 여유는 42757.7초이며 실행 승인 cap이 아니다. HEAD+LoRA는 prefix가 달라지므로 S1 특징 cache를 사용할 수 없다. 합산 단가는 실제 joint backward 측정이나 엄밀한 상한이 아니고 restore/checkpoint/guard I/O·TEST·배포·보고 비용을 초기 probe 뒤 따로 확정해야 한다.

[추정] joint trainables는2,821,248개, FP32 weights+grad+Adam m/v raw payload는 45,139,968bytes다. 별도 HEAD/LoRA Adam 이후 reserved peak 합은 위 표의 참고값이며 joint activation/allocator peak는 미측정이다. 첫 별도 probe 제안은12개 폐기용 부모×20updates+integrity reserve24=264technical updates, GPU wall cap 후보 900초와 process RAM/GPU allocated·reserved 각각8GiB다. joint actual memory·native head trainable 유지·초기함수 parity를 먼저 측정하여 본실험 견적을 다시 봉인해야 한다. S2는 별도 승인 전 실행하지 않으며 S3/S4 선택기, 새 PEFT, PCA/G 유지·압축·신규성 주장은 보류한다. 유지하는 것은 matched-parent 질문, 축소하는 것은 현재3자료에서의 관찰을 보편적 결론으로 확대하는 주장이다.

[확인] 기존 산출물76개를 현재 해시와 대조해 변경0개를 확인했다. PPT/PDF와 무관한 기존 연구 산출물을 포함하는 실제 보호 확인은 [preservation_check.json](preservation_check.json)에 기록되어 있다. 그 확인 시각·대상·해시를 별도로 참조하며 S1 성능 검증과 섞지 않는다.

검산 상세와 기계 판독 요약은 [verification.json](verification.json), 모든 기간·MAE·채널 수치와 source receipt는 [evaluation.json](evaluation.json), raw 비용·memory·drift는 [costrows.json](costrows.json)에 있다. 이 보고는 commit/push나 live remote SHA를 확인하지 않으며 게시 상태는 별도 절차로 보고한다.
