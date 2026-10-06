> 공개 사본: 개인 경로·채팅 식별자와 링크를 정리했습니다. 원본 및 사본 해시는 감사 공개 폴더의 `publication_manifest.json`에 분리 기록합니다. 아래 상태·검산은 원래 감사/실행 시점의 기록이며, 이번 게시 검증이 아닙니다.

# LEVEL와 Chronos-2 네이티브 다변량 대조

[확인] 이번 회차는 이미 노출된 Robin·Peacock Education·Jena에 고정 zero-shot 기준선 4개를 추가한 후속 비교입니다. 추가 적합·optimizer update·PCA 재적합·기존 모델 재선택은 0회입니다.

## 결과에 따른 주장

**Robin**

[확인] Chronos-2 MV의 MSE가 더 낮습니다. LEVEL의 채택 이유는 실제로 절약된 비용 축에 한정하며 정확도 손해를 함께 제시해야 합니다.

LEVEL의 Chronos-2 MV 대비 MSE 상대 차이는 19.57%, 조건부 95% 구간은 [3.18, 40.50]%입니다. 비용은 새 모델/LEVEL−1 기준으로 peak allocated 161.18%, batch4 처리량 -57.40%입니다.

[확인] Chronos-2-Small MV가 LEVEL보다 낮은 MSE·낮은 peak allocated·높은 batch4 처리량을 보였습니다. 이 측정 조건에서 LEVEL의 실용적 우위 주장을 축소해야 합니다.

LEVEL의 Chronos-2-Small MV 대비 MSE 상대 차이는 22.50%, 조건부 95% 구간은 [9.74, 43.55]%입니다. 비용은 새 모델/LEVEL−1 기준으로 peak allocated -15.43%, batch4 처리량 23.05%입니다.

**Peacock Education**

[확인] Chronos-2 MV의 MSE가 더 낮습니다. LEVEL의 채택 이유는 실제로 절약된 비용 축에 한정하며 정확도 손해를 함께 제시해야 합니다. 대응 MSE 구간이 0을 포함하여 정확도 차이에도 불확실성이 남습니다.

LEVEL의 Chronos-2 MV 대비 MSE 상대 차이는 1.85%, 조건부 95% 구간은 [-1.19, 6.41]%입니다. 비용은 새 모델/LEVEL−1 기준으로 peak allocated 156.06%, batch4 처리량 -47.18%입니다.

[확인] LEVEL의 관측 MSE 점추정이 Chronos-2-Small MV보다 낮습니다. 해당 자료·고정 조건의 결과이며 MAE와 비용의 방향은 별도로 제시합니다. 대응 MSE 구간이 0을 포함하여 정확도 차이에도 불확실성이 남습니다. MAE는 LEVEL 0.315004 대 Chronos-2-Small MV 0.301992로 LEVEL이 더 높으며, 상대 손해는 4.31%, 조건부 95% 구간 [2.34, 6.80]%입니다.

LEVEL의 Chronos-2-Small MV 대비 MSE 상대 차이는 -0.49%, 조건부 95% 구간은 [-3.31, 5.54]%입니다. 비용은 새 모델/LEVEL−1 기준으로 peak allocated -21.69%, batch4 처리량 21.07%입니다.

**Jena**

[확인] Chronos-2 MV의 MSE가 더 낮습니다. LEVEL의 채택 이유는 실제로 절약된 비용 축에 한정하며 정확도 손해를 함께 제시해야 합니다.

LEVEL의 Chronos-2 MV 대비 MSE 상대 차이는 27.71%, 조건부 95% 구간은 [19.89, 33.78]%입니다. 비용은 새 모델/LEVEL−1 기준으로 peak allocated 167.97%, batch4 처리량 -59.67%입니다.

[확인] Chronos-2-Small MV가 LEVEL보다 낮은 MSE·낮은 peak allocated·높은 batch4 처리량을 보였습니다. 이 측정 조건에서 LEVEL의 실용적 우위 주장을 축소해야 합니다. 메모리 이점은 peak allocated 축이며 모든 메모리 축의 지배가 아닙니다: B4 allocated는 Chronos-2-Small MV 195.612 대 LEVEL 217.904 MiB지만, reserved는 252.000 대 238.000 MiB로 새 모델이 더 높습니다.

LEVEL의 Chronos-2-Small MV 대비 MSE 상대 차이는 23.26%, 조건부 95% 구간은 [15.63, 30.56]%입니다. 비용은 새 모델/LEVEL−1 기준으로 peak allocated -10.23%, batch4 처리량 17.31%입니다.

## 지표별 절충과 주장 경계

[확인] Peacock Education에서 LEVEL의 MSE 0.203266는 Chronos-2-Small MV 0.204274보다 점추정상 낮습니다. 상대 차이는 -0.49%, 조건부 95% 구간 [-3.31, 5.54]%로 0을 포함합니다. 이를 안정적인 MSE 우위나 동등성으로 해석하지 않습니다.

MAE는 LEVEL 0.315004 대 Small MV 0.301992로 반대 방향이며 LEVEL의 상대 손해 4.31%, 조건부 95% 구간 [2.34, 6.80]%입니다. 낮은 MSE 점추정만으로 전체 정확도 우위를 주장하지 않습니다.

[확인] Jena에서 MSE-LoRA의 MSE 0.232892가 Chronos-2 MV 0.253987보다 낮지만, MAE는 Chronos-2 MV 0.224457가 MSE-LoRA 0.238943보다 낮습니다. 새 모델/MSE-LoRA−1 기준 MSE 9.06% [조건부 95%: 6.29, 13.13]%, MAE -6.06% [조건부 95%: -7.20, -4.73]%입니다. 목적별 절충이며 한 모델의 모든 정확도 지표 승리로 합치지 않습니다.

[확인] Jena에서 MSE-LoRA의 MSE 0.232892가 Chronos-2-Small MV 0.263155보다 낮지만, MAE는 Chronos-2-Small MV 0.230217가 MSE-LoRA 0.238943보다 낮습니다. 새 모델/MSE-LoRA−1 기준 MSE 12.99% [조건부 95%: 9.48, 17.57]%, MAE -3.65% [조건부 95%: -5.06, -2.17]%입니다. 목적별 절충이며 한 모델의 모든 정확도 지표 승리로 합치지 않습니다.

상세 원수치·기간·seed·주장 경계는 [ACCURACY_INTERPRETATION.md](ACCURACY_INTERPRETATION.md)에 연결됩니다.

## 정확도 원수치

```text
Method                Robin MSE  Robin MAE  Peacock MSE  Peacock MAE  Jena MSE  Jena MAE
b (compression only)  0.977023   0.661870   0.298832     0.421165     0.381187  0.381092
LEVEL                 0.275196   0.324593   0.203266     0.315004     0.324359  0.320450
Bolt F0               0.229095   0.262074   0.201521     0.300358     0.292300  0.264758
MSE-LoRA              0.218517   0.258718   0.198176     0.298839     0.232892  0.238943
Direct NLinear        0.327724   0.353410   0.219884     0.329450     0.273984  0.286714
Chronos-2 MV          0.230150   0.254024   0.199572     0.295145     0.253987  0.224457
Chronos-2-Small MV    0.224650   0.259768   0.204274     0.301992     0.263155  0.230217
Chronos-2 UNI         0.232587   0.254876   0.199978     0.296214     0.268380  0.237023
Chronos-2-Small UNI   0.235434   0.262740   0.205577     0.303619     0.271655  0.238883
```

모든 값은 TRAIN 표준화 좌표의 관측 target에 대한 채널 동일 가중 오차입니다. 기간별 채널 오차합/관측수를 먼저 합산하고 채널 평균을 낸 뒤 선택 seed 손실을 평균했습니다. 기존 학습형 방법은 2 seed, b/F0와 새 zero-shot은 해당 고정 출력 1개이며 예측 ensemble을 만들지 않았습니다.

### TEST-A/B도 모두 보존

**test_a**

```text
Method                Robin MSE  Robin MAE  Peacock MSE  Peacock MAE  Jena MSE  Jena MAE
b (compression only)  1.123537   0.692906   0.322713     0.435271     0.486863  0.440826
LEVEL                 0.273677   0.331865   0.234300     0.339193     0.422475  0.381041
Bolt F0               0.231539   0.283065   0.235011     0.324832     0.370756  0.313130
MSE-LoRA              0.229946   0.283501   0.230911     0.323183     0.278738  0.267552
Direct NLinear        0.327484   0.367210   0.253442     0.355312     0.332070  0.322967
Chronos-2 MV          0.221426   0.273224   0.232664     0.319034     0.303096  0.252617
Chronos-2-Small MV    0.241441   0.288877   0.235122     0.324298     0.323248  0.263088
Chronos-2 UNI         0.214492   0.269092   0.232718     0.320866     0.324581  0.269325
Chronos-2-Small UNI   0.242116   0.286653   0.237789     0.327581     0.330446  0.272300
```

**test_b**

```text
Method                Robin MSE  Robin MAE  Peacock MSE  Peacock MAE  Jena MSE  Jena MAE
b (compression only)  0.830506   0.630832   0.274944     0.407054     0.275510  0.321358
LEVEL                 0.276712   0.317319   0.172223     0.290810     0.226243  0.259859
Bolt F0               0.226649   0.241079   0.168020     0.275875     0.213844  0.216386
MSE-LoRA              0.207085   0.233932   0.165429     0.274487     0.187045  0.210335
Direct NLinear        0.327962   0.339608   0.186316     0.303581     0.215899  0.250461
Chronos-2 MV          0.238871   0.234821   0.166467     0.271248     0.204877  0.196296
Chronos-2-Small MV    0.207855   0.230654   0.173416     0.279680     0.203062  0.197346
Chronos-2 UNI         0.250679   0.240656   0.167227     0.271554     0.212178  0.204721
Chronos-2-Small UNI   0.228748   0.238824   0.173353     0.279649     0.212864  0.205465
```

관측 분모·개별 seed·채널·원점별 표는 accuracy_comparison.csv, accuracy_channels.csv, accuracy_origins.csv에 연결됩니다. 다른 자료의 MSE를 평균하여 종합 점수를 만들지 않았습니다.

## 원점 격리와 다변량 정보 공유

[확인] MV의 각 원점 C채널은 하나의 그룹이고 서로 다른 원점은 분리되었습니다. UNI는 원점/채널마다 별도 그룹입니다. 실제 모델 직전 group IDs와 group-attention mask, 배치·순서·다른 원점 변경·미래 배열 차단·동일 경로 복원 검사는 preflight_checks.json에 보존했습니다. 예측은 실제 0.5 분위수이며 입력과 같은 TRAIN 표준화 좌표로 반환됩니다.

```text
Dataset            Model MV            MV MSE    UNI MSE   MV/UNI-1 %  Conditional CI95 %
Robin              Chronos-2 MV        0.230150  0.232587  -1.05       [-5.44, 3.08]
Robin              Chronos-2-Small MV  0.224650  0.235434  -4.58       [-7.42, -0.72]
Peacock Education  Chronos-2 MV        0.199572  0.199978  -0.20       [-3.72, 3.68]
Peacock Education  Chronos-2-Small MV  0.204274  0.205577  -0.63       [-2.28, 0.61]
Jena               Chronos-2 MV        0.253987  0.268380  -5.36       [-7.56, -3.62]
Jena               Chronos-2-Small MV  0.263155  0.271655  -3.13       [-6.02, -0.21]
```

음수는 MV에 유리합니다. 차이가 없거나 불리한 기간도 paired_comparisons.csv에 포함됩니다. 같은 체크포인트의 관측 채널 공유 차이이며 전체 학습법·사전학습 지식의 순수 인과효과가 아닙니다. 구간이 0을 포함해도 동등성·비열등성을 선언하지 않습니다.

## 같은 회차 추론 비용

같은 RTX 4070·FP32·TF32 off·CPU 4 threads에서 첫 시간순 VAL 24원점을 사용했습니다. 이번 요청의 조건은 형상별 예열 10회, 전체 24원점 처리 10 passes, 순서를 바꾼 3개 블록입니다. 표준화 CPU 입력→전치/그룹·전송→온라인 전체 forward→원채널 H48 CPU 반환을 측정했습니다. 모델 로드·디스크·채점·hash·원장은 측정 밖이며 작업 시간에는 계상됩니다. B4 처리량의 단위는 원채널 전체 H48을 마친 원점/초입니다.

**Robin — B4 전체·chunk·CPU 대안**

```text
Method              Device  Execution   Rows M  MSE       ms/origin  origins/s  Peak alloc MiB  Peak reserv MiB  Block origins/s range
LEVEL               cuda    full        20      0.275196  6.694      149.515    214.489         216.000          132.678..153.817
Bolt F0             cuda    full        68      0.229095  6.970      143.464    265.489         292.000          128.875..147.716
Bolt F0             cuda    chunk_4k    20      0.229095  25.762     38.822     212.654         226.000          34.559..39.165
Bolt F0             cuda    chunk_k     5       0.229095  90.742     11.021     197.627         220.000          10.126..11.132
MSE-LoRA            cuda    full        68      0.218517  6.949      144.126    265.091         294.000          131.667..152.318
MSE-LoRA            cuda    chunk_4k    20      0.218517  25.936     38.558     214.334         246.000          16.905..39.462
MSE-LoRA            cuda    chunk_k     5       0.218517  89.500     11.175     196.627         222.000          8.832..11.290
Direct NLinear      cuda    direct      full    0.327724  0.150      7089.848   9.498           22.000           5249.199..10937.186
Chronos-2 MV        cuda    full_group  full    0.230150  15.701     63.689     560.193         636.000          63.530..64.499
Chronos-2-Small MV  cuda    full_group  full    0.224650  5.436      183.975    181.385         204.000          181.650..185.236
Direct NLinear      cpu     direct      full    0.327724  0.044      22790.679  N/A             N/A              20844.461..43506.583
```

**Robin — B1 지연**

```text
Method              Device  Execution   Rows M  MSE       ms/origin  origins/s  Peak alloc MiB  Peak reserv MiB  Block origins/s range
LEVEL               cuda    full        5       0.275196  26.898     37.193     196.628         220.000          34.494..38.579
Bolt F0             cuda    full        17      0.229095  26.421     37.849     209.871         216.000          34.235..38.330
Bolt F0             cuda    chunk_k     5       0.229095  101.588    9.844      197.518         220.000          8.883..9.920
MSE-LoRA            cuda    full        17      0.218517  26.757     37.374     210.959         218.000          29.229..38.682
MSE-LoRA            cuda    chunk_k     5       0.218517  102.121    9.793      196.518         222.000          7.849..9.864
Direct NLinear      cuda    direct      full    0.327724  0.468      2145.247   9.290           22.000           1834.054..2953.262
Chronos-2 MV        cuda    full_group  full    0.230150  38.040     26.288     489.610         502.000          26.151..26.448
Chronos-2-Small MV  cuda    full_group  full    0.224650  21.423     46.679     132.327         160.000          45.783..47.059
Direct NLinear      cpu     direct      full    0.327724  0.122      8199.779   N/A             N/A              7869.925..18696.438
```

**Peacock Education — B4 전체·chunk·CPU 대안**

```text
Method              Device  Execution   Rows M  MSE       ms/origin  origins/s  Peak alloc MiB  Peak reserv MiB  Block origins/s range
LEVEL               cuda    full        16      0.203266  6.783      147.439    210.021         218.000          140.858..157.173
Bolt F0             cuda    full        52      0.201521  6.745      148.257    248.557         266.000          140.973..154.385
Bolt F0             cuda    chunk_4k    16      0.201521  25.813     38.740     209.996         220.000          37.778..39.174
Bolt F0             cuda    chunk_k     4       0.201521  83.423     11.988     195.633         220.000          11.628..12.055
MSE-LoRA            cuda    full        52      0.198176  6.658      150.204    248.557         266.000          146.917..155.082
MSE-LoRA            cuda    chunk_4k    16      0.198176  25.426     39.331     209.996         220.000          38.284..39.951
MSE-LoRA            cuda    chunk_k     4       0.198176  83.833     11.931     195.633         220.000          11.304..12.100
Direct NLinear      cuda    direct      full    0.219884  0.178      6074.365   8.433           22.000           2943.991..10711.827
Chronos-2 MV        cuda    full_group  full    0.199572  12.841     77.877     537.784         586.000          75.111..79.502
Chronos-2-Small MV  cuda    full_group  full    0.204274  5.604      178.506    164.476         194.000          163.273..180.108
Direct NLinear      cpu     direct      full    0.219884  0.055      18361.813  N/A             N/A              17657.909..27682.476
```

**Peacock Education — B1 지연**

```text
Method              Device  Execution   Rows M  MSE       ms/origin  origins/s  Peak alloc MiB  Peak reserv MiB  Block origins/s range
LEVEL               cuda    full        4       0.203266  26.450     37.808     195.521         220.000          37.543..38.242
Bolt F0             cuda    full        13      0.201521  26.358     37.940     205.025         230.000          37.772..39.216
Bolt F0             cuda    chunk_k     4       0.201521  102.702    9.737      195.446         220.000          8.801..9.867
MSE-LoRA            cuda    full        13      0.198176  26.333     37.982     205.025         232.000          36.113..38.567
MSE-LoRA            cuda    chunk_k     4       0.198176  101.530    9.849      195.446         220.000          9.653..9.919
Direct NLinear      cuda    direct      full    0.219884  0.456      2223.447   8.273           22.000           1689.220..2574.983
Chronos-2 MV        cuda    full_group  full    0.199572  38.329     26.090     484.482         502.000          25.969..26.458
Chronos-2-Small MV  cuda    full_group  full    0.204274  22.397     44.649     127.822         170.000          43.927..45.110
Direct NLinear      cpu     direct      full    0.219884  0.136      7363.747   N/A             N/A              5904.624..16068.572
```

**Jena — B4 전체·chunk·CPU 대안**

```text
Method              Device  Execution   Rows M  MSE       ms/origin  origins/s  Peak alloc MiB  Peak reserv MiB  Block origins/s range
LEVEL               cuda    full        24      0.324359  6.943      144.139    217.904         238.000          133.655..152.857
Bolt F0             cuda    full        84      0.292300  7.141      140.038    281.436         304.000          137.642..142.282
Bolt F0             cuda    chunk_4k    24      0.292300  25.914     38.589     216.882         252.000          38.421..39.041
Bolt F0             cuda    chunk_k     6       0.292300  87.683     11.405     198.480         222.000          11.358..11.463
MSE-LoRA            cuda    full        84      0.232892  7.149      139.884    281.780         306.000          136.637..142.720
MSE-LoRA            cuda    chunk_4k    24      0.232892  26.132     38.269     218.180         252.000          37.532..38.731
MSE-LoRA            cuda    chunk_k     6       0.232892  87.854     11.383     197.730         224.000          11.199..11.523
Direct NLinear      cuda    direct      full    0.273984  0.128      8213.738   9.563           22.000           5874.340..14582.536
Chronos-2 MV        cuda    full_group  full    0.253987  17.201     58.136     583.921         674.000          57.711..58.569
Chronos-2-Small MV  cuda    full_group  full    0.263155  5.914      169.085    195.612         252.000          166.254..170.764
Direct NLinear      cpu     direct      full    0.273984  0.046      21881.491  N/A             N/A              19459.754..22791.326
```

**Jena — B1 지연**

```text
Method              Device  Execution   Rows M  MSE       ms/origin  origins/s  Peak alloc MiB  Peak reserv MiB  Block origins/s range
LEVEL               cuda    full        6       0.324359  25.941     38.556     197.710         222.000          37.832..39.188
Bolt F0             cuda    full        21      0.292300  25.811     38.744     214.457         236.000          38.234..39.277
Bolt F0             cuda    chunk_k     6       0.292300  100.150    9.985      198.341         220.000          9.722..10.000
MSE-LoRA            cuda    full        21      0.232892  26.164     38.222     214.315         238.000          37.293..38.776
MSE-LoRA            cuda    chunk_k     6       0.232892  100.464    9.954      197.591         222.000          9.576..10.045
Direct NLinear      cuda    direct      full    0.273984  0.441      2279.625   9.306           22.000           1909.982..3278.279
Chronos-2 MV        cuda    full_group  full    0.253987  38.744     25.810     494.201         522.000          25.564..26.203
Chronos-2-Small MV  cuda    full_group  full    0.263155  22.202     45.042     136.250         162.000          40.412..45.857
Direct NLinear      cpu     direct      full    0.273984  0.129      7754.942   N/A             N/A              7258.741..8309.576
```

집계는 instance별 3개 블록 중앙값의 중앙값 → 선택 seed 평균입니다. allocated와 reserved, 모델 로드 뒤 resident allocation, 파라미터/버퍼 bytes, 프로세스 CPU 메모리는 서로 다른 양입니다. CPU의 GPU 메모리는 N/A이며 0으로 바꾸지 않았습니다. 전체 60개 집계와 315개 원블록·sentinel·full/chunk는 cost_comparison.csv에 남겼습니다.

### 모든 블록과 환경 drift

```text
Dataset            Block  Batch  F0 pre ms  F0 post ms  post/pre-1 %
Robin              0      1      29.691     26.395      -11.10
Robin              0      4      7.650      6.776       -11.43
Robin              1      1      25.766     26.174      1.58
Robin              1      4      7.061      6.795       -3.78
Robin              2      1      27.045     25.509      -5.68
Robin              2      4      6.759      6.670       -1.32
Peacock Education  0      1      26.178     26.397      0.84
Peacock Education  0      4      6.723      7.167       6.61
Peacock Education  1      1      26.975     25.799      -4.36
Peacock Education  1      4      6.645      6.923       4.19
Peacock Education  2      1      26.887     25.757      -4.20
Peacock Education  2      4      6.433      6.265       -2.61
Jena               0      1      25.692     27.666      7.68
Jena               0      4      7.334      6.944       -5.32
Jena               1      1      25.331     26.420      4.30
Jena               1      4      7.088      7.173       1.20
Jena               2      1      25.988     26.706      2.76
Jena               2      4      6.876      6.813       -0.92
```

모든 앞뒤 sentinel 값을 보존했습니다. 블록 범위는 신뢰구간이 아니며 범위가 겹치는 작은 처리량 차이를 안정적인 순위로 해석하지 않습니다. drift의 원인을 온도·Windows·clock 중 하나로 확정하지 않았습니다. 과거 v4/v12 timing과 이번 값을 나누지 않았습니다.

### 실제 배포 상태

```text
Dataset            Method              Deploy params  Param bytes  Buffer bytes  Total tensor bytes  Previously fitted params  Inference trainable params
Robin              LEVEL               47736106       190944424    36            190944460           17920                     0
Robin              Bolt F0             47718016       190872064    36            190872100           0                         0
Robin              MSE-LoRA            47718016       190872064    376           190872440           294912                    0
Robin              Direct NLinear      24624          98496        340           98836               24624                     0
Robin              Chronos-2 MV        119477664      477910656    3156          477913812           0                         0
Robin              Chronos-2-Small MV  27934624       111738496    1588          111740084           0                         0
Peacock Education  LEVEL               47736040       190944160    244           190944404           17920                     0
Peacock Education  Bolt F0             47718016       190872064    244           190872308           0                         0
Peacock Education  MSE-LoRA            47718016       190872064    244           190872308           294912                    0
Peacock Education  Direct NLinear      24624          98496        208           98704               24624                     0
Peacock Education  Chronos-2 MV        119477664      477910656    3156          477913812           0                         0
Peacock Education  Chronos-2-Small MV  27934624       111738496    1588          111740084           0                         0
Jena               LEVEL               47736188       190944752    36            190944788           17920                     0
Jena               Bolt F0             47718016       190872064    36            190872100           0                         0
Jena               MSE-LoRA            47718016       190872064    540           190872604           294912                    0
Jena               Direct NLinear      24624          98496        504           99000               24624                     0
Jena               Chronos-2 MV        119477664      477910656    3156          477913812           0                         0
Jena               Chronos-2-Small MV  27934624       111738496    1588          111740084           0                         0
```

C2의 이번 과제 추가 적합은 0이지만 사전학습 비용·전체 모델 메모리가 0이라는 뜻은 아닙니다. 원래 모델의 학습 메모리를 이번 추론 결과에서 만들지 않았습니다. 모든 모델은 배포 시 동결되어 optimizer가 없습니다.

## 선택 가치와 발표 수정

[확인] Robin의 고정 grid에서 MSE·allocated·B4 처리량 점추정의 비지배 실행은 MSE-LoRA:full/M=68; MSE-LoRA:chunk_4k/M=20; MSE-LoRA:chunk_k/M=5; Direct NLinear:direct/M=None; Chronos-2-Small MV:full_group/M=None입니다.

[확인] Peacock Education의 고정 grid에서 MSE·allocated·B4 처리량 점추정의 비지배 실행은 LEVEL:full/M=16; Bolt F0:chunk_k/M=4; MSE-LoRA:full/M=52; MSE-LoRA:chunk_4k/M=16; MSE-LoRA:chunk_k/M=4; Direct NLinear:direct/M=None; Chronos-2-Small MV:full_group/M=None입니다.

[확인] Jena의 고정 grid에서 MSE·allocated·B4 처리량 점추정의 비지배 실행은 MSE-LoRA:full/M=84; MSE-LoRA:chunk_4k/M=24; MSE-LoRA:chunk_k/M=6; Direct NLinear:direct/M=None; Chronos-2-Small MV:full_group/M=None입니다.

이는 단일 고정 K와 실행 grid의 기술적 점추정입니다. 모델 선택 불확실성·측정 변동을 포함한 보편적 전선, 모든 K 탐색, 배포 추천이 아닙니다. 임의의 정확도 허용폭·가중 효율 점수는 도입하지 않았습니다.

### “F0보다 못한데 왜 하나?”

Robin에서 LEVEL의 F0 대비 관측 MSE 차이는 20.12%입니다. 압축만 한 b를 보완한 효과와 F0를 선택할 이유는 별개입니다. 위 full/chunk 비용을 함께 보고, 실제 절약된 축이 없거나 단순 실행 대안에 지배되면 LEVEL의 실용 주장을 축소합니다.

Peacock Education에서 LEVEL의 F0 대비 관측 MSE 차이는 0.87%입니다. 압축만 한 b를 보완한 효과와 F0를 선택할 이유는 별개입니다. 위 full/chunk 비용을 함께 보고, 실제 절약된 축이 없거나 단순 실행 대안에 지배되면 LEVEL의 실용 주장을 축소합니다.

Jena에서 LEVEL의 F0 대비 관측 MSE 차이는 10.97%입니다. 압축만 한 b를 보완한 효과와 F0를 선택할 이유는 별개입니다. 위 full/chunk 비용을 함께 보고, 실제 절약된 축이 없거나 단순 실행 대안에 지배되면 LEVEL의 실용 주장을 축소합니다.

### “그냥 Chronos-2를 쓰면 되지 않나?”

Robin: Chronos-2 MV의 MSE가 더 낮습니다. LEVEL의 채택 이유는 실제로 절약된 비용 축에 한정하며 정확도 손해를 함께 제시해야 합니다. Chronos-2-Small MV가 LEVEL보다 낮은 MSE·낮은 peak allocated·높은 batch4 처리량을 보였습니다. 이 측정 조건에서 LEVEL의 실용적 우위 주장을 축소해야 합니다.

Peacock Education: Chronos-2 MV의 MSE가 더 낮습니다. LEVEL의 채택 이유는 실제로 절약된 비용 축에 한정하며 정확도 손해를 함께 제시해야 합니다. 대응 MSE 구간이 0을 포함하여 정확도 차이에도 불확실성이 남습니다. LEVEL의 관측 MSE 점추정이 Chronos-2-Small MV보다 낮습니다. 해당 자료·고정 조건의 결과이며 MAE와 비용의 방향은 별도로 제시합니다. 대응 MSE 구간이 0을 포함하여 정확도 차이에도 불확실성이 남습니다. MAE는 LEVEL 0.315004 대 Chronos-2-Small MV 0.301992로 LEVEL이 더 높으며, 상대 손해는 4.31%, 조건부 95% 구간 [2.34, 6.80]%입니다.

Jena: Chronos-2 MV의 MSE가 더 낮습니다. LEVEL의 채택 이유는 실제로 절약된 비용 축에 한정하며 정확도 손해를 함께 제시해야 합니다. Chronos-2-Small MV가 LEVEL보다 낮은 MSE·낮은 peak allocated·높은 batch4 처리량을 보였습니다. 이 측정 조건에서 LEVEL의 실용적 우위 주장을 축소해야 합니다. 메모리 이점은 peak allocated 축이며 모든 메모리 축의 지배가 아닙니다: B4 allocated는 Chronos-2-Small MV 195.612 대 LEVEL 217.904 MiB지만, reserved는 252.000 대 238.000 MiB로 새 모델이 더 높습니다.

## 근거·범위·자원·게시 상태

[확인] 기록 시점 누적 GPU 작업 2844.00초/10800초, CPU 작업 24.82초/3600초, 새 다운로드 111750017 bytes, 새 폴더 저장 149859312 bytes입니다. 최종 종료 시간은 ledger.json과 STATUS.md를 기준으로 확인합니다.

추가 적합 0회이며 이번 원장에 기록한 최초 실패·재시도도 보존됩니다. preflight_checks.json은 사전 검사, final_checks.json은 최종 원수치·표·원장·보존 검산 상태입니다. 실행팀의 자체 검산을 독립 재현으로 부르지 않습니다.

[한계] 세 자료는 이전에 노출되었습니다. zero-shot은 대상 자료 추가 적합이 없다는 뜻이며 사전학습 비중복은 미확인입니다. 모델 크기·구조·사전학습·적응 이력이 다른 실용 대안 비교입니다. C2 MV와 UNI 비교를 제외한 모델 차이를 채널 공유만의 순수 효과로 설명하지 않습니다.

원본 PPT/PDF를 편집하지 않았으며 영어 표·캡션은 PRESENTATION_TABLES_EN.md, 수정 제안은 SLIDE_PATCHES.md에 별도로 남겼습니다. 이 회차 산출물은 로컬이며 commit/push 상태는 STATUS.md와 실제 Git 상태를 확인해야 합니다.

공식 모델·고정 revision: [Chronos-2](https://huggingface.co/amazon/chronos-2/tree/29ec3766d36d6f73f0696f85560a422f50e8498c), [Chronos-2-Small](https://huggingface.co/autogluon/chronos-2-small/tree/ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a). 실제 파일·라이브러리·group 경로는 model_manifest.json과 preflight_checks.json에 연결됩니다.
