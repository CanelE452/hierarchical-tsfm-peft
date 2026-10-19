# 기준 예측 이후 추가 보정: 실제 결과

이 문서는 공개 RUNS/COST/충분통계/학습곡선에서 다시 생성했다. V는 반복 사용한 개발 평가이며 독립 TEST가 아니다. M_select도 반복 선택에 사용했다.
두 seed의 점과 min/max는 관측 범위이며 신뢰구간이 아니다. 겹치는 예측 원점을 독립 표본으로 취급한 p-value는 계산하지 않았다.

G는 자료별 seed 92601의 기존 부모 하나이며 모든 S/보정 seed에서 공유한다. S는 Jena K12 cap3, PeMS32 K8 cap6의 기존 A60 부모다. A_stop도 A60 refit에 포함되므로 새 보정 선택에 재사용하지 않았다.
새 보정은 G/S가 학습·선택하지 않은 M_cal에서 M_fit 학습, 경계 정답을 purge한 M_select 선택을 거쳤다. 선택 후 전체 M_cal에서 같은 A60 초기 상태로 fresh refit했다. 보정 경로에 x_last를 중복 가산하지 않는다.

## jena

```text
method                 seed      K   MSE          MAE        selected_epoch  selected_prefix
G                      NA_shared_parent N/A    0.632217    0.441521            N/A            N/A
ALPHA_fitted           92601    N/A    0.572715    0.421198            N/A            N/A
ALPHA_fitted           92602    N/A    0.568282    0.420036            N/A            N/A
READOUT_ONLY           92601     12    0.666823    0.480888            1.0            N/A
READOUT_ONLY           92602     12    0.635848    0.446631           0.25            N/A
RESIDUAL_F             92601     12    0.643453    0.463478           0.25            N/A
RESIDUAL_F             92602     12    0.636034     0.45193           0.25            N/A
RESIDUAL_Q             92601     12    0.633319    0.459223           0.25            N/A
RESIDUAL_Q             92602     12    0.624594    0.445032           0.25            N/A
RESIDUAL_RAW_MLP       92601    N/A    0.632217    0.441521            0.0            N/A
RESIDUAL_RAW_MLP       92602    N/A    0.608351    0.436617           0.25            N/A
S                      92601    N/A    0.638959    0.482834            N/A            N/A
S                      92602    N/A    0.632572     0.48126            N/A            N/A
RAW                    NA_shared_parent N/A    0.849194    0.481388            N/A            N/A
MEAN                   NA_shared_parent N/A    0.631301    0.438319            N/A            N/A
OWN                    NA_shared_parent N/A    0.621753     0.45295            N/A            N/A
ALL                    NA_shared_parent N/A    0.643995    0.454273            N/A            N/A
SEQ_BLOCKS             92601      0    0.632217    0.441521     [0.25,1.0]             12
SEQ_BLOCKS             92601      6     0.64298    0.455664     [0.25,1.0]             12
SEQ_BLOCKS             92601     12    0.647094    0.464752     [0.25,1.0]             12
SEQ_BLOCKS             92602      0    0.632217    0.441521    [0.25,0.25]             12
SEQ_BLOCKS             92602      6    0.641032    0.448469    [0.25,0.25]             12
SEQ_BLOCKS             92602     12    0.636214    0.446305    [0.25,0.25]             12
JOINT_FULL             92601      0    0.632217    0.441521           0.25             12
JOINT_FULL             92601      6    0.641797    0.455249           0.25             12
JOINT_FULL             92601     12    0.637484    0.454845           0.25             12
JOINT_FULL             92602      0    0.632217    0.441521           0.25             12
JOINT_FULL             92602      6    0.639918    0.448827           0.25             12
JOINT_FULL             92602     12    0.635403    0.446999           0.25             12
JOINT_PREFIX           92601      0    0.632217    0.441521           0.25             12
JOINT_PREFIX           92601      6    0.641897     0.45496           0.25             12
JOINT_PREFIX           92601     12    0.638225    0.455086           0.25             12
JOINT_PREFIX           92602      0    0.632217    0.441521           0.25             12
JOINT_PREFIX           92602      6    0.640197    0.448252           0.25             12
JOINT_PREFIX           92602     12    0.636261    0.446896           0.25             12
ALPHA_0                92601    N/A    0.632217    0.441521            N/A            N/A
ALPHA_05               92601    N/A    0.570827    0.424944            N/A            N/A
ALPHA_1                92601    N/A    0.638959    0.482834            N/A            N/A
ALPHA_0                92602    N/A    0.632217    0.441521            N/A            N/A
ALPHA_05               92602    N/A    0.567053    0.422863            N/A            N/A
ALPHA_1                92602    N/A    0.632572     0.48126            N/A            N/A
```

RESIDUAL_F 평균 MSE는 0.639744, G는 0.632217이다. 차이(F−G)는 0.00752671이며 음수이면 개선이다.
ALPHA_fitted 대비 F 평균 MSE 차이: 0.0692452. 각 방법의 MAE와 seed별 음의 결과도 위 표에 보존했다.
READOUT_ONLY 대비 F 평균 MSE 차이: -0.0115914. 각 방법의 MAE와 seed별 음의 결과도 위 표에 보존했다.
RESIDUAL_Q 대비 F 평균 MSE 차이: 0.0107874. 각 방법의 MAE와 seed별 음의 결과도 위 표에 보존했다.
RESIDUAL_RAW_MLP 대비 F 평균 MSE 차이: 0.0194599. 각 방법의 MAE와 seed별 음의 결과도 위 표에 보존했다.
OWN 대비 F 평균 MSE 차이: 0.0179904. 각 방법의 MAE와 seed별 음의 결과도 위 표에 보존했다.
ALL 대비 F 평균 MSE 차이: -0.00425166. 각 방법의 MAE와 seed별 음의 결과도 위 표에 보존했다.

서로 다른 자료의 정규화 MSE 절대값으로 난이도 순위를 정하지 않는다. OWN/ALL은 차원이 다르므로 차이를 순수 교차채널 효과로 해석하지 않는다.

## pems32

```text
method                 seed      K   MSE          MAE        selected_epoch  selected_prefix
G                      NA_shared_parent N/A   0.0814654    0.185775            N/A            N/A
ALPHA_fitted           92601    N/A    0.076082    0.178634            N/A            N/A
ALPHA_fitted           92602    N/A   0.0763972    0.179257            N/A            N/A
READOUT_ONLY           92601      8   0.0814654    0.185775            0.0            N/A
READOUT_ONLY           92602      8   0.0786517    0.180825           0.25            N/A
RESIDUAL_F             92601      8   0.0784313    0.179955           0.25            N/A
RESIDUAL_F             92602      8   0.0779382    0.179194           0.25            N/A
RESIDUAL_Q             92601      8   0.0791771     0.18045           0.25            N/A
RESIDUAL_Q             92602      8   0.0787884    0.180015           0.25            N/A
RESIDUAL_RAW_MLP       92601    N/A   0.0798328    0.181092            1.0            N/A
RESIDUAL_RAW_MLP       92602    N/A   0.0800234    0.181644           0.25            N/A
S                      92601    N/A   0.0903477     0.19896            N/A            N/A
S                      92602    N/A   0.0914698    0.200695            N/A            N/A
RAW                    NA_shared_parent N/A   0.0905138    0.195211            N/A            N/A
MEAN                   NA_shared_parent N/A   0.0812314    0.184221            N/A            N/A
OWN                    NA_shared_parent N/A   0.0805922    0.182864            N/A            N/A
ALL                    NA_shared_parent N/A   0.0788069    0.179862            N/A            N/A
ALPHA_0                92601    N/A   0.0814654    0.185775            N/A            N/A
ALPHA_05               92601    N/A   0.0766076    0.179525            N/A            N/A
ALPHA_1                92601    N/A   0.0903477     0.19896            N/A            N/A
ALPHA_0                92602    N/A   0.0814654    0.185775            N/A            N/A
ALPHA_05               92602    N/A   0.0770522    0.180389            N/A            N/A
ALPHA_1                92602    N/A   0.0914698    0.200695            N/A            N/A
```

RESIDUAL_F 평균 MSE는 0.0781847, G는 0.0814654이다. 차이(F−G)는 -0.00328066이며 음수이면 개선이다.
ALPHA_fitted 대비 F 평균 MSE 차이: 0.00194507. 각 방법의 MAE와 seed별 음의 결과도 위 표에 보존했다.
READOUT_ONLY 대비 F 평균 MSE 차이: -0.00187382. 각 방법의 MAE와 seed별 음의 결과도 위 표에 보존했다.
RESIDUAL_Q 대비 F 평균 MSE 차이: -0.000798044. 각 방법의 MAE와 seed별 음의 결과도 위 표에 보존했다.
RESIDUAL_RAW_MLP 대비 F 평균 MSE 차이: -0.00174338. 각 방법의 MAE와 seed별 음의 결과도 위 표에 보존했다.
OWN 대비 F 평균 MSE 차이: -0.00240754. 각 방법의 MAE와 seed별 음의 결과도 위 표에 보존했다.
ALL 대비 F 평균 MSE 차이: -0.000622236. 각 방법의 MAE와 seed별 음의 결과도 위 표에 보존했다.

서로 다른 자료의 정규화 MSE 절대값으로 난이도 순위를 정하지 않는다. OWN/ALL은 차원이 다르므로 차이를 순수 교차채널 효과로 해석하지 않는다.

## 전체 경로 비용

비용은 physical 과거와 원래 past mask에서 시작해 과거만 채우기, A60 scaler, G와 실제 보정/P/E/F/Q/D, 역표준화, CPU 출력을 포함한다. 캐시 읽기와 head-only 시간을 전체 비용으로 쓰지 않는다. 모델 로드와 window slicing은 별도 기록이다.
배치·장치별 실측을 아래에 분리했다. B8/origin은 단일 요청 latency가 아니다. allocated/reserved, per-PID GPU, CPU RSS는 서로 다른 메모리 좌표다. per-PID 불가 사유는 COST 원자료를 따른다.

```text
dataset method                 seed   K device B  ms/batch  ms/origin allocatedMiB reservedMiB RSSMiB measurement_origin
jena    ALL                    92601  N/A cpu    1     0.3385    0.3385         N/A         N/A  514.117 NEW_FRESH_PROCESS
jena    ALL                    92601  N/A cpu    8    0.76175 0.0952188         N/A         N/A  527.598 NEW_FRESH_PROCESS
jena    ALPHA_fitted           92601  N/A cuda   1    12.5953   12.5953     204.218         274  1380.12 REUSED_IDENTICAL_CONTRACT
jena    ALPHA_fitted           92601  N/A cuda   8    20.8457   2.60572     294.308         308   1374.4 REUSED_IDENTICAL_CONTRACT
jena    ALPHA_fitted           92602  N/A cuda   1    12.4041   12.4041     204.218         274  1379.79 REUSED_IDENTICAL_CONTRACT
jena    ALPHA_fitted           92602  N/A cuda   8    22.2195   2.77743     294.308         308  1375.74 REUSED_IDENTICAL_CONTRACT
jena    G                      92601  N/A cuda   1     0.3854    0.3854     9.28027          22  752.746 REUSED_IDENTICAL_CONTRACT
jena    G                      92601  N/A cuda   8     0.6109 0.0763625     10.2002          24  754.418 REUSED_IDENTICAL_CONTRACT
jena    JOINT_FULL             92601    0 cuda   1     0.3526    0.3526     9.28027          22  775.566 NEW_FRESH_PROCESS
jena    JOINT_FULL             92601    0 cuda   8    0.47825 0.0597813     9.87207          22  775.707 NEW_FRESH_PROCESS
jena    JOINT_FULL             92601   12 cuda   1    13.3132   13.3132      204.28         274  1386.82 NEW_FRESH_PROCESS
jena    JOINT_FULL             92601   12 cuda   8    22.3958   2.79948     294.822         310  1381.71 NEW_FRESH_PROCESS
jena    JOINT_FULL             92601    6 cuda   1    12.9471   12.9471     197.869         264  1386.42 NEW_FRESH_PROCESS
jena    JOINT_FULL             92601    6 cuda   8    15.9424   1.99279      243.44         260  1381.73 NEW_FRESH_PROCESS
jena    JOINT_FULL             92602    0 cuda   1    0.52635   0.52635     9.28027          22  775.141 NEW_FRESH_PROCESS
jena    JOINT_FULL             92602    0 cuda   8     0.4853 0.0606625     9.87207          22  775.711 NEW_FRESH_PROCESS
jena    JOINT_FULL             92602   12 cuda   1     13.325    13.325      204.28         274  1386.71 NEW_FRESH_PROCESS
jena    JOINT_FULL             92602   12 cuda   8    23.1969   2.89961     294.822         310  1381.66 NEW_FRESH_PROCESS
jena    JOINT_FULL             92602    6 cuda   1    13.1154   13.1154     197.869         264  1387.01 NEW_FRESH_PROCESS
jena    JOINT_FULL             92602    6 cuda   8    15.9149   1.98936      243.44         260  1380.73 NEW_FRESH_PROCESS
jena    JOINT_PREFIX           92601    0 cuda   1    0.40035   0.40035     9.28027          22  774.883 NEW_FRESH_PROCESS
jena    JOINT_PREFIX           92601    0 cuda   8    0.51165 0.0639562     9.87207          22  775.152 NEW_FRESH_PROCESS
jena    JOINT_PREFIX           92601   12 cuda   1    13.4575   13.4575      204.28         274  1386.58 NEW_FRESH_PROCESS
jena    JOINT_PREFIX           92601   12 cuda   8    23.1318   2.89148     294.822         310  1381.84 NEW_FRESH_PROCESS
jena    JOINT_PREFIX           92601    6 cuda   1     13.086    13.086     197.869         264  1386.56 NEW_FRESH_PROCESS
jena    JOINT_PREFIX           92601    6 cuda   8    15.8065   1.97581      243.44         260  1381.46 NEW_FRESH_PROCESS
jena    JOINT_PREFIX           92602    0 cuda   1     0.3486    0.3486     9.28027          22  775.406 NEW_FRESH_PROCESS
jena    JOINT_PREFIX           92602    0 cuda   8    0.49515 0.0618938     9.87207          22  775.449 NEW_FRESH_PROCESS
jena    JOINT_PREFIX           92602   12 cuda   1    13.5236   13.5236      204.28         274  1386.42 NEW_FRESH_PROCESS
jena    JOINT_PREFIX           92602   12 cuda   8    22.6612   2.83265     294.822         310  1380.83 NEW_FRESH_PROCESS
jena    JOINT_PREFIX           92602    6 cuda   1    12.8846   12.8846     197.869         264  1386.96 NEW_FRESH_PROCESS
jena    JOINT_PREFIX           92602    6 cuda   8    15.8343   1.97929      243.44         260  1381.11 NEW_FRESH_PROCESS
jena    MEAN                   92601  N/A cpu    1    0.15925   0.15925         N/A         N/A  508.195 NEW_FRESH_PROCESS
jena    MEAN                   92601  N/A cpu    8    0.33125 0.0414062         N/A         N/A  509.539 NEW_FRESH_PROCESS
jena    OWN                    92601  N/A cpu    1     0.2674    0.2674         N/A         N/A  509.645 NEW_FRESH_PROCESS
jena    OWN                    92601  N/A cpu    8     0.5541 0.0692625         N/A         N/A  510.797 NEW_FRESH_PROCESS
jena    RAW                    92601  N/A cuda   1    11.7391   11.7391     213.558         260  1352.91 REUSED_IDENTICAL_CONTRACT
jena    RAW                    92601  N/A cuda   8     40.307   5.03837     330.083         354  1353.46 REUSED_IDENTICAL_CONTRACT
jena    READOUT_ONLY           92601   12 cuda   1    13.2967   13.2967     204.216         274  1386.38 NEW_FRESH_PROCESS
jena    READOUT_ONLY           92601   12 cuda   8    22.6475   2.83094     294.306         308  1377.34 NEW_FRESH_PROCESS
jena    READOUT_ONLY           92602   12 cuda   1    13.2422   13.2422     204.216         274  1386.69 NEW_FRESH_PROCESS
jena    READOUT_ONLY           92602   12 cuda   8    22.2779   2.78474     294.306         308  1377.77 NEW_FRESH_PROCESS
jena    RESIDUAL_F             92601   12 cuda   1     12.639    12.639     204.216         274   1387.8 NEW_FRESH_PROCESS
jena    RESIDUAL_F             92601   12 cuda   8    22.0543   2.75679     294.306         308  1381.57 NEW_FRESH_PROCESS
jena    RESIDUAL_F             92602   12 cuda   1    13.2471   13.2471     204.216         274  1386.95 NEW_FRESH_PROCESS
jena    RESIDUAL_F             92602   12 cuda   8    22.1536   2.76921     294.306         308  1382.58 NEW_FRESH_PROCESS
jena    RESIDUAL_Q             92601   12 cuda   1     0.9345    0.9345      9.7002          22   803.66 NEW_FRESH_PROCESS
jena    RESIDUAL_Q             92601   12 cuda   8    1.06075  0.132594     11.7505          24   799.34 NEW_FRESH_PROCESS
jena    RESIDUAL_Q             92602   12 cuda   1    0.90825   0.90825      9.7002          22  803.184 NEW_FRESH_PROCESS
jena    RESIDUAL_Q             92602   12 cuda   8    1.01995  0.127494     11.7505          24  799.027 NEW_FRESH_PROCESS
jena    RESIDUAL_RAW_MLP       92601  N/A cuda   1     0.5101    0.5101     9.43213          22  778.082 NEW_FRESH_PROCESS
jena    RESIDUAL_RAW_MLP       92601  N/A cuda   8    0.59395 0.0742437     10.1064          24  779.141 NEW_FRESH_PROCESS
jena    RESIDUAL_RAW_MLP       92602  N/A cuda   1    0.47385   0.47385     9.43213          22  779.094 NEW_FRESH_PROCESS
jena    RESIDUAL_RAW_MLP       92602  N/A cuda   8    0.62265 0.0778313     10.1064          24  779.078 NEW_FRESH_PROCESS
jena    S                      92601  N/A cuda   1    12.7855   12.7855     204.143         276  1379.48 REUSED_IDENTICAL_CONTRACT
jena    S                      92601  N/A cuda   8    20.6657   2.58321     294.206         310  1373.53 REUSED_IDENTICAL_CONTRACT
jena    S                      92602  N/A cuda   1     12.243    12.243     204.143         276  1379.77 REUSED_IDENTICAL_CONTRACT
jena    S                      92602  N/A cuda   8    20.8342   2.60427     294.206         310  1373.65 REUSED_IDENTICAL_CONTRACT
jena    SEQ_BLOCKS             92601    0 cuda   1    0.34655   0.34655     9.28027          22   775.82 NEW_FRESH_PROCESS
jena    SEQ_BLOCKS             92601    0 cuda   8     0.4624    0.0578     9.87207          22  775.707 NEW_FRESH_PROCESS
jena    SEQ_BLOCKS             92601   12 cuda   1    14.3324   14.3324      204.28         274  1386.65 NEW_FRESH_PROCESS
jena    SEQ_BLOCKS             92601   12 cuda   8    23.0841   2.88551     294.822         310   1382.6 NEW_FRESH_PROCESS
jena    SEQ_BLOCKS             92601    6 cuda   1    14.4509   14.4509     197.869         264  1386.85 NEW_FRESH_PROCESS
jena    SEQ_BLOCKS             92601    6 cuda   8    16.2132   2.02665      243.44         260  1381.41 NEW_FRESH_PROCESS
jena    SEQ_BLOCKS             92602    0 cuda   1    0.35265   0.35265     9.28027          22  775.203 NEW_FRESH_PROCESS
jena    SEQ_BLOCKS             92602    0 cuda   8    0.46735 0.0584188     9.87207          22  775.801 NEW_FRESH_PROCESS
jena    SEQ_BLOCKS             92602   12 cuda   1     13.425    13.425      204.28         274  1387.41 NEW_FRESH_PROCESS
jena    SEQ_BLOCKS             92602   12 cuda   8    25.0862   3.13577     294.822         310  1382.54 NEW_FRESH_PROCESS
jena    SEQ_BLOCKS             92602    6 cuda   1    13.1315   13.1315     197.869         264   1387.2 NEW_FRESH_PROCESS
jena    SEQ_BLOCKS             92602    6 cuda   8    17.9768    2.2471      243.44         260  1381.31 NEW_FRESH_PROCESS
pems32  ALL                    92601  N/A cpu    1    0.29695   0.29695         N/A         N/A  515.355 NEW_FRESH_PROCESS
pems32  ALL                    92601  N/A cpu    8    0.67265 0.0840812         N/A         N/A  523.449 NEW_FRESH_PROCESS
pems32  ALPHA_fitted           92601  N/A cuda   1    13.6681   13.6681     199.968         268  1386.43 REUSED_IDENTICAL_CONTRACT
pems32  ALPHA_fitted           92601  N/A cuda   8    17.1756   2.14695     260.267         280  1381.72 REUSED_IDENTICAL_CONTRACT
pems32  ALPHA_fitted           92602  N/A cuda   1    12.4893   12.4893     199.968         268  1386.29 REUSED_IDENTICAL_CONTRACT
pems32  ALPHA_fitted           92602  N/A cuda   8    20.0709   2.50886     260.267         280  1381.66 REUSED_IDENTICAL_CONTRACT
pems32  G                      92601  N/A cuda   1     0.3899    0.3899     9.31982          22  763.773 REUSED_IDENTICAL_CONTRACT
pems32  G                      92601  N/A cuda   8     0.5679 0.0709875     10.7222          24   765.34 REUSED_IDENTICAL_CONTRACT
pems32  MEAN                   92601  N/A cpu    1     0.1515    0.1515         N/A         N/A  512.473 NEW_FRESH_PROCESS
pems32  MEAN                   92601  N/A cpu    8     0.3888    0.0486         N/A         N/A  517.641 NEW_FRESH_PROCESS
pems32  OWN                    92601  N/A cpu    1    0.29975   0.29975         N/A         N/A  512.152 NEW_FRESH_PROCESS
pems32  OWN                    92601  N/A cpu    8     0.6365 0.0795625         N/A         N/A  513.773 NEW_FRESH_PROCESS
pems32  RAW                    92601  N/A cuda   1    14.1352   14.1352      225.29         260  1363.67 REUSED_IDENTICAL_CONTRACT
pems32  RAW                    92601  N/A cuda   8     49.815   6.22687     330.708         354  1360.57 REUSED_IDENTICAL_CONTRACT
pems32  READOUT_ONLY           92601    8 cuda   1    12.4968   12.4968     199.966         268  1388.91 NEW_FRESH_PROCESS
pems32  READOUT_ONLY           92601    8 cuda   8    18.6423   2.33029     260.265         280  1384.79 NEW_FRESH_PROCESS
pems32  READOUT_ONLY           92602    8 cuda   1    13.2746   13.2746     199.966         268  1388.38 NEW_FRESH_PROCESS
pems32  READOUT_ONLY           92602    8 cuda   8    18.5832   2.32291     260.265         280  1383.66 NEW_FRESH_PROCESS
pems32  RESIDUAL_F             92601    8 cuda   1    13.1574   13.1574     199.966         268  1388.55 NEW_FRESH_PROCESS
pems32  RESIDUAL_F             92601    8 cuda   8    18.4199   2.30248     260.265         280  1379.98 NEW_FRESH_PROCESS
pems32  RESIDUAL_F             92602    8 cuda   1    12.9454   12.9454     199.966         268  1388.45 NEW_FRESH_PROCESS
pems32  RESIDUAL_F             92602    8 cuda   8    19.1127   2.38909     260.265         280  1379.79 NEW_FRESH_PROCESS
pems32  RESIDUAL_Q             92601    8 cuda   1     0.9235    0.9235     9.77686          22  805.625 NEW_FRESH_PROCESS
pems32  RESIDUAL_Q             92601    8 cuda   8    1.11275  0.139094     12.4121          26  801.609 NEW_FRESH_PROCESS
pems32  RESIDUAL_Q             92602    8 cuda   1    0.98385   0.98385     9.77686          22  804.973 NEW_FRESH_PROCESS
pems32  RESIDUAL_Q             92602    8 cuda   8     1.2185  0.152312     12.4121          26  802.105 NEW_FRESH_PROCESS
pems32  RESIDUAL_RAW_MLP       92601  N/A cuda   1     0.4921    0.4921     9.47803          22  779.445 NEW_FRESH_PROCESS
pems32  RESIDUAL_RAW_MLP       92601  N/A cuda   8    0.63615 0.0795187     10.4844          24  782.066 NEW_FRESH_PROCESS
pems32  RESIDUAL_RAW_MLP       92602  N/A cuda   1     0.4693    0.4693     9.47803          22  779.965 NEW_FRESH_PROCESS
pems32  RESIDUAL_RAW_MLP       92602  N/A cuda   8    0.64735 0.0809187     10.4844          24  781.738 NEW_FRESH_PROCESS
pems32  S                      92601  N/A cuda   1    12.1961   12.1961     199.901         268  1390.29 REUSED_IDENTICAL_CONTRACT
pems32  S                      92601  N/A cuda   8    17.1766   2.14708     260.189         280  1386.46 REUSED_IDENTICAL_CONTRACT
pems32  S                      92602  N/A cuda   1     12.403    12.403     199.901         268  1390.25 REUSED_IDENTICAL_CONTRACT
pems32  S                      92602  N/A cuda   8    17.5115   2.18894     260.189         280  1385.81 REUSED_IDENTICAL_CONTRACT
```

## 선택·조건부 단계·검증

C 판정 봉인 상태: SEALED_BEFORE_V. 상세 수치·진입/미진입·예산 사유는 [C_DECISIONS.json](C_DECISIONS.json)을 따른다. 조건 미충족은 방법의 불가능성 증명이 아니다.
고정된 수치 검산 상태: PASS. [NUMERICAL_VALIDATION.json](NUMERICAL_VALIDATION.json)을 수치 근거로 사용한다. 정규화·마스크·부모·기울기·재개·그림의 최종 상태는 [VALIDATION_REPORT.json](VALIDATION_REPORT.json)을 따른다. 원격 게시 여부는 별도 PUBLICATION_VERIFICATION.json의 main SHA·파일 검증으로 확인한다. 누락 자료는 PASS로 바꾸지 않았다.
자원 스냅샷까지 경과시간(초): 3351.13; GPU 소유 프로세스 누적(초): 1713.01. 실제 프로세스 계보와 종료 상태는 [RESOURCE_LEDGER.json](RESOURCE_LEDGER.json)을 따른다. 이후 보고·게시까지 전체 경과시간은 게시 검증 영수증에 별도로 남긴다.
추가 search/refit·캐시·부모 학습비용은 RUNS/학습 영수증의 각 열을 사용한다. 누락된 비용을 0으로 채우지 않는다.
상류 Jena 역정규화/FP32 반올림, 과거 보호자료 노출, F 사전학습 중복은 unresolved다. 이 결과로 모든 데이터 우위·독립 검증·인과 설명·원천적 신규성을 주장하지 않는다.
가까운 선행·기존 V9와 실제 확인 범위는 [PRIOR_ART_AND_DELTA.md](PRIOR_ART_AND_DELTA.md)에 있다. 직접 CATS 전체 재현 없이 선행 대비 우위를 판정하지 않는다.

## 추가 학습·캐시 비용

아래는 새 보정기의 실제 단계 영수증이다. search/refit의 epoch·노출·update와 시간을 분리했다. 과거 G/S 부모학습은 새 보정 비용에 중복 합산하지 않는다. cache 생성은 별도 열이며 shared 자산을 model 수만큼 새 학습했다고 세지 않는다.

```text
dataset method                 seed    phase  updates exposures train_s eval_s ckpt_s cache_s worker_s
jena    READOUT_ONLY           92601   refit      129      2049 0.708467 0.228348 0.056488 8.13313  1.25936
jena    READOUT_ONLY           92601   search     228      3630 1.48178 0.46454 0.113072 8.16862  2.34282
jena    READOUT_ONLY           92602   refit       33       513 0.191517 0.224455 0.0275971 7.98819 0.701043
jena    READOUT_ONLY           92602   search     228      3630 1.25505 0.445425 0.0976863 8.11321  2.06835
jena    RESIDUAL_F             92601   refit       33       513 4.47638 15.8865 0.0309451       0  20.6825
jena    RESIDUAL_F             92601   search     228      3630 32.7786 33.1518 0.162783       0  66.3964
jena    RESIDUAL_F             92602   refit       33       513 4.34122 16.0945 0.0289191       0  20.7001
jena    RESIDUAL_F             92602   search     228      3630 31.8303 32.2511 0.159187       0  64.5098
jena    RESIDUAL_Q             92601   refit       33       513 0.572958 1.00571 0.0375106       0  1.86551
jena    RESIDUAL_Q             92601   search     228      3630 3.60399 1.96086 0.169311       0  6.01025
jena    RESIDUAL_Q             92602   refit       33       513 0.515989 0.943114 0.0347791       0  1.73456
jena    RESIDUAL_Q             92602   search     228      3630 3.74303 1.91065 0.167054       0  6.07783
jena    RESIDUAL_RAW_MLP       92601   refit        0         0       0 0.213263 0.0057163       0 0.481735
jena    RESIDUAL_RAW_MLP       92601   search     228      3630 2.56741 1.05279 0.148252       0  4.04133
jena    RESIDUAL_RAW_MLP       92602   refit       33       513 0.305122 0.409808 0.0298302       0  1.00903
jena    RESIDUAL_RAW_MLP       92602   search     228      3630 1.92428 0.865439 0.119339       0  3.16501
pems32  READOUT_ONLY           92601   refit        0         0       0 0.074822 0.0062598 5.87972 0.345631
pems32  READOUT_ONLY           92601   search     200      3180 1.15291 0.305281 0.0951973 6.27229  1.84895
pems32  READOUT_ONLY           92602   refit       28       444 0.188345 0.158712 0.0208322 6.48218 0.651779
pems32  READOUT_ONLY           92602   search     200      3180 1.16889 0.299761 0.0960415 6.16157  1.85055
pems32  RESIDUAL_F             92601   refit       28       444 3.55754 12.8523 0.0273076       0  16.7384
pems32  RESIDUAL_F             92601   search     200      3180 25.3338 24.6594 0.145888       0  50.4802
pems32  RESIDUAL_F             92602   refit       28       444 3.36268 12.2802 0.0251517       0  15.9375
pems32  RESIDUAL_F             92602   search     200      3180 25.3273 25.0815 0.136018       0  50.8413
pems32  RESIDUAL_Q             92601   refit       28       444 0.468743 0.820799 0.0294671       0  1.59531
pems32  RESIDUAL_Q             92601   search     200      3180 4.00099 2.05704 0.176788       0  6.52959
pems32  RESIDUAL_Q             92602   refit       28       444 0.453593 0.827769 0.0272699       0  1.58916
pems32  RESIDUAL_Q             92602   search     200      3180 3.55763 1.87266 0.165849       0  5.89094
pems32  RESIDUAL_RAW_MLP       92601   refit      111      1775 0.923316 0.275479 0.0560566       0   1.5299
pems32  RESIDUAL_RAW_MLP       92601   search     200      3180 1.86389 0.617433 0.125744       0  2.91074
pems32  RESIDUAL_RAW_MLP       92602   refit       28       444 0.198973 0.28278 0.0197161       0 0.797853
pems32  RESIDUAL_RAW_MLP       92602   search     200      3180 1.86975 0.574905 0.112695       0  2.83303
```

C 단계는 각 블록 update·목표노출·F forward/backward rows와 실제 wall을 별도로 공개했다. SEQ는 두 내부 학습 단계가 있어 단순 optimizer step 총합만으로 joint와 비용 동등이라고 쓰지 않는다.

```text
C/jena/JOINT_FULL_s92601.json
status: "COMPLETE"
selected_prefix: 12
selected_epoch: 0.25
search_accounting: {"checkpoint_seconds": 0.12231160006194841, "evaluation_seconds": 32.06842700000561, "optimizer_updates": 228, "target_exposures_per_block": [3630, 3630], "training_F_backward_rows": 43560, "training_F_forward_rows": 43560, "training_F_physical_rows": 43560, "training_seconds": 32.57523359998595, "updates_per_block": [228, 228]}
refit_accounting: {"checkpoint_seconds": 0.030864500018651597, "evaluation_seconds": 16.745251499989536, "optimizer_updates": 33, "target_exposures_per_block": [513, 513], "training_F_backward_rows": 6156, "training_F_forward_rows": 6156, "training_F_physical_rows": 6156, "training_seconds": 4.644329099988681, "updates_per_block": [33, 33]}
C/jena/JOINT_FULL_s92602.json
status: "COMPLETE"
selected_prefix: 12
selected_epoch: 0.25
search_accounting: {"checkpoint_seconds": 0.11797829999704845, "evaluation_seconds": 31.386895599993295, "optimizer_updates": 228, "target_exposures_per_block": [3630, 3630], "training_F_backward_rows": 43560, "training_F_forward_rows": 43560, "training_F_physical_rows": 43560, "training_seconds": 31.663686100000632, "updates_per_block": [228, 228]}
refit_accounting: {"checkpoint_seconds": 0.029648000025190413, "evaluation_seconds": 16.28321190000861, "optimizer_updates": 33, "target_exposures_per_block": [513, 513], "training_F_backward_rows": 6156, "training_F_forward_rows": 6156, "training_F_physical_rows": 6156, "training_seconds": 4.419935899990378, "updates_per_block": [33, 33]}
C/jena/JOINT_PREFIX_s92601.json
status: "COMPLETE"
selected_prefix: 12
selected_epoch: 0.25
search_accounting: {"checkpoint_seconds": 0.1407102000521263, "evaluation_seconds": 32.03891719998501, "optimizer_updates": 228, "target_exposures_per_block": [3630, 3630], "training_F_backward_rows": 43560, "training_F_forward_rows": 43560, "training_F_physical_rows": 43560, "training_seconds": 32.986482900043484, "updates_per_block": [228, 228]}
refit_accounting: {"checkpoint_seconds": 0.03310920001240447, "evaluation_seconds": 16.66400849999627, "optimizer_updates": 33, "target_exposures_per_block": [513, 513], "training_F_backward_rows": 6156, "training_F_forward_rows": 6156, "training_F_physical_rows": 6156, "training_seconds": 4.8919859000016, "updates_per_block": [33, 33]}
C/jena/JOINT_PREFIX_s92602.json
status: "COMPLETE"
selected_prefix: 12
selected_epoch: 0.25
search_accounting: {"checkpoint_seconds": 0.12070320002385415, "evaluation_seconds": 32.15409350000846, "optimizer_updates": 228, "target_exposures_per_block": [3630, 3630], "training_F_backward_rows": 43560, "training_F_forward_rows": 43560, "training_F_physical_rows": 43560, "training_seconds": 31.97491279990936, "updates_per_block": [228, 228]}
refit_accounting: {"checkpoint_seconds": 0.0280540999956429, "evaluation_seconds": 16.210136699999566, "optimizer_updates": 33, "target_exposures_per_block": [513, 513], "training_F_backward_rows": 6156, "training_F_forward_rows": 6156, "training_F_physical_rows": 6156, "training_seconds": 4.760572900020634, "updates_per_block": [33, 33]}
C/jena/SEQ_BLOCKS_s92601.json
status: "COMPLETE"
selected_prefix: 12
selected_epoch: [0.25, 1.0]
search_accounting: {"checkpoint_seconds": 0.2525399000151083, "evaluation_seconds": 54.51211029999831, "optimizer_updates": 456, "target_exposures_per_block": [3630, 3630], "training_F_backward_rows": 43560, "training_F_forward_rows": 43560, "training_F_physical_rows": 43560, "training_seconds": 54.173192900037975, "updates_per_block": [228, 228]}
refit_accounting: {"checkpoint_seconds": 0.09044510003877804, "evaluation_seconds": 27.6193229000055, "optimizer_updates": 162, "target_exposures_per_block": [513, 2049], "training_F_backward_rows": 15372, "training_F_forward_rows": 15372, "training_F_physical_rows": 15372, "training_seconds": 19.19560710003134, "updates_per_block": [33, 129]}
C/jena/SEQ_BLOCKS_s92602.json
status: "COMPLETE"
selected_prefix: 12
selected_epoch: [0.25, 0.25]
search_accounting: {"checkpoint_seconds": 0.24334179998550098, "evaluation_seconds": 55.77415369999653, "optimizer_updates": 456, "target_exposures_per_block": [3630, 3630], "training_F_backward_rows": 43560, "training_F_forward_rows": 43560, "training_F_physical_rows": 43560, "training_seconds": 53.457624200033024, "updates_per_block": [228, 228]}
refit_accounting: {"checkpoint_seconds": 0.04875699999684002, "evaluation_seconds": 26.917372399984743, "optimizer_updates": 66, "target_exposures_per_block": [513, 513], "training_F_backward_rows": 6156, "training_F_forward_rows": 6156, "training_F_physical_rows": 6156, "training_seconds": 7.376559699987411, "updates_per_block": [33, 33]}
```

## 현재 비교가 뒷받침하는 판단

M_select의 조건부 진입은 자원 배정 규칙이다. 그곳의 개선이 이후 V에도 유지된다는 보장이 아니다. V를 본 뒤 checkpoint, 학습량, prefix 또는 후보를 바꾸지 않았다.

jena K=12: SEQ 평균 MSE 0.6416538985, JOINT_FULL 0.6364432742, JOINT_PREFIX 0.637243002.
이 세 방법의 같은 K·seed별 값과 모든 미선택 prefix도 위 표에 남겼다. 평균 비교를 모든 seed의 동일한 방향으로 확대하지 않는다.

V에서는 모든 비영 prefix의 MSE가 G-only K0보다 높았다. 이는 선택 구간에서 정한 K의 전이 실패다. K0로 사후 재선택하지 않았다.

jena: 이번 등록 비교는 기존 fitted alpha 결합을 새 잔차 F로 교체할 근거를 주지 못했다.
pems32: 이번 등록 비교는 기존 fitted alpha 결합을 새 잔차 F로 교체할 근거를 주지 못했다.

비용 104개 facet 중 기존 동일 계약 실측 24개를 원 영수증·trial·PID·시간과 함께 재사용했고, 80개를 이번에 새 독립 프로세스로 측정했다. 새로 측정한 것처럼 재사용 시간을 원장에 합산하지 않았다.
각 timed trial 직전 GPU peak를 재설정해 timed20회의 최대 allocated/reserved를 요약했다. CPU RSS·per-PID 샘플 peak는 warmup과 timed 구간의 별도 관측이며 순간 최대의 보장이 아니다.
Q도 A60 S의 P/E 초기화를 사용하므로 S 부모의 과거 F 학습 정보를 물려받는다. no-F는 새 보정 학습·배포에서 F를 호출하지 않는다는 뜻이며, F를 한 번도 이용하지 않은 학습 이력이라는 뜻이 아니다.
B의 dataset 묶음 학습 worker에는 F가 계속 상주한다. Q/원본 MLP의 그 worker GPU peak를 독립 no-F 학습 메모리로 해석하지 않는다. 독립 배포 메모리는 COST.csv의 fresh-process 경로에서 확인한다.
실제 source-array의 보호 T 숫자 접근은 없었다. 과거 공개 역사 문서의 TEST 숫자가 우연히 노출됐는지는 현재 증거로 부재를 확정하지 못한다. [INFORMATION_ACCESS_AUDIT.json](INFORMATION_ACCESS_AUDIT.json)에 범위를 구분했다.
C 전체작업의 legacy calls=0 필드는 미계측 값을 뜻해 공개 사본에서 N/A로 해석했다. 원 기록은 보존했다. 훈련 F rows와 추론의 실측 F calls는 이 필드와 별개다. [COUNTER_INTERPRETATION.json](COUNTER_INTERPRETATION.json)을 따른다.
