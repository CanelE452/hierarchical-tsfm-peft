# 비교·절제 실험 결과

[확인] 저장된 V 예측을 FP64로 재계산하고 receipt와 1e-9 이내 일치를 확인했다.

종합 비교 판정: SUPPORTED_PROVISIONAL

```text
dataset             method   n    median V MSE    IQR          cap hits
jena                RAW     1    0.849194        0.000000     0
jena                C1      1    1.079489        0.000000     0
jena                C2      10   0.715067        0.017248     9
jena                C3      10   0.715753        0.012389     8
jena                OURS    10   0.697770        0.021174     2
jena                A1      10   0.766177        0.026572     10
jena                A2      10   0.723501        0.028942     2
jena                A3      0    NA              NA           0
jena                A4      0    NA              NA           0
jena                A5      0    NA              NA           0
ETTh1               RAW     1    0.283405        0.000000     0
ETTh1               C1      1    0.715743        0.000000     0
ETTh1               C2      10   0.678325        0.040459     5
ETTh1               C3      10   0.666153        0.058611     5
ETTh1               OURS    10   0.279643        0.005426     0
ETTh1               A1      10   0.279226        0.003130     3
ETTh1               A2      10   0.277454        0.003690     3
ETTh1               A3      0    NA              NA           0
ETTh1               A4      0    NA              NA           0
ETTh1               A5      0    NA              NA           0
ETTh2_safe17400     RAW     1    0.259555        0.000000     0
ETTh2_safe17400     C1      1    0.901934        0.000000     0
ETTh2_safe17400     C2      10   1.091104        0.102803     6
ETTh2_safe17400     C3      10   1.137333        0.109572     6
ETTh2_safe17400     OURS    10   0.277231        0.021597     2
ETTh2_safe17400     A1      10   0.263365        0.003370     10
ETTh2_safe17400     A2      10   0.278526        0.026056     0
ETTh2_safe17400     A3      0    NA              NA           0
ETTh2_safe17400     A4      0    NA              NA           0
ETTh2_safe17400     A5      0    NA              NA           0
appliances          RAW     1    0.347579        0.000000     0
appliances          C1      1    0.417315        0.000000     0
appliances          C2      10   0.324931        0.009727     3
appliances          C3      10   0.332662        0.004946     6
appliances          OURS    10   0.282019        0.006378     0
appliances          A1      10   0.283612        0.003923     1
appliances          A2      10   0.277510        0.003301     0
appliances          A3      0    NA              NA           0
appliances          A4      0    NA              NA           0
appliances          A5      0    NA              NA           0
```

비교 판정과 절제 판정의 상세는 COMPARATIVE.csv, ABLATION.csv에 있다.

오차 분포 설명:
- jena: 90·95·99% 오차는 Raw보다 낮고 50% 오차는 높다는 문장 = True. 기술 통계 설명이며 별도 판정은 아니다.
- ETTh1: 90·95·99% 오차는 Raw보다 낮고 50% 오차는 높다는 문장 = False. 기술 통계 설명이며 별도 판정은 아니다.
- ETTh2_safe17400: 90·95·99% 오차는 Raw보다 낮고 50% 오차는 높다는 문장 = False. 기술 통계 설명이며 별도 판정은 아니다.
- appliances: 90·95·99% 오차는 Raw보다 낮고 50% 오차는 높다는 문장 = True. 기술 통계 설명이며 별도 판정은 아니다.

비교 판정:

```text
dataset             best  gap         spread      decision
jena                C2    0.017297    0.021174    TIE_PROVISIONAL
ETTh1               C3    0.386511    0.058611    WIN_PROVISIONAL
ETTh2_safe17400     C1    0.624704    0.021597    WIN_PROVISIONAL
appliances          C2    0.042912    0.009727    WIN_PROVISIONAL
```

절제 집계 (jena·ETTh1·appliances):

```text
method  CONTRIBUTES  HARMS  UNCLEAR  INCOMPLETE
A1      1            0      2        0
A2      0            0      3        0
A3      0            0      0        3
A4      0            0      0        3
A5      0            0      0        3
```

위 절제 집계에는 다음 잠정(PROVISIONAL) 판정이 포함된다: A1/jena=CONTRIBUTES_PROVISIONAL.
해당 판정은 필요한 연장이 완료되지 않은 기본 결과에 근거한다. 집계의 CONTRIBUTES 수에 포함되어도 확정된 구성요소 기여로 해석할 수 없다.

Raw 대 OURS 보조 지표 (seed 중앙값):

```text
dataset              | method   | MAE          | R2           | MASE         | MAPE(a)%     | MAPE(c)%     | nearzero    
jena                 | RAW      | 0.481388     | 0.500973     | 9.618547     | 110.837943   | 45.978937    | 0.157484    
jena                 | OURS     | 0.497155     | 0.561646     | 10.650549    | 121.761145   | 43.530467    | 0.157484    
ETTh1                | RAW      | 0.324030     | 0.435846     | 1.518434     | 47.335493    | 34.052210    | 0.018022    
ETTh1                | OURS     | 0.348453     | 0.365474     | 1.692377     | 49.063759    | 36.722434    | 0.018022    
ETTh2_safe17400      | RAW      | 0.312753     | 0.415699     | 1.991622     | 40.637411    | 35.502063    | 0.208168    
ETTh2_safe17400      | OURS     | 0.357749     | 0.260870     | 2.460566     | 44.599826    | 40.745093    | 0.208168    
appliances           | RAW      | 0.333530     | 0.474509     | 10.378861    | 349920920942788.937500 | 18.027386    | 0.041334    
appliances           | OURS     | 0.337514     | 0.520304     | 10.393873    | 158829827865720.000000 | 16.997880    | 0.041334    
```

전체 family의 보조 지표는 SUMMARY.csv, 오차 분위수와 상위 오차 기여도는 ERROR_DISTRIBUTION_SUMMARY.csv에 있다.

Raw 대 OURS 절대오차 분위수 (seed 중앙값):

```text
dataset             method  q50         q90         q95         q99         better fraction
jena                RAW     0.204851    1.250836    1.957507    3.887369    0.000000
jena                OURS    0.298598    1.114967    1.659133    3.186592    0.402097
ETTh1               RAW     0.203269    0.677625    1.094758    2.280959    0.000000
ETTh1               OURS    0.233619    0.735358    1.118128    2.024841    0.440063
ETTh2_safe17400     RAW     0.162447    0.875195    1.198900    1.842405    0.000000
ETTh2_safe17400     OURS    0.226007    0.879736    1.200241    1.820080    0.378472
appliances          RAW     0.166498    0.833822    1.219942    2.318662    0.000000
appliances          OURS    0.207921    0.790464    1.075467    1.893698    0.447799
```

해석 범위와 한계:
- 최선 C1–C3는 V 중앙값으로 골랐다. 비교 대상에 유리한 선택이다.
- C2·C3는 AdaPTS식 어댑터이며 원 논문 재현이 아니다. 차이는 DEVIATIONS_FROM_ADAPTS.md에 있다.
- Mann–Whitney U 양측 p값은 기술 통계로만 보고하며 판정이나 다중비교 유의성 주장에 쓰지 않는다.
- 10 seed가 채워지지 않은 기본 비교·절제 그룹은 INCOMPLETE로 남긴다. 연장 10 seed가 모두 끝나면 연장을 주 결과로 사용한다. 연장이 미완료이면 기본 결과와 PROVISIONAL 표시를 유지하며 모든 실행은 RUNS.csv에 보존한다.
- MASE의 인접 관측쌍이 없거나 척도가 0인 채널이 있으면 전체 MASE는 NA다. 유효 채널만의 평균과 제외 수를 별도 열로 공개한다.
- MASE 출처: [Hyndman & Koehler (2006), Another look at measures of forecast accuracy](https://doi.org/10.1016/j.ijforecast.2006.03.001). 이 캠페인은 A60의 lag-1 관측쌍으로 척도를 고정한다.
- MAPE의 예측값은 저장된 FP32 정규화 예측과 FP32 학습 scaler를 FP64로 역변환한다. 정답과 0 여부는 같은 origin·horizon·채널·관측 마스크에 결속한 원단위 physical V 정답을 쓴다. 정답을 같은 FP32 scaler로 정규화하면 저장된 관측 target과 정확히 일치해야 한다. 이 방식은 정규화 왕복 반올림이 실제 0을 미세한 비영 값으로 바꾸는 문제를 막으며 epsilon은 쓰지 않는다. MSE·MAE·R2·MASE와 오차 분위수는 기존 정규화 단위를 유지한다.
- 정확한 0과 0 근처 비율 및 채널별 제외 수를 보고하며 0.1 표준편차 기준은 판정에 쓰지 않는다.
- Raw 상위 5%는 관측 점의 절대오차 내림차순에서 ceil(0.05*n)개를 고른다. 동률은 평탄화된 원래 순서를 유지한다.

실행 상태와 예산 기록:

```json
{
  "active": {},
  "elapsed_seconds": 138701.64415812492,
  "extension_status": {
    "A1/ETTh1": {
      "cap_count": 3,
      "n": 10,
      "performed": false,
      "status": "NOT_TRIGGERED",
      "triggered": false
    },
    "A1/ETTh2_safe17400": {
      "cap_count": 10,
      "n": 10,
      "performed": false,
      "status": "NOT_RUN_BUDGET",
      "triggered": true
    },
    "A1/appliances": {
      "cap_count": 1,
      "n": 10,
      "performed": false,
      "status": "NOT_TRIGGERED",
      "triggered": false
    },
    "A1/jena": {
      "cap_count": 10,
      "n": 10,
      "performed": false,
      "status": "NOT_RUN_BUDGET",
      "triggered": true
    },
    "A2/ETTh1": {
      "cap_count": 3,
      "n": 10,
      "performed": false,
      "status": "NOT_TRIGGERED",
      "triggered": false
    },
    "A2/ETTh2_safe17400": {
      "cap_count": 0,
      "n": 10,
      "performed": false,
      "status": "NOT_TRIGGERED",
      "triggered": false
    },
    "A2/appliances": {
      "cap_count": 0,
      "n": 10,
      "performed": false,
      "status": "NOT_TRIGGERED",
      "triggered": false
    },
    "A2/jena": {
      "cap_count": 2,
      "n": 10,
      "performed": false,
      "status": "NOT_TRIGGERED",
      "triggered": false
    },
    "A3/ETTh1": {
      "cap_count": 0,
      "n": 0,
      "performed": false,
      "status": "NOT_TRIGGERED",
      "triggered": false
    },
    "A3/ETTh2_safe17400": {
      "cap_count": 0,
      "n": 0,
      "performed": false,
      "status": "NOT_TRIGGERED",
      "triggered": false
    },
    "A3/appliances": {
      "cap_count": 0,
      "n": 0,
      "performed": false,
      "status": "NOT_TRIGGERED",
      "triggered": false
    },
    "A3/jena": {
      "cap_count": 0,
      "n": 0,
      "performed": false,
      "status": "NOT_TRIGGERED",
      "triggered": false
    },
    "A4/ETTh1": {
      "cap_count": 0,
      "n": 0,
      "performed": false,
      "status": "NOT_TRIGGERED",
      "triggered": false
    },
    "A4/ETTh2_safe17400": {
      "cap_count": 0,
      "n": 0,
      "performed": false,
      "status": "NOT_TRIGGERED",
      "triggered": false
    },
    "A4/appliances": {
      "cap_count": 0,
      "n": 0,
      "performed": false,
      "status": "NOT_TRIGGERED",
      "triggered": false
    },
    "A4/jena": {
      "cap_count": 0,
      "n": 0,
      "performed": false,
      "status": "NOT_TRIGGERED",
      "triggered": false
    },
    "A5/ETTh1": {
      "cap_count": 0,
      "n": 0,
      "performed": false,
      "status": "NOT_TRIGGERED",
      "triggered": false
    },
    "A5/ETTh2_safe17400": {
      "cap_count": 0,
      "n": 0,
      "performed": false,
      "status": "NOT_TRIGGERED",
      "triggered": false
    },
    "A5/appliances": {
      "cap_count": 0,
      "n": 0,
      "performed": false,
      "status": "NOT_TRIGGERED",
      "triggered": false
    },
    "A5/jena": {
      "cap_count": 0,
      "n": 0,
      "performed": false,
      "status": "NOT_TRIGGERED",
      "triggered": false
    },
    "C2/ETTh1": {
      "cap_count": 5,
      "n": 10,
      "performed": false,
      "status": "NOT_RUN_BUDGET",
      "triggered": true
    },
    "C2/ETTh2_safe17400": {
      "cap_count": 6,
      "n": 10,
      "performed": false,
      "status": "NOT_RUN_BUDGET",
      "triggered": true
    },
    "C2/appliances": {
      "cap_count": 3,
      "n": 10,
      "performed": false,
      "status": "NOT_TRIGGERED",
      "triggered": false
    },
    "C2/jena": {
      "cap_count": 9,
      "n": 10,
      "performed": false,
      "status": "NOT_RUN_BUDGET",
      "triggered": true
    },
    "C3/ETTh1": {
      "cap_count": 5,
      "n": 10,
      "performed": false,
      "status": "NOT_RUN_BUDGET",
      "triggered": true
    },
    "C3/ETTh2_safe17400": {
      "cap_count": 6,
      "n": 10,
      "performed": false,
      "status": "NOT_RUN_BUDGET",
      "triggered": true
    },
    "C3/appliances": {
      "cap_count": 6,
      "n": 10,
      "performed": false,
      "status": "NOT_RUN_BUDGET",
      "triggered": true
    },
    "C3/jena": {
      "cap_count": 8,
      "n": 10,
      "performed": false,
      "status": "NOT_RUN_BUDGET",
      "triggered": true
    }
  },
  "last_start_cutoff": "2026-10-27T21:00:00+09:00",
  "run_statuses": {
    "CA_A1_ETTh1_K4_s92601": "COMPLETE",
    "CA_A1_ETTh1_K4_s92602": "COMPLETE",
    "CA_A1_ETTh1_K4_s92603": "COMPLETE",
    "CA_A1_ETTh1_K4_s92604": "COMPLETE",
    "CA_A1_ETTh1_K4_s92605": "COMPLETE",
    "CA_A1_ETTh1_K4_s92606": "COMPLETE",
    "CA_A1_ETTh1_K4_s92607": "COMPLETE",
    "CA_A1_ETTh1_K4_s92608": "COMPLETE",
    "CA_A1_ETTh1_K4_s92609": "COMPLETE",
    "CA_A1_ETTh1_K4_s92610": "COMPLETE",
    "CA_A1_ETTh2_safe17400_K4_s92601": "COMPLETE",
    "CA_A1_ETTh2_safe17400_K4_s92602": "COMPLETE",
    "CA_A1_ETTh2_safe17400_K4_s92603": "COMPLETE",
    "CA_A1_ETTh2_safe17400_K4_s92604": "COMPLETE",
    "CA_A1_ETTh2_safe17400_K4_s92605": "COMPLETE",
    "CA_A1_ETTh2_safe17400_K4_s92606": "COMPLETE",
    "CA_A1_ETTh2_safe17400_K4_s92607": "COMPLETE",
    "CA_A1_ETTh2_safe17400_K4_s92608": "COMPLETE",
    "CA_A1_ETTh2_safe17400_K4_s92609": "COMPLETE",
    "CA_A1_ETTh2_safe17400_K4_s92610": "COMPLETE",
    "CA_A1_appliances_K10_s92601": "COMPLETE",
    "CA_A1_appliances_K10_s92602": "COMPLETE",
    "CA_A1_appliances_K10_s92603": "COMPLETE",
    "CA_A1_appliances_K10_s92604": "COMPLETE",
    "CA_A1_appliances_K10_s92605": "COMPLETE",
    "CA_A1_appliances_K10_s92606": "COMPLETE",
    "CA_A1_appliances_K10_s92607": "COMPLETE",
    "CA_A1_appliances_K10_s92608": "COMPLETE",
    "CA_A1_appliances_K10_s92609": "COMPLETE",
    "CA_A1_appliances_K10_s92610": "COMPLETE",
    "CA_A1_jena_K6_s92601": "COMPLETE",
    "CA_A1_jena_K6_s92602": "COMPLETE",
    "CA_A1_jena_K6_s92603": "COMPLETE",
    "CA_A1_jena_K6_s92604": "COMPLETE",
    "CA_A1_jena_K6_s92605": "COMPLETE",
    "CA_A1_jena_K6_s92606": "COMPLETE",
    "CA_A1_jena_K6_s92607": "COMPLETE",
    "CA_A1_jena_K6_s92608": "COMPLETE",
    "CA_A1_jena_K6_s92609": "COMPLETE",
    "CA_A1_jena_K6_s92610": "COMPLETE",
    "CA_A2_ETTh1_K4_s92601": "COMPLETE",
    "CA_A2_ETTh1_K4_s92602": "COMPLETE",
    "CA_A2_ETTh1_K4_s92603": "COMPLETE",
    "CA_A2_ETTh1_K4_s92604": "COMPLETE",
    "CA_A2_ETTh1_K4_s92605": "COMPLETE",
    "CA_A2_ETTh1_K4_s92606": "COMPLETE",
    "CA_A2_ETTh1_K4_s92607": "COMPLETE",
    "CA_A2_ETTh1_K4_s92608": "COMPLETE",
    "CA_A2_ETTh1_K4_s92609": "COMPLETE",
    "CA_A2_ETTh1_K4_s92610": "COMPLETE",
    "CA_A2_ETTh2_safe17400_K4_s92601": "COMPLETE",
    "CA_A2_ETTh2_safe17400_K4_s92602": "COMPLETE",
    "CA_A2_ETTh2_safe17400_K4_s92603": "COMPLETE",
    "CA_A2_ETTh2_safe17400_K4_s92604": "COMPLETE",
    "CA_A2_ETTh2_safe17400_K4_s92605": "COMPLETE",
    "CA_A2_ETTh2_safe17400_K4_s92606": "COMPLETE",
    "CA_A2_ETTh2_safe17400_K4_s92607": "COMPLETE",
    "CA_A2_ETTh2_safe17400_K4_s92608": "COMPLETE",
    "CA_A2_ETTh2_safe17400_K4_s92609": "COMPLETE",
    "CA_A2_ETTh2_safe17400_K4_s92610": "COMPLETE",
    "CA_A2_appliances_K10_s92601": "COMPLETE",
    "CA_A2_appliances_K10_s92602": "COMPLETE",
    "CA_A2_appliances_K10_s92603": "COMPLETE",
    "CA_A2_appliances_K10_s92604": "COMPLETE",
    "CA_A2_appliances_K10_s92605": "COMPLETE",
    "CA_A2_appliances_K10_s92606": "COMPLETE",
    "CA_A2_appliances_K10_s92607": "COMPLETE",
    "CA_A2_appliances_K10_s92608": "COMPLETE",
    "CA_A2_appliances_K10_s92609": "COMPLETE",
    "CA_A2_appliances_K10_s92610": "COMPLETE",
    "CA_A2_jena_K6_s92601": "COMPLETE",
    "CA_A2_jena_K6_s92602": "COMPLETE",
    "CA_A2_jena_K6_s92603": "COMPLETE",
    "CA_A2_jena_K6_s92604": "COMPLETE",
    "CA_A2_jena_K6_s92605": "COMPLETE",
    "CA_A2_jena_K6_s92606": "COMPLETE",
    "CA_A2_jena_K6_s92607": "COMPLETE",
    "CA_A2_jena_K6_s92608": "COMPLETE",
    "CA_A2_jena_K6_s92609": "COMPLETE",
    "CA_A2_jena_K6_s92610": "COMPLETE",
    "CA_A3_ETTh1_K4_s92601": "NOT_RUN_BUDGET",
    "CA_A3_ETTh1_K4_s92602": "NOT_RUN_BUDGET",
    "CA_A3_ETTh1_K4_s92603": "NOT_RUN_BUDGET",
    "CA_A3_ETTh1_K4_s92604": "NOT_RUN_BUDGET",
    "CA_A3_ETTh1_K4_s92605": "NOT_RUN_BUDGET",
    "CA_A3_ETTh1_K4_s92606": "NOT_RUN_BUDGET",
    "CA_A3_ETTh1_K4_s92607": "NOT_RUN_BUDGET",
    "CA_A3_ETTh1_K4_s92608": "NOT_RUN_BUDGET",
    "CA_A3_ETTh1_K4_s92609": "NOT_RUN_BUDGET",
    "CA_A3_ETTh1_K4_s92610": "NOT_RUN_BUDGET",
    "CA_A3_ETTh2_safe17400_K4_s92601": "NOT_RUN_BUDGET",
    "CA_A3_ETTh2_safe17400_K4_s92602": "NOT_RUN_BUDGET",
    "CA_A3_ETTh2_safe17400_K4_s92603": "NOT_RUN_BUDGET",
    "CA_A3_ETTh2_safe17400_K4_s92604": "NOT_RUN_BUDGET",
    "CA_A3_ETTh2_safe17400_K4_s92605": "NOT_RUN_BUDGET",
    "CA_A3_ETTh2_safe17400_K4_s92606": "NOT_RUN_BUDGET",
    "CA_A3_ETTh2_safe17400_K4_s92607": "NOT_RUN_BUDGET",
    "CA_A3_ETTh2_safe17400_K4_s92608": "NOT_RUN_BUDGET",
    "CA_A3_ETTh2_safe17400_K4_s92609": "NOT_RUN_BUDGET",
    "CA_A3_ETTh2_safe17400_K4_s92610": "NOT_RUN_BUDGET",
    "CA_A3_appliances_K10_s92601": "NOT_RUN_BUDGET",
    "CA_A3_appliances_K10_s92602": "NOT_RUN_BUDGET",
    "CA_A3_appliances_K10_s92603": "NOT_RUN_BUDGET",
    "CA_A3_appliances_K10_s92604": "NOT_RUN_BUDGET",
    "CA_A3_appliances_K10_s92605": "NOT_RUN_BUDGET",
    "CA_A3_appliances_K10_s92606": "NOT_RUN_BUDGET",
    "CA_A3_appliances_K10_s92607": "NOT_RUN_BUDGET",
    "CA_A3_appliances_K10_s92608": "NOT_RUN_BUDGET",
    "CA_A3_appliances_K10_s92609": "NOT_RUN_BUDGET",
    "CA_A3_appliances_K10_s92610": "NOT_RUN_BUDGET",
    "CA_A3_jena_K6_s92601": "NOT_RUN_BUDGET",
    "CA_A3_jena_K6_s92602": "NOT_RUN_BUDGET",
    "CA_A3_jena_K6_s92603": "NOT_RUN_BUDGET",
    "CA_A3_jena_K6_s92604": "NOT_RUN_BUDGET",
    "CA_A3_jena_K6_s92605": "NOT_RUN_BUDGET",
    "CA_A3_jena_K6_s92606": "NOT_RUN_BUDGET",
    "CA_A3_jena_K6_s92607": "NOT_RUN_BUDGET",
    "CA_A3_jena_K6_s92608": "NOT_RUN_BUDGET",
    "CA_A3_jena_K6_s92609": "NOT_RUN_BUDGET",
    "CA_A3_jena_K6_s92610": "NOT_RUN_BUDGET",
    "CA_A4_ETTh1_K4_s92601": "NOT_RUN_BUDGET",
    "CA_A4_ETTh1_K4_s92602": "NOT_RUN_BUDGET",
    "CA_A4_ETTh1_K4_s92603": "NOT_RUN_BUDGET",
    "CA_A4_ETTh1_K4_s92604": "NOT_RUN_BUDGET",
    "CA_A4_ETTh1_K4_s92605": "NOT_RUN_BUDGET",
    "CA_A4_ETTh1_K4_s92606": "NOT_RUN_BUDGET",
    "CA_A4_ETTh1_K4_s92607": "NOT_RUN_BUDGET",
    "CA_A4_ETTh1_K4_s92608": "NOT_RUN_BUDGET",
    "CA_A4_ETTh1_K4_s92609": "NOT_RUN_BUDGET",
    "CA_A4_ETTh1_K4_s92610": "NOT_RUN_BUDGET",
    "CA_A4_ETTh2_safe17400_K4_s92601": "NOT_RUN_BUDGET",
    "CA_A4_ETTh2_safe17400_K4_s92602": "NOT_RUN_BUDGET",
    "CA_A4_ETTh2_safe17400_K4_s92603": "NOT_RUN_BUDGET",
    "CA_A4_ETTh2_safe17400_K4_s92604": "NOT_RUN_BUDGET",
    "CA_A4_ETTh2_safe17400_K4_s92605": "NOT_RUN_BUDGET",
    "CA_A4_ETTh2_safe17400_K4_s92606": "NOT_RUN_BUDGET",
    "CA_A4_ETTh2_safe17400_K4_s92607": "NOT_RUN_BUDGET",
    "CA_A4_ETTh2_safe17400_K4_s92608": "NOT_RUN_BUDGET",
    "CA_A4_ETTh2_safe17400_K4_s92609": "NOT_RUN_BUDGET",
    "CA_A4_ETTh2_safe17400_K4_s92610": "NOT_RUN_BUDGET",
    "CA_A4_appliances_K10_s92601": "NOT_RUN_BUDGET",
    "CA_A4_appliances_K10_s92602": "NOT_RUN_BUDGET",
    "CA_A4_appliances_K10_s92603": "NOT_RUN_BUDGET",
    "CA_A4_appliances_K10_s92604": "NOT_RUN_BUDGET",
    "CA_A4_appliances_K10_s92605": "NOT_RUN_BUDGET",
    "CA_A4_appliances_K10_s92606": "NOT_RUN_BUDGET",
    "CA_A4_appliances_K10_s92607": "NOT_RUN_BUDGET",
    "CA_A4_appliances_K10_s92608": "NOT_RUN_BUDGET",
    "CA_A4_appliances_K10_s92609": "NOT_RUN_BUDGET",
    "CA_A4_appliances_K10_s92610": "NOT_RUN_BUDGET",
    "CA_A4_jena_K6_s92601": "NOT_RUN_BUDGET",
    "CA_A4_jena_K6_s92602": "NOT_RUN_BUDGET",
    "CA_A4_jena_K6_s92603": "NOT_RUN_BUDGET",
    "CA_A4_jena_K6_s92604": "NOT_RUN_BUDGET",
    "CA_A4_jena_K6_s92605": "NOT_RUN_BUDGET",
    "CA_A4_jena_K6_s92606": "NOT_RUN_BUDGET",
    "CA_A4_jena_K6_s92607": "NOT_RUN_BUDGET",
    "CA_A4_jena_K6_s92608": "NOT_RUN_BUDGET",
    "CA_A4_jena_K6_s92609": "NOT_RUN_BUDGET",
    "CA_A4_jena_K6_s92610": "NOT_RUN_BUDGET",
    "CA_A5_ETTh1_K4_s92601": "NOT_RUN_BUDGET",
    "CA_A5_ETTh1_K4_s92602": "NOT_RUN_BUDGET",
    "CA_A5_ETTh1_K4_s92603": "NOT_RUN_BUDGET",
    "CA_A5_ETTh1_K4_s92604": "NOT_RUN_BUDGET",
    "CA_A5_ETTh1_K4_s92605": "NOT_RUN_BUDGET",
    "CA_A5_ETTh1_K4_s92606": "NOT_RUN_BUDGET",
    "CA_A5_ETTh1_K4_s92607": "NOT_RUN_BUDGET",
    "CA_A5_ETTh1_K4_s92608": "NOT_RUN_BUDGET",
    "CA_A5_ETTh1_K4_s92609": "NOT_RUN_BUDGET",
    "CA_A5_ETTh1_K4_s92610": "NOT_RUN_BUDGET",
    "CA_A5_ETTh2_safe17400_K4_s92601": "NOT_RUN_BUDGET",
    "CA_A5_ETTh2_safe17400_K4_s92602": "NOT_RUN_BUDGET",
    "CA_A5_ETTh2_safe17400_K4_s92603": "NOT_RUN_BUDGET",
    "CA_A5_ETTh2_safe17400_K4_s92604": "NOT_RUN_BUDGET",
    "CA_A5_ETTh2_safe17400_K4_s92605": "NOT_RUN_BUDGET",
    "CA_A5_ETTh2_safe17400_K4_s92606": "NOT_RUN_BUDGET",
    "CA_A5_ETTh2_safe17400_K4_s92607": "NOT_RUN_BUDGET",
    "CA_A5_ETTh2_safe17400_K4_s92608": "NOT_RUN_BUDGET",
    "CA_A5_ETTh2_safe17400_K4_s92609": "NOT_RUN_BUDGET",
    "CA_A5_ETTh2_safe17400_K4_s92610": "NOT_RUN_BUDGET",
    "CA_A5_appliances_K10_s92601": "NOT_RUN_BUDGET",
    "CA_A5_appliances_K10_s92602": "NOT_RUN_BUDGET",
    "CA_A5_appliances_K10_s92603": "NOT_RUN_BUDGET",
    "CA_A5_appliances_K10_s92604": "NOT_RUN_BUDGET",
    "CA_A5_appliances_K10_s92605": "NOT_RUN_BUDGET",
    "CA_A5_appliances_K10_s92606": "NOT_RUN_BUDGET",
    "CA_A5_appliances_K10_s92607": "NOT_RUN_BUDGET",
    "CA_A5_appliances_K10_s92608": "NOT_RUN_BUDGET",
    "CA_A5_appliances_K10_s92609": "NOT_RUN_BUDGET",
    "CA_A5_appliances_K10_s92610": "NOT_RUN_BUDGET",
    "CA_A5_jena_K6_s92601": "NOT_RUN_BUDGET",
    "CA_A5_jena_K6_s92602": "NOT_RUN_BUDGET",
    "CA_A5_jena_K6_s92603": "NOT_RUN_BUDGET",
    "CA_A5_jena_K6_s92604": "NOT_RUN_BUDGET",
    "CA_A5_jena_K6_s92605": "NOT_RUN_BUDGET",
    "CA_A5_jena_K6_s92606": "NOT_RUN_BUDGET",
    "CA_A5_jena_K6_s92607": "NOT_RUN_BUDGET",
    "CA_A5_jena_K6_s92608": "NOT_RUN_BUDGET",
    "CA_A5_jena_K6_s92609": "NOT_RUN_BUDGET",
    "CA_A5_jena_K6_s92610": "NOT_RUN_BUDGET",
    "CA_C1_ETTh1_K4_s92601": "COMPLETE",
    "CA_C1_ETTh2_safe17400_K4_s92601": "COMPLETE",
    "CA_C1_appliances_K10_s92601": "COMPLETE",
    "CA_C1_jena_K6_s92601": "COMPLETE",
    "CA_C2_ETTh1_K4_s92601": "COMPLETE",
    "CA_C2_ETTh1_K4_s92602": "COMPLETE",
    "CA_C2_ETTh1_K4_s92603": "COMPLETE",
    "CA_C2_ETTh1_K4_s92604": "COMPLETE",
    "CA_C2_ETTh1_K4_s92605": "COMPLETE",
    "CA_C2_ETTh1_K4_s92606": "COMPLETE",
    "CA_C2_ETTh1_K4_s92607": "COMPLETE",
    "CA_C2_ETTh1_K4_s92608": "COMPLETE",
    "CA_C2_ETTh1_K4_s92609": "COMPLETE",
    "CA_C2_ETTh1_K4_s92610": "COMPLETE",
    "CA_C2_ETTh2_safe17400_K4_s92601": "COMPLETE",
    "CA_C2_ETTh2_safe17400_K4_s92602": "COMPLETE",
    "CA_C2_ETTh2_safe17400_K4_s92603": "COMPLETE",
    "CA_C2_ETTh2_safe17400_K4_s92604": "COMPLETE",
    "CA_C2_ETTh2_safe17400_K4_s92605": "COMPLETE",
    "CA_C2_ETTh2_safe17400_K4_s92606": "COMPLETE",
    "CA_C2_ETTh2_safe17400_K4_s92607": "COMPLETE",
    "CA_C2_ETTh2_safe17400_K4_s92608": "COMPLETE",
    "CA_C2_ETTh2_safe17400_K4_s92609": "COMPLETE",
    "CA_C2_ETTh2_safe17400_K4_s92610": "COMPLETE",
    "CA_C2_appliances_K10_s92601": "COMPLETE",
    "CA_C2_appliances_K10_s92602": "COMPLETE",
    "CA_C2_appliances_K10_s92603": "COMPLETE",
    "CA_C2_appliances_K10_s92604": "COMPLETE",
    "CA_C2_appliances_K10_s92605": "COMPLETE",
    "CA_C2_appliances_K10_s92606": "COMPLETE",
    "CA_C2_appliances_K10_s92607": "COMPLETE",
    "CA_C2_appliances_K10_s92608": "COMPLETE",
    "CA_C2_appliances_K10_s92609": "COMPLETE",
    "CA_C2_appliances_K10_s92610": "COMPLETE",
    "CA_C2_jena_K6_s92601": "COMPLETE",
    "CA_C2_jena_K6_s92602": "COMPLETE",
    "CA_C2_jena_K6_s92603": "COMPLETE",
    "CA_C2_jena_K6_s92604": "COMPLETE",
    "CA_C2_jena_K6_s92605": "COMPLETE",
    "CA_C2_jena_K6_s92606": "COMPLETE",
    "CA_C2_jena_K6_s92607": "COMPLETE",
    "CA_C2_jena_K6_s92608": "COMPLETE",
    "CA_C2_jena_K6_s92609": "COMPLETE",
    "CA_C2_jena_K6_s92610": "COMPLETE",
    "CA_C3_ETTh1_K4_s92601": "COMPLETE",
    "CA_C3_ETTh1_K4_s92602": "COMPLETE",
    "CA_C3_ETTh1_K4_s92603": "COMPLETE",
    "CA_C3_ETTh1_K4_s92604": "COMPLETE",
    "CA_C3_ETTh1_K4_s92605": "COMPLETE",
    "CA_C3_ETTh1_K4_s92606": "COMPLETE",
    "CA_C3_ETTh1_K4_s92607": "COMPLETE",
    "CA_C3_ETTh1_K4_s92608": "COMPLETE",
    "CA_C3_ETTh1_K4_s92609": "COMPLETE",
    "CA_C3_ETTh1_K4_s92610": "COMPLETE",
    "CA_C3_ETTh2_safe17400_K4_s92601": "COMPLETE",
    "CA_C3_ETTh2_safe17400_K4_s92602": "COMPLETE",
    "CA_C3_ETTh2_safe17400_K4_s92603": "COMPLETE",
    "CA_C3_ETTh2_safe17400_K4_s92604": "COMPLETE",
    "CA_C3_ETTh2_safe17400_K4_s92605": "COMPLETE",
    "CA_C3_ETTh2_safe17400_K4_s92606": "COMPLETE",
    "CA_C3_ETTh2_safe17400_K4_s92607": "COMPLETE",
    "CA_C3_ETTh2_safe17400_K4_s92608": "COMPLETE",
    "CA_C3_ETTh2_safe17400_K4_s92609": "COMPLETE",
    "CA_C3_ETTh2_safe17400_K4_s92610": "COMPLETE",
    "CA_C3_appliances_K10_s92601": "COMPLETE",
    "CA_C3_appliances_K10_s92602": "COMPLETE",
    "CA_C3_appliances_K10_s92603": "COMPLETE",
    "CA_C3_appliances_K10_s92604": "COMPLETE",
    "CA_C3_appliances_K10_s92605": "COMPLETE",
    "CA_C3_appliances_K10_s92606": "COMPLETE",
    "CA_C3_appliances_K10_s92607": "COMPLETE",
    "CA_C3_appliances_K10_s92608": "COMPLETE",
    "CA_C3_appliances_K10_s92609": "COMPLETE",
    "CA_C3_appliances_K10_s92610": "COMPLETE",
    "CA_C3_jena_K6_s92601": "COMPLETE",
    "CA_C3_jena_K6_s92602": "COMPLETE",
    "CA_C3_jena_K6_s92603": "COMPLETE",
    "CA_C3_jena_K6_s92604": "COMPLETE",
    "CA_C3_jena_K6_s92605": "COMPLETE",
    "CA_C3_jena_K6_s92606": "COMPLETE",
    "CA_C3_jena_K6_s92607": "COMPLETE",
    "CA_C3_jena_K6_s92608": "COMPLETE",
    "CA_C3_jena_K6_s92609": "COMPLETE",
    "CA_C3_jena_K6_s92610": "COMPLETE"
  },
  "started_utc": "2026-10-25T16:28:26.567108+09:00",
  "status": "PARTIAL",
  "stop_reason": "Budget, deadline or recorded family/job failure; see run_statuses."
}
```

그림: FIG_COMPARATIVE.png/.svg, FIG_ABLATION.png/.svg, FIG_ERROR_QUANTILES.png/.svg. 각 그림의 같은 이름 CSV에 긴 형식 데이터를 제공한다.

V MSE is the mean across original channels of squared forecast errors over observed validation targets, measured after normalization using statistics fitted only on the A60 training interval.

[확인] 근거: research/informs_broad_validation_20261015_v1/evaluation.py:39–61; data_contract.py:61–76; fit_base.py:46–49. V는 개발 검증 구간이며 보호 테스트 T가 아니다.

