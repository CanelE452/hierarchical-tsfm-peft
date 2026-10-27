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
jena                A1      10   0.775362        0.017211     3
jena                A2      10   0.723501        0.028942     2
jena                A3      0    NA              NA           0
jena                A4      10   0.685574        0.051290     4
jena                A5      0    NA              NA           0
ETTh1               RAW     1    0.283405        0.000000     0
ETTh1               C1      1    0.715743        0.000000     0
ETTh1               C2      10   0.678325        0.040459     5
ETTh1               C3      10   0.666153        0.058611     5
ETTh1               OURS    10   0.279643        0.005426     0
ETTh1               A1      10   0.279226        0.003130     3
ETTh1               A2      10   0.277454        0.003690     3
ETTh1               A3      0    NA              NA           0
ETTh1               A4      10   0.715743        0.000000     0
ETTh1               A5      0    NA              NA           0
ETTh2_safe17400     RAW     1    0.259555        0.000000     0
ETTh2_safe17400     C1      1    0.901934        0.000000     0
ETTh2_safe17400     C2      10   1.091104        0.102803     6
ETTh2_safe17400     C3      10   1.137333        0.109572     6
ETTh2_safe17400     OURS    10   0.277231        0.021597     2
ETTh2_safe17400     A1      10   0.263365        0.003370     10
ETTh2_safe17400     A2      10   0.278526        0.026056     0
ETTh2_safe17400     A3      0    NA              NA           0
ETTh2_safe17400     A4      10   0.901934        0.033964     0
ETTh2_safe17400     A5      0    NA              NA           0
appliances          RAW     1    0.347579        0.000000     0
appliances          C1      1    0.417315        0.000000     0
appliances          C2      10   0.324931        0.009727     3
appliances          C3      10   0.332662        0.004946     6
appliances          OURS    10   0.282019        0.006378     0
appliances          A1      10   0.283612        0.003923     1
appliances          A2      10   0.277510        0.003301     0
appliances          A3      0    NA              NA           0
appliances          A4      10   0.334393        0.003928     0
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
A4      2            0      1        0
A5      0            0      0        3
```

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
  "attempts": {
    "CA_A1_jena_K6_s92601_ext10": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793078434.4007125,
        "job_id": "CA_A1_jena_K6_s92601_ext10",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A1_jena_K6_s92601_ext10_attempt1.log",
        "pid": 14028,
        "result_present": true,
        "started_unix": 1793077492.2459118,
        "wall_seconds": 942.1548006534576,
        "worker": {
          "admitted_unix": 1793077501.3539047,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793078433.2659817,
          "job_id": "CA_A1_jena_K6_s92601_ext10",
          "status": "COMPLETE",
          "wall_seconds": 931.9120769500732
        }
      }
    ],
    "CA_A1_jena_K6_s92602_ext10": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793079645.6321638,
        "job_id": "CA_A1_jena_K6_s92602_ext10",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A1_jena_K6_s92602_ext10_attempt1.log",
        "pid": 45168,
        "result_present": true,
        "started_unix": 1793078446.8983238,
        "wall_seconds": 1198.7338399887085,
        "worker": {
          "admitted_unix": 1793078455.321791,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793079644.1597803,
          "job_id": "CA_A1_jena_K6_s92602_ext10",
          "status": "COMPLETE",
          "wall_seconds": 1188.8379893302917
        }
      }
    ],
    "CA_A1_jena_K6_s92603_ext10": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793080844.8075545,
        "job_id": "CA_A1_jena_K6_s92603_ext10",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A1_jena_K6_s92603_ext10_attempt1.log",
        "pid": 52188,
        "result_present": true,
        "started_unix": 1793079658.1186025,
        "wall_seconds": 1186.6889519691467,
        "worker": {
          "admitted_unix": 1793079666.8976994,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793080844.0262518,
          "job_id": "CA_A1_jena_K6_s92603_ext10",
          "status": "COMPLETE",
          "wall_seconds": 1177.1285524368286
        }
      }
    ],
    "CA_A1_jena_K6_s92604_ext10": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793082033.9120677,
        "job_id": "CA_A1_jena_K6_s92604_ext10",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A1_jena_K6_s92604_ext10_attempt1.log",
        "pid": 46372,
        "result_present": true,
        "started_unix": 1793080857.2758183,
        "wall_seconds": 1176.6362493038177,
        "worker": {
          "admitted_unix": 1793080865.9292495,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793082032.6562574,
          "job_id": "CA_A1_jena_K6_s92604_ext10",
          "status": "COMPLETE",
          "wall_seconds": 1166.7270078659058
        }
      }
    ],
    "CA_A1_jena_K6_s92605_ext10": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793083227.155368,
        "job_id": "CA_A1_jena_K6_s92605_ext10",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A1_jena_K6_s92605_ext10_attempt1.log",
        "pid": 24872,
        "result_present": true,
        "started_unix": 1793082046.4123611,
        "wall_seconds": 1180.7430069446564,
        "worker": {
          "admitted_unix": 1793082054.958732,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793083224.658389,
          "job_id": "CA_A1_jena_K6_s92605_ext10",
          "status": "COMPLETE",
          "wall_seconds": 1169.699657201767
        }
      }
    ],
    "CA_A1_jena_K6_s92606_ext10": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793084390.2427487,
        "job_id": "CA_A1_jena_K6_s92606_ext10",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A1_jena_K6_s92606_ext10_attempt1.log",
        "pid": 44504,
        "result_present": true,
        "started_unix": 1793083239.5293167,
        "wall_seconds": 1150.7134320735931,
        "worker": {
          "admitted_unix": 1793083248.0945086,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793084389.1351333,
          "job_id": "CA_A1_jena_K6_s92606_ext10",
          "status": "COMPLETE",
          "wall_seconds": 1141.0406246185303
        }
      }
    ],
    "CA_A1_jena_K6_s92607_ext10": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793085902.0591815,
        "job_id": "CA_A1_jena_K6_s92607_ext10",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A1_jena_K6_s92607_ext10_attempt1.log",
        "pid": 44160,
        "result_present": true,
        "started_unix": 1793084402.5569499,
        "wall_seconds": 1499.5022315979004,
        "worker": {
          "admitted_unix": 1793084410.9930549,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793085901.0373755,
          "job_id": "CA_A1_jena_K6_s92607_ext10",
          "status": "COMPLETE",
          "wall_seconds": 1490.0443205833435
        }
      }
    ],
    "CA_A1_jena_K6_s92608_ext10": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793087417.9922717,
        "job_id": "CA_A1_jena_K6_s92608_ext10",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A1_jena_K6_s92608_ext10_attempt1.log",
        "pid": 52508,
        "result_present": true,
        "started_unix": 1793085914.4809327,
        "wall_seconds": 1503.5113389492035,
        "worker": {
          "admitted_unix": 1793085923.0108774,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793087415.7263365,
          "job_id": "CA_A1_jena_K6_s92608_ext10",
          "status": "COMPLETE",
          "wall_seconds": 1492.7154591083527
        }
      }
    ],
    "CA_A1_jena_K6_s92609_ext10": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793088362.4365318,
        "job_id": "CA_A1_jena_K6_s92609_ext10",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A1_jena_K6_s92609_ext10_attempt1.log",
        "pid": 44004,
        "result_present": true,
        "started_unix": 1793087430.2723055,
        "wall_seconds": 932.1642262935638,
        "worker": {
          "admitted_unix": 1793087438.567382,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793088359.9607217,
          "job_id": "CA_A1_jena_K6_s92609_ext10",
          "status": "COMPLETE",
          "wall_seconds": 921.3933396339417
        }
      }
    ],
    "CA_A1_jena_K6_s92610_ext10": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793089882.5167453,
        "job_id": "CA_A1_jena_K6_s92610_ext10",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A1_jena_K6_s92610_ext10_attempt1.log",
        "pid": 26052,
        "result_present": true,
        "started_unix": 1793088374.923034,
        "wall_seconds": 1507.5937113761902,
        "worker": {
          "admitted_unix": 1793088383.471162,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793089880.04407,
          "job_id": "CA_A1_jena_K6_s92610_ext10",
          "status": "COMPLETE",
          "wall_seconds": 1496.572907924652
        }
      }
    ],
    "CA_A4_ETTh1_K4_s92601": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793066765.6555681,
        "job_id": "CA_A4_ETTh1_K4_s92601",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_ETTh1_K4_s92601_attempt1.log",
        "pid": 41592,
        "result_present": true,
        "started_unix": 1793066561.2587698,
        "wall_seconds": 204.39679837226868,
        "worker": {
          "admitted_unix": 1793066570.6248338,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793066763.386352,
          "job_id": "CA_A4_ETTh1_K4_s92601",
          "status": "COMPLETE",
          "wall_seconds": 192.76151823997498
        }
      }
    ],
    "CA_A4_ETTh1_K4_s92602": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793067887.8566086,
        "job_id": "CA_A4_ETTh1_K4_s92602",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_ETTh1_K4_s92602_attempt1.log",
        "pid": 29192,
        "result_present": true,
        "started_unix": 1793067691.4857123,
        "wall_seconds": 196.3708963394165,
        "worker": {
          "admitted_unix": 1793067700.404107,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793067886.992488,
          "job_id": "CA_A4_ETTh1_K4_s92602",
          "status": "COMPLETE",
          "wall_seconds": 186.58838081359863
        }
      }
    ],
    "CA_A4_ETTh1_K4_s92603": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793068916.066854,
        "job_id": "CA_A4_ETTh1_K4_s92603",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_ETTh1_K4_s92603_attempt1.log",
        "pid": 41352,
        "result_present": true,
        "started_unix": 1793068721.6750066,
        "wall_seconds": 194.39184737205505,
        "worker": {
          "admitted_unix": 1793068730.4533322,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793068914.9910667,
          "job_id": "CA_A4_ETTh1_K4_s92603",
          "status": "COMPLETE",
          "wall_seconds": 184.5377345085144
        }
      }
    ],
    "CA_A4_ETTh1_K4_s92604": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793070102.5786858,
        "job_id": "CA_A4_ETTh1_K4_s92604",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_ETTh1_K4_s92604_attempt1.log",
        "pid": 24264,
        "result_present": true,
        "started_unix": 1793069906.1918232,
        "wall_seconds": 196.3868625164032,
        "worker": {
          "admitted_unix": 1793069914.8726707,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793070100.6381788,
          "job_id": "CA_A4_ETTh1_K4_s92604",
          "status": "COMPLETE",
          "wall_seconds": 185.76550817489624
        }
      }
    ],
    "CA_A4_ETTh1_K4_s92605": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793071123.310823,
        "job_id": "CA_A4_ETTh1_K4_s92605",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_ETTh1_K4_s92605_attempt1.log",
        "pid": 49584,
        "result_present": true,
        "started_unix": 1793070924.895987,
        "wall_seconds": 198.4148359298706,
        "worker": {
          "admitted_unix": 1793070934.0668173,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793071120.9514081,
          "job_id": "CA_A4_ETTh1_K4_s92605",
          "status": "COMPLETE",
          "wall_seconds": 186.88459086418152
        }
      }
    ],
    "CA_A4_ETTh1_K4_s92606": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793072414.410965,
        "job_id": "CA_A4_ETTh1_K4_s92606",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_ETTh1_K4_s92606_attempt1.log",
        "pid": 46492,
        "result_present": true,
        "started_unix": 1793072215.9806662,
        "wall_seconds": 198.43029880523682,
        "worker": {
          "admitted_unix": 1793072225.151275,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793072413.0687938,
          "job_id": "CA_A4_ETTh1_K4_s92606",
          "status": "COMPLETE",
          "wall_seconds": 187.91751885414124
        }
      }
    ],
    "CA_A4_ETTh1_K4_s92607": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793073438.6548207,
        "job_id": "CA_A4_ETTh1_K4_s92607",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_ETTh1_K4_s92607_attempt1.log",
        "pid": 41608,
        "result_present": true,
        "started_unix": 1793073240.2149,
        "wall_seconds": 198.43992066383362,
        "worker": {
          "admitted_unix": 1793073249.3866718,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793073436.2779608,
          "job_id": "CA_A4_ETTh1_K4_s92607",
          "status": "COMPLETE",
          "wall_seconds": 186.8912889957428
        }
      }
    ],
    "CA_A4_ETTh1_K4_s92608": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793074527.6077154,
        "job_id": "CA_A4_ETTh1_K4_s92608",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_ETTh1_K4_s92608_attempt1.log",
        "pid": 29236,
        "result_present": true,
        "started_unix": 1793074331.1753454,
        "wall_seconds": 196.43236994743347,
        "worker": {
          "admitted_unix": 1793074340.1808503,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793074525.0648634,
          "job_id": "CA_A4_ETTh1_K4_s92608",
          "status": "COMPLETE",
          "wall_seconds": 184.88401317596436
        }
      }
    ],
    "CA_A4_ETTh1_K4_s92609": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793075733.5630467,
        "job_id": "CA_A4_ETTh1_K4_s92609",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_ETTh1_K4_s92609_attempt1.log",
        "pid": 23592,
        "result_present": true,
        "started_unix": 1793075547.1382542,
        "wall_seconds": 186.42479252815247,
        "worker": {
          "admitted_unix": 1793075555.7684813,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793075732.4209082,
          "job_id": "CA_A4_ETTh1_K4_s92609",
          "status": "COMPLETE",
          "wall_seconds": 176.6524269580841
        }
      }
    ],
    "CA_A4_ETTh1_K4_s92610": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793076936.114631,
        "job_id": "CA_A4_ETTh1_K4_s92610",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_ETTh1_K4_s92610_attempt1.log",
        "pid": 19292,
        "result_present": true,
        "started_unix": 1793076747.6887693,
        "wall_seconds": 188.42586159706116,
        "worker": {
          "admitted_unix": 1793076756.3311205,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793076934.549067,
          "job_id": "CA_A4_ETTh1_K4_s92610",
          "status": "COMPLETE",
          "wall_seconds": 178.21794652938843
        }
      }
    ],
    "CA_A4_ETTh2_safe17400_K4_s92601": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793066999.6301053,
        "job_id": "CA_A4_ETTh2_safe17400_K4_s92601",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_ETTh2_safe17400_K4_s92601_attempt1.log",
        "pid": 9964,
        "result_present": true,
        "started_unix": 1793066779.1926966,
        "wall_seconds": 220.4374086856842,
        "worker": {
          "admitted_unix": 1793066788.6272664,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793066997.3488379,
          "job_id": "CA_A4_ETTh2_safe17400_K4_s92601",
          "status": "COMPLETE",
          "wall_seconds": 208.7215714454651
        }
      }
    ],
    "CA_A4_ETTh2_safe17400_K4_s92602": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793068111.2283976,
        "job_id": "CA_A4_ETTh2_safe17400_K4_s92602",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_ETTh2_safe17400_K4_s92602_attempt1.log",
        "pid": 26844,
        "result_present": true,
        "started_unix": 1793067900.8048935,
        "wall_seconds": 210.423504114151,
        "worker": {
          "admitted_unix": 1793067909.9140892,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793068109.6513357,
          "job_id": "CA_A4_ETTh2_safe17400_K4_s92602",
          "status": "COMPLETE",
          "wall_seconds": 199.7372465133667
        }
      }
    ],
    "CA_A4_ETTh2_safe17400_K4_s92603": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793069123.6038096,
        "job_id": "CA_A4_ETTh2_safe17400_K4_s92603",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_ETTh2_safe17400_K4_s92603_attempt1.log",
        "pid": 44884,
        "result_present": true,
        "started_unix": 1793068929.2139275,
        "wall_seconds": 194.38988208770752,
        "worker": {
          "admitted_unix": 1793068938.2993984,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793069122.4068053,
          "job_id": "CA_A4_ETTh2_safe17400_K4_s92603",
          "status": "COMPLETE",
          "wall_seconds": 184.10740685462952
        }
      }
    ],
    "CA_A4_ETTh2_safe17400_K4_s92604": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793070311.9360602,
        "job_id": "CA_A4_ETTh2_safe17400_K4_s92604",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_ETTh2_safe17400_K4_s92604_attempt1.log",
        "pid": 40996,
        "result_present": true,
        "started_unix": 1793070115.5407925,
        "wall_seconds": 196.39526772499084,
        "worker": {
          "admitted_unix": 1793070124.7282412,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793070309.215113,
          "job_id": "CA_A4_ETTh2_safe17400_K4_s92604",
          "status": "COMPLETE",
          "wall_seconds": 184.48687171936035
        }
      }
    ],
    "CA_A4_ETTh2_safe17400_K4_s92605": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793071352.7893634,
        "job_id": "CA_A4_ETTh2_safe17400_K4_s92605",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_ETTh2_safe17400_K4_s92605_attempt1.log",
        "pid": 51252,
        "result_present": true,
        "started_unix": 1793071136.3268712,
        "wall_seconds": 216.46249222755432,
        "worker": {
          "admitted_unix": 1793071145.5405836,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793071350.769173,
          "job_id": "CA_A4_ETTh2_safe17400_K4_s92605",
          "status": "COMPLETE",
          "wall_seconds": 205.22858929634094
        }
      }
    ],
    "CA_A4_ETTh2_safe17400_K4_s92606": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793072624.026139,
        "job_id": "CA_A4_ETTh2_safe17400_K4_s92606",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_ETTh2_safe17400_K4_s92606_attempt1.log",
        "pid": 40964,
        "result_present": true,
        "started_unix": 1793072427.599324,
        "wall_seconds": 196.42681503295898,
        "worker": {
          "admitted_unix": 1793072436.5421603,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793072622.170735,
          "job_id": "CA_A4_ETTh2_safe17400_K4_s92606",
          "status": "COMPLETE",
          "wall_seconds": 185.62857460975647
        }
      }
    ],
    "CA_A4_ETTh2_safe17400_K4_s92607": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793073648.1816747,
        "job_id": "CA_A4_ETTh2_safe17400_K4_s92607",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_ETTh2_safe17400_K4_s92607_attempt1.log",
        "pid": 34100,
        "result_present": true,
        "started_unix": 1793073451.7604206,
        "wall_seconds": 196.42125415802002,
        "worker": {
          "admitted_unix": 1793073460.7854402,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793073645.7488043,
          "job_id": "CA_A4_ETTh2_safe17400_K4_s92607",
          "status": "COMPLETE",
          "wall_seconds": 184.9633641242981
        }
      }
    ],
    "CA_A4_ETTh2_safe17400_K4_s92608": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793074733.1878881,
        "job_id": "CA_A4_ETTh2_safe17400_K4_s92608",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_ETTh2_safe17400_K4_s92608_attempt1.log",
        "pid": 43824,
        "result_present": true,
        "started_unix": 1793074540.747633,
        "wall_seconds": 192.4402551651001,
        "worker": {
          "admitted_unix": 1793074549.7085433,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793074732.1723835,
          "job_id": "CA_A4_ETTh2_safe17400_K4_s92608",
          "status": "COMPLETE",
          "wall_seconds": 182.46384024620056
        }
      }
    ],
    "CA_A4_ETTh2_safe17400_K4_s92609": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793075932.5359333,
        "job_id": "CA_A4_ETTh2_safe17400_K4_s92609",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_ETTh2_safe17400_K4_s92609_attempt1.log",
        "pid": 43800,
        "result_present": true,
        "started_unix": 1793075746.118111,
        "wall_seconds": 186.41782236099243,
        "worker": {
          "admitted_unix": 1793075754.6679397,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793075930.6225739,
          "job_id": "CA_A4_ETTh2_safe17400_K4_s92609",
          "status": "COMPLETE",
          "wall_seconds": 175.9546341896057
        }
      }
    ],
    "CA_A4_ETTh2_safe17400_K4_s92610": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793077136.9167838,
        "job_id": "CA_A4_ETTh2_safe17400_K4_s92610",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_ETTh2_safe17400_K4_s92610_attempt1.log",
        "pid": 18644,
        "result_present": true,
        "started_unix": 1793076948.4786592,
        "wall_seconds": 188.43812465667725,
        "worker": {
          "admitted_unix": 1793076956.9429646,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793077134.7776012,
          "job_id": "CA_A4_ETTh2_safe17400_K4_s92610",
          "status": "COMPLETE",
          "wall_seconds": 177.83463668823242
        }
      }
    ],
    "CA_A4_appliances_K10_s92601": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793067348.6579175,
        "job_id": "CA_A4_appliances_K10_s92601",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_appliances_K10_s92601_attempt1.log",
        "pid": 42772,
        "result_present": true,
        "started_unix": 1793067013.993875,
        "wall_seconds": 334.66404247283936,
        "worker": {
          "admitted_unix": 1793067024.3066106,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793067347.0270934,
          "job_id": "CA_A4_appliances_K10_s92601",
          "status": "COMPLETE",
          "wall_seconds": 322.7204828262329
        }
      }
    ],
    "CA_A4_appliances_K10_s92602": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793068378.8635528,
        "job_id": "CA_A4_appliances_K10_s92602",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_appliances_K10_s92602_attempt1.log",
        "pid": 36520,
        "result_present": true,
        "started_unix": 1793068124.2786362,
        "wall_seconds": 254.5849165916443,
        "worker": {
          "admitted_unix": 1793068133.2554324,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793068377.1252017,
          "job_id": "CA_A4_appliances_K10_s92602",
          "status": "COMPLETE",
          "wall_seconds": 243.8697693347931
        }
      }
    ],
    "CA_A4_appliances_K10_s92603": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793069390.957174,
        "job_id": "CA_A4_appliances_K10_s92603",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_appliances_K10_s92603_attempt1.log",
        "pid": 4400,
        "result_present": true,
        "started_unix": 1793069136.456583,
        "wall_seconds": 254.5005910396576,
        "worker": {
          "admitted_unix": 1793069145.1866722,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793069388.825884,
          "job_id": "CA_A4_appliances_K10_s92603",
          "status": "COMPLETE",
          "wall_seconds": 243.63921189308167
        }
      }
    ],
    "CA_A4_appliances_K10_s92604": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793070581.5685933,
        "job_id": "CA_A4_appliances_K10_s92604",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_appliances_K10_s92604_attempt1.log",
        "pid": 43180,
        "result_present": true,
        "started_unix": 1793070325.0484748,
        "wall_seconds": 256.5201184749603,
        "worker": {
          "admitted_unix": 1793070333.8783066,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793070580.4396842,
          "job_id": "CA_A4_appliances_K10_s92604",
          "status": "COMPLETE",
          "wall_seconds": 246.5613775253296
        }
      }
    ],
    "CA_A4_appliances_K10_s92605": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793071692.928399,
        "job_id": "CA_A4_appliances_K10_s92605",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_appliances_K10_s92605_attempt1.log",
        "pid": 4556,
        "result_present": true,
        "started_unix": 1793071366.2377174,
        "wall_seconds": 326.6906816959381,
        "worker": {
          "admitted_unix": 1793071375.3916996,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793071690.5353537,
          "job_id": "CA_A4_appliances_K10_s92605",
          "status": "COMPLETE",
          "wall_seconds": 315.1436541080475
        }
      }
    ],
    "CA_A4_appliances_K10_s92606": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793072893.4836178,
        "job_id": "CA_A4_appliances_K10_s92606",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_appliances_K10_s92606_attempt1.log",
        "pid": 27352,
        "result_present": true,
        "started_unix": 1793072636.9291337,
        "wall_seconds": 256.554484128952,
        "worker": {
          "admitted_unix": 1793072645.9282892,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793072892.4153008,
          "job_id": "CA_A4_appliances_K10_s92606",
          "status": "COMPLETE",
          "wall_seconds": 246.48701167106628
        }
      }
    ],
    "CA_A4_appliances_K10_s92607": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793073984.037528,
        "job_id": "CA_A4_appliances_K10_s92607",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_appliances_K10_s92607_attempt1.log",
        "pid": 29876,
        "result_present": true,
        "started_unix": 1793073661.3233552,
        "wall_seconds": 322.7141728401184,
        "worker": {
          "admitted_unix": 1793073670.4982328,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793073982.4966347,
          "job_id": "CA_A4_appliances_K10_s92607",
          "status": "COMPLETE",
          "wall_seconds": 311.9984018802643
        }
      }
    ],
    "CA_A4_appliances_K10_s92608": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793075059.2079756,
        "job_id": "CA_A4_appliances_K10_s92608",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_appliances_K10_s92608_attempt1.log",
        "pid": 22316,
        "result_present": true,
        "started_unix": 1793074746.5229294,
        "wall_seconds": 312.6850461959839,
        "worker": {
          "admitted_unix": 1793074755.3447251,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793075058.26442,
          "job_id": "CA_A4_appliances_K10_s92608",
          "status": "COMPLETE",
          "wall_seconds": 302.9196949005127
        }
      }
    ],
    "CA_A4_appliances_K10_s92609": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793076255.66069,
        "job_id": "CA_A4_appliances_K10_s92609",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_appliances_K10_s92609_attempt1.log",
        "pid": 24456,
        "result_present": true,
        "started_unix": 1793075944.9625466,
        "wall_seconds": 310.69814348220825,
        "worker": {
          "admitted_unix": 1793075953.4999893,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793076253.8608077,
          "job_id": "CA_A4_appliances_K10_s92609",
          "status": "COMPLETE",
          "wall_seconds": 300.3608183860779
        }
      }
    ],
    "CA_A4_appliances_K10_s92610": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793077460.376863,
        "job_id": "CA_A4_appliances_K10_s92610",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_appliances_K10_s92610_attempt1.log",
        "pid": 47572,
        "result_present": true,
        "started_unix": 1793077149.549199,
        "wall_seconds": 310.827663898468,
        "worker": {
          "admitted_unix": 1793077158.1911592,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793077459.1792386,
          "job_id": "CA_A4_appliances_K10_s92610",
          "status": "COMPLETE",
          "wall_seconds": 300.9880793094635
        }
      }
    ],
    "CA_A4_jena_K6_s92601": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793066548.1993299,
        "job_id": "CA_A4_jena_K6_s92601",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_jena_K6_s92601_attempt1.log",
        "pid": 35312,
        "result_present": true,
        "started_unix": 1793066215.5894647,
        "wall_seconds": 332.60986518859863,
        "worker": {
          "admitted_unix": 1793066224.8680668,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793066546.1597612,
          "job_id": "CA_A4_jena_K6_s92601",
          "status": "COMPLETE",
          "wall_seconds": 321.2916944026947
        }
      }
    ],
    "CA_A4_jena_K6_s92602": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793067678.2824273,
        "job_id": "CA_A4_jena_K6_s92602",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_jena_K6_s92602_attempt1.log",
        "pid": 29044,
        "result_present": true,
        "started_unix": 1793067361.6836734,
        "wall_seconds": 316.5987539291382,
        "worker": {
          "admitted_unix": 1793067370.612273,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793067677.293151,
          "job_id": "CA_A4_jena_K6_s92602",
          "status": "COMPLETE",
          "wall_seconds": 306.68087792396545
        }
      }
    ],
    "CA_A4_jena_K6_s92603": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793068708.7146068,
        "job_id": "CA_A4_jena_K6_s92603",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_jena_K6_s92603_attempt1.log",
        "pid": 46504,
        "result_present": true,
        "started_unix": 1793068392.1000285,
        "wall_seconds": 316.6145782470703,
        "worker": {
          "admitted_unix": 1793068401.1257627,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793068706.655206,
          "job_id": "CA_A4_jena_K6_s92603",
          "status": "COMPLETE",
          "wall_seconds": 305.52944326400757
        }
      }
    ],
    "CA_A4_jena_K6_s92604": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793069893.0854576,
        "job_id": "CA_A4_jena_K6_s92604",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_jena_K6_s92604_attempt1.log",
        "pid": 43968,
        "result_present": true,
        "started_unix": 1793069404.1033926,
        "wall_seconds": 488.9820649623871,
        "worker": {
          "admitted_unix": 1793069413.0230937,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793069890.6731808,
          "job_id": "CA_A4_jena_K6_s92604",
          "status": "COMPLETE",
          "wall_seconds": 477.6500871181488
        }
      }
    ],
    "CA_A4_jena_K6_s92605": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793070911.2903395,
        "job_id": "CA_A4_jena_K6_s92605",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_jena_K6_s92605_attempt1.log",
        "pid": 39356,
        "result_present": true,
        "started_unix": 1793070594.5482101,
        "wall_seconds": 316.7421293258667,
        "worker": {
          "admitted_unix": 1793070603.5996926,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793070908.6334527,
          "job_id": "CA_A4_jena_K6_s92605",
          "status": "COMPLETE",
          "wall_seconds": 305.0337600708008
        }
      }
    ],
    "CA_A4_jena_K6_s92606": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793072202.9574986,
        "job_id": "CA_A4_jena_K6_s92606",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_jena_K6_s92606_attempt1.log",
        "pid": 10044,
        "result_present": true,
        "started_unix": 1793071705.9332654,
        "wall_seconds": 497.02423310279846,
        "worker": {
          "admitted_unix": 1793071715.0976791,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793072201.4049041,
          "job_id": "CA_A4_jena_K6_s92606",
          "status": "COMPLETE",
          "wall_seconds": 486.3072249889374
        }
      }
    ],
    "CA_A4_jena_K6_s92607": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793073227.114318,
        "job_id": "CA_A4_jena_K6_s92607",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_jena_K6_s92607_attempt1.log",
        "pid": 39432,
        "result_present": true,
        "started_unix": 1793072906.4181604,
        "wall_seconds": 320.69615745544434,
        "worker": {
          "admitted_unix": 1793072915.358912,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793073225.5865927,
          "job_id": "CA_A4_jena_K6_s92607",
          "status": "COMPLETE",
          "wall_seconds": 310.227680683136
        }
      }
    ],
    "CA_A4_jena_K6_s92608": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793074317.864744,
        "job_id": "CA_A4_jena_K6_s92608",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_jena_K6_s92608_attempt1.log",
        "pid": 48096,
        "result_present": true,
        "started_unix": 1793073997.1776083,
        "wall_seconds": 320.68713569641113,
        "worker": {
          "admitted_unix": 1793074006.1213071,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793074315.5832875,
          "job_id": "CA_A4_jena_K6_s92608",
          "status": "COMPLETE",
          "wall_seconds": 309.461980342865
        }
      }
    ],
    "CA_A4_jena_K6_s92609": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793075534.6072574,
        "job_id": "CA_A4_jena_K6_s92609",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_jena_K6_s92609_attempt1.log",
        "pid": 30800,
        "result_present": true,
        "started_unix": 1793075071.5780027,
        "wall_seconds": 463.0292546749115,
        "worker": {
          "admitted_unix": 1793075080.124491,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793075533.476392,
          "job_id": "CA_A4_jena_K6_s92609",
          "status": "COMPLETE",
          "wall_seconds": 453.3519010543823
        }
      }
    ],
    "CA_A4_jena_K6_s92610": [
      {
        "attempt": 1,
        "exit_code": 0,
        "finished_unix": 1793076735.2317595,
        "job_id": "CA_A4_jena_K6_s92610",
        "log": "C:\\Users\\User\\.codex\\worktrees\\comparative-ablation-20261025\\hierarchical-tsfm-peft\\.cache\\comparative_ablation_followup_20261027_v1\\logs\\CA_A4_jena_K6_s92610_attempt1.log",
        "pid": 9012,
        "result_present": true,
        "started_unix": 1793076268.1796901,
        "wall_seconds": 467.0520694255829,
        "worker": {
          "admitted_unix": 1793076276.7191126,
          "attempt": 1,
          "exit_code": 0,
          "finished_unix": 1793076732.675598,
          "job_id": "CA_A4_jena_K6_s92610",
          "status": "COMPLETE",
          "wall_seconds": 455.95648527145386
        }
      }
    ]
  },
  "budget": {
    "a4_jobs_remaining": 36,
    "admission_remaining_seconds": 45676.20914244652,
    "estimated_a4_finish_utc": "2026-10-27T04:59:37.631957+00:00",
    "estimated_a4_seconds_remaining": 9828.973032474518,
    "initial": {
      "A1_jena_10_estimated_seconds": 25133.31863030691,
      "A4_40_estimated_seconds": 22471.50537967682,
      "a1_jena_extension_seconds_per_job": 2513.331863030691,
      "all10_admission_rule": "No partial seed selection. If all 10 are not expected to finish, all ten are NOT_RUN_BUDGET.",
      "all_50_estimated_seconds": 47604.82400998373,
      "basis": "Original campaign actual RUNS stage times and RESULT elapsed_seconds; no followup outcomes used.",
      "extension_components": {
        "evaluation": 0.18721904001431539,
        "overhead_mean": 8.09306862388039,
        "refit_per_epoch": 136.55930010000787,
        "search_per_epoch": 113.9458574366717
      },
      "extension_formula": "mean(search_seconds/training_cap)*10 + mean(refit_seconds/selected_epoch)*10 + mean(evaluation_seconds) + mean(non-stage elapsed overhead); conservatively assumes refit at epoch 10",
      "interpretation": "Planner's ~11.3h arithmetic uses older K_tau profile for extension and omits per-job non-stage overhead. Actual first-campaign measured profile is used for admission. Extension is decided only after A4 and requires all 10 seeds to fit the remaining wall-clock/deadline window.",
      "latest_start": "2026-10-28T00:00:00+09:00",
      "limit_seconds": 46800,
      "per_dataset_a4_seconds": {
        "ETTh1": 528.3225228786469,
        "ETTh2_safe17400": 525.0935826897621,
        "appliances": 469.1060230255127,
        "jena": 724.6284093737602
      },
      "planner_legacy_profile_extension_seconds": 1841.6267243332309,
      "planner_legacy_profile_total_seconds": 40528.009423182426,
      "profiles": {
        "ETTh1": {
          "job_ids": [
            "CA_A1_ETTh1_K4_s92601",
            "CA_A1_ETTh1_K4_s92602",
            "CA_A1_ETTh1_K4_s92603",
            "CA_A1_ETTh1_K4_s92604",
            "CA_A1_ETTh1_K4_s92605",
            "CA_A1_ETTh1_K4_s92606",
            "CA_A1_ETTh1_K4_s92607",
            "CA_A1_ETTh1_K4_s92608",
            "CA_A1_ETTh1_K4_s92609",
            "CA_A1_ETTh1_K4_s92610",
            "CA_A2_ETTh1_K4_s92601",
            "CA_A2_ETTh1_K4_s92602",
            "CA_A2_ETTh1_K4_s92603",
            "CA_A2_ETTh1_K4_s92604",
            "CA_A2_ETTh1_K4_s92605",
            "CA_A2_ETTh1_K4_s92606",
            "CA_A2_ETTh1_K4_s92607",
            "CA_A2_ETTh1_K4_s92608",
            "CA_A2_ETTh1_K4_s92609",
            "CA_A2_ETTh1_K4_s92610"
          ],
          "max_job_elapsed_seconds": 801.7584314346313,
          "mean_job_elapsed_seconds": 528.3225228786469,
          "mean_stage_seconds": 520.2374284999853,
          "n": 20,
          "proxy_methods": [
            "A1",
            "A2"
          ]
        },
        "ETTh2_safe17400": {
          "job_ids": [
            "CA_A1_ETTh2_safe17400_K4_s92601",
            "CA_A1_ETTh2_safe17400_K4_s92602",
            "CA_A1_ETTh2_safe17400_K4_s92603",
            "CA_A1_ETTh2_safe17400_K4_s92604",
            "CA_A1_ETTh2_safe17400_K4_s92605",
            "CA_A1_ETTh2_safe17400_K4_s92606",
            "CA_A1_ETTh2_safe17400_K4_s92607",
            "CA_A1_ETTh2_safe17400_K4_s92608",
            "CA_A1_ETTh2_safe17400_K4_s92609",
            "CA_A1_ETTh2_safe17400_K4_s92610",
            "CA_A2_ETTh2_safe17400_K4_s92601",
            "CA_A2_ETTh2_safe17400_K4_s92602",
            "CA_A2_ETTh2_safe17400_K4_s92603",
            "CA_A2_ETTh2_safe17400_K4_s92604",
            "CA_A2_ETTh2_safe17400_K4_s92605",
            "CA_A2_ETTh2_safe17400_K4_s92606",
            "CA_A2_ETTh2_safe17400_K4_s92607",
            "CA_A2_ETTh2_safe17400_K4_s92608",
            "CA_A2_ETTh2_safe17400_K4_s92609",
            "CA_A2_ETTh2_safe17400_K4_s92610"
          ],
          "max_job_elapsed_seconds": 822.900025844574,
          "mean_job_elapsed_seconds": 525.0935826897621,
          "mean_stage_seconds": 516.9349884999799,
          "n": 20,
          "proxy_methods": [
            "A1",
            "A2"
          ]
        },
        "appliances": {
          "job_ids": [
            "CA_A1_appliances_K10_s92601",
            "CA_A1_appliances_K10_s92602",
            "CA_A1_appliances_K10_s92603",
            "CA_A1_appliances_K10_s92604",
            "CA_A1_appliances_K10_s92605",
            "CA_A1_appliances_K10_s92606",
            "CA_A1_appliances_K10_s92607",
            "CA_A1_appliances_K10_s92608",
            "CA_A1_appliances_K10_s92609",
            "CA_A1_appliances_K10_s92610",
            "CA_A2_appliances_K10_s92601",
            "CA_A2_appliances_K10_s92602",
            "CA_A2_appliances_K10_s92603",
            "CA_A2_appliances_K10_s92604",
            "CA_A2_appliances_K10_s92605",
            "CA_A2_appliances_K10_s92606",
            "CA_A2_appliances_K10_s92607",
            "CA_A2_appliances_K10_s92608",
            "CA_A2_appliances_K10_s92609",
            "CA_A2_appliances_K10_s92610"
          ],
          "max_job_elapsed_seconds": 596.5140025615692,
          "mean_job_elapsed_seconds": 469.1060230255127,
          "mean_stage_seconds": 459.74984484502346,
          "n": 20,
          "proxy_methods": [
            "A1",
            "A2"
          ]
        },
        "jena": {
          "job_ids": [
            "CA_A1_jena_K6_s92601",
            "CA_A1_jena_K6_s92602",
            "CA_A1_jena_K6_s92603",
            "CA_A1_jena_K6_s92604",
            "CA_A1_jena_K6_s92605",
            "CA_A1_jena_K6_s92606",
            "CA_A1_jena_K6_s92607",
            "CA_A1_jena_K6_s92608",
            "CA_A1_jena_K6_s92609",
            "CA_A1_jena_K6_s92610",
            "CA_A2_jena_K6_s92601",
            "CA_A2_jena_K6_s92602",
            "CA_A2_jena_K6_s92603",
            "CA_A2_jena_K6_s92604",
            "CA_A2_jena_K6_s92605",
            "CA_A2_jena_K6_s92606",
            "CA_A2_jena_K6_s92607",
            "CA_A2_jena_K6_s92608",
            "CA_A2_jena_K6_s92609",
            "CA_A2_jena_K6_s92610"
          ],
          "max_job_elapsed_seconds": 1001.3576462268829,
          "mean_job_elapsed_seconds": 724.6284093737602,
          "mean_stage_seconds": 714.2519561400229,
          "n": 20,
          "proxy_methods": [
            "A1",
            "A2"
          ]
        }
      },
      "status": "PASS_A4_WITHIN_LIMIT"
    },
    "measured_a4_per_dataset_seconds": {
      "ETTh1": 204.39679837226868,
      "ETTh2_safe17400": 220.4374086856842,
      "appliances": 334.66404247283936,
      "jena": 332.60986518859863
    }
  },
  "controller_pid": 42488,
  "elapsed_seconds": 23669.98355269432,
  "extension_status": {
    "A1/ETTh1": {
      "performed": false,
      "prior": {
        "cap_count": 3,
        "n": 10,
        "performed": false,
        "status": "NOT_TRIGGERED",
        "triggered": false
      },
      "status": "NOT_RUN_SCOPE"
    },
    "A1/ETTh2_safe17400": {
      "performed": false,
      "prior": {
        "cap_count": 10,
        "n": 10,
        "performed": false,
        "status": "NOT_RUN_BUDGET",
        "triggered": true
      },
      "status": "NOT_RUN_SCOPE"
    },
    "A1/appliances": {
      "performed": false,
      "prior": {
        "cap_count": 1,
        "n": 10,
        "performed": false,
        "status": "NOT_TRIGGERED",
        "triggered": false
      },
      "status": "NOT_RUN_SCOPE"
    },
    "A1/jena": {
      "cap10_count": 3,
      "cap_count": 10,
      "n": 10,
      "performed": true,
      "status": "EXTENDED",
      "triggered": true
    },
    "A2/ETTh1": {
      "performed": false,
      "prior": {
        "cap_count": 3,
        "n": 10,
        "performed": false,
        "status": "NOT_TRIGGERED",
        "triggered": false
      },
      "status": "NOT_RUN_SCOPE"
    },
    "A2/ETTh2_safe17400": {
      "performed": false,
      "prior": {
        "cap_count": 0,
        "n": 10,
        "performed": false,
        "status": "NOT_TRIGGERED",
        "triggered": false
      },
      "status": "NOT_RUN_SCOPE"
    },
    "A2/appliances": {
      "performed": false,
      "prior": {
        "cap_count": 0,
        "n": 10,
        "performed": false,
        "status": "NOT_TRIGGERED",
        "triggered": false
      },
      "status": "NOT_RUN_SCOPE"
    },
    "A2/jena": {
      "performed": false,
      "prior": {
        "cap_count": 2,
        "n": 10,
        "performed": false,
        "status": "NOT_TRIGGERED",
        "triggered": false
      },
      "status": "NOT_RUN_SCOPE"
    },
    "A3/ETTh1": {
      "performed": false,
      "prior": {
        "cap_count": 0,
        "n": 0,
        "performed": false,
        "status": "NOT_TRIGGERED",
        "triggered": false
      },
      "status": "NOT_RUN_SCOPE"
    },
    "A3/ETTh2_safe17400": {
      "performed": false,
      "prior": {
        "cap_count": 0,
        "n": 0,
        "performed": false,
        "status": "NOT_TRIGGERED",
        "triggered": false
      },
      "status": "NOT_RUN_SCOPE"
    },
    "A3/appliances": {
      "performed": false,
      "prior": {
        "cap_count": 0,
        "n": 0,
        "performed": false,
        "status": "NOT_TRIGGERED",
        "triggered": false
      },
      "status": "NOT_RUN_SCOPE"
    },
    "A3/jena": {
      "performed": false,
      "prior": {
        "cap_count": 0,
        "n": 0,
        "performed": false,
        "status": "NOT_TRIGGERED",
        "triggered": false
      },
      "status": "NOT_RUN_SCOPE"
    },
    "A4/ETTh1": {
      "cap_count": 0,
      "n": 10,
      "performed": false,
      "provisional": false,
      "status": "NOT_RUN_SCOPE",
      "triggered": false
    },
    "A4/ETTh2_safe17400": {
      "cap_count": 0,
      "n": 10,
      "performed": false,
      "provisional": false,
      "status": "NOT_RUN_SCOPE",
      "triggered": false
    },
    "A4/appliances": {
      "cap_count": 0,
      "n": 10,
      "performed": false,
      "provisional": false,
      "status": "NOT_RUN_SCOPE",
      "triggered": false
    },
    "A4/jena": {
      "cap_count": 4,
      "n": 10,
      "performed": false,
      "provisional": false,
      "status": "NOT_RUN_SCOPE",
      "triggered": false
    },
    "A5/ETTh1": {
      "performed": false,
      "prior": {
        "cap_count": 0,
        "n": 0,
        "performed": false,
        "status": "NOT_TRIGGERED",
        "triggered": false
      },
      "status": "NOT_RUN_SCOPE"
    },
    "A5/ETTh2_safe17400": {
      "performed": false,
      "prior": {
        "cap_count": 0,
        "n": 0,
        "performed": false,
        "status": "NOT_TRIGGERED",
        "triggered": false
      },
      "status": "NOT_RUN_SCOPE"
    },
    "A5/appliances": {
      "performed": false,
      "prior": {
        "cap_count": 0,
        "n": 0,
        "performed": false,
        "status": "NOT_TRIGGERED",
        "triggered": false
      },
      "status": "NOT_RUN_SCOPE"
    },
    "A5/jena": {
      "performed": false,
      "prior": {
        "cap_count": 0,
        "n": 0,
        "performed": false,
        "status": "NOT_TRIGGERED",
        "triggered": false
      },
      "status": "NOT_RUN_SCOPE"
    },
    "C2/ETTh1": {
      "performed": false,
      "prior": {
        "cap_count": 5,
        "n": 10,
        "performed": false,
        "status": "NOT_RUN_BUDGET",
        "triggered": true
      },
      "status": "NOT_RUN_SCOPE"
    },
    "C2/ETTh2_safe17400": {
      "performed": false,
      "prior": {
        "cap_count": 6,
        "n": 10,
        "performed": false,
        "status": "NOT_RUN_BUDGET",
        "triggered": true
      },
      "status": "NOT_RUN_SCOPE"
    },
    "C2/appliances": {
      "performed": false,
      "prior": {
        "cap_count": 3,
        "n": 10,
        "performed": false,
        "status": "NOT_TRIGGERED",
        "triggered": false
      },
      "status": "NOT_RUN_SCOPE"
    },
    "C2/jena": {
      "performed": false,
      "prior": {
        "cap_count": 9,
        "n": 10,
        "performed": false,
        "status": "NOT_RUN_BUDGET",
        "triggered": true
      },
      "status": "NOT_RUN_SCOPE"
    },
    "C3/ETTh1": {
      "performed": false,
      "prior": {
        "cap_count": 5,
        "n": 10,
        "performed": false,
        "status": "NOT_RUN_BUDGET",
        "triggered": true
      },
      "status": "NOT_RUN_SCOPE"
    },
    "C3/ETTh2_safe17400": {
      "performed": false,
      "prior": {
        "cap_count": 6,
        "n": 10,
        "performed": false,
        "status": "NOT_RUN_BUDGET",
        "triggered": true
      },
      "status": "NOT_RUN_SCOPE"
    },
    "C3/appliances": {
      "performed": false,
      "prior": {
        "cap_count": 6,
        "n": 10,
        "performed": false,
        "status": "NOT_RUN_BUDGET",
        "triggered": true
      },
      "status": "NOT_RUN_SCOPE"
    },
    "C3/jena": {
      "performed": false,
      "prior": {
        "cap_count": 8,
        "n": 10,
        "performed": false,
        "status": "NOT_RUN_BUDGET",
        "triggered": true
      },
      "status": "NOT_RUN_SCOPE"
    }
  },
  "first_a4_wall_seconds": {
    "ETTh1": 204.39679837226868,
    "ETTh2_safe17400": 220.4374086856842,
    "appliances": 334.66404247283936,
    "jena": 332.60986518859863
  },
  "last_start_cutoff": "2026-10-28T00:00:00+09:00",
  "limit_seconds": 46800,
  "prior_complete_job_ids": [
    "CA_C1_jena_K6_s92601",
    "CA_C1_ETTh1_K4_s92601",
    "CA_C1_ETTh2_safe17400_K4_s92601",
    "CA_C1_appliances_K10_s92601",
    "CA_C2_jena_K6_s92601",
    "CA_C2_ETTh2_safe17400_K4_s92601",
    "CA_C2_ETTh1_K4_s92601",
    "CA_C2_appliances_K10_s92601",
    "CA_C2_jena_K6_s92602",
    "CA_C2_ETTh1_K4_s92602",
    "CA_C2_ETTh2_safe17400_K4_s92602",
    "CA_C2_appliances_K10_s92602",
    "CA_C2_jena_K6_s92603",
    "CA_C2_ETTh1_K4_s92603",
    "CA_C2_ETTh2_safe17400_K4_s92603",
    "CA_C2_appliances_K10_s92603",
    "CA_C2_jena_K6_s92604",
    "CA_C2_ETTh1_K4_s92604",
    "CA_C2_ETTh2_safe17400_K4_s92604",
    "CA_C2_appliances_K10_s92604",
    "CA_C2_jena_K6_s92605",
    "CA_C2_ETTh1_K4_s92605",
    "CA_C2_ETTh2_safe17400_K4_s92605",
    "CA_C2_appliances_K10_s92605",
    "CA_C2_jena_K6_s92606",
    "CA_C2_ETTh1_K4_s92606",
    "CA_C2_ETTh2_safe17400_K4_s92606",
    "CA_C2_appliances_K10_s92606",
    "CA_C2_jena_K6_s92607",
    "CA_C2_ETTh1_K4_s92607",
    "CA_C2_ETTh2_safe17400_K4_s92607",
    "CA_C2_appliances_K10_s92607",
    "CA_C2_jena_K6_s92608",
    "CA_C2_ETTh1_K4_s92608",
    "CA_C2_ETTh2_safe17400_K4_s92608",
    "CA_C2_appliances_K10_s92608",
    "CA_C2_jena_K6_s92609",
    "CA_C2_ETTh1_K4_s92609",
    "CA_C2_ETTh2_safe17400_K4_s92609",
    "CA_C2_appliances_K10_s92609",
    "CA_C2_jena_K6_s92610",
    "CA_C2_ETTh1_K4_s92610",
    "CA_C2_ETTh2_safe17400_K4_s92610",
    "CA_C2_appliances_K10_s92610",
    "CA_C3_jena_K6_s92601",
    "CA_C3_ETTh1_K4_s92601",
    "CA_C3_ETTh2_safe17400_K4_s92601",
    "CA_C3_appliances_K10_s92601",
    "CA_C3_jena_K6_s92602",
    "CA_C3_ETTh1_K4_s92602",
    "CA_C3_ETTh2_safe17400_K4_s92602",
    "CA_C3_appliances_K10_s92602",
    "CA_C3_jena_K6_s92603",
    "CA_C3_ETTh1_K4_s92603",
    "CA_C3_ETTh2_safe17400_K4_s92603",
    "CA_C3_appliances_K10_s92603",
    "CA_C3_jena_K6_s92604",
    "CA_C3_ETTh1_K4_s92604",
    "CA_C3_ETTh2_safe17400_K4_s92604",
    "CA_C3_appliances_K10_s92604",
    "CA_C3_jena_K6_s92605",
    "CA_C3_ETTh1_K4_s92605",
    "CA_C3_ETTh2_safe17400_K4_s92605",
    "CA_C3_appliances_K10_s92605",
    "CA_C3_jena_K6_s92606",
    "CA_C3_ETTh1_K4_s92606",
    "CA_C3_ETTh2_safe17400_K4_s92606",
    "CA_C3_appliances_K10_s92606",
    "CA_C3_jena_K6_s92607",
    "CA_C3_ETTh1_K4_s92607",
    "CA_C3_ETTh2_safe17400_K4_s92607",
    "CA_C3_appliances_K10_s92607",
    "CA_C3_jena_K6_s92608",
    "CA_C3_ETTh1_K4_s92608",
    "CA_C3_ETTh2_safe17400_K4_s92608",
    "CA_C3_appliances_K10_s92608",
    "CA_C3_jena_K6_s92609",
    "CA_C3_ETTh1_K4_s92609",
    "CA_C3_ETTh2_safe17400_K4_s92609",
    "CA_C3_appliances_K10_s92609",
    "CA_C3_jena_K6_s92610",
    "CA_C3_ETTh1_K4_s92610",
    "CA_C3_ETTh2_safe17400_K4_s92610",
    "CA_C3_appliances_K10_s92610",
    "CA_A1_jena_K6_s92601",
    "CA_A1_ETTh1_K4_s92601",
    "CA_A1_ETTh2_safe17400_K4_s92601",
    "CA_A1_appliances_K10_s92601",
    "CA_A1_jena_K6_s92602",
    "CA_A1_ETTh1_K4_s92602",
    "CA_A1_ETTh2_safe17400_K4_s92602",
    "CA_A1_appliances_K10_s92602",
    "CA_A1_jena_K6_s92603",
    "CA_A1_ETTh1_K4_s92603",
    "CA_A1_ETTh2_safe17400_K4_s92603",
    "CA_A1_appliances_K10_s92603",
    "CA_A1_jena_K6_s92604",
    "CA_A1_ETTh1_K4_s92604",
    "CA_A1_ETTh2_safe17400_K4_s92604",
    "CA_A1_appliances_K10_s92604",
    "CA_A1_jena_K6_s92605",
    "CA_A1_ETTh1_K4_s92605",
    "CA_A1_ETTh2_safe17400_K4_s92605",
    "CA_A1_appliances_K10_s92605",
    "CA_A1_jena_K6_s92606",
    "CA_A1_ETTh1_K4_s92606",
    "CA_A1_ETTh2_safe17400_K4_s92606",
    "CA_A1_appliances_K10_s92606",
    "CA_A1_jena_K6_s92607",
    "CA_A1_ETTh1_K4_s92607",
    "CA_A1_ETTh2_safe17400_K4_s92607",
    "CA_A1_appliances_K10_s92607",
    "CA_A1_jena_K6_s92608",
    "CA_A1_ETTh1_K4_s92608",
    "CA_A1_ETTh2_safe17400_K4_s92608",
    "CA_A1_appliances_K10_s92608",
    "CA_A1_jena_K6_s92609",
    "CA_A1_ETTh1_K4_s92609",
    "CA_A1_ETTh2_safe17400_K4_s92609",
    "CA_A1_appliances_K10_s92609",
    "CA_A1_jena_K6_s92610",
    "CA_A1_ETTh1_K4_s92610",
    "CA_A1_ETTh2_safe17400_K4_s92610",
    "CA_A1_appliances_K10_s92610",
    "CA_A2_jena_K6_s92601",
    "CA_A2_ETTh1_K4_s92601",
    "CA_A2_ETTh2_safe17400_K4_s92601",
    "CA_A2_appliances_K10_s92601",
    "CA_A2_jena_K6_s92602",
    "CA_A2_ETTh1_K4_s92602",
    "CA_A2_ETTh2_safe17400_K4_s92602",
    "CA_A2_appliances_K10_s92602",
    "CA_A2_jena_K6_s92603",
    "CA_A2_ETTh1_K4_s92603",
    "CA_A2_ETTh2_safe17400_K4_s92603",
    "CA_A2_appliances_K10_s92603",
    "CA_A2_jena_K6_s92604",
    "CA_A2_ETTh1_K4_s92604",
    "CA_A2_ETTh2_safe17400_K4_s92604",
    "CA_A2_appliances_K10_s92604",
    "CA_A2_jena_K6_s92605",
    "CA_A2_ETTh1_K4_s92605",
    "CA_A2_ETTh2_safe17400_K4_s92605",
    "CA_A2_appliances_K10_s92605",
    "CA_A2_jena_K6_s92606",
    "CA_A2_ETTh1_K4_s92606",
    "CA_A2_ETTh2_safe17400_K4_s92606",
    "CA_A2_appliances_K10_s92606",
    "CA_A2_jena_K6_s92607",
    "CA_A2_ETTh1_K4_s92607",
    "CA_A2_ETTh2_safe17400_K4_s92607",
    "CA_A2_appliances_K10_s92607",
    "CA_A2_jena_K6_s92608",
    "CA_A2_ETTh1_K4_s92608",
    "CA_A2_ETTh2_safe17400_K4_s92608",
    "CA_A2_appliances_K10_s92608",
    "CA_A2_jena_K6_s92609",
    "CA_A2_ETTh1_K4_s92609",
    "CA_A2_ETTh2_safe17400_K4_s92609",
    "CA_A2_appliances_K10_s92609",
    "CA_A2_jena_K6_s92610",
    "CA_A2_ETTh1_K4_s92610",
    "CA_A2_ETTh2_safe17400_K4_s92610",
    "CA_A2_appliances_K10_s92610"
  ],
  "prior_run_statuses": {
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
  "remaining_seconds": 23130.01644730568,
  "run_statuses": {
    "CA_A1_ETTh1_K4_s92601_ext10": "NOT_RUN_SCOPE",
    "CA_A1_ETTh1_K4_s92602_ext10": "NOT_RUN_SCOPE",
    "CA_A1_ETTh1_K4_s92603_ext10": "NOT_RUN_SCOPE",
    "CA_A1_ETTh1_K4_s92604_ext10": "NOT_RUN_SCOPE",
    "CA_A1_ETTh1_K4_s92605_ext10": "NOT_RUN_SCOPE",
    "CA_A1_ETTh1_K4_s92606_ext10": "NOT_RUN_SCOPE",
    "CA_A1_ETTh1_K4_s92607_ext10": "NOT_RUN_SCOPE",
    "CA_A1_ETTh1_K4_s92608_ext10": "NOT_RUN_SCOPE",
    "CA_A1_ETTh1_K4_s92609_ext10": "NOT_RUN_SCOPE",
    "CA_A1_ETTh1_K4_s92610_ext10": "NOT_RUN_SCOPE",
    "CA_A1_ETTh2_safe17400_K4_s92601_ext10": "NOT_RUN_SCOPE",
    "CA_A1_ETTh2_safe17400_K4_s92602_ext10": "NOT_RUN_SCOPE",
    "CA_A1_ETTh2_safe17400_K4_s92603_ext10": "NOT_RUN_SCOPE",
    "CA_A1_ETTh2_safe17400_K4_s92604_ext10": "NOT_RUN_SCOPE",
    "CA_A1_ETTh2_safe17400_K4_s92605_ext10": "NOT_RUN_SCOPE",
    "CA_A1_ETTh2_safe17400_K4_s92606_ext10": "NOT_RUN_SCOPE",
    "CA_A1_ETTh2_safe17400_K4_s92607_ext10": "NOT_RUN_SCOPE",
    "CA_A1_ETTh2_safe17400_K4_s92608_ext10": "NOT_RUN_SCOPE",
    "CA_A1_ETTh2_safe17400_K4_s92609_ext10": "NOT_RUN_SCOPE",
    "CA_A1_ETTh2_safe17400_K4_s92610_ext10": "NOT_RUN_SCOPE",
    "CA_A1_appliances_K10_s92601_ext10": "NOT_RUN_SCOPE",
    "CA_A1_appliances_K10_s92602_ext10": "NOT_RUN_SCOPE",
    "CA_A1_appliances_K10_s92603_ext10": "NOT_RUN_SCOPE",
    "CA_A1_appliances_K10_s92604_ext10": "NOT_RUN_SCOPE",
    "CA_A1_appliances_K10_s92605_ext10": "NOT_RUN_SCOPE",
    "CA_A1_appliances_K10_s92606_ext10": "NOT_RUN_SCOPE",
    "CA_A1_appliances_K10_s92607_ext10": "NOT_RUN_SCOPE",
    "CA_A1_appliances_K10_s92608_ext10": "NOT_RUN_SCOPE",
    "CA_A1_appliances_K10_s92609_ext10": "NOT_RUN_SCOPE",
    "CA_A1_appliances_K10_s92610_ext10": "NOT_RUN_SCOPE",
    "CA_A1_jena_K6_s92601_ext10": "COMPLETE",
    "CA_A1_jena_K6_s92602_ext10": "COMPLETE",
    "CA_A1_jena_K6_s92603_ext10": "COMPLETE",
    "CA_A1_jena_K6_s92604_ext10": "COMPLETE",
    "CA_A1_jena_K6_s92605_ext10": "COMPLETE",
    "CA_A1_jena_K6_s92606_ext10": "COMPLETE",
    "CA_A1_jena_K6_s92607_ext10": "COMPLETE",
    "CA_A1_jena_K6_s92608_ext10": "COMPLETE",
    "CA_A1_jena_K6_s92609_ext10": "COMPLETE",
    "CA_A1_jena_K6_s92610_ext10": "COMPLETE",
    "CA_A2_ETTh1_K4_s92601_ext10": "NOT_RUN_SCOPE",
    "CA_A2_ETTh1_K4_s92602_ext10": "NOT_RUN_SCOPE",
    "CA_A2_ETTh1_K4_s92603_ext10": "NOT_RUN_SCOPE",
    "CA_A2_ETTh1_K4_s92604_ext10": "NOT_RUN_SCOPE",
    "CA_A2_ETTh1_K4_s92605_ext10": "NOT_RUN_SCOPE",
    "CA_A2_ETTh1_K4_s92606_ext10": "NOT_RUN_SCOPE",
    "CA_A2_ETTh1_K4_s92607_ext10": "NOT_RUN_SCOPE",
    "CA_A2_ETTh1_K4_s92608_ext10": "NOT_RUN_SCOPE",
    "CA_A2_ETTh1_K4_s92609_ext10": "NOT_RUN_SCOPE",
    "CA_A2_ETTh1_K4_s92610_ext10": "NOT_RUN_SCOPE",
    "CA_A2_ETTh2_safe17400_K4_s92601_ext10": "NOT_RUN_SCOPE",
    "CA_A2_ETTh2_safe17400_K4_s92602_ext10": "NOT_RUN_SCOPE",
    "CA_A2_ETTh2_safe17400_K4_s92603_ext10": "NOT_RUN_SCOPE",
    "CA_A2_ETTh2_safe17400_K4_s92604_ext10": "NOT_RUN_SCOPE",
    "CA_A2_ETTh2_safe17400_K4_s92605_ext10": "NOT_RUN_SCOPE",
    "CA_A2_ETTh2_safe17400_K4_s92606_ext10": "NOT_RUN_SCOPE",
    "CA_A2_ETTh2_safe17400_K4_s92607_ext10": "NOT_RUN_SCOPE",
    "CA_A2_ETTh2_safe17400_K4_s92608_ext10": "NOT_RUN_SCOPE",
    "CA_A2_ETTh2_safe17400_K4_s92609_ext10": "NOT_RUN_SCOPE",
    "CA_A2_ETTh2_safe17400_K4_s92610_ext10": "NOT_RUN_SCOPE",
    "CA_A2_appliances_K10_s92601_ext10": "NOT_RUN_SCOPE",
    "CA_A2_appliances_K10_s92602_ext10": "NOT_RUN_SCOPE",
    "CA_A2_appliances_K10_s92603_ext10": "NOT_RUN_SCOPE",
    "CA_A2_appliances_K10_s92604_ext10": "NOT_RUN_SCOPE",
    "CA_A2_appliances_K10_s92605_ext10": "NOT_RUN_SCOPE",
    "CA_A2_appliances_K10_s92606_ext10": "NOT_RUN_SCOPE",
    "CA_A2_appliances_K10_s92607_ext10": "NOT_RUN_SCOPE",
    "CA_A2_appliances_K10_s92608_ext10": "NOT_RUN_SCOPE",
    "CA_A2_appliances_K10_s92609_ext10": "NOT_RUN_SCOPE",
    "CA_A2_appliances_K10_s92610_ext10": "NOT_RUN_SCOPE",
    "CA_A2_jena_K6_s92601_ext10": "NOT_RUN_SCOPE",
    "CA_A2_jena_K6_s92602_ext10": "NOT_RUN_SCOPE",
    "CA_A2_jena_K6_s92603_ext10": "NOT_RUN_SCOPE",
    "CA_A2_jena_K6_s92604_ext10": "NOT_RUN_SCOPE",
    "CA_A2_jena_K6_s92605_ext10": "NOT_RUN_SCOPE",
    "CA_A2_jena_K6_s92606_ext10": "NOT_RUN_SCOPE",
    "CA_A2_jena_K6_s92607_ext10": "NOT_RUN_SCOPE",
    "CA_A2_jena_K6_s92608_ext10": "NOT_RUN_SCOPE",
    "CA_A2_jena_K6_s92609_ext10": "NOT_RUN_SCOPE",
    "CA_A2_jena_K6_s92610_ext10": "NOT_RUN_SCOPE",
    "CA_A3_ETTh1_K4_s92601": "NOT_RUN_SCOPE",
    "CA_A3_ETTh1_K4_s92601_ext10": "NOT_RUN_SCOPE",
    "CA_A3_ETTh1_K4_s92602": "NOT_RUN_SCOPE",
    "CA_A3_ETTh1_K4_s92602_ext10": "NOT_RUN_SCOPE",
    "CA_A3_ETTh1_K4_s92603": "NOT_RUN_SCOPE",
    "CA_A3_ETTh1_K4_s92603_ext10": "NOT_RUN_SCOPE",
    "CA_A3_ETTh1_K4_s92604": "NOT_RUN_SCOPE",
    "CA_A3_ETTh1_K4_s92604_ext10": "NOT_RUN_SCOPE",
    "CA_A3_ETTh1_K4_s92605": "NOT_RUN_SCOPE",
    "CA_A3_ETTh1_K4_s92605_ext10": "NOT_RUN_SCOPE",
    "CA_A3_ETTh1_K4_s92606": "NOT_RUN_SCOPE",
    "CA_A3_ETTh1_K4_s92606_ext10": "NOT_RUN_SCOPE",
    "CA_A3_ETTh1_K4_s92607": "NOT_RUN_SCOPE",
    "CA_A3_ETTh1_K4_s92607_ext10": "NOT_RUN_SCOPE",
    "CA_A3_ETTh1_K4_s92608": "NOT_RUN_SCOPE",
    "CA_A3_ETTh1_K4_s92608_ext10": "NOT_RUN_SCOPE",
    "CA_A3_ETTh1_K4_s92609": "NOT_RUN_SCOPE",
    "CA_A3_ETTh1_K4_s92609_ext10": "NOT_RUN_SCOPE",
    "CA_A3_ETTh1_K4_s92610": "NOT_RUN_SCOPE",
    "CA_A3_ETTh1_K4_s92610_ext10": "NOT_RUN_SCOPE",
    "CA_A3_ETTh2_safe17400_K4_s92601": "NOT_RUN_SCOPE",
    "CA_A3_ETTh2_safe17400_K4_s92601_ext10": "NOT_RUN_SCOPE",
    "CA_A3_ETTh2_safe17400_K4_s92602": "NOT_RUN_SCOPE",
    "CA_A3_ETTh2_safe17400_K4_s92602_ext10": "NOT_RUN_SCOPE",
    "CA_A3_ETTh2_safe17400_K4_s92603": "NOT_RUN_SCOPE",
    "CA_A3_ETTh2_safe17400_K4_s92603_ext10": "NOT_RUN_SCOPE",
    "CA_A3_ETTh2_safe17400_K4_s92604": "NOT_RUN_SCOPE",
    "CA_A3_ETTh2_safe17400_K4_s92604_ext10": "NOT_RUN_SCOPE",
    "CA_A3_ETTh2_safe17400_K4_s92605": "NOT_RUN_SCOPE",
    "CA_A3_ETTh2_safe17400_K4_s92605_ext10": "NOT_RUN_SCOPE",
    "CA_A3_ETTh2_safe17400_K4_s92606": "NOT_RUN_SCOPE",
    "CA_A3_ETTh2_safe17400_K4_s92606_ext10": "NOT_RUN_SCOPE",
    "CA_A3_ETTh2_safe17400_K4_s92607": "NOT_RUN_SCOPE",
    "CA_A3_ETTh2_safe17400_K4_s92607_ext10": "NOT_RUN_SCOPE",
    "CA_A3_ETTh2_safe17400_K4_s92608": "NOT_RUN_SCOPE",
    "CA_A3_ETTh2_safe17400_K4_s92608_ext10": "NOT_RUN_SCOPE",
    "CA_A3_ETTh2_safe17400_K4_s92609": "NOT_RUN_SCOPE",
    "CA_A3_ETTh2_safe17400_K4_s92609_ext10": "NOT_RUN_SCOPE",
    "CA_A3_ETTh2_safe17400_K4_s92610": "NOT_RUN_SCOPE",
    "CA_A3_ETTh2_safe17400_K4_s92610_ext10": "NOT_RUN_SCOPE",
    "CA_A3_appliances_K10_s92601": "NOT_RUN_SCOPE",
    "CA_A3_appliances_K10_s92601_ext10": "NOT_RUN_SCOPE",
    "CA_A3_appliances_K10_s92602": "NOT_RUN_SCOPE",
    "CA_A3_appliances_K10_s92602_ext10": "NOT_RUN_SCOPE",
    "CA_A3_appliances_K10_s92603": "NOT_RUN_SCOPE",
    "CA_A3_appliances_K10_s92603_ext10": "NOT_RUN_SCOPE",
    "CA_A3_appliances_K10_s92604": "NOT_RUN_SCOPE",
    "CA_A3_appliances_K10_s92604_ext10": "NOT_RUN_SCOPE",
    "CA_A3_appliances_K10_s92605": "NOT_RUN_SCOPE",
    "CA_A3_appliances_K10_s92605_ext10": "NOT_RUN_SCOPE",
    "CA_A3_appliances_K10_s92606": "NOT_RUN_SCOPE",
    "CA_A3_appliances_K10_s92606_ext10": "NOT_RUN_SCOPE",
    "CA_A3_appliances_K10_s92607": "NOT_RUN_SCOPE",
    "CA_A3_appliances_K10_s92607_ext10": "NOT_RUN_SCOPE",
    "CA_A3_appliances_K10_s92608": "NOT_RUN_SCOPE",
    "CA_A3_appliances_K10_s92608_ext10": "NOT_RUN_SCOPE",
    "CA_A3_appliances_K10_s92609": "NOT_RUN_SCOPE",
    "CA_A3_appliances_K10_s92609_ext10": "NOT_RUN_SCOPE",
    "CA_A3_appliances_K10_s92610": "NOT_RUN_SCOPE",
    "CA_A3_appliances_K10_s92610_ext10": "NOT_RUN_SCOPE",
    "CA_A3_jena_K6_s92601": "NOT_RUN_SCOPE",
    "CA_A3_jena_K6_s92601_ext10": "NOT_RUN_SCOPE",
    "CA_A3_jena_K6_s92602": "NOT_RUN_SCOPE",
    "CA_A3_jena_K6_s92602_ext10": "NOT_RUN_SCOPE",
    "CA_A3_jena_K6_s92603": "NOT_RUN_SCOPE",
    "CA_A3_jena_K6_s92603_ext10": "NOT_RUN_SCOPE",
    "CA_A3_jena_K6_s92604": "NOT_RUN_SCOPE",
    "CA_A3_jena_K6_s92604_ext10": "NOT_RUN_SCOPE",
    "CA_A3_jena_K6_s92605": "NOT_RUN_SCOPE",
    "CA_A3_jena_K6_s92605_ext10": "NOT_RUN_SCOPE",
    "CA_A3_jena_K6_s92606": "NOT_RUN_SCOPE",
    "CA_A3_jena_K6_s92606_ext10": "NOT_RUN_SCOPE",
    "CA_A3_jena_K6_s92607": "NOT_RUN_SCOPE",
    "CA_A3_jena_K6_s92607_ext10": "NOT_RUN_SCOPE",
    "CA_A3_jena_K6_s92608": "NOT_RUN_SCOPE",
    "CA_A3_jena_K6_s92608_ext10": "NOT_RUN_SCOPE",
    "CA_A3_jena_K6_s92609": "NOT_RUN_SCOPE",
    "CA_A3_jena_K6_s92609_ext10": "NOT_RUN_SCOPE",
    "CA_A3_jena_K6_s92610": "NOT_RUN_SCOPE",
    "CA_A3_jena_K6_s92610_ext10": "NOT_RUN_SCOPE",
    "CA_A4_ETTh1_K4_s92601": "COMPLETE",
    "CA_A4_ETTh1_K4_s92601_ext10": "NOT_RUN_SCOPE",
    "CA_A4_ETTh1_K4_s92602": "COMPLETE",
    "CA_A4_ETTh1_K4_s92602_ext10": "NOT_RUN_SCOPE",
    "CA_A4_ETTh1_K4_s92603": "COMPLETE",
    "CA_A4_ETTh1_K4_s92603_ext10": "NOT_RUN_SCOPE",
    "CA_A4_ETTh1_K4_s92604": "COMPLETE",
    "CA_A4_ETTh1_K4_s92604_ext10": "NOT_RUN_SCOPE",
    "CA_A4_ETTh1_K4_s92605": "COMPLETE",
    "CA_A4_ETTh1_K4_s92605_ext10": "NOT_RUN_SCOPE",
    "CA_A4_ETTh1_K4_s92606": "COMPLETE",
    "CA_A4_ETTh1_K4_s92606_ext10": "NOT_RUN_SCOPE",
    "CA_A4_ETTh1_K4_s92607": "COMPLETE",
    "CA_A4_ETTh1_K4_s92607_ext10": "NOT_RUN_SCOPE",
    "CA_A4_ETTh1_K4_s92608": "COMPLETE",
    "CA_A4_ETTh1_K4_s92608_ext10": "NOT_RUN_SCOPE",
    "CA_A4_ETTh1_K4_s92609": "COMPLETE",
    "CA_A4_ETTh1_K4_s92609_ext10": "NOT_RUN_SCOPE",
    "CA_A4_ETTh1_K4_s92610": "COMPLETE",
    "CA_A4_ETTh1_K4_s92610_ext10": "NOT_RUN_SCOPE",
    "CA_A4_ETTh2_safe17400_K4_s92601": "COMPLETE",
    "CA_A4_ETTh2_safe17400_K4_s92601_ext10": "NOT_RUN_SCOPE",
    "CA_A4_ETTh2_safe17400_K4_s92602": "COMPLETE",
    "CA_A4_ETTh2_safe17400_K4_s92602_ext10": "NOT_RUN_SCOPE",
    "CA_A4_ETTh2_safe17400_K4_s92603": "COMPLETE",
    "CA_A4_ETTh2_safe17400_K4_s92603_ext10": "NOT_RUN_SCOPE",
    "CA_A4_ETTh2_safe17400_K4_s92604": "COMPLETE",
    "CA_A4_ETTh2_safe17400_K4_s92604_ext10": "NOT_RUN_SCOPE",
    "CA_A4_ETTh2_safe17400_K4_s92605": "COMPLETE",
    "CA_A4_ETTh2_safe17400_K4_s92605_ext10": "NOT_RUN_SCOPE",
    "CA_A4_ETTh2_safe17400_K4_s92606": "COMPLETE",
    "CA_A4_ETTh2_safe17400_K4_s92606_ext10": "NOT_RUN_SCOPE",
    "CA_A4_ETTh2_safe17400_K4_s92607": "COMPLETE",
    "CA_A4_ETTh2_safe17400_K4_s92607_ext10": "NOT_RUN_SCOPE",
    "CA_A4_ETTh2_safe17400_K4_s92608": "COMPLETE",
    "CA_A4_ETTh2_safe17400_K4_s92608_ext10": "NOT_RUN_SCOPE",
    "CA_A4_ETTh2_safe17400_K4_s92609": "COMPLETE",
    "CA_A4_ETTh2_safe17400_K4_s92609_ext10": "NOT_RUN_SCOPE",
    "CA_A4_ETTh2_safe17400_K4_s92610": "COMPLETE",
    "CA_A4_ETTh2_safe17400_K4_s92610_ext10": "NOT_RUN_SCOPE",
    "CA_A4_appliances_K10_s92601": "COMPLETE",
    "CA_A4_appliances_K10_s92601_ext10": "NOT_RUN_SCOPE",
    "CA_A4_appliances_K10_s92602": "COMPLETE",
    "CA_A4_appliances_K10_s92602_ext10": "NOT_RUN_SCOPE",
    "CA_A4_appliances_K10_s92603": "COMPLETE",
    "CA_A4_appliances_K10_s92603_ext10": "NOT_RUN_SCOPE",
    "CA_A4_appliances_K10_s92604": "COMPLETE",
    "CA_A4_appliances_K10_s92604_ext10": "NOT_RUN_SCOPE",
    "CA_A4_appliances_K10_s92605": "COMPLETE",
    "CA_A4_appliances_K10_s92605_ext10": "NOT_RUN_SCOPE",
    "CA_A4_appliances_K10_s92606": "COMPLETE",
    "CA_A4_appliances_K10_s92606_ext10": "NOT_RUN_SCOPE",
    "CA_A4_appliances_K10_s92607": "COMPLETE",
    "CA_A4_appliances_K10_s92607_ext10": "NOT_RUN_SCOPE",
    "CA_A4_appliances_K10_s92608": "COMPLETE",
    "CA_A4_appliances_K10_s92608_ext10": "NOT_RUN_SCOPE",
    "CA_A4_appliances_K10_s92609": "COMPLETE",
    "CA_A4_appliances_K10_s92609_ext10": "NOT_RUN_SCOPE",
    "CA_A4_appliances_K10_s92610": "COMPLETE",
    "CA_A4_appliances_K10_s92610_ext10": "NOT_RUN_SCOPE",
    "CA_A4_jena_K6_s92601": "COMPLETE",
    "CA_A4_jena_K6_s92601_ext10": "NOT_RUN_SCOPE",
    "CA_A4_jena_K6_s92602": "COMPLETE",
    "CA_A4_jena_K6_s92602_ext10": "NOT_RUN_SCOPE",
    "CA_A4_jena_K6_s92603": "COMPLETE",
    "CA_A4_jena_K6_s92603_ext10": "NOT_RUN_SCOPE",
    "CA_A4_jena_K6_s92604": "COMPLETE",
    "CA_A4_jena_K6_s92604_ext10": "NOT_RUN_SCOPE",
    "CA_A4_jena_K6_s92605": "COMPLETE",
    "CA_A4_jena_K6_s92605_ext10": "NOT_RUN_SCOPE",
    "CA_A4_jena_K6_s92606": "COMPLETE",
    "CA_A4_jena_K6_s92606_ext10": "NOT_RUN_SCOPE",
    "CA_A4_jena_K6_s92607": "COMPLETE",
    "CA_A4_jena_K6_s92607_ext10": "NOT_RUN_SCOPE",
    "CA_A4_jena_K6_s92608": "COMPLETE",
    "CA_A4_jena_K6_s92608_ext10": "NOT_RUN_SCOPE",
    "CA_A4_jena_K6_s92609": "COMPLETE",
    "CA_A4_jena_K6_s92609_ext10": "NOT_RUN_SCOPE",
    "CA_A4_jena_K6_s92610": "COMPLETE",
    "CA_A4_jena_K6_s92610_ext10": "NOT_RUN_SCOPE",
    "CA_A5_ETTh1_K4_s92601": "NOT_RUN_SCOPE",
    "CA_A5_ETTh1_K4_s92601_ext10": "NOT_RUN_SCOPE",
    "CA_A5_ETTh1_K4_s92602": "NOT_RUN_SCOPE",
    "CA_A5_ETTh1_K4_s92602_ext10": "NOT_RUN_SCOPE",
    "CA_A5_ETTh1_K4_s92603": "NOT_RUN_SCOPE",
    "CA_A5_ETTh1_K4_s92603_ext10": "NOT_RUN_SCOPE",
    "CA_A5_ETTh1_K4_s92604": "NOT_RUN_SCOPE",
    "CA_A5_ETTh1_K4_s92604_ext10": "NOT_RUN_SCOPE",
    "CA_A5_ETTh1_K4_s92605": "NOT_RUN_SCOPE",
    "CA_A5_ETTh1_K4_s92605_ext10": "NOT_RUN_SCOPE",
    "CA_A5_ETTh1_K4_s92606": "NOT_RUN_SCOPE",
    "CA_A5_ETTh1_K4_s92606_ext10": "NOT_RUN_SCOPE",
    "CA_A5_ETTh1_K4_s92607": "NOT_RUN_SCOPE",
    "CA_A5_ETTh1_K4_s92607_ext10": "NOT_RUN_SCOPE",
    "CA_A5_ETTh1_K4_s92608": "NOT_RUN_SCOPE",
    "CA_A5_ETTh1_K4_s92608_ext10": "NOT_RUN_SCOPE",
    "CA_A5_ETTh1_K4_s92609": "NOT_RUN_SCOPE",
    "CA_A5_ETTh1_K4_s92609_ext10": "NOT_RUN_SCOPE",
    "CA_A5_ETTh1_K4_s92610": "NOT_RUN_SCOPE",
    "CA_A5_ETTh1_K4_s92610_ext10": "NOT_RUN_SCOPE",
    "CA_A5_ETTh2_safe17400_K4_s92601": "NOT_RUN_SCOPE",
    "CA_A5_ETTh2_safe17400_K4_s92601_ext10": "NOT_RUN_SCOPE",
    "CA_A5_ETTh2_safe17400_K4_s92602": "NOT_RUN_SCOPE",
    "CA_A5_ETTh2_safe17400_K4_s92602_ext10": "NOT_RUN_SCOPE",
    "CA_A5_ETTh2_safe17400_K4_s92603": "NOT_RUN_SCOPE",
    "CA_A5_ETTh2_safe17400_K4_s92603_ext10": "NOT_RUN_SCOPE",
    "CA_A5_ETTh2_safe17400_K4_s92604": "NOT_RUN_SCOPE",
    "CA_A5_ETTh2_safe17400_K4_s92604_ext10": "NOT_RUN_SCOPE",
    "CA_A5_ETTh2_safe17400_K4_s92605": "NOT_RUN_SCOPE",
    "CA_A5_ETTh2_safe17400_K4_s92605_ext10": "NOT_RUN_SCOPE",
    "CA_A5_ETTh2_safe17400_K4_s92606": "NOT_RUN_SCOPE",
    "CA_A5_ETTh2_safe17400_K4_s92606_ext10": "NOT_RUN_SCOPE",
    "CA_A5_ETTh2_safe17400_K4_s92607": "NOT_RUN_SCOPE",
    "CA_A5_ETTh2_safe17400_K4_s92607_ext10": "NOT_RUN_SCOPE",
    "CA_A5_ETTh2_safe17400_K4_s92608": "NOT_RUN_SCOPE",
    "CA_A5_ETTh2_safe17400_K4_s92608_ext10": "NOT_RUN_SCOPE",
    "CA_A5_ETTh2_safe17400_K4_s92609": "NOT_RUN_SCOPE",
    "CA_A5_ETTh2_safe17400_K4_s92609_ext10": "NOT_RUN_SCOPE",
    "CA_A5_ETTh2_safe17400_K4_s92610": "NOT_RUN_SCOPE",
    "CA_A5_ETTh2_safe17400_K4_s92610_ext10": "NOT_RUN_SCOPE",
    "CA_A5_appliances_K10_s92601": "NOT_RUN_SCOPE",
    "CA_A5_appliances_K10_s92601_ext10": "NOT_RUN_SCOPE",
    "CA_A5_appliances_K10_s92602": "NOT_RUN_SCOPE",
    "CA_A5_appliances_K10_s92602_ext10": "NOT_RUN_SCOPE",
    "CA_A5_appliances_K10_s92603": "NOT_RUN_SCOPE",
    "CA_A5_appliances_K10_s92603_ext10": "NOT_RUN_SCOPE",
    "CA_A5_appliances_K10_s92604": "NOT_RUN_SCOPE",
    "CA_A5_appliances_K10_s92604_ext10": "NOT_RUN_SCOPE",
    "CA_A5_appliances_K10_s92605": "NOT_RUN_SCOPE",
    "CA_A5_appliances_K10_s92605_ext10": "NOT_RUN_SCOPE",
    "CA_A5_appliances_K10_s92606": "NOT_RUN_SCOPE",
    "CA_A5_appliances_K10_s92606_ext10": "NOT_RUN_SCOPE",
    "CA_A5_appliances_K10_s92607": "NOT_RUN_SCOPE",
    "CA_A5_appliances_K10_s92607_ext10": "NOT_RUN_SCOPE",
    "CA_A5_appliances_K10_s92608": "NOT_RUN_SCOPE",
    "CA_A5_appliances_K10_s92608_ext10": "NOT_RUN_SCOPE",
    "CA_A5_appliances_K10_s92609": "NOT_RUN_SCOPE",
    "CA_A5_appliances_K10_s92609_ext10": "NOT_RUN_SCOPE",
    "CA_A5_appliances_K10_s92610": "NOT_RUN_SCOPE",
    "CA_A5_appliances_K10_s92610_ext10": "NOT_RUN_SCOPE",
    "CA_A5_jena_K6_s92601": "NOT_RUN_SCOPE",
    "CA_A5_jena_K6_s92601_ext10": "NOT_RUN_SCOPE",
    "CA_A5_jena_K6_s92602": "NOT_RUN_SCOPE",
    "CA_A5_jena_K6_s92602_ext10": "NOT_RUN_SCOPE",
    "CA_A5_jena_K6_s92603": "NOT_RUN_SCOPE",
    "CA_A5_jena_K6_s92603_ext10": "NOT_RUN_SCOPE",
    "CA_A5_jena_K6_s92604": "NOT_RUN_SCOPE",
    "CA_A5_jena_K6_s92604_ext10": "NOT_RUN_SCOPE",
    "CA_A5_jena_K6_s92605": "NOT_RUN_SCOPE",
    "CA_A5_jena_K6_s92605_ext10": "NOT_RUN_SCOPE",
    "CA_A5_jena_K6_s92606": "NOT_RUN_SCOPE",
    "CA_A5_jena_K6_s92606_ext10": "NOT_RUN_SCOPE",
    "CA_A5_jena_K6_s92607": "NOT_RUN_SCOPE",
    "CA_A5_jena_K6_s92607_ext10": "NOT_RUN_SCOPE",
    "CA_A5_jena_K6_s92608": "NOT_RUN_SCOPE",
    "CA_A5_jena_K6_s92608_ext10": "NOT_RUN_SCOPE",
    "CA_A5_jena_K6_s92609": "NOT_RUN_SCOPE",
    "CA_A5_jena_K6_s92609_ext10": "NOT_RUN_SCOPE",
    "CA_A5_jena_K6_s92610": "NOT_RUN_SCOPE",
    "CA_A5_jena_K6_s92610_ext10": "NOT_RUN_SCOPE",
    "CA_C2_ETTh1_K4_s92601_ext10": "NOT_RUN_SCOPE",
    "CA_C2_ETTh1_K4_s92602_ext10": "NOT_RUN_SCOPE",
    "CA_C2_ETTh1_K4_s92603_ext10": "NOT_RUN_SCOPE",
    "CA_C2_ETTh1_K4_s92604_ext10": "NOT_RUN_SCOPE",
    "CA_C2_ETTh1_K4_s92605_ext10": "NOT_RUN_SCOPE",
    "CA_C2_ETTh1_K4_s92606_ext10": "NOT_RUN_SCOPE",
    "CA_C2_ETTh1_K4_s92607_ext10": "NOT_RUN_SCOPE",
    "CA_C2_ETTh1_K4_s92608_ext10": "NOT_RUN_SCOPE",
    "CA_C2_ETTh1_K4_s92609_ext10": "NOT_RUN_SCOPE",
    "CA_C2_ETTh1_K4_s92610_ext10": "NOT_RUN_SCOPE",
    "CA_C2_ETTh2_safe17400_K4_s92601_ext10": "NOT_RUN_SCOPE",
    "CA_C2_ETTh2_safe17400_K4_s92602_ext10": "NOT_RUN_SCOPE",
    "CA_C2_ETTh2_safe17400_K4_s92603_ext10": "NOT_RUN_SCOPE",
    "CA_C2_ETTh2_safe17400_K4_s92604_ext10": "NOT_RUN_SCOPE",
    "CA_C2_ETTh2_safe17400_K4_s92605_ext10": "NOT_RUN_SCOPE",
    "CA_C2_ETTh2_safe17400_K4_s92606_ext10": "NOT_RUN_SCOPE",
    "CA_C2_ETTh2_safe17400_K4_s92607_ext10": "NOT_RUN_SCOPE",
    "CA_C2_ETTh2_safe17400_K4_s92608_ext10": "NOT_RUN_SCOPE",
    "CA_C2_ETTh2_safe17400_K4_s92609_ext10": "NOT_RUN_SCOPE",
    "CA_C2_ETTh2_safe17400_K4_s92610_ext10": "NOT_RUN_SCOPE",
    "CA_C2_appliances_K10_s92601_ext10": "NOT_RUN_SCOPE",
    "CA_C2_appliances_K10_s92602_ext10": "NOT_RUN_SCOPE",
    "CA_C2_appliances_K10_s92603_ext10": "NOT_RUN_SCOPE",
    "CA_C2_appliances_K10_s92604_ext10": "NOT_RUN_SCOPE",
    "CA_C2_appliances_K10_s92605_ext10": "NOT_RUN_SCOPE",
    "CA_C2_appliances_K10_s92606_ext10": "NOT_RUN_SCOPE",
    "CA_C2_appliances_K10_s92607_ext10": "NOT_RUN_SCOPE",
    "CA_C2_appliances_K10_s92608_ext10": "NOT_RUN_SCOPE",
    "CA_C2_appliances_K10_s92609_ext10": "NOT_RUN_SCOPE",
    "CA_C2_appliances_K10_s92610_ext10": "NOT_RUN_SCOPE",
    "CA_C2_jena_K6_s92601_ext10": "NOT_RUN_SCOPE",
    "CA_C2_jena_K6_s92602_ext10": "NOT_RUN_SCOPE",
    "CA_C2_jena_K6_s92603_ext10": "NOT_RUN_SCOPE",
    "CA_C2_jena_K6_s92604_ext10": "NOT_RUN_SCOPE",
    "CA_C2_jena_K6_s92605_ext10": "NOT_RUN_SCOPE",
    "CA_C2_jena_K6_s92606_ext10": "NOT_RUN_SCOPE",
    "CA_C2_jena_K6_s92607_ext10": "NOT_RUN_SCOPE",
    "CA_C2_jena_K6_s92608_ext10": "NOT_RUN_SCOPE",
    "CA_C2_jena_K6_s92609_ext10": "NOT_RUN_SCOPE",
    "CA_C2_jena_K6_s92610_ext10": "NOT_RUN_SCOPE",
    "CA_C3_ETTh1_K4_s92601_ext10": "NOT_RUN_SCOPE",
    "CA_C3_ETTh1_K4_s92602_ext10": "NOT_RUN_SCOPE",
    "CA_C3_ETTh1_K4_s92603_ext10": "NOT_RUN_SCOPE",
    "CA_C3_ETTh1_K4_s92604_ext10": "NOT_RUN_SCOPE",
    "CA_C3_ETTh1_K4_s92605_ext10": "NOT_RUN_SCOPE",
    "CA_C3_ETTh1_K4_s92606_ext10": "NOT_RUN_SCOPE",
    "CA_C3_ETTh1_K4_s92607_ext10": "NOT_RUN_SCOPE",
    "CA_C3_ETTh1_K4_s92608_ext10": "NOT_RUN_SCOPE",
    "CA_C3_ETTh1_K4_s92609_ext10": "NOT_RUN_SCOPE",
    "CA_C3_ETTh1_K4_s92610_ext10": "NOT_RUN_SCOPE",
    "CA_C3_ETTh2_safe17400_K4_s92601_ext10": "NOT_RUN_SCOPE",
    "CA_C3_ETTh2_safe17400_K4_s92602_ext10": "NOT_RUN_SCOPE",
    "CA_C3_ETTh2_safe17400_K4_s92603_ext10": "NOT_RUN_SCOPE",
    "CA_C3_ETTh2_safe17400_K4_s92604_ext10": "NOT_RUN_SCOPE",
    "CA_C3_ETTh2_safe17400_K4_s92605_ext10": "NOT_RUN_SCOPE",
    "CA_C3_ETTh2_safe17400_K4_s92606_ext10": "NOT_RUN_SCOPE",
    "CA_C3_ETTh2_safe17400_K4_s92607_ext10": "NOT_RUN_SCOPE",
    "CA_C3_ETTh2_safe17400_K4_s92608_ext10": "NOT_RUN_SCOPE",
    "CA_C3_ETTh2_safe17400_K4_s92609_ext10": "NOT_RUN_SCOPE",
    "CA_C3_ETTh2_safe17400_K4_s92610_ext10": "NOT_RUN_SCOPE",
    "CA_C3_appliances_K10_s92601_ext10": "NOT_RUN_SCOPE",
    "CA_C3_appliances_K10_s92602_ext10": "NOT_RUN_SCOPE",
    "CA_C3_appliances_K10_s92603_ext10": "NOT_RUN_SCOPE",
    "CA_C3_appliances_K10_s92604_ext10": "NOT_RUN_SCOPE",
    "CA_C3_appliances_K10_s92605_ext10": "NOT_RUN_SCOPE",
    "CA_C3_appliances_K10_s92606_ext10": "NOT_RUN_SCOPE",
    "CA_C3_appliances_K10_s92607_ext10": "NOT_RUN_SCOPE",
    "CA_C3_appliances_K10_s92608_ext10": "NOT_RUN_SCOPE",
    "CA_C3_appliances_K10_s92609_ext10": "NOT_RUN_SCOPE",
    "CA_C3_appliances_K10_s92610_ext10": "NOT_RUN_SCOPE",
    "CA_C3_jena_K6_s92601_ext10": "NOT_RUN_SCOPE",
    "CA_C3_jena_K6_s92602_ext10": "NOT_RUN_SCOPE",
    "CA_C3_jena_K6_s92603_ext10": "NOT_RUN_SCOPE",
    "CA_C3_jena_K6_s92604_ext10": "NOT_RUN_SCOPE",
    "CA_C3_jena_K6_s92605_ext10": "NOT_RUN_SCOPE",
    "CA_C3_jena_K6_s92606_ext10": "NOT_RUN_SCOPE",
    "CA_C3_jena_K6_s92607_ext10": "NOT_RUN_SCOPE",
    "CA_C3_jena_K6_s92608_ext10": "NOT_RUN_SCOPE",
    "CA_C3_jena_K6_s92609_ext10": "NOT_RUN_SCOPE",
    "CA_C3_jena_K6_s92610_ext10": "NOT_RUN_SCOPE"
  },
  "started_utc": "2026-10-27T01:56:38.063923+00:00",
  "status": "COMPLETE",
  "stop_reason": null,
  "updated_utc": "2026-10-27T08:31:34.851619+00:00"
}
```

그림: FIG_COMPARATIVE.png/.svg, FIG_ABLATION.png/.svg, FIG_ERROR_QUANTILES.png/.svg. 각 그림의 같은 이름 CSV에 긴 형식 데이터를 제공한다.

V MSE is the mean across original channels of squared forecast errors over observed validation targets, measured after normalization using statistics fitted only on the A60 training interval.

[확인] 근거: research/informs_broad_validation_20261015_v1/evaluation.py:39–61; data_contract.py:61–76; fit_base.py:46–49. V는 개발 검증 구간이며 보호 테스트 T가 아니다.

