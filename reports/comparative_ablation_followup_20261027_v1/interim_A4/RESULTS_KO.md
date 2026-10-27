# 비교·절제 후속 실험 결과

[확인] 단계=interim; 수집 RESULT.json=204개; 후속 완료=40/50개. 원 RAW 4개·OURS 40개는 등록 G1_REUSE_AUDIT를 통해 별도로 재사용한다.
[확인] 실행 상태=RUNNING; 중단 이유=없음/미기록.
[확인] 첫 새 학습부터 경과=11239.8초 (3.12시간); 시작 허용 예산 잔여=35560.2초. 한도는 실제 경과 13시간 또는 2026-10-28 00:00 KST이며 진행 중 job만 마친다.
[확인] 입력 스냅샷은 REPORTING_INPUTS.json, 무수정 등록 분석은 registered_analysis/, 판정은 DECISION.json·ABLATION.csv·COMPARATIVE.csv에 보존한다.

```text
dataset              method n/10 median V MSE IQR        effect%   decision / primary / cap
jena                 A1     10/10 0.766177     0.026572   9.80      CONTRIBUTES_PROVISIONAL; extended=False; cap=10; undertrained_at_10=False
jena                 A4     10/10 0.685574     0.051290   -1.75     UNCLEAR; extended=False; cap=4; undertrained_at_10=False
ETTh1                A1     10/10 0.279226     0.003130   -0.15     UNCLEAR; extended=False; cap=3; undertrained_at_10=False
ETTh1                A4     10/10 0.715743     0.000000   155.95    CONTRIBUTES; extended=False; cap=0; undertrained_at_10=False
ETTh2_safe17400      A1     10/10 0.263365     0.003370   -5.00     UNCLEAR_PROVISIONAL; extended=False; cap=10; undertrained_at_10=False
ETTh2_safe17400      A4     10/10 0.901934     0.033964   225.34    CONTRIBUTES; extended=False; cap=0; undertrained_at_10=False
appliances           A1     10/10 0.283612     0.003923   0.56      UNCLEAR; extended=False; cap=1; undertrained_at_10=False
appliances           A4     10/10 0.334393     0.003928   18.57     CONTRIBUTES; extended=False; cap=0; undertrained_at_10=False
```

주 집계 (jena·ETTh1·appliances, 잠정 판정도 포함하므로 확정 기여 수가 아님):

```text
method  CONTRIBUTES HARMS UNCLEAR INCOMPLETE provisional
A1      1           0     2       0          1
A4      2           0     1       0          0
```

[확인] effect(%)는 variant와 OURS 중앙값 차이를 OURS 중앙값으로 나눈 값이다. 등록 절제 판정은 절대 effect와 두 IQR 중 큰 spread를 비교한다. 예측의 +50%·10% 기준으로 판정을 바꾸지 않는다.

D0 고정 PCA 하한:

```text
dataset              A60 floor  V floor    outside energy C1 same cohort excluded
Pretraining D0 evidence: ../preflight/D0_SUBSPACE_FLOOR.csv (unchanged).
```

D1 학습 디코더 하한 (주 결과 seed별 비율의 중앙값):

```text
dataset              method n/10 floor      actual     median ratio status
jena                 C2     10/10 0.233940   0.715067   0.327237     PASS
jena                 C3     10/10 0.236901   0.715753   0.330043     PASS
jena                 A4     10/10 0.202541   0.685574   0.293243     PASS
ETTh1                C2     10/10 0.422568   0.678325   0.627389     PASS
ETTh1                C3     10/10 0.413784   0.666153   0.629555     PASS
ETTh1                A4     10/10 0.461371   0.715743   0.644605     PASS
ETTh2_safe17400      C2     10/10 0.773230   1.091104   0.708615     PASS
ETTh2_safe17400      C3     10/10 0.798323   1.137333   0.692266     PASS
ETTh2_safe17400      A4     10/10 0.691942   0.901934   0.767176     PASS
appliances           C2     10/10 0.158015   0.324931   0.485002     PASS
appliances           C3     10/10 0.155449   0.332662   0.471722     PASS
appliances           A4     10/10 0.144273   0.334393   0.424535     PASS
```

# Fixed planner prediction check

[확인] 아래 기준은 후속 지시문 §5의 사전 고정 예측이며 등록 판정을 바꾸지 않는다.
A4 effect(%) = 100 × (A4 median − OURS median) / OURS median.
ETTh1·ETTh2: effect ≥ +50%. Jena·Appliances: |effect| < 10% (경계 10%는 불일치).
D1: ETTh1·ETTh2에서 A4·C2·C3 각각의 **job별 하한/같은 대상점 실제 MSE 비율의 중앙값** ≥ 0.5.
D1 집계는 두 중앙값의 비율이 아니다. 원문은 집계 방식을 명시하지 않아 데이터별 중앙값 보고 규칙에 맞췄다.
n<10 또는 정해진 seed가 빠지면 INCOMPLETE이며 예측의 성공·실패를 주장하지 않는다.

```text
kind       dataset              method n/10 value           prediction status
A4_EFFECT  jena                 A4     10   -1.747846       MATCH
A4_EFFECT  ETTh1                A4     10   155.949048      MATCH
A4_EFFECT  ETTh2_safe17400      A4     10   225.337055      MATCH
A4_EFFECT  appliances           A4     10   18.571226       NOT_MATCH
D1_RATIO   ETTh1                A4     10   0.644605        MATCH
D1_RATIO   ETTh1                C2     10   0.627389        MATCH
D1_RATIO   ETTh1                C3     10   0.629555        MATCH
D1_RATIO   ETTh2_safe17400      A4     10   0.767176        MATCH
D1_RATIO   ETTh2_safe17400      C2     10   0.708615        MATCH
D1_RATIO   ETTh2_safe17400      C3     10   0.692266        MATCH
```

[추정] D1의 큰 비율은 출력 공간 제약과 수준 이동 설명에 부합할 수 있으나 마지막 관측값·lag의 개별 인과 효과를 식별하지 않는다.


남은 범위와 해석:

- [확인] A4 대 OURS는 D 전체(마지막 관측값 + 변화량 + lag)의 효과다. A5 미실행으로 마지막 관측값과 lag를 분리하지 않는다.
- [확인] 실행 범위는 첫 캠페인 결과를 본 뒤 발표상 공백을 메우기 위해 선택했다. job 정의·seed·판정 규칙은 첫 사전 등록을 그대로 사용했다.
- [확인] A3(E의 효과), A5(lag만의 효과), C2·C3 연장, A1/ETTh2 연장, A4 연장은 NOT_RUN_SCOPE다. 예산 실패와 구분한다.
- [확인] 필요한 A4 연장이 범위 밖이면 PROVISIONAL을 유지한다. A1/jena 연장도 10 seed 전부 있어야 주 결과로 채택한다. 상한 10 epoch를 5개 이상 고르면 undertrained_at_10=True를 그대로 보고한다.
- [확인] 4개 데이터를 모두 보고하며 주 집계는 jena·ETTh1·appliances 3개다. ETTh2_safe17400는 보조 보고다. n<10 그룹은 INCOMPLETE다.
- [확인] Jena 비교의 TIE_PROVISIONAL을 포함하여 미완료 비교 연장에 따른 잠정 판정은 이번 P·D 결과로 해제하지 않는다.
- [확인] V는 개발 검증 구간이며 보호 테스트 T가 아니다. 학습·선택은 A_fit·A_stop/A60에 한정한다. T 숫자를 읽은 job은 통과할 수 없다.
- [추정] 높은 D1 비율은 출력 공간이 오차의 큰 부분을 설명함을 뒷받침하지만 수준 이동 하나만을 원인으로 확정하지 않는다. D0는 고정 PCA C1에만 직접 적용한다.

등록 비교 판정:

```text
jena                 TIE_PROVISIONAL
ETTh1                WIN_PROVISIONAL
ETTh2_safe17400      WIN_PROVISIONAL
appliances           WIN_PROVISIONAL
```

실행·미실행 목록, 실제 경과시간, 예산·마감·중단 근거:

```json
{
  "active": {},
  "attempts": {
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
  "elapsed_seconds": 11239.834955215454,
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
  "remaining_seconds": 35560.165044784546,
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
  "status": "RUNNING",
  "stop_reason": null,
  "updated_utc": "2026-10-27T05:04:24.703022+00:00"
}
```

[확인] 기존 파일 보존 검사: PRESERVATION_REPORTING.json. 완료 산출물과 입력의 SHA-256: REPORTING_COMPLETION.json·REPORTING_INPUTS.json.
그림: FIG_ABLATION_UPDATED.png/.svg; 각 seed와 미완료 수: FIG_ABLATION_UPDATED.csv.
