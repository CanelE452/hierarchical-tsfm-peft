# P 가지 실험 확장 (Appliances·ETTh1·ETTh2) — CLI 실행 지시문

캠페인 이름 [새 이름]: `p_branch_appl_ett_20261029_v1`
- 새 코드: `research/p_branch_appl_ett_20261029_v1/`
- 새 보고서: `reports/p_branch_appl_ett_20261029_v1/`
- 새 캐시: `.cache/p_branch_appl_ett_20261029_v1/`

이 지시문은 계획자가 `2d3ffa94`(origin/main)의 코드·결과만 읽고 썼다. 전제가 틀렸으면 따르지 말고 멈춰서 근거(경로:행, 수치)와 함께 알려줘.

사용자가 이 지시문을 넘긴 것을 계획 승인으로 본다. 게이트나 중단 조건이 걸리지 않으면 중간 보고만 출력하고 끝까지 진행해.

**한눈에**
- Jena에서만 돌린 P 가지 실험(`P_TIME_ONLY`, `P_CHANNEL_ONLY`)을 appliances, ETTh1, ETTh2_safe17400에서 같은 방식으로 돌린다. 2개 × 데이터 3개 × 10 seed = 60회.
- 새 모델·새 레시피·새 판정 규칙은 없다. Jena 가지 실험 코드를 복사해 데이터 등록과 데이터별 상수만 넓힌다(3절의 허용 변경).
- 연장(10 epoch)은 하지 않는다. Jena 가지 실험과 다른 점이며, 이유는 3절에 있다.
- 예산 [추정]: 약 9시간, 최대 약 11시간. 학습 마감은 2026-10-30 09:00 KST다.
- T(보호 테스트 구간)는 열지 않는다. 끝나면 새 캠페인 폴더만 커밋하고 push한다(4절 Step 3의 7).

## 0. 목적

- 발표 16장 절제 표에서 "Time branch only", "Channel branch only" 두 줄의 Appliances·ETTh1·ETTh2 칸이 비어 있다("—: not run"). Jena와 같은 규칙으로 채운다.
- 질문: P 전체(OURS)에서 한 가지만 남기면 데이터별 V MSE가 얼마나 바뀌나? P 전체를 빼도(No P) 뚜렷한 차이가 없던 데이터에서 두 가지가 서로 반대로 작용하는지도 본다.
- 결과를 보는 사람: INFORMS 2026 발표 청중과 교수님 사전 검토.
- 결과로 16장 표의 빈칸을 채우고, 셋째 bullet("P helps only on Jena, and its size may explain part of that")을 유지할지 고칠지 정한다.
- 해석 범위:
  - 개발 V의 탐색 결과다. 유의성·동등성 주장을 하지 않는다.
  - 실행 범위는 기존 결과를 본 뒤 정했다. 보고서에 그대로 밝힌다.
  - 공동 LayerNorm과 E·D의 채널 결합이 남으므로 완전한 시간/채널 독립 실험이 아니다(Jena 사전 기록과 같음).

용어: V(개발용 검증 구간), T(보호 테스트 구간), A_fit(학습 구간), A_stop(epoch 선택 구간), A60(선택 후 재학습 구간), IQR(사분위 범위, Q3 − Q1), cap(학습 epoch 상한).

## 1. 확인된 사실 (계획자가 `2d3ffa94`에서 직접 읽음)

### 재사용할 Jena 가지 실험
- [확인] 코드: `research/jena_p_branch_20261027_v1/`의 models.py, runtime.py, run_one.py, controller.py, prepare.py, postprocess.py, package.py.
- [확인] 등록 소스 hash(`reports/jena_p_branch_20261027_v1/PREREGISTRATION.json`의 `registered_sources`)와 현재 파일이 같다.
  - models.py `3ab8ad80…`, runtime.py `e35d8f26…`, run_one.py `5890efd2…`, controller.py `79d9d103…`, prepare.py `2819817f…`
- [확인] 가지 구성(`models.py:12–38`):
  - 같은 seed로 전체 모델을 만든 뒤 `model.front.mixer`를 `SingleBranchMixer`로 바꿔 한 가지(LayerNorm + 잔차 MLP)만 남긴다.
  - P_TIME_ONLY는 `temporal_norm`·`temporal`을, P_CHANNEL_ONLY는 `feature_norm`·`feature`를 남긴다.
- [확인] 레시피(Jena `PREREGISTRATION.md`):
  - AdamW lr 0.001, weight decay 0, gradient clip 1, effective batch 16, microbatch 4, FP32, L 512, H 48.
  - checkpoint 0/.25/1/3(`run_one.py:35–40`).
  - A_stop에서 epoch를 고르고, A60으로 새로 학습한 뒤 V로 평가한다.
- [확인] 판정 규칙(`postprocess.py:251–254`):
  - spread = 두 대상 IQR 중 큰 값, gap = 중앙값(변형) − 중앙값(OURS).
  - gap > spread면 FULL_P_LOWER, gap < −spread면 VARIANT_LOWER, 아니면 UNCLEAR.
  - provisional이면 `_PROVISIONAL`을 붙인다.
- [확인] Jena 결과(바꾸지 않음): P_TIME_ONLY +0.2%(UNCLEAR), P_CHANNEL_ONLY +8.9%(FULL_P_LOWER).
  - cap을 고른 seed가 4/10, 1/10이라 연장이 필요 없었다(`reports/jena_p_branch_20261027_v1/CONTRASTS.csv`, `SUMMARY.csv`).
- [확인] 마감 처리: 마감 전에 끝나지 않을 것으로 추정되는 job은 시작하지 않는다(`controller.py:142–145`, 마감은 사전 기록의 `training_deadline`).

### Jena 전용이라 고쳐야 하는 곳 [확인]
- `runtime.py:18` `KS = dict(jena=6)`
- `run_one.py:18–26` job 검사: dataset == 'jena', K == 6, id `JPB_{method}_jena_K6_s{seed}`
- `prepare.py:48–63`, `84`: job 목록, 참조 run 선택(연장 A1 포함), 참조의 K == 6·normalization_fit_end == 12579 검사
- `runtime.py:226–232` `assert_protected_T`
  - Jena 묶음 loader의 구성원 목록(x.npy 등)을 요구한다.
  - Jena가 아닌 데이터의 접근 기록(`runtime.py:94–95`: prefix_path, config_path, prefix_sha256, T_numeric_values_parsed=0)에는 그 키가 없다. 그대로 두면 실패한다.
- `postprocess.py:51–55`, `66`, `73`
  - V 모양 (43, 48, 21), T 시작 16772
  - 정규화 정보가 없어도 되는 예외 archive 목록(`HISTORICAL_jena_K6_s92601`, `_s92602`)
- `postprocess.py:295–328`, `package.py:29`: 그림·파일 이름
- `controller.py:186–225`: 연장 판정과 실행

### 세 데이터의 고정값 [확인]
출처: `reports/comparative_ablation_followup_20261027_v1/RUNS.csv`, `reports/k_tau_rule_20261023_v1/CURRENT_DATA_FINGERPRINTS.json`, `runtime.py:19–20`.

| | appliances | ETTh1 | ETTh2_safe17400 |
|---|---|---|---|
| C → K | 26 → 10 | 7 → 4 | 7 → 4 |
| A_fit 끝 / A_stop 끝(= normalization_fit_end) | 9867 / 11841 | 8710 / 10452 | 8700 / 10440 |
| V_select | [13814, 15788] | [12194, 13936] | [12180, 13920] |
| T_final 시작 | 15788 | 13936 | 13920 |
| V 모양 (origin, H, C) | (41, 48, 26) | (36, 48, 7) | (36, 48, 7) |
| OURS 중앙 V MSE (10 seed) | 0.2820 | 0.2796 | 0.2772 |
| Raw V MSE | 0.3476 | 0.2834 | 0.2596 |
| OURS 학습 파라미터 | 37,114 | 34,023 | 34,023 |

- [확인] 참조 run: 세 데이터 모두 OURS·A1(No P)·A2가 seed 92601–92610 각 10개다. 연장 행은 없다.
- [확인] ETTh1·ETTh2의 OURS 중 `HISTORICAL_<dataset>_K4_s92601`, `_s92602` 2개는 Jena처럼 예전 archive다. 나머지 OURS는 `KTAU_*` job이다.
- [확인] 16장의 기존 값(위 RUNS.csv로 다시 계산해 일치 확인):
  - No P: +0.6% / −0.1% / −5.0%
  - 같은 크기 대조(A2): −1.6% / −0.8% / +0.5%
- [확인] No P가 cap을 고른 seed: 1/10, 3/10, 10/10. ETTh2의 No P는 연장하지 않아 잠정이다.
- [계산] 가지 변형의 학습 파라미터 수. 같은 식이 Jena 실측(34,320, 2,405)과 맞는다.
  - 시간 가지 = 2·512·32 + 32 + 512 = 33,312
  - 채널 가지 = 65C + 32

| | appliances | ETTh1·ETTh2 |
|---|---|---|
| P_TIME_ONLY (OURS − 채널 가지) | 35,392 | 33,536 |
| P_CHANNEL_ONLY (OURS − 시간 가지) | 3,802 | 711 |

- [확인] 참조 run 환경(`reports/comparative_ablation_followup_20261027_v1/ENVIRONMENT_VALIDATION.json`): Python 3.11.16, torch 2.10.0+cu128, RTX 4070. F_SHA `c2b4edf7…`(`runtime.py:21`).

### 시간
- [확인] Jena 가지 실험 첫 seed의 실제 소요는 697초, 664초다(`BRANCH_SMOKE_VALIDATION.json`). Jena A2의 search+refit 중앙값 10.1분과 비슷하다.
- [확인] 그래서 A2 시간을 대용으로 쓴다. A2 search+refit 중앙값(최대):

  | 데이터 | 분 |
  |---|---|
  | appliances | 7.4 (9.4) |
  | ETTh1 | 8.2 (12.9) |
  | ETTh2_safe17400 | 8.2 (8.4) |

- [계산] job마다 부대 시간 약 1분[추정]을 더하면 60회에 약 9시간, 최대값 기준 약 11시간이다.

## 2. 확인이 필요한 전제

- [추정] 3절의 허용 변경만으로 세 데이터에서 돈다.
  - → 학습·선택·평가 로직을 바꿔야 하면 멈추고 보고한다.
- [추정] `runtime.py:16–17`의 DEFAULT_SOURCE·DEFAULT_ASSET과, `load_task`(`runtime.py:70–114`)가 읽는 데이터·결과 캐시가 지금도 있다.
- [추정] 참조 run의 prediction 파일이 남아 있고 hash가 RUNS.csv와 같다.
- [추정] ETTh1·ETTh2의 HISTORICAL archive 2개에는 mean·std가 없다(Jena와 같은 경우).
  - → 다른 archive에도 없으면 목록을 보고한다.
- [추정] 실행 환경이 참조 run과 같다.

## 3. 승인된 고정값 (바꾸지 말 것)

### 실행 대상과 순서
- method: P_TIME_ONLY, P_CHANNEL_ONLY(Jena와 같은 정의)
- dataset·K: appliances K10, ETTh1 K4, ETTh2_safe17400 K4
- seed: 92601–92610
- job_id [새 이름]: `PBA_{method}_{dataset}_K{K}_s{seed}`
- 순서: seed → dataset(appliances, ETTh1, ETTh2_safe17400) → method(P_TIME_ONLY, P_CHANNEL_ONLY). 중간에 멈춰도 데이터·방법 사이에 seed 수가 고르게 남는다.
- 참조: OURS, A1, A2는 새로 학습하지 않는다. followup RUNS.csv의 10 seed를 재사용한다.

### 연장하지 않음 (Jena 가지 실험과 다른 점)
- cap 3을 고른 seed가 5/10 이상이어도 연장하지 않는다.
  - 해당 (dataset, method)의 연장 상태를 NOT_RUN_SCOPE로 기록하고 `_PROVISIONAL`을 붙인다.
- 이유:
  - 세 데이터의 비교 대상(OURS, No P, A2, 선형 디코더)은 모두 cap 3으로만 학습됐다. 연장하면 같은 열 안에서 학습 길이가 섞인다.
  - 예산도 줄어든다.

### 허용되는 코드 변경 (이 밖은 금지)
1. 캠페인 이름·경로 상수, job_id 형식.
2. 데이터 등록
   - KS = {appliances: 10, ETTh1: 4, ETTh2_safe17400: 4}
   - job 검사와 job 목록
   - 참조 run 선택: 세 데이터의 OURS·A1·A2 기본 10 seed, 참조의 K·normalization_fit_end 검사는 1절 표 값으로
3. 연장
   - 연장 job은 실행하지 않는다.
   - provisional 조건(`postprocess.py:218`)에 NOT_RUN_SCOPE를 더한다.
4. T 보호 검사
   - Jena가 아닌 데이터는 `load_task`가 이미 하는 확인(PREFIX_PHYSICAL.npz hash 일치, 배열 길이 = T_final 시작, `runtime.py:85–104`)과 T_numeric_values_parsed = 0을 결과에 기록하고 검사한다.
   - Jena용 검사는 지우지 않는다.
5. postprocess의 데이터별 상수
   - 1절 표의 V 모양과 T 시작
   - 예외 archive 목록: ETTh1·ETTh2의 `HISTORICAL_*_s92601`, `_s92602`
   - 그림·파일 이름
6. 마감: 사전 기록의 `training_deadline` = `2026-10-30T09:00:00+09:00`. job 시간 추정은 1절 시간표의 데이터별 최대값 + 1분으로 시작하고, 첫 seed 실측으로 갱신한다.

- 원본과의 diff를 `CODE_DIFF.patch`로 저장한다.
- 레시피·선택·평가·판정 로직이 diff에 들어가면 멈추고 보고한다.

### 그대로 유지
- 레시피·epoch 선택·지표·판정 규칙: Jena 가지 실험 코드 그대로.
- 한 번에 한 실행.
- 같은 설정의 기술적 실패는 1회 재시도한다. 과학적 게이트 실패는 설정을 바꿔 우회하지 않는다.

### 마감 뒤
- 마감 전에 끝나지 않을 것으로 추정되는 job은 시작하지 않는다. 진행 중인 job은 마친다.
- (dataset, method)마다 10 seed가 다 있어야 주 결과로 쓴다. 모자라면 INCOMPLETE로 n과 함께 따로 보고한다.

## 4. 단계

중간 보고는 출력만 하고 멈추지 말고 이어서 진행해. 멈추는 건 게이트나 중단 조건이 걸렸을 때뿐이야.

### Step 0 — 사전 점검 (읽기 전용, 학습 없음)
1. 6절 게이트 G1–G9를 확인한다.
2. 중간 보고 1: 게이트 표, 예산 추정.

### Step 1 — 코드와 사전 기록 (학습 전)
1. Jena 코드 7개를 새 폴더로 복사하고 3절의 허용 변경만 한다. `CODE_DIFF.patch`를 저장한다.
2. 새 보고서 폴더에 아래 파일을 쓴다.
   - `PREREGISTRATION.md`/`.json`: 이 지시문 0·3·5절 원문, job 목록, 예산, 실행 범위를 기존 결과를 본 뒤 정했다는 사실
   - `RUN_PLAN.json`
   - `REFERENCE_RUNS.json`: 참조 90행과 hash 확인 결과
   - `PRESERVATION_BASELINE.json`
3. 위 파일을 로컬 hash로 봉인한다(`REGISTRATION_SEAL.json`). commit은 하지 않는다.
4. 봉인 전에는 어떤 학습도 하지 않는다.

### Step 2 — 학습
1. 3절 순서대로 실행하고, 실행마다 시간을 기록한다.
2. 첫 seed의 6개 job이 끝나면 아래를 확인한다.
   - Jena `BRANCH_SMOKE_VALIDATION.json`과 같은 항목: cohort 일치, 저장 배열로 FP64 지표 재계산, T_numeric_values_parsed = 0
   - 학습 파라미터 수가 1절 표와 같은지
   - 통과하면 중간 보고 2(스모크 결과, 실제 시간, 남은 예산)를 출력하고 계속한다.
3. 이후 seed마다 한 줄로 보고한다: 완료 수, 경과 시간, 남은 추정.
4. V 값을 보고 순서·범위·seed를 바꾸지 않는다.

### Step 3 — 분석과 보고 (모두 새 보고서 폴더에)
1. Jena 가지 실험과 같은 열로 `ALL_NEW_RUNS.csv`, `SUMMARY.csv`, `CONTRASTS.csv`, `PAIRED_DIFFERENCES.csv`를 만든다.
   - 데이터마다 P_TIME_ONLY, P_CHANNEL_ONLY, A1, A2를 각각 OURS와 비교한다.
2. 검산: 다시 계산한 A1·A2 effect가 1절의 16장 값과 소수 첫째 자리까지 같아야 한다. 다르면 멈추고 보고한다.
3. `FIG_P_BRANCHES_APPL_ETT.png`/`.svg`
   - 데이터별 패널, 10 seed 점과 상자(OURS, P_TIME_ONLY, P_CHANNEL_ONLY, No P)
   - Raw는 점선, PROVISIONAL은 표시
4. `SLIDE_NUMBERS.md`: 16장 표 두 줄의 Appliances·ETTh1·ETTh2 값
   - 부호 있는 퍼센트, 소수 한 자리(예: +0.6%, −5.0%)
   - 칸마다 판정 레이블, cap을 고른 seed 수, n
   - Jena 값(+0.2%, +8.9%)은 바꾸지 않고 옆에 적는다.
   - 데이터별 두 변형의 학습 파라미터 수
5. `PREDICTION_CHECK.md`: 5절 계획자 예측이 맞았는지.
6. `RESULTS_KO.md`: 표, 판정, 예측 확인, 남은 공백(PROVISIONAL, INCOMPLETE), 실제 시간, 중단 여부와 이유.
7. 마무리
   - 보존 파일 hash를 시작 전과 비교한다.
   - 묶음 zip을 만든다.
   - `research/p_branch_appl_ett_20261029_v1/`와 `reports/p_branch_appl_ett_20261029_v1/`만 커밋하고 origin main에 push한다. `.cache/`와 다른 파일은 커밋하지 않는다. 계획자가 GitHub에서 이 결과를 읽어 16장 표를 채운다.
   - 마감이나 중단 조건으로 멈췄어도, 그때까지의 결과와 INCOMPLETE 보고를 같은 방식으로 커밋·push한다.

## 5. 정의와 계획자 예측

### 정의 (Jena와 같음)
- effect_pct = 100 × (중앙값(변형) − 중앙값(OURS)) / 중앙값(OURS). +면 그 가지만 남겼을 때 나빠진다.
- 판정 레이블: 1절. 기술적 비교이지 유의성·동등성 검정이 아니다.
- PROVISIONAL: cap 3을 고른 seed가 5/10 이상.

### 계획자 예측 (판정이 아니다. 맞았는지만 보고한다)
- [추정] P_TIME_ONLY: 세 데이터 모두 UNCLEAR.
- [추정] P_CHANNEL_ONLY: appliances·ETTh1은 UNCLEAR. ETTh2는 중앙값이 OURS보다 낮고(No P의 −5.0%처럼), cap을 고른 seed가 5/10 이상이라 PROVISIONAL.
- 근거: No P의 effect(+0.6%, −0.1%, −5.0%)와 cap 선택(1/10, 3/10, 10/10). P_CHANNEL_ONLY는 큰 시간 가지를 뺀 작은 모델이라 No P에 가깝다고 본다.

## 6. 게이트

아래 항목을 실행 **전에** 확인하고, 실패하면 그 자리에서 멈추고 보고해줘. 멈추는 게 기본값이야. 대충 맞으니 진행하지 마.

- G1 코드
  - HEAD에 `2d3ffa94`가 포함된다.
  - 1절의 등록 소스 hash 5개가 같다.
  - 이 지시문이 지목한 파일·행·이름이 실제와 다르면 실제 구조를 보고한다.
- G2 산출물 중복: 새 캠페인 폴더(research, reports, .cache)가 이미 있으면 만들지 말고 경로를 보고한다.
- G3 데이터
  - 세 데이터에서 `load_task` 검사(지문, split, L 512·H 48, A_fit·A60 scaler 지문)가 통과한다.
  - split 값이 1절 표와 같다.
- G4 참조 run
  - 세 데이터의 OURS·A1·A2가 각 10 seed다.
  - K·normalization_fit_end가 1절 표와 같다.
  - prediction hash가 RUNS.csv와 같고, 저장 배열 모양이 1절 표의 V 모양과 같다.
- G5 가지 구성
  - 첫 seed에서 두 변형의 학습 파라미터 수가 1절 표와 정확히 같다.
  - 남긴 가지의 초기 가중치가 같은 seed 전체 모델의 그 가지와 같다.
- G6 F·환경
  - F_SHA가 같다.
  - Python·torch·GPU가 참조 run 환경과 같다. 다르면 멈추고 보고한다(참조 OURS를 재사용하므로).
- G7 누출
  - epoch 선택은 A_stop만 쓴다.
  - V는 선택·순서·범위 결정에 쓰지 않는다.
  - T 숫자는 읽지 않는다(모든 job에서 T_numeric_values_parsed = 0).
- G8 보존: 기존 `reports/`·`research/` 파일과 참조 prediction 파일의 hash가 시작 전후 같다.
- G9 GPU: 다른 학습이 GPU를 쓰고 있지 않다.

## 7. 제약
- 기존 데이터·checkpoint·결과 파일 수정 금지. 출력은 새 캠페인 폴더에만.
- 학습은 3절의 60개 job만. 그 밖의 학습·재학습·hyperparameter 변경 금지.
- seed·데이터를 유리한 쪽으로 다시 고르지 않는다. 전체를 그대로 보고한다.
- commit·push는 Step 3의 7에 적은 결과 커밋뿐이다. 브랜치 조작과 force push는 하지 않는다.

## 8. 보고
- [확인]/[추정] 태그를 달아줘. [확인]에는 경로:행 또는 수치를 인용해줘.
- 내 전제가 틀렸으면 틀렸다고 말해줘. 지시문을 따르는 것보다 오류를 잡는 게 우선이야.
- 최종 보고에는 `SLIDE_NUMBERS.md` 내용을 그대로 붙이고, `RESULTS_KO.md`를 요약해줘.
