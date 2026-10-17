# N7 독립 산술 감사 준비

현재 상태는 **PREP_ONLY**다. 이 준비 작업은 실제 부모 refit·alias 예측·FP64 재채점·CPU6 재추론·비용 측정·감사 실행을 하지 않았다. 실제 진행 중 작업·프로세스를 조회하지 않았고, 원 체크포인트/NPZ/원자료/모델 payload와 실제 통계·trial 수치를 해석하지 않았다. 기존 frozen 소스·결과·PLAN은 수정하지 않는다.

소유 파일은 `audit_multik_resume.py`, `test_audit_multik_resume.py`, 이 문서뿐이다. `README_PREPARATION_KO.md`는 원 runner의 동결 문서이므로 별도로 보존한다.

## 목적과 한계

원 MASTER 공개본 §11(389–414행)의 다중 K 비교 중, 이미 등록된 Beijing B0의 shared `[3,6,11]`/max-only `[11]` 두 부모를 저장된 refit CP0에서 마무리한 후 여섯 prefix 경로를 검산하기 위한 코드다. 기존 독립 K 모델·nested PCA와 동등한 학습 예산을 보장하거나 전체 E4/MASTER를 완료하는 코드가 아니다. 부모 두 개를 여섯 개의 독립 학습으로 세지 않는다.

원 search는 완료되어 있고 신규 optimizer update/F call은 0이어야 한다. refit의 3 epoch·2262 update·36189 exposure는 원 recipe를 완료한 것이며 수렴을 입증하지 않는다. `UNDERTRAINED_OR_BUDGET_LIMIT`, seed92601, 반복 개발 V의 한계를 유지한다. 중단 직전의 미저장 작업은 `UNKNOWN_NOT_LOGGED`이고 uninterrupted bitwise parity를 주장하지 않는다. 원 search를 복원하는 새 호출 시간, 저장된 과거 segment 시간, 이번 refit 시간·과거 누적 F 카운터를 분리한다.

## 기본 호출과 실제 입장

기본 호출은 입력/출력 0이며 NumPy/Torch를 import하지 않는다. 실제 감사는 ROOT가 모든 30 POST 단계와 두 부모 guard의 종료·원 핀·현재 PID 생성시각 부재를 확인한 뒤 명시적으로 실행한다. 현재 감사자는 프로세스를 조회하지 않고 실제 기록된 guard/SPEC/STARTED/tree-join/remaining·error 목록을 대사한다.

```text
audit_multik_resume.py --execute --root-triggered
  --public-root <campaign public root> --cache-root <campaign cache root>
  --plan-sha 886ee23f88dcc4520fda9a409f883193bd8199bc879bbd4f8f0b66f1884a21cc
  --dispatch-sha <actual two-parent RESULT SHA>
  --dispatch-spec-sha <actual SPEC SHA>
  --dispatch-guard-sha <actual two-parent guard SHA>
  --post-sha <actual complete 30-step POST RESULT SHA>
  --audit-sha <current reviewed audit source SHA>
  --output <new JSON inside this namespace>
```

준비 시 미래 `dispatch_sha`, `dispatch_spec_sha`, `dispatch_guard_sha`, `post_sha`는 모두 **null/미확정**이다. 새 출력은 CreateNew이며 기존 파일이 있으면 입력을 읽기 전에 거부한다. 실제 dispatch/admission/PLAN seal·후처리 큐를 시작하지 않는다. 원 worker1792222550/campaign1792223150 마감·CPU2/RSS3GiB·기존 storage 제한은 원 실행 코드가 유지하며 본 감사는 이를 확장하지 않는다.

## 검사 계약

- 원 PLAN/source39+immutable9·의존 소스·실제 dispatch/POST 소스를 SHA로 결속한다. 원/백업 CP20과 선택 artifact·V archive는 **opaque 바이트 SHA/크기**만 확인한다. tensor/NPZ 수치 복원은 하지 않는다.
- 두 부모는 `COMPLETE_SAVED_POINT_REFIT_NO_V_YET`이며 BASE의 `full_V_completed=false`를 보존한다. 별도의 실제 predict/manifest만 여섯 V 완료를 입증한다. 원 spec/grid·A50/A60 scaler·초기화 contract·source 복사·선택 state SHA·저장점 provenance를 대사한다.
- alias는 C11/L512/H48/V43/20539 observed labels·A60 scaler·target/mask/origin/channel 지문을 공유한다. 공개 충분통계는 dense 43×48×11 point 수와 다른 **1419 origin×channel×3 band 행/alias**다. 여섯 조건 총8514행을 frozen FP64 `math.fsum` 채널 macro 산술로 재집계하고 SCORES/manifest/별도 실제 full-array FP64 영수증과 원 공차(atol1e-9/rtol1e-7)로 대사한다.
- 실제 CPU6는 4+2, 원 공차1e-5/1e-4, 같은 여섯 origin과 K별 F streams·row_calls·physical_calls를 요구한다. 검산 통과를 개선이나 독립 일반화로 바꾸어 말하지 않는다.
- 비용은 여섯 alias×B1/B8=12조건, 각각10 warm/20 timed/20 stage=50행, 총600행이다. B0 원 Route가 실제 기록하는 여섯 단계 열만 허용한다. 보통 whole trial의 단계 칸은 빈 문자열이어야 한다. 전체 physical 경로·CPU2·실제 stream/call·입력/마스크·origin·fresh120초 quiet·실제 argv validation proof SHA를 결속한다. 원 summary의 fsum mean/median/min/max/선형 p90를 **exact** 대사하며 ULP 차이도 새 공차로 통과시키지 않는다.
- N7 native 비용 영수증에는 raw total F counter·validation proof SHA map·출력 dtype가 없다. 이를 요구하거나 0/float64로 채우지 않는다. 공개 trial에서 독립 합산한 counter와 실제 argv proof/quiet 바인딩을 별도 필드로 기록한다. `memory_scope`는 원 memory dict이며 전체 resident weights와 F resident 실측값을 혼동하지 않는다.
- 원 부모/alias가 partial이거나 검산·공통 closure가 부족하면 `NOT_COMPLETE_AUDIT_EVIDENCE_REJECTED`로 남긴다. 자동 retry/기존 결과 수정·실패를 PASS로 바꾸는 경로가 없다.

## 출력 API

`INDEPENDENT_MULTIK_RESUME_AUDIT_v1`에 `parents[2]`, `accuracy[6]`, `cost_conditions[12]`, `rows[6]`, 입력 상대경로/SHA·보존 결과·실제 guard evidence를 기록한다. `parents`는 원 BASE 필드를 유지하며 `stage_F_counters`를 별도로 둔다. `rows[].cost_B1/cost_B8`에 `summaries`, 원 `memory_scope`, `guard_sampled_peak_RSS_bytes`, `derived_trial_F_counters`가 있다. 미기록 native 필드는 null 및 `NOT_RECORDED_BY_ORIGINAL_N7_RECEIPT` 범위다. `reference_rows=[]`이며 기존 nested PCA/독립 K6 참고는 별도 보고 builder가 기존 결과 재사용으로 표시한다. private 절대경로·원자료/예측 배열·checkpoint는 공개하지 않는다.

## 준비 검증 수준

구현 부재 RED 1개 실패/12개 skip 후 GREEN을 확인했고, 정수 argv/float STARTED deadline의 동등성 fixture를 추가해 별도 RED→GREEN을 확인했다. 최종 **stdlib 합성16개 PASS**, 기본0 I/O/NumPy0/Torch0다. 서로 다른 채널 count의 macro-vs-global, 실제 B0 여섯 단계·동적 K counter, missing/duplicate/wrong trial, 1 ULP summary 거부, guard tree/owned PID, partial 거부 등을 검사했다. 실제 조건·CP 복원·전체30단계 integration·실제600trial/8514통계 산술·실제 감사 결과는 **미검증/미실행**이며 ROOT의 종료 후 실행이 필요하다.
