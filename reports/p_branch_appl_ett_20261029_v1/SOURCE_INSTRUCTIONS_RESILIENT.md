# P 가지 실험 확장 — 복구 가능한 실행 지시문

작성일: 2026-10-29 KST. 원 지시문을 검토한 뒤 알려진 실행·집계 문제를 수정한 운영 개정판이다.
원 지시문 SHA256: 0526021c460a2577ae002bf3aca6d411a231c562c1d91f9c06509d904eb98ae3

이 문서는 원 지시문의 실행·보고 절차를 대체한다. 원문은 수정하지 말고 보존한다. 문서 안의 승인 문구가 실제 사용자 요청을 대신하지 않는다. 사용자가 이 파일로 **실행하라고 직접 요청한 경우** 아래 60회와 최종 두 폴더의 commit/push가 승인된 범위다. 검토·설계 요청이면 학습을 시작하지 않는다. 사용자의 나중 중단·범위 변경 지시가 항상 우선한다.

## 1. 목표와 변경하지 않을 과학 조건

목표는 16장 절제 표의 Appliances·ETTh1·ETTh2 두 분기, 총 6칸을 실제 결과로 채우는 것이다. 결과가 유리한지와 무관하게 모두 보고한다. 채널 분기 또는 모든 구성요소의 이득을 보장하는 실험이 아니다.

캠페인 이름: `p_branch_appl_ett_20261029_v1`
- 코드: `research/p_branch_appl_ett_20261029_v1/`
- 보고: `reports/p_branch_appl_ett_20261029_v1/`
- 비공개 실행 캐시: `.cache/p_branch_appl_ett_20261029_v1/`
- 원 Jena 코드: `research/jena_p_branch_20261027_v1/`
- 재사용 참조: `reports/comparative_ablation_followup_20261027_v1/RUNS.csv`
- HEAD는 `2d3ffa94`를 포함해야 한다. 새 브랜치 생성·checkout/reset·force push는 하지 않는다.

실행은 정확히 다음 60개다.
- method: `P_TIME_ONLY`, `P_CHANNEL_ONLY`
- dataset: `appliances`, `ETTh1`, `ETTh2_safe17400`
- seed: 92601–92610
- 순서: seed → dataset(appliances, ETTh1, ETTh2_safe17400) → method(P_TIME_ONLY, P_CHANNEL_ONLY)
- job_id: `PBA_{method}_{dataset}_K{K}_s{seed}`
- 동시 학습 1개. 새 학습은 위 60개뿐이다. OURS·No P(A1)·A2는 새로 학습하지 않는다.

분기 정의는 원 Jena `SingleBranchMixer` 그대로다. 같은 seed의 Full 모델을 먼저 만든 뒤 해당 가지의 원래 LayerNorm과 잔차 MLP를 남긴다. 남긴 가지를 다시 초기화하지 않는다. 공동 LayerNorm 및 E/D의 결합은 남는다. 완전한 축 독립 실험으로 부르지 않는다.

학습·선택·평가 조건:
- Frozen Chronos-Bolt-small, F_SHA `c2b4edf7db22d3f18e58ca79299a5c7e864e9b8f277e4e29e2f88d07dbb35874`
- Python3.11.16, torch2.10.0+cu128, RTX4070 환경을 검증한다. 설치·업그레이드로 임의 변경하지 않는다.
- L512, H48, FP32, AdamW lr0.001, weight_decay0, gradient_clip1
- effective_batch16, microbatch4, F_row_microbatch128, checkpoint_inputs=False, RevIN=False
- checkpoint 0/.25/1/3. A_fit 학습 → A_stop 최소 MSE로 epoch 선택(정확히 같으면 이른 epoch)·선택 봉인 → A60 fresh refit → V 평가
- 10epoch 연장 없음. 선택된 epoch·cap 선택 수는 모두 기록한다.
- 원 지표 정의·관측 mask·채널별 집계·판정식을 유지한다. 완전 관측일 때 원 global mean과 채널별 평균의 일치도 확인한다. Jena 전용 모양/관측 전제를 다른 데이터에 무조건 적용하지 말고 각 데이터의 실제 고정 cohort와 기존 지표 계약을 검증한다.
- V는 개발 탐색 결과다. V로 실행 순서·seed·범위·재시도·학습 길이를 바꾸지 않는다. 보호 T 숫자는 읽지 않는다.

고정 데이터 값:

```text
Dataset             C  K   A_fit_end A60_end V_start V_end=T_start V_shape
appliances         26 10      9867    11841    13814      15788      41x48x26
ETTh1               7  4      8710    10452    12194      13936      36x48x7
ETTh2_safe17400      7  4      8700    10440    12180      13920      36x48x7

Dataset             Full_params Time_only_params Channel_only_params
appliances                37114            35392                3802
ETTh1                     34023            33536                 711
ETTh2_safe17400            34023            33536                 711
```

참조 OURS/A1/A2는 데이터마다 각 10seed, 총90행이다. 현재 확인한 90개 archive에는 **모두 mean/std가 있다**. ETTh1·ETTh2 historical4개의 정규화 누락 예외를 새로 만들지 않는다. 이번 참조·신규 archive의 필수 정규화 metadata 누락은 검증 실패다.

등록 판정:
- effect_pct = 100 × (median_variant − median_OURS) / median_OURS
- spread = max(IQR_variant, IQR_OURS), gap = median_variant − median_OURS
- gap > spread: FULL_P_LOWER, gap < −spread: VARIANT_LOWER, 그 외 UNCLEAR
- 이 규칙은 기술통계이며 유의성·동등성 검정이 아니다.
- dataset×method가 정확히10seed일 때 cap3 선택이5/10 이상이면 `NOT_RUN_SCOPE`, `PROVISIONAL`. 새 두 분기뿐 아니라 재사용 OURS/A1/A2도 실제 cap 선택으로 계산하고 비교 양쪽의 잠정성을 전파한다.
- n<10은 `INCOMPLETE(n=...)`. 주 결과 10seed 표에 섞지 않는다. 유효한 완료 결과는 별도 공개한다.
- ETTh2 No P는 기존 cap3 선택10/10이므로 잠정 표시를 유지한다.

검산할 기존 effect(소수1자리):
```text
                      appliances ETTh1 ETTh2_safe17400
No_P(A1)                   +0.6%  -0.1%          -5.0%
A2                         -1.6%  -0.8%          +0.5%
```
Jena 기존 값 Time-only +0.2%, Channel-only +8.9%는 변경하지 않는다.

## 2. 허용되는 구현 변경

학습 전에 원7개 파일과 필요한 테스트를 새 코드 폴더에 복사/확장하고 `CODE_DIFF.patch`를 남긴다. 다음은 이번 개정에서 명시적으로 허용한 변경이다.

1. 새 namespace·경로·job_id·3개 데이터 등록·데이터별 K/fit/split/V shape/파라미터 수.
2. 60개 기본 job·연장0개·고정 순서의 등록/검증. 빈 extension 목록이 완료로 오인되지 않게 `bool(extended) and all(...)` 등 명시적으로 처리한다. 연장이 없으면 기본 결과를 선택한다.
3. 데이터별 cohort, dataset×method 집계·paired seed 비교·잠정성·INCOMPLETE 표기. 모든 데이터를 하나의 Jena canonical cohort나 하나의 method 집계로 합치지 않는다.
4. 0개/부분/전체 완료를 처리하는 후처리와 **결과 전용** 패키징. 원 package.py의 base_complete20·PPTX/PDF/대본/19장Office 검수 의존성을 제거한다. PPT·발표PDF·대본은 만들거나 수정하지 않는다.
5. 아래의 자원 대기·동일 설정 재시도·관리자 복구·실제 worker 식별·운영 timeout·입장 마감·원자적 상태 저장·시간 추정 갱신.
6. 비Jena loader의 prefix/config/array/split/scaler 지문·길이·보호T 경계 검증. 원 Jena bounded-loader 검증은 삭제하지 않는다.
7. 기계·경로·프로세스 상태의 검증과 로그 형식. 학습 수치를 바꾸는 최적화·배치 축소·dtype 변경·새 dependency 설치·판정식 변경은 허용하지 않는다.

원 Jena write guard는 보존파일 경로를 막는 guard다. crossdata에서 있었던 Windows NUL 오류가 반드시 재현된다고 단정하지 않는다. 새 guard를 추가한다면 TEMP/import/NUL의 정상 동작을 학습 전에 CPU smoke로 확인하되 T·기존 파일 보호를 약화하지 않는다.

## 3. 긴 실행 전에 반드시 끝낼 점검

사전 점검에서 발견한, 위 허용 범위의 코드/경로/상태/후처리 문제는 원인을 기록하고 **학습 전에 묶어서 수리·재검증**한다. 단순 문서 행 번호 차이나 이미 확인된 정규화 예외 정정 때문에 사용자 승인 대기를 다시 만들지 않는다. 모델·데이터·스케줄·지표의 실제 변경이 필요하면 별개 문제로 보고한다.

필수 검증:
- 원 Jena 등록 소스5개 SHA와 현재 파일 일치, 기준 commit 계보.
- 데이터3개의 실제 loader/지문/fit·split·scaler·L/H 및 pre-T prefix 경계.
- 참조90행의 정확한 dataset/method/seed/K/A60끝·prediction SHA·NPZ header/shape. origin과T경계를 확인한 뒤 허용V 수치만 읽는다. mean/std·정답·mask·채널 순서·origin을 데이터별 canonical과 비교하고 FP64로 기존 MSE/MAE를 검산한다.
- 두 분기 초기 가중치가 같은 seed Full의 해당 가지와 같고, 학습 파라미터 수가 위 표와 같음. 합성 CPU fixture로 확인할 수 있다.
- 실제 F·Python·torch·GPU 환경. 전혀 다른 모델·환경으로 참조를 조용히 재사용하지 않는다.
- 기존 research/reports/참조 예측의 시작 전 보존 hash와 출력 경계. 새 캠페인 자신의 파일은 baseline에서 제외한다.
- 다른 실제 학습·비용측정 worker 부재 또는 해제 대기, 공유 GPU 예약과 OS lock.

GPU 학습 전에 실행하는 합성 CPU 시험:
- 60개 job의 유일성·순서, 연장0개, dataset×method 각10seed.
- 빈 extension 목록에서도 기본 결과가 primary에 남는지.
- 서로 다른 데이터의 cohort를 섞으면 거절하는지.
- 0/1/9/10개 완료, 한 family10/다른 family9, 중복 seed, COMPLETE인데 RESULT 없음.
- cap 선택4/10·5/10, 재사용 No P의 잠정성 전파, n<10의 주 결과 제외.
- reference/scaler/shape/hash 불일치·T 경계 위반·비유한 값 거절.
- 정상 전체·부분·0개 완료에서 표·그림·빈 패널·보고·ZIP 생성 경로. 표는 결과 없음/미완료를 정직하게 표시하며 잘못된0%나 승리로 채우지 않는다.
- 기술 오류1회 재시도, 두 번 실패 후 다음 job, resource WAIT가 attempt를 소진하지 않음, 관리자 재시작 시 중복 worker 없음.
- PID 재사용, 살아 있는 worker, 실제 자식만 살아 있는 경우, 죽은 프로세스가 남긴 lock 파일, 진행 중 job의 캠페인 마감 통과.
- PPT 파일·대본이 없어도 최종 패키징 가능, cache/NPZ/checkpoint/secret/PPT의 공개 제외.

이 시험은 합성 테스트이며 본 실험 학습·V 탐색과 구분해 기록한다. 실제 모델을 학습해 보는 임의 pilot은 추가하지 않는다. 첫6개 실제 실행이 아래 smoke를 겸한다.

## 4. 사전등록과 재개 규칙

새 report에 `PREREGISTRATION.md/json`, `RUN_PLAN.json`, `REFERENCE_RUNS.json`(90행), `PRESERVATION_BASELINE.json`, `DATA_REGISTRY.json`, `CODE_DIFF.patch`, `CPU_TESTS.json`을 저장한다. 원 지시문과 이 개정 지시문도 copy/hash로 보존한다. PREREGISTRATION에는 과학 고정값·아래 운영 정책·허용 변경·실행 범위를 기존 결과를 본 뒤 정한 사실·각 추가 입력의 SHA를 기록한다.

`REGISTRATION_SEAL.json`으로 등록 문서와 전체 실행 소스/테스트를 학습 전에 봉인한다. run_one의 기존 '문서5개만 가능' 검사는 실제 등록된 정확한 문서 목록과 source 목록에 맞게 갱신한다. 최종 결과 전 commit은 하지 않는다. 로그가 있어도 봉인보다 앞선 실제 학습은 허용하지 않는다.

이미 폴더가 있을 때:
- 같은 캠페인 seal/RUN_PLAN/과학조건/입력 SHA가 모두 같고 상태가 검증되면 **재개**한다. 폴더 존재만으로 중단하거나 새 이름으로 완료 job을 다시 돌리지 않는다.
- seal이 없거나 다른 실행의 파일이면 덮어쓰지 않는다. 경로와 불일치를 보고하고 격리/정리를 사용자와 해결한다.
- 과거 terminal 파일은 `recovery_NNN/`에 원본 hash와 함께 보존하고 `RECOVERY_NNN.json`·`LAUNCH_RESUME_NNN.json`을 추가한다. 과거 STOPPED 문자열만 보고 현재 실행도 중단됐다고 단정하지 않는다.
- `PROGRESS.json`·실제 프로세스·lease/OS lock·검증된 RESULT를 함께 대조한다.
- 완료 RESULT·worker receipt·선택 봉인·모델/예측 SHA·cohort가 맞는 job은 COMPLETE로 재사용한다. 완료 학습을 다시 하지 않는다.

봉인 후 모델·runtime·run_one 등 등록된 학습 수치 코드를 무음 변경하지 않는다. 후처리/패키징의 구현 오류만 필요할 경우 원본을 유지하고 `postprocess_r2.py`/`package_r2.py` 같은 **새 분석 버전**과 수정 이유·diff·테스트·source SHA를 추가한다. 원 지표·판정·cohort·primary 선택은 유지한다. 최종 gate/manifest가 실제 사용한 분석 버전을 명시해야 한다.

## 5. 실행 관리자 — 기술 문제로 전체 종료하지 않기

### 5.1 상태와 영속 기록

상태는 READY → WAIT_RESOURCE 또는 RUNNING → COMPLETE / FAILED_TERMINAL / NOT_RUN_BUDGET로 구분한다. 과학 검증 실패는 SCIENTIFIC_SUSPEND, 사용자 중단은 USER_CANCELLED다. WAIT나 보고 실패를 TRAINING_COMPLETE로 표시하지 않는다.

각 상태는 임시파일+원자적 rename으로 저장한다. 필수 기록은 job_id, seed/dataset/method, attempt, phase, 실제 PID와 creation_time, 정확한 module/argv, 시작/최근 생존 시각, 종료코드, 로그 경로/SHA, 실패 분류다. private token/환경 credential/비공개 process command 원문은 공개하지 않는다.

로그가2분 안 늘었다는 이유만으로 worker를 죽이지 않는다. 로딩·보존 해시·평가에는 학습 event가 없을 수 있다. 관리자 heartbeat와 실제 process CPU/GPU/phase를 점검하는 경고로만 사용한다.

### 5.2 자원 대기와 실행 시도

worker를 띄우기 **전에** 다음을 확인한다.
- 동일 캠페인/다른 실제 학습·비용측정 worker와 GPU 메모리·공유 예약.
- 기존 OS lock 획득 가능 여부. lock 파일이 남았다는 사실만으로 busy라고 보지 않는다. 파일을 삭제하거나 lock을 강탈하지 않는다.
- PID·creation_time·정확한 module/job_id·실제 자식 프로세스의 소유 일치.

일시 충돌은 WAIT_RESOURCE로 두고60초 간격으로 다시 확인한다. 이미 등록한 입장 마감 이후에는 더 기다리며 신규 실행하지 않는다. 대기 사유·마지막 확인·누적 대기·다음 확인 시각을 기록한다. 이 대기에는 worker Popen/attempt 로그·디렉터리를 만들지 않고 attempt를 소진하지 않는다.

실제 worker가 시작되면 attempt를 소비한다. launch 뒤의 자원 race/실패를 조용히 지워서 attempt를 되돌리지 않는다. 모든 launch·실패는 보존한다.

**이번은 정확도 학습이지 비용 benchmark가 아니다.** Chrome/RustDesk/DWM 등 일반 desktop GPU context가 있다는 이유만으로 학습을 차단하지 않는다. 이전 비용측정용 background p95≤5%/max≤15% 기준을 새 학습 게이트로 끌어오지 않는다. 다른 실제 학습과 불명확한 compute 충돌은 확인·대기하되, GUI 새 PID 하나 때문에60개 job이 연속 실패하도록 설계하지 않는다. 다른 앱·서비스는 승인 없이 종료하지 않는다.

### 5.3 오류 분류와 재시도

종료코드20, ValueError, PermissionError라는 종류만으로 과학 실패를 단정하지 않는다. phase, 정확한 traceback, START/updates/학습시도 여부, source/data/scaler/F/selection/cohort/T 검증을 함께 확인한다.

- 알려진 일시적 인프라 오류(확인된 프로세스 소멸·일시 파일잠금·GPU 컨텍스트 장애·아래 wall timeout)는 원래 job/seed/레시피 그대로 **1회만 fresh 재시도**한다. 첫 시도 자료를 모두 남기고 optimizer/search/refit을 새로 시작한다. 중간 학습 결과를 성공 결과로 붙이지 않는다.
- 실패 worker와 자신의 실제 자식이 종료·GPU 예약 해제됐는지 확인한 뒤 retry/다음 job을 시작한다. 우리 캠페인 소유가 아닌 PID는 죽이지 않는다.
- 두 번 모두 기술 실패면 그 job을 FAILED_TERMINAL로 기록하고, 자원이 정상화됐으면 **다음 등록 job을 계속**한다. 전체60개 관리자를 종료하지 않는다. 해당 dataset×method가10개 미만이면 INCOMPLETE다.
- source/data/F/scaler/cohort/선택 봉인 불일치, T 접근, 비유한 결과 등 과학/무결성 실패는 SCIENTIFIC_SUSPEND. 설정을 바꿔 우회하거나 자동 `--resume-stopped`로 넘기지 않는다. 가능한 검증된 부분을 보고·패키징하고 원인을 알린다.
- 분류되지 않은 오류는 증거를 보존해 조사한다. 확인되지 않은 오류를 무조건 WAIT/technical로 바꿔 무한 반복하지 않는다.

### 5.4 관리자 복구와 프로세스 독립성

자동 복구를 수행할 별도의 CPU-only `supervisor.py` 또는 동등한 launcher를 새 코드 폴더에 둔다. 이 운영 보조 소스도 학습 전에 봉인한다. supervisor는 `supervisor.lock`으로 단일 인스턴스를 유지하고 관리자 생존·terminal 상태를 관찰한다. controller/worker의 OS lock은 각 소유 프로세스가 유지하며 supervisor가 대신 점유해 실행을 막지 않는다. 관리자 종료 후 재개를 실제로 실행하는 주체는 이 supervisor다. 재시작 전 동일 seal·영속 재시작 횟수·실제 worker를 검증하고, SCIENTIFIC_SUSPEND/USER_CANCELLED/의도된 terminal에는 재시작하지 않는다.

관리자는 Windows 숨김 background 프로세스로 실행하고 stdout/stderr를 새 cache에 남긴다. 현재 터미널이 닫혀도 worker가 유지돼야 한다. 시작 PID만 기록하지 말고 venv wrapper 아래 실제 Python PID/생성시각도 기록한다.

예상치 못한 **관리자 자체의 기술적 종료**는 동일 봉인/코드로 최대3회 재시작 가능하다. 과학 SUSPEND·사용자 중단·의도된 terminal은 재시작하지 않는다. 재시작 전 실제 worker가 살아 있으면 그 worker를 관찰·화해하고 새 worker를 중복 생성하지 않는다. worker가 없음을 확인하고 OS lock을 다시 획득한 뒤만 다음 작업을 시작한다. 실패 manager의 로그/상태를 지우지 않는다.

관리자 재시작은 각 job의 attempt 한도를 초기화하지 않는다. 실제 worker가 죽고 attempt1만 소진됐으면 attempt2로, 두 번 소진됐으면 다음 job으로 진행한다. 관리자 최대3회 재시작 횟수도 동일 캠페인의 영속 기록에서 이어받으며 관리자/감시자 재개로 초기화하지 않는다.

가능하면 관리자 수명 동안 process-scoped Windows sleep 방지(ES_SYSTEM_REQUIRED)를 사용하고 종료 시 해제한다. 영구 전원 설정·서비스·예약 작업·다른 앱을 임의 변경하지 않는다. 전원 차단·OS 강제재시작·사용자의 실제 중단을 막을 수 있다는 보장은 하지 않는다. 머신 복귀 후에는 동일 상태 검증으로 재개한다.

## 6. 시간 예산과 09:00 정책

**2026-10-30 09:00 KST는 신규 job 입장 마감**이다. 마지막 job의 실제 종료와 후처리 완료 시각을 보장하는 deadline은 아니다.

데이터별 초기 추정은 기존 A1/A2 중 최대 search+refit에60초를 더한 값으로 고정한다.
```text
Dataset            initial_estimate_seconds  fixed_hard_wall_seconds
appliances                        647.449                  2700.000
ETTh1                             852.185                  3408.740
ETTh2_safe17400                    873.753                  3495.012
```
초기값은 위로 반올림했다. hard wall은 학습 전에 고정한 `max(45분,4×initial_estimate)`이며 매번 관측치에 따라 늘리지 않는다.

새 job 입장 조건:
`now + current_estimate(dataset) + 60초 <= 2026-10-30T09:00:00+09:00`

시간 추정은 성공한 worker의 **전체 wall time**으로만 갱신한다.
`current_estimate = max(current_estimate, 1.35 × 해당 dataset의 관측 whole-worker wall)`
V 성능이나 선택 epoch에 따라 입장 우선순위·범위를 바꾸지 않는다. 첫 seed6개 완료 후 실제 추정과 남은 예산을 보고한다.

Hard-wall 경과 시간은 각 attempt의 실제 worker 최초 시작 시각부터 로딩·보존 해시·search·refit·V 평가를 모두 포함해 계산한다. 관리자 재시작·재부착으로 이 타이머를 초기화하지 않는다. 허용된 fresh attempt2는 별도의 실제 시작 시각을 기록한다.

worker에 캠페인 공통09:00 deadline을 search/refit 중단 시각으로 넘기지 않는다. 시작이 허용된 job은09:00을 지나도 정상 완료하도록 둔다. 별도의 고정 hard-wall watchdog만 적용한다. 이 watchdog 초과는 동일 설정 기술 실패로 기록하고, 입장 시간이 남으면 한 번 재시도한다. 종료는 정확한 본 job PID/creation/module/자식 확인 후 수행한다.2분 로그 정체로 강제 종료하거나 batch/epoch를 줄이지 않는다.

마감으로 시작하지 않은 job은 NOT_RUN_BUDGET다. 진행 중 job이 없으면 분석·검증·그림·ZIP·push를 마친다. 후처리는09:00 때문에 중간에 자르지 않는다.

시간은 학습 중심 약9–11시간 추정이며12–13시간 이상도 가능하다. 준비·참조90개 검산·최종 분석·공개는 별도다. '최대11시간'으로 약속하지 않는다.

## 7. 실제 실행과 smoke

사전 점검·CPUfixture·봉인 완료 후 고정60개를 순서대로 실행한다. 처음6개는 이미 등록된 seed92601의 실제 본실험이며 별도 pilot/중복학습이 아니다.

처음6개 완료 후:
- 데이터별 reference cohort와 정답/mask/origin/channel/mean/std 일치, FP64 지표 재계산, T_numeric_values_parsed=0
- 실제 trainable_parameters와 retained branch 초기 가중치 검증
- 실제 search/refit/전체wall, 고정recipe/F, 선택 봉인 후freshA60/V 순서
- 통과하면 중간 보고만 하고 다음seed로 진행한다. 통과한6개는 반복하지 않는다.

이후 각seed 묶음 완료 시 완료수/60, 경과시간, 데이터별 추정, 예상남은시간, WAIT/실패수 한 줄을 남긴다. 사용자에게 V 값으로 범위 변경을 제안하며 실행을 멈추지 않는다.

## 8. 보고·후처리는 학습 성공 여부와 독립적으로 끝내기

terminal 상태는 TRAINING_COMPLETE, TRAINING_INCOMPLETE, SCIENTIFIC_SUSPEND, USER_CANCELLED를 구분한다. 검증 가능한 완료0개여도 보고서는 만들 수 있어야 한다. 무결성 검증 PASS는 전체 실험 완료나 성능 우위를 뜻하지 않는다.

후처리 기본:
- 60개 job 전부를 상태표에 남긴다. 완료/실패/미실행을 누락하지 않는다.
- COMPLETE는 RESULT·attempt·source/data/선택 봉인·예측 SHA·실제 파일·FP64 검산까지 맞는 경우뿐이다.
- dataset별 canonical을 사용하고 dataset×method×seed로 유일성을 검증한다.
- 기본 결과를 primary 후보로 사용한다. 빈 연장 목록으로 기본 결과를 제외하지 않는다.
- 같은 dataset×method의 정확한10seed만 주 결과를 만든다. n<10은 별도 INCOMPLETE에 남긴다.
- A1/A2의 기존 effect 소수1자리 재현 및 모든 대상의 cap 선택/잠정성 전파를 검증한다.
- 새 선택 조건·임의 통계검정·유리한seed 제외·부분연장 혼합은 금지한다.

최종 산출물:
- `ALL_NEW_RUNS.csv`, `SUMMARY.csv`, `CONTRASTS.csv`, `PAIRED_DIFFERENCES.csv`, `JOB_STATUS.csv`
- `FIG_P_BRANCHES_APPL_ETT.png/.svg`: 데이터별 panel, OURS/두분기/NoP, Raw 점선, PROVISIONAL/INCOMPLETE 표시. 빈 family도 정직한빈panel로 처리한다.
- `SLIDE_NUMBERS.md`: 두줄×3열 signed1자리%, 판정, cap선택수, n. 미완료 칸을0%나정상값으로채우지않는다. Jena+0.2/+8.9는원값그대로 참고표시한다.
- `PREDICTION_CHECK.md`: 사전예측(Time-only세데이터UNCLEAR; Channel-onlyAppliances/ETTh1UNCLEAR, ETTh2낮은중앙값·상한가능)의적중여부. 예측을판정으로사용하지않는다.
- `RESULTS_KO.md`: 실제범위/시간/과학범위/부정·불명확·잠정·미완료/오류·회복·중단사유.
- `ANALYSIS_VALIDATION.json`, `INDEPENDENT_RESULT_CHECK.json`, `ROOT_RESULTS_GATE.json`
- 결과전용ZIP, manifest, 독립CRC/정확member집합/전체SHA·size 검증 영수증.

학습을 되돌리지 않고 끝낼 수 있는 CSV·cohort wiring·그림·JSON·ZIP 경로/구현 문제는 새 분석 버전으로 수리·시험 후 남은 보고를 계속한다. 잘못된 값·정의·T검증을 억지로 PASS로 만들지는 않는다. 완료된 학습을 다시 돌려 후처리 오류를 해결하지 않는다.

## 9. 공개와 push

원본 보존 hash를 다시 검사하고 그림을 실제로 시각 점검한다. 공개 점검을 `PUBLICATION_PRECHECK.json`으로 기록한다. ZIP에는 새source/publicreports/figures/manifest만 넣는다. `.cache`, 예측배열, checkpoint, 원데이터, credential, 비공개hostpolicy/trace, PPT/PDF/대본/초안은 제외한다. 코드·문서에 private파일의 검증용경로/해시를 기록하는 것과 실제privatebytes를공개하는것을구분한다.

전체완료뿐아니라 마감/과학SUSPEND/기술실패로부분완료일때도 검증된범위·상태표·실패근거·ZIP을 남긴다. 검증에실패한예측을정상결과로공개하지않는다.

최종변경은 다음두폴더의 공개할개별파일만 명시stage한다.
- `research/p_branch_appl_ett_20261029_v1/`
- `reports/p_branch_appl_ett_20261029_v1/`

`git add -A`, `git add .`, `--no-verify`, forcepush,reset,branch조작금지. ZIP이ignore되어있으면그ZIP정확한경로에만 `git add -f -- <zip>`을사용한다(forcepush와다름).

결과commit후 `git push origin HEAD:main`으로게시하고 `git ls-remote origin refs/heads/main`의SHA가로컬결과commitSHA와같은지확인한다. 일시네트워크실패는같은commit으로30초/120초후각1회재시도한다. 인증실패·non-fast-forward거부는원문을기록하고force/rebase/자동덮어쓰기를하지않는다. 로컬검증ZIP과commit을보존하고 publication_pending을명시한다.

최종전달: 실제완료수/60·실패/예산제외/중단여부, SLIDE_NUMBERS6칸, PROVISIONAL/INCOMPLETE, ZIP절대경로, push명령·결과·원격/로컬SHA일치를짧게알린다. 발표파일은이담당자가수정하지않는다.

## 10. 자동 진행의 경계

일시자원충돌·확인된기술실패·관리자기술종료·보고오류만으로전체캠페인을쉽게끝내지않는것이이설계의목적이다. 모델/데이터/선택/T검증실패를무시하거나60개모두성공한다고보장하는것이아니다.

사용자가명시적으로그만하라고하면관리자재시작/자동화를중지하고새job을시작하지않는다. 필요한경우본캠페인의실제worker만안전하게종료하고상태를보존한다. 나중의중단지시를이문서의'계속진행'으로무시하지않는다.
