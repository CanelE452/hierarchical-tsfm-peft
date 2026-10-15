# Cost / C scorer 소스 리뷰

작성: 2026-10-15. 검토 범위는 `cost_and_score.py` 전체와 연결된 비용·deployment·선택 잠금 생성 소스다. 변경한 파일은 이 문서뿐이다. 실제 자료·F·GPU를 로드하지 않았고 비용/C 점수를 실행하지 않았다. `[확인]`은 실행 흐름·차원·시간 범위를 소스에서 끝까지 추적했다는 뜻이다. 실제 비용/예측 receipt의 검산 PASS를 뜻하지 않는다.

최종 읽기 시점의 소스 SHA256:

```text
cost_and_score.py cb1329dded8e2d3d41fb5630fbae9e06c4604b5386cc338e4f636ded9aa24ee0
cost_measure.py   b5b9f96fd228b86da2ce0fd6467c4fae28986c3e5da96efa6a66b8017afaea91
deployment.py     6a187d8768519e868ff6574668eba0382e9462f326b9e57d9b6a4320440f43f2
followup.py       77d5ae26daccc2d361ac01b1cbe4af6c3e64e84f3e6c7120756b6bd90ef5c5f0
```

루트/비용 담당이 아래 지적을 반영한 뒤 소스에서 다시 확인했다. 실행 전 최종 source snapshot은 이 hash 이후 변경이 있는지도 함께 기록해야 한다.

## 전달했고 반영을 확인한 문제

1. **[P1, 확인·반영됨] C scorer 예외에서 LoRA hook 잔류.** 최초 소스는 정상 try 끝에서만 hook을 닫고 예외 후 다음 spec을 계속했다. 예측/JSON/NPZ 저장 실패 시 공유 F에 LoRA가 남아 다음 C 경로를 오염시킬 수 있었다. 현재 각 spec 앞에서 `model=None`으로 초기화하고 `finally`에서 내부 LoRA를 닫고 model을 해제한다. 생성/checkpoint load 예외는 deployment의 별도 cleanup도 적용된다. 실제 hook count/Raw parity 검증은 아직 이 리뷰에서 실행하지 않았다.

2. **[P1, 확인·반영됨] 비용과 선택의 hash 결속 누락.** 최초 exclusive_all은 이전 COST_SELECTION으로 측정한 뒤 현재 SELECTION_LOCK hash를 COST_LOCK에 넣을 수 있었다. 현재 비용 시작 시 selection hash를 고정하고 COST_SELECTION에 기록한다. GENERAL_COST_COMPLETE는 selection/COST_SELECTION hash를 저장한다. exclusive_all은 두 hash의 현재 일치를 검사하고 그 값을 COST_LOCK으로 전달한다. score도 selection/COST_SELECTION hash의 일치와 COST_LOCK 성공 상태를 먼저 검사한다. 따라서 단순 selection 교체로 옛 비용을 새 모델에 연결하던 경로는 제거됐다. 아래의 general receipt 최종 재검사는 별도 잔여 항목이다.

3. **[P2, 확인·반영됨] 동결 source/모델 검증.** `rows()`는 잠금의 과학 소스 hash, selected checkpoint/native weights hash, fold-statistics 파일 hash를 검사한다. followup의 bind는 nested base까지 재귀적으로 checkpoint/weights를 모으므로 G/SWAP의 base도 결속된다. 리뷰 문서를 수정하는 것은 동결 scientific tuple에 포함되지 않는다. 코드나 selected artifact가 바뀌면 C 이전에 거부하는 흐름이다.

4. **[P2, 확인·반영됨] exclusive의 경쟁 작업 검사.** `assert_no_training`을 general 및 exclusive_all/각 exclusive child 진입에서 사용한다. 프로젝트 campaign core/followup과 다른 exclusive child를 감지한다. 이는 알려진 command pattern 검사이며 모든 GPU 작업의 OS 수준 배타성 증명은 아니다. 실제 장치·다른 기존 job·측정 시간대의 경쟁 상태는 별도 자원 receipt로 보고해야 한다.

5. **[P2, 확인·반영됨] NoBolt의 F instance metadata.** storage가 비어 있는 exclusive NoBolt의 F instance 수는 현재 0으로 기록한다. resident F bytes도 실제 empty storage에서 0이다. 이전 상수 1 metadata를 native 경로의 실제 F 로드 증거로 해석하던 혼동은 제거됐다.

6. **[P2, 확인·반영됨] SPRINT matched Raw의 채움 순서.** SPRINT는 L512를 먼저 인과 채움한 뒤 effective length로 crop한다. 이전 Raw 대조는 effective-length window를 처음부터 다시 채워 crop 앞 결측의 carry 정보가 달랐다. 현재 `CropRaw`는 같은 filled L512를 받아 마지막 effective 행만 F에 넣는다. 원 history의 동일한 관측/초기 채움 정보에서 비교한다. 남는 길이/metadata 표기는 아래에서 구분한다.

7. **[P2, 확인·반영됨] SCORES의 실제 P/E/D/G/F 식별과 F 변경 상태.** `method_metadata`가 G의 nested base를 재귀적으로 풀고 SWAP은 native head로 F를 표시한다. R1은 variant를 P로 기록한다. SCORE_COMPLETE는 실패가 없고 실제 F hash도 같을 때만 COMPLETE_VALID다. 아래 matched Raw 파생 행은 이 새 metadata를 아직 다시 덮어쓰지 않는다.

## 최종 읽기 시점에 남는 항목

1. **[P2] general receipt의 기록된 hash도 소비해야 한다.** COST_LOCK에는 `general_cost = sha(GENERAL_COST_COMPLETE.json)`가 있지만 score 진입에서 이 필드와 현재 GENERAL_COST_COMPLETE의 SHA를 대조하지 않는다. selection/COST_SELECTION 체인은 이미 반영됐다. `general_cost`도 검사하면 최종 cost receipt 체인이 닫힌다. 이 항목은 루트에 전달했으며 위 hash 시점에는 미반영이다.

2. **[P2] matched Raw 파생 행의 P/decoder와 두 길이를 구분해야 한다.** `ref=dict(record, method='RAW', ...)`는 A11의 P=A11 및 R1 restore decoder를 상속한다. 참조 행에도 Raw metadata를 덮어써야 한다. A11 CropRaw는 물리 전처리 history L512와 F context 504/432 등이 다르므로 `length=effective` 하나만으로 둘을 표시하면 정보 범위가 잘못 읽힌다. 수치/target cohort 연결은 올바르다. 루트에 두 표기 문제를 전달했다.

3. **[P2, 조건부 실행 차단] CP0 parity lookup은 ADAPTER type을 한정하는 편이 안전하다.** 현재 kind/r만으로 첫 route를 찾는다. 정상 완전한 core에서는 core ADAPTER가 DIRECT보다 먼저 있어 기대한 경로를 고른다. r=C core가 누락되면 같은 kind의 DIRECT를 찾아 cp0가 Raw와 같아야 한다고 잘못 검사할 수 있다. `type=='ADAPTER'`를 명시하고 빠진 필수 CP0는 별도 미완료로 표시하면 실패 원인이 정확해진다. 실제 core 누락 여부는 이 리뷰에서 조사하지 않았다.

4. **오류 복구는 기존 산출물 덮어쓰기 없이 해야 한다.** FIRST_C_SCORE, 예측 폴더, SCORE JSON/CSV 및 cost output은 새 생성만 허용한다. 부분 실패 후 동일 CLI를 처음부터 재실행하면 기존 경로에서 정지한다. 동일 고정 모델의 저장·계산 오류를 복구할 때에는 이미 완료한 bytes/선정/receipt를 보존하고 별도 시도 ID/manifest를 사용해야 한다. C를 본 뒤 모델·CP·alpha를 다시 고르는 복구는 허용되지 않는다.

## 요청한 핵심 흐름 확인

**C selection/cost guard:** [확인] `score()`는 COST_LOCK COMPLETE_VALID, 현재 selection/COST_SELECTION hash, `rows()`의 source/artifact/statistics 검사를 통과한 뒤 FIRST_C_SCORE를 기록한다. 이 전에 cost/scoring용 F나 새 window를 만들지 않는다. followup은 모든 학습형/alpha/분기 선택을 끝낸 후 선택 잠금을 쓰며 C 점수를 만들지 않는다. GENERAL_COST_COMPLETE 성공 이후에만 exclusive_all이 진행하고, 모든 essential exclusive child의 exit0/COMPLETE_VALID가 있어야 COST_LOCK이 성공한다. 실패한 비용이 있으면 정상 흐름의 C gate는 닫힌다. 비용 대상으로 고른 경로와 전체 selection route는 서로 다른 집합이므로 COST_LOCK을 모든 ablation/seed/긴 R1 경로 각각의 비용 측정 증거라고 쓰면 안 된다.

**LoRA 정상/실패 정리:** [확인] general/exclusive 비용의 factory route는 비용 모듈 finally에서 `Route.close()`한다. C scorer도 성공/실패에 상관없이 LoRA를 닫는다. 다른 경로를 예측하는 동안 LoRA가 함께 붙어 있어야 할 이유는 없으며, F parameter hash만으로 hook의 부재를 증명할 수 없다. 실제 최초 LoRA 전후 Raw parity 또는 hook 수의 실행 검증은 별도 필요하다.

**general/exclusive NoBolt:** [확인] general은 F 한 개를 resident로 유지한 채 모든 후보를 순차 측정한다. 그 안의 native/G-only는 F를 호출하지 않아도 F weights가 resident다. exclusive는 `uses_bolt`가 G base까지 재귀적으로 판정하고 CURRENT/G-only/DIRECT/RIDGE/SWAP에 F를 로드하지 않는다. NoBolt는 0-element CUDA marker로 장치만 제공하며 호출되면 예외를 내므로 실수로 F를 쓰는 경로는 성공으로 측정되지 않는다. SWAP base의 P/E/D를 구성하는 것은 F forward가 아니다. 두 scope의 resident bytes를 같은 조건의 숫자로 섞으면 안 된다.

**공통 cohort와 긴 Raw:** [확인] A12/A13은 해당 길이의 합법 origins만 고르고 같은 origins의 원 Y/mask를 유지한다. 참조 Raw512도 바로 그 origins에서 새로 예측하므로 다른 cohort의 Raw 점수를 가져오지 않는다. A11 effective Raw도 같은 origins/Y/mask를 사용하며 수정된 채움/crop 순서가 같다. 원격자 H48에 대한 모든 예측은 같은 원 Y로 점수화한다. 별도 matched prediction은 해당 route의 NPZ에 저장된다.

**반복 수:** [확인] q 후보에 대해 실제 측정 round는 `ceil(10/q)*q`다. warmup5 이후 각 후보를 순환 위치에 한 번씩 놓는 완전 순환으로 올리므로 general의 측정 수를 항상 10이라고 쓰면 안 된다. q>10이면 후보당 q회, exclusive q=1이면 10회다. summary의 `valid_trials/measured_trials`는 batch1/batch8 각각의 측정 수이며 warmup은 포함하지 않는다. metadata의 `measured_rounds`와 raw CSV가 실제 receipt다. 이 리뷰는 후보 수나 실제 sample 수를 실행 결과로 확정하지 않았다.

**weights/buffers와 memory:** [확인] F resident는 고유 parameter/buffer storage이고 Route active는 F를 소유하지 않는 활성 adapter/head/G/LoRA의 고유 parameter/buffer storage다. tied storage를 두 번 세지 않는다. 이 값들은 training 가능 parameter 수와 다르다. peak allocated/reserved, peak−F, peak−pretrial allocated는 서로 다른 회계량이다. 마지막 두 차이를 순수 activation 메모리라고 부르면 안 된다. window/data cache 등 다른 CUDA allocation도 포함된다. Windows peak RSS는 process lifetime 값이다. 실제 bytes는 fresh process별 receipt에서 확인해야 한다.

**whole/stage 시간:** [확인] CPU 물리 history의 채움·표준화·GPU 전송·실제 경로 예측·CPU 반환·원 단위 복원을 whole timer가 포함한다. 모델/데이터 생성과 checkpoint load는 밖이다. F crop H48이 native head의 전체 연산을 48만큼 줄였다는 의미는 아니다. native/direct fallback의 head/D는 PE 묶음 timer에 포함되므로 Adapter와 PE/DG를 같은 세분화라고 해석하지 않는다. 자세한 연결은 REVIEW_FOLLOWUP.md에 기록했다.

현재 판정: 전달한 두 P1 문제의 소스 수정 반영을 확인했다. 실제 C 누출·비용 성공·예측 parity·측정값을 실행 검증한 판정은 아니다. 위 잔여 P2와 실제 receipt 검산을 분리해서 마무리해야 한다. 실제 C는 아직 이 리뷰에서 열거나 점수화하지 않았다.
