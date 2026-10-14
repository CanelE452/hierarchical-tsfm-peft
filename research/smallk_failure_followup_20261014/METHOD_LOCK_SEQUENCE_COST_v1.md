# RF1/RF3 전체 비용의 순서 균형 측정 잠금 v1

상태: **SOURCE_ONLY_NOT_EXECUTED**. 승인 파일과 continuation manifest가 없으므로 실제 실행하지 않는다. 현재 작성한 테스트도 미실행이다. 승인 후 root가 상한·source/API 검토·동시 작업 종료를 확인한 다음 실행한다.

원 Robin VAL 및 이미 노출된 Beijing EVAL의 같은 첫1/8 raw physical past·mask·origins·저장 FIT FP32 mean/std를 쓴다. 새 미래/보호 TEST에 접근하지 않는다. C current anchor를 보존하므로 strict K-only 압축이 아니다. 기존 완료 profiler·보고서·자료를 수정하지 않는다.

```text
자료/batch    Robin·Beijing ×B1/B8
각7 routes   RAWshared, oldMAIN seed92601/92602,
              STATIC_RF1/TEMPORAL_RF3 각각 seed92601/92602
전체         4개 group ×7 routes =28개 행
warmup       모든 경로를2번 실행한 뒤 timed blocks 시작
timed        7 blocks, block마다7 routes를 각각1번
순서         고정 RNG seed92714 permutation의 cyclic Latin square
검사         각 route가1~7번 위치를 정확히1번씩 차지
비교         같은 block의 RAW absolute seconds와 route/RAW ratio 모두 보존
전체경로     원 D3 vector causal CPU fill/norm,H2D,E/F/D/head,
              C anchor,inverse,physical D2H,sync
timer 밖     normalized output D2H, reference/load/SHA/save, 안전 검사
normalized   atol1e-5 / rtol2e-6
physical     atol.002 / rtol2e-5 (동일 batch 실제 reference)
분해 측정     전체 균형 trials를 모두 끝낸 뒤 별도 stage 진단
stage        fill/norm/H2D/E/F/oldD/head/anchor_inverse/D2H 각각 sync
stage반복    경로별 warm2/repeat7·동일 batch reference/whole parity
local cap    GPU process180초 (달성 보장/자동 연장 없음)
```

이 cyclic schedule은 위치를 균형화한다. 모든 predecessor/carryover 쌍을 균형화하거나 thermal·clock 효과를 배제했다는 주장이 아니다. RAW가 첫 번째였던 이전 joint28 측정과 달라진 여부를 실제 수치로 확인하며, 원인 확정이나 보편적인 효율 개선으로 바꾸어 말하지 않는다.

기존 `DeploymentPipeline.whole`·vector fill의 수치 계산을 그대로 쓴다. 새 경로는 worker의 실제 `forward_trace(codec,bolt,head,history)`를 wrapper의 `forward_path`로 연결한다. reference는 같은 batch의 실제 `predict_trace`와 원 controls physical deployment다. 새 head의 독립적인 CPU 구현/replay가 아니므로 검사의 적용 범위를 표시한다. 원 bulk science 출력과의 비교 flags는 별도 보존한다.

실제8 COMPLETE600 producer의 E/D/Head 변경·F input gradient·F frozen·zero-head initial MAIN parity·actual-Y TUNE 선택과 audit producer SHA를 확인한다. 원 Codec 구조에서 worker의 `restore_selected_state`로 codec/head를 strict 복원한다. 모든7 경로는 함께 resident 상태이며 method-exclusive 메모리 측정이 아니다. 현재 모델/CP load와 reference 비용은 timer 밖이다. 과거 학습/teacher 비용을0으로 가정하지 않는다.

Stage는 전체 비용과 별도로 실행한다. oldD는 미래/마지막 latent 두 decode를 포함하며 head는 과거 prefix·예측 latent 및 반복 마지막 latent의 residual 응답을 포함한다. RAW/원 MAIN의 head stage는 동기화·host identity 비용으로 실제 head kernel이 아니다. stage별 sync overhead 때문에 그 합은 원 전체 latency가 아니다. stage outputs/parity/미완료 attempt를 저장한다.28 stage 행이 끝나지 않으면 partial 또는 NOT_MEASURED 이유를 명시하며, 전체28만 통과한 상태를 전체 계획 완료로 쓰지 않는다.

블록 경계에 full ContinuationLimit, 매 sample에는 같은 fast base cap을 확인한다. canonical human 승인·manifest는 ContinuationLimit이 검증한다. local/shared 예산 초과 시 새 tag의 부분 배열·block/position·parity·상태를 보존한다.28개 complete가 아니면 완료라고 하지 않는다. 실제 duration을 확인하기 전180초 내 완료를 보장하지 않는다.

```powershell
# 승인 뒤에만 실행. 8 producer와 bound PASS audit가 실제로 존재해야 한다.
.venv/Scripts/python.exe research/smallk_failure_followup_20261014/profile_sequence_balanced_cost_v1.py --execute --input-bridge ORIGINAL_BRIDGE --sequence-results RESULT1 RESULT2 RESULT3 RESULT4 RESULT5 RESULT6 RESULT7 RESULT8 --sequence-audit PASS_AUDIT --tag rf_balanced28_v1 --no-contention-attested
```

새 자료/모델 학습·APP14 자동 재개·commit/push·논문/PPT는 하지 않는다.
