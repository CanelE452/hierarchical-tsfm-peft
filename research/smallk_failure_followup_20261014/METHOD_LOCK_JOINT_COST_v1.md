# 선택된 joint E/D의 실제 전체 배포 비용 잠금 v1

상태: **SOURCE_PREPARATION_ONLY / ACTUAL_NOT_STARTED**. 실제 GPU 실행은 root GO 뒤, 12 joint fit·CPU audit·report 작업이 끝나고 다른 비용 계측과 GPU 학습이 없을 때만 한다. CPU 테스트는 합성 fixture다.

원 audited bridge의 Robin VAL30·Beijing 이미 노출된 EVAL146에서 같은 첫1/8 physical past 창과 mask를 사용한다. L512/H48·저장 FIT FP32 mean/std·현재 C anchor를 그대로 유지한다. strict K-only 압축이 아니다. 보호 TEST나 새 EVAL을 개봉하지 않는다.

12조건은 2자료×원 seed92601/92602×Y/RAW/MIXED다. 각 실제 COMPLETE producer에 600 updates, E/D 변화, 유한·비영 frozen-F input gradient, F 동결, fixed_basis 불변 증거를 요구한다. 같은 원 MAIN CP·batch order와 실제 Y TUNE 선택(checkpoint0 포함)을 확인한다. 새 selected CP의 seed/objective/update/architecture/file hash/state hash를 검사하고 원 Codec 구조에 전체 state를 복원한다. selected0이면 원 초기 MAIN을 재사용한 결과이며 새 학습의 성능 이득으로 읽지 않는다.

```text
전체 행 수       selected12 ×batch1/8 =24; 자료별 공유 RAW2 ×batch1/8 =4; 합28
전처리           기존 D3 DeploymentPipeline.whole + vectorized causal CPU fill 그대로 재사용
전체 경로        fill/FITnorm,H2D,E/F/D/anchor,inverse,물리 출력D2H,synchronization
warmup/repeat     2/7
각 반복 저장      normalized/physical 실제 출력7개 (normalized D2H는 timer 뒤)
reference         같은 batch의 원 engine.predict + 원 controls.deployment 실제 추론
normalized gate   atol1e-5 / rtol2e-6
physical gate     원 D3 atol.002 / rtol2e-5
bulk science      별도 비교 flags 유지; matching-batch 검사로 실패를 고쳐 쓰지 않음
분해 계측         없음; D3의 sync stage profile을 전체 latency로 재사용하지 않음
co-resident       F와 12 selected codecs 모두 resident; method-exclusive memory 아님
측정 제외         모델/CP/disk load,reference/cache/SHA 준비,normalized 출력 복사,save/full guards
guard            row 전후 full FollowupLimit, 매 trial/reference batch 같은 base caps+process300초
공유 상한         현재 캠페인 잔여 GPU/CPU/wall/storage; 원 old Limit 재시작 안 함
```

원 CP·source·bridge/RAW/selected output·새 selected CP를 전후 해시로 확인한다. matching-batch parity 실패는 이미 측정한 행과 출력을 FAILED_PRESERVED로 남긴다. 부분 coverage를28행 완료라고 쓰지 않는다. 각 행 warm2/repeat7와 RSS/allocated VRAM을 기록하고, 원 full-C teacher의 과거 precompute 비용은 unknown/기존 비용으로 표시한다. 이번 deployment receipt에 teacher 생성이나 학습 비용을 포함했다고 하지 않는다.

PASS audit의 12 condition producer SHA가 실제 입력한 RESULT12개와 정확히 일치해야 한다. F와 12 selected CP의 현재 load/verify wall, 첫 reference 이전 준비 wall, 행별 실제 reference 추론·저장·검산 wall을 timer 밖 비용으로 별도 기록한다. 이 wall에는 해당 구간의 안전 검사가 포함되며 순수 모델 로딩 비용으로 해석하지 않는다.

측정 중 별도 CPU 보고서·검산 작업을 실행하지 않는다. root가 실제 실행 중 작업을 확인하고 `--no-contention-attested`로 명시한다. warmup2는 엄밀한 cold/warm 분리를 보장하지 않으며 짧은 단일 장비/프로세스 측정을 보편적인 속도 개선이나 통계적 유의성으로 부르지 않는다.

```powershell
.venv/Scripts/python.exe research/smallk_failure_followup_20261014/profile_joint_cost_v1.py --describe
# root GO 뒤 실제 bridge, 최종 12 producer RESULT, PASS joint audit와 fresh tag를 지정
.venv/Scripts/python.exe research/smallk_failure_followup_20261014/profile_joint_cost_v1.py --execute --input-bridge BRIDGE_RESULT --joint-results JOINT_RESULT_1 JOINT_RESULT_2 --joint-audit JOINT_PASS_AUDIT --tag selected_joint28_v1 --no-contention-attested
```

예시의 joint-results는 반드시 실제12개 전체 경로로 바꿔야 한다. CLI는12개 고유 조합이 아니면 실행을 거절한다. 출력은 `.cache/smallk_failure_followup_20261014/joint_cost/<tag>` 새 폴더만 사용한다. 기존 코드·완료 보고서·실험은 수정하지 않고 commit/push·논문/PPT를 하지 않는다.
