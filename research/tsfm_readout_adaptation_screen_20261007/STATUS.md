# 상태 — TSFM 출력·native head 대조

STATUS: **READY_FOR_BUDGET_APPROVAL**

- authorized_stage: S0 읽기 감사·실행 계약·작은 관련 산출물 게시.
- executed_stage: S0 읽기 감사·정적 데이터/함수/헤드/캐시 크기 확인·선행 지도·계획 작성.
- new_scientific_fits / technical_probe_updates / new_model_forwards / new_TEST_instances: **0 / 0 / 0 / 0**.
- S0 calibration / S1 / S2 / S3 / S4 execution authorization: **false**.
- stage1_complete: **false**. 목표는 active이며 승인 뒤 계속한다.
- local_sha / live_remote_sha at audit: `8646e54630add0777c90686b5a8e4bef20c7dcea`; 계획 게시 SHA는 후속 실제 게시 확인과 분리한다.

[확인] exact current OUTPUT/HEAD 완료 결과는 scoped search에서 발견하지 못했다. 기존 정확도·선택·예측 원장은 재사용한다. v18 resource stop은 성능 실패가 아니다.

[확인] native head2,526,336 scalars, bounded TRAIN-union+VAL 특징/point payload1,349,932,992bytes. 실제 새 HEAD+Adam/cache/online timing과 peak는 아직 미측정이다.

다음 승인 대상은 scientific fits0/TEST0/full cache0의 S0 calibration: GPU job600초, CPU-only job600초와 CPU process time600초, total wall1800초, disk256MiB, RAM/VRAM 각8GiB, disposable optimizer 최대384updates다. [계획](PLAN.md)과 [수치 원장](budget_estimate.json)을 먼저 검토한다. 그 실측으로 S1 전체 시간 cap을 수치화해 별도 승인받으며, 승인 후에는 run별 반복 승인을 받지 않는다.

S2_recommendation: **NOT_JUSTIFIED_YET**.

현재 remaining decision gaps: (1) 새 native HEAD+Adam/cache 실측과 S1 총예산, (2) S1 출력·헤드 적응 개선, (3) S2 이후 실제 내부 추가 이득/선택 문제 성립. 실행하지 않은 효과·기술 PASS·비용을 채우지 않는다.
