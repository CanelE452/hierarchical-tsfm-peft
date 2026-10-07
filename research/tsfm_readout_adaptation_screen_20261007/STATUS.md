# 상태 — TSFM 출력·native head 대조

STATUS: **STAGE1_DONE / VERIFICATION_PASS / MIXED_READOUT_RESULT**

- authorized_stage: S0/S1, 작은 관련 산출물 commit/push. 직접 사용자 `해줘`와 `이런걸로 멈추지 마`; 반복 run 승인 없이 고정 S1 범위를 실행했다.
- executed_stage: S0 native 검증 PASS, S1 최대24 fits 완료, 전체 선택 봉인 뒤12개 TEST 모델, 같은 회차 폐기용 학습·전체 배포 비용, CPU 원장 검산 PASS.
- S0 native technical_probe_updates: **384**, scientific fits / new_TEST_instances: **0 / 0**. 별도 CPU toy fixture40 updates는 실제 자료·모델 적합이 아니다.
- S1 scientific fits / scientific updates / new_TEST_models: **24 / 66,432 / 12**. 비용 technical updates **2,160**; 학습108·배포162 blocks·sentinel18쌍.
- S0/S1 execution authorization: **true**. S2/S3/S4 실행 금지. 최신 직접 계속 지시와 수치 상한은 authorization.json·s1_budget.json에 기록했다.
- stage1_complete: **true**. S2/S3/S4 미실행; 내부 LoRA 추가 가치와 진단 연구 성립은 아직 판정하지 않았다.
- local_sha / live_remote_sha at audit: `8646e54630add0777c90686b5a8e4bef20c7dcea`; 계획 게시 SHA는 후속 실제 게시 확인과 분리한다.

[확인] exact current OUTPUT/HEAD 완료 결과는 scoped search에서 발견하지 못했다. 기존 정확도·선택·예측 원장은 재사용한다. v18 resource stop은 성능 실패가 아니다.

[확인] native head2,526,336 scalars, bounded TRAIN-union+VAL 특징/point payload1,349,932,992bytes. 실제 S0 head+Adam/cache/online timing은 calibration_result.json·guard_repair.json에 기록했다. 구현 CPU fixture40 updates는 실제 자료를 적합하지 않았으며 별도 receipt에 남긴다.

S1 전체 계산 GPU job wall **2,322.440초(38.71분)**, CPU 보고 job 최종1.969초. 24 fits elapsed565.452초, 특징 준비256.363초, 선택0.352초, TEST 예측34.493초, 비용 probe1,440.327초, 나머지 I/O·채점·계측25.453초를 중복 없이 분리했다. 이는 전용 GPU kernel 시간이 아니라 CPU/I/O를 포함한 계산 job wall이다. 원장 측정 peak RAM2,843,209,728bytes, GPU allocated688,788,992 / reserved754,974,720bytes이며 사전 상한 이내였다.

```text
Combined TEST MSE  F0        OUTPUT    NATIVE_HEAD
Robin              .229095   .275511   .229719
Peacock            .201521   .199447   .201351
Jena               .292300   .265080   .243218
```

[확인] Jena MSE는 OUTPUT9.31%, HEAD16.79% 감소했다. Robin OUTPUT은20.26% 악화했다. Robin HEAD와 Peacock의 작은 차이는 조건부 block 구간에0을 포함한다. OUTPUT MAE는 세 자료 모두 악화했다. epoch0 선택7개·OOM0·상한 종료0은 기술 실패와 구분한다.

S2_recommendation: **PROPOSE**, TEST를 본 뒤 구체화한 탐색적 개발 제안. 봉인된 VAL·정상성·이번 회차 비용을 주 근거로 하고 historical LoRA 격차는 보조 참조로 둔다. 같은 HEAD 부모에서 두 경로·3자료·2seed의12 fits 제안은 별도 승인 대상이다. joint 실제 비용·메모리부터 확인할264 technical updates / GPU wall900초 / RAM·VRAM8GiB probe 후보와 본실험 추정의 한계를 [보고서](report.md)에 기록했다. 자동 실행하지 않았다.

[확인] 보호 원본76개 현재 hash 변경0, 시작 전 미추적1,889개 상태 목록 hash 동일. 기존 E/D·Conv, 원본 PPT/PDF, unrelated 파일을 변경하지 않았다. v18 자원중단을 과학적 실패로 바꾸지 않는다.

[검산](verification.json) · [정확도·기간·seed](evaluation.json) · [비용](costrows.json) · [실행 원장](execution_ledger.md). 최종 게시의 실제 local/live remote SHA는 계산 결과와 분리해 후속 게시 확인에서 기록한다.
