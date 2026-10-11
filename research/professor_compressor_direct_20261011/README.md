# 교수님 제안 GTConvAE·정상 크기 TS2Vec 직접 실험

GTConvAE의 native graph-time encoder/decoder와 공식 정상 크기 TS2Vec를 Robin/Jena × 두 seed에서 실제 학습했습니다. Canonical 표현 fit 16개, 실제 미래 평가 76개 경로, 비용 88개 조건, 전체 VAL 재구성 재추론 16개를 검산한 개발 결과입니다. 반복 VAL의 두 seed 결과이며 독립 TEST 확인과 학습 수렴을 뜻하지 않습니다.

Jena TS2Vec d320의 native ridge는 두 seed 모두 Raw F보다 MSE 약20%, MAE 약5% 낮았습니다. 같은 표현을 TSFM에 연결해 decoder를 calibration한 경로는 MSE만 좋아지고 MAE는 나빠졌습니다. 두 경로는 표현 export·label 창 노출·readout이 함께 다르므로 이 차이를 TSFM 하나의 인과 효과로 해석하지 않습니다. d320은 원채널21개보다 큰 표현 확장입니다. Robin에서는 실행한 후보가 Raw F를 개선하지 못했습니다.

![두 자료·두 seed의 실제 미래 성능](figures/METHOD_COMPARISON.png)

- [전체 결과·81개 그림·비용·한계](public/REPORT_KO.md), [실패 원인과 남는 판단](public/METHOD_FAILURE_DIAGNOSIS_KO.md)
- [GT 원형 대비 적용](public/GT_METHOD_DELTA.md), [TS2Vec 원형 대비 적용](public/TS2_METHOD_DELTA.md), [학습·추론·검산 소스와 재현 범위](public/SOURCE_README.md)
- [정확한 실행 계약](protocol.json), [실제 완료 목록](runs.json), [MSE/MAE 전체 CSV](metrics.csv), [실제 비용88개](COSTS.csv), [실제 학습·추론 stage202개](TRAINING_STAGES.csv)
- [완료 당시 최종 과학 감사](FINAL_EXECUTION_AUDIT.json), [strict CPU 검산](checkreceipts/direct_verification_1791724651692036200.json), [전수 재구성 검산](reconstruction_audit/summary_1791724673402087200.json)
- [완료 pilot의 별도 기록](EXTRA_ACTUAL_RUNS.json), [RAM 부분 실패8개](EXTRA_FAILED_ATTEMPTS.json), [GT guard 실패](FAILED_GT_robin_92601_False_1791722287225812200.json)
- [게시 범위·원본 보존·공개 clone의 한계](PUBLICATION_KO.md), [explicit allowlist](publication_files.json), [게시 검사](publication_checks.json)

이전 LEVEL A–H는 사용자 승인으로 중지된 **미완료** 범위입니다. 새 direct 실험 완료와 구 campaign 완료를 구분합니다. [중지 기록](OLD_FIT_CANCELLED.json), [과거 protected TEST 접근 사고의 범위](public/PROTECTED_ACCESS_CONTEXT.json)를 보존합니다. 새 Robin/Jena scope의 TEST 접근0이 과거 사건을 되돌리지는 않습니다.

공개 문서는 `public/`의 게시용 사본입니다. 실험 완료 당시 원본 source/result/그림/최종 감사는 byte 그대로 게시하고, 원본 한국어 문서의 SHA와 사본 SHA는 [publication_before](publication_before.json)에 기록했습니다. 원자료·weights·예측 배열·로그는 Git에 없으며, 아래 공개 검사는 이 파일들을 읽지 않습니다.

```text
python research/professor_compressor_direct_20261011/check_publication.py
```

새로 요청한 Jena d320의 동일 export·동일 label 조건 비교는 후속 실험이며 이 결과에 포함하지 않습니다.
