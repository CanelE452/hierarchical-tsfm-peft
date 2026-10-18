# B2 + RevIN 최소 목표 완료 대사

검사 단계: closing; snapshot 2026-10-18T03:51:25.392401+00:00. 등록한 과학 작업은 모두 terminal이며 상한에서의 개선 지속·일반화 한계와 게시 단계는 별도로 기록한다. 이 문서는 최종 원격 closing proof까지 확인했다는 선언이 아니다.

신규 개선 설정 4개, Jena 구성 대조 2개, 신규 search/refit arm-stage 10개, 기존 OFF 선택 결과 4개 재사용. 최종 cap은 {'jena': 3, 'pems32': 6}다. 신규 GPU 전체 경로 28개와 역사적 CPU Ridge 2개를 구분했다.

실행·저장 예측 FP64·source/bundle SHA·전체 비용 parity·공개 재생성·최종 테스트·보존 검사는 각 PASS proof와 실제 exit0에 결속되어 있다. 로그와 private artifact는 공개 raw 자료로 복사하지 않는다.

상한에서도 개선 중인 최종 arm은 RESUME_QUEUE.json의 bounded_scientific_questions에 그대로 남긴다. Jena B0는 같은 최종 B2 cap의 구성 대조이며 별도로 6epoch를 자동 실행하지 않았다. 반복 사용한 V를 독립 테스트로 바꾸지 않았고, 고정 K·두 seed·기존 T 노출·pretraining overlap UNKNOWN을 유지한다.

8개 목표별 실제 근거:

1. 신규 개선은 B2 + RevIN 하나; Jena·PeMS32 × 기존 두 seed의 ON 4개

   상태: COMPLETE_BOUNDED_SCOPE; 근거: CAMPAIGN_CONTRACT.json, RUNS.csv, EXPERIMENT_MATRIX.csv

2. 데이터·초기화·학습·선택·refit이 정확히 대응하는 같은 B2 OFF만 재사용

   상태: COMPLETE_BOUNDED_SCOPE; 근거: REUSE_AUDIT.json, PREDICTION_PROOF.json

3. 먼저 OFF/ON 3epoch; 개선 지속 데이터의 두 seed·양 arm 모두 6epoch; 상한은 수렴/실패로 단정하지 않음

   상태: COMPLETE_BOUNDED_SCOPE; 근거: DECISIONS.json, TRAINING_HISTORY.csv, COMPARISONS.csv, VALIDATION_REPORT.json

4. 최종 구성 기여는 같은 정규화·학습 조건의 Jena B0 두 seed로 확인

   상태: COMPLETE_BOUNDED_SCOPE; 근거: DECISIONS.json, T3_COMPOSITION_LIMITS.csv, EXPERIMENT_MATRIX.csv

5. Raw·기존 B2·RevIN·강한 기준선 예측 검산 연결; 최종 GPU B1/B8 비용; D 사용시에만 seed2 보완

   상태: COMPLETE_BOUNDED_SCOPE; 근거: PREDICTION_PROOF.json, COST_VALIDATION.json, T1_ACCURACY_COST.csv

6. 제외한 방법·AdaPTS 전체·추가 K/초기화 탐색을 실행하지 않음

   상태: COMPLETE_BOUNDED_SCOPE; 근거: CAMPAIGN_CONTRACT.json, EXPERIMENT_MATRIX.csv

7. 정확도–비용·RevIN 대응·구성 기여/한계의 세 묶음; 출처·상한·반복 개발 V 표시

   상태: COMPLETE_BOUNDED_SCOPE; 근거: REPORT_OUTPUTS.json, REPLAY_PROOF.json, T1_ACCURACY_COST.csv, T2_REVIN_MATCHED.csv, T3_COMPOSITION_LIMITS.csv

8. 코드·설정·검산·표·그림·한국어 분석의 정상 main 게시와 원격 확인·미완료 보고

   상태: REMOTE_PAYLOAD_VERIFIED_FINAL_CLOSING_PROOF_SEPARATE; 근거: REPLAY_PROOF.json, TEST_VALIDATION.json, PRESERVATION_CHECK.json, REMOTE_PROOF.json

원격 main 첫 payload의 실제 commit: `a23f3e66155deac9591eb66e9e73a93c542844d2`. 해당 proof에 포함된 파일 bytes를 다시 대사했다. closing metadata 게시와 최종 전체 파일 proof는 이후 정상 push 단계에서 확인해야 한다.

원본 dirty checkout은 Git status/HEAD 보존 범위이며 모든 dirty 파일의 content SHA 검증을 뜻하지 않는다. 별도 protected tracked 10062개는 실제 byte guard의 PASS를 따른다.
