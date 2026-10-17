# 전체 요청 충족 감사 — 로컬 검증 완료, main 게시 대기

과학 캠페인은 **PARTIAL**, 완료된 증거의 독립 검증은 **PASS**, 연구 판정은 **DESCRIPTIVE_DEVELOPMENT_EVIDENCE**, main 게시는 **PENDING**이다. Jena DCA 미수렴으로 필수 forecast 비교가 남았으므로 전체 실험 완료나 보편적인 성능 우위를 선언하지 않는다. 이 문서는 실제 로컬 증거 snapshot이며 첫 payload의 실제 원격 게시 후 상태를 갱신한다.

[확인] 감사 시각 2026-10-17T23:18:10.782833+00:00. 원문912행 SHA256은 55af78e3ade0d4cebcbc09f0f453b457afc7f748b2a2316486e4eadf8869e679다. 원문278개 명시 항목을 포함한657개 요구 ID/원문/section/행을 삭제·교체 없이 보존했다. [REQUIREMENT_COVERAGE.csv](REQUIREMENT_COVERAGE.csv)는 요구별 담당module/tests/job, 실제 증거·SHA, 상태와 확인 범위를 담는다. PROVEN은 해당 행의 한정된 증거 범위이며 연구 전체 성공을 뜻하지 않는다.

[확인] 원37job plan cab1bbc423e7ae7e5cf7029443a7be9a4e01246d76ec6af3a16d72b16fa7693f와 별도cost-only2job plan c6b0f84acf94e223f3e3dbe81d0cee3f5b47ead8663fffeb811e6a9d446c71fc을 유지했다. 봉인source 24개와 별도controller가 실제 바이트와 일치한다. 현재 report source dfa332fa872260d6277b0a79c5bc2e17606b86d796f91524c4315bf4cac1d575, verifier 546059a3a3d787799989539297c7b4889739d71fd91e06460a86e455fca781a7 및 전체 최종 test의 source/test SHA도 현재 파일과 일치한다. 과학 source/recipe와 보고/검증 source 수정을 구분했다.

## 실행 범위와 미완료

[확인] [EXECUTION_VALIDATION.json](EXECUTION_VALIDATION.json)의 두 실제 execute CLI가 exit0이다. 총39job은 {"COMPLETE": 30, "NOT_CONVERGED": 2, "NOT_RUN_DEPENDENCY": 5, "REUSED_VERIFIED": 2}로 모두 terminal이다. 신규 과학 실행15job, 신규 선택 결과24arm, 검증 PCA 재사용2job을 다른 단위로 센다. 초기화/profile/cost/audit를 신규 학습 개수에 더하지 않는다.

Jena D네head 및 P split/control은 seed92601/92602를 완료했다. PeMS32 P split/control과 fresh CUDA PCA/DCA는 두seed를 완료했고 D네head 전이는92601에서 완료했다. Jena DCA A_fit/A60은 등록 maxiter200에서 NOT_CONVERGED이므로 downstream을 정상 DCA실험으로 진행하지 않았다. PeMS 수렴/PI/정렬을 Jena forecast 완료로 대신하지 않는다. LIFT와 같은B2 output-gain은 필수 JenaE 의존성이 충족되지 않아 미입장이다.

- DCA_INIT_JENA_A_fit: NOT_CONVERGED — DCA fit did not converge under registered limits
- DCA_INIT_JENA_A60: NOT_CONVERGED — DCA fit did not converge under registered limits
- PROFILE_JENA_E_s92601: NOT_RUN_DEPENDENCY — Required scope/control preparation incomplete
- B2_DCA_JENA_s92601: NOT_RUN_DEPENDENCY — Required scope/control preparation incomplete
- PROFILE_JENA_E_s92602: NOT_RUN_DEPENDENCY — Required scope/control preparation incomplete
- B2_DCA_JENA_s92602: NOT_RUN_DEPENDENCY — Required scope/control preparation incomplete
- LIFT_JENA_s92601: NOT_RUN_DEPENDENCY — Required scope/control preparation incomplete

[RESUME_QUEUE.json](RESUME_QUEUE.json)의7개 항목은 terminal receipt 보존·원인 해결·새 승인 plan/identity 조건을 명시하며 같은 identity의 묵시 재시도나 solver/grid/budget 확대를 실행 가능한 재개로 표시하지 않는다. M5는4CSV+ZIP이 발견된 자산이다. 새 task/split/metric 계약 미등록으로 이번 수치 범위에서 제외되며 파일 없음 BLOCKED가 아니다.

## 결과와 최소 목표

[확인] 아래 변화율은 [COMPARISONS.csv](COMPARISONS.csv)의 MATCHED selected-refit/V_select 행에서 생성했다.100×(candidate−control)/control의 음수는 오차 개선이며 common endpoint와 각자의 A_stop 선택 뒤 fresh A60 결과를 구분한다.

- P 분해−같은 용량 대조: jena seed92601: ΔMSE +5.55164%, ΔMAE +5.91592%; jena seed92602: ΔMSE +6.11884%, ΔMAE +5.80281%; pems32 seed92601: ΔMSE +0.0238106%, ΔMAE +0.120639%; pems32 seed92602: ΔMSE +0.194016%, ΔMAE +0.150665%. 네 비교에서 분해 우위는 확인되지 않았다.
- E DCA−PCA: pems32 seed92601: ΔMSE +1.84432%, ΔMAE +1.11623%; pems32 seed92602: ΔMSE -0.0978383%, ΔMAE -0.000254763%. PeMS seed별 부호가 다르고 Jena forecast는 UNKNOWN이다.
- D는 Jena seed/지표별 방향이 다르며 PeMS92601에서 네head를 모두 비교했다. 선택 lambda/gamma와 recipe 차이, horizon tradeoff를 유지하고 자유도 하나를 유일한 원인으로 확정하지 않는다.
- 비용은 CPU BLAS2/GPU 및 B1/B8을 나눴다. GPU B1에서 Raw보다 느린 조건과 Ridge/LoRA/G의 적용범위·자연스러운 precision차이를 보존했다. batch8 ms/origin 개선을 모든 운영조건의 우위로 확대하지 않는다.

정확도 허용 열화율, 목표speedup, 필수 운영batch는 UNREGISTERED다. 임의 합격선이나 두seed의 보편적 유의성을 만들지 않았다. 최소 목표표는 실제 [PPT_T4_CLAIMS.csv](PPT_T4_CLAIMS.csv)에서 생성했다.

```text
Raw 정확도 : 부분/미판정
  COMPARISONS.csv; 임계값 UNREGISTERED
Ridge/LoRA 대비 가치 : 미판정
  RUNS.csv / COST.csv
device/batch 비용 : 부분
  COST.csv; UNKNOWN 유지
P 분해 기여 : 대응 실행 완료·우위 미입증
  COMPARISONS.csv; 선택-refit 실행과 지표/seed별 Δ를 구분; 우위 합격선 UNREGISTERED
E 초기화 기여 : 부분
  pems32 2 seed PCA/DCA 대응 완료; seed/지표 간 일관된 감소 없음; jena 2 seed DCA 초기화 NOT_CONVERGED; 예측 UNKNOWN
D 일반화 : 부분
  F2 / CHANNEL_HORIZON_STATS.csv
다중 K 공유 가치 : 부분
  MULTIK_AUDIT.csv; 누락 UNKNOWN
재현성 : 별도 증거 확인
  TEST_VALIDATION.json / VALIDATION_REPORT.json / PUBLIC_REPLAY_VERIFICATION.json / VISUAL_INSPECTION.json
```

## 검증과 재현

[확인] 최신 [TEST_VALIDATION.json](TEST_VALIDATION.json)은 357 passed in 49.76s, exit0, 현재 source불변과 기존 smoke 출력의 바이트 복구를 기록한다. 실제 Labour CUDA smoke를 포함한 suite wall은 동일 GPU journal에 보수적으로 누적했다. 이전 실행의 테스트 수를 합산하지 않는다. 초기 회귀 실패와 report-only 검증 실패/원인수정은 기존 history 및 다음 실제 lineage에 보존했다: [validation_attempts/AUDIT_FIX_LINEAGE.json](validation_attempts/AUDIT_FIX_LINEAGE.json), [validation_attempts/PUBLIC_TABLE_FIX_LINEAGE.json](validation_attempts/PUBLIC_TABLE_FIX_LINEAGE.json), [validation_attempts/PUBLIC_TABLE_INTEGER_FIX_LINEAGE.json](validation_attempts/PUBLIC_TABLE_INTEGER_FIX_LINEAGE.json), [validation_attempts/PPT_REPLAY_FIX_LINEAGE.json](validation_attempts/PPT_REPLAY_FIX_LINEAGE.json), [validation_attempts/PPT_SEED_LABEL_FIX_LINEAGE.json](validation_attempts/PPT_SEED_LABEL_FIX_LINEAGE.json). 과학 plan/receipt를 조용히 수정하거나 유리한 점수를 위한 재실행을 하지 않았다.

[확인] [VALIDATION_REPORT.json](VALIDATION_REPORT.json)은 FAIL0, 실제 저장 예측123archive와96valid whole-path route를 독립 검산했다. RUNS의299행 모두가 새 NPZ 검증이라는 뜻은 아니다. 과거 selected59/endpoint88행은 CONTEXTUAL이며 PCA재사용의 과거 endpoint8행은 metric-only UNVERIFIED로 남는다. source/selection/target/mask/origin/channel/precision 결속, FP64 SSE/SAE/count 및 밴드 합을 검산했다.63MATCHED 중 selected-refit 15개, paired-origin 1380행을 실제 배열 통계로 대사했다. origin macro 평균을 전체 channel-count macro로 대신하지 않고 stride/overlap을 공개했다.

[확인]96valid cost는 정확한 scored bundle/head/parent, fullV와 두reload parity 및10warmup/20timed 규약을 통과했다.4LIFT/gain NOT_MEASURED placeholder는 parity 실패가 아니고 새 순위에서 제외된다. Jena D/P 비용은 두seed, PeMS 비용은92601이며92602 비용은 UNKNOWN이다.46개 historicalcost(G4포함)는 해당 원receipt의 CONTEXTUAL 측정이며 새 hardware의 speedup에 섞지 않는다. F FP32/D FP64, stored/metric dtype, warm/cold·메모리·overlapping timer의 범위를 구분했다.

[확인] [HISTORICAL_REUSE_STATISTICS_CHECK.json](HISTORICAL_REUSE_STATISTICS_CHECK.json)은 과거142752개 통계를 FP64 bit-pattern SSE/SAE와 integer count의 동일 multiset으로 보존했고 과거metric147행을 legacy공차1e-8로 대사했다. 신규 동일배열 FP64 공차와 혼동하지 않는다. [MULTIK_AUDIT.csv](MULTIK_AUDIT.csv)의80행은 기존43source 감사이며 누락 노출/byte/switch time은 UNKNOWN이다. 새 다중K 학습이나 작은K 비용 측정은 아니다.

[확인] [PUBLIC_REPLAY_VERIFICATION.json](PUBLIC_REPLAY_VERIFICATION.json)의 실제 공개 CSV/JSON CLI replay는34개 exact byte PASS다: F1–F5 PNG/SVG/sourceCSV15개, public table9개, PPT table4개, 문서6개. [VISUAL_INSPECTION.json](VISUAL_INSPECTION.json)은 실제 읽은5개PNG와 마지막 표시 수정 후 동일15figure파일의 hash를 결속한다. F4는 Jena 미수렴을 표시하고 DCA곡선을 만들지 않았다. F6은 선택사항으로 생략하고 private곡선 재생성 PASS로 합산하지 않는다.

verifier의 외부3gate(시각 확인/공개 replay/원격 게시) UNVERIFIED는 수행하지 않은 외부 작업을 로컬에서 추론하지 않기 위한 계약이다. 앞2gate는 별도 실제 proof로 보완됐고 remote는 아직 PENDING이다. private 학습/배열 검산과 공개 통계 source-only 재생성의 경계는 [PUBLIC_PRIVATE_ARTIFACTS.md](PUBLIC_PRIVATE_ARTIFACTS.md)와 [REPRODUCE_KO.md](REPRODUCE_KO.md)에 기록했다.

## 보존, 예산, 게시

[확인] [ORIGINAL_PRESERVATION_CHECK.json](ORIGINAL_PRESERVATION_CHECK.json)은 원 checkout의 동일 HEAD와912개의 exact Git porcelain -z status 바이트 SHA 614d7291b6b5ca7e580cef0e5a2a74d6a1436693ce80fca419b19311d66ad7ae를 확인했다. 모든 원 dirty 사용자 파일의 byte별 검증 주장은 하지 않는다. 별도WORK의 SHA-inventory에 든 과거 공개1225파일은 parent:protected_bytes PASS다. 원 assets/environment를 읽고 편집·실행 출력은 격리 WORK/newcache에 두었다.

[확인] [RESOURCE_LEDGER.json](RESOURCE_LEDGER.json) snapshot에서 wall 10435.228초, 누적GPU charge 5370.250초, 새cache 720,025,422byte, 공식source 다운로드 305,232byte다. 남은 wall 11164.772초/GPU 9029.750초는 이 시점 기준이며 이후 감사/게시 wall은 최종 원장에서 갱신한다. 원6h/4h 및72m/48m 예약과 retained cache를 고려한 더 엄격한 상한을 유지하고 과거 누적 원장을 초기화하지 않았다. CPU-worker/parentCPU/GPU charge의 범위와 중복 가능성을 분리했다.

[확인] [ARTIFACT_MANIFEST.json](ARTIFACT_MANIFEST.json)의348개 현재 payload hash를 대사했다. 본 감사/요구표/VALIDATION/REMOTE 및 outer metadata를 자기hash순환 없이 별도 결속하는 정책을 유지한다. rootREADME 최신 결과 링크와 과거PeMS CPU상태/별도CUDA 완료를 구분했다.

**main 게시는 아직 PENDING이다.** 현재 REMOTE_PROOF.json/outer PUBLIC_ALLOWLIST·PUBLICATION_CHECK·PUBLIC_MANIFEST는 존재하지 않으며 존재하는 링크처럼 표시하지 않는다. root가 실제 allowlist/privacy/size/index gate를 통과하고 정상 commit/push한 후 remote ancestry 및 code/report/figure/repro 경로를 확인한다. force push나 보호규칙 우회는 허용되지 않는다. 첫 payload remote proof 이후 본 감사의 게시 상태를 갱신하며 closing metadata commit의 최종 원격 proof는 별도로 확인한다.

matrix 상태 집계: {"CONDITIONAL_NOT_ADMITTED": 12, "OPTIONAL_OMITTED": 3, "PARTIAL_NUMERICAL": 5, "PENDING_PUBLICATION": 14, "PROVEN": 275, "PROVEN_WITH_LIMITS": 337, "READY_FOR_FINAL_RESPONSE": 11}. READY_FOR_FINAL_RESPONSE는 아직 보내지 않은 최종 한국어 응답의 준비 checklist이며 응답을 이미 실행/검증했다는 뜻이 아니다.
