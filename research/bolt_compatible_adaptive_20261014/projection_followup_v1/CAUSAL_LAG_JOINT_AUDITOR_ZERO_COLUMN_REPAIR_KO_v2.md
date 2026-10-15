# 공동 출력공간 감사의 정확한 0열 처리

[확인] 원 producer `actual_causal_lag_objective_v1/RESULT.json`은 COMPLETE24D/72예측/72진단, SHA `4168cd05962e479130a722919310f4c192c1bab03d9b78cb7983754f2e502c17`다. 첫 독립 감사 source `4d89a02743d992b326d752ea4062439007c3dab71a18ce7a63282e797b259742`는 원 상태로 보존한다. 첫 감사 `CAUSAL_LAG_OBJECTIVE_AUDIT_v1.json` SHA `7d4999aed252e343dba18c2f009bb3dc20498af5ab2b90c6ac01c0d8c6c8df7a`는 FAILED_PARTIAL_AUDIT_PRESERVED_NO_FULL_PASS다. 24D/336응답열의 적합은 검산했지만, 일부 공동 진단의 rank assert가 먼저 실패해 57/72 split까지만 검산됐다. 나머지15개 예측이 틀렸다는 결과가 아니다.

[확인] ROOT는 새 `diagnose_causal_lag_rank_boundary_v1.py`에서 원 파일·배열 SHA/shape/dtype를 확인하고, 저장 joint_A 및 complete_horizons 두 member만 역직렬화했다. target/forecast를 역직렬화하거나 모델·D 적합·예측을 재실행하지 않았다. LS RHS는 같은 mask-group 크기의 0행렬이다. [실제 진단](audits/CAUSAL_LAG_RANK_BOUNDARY_DIAGNOSTIC_ROOT_v1.json) SHA `9280c189a503b13ccef65c0a259c52958e280f2203458c829f07f9bd6c6b50c6`:18개 Beijing L7 조건/split·1458개 비어 있지 않은 mask group 중 원1e-15에서 rank 불일치18개였다.

그18개 모두 `lstsq`가 반환한 rank가 **정확한 nonzero-column 수 상한을1 초과**했다. 같은 행렬의 SVD rank는 모두 그 상한과 같았다. SVD 마지막 retained 상대 singular의 최솟값은 `.004324598943780834`, 첫 discarded 상대 singular의 최댓값은 `4.440892098500626e-16`이다. 예를 들어187×96의 masked A는4개 exact zero columns가 있어 rank≤92인데 LS는93, SVD는92였다. 별도 dimension-aware cutoff 민감도에서는 rank 불일치0이었다. 그 민감도 값을 원 결과의 cutoff로 대체하지 않는다.

[판단] 첫 감사의 두 LAPACK 계산기 rank가 반드시 같다는 전제는 이 0열 수치 조건에서 성립하지 않는다. 실제 경계에 의미 있는 정보 방향이 밀집했다는 설명보다, exact zero directions를 LS가 수치적으로 세는 오류라는 구체적 증거가 있다. 단, 독립 출력 projection·minimum·원예측 검산은 새 감사에서 실제로 다시 확인해야 한다.

새 auditor v2의 수정 범위는 다음과 같다.

1. **masked 출력 행렬의 각 열이 정확히0인지** 확인한다. 원 full A·HK 좌표·output mask를 보존한다.
2. 독립 LS 안에서만 그0열을 제외하고 같은 rcond1e-15로 푼 뒤, 전체 HK solution에 해당 계수0을 다시 넣는다. 정확0열은 어떤 latent 값도 관측 출력에 기여하지 않으므로 출력공간과 목적이 대수적으로 같다.
3. 원 full SVD의 singular spectrum·retained numeric space와 reduced LS의 예측/projection을 원 공차로 비교한다. zero-column 수·원 좌표/축·reduced rank/shape와 증명 범위를 별도 기록한다. 원 LS rank를 맞았다고 고치지 않는다.
4. 관측이 없는 horizon의 latent를 통째로 삭제하지 않는다. 그 latent가 지연을 통해 뒤 관측 출력에 영향을 주면 masked A 열은0이 아니므로 반드시 남는다.

이 수정은 원 producer·D 계수·실제 예측·FIT/TUNE/DEV·목적·cutoff·공차·기존 최소값을 변경하지 않는다. 새 학습이나 F 재실행이 아니다. 원 실패·자원·진단을 보존하고 새 감사에만 기록한다. 새 감사가 통과하기 전 full 과학 검산 완료라고 쓰지 않는다. 앞서 준비하던 primary-only 감사 대안은 아직 실행하지 않았으며, 이 수학적으로 동등한 최소 복구를 먼저 검증한다.
