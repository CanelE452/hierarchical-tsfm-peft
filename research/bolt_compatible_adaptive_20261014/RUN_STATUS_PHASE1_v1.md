# 승인된 Phase 1 실행 상태

상태: **RUNNING — 첫 공정 readout 비교 준비 및 검산 중**.

사용자 답변 `call_72d48d8ce73647df9d34c9a54d7f817e:0`의 “계획과 자원 상한 승인”으로 새 캠페인을 시작했다. Phase 0 원본 보고서·승인안·감사 기록은 그대로 보존한다. 새 시계는 2026-10-14 20:14:37 KST부터 2026-10-15 00:14:37 KST까지며 기존 종료 캠페인은 재개하지 않는다.

상한: GPU process90분·CPU process120분·wall4시간·추가6GiB·RSS8GiB·GPU reserved9GiB. [승인 기록](AUTHORIZATION_v1.json), [고정 실행 manifest](CAMPAIGN_MANIFEST_v1.json), [실제 자원 receipts](resources_v1/).

실제 완료: 새 runtime9테스트와 readout 수치/마스크/선택/과거 입력8테스트 PASS. 실제 자료 fit·추론·비용·독립 검산·최종 결과 보고는 아직 완료되지 않았다.

관리 기록 생성 중 PowerShell generic List 타입 오류가 발생했다. null manifest는 [실패 원본](CAMPAIGN_MANIFEST_BOOTSTRAP_FAILED_v1.json)에 보존하고 [오류 기록](BOOTSTRAP_FAILURE_v1.json)을 남겼다. 유효 manifest는 원 승인 시작 시각을 유지한다. 이 오류 때 과학 실행0이며 기존 자료·원래 결과를 바꾸지 않았다.

TEST 열람·기존 job 종료·새 설치/다운로드·대규모 backbone 학습·commit/push는 이 범위에 없다. 구현/검사 PASS는 성능 개선이나 독립 일반화 검증이 아니다.
