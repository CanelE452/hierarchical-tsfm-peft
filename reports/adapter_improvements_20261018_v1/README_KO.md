# P/E/D 개선 실험 — 2026-10-18

실행 PARTIAL / 독립 검증 PASS / 연구 DESCRIPTIVE_DEVELOPMENT_EVIDENCE / 게시 SEPARATE_REMOTE_PROOF

[확인] 신규 과학 job 완료 15개, 검증 재사용 job 2개, 신규 선택 결과 arm 24개. 과거 contextual 선택 요약 59행과 과거 endpoint 88행은 별도 보존하며 신규 실행·재사용 job 수에 포함하지 않는다.

정확도 허용 열화율·목표 speedup·필수 운영 batch는 **UNREGISTERED**다. 임의 합격선을 만들지 않았다. V는 반복 사용된 개발 평가 구간이다. 과거 T 사용 이력은 유지하며 이번 캠페인에는 새 T 평가가 없다.

처음 볼 파일은 [분석](RESULT_ANALYSIS_KO.md), [정확한 수치](RUNS.csv), [대응 효과](COMPARISONS.csv), [origin별 대응 효과](PAIRED_ORIGIN_EFFECTS.csv), [비용](COST.csv), [발표 표](PPT_TABLES_KO.md)다. 그림 [F1](figures/F1.png)–[F5](figures/F5.png)는 각 source CSV에서 생성된다. 게시 상태는 로컬 build에서 추정하지 않으며 최종 remote 근거는 별도 REMOTE_PROOF.json을 따른다.

[확인] 등록 DCA 초기화 2개 scope가 NOT_CONVERGED다. 등록 optimizer 한도에서 종료했으며 downstream DCA 예측은 정상 완료로 기록하지 않는다. PCA 곡선만 있으면 DCA 미래 점수를 만들지 않는다.

선택 결과와 공통 endpoint 비교는 별도 행이다. 서로 다른 selected epoch/update는 같은 A_stop 선택 recipe의 결과 비교이며 같은 update 효과로 표현하지 않는다. 부정 결과·미실행·실패는 남긴다.

CHANNEL_HORIZON_STATS.csv는 row_id/horizon/channel_id/SSE/SAE/count의 6열이다. 각 통계의 method·cohort·seed·source·hash·상태는 RUNS.csv의 같은 row_id에 한 번만 저장한다. r00000 형식 ID는 해당 보고서 inventory 순서의 결정적 식별자이며 원 producer_row_id도 RUNS에 보존한다. 이 정규화로 모든 기존·신규 full precision 통계를 유지하면서 반복 metadata를 줄였다.

공개 통계만으로 표·그림을 재생성하는 실제 명령은 [재현 안내](REPRODUCE_KO.md)에 있다. 원 예측 배열과 checkpoint 검산은 비공개 자산을 요구한다. [남은 작업](RESUME_QUEUE.json)과 [원장](RESOURCE_LEDGER.json)을 함께 확인한다.
