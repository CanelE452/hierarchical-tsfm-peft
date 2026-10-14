# 자기 채널·다변량 정보와 고정 Bolt: 완료 진단

[한국어 상세 보고서·그림 8개](reports/adaptive_REPORT_KO.md) · [6장 설명 스토리 MD](reports/PRESENTATION_BRIEF_KO.md) · [완료·미실행 항목](reports/CAMPAIGN_CLOSURE_KO.md) · [판단 기록](reports/DECISION_LEDGER_FINAL_KO.md)

같은 과거 시간 요약·FIT/TUNE·학습 손실·정규화·선택 후보를 맞춰 자기 채널 SELF33과 모든 채널 JOINT33을 비교했다. Robin·Beijing의 이 선형 readout 계약에서 TUNE의 실용 이득을 확인하지 못했다. Robin에서는 동일 기간의 FIT512 진단도 실제 실행했지만 gate를 통과하지 못해 사전 계획에 따라 후속 표현 학습을 종료했다. 작은 K 연구 전체의 해결이나 공통 예측 정보의 부재를 증명한 결과는 아니다.

```text
TUNE 정규화 MSE / MAE        SELF33                  JOINT33
Robin FIT256                .139778 / .219167       .262253 / .294391
Beijing FIT256              .843454 / .554774       .953966 / .591202
Robin FIT512 수선 결과       .140339 / .218615       .260887 / .292043
```

SELF33·JOINT33은 같은 32개 16시점 평균과 현재값을 사용하고 모든 경로에 한 개의 macro TUNE 선택 규칙을 적용했다. 입력 차원·계수 용량과 규제 효과까지 같다는 뜻은 아니다. SELF512는 자기 채널 전체 과거를 사용하는 강한 단순 대안이며 채널 정보의 순수 대조와 구분한다. L512/H48, FIT256/TUNE64, Robin 개발 VAL30·Beijing 이미 노출된 EVAL146이다. 결정적 ridge에 중복 seed를 붙이지 않았다.

![개발 구간의 실제 MSE와 MAE](../reports/first_and_dense_v3/figures/01_selected_development_scores.png)

Beijing 개발 EVAL은 JOINT33 MSE .745597로 SELF33 .781774보다 좋지만 MAE는 .521992로 .484819보다 나쁘다. 이 반례를 삭제하지 않았다. Raw Bolt의 개발 MSE/MAE는 Robin .340402/.313192, Beijing .589575/.407917이다. 이 수치와 이전 표현 실험만으로 Bolt의 일반적 필요성·불필요성을 확정하지 않는다.

유효 신규 모델 묶음 42개와 검증 후 재사용 14개, 유효 pool 비교 72개를 기록했다. 수치 재현에 실패한 첫 FIT512 시도 14개까지 포함하면 실제 신규 실행은 56묶음·총 pool 비교 90개다. 수정 전 기록도 보존했다. FIT512는 최초 FIT256을 그대로 포함하는 추가 표본이 아니며 중첩 창을 독립 표본으로 세지 않는다.

![같은 실행의 배포 전체 경로 비용](../reports/first_and_dense_v3/figures/06_whole_deployment_cost.png)

첫 비용 실험은 실제 원단위 과거 입력→전처리→예측→원단위 CPU 반환까지 B1/B8 24조건·144반복을 측정했다. 수선된 FIT512의 CPU 4조건·24반복은 별도 실행이며 그때 Raw Bolt를 재측정하지 않았다. 두 실행의 절대 지연을 합쳐 속도 우위를 주장하지 않는다. 모델 로드·디스크 읽기·해시·감사는 타이머 밖이고 공유 상주 모델의 메모리 범위도 보고했다.

[점수 72행](../reports/first_and_dense_v3/tables/COMPARISON.csv) · [선택 pool 54행](../reports/first_and_dense_v3/tables/POOL_SELECTION.csv) · [FIT512 수선 점수](../reports/first_and_dense_v3/tables/EXPANDED_COMPARISON.csv) · [24조건 비용](../reports/first_and_dense_v3/tables/COST.csv) · [추가 CPU 비용](../reports/first_and_dense_v3/tables/EXPANDED_COST.csv) · [실패 포함 실행 시도](../reports/first_and_dense_v3/tables/EXECUTION_ATTEMPTS.csv)

[실제 배열의 첫 독립 감사](../audits/FIRST_READOUT_AUDIT_v2.json) · [FIT512 수선 감사](../audits/EXPANDED_READOUT_AUDIT_v2.json) · [비용 감사](../audits/COST_AUDIT_v2.json) · [추가 CPU 비용 감사](../audits/EXPANDED_COST_AUDIT_v1.json) · [보고서·CSV 4,288항목 검산](../audits/FINAL_REPORT_ARTIFACT_AUDIT_v2.json)

입력·통계·origin 계약 검사와 고정 모델의 합성 미래 교란 fixture를 수행했다. 이것을 모든 미래 원자료를 교란한 실제 전체 파이프라인 재실행으로 확대하지 않는다. 검산 PASS는 성능 개선이나 독립 holdout 검증이 아니다. 보호 TEST 접근과 새 신경망 학습은 0이다. ForeCA/예측 목적 E-D·새 native·잔차 결합·긴 L·다른 자료·대조학습/동적 그래프는 이번 gate 이후 실행하지 않았다. 해당 방법들의 성능 실패로 표시하지 않는다.

[코드와 재현 조건](CODE_AND_REPRODUCTION_KO.md) · [원 지시문](../INPUT_PLAN_20261014.md) · [승인된 계획 원문](../PHASE1_PLAN_KO.md) · [Phase0 원천 감사](../PHASE0_AUDIT_KO.md) · [공식 선행연구 조사](../LITERATURE_REVIEW_KO.md) · [인간 승인](../AUTHORIZATION_v1.json) · [고정 실행 시계](../CAMPAIGN_MANIFEST_v1.json)

원 보고서·source snapshot·producer JSON·감사·자원 기록은 변경하지 않았다. 원문에서 ‘승인 대기’, ‘push하지 않았다’, ‘자동 push 금지’라고 적은 부분은 각 실험 단계 당시 기록이다. **이번 게시 묶음은 이후 사용자의 명시적 main push 요청에 따른 것이다.** 새 실험 승인이나 기존 시계의 연장을 뜻하지 않는다.

원데이터·NPZ·모델 가중치·checkpoint·캐시 이진 파일은 게시하지 않는다. JSON의 Windows 절대 경로는 로컬 실행 계보다. Markdown의 이동 링크와 PNG/SVG·CSV는 GitHub용 상대 경로로 검사했다. 로컬 전체 git status는 당시 관찰 기록이며 해당 파일들의 게시 여부를 뜻하지 않는다.

[게시 파일 SHA 목록](FILES.json) · [상대 링크 검사](LINK_AUDIT.json) · [원 보고서 SHA와 링크 사본](ORIGINAL_REPORTS.json) · [게시 전 검증 기록](PUBLICATION_CHECKS.json) · [이전 MAIN 보정 결과](../../smallk_failure_followup_20261014/publication_main_20261014_v3/README_KO.md)
