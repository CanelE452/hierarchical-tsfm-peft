# MAIN 잔차 보정과 오차 분해: 완료 결과

[실험 보고서·그림 6개](reports/main_residual_REPORT_KO.md) · [오차 기여 분해](reports/error_diagnosis_REPORT_KO.md) · [현재 해석과 다음 판단](reports/interpretation_REPORT_KO.md) · [이전 matched-readout](../publication_main_20261014_v2/README_KO.md) · [그 이전 전체 결과](../publication_main_20261014_v1/README_KO.md)

현재 MAIN 예측을 보존한 뒤 과거 원 C·과거 K 표현으로 그 잔차를 보정했다. 새 ridge 84개, 고정 비교108개, 공유 FIT 평균4개, 선택 예측36개를 실제 실행했다. ZERO 후보를 포함했고 actual-Y TUNE 정규화 MSE로 선택했다. 검산은 성능 개선이나 독립 일반화 검증이 아니다.

```text
두 seed 평균       Robin MSE / MAE       Beijing MSE / MAE
RAW-F              .340402 / .313192     .589575 / .407917
MAIN               .569547 / .447339     .661957 / .470010
MEAN               .570543 / .446977     .662795 / .476079
RAW33C             .569547 / .447339     .654675 / .475351
E33K               .569547 / .447339     .657561 / .476598
E512K              .569547 / .447339     .661263 / .472481
```

Robin의 세 경로·두 seed 모두 ZERO를 선택했다. 기존 예측을 보존했으며 새 보정 학습의 개선은 확인되지 않았다. Beijing의 일부 MSE 개선에는 MAE 악화가 함께 있다. RAW33C는 추가 원 채널 정보와 다른 계수 용량을 사용하므로 encoder 단독 원인을 확정하지 않는다. 전체 방법의 정확도 문제는 아직 해결되지 않았다.

![최종 MSE와 MAE](../report_main_residual_v1/figures/01_final_seed_scores.png)

![같은 batch와 균형 순서 whole 비용](../report_main_residual_v1/figures/06_whole_path_cost.png)

whole 비용24행/144개 반복은 seed92601의 B1/B8, warmup2/repeat6, Latin6 순서다. ZERO 경로는 MAIN과 동일한 계산이며 feature 전송을 생략한다. 순서와 출력 parity를 검산했지만 데스크톱 배경 활동·thermal/carryover를 완전히 통제했다고 주장하지 않는다. 이전 RAW 지연의 차이도 보존했다.

[점수 CSV](../report_main_residual_v1/METRICS.csv) · [108후보](../report_main_residual_v1/CANDIDATE_GRID.csv) · [whole비용](../report_main_residual_v1/WHOLE_COST.csv) · [독립 과학 감사](../validation_v1/MAIN_RESIDUAL_AUDIT_v1_ACTUAL.json) · [독립 비용 감사](../validation_v1/MAIN_RESIDUAL_COST_AUDIT_v1_ACTUAL.json) · [보고서 최종 검산](../validation_v1/MAIN_RESIDUAL_REPORT_FINAL_VERIFICATION_v1.json)

오차 진단은 같은 4조건·3split·5비교의 선택 예측을 변경하지 않았다. 채널·윈도우 순서4분위·MAIN 절대오차 구간으로 signed MSE/MAE 차이를 분해하며 각 합을1e-10으로 검산했다. 독립/IID 표본·신뢰구간·새 선택의 근거로 사용하지 않는다.

[채널 기여](../report_main_residual_error_diagnosis_v1/CHANNEL_CONTRIBUTIONS.csv) · [블록 기여](../report_main_residual_error_diagnosis_v1/BLOCK_CONTRIBUTIONS.csv) · [오차 구간](../report_main_residual_error_diagnosis_v1/ERROR_BINS.csv) · [TUNE 여유](../report_main_residual_error_diagnosis_v1/TUNE_MARGINS.csv)

[오차 분해 최종 독립 검산 v2](../validation_v1/MAIN_ERROR_DIAGNOSIS_FINAL_VERIFICATION_v2.json). 초기 검증 v1의 파생 해석은 채널별 기여를 합산하지 않고 행 평균을 사용한 집계 오류가 있었다. v2에서 조건별 채널 합산 후 두 seed 평균으로 수정했다. 원 예측·CSV·전체 점수는 변경하지 않았으며 v1 기록도 역사적 시도로 보존했다.

두 seed와 반복 노출 VAL/EVAL 결과이며 보호 TEST·독립 holdout을 사용하지 않았다. 시간 지연의 유용성, 대조학습·그래프의 필수성, Chronos-Bolt의 일반적 유용성은 미확정이다. ExoTimer 전체 문장·LLM 시스템을 재현한 결과가 아니다.

코드·고정 조건·실패 기록·실제 producer/cost JSON 사본·검산·자원 기록과 PNG/SVG·CSV를 게시했다. 원데이터·NPZ·모델·checkpoint·캐시 이진 파일은 제외했다. JSON의 Windows/캐시 경로는 로컬 재현 계보이며 GitHub 파일 링크가 아니다. 다음 단계의 미실행 SOURCE 제안은 이 완료 snapshot에서 제외했다.

실험 당시 고정 지시문과 승인 기록의 자동 commit/push 금지는 그대로 보존했다. 이 게시 묶음은 이후 사용자의 명시적 main 게시 요청을 따른 것이며, 앞으로의 요청되지 않은 자동 게시를 허용하지 않는다.

[게시 파일 SHA](FILES.json) · [게시 링크 검사](LINK_AUDIT.json) · [원 보고서 SHA](ORIGINAL_REPORTS.json)
