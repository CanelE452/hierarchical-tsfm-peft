# Raw/context multiscale experiment

[한국어 결과와 해석](REPORT_INTERPRETATION_KO.md) · [그림14개](FIGURE_INDEX_KO.md) · [현재 검산·재생성 명령](README_USAGE_RECOVERY_KO.md)

실제 완료 범위는 V56경로, 전체추론비용120개, CP0진단10개, full-Mcal진단40개/seed평균20개, head-component비용16개/480trial이다. V는 개발 평가이며 독립 TEST가 아니다. 불리한 결과도 모두 보존했다.

원 main002와 첫 CPU 복구001의 실패를 보존하고 별도 [복구002 PASS](RECOVERY_VALIDATION_REPORT_002.json) 및 [독립 검산 PASS](INDEPENDENT_FINAL_AUDIT_001.json)를 확인했다. 학습·예측·비용 측정을 다시 만들거나 선택/공차를 바꾸지 않았다. 생성된 과거 README_KO/CLI_USAGE의 직접 verify 명령보다 위 현재 명령 문서를 우선한다.

- [최종 표와 원본 그림](final/README_KO.md)
- [Mcal 최종모델 오차](MCAL_FINAL_FIT_DIAGNOSTICS.csv) · [두 seed 평균](MCAL_FINAL_FIT_MEANS.csv)
- [Head component 비용](HEAD_COMPONENT_COST.csv) · [480 trial](HEAD_COMPONENT_COST_TRIALS.csv)
- [부모 식별표](PARENT_IDENTITIES.csv) · [재개 상태 보존 검산](EXACT_RESUME_VALIDATION_001.json)
- [14개 그림 시각 검토](FINAL_VISUAL_AUDIT_001.json)
- [472개 집중 테스트·소스 지문](PUBLICATION_CODE_VALIDATION_001.json) · [최종 자원 스냅샷](RESOURCE_COMPLETION_SNAPSHOT_001.json)

앞서 요청한 추가 데이터 ETTm1 검증은 [별도 결과](../b2_dataset_search_20261019_v1/waves/WAVE_003_results/RESULTS_KO.md)와 [23dfead 커밋](https://github.com/CanelE452/hierarchical-tsfm-peft/commit/23dfead446a03bb506d87fad99e9a159ce7f3ee6)에 게시했다. K4는 Raw 대비 평균 MSE14.40% 감소, B8 기본경로 약1.29배 속도, allocated10.12% 감소였으나 MAE는4.55% 증가했다. ETTm1은 ETTh1과 같은 관측소이므로 독립된 새 도메인으로 세지 않았다.

새 학습 cap 연장·PeMS B·같은 표현의 no-Bolt 대조는 이번 등록 범위 밖이다. 마지막 cap best는 수렴 미확정으로 남겼다. 새 보호 T 수치 연산은0이지만 과거 노출·상류 전처리·사전학습 중복 불확실성은 배제하지 않는다.
