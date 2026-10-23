# 공개 결과 독립 검증기

학습과 canonical 보고서 생성이 모두 끝난 후 저장소 루트에서 실행한다.

```powershell
python -B .cache/k_tau_rule_20261023_v1/verify_final_report.py --replay
```

기본 출력은 `reports/k_tau_rule_20261023_v1/FINAL_REPORT_VALIDATION.json`이다.
종료코드는 PASS=0, FAIL=1, INCOMPLETE=2다. 미완료·중단·누락 산출물에는 PASS를 내리지 않는다.
`--replay` 없이도 독립 수치 검증을 수행한다. 다시 replay할 때는 기존 디렉터리를 덮어쓰지 않으므로
`--replay-dir .cache/k_tau_rule_20261023_v1/report_replay_2`처럼 새 경로를 지정한다.

검증기는 표준 라이브러리만으로 공개 RUNS/K_TAU/BASELINE_RECALCULATION/RUN_PLAN/SUMMARY/DECISION을 읽는다.
신규 실행은 공개 RESULT/SELECTION/SEARCH_RESULT JSON의 완료 scalar·식별 필드와 대조한다.
checkpoint, selected bundle, prediction 배열, 원자료, 모델은 열지 않는다.

- 정확히 네 데이터 × seed92601–92610, A_fit K, 계획된 재사용6개/신규 main34개를 확인한다.
- G4 및 기존 K grid에서 실행·재사용 집합을 독립적으로 구성하고 RUN_PLAN/RUNS와 대조한다.
- 기존 두 seed baseline의 MSE/MAE 평균·최소·최대와 재사용 scalar를 재계산한다.
- G4 실패 시 새 comparison 두 seed 결과를 baseline으로 쓰며, 비교가 없거나 불완전하면 손해와 판정을 확정하지 않는다.
- stdlib median, 직접 선형 보간한 Q1/Q3, fsum/n 평균, IQR, 첫 두 seed 기준 손해, LIMITATION을 계산한다.
- 판정에는 jena/ETTh1/appliances만 쓰고 ETTh2는 기술 결과로 유지한다.
- cap3 checkpoint0/.25/1/3의 최소 A_stop와 정확한 동률의 이른 epoch, 선택 봉인 SHA, 완료 FP64 검산 영수증을 확인한다.
- 그림 PNG/SVG 네 파일은 존재와 크기만 확인한다. 이미지 bytes와 시각 품질은 검증하지 않는다.

선택적 replay는 독립 검증이 PASS일 때만 공개 입력 네 파일을 cache의 새 `report_replay` 디렉터리에 복사한다.
동결된 `analysis.build_report`를 그 복사본에만 실행하고 SUMMARY 및 DECISION(elapsed_seconds 제외)을 비교한다.
canonical 보고서는 수정하지 않는다. replay는 독립 계산의 근거가 아니라 같은 공개 입력의 재실행 확인이다.

합성 fixture 점검은 다음과 같이 실행한다. 실제 캠페인 산출물에는 접근하지 않는다.

```powershell
python -B .cache/k_tau_rule_20261023_v1/verify_final_report.py --self-test
```

독립 검증은 공개 scalar 영수증의 일관성을 검증한다. FP64 prediction 배열 재검산 자체는 기존 학습 코어의
`independent_FP64=PASS` 영수증에 근거하며, 이 스크립트가 배열을 새로 열어 검산했다고 주장하지 않는다.
