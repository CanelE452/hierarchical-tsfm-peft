# 재현과 재개

이 snapshot의 report 생성은 CPU 저장 결과 재채점이며 학습/모델 추론을 실행하지 않는다. 입력 SHA는 INPUT_MANIFEST.json, 산출물 SHA와 compiler source는 REPORT_MANIFEST.json에 기록한다. 원 데이터와 weights는 로컬 private cache에 남긴다.

1. 저장소에서 CONFIG_LOCK.json, DATA_CONTRACT.json, 초기/확장 registry, 공식 모델 pin을 확인한다.
2. INFORMS_ASSET_ROOT와 INFORMS_CACHE_ROOT를 기존 실제 자산 위치로 설정한다. 이 컴퓨터의 정확한 예시는 RESUME_COMMANDS.ps1에 있다.
3. 중앙 lease가 실행 중이면 기다린다. 기존 프로세스 종료나 중복 GPU 실행을 하지 않는다.
4. RESUME_QUEUE.json의 상태별 선행 조건을 만족한 route만 명령으로 실행한다. 구현/접근 계약이 없는 항목은 명령을 지어내지 않았다.
5. 재개 carry가 명시된 BUDGET_LIMIT 작업은 해당 receipt의 exact resume 명령을 사용한다. 이전 2400 구간의 RNG 복구를 가정하지 않는다.
6. 모든 저장 V 예측을 독립 감사하고 실제 checkpoint replay·비용을 측정한 뒤, 별도 판단으로 선택을 동결해야 T를 실행할 수 있다.
7. 모든 available V/T seed와 조건을 유지한다. T를 보고 seed나 방법을 다시 고르지 않는다.
8. 새 그림/report snapshot을 생성한다. 보고서는 원 입력·기존 결과·registry를 덮어쓰지 않는다.

학습 명령은 PRE_SELECTION_ONLY다. 선택이 locked된 뒤 새 학습은 현재 캠페인에서 재개하지 않는다. 새 campaign/별도 ID로 POST_T_DEVELOPMENT를 분리하고 기존 T를 학습·hyperparameter 선택에 재사용하지 않는다. E0의 과거 B 연장은 별도 개발 증거이며 고정된 E1–E4/T 선택으로 되먹이지 않는다.

현재 queue 항목 437개. 추가 GPU/CPU/전체 소요시간은 미측정 조건이 있어 합계 ETA를 계산하지 않았다. 각 runner의 기존 자원 상한과 실제 부과 ledger가 우선한다. M5는 공식 자산/접근이 생겨야 재개 가능하며 권한 우회나 자동 동의는 하지 않는다.

```powershell
& $py -X utf8 -m unittest test_report_campaign -v
& $py -X utf8 report_campaign.py --public-root . --private-root $env:INFORMS_CACHE_ROOT
```

이 문서에 commit/push 명령은 없다. 게시 여부는 실제 원격 SHA와 파일/링크 확인 후 별도 기록해야 한다.
