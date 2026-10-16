# Electricity Raw 원 parity 반환의 실제 진단

[실제 결과·그림·집계 CSV](../electricity_raw_parity_report_20261016_v1/REPORT_KO.md)를 먼저 읽는다. 원 cost worker는 반환 dict 저장 전에 assertion으로 종료되어 실패 종류를 구분하지 못했다. 이 후속은 원 함수의 반환을 보존하며, 새 학습이나 정답 평가를 수행하지 않는다.

실제 GPU 진단은 [capture_original_parity.py](capture_original_parity.py)와 [PLAN.json](PLAN.json)으로 실행했다. 원 source39·registry·selection lock·spec2·실패8건과 private log8건은 SHA로 전후 대사했다. B1/B8에서 각1회 원 함수를 호출했고, 8개 variant 모두 반환됐으며 분할6개가 원 공차에서 FAIL이었다. 이번 재현에서 OOM은 나타나지 않았다. 역사적 모든 실패가 같은 원인이었다고 확정하지 않는다.

[GPU_AUDIT.json](GPU_AUDIT.json)은 실제 CLOSED·누적 상한·정상 자식 종료 확인을 기록한다. 실제 실행은 timeout 없이 끝났으며, timeout 경로의 모든 상황이 GPU에서 검증됐다는 뜻은 아니다.

후속 검토에서 timeout 시 자손 하나가 열거 중 사라지면 다른 살아 있는 자손의 기록까지 버릴 수 있는 경계를 발견했다. 실행에 사용한 소스는 보존하고 [capture_original_parity_hardened.py](capture_original_parity_hardened.py)와 [PLAN_HARDENED.json](PLAN_HARDENED.json)을 별도로 만들었다. 후속 버전은 사라진 자손만 건너뛰고 전체 열거 실패를 종료 확인으로 바꾸지 않는다. 재실행을 구현할 때는 이 후속 버전을 사용한다. 이번 GPU 결과를 후속 버전으로 실행했다고 재표기하지 않는다.

[실행 버전 테스트](test_capture_metadata.py) 8개와 [후속 테스트](test_capture_metadata_hardened.py) 9개는 실제 통과했다. 공통 검사가 겹친다. 실제 Windows에서 새로 만든 CPU sleep 작업의 launcher·runtime을 PID/생성 시각으로 확인해 함께 종료·회수하는 검사도 포함한다. 후속 버전의 GPU 실행은 수행하지 않았다. [검증 기록](PREPARATION_VERIFICATION.json)

기본 CLI는 metadata만 검사한다. 실제 GPU 호출은 기존 audit owner를 인자로 받는 `run_with_owner`에서만 가능하다. owner 누적 상한은 원43200초이며 경과 상한도 그대로다. 공개 target·전체 예측 배열·백본 파일을 만들지 않는다.
