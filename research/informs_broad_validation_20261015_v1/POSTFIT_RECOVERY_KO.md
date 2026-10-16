# E2 실패 격리·Raw 재개·CPU 기준선 상태 대조

이 변경은 아직 실행하지 않은 E2의 실행 제어를 보완한다. 학습 목적, 비교 입력, calibration/selection 분할, checkpoint 선택 규칙과 기존 부모 artifact는 바꾸지 않는다. 기존 GPU 학습 및 12개 CPU V 예측 결과를 다시 실행하지 않았다.

## 경로별 실패 격리

`postfit.combine`은 각 E2 경로를 따로 실행한다. 한 B0/B2/B4 부모의 부재나 검증 오류는 해당 경로만 `EXECUTION_FAILED`로 기록하며, 원 오류와 traceback은 private cache의 `failures/`에 새 파일로 남긴다. 뒤의 다른 부모, 과거 입력 대조, Raw 보정은 계속 시도한다.

G 부모는 필요한 첫 경로에서만 읽는다. G가 없으면 G를 요구하는 alpha/head 계열에 명시적인 `MISSING_OR_INVALID_G_PARENT` 이유를 남긴다. Raw의 채널별 bias 보정에는 G가 들어가지 않으므로 계속할 수 있다. 자원 상한 예외와 예측 deadline 예외는 경로별 구현 실패로 바꾸지 않고 상위 실행기로 전달한다. F 불변성 검사는 pool 종료 시 유지한다.

## 완료 Raw 보정의 재사용

`raw_calibration.fit_raw_calibration`은 이미 완료된 같은 ID의 receipt가 있으면 재학습 전에 `validate_completed_receipt`를 호출한다. 검증 대상은 다음과 같다.

- 동일 ID·완료 상태·Raw 보정 family와 source/config/data/scaler/부모 모델 계약
- M/V 부모의 예측·정답·mask·origin·채널 순서 및 metadata 전체 fingerprint. 재사용 여부를 나타내는 operational flag만 제외한다.
- bias NPZ의 경로·SHA·키·유한 값·채널 차원·정규화·내부 계약
- M/V 출력 두 NPZ의 정확한 파일명·SHA·키와 부모 cohort 일치
- 출력이 고정된 `Raw + bias`를 FP32로 변환한 값과 같은지

통과하면 기존 receipt를 그대로 반환하고 기존 wrapper가 보고 단계만 이어간다. 완료 receipt가 없는 부분 생성 디렉터리는 추정 복구하거나 덮어쓰지 않는다. 이 경우 원 자료를 보존하고 실패를 기록한다. 새 receipt의 부모 fingerprint와 실행 시각은 추가 metadata이며, 기존 audit helper가 요구하는 contract 키는 유지했다.

## CPU 기준선 상태 갱신

`reconcile_cpu_baselines(lease)`는 `postfit.main`에서 exclusive `GPUReservation`을 획득한 뒤 호출한다. 과거의 CPU-only 실행 batch에 기록된 Beijing/Jena/Electricity/PeMS08의 CURRENT·SEASONAL_NAIVE·RAW_RIDGE만 허용한다. 새 예측·학습·F 호출은 하지 않는다.

batch source hash·V 분할·원래 registry family/차원·data identity·saved reference·NPZ SHA/배열 fingerprint·F stream 0을 대조한다. 별도의 `audit_predictions` 채점기로 저장 예측을 FP64 재채점하고 공개 통계와 점수 CSV를 재집계한다. 전체 후보 검증이 끝난 다음 해당 행의 상태만 갱신한다. 완료 상태였던 행을 다시 낮추지 않는다. 그 외 행은 갱신하지 않는다.

이 검사는 저장 reference를 이용한 재채점이다. 원 체크포인트 재실행, 새로운 독립 시간 fold, 추가 일반화 검증 또는 배포 비용 재측정으로 해석하지 않는다. CPU batch receipt 자체에는 최초 manifest SHA가 없으므로, 최초 batch가 기록한 SHA와 대조했다고 주장하지 않는다. 현재 실제 manifest의 private archive SHA와 사전에 생성된 reference를 함께 검증한다.

## CPU 검증

실패 재현 후 수선했다. 기존 코드에서 B2/G 부재가 전체 E2 반복으로 전파되고, Raw의 같은 ID 재실행이 `FileExistsError`가 되는 것을 확인했다. 수선 뒤 다음 검사를 통과했다.

- B2 실패 뒤 PAST_CONTROL 및 Raw 보정 진행, G 실패 뒤 Raw 보정 진행
- 완료 Raw 보정의 refit 차단과 동일 receipt 반환
- 부모 예측, source/config/data/scaler/부모 모델 계약 및 bias/M/V 파일 변조 거부
- inactive lease에서 상태 갱신 거부
- 실제 toy CPU 예측 NPZ의 독립 재채점 및 F stream 변조 시 상태 갱신 이전 거부
- campaign 전체 CPU 단위검사: `python -X utf8 -m unittest discover -s . -p 'test_*.py'`, 167개 PASS

실제 캠페인 registry 수정, 실제 E2 실행, 실제 CPU 기준선 reconciliation은 이 검증 작업에서 하지 않았다. 테스트는 임시 디렉터리의 synthetic 데이터와 toy 모델만 사용했다. profiler 단위검사에서 기존 PyTorch cycle 경고 1건이 발생했으며 실패는 없었다.
