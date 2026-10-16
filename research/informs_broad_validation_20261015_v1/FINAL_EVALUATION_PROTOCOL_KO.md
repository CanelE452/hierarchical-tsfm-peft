# 명시적 최종 선택과 재추론 실행 규약

`final_evaluation.py`는 호출자가 지정한 ID를 검증하고 평가하는 장치다. 방법·K·seed·checkpoint를 자동으로 고르지 않는다. import만으로 선택 잠금, T 평가 또는 모델 로딩을 수행하지 않는다. 실제 캠페인의 선택과 실행은 학습 종료 또는 자원 종료를 판단한 상위 실행기가 별도로 호출한다.

## API와 순서

1. `lock_selection(run_ids, decision, public, cache)`
2. `execute_replays(run_ids, cache, public, lease, deadline)`
3. `execute_T(cache, public, lease, deadline)`

`decision`에는 `stop_training=True`와 구체적인 `reason`이 있어야 한다. 이 결정에는 TSFM의 **median quantile 0.5** point 출력, deterministic native 출력, A60 정규화 원 채널의 observed channel-macro MSE 정의를 넣는다. 최종 평가에서 다른 quantile이나 다른 선정 지표로 바꾸지 못한다.

잠금 시 전체 registry에 RUNNING이 없어야 하고 살아 있는 training 목적 GPU lease도 없어야 한다. 실제 V manifest 각각에 대해 오류 없는 `audit_campaign`의 PASS 기록과 manifest SHA를 확인한다. 저장 reference·private V 예측·공개 통계도 독립 채점기로 현재 다시 검증하므로, 이전 감사 이후 reference만 변조된 경우도 거부한다.

그다음 `route_catalog.build_route`가 산출한 중첩 부모를 포함한 정확한 portable spec과 artifact SHA, V manifest/reference/archive SHA, data/scaler/source, registry 과학적 조건을 `SELECTION_LOCK.json`에 봉인한다. `allowed_route_sha`는 실제 route T gate와 연결되고, `allowed_artifact_sha`는 기존 부모 추론기의 gate와 호환된다. 같은 내용의 재호출은 기존 잠금을 반환하며, 다른 ID나 결정을 덮어쓰지 않는다.

미완료 mandatory 행은 원래 상태와 이유를 유지한 채 `mandatory_incomplete`에 남긴다. 예를 들어 `PARTIAL_SCREEN025`를 COMPLETE로 바꾸지 않는다. 이러한 선택 잠금은 `LOCKED_PARTIAL_CAMPAIGN`이며, 선택 경로의 T 실행 성공을 전체 캠페인 완료·수렴·독립 일반화 증거로 바꾸어 말하지 않는다. registry는 이 모듈이 수정하지 않는다.

## T 값과 데이터 로딩의 구분

선택 잠금에서는 `target_window`를 호출하지 않는다. route 구성과 scaler 대조에 사용되는 수치 연산은 A60 정규화 통계뿐이다. T 점수·T 정답·T 사례를 방법 선택에 사용하지 않는다.

기존 `data_contract.load_task`는 승인된 safeFIT archive 전체를 역직렬화할 수 있다. 따라서 **T 행의 byte가 메모리에 전혀 올라오지 않았다**고 주장하지 않는다. 여기서 보장하는 것은 선택 코드가 T target window를 만들거나 T 값으로 적합·선정·평가하지 않는다는 점이다. 기존 데이터 계약의 노출 이력·legacy 정규화 한계는 그대로 남는다.

`execute_T`는 task 로딩보다 먼저 선택 잠금, 고정 소스 전체, 선택 artifact, V 증거를 대조한다. 각 경로의 실제 Task data hash·A60 scaler·portable route가 잠금과 같아야 `route_predictions.predict_route(..., 'T_final')`를 호출한다. 이 함수의 T gate도 같은 route hash를 확인한 뒤 T 정답을 만든다. 최종 T의 공식 진입점은 이러한 상위 소스 검사를 수행하는 `execute_T`다.

## 실행 자원과 모델

두 실행 API는 외부에서 획득한 `purpose='audit'`의 활성 exclusive owner lease를 요구한다. 새 `GPUReservation`을 안에서 만들지 않는다. 상위 실행기는 reserved audit/cost 자원의 사용량과 전체 deadline을 관리해야 한다.

RAW_RIDGE·CURRENT·SEASONAL은 CPU 경로이며 F stream은 0이다. 그 외 학습 모델은 GPU 장치를 지정한다. 실제 활성 부모에 F가 있을 때만 해당 모델을 로드한다. Bolt-small은 `INFORMS_ASSET_ROOT`의 고정 revision snapshot을, mini/tiny는 `MODEL_ASSETS.json`에 고정된 private local snapshot을 사용한다. 설치·다운로드·모델 대체는 하지 않는다.

이 실행기는 경로 사이에 여러 foundation을 상주시켜 두지 않는다. 각 `Route`의 `finally: close()`가 LoRA 비활성화와 원래 F 상태 검사를 수행한다. 다른 size나 H를 한 pool에서 섞어 쓰지 않는다. 이는 배포 비용 측정기가 아니므로 모델 재로딩 시간을 제거한 비용 또는 여러 방법의 비용 우위를 보고하지 않는다.

한 경로의 실제 실행 실패나 replay 수치 불일치는 그 경로의 상태로 기록하고 다른 경로를 계속한다. 원 traceback은 private cache에 두며 공개 batch receipt에는 오류 유형과 상대 경로만 남긴다. 예산 또는 deadline이 소진되면 `DEFERRED_RESOURCE`와 아직 남은 ID를 기록한다. 완료된 artifact는 기존 helper의 hash 검증 후 재사용하며, 결과 파일을 덮어쓰지 않는다.

반환값과 공개 batch receipt에는 성공/실패/재사용 수, manifest·replay receipt 상대 경로와 SHA만 담는다. 원 정답과 전체 예측 배열은 공개하지 않는다. 최종 SCORES 집계와 T의 독립 재채점·보고서는 상위 실행기의 후속 단계다.

## 검증 범위

CPU toy 검사에서 다음을 확인했다.

- 감사 부재, stop 결정 부재, RUNNING 행, 활성 training lease 거부
- PARTIAL 상태와 registry byte 보존 및 동일 잠금 재사용
- 소스·artifact 변경 시 task/T 정답을 읽기 전에 거부
- 감사 이후 reference 변경 거부
- 실제 CURRENT toy T forward와 manifest 생성, 공개 반환값에 전체 배열 없음
- 한 replay 경로 실패 이후 다른 경로의 실제 CPU toy replay 진행
- 시작 전에 deadline이 끝난 경우 forward 없이 남은 경로를 DEFERRED로 기록

이 검증은 임시 synthetic 데이터와 toy artifact만 이용했다. 실제 캠페인에 대한 선택 잠금·T 평가·GPU 실행은 이 모듈의 구현 검증에서 수행하지 않았다.
