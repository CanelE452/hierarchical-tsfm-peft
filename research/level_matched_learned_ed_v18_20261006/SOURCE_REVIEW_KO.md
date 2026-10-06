# 중단 후 비용·평가 코드 계약 점검

**BLOCKED_STOP_RESOURCE. 학습·정확도·비용 비교는 여전히 미완료다.** 아래 수정은 아직 실행하지 않은 후속 코드의 결함을 정적 추적하고, 격리된 합성 CPU fixture로 확인한 것이다. 실제 learned E/D나 LEVEL의 성능·시간·메모리 결과로 해석하지 않는다. 점검 영수증과 코드 SHA는 `source_review_checks.json`에 있다.

비용 경로에서는 E/D에만 포함됐던 gradient 진단·동기화를 측정 구간에서 분리하고, 두 arm에 동일한 입력 전송 → mask-macro MSE → backward 경로를 사용하도록 수정했다. E/D는 frozen backbone 연산을 통해 gradient를 유지하고, LEVEL은 기존 main-path gradient 차단을 유지한다. optimizer.step은 없다. OOM 후 모델 참조를 해제하며 실패 행은 보존한다. 기존 manifest에 없던 fitted-parameter 필드를 참조하는 대신 실제 residual 파라미터를 열거한다. 누락·OOM·미완성 행을 성공으로 처리하지 않도록 그리드, 원시 반복, 입력 바인딩과 모델 상태 검산을 강화했다.

평가 경로에서는 TEST 등록 metadata가 `dataset::run_id`를 덮어쓰던 오류를 수정했다. 재사용 예측은 선택된 checkpoint·LR·epoch, 데이터·원점, selected adapter/backbone/PCA 및 완료 ledger 항목과 함께 확인한다. 파일을 저장한 뒤 종료 기록이 실패한 경우에는 그 complete receipt만으로 실패를 성공으로 바꾸지 않는다. 정확한 여섯 예측과 manifest·selection·seal 참조를 다시 검사하고, 기존 비교군 manifest를 Phase0의 고정 SHA로 읽는다.

[확인] 기존 preflight, 실패 attempt, 초기 모델 기록, ledger, resource stop, 학습·모델·runtime·preflight 코드, protocol과 네 그림 파일의 bytes/hash는 보존했다. 11,421.533초 projection을 다시 추정하거나 10,800초 상한을 바꾸지 않았다. 새 neural fit, optimizer update, 실제 TEST 예측, GPU job 및 실제 자료의 bootstrap은 추가하지 않았다. 기존 보호 파일도 별도로 검산한다.

CPU fixture 검사는 입력·mask가 다른 실패 사례, 누락·변경·교체된 영수증, 실패 ledger와 완료 판정 경로의 오류를 잡기 위한 소프트웨어 검사다. fake model·synthetic prediction은 새 연구 결과가 아니며 실제 비용 측정이나 독립 재현을 대신하지 않는다. 실패 원본과 수정 전 공개 코드·검산 영수증은 새 `.cache`의 source-review snapshot에 보존했다.

유지할 것은 기존 LEVEL 관측의 한정 범위와 preflight readiness다. matched 비교를 완료했다는 주장은 축소하고, G의 필요성·정확도 winner·backprop 및 inference 비용 우위는 보류한다. 가장 강한 “같은 K의 learned E/D가 G를 대체할 수 있다”는 반론은 아직 답하지 못했다. 이를 판단하려면 승인된 예산 안에서 동일 12-fit 계획, TEST 전 joint selection, 여섯 selected TEST 예측, paired uncertainty, 같은 회차의 비용 raw blocks와 실제 결과를 검산하는 완료 verifier가 필요하다.
