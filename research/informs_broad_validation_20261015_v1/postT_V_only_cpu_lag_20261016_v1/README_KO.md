# Jena 지연 압축의 별도 CPU 비교

[1 epoch까지의 실제 중간 보고서·곡선·CSV](PROGRESS_E1_KO.md). 최종 V와 비용은 아직 완료하지 않은 시점의 고정 snapshot이다.

현재 문서는 새 비교의 실행 계약이다. 실제 완료 여부는 각 diagnostic의 RESULT와 원 학습 receipt로 판정한다. 계획·합성 검사·1 update 속도 측정을 전체 학습 완료로 세지 않는다.

## 목적과 같은 조건

Jena의 같은 과거 512행, 미래 48행, 원 채널 21개, 잠재 차원 6개, seed92601에서 B0·B2·LAG_CAUSAL·LAG_BIDIR를 비교한다. 기존 `campaign.py`의 학습 템플릿을 사용하고 장치만 CPU로 명시한다. 같은 frozen Bolt-small, 선형 디코더와 현재값 anchor, 원 미래 정규화 channel-macro MSE, AdamW 학습률0.001, effective batch16/microbatch4를 유지한다. 실제 frozen Bolt를 통과한 입력 gradient로 압축기를 학습하며 백본 가중치는 갱신하지 않는다.

검색은 A50의 전체 legal origins9923개에 대해 epoch0.25·1·3까지 진행한다. A_stop43 origins로 checkpoint를 선택하고, 같은 선택 epoch를 A60에서 새로 적합한 뒤 V43 origins를 실제 추론한다. 전체 검색 업데이트는 endpoint의 나머지 batch를 각각 반영해156·622·1863회다. 선택 epoch가0이면 refit update도0이며, epoch3을 검색했다는 사실을 수렴 증명으로 바꾸지 않는다.

CPU 실행 속도 측정은 B0의 첫 effective update16sample만 사용한다. 이때도 전체 A50 cohort를 사용한 손실 정규화·PCA·seed를 유지한다. 측정 후 모델·optimizer·sampler를 폐기하고 네 학습은 처음부터 새로 시작한다. 한 번의 update 시간은 warmup과 동시 실행 영향을 포함하므로 전체 학습 ETA에 오차가 있다.

## 기존 평가와 분리

이 작업은 **기존 T 결과를 이미 관측한 뒤의 탐색적 V 전용 development**다. 새 ID와 별도 public/private root를 사용한다. 원 registry·selection lock·source39·T와 원 점수는 변경하지 않으며, 기존 중단 학습을 재개하지 않는다. 원 GPU3epoch 미달48개를 이 CPU 결과로 완료 처리하지 않는다.

원 `data_contract.load_task`의 승인된 safeFIT 전체 역직렬화 계약을 유지한다. 따라서 T 행 byte가 메모리에 전혀 올라오지 않는다는 주장을 하지 않는다. 학습·통계·선택에는 A50/A_stop/A60만 쓰고, 새 예측·채점은 V만 요청한다. 새 T target 구성·적합·선택·평가는 하지 않는다. legacy full-TRAIN 정규화의 FP32 역변환 이력과 독립 holdout 부재도 그대로 남는다.

4조건은 기존에 등록된 Jena E3 확장 및 그 B0/B2 대조를 기준으로 정했고 T 순위로 고르지 않았다. 양방향 지연도 origin 이전의 입력 창 내부만 사용한다. B0와 지연 인코더의 차이에는 파라미터·경계조건 차이도 있어 성능 차이만으로 인과성 효과를 증명할 수 없다.

## 실행과 검증

- [고정 계획](PLAN.json), [별도 등록](REGISTRY.csv), [실행 코드](cpu_lag_runner.py), [합성 검사](test_cpu_lag_runner.py).
- [저장 예측 검산·checkpoint 재추론](verify_cpu_outputs.py), [검사](test_verify_cpu_outputs.py), [CPU 비용8조건 준비](prepare_cpu_costs.py), [검사](test_prepare_cpu_costs.py).
- [CPU 프로세스 자원 감독 코드](cpu_group_supervisor.py), [검사](test_cpu_group_supervisor.py)는 실제 실행에 사용한 private 사본과 byte·SHA가 같은 공개 사본이다. 새로 생성한 자식의 PID/생성시간과 관측한 자손만 추적한다. 샘플링 RSS/CPU 기록이며 관측 전에 재부모화된 자손의 부재를 증명하지는 않는다.
- 기본 CLI는 metadata만 검사한다. `--execute`와 plan SHA·단일 run ID·새 diagnostic 경로·deadline이 있어야 실제 모델을 읽는다.
- 원 데이터·scaler·채널·source·foundation tensor hash를 검사한다. 실제 CPU runtime의 PID/생성시간·RSS·CPU 시간과 frozen 상태를 기록한다. 각 worker4GiB·전체8GiB 이내에서 실제 측정에 따라 동시 실행 수를 정한다.
- 실제 학습 종료 후 저장 V 예측을 별도로 재채점하고 선택 checkpoint를 재추론한다. CPU 추론 비용은 기존 GPU 비용과 별도 조건으로 보고한다.
- traceback과 가중치·예측·정답 배열은 private cache에만 보존하고 공개 결과는 집계·지문·상대경로로 제한한다.

원 캠페인의 미완료 범위는 [RESUME](../RESUME.md)에 유지한다. 이 네 비교의 성공 여부가 전체 미완료 항목의 완료 여부를 대신하지 않는다.
