# 기존 Jena B2 검증과 동일 조건 비용

[확인] 기존 B2(K6, seed92601)의 저장 CPU fullV 정확도를 독립 FP64로 다시 확인했고, 같은 선택 checkpoint의 fresh CPU6 추론과 CPU2/RSS3GiB 조건의 B1·B8 비용 측정을 완료했다. 이번 작업의 신규 학습은 **0**이다. 정확도는 기존 결과 재사용이며, 새 검증/비용을 새 학습 성과로 세지 않는다. 근거는 [독립 감사 PASS](INDEPENDENT_B2_REFERENCE_AUDIT_ROOT_v1.json), 원정밀도 [비용 CSV](B2_REFERENCE_COSTS_v1.csv), [입력·출력 SHA 결속](B2_REFERENCE_REPORT_BINDINGS_v1.json)이다.

기존 V43의 채널 macro 정규화 MSE는 **0.6834428963980711**, MAE는 **0.49388492449756366**이다. 관측 라벨43344개와 공개 충분통계2709행을 대사한 저장 V 검산이며, 새 독립 평가나 fullV 재추론은 아니다. C21/L512/H48/K6, 같은 원 A60 scaler와 원 target/mask/origin/channel 지문 및 원 선택 bundle을 유지했다. [원 manifest](../postT_V_only_cpu_lag_20261016_v1/predictions/postT_cpu_jena_E3_B2_k6_s92601_v1/V_select/PREDICTION_MANIFEST.json)와 [이번 FP64 영수증](diagnostics/jena_b2_reference_rescore_root_v1/RESCORE_RECEIPT.json)을 연결했다.

## 실제 추론 재현과 비용

[확인] fresh CPU6은 원43개 origin 중6개를4+2 batch로 다시 추론했다. 원 예측 대비 최대 절대 차이는 **9.5367431640625e-07**, 사전 공차 atol1e-5/rtol1e-4 안이다. 실제36 streams/2 physical calls이며 [CPU6 영수증](diagnostics/jena_b2_reference_replay_root_v1/REPLAY_RECEIPT.json)의 `executed_now=true`를 확인한 감사 결과다. 검사 통과는 원 선택 모델과 CPU physical route의 재현성 근거다.

전체 physical batch 지연의 원정밀도 수치는 다음과 같다. 초 단위이며 B8은8개 입력을 함께 처리한 한 batch의 시간이다. 단계 계측(stage_probe)은 별도 침습적 측정이므로 timed 전체 지연과 구분한다.

```text
phase        batch  trials median_seconds       p90_seconds
 timed       1      20     0.047770299948751926   0.04950220994651318
 timed       8      20     0.32702095014974475    0.33335708011873066
 stage_probe 1      20     0.047176399966701865   0.04834801009856165
 stage_probe 8      20     0.3413905999623239     0.3506813001353294
```

[확인] B1/B8 각각 warm10+timed20+stage20의50회, 총100개 trial과4개 summary/20개 수치 필드를 원 recipe로 별도 검산했다. 비용의 실제 F 합계는2700 streams/100 calls다. 매 trial의 B1은6 streams/1 call, B8은48 streams/1 call이다. 비용에는 실제 physical 전처리→원 P/E→동결 Bolt→원 D→FP64 물리 단위 역변환을 사용했고 cached Z/forecast는 읽지 않았다. 모델 로드는 whole trial timer에서 제외한다. [B1 비용 영수증](diagnostics/jena_b2_reference_cost_b1_root_v1/COST_RECEIPT.json)과 [B8 비용 영수증](diagnostics/jena_b2_reference_cost_b8_root_v1/COST_RECEIPT.json)에 실제 입력 SHA와 이번 rescore/replay 증거가 묶여 있다.

## 메모리·종료·측정 범위

[확인] 전체 Route CPU weights는 **182.1661491394043 MiB**다. 비용 worker의 관찰된 tree peak RSS는 B1 **1011871744 bytes**, B8 **1082167296 bytes**이며 각각 새3GiB 상한 안이다. 이는 비용 worker 전체의 관찰 peak이며 F만의 메모리나 학습 peak가 아니다. `F_resident_bytes`는 별도 기록이 없어 **NA / NOT_SEPARATELY_RECORDED**로 남긴다. 파일 크기·전체 weights·RSS를 F 상주 메모리로 대입하지 않는다.

Route의 `load_seconds`는 이미 앞에서 생성된 FoundationPool의 cold load를 포함하지 않는다. 감사 aggregate에는 cold-load 분리 실측 시간이 없으므로 해당 값도 NA다. whole latency·Route loading·F pool cold loading을 더하거나 비용 완전치로 바꾸지 않았다. 모델 로드가 포함된 cold-start latency는 이번 표의 평가 범위에 포함되지 않는다.

[확인] ROOT의 실제 종료/복원 확인에 따르면 replay·비용 후 F는 원 hash `c2b4edf7db22d3f18e58ca79299a5c7e864e9b8f277e4e29e2f88d07dbb35874`로 보존·복원되었다. 본 보고 작성자는 모델을 다시 로드하거나 hash를 재계산하지 않았다. 감사는 네 guard의 종료·현재 정확 PID 생성시각 identity 부재, 새 cost quiet, validation/cost 구간 겹침 없음과 입력 SHA 전후 보존을 확인했다. 원 raw/target/선택 checkpoint를 공개하지 않으며 보호 T에 대한 신규 target 연산은0이다.

## 이 결과로 말할 수 있는 범위

이번 결과는 기존 seed1 B2를 같은 CPU2/RSS3GiB quiet 비용 조건으로 비교표에 연결할 근거를 채웠다. 원 CPU 학습 guard는4GiB였고 그 ancestry를 새3GiB 학습으로 소급 변경하지 않는다. 원 TRAIN 정규화 계보 역시 유지된다. [새 reference PLAN](B2_REFERENCE_PLAN.json)과 [실행 entry](jena_b2_reference.py)는 기존 선택 weights에 대한 검산/비용만 허용한다.

한 seed, 반복 사용한 개발 V, 제한된 원 학습/epoch 선택이라는 조건은 유지된다. 성능 개선·수렴·독립 일반화·다른 family보다 우월함은 이번 검사로 확인되지 않았다. 향후 LoRA/B4와의 정확도/비용 비교에는 각 조건의 실제 완료·선택 학습량·모수·장치·정규화·물리 입력 지문을 함께 맞춰야 한다. 전체 MASTER는 여전히 미완료다. 이번 보고 작성에서는 추가 모델·추론·비용·통계 재집계·프로세스 조회를 실행하지 않고 완료된 감사 aggregate와 입력 SHA만 사용했다.
