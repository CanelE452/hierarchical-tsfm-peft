# 동일 추가 학습 예산과 전체 추론 비용 재검토

사용자가 승인한 범위: PCA·시점별 MLP·시간형 CNN을 동일 디코더 조건과 추가 학습 예산으로 다시 비교하고, 예측오차 및 전체 추론 시간·메모리를 확인한다. 새 방법 탐색·TEST·commit/push는 이번 범위에 포함하지 않는다.

## 목적과 판단

상위 목적은 고정 TSFM의 채널 처리 비용을 줄일 때 정확도와 실제 전체 비용의 교환관계를 확인하는 것이다. 소비자는 연구자와 교수님이며, 시간형 인코더가 필요하다는 주장을 유지할 근거가 있는지 판단한다.

학습 예산 확대의 목적 후보는 (A) 기존 최대 60 epoch의 학습 부족 때문에 순위가 달랐는지 확인, (B) 주어진 추가 update 예산 안에서 더 좋은 표현을 얻는지 확인이다. 같은 decoder 재적합의 목적 후보는 (A) encoder 변경 효과와 readout 조건 차이를 구분, (B) 각 좌표계에 적합한 미래 복원을 제공이다. 비용 측정의 목적 후보는 (A) C→K의 실제 추론 이익 확인, (B) 추가 인코더 비용이 이익을 상쇄하는지 확인이다.

예상 결과와 해석을 사전에 고정한다. 시간형의 정확도/비용이 개선되면 필요성을 조건부로 지지한다. 시점별과 비슷하거나 나쁘면 시간형 필요성은 지지하지 않는다. 압축 경로가 원채널보다 빠르지만 덜 정확하면 교환관계를 보고한다. 압축 경로가 더 느리거나 메모리 이익이 없으면 해당 조건의 경량화 장점을 주장하지 않는다. 어느 결과도 독립 TEST 우월성이나 일반적 수렴을 증명하지 않는다.

## 확인된 사실

- [확인] 기존 best checkpoint에는 Adam state/RNG/last state가 없다. best weights에 새 Adam을 붙이는 것은 정확한 이어학습이 아니다.
- [확인] 기존 자료·seed·K·입력512·예측48·학습origin512/epoch·batch32·D8·TRAIN RMS/ridge grid를 재사용할 수 있다.
- [확인] RTX 4070 12GiB가 있으며 실행 전 Python 학습 프로세스는 없다. 기존 과학 결과는 읽기 전용으로 보존한다.

## 확인할 전제 / 약한 고리

- [추정] 같은 초기화와 sampling의 fresh trajectory는 기존 실행의 실제 prefix를 재현한다. 기존 checkpoint 선택 epoch의 encoder/decoder state 및 공통 curve를 수치 검증한다. 불일치면 이유를 밝히기 전 legacy 대비 확대학습 효과를 단정하지 않는다.
- [추정] benchmark는 평가한 E-F-D와 같은 예측을 만든다. 각 정확한 checkpoint/D로 원시 VAL 예측과 parity를 검증한다. batch 차이에 의한 부동소수 오차는 허용치를 기록한다.
- 추가 120 epoch도 plateau에 도달하지 않을 수 있다. 마지막 10/30 epoch 변화, 선택 epoch 및 cap flag를 보고하고 무제한 최적화하지 않는다.
- 기존 VAL을 선택/보고에 다시 사용하므로 독립 확인이 아니다. 원채널 참고도 이번 동일 경로로 다시 평가한다.

## 실행 계약

1. Robin/Jena, seed 92601/92602를 고정한다. PCA는 encoder 학습 없음; MLP_POINT/TCN_TIME의 encoder 변수 수는 기존처럼 약1% 이내다.
2. fresh 180 epoch를 두 학습형 모두 수행한다. early stop 없이 각 2880 optimizer updates(512 origins/32 batch ×180), 공통 추가 예산은 60→180의 120 epoch/1920 updates다. prefix 재현 60 epoch 비용도 이번 실행 비용에 포함한다. legacy early stop 이후는 새 trajectory라고 명시한다.
3. 초기화·Adam lr.001·weight decay0·clip1·TRAIN 관측 마스크·burn16·sampling을 유지한다. epoch0 포함 가장 이른 full VAL reconstruction 최솟값을 60/180 milestone 각각 선택한다. 미래 예측으로 E를 고르지 않는다.
4. 각 milestone에서 같은 D8 구조, 공통 TRAIN512 origins, TRAIN-only feature RMS, lambda[.0001,.01,1], 같은 VAL 선택을 적용한다. D 가중치는 각 표현에 별도 적합한다. 주결과 FORECAST_RIDGE_D8, 보조 RECON_RIDGE_D8. 실제 TRAIN F 예측으로 주 D를 적합하며 미래 실제 latent를 decoder 입력에 넣지 않는다.
5. 전체 full VAL MSE/MAE를 관측 마스크의 channel-macro로 보고한다. seed 평균은 prediction ensemble이 아니다. PCA와 RAW_F0는 같은 full VAL 기준으로 재평가한다.
6. exact 선택 E180/주 D의 전체 E-F-D를 B1/B32로 측정한다. 입력 준비/표준화·입력 transfer·E·F 내부 scaling 및 median48·D8·출력 transfer/synchronization을 포함한다. startup/import/model load/disk I/O/scoring/hash는 제외한다. 고정 warmup5, timed20 이상; 순서는 rotate한다. checkpoint와 D receipt, 같은 입력origin/dtype/device, 원시 latency, median/p95, throughput, CUDA peak allocated/reserved, host RSS 관측치를 남긴다. 원채널 F0도 같은 환경/입력에서 측정한다.
7. 학습 process와 분리한 benchmark process에서 측정해 학습 optimizer/activation 잔류를 배제한다. benchmark 예측이 scientific 평가 예측과 허용 오차 안에서 일치해야 한다. GPU memory는 reset_peak_memory_stats 이후 진짜 allocator peak; RSS는 sampling 관측치로 구분한다.
8. F는 frozen/no_grad이며 역전파0, TEST0, offline only. 기존 파일 전후 hash 일치. 실패/재시도는 원장과 비용에 포함한다. 신규 cache/checkpoint에 Adam/RNG/last state를 저장한다.

## 완료 증거

8개의180epoch fit/update원장·60/180curve/checkpoint·D원시예측/선택receipt·PCA/F0 재평가·각자료/방법/seed B1/B32 cost rows·pipeline parity·독립 metrics/selection/update/source/preservation 검증·한국어 결과보고서와 비용-오차 그림·history가 필요하다. 검증 성공과 과학적 우월성은 별개다.

## 실행자 게이트

실제 API/배열/split이 존재하는지 먼저 읽는다. 공통 sampling/budget/decoder 조건이 수치로 성립해야 한다. 미래 정답의 입력 누수, 기존 출력 덮어쓰기, source seal 불일치가 있으면 실행을 멈추고 원인을 확인한다. 기록된 값이 부족하면 이름이나 관례로 완료를 추정하지 않는다. 새로운 실행만 이 폴더와 같은 이름 cache에 쓴다.
