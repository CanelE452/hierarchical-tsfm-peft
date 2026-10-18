# 방법과 실제 출처

이번 신규 방법은 기존 B2 앞뒤에 적용한 원채널 outer RevIN 하나다. 전체 RevIN 논문 재현이나 새로운 정규화 이론으로 주장하지 않는다.

[RevIN 공식 코드](https://github.com/ts-kim/RevIN/blob/fee40bc6c87cb536d048bcf1c14c4ed644b875e1/RevIN.py)를 실제 읽고 commit을 고정했다. 파일 SHA와 확인 범위는 [OFFICIAL_SOURCE_MANIFEST.json](OFFICIAL_SOURCE_MANIFEST.json)에 있다. 평균 중심화, biased variance, eps1e-5, detached mean/std, affine gamma1/beta0, inverse 분모 gamma+eps²를 따랐다. 공식 코드의 객체 내부 통계 저장을 이번 구현에서는 호출별 context로 바꿨다. 공식 소스 파일은 공개 payload에 복사하지 않는다.

B2는 프로젝트의 기존 `networks.ForecastAdapter`와 `fit_base.create_model`이다. 시간·채널 P → centered PCA로 초기화한 학습 E → frozen Chronos-Bolt → lag7 D와 원채널 마지막 값 anchor를 유지한다. ON/OFF의 P/E/D·PCA center·global scaler 초기값이 같다는 실제 대사는 [REUSE_AUDIT.json](REUSE_AUDIT.json)에 있다. Jena CUDA C02 PCA와 PeMS 개선 캠페인의 fresh CUDA PCA만 OFF로 재사용한다. 과거 CPU B03를 같은 B2라는 이름으로 대응 대조에 섞지 않는다.

Chronos-Bolt는 기존 local snapshot과 median quantile .5, FP32를 그대로 사용한다. F tensor-state SHA는 `c2b4edf7db22d3f18e58ca79299a5c7e864e9b8f277e4e29e2f88d07dbb35874`이며 실제 모델 로더가 이를 확인한다. 원 `gpu_runtime.FrozenBolt`와 `temporal_frontend_campaign_20261015_v1/runtime.py`의 zero-variance derivative 보정을 읽기 전용으로 재사용한다. F weights는 고정하고 입력 gradient는 유지한다.

기존 Raw·Ridge·DirectG·LoRA는 저장 prediction의 실제 byte SHA와 FP64 재집계로 연결한다. prediction-path fingerprint와 F tensor-state fingerprint를 다른 필드로 보존한다. Jena LoRA의 기존 CPU 예측과 CUDA replica를 동일한 숫자로 치환하지 않으며, GPU 표는 실제 CUDA replica와 검산된 배포를 사용한다. Ridge는 기존 NumPy FP64 CPU 경로이므로 GPU 표에 가짜 측정치를 넣지 않는다.

기존 그림과 K 한계 자료는 [HISTORICAL_ARTIFACTS.json](HISTORICAL_ARTIFACTS.json)의 실제 파일·SHA를 따라 재사용한다. 과거 여러 backbone/CPU→CUDA continuation 결과는 그 당시 조건의 배경 근거이며, 이번 새 RevIN의 matched 결과는 새 표·그림에서 확인한다. 추가 K·초기화 탐색이나 D seed2를 실행했다는 근거로 쓰지 않는다.
