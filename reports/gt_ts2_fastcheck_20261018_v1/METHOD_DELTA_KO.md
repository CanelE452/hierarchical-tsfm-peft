# 원 방법에서 가져온 부분과 이번에 새로 검증한 부분

[Graph-Time Convolutional Autoencoders, LoG 2022](https://proceedings.mlr.press/v198/sabbaqi22a.html)는 product graph 위의 시공간 convolution, encoder의 시간 downsampling과 decoder의 upsampling을 제안한다. 원 논문은 네트워크 시계열의 비지도 표현과 denoising·anomaly detection을 검증했다. 이번 저장소의 GT는 논문 기반 재구현이며, 기존 reconstruction+sparsity 사전학습 부모를 그대로 출발점으로 쓴다.

[TS2Vec, AAAI 2022](https://ojs.aaai.org/index.php/AAAI/article/view/20881)는 증강한 context view의 계층적 대조학습으로 시점별 문맥 표현을 학습한다. 이번에는 기존 Jena·같은 seed의 canonical d320 encoder를 고정하여 알려진 과거 전체에서 offline 표현을 추출한다. 이 경로를 내부 시점별 prefix-causal이라고 부르지 않는다. 예측 원점 이후의 실제 미래는 encoder에 넣지 않는다.

이번 신규 설계는 GT latent와 4시점 평균 TS2 표현을 정렬하고, TRAIN에서만 고정한 feature 통계로 표준화한 뒤 bias 없는 W(320→42)를 거쳐 residual로 더하는 것이다. W는 0에서 시작한다. 4 arm은 기존 GT E/D·공유 product 계수와 필요한 W만 forecast-only MSE로 짧게 적응한다. TS2·F·공간 인접 행렬은 고정한다. RAND는 같은 구조·용량의 무학습 encoder 대조이며 새 대조학습을 하지 않는다.

[AdaPTS, ICML 2025](https://proceedings.mlr.press/v267/benechehab25a.html)의 latent feature-space adapter 뒤에 단변량 foundation model을 적용하는 관점과 연결된다. 그러나 이 논문이 이번 GT/TS2 residual bridge의 효용을 검증했다는 뜻은 아니다.

긴 lookback의 고정 추론, 예측 목적의 공동 추가학습, GT/TS2 잠재 결합, 0.5 출력 평균은 이번 별도 비교다. 신규 GT reconstruction 사전학습이나 GTConvAE/TS2Vec 원 논문 benchmark의 재현으로 표시하지 않는다. GT는 시간축을 L→L/4로 4배 축소하고 feature가 2개라 scalar 수는 LC→LC/2로 2배 압축한다. 원채널 C=21과 F 시계열 42개를 유지하므로 B2의 채널 축소와 구분한다. 300 고정 updates 완료는 수렴 증명이 아니며, CP0·100·300의 고정 TRAIN 진단은 선택에 사용하지 않는 기술 통계다. 성능 미개선도 이 부모·데이터·고정 예산의 결과이며 원 논문 전체나 모든 결합 설계를 기각하지 않는다.
