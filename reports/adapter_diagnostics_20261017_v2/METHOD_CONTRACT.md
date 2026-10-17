# 고정 방법·변경 요인

B2: Z=E(P(X)); U=frozen_F(Z); Yhat=X_last+D(phi_lag(Z,U)). P는 temporal/channel residual MLP, E는 PCA prefix 초기화 학습행렬, D는 lag-major [7,K,C], lags=(0,1,2,4,8,16,32)다. D[0]=pinv(E), 나머지는0이다. h별 phi가 다르고 D는 공유한다. 원채널 미래 관측 mask를 반영한 정규화 macro MSE로 P/E/D를 공동 학습하며 F frozen과 input autograd를 함께 유지한다.

A_fit/A_stop/M_cal/V/T=50/10/10/10/20. 학습 stride1, 평가 strideH. A_stop 선택 뒤 A60 fresh refit하고 scaler/PCA도 새로 구한다. M_cal/T 선택 사용 금지. CP0/.25/1/3 endpoint 유지. SAME_UPDATE_COMPARISONS는 A_fit search 가중치의 같은 epoch/updates/exposures/split/scale 대조이며 A_stop 선택·selected fresh-A60 refit과 구분한다.

B01은 선택 A60 PE 고정 full [512,6] history, lambda1 고정, F/D 호출0. B02는 own32block 대 all-channel32block, 같은 alpha grid(.01,.1,1,10,100), SSE_c/N_c+alpha L2, intercept 비규제다. B03은 stage별 PE/cache·common RMS를 공유하고 shared penalty lambda/C, horizon penalty lambda/(C×H)를 맞춘다. A60 PE로 A_stop 선택 금지. C02는 PCA/QR만 변경한 matched fresh fit 쌍이다.

B03 parameter 각주: trainable_params는 D coefficient block 수(shared/원 D=882, horizon-specific=42336)다. 이번 진단에서 실제 적합한 fit_stage_params는 원 D=0, 재적합 shared=882, horizon-specific=42336이다. 앞서 target-trained된 retained P/E 34835개를 포함한 deployed_trainable_params는 원/shared=35717, horizon-specific=77171이다. 따라서 882개짜리 전체 모델로 해석하지 않는다. frozen backbone resident parameter 수는 다시 별도다.

Point output은 source-bound이며 Bolt/새 quantile 모델의 median0.5와 Ridge deterministic regression을 구분한다. Raw/C→K stream rows, 파라미터, 메모리, E2E latency는 별도 지표다.
