# 새 방향·용량 실험에서 F의 가치와 복원 병목 비교

앞선 [cross-moment 계획](PLAN_PREDICTIVE_CROSS_MOMENT_KO_v1.md)의 예정된 native/ridge/비용 단계다. 최신 같은 목적의 지속 지시 안에서 진행하며 승인 질문을 추가하지 않는다. 기존 금지와 원본 보존을 유지한다. 새 표현8개의 독립 과학 검산 뒤 실제 학습하며 SOURCE·합성 준비는 먼저 한다.

## 같은 E/D의 native16학습

Robin PCA/CROSS K4/K8, Beijing PCA/CROSS K2/K4의8개 frozen E/D를 그대로 쓴다. 기존 native_head_v1.py의 **stream별 affine512→48** head와 NativeLagForecaster를 사용해 seed92601/92602의16학습을 새 tag에서 수행한다. E/D/center는 동결하고 head만 갱신한다. 같은 FIT256/TUNE64·과거512·current-C anchor·old known-past/lags·actual-C observed-channel-macro MSE를 유지한다. AdamW lr.001/WD0/B4/clip1/600update, CP0/100/200/400/600와 정확 동률 이른 CP를 유지한다. seed는 같은 자료의 순서/초기화 조건이지 독립 데이터셋이 아니다.

같은 E/D 안의 F는 앞서 실제 실행한 cache를 재사용한다. 새 native 추론은 실제 수행하고 새 미래 latent·C 출력과 NumPy oldlag decode/원 mask 점수를 저장한다. 모든16개 성공/실패 선정 잠금 뒤 DEV를 연다. stream별 선형 head의 실패를 모든 native 최적화나 채널 혼합 예측기의 불가능성으로 확대하지 않는다.

## 직접 ridge로 정보와 D 제약 분리

각 dataset에 Raw CURRENT_C와 Raw HISTORY512 두 대조를, 각8 E에 ENCODED_HISTORY512 직접 C readout을 둔다. 총12개 fit이며 alpha={0,1e-4,.001,.01,.1,1,10}의84개 후보다. 별도 seed 반복이나 alpha 확장은 없다. 원 C actual-Y MSE/관측 채널 균등 지표로 own TUNE alpha 하나를 고르고 정확 동률은 사전 alpha 순서다. 모든12개 잠금 뒤 DEV를 연다.

- Raw CURRENT_C는 원 current-C의 C차원 특징으로 actual 미래 level을 예측한다. horizon별 complete-C 응답 origin을 공통 선택하고 같은 horizon 안의 C 응답을 동시에 적합한다. horizon별 FIT subset에서 mean/RMS를 구하고 intercept는 penalty에서 제외한다. 각 h의 목적은 SSE/n_h+alpha||B||²이다. complete mask에 따른 표본수 차이를 기록한다.
- Raw HISTORY512는 vec(history-currentC)의512C 특징에서 미래 innovation(Y-currentC)을 예측하고 마지막에 currentC를 더한다. 각(h,c)의 true-observed FIT 응답만 사용하며 미래 dummy를 label로 넣지 않는다. FIT256 전체의 특징 mean/RMS를 한 번 고정하고 intercept는 penalty에서 제외한다. 목적은 SSE/n_hc+alpha||beta||²이다.
- ENCODED_HISTORY512는 같은 state의 실제 canonical Torch32 과거 z에서 vec(z-z_last)의512K 특징을 사용한다. 응답/anchor/intercept·alpha·mask 규칙은 Raw HISTORY512와 같다. **D를 우회하므로 same-D native가 아니며 출력 공간/파라미터 수가 더 크다.** 이 경로가 좋아져도 latent future 예측이나 frozen F의 성공을 뜻하지 않는다. current-C anchor는 기존 방법과 같은 원 C 정보다.

feature RMS가0이면 scale1/해당 계수0으로 둔다. alpha0의 normalized design minimum-norm cutoff는1e-8이고 양수alpha는 ridge의 모든 singular 방향을 사용한다. 기본 해법은 각 관측 subset의 centered normalized design에 대한 FP64 thin SVD이며 같은 mask의 응답 열과7alpha에 분해를 재사용한다. design singular cutoff1e-8을 Gram eigen cutoff1e-8로 바꾸지 않는다. Gram을 제곱하면 임계값1e-16이 FP64 roundoff 부근이므로 alpha0를 임의의 Gram 수치 절단으로 대체하지 않는다. FP64 해와 FP32 배포를 분리한다. 선정은 실제 FP32 prediction에서 점수를 내며 직접 NumPy FP64 목적 해/불변/상수열/관측 subset·alpha0/큰alpha를 합성 검산한다.

원 C와 압축 특징의 입력 정보량·모수·추론 비용이 다르므로 capacity-matched라고 부르지 않는다. 같은 새 E의 입력 정보가 쓸 만한지, 같은 D의 병목이 있는지, 단순 직접 예측이 F보다 충분한지를 나누어 판단한다. 원 C HISTORY512의 이전 동일조건 결과가 exact source/feature/scaler/alpha/mask/FP32 계약으로 일치하면 그 lineage를 확인해 재사용 가능하나, 이름만 같은 과거 결과를 같은 조건으로 간주하지 않는다.

실제 전에 배포 산술도 고정한다. FP64 standardized coefficient를 원 feature 단위로 나눈 coefficient와 intercept에 합성한 뒤 FP32로 저장한다. 배포는 `feature32 @ coefficient32 + intercept32 + anchor32`다. 따라서 매 배포마다 별도 mean/RMS 나눗셈을 하는 구현과 FP32 byte 동등하다고 주장하지 않는다. FIT mean/RMS·FP64 objective와 합성 전후 계수를 보존하고 실제 FP32 출력을 TUNE로 선정한다. FIT 상태가 고정된 뒤 미래 target/mask는 forward 함수에 전달하지 않는다.

## 비용과 결론

새 선정 native/ridge와 F는 physical CPU input→원 window-local causal fill/FIT scaler→E(해당 경로)→실제 F/native/ridge→old D(해당 경로)→physical CPU 반환 전체를 같은 B1/B8 cohort에서 측정한다. 캐시 F를 반환하는 것은 배포 비용이 아니다. 사전 2warm/5실측 및 회전 순서·integer ns·shape/finite·first8 reference를 유지하며 CPU/GPU/RSS/sample 한계를 명시한다. 새 비용 SOURCE/검증과 결과 tag는 별도로 보존한다.

이 단계 뒤에도 별도 TEST/독립 평가가 없고, 현재 DEV는 적응적 탐색 자료다. 성공한 몇 줄만 최종 방법으로 제시하지 않으며 K별Raw대비정확도·전체비용·단순대안·실패경로를 함께 보고한다. 추가 대조학습/동적그래프는 단순 대조에서 남는 구체적 문제를 해결할 때만 근거가 있다.
