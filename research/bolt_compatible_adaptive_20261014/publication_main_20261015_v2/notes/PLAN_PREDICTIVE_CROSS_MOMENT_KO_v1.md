> 게시 사본입니다. 원문 작성 당시의 로컬·준비·미실행 상태는 역사로 보존합니다. 최신 완료/미해결 상태와 게시 범위는 [색인](../README_KO.md)을 확인하세요. 과학 원본은 바꾸지 않고 링크와 이 안내만 추가했습니다.

# 예측 관계로 고른 입력 방향과 작은 K 용량의 분리

최신 사용자 지시의 동일 연구 목적 안에서 진행하는 사전 계획이다. 추가 승인 대기가 아니다. 기존 TEST·설치/다운로드·큰 백본 학습·기존 작업 종료·자동 Git 금지는 유지한다. 원 파일은 보존하며 새 tag에 실행한다.

현재 고정 K4 Robin의 세 상태는 같은 D의 TUNE 수치 공간 최소 오차조차 Raw보다 높다. Q는 그 공간을 바꾸지 못한다. 따라서 같은 공간의 좌표 보정만 반복하지 않고 **입력 방향의 선택 목적과 K 용량**을 바꾸는 독립 대조를 준비한다. 이미 수행한 forecast-supervised E/D·FIT-target 방향 선택을 최초 예측 목적 학습이었던 것처럼 다시 소개하지 않는다.

## 사전 고정 비교

Robin C17의 K4/K8, Beijing C11의 K2/K4에서 PCA와 CROSS_MOMENT 두 방향 규칙을 비교한다. 총8 E/D이며 K가 다른 결과를 압축률이 같은 대조로 합치지 않는다. PCA는 FIT 과거 분산, CROSS_MOMENT는 FIT 현재 C와 미래48×C의 centered cross-covariance를 이용한다. 새로운 현재/지연 mask 또는 alpha sweep은 없다. 기존 [자료 계약](PREDICTABLE_PROJECTION_DATA_CONTRACT_KO_v1.md)의 (a)를 사용한다.

FIT history의 mean/covariance에서 relative eigenvalue cut1e-8의 whitening W를 만든다. 각 h에서 미래 C 전체가 실제 관측된 origin 집합 I_h만 쓰고 source current와 actual future level의 mean을 각각 빼 C_h를 만든다. T=[W^T C_0|...|W^T C_47]/sqrt(48)의 SVD로 상위 K 왼쪽 방향을 택해 E=W U_K로 둔다. 최종 E의 각 열에서 최대 절댓값 좌표를 양수로 고정한다. 미래 결측값을 0 응답으로 쓰지 않으며, 이를 exact masked RRR 또는 미관측 모집단의 unbiased covariance로 부르지 않는다.

같은 D 규칙을 모든 E에 다시 적용한다. 실제 배포와 같은 FP32 E/center로 FIT 과거를 encode한 뒤 L7 RECON의 공통[32,512) slice/lag0·1·2·4·8·16·32/관측 채널별 centered LS/rcond1e-4로 D를 적합한다. 복원 intercept는 delta/current-C anchor에서 상쇄한다. 새 E에 이전 D를 그대로 붙이지 않는다.

## 먼저 실제 실행할 단계

1. FIT-only E/D8개를 적합하고 moment rank/gap·mask별 origin수·고유 timestamp/중복·source fill과 latent covariance를 기록한다.
2. 같은 frozen Bolt median·canonical32를 사용해 FIT/TUNE에서 실제 새 F를 실행한다. PCA small-K는 기존 같은 규칙 결과와 원계수/FP32 입력·예측 parity를 확인한다. source 구현이 달라 동등하지 않으면 해당 차이를 남기며 캐시를 대신 반환하지 않는다.
3. flat latent·actual-future-encoded latent를 별도 계산한다. actual future가 필요한 oracle는 **원인 진단**이며 배포 입력이 아니다. 다른 채널 결측의 dummy를 통해 oracle를 유리하게 만들지 않도록 complete-C subset oracle와 all-observed scores를 구분한다. 원 finite numeric output-space minimum은 별도 진단이며 예측 가능한 성능이 아니다.
4.8개 상태/고정 규칙과 FIT/TUNE 결과를 잠근 뒤 기노출 DEV30/146을 연다. 같은8개를 모두 보고하며 DEV로 K/representation을 재선정하지 않는다. Raw는 원 mask/cohort의 실제 cache를 비교하고 이 단계에서 새 Raw 추론을 했다고 쓰지 않는다.

이 단계만으로 F의 고유 가치·전체 배포 비용·교수님께 제시할 최종 방법이 검증됐다고 하지 않는다. **모든8상태의 결과와 실패를 보존한 뒤 같은 E/D의 native predictor 및 원 C current/history ridge와 전체 비용을 이어서 비교한다.** 다른 E/D의 이전 native를 새 native로 재사용하지 않는다. 후속 native는 같은 stream별512→48 head/actual-C MSE600update/두 순서 seed를 사전 기본으로 유지하되 실제 source와 선정 잠금을 별도 보존한다.

## 검증과 해석

순수 함수는 미래 placeholder 변화 불변·horizon complete mask·rank 부족 실패·독립 loop cross moment·whitening·고유 부호·FIT 외 입력 배제·delayed-information 반례를 합성으로 검사한다. 실제 forward는 과거512×C와 고정 state만 받는다. 실제 E/D/F와 원 input binding·model dual hash·자원 receipt·독립 NumPy decode/score 검산을 남긴다.

cross-moment가 미래 Y 정보를 보존해도 해당 K stream 자신의 미래를 univariate F가 잘 예측한다는 보장은 없다. x2(t+1)=x1(t), x1 white noise 반례를 유지한다. K 증가의 개선은 비용 증가와 함께 보고하며, Q/contrastive/dynamic graph가 필요하다는 결론으로 바꾸지 않는다. DEV는 이미 반복 노출된 탐색 자료이고 독립 일반화 검증이 아니다.
