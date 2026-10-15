# 방법론 완성의 판단과 다음 분리 실험

## 목적과 실행 권한

최신 사용자 지시: “교수님한테 방법론 말할수있을정도로 방법론 완성시켜줘 오늘밤에는 멈추지마 뭐 허용관련 메시지도 뜨게 하지 말고”. 같은 연구 목적의 설계·구현·로컬 실행·검산·연구 메모 작성은 별도 하위 승인 없이 진행한다. 성공이나 학술적 신규성이 보장됐다는 의미는 아니다. 원 TEST 금지·원본/기존 작업 보존·설치/다운로드 금지·대형 백본 재학습 금지·자동 commit/push 금지는 유지한다. 종전 시간·누적 자원 상한을 종료 기준으로 재도입하지 않는다.

최상위 목적은 **원 채널 frozen TSFM을 쓰는 것보다 작은 K 경로가 필요한 이유와 작동 원리를 설명하고, 원수치·동일 조건 대조·실제 비용으로 채택/기각을 판단할 방법론**이다. 교수님은 압축 정보량, 예측 목표, TSFM의 필요성, 복원 제약, 비용 이득을 각각 물을 수 있어야 한다. 논문/PPT 대신 재현 가능한 코드와 한국어 방법론 메모·그림을 남긴다.

## 이미 확인한 반대 근거

- 재구성만 학습한 상태가 아니다. 이전 실제 Y/Raw-F teacher·공동 E/D·E-only 학습과 같은 L7 native 비교를 반복해 정확도 격차가 남았다.
- 현재 작은 q 보정은6조건 중5 identity, 나머지1개의 TUNE 이득 약.0114%/DEV 악화 약.0412%였다. q 전체 비용20셀100표본까지 실제 완료했고 독립 비용 감사 PASS도 저장했다. 통합 보고는 아직 준비 중이다.
- 현재 L7/K4 세 고정 공간의 TUNE retained numeric minimum은 모두 Raw보다 높고 DEV에서는 낮다. 등록된 수치 공간에 조건부인 진단이며 모든 실수/FP32 출력의 보편 하한이나 과거 예측 가능성 증명은 아니다. 현재 공간 안의 보정만으로 연구 전체 해결을 기대하지 않는다.
- 입력/출력 feature adapter는 [AdaPTS(ICML2025)](https://proceedings.mlr.press/v267/benechehab25a.html)에 이미 있고 orthogonal rotation도 [ORTCL(AAAI2026)](https://ojs.aaai.org/index.php/AAAI/article/view/39526)에 있다. 회전·압축·frozen F 연결 자체를 새 기여로 주장하지 않는다. 구체적 동등성과 적용 축은 후속 원문 검토에 남긴다.

## 후보 A: 정보·복원 공간을 그대로 두고 F 좌표 호환성을 분리

같은 원 E/D/center와 known-past latent, current C anchor를 고정하고 다음만 바꾼다.

```text
z_past = (X_past - center) E
u_Q    = F(z_past Q) Q^T
Y_hat  = old_lag_decoder(current_C, z_past, u_Q; old_D)
Q^T Q  = I
```

이 행동의 목적은 (1) 새로운 정보 없이 좌표 선택만으로 F가 달라지는지 확인하고 (2) E/D의 자유로운 학습에서 발생한 출력 공간 변화와 작은 F 호환 적합을 분리하는 것이다. reconstruction을 Q와 Q^T로 상쇄하는 것과 F의 비선형 연산을 통과한 뒤 상쇄되는 것은 다르다.

처음에는 Robin C17/K4 세 상태 E0/E1/E2, 원 FIT256/TUNE64/기노출 DEV30·L512/H48·scaler/origin/mask·원 D32·lags0/1/2/4/8/16/32를 유지한다. 사전 고정 I·채널 permutation·sign control·seed92711/92712의 dense orthogonal Q를 전부 보고하며 DEV로 Q를 골라 성능 순위를 만들지 않는다. Oct11의 K5/6 및 Q별 D-refit 실험과 구분한다.

analytic transport는 저장 canonical u를 Q로 옮겼다가 원공간으로 되돌리는 검산으로 먼저 확인한다. 실제 F(zQ)는 새로 실행한다. 이때 oldD와 oldpast를 그대로 쓰며 D_Q 재직렬화 오차를 섞지 않는다. 독립 FP64/FP32 식과 permutation/identity에 대한 모델·batch 경로를 검사하고 실제 결과를 본 뒤 tolerance를 완화하지 않는다. sign equivariance는 보장하지 않는 대조다.

고정 Q의 F 변화가 확인되면 Q=exp(skew(theta))의 FIT actual-C 손실 학습을 이어간다. K4의 trainable theta는6개이며 E/D/F는 동결한다. identity에서도 학습 경로는 미분을 유지하고, 비교용 무연산 identity와 구분한다. CP0와 사전 checkpoint를 TUNE에서 고르고 모든 선택을 잠근 뒤 DEV를 평가한다. 차이가 없더라도 후보 B를 막는 gate로 쓰지 않는다. 고정 회전·학습 회전의 실행은 따로 기록한다.

예상되는 반례는 Q에 따라 F는 크게 달라도 TUNE 선정 이득이 DEV로 옮겨지지 않는 경우다. 이것은 좌표 민감도의 증거이지 방법 성공이 아니다. 교수님의 첫 질문 “그래서 같은 정보에서 더 잘 예측하나?”에 원수치와 비용으로 답한다.

## 후보 B: 분산 대신 예측 가능한 성분을 선택

미래 Y의 에너지만 보는 기존 FIT-target PCA와 구분해, FIT 과거→미래 cross-predictive covariance/RRR 계열로 E를 찾는 후보를 조사한다. 목적은 (1) 압축 시 버리는 성분의 기준을 실제 예측과 연결하고 (2) 예측에 필요한 정보가 있는지와 F가 그 정보를 이용하는지를 분리하는 것이다. 이는 기존 통계 기법이며 이름만 바꿔 신규 방법으로 소개하지 않는다.

원 FIT-only causal fill/scaler/observed mask·정확한 응답/시간창·중복 가중치·regularization과 K를 사전에 정해야 한다. 실제 deployment는 과거만 사용하고 FIT 미래는 학습 응답으로만 쓴다. 같은 E/D에서 native와 frozen F를 비교하며 원 C의 같은 ridge 대조를 포함한다. PCA/D 선택 또는 큰 K로 바꾸면 정보·공간·비용이 함께 바뀌므로 A의 순수 Q 대조와 합치지 않는다. 코드/자산 조사에서 이미 같은 개입이 있으면 재발명하지 않는다.

예상 반례는 예측 목적 E에서 native만 좋아지고 F는 나빠지는 경우다. 이때 “TSFM compatible”을 주장하지 않으며 direct reduced-rank predictor의 강한 대조를 남긴다. 반대로 두 예측기가 모두 좋아지면 표현 이득과 F 고유 이득을 따로 계산한다.

## 독립 후보와 실행 순서

작은 horizon-dependent D는 동일 계수 수의 정적 D 보정과 비교할 때만 추가 후보로 둔다. 복잡한 동적 그래프·대조학습을 자동으로 붙이지 않는다. 단순한 후보가 왜 부족한지 실제 대조가 먼저다.

1. 현재 q의 비용 감사·통합 MD/CSV/그림을 마무리한다. 이 동안 다음 SOURCE/합성 준비와 원문·자산 조사만 병행한다.
2. A의 고정 Q/transport/실제 F 비교를 실행·검산한다.
3. A의 작은 Q 학습 및 B의 독립 비교를 판단한다. K4/K8 등의 capacity 대조가 필요하면 동일 원자료와 K별 PCA/D refit·전체 비용을 따로 등록한다.
4. 방법론 메모에는 문제·핵심 식·trainable/frozen·학습과 배포 입력·기존 방법과의 차이·반례·실제 비용·미완료를 제시한다. 개선이 없으면 완성된 성공 방법이라고 쓰지 않는다.

주지표는 관측 채널 macro 정규화 MSE, supplementary는 MAE·원단위 per-channel 진단·동일 cohort B1/B8 전체 경로 비용이다. 반복 노출 DEV에서의 선택적 탐색이며 독립 일반화/TEST 성능을 주장하지 않는다. 임의의 허용 손실률을 결과를 본 뒤 성공 기준으로 만들지 않는다.
