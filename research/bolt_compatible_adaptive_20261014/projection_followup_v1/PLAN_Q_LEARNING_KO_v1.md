# 같은 E/D에서 6개 회전 계수만 예측 목적으로 학습

고정 Q15조건의 실제 실행 뒤 작성한 후속 사전 계획이다. 원 RESULT/고정Q/부모 SOURCE는 수정하지 않는다. latest 사용자 방법론 완성·계속 실행 지시 범위이며 추가 승인 대기가 아니다. TEST/설치/다운로드/큰 백본 학습/기존 작업 종료/자동 Git 금지는 유지한다.

## 질문과 반대 근거

고정 Q의 항등·순열은 세 상태와 세 split에서 기존 baseline과 실제 byte 차이0이었다. sign/dense는 latent와 C 예측을 크게 바꿨고 모든 사전 고정 sign/dense9조건이 TUNE와 DEV에서 원 baseline보다 나빴다. 아직 이 기록 시점 독립 저장 산술 검산은 준비 중이며 원 실행 관측과 분리한다. 임의 회전의 실패로 모든 Q 학습을 생략하지 않는다.

질문은 **정보·복원 공간을 고정한 작은 학습 Q가 arbitrary Q와 달리 FIT/TUNE 잔차를 줄이고 DEV에 옮겨지는가**다. Q는 기존 D의 수치 공간 제약을 바꾸지 않으므로 현재 TUNE minimum>Raw라는 반대 근거를 유지한다. AdaPTS/ORTCL의 제약된 adapter 범주이며 회전 자체의 신규성은 주장하지 않는다.

## 사전 고정한 6학습 조건

Robin의 E0 RECON_CP0/E1 E_ONLY_s92601/E2 E_ONLY_s92602 세 frozen E/D×minibatch 순서 seed92601/92602다. 같은 zero θ에서 시작하므로 seed는 초기 표현이나 자료의 독립 반복이 아니다.

```text
Q = exp(S(theta)), S^T = -S, theta ∈ R^6
z = (Xpast-center)E
u = F(zQ)Q^T
Yhat = old_lag_decode(currentC,z,u;D)
```

E/D/center와 F 가중치는 동결하고 θ6개만 trainable이다. 학습 θ=0의 Q 경로는 미분을 위해 우회하지 않는다. 비교용 무연산 fixed-identity와 구분하고 CP0의 실제 C/future를 같은 canonical32 baseline에 사전 공차 atol1e-5/rtol2e-6으로 검산한다. D를 Q로 재저장하지 않고 old known-past/lag별 원점/current C를 그대로 쓴다. 모델의9분위 중0.5/48시점 반환 규칙도 이번에는 유지한다.

L512/H48, FIT256/TUNE64/기노출 DEV30·원 scaler/origin/mask를 그대로 사용한다. FIT 미래는 정상 supervised response이고 forward 입력은 과거뿐이다. 부모 fixed-controls의 독립 과학 감사가 PASS한 뒤 실제 학습하며 SOURCE/합성 준비는 먼저 가능하다.

## 학습·선정·검산

- AdamW θ만, lr=.001/weight_decay0, B4, gradient clip norm1, 각600update다. 기존 학습과 같은 global FIT observed-channel-macro 가중 actual-C MSE를 쓴다. FIT 관측수를 TUNE/DEV 평가 분모로 재사용하지 않는다.
- CP0/100/200/400/600을 모두 저장하고 TUNE MSE 최소를 선택한다. 정확 동률은 이른 CP다. CP0도 정상 후보이며 이후 학습을 무조건 선택하지 않는다.
- 모든6조건의 성공/실패 잠금·registry 뒤 DEV를 연다. DEV에서 CP나 seed를 고르지 않는다. 해당 경로 실패는 보존하고 독립 학습 조건을 계속한다.
- actual F gradient가 θ의 입력 경로에 도달하는지 첫 minibatch에서 검사한다. 전체 θ gradient, rotated F-input을 경유한 θ gradient, inverse-Q 경로 차이의 norm·finite·closure를 기록한다. 이는 모든 step의 독립 역전파 재생은 아니다.
- 매step θ gradient/clip·loss·동결 flag/F.grad None을 기록하고 CP마다 E/D/center byte 불변을 확인한다. F dual hash·가중치 파일 SHA는 전후 비교한다. 이를 매step 가중치 byte 검산으로 확대하지 않는다.
- canonical32 actual future와 C를 저장하고 독립 NumPy inverse-Q/old lag decode로 검산한다. Q orthogonality·determinant·θ 크기와 같은 E/D baseline/Raw MSE·MAE를 함께 기록한다. 합성 finite-difference/known linear null은 실제 F의 global optimum을 보장하지 않는다.

기존3상태의 학습 이력은 다르다. E1/E2는 이미 FIT 미래 감독을 받았으며 이번에 그 이력이 사라지는 것은 아니다. 학습 시간과 새 source/학습 로그/선정/실패를 보존하며 새 전체 추론 비용 측정은 이 학습 자체에 포함하지 않는다. 연구 성공이나 독립 일반화는 아직 미확정이다.

6조건이 TUNE에서 CP0를 택하거나 이득이 DEV로 옮겨지지 않으면 동일 회전 family의 seed/LR sweep을 늘리지 않는다. 큰 K·예측 가능한 방향·시점별 decoder·점요약 진단은 서로 다른 개입으로 보고, Q의 실패만으로 독립 후보를 막지 않는다.
