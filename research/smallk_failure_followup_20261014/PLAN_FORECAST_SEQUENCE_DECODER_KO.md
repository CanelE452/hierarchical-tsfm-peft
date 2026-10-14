# 예측 잠재 시퀀스를 이용하는 디코더 비교 — 실행 전 계획

상태: **SOURCE_ONLY_NOT_EXECUTED**. 새 모듈과 CPU 합성 검산을 준비한다. 실제 데이터·체크포인트·Chronos-Bolt를 읽거나 학습하지 않는다. 현재 실행 중인 APP 실험과 기존 v1/v2 소스·결과는 보존한다. 이 계획은 자원 승인을 대신하지 않는다.

## 목적과 검증할 가설

[확인] 기존 E/D의 학습 목적을 Y, RAW teacher, MIXED로 바꾼 12개 실행에서 RAW teacher는 기존 MAIN 대비 Robin 약 0.85%, Beijing 약 0.67%의 평균 MSE 개선을 보였다. 압축하지 않은 RAW의 성능에는 도달하지 못했다. E/F를 고정한 디코더 재적합도 Robin을 개선하지 못했다. 수치 및 조건은 `validation_v1/JOINT_ARRAY_AUDIT_v1.json`, `validation_v1/D2_ARRAY_AUDIT_v1.json`에 남아 있다. 이 결과만으로 실패 원인은 확정되지 않았다.

[가설] 기존 D는 각 미래 시점의 K차원 잠재값을 독립적으로 복원한다. 예측된 잠재 궤적에 남은 시간 지연 정보를 같은 시점의 D가 이용하지 못할 가능성이 있다. 현재 E, K, frozen F를 유지하고 **RF1과 RF3인 작은 잔차 디코더**를 비교한다. 개선이 나오더라도, 원인의 전부를 디코더에 귀속하거나 RF3의 일반적 우월성을 주장하지 않는다. 정보가 압축 단계에서 이미 사라졌거나 F가 잠재 궤적을 잘 예측하지 못한다면 RF3도 복구할 수 없다.

## 입력과 구조

기존 구조를 \(z=E(X_{\le t}),\ \hat z=F(z),\ \hat Y_0=X_t+D(\hat z)-D(z_t)\)로 둔다. 기존 current-C 상태 anchor는 유지한다. 새 raw C 입력, 실제 미래 Y, 미래 마스크, teacher label을 예측 경로에 넣지 않는다. 모든 head 입력은 같은 E가 만든 과거 표현과 frozen F가 예측한 미래 표현이다.

STATIC_RF1은 \(R(\hat z_h)-R(z_t)\)를 기존 예측에 더한다. R은 `Linear(K,width) → GELU → Linear(width,C)`이다. TEMPORAL_RF3은 마지막 두 과거 표현과 예측 미래 표현을 이어 붙여 causal `Conv1d(K,16,kernel=3,padding=0) → GELU → Conv1d(16,C,kernel=1)`를 적용한다. 동일한 두 과거 표현과 `z_t`를 H번 반복한 경로에도 같은 R을 적용하고, 두 경로의 미래 부분 차이를 더한다. 결과는 B×H×C이며 H=48이다. 첫 출력은 과거 두 점과 첫 예측값, 이후 출력은 해당 시점과 앞 두 예측값만 참조한다. 과거 인코더의 시점별 인과성은 바꾸지 않는다.

두 head의 마지막 weight와 bias를 모두 0으로 초기화한다. 따라서 초기 새 residual은 정확히 0이고 기존 MAIN의 연산 순서 그대로 만든 출력에 정확히 0을 더한다. 미래 잠재값이 마지막 과거 잠재값과 같으면 학습 후에도 두 R 경로가 동일해 새 residual은 정확히 0이다. **기존** `x_last + D(zF) - D(z_last)`의 FP32 연산은 반올림으로 anchor와 bit-exact하지 않을 수 있다. 전체 persistence는 고정 atol=1e-5, rtol=2e-6으로 검사한다. 이 문제를 피하려고 기존 계산 순서나 임의의 equality branch를 추가하지 않는다.

등록 파라미터 수는 Robin C=17,K=4에서 STATIC width22=501, TEMPORAL width16=497; Beijing C=11,K=2에서 STATIC width21=305, TEMPORAL width16=299이다. 차이를 두 개수의 평균으로 나눈 값은 각각 약 0.80%, 1.99%이다. 이는 새 head의 **등록 파라미터 수**에 대한 약 2% 이내의 비교이며 정확히 같지는 않다. 마지막 출력 bias C개는 두 경로의 차에서 구조적으로 상쇄된다. 이 bias를 뺀 잠재적 active 수는 Robin 484/480, Beijing 294/288이며 Beijing의 active 차이는 약 2.06%이다. 미세한 FP32 bias gradient나 모든 bias의 갱신을 학습 성공 조건으로 사용하지 않는다. zero final weight 때문에 첫 update에서는 hidden gradient가 0일 수 있으며, output이 갱신된 뒤 hidden의 gradient를 검사한다.

## 고정한 8개 조건과 학습 계약

Robin와 Beijing Aotizhongxin 각각 old MAIN seed 92601/92602에서 STATIC_RF1, TEMPORAL_RF3를 비교한다. 총 2×2×2=8개 **신규** 학습이며, 중단한 옛 실험을 재개하지 않는다. 각 dataset/seed의 두 head는 같은 기존 selected MAIN E/D 체크포인트를 복제하고 optimizer는 새로 만든다. 기존 MAIN 자체가 이미 actual-Y TUNE으로 선택되었다는 사실을 기록한다.

모든 조건에서 E/D와 새 head만 갱신한다. F는 기존 Chronos-Bolt-small이며 weights는 고정하고 입력 gradient는 유지한다. 600회 신규 update, batch4, `default_rng(seed).integers(0,Nfit,(600,4))`의 같은 순서, AdamW lr=.001, weight_decay=0, clip_norm=1을 쓴다. checkpoint 0/100/200/400/600을 저장한다. 학습 손실은 채널별 관측 count로 나눈 normalized actual-Y MSE의 활성 채널 평균이다. 누락값의 residual을 제곱 전에 0으로 처리하며 관측값이 없는 batch는 실패로 보존한다. checkpoint 선택은 두 head 모두 **actual-Y TUNE MSE**, exact tie는 이른 checkpoint이다. update0도 후보이므로 나쁜 학습을 강제로 선택하지 않는다. selected0은 기존 출력 유지이며 학습의 성능 개선으로 쓰지 않는다.

데이터는 기존 bridge의 같은 FIT/TUNE/이미 공개된 평가 subset, origin, mask, FIT mean/std를 재사용한다. Robin FIT256/TUNE64/VAL30, Beijing FIT256/TUNE64/EVAL146, L512/H48이다. 보호 TEST를 열거나 신규 독립 평가를 만들지 않는다. 기존 native/RAW/MAIN, joint Y/RAW/MIXED 결과를 같은 노출 subset의 참조로 보존한다. 새 평가 결과로 head·rank·폭·학습률을 선택하지 않는다. 앞선 반복 연구가 이 subset을 이미 노출시켰다는 한계를 보고한다.

## 실행 시 필수 저장과 검산

각 fit은 새 output/tag에 source/lock snapshot, original CP file/codec hash, F before/after의 old/Bi state hash, 통계·origins·배치 순서, warmstart 모든 split의 예측/target/mask, 600회 curve 및 각 checkpoint를 저장한다. checkpoint별 FIT/TUNE prediction과 점수, selected checkpoint와 최종 모든 split 예측을 저장한다. u600 상태와 selected 상태를 분리한다. E/D/head 별 실제 parameter change, frozen F parameter grad=None, F branch만의 input gradient를 검사한다. 기존 anchor shortcut을 F input gradient로 잘못 계산하지 않는다. 초기와 selected0의 기존 MAIN parity는 실행 전에 고정한 atol=1e-5/rtol=2e-6으로 검사한다.

독립 CPU 검산은 저장된 배열의 masked normalized MSE/MAE 및 채널별 count, 물리 단위의 채널별 오차, earliest TUNE selection, 원점·통계·CP/source SHA를 재계산한다. 저장된 latent와 head weights가 있다면 residual 경로를 CPU로 재구성한다. 저장하지 않은 매 update forward와 실제 F 재추론을 독립 검산했다고 표현하지 않는다. MSE 개선과 MAE 악화, 큰 오차와 특정 채널 악화도 함께 남긴다.

비용은 원 RAW/MAIN과 신규 선택 head의 batch1/batch8 inference를 같은 입력·fill·timing 조건에서 비교한다. GPU 동기화 후 전체 경로와 E/F/기존D/head 단계를 측정하고 output parity를 확인한다. 모델 load, 입력 준비, 학습, TUNE/선택, 검산, guard 비용을 구분한다. GPU process wall에는 CPU guard도 들어가므로 이를 GPU kernel 시간이라고 부르지 않는다.

## 자원과 실행 gate

이 파일과 `forecast_sequence_decoder.py --describe`는 계획과 CPU 합성 검산만 제공하며 실제 실행 기능이 없다. 기존 GPU 큐를 건드리지 않는다. 신규 실행은 기존 공유 ledger를 확인한 다음 진행해야 한다. 현재 campaign wall 종료 또는 잔여 cap 소진 시에는 **추가 자원 승인 없이 시작하지 않는다**. 기존 cap을 재설정하거나 연장하지 않는다.

검토용 예상 상한은 8 fits×각 GPU process wall150초=1,200초, paired 비용 측정 GPU process wall180초, 독립 CPU 검산300초이다. 모델 로딩·guard 포함이며 실제 비용은 미확정이다. 최소 약 25–35분의 새 campaign wall과 보고 작업 시간이 필요할 수 있다. 더 느려도 fit cap을 자동 증가시키지 않고 실패·최종 update·CP·재개 방법을 남긴다. 이후 concrete driver를 만들 때 공유 CPU/storage/RSS/VRAM 및 campaign cap을 매 batch/base와 10 update·checkpoint·IO/full 경계에서 강제하고, terminal guard 실패를 COMPLETE로 바꾸지 않는다. 원래 중단 checkpoint를 이어 학습하지 않는다.

## 합성 검산의 의미와 다음 판단

합성 예시는 독립 혁신 \(u_h\)에서 두 채널이 \((u_h,u_{h-1})\)일 때 RF3가 해당 함수를 표현할 수 있고, 동일한 현재 값과 다른 직전 값을 받은 pointwise R은 두 target을 동시에 맞출 수 없음을 보인다. GELU(u)−GELU(−u)=u를 이용해 명시적 weights로 구성하며 모델을 실제 자료에 학습한 결과가 아니다. 실제 E가 현재 z에 지연 정보를 이미 담거나 F가 미래 지연 패턴을 보존하지 않으면 이 예시는 현실 결과를 예측하지 않는다.

두 head의 평가 차이가 작거나 RF3도 RAW를 넘지 못하면 단순 pointwise 복원이 유일한 원인이라는 가설은 지지되지 않는다. RF3가 개선되더라도 작은 head capacity 차이, stochastic optimization, 압축 E의 동시 갱신, 이미 노출된 subset, seed2라는 제한을 남긴다. 구조적 개선과 단순 추가 capacity의 효과를 완전히 분리하지 못하면 다음 비교를 별도로 설계한다. 성능, 비용, 실패를 함께 보고한 뒤 더 큰 decoder·대조학습·동적 graph가 필요한지 판단한다. 이번 source 준비 자체를 real-data 개선 또는 실험 완료로 말하지 않는다.
