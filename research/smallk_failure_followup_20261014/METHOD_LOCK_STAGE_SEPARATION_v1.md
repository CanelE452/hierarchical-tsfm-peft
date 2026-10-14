# E·잠재 예측기·D 분리 진단의 실행 전 고정 조건

현재는 SOURCE_ONLY, 새 자원 승인은 대기 중이다. 이 파일은 구현의 모호한 부분을 실행 전에 고정한다. 기존 계획·실행 소스·결과·ledger를 바꾸지 않는다. 실제 자산 존재·계수 추정·학습·추론·검산의 완료를 주장하지 않는다.

## 원조건과 데이터

Robin C17/K4 및 Beijing Aotizhongxin C11/K2 × old MAIN seed92601/92602의 네 조건을 유지한다. 최근 RF1/RF3 체크포인트 대신 원래 selected MAIN을 쓴다. E·D hidden·F는 동결한다. L512/H48, FIT256/TUNE64, Robin 반복 VAL30/Beijing 노출 EVAL146, 원 FIT mean/std·origin·mask를 유지한다. 미래 oracle는 진단용이며 배포 입력은 과거뿐이다. TEST와 새 독립 holdout을 사용하지 않는다.

기존 D0 SAME_MAIN_NATIVE와 D2 F+Y-refitD를 실제 SHA·원조건이 일치할 때 재사용한다. D0의 native는 같은 E/D를 고정한 시간축 공유 선형 예측기를 actual-Y 손실로 학습한 결과다. 새 N1은 joint past features와 latent teacher 목표를 쓰므로 같은 native를 두 번 실행했다고 집계하지 않는다. 없는 자산·과거 비용은 존재/0으로 추정하지 않는다.

## Ridge 손실의 정확한 의미

N1/N2는 같은 FIT-only feature state를 쓴다. 과거 latent의 32개16-point mean과 마지막 latent를 펼친 33K feature를 FIT 평균·표준편차로 정규화하고 zero scale은1로 둔다. observed-subset intercept를 penalize하지 않는다. N2 target은 normalized Y−x_t, N1 target은 oracle z*−z_t다. N1은 실제-Y 예측 손실 학습이 아니며 그 목표 차이를 보고한다.

각 출력 채널 q의 모든 FIT 창 i·horizon h 관측 수를 Nq라고 할 때 목적은 active 채널에 대한 다음 평균이다.

```text
sum_(i,h) mask(i,h,q) * (prediction(i,h,q)-target(i,h,q))^2 / Nq
    + alpha * sum_h ||W(h,q)_scaled||^2
```

따라서 각 (h,q) 회귀의 ridge normal equation에는 alpha*Nq가 들어간다. horizon별 관측수 n(h,q)를 분모로 바꾸지 않는다. alpha0은 observed-subset centered minimum-norm 해이며 rank와 singular values/condition 정보를 저장한다. 동일 mask 그룹의 factorization 재사용은 목표·해를 바꾸지 않는다. 계수 precision, feature scaling과 실제 inference casting은 artifact에 명시하고 CPU/GPU 출력 parity를 실행 전에 정한 normalized atol1e-5/rtol2e-6으로 검사한다.

고정 alpha는 0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10이다. TUNE actual-Y normalized channel-macro MSE가 최소인 후보를 선택하며 exact tie는 목록의 이른 후보다. N1 선택에도 old D로 복원한 실제-Y 점수를 쓴다. 평가로 후보·rank·feature·mask·alpha를 바꾸지 않는다.

O1/N3는 기존 D hidden32를 유지하고 [z−z_t, GELU(Az+b)−GELU(Az_t+b)] feature의 outgoing weights만 새로 맞춘다. FIT RMS feature scaling, no intercept, 관측 Y 채널의 NH 전체 count로 masked mean loss를 쓴다. innovation0에서 current-C anchor를 유지한다. zero-observed channel은 원 D outgoing weight를 보존한다. O1의 oracle 계수와 선택을 N3 배포에 가져오지 않는다. N3는 먼저 선택한 N1 계수·잠재 예측을 고정하고 D만 맞춘다.

## 미래 mask와 읽기 경로

Oracle는 과거512+실제 미래48의 normalized560점을 causal E에 넣는다. 미래 결측은 해당 시점까지 알려진 값으로 carry-forward한다. 미래 뒤의 관측값으로 앞을 채우지 않고, 원 FIT 통계도 다시 맞추지 않는다. 과거 prefix parity와 뒤 미래 교란에 대한 앞 representation 불변성을 검사한다.

N1의 latent teacher mask는 horizon h까지 미래 모든 C가 관측된 complete-prefix다. 각 h에서 K에 같은 mask를 쓴다. teacher count0인 output은 innovation0을 유지하고 미학습으로 표시한다. N2는 원 Y-observed mask다. N1/N2의 훈련 표본수 차이는 교란으로 보고하며 N2 우위만으로 D가 유일한 원인이라고 쓰지 않는다. teacher coverage가 부족해도 N2·원 D 재사용 같은 독립 경로는 계속할 수 있다.

## Anchor와 단순 대조군

N2는 E features 외에 current raw-C anchor와 horizon/channel intercept를 가진다. E의 추가 예측 기여를 판단하려면 같은 mask·anchor의 CURRENT_C_PERSISTENCE와 FIT_INTERCEPT_ONLY를 같이 보고한다. 후자는 관측 FIT의 각(h,c) Y−x_t 평균, 관측0인 출력은0이며 alpha/평가 선택은 없다. 작은 계수 추정도 신규 학습·비용으로 기록한다. 기존 LAST 예측은 실제 배열 조건이 일치할 때 재사용할 수 있다.

N2가 원 F보다 좋아도 위 두 대조군보다 좋지 않으면 33K 과거 representation이 개선에 기여했다고 주장하지 않는다. 우위를 보여도 반복 개발 데이터·두 seed·mask/target/output capacity 차이가 남으며 독립 일반화를 입증하지 않는다.

## 실행·보존·비용 gate

새 human approval record와 고정 manifest가 없으면 어떤 실제 데이터/CP read·solver·추론·fixture 실행도 시작하지 않는다. 제안 상한은 추가 GPU process600초, CPU process900초, wall2400초, 저장 증가1GiB이며 기존 만료 ledger를 재설정하지 않는다. phase와 diagnostic-only oracle scope를 새 승인 record에 명시한다. 임의의 승인 파일·기한을 생성해 허용을 추정하지 않는다.

원 CP/state·원 데이터·cached latent provenance·source/계수·origin·통계·mask/teacher coverage·FIT/TUNE 후보·선택·평가 배열을 새 경로에 저장한다. oracle와 past-only deployment를 파일/표/함수명으로 구분한다. source·원 자산을 실행 중 수정하지 않고, 실패한 실행은 새 version/tag로 수정하되 원 실패 SOURCE/배열/receipt를 보존한다. 보호 TEST·옛 중단 학습 재개·큰 F 학습·commit/push는 하지 않는다.

배포 비용은 실제 과거 입력에서 원 RAW/MAIN과 선택된 native/direct-C/decoder 변경을 batch1/8로 위치 균형 측정한다. oracle는 배포 비용 비교에 포함하지 않는다. CPU/GPU process wall은 model load·IO·guard 포함이며 GPU kernel time으로 바꾸어 부르지 않는다. 독립 CPU 검산은 저장 배열/계수/선택/score 재계산 범위를 명시하고 F 재추론으로 확대하지 않는다.
