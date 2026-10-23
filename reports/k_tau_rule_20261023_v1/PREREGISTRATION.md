# Kτ 규칙 사전등록

캠페인: k_tau_rule_20261023_v1. 새 namespace 안에서만 출력한다. 기존 학습 핵심은 byte 단위로 유지한다.

규칙: 표준화한 A_fit 학습 행의 PCA 누적 설명분산이 τ 이상이 되는 가장 작은 K. 주 τ=0.9. τ=0.7,0.8,0.95는 K만 기록하며 학습하지 않는다. A60 계산은 기술 기록이고 A_fit과 다르면 A_fit K를 쓴다. K=C는 NO_COMPRESSION으로 기록하고 기존 create_model의 identity 초기화로 학습한다.

자료와 fit_end(상한 미포함): jena A_fit10482/A6012579, ETTh1 8710/10452, ETTh2_safe17400 8700/10440, appliances 9867/11841. source는 기존 task와 같은 pre-T prefix, DEFAULT128. K 계산에는 [0,A_fit_end)만, A60 기록에는 [0,A60_end)만 사용한다. V는 규칙 선택에 사용하지 않고 T는 평가/수치 접근하지 않는다.

seed는 92601–92610 정확히 열 개. 모델 L512/H48, RevIN off, family B2/init_kind PCA, P_B2→E→고정 Chronos-Bolt-small→lag D+원채널 마지막값. AdamW lr .001/weight decay0/clip1, effective16/micro4, FP32, F row128/checkpoint_inputsFalse. checkpoint0/.25/1/3, A_stop MSE의 최소(정확한 동률은 이른 epoch), 같은 초기화에서 선택 epoch만큼 A60 fresh refit. M_cal 사용 없음. V 원채널 정규화 MSE/MAE는 기존 저장 배열 FP64 검산 코드를 그대로 사용한다. 새 latency 측정 없음.

재사용: K_main이 기존 K grid에 있고 G4 통과이면 기존 두 seed 재사용+나머지8회; 새 K이면10회; G4 실패이면 K_main10회와 기존3K×2seed6회를 신규 실행한다. 지문 일부 unknown이면 기존 최소 K/seed92601을 등록된 단독 재실행으로 비교해 차이가 기존 2seed range보다 엄격히 작을 때만 재사용한다. 예산 계산은 Jena638s/ETTh1336s/ETTh2339s/appliances311s 상한×실행수이며 새 K 첫 실행의 실제 wall로 보수적으로 갱신한다. 시간 상한은6.75시간(24300초), 준비 시간 보수적 청구와 중단 대기는 EXECUTION_CLOCK에 따로 공개한다.

손해 = K_main의 seed92601/2 평균 MSE − min(기존 후보∪{K_main}의 같은2seed 평균 MSE). numpy percentile linear. V를 보고 K·seed·τ·학습량을 변경하지 않는다. 두 seed 범위는 신뢰구간이 아니며 V는 반복 개발 평가다.

## 승인된 Step 1

이 커밋 전에는 이번 규칙을 위한 누적 설명분산 곡선이나 τ별 K_τ를 계산하지 않는다. 기존 학습이 PCA 초기화 때 내부에서 계산한 고유값은 예외이며 G1에 따라 공개한다.

## 승인된 G1

멈춤: 사전 등록 커밋보다 먼저 네 데이터의 A_fit 누적 설명분산 곡선이나 τ별 K를 계산·출력한 산출물(파일·로그·노트북)이 있으면 멈추고 경로를 보고.
공개만 하고 진행:
(a) 기존 학습의 PCA 초기화 고유값 계산(fit_base.py 103행)
(b) 기존 산출물에 저장된 학습 K의 explained_variance(fit_base.py 106행): 있는지, 보고서·로그에 출력된 적이 있는지 목록
(c) 계획자가 τ를 정하기 전에 본 Jena 다른 구간 스펙트럼(reports/incremental_latent_adaptation_20261019_v1/COVARIANCE_SPECTRUM.csv)
PREREGISTRATION 커밋 시각은 이번 SPECTRUM.csv 생성보다 앞서야 함.

## 승인된 G5

과학적 학습 조건만 필드별로 비교하고, K·seed 외 차이가 있으면 멈춤: F SHA, 데이터 SHA·행 정책·분할 경계, L/H, RevIN, family·init_kind, P/E/D 구조, 손실, 옵티마이저·lr·weight decay·clip, effective/micro batch, 학습 F row, checkpoint·epoch 상한, A_stop 선택 규칙(동률 처리 포함), A60 재학습 규칙, 스케일러 적합 구간, 학습 핵심 코드 hash.
실행 식별 정보(job_id·run_id, 이름공간·출력 경로, 래퍼 hash, 시각, gate·protocol binding, 이들로 만든 fingerprint)는 달라도 됨.
다른 필드는 전부 과학 조건 / 실행 식별로 나눈 표로 보고하고, 어느 쪽인지 애매하면 멈춤.

## 결정 기준 원문

- **판정 대상**: 기존 후보 중 2 seed 평균 V MSE가 Raw보다 낮은 K가 하나라도 있는 데이터. 계획자 계산으로는 {jena, ETTh1, appliances}다(G7로 확인).
- **데이터 d 성공**: K_main 10 seed V MSE의 중앙값 < Raw V MSE.
- **판정**
  - 대상 3개 모두 성공 → ADOPT
  - 1개 실패 → ADOPT_CONDITIONAL(실패 이유 기록)
  - 2개 이상 실패 → REPLAN
- **한계 표시**: 손해가 그 데이터 K_main 10 seed의 사분위 범위보다 크면 LIMITATION.
- **ETTh2_safe17400**: 판정에서 빼고 같은 표로 따로 보고한다.
- 이 숫자들은 합의용 임시 기준이다. 결과를 본 뒤 바꾸지 않는다.

## 중단 조건 원문

- 게이트 실패
- 학습 전 예산 추정이 6.75시간 초과, 또는 학습 중 누적 시간이 6.75시간 초과
- 같은 실행이 두 번 실패
- REPLAN 판정: 보고서는 끝까지 쓰되, 변화량 기준 설명분산이나 검증 기반 K 선택 같은 추가 실험은 시작하지 않는다.
