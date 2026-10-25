# 비교·절제 실험 사전등록

캠페인: comparative_ablation_20261025_v1. 아래 고정값·결정 기준·중단 조건은 제공 문서에서 그대로 보존했다.
공식 AdaPTS 전체 재현이 아니라 동일 TSFM 조건의 어댑터 비교다. T 수치는 사용하지 않는다.

## 3. 승인된 고정값 (바꾸지 말 것)

### 공통 조건
- **데이터·K**: jena(K 6), ETTh1(4), ETTh2_safe17400(4), appliances(10). 모두 K_τ 값이다.
- **seed**: 92601–92610.
  - C1은 학습이 없어서 데이터마다 1회만 돌린다(seed 92601로 기록).
- **레시피**: 아래 표의 구조 차이 말고는 K_τ의 우리 방법과 같다.
  - L512/H48, RevIN off, A 구간 통계로 정규화, 손실은 채널 평균 MSE
  - AdamW lr 1e-3, weight decay 0, clip 1
  - effective batch 16 / micro batch 4, 학습 F row 128, FP32
  - checkpoint 0/.25/1/3. A_stop 최소(동률이면 앞쪽)로 고른 뒤 A60로 새로 재학습
  - F는 같은 F_SHA의 고정 Chronos-Bolt-small, 점예측은 중앙 분위수
- **학습 일정 예외**: C2·C3만 checkpoint 0/.25/1/3/6을 쓴다.
  - 이유: 우리 방법과 달리 무작위 초기화에서 시작하는 작은 선형 어댑터다. 이전 이식은 3 epoch 상한을 골랐다.
  - 비교 대상에게 학습을 더 주는 쪽이다.

### 비교·절제 대상

| ID | 표시 이름 | 우리 방법(B2) 대비 차이 | 구현 | 학습 파라미터(Jena, 계획자 계산) |
|---|---|---|---|---|
| OURS | Ours | — | K_τ 10 seed 재사용 | 35,717 |
| C1 | AdaPTS-PCA | P 없음, 학습 없음. E = A60 PCA로 고정, 역변환은 Eᵀ + center | 기존 `PCA_FIXED` | 0 |
| C2 | AdaPTS-LinearAE | P 없음. Linear(C→K) → F → Linear(K→C), bias 있음, PyTorch 기본 초기화. 마지막 관측값·lag 없음 | 기존 `ADAPTS` | 279 |
| C3 | AdaPTS-dropoutLAE | C2의 인코더 Linear 바로 뒤에 Dropout(p = 0.1) | 새 모듈 | 279 |
| A1 | No P | 앞단이 identity | 기존 `B0` | 1,008 |
| A2 | Capacity-matched control | 시간 MLP를 시점 공유 채널 MLP(C→h→C)로 바꿈 | 새 모듈(1절 정의를 C에 맞게 일반화) | 35,708 |
| A3 | Random-orthogonal E | E의 시작을 PCA 대신 QR 기저로 | 새 래퍼(1절 `qr_basis`) | 35,717 |
| A4 | Linear decoder | D 대신 Ŷ_h = U_h W + b (W ← pinv(E), b ← center). 마지막 관측값·lag 없음 | 새 모듈 | 34,982 |
| A5 | Decoder without lags | lag 1–32 제거. Ŷ_h = x_last + (U_h − Z_{L−1}) D₀ | 새 모듈 | 34,961 |

### 대상별 세부 규칙
- **A2의 h**
  - h = floor((P_t − C) / (2C + 1)). P_t는 B2 시간 MLP의 파라미터 수다(L = 512에서 33,312).
  - 결과: jena 774, appliances 628, ETTh1·ETTh2 2,220.
  - 실제 모델에서 다시 세어 우리 방법과의 차이를 보고한다(±10 이내 예상).
- **C3의 점예측**
  - 학습 중에는 dropout 표본 1개를 쓴다(AdaPTS 학습과 같음).
  - A_stop 선택과 V 평가에서는 dropout을 켠 표본 10개의 잠재 예측을 평균해 디코더에 넣는다(AdaPTS 예측과 같음).
  - 표본 seed는 실행 seed + 7000000으로 고정한다. 평가할 때마다 같은 seed로 다시 시작한다.
  - dropout을 끈 결정적 예측의 V 값도 부가로 기록한다(판정에는 쓰지 않음).
- **C2·C3가 AdaPTS와 다른 점**
  - 따르지 않는 것:
    - AdaPTS 학습 일정(early stopping, 최대 300 epoch)
    - 스케일러, RevIN, HEBO 하이퍼파라미터 탐색
    - batch 32, ReduceLROnPlateau
    - F로 쓴 Moment
  - 그래서 C2·C3는 원 논문 재현이 아니다. 같은 F·같은 데이터·같은 선택 규칙으로 통일한 "AdaPTS식 어댑터"다.
  - 차이는 `DEVIATIONS_FROM_ADAPTS.md`에 표로 공개한다.

### 상한 epoch 연장 규칙
A_stop 결과만 보고 정한다. V는 보지 않는다.

- 대상: 어떤 family×데이터에서 10 seed 중 5개 이상이 그 family의 마지막 checkpoint를 고른 경우.
  - 마지막 checkpoint는 C2·C3는 6, 나머지는 3이다.
- 조치:
  - 그 family×데이터의 10 seed를 checkpoint 0/.25/1/3/6/10으로 새로 학습한다(새 run_id).
  - 연장 결과를 주 결과로 쓰고, 기존 결과는 부록으로 남긴다.
- 연장에서도 5개 이상이 10을 고르면 UNDERTRAINED_AT_10으로 표시하고 더 늘리지 않는다.
- OURS에도 같은 규칙을 쓴다.
  - 92601·92602가 모두 3을 골랐어도 jena 4/10, ETTh2 3/10이라 대상이 아니다(계획자 계산).
  - 그래도 두 seed의 선택 epoch를 확인해 보고한다.
- 연장은 모든 기본 실행이 끝난 뒤 예산이 남을 때만 한다. 순서는 아래 우선순위를 따른다.

### 지표 정의 (A0)
모든 지표는 V의 관측된 점으로 계산하고 채널 평균한다.

- **MSE, MAE**: 기존 평가 그대로 계산한다. 저장된 receipt 값과 1e-9 이내로 같아야 한다.
- **R²(결정계수)**:
  - 채널마다 1 − SSE_c/SST_c를 구한다. SST_c는 그 채널 V 관측값의 평균을 기준으로 한다.
  - SST_c < 1e-12인 채널은 빼고 그 수를 보고한다.
- **MASE(평균 절대 척도 오차)**:
  - 채널마다 MAE_c / s_c를 구한다.
  - s_c는 그 실행의 정규화 구간 [0, A60 끝)에서 연속으로 관측된 쌍의 평균 |y_t − y_{t−1}|이다.
  - 정의 출처: "Another look at measures of forecast accuracy"(2006/International Journal of Forecasting).
  - 채널별 affine 변환에 불변이라 정규화 단위로 계산해도 된다.
- **MAPE(평균 절대 백분율 오차)**: 원 단위(그 실행의 정규화를 되돌린 값)로 세 가지를 보고한다.
  - (a) y ≠ 0인 모든 관측 점의 MAPE
  - (b) |y| < 0.1 × (A60 구간에서 그 채널의 표준편차)인 점의 비율
  - (c) (b)의 점을 뺀 MAPE
  - 0.1은 보고용 임시 기준이고 판정에 쓰지 않는다.
- **오차 분포**:
  - Raw 대 OURS를 먼저 계산하고, 이어서 모든 family를 같은 표로 계산한다.
  - 계산 방식: 데이터마다 관측 점(origin × horizon × 채널)을 정규화 단위로 모은다. seed별로 낸 뒤 seed 중앙값을 보고한다.
  - 항목:
    - |e|의 50·75·90·95·99% 분위수
    - |e_ours| < |e_raw|인 점의 비율
    - Raw 오차 상위 5% 점이 Raw 전체 SSE에서 차지하는 비율
    - 같은 점들에서의 OURS SSE

### 예산·마감·실행 방식
- **예산**: 학습 시작부터 실제 경과 시간 40시간이 상한이다.
  - 기본 실행 추정은 약 30시간, 최악 약 35시간이다(1절 시간 × 실행 수, 계획자 계산).
- **마감**: 2026-10-27 21:00 KST 이후에는 새 실행을 시작하지 않는다(사용자가 바꿀 수 있음).
- **동시 실행**:
  - 기본은 한 번에 한 실행이다.
  - 첫 C2 실행들로 처리량을 비교한다: jena 92601을 단독으로 돌리고, ETTh1·ETTh2 92601을 동시에 돌린다.
  - 동시 2개가 1.4배 이상 빠르고 GPU 메모리 피크가 10 GB 미만이면, 그 뒤로는 2개씩 돌려도 된다.
- **T**: 열지 않는다.
- **우선순위**:
  1. C1
  2. C2
  3. C3
  4. A1
  5. A2
  6. A3
  7. A4
  8. A5
  9. 연장. 순서는 다음과 같다:
     - C2·C3의 jena → appliances → ETTh1 → ETTh2
     - 그다음 A1–A3
     - 그다음 A4–A5
- **family 안의 순서**: seed를 데이터 사이에서 돌아가며 실행한다(92601을 4개 데이터에, 그다음 92602 …). 중간에 멈춰도 데이터마다 n이 비슷하게 남는다.
- **run_id 예시**: `CA_{ID}_{dataset}_K{k}_s{seed}`. 연장은 끝에 `_ext10`을 붙인다.

## 5. 결정 기준 (사전 등록에 그대로 복사)

표기:
- med: 10 seed V MSE의 중앙값
- IQR: numpy percentile 기본값(linear)으로 구한다
- spread: 비교하는 두 대상의 IQR 중 큰 값. 학습 없는 C1의 IQR은 0이다.

**비교 (Q1)**
- 판정 대상은 jena, ETTh1, appliances다. ETTh2_safe17400은 같은 표로 따로 보고한다.
- 최선 비교 대상: 데이터마다 C1–C3 중 med가 가장 낮은 것.
  - V로 고르므로 비교 대상에 유리하다. 이 점을 보고서에 밝힌다.
- gap = med(최선 비교 대상) − med(OURS).
  - WIN: gap > spread
  - LOSS: gap < −spread
  - TIE: 그 밖
- 종합 판정 (판정 대상 3개 기준):
  - SUPPORTED: WIN 2개 이상, LOSS 0개
  - NOT_SUPPORTED: WIN 0개
  - MIXED: 그 밖
- 연장 규칙이 걸렸는데 예산 때문에 연장하지 못한 비교 대상이 있으면, 그 데이터 판정에 PROVISIONAL을 붙인다.
- OURS 대 각 비교 대상의 Mann–Whitney U 양측 p값은 기술 통계로만 적는다.

**절제 (Q2)**
- 변형 v와 데이터 d마다 effect = med(v) − med(OURS).
  - CONTRIBUTES (그 모듈이 도움이 된다): effect > spread
  - HARMS: effect < −spread
  - UNCLEAR: 그 밖
- 4개 데이터를 모두 보고한다. 요약 집계는 판정 대상 3개로 한다.

**지표 (Q3)**
- 판정하지 않는다.
- 데이터마다 아래 문장이 맞는지만 기술한다: "OURS의 |e| 분위수가 90·95·99%에서는 Raw보다 낮고 50%에서는 높다."

이 숫자들은 합의용 임시 기준이다. 결과를 본 뒤 바꾸지 않는다.

## 6. 중단 조건

- **게이트 실패**: 멈추고 보고한다.
- **학습 전 예산 초과**: C1–C3 기본 실행만으로 40시간을 넘는다고 추정되면 학습 전에 멈춘다.
- **예산 소진·마감**: 학습 경과가 40시간을 넘거나 마감(2026-10-27 21:00 KST)이 지나면
  - 진행 중인 실행만 마치고 새 실행은 시작하지 않는다.
  - 끝난 것까지로 보고서를 쓰고, 미완료 family는 n을 표시한다.
- **실행 실패**:
  - 같은 실행이 두 번 실패하면 FAILED.
  - 한 family에서 FAILED가 3개면 그 family를 멈춘다.
- **기존 파일 변경**: 기존 파일 hash가 바뀌면 즉시 멈추고 보고한다.
- **계획 밖 실험 금지**: 시간이 남아도 RUN_PLAN 밖 실험(VAE 계열, 다른 K, T 평가 등)은 시작하지 않는다. 남은 시간만 보고한다.

