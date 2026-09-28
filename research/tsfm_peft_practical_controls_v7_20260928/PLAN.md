**A 필수 대조와 B의 Jena 기상 자료 확장을 포함하는 v7 실행계획으로 제안합니다.** B는 이미 사용된 자료이므로 독립 확증이 아니라 **노출을 명시한 다른 도메인 적용범위 시험**입니다. A의 결과가 불리하더라도 승인된 B를 수행합니다.

이번에는 저장소·기록·소스·메타데이터·최소 TRAIN 적격성만 확인했습니다. 새 학습·예측·채점·benchmark·프로젝트 파일 수정·push는 하지 않았습니다. 아래 실행 사항은 모두 **[제안·미검증]**입니다.

**1. 현재 상태와 재사용 범위**

[확인] 작업 디렉터리는 `E:/CODING/proj/hierarchical-tsfm-peft`, origin은 요청한 저장소, branch는 `main`입니다. 로컬 HEAD와 실시간 원격 main은 모두 `a3ccb0a3ae25c075c79015545a0b8d7013ad6550`입니다. 추적 파일 변경은 없으며 기존 무관한 미추적 항목 6개를 유지합니다. 실행 중인 Python 학습 프로세스는 없었습니다. 로컬 GPU는 RTX 4070 한 장입니다.

v6의 승인 계약·결과·코드와 다음 재사용 파일을 확인했습니다.

- TEST 예측 14개, TRAIN/VAL 및 TEST 데이터 2개.
- LEVEL 두 seed·LoRA 두 seed·F0의 checkpoint 5개, 대응 VAL 예측 5개, result receipt 5개.
- 위 **31개 모두 존재하며 manifest SHA256과 일치**합니다. 공통 초기값과 공식 Bolt 가중치도 확인했습니다.
- checkpoint는 `.cache/tsfm_peft_level_confirmation_v6_20260928/runs/{실제 run ID}/best.pt`, TEST 예측은 같은 cache의 `evaluation01_predictions/{실제 run ID}.npz`입니다. 예측 파일의 실제 키는 `prediction`, `origins`입니다.
- 정확한 ID·전체 hash는 [selected.json](E:/CODING/proj/hierarchical-tsfm-peft/research/tsfm_peft_level_confirmation_v6_20260928/selected.json), [evaluation01.json](E:/CODING/proj/hierarchical-tsfm-peft/research/tsfm_peft_level_confirmation_v6_20260928/evaluation01.json), [data_contract.json](E:/CODING/proj/hierarchical-tsfm-peft/research/tsfm_peft_level_confirmation_v6_20260928/data_contract.json)을 재사용 manifest로 연결합니다.

Robin의 기존 LEVEL·OLD·RAW·LEVEL_ONLY·COMPRESS·F0·LoRA·LEVEL_LINEAR_RES·SHARED는 재학습하지 않습니다. 기존 두 선형 대조를 NLinear로 재명명하지 않습니다.

승인 후 만들 폴더는 다음으로 고정합니다. 두 경로 모두 현재 존재하지 않습니다.

```text
공개 코드·작은 결과
E:/CODING/proj/hierarchical-tsfm-peft/research/tsfm_peft_practical_controls_v7_20260928/

비공개 배열·checkpoint
E:/CODING/proj/hierarchical-tsfm-peft/.cache/tsfm_peft_practical_controls_v7_20260928/
```

현재 사용자 지시와 승인될 v7 계약을 적용합니다. 과거 실험의 재실행 제한·봉인·위반 기록은 원래 기록에 보존하며 v7 전체에 자동 확장하지 않습니다.

**2. 이번에 판단할 기여와 가장 가까운 선행**

최상위 질문은 **“직접 선형 모델과 미압축 소배치 실행까지 고려해도, 같은 과제에서 LEVEL을 선택할 이유가 남는가?”**입니다.

[확인] 선행 검토에서 기여 설명을 다음처럼 좁히는 것이 타당합니다.

- **NLinear:** 공식 모델은 공유 `Linear(L,H,bias=True)`와 마지막 값의 detach·제거·복원입니다. 기본 난수 초기화를 사용합니다. 공식 데이터 loader도 TRAIN 표준화를 하므로, 외부 TRAIN 표준화를 우리만의 차이로 쓰지 않습니다. 모델 전체와 학습·선택 경로를 읽었으며, 공식 runner의 epoch별 TEST 조회는 복제하지 않습니다. [AAAI 2023 논문](https://ojs.aaai.org/index.php/AAAI/article/view/26317/26089), [저자 모델 코드](https://github.com/cure-lab/LTSF-Linear/blob/main/models/NLinear.py)
- **AdaPTS:** 잠재 E/D·차원 축소·동결 FM·예측 목적의 적응은 중복입니다. 읽은 저자 코드의 supervised 예측 경로는 `transform → FM → inverse_transform`이며, reconstruction loss는 별도 학습항입니다. 그 경로에는 관측 재구성잔차의 미래 보정 분기가 없습니다. 최종 ICML 출판 정보와 실제 코드는 확인했지만 최종 PDF 접근은 실패했으므로, 기존 preprint 방법절 검토와 구분합니다. COMPRESS는 전체 AdaPTS 재현이 아닙니다. [공식 출판 기록](https://proceedings.mlr.press/v267/benechehab25a.html), [저자 예측 코드](https://github.com/abenechehab/AdaPTS/blob/main/src/adapts/adapts.py)
- **RevIN:** 평균·scale 제거/복원과 선택적 affine이 핵심이며, 마지막 잔차 수준만 복원하는 LEVEL과 다릅니다. 코드·저자 설명을 확인했고 이번 OpenReview 본문 접근은 막혔습니다. [저자 코드](https://github.com/ts-kim/RevIN/blob/master/RevIN.py)
- **Deep Factors:** 전역 deep factor와 지역 확률효과를 결합하는 원리가 선행입니다. 최종 논문 §3.1–3.2, 식 (3)–(7)을 확인했습니다. 현재 방법의 고정 PCA·동결 TSFM·공유 결정론적 시간 선형 G와 학습·정보 조건을 구분합니다. [ICML 2019 원문](https://proceedings.mlr.press/v97/wang19k/wang19k.pdf)

본문에는 다음 등가 해석을 공개합니다. \(X\in\mathbb R^{L\times C}\), \(Q=I-UU^\top\), 마지막 값 반복을 \(P_n\)이라 하고,

\[
\mathcal N_G(X)=P_H(X)+G(X-P_L(X))
\]

라 정의하면, 고정 직교 PCA와 채널공유·무편향 선형 G에서

\[
\widehat Y_{\mathrm{LEVEL}}
=b+\mathcal N_G(XQ)
=b+\mathcal N_G(X)Q.
\]

즉 **동결 잠재 TSFM과 버려진 공간의 저랭크 NLinear를 결합한 설계**입니다. 이 대수적 해석 자체를 새로운 정규화 원리나 성능 원인의 증명으로 주장하지 않습니다. 표준 affine NLinear의 bias까지 포함하면 위 교환식은 그대로 적용되지 않습니다.

v7이 지지할 수 있는 기여는 이 구체적 배치의 대조 결과와 선택 조건입니다. retained-space 보존, offset equivariance, 유효 rank≤33은 조건부 성질로 남깁니다. 직접 NLinear·소배치 대조가 충분하면 실용 주장을 축소합니다.

ELF·EPOC의 온라인 예측오차 피드백과 현재 관측입력의 재구성잔차는 구분합니다. ELF의 ICML 2025 기록, EPOC의 실제 arXiv 기록과 preprint 상태도 재확인했습니다. 문헌 접근 한계는 해당 항목에만 남기고 실험 전체의 진입 조건으로 만들지 않습니다.

**3. A1 — Robin 직접 NLinear 4fit**

모델은 다음 하나로 고정합니다.

\[
p=\operatorname{stopgrad}(X[:,511,:]),\qquad
N(X)=P_{48}(X)+T(X-P_{512}(X)).
\]

- \(T\): 모든 원채널에 공유하는 `Linear(512,48,bias=True)`.
- `individual=False`, 학습 파라미터 **24,624개**.
- PCA·E/D·G·Chronos·문맥 표준편차 정규화를 넣지 않습니다.
- 입력에는 기존 Robin TRAIN 표준화와 결측 대체만 적용합니다.
- 공식 `nn.Linear` 기본 초기화를 사용합니다. 초기 함수는 일반적으로 persistence나 LEVEL_ONLY와 같지 않습니다.
- 학습 없이 같은 가중치를 넣은 공식 NLinear forward와 비교하는 검사는 **승인 후** 수행합니다.

첫 실행은 정확히 다음 4fit입니다.

```text
Dataset  Model           LR       Seed
Robin    DIRECT_NLINEAR  1e-3     92601
Robin    DIRECT_NLINEAR  1e-3     92602
Robin    DIRECT_NLINEAR  1e-4     92601
Robin    DIRECT_NLINEAR  1e-4     92602
```

같은 seed의 두 LR은 동일한 미학습 W/bias와 동일한 epoch별 TRAIN 원점 순서를 사용하고 hash를 남깁니다.

학습 계약은 v6에서 확인한 다음 규칙을 재사용합니다.

- AdamW, weight decay 0, clip 1, batch 4.
- epoch당 TRAIN 512원점, Robin의 24시간 위상균형 표집.
- 최대 120epoch, strict VAL nonimprovement 6회.
- plateau factor 0.5, patience 2, relative threshold \(10^{-4}\).
- 각 fit은 **step0 포함 전체 VAL MSE 최소**, 정확히 동률이면 먼저 나온 상태.
- LR은 **두 seed의 best VAL MSE 평균**으로 선택하며, 정확한 동률이면 목록의 첫 값 `1e-3`.
- 기존 split·채널·mask·원점을 그대로 사용합니다: TRAIN 1,945, VAL 30, TEST-A/B 각 40원점.
- TEST는 LR·epoch 선택에 사용하지 않습니다.

고정 TRAIN probe, 초기/선택 VAL, 실제 파라미터 갱신·유한 손실·저장/재시작 복원을 기록합니다. 낮은 성능만으로 추가 LR·초기화 탐색을 열지 않습니다. 명백한 미학습은 정상적인 음성과 구분합니다.

선택된 두 NLinear checkpoint를 기존 14개 모델의 저장 예측과 같은 원점·mask로 비교합니다. 주 비교는 **LEVEL 대 DIRECT_NLINEAR**이며, 나머지 기존 대조도 같은 표에 유지합니다. MSE·MAE·기간·seed·채널, 절대차와 참조 분모의 상대 변화율을 보고합니다. Robin의 이 추가 비교는 **v6 결과를 본 뒤 설계한 후속 비교**입니다.

**4. A2 — 미압축 소배치 실행 대조**

[확인: 코드 읽기] 기존 경로는 `[B,512,C] → [B×C,512]`로 펼쳐 raw Bolt를 호출하고, median quantile의 첫 48개를 `[B,48,C]`로 복원합니다. 설치된 Bolt의 정규화와 T5 attention에는 서로 다른 계열 행을 섞는 경로가 없습니다. 따라서 행 분할의 수학적 근거는 있지만, 배치 shape 변경에 따른 부동소수점 차이는 실제로 검사해야 합니다. [기존 forward](E:/CODING/proj/hierarchical-tsfm-peft/research/tsfm_peft_level_confirmation_v6_20260928/model_v6.py:118), [PyTorch 수치 정확도 안내](https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html)

다음 유한 grid를 고정합니다.

```text
Model instances         B=1                              B=4
F0 ×1, merged LoRA ×2    original, full M17, M5            original, full M68, M20, M5
LEVEL ×2                original, full M5                 original, full M20, M5
DIRECT_NLINEAR ×2        original GPU                     original GPU
```

여기서 `full`도 새 generic wrapper를 통과합니다. 원래 경로는 wrapper overhead를 구분하는 참조로 남깁니다.

- 블록당 primary 35개.
- 별도 F0 pre/post sentinel을 B1/B4에 각각 실행하여 블록당 4개 추가.
- 3블록으로 **GPU 측정 117행**.
- DIRECT_NLINEAR 두 seed의 CPU B1/B4는 **12행**.
- sentinel을 primary 결과로 재사용하여 수를 부풀리거나 줄이지 않습니다.
- B 기상 자료에는 이 M-grid를 추가하지 않습니다.

구현은 GPU에 올린 독립 계열 행만 `M`개씩 나눕니다. 시간축 512나 미래축 48은 자르지 않습니다. 각 chunk의 median을 미리 할당한 `[B×C,48]` 또는 `[B×K,48]` GPU buffer에 순서대로 복사하고, 원시 quantile 출력 참조를 해제합니다. 마지막에 전체 원채널 출력을 CPU로 반환합니다. **full-M과 chunk-M에 같은 buffer 정책**을 적용하며 매 chunk CPU 왕복·모델 reload·cache flush는 하지 않습니다. LEVEL의 E/D·G·수준 복원은 모두 실행합니다.

정합성과 측정은 다음으로 고정합니다.

- 공통 VAL 24원점에서 모든 B/M의 전체 출력과 행 순서를 검사.
- 허용오차는 v6와 같은 **atol \(10^{-5}\), rtol \(10^{-4}\)**.
- 최대 절대차·상대차·위반 원소 수를 기록하고, 통과할 때까지 tolerance를 늘리지 않음.
- LoRA의 병합 전후 검사와 병합 모델의 original/full/chunk 검사를 분리.
- 정합성이 통과한 경로는 Robin의 고정 TEST 80원점에서도 예측·MSE/MAE 차이를 확인. 이는 fit 0의 실행 동등성 확인이며 설정 선택에 사용하지 않음.
- 출력 차이가 크면 구현 오류를 수리하거나 해당 경로를 비동등으로 보고. 원래 모델의 유효한 정확도 결과까지 자동 폐기하지 않음.

비용은 같은 세션의 FP32·TF32 off·CPU threads 4에서 **표준화 CPU 입력 → H2D → 전체 CPU 출력**을 측정합니다. 추가 호출·reshape·조합을 포함하며 load·disk·채점·원장은 제외합니다. shape별 예열 10회, VAL 24원점 전체 20회 반복, 순서를 바꾼 3블록을 사용합니다. CUDA 동기화와 예열은 [공식 benchmark 안내](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)에 맞춥니다.

각 실행의 블록 중앙값 세 개에서 중앙값을 구하고, 두 seed 모델은 그 값을 평균합니다. batch4 throughput, batch1 latency, peak allocated/reserved, 로드 후 resident allocation, process memory, 전체 배포 bytes를 구분합니다.

**M별 모든 점과 블록 변동을 제시하고 단일 ‘승자 M’을 임의 선정하지 않습니다.** original→wrapper 차이와 full-wrapper→chunk 차이도 구분합니다. 변동이 남으면 최대 한 번, GPU 10분 이내의 상주 입력 또는 profiler 진단만 허용합니다.

**5. A3 — 남은 Robin 오차의 제한된 진단**

진단은 기존 예측·실제 target·TRAIN PCA를 재사용하며 새 fitting을 하지 않습니다.

1. **오차의 위치와 집중도:** LEVEL−F0/LoRA/OLD의 채널·원점·기간별 MSE/MAE 및 signed mean error를 계산합니다. LEVEL 자체의 큰 손실과 비교군 대비 초과손실을 분리하고, 양의 초과손실·음의 차이·순차이의 합을 각각 남깁니다.
2. **공간 안/밖 분해:** 각 `(origin, horizon)`에서 **17채널 target이 모두 관측된 행**만 공통 subset으로 사용합니다. \(e=\hat Y-Y\), \(P=UU^\top\), \(Q=I-P\)에 대해 \(\|eP\|^2/C\), \(\|eQ\|^2/C\)를 비교하고 기간별 coverage를 보고합니다. 결측 target을 0으로 채우지 않습니다.
3. **과거 잔차와 이후 오차:** 마지막 잔차 수준, 최근 24시간 평균·표준편차·변화량과 이후 잔차 보정오차의 관계를 채널·기간별 산점도와 기술 통계로 봅니다. 미래 target은 사후 정답에만 사용하며 입력·routing·특성 선택으로 되돌리지 않습니다.

공간 분해는 TRAIN 표준화 좌표의 완전관측 Euclidean 진단입니다. 원래 masked channel-macro MSE와 같은 수치라고 부르지 않습니다. 원점과 겹친 horizon을 독립 표본처럼 세지 않으며, 상관관계를 원인 증명으로 쓰지 않습니다.

공간 안쪽 손해는 주경로, 바깥 손해는 잔차 시간모델의 후속 검토 단서가 될 수 있습니다. **이번에는 그 결과로 K/G/loss/채널을 바꾸지 않습니다.**

**6. B — Jena 2024를 포함하되 노출된 도메인 확장으로 고정**

[확인] 사용할 자료는 다음 하나입니다.

[로컬 MPI Jena 2024 CSV](E:/CODING/proj/mltimeseries/data/jena_mpi_roof/mpi_roof_2024.csv)

```text
파일       mpi_roof_2024.csv
크기       11,068,144 bytes
SHA256     65a47db98ad85b0600f2deab1e11055af8fef9c5aedceadbdcd0011748120558
원천       MPI-BGC Jena, WS Beutenberg
행 수      52,704
시간       2024-01-01 00:10 ~ 2025-01-01 00:00
간격       정확한 10분 격자
C / K      21 / 6
다운로드   0
```

공식 archive의 로컬 hash와 과거 다운로드 receipt가 연결돼 있습니다. 현재 원격 archive와의 byte 동일성을 새로 확인한 것은 아닙니다. 공식 사이트의 이용 조건은 CC BY 4.0입니다. [공식 자료 안내](https://weather.bgc-jena.mpg.de/weather_data.html), [로컬 source receipt](E:/CODING/proj/mltimeseries/data/oa_resolution_pilot_v1_source_hashes.json)

타깃은 실제 CSV의 숫자 변수 21개 전체입니다.

```text
p(mbar), T(degC), Tpot(K), Tdew(degC), rh(%),
VPmax/VPact/VPdef(mbar), sh(g/kg), H2OC(mmol/mol), rho(g/m³),
wv/max.wv(m/s), wd(deg), rain(mm), raining(s),
SWDR(W/m²), PAR/max.PAR(µmol/m²/s), Tlog(degC), CO2(ppm)
```

source-naive timestamp를 그대로 사용하며 UTC/DST 변환을 만들지 않습니다. 풍향을 새 sin/cos 특징으로 바꾸거나 변수를 성능으로 제외하지 않습니다. 서로 다른 물리 단위의 오차를 직접 평균하지 않고 TRAIN 표준화 후 채널 동일가중 손실을 사용합니다.

**노출 근거가 있습니다.** 같은 SHA256 자료가 현재 저장소의 [과거 DATA_AUDIT](E:/CODING/proj/hierarchical-tsfm-peft/results/internal_adaptation_gap_v1_20260922/jena/DATA_AUDIT.json)에 TEST 사용까지 기록돼 있습니다. 그 실험은 Chronos-2·1152/144·다른 split/loss이므로 이번 Bolt LEVEL 비교와 동일 실행은 아닙니다. 기존 점수를 이번 점수로 재사용하지 않습니다.

조회 범위는 현재 저장소와 접근 가능한 `mltimeseries`, `covariate-trust-pilot`, `tsfm-peft-method-screen`의 관련 기록입니다. `forecast-revision-peft`는 확인한 로컬 경로에 없었습니다. 전체 대화·전체 원격 이력·백본 사전학습 비중복은 미확정입니다.

분할과 원점을 다음으로 고정합니다.

```text
역할     반개방 target 구간                              원점 수
TRAIN    [2024-01-01 00:10, 2024-07-01 00:00)             25,648
VAL      [2024-07-01 00:00, 2024-09-01 00:00)                371
TEST-A   [2024-09-01 00:00, 2024-11-01 00:00)                365
TEST-B   [2024-11-01 00:00, 2025-01-01 00:10)                365
```

이 원점 수는 **시간 인덱스로 확인한 값**이며 새로운 성능 결과가 아닙니다.

- L512/H48: 공칭 약 85시간 20분 과거 → 8시간 미래.
- TRAIN 후보는 stride 1. epoch당 512개를 **하루 144위상**에 균형 표집합니다.
- VAL/TEST는 stride 24관측, 즉 **4시간**입니다. 하루 여러 시간대를 포함합니다.
- 과거 문맥은 이전 split의 당시 관측 가능 값까지 허용하고, 목표 48개 전체만 해당 split 안에 둡니다. VAL/TEST 시작에서 문맥 512개를 다시 버리지 않습니다.
- TRAIN sentinel 결측 처리 후 최소 관측률은 약 0.999504이고 21개 모두 비상수였습니다.
- 비유한 값과 `-9999` 계열 sentinel을 missing으로 표시합니다. 다른 이상값 clipping이나 0값 결측화는 하지 않습니다.
- TRAIN 관측값으로 mean/std를 계산하고, 표준화된 입력 결측은 0으로 대체하여 PCA를 적합합니다. target mask는 원래 관측 상태를 유지합니다.

B의 기본 실행은 다음 **12fit**입니다.

```text
방법                 설정                             fit
LEVEL_RES            FIXED PCA K6 × 2seed                2
OLD_RES              FIXED PCA K6 × 2seed                2
LEVEL_RAW            FIXED PCA K6 × 2seed                2
미압축 LoRA          기존 r8/q,v/alpha16 × 2seed          2
DIRECT_NLINEAR       shared, 2LR × 2seed                  4
F0/COMPRESS/ONLY      무학습 참조                         0
합계                                                    12
```

LEVEL·OLD·RAW는 G 512→32→48, bias/activation 없음, 출력층 0 초기화, G LR \(10^{-3}\)을 유지합니다. LoRA는 LR \(10^{-4}\)와 native quantile loss, 나머지는 masked macro MSE입니다. 모든 학습군은 같은 공통 학습·checkpoint 규칙을 적용합니다. NLinear는 A와 같은 **공유형 2LR**이며 individual 모델을 추가하지 않습니다.

v6의 C17/K5 guard를 해제하지 않고, 새 runner에서 승인된 `Robin 17/5`, `Jena 21/6`만 지원합니다. B의 E/D/G는 새 TRAIN PCA와 새 미학습 초기값에서 시작합니다.

B는 **정확도만 확인**합니다. 원형 full inference를 고정하고 Robin에서 유리했던 M을 선택해 이식하지 않습니다. 주 비교 LEVEL−OLD, RAW·ONLY·F0·LoRA·NLinear 비교를 함께 평가합니다. 불확실성은 4시간 stride에 맞춘 **42원점 블록=7일, 2,000회**, TEST-A/B 내부에서 각각 재표집하고 채널·방법을 함께 움직입니다.

**7. 선택·수리·예산의 경계**

A와 B의 자료·후보·비교·지표·비용 grid를 **A 새 결과 이전에 함께 고정**합니다. 각 자료의 모든 VAL 선택과 필요한 보완을 마친 뒤 checkpoint를 봉인하고 예정된 TEST 비교를 수행합니다. A의 평가 완료가 이미 승인된 B 학습을 막지 않도록 단계별 guard를 사용합니다.

예비 4attempt는 다음에만 씁니다.

- API·경로·직렬화·복원·실제 구현 오류로 무효가 된 fit의 복구.
- 120epoch까지 도달했고 정체 종료가 아니며 `120−best_epoch<6`인 실행의 한 차례 240epoch 연장.
- 연장은 마지막 모델·optimizer·scheduler·RNG에서 이어가고 기존 best를 보존합니다.
- 같은 자격 규칙을 모든 신규 학습군에 적용합니다. 적격 보완 전체가 남은 예비를 초과하면 유리한 군만 골라 연장하지 않고 학습상한 한계를 보고합니다.
- 실패·실자료 optimizer smoke·재시작·연장은 모두 추가 attempt입니다.

```text
적합 배분                              기본   예비
A: Robin DIRECT_NLINEAR                   4      -
B: Jena 고정 비교                       12      -
기술 복구·자격 있는 상한 보완              -      4
전체                                    16      4
하드 상한                                      20
```

GPU 작업점유 상한은 **14,400초**, 동시 신경망 학습 1개입니다. 다음은 완료시간 예측이 아닌 초기 운영 배분입니다.

```text
A 기본 학습                    20분
B 기본 학습                    90분
fit 복구·허용 연장              30분
정합성·고정 평가                20분
비용 grid                      60분
조건부 진단                    10분
미배정 여유                    10분
합계 상한                     240분
```

[확인] v6 GPU 비용 78행은 635.49초였습니다. [추정] 이번에는 B4/M5가 14회 호출로 나뉘므로 단순 행 수 비례가 부적절합니다. 기존 호출 시간에 근거해 비용 단계에 30–60분 범위를 배분했으며 새 실측으로 일정만 조정합니다. 총상한은 늘리지 않습니다.

합성 optimizer 최대 6세션·세션 전체 10update 이하, CPU 검사 누적 900초, 신규 저장 5GiB, 다운로드 0을 유지합니다. 모든 실행을 공통 원장에 사전 예약하고 부모 GPU job과 내부 fit 시간을 이중 가산하지 않습니다. 문서·구문 검사는 예약된 검사 세션에 묶습니다.

다른 작업 종료·설치·전역환경/driver/power 변경은 하지 않습니다. 자원 충돌 대기는 회당 30분·누적 60분을 넘기지 않습니다.

TEST 이후에는 구조·loss·LR·K·채널·기간·자료·선택을 바꾸지 않습니다. 재현 가능한 채점/복원 오류만 원본 노출과 영향을 남겨 관련 비교 전체를 정정합니다. 선택 자체가 바뀌어야 한다면 그 범위만 별도 결정 대상으로 보고합니다.

**8. 원고 수정과 완료 판단**

승인 후 v6 원고를 보존하고 v7 `PAPER_DRAFT.md`와 `APPENDIX.md`를 만듭니다. 긴 원고를 쓰기 전에 독자의 질문과 답변 위치를 먼저 정리합니다.

본문은 다음 흐름으로 고칩니다.

1. 압축된 잠재 채널이 만드는 표현 제약과 문제의 중요성.
2. projected NLinear라는 중복을 포함한 정확한 방법·정보·초기함수 차이.
3. 조건부 공간 성질과 성능 원인 증명의 구분.
4. Hog 등 개발 → Robin v6 고정 확인 → v7 사후 대조·진단 → 노출된 Jena 확장의 증거 단계.
5. 직접 NLinear와 소배치 F0/LoRA까지 포함한 정확도·메모리·시간의 선택 조건.
6. 불리한 결과와 남은 적용 한계.

해시·원장·전체 seed/channel 표·세부 환경·재시작·상세 증명·모든 비용 블록은 부록과 실행 기록으로 이동합니다. 핵심 정확도 손해, 실용 반례, leakage 방지와 선택 규칙은 본문에 남깁니다.

결과별 결정은 다음과 같습니다.

- **직접 NLinear·소배치 대조 뒤에도 가치가 남음:** 그 조건의 방법 기여를 유지합니다.
- **NLinear가 더 좋은 선택:** TSFM 필요성과 배포 주장을 축소하되, 기존 matched-control 양성 결과와 분리합니다.
- **소배치 F0/LoRA가 더 정확하고 메모리·시간도 유리함:** LEVEL의 배포 기여를 낮춥니다.
- **메모리를 맞추는 소배치가 더 느림:** 실제 시간 손해를 포함한 절충으로 기술합니다.
- **Jena에서 이득이 사라짐:** 전력 범위로 제한하거나 성립조건 미확정으로 남깁니다. 자료를 교체하지 않습니다.
- **원인·문헌 일부 미확정:** 그 한계만 남기고 유효한 비교와 원고를 완료합니다.

승인 후 산출물은 `PLAN.md`, `APPROVAL.txt`, `STATUS.md`, 자료·재사용 manifest, 실행별 result/curve/receipt, 선택·평가 기록, NLinear 비교, microbatch 정합성·전체 grid, Robin 진단, B 계약·봉인·평가, `CLAIMS_PRIOR_ART.md`, 원고·부록·references, `TOPIC_DECISION.md`, 수치/소스 검산과 게시확인입니다.

숫자·분모·단위·수식·그림·선행 연결을 검산한 후 관련 파일과 결과 색인만 개별 staging하여 일반 commit·`origin/main` push·실시간 SHA 확인까지 진행합니다. raw·가중치·예측 배열과 무관한 dirty 파일은 게시하지 않습니다. 원고 완료와 제출 준비 완료를 구분하며 논문을 직접 제출하지 않습니다.

실행 전 검토용 요약은 다음과 같습니다.

```text
항목                 이번 전체 승인안
────────────────────────────────────────────────────────────────────
A 필수 4fit          Robin shared DIRECT_NLINEAR
                     LR {1e-3,1e-4} × seed {92601,92602}; LEVEL 재학습 없음

소배치 대조          F0/LoRA: B1 M17·M5, B4 M68·M20·M5
                     LEVEL: B1 M5, B4 M20·M5; 원형 경로도 별도 측정
                     GPU117행 + NLinear CPU12행
                     atol1e-5/rtol1e-4; CPU입력→전체 CPU출력 시간

진단 질문            손해의 채널·원점·bias·집중도는?
                     공통 완전관측 벡터의 PCA 안/밖 손해는?
                     과거 잔차 수준·변동과 이후 보정오차의 관계는?
                     진단을 이용한 LEVEL 수정은 이번 범위 밖

B 포함·계약          포함: 로컬 공식 Jena 2024, 21채널/K6, 10분 자료
                     L512/H48, 고정 월별 split, 평가 stride24=4시간
                     노출된 도메인 확장, 정확도만 확인, 추가12fit
                     Robin 결과의 유불리와 무관하게 수행

전체 예비·상한       기본16 + 복구/자격 있는 연장4 = 최대20attempt
                     GPU4시간·동시학습1·합성6세션×최대10update
                     CPU검사900초·신규5GiB·다운로드0

원고·완료            본문에 직접 대조·선행 중복·불리한 결과를 반영
                     세부 감사자료는 부록; 수치 검산 후 main 게시
                     유지/축소/보류의 근거 있는 결정이면 완료
────────────────────────────────────────────────────────────────────
```

**위 A+B 전체 v7 실행계획을 승인하시겠습니까?**
