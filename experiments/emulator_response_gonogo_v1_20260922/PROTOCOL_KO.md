# 예측 변화 보존 대리모델 PEFT — 구현 프로토콜 (결과 전 고정)

실행 계약은 같은 폴더의 `MASTER_CLI.txt`와 `config.json`(번들 `config_proposal.json`과 바이트 단위 동일)이다. 이 문서는 계약이 CLI에 맡긴 구현 세부를 결과를 보기 전에 고정한다. 둘이 충돌하면 계약이 우선한다. 이 문서의 해시는 `SOURCE_SEAL.json`에 기록된다.

## 1. 목적과 소비처

- 소비처: 사용자가 E_RESPONSE_DELTA를 후속 방법론 주제로 개발할지 정하는 결정.
- 문장: Jena에서, 대리모델에서 학습한 LoRA를 원래 Chronos-2로 옮긴 결과가 F_FULL의 품질을 유지하면서 준비비를 포함한 전체 적응시간을 줄이는지, 그리고 반응 보정이 VALUE 보정만으로 얻지 못하는 추가 가치를 주는지 판정한다.
- GO는 개발 투자 신호다. 논문 PASS나 통계적 동등성 증명이 아니다.

## 2. 재사용 코드

`experiments/internal_adaptation_gap_v1_20260922/gap_screen`의 자료 준비(`prepare_frame`, `read_panel`), 모델 로드(`load_base`, pinned revision `29ec3766…`), native loss(`normalized_loss`), 지표(`score`, `scalar_primary`, `paired_bootstrap`), 체크포인트 선택(`choose_checkpoint`), 업데이트 장부(`Ledger`)를 수정 없이 import해 쓴다. 이 파일들과 이전 Jena 감사 결과의 SHA256은 봉인에 기록되고, 모든 GPU 단계 시작 때 다시 대조된다. 기준 commit은 `e751052`, 읽은 최신 HEAD는 `bf8e992`다.

`reference_core.py`와 `tests/test_reference_core.py`는 번들 그대로다. 이 파일들은 텐서 부품일 뿐이다. 실제 Chronos 연결은 `emulator_screen/`의 새 코드가 맡는다.

## 3. 자료

- 로컬 공식 Jena 2024 CSV를 다시 읽어 이전 실험과 같은 규칙(TRAIN 60 / VAL 20 / TEST 20, 일 경계, TRAIN sigma)으로 panel을 새로 만든다. 그 결과가 이전 DATA_AUDIT의 값·scale 해시, 이전 panel cache의 원점·값과 모두 같아야 진행한다. 다르면 `BLOCKED_DATA`.
- 입력 1152, 출력 144, 21계열, 그룹 단위는 한 원점의 21계열이다.
- 자료 준비 시간은 이 실행에서 실제로 잰 값을 모든 군에 청구한다.

## 4. 대리모델

- 압축 대상은 encoder block 안의 실제 `nn.Linear` 120개다. block마다 time attention q/k/v/o, group attention q/k/v/o, MLP wi/wo가 있다. 입력 patch embedding, 출력 head, LayerNorm, REG embedding은 그대로 둔다. layers.py를 읽어 이 층들이 `.weight`를 직접 읽지 않고 호출로만 쓰임을 확인했다.
- activation 통계는 TRAIN 원점 16개(TRAIN 합법 원점 index 전체에 대한 linspace, 반올림, 중복 없음)에서 수집한다. 원래 모델의 BF16 autocast forward에 미래 y 없이 넣고, 각 층 입력의 uncentered XᵀX/n을 FP32로 누적한다.
- rank는 `floor(0.5·out·in/(out+in))`이다. 768×768 층은 192, 768↔3072 층은 307이다. 저장량이 줄지 않는 층은 제외하고 기록한다.
- `reference_core.activation_factors`는 FP32로 실행한다. 공식 EMLoC `getScaling`/`SVD`(eigh, 상대 floor 1e-7, W·S의 truncated SVD, S⁻¹)와 수식이 같다.
- 대리모델은 CPU에서 pinned 모델을 만든 뒤 대상 층을 바꾸고 GPU로 옮긴다. 바뀐 층의 dense 원본은 GPU에 올라가지 않는다. forward는 `right`와 `left`를 차례로 곱하는 두 번의 GEMM이다.
- 층의 형태는 두 가지다. gamma를 학습하는 calibration 동안은 `reference_core.FactorizedLinear`를 쓴다. gamma가 동결된 모든 곳(본학습, E0 cache, 대리모델 진단)은 exp(γ)를 left에 한 번 미리 곱한 `FrozenFactorizedLinear`를 쓴다. 수학적으로 같은 분해이고 저장도 factor 그대로다. 번들 층은 매 호출마다 `low * gamma.exp()`에서 BF16을 FP32로 승격하는데, 봉인 전 실측에서 이 때문에 대리모델 step이 약 10% 느려졌다(0.401초 → 0.366초/원점). 이 비효율이 대리모델 군에만 불리하게 작용하므로 동결 구간에서 제거했다.
- LoRA는 원래 in→out 투영 옆에 붙는 additive branch(`AdapterLinear`)다. 대리모델과 원래 모델의 LoRA 이름·shape가 같다.

## 5. 비교군 구현

- 공통: seed마다 canonical LoRA 초기값을 원래 모듈 경로에 한 번 저장해 모든 군이 재사용한다. A ~ U(±1/√in)(PEFT 기본 kaiming(a=√5)과 같은 분포), B = 0이다. TRAIN 원점 schedule은 `panel.schedule(seed,128,4)`다. LoRA+는 A 1e-5, B 1.6e-4, AdamW(.9,.999, eps 1e-8, wd 0), clip 1, scheduler 없음이다. PEFT `create_loraplus_optimizer`와 파라미터별 LR 매핑이 같은지 실모델에서 확인한다. 정밀도는 FP32 master + BF16 autocast(cache off)이고, dropout과 TF32는 끄며 non-reentrant gradient checkpointing을 쓴다.
- F_FULL: 원래 모델 + 97개 모듈 LoRA+.
- F_TOP3: 마지막 3 block(9~11)의 attention 24개 + 출력 투영 1개, 합 25개 모듈만 LoRA를 붙인다. block 0~8은 `torch.no_grad`로 계산하고 경계 출력을 detach한다. block 9~11만 checkpointing한다.
- E_EMLOC: 대리모델 E_svd(gamma 0)에서 대리모델 예측에 native loss를 걸어 학습한다. 체크포인트마다 복사본에 EMLoC-style 보정(L=3, alpha/r=2 명시 포함)을 적용해 원래 모델로 옮긴다. 보정 없이 옮긴 결과는 진단으로만 저장한다.
- E_DELTA / E_VALUE_DELTA / E_RESPONSE_DELTA: 각각 E_svd / VALUE 보정 대리모델 / RESPONSE 보정 대리모델에서 `train_raw = F0_raw + E_θ_raw − E0_raw`를 raw 공간에서 만든다. 이를 원래 모델과 같은 context loc/scale로 정규화하고 asinh한 뒤, 기존 native pinball을 적용한다. loc/scale은 매 원점에서 F0 cache와 정확히 같아야 한다(다르면 `BLOCKED_MAPPING`). 학습 뒤 LoRA만 원래 F0에 그대로 복사한다.

## 6. 반응 calibration

- 각 retained SVD 성분에 multiplier exp(gamma)를 둔다(gamma0 = 0). 총 개수는 retained rank의 합이다. AdamW lr 1e-2, clip 1, 매 step 뒤 [−log 2, log 2]로 투영한다. VALUE와 RESPONSE 모두 같은 E_svd에서 시작해 16 update씩 학습한다. calibration 원점 j에는 probe j mod 4를 쓴다.
- probe: seed 92703, 두 Gaussian 방향 × 상대 크기 {1e-4, 1e-3}이다. index는 2·방향 + 크기다. 모듈마다 ‖s·B·A‖_F를 원래 W 노름에 맞춘다. plus/minus는 B 부호만 바꾼다.
- calibration forward는 gradient checkpointing을 쓰지 않는다. zero/plus/minus 세 forward를 한 번에 역전파하는데, 재계산 시점의 probe 부호가 바뀌면 안 되기 때문이다.
- calibration 단계의 모든 forward(교사 F0/F±, 대리모델 E0/E±, 초기 오차, 검사용 진단)는 교사와 대리모델 모두 FP32(autocast 없음, TF32 끔)로 대칭 실행한다. 봉인 전 실측에서 BF16 autocast로 잰 probe 반응(F⁺−F⁻)/2는 FP32 대비 상대오차가 5~60배였다. 원소의 55~81%는 정확히 0이었고, FP32 참 반응 크기는 MSE/σ² 3×10⁻⁹(1e-4 probe), 3×10⁻⁷(1e-3 probe)이었다. BF16이면 RESPONSE 보정이 반올림 잡음을 맞추게 된다. probe 크기·LR·update 수는 바꾸지 않았다. VALUE와 RESPONSE가 같은 정밀도라서 둘의 차이는 반응 loss 하나뿐이다. 본학습은 계약대로 BF16이다.
- 검사용 TRAIN 원점 8개는 calibration 16개를 뺀 나머지에서 linspace로 고정했다. probe seed는 92704이며, 결과로 아무것도 다시 고르지 않는다.
- 두 calibration 대리모델의 fingerprint는 봉인되고, 본학습 시작 때 대조된다.

## 7. 선택과 평가

- 모든 군은 체크포인트 0/16/32/64/128에서 LoRA를 원래 모델에 이식한 상태로 VAL을 평가한다. 최소 VAL을 고르고, 동률이면 더 이른 step을 고른다. E 군은 128 update를 모두 마친 뒤 대리모델을 내리고 원래 모델을 올려 다섯 체크포인트를 차례로 이식·평가한다. 두 모델을 GPU에 동시에 올리지 않는다(사전검사 단계는 예외이며 비용에 넣지 않는다).
- 대리모델 쪽 VAL(학습용 예측)은 진단으로만 기록한다. 선택에 쓰지 않는다.
- 12개 선택을 `MODEL_SELECTION.json`으로 봉인한 뒤, 모든 TEST 예측(F0, 12개 선택 모델, E_EMLOC 보정 전 진단)을 저장·해시하고 그다음에 채점한다.

## 8. 비용 장부 (결과 전 고정)

주 비용 T_total은 한 사용자가 한 군을 처음부터 적용할 때의 비용이다. 128회 고정 절차와 선택까지 모두 포함한다.

| 항목 | 청구 대상 |
|---|---|
| 자료 준비 | 전 군 |
| fit process 안의 모델 로드, 128 update, 체크포인트 저장, 이식·보정, 원래 모델 VAL 5회, 선택·export | 전 군 |
| 원래 모델 로드, activation 수집, SVD, 대리모델 구성 | E 군 |
| teacher 0 예측(calibration 원점), 초기 오차 forward, gamma 16 update | E_VALUE_DELTA |
| 위 항목 + teacher plus/minus 예측 | E_RESPONSE_DELTA |
| F0 TRAIN/VAL cache, 자기 E0 TRAIN/VAL cache (TRAIN·VAL 따로 계측) | delta 3군 |

- 같은 대리모델·cache를 물리적으로 공유해도 각 군에 전체를 청구한다. 실험 전체 실비 장부에서는 한 번만 센다.
- calibrate·caches 단계 안의 모델 재로드는 이 실행의 process 분리 때문에 생긴 것이다. cold 파이프라인은 그 모델을 이미 들고 있으므로 실비 장부에만 기록한다. 반면 fit process 안의 대리모델 로드와 이식용 원래 모델 재로드는 청구한다(후보에 불리한 쪽).
- 청구하지 않는 진단: 대리모델 쪽 VAL, 보정 전 EMLoC 이식 VAL, 검사용 반응 오차, 사전검사, TEST 채점, 그림.
- update 시간은 forward/backward/step만 잰다. 장부 fsync는 빼고, 전체 process wall time은 따로 공개한다.
- 선택된 체크포인트까지의 prefix 비용은 보조표에만 싣는다.
- 계약(§5)대로 TRAIN 합법 원점 전체(565)와 VAL(73)의 F0/E0 cache를 모두 청구한다. 그런데 seed별 schedule이 실제로 쓰는 TRAIN 고유 원점은 이보다 적고, VAL cache는 청구하지 않는 대리모델 진단에만 쓰인다. 그래서 보조값으로 "민감도 총시간"을 함께 싣는다. VAL cache를 빼고, 미사용 TRAIN 원점 몫을 원점 수 비례로 추정해 뺀 값이다. 판정은 계약 청구액으로만 한다.
- update 측정 구간 안에는 forward/backward/clip/step과 cache 읽기만 둔다. loss·gradient finite 검사와 loc/scale 대조는 측정 구간 밖에서 한다.

## 9. 판정 (결과 전 고정, `emulator_screen/decision.py`에 구현·봉인)

1. 입력: 원래 모델 TEST 주지표의 seed별 값, total_seconds = 준비비 + fit 절차 + 이식·선택 비용(§8). `reference_core.outcome(F_FULL, E_RESPONSE_DELTA, {F_TOP3, E_EMLOC, E_DELTA, VALUE_DELTA=E_VALUE_DELTA})`로 산술 분류를 먼저 계산한다.
2. EMLoC 공식 함수와의 FP32 parity(scaling 1) 또는 active-subspace 항등식이 실패하면 `BLOCKED_BASELINE`이다.
3. 산술 분류가 `GO_METHOD_SCREEN`이면 `GO_METHOD_SCREEN`이다. 단, 이 계약의 LoRA 지도에서는 EMLoC가 압축된 MLP를 보정할 수 없어 `LIMITED_BASELINE`(§11)이다. 계약상 강한 대조가 빠진 상태의 METHOD GO는 금지이므로 `HOLD_NO_AUTO_RESCUE`로 바꾼다.
4. 그렇지 않고, 대조군 중 하나라도 F_FULL 대비 품질 0.5% 이내(평균과 두 seed 모두)와 총시간 20% 이상 절감(평균, 두 seed 모두 더 빠름)을 함께 충족하면 `GO_STANDARD_ONLY`다. 계약 정의 "고정압축/EMLoC/부분층LoRA/출발예측보존만으로 같은 목표가 해결됨"에 해당한다. 반응 보정 없이 VALUE 보정만으로 해결된 경우도 "반응 보정의 필요성 미확보"이므로 여기에 넣는다.
5. 그 밖에는 산술 분류(`NO_GO_CURRENT_IMPLEMENTATION` 또는 `HOLD_NO_AUTO_RESCUE`)를 따른다. 산술 분류가 `GO_STANDARD_ONLY`인데 4의 seed별 조건을 못 넘으면 `HOLD_NO_AUTO_RESCUE`다.
6. 시간 신뢰도: 어떤 fit이든 update 시간 CV(앞 16 step 제외)가 0.2를 넘거나, 같은 군의 두 seed throughput 차이가 20%를 넘으면 `TIMING_HOLD`다. 계약 §9에 따라 이 경우 속도 근거의 GO(`GO_METHOD_SCREEN`, `GO_STANDARD_ONLY`)만 `HOLD_NO_AUTO_RESCUE`로 보류한다. NO_GO는 시간 잡음으로 구제하지 않는다. 정확도 수치는 그대로 보고한다.
7. 무결성 검산(장부·해시·선택 재계산·지표 재계산·판정 독립 재계산)이 PASS가 아니면 결과를 게시하지 않는다.
8. F_FULL 또는 다른 군이 128에서 선택되고 마지막 구간 VAL 개선이 0.2%를 넘으면 `LIMITED_BUDGET`을 표시한다. 판정 등급은 바꾸지 않고 한계로 적는다.
9. 보조: 7일 paired block bootstrap 2000회(seed 92705)를 쓴다. 날짜 단위로 21계열과 두 seed를 함께 유지한다.

## 10. 사전 고정 허용오차

| 검사 | 기준 |
|---|---|
| 공식 forward vs wrapper raw 예측, 초기 LoRA = F0, 이식 θ0 = F0, anchored train0 = F0, PEFT parity, TEST 복원 | rtol = atol = 1e-5 (이전 실험 config 값) |
| native loss parity | 1e-5 |
| full-rank factor의 W 복원(float64) | 상대 1e-8 |
| full-rank 대리모델 예측(FP32, native 공간) | 절대 1e-3 [설계] |
| 공식 EMLoC 함수 vs 이식(scaling 1) | FP32 상대 1e-5, BF16 상대 5e-2 [설계] |
| gamma 저장→동결 층 재구성: 층별 dense 사상 / native 예측(FP32) | 상대 1e-6 / 절대 1e-3 [설계] (봉인 전 실측 1.3e-7 / 3.1e-6) |

[설계] 표시는 선행 기준이 없어 이 실행에서 정한 값이다. 실패하면 허용오차를 넓히지 않고 BLOCKED로 보고한다.

## 11. EMLoC 공식 코드와의 확인·차이 (commit 332d70e)

- 확인한 것: 공식 `getCorrectedLoRA`는 A의 SVD로 basis를 바꾸고, 보정량을 `fullModule(A_ᵀ) − svdModule(A_ᵀ)`로 계산한다. `clamp`는 basis별 mean|B_| 대비 mean|correction|이 L을 넘으면 L/ratio로 줄인다. 공식 `config/peft/8B.json`은 r=8만 주고 lora_alpha는 지정하지 않아 PEFT 기본값 8, 즉 scaling 1이다. 그래서 공식의 `B_ − correction`은 scaling 1에서만 정확하다. 이번 alpha/r=2에서는 보정량을 2로 나눠야 같은 알고리즘이 되며, `reference_core.emloc_correct`가 이를 명시적으로 처리한다.
- 차이: 모델(InternVL → Chronos-2 encoder), 압축 대상(모든 encoder Linear), 정밀도(보정을 FP32로 계산, 공식은 BF16), 보정 시점(공식은 최종 LoRA 1회, 여기서는 선택을 위해 체크포인트마다), 자료, 압축비가 다르다. 원논문 VLM 결과의 재현이 아니다.
- 대조 강도의 제한(`LIMITED_BASELINE`): 공식 LoRA 대상 `language_model.model.layers.*w([o123]|qkv)`는 attention wqkv/wo와 MLP w1/w2/w3를 모두 포함한다. 그래서 공식 EMLoC는 압축된 모든 층의 차이를 LoRA 보정으로 고칠 수 있다. 이 계약은 모든 군의 LoRA 지도를 attention q/k/v/o 96개 + 출력 head로 고정하고 MLP wi/wo 24개도 압축한다. 따라서 EMLoC 보정은 MLP 압축 오차에 손을 댈 수 없다. 반면 delta 3군은 출발 예측 보존으로 0차 오차를 없애고, VALUE/RESPONSE의 gamma는 MLP까지 건드린다. 계약의 LoRA 지도를 바꾸지 않고 이 제한을 판정표(§9-3)와 보고서에 명시한다.

## 12. 예산

- main: 6군 × 2 seed × 128 = 1,536 update
- calibration: 32 update
- smoke: 16 update(main 연결 12 + calibration 연결 4)
- 합계 상한: 1,584 update

장부가 중복과 초과를 막고, 부분 fit은 자동으로 다시 실행하지 않는다. 새 자료·압축비·rank·LR·seed 탐색과 자동 후속 실행은 없다.

## 13. 실행 환경

Windows 11, RTX 4070(WDDM), 기존 CUDA venv(Python 3.11, torch 2.10+cu128, peft 0.21.0)를 쓴다. GPU 점유 검사는 WDDM의 데스크톱 C+G 항목을 제외하고, 순수 compute(Type C) 프로세스만 센다. 한국어 Windows에서의 인코딩 오류를 막기 위해 `PYTHONUTF8=1`을 설정한다.

## 14. 결과 전 투영과 예상 (봉인 전 실측, 판정에 쓰지 않음)

- 원래 모델 한 원점 forward는 0.056초, fwd+bwd는 0.313초였다. 동결 대리모델 fwd+bwd는 0.366초로 원래보다 약 17% 느렸다(계약 §4 [한계]의 "작은 GEMM 두 번이 큰 GEMM 한 번보다 느릴 수 있음"). F0/E0 TRAIN+VAL cache는 각각 약 36초로 투영된다. 따라서 후보가 시간 20% 절감 기준을 넘기는 어렵다고 결과 전에 예상한다. 계약상 이것 자체가 보고할 결과이며 설계를 바꾸지 않는다. 반응 보정의 추가 가치(기준 3)는 시간과 별도로 읽는다.
- F_TOP3는 앞 9 block의 역전파가 없어 fwd+bwd가 크게 짧다(backward 0.053초 실측). 품질 손해가 0.5% 이내면 `GO_STANDARD_ONLY`가 될 수 있다.
- 기준 3의 0.3%는 두 seed 차이(이전 Jena LoRA+ TEST에서 0.445%)와 같은 크기다. gamma는 lr 1e-2 × 16 update라 |γ| ≲ 0.16에 머물러 [−log 2, log 2] 경계에 거의 닿지 않을 것이다. 따라서 VALUE와 RESPONSE 대리모델의 차이도 작을 수 있다.
- 계획 검증(plan-critic, 봉인 전)의 지적은 COMPAT_REVISION_KO.md에 정리했다.
