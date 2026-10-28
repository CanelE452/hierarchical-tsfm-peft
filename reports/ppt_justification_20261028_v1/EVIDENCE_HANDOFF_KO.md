# 실험 근거 인계

2026-10-28. 과학적 주장과 근거의 범위를 전달하는 문서다. 발표 파일·대본·슬라이드 편집안이 아니다. 주 실험 30개와 별도 E 고정 실험 5개가 완료됐다. 이번 Appliances 비용은 측정 전 게이트 실패로 미측정이다.

## 읽는 기준과 검증 상태

- **주 실험:** 신규 30개 = 필수 20개 + 선택 10개. 여섯 family가 각각 seed 92601–92605를 완결했다. A_stop 상한 선택수가 연장 기준에 못 미쳐 연장 후보 30개는 모두 `NOT_REQUIRED`다. 중단·예산 제외·미완료 연장에 따른 잠정 결과는 없다. [M1, M2]
- **수치 검산:** 신규 30개와 재사용 기준선 20개의 예측 archive 50개를 독립 대조했다. NPZ header·origin·보호 T 경계를 먼저 확인한 뒤, 관측된 오차를 채널별 FP64 `math.fsum`으로 합산한 MSE/MAE를 재계산했다. 최대 절대차이는 1.78×10⁻¹⁵, 보호 T 수치 열람은 0건이다. 공개 파일 305개와 등록 소스 5개의 해시도 대조했다. 이 검산은 기존 `INDEPENDENT_FINAL_RESULTS_AUDIT.json`을 재사용하며, 이 문서 작성을 위해 예측 배열을 다시 열거나 학습하지 않았다. [M2]
- **별도 E 검산:** 신규 5개와 같은 Full 기준선 5개의 V 예측을 독립 재계산했으며 최대 절대차이는 1.11×10⁻¹⁶였다. Search/refit 10단계와 실제 checkpoint 30개에서 E 불변성을 확인했다. 과거 seed92601·92602 NPZ에는 scaler metadata가 없어 scaler 자체의 직접 비교에는 예외가 있다. 이 두 파일도 origin·채널·정규화된 정답·관측 mask는 다른 기준선과 정확히 같았으며, 예외를 감사 기록에 보존했다. T 수치 열람은 0건이다. [E2]
- **표본과 판정:** 새 비교는 동일 5seed, 기존 Jena 분기는 10seed다. 두 집단을 합쳐 새로운 표본수로 만들지 않는다. 아래 %는 별도 표기가 없으면 `100 × (변형의 중앙 MSE − 기준의 중앙 MSE) / 기준의 중앙 MSE`다. 양수는 변형의 오차가 더 크다. 등록 판정은 중앙값 차이를 큰 쪽 IQR과 비교한 기술통계이며, `UNCLEAR`는 동등성·효과 없음·통계적 유의성의 판정이 아니다. [M3, P1]
- **선택 절차:** A_fit 0–50%에서 탐색 학습하고 A_stop 50–60%에서 epoch를 선택·봉인한 후, A60 0–60%에서 새로 적합해 V 70–80%를 평가한다. M_cal 60–70%는 이번 epoch 선택에 사용하지 않는다. 반복 개발 V의 결과이며 독립 미래 평가가 아니다. [M4]

## 1. 외부 적응과 압축: 무엇이 연구 문제인가

**질문:** 고정된 단변량 TSFM을 재사용하면서 다변량 입력의 계산을 그 바깥에 둘 수 있는가?

**확인과 범위:** Raw의 각 F 호출은 한 시계열만 받으며, P/E/D는 원채널 간 계산을 허용한다. 고정 F의 재사용은 연구 조건이지 내부 미세조정의 열등성을 입증한 결과가 아니다. 일반적인 우회 없는 E→F→D 구조에서는 예측에 쓰는 입력이 K개 latent를 통과한다고 설명할 수 있다. 제안 방법은 직접 last-context 경로와 과거 latent 참조가 있으므로 전체 구조를 같은 조건으로 묶으면 안 된다. [H1, S1]

**미해결:** 입력 분산 손실이 예측 정보 손실과 같다는 보장은 없다. 고정 PCA/선형 출력공간의 조건부 하한도 모든 학습형 decoder의 하한이 아니다. D0 진단에서 ETT의 약 90%라는 설명은 그 하한의 평균 이동 성분 비율이지, 전체 예측 오차나 개선량의 90%가 아니다. 기존 own-channel/all-channel Ridge 진단도 Jena와 PeMS32에서 이득 방향이 달랐다. [H1, H2]

## 2. P의 성능 기여와 시간·채널 분기

**질문:** 현재 Jena 구성에서 P 전체와 각 분기를 제거하면 어떤 변화가 있는가?

**확인:** 10seed에서 Full P 대비 No P는 **+11.12%**, 시간 분기를 제거한 Channel-only는 **+8.92%**이며 등록 판정은 Full P가 낮았다. 채널 분기를 제거한 Time-only는 **+0.17%, UNCLEAR**다. 용량이 가까운 채널 대체 A2도 **+3.69%, UNCLEAR**다. [P1, P2]

**말할 수 있는 범위:** 이 Jena·모델·선택 규칙에서 P 전체와 시간 분기를 포함한 구성의 조건부 성능 근거가 있다. P는 압축 전에 과거 창과 채널을 섞을 계산 경로를 제공한다.

**미해결:** 채널 분기의 추가 이득이나 필요성은 확정하지 않았다. 시간/채널 분기의 용량은 다르고, A2도 연산·구조까지 완전히 통제한 대조가 아니다. 전체 창 LayerNorm 및 E/D의 결합이 남으므로 “시간만/채널만”은 유지된 P 분기의 이름이며 모델 전체의 축 독립성을 뜻하지 않는다. No P는 기존 cap10, Full P와 새 분기는 cap3이라는 실제 일정 차이를 함께 공개한다. [P2, S1]

## 3. F의 가치와 P×F의 조건부 비교

**질문:** 같은 P/E/D에서 고정 F를 학습형 선형 예측기로 바꾸면 좋은가? P 제거 효과는 예측기에 따라 달라지는가?

**확인:** 같은 5seed에서 F→Linear 대체는 Jena **+4.39%**(Full이 낮은 seed 4/5), Appliances **−5.08%**(Linear가 낮은 seed 4/5)였다. Jena의 P 제거는 고정 F에서 **+12.02%**, Linear에서 **−0.28%, UNCLEAR**였다. P가 없는 조건에서 F→Linear 대체는 **−7.07%**, Linear가 낮은 seed 5/5였다. seed별 차이의 차이 중앙값은 **−0.086187**이다. [M3]

**말할 수 있는 범위:** 고정 F의 이득과 P의 기여는 이 조건에서 예측기·데이터에 의존한다. Jena의 유리한 결과와 Appliances의 불리한 결과를 함께 보고해야 한다. 과거 Jena·PeMS의 1seed F 대체 근거는 역사적 자료로 남기며 현재 GPU 5seed에 합치지 않는다. [M3, H1, H3]

**미해결:** Linear는 추가 학습 파라미터 **24,624개**를 갖고, frozen F와 함수 공간·사전학습·최적화도 다르다. Jena Full cap3와 No P cap10의 일정 차이도 남는다. 따라서 사전학습만의 인과 효과, 보편적 F 우월성, 인과 상호작용 검정으로 해석하지 않는다. P의 정보 보존·F 정렬 메커니즘을 직접 측정한 결과도 아니다. 기존 고정 P/E 전체-history 선형 reader는 Jena 두 seed에서 B2가 B0보다 MSE **+2.28%, +0.71%**로 나빴으며, 이 반례를 삭제하지 않는다. [M3, H1, H2]

## 4. 직접 last-context 경로와 추가 lag는 다른 질문이다

**질문:** 원채널 현재 수준을 직접 더하는 경로와 여러 lag 변화 특징이 각각 추가 기여를 하는가?

**확인:** 직접 가산만 제거한 No-anchor는 ETTh1 **+189.35%**, Appliances **+48.63%**였다. 두 데이터 모두 Full의 MSE가 5/5seed에서 낮았다. 반면 anchor를 유지하고 추가 lag1/2/4/8/16/32를 제거한 ETTh1 Lag0는 **+0.85%, UNCLEAR**, Full이 낮은 seed 3/5였다. [M3]

**말할 수 있는 범위:** 현재 수준을 K차원 주 경로 바깥에 직접 보존하는 설계 이유와, 그 경로의 두 데이터상 성능 기여를 구분해 설명할 수 있다. 입력은 origin o에 대해 [o−L,o), 정답은 [o,o+H)다. 직접 더하는 값은 입력 끝의 전처리된 context 값이며, 결측이면 과거 관측으로 채운 값일 수 있다. 미래 정답을 직접 더하는 구조가 아니다. [H1, S1, S2]

**미해결:** No-anchor에도 과거 입력과 latent z_last 차분은 남는다. 모든 관측값 사용의 필요성이나 모든 수준 이동에 대한 강건성을 검증한 것이 아니다. 추가 lag의 이득은 ETTh1에서 불명확하며 개별 lag의 최적성·물리 주기·인과 시차는 확인하지 않았다. Lag0는 decoder 파라미터 수도 달라진다. 기존 A4는 anchor·lag·차이 특징이 함께 바뀌므로 이번 두 질문의 단독 근거로 대체하지 않는다. [M3, H1]

## 5. PCA 초기화와 Kτ는 최적성 주장이 아니다

**질문:** 초기값과 압축 폭을 어떻게 재현 가능하게 정했는가?

**확인:** 과거 Jena 2seed에서 QR 초기화의 MSE는 PCA보다 **+2.86%, +6.21%**였다. 다만 E와 D0=pinv(E)를 함께 바꿨으므로 PCA 기반 E/D 결합 초기화에 대한 제한된 지지다. [H1, H2]

Kτ는 A_fit 입력 분산을 기준으로 사전 고정한 τ=.9 규칙이다. Jena K6, Appliances K10, ETTh1/ETTh2 K4이며 표시 누적분산은 각각 **91.1%, 90.7%, 93.0%, 93.6%**다. 이것은 A_fit 0–50% 통계다. 최종 A60 0–60% refit에서는 PCA와 정규화를 새로 적합하며, 네 데이터의 K는 같았지만 누적분산 값은 달랐다. V를 보고 K를 다시 선택하지 않았다. [K1, K2]

**미해결:** τ=.9가 최적 예측 폭이거나 예측 정보 90%를 보장하지 않는다. 기존 Jena 2seed K 민감도에서 K12가 규칙의 K6보다 좋았고 이 손해는 원 보고서의 `LIMITATION`으로 남았다. 2seed 평균/범위와 10seed 중앙값/IQR을 동일한 추정량처럼 비교하면 안 된다. MLP 폭·lag 집합도 모든 대안을 최적화한 값이 아니라 고정 설계값이다. [K2]

## 6. 비교 방법 대비 성능과 일반화

**확인:** 기존 10seed 중앙 MSE는 Raw 대비 Jena **17.83%**, Appliances **18.86%**, ETTh1 **1.33%** 낮았지만, 별도 보조 데이터 ETTh2에서는 **6.81% 높았다**. C2/C3 비교는 여러 family의 필요한 추가 학습이 미완료라 Jena `TIE_PROVISIONAL`, 다른 데이터의 우위 판정도 `WIN_PROVISIONAL`이다. 이번 30개 실험은 그 최적화 공백을 해결하지 않았다. [C1]

**지표별 반례:** 기존 공개 표의 Raw 1개 결과와 Ours 10seed 중앙값을 비교하면, **MAE와 MASE는 네 데이터 모두 Raw가 낮았다.** 따라서 일부 데이터의 MSE 감소를 평균 절대오차·스케일 보정 절대오차까지 개선한 것으로 확대하면 안 된다. MSE 전용인 PAPER_COMPARISON이 아니라 원 SUMMARY의 MAE_median·MASE_median 열에 근거한 비교다. [C3]

**오차 분포의 범위:** ETTh1·ETTh2에서는 보고된 절대오차 **q50·q75·q90·q95가 모두 Raw보다 높고, 표시한 분위수 중 q99만 낮았다.** Ours 값은 각 seed에서 계산한 분위수의 seed 중앙값이다. 이는 “상위 1% 오차가 전부 줄었다”거나 같은 개별 예측점들의 오차가 개선됐다는 뜻이 아니다. 분위수 값의 비교와 개별 오차의 대응 비교를 구분한다. [C4]

**말할 수 있는 범위:** 선택한 지표·데이터·학습 예산에서의 관측 성능과 구성요소 대조를 제시할 수 있다. C2/C3는 동일 F·데이터 환경으로 이식한 어댑터이며 원 AdaPTS 전체 재현으로 부르지 않는다. [C1, C2]

**미해결:** 모든 데이터·지표·충분히 학습한 비교 방법에 대한 우월성은 미확정이다. seed 반복은 새로운 데이터 표본이 아니다. 이번 캠페인은 보호 T를 평가하지 않았으며, 반복 확인한 V와 과거 노출 이력으로 독립 테스트 주장을 할 수 없다. 충분 최적화 C2/C3, 새로운 독립 평가, 다른 모델·데이터의 재현은 별도 과제다. [M4, H1, K2]

## 7. 계산량의 구조적 이유와 실제 비용

**확인:** K<C는 F에 보내는 논리 시계열 수를 줄인다. 실제 지연·메모리는 전체 P/E/F/D 경로와 장치·batch·F 행 배치 조건에 의존한다. 과거 Jena CUDA B8·고정 48행·warmup10/timed20 조건에서 Raw 중앙 배치시간은 **95.39ms**, K6의 두 seed 중앙 배치시간 평균은 **26.59ms**였다(약 **3.59배**). 같은 조건의 peak allocated 메모리는 Raw **243.13MiB**, K6 **242.91MiB**로 거의 같았다. K6→K12의 시간은 **26.59→49.56ms**, 메모리는 **242.91→243.21MiB**였다. 이 수치는 기존 측정의 해당 조건에 한정하며 기본 128행 상한 정책 결과와 섞지 않는다. 메모리는 추론 중 PyTorch peak allocated이고 전체 GPU 사용량이 아니다. [B1]

**이번 Appliances:** 상태는 `CONTAMINATED`, 단계는 `PREFLIGHT`다. 측정 전 61표본·약 62.22초 관찰에서 background 3D p95가 **6.0449%**로 등록 한도 **5%**를 넘었다. 독립 감사가 이 실패 판정을 확인했다. 실제 추론 측정·warmup·timed trial은 모두 **0회**다. 따라서 Appliances 지연·메모리 비교는 **미측정**이며, 오염된 성능 수치가 얻어졌다는 뜻도 아니다. 조건을 완화하거나 좋은 시도만 골라 채우지 않는다. [B2, B3]

**미해결:** K 감소가 항상 전체 메모리 감소나 고정 배수의 속도 향상을 보장하지 않는다. No-F 학습이 빨랐다는 기록도 같은 프로토콜의 추론 비용 측정을 대신하지 않는다.

## 8. E를 학습하는 것의 추가 기여

**질문:** Jena K6에서 같은 P·고정 F·7-lag D·직접 last-context 경로와 초기화·선택 규칙을 유지하고, PCA로 초기화한 E만 고정하면 성능이 달라지는가?

**확인:** 기본 5/5개가 첫 시도에 완료됐다. E 고정의 cap3 선택은 1/5로 등록 연장 기준 3/5에 못 미쳐 연장 5개는 모두 `NOT_REQUIRED`다. 학습 E의 V MSE 중앙값은 **0.686402**, 고정 E는 **0.698947**로 고정 시 **+1.83%**였다. 학습 E가 낮은 seed는 **4/5**지만, 중앙값 차이 0.012545가 두 그룹의 큰 쪽 IQR 0.020333보다 작아 등록 판정은 **UNCLEAR**다. seed별 짝 차이 중앙값은 별도로 **+0.011077**이다. [E1]

**말할 수 있는 범위:** 이 다섯 seed에서는 학습 E의 중앙 MSE가 낮은 방향이 관측됐으나, E 학습의 추가 이득은 등록 기준에서 명확하지 않았다. 학습 파라미터는 E 126개를 고정해 35,717개에서 35,591개로 줄며 전체 저장 구조는 같다. P와 D는 계속 학습한다. Search의 PCA는 A_fit, fresh refit의 PCA는 A60에서 각각 계산하므로 E 불변성은 각 단계 안에서 검증한다. 두 단계의 PCA가 서로 같아야 하는 것은 아니다. [E1]

**미해결:** 이 결과는 E 학습의 필연성, 고정 E와의 동등성, PCA·K의 최적성이나 독립 테스트 일반화를 확정하지 않는다. 두 그룹의 cap은 3으로 같지만 선택 epoch는 학습 E `[1,1,1,1,.25]`, 고정 E `[1,1,1,1,3]`으로 달랐다. 같은 선택 규칙과 같은 실제 학습 길이를 구분한다. 불명확한 결과를 유리한 판정으로 바꾸기 위해 추가 seed나 학습을 반복하지 않는다. [E1]

## 인계 결론

현재 가장 명확한 새 근거는 **직접 last-context 경로의 두 데이터상 기여**다. **Jena P와 시간 분기를 포함한 구성**에는 조건부 지지가 있고, **F의 이득은 데이터·예측기 구성에 의존**한다. **채널 분기·추가 lag·E 학습의 추가 이득, P의 정보 보존 메커니즘, Kτ의 최적성**은 확정하지 않는다. 설계 이유를 설명할 수 있다는 것과 모든 부품의 실증적 필요성을 입증했다는 것은 다르다. 유리·불리·불명확·미측정 결과를 모두 보존한다.

## 근거 파일 — 저장소 상대경로

- [M1] `reports/ppt_justification_20261028_v1/EXECUTION_COMPLETION.json`
- [M2] `reports/ppt_justification_20261028_v1/INDEPENDENT_FINAL_RESULTS_AUDIT.json`
- [M3] `reports/ppt_justification_20261028_v1/RESULTS_KO.md`, `reports/ppt_justification_20261028_v1/CONTRASTS.csv`, `reports/ppt_justification_20261028_v1/INTERACTION.json`
- [M4] `reports/ppt_justification_20261028_v1/PREREGISTRATION.json`, `reports/ppt_justification_20261028_v1/ANALYSIS_VALIDATION.json`
- [P1] `reports/jena_p_branch_20261027_v1/CONTRASTS.csv`
- [P2] `reports/jena_p_branch_20261027_v1/SUMMARY.csv`, `reports/jena_p_branch_20261027_v1/RESULTS_KO.md`
- [H1] `reports/ppt_justification_20261028_v1/PPT_CLAIM_EVIDENCE_AUDIT_KO.md` — 준비 시점 감사이며, 당시 PENDING 표시는 위 최종 주 실험 결과와 구분한다.
- [H2] `reports/adapter_diagnostics_20261017_v2/QUESTIONS_AND_DECISIONS_KO.md`, `reports/adapter_diagnostics_20261017_v2/METHOD_CONTRACT.md`
- [H3] `research/informs_broad_validation_20261015_v1/priority_core_followup_20261016_v1/RESULTS_v2.csv`
- [K1] `reports/k_tau_rule_20261023_v1/K_TAU.csv`
- [K2] `reports/k_tau_rule_20261023_v1/RESULTS_KO.md`, `reports/k_tau_rule_20261023_v1/DECISION.json`
- [C1] `reports/comparative_ablation_followup_20261027_v1/PAPER_COMPARISON.csv`
- [C2] `reports/comparative_ablation_20261025_v1/DEVIATIONS_FROM_ADAPTS.md`
- [C3] `reports/comparative_ablation_followup_20261027_v1/SUMMARY.csv`
- [C4] `reports/comparative_ablation_20261025_v1/ERROR_DISTRIBUTION_SUMMARY.csv`
- [B1] `reports/memory_budget_20261018_v1/MEMORY_COST_SUMMARY.csv`
- [B2] `reports/ppt_justification_20261028_v1/cost_profile/COST_DISPOSITION.json`
- [B3] `reports/ppt_justification_20261028_v1/cost_profile/COST_PREFLIGHT_VALIDATION.json`
- [S1] `research/informs_broad_validation_20261015_v1/networks.py`, `research/informs_broad_validation_20261015_v1/training.py`
- [S2] `research/ppt_justification_20261028_v1/models.py`
- [E1] `reports/ppt_e_fixed_20261028_v1/RESULTS_KO.md`, `reports/ppt_e_fixed_20261028_v1/SUMMARY.csv`, `reports/ppt_e_fixed_20261028_v1/CONTRASTS.csv`, `reports/ppt_e_fixed_20261028_v1/ANALYSIS_VALIDATION.json`, `reports/ppt_e_fixed_20261028_v1/COMPLETION_VALIDATION.json`
- [E2] `reports/ppt_e_fixed_20261028_v1/INDEPENDENT_RESULT_CHECK.json`
