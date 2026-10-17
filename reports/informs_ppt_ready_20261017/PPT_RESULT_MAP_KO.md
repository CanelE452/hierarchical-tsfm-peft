# INFORMS 본문 결과 지도 — Priority 1 패키지

이 문서는 PPT 파일이 아닌 결과 배치·발표 판단 자료다. 작성 시점에는 표·그림 생성 및 패키지 검산이 별도로 진행 중이다. 실제 파일 상태와 포함 행은 [주 결과 CSV](PPT_MAIN_TABLES.csv), [보조 결과 CSV](PPT_ABLATION_TABLES.csv), [주장–근거 행렬](PPT_CLAIM_EVIDENCE_MATRIX.csv), [남겨야 할 빈칸](PPT_PENDING_CELLS.csv), 최종 `SOURCE_BINDINGS.json`과 `PACKAGE_VALIDATION.json`을 함께 확인한다. 이 문서 자체가 생성·게시 성공 영수증은 아니다.

주 후보는 B2 시간·채널 처리 → C→K 인코더 → 동결 Chronos-Bolt → 지연 디코더+현재값 anchor다. P/E/D는 이미 원채널 미래 MSE로 학습한다. 이번 패키지에서 새 학습·checkpoint 추론·보호 T 평가는 하지 않는다. 원 MASTER와 b2 자동화는 paused 상태로 보존한다.

## 먼저 분리할 세 코호트

```text
Jena 본문        C21 K6  L512 H48  V43  seed92601/92602
PeMS 기존 진단   C170 K43 L512 H12 V148 seed92601
PeMS 새 matched  C32 K8(B0/B2) L512 H12 V148 계획, seed92601
```

Raw는 원 C32를 F에 입력하고 G는 F를 쓰지 않는다. C32 B0/B2는 search 중단으로 final V가 없다. 기존 C170 표나 Raw/G C32 성능을 그 빈칸에 대입하지 않는다. 새 지시문 §7(5)의 학습 불충분 종료 기준에 해당하며 방법 실패의 판정은 아니다. P1-2의 긍정 gate는 충족되지 않았고 Priority 2는 실행하지 않는다. 원 마감이 지난 조건에 새 예산이나 검증된 resume CLI가 생겼다고 가정하지 않는다. 근거: [S13: PRIORITY_FINAL_JUDGMENT_KO_20261017_v2.md](../../research/informs_broad_validation_20261015_v1/priority_pems_subset_20261016_v1/PRIORITY_FINAL_JUDGMENT_KO_20261017_v2.md), [S14: B0_B2_TERMINAL_PROGRESS_20261017_v2.json](../../research/informs_broad_validation_20261015_v1/priority_pems_subset_20261016_v1/B0_B2_TERMINAL_PROGRESS_20261017_v2.json).

## 본문 1 — Background: TSFM과 B2의 질문

- 넣을 자료: [S01: METHOD_CONTRACT.md](../../research/informs_broad_validation_20261015_v1/priority_b2_validation_20261016_v1/METHOD_CONTRACT.md)의 계산 계약. TSFM을 사전학습 시계열 예측기로 소개하고 이 연구에서 Chronos-Bolt를 고정해 쓰는 이유와 C채널 의존성의 입력 구성 문제를 설명한다.
- 실제로 말할 주장: K개의 잠재 시계열을 동결 F에 넣고 원 C의 미래를 출력하는 계산 경로를 구현했다. Jena P0의 K 입력·역전파·동결 검사가 있다.
- 말할 수 없는 주장: 구현 통과가 정확도 향상·충분 학습·역사 전체 무누출을 증명한다는 주장. anchor는 원 C 정보를 우회하므로 전체 정보/메모리가 C/K로 줄었다는 주장.
- 각주: P0은 optimizer update0의 연결 검사다. 이미 사용한 개발 V와 legacy 전처리 계보를 유지한다.
- 원 파일/SHA: S01·S02·S03. 아래 전체 SHA 색인을 따른다.

## 본문 2 — Temporal Context Effect

- 넣을 표/그림: `PPT_MAIN_TABLES.csv`의 `Temporal_Context_Effect`, [JENA_B0_B2_TWO_SEEDS.svg](figures/JENA_B0_B2_TWO_SEEDS.svg).
- 실제로 말할 주장: Jena 두 seed 모두 B2의 MSE가 B0보다 낮았다. seed92602의 MAE 차이는 작다. C32 PeMS 행은 NA/PENDING이다.
- 말할 수 없는 주장: 용량·선택 refit을 맞춘 순수 시간혼합 인과효과, 두 도메인 재현, 유의성 또는 독립 일반화.
- 각주: 같은 Jena C21/K6/L512/H48, V43/43344 관측 라벨. search0/.25/1/3와 A_stop 선택, fresh refit은 B0 3/B2 1epoch. 두 seed는 같은 반복 개발 V를 쓴다.
- run ID: `postT_cpu_jena_E3_B0_k6_s92601_v1`, `postT_cpu_jena_E3_B2_k6_s92601_v1`, `priority_seed2_jena_B0_k6_s92602_v1`, `priority_seed2_jena_B2_k6_s92602_v1`.
- 원 파일/SHA: S04·S05·S06. 새 C32의 미완료 근거는 S13·S14.

## 본문 3 — Frozen TSFM Contribution

- 넣을 표/그림: `Frozen_TSFM_Contribution`, [BOLT_VS_NO_BOLT_TWO_DATASETS.svg](figures/BOLT_VS_NO_BOLT_TWO_DATASETS.svg).
- 실제로 말할 주장: Jena B2는 같은 최종 P/E/D를 고정한 R5a와 같은 구조를 no-Bolt 목적에 맞춰 별도 학습한 R5b보다 MSE가 낮았다. 기존 PeMS C170 CPU 부모는 고정 표현·디코더의 선형 head보다 MSE가 낮았다.
- 말할 수 없는 주장: PeMS joint no-Bolt도 실행됐다는 주장, 모든 직접 예측기보다 Bolt가 좋다는 주장. R5a와 R5b를 동일 가중치 표현 비교로 합치지 않는다.
- 각주: Jena H48와 PeMS H12는 별도 패널이다. PeMS 부모는 search 최대1/선택1epoch, 고정 head는 추가 A60 3epoch다. Jena R5a도 부모 학습에 추가한 head 비용이며 R5b는 새 P/E/head/D 가중치다.
- run ID: `postT_cpu_jena_E3_B2_k6_s92601_v1`, `priority_jena_R5a_fixedB2_linear_k6_s92601_v1`, `priority_jena_R5b_joint_linear_k6_s92601_v1`, `diagnostic_pems08_B2_parent_full_CPU_V148_k43_s92601_v1`, `priority_pems08_fixedB2_linear_k43_s92601_v1`.
- 원 파일/SHA: S03·S04·S05·S06. P1-2의 새 C32 joint 행은 `NOT_ENTERED_GATE_UNMET`다.

## 본문 4 — Overall Forecasting Results

- 넣을 표/그림: `Overall_Jena`, `Overall_PeMS_C170`, `Matched_PeMS_C32`; [MAIN_ACCURACY_TABLE.svg](figures/MAIN_ACCURACY_TABLE.svg).
- Method 순서: Raw → Ridge → Direct G → PCA → B0 → LoRA → Proposed B2. 각 코호트에서 제안 방법은 마지막에 둔다. 없는 행을 0으로 채우지 않는다.
- 실제로 말할 주장: Jena B2는 Ridge·Direct G·선택한 새 LoRA CPU 실행보다 MSE가 높다. 새 C32에서는 완료된 Raw/G만 숫자가 있고 B0/B2 최종 성능은 없다.
- 말할 수 없는 주장: 장치·학습량이 다른 행을 합친 엄밀한 최종 순위, C170 수치를 C32의 matched 결과로 이동하는 해석.
- 각주: LoRA 본문 run은 `postT_cpu_jena_E1_LORA_R8_k0_s92601_family_v1` 하나다. search 최대3, 선택 fresh refit1epoch이며 옛 CUDA .25 screen과 다르다. Jena Raw/PCA와 PeMS C170 일부 baseline의 saved full-V 장치는 CUDA다. CPU6 통과를 full-V CPU 정확도로 바꾸지 않는다. Ridge는 FP64 닫힌해이며 optimizer 수렴 판정 대상이 아니다.
- 원 파일/SHA: S04·S06·S07·S08·S11·S12. 조건·원 장치·recipe는 각 CSV 행을 따른다.

## 본문 5 — Accuracy–Cost Tradeoff

- 넣을 표/그림: `Accuracy_Cost`, [ACCURACY_COST_TRADEOFF.svg](figures/ACCURACY_COST_TRADEOFF.svg).
- 실제로 말할 주장: 작은 K의 B2는 새 Jena LoRA보다 warm 추론이 빠르지만 정확도는 낮다. 기존 full-history no-Bolt head는 Bolt 경로보다 훨씬 저렴하지만 해당 고정/별도 학습 비교에서 MSE가 높다.
- 말할 수 없는 주장: 전체 수명주기·cold start·학습비용·모든 대안에 대한 효율 우위, 서로 다른 시점의 기록 비율을 범용 가속비로 표현하는 것.
- 각주: CPU2, B1/B8, warm10·timed20·별도 stage20, physical input→전처리→예측/복원→inverse→CPU return. 단위는 초/배치이며 B8을8로 나누지 않는다. 모델 로딩/LoRA merge는 warm timer 밖이다. 기존 B0/no-Bolt와 새 LoRA/B2 reference의 cost provenance·RSS 상한·측정 시점 차이를 남긴다. PeMS C170 B0/G/LoRA의 동조건 CPU 비용과 새 C32 B0/B2/joint 비용은 NA다. Raw parity 실패는 FAILED다.
- 원 파일/SHA: S03·S04·S05·S06·S07·S08·S09·S10·S11·S12. B2 reference와 원 priority B2의 다른 비용 영수증을 한 run의 단일 측정으로 합치지 않는다.

## 본문 6 — Qualitative Forecast

- 넣을 그림: [QUALITATIVE_FORECASTS.svg](figures/QUALITATIVE_FORECASTS.svg). 실제 포함된 모델·cohort·단위와 재생성 검산은 그림 caption 및 최종 SOURCE_BINDINGS/PACKAGE_VALIDATION을 따른다.
- 선택 규칙: 같은 V origin ID의 교집합을 정렬해 첫/중앙/끝, 원 채널 순서의 첫/중앙/끝을 선택하는 3×3 규칙이다. 오차값을 보기 전 scope SHA `855894caf685d13fa101c6458db6ea3499a5a47a35cd188acec28c37bb5943cd`로 고정했다. H 전체와 원 관측 mask를 사용하며 결측을 채워 그리지 않는다.
- 실제로 말할 주장: 고정한 예시에서 방법별 예측 모양을 비교한다. 강한 기준선은 지표로 고르지 않고 사전 고정한 모델 유형/가용성 규칙을 따른다.
- 말할 수 없는 주장: 예시가 평균적 우열·일반화를 입증한다는 주장, 없는 PeMS joint 곡선을 고정 head로 위장하는 것. 이 그림의 PeMS는 기존 C170이며 새 C32 final forecast가 아니다.
- 각주: Jena는 원단위가 다른 채널을 각각 패널로 둔다. 기존 saved scaler로 원단위 변환이 가능한 경우에만 사용하며 그렇지 않으면 normalized라고 표시한다. 제공 불가능한 모델은 NA로 표시한다.
- 원 성능 근거/SHA: S04·S06. 이 문서 작성은 예측 배열을 읽거나 새 추론한 작업이 아니다.

## 보조 결과와 결론

[보조 결과 CSV](PPT_ABLATION_TABLES.csv)의 Jena 방법군, Beijing K6/K11, Beijing variable-K는 appendix다. Beijing K6의 B2 악화는 dataset dependence의 반례로 남긴다. variable-K는 B0 계열 부모2개/출력6개이며 주 B2의 가변 K 검증이나 독립 학습6개가 아니다. 원 파일/SHA는 S15·S16·S17이다.

결론은 새 수식보다 질문의 답으로 끝낸다. 구현은 확인됐고 Jena의 조건부 이득은 있으나 강한 기준선을 넘는 보편적 정확도–비용 우위는 미확정이다. 새 PeMS matched 셀은 학습 불충분으로 남으며 P1-2·Priority 2는 미진입이다. 이는 새 모듈·loss·graph·K를 자동 추가할 이유가 아니다.

## 원 source SHA 색인

아래 경로는 모두 공개 원 파일이며 저장 예측 배열·모델·체크포인트를 포함하지 않는다. 표의 실제 행과 원정밀도 수치는 원 CSV를 우선한다.

- [S01: METHOD_CONTRACT.md](../../research/informs_broad_validation_20261015_v1/priority_b2_validation_20261016_v1/METHOD_CONTRACT.md) — SHA256 `9e08b2528930d408ecbfc22227fb3066845cdf5c5e622e0bf4fe4d3ef645c4a1`.
- [S02: RESULT.json](../../research/informs_broad_validation_20261015_v1/priority_b2_validation_20261016_v1/diagnostics/p0_root_v1/RESULT.json) — SHA256 `d6b20943b949751ddef7c14a2b101e9bfebf06e3e2d794d324298c5aeb28a722`.
- [S03: INFORMS_METHOD_STATUS_KO.md](../../research/informs_broad_validation_20261015_v1/priority_b2_validation_20261016_v1/INFORMS_METHOD_STATUS_KO.md) — SHA256 `ea9c52fcb316ae911e848926afed53246969f604ccac633ae8e03ca2d48434f7`.
- [S04: PRIORITY_RESULTS.csv](../../research/informs_broad_validation_20261015_v1/priority_b2_validation_20261016_v1/PRIORITY_RESULTS.csv) — SHA256 `938a30cd189dd65136f52a208486a26975546bbd5621c44f758486857ef6f77e`.
- [S05: RESULTS_KO_v2.md](../../research/informs_broad_validation_20261015_v1/priority_core_followup_20261016_v1/RESULTS_KO_v2.md) — SHA256 `8db61e64f396e095e70edceeab35cf238be214de3058803de182b286d0d14e4b`.
- [S06: RESULTS_v2.csv](../../research/informs_broad_validation_20261015_v1/priority_core_followup_20261016_v1/RESULTS_v2.csv) — SHA256 `0cfd809a0345ac5a35a1135553786eb945cbe114833700108e132a7bb6266393`.
- [S07: INITIAL_FAMILY_KO_v1.md](../../research/informs_broad_validation_20261015_v1/priority_valid_family_20261016_v1/INITIAL_FAMILY_KO_v1.md) — SHA256 `7410b99221e29657000ac15afed0d05b628eee5c8088ca831b22c8ac66d08c40`.
- [S08: INITIAL_FAMILY_RESULTS_v1.csv](../../research/informs_broad_validation_20261015_v1/priority_valid_family_20261016_v1/INITIAL_FAMILY_RESULTS_v1.csv) — SHA256 `6e66eb183f784d80c0c66ae551c1f0f93a5f74b6df4900b468fdc87bf956178e`.
- [S09: B2_REFERENCE_VALIDATION_KO_v1.md](../../research/informs_broad_validation_20261015_v1/priority_valid_family_20261016_v1/B2_REFERENCE_VALIDATION_KO_v1.md) — SHA256 `362b7b970cef2783ed8af416939cb50797e06eec720e7f4a1982cafd43551206`.
- [S10: B2_REFERENCE_COSTS_v1.csv](../../research/informs_broad_validation_20261015_v1/priority_valid_family_20261016_v1/B2_REFERENCE_COSTS_v1.csv) — SHA256 `e87b84a1639322d6f8cee91f6cf2dc7c048e8e0121101002ffbcf82cdde121fa`.
- [S11: RAW_G_KO_v1.md](../../research/informs_broad_validation_20261015_v1/priority_pems_subset_20261016_v1/RAW_G_KO_v1.md) — SHA256 `0a3fafef7b6a9100802f39c750d1fcb453a0e44cda3f89f4466bf29a2186269b`.
- [S12: RAW_G_RESULTS_v1.csv](../../research/informs_broad_validation_20261015_v1/priority_pems_subset_20261016_v1/RAW_G_RESULTS_v1.csv) — SHA256 `ac65ef6839fb9578babeb89f45a29e6f8e657528241ca6814b2a1c950305ae21`.
- [S13: PRIORITY_FINAL_JUDGMENT_KO_20261017_v2.md](../../research/informs_broad_validation_20261015_v1/priority_pems_subset_20261016_v1/PRIORITY_FINAL_JUDGMENT_KO_20261017_v2.md) — SHA256 `cca4b1c945d3e1f346beae7d8b8fdad8e84b869d4b338944d1d7d99a29860798`.
- [S14: B0_B2_TERMINAL_PROGRESS_20261017_v2.json](../../research/informs_broad_validation_20261015_v1/priority_pems_subset_20261016_v1/B0_B2_TERMINAL_PROGRESS_20261017_v2.json) — SHA256 `4c73055b768a0ae9b64a2e1b4deaa98aa87e519bf982a386ba3da7b0d685c14c`.
- [S15: CORE_CONTRAST_KO_v1.md](../../research/informs_broad_validation_20261015_v1/priority_valid_minima_20261016_v1/CORE_CONTRAST_KO_v1.md) — SHA256 `36811aa67c6fedc0d82c5d2c3d57464aafca6864e83301d6483f9954df6486ba`.
- [S16: REMAINING_FAMILY_KO_v1.md](../../research/informs_broad_validation_20261015_v1/priority_valid_family_20261016_v1/REMAINING_FAMILY_KO_v1.md) — SHA256 `76666b40885a5c6662e2f02a85509bb02b826ad0792993ca7668dea8f4f8eb71`.
- [S17: MULTIK_RESUME_KO_v1.md](../../research/informs_broad_validation_20261015_v1/priority_multik_resume_20261016_v1/MULTIK_RESUME_KO_v1.md) — SHA256 `b160abf1ee29e24753d1897cac2bc1a5029ab71c07494ac1d8817a4d4a1e508d`.
