# 후속 연구의 6항목 누적 결정 장부

작성일: 2026-10-15. **이 문서는 기존 계획·실제 결과·검산·채택 보고서를 연결한 사후 추적 장부다. 이 문서를 실험 전에 작성했다거나 사전 결정을 새로 등록했다고 주장하지 않는다.** 사전 계약은 각 항목에 연결한 원 PLAN 및 실행 당시 source snapshot·선정 잠금으로 확인한다. 이 작성 작업에서 배열·원자료·모델을 열거나 새 적합·F 추론·학습·비용 측정을 하지 않았고 과학 원본도 수정하지 않았다.

원 지시의 매 반복 6항목 기록 형식을 보완한다. 초기 self/joint·dense·초기 비용·보고 D1~D5는 [기존 최종 결정 장부](../DECISION_LEDGER_FINAL_KO.md)를 그대로 따르며 여기서 중복하지 않는다. 아래 C01~C25는 후속 연구의 의사결정 묶음이다. 관련 probe·검산·비용·수선을 같은 묶음에 모았으므로 번호를 모든 process의 실행 시각 순서나 독립 실험 수로 해석하지 않는다.

`현 증거`는 확인된 기록과 남은 추정을 구분하고, `선택한 검사`는 당시 원 계획의 대조를 요약한다. `다음 결정`은 그 결과에 따른 판단과 이후에 실제 이어진 검사의 연결이며 현재 새 실험을 시작하라는 지시가 아니다. 결과에 맞춰 K·지표·공차·성공 기준을 바꾼 사전 계획으로 꾸미지 않는다.

공통 주지표는 각 비교에 등록된 관측 채널 macro 정규화 MSE이며 MAE·원단위 per-channel 진단·동일 cohort 전체 비용을 구분한다. FIT 미래를 정상 학습 label로 쓰는 것과 배포 입력에 실제 미래를 넣는 것은 다르다. 반복 노출된 DEV는 독립 holdout이 아니고, 두 minibatch seed·중첩 창·horizon·mask 진단을 독립 자료로 세지 않는다. Numeric minimum은 해당 D·known-past offset·mask·등록 cutoff의 retained 공간에 조건부이며 모든 실수/FP32 출력의 보편 하한이나 과거 예측 가능성 증명이 아니다.

큰 JSON은 [압축 계보](../publication_main_20261015_v2/EVIDENCE_ARCHIVES.json)의 원 경로→게시 `.json.gz` 사본으로 연결했다. 이는 원 JSON의 무손실 사본이며 새로운 결과가 아니다. 작은 K가 Raw 정확도와 비용을 함께 유지하는 성공 방법은 아직 얻지 못했다. 현재 continuation의 보호 TEST 금지와 과거 다른 캠페인의 별도 접근 incident를 구분하며, 이 장부로 다른 캠페인 전체 완료를 선언하지 않는다.

## C01 — JOINT gate와 독립적으로 PCA·ForeCA를 직접 비교

1. **질문**: 단순 JOINT ridge가 유리하지 않아도 작은 K의 선형 표현이 원미래를 보존하고 고정 Bolt에 쓸 이유가 있는가? 같은 E/D에서 F만 native로 바꾸면 실패 단계를 더 좁힐 수 있는가?

2. **현 증거**: [확인] 자기 채널만으로 공통 신호를 알면 JOINT 추가 이득 없이도 압축이 가능한 논리적 반례가 있어 기존 gate는 압축 가능성의 필요조건이 아니다. [추정] 실제 자료에서는 저차원 제약의 이득, 미래 보존 부족, F 좌표 호환성 또는 D 제약이 각각 남을 수 있었다.

3. **경쟁 설명**: 과거 분산을 보존하는 방향과 예측에 유용한 방향이 다를 수 있고, 낮은 spectral entropy가 실제 미래 C 보존을 보장하지 않을 수 있다. 같은 표현에서도 Bolt·단순 time head의 이용 능력과 복원 D의 제약이 서로 다를 수 있다.

4. **선택한 검사**: 승인된 gate 변경 뒤 Robin K4·Beijing K2의 PCA/ForeCA 원리 투영과 같은 FIT 관측 과거 D 적합 규칙을 비교했다. E/D를 고정한 native 두 seed·각600 update, 실제 F 추론, oracle 진단과 Raw를 포함한 20조건·120표본 전체 비용을 사전 계약으로 두었다.

5. **실제 결과**: [확인] 채택 묶음은 결정적 표현4개·D4개·native8조건과 실제 whole 비용을 완료하고 독립 검산을 통과했다. 작은 K 경로는 Raw 정확도를 유지하지 못했고 ForeCA+Bolt의 개발 MSE는 두 자료 모두 PCA+Bolt보다 컸다. 최초 미채택 시도의 CP 평가 문제와 수정 실행은 보존했으며 물리적 재실행을 독립 조건으로 부풀리지 않았다.

6. **다음 결정**: JOINT 결과로 모든 표현을 기각하지 않되 spectral entropy 개선을 예측 성공으로 채택하지 않는다. 고정 D 공간과 실제 미래 latent의 문제를 다음 진단에서 나누기로 했고, 이 묶음의 실행 완료를 Raw 대체 성공이나 독립 일반화로 쓰지 않는다.

근거: [사전 gate 재검토](../GATE_REVIEW_AND_MINIMAL_FOLLOWUP_KO_v1.md) · [채택 실제 결과](runs/actual_projection_v2/RESULT.json) · [게시 보고서](../publication_main_20261015_v1/reports/matched_projection_v1/REPORT_KO.md).

## C02 — 고정 D 공간과 미래를 아는 rank-K 공간 분리

1. **질문**: 선택된 고정 D에서 오차가 커도 모든 K차원 선형 출력 공간이 불가능하다고 할 수 있는가? 실제 E(Y) 복원과 수학적 공간 진단은 같은 의미인가?

2. **현 증거**: [확인] E(Y) 복원은 특정 E/D의 출력일 뿐 해당 D의 최적 좌표 해라고 보장되지 않는다. [추정] 고정된 과거 복원 방향이 부적절할 수도 있고, 좋은 공간이 존재해도 과거에서 그 미래 좌표를 예측하기 어려울 수 있었다.

3. **경쟁 설명**: 선택한 D의 rowspace가 미래 innovation을 놓치는 설명과, 그 공간 안에서 F가 잘못된 좌표를 예측하는 설명이 경쟁한다. 각 split의 정답으로 방향까지 고르면 낮은 오차가 가능해도 하나의 FIT-only 공간이 이후 구간에 일반화한다는 설명은 별도로 필요하다.

4. **선택한 검사**: 같은 complete-C mask에서 고정 D32의 수치 공간 최소와 E(Y)·Raw·실제 예측을 비교했다. 별도 rank-K oracle는 각 split의 알려진 미래로 방향까지 맞추고 Gram 고유값·SVD 에너지의 교차 검산을 수행하며 방법 선정에는 쓰지 않았다.

5. **실제 결과**: [확인] 고정 D의 개발 최소가 Raw보다 높은 경우가 있어도, 미래로 방향까지 정한 rank-K 진단은 6개 dataset/split cell 모두 Raw보다 낮았다. 고정 공간의 실패를 모든 K 선형 공간의 불가능성으로 확대할 근거가 없었으며 이 oracle는 배포 성능이 아니다. MAE·원단위 점수는 정규화 MSE 해의 진단값이지 각 지표의 최소라는 뜻이 아니다.

6. **다음 결정**: 공간 선택·공간의 시간 이전·미래 좌표 예측을 서로 다른 질문으로 유지한다. 미래 oracle를 학습 입력·분기·checkpoint 선택으로 옮기지 않고 실제 D 개입을 다음 대조로 삼았으며 oracle의 배포 비용을 측정한 것으로 주장하지 않는다.

근거: [rank 진단 계획](PLAN_ORACLE_RANK_DIAGNOSTIC_KO_v1.md) · [독립 rank 검산](audits/ORACLE_RANK_CEILING_AUDIT_v1.json) · [게시 보충 보고서](../publication_main_20261015_v1/reports/oracle_rank_addendum_v2/REPORT_KO.md).

## C03 — 과거 LS D를 pinv(E)로만 바꾸는 개입

1. **질문**: E와 예측 미래를 유지한 채 D만 바꾸면 예측 오차가 해결되는가? 원 D의 비직교성이 충분한 단독 설명인가?

2. **현 증거**: [확인] 원 past-LS D는 D와 E의 곱이 단위행렬에 가까워도 C 공간의 직교 사영을 보장하지 않았다. [추정] 출력 공간·scale·오차 증폭 또는 실제 latent 예측의 부적합이 각각 경쟁 설명으로 남았다.

3. **경쟁 설명**: 잘못된 복원 metric만의 문제일 수도 있고, D를 바꿔도 입력 표현이나 미래 예측이 부족할 수 있다. 원 D용으로 학습한 native head의 악화는 새 D 자체의 한계와 head 학습 불일치가 섞인 결과일 수도 있다.

4. **선택한 검사**: 기존 E/F/head를 고정하고 D를 배포 E32의 pinv로 교체하며 캐시 미래에서 두 D의 산술과 동일 mask 점수를 비교했다. 이후 새 비용 시행에서 Raw·두 표현의 원/대체 D Bolt 경로 20조건·120표본을 실제 측정했고 native는 이 비용 메뉴에 넣지 않았다.

5. **실제 결과**: [확인] 원 D 재현48건의 byte 일치를 확인했고 pinv 교체만으로 작은 K 문제는 해결되지 않았다. PCA 점수는 거의 같았고 Robin ForeCA는 TUNE 개선과 DEV 악화가 함께 나타났으며, 실제 비용 감소를 정확도 유지로 바꾸어 쓰지 않았다.

6. **다음 결정**: 비직교성 하나를 원인으로 확정하지 않고 FIT 미래에 맞춘 공간과 새 D용 matched head를 따로 검사하기로 했다. 원 head에 D만 바꾼 결과를 새 D에 최적화한 native benchmark나 새로운 표현학습의 실패로 채택하지 않는다.

근거: [D 개입 계획](PLAN_DECODER_METRIC_PROBE_KO_v1.md) · [실제 D-only 결과](runs/decoder_metric_probe_v1/RESULT.json) · [게시 보고서](../publication_main_20261015_v1/reports/decoder_metric_intervention_v2/REPORT_KO.md).

## C04 — FIT 미래 innovation의 방향으로 공간 학습

1. **질문**: 과거 PCA 대신 FIT 미래 변화가 놓이는 방향을 보존하면 같은 K·D 규칙·예측기에서 원미래 예측이 나아지는가? 공간의 미래 보존과 실제 predictor 이용 능력을 나눌 수 있는가?

2. **현 증거**: [확인] 과거 재구성·entropy·D pinv만으로 Raw 격차를 해결하지 못했고 이미 이전 actual-Y 학습도 존재했다. [추정] 과거 분산과 미래 innovation 방향의 불일치가 추가 설명일 수 있지만 큰 미래 분산이 예측 가능성을 뜻하지는 않는다.

3. **경쟁 설명**: FIT에서 찾은 미래 방향이 DEV로 이전되지 않을 수 있고, 이전되는 공간이 있어도 Bolt가 적절한 latent 좌표를 못 맞출 수 있다. 예측 불가능한 큰 innovation을 고르는 감독 목적의 한계도 단순 최적화 실패와 경쟁한다.

4. **선택한 검사**: PCA와 FIT-target 공간에 같은 pinv(E32) 규칙을 적용하고 양쪽 native를 두 seed·600 update로 새로 학습했다. FIT-only 미래 감독·원 scaler/mask·현재 C anchor·TUNE 선택을 고정하고 8학습조건과 같은 20조건·120표본 전체 비용을 수행했다.

5. **실제 결과**: [확인] Robin DEV에서는 FIT-target 공간의 최소와 실제 Bolt가 PCA보다 나빴다. Beijing에서는 새 공간 최소가 낮아도 실제 Bolt가 Raw를 넘지 못해 공간의 여지와 predictor 격차가 분리됐다. 실제8학습·비용·검산·보고를 완료했고 감사 v1의 JSON bool 직렬화 실패는 과학 재학습 없이 새 감사 v2로 수선했다.

6. **다음 결정**: 단순히 예측 목적을 넣으면 해결된다는 결론을 채택하지 않고 현재 C별 상수성과 과거 관계의 시간 변화를 다음 설명으로 조사했다. E뿐 아니라 E에서 만든 D도 FIT 미래의 간접 감독을 받으며 전체 설계가 DEV/oracle의 영향을 전혀 받지 않았다고 주장하지 않는다.

근거: [FIT-target 계획](PLAN_FIT_TARGET_SUBSPACE_KO_v1.md) · [실제 결과](runs/actual_fit_target_v1/RESULT.json) · [게시 보고서](../publication_main_20261015_v1/reports/FIT_target_comparison_v1/REPORT_KO.md).

## C05 — 과거 상수 채널의 출력 고정

1. **질문**: 과거가 정확히 상수인 C 좌표를 현재값으로 유지하면 오차가 줄어드는가? 표현 개선과 추가 decoder side information의 효과를 구분할 수 있는가?

2. **현 증거**: [확인] 이 대조 전에도 현재 C anchor가 있었지만 각 채널이 과거 전체에서 상수인지 decoder에 전달하는 조건은 없었다. [추정] 상수 채널의 불필요한 변화 예측이 일부 오차를 만들 수 있으나 미래 변화가 있는 채널을 잘못 고정할 위험도 있었다.

3. **경쟁 설명**: 출력의 불필요한 변동과 진짜 미래 변화가 경쟁한다. 정규화 FP32의 exact flat이 물리 값의 작은 변동을 반올림한 결과일 가능성도 있어 실제 물리 상수성과 같은 개념으로 볼 수 없다.

4. **선택한 검사**: 512개 normalized 과거가 현재값과 exact equal이고 실제 관측 과거가 두 개 이상인 좌표에만 고정 gate를 적용했다. 기존 7방법의 42쌍 cached 개입을 평가하고 별도 six-route whole 비용 24조건·144표본을 Raw gate 통제와 함께 측정했다.

5. **실제 결과**: [확인] CPU 개입과 실제 whole 비용·독립 검산을 완료했으나 Raw 정확도·비용 동시 채택 근거를 얻지 못했다. 비용 first8에서 Robin은 gate 대상59 C-window쌍, Beijing은0개였으므로 Beijing의 빠른 gated latency를 출력 변경의 인과적 가속 효과로 해석하지 않았다. native는 고정 출력 개입이며 새 gate native 학습이나 비용 측정이 아니다.

6. **다음 결정**: gate를 표현학습의 성공이나 strict K-only 개선으로 부르지 않고 과거에 맞춘 decoder 자체를 검사하기로 했다. 논리적 C개 flag와 실제 bool byte·count 저장을 구분하며 네트워크 bit-packed 전송 비용은 미측정으로 남겼다.

근거: [상수 gate 계획](PLAN_PAST_CONSTANT_GATE_KO_v1.md) · [실제 CPU 개입](runs/past_constant_gate_v1/RESULT.json) · [게시 보고서](../publication_main_20261015_v1/reports/past_constant_gate_v1/REPORT_KO.md).

## C06 — origin별 과거만으로 local D 적합

1. **질문**: FIT에서 고정한 D 대신 각 origin의 알려진 과거로 D[n]을 맞추면 복원·예측이 개선되는가? 그 이득이 적합 비용까지 포함해 남는가?

2. **현 증거**: [확인] 같은 E와 캐시 미래를 유지한 D 개입이 가능했고 원 FIT-target E는 이미 FIT 미래 감독을 받은 상태였다. [추정] 기간에 따라 변하는 채널 관계, 고정 D의 한계, latent 예측 오차 또는 불안정한 local 역문제가 경쟁했다.

3. **경쟁 설명**: local 과거 복원이 좋아져 미래도 나아질 수 있지만 작은 singular 방향의 증폭이나 과거 관계의 미래 불일치가 오히려 악화시킬 수도 있다. 새 D에 맞춰 재학습하지 않은 native의 결과에는 head 불일치도 남는다.

4. **선택한 검사**: 원 observed 과거만으로 centered affine LS와 고정 cutoff1e-4를 적용하고 같은 cached 미래를 새 D로 복원했다. 새 D[n]의 complete-C 공간 진단을 따로 계산했으며 whole 비용 20조건·120표본에는 매 callback의 실제 local 적합·구현 진단·입력 hash도 포함했다.

5. **실제 결과**: [확인] 3,264개 창 적합·44,304개 관측 채널 solve를 완료했고 fallback은 없었다. Robin target/Bolt는 DEV MSE가 0.564240397에서 0.373466779로 낮아져도 Raw 0.340401887보다 높고 TUNE는 악화했다. Beijing 개선도 Raw를 넘지 못했고 local 적합 비용이 추가됐다.

6. **다음 결정**: 개발 점수 개선만으로 local D를 채택하지 않고 등록된 shrinkage로 증폭을 제한하는 대조를 이어갔다. decoder로 전달되는 K×C 과거 계수라는 추가 정보와 비용을 명시하며 이 결과를 단순 표현의 개선으로 분류하지 않는다.

근거: [local D 계획](PLAN_LOCAL_PAST_DECODER_KO_v1.md) · [실제 결과 무손실 사본](../publication_main_20261015_v1/evidence/projection_followup_v1/runs/local_past_decoder_v1/RESULT.json.gz) · [게시 보고서](../publication_main_20261015_v1/reports/local_past_decoder_v1/REPORT_KO.md).

## C07 — local D shrinkage·선정 비용·새 출력 공간

1. **질문**: 원 static D 쪽으로 local LS를 축소하면 불안정성과 실제 오차가 함께 줄어드는가? 선정한 D의 공간 여지와 실제 F 오차는 어느 정도 분리되는가?

2. **현 증거**: [확인] unregularized local D의 일부 계수 크기가 컸고 TUNE/DEV 방향이 달랐다. [추정] 역문제 증폭이 경쟁 설명이지만 shrinkage가 방향·gain·공간을 함께 바꾸므로 증폭만의 원인으로 확정할 수 없었다.

3. **경쟁 설명**: 규제가 local 적합의 분산을 줄이는 설명과, 과거 관계를 바꾸어 미래에 맞지 않는 편향을 늘리는 설명이 경쟁한다. 공간이 충분해도 F 미래 좌표가 부정확하면 규제만으로는 해결되지 않을 수 있다.

4. **선택한 검사**: 사전 rho 메뉴를 같은 TUNE Bolt MSE로 고르고 네 선택을 잠근 뒤 DEV를 평가했다. 선정된 실제 backend D[n]에 새 출력 공간·E(Y) 진단을 계산하고 별도 Raw 포함 20조건·120표본 live-F whole 비용을 측정했다.

5. **실제 결과**: [확인] 양의 rho backend state88개·13,504창 적합을 검산했고 Robin 두 표현은 rho .1, Beijing PCA는 .1·target은1을 골랐다. 규제는 큰 계수와 일부 TUNE 오차를 줄였지만 선정 Bolt의 DEV는 두 자료 모두 Raw보다 높았다. 미래를 이미 아는 좌표의 여지와 실제 F 격차는 동시에 남았다.

6. **다음 결정**: DEV에서 더 좋은 unregularized rho를 사후 선택하지 않고 TUNE 잠금을 유지했다. 원 static D용 head의 결과로 native 전체를 판단하지 않도록 같은 shrinkage D에 맞춘 fresh native를 다음 비교로 삼았고 기존 다른 cohort latency를 새 비용으로 대체하지 않았다.

근거: [shrinkage 계획](PLAN_LOCAL_DECODER_SHRINKAGE_KO_v1.md) · [실제 결과 무손실 사본](../publication_main_20261015_v1/evidence/projection_followup_v1/runs/local_decoder_shrinkage_v1/RESULT.json.gz) · [선정 공간·비용 통합 보고](../publication_main_20261015_v1/reports/shrinkage_matched_native_v1/REPORT_KO.md).

## C08 — 같은 shrinkage D용 fresh native와 비용 반복

1. **질문**: 새 D를 붙인 옛 native의 부진이 head 학습 불일치 때문인가? 같은 E·rho·과거 D[n]에 head를 새로 학습하면 Bolt를 대체할 수 있는가?

2. **현 증거**: [확인] 이전 native head는 static D를 위해 학습했으며 조건부 decoder에는 최적화되지 않았다. [추정] 불일치가 일부 원인일 수 있으나 affine time head의 제한, clipping·일반화와 원 표현의 한계도 남았다.

3. **경쟁 설명**: 새 decoder용 학습이 개선을 줄 수 있는 설명과, head를 다시 맞춰도 미래 latent 예측이 어려운 설명이 경쟁한다. 빠른 native 배포와 정확도 보존 여부도 서로 다른 판단이다.

4. **선택한 검사**: 같은 NativeTorch backend의 잠재 과거·선정 D[n]을 사용해 8head를 각600 update 학습하고 모든 CP 잠금 뒤 DEV를 열었다. 새 7-route whole 비용 28조건·196표본을 수행한 뒤 같은 frozen source로 한 번 더 측정하되 두 cohort를 모두 보존했다.

5. **실제 결과**: [확인] 새8fit·4,800 update와 두 실제 비용 cohort를 완료하고 검산했다. 새 native가 Raw를 대체할 정확도 근거는 얻지 못했고 원/반복의 Raw latency 차이를 방법 이득으로 쓰지 않았다. 첫 비용 시도의 canonical physical manifest schema 실패는 0셀·0표본으로 보존하고 새 v2에서 같은 파일·SHA의 전체 member manifest를 연결했다.

6. **다음 결정**: head 재학습만으로 해결됐다고 하지 않고 원 채널 subset과 혼합 표현을 같은 global D 규칙으로 비교하기로 했다. 두 비용 cohort 중 빠른 값만 골라 보고하지 않으며 이 제한된 head의 결과를 모든 native나 Bolt의 보편적 필수성으로 확대하지 않는다.

근거: [matched head 계획](PLAN_MATCHED_SHRUNK_NATIVE_KO_v1.md) · [실제8head 결과](runs/matched_shrunk_native_v1/RESULT.json) · [원·반복 비용까지 연결한 게시 보고](../publication_main_20261015_v1/reports/shrinkage_matched_native_v1/REPORT_KO.md).

## C09 — 원 채널 subset 대 혼합 표현

1. **질문**: 채널을 섞은 좌표 때문에 F가 나빠진다면 실제 raw 채널 K개를 선택할 때는 나아지는가? 같은 global FIT D와 새 head 조건으로 이 설명을 검사할 수 있는가?

2. **현 증거**: [확인] static·local·shrinkage D와 matched head 후에도 Raw 격차가 남았다. [추정] 새로운 좌표의 F 호환성 문제와 작은 K가 버리는 정보, 과거 복원 목적의 채널 선택 실패가 경쟁했다.

3. **경쟁 설명**: raw 채널 선택이 pretrained F의 입력 분포를 보존해 유리할 수 있지만 greedy 과거 재구성 기준이 예측에 유용한 채널을 고르지 못할 수 있다. 선택 채널의 미래를 알아도 고정 D가 원 C 미래를 보존하지 못하는 경우도 가능하다.

4. **선택한 검사**: FIT observed 과거만으로 greedy 채널을 고정하고 PCA·FIT-target·subset에 같은 observed-centered global LS D 규칙을 적용했다. 12fresh head·실제 subset K-F 추론·새 D 공간 진단 및 10-route 40조건·400표본 whole 비용을 사전 계약으로 두었다.

5. **실제 결과**: [확인] 최초 Gram 잔차의 cancellation 실패를 보존하고 직접 centered LS residual 제곱으로 계산한 새 v2를 완료했다. accepted12head·7,200 update와 비용을 완료했으며 앞선 혼합8head 재실행은 exact identity를 확인해 전체 물리20fit·서로 다른12조건으로 구분했다. Subset/Bolt DEV MSE는 Robin 0.812393864·Beijing 0.866360117로 Raw보다 높았고, 고정 공간과 actual predictor의 격차를 별도로 확인했다.

6. **다음 결정**: 원 좌표를 유지하는 단순 대안도 성공하지 않았다는 제한된 결과로 남기고 lag 구조와 decoder 학습 목적을 교차하기로 했다. Gram 수치 실패를 subset 방법의 성능 실패로 혼동하지 않으며 greedy의 최적성·미래 예측 적합성은 주장하지 않는다.

근거: [subset 계획](PLAN_RAW_CHANNEL_SUBSET_KO_v1.md) · [채택 실제 v2 결과](runs/raw_subset_comparison_v2/RESULT.json) · [게시 보고서](../publication_main_20261015_v1/reports/raw_subset_comparison_v1/REPORT_KO.md).

## C10 — causal lag와 decoder 목적의 2×2

1. **질문**: 동일 E·F에서 시점별 D의 시간 제약과 과거 복원 학습 목적을 나누면 어느 단계의 오차가 줄어드는가? lag를 늘리는 것과 실제 미래 label로 D를 맞추는 것은 같은 개입인가?

2. **현 증거**: [확인] 앞선 비교는 같은 시점의 선형 복원에서 공간 손실과 미래 좌표 예측 격차를 모두 남겼다. [추정] 미래 innovation에 맞지 않는 RECON 목적, 지연 관계 누락, E/F의 부적합이 서로 다른 설명이었다.

3. **경쟁 설명**: known-past 지연을 활용하면 복원이 좋아질 수 있지만 계수 증가나 기간 이전 실패가 악화시킬 수도 있다. FORECAST D가 TUNE를 잘 맞춰도 F의 미래 예측 오차와 노출 DEV의 분포 차이가 남을 수 있다.

4. **선택한 검사**: 두 자료·세 E에서 L0/L7×RECON/FORECAST의 24 D를 적합하고 모두 잠근 뒤 split별72출력을 계산했다. 같은 complete-C mask의 joint HK 수치 공간·E(Y)·FLAT을 검산하고 별도 13-route whole 비용 52조건·676표본을 실제 측정했다.

5. **실제 결과**: [확인] 예측 목적 D가 같은 lag의 RECON보다 MSE를 줄인 비교는 TUNE 12/12·기노출 DEV 5/12였고 Raw보다 낮은 DEV 조건과 사전 정확도·비용 동시 기준은 0/24였다. joint 감사의 exact-zero 열에서 생긴 LS phantom rank 문제는 원 full HK·cutoff·공차를 유지한 reduced 검산으로 수선했다. 이는 실제 D나 예측을 고친 재학습이 아니다.

6. **다음 결정**: 목적만 바꾸면 해결된다고 채택하지 않고 같은 초기 E에서 E 고정/갱신을 비교하기로 했다. lag별 원점과 known-past prefix를 유지하며 이 선형 lag 구조를 학습된 directed causal graph나 채널 인과성으로 소개하지 않는다.

근거: [2×2 계획](PLAN_CAUSAL_LAG_OBJECTIVE_KO_v1.md) · [실제 결과 무손실 사본](../publication_main_20261015_v1/evidence/projection_followup_v1/runs/actual_causal_lag_objective_v1/RESULT.json.gz) · [게시 보고서](../publication_main_20261015_v1/reports/causal_lag_objective_v1/REPORT_KO.md).

## C11 — 같은 초기값의 E 고정·갱신과 실제 F gradient

1. **질문**: 같은 linear lag 구조에서 E까지 실제 F를 통해 갱신하면 D만 학습하는 것보다 유리한가? F 입력 gradient가 실제 E에 도달하는가?

2. **현 증거**: [확인] 기존 연구에도 actual-Y joint 학습과 F 입력 gradient가 존재했으므로 예측 목적을 처음 도입하는 실험이 아니었다. [추정] 현재 origin-reference linear 구조의 E 적응, D-only 최적화 악화, 초기화와 일반화가 경쟁했다.

3. **경쟁 설명**: E가 더 예측하기 쉬운 정보를 F에 제공할 수 있지만 E/D 공간 이동이나 직접 past-reference 경로만으로 차이가 날 수도 있다. F가 frozen이라는 사실 자체가 입력 gradient를 막지는 않으며 학습 통과가 충분 수렴을 증명하지도 않는다.

4. **선택한 검사**: 동일 PCA/FORECAST-LS 초기값의 L0/L7×E고정/갱신×두 자료·두 seed 16조건을 600 update로 학습했다. 먼저 실제4probe에서 gradient 경로·CP0 NumPy parity·canonical32/B4 차이를 확인하고 16잠금 뒤 DEV 및 36조건·324표본 whole 비용을 수행했다.

5. **실제 결과**: [확인] 본 학습16fit·9,600 update·80CP와 기술 probe4update를 구분해 완료했다. E 갱신은 matched E 고정보다 TUNE와 DEV 각각8/8에서 MSE를 낮췄지만 Raw보다 DEV가 낮은 조건은 0/16이었다. 저장 산술·점수·선정·동결 상태와 비용을 검산했으며 모든 backward/optimizer를 독립 재생한 것은 아니다.

6. **다음 결정**: E 적응의 조건부 효과는 남기되 F gradient의 기여와 D 초기화/공간 이동을 따로 조사하기로 했다. 단순히 학습을 더 하면 Raw가 해결된다고 판단하지 않았고 이 단계의 실제 whole 비용을 이후 다른 선정 상태에 붙이지 않는다.

근거: [E 적응 계획](PLAN_LINEAR_E_ADAPTATION_KO_v1.md) · [실제16조건 결과](runs/linear_e_train_v1/RESULT.json) · [게시 보고서](../publication_main_20261015_v1/reports/linear_e_adaptation_v2/REPORT_KO.md).

## C12 — F 입력 gradient만 끊는 ablation

1. **질문**: E 적응의 이득 중 F 입력으로 되돌아오는 gradient가 이 구성에서 도움이 되는가? F 예측값의 사용과 F 입력 미분의 사용은 분리할 수 있는가?

2. **현 증거**: [확인] 앞 probe는 전체 E gradient뿐 아니라 F 경유와 직접 past 경로를 구분했지만 성능상 필요성을 입증하지 않았다. [추정] 직접 경로만으로 충분한 학습과 F 경유 경로의 추가 기여가 경쟁했다.

3. **경쟁 설명**: stop-gradient가 불필요한 F 미분을 줄여 유리할 수 있고 반대로 E가 F 친화적인 방향을 배우는 데 필요한 경로를 제거할 수도 있다. 두 결과 모두 F 예측값 자체나 모든 TSFM의 필요성을 답하는 것은 아니다.

4. **선택한 검사**: L0의 같은 초기 E/D·seed·600update에서 F 출력만 detach하고 직접 과거 경로와 D 학습을 유지했다. 새4fit·20CP와 기존 full-gradient4조건을 비교하고 네 zero-update probe, 잠금 뒤 DEV 및 별도20조건·100표본 whole 비용을 두었다.

5. **실제 결과**: [확인] 새2,400 update와 4잠금·선정 FIT/DEV8출력을 완료했다. stop-gradient는 full-gradient보다 TUNE·DEV 모두4/4에서 MSE가 높았고 양쪽8조건 중 Raw보다 DEV가 낮은 경우는0이었다. probe의 chain-rule·D gradient·CP0 byte·순서 검산과 실제 비용을 완료했다.

6. **다음 결정**: 현재 L0 절차에서 F gradient가 유리했다는 제한된 판단으로 남겼다. 모든 native의 불가능성이나 충분 수렴을 선언하지 않고, 선정된 E/D의 출력 공간과 초기화 민감도를 다음 검사로 연결했다.

근거: [gradient ablation 계획](PLAN_F_INPUT_GRADIENT_ABLATION_KO_v1.md) · [실제 새4fit 결과](runs/f_input_gradient_ablation_v1/RESULT.json) · [게시 보고서](../publication_main_20261015_v1/reports/f_input_gradient_ablation_v1/REPORT_KO.md).

## C13 — RECON D 초기화 민감도

1. **질문**: FORECAST-LS 초기 D 때문에 joint E/D 학습이 불리했는가? 같은 PCA E에서 RECON D로 시작하면 결과와 선정 출력 공간이 달라지는가?

2. **현 증거**: [확인] E 갱신·F gradient가 유리해도 원 Raw 격차는 남았고 선정 D의 공간 이동이 경쟁 설명이었다. [추정] 초기화에 의한 최적화 경로 차이와 학습 후 출력 공간·기간 이전의 악화를 나눌 필요가 있었다.

3. **경쟁 설명**: 다른 초기 D가 더 나은 경로를 열 수도 있지만 같은 actual-C 손실이 TUNE에 맞추며 DEV 공간을 악화시킬 수도 있다. 초기화 민감도와 단순한 update 부족은 같은 설명이 아니다.

4. **선택한 검사**: Robin L7의 두 seed에서 RECON 초기 E/D를 600update씩 새로 학습하고 원 FORECAST-init 두 조건을 재사용했다. CP0 산술·같은 batch 순서·선정 잠금과 새 선정 TUNE/DEV4공간을 분리해 검산했다.

5. **실제 결과**: [확인] 새2fit·1,200 update·10CP를 완료했다. RECON-init은 이전 FORECAST-init보다 DEV가 낮아도 원 RECON CP0와 Raw보다 두 조건 모두 높았고, 새 D의 DEV numeric minimum도 Raw보다 높았다. 감사 v1의 추가 bytes record 동일성 실패는 원 과학 출력 그대로 새 감사 v2에서 수선했다.

6. **다음 결정**: 초기화만 바꾸면 해결된다고 하지 않고 D를 고정한 E-only를 다음 대조로 정했다. 이 계획은 새 비용을 현재 단계에서 제외하고 개선 확인 뒤의 후속 측정으로 분리했으므로 새 RECON-init 배포 비용은 미측정으로 남겼다.

근거: [초기화 계획](PLAN_RECON_INIT_SENSITIVITY_KO_v1.md) · [실제2fit 결과](runs/recon_init_sensitivity_v1/RESULT.json) · [게시 보고서](../publication_main_20261015_v1/reports/recon_init_sensitivity_v1/REPORT_KO.md).

## C14 — RECON D를 고정한 E-only

1. **질문**: joint 학습의 D 이동을 막고 E만 바꾸면 DEV 악화가 줄어드는가? 같은 D라도 E의 known-past offset 변화가 남는가?

2. **현 증거**: [확인] RECON-init joint는 TUNE를 낮췄지만 DEV와 선정 D 공간에서 원 CP0보다 불리했다. [추정] D 공간 이동이 원인일 수도 있고 E 변화·최적화·시간 이전의 문제가 독립적으로 남을 수도 있었다.

3. **경쟁 설명**: 고정 D가 공간의 과적합을 줄이는 설명과 E-only가 예측 손실을 충분히 줄이지 못하는 설명이 경쟁한다. D가 같아도 E가 바꾼 과거 reference 때문에 numeric minimum이 같아진다고 보장할 수 없다.

4. **선택한 검사**: 같은 fresh RECON 초기값·순서의 Robin L7 두 seed를 600update씩 학습하되 D.requires_grad=False·grad 없음·CP별 bytes 불변을 검사했다. E-only clipping 범위를 명시하고 joint 두 결과와 실제 CP0·Raw·새 공간4개를 비교했다.

5. **실제 결과**: [확인] 새2fit·1,200 update·10CP를 완료했고 D 고정 기록을 검산했다. E-only는 joint보다 DEV 두 조건 모두 낮았지만 TUNE는 높았고 원 RECON CP0·Raw도 이기지 못했다. DEV numeric minimum은 Raw보다 낮아도 실제 F 출력의 성공을 뜻하지 않았다.

6. **다음 결정**: D 이동이 일부 악화를 줄이는 조건은 남기되 이를 Raw 대체 성공으로 채택하지 않았다. 이 E-only 학습 단계의 단독 배포 비용은 측정하지 않았고 같은 고정 E/D의 native 비교를 진행했다. 이후 C15의 matched L7 및 C16의 q 전체 비용에서는 이 E-only 두 상태와 원 RECON 상태를 실제로 측정했다.

근거: [고정 D 계획](PLAN_FIXED_RECON_D_E_ADAPTATION_KO_v1.md) · [실제 E-only 결과](runs/fixed_recon_d_e_adaptation_v1/RESULT.json) · [게시 보고서](../publication_main_20261015_v1/reports/fixed_recon_d_e_adaptation_v1/REPORT_KO.md).

## C15 — 학습 후 공간 진단과 같은 L7 E/D의 matched native

1. **질문**: 학습한 D가 Raw에 도달할 수 없는 공간을 만들었는가, 아니면 실제 미래 좌표가 그 여지를 이용하지 못했는가? 고정 RECON D와 E-only 상태에서 제한된 native가 Bolt를 대체하는가?

2. **현 증거**: [확인] full-gradient·stop-gradient 선정 상태의 공간 진단24cell은 Robin에서 TUNE와 DEV의 공간 여지가 달랐고 Beijing에는 공간 여지와 실제 F 오차가 함께 남았다. [추정] 공간의 시간 이전과 predictor의 좌표 오차는 단일 원인으로 합칠 수 없었다.

3. **경쟁 설명**: 선택된 E/D 공간 제약이 지배적일 수 있고 공간 안에 여지가 있어도 현재 predictor가 이용하지 못할 수 있다. E1/E2가 Bolt를 통해 최적화된 이력과 affine head의 한계·clipping·선정 CP가 native 비교에 영향을 줄 수도 있다.

4. **선택한 검사**: 선행 선정 E/D에 같은 complete-C joint 진단을 적용하고, Robin의 RECON CP0·E-only 두 상태 각각에서 head 두 seed를 fresh 학습했다. 6fit·3,600 update·30CP·6잠금 뒤 DEV, 현 D의 cached Bolt 재복원9개와 새 공간12cell·live-F/native whole 비용20조건·100표본을 비교했다.

5. **실제 결과**: [확인] native는 같은 E/D의 Bolt보다 TUNE·DEV 모두6/6에서 높았고 Raw보다 DEV가 낮은 조건은0/6이었다. 세 상태의 complete-C TUNE numeric minimum은 모두 Raw보다 높았지만 DEV minimum은 낮아 구간별 공간 제약과 실제 예측 격차를 함께 남겼다. native는 더 저렴했어도 정확도 격차가 남았으며 보고 v1의 state label 충돌은 원본 보존 후 새 v2에서 수선했다.

6. **다음 결정**: 특정 affine512→48 head의 대체 근거 부족으로 판단하고 Bolt의 보편적 필수성을 선언하지 않았다. 먼저 현재 E/D 공간을 바꾸지 않는 작은 F 미래 scale/bias 보정으로 설명되는 오차를 조사했으며, 선행 공간 계산의 두 수학 경로 교차를 독립 F 재실행이나 제3 외부 재현이라고 부르지 않는다.

근거: [matched L7 계획](PLAN_MATCHED_L7_NATIVE_KO_v1.md) · [선행 학습 후 공간 진단 무손실 사본](../publication_main_20261015_v1/evidence/projection_followup_v1/runs/learned_output_space_v1/RESULT.json.gz) · [matched L7 게시 보고서](../publication_main_20261015_v1/reports/matched_l7_native_v2/REPORT_KO.md).

## C16 — 고정 L7의 작은 잠재 미래 scale·bias 보정

1. **질문**: E/D/F를 그대로 두고 F 미래의 전역 scale·bias만 바꾸면 남은 오차를 줄일 수 있는가? 학습되지 않는 identity와 직접 solver를 함께 두면 이 작은 family의 적합·이전을 구분할 수 있는가?

2. **현 증거**: [확인] 세 고정 L7 공간에서 native는 더 부정확했고 Bolt도 Raw를 이기지 못했다. [추정] 작은 전역 calibration으로 설명되는 오차일 수도 있고 horizon 상호작용·좌표 혼합·공간의 한계가 남을 수도 있었다.

3. **경쟁 설명**: FIT 잔차의 scale/bias 편향이 이후에도 안정적으로 남는 설명과 FIT에만 맞는 작은 보정이라는 설명이 경쟁한다. FP64 surrogate와 실제 FP32 배포 연산의 차이를 optimizer 실패나 정확한 FP32 최적성으로 혼동할 수도 있었다.

4. **선택한 검사**: 세 상태×SCALE_ONLY/SCALE_BIAS의 6fit에 7ridge와 identity를 두고 실제 CUDA FP32 TUNE MSE로 선택했다. 6잠금 전후를 나눠 96 FIT/TUNE 평가·선정 DEV6개를 저장하고 독립 NumPy q/lag parity 및 live-F whole 비용20조건·100표본을 검산했다.

5. **실제 결과**: [확인] 6조건 중5개는 identity였다. E2 SCALE_ONLY의 작은 TUNE 감소 약0.0114%는 DEV 약0.0412% 악화와 함께 나타났으며 Raw보다 DEV가 낮은 선정 q는0/6이었다. 모든 family의 FIT 후보 최솟값은 identity보다 낮아 적합 자체와 이전 실패를 구분했고 실제 FP32와 surrogate 차이를 보존했다.

6. **다음 결정**: 이 작은 global family로 Raw 격차를 해결했다고 채택하지 않았고 같은 공간의 복잡한 보정만 늘리지 않았다. 비용에서 다섯 identity 경로를 서로 다른 학습 성공이나 속도 우위로 세지 않으며 정보·공간을 고정한 F 좌표 호환성과 다른 정보 방향을 다음 질문으로 삼았다.

근거: [잠재 보정 계획](PLAN_FIXED_L7_LATENT_CALIBRATION_KO_v1.md) · [실제6조건 결과](runs/fixed_l7_latent_calibration_v1/RESULT.json) · [과학·실제 비용을 합친 게시 보고서](../publication_main_20261015_v2/reports/fixed_l7_latent_calibration_v1/REPORT_KO.md).

## C17 — 같은 정보·old D의 고정 orthogonal Q

1. **질문**: 표현 정보와 decoder를 그대로 두고 F가 보는 좌표만 바꾸면 예측이 달라지는가? 재구성 항등성과 비선형 F를 통과한 좌표 호환성은 같은가?

2. **현 증거**: [확인] joint E/D 적응은 보존 정보·공간·scale을 함께 바꿨으므로 순수한 F 좌표 효과가 아니었다. [추정] F는 동일 정보의 서로 다른 기저에 민감할 수 있지만 그 민감도가 유리한 성능으로 이어진다는 보장은 없었다.

3. **경쟁 설명**: dense mixing의 F 분포 부적합과 단순 배치·좌표 연결 오류를 분리해야 했다. sign 대칭·permutation 대칭의 여부와 실제 predictive utility도 각각 다른 설명이다.

4. **선택한 검사**: Robin 세 고정 상태에서 identity·permutation·sign·사전 seed의 dense Q를 모두 비교했다. cached future의 analytic transport를 먼저 검산하고 실제 F(zQ)를 새로 실행한 뒤 Q의 역변환과 원 old-past/D로 복원했으며 DEV로 Q를 고르지 않았다.

5. **실제 결과**: [확인] identity·permutation은 기존 경로와 byte 차이0이었고 sign/dense는 실제 출력을 바꾸었으나 DEV의9개 비교 모두 원 median 경로보다 나빴다. 이는 고정 정보 안의 좌표 민감도 관측이며 압축의 정보 보존 성공이나 모든 Q의 실패가 아니다.

6. **다음 결정**: arbitrary Q 실패만으로 학습 Q를 생략하지 않고 같은 정보·공간에서 소수 theta를 FIT actual-C로 갱신하기로 했다. 이 고정 Q 과학 실행의 자료를 새 Q 전체 deployment latency로 대신 쓰지 않고 비용 미측정 범위를 유지했다.

근거: [최상위 방법론 계획의 후보 A](PLAN_METHODOLOGY_COMPLETION_KO_v1.md) · [실제 고정 Q 결과](runs/orthogonal_basis_controls_v1/RESULT.json) · [게시 진단 보고서](../publication_main_20261015_v2/reports/methodology_diagnostics_v1/REPORT_KO.md).

## C18 — theta 6개의 예측 목적 Q 학습

1. **질문**: 임의의 Q와 달리 actual-C loss로 학습한 작은 회전이 FIT/TUNE 잔차를 줄이고 DEV로 이전되는가? E/D 정보와 출력 공간을 바꾸지 않는 적응이 유용한가?

2. **현 증거**: [확인] 고정 sign/dense는 모두 악화했지만 임의 좌표 실패가 전체 학습 family의 실패를 뜻하지 않았다. [추정] 작은 Q의 최적화 가능성과 기존 고정 D의 TUNE 공간 제약이 동시에 남았다.

3. **경쟁 설명**: FIT가 요구하는 기저를 학습해 F 호환성이 나아질 수 있으나 그 이득이 시간 구간을 넘어 유지되지 않을 수 있다. 낮은 TUNE와 마지막 CP 선택이 충분 수렴·global optimum을 뜻하는 것도 아니다.

4. **선택한 검사**: 같은 세 상태×두 minibatch 순서의 6조건에서 Q=exp(skew(theta))의 theta6개만 각600 update 학습했다. F 입력·inverse-Q gradient를 분리하고 CP0/100/200/400/600의 TUNE 선택 및 모든6잠금 뒤 DEV를 검사했다.

5. **실제 결과**: [확인] 6fit·3,600 update를 완료했고 TUNE는6/6, DEV는2/6에서 원 baseline보다 낮았지만 Raw median보다 좋은 DEV는0/6이었다. 여섯 조건 모두 CP600을 골랐으므로 이 기록만으로 충분한 학습이나 수렴을 확정하지 않았다.

6. **다음 결정**: 회전 자체를 새 기여로 채택하거나 같은 family의 seed/LR sweep을 무한히 늘리지 않았다. Q 학습 자체의 새 whole 비용은 계획 범위에 포함하지 않았으며 정보 방향·K capacity·점요약의 독립 질문을 이어갔다.

근거: [Q 학습 계획](PLAN_Q_LEARNING_KO_v1.md) · [실제6학습 결과](runs/orthogonal_q_learning_v1/RESULT.json) · [게시 진단 보고서](../publication_main_20261015_v2/reports/methodology_diagnostics_v1/REPORT_KO.md).

## C19 — 같은 full9 분위 출력의 점요약

1. **질문**: MSE를 평가하면서 median만 쓰는 점요약이 일부 격차를 설명하는가? 같은 요약 메뉴를 Raw에도 적용했을 때 압축 경로에 고유한 이득이 남는가?

2. **현 증거**: [확인] 기존 wrapper는 0.5 분위의 앞48시점을 썼고 설치 pipeline의 mean 반환을 진짜 조건부 평균으로 간주할 수 없었다. [추정] 점요약 목표의 불일치와 E/D·F의 정보/공간 문제가 경쟁했다.

3. **경쟁 설명**: 여러 분위의 가중 평균 proxy가 MSE를 줄일 수 있지만 제한된 tail 가정·crossing·음의 D 때문에 실제 평균이나 원 C 분포를 식별하는 것은 아니다. Raw도 같은 이득을 얻는다면 압축 호환성의 해결로 해석할 수 없다.

4. **선택한 검사**: Raw와 세 고정 E/D의 모델 full9×64 출력을 직접 저장하고 MEDIAN·NINE_MEAN_PROXY·TAIL_CLAMPED_PROXY를 고정 비교했다. 네 경로의 24 FIT/TUNE 점수와4잠금 뒤 DEV의 사전12진단을 계산하고 NumPy 가중합·old lag decode를 독립 검산했다.

5. **실제 결과**: [확인] 같은 메뉴에서 선정한 Raw는 fresh median DEV MSE 0.340401882에서 0.334882911로 낮아졌지만 압축 세 경로는 자기 median보다 모두 악화했다. 세 요약을 실제 조건부 평균이나 C-channel probabilistic forecast로 해석하지 않았고 fit·optimizer는0이었다.

6. **다음 결정**: 요약을 바꾸는 것만으로 압축 문제를 해결했다고 하지 않고 선택된 Raw를 공정한 통제로 유지했다. 이 과학 단계는 whole-deployment latency가 아니며 선정 summary의 추가 전체 비용은 미측정으로 남겼다.

근거: [점요약 계획](PLAN_QUANTILE_POINT_SUMMARY_KO_v1.md) · [실제 full9·요약 결과](runs/quantile_point_summary_v1/RESULT.json) · [게시 진단 보고서](../publication_main_20261015_v2/reports/methodology_diagnostics_v1/REPORT_KO.md).

## C20 — 예측 cross-moment·K capacity·native·직접 ridge

1. **질문**: 과거 분산 대신 예측 cross-moment 방향을 선택하거나 K를 두 배로 하면 정보 보존과 실제 예측이 달라지는가? 같은 E/D의 native와 D를 우회한 직접 readout이 실패 위치를 좁히는가?

2. **현 증거**: [확인] 고정 K4 공간에는 TUNE 제약이 있었고 Q는 그 공간을 바꾸지 않았다. [추정] 방향 선택·K capacity·F 이용 능력·D 제약과 유한 표본의 고차원 규제가 서로 다른 설명이었다.

3. **경쟁 설명**: cross-predictive 방향이 더 유용할 수 있지만 제한된 FIT나 관측 mask 때문에 이후 구간에 불리할 수도 있다. K 확대가 정보를 늘려도 F/native의 추정과 D 복원이 실패할 수 있고 직접 ridge의 실패는 alpha 범위 제한 때문일 수도 있다.

4. **선택한 검사**: 두 자료×PCA/CROSS×기본K/2K의8 E/D와 실제 F를 비교하고 같은 공간의 native16개를 각600 update 학습했다. Raw current/history와 encoded-history 직접 ridge12fit·7alpha, 모든 선택 잠금 뒤 DEV 및 새19-route whole 비용76조건·380표본을 수행했다.

5. **실제 결과**: [확인] PCA의 K 확대는 일부 F 오차를 낮췄지만 모든 frozen-F 방향/K는 Raw DEV보다 높았다. 같은 D에서의 native 이득은 제한적이었고 직접 ridge history10경로 모두 alpha10 상단을 골라 충분히 넓은 규제 탐색으로 볼 수 없었다. Robin PCA K8의 E(Y) 진단에는 낮은 오차의 여지가 있어도 실제 F 성공이나 과거 예측 가능성이 증명된 것은 아니다.

6. **다음 결정**: 원 ridge 메뉴의 경계 선택에 대해서만 등록된 alpha 확장을 시행하고 현재 공간·예측기 전체를 기각하지 않았다. K 효과·방향 효과·F/native 차이를 합치지 않으며 v1 전체 비용은 v1 선정 상태의 측정으로 고정했다.

근거: [예측 방향 계획](PLAN_PREDICTIVE_CROSS_MOMENT_KO_v1.md) · [native·직접 ridge 대조 계획](PLAN_NEW_DIRECTION_NATIVE_BASELINES_KO_v1.md) · [실제 결과·비용을 연결한 게시 보고서](../publication_main_20261015_v2/reports/new_direction_comparison_v1/REPORT_KO.md).

## C21 — ridge 규제 상단의 등록 확장

1. **질문**: history10경로가 alpha10을 고른 것이 단순한 규제 탐색 범위 부족 때문인가? mean과 현재 C 제어까지 포함해도 같은 판단이 남는가?

2. **현 증거**: [확인] 원12fit 중 Raw-current 두 경로만 .1을 골랐고 history10은 모두 grid 상단이었다. [추정] 더 큰 shrinkage가 일반화에 유리할 수 있으나 표현·feature·모델 한계가 독립적으로 남았다.

3. **경쟁 설명**: 고차원 slope의 분산을 충분히 줄이지 못한 설명과 선형 history readout 자체의 부적합이 경쟁한다. 무한 규제의 해석상 끝점인 intercept-only와 실제 유한 alpha의 최적성도 구분해야 했다.

4. **선택한 검사**: history10경로에 alpha100/1000/10000과 INTERCEPT_ONLY·CURRENT_C_ANCHOR를 추가했다. 원70 FIT/TUNE 후보를 재사용하고 새50후보·100 FIT/TUNE 평가, 모든10잠금 뒤 새/재사용 DEV를 원 계약대로 분리했다.

5. **실제 결과**: [확인] Robin5경로는 alpha10을 유지했고 Beijing5경로는100으로 바뀌었다. 새 상단10000 선택과 zero-slope 선택은 각각0/10이었지만 Raw 정확도 격차는 남았고 유한 메뉴를 global optimum으로 주장하지 않았다.

6. **다음 결정**: 규제를 더 키우면 모든 실패가 해결된다는 설명은 채택하지 않았다. 바뀐 계수의 새 deployment 비용은 미측정이며 이전76셀 비용을 새 정확도에 붙이지 않고, 다음 decoder 개입을 별도 등록했다.

근거: [규제 확장 계획](PLAN_RIDGE_REGULARIZATION_EXTENSION_KO_v1.md) · [실제 확장 결과](runs/ridge_regularization_extension_v1/RESULT.json) · [게시 보고서](../publication_main_20261015_v2/reports/ridge_regularization_extension_v1/REPORT_KO.md).

## C22 — 동일 추가 slope의 STATIC/RAMP와 비용 반복

1. **질문**: 같은 E/F·old RECON D에서 horizon에 관계없이 같은 크기로 보정해야 한다는 제약을 한 번 풀면 도움이 되는가? 같은 자유 slope 수의 STATIC과 고정 RAMP 차이가 Raw까지 남는가?

2. **현 증거**: [확인] 기존 학습·Q·점요약·예측 방향에서 Raw 격차가 남았고 미래를 아는 K8 진단에는 여지가 있었다. [추정] decoder의 horizon 제약, 실제 F 좌표 오차, 보정의 일반화와 크기가 경쟁했다.

3. **경쟁 설명**: 멀리 갈수록 보정을 키우는 ramp가 유리할 수 있지만 유효 rank·규제 prior가 달라져 오히려 악화할 수도 있다. Raw에도 같은 보정이 유리하면 그 효과는 압축 고유 이득이 아니며 FLAT과 F의 차이는 별도로 읽어야 한다.

4. **선택한 검사**: 두 자료의 Raw/PCA기본K/2K×F/FLAT×STATIC/RAMP의24fit에 공통 actual-F FIT RMS·같은 observed residual 목적·10alpha·ZERO를 사용했다. 24잠금 뒤 DEV를 열고 실제 18-route whole 비용72조건·360표본과 같은 frozen source의 반복72/360을 모두 보존했다.

5. **실제 결과**: [확인] 240alpha·24ZERO·528 FIT/TUNE·24선정 DEV를 완료했다. 같은 class의 F는 FLAT보다 TUNE/DEV 모두12/12에서 낮았지만 PCA RAMP 네 조건은 자기 무보정 F보다 DEV가 높고 F 기반 작은 K 보정8조건 중 Raw를 이긴 경우는0이었다. 비용감사 v1의 지역 이름 충돌 실패 두 개는 보존하고 새 v2 두 PASS로 수선했으며 과학 출력·선정·타이머 표본은 바꾸지 않았다.

6. **다음 결정**: Raw Beijing 보정의 이득을 압축 고유 기여로 채택하지 않았고 비용의 빠른 cohort만 고르지 않았다. 원/반복 B8 절감과 B1의 불일관성을 남기고, 보정 방향과 크기 및 시간 이전을 추가 진단하기로 했다.

근거: [horizon 대조 계획](PLAN_HORIZON_DECODER_DIAGNOSTIC_KO_v1.md) · [실제 결과 무손실 사본](../publication_main_20261015_v2/evidence/projection_followup_v1/runs/horizon_decoder_diagnostic_v1/RESULT.json.gz) · [원·반복 비용을 포함한 게시 보고서](../publication_main_20261015_v2/reports/horizon_decoder_diagnostic_v1/REPORT_KO.md).

## C23 — 보정 방향 interaction과 correction energy 분해

1. **질문**: TUNE의 보정 이득이 DEV에서 사라지는 것은 방향이 옮겨지지 않아서인가, 보정 크기가 커서인가? 같은 mask의 오차 항등식으로 두 양상을 분리해 설명할 수 있는가?

2. **현 증거**: [확인] STATIC/RAMP의 선정 결과에는 TUNE 개선과 DEV 악화가 함께 있었다. [추정] 과적합·공변량 변화·표현 부족·최적화 문제 중 어떤 것이 원인인지는 오차 분해만으로 확정할 수 없었다.

3. **경쟁 설명**: baseline 잔차와 보정의 interaction이 불리하게 바뀌는 경우와 유리한 interaction보다 correction energy가 큰 경우가 경쟁한다. 이 두 관측을 곧바로 독립적인 인과 원인이나 임의 개입의 효과로 읽는 설명은 배제해야 했다.

4. **선택한 검사**: 실제 FP32 endpoint의 baseline b·선정 출력 p에서 c=p-b를 잡고 동일 관측 mask·macro 가중치로 ΔMSE의 interaction과 energy를 계산했다. 기존 선정·예측을 바꾸지 않고 전체·horizon·구간·per-C의 저장 표와 합계를 검산했다.

5. **실제 결과**: [확인] 72전체·3,456h·216구간·52,416 per-C 진단을 완료했다. Robin K4 STATIC은 interaction의 부호가 TUNE와 DEV 사이에서 바뀌었고 K8 STATIC은 DEV의 유리한 interaction보다 energy가 컸다. 이는 실제 오차의 분해이며 새 모델 학습·선정·독립 인과 실험이 아니다.

6. **다음 결정**: 모호한 D 불안정이라는 설명 대신 관측된 두 양상을 구분하되 고유 원인은 미확정으로 남겼다. 새 latency를 측정할 개입이 아니므로 기존 hD whole 비용을 원 상태에만 연결하고 시간순 이전의 최소 실험을 다음으로 삼았다.

근거: [오차항 진단 계획](PLAN_HORIZON_CORRECTION_ERROR_TERMS_KO_v1.md) · [실제 분해 결과 무손실 사본](../publication_main_20261015_v2/evidence/projection_followup_v1/runs/horizon_correction_error_terms_v1/RESULT.json.gz) · [게시 보고서](../publication_main_20261015_v2/reports/horizon_error_terms_v1/REPORT_KO.md).

## C24 — 고정 upstream 조건의 purged 시간 이전

1. **질문**: 앞 구간으로 맞추고 중간 구간으로 고른 보정의 이득이 뒤 구간에도 남는가? 기존 full-FIT upstream을 고정한 상태에서 readout 이전만 분리할 수 있는가?

2. **현 증거**: [확인] hD의 TUNE 개선이 DEV 악화로 바뀌었고 보정 방향·크기의 차이가 관측됐다. [추정] 시간 이전 실패와 upstream의 전 FIT 적합·이미 가용했던 C label이라는 조건을 분리하지 않으면 독립 일반화로 과장될 수 있었다.

3. **경쟁 설명**: 같은 기간 내 관계 변화 때문에 계수가 이전되지 않을 수 있고, 적합·선택 구간의 중첩 response가 낙관적인 선택을 만들 수도 있다. full-FIT scaler/E/D를 유지한 진단은 readout의 조건부 이전을 보지만 전체 pipeline의 시간순 학습 검증을 대신하지 못한다.

4. **선택한 검사**: Robin FIT256을 시간순 A128/B64/C64 후보로 나누고 A/B response가 다음 구간 origin 전에 끝나도록 purge했다. PCA K8·Raw의 STATIC 두 경로에서 A-only RMS/계수를 적합하고 B로 잠근 뒤 C를 평가하며 원 TUNE/DEV와 새 F는 사용하지 않았다.

5. **실제 결과**: [확인] A/B에서 각각11창을 제외한117/53/64 origin, 2fit·22후보·44 A/B·선정2 C를 완료했다. PCA K8과 Raw 모두 B의 보정 MSE 이득이 C 악화로 바뀌었고 선정 해·출력과 보고를 검산했다. 고정 upstream과 C label이 기존 full-FIT archive에 가용했던 사실을 숨기지 않았다.

6. **다음 결정**: 조건부 시간 이전의 반례로 남기고 독립 holdout·전체 temporal validation·모든 신경망 최적화 문제의 배제로 쓰지 않는다. 새 계수의 whole 비용은 계획상 미측정이며 이전 hD 비용을 이 새 정확도에 붙이지 않고 현재 설명 자료를 마무리하기로 했다.

근거: [시간순 진단 계획](PLAN_PURGED_READOUT_STABILITY_KO_v1.md) · [실제2경로 결과](runs/purged_readout_stability_v1/RESULT.json) · [게시 보고서](../publication_main_20261015_v2/reports/purged_readout_stability_v1/REPORT_KO.md).

## C25 — 자산·학습 조건 조사와 사용자 마무리 범위

1. **질문**: 현재 자료와 학습 조건으로 어디까지 설명할 수 있고, 무엇을 새 실험 없이 완료 산출물로 남겨야 하는가? 추가 TRAIN·시간순 재학습 제안과 최신 마무리·게시 요청을 구분할 수 있는가?

2. **현 증거**: [확인] 실제 예측 목적 E/D·E-only·Q·native·직접 회귀·보정·전체 비용과 설명 자료가 존재하며 작은 K의 Raw 정확도 격차는 남았다. [추정] 학습 노출·기간·밀도·모델 제약이 다음 설명이지만 현재 결과만으로 하나를 확정할 수 없었다.

3. **경쟁 설명**: Robin의 중첩된 supervision이 효과적 표본을 제한할 수 있지만 Beijing도 같은 상태라는 일반화는 맞지 않는다. 기존 upstream을 A-only로 맞추는 변화, supervision 밀도 확대, 새로운 기간·regime 추가는 각각 다른 개입이며 단순 update 확대와도 다르다.

4. **선택한 검사**: 최신 사용자의 현재 작업 마무리·main 게시 요청에 따라 기존 source·manifest·학습 로그·자산 존재와 가드를 읽기 전용으로 조사했다. TRAIN 자산의 새 역직렬화·학습·F·비용을 시작하지 않고 학습 이력·자산 조사·조건부 다음 설계와 교수 메모를 기존 결과와 구분해 작성했다.

5. **실제 결과**: [확인] 교수 메모에는 목적·배경·배포식·trainable/frozen·FIT/TUNE/DEV·공정 대조·원수치·실제 비용·실패·선행 한계를 모았다. 원수치 연결·그림·파일을 확인하고 코드·MD·CSV·PNG/SVG·실패 기록·큰 JSON의 무손실 공개 사본을 게시했으며 새 TRAIN 실험 결과는 만들지 않았다. 이 장부는 그 완료 산출물에 빠져 있던 후속 누적6항목 형식을 사후 보완한 것이다.

6. **다음 결정**: 현재 실행·검산·설명·게시 범위를 마무리하고 새로운 학습이나 자료 확장을 자동 시작하지 않는다. 조건부 TRAIN/시간순 재학습은 미실행 제안이며 독립 검증·학술적 신규성·작은 K 성공은 미해결로 남긴다. 과거 다른 캠페인의 p49K 준비 경로까지 완료했다고 주장하지 않고 보호 TEST·설치·큰 백본 학습·기존 작업 종료/재개 금지를 유지한다.

근거: [학습 이력 조사](LEARNING_REGIME_REVIEW_KO_v1.md) · [TRAIN 자산 조사](TRAIN_ASSET_REVIEW_KO_v1.md) · [미실행 다음 단계의 조건과 범위](NEXT_PHASE_LEARNING_DESIGN_REVIEW_KO_v1.md) · [최종 교수 미팅용 게시 메모](../publication_main_20261015_v2/reports/professor_method_brief_v1/REPORT_KO.md).

## 장부의 완료 범위

기록을 새로 꾸며 성능 성공을 만들지 않았다. 초기 D1~D5 이후 위25개 후속 묶음에 대해 질문·증거·경쟁 설명·선택한 검사·실제 결과·다음 결정을 추적할 수 있도록 모았다. 원 보고서의 당시 `로컬`, `미실행`, `검산 중` 문장은 역사기록으로 유지하고 최신 게시 여부와 혼동하지 않는다.

성능상 미해결, 계획상 미측정, 선택적 미실행 후속은 위 각 항목의 다음 결정에 남겼다. 이 장부의 구조·링크 검사는 과학 산술·GPU·optimizer·타이머의 독립 재실행을 대신하지 않으며, 새 보호 TEST 사용이나 과거 전체 연구의 완료를 뜻하지 않는다.
