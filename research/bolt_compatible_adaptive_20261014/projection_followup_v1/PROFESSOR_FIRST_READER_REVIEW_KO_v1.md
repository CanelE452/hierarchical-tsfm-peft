# 처음 읽는 교수 관점의 방법론 검토

검토일: 2026-10-15. 독립적인 첫 독자의 관점에서 문제 정의·선행 대비·대조의 의미·주장 범위를 검토했다. 기존 Markdown, JSON 메타데이터와 `reports/professor_method_brief_v1/figures_v2/method_spec.png`만 읽었다. 원자료·NPZ·모델을 열거나 F/GPU/학습/비용/과학 감사를 실행하지 않았다. 기존 파일을 수정하지 않았다. 아래 `[확인]`은 문서·메타데이터 확인 범위이며 이 검토자가 실험을 독립 재현했다는 뜻이 아니다. `[판단]`은 검토 결론, `[추정]`은 아직 검증되지 않은 해석이다.

**판정: 절차와 실패 조건을 설명할 정도의 방법론 후보는 있다. 작은 K 경로가 정확도와 비용을 함께 만족하는 성공 방법이라는 결론은 현재 근거로 내릴 수 없다.** 교수님께 가져갈 수 있는 것은 재현 가능한 후보와 반례를 갖춘 연구 판단 자료다. “완성”을 성능 성공·신규성·독립 일반화까지 포함하는 말로 사용하면 안 된다.

## 1. 가장 큰 앞단 논리 구멍 세 가지

### 구멍 1 — 왜 작은 K가 필요한지와 무엇을 압축했는지가 아직 한 문장으로 닫히지 않는다

교수님은 첫 단계에서 “원 C를 그대로 예측하면 더 정확한데 왜 K로 줄이나? 무엇의 비용을 줄였나?”를 묻는다. `[확인]` 현재는 F에 들어가는 stream 수를 C에서 K로 줄인다. 배포에는 current-C anchor와 K의 과거 buffer가 남는다. 원 C 과거를 입력받는 E도 필요하다. 따라서 전체 정보·저장량·통신량이 K차원만 남는 병목이라고 설명할 수 없다. `strict_K_only=false`와 도식의 우회 화살표는 이 점을 올바르게 드러낸다.

`[확인]` Robin PCA K8의 원 F DEV MSE는 0.386399로 Raw 0.340402보다 높다. 새 STATIC/RAMP 보정은 각각 0.525296/0.441554로 더 나빠졌다. Beijing PCA K4의 STATIC은 0.641731→0.633718로 조금 낮아졌지만, 같은 기회로 보정한 Raw STATIC/RAMP는 0.579185/0.574256이다. “보정 전 작은 K보다 개선”이 “Raw 대신 채택할 이유”로 이어지지 않는다.

`[판단]` 문제 정의의 앞문장은 **frozen univariate TSFM의 실행 stream 수를 줄이는 계산 경로를 검토한다**여야 한다. 그 다음에 “허용되는 정확도·비용 조건에서 Raw를 대체할 수 있는가?”를 질문으로 둬야 한다. 허용 손실률이나 배포 제약이 사전에 정해지지 않았다면 현재 관측한 손실을 정당화하는 기준을 사후에 만들면 안 된다. B1/B8 전체 경로의 실측 근거 없이 K/C를 속도 향상률로 바꿀 수도 없다.

최소 보완: 압축 대상을 F stream으로 명시하고, 남는 입력·state를 밝힌 뒤, Raw 대비 주지표와 실제 전체 경로 비용을 같은 상태에 연결한다. 이 연결을 끝내기 전에는 “효율적인 압축 예측법을 얻었다”로 시작하지 않는다.

### 구멍 2 — 표현에 남는 정보, F가 쓰는 정보, 최종 복원의 이득이 서로 다른데 성공 근거로 이어 붙이기 쉽다

“PCA가 정보를 보존한다” 또는 “미래 정답을 넣으면 복원이 가능하다”는 사실로 frozen F의 예측 성공을 설명할 수 없다. 과거에서 예측 가능한 정보인지, 각 latent stream의 과거만 받는 F가 활용할 수 있는지, D를 통과한 C 오차가 줄어드는지는 별도 질문이다.

`[확인]` 새 방향 비교에서 frozen F가 같은 E/D의 native보다 나은 조건이 많았고, horizon 보정에서는 같은 공간·같은 보정 class의 F가 FLAT보다 DEV MSE가 낮은 12/12 대조를 가진다. 반면 새 방향 비교의 작은 K F는 두 dataset 모두 Raw보다 낮은 DEV가 0/4였다. horizon 보정의 PCA-F 8조건도 원 Raw보다 낮은 DEV가 없다. native는 유한한 600 update, 두 seed, stream별 affine head의 대조다. 직접 C ridge는 D를 우회하며 정보와 용량이 같은 대조가 아니다.

`[판단]` 여기서 말할 수 있는 것은 **이 대조군 집합에서는 F 미래를 넣는 편이 단순 반복보다 도움이 된다**까지다. “F가 반드시 필요하다”, “표현이 충분하므로 D만 고치면 된다”, “native가 실패했으므로 작은 predictor는 불가능하다”는 결론은 나오지 않는다. 특히 E(Y)는 미래를 알고 만드는 진단이며 배포 예측기나 과거 예측 가능성의 증거가 아니다. 기존 D의 retained-space minimum을 새 horizon D의 하한으로 재사용할 수 없다.

최소 보완: 각 결과가 답하는 질문을 붙인다. F 대 FLAT은 등록된 단순 미래 대비 증분, 같은 E/D native는 지정된 학습기 대비 증분, D 우회 ridge는 직접 예측 대조, Raw는 배포 대체의 기준이다. 이 네 역할을 한 “baseline 승리 횟수”로 합치지 않는다.

### 구멍 3 — 계산 절차의 완결성과 효과·원인·신규성의 입증 수준이 다르다

`[확인]` horizon 진단은 COMPLETE이며 과학 감사 v1은 PASS다. 24 fit, alpha 후보 240개와 ZERO 24개, FIT/TUNE 평가 528개, 잠금 후 DEV 평가 24개가 기록돼 있다. 하지만 DEV는 이미 반복 노출된 적응적 탐색 자료다. 이 묶음에서 24개 잠금을 DEV보다 먼저 저장했어도 연구 전체의 독립 holdout이 되지 않는다. 감사도 독립 OS 수준 DEV 접근 사건을 검증한 것은 아니라고 명시한다.

`[확인]` 새 비용 실행과 반복 실행은 각각 COMPLETE로 기록됐지만, 이 검토에 고정한 비용 감사 v1은 `FAILED_PRESERVED`다. 별도 auditor 수정·후속 PASS를 이 문서에서 확인한 것으로 취급하지 않는다. 과학 감사 PASS를 비용 PASS나 최종 통합 보고 완료로 확대할 수 없다. 기존 ridge 규제 확장의 변경된 Beijing 계수 비용도 미측정이다.

`[확인]` 기존 검토는 AdaPTS의 frozen F 앞뒤 feature adapter 및 역변환, ORTCL의 orthogonal 입출력 변환, 기존 horizon별 decoder/잔차 보정을 선행으로 명시한다. 이 검토자는 해당 원문을 새로 외부 검증하지 않았다. 그 기록만으로도 “회전/압축+frozen F+decoder” 자체를 새로운 원리로 소개하기에는 반대 근거가 충분하다.

`[판단]` 산술 검산의 성공을 학술적 효과의 성공으로 바꾸는 문장에 가장 큰 위험이 있다. 현재는 절차의 명료화와 제한된 실패 진단이 성과이며, 독립 일반화·보편 원인·신규 방법의 성능 우위는 남은 주장이다.

## 2. 독자 질문 체인과 현재 답변 위치

1. **무슨 문제인가?** `PLAN_METHODOLOGY_COMPLETION_KO_v1.md`의 목적과 도식이 답한다. 다만 첫 설명은 “전체 C 정보를 K로 압축한다”가 아니라 F 실행 stream 절감이어야 한다.
2. **왜 중요한가?** 전체 E/F/D 배포 비용을 줄이려는 목적은 있다. 어느 배포 조건에서 정확도 손실을 감수할 가치가 있는지는 아직 정해진 성공으로 답할 수 없다. 가장 먼저 남는 구멍이다.
3. **왜 이 방법과 baseline인가?** `METHODOLOGY_DECISION_REVIEW_KO_v1.md`가 정보·좌표·horizon 제약을 분리한다. Raw, F/FLAT, STATIC/RAMP, ZERO, native, 직접 ridge의 역할이 구분될 때 설득력이 생긴다. 장황한 실험 목록부터 말하면 이 이유가 가려진다.
4. **설정의 근거는 무엇인가?** `PLAN_HORIZON_DECODER_DIAGNOSTIC_KO_v1.md`가 같은 lag·추가 계수·mask·RMS·alpha 메뉴와 선택 순서를 제시한다. K doubling과 다음 후보 선택은 이전 결과를 본 적응적 설계다. K4/K8이 최적 K이거나 모든 설정을 대표한다고 말할 수 없다.
5. **결과가 주장을 지지하는가?** Raw 대체 성공은 지지하지 않는다. F 대 FLAT의 제한된 이점, TUNE 보정 이득이 DEV로 옮겨지지 않는 현상은 지지한다. 주장은 이 수준에 맞춰야 한다.
6. **한계와 다음 판단은?** 노출 DEV, 두 dataset, 겹치는 horizon/origin, 유한 head/규제 메뉴, 새 상태 비용 검산과 신규성 미입증을 한 번에 명시할 수 있다. 다음 단계는 가설을 더 붙이는 자동 실험 나열보다 채택 기준·새 평가 범위·독립 비용 근거를 먼저 닫아야 한다.

`[끊김]` 입력/출력 그림에서 곧바로 264후보·SVD·감사 수치로 넘어가면 “이 방법이 왜 필요한가?”가 사라진다. 교수님께는 목적→배포식→판별 대조→가장 불리한 Raw 비교→조건부 해석 순서로 읽혀야 한다.

`[불일치/생략]` 도식은 candidate라는 제목, current-C 우회, FIT label만 사용, exposed DEV를 명시해 과장을 막고 있다. 다만 X가 이미 표준화된 값인지, encoder 식의 μ가 FIT scaler 평균인지 PCA center인지 구분하는 정의가 필요하다. 또한 `φ_h`는 단순 예측값이 아니라 각 lag의 알려진 과거/예측 미래와 **lag별 origin 참조**의 차이다. 첫 설명에서 이 정의가 빠지면 current-C persistence와 FLAT을 같은 baseline으로 오해하기 쉽다. PCA-FLAT은 일반적으로 current-C persistence가 아니다. Raw FLAT은 지정한 identity 경로에서 current-C persistence다.

## 3. 방법론 설명 가능과 효과 입증 성공을 구분하는 문장

**설명 가능한 것:** “원 C 과거를 고정 E로 K개 stream으로 변환하고, 동결된 Chronos-Bolt F가 만든 미래와 알려진 과거로 lag 특징을 구성합니다. current-C에 고정 D의 변화량 복원을 더하고, FIT 미래 관측만으로 작은 잔차 ΔD를 적합합니다. 추가 계수 수가 같은 STATIC과 사전 고정 RAMP를 비교하며 TUNE로 각 arm의 규제를 고른 뒤 노출된 DEV에서 진단합니다.”

```text
J = {0,1,2,4,8,16,32},  h = 1,...,48
z_past = (normalized_X - PCA_center) E
u = frozen_F(z_past)
φ_h = concat_l [ztilde_(h-l) - z_origin_reference(l)]
b_h = current_C + φ_h D
ŷ_h = b_h + a_h φ_h ΔD
STATIC: a_h=1      RAMP: a_h=(h-1)/47
```

`[확인]` 배포는 원 baseline FP32에 별도 FP32 correction을 더하는 순서다. 실수 대수의 `(D+a_hΔD)` 합성식과 byte 동일하다고 주장하지 않는다. E/F/old D는 고정하며 학습 대상은 추가 7KC slope다. Raw와 PCA의 계수 수는 다르고, STATIC/RAMP도 같은 열 수만 가질 뿐 유효 rank·regularization prior까지 완전히 같지 않다. full 48개 D_h를 학습한 결과가 아니다.

**아직 할 수 없는 성공 문장:** “K stream으로 원 C 정확도를 유지하면서 더 빠른 새 TSFM 압축법을 완성했다.” 정확도 반대 결과, 새 비용 검산 상태, 적응적 DEV, 선행 대비 미입증을 한꺼번에 건너뛰기 때문이다.

**현재 결과를 맞게 요약한 문장:** “작은 K 경로의 제한된 후보와 대조를 구현·검산했습니다. frozen F는 단순 미래 반복보다 유리했지만, 현재 예측 보정은 작은 K를 Raw보다 유리하게 만들지 못했고 Robin에서는 TUNE 이득이 DEV 악화로 바뀌었습니다. 성공 방법의 확정이 아니라 어디서 실패하는지 설명할 수 있는 단계입니다.”

## 4. 교수 예상 질문 여섯 개와 증거로 답할 수 있는 범위

### Q1. 원 채널이 더 정확한데 왜 압축하나? 압축률이 바로 속도 향상인가?

답할 범위: C개 F stream을 K개로 줄이는 계산 경로가 연구 대상이다. current-C·과거 buffer·E/D가 유지되므로 전체 입력·state의 K 병목은 아니다. 기존 실측에서 batch 조건별 이득이 다르며 B1의 근접한 차이를 강한 이득으로 말할 수 없다. 이 검토 시점의 새 horizon 비용은 실행 COMPLETE와 독립 감사 PASS를 구분해야 한다. **정확도와 비용의 동시 채택 성공은 답할 수 없다.**

### Q2. 왜 frozen F를 써야 하나? 작은 선형 예측기면 되지 않나?

답할 범위: 같은 E/D native16개, 원 직접 ridge12개 및 history10경로 규제 확장이 수행돼 단순 대조를 회피하지 않았다. horizon 보정의 같은 class F/FLAT 12개에서 F가 모두 낮은 DEV MSE다. 그러나 native의 유한 head/최적화, ridge의 별도 출력 구조, 두 dataset의 노출 DEV라는 제한이 있다. **F가 모든 작은 predictor보다 낫거나 반드시 필요하다는 결론은 아니다.**

### Q3. 재구성 목적이 문제라면 예측 목적 D로 바꾸면 해결되는 것 아닌가?

답할 범위: 실제 Y·teacher·공동 E/D·E-only·forecast-D가 이미 시도됐다. 이전 공통 FORECAST-D는 TUNE 12/12 개선, DEV 5/12 개선이며 Raw보다 낮은 DEV는 0/24였다. 현재 horizon 잔차도 Robin PCA K8 TUNE 0.085062→STATIC 0.077317/RAMP 0.078367이나 DEV는 0.386399→0.525296/0.441554다. **“예측 목적을 넣으면 해결”은 반대 근거가 있다.** 반대로 모든 예측 목적·비선형 decoder가 불가능하다고 말할 수도 없다.

### Q4. horizon ramp가 정적 보정보다 낫다고 말할 수 있나?

답할 범위: 같은 추가 slope 수의 좁은 함수 대조다. Robin K8에서는 RAMP가 STATIC보다 DEV가 낮지만 둘 다 무보정보다 나쁘다. Beijing K4에서는 STATIC 0.633718, RAMP 0.670856으로 순서가 반대다. Raw 보정도 함께 비교했다. **일관된 ramp 우위나 일반적인 시간 의존 decoder의 우위를 주장할 수 없다.** Raw와 PCA는 capacity match가 아니며 STATIC/RAMP의 유효 regularization도 동일하다고 할 수 없다.

### Q5. TUNE에서 좋아졌는데 DEV에서 나빠지는 원인은 확인했나?

답할 범위: 저장된 FP32 예측과 baseline의 실제 차이 c에 대해 같은 split의 true-observed 채널 macro로 `ΔMSE = 2 mean((baseline−Y)c) + mean(c²)`를 재계산했다. Robin K4 STATIC의 interaction은 TUNE −0.029341→DEV +0.037270, RAMP는 −0.026761→+0.038002다. K8 STATIC DEV는 interaction −0.007181에 correction energy +0.146078, RAMP는 −0.009391에 +0.064545로 증가 항이 더 크다. **관측된 오차 증가를 항별로 설명할 수 있다.** 이 진단은 별도 외부 감사가 아니며 분포 변화·과적합·오류 증폭 중 하나를 고유한 원인으로 확정하지 않는다. `[추정]` 분포 차이나 선택의 불안정성은 후속 가설이다.

### Q6. 무엇이 새로운 기여이며 일반화는 어디까지 확인했나?

답할 범위: feature-space adapter·orthogonal 변환·시간별 잔차 decoder라는 큰 구성 자체의 신규성은 선행 기록 때문에 주장하지 않는다. 현재 차별점 후보는 정해진 causal lag/current-C 구조 안에서 정보·F 미래·보정 제약을 분리한 검증 절차와 반례다. **새 알고리즘의 학술적 신규성도, 독립 일반화도 입증되지 않았다.** 264후보·24조건·48horizon·두 seed를 독립 dataset 수로 세지 않는다. 이번 묶음의 선택 잠금은 필요 절차지만 이전 DEV 노출을 없애지 않는다. 보호 TEST 새 사용 없이 후속 평가 설계를 별도로 정해야 한다.

## 5. 지금 강하게 주장하면 안 되는 것

- K/C의 비율만큼 전체 배포가 빨라진다거나 전체 저장·통신 정보가 K로 줄었다는 주장.
- 일부 작은 K 조건의 자체 baseline 개선을 원 Raw 대비 우위로 바꾸는 주장. 같은 보정 기회를 준 Raw 수치도 함께 보여야 한다.
- F/FLAT 12개 또는 유한 native/ridge 대조를 F의 보편적 필요성·작은 predictor의 불가능성으로 바꾸는 주장.
- TUNE 향상, oracle E(Y), retained numeric minimum을 각각 독립 일반화·과거 예측 가능성·모든 실수/FP32의 보편 하한으로 바꾸는 주장.
- 현재 STATIC/RAMP로 모든 horizon decoder를 대표하거나, 같은 계수 수를 완전 capacity/regularization match라고 부르는 주장.
- K4 interaction 부호 변화와 K8 correction energy 증가를 과적합 또는 분포 변화의 인과적 증명이라고 부르는 주장.
- 적응적 DEV의 최저 조건을 새 winner로 택한 뒤 독립 검증 성능처럼 보고하는 주장.
- 과학 PASS로 새 비용 감사 PASS·최종 통합 보고 완료를 대신하거나, 이전 계수/다른 cohort의 시간을 새 상태 비용으로 대입하는 주장.
- frozen E/F/D·orthogonal adapter·horizon 보정 자체가 새 방법이라는 주장.

## 6. 현재 가장 좁고 방어 가능한 문제 정의와 기여 후보

**문제 정의:** “고정된 두 다변량 시계열의 기존 노출 split에서, 원 C 과거와 current-C anchor를 허용한 채 frozen Chronos-Bolt의 실행 stream을 K로 줄였을 때, 같은 E/D 공간의 F 미래와 단순 미래 반복, 그리고 같은 추가 slope 수의 정적/고정 ramp 잔차 보정이 C 예측 오차와 실제 전체 경로 비용에 주는 조건부 차이를 구분한다.”

이는 범용 TSFM 압축 성공을 전제하지 않는 평가 질문이다. 큰 K가 좋아졌다는 결과를 동일 압축률의 개선으로 합치지 않고, 미래 정답 진단과 실제 예측을 구분한다. 정확도와 비용을 함께 제시하려면 새 상태 비용의 독립 검산이 추가로 닫혀야 한다.

**기여 후보 1 — 특정 구조의 실패를 분해한 재현 가능한 진단 절차.** current-C anchor·known-past lag와 실제 F 미래를 보존한 대조에서, decoder 유연성의 증가와 F 미래의 증분을 구분했다. 기존 방식보다 새 원리라는 주장이 아니라, 이 실험 구조에서 어떤 설명이 반례에 막히는지를 명료하게 만든 기여 후보로 제시한다.

**기여 후보 2 — 제한된 음성 결과와 후보 기각 근거.** 현재 두 dataset에서는 예측 목적 보정·PCA 차원 확대·등록된 native/ridge 대조가 작은 K의 Raw gap을 해결하지 못했다. 특히 TUNE에서 선택된 보정이 Robin DEV를 악화시키며, 그 증가가 K4와 K8에서 다른 오차항 조합으로 나타났다. 이를 모든 저차원/TSFM 방법의 불가능성으로 일반화하지 않는다.

`[판단]` 이 둘은 교수 상담에서 연구의 현 상태를 방어하는 기여 후보다. 그 자체로 충분한 논문 기여이거나 새 성공 방법이라는 보장은 없다. 현재 자료를 더 긍정적으로 표현하는 것보다 **“후보는 설명 가능하고, Raw 대체 성공은 아직 아니다”를 처음부터 분리하는 것**이 가장 방어 가능하다.

## 7. 확인한 증거와 고정 범위

- [최상위 계획](PLAN_METHODOLOGY_COMPLETION_KO_v1.md), [horizon 설계](PLAN_HORIZON_DECODER_DIAGNOSTIC_KO_v1.md), [방법론 결정 검토](METHODOLOGY_DECISION_REVIEW_KO_v1.md).
- [새 방향/native/ridge 비교](reports/new_direction_comparison_v1/REPORT_KO.md), [ridge 규제 확장](reports/ridge_regularization_extension_v1/REPORT_KO.md).
- [실제 horizon 진단 메타데이터](runs/horizon_decoder_diagnostic_v1/RESULT.json): SHA256 `5f158382640dac74ec63cc3d1005b385c898821d3c032d51b543f2c9b27d7dac`, COMPLETE. 보호 TEST 접근 false, 독립 holdout false, strict-K-only false. 이 단계 자체의 whole cost 필드는 false이며 이후 별도 비용 실행과 구분한다.
- [horizon 과학 감사 v1](audits/HORIZON_DECODER_DIAGNOSTIC_AUDIT_v1.json): SHA256 `0f65053536403a3b7b8f53fca3b9f8274fb1a406eaf6dfdcb73bb0467fffe698`, PASS. 독립 재풀이된 선정 fit 22개와 ZERO 2개, 모든 후보 점수·선정 잠금 등 지정 범위를 확인했다. 미선정 계수 전체를 독립 재적합한 감사는 아니다. 새 joint output-space minimum과 C-output E(Y)는 계산하지 않았다.
- [오차항 진단](runs/horizon_correction_error_terms_v1/RESULT.json): SHA256 `605a2dec55f9d7a48bea3df99d177b5c54706d06cf14268de4f8c6fcc32204eb`, COMPLETE. scope는 저장 endpoint 진단이며 외부 감사·새 선택·고유 인과 설명이 아님을 명시한다. 24조건/72전체 split cell, 최대 closure 절대값 약 `9.70e-15`가 기록돼 있다.
- [새 horizon 비용 실행](runs/horizon_decoder_cost_v1/RESULT.json)과 [반복 실행](runs/horizon_decoder_cost_repeat_v1/RESULT.json)은 읽은 시점 각각 COMPLETE·72셀/360표본이고, [비용 감사 v1](audits/HORIZON_DECODER_COST_AUDIT_v1.json)은 FAILED_PRESERVED다. 후속 수정·감사·최종 보고의 완료는 이 검토의 확정 근거에 넣지 않았다. 이 상태 문장은 v1 snapshot에 관한 것이며 후속 결과를 부정하는 뜻이 아니다.

읽은 계획의 “설계만/실행 0” 상태는 해당 문서 작성 시점의 기록이다. 최신 horizon 실행 상태는 위 COMPLETE/PASS JSON으로 판단했다. 이 검토 문서는 새 실험·실제 배열 검산·최종 통합 보고를 대신하지 않는다.
