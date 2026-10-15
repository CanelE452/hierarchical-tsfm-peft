> 게시 사본입니다. 원문 작성 당시의 로컬·준비·미실행 상태는 역사로 보존합니다. 최신 완료/미해결 상태와 게시 범위는 [색인](../README_KO.md)을 확인하세요. 과학 원본은 바꾸지 않고 링크와 이 안내만 추가했습니다.

# 다음 연구 판단: 더 복잡한 방법보다 보정의 시간 이전 가능성을 한 번만 검증

작성일: 2026-10-15. 상태: **의사결정 메모 / 제안한 새 분할·fit·배열 접근·F·모델·GPU·비용 실행 0**. 기존 계획·보고서·JSON 메타데이터와 SOURCE의 관련 계약만 읽었다. 원자료·NPZ는 열지 않았으며 기존 결과·선정·보고서를 수정하지 않았다. 기존 검토의 후속이고 ROOT의 최종 교수 메모를 대신하지 않는다.

**한 개만 더 한다면, Robin PCA K8의 STATIC 잔차 보정 하나를 고정해 FIT 내부에서 시간순 A/B/C 분할을 만들고, 앞 구간에 적합·중간 구간에서 alpha 선택·마지막 구간에서 무보정 대비 차이를 보는 조건부 안정성 진단을 권한다.** 새 encoder·horizon 함수·gate를 추가하지 않는다. 성공 방법을 만들기 위한 마지막 손질이 아니라, 현재 보정 계열을 더 연구할 최소 근거가 있는지 기각하는 실험이다.

이 권고에는 중요한 제한이 있다. 원 full-FIT scaler/E/D와 F cache를 유지하므로 **고정 표현을 조건으로 한 추가 readout의 안정성 진단**이다. 완전한 end-to-end 시간순 검증도 독립 일반화 평가도 아니다. 현재 계약의 immutable state를 유지하면서 가능한 가장 좁은 질문으로 제한한다. 이 질문조차 연구상 가치가 작다고 판단하면 새 architecture 없이 현재 Raw 대체 실패를 결론으로 닫는 것이 합리적이다.

## 왜 이 질문인가

`[확인]` 새 예측 목적 E·cross-moment 방향·Q 학습·native·직접 ridge와 규제 확장·horizon 보정은 이미 조사했다. “예측 목적을 처음 넣자”, “native를 하나 더 붙이자”, “D를 더 유연하게 하자”는 다음 핵심 질문이 아니다. 현재 hD는 사전에 정한 STATIC/RAMP와 Raw/F/FLAT 대조까지 갖췄지만 작은 K의 Raw gap을 닫지 못했다.

`[확인]` Robin K8 STATIC은 TUNE 0.085062→0.077317, DEV 0.386399→0.525296이다. 저장 endpoint의 오차항 진단에서 DEV interaction은 −0.007181로 유리한 방향이지만 correction energy +0.146078이 이를 넘는다. RAMP도 같은 방향의 TUNE/DEV 역전을 보인다. K4에서는 interaction 자체가 TUNE 음수에서 DEV 양수로 바뀐다. 이것은 관측된 오차 분해이며 과적합·분포 변화 중 한 원인을 확정하지 않는다.

`[판단]` 새로운 함수가 표현력이 부족해서 필요한지보다 **이 자료량과 고정 특징에서 학습한 보정의 이득이 시간 구간을 넘어 유지되는가**를 먼저 묻는 편이 직접적이다. 한 TUNE에서 계수가 좋아졌다는 이유로 함수 수를 늘리면, 현재 드러난 이전 실패를 해결하기보다 적응적 선택의 자유도를 늘릴 수 있다. `[추정]` 시간 구간에 따라 residual 관계가 달라질 가능성이 있지만, 현재 메타데이터만으로 그 원인이나 안정성을 알 수 없다.

비용은 이 질문과 분리된다. `[확인]` 새 비용 감사 v2 두 개는 실제 JSON에서 PASS이며 SHA는 각각 아래와 일치했다.

```text
원 비용 v2    84e32f1f937733990140e7e70c1dcfd389aec807d116e600e3e91a4a52aa2f37
반복 비용 v2  27d721c6f248caf521bf11dd045af2eb61151629bdb64fbda76750d4b056566c
```

따라서 현재 다음 한 개의 연구 질문을 감사 버그 수정이나 또 다른 timer 반복으로 잡을 이유는 약하다. 이 PASS가 정확도 성공을 의미하지도 않는다. 앞선 첫 독자 검토의 비용 v1 실패 언급은 당시 고정한 v1 snapshot이고 여기서는 새 상태를 별도로 확인했다.

## 정확히 무엇을 반증하려는가

운영 가설을 다음 한 문장으로 한정한다.

> 현재 고정 PCA K8/L7/F 특징에서 앞 시간 구간으로 적합하고 중간 구간으로 선택한 STATIC ΔD가, 그보다 뒤의 사전 고정 구간에서 ZERO보다 낮은 관측 채널 macro 정규화 MSE를 낸다.

마지막 구간의 signed 차이 `MSE(selected)-MSE(ZERO)`가 0 이상이면 이 사전 고정된 국소 가설은 지지되지 않는다. ZERO가 선택되면 “보정이 필요 없었다”로 그대로 기록한다. C 결과가 나쁜데 B/C 경계·K·arm·alpha 메뉴·RMS를 바꿔 같은 가설의 성공으로 합치지 않는다.

이것은 “미래 전체에서 보정이 불가능하다” 또는 “이번 DEV 실패의 원인이 과적합이다”라는 보편 가설의 검정이 아니다. 한 분할의 실패만으로 모든 보정·자료 구간을 기각할 수 없다. 이 진단에서 성공해도 기존 DEV·Raw gap이 해결되는 것이 아니다. 이 구분을 할 수 없다면 이 추가 실험의 해석 비용이 정보 이득보다 크다.

## 한 번의 사전 고정 프로토콜

### 1. 범위와 대조

Robin C17, PCA K8, L512/H48, lag `0/1/2/4/8/16/32`, 기존 median F·FP32 배포식을 고정한다. PCA K8을 택하는 이유는 기존 작은 K 후보 중 원 Raw에 상대적으로 가까웠고, STATIC의 TUNE/DEV 이전 실패가 뚜렷했기 때문이다. 이전 결과를 본 적응적 진단 대상으로 선택한 것이며 독립 사전 선택으로 포장하지 않는다.

새롭게 바꾸는 것은 **ΔD의 적합·alpha 선정에 쓰는 시간 구간** 하나다. 새 K·Q·E·native·RAMP·spline·horizon bin·채널별 alpha·학습 gate는 없다.

동일한 A/B/C에서 Raw-C STATIC+ZERO도 함께 둔다. PCA는 추가 7KC, Raw는 7C² slope이므로 capacity match가 아니다. Raw는 이 시간 분할에서 압축 후보를 채택할 여지가 있는지 판단할 경쟁 기준이다. Raw를 제외하면 “추가 보정이 자기 baseline보다 조금 좋아졌다”는 결론을 다시 목적의 성공으로 오해하기 쉽다.

### 2. 날짜로 순서를 정하고 response 중첩을 제거

기존 FIT256 origin을 실제 timestamp 순서로 정렬한 뒤, index상 앞 128개를 A 후보, 다음 64개를 B 후보, 마지막 64개를 C로 고정한다. 실제 배열을 읽기 전이므로 origin 간격·purge 후 개수·관측수는 현재 미확인이다. 같은 timestamp 중복이나 순서 불일치가 있으면 임의 정리하지 말고 계약 불일치로 보존한다.

기존 계약에서 origin o는 미래 첫 timestamp index이며 target은 `o,...,o+47`이다. A에서는 `o+47 < first_origin(B)`인 origin만 유지하고, B에서는 `o+47 < first_origin(C)`인 origin만 유지한다. 경계가 부족해지는 경우 C 시작을 옮겨 유리한 분할을 만들지 않는다. 각 경계와 제거 origin·고유 target timestamp·반복도를 기록한다.

이 purge는 **이전 단계에서 쓰는 response가 다음 단계의 예측 origin보다 앞에서 모두 관측 가능하게 만드는 것**이다. B/C history가 이전 A/B 기간을 포함하는 것은 실배포에서 허용되는 과거 관측 공유다. 겹치는 512 history를 전부 지우는 L+H 완전 분리와 혼동하지 않는다. 같은 window의 horizon이나 서로 겹친 origin을 독립 표본으로 세지 않는다.

### 3. 고정 state와 새 readout을 분리

기존 full-FIT scaler·PCA E/center·old D·known-past/current-C와 원 실제 F cache는 immutable로 고정한다. 새 특징을 계산해도 같은 저장 경로·byte 계약을 재현해야 한다. **이 고정 상태가 A 밖 자료를 이미 사용해 적합됐다는 사실은 숨기지 않는다.** 이 선택 때문에 이 실험은 조건부 readout 진단이다.

PCA와 Raw 각각에서 actual-F의 ungated A 특징만으로 uncentered RMS를 다시 계산한다. B/C 통계를 RMS에 넣지 않는다. 원 hD의 full-FIT RMS를 그대로 재사용하면 새 readout 단계의 미래 통계를 섞으므로 이 조건부 질문조차 흐려진다. 기존 source의 관측 응답 처리·FP32 correction·alpha0 minimum-norm·양수 ridge 식은 유지한다. RMS가 바뀌므로 기존 선정 계수를 재사용하는 실험이 아니라 **동일한 보정 class의 새 A-only 적합**이다.

alpha 메뉴는 기존 `0,1e-4,.001,.01,.1,1,10,100,1000,10000`과 ZERO 그대로다. 추가 절편·mean·bias를 붙이지 않는다. A에서 fit, B에서 route별 observed channel-macro normalized MSE 하나로 alpha/ZERO를 선택한다. 정확 동률 순서는 기존 alpha 순서 뒤 ZERO다. PCA와 Raw 두 선택을 모두 잠근 뒤 C 예측·점수를 만든다.

purge 후 관측수가 없거나 source가 요구하는 finite/rank/shape 계약을 만족하지 못하면 해당 경로를 실패로 보존한다. 이를 피하려고 C를 학습에 넣거나 K를 낮추지 않는다. C를 읽고 B alpha를 다시 고르지 않는다. 이번 프로토콜에 원 TUNE/DEV 재선정이나 TEST 접근을 추가하지 않는다.

### 4. 평가와 기록

주 결과는 C에서 다음 세 signed 차이다.

```text
PCA 보정의 이전     MSE(PCA_selected) - MSE(PCA_ZERO)
Raw 보정의 이전     MSE(Raw_selected) - MSE(Raw_ZERO)
배포 경쟁의 차이    MSE(PCA_selected) - MSE(Raw_selected)
```

B의 선택 점수도 같은 표에 남겨 선택 시점과 평가 시점의 방향을 분리한다. A/B/C마다 자체 true-observed 분모로 채널 macro를 계산하며, per-C·사전 1–16/17–32/33–48 horizon 구간과 correction interaction/energy를 보조로 남긴다. C의 부분집합 중 유리한 채널이나 horizon만 성공으로 고르지 않는다. 이번 단계의 겹치는 표본에서 통상적인 독립 표본 CI·p값을 만들지 않는다.

새 readout 계수의 실제 전체 비용은 이 진단에서 미측정이다. 기존 v2 비용은 기존 선정 계수의 결과로 유지한다. 이번 목적은 상태 변경의 비용 최적화가 아니라 시간 이전의 방향을 기각하는 것이므로, 새 정확도 수치에 옛 timer 수치를 붙여 Pareto 성공을 선언하지 않는다.

## 기존 작업과 무엇이 같고 다른가

- 기존 hD와 **같은 STATIC class·ridge 메뉴·true-observed 목적·baselineFP32+correctionFP32**다. 새로운 예측법이나 새로운 CV 알고리즘으로 소개하지 않는다.
- 기존 hD는 full FIT fit→기존 TUNE 선택→기노출 DEV 진단이었다. 이 제안은 FIT 내부의 시간순 A fit→B 선택→C 잠금 평가로, 새로운 함수 없이 readout의 시간 이전을 묻는다.
- 기존 `PLAN_HORIZON_CORRECTION_ERROR_TERMS_KO_v1.md`는 이미 선정된 endpoint의 산술 분해다. 이 제안은 ΔD를 A에서 다시 적합하므로 동일한 진단의 단순 재포맷은 아니다. 다만 실패 원인을 유일하게 식별하지 못한다는 제한은 같다.
- 기존 past-constant gate는 C 과거의 정확한 상수 여부로 출력을 바꿨고, local/shrinkage D는 origin별 적합이나 원 D 주변의 규제를 다뤘다. 여기서는 입력 조건·출력 함수를 늘리지 않는다.
- `PREDICTABLE_PROJECTION_DATA_CONTRACT_KO_v1.md`에는 이미 시간순 fold·purge와 full-FIT state 재사용의 한계가 적혀 있다. 따라서 시간순 검증이라는 아이디어를 처음 제안한 것이 아니다. 지정 완료 보고와 파일명 검색에서는 이 A/B/C 진단의 완료 기록을 찾지 못했다. 전체 저장소에 존재하지 않는다는 보편 단정은 하지 않는다.

## 현재 계약으로 가능한 것과 불가능한 것

**가능:** 기존 허용 FIT origin/관측/mask/state/cache만 사용한 새 A-only residual ridge 적합과 B/C 조건부 채점이다. 새 데이터·TEST·설치·download·백본 학습이 필요하지 않는다. 기존 파일 보존·새 run 출력·독립 검산 계약을 유지할 수 있다. 실제 구현·배열 접근·fit·검산은 별도 실행 단계이며 이 메모에서는 수행하지 않았다.

**이 계약으로 불가능:** 원 full-FIT scaler/E/D·cache를 유지한 결과를 완전한 end-to-end 시계열 검증이나 독립 holdout 성능이라고 부르는 것이다. 엄밀한 전체 pipeline의 시간순 평가에는 A에서만 scaler·fill 초기 통계·E·D를 적합하고 그 새 상태로 Raw/PCA F를 다시 실제 추론해야 한다. 원 immutable state/cache를 유지하는 현재 작은 진단과 다른 데이터/전처리 계약이다. 그 확장까지 이번 한 개의 권고에 자동 포함하지 않는다.

더 근본적으로, A/B/C의 값과 원 FIT 결과를 연구자가 이미 본 이력이 사라지는 것도 아니다. label을 이번 readout fitting에 넣지 않았다는 성질과 연구 전체에서 미노출 평가라는 성질은 다르다. 이번 결과로 독립 확증을 대체할 수 없다.

## 결과별 연구 결정

1. **B가 ZERO를 선택하거나, B 이득이 C에서 사라진다.** 이 사전 고정 후보의 시간 이전 근거가 없다. 함수 수를 늘려 같은 exposed 자료에서 성공을 찾는 우선순위를 낮춘다. 현재 조건에서 작은 K의 예측 보정을 통해 Raw를 대체한다는 목표는 미달이라고 정리한다. 실패가 모든 미래 decoder의 불가능성이라는 뜻은 아니다.
2. **PCA가 C에서 자기 ZERO보다 좋아져도 Raw_selected보다 나쁘다.** 국소 보정의 조건부 이전만 확인한 것이다. 압축 경로를 채택할 정확도 근거는 없다. 새 성공 방법으로 승격하지 않는다.
3. **PCA가 C에서 ZERO와 Raw_selected보다 모두 낮다.** 한 과거 구간에서의 조건부 긍정 관찰로 남긴다. 기존 DEV 악화와 독립성 부족은 그대로다. 새 성능 주장·새 timer 요구·더 많은 함수 탐색을 자동으로 시작하는 신호가 아니다. 배포 가치와 별도 평가 설계가 없으면 현 연구의 성공 결론은 바뀌지 않는다.

`[판단]` 이 한 번의 진단이 추가 모델보다 낫다고 권하는 이유는 성공 가능성이 높아서가 아니다. 현재 남은 가장 직접적인 반례인 **선택된 보정의 시간 이전 실패**를 함수 자유도를 늘리지 않고 확인하며, 음성 결과여도 현재 계열을 정리할 수 있기 때문이다. 반대로 “원 Raw 정확도를 유지하는 새 방법”이 당장 산출되어야 한다는 기준이라면 이 제안도 그 요구를 충족하지 않는다. 현 자료·계약만으로 그 성공을 약속할 근거는 없다.

## 참조한 상태

- [방법론 완성 계획](PLAN_METHODOLOGY_COMPLETION_KO_v1.md), [결정 검토](METHODOLOGY_DECISION_REVIEW_KO_v1.md), [예측 방향 데이터 계약](PREDICTABLE_PROJECTION_DATA_CONTRACT_KO_v1.md).
- [horizon 진단 계획](PLAN_HORIZON_DECODER_DIAGNOSTIC_KO_v1.md), [오차항 진단 계획](PLAN_HORIZON_CORRECTION_ERROR_TERMS_KO_v1.md), [past-constant gate 계획](../../projection_followup_v1/PLAN_PAST_CONSTANT_GATE_KO_v1.md).
- [새 방향 비교](../reports/new_direction_comparison_v1/REPORT_KO.md), [ridge 규제 확장](../reports/ridge_regularization_extension_v1/REPORT_KO.md), [causal lag 목적 비교](../../publication_main_20261015_v1/reports/causal_lag_objective_v1/REPORT_KO.md).
- [실제 hD 메타데이터](../evidence/projection_followup_v1/runs/horizon_decoder_diagnostic_v1/RESULT.json.gz), [과학 감사](../evidence/projection_followup_v1/audits/HORIZON_DECODER_DIAGNOSTIC_AUDIT_v1.json.gz), [저장 endpoint 오차항 진단](../evidence/projection_followup_v1/runs/horizon_correction_error_terms_v1/RESULT.json.gz).
- [원 비용 감사 v2](../../projection_followup_v1/audits/HORIZON_DECODER_COST_AUDIT_v2.json), [반복 비용 감사 v2](../../projection_followup_v1/audits/HORIZON_DECODER_COST_REPEAT_AUDIT_v2.json). 이 두 JSON의 PASS와 SHA를 직접 읽어 확인했으며 비용을 재실행하거나 재감사하지 않았다.

새 제안은 미실행이다. 실제 timestamp 간격·purge 후 개수·부분 구간 관측수·새 계수·점수는 이 문서의 확인 사실이 아니다.
