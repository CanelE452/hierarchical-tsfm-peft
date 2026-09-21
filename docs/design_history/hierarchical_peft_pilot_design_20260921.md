# 계층형 시계열 PEFT — 제한된 직접 비교 설계안

작성일: 2026-09-21
상태: 계획 검토용. 학습·평가·저장소 변경을 승인하는 실행 지시문이 아니다.
기존 저장소 확인 기준: CanelE452/tsfm-peft-method-screen, e84ef88579382574d9c82985c2c0c4672e262b21.

## 1. 목표와 예상 결과

목표: 실제 부분–전체 합계 관계를 가진 월별 시계열에서, 같은 합계 조정을 받은 표준 PEFT보다 계층 관계를 이용한 작은 표현 보정이 미래 예측 오차를 더 줄이는지 판단한다.
소비처: 사용자와 지도교수의 '이 문제에 방법론 연구 비용을 더 투자할 것인가' 판단.

예상 결과:
- 모든 조정된 예측은 합계가 맞는다. 이 사실만으로는 제안 효과가 아니다.
- 계층 정보를 이용한 군이 일반 LoRA, 전체 계열을 단순 결합한 어댑터, 잘못 연결한 계층 어댑터보다 정확하면 추가 개발 근거가 생긴다.
- 일반 LoRA+MinT 또는 단순 결합으로 충분하면 현재 계층 어댑터의 필요성은 낮다.
- 한 자료의 파일럿은 계층형 PEFT 전체의 가능성이나 논문 채택을 판정하지 않는다.

이번에 제외: MAG·클리핑·간헐수요·센서 지연·과거 검색·변수 구성 전이·연속 학습의 동시 실험. 확률적 계층 정합과 비음수 제약도 이번 주목표에서 제외한다.

## 2. 기존 연구와의 관계

[확인] MinT는 모든 수준의 예측을 오차 공분산에 근거해 조정하는 기존 방법이다. 합계 조정 자체는 신규성이 아니다.
[확인] End-to-End Learning of Coherent Probabilistic Forecasts for Hierarchical Time Series(ICML 2021)는 조정을 학습 모델에 포함한다.
[확인] Coherent Probabilistic Forecasting of Temporal Hierarchies(AISTATS 2023)는 집계 수준의 표현과 그래프 신경망을 활용한다. 다루는 계층이 시간 집계라는 차이가 있지만, 그래프 정보 전달 자체도 새 원리가 아니다.
[미검증] 본 설계의 '동결 TSFM 인코더 표현에 대한 작은 계층 관계 보정'이 기존 방법을 넘어서는 효과·신규성이 있는지는 아직 확인하지 않았다.

이전 합계 감독과 차이:
- 이전에는 일부 상세 정답이 없는 상태에서 집계 정답으로 상세 예측을 개선하려 했다.
- 이번에는 같은 시각의 최하위 관측과 공식 집계 구조를 사용한다. 상위 학습 정답도 모두 제공한다.
- 집계값은 독립적으로 새로 관측한 정보가 아니라, 같은 최하위 자료를 다른 수준으로 표현한 것이다.

## 3. 선택한 접근과 보류한 접근

A. 합계 위반 벌점만 추가: 빠르지만 기존 종단간 정합 학습과 가깝고, 테스트에서 합계만 맞은 효과를 분리하기 어렵다. 이번 제안군으로 선택하지 않는다.
B. 계층 이웃의 과거 표현을 공유하는 작은 어댑터: 이번 직접 검증 대상으로 선택한다.
C. 부모 예측에서 형제 예측을 빼 타깃의 대안 예측을 만드는 방식: 좋은 단순 발상이지만 선형 예측 결합·reconciliation과 겹친다. 새 PEFT 주장으로 추가하지 않는다.

## 4. 데이터 — Labour 하나로 첫 비교

자료: datasetsforecast.hierarchical의 Labour 공개 벤치마크.
공식 loader의 계약:
- 월별 자료, 계절 주기 12.
- Country / Country-Region / Country-Gender-Region / Country-Employment-Gender-Region 계층.
- Y_df, summing matrix S_df, levels/tags를 함께 제공한다.
- 공식 loader는 ds < 2020-01-01을 사용한다. 이 canonical 전처리를 명시하고, 결과를 보고 추가 기간을 제거하지 않는다.

[미확인] 이번 응답에서는 실제 원본 archive를 확보하지 못했으므로, 정확한 노드 수·시작/종료일·결측·반올림 오차는 실제 다운로드 후 기록해야 한다. 모르는 숫자를 완료값으로 채우지 않는다.

사용 범위:
- 제공된 계층의 모든 최하위 계열과 그 상위 노드를 사용한다.
- 성능 때문에 일부 지역·계열만 고르지 않는다.
- 공식 계층을 사용하며 임의 시계열 몇 개를 모아 새로운 계층으로 만들지 않는다.
- 공급된 S의 행·열 ID, 최하위 identity block, 레벨별 분할, 부모–자식 관계, 동일 시각을 검사한다.
- 실제 최하위 관측으로 S*b를 구성한 값을 일관된 상위 정답으로 사용한다. 원본에 상위값도 있으면 차이를 먼저 보고한다. 단순 반올림 외의 단위/정의 차이가 있으면 조용히 덮어쓰지 않는다.
- 원천 노출·사전학습 포함 여부를 확인할 수 없으면 개발 벤치마크로 명시한다.

기간·단위·합계 계약이 잘못되면 데이터 오류로 중단한다. 모델 성능 실패로 해석하지 않는다. 다른 자료로 자동 교체하지 않는다.

## 5. 시간 분할

입력 C=96개월, 미래 H=12개월. 실험 조건의 선택이지 Labour의 공식 평가 길이를 그대로 재현한다는 주장이 아니다.
총 관측 수를 T라 할 때 시간순으로:
- TRAIN: 처음부터 마지막 132개월 직전까지.
- CAL: 그 다음 36개월. reconciliation 공분산 추정 전용.
- V_SELECT: 그 다음 36개월. 학습률·체크포인트 선택 전용.
- E: 마지막 60개월. 최종 파일럿 평가.

TRAIN에서 C+H를 만족하는 원점을 월 단위로 전부 만든다. 원점을 64개 등으로 자르지 않는다.
CAL에서는 각 월 원점의 1-step 오차만 공분산에 쓴다. target 한 시점이 CAL 안에 있어야 한다.
V/E에서는 미래 12개월 전체가 각 구간 안에 들어오는 월 원점을 모두 사용한다. 정상적인 연속 자료라면 V 25개, E 49개 원점이다.
각 원점의 과거 문맥은 이전 구간으로 넘어갈 수 있지만, 그 원점 이후의 값을 포함하지 않는다.
학습 후 모델·reconciliation을 E에서 갱신하지 않는다. 다음 E 원점에서는 그때까지 관측된 실제 과거를 입력할 수 있다.

TRAIN에 적법한 원점이 아예 없으면 길이 계약 불충족으로 중단한다. 희귀 조건의 최소 발생 횟수 같은 별도 gate는 만들지 않는다.

## 6. 모델과 네 학습군

백본: 현재 프로젝트에서 사용한 pinned Chronos-Bolt-small snapshot. 새 버전 자동 업그레이드 금지.
현재 공개 API는 입력 길이 96과 출력의 첫 12개 사용이 가능하지만, 과거 프로젝트의 512/64 고정 wrapper는 그대로 사용하지 않는다.
실제 wrapper는 native forward와 초기 동일성을 확인해야 한다.

모든 군은 같은 사전학습 F0에서 시작한다. 먼저 별도 B0를 학습한 뒤 두 번째 모듈을 붙이는 절차를 요구하지 않는다.

L — LORA:
- 현재 검증된 q/v rank8 LoRA 방식.
- 모든 노드가 하나의 LoRA를 공유한다.
- 계열별 독립 LoRA를 N개 만들지 않는다.

G — GLOBAL_ADAPTER:
- 인코더·디코더·기존 출력층을 모두 동결한다.
- 인코더 출력의 자기 노드 표현과 같은 시각/패치의 다른 모든 노드 평균 표현을 입력으로 받는 작은 잔차 어댑터.
- 학습하는 것은 새 어댑터뿐이다.

H — HIERARCHY_ADAPTER (유일한 제안 후보):
- G와 같은 위치·크기·초기값.
- 다른 노드 전체 평균 대신 공식 계층에서 직접 연결된 부모·자식 노드의 평균 표현을 사용한다.
- 부모/자식 선택은 공식 S 및 계층 메타데이터만 사용한다.

P — PERMUTED_ADAPTER:
- H와 같은 구조.
- 계층 레벨 안에서 고정 seed로 노드 연결을 섞는다. 데이터·노드 ID·정답·최종 summing matrix S는 바꾸지 않는다.
- 연결 그래프의 크기·레벨·연결 수 분포는 유지한다. 실제 연결과 다른지 사전 검사한다.
- 성능을 보고 permutation을 다시 뽑지 않는다.

G/H/P의 정보와 파라미터:
- 같은 예측 원점의 전체 노드 과거가 모두 제공된다.
- 다른 노드의 미래 정답은 입력으로 주지 않는다.
- h_{i,j}: i번째 노드의 j번째 시간 패치에 대한 동결 인코더 출력.
- m_{i,j}: 지정된 이웃의 원래 인코더 출력 평균. 업데이트한 이웃 표현을 재귀적으로 사용하지 않는다.
- delta = W_up GELU(W_self h + W_message m + b_down) + b_up.
- hidden width 512, bottleneck 8이면 12,808개 학습 파라미터.
- W_up,b_up=0으로 초기화한다. 초기 예측은 F0와 같아야 한다.
- 특수 REG token은 변경하지 않는다.
- 각 노드와 모든 패치에 동일한 어댑터 가중치를 공유한다.

L과 G/H/P는 파라미터 예산이 같지 않다. G/H/P끼리의 비교가 구성요소 효과의 직접 대조다. L은 강한 표준 PEFT 기준이다. 파라미터가 작다는 이유만으로 정확도 목표의 성공으로 세지 않는다.

## 7. 빠르게 계산하는 방법

G/H/P의 인코더는 동결돼 있으므로, 각 원점·노드의 encoder hidden state, attention mask, 복원에 필요한 normalization 통계를 한 번 계산해 캐시한다.
같은 cache를 G/H/P에 재사용한다. 캐시는 자신의 과거 입력으로만 만든다.
학습 시 decoder는 동결된 상태지만, 수정된 hidden state에 대한 gradient는 통과해야 한다. decoder 전체를 no_grad로 실행하면 안 된다.
LoRA는 encoder/decoder를 바꾸므로 이 cache를 학습에 재사용하지 않는다.

한 학습 batch는 같은 원점의 전체 계층이다. 서로 다른 월의 부모·자식을 같은 집단으로 잘못 섞지 않는다.
GPU 메모리가 부족하면 정확한 gradient accumulation을 사용한다. 특정 계층 노드를 삭제하거나 gradient를 임의 detach하지 않는다.

## 8. 점예측·학습 목적

이번 파일럿은 점예측만 다룬다.
Chronos의 0.5 출력 채널에서 첫 12시점을 점예측으로 사용하고, 같은 scaled MSE로 학습한다. 이 출력이 학습 후 올바른 분위수라는 주장은 하지 않는다.
기존 quantile head 전체는 동결하되, gradient는 필요한 적응 파라미터에 전달한다.

각 노드 i의 scale q_i는 TRAIN의 12개월 seasonal difference RMS로 고정한다.
q_i가 0인 경우는 사전 정의된 1e-6*max(TRAIN RMS_i,1) floor를 쓰고 해당 노드를 별도로 보고한다.
계층 깊이마다 같은 총 가중치, 같은 깊이 안에서는 노드별 같은 가중치로 scaled squared error를 평균한다.
같은 목적·같은 노드 가중치를 L/G/H/P에 모두 적용한다.
계층 불일치 벌점은 H에만 추가하지 않는다.
LoRA가 F0보다 나빠야 한다는 진입 조건은 없다. 초기 합계 불일치가 작다는 이유로도 직접 비교를 차단하지 않는다.

## 9. 합계 조정과 강한 기준선

주 조정: MinT-shrink.
- 공식 구현 및 estimator를 사용한다. 새 covariance shrinkage 수식을 발명하지 않는다.
- 각 모델/체크포인트의 CAL 1-step 예측 오차로 공분산을 추정한다.
- shrinkage, numerical regularization의 방식은 동일하게 고정하고 실제 수치를 기록한다.
- 이는 모델마다 다른 covariance로 만든 같은 조정 알고리즘이다. '완전히 같은 행렬'이라고 쓰지 않는다.
- V에서 조정 후의 주지표로 checkpoint와 LR을 선택한다.
- 선택 후 W와 모델을 고정하고 E를 채점한다.
- E 오차로 W를 다시 추정하지 않는다.

학습0회의 CPU/추론 기준선:
- Seasonal Naive: 12개월 주기.
- AutoETS: 원점 이전의 같은 관측 권한 사용, 미래 정답 사용 금지.
- F0.
각 기준선과 신경망에 post-processing Bottom-up 및 OLS도 함께 계산해, 적절한 단순 조정이 더 좋은지 확인한다. 추가 neural fit은 없다.
주 비교는 MinT-shrink로 고정한다. E에서 가장 좋은 조정만 골라 주 결과를 바꾸지 않는다. V에서 고른 고전 기준선 및 단순 조정이 더 좋으면 그 결과도 함께 보고한다.

보조 통제:
- G/H/P의 선택된 예측을 L의 고정 MinT 행렬로도 조정해 공통 조정 아래의 효과를 본다.
- 이 보조 결과가 주 결과보다 좋아도 사후에 주 결과로 바꾸지 않는다.

수치 합계는 반드시 원래 단위의 점예측에 적용한다. 노드별 표준화된 값을 그대로 더하지 않는다.
음수 예측을 0으로 사후 clamp하면 합계가 깨질 수 있으므로 임의 clipping하지 않는다. 음수 발생을 보고하고 비음수 reconciliation은 후속 주제로 분리한다.
각 노드의 분위수를 더해 coherent probabilistic forecast라고 주장하지 않는다.

## 10. 평가 지표

주지표: 계층 깊이별 동일 가중 RMSSE.
1. 노드마다 TRAIN seasonal scale로 나눈 평가 MSE의 제곱근(RMSSE)을 계산한다.
2. 각 계층 깊이 안에서 노드 평균.
3. 깊이별 평균을 동일 가중으로 평균.

보조:
- 각 깊이의 RMSSE.
- 최하위 RMSSE.
- 전체 MAE 및 노드별 scaled MAE.
- 미조정 예측과 조정 후 예측을 나란히 보고.
- 합계 불일치는 correctness 지표이지 제안의 성공 지표가 아니다.
- 연도/12개월 블록별 차이와 optimizer seed별 차이.
- 실제 학습/추론 시간, 메모리, 추가 파라미터와 캐시 크기.

95% paired time-block 구간은 보조로만 보고한다. 마지막60개월의 월별 겹치는 예측을 독립 사건으로 세지 않는다. seed2개·한 자료에서 확증이나 모집단 우위를 주장하지 않는다.

## 11. 학습 예산

네 학습군 L/G/H/P.
선택용 seed92150에서 LR {1e-4,3e-4}: 4*2=8 fits.
선택 LR을 반복 seed92151/92152에서 실행: 4*2=8 fits.
합계16 fits.
각 fit 최대512 updates; 체크포인트0/128/256/512.
본학습 총상한8,192 updates.
네 군 실제 모델 smoke 2 updates씩, 총8 updates; 폐기 후 본학습.
총상한8,200 updates.

동일 학습 원점 순서/초기값 규칙, AdamW(.9,.999), eps1e-8, wd0, clip-norm1, FP32, TF32 off, dropout0.
학습 원점은 한 번씩 섞어서 순회한 후 반복한다. 잘 맞는 월만 sampling하지 않는다.
동률은 작은 LR, 이른 checkpoint를 선택한다.
선택이 모두 최대 checkpoint 또는 LR 경계에 몰리면 budget-limited라는 한계를 남긴다. 추가 LR/epoch를 자동 시작하지 않는다.
학습 성능이 나쁘다고 계획된 정상 대조를 중간에 제거하지 않는다. 기술적 오류·자원 상한에서는 정확한 재개 상태를 남긴다.

## 12. 성공·중단의 해석

다음은 서로 다른 판단이다.

- L+MinT가 좋아짐: 기존 PEFT+조정의 유용성. 새 방법의 성공이 아니다.
- H가 L보다 좋아도 G와 비슷함: 정보 결합/추가 용량으로 설명될 수 있음.
- H가 G보다 좋아도 P와 비슷함: 정확한 계층 연결의 필요성은 약함.
- H가 L/G/P보다 좋아짐: 계층 관계를 이용하는 적응의 제한된 후속 개발 근거.
- 차이가 작거나 seed 방향이 다름: 불확실. 동등성 또는 명확한 실패로 강제하지 않는다.
- H가 직접 대조들보다 반복적으로 나쁨: 이 어댑터 구현의 추가 개발은 중단. 계층형 PEFT 전체의 불가능성으로 쓰지 않는다.
- 합계만 잘 맞고 미래 오차가 줄지 않음: 목표 미지지.
- raw/shape/모델 연결 문제: 성능 실패가 아니라 구현 또는 데이터 문제.

프로젝트 범위 종료:16 fits와 고정 비교를 마치면 무조건 멈춘다. 추가 seed, 다른 branch, 다른 relation/gate/rank/loss를 자동 탐색하지 않는다.
진짜 양성 근거가 있을 때만 TourismLarge의 동일 질문 재현을 다음 계획으로 제안한다. 자동 실행하지 않는다. TourismLarge는 규모와 grouped hierarchy 계약이 달라 별도 확인이 필요하다.

## 13. 구현 전 필수 확인

- 공식 raw 파일 확보·라이선스/출처·노드 ID·S/계층 계약.
- TRAIN/CAL/V/E의 실제 timestamp 경계.
- 미래 y를 바꿔도 입력·연결·캐시·예측이 바뀌지 않는지.
- H/G/P의 초기 출력 F0 동일성, 학습 파라미터 이름·수.
- 같은 원점의 전체 계층을 묶는 batch 축.
- decoder gradient 전달 및 frozen parameter 불변.
- permutation 대조에서 S와 정답이 원래대로인지.
- MinT의 covariance가 CAL에서만 계산되는지.
- 조정 후 projection identity 및 합계 일치.
- 모든 선택을 고정한 뒤 E 채점.

죽은 검사나 불가능한 조건을 숫자로 채우지 않는다. 기술적 미확인을 기록한다.

## 14. 결과 문서가 답할 질문

1. 계층 데이터와 실제 정답이 일치했는가?
2. 일반 LoRA+MinT가 얼마이고, H가 얼마를 더 줄였는가?
3. G/P를 넘어 계층 연결의 추가 가치가 있는가?
4. 어떤 수준이 좋아지고 어떤 수준이 손해인가?
5. F0/SeasonalNaive/AutoETS 같은 강한 단순 기준선보다 쓸 이유가 있는가?
6. 계산 비용과 불확실성을 고려해 후속 검토할 가치가 있는가?

권장 산출물: PROTOCOL, DATA_AUDIT, S/노드 매핑, split/origin manifest, fit ledger, selection/calibration manifest, raw forecasts, before/after reconciliation scores, paired effects, resource report, verification, REPORT, NEXT_DECISION.
많은 그림 대신 주 결과표1개, 수준별 오차 그림1개, 기간별 효과 그림1개를 우선한다.

## 15. 확인한 일차 자료

- Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization (2019/JASA), DOI:10.1080/01621459.2018.1448825.
- End-to-End Learning of Coherent Probabilistic Forecasts for Hierarchical Time Series (2021/ICML), PMLR139:8832–8843.
- Coherent Probabilistic Forecasting of Temporal Hierarchies (2023/AISTATS), PMLR206:9362–9376.
- Nixtla datasetsforecast: datasetsforecast/hierarchical.py, Labour 데이터/계층/전처리 계약.
- Nixtla HierarchicalForecast: Quick Start, hierarchical_baselines 예제.
- Forecasting: Principles and Practice, 3rd ed., 11.3 Forecast reconciliation.
- amazon-science/chronos-forecasting: src/chronos/chronos_bolt.py, encode/decode 분리.

설계 단계 검산: 작은 인공 합계행렬에서 projection identity/합계 일치를 확인했고, 어댑터12,808개·16fits·8,192+8updates의 산술을 확인했다.
실행하지 않은 것: 실제 Labour 자료 판독, 새 Chronos wrapper 구현, backbone inference, optimizer 학습, 실데이터 평가. 이 문서는 실행 결과가 아니다.
