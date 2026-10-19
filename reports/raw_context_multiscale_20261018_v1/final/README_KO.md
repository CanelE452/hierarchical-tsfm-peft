# 원본·문맥 표현 분리와 다중 해상도 잠재 예측

[확인] 원채널 normalized observed channel-macro MSE가 주지표이고 MAE는 보조지표다. 모든 완료 seed와 불리한 결과를 포함했다.

S는 ALL_CONTEXT B2, A는 앞 K/2 PCA 축의 원본 투영과 뒤 K/2 축의 문맥 투영, B는 A의 문맥 잠재 경로만 2배 평균한 모델이다. P는 모두 길이512에서 계산하며 anchor는 한 번만 더한다. B의 F 입력은 실제512/256이고 decoder lag는 원격자에 적용한다.

V는 반복 사용된 개발 평가다. 새 모델·전이·head 선택은 V 전에 봉인했지만 V의 과거 노출, 상류 전처리와 사전학습 중복 가능성을 지우지 않는다. 보호 T에 새 수치 접근을 하지 않았다.

```text
dataset   mode          seed       MSE           MAE
jena      S             92601      0.6389587062  0.4828335372
jena      S             92602      0.6325724866  0.4812597265
jena      A             92601      0.7173798385  0.5044833698
jena      A             92602      0.7796831497  0.537571334
jena      B             92601      0.7210013191  0.4987407039
jena      B             92602      0.7097857276  0.5210459219
pems32    S             92601      0.09106836154  0.2000302368
pems32    S             92602      0.09221473521  0.2026237781
pems32    A             92601      0.09578883944  0.2073904497
pems32    A             92602      0.09648034572  0.2065680412
jena      G             92601      0.6322170504  0.4415212863
jena      Raw           None       0.8491941206  0.4813883423
pems32    G             92601      0.08146537301  0.1857749764
pems32    Raw           None       0.0905137861  0.1952114879
jena      LoRA          92601      0.5639229488  0.3990099355
jena      Ridge         92601      0.5566463357  0.4055218018
```

두 seed 산술평균과 각 seed는 RUNS.csv에 모두 있다. 원점·채널·겹치는 창을 독립 표본으로 간주한 신뢰구간이나 유의성 검정은 만들지 않았다.

현재 등록 계획: EXPERIMENT_PLAN_REPAIR_002.json. 수리 리비전이 있으면 원래 계획의 바이트를 보존하고, 같은 과학적 계약을 승계하는 해당 리비전의 지문을 검산에 연결한다.

B8 native 비용의 같은 장치·메모리 계층 안에서 비교한다. 아래 ms는 배치 전체와 원점당 값을 구분한다.

```text
dataset   mode    seed    device  ms/batch   ms/origin  allocatedMiB  reservedMiB
jena      S       92601   cuda    20.6657    2.58321    294.206       310
jena      S       92602   cuda    20.8342    2.60427    294.206       310
jena      A       92601   cuda    20.4626    2.55782    294.394       310
jena      A       92602   cuda    20.8177    2.60221    294.394       310
jena      B       92601   cuda    32.0855    4.01069    243.153       270
jena      B       92602   cuda    31.9595    3.99494    243.153       270
pems32    S       92601   cuda    17.2598    2.15747    260.438       262
pems32    S       92602   cuda    16.6627    2.08284    260.438       262
pems32    A       92601   cuda    19.7447    2.46809    260.563       262
pems32    A       92602   cuda    17.315     2.16437    260.563       262
jena      G       92601   cuda    0.6109     0.0763625  10.2002       24
jena      Raw     None    cuda    40.307     5.03837    330.083       354
pems32    G       92601   cuda    0.5679     0.0709875  10.7222       24
pems32    Raw     None    cuda    49.815     6.22687    330.708       354
jena      LoRA    92601   cuda    38.5048    4.8131     330.083       354
jena      Ridge   92601   cpu     0.5676     0.07095    N/A           N/A
jena      S       mean    cuda    20.7499    2.59374    294.206       310
jena      A       mean    cuda    20.6401    2.58002    294.394       310
jena      B       mean    cuda    32.0225    4.00282    243.153       270
pems32    S       mean    cuda    16.9612    2.12015    260.438       262
pems32    A       mean    cuda    18.5299    2.31623    260.563       262
```

mean은 두 seed facet 요약값의 산술평균이며 별도의 실행 측정치나 독립 반복을 뜻하지 않는다. per-PID GPU 측정 상태와 RSS는 COST.csv에 별도 열로 공개했다.

jena S_to_A: 두 seed 모두 MSE 증가/동일; 평균 ΔMSE=+0.1127658977, 평균 ΔMAE=+0.03898072003.
pems32 S_to_A: 두 seed 모두 MSE 증가/동일; 평균 ΔMSE=+0.004493044205, 평균 ΔMAE=+0.005652238012.
jena A_to_B: seed별 방향이 다른 혼합 결과; 평균 ΔMSE=-0.03313797077, 평균 ΔMAE=-0.01113403901.
pems32 A_to_B: 등록된 전이 family에 포함되지 않아 대응 비교가 없다.
jena 원본 예측 결합: 두 seed 모두 conditional MSE가 낮았다; conditional−past_control 평균 ΔMSE=-0.02860071064, ΔMAE=-0.001202477948.
pems32 원본 예측 결합: 두 seed 모두 conditional MSE가 낮았다; conditional−past_control 평균 ΔMSE=-0.003414716441, ΔMAE=-0.00023853064.

정확도와 비용은 T1_ACCURACY_COST.csv, 원인 대조는 T2_SINGLE_HYPOTHESIS.csv, 결합은 T3_OUTPUT_COMBINATION.csv 및 COMPLEMENTARITY.csv에 있다. F3_NATIVE_ALL 그림은 COST의 모든 완료 native 경로를 데이터·장치·B1/B8별로 분리하며 alpha·conditional·past_control과 CPU Ridge도 포함한다. F3_MATCHED_TWO_CALL은 B1/B8을 나누어 각 seed의 A native/two_call과 B를 보여준다. FIGURE_POINTS.csv와 FIGURE_MANIFEST.json에 실제 그림의 경로별 식별자를 기록했다.

A의 native 한 호출과 비용 전용 two_call을 함께 공개한다. B의 낮은 요청 미래 길이는 Bolt 고정 head의 연산이 비례해 줄었다는 증거가 아니다. T3의 head_search_seconds/head_refit_seconds는 적합 비용이며 standalone 부모 지연을 빼서 head-only 추론 시간으로 해석하지 않는다.

G가 MSE·whole-path 지연·allocated 피크에서 모두 지배하는 행은 T1의 G_dominates_MSE_latency_allocated에 표시했다. 이런 조건에서는 F를 유지할 실용 선택 이유가 확보되지 않는다. Conditional이 past_control보다 나아야 추가 B2 조건 정보 가치의 근거가 생기며, 이것만으로 F의 인과적 필요성이나 일반적 우수성이 증명되지는 않는다.

M_fit 적합, M_select 선택, A_stop 선택, V 보고 점수는 다른 열이다. M_cal_batch_fit_loss 및 TRAIN_segment_batch_loss는 학습 도중 배치 평균 손실로서 전체 구간의 현재 checkpoint MSE와 다르다. 새 본체 segment는 기록했고, 과거 재사용 S의 미기록 segment를 복원하거나 만들어 넣지 않았다.

Ridge는 CPU 지연·RSS 맥락이다. CPU/GPU 속도비를 만들지 않으며 allocated, reserved, 측정 가능한 per-PID GPU memory를 다른 열로 유지한다. WDDM에서 per-PID 측정이 불가하면 N/A다. p95는20회 trial의 서술 분위수이며 서비스 꼬리 지연 보장이 아니다.

실제 캠페인 wall=41431.873초; GPU 소유 wall 상한=8167.840초. GPU kernel busy 시간은 측정하지 않았다. 새 본체 설정 6개, 본체 fresh refit 8개, 재사용 S 2개, head search/fresh refit 각각 8/8개다.

cap에서 계속 좋아지는 경로는 BUDGET_LIMITED_CONVERGENCE_UNRESOLVED이며 수렴/구조적 실패로 판정하지 않는다. 실제 제외·연장 이유와 재개 명령은 RESUME_QUEUE.md를 따른다.



원격 게시 여부는 별도 실제 원격 검증 기록을 따른다. 이 보고 생성만으로 remote verified를 선언하지 않는다.

## 최종 보고 보완의 범위

이 final/ 보고서는 primary 공개 CSV와 독립 검산을 입력으로 생성했다. 기존 root 보고·13개 producer 소스·선택·예측·오차 원수치를 바꾸지 않았다. T1은 K 분할과 scales, T2는 동일한 실제 V 원점 identity와 후보·기준선 각각의 search/refit 학습량을 추가했다.

RUNS의 two_seed_mean은 V MSE·MAE의 산술평균이다. selected epoch/A_stop/M_select/alpha/수렴 상태는 평균으로 가장하지 않고 빈칸 및 by_seed_auxiliary_values로 남겼다. M_cal mean batch loss는 checkpoint 전체 M_cal MSE가 아니다. CP0 A_stop MSE·MAE는 학습 전 고정 모델의 추가 진단이며 체크포인트·전이 선택을 다시 하지 않았다.

정확한 반열린 행 범위와 원점 규칙은 DATA_SCOPES.csv에 있다. 보호 T 행은 범위만 인용하고 새 입력·통계·평가를 하지 않았다.

```text
jena    A_fit     [0,10482) origins=9923 stride=1
jena    A_stop    [10482,12579) origins=43 stride=48
jena    A60       [0,12579) origins=12020 stride=1
jena    M_cal     [12579,14675) origins=2049 stride=1
jena    M_fit     [12579,13836) origins=1210 stride=1
jena    M_select  [13836,14675) origins=792 stride=1
jena    V_select  [14675,16772) origins=43 stride=48
jena    T_final   [16772,20965) origins=PROTECTED stride=None
pems32  A_fit     [0,8928) origins=8405 stride=1
pems32  A_stop    [8928,10713) origins=148 stride=12
pems32  A60       [0,10713) origins=10190 stride=1
pems32  M_cal     [10713,12499) origins=1775 stride=1
pems32  M_fit     [10713,11784) origins=1060 stride=1
pems32  M_select  [11784,12499) origins=704 stride=1
pems32  V_select  [12499,14284) origins=148 stride=12
pems32  T_final   [14284,17856) origins=PROTECTED stride=None
```

F1_CAP_MARKED는 실제 update 축에서 선택 star와 cap square를 분리한다. F1_CP0_DIAGNOSTIC는 모든 완료 S/A/B의 CP0 두 지표를, F2_HORIZON_THIRDS는 각 seed의 near/middle/far ΔMSE·ΔMAE를 생략 없이 보여준다. F3_NATIVE_ALL/FIGURE_POINTS의 모든 32 alpha·8 head·직접 기준선·CPU Ridge·B1/B8 facet은 primary와 동일하다. 장치와 memory layer가 다른 결과의 비율을 만들지 않는다.

자세한 실제 사용법·immutable deadline·재개 제한은 CLI_USAGE_KO.md, 미완료 extension은 RESUME_QUEUE.md를 따른다. V는 반복 사용된 개발 평가이며 미사용 독립 테스트가 아니다. 추가 fixed-inference/보고 시간은 FINAL_CONTEXT.json의 실제 resource receipt와 원 primary RESOURCE_LEDGER를 구분해 읽는다.

발표에서 지지 가능한 문장:
- 이 고정 분할·PCA 축 배분·총 K·학습 기회에서 관찰한 seed별 방향과 평균을 함께 제시한다.
- A와 B의 대조는 문맥 평균화·낮은 격자 F 예측·원격자 반복 복원의 조합에 한정한다.
- 정확도와 실제 native/matched 실행비용의 교환관계 및 DirectG와의 열세를 함께 밝힌다.
- cap에서 개선 중이면 등록 예산에서의 결과이며 수렴은 미확정이라고 말한다.

아직 쓰면 안 되는 문장:
- 이 V를 미사용 독립 테스트라고 부르거나 두 seed·겹치는 창·채널로 유의성/신뢰구간을 만든다.
- 같은 K가 동일 FLOPs/메모리를 보장하거나 coarse 출력 일부를 읽으면 F head 연산도 비례 감소한다고 말한다.
- 이 결과만으로 F의 필수성·모든 데이터에서의 우위·TimeMixer/NHITS 전체 재현을 입증했다고 말한다.
- 학습 상한 또는 반복 복원의 한계를 구조적 실패/고해상도 완전 복원으로 단정한다.
