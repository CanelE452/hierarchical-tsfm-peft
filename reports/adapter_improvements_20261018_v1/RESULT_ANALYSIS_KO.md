# 결과 해석과 적용 범위

실행 PARTIAL / 독립 검증 PASS / 연구 DESCRIPTIVE_DEVELOPMENT_EVIDENCE / 게시 SEPARATE_REMOTE_PROOF

[확인] 신규 과학 job 완료 15개, 검증 재사용 job 2개, 신규 선택 결과 arm 24개. 과거 contextual 선택 요약 59행과 과거 endpoint 88행은 별도 보존하며 신규 실행·재사용 job 수에 포함하지 않는다.

선택 결과 핵심 해석 — MATCHED selected_refit / V_select만 사용한다.

[확인] P 선택-refit V의 분해−대조 효과(음수 개선)는 jena seed 92601: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE +0.0410513 (악화), MAE +0.0297769 (악화); jena seed 92602: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE +0.0447922 (악화), MAE +0.028867 (악화); pems32 seed 92601: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE +2.35799e-05 (악화), MAE +0.00025385 (악화); pems32 seed 92602: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE +0.000192861 (악화), MAE +0.000315723 (악화).

[확인] E 초기화의 미래 예측 효과는 pems32 seed 92601: B2_DCA−B2_PCA MSE +0.00166681 (악화), MAE +0.00221688 (악화); pems32 seed 92602: B2_DCA−B2_PCA MSE -8.92933e-05 (개선), MAE -5.08465e-07 (개선); jena DCA A60/A_fit NOT_CONVERGED로 정상 완료 PCA/DCA 선택-refit 효과는 UNKNOWN. 미수렴 초기화의 objective를 예측 성능으로 바꾸지 않는다.

[확인] D의 같은 fixed P/E 선택 recipe에서 jena seed 92601: D_LINEAR−D_SHARED MSE +0.00717886 (악화), MAE -0.00100378 (개선); D_LINEAR−D_GAIN MSE +0.0054466 (악화), MAE +0.00173157 (악화); jena seed 92602: D_LINEAR−D_SHARED MSE -0.000220504 (개선), MAE -0.00365156 (개선); D_LINEAR−D_GAIN MSE -0.00391076 (개선), MAE -0.00291477 (개선); pems32 seed 92601: D_LINEAR−D_SHARED MSE -0.000450248 (개선), MAE -0.000527012 (개선); D_LINEAR−D_GAIN MSE -0.000383894 (개선), MAE -0.000372256 (개선). MSE와 MAE 및 두 seed의 방향을 각각 읽으며 이를 단일 승자로 합치지 않는다.

[확인] 같은 측정 job/device/batch와 정확한 저장 예측·bundle의 full-V parity를 통과한 Raw 대비 비용은 jena cpu B1 [reports/adapter_improvements_20261018_v1/jobs/COST_JENA_CPU.json]: P_SPLIT_PCA seed 92601 49.6176 ms/origin, Raw speedup 2.71763x ~ D_HORIZON seed 92601 47.4045 ms/origin, Raw speedup 2.84451x; jena cpu B8 [reports/adapter_improvements_20261018_v1/jobs/COST_JENA_CPU.json]: P_LOCAL_CONTROL_PCA seed 92601 42.2184 ms/origin, Raw speedup 3.21962x ~ D_GAIN seed 92601 40.2125 ms/origin, Raw speedup 3.38022x; jena cpu B1 [reports/adapter_improvements_20261018_v1/jobs/COST_JENA_CPU_s92602.json]: P_LOCAL_CONTROL_PCA seed 92602 48.666 ms/origin, Raw speedup 2.76465x ~ D_SHARED seed 92602 47.2832 ms/origin, Raw speedup 2.84551x; jena cpu B8 [reports/adapter_improvements_20261018_v1/jobs/COST_JENA_CPU_s92602.json]: P_SPLIT_PCA seed 92602 42.0134 ms/origin, Raw speedup 3.23207x ~ D_LINEAR seed 92602 40.3129 ms/origin, Raw speedup 3.36841x; jena cuda B1 [reports/adapter_improvements_20261018_v1/jobs/COST_JENA_CUDA.json]: P_SPLIT_PCA seed 92601 14.9355 ms/origin, Raw speedup 0.799076x (Raw보다 느림) ~ D_GAIN seed 92601 12.121 ms/origin, Raw speedup 0.984622x (Raw보다 느림); jena cuda B8 [reports/adapter_improvements_20261018_v1/jobs/COST_JENA_CUDA.json]: P_SPLIT_PCA seed 92601 1.98829 ms/origin, Raw speedup 1.89071x ~ D_HORIZON seed 92601 1.84084 ms/origin, Raw speedup 2.04215x; jena cuda B1 [reports/adapter_improvements_20261018_v1/jobs/COST_JENA_CUDA_s92602.json]: P_LOCAL_CONTROL_PCA seed 92602 12.4297 ms/origin, Raw speedup 0.973052x (Raw보다 느림) ~ P_SPLIT_PCA seed 92602 11.6479 ms/origin, Raw speedup 1.03836x; jena cuda B8 [reports/adapter_improvements_20261018_v1/jobs/COST_JENA_CUDA_s92602.json]: P_LOCAL_CONTROL_PCA seed 92602 1.82791 ms/origin, Raw speedup 2.03988x ~ D_SHARED seed 92602 1.77644 ms/origin, Raw speedup 2.09897x. 그 외 선택 run 12개의 같은 조건 Raw 비용 비교는 UNKNOWN. Historical 비용은 별도 contextual 조건이며 이 speedup에 섞지 않는다.

선택-refit V 상세 — 각자의 A_stop 선택 epoch를 허용한 recipe 대응이다.

- [확인] jena seed 92601: D_HORIZON−D_SHARED MSE 0.0115681 (1.70705%), MAE -0.00230992 (-0.466711%). 대응 상태 MATCHED_RECIPE_SELECTED_REFIT. 비용은 동일 bundle/device/batch의 COST.csv가 있는 경우에만 연결한다.
- [확인] jena seed 92601: D_LINEAR−D_SHARED MSE 0.00717886 (1.05935%), MAE -0.00100378 (-0.202811%). 대응 상태 MATCHED_RECIPE_SELECTED_REFIT. 비용은 동일 bundle/device/batch의 COST.csv가 있는 경우에만 연결한다.
- [확인] jena seed 92601: D_GAIN−D_SHARED MSE 0.00173226 (0.255622%), MAE -0.00273535 (-0.552667%). 대응 상태 MATCHED_RECIPE_SELECTED_REFIT. 비용은 동일 bundle/device/batch의 COST.csv가 있는 경우에만 연결한다.
- [확인] jena seed 92602: D_HORIZON−D_SHARED MSE 0.00278211 (0.421886%), MAE -0.003204 (-0.653046%). 대응 상태 MATCHED_RECIPE_SELECTED_REFIT. 비용은 동일 bundle/device/batch의 COST.csv가 있는 경우에만 연결한다.
- [확인] jena seed 92602: D_LINEAR−D_SHARED MSE -0.000220504 (-0.0334378%), MAE -0.00365156 (-0.744269%). 대응 상태 MATCHED_RECIPE_SELECTED_REFIT. 비용은 동일 bundle/device/batch의 COST.csv가 있는 경우에만 연결한다.
- [확인] jena seed 92602: D_GAIN−D_SHARED MSE 0.00369026 (0.5596%), MAE -0.000736793 (-0.150175%). 대응 상태 MATCHED_RECIPE_SELECTED_REFIT. 비용은 동일 bundle/device/batch의 COST.csv가 있는 경우에만 연결한다.
- [확인] jena seed 92601: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE 0.0410513 (5.55164%), MAE 0.0297769 (5.91592%). 대응 상태 MATCHED_RECIPE_SELECTED_REFIT. 비용은 동일 bundle/device/batch의 COST.csv가 있는 경우에만 연결한다.
- [확인] jena seed 92602: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE 0.0447922 (6.11884%), MAE 0.028867 (5.80281%). 대응 상태 MATCHED_RECIPE_SELECTED_REFIT. 비용은 동일 bundle/device/batch의 COST.csv가 있는 경우에만 연결한다.
- [확인] pems32 seed 92601: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE 2.35799e-05 (0.0238106%), MAE 0.00025385 (0.120639%). 대응 상태 MATCHED_RECIPE_SELECTED_REFIT. 비용은 동일 bundle/device/batch의 COST.csv가 있는 경우에만 연결한다.
- [확인] pems32 seed 92601: B2_DCA−B2_PCA MSE 0.00166681 (1.84432%), MAE 0.00221688 (1.11623%). 대응 상태 MATCHED_RECIPE_SELECTED_REFIT. 비용은 동일 bundle/device/batch의 COST.csv가 있는 경우에만 연결한다.
- [확인] pems32 seed 92602: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE 0.000192861 (0.194016%), MAE 0.000315723 (0.150665%). 대응 상태 MATCHED_RECIPE_SELECTED_REFIT. 비용은 동일 bundle/device/batch의 COST.csv가 있는 경우에만 연결한다.
- [확인] pems32 seed 92602: B2_DCA−B2_PCA MSE -8.92933e-05 (-0.0978383%), MAE -5.08465e-07 (-0.000254763%). 대응 상태 MATCHED_RECIPE_SELECTED_REFIT. 비용은 동일 bundle/device/batch의 COST.csv가 있는 경우에만 연결한다.
- [확인] pems32 seed 92601: D_HORIZON−D_SHARED MSE -0.000660581 (-0.735407%), MAE -0.000645384 (-0.326584%). 대응 상태 MATCHED_RECIPE_SELECTED_REFIT. 비용은 동일 bundle/device/batch의 COST.csv가 있는 경우에만 연결한다.
- [확인] pems32 seed 92601: D_LINEAR−D_SHARED MSE -0.000450248 (-0.501248%), MAE -0.000527012 (-0.266684%). 대응 상태 MATCHED_RECIPE_SELECTED_REFIT. 비용은 동일 bundle/device/batch의 COST.csv가 있는 경우에만 연결한다.
- [확인] pems32 seed 92601: D_GAIN−D_SHARED MSE -6.63536e-05 (-0.0738696%), MAE -0.000154756 (-0.0783115%). 대응 상태 MATCHED_RECIPE_SELECTED_REFIT. 비용은 동일 bundle/device/batch의 COST.csv가 있는 경우에만 연결한다.

공통 endpoint 상세 — 선택-refit 점수와 별개이며 같은 epoch/update/exposure/sampler 조건이다.

- [확인] jena seed 92601 A_fit epoch 0: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE 0, MAE 0. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] jena seed 92601 A_stop epoch 0: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE 0, MAE 0. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] jena seed 92601 A_fit epoch 0.25: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE -0.00038468, MAE -0.00109741. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] jena seed 92601 A_stop epoch 0.25: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE -0.00955431, MAE -0.00735945. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] jena seed 92601 A_fit epoch 1: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE 1.06278e-05, MAE -0.000792583. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] jena seed 92601 A_stop epoch 1: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE -0.0104473, MAE -0.008727. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] jena seed 92601 A_fit epoch 3: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE 0.00082103, MAE 0.00244178. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] jena seed 92601 A_stop epoch 3: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE 0.0025242, MAE -0.00163408. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] jena seed 92602 A_fit epoch 0: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE 0, MAE 0. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] jena seed 92602 A_stop epoch 0: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE 0, MAE 0. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] jena seed 92602 A_fit epoch 0.25: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE 0.00151425, MAE 0.00186469. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] jena seed 92602 A_stop epoch 0.25: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE 0.00348617, MAE 0.0025362. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] jena seed 92602 A_fit epoch 1: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE 0.00480705, MAE -0.00178838. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] jena seed 92602 A_stop epoch 1: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE -0.0101555, MAE -0.00803835. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] jena seed 92602 A_fit epoch 3: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE 0.00135966, MAE 0.000988219. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] jena seed 92602 A_stop epoch 3: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE 0.00516189, MAE 0.0012704. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] pems32 seed 92601 A_fit epoch 0: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE 0, MAE 0. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] pems32 seed 92601 A_stop epoch 0: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE 0, MAE 0. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] pems32 seed 92601 A_fit epoch 0.25: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE 0.000415861, MAE 0.000530317. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] pems32 seed 92601 A_stop epoch 0.25: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE 0.000474685, MAE 0.000774236. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] pems32 seed 92601 A_fit epoch 1: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE -9.86608e-05, MAE 3.93661e-05. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] pems32 seed 92601 A_stop epoch 1: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE 9.0328e-06, MAE 3.79222e-05. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] pems32 seed 92601 A_fit epoch 3: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE -1.17239e-05, MAE -0.00020635. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] pems32 seed 92601 A_stop epoch 3: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE 0.000423423, MAE 3.86424e-05. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] pems32 seed 92601 A_fit epoch 0: B2_DCA−B2_PCA MSE 0.00676765, MAE 0.00534339. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] pems32 seed 92601 A_stop epoch 0: B2_DCA−B2_PCA MSE 0.0103348, MAE 0.00667097. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] pems32 seed 92601 A_fit epoch 0.25: B2_DCA−B2_PCA MSE 0.00379266, MAE 0.00143667. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] pems32 seed 92601 A_stop epoch 0.25: B2_DCA−B2_PCA MSE 0.00792801, MAE 0.00476314. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] pems32 seed 92601 A_fit epoch 1: B2_DCA−B2_PCA MSE 0.00356934, MAE 0.00080862. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] pems32 seed 92601 A_stop epoch 1: B2_DCA−B2_PCA MSE 0.00665783, MAE 0.0011403. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] pems32 seed 92601 A_fit epoch 3: B2_DCA−B2_PCA MSE 0.00128892, MAE 0.000559662. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] pems32 seed 92601 A_stop epoch 3: B2_DCA−B2_PCA MSE 0.000246717, MAE -0.000275416. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] pems32 seed 92602 A_fit epoch 0: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE 0, MAE 0. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] pems32 seed 92602 A_stop epoch 0: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE 0, MAE 0. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] pems32 seed 92602 A_fit epoch 0.25: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE 0.000241434, MAE 0.000336464. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] pems32 seed 92602 A_stop epoch 0.25: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE 0.000435339, MAE 0.000627961. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] pems32 seed 92602 A_fit epoch 1: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE -0.000386659, MAE -2.29527e-06. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] pems32 seed 92602 A_stop epoch 1: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE -0.000731159, MAE -0.000719732. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] pems32 seed 92602 A_fit epoch 3: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE -0.000491632, MAE -0.000989236. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] pems32 seed 92602 A_stop epoch 3: P_SPLIT_PCA−P_LOCAL_CONTROL_PCA MSE -0.000574461, MAE -0.00106528. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] pems32 seed 92602 A_fit epoch 0: B2_DCA−B2_PCA MSE 0.00676765, MAE 0.00534339. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] pems32 seed 92602 A_stop epoch 0: B2_DCA−B2_PCA MSE 0.0103348, MAE 0.00667097. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] pems32 seed 92602 A_fit epoch 0.25: B2_DCA−B2_PCA MSE 0.004902, MAE 0.00236773. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] pems32 seed 92602 A_stop epoch 0.25: B2_DCA−B2_PCA MSE 0.00778214, MAE 0.004339. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] pems32 seed 92602 A_fit epoch 1: B2_DCA−B2_PCA MSE 0.00481278, MAE 0.00174544. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] pems32 seed 92602 A_stop epoch 1: B2_DCA−B2_PCA MSE 0.00622096, MAE 0.000990769. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] pems32 seed 92602 A_fit epoch 3: B2_DCA−B2_PCA MSE 0.00144946, MAE 0.000252588. 대응 상태 MATCHED_COMMON_ENDPOINT.
- [확인] pems32 seed 92602 A_stop epoch 3: B2_DCA−B2_PCA MSE 0.00111464, MAE 0.00157644. 대응 상태 MATCHED_COMMON_ENDPOINT.

[확인] 등록 DCA 초기화 2개 scope가 NOT_CONVERGED다. 등록 optimizer 한도에서 종료했으며 downstream DCA 예측은 정상 완료로 기록하지 않는다. PCA 곡선만 있으면 DCA 미래 점수를 만들지 않는다.

P 분해 효과의 주 대조는 P_SPLIT−P_LOCAL_CONTROL이다. B2 대비 차이는 입력 구조·용량·연산 방식 전체의 차이다. E 비교는 B2+PCA 대 B2+DCA이며 초기 objective/수치 유효성을 미래 예측 개선으로 대신하지 않는다. D 비교는 같은 고정 P/E에서 각 head가 A_stop으로 선택한 lambda/gamma를 사용한 recipe 비교다.

Jena 밴드는 h1–16/17–32/33–48, PeMS32는 h1–4/5–8/9–12다. 전체 macro MSE/MAE는 채널별 SSE/SAE/count를 먼저 합산한 뒤 고정 채널 집합의 ratio를 평균한다. 밴드 macro 값을 직접 더하지 않는다. 관측이 없는 채널을 조용히 빼지 않으며 해당 밴드는 UNDEFINED로 표시한다.

두 seed를 전체 분포의 안정성이나 보편적 유의성으로 해석하지 않는다. origin별 의존성이 있어 IID bootstrap/p-value를 자동 추가하지 않았다. delta는 candidate−control이며 음수가 오차 개선이다.

PAIRED_ORIGIN_EFFECTS.csv는 MATCHED selected-refit V 비교의 origin_index별 candidate/control MSE·MAE와 delta다. 공개 producer JSON의 채널 SSE/SAE/count를 고정 C채널 macro로 계산하며 어느 채널의 count가 0이면 해당 origin은 UNDEFINED다. n_origins·평가 stride·horizon overlap·양쪽 source binding을 함께 남긴다. 이 origin별 macro를 단순 평균해 전체 macro라고 주장하지 않는다. 전체 지표는 RUNS/채널·horizon 통계의 전역 채널 count 집계를 따른다.

과거 Jena GPU batch1의 B2 지연이 Raw보다 나빴던 조건과 Ridge의 강한 기준선 지위를 보존한다. CPU/GPU, batch1/8, historical/new hardware scope를 한 순위로 합치지 않는다. 압축 P/E/D 파라미터 감소는 동결 F 상주 메모리 감소를 뜻하지 않는다. L=512는 유지되며 K는 채널 축 압축이다.

Ridge 파생 레이블은 CPU core FP64 / stored FP32다. 과거 저장 dtype 표기가 연산 dtype처럼 읽힐 수 있던 부분을 이 파생 보고서에서 구분했다. 과거 봉인 파일과 예측값은 수정하지 않는다.

G-only의 기존 Jena/PeMS32 CPU batch1/8 비용 4행은 exact run/checkpoint/scaler 및 원 receipt/trials SHA를 대사한 HISTORICAL_CPU2_CONTEXT다. 실제 host identity는 UNKNOWN이며 새 하드웨어의 Raw speedup 계산에 사용하지 않는다. 원 p90을 p95로 바꾸지 않았다. Jena의 이전 prediction manifest는 현재 clean tree에 없어 기존 진단 score/checkpoint/scaler를 통한 transitive binding으로 제한한다. PeMS32는 원 공개 prediction manifest를 직접 대사한다. 이 4행은 신규 실행·신규 측정 수에 포함하지 않는다.

현재 seed/data/method 커버리지:

```text
cohort/seed/method                 selected rows
jena s92601 B2_PCA                 1            
jena s92601 D_GAIN                 1            
jena s92601 D_HORIZON              1            
jena s92601 D_LINEAR               1            
jena s92601 D_SHARED               1            
jena s92601 P_LOCAL_CONTROL_PCA    1            
jena s92601 P_SPLIT_PCA            1            
jena s92602 B2_PCA                 1            
jena s92602 D_GAIN                 1            
jena s92602 D_HORIZON              1            
jena s92602 D_LINEAR               1            
jena s92602 D_SHARED               1            
jena s92602 P_LOCAL_CONTROL_PCA    1            
jena s92602 P_SPLIT_PCA            1            
pems32 s92601 B2_DCA               1            
pems32 s92601 B2_PCA               1            
pems32 s92601 D_GAIN               1            
pems32 s92601 D_HORIZON            1            
pems32 s92601 D_LINEAR             1            
pems32 s92601 D_SHARED             1            
pems32 s92601 P_LOCAL_CONTROL_PCA  1            
pems32 s92601 P_SPLIT_PCA          1            
pems32 s92602 B2_DCA               1            
pems32 s92602 B2_PCA               1            
pems32 s92602 P_LOCAL_CONTROL_PCA  1            
pems32 s92602 P_SPLIT_PCA          1            
```


미완료와 UNKNOWN은 RESUME_QUEUE.json에 남긴다. F6 원 곡선은 비공개 배열을 요구하므로 공개 통계 재생성 범위에서 제외했다.
