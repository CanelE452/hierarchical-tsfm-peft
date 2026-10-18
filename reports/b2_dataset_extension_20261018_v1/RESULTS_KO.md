# B2 데이터 확장 실험 결과

[확인] 이 고정 조건에서 두 후보 모두 B2의 MSE가 Raw보다 나빴고, Direct MLP는 두 데이터 모두에서 B2보다 MSE·전체 지연·allocated 메모리가 우수했다.

[확인] Household와 SML2010의 고정 비교를 완료했다. 4개 B2 search 전체를 봉인한 뒤 fresh A60 refit/V를 실행했고, 동일 seed의 Direct MLP 4개와 Raw 2개를 포함해 총 10개 경로를 새 프로세스에서 측정했다. 후보를 결과에 따라 교체하지 않았다.

[전체 seed별 수치](COMPARISON.csv) · [비교 그림](COMPARISON.png) · [영문 슬라이드 문안](SLIDE_CONTENT_EN.md)

MSE·MAE는 각 데이터의 고정 A60 정규화 단위에서 관측 equal-channel macro로 계산했다. L512/H48은 같은 스텝 수이고 실제 시간 범위는 다르다(H48: Jena 8시간, SML 12시간, Household 48시간). 주파수를 강제로 리샘플하지 않았으며 동일 조건은 데이터별 방법 간 비교를 뜻한다. 서로 다른 데이터의 MSE 크기를 직접 비교하지 않는다. 아래 평균·범위는 두 seed의 기술 통계이며 신뢰구간이 아니다.

## Household

```text
method       seeds   MSE mean [min, max]          MAE mean    B8 median ms    allocated MiB    reserved MiB
Raw          1       0.755472 [0.755472, 0.755472]  0.509199    45.658           209.077          270.000
Ours B2      2       0.774520 [0.766500, 0.782540]  0.601801    12.324           208.498          262.000
Direct MLP   2       0.712240 [0.710379, 0.714101]  0.569324    0.397           9.530          22.000
```

B2의 Raw 대비 MSE 변화는 +2.52%, MAE 변화는 +18.19%, 전체 B8 지연 기준 속도비는 3.70배다. 낮은 오차와 빠른 실행을 별개 축으로 해석한다.

Direct MLP가 B2를 MSE·allocated 피크·지연 세 축에서 지배하는지: 평균=True, seed92601=True, seed92602=True. 세 값 모두 작거나 같고 하나 이상 엄격히 작을 때만 지배로 판정했다. MAE와 reserved는 별도 보고한다.

B2 두 seed의 B8 median 지연 범위는 12.281–12.367 ms, allocated 피크 범위는 208.498–208.498 MiB, reserved 피크 범위는 262.000–262.000 MiB다.

Direct MLP 두 seed의 B8 median 지연 범위는 0.391–0.402 ms, allocated 피크 범위는 9.530–9.530 MiB, reserved 피크 범위는 22.000–22.000 MiB다.

## SML2010

```text
method       seeds   MSE mean [min, max]          MAE mean    B8 median ms    allocated MiB    reserved MiB
Raw          1       0.321352 [0.321352, 0.321352]  0.315510    56.274           243.127          262.000
Ours B2      2       0.671312 [0.575660, 0.766964]  0.543829    18.534           242.915          262.000
Direct MLP   2       0.440934 [0.432779, 0.449088]  0.426263    5.073           10.200          24.000
```

B2의 Raw 대비 MSE 변화는 +108.90%, MAE 변화는 +72.37%, 전체 B8 지연 기준 속도비는 3.04배다. 낮은 오차와 빠른 실행을 별개 축으로 해석한다.

Direct MLP가 B2를 MSE·allocated 피크·지연 세 축에서 지배하는지: 평균=True, seed92601=True, seed92602=True. 세 값 모두 작거나 같고 하나 이상 엄격히 작을 때만 지배로 판정했다. MAE와 reserved는 별도 보고한다.

B2 두 seed의 B8 median 지연 범위는 18.280–18.788 ms, allocated 피크 범위는 242.915–242.915 MiB, reserved 피크 범위는 262.000–262.000 MiB다.

Direct MLP 두 seed의 B8 median 지연 범위는 4.934–5.211 ms, allocated 피크 범위는 10.200–10.200 MiB, reserved 피크 범위는 24.000–24.000 MiB다.

## 관측 범위와 학습 상태

Household는 V 72개 origin을 모두 평가했다. SML2010은 V 9개 origin 중 한 창의 목표가 전부 결측이어서 실제 오차 기여 origin은 8개다. 그 창을 결과에 따라 제거하거나 보간하지 않았다. SML의 3개 motor 채널은 A_fit에서 상수이며 21개 채널의 원래 macro 집계에 그대로 포함했다. SML의 짧은 기간과 결측 블록은 결과의 적용 범위를 제한한다.

```text
run_id  selected_epoch  cap  search_train_origins  training_status
B2_uci235_household_hourly_K2_OFF_s92601  3.0  3  16734  CAP3_STILL_IMPROVING
B2_uci235_household_hourly_K2_OFF_s92602  3.0  3  16734  CAP3_STILL_IMPROVING
B2_uci274_sml2010_K6_OFF_s92601  1.0  3  1833  CAP3_CONVERGENCE_UNRESOLVED
B2_uci274_sml2010_K6_OFF_s92602  3.0  3  1833  CAP3_STILL_IMPROVING
DirectMLP_uci235_household_hourly_s92601  3.0  3  16734  CAP3_STILL_IMPROVING
DirectMLP_uci235_household_hourly_s92602  3.0  3  16734  CAP3_STILL_IMPROVING
DirectMLP_uci274_sml2010_s92601  3.0  3  1833  CAP3_STILL_IMPROVING
DirectMLP_uci274_sml2010_s92602  3.0  3  1833  CAP3_STILL_IMPROVING
```

TRAIN origin 수는 실제 stride1 학습 receipt의 수이며 데이터 진단용 stride48 목록 길이가 아니다. cap3 끝이 계속 최선인 arm은 학습 상한에서 개선 중이며 수렴한 것으로 해석하지 않는다. epoch0 선택도 그대로 보고한다.

## 공정한 해석과 비용의 범위

두 후보 모두 과거 다른 방법의 개발에 사용된 데이터이고 V는 반복 개발 구간이다. 새 독립 TEST나 사전학습 비중복을 입증하지 않는다. LoRA와 CPU Ridge는 이번 확장에서 실행하지 않아 그 대안보다 우수하다는 주장을 하지 않는다.

그림의 Jena* 4행은 이전 memory_budget_20261018_v1의 B8·confirm·row48·K6 Raw/B2 두 seed/merged LoRA 문맥이다. 새 후보의 비용과 별도의 과거 실행이며, Jena의 기존 데이터·정규화 ancestry 및 과거 TEST 노출 이력도 유지된다.

실측은 경로별 10 warmup+20 timed, 첫 8개 고정 V origin을 반복했다. 전체 V·불균등 마지막 batch의 row128/공통 cap parity와 F calls/streams를 따로 검증했다. 시간은 physical+mask부터 causal fill·A60 scaler·실제 모델·inverse·CPU 반환까지이며 window slicing·model load·검산·hashing은 제외했다. GPU 수치는 CUDA allocated/reserved 텐서 피크이며 전체 프로세스 메모리 상한이 아니다. 20회 p95는 서비스 tail latency 보장이 아니다.

CSV의 K는 F에 들어가는 stream 폭이다(Raw=C, B2=압축 K, Direct MLP=0). 공통 실행 제한은 별도 row_cap 열이며 Raw 채널을 압축하지 않았다.

새 경로 10개, warmup 100회, timed 200회 및 private prediction 파일 SHA·array 지문·FP64 오차·raw timing quantile·학습/선택/체크포인트 연결을 독립 검산했다. seed별 전체 수치와 각 원 receipt 경로는 CSV에 있다.

초기 비용 10개는 새로 강제한 `CUBLAS_WORKSPACE_CONFIG=:4096:8`을 사용했으므로 [환경 정합 기술 반복 기록](MEASUREMENT_AMENDMENT_001.json)에 `AUXILIARY_WORKSPACECONFIG`로 보존했다. 기존 Jena 코드가 부모 환경을 상속했다는 사실은 확인했지만 당시 실제 환경값은 기록이 없어 확정하지 않는다. 본표는 강제 설정을 제거하고 현재 부모의 unset 값을 기록해 비용만 다시 측정한 10개다. 이번 두 반복의 allocated 피크 차이(auxiliary−본표)는 23.875–23.875 MiB다. 보조 측정을 실패나 별도 정확도 반복으로 처리하지 않았다. 합계는 비용 경로 측정 20개, warmup 200회, timed 400회이며 학습 재실행은 0회다. 학습 모델은 B2 4개와 Direct MLP 4개(각 search+fresh refit)로 그대로다.
