# 선택된 축소 디코더와 새 matched-native의 전체 배포 비용 비교

## 목적과 현재 상태

[확인] 같은 과거 전용 D[n]에 맞춘 새 native 8조건 학습·추론 producer는 실제 `COMPLETE`다. `runs/matched_shrunk_native_v1/RESULT.json`의 SHA256은 `e13587c86a9f28b3dccc5bf445cb85cd855a8a20a0ce1d9caad7dd57b93ba77a`다. 이 문서를 처음 작성하는 시점에는 그 결과의 독립 감사와 이번 새 7경로 비용 실행이 아직 완료되지 않았다. 감사 `PASS`의 정확한 SHA·coverage가 source gate에 등록되기 전 실행을 허용하지 않는다.

[판단] 작은 K와 고정 Bolt로 정확도·비용을 유지할 가능성을 판단하려면 과거를 입력하고 원 C 예측을 반환하는 전체 경로를 같은 조건에서 측정해야 한다. cached Z/D의 head forward 시간이나 이전 Bolt 비용을 새 matched-native 전체 비용으로 부르지 않는다. 성능이 나쁜 방법도 모두 실제 측정하며, 지연 시간을 예측 성능으로부터 추정하지 않는다. 비용을 본 뒤 rho·head CP·seed·방법을 다시 선택하지 않는다.

선행 [matched-native 계획](PLAN_MATCHED_SHRUNK_NATIVE_KO_v1.md), [새 실제 producer](runs/matched_shrunk_native_v1/RESULT.json), [선행 shrinkage producer](runs/local_decoder_shrinkage_v1/RESULT.json) 및 독립 감사가 선택 조건의 기준이다. 이미 실행한 5경로·6회 비용 결과는 보존하고 이번 결과로 덮거나 같은 타이밍 cohort로 합치지 않는다.

## 고정 비교 조건

```text
원 데이터                 원 C   K   E                  이미 선택된 rho
Robin                      17    4   PCA_PINV           0.1
Robin                      17    4   FIT_TARGET_PINV    0.1
Beijing Aotizhongxin        11    2   PCA_PINV           0.1
Beijing Aotizhongxin        11    2   FIT_TARGET_PINV    1.0
```

rho는 이전 Bolt TUNE 선택 잠금 그대로다. 네 native 조건의 seed 92601/92602는 각각 새 학습에서 TUNE로 선택한 head CP를 사용한다. 원 static prior D는 고정 E의 기존 decoder를 원 FP32 규칙으로 cast한 행렬이다. 새로운 E·head·backbone 학습, rho sweep, 비용을 사용한 선택을 하지 않는다. FIT에서 정한 평균·표준편차, 과거 512, 미래 48, 원 true-observed 과거 mask와 노출 DEV origin을 유지한다.

각 데이터에서 아래 정확히 7경로를 B1·B8로 측정한다.

```text
RAW_BOLT
PCA_SHRUNK_BOLT
FIT_TARGET_SHRUNK_BOLT
PCA_SHRUNK_NATIVE_s92601
PCA_SHRUNK_NATIVE_s92602
FIT_TARGET_SHRUNK_NATIVE_s92601
FIT_TARGET_SHRUNK_NATIVE_s92602
```

2개 데이터 × 2개 batch × 7개 경로 = 28개 cell이다. 각 cell warmup 2회, 측정 7회로 실제 timer sample 목표는 196개다. seed `20261015`로 초기 경로 순서를 섞고 7개 circular rotation을 모두 사용한다. 각 경로는 각 위치에 정확히 한 번 나온다. 기존의 5개 rotation+6번째 extra permutation이나 6회 반복을 사용했다고 기록하지 않는다. 모든 경로가 각 데이터의 같은 첫 B1/B8 origin을 공유하며 이를 7회 반복한다. 반복 sample을 서로 다른 데이터나 독립적인 일반화 표본으로 계산하지 않는다.

## 타이머에 들어가는 실제 작업

타이머 경계는 원 전체 비용 helper의 CPU causal fill→FIT 정규화→예측 callback→원단위 반환→최종 동기화 경계를 유지한다. Raw는 고정 F를 직접 호출한다. 축소 경로는 매 호출에서 고정 E로 과거 잠재값을 만들고, 그 호출의 관측 과거 C와 mask로 D[n]을 새로 적합한다. prior D와 선택 rho는 같으며 lambda는 해당 backend 잠재 과거의 `rho × trace(G) / K`다. Bolt의 NumPy E와 native의 Torch CPU E에 같은 rho·규칙을 적용하지만 실제 D/lambda의 byte 동일성을 가정하지 않는다.

Bolt 경로는 NumPy E→과거 shrinkage 적합→실제 고정 F→D innovation→원 C 출력을 수행한다. native 경로는 Torch CPU2/batch32의 고정 E→과거 shrinkage 적합→실제 선택 matched head→Torch FP32 delta/einsum→원 C 출력을 수행한다. native도 현재 C anchor와 origin별 D32[n]를 사용한다. 두 seed가 동일한 D 조건을 사용한다는 사실을 비용 경로의 D cache 공유로 바꾸지 않는다. 각 seed callback이 매번 자기 호출의 D를 다시 적합한다.

과거 적합 함수가 실제 구현한 LS/ridge solve, 관측 개수·trace·condition·재구성·stationarity 진단과 입력 배열 binding hash는 모두 callback 안에서 계산하므로 타이머에 포함한다. 별도 `LocalPastDecoder.metadata()`의 state-array hash, source/file hash, NPZ 저장, 선택 state·모델 load, 참조 배열 load, guard와 memory snapshot은 타이머 밖이다. F 경로의 H2D/F/D2H와 native CPU head의 실제 연산을 각각 포함하고, 최종 physical CPU 배열을 materialize한 뒤 시간을 끝낸다. 별도 inverse proxy 시간으로 대체하지 않는다.

모든 고정 E 객체와 선택 native 8개 head, 고정 F가 같은 process에 머무른다. RSS와 CUDA peak counter는 이 공유 process/기존 counter의 관측값이다. 경로별 독점 메모리나 native 모델만의 VRAM peak라고 주장하지 않으며 counter를 경로마다 reset하지 않는다. 단독 머신·조용한 벤치마크 환경이 검증됐다고 말하지 않는다.

## 실제 실행 전 검증

1. 새 matched producer `COMPLETE`, 독립 감사 `PASS`, 정확한 producer/auditor/source SHA, 8개 새 CP lock와 전체 registry를 확인한다. 8 fit·4,800 updates·40 CP·24개 FIT/TUNE/DEV 저장 예측·12개 고정 input context·8개 fixed-cached future perturbation 검사의 coverage를 확인한다. 선행 shrinkage/FIT-target chain도 exact bind한다.
2. 보호 TEST guard와 기존 uncapped continuation meter를 유지한다. 기존 source·결과·checkpoint·모델 파일·선택 lock의 SHA를 실행 전후 확인한다. 새 출력 경로가 이미 있으면 덮지 않는다.
3. physical 첫 8개 origin을 실제 causal fill/FIT 정규화하고, 각 E·backend에서 잠재 과거와 새 D를 계산한다. 원 past/current C, native cached Z, D32·bias, lambda·trace·condition/stationarity는 원 numerical tolerance로 확인한다. 관측 mask·count·zero-trace·no-observation fallback·positive lambda 결정은 exact 비교한다. shrinkage에 없는 SVD cutoff나 별도 rank 선택을 만들어 검증했다고 쓰지 않는다.
4. native backend에서는 두 selected head를 실제 forward해 같은 첫 8개 저장 예측과 normalized parity를 확인한다. 각 B1/B8 비용 helper도 scientific normalized/physical 참조에 대한 parity를 측정 전과 각 sample에서 다시 확인한다. normalized `atol=1e-5, rtol=2e-6`, physical `atol=2e-3, rtol=2e-5`를 완화하지 않는다. 출력 tolerance만 통과한 것으로 다른 D fitting 결정을 인정하지 않는다.
5. 모든 7경로가 갖춰졌을 때만 해당 데이터의 14개 cell을 측정한다. 경로 실패 때문에 더 작은 cohort로 바꾸지 않는다. 한 데이터가 실패해도 다른 데이터의 독립적인 cohort는 계속한다. 실패 전에 저장한 sample·부분 receipt·source snapshot을 보존하며, 실제 sample 수와 완료 cell의 sample 수를 분리한다.

## 결과와 해석

원 timer 시작/끝·position·origin·7개 sample·normalized/physical 예측 SHA·parity·RSS/VRAM counter를 저장한다. 각 cell의 median latency·window당 latency·같은 block의 Raw와 나눈 7개 paired ratio·그 median을 보고한다. paired ratio의 median과 두 latency median의 비율을 같은 값으로 섞지 않는다.

[판단] 정확도가 나쁘고 비용만 낮은 결과는 정확도·비용 유지 목적을 해결한 것으로 인정하지 않는다. 비용과 정확도는 같은 조건의 원수치로 제시하되, 7회 타이머 반복의 차이를 독립 데이터에 대한 유의한 속도 개선으로 과장하지 않는다. 특히 공용 F와 모든 head가 resident인 조건의 결과라는 한계를 유지한다. 이번 결과는 노출 DEV에서의 탐색적 실제 실행이며 새 독립 TEST 일반화 검증이 아니다.

실행 CLI는 새 controller `run_matched_shrunk_native_cost_v1.py --tag <새이름> --matched-diagnostic <accepted RESULT.json> --matched-audit <실제 PASS 감사.json>`이다. parent가 전체 source와 합성 fixture를 검토하고 실제 PASS gate가 연결된 뒤 한 번 실제 실행한다. 준비 단계에서는 실제 모델·자료를 열거나 타이밍 실험을 수행하지 않는다. 보호 TEST, 설치, 기존 중단 실험 재개, 기존 작업 종료, 큰 모델 재학습, Git commit/push는 이 단계에 포함하지 않는다.
