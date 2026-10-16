# B2 우선 검증 후속: Jena 두 seed와 PeMS 전체 CPU 비교

**[확인] 교수님께 설명할 방법론 후보의 구현은 검증됐고, Jena에서는 두 학습 seed 모두에서 B2의 MSE가 B0보다 낮았다. 다만 강한 직접 예측·적응 기준선 전체를 앞선 방법은 확보하지 못했다.** PeMS의 새 전체 CPU148 부모와 고정 no-Bolt head 비교까지 끝냈으며, 이 문서는 남은 MASTER 전체 완료 보고가 아니다.

1. 구조는 맞는가: 허용 과거512→시간·채널 앞단 P→C에서K로 E→동결 Bolt→lag D+현재값 경로, 실제 K streams·F 입력 역전파·동결·미래 경계를 확인했다. [P0 근거](../priority_b2_validation_20261016_v1/diagnostics/p0_root_v1/RESULT.json). B0도 F와 D를 통해512과거를 사용한다. 현재값 우회가 있으므로 모든 정보를 오직K로만 전달하는 통신압축은 아니다.
2. B0에 무엇을 더했는가: 아래 두 seed에서 MSE가 낮았고 선택 refit은 B0/B2 모두3/1epoch였다. B2의 추가 파라미터와 학습 구조가 함께 달라 **용량을 맞춘 시간혼합의 인과 효과**로 해석하지 않는다. MAE 이득의 크기도 seed에 따라 다르다.
3. Bolt를 붙일 근거는 있는가: Jena R4는 같은 최종 P/E/D의 전체512K 선형 head R5a와, 동형 구조를 새로 학습한 no-F R5b보다 MSE가 낮다. 그러나 native Ridge·CPU fresh G·제한 학습 LoRA의 V 수치보다 높아 전체 대안 대비 필요성은 확정하지 못했다. LoRA/Raw/PCA의 옛 전체 V 장치는CUDA이며 CPU6 subset과CPU비용을 전체CPU 정확도로 바꾸지 않는다.
4. 전체 비용은 어떤가: PeMS 같은 CPU148에서 부모 MSE 0.104259697, 고정 head 0.12062289로 head가 15.6946% 높다. CPU B1/B8 부모/head 시간비는 각각 368.6588/350.5368다. 학습시간·모델 적재·정밀추론은 분리하며 전체 시스템의 무조건적 비용 우위로 확대하지 않는다.
5. 후순위는 무엇인가: 먼저 이 진단을 근거로 정확도·학습·배포비용 중 목표를 좁혀야 한다. 승인된 기존 후속은 원예산·T잠금 안에서만 별도로 실행한다. nested 비용6개/Beijing 최소3조건은 별도 준비이며 기존 미시작13/core29/M5·추가자료·K·학습량 등 전체 잔여를 대체하지 않는다.

[원정밀도 전체 결과](RESULTS_v2.csv) · [학습 비용·학습량](TRAINING_COST_v2.csv) · [입력 SHA와 출력 결속](REPORT_INPUT_BINDINGS_v2.json) · [재현 코드](build_followup_report.py)

![Jena 두 seed와 PeMS CPU 비교](figures/FOLLOWUP_SUMMARY_v3.svg)

## Jena: 두 학습 seed, 같은 개발 V43

```text
seed       B0 MSE       B2 MSE       B0 MAE       B2 MAE  selected refit B0/B2
92601   0.784103931  0.683442896   0.52096221  0.493884924  3.0/1.0
92602   0.755008778  0.678813925  0.502520209  0.501871227  3.0/1.0
```

L512/H48/C21/K6, A50[0,10482), A_stop[10482,12579), fresh A60[0,12579), V[14675,16772)의 43 origins·43,344 관측 정답과 동일한 scaler/target/mask/channel을 유지했다. 검색은 0/.25/1/3epoch, 선택은 A_stop이며 V를 checkpoint 선택에 쓰지 않았다. seed92602도 B0 search1863/refit2254update, B2 search1863/refit752update다. B0의 마지막 선택 3epoch와 B2의 선택 1epoch는 수렴 검증이 아니다. 두 seed와 반복해서 본 V를 독립시험·유의성 검정으로 부르지 않는다.

R5a는 **R4 최종 P/E/D와 scaler를 고정**하고 전체 Z[512,6]를 공유 Linear(512,48)에 넣어 A60 고정 3epoch 동안 head만 학습했다. R5b는 같은 형태의 P/E/head/D를 fresh로 함께 학습하고 A_stop으로 선택한 별도 가중치다. 부모 표현을 얻는 학습 비용은 R5a head의 추가 비용과 별도다. 이미 미래 MSE 목적을 사용했으므로 현재 실패를 재구성 목표 때문이라고 설명하지 않는다.

## PeMS08: 새 부모 전체 CPU148이 주 비교

새 CPU 부모는 기존 `primary_pems08_E1_B2_k43_s92601__e1`의 동일 checkpoint를 재학습 없이 전체 148 origins에서 추론하고 별도로 FP64 검산했다. H12/C170/K43, A50end8928/A60end10713, V[12499,14284), 301,920 정답과 scaler94e8c71c…69c60을 head와 맞췄다. 실제 원자료의 앞 14284행/feature0만 수치 변환했고 보호 T 수치는 파싱하지 않았다. 옛 GPU 부모 V는 별도 reference 행이다. GPU/CPU 출력 비교 상태는 `PASS_ALL148_GPU_CPU_PARITY`이며, 새 CPU 예측·독립 재점수의 완료와 구분한다.

부모는 검색 0/.25/1epoch·선택 1epoch의 제한 학습이다. 직접 head는 그 최종 P/E/D를 고정하고 전체 Z[512,43]를 Linear(512,12)에 넣어 head 파라미터 6,156개만 추가 3A60epoch/1911update/30,570 노출로 학습했다. 부모 1epoch와 추가 head 3epoch의 학습량은 같지 않고, PeMS에서 별도의 joint no-Bolt 학습은 수행하지 않았다. 이 결과는 고정 Bolt용 표현/고정 D 아래의 선형 head 진단이며 Bolt의 보편적 필요성이나 표현 학습 방법의 일반 한계를 확정하지 않는다.

## 실제 V 수치와 장치

```text
dataset role                                        MSE          MAE accuracy device
jena    R0_Raw                              0.849194139   0.48138834 cuda
jena    R1_Ridge                            0.556646336  0.405521802 cpu
jena    R1_G                                0.632217057   0.44152129 cpu
jena    R2_PCA                               1.07948864  0.642719116 cuda
jena    R3_B0                               0.784103931   0.52096221 cpu
jena    AUX_LAG_CAUSAL                      0.743206418   0.51245722 cpu
jena    AUX_LAG_BIDIR                       0.718890348  0.503302599 cpu
jena    R5a                                 0.737851378  0.503803264 cpu
jena    R5b                                 0.720197904  0.509695606 cpu
jena    R6_LoRA                             0.626266015  0.416788307 cuda
jena    R4_B2                               0.683442896  0.493884924 cpu
jena    B0_SEED92602                        0.755008778  0.502520209 cpu
jena    B2_SEED92602                        0.678813925  0.501871227 cpu
pems08  B0                                  0.105338534  0.212696673 cuda
pems08  B2_PARENT_GPU_REFERENCE             0.104259699  0.205841794 cuda
pems08  RAW                                0.0953549334  0.196333829 cuda
pems08  RIDGE                               0.228995943  0.232532032 cpu
pems08  G_MATCHED_ONE_EPOCH_REFERENCE        0.10605044  0.196226558 cuda
pems08  LORA                               0.0847248737  0.181742605 cuda
pems08  G_CPU_FRESH_THREE_EPOCH_REFERENCE  0.0993884698  0.189054279 cpu
pems08  B2_PARENT_CPU_FULL                  0.104259697  0.205841792 cpu
pems08  FIXED_B2_NO_BOLT_HEAD                0.12062289  0.218175432 cpu
```

CSV는 원정밀도를 유지한다. CUDA와 CPU 전체 정확도 행을 섞어 동일 장치 전체 순위를 만들지 않는다. 기존 G 1epoch와 CPU fresh G 3epoch, LoRA .25epoch도 서로 다른 학습량이다. native Ridge는 32개 시간 블록 평균+현재값을 쓰는 별도 FP64 닫힌해이며 모든 ridge 대안을 대표하지 않는다. Ridge는 closed-form/no optimizer로 다루며 신경망의 수렴 미확정 판정 대상이 아니다.

## CPU 전체 경로 추론과 학습 비용

```text
dataset role                               CPU B1 seconds/batch     CPU B8 seconds/batch
jena    R0_Raw                             NOT_MEASURED             NOT_MEASURED
jena    R1_Ridge                           0.00022649997845292091   0.0005060497205704451
jena    R1_G                               0.0001355998683720827    0.0003303999546915293
jena    R2_PCA                             0.04734520008787513      0.3246007999405265
jena    R3_B0                              0.04765265015885234      0.34607005026191473
jena    AUX_LAG_CAUSAL                     0.0505663501098752       0.3432335499674082
jena    AUX_LAG_BIDIR                      0.048634300008416176     0.3454676503315568
jena    R5a                                0.00040840008296072483   0.0008226498030126095
jena    R5b                                0.0003942498005926609    0.0008136499673128128
jena    R6_LoRA                            0.13647314999252558      1.1252111999783665
jena    R4_B2                              0.04868765012361109      0.35482815001159906
jena    B0_SEED92602                       0.046653299825266004     0.3240838497877121
jena    B2_SEED92602                       0.0474398999940604       0.3300462500192225
pems08  RIDGE                              0.0003635501489043236    0.005099399946630001
pems08  B2_PARENT_CPU_FULL                 0.28220830019563437      2.2950869500637054
pems08  FIXED_B2_NO_BOLT_HEAD              0.0007654998917132616    0.006547349970787764
```

원 physical 입력·관측 mask→과거 채움/정규화→P/E→F 또는 직접 head→D→FP64 역변환/CPU 반환을 실제 측정했다. CPU2, B1/B8, 10warm/20timed/별도20stage 조건이다. cached Z/forecast는 쓰지 않았다. stage는 계측이 추가된 별도 범위며 stage 합이 whole과 같다고 하지 않는다. 부모/표현 학습·head 추가 학습·model load·steady-state 추론을 총비용 하나로 더하지 않는다. [학습CSV](TRAINING_COST_v2.csv)의 빈 시간은 미기록이며 0이 아니다. 원 장부에서 다른 장치·동시 실행으로 측정된 학습시간은 인과적 속도 비교가 아니다.

Jena Raw의 원 row-parity 실패는 미해결이라 timed 비용을 채우지 않았다. PeMS Raw/G/LoRA의 옛 CUDA 비용을 CPU 열에 복사하지 않았다. PeMS native Ridge의 기존 CPU 영수증은 thread 필드가 없고 pinned 실행 source의 2threads를 근거로 한다. CPU active_adapter_MiB의 0은 GPU 전용 계수의 0이며 CPU 어댑터 메모리 0을 뜻하지 않는다.

## 검증 범위와 다음 판단

[새seed2독립검산](INDEPENDENT_SEED2_REVIEW_ROOT_v1.json)은 공개 5,418통계행·86,688label과 200costtrial/8summary를 재집계했고 기존 actual CPU6 및 원수치 FP64 검산을 결속했다. [PeMS head와부모CPU6/비용](INDEPENDENT_PEMS_REVIEW.json), [기존baseline](PEMS_BASELINE_REUSE.json), [새부모전체CPU추론](diagnostics/pems_parent_full_cpu_inference_root_v1/RESULT.json)와 [별도전체FP64검산](diagnostics/pems_parent_full_cpu_rescore_root_v1/RESULT.json)을 구분한다. [새 부모 전체 CPU의 독립 충분통계 검산](INDEPENDENT_PARENT_CPU_REVIEW_ROOT_v1.json)도 별도 근거다. 단순 검사 PASS를 성능 개선이나 새 일반화로 해석하지 않는다.

확정된 것은 구현 경계와 조건부 비용·오차 차이이다. 미확정은 추가 앞단이 작은 K에서 강한 직접 예측보다 유리한 조건, 충분 학습 후의 안정성, 미노출 구간 일반화와 학술적 신규성이다. 현재 근거로는 B2→K→frozenBolt→lagD+anchor를 **검증 중인 후보**로 설명하고, 전체 우월성을 기여로 내세우지 않는다. 원자료·checkpoint·보호 T를 재선택하거나 G/H·새 graph로 주방법을 전환하지 않는다.
