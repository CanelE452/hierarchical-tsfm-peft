# Nested PCA 3조건 완료와 학습 진행 snapshot

**[확인] 고정 nested PCA의 K3/6/11 전체 V43, 독립 FP64 재채점3개, 새 CPU6 replay3개가 완료됐다. 별도 신경망15 fit의 완성 BASE는 이 snapshot에서0개이며, nested3을 학습 완료로 세지 않는다.**

작성 기준: 2026-10-16 05:55:46 UTC. [PLAN](PLAN.json)의 창은 신경망15 fit+nested3의18단위, 예정 V22개다. 완료18/22라는 뜻은 아니다.

## 점수와 검산

Beijing/primary, L512/H48/C11, seed92601, CPU2 threads, 같은 data/scaler/cohort·target/mask/origins/channel 지문을 확인했다. 단위는 A 정규화 원 C채널 오차다.

```text
K    MSE                  MAE                  전체 V F streams
3    0.7979250373047883   0.6139606908709884   129
6    0.7794565960483851   0.59563573063231     258
11   0.7517899596417101   0.5765475116735715   473
```

- K3: [점수](ordered_predictions/postT_cpu_beijing_E4_NESTED_PCA_k3_s92601_v1/SCORES.csv), [V](diagnostics/rolling_root_v1_nested_3_V_464962a4e7/RESULT.json), [FP64](diagnostics/rolling_root_v1_nested_3_rescore_464962a4e7/RESULT.json), [CPU6](diagnostics/rolling_root_v1_nested_3_replay_464962a4e7/RESULT.json).
- K6: [점수](ordered_predictions/postT_cpu_beijing_E4_NESTED_PCA_k6_s92601_v1/SCORES.csv), [V](diagnostics/rolling_root_v1_nested_6_V_8246ac2ead/RESULT.json), [FP64](diagnostics/rolling_root_v1_nested_6_rescore_8246ac2ead/RESULT.json), [CPU6](diagnostics/rolling_root_v1_nested_6_replay_8246ac2ead/RESULT.json).
- K11: [점수](ordered_predictions/postT_cpu_beijing_E4_NESTED_PCA_k11_s92601_v1/SCORES.csv), [V](diagnostics/rolling_root_v1_nested_11_V_a9916278ae/RESULT.json), [FP64](diagnostics/rolling_root_v1_nested_11_rescore_a9916278ae/RESULT.json), [CPU6](diagnostics/rolling_root_v1_nested_11_replay_a9916278ae/RESULT.json).

원 Raw의 같은 평가 문제 점수는 MSE 0.7968786536971474/MAE 0.5618622043428347다([점수](../predictions/primary_beijing_E1_RAW_BOLT_SMALL_k0_s92601/V_select/SCORES.csv)). K6/11의 MSE는 더 낮지만 K3은 더 높고, MAE는 모두 Raw보다 높다. 과거 Raw 전체 V의 장치까지 새 CPU와 같다고 주장하지 않는다.

전체 V는43 origins의 FP64 검산(atol1e-9/rtol1e-7)이다. Replay는 각6 origins를 batch4로 새 CPU 추론했고, 실제 F calls는 **각2회**, logical/physical streams는 K별18/36/66이다(atol1e-5/rtol1e-4). 전체 V의 실제 호출은 각11회다. 저장 `cost_record.F_calls=1`은 bundle bookkeeping이며 실제 호출수나 비용 계측이 아니다.

## 고정 변환과 남은 범위

[원 구현](../ordered_pilot.py)의 `_nested_artifact`는 A60 끝12622까지 정규화·11×11 PCA basis/center를 적합하고 같은 basis의 앞 K축을 쓴다([실제 basis binding](diagnostics/rolling_root_v1_nested_3_rescore_464962a4e7/NESTED_FIXED_BASIS_BINDING.json)). A50 검색·학습된 shared/max 부모가 필요 없는 별도 고정 대조이며, optimizer updates/exposures는0이다. K11=C도 PCA 좌표로 회전한 후 비선형 F를 거치므로 Raw 원채널 항등 경로나 full-C 학습과 같지 않다.

신경망 shared는 [search1epoch](runs/postT_cpu_beijing_E4_ORDERED_MULTI_K_k11_s92601_v1/search/event_000002.json), max-only는 [search0.25epoch](runs/postT_cpu_beijing_E4_MAXK_PCA_TRAINED_k11_s92601_v1/search/event_000001.json) milestone이 저장됐지만 최종 선택 A60 refit·V 완료를 뜻하지 않는다. 이 문서의 nested 정밀 CPU 비용은 아직0조건이며, 경과시간을 배포 비용으로 쓰지 않는다.

원 source39·등록·선택·T 보존은 실행 receipt와 [PLAN](PLAN.json)의 핀 범위다. 이는 새 수렴·방법 우위·인과 그래프·독립 일반화 PASS가 아니다. 원 [core56 대응](../diagnostics/postT_core56_scope_20261016_v1/README_KO.md)은 현재19/잔여37이고 nested3으로 늘지 않는다. Post-T V-only·단일 seed·전체 연구 PARTIAL을 유지한다. 보고서 작성 중 모델·배열·통계 gzip·새 학습/추론/비용/GPU/Git을 실행하지 않았다.
