# LEVEL vs matched learned E/D — resource stop report

BLOCKED_STOP_RESOURCE. `full_campaign_complete=false`. [확인] 새 neural fit 0, 새 TEST model prediction 0, optimizer update 0이다. fixed E/D + external residual G가 같은 K의 learned E/D보다 필요한지에 대한 질문은 **아직 답하지 못했다**. 사전 점검 통과를 정확도·훈련 비용·추론 비용 우위로 바꾸지 않는다. 비교 전 계약과 실제 parameter enumeration은 [FAIRNESS_TABLE.md](FAIRNESS_TABLE.md)에 있다.

[확인] 전체 계약의 계획 추정치 11421.533초(3시간 10분 21.5초)가 승인된 cumulative GPU-job wall-time cap 10,800초를 621.533초 넘었다. 계약 15절에 따라 학습 전에 STOP_RESOURCE로 종료했다. 실제 preflight GPU job wall time은 21.847596초이며 3시간을 실제로 소비한 것이 아니다. failed job, traceback, 부분 readiness 자료는 `ledger.json`, `preflight_attempt01.json`, `preflight.json`, `resource_stop.json`에 보존됐다. resource gate 실패로 `all_checks_passed=false`인 원본을 유지했다.

이 추정은 **12 fits 모두 120 epochs, fit마다 step0 포함 전체 VAL 121회, early stopping 미가정**을 포함한다. dataset당 zero-update forward+backward **3회 중 최댓값**에 1.25를 곱하고, full-VAL 측정, optimizer CPU allowance, checkpoint allowance, 후속 모델 load 40회 allowance 및 TEST/cost/diagnostics 범위를 넣었다. Peacock 원시 F+B seconds는 0.052150800 / 0.053566600 / 0.036708700이다. 계획용 conservative upper projection이지 실제 convergence time, 통계적 upper bound, 또는 3시간 실행 불가능의 증명이 아니다. 선택 checkpoint의 활성화/시간 변화도 측정하지 않았다. 측정 뒤 estimate를 재조정하거나 scientific scope를 줄이지 않았다.

```text
dataset              four_full_fits_s    post_training_s
robin                        2950.644          565.651
peacock_education            4328.593          183.016
jena                         3219.058          152.735
preflight_at_projection_s: 21.836285
projected_total_s: 11421.533380
adopted_cap_s: 10800
```

[확인] 답한 부분은 정확한 matched 초기 구조가 실행 가능하다는 제한된 질문이다. 각 parent TRAIN PCA bytes/hash, canonical VAL 전체 출력, q=0.5 원래 채널 순서, backbone frozen/eval 및 encoder까지 autograd를 확인했다. microbatch 4/2/1에서 두 weight의 gradient가 finite/nonzero였고 optimizer.step을 하지 않아 E/D와 backbone 상태 hash가 모두 같았다. dataset마다 2 seeds × 120 epochs의 240개 schedule을 literal parent sampler와 대조했다. 이는 학습 완료나 selected model update 증거가 아니다.

```text
dataset              VAL_origins   max_abs_error   E/D_parameters   gradients_mb4/2/1
robin                         30   7.152557e-07              170   finite/nonzero
peacock_education             30   2.384186e-07              104   finite/nonzero
jena                         371   3.337860e-06              252   finite/nonzero
tolerance: elementwise abs_error <= 1e-5 + 1e-4 * abs(reference)
same_path_replay_max_absolute: 0 for all three datasets
```

미실행 항목은 selected LR/checkpoint, selected ED TEST-A/B/combined MSE/MAE, signed bias 비교, paired relative MSE/MAE CI, same-session training-path probe, B1 latency/B4 throughput, allocated/reserved/resident memory와 raw cost blocks, selected checkpoint diagnostics다. 모두 **unavailable, gate prevented**다. 없음은 0 error, equal performance, uncertain CI 또는 equivalence를 뜻하지 않는다. Case A/B/C/D 어느 것도 판정하지 않았다. `DATA_DEPENDENT_NO_GENERAL_WINNER` 역시 실제 dataset별 결과나 CI가 있어야 쓰므로 이번 상태의 대용 문구로 사용하지 않는다.

아래 LEVEL 값은 기존 저장된 TEST 결과의 문맥만 제공한다. 새 ED 비교나 재평가가 아니다. 기존 scorer가 각 seed에서 A/B channel SSE와 observed count를 먼저 합쳐 combined macro를 계산한 뒤, 두 selected seed **loss**를 평균한 저장 row를 그대로 복사했다. prediction ensemble이나 dataset 총점은 만들지 않았다.

```text
dataset              period      LEVEL_MSE       LEVEL_MAE       LEARNED_ED
-------------------  ----------  --------------  --------------  --------------------------
robin                test_a      0.273677231     0.331865000     unavailable, gate prevented
robin                test_b      0.276712128     0.317318719     unavailable, gate prevented
robin                combined    0.275195854     0.324593200     unavailable, gate prevented
peacock_education    test_a      0.234299749     0.339193290     unavailable, gate prevented
peacock_education    test_b      0.172223260     0.290809647     unavailable, gate prevented
peacock_education    combined    0.203266171     0.315004275     unavailable, gate prevented
jena                 test_a      0.422474863     0.381040557     unavailable, gate prevented
jena                 test_b      0.226243453     0.259859279     unavailable, gate prevented
jena                 combined    0.324359158     0.320449918     unavailable, gate prevented
```

기존 source: `research/level_bolt_backbone_scaling_v1_20261006/accuracy_comparison.csv`; SHA256 `8f58c8fb8129120dd3a10ba9231967063fcfd7e453e204a21b706128e799ab1b`. 정확한 0-based CSV record/physical line와 seed IDs는 `report_summary.json`의 `reused_LEVEL_rows`에 있다. JSON parent 원본 pointer는 `parent_reuse_manifest.json`의 각 `levels[].source_evaluation`, `source_pointer`로 이어진다. 기존 LEVEL의 숫자를 새 learned model 성능으로 해석할 수 없다.

가장 강한 반론 네 가지는 다음과 같다.

1. **핵심 대조가 아직 없다.** 같은 K의 E/D를 forecast loss로 학습하면 G를 대체할 수 있다는 반론은 그대로 유효하다. 초기 COMPRESS parity와 finite gradient는 이에 대한 반증이 아니다.
2. **선행 연구와 완전히 같지 않다.** [ICML 2025 AdaPTS](https://proceedings.mlr.press/v267/benechehab25a.html)와 [공식 pinned supervised path](https://github.com/abenechehab/AdaPTS/blob/8bf57c7ee3b97bfd3f1852ad8dc8d0695a806278/src/adapts/adapts.py#L373-L559)를 source audit했다. 공식 LinearAutoEncoder의 bias 기본값은 True, 본 arm은 False다. 공식 default reconstruction coefficient 0.0와 forecast-MSE 경로를 참고했지만 full AdaPTS probabilistic framework를 재현하지 않았다. 공식 `FM.eval()`와 optimizer exclusion만으로 모든 backbone parameter의 requires_grad=False를 추정하지 않았다.
3. **parameter 수와 gradient path가 다르다.** E/D 170/104/252와 historical G 17,920은 parameter matched가 아니다. 같은 압축 손실 처리 전략을 묻는 control이다. backbone-through backprop 비용은 측정할 핵심 tradeoff지만 새 LEVEL/ED 동시 비용 probe가 없으므로 preflight timing/peak memory로 우위를 주장하지 않는다.
4. **노출된 세 dataset과 planning projection의 범위가 좁다.** TEST가 이미 노출된 후속 개발 자료다. 세 자료의 결과가 생겨도 범용성/독립 확인을 뜻하지 않는다. 현재 시간 추정은 짧은 zero-update 측정과 보수적 allowance에 의존하고 실제 early stopping은 더 짧을 수 있다.

판단을 바꿀 증거는 동일 recipe의 최대 12 TRAIN/VAL fits, TEST 이전 joint selection seal, 선택 LR의 6 model TEST predictions, A/B 경계를 지키는 paired bootstrap와 동일 session의 training/inference cost raw blocks다. ED lower MSE CI excludes 0 및 유사/낮은 training cost이면 Case A, ED 더 정확하지만 상당히 높은 cost이면 Case B, LEVEL lower MSE CI excludes 0 및 낮은 training cost이면 Case C, 실제 결과 혼재/CI0이면 Case D를 검토한다. 기존 1.25/2.0 operational cost bins는 사전 해석 기준일 뿐 equivalence margin이 아니며 continuous ratios와 raw ranges를 함께 보아야 한다.

```text
claim_changes:
  keep: 기존 LEVEL 관측 결과의 원래 범위; literal matched step0 구조와 zero-update gradient feasibility
  shrink: matched learned E/D 검증 완료 -> 학습 전 readiness만 확인, resource gate에서 중단
  hold: external residual G 필요성; trained accuracy winner; backprop/inference cost advantage; novelty/generalization
```

최소 다음 계획은 scientific recipe와 3×2LR×2seed=12 fits / selected TEST 6 models를 유지하면서 **resource budget만 명시적으로 재검토**하는 것이다. 예를 들어 14,400초(4시간)는 검토 가능한 상한 예시이며 아직 승인되지 않았다. 원래 STOP_RESOURCE attempt를 보존하고 변경 예산에 대한 continuation gate와 새 source binding을 검토한 뒤에만 재개할 수 있다. 현재 `train_v18.py`는 STOP_RESOURCE preflight를 거부한다. 단순 cap 편집, 자동 fit, bias/K/모델/보조 loss sweep을 실행하지 않는다.

남은 gap은 (1) 승인된 실행 예산, (2) TRAIN/VAL 선택·TEST paired accuracy, (3) matched 비용·선택 checkpoint diagnostics다. 이 회차에서 실행 가능한 부분만 audit하고 핵심 비교는 BLOCKED로 남긴다.

[확인] `verify_v18.py`의 PASS는 저장된 stop 기록과 source/hash/수학의 **자체 검산**에만 해당한다. `full_campaign_complete=false`이며 독립 재현, full AdaPTS reproduction 또는 novelty 인증이 아니다. verifier는 모델 forward/fit/TEST/bootstrap을 실행하지 않고 이미 저장된 step0 VAL score만 재검산한다. 미실행 mandatory checks는 `gate_prevented`로 기록한다. 결과는 [예산 그림](figures/resource_projection.png), [초기화 점검 그림](figures/preflight_readiness.png), source CSV 및 `figures/manifest.json`으로 추적할 수 있다.

```text
BLOCKED
remote_sha: final_checks.json live_remote_main_SHA receipt; publication SHA is recorded after push
new_fits: 0
new_test_predictions: 0
selected_lr: robin/peacock/jena = unavailable, gate prevented
combined_mse: LEVEL = historical rows above; LEARNED_ED = unavailable, gate prevented
paired_relative_mse_ci: unavailable, gate prevented
matched_training_cost: unavailable, gate prevented
inference_cost: unavailable, gate prevented
verdict: unavailable; no scientific Case A/B/C/D
```
