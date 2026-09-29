# V8 topic decision — subspace P/Q correction

## Decision

[확인] v8의 P/Q split은 **조건부 개발 신호만 남긴다**. Jena에서는 보존 공간(P) 보정이 기존 LEVEL 잔차 PEFT보다 분명히 낫고, 같은 35,840개 학습 파라미터를 가진 RAW64보다도 낫다. 그러나 Robin에서는 기존 LEVEL을 대체할 근거가 없고, 두 자료 모두에서 LoRA 또는 DIRECT 격차가 남아 있다. 따라서 v8 결과는 **LEVEL의 기본 대체안으로 채택하지 않는다**. 다만 Jena와 같은 경우에는 “버린 공간(Q)만 고치던 LEVEL의 한계를 보존 공간(P) 보정으로 줄일 수 있다”는 후속 개발 가설로 보존한다.

[확인] 이 판단은 노출된 개발 자료 Robin/Jena의 고정 비교다. 독립 보호 평가나 범용성 확증이 아니다. Robin에서 `pq_split`이 `level_res`보다 불리한 평균을 보였지만, paired block interval이 0을 가로지르므로 “P/Q split이 보편적으로 열등하다”는 주장도 하지 않는다.

## Method fixed before evaluation

승인된 v8 수식은 다음이다. `P`는 TRAIN PCA 보존 공간, `Q=I-P`, `b = D(F0(E(X)))`, `delta X = X - x_last`이다.

```text
pq_split:
  Yhat = b + repeat48(x_last Q) + G_Q(delta X Q) + H_P(delta X P)

raw64 parameter control:
  Yhat_RAW64 = b + repeat48(x_last Q) + G64(delta X)
```

[확인] `G_Q`와 `H_P`는 서로 독립적인 공유 시간 선형 head `512→32→48`이고, 출력층은 0 초기화다. `raw64`는 `512→64→48`로 같은 총 학습 파라미터 35,840개를 갖는다. 두 family 모두 `fixed_ed`이며 F0, PCA basis, K를 고정했다. 근거 위치는 [PLAN.md](PLAN.md)의 “첫 변경”과 “실행 구체화”, [model_v8.py](model_v8.py)의 `FixedAdapter.model_config`, [protocol.json](protocol.json)의 `fits`, [selected_robin.json](selected_robin.json)·[selected_jena.json](selected_jena.json)의 `matching_checks`이다.

[확인] 두 방법 모두 동결 백본을 포함한 전체 파라미터는 Robin47,754,026개/Jena47,754,108개다(`runs/*/result.json.total_parameters`). FP32 파라미터 텐서만의 산술 크기는 각각191,016,104/191,016,432 bytes이며 buffer·직렬화 overhead·activation·allocator·프로세스 메모리를 포함하지 않는다. 작은 학습 파라미터 수가 작은 전체 배포 모델을 뜻하지 않는다. 마지막 수준 제거·복원, PCA, 잠재 E/D 자체는 기존 원리이며 v8의 기여 근거는 한정된 직접 대조 결과다. 선행 중복 경계는 [v7 CLAIMS_PRIOR_ART](../tsfm_peft_practical_controls_v7_20260928/CLAIMS_PRIOR_ART.md)를 유지하며 새 이름이나 이 실험만으로 최초성을 주장하지 않는다.

[확인] 초기·학습 통제는 다음과 같다.

```text
공통 설정: seed 92601/92602, LR 1e-3, max 120 epoch, patience 6, VAL 최소 MSE checkpoint
Robin: C17/K5, v8_robin_{pq_split,raw64}_{92601,92602}
Jena:  C21/K6, v8_jena_{pq_split,raw64}_{92601,92602}
초기 제어: 같은 seed에서 shared initial source true, same initial VAL score true
선택 금지: selected_*.json의 test_used_for_selection=false
```

## Training and execution validity

[확인] 실자료 fit은 8개 모두 `complete`이고 실패한 실자료 fit은 0건이다. 복구용 attempt는 쓰지 않았다. 각 run은 `final_stale=6`, `cap_reached_while_improving=false`, `restart_roundtrip.model_serialization=true`, `replay_error=0`이었다. 결과 파일은 [runs/](runs/) 아래 각 `result.json`에 있다.

```text
run                         selected_epoch  val_mse     train_probe initial→selected  steps
v8_robin_pq_split_92601      5              0.380618    0.239223→0.175325            1408
v8_robin_pq_split_92602      4              0.384782    0.249672→0.171686            1280
v8_robin_raw64_92601         6              0.466725    0.239223→0.219170            1536
v8_robin_raw64_92602         7              0.471757    0.249672→0.233767            1664
v8_jena_pq_split_92601       25             0.518065    0.430776→0.348918            3968
v8_jena_pq_split_92602       9              0.521127    0.464714→0.377185            1920
v8_jena_raw64_92601          6              0.548095    0.430776→0.369831            1536
v8_jena_raw64_92602          16             0.542522    0.464714→0.388555            2816
```

[확인] seal과 TEST 평가 후 추가 fit은 하지 않았다. [ledger.json](ledger.json)의 `v8_seal_*`, `v8_evaluate_*`, `v8_analyze_analysis01` job은 완료 상태다. [STATUS.md](STATUS.md)에 현재 결정과 예산을 기록했다. 실행팀의 최종 저장 산출물 검산 상태는 [final_checks.json](final_checks.json)의 실제 status를 따른다. 이 문서의 해석이나 LLM의 동의는 독립 재현이 아니다.

## Accuracy evidence

주 지표는 TRAIN 표준화 raw-space observed-channel macro MSE이고, MAE는 보조 지표다. 역할별 점수는 seed별 손실 평균이며 ensemble이 아니다. 근거 위치는 [robin_eval01.json](robin_eval01.json)과 [jena_eval01.json](jena_eval01.json)의 `role_scores`, `comparisons`, [analysis01/analysis_summary.json](analysis01/analysis_summary.json)이다.

[확인] 아래 상대 변화는 `(수정 MSE / 비교 MSE - 1) × 100%`이며 퍼센트포인트가 아니다. 95% 구간은 고정된 두 seed 손실 평균에서 기간 내부 연속 블록(Robin7/Jena42원점,2,000회)을 대응 재표집했다. 기간 경계를 넘지 않고 채널·방법을 함께 움직인다. 학습 seed 모집단이나 개발 선택 편향 전체를 포괄하는 구간이 아니며, 0 포함은 동등성 증명이 아니다. 채널별 MSE/MAE/부호오차는 각 평가 `models.*.scores.*`에, seed·기간 전체 값은 [model_scores.csv](analysis01/model_scores.csv)에 보존한다.

### Robin

[확인] Robin에서는 `pq_split`이 parameter-matched `raw64`와 기존 `level_raw`보다는 낫지만, 기존 `level_res`를 대체하지 못한다. `pq_split`은 LoRA와 F0보다도 크게 나쁘다.

```text
Robin combined role scores
role             MSE        MAE        test_a MSE  test_b MSE
pq_split         0.279019   0.331915   0.281026    0.277010
raw64            0.347336   0.386750   0.359342    0.335327
level_res        0.275196   0.324593   0.273677    0.276712
level_raw        0.346630   0.385883   0.358622    0.334635
direct_nlinear   0.327724   0.353410   0.327484    0.327962
f0               0.229095   0.262074   0.231539    0.226649
lora             0.223610   0.256347   0.230535    0.216683
old_res          0.538776   0.483495   0.563077    0.514472
```

[확인] paired block 비교는 다음과 같다.

```text
comparison                         combined delta MSE  relative       block 95% relative interval
pq_split - raw64                   -0.068317           -19.67%        [-29.80%, -9.98%]
pq_split - level_res               +0.003823           +1.39%         [-0.76%, +3.77%]
pq_split - level_raw               -0.067611           -19.51%        [-29.42%, -9.86%]
pq_split - direct_nlinear          -0.048705           -14.86%        [-19.87%, -8.89%]
pq_split - f0                      +0.049924           +21.79%        [+10.01%, +39.78%]
pq_split - lora                    +0.055409           +24.78%        [+11.66%, +45.13%]
raw64 - level_raw                  +0.000706           +0.20%         [-0.38%, +0.76%]
```

[확인] seed별로도 `pq_split`의 Robin LEVEL 대체 근거는 약하다. `pq_split_vs_level_res` combined에서 seed 92601은 0.275013 vs 0.275009로 사실상 동일하고, seed 92602는 0.283025 vs 0.275383으로 `pq_split`이 나쁘다. 이 때문에 Robin에 대해서는 “개선 없음”으로 본다. 다만 interval이 0을 가로지르므로 “항상 더 나쁨”으로 확장하지 않는다.

### Jena

[확인] Jena에서는 `pq_split`이 `raw64`, `level_res`, `level_raw`, F0보다 MSE 기준으로 좋다. 그러나 DIRECT에는 combined에서 약간 밀리고 interval이 0을 가로지르며, LoRA에는 명확히 밀린다.

```text
Jena combined role scores
role             MSE        MAE        test_a MSE  test_b MSE
pq_split         0.279714   0.299361   0.346200    0.213229
raw64            0.291395   0.306281   0.360751    0.222038
level_res        0.324359   0.320450   0.422475    0.226243
level_raw        0.291159   0.306634   0.360648    0.221671
direct_nlinear   0.273984   0.286714   0.332070    0.215899
f0               0.292300   0.264758   0.370756    0.213844
lora             0.240590   0.226584   0.285445    0.195735
old_res          0.323305   0.321819   0.422585    0.224024
```

[확인] paired block 비교는 다음과 같다.

```text
comparison                         combined delta MSE  relative       block 95% relative interval
pq_split - raw64                   -0.011680           -4.01%         [-5.22%, -2.92%]
pq_split - level_res               -0.044645           -13.76%        [-16.31%, -10.22%]
pq_split - level_raw               -0.011445           -3.93%         [-5.06%, -2.75%]
pq_split - direct_nlinear          +0.005730           +2.09%         [-0.58%, +4.82%]
pq_split - f0                      -0.012586           -4.31%         [-7.01%, -0.62%]
pq_split - lora                    +0.039125           +16.26%        [+12.83%, +19.70%]
raw64 - level_raw                  +0.000235           +0.08%         [-0.19%, +0.38%]
```

[확인] Jena의 개선은 두 seed에서 모두 나타난다. `pq_split_vs_raw64` combined seed 92601은 -4.22%, seed 92602는 -3.80%이고, `pq_split_vs_level_res` combined seed 92601은 -13.80%, seed 92602는 -13.73%이다. 다만 `pq_split_vs_direct_nlinear`은 combined +2.09% 손해이고 interval이 0을 가로지르며, test_a에서는 +4.26% 손해로 interval도 양수다. `pq_split_vs_lora`는 combined +16.26% 손해이고 두 기간 모두 손해다.

## P/Q decomposition

[확인] P/Q 분해는 완전관측 target row에만 수행했다. 결측 target을 0으로 채워 투영하지 않았다. 따라서 이 값은 masked macro MSE의 대체 지표가 아니라, 보존 공간(P)과 버린 공간(Q)에서 오차가 어떻게 달라졌는지 보는 진단이다. 근거 위치는 [robin_eval01.json](robin_eval01.json)·[jena_eval01.json](jena_eval01.json)의 `role_space_scores`이다.

[확인] coverage는 Robin combined 3,838/3,840 complete rows(test_a 1,920, test_b 1,918), Jena combined 35,040/35,040 complete rows(test_a 17,520, test_b 17,520)이다. 이는 원점×horizon 행 수로, 겹친 목표시각을 독립 관측 수로 세지 않는다. Robin은 incomplete target row가 있어 P+Q 진단 합과 masked macro MSE가 완전히 일치하지 않는다. Jena는 평가 target이 완전관측이므로 수치 정밀도 범위에서 진단 합이 role MSE에 맞는다. 이 분해는 모든 비교군의 같은 complete-row coverage 안에서만 비교한다.

```text
Robin P/Q complete-row MSE
role             P MSE      Q MSE      complete rows
pq_split         0.124528   0.153146   3838
raw64            0.133290   0.213094   3838
level_res        0.121813   0.152047   3838
level_raw        0.131638   0.214010   3838
direct_nlinear   0.154538   0.171680   3838
f0               0.102494   0.125240   3838
lora             0.100528   0.121749   3838
```

[확인] Robin에서는 `pq_split`이 `raw64`보다 P/Q 모두 낫지만, `level_res`보다 P와 Q 모두 조금 나쁘다. 이는 Robin에서 P-head 추가가 기존 LEVEL의 선택을 바꿀 정도의 개선을 만들지 못했다는 해석과 일치한다.

```text
Jena P/Q complete-row MSE
role             P MSE      Q MSE      complete rows
pq_split         0.211579   0.068136   35040
raw64            0.212690   0.078705   35040
level_res        0.255840   0.068519   35040
level_raw        0.213403   0.077756   35040
direct_nlinear   0.201671   0.072313   35040
f0               0.213780   0.078520   35040
lora             0.172211   0.068379   35040
```

[확인] Jena에서는 `level_res`의 큰 손해가 P-space에 집중되어 있고, `pq_split`은 그 P-space MSE를 0.255840에서 0.211579로 낮췄다. Q-space는 0.068519에서 0.068136으로 거의 같다. 이 결과는 “LEVEL이 Q-only 보정이라 P-space TSFM 오차를 직접 고치지 못한다”는 v8 가설에 맞는다. 다만 LoRA의 P-space MSE 0.172211과 DIRECT의 P-space MSE 0.201671보다 아직 크기 때문에, P-head 하나로 정확도 격차가 해결됐다고 볼 수는 없다.

[확인] RAW64 대비 P/Q split의 P 차이는 작고(0.212690→0.211579), Q 차이가 더 크다(0.078705→0.068136). P 접근 추가와 Q 보정 분리 보존이 결합된 관찰이며, 모든 개선을 P-head 단독 인과효과로 돌리지 않는다. 두 분기의 공동학습·VAL checkpoint 선택도 달라질 수 있다.

## Cost decision

[확인] 새 비용 측정은 수행하지 않았다. 계획은 유지 근거가 있을 때 Robin 대상의 비용 측정을 허용했다. 이번에는 Jena에 한정된 방법 근거를 보존하되 Robin 구성을 바꿀 근거가 없고, Robin 측정만으로 Jena 배포 가치를 판단할 수 없다. 따라서 이번 선택을 바꿀 정보가 적다고 판단해 허용된 비용 측정을 생략했다. 이는 원래 승인 조건을 소급해서 좁힌 해석이 아니라 [cost_decision.json](cost_decision.json)에 남긴 종료 판단이다.

[확인] 따라서 이 문서에서는 Jena 비용 우위, v8 새 timing, 또는 과거 v4/v7 timing 대비 개선율을 주장하지 않는다. 비용 판단이 필요하면 Jena를 포함하는 별도 승인 범위에서 같은 회차 대응 측정으로 해야 한다.

## Limits

[확인] 이 회차의 주요 한계는 다음과 같다.

- Robin/Jena는 모두 노출된 개발 자료다. 개선이 있어도 독립 확증이 아니다.
- Jena에서는 P/Q split이 좋지만 LoRA와 DIRECT 격차가 남는다. 특히 LoRA 대비 combined MSE는 +16.26% 손해다.
- Robin에서는 기존 LEVEL 대비 평균상 +1.39% 손해이고 interval이 0을 가로지른다. 개선도, 보편적 열등도 확정할 수 없다.
- Q head rank는 그대로 32다. 이번 결과는 “보존 공간 head 추가”의 효과이지 Q 표현력 전체 탐색이 아니다.
- RAW64는 parameter-matched 대조지만 함수 공간이 P/Q split과 동일하다는 뜻은 아니다.
- P/Q 분해는 complete target row 기준 진단이다. masked macro MSE와 역할이 다르고, Robin에서는 coverage 차이 때문에 전체 지표로 대체할 수 없다.
- 코드/수치 검산의 실제 상태는 final_checks.json에서 확인한다. PASS는 독립 재현이나 과학적 주장·논문 채택의 보증이 아니다.

## Final decision and next one decision

[확인] v8은 `pq_split`을 기본 방법으로 승격하지 않는다. Robin에서 기존 LEVEL을 이번 수정으로 교체하지 않으며, 과거 자료별 선택을 일괄 LEVEL로 바꾸지 않는다. `pq_split`은 Jena에서 보존 공간(P) 보정의 조건부 신호를 보인 개발 후보로만 보존한다.

다음 의사결정은 하나다. **기존 자료별 구성을 유지하면서, Jena에서 남은 보존 공간의 예측 손해를 별도 개발 대상으로 계속 다룰지 결정한다.** 이번 결과는 그 질문을 좁혔지만 범용 수정을 확보하지 않았다. 새 후보·K·학습률·미사용 자료는 이번 회차에서 추가하지 않는다.

## Source index

- 계획과 수식: [PLAN.md](PLAN.md), [protocol.json](protocol.json), [model_v8.py](model_v8.py)
- 데이터·재사용 계약: [data_contract_robin.json](data_contract_robin.json), [data_contract_jena.json](data_contract_jena.json), [reuse_manifest.json](reuse_manifest.json)
- 선택 파일: [selected_robin.json](selected_robin.json), [selected_jena.json](selected_jena.json)
- 평가 seal과 노출: [evaluation_seal_robin.json](evaluation_seal_robin.json), [evaluation_seal_jena.json](evaluation_seal_jena.json), [test_exposure_robin.json](test_exposure_robin.json), [test_exposure_jena.json](test_exposure_jena.json)
- 최종 평가: [robin_eval01.json](robin_eval01.json), [jena_eval01.json](jena_eval01.json)
- 분석 요약: [analysis01/analysis_summary.json](analysis01/analysis_summary.json), [analysis01/role_scores.csv](analysis01/role_scores.csv), [analysis01/comparisons.csv](analysis01/comparisons.csv), [analysis01/space_scores.csv](analysis01/space_scores.csv)
- 실행 원장과 run 결과: [ledger.json](ledger.json), [runs/](runs/)
