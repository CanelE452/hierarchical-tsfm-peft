# Hierarchical TSFM PEFT 실행 보고

최종 판단: **STOP_CURRENT_HIER_ADAPTER**

[확인] MASTER 실행 계약에 따라 Labour screen을 완료했습니다. 이 판단은 후속 연구 예산에 관한 결정이며 논문 PASS 선언이 아닙니다.

Labour continuation: `STOP_CURRENT_HIER_ADAPTER_AFTER_LABOUR_SCREEN`

## 환경과 검산

Python 3.11.16, PyTorch 2.10.0+cu128 / CUDA 12.8, RTX 4070 12GB. 정확한 전체 버전·wheel provenance는 ENVIRONMENT.json과 requirements-lock.txt에 기록했습니다.
Chronos model revision은 MODEL_SOURCE_MANIFEST.json에 고정했습니다. F0 공식 parity와 세 seed의 adapter step0 parity는 실제 Labour 입력에서 차이 0으로 확인했습니다.
PEFT의 embedding accessor 호환성 오류는 기존 shared embedding을 반환하도록 연결하여 해결했고, 파라미터 수·동결 가중치·예측 parity를 검증했습니다.

## 데이터와 평가 범위

공식 datasetsforecast.HierarchicalData로 Labour와 TourismLarge를 받았으며 전체 시간축의 합계·누락·중복을 감사했습니다. 원본과 weights는 Git에서 제외했습니다.
Context 48개월, horizon 12개월입니다. TRAIN/CALIBRATION/VALIDATION/TEST를 분리했고 TEST는 LR·checkpoint·reconciler 선택에 쓰지 않았습니다.
Primary는 series RMSSE → 각 공식 level 평균 → level 동일 가중 평균입니다. denominator는 TRAIN seasonal scale입니다.
MinT residual은 13개 CALIBRATION origins × 12 horizons이며, 서로 겹치는 달의 예측 오차도 별도 관측으로 유지했습니다. TEST의 37 rolling origins 또한 겹치므로 독립 표본처럼 통계적 유의성을 주장하지 않습니다.
모든 neural arm에 unreconciled / BottomUp / OLS / MinT-shrink를 동일하게 제공했습니다. Statistical baseline도 같은 48개월 context와 reconciliation을 사용했습니다.

## Labour 결과

아래 neural 값은 선택된 seed들의 평균입니다. Seed별 방향은 별도로 제시합니다.

```text
                              primary  bottom_rmsse   raw_mae
arm           reconciliation
AutoETS       bottom_up      0.991299      1.153678 15.091965
              mint_shrink    0.953704      1.193945 13.727674
              ols            1.037989      1.239631 13.000745
              unreconciled   0.898294      1.153678 13.001295
F0            bottom_up      1.396376      1.335454 22.695941
              mint_shrink    1.090013      1.274303 15.011049
              ols            1.073237      1.203647 15.193725
              unreconciled   1.069613      1.335454 16.545204
LORA          bottom_up      0.822949      1.048550 12.012981
              mint_shrink    0.906933      1.143659 13.089552
              ols            0.848596      1.055403 12.434074
              unreconciled   0.834648      1.048550 12.352611
LORA_HIER     bottom_up      0.822308      1.048403 12.000614
              mint_shrink    0.905467      1.142501 13.068984
              ols            0.847991      1.055208 12.421979
              unreconciled   0.834182      1.048403 12.340904
LORA_POOL     bottom_up      0.814569      1.054826 11.661023
              mint_shrink    0.889541      1.138829 12.722460
              ols            0.824582      1.052910 11.913758
              unreconciled   0.815732      1.054826 11.861156
LORA_SELF     bottom_up      0.822589      1.048499 12.008277
              mint_shrink    0.906169      1.143188 13.081305
              ols            0.848458      1.055409 12.430887
              unreconciled   0.834553      1.048499 12.349709
SeasonalNaive bottom_up      1.335655      1.393398 22.974934
              mint_shrink    1.335655      1.393398 22.974934
              ols            1.335655      1.393398 22.974934
              unreconciled   1.335655      1.393398 22.974934
```

- LoRA 자체가 F0 대비: MinT primary -16.80%, reconciliation 전 -21.97% (음수가 유리).
- SELF가 LoRA 대비: MinT primary -0.08%, reconciliation 전 -0.01% (음수가 유리).
- POOL이 SELF 대비: MinT primary -1.84%, reconciliation 전 -2.26% (음수가 유리).
- HIER가 SELF 대비: MinT primary -0.08%, reconciliation 전 -0.04% (음수가 유리).
- HIER가 POOL 대비: MinT primary +1.79%, reconciliation 전 +2.26% (음수가 유리).
- HIER가 LoRA 대비: MinT primary -0.16%, reconciliation 전 -0.06% (음수가 유리).
- 가장 낮은 통계 baseline primary는 AutoETS 0.953704; HIER는 0.905467입니다.

Seed 효과: HIER − 비교 arm. 양수는 HIER의 오차가 더 큽니다.

```text
 seed comparator reconciliation  primary_difference  bottom_relative_difference
92110       LORA   unreconciled           -0.001184                   -0.000037
92110  LORA_SELF   unreconciled           -0.001204                   -0.000091
92110  LORA_POOL   unreconciled           -0.000543                   +0.000365
92110       LORA    mint_shrink           -0.002396                   -0.000577
92110  LORA_SELF    mint_shrink           -0.001938                   +0.000530
92110  LORA_POOL    mint_shrink           -0.001487                   -0.000276
92111       LORA   unreconciled           -0.000259                   +0.000125
92111  LORA_SELF   unreconciled           -0.000029                   +0.000134
92111  LORA_POOL   unreconciled           +0.055910                   -0.018286
92111       LORA    mint_shrink           -0.003602                   -0.001442
92111  LORA_SELF    mint_shrink           -0.000643                   -0.001063
92111  LORA_POOL    mint_shrink           +0.049071                   +0.011138
92112       LORA   unreconciled           +0.000046                   -0.000513
92112  LORA_SELF   unreconciled           +0.000120                   -0.000322
92112  LORA_POOL   unreconciled           -0.000019                   -0.000083
92112       LORA    mint_shrink           +0.001602                   -0.001014
92112  LORA_SELF    mint_shrink           +0.000476                   -0.001263
92112  LORA_POOL    mint_shrink           +0.000196                   -0.001231
```

각 level과 raw MAE는 LEVEL_SCORES.csv / RAW_SCORES.csv, 각 origin은 ORIGIN_SCORES.csv에 보존했습니다.

Neural fits 16개, main updates 8,192회; 기록된 fit/baseline 실행 시간 합 7.7분.

MinT 이후 level별 HIER 효과(선택된 세 seed 평균, 음수가 유리):

```text
                                             difference  relative_difference
comparator level
LORA       Country                            -0.001666            -0.001990
           Country/Employment/Gender/Region   -0.001159            -0.001011
           Country/Gender/Region              -0.001683            -0.001826
           Country/Region                     -0.001353            -0.001529
LORA_POOL  Country                            +0.038853            +0.066340
           Country/Employment/Gender/Region   +0.003672            +0.003211
           Country/Gender/Region              +0.010492            +0.012005
           Country/Region                     +0.010690            +0.012987
LORA_SELF  Country                            -0.000951            -0.001113
           Country/Employment/Gender/Region   -0.000687            -0.000599
           Country/Gender/Region              -0.000737            -0.000798
           Country/Region                     -0.000433            -0.000471
```

전체 seed/level/reconciliation 효과는 LEVEL_EFFECTS.csv에 있습니다.

이 선택된 모델들에서 HIER primary는 raw 0.834182, MinT 0.905467입니다. 합계 일치 자체를 예측 성능 개선으로 해석하지 않았고, TEST를 보고 reconciliation을 변경하지 않았습니다.

## 후속 결정과 한계

```json
{
  "decision": "STOP_CURRENT_HIER_ADAPTER_AFTER_LABOUR_SCREEN",
  "checks": {
    "LORA_SELF_mean_primary_favorable": true,
    "LORA_SELF_both_repeats_favorable": false,
    "LORA_SELF_bottom_no_large_harm": true,
    "LORA_SELF_raw_mean_favorable": true,
    "LORA_POOL_mean_primary_favorable": false,
    "LORA_POOL_both_repeats_favorable": false,
    "LORA_POOL_bottom_no_large_harm": true,
    "LORA_POOL_raw_mean_favorable": false
  },
  "nature": "project budget continuation rule, not statistical significance"
}
```

Labour continuation 조건을 모두 충족하지 않아 TourismLarge 학습은 실행하지 않았습니다. TourismLarge 다운로드·감사는 Stage B 학습과 구분합니다.

선택 seed는 개발 과정에 사용되었으며 두 repeat seeds를 별도로 보고했습니다. overlapping origins와 작은 seed 수를 고려할 때 이 결과로 보편적 우월성이나 통계적 유의성을 주장할 수 없습니다.
`OPTIMIZATION_LIMIT_REACHED` 표시는 마지막 checkpoint의 validation이 계속 개선된 fit에만 남겼으며 학습을 연장하지 않았습니다.
추가 LR/rank/adapter/graph/loss 탐색은 실행하지 않았습니다. 환경·구현 오류 기록은 IMPLEMENTATION_EVENTS.json에 있으며 과학적 결과와 분리했습니다.

모든 fit과 TEST 평가가 끝난 뒤 continuation boolean의 JSON 직렬화 오류가 발생했습니다. 완료된 점수표에서 동일 조건을 재계산하고 Python bool로 저장해 복구했습니다. 재학습·추가 update는 0회입니다. 실행 당시 source seal은 보존했고, 재발 방지용 후처리 수정과 회귀 테스트는 POST_RUN_REPAIRS.json에 별도 기록했습니다.
