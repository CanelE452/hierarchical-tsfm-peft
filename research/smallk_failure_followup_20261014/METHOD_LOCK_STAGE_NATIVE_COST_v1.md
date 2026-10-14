# N1/N2/N3 전체 배포 비용의 고정 프로토콜

상태: **SOURCE_ONLY / 실제 fixture·자산 읽기·비용 측정 미실행**. 실제 실행은 ROOT의 fresh stage authorization/manifest, 원조건4개의 CPU producer COMPLETE, selected-model SHA에 결합된 독립 audit PASS, 다른 실제 CPU/GPU job이 끝났다는 ROOT attestation 후만 가능하다. 기존 실행 소스·결과를 덮지 않는다.

Robin C17/K4, Beijing Aotizhongxin C11/K2, 각각 원 MAIN seed92601/92602를 유지한다. 첫8개의 기존 노출 VAL/EVAL 과거를 사용하고 batch1/8을 나눈다. 같은 original FIT FP32 mean/std, origin, original observation mask, 원 physical past에 동일한 vectorized causal CPU fill을 적용한다. inverse normalization과 physical D2H도 전체 시간에 포함한다. TEST·신규 holdout·추가 coefficient 학습·encoder/backbone 학습은 없다.

각 dataset/batch에서 경로는 정확히9개다.

```text
RAW_BOLT
MAIN_s92601, MAIN_s92602
N1_s92601, N2_s92601, N3_s92601
N1_s92602, N2_s92602, N3_s92602
```

따라서 whole 행은2 datasets×2 batches×9 routes=36개다. O0/O1 미래 oracle는 배포 비용·배포 순위에 포함하지 않는다. 현재값/평균잔차 진단의 별도 비용을 측정한 것으로 쓰지 않는다.

order seed는92725다. 고정한 permutation을 cyclic shift한9개의 Latin block을 사용한다. 각 route는 각 위치에 정확히1회 나온다. 각 route를2회 warmup한 다음9개 block을 측정한다. 같은 block의 RAW 시간으로 route/RAW ratio를 계산하고 RAW absolute time도 남긴다. median(ratios)를 median(route)/median(RAW)와 같다고 가정하지 않는다. 모든 carryover pair나 clock/thermal 상태를 균형화한 설계는 아니다.

원 RAW/MAIN은 기존 GPU F 경로를 그대로 사용한다. 신규 경로는 scientific CPU runner와 동일 계산 recipe를 사용한다.

```text
동일 physical past
  → CPU causal fill + FIT normalization
  → normalized H2D → frozen original E GPU
  → latent와 current-C D2H
  → CPU 33K block features + FIT feature state + FP64 ridge
  → N1: CPU 원 D의 기존 FP32 anchored operation order
     N2: CPU direct-C current anchor
     N3: selected N1을 내부 계산하는 bound past-only 공개 API + CPU calibrated D
  → normalized prediction H2D
  → 기존 GPU inverse normalization → physical D2H → final sync
```

N1은 `frozen_decoder_refit.original_anchored_prediction`, N2는 `NativeModel.direct_forecast`, N3는 `DecoderCalibration.forecast_from_selected_n1`을 사용한다. 전체 native 경로의 F 인수는 사용하지 않는다. F는 원 RAW/MAIN을 위해 로드된 상태이므로 memory는 model-exclusive 증가량이 아니다. 이 hybrid CPU/GPU 구현이 최적 구현이라는 주장은 하지 않는다. 전송·CPU feature/ridge/decoder 시간은 숨기지 않고 whole 시간에 포함한다.

각 route의 같은 batch에 fresh reference를 만든다. normalized reference는 원 history에서 E/GPU와 동일 scientific CPU math를 다시 실행한다. physical reference는 기존 loop causal preprocessing 및 기존 controls deployment/inverse를 사용한다. 이는 새로운 native 수학의 독립 구현이 아니라 matching-batch 실행 참조이며, 독립 saved-array audit와 구분한다.

모든9번의 normalized·physical 출력, block/position, seconds, same-block RAW seconds/ratios, memory를 저장한다. mandatory same-batch parity는 normalized atol1e-5/rtol2e-6, physical atol0.002/rtol2e-5를 고정한다. 기존 bulk-selected 출력과의 parity는 별도 diagnostic flag로 모두 보존한다. normalized 출력의 추가 D2H 복사는 timer 밖이고, physical D2H는 timer 안이다. disk/model load, coefficient restore, reference inference, source hashing, 저장, resource guard는 측정 timer 밖이지만 **GPU process wall budget에는 포함**한다. 기존 teacher precompute 비용은 unknown이며 무료로 기록하지 않는다.

full storage/source/resource guard는 group/reference/Latin block 경계에 사용하고 fast process/GPU/RSS/deadline/local180 guard는 warmup과 각 trial 앞뒤에 적용한다. 최종 original codec/native coefficient/original CPU D/F hashes와 no-F-gradient, 입력/source snapshot SHA를 확인한다. 이 whole 프로토콜에는 stage decomposition이 없다.

local GPU process180초와 fresh shared GPU600초/CPU900초/wall2400초/저장 증가1GiB를 유지한다. 실제 close/최종 guard 시간도 청구하고 실패·budget terminal이 producer COMPLETE로 남지 않도록 한다. 상한을 자동 늘리지 않는다. 전체36 route가 실제9회 출력·same-batch parity·preservation을 통과해야 COMPLETE다. 실패나 budget 중단의 존재하는 배열/행/receipt는 보존한다. 미완료 family 때문에 default full9-route gate가 없으면 비용은 NOT_EXECUTED이며 method/teacher count 조건을 바꾸지 않는다.

SOURCE CLI:

```powershell
.venv/Scripts/python.exe research/smallk_failure_followup_20261014/profile_stage_native_cost_v1.py --describe

# 아래 actual 경로는 terminal queue의 producer record와 실제 PASS audit에서 확인 후 전달한다.
.venv/Scripts/python.exe research/smallk_failure_followup_20261014/profile_stage_native_cost_v1.py --execute --input-bridge <original_bridge_RESULT.json> --native-results <actual_robin_seed1_RESULT.json> <actual_robin_seed2_RESULT.json> <actual_beijing_seed1_RESULT.json> <actual_beijing_seed2_RESULT.json> --native-audit <actual_bound_PASS_audit.json> --tag <fresh_tag> --no-contention-attested
```
