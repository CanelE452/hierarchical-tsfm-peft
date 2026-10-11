> 게시용 사본입니다. 실험 완료 당시 원본 문서의 SHA256은 `b5b028baa64967e7a7d1b81080480ad3f58792bc8003d58ce5b787fa9f328244`입니다. 본문 수치·판단을 유지하고 링크만 GitHub 경로로 조정했습니다. 원문의 “이번 작업에서 push하지 않음”은 과학 실험 완료 당시 기록이며, 후속 사용자 승인으로 이 공개 사본을 게시합니다. 구 TEST 사고는 [범위·원본 SHA](PROTECTED_ACCESS_CONTEXT.json)만 공개하며 보호된 통계는 재게시하지 않습니다.

# Jena TS2 batch4 재시도 메모

작성 시각: 2026-10-11 KST  
범위: 읽기 전용 진단. 과학 결과, runner, protocol, checkpoint, failure/result 파일은 수정하지 않았다. 이 파일만 새로 작성했다.

## 결론

[확인] `jena_TS2_d16_s92601`, `jena_TS2_d320_s92601`의 batch 8 실패는 PyTorch CUDA OOM 예외가 아니라 `common_direct.Resources.check()`의 `process RAM/allocated VRAM cap` 가드가 발생한 것이다. 두 실행 모두 `peak_cuda_allocated_bytes`는 8GiB 상한보다 낮았고, `peak_rss_bytes`가 8GiB RAM 상한을 넘었다.

```text
run                         iter/expected   peak RSS              CUDA allocated        CUDA reserved        중단 지점
jena_TS2_d16_s92601         63/600          12.13 GB / 11.30 GiB  5.18 GB / 4.82 GiB    22.22 GB / 20.69 GiB official_TS2Vec_representation_fit
jena_TS2_d320_s92601        63/600          11.35 GB / 10.57 GiB  5.49 GB / 5.11 GiB    21.39 GB / 19.92 GiB official_TS2Vec_representation_fit
budget cap                  -               8.59 GB / 8.00 GiB    8.59 GB / 8.00 GiB    reserved는 가드 조건이 아님
```

[판단] `--encoder-batch 4` 재시도는 타당하다. 이유는 실패 stage가 공식 TS2Vec representation fit 내부이고, Jena TRAIN이 `max_train_length=3000` 규칙으로 8개 section으로 나뉜 뒤 batch 8에서 한 번에 8개 section을 처리하기 때문이다. batch 4로 낮추면 모델 설정(`hidden_dims=64`, `depth=10`, `max_train_length=3000`, `600` iterations)은 유지하면서 한 step에서 동시에 처리하는 section 수만 줄인다.

## 확인 근거

[확인] 실패 receipt:

- `research/professor_compressor_direct_20261011/runs/jena_TS2_d16_s92601/failure.json`
- `research/professor_compressor_direct_20261011/runs/jena_TS2_d320_s92601/failure.json`

두 receipt 모두 다음을 기록한다.

```text
status                    incomplete
error                     BudgetExceeded('process RAM/allocated VRAM cap')
actual_encoder_iterations 63
stage                     official_TS2Vec_representation_fit
F_forward_calls           0
F_backward_calls          0
```

[확인] 초기 protocol:

- `.cache/professor_compressor_direct_20261011/runs/jena_TS2_d16_s92601/initial_protocol.json`
- `.cache/professor_compressor_direct_20261011/runs/jena_TS2_d320_s92601/initial_protocol.json`

```text
TRAIN_shape                  [1, 26207, 21]
expected_official_iterations 600
contiguous_TRAIN_only        true
input_missing                official NaN masking
hidden_dims                  64
depth                        10
batch_size                   8
max_train_length             3000
protected_TEST_new_access    0
```

[확인] 공식 TS2Vec fit 경로는 `research/level_k_tradeoff_20261011/vendor/ts2vec/ts2vec.py` 기준이다. `fit()`은 `train_data.shape[1] // max_train_length`가 2 이상이면 `split_with_nan(..., axis=1)`로 긴 시계열을 여러 section으로 나눈다. Jena의 `26207 // 3000 = 8`이므로 batch 8은 이 8개 section을 한 batch로 처리한다. 이후 각 iteration에서 `x[:, window_offset:window_offset + max_train_length]`로 길이 3000 창을 뽑고, 두 augmented crop을 만들어 `_net`을 두 번 호출한 뒤 hierarchical contrastive loss를 계산한다.

[확인] `runner_ts2.py`는 `--encoder-batch` 인자를 이미 가지고 있고 기본값만 8이다. `create_ts2(..., batch_size=args.encoder_batch)`로 공식 TS2Vec batch size에 전달하며, 완료 result에는 `encoder_batch_adjustment = args.encoder_batch != 8`이 기록되도록 되어 있다.

## batch4 재시도의 의미

[확인] batch4는 다음을 바꾸지 않는다.

```text
TS2Vec vendor source       pinned official source 그대로
hidden_dims                64 유지
depth                      10 유지
max_train_length           3000 유지
expected iterations        Jena 600 유지
TRAIN/VAL data contract    동일 hash 유지
TEST access                새 접근 없음
```

[확인] batch4는 다음만 바꾼다.

```text
encoder batch size         8 -> 4
동시 처리 section 수        8개 -> 4개
mini-batch gradient 구성   변경됨
metadata                   encoder_batch_adjustment=true로 기록 가능
```

[판단] 따라서 batch4는 “모델을 작게 만든다”기보다 “같은 공식 모델/길이/iteration 예산을 작은 mini-batch로 실행한다”는 자원 회피 재시도다. 결과 평균 비교에서는 batch size 변경을 명시해야 하지만, `hidden64/depth10/max_train_length3000/600iters` 조건은 유지된다.

## 원인 분리

[확인] 직접 중단 조건은 host RSS 초과다. CUDA allocated는 8GiB보다 낮다. 즉 “allocated VRAM이 8GiB를 초과해서 실패했다”는 설명은 현재 receipt와 맞지 않는다.

[확인] CUDA reserved는 20GiB 안팎으로 크지만 `Resources.check()`는 reserved를 중단 조건으로 사용하지 않는다. reserved 증가는 PyTorch CUDA caching allocator가 이미 확보한 영역일 수 있으므로, 그 자체가 실패 조건은 아니다.

[추정] 가능 원인 1: 공식 TS2Vec의 dynamic crop length가 매 iteration 달라져 convolution/autograd/caching allocator가 다양한 shape의 workspace/cache를 만들고, host RSS와 reserved CUDA memory가 iteration이 진행되며 증가했을 수 있다. 근거는 두 dim 모두 정확히 63 iterations에서 RAM guard에 걸렸고, 공식 fit이 매 iteration 임의 crop 길이로 두 번 `_net` forward를 수행한다는 점이다. 다만 현재 로그만으로 cuDNN algorithm cache 또는 host cache가 직접 원인이라고 확정할 수는 없다.

[추정] 가능 원인 2: batch 8에서 Jena section 8개를 동시에 처리하므로, 길이 3000 근처 crop이 나오는 iteration에서 activation/workspace가 크게 증가했을 수 있다. batch4는 이 동시 처리량을 줄여 RSS/allocated/reserved를 모두 낮출 가능성이 높다.

[확인] 원인으로 보기 어려운 것:

```text
TEST 누출              initial_protocol protected_TEST_new_access=0
F 호출/학습            failure resources F_forward_calls=0, F_backward_calls=0
decoder/calibration    실패가 encoder fit stage에서 발생해 아직 도달하지 않음
output dim만의 문제    d16, d320 모두 63/600에서 중단됨. d320 allocated가 더 크긴 하나 d16도 동일 증상
```

## 재시도 제안

[판단] 다음 retry는 새 ID와 새 retry tag로 보존 실행하는 것이 맞다. 기존 `jena_TS2_d16_s92601`, `jena_TS2_d320_s92601` 실패 결과와 partial checkpoint는 유지해야 한다.

권장 실행 조건:

```text
--dataset jena
--dim 16 또는 320
--seed 92601
--encoder-batch 4
--retry-tag batch4 또는 protocol에서 정한 canonical retry tag
```

[판단] batch4에서도 RAM cap을 넘으면 그때는 batch size만의 문제가 아니라 dynamic shape/cache 누적 가능성이 커진다. 그 경우에는 source를 바꾸기 전에, 같은 모델 설정을 유지하면서 다음 메타데이터를 더 기록하는 짧은 진단 run이 필요하다.

```text
iteration별 RSS
iteration별 cuda_allocated / cuda_reserved
crop_l 또는 crop tensor length 요약
torch.cuda.empty_cache 사용 여부는 별도 프로토콜 변경으로 분리
cuDNN enabled/benchmark/deterministic 상태
```
