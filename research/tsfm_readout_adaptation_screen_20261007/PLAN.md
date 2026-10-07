# TSFM 출력·native head 대조 실행 계획 — 2026-10-07

> 실행자는 [상세 계약](REQUEST_GOAL_PLAN.md)과 이 계획을 함께 읽는다. 승인 뒤에는 superpowers:subagent-driven-development 또는 executing-plans의 검증 절차를 적용하되, 이 계약의 단계·예산·commit 범위를 우선한다.

**Goal:** 출력 affine 보정과 동결 표현을 읽는 원래 예측 헤드의 적응으로 얻는 개선을 먼저 확인하고, 동일한 HEAD 부모 이후 내부 LoRA의 추가 가치와 저비용 진단 문제가 성립하는지 후속 단계로 구분한다.

**Architecture:** 원채널 Bolt-small F0를 유지한다. OUTPUT는 최종 point forecast의 C→C affine만, HEAD는 원래 output_patch_embedding 전체만 학습한다. prefix가 고정된 S1은 정확한 특징 캐시를 사용할 수 있다.

**Tech Stack:** 현재 Python 환경의 torch2.10.0+cu128, chronos-forecasting2.3.2, transformers5.17.0, peft0.21.0, numpy2.4.6. RTX4070 12,282MiB, CPU threads4, FP32, TF32 off. 설치·업그레이드·새 모델 다운로드는 포함하지 않는다.

**Spec:** [REQUEST_GOAL_PLAN.md](REQUEST_GOAL_PLAN.md), SHA256 `18e4f9e3254e5be7bbd7c4a8d4c6cffa3d443d097739f54d3d62bb863763f737`.

## 현재 상태와 권한

[확인] 2026-10-08 S0 native calibration을 완료했다. 세 자료 전체 VAL 초기 F0, cache 값·loss·gradient, 결측 B4/micro4/2/1, frozen 전체 parameters/buffers, Adam/scheduler/RNG 복원이 통과했다. native disposable updates384, scientific fits0/TEST0이며, 실제 비용은 calibration_result.json·guard_repair.json·s0_budget_ledger.json에 기록했다. S1 실행 직전 상태다.

사용자의 직접 `/goal` 요청에 따라 목표를 등록했다. 첨부 문서의 slash 명령은 인용한 상세 계약이며 별도 명령으로 재실행하지 않는다. 2026-10-08 직접 `해줘`로 S0 상한을 승인했고, 이어 `이런걸로 멈추지 마`라는 최신 계속 지시를 S1 최대24 fits 범위의 실행 지시로 적용한다. 예산을 다시 묻는 중간 정지는 생략하되, 수치 상한을 S1 전에 공개·기록하고 준수한다. S2 권한은 확대하지 않는다.

첨부 §4의 비용 게이트에 따라 S0 실측을 먼저 수행했다. [S1 수치 예산](s1_budget.json)은 조기정지 가정 없이 비용을 추정하며 GPU 작업 누적8시간, CPU-only 작업30분, CPU process117,000초, 전체9시간, campaign 총disk3GiB, RAM/VRAM 각8GiB를 상한으로 한다. 원래20-update 계측은 느린 원장 I/O를 포함하므로 별도 보존하고, 수정한 계측의6개 single-update 진단은 steady-state 측정으로 주장하지 않는다. 냉시작 이상치와 비용 불확실성을 보수적 상한에 포함한다. 실제 사용량은 별도 보고하며 run별 반복 승인은 없다.

S2는 별도 승인 전 실행하지 않는다. S3/S4의 selector/새 PEFT/새 데이터·백본/새 loss/Conv/K 탐색은 현재 실행 금지다. PCA·채널 압축·residual G·큰 백본의 필요성, 신규성, HEAD 성공을 미리 결론으로 두지 않는다.

## S0에서 확인한 출발점

- [확인] 감사 당시 branch main, local HEAD/tracking/live remote 모두 `8646e54630add0777c90686b5a8e4bef20c7dcea`. [S0 snapshot](s0_audit.json)에 시각·dirty census·보호 해시를 기록했다.
- [확인] staged/tracked diff 없음, 미추적1,889개. 기존 산출물·PPT/PDF·E/D·Conv 계획은 보존한다. 초기 Windows 프로세스 검색에 Python/Conda/WSL/SSH 연구 작업이 없었고, 조회된 프로젝트 채팅 중 현재 채팅만 active였다. 이는 이후 실행 상태의 보장이 아니다.
- [확인] GPU는 약1.5GiB를 다른 앱이 사용 중이었다. 장치가 비었다고 가정하거나 무관한 작업을 종료하지 않는다. 실행 직전에 다시 조회한다.
- [확인] v18 실제 GPU preflight21.847596초, 다른 E/D 전체 projection11,421.533초, 당시 cap10,800초. fit/update/새 TEST 모두0인 RESOURCE_BLOCKED다. 과학적 실패나 실제3시간 소비로 취급하지 않는다.
- [확인] 정확한 현재 OUTPUT/HEAD 완료 fit chain은 검색 범위에서 찾지 못해 exact fit 차감0, S1 최대24 fits가 남는다. 오래된 Chronos-2 HEAD는 실제 관련 실험이므로 “헤드 실험이 전혀 없다”라고 하지 않는다.
- [확인] 기존 F0/LEVEL/MSE-LoRA/native-LoRA/NLinear/Mini/Tiny/Chronos-2 정확도와 checkpoint·선택·예측 원장은 재사용한다. 정확도 재사용과 같은 회차 비용 측정은 구분한다. [재사용 분류](reuse_manifest.json).
- [확인] 설치 source·pinned 모델 hash와 데이터 원장은 [source_binding.json](source_binding.json), [data_contract.json](data_contract.json)에 연결했다. 숫자 parity/gradient는 아직 실행 전이다.

### 선행연구가 이미 포함하는 대조

[확인] Time-PEFT에는 HeadOnly가 있고, appendix는 Bolt-small의 원래 encoder-decoder HeadOnly/LoRA도 보고한다. TRACE 원문에도 head-only 및 head+LoRA가 있다. AdaPTS에는 decoder-only affine 및 head-first 절차가 있다. 따라서 “헤드 대조를 최초로 추가했다” 또는 “헤드 후 내부 적응 순서가 새롭다”는 주장을 하지 않는다.

읽은 원문/코드 범위와 미확인 exact 계약은 [prior_art_map.md](prior_art_map.md)에 기록한다. 이번 질문의 후보 차이는 **같은 선택 HEAD 부모와 같은 추가 학습 기회에서 남는 이득을 사전에 판단하고, 표준 pilot보다 총비용·선택 위험을 줄이는가**다. 이 차이와 연구 문제의 성립 자체는 미검증이다.

## 고정할 데이터·선택 계약

모든 숫자·origin 배열·채널 순서는 hash-bound 기존 canonical npz/contract에서 가져온다. 새 표준화·imputation·자료 제외·context extension은 없다.

```text
자료      C   L    H   TRAIN   VAL   TEST-A/B   간격    sampler period
Robin    17  512  48    1945    30     40/40    1시간         24
Peacock  13  512  48    1945    30     40/40    1시간         24
Jena     21  512  48   25648   371    365/365   10분         144
```

입력은 canonical standardized Z의 `[origin-512:origin]`만이다. target와 mask의 `[origin:origin+48]`은 loss/평가에만 사용한다. TRAIN nanmean/nanstd와 standardized0 imputation을 그대로 쓴다. 기존 raw-channel forward는 imputed context만 전달하므로 새 historical observation mask를 encode에 추가하지 않는다. 누락 target는 raw finite mask로 제외하며 실제 관측0은 제외하지 않는다. PCA basis를 포함하는 기존 파일을 읽더라도 S1 함수는 Z만 사용한다.

공통 fit recipe는 LR 순서 `[1e-3,1e-4]`, seeds92601/92602, 최대120 epochs, patience6, epoch_origins512, effective_B4, AdamW wd0, clip1이다. Plateau factor.5/patience2/relative threshold1e-4, epoch>=1의 VAL 뒤만 scheduler step을 사용한다. epoch0은 평가·선택에 포함하지만 scheduler observation에 넣지 않는다.

각 epoch1..120의 표본은 parent의 `SeedSequence([seed,epoch])`와 phase별 without-replacement 후 permutation을 재사용한다. Robin/Peacock은 phase0..7 각22개, 나머지21개; Jena는 phase0..79 각4개, 나머지3개다. 같은 자료/seed/epoch의 두 arm·두 LR origin 순서가 같아야 한다.

각 run은 전체 VAL masked channel-macro MSE의 strict minimum을 선택하며 epoch0을 포함하고 exact tie는 earliest다. 자료/arm마다 두 seed의 선택 VAL **손실 평균**으로 LR 하나를 선택한다. 이번 계약의 LR exact tie는 모든 자료에서 first-listed1e-3이다. Peacock 기존 v12 numeric1e-4 tie를 그대로 복사하지 않는다. 이는 최신 명시 계약에 따른 차이다.

v6/v7 원래 LoRA의 native quantile loss와 이번 point MSE를 혼용하지 않는다. 기존 MSE-LoRA 참조는 v11/v12 full_mse 원장에 연결한다. q0.5 위치 첫48을 point forecast로 사용하지만 MSE 적응 뒤 통계적 분위수 보정을 주장하지 않는다.

감독 loss는 inverse normalization 후 TRAIN-standardized 좌표에서 계산한다.

```python
count = effective_mask.sum((0, 1))     # entire effective B4, channel-wise
valid = count > 0
loss = (squared_error_sum[valid] / count[valid]).mean()
```

microbatch4→2→1로 줄여도 전체 B4의 count/valid channels를 공유한다. 각 micro SSE/count 기여를 합해 backward하고 clip/step은 한 번 한다. micro loss의 단순 평균이나 추가 `/num_microbatches`는 금지다. update/표본/VAL 기회를 줄이지 않는다. full VAL은 각 채널 count>0이어야 한다.

## S1 함수·학습 경계

[정적 흐름 확인] `architecture_boundary.json`에 installed source와 safetensors header 검산을 기록했다.

```text
x[B,512,C] → rows[B*C,512]
  frozen encode → frozen Transformer decode
  h[B*C,1,512], loc/scale[B*C,1]
  → trainable entire output_patch_embedding
  → reshape[B*C,9,64] → original inverse normalization
  → q index4, first48 → y[B,48,C]
```

### OUTPUT_AFFINE

```python
y0 = frozen_F0(x)               # dataset-standardized [B,48,C]
prediction = y0 @ W + beta      # W[C,C], beta[C], shared horizons
# W=eye(C), beta=zeros(C); train only W and beta
```

306/182/462 scalars for Robin/Peacock/Jena. No raw-history branch, activation, horizon mixing, PCA/E/G/internal adaptation. Initial function must match F0.

### NATIVE_HEAD

원래 `output_patch_embedding`의6 tensors 전체를 pretrained 값으로 시작한다. ReLU/dropout 비선형 hidden512→2048→576와 residual512→576, bias를 모두 유지한다. 총2,526,336 parameters이며 기존 LoRA294,912보다 작지 않다. 새 linear probe로 대체하지 않는다.

```python
with torch.no_grad():
    encoded, loc_scale, embeds, attention = model.encode(context=rows)
    h = model.decode(embeds, attention, encoded)
raw = model.output_patch_embedding(h).view(B*C, 9, 64)
native = model.instance_norm.inverse(raw.view(B*C, -1), loc_scale)
prediction = native.view(B*C, 9, 64)[:, 4, :48]
prediction = prediction.reshape(B, C, 48).transpose(1, 2)
```

실행 시 실제 effective normalization callable을 부모 wrapper로부터 그대로 바인딩한다. v11/v12의 zero-variance forward patch도 source binding에 남긴다. `pipeline.embed()`는 encoder token과 squeeze된 norm을 반환하므로 위 prefix를 대체할 수 없다.

전체 dropout을 eval/off로 유지하고 head의 requires_grad만 켠다. 전체 forward를 no_grad/inference_mode로 감싸지 않는다. 캐시 h는 autograd가 저장할 수 있는 일반 tensor로 사용한다.

registered/trainable 전체 scalars2,526,336과 계산상 직접 loss에 연결되는1,173,600 scalars를 구분한다. output/residual row256:304가 q0.5 첫48에 해당하므로 다른 행의 zero gradient는 정상이다. 실제 nonzero/finite gradient와 optimizer state는 실행으로 확인한다. frozen 불변 해시는 parameters와 **모든 named_buffers**를 포함한다. state_dict에 없는 persistent=False quantiles도 빠뜨리지 않는다.

OUTPUT는 채널을 섞고 HEAD는 가중치를 공유하는 채널독립 함수다. 두 arm 차이를 순수 정보량 효과로 해석하지 않는다.

## 캐시·저장 범위

정해진 seeds×epoch1..120 schedule 합집합을 CPU로 확인했다. Robin/Peacock은 TRAIN 전체1945, Jena는25403개다. VAL 전체를 더하면 캐시 origins는1975/1975/25774다. 이 사전 지정 범위를 VAL 성능과 무관하게 먼저 봉인한다. full TRAIN보다 줄어드는 storage는0.85%로 작아 큰 비용 절감으로 표현하지 않는다.

```text
자료      h+loc/scale bytes  OUTPUT point bytes   합계
Robin          69,030,200          6,446,400        75,476,600
Peacock        52,787,800          4,929,600        57,717,400
Jena        1,112,818,224        103,920,768     1,216,738,992
전체        1,234,636,224        115,296,768     1,349,932,992
```

FP32 payload 공식은 `N*C*(512+2)*4`와 `N*48*C*4`다. full TRAIN+VAL은1,361,498,952bytes, 차이는11,565,960bytes다. file headers, origin arrays, target/mask, checkpoints는 별도다. sampled TRAIN curves와 고정 schedule-union TRAIN 평가를 명확히 표기한다. full canonical TRAIN 점수가 필요하면 추가245 Jena origins의 생성·평가 비용을 명시하고 승인 상한 안에서 처리한다.

현재 exact raw TRAIN F0/native h cache는 scoped search에서 찾지 못했다. 기존 Robin/Jena F0 VAL과 모든 F0 TEST는 원장 검증 후 재사용할 수 있다. Peacock stored F0 VAL은 미확인이라 대체하지 않는다. v14 direct/LEVEL cache, v15 K-channel cache, 오래된 Chronos-2 HEAD cache는 exact S1 특징이 아니다.

cache key는 원자료·preprocessing·origin 순서·채널·prefix weights·effective code·dtype·norm state·q index/horizon을 묶는다. feature 생성 함수는 x만 받고 target/mask를 인자로 받지 않는다. labels는 별도 감독 역할로 관리한다. 같은 prefix cache를 두 seeds/LRs가 공유해도 실제 생성/I/O를 총비용에서 한 번 계상한다. optimizer steps를 arm마다 늘리지 않는다. S2 LoRA가 prefix를 바꾸면 이 cache를 재사용하지 않는다.

계산상 HEAD parameter bytes10,105,344(9.64MiB), gradient+Adam m/v 포함40,421,376bytes다. 실제 peak가 아니다. 12 HEAD fits마다 best head10,105,344bytes + resume head/Adam30,316,032bytes(28.91MiB)를 유지하면 raw payload485,056,512bytes(462.59MiB)다. OUTPUT12 fits는 작다. full backbone을 run마다 저장하지 않는다. static S1 증분 disk 제안은3GiB이며 actual caches+I/O+checkpoints+reports의 중간 peak도 감시한다.

## 승인 대상: S0 비용보정 probe만

목적은 **학습 정상성·exact 새 경로 비용을 측정하여 S1 전체 예산을 산출**하는 것이다. LR 선택·성능 평가를 수행하지 않는다. TEST 접근, scientific fits, full cache는0이다.

```text
승인할 최대값
GPU job 누적 wall                600초 (10분)
CPU-only job 누적 wall           600초 (10분)
CPU process time                 600초 (전체 자식 포함 별도 기록)
작업 시작→종료 elapsed wall     1800초 (30분)
새 disk                         256MiB
process peak RAM                  8GiB
GPU allocated/reserved 각         8GiB
폐기용 technical optimizer       384 updates
full-prefix batch 경로             2000 calls (B<=4)
head/output-only batch 경로        8000 calls (B<=4)
```

3자료×2arm×2경로(online/cached)에 각각 warmup10+measured20=360 updates. gradient/restore integrity reserve는 최대24 updates, 총384다. fixed chronological TRAIN 첫24와 각 자료 canonical VAL 전체30/30/371개를 sample cache로 사용한다. cache origins54/54/395개, raw h/norm+F0 payload는22,288,920bytes다. 전체 VAL online/cached 평가 시간을 직접 측정하기 위한 범위이며 전체 TRAIN cache는 만들지 않는다. Adam 상태가 생긴 이후의 timing·peak memory를 측정한다. disposable 초기 상태 사본만 갱신하고 scientific checkpoint·선택 state에는 쓰지 않는다.

초기 parity, cache 값/loss/gradient, 결측 B4/micro gradients, frozen buffers, dropout/allowlist, target perturbation, save/restore gate를 이 범위에서 검사한다. full canonical VAL 초기 parity와 cache 평가 모두 전체431 origins에 적용하며 모든 시간을 위 cap에 포함한다. sample cache 생성은503 origin forecasts, B4 이하127 prefix batches를1회만 수행한다. 비용 평가의 full VAL은 각 dataset/arm/path당 warmup1+measured3 passes, 별도 초기 F0 reference는 각 자료1pass다. 부가 parity/교란/micro/restore forwards도 위2000/8000 counter cap에 모두 포함한다. 고정 probe LR1e-4로 동일 초기 사본을 사용하며 LR/성능 선택을 수행하지 않는다. 비용보정 sample cache는 수치·gradient 검증 뒤 S1 전체 cache의 부분 생성비로 계상해 중복 생성비를 피한다.

실패는 STOP_DEBUG, 자원 상한 초과는 STOP_RESOURCE다. tolerance를 늘려 통과하거나 scope를 몰래 줄이거나 cap을 자동 증액하지 않는다. probe cap은 실행 허가와 함께 authorization.json에 기록한 뒤만 사용한다. 지금 값은 **요청값**이다.

probe 이후 S1의 cached step, online step, VAL, generation, I/O를 별도로 수치화한다. 최대epoch projection은 조기정지 가정 없이 계산한다.

```text
cached fit work =
  Σ자료 4 × [
    15360 × HEAD_cached_step + 121 × HEAD_full_VAL
  + 15360 × OUTPUT_cached_step + 121 × OUTPUT_full_VAL
  ] + checkpoint/selection I/O

online fit projection = 같은 update/VAL 수에 online 경로 단가 적용

총 적응비용 = feature generation/write + 모든 cached fit work
            + fit work 밖의 LR-selection metadata I/O + 모든 probes
# fit work에 VAL/cache-read/checkpoint I/O가 이미 포함되면 다시 더하지 않는다.
총 campaign = 총 적응비용 + TEST + matched deployment cost + 검산/보고
```

원시 실측 단가, 느린 measured block 기준의 보수적 projection와 margin, CPU process/job wall, total operation wall, peak RAM/VRAM, disk를 제시한 후 **S1 전체 실행·시간 상한**을 승인받는다. 현재 S1 시간 cap은 미승인/null이고24 fits를 자동 시작할 수 없다. 옛 E/D 시간을 HEAD+Adam 실측이나 새 speed ratio로 바꾸지 않는다.

기존 비용 CSV4개의 실제 측정값·행수·source hash도 budget_estimate.json에 context-only로 목록화했다. v15 일부 비용은 원래 campaign 미완료 상태를 유지하며 나머지 회차의 완성도와 혼합하지 않는다. 이 값들은 S1의 새 측정 scope/ratio를 대신하지 않는다.

## 승인 뒤 구현·검증 작업

미실행 파일은 만들지 않았다. 아래 파일은 승인 뒤 새 campaign root 안에만 생성하는 실행 코드 역할이다. 기존 연구 코드를 수정하지 않는다.

### 작업1: 함수와 기술 gate

Files: `model_readout.py`, `runtime_readout.py`, `preflight_readout.py`.

Interfaces:
- `frozen_prefix(x) -> (h, loc_scale)`: canonical [B,512,C]만 입력.
- `native_point(h, loc_scale) -> [B,48,C]`: original block/inverse/select.
- `output_point(y0) -> [B,48,C]`: W/beta only.
- `backward_effective_batch(..., channel_count)`: B4 denominator, one clip/step.
- `enumerate_state(module)`: persistent/nonpersistent buffers 포함.

- [ ] source/데이터 hash를 검사하고 model allowlist6/head 또는 W/beta를 실제 enumeration한다.
- [ ] full VAL step0=F0; atol1e-5/rtol1e-4, 저장 array replay atol1e-6/rtol0의 부모 계약을 적용한다.
- [ ] shape/ordering/q index/normalization, singleton/batch/permutation, 채널독립, 미래 target 교란 불변을 검사한다.
- [ ] online/cache 예측·loss·head gradients 및 결측 effective_B4/micro4/2/1 gradients를 같은 초기 사본에서 비교한다. prediction/features/norm atol1e-5/rtol1e-4, parameter별 gradient atol1e-6/rtol1e-4, scalar loss abs差<=1e-6+1e-4*abs(reference_loss)를 고정한다. gradient/loss 수치는 v18 model_v18.py:16와 preflight_model_v18.py:118–123에 연결하며, 수치 결과를 보고 완화하지 않는다.
- [ ] finite loss/gradient와 실제 disposable update, frozen parameters/all buffers, optimizer/scheduler/RNG reload를 검증한다.
- [ ] 모든 통과·실패·실제 update/cost를 기록한다. PASS는 scientific effect·독립 재현·신규성 판정이 아니다.

### 작업2: S1 예산 확정과 캐시 봉인

Files: `prepare_readout.py`, `cache_readout.py`, 실행 후 `protocol.json`, `cache_manifest.json`, `preflight_checks.json`.

- [ ] 승인된 probe만 실행해 budget_estimate.json을 실측으로 갱신한다. 현재 source-only/native gate를 PASS로 미리 쓰지 않는다.
- [ ] S1 전체 cap 승인과 source/protocol/cache origin seal 이후 TRAIN-union·VAL 캐시를 생성한다.
- [ ] features/target 역할, cache hash와 online gradient parity를 검증한다.
- [ ] 모든 scientific fits를 시작하기 전 예산의 전체120epoch projection을 확인한다. 남은 cap이 계획을 감당하지 못하면 STOP_RESOURCE로 보고한다.

### 작업3: 적합과 TEST 전 joint 선택

Files: `train_readout.py`, 실행 후 `runs/<id>/result.json`, curves, `selection.json`, `selection_seal.json`, `exposure.json`.

- [ ] exact reuse가 추가 발견되면 계약 증거를 검증하고 fit 수에서 차감한다. 코드만 같다는 이유로 완료를 가정하지 않는다.
- [ ] 2arm×3자료×2LR×2seed 최대24 fits, 각128updates/epoch 이하, 총368640updates 이내로 실행한다.
- [ ] epoch0 포함 전체 VAL, strict earliest checkpoint, seed 손실 평균 LR 선택을 적용한다.
- [ ] epoch0 선택·상한 도달·비유한 gradient·TRAIN/VAL 반전·정상 악화를 분리한다. 결과를 보고 LR/epoch/head 구조를 추가하지 않는다.
- [ ] 모든 자료/arm의 LR/checkpoint/hash와 bootstrap/비용 protocol, 가능한 S2 사전 권고 초안을 **새 TEST 접근 전에** joint seal한다.

### 작업4: 정확도·동일 회차 비용·보고

Files: `evaluate_readout.py`, `cost_readout.py`, `verify_readout.py`, `report_readout.py`. 생성된 결과만 accuracy CSV/prediction receipts/cost rows/final checks/report에 기록한다.

- [ ] 선택된 LR의 두 seeds만 TEST-A/B를 예측한다. 선택 모델 최대12개이며 기간2개를 독립24fits로 세지 않는다.
- [ ] canonical observed mask의 MSE primary/MAE secondary, A/B channel sums/count를 pooled 뒤 macro, seed **손실 평균**을 계산한다. 예측 ensemble과 자료 간 raw MSE 총점은 없다.
- [ ] 기존 비교는 동일 checkpoint/origin/mask/좌표 증거로 재사용한다. 새 모델과의 paired 차이는 origin/channel/seed/기간 pairing을 보존한다.
- [ ] bootstrap2000, seed9262026, A/B stratified blocks Robin/Peacock7, Jena42, 기존 dataset salts0/0/1을 사용한다. `[9262026,salt,period_index]`의 moving contiguous noncircular blocks, ceil(n/block)개 draw 후 truncate, A/B weights 합산을 source와 연결해 봉인한다.
- [ ] total adaptation의 generation/cache-I/O/전체 LR·seed fits/VAL·선택비를 분리한다. cached fit과 online training step의 별도 표를 만든다.
- [ ] training probe는 선택 checkpoint의 폐기용 사본으로 warmup10/measured10×3blocks, Adam 상태 생성 뒤 full forward/loss/backward/optimizer.step을 측정한다. 새 두 arm의 selected12 instances×2paths=1440updates, 기존 MSE-LoRA/LEVEL 각2seed×3자료 online=720updates, 기본2160 technical updates는 별도 S1 예산에 포함한다. selected checkpoint pre/post hash는 동일해야 한다. 캐시 적합 중 prefix를 GPU에서 내렸다면 그 memory는 cached fitting에만 해당하며 배포 모델 메모리 감소로 해석하지 않는다.
- [ ] deployment는 CPU standardized x→전체 backbone+arm→CPU contiguous[48,C] 경로. B1/B4, VAL 첫24origins, warmup10, complete24origin passes10, rotated3blocks, F0 sentinels. HEAD/OUTPUT/F0/기존 MSE-LoRA/LEVEL만 같은 회차 실용 비용 비교에 포함한다.
- [ ] OOM·실패 row·sentinel drift를 보존한다. training microbatch와 deployment B4 조건을 혼동하지 않는다. block range는 CI가 아니다.
- [ ] 채점/분모/relative change/선택 seal/파일·hash/행수/자원/counter를 검산하고 보고한다.
- [ ] 관련 작은 결과·코드만 개별 staging/ordinary commit/push, live remote SHA 확인. raw data/cache/weights/checkpoints/무관 dirty/PPT/PDF는 게시하지 않는다. S0 계획 게시와 S1 결과 게시의 상태를 구분한다.

## S1 종료와 다음 승인

보고 순서는 문제→학습 경계/정보/혼입→기간·seed 정확도/정상성→비용 scope→가장 강한 반론→S2 권고다. OUTPUT/HEAD 각각 F0 변화, 두 경로의 절충, 기존 LoRA와의 참조 격차를 보고하되 pure internal gain이라고 하지 않는다.

S2는 S1의 같은 dataset/seed selected HEAD를 byte-identical 부모로 복사해 HEAD_CONTINUE와 HEAD_PLUS_LORA를 비교하는 별도 승인 단계다. LoRA 초기 수정0, head 초기값/입력순서/추가 update/LR 탐색/optimizer 재시작 규칙을 맞춘다. LoRA 래핑 후 head freeze 실수를 검사하고, 같은 update 성능과 실제 time/memory를 분리한다. 기본3자료 모두이며 결과가 나쁜 자료만 제외하지 않는다.

S1 정상 음성/혼합은 HEAD 정보 부재, 동등성, 내부 적응 불필요/필수의 증명이 아니다. S2 이득이 남고 대상별 차이·비용 차이가 있어야 S3/S4의 저비용 진단·경로 선택을 검토한다. LogME/Time-PEFT complexity/짧은 pilot보다 실제 decision risk·총비용이 나은지는 별도 미래 held-out 계약에서 검증해야 한다. 겹친 origins를 독립 도메인처럼 세지 않는다.

현재 S2 권고는 **NOT_JUSTIFIED_YET**이다. S1 결과·학습 정상성·비용 없이 실행하지 않는다. 유지할 것은 질문과 검증 계약, 보류할 것은 신규성·선택기·내부 적응 필수성·PCA/G/압축·배포 우위 주장이다.
