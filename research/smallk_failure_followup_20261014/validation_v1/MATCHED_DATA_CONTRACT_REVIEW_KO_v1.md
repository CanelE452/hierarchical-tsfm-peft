# Matched readout 데이터 계약 소스 검토

상태: SOURCE_ONLY_REVIEW / 실제 자산 미검산 / solver 미실행.

검토 범위는 `PLAN_MATCHED_READOUT_INFORMATION_KO_v1.md`, `METHOD_LOCK_MATCHED_READOUT_v1.md`, `matched_readout_contract_v1.py`, `matched_readout_math_v1.py`, `run_stage_native_ridge_v1.py`, `run_condition_probes.py`, `prepare_inputs.py`, 그리고 `_docs/history/2026-10-14.md`의 메타 히스토리다. `run_matched_readout_v1.py`는 아직 준비 전제로 보아 검토하지 않았다. 결과 JSON/NPZ/checkpoint/data/model 자산은 검산하지 않았고, 테스트/solver/GPU도 실행하지 않았다.

주의: 검토 중 한 번 `rg` 검색 범위가 넓어져 `validation_v1`/report 경로명과 일부 텍스트 결과 줄이 터미널에 출력되었다. 이 메모의 판단에는 해당 결과 수치나 결과 파일 내용을 쓰지 않았고, 바이너리 자산·checkpoint·NPZ·모델은 열지 않았다.

## 판단 맥락

- [확인] 도메인은 시계열 예측 모델의 고정 split 소표본 readout 비교다. downstream 결정은 RAW33C, E512K, 기존 E33K(N2 reuse)가 같은 target/mask/current-C anchor/FIT-only scaling/actual-Y TUNE 선택 조건에서 비교 가능한지 확인하는 것이다.
- [확인] 품질 기준은 publication에 가까운 사전 실행 계약이다. 누수 허용치는 0에 가깝고, 특히 시간순 split, future oracle 배제, old N2 lineage, target/mask/anchor 동일성이 핵심이다.
- [확인] 계획 문서는 Robin C17/K4, Beijing C11/K2, seeds 92601/92602, L512/H48, FIT256/TUNE64, Robin reused VAL30, Beijing exposed EVAL146을 고정한다. 새 경로는 RAW33C/E512K만 fit하고 old E33K는 기존 N2를 재사용한다.

## 강한 계약으로 보이는 부분

- [확인] `prepare_inputs.py`는 dataset spec에서 Robin `fit/tune/val = 256/64/30`, Beijing `fit/tune/eval = 256/64/146`, channel 수 17/11을 고정한다. 소스: `prepare_inputs.py:55-73`.
- [확인] input bridge의 split payload 검증은 `history`/`raw_forecast`/`target`/`raw_last`를 FP32, observed/history mask를 bool, origins를 int64로 제한하고, `raw_last == history[:, -1]`를 exact로 확인한다. observed target만 finite면 되고, unobserved target의 NaN은 허용된다. 소스: `prepare_inputs.py:316-361`.
- [확인] Robin FIT/TUNE/VAL history는 old `load_data`와 old `experiment.preprocess`에서 재구성하고, raw prediction bundle의 origins/target observed와 맞춘다. Beijing FIT/TUNE도 같은 방식이며, Beijing EVAL은 stored repaired normalized history를 쓰고 future target open 호출을 피한다. 소스: `prepare_inputs.py:534-579`, `prepare_inputs.py:617-680`.
- [확인] `run_condition_probes.py`의 bridge loader는 COMPLETE bridge, old NPZ/hash gate, Bi phase barrier, split coverage, split별 mean/std 동일성, `raw_last == history[:, -1]`, RAW forecast 원본 일치를 확인한다. 소스: `run_condition_probes.py:125-175`.
- [확인] D2 latent producer는 `values['history']`를 original codec encode에 넣어 `latent_past`, `z_future`, `z_last`, source GPU prediction을 만들고, stored MAIN prediction과 parity를 요구한다. 소스: `run_condition_probes.py:268-294`, `run_condition_probes.py:417-426`.
- [확인] matched readout math는 RAW33C/E512K route만 허용하고, 입력은 FP32 finite past만 받는다. RAW33C는 `N x 512 x C`를 `32 x 16` block mean + last C로 만들고, E512K는 `512K`를 시간순 flatten한다. 소스: `matched_readout_math_v1.py:23-35`.
- [확인] feature state는 FIT past에서만 center/scale을 추정하며 FP64 statistics, zero scale -> 1을 기록한다. transform은 fit artifact SHA를 선택적으로 강제할 수 있다. 소스: `matched_readout_math_v1.py:52-75`.
- [확인] target innovation은 FP32 `Y - last`를 먼저 계산한 뒤 masked ridge로 넘긴다. solver는 observed target 값만 finite로 요구하므로 unobserved NaN은 직접 학습에 들어가지 않는다. 소스: `matched_readout_math_v1.py:91-97`, `matched_readout_math_v1.py:201-214`.
- [확인] prediction API는 `past`와 `raw_last`만 받으며 target/future mask/oracle 인자를 받지 않는다. current-C anchor는 FP32 innovation 후 더한다. 소스: `matched_readout_math_v1.py:139-150`.
- [확인] capacity 해석용 metadata는 coefficient/intercept 수, observed output 수, zero-observed output 수, rank min/max, condition inf/nan count, route 간 capacity 차이를 기록한다. 소스: `matched_readout_math_v1.py:152-175`.

## Material issues

1. [확인] `validate_stage_reuse()`는 old N2 reuse의 필수 증명 중 일부만 확인한다.

   method lock은 COMPLETE stage result, exact old MAIN checkpoint/dataset/seed/state metadata, 실제 coefficient/feature state/selected prediction SHA 확인, seven alpha finite, earliest tie 선택, 저장 계수로 과거 예측 복원을 요구한다. 현재 함수는 stage status/dataset/seed, old E/D/F update flags, codec hash before/after/terminal, N2 alpha grid와 TUNE score finite, selected index/alpha/model/coefficient artifact pointer 일치까지만 본다. 소스: `matched_readout_contract_v1.py:43-62`.

   누락된 증명은 다음이다.

   - selected coefficient artifact 파일 SHA/content 검증
   - feature state artifact와 33K feature shape 검증
   - selected FIT/TUNE/VAL/EVAL prediction artifact SHA/content 검증
   - 저장 계수 restore 후 old N2 prediction 재생성
   - old N2의 selected model kind/feature count가 정말 E33K인지 확인

   따라서 현 상태에서 `validate_stage_reuse()` 통과만으로 “oldN2E33K reuse lineage가 완결됐다”고 쓰면 과장이다. `run_matched_readout_v1.py`가 별도 restore/replay 검산을 구현한다면 보완 가능하지만, 현재 검토한 계약 함수에는 없다.

2. [확인] raw history와 cached latent의 동일 입력 lineage는 소스 경로상 강하지만, 순수 계약 함수 하나로 닫히지는 않는다.

   `validate_split()`은 bridge `history`의 shape/dtype/finite, cached `latent_past`의 shape/dtype/finite, target/observed/origins/raw_last/history_observed/raw_forecast/mean/std exact match, `raw_last == history[:, -1]`, `z_last == latent[:, -1:]`를 확인한다. 소스: `matched_readout_contract_v1.py:14-40`.

   하지만 cached D2 payload에는 raw `history` 자체가 들어 있지 않으므로 `validate_split()`은 “이 latent가 이 exact history에서 encode되었다”를 직접 재생성하지 않는다. 그 증명은 D2 producer의 `extract_absolute_latents()`가 bridge history에서 latent를 만들고 source prediction parity를 저장했다는 producer lineage에 걸려 있다. 소스: `run_condition_probes.py:268-294`, `run_stage_native_ridge_v1.py:252-273`.

   따라서 최종 runner는 `validate_split()`만 호출하지 말고, D2 result identity/source snapshot/parity와 matched input bridge를 함께 묶어야 한다. 그렇지 않으면 “original normalized raw history vs old stage cached INPUT latents identity proof”가 절반짜리 증명으로 남는다.

3. [확인] old N2 reuse의 비용/자원 해석은 아직 결과 runner 수준에서 고정되지 않았다.

   plan/method lock은 old N2의 학습·미래표현 생성 비용을 신규 사용량으로 중복 집계하거나 0으로 추정하지 말라고 한다. 현재 검토 대상 중 `matched_readout_contract_v1.py`/`matched_readout_math_v1.py`에는 cost ledger나 reuse cost annotation이 없다. `run_stage_native_ridge_v1.py`에는 stage-native 실행의 resource receipt가 있지만, matched readout 신규 runner가 아직 없으므로 oldN2 reuse cost 표기가 어떻게 강제될지는 확인되지 않는다. 소스: `run_stage_native_ridge_v1.py:526-535`.

## Ambiguous or needs-runner-proof

- [추정] `prepare_thin_masked_ridge()`의 denominator 계산은 method lock의 `alpha * Nc` 목적과 맞게 설계된 것으로 보인다. channel count를 horizon별로 tile하고 mask group별 centered thin SVD를 만든다. 다만 실제 `RidgeSolution.fit(alpha)` 내부는 이번 허용 파일 목록 밖의 `stage_separation_math_v1.py`에 있으므로, 본 검토에서는 normal-equation penalty 등가성을 완결 확인하지 않았다. 소스: `matched_readout_math_v1.py:100-125`.
- [확인] new RAW33C/E512K는 target/future/oracle 없이 예측할 수 있는 API다. 다만 alpha 선택과 selected prediction 저장은 아직 `run_matched_readout_v1.py`가 없어서 runner 수준으로는 확인되지 않는다.
- [확인] unobserved target NaN은 학습에는 들어가지 않게 되어 있다. 다만 selected artifact가 target/observed를 저장할 때 unobserved NaN을 그대로 보존하는지는 matched runner 부재로 아직 확인할 수 없다.
- [확인] capacity/sample-mask 해석 문구는 metadata에 들어가 있지만, 보고서가 primary metric을 normalized channel-macro MSE로 고정하고 physical metric을 diagnostic으로만 쓰는지는 matched runner/report 부재로 확인되지 않는다.

## Source-only verdict

현재 소스는 matched readout의 핵심 데이터 계약을 대체로 올바른 방향으로 준비했다. 특히 원 input bridge, target/mask/origin/stat/anchor exact match, FIT-only scaling, observed-only ridge, future/oracle-free prediction API는 소스상 확인된다.

다만 oldN2 E33K reuse를 비교축으로 세우려면 `validate_stage_reuse()`와 최종 runner가 더 강해야 한다. 최소한 selected coefficient artifact와 feature state를 실제로 복원하고, old selected prediction을 재계산해 저장 artifact SHA/score와 맞추는 검산이 필요하다. 또한 D2 cached latent가 bridge history에서 온 것이라는 증명은 producer lineage와 묶어서 보고해야 하며, `validate_split()` 단독 통과를 “raw history와 cached latent identity proof”로 부르면 안 된다.

실행 전 권장 gate:

- old stage N2 result JSON의 source snapshot, D2 source, checkpoint/case metadata, N2 feature state, selected coefficient artifact, selected split artifacts를 모두 SHA 검증한다.
- selected old N2 coefficient를 restore해서 FIT/TUNE/final split prediction과 normalized channel-macro MSE를 재생성한다.
- RAW33C/E512K runner에서 `validate_split()`을 bridge split과 D2 cached split 각각에 대해 호출하고, D2 producer parity/source snapshot도 같은 manifest에 묶는다.
- selected output artifact에는 prediction, target, observed, origins, mean/std, raw_last, feature/model SHA를 남긴다.
- 보고서에는 E33K reuse, RAW33C, E512K의 feature count/rank/observed output count/zero output count를 같은 표에 두고, capacity 차이 때문에 encoder 단독 효과를 주장하지 않는다고 명시한다.
