# Matched readout 실행 전 소스·계약 검토

상태: **SOURCE_REVIEW_ONLY / 실제 자료·solver·fixture 미실행**. [새 계획](../PLAN_MATCHED_READOUT_INFORMATION_KO_v1.md), [기존 native driver](../run_stage_native_ridge_v1.py), [bridge loader](../run_condition_probes.py), [원 input 생성 소스](../prepare_inputs.py), [기존 E33K 수학](../stage_separation_math_v1.py)을 읽었다. 실제 점수 JSON, NPZ, checkpoint, 모델, 데이터는 읽지 않았다. 기존 결과·실행 소스·보고서를 수정하지 않았다. 이 문서는 새 승인이나 실행 manifest를 대신하지 않는다.

[판단] 계획의 질문은 적절하다. 다만 “같은 readout”을 만들려면 같은 데이터·목표·산술 순서·선택 규약을 먼저 고정해야 한다. 입력 차원 차이는 남기되, 아래 세 항목은 실행 전 method lock과 실제 감사에서 다뤄야 한다.

## 1. RAW33C와 E512K의 원본 연결을 강제해야 한다

[확인·소스] 기존 native driver가 저장하는 `INPUT_<split>`은 D2의 `ABSOLUTE_LATENTS_<split>`에서 복사한 payload다. `latent_past`, `history_observed`, `raw_last`, `target`, `observed`, `origins`, `mean/std` 등을 가지지만 normalized 원 C의 전체 `history`를 저장하지 않는다. 따라서 RAW33C를 old INPUT의 `raw_last`나 latent에서 역산할 수 없다.

RAW33C의 올바른 원본은 `probes.load_bridge`가 읽는 원 bridge의 `history`: normalized FP32 `[N,512,C]`다. 원 input 생성은 각 창 시작에서 FIT 평균으로 carry를 초기화하는 causal fill 및 원 stored FP32 통계를 사용한다. Beijing EVAL은 원 repair의 저장 normalized history를 유지한다. 신규 경로가 원 physical 값에서 다른 fill을 적용하거나 새 통계를 추정하면 matched 비교가 아니다. `raw_last`는 이름과 달리 normalized current-C anchor이며, physical current 값으로 대체하면 안 된다.

실제 loader는 다음을 모두 강제해야 한다.

- 새 승인 runtime과 protected-TEST guard를 **파일 읽기 전에** 설치한다. 기존 bridge의 phase barrier 검사는 계보 검사용으로 재사용하되, 이전 phase의 만료 runtime이나 실행 큐를 시작하지 않는다.
- bridge JSON 및 artifact SHA, native producer/audit의 COMPLETE/PASS와 dataset·seed·split 역할을 검증한다. native 감사가 같은 producer SHA를 대상으로 했는지 결합한다.
- old INPUT과 bridge에서 `target`(dtype·shape·NaN 포함), `observed`, `origins`, `raw_last`, `history_observed`, `raw_forecast`, `mean/std`를 exact 대조한다. `raw_last == history[:,-1]`도 유지한다. 통계는 canonical 원 통계이며 재학습하지 않는다.
- E512K는 같은 old INPUT의 FP32 `latent_past[N,512,K]`만 펼친다. `z_last == latent_past[:,-1:]`, source CP·D2·latent artifact SHA와 원 codec의 결합을 검증한다. 같은 CP의 latent라는 기존 계보를 사용하는 것이며, 새 E 순전파를 독립 검증한 것으로 말하지 않는다.
- RAW33C의 32개 block mean 뒤 last를 붙이는 flatten 순서를 E33K와 맞춘다. 모든 신규 feature/scaling은 FIT의 past에서만 만든다. `z_future`, oracle teacher, query future target/mask는 deployment feature에 들어가지 않는다.

현재 source의 `probes.load_bridge`는 `history`를 반환하고 stats·last equality를 검사하므로 필요한 원본 API는 있다. bridge loader 자체만으로 native INPUT·계수·선택의 일치까지 증명되는 것은 아니다.

## 2. 재사용 N2와 신규 목표의 FP32 순서를 맞춰야 한다

[확인·소스] 기존 `prepare_n2`는 cached FP32 `target - raw_last[:,None]`를 먼저 계산하고, `prepare_masked_ridge`에서 이 innovation을 FP64로 바꾼다. `fixed_past_features`도 FP32 block mean을 먼저 구한 뒤 FIT center/scale을 FP64로 추정한다. 출력은 FP64 ridge innovation을 FP32로 cast한 뒤 FP32 anchor를 더한다.

반면 기존 `fit_intercept_only`는 Y와 last를 **먼저 FP64로** cast한 뒤 빼고, 평균 innovation을 FP32로 cast한다. 같은 수학식처럼 보여도 기존 N2와 operation order가 같지 않다. 신규 RAW33C/E512K는 **FP32 residual → FP64 ridge → FP32 innovation → FP32 anchor addition**을 고정해야 old N2를 동일 목표로 재사용할 수 있다. 새 feature block mean의 dtype과 flatten order도 고정해야 한다. 기존 FIT-intercept는 보존된 대조로 표시하고, FP32 residual recipe의 exact zero-feature equivalence를 증명하는 객체로 취급하지 않는다.

N2 점수만 복사하는 재사용은 불충분하다. 실제 재사용 감사에는 다음이 필요하다.

- original N2 family COMPLETE, 고정 순서의 7 alpha, actual-Y TUNE MSE·earliest exact tie 선택 및 원 coefficient artifact SHA 검증.
- feature center/scale, FIT raw33K feature SHA/행 수, solution kind·with-intercept·alpha·feature-state SHA·model SHA를 복원하여 검증.
- 모든 후보의 FIT/TUNE 예측·score 및 선택된 all-split prediction/target/mask/origins/stats/anchor의 계보 검증. 최소한 선택 계수의 past-only 예측을 원 selected artifact와 고정 tolerance로 대조하고, 저장 TUNE grid의 선택을 독립 계산한다.
- 재사용된 모델을 새 fit/새 seed 결과로 세지 않는다. 이전 학습 비용은 원 기록의 값 또는 미확인으로 남기고, 현재 hash/load/검산/추론 비용은 별도로 측정한다.

N2만 재사용할 때 기존 `load_selected_models`는 N1/N3와 CPU decoder도 함께 복원한다. 새 경로에 이들 객체가 필요하지 않으면 N2의 FeatureState·RidgeSolution만 복원하는 제한된 adapter가 더 명확하다. 기존 frozen 함수를 수정할 필요는 없다.

## 3. 고차원 solver는 목적뿐 아니라 alpha0 정의도 유지해야 한다

[확인·소스] 기존 ridge는 identical mask별로 `lstsq`와 thin SVD를 각각 수행하고 7 alpha에 factorization을 공유한다. 목표는 active channel별 전체 NH 오차를 그 채널의 total count로 나눈 macro MSE이며, penalty는 `alpha * sum_h ||W_hc_scaled||²`다. 따라서 각 output의 normal equation에는 **lambda = alpha × N_C(channel-total NH count)**를 사용한다. horizon별 `n_hc`로 바꾸면 같은 mask의 목표도 달라진다.

E512K의 p는 Robin 2048, Beijing 1024이고 FIT n은 256이다. 중심화된 mask group의 rank는 최대 `n_observed-1`이며 p>n이다. alpha0은 단순 역행렬이 아니라 원 centered design의 minimum-norm solution이고 intercept는 penalize하지 않는다. 양의 alpha의 condition 진단도 관측 singular값만의 최소값이 아니라 p차원 nullspace를 포함해야 한다. zero-observed output은 innovation/intercept0과 `NOT_ESTIMATED`를 유지한다.

[확인·구현 담당자의 준비 범위] 담당자는 Gram EVD 대신 centered design의 **단일 direct thin SVD/group**를 준비하고, alpha0 cutoff를 `float64 eps * max(n,p) * s_max`로 두며 기존 `PreparedRidge.fit` 목적을 유지한다고 전달했다. Gram EVD의 squared-condition 수치 손실을 피하는 선택이다. 새 소스는 준비 후 별도로 읽어야 하며 이 메시지를 fixture PASS로 바꾸어 말하지 않는다.

승인 이후의 synthetic 등가 fixture는 p>n·rank deficiency·ill-conditioned spectrum·masked NaN·zero count/rank·불균등 horizon count·unpenalized intercept·alpha0 minimum norm·FP32 최종 forecast를 포함해야 한다. 작은 행렬에서 기존 `lstsq`/원 수학과 비교하고 원 solver를 대규모로 중복 실행하지 않는다. 7 alpha마다 factorization을 다시 하는 구현은 피한다.

Beijing의 actual-Y mask가 output별로 다르므로 mask group 수가 실행 비용에 영향을 준다. 그룹 수, p/n/rank, 준비와 apply/save wall 및 process receipt를 남긴다. CPU900초 안에 완료된다고 소스만으로 보장할 수 없으며, cap 도달 시 실제 partial 상태와 재개에 필요한 자원을 남긴다. 새 budget에는 source fixture·독립 감사·보고 비용도 포함해야 한다.

## 허용되는 해석과 남는 교란

고정 차원은 다음과 같다. 계수 수는 intercept를 제외한 `p×48×C`이며 유효 rank와 관측 output 수는 실제 승인 후 따로 기록한다.

```text
조건                  Robin p   Beijing p
E33K 재사용 N2             132          66
RAW33C                    561         363
E512K                    2048        1024
```

동일 ridge recipe는 동일 계수 수·유효 자유도·effective regularization을 뜻하지 않는다. 같은 alpha에서도 입력 p와 상관구조가 달라 prior와 overfitting 여지가 다르다. E512K 개선은 “시간 요약을 없앤 효과 + readout capacity 변화”이고, RAW33C 대비 E33K 차이는 “현재 E+요약 경로”의 조건부 비교다. 새 계획이 이 제한을 이미 명시한 것은 적절하다.

RAW33C는 block마다 원 C의 평균을 직접 보지만 E33K는 causal E의 temporal filtering 후 평균을 본다. 두 경로는 같은 512 past를 사용해도 effective temporal feature가 같다거나 순수 선형 channel projection만 다르다고 하면 안 된다. 모든 경로는 current-C anchor를 받으므로 strict K-only deployment가 아니다.

두 seed는 같은 RAW past·target을 재사용하며 서로 다른 E checkpoint를 비교한다. RAW33C의 deterministic recipe가 두 seed에서 같아질 수 있고, 이를 독립 데이터 검증 두 번으로 세지 않는다. TUNE 선택과 반복 노출 VAL/EVAL은 유지하되, 새 method/alpha를 평가 점수를 보고 고르지 않는다. 새 결과가 나빠도 대조학습·동적 그래프의 필요성을 자동으로 결론내리지 않는다.

현재 판정: **데이터 API와 목적을 맞출 소스 경로는 있으나, 실제 artifact/계수 재사용·고차원 solver의 수치 등가·비용은 아직 미검증이다.** 위 계약을 실행 전 lock과 실제 감사에 넣고, 새 human 승인 및 fresh 실행 manifest가 유효한 뒤 독립 경로를 진행할 수 있다.

## 새 math·runtime 소스 후속 검토

[확인·소스] [새 math](../matched_readout_math_v1.py) SHA `304b5ae9bf5fa1cc3957cf14c17bea4d2f24fd9c87b00e271e1fc77d01c0088f`, [12개 fixture 정의](../test_matched_readout_math_v1.py) SHA `0ac15ea0426a7dabe7cc0b717b7b8d4abcc698a0295133aff73387f155f28bb5`, [새 runtime](../matched_readout_runtime_v1.py) SHA `1729067cf8bc9c929eab56fb358dc9187627afa502e9fec5a39d770252234664`의 실제 소스를 읽었다. 이 SHA는 검토 시점의 source이며 이후 수정 또는 실행 고정 여부를 추정하지 않는다.

새 math는 RAW33C의 FP32 block mean, E512K의 FP32 flatten, FIT-only FP64 center/scale, FP32 label subtraction, FP64 solver, FP32 innovation/anchor addition 순서를 구현한다. 동일-mask마다 direct thin SVD를 한 번 하고 기존 `stage.PreparedRidge.fit`을 재사용해 7 alpha의 lambda/count/intercept/zero-output 정의를 유지한다. p>n의 alpha0 minimum-norm cutoff도 명시돼 있다. 공개 `ReadoutModel.forecast(past, raw_last)`에는 future/target/mask 인수가 없고, 복원은 계수와 FeatureState SHA를 확인한다. 현재 이 흐름에서 실행을 막을 수학 오류는 찾지 않았다. **fixture PASS나 실제 N2 parity를 확인한 판정은 아니다.**

12개 fixture 정의는 p>n random design의 old/primal/dual 비교, rank deficiency, 불균등 count, masked NaN/zero output, FIT feature SHA, FP32 subtraction, no-future public API, model restore, selection tie를 다룬다. alpha0 cutoff를 추가로 검증하려면 ill-conditioned centered design에서 singular값이 threshold보다 확실히 위/아래에 있는 작은 fixture를 원 `lstsq`와 비교하는 것이 좋다. 경계의 exact floating tie를 테스트 기대값으로 강요하지 않는다. 이 권고를 구현 담당자에게 전달했고 fixture를 실행하거나 다른 소스를 수정하지 않았다.

새 runtime은 fresh matched-readout authorization/manifest, plan/method-lock/runtime SHA, human response origin과 고정 expiry를 요구한다. GPU0/CPU900초/wall1800초/growth1GiB를 상한으로 두고, 원 receipt 보존 목록의 비어있지 않음·SHA와 이전/current active lease를 검사한다. 실제 receipt는 승인 및 manifest SHA에 결합하며 process creation부터 charge한다. CUDA 초기화나 명시적인 `flag_forbidden_access`를 오류로 처리한다. 실제 F 접근 차단은 runner가 F/engine을 import·open하지 않는 소스 경로와 함께 확인해야 하며, `F_access_allowed:false` 메타데이터 자체가 모델 파일을 읽지 않았다는 증거는 아니다.

검토 시 신규 runner/method lock은 아직 읽지 않았으므로 위 데이터 alignment·N2 재사용·불필요한 CP/F 접근 배제의 **연결 gate는 남아 있다**. 실제 승인 기록이나 asset 존재를 읽고 판단하지 않았다.

## 신규 runner·method lock 연결 검토

[확인·소스] 이후 [runner](../run_matched_readout_v1.py) SHA `622f312cff70613e4e353bfb1c97c003f00f4fbbc950a80b1cbebe268a88ad17`, [passed-array contract](../matched_readout_contract_v1.py) SHA `68831551d35cc56adfc95ca54302681d77f88fede6743a60e1f0ba0222c78302`, [method lock](../METHOD_LOCK_MATCHED_READOUT_v1.md) SHA `ff4da28461baa6d641f6d0460dc9154b35fffa774286c242a1776dcc20f82cf0`를 읽었다. 작성 중인 새 소스이므로 이 검토는 아직 실행 고정 버전의 승인 판정이 아니다.

확인한 적절한 연결은 다음과 같다.

- 새 runtime 이후 source imports 및 protected guard를 설치하고, RAW33C는 bridge history로 먼저 실행한다. missing cached E가 유효한 RAW 경로를 취소하지 않는다.
- 새 목표·선택은 same actual-Y mask 및 원 FP32 residual recipe를 사용한다. 7 alpha의 FIT/TUNE 저장과 선택 모델의 VAL/EVAL 저장 흐름이 구분돼 있다.
- 기존 N2만 FeatureState/RidgeSolution으로 복원하며 E/D/F checkpoint를 deserialize하지 않는다. `frozen_decoder_refit` import는 기존 Torch CPU score 구현을 사용하기 위한 것이고, 그 import 자체에서 CUDA engine/F를 생성하는 source 흐름은 찾지 않았다. GPU/F 미사용 실제 판정은 runtime/실행 기록과 별도다.
- `MAIN_checkpoint.selected_update`는 원 `probes.load_main_codec`가 실제 생성하는 record 필드다. `selected_codec_hash_old_algorithm`, `selected_codec_hash_bi_algorithm`, `architecture`, `K`, `original_prediction_batch_size`도 그 record에 존재한다. 해당 필드명을 추정 오류로 수정할 필요는 없다.

**실행 전 보완할 연결 gate 세 가지**를 ROOT에 전달했다.

1. **원 raw history와 cached E의 생성 계보.** 공통 target/mask/origin/anchor를 맞추는 것은 필요하지만 latent가 동일 raw history에서 생성됐다는 증명은 아니다. `stage.D2_source`를 SHA 검증하고 D2의 `bridge_provenance.input_bridge` 및 `input_artifacts`를 현재 bridge와 묶어야 한다. 현재 원 D2 source의 필드명은 `bridge_provenance`이며 `input_provenance`가 아니다. D2의 `ABSOLUTE_LATENTS_<split>` declared array metadata/SHA와 stage가 복사한 `INPUT_<split>`의 `latent_past/z_last/common arrays`를 결합하면 재생성 없이 provenance를 닫을 수 있다. 독립 E forward 검증으로 확대하지 않는다. 기존 native PASS audit가 같은 4 producer SHA를 검증했다는 binding도 새 manifest/실제 재사용 gate에 필요하다.
2. **재사용 FIT feature와 계수 선택의 closure.** FeatureState/solution/model SHA 및 selected prediction parity 구현은 적절하다. 다만 복원 직후 `feature_state.transform(cached_fit_latent, require_fit_artifact=True)`로 원 FIT33K raw-feature SHA와 actual cached FIT를 직접 묶는 검사가 추가로 필요하다. 후보 grid의 metadata min/tie만 믿지 않고, 원 감사의 producer SHA binding 및 coefficient/prediction artifact SHA를 확인한다. 기존 N2를 새로 학습할 필요는 없다.
3. **N2 재사용 실패와 E512K 독립성.** 검토 버전은 N2 coefficient restore·selected parity 뒤에 E512K fit을 호출한다. cached E의 계보가 유효하지만 N2 coefficient만 누락되면 신규 E512K도 건너뛰게 된다. 유효 E cache 검증 후 E512K를 재사용 branch와 분리하면 독립 경로 계속 원칙을 충족한다. RAW33C 독립 실행은 이미 구현돼 있다.

초기 raw-only contract의 `past_arrays_exactly_matched/target_mask_anchor_stats_exactly_matched=True`는 구조·유한성 검사가 끝났다는 뜻과 cached E exact 대조를 혼동시킬 수 있다. raw validation과 cached matching 증거를 별도 flags로 두어야 한다는 ROOT의 수정 예정 사항을 확인했다. mean/std FP32 dtype도 contract에 명시하면 method lock과 맞는다. 실제 통계가 틀렸다고 발견한 것은 아니다.

후속 math fixture source는 SHA `c43a54345d5dc0931a9d5312d6be6b146eb59a74285f056896e6ec6d48cfc4bb`로 갱신됐고 13개 정의를 가진다. 새 cutoff fixture는 centered disjoint signed pairs의 p11/n6 design과 singular spectrum `1, 128×cutoff, cutoff/128`에서 원 lstsq와 rank2/minimum-norm/prediction을 비교한다. 권고한 source 검산 범위를 추가한 것을 읽었으며 실행하지 않았다.

신규 실제 score JSON/NPZ/CP/모델/data를 읽거나 solver·fixture·AST 검사·기존 실험 재개·Git 작업을 수행하지 않았다. 위 gate가 새 소스에서 보완되면 다시 source만 확인할 수 있고, 실제 연결·수치·비용 통과는 새 승인 후 기록으로 판정해야 한다.
