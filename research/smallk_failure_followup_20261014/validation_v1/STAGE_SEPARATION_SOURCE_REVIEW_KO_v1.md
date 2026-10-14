# 표현·잠재 예측기·복원 분리 계획의 독립 소스 검토

상태: **SOURCE_REVIEW_ONLY / 신규 실험 미실행**. 소스와 계획 문서만 읽었다. 실제 데이터·CP·NPZ·모델을 열거나 import하지 않았고 fixture·학습·추론을 실행하지 않았다. 자산 존재, 실제 coverage, 새 수치 결과, real-device parity는 미확인이다. 원 계획과 실행 완료된 소스·결과를 수정하지 않았다.

검토 대상은 [원 계획](../PLAN_STAGE_SEPARATION_KO_v1.md), [원 engine](../../smallk_predictive_alignment_20261013/engine.py), [원 controls](../../smallk_predictive_alignment_20261013/controls.py), [원 encoder](../../forecast_component_study_20261013/component_experiment.py), [기존 D2 helper](../frozen_decoder_refit.py), [D0 native 계보 검사](../run_saved_diagnostics.py), [동일 E/F 잠재 추출](../run_condition_probes.py)다. `_docs/history/2026-10-14.md`의 이전 단계 기록도 먼저 확인했다. 이 문서는 실행 승인이나 과학 검산 PASS가 아니다.

[판단] O0/O1, N1/N2, 동일 N1을 고정한 N3를 나누는 방향은 현재 실패의 위치를 좁히기에 적절하다. 원 계획은 oracle를 배포 순위에서 분리하고, N1↔N2의 target·출력 차원·표본 수 차이, N1↔F의 목표·capacity 차이, 원 E/D가 재구성 AE가 아니라 F를 통과한 예측 손실로 학습됐다는 점을 이미 제한하고 있다. 다음 세 항목을 새 method lock과 구현 계약에서 명확히 해야 한다.

## 1. ridge 목적의 분모는 개별 horizon count가 아니라 채널 전체 count다

[확인·수식] 주지표인 normalized channel-macro MSE에 맞춘 N2의 손실은 active channel 집합 A와 채널 전체 관측수 N_c에 대해 다음과 같다. 입력 feature를 FIT-only로 중심화·스케일한 것을 u, 각 horizon/output의 weight를 W_hc, unpenalized intercept를 b_hc라 둔다.

$$
N_c=\sum_{i,h}m_{ihc},\qquad
L=\frac{1}{|A|}\sum_{c\in A}\left[
\frac{\sum_{i,h}m_{ihc}(u_i^\top W_{hc}+b_{hc}-r_{ihc})^2}{N_c}
 +\alpha\sum_h\|W_{hc}\|_2^2\right].
$$

개별 (h,c) group의 normal equation에는 **lambda=alpha×N_c**가 들어간다. lambda=alpha×n_hc는 각 horizon을 별도로 평균한 다른 목적이다. 같은 alpha grid를 사용해도 estimator가 달라진다. alpha=0에서는 이 차이가 드러나지 않으므로 양의 alpha와 불균등 horizon 관측수가 있는 fixture가 필요하다.

소스 수준 반례: 채널 하나, 중심화한 scalar x=±1, y=x, h0 관측수4, h1 관측수2, N_c=6, alpha=1이면 위 목적의 두 slope는 4/(4+6)=0.4, 2/(2+6)=0.25다. 개별 horizon mean 방식은 둘 다0.5가 된다. 이는 실제 데이터 실험 결과가 아니라 목적함수의 차이를 보이는 대수적 예다.

N1에도 같은 형태를 적용하되 output q는 latent K이고 mask는 K에 동일하게 확장한 complete-future-prefix mask다. **N1의 학습 target은 실제-Y가 아니라 원 E 좌표의 latent innovation**이다. actual-Y TUNE는 alpha 선택 지표이며 latent 학습 목적과 동일하다고 쓰면 안 된다. D2/O1/N3 decoder는 NH를 flatten한 채널별 문제이므로 기존 D2의 전체 관측수 분모와 일치한다.

[확인·소스] 기존 `fit_output_head`는 채널별 전체 NH observed mean error + alpha×scaled weight norm을 사용한다. no-intercept/FIT RMS feature rule을 새 predictor의 centered feature/unpenalized intercept와 혼동하지 않아야 한다. 원 D2를 고쳐 새 목적에 맞췄다고 쓰지 않는다.

권고 상태: ROOT와 math 담당자가 위 식과 불균등 count fixture를 새 immutable method lock에 명시하기로 수용했다. 새 구현·fixture의 실제 통과 여부는 아직 확인하지 않았다. zero-observed (h,q)는 innovation/bias0으로 미학습 표시하고, zero-observed decoder C는 기존 outgoing weight를 유지한다.

## 2. N2 우위만으로 E의 과거 feature가 도움이 됐다고 결론 내릴 수 없다

[확인·소스] N2는 33K feature 외에 raw current-C anchor와 horizon/channel intercept를 갖는다. 원 MAIN보다 N2가 좋더라도 현재값 지속 또는 FIT의 평균 미래 잔차가 개선을 설명할 수 있다. N1과 N2의 coefficient 수도 다르므로 두 방법의 차이를 순수 decoder 효과로 읽을 수 없다.

최소 대조는 같은 anchor와 Y-observed mask를 쓰는 다음 두 경로다.

- **CURRENT_C_PERSISTENCE:** 모든 미래 horizon의 예측은 x_t다. 신규 학습은 없다.
- **FIT_INTERCEPT_ONLY:** 각 (h,c)의 관측 FIT Y−x_t 평균을 x_t에 더한다. 관측수0이면 잔차0이다. alpha 선택은 없지만 **신규 coefficient 추정**이므로 비용·관측수·계수·실제 예측을 기록한다.

이 두 대조보다 N2가 낫다면, 같은 raw-current anchor와 평균 잔차를 넘어 추가 feature readout의 근거가 생긴다. 그래도 반복 개발 VAL/노출 EVAL·seed2개에서의 결과이며, 모든 E 정보가 충분하거나 지연 채널 구조가 해결됐다는 증거는 아니다. 추가 raw-C 과거512점을 N2에 넣으면 same-E feature 계약이 바뀐다.

권고 상태: ROOT가 위 두 고정 대조를 새 method lock에 포함하기로 수용했다. cached LAST와 같은지 실제 SHA/origin/stat/mask를 확인하기 전에는 재사용 가능으로 간주하지 않는다. 새 구현·실제 계수 추정은 미실행이다.

## 3. latent teacher coverage는 목표 정보보다 미래 관측 가능성을 조건으로 표본을 고른다

[확인·수식] complete-future-prefix mask는 `all_C(observed)`를 미래 horizon에 따라 누적 AND한 것이다. 따라서 h가 길수록 teacher count가 증가할 수 없다. 이 subset은 미래에 모든 채널을 관측한 경우에 조건부인 표본이며, 원 Y의 채널별 observed 표본과 다르다. 실제 두 데이터에서 얼마나 줄어드는지는 이번 검토에서 확인하지 않았다.

새 실행 gate/보고에는 FIT 각 h의 teacher count·design rank·conditioning, 모든 split의 complete-prefix count와 원 Y-observed count를 분리해 남겨야 한다. count0인 h의 persistence는 학습 성공이나 latent 예측 실패로 세지 않는다. 전체 teacher가0이면 N1 학습은 **NOT_ESTIMATED**이며 alpha tie를 선택했다고 실제 학습으로 바꾸어 쓰지 않는다. 부족한 coverage 때문에 N1 해석을 보류하더라도 원 관측 Y를 쓰는 N2와 고정 대조는 독립적으로 진행할 수 있다. 결과를 본 뒤 complete-prefix 조건을 완화하면 안 된다.

원 Y mask 점수와 complete-prefix subset 점수는 서로 다른 모집단이다. subset 점수를 full observed score와 직접 빼거나, count 변화 때문에 사라진 채널을 더 좋은 성능으로 읽으면 안 된다. count/rank가 작은 양성·음성 결과 모두 제한된 표본의 결과로 남긴다. 임의의 사후 coverage threshold로 좋은 방법만 선택하지 않는다.

추가 해석 한계: N3 calibration은 같은 FIT으로 학습한 N1의 **in-sample** latent predictions를 사용한다. FIT에서의 interpolation과 TUNE에서의 predictor error 분포가 다를 수 있다. N1 alpha를 oldD TUNE로 먼저 고정한 다음 N3 alpha를 TUNE로 고르는 순차 선택도 남는다. N1↔N3는 고정 predictor 아래 calibration 기여를 보는 비교지만, 모든 decoder의 최적 한계나 독립적인 일반화 검증은 아니다. 이 단계에 cross-fitting을 자동 추가하지 않고 해당 조건·FIT/TUNE 차이를 기록하는 것으로 범위를 유지한다.

## 구현에서 확인할 API·계보 계약

아래는 구현 검토와 승인 뒤 실제 검산에 필요한 항목이다. 현재 체크 완료를 뜻하지 않는다.

```text
항목                         필요한 계약
원조건                       Robin C17/K4, Beijing C11/K2 × old MAIN seed2 = 정확히4
동일 E/기존 D                원 CP identity·file SHA·전체 state·통계·origin·mask 결합
same-past features           N1/N2 동일33K·동일FIT mean/std·scale0→1·공유feature SHA
N3 predictor 동결            selected N1 계수/alpha/latent predictions SHA를 변경 없이 재사용
oracle/deployment 선택       O1 oracle 계수·TUNE oracle 점수를 N1/N2/N3 배포 선택에 전달 금지
zero-count output            native innovation0; decoder는 원 outgoing C weight 유지
기존 control 재사용          D0 NATIVE/D2 Y-refit/RAW의 실제artifact SHA와 조건 일치 후만 사용
보호 및 비용                 새 승인 runtime만 사용, old Limit/campaign 시작 금지, TEST 금지
```

[확인·소스] 원 encoder의 시간 convolution은 왼쪽 padding만 사용한다. 이것은 causal 구조의 소스 근거지만, 실제560점 실행의 prefix parity가 통과했다는 증거는 아니다. 새 oracle builder는 **이미 normalized·filled 과거512점과 동일 FIT 통계로 normalized한 observed 미래48점**을 이어야 한다. 미래를 두 번 정규화하거나 physical 미래를 normalized 과거에 이어도 prefix parity는 통과할 수 있으므로 미래 target의 단위·mask·SHA 계약도 따로 결합해야 한다.

Oracle와 배포 검사는 다음처럼 분리한다.

- 배포 predictor 함수는 past feature, current anchor, 고정 coefficient만 받는다. query future Y/mask를 바꿔도 배포 출력은 같아야 한다.
- oracle는 실제 observed 미래에 의존할 수 있다. 뒤 미래의 변경은 이전 latent에 영향을 주면 안 되고, unobserved 미래의 저장값 변경은 carry-forward 결과에 영향을 주면 안 된다.
- oracle가 모든 observed perturbation에 반드시 반응해야 한다고 요구하면 안 된다. E가 어떤 방향을 버리거나 constant로 매핑할 수 있다. 반응 witness가 없다는 진단과 causal 위반을 구분한다.
- oracle의 원 Y-observed score는 imputation이 섞인 latent도 포함한다. complete-prefix score/count를 함께 제시하고, oracle 성공을 past-only 예측 가능성·미래 누수 부재·보편적 decoder bound로 확대하지 않는다.

[확인·소스] 기존 `NativeShared`는 latent별 시간축512→48 가중치를 공유하고 decoded actual-Y로 학습한다. 이번 N1의 joint33K latent-target ridge와 다른 모델이다. 기존 native를 반복 학습하지 않고 조건 일치 후 결과를 재사용하는 계획은 중복 방지에 적절하다. 기존 `engine.oracle`의 fixed-linear projection은 nonlinear D에서 적용 불가하며 이번560점 true-future E oracle의 대체물이 아니다.

## 현재 판정

[판단] 새 진단은 진행할 가치가 있으나, 원인이 해결됐다고 전제하는 실험이 아니다. 위 목적함수와 고정 대조를 method lock에 명시하고, teacher coverage·API·계보를 검산하면 E의 제한된 feature readout, F 경로, predictor별 D calibration을 지금보다 좁게 비교할 수 있다. 실패해도 representation이나 temporal decoder의 보편적 불가능을 입증하지 않는다.

새 math/oracle module은 이 문서 최초 작성 시 아직 소스 준비 중이었다. 이후 받은 source readiness에 대한 검토는 아래에 덧붙였다. 자산/CP/NPZ/model을 읽거나 실제 fixture를 실행하지 않는다.

## 새 math API의 read-only 검토

[확인·소스] [stage_separation_math_v1.py](../stage_separation_math_v1.py) 전체와 [14개 fixture 소스](../test_stage_separation_math_v1.py)를 읽었다. 담당자가 전달한 source SHA는 각각 `d64316aed4976fdfed489d691759ea8d2312d06cc66ad250ed7f1b839bd41e2b`, `3b3f9fe022b6f0b6f4f3cb3a4e243aaa7e34ef858f3fefc4fc45a35e1dbf71a9`다. 이 SHA는 전달 당시 버전 식별이며 실제 계산/테스트 PASS가 아니다.

소스 흐름에서 다음을 확인했다.

- `FeatureState.transform(require_fit_artifact=True)`는 FIT raw33K feature SHA/row 수를 결합하고 N1/N2는 같은 state를 사용한다.
- `prepare_masked_ridge`의 N2 denominator는 채널 전체 NH count를 H회 tile하고, 동일 mask group의 SVD를 재사용하되 각 column에는 alpha×해당 채널 전체 count를 적용한다. 양의 alpha에서 intercept는 penalize하지 않고, alpha0은 mask group 중심화 후 weight minimum-norm과 intercept 복원을 수행한다.
- `prepare_n1`의 teacher mask는 complete-future-prefix를 K에 동일하게 확장한다. 관측수0인 horizon/output은 group을 만들지 않아 innovation/intercept0을 유지한다.
- `prepare_n3`의 **FIT 준비**는 제공된 N1 model로 past에서 latent를 다시 예측하고 N1 SHA를 저장한다. no-intercept decoder/RMS와 zero-observed C outgoing fallback도 구현돼 있다.
- current-C persistence 및 FIT-intercept-only 대조가 추가됐고, fixture에는 zero-feature N2가 intercept-only와 같아지는 조건, 불균등 horizon count, masked NaN, zero-rank/zero-count, earliest actual-Y TUNE selection이 포함된다. 이 fixture를 실행하지 않았다.

[판단] 위 대수·mask 흐름에서 현재 막아야 할 solver 오류는 찾지 않았다. 실제 numerical fixture와 old D/device parity가 통과했다는 판정은 아니다. 새 exact-erf GELU와 NumPy FP32 matmul은 기존 Torch CPU/CUDA의 operation order와 다를 수 있으므로 원 hidden/output의 고정 tolerance parity를 실제 단계에서 확인해야 한다. alpha0 rank-deficient condition의 infinity 및 zero-observed condition의 NaN은 배열 진단 값으로 정상일 수 있지만, JSON 보고 시 finite value로 위장하지 않고 null+reason 또는 별도 condition-status로 직렬화해야 한다.

**남은 integration gate:** `DecoderCalibration.forecast(z_future, ..., fixed_n1_model_sha256=...)`는 전달한 문자열이 N1 SHA와 같은지만 검사한다. 그 latent 자체가 해당 N1의 past-only 출력인지 강제하지 않는다. oracle latent와 올바른 문자열을 함께 넘기는 잘못된 wiring을 이 API만으로는 차단하지 못한다. 현재 `cannot_import_oracle` fixture도 wrong-hash 거부를 검사하며 이러한 latent 출처 검사는 하지 않는다.

실제 N3 배포 wrapper는 selected N1 model과 past/anchor만 받아 내부에서 `selected_N1.latent_forecast(past)`를 호출해야 한다. 또는 최소한 실제 runner의 생성 경로와 저장 N1 prediction SHA/배열 equality를 독립 검산해야 한다. O1/O0 arrays·coefficients는 별도 diagnostic 객체/이름으로 관리한다. FIT의 미래 teacher로 coefficient를 학습하는 것은 허용된 supervised 학습이며, query의 실제 미래를 배포 inference에 넣는 것과 구분한다. 이 문제는 helper의 ridge 수학 결함이 아니라 **새 runner 연결 계약의 미검증 항목**이다. ROOT와 구현 담당자에게 전달했고 다른 소스를 수정하지 않았다.

후속 수정 확인: math source `8ce0ceb4d7247e05d517f2658aad92e3835e46134cfd48d7c2ad482fe69a2eab`, 15개 fixture source `a629edc968c9a3f1aef9881f943b9807ae48048b7ddef7a2826b992543161526`를 실제 파일 SHA로 확인했다. 새 `forecast_from_selected_n1(model, latent_past, raw_last)`는 N1 model 객체/type/결합 SHA를 확인한 후 latent를 내부 계산한다. 미래 latent 인수가 없으므로 위 N3 공개 API 문제를 소스에서 보완했다. low-level `forecast`는 산술 감사용이고 hash만으로 latent origin을 증명하지 못함을 명시했다. 실제 runner도 이 공개 API를 사용해야 한다. FP32 casting 뒤 native forecast/decoder effective weight의 finite 검사도 추가됐다. fixture 실제 실행은 여전히 미확인이다.

## Oracle·새 runtime의 read-only 검토

[확인·소스] [oracle helper](../stage_separation_oracle_v1.py), [oracle fixture 소스](../test_stage_separation_oracle_v1.py), [새 runtime](../stage_separation_runtime_v1.py), [새 method lock](../METHOD_LOCK_STAGE_SEPARATION_v1.md)을 읽었다. 검토 시 파일 SHA는 oracle `0c3b7d52781fdfa44e0f1710f5521abd539296934b4ff86375bee54c6b28089c`, runtime `433381091138b850344b91796801fb358c8d11e010325849c8338252059805e0`, method lock `ae335e6fee09c9642f28abc2d6f57e523e3140cbaed2d0384e798ffc2b74eaa6`다. 작성 중인 소스는 이후 변경될 수 있으며 실행 전 마지막 SHA를 다시 고정해야 한다.

Oracle의 causal fill은 각 채널의 carry를 filled past last에서 시작해 observed 미래만으로 갱신한다. 실제 observed flag와 fill-source code를 별도로 남기고, complete-prefix를 K에 확장한다. normalized helper는 이미 normalized한 값을 받아 재정규화하지 않는다. encoder512와560 prefix를 비교하며, 실제 observed 미래가 있는 마지막 시점/sample을 골라 perturb한다. filled context가 실제로 안 바뀌면 PASS가 아니라 NOT_TESTED/None으로 보고한다. 뒤 latent의 반응을 모든 encoder의 필수 조건으로 강요하지 않는 범위도 명시한다. 이전의 첫 sample/누락 future를 교란하는 vacuous 검사를 이 소스에서 보완한 것을 확인했다.

`prepare_past_only_deployment_inputs`도 후속 소스에서 past-only validator를 사용하도록 분리됐다. future target/mask가 없어도 과거 payload를 준비하며, fixture에는 poison future 배열을 읽지 않는 조건이 있다. 이 fixture는 실행하지 않았다. actual caller는 원 `history_observed`를 명시해야 하고, 생략 시 all-observed context-mask fallback이라는 메타데이터를 실제 관측 mask로 보고하면 안 된다.

**아직 실제 계약으로 확인할 내용:** helper의 normalizer 검사는 dtype/shape/finite/positive-std이며 canonical 원 FIT 통계와 동일하다는 증명이 아니다. D2 cache validator도 구조·dtype 중심 검사이므로 이것만으로 원 CP/array/origin/mask/단위 계보가 검증됐다고 하면 안 된다. 실제 runner는 bridge/원 CP/normalizer/캐시 producer SHA를 결합하고 `z_last == latent_past[:,-1:]`, source raw_last/stat/origin/target/mask의 일치를 독립 검산해야 한다. oracle target의 observed normalized 단위와 560 real-device prefix/교란 parity도 미확인이다.

새 runtime은 기존 만료 authorization/manifest를 사용하지 않고 canonical **stage-separation** 파일을 요구한다. human response origin·response identifier·phase·diagnostic-only oracle scope, 원 계획/새 method lock SHA, runtime SHA, approved 이후 고정 start/expiry를 확인한다. 상한은 GPU process600초, CPU process900초, wall2400초, 추가 저장1GiB 및 RSS/allocated VRAM8GiB다. 같은 phase의 terminal receipt를 누적하고, prior continuation ACTIVE 및 자기 phase ACTIVE가 있으면 시작을 거부한다. 자기 lease는 exclusive create하며 final guard 실패를 terminal status에 반영한다. CPU/GPU process wall은 kernel 시간이 아니다.

Runtime에서 **provided** protected-prior receipt들의 SHA를 검사하는 것을 확인했다. 다만 검토 버전은 `run_manifest.get('protected_prior_receipts', [])`이므로 목록이 없거나 비어도 이 검사는 통과할 수 있다. 새 실제 manifest에는 보존할 원 ledger/receipt 전체 목록을 반드시 채우고, presence/비어있지 않음/record 형식의 실행 전 gate를 두도록 ROOT에 권고했다. phase 폴더에 있는 이전 receipt도 현재 authorization/run-manifest binding을 실제로 확인해야 한다. fresh 승인 파일이 없다는 것은 의도한 gate이며, 향후 유효한 승인이 생긴 후에도 불필요하게 재승인을 요구하라는 뜻이 아니다.

현재 신규 자원 승인·authorization/manifest·보호 원 receipt 내용을 읽거나 생성하지 않았다. 실제 fixture·asset 존재·원 CP/캐시 계보·solver·추론·과학 audit·배포 비용은 미실행이다. 결과적으로 **소스 수학/분리 API는 실행 전 검산 준비가 됐으나, 실제 자산·단위·coverage·device parity는 아직 판단할 수 없다**.
