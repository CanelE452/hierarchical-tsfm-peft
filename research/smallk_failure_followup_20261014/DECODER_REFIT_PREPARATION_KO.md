# D2: 원 MAIN E/F·비선형 hidden을 동결한 출력 재적합 준비

상태: **PREPARATION_ONLY**. Bi의 전체 비용 proof·최종 독립 검산·보고가 완료되지 않았으므로 실제 실행 gate는 닫혀 있다. 이번 작업은 Python 소스와 기존 범위 문서를 읽고 합성 CPU fixture로 helper를 검증했다. 실제 과학 NPZ/CP·모델·GPU·보호 TEST를 열거나 실행하지 않았다. 원 job 재개·Git 작업도 없다.

준비 파일은 [frozen_decoder_refit.py](frozen_decoder_refit.py), 합성 검증은 [test_frozen_decoder_refit.py](test_frozen_decoder_refit.py)다. 입력 복원·새 실제 latent 추론·원 MAIN/BASE_Q state equality·원 native 대조·실제 실행은 root의 후속 runner 책임이다. 이 문서는 실험 결과나 원인 확정 보고서가 아니다.

## 질문과 고정하는 조건

같은 원 MAIN의 E/F 출력을 고정한 상태에서 D 출력 가중치를 FIT에 다시 적합하면 원 D보다 예측 오차가 줄어드는가. 이는 현재 학습된 hidden map을 조건으로 decoder 출력 최적화를 분리하는 진단이다. hidden까지 다시 학습한 nonlinear D 전체의 최선 성능, E/F의 원인, 보편적인 복원 하한을 입증하지 않는다.

[확인: 읽은 코드] 원 `engine.Codec.forward_path`는 다음 순서로 계산한다.

```text
z_past = original_MAIN_E(preprocessed_normalized_past)
z_last = z_past[:, -1:]                           [N,1,K]
z_F    = frozen_original_F(z_past)                [N,48,K]
D(z)   = W_linear z + b_linear
       + W_out GELU(A z + b_hidden) + b_out
pred   = x_last + D(z_F) - D(z_last)              [N,48,C]
```

E는 원 linear project+causal TCN, F는 원 FrozenBolt+원 zero-safe normalizer다. 이번 helper는 E/F를 호출하거나 구성하지 않는다. 원 D의 GELU는 `approximate='none'`, hidden width32다. `controls.load_codec`의 CP/state 확인 흐름을 소스로 읽었지만 해당 함수는 CUDA로 옮기므로 호출하지 않았다.

[확인: 수학 및 합성 CPU 검산] outgoing 두 bias는 anchor에서 수학적으로 상쇄된다. 따라서

```text
phi = concat(z_F - z_last,
             GELU(A z_F + b_hidden) - GELU(A z_last + b_hidden))  [N,48,K+32]
pred_new = x_last + phi @ W_new                                 [N,48,C]
```

의 절편 없는 출력 적합은 원 anchored decoder class 안에 남는다. E/F·A·b_hidden은 고정하고 두 outgoing weight만 바꾼다. `W_new[:K].T`는 `codec.decode.weight`, `W_new[K:].T`는 `residual.2.weight`로 다시 내보낼 수 있다. 원 outgoing bias도 source state로 보존한다. 원 연산 순서의 FP32 상쇄는 정확 산술과 다를 수 있으므로 원 D·factored 원 D·저장 원 prediction의 실제 수치 차이를 별도 기록한다.

비선형 hidden 차이는 절대 z_F와 z_last에 의존한다. 기존 `SELECTED_FIT_ALIGNMENT.npz`의 `latent_fit_pre_q = z_F-z_last`만으로 만들 수 없다. post_q 또는 실제 미래 target_latent를 예측 feature에 넣지 않는다. `z_F=z_last`이면 phi=0이므로 새 factored 출력은 x_last를 정확히 유지한다. C개 현재 상태의 anchor 우회도 그대로 있으며 strict K-only 압축이 아니다.

## 적합·선택 규칙

후보는 `Y`, `RAW`, `MIXED` 세 가지다. Y는 실제 미래 정답, RAW는 원 RAW_BOLT의 과거 기반 forecast, MIXED는 `.5*Y+.5*RAW`다. MIXED의 같은 마스크 least-squares minimizer는 `.5*loss_Y+.5*loss_RAW`와 같지만 두 objective의 loss 상수항은 다르다.

세 목적 모두 실제 Y의 **같은 FIT observed mask**를 사용한다. teacher 값이 모두 유한하다는 이유로 RAW 조건에 더 많은 학습 행을 주지 않는다. RAW forecast는 미래 정답 oracle이 아니며 원 teacher의 오류를 복제할 수 있다. 모델 선택은 세 목적 모두 **TUNE 실제 Y의 normalized observed channel-macro MSE**로 한다.

FIT 전체 feature 행으로 각 feature의 RMS만 적합하고 평균 중심화·절편은 넣지 않는다. zero RMS는1로 둔다. feature scale은 target이나 TUNE/VAL/EVAL을 읽지 않는다. 각 채널은 자기 observed FIT 행으로 다음 문제를 FP64에서 푼다.

```text
min_w mean_over_observed_fit_rows((phi/RMS @ w - (target-x_last))^2)
      + alpha * sum(w^2)

alpha = 0, 0.0001, 0.001, 0.01, 0.1, 1, 10
```

alpha0은 `numpy.linalg.lstsq`의 최소 노름 해, 양수 alpha는 정규화한 Gram matrix에 ridge penalty를 더한 solve다. 출력 effective weight `w/RMS`를 FP32로 저장하고 FP32 feature에서 예측한다. FIT 관측이 없는 채널은 원 outgoing weight를 유지하며 해당 채널을 누락 목록으로 기록한다. FIT/TUNE의 전체 mask가 비어 있으면 실행을 거부한다. TUNE 점수가 완전히 같은 경우 grid의 앞선 alpha를 고른다.

모든 alpha의 출력 head와 TUNE normalized/physical/RAW-teacher 오차, 선택 alpha, feature RMS·가중치 SHA·mask count·걸린 시간을 남긴다. 선택 출력은 FIT/TUNE/반복 VAL 또는 이미 노출 EVAL에서 같은 Y mask로 채점한다. MSE/MAE와 채널별 값 및 정규화/원단위를 함께 기록한다. VAL/EVAL은 선택이나 feature scale에 관여하지 않는다. 해당 평가는 새 독립 holdout 증거가 아니다.

원 D·정확 수식의 factored 원 D·source D FP32·RAW·각 refit을 함께 비교한다. 이 조건은 **추가 D 최적화와 TUNE 선택을 수행**한다. 추가 E/F 최적화가 없다는 이유로 추가 최적화 자체가 없다고 쓰지 않는다. 학습·저장·검산을 포함한 helper wall과 각 목적/alpha wall은 기록하며 deployment latency로 해석하지 않는다.

## API와 caller 검증 계약

```python
run_decoder_refit(
    fresh_output,
    splits=verified_bundle,
    codec_state=exact_selected_MAIN_CP_codec_state,
    mean=exact_stored_mean,
    std=exact_stored_std,
    source_record=verified_lineage,
    shared_limit=existing_budget_limit,
    execution_authorized=True,  # root: Bi 완료 뒤 전환할 때만
)
```

`execution_authorized` 기본값은False다. CLI는 `--describe`만 지원하며 실제 source loader/실행 기능은 없다. import도 원 캠페인·모델을 불러오지 않는다. torch 호출은 제공된 CPU 배열의 GELU/linear 연산에만 사용한다. root가 실제 CP를 CPU로 읽어 검증한 뒤 `['codec']` state를 공급한다. CPU tensor와 FP32 ndarray를 받으며 CUDA tensor는 거부한다.

```text
splits keys   fit/tune/val (Robin), fit/tune/eval (Beijing); test 금지
z_future      float32 [N,48,K], 원 F(E(X)) 절대 좌표
z_last        float32 [N,1,K], 같은 원 E(X)의 마지막 절대 좌표
raw_last      float32 [N,C], 원 전처리된 normalized 과거의 현재 상태
original_prediction
raw_forecast
target        float32 [N,48,C], normalized; target는 unobserved 위치 NaN 허용
observed      bool    [N,48,C], 정확한 원 실제 Y mask
origins       int64   [N], 정확한 원 순서
mean/std      float32 [C], 원 저장 통계를 그대로 사용
```

실제 조건은 원 Robin C17/K4·seed92601/92602·창256/64/30 또는 Beijing C11/K2·동일 seed·창256/64/146, L512/H48와 `ANCHOR_NONLINEAR_D`다. helper는 shape/dtype·유한성·정렬 origin·그 조건을 검사한다. 원 layout으로 통계를 재계산하거나 raw_last를 교체하지 않는다. exact origin/stats/mask equality 및 past-only latent 생성·MAIN/BASE_Q 귀속은 **caller가 배열과 source를 실제 검증해야 한다**. helper의 caller attestations 검사는 독립 파일 감사가 아니다.

`source_record`는 JSON으로 저장할 수 있는 객체이며 적어도 다음 항목을 채운다.

```text
bi_complete: true
dataset, seed, architecture='ANCHOR_NONLINEAR_D', input_length=512
decoder_state_sha256:
    decoder_state_sha256(DecoderState.from_codec_state(exact_MAIN_CP['codec']))
verification: 아래 이름 모두 true
    source_checkpoint, source_snapshots, same_original_main_E_F,
    absolute_latents, original_stats_origins_masks, split_roles
source_artifacts: 각 {alias, path, sha256, verified_sha256}; sha 두 개가 같아야 함
    필수 alias MAIN_CP, MAIN_PREDICTION, RAW_PREDICTION, ORIGINAL_SOURCE
    추가 source·stats/origins/latent·F/model snapshot 근거도 같은 목록에 기록
```

decoder digest는 명시된 원 D의 여섯 key 이름+FP32 bytes를 순서대로 hash한 값이다. 전체 codec hash와 다르며, 전체 CP file hash·선택 MAIN codec hash를 source lineage에도 보존한다. source artifact 경로는 저장소 기준 상대 경로로 작성한다. seed·선택 update·모델/F hash·원 source별 SHA·latent 추출 시각/시간·F 호출 수·새 호출/예산 receipt·검증 근거 경로도 caller metadata에 남긴다. `source_record`는 RESULT에 그대로 기록된다.

실제 실행은 새 `.cache/smallk_failure_followup_20261014/` 하위의 **존재하지 않는** output만 받는다. 기존 Limit의 `check()`를 각 입력·alpha·선택 출력 전후 호출하고, 실제 프로세스 wall300초/RSS8GiB 협력 guard를 추가한다. helper가 기존 ledger를 만들거나 닫지 않는다. 외부 watchdog·진짜 전체 프로세스/cumulative 자원 집계·terminal receipt는 caller 책임이다. CPU LAPACK 호출 중에는 협력 guard만으로 강제 중단할 수 없다.

기본 normalized parity tolerance는 atol2e-5/rtol2e-6이며 fixture에서만 검증됐다. 실제 시작 전에 root가 원 조건의 batch/backend 수치 검산 및 tolerance 근거를 고정해야 한다. 실패를 보고 뒤 tolerance를 자동 완화하지 않는다. source D↔저장 original 또는 source D↔factored 원 D가 통과하지 않으면 refit 전에 새 입력/reference와 실패 위치를 보존하고 예외를 낸다.

## 새 출력과 실패 보존

모든 파일을 exclusive create로 작성한다. 원 source 파일을 쓰거나 overwrite하지 않는다.

```text
SOURCE_DECODER.npz              원 D 여섯 state 배열
INPUT_<split>.npz               absolute latent/current last·원 prediction/RAW/Y/mask/origins/stats
REFERENCE_<split>.npz           원 D/source D/factored D/RAW 출력·실제 phi·origins
<objective>_alpha_<alpha>_HEAD.npz
                               모든 alpha의 FP32 weight/scaled weight, FP64 RMS, observed count
<objective>_SELECTED_DECODER.npz 원 architecture로 재주입할 decoder state
<objective>_SELECTED_<split>.npz
                               선택 출력과 origins
METRICS.csv                    모든 alpha TUNE 및 기준/선택 출력의 macro·채널 MSE/MAE
RESULT.json                    lineage·SHA/schema·선택·전체 grid·시간·parity·성공/실패
REPORT_KO.md                   한국어 해석 경계와 선택 결과
```

입력/schema/source gate 오류는 output 생성 전에 거부한다. output 생성 뒤 local budget/parity/solve/save 오류는 `FAILED_PRESERVED`, phase/split/objective/alpha/예외를 기록하고 이미 생긴 파일을 남긴다. 부분 head 결과가 있으면 그것도 남긴다. 디스크 자체가 쓰기 불가능한 실패에서 terminal 파일 기록을 보장하지 않는다.

## 실제로 확인한 준비 검증

[확인] 프로젝트 `.venv/Scripts/python.exe`에서 합성 CPU 테스트24개가 통과했다. 원 GELU/direct-source FP32 parity, factored/linear class parity, 동일 delta의 다른 absolute latent 반례, zero-anchor·절편/중심화 금지, 부분 mask·NaN·무관측 채널 fallback, RMS FIT-only, ridge alpha mean-loss 의미, 원 architecture 재주입, Y/RAW/MIXED 구분, 실제 Y TUNE 선택·tie, VAL 변경의 fit/선택 독립성, 채널 macro/원단위 계산, readonly 입력, 새 출력·overwrite 거부, schema/TEST/root gate, source decoder digest·alias/hash·split guard, budget/parity 실패 보존을 검증했다.

최종 검증은 준비 폴더의 합성 테스트와 compile/`--describe`로 한정한다. 저장소 전체 테스트에는 실제 모델·자료 경로가 있을 수 있으므로 현재 준비 gate에서 실행하지 않는다. 합성 output은 pytest 임시 fixture이며 원 캠페인 과학 산출물이 아니다.

[확인: 현재 작은 source 파일 직접 hash]

```text
engine.py
05f2cf63b3f7f6a01bf722c51771455be9c73c21ab301d7eb950b3dd58bab817
controls.py
324c75a65a001ac0cae30610abb148499718cbad838a69e0012df4a85bd83393
forecast_component_study_20261013/component_experiment.py
ff4cf8c9ceeaec280a6af69e8c4e586685457030feece790faba464afd282d03
```

위 engine/controls는 ASSET_MAP의 선언값과 일치했다. 이 확인을 source CP/NPZ/모델 bytes까지 실제 해시한 것으로 확장하지 않는다. 실제 실행 source snapshot 선택과 전체 source chain 확인은 root가 실행 전에 끝내야 한다.

남은 blocker는 Bi 완료 gate와 정확한 원 CP/source/arrays 검산, 누락된 절대 z_F/z_last/current last의 원 조건 복원이다. helper 준비 완료가 실제 D2 개선 입증이나 작은 K 상위 목표 완료를 뜻하지 않는다.
