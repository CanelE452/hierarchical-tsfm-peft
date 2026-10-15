# 예측 시점의 관측 과거로만 갱신하는 국소 디코더 진단

상태: **SOURCE·사전 규칙·작은 CPU 합성 검산 단계**. 원 실제 자료·NPZ·checkpoint·모델을 읽거나 새 CLI를 실행한 결과가 아니다. 실제 baseline 비용·독립 감사·보고 이후 ROOT가 source를 읽고 실행 여부를 판단한다. 기존 실행 소스·결과·gate·선정은 수정하지 않는다.

## 직접 질문과 기존 비교의 차이

기존 PCA/ForeCA의 pooled FIT 과거 최소제곱 D, encoder inverse D, FIT 실제 미래 목표 공간의 D는 모두 자료별로 고정된 K×C 출력 공간이었다. exact-constant gate는 특정 C채널의 출력을 현재 값으로 제한하는 사전 규칙이었다. 이번의 질문은 **매 forecast origin에서 이미 알고 있는 C 과거를 이용해 K→C 복원 관계를 바꾸면 고정 출력 공간의 한계가 달라지는가**다. E·latent 미래 예측을 그대로 두고 D의 조건부 적합만 바꾸어 진단한다. 기존 static LS를 같은 자료에 반복해 적합하는 실험이 아니다.

accepted FIT-target의 PCA_PINV/FIT_TARGET_PINV E, K, L512/H48, FIT scaler, 자료/split/origins, current-C anchor, frozen Bolt와 원 selected native head를 유지한다. Bolt와 native의 **해당 원 backend에서 이미 계산된 latent_history/future/last**를 각각 쓰고 E/F/head를 다시 실행하지 않는다. local solver에는 rawC 과거·true observed mask·cached latent 과거만 입력한다. 미래 정답은 적합에 없다. native는 원 static-D 조건의 selected head를 고정한 출력 개입이며 새 D에 최적화된 native 비교가 아니다.

## 마스크와 중심화·rank 규칙

각 window n과 rawC 채널 c의 **true observed past subset**에서 zmean과 xmean을 구한다. causal forward-fill 값은 원 E의 input과 cached latent에 남지만 관측되지 않은 filled x를 decoder response로 세지 않는다. `Zc = Zobserved−zmean`, `xc = Xobserved−xmean`에 FP64 NumPy lstsq를 수행하고 `bias = xmean−zmean·D`로 analytic intercept를 복원한다. 모델은 K개 slope와1개 intercept인 affine 모델이며 decoder bias는 미래 변화량 예측에서 상쇄된다.

사전 고정한 **centered slope singular-relative cutoff는1e−4**다. 이는 같은 observed-subset centered covariance의 eigen-relative cutoff1e−8과 대응한다. singular cutoff1e−8과 covariance eigen cutoff1e−8은 다른 규칙이다. augmented `[Zc,1]` 전체 행렬에 cutoff를 걸면 상수 열의 scale이 기준에 들어가 이 대응도 깨진다. 따라서 intercept는 별도 analytic·unpenalized로 처리한다. rcond/λ/K/상대 threshold를 TUNE/DEV 지표로 고르지 않고 자료를 읽기 전 수치 조건을 보고 정한다.

이 해는 **retained singular directions 안의 truncated minimum-norm LS**다. 작은 비영 방향을 버리면 원 full centered design의 무제약 최소제곱 최적해라는 주장은 성립하지 않는다. residual은 retained directions에는 직교하지만 버린 방향 때문에 full residual gradient는 비영일 수 있다. 최소 norm도 centered slope 좌표에서 정의되며 uncentered `[Z,1]`의 slope/intercept 전체 norm 최소와 다르다. rank-deficient 또는 거의 공선인 과거에서 off-past-span 미래 변화량의 복원은 이 사전 좌표 규칙에 의존한다.

rank 부족을 숨기기 위해 원 global D나 adaptive penalty로 갈아타지 않는다. true observed count=0인 채널만 D열0·재구성 bias=currentC의 표준 fallback을 쓴다. count=1은 centered slope rank0의 명시적 최소 norm D=0과 단일 관측값 intercept다. 둘의 이유와 count/rank를 구분한다. count<2는 복원 관계의 단서가 부족하다고 표시한다. count≥2도 K방향을 식별하거나 미래 persistence가 최적이라는 보장은 아니다.

각 origin/channel의 observed count·centered slope rank·intercept 포함 affine rank·singular values·retained condition·mask별 과거 reconstruction MSE, 각 origin의 D최대절댓값·Frobenius/operator norm·FP32 finite 상태를 기록한다. 1e−4는 retained 상대 condition≤1e4의 수치 규칙일 뿐 절대 Dgain이나 미래 오차 상한이 아니다. 작은 local singular scale, 큰 rawC response scale, 큰 latent forecast Δ로 여전히 오차가 커질 수 있다. FP32 coefficient/forecast overflow는 경로 실패로 남기고 clip·다른 penalty·global fallback으로 조용히 바꾸지 않는다.

## 미래 연산과 정보 조건

fit은 FP64, deployment D는 FP32다. 미래 연산은 미리 고정한 `delta32 = cached_future32−cached_last32`, `currentC32 + einsum('nhk,nkc->nhc',delta32,Dlocal32,optimize=False)` 순서다. 0 innovation에서 현재 C의 FP32 수치는 정확히 유지되지만 일반 부동소수점 덧셈은 signed-zero bit 보존까지 보장하지 않는다. 원 static control은 원 NumPy 전체 행렬·native Torch CPU2/chunk32의 저장 예측을 그대로 사용하고 byte 보존한다. 새 dynamic native는 동일 cached future에 대한 새 NumPy row-wise 출력 연산이다. controller는 static D를 새 einsum 순서로 넣는 산술 대조도 기록해 backend rounding과 D변화의 효과를 혼동하지 않도록 한다.

디코더의 정보 조건은 동일하지 않다. 새 interface는 **K future streams + current C + C의 known-past512/mask로 적합한 origin별 K×C coefficients**다. decoder 측에 C 과거의 조건부 metadata가 추가된다. strict K-only 압축·표현학습 자체의 개선·같은 decoder-side information 대조라고 쓰지 않는다. 이후 비용에는 local past masking/centering/LS·D생성·출력 연산 전체를 포함해야 한다.

TUNE/DEV의 **forecast origin 이전 과거**도 해당 예측의 local D 적합에 사용한다. 따라서 TUNE/DEV 자료를 어떤 fit에도 전혀 쓰지 않았다고 쓰면 틀리다. 확인할 계약은 future labels를 적합에 쓰지 않고, threshold/공간/head 선택을 새 TUNE/DEV 결과로 하지 않았다는 것이다. 새로운 supervised E/F/head 훈련과 달리 inference-time past-only decoder refitting이다.

## oracle와 해석 범위

origin별 D[n]의 rowspace가 달라지므로 원 static fixed-D 최소 오차를 새 출력의 lower bound로 재사용하지 않는다. 이전 global rank-K future oracle도 전체 split의 flattened 미래 행렬에 공통 K차원 출력 공간을 둔 조건의 최소였다. origin별 조건부 공간을 합치면 전체 rank가 K보다 커질 수 있어 그 oracle보다 낮아도 모순·누출·미래 정보 복구의 증거가 아니다. 이는 C 과거로 만든 조건부 coefficients를 추가하여 비교 클래스를 바꾼 결과다.

CPU producer 완료 후 독립 감사에서 **새 D[n]별 출력 공간의 최소 오차**를 별도 진단할 수 있다. 같은 complete-C future mask와 current-C anchor에서 알려진 미래 변화량만으로 `lstsq(D[n].T,V[n].T,rcond=1e−15)`의 real-arithmetic 최적 Δz를 구한다. 이는 새 per-origin D32를 FP64로 해석한 선형 공간 안의 normalized MSE 최소다. raw physical MSE/MAE 각각의 최소나 과거 입력만으로 달성 가능한 오차라고 하지 않는다. 그 미래 정답은 local D 적합·threshold·checkpoint·branch 선택에 사용하지 않고, encoder/F/head forward를 새로 실행하지 않는다. FP32 실제 trace와 real trace 차이를 따로 측정해 rounding allowance를 적용하고 RAW에 span 부등식을 적용하지 않는다. 과거 정보가 충분했다거나 미래를 복구했다는 증거로 확대하지 않는다.

과거를 완벽히 복원해도 미래의 C관계가 바뀌면, 심지어 latent 미래값이 정확해도 rawC 미래 복원은 틀릴 수 있다. 약한 방향을 버리면 불안정성이 줄면서 실제 예측 신호도 사라질 수 있다. threshold 부근 작은 입력 변화로 rank가 바뀌는 불연속도 남는다. 합성 반례는 수학적 경계를 확인하며 actual Robin/Beijing 실패 원인이나 실제 개선을 증명하지 않는다.

## 준비와 실행 전 gate

[순수 모듈](local_past_decoder_v1.py)·[합성 검산](test_local_past_decoder_v1.py)은 scientific IO/import 실행·모델 호출이 없다. raw/mask/cached latent array SHA를 묶고 window 순서 순열 동치, zero innovation, filled response 제외, 중복/rank 부족·0/1obs, 최소 norm의 좌표 의존, near-collinear FP32 오차 증폭, 약한 예측 방향 소실, rank 경계, 상대 condition과 절대gain 차이, 과거/정확latent 미래만으로 raw 미래를 보장할 수 없는 반례를 확인한다.

SOURCE controller의 실행 전 prerequisite는 accepted FIT-target COMPLETE+독립v2PASS, past-constant CPU COMPLETE+독립PASS, 그 whole deployment cost COMPLETE+독립v2PASS다. exact SHA·scope·counts를 먼저 묶고 actual assets를 읽는다. 아직 pending이면 새 적합을 시작하지 않는다. 자료2×split3×space2×Bolt/native2의36 static/local 쌍과 원 RAW6통제 지표를 같은 observed/complete-C mask로 저장할 예정이다.

native 두 seed의 cached latent past가 byte-identical일 때만 local D를 공유한다. 최대 cohort는 두 자료 합계816 origins×space2×backend2의3264 window matrices·44304 channel slots다. 이 수치는 완료 횟수가 아니다. 각 actual solver 호출/0obs fallback/재사용/실패 횟수를 별도로 기록해야 한다. local D가 비정상인 경로만 실패로 남기고 독립 자료·space·backend는 계속 확인한다. static source/예측/선정/모델 SHA를 보존한다. 미래 MSE/MAE·물리 단위 진단과 과거 복원 지표는 역할을 구분하고 원 static floor를 게이트 lower bound로 꾸미지 않는다.

현재 CPU probe는 local fit·과거 복원·캐시 future 출력 연산만 수행하며 전체 배포 비용은 미측정이다. 독립 CPU PASS 이후 필요하면 ROOT가 RAW_BOLT·PCA_PINV_BOLT·FIT_TARGET_PINV_BOLT·PCA_LOCAL_BOLT·FIT_TARGET_LOCAL_BOLT의 자료2×batch1/8, 총20조건을 같은 물리 입력으로 비교한다. CPU fill/정규화/E·과거 local LS·F·D·물리 단위 CPU 반환을 모두 timer 안에 넣고 첫8 physical-prefix와 원 cache의 parity를 먼저 확인해야 한다. 비용 소스·실행은 지금 완료된 것이 아니다.

`[판단]` 지속 수행하도록 한 사용자의 원인 분리 목표의 최소 후속이다. 새 exact-human 개별 승인이 있었다고 주장하지 않는다. 이미 기노출 DEV와 이전 진단을 본 뒤 제안한 탐색적·적응적 연구이며 독립 일반화 확증이 아니다. 지금 실제 데이터·NPZ·model·CLI·새 whole cost는 미실행이다. TEST·큰 모델 retrain·기존 job 종료/재개·설치·자동 commit/push는 하지 않는다.
