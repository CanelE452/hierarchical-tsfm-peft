> 게시 사본입니다. 원문 작성 당시의 로컬·준비·미실행 상태는 역사로 보존합니다. 최신 완료/미해결 상태와 게시 범위는 [색인](../README_KO.md)을 확인하세요. 과학 원본은 바꾸지 않고 링크와 이 안내만 추가했습니다.

# FIT 과거→미래 방향 선택의 데이터 계약

상태: **SOURCE와 기존 JSON metadata만 읽은 설계 검토**. 새 코드·테스트·NPZ/원자료 읽기·F/모델·GPU·학습·비용 측정은 모두 0이다. 기존 RRR/PLS를 새 통계 방법으로 주장하지 않는다. 같은 연구 목적의 후속 실행은 이미 허용된 범위이며, 이 문서는 추가 사용자 승인 gate가 아니다.

먼저 [방법론 완성 계획](PLAN_METHODOLOGY_COMPLETION_KO_v1.md)과 [판별 질문 검토](METHODOLOGY_DECISION_REVIEW_KO_v1.md)를 읽었다. 목적은 variance가 큰 축 대신 **FIT의 현재 C 벡터와 미래 C 벡터 사이의 교차 구조로 pointwise C×K encoder를 고를 수 있는지**, 그리고 그 방향이 같은 E/D의 frozen F에도 유용한지를 분리하는 것이다.

## 먼저 내리는 판단

**첫 결정 실험은 (a) FIT origin의 마지막 current-C → 48개 미래 horizon의 masked cross-covariance SVD 하나를 권한다.** 입력은 C차원이고 응답 쪽만 HC차원으로 쌓는다. Robin K4/Beijing K2·L512/H48과 기존 전처리·origin·mask를 유지한다. whitening과 E의 scale은 원 FIT 과거 covariance에 맞춘다. E 적합에 TUNE/DEV를 쓰지 않으며 새 ridge/lag/K sweep을 붙이지 않는다.

다만 Beijing의 미래 결측 때문에 이것을 **256×HC 완전 응답 행렬에 대한 exact RRR/표준 PLSSVD**라고 부르면 안 된다. 가장 작은 공통 계약은 horizon별 complete-C 응답만 이용해 교차 moment를 만든 뒤 연결하는 방식이다. 정확한 RRR는 모든 HC가 관측된 origin을 공통 선택하거나 마스킹 목적을 따로 최적화해야 한다. 이는 별도 후보다.

(b) FIT 과거 window 안의 lag1/4/16/48 pair는 미래 FIT label을 전혀 요구하지 않는 장점이 있지만, 겹친 window의 중복 가중치·window별 fill 초기값·lag별 표본 불균형을 더 결정해야 한다. 첫 개입으로는 (a)보다 작지 않다. 어느 쪽도 개선이나 frozen univariate F 호환성을 보장하지 않는다.

## 원 입력과 기존 방법에서 확인한 사실

[확인] [prepare_inputs.py](../../../smallk_failure_followup_20261014/prepare_inputs.py)의 `DatasetSpec`, `old_window_preprocess`, `build_split_from_physical` 실행 흐름상 다음 계약이다.

```text
dataset                  C   K   FIT   TUNE   DEV          L    H
robin                   17   4   256     64    val:30     512   48
beijing_aotizhongxin     11   2   256     64    eval:146   512   48

history X       float32[N,512,C], 원 FIT mean/std로 정규화
history mask    bool[N,512,C], 실제 관측 여부
target Y        float32[N,48,C], mask가 true인 값만 응답으로 유효
future mask M   bool[N,48,C], 실제 관측 여부
origins o       int64[N], 미래 첫 timestamp의 index
current a       X[:,511,:] == raw_last, timestamp o-1
history 시각    o-512,...,o-1
future h 시각   o+h, h=0,...,47
```

[확인] fill은 **origin별 512 window마다 원 FIT mean으로 초기화**하고 시간순 forward fill한다. 미래를 역방향으로 채우지 않는다. 모든 과거 값을 global series에서 한 번 fill하는 방식으로 바꾸면 기존 입력과 달라진다. target의 결측 자리에 저장된 0/대체값은 실제 Y 관측이 아니므로 응답으로 넣지 않는다. source predictor의 filled current는 유효한 과거 입력이지만 `observed current`로 재표기하지 않는다.

[확인] [fit_projection](../../projection_followup_v1/projection_math_v1.py)은 FIT history를 N·L×C로 펴서 filled 값과 반복 window 위치를 포함한 mean/covariance를 만든다. PCA는 이 covariance의 방향, ForeCA는 whitened 과거 spectrum의 entropy 방향을 사용한다. D는 원 채널마다 true-observed FIT 과거만 쓰는 masked affine LS다. 따라서 지금 후보의 **과거와 미래 사이 교차 moment**와 목적이 다르다.

[확인] [fit_target_subspace](../../projection_followup_v1/future_target_projection_v1.py)는 complete-C FIT 미래 위치의 `Y-current_C`에 **uncentered SVD**를 하고, 그 C차원 output 방향을 FIT 과거에서 whiten한다. 미래 mean/intercept를 제거하는 예측 회귀나 cross-covariance가 아니다. D는 deployed E의 pseudoinverse다. 그러므로 PCA/ForeCA/FIT-target을 단순히 예측 학습 전혀 없는 기법으로 묶어서는 안 된다.

[확인] 기존 [실제 FIT-target RESULT JSON](../../projection_followup_v1/runs/actual_fit_target_v1/RESULT.json)의 `fit_metadata`에는 다음이 저장돼 있다. 이 문서에서 mask 배열을 다시 열거나 complete-H origin 교집합을 계산한 것은 아니다.

```text
dataset                 complete-C future positions   per-horizon complete origins
robin                         12288 / 12288                  256 / 256
beijing_aotizhongxin           10881 / 12288                  220~234 / 256
```

JSON SHA256: `5d213da750bc6eb3b43dcb9d25fa08c3313f4c2662148cc2bad45adcc2cf6f98`.

Robin은 이 metadata상 모든 FIT 미래 HC가 관측됐다. Beijing의 **48 horizon 모두 complete인 origin 수**, 실제 고유 timestamp 수·중복도·현재값 fill 비율은 이 요약만으로 알 수 없다. per-horizon 최소220을 complete-H origin220으로 해석하지 않는다.

## (a) 현재 C → 미래 HC: 권장하는 정확한 정의

아래는 성능을 확인한 구현이 아니라 후속 구현에 고정할 [설계]다.

1. 원 FIT256개 origin의 `a_i=X_i,511`과 `Y_i,h`만 E 방향 적합에 쓴다. 응답은 **정규화된 actual 미래 level Y**다. 현재 C를 뺀 delta 응답은 첫 후보에 섞지 않는다.
2. `m_i,h=all_c M_i,h,c`로 horizon별 complete-C 집합 `I_h`를 만든다. source current의 complete 여부로 origin을 추가 탈락시키지 않는다. source는 기존 causal fill 입력이며 해당 fill 비율을 별도로 남긴다.
3. `n_h=|I_h|`와 각 h의 source/response mean을 FIT에서만 구한다. 어느 h라도 n_h<2이면 horizon을 빼지 않고 실패를 보존한다. 현재 JSON summary상 이 gate는 두 자료 모두 충족하지만 actual 배열의 동일성까지 확인한 것은 아니다.

   `mu_a,h = mean_{i in I_h} a_i`, `mu_y,h = mean_{i in I_h} Y_i,h`

   `C_h = (1/n_h) sum_{i in I_h} (a_i-mu_a,h)^T (Y_i,h-mu_y,h)`

   행벡터 convention에서 `C_h`는 C×C다. horizon별 같은 mask가 모든 C 응답에 적용되므로 서로 다른 채널의 결측 placeholder를 covariance에 넣지 않는다. 이는 **그 horizon에서 complete-C였던 표본에 조건부인 covariance**다. 관측 누락이 비무작위면 전체 모집단의 unbiased covariance라고 주장하지 않는다.
4. 원 PCA/ForeCA와 같은 FIT filled history의 mean `mu_hist`, covariance `S_hist`를 계산한다. `eig > maxeig*1e-8`만 남긴 whitening `W`의 shape은 C×r이며 `W^T S_hist W=I_r`다. rank r<K면 실패 상태를 보존하며 K를 바꾸거나 원 채널을 임의로 빼지 않는다.
5. `T=[W^T C_0 | ... | W^T C_47]/sqrt(48)`의 shape은 r×(48C)다. FP64 SVD의 왼쪽 상위 K 방향 `U_K`로 `E=W U_K`를 만든다. 큰 절댓값 좌표의 부호를 양수로 고정한다. singular gap/rank와 `E^T S_hist E`를 기록한다. 수치 rank cut은 상대1e-8로 사전 고정하고 rank<K이면 보존된 실패로 끝낸다. 경계 singular value tie는 방향의 고유성이나 플랫폼 간 bit 동일성을 보장하지 않는다.
6. 배포는 모든 history timestamp에 **동일한** `z_i,t=(X_i,t-mu_hist) E`를 FP32로 적용한다. deployment에 `mu_y,h`, FIT 미래 값, 새 origin의 미래 mask를 전달하지 않는다. 입력은 N×512×K이고 F는 원 K개 stream 경로를 그대로 쓴다.

정확한 최적화 대상은 `U^T U=I_K` 아래 `||U^T T||_F^2` 최대화다. 이 weighted cross-moment SVD는 기존 supervised covariance 방향 선택의 한 계약이며 **masked least-squares prediction error의 전역 RRR 해와 같다고 주장하지 않는다**. 출력 singular vectors는 HC차원이므로 pointwise E로 사용할 수 없다. 표준 PLSSVD가 cross-covariance SVD의 입력 singular vectors로 X를 투영한다는 점은 [scikit-learn 공식 cross-decomposition 설명](https://scikit-learn.org/stable/modules/cross_decomposition.html#plssvd)으로 확인했다. 위 horizon별 mask/whitening은 이 문서의 설계 선택이며 그 라이브러리의 missing-data 지원 기능을 뜻하지 않는다.

**왜 actual Y인가.** [확인: 대수] delta 응답을 쓰면 `Cov(a,Y-a)=Cov(a,Y)-Cov(a,a)`다. 미래 Y가 a와 독립인 white noise여도 두 번째 항이 남는다. 이는 current에서 예상되는 mean reversion을 예측하는 데에는 의미가 있지만, 큰 negative cross-covariance를 미래 level의 공통 예측 성분 발견으로 부를 수는 없다. 기존 anchor 복원과 목적을 맞추려 delta를 연구한다면 `DELTA_RESPONSE`라는 별도 사전 후보로 기록해야 한다. 첫 실험은 centered future level로 이 혼동을 줄인다. centered level도 persistence를 활용할 수 있으며 어려운 혁신만을 분리했다는 뜻은 아니다.

**가중치와 중복.** 각 h의 covariance moment 안에서 origin은 동일 가중치1/n_h다. T에서 block 계수는1/sqrt(48)이므로 **제곱 cross-moment 목적**에서 horizon별 비중은1/48이다. 이 목적은 각 sample의 squared prediction error 합이 아니며 pair별 위험 가중치 `1/(48*n_h)`와 동치라고 쓰지 않는다. 같은 실제 timestamp가 서로 다른 origin/horizon의 응답으로 반복되더라도 이번에는 그대로 보존한다. 서로 다른 lead-time의 예측 과제가 원 평가에도 있기 때문이다. complete 데이터에서는 모든 origin이 각 horizon의 moment에1/256로 기여한다. timestamp를 deduplicate하거나 역빈도 가중하면 다른 moment가 되므로 첫 후보에 섞지 않는다. 이 반복은 독립 표본 수 증가를 뜻하지 않으며 `256*48`을 iid N으로 보고하지 않는다.

## exact RRR의 왼쪽 factor와 K stream은 어디까지 같은가

[확인: 문헌/대수] 공통 complete 응답과 같은 행 가중치가 있을 때, centered predictor `A:n×C`, response `R:n×(HC)`, 행 가중치 `Omega`, 양의 lambda에 대해

`min_{rank(B)<=K} ||Omega^(1/2)(R-A B)||_F^2 + lambda ||B||_F^2`

를 생각한다. `S=A^T Omega A+lambda I`, `G=S^(-1/2) A^T Omega R`라 두고 `G=U Sigma V^T`를 SVD하면

`B_K=S^(-1/2) U_K Sigma_K V_K^T = E_RRR Aout`

이고 `E_RRR`는 C×K, `Aout`은 K×(HC)다. 공통 complete matrix에서 rank와 ridge를 함께 다루는 기존 문제는 [Mukherjee & Zhu, 2011 원문 §2, 식4~6](https://dept.stat.lsa.umich.edu/~jizhu/pubs/Mukherjee-SADM11.pdf)에 있다. 위 factor 식은 그 목적을 square completion으로 풀어 이 배포 축에 맞춰 쓴 대수다. `lambda=0`에서는 S의 retained range/pseudoinverse를 따로 정의해야 한다. 일반 coefficient B의 보통 truncated SVD만으로 위 ridge-RRR 목적의 해가 된다고 간주하지 않는다.

`(a_i-mu_a)E_RRR`와 매 timestamp의 `(X_i,t-mu_hist)E_RRR`는 같은 C차원 선형 방향을 적용한다. center 차이는 상수 translation이며 회귀 원식 재현에는 response intercept와 함께 반영해야 한다. E를 FIT history에서 재whiten하는 invertible K×K 변환 `J`를 붙이면 `E'=E_RRR J`, `Aout'=J^(-1)Aout`으로 direct RRR 예측을 보존할 수 있다. 그러나 비선형 F를 통과한 결과가 보존되는 것은 아니므로 E의 scale/rotation을 별도 확정하고 기록해야 한다.

**RRR를 그대로 새 F 경로라고 부를 수 없는 이유는 두 가지다.**

- predictor를 `vec(X_past):LC`로 바꾸면 왼쪽 factor가 `(LC)×K`가 된다. 이는 origin당 latent state K개이며, 기존의 **512 시점×K stream**인 E가 아니다. factor를 reshape/평균해서 C×K로 바꾸면 그 회귀 해의 보존을 잃는다. 원 C 직접 ridge의 대조가 LC를 써도 되지만 그것을 배포 encoder로 전용하지 않는다.
- RRR의 `Aout:K×HC`는 horizon별 C 회귀 출력이다. 기존 pointwise RECON D(K×C)나 L7 D(7K×C)와 같은 객체가 아니다. `Aout`으로 예측이 잘된다고 `(predicted future z)D`가 잘된다는 결론은 나오지 않는다.

Beijing에서 horizon마다 mask가 다르면 각 response block의 `A^T Omega_h A`도 달라진다. 이때 위 **하나의 S에 대한 closed form을 그대로 쓰면 다른 문제**를 푼다. 다음 둘만 exact-RRR의 정직한 별도 계약이다: (i) 모든 HC가 complete인 origin의 공통 교집합으로 제한하고 표본수/탈락 bias를 보고, (ii) 실제 mask별 squared error 아래 rank factor를 최적화하고 비볼록 최적화 한계를 보고한다. 아직 공통 교집합 수를 모르는 상태에서 (i)가 충분하다고 약속하지 않는다. 결측 Y=0로 채우는 제3 경로는 허용하지 않는다.

PLSRegression을 택할 경우에도 raw `x_weights_`를 무조건 최종 배포 projection으로 쓰지 않는다. deflation을 합성한 `x_rotations_`와 입력 centering/scaling까지 실제 `transform` 경로로 맞춰야 한다. [공식 설명](https://scikit-learn.org/stable/modules/cross_decomposition.html#transforming-data)이 이 차이를 명시한다. 이 추가 선택을 줄이기 위해 첫 후보는 반복 deflation 없는 위 single SVD로 고정한다.

## (b) FIT 과거 window의 lag pair: 비교용 정확한 정의

[설계] `ell in {1,4,16,48}`, `r=0,...,511-ell`에서 predictor `a_i,r=X_i,r`와 response `b_i,r,ell=X_i,r+ell`를 만든다. 실제 timestamp는 각각 `tau=o_i-512+r`, `tau+ell`이다. 응답 mask는 `all_c history_observed[i,r+ell,c]`이며, true-observed response만 사용한다. filled source predictor는 허용하되 source 관측 여부를 따로 기록한다. pair는 FIT history 내부에만 있으며 `r+ell<512`다. FIT future Y나 TUNE history를 추가 pair로 가져오지 않는다.

한 window당 결측 전 pair 수는 ell별511/508/496/464다. **실제 채택 pair 수, 고유 `(tau,ell)` 수, 중복도**를 저장해야 한다. 결측 전 정확한 slot 합은 `(511+508+496+464)*256=506624`이고 독립 표본 수가 아니다.

같은 `(tau,ell)`이 여러 origin window에 중복된다. 이때 target이 true observed이면 값은 같지만 source는 **window 초기화 mean과 이용 가능한 prefix가 달라** filled 값이 다를 수 있다. 따라서 timestamp만 같다고 평균해 하나의 '실제 source'로 합치지 않는다. 이 후보를 실제 채택할 때는 다음 보존 계약을 권한다.

- 각 unique `(tau,ell)`의 유효 중복수 `d_tau,ell`을 계산한다. 원 window occurrence는 유지하고 각 가중치를 `1/d_tau,ell`로 해 unique pair마다 총질량1을 준다.
- 각 lag 안에서 가중치 총합을1로 정규화해 weighted source/response mean과 cross moment `C_ell`을 만든다. `T_b=[W^T C_1|W^T C_4|W^T C_16|W^T C_48]/2`로 연결해 제곱 cross-moment 목적에서 네 lag가 각각1/4이 되게 한다. W와 배포 center/scale은 (a)와 같다.
- 동일 `(tau,ell)` source의 fill 값이 여러 개인 수/최대차, source filled 비율, unique response timestamp multiplicity를 남긴다. 이것은 deployment window 상황을 평균한 통계이며, 엄밀한 하나의 global causally filled process의 covariance라고 부르지 않는다.

이 후보는 window 반복을 그대로 주는 기존 PCA covariance와 중복 가중치가 다르다. W/center만 기존 PCA 계약을 재사용하더라도 cross moment의 유효분포는 달라진다. inverse-multiplicity가 성능을 높인다는 근거는 없고, 다른 가중치를 결과를 본 뒤 고르는 것은 추가 탐색이다. lag48은 원48-step 목표와 관련되지만 lag1/4/16만 강조하면 단기 자기상관을 크게 고를 수 있다. 반대로 `(a)`는 실제 원 horizon 전체를 직접 포함해 이 정합성 선택을 줄인다.

## input 예측 성분과 self-forecastable stream의 불일치

[확인: 반례 대수] `x1(t)=epsilon(t)`, `x2(t+1)=epsilon(t)`, epsilon은 시간 독립 평균0 noise라고 하자. current의 `E=e1`은 미래 x2를 예측하는 정보를 보존한다. 하지만 latent `z(t)=x1(t)` 자체의 미래는 white noise이고, 동시점 x1와 x2가 무상관이면 pointwise RECON D는 x2를 복원하지 못한다. input-side predictive E, output-side recon span, stream 자신의 미래 예측 가능성은 세 가지 다른 조건이다.

L7 decoder는 일부 지연 관계를 표현할 수 있으므로 위 pointwise-D 반례를 그대로 모든 L7 실패 증명으로 확대하지 않는다. 그러나 H48에서 D의 최대lag32와 예측 causal 구조, known-past offset까지 따로 확인해야 한다. 작은 raw/current RRR가 성공해도 frozen **univariate** F가 K stream 각각에서 필요한 미래를 만들 수 있다는 보장은 없다. 이 구분을 생략하면 '예측 covariance를 썼으니 TSFM compatible'이라는 순환 주장이 된다.

## 가장 작은 공정 대조와 D 계약

첫 실험에서는 새로운 E 하나와 같은 원자료의 PCA E, FIT-target E를 비교한다. 모든 E를 같은 FIT-history whitening scale에 놓고, **D 규칙은 representation 간에도 하나로 고정**한다. 기존 PCA의 LS-D와 FIT-target의 pinv-D를 원 상태 그대로 섞어 E 효과로 보고하지 않는다. ForeCA를 추가한다면 같은 D 규칙과 새 native가 필요하며 첫 결정 후보 수를 늘리는 선택이다.

현재 L7 RECON 연구 질문을 유지하려면 각 E에서 [fit_reconstruction](../../projection_followup_v1/causal_lag_decoder_v1.py)을 동일하게 다시 적용한다: lags0/1/2/4/8/16/32, 공통 history slice[32,512), true-observed C 응답만 사용, channel별 centered LS, rcond1e-4, FP32 D 저장·배포. E가 바뀌는데 old numerical D를 붙이는 것은 같은 reconstruction 규칙 대조가 아니다. **한 E 안에서는 그 refit D/center/current-C/known-past buffer를 완전히 고정하고** frozen F와 native만 교체한다. pinv-D 실험을 택하면 모든 E에서 동일 pinv rule을 써야 하며 L7 결과와 합치지 않는다.

필수 예측 경로는 다음과 같다.

1. Raw frozen Bolt: 원 C·원 L512/H48·원 mask와 기존 actual-C 점수를 같은 cohort에서 비교한다.
2. 같은 E/D의 frozen Bolt와 native: native는 같은 N×512×K를 받아 K×48 미래를 내며 같은 lag decoder를 거친다. 원 [native_head_v1.py](../../projection_followup_v1/native_head_v1.py)의 stream별 affine512→48은 명시적 대조다. L7를 쓸 때 static-D wrapper를 실수로 재사용하지 않고 현재 L7 native wrapper를 맞춘다. 실제 C의 관측채널 macro MSE로 적합·TUNE 선택하고, 새 E/D에는 새 native를 적합한다. 이전 다른 D의 head를 그대로 'optimized native'로 재사용하지 않는다.
3. 원 C direct ridge, **동일 current feature 대조**: (a)의 a_i∈R^C에서 각 horizon의 C actual-Y를 직접 예측한다. complete-C-per-horizon subset/균등 horizon 가중치, intercept와 FIT-only feature RMS, 사전 고정 alpha grid `{0,1e-4,.001,.01,.1,1,10}`를 쓴다. 각 h에서 `I_h`의 mean/RMS로 a를 표준화한 A를 만들고 `||Y-A B-b||_F^2/n_h + alpha||B||_F^2`를 적합한다. b는 penalty에서 제외하며 alpha0은 rcond1e-8 minimum-norm, 양수 alpha는 그 ridge 해다. 같은 alpha 하나를 48개 h에 적용해 TUNE로 고른다. 이 대조는 E를 고른 current→future 관계가 직접 예측기로 충분한지 답한다. 512 history를 쓰는 F와 입력 범위가 같다는 주장은 하지 않는다.
4. 원 C direct ridge, **동일 history 정보 대조**: F의 필요성을 말하려면 같은 512×C 과거를 받는 직접 ridge도 둔다. 여기서의 `vec(history-last):LC`→HC 회귀는 baseline으로 허용되지만 encoder가 아니다. 원 current-C anchor+intercept를 쓰고 각 실제 observed response column의 mask로 적합한다. 모든 FIT256의 feature별 mean/RMS를 한 번 고정하고, 각 `(h,c)`의 관측 subset 크기 n_hc에 대해 `sum(error_hc^2)/n_hc + alpha||beta_hc||_2^2`를 푼다. 같은 alpha grid/절편 무penalty/minimum-norm 규칙을 쓰고 TUNE macro 점수로 alpha 하나를 고른다. RMS0인 feature는 scale1/학습계수0으로 처리하며 unobserved response를 0 label로 만들지 않는다. coefficient 수/메모리·배포 비용·정규화/alpha 선정까지 별도 기록한다. 이 경로는 K/decoder 출력 공간을 우회하므로 native와 capacity match라고 부르지 않는다.

첫 단계가 비용상 축소돼3까지만 실행되면 '간단한 current 예측기 대조'로 보고하고 **frozen F 고유 가치 검증 완료**로 쓰지 않는다. 4까지 포함한 경로를 기준으로 결론을 내리는 것이 권장 최소 결론 범위다. native/ridge의 FIT 손실은 full observed macro와 complete-C 진단을 분리하고, E의 complete-C fit mask를 전 채널별 관측학습과 같은 표본이라고 서술하지 않는다.

E의 방향 적합은 TUNE로 고르지 않는다. native checkpoint와 ridge alpha는 TUNE actual-C macro MSE로만 사전 규칙에 따라 선택하고 exact tie는 사전 나열 순서로 고정한다. E·D·predictor 선정·mask·alpha를 잠근 뒤 DEV를 평가한다. 이미 노출된 DEV30/146은 독립 holdout이 아니며 TEST를 대신하지 않는다.

## 누수·검증·기록 계약

- 원 FIT mean/std, origin 배열, history/target/observed bytes와 state/metadata bindings를 기존 승인된 source lineage에 연결한다. 전처리·시간 origin·mask를 후보별로 재생성해 비교 cohort를 바꾸지 않는다.
- `o_i-512>=fit_lo`, `o_i+48<=fit_hi`와 source split 경계를 기존 metadata/허용된 actual 단계에서 검증한다. FIT-only는 파일 이름이 FIT라는 뜻이 아니라 모든 E/D 적합의 입력과 응답 timestamp가 FIT 범위라는 뜻이다.
- E 방향·whitening·SVD threshold·center, D, source feature RMS와 ridge coefficient는 FIT에서만 적합한다. TUNE/DEV 미래로 E를 갱신하거나 complete-C 채널 subset을 고르지 않는다. 원 C17/C11 중 결측이 잦은 채널을 빼서 새 C를 만들지 않는다.
- 이번 첫 실험은 FIT 내부 random fold를 쓰지 않는다. 이후 검증을 추가할 때 겹친 `(timestamp,horizon)`/lag pair를 random fold로 나누지 않고 시간순 fold와 purge를 쓴다. 최소한 training response timestamp가 validation response 구간과 겹치지 않아야 한다. 엄격한 전구간 분리가 목적이면 history L까지 포함한 interval 중첩을 purge한다. validation의 과거가 training 기간을 포함하는 것은 실배포와 맞을 수 있으므로 이 두 purge 목표를 혼동하지 않는다.
- fold별 E/whitening/D/scaler까지 refit하지 않고 기존 전체-FIT scaler/E를 재사용하면 조건부 predictor 검증이다. 이를 end-to-end cross-validation 또는 독립 추정으로 부르지 않는다.
- actual mask bytes를 읽지 않은 현재 단계에서는 complete-H origin·timestamp duplicate·source fill 비율을 미확인으로 남긴다. 후속 actual에서 먼저 기록하고 정해 둔 rank/finite 계약이 실패하면 실패 상태를 보존한다. 결과를 본 뒤 mask·K·lag·ridge grid를 바꿔 성공 조건으로 만들지 않는다.
- 직접 수치 검증은 E shape C×K, N×512×K deployment output, FIT latent covariance, 미래/mask 교체 시 배포 예측 불변, FIT 외 데이터로 E 불변, direct RRR factor 재합성(해당 경로만), 같은 E/D의 F/native C 복원식을 포함해야 한다. 현재 문서에서 이 검증을 실행했다고 주장하지 않는다.
- 주요 점수는 원 full-observed channel macro normalized MSE다. MAE·채널별 원단위와 complete-C subset 점수를 보조로 남긴다. complete-C만 유리한 결과를 원 all-observed Raw와 비교하지 않는다. 실제 배포 비용은 physical input→원 fill/scaler→E→F/native→D→physical CPU 반환 전체 경로에서 같은 B1/B8 cohort로 측정한다. cached F fitting 시간은 이 비용을 대신하지 않는다.

## 후속 실행자가 먼저 확인할 미확정 항목

현재 확정한 것은 SOURCE의 목적/축과 JSON summary의 관측 수다. 실제 numeric rank와 교차 singular gap, C predictor→future 관계의 신호 크기, complete-C 선택의 분포 이동, same-E/D native 성능, F 이득 및 전체 비용은 아직 미확정이다. 이 중 어느 것도 문서만으로 성공이라고 예고하지 않는다.

권장 순서는 **(a)의 정의된 cross-moment E 한 개 → 같은 규칙의 E/D 대조 → 같은 E/D F/native와 두 원 C ridge 대조 → TUNE lock 뒤 DEV·전체 비용**이다. cross-moment E가 불리하면 origin/lag/response 방식을 계속 바꿔 최초 가설 성공으로 합치지 않는다. (b)나 exact RRR로 확장할 때에는 그 질문·결측/중복 계약·추가 자유도를 새로 기록한다.
