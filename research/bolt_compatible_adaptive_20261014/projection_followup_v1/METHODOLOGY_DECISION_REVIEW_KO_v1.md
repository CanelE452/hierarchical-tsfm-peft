# 다음 방법론의 판별 질문과 최소 대조

상태: **읽기 전용 전략 검토와 설계 문서 / 새 자료·NPZ·모델·F·GPU·학습·타이머 실행 0**. 이미 실행된 실험과 아래 가설을 구분한다. 새 방법의 성능이나 신규성을 입증한 문서가 아니다. 사용자의 지속 연구 지시 범위 안에서 작성했으며 추가 승인 대기를 뜻하지 않는다.

목적은 작은 K의 정확도와 실제 전체 E/F/D 비용을 함께 유지하는 방법을 교수님께 설명하고 판단할 수 있게 만드는 것이다. 다음 실험은 중간 지표를 더 만드는 것보다 **정보·복원 공간의 제약, 고정 F가 쓰는 좌표, decoder의 horizon 제약 중 무엇을 바꿔야 하는가**에 답해야 한다. 추천 순서는 1 → 2이고 3은 대안이다.

## 이미 배제해야 하는 설명

- [확인] 실제 Y·Raw-F teacher·latent future teacher, frozen-F D2 ridge, 직접 C 잔차, joint E/D, E-only, stop-gradient, matched native가 이미 실행됐다. “이제 예측 목적/teacher/native를 처음 해보자”는 새 가설이 아니다. [기존 작은 K 보고](../../smallk_predictive_alignment_20261013/README.md), [좌표 호환성 검토](../../smallk_failure_followup_20261014/FORECAST_COORDINATE_COMPATIBILITY_REVIEW_KO_v1.md), [현황 v8](CONTINUATION_STATUS_KO_v8.md).
- [확인] 현재 K4/L7의 matched 독립 affine512→48 native 6조건은 TUNE/DEV 모두 같은 E/D의 Bolt보다 나빴다. 이 head·손실·최적화 경로의 결과이며 모든 native가 불가능하다는 증거는 아니다. [완료 보고](reports/matched_l7_native_v2/REPORT_KO.md).
- [확인] 전역 scale4/scale+bias8 직접 보정의 6조건 중5개는 identity, E2 scale-only만 λ10이었다. 그 한 조건은 TUNE .10092611978772857→.10091457870111374, DEV .5052500038642814→.5054581188842874였다. 작은 전역 affine 보정이 실패한 것이며 F 앞 mixing, horizon interaction, 모든 비선형 보정을 배제하지 않는다. [실제 JSON](runs/fixed_l7_latent_calibration_v1/RESULT.json), [독립 과학 감사](audits/FIXED_L7_LATENT_CALIBRATION_AUDIT_v1.json).
- [확인] 고정 E/D의 complete-C TUNE retained numeric minimum은 E0/E1/E2에서 .0609840323/.0597340174/.0595795763으로 Raw .0569510909보다 높다. DEV minimum은 .3068850413/.3069056211/.3111773358로 Raw .3404018870보다 낮지만 실제 Bolt는 더 나쁘다. 따라서 TUNE 공간 제약과 DEV의 실제 미래 예측 격차를 같이 설명해야 한다. 이 수치는 D32·known-past offset·전체 HK 미지수·complete-C 행·rcond1e-15 retained 공간에 조건부다. 모든 실수/FP32 출력의 보편 하한으로 바꾸지 않는다.

## 1. 같은 정보와 old D를 보존한 F 좌표 호환성

**교수님 첫 질문:** “이미 E/D를 학습했는데 왜 다시 회전인가? 무엇을 고정해서 원인을 분리하는가?”

[가설] 기존 joint 학습은 E가 보존하는 정보, D 공간, 스케일과 최적화 경로를 함께 바꿨다. 반면 orthogonal Q의 conjugation은 그 변화를 막고, **동일한 K stream 정보를 고정 F가 어느 기저에서 잘 예측하는가**만 시험한다. 일반 E/D 학습의 제한된 부분집합이지만 원인 분리의 의미는 다르다.

행벡터 convention에서 현재 과거 z=(X-center)E와 lag별 D_l를 쓰자. 고정 QᵀQ=I에 대해 E_Q=EQ, D_Q,l=QᵀD_l이다. 과거·미래·각 lag 원점 참조를 모두 함께 변환하면

\[
(z_{h+1-l}Q-z_{-1-l}Q)Q^T D_l
=(z_{h+1-l}-z_{-1-l})D_l.
\]

따라서 [수학 확인] 재구성, lag별 원점, known-past offset, joint 미래 출력 공간은 실수 대수에서 보존된다. center는 원 C 좌표 그대로다. 원점 참조를 모든 lag에서 z_last로 바꾸면 다른 실험이 된다.

실제 구현은 D_Q를 다시 FP32로 저장하기보다 다음 형태가 더 분명하다.

\[
u_Q=F(z_{past}Q)Q^T,\quad
\hat Y_Q=\operatorname{old\_lag\_decode}(C_t,z_{past},u_Q;D).
\]

old E/D/center·과거 buffer·current-C를 그대로 두고 미래 u만 교체한다. h를0부터 세면 원 예측과의 실수 대수 차이는

\[
\Delta\hat Y_h=\sum_{l\le h}
\left[F(zQ)_{h-l}Q^T-F(z)_{h-l}\right]D_l.
\]

미래 좌표의 commutator 차이가 D에서 지워질 수도 있으므로 latent 차이와 최종 C 차이를 함께 본다. 차이가 크다는 사실만으로 어느 쪽이 더 정확한지, 현재 실패의 주원인인지 결정할 수 없다.

**가장 작은 결정 실험.** 원 PCA/RECON인 E0부터 identity, 고정 permutation, 고정 dense orthogonal Q 두 개를 비교한다. identity와 permutation은 같은 shared per-stream F의 배치·좌표 연결 검사다. sign flip은 선택적인 대칭 진단이며 dense mixing과 같은 질문으로 합치지 않는다. 새 D 적합은 하지 않는다. cached 원 u를 uQ로 transport했다가 되돌리는 대조에서 원 C 재현을 먼저 확인하고, 실제 F(zQ)로 교체한 뒤에만 성능 차이를 계산한다. FP32의 행렬 결합 순서·batch 차이는 별도 오차이며 실수 항등식을 byte-exact 보장으로 쓰지 않는다. 실제 tolerance와 실패 보존 계약은 결과를 보기 전에 고정한다.

다음 학습형 후보는 Q=exp(S), Sᵀ=-S의 K(K−1)/2개 계수만 FIT actual-C macro MSE로 갱신한다. K4는6개, K8은28개다. Cayley도 가능하지만 단일 chart는 고유값−1을 포함하는 Q를 직접 표현하지 못한다. orthogonality·초기 identity·F 입력 gradient·old E/D 불변·무보정 후보를 검사한다. F 가중치 동결은 F 입력 gradient 제거가 아니다. 학습 Q는 작은 비선형 최적화이므로 global optimum이나 “optimizer 부족 배제”를 주장할 수 없다. TUNE는 checkpoint/Q 선택만, DEV는 선택 잠금 뒤 진단에 쓴다.

**반대근거와 결정.** Q는 old D 출력 공간을 바꾸지 않는다. 따라서 등록된 retained 공간에서 TUNE minimum>Raw인 반대근거를 그대로 가진다. Q만으로 Raw gap 전체가 해결될 것이라고 예고하면 안 된다. fixed 두 Q가 실패해도 모든 Q가 무효라고 결론내릴 수 없다. 학습 Q까지 실제 TUNE 이득 없이 identity를 고르면 이 고정 공간의 basis 탐색 우선순위를 낮추고 같은 계열의 seed/각도 sweep을 늘리지 않는다. 이득이 나도 외부 일반화와 비용은 별도 검증 대상이다.

**K8과의 관계.** K4→K8은 공간·보존 정보·D 적합·F stream 수가 함께 바뀌는 capacity 실험이다. [K4,K8]×[identity,Q]를 나란히 두면 같은 K 안의 Q 효과와 K 증가 효과를 분리해 읽을 수 있다. K8은 현재 K의 사전 doubling 대조이며 최적 K라는 근거는 없다. 새 PCA/D를 적합하면 known-past offset도 달라지므로 minimum이 자동으로 단조 개선된다고 보장하지 않는다. K8 actual identity와 새 공간 진단을 먼저 보고 Q의 역할을 해석한다. K8의 정확도 개선을 K4와 같은 압축률의 성공으로 합치지 않는다.

**기존과의 차이/선행 반증.** [Oct11 회전 protocol](../../level_compressor_fullscreen_20261011/ROTATION_PROTOCOL.json)은 RobinK5/JenaK6, 고정 Q 두 개와 새 F 및 D 재적합이었다. 현재 K4/L7의 old-D conjugation과 같은 원인 대조가 아니다. 그러나 회전 자체는 이미 해봤으므로 첫 회전 실험이라고 쓸 수 없다.

[AdaPTS](https://arxiv.org/html/2502.10235v1)는 frozen univariate FM 앞뒤의 feature-space adapter를 학습하며, appendix D.1에는 F(XW)W⁻¹ 형태의 목적도 명시한다. 우리 Q conjugation은 이 기존 개념의 제약된 경우로 보아야 한다. [ORTCL, AAAI2026](https://ojs.aaai.org/index.php/AAAI/article/view/39526)도 streaming model editing의 input/output orthogonal 변환을 다룬다. 그 input SVD·output least-squares/Procrustes 구성과 현재 전체 F의 같은-Q 역변환은 세부가 다르지만, orthogonal adapter 자체의 신규성 주장을 배제하는 강한 선행이다. [공식 Bolt 소스](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos_bolt.py)는 stream별 정규화와 inverse를 보여주며, 이것만으로 cross-channel mixing equivariance가 생기지는 않는다. upstream main은 과거 로컬 실행본의 byte 동일성 증거가 아니다.

## 2. horizon이 달라도 같은 D_l을 쓰는 제약

**교수님 첫 질문:** “local D와 forecast D도 이미 했는데 dynamic decoder라는 이름만 바꾼 것 아닌가?”

[가설] 현재 L7은 horizon 전체에 같은 7개 D_l를 쓴다. 같은 실제 F 미래에서 horizon에 따라 C와 잠재 변화의 관계가 달라지면, 전역 q나 공유 D refit으로는 그 제약을 풀지 못한다. 이 가설은 F의 과거 예측 가능성과 별개이며 decoder 편향을 시험한다.

첫 개입은 큰 hypernetwork 대신 사전 고정한 scalar g(h)와 D_l(h)=D_l+g(h)ΔD_l이다. 예를 들어 g는 48개 horizon의 고정 선형 ramp이고, 새로운 계수는7KC=476개다. 동일476개를 갖는 horizon-invariant D_l+ΔD_l도 반드시 같이 적합한다. 같은 frozen E/F, 실제 FIT future response, 관측 채널 macro 가중치, FIT-only column RMS, identity/zero correction과 같은 ridge grid를 사용한다. ΔD는 old D 주변의 잔차이며 future mean/intercept를 별도로 추가하지 않는다. 실제 FP32 decode를 TUNE 선정 대상으로 삼는다. g(h)의 정의·h0 indexing은 결과를 보고 바꾸지 않는다.

두 arm의 열 수가 같아도 rank·scale·effective prior까지 같지는 않다. 각 feature singular values와 적합 상태를 남겨 “완전 capacity match”라고 과장하지 않는다. global+linear 두 항을 동시에 학습하면952개로 늘어나는 다른 모델이다. 한 번의 horizon interaction이 개선되지 않으면 spline 수·horizon bin을 계속 늘리는 방향은 재평가한다.

**핵심 대조.** actual F 미래와 latent-last/등록된 단순 미래를 같은 새 decoder 규칙 아래 비교한다. 둘 다 비슷하게 개선되면 decoder가 과거 prefix/current-C를 이용한 효과일 수 있어 F의 고유 가치를 입증하지 못한다. 원 Raw에도 같은 종류의 미래 보정 대조가 없으면 작은 K의 고유 이득이라고 부르지 않는다. 새 D(h)의 joint 공간/known-past offset 진단은 원 D minimum을 재사용하지 않고 다시 정의해야 한다.

현재 q는 고정 D 안의4/8계수 미래 보정이고 이 개입은 D 공간 자체를 바꾸므로 동일하지 않다. 반면 horizon 독립 ΔD는 기존 D2/forecast-D refit의 현 L7 matched 재대조에 가깝다. per-origin 과거 C로 D를 적합하는 local D는 이미 실행됐다. C 과거를 받는 dynamic hypernetwork를 추가하면 decoder 정보량과 비용까지 늘어나므로 “예측기만 변경”으로 해석할 수 없다.

[TiDE](https://arxiv.org/html/2304.08424v3)는 horizon별 decoded vector와 그 시점의 알려진 covariate를 결합하는 temporal decoder를 사용한다. 이는 시간별 decoder가 기존 설계임을 보여주지만, 우리 자료에서 미래 covariate가 합법적으로 주어진다는 근거는 아니다. [CoRA, ICLR2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ae3e173398d6e43fea63cbcc16fbaa98-Abstract-Conference.html)도 time-varying/invariant correlation과 learnable polynomial adapter를 다룬다. 따라서 “horizon/dynamic decoder”라는 넓은 이름을 신규성으로 쓰지 않는다.

## 3. 큰 미래 분산 대신 과거로 예측 가능한 C 방향

**교수님 첫 질문:** “FIT 미래 PCA와 teacher까지 실패했는데 표현을 또 바꿀 근거는 무엇인가? 직접 작은 선형 예측기면 충분하지 않은가?”

[가설] FIT 미래 innovation 자체의 큰 분산 방향은 예측 불가능한 변동도 포함한다. 과거→미래의 예측 가능한 교차 구조로 K 방향을 정하면 다른 출력 공간을 얻을 수 있다. 이는 모든 actual-Y 학습의 첫 시도가 아니라, marginal future variance와 predictable covariance의 구별이다. [현재 FIT-target SOURCE](future_target_projection_v1.py)는 complete-C FIT의 Y−current-C SVD이며 과거→미래 예측 회귀를 방향 선택에 쓰지 않았다. ForeCA와도 기준이 같지 않다.

작은 결정 실험은 고정 past feature로 FIT-only 다중 horizon ridge/축소 rank 회귀를 먼저 적합하고, 그 예측 가능한 C 방향을 K개 택하는 것이다. 같은 rank의 미래-Y PCA·과거 PCA와 비교하되, 같은 past feature의 직접 C ridge/RRR 예측도 함께 보여야 한다. 직접 작은 모델이 같은 표현의 frozen F보다 좋다면 “F 재사용의 이점”은 지지되지 않는다. N1 teacher→미래 latent, N2 과거 E→C 직접 ridge의 기존 결과와 source/objective가 실제로 다른지 먼저 확정한다. 차이가 없으면 새 조건으로 만들지 않는다.

선형 RRR를 horizon까지 flatten하면 latent state 하나의 rank K와 기존 K개 시간 stream의 HK 좌표는 다른 병목이다. 그대로 F에 넣을 수 있다고 간주하지 않는다. E를 pointwise C×K로 유지할지 temporal encoder로 바꿀지 먼저 정의해야 하며, 후자는 encoder 입력 범위·연산량을 함께 바꾼다. FIT 내 교차 예측으로 방향을 정할 때 중첩 target timestamp를 random fold에 나누지 말고 시간순 purge와 fold-local 통계를 사용해야 한다. 기존 전체 FIT 정규화/표현을 재사용하면 완전한 end-to-end 교차 검증이 아닌 조건부 진단임을 남긴다.

[Low Rank Forecasting](https://web.stanford.edu/~boyd/papers/pdf/low_rank_forecasting.pdf)은 과거에서 latent state를 추정하고 미래로 decode하는 저차원 선형 예측과 시간순 검증을 다룬다. [Box–Tiao의 원논문](https://pages.stern.nyu.edu/~dbackus/BCZ/HS/BoxTiao_canonical_Bio_77.pdf)도 선형 조합을 predictability로 구별한다. 다만 다변량 과거로 잘 예측되는 축이 그 한 축의 과거만 받는 frozen univariate F에도 잘 예측된다는 보장은 없다. 이 불일치를 덮으면 새로운 표현 실험도 같은 질문을 반복한다. 따라서 이 가설은 Q/단순 horizon 대조보다 우선순위가 낮다.

## 교수님께 제시할 결정 구조

1. “공간을 보존해도 F가 기저에 민감한가?”에는 old-D conjugate Q 대조가 답한다. 재구성 항등식은 보조 증거이고 실제 C 정확도가 판정 대상이다.
2. “공간을 바꿔야 하는가?”에는 K4/K8의 actual identity·새 공간 진단과 작은 horizon-D 대조가 답한다. capacity 증가와 basis 선택을 하나의 효과로 합치지 않는다.
3. “왜 frozen F인가?”에는 같은 E/D와 같은 학습 label/선정 규칙의 직접 작은 predictor 대조가 답한다. 현재 native6조건의 부진만으로 그 질문이 끝난 것은 아니다.

어떤 가설도 현재 exposed DEV를 새 독립 일반화 자료로 바꾸지 않는다. 성능·선택의 방향을 보고 다음 설계를 고르는 적응적 탐색임을 남긴다. 실제 비용은 물리 입력→E→actual F/단순 predictor→D→물리 CPU 반환 전체 경로에서 새 상태별로 측정해야 하며, cached F를 이용한 과학 fitting 시간과 혼합하지 않는다. 새 방법 주장은 관련 adapter 선행과의 차이, Raw 정확도, 비용, 독립적인 평가 설계가 모두 갖춰진 뒤에 판단한다.
