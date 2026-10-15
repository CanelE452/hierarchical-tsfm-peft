> 게시 사본입니다. 원문 작성 당시의 로컬·준비·미실행 상태는 역사로 보존합니다. 최신 완료/미해결 상태와 게시 범위는 [색인](../README_KO.md)을 확인하세요. 과학 원본은 바꾸지 않고 링크와 이 안내만 추가했습니다.

# 같은 추가 계수의 공통·horizon ramp 디코더 보정 진단

상태: **설계 메모만 작성 / 이 문서의 새 적합·자료/NPZ·모델·F·GPU·비용 실행 0**. 원 실행 SOURCE·결과는 수정하지 않는다. [최상위 방법론 계획](PLAN_METHODOLOGY_COMPLETION_KO_v1.md)과 [기존 의사결정 검토 §2](METHODOLOGY_DECISION_REVIEW_KO_v1.md)의 작은 horizon 개입을 구체화한 것이다. 사용자의 같은 목적에 대한 지속 실행 지시 안의 설계이며 추가 승인 대기가 아니다. 실제 실행은 현재 native/ridge·비용·보고 묶음 검산 뒤 별도 SOURCE 검토를 거친다.

## 먼저 답할 질문과 이미 한 일

최상위 목적은 작은 K 경로가 원 채널 frozen Bolt보다 정확도·전체 비용에서 필요한지를 설명하는 것이다. 이번 좁은 질문은 **같은 E/F·원 RECON D·동일 추가 계수 수에서, 보정이 모든 예측 거리에 같은 크기로 작동해야 한다는 제약을 한 번 풀면 도움이 되는가**다. 재구성 손실을 예측 손실로 처음 바꾸거나, 기존 실패를 모두 재구성 때문이라고 설명하는 실험이 아니다.

[확인: 아래 SOURCE/완료 보고서에서 확인한 범위] 다음 대조는 이미 있다.

- [기존 D2](../../../smallk_failure_followup_20261014/frozen_decoder_refit.py)는 고정 E/F/GELU 특징에 실제 FIT Y 또는 Raw-F teacher를 사용해 **horizon을 합친 공통** RMS ridge 출력 가중치를 적합했다. 예측 목적 D·teacher 자체는 새 시도가 아니다.
- [현재 L0/L7 × RECON/FORECAST 결과](../../publication_main_20261015_v1/reports/causal_lag_objective_v1/REPORT_KO.md)는 같은 E/F에서 공통 D를 과거 복원 또는 실제 FIT 미래 예측으로 적합했다. FORECAST는 TUNE 12/12 개선했지만 DEV는 5/12였고 Raw보다 낮은 DEV 조건은 0/24였다. 이 결과는 목적 변경만으로 문제가 해결되지 않았다는 반대 근거다.
- per-origin C 과거를 적합하는 local/shrinkage D, joint E/D·E-only, 고정 D의 전역 latent scale/bias와 Q, 같은 E/D의 stream별 native는 이미 실행했다. origin별 변화와 horizon별 고정 함수는 다른 축이다.
- [새 직접 ridge 계획](PLAN_NEW_DIRECTION_NATIVE_BASELINES_KO_v1.md)은 과거 전체로 각 (h,c)를 직접 예측한다. horizon별 계수는 있지만 **실제 F 미래·같은 lag 특징·같은 추가 계수의 정적 대조**를 사용하는 실험이 아니다. [진행 중 규제 확장](PLAN_RIDGE_REGULARIZATION_EXTENSION_KO_v1.md)을 이 문서에서 다시 실행하거나 새 실험으로 세지 않는다.

[판단] 이 기록만으로 아래 STATIC_DELTA 대 RAMP_DELTA 대조가 완료됐다고 볼 수 없다. 반대로 “horizon별 복원을 처음 생각했다”는 주장은 할 수 없다. 직접 horizon 모델/잔차 보정은 이미 [Ben Taieb·Hyndman, ICML2014](https://proceedings.mlr.press/v32/taieb14.html)에 있고, frozen univariate FM을 feature adapter로 다변량에 연결하는 틀은 [AdaPTS, ICML2025](https://proceedings.mlr.press/v267/benechehab25a.html)에 있다. 여기서는 새 이름의 신규 방법보다 **현재 실패 설명을 나누는 제한된 진단**을 제안한다.

## 확정 범위: PCA baseK·2K와 Raw 대조

Robin C17/K4·K8, Beijing Aotizhongxin C11/K2·K4, L512/H48, FIT256/TUNE64/기노출 DEV30·146, 같은 origin·causal fill·FIT mean/std·true-observed mask·median F를 유지한다. PCA E/center와 RECON D는 [새 방향 결과](../../projection_followup_v1/runs/predictive_cross_moment_v1/RESULT.json) 및 [독립 감사](../../projection_followup_v1/audits/PREDICTIVE_CROSS_MOMENT_AUDIT_v1.json)의 **실제 FP32 state와 캐시**에 결속한다. 옛 NumPy64 E·다른 backend D를 이름만 보고 대체하지 않는다. ROOT가 기록한 결과 SHA는 `99bffd5831143ca35947099fef9567f89dd209d83451661decee0682b2b8e13c`, 감사 SHA는 `90c68a9057909fb5b6c23284e5a45523cdf126da178a5ba0d9d9e96a50dede30`다.

원 C를 섞은 PCA의 baseK·2K를 모두 사전 고정한다. 새 CROSS/FIT-target/E-only·Q·quantile summary·native를 섞지 않는다. 아래 네 arm을 자료2×K2에 적용해 **16회 잔차 ridge 적합**을 한다.

```text
미래 latent 입력             추가 계수 형태
실제 고정 F 미래             STATIC_DELTA
실제 고정 F 미래             RAMP_DELTA
latent-last 반복 FLAT         STATIC_DELTA
latent-last 반복 FLAT         RAMP_DELTA
```

같은 K의 F/FLAT 대조 없이 D만 유연하게 만들면, 개선이 F 미래 내용인지 과거 buffer 활용인지 나눌 수 없다. FLAT도 origin에서 알려진 같은 K 과거와 current-C를 쓴다. L7의 알려진 prefix 및 lag별 참조 때문에 FLAT 출력은 일반적으로 current-C persistence가 아니며, prefix가 끝난 뒤에도 `sum_l (z_last-z_past[-1-l]) D_l`의 plateau가 남을 수 있다. 원 current-C persistence를 별도 공통 기준으로 기록한다.

**방법 채택을 논할 때 필요한 Raw 대조 8회도 함께 수행한다.** 자료별 원 C F와 원 C FLAT에 STATIC/RAMP를 똑같이 준다. E=I/center=0, 원 D는 lag0=I·다른 lag=0으로 정의하면 F baseline은 같은 Raw F 예측, FLAT baseline은 정확한 current-C persistence다. 배포 baseline은 RawF의 원 `raw_forecast`와 currentC 복사본을 그대로 사용하며 FP32 delta를 다시 더한 값으로 대체하지 않는다. 보정 특징은 같은 7개 lag의 원 C delta이며 새 Raw F를 학습하지 않는다. 확정 묶음은 PCA16+Raw8=**24회 적합**이다. Raw 보정은 추가 7C²개(2023/847), PCA는 7KC개(작은 K 476/154, 큰 K 952/308)로 **서로 용량·정보량이 같지 않다**. 이는 같은 보정 기회를 준 경쟁 기준이며 capacity match가 아니다.

초기 메모의 baseK8회 최소안에서, 실행 전 ROOT가 baseK·2K를 모두 포함하는24회로 확정했다. Robin PCA K8의 실제 미래 E(Y) 진단이 Raw보다 낮아 큰 K에는 다른 여지가 있었으므로, 작은 K만의 부정 결과로 시간 보정을 닫지 않기 위해서다. 이 선택 자체는 이미 본 결과에 반응한 적응적 설계이며 독립 사전 일반화가 아니다. K 증가는 공간·F stream 수·보존 정보를 함께 바꾸므로 같은 K 안의 STATIC/RAMP·F/FLAT 대조를 먼저 읽고, K에 따른 효과와 합치지 않는다. E(Y)의 여지는 실제 과거 예측 가능성이나 이 보정 성공의 보장은 아니다.

## 실제 배포식과 시간 index

origin 현재는 t=0, 실제 미래 시점은 h=1..48이다. 저장 배열 index i=h−1은 0..47이다. lag 집합은 `J=(0,1,2,4,8,16,32)`다. 아래의 ztilde는 시점≤0에서 알려진 과거, >0에서 해당 arm의 F 또는 FLAT 미래를 쓴다.

```text
phi[n,i,l,:] = ztilde[n,h-l,:] - zpast[n,-1-l,:]
base[n,i,:]  = currentC[n,:] + concat_l(phi[n,i,l,:]) @ D_old

STATIC_DELTA: a[i] = 1
RAMP_DELTA:   a[i] = i / 47   (i=0..47, 사전 고정, FP32 vector 저장)

chi[n,i,:]   = FP32(a[i] * phi[n,i,:])
prediction  = base_FP32 + chi_FP32 @ DeltaD_FP32
```

첫 미래 h=1에서 ramp 보정은 0이고 h=48에서 1이다. 이는 origin h=0과 다른 index이며 첫 미래 baseline의 정확성을 보장한다는 뜻도 아니다. g의 기울기·부호·절점·형태를 결과를 보고 조정하지 않는다. 두 arm 모두 추가 slope는 7K×C 하나다. `D_old+DeltaD`와 `D_old+g(h)DeltaD`의 실수 식에 대응하지만, 배포는 위 **원 baseline32+별도 correction32** 연산 순서로 고정한다. `D_old+DeltaD`를 먼저 FP32 합성한 뒤 한 번 곱하는 것과 byte 동일하다고 주장하지 않는다.

이것은 **전 horizon 독립 D_h를 48개 학습하는 실험이 아니다.** full D_h는 계수가 48배 늘어 최소 동일 계수 질문과 섞인다. 또한 static+ramp 두 항을 동시에 학습하면 추가 계수가 2배인 다른 모델이다. 현재 두 arm은 같은 free slope 수만 가지며 rank·effective regularization·함수집합까지 같지 않다. ramp 실패로 모든 horizon별/비선형 D를 기각하지 않는다.

## 같은 FIT 손실·스케일·정규화 선택

각 input arm의 원 baseline은 그대로 고정한다. FIT 정답의 true-observed cell에서만 `r64 = Y32_as64 − base32_as64`를 만들며 미관측 target을 0 label로 쓰지 않는다. response C별 mask로 적합하고, N_c는 FIT 전체 origin×horizon에서 해당 C의 실제 관측수다. 주목적은 현재와 같은 채널 macro다.

```text
L_data = (1/C_active) sum_c [ sum_(n,i:observed) (r64 - (chi32_as64/s) W_c)^2 / N_c ]
L_ridge = L_data + alpha * (1/C_active) sum_c ||W_c||^2
DeltaD64 = W / s[:,None] ; DeltaD32 = FP32(DeltaD64)
```

절편·mean response·별도 시간 bias는 추가하지 않는다. 각 자료·입력 공간(PCA 또는 Raw)에서 **실제 F의 ungated FIT phi32를 전체 origin×horizon에 걸쳐 RMS**한 s를 한 번 고정하고, STATIC/RAMP·F/FLAT 네 arm이 모두 같은 s를 쓴다. 평균을 빼지 않으며 s=0인 열은 scale1·해당 slope0으로 기록한다. FLAT이나 ramp에 각자 RMS를 다시 맞추면 penalty prior까지 바뀌어 다른 비교가 된다. 의도적으로 ungated 기준을 고정하더라도 ramp 열의 크기·유효 rank·shrinkage는 static과 달라질 수 있으므로 spectrum과 norm을 보고하며 완전한 capacity match라고 쓰지 않는다. PCA와 Raw의 s 자체는 서로 다르다.

alpha0는 직접 thin SVD/lstsq의 design singular cutoff1e−8 최소노름, alpha>0는 모든 singular 방향을 쓰는 SSE/N_c ridge로 계산한다. Gram 고유값의 같은 숫자 cutoff로 바꾸지 않는다. per-C mask가 같을 때만 분해를 재사용한다. 무관측 C는 학습되지 않은 correction0, 관측1개·rank 부족·all-zero design은 각각 식별 범위/최소노름을 명시한다. 실제 coefficient64/32 finite 실패는 그 경로를 보존하고 독립 arm은 계속한다.

모든 arm에 동일한 alpha 메뉴 `0,1e-4,.001,.01,.1,1,10,100,1000,10000`과 **ZERO_DELTA** 비학습 후보를 둔다. ZERO는 baseline 배열을 그대로 반환해 byte 동일성을 확인한다. 정확 동률의 순서는 위 alpha 순서 뒤 ZERO다. 후보 선택은 각 arm의 **실제 배포 FP32 TUNE all-observed 채널 macro 정규화 MSE 한 개**로 한다. horizon별 alpha·channel별 alpha·DEV 최저 arm 선택은 없다. alpha10000 상단 선택도 유한 메뉴의 경계로 보고하며 최적 규제 소진으로 바꾸지 않는다.

24회 묶음은 **alpha 후보240개(0 포함)+ZERO24개의264개 후보 비교, FIT/TUNE 최대528회 후보 출력, 선정 DEV24개**다. 양수 alpha만 세면216개이며 alpha0도 양수 ridge라고 쓰지 않는다. 같은 input/space의 ZERO baseline은 static/ramp 사이에서 byte binding으로 재사용 가능하며 실제 새 추론 횟수와 비교 행 수를 구분한다. SVD 횟수는 C별 관측 mask 그룹과 all-zero design에 따라 달라지므로 무조건24라고 쓰지 않는다. 기존 history ridge의 규제 확장/계수/점수를 이 횟수에 넣지 않는다.

## 무엇을 측정하고 어떤 답을 제한할 것인가

1. 모든 후보의 FIT/TUNE, 잠금 뒤 선정 DEV에 all-observed 및 complete-C 정규화 MSE/MAE·per-C·관측수·원단위 진단을 따로 저장한다. h별·사전 1–16/17–32/33–48 구간 오차와 baseline 대비 signed 차이를 원수치로 보인다. horizon 수와 겹치는 창을 독립 데이터셋/독립 반복으로 세지 않는다.
2. 고정 E에서 실제 F와 FLAT의 latent future 오차도 함께 본다. 실제 미래 C 전체가 관측된 h만 안전하게 E(Y)를 구성하고 같은 mask에서 latent별/h별 MSE·MAE를 계산한다. delta-feature 진단이나 L7 C-output E(Y)를 추가할 때는 **현재 target h와 참조되는 모든 양수 미래 h−lag가 complete-C**인 공통 mask로 줄여 F/FLAT/Raw도 같이 점수 낸다. 단순히 target h만 complete라고 oracle 전체를 유효 처리하지 않는다. latent 단위 오차를 C normalized MSE와 같은 단위로 비교하지 않는다.
3. E(Y)는 미래를 아는 원인 진단이며 fitting·TUNE 선정·배포에 넣지 않는다. 새 joint output-space minimum은 1차 필수 측정에서 제외해 **미측정**으로 명시한다. old D의 minimum을 새 D(h)에 재사용하지 않는다. 큰 SVD·새 oracle가 낮아야만 다른 arm을 계속하는 gate도 두지 않는다.
4. STATIC 대 RAMP의 차이는 동일한 추가 slope 수의 고정 시간 modulation 효과다. F 대 FLAT 차이는 같은 보정 class·과거 정보에서 F 미래가 더 유용했는지를 묻는다. 둘이 비슷하게 좋아지면 과거 buffer 기반 readout 개선일 수 있으며 F의 고유 이득이나 표현 학습 성공으로 설명하지 않는다. RAMP만 좋아져도 효과는 이 g와 적합·선정 조건에 한정된다.
5. PCA 보정이 원 Raw F를 넘더라도 보정 Raw가 더 좋으면 압축 고유 정확도 이득을 주장하지 않는다. 반대로 Raw가 좋아지고 PCA가 그대로면 범용 forecast calibration 효과라는 반대 근거다. 실패하면 재구성 D만의 탓·학습량 부족·모든 native 불가능 중 어느 것도 자동 결론이 아니다.

## 실행·검산·중단 규칙

SOURCE/합성 검사에서는 고정 lag reference/known prefix·h=1/48 ramp index·추가 계수 수·ZERO byte parity·observed mask·per-C macro·공통 RMS·직접 ridge 해·FP64/32 fusion을 독립 확인한다. 실제 과학 단계는 이미 수행한104개 F 호출의 해당 캐시를 재사용한 CPU 적합/배포 D 산술이며 새 F forward0이다.104개를 이번 새 호출로 세거나 모든 캐시를 다시 쓰는 개수로 간주하지 않는다. 원 actual source/state/input/array SHA와 원 baseline 재현을 먼저 확인하고, 모든24개 성공/실패 잠금 및 registry 뒤에 DEV 출력/진단을 연다. 이 파일 작성 중 실제 자료는 열지 않았다.

독립 검산은 선정 계수의 LS/ridge 해·실제 FP32 후보 점수·자기 TUNE 선택·원본 보존·같은 mask·등록된 자원 영수증을 확인한다. 저장된 F를 독립 GPU 재실행했다고 말하지 않는다. 기존 실패와 identity/경계 선택도 함께 보고한다. 현재 묶음의 파일을 고치거나 기존 중단 작업을 재개/종료하지 않는다.

**한 번의 고정 ramp 진단 이후 실패를 보고 spline·horizon bin·고차 polynomial을 자동 증가시키지 않는다.** TUNE에서 RAMP가 STATIC보다 유리하지 않으면 이 작은 시간 modulation의 우선순위를 낮춘다. TUNE 이득이 DEV로 옮겨지지 않으면 적응적 설계/과적합 가능성을 남긴다. DEV는 이미 노출된 탐색 자료이므로 원인/일반화의 확정 증거가 아니다. full D_h는 미실행이며 한계를 그대로 기록한다.

채택 판단에는 같은 선정 상태의 physical input→causal fill/FIT scaler→E→실제 frozen F 또는 FLAT→correction→physical CPU 반환 B1/B8 전체 비용이 별도로 필요하다. Raw에도 같은 실행 경계를 준다. cached F를 반환한 과학 단계 시간을 F 비용으로 세지 않는다. 이 메모는 새 비용 SOURCE/실측을 수행하거나 측정했다고 주장하지 않는다. TEST·설치·큰 백본 재학습·자동 Git·논문/PPT 작성은 범위에 없다.
