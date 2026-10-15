# 같은 E/D 안에서 F 미래의 작은 보정으로 남은 격차를 줄일 수 있는가

후속 사전 설계이며 본 문서 최초 작성 당시 SOURCE·실제 적합·새 추론·비용은0이었다. **현재 matched L7 native의 비용·독립 검산·보고까지 완료한 뒤 실제 적합·추론을 실행한다.** 검산된 상태를 쓰는 순수 SOURCE·합성 준비는 보고 검산과 병행할 수 있다. 이전 작업을 대체하거나 중단하지 않는다. 같은 연구 목적·자료·주지표의 다음 원인 대조에 대한 작업 판단이며 새 개별 승인을 받았다고 쓰지 않는다.

## 좁은 질문과 이미 한 일

같은 고정 L7 RECON D에서 단순 native6조건은 실제 Bolt보다 나빴지만 Bolt 역시 Raw보다 나빴다. native는600update·정해진5CP 중 모두CP100을 골랐고 gradient clipping도 빈번했다. 이 결과로 모든 native·충분 수렴·Bolt 필수성을 확정하지 않는다. 무조건 장기 학습 전에, **현재 공간을 바꾸지 않는 작은 F 출력 보정**으로 설명되는 오차가 있는지 살핀다.

기존 전체 D forecast-LS24fit·D2 frozen-F ridge·MAIN direct-C residual ridge·고정 teacher q는 실제 있었다. 현재 L7 D를 고정한 global latent scale/bias identity-ridge는 해당 실제 lineage에 없다는 읽기 전용 조사를 마쳤다. frozen-F calibration이나 예측 목적 표현의 첫 시도로 부르지 않는다.

## 고정 계약과 보정식

Robin C17/K4·L512/H48·FIT256/TUNE64/기노출 VAL30, 같은 raw input/origin/scaler/mask·현재 C anchor·lag0/1/2/4/8/16/32를 유지한다. 원 RECON CP0와 E-only 선정 두 상태의 E/D/center를 고정하고 검산된 실제 Bolt latent/history 및 현재 D로 decode한 C 예측을 재사용한다. native head 가중치는 이 적합에 넣지 않는다. 새 E/D/F 학습·optimizer·F model load/forward는0이며 FIT 미래는 정상 회귀 response로만 사용한다.

F 미래를 u, origin의 마지막 잠재를 z_last라 할 때 다음 두 family를 각각 적합한다.

```text
scale-only(P4):  u' = u + a_k (u - z_last)
scale+bias(P8):  u' = u + a_k (u - z_last) + b_k
identity:       a=b=0
```

a/b는 origin·미래 horizon에 공유된다. 별도 C intercept나 새 D는 없다. 원 decoder는 lag마다 **z_past[-1-lag]**를 빼며 모든 lag에서 z_last를 빼는 식이 아니다. 과거 prefix와 lag별 origin reference는 그대로 둔다. 따라서 h를0부터 셀 때 실제 미래만 바꾸는 설계 열은 다음과 같다.

```text
phi_a[n,h,c,k] = sum_l 1[h>=l] (u[n,h-l,k]-z_last[n,k]) D_l[k,c]
phi_b[n,h,c,k] = sum_l 1[h>=l] D_l[k,c]
```

회귀 baseline은 실제 cached F→C 예측이며 잘못된 공통 origin reference로 다시 만들지 않는다. 실수식에서 a=-1,b=0은 잠재 미래를 z_last로 만들지만 C persistence를 뜻하지 않는다. 과거 prefix가 끝난 뒤에도 currentC+sum_l(z_last-z_past[-1-l])D_l의 plateau가 남는다. 실제 FP32의 u-(u-z_last)는 cancellation 때문에 z_last와 byte-exact임을 보장하지 않으므로 실제 산술 차이는 별도로 기록한다. 영계수를 산술 없이 반환하는 identity 경로의 byte 보장과 구분한다.

## 적합·선정·수치 검산

FIT의 원 관측 위치만 펼친다. 채널 관측 수 N_FIT,c와 활성 채널 수 C_active로 w=1/(C_active*N_FIT,c)를 사용한다. 관측 행의 w 합은1이므로 FIT macro MSE의 weighted-mean 목적이다. r=Y-실제 cached baseline, X=phi, s_j=sqrt(sum_i w_i X_ij^2)를 FIT에서만 계산한다. 열이나 response를 중심화하지 않는다. A=sqrt(w)X/s, r_w=sqrt(w)r에서 ||A theta-r_w||²+lambda||theta||²를 SVD로 적합하고 원 계수는 theta/s다. 이 규제는 identity 중심이며 표준화된 계수에 적용된다.

lambda=[0,1e-6,1e-4,1e-2,.1,1,10]을 미리 고정한다. 각3상태×2family의6개 적합 조건, 최대6SVD와42계수 후보,6identity 비교를 구분한다. 정확0인 관측 FIT feature 열만 제거하고 원 위치에0을 재삽입한다. 관측 FIT design 전체가0이면 SVD를 호출하지 않고7개 영계수 후보와 identity를 보존하며 실제 SVD 수를0으로 기록한다. lambda0의 relative cutoff1e-12와 양수lambda의 sigma/(sigma²+lambda)를 구분한다. 이것은 기존 출력 공간 minimum의 cutoff1e-15를 바꾸는 것이 아니다. rank·singularvalues·제외 열·scales·계수 크기/음수 gain을 보존하며 계수를 결과에 맞춰 clip하지 않는다. rank가 부족하면 계수 자체를 고유한 잠재 의미나 인과효과로 해석하지 않는다.

TUNE 선정은 실제 CUDA FP32 q+원 lag D의 C 예측을 원 TUNE 자체 관측 수로 계산한 macro MSE로 한다. FIT 관측 수를 TUNE/DEV 평가 분모로 재사용하지 않는다. identity는 같은 원 baseline을 byte-exact로 재현하는 별도 경로다. 정확 동률이면 identity 우선, 그 다음 큰lambda 순으로 정한다.6성공/실패 잠금·registry 뒤 DEV를 연다. 비유한 후보는 해당 경로 실패로 보존하고 다른 후보/family/state는 계속한다.

계수 적합은 FP64 선형 surrogate이고 배포 계산은 FP32다. FIT/TUNE의 actual FP32와 baseline+X계수 사이 오차를 별도 저장하며0이라고 가정하지 않는다. 숫자 불안정/검산 공차 계약은 SOURCE를 동결하기 전에 합성 stress fixture로 정하고 실제 결과 뒤 완화하지 않는다. 독립 직접 lag decode와 설계 열·lag 경계0/1/31/32·불균등 mask의 알려진 해·0열/비례열/near-rank·가려진 label 불변·FIT scaling/계수의 TUNE/DEV label 불변·identity tie/plateau·FP32 실제 경로를 검증한다. 미래 oracle나 DEV를 계수/선정에 넣지 않는다.

## 비용과 해석

이미 검산된 같은 complete-C TUNE의 retained numeric minimum은 E0 .06098403233470353, E1 .05973401737367974, E2 .05957957628725291로 Raw .05695109093352815보다 높다. DEV minimum은 반대로 Raw보다 낮다. 따라서 이 보정은 현재 고정 D 안의 격차를 얼마나 줄이는지 확인하는 원인 대조이며 TUNE에서 Raw를 이길 수 있다고 전제하지 않는다. 이 수치는 모든 실수/FP32 경로의 보편 하한이나 과거 예측 가능성의 증명이 아니다. 보정 실패를 근거로 다른 E/D 또는 K의 가능성까지 닫지 않는다.

계획이 실제 진행되면 선정 q6+같은 E/D의 Bolt3+Raw1을 같은 물리 TUNE B1/8·warm2·seed20261015의5순환으로20셀100표본 측정한다. q는 H×K 원소 연산을 추가하며 F/E/D 비용이 계속 필요하므로 무료 보정이라고 하지 않는다. 현재 native 비용과 다른 cohort의 timing을 차용하지 않는다. 실제 비용 SOURCE는 현재 묶음이 끝난 뒤 별도로 준비한다.

Raw를 이기지 못하면 연구 목표 해결이 아니다. 작은 보정의 실패도 모든 forecast-compatible 표현의 불가능성을 뜻하지 않는다. 모든6조건·42후보와identity·원수치/실패·자원·한계를 MD/CSV/그림으로 남긴다. 독립 일반화가 아닌 기노출 DEV의 적응적 탐색이며 strict K-only도 아니다. TEST·설치/다운로드·큰 백본 학습·기존 작업 종료/중단 학습 재개·자동 commit/push·논문/PPT 금지는 유지한다.
