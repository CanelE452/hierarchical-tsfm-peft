# comparative_ablation_followup_20261027_v1 — 사전등록 추가본

작성: 2026-10-27T10:55:17.039363+09:00

기준 커밋: `06f142dbba1ab0370fb69366eb6cd28b721d5505`

새 학습 전 작성 및 로컬 커밋. 원격 push는 별도 사용자 확인 후에만 수행한다.

## 범위 선택의 시점과 해석

The subset was chosen after inspecting the first campaign results, to fill the D-whole-structure ablation gap and the provisional Jena No-P result under the available time. Job definitions, seeds, original decision rules, and original training code remain fixed. This is a disclosed post-hoc scope selection, not an outcome-independent choice of which experiments to run.

A4는 D 구조 전체의 효과이며 마지막 관측값과 lag를 분리하지 않는다. V는 개발 평가이며 보호 T 숫자를 읽지 않는다.

## 예산의 실측 갱신

A4 40개: 6.242085시간. A1/Jena 연장 10개: 6.981477시간. 총 13.223562시간 [추정].

지시문의 약 11.3시간은 이전 K_tau 프로파일의 연장 추정에 근거했다. 이번에는 첫 캠페인 A1 실측 search/refit 시간과 실행 부대 시간을 반영한다. A4만으로 13시간 이내이므로 FG6은 통과한다. A4 종료 시 원래 13시간 시계와 2026-10-28 00:00 KST 마감 안에 연장 10개가 모두 끝날 것으로 추정될 때만 묶음으로 진입한다. 그렇지 않으면 10개 모두 NOT_RUN_BUDGET이다. 진행 중인 job은 마친다.

## 학습 전 D0 결과

```csv
dataset,K,C,status,A60_fit_end,A60_rows,A60_explained_variance,A60_floor_MSE,V_floor_MSE,V_energy_MSE,V_energy_outside_ratio,V_optimal_rankK_floor_MSE,V_mean_shift_vs_A60_causalfilled_mean_outside_MSE,V_mean_shift_vs_A60_observed_scaler_mean_outside_MSE,V_mean_shift_vs_PCA_center_outside_MSE,C1_actual_same_cohort_MSE,C1_floor_fraction,total_points,complete_points,excluded_points,excluded_fraction,rank,rank_tolerance,singular_values,prediction_space_residual_MSE,prediction_space_residual_limit,lower_bound_tolerance,ref_RAW_MSE,ref_RAW_n,ref_RAW_extended,ref_C1_MSE,ref_C1_n,ref_C1_extended,ref_C2_MSE,ref_C2_n,ref_C2_extended,ref_C3_MSE,ref_C3_n,ref_C3_extended,ref_OURS_MSE,ref_OURS_n,ref_OURS_extended,ref_A1_MSE,ref_A1_n,ref_A1_extended
jena,6,21,PASS,12579,12579,0.9088954938978033,0.09110450616819522,0.17473377007276672,1.7812058562879827,0.09809858274153127,0.0830917608024298,0.0034782742251612177,0.0034782693915226424,0.0034782742251611826,1.0794886734636069,0.16186716393431205,2064,2064,0,0.0,6,4.6629367372484656e-15,"[1.0000000072535422, 1.0000000017066075, 0.9999999978010584, 0.9999999931762075, 0.9999999919005058, 0.9999999813594924]",2.8939619958931442e-15,2.34134245324924e-09,4.497436158658862e-08,0.8491941389472366,1,False,1.079488673463609,1,False,0.7150672744180276,10,False,0.7157531985161767,10,False,0.6977704154364386,10,False,0.7661766256358824,10,False
ETTh1,4,7,PASS,10452,10452,0.920902503908644,0.07909749545758728,0.46137119957766604,1.0486954431510844,0.43994774897786676,0.008380127262615052,0.42393875764270667,0.4239387520034713,0.42393875764270766,0.7157426458978436,0.6446048760988827,1728,1728,0,0.0,4,1.5543122628030744e-15,"[1.0000000182253312, 1.0000000109061635, 0.999999990941056, 0.9999999740385206]",4.867463906247668e-16,5.598986829724595e-10,2.997138140633676e-08,0.28340453716596214,1,False,0.7157426458978442,1,False,0.6783253016312557,10,False,0.6661532687941905,10,False,0.2796426202734389,10,False,0.27922592486593145,10,False
ETTh2_safe17400,4,7,PASS,10440,10440,0.9331520953231541,0.06684790479481184,0.6919420115123488,3.2351327108563814,0.2138836558977461,0.012011489500532649,0.6255406147203003,0.6255406411905314,0.6255406147202999,0.9019343168197984,0.7671756120247447,1728,1728,0,0.0,4,1.5543122576778948e-15,"[1.0000000149279373, 1.0000000130537563, 0.9999999933121944, 0.9999999729653372]",4.355788363537078e-15,2.5914146384309333e-09,1.0979897822437694e-07,0.2595548743245539,1,False,0.9019343168197952,1,False,1.0911035753073155,10,False,1.1373325400074006,10,False,0.27723074241953666,10,False,0.263364830221251,10,False
appliances,10,26,PASS,11841,11841,0.9102052500190577,0.08979474997919101,0.1346360977566276,1.397011742984494,0.09637434934440776,0.04711373391041925,0.03974719945675774,0.03974721035857187,0.03974719945675842,0.41731451276976445,0.3226250073668235,1968,1968,0,0.0,10,5.773159830381884e-15,"[1.000000017725314, 1.000000008886319, 1.0000000047302275, 1.0000000013575372, 0.9999999997807147, 0.9999999941699032, 0.9999999926088651, 0.9999999850596131, 0.9999999716133186, 0.9999999684177852]",4.928455309004279e-15,6.753928971647838e-09,5.1518861419486613e-08,0.347579137172896,1,False,0.41731451276976345,1,False,0.32493083911635734,10,False,0.3326621451263595,10,False,0.2820186407011106,10,False,0.283611553958266,10,False
```

## 기존 C2/C3의 학습 전 D1 요약

```csv
dataset,method_id,n,expected_n,complete,skipped_n,status,aggregation,floor_MSE_median,actual_same_cohort_MSE_median,floor_fraction_median,excluded_fraction_median,prediction_space_residual_MSE_median
jena,C2,10,10,True,0,PASS,median of primary per-job values; floor_fraction is median of ratios,0.23394020994944809,0.7150672744180272,0.3272373945193644,0.0,3.0099597083953005e-15
jena,C3,10,10,True,0,PASS,median of primary per-job values; floor_fraction is median of ratios,0.2369005951001198,0.7157531985161765,0.3300434416061121,0.0,3.1337815429486736e-15
ETTh1,C2,10,10,True,0,PASS,median of primary per-job values; floor_fraction is median of ratios,0.422568160143868,0.6783253016312547,0.6273893658704971,0.0,5.843683626917032e-16
ETTh1,C3,10,10,True,0,PASS,median of primary per-job values; floor_fraction is median of ratios,0.4137835664582573,0.6661532687941896,0.6295552115427436,0.0,5.145487733394556e-16
ETTh2_safe17400,C2,10,10,True,0,PASS,median of primary per-job values; floor_fraction is median of ratios,0.7732298486640334,1.091103575307315,0.7086150147743007,0.0,2.1004126600114413e-15
ETTh2_safe17400,C3,10,10,True,0,PASS,median of primary per-job values; floor_fraction is median of ratios,0.7983225027161247,1.1373325400074017,0.6922663094775943,0.0,1.9283144170439835e-15
appliances,C2,10,10,True,0,PASS,median of primary per-job values; floor_fraction is median of ratios,0.15801456960739008,0.324930839116358,0.48500222207384397,0.0,2.239182284017291e-15
appliances,C3,10,10,True,0,PASS,median of primary per-job values; floor_fraction is median of ratios,0.15544905123552766,0.33266214512636033,0.4717223507909152,0.0,2.1778989637714284e-15
```

독립 최소제곱 검산과 원 파일 보존 점검은 PASS이며 원본 CSV/검증 영수증 SHA는 JSON에 기록한다.

## 실행 목록과 순서

A4는 seed별 4개 데이터를 순회한 뒤 A1/Jena 연장 10개를 조건부 실행한다.

- `CA_A4_jena_K6_s92601`
- `CA_A4_ETTh1_K4_s92601`
- `CA_A4_ETTh2_safe17400_K4_s92601`
- `CA_A4_appliances_K10_s92601`
- `CA_A4_jena_K6_s92602`
- `CA_A4_ETTh1_K4_s92602`
- `CA_A4_ETTh2_safe17400_K4_s92602`
- `CA_A4_appliances_K10_s92602`
- `CA_A4_jena_K6_s92603`
- `CA_A4_ETTh1_K4_s92603`
- `CA_A4_ETTh2_safe17400_K4_s92603`
- `CA_A4_appliances_K10_s92603`
- `CA_A4_jena_K6_s92604`
- `CA_A4_ETTh1_K4_s92604`
- `CA_A4_ETTh2_safe17400_K4_s92604`
- `CA_A4_appliances_K10_s92604`
- `CA_A4_jena_K6_s92605`
- `CA_A4_ETTh1_K4_s92605`
- `CA_A4_ETTh2_safe17400_K4_s92605`
- `CA_A4_appliances_K10_s92605`
- `CA_A4_jena_K6_s92606`
- `CA_A4_ETTh1_K4_s92606`
- `CA_A4_ETTh2_safe17400_K4_s92606`
- `CA_A4_appliances_K10_s92606`
- `CA_A4_jena_K6_s92607`
- `CA_A4_ETTh1_K4_s92607`
- `CA_A4_ETTh2_safe17400_K4_s92607`
- `CA_A4_appliances_K10_s92607`
- `CA_A4_jena_K6_s92608`
- `CA_A4_ETTh1_K4_s92608`
- `CA_A4_ETTh2_safe17400_K4_s92608`
- `CA_A4_appliances_K10_s92608`
- `CA_A4_jena_K6_s92609`
- `CA_A4_ETTh1_K4_s92609`
- `CA_A4_ETTh2_safe17400_K4_s92609`
- `CA_A4_appliances_K10_s92609`
- `CA_A4_jena_K6_s92610`
- `CA_A4_ETTh1_K4_s92610`
- `CA_A4_ETTh2_safe17400_K4_s92610`
- `CA_A4_appliances_K10_s92610`
- `CA_A1_jena_K6_s92601_ext10`
- `CA_A1_jena_K6_s92602_ext10`
- `CA_A1_jena_K6_s92603_ext10`
- `CA_A1_jena_K6_s92604_ext10`
- `CA_A1_jena_K6_s92605_ext10`
- `CA_A1_jena_K6_s92606_ext10`
- `CA_A1_jena_K6_s92607_ext10`
- `CA_A1_jena_K6_s92608_ext10`
- `CA_A1_jena_K6_s92609_ext10`
- `CA_A1_jena_K6_s92610_ext10`

## 지시문 원문: 0·3·5·6절

## 0. 목적

첫 캠페인 결과(`06f142db`)에 남은 공백 중 발표에 가장 중요한 두 개를 메운다.
- D의 순수 효과가 없다(A4 미실행). "ETT에서 AdaPTS식 어댑터가 무너지는 이유는 마지막 관측값을 압축 없이 더하는 경로가 없어서"라는 설명이 [추정]으로 남아 있다.
- Jena에서 A1(No P)의 CONTRIBUTES가 잠정이다. A1은 10 seed 모두 상한 3 epoch를 골랐다(아직 개선 중).

이번 실험이 답할 질문:
- **Q4 D**: D 구조 전체(마지막 관측값 + 변화량 + lag)를 단순 선형 디코더로 바꾸면 데이터별 V MSE가 얼마나 바뀌나?
- **Q5 기제**: AdaPTS식 어댑터와 A4의 오차가 "출력이 K차원 공간에 갇혀 V 구간의 수준 이동을 따라가지 못한다"로 설명되나?
- **Q2 갱신**: A1을 10 epoch까지 학습해도 Jena에서 P 효과가 남나?

이번에도 남는 공백(보고서에 명시): Jena 비교의 TIE_PROVISIONAL, lag만의 효과(A5), E의 효과(A3).

결과를 보는 사람: INFORMS 2026 Annual Meeting(11월 1–4일) 발표 청중과 교수님 사전 검토.
결과로 16·18장의 절제 문장과, 발표에서 P와 D 중 무엇을 중심에 둘지를 정한다.

용어: V(개발용 검증 구간), T(보호 테스트 구간), A_fit(학습 구간), A_stop(선택 구간), A60(재학습 구간), IQR(사분위 범위, Q3 − Q1).


## 3. 승인된 고정값 (바꾸지 말 것)

### 실행 대상과 순서
모두 원 `RUN_PLAN.json`에 있는 job이고, 필드를 그대로 쓴다.
1. A4 기본 40개.
   - seed를 데이터 사이에서 돌아가며 실행한다(92601을 4개 데이터에, 그다음 92602 …).
2. A4가 다 끝나면 중간 분석을 보고한다(4절 Step 2.5). 학습은 멈추지 않고 바로 3으로 넘어간다.
3. A1/jena 연장 10개(`CA_A1_jena_K6_s{seed}_ext10`).

### 실행하지 않는 것
- A5, A3, C2·C3의 모든 연장, A1/ETTh2 연장, A4 연장(규칙이 걸려도), RUN_PLAN 밖의 모든 실험.
- 사유는 NOT_RUN_SCOPE로 기록한다(예산 부족과 구분).
- A4에서 연장 규칙이 걸리면, 연장 없이 해당 판정에 PROVISIONAL을 붙인다.

### 그대로 유지하는 것
- 레시피·선택 규칙·지표 정의: 첫 캠페인 등록 코드 그대로.
- 한 번에 한 실행. 첫 캠페인의 동시 실행 시험이 직렬을 택했다(`CONCURRENCY_PROBE.json`).
- 실패 처리: 첫 캠페인과 같다. 한 번 재시도하고, 같은 family에서 FAILED가 3개면 그 family를 멈춘다.
- 진행 파일 교체는 WinError 5/32/33 재시도 수정이 들어간 방식으로 한다(첫 캠페인 `RECOVERY_001.json`).


## 5. 정의와 결정 기준

### 기존 기준 (첫 사전 등록 그대로)
- 절제 판정, spread(비교하는 두 대상의 IQR 중 큰 값), PROVISIONAL, 연장 규칙은 `reports/comparative_ablation_20261025_v1/PREREGISTRATION.md` 그대로 쓴다. 바꾸지 않는다.
- 해석 고정: A4 대 OURS는 **D 구조 전체**(마지막 관측값 + 변화량 + lag)의 효과다. 마지막 관측값과 lag의 효과를 나눠 말하지 않는다(A5를 돌리지 않으므로).
- 4개 데이터를 모두 보고하고, 요약은 판정 대상 3개(jena, ETTh1, appliances)로 한다.

### D0 — 고정 PCA 부분공간 하한 (학습 없음)
데이터마다 등록 로더, A60 scaler, `initialize_pca`와 같은 PCA(K = 등록 K)로 계산한다.
- **A60 기준값**: A60 누적 설명분산, A60 행의 공간 밖 MSE.
- **하한**: V 평가 대상점(origin × horizon, 모든 채널이 관측된 점)에서 공간 밖 성분 제곱합 ÷ C의 평균.
  - 이것이 C1 V MSE의 하한이다.
  - 일부 채널만 관측된 점은 빼고 그 비율을 보고한다.
- **추가 값**:
  - V 에너지(A60 center 기준) 중 공간 밖 비율
  - V 자체 최적 K차원 공간의 하한
  - V 평균 이동(채널별, A60 σ 단위)과 그중 공간 밖 성분
- **점검**: 같은 대상점 집합에서 저장된 C1 예측으로 MSE를 다시 계산하고, 그 값이 하한 이상인지 본다. 어기면 멈추고 보고한다(계산이나 해석 오류).
- **참고 열**: Raw, C1, C2, C3, OURS, A1의 중앙값.

### D1 — 학습된 디코더 출력 공간 하한 (학습 없음)
- C2·C3·A4의 출력은 b + (W의 K차원 행공간)에 갇힌다.
- 각 job의 선택 모델 디코더(W, b)로, V 평가 대상점에서 공간 밖 성분 제곱합 ÷ C의 평균을 계산한다. 대상점 규칙은 D0와 같다.
- job별로 실제 V MSE(같은 대상점), 하한, 비율(하한 ÷ 실제)을 보고하고, 데이터별 중앙값을 낸다.
- C3는 디코더 가중치로 계산한다. 출력 공간은 dropout 표본 평균과 무관하다.

### 계획자 예측 (판정이 아니다. 맞았는지 보고만 한다)
- A4: ETTh1·ETTh2에서 OURS 대비 중앙값이 +50% 이상 높다. Jena·Appliances에서는 |effect| < OURS 중앙값의 10%다.
- D1: ETTh1·ETTh2에서 A4·C2·C3의 하한이 실제 V MSE의 절반 이상이다.
- A1/jena 연장: 예측 없음.

이 숫자들은 합의용 임시 기준이다. 결과를 본 뒤 바꾸지 않는다.


## 6. 중단 조건

- **게이트 실패**: 멈추고 보고한다.
- **예산·마감**: 학습 경과 13시간 또는 2026-10-28 00:00 KST 중 먼저 오는 시점이 지나면
  - 진행 중인 실행만 마치고 새 실행은 시작하지 않는다.
  - 끝난 것까지로 분석·보고하고, 미완료 family는 n을 표시한다.
- **실행 실패**: 첫 캠페인 규칙과 같다.
- **기존 파일 변경**: 원 이름공간이나 이전 캠페인 파일의 hash가 바뀌면 즉시 멈추고 보고한다.
- **계획 밖 실험 금지**: 시간이 남아도 3절 밖(A5, A3, C2·C3 연장, 다른 K, VAE 계열, T 평가 등)은 시작하지 않는다. 남은 시간만 보고한다.

