# B2 후보와 핵심 직접 예측 대조

후보는 **B2 시간·채널 처리 → C21에서 K6로 압축 → 동결 Chronos-Bolt → 지연 디코더와 현재값 보존 → 원 C21 미래 예측**이다. P/E/D는 원채널 미래 예측 MSE로 학습하며 F 가중치는 고정한다. G/H 결합·가변 K·새 그래프를 주 방법으로 바꾸지 않는다. 구현 타당성, 성능, 신규성은 각각 판단한다. PPT/PDF는 만들지 않는다.

## 실제 계산

Jena primary, L512/H48/C21/K6, seed92601을 사용한다. 학습된 원채널 정규화와 초기 PCA 중심을 각각 `mu_A, sigma_A, c`라 두면 다음과 같다.

```text
x = (허용 과거의 causal fill(X) - mu_A) / sigma_A  [B,512,21]
z = (P_B2(x) - c) E                                [B,512,6]
u = frozen_Bolt(z)                                 [B,48,6]
joined = concat_time(z, u)
lags = {0,1,2,4,8,16,32}
features_l = joined[:,512-l:512-l+48] - z[:,-1-l,None]
y_hat_normalized = x[:,-1:] + concat_feature(features_l) D
y_hat_physical = y_hat_normalized * sigma_A + mu_A
```

이것은 `networks.ForecastAdapter`의 실제 decode다. 현재값 우회 경로가 있으므로 원 C의 정보가 모두 K개만을 통과한다는 통신 압축 주장은 하지 않는다. B0는 P만 항등이며 E/D가 학습된다. B0에도 F와 지연 D를 통한 시간 처리가 있다. B2의 과거 창 내부 혼합은 예측 원점 이후 실제 미래를 사용하는 것과 다르며, 매 시점 strictly causal 표현이라고 부르지 않는다.

## 고정된 데이터와 선택

```text
A_fit    [0,10482)
A_stop   [10482,12579)
M_cal    [12579,14675)
V_select [14675,16772)  origins43, H48, 관측정답43344
T_final  [16772,20965)  이번 비교의 수치 파싱·정답 계산·선택 없음
```

주지표는 사전에 고정한 정규화 observed channel-macro MSE이고, 같은 공간의 MAE를 함께 보고한다. 채널별 관측 SSE/count를 구한 뒤 채널을 균등 평균한다. 같은 origin·channel·mask·A60 scaler를 사용한다. 반복 개발 V이므로 독립 일반화 검증으로 부르지 않는다.

새 prefix 로더는 실제 Fortran-order 원 배열에서 각 채널의 `[0,16772)` 값만 수치 변환한다. 전체 원 파일의 바이트 SHA 읽기나 ZIP 내부의 opaque tail 처리를 물리 I/O 0으로 과장하지 않는다. 기존 전체20965행의 데이터 지문은 상속 metadata이며, 실제 materialized prefix 지문을 별도로 기록한다. 원 DATA_CONTRACT·분할 경계는 덮어쓰지 않는다.

## R5a: 동일한 최종 표현·복원 가중치

R4 `postT_cpu_jena_E3_B2_k6_s92601_v1`의 정확한 A60 artifact, P/E/D와 scaler를 고정한다. 전체 Z[512,6]에 채널 공유 `Linear(512,48,bias=True)` 24,624파라미터를 적용해 F만 바꾼다. 원 `adapter.decode(x,z,u,6)`와 현재값 보존을 그대로 쓴다. P/E 추출은 no_grad로 할 수 있으나 동결 D를 통한 head의 역전파는 유지한다.

Head는 latent persistence로 초기화하고 A60의 모든 합법 origin에서 **고정3epoch** 학습한다. AdamW lr0.001·weight_decay0·effective16/micro4·gradient_clip1·seed92601을 사전 고정한다. 학습 길이·LR를 결과로 선택하지 않고 M_cal 정답을 추가로 사용하지 않는다. A_stop은 기존 표현이 이미 본 구간이므로 미노출 head 선택 자료로 쓰지 않는다. R4의 선택 refit1epoch와 head3epoch의 추가 적합량도 구분해 보고한다.

이 비교는 Bolt용 표현과 동결 D 아래의 예측기 진단이다. R5a가 나빠도 모든 직접 예측기의 한계나 Bolt의 보편적 필요성을 뜻하지 않는다.

## R5b: 같은 구조의 별도 학습

같은 P/E·K6·D·현재값 규칙을 새로 초기화하고 같은 시간 선형 head와 함께 학습한다. A50에서 0/.25/1/3epoch 검색, A_stop MSE 선택, 선택 길이의 fresh A60 refit을 적용한다. 원 미래 MSE·원 학습 recipe·초기화 원리를 유지한다. 최종 P/E/D 가중치는 R4와 다르므로 같은 표현 비교라고 쓰지 않는다. 기존 DirectG나 DirectMixer를 matched R5b로 이름만 바꾸지 않는다.

1차 Jena 새로 적합/재개하는 조건은 최대4개이며 현재 사전 등록은 R5a/R5b 두 개다. 학습·비용은 동시에 실행하지 않는다. 기존 유효 비용은 재사용하고 새 경로는 B1/B8의 실제 physical 입력부터 전체 배포 경로를 측정한다. 캐시 Z·저장 forecast를 비용에서 사용하지 않는다. Raw의 기존 parity 실패는 유지하고 미측정을 0으로 쓰지 않는다.

우선 비교와 근거를 판단할 수 있게 정리한 다음, 필요한 PeMS08·추가 seed와 기존 후순위 작업을 남은 원 예산·잠금 안에서 순차 진행한다. 성능이 나쁘다는 이유로 자동 모듈 추가나 무한 연장 학습을 하지 않는다.
