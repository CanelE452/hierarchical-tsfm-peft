# RF3의 기준 경로가 과거의 지연 전파를 상쇄하는 반례

상태: **SOURCE_ONLY_NOT_EXECUTED**. 이 문서는 고정된 소스의 대수적 분석과 CPU 합성 검산 소스 준비이다. 실제 데이터·체크포인트·잠재 배열·모델을 읽지 않았으며 합성 검산도 아직 실행하지 않았다. 진행 중인 승인된 RF1/RF3 8개 학습과 frozen decoder/runner를 수정하거나 중단하지 않는다. 실제 CPU 실행 결과는 별도 로그·proof에 기록해야 한다.

## 직접 답변

[확인: 대수와 소스] 현재 `TemporalResidualHead.forward`는 `R(실제 과거 last2 + 예측 미래)`에서 `R(동일 과거 last2 + 마지막 잠재값의 반복)`을 뺀다. 따라서 미래 잠재 예측이 마지막 과거 잠재값과 모두 같으면, 어떤 hidden/output weight와 GELU를 사용해도 두 입력 시퀀스가 같아 residual이 정확히 0이다. 이는 구현이 약속한 동작이다.

[판단] 그러나 **미래 잠재값의 혁신이 없으면 raw C채널 전체도 그대로여야 한다**는 가정은 일반적인 지연 관계에서 성립하지 않는다. 현재 head가 검증하는 범위는 마지막 잠재값의 반복으로 설명되지 않는 추가 예측 혁신의 복원이다. 과거에 이미 관측한 정보가 시간 진행만으로 다른 채널에 전파되는 모든 형태를 포함하는 디코더라고 해석하면 범위를 넘는다. 이 구조적 반례가 Robin/Beijing 실패의 실증 원인이라는 증거는 없다.

## 최소 반례: 새로운 미래 정보 없이도 lag 채널은 변한다

스칼라 잠재값을 `z_s = u_s`, 원 채널을 `X_s = (u_s, u_(s-1))`로 둔다. 과거 끝 세 점은 `[-1,2,5]`이다. 현재 원 채널은 `X_t=(5,2)`이고 frozen forecaster가 다음 세 잠재값을 `[5,5,5]`로 예측한다고 가정한다. 이 합성 예에서는 실제 미래 u도 5로 유지된다고 정의한다. 실제 미래값은 검산용 정답에만 쓰며 head의 입력에는 없다.

첫 미래 시점의 lag 채널 값은 이미 관측한 `u_t=5`이다. 따라서 원 채널의 미래 정답은 `(5,5)`이며, 현재 raw anchor `(5,2)`에 필요한 변화는 `(0,3)`이다. 과거의 변화가 끝나고 미래 u가 평탄해도 두 번째 채널은 한 번 더 이동해야 한다.

현재 paired-reference RF3에서는 예측 경로와 기준 경로가 모두 `[2,5,5,5,5]`라서 residual은 `(0,0)`이다. 기존 pointwise D의 미래/현재 차도 flat 잠재값에서 0이므로 anchor 기반 전체 예측은 `(5,2)`를 유지한다. 단순 미래 잠재 추론 오류가 전혀 없어도 lag 복원이 실패할 수 있다.

원점 기준 대안은 미래 경로의 R에서 **실제 origin 시점의 R**을 뺀다. `R`이 윈도우의 `(현재 u,직전 u)`를 출력하면 origin 응답은 `R([-1,2,5])=(5,2)`, 첫 미래 응답은 `R([2,5,5])=(5,5)`다. 차 `(0,3)`을 anchor에 더하면 미래 lag 값을 복원한다. origin 기준 입력은 실제 과거 세 점뿐이다.

```text
입력/출력                  값
실제 과거 잠재             -1, 2, 5
예측 미래 잠재              5, 5, 5
현재 raw anchor             (5, 2)
현재 RF3 residual           (0, 0)
원점 기준 residual          (0, 3)
현재 RF3 전체 예측          (5, 2)
원점 기준 전체 예측         (5, 5)
합성 미래 정답              (5, 5)
```

이 예에서 3시점×2채널의 균등 MSE는 현재 방식 4.5, 원점 기준 0이다. 이 수치는 합성 값의 대수이며 실제 연구 자료의 성능 수치가 아니다. 검산 소스는 같은 계산을 CPU로 확인하도록 준비했다.

## 선형 RF3에서 무엇이 상쇄되는가

RF3의 선형 응답을 `R(v_(s-2),v_(s-1),v_s)=A0 v_(s-2)+A1 v_(s-1)+A2 v_s+b`라 하자. `e_h = predicted_z_(t+h)-z_t`로 놓는다. 현재 두 경로의 prefix가 같으므로 residual은 다음과 같다.

```text
h=1:  A2 e_1
h=2:  A1 e_1 + A2 e_2
h>=3: A0 e_(h-2) + A1 e_(h-1) + A2 e_h
```

실제 과거 `z_(t-1)` 등에서 출발하는 항은 차분에서 상쇄된다. 마지막 `z_t`는 혁신의 공통 기준으로 남는다. 즉 선형 RF3에서는 예측 잠재 혁신들 사이의 시간 필터링은 가능하지만, 혁신이 0이어도 필요한 알려진 과거의 lag 이동은 만들 수 없다.

GELU가 있는 실제 head는 선형식보다 넓다. 두 preactivation이 서로 다르면 같은 과거 항이 activation의 작동 영역을 바꾸므로 **혁신이 0이 아닐 때** 과거가 출력 변화의 크기·방향에 영향을 줄 수 있다. 따라서 nonlinear head가 모든 과거 정보를 항상 지운다는 주장은 틀리다. 다만 flat 잠재 예측에서는 두 전체 입력이 동일하므로 nonlinear head도 residual을 만들 수 없다.

## 준비한 최소 대안과 공정한 후속 비교의 범위

새 `temporal_origin_reference_v1.py`의 helper는 기존 동일 RF3 `head.response`를 두 번 호출한다. 미래 응답은 `last2 actual past + predicted future`이며, origin 응답은 `last3 actual past`의 마지막 시점이다. 반환값은 `future_response - origin_response`이다. 원 codec/D/current-C anchor는 유지하며 실제 Y·future mask·teacher·추가 raw-C 정보는 받지 않는다. 새 학습·실행 driver는 만들지 않았다.

등록 파라미터와 RF3 폭/kernel은 동일하다. 원점 경로는 출력 길이 1이므로 현재처럼 H개 기준 응답을 계산할 필요가 없지만, 실제 GPU latency/메모리 이득은 측정 전이다. `past_z`는 기존에 이미 계산한 E의 출력에서 세 점을 사용한다. 현재 saved trace의 last2만으로는 origin을 정확히 재구성할 수 없으므로 **추후 신규 fit은 last3를 별도 저장**해야 한다. old 배열의 없는 세 번째 값을 추정하거나 보완하지 않는다.

출력 final weight/bias가 0이면 새 origin residual도 정확히 0이어서 원 MAIN 초기화의 parity를 유지할 수 있다. 전체 과거와 미래가 동일한 상수이면 origin과 미래의 RF3 윈도우가 같아 C anchor를 보존한다. 반면 **변화한 과거 + 평탄한 미래**에는 정당한 lag 이동을 허용한다. 기존의 flat-future→C-persistence 보장을 의도적으로 완화하는 별도 구조 비교이다.

원점 기준이 무조건 우월하지는 않다. 과거 표현의 origin 응답이 잘 복원되지 않으면 reference 오류가 전체 horizon에 공통으로 들어갈 수 있다. 처음 두 시점은 실제 과거와 예측을 혼합하고 이후 시점은 예측만 쓰므로 경계의 분포 차이도 남는다. 오래된 E가 시점별 last latent에 이미 lag 상태를 충분히 담고 있다면 현재 pointwise 방식으로도 문제가 없을 수 있다. E/D를 같이 갱신하면 실험 차이를 순수 decoder 원인으로 귀속할 수 없으며, 기존 subset의 반복 노출과 seed2 제한도 그대로다.

## 합성 검산 소스와 이후 판단

`test_temporal_reference_counterexample_v1.py`는 미래 입력이 flat인 lag 반례, 선형 prefix 상쇄식, nonlinear modulation의 잔존, globally constant 과거/미래의 anchor 보존, causal future support, zero-final 초기화, past3 요구 조건을 확인하는 CPU 합성 테스트를 제공한다. 명시적인 GELU 쌍 `GELU(s)-GELU(-s)=s`와 고정 weight를 사용한다. 실제 모델 학습과 함수 표현성 증명을 혼동하지 않는다.

승인된 현재 8개 fit은 그대로 마무리해야 한다. 그 결과가 좋든 나쁘든 이번 반례를 근거로 현재 결과를 삭제하거나 정의를 바꾸어 승리로 만들지 않는다. 이후 비교한다면 기존 paired RF3와 신규 origin-reference RF3를 같은 original CP·seed order·600 update·normalized Y loss·TUNE MSE incl0·자료 subset·용량·측정 조건으로 맞추어야 한다. 비용·자원이 남고 계약을 확정한 범위에서만 실행한다. 없으면 미실행으로 남긴다.

[판단] 대수적 반례는 **현재 디코더가 모든 time-lag restoration을 검증했다는 결론을 막는 근거**다. 원점 기준 대안의 실제 유용성, 현재 자료의 실패 원인, 압축 없는 RAW를 넘는지 여부는 아직 별개 질문이다.
