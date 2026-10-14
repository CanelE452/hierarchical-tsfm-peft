# 원 C·K 표현·시간 요약 비교의 실행 전 고정 조건

상태: SOURCE_ONLY / 새 범위 승인 대기. 이 문서는 [다음 계획](PLAN_MATCHED_READOUT_INFORMATION_KO_v1.md)의 모호한 구현 조건을 고정한다. 구현 준비는 실제 검증이나 실험 완료를 뜻하지 않는다. 원 stage source·모델·결과·실행 시계는 바꾸지 않는다.

## 자료와 재사용

원 Robin C17/K4 및 Beijing Aotizhongxin C11/K2, seeds92601/92602, L512/H48, FIT256/TUNE64, 반복 VAL30/노출 EVAL146, 원 FIT mean/std·origin·history observed·future Y observed를 유지한다. 원 E/D checkpoint는 hash만 확인하며 E/D/F를 새로 load·forward·update하지 않는다. 기존 cached past latent와 실제 학습한 N2 E33K 결과를 재사용한다. 미래 oracle latent/teacher 파일은 새 경로에 사용하지 않는다.

RAW33C의 normalized causal-filled history는 원 input bridge의 history다. 기존 native INPUT 파일에는 raw history가 없으므로 latent를 raw history로 간주하지 않는다. 두 입력의 shape/dtype·origin·target·observed·raw_last·history_observed·mean/std를 정확히 맞춘다. NaN은 미관측 target에서만 허용하며 exact 비교는 equal_nan=True다. raw_last는 normalized history의 마지막 C값과 정확히 같아야 한다. 관측 target과 모든 과거 입력·anchor·통계는 유한해야 한다.

기존 N2는 COMPLETE stage result, exact old MAIN checkpoint/dataset/seed/state metadata, 실제 coefficient·feature state·selected prediction의 SHA를 확인한다. seven alpha와 actual-Y TUNE 점수가 유한한지, lower-is-better earliest tie로 선택한 alpha/model/coeff가 맞는지 검산하고 저장 계수로 과거 예측을 복원한다. 원 N2의 학습·미래표현 생성 비용을 신규 사용량으로 중복 집계하거나 0으로 추정하지 않는다.

## 입력과 예측 연산

신규 RAW33C: FP32 normalized C 과거512점을 32개16-point mean과 마지막 C값으로 만든 33C 벡터. 신규 E512K: 원 FP32 past latent512K를 시간 순서 그대로 펼친 벡터. E33K는 기존 N2와 결과를 재사용한다. 모든 경로는 동일 actual-Y innovation, C 출력, current-C anchor 및 horizon/channel intercept를 갖는다. 새 예측 함수의 입력은 past와 current C이며 target·미래 mask·oracle가 없다.

각 경로 feature의 FIT-only 평균·표준편차는 FP64에서 추정한다. zero scale은1이다. 테스트/선택/평가에서 feature state를 재학습하지 않는다. block mean은 FP32에서 먼저 계산한다. target innovation은 기존 N2와 동일하게 FP32 Y−last를 먼저 계산한 뒤 FP64 solver로 넘긴다. 추론은 FP64 standardized feature @ coefficient + intercept → FP32 innovation → FP32 current-C anchor 더하기 순서다. FIT_INTERCEPT_ONLY는 과거 실험의 다른 FP64 residual recipe이며 정확한 zero-feature solver 등가 대조라고 쓰지 않는다.

## 목적·solver·선택

원 출력 채널 c의 FIT 관측 origin/horizon 총수를 Nc라 할 때 목적은 active 채널 평균의 sum_NH(mask*error²)/Nc + alpha*sum_h||W_scaled(h,c)||²다. 독립 (h,c) normal equation의 penalty는 alpha*Nc이며 horizon별 n(h,c)가 아니다. intercept를 penalize하지 않는다. 출력 관측0이면 coefficient/intercept0, innovation0으로 두고 미학습으로 표시한다.

알파는 0, 1e-4, 1e-3, 1e-2, .1, 1, 10으로 고정한다. 동일 mask의 centered thin SVD를 한 번만 계산해 일곱 해에 공유한다. P×P Gram/inverse를 만들지 않는다. alpha0은 centered minimum-norm, singular cutoff는 float64 eps*max(n_observed,p)*smax다. 양의 alpha는 원 stage와 같은 spectral ridge를 적용한다. NumPy 공식 [lstsq 정의](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html)와 [thin SVD 정의](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html)를 확인했다. 실제 설치 버전과 원 rcond=None 등가성은 새 승인 후 synthetic fixture로 검증하며 현재 검사 통과를 주장하지 않는다.

모든 후보의 FIT/TUNE actual-Y 정규화 observed-channel macro MSE/MAE와 원단위 채널 오차를 저장한다. 선택은 TUNE MSE 최솟값, exact tie는 목록에서 먼저 나온 alpha다. 선택한 모델만 원 VAL/EVAL에 추론하며 평가로 alpha·feature·mask·rank를 바꾸지 않는다. solver/계수/추론 실패는 해당 route를 보존하고 다른 독립 route/condition을 계속할 수 있으나, 전체 자원·자료 계약 실패는 관련 실행을 멈춘다.

## Capacity와 해석

```text
dataset       E33K(reuse)       RAW33C(new)       E512K(new)
Robin         p132             p561              p2048
Beijing       p66              p363              p1024
```

등록·관측 출력의 계수 수, intercept 수, effective design rank, mask 그룹 수, condition/count 배열, feature variance/scale를 기록한다. 입력 차원과 total feature energy가 달라 같은 alpha도 동일 capacity/regularization 효과를 보장하지 않는다. 이 비교는 실제 pipeline의 조건부 유용성을 확인하는 것이며 encoder 단독의 인과 효과, 압축의 필요성, 지연 정보의 인과적 유용성을 증명하지 않는다. 비용은 CPU 계수 추정·적용·검산 process wall이며 GPU0 단계에서 새 end-to-end deployment latency를 측정했다고 쓰지 않는다.

## 실행 gate와 산출물

최대 신규8 route fits ×7 alpha =56후보, E33K4조건은 재사용한다. 새 human authorization/고정 manifest 전에는 fixture·실제 asset read·solver·추론을 실행하지 않는다. 새 실행 상한은 제안 CPU process900초/GPU0/wall1800초/저장 증가1GiB/RSS8GiB다. 이전 stage ledger·receipt·시계는 보존하고 새 기록은 별도 경로에 둔다. 실제 solver 호출은 도중 cooperative interruption이 불가능하므로 guard를 factorization 전후에 적용하고 외부 프로세스 감독의 필요성을 밝힌다.

SOURCE snapshot과 SHA, 기존 N2 lineage, 같은 입력 계약, FIT feature state·all-alpha 계수·선택과 실제 예측·정규화/원단위 오차·훈련/추론 wall·검산·한국어 보고와 시각 자료를 새 경로에 남긴다. 실패/미실행/추가 자원 필요를 완료로 바꾸지 않는다. TEST·큰 모델 재학습·중단 실험 재개·자동 commit/push는 금지한다.
