# Subset 선정의 수치 오류와 직접 잔차 계산 복구

상태: 원 v1 부분 실행 후, 새 v2 실제 실행 전의 복구 기록이다. v1 source·결과·영수증·성공한8조건을 덮어쓰거나 성능 실패로 바꾸지 않는다. 원 subset 실제 선정과 K-channel F 추론은0이었다.

## 실제 실패와 근거

`runs/raw_subset_comparison_v1/RESULT.json` SHA `4240cedea826e052b227ef2fba4e16ff52cdcd6db69281844485eca6c0b0bd6b`는 PARTIAL_COMPLETED다. 두 자료의 subset 선정이 `NEGATIVE_RECONSTRUCTION_MSE_BEYOND_FP64_ROUNDOFF`에서 중단됐다. 독립적인 PCA/FIT-target의 global D8개·native8fit/4800update/40checkpoint·24native/12Bolt split 예측은 완료했고 F hash는 변하지 않았다. Subset 실패는 해당 경로에만 적용했다.

ROOT가 원 FIT NPZ의 `history`, `history_observed` 두 member만 checksum/shape/dtype까지 검증해 열고 첫 실패를 독립 재계산했다. 미래 target·TUNE·DEV·모델은 읽지 않았다. [실제 진단 JSON](audits/RAW_SUBSET_GRAM_FAILURE_DIAGNOSTIC_ROOT_v1.json)의 원수치는 다음과 같다.

```text
자료       원 후보/응답 index  관측행수   Gram 식 MSE            원 음수 허용값          직접 centered LS 잔차 MSE
Robin       1 / 1             131072    -1.09023901018e-12     1.31320626148e-13       2.15554472008e-25
Beijing     4 / 4             121356    -7.54951656745e-13     1.14842822114e-13       2.06545362237e-26
```

둘 다 자기 채널을 자기 값으로 회귀하는 rank1 대조다. 원 후보 함수는 응답 분산·cross product·Gram quadratic을 서로 다른 합산으로 계산한 뒤 빼서 MSE를 구한다. 약1 크기의 수를 상쇄하면서 각각 약1e-12의 음수가 생겼다. 실제 squared residual은 음수가 아니며 약1e-25다. 음수 검사의 허용값만 키우거나 무조건0으로 잘라 통과시키는 복구를 하지 않는다. 이 사례는 subset 예측의 유용성이나 학습 실패의 결과가 아니다.

## 새 source의 변경 범위

원 실행 source `raw_subset_projection_v1.py` SHA2583b221.../`raw_subset_native_head_v1.py` SHA40811301.../`run_raw_subset_comparison_v1.py` SHA33b21b10...를 보존한다. 새 `raw_subset_projection_v2.py`는 같은 FIT 관측 과거·response별 중심화·singular rcond1e-4·동일 greedy 순서/원 index exact tie 규칙을 유지하면서, 직접 centered `lstsq`와 실제 잔차 제곱 평균으로 후보 점수를 계산한다. Gram eigen cutoff 계산과 subtraction MSE를 선정에서 사용하지 않는다. 수학적 목적은 같지만 두 finite 계산의 scalar나 exact tie가 bit-identical하다고 주장하지 않는다. 원 v1에서는 subset 승자가 선정되지 않았다.

global FIT decoder fitting·E/F/native 구조·실제 손실·600update·checkpoint0/100/200/400/600·TUNE 선정·기노출 개발 자료·주지표·K·정규화·관측 mask는 변경하지 않는다. 새로운 실제 source는 첫 실행 전에 ROOT 검토/합성검산/SHA 고정을 거친다. 새 controller는 fresh `raw_subset_comparison_v2` 디렉터리에 저장하며 v1 partial 및 본 addendum을 결속한다.

12조건의 새 선정 잠금/registry를 모두 저장한 다음 새 DEV 입력을 여는 원 실행 계약을 단순하게 유지하기 위해 v2에서12조건을 zero-init부터 새로 실행한다. 이 중8 mixed 조건은 v1 성공조건과 같은 반복이다. 따라서 두 시행의 실제학습20회를 독립조건20개로 세지 않으며 distinct 조건은12개다. 원8조건의 E/D·checkpoint states·학습곡선·TUNE/DEV 예측·선정 update와 새 실행의 동일성을 확인한다. 차이가 있으면 숨기거나 허용오차를 늘리지 않고 원수치로 보존한다. v1의 이미 노출된 DEV를 독립 holdout으로 바꾸지 않는다.

원 수치 실패·setup·후속 직접 잔차 계산·중복 학습의 사용량을 모두 계측/기록한다. 이전 자원 상한은 최신 직접 지시로 해제됐지만 기록을 초기화하지 않는다. 이 수치 복구는 새 연구 주제나 실제 성공이 아니며 TEST·설치·큰 백본 학습·기존 실행 종료/재개·자동 commit/push를 포함하지 않는다.
