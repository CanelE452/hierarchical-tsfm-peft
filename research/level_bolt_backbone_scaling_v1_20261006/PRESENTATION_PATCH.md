# 발표 수정 제안: 동일 Chronos-Bolt 계열 allocation

이 문서는 원본 PPT/PDF를 편집한 결과가 아닌 역할별 추가 제안입니다. 기존 Chronos-2 native multivariate system comparison과 별도 표·캡션으로 유지합니다.

- **질문/동기:** ‘큰 Small을 압축하는 대신 더 작은 Bolt에 원채널을 넣으면?’을 직접 질문으로 추가합니다.

- **자료·정보 계약:** Robin C17/K5, Peacock C13/K4, Jena C21/K6, L512/H48와 기존 TRAIN 통계·결측대체·origin·target mask를 명시합니다. F0들은 원채널 독립 단변량이라는 공통점을 표시합니다.

- **공정성/파라미터:** `FAIRNESS_TABLE.md`를 정확도 결론 전에 둡니다. G 이전 학습 변수, inference gradient0, backbone params, 전체 배포 params/bytes와 K/C task rows를 구분합니다.

- **MSE/MAE 주표:** Small+LEVEL, Small F0 anchor, Mini F0, Tiny F0의4행을 추가하되 세 자료를 평균하지 않습니다. combined 본문과 TEST-A/B 부록을 연결합니다. LEVEL의 두 선택 seed는 손실 평균이며 zero-shot은1개입니다.

- **대응 구간:** LEVEL/Mini와 LEVEL/Tiny의 상대 MSE·MAE와 조건부95% 구간을 표시합니다. 0 포함은 ‘차이 불확실’로 쓰고 ‘동등’이라고 바꾸지 않습니다.

- **비용:** 이번 같은 RTX4070 세션 B1 latency, B4 throughput, peak allocated/reserved를 사용합니다. 작은 Mini/Tiny의 전체 가중치 bytes도 명시합니다. raw block 범위·resident·sentinel drift는 부록에 두며 범위를 CI라고 부르지 않습니다.

- **내부 구조 대조:** compression-only b, LEVEL/RAW, 내부 ablation 결과는 현재 same-family allocation과 별도 근거로 유지합니다. deployment 선택에서 지배된 조건이 구조 효과의 부정이라는 문장을 추가하지 않습니다.

## 추가할 combined accuracy table

| Arm | Robin MSE | Robin MAE | Peacock MSE | Peacock MAE | Jena MSE | Jena MAE |
| --- | --- | --- | --- | --- | --- | --- |
| Small + LEVEL | 0.275196 | 0.324593 | 0.203266 | 0.315004 | 0.324359 | 0.320450 |
| Small F0 | 0.229095 | 0.262074 | 0.201521 | 0.300358 | 0.292300 | 0.264758 |
| Mini F0 | 0.227636 | 0.264707 | 0.202057 | 0.301707 | 0.302064 | 0.271424 |
| Tiny F0 | 0.243654 | 0.269697 | 0.202745 | 0.304780 | 0.321409 | 0.281006 |

**English caption:** Same-family practical allocation comparison on previously exposed datasets. All F0 models process each original channel as an independent univariate task. LEVEL uses a fixed TRAIN PCA, the pinned Small backbone, and a previously selected frozen G. Errors pool period sums/counts before channel averaging; LEVEL averages selected-seed losses. This is neither a method ablation nor independent confirmation.

## 자료별 본문 교체 문장

- **Robin / Mini F0:** LEVEL 상대 MSE 20.89%, 조건부95% 구간 [10.38, 36.79]%. `CHANNEL_COMPRESSION_DEPLOYMENT_ADVANTAGE_REJECTED_FOR_TESTED_SETTING`. Mini F0가 MSE·B4 peak allocated·B4 처리량의 세 주축에서 유리합니다. 이 조건의 압축 deployment 선택 주장을 축소합니다.

- **Robin / Tiny F0:** LEVEL 상대 MSE 12.95%, 조건부95% 구간 [0.75, 33.00]%. `CHANNEL_COMPRESSION_DEPLOYMENT_ADVANTAGE_REJECTED_FOR_TESTED_SETTING`. Tiny F0가 MSE·B4 peak allocated·B4 처리량의 세 주축에서 유리합니다. 이 조건의 압축 deployment 선택 주장을 축소합니다.

- **Peacock Education / Mini F0:** LEVEL 상대 MSE 0.60%, 조건부95% 구간 [-2.14, 5.82]%. `NO_SINGLE_DOMINANT_ALLOCATION`. 대응 MSE 구간이 0을 포함하므로 정확도 차이는 불확실합니다. 동등성의 증거가 아닙니다.

- **Peacock Education / Tiny F0:** LEVEL 상대 MSE 0.26%, 조건부95% 구간 [-2.50, 5.57]%. `NO_SINGLE_DOMINANT_ALLOCATION`. 대응 MSE 구간이 0을 포함하므로 정확도 차이는 불확실합니다. 동등성의 증거가 아닙니다.

- **Jena / Mini F0:** LEVEL 상대 MSE 7.38%, 조건부95% 구간 [4.71, 10.88]%. `CHANNEL_COMPRESSION_DEPLOYMENT_ADVANTAGE_REJECTED_FOR_TESTED_SETTING`. Mini F0가 MSE·B4 peak allocated·B4 처리량의 세 주축에서 유리합니다. 이 조건의 압축 deployment 선택 주장을 축소합니다.

- **Jena / Tiny F0:** LEVEL 상대 MSE 0.92%, 조건부95% 구간 [-1.88, 4.35]%. `NO_SINGLE_DOMINANT_ALLOCATION`. 대응 MSE 구간이 0을 포함하므로 정확도 차이는 불확실합니다. 동등성의 증거가 아닙니다.

## 주장 조정

**유지:** 같은 Bolt-small 내 compression-only b 보완과 LEVEL/RAW 등의 기존 구조적 결과는 해당 내부 대조의 범위에 유지합니다. 이번 allocation에서 남는 근거: 내부 구조 대조의 기존 근거와 이번 자료별 조건부 결과.

**축소:** 큰 backbone을 유지하며 채널을 줄이면 작은 원채널 backbone보다 deployment가 유리하다는 주장은 Robin vs Mini F0; Robin vs Tiny F0; Jena vs Mini F0에 맞춰 축소합니다. 한 자료의 결과를 세 자료의 평균 총점이나 범용 우위로 확장하지 않습니다.

**보류:** 3개 주 비교는 차이 불확실·비용 축 교차·측정 범위 중첩 등의 이유로 단일 우위를 보류합니다. 선택 불확실성, 사전학습 중복 여부, 독립 재현과 보호된 외부 확인은 이 회차에서 해결하지 않았습니다.

그림은 `figures/allocation_<dataset>.png`/`.svg`입니다. 각 자료의 두 영어 패널은 combined MSE와 B4 throughput, combined MSE와 B4 allocated를 보여 줍니다. 축은0을 포함하며 whisker는 observed block range입니다. 원본 PPT/PDF 편집, Mini/Tiny+LEVEL 또는 새 실험 실행은 수행하지 않습니다.
