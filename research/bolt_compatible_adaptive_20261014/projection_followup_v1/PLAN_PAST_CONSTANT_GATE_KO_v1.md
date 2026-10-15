# 과거가 정확히 상수인 원 C 채널의 조건부 출력 진단

상태: SOURCE·계획·작은 합성 검산 준비. 실제 자료를 읽어 초기 숫자를 확인하거나 기존 학습·모델을 다시 실행하는 단계가 아니다. ROOT의 accepted FIT-target COMPLETE와 독립 v2 PASS exact-SHA gate 이후 fresh CPU 진단을 한 번 실행할 예정이다.

## 질문과 결정적 규칙

현재 C anchor와 고정 D를 사용해도 mixed latent의 예측 변화량이 과거창에서 상수였던 C 채널에 오차를 전파할 수 있는가를 확인한다. 모든 자료·K·L/H·E/D·기존 FIT-target selected native head·FIT 통계·origins·checkpoint 선택을 유지한다. 미래 정답, threshold 탐색, 학습, 새 encoder/head/F forward는 사용하지 않는다.

과거 입력은 기존 causal forward-filled 정규화 FP32 `X[n,512,c]`와 true history observed mask다. `G[n,c]=0`은 **512개 값이 모두 마지막 값과 FP32 값 비교에서 정확히 같고, 해당 원 채널의 true observed count가 2 이상일 때만** 허용한다. 다른 경우는 G=1이다. 임계 오차·분산 추정·데이터별 threshold 선택을 넣지 않는다. +0와 −0는 수치상 같은 값으로 판정하며 실제 교체 anchor의 FP32 bits는 보존한다.

출력은 `where(G==0,currentC,originalPrediction)`으로 만든다. G=1 좌표는 원 예측과 byte-exact, G=0 좌표는 원 current-C anchor와 byte-exact여야 한다. 기존 E의 window 안 표현·F·native head·latent trajectory를 바꾸지 않고 **forecast origin에서 이미 알려진 전체 과거창을 읽는 출력 단계**에만 규칙을 넣는다. 창 안의 뒤쪽 알려진 과거를 바꾸면 G가 달라질 수 있으며 이를 시점별 표현 prefix 불변성과 혼동하지 않는다.

## 정보 조건의 변화와 대조

원 K-stream/current-C 디코더보다 **과거 C조건 metadata가 추가**된다. 새 인터페이스는 `K streams + current C + past-constant flags (C bits)`다. E 역시 원 C 과거를 읽더라도 디코더 측 정보 조건이 동일해진 것은 아니다. strict K-only 압축이라고 주장하지 않는다. 성능이 바뀌면 과거 C의 조건부 출력을 추가한 효과이지 표현학습 자체의 개선 증거가 아니다. 비용에서는 G를 계산하는 C별512과거·관측 mask 읽기와 flags 생성/출력 교체까지 포함해야 한다.

같은 규칙을 RAW_BOLT에도 적용한다. 자료2개×FIT/TUNE/기노출 DEV3split에서 RAW와 PCA_PINV/FIT_TARGET_PINV 각각의 Bolt·native두 seed, 총7원 예측의 ungated/gated 쌍을 저장하고 채점한다. 42쌍의 실제 캐시 CPU 출력 연산이며 F·E·native 전체 추론을 다시 수행했다고 세지 않는다. native head는 게이트 없는 decoder 조건으로 학습된 원 selected head를 고정하므로 **고정 head 출력 개입**이다. 새 gated-native를 최적화한 공정 학습 대조라고 하지 않는다. E(Y)는 이번 배포 gate 비교·순위에 포함하지 않는다.

## 안전성이라고 확대할 수 없는 이유

과거가 상수여도 미래가 바뀔 수 있다. true observed count≥2는 최소 관측 증거일 뿐 미래 상수성·최적 예측·잡음 제거를 보장하지 않는다. 정규화 FP32의 반올림이나 큰 mean 차감으로 서로 다른 원 물리 값이 같은 정규화 값이 될 수도 있다. 원 physical history의 정확한 상수성과 같다고 쓰지 않는다. 합성 검산에는 과거 상수 뒤 미래 변화로 gate가 악화되는 경우와 FP32 정규화가 작은 원 값 차이를 숨기는 반례를 포함한다.

미래 정답은 gate 생성에 들어가지 않고 평가·진단에만 쓴다. eligible 채널 중 관측 미래가 current-C와 달라진 위치/창을 별도로 센다. 이것도 **저장된 normalized FP32 비교**여서 물리 값의 작은 변화가 반올림으로 숨을 수 있다. 미래 관측이 전혀 없는 창은 미래가 그대로였다고 해석하지 않는다. 전체 observed 채널 macro MSE/MAE·원단위 차이와 exact complete-C mask 지표를 함께 보존한다. 겹친 window의 counts는 독립 사건 수가 아니다.

row-vector latent와 K×C D 계약에서 gate는 origin별 `D·Diag(G)` 형태의 조건부 출력 공간을 만들고 기존 고정 D와 달라질 수 있다. 즉 `currentC + Δz[D·Diag(G)]`다. 이전 static fixed-D 최소 오차를 이번 gated 출력의 lower bound로 재사용하지 않는다. synthetic 반례에서는 gate가 이전 fixed-D span 밖의 출력에 도달할 수 있음을 보인다. 이를 데이터의 미래 정보가 복구됐다는 증거로 확대하지 않는다.

## 실행·검산과 남은 비용

[모듈](past_constant_gate_v1.py)과 [합성 검산](test_past_constant_gate_v1.py)은 actual data/model/fit IO 없이 순수 NumPy 연산을 정의한다. [CPU controller](run_past_constant_gate_v1.py)는 COMPLETE FIT-target producer와 정확한 v2 PASS·audit-source SHA·전체 counts를 먼저 확인한다. pending/불일치에서는 과학 라이브러리·Meter·NPZ 접근 전에 멈춘다. ROOT가 실행할 fresh 결과 디렉터리에 SCOPE.json, 원 source/mask/origin/예측 결속, G·counts·ungated/gated predictions·지표와 failure-path를 보존한다. 원 checkpoint/선정/캐시/모델 파일은 전후 SHA가 같아야 한다.

이 단계의 실제 scope는 **캐시 CPU gate 연산·채점**이다. 전체 배포 비용은 미측정으로 남기고 별도 ROOT 작업에서 ungated/gated 조건을 같은 origin/batch/모델로 실제 측정한다. E/F 비용이나 GPU 절약·개선은 이 단계에서 주장하지 않는다. TEST, 기존 job 종료·재개, 설치, 새 학습, 새 dataset, 자동 commit/push는 없다.

`[판단]` 사용자의 지속 원인 진단 목표 아래 결정적 최소 후속으로 해석한다. 개별 gate 방법의 새 exact-human 승인이나 독립 확증이라고 쓰지 않고 원 승인 기록을 보존한다. 이미 노출 DEV와 선행 진단을 본 후 정한 탐색적·적응적 후속이다. 실제 캐시·지표를 독립 검산하고 전체 비용을 확인하기 전에는 해결이나 전체 연구 완료를 선언하지 않는다.
