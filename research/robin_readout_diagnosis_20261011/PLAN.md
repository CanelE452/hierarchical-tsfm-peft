# Robin 원채널 readout 진단과 비교 계획

사용자 요청: 압축 없이 디코더가 Robin 성능을 떨어뜨리는 원인을 분석하고 같은 조건의 개선 방법을 실제 비교한다. 기존 main 게시 실험은 그대로 보존하고 새 연구 폴더와 같은 이름의 cache만 사용한다. 이번 요청에는 새 commit/push가 포함되지 않는다.

## 출발점과 확인 범위

[확인] 기존 identity E/K=C + D8은 TRAIN MSE 0.169016→0.151228, VAL 0.408587→0.625913이다. 원래 VAL F 예측은 identity 경로와 bitwise 동일하다. 압축이 없어도 발생하는 readout 일반화 실패이며, 이것만으로 과적합·분포 변화·규제 prior 중 하나를 단독 원인으로 확정할 수 없다.

고정 입력: C17, context512/horizon48, 저장된 동일 TRAIN512/VAL30 시작점·F 예측·과거8시간·관측 마스크. 신규 F/TEST/G/encoder 학습·GPU benchmark는 하지 않는다. 예측 시작점마다 target window가 겹치므로 512/30을 독립 표본 수로 부르지 않는다.

## 사전 고정 비교

Raw F, zero/full D8(기존 grid 재사용), identity residual/full D8, zero/diagonal D8, identity residual/diagonal D8, identity residual/full D0, identity residual/lag-minus-current D8를 비교한다. 공통 10 λ는 이전 연구의 확장 grid 그대로다. 새 후보 결과를 보기 전에 이 목록과 선택 규칙을 고정한다. identity fallback은 선택 후보에 추가하지 않는다.

직접 D는 y를 예측하고 잔차 D는 y−F를 예측해 F에 더한다. 같은 full D8의 두 방식은 선형 함수 클래스가 같고 규제 중심만 다르다. RMS 좌표의 identity prior는 단순 I가 아니라 scale×identity selector다. Diagonal D8은 각 출력의 자기 채널 9개 입력만 사용한다. DC-difference는 lag1..8−현재 F의 full cross-channel 입력으로 잔차를 예측하여 history와 forecast에 같은 상수 벡터를 더할 때 그 상수만큼 출력을 이동시키는 구조다. D0/DC-difference 비교는 차원·규제 기하도 달라지는 구조 개입이며 순수 단일 요인 실험으로 과장하지 않는다.

기존 VAL 최소 MSE 선택 결과와 TRAIN 시간순 3-fold 선택 후 all-TRAIN refit 결과를 모두 제시한다. Fold timestamp cuts=[528,1026,1524,2022,2520], fit target 마지막 시간이 heldout 시작 이전이도록 48시간을 purge한다. Fit 114/243/380, heldout119/125/123 시작점, fold 내 RMS 재적합, 세 fold macro MSE 동등 가중. 부모 데이터/F trace는 full-TRAIN normalization을 이미 사용하므로 이 CV는 **고정 trace에 조건부인 readout CV**이며 완전한 end-to-end 미래 누수 없는 backtest가 아니다. 기존 VAL도 반복 개발에 사용되었다.

## 실행·진단·검증

- [x] 실행 전 protocol/source/input SHA 및 기존 연구/Git/history 보존 snapshot.
- [x] 잔차 identity, diagonal support, DC shift, causal lag, purge, prior equivalence에 대한 의미 있는 fixture tests.
- [x] 새 full-TRAIN 50 + CV180 =230 ridge 후보/3910채널 solve, 기존10 후보 재사용. CPU 고정 정밀도·단일 BLAS thread. 완료 result 덮어쓰기 금지.
- [x] 전체 grid의 TRAIN/VAL/horizon 결과·계수·예측 저장. 채널별/시작점별/오차 분해/평균/DC 왜곡 진단.
- [x] 독립 verifier가 원본 trace·target/mask·선택·계수 normal equation·예측·수치·counter·source SHA·보존 상태 재검증. 새 fit/F 호출 없이 검증한다.
- [x] 상세 한국어 보고서, PNG/PDF 그림, 상대 이미지/링크 경로, 적용 범위와 기여 후보 정리.
- [x] 기존 history prefix를 보존하여 결과 확인 후 변경 사항 append. 새 commit/push 없이 로컬 완료.

원인 설명은 관측 증상, 수학적 구조, 실제 개입 효과를 나누어 기술한다. 한 자료·한 고정 forecast·반복 VAL의 결과를 일반적 decoder 우월성이나 압축기 이익으로 확대하지 않는다. 비용은 학습/CPU readout 범위만 기록하며 전체 TSFM 추론 속도를 재측정한 것으로 표현하지 않는다.
