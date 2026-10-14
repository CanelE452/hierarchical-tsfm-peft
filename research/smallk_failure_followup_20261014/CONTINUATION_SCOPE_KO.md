# 작은 K 미해결 문제의 후속 분석·개선 범위

상태: **PREPARATION_ONLY — 현재 Bi-Hash 연구를 먼저 완료한 뒤 실행한다.**

## 사용자 목적과 진행 순서

[확인] 사용자는 진행 중인 Bi-Hash의 표현 학습 가능성·예측 유용성 실험을 마친 다음, 이전 작은 K 연구의 성능·지표·속도 문제로 돌아가 분석하고 개선하도록 요청했다. 사용자가 “끝”이라고 하기 전까지 상위 목표를 유지한다. Bi-Hash 하위 연구의 완료를 전체 목표 완료로 처리하지 않는다.

소비처는 교수님과의 연구 논의다. 원하는 근거는 구현 가능성만이 아니라, 같은 조건의 기준군과 비교한 예측 성능·반례·비용과 설명 가능한 문제 정의다. 결과가 좋아 보이는 자료나 seed만 골라 방법론이 해결됐다고 쓰지 않는다.

[확인] 기존 실험·원 결과·보호 TEST·실행 중 작업을 보존하며, 이전 중단 캠페인을 자동 재개하지 않는다. 후속 실험은 새 버전으로 실행한다. 대형 백본 재학습, 자동 commit/push, 논문/PPT 작성은 기존 금지 범위를 유지한다. MD 결과 보고와 실제 시각자료는 연구 진행 기록으로 작성한다.

## 출발점의 실제 증거

[확인] 이전 보고서는 `../smallk_predictive_alignment_20261013/report_v2/REPORT_KO.md`이며 SHA256은 `0c7369a82304d35d5b21fdb7d8df1245702f9934999c807dc4d97d609d61eab7`이다. 그 보고서의 공동 새자료 EVAL 순서 위반은 보존하며 통과했다고 고쳐 쓰지 않는다.

```text
비교                  RAW MSE       작은K BASE MSE   상대 차이
Jena, 반복 VAL        .848631966    .848224944       -.05%
Robin, 반복 VAL       .340401887    .569546544       +67.32%
Beijing, 노출 EVAL    .589575414    .661957198       +12.28%
Appliances learned    .348920468    .342723827       -1.78%
```

[확인] Appliances learned의 정규화 MAE는 .367625077로 RAW .343643728보다6.98% 높고, MSE도 RAW+R .322202447보다 높다. 한 seed는 RAW보다 나쁘다. 전체 경로 batch1 지연은 Robin RAW11.8964ms/BASE12.4555ms, Beijing RAW11.7792ms/BASE12.59215ms였으며, 기존 자료에 E/F/D/전처리별 분해 계측은 없다.

[확인] 읽기 전용 탐색에서 기존 full prediction NPZ, 선택 CP, teacher FIT forecast/last/basis, 채널 오차 CSV, 전체 비용 기록의 존재와 SHA를 확인했다. 다음 진단은 있는 자산을 먼저 사용하고 필요한 값만 새 실제 추론으로 보충한다.

[확인] Robin의 저장 VAL 평균 gap에서 Zelma 채널의 MSE는 RAW .700793099→BASE3.554516109다. 해당 차이는 전체 채널합 gap의 약73.26%다. 이것은 오차가 집중된 위치이며 원인 지목이나 그 채널을 VAL 성능에 맞춰 선택하라는 지시가 아니다.

[확인] RAW FIT forecast−현재 상태의 centered rank-K 분산 설명률은 Robin K4에서96.010670%, Beijing K2에서66.925581%였다. Robin의 저장 teacher FIT forecast는 RAW와 maxdiff0이지만 Beijing teacher는 RAW와 다른 학습된 full-C 모델이다. 두 실패를 같은 기전으로 단정하지 않는다.

## 분석에서 고정할 경계

1. Jena/Robin의 기존 VAL과 Appliances/Beijing의 이미 확인한 EVAL은 이제 반복 사용한 개발 증거다. 같은 자료의 후속 개선을 새 독립 일반화라고 부르지 않는다.
2. 저장된 원 정규화 통계·mask·origin을 재사용하는 정확 재현과 새 FIT-only 통계로 다시 학습하는 새 조건을 분리한다. old FP32 F-layout 통계를 무심코 재계산해 기존 수치와 같다고 하지 않는다.
3. 현재 상태 C개를 우회로 보존하는 anchor는 TSFM 입력 K개라는 조건이지 strict K-only 압축이 아니다. 새 우회 정보·입력 길이·추가 예측기를 넣으면 그 비용과 정보량을 명시한다.
4. 실제 미래 정답 투영은 배포 방법이 아니다. RAW의 과거 기반 forecast를 저차원으로 근사한 진단도 RAW C채널 추론을 먼저 요구하므로 경량 배포 성능으로 세지 않는다.
5. centered covariance 설명률, through-origin projection 오차, nonlinear decoder 복원 가능성, 실제 target 예측 오차를 별도로 계산한다. 선형 rank-K 진단을 학습된 nonlinear D의 보편적 하한으로 부르지 않는다.
6. latent 좌표·원단위·정규화 MSE/MAE를 혼합하지 않는다. q 이후 정렬 개선을 q 이전 예측 개선으로 해석하지 않는다. MSE 감소와 MAE 악화가 공존하면 원수치와 tail contribution을 보고한다.
7. 모델 선택은 FIT/TUNE에서 하고, 같은 조건·seed·초기화·업데이트·주지표를 맞춘다. 보호 TEST는 그대로 닫는다. 새 독립 평가가 필요해지면 방법과 전체 자료의 선택을 먼저 봉인한다.

## 목적 트리와 먼저 필요한 진단

뿌리: 작은 K를 사용하는 TSFM adapter가 유용해지는 조건을 확인하고, 현재 실패를 개선할 수 있는 방법과 근거를 만든다.

- 오차 분해 → 입력/출력 표현 손실과 예측기의 변환 문제를 구분 → 처방 선택이 달라져야 한다.
- MAE·tail 분해 → 주 MSE 개선이 일부 큰 오차 감소인지 전반적 정확도인지 구분 → loss/readout 선택과 설명이 달라져야 한다.
- 실제 지연 분해 → 비용이 전처리·GPU launch·E·F·D 중 어디에 드는지 확인 → 같은 출력의 공통 최적화와 모델 단순화 선택이 달라져야 한다.
- 새 조건의 재현·검산 → 구현·정보 경계·선택 편향을 분리 → 개발 개선을 독립 효과로 과장하지 않아야 한다.

### D0: 원수치·시간/채널/꼬리 오차

기존 저장 arrays에서 FIT/TUNE/반복 평가의 채널·horizon·시간 구간·signed bias·absolute error 분위수·tail squared error 기여를 계산한다. Robin Zelma가 FIT/TUNE부터 나쁜지, 평가 구간의 level/variance shift에서 차이가 커졌는지 확인한다. 모든 채널을 함께 보고하고 관측된 VAL worst channel을 새 독립 조건 선택 기준으로 쓰지 않는다.

### D1: 예측 출력의 저차원 근사와 변환 순서

FIT RAW forecast 변화로 고정한 rank-K basis를 TUNE/기존 평가에 적용한다. through-origin과 mean+centered를 구분하고 실제 Y 오차도 별도 채점한다. 동일 projection에서 `RAW forecast를 투영`한 값과 `과거를 투영한 뒤 F로 예측`한 값을 비교한다. 전자는 잘 근사되는데 후자가 나쁘면 K 크기 하나의 설명보다 forecast 변환 순서·입력 분포/좌표 문제가 중요해진다. 둘 다 나쁘면 표현 예산·decoder 선택을 먼저 검토한다. 현재 E/D의 nonlinear 구조와 이 선형 진단은 같지 않다.

### D2: 실제 E/F/D 분리

이미 저장된 FIT latent 출력을 먼저 사용한다. TUNE/기존 평가 latent가 없으면 현재 Bi GPU 작업 종료 후 기존 CP의 새 실제 추론으로 저장한다. 같은 E에서 F와 단순 예측기의 readout을 비교하고, 실제 F latent를 고정한 FIT-only D 재적합을 TUNE 선택해 decoder 좌표/용량 문제를 구분한다. Bolt를 통해 학습된 E를 고정한 native readout을 native predictor의 최선 성능이라고 하지 않는다.

### D3: batch1/8 비용 분리

현재 동결 모델의 출력 parity를 먼저 저장하고 CPU fill/norm·H2D·E·F·D·anchor/inverse·D2H를 실제 계측한다. 분해 자체의 sync overhead와 원 전체 경로 시간을 구분한다. 공통 전처리 개선이나 compile/graph 최적화는 RAW와 후보 모두에 동일하게 적용하고 compile/load 비용도 남긴다.

## 다음 후보는 진단 결과로 좁힌다

[추정] 과거에 계산한 RAW forecast 행동을 작은 경로에 distill하는 목적은 실제 미래 정답의 어려운 잡음을 그대로 맞추는 보조 정렬과 다를 수 있다. 하지만 RAW가 최선 teacher라는 보장은 없고, teacher 오류를 복제하거나 기존 supervised loss와 충돌할 수 있다. D1/D2가 변환 순서 문제를 지지할 때 시험한다.

[추정] 적은 GPU kernel의 causal encoder와 출력 readout, 또는 FIT/TUNE에서 고정한 K 예산 변화는 latency/표현 병목을 다르게 개선할 수 있다. K=C로 바꾼 결과만으로 압축 문제 해결을 주장하지 않는다.

[추정] 물리적 원채널을 유지하는 representative/convex channel mixture는 PCA의 부호·혼합으로 달라지는 입력 분포를 줄일 수 있다. 그러나 여전히 F의 비선형성은 남고, 채널 수 축소가 idiosyncratic 미래 정보를 지울 수 있다. 이 후보도 같은 정보·예산·decoder와 대조해야 한다.

[추정] MSE와 MAE의 충돌이 tail/bias에 집중된다면 robust 보조항 또는 point forecast 요약의 차이를 시험할 수 있다. Chronos의 반환 median을 조건부 mean이라고 가정하지 않는다. 기존 모든 RAW baseline에도 같은 선택 규칙을 적용해야 한다.

후속 학습의 구체적인 방법·weight·K·seed·resource와 독립 평가 조건은 현재 Bi 결과 및 D0–D3 뒤 새 계획으로 고정한다. 후보 셋을 먼저 고정하고 결과가 나쁜 같은 계열을 세 번 반복하면 그 계열의 전제와 비교 조건을 재평가한다. 이는 상위 목표 중지가 아니라 다른 근거 있는 해결 경로를 찾는 gate다.

## 선행 근거와 한계

- [Hinton 등, Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531): teacher 행동을 작은 모델로 옮기는 일반적 근거. 이 연구의 TSFM channel compression 개선을 직접 입증한 논문은 아니다. arXiv 원문이다.
- [Chronos 공식 구현](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos_bolt.py): quantile 예측과 median을 point 값으로 반환하는 구현. 후속 실제 실험은 웹 main 대신 로컬 설치 source·모델 snapshot을 확인해 봉인한다.
- [Chronos 공식 저장소](https://github.com/amazon-science/chronos-forecasting): foundation interface 배경. 새로운 모델 버전으로 바꿔 기존 baseline 실패를 덮지 않는다.

현재 가장 약한 전제는 낮은 teacher 출력 rank가 압축된 과거로부터 그 좌표를 예측할 수 있다는 가정이다. D1/D2가 이를 검증한다. 현재 문서는 원인 확정·실행 완료 보고서가 아니다.

