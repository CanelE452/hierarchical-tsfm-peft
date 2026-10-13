# 평가 기준을 맞춘 학습 일반화·압축·예측기 대조

2026-10-13. 사용자가 앞서 제시한 “평가 목표 고정 → 학습 일반화 → K·시간 범위 → Bolt의 추가 가치” 순서에 `그렇게 해줘`로 실행을 승인했다. 기존 연구를 재개하지 않고 새 연구 폴더와 cache에서 수행한다.

## 목적과 소비처

연구 목적은 다변량 과거로 만든 작은 성분을 동결 TSFM에 연결할 실질적 근거를 찾는 것이다. 이번에는 학습·선택 지표 불일치와 작은 학습 변경으로 해결 가능한 일반화 문제부터 분리한다. 사용자와 교수님이 후속 방법을 고를 수 있도록 원수치·반례·비용을 보고한다. 성능 우위가 없어도 정해진 대조를 완료한다.

시작 HEAD는 `e57d0f01f148dd4004c53531c7c24c45198636ab`다. 기존 연구 파일·결과·cache·다른 프로젝트의 프로세스는 보존한다. 보호 TEST, 이전 중단 캠페인, 대형 백본 재학습, 자동 commit/push를 실행하지 않는다.

## 주평가지표와 데이터 계약

- 사용자가 주지표 질문에 **정규화 MSE — 채널 균형과 큰 오차 감소를 우선**으로 답하여 확정했다. FIT 표준편차로 정규화한 channel-macro MSE에 학습 손실, 미래 TUNE checkpoint 선택, ridge lambda 선택을 맞춘다. 정규화 MAE 및 물리 단위 MSE/MAE는 모든 조건에서 함께 보고한다. 다른 지표로 변경 요청이 오면 이미 실행된 결과를 보존하고, 변경 후 실행과 구분한다.
- Jena C21, Robin C17의 기존 bounded TRAIN/VAL 원자료만 사용한다. FIT/TUNE 경계·통계·FIT256/TUNE64/VAL371·30 target origins는 앞 연구와 정확히 같다. source/origin/mask/stat SHA로 검사한다.
- TUNE는 선택, VAL은 개발 진단에 사용한다. VAL로 hyperparameter·K·history·checkpoint를 선택하지 않는다. 이미 반복 사용한 VAL이므로 독립 일반화 검증이라고 주장하지 않는다.
- H48, float32, pinned Chronos-Bolt-small median의 첫48개, 가중치 고정. 기본 L512, RF127, K=floor(C/4), seed92601/92602다. 같은 설정의 REC/PRED와 optimizer 비교는 초기 codec와 600×4 batch index를 공유한다.

## 선행 근거와 후보

- [AdamW, ICLR2019](https://arxiv.org/abs/1711.05101)는 Adam의 coupled L2와 decoupled weight decay를 구분한다. 이미지 분류의 일반화 결과를 우리 자료의 성능 보장으로 확장하지 않는다.
- [PyTorch AdamW 공식 문서](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)와 설치된 PyTorch2.10 소스를 따라 optimizer를 구현한다. 기본 betas/eps를 유지한다.
- [StandardScaler 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)의 학습 자료 기반 표준화 원칙을 적용한다. 기존 FIT-only 통계를 그대로 사용한다.
- [CPC](https://arxiv.org/abs/1807.03748)의 미래 표현 학습은 후속 후보다. 이번에는 실제 Bolt 미래 손실이라는 직접 기준보다 무엇을 더 개선해야 하는지 먼저 확인한다.

후보는 기본 학습, 낮은 학습률, decoupled weight decay의 세 가지다. 낮은 학습률은 초기 PCA 근처의 급격한 변화 가능성, weight decay는 학습 자료에 치우치는 변화 가능성을 검사한다. 가장 단순한 후보인 학습 전 PCA/항등을 항상 남긴다. 특히 **K=C여도 PCA 회전은 Bolt의 입력을 바꾸므로 RAW와 같지 않다**는 반례를 활용해 full-rank PCA와 identity 초기값을 둘 다 비교한다. 이는 압축하지 않는 조건을 넣어 연구의 압축 가정을 반증하는 대조다.

## G0 — 기존 결과 재사용과 선택 기준의 영향

기존 Stage B의 저장 checkpoint0/100/200/400/600 및 과거 재구성 진단의 TUNE 점수에서 정규화 주지표의 최저점을 선택한다. REC는 과거 재구성 TUNE와 미래 예측 TUNE의 선택을 각각 기록한다. 선택은 TUNE 점수만 사용하며, 선택된 checkpoint의 실제 미래 예측은 새 폴더에 재평가한다. 같은 state의 중복 추론을 재사용하면 출처를 명시한다. 기존 모델 fit과 신규 추론을 구분하고 기존 결과를 덮지 않는다.

## G1 — 고정 조건에서 학습 일반화

두 자료×두 seed×REC/PRED×세 설정, 총24개 fit을 최대600 updates씩 실행한다.

```text
BASE      AdamW lr0.001   weight_decay0
LOW_LR    AdamW lr0.0003  weight_decay0
DECAY     AdamW lr0.001   weight_decay0.01
```

기존 Adam(wd0)과 AdamW(wd0)의 수학적 동등성은 첫5 update의 실제 parameter parity fixture로 확인하고 RNG·초기값을 복구한다. 새 내부 세 설정은 같은 optimizer class를 사용한다. 이 검사만으로 기존 전체 학습 궤적의 bitwise 재현을 주장하지 않는다.

E는 이전 PCA 초기 projection+causal convolution(hidden16, kernel3, 6층), D는 linear다. REC는 정규화 과거 재구성 MSE, PRED는 실제 동결 Bolt를 통과한 미래 MSE로 E/D를 학습한다. batch4, gradient clipping1, 같은 배치 순서, 체크포인트0/100/200/400/600을 유지한다.

각 checkpoint에서 같은 FIT/TUNE 창의 재구성 및 미래 예측 MSE/MAE를 평가한다. stochastic minibatch loss를 고정 데이터의 일반화 증거로 쓰지 않는다. REC는 복원 기준과 미래 예측 기준의 선택 state를 모두 보존하고 각각 VAL을 평가한다. PRED는 미래 TUNE 기준을 사용한다. update0을 선택 후보에서 제외하지 않는다.

TUNE 과거 창에는 FIT 시점이 섞이므로, 각 checkpoint의 TUNE 재구성에 **FIT 경계 이후의 관측 위치만** 포함한 별도 점수와 관측 수를 추가한다. 전체 창의 기존 선택 규칙은 바꾸지 않는다. 겹치는 창에서 같은 시점이 반복 가중되는 점은 유지·명시하며, 이 보조 점수도 독립 일반화 표본이라고 부르지 않는다.

## G2 — 성분 수와 실제 과거 범위

G1 완료 후 자료별 PRED의 **두 seed 평균 미래 TUNE 정규화 MSE**로 설정을 정한다. 동일값(1e-12 이내)은 BASE 우선이다. 최선 설정의 BASE 대비 개선이1% 미만이면 G2에서는 BASE를 유지한다. 이1%는 제한된 실용 screen 기준이지 통계적 유의성 기준이 아니다. G1 모든 수치는 그대로 보고한다.

아래 PRED 변형을 같은 target origins·H·선택 규칙·600 updates·두 seed에서 수행한다. G1의 같은 quarter-K 기본 조건은 재사용한다.

- K=floor(C/2), L512, RF127, PCA 초기화.
- K=C, L512, RF127, full-rank PCA 초기화.
- K=C, L512, RF127, identity basis/zero mean 초기화. 초기 E/D 항등 조건을 확인하고 RAW Bolt를 별도 비교한다.
- K=floor(C/4), L128, RF127. 각128 창의 시작을 FIT mean으로 초기화해 다시 forward fill한다. 512 창을 먼저 채운 뒤 잘라서 이전 관측을 끌고 오지 않는다. 길이와 결측 처리 경계가 함께 바뀌는 조건임을 명시한다.
- K=floor(C/4), L512, RF31. RF127의 `(1,2,4,8,16,32)`와 같은6개 convolution의 dilation을 `(1,1,1,2,4,6)`으로 바꾼다. 파라미터 개수·초기 weight는 같고, RF=1+2Σd이다. dilation 위치도 바뀌므로 순수 정보 범위만의 효과라고 과장하지 않는다.

G2는 총20개 추가 PRED fits다. 성능이 나쁘다는 이유로 성분 수·시간 범위 대조를 생략하지 않는다. 누출·구현 오류·자원 초과는 해당 경로만 중단하고 가능한 독립 대조를 계속한다.

## G3 — 예측기·복원·보정과 비용

G1 PRED 선택 모델과 G2 모델에 동일 E/D를 고정한 Bolt와 shared linear L→48 native를 연결한다. native는 최대600회 추가 학습하며 필요 학습 비용을 함께 기록한다. persistence, 같은 규칙의 reconstruction D 재적합, 원자료 Bolt, 원자료 joint ridge 단독·SELF33 단순 기준군을 포함한다.

보정은 기존33*C 특징(전체512의 16시점 평균32개+마지막값)을 유지하고, 미래 예측 오차에 fit한다. lambda는 `[1e-4,0.01,1,100,10000]` 중 정규화 TUNE MSE로 선택한다. Bolt+R와 native+R의 구조·FIT/TUNE origin·정답·선택 기준을 맞춘다.

L128 성분 경로에 R512를 결합하면 **성분 경로의 지원 범위는128, 전체 결합의 지원 범위는512**다. 비용과 정보 범위를 따로 표시한다. 공통 D는 E마다 같은 적합 규칙으로 얻으며, 동일한 수치의 D를 강제로 공유한 인과 대조로 부르지 않는다.

batch1/8, warmup2, 반복7, CUDA synchronize로 전처리·H2D·E/F/D/R·역정규화·D2H 전체 경로를 측정한다. 모델 로드·디스크 읽기는 별도다. process 공존 메모리는 방법별 독립 절감으로 주장하지 않는다.

새 미래 정렬·대조학습·지연 그래프는 자동으로 추가하지 않는다. G1–G3에서 남은 병목과 직접 예측 기준 대비 요구되는 추가 이득을 보고한 뒤 판단할 후보다.

## 검증·보고와 완료 조건

1. 실제 HEAD/자산/실행 작업과 source·data·origin·초기값·배치 SHA 확인.
2. 실제 frozen Bolt gradient/update·공식 추론 parity·causal counterfactual·L128 경계·RF/K 초기값 대조 통과.
3. G0 재사용/재추론, G1 신규24 fits, G2 신규20 fits, G3 predictor/control/비용을 구분한 결과와 정상 종료 기록.
4. 독립 재채점으로 선택·점수·파일SHA·정보 경계·동결·실제 비용 경로를 확인. channel/seed/time-block 결과를 함께 보고.
5. 한국어 보고서·CSV·그림을 작성하고 실물 이미지·상대링크를 확인. 안 된 경로는 상태·이유·재개 방법·추가 자원을 표시한다.

## 자원과 중단

이번 후속의 보수적 상한은 GPU 사용 process wall45분, CPU-only 검산/보고 wall30분, 전체 wall2시간, 새 저장4GiB, process RAM/allocated VRAM8GiB다. 과거 연구의 resource ledger를 수정하지 않는다. 실제 profiler로 예상 시간을 갱신한다. 낮은 정확도만으로 원인 분리 대조를 중단하지 않으며, 설정 수를 계속 늘리는 탐색은 하지 않는다.

예상되는 결론은 학습 변경의 일반화 효과, 평가 기준의 영향, 압축량·시간 범위의 교환, 특정 자료에서 동결 Bolt가 단순 대안보다 남기는 이득이다. 두 자료·두 seed와 반복 개발 VAL의 제한을 유지한다.
