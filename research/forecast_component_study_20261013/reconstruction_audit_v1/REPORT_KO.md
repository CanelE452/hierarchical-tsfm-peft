# 고정 과거 창의 재구성 학습 재평가

[확인] 40개 저장 codec을 다시 실행했다. REC final600의 정규화 재구성 MSE는 초기값 대비 고정 FIT에서 4/4, 고정 TUNE에서 0/4 조건이 낮았다. 같은 checkpoint의 미래 TUNE 물리 MSE는 4/4 조건에서 높아졌다. 아래 원수치와 지표별 차이를 함께 보아야 한다.

## REC 초기값과 final600: 같은 입력에서의 비교

```text
자료   seed     FIT norm MSE (0→600)    TUNE norm MSE (0→600)    미래 TUNE phys MSE (0→600)
jena   92601   0.147215→0.145960       0.400644→0.460918       20896.801→21207.776
jena   92602   0.147215→0.146069       0.400644→0.495028       20896.801→21223.095
robin  92601   0.116615→0.110079       0.038509→0.047335       81.964→87.416
robin  92602   0.116615→0.110178       0.038509→0.049814       81.964→90.045
```

학습 목적은 정규화 channel-macro MSE다. 물리 MSE는 채널별 단위가 달라 같은 순위를 보장하지 않는다. [모든 snapshot의 정규화·물리 MSE/MAE](metrics.csv), [채널별 수치와 latent 통계](RESULT.json)에 원값을 남겼다.

## 해석 범위

[판단] 같은 고정 입력에서 재구성 오차가 낮아졌는데 미래 예측 오차가 높아진 조건은, 단순한 loss 미감소보다 재구성 목적과 실제 예측 목적이 어긋날 수 있다는 구체적 증거다. 단, 이 결과만으로 encoder의 정보 손실, 잠재 미래의 예측 불가능성, decoder 민감도 중 하나를 유일 원인으로 특정하지는 못한다. 재구성도 나빠진 조건은 학습/일반화 문제를 별도로 남긴다.

PRED snapshot도 같은 방식으로 평가했지만, PRED는 미래 예측 오차로 E/D를 함께 최적화한 모델이다. 그 D를 과거의 실제 latent에 적용한 재구성 오차는 학습 목표가 아니며, 나쁘다고 PRED의 예측 목적 실패로 판정하지 않는다.

원래 REC 네 조건의 checkpoint 선택은 미래 TUNE 예측 MSE로 초기0을 선택했다. 이번 고정 재구성 검사로 checkpoint나 방법을 다시 선택하지 않는다. latent의 분산·최소std가 0보다 큰 것은 상수 출력 붕괴 검사이며, 유용한 정보 보존이나 동결 Bolt 호환의 증거는 아니다.

## 정보 경계와 검산

- FIT256/TUNE64의 과거512만 E/D에 입력했다. 목표는 각 창의 같은 과거값이고 observed mask만 채점했다. TUNE 과거 창이 FIT와 겹치는 기존 계약을 유지하므로 독립적인 TUNE 시점만의 재구성 평가라고 부르지 않는다.
- 원자료는 기존 단일 bounded TRAIN/VAL 파일을 읽었지만 VAL 창·target·새 점수는 만들지 않았다. TEST guard를 등록했다. Bolt 모델 로드/실행, 추가 학습은 0이다.
- data/source/checkpoint byte SHA 및 Stage B의 고정 origin·FIT mean/std 일치를 확인했다. 저장 배열을 재독해 별도 NumPy 채널 루프로 MSE/MAE를 계산했다. 40개 snapshot의 causal 변형·최근 관측 양성대조도 수행했다.
- 미래 TUNE 예측 값은 기존 Stage B RESULT의 checkpoint metadata를 SHA로 연결해 재사용했다. 이번 CPU 재구성 실행과 GPU에서 생성됐던 미래 예측 실행은 같은 새 실행으로 표현하지 않는다.
- 재구성/latent 배열과 고정 입력/mask는 새 로컬 cache에 저장하고 SHA를 기록했다. 원자료·모델은 공개 저장소 산출물에 복사하지 않았다.

실제 본체 CPU 21.812초, wall 13.716초, GPU0, threads2. Python import/프로세스 시작은 제외한다. 기존 파일·결과 수정 및 commit/push는 없다.

```powershell
.venv/Scripts/python.exe research/forecast_component_study_20261013/diagnose_reconstruction.py
```

고정 v1 산출물이 존재하면 덮지 않고 중단한다. 새 실행은 별도 버전이 필요하다.
