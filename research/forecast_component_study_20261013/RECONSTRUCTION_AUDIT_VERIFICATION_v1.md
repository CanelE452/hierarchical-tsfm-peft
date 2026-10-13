# reconstruction_audit_v1 독립 검산 메모

판정: **PASS_WITH_NUMERIC_TOLERANCE**

[확인] 새 학습이나 Bolt 실행 없이, 저장된 fixed input과 reconstruction/latent NPZ 배열을 다시 읽어 40개 snapshot의 FIT/TUNE 재구성 MSE/MAE를 독립 계산했다. 최대 차이는 Jena PRED seed92601 update200 TUNE physical channel18 MSE 약 91,280.63에서 0.00313이며, float32 저장 배열을 float64로 재계산한 정밀도 차이로 판단했다.

## 검산 결과

```text
cases          8
snapshots      40
metrics rows   80
metric max diff 0.0031301474664360285
csv max diff    0.00025845070922514424
```

[확인] `VERIFICATION.json`의 RESULT/REPORT/metrics SHA는 현재 파일 SHA와 일치한다. 저장 배열 SHA도 40개 모두 일치했다. causal fixture는 40개 모두 pass이고 latent는 finite/non-constant 조건을 통과했다.

## REC 0→600 요약

```text
dataset seed  FIT norm MSE 0->600   TUNE norm MSE 0->600   forecast TUNE MSE 0->600
jena   92601  0.147215->0.145960       0.400644->0.460918       20896.801->21207.776
jena   92602  0.147215->0.146069       0.400644->0.495028       20896.801->21223.095
robin  92601  0.116615->0.110079       0.038509->0.047335       81.964->87.416
robin  92602  0.116615->0.110178       0.038509->0.049814       81.964->90.045
```

[확인] REC final600은 FIT 재구성 MSE에서 4/4 개선, TUNE 재구성 MSE에서 0/4 개선이다. 즉 TUNE에서는 4/4 악화했다. 미래 TUNE physical MSE는 4/4 조건에서 악화했다.

## 해석 경계

[확인] 이 검산은 reconstruction audit 자체의 배열·수치 일관성 확인이다. 재구성 목적과 미래 예측 목적의 불일치 가능성을 지지하지만, encoder 정보 손실·latent 예측성·decoder 민감도 중 하나를 단일 원인으로 확정하지는 않는다.

[확인] 이 진단은 report_v1 CSV 검산 및 CLOSURE_SCOPE_REVIEW_V2와 별도 산출물이다. report_v1에 reconstruction 문단이 추가되면 해당 report SHA는 다시 묶어야 한다.