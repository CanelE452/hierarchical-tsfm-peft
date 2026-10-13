# Forecast Component Study 종료 범위 감사 V2

작성일: 2026-10-13  
성격: 기존 `CLOSURE_SCOPE_REVIEW.md/json`을 보존하고, `report_v1` 생성 이후 CSV·그림·링크 검산과 해석 정정을 반영한 V2 감사다.

## 판정

[확인] 핵심 실험과 `report_v1` 보고 산출물은 닫을 수 있다. Stage A/A2/B/self-lag의 실행과 독립 검산은 이미 PASS였고, 이번에는 `report_v1`의 78개 method 표, 156개 batch latency, 52개 paired contrast, 208개 chronological block, 1482개 channel metric을 원본 JSON과 prediction NPZ로 다시 대조했다.

[확인] `REPORT_VERIFICATION_v1` 판정은 `PASS_WITH_NUMERIC_TOLERANCE`다. 허용한 차이는 부동소수점/출력 정밀도 수준이다.

```text
all_78_methods.csv       78 rows
batch_latency.csv       156 rows
paired_contrasts.csv     52 rows
chronological_blocks.csv 208 rows
val_channel_metrics.csv 1482 rows
```

[확인] `all_78_methods.csv`는 Stage B/self-lag `RESULT.json`과 최대 차이 0.0으로 일치했다. prediction SHA도 모두 일치했다. `paired_contrasts.csv`는 method table에서 다시 계산했을 때 최대 차이 `1.42e-14`였다. Stage B latency 원본과 batch latency CSV도 0.0ms 차이로 일치했다.

[확인] NPZ 재채점에서 남은 최대 차이는 다음과 같다.

```text
VAL aggregate max abs diff       0.00021115567506058142
VAL channel max abs diff         0.004750268184579909
Chronological block max abs diff 0.0002235882930108346
```

가장 큰 channel 차이는 Jena PAR 채널 physical MSE 약 368,463에서 0.00475 차이다. 이는 float32 저장 prediction을 float64로 재채점한 수준의 상대 오차라서 metric 불일치로 보지 않는다.

## V1 감사 정정

[확인] 기존 V1 감사의 “Jena compressed+correction이 RAW_R_ONLY나 RAW_BOLT_RAW_R를 이기지는 못한다”는 표현은 과했다. 정확한 결론은 metric tradeoff다.

```text
Jena PRED_BOLT_RAW_R seed92601: MSE 13058.848, MAE 37.560, nMSE 0.9607, nMAE 0.5655
Jena PRED_BOLT_RAW_R seed92602: MSE 12185.094, MAE 36.941, nMSE 0.9436, nMAE 0.5635
Jena RAW_BOLT_RAW_R:           MSE 12520.367, MAE 35.287, nMSE 0.8445, nMAE 0.4664
Jena RAW_R_ONLY:               MSE 10978.293, MAE 34.266, nMSE 0.8782, nMAE 0.5111
```

따라서 seed92602에서는 `PRED_BOLT_RAW_R`가 `RAW_BOLT_RAW_R`보다 physical MSE만 낮다. 그러나 MAE와 normalized metric은 더 나쁘고, seed92601에서는 physical MSE도 더 나쁘다. 두 seed 모두 `RAW_R_ONLY`보다 physical MSE/MAE가 나쁘다. 보고서 본문은 이 과장을 하지 않고 두 seed와 metric 차이를 보여준다.

## 현재 말할 수 있는 것

[확인] 말할 수 있는 결론은 다음이다.

- 기존 FORECAST 부진은 sparse origin pool 설계가 큰 원인 중 하나였다.
- PRED 목적은 compressed path 안에서 REC/PCA보다 좋아지는 조건이 있지만, raw strong baseline을 안정적으로 넘는 정확도 근거는 아직 약하다.
- compressed Bolt는 측정된 batch8 pipeline에서 raw Bolt보다 latency가 낮지만, native/ridge 계열과 비교한 비용 우위나 method-exclusive memory saving으로 읽으면 안 된다.
- Jena 일부 metric에는 compressed+correction의 physical-MSE tradeoff가 있으므로 “항상 못 이긴다”나 “우위다” 둘 다 부정확하다.

[확인] 말하면 안 되는 결론은 그대로다. 독립 TEST 일반화, 보편적 정확도 우위, D-only 인과 결론, physical/normalized metric 동일 순위, REC 학습 개선, native와 frozen Bolt의 동일비용 공정비교, method-exclusive VRAM 절감은 주장할 수 없다.

## Reconstruction audit 경계

[확인] 이 V2 감사 시점에는 `reconstruction_audit_v1` 디렉터리가 없었다. 이후 40개 saved codec E/D-only CPU 진단이 생성되면 이 V2의 일부가 아니라 별도 진단으로 검산해야 한다. 해당 검산은 저장 reconstruction 배열, source/report SHA, FIT/TUNE past-only 조건, no VAL/Bolt/no new training 조건을 따로 확인하면 된다.

## 묶인 파일

- `REPORT_VERIFICATION_v1.md`
- `REPORT_VERIFICATION_v1.json`
- `report_v1/REPORT_KO.md`
- `report_v1/all_78_methods.csv`
- `report_v1/batch_latency.csv`
- `report_v1/paired_contrasts.csv`
- `report_v1/chronological_blocks.csv`
- `report_v1/val_channel_metrics.csv`
