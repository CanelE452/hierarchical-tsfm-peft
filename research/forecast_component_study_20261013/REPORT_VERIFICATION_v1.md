# report_v1 독립 검산 메모

작성일: 2026-10-13  
대상: `research/forecast_component_study_20261013/report_v1`  
판정: **PASS_WITH_NUMERIC_TOLERANCE**

## 검산 범위

[확인] 새 실험이나 학습은 실행하지 않았다. `REPORT_KO.md`, `all_78_methods.csv`, `batch_latency.csv`, `paired_contrasts.csv`, `chronological_blocks.csv`, `val_channel_metrics.csv`를 Stage B/self-lag 원본 JSON과 prediction NPZ로 대조했다.

[확인] 이 검산은 아래 SHA에 묶인 point-in-time 검산이다. 이후 `REPORT_KO.md`에 reconstruction audit 문단이 추가되면 그 변경분은 별도 검산 대상이다.

```json
{
  "report_sha256": "a1bfc07bb889bf0748b9f89d73cb769d77281817c8d03c0d63a940510a37427e",
  "build_check_sha256": "af53b9ee55e4494ffefc81b8280b38a7710551a9bdab67ccd4f22fac8178e4e7",
  "stage_b_result_sha256": "c5b0b9ed505dab0cbd4de54faec59912f41beb84725ef95aaad820022ef77077",
  "self_lag_result_sha256": "8dcec05d906509e14f55eface5ddb790c09f836ac80cab385f595b1a3f6ae667"
}
```

## 행 수와 링크

[확인] `REPORT_BUILD_CHECK.json`은 PASS이고 markdown 상대 링크 존재 여부도 PASS다.

```text
all_78_methods.csv       78 rows
batch_latency.csv       156 rows = 78 methods × batch(1,8)
paired_contrasts.csv     52 rows
chronological_blocks.csv 208 rows = 52 paired contrasts × 4 chronological blocks
val_channel_metrics.csv 1482 rows = Jena 39×21 + Robin 39×17
```

## 수치 대조

[확인] `all_78_methods.csv`의 FIT/TUNE/VAL metric은 Stage B와 self-lag `RESULT.json` 값과 최대 차이 0.0으로 일치했다. prediction SHA도 모두 일치했다.

[확인] `paired_contrasts.csv`는 `all_78_methods.csv`에서 candidate-reference를 다시 계산했을 때 최대 차이 `1.42e-14`였다.

[확인] `batch_latency.csv`는 Stage B latency 원본과 최대 차이 0.0ms로 일치했다. SELF33 네 행은 CPU reference 행으로 들어가며, CSV 내부의 repeat list median/min/max와 일치한다.

[확인] NPZ 재채점은 허용 가능한 부동소수점 차이만 있었다.

```text
VAL aggregate max abs diff       0.00021115567506058142
VAL channel max abs diff         0.004750268184579909
Chronological block max abs diff 0.0002235882930108346
```

가장 큰 channel 차이는 Jena `PCA_PERSISTENCE`, PAR 채널 physical MSE 약 368,463에서 0.00475 차이다. 이는 float32 prediction 저장값을 float64로 재채점한 수준의 상대 오차이며 metric 불일치로 보지 않는다.

## V1 scope review 정정

[확인] 기존 `CLOSURE_SCOPE_REVIEW.md`의 “RAW_R_ONLY나 RAW_BOLT_RAW_R를 이기지는 못한다”는 표현은 Jena의 일부 metric에 대해 과했다. 정확한 표현은 다음이다.

```text
Jena PRED_BOLT_RAW_R seed92601: MSE 13058.848, MAE 37.560, nMSE 0.9607, nMAE 0.5655
Jena PRED_BOLT_RAW_R seed92602: MSE 12185.094, MAE 36.941, nMSE 0.9436, nMAE 0.5635
Jena RAW_BOLT_RAW_R:           MSE 12520.367, MAE 35.287, nMSE 0.8445, nMAE 0.4664
Jena RAW_R_ONLY:               MSE 10978.293, MAE 34.266, nMSE 0.8782, nMAE 0.5111
```

따라서 Jena seed92602의 `PRED_BOLT_RAW_R`는 `RAW_BOLT_RAW_R`보다 physical MSE만 낮다. 그러나 MAE와 normalized metrics는 더 나쁘고, 두 seed 모두 `RAW_R_ONLY`보다 physical MSE/MAE가 나쁘다. 결론은 “metric tradeoff가 있다”이지 “항상 못 이긴다”가 아니다.

## Reconstruction audit 분리

[확인] 검산 시점에 `reconstruction_audit_v1` 디렉터리는 없었다. 이후 40개 saved codec E/D-only CPU 진단 결과가 생성되면, 이 보고서 검산과 분리해서 저장 reconstruction 배열을 1회 독립 재채점해야 한다.
