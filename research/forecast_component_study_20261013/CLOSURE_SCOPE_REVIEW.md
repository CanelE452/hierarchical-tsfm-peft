# Forecast Component Study 종료 범위 감사

작성일: 2026-10-13  
감사 방식: 새 실험 없이 `PLAN_KO.md`, Stage A/A2/B/self-lag 결과, 소스 해시, 독립 검산 파일을 읽어 범위 충족 여부만 확인했다.

## 결론

[확인] 실험 본체는 닫을 수 있다. `PLAN_KO.md`의 핵심 목표였던 “기존 FORECAST 실패 원인 좁히기”와 “learned low-dimensional representation + frozen Bolt + correction의 정확도/비용 진단”은 Stage A, Stage A2, Stage B full, self-lag reference, 독립 검산까지 실제 산출물이 있고 모두 PASS다.

[확인] 다만 완료 조건 5의 독자용 산출물은 아직 완전히 닫히지 않았다. `stage_b_full_v1` 아래에는 Stage B 통합 한국어 보고서와 그림이 없고, 시간블록별 paired 차이를 바로 읽을 수 있게 정리한 요약도 별도 산출물로 보이지 않는다. 이는 새 실험 누락이 아니라 보고 산출물 누락이다. 기존 `RESULT.json`에는 horizon/per-channel metric과 prediction NPZ가 있어 계산은 가능하다.

## 완료 조건별 판정

- [확인] 조건 1은 충족했다. Stage A는 저장된 FORECAST 원래 head, final checkpoint, TUNE 선택 checkpoint, 같은 표현의 새 ridge를 평가했고, 독립 검산은 Stage A `records_rescored=38`, 실패 체크 0개로 PASS다. Stage A 보고서는 선택 기준이 TUNE physical channel-macro MSE이며 VAL은 선택 checkpoint와 final600 비교에만 사용했다고 명시한다.

- [확인] 조건 2는 충족했다. Stage B에는 PCA, REC, PRED가 있고 REC/PRED nonlinear case는 두 seed로 수행됐다. `component_experiment.py`는 REC/PRED의 초기 codec hash와 batch order가 다르면 중단하도록 되어 있다. PCA는 deterministic baseline으로 seed 92601만 있다. 주의할 점은 Stage B REC가 확인된 모든 REC case에서 selected update 0이라는 것이다. 따라서 REC를 “학습된 재구성 codec의 성능”처럼 쓰면 안 되고, “Bolt 기준 TUNE 선택에서 초기 PCA형 codec이 선택됐다”고 써야 한다.

- [확인] 조건 3은 충족했다. Stage B method family에는 `RAW_R_ONLY`, `RAW_BOLT`, `RAW_BOLT_RAW_R`, compressed `*_BOLT`, `*_BOLT_RAW_R`, `*_BOLT_RESIDUAL_R`가 모두 있다. 전체 Stage B method count는 76이다.

- [확인] 조건 4는 충족했다. Stage B에는 native trainable shared linear 경로(`*_NATIVE`, `*_NATIVE_RAW_R`), persistence, common decoder(`*_BOLT_COMMON_D`)가 있다. 다만 native는 frozen Bolt와 동일 비용 조건의 공정한 대체물이 아니라 “Bolt를 유지할 근거가 있는지 보는 스트레스 테스트”로 해석해야 한다. common decoder도 D-only causal proof가 아니라 진단용 대조다.

- [부분 충족] 조건 5의 실험 기록과 검산은 충족했다. Stage B `RESULT.json`에는 physical/normalized MSE·MAE, per-channel metric, horizon metric이 있고 `IMPLEMENTATION_AND_REPRODUCIBILITY.json`에는 `train_updates=10800`, `timed_batches=152`, prediction hash parity, cost notes가 있다. 독립 검산 `validation_final_v2.json`은 Stage A, A2, Stage B preflight, Stage B full, self-lag 모두 PASS다. 남은 것은 Stage B 통합 한국어 보고서/그림과 시간블록 paired delta의 독자용 정리다.

## 누수·동결·소스 무결성

[확인] TEST 접근 차단은 기록된 두 protected TEST path에 대해 걸려 있다. `common.py`의 `access_guard()`는 data contract에서 TEST path를 읽어 audit hook으로 차단하고, `component_experiment.py`, `diagnose_forecast.py`, `diagnose_dense_forecast.py`, `self_lag_baseline.py`가 실행 초기에 이를 호출한다. Stage B 결과에도 보호 TEST path 두 개가 기록되어 있다. 단, 이 hook은 기록된 TEST 경로에 대한 차단이지 이름이 바뀐 복사본까지 원천적으로 증명하는 장치는 아니다.

[확인] Stage B leakage fixture는 Jena/Robin 모두 `future_perturbation_past_max_abs=0.0`, `official_median_max_abs=0.0`, `prefix_full_max_abs=9.5367431640625e-07`, positive control 양수로 기록되어 있다. 이는 현재 구현에서 미래 교란이 과거 표현에 새지 않았다는 강한 구현 검산이다.

[확인] Bolt는 동결 상태로 유지됐다. Stage B fixture의 `trainable_bolt_parameters=0`이고 `bolt_initial_hash`와 `bolt_final_hash`가 같다. `IMPLEMENTATION_AND_REPRODUCIBILITY.json`에서 `component_experiment.py`, `common.py`, `PLAN_KO.md`는 실행 hash와 현재 hash가 일치한다.

## 핵심 수치와 해석

Stage A2는 기존 FORECAST 실패 원인을 상당히 좁혔다. 같은 구조·초기값·loss·batch/update 조건에서 origin pool만 sparse에서 dense256으로 바꾸자 TUNE 선택 VAL MSE가 네 조건 모두 개선됐다.

```text
Jena  seed92601: sparse 32063.406 -> dense 12887.576
Jena  seed92602: sparse 33796.728 -> dense 15812.935
Robin seed92601: sparse  1376.123 -> dense   646.548
Robin seed92602: sparse  1312.848 -> dense   819.041
```

[확인] 이는 “기존 FORECAST가 단순히 head 때문에 망했다”보다 “학습 origin pool 설계가 큰 원인 중 하나였다”는 결론을 지지한다. 다만 이 결과만으로 learned low-dimensional frozen Bolt 경로가 성공했다고 말할 수는 없다.

Stage B의 bounded VAL 정확도는 압축 경로의 채택 근거를 강하게 주지 않는다.

```text
Jena physical VAL MSE
RAW_R_ONLY              10978.293
PRED_NATIVE_RAW_R       11702.367
PRED_BOLT               12988.519
RAW_BOLT_RAW_R          12520.367
RAW_BOLT                13981.151
REC_BOLT                18935.994

Robin physical VAL MSE
RAW_BOLT                  151.554
RAW_BOLT_RAW_R            154.439
PRED_BOLT                 243.308
REC_BOLT                  257.023
PRED_NATIVE_RAW_R         292.330
RAW_R_ONLY                826.786
RAW_SELF_LAG              512.196
```

[확인] Jena에서는 PRED low-dimensional path가 REC/PCA보다 낫고 raw Bolt 단독보다 physical MSE를 낮추는 경우가 있지만, strongest raw correction baseline인 `RAW_R_ONLY`나 `RAW_BOLT_RAW_R`를 이기지는 못한다. Robin에서는 raw Bolt가 압축 Bolt보다 훨씬 낫다. 따라서 “learned low-dimensional frozen Bolt + correction이 정확도에서 이긴다”는 결론은 현재 bounded VAL에서는 지원되지 않는다.

[확인] 비용 측면에서는 compressed Bolt가 raw Bolt보다 batch8 latency가 낮다.

```text
Jena  RAW_BOLT 0.003992 s/window, PRED_BOLT 0.001879 s/window
Robin RAW_BOLT 0.003416 s/window, PRED_BOLT 0.001818 s/window
```

하지만 native/ridge 계열은 더 빠르고, 메모리 수치는 co-resident process 기준이다. Stage B 전체 resource는 wall 157.295초, cumulative GPU process wall 175.315초, peak allocated VRAM 400,394,240 bytes, peak sampled RSS 2,004,234,240 bytes다. 그러므로 비용 이득은 “raw Bolt 대비 compressed Bolt 추론 latency 감소” 정도로만 제한해야 한다.

## 보고서에서 피해야 할 주장

- [확인] 독립 일반화 검증이라고 쓰면 안 된다. VAL은 이미 개발 진단에 반복 사용됐고 TEST는 보호됐다.
- [확인] universal success/failure라고 쓰면 안 된다. 두 데이터셋, REC/PRED 두 seed, `K=floor(C/4)` 한 압축률, 중첩 VAL window 조건이다.
- [확인] physical MSE와 normalized MSE 순위가 항상 같다고 쓰면 안 된다. 예를 들어 Robin의 `PRED_BOLT`와 `PRED_BOLT_COMMON_D`는 physical/normalized 순위가 다르다. 데이터셋 간 physical metric도 채널 단위가 달라 풀링하면 안 된다.
- [확인] REC가 학습 후 좋아졌다고 쓰면 안 된다. 확인된 REC selected update는 모두 0이다.
- [확인] common decoder 결과를 D-only 인과 결론으로 쓰면 안 된다. E/D와 보정기는 진단적으로 분리했지만 완전한 causal isolation은 아니다.
- [확인] native linear 결과를 frozen Bolt와 동일 비용·동일 조건의 공정 비교라고 쓰면 안 된다. native는 추가 학습된 예측 head를 가진다.
- [확인] VRAM/RSS를 method-exclusive memory saving으로 쓰면 안 된다. 기록 자체가 co-resident benchmark process 기준이라고 명시돼 있다.

## 종료 판단

[확인] 새 실험은 필요하지 않다. 현재 결과는 “기존 FORECAST 실패는 sparse origin pool 문제가 컸고, learned low-dimensional PRED는 일부 압축 비교에서 REC/PCA보다 낫지만 strongest raw baseline을 넘지 못했으며, frozen Bolt를 압축 경로에 붙이는 정확도 근거는 약하고 raw Bolt 대비 latency 절감 근거만 제한적으로 남는다”까지 말할 수 있다.

[확인] 최종 닫기 전에 필요한 일은 기존 Stage B 산출물에서 통합 한국어 보고서·그림·시간블록 paired delta 요약을 만드는 것이다. 이 작업은 새 실험이 아니라 기존 결과의 보고 정리다.
