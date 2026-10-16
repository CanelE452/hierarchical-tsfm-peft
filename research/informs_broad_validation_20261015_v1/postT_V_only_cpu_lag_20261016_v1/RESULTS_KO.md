# Jena CPU 표현 비교: 전체 V 검산 결과와 남은 판단

**[확인] 네 인코더의 검색·fresh refit·전체 V43 예측, 독립 FP64 재채점, 선택 checkpoint의 새 CPU 6-origin 재추론을 완료했다. 시간 정보를 사용하는 세 조건은 B0보다 MSE·MAE가 낮았지만, 동일 V의 native ridge와 A-only CPU fresh G를 이기지 못했다.** CPU 비용과 기준선의 새 CPU 비교는 이 문서 작성 시 계측·검증 중이다. 이 보고서는 해당 네 조건의 검증된 정확도 결과를 정리한 부분 보고서이며 원 연구 전체 완료 보고서가 아니다.

## 무엇을 같은 조건에서 비교했나

Jena의 origin 이전 512행에서 미래48행을 예측한다. 원 채널은21개, 네 압축 조건의 잠재 차원은6, seed는92601이다. B0는 현재 시점의 채널 투영, B2는 과거 창 전체의 시간 혼합, LAG_CAUSAL은 과거 방향의7개 지연, LAG_BIDIR는 같은 과거 창 내부 양방향의7개 지연을 사용한다. 양방향이라는 이름이 예측 origin 이후 정답의 사용을 뜻하지 않는다.

네 조건은 같은 frozen Bolt-small, 선형 디코더, 현재값 anchor, 원 미래의 정규화 channel-macro MSE 손실, AdamW 학습률0.001, effective16/micro4, CPU2 threads를 사용했다. 입력 복원 오차만으로 학습한 비교가 아니라, frozen Bolt를 통과한 예측 오차의 gradient로 인코더를 적합한 비교다. 실행 계약과 실제 코드는 [PLAN](PLAN.json), [별도 REGISTRY](REGISTRY.csv), [cpu_lag_runner.py](cpu_lag_runner.py)에 있다.

A50의 전체 legal origins9923개에서0.25·1·3epoch까지 검색하고 A_stop으로 선택했다. 선택한 epoch 수만큼 A60에서 처음부터 새로 refit한 뒤 V43을 예측했다. 검색의 후보·노출 일정은 같지만, 선택된 refit 길이와 모델 용량까지 같다는 뜻은 아니다. 아래 노출 수는 반복을 포함한 누적 origin 노출이며 독립 표본 수가 아니다.

```text
조건          검색 최대  검색 update  검색 노출  선택/refit epoch  refit update  refit 노출
B0            3          1863         29769      3                 2254          36060
B2            3          1863         29769      1                  752          12020
LAG_CAUSAL    3          1863         29769      3                 2254          36060
LAG_BIDIR     3          1863         29769      1                  752          12020
```

선택·학습량의 원 영수증은 [B0 BASE](runs/postT_cpu_jena_E3_B0_k6_s92601_v1/BASE_FIT_RECEIPT.json)와 [refit](runs/postT_cpu_jena_E3_B0_k6_s92601_v1/refit/RECEIPT.json), [B2 BASE](runs/postT_cpu_jena_E3_B2_k6_s92601_v1/BASE_FIT_RECEIPT.json)와 [refit](runs/postT_cpu_jena_E3_B2_k6_s92601_v1/refit/RECEIPT.json), [과거 지연 BASE](runs/postT_cpu_jena_E3_LAG_CAUSAL_k6_s92601_v1/BASE_FIT_RECEIPT.json)와 [refit](runs/postT_cpu_jena_E3_LAG_CAUSAL_k6_s92601_v1/refit/RECEIPT.json), [양방향 지연 BASE](runs/postT_cpu_jena_E3_LAG_BIDIR_k6_s92601_v1/BASE_FIT_RECEIPT.json)와 [refit](runs/postT_cpu_jena_E3_LAG_BIDIR_k6_s92601_v1/refit/RECEIPT.json)이다. 네 BASE 상태는 모두 `COMPLETE_VALID`다. B0·과거 지연은 검색 마지막3epoch가 선택됐으므로, 검색 완료를 충분 수렴으로 해석하지 않는다.

## 전체 V43의 원수치

주지표는 원 채널을 균등하게 보는 정규화 channel-macro MSE이며 MAE를 함께 보고한다. 아래 일곱 결과는 같은 V43 cohort, data·scaler, C21/H48 조건이다. MSE·MAE 모두 원 물리 단위의 오차가 아니다. 기준선과 네 인코더가 같은 디코더·파라미터 수·학습량까지 사용하는 비교는 아니다.

```text
방법                       정규화 MSE         정규화 MAE
Raw Bolt-small             0.8491941389472357  0.4813883402485757
Native ridge               0.5566463357048567  0.40552180175406594
A-only CPU fresh G         0.632217057436123   0.44152128982113786
B0                         0.7841039310324822  0.5209622100413023
B2                         0.6834428963980711  0.49388492449756366
LAG_CAUSAL                  0.7432064184643359  0.5124572203253038
LAG_BIDIR                   0.7188903475533631  0.5033025985012579
```

네 신규 점수 CSV는 [B0](predictions/postT_cpu_jena_E3_B0_k6_s92601_v1/V_select/SCORES.csv), [B2](predictions/postT_cpu_jena_E3_B2_k6_s92601_v1/V_select/SCORES.csv), [LAG_CAUSAL](predictions/postT_cpu_jena_E3_LAG_CAUSAL_k6_s92601_v1/V_select/SCORES.csv), [LAG_BIDIR](predictions/postT_cpu_jena_E3_LAG_BIDIR_k6_s92601_v1/V_select/SCORES.csv)다. 기준선의 저장 점수는 [Raw](../predictions/primary_jena_E1_RAW_BOLT_SMALL_k0_s92601/V_select/SCORES.csv), [ridge](../predictions/primary_jena_E1_RAW_RIDGE_k0_s92601/V_select/SCORES.csv), [A-only CPU fresh G](../predictions/primary_jena_E1_DIRECT_G_k0_s92601__cpu_fresh_e3/V_select/SCORES.csv)다.

**이 표의 G는 `primary_jena_E1_DIRECT_G_k0_s92601__cpu_fresh_e3`다.** 다른 deep G checkpoint나 다른 부모에 연결한 H의 점수를 같은 G 조건으로 섞지 않았다. 기준선의 새 CPU parity·비용 완료 여부는 이 저장 V 점수의 존재와 별도로 확인해야 한다.

## 지금 허용되는 해석

**[확인] 시간 정보를 쓰는 표현은 이번 K6의 B0보다 나았다.** B2·과거 지연·양방향 지연 모두 B0보다 V MSE와 MAE가 낮다. 다만 이는 인코더 구조·용량·경계조건 및 선택된 refit 길이를 포함한 전체 방법 차이다. 시간 정보 하나만의 인과 효과나 지연 그래프의 필요성을 분리한 결과는 아니다. 네 인코더 중에서는 B2의 V MSE·MAE가 가장 낮았으므로, 지연 구조를 추가해야만 개선된다는 주장도 현재 결과로 뒷받침되지 않는다.

**[확인] 네 인코더는 native ridge·A-only CPU fresh G보다 두 지표 모두 나빴다.** 따라서 이 Jena 조건에서는 압축 표현과 frozen Bolt의 결합이 강한 단순 기준선보다 필요한 이유를 아직 확보하지 못했다. Bolt를 유지할 근거, 압축의 실제 배포 이득, 논문의 고유 기여는 미확정이다. 다만 원 채널 기준선과 K6 경로가 받는 정보·모델 구조가 달라, 이 결과 하나로 압축·잠재 미래 예측·복원 디코더 중 어느 단계가 원인인지 확정할 수는 없다.

**[확인] Raw 대비 결과는 지표에 따라 다르다.** 네 조건 모두 Raw보다 MSE는 낮지만 MAE는 높다. 정규화 MSE를 주지표로 정한 선택 규칙은 유지하며, 이를 모든 오차 지표의 개선으로 바꾸어 말하지 않는다.

**[추정·판단] 실패를 단순히 “입력 복원 목표로만 학습했기 때문”으로 설명하기에는 근거가 부족하다.** 이번 조건은 원 미래 예측 MSE로 실제 학습했음에도 강한 기준선을 이기지 못했다. 예측 목적 학습으로 바꿨다는 사실만으로 문제가 해결됐다고 볼 수 없다. 반대로 검색3epoch와 일부 checkpoint의 재추론 일치만으로 최적화 문제나 방법의 한계를 모두 판정할 수도 없다. 특히 마지막 검색점이 선택된 B0·과거 지연의 추가 학습 필요성과, 같은 E·F·D를 분리하는 원인 진단은 별도 질문이다.

## 무엇을 실제로 검산했나

ROOT는 원 학습 supervisor의 네 worker returncode0, 관측한 owned tree의 종료와 잔존 프로세스 없음, source 보존을 확인했다. 각 public fit RESULT도 전체 V 완료·frozen foundation 유지·원 source/registry/selection/T 지문 보존·CUDA 미초기화를 기록한다: [B0](diagnostics/fit_1792112761493733800_B0/RESULT.json), [B2](diagnostics/fit_1792112761493733800_B2/RESULT.json), [과거 지연](diagnostics/fit_1792112761493733800_LAG_CAUSAL/RESULT.json), [양방향 지연](diagnostics/fit_1792112761493733800_LAG_BIDIR/RESULT.json).

저장된 전체 V43을 FP64로 독립 재채점한 네 작업과, 선택 checkpoint를 실제 새 CPU forward로6 origins에 재추론한 네 작업이 모두 PASS다. 전체 V43 재채점은 새 모델 추론이 아니며, 6-origin 재추론은 전체 V43을 다시 추론한 결과가 아니다. 코드와 고정 공차는 [verify_cpu_outputs.py](verify_cpu_outputs.py)에 있다.

- B0: [전체 V43 FP64](diagnostics/root_cpu_sequential_v1_j01_lag_rescore/RESULT.json), [새 CPU6 재추론](diagnostics/root_cpu_sequential_v1_j02_lag_replay/RESULT.json).
- B2: [전체 V43 FP64](diagnostics/root_cpu_sequential_v1_j03_lag_rescore/RESULT.json), [새 CPU6 재추론](diagnostics/root_cpu_sequential_v1_j04_lag_replay/RESULT.json).
- LAG_CAUSAL: [전체 V43 FP64](diagnostics/root_cpu_sequential_v1_j05_lag_rescore/RESULT.json), [새 CPU6 재추론](diagnostics/root_cpu_sequential_v1_j06_lag_replay/RESULT.json).
- LAG_BIDIR: [전체 V43 FP64](diagnostics/root_cpu_sequential_v1_j07_lag_rescore/RESULT.json), [새 CPU6 재추론](diagnostics/root_cpu_sequential_v1_j08_lag_replay/RESULT.json).

이 PASS는 저장·집계 수치와 명시한 checkpoint 일부 재추론의 일치 증거다. 독립 일반화, 충분 수렴, 전체 누출 검사 또는 학습 목표의 우월성 검증을 대신하지 않는다. “검산 작업8개”를 독립 실험8회나 seed8개로 세지 않는다.

## CPU 비용과 그림은 아직 무엇이 남았나

이 문서 작성 시 ROOT가 CPU 정밀 비용을 계측 중이며, 같은 CPU에서 기준선 Raw·ridge·G의 새 checkpoint parity와 비용 비교는 대기 중이다. 목표 프로토콜은 CPU2 threads의 fresh process, batch1/8, warmup10·timed20·별도 stage20, row128, checkpoint inputs off다. 실제 완료 receipt와 비용 검산 전에는 지연시간·메모리·속도 배율을 추정하지 않는다. 실패·미측정 조건을0으로 채우거나 다른 경로의 시간으로 대체하지 않는다. 과거 GPU 시간과 새 CPU 시간을 직접 합쳐 속도 향상을 주장하지 않는다.

정확도·비용·정확도–비용 그림은 실제로 검산된 최종 집계 snapshot이 준비된 뒤 ROOT가 생성하고 이 문서에 연결한다. 현재 없는 그림 파일을 먼저 링크하지 않았다. 준비된 렌더러는 집계 CSV를 표시하는 도구이며, 렌더링 성공 자체가 새 예측·비용·일반화 검증을 수행한다는 뜻은 아니다.

## 적용 범위와 연구 상태

이 비교는 T 결과를 이미 관측한 뒤 별도로 등록한 **post-T 탐색적 V 전용 development**이며, 한 seed와 반복 사용한 개발 자료의 결과다. CI나 독립 holdout 일반화를 보고하지 않는다. 새 T target 구성·적합·선택·평가를 요청하지 않았으며, 이미 관측한 T 값으로 모델을 다시 선택하지 않는다. 승인된 원 safeFIT loader가 전체 archive의 T bytes까지 물리적으로 역직렬화·hash할 수 있다는 계약과 legacy 정규화 이력도 유지한다. 전체 작업의 T bytes 접근이0이었다는 주장은 하지 않는다.

[1epoch 기록](PROGRESS_E1_KO.md)과 [검색3epoch 당시 기록](PROGRESS_E3_KO.md)은 그 시점의 역사적 snapshot으로 보존한다. 이 CPU 네 조건의 완료는 원471행 중 미완료192개나 GPU3epoch 미달48개를 완료 처리한 것이 아니다. CPU 보강 조건이 검토한 과학적 범위와 원 GPU 실행 장부를 구분하며, 원 MASTER의 전체 최소 요구는 아직 미완료다. 남은 전체 범위는 [원 RESUME](../RESUME.md), 별도 비교는 [고정 PCA 계획](../postT_V_only_cpu_fixed_pca_20261016_v1/README_KO.md), [다중 K 계획](../postT_V_only_cpu_multik_20261016_v1/README_PREPARATION_KO.md)에 남아 있다.

현재 확정한 것은 네 CPU 표현 조건의 적합·전체 V 예측과 명시한 수치 검산 범위다. 강한 기준선 대비 우위, Bolt 필요성, 최종 비용·그림·전체 요구 충족·main 게시 완료는 이 보고서의 선행 주장이 아니다.
