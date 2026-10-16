# 고정 PCA 6조건: 전체 V 결과와 남은 판단

**[확인] 별도 등록한 CPU 고정 PCA 6조건에서 PCA 적합·전체 V 추론, 독립 FP64 재채점 6개, 선택된 고정 변환의 새 CPU 6-origin 재추론 6개를 완료했다.** Electricity와 PeMS08의 두 K 비교에서는 큰 K의 MSE·MAE가 낮았다. 그러나 아래 6조건 중 Raw Bolt-small보다 MSE가 낮은 경우는 Beijing K9 한 조건이고, MAE가 낮은 경우는 없다. 고정 PCA의 보편적 이득이나 frozen Bolt의 필요성을 확보한 결과는 아니다.

이 문서는 **post-T V 전용 개발의 부분 결과**다. 작성 시 PCA의 CPU 비용은 별도 큐에서 계측 중이고 Raw·ridge·G 기준선의 같은 CPU 비용 비교도 종료를 확인하지 않았다. 비용·속도·메모리 우위, 새 그림, 이번 산출물의 게시 완료 및 원 연구 전체 완료를 주장하지 않는다. 실제 실행 계약은 [PLAN](PLAN.json), [별도 등록 6조건](REGISTRY.csv), [실행기](fixed_pca_runner.py)에 있다.

## 무엇을 실행했나

원 `PCA_FIXED_BOLT`는 A 구간의 정규화된 과거 창을 A-fitted PCA의 center와 E로 투영하고, 고정 Bolt-small로 K개 잠재 시계열을 예측한 뒤 같은 basis의 전치로 원 C채널에 복원한다. 실제 식은 `F((X-center)E)Eᵀ+center`이며 [원 고정 변환](../fit_base.py#L55)을 그대로 사용한다. 별도 학습 디코더나 현재값 anchor를 가진 B0/B2와 같은 구조라고 부르지 않는다.

모든 조건은 입력 L512, primary fold, seed92601, CPU 2 threads, frozen Bolt-small, row microbatch128, checkpoint inputs off를 유지한다. 자료별 H·C·V origin 수와 K는 다음과 같다. 정규화 MSE·MAE는 원 C채널에서 계산하며 채널을 균등하게 보는 주 MSE 규칙을 유지했다.

```text
자료          C    H   V origins   실행 K       A50 끝   A60 끝
Beijing      11   48      43          9          10519    12622
Jena         21   48      43         16          10482    12579
Electricity 321   48      50         41,241      12120    14544
PeMS08      170   12     148         22,128       8928    10713
```

원 데이터·분할·열·source39·정규화 계약을 유지했고 별도 public/private namespace의 새 ID로 실행했다. 전체 V의 실제 F stream은 각각 `V origins × K`인 387, 688, 2050, 12050, 3256, 18944다. 이 값은 F에 전달한 논리적 시계열 수이며 전체 지연시간·메모리 또는 속도 배율이 아니다. V receipt의 `F_calls=1`은 bundle bookkeeping이며 실제 row-call 횟수로 해석하지 않는다.

## PCA 적합과 신경망 학습을 구분한다

**[확인] 6조건의 center/E는 buffer이고 trainable parameter, search/refit optimizer update, training exposure가 모두 0이다.** A50과 A60의 데이터로 정규화·PCA 통계를 적합한 작업은 실제 수행했다. 그러나 gradient로 신경망 가중치를 갱신하지 않았다. 원 API의 `[0,0.25,1,3]` search 표시는 같은 고정 상태를 A_stop에서 반복 평가한 일정이다. 4개 독립 학습이나 신경망 3epoch 완료로 세지 않으며, 모든 조건의 selected epoch는 0이고 A60의 고정 변환을 새로 구성했다.

원 BASE receipt의 `optimizer="AdamW"`는 공통 기록 코드의 고정 문자열이다. 실제 search/refit stage의 `optimizer="NONE_FIXED_TRANSFORM"`, `actual_epochs=0`, updates/exposures0을 우선해 해석한다. 같은 상태의 반복 평가를 독립 seed·표본·방법으로 부풀리지 않는다. 예를 들어 [Beijing BASE](runs/postT_cpu_beijing_E1_PCA_FIXED_BOLT_k9_s92601_v1/BASE_FIT_RECEIPT.json), [search](runs/postT_cpu_beijing_E1_PCA_FIXED_BOLT_k9_s92601_v1/search/RECEIPT.json), [refit](runs/postT_cpu_beijing_E1_PCA_FIXED_BOLT_k9_s92601_v1/refit/RECEIPT.json)에 이 구분이 남아 있다.

## 전체 V 원수치

다음은 저장된 `SCORES.csv`와 manifest에 기록된 반올림 전 값이다. 서로 다른 자료·H·정규화의 값을 합쳐 평균 성능으로 비교하지 않는다.

```text
자료          K      정규화 MSE          정규화 MAE
Beijing       9      0.753197402107896     0.5802058746215492
Jena         16      1.0328224546410856    0.5891090781334987
Electricity  41      0.28218578261296384   0.34181553288311894
Electricity 241      0.17175415905886102   0.27041468602564833
PeMS08       22      0.34667652090077167   0.3320666797683997
PeMS08      128      0.09887213009597785   0.20689401731789153
```

점수와 실제 full V 생성 영수증은 다음과 같다.

- Beijing K9: [점수](predictions/postT_cpu_beijing_E1_PCA_FIXED_BOLT_k9_s92601_v1/V_select/SCORES.csv), [실행](diagnostics/root_cpu_sequential_v1_j18_pca_fit/RESULT.json).
- Jena K16: [점수](predictions/postT_cpu_jena_E1_PCA_FIXED_BOLT_k16_s92601_v1/V_select/SCORES.csv), [실행](diagnostics/root_cpu_sequential_v1_j19_pca_fit/RESULT.json).
- Electricity K41: [점수](predictions/postT_cpu_electricity_E1_PCA_FIXED_BOLT_k41_s92601_v1/V_select/SCORES.csv), [실행](diagnostics/root_cpu_sequential_v1_j20_pca_fit/RESULT.json).
- Electricity K241: [점수](predictions/postT_cpu_electricity_E1_PCA_FIXED_BOLT_k241_s92601_v1/V_select/SCORES.csv), [실행](diagnostics/root_cpu_sequential_v1_j21_pca_fit/RESULT.json).
- PeMS08 K22: [점수](predictions/postT_cpu_pems08_E1_PCA_FIXED_BOLT_k22_s92601_v1/V_select/SCORES.csv), [실행](diagnostics/root_cpu_sequential_v1_j22_pca_fit/RESULT.json).
- PeMS08 K128: [점수](predictions/postT_cpu_pems08_E1_PCA_FIXED_BOLT_k128_s92601_v1/V_select/SCORES.csv), [실행](diagnostics/root_cpu_sequential_v1_j23_pca_fit/RESULT.json).

## 같은 평가 조건의 Raw·ridge와 비교

**[확인] 각 자료의 기존 Raw Bolt-small·native ridge와 PCA의 data/scaler SHA, V cohort, target/observed/origins/channel_ids의 shape·dtype·SHA가 일치한다.** 6 PCA × 2 기준선의 12쌍을 metadata로 대조했다. 공통 예측 소스 7개의 SHA도 같고 ridge에는 원 `ridge_baseline.py`가 추가된다. 전체 prediction source/model/config SHA는 방법·학습 출처가 달라 서로 같지 않으며, 이를 같은 모델이라고 주장하지 않는다. 이번 문서 작성자는 배열을 다시 열거나 기준선을 새로 추론하지 않았다.

따라서 아래 저장 V의 오차를 같은 평가 문제에서 비교할 수 있다. 장치·파라미터 수·적합 목표·디코더·계산량까지 맞춘 인과 대조 또는 비용 비교는 아니다.

```text
자료          기준선             정규화 MSE          정규화 MAE
Beijing       Raw Bolt-small     0.7968786536971474    0.5618622043428347
Beijing       Native ridge       0.8315416367816169    0.6073577987792294
Jena          Raw Bolt-small     0.8491941389472357    0.4813883402485757
Jena          Native ridge       0.5566463357048567    0.40552180175406594
Electricity   Raw Bolt-small     0.12662080012059046   0.2027005720721219
Electricity   Native ridge       0.3193732185113921    0.40482331800492677
PeMS08        Raw Bolt-small     0.0953549334407595    0.19633382925361176
PeMS08        Native ridge       0.22899594314673508   0.2325320321186678
```

원 기준선 점수: Beijing [Raw](../predictions/primary_beijing_E1_RAW_BOLT_SMALL_k0_s92601/V_select/SCORES.csv)·[ridge](../predictions/primary_beijing_E1_RAW_RIDGE_k0_s92601/V_select/SCORES.csv), Jena [Raw](../predictions/primary_jena_E1_RAW_BOLT_SMALL_k0_s92601/V_select/SCORES.csv)·[ridge](../predictions/primary_jena_E1_RAW_RIDGE_k0_s92601/V_select/SCORES.csv), Electricity [Raw](../predictions/primary_electricity_E1_RAW_BOLT_SMALL_k0_s92601/V_select/SCORES.csv)·[ridge](../predictions/primary_electricity_E1_RAW_RIDGE_k0_s92601/V_select/SCORES.csv), PeMS08 [Raw](../predictions/primary_pems08_E1_RAW_BOLT_SMALL_k0_s92601/V_select/SCORES.csv)·[ridge](../predictions/primary_pems08_E1_RAW_RIDGE_k0_s92601/V_select/SCORES.csv).

Beijing K9는 두 기준선보다 MSE가 낮지만 MAE는 Raw보다 높다. Jena K16은 두 기준선보다 MSE·MAE 모두 높다. Electricity 두 K는 ridge보다 두 지표가 낮지만 Raw보다 높다. PeMS08 K22는 두 기준선보다 나쁘고 K128은 ridge보다 낫지만 Raw보다 두 지표가 높다. 따라서 PCA+Bolt가 원 C 입력이나 단순 예측기를 일관되게 대체한다는 결론은 허용되지 않는다.

## K가 커지면 무엇이 달라졌나

Electricity K41→241, PeMS08 K22→128에서 두 지표 모두 낮아졌다. 같은 자료·scaler·F·고정 투영 규칙에서 더 많은 주성분을 유지한 두 점의 관찰이다. 다른 자료·모든 K에서 단조 개선된다는 증거나 학습형 표현의 한계 판정은 아니다. Beijing·Jena는 이 새 묶음에 K가 한 개씩이므로 이 묶음만으로 K 추세를 그리지 않는다.

원 C의 Raw와 K의 PCA는 압축 차원뿐 아니라 투영·역투영과 F가 받는 신호의 좌표계도 다르다. 큰 K는 F stream과 복원 비용도 늘리므로 오차 감소만으로 효율이 낫다고 말할 수 없다. 고정 PCA는 동시점 채널 공분산을 활용하는 기준선이며 시간 지연 그래프를 학습한 결과가 아니다. 이 6조건은 gradient 학습이 없어서 짧은 optimizer 학습량 때문에 PCA 가중치가 덜 학습됐다고 설명할 수도 없다. 남은 문제는 버린 정보, 잠재 시계열의 F 적합성, 고정 복원의 제약을 분리하는 것이다.

## 실제 검산 범위

각 full V 생성 RESULT는 frozen foundation 불변, 원 source/registry/selection/T 지문 보존, CUDA 미초기화를 기록한다. 독립 FP64 재채점은 전체 V와 공개 충분통계를 확인했고, replay는 선택 artifact를 새 CPU forward로 6 origins에만 재추론했다. 전자는 새 모델 추론이 아니고 후자는 full V 전체의 재추론이 아니다.

- Beijing K9: [전체 V43 FP64](diagnostics/root_cpu_sequential_v1_j24_pca_rescore/RESULT.json), [새 CPU6 replay](diagnostics/root_cpu_sequential_v1_j25_pca_replay/RESULT.json).
- Jena K16: [전체 V43 FP64](diagnostics/root_cpu_sequential_v1_j26_pca_rescore/RESULT.json), [새 CPU6 replay](diagnostics/root_cpu_sequential_v1_j27_pca_replay/RESULT.json).
- Electricity K41: [전체 V50 FP64](diagnostics/root_cpu_sequential_v1_j28_pca_rescore/RESULT.json), [새 CPU6 replay](diagnostics/root_cpu_sequential_v1_j29_pca_replay/RESULT.json).
- Electricity K241: [전체 V50 FP64](diagnostics/root_cpu_sequential_v1_j30_pca_rescore/RESULT.json), [새 CPU6 replay](diagnostics/root_cpu_sequential_v1_j31_pca_replay/RESULT.json).
- PeMS08 K22: [전체 V148 FP64](diagnostics/root_cpu_sequential_v1_j32_pca_rescore/RESULT.json), [새 CPU6 replay](diagnostics/root_cpu_sequential_v1_j33_pca_replay/RESULT.json).
- PeMS08 K128: [전체 V148 FP64](diagnostics/root_cpu_sequential_v1_j34_pca_rescore/RESULT.json), [새 CPU6 replay](diagnostics/root_cpu_sequential_v1_j35_pca_replay/RESULT.json).

6 rescore와 6 replay는 모두 PASS이며 replay는 모두 `executed_now=true`다. FP64 재집계의 공차는 atol1e-9/rtol1e-7, 실제 replay는 원 atol1e-5/rtol1e-4를 유지했다. 예를 들어 Beijing MSE는 저장값0.753197402107896과 FP64 재집계값0.7531974021078962처럼 마지막 자리의 부동소수점 차이가 있으며 공차 안의 일치이지 byte 동일 수치라는 뜻은 아니다. 6 replay의 최대 절대 차이는 각각 8.344650268554688e-7, 1.430511474609375e-6, 9.5367431640625e-7, 4.76837158203125e-7, 1.0132789611816406e-6, 0이다. 검산 코드는 [verify_fixed_pca_outputs.py](verify_fixed_pca_outputs.py)에 있다.

이 PASS는 저장 수치·집계와 명시된 일부 checkpoint 재추론의 일치다. 충분 수렴·독립 일반화·모든 누출 검사·Bolt 필요성·비용 우위를 증명하지 않는다. 12개 검산 작업이나 고정 상태의 반복 A_stop 평가를 독립 실험·seed 수로 세지 않는다.

## 비용과 연구 전체 상태

CPU 비용은 작성 시 별도 실행 중이다. 목표는 CPU 2 threads의 fresh single worker, batch1/8, warmup10·timed20·별도 stage20, row128, checkpoint inputs off다. 완료 receipt·quiet-window·독립 비용 검산 전에는 속도나 메모리 숫자를 이 문서에 넣지 않는다. 과거 GPU 비용과 새 CPU 비용을 합쳐 속도 향상을 계산하지 않으며, 실패·미측정 조건을 0으로 채우지 않는다. 비용 준비 코드는 [prepare_fixed_pca_costs.py](prepare_fixed_pca_costs.py), 원 캐시 접근의 공개 재현·검증은 [prepare_cost_assets.py](prepare_cost_assets.py)와 [원 bytes 검증 영수증](diagnostics/cost_cache_assets_verified_root_v1/COST_CACHE_ASSET_ACCESS.json)에 있다. 이 접근 검증은 NPZ의 opaque byte copy/hash 범위이며 새 배열 검산이나 비용 실측이 아니다.

그림은 최종 검산된 집계와 비용이 준비된 뒤 연결한다. 이 문서에는 아직 없는 그림 또는 비용 파일의 링크를 만들지 않았다. 원 PCA 실행 계획의 당시 대기 표현은 [README](README_KO.md)에 역사적 준비 상태로 보존하며 현재 실행 여부는 위 실제 RESULT를 따른다.

이 결과는 이미 T를 관측한 뒤 따로 등록한 V-only 개발이고 seed92601 하나와 재사용한 개발 구간에 한정된다. 새 T target·적합·선택·평가를 요청하지 않았다. 승인된 원 safeFIT loader가 전체 archive의 T bytes까지 물리적으로 역직렬화할 수 있는 계약과 legacy 정규화 이력을 유지하므로 세션의 T byte 접근이 0이었다고 말하지 않는다. 이번 보고서 작성 과정은 source와 작은 metadata·점수 텍스트만 읽었고 배열·Task·모델·비용 trial을 열거나 실행하지 않았다.

원 neural56/GPU 미달48·전체471행의 미완료 장부를 이 고정 PCA의 0update 평가로 완료 처리하지 않는다. 원 연구의 전체 K·장기 학습·추가 seed·시간 전이·비용 요구와 충분한 기여 검증은 [원 RESUME](../RESUME.md)에 남아 있다. 고정 PCA 6조건의 실제 V·검산 완료와 원 연구 전체 PARTIAL 상태, 같은 CPU 비용의 미확정 상태를 분리한다.
