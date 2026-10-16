# 고정 PCA 6조건: 전체 V 검산과 CPU 비용

**[확인] 고정 PCA 6조건의 전체 V 추론, 독립 FP64 재채점 6개, 새 CPU 6-origin replay 6개가 완료됐다. CPU 비용도 6조건 × batch1/8의 12개 receipt에서 timed20 원 trial을 재집계했다.** 새 6조건 중 Raw Bolt-small보다 정규화 MSE가 낮은 것은 Beijing K9 하나이고, MAE는 6조건 모두 Raw보다 높다. Electricity·PeMS08에서 K를 늘리면 오차가 줄었지만 CPU 지연시간도 늘었다. 이 결과로 PCA+Bolt의 보편적 우위나 Bolt의 필수성을 주장하지 않는다.

이 문서는 [기존 성능 보고서](RESULTS_KO.md)에 종료된 비용의 수치와 해석을 추가한 별도 스냅샷이다. 기존 보고서의 “비용 진행 중” 문장은 그 작성 당시 상태로 보존한다. 실행 조건은 [PLAN](PLAN.json), [등록 6조건](REGISTRY.csv), [고정 PCA 실행기](fixed_pca_runner.py), [검산 실행기](verify_fixed_pca_outputs.py)에 있다.

## 비교 조건과 실제 학습량

입력 길이 L512, primary fold, seed92601, frozen Bolt-small, CPU 2 threads, row microbatch128, checkpoint inputs off를 유지했다. 원 C채널의 A 정규화 공간에서 MSE·MAE를 계산했다. 자료·예측 길이가 다른 점수는 합산해 순위를 만들지 않는다.

```text
자료          C    H   전체 V origins   K          A50 끝   A60 끝
Beijing      11   48          43          9         10519    12622
Jena         21   48          43         16         10482    12579
Electricity 321   48          50         41,241     12120    14544
PeMS08      170   12         148         22,128      8928    10713
```

원 방법은 `F((X-center)E)Eᵀ+center`이다. A 구간에서 center와 PCA basis E를 적합하고, K개 잠재 시계열의 Bolt 예측을 같은 basis로 복원한다. 별도 학습 디코더나 현재값 anchor를 가진 신경망 B0/B2와 동일 구조가 아니다.

**[확인] trainable parameter·search/refit optimizer update·training exposure는 모두 0이다.** A50/A60 PCA 적합은 실제 수행했지만 gradient 학습은 없었다. 원 `[0,0.25,1,3]` 검색 일정은 같은 고정 상태를 반복 평가한 기록이며, 신경망 3epoch 완료나 독립 실험 4개로 세지 않는다. selected epoch는 모두 0이다. 공통 BASE receipt의 `optimizer="AdamW"` 문자열보다 실제 stage의 `NONE_FIXED_TRANSFORM`, updates0/exposures0 기록을 따라 해석한다. [Beijing search receipt](runs/postT_cpu_beijing_E1_PCA_FIXED_BOLT_k9_s92601_v1/search/RECEIPT.json)와 [refit receipt](runs/postT_cpu_beijing_E1_PCA_FIXED_BOLT_k9_s92601_v1/refit/RECEIPT.json)가 한 예다.

## 전체 V 원수치와 같은 평가 문제의 기준선

다음은 저장된 점수의 원수치다. Raw·ridge의 data/scaler/cohort와 target·observed·origins·channel_ids identity가 일치한다는 기존 12쌍 metadata 대조를 [기존 보고서](RESULTS_KO.md)에서 재사용했다. 이번 문서 작성에서 배열이나 기준선을 다시 추론하지 않았다. 같은 평가 문제의 예측오차 비교이며, 장치·학습 목적·파라미터·복원 구조까지 통제한 인과 대조는 아니다.

```text
자료          K      PCA 정규화 MSE       PCA 정규화 MAE
Beijing       9      0.753197402107896     0.5802058746215492
Jena         16      1.0328224546410856    0.5891090781334987
Electricity  41      0.28218578261296384   0.34181553288311894
Electricity 241      0.17175415905886102   0.27041468602564833
PeMS08       22      0.34667652090077167   0.3320666797683997
PeMS08      128      0.09887213009597785   0.20689401731789153

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

Beijing K9는 Raw보다 MSE는 낮고 MAE는 높다. Jena K16은 Raw·ridge보다 두 지표 모두 높다. Electricity 두 K는 ridge보다 두 지표가 낮지만 Raw보다 높다. PeMS08 K22는 두 기준선보다 높고 K128은 ridge보다 낮지만 Raw보다 높다. 강한 기준선에 대한 일관된 우위는 확보하지 못했다.

원 PCA 점수와 검산 근거는 아래 여섯 묶음이다. FP64는 전체 V 저장 예측·공개 충분통계의 검산이고, replay는 선택 artifact를 실제 새 CPU forward로 6 origins에만 재추론한 검사다.

- Beijing K9: [점수](predictions/postT_cpu_beijing_E1_PCA_FIXED_BOLT_k9_s92601_v1/V_select/SCORES.csv), [전체 V43 FP64](diagnostics/root_cpu_sequential_v1_j24_pca_rescore/RESULT.json), [CPU6 replay](diagnostics/root_cpu_sequential_v1_j25_pca_replay/RESULT.json).
- Jena K16: [점수](predictions/postT_cpu_jena_E1_PCA_FIXED_BOLT_k16_s92601_v1/V_select/SCORES.csv), [전체 V43 FP64](diagnostics/root_cpu_sequential_v1_j26_pca_rescore/RESULT.json), [CPU6 replay](diagnostics/root_cpu_sequential_v1_j27_pca_replay/RESULT.json).
- Electricity K41: [점수](predictions/postT_cpu_electricity_E1_PCA_FIXED_BOLT_k41_s92601_v1/V_select/SCORES.csv), [전체 V50 FP64](diagnostics/root_cpu_sequential_v1_j28_pca_rescore/RESULT.json), [CPU6 replay](diagnostics/root_cpu_sequential_v1_j29_pca_replay/RESULT.json).
- Electricity K241: [점수](predictions/postT_cpu_electricity_E1_PCA_FIXED_BOLT_k241_s92601_v1/V_select/SCORES.csv), [전체 V50 FP64](diagnostics/root_cpu_sequential_v1_j30_pca_rescore/RESULT.json), [CPU6 replay](diagnostics/root_cpu_sequential_v1_j31_pca_replay/RESULT.json).
- PeMS08 K22: [점수](predictions/postT_cpu_pems08_E1_PCA_FIXED_BOLT_k22_s92601_v1/V_select/SCORES.csv), [전체 V148 FP64](diagnostics/root_cpu_sequential_v1_j32_pca_rescore/RESULT.json), [CPU6 replay](diagnostics/root_cpu_sequential_v1_j33_pca_replay/RESULT.json).
- PeMS08 K128: [점수](predictions/postT_cpu_pems08_E1_PCA_FIXED_BOLT_k128_s92601_v1/V_select/SCORES.csv), [전체 V148 FP64](diagnostics/root_cpu_sequential_v1_j34_pca_rescore/RESULT.json), [CPU6 replay](diagnostics/root_cpu_sequential_v1_j35_pca_replay/RESULT.json).

6 FP64·6 replay 모두 PASS이고 replay는 모두 실제 forward가 새로 수행됐다. 공차는 FP64 atol1e-9/rtol1e-7, replay atol1e-5/rtol1e-4를 유지했다. 이 일치는 독립 일반화나 충분 수렴·Bolt 필요성의 증거로 확대하지 않는다.

## 종료된 CPU 비용: timed20의 전체경로 지연시간

수치 출처는 [독립 비용 산술 검산 AUDIT_RESULT](diagnostics/root_cpu_cost_arithmetic_20261016_v1/AUDIT_RESULT.json)의 **12 records 중 `recomputed_summaries.phase="timed"`만**이다. 12개 모두 원 receipt 상태는 `COMPLETE_PROTOCOL`, 각 phase는 warmup10·timed20·별도 stage_probe20이다. 아래 median/p90는 timed20 분포의 초 단위 원값이다. Batch8 값은 8개 입력을 처리한 한 batch의 전체 시간이며 입력 1개당 시간으로 바꾸지 않았다. stage_probe의 시간을 섞거나 stage 합을 전체 시간이라고 주장하지 않는다.

```text
자료          K    B    timed trials   median seconds        p90 seconds
Beijing       9    1          20       0.06581165012903512    0.06827926975674928
Beijing       9    8          20       0.4773309000302106     0.48480951027013364
Jena         16    1          20       0.11176185007207096    0.11776564978063106
Jena         16    8          20       0.8658131002448499     0.8870924399234356
Electricity  41    1          20       0.28759984974749386    0.29284457988105717
Electricity  41    8          20       2.241482650162652      2.333791770134121
Electricity 241    1          20       1.6696278499439359     1.7351426399312913
Electricity 241    8          20      13.329069199971855     13.607346489652992
PeMS08       22    1          20       0.1489165499806404     0.15686008990742267
PeMS08       22    8          20       1.2088229497894645     1.2684966696891933
PeMS08      128    1          20       0.8905439001973718     0.9223303098231554
PeMS08      128    8          20       7.093275299761444      7.264548809872941
```

원 비용 영수증은 자료·K별로 다음과 같다. 산술 검산은 각 receipt/trial의 SHA와 artifact·V manifest·data/scaler binding을 확인했다. 이 문서는 trial gzip을 다시 열거나 비용 worker를 실행하지 않았다.

- Beijing K9: [B1](costs/1792124498526258000_973ea04c82c8448c8737ae31ed718733/PCA_FIXED_BOLT_k9_B1.json), [B8](costs/1792124498526258000_973ea04c82c8448c8737ae31ed718733/PCA_FIXED_BOLT_k9_B8.json).
- Jena K16: [B1](costs/1792124556976712600_2455638deefe4b0c875b89b2f2b98a10/PCA_FIXED_BOLT_k16_B1.json), [B8](costs/1792124556976712600_2455638deefe4b0c875b89b2f2b98a10/PCA_FIXED_BOLT_k16_B8.json).
- Electricity K41: [B1](costs/1792124638173262700_ef0d9bb925a34bdf86fcbcfd58e40045/PCA_FIXED_BOLT_k41_B1.json), [B8](costs/1792124638173262700_ef0d9bb925a34bdf86fcbcfd58e40045/PCA_FIXED_BOLT_k41_B8.json).
- Electricity K241: [B1](costs/1792124800239271300_9b270d98e78642e3a92f1707ee34e398/PCA_FIXED_BOLT_k241_B1.json), [B8](costs/1792124800239271300_9b270d98e78642e3a92f1707ee34e398/PCA_FIXED_BOLT_k241_B8.json).
- PeMS08 K22: [B1](costs/1792125591748183400_63828d4d6047452eae2e84cf36e702a5/PCA_FIXED_BOLT_k22_B1.json), [B8](costs/1792125591748183400_63828d4d6047452eae2e84cf36e702a5/PCA_FIXED_BOLT_k22_B8.json).
- PeMS08 K128: [B1](costs/1792125694161941400_10a72a3eea124e0cbefb2ed477246be9/PCA_FIXED_BOLT_k128_B1.json), [B8](costs/1792125694161941400_10a72a3eea124e0cbefb2ed477246be9/PCA_FIXED_BOLT_k128_B8.json).

**검산 범위의 구분:** `PASS_COST_NUMBERS_AND_METADATA_ONLY`는 저장 trial의 산술·선언 identity 검산이다. audit의 `resource_guard_verified=false`, `quiet_window_verified=false`, `fresh_CPU_prediction_parity_verified=false`를 PASS로 바꾸어 읽지 않는다. CPU 자원·동시 작업 조건과 실제 forward의 근거는 별도 supervisor/closure와 위 replay에 속한다. 이 감사 단독으로 동시 실행 부재·메모리 상한·재현 정확도를 새로 증명한 것은 아니다.

## K와 예측·계산의 관계

Electricity K41→241에서는 MSE가 0.28218578261296384→0.17175415905886102, MAE가 0.34181553288311894→0.27041468602564833으로 낮아졌다. 같은 때 CPU median은 B1에서 0.28759984974749386→1.6696278499439359초, B8에서 2.241482650162652→13.329069199971855초로 늘었다. PeMS08 K22→128에서도 MSE·MAE 감소와 두 batch의 시간 증가를 함께 관찰했다. 유지한 정보와 F stream이 함께 늘어난 두 점 비교이며, 압축률만의 인과 효과나 모든 K에서의 단조 개선을 뜻하지 않는다.

같은 CPU에서 측정한 PCA 조건끼리의 기록은 비교할 수 있지만 Raw·ridge와의 속도 배율은 여기서 계산하지 않는다. 4자료 모두의 같은 CPU·같은 입력·같은 batch protocol 기준선 비용을 이 감사만으로 갖춘 것은 아니다. 이전 GPU 비용과 새 CPU 비용을 섞지 않고, F stream 감소를 그대로 속도 향상으로 바꾸지 않는다. PCA 적합·모델 로드·학습시간이나 메모리 우위도 이 timed 표의 결론에 포함하지 않는다.

## 남은 판단과 연구 전체 상태

고정 PCA는 동시점 채널 공분산을 사용하는 기준선이며 시간 지연 그래프를 학습한 결과가 아니다. gradient update0이므로 이번 악화를 “PCA optimizer 학습 부족”으로 설명할 근거는 없다. 그러나 버린 정보, 잠재 시계열에 대한 Bolt 적합성, 고정 복원의 제약 중 무엇이 지배적인지는 이 6조건만으로 확정하지 못했다. 같은 E/D에서 native 예측기와 Bolt를 비교하는 등 단계를 나눈 비교가 Bolt를 남길 근거의 판단에 필요하다.

이 결과는 T 관측 후 별도 등록한 V-only 개발, seed92601 하나, 재사용한 개발 구간에 한정된다. 새 TEST target·선택·평가를 수행하지 않았다는 실행 receipt를 유지한다. 승인된 원 safeFIT loader가 archive의 T bytes를 물리적으로 역직렬화할 수 있는 계약과 legacy 정규화 이력을 유지하며, 세션 전체의 T byte 접근0을 주장하지 않는다. 이번 보고서 작성에서는 작은 JSON/CSV와 기존 문서만 읽었고 Task·모델·배열·통계 gzip·trial gzip·학습·추론·새 측정을 실행하지 않았다.

원 source39와 원 등록·선택·T의 고정을 유지한다. MASTER 전체는 PARTIAL이며 M5의 자료 부족, 원 coregap37이나 다른 미완료를 이 고정 PCA0update 6조건으로 채우지 않는다. 별도 15 fresh fit·22 V 출력을 목표로 하는 보강 큐는 이 문서의 완료 분모에 포함하지 않으며 전부 완료됐다고 주장하지 않는다. 원 연구의 재개 조건은 [RESUME](../RESUME.md)에 있다. 후속으로 아래 PCA6 전용 그림을 실제 생성하고 PNG를 열어 수치·범례·서로 다른 H/V 표시의 가독성을 확인했다. 원 집계6행만 렌더링했으며 새 모델 추론이나 비용 측정을 수행한 것은 아니다.


## 고정 PCA6의 실제 그림과 무손실 원수치

![같은 자료의 Raw 대비 PCA 오차와 PCA CPU 비용](diagnostics/final_pca_cpu_summary_root_v1/rendered_root_v1/PCA_ACCURACY_CPU_COST.svg)

[그림 입력6행](diagnostics/final_pca_cpu_summary_root_v1/AGGREGATE.csv), [실제 source·artifact 결박](diagnostics/final_pca_cpu_summary_root_v1/SOURCE_BINDINGS.json), [렌더러와 출력 SHA](diagnostics/final_pca_cpu_summary_root_v1/rendered_root_v1/FIGURE_BINDING.json)를 함께 제공한다. Raw 네 기준선을 각 K에 다시 표시했으므로 독립 기준선6개가 아니다. 자료 간 평균이나 Raw 대비 CPU 속도 배율은 만들지 않았다. PNG는300dpi 로컬 출력으로 보존하고 이 문서에는 SVG를 연결한다.

원 충분통계251,388행은 행·순서·float 정밀도·CSV 개행과 원 gzip byte를 바꾸지 않는 별도 numeric 포장으로 전수 복원했다. [독립 목적지의 원 gzip6파일 SHA 대사](diagnostics/numeric_pack_independent_restore_root_v1/RESULT.json)는 별도 CLI 실행과 원본 대사를 기록한다. 포장 원본과 local 결과는 삭제하거나 덮어쓰지 않았다. 최종 main 게시 범위·포장 파일과 복원 명령은 캠페인의 게시·복원 색인에서 확인해야 하며, 이 로컬 복원 PASS 자체가 main 게시 완료를 뜻하지 않는다.
