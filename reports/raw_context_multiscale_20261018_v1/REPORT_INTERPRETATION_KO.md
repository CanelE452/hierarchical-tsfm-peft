# 실제 결과와 해석 범위

[확인] 이번 원본·문맥 분리 가설은 지지되지 않았다. 다중 해상도는 Jena에서 평균 오차와 메모리를 줄였지만 seed별 MSE 방향이 달랐고 native 지연은 늘었다. 출력 결합에는 개선이 있었으나 단순 alpha 결합도 비슷하거나 더 좋은 결과를 냈다. V는 반복 사용된 개발 평가이며 독립 TEST 결과가 아니다.

## 실행한 범위

Jena는 C=21, K=12, L=512, H=48, V 원점43개이고 PeMS32는 C=32, K=8, L=512, H=12, V 원점148개다. 원채널·원격자의 observed equal-channel macro MSE를 주지표, MAE를 보조지표로 삼았다. 원점/target/bool mask/channel order/A60 scaler 지문을 검산했고, 겹치는 창·채널·두 seed를 독립 표본으로 간주한 신뢰구간이나 p-value를 만들지 않았다.

Jena A/B 각 두 seed, A_stop으로 고른 A의 PeMS 두 seed를 실행했다. 새 body 설정은6개이고, 새 검색 단계는10개(Jena4, PeMS 초기A2, PeMS S/A 전체 cap12 연장4)다. 검색 seed 경로는8개, fresh A60 refit은8개이며 Jena S2개는 기존 완료 결과를 재사용했다. PeMS S2개를 새로 연장·refit한 작업은 새 후보6개와 별도로 셌다. A=B(s1) 대조를 위해 모델을 중복 학습하지 않았다.

고정 baseline6개, body10개, alpha32개, conditional/past_control8개로 V 경로는56개다. alpha32 중24개는 고정 endpoint이고8개가 적합 계수를 사용한다. head는8개 검색·8개 전체 M_cal fresh fit이다. 120개 전체 경로 비용은 56×B1/B8의112개와 A4개 모델의 추가 two_call×B1/B8 8개로 구성된다. 별도 head component16개는 기존8개 head의 B1/B8 고정 추론이며 새 학습 모델이 아니다.

## 가설1: 같은 K의 원본·문맥 분리

아래는 두 seed 산술평균이다. 모든 개별 seed는 [final/RUNS.csv](final/RUNS.csv)에 보존했다.

```text
data       mode   MSE          MAE
Jena       S      0.635765596  0.482046632
Jena       A      0.748531494  0.521027352
Jena       B      0.715393523  0.509893313
PeMS32     S      0.091641548  0.201327007
PeMS32     A      0.096134593  0.206979245
```

S→A에서 Jena 평균 MSE는17.74%, PeMS는4.90% 증가했다. 두 데이터의 두 seed 모두 MSE와 MAE가 나빠졌다. Jena B8 allocated는 S294.206→A294.394MiB이고 지연은20.750→20.640ms로 비슷했다. 이 조건에서 분리 자체의 이득을 확보하지 못했다.

이 대조는 P/E/D 용량을 늘리는 실험이 아니다. S는 전체 K를 문맥 P에 보내고 A는 앞 K/2 PCA 투영을 P에서 우회시킨다. 뒤 K/2만 문맥 P를 통과한다. 같은 PCA center/basis, seed의 P 초기화와 전체 E의 pseudo-inverse 기반 lag0 D를 사용한다. 실제 zero-update CP0에서 S와 A의 일치를 검산했다. E도 학습하므로 원신호를 무손실 보존했다고 해석할 수 없으며, 앞 절반 PCA 축을 raw로 배정한 한 설계의 결과다.

## 가설2: 문맥의 낮은 시간 해상도를 미래까지 유지

Jena A→B의 평균 MSE는4.43%, MAE는 약2.14% 감소했다. MSE는 seed92601에서0.50% 증가했고92602에서8.96% 감소해 일관된 개선은 아니다. B는 S보다 평균 MSE가12.52% 높다. B8 allocated는 A294.394→B243.153MiB로17.41% 줄었지만 native 지연은20.640→32.023ms로55.15% 늘었다. 현재 결과는 메모리와 지연·정확도의 교환관계다.

B는 P를 길이512에서 계산한 뒤 문맥 잠재의 과거를 두 칸 평균해 실제256 길이로 F에 보낸다. coarse 미래를 반복 복원하고 원격자 lag=(0,1,2,4,8,16,32)와 원채널 anchor를 한 번 적용한다. H48/H12의 원래 개별 target을 유지했으며 더 긴 lookback이나 평균 target을 예측한 실험이 아니다. 반복 복원은 고해상도 세부를 학습해 복구하는 연산이 아니다.

S/A native는 잠재를 묶은 한 F 호출이고 B는 공유된 frozen F 한 인스턴스에 실제512/256을 순차 호출한다. 같은 K가 같은 FLOPs·시간·메모리를 뜻하지 않는다. F 고정 head에서 앞 H/2를 읽는다고 head 연산이 절반이 되지는 않는다. A의 같은 가중치 two_call 비용도 별도로 측정했으며 [matched 그림](final/F3_MATCHED_TWO_CALL.png)에 native와 함께 표시했다.

## 선택과 학습 상한

전이는 V 전에 봉인했다. Jena의 두 seed A_stop 평균은 A0.3841795154, B0.4009291987이어서 A를 선택했다. 나중에 V에서 B의 평균이 낮게 나왔어도 PeMS 후보를 바꾸지 않았다. PeMS B는 등록 범위에 없어 실행하지 않았다.

Jena S/A/B는 cap3의 같은 검색 기회를 받았다. 연장 규칙이 발동하지 않았다는 사실은 수렴 증명이 아니다. PeMS에서는 S/A 각 두 seed를 cap6→12로 함께 연장했다. 선택 epoch와 fresh refit의 실제 학습량까지 같다는 뜻은 아니며 실제 updates·exposures·CP0·segment loss는 T2와 훈련표에 구분했다. 마지막 cap가 strict best인 경로는 `BUDGET_LIMITED_CONVERGENCE_UNRESOLVED`다. 빈 resume queue는 입장한 작업의 완료를 뜻하며 수학적 수렴을 뜻하지 않는다.

## 출력 결합과 F의 가치

부모는 원래의 Jena S cap3/PeMS S cap6이다. 최종 body의 PeMS S cap12와 다르다. [부모 식별표](PARENT_IDENTITIES.csv)는 고정 역할8개와 최종 S4개를 구분하고 artifact SHA·epoch·cap·적합 범위를 기록한다. T3 비용의 alpha1 endpoint도 원래 S의 SHA로 역추적했다. G는 데이터마다 seed92601 한 artifact를 두 head seed가 공유하며 G 두 번의 독립 학습으로 세지 않았다.

```text
data       G MSE        past_control   conditional    G+S alpha0.5  G+S fitted alpha
Jena       0.632217050  0.624665559    0.596064848    0.568939820   0.570498558
PeMS32     0.081465373  0.079800711    0.076385994    0.076829897   0.076239638
```

Conditional은 두 데이터의 두 seed 모두 past_control보다 MSE가 낮았고 평균 감소는 Jena4.58%, PeMS4.28%다. 하지만 Jena는 단순 G+S alpha0.5가 conditional보다 좋고, PeMS fitted alpha도 conditional과 비슷하며 조금 낮다. 복잡한 출력부가 단순 결합보다 일관되게 우수하다는 근거는 없다. Jena의 LoRA0.563922949와 CPU Ridge0.556646336도 함께 보고했다.

추가 S 조건의 이득에는 비용이 따른다. B8 conditional 전체 경로는 Jena21.110ms/PeMS18.620ms이고 past_control은0.733/0.737ms다. 소형 head 자체의 GPU-resident B8 forward 중앙값은 conditional0.13345–0.13530ms, past_control0.13960–0.14275ms다. 이 component 진단은 feature 구성을 포함하지만 H2D·부모 계산·cold load·CPU 반환은 제외한다. 전체 경로 시간에 더하거나 빼서 분해한 값이 아니며 전체 경로 지연을 대체하지 않는다. [16개 측정과480개 trial](HEAD_COMPONENT_COST_VALIDATION.json)을 별도로 남겼다.

G는 S/A/B를 이 측정의 MSE·whole-path 지연·allocated에서 지배했다. 결합에서 확인된 것은 이 고정 S 예측의 추가 조건 정보 가치다. 같은 표현의 no-Bolt 대조는 실행하지 않았으므로 F 자체의 필수성은 확인되지 않았다. G/LoRA/Ridge는 원래 recipe의 실용·역사 기준선이며 S/A/B와 같은 총 학습예산의 인과적 대조로 해석하지 않는다.

## M_cal 적합오차와 평가 범위

Head는 target이 겹치지 않는 M_fit/M_select에서 epoch를 고르고 전체 M_cal에서 fresh refit했다. 최종 모델40개의 full-M_cal in-sample MSE/MAE를 별도 CPU 진단으로 재계산하고 FP32 저장값에서 FP64 fsum/NumPy/reload 통계를 대조했다. [원40행](MCAL_FINAL_FIT_DIAGNOSTICS.csv)과 [평균20행](MCAL_FINAL_FIT_MEANS.csv)을 구분한다. M_fit/M_select는 선택 당시 모델의 복사된 진단이며 최종 full-M_cal 모델을 다시 평가한 값이 아니다. Batch 평균 loss도 full-M_cal MSE가 아니다.

예를 들어 conditional 평균 MSE는 Jena M_fit0.39456/M_select0.71134, 최종 M_cal0.47531, V0.59606이다. PeMS는 각각0.07567/0.05939/0.06892/0.07639다. 서로 다른 구간·모델 상태의 숫자이므로 같은 오차로 섞지 않았다. 허용 pre-T prefix loader가 V 숫자 prefix까지 decode할 수 있다는 사실과 이 진단이 M_cal에서만 계산하고 새 V 예측·선택을 하지 않았다는 사실도 구분했다.

새 보호 T 수치 연산은0이지만 과거 T 노출 이력, 상류 cache 전처리·정규화와 사전학습 중복의 불확실성은 남는다. 현재 A60 scaler와 V 전 선택 봉인이 이 이력을 지우지는 않는다. 두 가설의 결론은 이 데이터·PCA 축 배분·두 seed·등록 예산에 한정한다.

## 검증·복구·그림·재현

56개 V와120개 전체 비용을 생산한 main002는 마지막 검산에서 Direct의 `NO_FOUNDATION`을 잘못 거부해 실패했다. 첫 CPU 복구는 이를 처리한 뒤 과거 LoRA config 형식에서 실패했다. 두 실패 기록을 보존하고 두 번째 CPU 복구에서 LoRA와 PeMS Direct의 원래 저장 공식으로 config SHA를 정확히 검증했다. 일반 모델의 설정·상태·scaler·source 검사와 모든 점수/공차/모델/선택은 유지했다. 실패를 성공으로 바꾸지 않았으며 별도 실제 exit0/COMPLETE와 PASS 파일을 만들었다.

[복구 PASS](RECOVERY_VALIDATION_REPORT_002.json), [독립 V56/CP010/비용120 검산](INDEPENDENT_FINAL_AUDIT_001.json), [공개 재생성 바이트 대조](REPLAY_VALIDATION_RECOVERY_002.json), M_cal40·평균20·head16의 검산을 각각 보존했다. checkpoint/optimizer/sampler/RNG와 PeMS S migration의 상태 보존 검산도 별도로 남긴다. 이는 새 업데이트를 실행한 동등성 실험과 구분한다.

최종14개 PNG/SVG를 실제로 검토했다. 원본 CPU 그림2개는 제목이 잘려 원본 바이트를 보존하고 같은 공개 좌표의 [B1](readable_cpu_figures/F3_NATIVE_ALL_jena_cpu_B1_READABLE.png)/[B8](readable_cpu_figures/F3_NATIVE_ALL_jena_cpu_B8_READABLE.png) 가로 버전을 추가했다. 나머지12개는 [final 그림 manifest](final/FIGURE_MANIFEST.json)를 따른다. 읽기 쉬운 버전은 원본14개에 새 실험점을 추가한 것이 아니다.

실제 원본 시작·실패·대기·CP0·M_cal·head·검산·보고·게시 시간은 별도 최종 자원 원장을 따른다. GPU 소유 process wall 상계는 base8167.840초+head36.512초=8204.352초(약2시간16분44초)다. kernel busy 시간은 측정하지 않았다. 부모/전체 경로와 component 시간을 중복 합산하지 않는다. 원래 wall21600초/GPU14400초와 deadline을 유지했다.

현재 사용법은 [복구를 반영한 명령](README_USAGE_RECOVERY_KO.md)을 우선한다. 원본 생성 보고서의 과거 `run verify`/직접 supplement 명령은 보존된 verifier의 호환성 오류로 실패할 수 있다. 공개 CSV/그림 replay는 private 배열·가중치 없이 가능하다. 실제 모델 검산은 원래 로컬 자산·private 예측·receipt가 필요하다. 추가 예산의 cap 연장·미실행 PeMS B·no-Bolt 대조는 이번 실행 범위 밖이고 새 실험으로 등록해야 한다.

## 관련 원리의 한계

[TimeMixer](https://proceedings.iclr.cc/paper_files/paper/2024/hash/a7ac8a21e5a27e7ab31a5f42a0117bdb-Abstract-Conference.html)의 multiscale mixing, [NHITS](https://ojs.aaai.org/index.php/AAAI/article/view/25854)의 multirate sampling/interpolation, [Bates–Granger](https://link.springer.com/article/10.1057/jors.1969.103)의 forecast combination은 설계 배경이다. 이 논문들의 전체 모델을 재현하거나 본 변형의 성공을 입증한 실험으로 해석하지 않는다.
