# Jena: 초기 LoRA·B4 방법군 최소 비교

말단 wrapper 기록과 과학적 validation·비용을 구분한 복구

원 POST wrapper는 mixed-slash cache-key 조회 때문에 manifest 및 fit guard SHA를 찾지 못해 **PARTIAL_VALIDATIONS_COSTS_PRESERVED**로 끝났다. 원 결과 SHA `28911ed451a95ce4af157dd863914f9078f5b0f3f11ab82d0ee5ae2efd6d6080`와 네 `success=false` 행·null fit-guard 기록은 그대로 보존했다. 실제 원 네 validation의 science PASS·guard0·STARTED/선택 artifact를 별도 독립 검증하고, 학습·validation을 다시 실행하지 않은 **새 비용4조건**을 결속했다. 별도 recovery RESULT SHA는 `15019bbbcd81793510b6f7c8cbac2a4accc47698cb12ef20929f77df6a28541c`다. 원 wrapper를 성공으로 수정한 결과가 아니다. [원 기록·복구 경로의 독립 감사](INDEPENDENT_INITIAL_FAMILY_RECOVERY_AUDIT_ROOT_v2.json).

이번 두 신규 방법과 기존 B2 reference의 학습 metadata는 모두 search 누적3epoch, **선택 fresh refit1epoch**, search/refit **1863/752 update·29769/12020 exposure**로 일치한다. 동일 노출이 LoRA C21 대 B4/B2 K6의 동일 F 연산·동일 학습률·동일 모수 수를 뜻하지 않는다. B2는 기존 selected artifact의 정확도를 재사용한다.

첫 recovery 감사는 LoRA에 없는 P/E·D stage 열까지 요구해 거절되었으며 [실패 감사](INDEPENDENT_INITIAL_FAMILY_RECOVERY_AUDIT_ROOT_v1.json)를 그대로 보존했다. 새 감사는 실제 LoRA4열/B4 6열 schema와 같은50trial·F counter·정밀 요약을 검증했다. 없는 stage 값을0으로 만들거나 비용을 다시 측정하지 않았다.


이 보고가 답하는 질문은 **등록된 두 다른 적응 원리를 실제로 학습·전체 V 추론·검산·단독 비용까지 수행했는가, 그 원수치는 무엇인가**다. 아래 결과는 같은 Jena C21/L512/H48, seed92601, CPU2, 반복 개발 V43의 normalized channel-macro MSE와 MAE다. 이 두 경로의 완료가 주방법 우위나 전체 연구 해결을 뜻하지 않는다.

[원정밀도 CSV](INITIAL_FAMILY_RESULTS_v1.csv) · [입력·출력 SHA 결속](INITIAL_FAMILY_BINDINGS_v1.json)

```text
role           MSE                 MAE                 registered-K/effective-F-width
LORA_R8_C21    0.56392294174335067 0.39900992921234152 0/21
B4_K6          0.70435308018904963 0.49675056670884404 6/6
```

LoRA는 등록 K0/rank8/alpha16이고 실제 F 입력은 원 C21이다. B4는 실제 K6 시계열을 F에 넣는다. 등록 K0를0개 입력으로 해석하거나, LoRA C21과 B4 K6를 같은 입력 구조·같은 모수 수의 격리 실험으로 해석하지 않는다. 원 등록 LoRA lr0.0001과 B4 lr0.001도 보존했다. 같은 방식의 decoder만 비교한 실험이 아니다.

```text
role           selected-refit search/refit-updates search/refit-exposures search/refit-seconds lr training-status
LORA_R8_C21    1.0 1863/752 29769/12020 10727.879892699886/4058.1899608001113 0.0001 SCREEN_COMPLETE_CONVERGENCE_UNRESOLVED
B4_K6          1.0 1863/752 29769/12020 3428.3985977000557/1376.1958650001325 0.001 SCREEN_COMPLETE_CONVERGENCE_UNRESOLVED
```

두 조건은 search A50[0,10482), A_stop[10482,12579), fresh refit A60[0,12579), V[14675,16772)의43 origins·43344 observed labels를 공유한다. 43344는 원 manifest에서 확인한 유효 label 수이며 두 결과를 합쳐 독립 표본 수를 늘리지 않는다. 원 scaler·target·mask·origin·channel 지문을 대사했다. 이번 두 조건과 기존 B2 reference는 search3/refit1 및 실제 update/exposure가 같다. 동일 F 연산·학습률·용량 대조는 아니다. **단일 seed**, 제한 학습, 반복 개발 V이고 충분한 수렴·독립 일반화는 미확정이다. MSE 목적의 실제 학습을 재구성 학습 실패로 바꾸어 설명하지 않는다.

현재 entry의 새 계산은 허용한 과거 prefix만 사용하며 보호 T를 새로 파싱하거나 선택하지 않았다. 그러나 원 Jena **전체 TRAIN 정규화 ancestry**와 과거 quarter/probe의 **full-loader ancestry**는 그대로 남는다. prefix 경계 검사 통과를 원자료 전체의 독립적인 causal 생성·정규화 검증으로 확대하지 않는다. 이 보고를 위해 전처리나 기존 모델을 재학습하지 않았다.

```text
role           B1 median/p90 seconds B8 median/p90 seconds CPU weights MiB B1/B8 observed-tree RSS bytes B1/B8
LORA_R8_C21    0.13867014995776117/0.14114590981043876 1.1420140000991523/1.2185885000042618 183.15481948852539/183.15481948852539 1101574144/1265180672
B4_K6          0.04856874980032444/0.049496910162270071 0.34662930015474558/0.37185468007810413 182.6283073425293/182.6283073425293 984928256/1045618688
```

각 비용은 물리 history/mask→채움·정규화→해당 적응 경로/F→원 C 출력→FP64 inverse·CPU 반환의 **warm 한 batch 전체**다. B8은8 origin을 담은 한 batch이며8로 나눈 시간이 아니다. ordinary timed20의 median/p90를 표시하고 warm10·별도 intrusive stage20은 구분했다. intrusive stage 합=ordinary whole을 주장하지 않는다. LoRA B8은168 streams·2 F calls/trial, B4 B8은48 streams·1 call/trial이다. 실제 whole cost의200 trial·8 summary·40 지연시간 요약 필드를 원 exact 정책으로 검산했다.

```text
role           Route-model-load-seconds B1/B8 load-and-LoRA-merge-in-whole B1/B8 restore-snapshot-in-Route-weights B1/B8
LORA_R8_C21    0.15483110025525093/0.14895740011706948 False/False False/False
B4_K6          0.03683099988847971/0.03582199988886714 False/False NOT_APPLICABLE/NOT_APPLICABLE
```

**cold load와 LoRA merge는 whole trial timer에 포함되지 않는다**(`LoRA_merge_and_model_loading_in_whole_trial_timer=False`). 표의 Route `model_load_seconds`는 원 기록의 별도 소요시간으로, F 전체 cold startup·서비스 전체 응답시간으로 확대하지 않는다. LoRA는 CPU restore snapshot이 Route weights 집계에 포함되지 않았다(`CPU_restore_snapshot_bytes_in_Route_memory_state=False`). snapshot 용량을 추정해 채우지 않는다. CPU weights는 원 Route의 모듈 집계이고 총 process memory가 아니다. 관측 tree RSS를 별도 보존했고, F만의 resident bytes는 미측정 NA다. F instances=1은 모델 메모리가 K배로 줄었다는 의미가 아니다.

LoRA 비용 중 effective merged F hash와 Route.close 이후 복원한 원 unmerged F hash를 CSV·감사에 따로 보존했다. **LoRA는 내부 저랭크 adapter를 학습하고 이를 merge한 effective F를 사용한다. 원 base weights의 보존·복원 hash 일치가 모든 effective F의 동결을 뜻하지 않으며, B4/B2의 frozen F 대조와 구분한다.** 두 hash가 항상 다르다고 가정하지 않는다. LoRA의 fresh CPU6 replay126 streams/2calls와 추가 merge parity42 streams/2calls는 별도 활동이다. 검산 완료를 새 모델 추론·새 독립 일반화 증거로 부르지 않는다.

기존 [Jena B2 K6 manifest](../postT_V_only_cpu_lag_20261016_v1/predictions/postT_cpu_jena_E3_B2_k6_s92601_v1/V_select/PREDICTION_MANIFEST.json)는 원 fullV outer·STARTED·guard/SPEC·실제 CPU forward 소스까지 추적된 기존 참고다. 해당 manifest의 원 SHA는 `6a885f3c8f298009ae93c731409baff116263a41c9c17ecaff713cf5e9499148`이다. 기존 단독 CPU 비용은 guard4GiB 등 원 resource/quiet 감사 범위가 현재3GiB 계약과 다르므로 여기 새 두 row의 동일 계약 Pareto로 섞지 않았다. 미확인 장치의 Raw/G/PCA도 새 CPU 공정 비교표에 넣지 않았다. B2 대비 추가 가치의 최종 판단에는 기존 결과의 정확한 비용·검산 scope를 따로 대사해야 한다.

**전체 MASTER 미완료**다. 이 초기 묶음은 네 family recipe 중 Jena LoRA·B4 두 조건만 포함하며 Jena ADAPTS와 Beijing B4는 자동 완료로 표시하지 않는다. AdaPTS 원 논문 전체 재현을 뜻하지 않는다. 실제 두 학습·fullV·독립 FP64·CPU6·4비용과 source48 보존·강한 recorded process tree·quiet/nonoverlap가 모두 확인될 때만 이 완성 보고를 생성한다. 부분 종료면 builder가 거절하며 ROOT는 성공·실패·미실행 및 저장점·재개 방법을 별도 정직한 부분 보고로 남길 수 있다. 다음 판단은 이 반복 V의 오차·whole 비용·학습량·구조 혼재를 함께 검토하는 것이며 자동 새 모듈·학습 연장·주방법 변경은 없다.
