# 내부 적응의 성능–비용 격차: 제한된 두 자료 확인

**실행 계약 v1 · 2026-09-22**

## 1. 목표와 소비처

같은 관측 정보에서 동결 모델의 출력 보정으로 얻지 못하는 미래 예측의 이득이 내부 LoRA에 남고, 그 이득을 얻는 데 실제 추가 학습비용이 드는지 확인한다. 사용자는 두 자료의 독립 보고서를 읽고 **새로운 효율적 PEFT를 개발할 문제의 여지가 있는지** 결정한다.

이것은 새 PEFT 후보를 주장하는 실험이 아니다. 과거 A/B·MAG·HIER·TEMP·Query의 중단을 해제하지 않는다. 기존 LoRA가 먼저 F0보다 실패해야 한다는 관문도 없다. 새로운 생성기·초기화·side network·loss를 넣지 않는다.

예상하는 연구 진입 근거는 `동일 정보의 싼 해법 < 내부 적응의 예측 품질`, 그리고 그 품질을 얻는 준비비 포함 비용 차이가 함께 남는 것이다. 싼 해법으로 충분하면 그 설정에서 새 모듈을 만들지 않는다. 서로 다른 자료의 숫자를 합쳐 승자 하나를 만들지 않는다.

## 2. 목적에서 도출한 비교와 가지치기

| 행동 | 필요한 이유 | 기대 결과 / 목적 지지 / 읽는 사람이 물을 질문 |
|---|---|---|
| 긴 문맥의 F0와 출력 보정 | 기존 기능만으로 해결되는지 확인; 피드백 요약과 원자료 사용의 혼선을 제거 | 싼 기준을 확정 / 직접 / 그냥 과거를 더 읽으면 되지 않나? |
| 같은 입력의 내부 LoRA | 달성 가능한 정확도 참조; 비용 차이를 실제로 측정 | 정확도 이득과 비용을 분리 / 직접 / 내부 적응이 필요한가? |
| LoRA+ | 알려진 단순 학습 개선의 설명을 배제; 비싼 LoRA를 일부러 기준으로 삼지 않음 | 표준 최적화 뒤에도 차이가 있는지 / 직접 / A와 B의 학습률만 바꾸면 되지 않나? |
| 두 자료를 독립 실행 | 과거 양성 ETTm2와 반대 사례 Jena를 함께 보존; 한 자료의 패턴을 일반화하지 않음 | 자료별 다른 판정도 허용 / 직접 / 어느 설정에서 필요한가? |
| 캐시·준비·검증 비용 공개 | 동결 방법에만 공짜 준비비를 주지 않음; 계산 편차와 수렴을 구분 | 같은 품질의 실비 비교 / 직접 / 빠르다는 비용에 무엇이 포함됐나? |

가지치기: 전체 미세조정, 새로운 어댑터, 새 데이터 검색, 다수 문맥/HPO 조합, 원인별 대형 분석은 이번 범위에서 제외했다. 전체 미세조정이 상한이라는 전제도 없다. 격차가 확인된 다음의 방법 개발은 별도 승인이다.

## 3. 이전 S1과 유지·변경한 부분

[확인: 저장소 코드] 이전 S1은 Chronos-2의 ETTm2 384/96 관측(15분 간격의 4일/1일), Jena 576/144 관측(10분 간격의 4일/1일), 마지막 116일 구간을 사용했다.

| 항목 | 이번 고정 설정 |
|---|---|
| backbone | `amazon/chronos-2`, revision `29ec3766d36d6f73f0696f85560a422f50e8498c` |
| 시간 설정 | 모든 학습군에 **8일 과거 → 1일 미래**. ETTm2 768/96, Jena 1152/144 |
| 데이터 기간 | ETTm2 저자 CSV 전체; Jena 공식 2024년 CSV 전체 |
| 분할 | 전체 원시간축 TRAIN60% / VAL20% / TEST20%, 일 경계 |
| 대상 | ETTm2 숫자 7개 변수, Jena 숫자 21개 변수. TRAIN 상수만 사전 제외·공개 |
| 문맥 진단 | 무학습 F0만 4일·8일 둘 다 계산; 학습군은 8일 하나만 |
| 주 목표 | raw-space TRAIN 표준편차 정규화 mean twice-pinball, 낮을수록 좋음 |

이는 **과거 S1의 정확한 재현이 아니라 더 긴 공통 정보와 더 넓은 시간분할의 개발 재검토**다. 예전 점수를 새 점수와 같은 조건인 것처럼 비교하지 않는다. 이미 개발/사전학습에 사용됐을 수 있어 외부 독립 확증이 아니다.

## 4. 실제 학습군

| 학습군 | 구현 | 고정 학습률 |
|---|---|---|
| HEAD / ETTm2 | 동결된 미래 patch 표현 → 768–533–336 ReLU residual MLP. 마지막 층 0 초기화 | AdamW 3e-4 |
| HEAD / Jena | 기존 output_patch_embedding 전체만 학습. backbone hidden 캐시 | AdamW 3e-4 |
| LORA | attention q/k/v/o 96개 + native output projection 1개, rank8 alpha16 | AdamW 1e-4 |
| LORAPLUS | 같은 rank·위치·초기값. LoRA A/B 학습률만 구분 | A 1e-5, B 1.6e-4; 비율16 |

HEAD 종류는 과거 S1의 source별 검증 선택을 근거로 **이번 결과 전에 고정**했다. 새 설정에서 모든 HEAD 계열을 튜닝한 최적 출력 보정이라고 주장하지 않는다. ETTm2 MLP 589,301개, Jena native head 약 3,653,280개, 내부 LoRA 1,206,912개를 예상하되 실제 모델 검사로 기록한다. 내부군에 output projection도 포함되어 있으므로 **encoder-only 수정의 순수 인과효과**로 해석하지 않는다.

LoRA+는 공식 PEFT `create_loraplus_optimizer`의 실제 파라미터별 LR 그룹과 로컬 그룹을 runtime에서 비교한다. 공식 LoRA+ 전체 논문 재현이나 충분히 튜닝한 최선의 학습률이라는 뜻은 아니다. 같은 base LR에 비율만 바꾼 원인 분해가 아니라, **사전 고정한 두 학습 recipe의 실용 비교**다.

F0_LONG, F0_SHORT, 계절 naive는 optimizer 0의 기준선이다. naive는 직전 1일 관측을 미래 1일로 반복하고 모든 분위수에 동일 점값을 준 퇴화분포이므로, 확률적 불확실성을 제대로 모델링한다고 부르지 않는다. AutoETS·새 출력 calibration·추가 LR grid는 이번 자동 실행 범위에 없다.

## 5. 데이터 출처와 무결성

- ETT: https://github.com/zhouhaoyi/ETDataset
- ETTm2 CSV: https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv
- Jena 공식 안내: https://weather.bgc-jena.mpg.de/weather_data.html
- Jena 2024 archive: https://weather.bgc-jena.mpg.de/mpi_roof_2024.zip
- Jena 관측의 출처·라이선스는 공식 사이트(CC BY 4.0 표기)를 보고서의 SOURCE manifest와 함께 보존한다.

원본 SHA256·파일 크기·날짜·컬럼·원 간격·단위명·결측·제외 구간을 저장한다. 명시적인 `--ettm2-file`, `--jena-file`로 실제 로컬 CSV를 지정할 수 있고, 경로를 추정해서 다른 자료를 사용하지 않는다. Jena -9999 센티널은 NaN으로 표시하고 보간하지 않는다. 중복·불규칙 시간축이면 조용히 resampling하지 않고 BLOCK한다.

모든 군에 동일한 완전 관측 문맥/정답 원점을 사용한다. 결측 검사로 탈락한 비율은 자료 감사에 남는다. TRAIN 상수 변수 제외와 scale 계산만 TRAIN으로 수행한다. 미래값의 크기·사건 조건·모델 성능으로 원점을 선택하지 않는다. 마스킹된 데이터셋 전체를 운영 수준으로 다루는 실험이라고 일반화하지 않는다.

학습 원점 후보는 6시간 간격, 그 전체에서 균등 표집. VAL/TEST는 24시간 간격, 미래 1일 전체가 해당 역할 안에 들어가는 모든 원점. TEST context는 이전에 관측된 TRAIN/VAL 부분까지 허용한다. 전체 원자료가 같은 프로세스에 로딩될 수는 있지만 모델·특징 생성에는 문맥만 전달한다. 물리적으로 TEST 파일을 접근하지 않았다는 주장은 하지 않는다.

## 6. 학습과 선택

두 반복 seed92601/92602, 학습군3개, 자료2개다. 선택 전용 seed는 없으며 두 seed의 원점수 평균을 보고한다. 동일 seed에서는 학습할 원점·샘플 순서·LoRA 초기값을 맞춘다.

- 경로당512 updates; 저장 checkpoint 0/32/64/128/256/512.
- effective batch = 동일 원점의 전체 다변량 계열 묶음 4개. microbatch는 원점1개 전체채널.
- base FP32 파라미터 + BF16 autocast, cache_enabled=False. TF32/dropout off.
- 두 내부군 모두 non-reentrant block checkpointing. CPU로 본학습을 조용히 대체하지 않는다.
- native normalized pinball 학습. raw-space 정규화 pinball 평가와 구분한다.
- gradient clip1, weight_decay0, scheduler없음.
- BASE 및 buffers 동결. HEAD는 hidden 캐시를 만들어 반복 인코더 역전파를 피한다.
- checkpoint 선택은 각 모델의 VAL 주지표 최소, tie이면 이른 시점.
- cheap reference는 **HEAD/F0_LONG/계절naive 중 VAL 최선**, internal reference는 **LORA/LORAPLUS 중 VAL 최선**. TEST에서 기준선을 바꾸지 않지만 모든 TEST 결과를 공개한다.
- TEST 예측을 전부 저장·hash 봉인한 다음 채점한다.

한 자료에서 음성이어도 다른 자료는 독립 실행한다. 실패한 데이터 검사나 partial checkpoint를 숨기고 다음 단계로 가지 않는다. 부분 fit은 자동 재실행하지 않는다. 완결된 fit만 hash 검증 후 재사용한다.

## 7. 비용과 평가를 섞지 않기

`cold_ready_s`는 데이터 준비 + model load + HEAD의 전체 TRAIN/VAL feature cache + 해당 시점까지의 update 계산 + 방문한 validation(추론/채점/예측 저장) + checkpoint 저장을 합친다. journal fsync와 부가 무결성 hash 검사 시간은 비용모형에서 제외하고 실제 fit wall time에 별도 공개한다. 전체 job wall과 이 모델링된 cold cost는 같은 값이 아니다. head fit wall에는 이전 프로세스의 cache 준비 시간이 들어가지 않으므로 cache 항을 따로 보아야 한다.

캐시를 두 seed가 물리적으로 공유해도 **독립적인 HEAD 도입 비용에는 전체 수집 비용을 각각 청구**한다. 반대로 실제 전체 실험 지출에서는 캐시를 2회 생성했다고 부풀리지 않는다. TEST용 캐시는 평가 비용이고 adaptation 시간에 포함하지 않는다.

내부 LoRA 기준의512회 cold time 중앙값을 기준으로 0.5×/1.0× 예산에 들어오는 저장 checkpoint 중 VAL 최소를 고른 동시간 표도 제공한다. 보간으로 존재하지 않는 checkpoint를 만들지 않는다. 비용이 예산을 넘으면 해당 방법은 NO_CHECKPOINT_IN_BUDGET이다.

목표 품질 도달시간은 동일한 VAL 목표에 처음 도달한 저장 checkpoint의 상한이다. 목표는 V-selected internal method의 두 seed512회 VAL 평균 ×1.01. F0에서 이미 달성되면 NO_ADAPTATION_HEADROOM, 미도달은 CENSORED. 초기 상태에서 목표가 만족되지 않은 경우에만 적응 도달시간을 비교한다. 동일 checkpoint에 도달했으면 더 적은 updates의 가속이라고 주장하지 않는다.

훈련 GPU peak와 캐시 peak를 따로 기록한다. 메모리의 작은 측정차를 유의한 구조 개선으로 주장하지 않는다. 실행 순서는 seed2에서 반대로 하고 각 fit은 새 subprocess. 다른 GPU 프로세스를 종료하지 않는다. 타이밍 변동이 크면 속도 결론을 유보한다.

## 8. 판정은 연구 문제의 여지에 한정

아래 숫자는 학술적 법칙이나 논문 PASS 기준이 아니라 **사전 제안한 투자 선별값**이다. TEST 결과를 보고 바꾸지 않는다.

- 내부군이 VAL로 정한 cheap reference보다 평균1% 이상 개선하고 두 seed 모두 개선: 정확도 격차의 개발 신호.
- 같은 시점까지의 cold-ready 비용이 cheap reference의1.5배 이상: 현재 비용 격차도 남는 후보 상황.
- HEAD가 마지막 checkpoint를 고르고 마지막 구간 VAL이0.2% 이상 계속 좋아지면 `QUALITY_GAP_BUDGET_LIMITED`. 출력 보정의 충분한 학습 한계라고 주장하지 않는다.
- timing CV가0.2 이상이면 `QUALITY_GAP_TIMING_UNCERTAIN`. 원인 불명의 느린 실행을 알고리즘 가속 근거로 쓰지 않는다.
- cheap recipe가 비슷하거나 더 좋으면 `OUTPUT_SUFFICIENT_AT_CURRENT_RESOLUTION`. 통계적 동등성 증명이 아니라 이번 개발 해상도의 판단이다.
- 내부 정확도는 좋지만 비용차가 작으면 `QUALITY_GAP_NO_COST_BOTTLENECK`.
- 조건이 엇갈리면 `HOLD_NO_NEW_METHOD_AUTHORIZATION`.

모든 판정은 한 모델·두 source·두 seed·고정 recipe의 범위다. 별도의 HEAD HPO/LoRA HPO를 한 최종 frontier가 아니다. 특히 LoRA+와 표준LoRA의 학습률 범위 차이는 공개한다. 작은 양성값을 실용적 격차로 승격하거나 한 음성 결과로 분야 전체를 기각하지 않는다.

시간축7일 paired block bootstrap2000회는 같은 날짜의 전체계열/seed를 함께 유지한다. 두seed로 optimizer 모집단의 불확실성 전체를 알았다고 하지 않는다. 채널별 손익·정규화MAE·coverage·width·crossing을 모두 보고한다.

## 9. 산출물과 종료

각 자료에 `REPORT_KO.md`, `DECISION.json`, `FINAL_DECISION.md`, SCORES/CHANNEL_SCORES/ORIGIN_SCORES/EFFECTS/CURVES/RESOURCES/TIME_TO_TARGET CSV, source/model/cache/prediction/selection manifests를 만든다. 보고서에는 실제 생성된 PNG 다섯 장이 상대경로로 연결된다.

1. updates별 VAL 곡선
2. 준비비 포함 비용별 VAL 곡선
3. 선택된 TEST 점수와 seed점
4. 실제 비용–품질 산점도
5. 채널별 internal 대 HEAD 손익

통합 `SUMMARY_KO.md`는 두 자료의 개별 결론을 연결한다. `VERIFICATION.json`은 source/config seal, 모델 상태·optimizer finite, 실제 업데이트 수, 선택 재계산, seed 일정, 독립 scalar metric, 개선율, 이미지 링크를 검사한다.

**최대 main12fits=6,144 updates, smoke12회.** 새로운 생성기0, 새로운 데이터0, LR/rank/epoch 자동 확장0. 수행 후 code+small reports+CSV+PNG만 scoped commit/push. raw자료·모델가중치·예측npz·optimizer checkpoint·credential은 제외한다. `PUSH_RECEIPT.json`은 실제 확인된 artifact commit과 보고서 URL을 남긴다. force push, unrelated staged files, 기존 unpushed commits를 함께 push하는 행동은 금지한다.

## 10. 가까운 일차 출처

- LoRA+: Efficient Low Rank Adaptation of Large Models(2024/ICML): https://proceedings.mlr.press/v235/hayou24a.html
- PEFT LoRA+ optimizer API: https://huggingface.co/docs/peft/en/package_reference/lora
- Chronos-2 official model code: https://github.com/amazon-science/chronos-forecasting/blob/10afa9ebe016e514f9d7dc1aa873f66af57e116b/src/chronos/chronos2/model.py
- Model config: https://huggingface.co/amazon/chronos-2/blob/29ec3766d36d6f73f0696f85560a422f50e8498c/config.json
- 기존 S1 모델/학습/자료: https://github.com/CanelE452/mltimeseries/tree/main/experiments/peft_adaptation_scope_v1
- 기존 S1 수치·비용: https://github.com/CanelE452/mltimeseries/blob/main/results/peft_adaptation_scope_v1/summary_and_costs.json

이 파일의 새 설정·선별값은 [설계]이며 선행논문의 실험 계약을 그대로 복제한 것이 아니다. 결과가 양성이어도 새 PEFT 기여·신규성·논문 가능성을 자동 선언하지 않는다.
