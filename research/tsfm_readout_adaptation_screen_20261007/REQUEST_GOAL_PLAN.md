/goal

# TSFM 적응 경로 진단 — 원채널 출력·헤드 적응부터 검증
작성일: 2026-10-07
대상 저장소: CanelE452/hierarchical-tsfm-peft
문서 성격: 단계별 실행 범위를 분리한 계획. 현재 응답에서 실험을 실행한 것은 아니다.
용어: TSFM=시계열 파운데이션 모델, PEFT=파라미터 효율적 미세조정, LoRA=저랭크 수정분을 학습하는 적응, F0=추가 적응 없는 원래 모델, head=내부 표현을 미래값으로 바꾸는 출력부. C/L/H는 원채널 수/과거 관측 개수/미래 관측 개수이며 seed는 난수 초기화·표본 순서의 재현 식별자다.

## 1. 최상위 목적

목표 시계열에서 동결된 사전학습 모델의 최종 출력을 보정하는 것, 기존 내부 표현을 읽는 예측 헤드를 적응시키는 것, 백본 내부에 LoRA를 추가하는 것의 실제 추가 가치와 비용을 구분한다. 그 결과로 “내부 적응이 추가로 유리한 경우를 저비용으로 미리 판단한다”는 연구 질문 자체가 성립하는지 검토한다.

교수님과 연구자가 결정할 것은 다음이다.
- 동결된 표현의 읽기 방식을 바꾸는 표준 대안으로 충분한가?
- 헤드에 같은 추가 학습 기회를 주어도 내부 적응의 이득이 남는가?
- 그 이득이 대상에 따라 달라지고 실제 비용 차이도 있어, 적응 경로를 선택할 가치가 있는가?
- 더 나아가 저비용 진단이 기존 전이 지표·데이터 복잡성·짧은 파일럿보다 좋은 선택을 가능하게 하는가?

처음 세 질문이 성립하지 않으면 새 선택기나 새로운 PEFT를 만들지 않는다. LEVEL을 반드시 살리거나, 모든 적응법 중 하나의 보편적 승자를 만드는 것이 목표가 아니다.

## 2. 문제정의 후보와 아직 모르는 것

[추정·미검증]
같은 예측 오차라도 (a) 출력의 보정, (b) 동결 내부 표현의 재읽기, (c) 내부 표현을 바꾸는 적응이 각각 유리한 경우가 있을 수 있다. 원래 F0 오차만으로 내부 적응의 추가 이득을 판단하기 어려울 수 있다. 이 가능성을 검증한 뒤에만 진단·선택 방법으로 발전시킨다.

- “HEAD가 실패했다”는 것은 정보 부재의 증명이 아니다.
- “LoRA가 HEAD보다 좋다”만으로 내부 적응이 원리적으로 필수라는 결론을 내리지 않는다.
- “HEAD 후 LoRA”라는 학습 순서, 동결 특징의 전이 적합성 평가, 자동 PEFT 선택은 기존 원리다.
- “시계열에 적용했다”, 새로운 이름, 모듈 조합, 테스트 개선만으로 신규성을 확정하지 않는다.
- PCA, C→K 압축, residual G, Conv, 고정 Bolt-small의 배포 필수성은 미리 정답으로 고정하지 않는다.

## 3. 목적 트리와 가지치기

최상위: 불필요한 적응비용을 줄이면서 미래 예측을 개선하는 타당한 적응 경로 판단.
  ├─ S1: 출력/헤드라는 단순한 대안의 실제 이득 확인.
  │    없으면: 약한 대조를 이긴 내부·외부 모듈을 필요하다고 오판할 수 있음.
  ├─ S2: 같은 HEAD 부모·추가 학습 기회에서 내부 적응의 추가 가치 확인.
  │    없으면: 초기화, 부모 성능, 더 오래 학습한 효과를 내부 적응 효과와 혼동.
  └─ S3/S4: 그 이득의 저비용 예측과 실제 의사결정 가치 확인.
       없으면: 비교 분석을 새 적응 선택 방법으로 부를 수 없음.

이번 우선순위에서 제외/강등:
- Conv/K/회전: 원채널 표준 적응의 부족을 먼저 확인하므로 보류. 기존 결과 삭제 금지.
- 새 데이터/백본, 새 게이트/새 loss: 첫 회차에는 불필요. 성공 조건을 만들기 위해 추가하지 않음.
- 완료된 모델 정확도 재실행: 동일 계약 증거를 재사용.
- full fine-tuning: 초기 누락 헤드 대조를 채우는 데 필수는 아님. 최고성능 상한이라고 가정하지 않음.
- 진단 신호가 없는 선택기 개발: 하지 않음.

## 4. 승인·실행 범위

첫 실행 범위는 S0 감사·계약·견적과 S1 OUTPUT/HEAD뿐이다. S1 최대 24개 과학적 적합 구성, 신규 선택 모델 최대 12개(각각 TEST-A/B를 출력)다. S2는 조건부 최대 24개 적합의 별도 단계이며 S3/S4는 현재 연구 방향 명세만 포함한다.

이 문서를 전달받으면 먼저 로컬의 승인 범위를 확인한다.
- 읽기 감사·기존 증거 조사·코드 기반 계획 구체화는 진행한다.
- 새 GPU 작업·적합·캐시 생성·실제 비용 측정은 이 문서의 S0/S1 범위와 실측 견적을 반영한 자원 상한이 승인된 뒤 실행한다.
- 이전 다른 회차의 3/4/12시간 예산을 가져오지 않는다. 현재 시간/디스크 상한은 로컬 견적으로 수치화하여 승인받는다.
- 사용자가 S0/S1 및 그 예산을 승인한 이후에는 각 데이터·학습률·seed마다 반복 승인을 받지 않는다. 같은 범위의 구현→선택→평가→비용→검산→보고까지 진행한다.
- S2나 새 알고리즘으로 자동 확장하지 않는다. 목표·데이터·함수·비용의 실질적 변경만 재승인한다.
- 이전 E/D·Conv 회차가 이미 승인되어 실행 중이면 임의로 kill하지 않는다. 현재 상태를 기록하고 사용자와 전환/병행 우선순위를 정한다.
- 이 계획은 이전 산출물을 덮어쓰지 않는 후속 우선순위안이다.

/plan

# S0. 원격·로컬 감사, 선행 확인, 실행 계약

## S0-1. 확인한 출발점

[확인: 2026-10-07 이번 조회]
- remote main: 8646e54630add0777c90686b5a8e4bef20c7dcea. 실행 때 다시 조회하고 차이를 기록한다.
- v18 STATUS: BLOCKED_STOP_RESOURCE; 신규 fit/optimizer update/TEST 예측 모두 0.
- v18 보수적 전체 추정 11,421.533초, 기존 cap 10,800초. 실제 preflight 약21.848초. 실제 3시간 소모나 성능 실패가 아니다.
- head_only 검색의 일부 결과는 v14 보정 모듈 저장/복원 검사다. 실제 native forecast-head-only 학습 완료와 구분한다.
- 기존 LoRA helper는 r=8, alpha=16, dropout=0, target_modules q/v, bias=none이고 294,912개를 검산한다.
- 공식 최신 Bolt source의 output_patch_embedding은 비선형 경로와 residual 경로를 가진 ResidualBlock이다. 이번 실험의 기준은 최신 main이 아니라 기존 local pinned 모델·라이브러리다.

직접 읽은 기존 파일:
- research/level_matched_learned_ed_v18_20261006/STATUS.md
- research/level_matched_learned_ed_v18_20261006/model_v18.py
- src/hier_peft/lora.py
- 이전 고정 모델·LoRA·LEVEL의 selection, prediction, scoring, cost 원장

위 위치는 원격에서 확인한 경로다. 이후 제안하는 새 디렉터리·arm·산출물 이름은 기존 파일/변수라는 뜻이 아니다.

## S0-2. 이름이 아니라 함수와 증거로 중복 확인

branch/HEAD/origin/dirty files/processes/local untracked/cache를 읽기 전용으로 확인한다. 권한 없는 다른 프로세스나 파일을 정리하지 않는다.

찾을 것:
1. 최종 F0 예측의 C→C affine OUTPUT와 정확히 같은 완료 실험.
2. 동일한 Bolt-small의 실제 output prediction block 전체만 학습한 HEAD 실험.
3. 기존 원채널 MSE-LoRA, native-quantile-loss LoRA의 구분.
4. v6 Robin, v7 Jena, v12 Peacock의 canonical 부모·데이터·선택·예측 연결.
5. v15 independent decoder, v14 external head, v18 E/D와 새 OUTPUT/HEAD의 차이.
6. 안전하게 재사용 가능한 원채널 TRAIN/VAL F0 예측 또는 native head-input feature cache.

분류는 EXACT_REUSABLE / RELATED_DIFFERENT_CONTRACT / CODE_ONLY / PREFLIGHT_ONLY / RESOURCE_BLOCKED / COMPLETED_NEGATIVE / NOT_FOUND_IN_SEARCH_SCOPE로 한다.
검색 실패를 “없다”로 단정하지 않는다. 다른 함수의 성공이나 step0 부모 재선택을 새 방법 효과로 세지 않는다.

동일 체크포인트·원점·마스크의 기존 accuracy는 재사용한다. 다른 회차의 과거 실행시간과 새 실행시간을 나눠 속도비를 만들지 않는다. 재채점이 필요하면 새 학습/새 독립 결과와 구분한다.

## S0-3. 선행 중복 지도

공식 초록/원문/코드를 실제 읽은 깊이별로 기록한다. 아래 연구의 전체 재현을 S1에서 요구하지 않는다.

- LogME: Practical Assessment of Pre-trained Models for Transfer Learning(2021/ICML)
  동결 특징을 통한 전이 적합성 평가. 회귀도 다룸. 시계열에서 성공하는지는 별도 검증.
- AutoPEFT: Automatic Configuration Search for Parameter-Efficient Fine-Tuning(2024/TACL)
  적응 구조·위치·크기의 자동 성능-비용 탐색. 목표 영역은 언어 모델.
- Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution(2022/ICLR)
  linear probing 후 fine-tuning이라는 기존 순서. 이 연구의 HEAD는 비선형 native head이므로 LP와 같다고 부르지 않음.
- Time-PEFT: Temporal and Multichannel Complexity-Based Fine-Tuning for Time-Series Foundation Models(2026/ICML)
  자료 복잡성과 적응 이득의 관계. 공식 수식·코드 미확인 지표를 새로 만들어 같은 이름으로 쓰지 않음.
- TRACE: Time Series Parameter Efficient Fine-Tuning(2026/Neurocomputing)
  LoRA 선택과 예측 헤드의 구성. native HEAD-only를 빠뜨린 비교가 아닌지 점검.
- AdaPTS: Adapting Univariate Foundation Models to Probabilistic Multivariate Time Series Forecasting(2025/ICML)
  입력/출력 및 decoder-only 대조의 가까운 출발점. OUTPUT는 공식 확률형 전체를 대표하지 않음.
- Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization(2018/JMLR)
  짧은 파일럿 기반 자원 배분은 표준 비교 대상.

각 선행에 problem / input information / training variables / gradient path / decision cost / evaluation domain / 우리 후보와 중복 / 아직 미검증인 차이를 적는다.
신규성 후보는 '헤드 이후 내부 적응의 추가 이득을 사전에 판별하는 정보와 실제 총비용의 개선'이다. 이 차이도 미검증이며 검색에서 안 보였다는 사실로 확정하지 않는다.

인접 원리의 전이 위험:
- 전이 지표: 좋은 회귀 적합성이 반드시 내부 적응의 추가 이득을 뜻하지 않는다.
- 자동 탐색: 진단이 짧은 후보 탐색보다 비싸면 가치가 없다.
- 두 단계 미세조정: 순서 자체는 기존 것이며 부모·추가 학습시간·optimizer 재시작이 혼입될 수 있다.

## S0-4. 첫 회차 공통 데이터·정보 계약

기존 local manifest/source를 읽어 다음을 고정한다.
- Robin: C=17, L=512, H=48; 관측 간격1시간.
- Peacock Education: C=13, L=512, H=48; 관측 간격1시간.
- Jena: C=21, L=512, H=48; 관측 간격10분.
- 같은 TRAIN/VAL/TEST-A/TEST-B boundaries, origin arrays/order, channels, imputation, target observation mask.
- 같은 TRAIN 통계의 외부 표준화, 같은 모델 내부 context normalization/inverse normalization.
- 같은 Chronos-Bolt-small checkpoint/revision과 라이브러리 코드. 패키지 업그레이드 금지.
- 같은 native q=0.5 위치의 첫48 출력, FP32/TF32 off, CPU thread 정책.
- TEST 중 업데이트, 미래 공변량, 문맥 확장, 채널 재선택, 자료 제외 금지.

S1의 신규 경로는 모두 원채널 C개를 유지한다. PCA·E·K·G가 들어가지 않는다.
F0/HEAD는 채널별 독립 실행이면서 가중치를 공유한다. OUTPUT는 원채널 예측 사이를 섞는다. 서로 같은 정보 결합 구조라고 쓰지 않는다.

q=0.5는 사전학습 모델의 출력 인덱스다. MSE로 HEAD를 적응시키거나 선형 조합한 값이 계속 정확한 조건부 중앙값/평균/분포 분위수라는 보장은 없다. point forecast라고 보고하고 확률 보정 성능은 이번에 주장하지 않는다.

# S1. 첫 실행 — OUTPUT 대 실제 HEAD 적응

## S1-1. 문제와 가설

[추정·미검증]
- OUTPUT 가설: 이미 만들어진 예측의 affine 채널 조합만으로 일부 오차를 줄일 수 있다.
- HEAD 가설: 최종 출력 직전의 동결 표현을 읽는 native head를 적응시키면, 출력 보정과 다른 추가 가치를 얻을 수 있다.
- 경쟁 설명: capacity/채널 혼합/초기화/optimizer/표본수/정규화/손실 정의가 차이를 설명할 수 있다.

OUTPUT 대 HEAD는 하나의 변수만 바꾼 인과 실험이 아니다. 표현, 용량, 채널 결합 구조가 다르다. 이 단계에서는 두 표준 완성 적응 경로를 비교한다.

## S1-2. 신규 명칭 OUTPUT_AFFINE

함수:
  y0 = F0(X)                     # dataset-standardized [B,48,C]
  yhat = y0 W + beta             # W:[C,C], beta:[C], 모든 h에서 공유

- 원채널을 각각 원래 Bolt-small로 예측한 뒤 조합한다.
- W=I, beta=0으로 시작. step0=F0.
- W와 beta만 학습. C²+C개(계산상 Robin306/Peacock182/Jena462); 실제 enumeration 기록.
- 추가 raw-history branch, horizon mixing, channel embedding, activation, residual predictor 없음.
- 미래 보조정보 없음. 훈련 목표와 접속 경로를 바꾸지 않음.
- 이 기능은 이전 계획의 D_ONLY_C와 정확히 같은 함수/recipe면 재사용 가능하다.

단순한 편향/스케일 보정만 분리하는 새 arm은 첫 회차에 자동 추가하지 않는다. OUTPUT가 무엇으로 개선되는지 세부 분해가 결정적일 때만 별도 계획으로 제시한다.

## S1-3. 신규 명칭 NATIVE_HEAD

이 arm은 post-output channel decoder나 새 linear probe가 아니다. 원래 Bolt의 마지막 forecast-output block을 사전학습 가중치 그대로 초기화하고 그 블록 전체만 적응한다.

공식 source에서 확인한 경계(실행 전에 pinned local source로 확정):
  context -> encode -> decode -> sequence_output
          -> output_patch_embedding -> reshape quantile outputs -> inverse normalization

- encode와 Transformer decode까지가 동결 prefix다.
- output_patch_embedding의 전체 ResidualBlock이 학습 가능한 head다.
- 새로운 작은 선형층으로 교체하지 않는다. 기존 비선형 경로·residual projection·bias를 유지한다.
- 원래 quantile 수와 native horizon 모양을 유지한다. loss는 q0.5의 첫48 관측 출력에만 걸어 기존 point contract와 맞춘다.
- 단일 encoder token이나 임의 pooling으로 sequence_output를 대체하지 않는다.
- instance_norm의 loc/scale 및 exact inverse 동작을 유지한다. 외부 TRAIN 표준화 좌표까지 되돌린 후 기존 loss를 계산한다.
- Encoder/Transformer decoder 전체를 '헤드'에 몰래 포함하지 않는다. 학습 경계를 architecture manifest로 그린다.
- head 전체 registered params, gradient-enabled params, 실제 loss에 연결된 parameter groups를 따로 기록한다. 기존 LoRA보다 파라미터가 적다고 가정하지 않는다.

최소 계산 구조(실제 API는 local source로 검증):
  with torch.no_grad():
      h, norm_state = frozen_prefix(X)
  native_outputs = trainable_original_head(h)
  yhat = original_unscale_and_select(native_outputs, norm_state)
  loss = parent_masked_macro_mse(yhat, target, mask)

주의:
1. eval()와 requires_grad는 다르다. 캐시와 online이 같은 함수를 갖도록 dropout은 전체에서 꺼 둔 상태로 head에 gradient만 켠다. 이 선택을 source에 기록한다.
2. model.train()가 뒤에서 frozen prefix/dropout을 다시 켜지 않도록 실제 module state를 검사한다.
3. forward 전체를 no_grad/inference_mode로 감싸면 head 학습이 끊긴다.
4. inference_mode tensor는 이후 autograd 저장과 충돌할 수 있으므로 학습용 h는 no_grad 또는 안전하게 복사된 일반 tensor를 사용한다.
5. 기대하는 것은 head의 loss-relevant gradient다. 미사용 분위수/미래 출력 행에서 gradient가0인 것은 곧 버그가 아니다. 모든 scalar가 업데이트됐다는 PASS 조건을 만들지 않는다.
6. native_head는 비선형 block이라 'linear probing 완전 재현'이라고 이름 붙이지 않는다.
7. frozen prefix 불변은 파라미터·버퍼 해시로 확인한다. head 변화가 다른 양자 출력도 바꿀 수 있으므로 probabilistic fidelity를 주장하지 않는다.

## S1-4. 기준선 역할

- PRIMARY: OUTPUT_AFFINE vs F0; NATIVE_HEAD vs F0; NATIVE_HEAD vs OUTPUT_AFFINE.
- contextual within-backbone: 현재 선택된 원채널 MSE-LoRA 및 LEVEL.
- practical external systems: NLinear, Bolt Mini/Tiny, Chronos-2/Small 기존 정확도.

NATIVE_HEAD vs 기존 LoRA는 탐색기회·초기화·학습 변수 차이가 남는 참조다. 이를 S2의 부모가 같은 내부 적응 추가이득이라고 쓰지 않는다.
새 비용 회차에 포함하지 않은 기준선의 과거 속도/메모리를 새 비교의 승패에 사용하지 않는다.

## S1-5. 학습 recipe와 적합 수

[추정·미검증: 이전 recipe를 유지하는 첫 대조안, 각 모델의 최적 튜닝이라는 뜻 아님]
- 두 arm × 세 자료 × LR {1e-3,1e-4} × seeds {92601,92602}: 최대24 scientific fits.
- 최대120 epochs; parent의 patience6, epoch_origins512, effective_batch4, AdamW, weight_decay0, clip_grad_norm1을 source와 대조한다.
- scheduler의 factor/patience/threshold, sampler의 실제 phase balance, schedule seed 정의, batch loss 분모는 parent source를 그대로 검증해 사용한다.
- 같은 dataset/seed/epoch에서 두 arm과 LR들이 같은 origin 순서를 받도록 한다.
- head/OUTPUT 초기 함수는 모든 LR/seed에서 원래 F0와 동일. 차이는 학습 순서 seed이지 별도 무작위 헤드 초기화가 아니다.
- run 내부 선택은 full-VAL macro MSE 최소, step0 포함, exact tie earliest.
- dataset/arm별 두 seed 최선 VAL 손실 평균으로 LR 하나 선택; exact tie first-listed LR.
- TEST 이전 모든 S1 선택을 봉인한다. 선택된 LR의 두 seed만 TEST-A/B 평가.
- 신규 선택 TEST 모델은 최대12개. 이를 기간 두 개 때문에 독립24개 실험이라고 세지 않는다.

감독 손실:
  R = mean_c (sum_observed (prediction-target)^2 / count_c)

사용자 평가 좌표에서 계산한다. native quantile loss를 혼용하지 않는다. normalized-head output에서 MSE를 계산하면 context scale에 따른 가중치가 달라질 수 있으므로 원래 inverse normalization 뒤 loss가 필수다.

유효배치4가 OOM이면4→2→1 microbatch만 허용하고 gradient accumulation으로 유효배치4/업데이트 수/순서를 유지한다. 전체 유효배치의 채널별 count를 공유해 summed gradient가 full batch와 같아야 한다. 각 microbatch macro mean의 단순 평균 금지.

동일 recipe가 충분한 최적화를 보장하지 않으므로 학습곡선·TRAIN/VAL 반전·비유한 gradient·선택epoch0·상한 도달을 분리해 보고한다. 성능을 보고 새 LR·더 많은 epoch·정규화·head 구조를 자동 추가하지 않는다.

## S1-6. 캐시와 사용 정보

두 arm의 prefix는 고정된다. 안전한 캐시가 있으면 사용해도 된다.
- OUTPUT: F0의 원채널 point output을 캐시.
- HEAD: 실제 head 직전 sequence_output과 역정규화에 필요한 loc/scale를 캐시. 필요한 shape/source/hash를 먼저 조사.
- feature값과 target/mask를 서로 다른 역할로 관리. 캐시 생성 함수가 미래 target를 참조하지 않는지 테스트.
- 생성은 TRAIN/VAL 입력만. 정답은 감독 loss에만 사용. TEST는 선택 봉인 후.
- 미리 정해진 최대 epoch schedule의 합집합 또는 그 이상인 TRAIN 전체 origins 중 비용이 작은 안전한 방식을 선택하고 사전 고정한다. validation 값으로 캐시 범위를 선택하지 않음.
- cache key는 raw data/preprocessing/origin/channel/prefix weights/code/dtype/norm state/output index에 연결.
- 캐시가 제공하는 더 많은 반복을 특정 arm만 이용해 추가 optimizer steps로 바꾸지 않는다.
- HEAD 내부 adapter가 바뀌어도 prefix cache는 유효하다. S2 HEAD+LoRA에서는 prefix가 바뀌므로 이 캐시는 사용 금지.
- online vs cache가 값/손실/head gradient 수준에서 일치하는지 검증한다.
- 캐시를 만들었다고 비용0으로 기록하지 않는다. 생성·전송·저장·읽기·fit을 분리하여 총비용에 포함한다.

## S1-7. TEST 이전 기술 게이트

- pinned F0 original forward 재생 및 무결성.
- OUTPUT step0 = F0; HEAD step0 = F0: 전체 canonical VAL에서 기존 tolerance로 비교.
- shape [B,48,C], channel/origin 순서, q0.5 index, 정규화 이중 적용/역변환 누락 없음.
- same origin standalone/batch/permutation parity; 미래 target 교란에 입력/예측 불변.
- F0/HEAD는 다른 채널 입력을 바꿔도 해당 채널 출력 불변. OUTPUT는 실제 channel mixing이 있으므로 이 불변을 요구하면 안 됨.
- trainable allowlist exact. HEAD에서는 native head만, OUTPUT에서는 W/beta만.
- finite loss/필요 gradient, 실제 갱신 가능한 경로; frozen prefix state 불변.
- initial state 및 optimizer/scheduler/RNG 저장/복원 일치.
- cache/online value+gradient, full/microbatch gradient 동등성.
- dropout off/eval과 학습 변수 상태를 별도 검사.
- 작은 폐기용 인스턴스의 optimizer 갱신 검사는 technical update로 계상하고 과학적 모델·캐시에 섞지 않음.
- 목적과 무관한 전수 v1-v18 감사는 하지 않음.

실패는 STOP_DEBUG. tolerance를 늘려 통과시키거나 성능 음성으로 재분류하지 않는다.

## S1-8. 평가와 불확실성

MSE primary, MAE secondary. 기존 관측 target/channel-macro를 동일 적용.
- TRAIN/VAL 및 TEST-A/TEST-B/combined를 분리.
- combined는 기간별 채널 SSE/absolute sums/count부터 합친 뒤 channel ratio→macro.
- 학습 seed별 loss 평균. seed 예측 평균 ensemble 금지.
- 자료 간 raw MSE를 평균한 총점 금지.
- 새 모델과 참조의 데이터/표본/마스크/단위/사용정보/모델선택 기회/자원 조건을 먼저 나란히 확인.
- 상대변화 =100*(candidate/reference-1); 반대로 표현할 때 분모를 다시 계산.
- 기존 paired time-block bootstrap을 재사용한다면 원점·채널·method pairing과 A/B 경계, seed aggregation을 보존. 설정은 TEST 이전 봉인.
- 기존 설정 예시는 Robin/Peacock block7 origins, Jena42, draws2000, bootstrap seed9262026이다. salt/stratification까지 실제 source를 읽어 확정한다.
- 구간에0 포함≠동등/충분/비열등. 작은 p값≠중요한 비용 절약.
- 모든 구간은 선택된 모델·기존 노출자료 조건부이며 전체 탐색·모델 선택의 불확실성을 포괄하지 않음.

S1 출력 간 차이가 났어도 정보량의 근본 한계나 원인 증명으로 확대하지 않는다.

# 공통 자원·비용 계약

## BUDGET: 수치 없는 무제한 실행 금지, 임의 예산 승계 금지

S0에서 BUDGET_REQUEST를 작성한다.
- 실행할 과학적 fits 최대수:24에서 exact reuse만큼 차감.
- maximum optimizer steps = fits ×120×(512/4), 실제 sampler를 검증한 조건부 상한.
- stage1 선정 TEST 인스턴스 최대12.
- 필요한 cache bytes=실제 h/norm tensor shape×origin 수×dtype bytes로 계산.
- GPU job wall, CPU time, 총 작업 wall, peak RAM/VRAM, disk upper estimate를 따로 제시.
- cached-fit와 online-fit의 시간 예상, 최대epoch 전체의 보수적 projection, 추정 근거의 불확실성을 분리.
- 단기 probe 횟수·시간·업데이트 수까지 계상. 새 probe GPU 예산도 먼저 확인.
- 승인된 값은 AUTHORIZATION에 기록하고 작업 시작시 확정. UNKNOWN cap 상태로 무제한 시작하지 않음.

최대epoch 기반 projection이 cap을 넘었다는 것과 실제 수렴에 그 시간이 든다는 것은 다르다. 멈춘 경우 STOP_RESOURCE로 기록하고 추정치·남은 cap·가정과 대안을 한 번에 보고한다. 과거 v18의 정지 조건을 몰래 편집하거나 HEAD 비용의 증거로 사용하지 않는다.

## COST: 저장·학습·배포를 구분

세 가지 표를 별도로 만든다.
1. 실제 총 적응비용: 특징 생성 + 캐시I/O + 모든 LR/seed 탐색 + validation/선택; shared work 중복계상 금지.
2. 같은 배치의 online 학습 연산비용: CPU input→prefix/head/output→동일 loss→backward→optimizer.step.
3. 배포 inference: CPU standardized input→전체 model→CPU contiguous H×C output.

캐시로 적응할 때 prefix를 GPU에서 내렸다면 그것은 cached head fitting의 memory다. 배포 모델 memory가 줄었다고 쓰지 않는다.
선택 checkpoint의 parameter bytes/buffer bytes/deployed bytes, trainable params, peak allocated/reserved, optimizer state memory를 기록한다.

학습비용 probe:
- training Adam 상태는 첫 step 후 생길 수 있다. optimizer.step 없이 잰 값만으로 완전한 학습 peak를 대표하지 않는다.
- 폐기용 사본에서 optimizer states까지 만든 뒤 고정 warmup/measured step block을 측정한다. 실제 업데이트 수와 폐기 경로를 기록한다.
- 선택한 과학적 checkpoint는 덮어쓰지 않으며 pre/post hash로 불변 확인.
- forward+backward-only probe가 있으면 별도 명칭으로 보존한다.

같은 회차 비교:
- S1 HEAD/OUTPUT과 실용 비용 판단에 사용할 F0·기존 MSE-LoRA·LEVEL만 포함. 다른 기존 모델 cost는 동일 회차 없으면 context only.
- 같은 장치·FP32·TF32 off·thread·입출력·warmup·동기화.
- inference B1/B4; 각 자료 VAL 첫24 시간순 원점; warmup10, 전체24-origin passes10, 순서회전3 blocks는 기존 최근 회차와 연결 가능한 기본안이다. 실제 data contract에서 가능함을 확인해 봉인한다.
- full raw block, sentinel drift, 시간과 처리량 단위, 통계 집계를 기록.
- block 범위는 신뢰구간이 아니다. 측정 중앙값의 우위를 모든 하드웨어의 속도 우위로 일반화하지 않음.
- OOM 자체를 기록하고 primary B4를 임의 B2로 바꾸지 않음. 기술적 microbatch 학습과 inference 조건을 혼동하지 않음.
- compiled/quantized/CPU offload/gradient checkpointing 새 최적화는 첫 회차에 한 arm만 적용하지 않음.

# S1 종료 판정과 다음 단계 게이트

필수 산출물은 성공 선언이 아니라 다음 질문에 대한 답이다.
- OUTPUT/HEAD 각각 F0를 개선했나? 원점/기간/seed별 방향은 무엇인가?
- HEAD와 OUTPUT 중 같은 점수·비용에서 어떤 절충이 생겼나?
- 기존 MSE-LoRA와의 격차 및 비용은 어떠한가? 비교 조건 차이는 무엇인가?
- 데이터에 따라 적응 경로 선택이 달라질 가능성이 있나?
- 현재 손실은 원래 함수·정규화·최적화 오류로 설명되는가?

S1을 성공/실패 토큰 하나로 과장하지 않는다. 가능한 요약은 STANDARD_READOUT_HELPFUL / NO_CLEAR_READOUT_GAIN / MIXED_READOUT_RESULT / OPTIMIZATION_UNRESOLVED 정도의 기술적 설명이며, 새 문제·논문 성립 판정과 다르다.

S2 권고는 VAL·비용과 충분히 검증된 학습 상태를 주 근거로 한다. 가능하면 S1 TEST 전에 continuation 필요성과 protocol 초안을 기록한다. 이미 TEST를 본 뒤 정한 방향은 사후 개발 결정으로 공개한다. 불리한 자료만 빼거나 좋아 보이는 자료만 S2에 넣지 않는다. 기본 범위는 같은3자료 모두이며 축소는 별도 목적·예산 합의가 필요하다.

S1만으로 “HEAD가 LoRA와 동등”, “내부 적응이 불필요/필수”, “표현에 정보가 없음”, “진단 선택기 성공”, “신규 PEFT 확보”는 주장하지 않는다.

# S2. 조건부 후속 — 같은 HEAD 부모에서 내부 적응의 추가 가치

S2는 이 문서의 첫 자동 실행 범위가 아니다. S1 결과, 구체적 continuation steps/예산에 대한 승인 후 진행한다.

## 같은 부모·초기 함수

dataset/seed별 S1 NATIVE_HEAD 선택 checkpoint를 바이트 동일하게 두 갈래 복사.
- HEAD_CONTINUE: head만 추가 적응.
- HEAD_PLUS_LORA: 같은 head + 내부 q/v LoRA 추가 적응.

기존 helper의 r8/alpha16/dropout0/qv/biasnone을 출발점으로 검토하고 실제 부착 위치를 enumeration한다. LoRA의 초기 수정분은0. 두 arm은 초기 예측·native head weights가 같아야 한다.

주의: get_peft_model은 head까지 freeze할 수 있으므로 래핑 후 HEAD_PLUS_LORA의 native head를 다시 명시적으로 trainable로 설정하고 allowlist 검사. LoRA+HEAD의 trainable 수는 기존294,912뿐이 아니라 head params를 포함한다.

## 추가 학습량 대조

- 두 arm 모두 새 optimizer/scheduler를 동일 규칙으로 시작하거나 같은 부모 상태로 시작한다. 어느 규칙인지 TEST 전에 결정하고 혼용하지 않는다.
- 부모 selected epoch 이후 재현 가능한 동일 sample suffix를 사용한다. 조기정지로 가변인 전반 이력을 기록한다.
- 동일 effective batch, optimizer update budget, 동일 LR 탐색 기회. 공통 fixed update checkpoints를 저장하여 같은 update 수 비교도 제공한다.
- 각 arm의 VAL-best early-stop 결과와 fixed-update 결과를 구분한다.
- 시간 예산 비교는 또 다른 estimand이므로 같은 update 수≠같은 시간이라 명시한다. equal-time claim에는 사전 시간예산과 trajectory snapshots 필요.
- HEAD_CONTINUE는 안전한 고정 prefix cache 재사용 가능. HEAD_PLUS_LORA는 prefix가 변하므로 그 cache를 쓰지 못함.
- 추가 LR배수/LoRA rank/모듈탐색 자동 확장 금지.

과학적 최대 규모안:
  2 arms ×3자료 ×2LR ×2seeds=24 fits.
선택된 신규 TEST 모델 최대12개. S1 비용은 두 경로의 공통 과거 비용이며 총 adaptation 비용에 한 번 포함한다.

측정:
  Delta_internal = R(HEAD_CONTINUE) - R(HEAD_PLUS_LORA)
양수는 해당 조건에서 내부 LoRA 추가 arm의 오차가 낮다는 뜻. 같은 원점·mask·metric·선택 규칙에서만 계산한다.

통제해도 남는 한계: 추가 trainable capacity와 비선형 최적화가 함께 바뀐다. 모든 head 클래스의 최적 오차를 구한 것이 아니며 내부 표현 정보부족의 증명도 아니다.

S2 이후:
- 내부 적응의 이득/비용이 실제로 달라 선택할 문제를 만들면 S3 계획 검토.
- 한 단순 경로로 목적을 달성하면 진단기 개발 보류.
- 차이가 불확실하면 동등이 아니라 증거 부족으로 기록.
- 정상 음성 결과를 새로운 hyperparameter 탐색의 자동 허가로 쓰지 않음.

# S3/S4. 연구 방향만 명세 — 별도 승인 전 구현·실험 금지

## S3: 저비용 진단 후보

가설: F0 오차, 짧은 head 적응으로 줄어든 오차, head 이후의 미래거리/채널/기간별 잔여오차가 이후 Delta_internal에 추가 정보를 줄 수 있다.

아직 수식·회귀기·threshold를 확정하지 않는다. 자료별 학습된 최종 HEAD의 TEST 오차를 입력으로 사용해 같은 TEST의 LoRA 이득을 맞히는 순환 검증은 금지한다.

현재3자료/2seed/겹친 원점은 독립 수백 도메인이 아니다. 이들로 큰 selector를 학습하고 일반화했다고 주장하지 않는다. S3 전에 task unit과 새로운 held-out 대상·백본·시간 구간에 대한 별도 데이터 계약이 필요하다.

시간별 의사결정:
  과거 support fit -> 그 뒤 diagnostic/selection interval -> 더 뒤 outer evaluation

각 예측 시점에서 사용 가능한 정답만 진단에 사용한다. 과거 진단 원점에 전체 TRAIN로 학습된 head·scaler를 가져가면 뒤쪽 정보가 섞일 수 있으므로 prefix별 적합·통계·가용성을 따로 검증한다. forecast label intervals가 다음 선택/평가 구간을 넘어가지 않도록 purge한다. 과거 context 중첩 자체와 미래 label 누출을 구분한다.

## S4: 진단의 실제 선택 가치

새 데이터·백본 범위와 budget을 승인한 뒤에만 다음 표준 선택기들과 비교한다.
- 항상 HEAD / 항상 내부 LoRA / no-adaptation F0 옵션.
- 과거 개발 자료에서 정한 공통 best policy.
- 같은 총예산의 짧은 후보 파일럿·successive halving 계열.
- 데이터 복잡성 지표(Time-PEFT 원리)의 정확히 검증된 구현.
- frozen feature transferability(LogME)의 회귀·결측·시간 의존 조건을 공개한 적용.

Time-PEFT/LogME 재현과 변경 적용을 구분하며 원문 설정을 임의 대체하지 않는다. full AutoPEFT 구현을 무조건 돌리기보다 같은 문제의 최소 표준 탐색과의 비교를 먼저 정의한다.

선택에 쓴 원점과 결과 평가 원점을 분리한다. 진단비·후보 탐색비·feature generation·선택된 fit/continuation을 모두 합산한다. 짧은 파일럿의 상태를 이어쓴 경우 중복계상하지 않는다. 연구용 모든-candidate 학습비와 배포 decision path 비용도 별도로 기록한다.

동일 TEST를 보고 가장 좋은 모델을 고른 사후 best는 attainable policy가 아니라 oracle comparator(평가용 참조)로만 쓴다. 관측 성능差만으로 oracle보다 열등한 선택을 고쳐 쓰지 않는다.

원수치 상관이나 AUROC 하나가 아니라 선택 후 실제 forecast risk, decision 비용, 총 적응비용, fallback 오류, 비용대비 단순 정책 이득을 평가한다. 허용 오차/비열등 margin/자원예산은 결과 전에 정해야 하며 현재 임의1%/2배 합격선을 만들지 않는다.

# 공통 반론과 주장 한계

1. “HEAD부터 적응하고 LoRA”는 기존 순서다. 신규성으로 쓰지 않는다.
2. 새 지표가 LogME/복잡성/짧은 pilot와 같은 결정을 더 비싸게 내리면 기여가 부족하다.
3. 단순 head 적합 실패는 representation insufficiency의 증명이 아니다.
4. S1 OUTPUT는 cross-channel, HEAD는 공유 univariate다. 결과를 순수 feature availability 효과로 해석하지 않는다.
5. S1/S2의 기존 TEST는 노출되어 있으므로 보호확증/일반선택 규칙 검증으로 쓰지 않는다.
6. HEAD가 다른 head-class보다 최적이라는 증거는 없다. 무리한 반론을 만들기 위해 약한 linear probe로 native head를 바꾸지 않는다.
7. 수치 개선·PASS·실험수·복잡함만으로 논문/신규성 점수를 만들지 않는다.
8. 모든 것을 해도 비교연구로만 남을 수 있다. 그것을 사용자의 동의 없이 새 분석 논문 목표로 바꾸지 않는다.

# 산출물·검산·게시

다음은 신규 제안 경로다. 기존 경로와 충돌하면 새 이름을 기록하되 기존 연구를 덮어쓰지 않는다.
  research/tsfm_readout_adaptation_screen_20261007/
  .cache/tsfm_readout_adaptation_screen_20261007/

첫 회차 최소 산출물(같은 정보의 중복 문서 생성은 줄인다):
- REQUEST_GOAL_PLAN.md, PURPOSE.md, PLAN.md, STATUS.md
- authorization.json / budget_estimate.json / ledger.json
- reuse_manifest.json, source_binding.json, prior_art_map.md
- protocol.json, data_contract.json, architecture_boundary.json, preflight_checks.json
- stage1 model/train/evaluate/cost의 최소 재사용 코드
- runs/<id>/result.json, curves, selected checkpoint/prediction receipts
- selection.json, selection_seal.json, exposure.json, prediction_manifest.json
- accuracy_models.csv, accuracy_channels.csv, paired_comparisons.csv
- training_work.csv, training_probe_rows.json, inference_cost_rows.json
- cache_manifest.json, cost_summary.json, final_checks.json
- FINAL_REPORT_KO.md, CLAIM_EVIDENCE.md, NEXT_STAGE_DECISION.md
- 필요하면 PRESENTATION_PATCH.md(문장 제안만, 원본 PPT/PDF 편집 금지)

모든 파일을 첫날 dummy로 만들지 않는다. 미실행 산출물은 gate_prevented/unauthorized로 표시한다. 원본 데이터·모델·대형 cache/checkpoint는 Git에 올리지 않는다. 개인 절대경로·연락정보를 공개 사본에서 제거하되 source와 sanitized copy 해시를 구분한다.

최종 검산:
- 정확한 checkpoint와 local code binding.
- F0 step0 parity / trainable allowlist / frozen hash / actual non-step0 updates.
- 캐시/online·full/microbatch parity 및 정규화·마스크 분모.
- joint VAL choice before TEST, loss average≠ensemble.
- 지표 계산·relative denominator·시간원점/기간 pairing 재생.
- 비용 측정의 전체 scope·optimizer states·cache 비용·raw blocks.
- completed-negative와 technical/resource block 구분.
- stage1_complete와 stage2_authorized/executed를 별도 boolean으로 기록.
- verification PASS≠새 방법 검증≠독립 재현≠학술 최초성.

과학적 결과를 미리 채워넣지 않는다. 최종 상태는 READY_FOR_BUDGET_APPROVAL / RUNNING_STAGE1 / STAGE1_DONE / STOP_DEBUG / STOP_RESOURCE 등 실제 상태로 쓴다.

관련 결과의 commit/push 권한이 확인되면 무관한 dirty files를 제외해 ordinary commit/push한다. force-push/reset/rebase/작업삭제 금지. 브랜치 보호·권한 때문에 push 불가하면 로컬 SHA와 상태를 보고한다. push한 경우 live remote SHA를 확인해서만 게시 완료라고 한다.

FINAL_REPORT_KO.md 순서:
1. 문제: 표준 출력/헤드 적응으로 충분한가?
2. 왜 중요한가: 새 모듈 이전에 단순 대안 누락을 제거.
3. 실제 학습 경계·입력정보·비교의 혼입.
4. 정확도·선택·기간/seed·실행 정상성.
5. total fit/cached fit/online train/deployment cost의 구분.
6. 표준 대안을 넘어서 남은 질문과 가장 강한 반론.
7. S2 진행 여부·근거·불확실성·필요 예산(자동 실행 금지).

CLI 최종 보고 형식:
STATUS:
local_sha / verified_remote_sha:
authorized_stage / executed_stage:
new_scientific_fits / technical_probe_updates / selected_TEST_instances:
reused_artifacts:
Robin/Peacock/Jena OUTPUT / HEAD / F0 / historical MSE-LoRA MSE and MAE:
paired differences and uncertainty:
cache creation / fitting / selection / online training / inference costs:
what_this_does_not_establish:
S2_recommendation: PROPOSE / DEFER / NOT_JUSTIFIED_YET
remaining_decision_gaps: 최대3개

# 확인한 출처와 열람 범위

[공식 웹 확인: 이번 문서 작성 시 metadata/초록 및 제공된 본문 범위]
- LogME(2021/ICML): https://proceedings.mlr.press/v139/you21b.html
- AutoPEFT(2024/TACL): https://aclanthology.org/2024.tacl-1.29/
- LP-FT 관련 논문(2022/ICLR): https://arxiv.org/abs/2202.10054
- Time-PEFT(2026/ICML): https://proceedings.mlr.press/v306/na26b.html
- TRACE(2026/Neurocomputing): https://www.sciencedirect.com/science/article/pii/S0925231225027705
- AdaPTS(2025/ICML): https://proceedings.mlr.press/v267/benechehab25a.html
- Hyperband(2018/JMLR): https://jmlr.org/papers/v18/16-558.html

[코드 읽기: 실제 실험에서는 local pinned library hash 우선]
- https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos_bolt.py
  이번 읽은 blob sha: db44821f4c657fa1da1219d8ff7f996594e375cc.
  main을 영구 실행 revision으로 쓰지 말고 clone의 commit과 설치파일 SHA256를 실제 기록.
- https://github.com/CanelE452/hierarchical-tsfm-peft/blob/8646e54630add0777c90686b5a8e4bef20c7dcea/src/hier_peft/lora.py
- https://github.com/CanelE452/hierarchical-tsfm-peft/blob/8646e54630add0777c90686b5a8e4bef20c7dcea/research/level_matched_learned_ed_v18_20261006/model_v18.py
- https://github.com/CanelE452/hierarchical-tsfm-peft/blob/8646e54630add0777c90686b5a8e4bef20c7dcea/research/level_matched_learned_ed_v18_20261006/STATUS.md

위 선행의 성공은 새 진단의 성공 근거가 아니다. 실제 원문/코드를 읽지 못한 세부사항은 미확인으로 기록하고 계획의 빈칸을 임의로 채우지 않는다.
