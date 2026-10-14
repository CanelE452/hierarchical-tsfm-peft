# TS2Vec 정상 크기 표현과 새로운 복원기의 직접 실험

[확인] encoder와 학습 알고리즘은 [공식 TS2Vec](https://github.com/zhihanyue/ts2vec/tree/b0088e14a99706c05451316dc6db8d3da9351163)의 revision `b0088e14a99706c05451316dc6db8d3da9351163`을 사용한다. 기존 vendor의 encoder, dilated convolution, hierarchical contrast, crop/mask, AdamW, SWA와 save/load 소스 7개는 파일 SHA256을 확인하고 **수정 없이** 재사용한다. 임시 import alias는 기존 `models`/`utils` 모듈을 복원하므로 다른 모델과 섞이지 않는다.

[확인] `hidden_dims=64`, `depth=10`이며 `output_dims=320`은 공식 API 기본값이다. 모든 원논문 실험의 최적 설정이라고 주장하지 않는다. d16은 같은 hidden/depth에서 출력 차원만 줄인 이번 적용이며 교수님이 정한 압축률은 아니다. d320은 Robin C17/Jena C21보다 커서 표현 확장 대조다. 이전의 작은 d5/6 또는 hidden16/depth3 실행을 정상 크기 실행으로 다시 이름 붙이지 않는다.

```text
encoder                online parameters   separate SWA parameters
Robin C17, d16                  253104                  253104
Robin C17, d320                 638272                  638272
Jena C21, d16                   253360                  253360
Jena C21, d320                  638528                  638528
```

학습 입력은 `[1,T_train,C]`의 원래 연속 TRAIN 구간이다. 관측되지 않은 입력은 공식 NaN masking에 맡기며, overlapping 512-window 전체를 independent negative instance로 취급하지 않는다. [공식 CLI](https://github.com/zhihanyue/ts2vec/blob/b0088e14a99706c05451316dc6db8d3da9351163/train.py)의 초기 설정 batch8, lr0.001, max_train_length3000을 사용한다. `fit()`의 실제 기본 업데이트 수는 입력 scalar 수가 100000 이하이면 200, 더 크면 600이다. 긴 입력에 대한 공식 split/crop 및 SWA는 그대로 사용한다. 실제 데이터별 scalar 수·업데이트 수·batch 조정은 실행 기록에 남긴다.

[확인·실제 적용] Robin의 TRAIN `[1,1992,17]`은 max_train_length3000보다 짧아 공식 fit의 분할 후 instance가1개다. 설정 batch8과 달리 DataLoader의 실제 batch는 `min(8,1)=1`이다. 수정 없는 [공식 loss](https://github.com/zhihanyue/ts2vec/blob/b0088e14a99706c05451316dc6db8d3da9351163/models/losses.py)는 B1의 instance contrastive 항을0으로 반환한다. 따라서 Robin d16/d320·두 seed는 정상 크기 encoder로200 update·SWA201을 수행했지만 **실제 비영 contrast는 hierarchical temporal 항**이다. 원형 알고리즘을 바꾼 것은 아니나 두 contrast 항이 모두 작동한 학습으로 표시하지 않는다.

[확인·실제 적용] Jena는 TRAIN `[1,26207,21]`을 공식 max-length 규칙으로8개 contiguous instance로 나눈다. 원래 B8 및 cache-only B8 시도가 RSS 상한을 넘어, canonical4조건은 **새 경로에서 B4**로 실행했다. hidden64/depth10, d16/d320, max_train_length3000,600 update·SWA601은 유지했고 각각300 epoch였다. B4는 한 update의 instance negative 수를 바꾸며 같은600 update에서 instance 노출 횟수도 B8의 절반이다. crop 시간점 노출까지 정확히 같은 조건은 아니다. 따라서 정상 encoder/공식 알고리즘과 **공식 기본 batch의 정확한 재현**을 구분한다. 실패·cache-only 시도와 최초 Robin epoch0 누락 pilot은 보존하고 canonical8개 집계에서 제외한다.

## TSFM 연결에 추가한 부분

TS2Vec 원형에는 reconstruction decoder가 없다. 새 decoder는 시점별 `d→128→ReLU→C`이고 출력은 linear다. 이는 원논문의 구성이나 기존 선형 readout으로 표시하지 않는다. frozen E가 각 관측 과거 512-window에서 산출한 **offline full-context** 표현을 사용해 TRAIN observed 입력을 복원하도록 D를 학습한다. VAL 역시 같은 window export 규칙이다. 과거 window 내부에서는 양쪽 temporal context를 사용하지만 forecast origin 이후 실제 target을 읽지는 않는다.

TSFM은 그 표현의 d개 좌표를 각각 time series 행으로 받는다. `[B,512,d]→[B*d,512]`이며 좌표를 시간 축에 이어 붙이지 않는다. frozen F의 실제 미래 48개 출력을 같은 D로 복원하는 transfer 경로와, actual F TRAIN predictions로 D만 따로 적응한 calibration 경로를 구분한다. calibration은 미래 label supervision을 추가하므로 비지도 표현학습의 순수 이득으로 표시하지 않는다. latent 마지막값 유지도 같은 두 decoder에서 부품 교체 진단으로 평가한다.

## 원형 forecasting 사용 대조

[공식 forecasting 코드](https://github.com/zhihanyue/ts2vec/blob/b0088e14a99706c05451316dc6db8d3da9351163/tasks/forecasting.py)는 `causal=True, sliding_length=1, sliding_padding=200`으로 representation을 산출한다. 마지막 표현에서 다음 H개 원채널을 ridge로 직접 예측하는 대조에 이 규칙을 사용한다. main E→F→D의 offline representation과 다른 export 경로라는 점을 결과에 기록한다.

마지막 표현만 필요하므로 `export_last_causal`은 공식 rolling loop의 마지막 201점 window와 `[200:201]` output slice만 계산한다. 짧은 과거는 같은 NaN left padding을 적용한다. full rolling을 실행한 CPU fixture에서 d16/320, L17/213의 최대 차이가 1.79e-7 이하였으며, CUDA 실제 경로에서도 별도 검산한다. 양방향 convolution을 그대로 쓰므로 이 규칙을 네트워크 자체의 causal convolution이라고 부르지 않는다.

[공식 fit_ridge](https://github.com/zhihanyue/ts2vec/blob/b0088e14a99706c05451316dc6db8d3da9351163/tasks/_eval_protocols.py)는 alpha 13개 `0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000`을 사용한다. intercept를 포함하고 validation의 RMSE+MAE를 최소화한 첫 alpha를 선택한 뒤 TRAIN으로 다시 적합한다. 새 데이터에서는 원래 관측 mask를 적용하고 동일 outer VAL을 사용한다. 사용된 TRAIN origins의 stride, target mask 처리와 규제 선택은 원논문의 정확한 benchmark 재현과 구분한다.

[확인·비교 범위] native ridge는 Robin/Jena의 모든 legal TRAIN origins1433/25648개(stride1, 공통 최소 과거512)를 사용한다. F-decoder calibration은12/201개(stride128)를 사용한다. 중첩 origin/window 수를 독립 표본 수로 세지는 않는다. native의 causal last201 표현·선형 multi-horizon head와 main의 offline512 표현·실제 F 출력·시점별 비선형 D는 **export, readout, supervised label-window 노출이 함께 다르다**. 같은 개발 VAL에서 native ridge가 낫다는 실제 수치는 유효하지만 F의 필요성이나 F 연결만의 실패 원인을 분리한 ablation은 아니다. 후속 matched export/readout supervision 비교는 별도 연구 질문이며 이번에 추가 실행하지 않았다.

공식 `save/load`는 SWA **inference state**를 보존한다. online network, optimizer 또는 iteration counter를 자동으로 복구하지 않는다. 실제 학습 업데이트·학습 곡선·초기 seed 및 checkpoint provenance는 별도 저장한다. 단순 load 후 새로 학습하는 것을 optimizer resume이라고 표시하지 않는다.

## 현재 검증 범위

[확인] `TS2_CPU_VERIFICATION_1791721997040036200.json`: synthetic CPU fixture 26개 PASS. 정상 크기, source SHA, 실제 3-update 학습과 SWA 변경, NaN 입력 유한 출력, 공식 save/load bitwise parity, efficient last causal export equivalence, past-only export의 target perturbation 불변성, online/SWA freeze를 확인했다. GPU job·실제 데이터 fit·보호된 TEST 접근은 모두 0이다. 이 검증 자체를 연구 성능 실험 완료로 세지 않는다. 실제 E/F/D 연결·학습·예측 수치는 `runner_ts2.py`가 별도 산출한다.

[확인] 위 CPU fixture와 별개로 현재 canonical TS2 8개에 실제 학습·추론 완료 JSON이 있다. [최종16조건 진단](METHOD_FAILURE_DIAGNOSIS_KO.md)은 원수치·결과 SHA와 해석 범위를 연결한다. 전체 예측·비용의 독립 검산 완료 여부는 별도 검산 보고서에서 확인한다.

공식 CLI/forecasting/ridge 파일은 `vendor_reference/ts2vec/`에 같은 revision의 수정 없는 참고 파일로 저장했고 `provenance.json`에 SHA256과 직접 source URL을 남겼다. 이 참고 파일의 TEST 평가 함수는 실행하지 않는다.
