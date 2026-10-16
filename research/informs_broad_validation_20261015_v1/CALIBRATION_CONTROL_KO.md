# 같은 M 정답을 사용하는 강한 A+M 대조군

## 목적과 공정성의 범위

보정기 H와 α는 M_cal 정답을 추가로 사용한다. 이를 A까지만 학습한 G/LoRA와 비교하면 추가 지도정보의 효과가 결합 구조의 효과처럼 보일 수 있다. `G_REFIT_APLUSM`, `LORA_REFIT_APLUSM`은 같은 M 정답을 원 예측기에 직접 사용하는 강한 실제 사용 대안이다. 결합기 개선이 단순한 추가 학습으로 설명되는지를 확인하기 위한 대조다.

이 대조는 정보 범위를 맞추는 것이며, 구조·parameter 수·GPU 시간·체크포인트 후보 수까지 자동으로 동일하다는 뜻이 아니다. 실제 노출·선정·비용을 따로 보고해야 한다. V는 각 방법의 선택 구간이므로 이 대조의 V 점수도 독립 일반화 검증이 아니다. 선택을 잠근 뒤 T에서 평가해야 한다.

## API와 실제 적합 절차

```python
fit_control.fit_control(spec, task, cache_root, public_root, foundation, deadline)
```

허용 family는 `DIRECT_G`, `LORA`이며 `role="AplusM_control"`을 고정한다. 로컬 데이터와 원 frozen foundation은 호출자가 제공한다. 기존 G/adapter의 학습 weights나 부모 예측 캐시는 입력으로 받지 않는다.

1. `fit_base.create_model(spec, task, A_stop.end, foundation)`을 호출해 **A60 통계와 같은 seed의 새 모델**을 만든다. G와 LoRA adapter를 새로 초기화한다. 원 foundation의 pretrained weights는 frozen 상태로 재사용한다.
2. Normalization의 `fit_end=A_stop.end`는 유지하고, `train_end=M_cal.end`를 명시한다. 학습·평가 모두 A60 척도로 계산한다. M 정답을 추가했다고 scaler/PCA를 M까지 다시 적합하지 않는다.
3. 공유 학습 engine은 0부터 M_cal 끝까지의 모든 합법 origin을 stride1 순열로 사용한다. 학습 origin의 모든 H target은 이 prefix 안에 있다. A/M 경계를 가로지르는 창도 같은 연속 prefix의 합법 창이다.
4. CP0와 요청한 `.25/1/3` 등 후보를 **V_select의 strideH**에서 평가하고, 채널 macro normalized MSE가 가장 작은 CP를 선택한다. V에서 backpropagation하거나 T를 읽지 않는다. 동점은 먼저 평가한 CP를 유지한다.
5. 선택된 weights를 그대로 V/T용 artifact로 고정한다. A까지만의 search/refit을 수행한 것처럼 결과 구조를 만들지 않는다. 별도 `control_summary`, `selected_V_mse`, `checkpoint_selection_split`을 저장한다.

AdamW·weight decay0·clip1·전체 origin 관측수로 보정하는 macro loss·micro4/effective16은 기본 실행기와 같다. F를 통한 LoRA gradient는 유지하며 F 자체는 고정한다. 정상·예외 종료 시 hook과 원 F를 복구한다.

## 예측 역할과 시간 경계

`prediction_cache.predict_parent(..., role="AplusM_control")`로만 읽을 수 있다. V/T에 한정하고 `normalization.train_end <= M_cal.end`를 확인한다. M_cal에서 이 모델의 예측을 보정기 훈련 입력으로 만들면 자기 학습 target을 재사용하게 되므로 거부한다. 기본 calibration parent로 역할을 바꾸는 것도 거부한다.

Metadata의 `parent_fit_end`는 label 학습 경계인 train_end를 뜻하고, `parent_normalization_fit_end`는 scaler를 적합한 A60 경계를 따로 기록한다. 두 값의 의미를 섞지 않는다. T는 별도의 선택 잠금·정확한 artifact SHA 확인이 있어야 열린다.

## 산출물과 재개

Private `runs/{run_id}/control/`에는 immutable optimizer/RNG/sampler checkpoint를 저장한다. 최종 `selected_bundle.pt`에는 spec/model/normalization/contract/model_state_sha와 control metadata를 저장한다. 공개 출력은 `CONTROL_SOURCE_CONTRACT.json`, `CONTROL_FIT_RECEIPT.json`, `control/CURVE.csv`, `control/RECEIPT.json`과 stage events다. 개인 cache 절대경로나 실제 label 배열을 공개하지 않는다.

공유 stage engine은 자신의 source hash를 매 CP에서 검사한다. 그 checkpoint contract에 `control_source_sha`도 묶고, outer helper는 실행 전후 전체 control+engine source를 확인한다. 최종 bundle은 결합 source SHA와 별도의 engine SHA를 모두 가진다. 기존 `fit_base.py`를 수정하지 않아 진행 중인 base 적합의 과학 소스를 바꾸지 않는다.

같은 ID에 source/config/data/F hash가 바뀌면 거부한다. 완료 artifact의 SHA를 검산한 뒤에만 재사용한다. 제한 epoch schedule을 늘리는 것은 기존 계약 변경이므로 새 ID/상위의 명시 확장 절차가 필요하다.

공유 engine의 private checkpoint에 있는 legacy 필드 `astop_mse`는 이 호출에서는 **V_select**의 값이다. 공개 curve는 `V_mse`, summary는 `selected_V_mse`로 이름을 바꾸고 선택 split을 명시한다. A_stop에서 고른 결과라고 해석하면 안 된다.

`COMPLETE_VALID`는 요청한 적합·선정 artifact 완성을 뜻한다. `.25`만 실행하면 `LIMITED_SCREEN`이다. 마지막 CP가 best이거나 마지막 두 점이 감소한 상태는 receipt에 별도 표시하며 수렴/방법 실패를 단정하지 않는다. Budget 중단 후 weights/optimizer/다음 batch/RNG 재개는 검증했다. 다만 공유 engine의 budget 조기 반환은 중단 segment의 clipping/norm 진단을 전부 돌려주지 않으므로, 이 경로의 누적 clipping 진단까지 exact하다고 주장하지 않는다.

## CPU 합성 검증

최초 3 RED 실패 후 구현했고 최종 4 unittest가 통과했다.

- G의 실제 A+M 학습 노출, V 선택, A60 mean의 FP64 계산→FP32 결과 exact 일치, 새 초기 weights SHA.
- 완성 artifact 재사용과 epoch schedule 변경 거부.
- Budget 중단·재개와 무중단 완료의 최종 model-state SHA 일치.
- 생성 artifact의 V 예측·M 부모 거부, LoRA 실제 toy update 후 F hash/gradient/hook 복구와 V 추론 row 수.

이 검사는 합성 CPU 데이터와 작은 q/v 선형 모형만 사용했다. 실제 자료, 실제 Bolt, GPU, TEST 또는 Git 작업을 수행하지 않았다. 실제 모델의 개선·학습 충분성·독립 T 성능은 상위 캠페인 실행 결과로 판단해야 한다.
