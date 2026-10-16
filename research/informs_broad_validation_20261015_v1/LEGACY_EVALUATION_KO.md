# E0의 구 B 저장 가중치 재추론·독립 검산

이 경로는 E0의 새 학습으로 만들어진 체크포인트를 다시 로드하여 구 B 전체를 실제로 예측하고, 저장한 예측을 FP64로 독립 검산한다. 학습을 다시 실행하지 않는다. 구 B는 이미 학습 체크포인트·설계 선택에 사용한 개발 자료다. `historically_unseen=false`이며 E1의 새로운 `V_select` 또는 `T_final` 성능으로 합산하지 않는다. 평가 코드를 구현하고 CPU 검사를 통과한 사실은 실제 E0 GPU 재추론 완료를 뜻하지 않는다.

## API와 실행 주체

`inventory_checkpoints(asset_root, cache_root, public_root=None, legacy_ids=None)`는 데이터 배열과 F를 로드하지 않고 작은 학습 상태·공개 receipt를 검사한다. 완료 endpoint의 last 파일과 selected 파일 SHA, 내부 tensor SHA, carry의 체크포인트 목록과 receipt가 일치해야 한다. budget carry도 공개 budget receipt, private sidecar와 내부 contract를 대조한다. 구 과학 소스 4개, 구 통계, continuation 실행 소스, 구 2400 receipt/last, 실제 foundation tensor hash의 pin을 유지한다.

영수증의 prefix 검색 결과를 그대로 같은 ID로 취급하지 않는다. fits의 정식 파일명은 `<id>_<endpoint>.json`, budgets는 `<id>_<endpoint>_<숫자 timestamp>.json`이다. 각 JSON의 id·고정 endpoint·정식 파일명이 먼저 일치해야 하며, 그 뒤 정확히 다른 ID인 정상 이웃만 제외한다. 예를 들어 `primary_jena_B0_r5_s92601`은 `_G_s92601` 등 긴 이웃 ID의 영수증을 자신의 것으로 검사하지 않는다. 반대로 own 파일명에 다른 id를 넣거나 own id를 이웃 파일명에 넣은 경우, endpoint·내용·파일명이 다른 경우, 손상된 JSON은 계속 INVENTORY_FAILED다. 초기화되지 않은 budget도 이 identity 검사보다 먼저 건너뛰지 않는다. 자신의 source/statistics/F/legacy/carry/last/selected tensor·파일 SHA 검증은 그대로 유지한다.

`evaluate_legacy(asset_root, cache_root, public_root=None, legacy_ids=None, *, lease=None, deadline=None, batch_size=8, row_microbatch=None)`가 실제 실행 API다. root가 중앙 GPU lease를 유지한 새 subprocess에서 호출한다. lease를 생략하면 중앙 owner가 서명한 child 환경을 검증한다. 기존 일반 `runtime`, `data`, `campaign`, `models` 모듈이 이미 import된 프로세스에서는 구 loader가 거부한다. 기본 batch 8은 구 `assess → infer`와 같다. 다른 microbatch 설정은 실제 기록하며 수치 일치가 보장된다고 미리 주장하지 않는다.

중앙 예약을 확보하고 인증된 child 환경을 전달한 다음에만 다음 명령을 사용한다. 아래 명령 자체가 별도의 GPU 예산을 발급하지 않는다.

```powershell
python -X utf8 -B legacy_evaluation.py run `
  --asset-root $env:INFORMS_ASSET_ROOT `
  --cache-root $env:INFORMS_CACHE_ROOT `
  --execute-evaluation
```

`--id <legacy_id>`를 여러 번 주어 일부 경로만 평가할 수 있다. `inventory` 모드는 `--execute-evaluation` 없이 사용할 수 있다. `--result-json`은 새 파일에만 기록한다. 이 모듈은 기존 E0 12개 unresolved ledger 또는 28개 큐를 변경하지 않는다.

## 어떤 가중치를 검사하는가

새 private `legacy_e0/checkpoints/<id>/cp*.pt` 중 update 2800 이상인 모든 저장 상태가 대상이다. 완료 endpoint 또는 budget carry가 선택한 구 2400 이하 CP가 있으면 그 상태도 추가한다. 각 CP 파일 SHA와 tensor SHA를 검증한다. CP 한 개가 잘못되면 해당 CP를 실패로 남기고 같은 경로의 다른 CP를 계속한다. last/selected 또는 전체 receipt의 계약이 잘못되면 그 경로의 provenance가 성립하지 않으므로 그 경로를 중단하고 다른 ID를 계속한다.

기록된 B 점수가 있는 CP는 `BOUND_RECORDED_B_SCORE`로 표시한다. CP 파일은 있지만 B 점수가 carry/receipt에 아직 영속화되지 않은 경우도 저장 가중치로 추론할 수 있으나 `SAVED_STATE_WITHOUT_PERSISTED_B_SCORE`로 분리한다. 이 경우 기존 곡선과 일치했다고 주장하지 않는다. 새 endpoint가 추가되어도 이미 검증한 CP의 모델·데이터 identity는 불필요하게 바뀌지 않도록, 변할 수 있는 endpoint provenance 전체는 실행 registry에 보관하고 CP identity에는 고정 SHA를 넣는다.

## 구 B 계약과 정보 접근

고정된 구 모듈을 `legacy_continuation.load_legacy`로 import하고 `model_spec`, `build_legacy_model`로 원래 모델을 복원한다. 실제 구 API는 `Windows.truth(origins)`이며 `target('B')`가 아니다. 구 `make_fold`의 B 원점, H 간격과 마지막 endpoint 원점을 그대로 쓴다. 마지막 두 원점의 표적이 겹칠 수 있다는 구 계약도 유지한다. 새 비중첩 코호트로 바꾸거나 독립 표본이라고 해석하지 않는다.

평가 loader는 pinned safe archive에서 구 FIT의 A+B 끝까지 즉시 slice한다. legacy-normalized 자료는 slice 후 원래의 FP32 `x*std+mean` 역변환을 한다. 구 `Windows`와 구 A-only 통계를 쓰고, 저장된 `FOLD_STATISTICS.json`의 A mean/std와 정확히 일치하는지 검사한다. C 원점 목록은 비워 두며 C 표적을 추출하거나 평가하지 않는다. 보호된 원 TEST 파일은 열지 않는다.

이는 **파일 바이트 수준의 C 미읽기 증명은 아니다**. 안전 NPZ의 필요한 압축 멤버는 전체 해제되고, 소스 SHA 검증도 파일 전체를 읽는다. 그 멤버에는 구 C나 과거 TUNE/VAL 행이 포함될 수 있다. 계산·통계·GPU 적재는 A+B prefix만 사용하며 C 표적을 계산하거나 점수화하지 않는다는 범위로 기록한다. 기존 Jena/Robin 역변환의 whole-TRAIN 정규화 계보·FP32 반올림도 없어졌다고 주장하지 않는다.

## 예측 전 봉인과 검산

각 CP에서 예측 호출 전에 B `target_normalized`, 원 관측 `observed`, `origins`, `channel_ids`를 모델 출력과 독립적으로 추출한다. 배열 shape/dtype/SHA와 모델·소스·데이터·scaler·구 곡선 점수 SHA를 공개 `REFERENCE_CONTRACT.json`에 먼저 봉인한다. 이미 예측 파일이 있는데 봉인이 없으면 소급 생성하지 않는다.

private 예측 NPZ에는 다음 5키만 저장한다.

```text
pred_normalized
target_normalized
observed
origins
channel_ids
```

공개 `PUBLIC_ERROR_STATS.csv.gz`에는 원점·채널별 정규화 SSE, SAE, 실제 관측 count만 저장한다. 결측 표적은 오차·분모에 포함하지 않는다. 채널별 전체 관측 오차 평균을 먼저 구한 후 관측이 있는 채널을 균등 평균한다. 원점별 평균을 다시 평균하지 않는다. 구 A 정규화 공간의 원 채널 MAE/MSE이며 물리 단위 오차로 바꾸어 부르지 않는다.

`rescore_checkpoint(private_dir, public_dir)`는 모델이나 데이터 loader를 호출하지 않는다. NPZ 파일 SHA, 5키, 배열 SHA, 사전 봉인, 시간 순서와 공개 충분통계를 재검사한다. 구 `runtime.score`와 새 일반 `evaluation` 모듈을 호출하지 않는 별도 FP64 구현으로 수치를 재계산한다. 구 곡선 receipt의 MSE와 MAE 각각 절대 오차 `1e-7`, 상대 오차 0으로 비교한다. 불일치하면 새 원수치를 보존하고 `CURVE_SCORE_MISMATCH`로 남긴다. 임계값을 느슨하게 바꾸거나 기존 점수로 덮어쓰지 않는다.

## 출력·자원·한계

public과 private에 각각 `legacy_e0/evaluation/<id>/cp<update>/`를 사용한다. public은 reference, manifest, 충분통계, 독립 검산만 담고 private은 전체 예측 NPZ를 담는다. 공개 manifest에는 개인 절대경로·token·전체 예측/표적을 넣지 않는다. 실행별 `REGISTRY_<timestamp>.json`은 실제 재추론 수, 재검산 수, 곡선 일치 수, 실패·budget pending, 소스 pin, 남은 deadline, 실제 F calls/streams를 따로 기록한다. 기존 archive를 재검산한 경우 `executed_now=false`이며 이번 F 호출로 재계상하지 않는다. 대상 CP가 없으면 `NO_ELIGIBLE_CHECKPOINTS`다.

GPU 작업 전과 모든 원점 batch에서 lease를 확인한다. F 모델의 forward pre-hook은 실제 foundation row microbatch마다 lease/deadline을 확인한다. 실제 F tensor hash는 로드 직후와 재추론 후 원 pin과 대조한다. 모델도 저장 CP와 tensor SHA가 같아야 한다. F streams는 원점 수 × 실제 F 입력 폭과 비교한다. 학습·대규모 백본 갱신은 수행하지 않는다. 중앙 GPU process-wall 계상은 호출자의 살아 있는 owner reservation이 담당하고 이 모듈은 별도 예산을 만들지 않는다.

CPU toy 8개는 마스크가 있는 FP64 macro 계산, B reference가 예측보다 먼저 생기는 순서, C 원점 차단, 관측 C 값이 잘못되어도 A+B prefix만 계산되는 조건, GPU lease 부재 차단, NPZ 변조, last/selected 파일 변조, 소급 reference 금지를 검사한다. 구 continuation CPU 검사와 함께 13개가 통과했다. 실제 F 실행과 모든 구 B 값의 일치 여부는 root 실행 후 생성되는 registry로만 판단한다. 복원한 가중치의 재추론 검사는 독립 학습 재현이나 새 일반화 검증이 아니다.

[확인] exact-ID 수선 뒤에는 지정한 `test_legacy_evaluation` 모듈의 14/14 synthetic 검사가 통과했다. 세 종류의 prefix 이웃을 fits와 budgets 각각에서 재현해 수정 전 실패·수정 후 통과를 확인했다. own malformed/id/endpoint/status/state 변조, 잘못된 budget source/file SHA 거부와 정상 own budget도 검사했다. CUDA_VISIBLE_DEVICES=-1, OMP/MKL/OPENBLAS=2 조건에서 임시 toy 배열·checkpoint만 사용했으며 실제9개 inventory·본 모델·원자료·GPU·B 재추론·T 실행은 하지 않았다. 실제9개 inventory 재검사는 별도 담당 CPU 실행자의 후속 결과로 판단한다. [원문 SHA·패치·검사 근거](SOURCE_PATCH_LEGACY_EVALUATION_EXACT_ID.json)를 보존한다.
