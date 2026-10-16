# 실제 배포 경로와 체크포인트 재추론 계약

`route_catalog.py`와 `route_predictions.py`는 저장 예측 재채점과 **실제 체크포인트 재추론**을 구분한다. 이 담당자가 실행한 모델 계산은 CPU 합성 자료의 toy forward뿐이다. 공식 Bolt 모델 로드·GPU 실행·실자료 재추론·새 학습은 수행하지 않았다. 실자료에서는 네 ridge와 Beijing/PeMS G의 실제 저장 artifact 6개를 CPU에서 읽고 SHA·계약이 맞는 배포 명세를 구성한 것까지만 확인했다. 당시 Jena/Electricity G는 완료 receipt가 없어 `NOT_READY`로 거부했다.

## 호출 API와 실제 가중치 결속

```python
route_spec, model_size = route_catalog.build_route(run_id, task, cache_root, public_root)
route_sha = training.fingerprint(route_spec)
expected_rows = route_catalog.expected_F_streams(route_spec, batch_size, len(task["columns"]))
selected = route_catalog.artifact_bindings(route_spec)

result = route_predictions.predict_route(
    run_id, task, "V_select", cache_root, public_root,
    model_path=local_model_snapshot, device="cuda", deadline=lease.deadline, lease=lease,
)
replay = route_predictions.replay_route(
    run_id, task, cache_root, public_root,
    model_path=local_model_snapshot, device="cuda", deadline=lease.deadline, lease=lease,
)
```

`build_route`는 `(deployment.Route에 직접 전달하는 dict, model_size)`를 반환한다. size는 `small/mini/tiny` 또는 F가 필요 없는 경우 `None`이다. 실제 registry, BASE/CONTROL/RIDGE receipt, calibration receipt와 private spec, E4 alias manifest를 확인한다. 활성 경로의 모든 adapter/head/bias/PCA artifact는 private 상대 경로와 실제 파일 SHA로 결속한다. 공개 명세에 로컬 절대 경로나 token을 넣지 않는다. 경로가 private root 밖으로 나가거나 artifact가 바뀌면 거부한다.

등록 상태나 방법 이름으로 가중치의 존재를 추정하지 않는다. 실제 완료 receipt가 없으면 그 경로만 거부한다. 학습 부모의 data/fold/허용 학습 끝을 확인하며, calibration spec의 config hash와 실제 부모 forecast model hash도 적합 당시 계약과 대조한다. 존재하는 표준 V manifest와 archive의 실제 hash를 route에 결속한다. mini/tiny는 `MODEL_ASSETS.json`의 공식 revision·파일 SHA를 사용한다. small은 기존 실제 GPU probe의 foundation state SHA/revision을 결속한다.

반환 `model_size=None`은 항상 CPU 실행을 뜻하지 않는다. G 같은 직접 신경망은 선택한 device에서 동작할 수 있다. **RAW_RIDGE는 NumPy FP64이므로 명시적으로 `device="cpu"`를 사용해야 한다.** 다른 GPU 경로와 별도 하드웨어 비용 범위로 보고한다. GPU 재추론은 유효한 `lease`를 반드시 전달해야 하며, 예산을 예측 microbatch마다 검사한다.

## 활성 경로와 F 행 수

- alpha0는 실제 G-only 경로이며 b/F를 생성하거나 호출하지 않는다.
- alpha1은 실제 b-only 경로이며 G를 생성하거나 호출하지 않는다.
- alpha0.5와 M-fitted alpha는 두 부모를 계산한다. 적합 alpha가 우연히 0이나 1이어도 임의로 계산 생략을 추가하지 않는다.
- conditional H는 selected.pt와 고정 G/b, past H는 selected.pt와 G만 사용한다. legacy residual은 selected.pt와 b만 사용한다.
- Raw calibration은 실제 Raw 예측에 M_cal에서 적합한 channel bias NPZ를 더한다.
- E4 shared/posthoc는 실제 full-C 부모 artifact와 요청 K를 사용한다. posthoc는 저장된 `[C]` 학습 계약을 바꾸지 않고 배포 시 허용 prefix에 요청 K를 추가한다.
- nested PCA는 private `ordered/<fold>_<dataset>/nested_pca.npz`와 실제 foundation SHA를 사용한다.

F 행은 명세의 활성 경로를 재귀적으로 합산한다. Raw/LoRA는 batch×C, 압축·PCA는 batch×K, 직접 모델은 0이다. 실제 `deployment.Route.last_receipt`의 logical/physical F rows와 호출 수를 합산해 예상 logical rows와 일치해야 저장한다. 저장 예측을 읽어서 F를 호출한 것으로 세지 않는다.

## V/T 독립 목표 seal과 저장 위치

`predict_route`는 전체 V 또는 T cohort의 실제 물리 입력 → 창별 채움/정규화 → P/E/F/D/G/H → 물리 미래 반환을 실행한다. 평가 좌표는 해당 fold의 A60 정규화다. origin stride는 H이며 마지막 중첩 origin을 추가하지 않는다.

T에서는 다음 조건을 **target_window 호출·모델 로드·추론 전에** 검사한다.

```text
SELECTION_LOCK.json.locked == true
SELECTION_LOCK.json.allowed_route_sha[run_id] == training.fingerprint(route_spec)
```

검사 뒤 Task 진리값으로 target/mask/origin/channel을 독립 구성하고, source/model/scaler/cohort와 예상 F rows를 담은 `REFERENCE_CONTRACT.json`을 추론 전에 생성한다. 생성 시각도 보존한다. 기존 예측이 있는데 reference가 사라졌다면 사후 reference를 새로 만들어 사전 seal인 것처럼 쓰지 않는다.

기존 표준 V writer를 덮지 않도록 별도 namespace를 사용한다.

```text
private: route_predictions/<run_id>/<V_select or T_final>.npz
public:  route_predictions/<run_id>/<split>/REFERENCE_CONTRACT.json
         route_predictions/<run_id>/<split>/PREDICTION_MANIFEST.json
         route_predictions/<run_id>/<split>/SCORES.csv
         route_predictions/<run_id>/<split>/PUBLIC_ERROR_STATS.csv.gz
```

NPZ는 기존과 같은 다섯 배열 `pred_normalized/target_normalized/observed/origins/channel_ids`를 private에만 보존한다. 공개 오류 통계는 원점×채널×horizon-band SSE/SAE/count다. 모든 배열과 archive를 hash로 결속한다. 반환값은 배열·metadata·manifest·reference_contract·score·cost_record·public_stats_relative_path를 포함한다. 기존의 동일 결과를 재사용하면 `executed_now=False`이며 새 추론을 했다고 주장하지 않는다.

## 고정 V subset 체크포인트 재추론

`replay_route`는 기존 표준 `predictions/<run>/V_select` 또는 E4 `ordered_predictions/<run>`의 실제 manifest/archive hash를 먼저 확인한다. 전체 등록 V 원점에서 first/middle/last와 나머지 중 seed92601로 3개를 뽑아 중복 없이 고정한다. 원점·Task target·mask·channel·artifact SHA를 별도 pre-inference reference에 고정한 뒤 실제 배포 경로를 호출한다.

```text
private: route_replays/<run_id>/V_subset.npz
public:  route_replays/<run_id>/PRE_INFERENCE_REFERENCE.json
         route_replays/<run_id>/REPLAY_RECEIPT.json
```

저장 표준 예측과 실제 재추론 예측의 허용 오차는 `atol=1e-5, rtol=1e-4`다. LoRA는 병합된 배포 경로로 재추론한다. 실제 F rows, 재추론 최대 차이, 모든 활성 selected artifact SHA를 기록한다. 실패하면 `FAIL_NUMERICAL_PARITY`로 보존하고 허용 오차를 사후 완화하지 않는다. 이는 고정 V 부분집합의 추론 재현이며 전체 origin replay·독립 학습 재실행·독립 일반화 검증이 아니다.

## 실제 CPU 검사와 수치 수정

초기 두 모듈 부재로 4실패를 확인한 뒤 구현했다. 최종 전용 검사 10개는 실제 toy forward로 alpha0의 F0, alpha1의 C rows, Hpast의 불필요 Raw 로드0, legacy의 G 로드0, Raw bias, nested PCA와 posthoc의 요청 K rows를 확인했다. T gate 이전 target 접근 금지, 실제 artifact 손상 거부, 기존 예측의 사후 seal 금지, 저장 표준 V 대비 고정 subset replay도 검사했다.

추가로 물리 값이 약 1e6이고 G 혁신량이 정규화 좌표에서 .5인 합성 사례에서 FP32 물리 역변환이 normalized parity를 깨는 문제를 재현했다. 수정 전 최대 차이는 **0.0051937103**이었다. root가 `deployment.py`의 역변환과 CPU 반환을 FP64로 바꾼 뒤 같은 사례의 최대 차이는 **0**, 허용 오차를 바꾸지 않고 통과했다. 내부 F/신경망 계산은 FP32, native ridge는 FP64를 유지하며 최종 물리 반환/역변환 FP64를 비용 scope에 명시한다.

```text
python -X utf8 research/informs_broad_validation_20261015_v1/test_route_predictions.py
Ran 10 tests in 0.470s
OK
```

동시에 실행한 캠페인 전체 CPU 회귀는 **149 passed, 37 subtests passed**였다. 당시 다른 담당자가 구현 중인 `report_figures.py`가 아직 없어 `test_report_figures.py`의 5개가 ModuleNotFoundError로 실패했다. 따라서 그 시점의 전체 suite를 PASS로 표시하지 않는다. 이 경로 담당의 10개에는 실패가 없었다. 이후 전체 회귀 완료 여부는 캠페인 최종 검증 기록을 따른다.
