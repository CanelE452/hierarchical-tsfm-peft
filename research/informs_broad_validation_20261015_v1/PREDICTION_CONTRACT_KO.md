# 부모 예측 캐시와 초기 채점 계약

## 호출

```python
predict_parent(
    run_id, task, split, cache_root, public_root, foundation,
    dense=False, receipt=None, baseline=None, fold="primary", role="base_parent", deadline=None,
)
```

`split`은 M_cal/V_select/T_final이다. `dense=True`는 M_cal에만 허용하며 stride1을 사용한다. 다른 경우 strideH이고 마지막 끝점을 임의로 덧붙이지 않는다. `task`와 `foundation`은 호출자가 이미 준비해 전달한다. 이 helper는 데이터 다운로드나 모델 로딩을 하지 않는다.

운영용 `deadline`은 Unix wall-clock 시각이며 각 실제 예측 microbatch 전에 확인한다. 시각에 도달하면 TimeoutError로 중단하고 부분 예측을 완성 캐시로 저장하지 않는다. 과학 config/identity에는 넣지 않으므로 같은 완성 캐시는 deadline이 지나도 검산 후 재사용할 수 있다. 이미 시작한 연산이나 CPU 준비·저장 전체의 강제 실시간 종료를 보장하는 기능은 아니다.

반환 dict의 `pred_normalized`, `target_normalized`, `observed`, `origins`, `channel_ids`, `metadata`는 combination 실행기에 바로 전달할 수 있다. 추가 필드는 `manifest`, `reference_contract`, `score`, `cost_record`, `public_stats_relative_path`다.

## 입력 표현과 공통 평가 척도

부모는 `BASE_FIT_RECEIPT.json`의 상태·run ID·artifact SHA를 검사한 뒤 private selected_bundle에서 복원한다. 원 spec으로 같은 모델 클래스를 만들고 state를 strict하게 로드한다. Parent data/fold/normalization/foundation hash를 검사하고, 자체 적합 범위의 평균·표준편차로 과거 입력을 만든다. 과거 창 안의 관측값만 forward-fill하며 창 이전의 값은 읽지 않는다.

모든 원 C 출력의 주평가 척도는 해당 fold의 **A_fit+A_stop 통계**로 통일한다. 원 부모 normalization이 다르면 예측을 원 물리 단위로 변환한 뒤 공통 척도로 옮긴다. 이 변환은 FP64로 계산하고 결과를 FP32로 저장한다. 두 normalization이 동일하면 불필요한 왕복 연산을 생략한다. 원 단위 예측 오차와 공통 척도 오차를 혼용하지 않는다.

기본 `role="base_parent"`는 E2용 부모 조건인 `parent_fit_end <= M_cal.start`를 강제한다. `role="AplusM_control"`은 spec에도 같은 역할을 명시한 새로운 적합 artifact에만 허용하며 V/T에서만 예측한다. 학습 경계는 `normalization.train_end`를 우선 읽고, 이 역할은 `M_cal.end`를 넘을 수 없다. 입력 정상화의 `fit_end`와 학습 label의 `train_end`를 분리한다. A+M artifact를 기본 부모로 바꿔 표시하거나 그 artifact로 M_cal의 학습용 예측을 만들면 거부한다. 두 역할 모두 공통 A60 scorer를 사용한다.

고정 baseline은 `RAW`, `CURRENT`, `SEASONAL`을 지원한다. PCA는 `baseline={"kind": "PCA", "latent_k": K}`로 호출한다. Seasonal은 metadata/명시 period가 과거 창 안에 들어가는 경우에만 마지막 period를 반복한다. PCA는 A60에서 새로 계산한 중심·basis와 고정 선형 inverse를 사용한다. Baseline의 model/artifact SHA는 baseline 설정·F hash·공통 scaler·data hash로 정의한 논리적 artifact hash이며 실제 weight 파일 hash와 구분한다.

`RAW_RIDGE`는 학습 완료한 자체 채널 raw-history ridge artifact를 지원한다. `baseline=None`과 RIDGE_FIT_RECEIPT.json의 자동 탐색 또는 `baseline="RAW_RIDGE"`를 사용할 수 있다. 실제 ridge 적합은 별도 실행기가 수행하며, 예측 helper는 `ridge_baseline.load_artifact`로 private NPZ를 읽고 `predict_normalized`를 호출한다. Torch checkpoint로 읽거나 encoder/F를 추가하지 않는다. Coefficient C×33×H, block 경계, normalization, data/source/feature hash, fold와 학습 끝을 검산한다. F logical rows는 0이다.

Ridge artifact의 scaler hash는 생산자 API `ridge_baseline.scaler_fingerprint(mean, std)`로 검산하고 `ridge_native_scaler_sha`로도 보존한다. 현재 생산자 API와 scoring/combination의 `scaler_sha`는 같은 shape/dtype/raw-byte 기반 공통 A60 알고리즘이다. 예전 `data_contract.array_sha256`를 한 번 더 감싼 방식으로 새 artifact를 검사하면 안 된다. 생산자 변경을 반영한 CPU fixture에서 먼저 fingerprint mismatch RED를 확인한 후 검산 호출을 수정했다. Ridge도 같은 독립 정답 봉인, base_parent의 학습 경계, V/T stride, T 선택 잠금, 원 C macro score 및 독립 auditor를 사용한다. Closed-form ridge의 전체 A origin 노출과 neural 0.25 epoch 노출은 별도 적합 보고서에서 구분해야 한다.

## 정답의 독립 봉인과 T 잠금

예측을 만들기 **전에** data_contract.target_window를 Task 원자료에 적용해 허용 cohort의 정답·관측 mask를 따로 생성한다. Origin과 원 채널 순서까지 shape/dtype/raw-byte fingerprint로 묶어 `REFERENCE_CONTRACT.json`에 저장한다. NPZ에서 정답을 다시 읽어 reference로 삼지 않는다. 예상 F rows도 route의 실제 원/잠재 폭으로 계산해 reference에 함께 봉인한다.

T_final은 `public_root/SELECTION_LOCK.json`이 존재하고 아래 세 조건을 모두 만족한 뒤에만 target_window를 호출한다.

1. `locked`가 JSON boolean true다.
2. run ID가 `allowed_run_ids`에 포함된다.
3. `allowed_artifact_sha[run_id]`가 해당 부모의 정확한 artifact SHA와 같다.

잠금 없이 T를 열거나 다른 부모 weight를 잠금 이후 대체하면 거부한다. 선택 잠금의 생성·허용 run 집합 확정은 상위 실행기의 책임이다.

## 저장·계수·초기 점수

Private NPZ는 `predictions/{run_id}/{split}[_dense].npz`에 exclusive 생성하며 키는 위의 다섯 배열로 고정한다. 공개 record 폴더에는 reference fingerprint, manifest, 정규화 오차 통계를 저장한다. 실제 정답·예측 배열과 private 절대경로는 공개하지 않는다. Model SHA는 artifact/weights/normalization을 함께 묶으므로 M과 V는 같은 부모 SHA를 갖는다. 배포 source chain과 학습 source chain은 둘 다 manifest에 남긴다.

Foundation의 실제 logical-stream counter 차이를 검사한다. Raw/LoRA는 N×C, 압축 부모는 N×K, Direct/Current/Seasonal은 0이어야 한다. 호출을 누락하거나 Kmax를 실행하고 작은 K로 표시하면 실패한다. 이 계수는 시간·memory의 독립 재측정이나 수렴 증명이 아니다.

Dense M_cal은 학습용 private 배열을 저장하며 공개 score/SSE 행을 대량 생성하지 않는다. V/T 및 non-dense 요청은 evaluation.error_statistics의 FP64 macro MSE/MAE와 전체 origin×channel×horizon-band SSE/SAE/count를 저장한다. count0 행도 유지한다. Gzip은 mtime0/빈 filename으로 만들어 동일 통계 내용에 대해 header가 재현 가능하다. 이 초기 점수는 별도 독립 auditor의 NPZ 재채점을 대체하지 않는다.

기존 캐시는 source/parent/archive SHA와 모든 배열 fingerprint를 확인한 경우에만 재사용한다. 불완전 파일이나 변경 계약은 덮지 않고 실패한다. LoRA는 정상·load 예외 경로 모두 hook을 제거하고 원 F hash를 복구한다.

## 합성 CPU 검산 결과

최초 6 RED 실패를 확인한 뒤 구현했으며 최종 9 unittest가 통과했다. 실제 데이터·F·GPU를 사용하지 않았다.

- CURRENT 전체 origin/mask coverage와 dense M_cal 비공개 통계.
- Raw 실제 F rows, 거짓 counter 거부, 캐시 재사용 시 F 추가 호출0.
- T 잠금 전 target 미접근과 잘못된 artifact hash 거부.
- Deterministic gzip header와 전체 band 행 수.
- 저장 DIRECT_G를 자체 A50 척도로 복원한 뒤 공통 A60으로 변환한 수치 및 M/V 동일 부모 SHA.
- 생성 Raw 캐시의 별도 audit_predictions FP64 재채점·공개 재집계 PASS.
- LoRA state load 오류 이후 hook 제거와 원 F hash 복구.
- RAW_RIDGE의 NPZ 전용 로드, direct CPU predictor와의 수치 일치, F0, M/V 동일 모델 SHA, 공통/native scaler 해시 구분, 독립 재채점 PASS, T 잠금·허용 후 F0, 잘못된 data 계약 거부. 최종 확장 suite는 12 unittest다.

이 검사는 실제 자료의 예측 개선, 실제 Bolt 추론, GPU 비용 또는 학습 수렴을 주장하지 않는다.
