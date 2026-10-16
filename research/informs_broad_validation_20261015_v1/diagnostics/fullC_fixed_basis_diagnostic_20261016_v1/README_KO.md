# Full-C fixed PCA 좌표 변환 진단

이 폴더는 **새 학습·모델 추론 없이** Beijing의 저장된 developmental V 충분통계와 작은 고정 basis/scaler 파라미터만 검사한다. 실제 실행 결과는 아직 없으며 ROOT의 코드 검토 후 명시 입력을 고정하고 실행한다.

## 먼저 분리할 질문

`K=C=11`이면 저장된 전체 PCA basis가 원 채널 정보를 사실상 보존하는가? 그렇더라도 원 채널에 바로 F를 적용하는 Raw와, 회전·중심화한 좌표에 F를 적용하고 역투영하는 경로가 같은가? `K3/6/11`의 평균 오차 차이는 특정 시기·horizon·채널에 집중되는가?

전체 basis의 rank/condition/Gram/identity 잔차는 변환의 수치 특성을 보여준다. 실제 과거 입력을 복원한 검사는 아니다. 임의의 잔차 공차를 만들어 성능 성공이나 독립 일반화 통과로 바꾸지 않는다.

## 끝까지 추적한 원 경로

- [`ordered_pilot.py:133`](../../ordered_pilot.py)는 A60 끝12622에서 전체11×11 PCA basis, center, mean/std를 `ordered/primary_beijing/nested_pca.npz`에 고정한다. 앞 K열을 모든 nested prefix가 공유한다. `:263`에서 고정 `_FixedPCA`를 생성한다.
- [`fit_base.py:55`](../../fit_base.py)의 고정 경로는 `F((x-center) E_K) E_Kᵀ + center`이다. E와 center는 buffer이고 optimizer update는 없다. 전체 basis가 직교하면 F 적용 전후의 투영 자체는 항등에 가깝지만 F의 회전 등변성은 보장되지 않는다.
- [`prediction_cache.py:271`](../../prediction_cache.py)의 Raw는 `F(x)`이다. 같은 canonical A60 mean/std로 입력/점수를 정규화한다. 별도 PCA center·역투영·학습형 D·current-value anchor는 없다.
- [`networks.py:284`](../../networks.py)의 **학습형** ForecastAdapter는 원 현재값에 latent lag7 innovation을 더한다. 이 anchored D를 고정 nested 경로의 decoder와 혼동하지 않는다.
- [`networks.py:316`](../../networks.py)의 Direct G도 원 현재값을 anchor로 사용하지만 별도 MLP 예측기이고 F를 호출하지 않는다. [`ridge_baseline.py`](../../ridge_baseline.py)의 ridge도 별도 학습 경로다. 최종 점수 cohort가 같다는 사실만으로 모든 학습 조건이 같다고 하지 않는다.
- [`training.py:51`](../../training.py)의 FastWindows는 supplied L512 안에서 과거 관측을 forward-fill하며 창 이전 관측은 0(정규화 mean)으로 처리한다. [`data_contract.py:109`](../../data_contract.py)의 PCA center는 A60 causal fill 후 정규화한 자료의 중심이다. 결측 처리 때문에 이 center가 정확히 0이라고 가정하지 않는다.
- [`gpu_runtime.py:31`](../../gpu_runtime.py)는 각 original/latent 채널을 독립 row로 F에 전달한다. 따라서 전체 회전은 F가 보는 각 단변량 경로의 모양을 바꾼다. 원 source·SafeInstanceNorm·실제 backbone은 이 감사에서 수정하거나 로드하지 않는다.

## 명시 입력 계약

실행 스펙 JSON의 `format`은 `fullC_fixed_basis_saved_V_diagnostic_v1`, `expected_dimensions`는 `{"origins":43,"horizon":48,"channels":11}`이다. 모든 입력은 `path/label/bytes/sha256`을 명시한다. 스펙은 private 위치에 두고 공개하지 않는다.

- `basis`: 위 작은 NPZ만 허용한다. key는 basis/center/mean/std/fit_end/data_sha/source_sha 7개이며 압축 해제 합계64KiB 이하다. Task, safeFIT, 원 관측, 정답, 예측 배열, weights는 허용하지 않는다.
- `basis_contract`: `nested_pca_contract.json`.
- `sources`: ordered_pilot.py, prediction_cache.py, fit_base.py, training.py, networks.py, data_contract.py, evaluation.py, gpu_runtime.py, ridge_baseline.py 9개의 실제 원 파일 SHA. 원 basis의 source fingerprint와 대조한다.
- `routes`: RAW, DIRECT_G, RAW_RIDGE, PCA_K3, PCA_K6, PCA_K11 순서의 객체. 각 항목은 기존 `PUBLIC_ERROR_STATS.csv.gz`, `PREDICTION_MANIFEST.json`, `REFERENCE_CONTRACT.json`, `SCORES.csv`를 각각 `stats/manifest/reference/scores`로 고정한다.

Raw/G/ridge는 캠페인 `predictions/primary_beijing_E1_{RAW_BOLT_SMALL,DIRECT_G,RAW_RIDGE}_k0_s92601/V_select/`를 사용한다. PCA3/6/11은 `postT_V_only_cpu_multik_20261016_v1/ordered_predictions/postT_cpu_beijing_E4_NESTED_PCA_k{3,6,11}_s92601_v1/`이다. 경로 이름을 근거로 수치를 추정하지 않고 각 SHA와 manifest를 실제 확인한다.

```powershell
python -B audit_fixed_basis.py
python -B test_audit_fixed_basis.py
python -B audit_fixed_basis.py --execute --spec <explicit-private-spec.json> --spec-sha256 <SHA256> --output <fresh-output-directory>
```

기본 실행은 stdlib metadata만 출력하고 입력 파일0개/NumPy import0회다. ROOT가 코드를 읽고 실제 실행을 지시하기 전에는 마지막 명령을 수행하지 않는다. 실행 디렉터리가 있으면 덮어쓰지 않는다.

## 출력과 해석 경계

`RESCORED_V.csv`는 원 SSE/SAE/count로 전체43 origins의 채널 균등 FP64 MSE/MAE를 다시 구성한다. 원 점수 공차는 atol1e-9/rtol1e-7이다. `PAIRED_GROUPS.csv`는6방법의15쌍을 전체·3연속 시간 block(15/14/14 origins)·horizon[0,12)/[12,24)/[24,48)·11채널에서 비교한다. Delta는 left−right이고 작은 오차가 좋다. 승패는 관측 count>0인 origin×channel×band cell의 SSE/SAE 엄격 비교이며 동률을 따로 센다.

`RESULT.json`은 basis 수치 진단, cohort 지문, 재채점 상태와 불확실성을 기록한다. `INPUT_BINDINGS.json`은 공개 가능한 label/크기/SHA만 남기고 private 절대경로·basis 원값은 싣지 않는다. 모든 원 입력은 마지막에 다시 hash 확인한다.

충분통계의 row key/count와 manifest의 원 target/mask/scaler/cohort fingerprint를 대조할 수 있지만, band count만으로 band 내부 개별 관측 위치나 원 정답을 재구성할 수 없다. 이는 새 prediction-array 독립 감사가 아니다. 단일 seed·반복 V·post-T 탐색이라는 한계, 기존 Raw 전체 V의 장치 불확실성, 학습형 G/ridge와 고정 경로의 학습/anchor 차이는 남는다. 개별 cell/시간 block을 독립 유의성 표본으로 세지 않으며, 좌표 변환이 가역이어도 TSFM만이 성능 차이의 유일 원인이라고 단정하지 않는다.
