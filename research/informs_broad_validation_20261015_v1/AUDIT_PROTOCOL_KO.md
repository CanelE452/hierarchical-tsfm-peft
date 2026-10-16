# 독립 재채점과 조합 reference의 두 단계 봉인

[audit_campaign.py](audit_campaign.py)는 기존 [audit_predictions.py](audit_predictions.py)의 독립 FP64 재채점기를 실행하고 전체 실험 registry를 보존한다. 이 담당자가 실행한 것은 CPU 합성 10개 검사다. 실제 캠페인의 검산 수·성패는 `audits/audit_<timestamp>/AUDIT_REPORT.json`을 확인해야 한다. 문서 작성이나 합성 PASS를 실자료 감사 완료로 세지 않는다.

## 조합을 실행하기 전에 해야 하는 일

```python
refs = prepare_combination_reference(
    run_id, task, parent_v, g_v, mean, std,
    cache_root, public_root, spec=spec, fold="primary",
    kind="combination",
)
receipt = fit_combination(...)
standard_manifests = wrap_combination_output(
    receipt, refs, cache_root, public_root, kind="combination",
)
```

Raw의 M-only bias 보정은 같은 API의 `kind="raw_calibration"`을 사용한다. 이때 spec에는 해당 run_id·variant를 주고 Raw V를 부모로 전달한다. `public_root`는 캠페인 최상위이며 `fit_combination`의 공개 하위 출력 디렉터리와 구분한다.

prepare는 실제 Task의 원 관측 mask·원 채널 순서·등록 V origin·A60 정규화 정답을 따로 계산한다. 부모의 실제 V cache와 일치하는지 확인하고 이미 완료된 실제 부모 manifest 파일과 archive/model SHA를 결속한다. source/config/data/scaler·부모·cohort는 fit 전 `predictions/<id>/V_select/PRE_TARGET_CONTRACT.json`에 저장한다. 해당 reference 없이 private 조합 V NPZ가 먼저 존재하면 사후 reference 생성으로 거부한다. 기존 동일 reference가 있으면 hash를 확인해 재사용한다.

학습 후 선택될 H 가중치는 fit 전에 알 수 없다. 따라서 이 사전 계약의 모델 클래스/L/H hash를 실제 학습 가중치 hash라고 부르지 않는다. wrap은 fit receipt의 source/config/data/scaler와 사전 계약을 대조하고 실제 `selected.pt` 또는 `bias.npz`의 정확한 SHA를 검산한다. alpha는 고정 또는 M-fitted 실제 계수를 결속한다. 이 실현 artifact·실제 필요한 부모의 model SHA·출력 label로 실제 model identity를 만들고 `REFERENCE_CONTRACT.json`과 `PREDICTION_MANIFEST.json`을 작성한다.

표시는 `target_cohort_sealed_before_fit=true`, `selected_model_identity_bound_after_fit=true`, `all_identity_fields_preinference_claimed=false`다. 전체 모델 identity가 첫 V checkpoint 평가 전에 모두 확정됐다는 주장은 하지 않는다. 사후 model binding을 이용해 사전 target/cohort를 바꿀 수 없도록 두 파일의 연결 SHA와 고정 필드를 감사한다. 정확한 artifact binding은 그 checkpoint가 실제로 해당 예측을 생성했다는 재추론 증명이 아니므로 checkpoint replay를 별도로 수행해야 한다.

## alpha alias와 실제 F 사용량

alpha family의 네 출력은 원 ALPHA_FIT registry ID와 ALPHA0/ALPHA05/ALPHA1 ID로 연결한다. 내부 `run_id__fitted` 같은 임시 label을 새 독립 실험으로 세지 않는다. 모든 alias의 source/cohort/부모가 연결되며 한 번의 부모 학습을 네 번의 학습으로 세지 않는다.

wrap은 기존 private `combinations/<family-id>/<label>_V_select.npz` 또는 `V_select.npz`를 읽어 reference와 정답·mask·origin·채널을 대조한다. 그 저장 예측의 V 점수와 origin×channel×horizon-band SSE/SAE/count를 표준 공개 위치에 생성한다. Raw bias도 같은 형식이다. 원본 NPZ나 원래 fit 영수증을 덮지 않는다.

이 단계는 저장 부모와 소형 head/calibration의 결과를 포장·재채점하는 범위다. 새 F 추론으로 N×K streams를 꾸미지 않는다. cost ledger는 `actual_F_streams=expected_F_streams=F_calls=latent_k=0`, scope는 `cached_parent_calibration_output`이다. 실제 배포에서 b/G/H를 실행하는 비용과 부모의 앞선 실제 F inference 영수증은 따로 유지한다. 저장 결과의 산술 재검산과 배포 비용 측정을 혼동하지 않는다.

## 전체 감사 실행

```text
python research/informs_broad_validation_20261015_v1/audit_campaign.py --public-root <campaign> --private-root <private-cache>
```

현재의 모든 표준 manifest와 reference를 찾아 원 archive SHA·배열 fingerprint·identity·실제 mask를 확인한다. 공개 통계가 있는 V/T는 독립 구현으로 저장 예측을 FP64 재채점하고 gzip 통계를 다시 집계한다. 전역 SCORES.csv가 있으면 행의 MSE/MAE·관측수·origin 수·C/H도 대조하고, 검산 가능한 manifest 없이 존재하는 V/T 점수행을 거부한다. 값이 없는 dense M cache는 공개 평가 점수로 만들지 않고 private array integrity와 독립 FP64 오차만 검산한다. 일부 manifest가 깨져 있어도 다른 경로 검산을 계속하며 그 오류를 해당 감사 batch에 남긴다.

전체 실험 범위는 `EXPERIMENT_REGISTRY_INITIAL.csv`와 모든 `EXPERIMENT_REGISTRY_EXTENSION_*.csv`의 ID 합집합이다. 확장 파일의 같은 stem JSON은 다음을 가져야 한다.

```json
{
  "registered_unix": 1234567890.0,
  "csv_sha256": "실제 확장 CSV의 SHA256",
  "before_execution": true,
  "before_T_selection": true,
  "run_ids": ["새 ID"]
}
```

확장의 schema·파일 SHA·고유 ID·사전 등록 timestamp·실행/선정 이전 선언을 검산한다. 이용 가능한 실제 시작 시각과 선정 잠금 시각도 비교한다. 기존 ID 재정의나 등록하지 않은 새 current 행, 기존 dataset/fold/모델/C/K/seed/L/H/필수 범위/계획 epoch 변경을 거부한다. 상태·reason·resume_command만 갱신 가능하다. 확장 뒤 current registry는 이 합집합과 같아야 한다.

기본 0.25-epoch 결과는 실제 저장 예측이 있어 검산할 수 있지만 원 registry의 `PARTIAL_SCREEN025`는 그대로 보존한다. 하위 검산기의 registry projection은 **그 prediction artifact를 생성한 실행**만 COMPLETE_VALID로 다루며 전체 mandatory 실험이나 충분 학습이 끝났다는 뜻이 아니다. pending/blocked/실패 행을 포함한 전체 원 registry를 결과 옆에 보관한다. E0 legacy 학습 장부, 최종 배포 비용, checkpoint replay와 학습 수렴은 이 ID/산술 감사와 다른 근거다.

감사 결과는 매번 새 `audits/audit_<timestamp>/` 아래 JSON으로 남긴다. 이전 실패나 부분 감사를 덮지 않는다. 진행 중이거나 막힌 필수 항목이 있으면 `mandatory_coverage_complete=false`다. 유효한 저장 예측만 통과한 경우 `PASS_AVAILABLE_PREDICTIONS_COVERAGE_INCOMPLETE`, 아무 예측도 없으면 `NO_PREDICTIONS_TO_RESCORE`다. 검산 성공을 성능 개선·독립 holdout 일반화·충분 학습·main 게시로 바꾸어 말하지 않는다. `checkpoint_replay_count`, `training_rerun_count`, `timer_reexecution_count`는 이 CPU 산술 감사에서 모두0이다.

## 실제 CPU 검사 범위

```text
python research/informs_broad_validation_20261015_v1/test_audit_campaign.py
```

모듈 부재 RED 후 합성 10개 GREEN을 확인했다. 실제 CPU Current 부모 cache를 생성하여 다음을 검사했다: alpha alias 네 개의 registry 연결과 독립 재채점, 다시 hash한 origin shift 거부, 예측 후 reference 생성 거부, 실제 selected artifact SHA 변경 거부, 부분 실행 상태 보존과 blocked 행 누락 탐지, cached 조합의 가짜 F streams 탐지, Raw bias exact artifact와 dense 부모 재채점, 합법적 사전 등록 확장과 무등록 추가 행 구분, 기존 scientific 계약 변경 거부, 깨진 manifest를 보고하면서 다른 bundle 감사 계속하기다.

기존 `test_evaluation.py`는 정답 shift·채널 swap·잘못된 mask·SHA·실패 경로 누락·Kmax 비용 위장·점수 변조를 별도로 검사한다. 이 문서는 그 기존 검사를 새 실자료 실행으로 중복 계산하지 않는다. 실제 모델 재추론과 재학습 여부는 각각 별도 영수증으로 확인해야 한다.


## Legacy B의 별도 검산

`legacy_e0/evaluation/<legacy-id>/cp<update>/PREDICTION_MANIFEST.json`은 새 V/T와 schema가 다르다. 일반 감사는 이 명시적 namespace의 경로·ID·update·`legacy_B_development` 구간·기존 개발 노출·보호 TEST 미사용 선언을 먼저 확인한 뒤 `legacy_evaluation.rescore_checkpoint`를 실제 호출한다. private 예측 archive/독립 target 봉인/공개 통계 SHA·FP64 점수·원 checkpoint curve를 재검산하고 저장 `INDEPENDENT_RESCORE.json`과 대조한다. 손상·curve mismatch·다른 split을 이 namespace로 숨긴 경우 감사 오류다.

`LEGACY_B_AUDITS.json`과 `legacy_B_rescoring_passed`는 새 V/T의 `PREDICTION_AUDITS.json`·검산 수와 분리한다. 따라서 전체 manifest를 빠짐없이 확인하면서도 과거 B를 새 T 또는 독립 일반화로 세지 않는다. 학습 재실행이나 이번 감사 안에서의 checkpoint 재추론을 뜻하지 않는다. 합성 CPU20검사 PASS를 확인했으며 실자료 B의 수는 실제 감사 영수증만 따른다.
