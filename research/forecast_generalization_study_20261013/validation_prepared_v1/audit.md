# Forecast generalization 독립 검증

- 생성 시각: 2026-10-13 18:10:20
- 상태: `PREPARED_PARTIAL`
- 모드: `prepare`
- 수집 case 수: 70
- failure artifact 수: 8

## 핵심 검증 결과

- 독립 재채점 method 수: 236
- 재채점 상태: `{'pass': 236}`
- checkpoint 선택 검증: 56건 중 failure 0건
- unique core fit: 44 / expected final 44
- native fit: 32 / expected final 32
- method count: 236 / expected final 236
- batch cost count: 472 / expected final 472
- TEST 의심 경로 접근 수: 0
- resource/ledger JSON 접근 수: 0

## 해석

[확인] 최종 `RESULT.json`이 아직 없어서 완료된 case JSON과 NPZ 기준으로 검증기를 준비·부분 실행했다. 최종 manifest가 생성되면 같은 스크립트로 전체 44 core fit / 32 native fit / 236 methods / 472 batch cost를 다시 검산해야 한다.

## warnings

- research/forecast_generalization_study_20261013/run_repair_v1/jena_contract.json 없음: jena 데이터 계약 검증은 부분 수행
- research/forecast_generalization_study_20261013/run_repair_v1/robin_contract.json 없음: robin 데이터 계약 검증은 부분 수행

## errors

- 없음

## repair source 정적 리뷰

```json
{
  "repair_experiment_exists": true,
  "diagnostic_exists": true,
  "checks": {
    "uses_source_failed_paths_only": true,
    "does_not_iterate_all_cases_for_repair": true,
    "has_retry_replace_for_windows_lock": true,
    "identity_gradient_fix_is_conditional": true,
    "records_repair_of_metadata": true,
    "checks_frozen_bolt_hash": true,
    "checks_source_hashes_unchanged": true,
    "may_run_extra_g2_if_g1_selection_changes": true
  },
  "notes": [
    "repair_experiment.py는 source RESULT가 완료된 뒤 failed_paths만 순회하도록 작성되어 있다. 다만 G1 수리 후 config 선택이 바뀌면 사전 규칙에 맞춰 추가 G2를 실행할 수 있으므로, 최종 검증은 repair RESULT까지 병합해야 한다."
  ],
  "diagnostic_checks": {
    "zero_safe_sqrt_defined": true,
    "forward_unchanged_asserted": true,
    "positive_variance_gradient_unchanged_asserted": true,
    "no_training_updates_recorded": true
  }
}
```
