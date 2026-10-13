# Forecast generalization 독립 검증

- 생성 시각: 2026-10-13 18:13:11
- 상태: `PASS`
- 모드: `final`
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
- model/checkpoint artifact SHA: 61개 확인, missing 0, mismatch 0
- TEST 의심 경로 접근 수: 0
- resource/ledger JSON 접근 수: 0

## 해석

[확인] 최종 manifest 기준 독립 재채점과 계약 검사가 통과했다.

## warnings

- 없음

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

## clean-cost 정적/결과 리뷰

```json
{
  "script_exists": true,
  "static_checks": {
    "resolves_full_and_repair": true,
    "enforces_70_cases_236_methods": true,
    "loads_selected_models_and_checkpoint_sha": true,
    "uses_existing_prediction_for_latency_parity": true,
    "records_new_training_zero": true,
    "guards_test_access": true,
    "checks_competing_gpu_processes": true,
    "checks_backbone_unchanged": true,
    "does_not_call_train_codec": true
  },
  "terminal_results": [
    {
      "path": "research/forecast_generalization_study_20261013/run_clean_cost_v1/RESULT.json",
      "sha256": "272118b91d928abd9593e8b29866b4cead5015122390ba0d8a6963708ed52356",
      "status": "COMPLETE",
      "record_count": 236,
      "resolved_case_count": 70,
      "resolved_method_count": 236,
      "new_training_updates": 0,
      "source_runs": {
        "full": {
          "path": "research\\forecast_generalization_study_20261013\\run_full_v1\\RESULT.json",
          "sha256": "5af9567f20cbb88b3a0bc9f81b1eec618cb2246ee0f5cedcb1cd3a4c916b4888"
        },
        "repair": {
          "path": "research\\forecast_generalization_study_20261013\\run_repair_v1\\RESULT.json",
          "sha256": "eccb9d40047a270f9352cd5df92e12ea18987bdc9d532c5879497a3f2b178cbb"
        }
      },
      "resources": {
        "pid": 49612,
        "tag": "clean_cost_v1",
        "status": "COMPLETE",
        "gpu_process": true,
        "started_unix": 1791882716.1087995,
        "wall_seconds": 45.96761070005596,
        "peak_sampled_rss_bytes": 1580142592,
        "peak_allocated_vram_bytes": 396548096,
        "cpu_seconds": 44.4375,
        "new_cache_bytes": 734804589,
        "new_research_bytes": 134168895,
        "cumulative_gpu_process_wall_seconds": 1279.5839344998822,
        "external_completed_gpu_process_receipts": [
          {
            "path": "research\\forecast_metric_alignment_20261013\\run_v1\\RESOURCE_USAGE.json",
            "sha256": "d582ae64141e3b677f546148d33d0bfcf4675ccf7c2af401a76fb9b187d9a953",
            "wall_seconds": 63.32161619979888,
            "status": "FAILED_PRESERVED"
          },
          {
            "path": "research\\forecast_metric_alignment_20261013\\run_v2\\RESOURCE_USAGE.json",
            "sha256": "d1c23a2de5216cf71d85aa499cb8f3af402f77a59796d42c6f4cd0f53fe862d3",
            "wall_seconds": 320.24091779999435,
            "status": "FAILED_PRESERVED"
          },
          {
            "path": "research\\forecast_metric_alignment_20261013\\run_v3\\RESOURCE_USAGE.json",
            "sha256": "89d621f13896412d37765ceb394edbf70c6103b7255501f2ae1a22d9f9c58163",
            "wall_seconds": 402.12076950003393,
            "status": "COMPLETE"
          }
        ],
        "external_completed_gpu_process_wall_seconds": 785.6833034998272,
        "all_new_campaign_gpu_process_wall_seconds": 2065.2672379997093,
        "external_new_storage_bytes": 3425702,
        "all_new_storage_bytes": 872399186,
        "error": null
      }
    }
  ],
  "notes": [
    "measure_resolved_cost.py는 full_v1+repair_v1을 병합하고 selected model/checkpoint/prediction SHA를 확인한 뒤 기존 science prediction으로 latency만 다시 측정하도록 작성되어 있다."
  ]
}
```
