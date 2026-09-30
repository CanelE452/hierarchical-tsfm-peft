# v11 execution and reuse

Run location: repository root, Windows PowerShell, existing `.venv/Scripts/python.exe`. No package installation, raw download, or backbone download is part of this run. Exact environment, source snapshots, frozen parents, data bytes and checkpoint receipts are linked through `provenance.json`, `reuse_manifest.json`, `protocol.json`, and `ledger.json`. Raw data, weights and prediction arrays remain in ignored local cache.

These commands describe the approved execution order. The ledger is authoritative for whether a command has actually completed. All scheduled stages below have now completed, including 40 neural fits, 12 coefficient fits, 57 saved-output evaluations, 621 GPU cost rows, 36 CPU cost rows, reporting, and final numeric verification. Original failures and zero-fit repairs remain recorded. Listing a command is not a reason to repeat completed work.

Do not rerun into these output names. Completed results and failed attempts are preserved, the ledger rejects duplicate reservations, and fitting is prohibited after the joint selection/exposure seal. A separately authorized reproduction needs its own output namespace and resource ledger with the same pinned sources, rather than deleting these records.

```powershell
.venv/Scripts/python.exe research/tsfm_peft_internal_vs_subspace_v11_20260930/prepare_v11.py --initialize
.venv/Scripts/python.exe research/tsfm_peft_internal_vs_subspace_v11_20260930/test_model_v11.py --output research/tsfm_peft_internal_vs_subspace_v11_20260930/synthetic_checks01.json
.venv/Scripts/python.exe research/tsfm_peft_internal_vs_subspace_v11_20260930/train_v11.py --stage hog_direct --job hog_direct01 --reserve-seconds 1200
.venv/Scripts/python.exe research/tsfm_peft_internal_vs_subspace_v11_20260930/check_gpu_v11.py --job actual_bolt_checks01
# The preceding no-optimizer check failed on constant-context backward; see REPAIRS.md.
.venv/Scripts/python.exe research/tsfm_peft_internal_vs_subspace_v11_20260930/check_gpu_v11.py --job actual_bolt_checks02
.venv/Scripts/python.exe research/tsfm_peft_internal_vs_subspace_v11_20260930/check_analysis_v11.py
.venv/Scripts/python.exe research/tsfm_peft_internal_vs_subspace_v11_20260930/train_v11.py --stage main --job main_neural01 --reserve-seconds 16200
.venv/Scripts/python.exe research/tsfm_peft_internal_vs_subspace_v11_20260930/train_v11.py --stage select --job neural_selection01
.venv/Scripts/python.exe research/tsfm_peft_internal_vs_subspace_v11_20260930/gamma_v11.py collect --reserve-seconds 900 --name gamma_collect01
.venv/Scripts/python.exe research/tsfm_peft_internal_vs_subspace_v11_20260930/gamma_v11.py fit --reserve-seconds 1200 --name gamma_fit01
.venv/Scripts/python.exe research/tsfm_peft_internal_vs_subspace_v11_20260930/gamma_v11.py verify --reserve-seconds 900 --name gamma_verify01
.venv/Scripts/python.exe research/tsfm_peft_internal_vs_subspace_v11_20260930/evaluate_v11.py seal --reserve-seconds 60 --name evaluation_seal01
.venv/Scripts/python.exe research/tsfm_peft_internal_vs_subspace_v11_20260930/evaluate_v11.py predict --reserve-seconds 1800 --name evaluation_predict01
# Original inference failed on a transient ledger PermissionError; see REPAIRS.md.
.venv/Scripts/python.exe research/tsfm_peft_internal_vs_subspace_v11_20260930/evaluate_v11.py predict --reserve-seconds 1800 --name evaluation_predict02
.venv/Scripts/python.exe research/tsfm_peft_internal_vs_subspace_v11_20260930/evaluate_v11.py score --reserve-seconds 1200 --name evaluation_score01
.venv/Scripts/python.exe research/tsfm_peft_internal_vs_subspace_v11_20260930/cost_v11.py --gpu --job v11_cost_gpu01 --reserve-seconds 7200
.venv/Scripts/python.exe research/tsfm_peft_internal_vs_subspace_v11_20260930/cost_v11.py --cpu --job v11_cost_cpu01 --reserve-seconds 300
.venv/Scripts/python.exe research/tsfm_peft_internal_vs_subspace_v11_20260930/report_v11.py --job report01 --reserve-seconds 300
.venv/Scripts/python.exe research/tsfm_peft_internal_vs_subspace_v11_20260930/verify_v11.py --job final_verification01
```

`--reserve-seconds` reserves work against the common resource ledger. It does not automatically increase the global limit and it is not a forecasting-quality criterion. Parent GPU jobs account elapsed time once; their nested fit jobs account independent fitting attempts and optimizer updates. Read/forward/restore/score uses zero fits but charged resource time. Synthetic optimizer checking is separately bounded. Explicit technical retries preserve the original run, use a new attempt name, and consume reserve; no such retry is implied by this command list.

The original model and data contracts are unchanged. The one current numerical repair preserves the installed native scaling forward while making its exact-zero-variance backward finite, applied consistently to all v11 TSFM families. It is documented with the original failure and follow-up checks. Cost observers are stopped outside timed spans; manual progress/budget checks run between passes. Neither repair is claimed as a new forecasting method.

All Robin, Jena and Hog evaluation periods are previously exposed development data. Neural LR/checkpoint choices precede the twelve VAL coefficient fits; the joint selection seal precedes any v11 TEST output. Reported means average seed losses, not forecast ensembles, and pooled period scores use each channel's total observed count. Our saved-array verification is not independent reproduction.


Presentation correction: after report01, `plot_accuracy_cost` was called with the existing `report_family_comparison.json` datasets and `report_values.json` batch4 GPU options into `figures/v11_accuracy_cost_batch4_v2`. Only the throughput log axis and separated legend layout changed. `figure_revision01.json` links original/revised hashes; original report_values and its verified numeric sources remain unchanged.
