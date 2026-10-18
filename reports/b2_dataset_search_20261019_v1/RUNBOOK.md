# Run and resume the continuing search

Use the existing pinned Python3.11/OpenBLAS interpreter and local
Chronos-Bolt-small snapshot. No package installation or model download is
needed. Preserve the current parent `CUBLAS_WORKSPACE_CONFIG` and record it;
the worker does not force a value. One GPU worker runs at a time.

```powershell
$taskPython = '<PINNED_PYTHON_EXE>'
$taskAssets = '<ORIGINAL_ASSET_ROOT>'
$taskModel = '<PINNED_BOLT_SNAPSHOT_DIRECTORY>'
```

The concrete first candidate list was registered in `CANDIDATE_PLAN_001.json`
before numerical preparation. The data stage materializes only approved pre-T
prefixes, retains target masks and writes per-role observability diagnostics.
An inadmissible candidate is recorded rather than silently changed. This stage
refuses to overwrite prepared assets.

```powershell
& $taskPython -m research.b2_dataset_search_20261019_v1.data `
  --asset-root $taskAssets `
  --output-root .cache/b2_dataset_search_20261019_v1/data `
  --public-output reports/b2_dataset_search_20261019_v1/DATA_DIAGNOSTICS.json `
  --dataset all
```

Seal the ready dataset list and exact config/prefix SHA256 identities into
`waves/WAVE_001.json` before launching the batch. Keep the pre-registered order
and exclusions. The trainer separately seals all expected K/seed jobs before
the first model load, then seals all A_stop selections before any B2 fresh
refit/V evaluation. The coordinator checks input bytes on each invocation.

```powershell
& $taskPython -u -m research.b2_dataset_search_20261019_v1.run all `
  --wave 2 --model-dir $taskModel
```

The first prepared registry (wave1) was superseded before any GPU training by
`PREPARATION_AMENDMENT_001.json`. The cross-resolution ETTh2/ETTm2 boundary
requires the custom ETTh2_safe17400 cohort, with a13920-row prefix. ETTh1 and
Appliances retain their unchanged prepared inputs. The actual first training
batch is wave2; `CANDIDATE_PLAN_002.json` fixes all three datasets and their K
grids before its preparation/training. The initial16 potentially overlapping
hourly observations were prepared but never used for new training/forecasting.
This must remain disclosed when reporting the result.

The safe replacement can be prepared into an empty output location with:

```powershell
& $taskPython -m research.b2_dataset_search_20261019_v1.data_safe_etth2 `
  --asset-root $taskAssets `
  --output-root .cache/b2_dataset_search_20261019_v1/data `
  --dataset ETTh2_safe17400 `
  --amendment reports/b2_dataset_search_20261019_v1/PREPARATION_AMENDMENT_001.json
```

## Conditional next batch

`CANDIDATE_PLAN_003.json` was registered before the new prefixes or forecast
scores. ETTm1 (native15min, all7 channels) and Solar (native10min, all137
channels) are prepared while wave2 trains. `WAVE_003_PREPARED.json` has status
`PREPARED_CONDITIONAL_NOT_ACTIVATED`; it is not an executable training registry.
No GPU model or forecast score exists for these two candidates at preparation.
If wave2 yields a verified positive, retain them as READY_NOT_EXECUTED. Otherwise
activate the registered next batch after recording the wave2 outcome.

Use `data_next.source_locations($taskAssets, dataset)` to locate the already
existing metadata documents. The API requires explicit inputs: ETTm1 takes
`metadata_path` (N03 receipt), `manifest_path` (external file inventory) and
`master_manifest_path` (native frequency/bounds); Solar takes `metadata_path`
(continuation audit). It validates metadata/size before streaming only the
registered numeric prefix. Full source hashes are historical metadata, not
newly recomputed hashes over protected tail records.

ETTm1 has55744 admitted records ending exclusively at2018-02-01 16:00. Solar
has42048 admitted records. Its int64 time coordinates are elapsed600s from0;
they have no calendar date, UTC or timezone interpretation. All source channels
remain present; the historical8-channel Solar subset is not used.

The `train`, `baselines` and `costs` stages can also be invoked individually.
Checkpoint/RNG/optimizer identities and complete matching results are reused;
source/data/selection or environment mismatches require inspection. Never
erase an attempt or restart training merely because a log observer timed out.
The private `PROCESS_ATTEMPTS.jsonl` identifies the live process and each stage.

Every dataset has three widths times two B2 seeds, two Direct MLP seeds and one
zero-shot Raw identity. Both DEFAULT128 and COMMON_CAP are measured separately,
18 cost routes per dataset. Raw/Direct accuracy is not counted twice when its
same prediction identity appears in both cost facets. Each route validates
whole-V parity and runs ten warmups plus twenty timed trials in a fresh process.

Focused CPU checks:

```powershell
& $taskPython -m pytest -q tests/test_b2_dataset_search_data.py `
  tests/test_b2_dataset_search_training.py tests/test_b2_dataset_search_costs.py `
  tests/test_b2_dataset_search_run.py
```

A positive development result triggers full independent prediction/cost checks
and a three-width accuracy/memory/time table resembling the reference slide.
Negative batches remain visible and lead to the next pre-registered dataset
batch. No result is an independent TEST claim. No commit/push is part of this
request. The previous two-candidate extension and original dirty checkout are
guarded and remain separate.
