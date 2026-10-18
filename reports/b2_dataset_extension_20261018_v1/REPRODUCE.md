# Reproduce or resume

Run from the checkout root with the pinned Python3.11/OpenBLAS interpreter and
the existing local Chronos-Bolt-small snapshot. Installed versions are recorded
in `ENVIRONMENT.json`. Keep one GPU stage active at a time. These commands do
not download data or model weights.

```powershell
$taskPython = '<PINNED_PYTHON_EXE>'
$taskAssets = '<ORIGINAL_ASSET_ROOT>'
$taskModel = '<PINNED_BOLT_SNAPSHOT_DIRECTORY>'
```

Prepare only the two previously isolated FIT_TUNE payloads; their source
contracts must already exist. This command refuses to overwrite an existing
prepared data directory. Skip it when resuming prepared data.

```powershell
& $taskPython -m research.b2_dataset_extension_20261018_v1.data `
  --asset-root $taskAssets `
  --output-root .cache/b2_dataset_extension_20261018_v1/data `
  --public-output reports/b2_dataset_extension_20261018_v1/DATA_DIAGNOSTICS.json `
  --dataset all
```

Run the three GPU stages sequentially. The first stage performs all four B2
searches, seals them together and only then performs four fresh A60 refits/V
evaluations. The baseline stage performs four fixed DirectG searches/refits.
The cost stage launches ten separate measurement processes.

```powershell
& $taskPython -m research.b2_dataset_extension_20261018_v1.run train --model-dir $taskModel
& $taskPython -m research.b2_dataset_extension_20261018_v1.run baselines --model-dir $taskModel
& $taskPython -m research.b2_dataset_extension_20261018_v1.run costs --model-dir $taskModel
```

An interrupted training stage resumes checkpoints with the same source,
configuration, data, optimizer, sampler and RNG identities. Do not edit those
sources or reset a failed attempt. Complete matching cost receipts are reused
only with their private prediction file SHA; incomplete cost attempts require
inspection and a recorded repair before retrying. Existing attempts are never
silently overwritten.

The actual first training launch directly called `train.execute_many` with the
same ordered two inputs (SML2010, Household) and the same model snapshot; it is
scientifically identical to the `run train` command above. Its private log is
`cache://TRAINING_ATTEMPT_001.log`. Subsequent stages use the coordinator CLI.

The first ten cost measurements forced `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
They remain available under `cost_attempts/workspace4096_001` as auxiliary
measurements. The final ten measurements inherit the current parent environment
(unset for this run), matching the historical coordinator's inheritance behavior.
The actual historical Jena environment value was not recorded. The amendment,
old source snapshot, both logs and all predictions are preserved; no model was
retrained. See `MEASUREMENT_AMENDMENT_001.json` and
`ENVIRONMENT_AMENDMENT_001.json` for the exact source and environment bindings.
The initial `ENVIRONMENT.json` records the initial cost source, while the
amendment records the final cost source. Keep the parent setting unchanged when
resuming final measurements; each receipt records its actual value.

Validate and render only after all stages complete. The renderer independently
checks private arrays, training selections, checkpoints, whole-V parity and raw
timing statistics before writing the five comparison artifacts.

```powershell
& $taskPython -m research.b2_dataset_extension_20261018_v1.analysis --check-only
& $taskPython -m research.b2_dataset_extension_20261018_v1.analysis
& $taskPython -m research.b2_dataset_extension_20261018_v1.analysis `
  --output-dir .cache/b2_dataset_extension_20261018_v1/replay
```

The five replay artifacts must match the public bytes and the SHA256 entries in
`VALIDATION.json`. An initial rendering with an overlapping SML2010 label, its
successful replay, and its source snapshot are retained privately under
`analysis_attempts/001` and `source_snapshots/analysis_attempt_001.py`. The final
render only changes label placement and adds the explicit outcome paragraph;
all comparison numbers are unchanged.

Focused CPU checks use synthetic fixtures and approved prefix metadata:

```powershell
& $taskPython -m pytest -q tests/test_b2_dataset_extension_data.py `
  tests/test_b2_dataset_extension_training.py tests/test_b2_dataset_extension_costs.py `
  tests/test_b2_dataset_extension_baselines.py
```

Do not run unrestricted legacy tests: `tests/test_model_parity.py` launches an
unrelated old GPU smoke and `tests/test_baselines.py` reads unrelated old data.
Their exclusion protects existing campaigns; it is unrelated to new results.

Public files contain aggregates, identities and figures only. Prefix arrays,
model checkpoints, full forecasts, worker logs and local paths stay in the
ignored `.cache/b2_dataset_extension_20261018_v1` directory. Neither old EVAL
payloads nor TEST arrays are required for any stage.
