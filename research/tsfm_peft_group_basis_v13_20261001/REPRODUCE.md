# V13 reproduction and artifact verification

The executable contract is [protocol.json](protocol.json), bound with [PLAN.md](PLAN.md) and [AUTHORIZATION.txt](AUTHORIZATION.txt) by [provenance.json](provenance.json). [METHOD_UPDATE.md](METHOD_UPDATE.md) defines the models; [TOPIC_DECISION.md](TOPIC_DECISION.md) interprets evidence. This guide records the actual CLI order and its boundaries, not authorization to rerun this completed selection or use unused technical attempts for new methods.

## Required environment and local sources

Run from the repository root using its `.venv/Scripts/python.exe`. The fit receipt records Python 3.11.16, torch 2.10.0+cu128, NumPy 2.4.6, SciPy 1.17.1, PEFT 0.21.0, chronos-forecasting 2.3.2, Transformers 5.17.0 and pandas 2.3.3. Actual executable, command, working directory and versions are retained in `runs/<fit>/result.json`; cost artifacts also retain their measured environment. CUDA uses FP32 with TF32 disabled; CPU numeric/timing work uses four torch threads. Do not substitute another installed Python environment silently.

Required inputs are the exact receipts in [reuse_manifest.json](reuse_manifest.json): existing standardized TRAIN/VAL and evaluation archives, original untrained LEVEL G for both seeds, selected reference checkpoints, stored reference predictions and their source results. Original v1-v12 files remain unchanged. V13 imports numeric/model primitives from `research/tsfm_peft_internal_vs_subspace_v11_20260930/` and evaluation/cost primitives from `research/tsfm_peft_a_confirmation_v12_20261001/`; it writes only its own ledger and outputs.

The backbone is the local `amazon/chronos-bolt-small` snapshot at `.cache/huggingface/models--amazon--chronos-bolt-small/snapshots/772f3d25d38aec6d914c8949dab4462e2d46f5d8`. Downloads are outside this campaign's zero-download allowance. Raw arrays, copied initial G, main caches, checkpoints and new prediction arrays live under ignored `.cache/tsfm_peft_group_basis_v13_20261001/`; old referenced caches are also required. Public JSON/CSV/source files alone are insufficient to repeat the computation. Receipts include absolute paths, so relocation requires an explicitly separate provenance record rather than editing the original receipts.

## Recorded execution order

The commands below use the original job labels and reservations visible in [ledger.json](ledger.json). They document the historical sequence; **do not replay them against the existing output directory**. Labels are unique, preparation refuses overwrite, and fitting is prohibited after the selection seal or exposure. The CLIs have no independent output-root switch. A separate reproduction must preserve this campaign and establish its own output/provenance/accounting context first; deleting the ledger or seal is not a reproduction procedure.

```powershell
$v13Python = '.venv/Scripts/python.exe'
$v13Folder = 'research/tsfm_peft_group_basis_v13_20261001'

& $v13Python -B "$v13Folder/check_v13.py" --job v13_synthetic01 --reserve-s 90
& $v13Python -B "$v13Folder/prepare_v13.py" prepare --job v13_group_basis_prepare_01 --reserve-s 150
& $v13Python -B "$v13Folder/train_v13.py" --stage cache --label v13_cache01 --reserve-s 1800
& $v13Python -B "$v13Folder/train_v13.py" --stage fit --label v13_fit01 --reserve-s 3600
& $v13Python -B "$v13Folder/evaluate_v13.py" seal --job v13_seal01 --reserve-s 60
& $v13Python -B "$v13Folder/evaluate_v13.py" predict --job v13_predict01 --reserve-s 600
& $v13Python -B "$v13Folder/evaluate_v13.py" score --job v13_score01 --reserve-s 600
& $v13Python -B "$v13Folder/cost_v13.py" decision --job v13_cost_decision01 --reserve-s 30
& $v13Python -B "$v13Folder/cost_v13.py" gpu --job v13_cost_gpu01 --reserve-s 1800
```

`preflight01.json` is a separately reserved source/import/receipt check recorded as `v13_preflight01` while caching was active; it has no standalone CLI. The accounting reconciliation also has its own receipt. Do not invent corresponding runner commands. The stage logs (`cache01.*.log`, `fit01.*.log`, `predict01.*.log`, `score01.*.log`, `cost_gpu01.*.log`) supplement the ledger and artifact status; a warning-free log or exit alone is not the completion criterion.

Preparation computes the four TRAIN-only group bases and writes `data_manifest.json` plus dataset contracts, preserving original PCA bases as `pca_basis`. It copies only verified original untrained G. Caching computes frozen main predictions for all TRAIN/VAL origins, recording source/data bindings and initial online/cache parity in `main_cache.json`. Both paired families use the same cache and original G. Deployment and costs always execute F0 online.

Fitting follows the sixteen `protocol.fits` entries in their fixed order, with LR 1e-3, seeds 92601/92602, batch 4, 512 phase-balanced origins, maximum 120 epochs and patience 6. Epoch zero is eligible; complete VAL masked channel-macro MSE selects the earliest exact minimum. `selected.json` is written only once all prescribed fits are valid. The GPU parent job contains nested `neural_fit` attempts: update/attempt counts are separate, and the nested elapsed time is not added again to GPU time.

The joint seal binds all four datasets, selected checkpoints and evaluation sources before `predict` creates `test_exposure.json` and new predictions. Sixteen new model instances and forty-four existing reference instances form sixty saved predictions. `score` checks reference replay and emits `evaluation01.json` and its model/channel CSVs. Period error sums/counts are pooled per channel before channel averaging, then seed losses are averaged; predictions are never ensembled.

## Cost, diagnosis and reporting

The predeclared cost trigger is operational, not a significance test: at least one normally trained GROUP_RES must have lower pooled MSE than PCA LEVEL. If triggered, all four datasets remain. The GPU grid contains 432 core rows and 48 F0 sentinels; each uses the same first 24 VAL origins, batch 1 or 4, 10 warmups, three timed passes and three blocks. Full-MSE LoRA includes the fixed K/4K row-chunk controls. Timed work runs from standardized CPU input to the complete original-channel CPU forecast; loading, hashing, scoring and ledger work are outside the timed interval. Keep raw passes, three blocks, sentinels and parity receipts. Three passes do not establish a stable speed ranking for small differences.

After GPU timing has ended, the actual remaining CLIs are:

```text
cost_v13.py cpu --job v13_cost_cpu01 --reserve-s 300
diagnose_space_v13.py --job v13_space_diagnosis01 --reserve-s 300
report_v13.py --job v13_report01 --reserve-s 120
verify_v13.py --job v13_verify01 --reserve-s 180
verify_space_v13.py --job v13_space_verify01 --reserve-s 120
```

These commands also completed, with the same Python and folder prefix; preserve their recorded unique labels rather than rerunning them. Timing was isolated from concurrent computational checks/analysis. The diagnostic specified in [DIAGNOSIS.md](DIAGNOSIS.md) reuses saved predictions in the common new GROUP space and reports complete-target-vector P/Q energies and coverage. It creates no predictor and cannot replace the primary masked MSE. Its output is `space_diagnosis01.json`. Supplementary `verify_space_v13.py` checks receipts, original coverage, basis dtype/hash/Gram, energy algebra and count-weighted period aggregation without regenerating predictions or reprojecting the full arrays. Both artifact verifiers passed. `interpretation_values.json` is a compact extraction of cost groups and mean seed space energies, with input receipts; it adds no fitted model or forecast.

Terminal cost status is in `cost_cuda_rows.json`, `cost_cuda_summary.json` and `cost_summary.json`, with optional CPU equivalents. `partial` or `budget_stopped` must remain incomplete. Reporting writes `report_values.json` and figure receipts. Actual final elapsed time and terminal job/PID state must be read from the terminal ledger and `final_checks.json`, not from the reservations above. Scientific completion is separate from the repository publication receipt/live commit record.

The publication includes complete `cost_cuda_rows.json` and `cost_cpu_rows.json` with every raw pass and parity receipt. Duplicate per-row recovery files in `cost_rows/` and successful rolling `*_partial.json` copies remain unchanged locally but are ignored for publication. This removes duplicated serialization, not unfavorable rows or failed attempts. Original accounting deviations remain public.

## Immutability, recovery and verification scope

The shared caps are 16 basic predictive fits plus at most 4 technical attempts (20 total), GPU 7,200 seconds, CPU analysis 7,200 seconds, CPU checks 900 seconds, 5 GiB new storage, zero downloads, and at most six synthetic sessions of at most ten updates each. The cost campaign has its additional 1,800-second GPU bound. GPU conflicts are observed without terminating other work; the fixed wait bounds are in the protocol. Unused resources are not extra search allowance.

Cache bindings include model/train/runtime source hashes; changing those after caching invalidates reuse. Preserve original attempts, selected states, seals and exposure. An interrupted fit requires a new GPU job label, one prescribed `--fit-id`, `--retry-tag` and concrete `--retry-reason`; optional `--resume-from` accepts only the matching complete-epoch `last.pt` before a terminal stopping boundary. The checkpoint restores model, optimizer, scheduler and RNG after VAL/scheduler processing. Failed incomplete updates remain consumed; restored steps and new-attempt updates are recorded separately. `replacements/<canonical_id>.json` identifies technical replacements. A valid completion cannot be retried for performance, and no fit may follow sealing/exposure.

`verify_v13.py` verifies selected fit/ledger receipts, initial-G and overlapping schedule hashes, recorded update/freeze/replay checks, minimum-VAL selection and seal ordering. It independently reaggregates saved arrays against original origins/masks for MSE, MAE, bias, channels, periods and seed means, then checks CSV/report consistency, cost grid/raw timing arithmetic and stored deployment parity receipts. It performs zero model calls and zero optimizer updates. It does **not** rerun the fits, regenerate deployment parity or bootstrap intervals, independently validate scientific generalization, or certify publication. Earlier final-check output is preserved if a later verification replaces it. A passed check on an explicitly budget-stopped cost record does not make that grid complete.

[accounting_deviation01.json](accounting_deviation01.json) discloses 5.1755 seconds of agent syntax/CLI/manifest checks that were not pre-reserved. The charge uses reported whole-tool wall times; exact absolute start/end times and Python executable were not recorded. It involved no model fitting, optimizer updates, new basis, predictions or scores and is charged retrospectively in the ledger. This campaign must not be described as having flawless pre-reservation compliance. All verification here is self-verification, not independent reproduction.
