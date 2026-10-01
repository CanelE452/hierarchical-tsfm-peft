# Reproducing the v14 run record

This note is a reproduction guide for `research/tsfm_peft_nonlinear_correction_v14_20261001`. It is based on the scripts and manifests currently in this folder. It is not a claim of independent confirmation: the protocol itself records four exposed development datasets and `test_used_for_selection: false`, followed by a sealed TEST exposure.

Do not rerun the commands blindly in this directory. The current directory already contains completed fitting records and a shared ledger. Use the existing files for inspection. A fresh reproduction must reestablish a fresh output declaration, provenance record, and ledger in a clean output area with the same prior v6/v7/v11/v12/v13 artifacts available.

## Current authoritative state

The fixed protocol is `protocol.json`:

- schema: `v14_nonlinear_correction_protocol_1`
- base commit: `58724610383920f40ddecd3573c6c9ec7c6771a5`
- datasets: `robin`, `jena`, `hog`, `peacock_education`
- seeds: `92601`, `92602`
- families: `level_gelu`, `level_linear`, `direct_gelu`
- prescribed fits: 48
- scope: exposed development data; no independent confirmation

The current ledger records 48 completed `neural_fit` jobs and 48 `runs/*/result.json` files. `training_diagnosis.json` is complete with 48 fit rows and 24 selected seed rows. `selection_seal.json` is sealed, and `test_exposure.json` records the first new evaluation exposure for all four datasets.

Prediction, scoring, Jena GPU132/CPU24 cost rows, reporting and all three saved-artifact verification groups are complete. `prediction_manifest.json`, `evaluation01.json`, `cost_summary.json`, `final_checks.json`, and `report_values.json` are the authoritative outputs. Publication is a separate status recorded only after a live remote SHA check. Reading the changing ledger's earlier hash in `training_diagnosis.json` describes that stage's state, not an immutable final-ledger receipt.

## Fixed hashes and bindings

The selection seal binds the main inputs and source code:

```text
selection_seal.json status: sealed
protocol.json sha256:      e33ccc9016e56056beaeb55c83d91fd9c0d6dde160c67dc129f8fa55cb6f1e53
selected.json sha256:      b474606520196dd53d69bafe16d545a513419ba351f17b238e2e9ce39b84fcac
data_manifest.json sha256: a4e5c07c355a5946619a1847b10999cde2046965fda11cbc218cd9cc62aaef30
test exposure seal sha256: 3c93683cf66cc68658fd5b29e18e027ac7e2c03085d29e76ad2cf447992d963c
```

`data_manifest.json` binds the reused v13 train/validation and test archives. The training code uses `trainval.npz`; TEST loading requires `selection_seal.json` and `test_exposure.json`.

```text
dataset            C   K   trainval sha256                                                   test sha256
robin              17  5   2c2d6e1312ad53f61ff93df5a417bcc5f614c31aa0eebef8a1da68569067fb61  5f0c615ecbd7c2f16cb12f3c96e3ef2edb442cd348826b0db4d4cce318a4af25
jena               21  6   74f8e7487ba55a71a9bfb26224b756921723dddb91165094a71b6f92200f1fb9  f4b12a44e6d64c20fa332bc22b8b39895f551874af7d28e99e23e9c2f4d829a1
hog                32  8   920f9b7399c497d405cb62afbee0eaef1bb5cb29188b9d5fafd850d856e657b7  a060aa49e97dc87f232924c5f20711cf3fa9384244540e24d654e810a3335811
peacock_education  13  4   bf8ce1c72ef8b1644d4148cd25adbfff959b4ca2a7564110132c2d95844dfb2e  e5e76457afd45d4bd45cb6bad5a4c131b177c6fd67f904db0f62e6fc9585ee85
```

Initial G receipts are also in `data_manifest.json` and `initial_manifest.json`. The down tensor is copied from the original untrained G; the up tensor is the original zero tensor. Representative initial receipts:

```text
dataset            seed 92601 sha256                                                seed 92602 sha256
robin              45b6b98f1d594e374ff7246303da0bc3de2d5492293c2cb34d0156d0bf4791f8  d69136b8a392a50df88f8d70cc5b090f604e5e3f04d5dd2a1664600c3f12eff0
jena               6e115b963fde71f8eee96617c2bd2aef98a61ecdb10f4f4651924c295baf3c26  544696ab7ffd485d519afdbffe559fffd9bbd1276c996ac94f82f27b1057ac5c
hog                40454908818ee2781b1e8b38cc818d06ed87b7d2130fb343e26338e583ff19dd  8e39e62b5a63b16b18def9258d8b0259150c503c0b987d2529f5624505f8d467
peacock_education  e6bbb3a1f18e23052ce39aaa3dddbc0607b371408a9de60ef25aee33468aecd6  5f2449954a2b3c331b2328aea73a28ca25cb38ce6230114bfb4dc0d90a409ffa
```

## Existing completed run versus fresh reproduction

For the existing completed run, inspect these files rather than starting jobs:

- `ledger.json` for job accounting and source snapshots
- `parent_cache.json` for frozen parent TRAIN/VAL cache receipts
- `runs/*/result.json` and `runs/*/curve.json` for each neural fit
- `selected.json` for the mean-seed VAL selection
- `training_diagnosis.json` for post-hoc TRAIN-probe and VAL-curve summaries
- `selection_seal.json` and `test_exposure.json` for selection and TEST exposure binding
- `prediction_manifest.json`, `evaluation01.json`, and `cost_decision.json` for completed TEST prediction, scoring, and cost-eligibility decisions

The existing directory should not be used for a fresh rerun of the fit matrix. `runtime_v14.Job` rejects duplicate job labels and duplicate valid fit attempts, the scripts refuse overwriting key outputs, and `runtime_v14.Job` rejects new `neural_fit` jobs after `selection_seal.json` or `test_exposure.json` exists. Changing only job labels is not a valid way to redo sealed fits. A fresh reproduction should start from a clean v14 output area with the same prior artifacts, a fresh output declaration, a fresh provenance record, and a new ledger.

## Script sequence for a fresh reproduction

The commands below show the intended order and the job reservations recorded or planned for this run. Do not run the fit commands in the current sealed directory.

Prepare manifests:

```powershell
./.venv/Scripts/python.exe -B research\tsfm_peft_nonlinear_correction_v14_20261001\prepare_v14.py --job prepare01 --reserve-s 150
```

Optional synthetic/source sanity check:

```powershell
./.venv/Scripts/python.exe -B research\tsfm_peft_nonlinear_correction_v14_20261001\check_v14.py --job synthetic01 --reserve-s 90
```

Build frozen parent TRAIN/VAL caches:

```powershell
./.venv/Scripts/python.exe -B research\tsfm_peft_nonlinear_correction_v14_20261001\train_v14.py --stage cache --label parent_cache01 --reserve-s 1200
```

Run the prescribed 48 fits:

```powershell
./.venv/Scripts/python.exe -B research\tsfm_peft_nonlinear_correction_v14_20261001\train_v14.py --stage fit --label training01 --reserve-s 4000
```

The fit stage runs all protocol fits when `--fit-id` is omitted. A single fit can be targeted with `--fit-id`, but completed fits are reused only if their `result.json` passes `result_for(spec)` and ledger checks. Technical retries require `--retry-tag`, `--retry-reason`, and a failed prior attempt; they are not performance retries.

After all fits finish, write the metadata-only training diagnosis:

```powershell
./.venv/Scripts/python.exe -B research\tsfm_peft_nonlinear_correction_v14_20261001\training_diagnosis_v14.py --job training_diagnosis01 --reserve-s 120
```

Seal joint selection:

```powershell
./.venv/Scripts/python.exe -B research\tsfm_peft_nonlinear_correction_v14_20261001\evaluate_v14.py seal --job seal01 --reserve-s 90
```

Generate TEST predictions only after the seal:

```powershell
./.venv/Scripts/python.exe -B research\tsfm_peft_nonlinear_correction_v14_20261001\evaluate_v14.py predict --job prediction01 --reserve-s 300
```

Score only after `prediction_manifest.json` is complete:

```powershell
./.venv/Scripts/python.exe -B research\tsfm_peft_nonlinear_correction_v14_20261001\evaluate_v14.py score --job scoring01 --reserve-s 600
```

Cost measurement is conditional on `evaluation01.json`:

```powershell
./.venv/Scripts/python.exe -B research\tsfm_peft_nonlinear_correction_v14_20261001\cost_v14.py decision --job cost_decision01 --reserve-s 60
./.venv/Scripts/python.exe -B research\tsfm_peft_nonlinear_correction_v14_20261001\cost_v14.py gpu --job cost_gpu01 --reserve-s 1800
./.venv/Scripts/python.exe -B research\tsfm_peft_nonlinear_correction_v14_20261001\cost_v14.py cpu --job cost_cpu01 --reserve-s 300
```

After complete evaluation and cost files, generate the report first; the verifier requires its tables and figure hashes:

```powershell
./.venv/Scripts/python.exe -B research\tsfm_peft_nonlinear_correction_v14_20261001\report_v14.py --job report01 --reserve-s 300
./.venv/Scripts/python.exe -B research\tsfm_peft_nonlinear_correction_v14_20261001\verify_v14.py --job verify01 --reserve-s 180
```

## Training cache versus deployment

Training uses a frozen parent forecast cache for TRAIN and VAL only. The cache is recorded in `parent_cache.json`, and each cache entry stores the parent row, source/data binding, train and val NPZ receipts, online/cache parity, and frozen-parent state digest. Fit results record `cached_parent: true` and `deployment: online frozen parent once, no prediction cache`.

Deployment and TEST prediction do not use the TRAIN/VAL parent cache. `evaluate_v14.py predict` restores each selected v14 checkpoint with `model_v14.restore_model(..., load_parent=True)`, which attaches the online frozen parent and runs the full model on TEST origins. Reference models may reuse their old prediction receipts, but new v14 selected models run online.

## Resource limits and guards

The hard limits are in `runtime_v14.py`:

```text
real_fit: 52
neural_basic: 48
technical_reserve: 4
gpu_seconds: 7200
cpu_analysis_seconds: 1800
cpu_check_seconds: 900
storage_bytes: 5 GiB
download_bytes: 0
```

`Job` records source snapshots in `.cache\tsfm_peft_nonlinear_correction_v14_20261001\source_snapshots`. It rejects an unclosed same-category job, duplicate labels, duplicate valid fits, invalid technical retries, and fitting after selection or TEST exposure.

## Selection already made in this run

The existing `selected.json` has schema `v14_joint_selection_1`, `test_used_for_selection: false`, `matched_initial_groups: 8`, and `matched_schedule_groups: 208`. The criterion is earliest exact minimum complete VAL MSE including epoch0 per fit, with shared LR chosen by mean two-seed best VAL MSE and exact LR ties going to `1e-4`.

Selected new families:

```text
dataset            family        selected lr  selected epochs  parent role
robin              level_gelu    1e-4         0,0              level
robin              level_linear  1e-4         0,0              level
robin              direct_gelu   1e-3         7,21             direct
jena               level_gelu    1e-4         29,35            level
jena               level_linear  1e-4         15,16            level
jena               direct_gelu   1e-3         37,17            direct
hog                level_gelu    1e-4         12,14            level
hog                level_linear  1e-3         12,19            level
hog                direct_gelu   1e-3         9,18             direct
peacock_education  level_gelu    1e-4         0,0              level
peacock_education  level_linear  1e-4         0,0              level
peacock_education  direct_gelu   1e-3         14,11            direct
```

An epoch0 selection means the original frozen parent fallback was selected by VAL. It does not mean no optimization updates occurred in that fit. `training_diagnosis.json` distinguishes `step0_exact_parent_fallback` from `learned_selected` and reports TRAIN-probe deltas separately from VAL deltas. The TRAIN probe is a fixed saved probe, not full TRAIN scoring.

## Plan erratum and claim boundary

`PLAN_ERRATUM.md` corrects the motivation label for the v13 P/Q diagnosis: that diagnosis used a common GROUP basis, not PCA coordinates. The v14 parent and stored basis remain the original fixed PCA LEVEL policy where specified by `protocol.json`, `model_v14.py`, and `data_manifest.json`.

The method is a bounded exposed-development experiment, not a proof of novelty, causality, or independent generalization. The nonlinear head is compared to a matched linear control and to a direct NLinear parent system, but those comparisons do not isolate every function-space and optimization effect.

## Final status files

Use these files as authority when they exist:

- predictions: `prediction_manifest.json`
- accuracy: `evaluation01.json`, `evaluation01_models.csv`, `evaluation01_channels.csv`
- cost: `cost_decision.json`, `cost_summary.json`
- verification: the final verifier output written by `verify_v14.py`
- report/package values: `report_values.json`, `comparison.csv`, `paired_differences.csv`, `cost_comparison.csv`, `figures/*`

If a file is absent, do not infer its result from partial manifests, running jobs, or planned commands.

## Post-publication component attribution

POSTHOC_COMPONENT_PLAN.md fixes a saved-output-only diagnostic after the scientific v14 campaign. COMPONENT_DIAGNOSIS.md reports its interpretation; component_swap01.json and three CSVs retain all source/composition results. The rolling component_swap01_partial.json is preserved locally and excluded as a duplicate serialization, not as an unfavorable result.

The original commands were run once under the same ledger with no fits or model forward:

```powershell
.venv/Scripts/python.exe -B research/tsfm_peft_nonlinear_correction_v14_20261001/diagnose_component_swap_v14.py --job component_attribution01 --reserve-s 300
.venv/Scripts/python.exe -B research/tsfm_peft_nonlinear_correction_v14_20261001/verify_component_swap_v14.py --job component_verify01 --reserve-s 60
```

These commands intentionally reject completed output or reused job labels. Reproduction must preserve the original results/ledger and use a separately recorded output/run namespace; do not delete prior evidence to rerun in place. No new reproduction is included in the completed computation. The checker recomputes point metrics/algebra and checks CI settings and formatting, but does not rerun bootstrap sampling. Sources and fixed canonical PCA bytes are bound in the result. This is post-hoc exposed-development analysis, not a new cheap model or independent confirmation.

## Original publication formatting record

The first reserved publication check failed only because default `git diff --cached --check` flags a final blank line in AUTHORIZATION.txt, PLAN.md, protocol.json and STATUS.md. The first three are immutable provenance inputs. Their original bytes and hashes are preserved, as is the failed publication_check01 traceback in the ledger; this is not a fitting, scoring or scientific verification failure.

The second publication check used an incorrect per-command whitespace override: it omitted the existing Windows `cr-at-eol` setting and did not explicitly disable `blank-at-eof`. That generated a verbose formatting-only failure. Its exact original UTF-8 traceback is preserved under the local cache `publication_errors/publication_check02_traceback.txt`, with path/hash/size and an excerpt in the ledger. The archive avoids publishing a34MB duplicate diagnostic, not a scientific result. publication_format_repair.json records the correction: `git -c core.whitespace=blank-at-eol,space-before-tab,cr-at-eol,-blank-at-eof diff --cached --check`. It checks trailing spaces and space-before-tab, preserves CRLF handling and allows the recorded EOF blank lines; no repository/global configuration or hook was changed. Complete raw cost aggregates retain all156 measured rows; duplicate per-row and four rolling serialization files remain locally intact and are excluded only from publication.
