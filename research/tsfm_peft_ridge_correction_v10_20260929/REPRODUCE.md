# Executed commands and reuse boundaries

Run from the repository root using the existing Windows `.venv/Scripts/python.exe -B`. No packages or data were downloaded. PLAN/APPROVAL and protocol.json define the experiment; reuse_manifest.json binds source data/parents, parent_cache.json binds TRAIN/VAL forecasts, each runs/<id>/result.json binds solved weights and predictions, and selection_seal/test_exposure bind evaluation order. Weights/raw/arrays stay in local ignored cache and are not published.

The commands below document this completed run. Do not rerun into these sealed output names: guards preserve existing attempts and prohibit fitting after exposure. A separate authorized reproduction needs a separate output scope and its own ledger; the reader's cache paths must be resolved from manifests rather than guessed.

```powershell
.venv/Scripts/python.exe -B research/tsfm_peft_ridge_correction_v10_20260929/runtime_ridge.py
.venv/Scripts/python.exe -B research/tsfm_peft_ridge_correction_v10_20260929/prepare_ridge.py reuse --job reuse01
.venv/Scripts/python.exe -B research/tsfm_peft_ridge_correction_v10_20260929/prepare_ridge.py parents --job parents01 --reserve-seconds 1000
.venv/Scripts/python.exe -B research/tsfm_peft_ridge_correction_v10_20260929/run_ridge.py check --job synthetic_check01
.venv/Scripts/python.exe -B research/tsfm_peft_ridge_correction_v10_20260929/seal_ridge.py static --job static01
.venv/Scripts/python.exe -B research/tsfm_peft_ridge_correction_v10_20260929/run_ridge.py fit --job ridge_fit01 --reserve-seconds 1200
.venv/Scripts/python.exe -B research/tsfm_peft_ridge_correction_v10_20260929/seal_ridge.py online --job online01
.venv/Scripts/python.exe -B research/tsfm_peft_ridge_correction_v10_20260929/seal_ridge.py expose --job seal01
.venv/Scripts/python.exe -B research/tsfm_peft_ridge_correction_v10_20260929/evaluate_ridge.py --dataset robin --name robin_eval01 --reserve-seconds 250
.venv/Scripts/python.exe -B research/tsfm_peft_ridge_correction_v10_20260929/evaluate_ridge.py --dataset jena --name jena_eval01 --reserve-seconds 300
# Previous command failed at Windows ledger replacement; preserve its arrays/partial/error.
.venv/Scripts/python.exe -B research/tsfm_peft_ridge_correction_v10_20260929/seal_ridge.py static --job static02
.venv/Scripts/python.exe -B research/tsfm_peft_ridge_correction_v10_20260929/evaluate_ridge.py --dataset jena --name jena_eval02 --reserve-seconds 250 --reuse-partial research/tsfm_peft_ridge_correction_v10_20260929/jena_eval01_partial.json
.venv/Scripts/python.exe -B research/tsfm_peft_ridge_correction_v10_20260929/summarize_ridge.py
.venv/Scripts/python.exe -B research/tsfm_peft_ridge_correction_v10_20260929/cost_ridge.py --dataset jena --job jena_cost01 --reserve-seconds 600
# Run while the above cost job is active; only supervises that verified job's approved ceiling.
.venv/Scripts/python.exe -B research/tsfm_peft_ridge_correction_v10_20260929/guard_cost.py
.venv/Scripts/python.exe -B research/tsfm_peft_ridge_correction_v10_20260929/final_verify.py
```

Final verification/publication commands and receipts are separate from numerical experiments. CPU fitting is deterministic conditional on the two already-fitted parent seeds. Failed/interrupted execution consumes wall-clock budget; no new fit was needed to repair the evaluation writer. Parent cache is an optimization for fitting/evaluation only. Cost always includes live TSFM+adapter+ridge computation and complete CPU output.
