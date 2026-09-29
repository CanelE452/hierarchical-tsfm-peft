# V9 reproduction and evidence map

Run from the repository root with the existing `.venv/Scripts/python.exe -B`. `provenance.json` binds the approval, actual plan, protocol and base commit. `reuse_manifest.json` binds the existing data, parent checkpoints, untrained shared head initialization, backbone revision and baseline predictions. No raw data, backbone weights or prediction arrays are published.

The completed dataset/seed/family settings are the eight entries of `protocol.json`. Each `runs/<id>/result.json` links the actual command/environment, initial/selected/best-trained/restart states, parent, origin order, curve, VAL predictions, alpha choice and hashes. `.cache/tsfm_peft_staged_p_v9_20260929/` contains local checkpoints and arrays. Both parent datasets must already be available; no automatic substitute or download is provided.

The executed sequence was preparation, one failed zero-update synthetic checker and its corrected ten-update session, initial states, Robin/Jena four-fit launchers, both selections, both evaluation seals, both evaluations, analysis, figures, conditional Jena-only cost, reporting and saved-artifact verification. All computational jobs are reserved in `ledger.json`. Failed jobs remain charged. Nested real-fit elapsed time is not added again to the parent GPU-job clock.

The following are command forms documenting this execution, **not instructions to rerun the exposed experiment in place**. Exact names and commands reside in receipts; `run_fits_v9.py` rejects execution after either TEST exposure, and output guards preserve existing artifacts.

```powershell
.venv/Scripts/python.exe -B research/tsfm_peft_staged_p_v9_20260929/prepare_reuse_v9.py
.venv/Scripts/python.exe -B research/tsfm_peft_staged_p_v9_20260929/run_fits_v9.py --dataset robin --reserve-seconds 1200
.venv/Scripts/python.exe -B research/tsfm_peft_staged_p_v9_20260929/run_fits_v9.py --dataset jena --reserve-seconds 1200
.venv/Scripts/python.exe -B research/tsfm_peft_staged_p_v9_20260929/cost_v9.py --dataset jena --phase parity --name cost_jena_parity02 --reserve-seconds 400
.venv/Scripts/python.exe -B research/tsfm_peft_staged_p_v9_20260929/cost_v9.py --dataset jena --phase gpu --name cost_jena_gpu01 --reserve-seconds 1800
```

`selected_<dataset>.json`, `evaluation_seal_<dataset>.json` and `test_exposure_<dataset>.json` establish the selection/exposure order. `<dataset>_eval01.json` links every reused and new prediction and records masked macro scores, complete-target P/Q diagnostics and paired within-period block resampling. `analysis01/` contains compact CSVs; `report_values.json` collects traceable quantities for the final documents and links `training_summary.csv` for historical parent and incremental head costs. `numeric_claims.json` binds rounded document values to source JSON. `figures01.json` binds the inspected PNG/SVG accuracy figures.

`checker_correction01.json` preserves the original synthetic-reference mismatch. `cost_checker_correction01.json` preserves the cost-replay failure, VAL-only diagnostic and common execution-context correction. Neither changed fitted model code, checkpoint selection, data, TEST outputs or tolerances. Same-session timing uses online parent forwards and all declared output channels; no training forecast cache substitutes for inference.

`verify_v9.py` reconstructs metrics from saved predictions and checks checkpoint states, selections, hashes, budget and timing aggregation. It does not instantiate a model, run inference, fit parameters or claim independent reproduction. A future independently authorized replication requires a new output/ledger scope and the same pinned inputs; do not delete exposure guards to rerun this completed round.

`publication_receipt.json` will identify the verified artifact commit on origin/main. A subsequent metadata-only commit may publish that receipt; its live SHA is reported separately without a self-referential hash claim.
