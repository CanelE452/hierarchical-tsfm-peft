# Reproduction and scope

Run from the repository root with the existing `.venv/Scripts/python.exe -B`. This is a new bounded development campaign; it does not replace previous manifests, exposure records, failures, or budget violations. `PLAN.md` is the prospective contract, `AUTHORIZATION.txt` records the actual delegation, and `protocol.json` fixes every dataset, seed, coefficient fit and cost grid.

The scripts below have no automatic permission to repeat a valid fit. Read `STATUS.md`, `ledger.json`, actual processes and existing receipts first. A new label alone does not authorize duplicate fitting. The bootstrap is one-time and refuses to overwrite a contract or ledger. Local cache paths in the manifests are references to actual source bytes; downloading is outside this contract.

```powershell
.venv/Scripts/python.exe -B research/tsfm_peft_coordinate_decoder_v15_20261001/bootstrap_v15.py
.venv/Scripts/python.exe -B research/tsfm_peft_coordinate_decoder_v15_20261001/prepare_v15.py --job prepare01 --reserve-s 90
.venv/Scripts/python.exe -B research/tsfm_peft_coordinate_decoder_v15_20261001/check_v15.py --job implementation_check01 --reserve-s 30
.venv/Scripts/python.exe -B research/tsfm_peft_coordinate_decoder_v15_20261001/cache_v15.py --phase trainval --job latent_trainval01 --reserve-s 900
.venv/Scripts/python.exe -B research/tsfm_peft_coordinate_decoder_v15_20261001/fit_v15.py fit --job coefficient_fit01 --reserve-s 1000
.venv/Scripts/python.exe -B research/tsfm_peft_coordinate_decoder_v15_20261001/fit_v15.py select --job selection01 --reserve-s 60
.venv/Scripts/python.exe -B research/tsfm_peft_coordinate_decoder_v15_20261001/evaluate_v15.py seal --job selection_seal01 --reserve-s 60
.venv/Scripts/python.exe -B research/tsfm_peft_coordinate_decoder_v15_20261001/cache_v15.py --phase test --job latent_test01 --reserve-s 240
.venv/Scripts/python.exe -B research/tsfm_peft_coordinate_decoder_v15_20261001/evaluate_v15.py predict --job prediction01 --reserve-s 60
.venv/Scripts/python.exe -B research/tsfm_peft_coordinate_decoder_v15_20261001/evaluate_v15.py score --job evaluation01 --reserve-s 120
.venv/Scripts/python.exe -B research/tsfm_peft_coordinate_decoder_v15_20261001/cost_v15.py gpu --job cost_gpu01 --reserve-s 2400
.venv/Scripts/python.exe -B research/tsfm_peft_coordinate_decoder_v15_20261001/cost_v15.py cpu --job cost_cpu01 --reserve-s 120
.venv/Scripts/python.exe -B research/tsfm_peft_coordinate_decoder_v15_20261001/summarize_available_cost_v15.py --job available_cost01 --reserve-s 30
.venv/Scripts/python.exe -B research/tsfm_peft_coordinate_decoder_v15_20261001/report_v15.py --job report01 --reserve-s 120
.venv/Scripts/python.exe -B research/tsfm_peft_coordinate_decoder_v15_20261001/verify_v15.py --scope all --job final_check01 --reserve-s 180
```

These commands describe the intended sequence, not proof that every command has completed. Actual source snapshots, reservations, PIDs, durations, failures and outputs are recorded in `ledger.json` and receipts. The current interpreter, pinned model revision and library versions are captured by preparation/model jobs. No optimizer or new neural training is used; each independent real-data decoder solve still counts as a fitting attempt. CPU fitting children do not add their time again to the enclosing CPU-analysis job.

Actual terminal state:48 coefficient fits,84 forecast instances,606/612 GPU cost rows and48/48 CPU cost rows. The GPU guard prevented the final six Robin rows; `cost_available_summary.json` summarizes only datasets with every fixed row and preserves the complete raw record. `final_checks.json` passes saved-artifact verification but correctly keeps `complete_campaign_verified=false`. No missing measurement is fabricated or automatically retried. Visual inspection triggered a layout-only report revision (`report02`); the initial report bytes/source are preserved in `report_initial/` and mapped by `report_layout_archive.json`. Numerical inputs did not change.

The original full TRAIN/VAL DIRECT caches are reused; only the required K-series frozen TSFM predictions are generated. All forty-eight prescribed coefficient fits and common two-seed VAL selections precede the joint seal and new evaluation. Each dataset was exposed in earlier development, so the new seal is a sequencing record, not independent confirmation.

Accuracy uses the original masks and channel order. Periods are pooled by channel error sums/counts before the channel macro and seed-loss mean. Fallback is exactly the direct parent. Missing targets never become zero-valued regression observations. No cached forecast enters the timed deployment forward. Scalar coefficients stored as buffers are included in deployment tensor bytes even though `requires_grad=False` and `parameters()` exclude those buffers.

`coefficient_runs/*/{result,objective,receipt}.json`, selected coefficients and immutable local source snapshots establish fitting provenance. Large arrays/checkpoints remain in local caches. The final tables/figures retain adverse periods, controls, external baselines, small-batch executions and timing drift. Arithmetic checks and agent reviews are internal verification, not independent reproduction.
