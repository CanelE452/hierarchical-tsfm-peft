# Reuse and execution receipts

Run from the repository root with the existing `.venv/Scripts/python.exe`; no installation or download was performed. These are the actual commands, not instructions to repeat completed work blindly. Read STATUS/ledger and existing outputs first. The analysis scripts reject existing attempt names to preserve records; any technical rerun needs a fresh recorded name.

```text
.venv/Scripts/python.exe research/tsfm_peft_a_confirmation_v12_20261001/q_contribution_v12.py --name q_contribution02
.venv/Scripts/python.exe research/tsfm_peft_a_confirmation_v12_20261001/q_uncertainty_v12.py --name q_uncertainty01 --source q_contribution02
.venv/Scripts/python.exe research/tsfm_peft_a_confirmation_v12_20261001/equivalence_v12.py --stage cpu --job equivalence_cpu01 --reserve-s 120
.venv/Scripts/python.exe research/tsfm_peft_a_confirmation_v12_20261001/equivalence_v12.py --stage gpu --job equivalence_gpu01 --reserve-s 900
.venv/Scripts/python.exe research/tsfm_peft_a_confirmation_v12_20261001/equivalence_v12.py --stage report --job equivalence_report01 --reserve-s 60
.venv/Scripts/python.exe research/tsfm_peft_a_confirmation_v12_20261001/render_v12.py --source q_contribution02.json
.venv/Scripts/python.exe research/tsfm_peft_a_confirmation_v12_20261001/report_v12.py
.venv/Scripts/python.exe research/tsfm_peft_a_confirmation_v12_20261001/verify_v12.py
```

The first Q attempt and UTF-8 helper repair remain in REPAIRS.md and the ledger. Q scripts never load the TSFM; equivalence alone restores actual A models for bounded backward with optimizer zero. Time reservations and source snapshots belong to the isolated v12 ledger/cache. Original runtime modules are imported under explicit unique names for read-only loading/scoring; no old ledger is reused.

reuse_manifest.json links the six A/LEVEL states, data arrays, channel order, origins and actual hashes. Derived prediction arrays and bootstrap draws remain under the ignored `.cache/tsfm_peft_a_confirmation_v12_20261001/`. Public tables link their hashes but do not publish weights, raw data or prediction arrays.

At the original published checkpoint, cost values were historical v11 evidence with their campaign identity in cost_reference.json, and the missing confirmation contract prevented further execution. This is a historical boundary; the prospectively delegated continuation below supersedes that waiting state without changing its record.

## Prospectively delegated confirmation

AUTHORIZATION_SUPPLEMENT.txt preserves the user's delegation. PLAN_SUPPLEMENT.md, confirmation_protocol.json and confirmation_binding.json fix Peacock Education C13/K4 before fitting. No raw or backbone download is needed. Read STATUS and the ledger before any invocation: these commands are a reproduction record, not permission to duplicate a running or completed job. All real retries consume separate attempts.

The prescribed fit stages are:

```text
.venv/Scripts/python.exe -B research/tsfm_peft_a_confirmation_v12_20261001/train_confirmation_v12.py --stage parents --label confirmation_parents01 --reserve-seconds 1800
.venv/Scripts/python.exe -B research/tsfm_peft_a_confirmation_v12_20261001/train_confirmation_v12.py --stage all --label confirmation_main01 --reserve-seconds 7200
```

The first parent stage failed during resource heartbeat after3744 optimizer updates. REPAIRS.md records the exact failure and technical repair. The retry restored complete epoch29 (3712 updates), retained the failed attempt, and used a new label/output directory:

```text
.venv/Scripts/python.exe -B research/tsfm_peft_a_confirmation_v12_20261001/train_confirmation_v12.py --stage parents --label confirmation_parent_recovery01 --reserve-seconds 600 --retry-fit peacock_education_level_lr1e-3_s92601 --retry-tag heartbeat01 --retry-reason "Resource heartbeat transient-file stat failure; retain failed attempt and resume complete epoch 29" --resume-from .cache/tsfm_peft_a_confirmation_v12_20261001/confirmation_runs/peacock_education_level_lr1e-3_s92601/last.pt
.venv/Scripts/python.exe -B research/tsfm_peft_a_confirmation_v12_20261001/train_confirmation_v12.py --stage parents --label confirmation_parents02 --reserve-seconds 1800
```

The replacement manifest distinguishes consumed updates from the restored trajectory. No learned parameter definition, source data, selection rule, loss or tolerance changed. The complete matrix has18 canonical fits and currently one technical recovery; the ledger and final verification give the actual terminal count. Preparation/check jobs and their exact source snapshots are in the shared ledger. TEST is inaccessible through the prescribed loader until every selected fit is jointly sealed.

The fixed post-training sequence was seal, predict, score, GPU cost, CPU cost, report, verification. All commands below completed successfully; finalconfirmationchecks.json records the8 verification groups. Actual command/environment/source receipts are in the shared ledger:

```text
.venv/Scripts/python.exe -B research/tsfm_peft_a_confirmation_v12_20261001/evaluate_confirmation_v12.py seal --job confirmation_seal01 --reserve-s 60
.venv/Scripts/python.exe -B research/tsfm_peft_a_confirmation_v12_20261001/evaluate_confirmation_v12.py predict --job confirmation_predict01 --reserve-s 900
.venv/Scripts/python.exe -B research/tsfm_peft_a_confirmation_v12_20261001/evaluate_confirmation_v12.py score --job confirmation_score01 --reserve-s 180
.venv/Scripts/python.exe -B research/tsfm_peft_a_confirmation_v12_20261001/cost_confirmation_v12.py --gpu --job confirmation_cost_gpu01 --reserve-s 3600
.venv/Scripts/python.exe -B research/tsfm_peft_a_confirmation_v12_20261001/cost_confirmation_v12.py --cpu --job confirmation_cost_cpu01 --reserve-s 120
.venv/Scripts/python.exe -B research/tsfm_peft_a_confirmation_v12_20261001/report_confirmation_v12.py --job confirmation_report01 --reserve-s 120
.venv/Scripts/python.exe -B research/tsfm_peft_a_confirmation_v12_20261001/verify_confirmation_v12.py --job confirmation_final_check01 --reserve-s 180
```

GPU costs must precede CPU costs because the CPU direct-model parity uses the saved same-campaign GPU reference. Exact array/model/prediction hashes remain in manifests; raw data, weights, dense predictions and source snapshots remain in local ignored cache. The new population is a same-domain building/use-type group; bounded non-use evidence is not proof of no backbone pretraining overlap.
