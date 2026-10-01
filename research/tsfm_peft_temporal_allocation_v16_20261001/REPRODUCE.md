# V16 commands and artifact boundaries

Run from the repository root with the recorded local `.venv/Scripts/python.exe -B`. This campaign does not download data or weights. `prepare_receipt.json`, the three manifests and each ledger source snapshot identify the actual local inputs. Large arrays and checkpoints are deliberately excluded from publication; their paths and SHA256 receipts remain in the small records.

The commands below describe one campaign, not permission to rerun an already completed fit or reopen a closed ledger. On resume, first read PLAN, STATUS, ledger, actual PID and existing results. `train01` performs all sixteen planned fits sequentially and writes only TRAIN/VAL selection. An additional optimizer update after a restart consumes another attempt. Technical recovery must preserve the first failure and source diff.

```powershell
.venv/Scripts/python.exe -B research/tsfm_peft_temporal_allocation_v16_20261001/prepare_v16.py --label prepare01 --reserve-s 90
.venv/Scripts/python.exe -B research/tsfm_peft_temporal_allocation_v16_20261001/check_v16.py --label implementation_check01 --reserve-s 45
.venv/Scripts/python.exe -B research/tsfm_peft_temporal_allocation_v16_20261001/train_v16.py --all --label train01 --reserve-s 9000
.venv/Scripts/python.exe -B research/tsfm_peft_temporal_allocation_v16_20261001/evaluate_v16.py seal --job seal01 --reserve-s 60
.venv/Scripts/python.exe -B research/tsfm_peft_temporal_allocation_v16_20261001/evaluate_v16.py predict --job predict01 --reserve-s 600
.venv/Scripts/python.exe -B research/tsfm_peft_temporal_allocation_v16_20261001/evaluate_v16.py score --job score01 --reserve-s 180
.venv/Scripts/python.exe -B research/tsfm_peft_temporal_allocation_v16_20261001/verify_v16.py --group contracts --label verify_contracts01 --reserve-s 180
.venv/Scripts/python.exe -B research/tsfm_peft_temporal_allocation_v16_20261001/verify_v16.py --group predictions --label verify_predictions01 --reserve-s 180
.venv/Scripts/python.exe -B research/tsfm_peft_temporal_allocation_v16_20261001/diagnose_v16.py --job temporal_diagnosis01 --reserve-s 180
.venv/Scripts/python.exe -B research/tsfm_peft_temporal_allocation_v16_20261001/cost_v16.py gpu --job cost_gpu01 --reserve-s 3600
.venv/Scripts/python.exe -B research/tsfm_peft_temporal_allocation_v16_20261001/cost_v16.py cpu --job cost_cpu01 --reserve-s 180
.venv/Scripts/python.exe -B research/tsfm_peft_temporal_allocation_v16_20261001/report_v16.py --job report01 --reserve-s 180
.venv/Scripts/python.exe -B research/tsfm_peft_temporal_allocation_v16_20261001/verify_v16.py --group cost_report --label verify_cost_report01 --reserve-s 180
```

Actual command, status, elapsed time and environment belong to the ledger/receipt, not this example sequence. The CPU synthetic check uses no optimizer or actual backbone. A numerical verifier separately checks recorded contracts, prediction-derived errors and report/cost consistency; those checks do not count as independent reproduction or scientific improvement.

Selection is per dataset: mean of the two seeds' best whole-VAL MSE for each LR, ties favoring1e-4; epoch ties favor the earlier state, including step0. All selections are fixed before `selection_seal.json` and the first v16 TEST prediction. The four evaluation datasets were already exposed before v16. Neither the seal nor repeated scoring creates independent confirmation.

Cost is a single fixed same-session grid:588 GPU and48 CPU rows, including adverse blocks and F0 sentinels. A budget interruption leaves partial rows and missing configurations visible. Do not divide old timing by new timing, silently replace missing rows, or use unused technical fit reserves for new candidates.
