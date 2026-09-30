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

Cost values are historical v11 evidence with their original campaign identity in cost_reference.json. No fresh cost or new confirmation was performed. The full next-unit recipe cannot run until the missing contract described in CONFIRMATION_PROTOCOL.md is recovered; the current empty new-unit list is not a scientific exclusion based on results.
