# Confirmation exposure search note

Scope: bounded local-record note only. This is not a complete non-use proof, and no new search was run while writing this file.

## Selected unit

- [확인] Final v12 supplement fixes one fresh confirmation unit, `BDG2 Peacock Education`, and records why local Jena weather periods are not claimed fresh: `E:\CODING\proj\hierarchical-tsfm-peft\research\tsfm_peft_a_confirmation_v12_20261001\PLAN_SUPPLEMENT.md`.
- [확인] The frozen prospective protocol records `peacock_education` with C=13, K=4, the 13 columns, source hashes, source-naive hourly time basis, and US/Eastern metadata timezone: `E:\CODING\proj\hierarchical-tsfm-peft\research\tsfm_peft_a_confirmation_v12_20261001\confirmation_protocol.json`.
- [확인] TRAIN-only eligibility receipt records metadata candidates 17, eligible columns 13, and no validation/test target values read for this eligibility decision: `E:\CODING\proj\hierarchical-tsfm-peft\research\tsfm_peft_a_confirmation_v12_20261001\confirmation_eligibility_peacock_education.json`.

## Weather prior-use examples found before this note

- [확인] Jena 2024 source and exposure note are recorded in `E:\CODING\proj\hierarchical-tsfm-peft\research\tsfm_peft_internal_vs_subspace_v11_20260930\data_contract_jena.json`.
- [확인] The local exposure audit records Jena 2018, 2019, 2021, 2022, 2023, and 2024 as opened in PEFT/S1, and Jena 2020 as excluded due prior weather/raw exposure: `E:\CODING\proj\mltimeseries\results\peft_paper_closure_v1\data_exposure_audit.md`.
- [확인] The same audit gives concrete mapped examples: `adaptation_scope_s1_jena2024`, `contribution_freeze_jena2023`, `decision_transfer_dev_jena2021`, `decision_transfer_test_jena2022`, `initial_headroom_jena2019`, `head_convergence_jena2018`, and `excluded_jena2020`: `E:\CODING\proj\mltimeseries\results\peft_paper_closure_v1\data_exposure_audit.md`.

## Peacock Office retry receipt repair note

- [확인] `fresh_unit_train_eligibility_01_retry2.json` reported `Peacock Office` metadata channel count 0 because the script converted the BDG2 metadata `electricity` flag with numeric parsing. The local metadata stores this flag as the string `Yes`, so the candidate filter was technically wrong. Treat retry2 as a superseded technical schema receipt, not as valid evidence that Peacock Office has zero channels.
- [확인] `fresh_unit_train_eligibility_01_retry3.json` corrected the flag check to `electricity == "Yes"`; it found 11 Peacock Office metadata channels and 7 TRAIN-eligible channels. Peacock Office remains below the C>=8 start rule, but for that reason, not because of the retry2 zero-channel result.

## Source license note

- [확인] Root supplied the pinned BDG2 source license URL for the existing source revision: `https://raw.githubusercontent.com/buds-lab/building-data-genome-project-2/9b97ccbe90096aff42ed4fd6493bf7ae692d7118/LICENSE`, title `Attribution-ShareAlike 4.0 Unported`. The already-bound protocol is not changed; downstream data contracts/notes should refer to this pinned license exactly.
