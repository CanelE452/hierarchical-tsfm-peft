# Bounded implementation repairs

The first CPU Q analysis reproduced the parent/full output but stopped because its pooling assertion applied a score-unit absolute tolerance to a large raw SSE sum. Maximum raw-sum difference was 1.1096e-10, while the corresponding MSE difference was 1.55e-15. Counts now match exactly and raw-sum discrepancies are divided by each channel's observed count before comparison with the unchanged score tolerance 1e-10. Forecast parity remains atol1e-5/rtol1e-4. No model, source mask, prediction or selection changed.

The failed q_contribution01 job, source snapshot, traceback and partial arrays remain. q_contribution_repair01.json links the before/after source, numerical diagnosis and positive/negative checks. The corrected q_contribution02 completed all three datasets and the original-score replay.

A repair-check helper also failed on Windows' default cp949 text decoding. Its ledger traceback remains; explicit UTF-8 fixed the helper. Both failed jobs consume their recorded time. Neither failure performed a parameter update or TSFM forecast. No fit attempt was hidden as a synthetic test.

The detailed prior assistant PLAN was not found. PLAN.md explicitly records that absence; REQUEST_PLAN.txt is not mislabeled as an approved detailed plan, and no new-unit identities were filled in by inference.
