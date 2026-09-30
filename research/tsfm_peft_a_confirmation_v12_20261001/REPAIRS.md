# Bounded implementation repairs

The first CPU Q analysis reproduced the parent/full output but stopped because its pooling assertion applied a score-unit absolute tolerance to a large raw SSE sum. Maximum raw-sum difference was 1.1096e-10, while the corresponding MSE difference was 1.55e-15. Counts now match exactly and raw-sum discrepancies are divided by each channel's observed count before comparison with the unchanged score tolerance 1e-10. Forecast parity remains atol1e-5/rtol1e-4. No model, source mask, prediction or selection changed.

The failed q_contribution01 job, source snapshot, traceback and partial arrays remain. q_contribution_repair01.json links the before/after source, numerical diagnosis and positive/negative checks. The corrected q_contribution02 completed all three datasets and the original-score replay.

A repair-check helper also failed on Windows' default cp949 text decoding. Its ledger traceback remains; explicit UTF-8 fixed the helper. Both failed jobs consume their recorded time. Neither failure performed a parameter update or TSFM forecast. No fit attempt was hidden as a synthetic test.

The detailed prior assistant PLAN was not found. PLAN.md explicitly records that absence; REQUEST_PLAN.txt is not mislabeled as an approved detailed plan, and no new-unit identities were filled in by inference.

## Delegated continuation

The user subsequently delegated the missing unit/split choices and continued execution. This is a prospective authorization change, not recovery of the missing old plan. AUTHORIZATION_SUPPLEMENT.txt and the separately hashed confirmation protocol preserve that distinction. runtime_v12 now validates the supplemental fit matrix while retaining the original immutable provenance and common ledger; it also blocks fits after the new confirmation exposure marker.

The bounded eligibility helper had a metadata-schema assertion failure and a nonexistent Job.save_json call; both CPU failures remain in the shared ledger. Its retry2 also incorrectly parsed the electricity flag numerically and reported zero Peacock Office channels. Retry3 uses the actual string Yes, giving11 metadata channels and7 TRAIN-eligible channels. Retry2 is superseded technical evidence, not a scientific negative. The subsequent Peacock Education selection used its own fixed TRAIN-only eligibility receipt (17 candidates,13 eligible) and no model performance. See confirmation_exposure_search.md. These data/record checks made zero optimizer updates and consume CPU time rather than fit attempts.

First LEVEL/92601 attempt stopped after3744 updates because a background resource heartbeat raised FileNotFoundError; the old handler retained only repr, so its exact internal frame was not established. The storage census had an is_file→stat race with atomically replaced JSON temporary files. It now skips only files disappearing at that boundary and retains full background tracebacks. The concurrent stress helper01 was inconclusive because its writer hit Windows PermissionError; its wording that the race was independently reproduced is invalid and superseded by helper02. Helper02 deterministically removes a file between is_file and stat: the old expression fails and the repaired census passes. This establishes coverage of that failure mode, not the omitted original stack.

Preserve the original failed attempt and3744 updates. Resume from the last complete epoch29/3712-step checkpoint with optimizer/scheduler/RNG and previous selected checkpoint intact; the extra32 updates in the failed incomplete epoch remain consumed and are not retained in the resumed model. The resume is a technical-reserve attempt. No method/data/LR/selection/tolerance changed.

## Publication whitespace check and accounting correction

The first exact-path Git staging check returned exit2 only for generated SVG path-attribute trailing spaces. No scientific computation failed. The follow-up retained the frozen figure bytes and their receipts, parsed both SVGs as valid XML, and confirmed all other staged paths pass the default whitespace check. This explicit generated-file exception is recorded in confirmation_staging_check.json; no hook or Git configuration was bypassed. The first staging helper ran outside its planned CPU reservation; its observed1.4055728s is late-accounted with reservation_rule_deviation=true. This bookkeeping deviation is preserved, not retroactively described as prior reservation. No optimizer, model run or fit was added.
