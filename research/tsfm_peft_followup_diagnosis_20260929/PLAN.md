# Follow-up diagnosis after completed v9

User request: `해결해줄수있어?` (2026-09-29), following the explanation that saved Robin TRAIN probes did not evaluate a trained v9 P checkpoint. This authorizes the diagnostic follow-up; it is not a retroactive change to the approved v9 experiment.

First action: evaluate saved initial, best-trained, and final Robin P/RAW checkpoints on the exact original fixed TRAIN probe and whole VAL. No optimizer, new fit, TEST access, checkpoint selection, model change, or timing benchmark. Preserve all v1-v9 artifacts. Hash sources/checkpoints before and after, check initial-score and best-trained VAL replay, save predictions only in ignored local cache. Four existing runs, two seeds; seed scores are not an ensemble.

Diagnostic operating ceiling: one GPU job up to 300 seconds, zero fits/optimizer updates, local storage below 100 MiB. One sequential model instance. No downloads, installations, new data, external services, process termination, or automatic resource increase. Reserve the job before importing torch/loading models. This new diagnostic budget is separate from the closed v9 ledger.

Interpretation: TRAIN improvement with worse VAL supports failure to transfer this learned correction, not a proof of a specific distribution shift or irreducible representation limit. No TRAIN improvement leaves optimization/representation unresolved. A fixed 96-origin probe is evidence on those origins only. Do not use changing epoch TRAIN losses as substitutes. The result will determine the smallest concrete repair proposal; new method/rank/loss/data changes are not authorized by the old v9 contract.

The latest user requests action, not another paper summary. Complete this diagnostic before deciding which repair is justified. Do not claim that a diagnosis itself solves accuracy or generalization.
