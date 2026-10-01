# Coordinate-label clarification before new evaluation

The PLAN.md motivation says "PCA-related components" when referring to the v13 complete-vector diagnosis. The actual saved v13 diagnosis uses the same **GROUP basis** for every compared method: `research/tsfm_peft_group_basis_v13_20261001/space_diagnosis01.json`. Its P/Q excess errors must be described in that common GROUP coordinate system, not as PCA-coordinate measurements.

The supported motivation is only that a universal correction confined to one of those two components was not established across the four tasks. It does not prove that nonlinearity is the cause. v14's actual parent and stored basis are the original fixed PCA LEVEL on every dataset, as explicitly defined in protocol/model/data manifests.

This clarification is recorded while the fixed48-fit matrix is running and before v14 selection or new TEST evaluation. It changes no parent, tensor, data, initialization, model, optimizer, selection rule, comparison, cost rule or resource limit. Original PLAN.md and its provenance hash remain preserved.
