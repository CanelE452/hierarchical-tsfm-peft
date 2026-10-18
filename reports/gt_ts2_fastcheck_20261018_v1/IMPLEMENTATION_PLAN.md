# GTConvAE lookback / TS2Vec latent fusion execution plan

Approved specification: `CLI_GTConvAE_TS2Vec_Fast_Verification_20261018_KO.md`; exact source SHA and start commit are in START_STATE.json. This is implementation of the approved experiment, not a new model search.

Purpose: determine whether longer physical history and pretrained TS2Vec residual latent features improve the existing GT forecast route. The consumer needs evidence for further investigation and presentation claims, including negative findings.

## Checklist and evidence

- [x] Bind canonical GT full and TS2Vec d320 parents, their selected decoder checkpoints, source snapshots, and private artifacts. Reproduce both GT L512 historical CALIBRATED_F predictions before assigning a lookback effect to either parent.
- [x] Freeze common legal origins, window-local causal fill, TRAIN statistics, four Phase B arms, seeds, equal sampling, optimizer, update decision rule, diagnostics, and resource budget before new VAL scores.
- [x] Verify native GT temporal alignment and F forward parity / finite nonzero input gradients. F and TS2 remain fixed; shared GT product coefficients, native E/D, and the zero-initialized feature bridge are trainable.
- [x] Perform Phase A Raw and frozen GT L512/1024/2048 on the same 371 VAL origins. Separately perform TS2 native D and fixed 0.5 output ensemble context at L1024; never count the ensemble as latent fusion.
- [x] Profile discarded TRAIN smoke; seal 300 or uniformly 100 updates by the registered time rule. Cache only fixed TS2 window features. Execute four forecast adaptation arms times two seeds from identical seed-specific GT parents and sample orders.
- [x] Evaluate CP0 and fixed final update, independently aggregate FP64 observed channel statistics / horizon bands / paired origins, and diagnose branch use and immutable frozen weights.
- [x] Measure actual full online routes at B1/B8 in separate processes, preserving model/feature work and distinguishing allocated/reserved memory.
- [ ] Render three figures and Korean analysis; verify replay, meaningful tests, actual coverage, preservation, scoped normal main push, and actual remote blob bytes.

## Interfaces and scope

`data.py` supplies physical inputs, mask, frozen normalization, origin pools and exact channel-macro loss. `parents.py` verifies and loads private canonical assets without historical constructors' writes. `model.py` supplies a differentiable frozen F and native GT residual-fusion route. Root-owned execution code consumes these interfaces and owns the one-GPU scheduler, receipts, training, evaluation, cost and reporting.

Shared interface checks: features are pooled raw R4 `[B,L/4,320]`, then standardized once inside ForecastRoute using TRAIN-only statistics. Targets retain the canonical normalized coordinates. Historical A0 uses legacy globally filled inputs; all new arms use independently filled physical windows. Foundation weights are excluded from adapter checkpoint state. Private paths and arrays remain in the new cache.

The wall limit is 3 hours from the actual goal start; GPU process wall limit is 90 minutes, with the final 20% reserved for evaluation/cost/report/publication. Cache limit is 6 GiB of new bytes. The 300/100 choice uses TRAIN timing and the full eight-arm estimate, never Phase A accuracy. No fresh TS2 pretraining, GT reconstruction pretraining, L2048 adaptation, loss/width search, new seeds, or other experiment resumption.

Failure evidence is immutable. A legitimate bounded or blocked arm is reported separately from a completed arm with poor accuracy. Prior TEST exposure and repeated development VAL use remain limitations.
