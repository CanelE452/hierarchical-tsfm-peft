# Bounded post-hoc component attribution after v14

Authority: the user's continuing goal, "해결할수 있게 분석하고 해결해줘", and delegated judgment, "너가 직접 판단해서 쭉 진행하게 해줘". The prior goal turn made concrete progress by completing and publishing v14 at260aa5ab96e22af3b64fdfc622cd6e32038a8e99. Its negative common-adoption decision stands. This analysis does not reopen the head/activation/LR search or retroactively modify PLAN, protocol, selection, predictions or the original decision evidence.

## Question and reason

Normal optimization, full-rank ridge, internal LoRA, learned latent directions, group compression and the v14 matched nonlinear correction have not supplied a common accurate replacement for full-MSE LoRA. Yet the retained and discarded components need not fail for the same reason on each dataset. The earlier v13 diagnosis used common GROUP coordinates; it does not directly identify the canonical PCA LEVEL paths.

For canonical fixed PCA U, distinguish predicting the compressed input from projecting an uncompressed forecast: F0(XU)U^T versus F0(X)UU^T. The complete native point-prediction function includes context scaling and median quantile selection. Noncommutation does not isolate Transformer nonlinearity, pure information loss or pretraining knowledge. This saved-output analysis tests which component replacement would change the current error, without fitting another model or declaring a causal explanation.

## Fixed analysis before computation

All four existing exposed-development datasets, channels, origins, masks and two seeds remain fixed. No site, period or channel is chosen by this analysis. Use the exact prediction_manifest.json and data_manifest.json receipts. Canonical U is data.pca_basis, never the v13 group basis. Use FP64 projection of the recorded FP32 basis; do not refit or rotate U.

Original source roles are PCA_LEVEL, A, F0 and FULL_MSE. Each parent R in {PCA_LEVEL,A} is paired with donor D in {F0,FULL_MSE} at the same seed. F0 has one prediction, reused for both pairs and never counted as two independent models. Define P=UU^T and Q=I-P:

```text
swapP(R,D) = D P + R Q
swapQ(R,D) = R P + D Q
```

These are fixed diagnostic forecast compositions of saved causal forecasts, not deployable low-cost proposals or target-informed oracles. The donor's complete uncompressed forecasting cost would remain; no speedup is inferred. No coefficients, gates, mixing penalties or TEST-based model selection are fitted. Sixteen composed seed forecasts per dataset cover the2 parents x2 donors x2 swaps x2 seeds. Original source and both compositions are retained even when unfavorable. These are not sixteen independent pieces of evidence: A and LEVEL preserve the same Q, so their swapP forecasts with the same donor should also agree within recorded numerical tolerance. Report this duplication rather than multiplying apparent support.

Report original observed masked channel-macro MSE, MAE and signed error, periods A/B/pooled, channels and seed-loss means. Pool channel error sums/counts before macro averaging; never use an ensemble or cross-dataset raw-score average. Conditional paired time blocks reuse the established7 hourly/42 Jena origins,2,000 draws and seed9262026, with period boundaries respected and the same channel/seed pairing. Their scope is fixed exposed models, not full method-selection uncertainty.

Separately report original-source P/Q error energies only on common complete target vectors, with coverage. Never fill missing targets with zero before projection. The Euclidean complete-vector decomposition does not replace masked macro MSE. Check Gram maxabs<=1e-5, prediction recombination, A/LEVEL Q consistency and score-unit P+Q reconstruction within1e-6 absolute plus1e-6 relative tolerance. Replay original observed metrics at atol1e-10/rtol1e-9. Existing FP32 approximation and new FP64 diagnostics remain explicitly distinct.

## Resources and implementation

No neural or coefficient fit, no optimizer update, no model forward, no GPU work, no new data/download, no benchmark. This is a post-hoc analysis supplement under the existing v14 ledger and unchanged total limits, not a new experiment budget or use of technical fit reserve. Reserve up to300 CPU-analysis seconds and90 CPU-check seconds; cumulative caps remain1,800/900 seconds and5GiB storage. Existing consumption is approximately72.473/23.803 seconds. Use CPU4 threads. Store compact source-bound JSON/CSV and a short interpretation; large composed prediction arrays need not be published or retained because sources plus the fixed equations determine them. Preserve failed attempts and corrections.

New code: diagnose_component_swap_v14.py and verify_component_swap_v14.py. Original sealed source files stay unchanged. Separate checks recompute original/composed metrics and algebra from the saved source arrays; they are self-verification, not independent reproduction. Do not rerun all prior model experiments.

## Decision use and stopping

If replacing P is consistently helpful while replacing Q is not, the next question concerns the retained forecast path and coordinate/scaling change; it does not justify another raw-output head. If Q replacement is helpful, record the unresolved temporal residual prediction requirement without assuming a larger G is the answer. If both components are dataset dependent or neither replacement supplies a consistent advantage, the evidence argues against another common-component fix. If a composition improves both of its source forecasts R and D, retain it as an expensive diagnostic only; no cheap implementation has been established. Report F0 and FULL_MSE donors separately: a benefit seen only with FULL_MSE does not isolate compression from the donor's internal adaptation. Complete-vector P/Q energy is not an additive decomposition of the masked channel-macro primary score, whose mask and channel denominators change the metric geometry.

Complete this one attribution comparison, verify and publish it with adverse results. No new fitting follows automatically from a favorable cell. Any next implementation must have a concrete new hypothesis, the closest simple control and a bounded prospective contract; the broader research goal is not completed by this analysis alone.
