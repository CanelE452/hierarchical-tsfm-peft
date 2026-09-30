# Claim–evidence boundaries

1. **Supported on the exposed development set:** adding the fixed learned Q to the same A main forecast improves pooled MSE/MAE relative to no Q and last-residual persistence. Source: q_contribution02.json plus q_uncertainty01.json; saved selected checkpoints, observed masks and paired time blocks. This is a same-checkpoint output comparison, not independently trained no-Q superiority.

2. **Supported conditional mathematics:** shared bias-free linear G preserves the exact Q space; complete uniform MSE then has a φ-independent full/main loss offset. With actual mask weights the difference is 2 J_bᵀ Wq. Source: actual model/loss code, TRAIN/VAL contract scan, FP64 counterexample and 24 real paired backwards. Do not turn a few matching batches into full-trajectory identity.

3. **Not observed in the tested actual offset replay:** scheduler divergence on Robin/Hog. Their eight conditional replays kept the same LR schedules; Jena's VAL missingness prevents a constant-shift claim. The relative scheduler counterexample is synthetic. Source: equivalence_scheduler_replay01.json and installed PyTorch source receipt.

4. **Not established:** accuracy-gap resolution against full MSE-LoRA, independently optimized no-Q advantage, universal cost superiority, or fresh generalization. All three full MSE-LoRA pooled MSE values remain lower. The prior native-LoRA MAE counterexamples stay in the main table, not hidden in an appendix.

5. **Historical conditional deployment evidence only:** unchanged v11 A has useful batch4 memory/throughput tradeoffs on some tasks, while small-batch full LoRA and direct NLinear remain real alternatives. Source: cost_reference.json, which retains all selected methods, execution variants and sentinels from the original campaign. No v12 timing was run; no ratio combines campaigns.

6. **Pending access to the specified contract:** new selection-after-development confirmation. Source: CONFIRMATION_PROTOCOL.md and PLAN.md recovery record. The explicit approval referred to already-fixed units, but the request and bounded transcript search did not identify them. No new unit was invented. This is a contract gap, not a negative experiment or completed confirmation.

Claims 1–3 narrow and strengthen the explanation of the existing A design. They do not supply claims 4–6. Self-checks and agent review verify particular code/numbers; neither is independent scientific replication. The result may be scientifically useful without satisfying the full pending confirmation objective.
