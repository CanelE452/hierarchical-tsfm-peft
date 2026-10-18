# Comparison under execution constraints

B = 8. All frozen-F routes share the same per-call row cap: Household 16; SML2010 48. Raw uses four F calls; B2 uses one. Direct MLP uses no F.

## Household

- B2 vs Raw: normalized MSE +2.52%; MAE +18.19%.
- Whole B8 speed ratio vs Raw: 3.70×.
- Direct MLP dominates B2 in MSE, allocated peak and latency: mean True; seed 92601 True; seed 92602 True.

## SML2010

- B2 vs Raw: normalized MSE +108.90%; MAE +72.37%.
- Whole B8 speed ratio vs Raw: 3.04×.
- Direct MLP dominates B2 in MSE, allocated peak and latency: mean True; seed 92601 True; seed 92602 True.

## Figure and footnote

Use COMPARISON.svg for vector export. Bars are seed means; whiskers are two-seed min–max ranges, not confidence intervals. Compare normalized MSE only within a dataset. Jena* is historical context from a separate earlier measurement.

L512/H48 denote equal step counts, not equal durations: H48 is 8 h for Jena, 12 h for SML2010, and 48 h for Household. No common-frequency resampling was imposed.

Development V, two seeds, cap3, fixed first B8 group, 10 warmups + 20 timings, allocated tensor memory. SML2010: 9 V origins, 8 with observed targets; three constant motor channels remain included. No new LoRA or Ridge baseline. Direct MLP competition must remain visible. No independent TEST claim.

An initial forced `CUBLAS_WORKSPACE_CONFIG=:4096:8` cost pass is preserved as `AUXILIARY_WORKSPACECONFIG`. The canonical pass inherits the current unset parent value; the historical Jena environment value was not recorded. Cost-only repeat: 20 route measurements, 200 warmups and 400 timings in total; no retraining.
