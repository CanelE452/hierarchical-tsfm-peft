# Registered K trade-offs — wave 2

All candidates, widths and both seeds are disclosed. Positive is a fixed exploratory rule relative to Raw under DEFAULT128; it is not an independent TEST claim.

## ETTh1

- K=1: MSE gain vs Raw -66.77%; MAE change +51.74%; default speed ratio 1.34×; positive=False; Direct MLP dominates mean=True.
- K=2: MSE gain vs Raw -32.76%; MAE change +31.90%; default speed ratio 1.24×; positive=False; Direct MLP dominates mean=True.
- K=4: MSE gain vs Raw +2.19%; MAE change +7.49%; default speed ratio 1.30×; positive=True; Direct MLP dominates mean=False.
## ETTh2 (safe 17400 cohort)

- K=1: MSE gain vs Raw -31.22%; MAE change +27.95%; default speed ratio 1.45×; positive=False; Direct MLP dominates mean=True.
- K=2: MSE gain vs Raw -12.68%; MAE change +18.61%; default speed ratio 1.46×; positive=False; Direct MLP dominates mean=True.
- K=4: MSE gain vs Raw -10.81%; MAE change +18.22%; default speed ratio 1.26×; positive=False; Direct MLP dominates mean=True.
## Appliances

- K=4: MSE gain vs Raw +14.94%; MAE change +4.86%; default speed ratio 3.16×; positive=True; Direct MLP dominates mean=True.
- K=7: MSE gain vs Raw +15.89%; MAE change +3.38%; default speed ratio 2.64×; positive=True; Direct MLP dominates mean=True.
- K=14: MSE gain vs Raw +16.57%; MAE change +3.49%; default speed ratio 1.73×; positive=True; Direct MLP dominates mean=True.

## Figure footnote

Use each dataset SVG: rows are DEFAULT128 and COMMON_CAP=8*ceil(C/4); columns are normalized MSE, whole B8 latency and allocated peak. Bars show means, whiskers seed min–max (not confidence intervals); Raw has one deterministic model. MAE and reserved memory remain in the tables. Direct MLP is displayed even when it dominates B2. Compare MSE within datasets only.

Development V, adaptive candidate search, full stride1 TRAIN, two fixed seeds, cap3; no new independent TEST, LoRA or Ridge comparison. Equal L512/H48 steps imply different durations. Historical Jena is a separate earlier execution, with its existing ancestry/exposure limits.

Preparation amendment: initial ETTh2 prefix materialized 16 potentially shared-source protected hours, with zero new training/forecast use. WAVE001 and its preparation are preserved as superseded. ETTh2_safe17400 is a custom first17400-source-coordinate cohort, with only13920 prefix rows prepared from the physical source; not the official ETT split.

Conditionally prepared ETTm1 and full137-channel Solar are registry metadata only, with zero performance experiments counted here. Activate their next GPU wave only if the complete parent wave has no verified positive; otherwise preserve them as READY_NOT_EXECUTED.