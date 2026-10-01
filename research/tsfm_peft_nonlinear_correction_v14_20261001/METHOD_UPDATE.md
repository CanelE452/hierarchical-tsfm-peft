# Fixed-parent temporal correction

The tested change adds one small temporal correction to an already selected forecast. All datasets use the same canonical PCA LEVEL parent policy; no dataset receives a parent chosen by its exposed TEST result.

Let `dX = X - repeat_L(last(X))`, and let `h_a(dX) = W_up a(W_down dX)` act independently on each original channel with shared temporal weights. The three systems are:

```text
LEVEL_GELU   = frozen PCA_LEVEL(X)       + h_GELU(dX)
LEVEL_LINEAR = frozen PCA_LEVEL(X)       + h_Identity(dX)
DIRECT_GELU  = frozen DIRECT_NLINEAR(X)  + h_GELU(dX)
```

The head is bias-free512->32->48, with exact GELU or Identity and17,920 registered parameters. Down weights start from the recorded original untrained G initialization; up weights are zero. Initial outputs equal their respective frozen parents. The parent already restores levels, so the new branch does not add another level term. Its raw-history input and original-channel output are not restricted to P or Q.

All original parent weights, statistics and basis are fixed. LEVEL's total staged fitted parameters are35,840 (old G17,920 plus new head17,920); DIRECT's are42,544 (old NLinear24,624 plus new head17,920). These are not total deployment parameters. Because the last centered input coordinate is always zero, the new down layer's32 corresponding scalar weights have a structurally zero input; requiring every registered scalar to change would be incorrect. Actual gradient and update counts are retained per run.

The frozen parent's full TRAIN/VAL forecast may be cached during new-head training. Deployment executes the parent online once. Cached-head training time/memory does not describe a full online end-to-end training pipeline; parent cache construction is separately charged.

## What the controls identify

LEVEL_GELU versus LEVEL_LINEAR tests the activation change within matched head dimensions, initial tensors, sample order and LR selection opportunities. It does not separate function-space changes from optimization effects. LEVEL_GELU versus the parent tests the entire added stage. DIRECT_GELU tests whether the same cheap nonlinear correction is useful without a TSFM; the two different parents make this a practical system comparison rather than a pure causal effect of pretraining.

Existing F0, full-MSE LoRA, native-loss LoRA, internal compressed LoRA A, direct NLinear and GROUP_RES remain external references. Improving only the canonical PCA parent is insufficient to establish a new practical choice. An epoch0 selection is the original parent, not a learned improvement.

For LEVEL_GELU and DIRECT_GELU, the new head is nonlinear; LEVEL_LINEAR remains the matched linear control. The original LEVEL's linear commutation, Q-preservation and effective-rank arguments cannot be transferred to the whole nonlinear system. The added raw-input correction is not confined to Q even in LEVEL_LINEAR. These comparisons also cannot establish that the old residual-input branch remains necessary after adding this raw-input correction; that is not the direct question tested here.

## Prior-art boundary

Latent adapters are established in [AdaPTS, ICML2025](https://proceedings.mlr.press/v267/benechehab25a.html). The [TiDE author preprint v5, section4](https://arxiv.org/html/2304.08424v5) and [N-BEATS author implementation](https://github.com/ServiceNow/N-BEATS/blob/master/models/nbeats.py) already combine nonlinear forecasting blocks with residual/linear paths. The [NLinear author implementation](https://github.com/cure-lab/LTSF-Linear/blob/main/models/NLinear.py) supplies the direct last-centered affine reference principle. This small head is not a reproduction of those complete systems or an originality claim based on a new name.

The prior v13 complete-vector P/Q diagnosis was in common GROUP coordinates; [PLAN_ERRATUM.md](PLAN_ERRATUM.md) corrects the motivation's coordinate label while preserving the immutable plan and experiment contract. It supplied a reason to test a correction that can affect both components, not proof that nonlinearity caused the remaining errors.

## Evidence status

The fixed48-fit matrix and joint76-instance evaluation are complete. In Jena, LEVEL_GELU reduces pooled MSE by12.329% and MAE by6.454% versus the PCA LEVEL parent. The parameter-matched linear head already reaches MSE0.286657, compared with GELU's0.284368: GELU's incremental MSE change is-0.799%, with a conditional time-block interval[-1.679%,-0.457%]. In period B alone, its MSE is0.0235% higher than the linear control. These intervals condition on the selected models and seeds; they do not cover all development selection uncertainty.

Jena DIRECT_GELU is more accurate (MSE0.268713/MAE0.280965), as is full-MSE LoRA (0.232892/0.238943). Improving the frozen LEVEL parent therefore does not establish the need for a TSFM in this added correction. Hog LEVEL_GELU worsens both parent metrics; Robin and Peacock select epoch0 for both seeds and retain the parent function. Tiny score replay differences at approximately1e-9 are not learned gains. All48 fits made real updates and stopped by the fixed patience rule; recorded checkpoint replay error is zero. Fixed TRAIN-probe improvement does not prove full TRAIN convergence or identify a generalization failure's unique cause.

The prespecified cost-allocation rule selects Jena only. Its same-session132 GPU and24 CPU cost rows are complete. LEVEL_GELU uses218.04MiB peak allocated at batch4 and reaches241.16 origins/s, while DIRECT_GELU uses8.81MiB and reaches2,220.55 origins/s with lower MSE and MAE. Full MSE-LoRA uses281.94MiB at189.99 origins/s; a K-row chunk reduces this to197.89MiB at24.88 origins/s with its more accurate forecasts preserved within the fixed parity tolerance. The smaller new head does not remove the TSFM's roughly191MB deployment state. Sentinel drift up to23.23% and three passes/block prevent a stable small speed-advantage claim. CPU direct-based options are reported separately. No old/new timing ratio is used.

This correction is not adopted as a common replacement, and the same bounded head/activation question is closed without spending its technical reserve on new ideas. The Jena activation effect remains a supported conditional development observation. All four datasets are exposed development data. No independent confirmation, universal accuracy improvement, stable cost advantage or paper-readiness claim is made. Full adverse periods, seeds and external references are retained in evaluation01.json and the comparison tables. Completed saved-artifact checks are in final_checks.json; they are not independent reproduction.
