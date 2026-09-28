# Practical Controls for Level-Preserving Residual Adaptation of Compressed Time-Series Foundation Models

**Working English manuscript; empirical comparisons complete.** Robin and Jena evaluations, the Robin inference grid and corrected diagnostics are measured results. Final artifact verification and publication status are separate records; this draft does not imply submission readiness. Source access and contribution boundaries are in [CLAIMS_PRIOR_ART.md](CLAIMS_PRIOR_ART.md); audit and technical detail are in [APPENDIX.md](APPENDIX.md).

## Abstract

Channel compression reduces the number of series presented to a frozen time-series foundation model, but constrains decoded forecasts to a small channel subspace. We study a level-preserving reconstruction-residual adapter that complements this path. Under fixed orthonormal PCA, the method is exactly a frozen latent forecaster plus a projected, bias-free, low-rank NLinear-style predictor in the discarded space. Its normalization and factor-plus-residual construction are established principles. On Robin building electricity, LEVEL yields MSE 0.275196 versus 0.327724 for direct shared NLinear, a 16.03% reduction under a fixed recipe, while uncompressed LoRA remains better at 0.223610. In the same-session inference grid, uncompressed microbatches match LEVEL's allocation with substantially lower throughput; direct NLinear is much cheaper, and throughput varies with execution path and block. The exposed Jena weather extension is a counterexample: LEVEL MSE 0.324359 is worse than matched RAW 0.291159 and direct NLinear 0.273984. Corrected Robin diagnostics localize remaining loss in both retained and discarded spaces without identifying a causal remedy. The findings retain a conditional building-electricity contribution and a measured three-axis tradeoff, while rejecting broad cross-domain residual superiority, unique memory savings and a new-normalization claim.

## 1. Introduction

A univariate time-series foundation model can forecast each channel of a multivariate task, but the amount of inference work depends on how many series it processes and how they are batched. Encoding many channels into fewer latent series offers one way to reduce that work. A fixed linear decoder then imposes an equally concrete restriction: each predicted future channel vector lies in its retained subspace. Recent patterns omitted by the encoder cannot be represented by the decoded main forecast alone.

The omitted component is observable in the input as a reconstruction residual. Forecasting that component can recover useful information, but its existence does not prove that it is predictable, that a TSFM is needed, or that compression is the best way to lower resource use. A direct linear model can avoid the TSFM entirely. Smaller batches of independent series can reduce the memory of an uncompressed model without changing its learned parameters or statistical task. Both are practical countercomparisons to a compressed adapter.

The fixed method considered here restores the last observed residual level and learns its subsequent changes with a small shared temporal map. Its construction was developed on exposed building-electricity data. A subsequent fixed Robin comparison improved the old residual path and matched controls, while retaining a substantial accuracy disadvantage against uncompressed models. The present study does not modify the method to remove that disadvantage. It asks whether the existing design still offers a useful choice after direct NLinear and uncompressed microbatch inference are included, and whether its residual-path advantage extends to a previously exposed weather task.

The intended contribution is bounded: an explicit arrangement of known forecasting principles, a transparent projected-NLinear interpretation, and comparisons that can support or weaken the complete deployment. Positive residual ablations and a useful deployment are separate conclusions. We preserve both the earlier favorable controls and the stronger alternatives that may limit their practical significance.

## 2. Closest principles and the contribution boundary

AdaPTS projects multivariate inputs into latent channels, applies a univariate foundation model, and maps predictions back. Dimension reduction and forecast-loss adaptation of encoders and decoders are therefore direct predecessors. The author implementation path reviewed for this study returns encoded-FM-decoded predictions and treats reconstruction loss as a separate objective. Our additional observed-input residual path is the difference tested here. COMPRESS isolates the main-path structure; it is not a reproduction of AdaPTS's full probabilistic framework or optimal tuning. The final publication metadata is verified, while accessible preprint sections and the current author code, rather than an inaccessible final PDF, support this method-level account. [Benechehab et al., 2025](https://proceedings.mlr.press/v267/benechehab25a.html)

NLinear subtracts the final input value, predicts with a temporal affine map and restores that value. We apply the same level treatment to a reconstruction residual. RevIN removes and restores instance mean and scale, with optional affine parameters; this is a related but different normalization. Neither level removal/restoration nor reversible normalization is claimed as new. Deep Factors provides an earlier global-factor/local-component construction, using learned deep factors and probabilistic local effects. Our fixed PCA, frozen TSFM and deterministic shared temporal correction differ from that model, while retaining the broad hybrid idea. [Zeng et al., 2023](https://ojs.aaai.org/index.php/AAAI/article/view/26317/26089), [Kim et al., 2022](https://seharanul17.github.io/RevIN/), [Wang et al., 2019](https://proceedings.mlr.press/v97/wang19k.html)

Online correction methods offer another related direction. ELF uses feedback in lightweight adaptation of foundation-model forecasts, and the EPOC preprint represents completed forecast-error blocks using compressed state and an endpoint. Here the residual is available from the observed input at forecast time, and adapter parameters remain fixed during evaluation. These distinctions concern information and update rules; they do not establish priority for residual or endpoint-preserving correction. [Lee et al., 2025](https://proceedings.mlr.press/v267/lee25ag.html), [Fujimoto and Nishi, 2026](https://arxiv.org/abs/2609.30929)

The strongest simple objection is that a standard NLinear predictor may already provide the relevant accuracy at much lower cost. A second objection is that the original uncompressed inference implementation may use more memory than necessary. The v7 comparisons address those objections directly. A completed comparison would not by itself establish a new normalization mechanism, universal TSFM superiority, or paper acceptance.

## 3. Fixed method and matched controls

### 3.1. Two channel spaces

Let \(X\in\mathbb R^{L\times C}\) be a context after TRAIN-only standardization and input imputation. Let \(U\in\mathbb R^{C\times K}\) contain orthonormal TRAIN PCA directions and put \(P=UU^\top\), \(Q=I-P\). The main forecast and observed reconstruction residual are

\[
b(X)=F_0(XU)U^\top,\qquad r(X)=XQ.
\]

The frozen \(F_0\) is Chronos-Bolt-small at the existing pinned revision. We use its median quantile for the first H future observations. E/D remain fixed PCA maps throughout v7. The experiment uses L=512 and H=48 observations, whose elapsed-time meaning differs between hourly Robin and ten-minute Jena.

For any context matrix V, let \(P_n(V)\) repeat its last row n times. That row is detached when used as a level statistic. Define

\[
\mathcal N_G(V)=P_H(V)+G(V-P_L(V)),
\qquad \widehat Y_{\rm LEVEL}=b(X)+\mathcal N_G(r(X)).
\]

G is shared across channels, bias-free and without an activation, with temporal layers 512→32→48. Its output layer is initialized to zero, so LEVEL begins at \(b+P_H(r)\). Only its 17,920 parameters are learned. The last level comes from the available imputed context, never a future statistic. The statistical task predicts the full future in original standardized channels, rather than fitting a separately constructed future-residual label.

### 3.2. Projected NLinear interpretation

For fixed orthonormal U and shared linear bias-free G,

\[
\widehat Y_{\rm LEVEL}
=b+\mathcal N_G(XQ)
=b+\mathcal N_G(X)Q.
\]

The final equality follows because the temporal map and last-row operation commute with the channel projection. Thus the residual path is a projected low-rank NLinear-style predictor. It does not introduce a new temporal normalization. Its correction lies in the discarded subspace, so \(\widehat Y_{\rm LEVEL}P=b\); it cannot repair retained-subspace errors through that path. Constant offsets a with aP=0 are preserved through the discarded path.

These are conditional forward identities, not explanations of observed accuracy. If G has temporal matrix A with rank at most 32, its level-preserving effective matrix is

\[
A_{\mathrm{eff}}=A+(\mathbf1_H-A\mathbf1_L)e_L^\top,
\quad A_{\mathrm{eff}}\mathbf1_L=\mathbf1_H,
\quad \operatorname{rank}(A_{\mathrm{eff}})\le33.
\]

Equal learned counts therefore do not imply the same function class as the old residual map. Standard NLinear includes a bias and does not automatically satisfy the projection-commutation equality. Learned nonorthogonal E/D, channel-specific G and masked losses also require separate reasoning. Appendix A provides proofs and the precise coordinate and missingness conditions.

### 3.3. Comparisons with distinct roles

The old model is \(b+G_{\rm old}(r)\), initialized at b. LEVEL_ONLY is \(b+P_H(r)\), and COMPRESS is b. The matched raw-input model is

\[
\widehat Y_{\rm RAW}
=b(X)+P_H(r(X))+G_{\rm raw}(X-P_L(X)).
\]

LEVEL and RAW restore the same residual level, start from the same function, and have matched G tensors, size, sample order and selection opportunities. RAW does not add the raw input level on top of the main forecast. Its correction can modify the retained space. LEVEL versus OLD measures the combined change in centering, level treatment, effective operator and initialization; LEVEL versus RAW tests the input-path restriction under the common treatment; LEVEL versus ONLY tests learned changes beyond the shared initial function.

DIRECT_NLINEAR operates on the original standardized channels without PCA, G or a TSFM:

\[
\widehat Y_{\rm NLinear}=P_H(X)+T(X-P_L(X)),
\qquad T=\operatorname{Linear}(512,48,\mathrm{bias=True}).
\]

It shares the temporal affine map across channels and has 24,624 trainable parameters. Its default affine initialization generally differs from persistence and LEVEL_ONLY. Both learning rates share exactly the same initial tensors for a given seed. This direct model is distinct from the previously tested normalized latent LINEAR_RES and the full-TRAIN SHARED ridge predictor; those references retain their original names and roles.

Uncompressed F0 and merged LoRA are practical accuracy alternatives. Their inference microbatch grid changes only execution over independent series rows, not time indices, forecast horizon, precision, checkpoint or task-level batch size. A valid smaller-row batch must return the same complete original-channel forecast within the fixed numerical tolerance. No future prediction cache removes work from deployment.

## 4. Evaluation design

### 4.1. Evidence stages and selection

Hog, Bull and Electricity are exposed development data. Robin v6 evaluated the fixed method on a group not used in its selection in the searched records; pretraining overlap was not excluded. The new Robin NLinear comparison, cost grid and diagnosis were designed after those results and are follow-up analyses. Jena 2024 has explicit earlier-use records, including earlier TEST use with another setup. It is an exposed other-domain scope test, not independent confirmation. This distinction remains in the main conclusions because repeated selection can bias performance estimates. [Cawley and Talbot, 2010](https://jmlr.org/papers/v11/cawley10a.html)

All v7 methods and comparisons were fixed together before new A results. Robin retains its 17 hourly office electricity channels and K=5, TRAIN/VAL split and 40 origins in each existing TEST period. Existing TSFM models and predictions are reused rather than fitted again. Four new DIRECT_NLINEAR fits cover learning rates 0.001 and 0.0001 and seeds 92601 and 92602. Each fit selects the earliest minimum full-VAL MSE checkpoint, including step 0; the lower mean best-VAL score across seeds chooses the learning rate. Exact learning-rate ties choose 0.001. TEST never chooses a learning rate or epoch.

Jena uses 21 recorded meteorological variables and K=6. L512 corresponds nominally to 85 hours 20 minutes, and H48 to eight hours. TRAIN occupies the first half of 2024, VAL July–August, TEST-A September–October, and TEST-B November–December under the exact half-open timestamp contract in Appendix B. Evaluation origins advance by 24 observations, or four hours. Targets fit wholly within their assigned period, while contexts may use already available observations before the period. There are no online TEST updates. Fresh TRAIN statistics, PCA and untrained adapters are used, and B is performed regardless of whether A favors LEVEL. [Official Jena source](https://weather.bgc-jena.mpg.de/weather_data.html)

The main endpoint is observed-target channel-macro MSE after TRAIN standardization. For the combined periods, each channel's error sums and observed counts are pooled before channel averaging. MAE and every fixed period, seed and channel remain visible. Seed means average losses, not predictions. Robin's paired time blocks contain seven daily origins; Jena's contain 42 four-hour origins. Each resampling moves methods and channels together and stays inside a period. These conditional intervals do not establish equivalence or cover all learning-rate and checkpoint selection uncertainty.

### 4.2. Fitting and practical execution controls

The shared recipe uses AdamW, zero weight decay, gradient clipping at 1, task batch 4, and 512 phase-balanced TRAIN origins per epoch. Maximum training is 120 epochs with six nonimproving VAL checks and the existing plateau scheduler. G learns at 0.001. LoRA retains rank 8, alpha 16, q/v targets, learning rate 0.0001 and its native quantile objective; this loss difference limits a purely causal interpretation of its comparison with MSE-trained adapters. The new NLinear receives the two fixed learning rates on each dataset. Technical repair and the narrow pre-specified training-cap extension are permitted before that dataset's TEST selection seal; poor scores alone do not open a new search.

For Robin inference, the grid includes original and generic-wrapper full-row execution as well as smaller row batches. F0 and LoRA process rows of the flattened B×C contexts, whereas LEVEL processes B×K latent rows and still executes its complete E/D and residual path. The generic full and chunked wrappers use the same preallocated median-output buffer. Thus full-wrapper versus chunk-wrapper isolates row batching more closely than original versus wrapper, which also includes implementation changes. All configurations and blocks are retained rather than selecting a favorable minimum time. Jena keeps original full inference and has no new cost grid.

Costs use the same RTX 4070, FP32, disabled TF32, four CPU threads and 24 common Robin VAL origins. The deployment boundary is standardized CPU input through transfer and complete CPU-returned forecast, including sequential calls and recombination. Shape-specific warm-ups precede synchronized measurements. CPU NLinear deployment is reported separately. Peak allocated, reserved, resident, process memory and deployed tensors are different quantities. The whole grid, parity and timing distributions are in Appendix C and the completed source records. Earlier v6 timing is not divided by v7 timing to manufacture a speedup.

### 4.3. Remaining-error diagnosis

The descriptive Robin analysis separates channel, origin and period losses, signed bias, and positive versus negative excess loss against F0, LoRA and OLD. For the spatial diagnostic, only an `(origin,horizon)` row with all 17 targets observed is admitted. On this common subset, error is projected into P and Q and its squared norm is divided by C. Coverage is reported per period, and the resulting complete-vector loss is not substituted for the full masked macro endpoint.

Recent residual level, variability and change are compared with subsequent forecast errors using fixed descriptive summaries. Target values appear only as retrospective outcomes, not predictor inputs, routing features or channel-selection criteria. Associations do not prove a cause. No diagnosis changes K, G, level treatment, loss, model selection or the included dataset in this study.

## 5. Results

### 5.1. Reused Robin evidence

The following are measured v6 combined TEST results, reproduced without retraining. They establish the starting evidence, not the outcome of the new practical controls. Lower MSE and MAE are better.

| Existing model | MSE | MAE |
|---|---:|---:|
| LEVEL_RES | 0.275196 | 0.324593 |
| OLD_RES | 0.538776 | 0.483495 |
| LEVEL_RAW | 0.346630 | 0.385883 |
| LEVEL_ONLY | 0.349041 | 0.386300 |
| COMPRESS | 0.977023 | 0.661870 |
| F0 | 0.229095 | 0.262074 |
| LoRA | 0.223610 | 0.256347 |
| LEVEL_LINEAR_RES | 0.315646 | 0.354268 |
| SHARED ridge | 0.337277 | 0.376721 |

LEVEL improved the old residual branch, the matched RAW control and level-only restoration in the combined endpoint. F0 and LoRA nevertheless remained more accurate, with LEVEL MSE higher by 20.12% and 23.07%, respectively, and worse MAE. These disadvantages are central to the practical question. Historical Hog recovery and the Electricity non-improvement also remain development evidence rather than being dropped from the narrative. Exact periods, seeds, intervals and counts are retained in the [v6 result](../tsfm_peft_level_confirmation_v6_20260928/evaluation01.json) and original manuscript.

The previous Robin original-path costs showed lower peak allocated memory for LEVEL but no stable throughput advantage over uncompressed TSFMs. That observation did not test whether uncompressed row batching could recover the memory difference. It therefore cannot settle the v7 practical deployment question.

### 5.2. Direct NLinear follow-up

The four direct NLinear fits completed without a training-cap extension. VAL selected learning rate 0.001: its mean best-VAL MSE was 0.435458, compared with 0.437953 for 0.0001. Every fit updated its temporal weights and bias, reduced the fixed TRAIN-probe loss, and reproduced its selected VAL output after restoration with recorded maximum error zero. The selected checkpoints were epochs 7 and 30. Thus the unfavorable comparison below is not a step-zero selection or an observed failure to learn. It remains a finite shared-model recipe, not the best possible linear predictor. [Selection](selected_robin.json), [run and convergence details](APPENDIX.md#b4-completed-robin-direct-nlinear-fits).

| Robin endpoint | LEVEL MSE | NLinear MSE | LEVEL MAE | NLinear MAE | LEVEL MSE change versus NLinear |
|---|---:|---:|---:|---:|---:|
| TEST-A | 0.273677 | 0.327484 | 0.331865 | 0.367210 | −16.43% |
| TEST-B | 0.276712 | 0.327962 | 0.317319 | 0.339608 | −15.63% |
| Combined | 0.275196 | 0.327724 | 0.324593 | 0.353410 | −16.03% |

The combined absolute MSE difference is −0.052528. The conditional paired-block interval for relative MSE change is [−19.84%, −11.84%], with the trained seeds fixed; it does not account for all training, selection or prior-exposure uncertainty. Both paired seeds favor LEVEL in each period. Combined NLinear MSE is 0.334129 and 0.321319 for seeds 92601 and 92602, respectively; corresponding LEVEL MSE is 0.275009 and 0.275383. These are means of per-model losses, not prediction ensembles. [Complete evaluation](robin_eval01.json).

This result supports predictive value of the complete compressed-TSFM design over the specified direct NLinear control on exposed Robin. It does not identify the pure causal contribution of pretraining, since the models also differ in spatial constraints, architecture and optimization. Nor does it establish deployment preference: direct NLinear has far fewer total parameters, and F0 and LoRA remain more accurate than LEVEL.

### 5.3. Uncompressed microbatch costs

All 117 GPU and 12 CPU measurement rows completed after the fixed numerical-parity checks passed. Full and chunked execution preserved the model's forecast within the specified combined absolute/relative tolerance, including the fixed Robin TEST consistency check. The table shows same-session throughput centers and allocated memory; MiB denotes bytes divided by 1,048,576. Every tested GPU B4 execution configuration is included. Seed-specific block values and all B1 configurations remain in the source records and Appendix C. Converted values are in [cost_comparison.csv](cost_comparison.csv).

| GPU model and B4 execution | M | Throughput, origins/s | Peak allocated, MiB |
|---|---:|---:|---:|
| F0 original | — | 205.27 | 265.49 |
| F0 generic full | 68 | 215.21 | 265.50 |
| F0 generic chunk | 20 | 74.73 | 212.66 |
| F0 generic chunk | 5 | 19.61 | 197.63 |
| Merged LoRA original | — | 204.41 | 265.49 |
| Merged LoRA generic full | 68 | 200.10 | 265.50 |
| Merged LoRA generic chunk | 20 | 64.39 | 212.66 |
| Merged LoRA generic chunk | 5 | 19.35 | 197.63 |
| LEVEL original | — | 254.29 | 214.49 |
| LEVEL generic full | 20 | 226.87 | 214.49 |
| LEVEL generic chunk | 5 | 67.16 | 196.90 |
| Direct NLinear | — | 15,300.47 | 9.50 |

The strongest memory-only objection is supported: F0 and LoRA at M20 allocate less memory than original LEVEL while retaining their lower forecast errors. Thus LEVEL is not necessary to obtain that allocated-memory level. This is not a three-axis domination. M20 uncompressed throughput is far below original or full-wrapper LEVEL, and its reserved memory is 236,978,176 bytes versus LEVEL's 226,492,416 bytes. M5 further reduces allocated memory but adds more sequential calls and lower throughput. Conversely, original uncompressed execution offers better accuracy than LEVEL with larger allocation. Compression therefore changes the choices among accuracy, allocation and throughput rather than minimizing every cost.

The timing advantage is conditional on the implementation and measured session. Original LEVEL B4 block medians across seeds range from 238.59 to 301.07 origins/s. The common generic full wrapper spans 171.75–248.60 for LEVEL and 184.65–236.50 for LoRA; some block/seed orderings reverse. F0 B4 sentinels also drift within blocks, including 4.368 ms before versus 5.084 ms after the first block; B1 first-block sentinels move from 12.808 to 17.104 ms. These are preserved observations, not diagnosed causes. The experiment supports a pronounced throughput cost of small uncompressed chunks, but not an environment-independent stable speedup for compression.

B1 original latency is 13.203 ms for F0, 16.074 ms for LoRA and 15.127 ms for LEVEL. Direct NLinear gives 0.201 ms on GPU and 0.049 ms on CPU; its CPU B4 throughput center is 43,879.94 origins/s. CPU and GPU paths are different deployment choices, and these small times also vary across blocks. NLinear's 24,624 deployed parameters occupy 98,496 tensor bytes, compared with LEVEL's 47,736,106 parameters and 190,944,460 bytes. LEVEL trains only 17,920 parameters but still deploys the frozen backbone. No user-defined error tolerance makes one of these choices automatically acceptable. [Cost summary](cost_summary.json), [GPU rows](cost_gpu_rows.json), [CPU rows](cost_cpu_rows.json), [parity record](microbatch_parity.json).

### 5.4. Corrected Robin error diagnosis

The valid diagnosis calculates each model's loss before averaging seeds. It covers 1,920/1,920 complete target vectors in TEST-A and 1,918/1,920 in TEST-B, using a common mask across all models and no zero-filled targets. The following P/Q values are complete-vector squared-error contributions divided by the channel count. They are diagnostic subset quantities; TEST-B totals differ from the primary masked macro score.

| Period and model | Retained-space P MSE | Discarded-space Q MSE |
|---|---:|---:|
| TEST-A LEVEL | 0.101122 | 0.172555 |
| TEST-A OLD | 0.101122 | 0.461955 |
| TEST-A LoRA | 0.094777 | 0.135757 |
| TEST-A NLinear | 0.141710 | 0.185774 |
| TEST-B LEVEL | 0.142525 | 0.131518 |
| TEST-B OLD | 0.142525 | 0.370653 |
| TEST-B LoRA | 0.106285 | 0.107726 |
| TEST-B NLinear | 0.167380 | 0.157572 |

LEVEL's improvement over OLD lies in Q, consistent with the fixed-projection identity. Its remaining loss against LoRA is not confined to one space: Q contributes more of the gap in TEST-A, whereas P contributes more in TEST-B. This is a localization of errors, not proof that changing K or the main path would improve the endpoint. The direct NLinear disadvantage is also present in both spaces on the complete subset.

The combined channel loss gap against LoRA is concentrated but not a single-channel event. Donald accounts for 39.22% of positive channel excess MSE; the three largest contributors account for 60.23%. In contrast, the largest origin accounts for 7.93% of positive origin excess, and the top ten of 80 origins for 44.79%. These shares use the positive part of per-channel or per-origin MSE differences, not net loss or raw target magnitudes. Negative differences are separately retained. The pattern therefore includes dispersed temporal losses as well as concentrated channel contributions.

Residual-context correlations vary with period, channel and outcome. For example, Donald's last residual level correlates negatively with subsequent mean signed error in both periods, while its Pearson association with future MSE changes sign between periods. This does not establish a residual-level cause, a universal predictor of harm, or a routing rule. All correlations remain descriptive, and the method is unchanged. The initial ensemble-based diagnosis is preserved as invalid for the intended contract; only the corrected record is used here. [Corrected manifest and coverage](robin_diagnostics_corrected_v7.json), [space summary](diagnostics_corrected/robin_complete_row_space_summary_corrected.csv), [channel excess](diagnostics_corrected/robin_channel_excess_concentration_corrected.csv), [origin excess](diagnostics_corrected/robin_origin_excess_concentration_corrected.csv), [correlations](diagnostics_corrected/robin_residual_context_correlations_corrected.csv).

Those correlations concern the total LEVEL forecast error. A separate additive diagnostic explicitly projects each seed's error through Q before computing signed error or squared loss, isolating the residual-correction error on common complete horizons. It reuses the same past features and supplies all-channel scatter atlases, rather than selecting one favorable channel. The distinction prevents total forecast error from being mislabeled as residual-branch error. Neither set of associations was used for a model update. [Q-error diagnostic](robin_q_residual_error_diagnostic_v7.json), [all-channel level/Q-error atlas](diagnostics_residual_error/robin_last_residual_to_q_signed_error_atlas_v7.pdf), [all-channel variability/Q-loss atlas](diagnostics_residual_error/robin_recent24std_to_q_mse_atlas_v7.pdf).

### 5.5. Jena scope test

All twelve planned Jena fits completed and stopped before the cap. The direct NLinear learning rate selected on VAL was 0.001. There was no reserve extension, failed fit or TEST-driven change of arm, channel or period. Training probes improved and selected-output replay errors were zero, so the observed negative result is not attributed to an identified failure to train or restore. This does not prove optimal training for every architecture. Jena remains an exposed scope test, and its errors are not pooled numerically with Robin.

| Jena model | TEST-A MSE | TEST-B MSE | Combined MSE | Combined MAE |
|---|---:|---:|---:|---:|
| LEVEL_RES | 0.422475 | 0.226243 | 0.324359 | 0.320450 |
| OLD_RES | 0.422585 | 0.224024 | 0.323305 | 0.321819 |
| LEVEL_RAW | 0.360648 | 0.221671 | 0.291159 | 0.306634 |
| LEVEL_ONLY | 0.462181 | 0.249258 | 0.355720 | 0.326104 |
| COMPRESS | 0.486863 | 0.275510 | 0.381187 | 0.381092 |
| F0 | 0.370756 | 0.213844 | 0.292300 | 0.264758 |
| LoRA | 0.285445 | 0.195735 | 0.240590 | 0.226584 |
| Direct NLinear | 0.332070 | 0.215899 | 0.273984 | 0.286714 |

LEVEL improves over level-only restoration and COMPRESS, but not over the closer learned alternatives. Its combined MSE is 11.40% above matched RAW, with conditional fixed-seed block interval [7.52%, 14.20%], and 18.39% above direct NLinear, interval [11.86%, 24.08%]. Both paired seeds have these unfavorable combined directions. TEST-B mean differences are smaller and its intervals versus RAW and NLinear include zero; the stronger combined losses are not presented as identical effects in every period. LEVEL is also 10.97% worse than F0 and 34.82% worse than LoRA in combined MSE.

Compared with OLD, LEVEL's combined MSE is 0.33% higher while its MAE is slightly lower. Its near-equality on TEST-A and small loss on TEST-B do not reproduce Robin's large improvement. The relevant result is therefore a failure of the residual-input advantage to extend under this fixed Jena contract, rather than a failure of every learned correction: RAW remains useful and LoRA is the most accurate compared model. No Jena cost grid was run, so Robin timings are not transferred to this task. [Jena evaluation and intervals](jena_eval01.json), [selection](selected_jena.json), [all seed/period scores](accuracy_comparison.csv), [training records](training_summary.csv).

[Figure 1: fixed Robin and Jena accuracy comparisons](figures/accuracy_fixed_comparisons.pdf) shows the eight common comparison roles; Robin's additional latent-linear and SHARED controls remain in the complete accuracy table. [Figure 2: Robin accuracy–throughput and accuracy–memory choices](figures/robin_microbatch_tradeoffs.pdf) keeps the execution configurations and variation visible. [Figure 3: Robin complete-row error spaces](figures/robin_complete_row_error_spaces.pdf) visualizes the restricted P/Q diagnosis. Figure sources and hashes are in [figure_manifest.json](figure_manifest.json); variation whiskers are observed block ranges, not independent-sample confidence intervals.

## 6. Discussion and current scope

The evidence supports improvement of a particular compressed residual architecture on Robin and lower error there than the specified direct NLinear control. The Jena result is an explicit counterexample to a domain-general residual-input advantage: matched RAW and direct NLinear both outperform LEVEL. It narrows, rather than erases, the earlier building-electricity evidence. Neither the positive Robin result nor the negative Jena result licenses a claim about every electricity or weather task. The full Robin inference grid supports a further restriction: LEVEL exchanges accuracy for higher throughput than memory-reduced uncompressed chunks, but direct NLinear is much cheaper and uncompressed chunks remove any claim of uniquely attaining low allocation.

The projected-NLinear identity makes the design boundary explicit: spatial allocation and information restriction are being tested, while the level-normalization operation itself is inherited. P-space error found against LoRA shows why a correction forbidden from that space can leave a meaningful limitation; it does not establish the outcome of an untested modification. Jena's reversal cannot be explained causally by sampling interval, noncircular direction values, related meteorological channels or compression from these comparisons alone. Those are contract limitations and possible future questions, not demonstrated causes or permission for post-evaluation tuning.

Practical value can survive without highest accuracy or lowest latency, but it must be described through actual losses and measured costs. A memory reduction that disappears under an equally valid uncompressed batch policy has a different significance from a reduction obtained only by accepting slower sequential execution. Similarly, a direct linear method that is sufficiently accurate at much lower cost can limit the reason to retain F0 without negating a matched residual-input effect. The study supplies comparisons, not an invented deployment tolerance or a weighted efficiency score.

The scope is one frozen backbone, fixed L/H in observations, two training seeds and a finite selection policy. Robin and Jena differ in sampling interval, physical variables, missingness and source history; a contrast between them does not identify which difference caused transfer or failure. Jena retains numeric wind-direction degrees without circular encoding and several related or derived meteorological variables. Its macro score weights TRAIN-standardized channels equally, not physical-unit errors or independent physical quantities. Jena's prior exposure prevents an independent-confirmation claim. Fixed-seed intervals, complete-case geometry and block timing summaries have their own uncertainty and selection limitations. Artifact checks by the execution team are not independent reproduction.

## 7. Conclusion

The method is a frozen latent TSFM supplemented by a projected low-rank level-preserving predictor, with known overlap with NLinear, latent adapters and factor-residual modeling. Its Robin accuracy gain over direct shared NLinear survives the added control, but its residual-input advantage does not extend to the fixed Jena task. Equivalent-output microbatching removes a claim of unique allocated-memory savings while leaving a conditional Robin accuracy–memory–throughput tradeoff. The result supports retaining a limited contribution for the tested building-electricity setting and withholding broad residual superiority or general deployment recommendations. The remaining decision is whether that restricted contribution warrants further research; a new method, favorable replacement dataset or post-TEST tuning was not used to turn this counterexample into a success.

## References and supporting records

Bibliographic entries are in [references.bib](references.bib), with reading depth and publication status in [CLAIMS_PRIOR_ART.md](CLAIMS_PRIOR_ART.md). [APPENDIX.md](APPENDIX.md) contains complete assumptions, data/selection contracts, grid details and record locations. [report_values.json](report_values.json) connects the measured tables to source files. [PLAN.md](PLAN.md), [APPROVAL.txt](APPROVAL.txt) and [STATUS.md](STATUS.md) distinguish authorization, current progress and final verification/publication status. Execution-team code and numeric checks have their documented scope and are not independent reproduction.
