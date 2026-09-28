# Appendix: conditional geometry, practical controls and reproducibility

Status: the approved Robin and Jena comparisons, cost grid and corrected diagnostics are complete. This appendix accompanies [PAPER_DRAFT.md](PAPER_DRAFT.md); final verification and publication status are recorded separately in [final_checks.json](final_checks.json), [publication_receipt.json](publication_receipt.json) and [STATUS.md](STATUS.md). The prior v6 record is preserved.

## A. Conditional forward identities

### A.1. Projection and the NLinear-style map

Let X be L by C in TRAIN-standardized, post-imputation coordinates. U is C by K with U^T U=I. Define P=UU^T, Q=I-P, and m(V)=e_L^T V, the final row of a context V. The repeated level is P_n(V)=1_n m(V). A shared bias-free temporal linear map can be written G(V)=A V for A of shape H by L. Its channel sharing means the same A acts on every column. Therefore

\[
G(VQ)=A(VQ)=(AV)Q=G(V)Q,
\qquad P_n(VQ)=P_n(V)Q.
\]

For \(\mathcal N_G(V)=P_H(V)+G(V-P_L(V))\), both identities give

\[
\mathcal N_G(XQ)=\mathcal N_G(X)Q.
\]

The model forecast is \(b+\mathcal N_G(XQ)\) with \(b=F_0(XU)U^\top\). Since bP=b and QP=0, multiplying the forecast by P returns b. No approximation to F0 or linearity of F0 is needed. The argument concerns channel projection, not a linear approximation to the backbone.

### A.2. Offset behavior and effective rank

For a constant row a with aP=0, replace X by X+1_L a. The latent input XU is unchanged, and the centered residual is unchanged because the final residual row also increases by a. The resulting forecast increases by 1_H a. This establishes discarded-space constant-offset equivariance in the stated coordinates. It says nothing about a future change not present in the input or an offset with a retained component.

Expanding the level map gives

\[
\mathcal N_G(V)
=\left[A+(\mathbf1_H-A\mathbf1_L)e_L^\top\right]V
=A_{\rm eff}V.
\]

Thus \(A_{\rm eff}\mathbf1_L=\mathbf1_H\). The learned low-rank map A has rank at most 32, and the rank-one adjustment gives rank at most 33 for A_eff. This bound is not a claim that the maximum is attained, nor that LEVEL and OLD have identical expressivity. At zero output-layer initialization A=0, so A_eff is the persistence matrix; OLD instead contributes zero. Their comparison changes the initial function and the parameterized effective map, not just a label on the input.

### A.3. Bias, raw input and masks

A standard shared affine NLinear adds a horizon-specific bias in every channel. For a general Q, adding that bias after input projection is different from projecting the affine output. The commutation statement therefore does not apply to DIRECT_NLINEAR unchanged. A channel-specific map also need not commute with Q, and nonlinear maps generally do not. Learned CURRENT E/D from earlier development need not remain orthogonal. All such settings are outside this proof.

Matched RAW uses \(b+P_H(XQ)+G_{\rm raw}(X-P_L(X))\). If the two fitted maps were equal, the difference between LEVEL and RAW would be confined to the retained space. In the experiment they are fitted independently, so this identity is not a causal explanation of their observed error difference. Both must retain their full measured outputs and losses.

For an observed complete row error e, orthogonal projection gives \(\|e\|^2=\|eP\|^2+\|eQ\|^2\). The diagnostic divides each term by C and averages over the same admitted rows. A missing-target mask or channel-dependent observation count changes the weighting and need not commute with P. Hence this decomposition cannot be applied to zero-filled targets and called a decomposition of the full masked channel-macro MSE. Per-period coverage and the complete-row total are required alongside the original endpoint. Imputation also prevents interpreting the post-imputation offset property as unconditional invariance to shifts in raw missing observations.

## B. Data, fitting and exposure contracts

### B.1. Robin reuse

Robin has C=17, K=5, hourly observations, L512/H48, and the existing v6 source and channel identifiers. The intervals are TRAIN [2016-01-23,2016-04-15), VAL [2016-04-17,2016-05-18), TEST-A [2016-05-20,2016-06-30) and TEST-B [2016-06-30,2016-08-10), with context available from 2016-01-01. The existing counts are 1,945 TRAIN origins, 30 VAL origins and 40 origins in each TEST period. Target intervals are half-open, evaluation stride is 24 observations, and each full horizon must remain in its period.

Existing arrays, the fourteen-model v6 evaluation, five required backbone/LEVEL/LoRA checkpoints and corresponding receipts are reused through the exact v6 manifests. The approval-stage existence/hash review was not a new prediction or score. The completed v7 direct NLinear scores are a follow-up designed after v6 exposure. A downloaded complete source file is not by itself evidence that every contained group was previously selected on, but actual Robin exposure after v6 is unequivocal.

| Existing record | Role |
|---|---|
| [v6 data contract](../tsfm_peft_level_confirmation_v6_20260928/data_contract.json) | Source, channels, split and array identity |
| [v6 selection](../tsfm_peft_level_confirmation_v6_20260928/selected.json) | Selected runs and checkpoints |
| [v6 evaluation](../tsfm_peft_level_confirmation_v6_20260928/evaluation01.json) | Exact old predictions, losses and paired comparisons |
| [v6 cost values](../tsfm_peft_level_confirmation_v6_20260928/report_values.json) | Historical original-path cost summary; not a same-session v7 comparator |

### B.2. Jena 2024

The approved local file is `E:/CODING/proj/mltimeseries/data/jena_mpi_roof/mpi_roof_2024.csv`, 11,068,144 bytes, SHA256 `65a47db98ad85b0600f2deab1e11055af8fef9c5aedceadbdcd0011748120558`. It contains 52,704 timestamps from 2024-01-01 00:10 to 2025-01-01 00:00 on a ten-minute grid. The source is MPI-BGC Jena, WS Beutenberg; source-naive timestamps are preserved. Its source and exposure review is recorded in PLAN. No current remote byte-identity check or pretraining-disjointness claim is added here.

All 21 numerical variables are retained: pressure; air, potential and dew-point temperature; relative humidity; maximum, actual and deficit vapor pressure; specific humidity; water concentration; density; wind and maximum wind speed; wind direction; rainfall amount and raining duration; shortwave radiation; PAR and maximum PAR; logger temperature; and carbon dioxide. Exact CSV headers remain in the data contract. Wind direction is not converted to sine/cosine features, and observed zeros are not treated as missing. Nonfinite and documented `-9999`-type sentinel entries are masked before TRAIN statistics. Inputs are TRAIN-mean imputed, while targets preserve observed masks.

| Role | Half-open target interval | Time-index origin count |
|---|---|---:|
| TRAIN | [2024-01-01 00:10, 2024-07-01 00:00) | 25,648 |
| VAL | [2024-07-01 00:00, 2024-09-01 00:00) | 371 |
| TEST-A | [2024-09-01 00:00, 2024-11-01 00:00) | 365 |
| TEST-B | [2024-11-01 00:00, 2025-01-01 00:10) | 365 |

TRAIN candidates have stride one; each epoch samples 512 origins balanced over the 144 daily phases. VAL and TEST use stride 24 observations, or four hours. Targets cannot cross period boundaries, but already available context can cross them. C=21 gives K=6. TRAIN-only observation rate and nonconstant checks passed at planning stage; these are data-eligibility facts, not forecast results.

The identical source hash was used by an earlier local experiment whose audit includes TEST use (`results/internal_adaptation_gap_v1_20260922/jena/DATA_AUDIT.json`). Its backbone, horizon and split differ from v7, so its scores are not reused. The historical exposure remains relevant even when the experimental setup changes. Jena therefore tests scope on an exposed other-domain dataset.

### B.3. Fit selection and repair

| Dataset | Model | Independent settings | Base attempts |
|---|---|---|---:|
| Robin | DIRECT_NLINEAR | Two learning rates × two seeds | 4 |
| Jena | LEVEL_RES, OLD_RES, LEVEL_RAW, LoRA | Four families × two seeds | 8 |
| Jena | DIRECT_NLINEAR | Two learning rates × two seeds | 4 |
| Jena | F0, COMPRESS, LEVEL_ONLY | Frozen/statistical references | 0 |

The seeds are 92601 and 92602. NLinear learning rates are 0.001 and 0.0001; an exact best-VAL mean tie favors 0.001. Each seed's checkpoint includes step 0 and selects the earliest strict minimum full-VAL MSE. Shared initial NLinear tensors and epoch-origin orders are explicitly identified, not merely assumed identical from seed labels. G's initial tensors are also shared between corresponding fresh Jena controls. No previous site's learned adapter is used as initialization.

The common optimizer is AdamW with weight decay zero, clipping at one and batch four. The plateau scheduler uses factor 0.5, patience two and relative threshold 0.0001. Training stops after six strict nonimproving full-VAL checks or at 120 epochs. LoRA uses its native quantile loss while other candidates use masked original-channel macro MSE. A fixed TRAIN probe and initial/selected VAL losses distinguish nonlearning from a valid unfavorable result; differing epoch training samples are not a fixed convergence curve.

Four reserve attempts are limited to technical recovery or a single extension of an eligible cap-limited run to 240 epochs. Eligibility requires reaching epoch 120 without plateau stopping and 120−best_epoch<6. Extension resumes the final optimizer, scheduler and RNG state while preserving the old best. All eligible runs use the same rule; if they exceed the remaining reserve, favorable arms are not selectively extended. Every real-data fitting failure, smoke, restart or extension consumes an attempt. TEST is sealed separately for each dataset after its choices, so finishing A does not block the approved B fitting.

### B.4. Completed Robin direct NLinear fits

The four fits stopped after six nonimproving VAL checks, before the 120-epoch cap. No fit met the extension criterion. The fixed TRAIN probe contains 96 origins, and the columns below refer to the same probe before fitting and at the selected checkpoint. The VAL learning-rate choice used means 0.435458 for 0.001 and 0.437953 for 0.0001, selecting 0.001 before TEST evaluation. The two learning rates share the recorded initial tensors within each seed.

| LR | Seed | Epochs run | Selected epoch | Initial TRAIN MSE | Selected TRAIN MSE | Initial VAL MSE | Best VAL MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.001 | 92601 | 13 | 7 | 2.385128 | 0.219901 | 1.996529 | 0.437619 |
| 0.001 | 92602 | 36 | 30 | 2.281710 | 0.197772 | 1.844349 | 0.433297 |
| 0.0001 | 92601 | 32 | 26 | 2.385128 | 0.208701 | 1.996529 | 0.435873 |
| 0.0001 | 92602 | 40 | 34 | 2.281710 | 0.210984 | 1.844349 | 0.440033 |

Each result records 24,624 trainable and total parameters, actual temporal-weight and bias updates, maximum VAL replay error zero, and restoration of model, optimizer, scheduler and RNG states without an extra optimizer update. These checks show that the implemented baseline learned and was restored; they do not certify a global optimum. The selected-model combined MSE/MAE is 0.334129/0.362302 for seed 92601 and 0.321319/0.344519 for seed 92602.

Sources: [0.001/92601](runs/v7_robin_direct_nlinear_lr0.001_92601/result.json), [0.001/92602](runs/v7_robin_direct_nlinear_lr0.001_92602/result.json), [0.0001/92601](runs/v7_robin_direct_nlinear_lr0.0001_92601/result.json), [0.0001/92602](runs/v7_robin_direct_nlinear_lr0.0001_92602/result.json), [selection](selected_robin.json), [extension decision](extension_decision_robin.json), [evaluation](robin_eval01.json). Prediction and checkpoint hashes are stored with these records.

### B.5. Completed Jena fitting and all-period outcomes

All twelve Jena fits stopped before epoch 120, after the prescribed patience, and none was eligible for extension. Every selected checkpoint has recorded output replay error zero. LEVEL used selected epochs 11/9; OLD 19/20; RAW 5/16; and LoRA 39/17 for seeds 92601/92602. NLinear's selected LR0.001 checkpoints are epochs 34/39, chosen by mean best-VAL MSE 0.488383 versus 0.506151 for LR0.0001. The lower-LR checkpoints at epochs 50/49 were not selected through TEST. The observed fixed-probe improvements establish successful learning under this recipe, not complete convergence or optimal tuning. [All twelve training rows](training_summary.csv), [selection](selected_jena.json), [extension decision](extension_decision_jena.json).

The following seed-wise combined losses complement the main period-MSE table. Losses precede seed averaging throughout.

| Jena family | Seed 92601 MSE | Seed 92602 MSE | Seed 92601 MAE | Seed 92602 MAE |
|---|---:|---:|---:|---:|
| LEVEL_RES | 0.324219 | 0.324499 | 0.320484 | 0.320416 |
| OLD_RES | 0.323437 | 0.323172 | 0.322329 | 0.321309 |
| LEVEL_RAW | 0.291516 | 0.290802 | 0.307515 | 0.305753 |
| LoRA | 0.240184 | 0.240996 | 0.226027 | 0.227141 |
| Direct NLinear | 0.273152 | 0.274817 | 0.288104 | 0.285324 |

| Jena family | TEST-A mean-seed MAE | TEST-B mean-seed MAE |
|---|---:|---:|
| LEVEL_RES | 0.381041 | 0.259859 |
| OLD_RES | 0.383491 | 0.260147 |
| LEVEL_RAW | 0.353870 | 0.259398 |
| LEVEL_ONLY | 0.388289 | 0.263920 |
| COMPRESS | 0.440826 | 0.321358 |
| F0 | 0.313130 | 0.216386 |
| LoRA | 0.253981 | 0.199188 |
| Direct NLinear | 0.322967 | 0.250461 |

Combined LEVEL relative-MSE intervals from 2,000 within-period, 42-origin block resamples are [0.05%, 0.78%] versus OLD, [7.52%, 14.20%] versus RAW, [−10.03%, −7.30%] versus ONLY, [−19.17%, −10.30%] versus COMPRESS, [8.48%, 13.44%] versus F0, [26.75%, 41.87%] versus LoRA and [11.86%, 24.08%] versus NLinear. These are conditional on the fitted models. The small OLD comparison is not an assertion of practically meaningful harm. For RAW and NLinear, TEST-B intervals include zero despite unfavorable mean differences; all periods and seeds remain in [jena_eval01.json](jena_eval01.json) and [accuracy_comparison.csv](accuracy_comparison.csv). No favorable period replaced the combined endpoint.

## C. Inference grid, parity and measurement

### C.1. Fixed execution grid

The underlying task batch B counts complete multivariate contexts. M counts independent series rows after flattening B×C for an uncompressed TSFM or B×K for LEVEL. Neither L nor H is split. Generic-wrapper full and chunk configurations share the same median-output buffer policy; original execution remains a distinct implementation reference.

| Model instances | Task B=1 | Task B=4 |
|---|---|---|
| F0 once; merged LoRA twice | Original, full M17, M5 | Original, full M68, M20, M5 |
| LEVEL twice | Original, full M5 | Original, full M20, M5 |
| Selected DIRECT_NLINEAR twice | Original GPU | Original GPU |

This is 35 primary rows per order block, plus four F0 before/after sentinel rows, repeated in three blocks: all 117 GPU rows completed, along with the 12 NLinear CPU rows. The records are [cost_gpu_rows.json](cost_gpu_rows.json) and [cost_cpu_rows.json](cost_cpu_rows.json). There is no Jena M-grid and no transfer of a favorable Robin M to Jena evaluation.

The chunk wrapper assigns every row to its original position, retains only the needed median horizon in a preallocated GPU buffer, and releases the raw quantile output before the next chunk. It returns the full original-channel CPU tensor only after all necessary calls and LEVEL's decoding/residual work. It does not flush the allocator, reload a model or return to CPU on every chunk. The original-to-wrapper difference contains implementation effects; the full-wrapper-to-chunk difference addresses batching more directly.

### C.2. Equivalence boundary

Before timing, all grid configurations replay 24 common VAL origins against the original selected model. The fixed tolerance is absolute 0.00001 and relative 0.0001; maximum differences and violating-element counts are retained. LoRA merge parity is checked separately from chunk parity. An error does not authorize increasing tolerance until it passes. Qualified full/chunk paths are also compared on the fixed Robin 80 TEST origins, with prediction and MSE/MAE differences recorded as inference consistency, not a new configuration-selection endpoint.

Independent-row mathematics does not promise bitwise equality after floating-point batch-shape changes. Conversely, an algebraic argument alone cannot validate row ordering or output slicing. Invalid execution paths are repaired with their first failure preserved or marked non-equivalent. The original models' valid forecasts do not become invalid merely because a new wrapper fails.

The completed [microbatch_parity.json](microbatch_parity.json) has status PASS under the specified tolerance and records VAL and fixed-TEST discrepancies for the grid, including CPU NLinear equivalence. Related code, candidate checkpoints, selection and data contracts are hash-bound to the timing stage. The recorded TEST MSE differences are inference-consistency quantities and did not select M or any model. This implementation-team check is not independent reproduction.

### C.3. Cost scope and variation

The primary boundary is standardized FP32 CPU input → GPU transfer → full model execution → complete original-channel CPU output. It includes sequential calls, slicing, copying, recombination and synchronization. Model loading, disk access, scoring and ledger updates are outside. GPU, dtype, TF32 and CPU-thread settings are fixed; each shape receives ten warm-up calls and twenty passes over the 24 VAL origins in each order block. Peak allocation is reset after normal warm-up.

For a fit/configuration, the timing center is the median of three block medians. For a two-seed family, those fit centers are averaged. Every block and M remains visible. No single favorable seed, pass or M is retrospectively promoted to a stable winner. CPU NLinear is a distinct deployment option. Peak allocated/reserved, post-load resident allocation, process memory and tensor bytes have separate interpretations and are not substituted for each other.

Only one additional resident-input or profiler diagnosis, within the approved ten GPU minutes, is available if needed. CUDA-event intervals need not equal pure kernel time, and a different diagnostic scope does not replace deployment measurements. Prior v6 timing is historical context and is not a denominator for a v7 improvement ratio.

### C.4. Completed measurements and variation

The main text reports every B4 GPU configuration. These are the corresponding B1 centers; M is still an independent-series row count, and MiB equals 1,048,576 bytes. Allocated peaks were unchanged across the three recorded blocks for each configuration; reserved and process memory remain distinct recorded quantities. Unit conversions are recorded in [cost_comparison.csv](cost_comparison.csv).

| B1 model and execution | M | Latency, ms/origin | Peak allocated, MiB |
|---|---:|---:|---:|
| F0 original | — | 13.203 | 209.87 |
| F0 generic full | 17 | 14.938 | 209.87 |
| F0 generic chunk | 5 | 58.425 | 197.52 |
| Merged LoRA original | — | 16.074 | 209.87 |
| Merged LoRA generic full | 17 | 15.537 | 209.87 |
| Merged LoRA generic chunk | 5 | 56.151 | 197.52 |
| LEVEL original | — | 15.127 | 196.63 |
| LEVEL generic full | 5 | 15.455 | 196.63 |
| Direct NLinear GPU | — | 0.201 | 9.29 |
| Direct NLinear CPU | — | 0.049 | Not a GPU allocation |

For B4, the observed peak-reserved bytes are 306,184,192 for original F0, 308,281,344 for original merged LoRA, and 226,492,416 for original LEVEL. F0/LoRA M20 reserve 236,978,176 despite their lower allocation than LEVEL; NLinear GPU reserves 23,068,672. The distinction matters to a hardware-capacity claim. The summary's mean of per-seed median peaks is a center, not a worst-case bound; raw observations and their range are retained. These finite measurements do not establish a universal memory bound.

| Family | Trainable parameters before merge | Deployed parameters | Deployed tensor bytes |
|---|---:|---:|---:|
| F0 | 0 | 47,718,016 | 190,872,100 |
| LoRA | 294,912 | 47,718,016 | 190,872,100 |
| LEVEL | 17,920 | 47,736,106 | 190,944,460 |
| Direct NLinear | 24,624 | 24,624 | 98,496 |

The LoRA trained model has 48,012,928 total parameters before merging; the deployment column uses the returned merged model. Stored tensor bytes include buffers and are distinct from checkpoint archive size, CUDA peak memory and process memory. LEVEL's low trainable count does not remove the backbone from deployment.

All six F0 pre/post sentinel pairs are shown below in ms/origin. They quantify the environment's observed within-block timing changes without assigning a cause.

| Order block | B1 before | B1 after | B4 before | B4 after |
|---|---:|---:|---:|---:|
| 0 | 12.808 | 17.104 | 4.368 | 5.084 |
| 1 | 15.643 | 17.325 | 4.870 | 4.855 |
| 2 | 15.366 | 14.972 | 4.891 | 4.916 |

Small raw time differences are therefore not sufficient to declare a stable ranking. For example, full-wrapper LEVEL seed 92602 spans 171.75–248.60 B4 origins/s, while full-wrapper LoRA seed 92602 spans 201.32–236.50. The original path and common full wrapper do not have identical rankings in every block. NLinear CPU B4 block medians across seeds span 36,122.96–47,193.70 origins/s; GPU B4 spans 8,075.10–23,101.85. Both are much faster than the TSFM paths here, but their absolute measurements are also variable. Complete distributions and measurement order are preserved; no block was excluded to improve an outcome. [Summary](cost_summary.json), [GPU measurements](cost_gpu_rows.json), [CPU measurements](cost_cpu_rows.json).

## D. Error analysis and uncertainty

The Robin diagnosis retains channel, origin and period MSE/MAE; signed mean error; positive excess loss, negative differences and their net difference. This avoids conflating large absolute LEVEL errors with losses uniquely worse than a comparator. The spatial calculation uses all-channel-observed `(origin,horizon)` rows shared by the compared methods, reports period coverage, and keeps the full masked macro endpoint separate. Losses are computed per seed before averaging; averaging predictions is not used as a substitute.

Past residual last level and fixed recent-24-hour mean, standard deviation and change are descriptive inputs for plots and summaries. Total-forecast and specifically Q-projected correction errors are distinguished below; associations with either cannot establish a cause or authorize new routing. Complete-row subsets and overlapping forecasts are not independent samples. Robin uses seven-origin and Jena 42-origin paired blocks, 2,000 resamples and within-period boundaries under the fixed protocol. Results from the two domains are reported separately rather than pooled into an arbitrary composite.

The completed corrected analysis uses 3,838 of 3,840 `(origin,horizon)` rows: all 1,920 TEST-A rows and 1,918 TEST-B rows. The shared complete-row mask is identical for every model. Per-member P/Q losses are calculated before averaging, with detailed rows retained in the private cache and compact summaries published. The main text's P/Q table is drawn from [the corrected summary](diagnostics_corrected/robin_complete_row_space_summary_corrected.csv), not from the invalid prediction-ensemble version. Near-equality of LEVEL/OLD/COMPRESS P losses is consistent with ideal orthonormal-PCA algebra up to numerical errors; it is not a claim of bitwise equality.

The positive channel excess against LoRA sums to 0.886383 and the negative contributions to −0.009428 across channel-wise combined MSE differences; their net is 0.876955 before division by 17. The origin calculation instead aggregates per-origin MSE differences and therefore has its own denominator and missingness weights. Its positive sum is 4.940954, its negative sum −0.813883, and its net 4.127071 across 80 origins. These two descriptive sums must not be substituted for the main pooled observed-count endpoint or directly compared as magnitudes. Concentration shares are normalized by the appropriate positive sum, retaining negative improvements separately. [Channel concentration](diagnostics_corrected/robin_channel_excess_concentration_corrected.csv), [origin concentration](diagnostics_corrected/robin_origin_excess_concentration_corrected.csv).

The correlation file uses 40 origins per period and channel for each fixed context feature and outcome; temporal overlap and shared channels preclude treating all rows as independent. In Donald, Pearson correlations of last residual level with future mean signed error are −0.312147 and −0.239701 in TEST-A/B, while its correlations with future MSE are −0.074506 and 0.071256. This specific example illustrates dependence on outcome and period rather than identifying a reliable causal mechanism. The full table includes all channels and predeclared features. [Correlation table](diagnostics_corrected/robin_residual_context_correlations_corrected.csv).

That corrected table's outcomes are total LEVEL forecast errors. The additive [Q-error diagnostic](robin_q_residual_error_diagnostic_v7.json) instead computes \(e_Q^{(s)}=(\widehat Y^{(s)}-Y)Q\) for each seed on common complete target horizons. Under the fixed projection, this equals the residual correction minus \(YQ\), up to numerical precision. It then aggregates each seed's squared/signed error over observed complete horizons and averages seeds. It does not square a seed-mean prediction and does not fill missing targets. The context features are reused without introducing a new selection criterion.

This produces 1,360 origin–channel rows over all 17 channels, with the same 3,838/3,840 complete-horizon coverage. In the Donald example, last-level correlations with Q signed error are −0.356473/−0.282334 and with Q MSE −0.121751/0.076653 for TEST-A/B. Recent-24-hour variability has Q-MSE correlations 0.257587/0.223105. These illustrative associations are not universally signed effects, formal causal evidence or validated routing predictors. The complete [Q-error correlation CSV](diagnostics_residual_error/robin_level_res_q_residual_error_correlations_v7.csv) and the [last-level atlas](diagnostics_residual_error/robin_last_residual_to_q_signed_error_atlas_v7.pdf) and [variability atlas](diagnostics_residual_error/robin_recent24std_to_q_mse_atlas_v7.pdf) retain every channel and both periods.

## E. Resource, provenance and result status

The approved budget is 16 base real-data attempts plus four restricted reserves, at most 14,400 GPU job-seconds, one concurrent neural fit, six synthetic optimizer sessions of at most ten total updates each, 900 CPU-check seconds and 5 GiB of new storage. No download, package installation, driver or global-environment change is authorized. Parent GPU jobs and their internal fit intervals are not added twice. The current execution state belongs in STATUS and the shared ledger; this appendix does not replace them.

Run configurations, code/source/model hashes, initial tensors, sample orders, curves, checkpoints, predictions, selection seals, exposure and failure receipts remain attached to actual runs. Large arrays and weights stay in the private cache. Completed source results populate the tables and [report_values.json](report_values.json). All 16 base fits completed without consuming a reserve; there was no cap extension. Final resource totals belong to the shared ledger and STATUS because later analysis and verification also consume their own budget categories. Repairs preserve their original failure and any affected result is marked invalid or corrected; implementation-team checks are not independent reproduction.

Robin [evaluation](robin_eval01.json), Jena [evaluation](jena_eval01.json), the full cost grid and [corrected diagnosis](robin_diagnostics_corrected_v7.json), including its Q-error supplement, are complete. The first [diagnostic record](robin_diagnostics_v7.json) and its CSV files are preserved as an invalid attempt for the intended mean-seed-loss analysis: predictions were averaged before errors were calculated. Those quantities are not used in the manuscript's findings; [diagnostic_correction.json](diagnostic_correction.json) connects the original attempt and the corrected per-member analysis. This diagnosis error does not affect the separate valid Robin evaluation. Final verification and publication status must be read from [final_checks.json](final_checks.json), [publication_receipt.json](publication_receipt.json) and [STATUS.md](STATUS.md), not inferred from the existence of this manuscript.
