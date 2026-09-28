# Level-Preserving Residual Adaptation for Compressed Time-Series Foundation Models

**Working English manuscript.** Development results and Robin accuracy and deployment costs below are measured results. Artifact verification is reported separately in [final_checks.json](final_checks.json). Author names and affiliations have not been assigned. Completing this fixed comparison and draft does not imply submission readiness. The approved execution contract is [PLAN.md](PLAN.md), and source status is documented in [CLAIMS_PRIOR_ART.md](CLAIMS_PRIOR_ART.md).

## Abstract

Applying a univariate time-series foundation model independently to many channels can be expensive. Channel compression reduces the number of foundation-model inputs, but decoding a forecast from a small latent space restricts the channel patterns it can represent. We study a small residual adapter that preserves the last observed reconstruction-residual level and predicts changes through a shared low-rank temporal linear map. Its components are established principles; the contribution is their use at the channel-compression bottleneck and an explicit separation of level, residual-input, backbone, and cost comparisons. Under fixed orthonormal PCA, the correction remains in the discarded channel subspace and is equivariant to constant offsets in that subspace. Development on Hog reduced MSE from 3.220290 to 2.666809, but Electricity worsened and Hog retained disadvantages against LoRA. We then fixed the design before evaluating 17 office electricity channels at Robin. LEVEL achieved MSE 0.275196 versus 0.538776 for the previous residual adapter, 0.346630 for the matched raw-input control, and 0.349041 for level-only restoration. It also outperformed the tested small linear alternatives. Nevertheless, MSE remained 20.12% above uncompressed F0 and 23.07% above LoRA, with worse MAE. Peak allocated GPU memory at batch 4 was 19.21% lower than both uncompressed models, while a stable throughput advantage was not established. These results support the specific residual design and a bounded accuracy–memory choice on this additional group, not highest accuracy, reliable acceleration, or universal transfer.

## 1. Introduction

A pretrained univariate forecaster can be reused across the channels of a multivariate task without adapting its large parameter set. This does not make prediction cost independent of the number of channels. A feature-space encoder offers a direct way to control that cost: forecast a smaller set of latent series and decode their forecasts to the original channels. The practical question is what to do with information omitted by this representation.

For a fixed linear decoder, every decoded future lies in its channel subspace. A simple compressed forecast therefore cannot express a component orthogonal to that subspace, even if such a component is present in the recent observed history. This observation motivates a complementary path, but does not establish that the omitted component is predictable or worth modeling. Some residual patterns are noise; others may be adequately handled by persistence or a small forecaster without a foundation model. Those alternatives must be tested rather than dismissed by the small number of trainable parameters.

We examine a level-preserving reconstruction-residual adapter. A frozen foundation model forecasts PCA components. A small, shared temporal map predicts changes in the reconstruction residual, and the most recent residual value is restored across the horizon. This separates the retained-channel forecast from a complementary residual forecast without adding learned parameters to the previous residual branch. Level centering and restoration are inherited from established forecasting practice, especially NLinear; they are not presented as a new normalization principle.

The empirical motivation arose during development on Hog, a building-electricity group where an earlier compressed residual adapter lost accuracy relative to uncompressed F0 and LoRA. LEVEL reduced that loss. However, it did not improve all development datasets or all metrics, and both proposed and raw-input variants benefited from its common level treatment. We therefore distinguish the usefulness of level treatment, the incremental value of residual input, and the practical justification for retaining the frozen TSFM. We then freeze the design and selection policy for one new group, rather than continue tuning Hog or choose a favorable new group by forecast performance.

Our contribution is a concrete adaptation design and its bounded evaluation. We provide conditional structural properties and matched controls with explicit initialization differences. The Robin comparison supports gains over the previous residual adapter, matched RAW, persistence-only restoration, and the tested linear alternatives, while preserving a clear loss against uncompressed F0 and LoRA. This is an additional fixed-group result beyond the development evidence, not a reason to erase the development counterexamples. Measured peak allocated memory is lower than for the uncompressed TSFMs, but their deployment throughput is essentially matched within observed variation. The remaining practical argument is an accuracy–memory choice, with much cheaper linear alternatives also retained in view.

## 2. Related work

AdaPTS adapts frozen univariate foundation models through latent encoders and decoders, including dimension reduction and prediction-loss adaptation of E/D. These are direct predecessors of our main path. We examine an additional observed reconstruction-residual path. Our COMPRESS reference isolates that structural main path; it is not a reproduction of the complete probabilistic AdaPTS framework or its tuning. [Benechehab et al., 2025](https://proceedings.mlr.press/v267/benechehab25a.html)

NLinear subtracts the last input value, applies a temporal linear predictor, and restores that value. We use this established operation on a channel-reconstruction residual beside a compressed frozen TSFM. RevIN instead removes and restores instance mean and scale, with optional affine parameters. LEVEL does not implement that full transformation. These connections constrain the contribution to a specific composition and comparison, not priority for remove-and-restore normalization. [Zeng et al., 2023](https://ojs.aaai.org/index.php/AAAI/article/view/26317), [Kim et al., 2022](https://seharanul17.github.io/RevIN/)

Lightweight post-processing also appears in online forecast adaptation. ELF combines a separate forecaster with foundation-model predictions using online feedback. EPOC, currently a preprint, retains coefficients and an endpoint from completed forecast-error blocks for online correction. Our input is a reconstruction residual already available at forecast time, and our adapter is fitted only on TRAIN. We make no priority claim for residual correction or endpoint preservation in general. [Lee et al., 2025](https://proceedings.mlr.press/v267/lee25ag.html), [Fujimoto and Nishi, 2026](https://arxiv.org/abs/2609.30929)

Repeated optimization of a finite validation criterion can induce selection bias. We consequently label Hog, Bull, and Electricity as development data and separate their evidence from the frozen Robin evaluation. This procedural separation limits one source of bias; it does not prove global dataset independence or absence from backbone pretraining. [Cawley and Talbot, 2010](https://jmlr.org/papers/v11/cawley10a.html)

## 3. Method

### 3.1. Compressed frozen main path

Let \(X\in\mathbb R^{L\times C}\) be the TRAIN-standardized context. Let \(U\in\mathbb R^{C\times K}\) contain orthonormal PCA directions fitted to TRAIN. Define

\[
E(X)=XU,\qquad D(Z)=ZU^\top,\qquad
b(X)=D(F_0(E(X))),\qquad r(X)=X-D(E(X)).
\]

The backbone \(F_0\) is Chronos-Bolt-small at revision `772f3d25d38aec6d914c8949dab4462e2d46f5d8`, frozen and evaluated in FP32 with dropout disabled. We take its median-quantile output for the first 48 future observations. The new-group design fixes E/D to the fresh Robin TRAIN PCA. It does not select between fixed and trainable E/D using Robin scores.

For a matrix \(u\), let \(P_n(u)\) repeat its final row \(n\) times. That row is detached when used as the level statistic. With \(L=512\), \(H=48\), the proposed forecast is

\[
\widehat Y_{\mathrm{LEVEL\_RES}}
=b(X)+P_H(r(X))+G_{\mathrm{res}}\bigl(r(X)-P_L(r(X))\bigr).
\]

The map G consists of bias-free linear layers 512→32→48, shared across channels and without an activation. It has 17,920 trainable parameters. Its first matrix is randomly initialized and its output matrix is zero. The initial forecast is therefore \(b+P_H(r)\), not the compressed forecast alone. The level term is computed from available context, including TRAIN-mean imputation if the final input observation is missing. No future statistic is used.

### 3.2. Controls and interpretation

The matched raw-input control is

\[
\widehat Y_{\mathrm{LEVEL\_RAW}}
=b(X)+P_H(r(X))+G_{\mathrm{raw}}\bigl(X-P_L(X)\bigr).
\]

Both models restore the same **residual** level. The raw control does not add the raw input level on top of the main forecast. Their initial G tensors, parameter counts, sampling orders, optimizers, and selection opportunities are matched. After independent fitting their temporal matrices can differ.

OLD_RES uses \(b+G_{\mathrm{old}}(r)\) and starts from b. Its comparison with LEVEL measures the combined effect of persistence, centering, altered effective function class, and initialization; it is not a pure normalization intervention. LEVEL_ONLY uses \(b+P_H(r)\) without any predictive fitting, while COMPRESS returns b. The LEVEL_ONLY comparison asks whether learned residual changes improve on the common initial function.

Uncompressed F0 and merged LoRA provide practical TSFM alternatives. LoRA learns low-rank weight updates while retaining the pretrained weights; our implementation adapts the established q/v targets rather than claiming a new LoRA variant. [Hu et al., 2021](https://github.com/microsoft/LoRA) LEVEL_LINEAR_RES replaces F0 by a channel-shared affine temporal predictor on the latent series, retaining the same fixed E/D and LEVEL residual path. Its independent \(T\) matrix has 24,624 parameters and uses the established v3 context mean/scale normalization and inverse transformation. It is therefore described as a normalized linear replacement, not a globally linear function or the pure causal removal of pretraining. T and G are distinct parameter objects, and this deployment path contains no unused F0 instance.

A full-TRAIN shared affine ridge predictor is a second TSFM-free alternative. It operates directly on the original standardized channels with 512→48 temporal weights and an unpenalized intercept. The penalties 0.001 and 0.1 are the only candidates and are selected by VAL. This prevents the practical comparison from depending solely on a small neural alternative with a potentially difficult optimization path.

### 3.3. Conditional structural properties

For clarity, transpose the context in this subsection so \(X_c\in\mathbb R^{C\times L}\). Put \(P=UU^\top\), \(Q=I-P\), \(R=QX_c\), and \(m=R e_L\). Write the shared temporal map as \(G(R)=RA^\top\). The LEVEL correction is

\[
c=m\mathbf1_H^\top+(R-m\mathbf1_L^\top)A^\top.
\]

**Retained-subspace preservation.** Since \(PR=0\) and \(Pm=0\), \(Pc=0\). The main forecast lies in the retained subspace, so \(P\widehat Y=b\). Thus this correction cannot repair an error inside the retained PCA space; that is a limitation as well as a clean separation.

**Discarded-subspace offset equivariance.** If \(Pa=0\), adding \(a\mathbf1_L^\top\) to the post-imputation context leaves its latent input unchanged. It shifts m by a and leaves the centered residual unchanged. Consequently the forecast shifts by \(a\mathbf1_H^\top\). The property does not cover arbitrary retained-space offsets, scaling, preprocessing before imputation, or future level changes.

The equivalent residual temporal matrix is

\[
A_{\rm eff}=A+(\mathbf1_H-A\mathbf1_L)e_L^\top,
\qquad A_{\rm eff}\mathbf1_L=\mathbf1_H.
\]

Although \(\operatorname{rank}(A)\le32\), this effective matrix can have rank 33. Unchanged learned-parameter counts therefore do not imply identical function spaces. These statements depend on fixed orthonormal PCA and shared bias-free temporal linear maps. They do not automatically apply to the trainable E/D used in the earlier Electricity development setting. They also do not decompose the primary masked macro loss, whose missingness-dependent weights need not respect Euclidean orthogonality. All geometric claims concern standardized coordinates.

## 4. Experimental protocol

### 4.1. Development and new-group separation

Hog, Bull and Electricity have been repeatedly used during development. Their published v5 measurements are reused without retraining; they are not independent confirmation. The new comparison fixes the Robin site in BDG2 raw electricity. BDG2 supplies hourly building-meter series and distinct raw and cleaned variants. We use the pinned raw source and do not silently substitute the cleaned data. [Miller et al., 2020](https://doi.org/10.1038/s41597-020-00712-x)

Robin was selected using site identifiers, searched exposure records, and TRAIN-only eligibility, not covariance structure or forecast performance. The group contains 17 office-labelled channels within educational facilities; K=ceil(17/4)=5. Eligible channels have TRAIN observation rate at least 0.95 and are nonconstant. All 17 eligible identifiers are fixed in PLAN and the data contract. The metadata assigns Europe/London, while the supplied hourly timestamps are local-naive; we preserve those timestamps rather than reconstructing daylight-saving-time offsets. One educational-office site is not a new-domain benchmark of building energy in general.

The bounded record search found no use of Robin in selecting this method in the searched accessible repositories. It did not establish exhaustive historical non-use, pretraining disjointness, or statistical independence from all other BDG2 sites. Keeping a full source CSV is distinguished from using Robin performance for selection.

### 4.2. Splits, targets, and metrics

The TRAIN interval is [2016-01-23, 2016-04-15), with earlier observations from January 1 available as context. VAL is [2016-04-17, 2016-05-18). TEST-A is [2016-05-20, 2016-06-30), and TEST-B is [2016-06-30, 2016-08-10). Every 48-observation target must fit within its assigned interval. Evaluation origins advance by 24 observations; the pre-specified time-only counts are 1,945 TRAIN origins, 30 VAL origins, 40 TEST-A origins and 40 TEST-B origins, subject to the recorded index verification. Subsequent test origins can use already observed history, but no weights are updated from TEST targets.

Means, standard deviations, PCA and input-imputation values come only from TRAIN. Missing inputs become the TRAIN mean, or zero after standardization. Zero-valued observations remain observed. Targets retain their original finite mask and are not interpolated for scoring. The main endpoint is channel-macro MSE in TRAIN-standardized units: within each channel, squared errors are summed over observed targets and divided by the observed count, then channel losses are averaged. For the combined result, error sums and counts are pooled across TEST-A/B before averaging channels. MAE, period, seed and channel results are also reported. The two periods are not counted as independent datasets.

Reported stochastic-method means average the two seed losses, not predictions. Relative change is \(100(MSE_{method}/MSE_{reference}-1)\), and is never labelled a percentage-point change. Paired moving-block intervals use 7 consecutive origins, 2,000 resamples and seed 9262026, with channels and compared methods moving together and blocks restricted to their own period. Such intervals are conditional on the fitted seeds and do not cover all training or selection uncertainty. Similar estimates or wide intervals will not establish equivalence or noninferiority.

### 4.3. Fitting and selection

LEVEL_RES, OLD_RES, LEVEL_RAW, LEVEL_LINEAR_RES and uncompressed LoRA each receive seeds 92601 and 92602. SHARED receives two closed-form fits, one per fixed penalty. The base design therefore contains 12 real-data fitting attempts; four further attempts are reserved only for defined technical repair or learning-cap continuation. F0, COMPRESS and LEVEL_ONLY require no predictive fitting and are not duplicated as independent seed observations.

G and T use learning rate 0.001; LoRA uses 0.0001 with rank 8, alpha 16 and q/v targets. Training uses AdamW, zero weight decay, gradient clipping 1, batch 4 and 512 phase-balanced TRAIN origins per epoch. The same seed and epoch identify the same order. Maximum training is 120 epochs, with early stopping after six epochs without a strict full-VAL improvement. The existing plateau scheduler uses factor 0.5 and patience 2. Step 0 is eligible; exact checkpoint ties select the earliest state. The two ridge penalties are selected by VAL MSE, with exact ties resolved to 0.001. No new learning-rate, K, mode or structure search is allowed.

LoRA retains its native quantile training objective; other neural candidates use original-channel masked macro MSE. All checkpoints are selected by the same VAL MSE definition. This loss difference is a limitation of the practical comparison. Fresh PCA and matched untrained G states are used; no learned Hog adapter is transferred. Only a run still improving near its 120-epoch cap can qualify for the approved extension rule, applied before TEST and symmetrically within the reserve. A synthetic pass is not evidence of sufficient real-data training.

All settings and selected checkpoints are sealed before the scheduled TEST comparison. Poor VAL performance does not remove a comparison arm. After TEST, structure, loss, learning rate, normalization, channels, dates and site cannot be changed in response to scores. A reproducible restoration or scoring bug can lead to a marked correction across all affected comparisons, preserving the original exposure; a changed model selection would invalidate the unexposed-confirmation claim.

### 4.4. Deployment measurements

The primary cost quantities are batch 4 throughput and peak allocated GPU memory; batch 1 latency, reserved memory, deployed parameter/buffer bytes and trainable counts remain visible. Costs use one session, the same device, FP32 with TF32 disabled, four CPU threads and 24 common VAL origins. The timed scope is standardized CPU input through GPU transfer to complete CPU-returned forecasts, with synchronization. Model loading, disk operations and scoring are excluded. Each configuration has 10 warm-ups, 20 whole-input passes and three order blocks with F0 sentinels. LoRA uses the returned merged model only after output parity is checked.

All blocks and seed results are retained. CPU deployments of the linear alternatives are separate practical paths, not GPU-kernel speed comparisons. No backbone-output cache removes work from a compressed deployment. A limited resident-input diagnostic is allowed if drift recurs, but profiler time is not substituted for deployment timing. Historical v5 time and new Robin time will not be divided to create an apparent speedup.

## 5. Development evidence

Table 1 reproduces measured v5 results. All entries are exposed development scores; lower is better, and values from different datasets are not averaged into one score. The old Bull residual and raw references use FIXED_ED K4; the historical best-selected Bull RAW was a different mode and is not substituted here.

| Group | Model | MSE | MAE |
|---|---|---:|---:|
| Hog | OLD_RES | 3.220290 | 0.996522 |
| Hog | LEVEL_RES | 2.666809 | 0.802649 |
| Hog | LEVEL_RAW | 2.949584 | 0.886455 |
| Hog | LEVEL_ONLY | 3.132837 | 0.886585 |
| Hog | F0 | 2.738044 | 0.636102 |
| Hog | LoRA | 2.683560 | 0.618902 |
| Bull | OLD_RES | 0.727659 | 0.354818 |
| Bull | LEVEL_RES | 0.708616 | 0.341326 |
| Bull | LEVEL_RAW | 0.697731 | 0.401288 |
| Bull | LoRA | 0.676191 | 0.304493 |
| Electricity | OLD_RES | 0.165850 | 0.270891 |
| Electricity | LEVEL_RES | 0.167458 | 0.271637 |
| Electricity | LEVEL_RAW | 0.233356 | 0.343038 |
| Electricity | LoRA | 0.161096 | 0.250419 |

Hog LEVEL_RES reduced MSE by 17.187% relative to OLD_RES and 9.587% relative to LEVEL_RAW. The corresponding conditional 95% relative-change intervals were [-22.920, -14.955]% and [-13.194, -4.449]%. LEVEL_ONLY was also improved upon. These results support the development objective of recovering compressed-model MSE and indicate additional value beyond the matched raw-input path in this group. Because RAW improved over its old version too, the common LEVEL treatment cannot be credited entirely to residual input.

The Hog comparison with LoRA is less decisive. Aggregate LEVEL MSE was 0.624% lower, with conditional interval [-7.865, +2.582]%, while its MAE was 0.802649 versus 0.618902. In period B, MSE was 3.119827 versus LoRA's 2.989048. Thus the aggregate MSE result neither establishes uniform accuracy dominance nor licenses equivalence. Bull LEVEL improved the old residual reference but had worse aggregate MSE than LEVEL_RAW, despite lower MAE. Electricity LEVEL worsened OLD_RES MSE by 0.969%. Its large advantage over LEVEL_RAW was partly associated with degradation of that raw control, not a recovery over the old residual method.

These historical comparisons also have implementation limits. Electricity used trainable E/D, and complete archived historical initialization tensors were unavailable for its old reference; within-v5 RES/RAW initializations were explicitly matched. The v5 execution completed 12 real fits and did not exhaust reserved candidate searches. These details are preserved rather than treating all previous negative settings as implementation failures.

In the same v5 Hog cost session, LEVEL_RES reached 353.725 batch 4 origins/s versus 167.330 for merged LoRA, with peak allocated memory 225.859 versus 328.676 MiB. The approximately 2.11-fold throughput difference was observed in that session, not inferred from C/K. OLD_RES throughput was 353.649, providing no meaningful speed-improvement claim for LEVEL itself. Batch 1 latency was 10.2416 ms/origin for LEVEL and 10.8257 for LoRA; small timing gaps are not asserted to be stable. Two of six sentinel comparisons had nonoverlapping interquartile ranges, and nearby variants exhibited order reversals. Both deployments still retained approximately 191 MB of model tensors, so fewer trainable parameters did not yield a comparable reduction in deployed weights.

Sources for Table 1 and these statements are the published [v5 final comparison](../tsfm_peft_accuracy_recovery_v5_20260927/final_comparison.json), [Hog results](../tsfm_peft_accuracy_recovery_v5_20260927/hog_level_01.json), [Bull results](../tsfm_peft_accuracy_recovery_v5_20260927/bull_level_01.json), [Electricity results](../tsfm_peft_accuracy_recovery_v5_20260927/electricity_level_01.json), and [cost measurement](../tsfm_peft_accuracy_recovery_v5_20260927/cost01.json). Their read-version hashes are recorded in CLAIMS_PRIOR_ART.

## 6. Robin fixed confirmation

The scheduled TEST evaluation completed for all 14 selected or deterministic models after the [selection seal](evaluation_seal.json). [evaluation01.json](evaluation01.json) records the [first TEST exposure](test_exposure.json), no reselection, and no corrected or reused evaluation. The main contrast remained LEVEL_RES versus OLD_RES. The planned ten neural fits and two ridge fits supplied all comparison arms. Selected epochs for seeds 92601/92602 were LEVEL_RES 10/16, OLD_RES 15/6, LEVEL_RAW 6/6, LEVEL_LINEAR_RES 19/17, and LoRA 2/1. Every selected neural checkpoint followed the initial state. SHARED selected penalty 0.001 by VAL; no TEST result selected a penalty or epoch. The completed accuracy record is available; cross-artifact verification is reported separately in [final_checks.json](final_checks.json).

Table 2 reports every scheduled method, both periods, and the combined endpoint. These are mean individual-seed losses; F0, COMPRESS, LEVEL_ONLY and selected SHARED are counted once. There were 1,920 observed horizon targets per channel in TEST-A. TEST-B had 1,918 for Robin_office_Soledad and 1,920 for each other channel, so the combined score uses actual counts rather than an unweighted average of the two period scores. Overlapping horizons are not independent observations.

| Model | Combined MSE | Combined MAE | TEST-A MSE | TEST-A MAE | TEST-B MSE | TEST-B MAE |
|---|---:|---:|---:|---:|---:|---:|
| LEVEL_RES | 0.275196 | 0.324593 | 0.273677 | 0.331865 | 0.276712 | 0.317319 |
| OLD_RES | 0.538776 | 0.483495 | 0.563077 | 0.485327 | 0.514472 | 0.481659 |
| LEVEL_RAW | 0.346630 | 0.385883 | 0.358622 | 0.397713 | 0.334635 | 0.374049 |
| LEVEL_ONLY | 0.349041 | 0.386300 | 0.355840 | 0.399788 | 0.342236 | 0.372808 |
| COMPRESS | 0.977023 | 0.661870 | 1.123537 | 0.692906 | 0.830506 | 0.630832 |
| F0 | 0.229095 | 0.262074 | 0.231539 | 0.283065 | 0.226649 | 0.241079 |
| LoRA | 0.223610 | 0.256347 | 0.230535 | 0.278978 | 0.216683 | 0.233713 |
| LEVEL_LINEAR_RES | 0.315646 | 0.354268 | 0.319826 | 0.367575 | 0.311464 | 0.340957 |
| SHARED ridge | 0.337277 | 0.376721 | 0.340548 | 0.383573 | 0.334003 | 0.369866 |

Table 3 separates the fixed-seed outcomes. LEVEL_RES improved OLD_RES and LEVEL_RAW for both seeds in both periods, rather than obtaining its mean gain from one unusually favorable training seed. The two seeds do not estimate the full distribution of possible training outcomes.

| Model | Seed 92601 MSE | Seed 92601 MAE | Seed 92602 MSE | Seed 92602 MAE |
|---|---:|---:|---:|---:|
| LEVEL_RES | 0.275009 | 0.324438 | 0.275383 | 0.324748 |
| OLD_RES | 0.572888 | 0.505539 | 0.504665 | 0.461450 |
| LEVEL_RAW | 0.345056 | 0.384607 | 0.348204 | 0.387159 |
| LEVEL_LINEAR_RES | 0.311760 | 0.351671 | 0.319531 | 0.356865 |
| LoRA | 0.221881 | 0.255153 | 0.225339 | 0.257541 |

The primary combined MSE change was -48.92% relative to OLD_RES, with a conditional 95% interval [-60.58, -31.35]%. The reduction was present in TEST-A (-51.40%) and TEST-B (-46.21%). LEVEL_RES also improved the matched LEVEL_RAW by -20.61% and LEVEL_ONLY by -21.16% in combined MSE. Thus the confirmation supports both the LEVEL modification relative to the old residual branch and an incremental learned-residual contribution beyond the common level term. This statement is about the tested design as a whole: its centering, initialization and effective temporal operator are not independently identified causal effects.

| Reference | Combined relative MSE change (%) | Conditional 95% interval (%) |
|---|---:|---:|
| OLD_RES | -48.92 | [-60.58, -31.35] |
| LEVEL_RAW | -20.61 | [-31.58, -9.44] |
| LEVEL_ONLY | -21.16 | [-32.06, -9.16] |
| COMPRESS | -71.83 | [-80.08, -57.42] |
| F0 | +20.12 | [+9.29, +36.49] |
| LoRA | +23.07 | [+10.09, +41.43] |
| LEVEL_LINEAR_RES | -12.82 | [-17.49, -8.71] |
| SHARED ridge | -18.41 | [-25.63, -10.93] |

The tested linear replacements were not sufficient to match LEVEL_RES accuracy: the combined MSE changes were -12.82% against LEVEL_LINEAR_RES and -18.41% against SHARED. Both periods favored LEVEL_RES in the point estimates and conditional intervals. This provides a practical reason to retain the frozen latent TSFM in this configuration, while leaving the amount of acceptable additional cost open. It is not an upper bound on all linear models or a pure causal effect of pretraining.

Uncompressed methods remained more accurate. LEVEL_RES had +20.12% MSE relative to F0 and +23.07% relative to LoRA; both combined intervals were entirely positive under the fixed-seed resampling. LoRA's advantage appeared in each period: LEVEL's relative MSE changes were +18.71% in TEST-A and +27.70% in TEST-B. MAE likewise favored F0 and LoRA. Therefore the result supports improvement of the compressed residual design, not replacement of uncompressed models when forecast accuracy is the sole criterion. The fixed group was not changed after this disadvantage became visible.

The period-specific LEVEL-versus-RAW interval is also informative: TEST-B's upper relative endpoint was -0.03%, close to zero, despite the larger combined improvement. We retain that uncertainty rather than claim every possible period or channel benefits. Full channel losses, observed counts, absolute error differences and paired-seed/period comparisons are retained in the source result. No post-hoc channel subset, timing-adjusted score, or cross-dataset mean substitutes for the fixed combined endpoint.

## 7. Robin accuracy–cost comparison

### 7.1. Same-session GPU deployment

The completed GPU measurements used one NVIDIA GeForce RTX 4070 and the fixed VAL inputs under the scope in Section 4.4. Table 5 reports every measured primary GPU family. Each timing center is the median of three block medians within a fit, followed by the arithmetic mean over selected fits. Memory is the largest measured peak over those blocks and fits. Timing centers are not independent-sample confidence intervals; the full pass distributions and seed/order records remain in [cost_gpu01.json](cost_gpu01.json). The accuracy column uses the fixed combined TEST endpoint, without reselecting a checkpoint by timing.

| Model | Combined MSE | Batch 1 ms/origin | Batch 4 origins/s | Batch 4 allocated MiB | Batch 4 reserved MiB |
|---|---:|---:|---:|---:|---:|
| LEVEL_RES | 0.275196 | 23.680 | 162.32 | 214.489 | 216.0 |
| OLD_RES | 0.538776 | 23.661 | 163.55 | 214.489 | 216.0 |
| LEVEL_RAW | 0.346630 | 23.663 | 164.81 | 214.489 | 216.0 |
| LEVEL_ONLY | 0.349041 | 23.571 | 164.50 | 214.421 | 216.0 |
| F0 | 0.229095 | 23.538 | 162.30 | 265.489 | 292.0 |
| Merged LoRA | 0.223610 | 23.591 | 161.39 | 265.489 | 294.0 |
| LEVEL_LINEAR_RES | 0.315646 | 1.391 | 2124.94 | 8.893 | 22.0 |

LEVEL's batch 4 throughput was only +0.58% relative to merged LoRA and +0.01% relative to F0. Its block-specific mean throughputs were 164.91, 163.28 and 161.21 origins/s. F0 and LoRA showed overlapping nearby values, and the retained pairwise comparisons included numerous order reversals. Every F0 before/after sentinel pair had overlapping interquartile ranges. Those observations do not establish equivalence, but they do prevent a stable acceleration claim. The earlier Hog throughput advantage did not transfer to this Robin deployment. LEVEL also has no demonstrated batch 1 latency advantage here.

Peak allocated batch 4 memory was 214.489 MiB for LEVEL versus 265.489 MiB for F0 and LoRA, a measured 19.21% reduction. This footprint was constant across the recorded primary blocks and seeds for each of those families. Reserved memory is reported separately and is allocator-dependent. Process resident memory also includes the Python runtime, libraries and task arrays, so its row-level measurements are not interpreted as model-only size. Lower GPU allocation does not mean the backbone weights disappeared: LEVEL still deployed 190,944,460 tensor bytes versus 190,872,100 for F0 and merged LoRA.

| Model | Fitted coefficients | Deployed parameters | Deployed tensor bytes |
|---|---:|---:|---:|
| LEVEL_RES | 17,920 | 47,736,106 | 190,944,460 |
| OLD_RES | 17,920 | 47,736,106 | 190,944,460 |
| LEVEL_RAW | 17,920 | 47,736,106 | 190,944,460 |
| LEVEL_ONLY | 0 | 47,718,186 | 190,872,780 |
| F0 | 0 | 47,718,016 | 190,872,100 |
| Merged LoRA | 294,912 | 47,718,016 | 190,872,100 |
| LEVEL_LINEAR_RES | 42,544 | 42,714 | 170,856 |
| SHARED ridge (CPU) | 24,624 | 0 | 98,496 |

Table 6 distinguishes the learned count from the deployed model. For LoRA the learned count is measured before merging, while the deployment contains the merged backbone. SHARED's coefficients are stored as inference buffers; its absence of registered trainable parameters at deployment is not absence of fitted coefficients. Tensor bytes include parameters and buffers and do not equal serialized checkpoint file size or process memory.

### 7.2. CPU alternatives and limited diagnosis

The separate CPU FP32 deployments use the same standardized input/output boundary without GPU transfer. They are practical alternatives, not GPU-kernel benchmarks. Their combined forecast scores are identical to the selected models' accuracy entries above, subject to the recorded output parity check.

| CPU model | Combined MSE | Batch 1 ms/origin | Batch 4 origins/s |
|---|---:|---:|---:|
| LEVEL_LINEAR_RES | 0.315646 | 0.5468 | 4696.46 |
| SHARED ridge | 0.337277 | 0.0442 | 54675.98 |

CPU block variation remains material. In particular, SHARED's batch 4 block centers were 111,967.15, 49,120.53 and 54,675.98 origins/s. These values remain visible rather than being replaced by the favorable block. The linear alternatives are substantially smaller and faster in these measured deployments, but have worse MSE and MAE than LEVEL. An application prioritizing resource use may still prefer them; the experiment does not supply a user-specific acceptable accuracy loss.

A single [GPU-resident diagnostic](cost_diagnostic01.json), excluded from the primary table, replayed full forwards with inputs already on the device. Compressed and uncompressed TSFM times remained close. The recorded CUDA-event intervals can include CPU-dispatch idle gaps and are not sums of pure kernel execution. Subtracting this separate diagnostic from deployment time would not isolate transfer cost. This narrows the observation without identifying a proven bottleneck or a Windows, temperature, clock or dispatch cause. No repeated profiling or favorable-block deletion was used to recover the desired speed claim.

### 7.3. Decision supported by the tradeoff

On this fixed group, LEVEL improves accuracy over the old compressed residual adapter and matched RAW at the same measured peak allocation and approximately similar deployment times. Compared with uncompressed F0/LoRA, it exchanges a clear accuracy loss for lower peak allocated memory, with no established throughput benefit. Compared with the tested linear alternatives, it buys better accuracy using much more model storage, memory and time. These are distinct comparisons, not a single efficiency score. The measured accuracy–memory choice justifies retaining this particular compressed design for further study, while reliable acceleration and a general deployment recommendation remain unsupported.

The [method diagram](figures_layout02/method.png), [overall and period accuracy](figures_layout02/robin_accuracy.png), [GPU accuracy–cost plots](figures_layout02/robin_gpu_accuracy_cost.png), and [separate CPU alternatives](figures_layout02/robin_cpu_accuracy_cost.png) display these paths and comparisons. The figure source records and hashes are in [figures_layout02/manifest.json](figures_layout02/manifest.json). Exact summary calculations are exposed in [report_values.json](report_values.json), beside the raw measurement distributions.

## 8. Limitations and scope

This study tests one frozen backbone, a fixed horizon and a modest number of channels. Robin is a single additional group in the same building-electricity domain, with only two training seeds. Source overlap with backbone pretraining remains unresolved. Bounded local record searches cannot prove universal non-use. The validation rule still selects epochs and a ridge penalty, and conditional block intervals do not capture all uncertainty induced by those choices.

The method combines known operations and carries an explicit restriction: under fixed PCA it cannot correct retained-subspace errors through the residual branch. Persistence can be wrong when future residual levels change. The shared linear G cannot represent arbitrary channel-specific or nonlinear dynamics. Although its learned count is unchanged, LEVEL changes its effective temporal operator and initialization. Matched RAW controls that common change, but the absence of a full old-RAW/new-RAW factorial experiment on Robin prevents a complete interaction estimate. That omission preserves the approved finite comparison; historical Hog results do not replace the missing Robin cell.

LoRA uses a different training loss, and the alternatives have different pretraining, capacity and optimization. Their comparison is practical rather than a pure causal estimate of pretraining knowledge. CPU/GPU choices likewise represent deployments rather than kernel-only experiments. Small memory usage for learned adapters does not remove the frozen model from deployment. The current v5 evidence already contains non-transfer and metric tradeoffs, which must remain in any final claim.

Execution-team checks can establish artifact consistency, not independent reproduction. A completed draft does not establish scientific novelty, peer-review acceptance, or submission readiness. No authorship, external submission or sharing-permission action is included in this work.

## 9. Conclusion

Level-preserving reconstruction-residual adaptation complements a compressed frozen TSFM while leaving its retained-subspace forecast unchanged under explicit conditions. The fixed Robin comparison strengthens the design's evidence beyond Hog development: it improved the previous residual adapter, matched RAW, level-only restoration and the two tested linear alternatives. This is a concrete positive result within the selected scope, while the existing Bull/Electricity limitations remain part of the evidence. F0 and LoRA were still more accurate on Robin. Their throughput was not reliably lower, whereas LEVEL retained a measured peak-allocation benefit. We therefore retain the design as a conditional accuracy–memory option and do not claim stable acceleration. The next separately approved decision is whether to extend confirmation of this fixed configuration to one other domain, retaining the same accuracy and cost countercomparisons. Such a study would test the scope of transfer; it would not erase the known accuracy loss or supply an application-specific acceptable-loss threshold. No additional method or site is selected or evaluated from these TEST results.

## References and reproducibility

Full bibliographic metadata is supplied in [references.bib](references.bib). Source access depth, publication status and algebraic conditions are recorded in [CLAIMS_PRIOR_ART.md](CLAIMS_PRIOR_ART.md). The full list of Robin channels, data hashes, approved budgets, selection policy and post-TEST repair boundaries are in [PLAN.md](PLAN.md) and [data_contract.json](data_contract.json). [selected.json](selected.json) connects run/checkpoint records; [evaluation01.json](evaluation01.json) connects predictions and metrics. [final_comparison.json](final_comparison.json), [report_values.json](report_values.json), the two primary cost records, and the figure manifest expose the reported numbers. [manuscript_claims.json](manuscript_claims.json) registers rounded manuscript values against exact source paths and hashes. [final_checks.json](final_checks.json) is the separate artifact-verification record; its actual status governs any claim that verification passed. Execution-team checks are not independent reproduction, and this manuscript has not been submitted.
