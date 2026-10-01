# v14: Fixed-parent nonlinear correction and matched linear/direct controls

Authority: the human delegated judgment and continued execution ("너가 직접 판단해서 쭉 진행하게 해줘") under the active goal "해결할수 있게 분석하고 해결해줘". This is a new bounded development campaign, not a fabricated explicit approval of this exact prewritten plan. No previous experiment budget is increased or retrospectively approved.

## Purpose and evidence
v13 improved PCA LEVEL MSE in Robin/Jena but not Hog/Peacock. All four remained behind full MSE LoRA. Complete-vector diagnostics found remaining excess error in both PCA-related components across tasks; these are observational clues, not proof that nonlinearity is the cause. v8-v10 already tried linear corrections, so the linear head here is a matched activation control, not a newly claimed method. TRAIN-to-VAL overfitting remains a serious counterargument.

We test one finite change: a shared nonlinear correction of a fixed existing forecast. Canonical fixed-PCA LEVEL is the parent on all four datasets, regardless of which older model had the best exposed TEST. A direct NLinear parent receives the same nonlinear correction to test whether the TSFM is unnecessary. No new dataset, K, loss, gate, backbone, LoRA, or per-dataset parent selection.

## Fixed equations
Let dX = X - repeat_L(last(X)), and h_a(dX) = W_up a(W_down dX) applied to each channel's temporal vector.
LEVEL_GELU = frozen PCA_LEVEL(X) + h_GELU(dX).
LEVEL_LINEAR = frozen PCA_LEVEL(X) + h_Identity(dX).
DIRECT_GELU = frozen selected DIRECT_NLINEAR(X) + h_GELU(dX).
GELU uses approximate='none'. Head is bias-free shared 512->32->48, 17,920 parameters. The parent already restores levels; do not add them again. The new head can correct both P and Q components.
Down starts from the recorded original untrained LEVEL G down tensor; up is exactly zero. Initial functions therefore equal their respective parents. All families and LRs share actual initial tensors for a dataset/seed. All parent parameters and buffers stay fixed. LEVEL cumulative adapter learning includes old G plus new head =35,840; DIRECT cumulative learned parameters =24,624+17,920=42,544. Backbone storage remains.

## Sources and novelty limits
AdaPTS (ICML2025 PMLR267) already supplies latent adapters; no new encoder/decoder claim. TiDE author preprint v5, section4, uses shared nonlinear MLP forecasting with a linear skip, and N-BEATS author implementation uses nonlinear backcast/forecast residual blocks. Our small GELU head is neither full reproduction nor evidence of originality. Those sources motivate a simple nonlinear versus linear control only. Standard direct NLinear remains an external practical alternative.
Read sources: https://proceedings.mlr.press/v267/benechehab25a.html ; https://arxiv.org/html/2304.08424v5 ; https://github.com/ServiceNow/N-BEATS/blob/master/models/nbeats.py ; https://github.com/cure-lab/LTSF-Linear/blob/main/models/NLinear.py .
No publication-status inference for TiDE beyond the read author preprint.

## Data and reuse
Reuse exact v13 archives and original parent receipts; no raw download or statistic refit. Robin C17/K5 hourly, Jena C21/K6 ten-minute, Hog C32/K8 hourly, Peacock Education C13/K4 hourly. L512/H48 observations. Original TRAIN/VAL/testA/testB, column order, masks, standardization, imputation and origins are unchanged. data.pca_basis is canonical PCA; data.basis is the v13 group basis and must not be substituted.
All four evaluations are exposed development. No new seal changes this status.
Canonical LEVEL parents: v6 Robin, v7 Jena, v5 Hog, v12 Peacock. DIRECT parents are the actually selected original-channel checkpoints. New compact manifests must verify existing bytes and record source hashes before reuse. v13 main-only GROUP caches cannot substitute for full PCA LEVEL parent forecasts.

## Training and selection
48 basic CUDA head fits =4datasets x3families x2LR(1e-4,1e-3) x2seeds(92601,92602).
Order is fixed in protocol: seed, dataset, LR, family. TRAIN schedule is independent of model construction and shared per dataset/seed/epoch. Phase24 hourly,144 Jena;512 origins/epoch, batch4.
AdamW wd0, clip1, max120epochs, strict VAL nonimprovement6, ReduceLROnPlateau factor.5/patience2/relative threshold1e-4. Step0 is eligible, minimum full VAL macro MSE, earliest exact tie. Choose one LR per dataset/family by mean two-seed best VAL; exact LR tie1e-4. No automatic240epochs, LR expansion or result-driven family/data deletion.
Full frozen parent TRAIN/VAL predictions may be cached; deployment computes parent online once. Cached head training must not retain the unused Chronos weights. Fixed TRAIN probe initial/best/final, gradients/actual updates, frozen-parent preservation and checkpoint restoration distinguish valid negative results from implementation failures. No requirement that all scalars change or noisy TRAIN loss decrease monotonically.

## Checks, evaluation, interpretation
Reserve every actual computational job before running. Synthetic optimizer <=6sessions, <=10updates/session. Initial-parent parity, correct temporal axes, two-step head update (zero-up can make down gradient zero first step), same initial state/sampling, immutable parents, cached-online parity, restore/restart state, loss/mask and export are the relevant checks.
Finish all48 fits and LR selections before joint new evaluation. Existing references: PCA LEVEL, A, F0, full-MSE LoRA, native LoRA, direct NLinear, GROUP_RES. New6 plus reference13 instances per dataset =76 total.
MSE primary; MAE, signed bias, period, seed, channel and pooled counts mandatory. Pool each channel's errors/counts across periods, macro-average, then mean seed loss; no ensemble or cross-dataset raw score average. Paired blocks remain inside periods, 7origins hourly/42 Jena,2000resamples,seed9262026. Intervals condition on selected models/seeds and do not include all historical selection uncertainty.
Primary questions: nonlinear versus parameter-matched linear, improvement of fixed parent, TSFM versus direct nonlinear alternative, remaining gap to F0/LoRA/A. A cheap direct model can refute TSFM utility even if the new LEVEL head improves its parent.

## Conditional costs, fixed before results
Accuracy is completed for every dataset. Cost is measured only in datasets where selected LEVEL_GELU mean-seed MSE and MAE are both no worse than both PCA_LEVEL and LEVEL_LINEAR, with at least one strict metric gain against each. This is a predeclared resource allocation rule, not a significance test or scientific success threshold.
Eligible dataset roles: three new families(6instances),PCA_LEVEL2,A2,F0(1),FULL_MSE2,NATIVE2,DIRECT2 =17instances. All B1/B4 full, plus FULL_MSE chunks B1K/B4{4K,K};3blocks and F0 before/after sentinels =132GPU rows/dataset,max528. CPU DIRECT/DIRECT_GELU=24rows/dataset,max96. Limited microbatch grid is explicit; do not claim an exhaustive frontier or extrapolate old timings.
Same local RTX4070,FP32,TF32off,CPUthreads4,first24VAL origins,10warmups,3passes,3balancedblocks. Time from standardized CPU input through complete CPU forecast, all online parent/transfer/assembly included; load/disk/scoring/hash/ledger excluded. Shape warmup and peak reset, allocated/reserved/resident/process distinctions, full/chunk/merge parity at fixed tolerances. Preserve unfavorable blocks/drift. No clock/power/environment changes. Cost slice1800GPU seconds inside total; partial grid remains partial.

## Budget and stopping
New independent bounded allocation: realfit52=48basic+4technical recovery, GPU7200s, CPUanalysis1800s, CPUchecks900s,synthetic6x<=10updates,newstorage5GiB,download0. No obligation to spend it. Illustrative GPU allocation cache900+training4000+evaluation300+cost1800+technical200=7200s; estimates, not measured completion promises. First actual throughput checks feasibility without deleting unfavorable tasks.
Failed/interrupted/restarted fitting consumes attempts. No no-fit replay charged as fitting; all GPU time charged once at parent-job level. Technical reserve only repairs same contract, not new ideas. Preserve errors and invalidate affected outputs. No new paid resources/global installs, no terminating other projects. Conflict waits30min perincident/60mintotal. Stop at real bounds, access block or sufficient interpretable comparison.

## Decisions and artifacts
If LEVEL_GELU improves parent and matched linear yet DIRECT_GELU or full LoRA remains preferable, recognize the local improvement and limit TSFM contribution. If GELU has no reliable advantage or overfits, reject this added nonlinearity; do not expand heads/activations. Mixed results imply restricted scope, not universal solution. Normal negatives and technical invalidity remain separate. The overall human accuracy/generality goal is not marked complete simply because this campaign ends.
Produce PLAN,AUTHORIZATION,protocol,provenance,STATUS; compact reuse/data/init manifests; result/curve/receipt perrun; selected/seal; joint evaluation and comparison JSON/CSV; conditional cost raw/summary; evidence/decision/verification and minimal comparison plots. Large arrays/checkpoints stay local.
Publish only related new files, result index and existing history via normal commit/push origin/main after verification and liveSHA confirmation. Preserve v1-v13, historical violations, failures and unrelated dirty files. No force/reset/newbranch/PR/hook bypass.

