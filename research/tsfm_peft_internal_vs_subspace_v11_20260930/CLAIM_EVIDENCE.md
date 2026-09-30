# v11 Claim and Evidence Map

This file maps the claims that v11 is allowed to make, the evidence behind each claim, and the final v11 status under the approved scope. It has been updated after the fixed accuracy evaluation, same-session GPU/CPU cost summaries, report-value generation, adaptation-cost accounting, final checks, and [TOPIC_DECISION.md](TOPIC_DECISION.md). It is an evidence map; [TOPIC_DECISION.md](TOPIC_DECISION.md) remains the decision file.

## Current Evidence State

Available now:

- approved v11 plan and scope: [PLAN.md](PLAN.md), [APPROVAL.txt](APPROVAL.txt);
- reuse/data/protocol manifests: [reuse_manifest.json](reuse_manifest.json), [protocol.json](protocol.json);
- three exposed data contracts: [data_contract_robin.json](data_contract_robin.json), [data_contract_jena.json](data_contract_jena.json), [data_contract_hog.json](data_contract_hog.json);
- selected direct NLinear controls for Robin/Jena reuse and Hog v11: [selected_direct.json](selected_direct.json);
- implementation-path checks and normalization repair record: [synthetic_checks01.json](synthetic_checks01.json), [REPAIRS.md](REPAIRS.md), `checks/actual_bolt_checks02.json`;
- VAL-selected A/FULL/B checkpoints and run receipts: [selected.json](selected.json);
- bounded Gamma fit records and verification: [gamma_selection.json](gamma_selection.json), [gamma_verification.json](gamma_verification.json);
- fixed saved-array evaluation: [evaluation01.json](evaluation01.json), [evaluation01_models.csv](evaluation01_models.csv), [evaluation01_channels.csv](evaluation01_channels.csv);
- TEST-selection seal: [selection_seal.json](selection_seal.json);
- same-session CUDA cost rows and summary: [cost_binding.json](cost_binding.json), [cost_cuda_rows.json](cost_cuda_rows.json), [cost_cuda_grid.json](cost_cuda_grid.json), [cost_cuda_summary.json](cost_cuda_summary.json);
- consolidated GPU/CPU cost summary: [cost_summary.json](cost_summary.json);
- report value mapping and figures: [report_values.json](report_values.json), [figure_revision01.json](figure_revision01.json);
- adaptation-cost accounting: [adaptation_cost.json](adaptation_cost.json), [adaptation_cost.csv](adaptation_cost.csv);
- final numeric/provenance checks: [final_checks.json](final_checks.json);
- bounded final decision: [TOPIC_DECISION.md](TOPIC_DECISION.md);
- current execution status: [STATUS.md](STATUS.md).

Outside this evidence map:

- publication, commit/push, and live remote receipt status are tracked elsewhere;
- independent confirmation and broad generality are unavailable in this exposed-development scope.

Therefore the accuracy, memory/throughput, adaptation-cost, and decision statements below can use saved evidence. [final_checks.json](final_checks.json) passed, but it is a self-check of saved numeric/provenance/contract state, not independent reproduction or scientific proof of broad superiority.

## Fixed Accuracy Snapshot

All values in this section are [확인] from `evaluation01.json` / `evaluation01_models.csv`. They are combined TEST-A/B channel-macro MSE and MAE in the stored TRAIN-standardized coordinates, aggregated as mean seed losses rather than prediction ensembles. Robin, Jena, and Hog are all exposed development evaluations, not independent confirmation.

- Robin: A improves BASE_LEVEL slightly, 0.273071/0.323059 versus 0.275196/0.324593. B_LEARNED_U improves B_FIXED_U slightly, 0.288584/0.328083 versus 0.294975/0.334384, and B_LEARNED_GAMMA is 0.292107/0.331575. F0, native LoRA, and full MSE LoRA remain much better on MSE/MAE: 0.229095/0.262074, 0.223610/0.256347, and 0.218517/0.258718.
- Jena: A and learned U show the clearest internal gains. A is 0.253093/0.275047 versus BASE_LEVEL 0.324359/0.320450. B_LEARNED_U is 0.257653/0.276000 versus B_FIXED_U 0.328154/0.323134; B_LEARNED_GAMMA is 0.255666/0.271623. They beat direct NLinear on both MSE/MAE and F0 on MSE, but not native LoRA or full MSE LoRA on MSE; their MAE remains worse than F0/native/full LoRA.
- Hog: A and learned U are small changes. A is 2.658552/0.796149 versus BASE_LEVEL 2.666809/0.802649. B_LEARNED_U is 2.707084/0.801910 versus B_FIXED_U 2.751190/0.808313; B_LEARNED_GAMMA is 2.703102/0.766173. A slightly beats F0/native LoRA on MSE but has much worse MAE; full MSE LoRA is better on both MSE/MAE at 2.625304/0.623919.

The current accuracy evidence supports narrow internal effects, strongest on Jena. Accuracy alone does not establish an overall practical winner, a cost claim, or an independent generalization claim; the CUDA cost evidence is recorded separately below.

## Fixed CUDA Cost Snapshot

All values in this section are [확인] from `cost_cuda_summary.json`. They use the same-session CUDA deployment measurement, not cross-session timing ratios. The summary aggregation is median of three block medians per instance, then paired-seed mean.

- Jena gives A its strongest conditional practical case: A has 0.253093/0.275047 MSE/MAE and batch4 full execution of 309.48 origins/s at 218.053 MiB peak allocated. Full MSE LoRA is more accurate at 0.232892/0.238943 but costs 224.52 origins/s and 282.312 MiB in full execution. Its chunk_4K option reaches 219.336 MiB but slows to 92.01 origins/s; chunk_K reaches 198.605 MiB but slows to 28.17 origins/s.
- Hog shows a memory/throughput advantage for A but an accuracy counterexample. A is 278.59 origins/s at 225.672 MiB, while full MSE LoRA is 160.72 origins/s at 328.677 MiB; however full MSE LoRA has better MSE/MAE, 2.625304/0.623919 versus A's 2.658552/0.796149.
- Robin blocks a universal speed claim. A uses less memory than full MSE LoRA, 214.521 versus 265.966 MiB, but is slower, 207.10 versus 245.96 origins/s, and is less accurate.
- B_LEARNED_GAMMA has similar memory to A and keeps learned-U/Gamma effects visible, but it is not automatically better than A: batch4 full execution is 243.22/211.717 MiB on Robin, 302.01/218.095 MiB on Jena, and 269.29/225.723 MiB on Hog.
- DIRECT_NLINEAR remains a separate low-cost control at 7109.21, 6280.45, and 4282.35 origins/s with about 9.5-9.7 MiB allocated on Robin/Jena/Hog.

The sentinel record limits timing certainty: Robin batch1 block0 F0 sentinel changed from 11.019 ms to 13.200 ms per origin, a +19.79% post/pre shift without an assigned environmental cause. This is not a reason to discard the completed CUDA evidence, but it should prevent a universal or overprecise speed claim.

## Claims

### Claim 1: A tests internal latent TSFM adaptation without changing the learned Q branch

Allowed wording now:

> v11 defines A as an internal LoRA adaptation of the latent TSFM P path while preserving the selected LEVEL Q branch \(q^*(X)\).

Current support:

- method equation in [METHOD_UPDATE.md](METHOD_UPDATE.md);
- frozen/trainable contract in [protocol.json](protocol.json);
- parent reuse in [reuse_manifest.json](reuse_manifest.json);
- LoRA setup in [src/hier_peft/lora.py](../../src/hier_peft/lora.py);
- VAL-selected A checkpoints in [selected.json](selected.json);
- fixed saved-array accuracy in [evaluation01.json](evaluation01.json);
- same-session CUDA cost in [cost_cuda_summary.json](cost_cuda_summary.json).

Accuracy evidence:

- Robin and Hog show only small BASE_LEVEL improvements;
- Jena shows a clear BASE_LEVEL improvement;
- external F0/native LoRA/full MSE LoRA comparisons still limit the practical claim, especially on Robin and on MAE for Jena/Hog.

Final status:

- final checks passed and [TOPIC_DECISION.md](TOPIC_DECISION.md) conditionally retains A, primarily on Jena;
- A is not adopted as a universal accuracy recovery method across Robin/Jena/Hog.

Do not claim:

- A recovers the LoRA/F0 accuracy gap;
- A is a better PEFT method than full LoRA.
- A generalizes beyond the exposed development datasets.
- A has a universal speed advantage.

### Claim 2: B directly compares fixed PCA directions against learned orthogonal latent directions

Allowed wording now:

> v11 defines B as a fixed direct NLinear forecast plus a frozen-TSFM correction in a K-dimensional orthogonal subspace, comparing \(U_0\) with learned \(U\).

Current support:

- B equation in [METHOD_UPDATE.md](METHOD_UPDATE.md);
- direct NLinear S* selection in [selected_direct.json](selected_direct.json);
- U/Gamma training and fit lists in [protocol.json](protocol.json);
- selected learned-U checkpoints in [selected.json](selected.json);
- completed Gamma coefficient fits in [gamma_selection.json](gamma_selection.json);
- Gamma verification pass in [gamma_verification.json](gamma_verification.json);
- fixed saved-array accuracy in [evaluation01.json](evaluation01.json);
- same-session CUDA cost in [cost_cuda_summary.json](cost_cuda_summary.json).

Accuracy evidence:

- B_LEARNED_U improves B_FIXED_U on all three datasets by MSE, with the largest effect on Jena;
- Gamma fits completed without TEST use and did not collapse to an all-zero TSFM correction, but Gamma is a VAL mixture check, not an independent confirmation;
- the Gamma/simple-mixture step is strong enough to change B results, so it must be reported separately from the learned-U-only effect;
- B remains externally limited by F0/LoRA/full MSE LoRA accuracy on several MSE/MAE comparisons;
- B_LEARNED_GAMMA has similar CUDA memory to A, but its accuracy/cost tradeoff is dataset-dependent and not automatically better than A.

Final status:

- final checks passed and [TOPIC_DECISION.md](TOPIC_DECISION.md) retains B as a valid internal learned-subspace alternative;
- B is not automatically better than A, and Gamma/simple mixture effects must remain separated from learned-U-only effects.

Do not claim:

- learned U finds a forecastable component;
- U movement is purely subspace movement rather than rotation/sign adaptation;
- B proves TSFM value if Gamma mostly returns to zero.
- B is a practical replacement for F0/LoRA/full MSE LoRA.
- B is automatically preferable to A.

### Claim 3: Full MSE LoRA and native LoRA separate objective effects from compression effects

Allowed wording now:

> v11 separates the existing native-loss full LoRA baseline from a new full original-channel LoRA trained with the same point-MSE family used by A/B.

Current support:

- native LoRA reuse in [reuse_manifest.json](reuse_manifest.json);
- `LORA_FULL_MSE` fit list in [protocol.json](protocol.json);
- LoRA implementation guard in [src/hier_peft/lora.py](../../src/hier_peft/lora.py);
- VAL-selected full MSE LoRA checkpoints in [selected.json](selected.json);
- fixed saved-array accuracy in [evaluation01.json](evaluation01.json);
- same-session full and chunked CUDA cost in [cost_cuda_summary.json](cost_cuda_summary.json).

Accuracy evidence:

- full MSE LoRA is the strongest MSE result among the listed external baselines on Robin, Jena, and Hog in this saved evaluation;
- native LoRA remains strongest on MAE for Jena and Hog and is close to full MSE LoRA on Robin.
- full MSE LoRA chunking can lower peak allocated memory near or below A, but at much lower throughput in the recorded batch4 chunk_4K/chunk_K settings.

Final status:

- final checks and report values are complete;
- full MSE LoRA remains the strongest MSE external baseline in this saved evaluation, while native LoRA remains a strong MAE baseline.

Do not claim:

- compression is better than full LoRA;
- the native objective is the reason prior LoRA won or lost.

### Claim 4: Direct NLinear is the practical simple predictor control

Allowed wording now:

> v11 uses original-channel shared NLinear as \(S^*\), reusing Robin/Jena selected controls and fitting Hog under the same LR/seed rule.

Current support:

- Robin/Jena direct checkpoints from v7, bound in [selected_direct.json](selected_direct.json);
- Hog direct checkpoint selection in [selected_direct.json](selected_direct.json);
- NLinear-style formula in [METHOD_UPDATE.md](METHOD_UPDATE.md);
- v7 control discussion in [../tsfm_peft_practical_controls_v7_20260928/CLAIMS_PRIOR_ART.md](../tsfm_peft_practical_controls_v7_20260928/CLAIMS_PRIOR_ART.md);
- fixed v11 comparison in [evaluation01.json](evaluation01.json);
- CUDA direct-path cost in [cost_cuda_summary.json](cost_cuda_summary.json).

Accuracy evidence:

- direct NLinear is weaker than A and B_LEARNED_GAMMA on all three datasets by MSE;
- it remains a necessary practical control because it is orders of magnitude cheaper in CUDA allocated memory and throughput on these small direct paths.

Final status:

- CPU/GPU cost report values are complete;
- direct NLinear remains the simple low-cost control, not the selected TSFM PEFT method.

Do not claim:

- direct NLinear is sufficient on all three v11 endpoints;
- that beating direct NLinear on saved accuracy alone is enough for a deployment or topic decision.

### Claim 5: The normalization repair is implementation support, not a scientific contribution

Allowed wording now:

> v11 repairs a zero-variance backward issue in the installed Chronos native-scaling path while preserving the checked forward outputs and identities.

Current support:

- [REPAIRS.md](REPAIRS.md);
- `checks/actual_bolt_checks02.json`;
- no package or pretrained-weight change.

Final status:

- final checks did not change the repair boundary; it remains an implementation support item only.

Do not claim:

- new normalization method;
- performance improvement from the repair;
- evidence for A or B from common-forward equality alone.

## Prior-Art Overlap and Read Depth

**LoRA.** LoRA is prior art for frozen-base low-rank internal adaptation. v11 uses a standard LoRA baseline and a standard LoRA-in-latent-path candidate; it does not invent LoRA or a new merge rule. Local implementation: [src/hier_peft/lora.py](../../src/hier_peft/lora.py). External source noted in PLAN: Microsoft LoRA paper page and Hugging Face PEFT documentation.

**AdaPTS.** [확인] The official PMLR final PDF method sections read for this update define feature-space adapters around a univariate foundation model, including linear encoder/decoder adapters, PCA-related adapter/preprocessing discussion, and supervised joint forecast-loss training of adapter components. Therefore latent linear/PCA adapters and forecasting-objective E/D training are prior art. In the method sections read, I did not find a separate reconstruction-residual forecast branch, a LEVEL last-residual-level path, A's fixed-q* latent LoRA path, or B's direct-predictor-plus-subspace-correction/Gamma mixture. v11's remaining question is whether those narrower systems give value under the local exposed-development contract, not whether latent adapters are new.

**NLinear.** Last-value subtraction/restoration and a shared temporal affine map are prior art. v11's \(S^*\) is a direct control, and LEVEL/B equations use related linear structure. A result against this control would not prove superiority over all linear forecasters.

**ForeCA.** Forecastable components motivate asking whether PCA directions are the right directions for prediction. v11's learned \(U\) is not ForeCA and does not prove it found intrinsically forecastable components unless supported by the planned controls.

**Model-selection bias.** Robin, Jena, and Hog are exposed development evaluations. Any positive v11 result can support a candidate for further confirmation, but not independent generalization.

Existing read-depth and overlap records remain in:

- [v6 CLAIMS_PRIOR_ART.md](../tsfm_peft_level_confirmation_v6_20260928/CLAIMS_PRIOR_ART.md)
- [v7 CLAIMS_PRIOR_ART.md](../tsfm_peft_practical_controls_v7_20260928/CLAIMS_PRIOR_ART.md)

The v11 planning read record is [PLAN.md, section 4](PLAN.md). It supports the following bounded source coverage, not a comprehensive novelty clearance:

- [LoRA, ICLR 2022 official paper page](https://www.microsoft.com/en-us/research/publication/lora-low-rank-adaptation-of-large-language-models/) and [PEFT LoRA API](https://huggingface.co/docs/peft/en/package_reference/lora): frozen base, default zero update, target modules and merge; installed PEFT initialization/merge source was also read.
- [AdaPTS, ICML 2025 proceedings](https://proceedings.mlr.press/v267/benechehab25a.html), [official PMLR PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/benechehab25a/benechehab25a.pdf), and [author repository](https://github.com/abenechehab/AdaPTS): proceedings metadata and PDF link were verified from PMLR. The final PDF method sections were read at pp. 2-5, training modes at p. 9, and preprocessing/linear-adapter implementation details at pp. 16-17. They support overlap for latent feature adapters, linear/PCA adapter discussion, and supervised joint forecast-loss training. The method sections read did not show a separate reconstruction-residual branch or forecast-mixture/Gamma path. This update compares the PDF with the existing author-code review record; it does not perform a new full code audit.
- [NLinear author implementation](https://github.com/cure-lab/LTSF-Linear/blob/main/models/NLinear.py): last-value detachment, subtraction/restoration and shared affine time predictor were read and compared with the saved DIRECT definition. This does not reproduce every training experiment of the AAAI 2023 paper.
- [Forecastable Component Analysis, ICML 2013](https://proceedings.mlr.press/v28/goerg13.html): introduction and relevant method discussion on variance versus temporal forecastability; not a result about frozen TSFM improvements.
- [Cawley and Talbot, JMLR 2010](https://jmlr.org/papers/v11/cawley10a.html): selection-bias motivation, used to limit repeated-VAL and exposed-development claims.
- [PyTorch orthogonal parametrization](https://docs.pytorch.org/docs/main/generated/torch.nn.utils.parametrizations.orthogonal.html), [SciPy bounded least squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html), and [PyTorch benchmark recipe](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html): relevant API/source paths, including local PyTorch 2.10 and SciPy 1.17.1 support, were checked before implementation. Web main documentation is not substituted for the installed version.

## Final Evidence Boundary

[final_checks.json](final_checks.json) passed and links the selected runs, saved prediction checks, Gamma fits, costs, budget snapshot, and source receipts. The run used 40 neural fit attempts and 12 coefficient-fit attempts, with no technical reserve attempts consumed. Its scope is saved-array numeric/provenance/contract verification; it explicitly does not provide independent reproduction or a scientific superiority assertion.

The final decision is bounded: A is conditionally retained, primarily because Jena preserves both an internal adaptation gain and a CUDA memory/throughput tradeoff. B is retained as a valid internal alternative and diagnostic comparison. Neither result supports averaging raw MSE across Robin/Jena/Hog, choosing a TEST-based dataset policy, claiming independent confirmation, or asserting broad generality.
