# v6 claim boundaries and source record

Status: contribution analysis, Robin fixed accuracy evaluation, and deployment measurements completed. Final artifact verification has its own status in [final_checks.json](final_checks.json). This document distinguishes algebraic statements, observed v5 development evidence, and v6 measured results. A source search is not proof of novelty. The approved scope is in [PLAN.md](PLAN.md); bibliographic entries are in [references.bib](references.bib).

## Proposed contribution and its boundaries

The contribution being evaluated is a level-preserving reconstruction-residual adapter around a channel-compressed, frozen TSFM, together with matched controls and a fixed evaluation on one group not used for method selection in the searched records. This combines existing principles for a specific information bottleneck. It is not a new normalization principle, a new PCA algorithm, the first latent adapter, or a guarantee that compression improves forecasting.

1. **Design claim — implemented, not a claim of predictive superiority.** Retain the last level of the observed reconstruction residual and train a shared small temporal map on its centered history. Compare with an old residual map, the same level term with raw-input changes, and an untrained persistence reference. The matched RAW does not restore the raw input level on top of the TSFM forecast.
2. **Structural claim — conditional algebra.** Fixed orthonormal PCA and a channel-shared, bias-free linear temporal map keep the correction in the discarded channel subspace and preserve constant offsets in that subspace. The derivation below establishes this property; the recorded synthetic check addresses implementation consistency, not empirical usefulness or novelty.
3. **Empirical claim — separated by evidence stage.** v5 supports a Hog development improvement, with known Bull/Electricity limitations. The fixed Robin accuracy comparison now supports gains over OLD_RES, LEVEL_RAW, LEVEL_ONLY and the tested linear alternatives, while preserving worse accuracy than uncompressed F0/LoRA. The measured batch 4 peak-allocation benefit supports a bounded accuracy–memory choice, while stable throughput superiority is not established. Artifact verification is reported separately. A completed paper draft, a completed new-group comparison, and readiness to submit a paper are different states.

## Closest sources and reading depth

- **AdaPTS — ICML 2025, published.** Read the official proceedings entry, arXiv v1 Sections 3.1 and 4.1, and the author's implementation overview. Frozen univariate FMs operating on latent channels, dimension reduction, and E/D adaptation by forecast loss are prior principles. The specific reconstruction-residual level path is the difference studied here. COMPRESS is a structural control, not a reproduction of AdaPTS's full probabilistic method or tuning. [Proceedings](https://proceedings.mlr.press/v267/benechehab25a.html), [method text](https://arxiv.org/html/2502.10235v1), [author repository](https://github.com/abenechehab/AdaPTS). The accessible method text is the February preprint; the proceedings establishes final publication status.
- **LTSF-Linear / NLinear — AAAI 2023, published.** Read the official paper's LTSF-Linear method passage and the complete `NLinear.py` implementation. Last-value subtraction, detached statistics, a temporal linear prediction, and last-value restoration are prior art. LEVEL applies that principle to channel-reconstruction residuals beside a frozen TSFM; it is not a new normalization mechanism or the full NLinear experimental recipe. [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/26317), [author code](https://github.com/cure-lab/LTSF-Linear/blob/main/models/NLinear.py).
- **RevIN — ICLR 2022, published.** Read the author project page, repository, and complete `RevIN.py`; OpenReview's full paper was blocked by browser verification during this review. RevIN removes/restores instance mean and scale and optionally learns affine parameters. LEVEL uses a last residual value, without this variance normalization or affine transformation. [Author project](https://seharanul17.github.io/RevIN/), [code](https://github.com/ts-kim/RevIN/blob/master/RevIN.py). Do not describe an unread full paper as reviewed.
- **ELF — ICML 2025, published.** Read the official abstract only. A lightweight additional forecaster and weighting mechanism use online feedback to adapt frozen-FM forecasts. This overlaps the broad hybrid-prediction idea. v6 fits only on TRAIN and does not update from TEST feedback. No full-method or numerical reproduction claim is made. [Proceedings](https://proceedings.mlr.press/v267/lee25ag.html).
- **EPOC — arXiv v1, submitted 25 September 2026; preprint.** Read the official abstract and Sections 3.1–3.3. It retains DCT summaries and the endpoint of a completed forecast-error block for online ridge correction. Our residual is an observed-input channel reconstruction error, with offline fitting and a different spatial constraint. Broad priority claims for endpoint-preserving residual correction are inappropriate. [Official record](https://arxiv.org/abs/2609.30929), [method text](https://arxiv.org/html/2609.30929v1). No peer-reviewed publication status was verified.
- **BDG2 — Scientific Data 2020, published.** Verified title, authors and citation through the author repository, plus its hourly/raw-versus-cleaned description. The Nature page was inaccessible in this bounded revisit. Exact Robin data/selection facts come from the local v6 contract, not an assumption about every building in the paper. [Author repository](https://github.com/buds-lab/building-data-genome-project-2), [paper](https://doi.org/10.1038/s41597-020-00712-x).
- **Model selection bias — JMLR 2010, published.** Read the official abstract and metadata. Finite validation selection motivates separating development from the fixed Robin evaluation; it does not certify independence or provide an acceptance threshold. [Official entry](https://jmlr.org/papers/v11/cawley10a.html).
- **Chronos-Bolt-small — official model artifact.** Verified the pinned revision's official file listing. No claim that Robin is absent from pretraining is made. [Pinned artifact](https://huggingface.co/amazon/chronos-bolt-small/tree/772f3d25d38aec6d914c8949dab4462e2d46f5d8).
- **LoRA — established baseline.** Verified title/authors, the 2021 preprint link and q/v low-rank-update description in the official Microsoft repository. The original full paper was not reread here; no new LoRA principle is claimed. [Author repository](https://github.com/microsoft/LoRA).

The search was bounded to the named nearest principles and a small search for channel-compressed foundation-model residual adapters. Other search hits were not promoted to evidence without reading an official source. This is a contribution boundary, not a systematic-review coverage guarantee.

## Exact structural statement

Use column-channel notation for this derivation: X is C by L, U is C by K with U^T U=I, P=UU^T, Q=I-P. Let b=U F0(U^T X), R=QX, m=R e_L, and G(R)=R A^T, where A is H by L and rank(A)<=32. Let 1_n denote an n-vector of ones. The implemented level statistic is detached; detachment does not alter these forward equalities.

The correction is c=m 1_H^T+(R-m 1_L^T)A^T. Since PR=0 and Pm=0, Pc=0. Also Pb=b, hence P(b+c)=b. For Pa=0, replacing X by X+a 1_L^T leaves U^T X unchanged and changes m by a; the centered residual is unchanged. Therefore the output increases by a 1_H^T. These are exact-real-arithmetic statements, subject to numerical tolerance in FP32 implementations.

The equivalent residual temporal map is A_eff=A+(1_H-A1_L)e_L^T. It satisfies A_eff 1_L=1_H and rank(A_eff)<=33. Thus equal trainable-parameter counts do not imply the same function class as OLD_RES. Its initialization also differs: LEVEL starts at b+m 1_H^T, while OLD_RES starts at b.

The matched RAW correction is m 1_H^T+(X-x_L 1_L^T)A_raw^T. It can modify the retained subspace. If A_raw=A_res, their difference lies in that retained subspace, but separately trained maps need not be equal; this identity is not a causal explanation of the empirical comparison.

These conclusions need fixed orthonormal E/D, a shared bias-free temporal map, and the stated input processing. Learned CURRENT E/D need not remain orthogonal. Channel-specific maps, arbitrary biases or nonlinear maps do not inherit the argument. Missing-target masks and channel-dependent observation weights prevent a general orthogonal decomposition of the primary masked loss. The geometry is in TRAIN-standardized coordinates, not a claim about unweighted physical-unit error. Input imputation can also break an offset argument defined on pre-imputation observations.

## Required counterevidence and interpretation rules

- LEVEL_ONLY may match LEVEL: then learned residual dynamics add little despite a useful persistence baseline.
- LEVEL_RAW may match or beat LEVEL: then the residual-input restriction has weak incremental support.
- LEVEL may lose to OLD_RES on Robin: the Hog recovery did not transfer under this contract.
- LINEAR_RES or SHARED may suffice: then frozen-TSFM cost has weak justification in this setting.
- F0/LoRA may have better MAE or period-specific MSE despite a favorable aggregate MSE tradeoff.
- Timing variation can make nearby methods indistinguishable in cost. The C/K ratio and trainable count do not establish a speed or memory benefit.
- One site, one backbone, two seeds, and conditional block intervals do not establish broad-domain superiority. No p-value, wide interval or similar point estimates will be called equivalence or noninferiority.

## Local evidence manifest used while drafting

All paths below are relative to this folder. SHA256 values identify the versions read; these existing v5 files are read-only inputs.

```text
../tsfm_peft_accuracy_recovery_v5_20260927/final_comparison.json
9fc9396fb9215dff99bc352d620f2dae4e517b09612d83969f9fee6aa054e90f
../tsfm_peft_accuracy_recovery_v5_20260927/hog_level_01.json
40934349ec1f310cac13a74c6380c1913c99cdb3d8e81f399102123ef542bf56
../tsfm_peft_accuracy_recovery_v5_20260927/bull_level_01.json
224df31159839e65f810cf0e7cbc06c852b8a45a58403ef7606b76469e66eed6
../tsfm_peft_accuracy_recovery_v5_20260927/electricity_level_01.json
077e7b6ee42d5d3a151994669309430c5058fe864f43e825d2c694d9f6eeb3f8
../tsfm_peft_accuracy_recovery_v5_20260927/cost01.json
2ce69d366a1249b0c1a603f763c3594e66c2313d25c8378b7f34044d372d8bb2
../tsfm_peft_accuracy_recovery_v5_20260927/model_v5.py
5775abb214269d0be997454e39e48cee34d92d7cc29d236da665038bbc4093af
PLAN.md
fe971912073704af18a1a283255f37bfa23a2b6d8f0eaa77cb0f7757863334c4
```

## Robin outcome and revised empirical claim

The complete [evaluation01.json](evaluation01.json), [selection](selected.json), [evaluation seal](evaluation_seal.json), and [first exposure](test_exposure.json) now exist. All fourteen scheduled models were evaluated without TEST reselection or a corrected evaluation. Robin's combined LEVEL_RES MSE/MAE is 0.275196/0.324593, compared with OLD_RES 0.538776/0.483495 and LEVEL_RAW 0.346630/0.385883. The relative MSE changes are -48.92% against OLD_RES, -20.61% against matched RAW, -21.16% against LEVEL_ONLY, -12.82% against LEVEL_LINEAR_RES and -18.41% against SHARED. These improvements occur in both period point estimates and both trained seeds. The TEST-B RAW comparison's conditional upper endpoint is close to zero, so no universal period/channel guarantee follows.

The remaining accuracy disadvantage is explicit: F0 MSE/MAE is 0.229095/0.262074 and LoRA is 0.223610/0.256347. LEVEL's MSE is +20.12% and +23.07% higher, respectively, with positive combined conditional intervals. Therefore the supported claim is additional value within this fixed compressed-residual design on one new group; it is not highest-accuracy superiority. Existing development counterexamples are unchanged. The completed same-session deployment records show LEVEL at 162.32 batch 4 origins/s versus F0 162.30 and merged LoRA 161.39. The +0.58% center difference against LoRA is not a stable throughput advantage: nearby ordering reverses across blocks. In contrast, LEVEL peak allocated memory was 214.489 MiB versus 265.489 MiB for both uncompressed TSFMs, a 19.21% reduction under the measured conditions. Its full deployed tensor size remains 190,944,460 bytes, so this is a measured runtime-allocation benefit, not removal of the frozen model. The CPU and GPU linear alternatives remain substantially cheaper but less accurate. The revised empirical claim is conditional value as a compressed accuracy–memory option, without a reliable speed advantage or universal recommendation. [report_values.json](report_values.json), [GPU deployment](cost_gpu01.json), [CPU deployment](cost_cpu01.json), and the separate [resident diagnostic](cost_diagnostic01.json) preserve these scopes. The manuscript's numeric claims and source hashes are registered in [manuscript_claims.json](manuscript_claims.json); the separate [final_checks.json](final_checks.json) records actual artifact-verification status.
