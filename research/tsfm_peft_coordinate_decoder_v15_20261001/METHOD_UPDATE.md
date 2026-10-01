# Coordinate forecasts with an untied innovation decoder

**Status: accuracy complete and internally verified; GPU costs606/612 and CPU48/48.** The fixed GPU guard left six Robin rows unmeasured. Complete three-block costs are reported for Jena, Hog and Peacock; all completed Robin measurements remain available. Both fitted decoders reduce the direct parent's pooled MSE on the four exposed development datasets, but neither resolves the gap to full-MSE LoRA. Coordinate selection is not better than matched PCA on three of four pooled MSE point estimates. These results do not establish generalization or novelty. The detailed decisions and counterexamples are in [TOPIC_DECISION.md](TOPIC_DECISION.md); the prospective contract remains in [PLAN.md](PLAN.md), [model_v15.py](model_v15.py), and [fit_v15.py](fit_v15.py).

The remaining question is whether mixing channels before a frozen univariate forecaster is an avoidable restriction. We therefore distinguish two choices: which series the TSFM receives, and how their forecast innovations are transferred to the original channels. A previously selected, frozen, original-channel NLinear predictor supplies the baseline. No LEVEL residual head or LoRA is added to the new models.

Let \(X\in\mathbb R^{B\times512\times C}\), with products below acting on the channel axis. Let \(S^*(X)\in\mathbb R^{B\times48\times C}\) be the frozen NLinear forecast, and let \(F_0\) denote the pinned Chronos-Bolt-small point path, including native context normalization, inverse transformation, median quantile, and the first 48 outputs. For a fixed encoder \(E\in\mathbb R^{C\times K}\), define

\[
Z_E(X)=F_0(XE)-S^*(X)E,\qquad
f_{E,W}(X)=S^*(X)+Z_E(X)W,
\]

where \(W\in\mathbb R^{K\times C}\) is an independent, bias-free decoder shared across origins and horizons. The projection is explicitly \(S^*(X)E\), not a separate evaluation of \(S^*\) on latent inputs.

RAW_FREE uses a coordinate selector \(J\): pivoted QR of the recorded TRAIN-PCA basis transpose chooses exactly K original channels without performance-based selection. PCA_FREE uses that same PCA basis \(U_0\). Both adapt only \(CK\) decoder coefficients, using the same observed-target objective, three ridge penalties, two frozen parent seeds, and mean-seed VAL selection. Identical coefficient counts and selection opportunities do not imply identical feature spaces, conditioning, or regularization effects. TRAIN fitting uses FP64 least squares; selection uses exported FP32 coefficients and FP32 forecast features.

Because the backbone operates on independent series rows, coordinate selection satisfies

\[
F_0(XJ)=F_0(X)J
\]

in exact arithmetic under the same forecast contract. Full-versus-selected row batching can introduce floating-point differences, which are checked on fixed VAL inputs. This preserves the existing preprocessed original-channel inputs; it does not prove that their distribution matches pretraining. Dense PCA mixing generally does not commute with the nonlinear forecast path, including its native scaling.

The v11 mixture \(S^*+[F_0(XU)-S^*U]\Gamma U^T\) ties the output directions to the input directions. RAW_TIED and PCA_TIED recover its coordinate/PCA special cases with \(\Gamma=I\). An independent W can instead transfer an innovation to unselected channels. For example, \(J=e_1\) and \(W=[0,1]\) transfer the first channel's innovation to the second; \(W=\gamma J^T\) cannot. This difference does not establish that such transfer is predictable or useful. A general W has no orthogonal P/Q decomposition or preservation guarantee.

RAW_INTERP uses \(W=(U_0^TJ)^{-1}U_0^T\), when available, as an unfitted interpolation control. [Q-DEIM](https://arxiv.org/pdf/1505.00370) provides the established coordinate-selection and reconstruction principle. Its approximation argument depends on an appropriate nonlinear-output basis and conditioning. Here, input PCA is only a selector heuristic, not a forecast-innovation basis or forecast-error guarantee. The prospective reading record covers its Sections 1.4/2.1 and Theorem 2.1. [AdaPTS](https://proceedings.mlr.press/v267/benechehab25a.html) already studies latent input/output adapters and forecast-based adaptation. Encoders, decoders, PCA, and forecast combination are therefore not claimed as new primitives. The retained [NLinear](https://github.com/cure-lab/LTSF-Linear/blob/main/models/NLinear.py) baseline uses last-value removal/restoration and a shared affine temporal layer.

W=0 exactly returns the direct parent, and deployment then loads no TSFM. Such selection is evidence for fallback, not TSFM benefit. The comparison fixes one K per dataset across Robin, Jena, Hog, and Peacock Education, all previously exposed development data. RAW_FREE versus PCA_FREE tests the complete representation-and-decoder choices; learned versus tied/interpolated controls tests decoder adaptation. External F0, LoRA, and direct alternatives remain necessary for practical interpretation. A favorable development result would still require a separately fixed, unexposed confirmation.
