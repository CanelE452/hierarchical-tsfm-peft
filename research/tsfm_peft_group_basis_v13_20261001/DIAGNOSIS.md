# Evidence that changes the next action

This is a read-only interpretation of saved v8-v12 results, not a new inference or score computation. Read on base2aefcee before any v13 fit. P/Q values below are complete-target-vector diagnostics in TRAIN-standardized coordinates, not a replacement for the primary observed-mask channel-macro MSE.

```text
Dataset / space   A or LEVEL, seed92601 /92602   Full MSE-LoRA, same seeds
Peacock P         .095821 / .095821             .095107 / .095198
Peacock Q         .107763 / .106827             .102928 / .102908
Robin P           .128109 / .110911             .101294 / .095297
Robin Q           .151860 / .152234             .116406 / .121344
Jena P            .184898 / .184251             .166509 / .168794
Jena Q            .068378 / .068659             .065052 / .065429
```

Peacock coverage3824/3840, Robin3838/3840, Jena100%. Hog coverage84.17% reverses the primary masked A/LoRA ranking, so a decomposition of that subset cannot establish the cause of Hog's primary loss. Sources: v12 confirmation_evaluation01.json and v11 evaluation01.json, `datasets.<dataset>.models.<model>.space_common_U0.combined`; exact unrounded values and coverage remain in those source records.

Peacock's four latent-LoRA configurations completed6epochs/768updates each and all selected step0. For LR1e-4, fixed TRAIN probe .331147->.291538 and .325981->.283779 improved while every learned VAL checkpoint was worse than its parent. LR1e-3 reduced TRAIN further with worse VAL. Real parameter updates and restore checks passed. This supports normal VAL rejection of adaptation, not permanently disconnected gradients. It does not identify the unique generalization mechanism or prove the recipe's optimum.

v8 learned joint linear P/Q corrections, v9 staged linear corrections with finite shrinkage, and v10 full-TRAIN ridge do not close Robin's external gap. Jena retains P improvement but still loses to stronger full-channel alternatives. v10 changes rank, fitting and regularization together, so it weakens a pure rank32/SGD explanation without proving every possible linear remedy impossible. v5 FULL/K branches were unexecuted; they are not counted as failures. No generic `all residual approaches failed` conclusion follows.

Both code and saved results leave a representation question open: applying a nonlinear univariate F to signed latent mixtures then decoding is not generally equivalent to forecasting the original channels. PCA's historical reconstruction optimum does not make F(XU)U^T a forecast optimum. This mathematical mismatch is a plausible target, not evidence that signed PCA mixing caused the observed errors. A positive disjoint group basis still averages channels and can also fail, especially for different physical variables and useful negative correlations.

The selected test changes that basis only, keeps K/F0/G capacity and the prior LEVEL learning recipe, refits the new G, and includes the matched RAW input at the same restored residual level. Old PCA LEVEL is the direct reference and old A/LoRAs/direct NLinear remain external alternatives. A gain over PCA reconstruction error or matched RAW alone is not the goal: remaining MSE/MAE loss against the stronger alternatives must still be reported. Later favorable results on these already examined datasets remain development evidence.

Deferred alternative: a zero-initial nonlinear P/Q correction could change both spaces with explicit output projection. It was not tested by the old linear experiments and is not rejected as ineffective. It is not run in this campaign because adding capacity despite TRAIN/VAL divergence is a less controlled first change. The choice of group basis is the primary agent's judgment, not a scientific result or an agent-voting certificate.

## Post-evaluation diagnostic question, fixed before its computation

The completed v13 evaluation shows lower GROUP_RES MSE than PCA LEVEL on Robin and Jena, no pooled MSE improvement on Hog or Peacock, and remaining loss versus full-MSE LoRA. Normal updates, selected epochs greater than zero and patience stopping do not support treating these outcomes as a disconnected model or a 120-epoch truncation. Jena's comparable RAW improvement also limits residual-specific interpretation.

The next question is whether the remaining error difference lies mainly in the new group's main space P, its residual space Q, or both. The old PCA-space diagnosis cannot answer that after U changes. Before any such calculation, fix this limited post-hoc analysis: reuse every existing role/seed prediction on all four datasets, project errors into the same new GROUP U for every role, retain only target vectors observed on every channel, and report P/Q/full Euclidean mean-square energies divided by C, orthogonal-sum residual, period/seed and complete-vector coverage. Keep the original masked macro MSE alongside it; no zero-filled targets, mixed forecast, fitting, channel selection or model change. This analysis is diagnostic, not another protected evaluation or a proof of a unique cause.

Run it as a reserved CPU-analysis job after GPU timing ends. It can inform one subsequent methodological question, but no new corrective head, LoRA or additional fit is included in the v13 technical reserve. The final report must retain counterexamples and subset-ranking changes, especially Hog's missing-target limitation.
