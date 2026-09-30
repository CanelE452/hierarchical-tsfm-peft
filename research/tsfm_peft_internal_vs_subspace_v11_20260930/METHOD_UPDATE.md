# v11 Method Update

This document records the method definitions for v11 and the completed accuracy, cost, adaptation-cost, report-value, and verification evidence. The final bounded interpretation is in [TOPIC_DECISION.md](TOPIC_DECISION.md): A is retained conditionally, primarily on Jena's accuracy-memory-throughput tradeoff, while B remains a valid internal learned-subspace comparison rather than an automatic replacement for A. This document does not claim broad generalization, independent confirmation, publication, or a universal cost advantage. The authoritative execution contract is [PLAN.md](PLAN.md), with the exact user approval in [APPROVAL.txt](APPROVAL.txt).

## Scope

v11 compares two bounded follow-up systems on the already exposed Robin, Jena, and Hog development datasets:

1. **A_P_LORA_QFIXED**: adapt the frozen latent TSFM internally with ordinary LoRA while preserving the previously learned LEVEL residual branch.
2. **B_LEARNED_U / B_FIXED_U**: keep a direct original-channel NLinear predictor fixed and learn only the orthogonal latent directions given to the frozen TSFM, with a separate bounded Gamma mixture check.

The comparison is deliberately limited. It does not combine A and B, change K, add QLoRA/DoRA, add gates, change losses, introduce a new backbone, or retune the existing LEVEL parent.

## Common Definitions

Let \(X \in \mathbb{R}^{B \times 512 \times C}\), \(Y \in \mathbb{R}^{B \times 48 \times C}\), \(U_0 \in \mathbb{R}^{C \times K}\), \(P_0=U_0U_0^\top\), and \(Q_0=I-P_0\). \(F_0\) is the pinned Chronos-Bolt-small point forecast path with the existing native scaling, 0.5 quantile selection, and H48 crop. \(G^*\) is the selected trained LEVEL temporal branch from the parent run.

\[
r_0 = X - XP_0
\]

\[
q^*(X) = \operatorname{repeat}_{48}(\operatorname{last}(r_0)) +
G^*\left(r_0 - \operatorname{repeat}_{512}(\operatorname{last}(r_0))\right)
\]

\[
\operatorname{BASE\_LEVEL}(X) = F_0(XU_0)U_0^\top + q^*(X)
\]

The reused parent checkpoints are bound in [reuse_manifest.json](reuse_manifest.json). The data contracts keep the existing channel order, train-only standardization, PCA basis, masks, and origin lists in [data_contract_robin.json](data_contract_robin.json), [data_contract_jena.json](data_contract_jena.json), and [data_contract_hog.json](data_contract_hog.json).

## Candidate A: Internal Latent LoRA With Fixed Q Branch

\[
\operatorname{A\_P\_LORA\_QFIXED}(X)=F_\phi(XU_0)U_0^\top+q^*(X)
\]

Frozen:

- base TSFM weights;
- \(U_0\), decoder \(U_0^\top\), train-only statistics, and parent buffers;
- the learned \(G^*\) branch and its LEVEL input contract.

Trainable:

- only the new ordinary LoRA adapter inside the latent TSFM path.

The LoRA setting follows the repository baseline: rank 8, alpha 16, dropout 0, target modules `q` and `v`, no bias, expected registered trainable parameters 294,912. The corresponding implementation guard is in [src/hier_peft/lora.py](../../src/hier_peft/lora.py). The initial zero-update adapter identity with BASE_LEVEL was treated as an implementation-path check, not a performance claim; the verification receipts are linked from [final_checks.json](final_checks.json) and the implementation check records.

The matching external LoRA control is:

\[
\operatorname{LORA\_FULL\_MSE}(X)=F_\phi(X)
\]

It uses the same LoRA structure, initialization policy, LR candidates, seeds, and original-channel masked macro MSE, but it runs on the full original-channel input and does not use LEVEL or \(G^*\). Existing `LORA_NATIVE` remains a separate practical baseline trained with the native quantile objective.

## Candidate B: Direct Predictor Plus Learned TSFM Subspace

The direct predictor \(S^*\) is a shared original-channel NLinear predictor:

\[
S^*(X)=\operatorname{repeat}_{48}(\operatorname{last}(X))+
T^*(X-\operatorname{repeat}_{512}(\operatorname{last}(X)))
\]

where \(T^*\) is a shared `Linear(512,48,bias=True)`. Robin and Jena reuse v7 direct NLinear checkpoints; Hog selected direct NLinear checkpoints are recorded in [selected_direct.json](selected_direct.json).

B is:

\[
B(U,\Gamma;X)=S^*(X)+[F_0(XU)-S^*(X)U]\Gamma U^\top
\]

where \(\Gamma=\operatorname{diag}(\gamma)\) and \(0 \le \gamma_j \le 1\).

Important constraints:

- the term is \(S^*(X)U\), not \(S^*(XU)\);
- encoder and decoder share the same \(U\);
- \(S^*\) and \(F_0\) weights stay frozen;
- no LoRA, LEVEL \(G^*\), extra channel mixer, or new residual branch is added to B;
- \(\Gamma=0\) is exactly the \(S^*\) fallback and is not evidence of TSFM value;
- \(\Gamma=I\) keeps all K latent TSFM correction directions.

`B_FIXED_U` uses \(U=U_0\) and \(\Gamma=I\), with no neural fit. `B_LEARNED_U` starts from \(U_0\), keeps \(\Gamma=I\), and trains only \(U\). The learned \(U\) effect includes subspace movement and possible in-subspace rotation/sign effects because \(F_0\) is nonlinear in the latent coordinates.

Gamma is a later bounded coefficient fit on VAL only. It is a prediction mixture check, not a sparsity or cost optimizer. Exact zero coefficients may allow a deployment shortcut after parity is checked; small positive coefficients are not pruned after TEST.

## Direct Controls

The direct controls define the closest practical objections:

- `F0`: unadapted full original-channel TSFM.
- `LORA_NATIVE`: existing native-loss full LoRA baseline.
- `LORA_FULL_MSE`: v11 full original-channel LoRA trained with the same point-loss policy as A.
- `DIRECT_NLINEAR`: original-channel NLinear \(S^*\).
- `BASE_LEVEL`: previously selected LEVEL parent.
- `B_FIXED_U`: fixed PCA version of the B system.
- `B_FIXED_GAMMA` and `B_LEARNED_GAMMA`: bounded VAL mixture checks.

TEST comparisons will distinguish internal evidence (`A - BASE_LEVEL`, `B_LEARNED_U - B_FIXED_U`) from practical evidence against external alternatives. Dataset-wise winners will not be combined into a post-hoc selector.

## Fixed Accuracy Evidence Available

The scheduled neural fits, VAL selections, Gamma coefficient fits, and fixed saved-array TEST evaluation have completed for the accuracy side of v11. The source records are [selected.json](selected.json), [gamma_selection.json](gamma_selection.json), [gamma_verification.json](gamma_verification.json), [selection_seal.json](selection_seal.json), [evaluation01.json](evaluation01.json), [evaluation01_models.csv](evaluation01_models.csv), and [evaluation01_channels.csv](evaluation01_channels.csv). These are still exposed development evaluations, not independent confirmation.

Combined TEST values below use the stored channel-macro MSE/MAE and mean seed losses, never prediction ensembling.

- Robin: A gives a small BASE_LEVEL improvement, MSE 0.273071 versus 0.275196 and MAE 0.323059 versus 0.324593. B_LEARNED_U improves B_FIXED_U modestly, MSE 0.288584 versus 0.294975 and MAE 0.328083 versus 0.334384. B_LEARNED_GAMMA is 0.292107 MSE and 0.331575 MAE. External accuracy remains the limiting point: F0 is 0.229095/0.262074, native LoRA is 0.223610/0.256347, and full MSE LoRA is 0.218517/0.258718.
- Jena: A and learned-U show the clearest internal improvements. A improves BASE_LEVEL from 0.324359/0.320450 to 0.253093/0.275047. B_LEARNED_U improves B_FIXED_U from 0.328154/0.323134 to 0.257653/0.276000, and B_LEARNED_GAMMA is 0.255666/0.271623. These results also beat direct NLinear on MSE/MAE and F0 on MSE, but not native LoRA or full MSE LoRA on MSE; their MAE also remains worse than F0/native/full LoRA.
- Hog: A and learned-U changes are small. A improves BASE_LEVEL from 2.666809/0.802649 to 2.658552/0.796149. B_LEARNED_U improves B_FIXED_U from 2.751190/0.808313 to 2.707084/0.801910, and B_LEARNED_GAMMA is 2.703102/0.766173. A slightly beats F0/native LoRA on MSE, but has much worse MAE than F0/native/full LoRA; full MSE LoRA has lower MSE and MAE at 2.625304/0.623919.

The accuracy evidence supports narrow internal effects, especially on Jena, but it does not establish a universal external winner. Same-session CUDA and CPU cost evidence, report consolidation, final numeric verification, and the final method decision are now available in [cost_summary.json](cost_summary.json), [report_values.json](report_values.json), [final_checks.json](final_checks.json), and [TOPIC_DECISION.md](TOPIC_DECISION.md).

## Same-Session CUDA Cost Evidence Available

[cost_cuda_rows.json](cost_cuda_rows.json) completed 621/621 CUDA rows, and [cost_cuda_summary.json](cost_cuda_summary.json) summarizes the measurements as the median of three block medians per instance, then the arithmetic mean of paired seed medians. The scope is the same-session CUDA deployment path, not a cross-session speed ratio.

Batch4 full-execution comparisons show a conditional cost advantage for A on Jena and Hog, but not a universal throughput advantage:

- Robin: A is 207.10 origins/s at 214.521 MiB peak allocated, while full MSE LoRA is faster at 245.96 origins/s but uses 265.966 MiB. Robin therefore supports a memory reduction, not a speed claim.
- Jena: A is 309.48 origins/s at 218.053 MiB, while full MSE LoRA is 224.52 origins/s at 282.312 MiB. FULL chunk_4K reduces memory to 219.336 MiB but drops to 92.01 origins/s; chunk_K lowers memory to 198.605 MiB but drops to 28.17 origins/s. This is the strongest accuracy-memory-throughput case for A, although full MSE LoRA remains more accurate on MSE/MAE.
- Hog: A is 278.59 origins/s at 225.672 MiB, while full MSE LoRA is 160.72 origins/s at 328.677 MiB. The accuracy side is a counterexample to a clean practical win: A has worse MSE/MAE than full MSE LoRA and much worse MAE than F0/native LoRA.

B keeps the learned-U effect visible against fixed-U, but it is not automatically better than A. The bounded Gamma/simple-mixture step is strong enough to change the B comparison, so it should be interpreted as a separate mixture effect rather than proof that learned \(U\) alone is the useful component. In batch4 full execution, B_LEARNED_GAMMA is 243.22 origins/s at 211.717 MiB on Robin, 302.01 origins/s at 218.095 MiB on Jena, and 269.29 origins/s at 225.723 MiB on Hog. These points are similar in memory to A and sometimes faster or slower, while accuracy differs by dataset and metric. DIRECT_NLINEAR remains a separate low-cost control at 7109.21, 6280.45, and 4282.35 origins/s with about 9.5-9.7 MiB allocated on Robin/Jena/Hog, so any TSFM-based candidate must be justified by accuracy or representational value rather than raw speed alone.

The sentinel check still limits strong timing claims. Robin batch1 block0 F0 sentinel changed from 11.019 ms to 13.200 ms per origin, a +19.79% post/pre shift with no assigned environmental cause. Other sentinel shifts were smaller, but this unresolved drift should stay visible in the final report.

## Report, Adaptation-Cost, and Verification Evidence

[cost_summary.json](cost_summary.json) is complete for GPU and CPU measurements. [report_values.json](report_values.json) is complete and maps the reported accuracy, cost, source keys, figures, and adaptation-cost outputs. [adaptation_cost.json](adaptation_cost.json) is complete and separates reused historical parent/S*/native LoRA costs from this-round incremental fits; the rows should not be read as controlled training-speed ratios. [figure_revision01.json](figure_revision01.json) records a layout-only revision of the batch4 accuracy-cost figure with no numeric, method, or selection change; the original report numbers remain unchanged.

[final_checks.json](final_checks.json) passed. It verifies the saved-array numeric/provenance/contract state, including 40 neural fits, 12 Gamma fits, saved prediction checks, and the linked cost/report sources. It also records `independent_reproduction: false` and `scientific_superiority_asserted: false`. That means the numeric audit is complete for this run, but it is not independent reproduction and not scientific proof of broad superiority.

## Prior-Art Boundary

This work does not claim new LoRA mathematics, a new NLinear normalization principle, or the first latent adapter for time-series foundation models.

- **LoRA** supplies the frozen-base, low-rank internal-adapter principle. v11 uses the existing q/v LoRA setup and treats any benefit as a system result, not a new LoRA method. Sources: Microsoft LoRA paper page and Hugging Face PEFT documentation.
- **AdaPTS** already covers latent input/output adapters around univariate foundation models and forecast-loss adaptation. v11 differs by testing a fixed residual branch in A and a direct-predictor-plus-subspace correction in B; neither is a reproduction of AdaPTS's full probabilistic model.
- **NLinear / LTSF-Linear** supplies last-value subtraction and restoration through a temporal affine predictor. v11 uses that idea directly for \(S^*\) and historically for LEVEL-style residual processing; it does not claim NLinear itself as new.
- **Forecastable Component Analysis** motivates the distinction between high-variance PCA directions and forecast-useful directions. It does not show that the learned v11 \(U\) will improve a TSFM system.
- **Model-selection-bias literature** motivates treating Robin/Jena/Hog as exposed development evaluations. v11 is not an independent confirmation stage.

The closest-overlap notes and read-depth records from v6/v7 remain relevant: [v6 CLAIMS_PRIOR_ART.md](../tsfm_peft_level_confirmation_v6_20260928/CLAIMS_PRIOR_ART.md) and [v7 CLAIMS_PRIOR_ART.md](../tsfm_peft_practical_controls_v7_20260928/CLAIMS_PRIOR_ART.md).

## Normalization Repair Boundary

[REPAIRS.md](REPAIRS.md) records a v11-only technical repair for the installed Chronos native scaling path. The original forward values, location, scale, parent outputs, F0 outputs, merge/export identities, row chunking, and Gamma identities were verified under the fixed tolerances in `checks/actual_bolt_checks02.json`. The repair gives finite gradients for exactly zero-variance contexts while preserving forward behavior.

This is a technical fix that allows the intended differentiable path to be trained. It is not a method contribution, not a normalization innovation, and not evidence that A or B improves accuracy. Likewise, common-forward equivalence checks establish that wrappers preserve the intended function; they are not scientific evidence of forecasting value.

## Final v11 Interpretation Boundary

The v11 methodological evidence is complete under the approved scope. The decision in [TOPIC_DECISION.md](TOPIC_DECISION.md) is to conditionally retain A, primarily on Jena, and to keep B as a valid internal learned-subspace alternative rather than a replacement for A. A did not solve the accuracy gap across all three exposed datasets, and B did not become a generally better system after Gamma.

Independent confirmation and broad generality are not v11 outputs. Robin, Jena, and Hog are exposed development evaluations, so those claims remain unavailable in this scope regardless of the completed cost and verification results. Publication, push, and repository-posting status are tracked outside this method note.
