# V13: test the compression representation before adding more capacity

## Authority and purpose

The human delegated judgment and continued execution ("너가 직접 판단해서 쭉 진행하게 해줘") and then set the active goal "해결할수 있게 분석하고 해결해줘". This is a new development action under that delegation, not expenditure of v12's unused experiment allowance or retroactive modification of its fixed confirmation. Base main is 2aefcee0ebb47fd83f82334bbcca2d5e8adcdb3d. Original v1-v12 artifacts, failed attempts and exposure records remain unchanged. This plan is the actual operational design, not a request for another plan. No additional approval is required for these bounded actions under the delegation; resource or method expansion beyond this contract is not implicit.

The top-level goal remains reducing the external accuracy loss while retaining a useful TSFM PEFT choice across tasks. Completing these comparisons is progress toward that goal, not automatically proof the accuracy or generality problem is solved. No paper-acceptance or universal-success threshold is invented.

## Evidence and the one change

Saved v11/v12 results show both P/Q loss on Robin, mainly P excess on Jena, and mainly Q excess on Peacock's complete-target subset. Hog's complete subset reverses the primary masked ranking; it cannot stand in for the primary metric. Peacock latent LoRA performed real updates and reduced TRAIN probe loss but every epoch worsened VAL: step0 was legitimately retained. v8-v10 normal linear-head experiments do not support repeating the same rank/LR remedy as a new solution.

First test the compression basis, holding trainable capacity and learning policy fixed. PCA optimizes historical reconstruction, not the suitability of synthetic latent signals for a nonlinear univariate foundation model. Replacing signed global mixtures with averages of positively correlated channel groups is a plausible alternative. Signed cancellation is NOT established as the cause, and group averaging can lose useful anti-correlated or cross-group information. This is a direct falsifiable development comparison, not a claimed cure.

An alternative small nonlinear P/Q side correction was considered. It is untested, not a failed method. It is deferred because current TRAIN/VAL divergence makes adding freedom less directly supported than a capacity-preserving representation test. It is not an automatic reserve in this campaign. v11's learned-U/NLinear system is different from the fixed disjoint basis tested here.

## Fixed data and basis

Use all four existing exposed datasets: Robin C17/K5; Jena C21/K6; Hog C32/K8; Peacock Education C13/K4. Keep existing source bytes, columns, timestamps, TRAIN/VAL/evaluation periods, L512/H48, standardization, original target mask, missing-input replacement and origins. Robin/Hog/Peacock are hourly; Jena is ten-minute data. No new source, download, column or period selection. Peacock is now exposed development data; its v12 prospective history is preserved.

Compute channel correlation only over the actual historical TRAIN segment, pairwise over finite original observations using their existing TRAIN-standardized coordinates. Correlation requires at least two joint observations and nonzero variance; otherwise set correlation0 and record the pair. Distances are sqrt(max(0,2-2*clip(rho,-1,1))), never absolute correlation. Start singleton clusters in original channel order. Repeatedly merge the pair with smallest mean original between-channel distance (average linkage); exact ties use lexicographic sorted leaf-index tuples. Stop at exactly K clusters, ordered by smallest member index. No size tuning, sign flips or outcome-dependent clustering.

For group j, U[c,j]=1/sqrt(group_size_j) when c belongs to j, else0. Thus U has orthonormal disjoint positive columns, and XU is a scaled group mean (not literally an unscaled mean). U and U transpose are fixed tied encoder/decoder. There is no learned clustering, routing, individual head, K search or additional trainable parameter. Store correlation/group membership/basis hash and source TRAIN boundaries. Standardized grouping does not imply common physical units for Jena's raw variables.

## Models and initial functions

Let P=UU^T, r=X-XP, ell=last(r), DeltaX=X-repeat_L(last(X)), and b=F0(XU)U^T. F0 is the same pinned Chronos-Bolt-small native-normalized median cropped to48.

    GROUP_RES = b + repeat_H(ell) + G(r-repeat_L(ell))
    GROUP_RAW = b + repeat_H(ell) + G(DeltaX)

Both start at b+repeat_H(ell), use the same original-channel information, same fixed U and same G initialization, and train only G. G is shared, bias-free, activation-free512->32->48,17,920 parameters with zero output layer. RAW uses the same residual-level restoration term, never adds the whole raw level twice. RES's linear shared G remains in Q up to floating-point error; RAW need not.

Use each dataset/seed's original untrained LEVEL G initialization when bytes and provenance can be verified. Never transfer learned PCA G or latent LoRA onto the changed basis. If original G initialization cannot be recovered exactly, record the limitation before fitting and reconstruct from verified original constructor/RNG order; do not assert equality without comparing bytes. Both new families must share literal tensors and origin schedules.

Reuse fixed selected PCA LEVEL, A, F0, MSE-LoRA, native-LoRA and direct NLinear as references, with actual checkpoint/prediction/source receipts. They are separately named. Main question is GROUP_RES versus PCA LEVEL under the same F0/K/G recipe; group RAW distinguishes group representation effects from residual-input effects. A and both LoRAs are stronger external/nearby alternatives. Historical search opportunities differ and remain reported. This is not a pure signedness ablation or an AdaPTS/CCM full reproduction.

## Fits, validity and selection

Basic matrix: four datasets x {GROUP_RES,GROUP_RAW} x seeds92601/92602 =16 new fits. LR1e-3 is fixed, matching the original LEVEL starting recipe; no new LR sweep. AdamW wd0, clip1, batch4,512 phase-balanced TRAIN origins/epoch, max120epochs, strict nonimprovement6, ReduceLROnPlateau factor.5/patience2/relative threshold1e-4. Keep the existing scheduler initialization policy; do not fix it simultaneously with the basis change. Step0 is a candidate, smallest full VAL masked channel-macro MSE wins, exact ties earliest. Sampling uses independent SeedSequence([seed,epoch]) and period24 except Jena144. Reuse original train schedules where equivalent and record hashes.

G may train from cached deterministic frozen b for TRAIN/VAL. This removes redundant frozen F0 execution only; inputs/targets/masks, gradients and loss remain the online model's. Cache provenance and online/cached parity are required before trusting results. Inference cost must always include online F0, never these caches. No TEST target or score is used for grouping, training, LR or checkpoint selection.

Check initial RES=RAW function, shared G tensors, U Gram tolerance1e-5, frozen F0/U, G gradients and actual changes, expected first-step zero down gradient, axes/mask, finite loss, checkpoint replay and restart state. Ordinary parity atol1e-5/rtol1e-4; same-route replay atol1e-6/rtol0. No tolerance inflation or arbitrary loss decrease assertions. Real-data optimizer smoke counts as a fit; use the scheduled fit's first updates instead of a separate smoke. CPU synthetic optimizer sessions are separately bounded.

All16 prescribed choices are finalized before new evaluation. A dataset is not cancelled because an earlier one is unfavorable. Only reproducible technical invalidity or hard resource/access limits can interrupt this matrix. No automatic240epoch extension. Reserve attempts repair technical failures/restores, not new candidates.

## Evaluation and costs

Use primary TRAIN-standardized observed channel-macro MSE; report MAE, signed bias, each seed/period/channel, and residual gap versus F0, both LoRAs, direct NLinear, PCA LEVEL and A. Pool period SSE/count per channel, then channels, then seed losses. No prediction ensemble or cross-dataset raw-score average. Existing reference arrays require exact hash/order/mask/provenance checks and score replay. Conditional paired time blocks:7 origins on hourly tasks,42 on Jena,2,000 draws, same methods/channels together, no crossing period boundaries. These intervals exclude repeated development and selection uncertainty. All four evaluations are exposed development evidence.

Only if a normally learned GROUP_RES offers an accuracy improvement over PCA LEVEL in any fixed task will a same-session cost comparison be run, with all four tasks retained. Cost is not a reward that removes negative accuracy records. Compare selected GROUP_RES/GROUP_RAW/PCA LEVEL/A/F0/full-MSE-LoRA/native-LoRA/direct, B1/B4; include full-MSE-LoRA row chunksK and4K as applicable. Same VAL24 origins, FP32/TF32off/CPU4threads,10 warmups/3 full passes/3 blocks and F0 sentinel. This reduced repeat count is fixed before the first fit: v12's135 GPU rows took2,135.39s, so blindly copying10passes for480 rows would exceed the bounded cost allocation. Each model is loaded/verified once per block for its shape options. This is a planning estimate, not a measured v13 runtime; fewer repeats limit precision. There are432 core GPU rows plus48 sentinel rows, optional48 direct CPU rows. CPU standardized input through all online prediction/chunk assembly to complete H48 original-channel CPU output is timed; load/disk/hash/scoring/ledger is outside. Report allocated/reserved/resident/parameter bytes and latency/throughput separately. Exact duplicate functions may share a record only with verified provenance. Keep every block and drift. No profiler sweep or claimed whole frontier. Bound cost before execution to <=1,800GPU seconds within the total cap; an incomplete grid is explicitly incomplete.

## Operational bounds and autonomy

For this new delegated development, self-limit to20 real fits (16basic+4technical), one local GPU and one neural fit at a time, GPU total7,200s, CPU data/analysis/cached fitting7,200s, CPU checks900s, synthetic optimizer<=6sessions each<=10updates, new storage5GiB, downloads0. These stricter operational limits are not expected completion times or academic acceptance thresholds. No use of old unspent ledger allowances; no automatic increase. Parent GPU jobs and nested fit times are not double-counted. Reserve jobs and attempts before work, count failures/restarts, record actual updates and time. No other-project termination, power/driver/global environment changes, paid services or servers. Conflict wait<=30min/incident and60min total; verify live processes.

Same-scope API/path/serialization/accounting errors may be repaired with original failure and diff preserved. Data/evaluation errors invalidate and recalculate affected comparisons. No post-evaluation method/data/metric change. Different methods or new data are not silently appended to this campaign.

If the changed representation reduces the gap with clear matched-control interpretation, retain that bounded evidence and identify remaining external losses/costs. If RAW suffices, reduce residual-specific claims. If normal group models fail, reject this basis rule rather than blaming every negative on optimization. Completing or rejecting this hypothesis does not by itself complete the overarching accuracy/generality goal; record the unresolved requirements and next evidence-based action. Stop this campaign at its fixed comparison completion, technical impossibility or hard cap without filling unused resources.

## Records and publication

Keep PLAN, exact authority text, STATUS, reuse/data/initial manifests, protocol and source hashes, reservations, per-fit result/curve/checkpoint receipts, joint selection, predictions local, comparison JSON/CSV, conditional cost records, diagnostics and TOPIC_DECISION/final checks. Raw/checkpoints/large arrays stay in ignored cache. State which gaps are actually resolved. Original v1-v12 remain untouched except related README/history index additions. Check only relevant files, ordinary commit and origin/main push, verify live SHA; no force/history rewrite/hooks bypass/new branch/PR. Self-verification or agent agreement is not independent scientific evidence.

## Targeted prior art read

AdaPTS, ICML2025: official PMLR entry and author repository adapter table; latent adapters and forecast learning are existing principles. https://proceedings.mlr.press/v267/benechehab25a.html and https://github.com/abenechehab/AdaPTS .

Chen et al., From Similarity to Superiority: Channel Clustering for Time Series Forecasting, NeurIPS2024: official PDF pp1-5/sections4.1-4.3 read. CCM learns assignments/prototypes and cluster-specific feed-forward weights. Our fixed TRAIN average-linkage, orthonormal group means and shared residual G differ, but clustering channels itself is not novel and the paper does not establish our compressed-TSFM benefit. https://papers.nips.cc/paper_files/paper/2024/file/eb9b18ccb76a1156af5779ffdca1d91f-Paper-Conference.pdf .

SciPy linkage/cut_tree official docs read; local scipy1.17.1 differs from current web1.18. A small explicit average-linkage/tie implementation is specified for deterministic provenance, not a new clustering algorithm. Its merge distances should be checked against SciPy on non-tied synthetic data. No comprehensive novelty search or priority claim.
