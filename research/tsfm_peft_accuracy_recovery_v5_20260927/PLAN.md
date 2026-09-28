# Residual PEFT v5: approved execution contract

Approval: user requested execution ("실행해줄수있어?") after the detailed v5 plan. This file records the operational contract, not a verbatim transcript. The original user request is preserved in approved_request.txt. Baseline main: 4e0107c0ce258056cad1ea46bbc14c0d289e5374. Approval does not retrospectively authorize v2's 22/20 violation.

## Objective and scope
Reduce Hog compressed RES accuracy loss against uncompressed F0/LoRA. Determine one useful configuration or reject bounded modifications. Reconfirming residual usefulness alone is insufficient. Hog TEST-A/B, Bull E1/E2 and Electricity DEV are exposed DEVELOPMENT evaluations. No new site, new topic, protected confirmation, new modules beyond the finite candidates, future information, or success guarantee.

Preserve v1-v4 contracts/results/failures/seals and unrelated dirty files. Use existing raw data, frozen Chronos-Bolt-small revision 772f3d25d38aec6d914c8949dab4462e2d46f5d8 and saved predictions/checkpoints. Publish only related source and small result records; caches/weights/prediction arrays remain local.

## Bounded diagnosis and pre-fit choice
CPU diagnosis <=1800s; diagnostic GPU <=1200s within total GPU budget. Reproduce existing Hog scores after origin/channel/mask alignment. Compare channel/origin/period/seed MSE, MAE, signed error (prediction-target), absolute loss and positive excess concentration. Fixed TRAIN PCA decomposition uses only completely observed target vectors; report coverage and do not substitute missing targets with zero. Compare Q8, Q16-Q8 and I-Q16; relate correction y-RES to past residual last value and past24 mean/std. Inspect existing TRAIN probe/VAL curves and freeze/update records. LEVEL_ONLY is saved COMPRESS plus repeated last reconstruction residual, fit0.

Select exactly one first candidate and one reserve before first fit:
- LEVEL_ONLY improves combined COMPRESS MSE and residual level aligns with remaining correction: LEVEL first, FULL reserve.
- Otherwise positive excess in added PCA directions exceeds remaining outside directions and complete-case direction agrees with full masked comparison: K first, FULL reserve.
- Otherwise nonlevel/outside error or uncertain evidence: FULL first, LEVEL reserve.
LEVEL wins a tie for lower added complexity; this is an operational choice, not a success threshold. K cannot repair old Q8 predictions under fixed nested PCA and channel-independent F0. Negative outside relative excess does not mean zero outside absolute error; diagnosis alone does not prohibit the smallest direct fit.

## Formulas
b_K=D_K(F0(E_K(X))); r_K=X-D_K(E_K(X)); p(u)=stopgrad(u_last), P_n(u)=repeat(p(u),n).
LEVEL RES=b_K+P_H(r_K)+G(r_K-P_L(r_K)). Matched RAW=b_K+P_H(r_K)+G(X-P_L(X)). Both start at b0+P_H(r0), unlike original b0. LEVEL_ONLY is this no-G, unadapted PCA function. Only level statistics detach; CURRENT main and residual-input gradient paths remain as defined.
FULL RES=b_K+W r_K; RAW=b_K+W X, shared bias-free Linear(512,48), W0=0. Both start b0. Capacity, parameterization and optimization change jointly; no pure-rank causal claim.
K uses K'=min(2K,C) with old PCA columns/signs preserved as prefix and added TRAIN complement directions: RES=b_K'+G(r_K'), RAW=b_K'+G(X). Start b_K' (K_ONLY). Existing G initialization explicitly shared; no LEVEL or FULL combination.
G is shared bias-free activation-free 512->32->48 with output weight zero. Hog FIXED_ED K8/C32; Bull FIXED_ED K4/C16; Electricity CURRENT K8/C32. K doubles to16/8/16.
Trainable parameters RES=RAW: LEVEL17920 fixed/18432 Electricity; FULL24576/25088; K17920/18944. Frozen F0 contributes to deployed size and inference cost.

## Learning and paired comparison
First Hog quartet: modified RES seed92601/92602 and matched RAW seed92601/92602. Fresh TRAIN PCA and untrained E/D/G, no learned-site warm start. Paired initial tensors and origin order must be hash-verified. Missing historical Electricity initial.pt prevents claiming exact historical initialization equality.
FP32, dropout disabled; AdamW LR.001, wd0, clip1, effective/microbatch4. Phase-balanced512 TRAIN origins per epoch, SeedSequence([seed,epoch]). Max120epochs, patience6. ReduceLROnPlateau factor.5 patience2 threshold1e-4 relative min_lr0 eps1e-8. Select minimum full VAL MSE including step0; exact ties earliest. No development-score checkpoint selection.
Check actual parameter updates, intended freeze, finite gradients/loss, axes/masks, initialization correspondence, selected checkpoint restoration and restart state. Initial low-rank G down gradient zero is expected when up=0. Synthetic check success is not learning sufficiency.
One bounded supplement pool: capped run with strict improvement in last6 and no patience stop can continue once to240 preserving optimizer/scheduler/RNG; each restart consumes a new attempt. Apply same eligibility across arms/seeds. No arbitrary LR/architecture search for bad VAL.

## Branches
After valid Hog comparison: both improve -> separate common modification from residual extra; RES only improves -> residual-specific hypothesis; RAW equal/better -> residual-specific claim weakens; neither/RES worsens -> inspect learning validity then reserve if its predeclared evidence remains. No arbitrary improvement cutoff.
If modified RES mean-seed whole Hog MSE is below old RES with valid learning, extend the SAME modification and paired RAW to Bull and Electricity (4fits each). Small improvement justifies a development probe, not proof. Do not pool dataset numbers into one score. If LEVEL_ONLY suffices, a simpler alternative can end development without a new PEFT claim.
Reserve only once: LEVEL->FULL for remaining nonlevel temporal error; K->FULL at originalK if allocation insufficient; FULL->LEVEL if level/bias clue remains. Record observation/question/change/expected distinction before switching. A late reserve after old-data extension gets Hog4 only, without old-data generalization claim.
Bull K8 v2 fixed RES/RAW two-seed results already valid and VAL worse thanK4: reuse, do not retrain or call it fresh positive. Historical K4/K8 G initial differs; withinK8 pairs match. New Hog/Electricity K uses explicit sharedG.

## Evaluation and costs
Primary masked channel-macro MSE in TRAIN-standardized units; MAE plus seed/period/channel, oldRES, matchedRAW, F0/LoRA remaining loss. Pool per-channel error sums/counts across HogA/B before macro then average seed LOSSES (not ensemble). Pair block bootstrap7origins/2000draws/seed9262026 within each period and all channels/methods together. Exposed developmental uncertainty only; wide intervals are not equivalence.
Only a retained modification receives new cost benchmark: Hog oldRES,newRES,newRAW,F0,mergedLoRA, both seeds where relevant. Same-session VAL24, FP32,CPUthreads4,TF32off,batch1/4,10warmups,20whole-input passes,3orderblocks. Scope standardized CPUinput->H2D->full CPUoutput with sync; exclude load/disk/scoring/ledger. Report latency, throughput, allocated/reserved/process memory, trainable/total params and bytes separately. Merge parity, preserve drift/badblocks; never divide new timing by old timing.

## Hard budgets and accounting
First Hog4 + one chosen modification Bull/Electricity8 + reserve Hog4 + paired learning supplement4 + failures/restarts/mandatory recovery4 =24 real-data attempts maximum. Typical12; early reserve16; late reserve16 (no second complete three-dataset expansion). K path reuses Bull so only8 new fits if retained. No obligation to fill pools.
GPU1, concurrent training1, totalGPU14400seconds includes training/inference/restore/warmup/smoke/failures/profile; nested fit duration not double-counted. Synthetic CPU optimizer max6sessions, <=10totalupdates each, CPUchecks<=900s. Storage<=5GiB. No new raw/backbone download. Real-data optimizer smoke/interruption/restart and closed-form fitting count; TRAIN/PCA statistics and fit0 references are separately timed. Reserve before start, keep failures and actual updates. Conflict wait1800s/incident and3600s cumulative; do not kill other projects. No automatic limit increase.

## Autonomy, completion and publication
Approval authorizes bounded implementation, comparison, evidence-based changes, repair/retry, evaluation, reporting, related normal commit and origin/main push. No global installs/driver/power/server changes, force/history rewrite/hooks bypass/new branch/repo/PR. API/path/serialization bugs retain original failure and repair diff; data/evaluation errors invalidate affected results and recompute related controls.
Stop on interpretable keep/reject decision, simpler alternative sufficient, reserve unsupported, budget or access limit. Improvement may remain unmet. No independent-reproduction or universal-success claim.
Artifacts: PLAN,STATUS,provenance/reuse manifest/common ledger, diagnosis, per-run result/curve/receipt, comparisonJSON/CSV, cost if warranted, TOPIC_DECISION, final verification, related index/history and commit/push confirmation. Resume by checking STATUS/ledger/PIDs; never duplicate ongoing jobs.
