# V13 decision: fixed group compression is a conditional improvement, not a general replacement

The fixed campaign is complete: sixteen fits, sixty saved prediction instances,480 GPU and48 CPU cost rows, the limited space diagnosis and artifact verification. Fixed group compression is retained as a bounded Robin improvement, not adopted as a general replacement. The overarching accuracy/generality goal is not complete. Publication is a separate state recorded by the live commit receipt.

## What changed and what the comparison answers

V13 replaces the fixed PCA basis by a deterministic TRAIN-only, signed-correlation average-linkage grouping at the same K. Its orthonormal nonnegative columns encode scaled group means. The frozen Chronos-Bolt-small, original datasets/splits/masks, shared bias-free linear G (17,920 learned parameters), initialization and learning recipe are retained. Only the new G is fitted. GROUP_RAW has the same group main path and residual-level restoration, with centered raw history instead of centered reconstruction residual entering G. Both start at the same function. See [METHOD_UPDATE.md](METHOD_UPDATE.md) for formulas and scope.

All sixteen prescribed fits completed without a real-data retry. Every fit selected a trained state, with selected epochs from3 to46; all stopped by the prescribed patience rule before120. Real updates, fixed TRAIN probes, frozen parameters and checkpoint replay are recorded in `runs/`. This does not establish an optimum, but the observed negatives are not evidence of an unexecuted or disconnected method.

## Accuracy evidence

The following are observed-mask channel-macro values, pooled across periods by per-channel error sums/counts and then averaged across the two seed losses. F0 is a single deterministic model. They are not prediction ensembles or cross-dataset scores. Exact values, every seed/period/channel, source receipts and paired conditional intervals are in [evaluation01.json](evaluation01.json), [model table](evaluation01_models.csv) and [channel table](evaluation01_channels.csv).

```text
Pooled MSE
Dataset            GROUP_RES  GROUP_RAW  PCA_LEVEL  A         F0        MSE_LoRA  Native    Direct
Robin              0.258453   0.338095   0.275196   0.273071  0.229095  0.218517  0.223610  0.327724
Jena               0.295611   0.298699   0.324359   0.253093  0.292300  0.232892  0.240590  0.273984
Hog                2.674576   2.905489   2.666809   2.658552  2.738044  2.625304  2.683560  3.059351
Peacock Education  0.210620   0.298427   0.203266   0.203266  0.201521  0.198176  0.200123  0.219884

Pooled MAE
Dataset            GROUP_RES  GROUP_RAW  PCA_LEVEL  A         F0        MSE_LoRA  Native    Direct
Robin              0.302964   0.366835   0.324593   0.323059  0.262074  0.258718  0.256347  0.353410
Jena               0.289578   0.287173   0.320450   0.275047  0.264758  0.238943  0.226584  0.286714
Hog                0.696735   0.826774   0.802649   0.796149  0.636102  0.623919  0.618902  0.775475
Peacock Education  0.314188   0.402925   0.315004   0.315004  0.300358  0.298839  0.299731  0.329450
```

Robin GROUP_RES improves on PCA LEVEL by6.084% MSE and6.664% MAE. Both seeds improve; pooled conditional MSE difference interval is[-0.031191,-0.002331]. Period A's MSE interval includes zero, while period B carries a larger improvement. GROUP_RES has23.556% lower pooled MSE than RAW, using RAW as the denominator. The5.353% MSE advantage over A is a point estimate whose interval includes zero; superiority over A is not established. GROUP_RES MSE remains18.276% higher than full MSE-LoRA, using full MSE-LoRA as the denominator.

Jena improves on PCA LEVEL by8.863% MSE and9.634% MAE, but RAW also improves on PCA. RES versus RAW is only-1.034% MSE and+0.838% MAE, with both conditional intervals including zero. This does not establish a residual-input-specific Jena gain. A and direct NLinear remain better: GROUP_RES MSE is16.799% and7.894% higher respectively; its full MSE-LoRA gap is26.931%.

Hog presents a metric tradeoff: GROUP_RES versus PCA LEVEL MSE is+0.291%, with a wide interval spanning zero, while MAE falls13.196%. MSE worsens in period A and improves in period B; the seed directions differ. The1.877% MSE loss against full MSE-LoRA is uncertain, not an equivalence result, and the11.671% MAE loss remains. Lower MAE must not be renamed a primary-MSE improvement.

Peacock Education is a counterexample to a universal basis replacement. Both seeds and both periods have higher MSE than PCA LEVEL/A; pooled MSE is3.618% higher, although its conditional interval includes zero. MAE's0.259% reduction is small and uncertain. GROUP_RES still beats matched RAW and direct NLinear, but retains6.279% MSE and5.136% MAE loss against full MSE-LoRA.

All percentages above are relative changes with the named reference as denominator, not percentage points. Intervals are conditional on these selected seeds and exposed periods; they omit repeated development, grouping/method selection and parent search uncertainty. Four datasets do not make the procedure universally superior. Peacock's original v12 prospective record is preserved, but its v13 reuse is development evidence.

## Current methodological judgment

Retain the measured Robin improvement over fixed PCA LEVEL as a useful bounded result. Preserve Hog's MAE tradeoff separately. Do not replace PCA by this grouping rule on all tasks, claim that signed PCA mixing was the unique cause, or claim that the accuracy/generality problem is solved. Group membership is highly unequal, and a positive basis does not guarantee every within-group correlation is positive. Historical search opportunities also differ.

The matched RAW results support a residual-input benefit in Robin, Hog and Peacock under this grouping, while Jena supplies a concrete limitation. Strong full-channel alternatives remain visible in the primary tables. A TEST-selected patchwork of the best method for each dataset is not a validated selection policy.

## Same-campaign costs and practical choice

The cost trigger was defined before fitting and fired on Robin/Jena; unfavorable Hog/Peacock configurations remained in the fixed grid. The GPU campaign completed480/480 rows in1,420.154 seconds; the optional CPU direct comparison completed48/48 rows. Exact measurements, raw blocks, resident allocation, reserved memory, deployment bytes and parity are linked in [cost_summary.json](cost_summary.json) and [cost_comparison.csv](cost_comparison.csv). The following point estimates are medians across three block medians per model, then means across seed models. Latency and throughput are separately aggregated and are not reciprocal means.

```text
                       GROUP_RES                         Full MSE-LoRA, unchunked
Dataset            B1 ms   B4 origins/s  B4 alloc MiB     B1 ms   B4 origins/s  B4 alloc MiB
Robin              11.941  269.72        213.107          11.973  208.31        265.224
Jena               11.767  232.03        217.975          11.843  210.15        281.944
Hog                12.144  233.79        225.672          12.478  167.89        328.927
Peacock Education  11.755  253.85        210.021          11.534  199.35        248.557

Full MSE-LoRA, same checkpoint and accuracy, B4 row-chunk alternatives
Dataset            chunk4K origins/s / alloc MiB          chunkK origins/s / alloc MiB
Robin               84.00 /214.467                        25.45 /196.760
Jena                86.80 /218.344                        25.51 /197.895
Hog                 80.51 /225.745                        22.00 /200.199
Peacock Education   88.29 /209.996                        27.61 /195.633
```

All full/chunk/merge/export checks passed at the fixed tolerances. Smaller LoRA chunks provide lower allocation and better accuracy than GROUP_RES but take more time in this measurement. They invalidate a claim that channel compression is the only memory-saving option. Grouping and PCA LEVEL have the same allocated memory for the same shape; grouping is not an additional memory reduction. The lower allocation relative to unchunked full-channel LoRA is real for these measured instances, but it is an accuracy-memory tradeoff, not a simultaneous accuracy win.

Timing is unstable: F0 pre/post sentinel changes range from-28.21% to+5.35% on Robin,-14.67% to+2.63% on Jena,-10.57% to+8.99% on Hog, and-39.60% to+30.43% on Peacock. These are drift ranges, not confidence intervals or identified thermal/OS causes. Three timed passes per block do not support declaring the small B1 differences or all B4 point differences stable. Every block is retained; no historical timing is divided by this run's timing. The large sequential-chunk slowdown is visible, while a stable speed advantage over unchunked models remains unestablished.

Direct NLinear's B4 CPU deployment is about30,994-32,265 origins/s on these small arrays (separate device comparison), with only24,624 learned parameters and98,496 parameter bytes. Its Jena accuracy is also better than GROUP_RES, so Jena does not support adopting the group model as a practical replacement. Other tasks retain direct-linear accuracy losses. GROUP_RES learns17,920 parameters, but its deployed Robin model has47,736,106 total parameters and190,944,800 parameter-plus-buffer bytes, versus190,872,440 bytes for merged full LoRA. Compression reduces active series/activation memory, not backbone storage. Reused checkpoints marked `registered_trainable_parameters=0` by the deployment loader are frozen restored objects, not proof their historical training used zero parameters. Original adaptation counts remain in the source run receipts.

## What the remaining error diagnosis changes

The diagnostic question was recorded in [DIAGNOSIS.md](DIAGNOSIS.md) before computation. [space_diagnosis01.json](space_diagnosis01.json) projects every method's error into the same new GROUP U, using only target vectors observed on every channel. Entries below are GROUP_RES minus full MSE-LoRA, mean seed squared Euclidean energy divided by C. Positive values are excess error; these are not the primary masked macro MSE or a causal attribution.

```text
Dataset            Complete-vector coverage     Delta P     Delta Q
Robin              3838/3840  (99.95%)           +0.007192   +0.032690
Jena               35040/35040(100.00%)          +0.053113   +0.009607
Hog                3232/3840  (84.17%)           +0.076898   -0.059923
Peacock Education  3824/3840  (99.58%)           +0.003545   +0.008904
```

Robin's improvement over PCA LEVEL lies mostly in P in this common coordinate system, yet most of its remaining LoRA gap lies in Q. Jena's improvement over PCA lies mostly in Q, while its remaining gap lies mainly in P. Peacock retains both errors, especially Q relative to LoRA, and worsens against PCA mainly in P. Hog trades higher P for lower Q on only84.17% coverage; opposite components must not be converted into an exaggerated contribution percentage or used to explain the full masked metric. The largest numerical P+Q sum residual is8.153e-9, consistent with the stored FP32 basis's Gram error; [space_checks01.json](space_checks01.json) verifies this without imputing targets.

Consequently, another common PCA/group switch or automatic P-LoRA retry is not supported as a universal cure. Nor is Q-only correction a supported universal prescription. The next single methodological question is whether a small nonlinear forecast correction, with backbone/basis fixed and allowed to correct both P and Q, can reduce the external gap beyond a matched linear correction with comparable capacity and input information. The old linear negatives and different remaining error locations motivate this question; they do not establish nonlinear insufficiency as the cause or predict success. Prior TRAIN/VAL divergence is an explicit counterargument. No new head, larger K, LoRA or additional fit is executed under v13's unused technical reserve.

## Completion and limits

Retain the Robin improvement and Hog MAE tradeoff; reject this group rule as a common default. The exact four-task comparison is complete, but external accuracy loss and generality are not solved. No independent confirmation or paper-readiness claim follows from these exposed-data results.

The campaign used16 predictive fits with no technical fit retries, one synthetic optimizer session/eight updates, and2,847.218 GPU job seconds (47.45 minutes) within7,200 seconds. Four TRAIN grouping computations are data preparation, not predictive fits. Frozen-cache creation, G-only training, inference and benchmarking are separately recorded; the cached training peak is not a full online-training peak. [final_checks.json](final_checks.json) passed selection/receipt,60-model saved-score and480+48-row cost checks; bootstrap intervals were not independently rerun. Figures were visually inspected. [accounting_deviation01.json](accounting_deviation01.json) discloses5.1755 seconds of unreserved agent syntax/manifest checks, retrospectively charged rather than called pre-reserved. Scientific self-checks and agent review are not independent reproduction. Original v1-v12 artifacts and unrelated dirty files are preserved.
