# Coordinate selection and independent decoder: decision

Accuracy and the available cost artifacts have passed internal verification. GPU costs stopped at the fixed time guard after606/612 rows; CPU costs completed48/48. The full four-dataset cost campaign is incomplete, and publication status is recorded separately in STATUS.md. The completed accuracy comparison does not support promoting coordinate selection plus an independent decoder as the common solution to the remaining accuracy gap. Both fitted decoder families improve the direct parent in pooled MSE point estimates, but every pooled MSE remains above the existing internal-LoRA A and full-MSE LoRA. This is a useful distinction between improving a cheap parent and justifying the resulting complete TSFM system.

## What was actually tested

The frozen original-channel DIRECT_NLINEAR parent supplied S*. RAW_FREE used K actual input channels selected by TRAIN-PCA pivoted QR, while PCA_FREE used the original K PCA directions. Each independently fitted a bias-free K-by-C output matrix to the innovation F0(XE)-S*(X)E. Nothing in F0, S*, the encoder, normalization or data split was trained again. Three ridge penalties and the direct-parent fallback received the same two-seed VAL selection rule. All48 coefficient fits completed normally without retries; every family selected a fitted decoder rather than the fallback. The original masks, full TRAIN origins, fitted objectives, FP32 exports and all84 forecast instances passed arithmetic verification.

This is not the previous LEVEL residual input path, a new LoRA variant, or an online gate. RAW_TIED/PCA_TIED and untuned RAW_INTERP separate learned decoding from fixed reconstruction. The [method note](METHOD_UPDATE.md) distinguishes standard adapter/forecast-combination and Q-DEIM principles from this concrete experiment.

## External position and internal contrasts

The following are mean seed losses over pooled original-period channel error sums/counts. They are not ensembles or a cross-dataset average.

| Dataset | RAW_FREE MSE / MAE | PCA_FREE MSE / MAE | A MSE / MAE | FULL_MSE MSE / MAE |
|---|---:|---:|---:|---:|
| Robin | 0.301136 / 0.330507 | 0.289623 / 0.326954 | 0.273071 / 0.323059 | 0.218517 / 0.258718 |
| Jena | 0.253811 / 0.265235 | 0.257625 / 0.273149 | 0.253093 / 0.275047 | 0.232892 / 0.238943 |
| Hog | 2.840448 / 0.806972 | 2.762650 / 0.773428 | 2.658552 / 0.796149 | 2.625304 / 0.623919 |
| Peacock Education | 0.212821 / 0.321706 | 0.207880 / 0.315635 | 0.203266 / 0.315004 | 0.198176 / 0.298839 |

PCA_FREE reduces DIRECT pooled MSE on all four datasets with negative conditional time-block difference intervals. This supports the complete fitted correction formula over the selected direct parent within these exposed data and fixed selection rules. It does not isolate F0's pretrained knowledge from learned decoding and the −S*EW recombination, or establish superiority to the stronger full TSFM/LoRA alternatives. Hog PCA_FREE's small MAE point improvement has an interval crossing zero.

RAW_FREE's pooled MSE point reductions versus DIRECT are8.113%,7.363%,7.155%,3.212% in dataset order. Hog's MSE interval crosses zero and its MAE increases4.062% with a positive interval. Thus the result is not consistent improvement across error metrics. RAW_FREE is worse than matched PCA_FREE on three dataset MSE point estimates. Its Jena MSE advantage over PCA_FREE is1.480%, but the conditional interval crosses zero and TEST-B reverses direction. Preserving native channel rows is therefore not established as a generally better representation.

Learning the decoder is not universally better than a fixed decoder either. In Jena, RAW_FREE and PCA_FREE reduce their corresponding tied-control MSE by9.489% and21.493%. Hog RAW_FREE instead has MSE+1.732% and MAE+14.486% versus RAW_TIED. Peacock PCA_FREE has MSE+1.392% and MAE+0.809% versus PCA_TIED, with a positive MAE difference interval. The decoder is learned against TRAIN and selected on VAL; it is not a TEST error-minimizing oracle.

Jena is the limited retained observation. RAW_FREE versus A has MSE+0.284% with an interval crossing zero, and MAE−3.567% with a negative interval. No equivalence or noninferiority margin was specified, so these values do not establish equal accuracy. The method uses126 newly fitted decoder coefficients plus the previously fitted24,624-parameter direct parent, rather than another neural-adaptation stage. Its complete deployment still requires the frozen TSFM. Against FULL_MSE, meaningful error gaps remain. Historical v14 DIRECT_GELU is also worse in Jena point estimates; this is historical context, not a newly paired comparison or same-session cost claim.

All controls, F0, native-loss LoRA, each seed and both periods remain in [evaluation01.json](evaluation01.json). Conditional block intervals use the selected two seeds and exposed parents; they do not cover all prior method/parent/penalty selection uncertainty.

## What the negative result rules out

The experiment does not support the sufficient-condition claim that removing PCA mixing alone resolves the common accuracy gap. It also does not prove that PCA mixing is harmless: coordinate selection changes the available information, frozen nonlinear features and their conditioning. The fitted W problem is convex least squares/ridge and its objective was verified; there is no evidence to blame this W result on too few neural epochs or a missed learning rate. Other model classes, features or untested penalties are not ruled out, but they are outside this finished finite comparison.

## Cost and next decision

The guard stopped GPU work after2,372.414s, inside the2,400s cost allocation. The six uncompleted rows are all Robin block2: DIRECT seed92602 B1; RAW_FREE seed92601 B1, B4-full and B4-K; and both closing F0 sentinels. Its147 completed rows remain in [the complete raw record](cost_cuda_rows.json), but no Robin GPU aggregate is presented. The performance-independent completeness rule permits unchanged three-block aggregation for Jena, Hog and Peacock Education,459 rows in total. [cost_available_summary.json](cost_available_summary.json) links the original606-row record, guard, exact missing grid and this subset. It is not a completed frontier or a new favorable-dataset selection.

Jena illustrates the limited tradeoff. Values below use same-session three-block medians and then mean seed medians. CPU and GPU execution are separate alternatives. Peak allocated memory is not model size or reserved memory.

| Jena execution | B4 origins/s | B4 allocated MiB | B4 reserved MiB | B1 full latency ms/origin |
|---|---:|---:|---:|---:|
| RAW_FREE full | 201.78 | 217.78 | 236 | 12.01 |
| PCA_FREE full | 236.13 | 217.78 | 236 | 11.39 |
| A full | 207.72 | 217.97 | 238 | 11.99 |
| F0 full | 221.27 | 281.94 | 306 | 12.45 |
| FULL_MSE full | 201.17 | 281.94 | 306 | 12.16 |
| FULL_MSE chunk4K | 80.99 | 218.34 | 252 | not this execution |
| FULL_MSE chunkK | 25.40 | 197.89 | 222 | not this execution |
| DIRECT GPU | 5,646.38 | 8.56 | 22 | 0.307 |
| DIRECT CPU | 32,249.44 | N/A | N/A | 0.070 |

RAW_FREE therefore uses less allocated memory than full uncompressed execution, but smaller LoRA chunks can use still less memory while retaining their better accuracy, at substantially slower measured throughput. RAW_FREE does not have a meaningful allocation advantage over A. Its201.78 versus201.17 origins/s does not establish a stable speed advantage: all22 available F0 sentinel pairs are retained, and pre/post changes range from−27.81% to+28.32%. This is observed drift, not a causal attribution to Windows, temperature or clocks. Bars in the figures span seeds and blocks, not statistical confidence intervals.

The Jena new model fits126 decoder coefficients while reusing the already fitted24,624-parameter NLinear. It still deploys47,742,640 parameters plus buffers,190,973,116 tensor bytes in total; W is stored as a buffer and is included in bytes. No neural optimizer was run in this campaign. The48 coefficient fits share frozen feature generation, but the measured incremental work includes492.77s TRAIN/VAL GPU feature work and52.43s evaluation features, plus110.25s CPU coefficient fitting. Historical parent adaptation is additional; these increments are not a controlled training-speed comparison. Total GPU occupation was2,917.609s. [decision_cost_values.json](decision_cost_values.json) links these table inputs; [cost_comparison.csv](cost_comparison.csv) retains all available B1/B4, seed and execution alternatives through the underlying summaries.

The decision is to decline the proposed common coordinate-decoder replacement, retain only the bounded Jena MSE–MAE/adaptation-cost observation, and close this finite candidate comparison with its disclosed cost limit. No new selector, basis, loss, gate, dataset, penalty sweep or repeat fitting is authorized by unused technical reserve. Accuracy loss and generality have not been solved, and internal verification is not independent reproduction. The next scientific decision is whether the limited Jena tradeoff warrants a separately fixed confirmation; these four reused development datasets cannot provide that confirmation. It would test a restricted tradeoff claim, not certify recovery of the common full-LoRA accuracy gap.
