> 공개 사본: 개인 경로·채팅 식별자와 링크를 정리했습니다. 원본 및 사본 해시는 감사 공개 폴더의 `publication_manifest.json`에 분리 기록합니다. 아래 상태·검산은 원래 감사/실행 시점의 기록이며, 이번 게시 검증이 아닙니다.

# Accuracy interpretation of the Chronos-2 controls

[확인] This note reads the completed, saved evaluation results. It does not add forecasting, fitting, scoring, or benchmarking. The same-session cost experiment was still running when this note was written; accuracy alone does not establish a deployment recommendation or a Pareto advantage.

The new zero-shot controls leave LEVEL with substantially higher combined MSE and MAE on Robin and Jena. Peacock is different: its LEVEL-versus-Chronos-2 MSE differences are small and their conditional intervals include zero. LEVEL has slightly lower point-estimate MSE than Chronos-2-Small MV on Peacock, while both native MV controls have lower MAE. Consequently, neither “both Chronos-2 variants beat LEVEL on every accuracy measure” nor “LEVEL remains competitive everywhere” describes these results.

The original-channel MSE-LoRA reference has the lowest combined MSE among the nine reported methods on all three datasets. Chronos-2 MV has the lowest combined MAE on all three. These are different objectives and must remain separate in the presentation.

## Evidence and aggregation

Sources are [evaluation_summary.json](evaluation_summary.json), old_score_replay.json (로컬 보존, 공개 사본 미포함: `old_score_replay.json`), [accuracy_comparison.csv](accuracy_comparison.csv), accuracy_channels.csv (로컬 보존, 공개 사본 미포함: `accuracy_channels.csv`), accuracy_origins.csv (로컬 보존, 공개 사본 미포함: `accuracy_origins.csv`), and [paired_comparisons.csv](paired_comparisons.csv). Exact scores are at `/units/{dataset}/methods/{method}/periods/{period}/{metric}`. Individual seed results remain under `/units/{dataset}/models/*/periods`. All decimals in this note are rounded for display; JSON/CSV retain the underlying values.

[확인] The saved replay record reports 144 passing historical MSE/MAE checks, with maximum absolute difference `5.551115123125783e-17`. This is an internal numerical replay, not independent replication. It does not substitute for the campaign's separate final verification.

The evaluator pools each channel's error sums and observed-target counts across TEST-A/B, divides within each channel, and then averages channels. Learned method results average the two selected seed losses; they do not average predictions into an ensemble. Each zero-shot model is counted once. The same observed mask is used for every method, including observed zeros. Targets remain in the frozen TRAIN-standardized original-channel coordinates.

```text
Dataset              C/K   Origins A/B/combined   Observed targets A/B/combined
Robin                17/5          40/40/80           32640/32638/65278
Peacock Education    13/4          40/40/80           24958/24946/49904
Jena                 21/6        365/365/730        367920/367920/735840
```

The combined per-channel count vectors are identical across all methods. Robin has 3,840 observed targets in every channel except one with 3,838. Peacock has 3,840 except three channels with 3,838, 3,836, and 3,830. Jena has 35,040 per channel. The counts are denominators for the channel-macro metric, not independent sample sizes: neighboring 48-step targets overlap.

## All methods and both periods

`B` is the presentation's fixed PCA plus frozen main-path baseline `b`. `MSE_LORA` is the selected original-channel point-MSE LoRA, not latent internal-LoRA candidate A and not the older native-quantile LoRA. `C2` denotes Chronos-2 and `SMALL` Chronos-2-Small; MV and UNI use their fixed grouped and isolated-channel contracts.

```text
Robin             MSE A      MSE B      MSE combined    MAE A      MAE B      MAE combined
B                 1.123537   0.830506   0.977023        0.692906   0.630832   0.661870
LEVEL             0.273677   0.276712   0.275196        0.331865   0.317319   0.324593
F0                0.231539   0.226649   0.229095        0.283065   0.241079   0.262074
MSE_LORA          0.229946   0.207085   0.218517        0.283501   0.233932   0.258718
DIRECT_NLINEAR    0.327484   0.327962   0.327724        0.367210   0.339608   0.353410
C2_MV             0.221426   0.238871   0.230150        0.273224   0.234821   0.254024
C2_SMALL_MV       0.241441   0.207855   0.224650        0.288877   0.230654   0.259768
C2_UNI            0.214492   0.250679   0.232587        0.269092   0.240656   0.254876
C2_SMALL_UNI      0.242116   0.228748   0.235434        0.286653   0.238824   0.262740

Peacock Education
B                 0.322713   0.274944   0.298832        0.435271   0.407054   0.421165
LEVEL             0.234300   0.172223   0.203266        0.339193   0.290810   0.315004
F0                0.235011   0.168020   0.201521        0.324832   0.275875   0.300358
MSE_LORA          0.230911   0.165429   0.198176        0.323183   0.274487   0.298839
DIRECT_NLINEAR    0.253442   0.186316   0.219884        0.355312   0.303581   0.329450
C2_MV             0.232664   0.166467   0.199572        0.319034   0.271248   0.295145
C2_SMALL_MV       0.235122   0.173416   0.204274        0.324298   0.279680   0.301992
C2_UNI            0.232718   0.167227   0.199978        0.320866   0.271554   0.296214
C2_SMALL_UNI      0.237789   0.173353   0.205577        0.327581   0.279649   0.303619

Jena
B                 0.486863   0.275510   0.381187        0.440826   0.321358   0.381092
LEVEL             0.422475   0.226243   0.324359        0.381041   0.259859   0.320450
F0                0.370756   0.213844   0.292300        0.313130   0.216386   0.264758
MSE_LORA          0.278738   0.187045   0.232892        0.267552   0.210335   0.238943
DIRECT_NLINEAR    0.332070   0.215899   0.273984        0.322967   0.250461   0.286714
C2_MV             0.303096   0.204877   0.253987        0.252617   0.196296   0.224457
C2_SMALL_MV       0.323248   0.203062   0.263155        0.263088   0.197346   0.230217
C2_UNI            0.324581   0.212178   0.268380        0.269325   0.204721   0.237023
C2_SMALL_UNI      0.330446   0.212864   0.271655        0.272300   0.205465   0.238883
```

All selected learned-seed combined values are retained here to show the aggregation:

```text
Dataset   Method           MSE seed92601/92602       MAE seed92601/92602
Robin     LEVEL            0.275009 / 0.275383        0.324438 / 0.324748
          MSE_LORA         0.219055 / 0.217979        0.261099 / 0.256338
          DIRECT_NLINEAR   0.334129 / 0.321319        0.362302 / 0.344519
Peacock   LEVEL            0.203727 / 0.202805        0.315513 / 0.314495
          MSE_LORA         0.198143 / 0.198210        0.299013 / 0.298666
          DIRECT_NLINEAR   0.219777 / 0.219991        0.328880 / 0.330020
Jena      LEVEL            0.324219 / 0.324499        0.320484 / 0.320416
          MSE_LORA         0.231561 / 0.234222        0.236931 / 0.240955
          DIRECT_NLINEAR   0.273152 / 0.274817        0.288104 / 0.285324
```

## LEVEL versus native multivariate zero-shot

Every relative difference below is `100 * (left / right - 1)`, with the **right-hand method's metric as denominator**. Thus positive `LEVEL / C2` means higher error for LEVEL. These are relative percentages, not percentage-point differences. The exact paired row keys are `left`, `right`, `period`, `metric`, `left_value`, `right_value`, `absolute_difference`, `relative_percent`, `denominator`, `absolute_ci95`, and `relative_ci95_percent`.

```text
Combined contrast           Metric    Relative %       Conditional 95% interval %
Robin LEVEL / C2_MV         MSE          +19.572          [+3.184, +40.500]
                            MAE          +27.780         [+16.133, +38.649]
Robin LEVEL / SMALL_MV      MSE          +22.500          [+9.742, +43.545]
                            MAE          +24.955         [+14.141, +38.220]
Peacock LEVEL / C2_MV       MSE           +1.851          [-1.190,  +6.406]
                            MAE           +6.729          [+4.657,  +8.625]
Peacock LEVEL / SMALL_MV    MSE           -0.493          [-3.313,  +5.535]
                            MAE           +4.309          [+2.337,  +6.804]
Jena LEVEL / C2_MV          MSE          +27.707         [+19.887, +33.785]
                            MAE          +42.767         [+36.876, +46.400]
Jena LEVEL / SMALL_MV       MSE          +23.258         [+15.630, +30.564]
                            MAE          +39.195         [+33.186, +42.966]
```

For the exact combined MSE denominators, C2_MV/SMALL_MV are respectively Robin `0.23015048867707527 / 0.2246504656523076`, Peacock `0.19957182242794386 / 0.2042740641119311`, and Jena `0.25398682694924746 / 0.2631551031209199`.

Period qualifications matter. Robin's LEVEL-versus-C2_MV MSE intervals include zero separately in A and B; the combined interval is positive. Robin's LEVEL-versus-Small MSE interval includes zero in A but is positive in B. Robin MAE is higher for LEVEL in both periods for both controls, with positive conditional intervals. Peacock's MSE intervals include zero in both periods for both controls; the point estimate favors LEVEL over Small in both periods. Peacock MAE favors both MV controls in both periods, with positive LEVEL-relative intervals. Jena's LEVEL MSE and MAE are higher in both periods versus both MV controls, with positive intervals throughout. Complete period-specific differences and intervals remain in the paired CSV/JSON, including unfavorable periods.

## Does channel sharing help?

The implementation's same-origin grouping contract and an accuracy benefit are distinct questions. MV is not required to improve every task for the grouping implementation to be valid.

```text
Combined MV / corresponding UNI   Metric   Relative %     Conditional 95% interval %
Robin C2                         MSE        -1.048         [-5.437, +3.075]
                                 MAE        -0.334         [-1.929, +1.556]
Robin SMALL                      MSE        -4.580         [-7.415, -0.725]
                                 MAE        -1.131         [-2.460, +0.196]
Peacock C2                       MSE        -0.203         [-3.718, +3.681]
                                 MAE        -0.361         [-1.749, +1.205]
Peacock SMALL                    MSE        -0.634         [-2.283, +0.605]
                                 MAE        -0.536         [-1.099, -0.056]
Jena C2                          MSE        -5.363         [-7.565, -3.619]
                                 MAE        -5.302         [-6.551, -4.178]
Jena SMALL                       MSE        -3.129         [-6.017, -0.206]
                                 MAE        -3.628         [-4.874, -2.181]
```

Jena provides the clearest conditional evidence of a grouped-input improvement. For C2, both Jena periods improve in both metrics with negative intervals. For Small, Jena MSE-A includes zero, MSE-B is negative, and both MAE periods are negative. Robin C2 has uncertain combined differences. Robin Small improves combined MSE, but its MAE effects reverse across periods: A `+0.776%` (interval `[+0.216,+2.145]`) and B `-3.421%` (`[-5.420,-1.537]`). Peacock C2's combined changes are approximately two to four tenths of a percent and uncertain; Peacock Small's combined MSE is uncertain and its MAE reduction is small. These results do not establish a universal multivariate advantage.

## External accuracy alternatives remain strong

[확인] Both MV controls improve over DIRECT_NLINEAR in combined MSE and MAE on all three datasets, with negative conditional intervals. Against Bolt F0 and original-channel MSE-LoRA, the result depends on the dataset and metric:

- Robin: C2_MV MSE is `+0.461%` versus F0 and `+5.324%` versus MSE-LoRA; both intervals include zero. Small-MV MSE is `-1.940%` versus F0 and `+2.807%` versus MSE-LoRA, again uncertain. Their MAE comparisons to these references also include zero.
- Peacock: C2_MV MSE is `-0.967%` versus F0 and `+0.704%` versus MSE-LoRA, with intervals including zero. Its MAE versus F0 is `-1.735%` with interval `[-2.710,-0.103]`; versus MSE-LoRA it is `-1.236%` with an interval including zero. **Small-MV is worse than MSE-LoRA in both combined metrics:** MSE `+3.077%` (`[+1.105,+5.183]`) and MAE `+1.055%` (`[+0.190,+2.177]`). Small-MV versus F0 is uncertain in both metrics. This unfavorable Small result must remain visible.
- Jena: both MV models improve combined MSE and MAE over F0, but MSE-LoRA retains lower MSE. C2_MV versus MSE-LoRA is MSE `+9.058%` (`[+6.286,+13.125]`) and MAE `-6.063%` (`[-7.200,-4.725]`); Small-MV is MSE `+12.995%` (`[+9.479,+17.570]`) and MAE `-3.652%` (`[-5.062,-2.168]`). This is an error-objective tradeoff, not uniform dominance.

## Signed mean error and interpretation limits

The signed error is prediction minus observed target, pooled within channel and then averaged channels and selected seed losses. The following combined values describe average standardized bias; positive and negative channel/time errors can cancel. A small signed mean error does not establish small MSE, calibration, or the cause of a performance difference.

```text
Method             Robin       Peacock      Jena
B                 +0.019811   -0.015910    -0.034748
LEVEL             +0.007139   -0.013859    -0.012159
F0                +0.002460   -0.022447    -0.027040
MSE_LORA          +0.032866   -0.017744    -0.000747
DIRECT_NLINEAR    +0.019396   -0.042300    +0.011269
C2_MV             +0.001776   -0.025339    -0.005112
C2_SMALL_MV       -0.006782   -0.028664    -0.010980
C2_UNI            +0.001933   -0.028225    -0.011886
C2_SMALL_UNI      -0.008623   -0.032985    -0.015721
```

The paired moving-block intervals use the fixed stored protocol: 2,000 draws, seed `9262026`, block lengths Robin/Peacock/Jena `7/7/42` origins, dataset salts `0/0/1`, and period-specific RNG sequences. A and B are resampled separately and their channel error sums/counts are pooled; blocks never cross their boundary. All channels/methods are paired within a draw. Intervals are conditional on the selected stored checkpoints and seeds, and exclude full model-selection uncertainty. They are not equivalence or noninferiority tests. No multiplicity-adjusted significance claim is made from the collection of reported contrasts.

All three datasets and periods are already exposed development evaluations. New zero-shot predictions do not turn these data into independent confirmation. The comparison is between complete pretrained systems and information-sharing modes; it is not a pure causal estimate of cross-channel information or pretraining knowledge. No causal explanation for LEVEL's remaining error has been established here.

The supported presentation revision at this stage is: **LEVEL's accuracy limitations are clearer after adding native Chronos-2 alternatives; any practical argument must be stated as a measured accuracy–memory–throughput tradeoff against these alternatives and normal execution variants.** A claim that LEVEL is the most accurate choice is unsupported. Whether LEVEL retains a cost-based choice region awaits the same-session completed cost results and final verification; this note makes no deployment recommendation.
