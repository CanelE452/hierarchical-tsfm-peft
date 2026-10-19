# ETTm1 registered K trade-offs — wave 3

Outcome: POSITIVE_EXPLORATORY. ETTm1 was executed at all registered K=1, K=2 and K=4 with seeds92601/92602.

Positive requires both seeds below Raw MSE, their mean at least1% below Raw, and both seeds below Raw whole B8 median latency and allocated peak under DEFAULT128. MAE and the practical Direct MLP comparison remain separate disclosed axes.

Full original C7, N69680, consumed prefix55744, native15min; H48=12h and L512=128h. ETTm1 and ETTh1 share the same transformer station; they are related frequency variants, not independent domains. ETTh1 H48=48h; historical Jena H48=8h. Compare methods within this dataset; no frequency resampling.

## DEFAULT128

```text
Method       seeds  MSE mean [min,max]           MAE mean [min,max]           B8 ms mean [min,max]         allocated MiB [min,max]      reserved MiB [min,max]
Raw          1      0.278007 [0.278007,0.278007]  0.292479 [0.292479,0.292479]  16.379350 [16.379350,16.379350]  250.995605 [250.995605,250.995605]  266.000000 [266.000000,266.000000]
B2 K=1       2      0.339939 [0.333789,0.346089]  0.404090 [0.400861,0.407320]  12.001075 [11.724700,12.277450]  199.934082 [199.934082,199.934082]  270.000000 [270.000000,270.000000]
B2 K=2       2      0.294847 [0.290531,0.299164]  0.361427 [0.357794,0.365061]  12.309800 [12.198950,12.420650]  208.497559 [208.497559,208.497559]  262.000000 [262.000000,262.000000]
B2 K=4       2      0.237962 [0.237670,0.238254]  0.305800 [0.303062,0.308539]  12.657250 [12.639100,12.675400]  225.593750 [225.593750,225.593750]  264.000000 [264.000000,264.000000]
Direct MLP   2      0.241710 [0.234186,0.249235]  0.310964 [0.308410,0.313518]  0.407025 [0.401900,0.412150]  9.530273 [9.530273,9.530273]  22.000000 [22.000000,22.000000]
```

## COMMON_CAP

```text
Method       seeds  MSE mean [min,max]           MAE mean [min,max]           B8 ms mean [min,max]         allocated MiB [min,max]      reserved MiB [min,max]
Raw          1      0.278007 [0.278007,0.278007]  0.292479 [0.292479,0.292479]  45.809850 [45.809850,45.809850]  209.076660 [209.076660,209.076660]  270.000000 [270.000000,270.000000]
B2 K=1       2      0.339939 [0.333789,0.346089]  0.404090 [0.400861,0.407320]  12.184700 [12.071350,12.298050]  199.934082 [199.934082,199.934082]  270.000000 [270.000000,270.000000]
B2 K=2       2      0.294847 [0.290531,0.299164]  0.361427 [0.357794,0.365061]  12.098050 [12.094700,12.101400]  208.497559 [208.497559,208.497559]  262.000000 [262.000000,262.000000]
B2 K=4       2      0.237962 [0.237670,0.238254]  0.305800 [0.303062,0.308539]  23.899775 [23.448200,24.351350]  208.595703 [208.595703,208.595703]  264.000000 [264.000000,264.000000]
Direct MLP   2      0.241710 [0.234186,0.249235]  0.310964 [0.308410,0.313518]  0.405300 [0.395450,0.415150]  9.530273 [9.530273,9.530273]  22.000000 [22.000000,22.000000]
```

- K=1: positive=False; seed MSE 92601: 0.346089 (below Raw=False), 92602: 0.333789 (below Raw=False); Raw=0.278007. Mean MSE gain vs Raw -22.28%; MAE change +38.16%; default speed ratio 1.365x; allocated change -20.34%. Predicates: {"both_seed_MSE_below_raw": false, "both_seed_default_allocated_below_raw": true, "both_seed_default_latency_below_raw": true, "mean_MSE_gain_at_least_1pct": false}.

  Direct MLP comparison: B2 MSE change +40.64%; B2/Direct latency 29.485x; allocated 20.979x. Direct dominates all three mean axes=True; paired seed dominance={"92601": true, "92602": true}.

- K=2: positive=False; seed MSE 92601: 0.290531 (below Raw=False), 92602: 0.299164 (below Raw=False); Raw=0.278007. Mean MSE gain vs Raw -6.06%; MAE change +23.57%; default speed ratio 1.331x; allocated change -16.93%. Predicates: {"both_seed_MSE_below_raw": false, "both_seed_default_allocated_below_raw": true, "both_seed_default_latency_below_raw": true, "mean_MSE_gain_at_least_1pct": false}.

  Direct MLP comparison: B2 MSE change +21.98%; B2/Direct latency 30.243x; allocated 21.877x. Direct dominates all three mean axes=True; paired seed dominance={"92601": true, "92602": true}.

- K=4: positive=True; seed MSE 92601: 0.237670 (below Raw=True), 92602: 0.238254 (below Raw=True); Raw=0.278007. Mean MSE gain vs Raw +14.40%; MAE change +4.55%; default speed ratio 1.294x; allocated change -10.12%. Predicates: {"both_seed_MSE_below_raw": true, "both_seed_default_allocated_below_raw": true, "both_seed_default_latency_below_raw": true, "mean_MSE_gain_at_least_1pct": true}.

  Direct MLP comparison: B2 MSE change -1.55%; B2/Direct latency 31.097x; allocated 23.671x. Direct dominates all three mean axes=False; paired seed dominance={"92601": true, "92602": false}.

## Training and scope

Full V: 145 origins, 145 with observed targets, 48720 observed scalar targets. No target imputation or outcome-dependent origin removal.

```text
run_id  selected_epoch  search_train_origins  training_status
B2_ETTm1_K1_OFF_s92601  3.0  34281  CAP3_STILL_IMPROVING
B2_ETTm1_K1_OFF_s92602  3.0  34281  CAP3_STILL_IMPROVING
B2_ETTm1_K2_OFF_s92601  0.25  34281  CAP3_CONVERGENCE_UNRESOLVED
B2_ETTm1_K2_OFF_s92602  1.0  34281  CAP3_CONVERGENCE_UNRESOLVED
B2_ETTm1_K4_OFF_s92601  0.25  34281  CAP3_CONVERGENCE_UNRESOLVED
B2_ETTm1_K4_OFF_s92602  0.25  34281  CAP3_CONVERGENCE_UNRESOLVED
DirectMLP_ETTm1_s92601  3.0  34281  CAP3_STILL_IMPROVING
DirectMLP_ETTm1_s92602  1.0  34281  CAP3_CONVERGENCE_UNRESOLVED
```

6 B2 searches, 6 fresh A60 refits, 2 Direct MLP search/refit pairs; 18 cost facets, 180 warmups and 360 timed trials. Actual score reading follows driver exit0 and completed train/baselines/costs journal entries.

Full stride1 TRAIN, strict minimum A_stop selection with earliest ties, all B2 selections sealed before fresh A60 refit/V; Direct MLP selects separately under the fixed recipe. Cap3 does not prove convergence. FP32, RevIN OFF, L512/H48.

Each cost facet has a distinct (PID, process creation time). Whole B8 physical+mask -> causal fill -> fixed scaler -> online model -> inverse -> CPU output excludes slicing/load/hash/verification. Full V forecasts, row-policy parity, FP64 observed equal-channel macro MSE/MAE, A60 scaler/history/target-mask, raw trial timing quantiles, tensor memory peaks and source/model/checkpoint/selection bindings are independently checked. Physical output bytes were not saved; their hash is not recomputed.

Figure rows: DEFAULT128 and COMMON_CAP=16; columns: normalized MSE, whole B8 latency, allocated peak. Bars show means and whiskers seed min-max, not confidence intervals. Raw is one deterministic model. MAE and reserved peaks remain in tables. Timed20 p95 is not a service-tail guarantee; tensor peaks are not process hard caps.

Adaptive development V and two seeds do not establish independent TEST performance or statistical significance. Prior data exposure, shared station ancestry and possible foundation pretraining overlap remain limitations. No new LoRA/Ridge comparison. Historical Jena B8 confirm K3/6/12, Raw and merged LoRA remain a separate prior execution with existing ancestry/TEST-exposure limits; they do not enter ETTm1 classification. Prior CUBLAS values were not recorded; new receipts record inherited settings.

## Activation and preserved provenance

ACTIVATION_AMENDMENT_WAVE_003 registers the latest explicit request before new scores and activates ETTm1 despite the old conditional stop after wave2 positives. Prior Appliances and ETTh1 positives and the original plan are preserved.

Historical preparation amendment: initial ETTh2 prefix13936 materialized16 potentially overlapping protected hours relative to same-source ETTm2; zero new training/forecast use. Superseded WAVE001 and source preparation are preserved. Replacement ETTh2_safe17400 uses a custom cohort with freshly streamed prefix13920. This historical ETTh2/ETTm2 incident is unrelated to ETTm1 fresh prefix preparation beyond inherited cohort history. ETTm1 uses its separately checked N69680/prefix55744/native15min and same-station ETTh1 cutoff.

No new protected T numeric access or whole-raw numeric hashing. Consumed prefix/array hashes are separate from historical full-source metadata hashes. Unexecuted Solar numeric archives are not opened or hashed.

```text
wave dataset status detail
1 ETTh2 SUPERSEDED_BEFORE_TRAINING Original ETTh2 preparation possible shared-source 16-hour overlap; wave never trained
1 appliances SUPERSEDED_BEFORE_TRAINING Original ETTh2 preparation possible shared-source 16-hour overlap; wave never trained
1 ETTh1 SUPERSEDED_BEFORE_TRAINING Original ETTh2 preparation possible shared-source 16-hour overlap; wave never trained
1 uci360_air_quality NOT_READY_UNOBSERVED_CHANNEL [{"role": "A_stop", "unobserved_columns": ["NMHC(GT)"]}]
2 ETTh2_safe17400 PRESERVED_PRIOR_WAVE All K and both seeds disclosed; development V, not independent TEST
2 appliances PRESERVED_PRIOR_WAVE All K and both seeds disclosed; development V, not independent TEST
2 ETTh1 PRESERVED_PRIOR_POSITIVE_EXPLORATORY Prior wave2 positive preserved by activation amendment; not rerun or counted in wave3.
3 ETTm1 POSITIVE_EXPLORATORY All K and both seeds disclosed; development V, not independent TEST
NEXT_UNREGISTERED solar_full137 READY_NOT_EXECUTED Unexecuted full137-channel Solar; metadata only, numeric prefix not opened or hashed. Additional example found in wave3; retain prepared candidate.
```

Solar full137 remains READY_NOT_EXECUTED after the additional example. Solar has zero performance experiments counted in this report. The separately authorized RawContext campaign has a separate execution and budget.

RUNS.csv discloses every seed; SUMMARY.csv gives means/ranges; CANDIDATES.csv preserves all candidate outcomes.