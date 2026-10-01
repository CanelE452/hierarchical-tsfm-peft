# V16 decision: temporal allocation does not resolve the accuracy gap

Accuracy, temporal diagnostics and all predeclared same-session deployment measurements are complete. The three saved-artifact verification groups passed. This is a completed finite comparison, not a full frontier or an independent confirmation. All four datasets are exposed development.

The fixed all-channel temporal candidate successfully learns from its poor zero-update initialization, but does not match full-resolution MSE-LoRA on any dataset's pooled MSE or MAE. It therefore does not solve the requested common accuracy problem. Jena supplies a narrower observation: temporal allocation improves the earlier channel-compressed A model's point MSE and MAE, but its MSE difference remains uncertain and the frozen fine prediction actually hurts relative to the same selected coarse output. Matched costs identify a Jena batch4 accuracy-throughput tradeoff; they do not establish a general accuracy recovery or stable latency advantage.

## What was actually changed

Every original channel reaches Chronos-Bolt-small. Consecutive groups of four historical observations are averaged, reducing its context512 to128. The native median head still emits64 steps; its first12 are repeated to48 original steps. A frozen, previously selected original-channel NLinear supplies its within-four-step predicted variation. Only ordinary q/v LoRA r8/alpha16 is newly trained:

    f(X) = R F_phi(A X)[:12] + (I - R A) S*(X).

There is no new G, gate, learned decoder or stride search. The fixed fine forecast cannot change the coarse forecast's four-step means. Step0 is the same formula with F0, not full-resolution F0 or DIRECT. The parent DIRECT fit is historical adaptation, not free knowledge; its24624 parameters and the full backbone remain in deployment alongside294912 newly fitted LoRA parameters before merging.

All16 planned fits completed without failure or retry, using two LRs and two seeds per dataset. New training GPU-job occupancy was1529.442 seconds. Whole-VAL paired-seed selection chose LR1e-3 for Robin/Hog/Peacock Education and1e-4 for Jena. Best epochs were respectively(5,2),(5,1),(5,3),(25,17). Every selected state is after step0. All choices were jointly sealed before the new evaluation. Initial-function parity, actual updates, frozen-state checks, complete-epoch restart payload, selected restore and original schedule/selection arithmetic passed. These checks establish implementation validity, not improvement or an optimization optimum.

## External accuracy, including adverse results

Pooled values combine error sums and observed counts within each channel across the two fixed periods, then average channels and seed losses. They are not prediction ensembles or a cross-dataset score. Entries are MSE / MAE; lower is better.

| Dataset | Temporal LoRA | Earlier A | DIRECT | Full MSE-LoRA | Native LoRA |
|---|---:|---:|---:|---:|---:|
| Robin |0.282457 /0.325842|0.273071 /0.323059|0.327724 /0.353410|0.218517 /0.258718|0.223610 /0.256347|
| Jena |0.248391 /0.256284|0.253093 /0.275047|0.273984 /0.286714|0.232892 /0.238943|0.240590 /0.226584|
| Hog |3.078891 /0.751017|2.658552 /0.796149|3.059351 /0.775475|2.625304 /0.623919|2.683560 /0.618902|
| Peacock Education |0.245750 /0.349818|0.203266 /0.315004|0.219884 /0.329450|0.198176 /0.298839|0.200123 /0.299731|

MSE losses relative to full MSE-LoRA are29.26%,6.66%,17.28%,24.01%; MAE losses are25.94%,7.26%,20.37%,17.06% in the same dataset order. The paired conditional block95% MSE relative intervals are[16.58,41.02],[3.66,9.98],[8.04,23.31],[18.89,30.99] percent. They condition on the selected seeds/models and omit accumulated model/parent/selection uncertainty. They are not equivalence tests.

Jena versus A improves point MSE1.86% and MAE6.82%. Its MSE interval[-4.84,+1.24]% crosses zero; the MAE interval[-7.97,-5.79]% does not. MSE is0.17% worse in TEST-A and4.93% better in TEST-B. This is a limited development result, not a common recovery. Hog has lower MAE than A but higher MSE; versus DIRECT the point MSE is0.64% worse and MAE3.15% better, with both conditional intervals spanning zero. Its DIRECT-relative MSE direction reverses between seeds. Peacock is worse than the cheap DIRECT in both metrics. Full MSE-LoRA also has lower seed-mean MSE and MAE in each individual period of all four datasets. F0, every seed, both periods, all channels and all seven fixed paired contrasts remain in evaluation01.json and its CSVs.

## What the controls and diagnosis explain

LoRA substantially improves the same temporal allocation's initial function on all four datasets. That validates adaptation within this allocation, not superiority to the full-resolution alternatives.

The fixed fine prediction improves pooled MSE over the same-checkpoint COARSE_ONLY on Robin17.38%, Hog2.21% and Peacock19.78%. It worsens Jena MSE1.21% and MAE1.52%; both periods and both seeds favor COARSE_ONLY on those two metrics, as do the paired pooled intervals. This is an output-contribution contrast using the same learned main path, not a separately optimized no-fine training policy. We do not silently replace the selected method with COARSE_ONLY or assign it unmeasured deployment costs.

The temporal diagnostic uses only completely observed blocks of four future positions for one channel. It selects valid blocks before projecting errors; it never substitutes zero for missing targets. It retains99.991%,100%,99.751%,99.936% of observed targets for Robin/Jena/Hog/Peacock and represents every fixed channel. The subset metric remains distinct from the original masked macro MSE.

| Dataset | Excess complete-block MSE vs full MSE-LoRA | Coarse P excess | Fine Q excess |
|---|---:|---:|---:|
| Robin |0.063923|0.051596|0.012327|
| Jena |0.015500|0.012216|0.003284|
| Hog |0.461103|0.429369|0.031734|
| Peacock Education |0.047631|0.042772|0.004859|

The larger excess is in the coarse component in every dataset. The current fine component alone cannot explain the observed gap. More strongly, Robin/Hog's current P energies(0.241285/2.817982) already exceed full MSE-LoRA's complete-block total errors(0.218061/2.554818): holding P fixed, even a perfect Q cannot close those subset gaps. This lower-bound argument does not extend unchanged to the original masked metric or rule out useful Q improvements elsewhere.

Preserving channel identity while shortening history did not suffice. This does not uniquely identify temporal information loss, pretrained temporal-scale mismatch or remaining optimization/selection effects as the cause: the experiment changes those factors together. A new fine head or longer training is not justified merely by a negative result. The data support rejection of this fixed common-recovery design; they do not prove all temporal compression or all PEFT impossible.

## Cost and final disposition

All588 GPU rows and48 CPU rows completed, including every predeclared full/chunked F0, full MSE-LoRA, native LoRA, earlier A, DIRECT and temporal configuration. FP32, TF32 off, four CPU threads and the same24 VAL origins were used. Timing includes standardized CPU input, transfers, complete online prediction and original-channel CPU output; loading, disk, scoring and bookkeeping are outside. Each shape received10 warmups, three full-origin passes and three balanced blocks. Median block values are aggregated per seed, then seed values are averaged. These are same-session measurements; no old/new timing ratio is used.

Batch4 entries below are peak allocated MiB / origins per second. Reserved memory, seed/block ranges and CPU DIRECT are summarized in cost_comparison.csv and cost_summary.json. Individual timing repetitions are retained in cost_cuda_rows.json and cost_cpu_rows.json.

| Dataset | Temporal full | Earlier A full | Full MSE-LoRA full | Full MSE-LoRA chunk4K | Full MSE-LoRA chunkK |
|---|---:|---:|---:|---:|---:|
| Robin |213.51 /220.29|213.11 /303.40|265.22 /213.84|214.47 /86.62|196.76 /25.49|
| Jena |218.68 /280.14|217.97 /269.62|281.94 /206.70|218.34 /81.43|197.89 /25.13|
| Hog |233.13 /264.80|225.67 /265.93|328.93 /165.52|225.74 /82.14|200.20 /22.71|
| Peacock Education |207.76 /244.42|210.03 /251.64|248.56 /228.63|210.00 /84.92|195.63 /27.84|

Jena illustrates the supported choice: temporal MSE0.248391 at218.68MiB and280.14origins/s versus full MSE-LoRA MSE0.232892 at281.94MiB and206.70origins/s. Chunking the same stronger LoRA reaches218.34MiB but81.43origins/s, or197.89MiB but25.13origins/s. Thus comparable memory with better accuracy is already available; the temporal candidate buys throughput at an accuracy cost. Its Jena seed/block throughput range is245.61–342.98 versus180.26–232.82 for full MSE-LoRA and203.58–331.65 for A. These ranges are not confidence intervals. The small point throughput difference against A is not a stable superiority claim.

Batch1 full latency is12.330/12.628/12.103/12.727ms for temporal versus12.055/11.707/11.896/11.756ms for full MSE-LoRA in dataset order. There is no point latency advantage. CPU DIRECT has approximately0.070–0.071ms per origin and GPU allocation is N/A, not zero. It remains a separate practical device alternative. In Hog, A has lower MSE, lower allocated memory and similar batch4 throughput than temporal, while temporal retains a MAE tradeoff. In Peacock, even DIRECT is more accurate on both metrics and far cheaper.

All24 F0 sentinel pairs are preserved. Within-block latency changes range from−7.74% to+27.57%, with substantial Robin/Peacock variation. No unfavorable block is discarded and no Windows, temperature or clock explanation is asserted. The candidate is not a generally stable speed winner. Temporal deployment contains47,742,640 parameters and190,970,560 parameter bytes, slightly more than full MSE-LoRA's47,718,016 and190,872,064; dataset-specific buffers are additional. Lower activation allocation is not a smaller weight model. Its registered new LoRA294912 parameters and historical DIRECT24624 parameters are reported separately.

Final decision: reject this fixed configuration as a common accuracy-recovery solution and do not promote it to the main paper method. Retain the limited Jena accuracy-throughput observation, alongside the negative fine-branch control and repeated-development limitation. When accuracy is the priority, the tested full-resolution MSE-LoRA is preferred on these four comparisons; under a memory constraint its slower chunked modes are explicit alternatives. No application-specific error tolerance or utility score has been invented.

The one remaining research decision is whether the limited Jena tradeoff warrants a separately fixed simplification and fresh confirmation. The current results do not authorize treating COARSE_ONLY as a newly selected deployable winner or starting another stride/head sweep. This campaign stops with its finite question answered; common accuracy and generality remain unresolved.

Operational totals before publication checks:16/20 neural attempts, zero technical retries, zero coefficient fits,3729.503 GPU-job seconds (62.16min),124.670 CPU-analysis seconds,12.763 CPU-check seconds and about180MiB new storage. The one synthetic session used zero optimizer updates. All jobs are closed. terminal_budget.json records the final publication-check accounting; unused technical reserve is not a new-method budget.

## Evidence

- PLAN.md, protocol.json and selection_seal.json: prospective scope and selection boundary.
- runs/*/result.json, curve.json and selected.json: all fitting attempts and selected states.
- evaluation01.json and model/channel CSVs:60 matched instances, all periods/seeds and source-score replay.
- temporal_diagnosis01.json/CSV: complete-block temporal P/Q energies, coverage and saved-forecast identity checks.
- verification_contracts.json / verification_predictions.json / verification_cost_report.json and final_checks.json: all three CPU self-check groups passed; not independent reproduction.
- cost_cuda_rows.json / cost_cpu_rows.json, cost_summary.json and cost_comparison.csv: complete raw matched measurements and aggregation.
- report_values.json, decision_values01.json and the two PNG/SVG figures: numeric provenance and visually inspected presentation.
- publication_checks.json, terminal_budget.json and publication_receipt.json: publication validation, terminal accounting and verified scientific-commit receipt. V1-v15 originals, failures and closed ledgers remain unchanged.
