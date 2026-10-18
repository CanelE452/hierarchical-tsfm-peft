# Continuing dataset and latent-width search

Registered before new candidate numerical preparation or training. Parent:
`d2b4145c7c2b67ee580c2e9a1c68c11b083ad359`. User explicitly requested more
datasets, continuing until a useful example resembling the Jena K trade-off
slide is found. This authorizes rolling dataset batches and the fixed K grid;
it does not authorize changing old experiments, commit/push, or claiming
independent generalization from an adaptive development search.

## Purpose and concrete outcome

Find additional real multivariate datasets on which compression provides a
useful accuracy/time/memory choice under the existing B2 recipe. The consumer
is the user's experiment presentation. Each dataset receives a three-width
comparison against Raw and a practical Direct MLP alternative. All failed,
negative, excluded and successful candidates remain visible in the registry.

A positive exploratory example requires one registered K with both seeds'
normalized V MSE below Raw, their mean at least 1% below Raw, and both seeds'
median whole-path latency and allocated tensor memory below Raw under the
default row128 path at batch8. Report MAE even if worse. Direct MLP results
remain visible; a positive relative to Raw does not establish superiority over
MLP, LoRA, Ridge or every competing forecast model. The common-row-cap facet
must also be measured and shown. Do not silently equate its claim with the
default-path claim. This operational rule is fixed before new scores.

Inspect at least the first rolling batch of multiple new datasets, even if its
first dataset is positive. Continue further batches if no verified positive
example exists. Every batch's datasets and execution order are sealed from
availability/schema/TRAIN-only evidence before that batch's new forecast scores.
All B2 searches within that batch are selected and sealed before fresh refit/V.
Each Direct MLP baseline uses its own fixed A_stop selection before its fresh
refit/V; its recipe never adapts to B2 or other baseline V scores.
Do not adapt K, seed, loss, epoch cap, channel subset or preprocessing to V.

## Candidate families and reasons

Candidates: ETTh1/ETTh2/ETTm1/ETTm2 (related electrical loads and transformer
temperature); Exchange (several currencies, contrasting nonperiodic common
trends); UCI Air Quality and Beijing pollution/sensor series (related physical
measurements); Solar (shared astronomical driver across sites); full Electricity
and Traffic (larger correlated collections). Prefer admissible local sources.
The ETT frequency variants share physical ancestry and are not independent
domains. Weather derived from Jena is not a new independent dataset. Existing
Household/SML results stay in their previous namespace and are not relabeled.

Solar is the less certain shared-driver candidate; daily darkness may make its
correlations easy without improving forecasts. Exchange is deliberately a
different, less periodic domain; cross-channel dependence does not guarantee
predictability. Neither is assumed to win. Prioritize smaller complete datasets
to reach evidence sooner, then larger ones when needed; never shrink TRAIN
origin coverage or silently select channels to save runtime.

Primary references:
[ETT](https://github.com/zhouhaoyi/ETDataset),
[LSTNet datasets](https://github.com/laiguokun/multivariate-time-series-data),
[UCI Air Quality](https://archive.ics.uci.edu/dataset/360/air+quality),
[UCI Beijing PM2.5](https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data),
[UCI Beijing multi-site](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data).
The split below is the inherited development protocol, not each benchmark's
official split. Pretraining overlap and previous development exposure remain
unknown unless authoritative source evidence says otherwise.

## Fixed method, data and split

Keep the pinned local Chronos-Bolt-small FP32 and Python3.11/OpenBLAS environment,
P_B2 -> PCA-initialized trainable E -> frozen F -> lag D with original-channel
last-value anchor, RevIN OFF, L512/H48. q=ceil(C/4), widths are distinct sorted
{max(1,ceil(q/2)), q, min(C,2q)}. This gives Jena's 3/6/12, and C7's 1/2/4.
Each K is separately trained at seeds92601/92602. No K is selected using V as
an estimate of unbiased performance; all K scores are disclosed.

Raw is zero-shot. Direct MLP is the existing channel-shared L->32->H GELU
innovation plus original last value, same two seeds and selection/refit recipe.
AdamW lr.001, weight_decay0, clip1, effective batch16/microbatch4, training
F row128/checkpoint_inputs=False, checkpoints0/.25/1/3. Select strict minimum
A_stop macro MSE, earlier ties; fresh A60 refit for the selected epoch.
Do not declare cap3 models converged; retain last-checkpoint improvement flags.

For original N, A_fit=[0,.5N), A_stop=[.5N,.6N), fresh A60=[0,.6N),
unused M_cal=[.6N,.7N), V=[.7N,.8N), protected T=[.8N,N), integer floors.
TRAIN stride1, A_stop/V stride48, full H targets within role. Keep source
physical channels/time grid and target masks. Causal history filling resets to
the fitted mean at each window. Fit search scaler on A_fit and refit scaler on
A60 only. Never fill targets, inspect V to choose channels, or standardize on
the full dataset. Equal step counts may imply different physical horizons.

Only materialize numeric rows through floor(.8N). Prefer separately isolated
FIT_TUNE payloads whose headers/contracts prove prefix boundaries. For CSV/gzip
sources, establish N/schema from authoritative metadata and stream only the
registered prefix; do not load/hash full numeric files or read protected tail
records. Record prefix-byte/array fingerprints and source size/metadata without
claiming a full-file numeric SHA. Calendar index/day-of-week encodings are not
physical targets; channel exclusions must be specified before numeric loading.
If only a normalized full-data cache exists, find a physical source instead.

## Measurement and verification

Both facets use B8 and the same selected models/cohort:
default=F row128; common=F row8q fixed across all widths and Raw for that dataset.
Record actual calls and streams per arm, including K above q and unequal final
batches. Regenerate all V forecasts and require normalized predictions/metrics
parity across row chunking, atol1e-5/rtol1e-4, not just a smoke batch.

Fresh OS process per route/facet. physical+mask -> per-window fill -> fixed
scaler -> real online model -> inverse -> CPU output. Exclude slicing/load and
verification, synchronize CUDA, 10 warmups/20 timed repetitions on fixed first
eight V origins. Record all raw trials, p50/IQR/p95, loaded/peak allocated and
reserved tensors, states before/after, source/data/model/prediction identities.
Inherit parent CUBLAS workspace configuration and record its actual value.
No cached forecasts in timing, training/cost overlap, compilation or quantization.
Allocated tensor memory is not a process hard cap; technical timing repetitions
are not seed replications or production tail latency guarantees.

Independently recompute FP64 observed equal-channel MSE/MAE, model selection and
cohort bindings; report origins, observed targets, constant/unobserved channels.
Two seeds and adaptively searched development V do not establish significance.
Preserve every attempt and repair. One GPU worker at a time; do not terminate
other applications. Pause only for an explicit user request or actual resource
failure, preserving checkpoints for an identical resume. A block's evidence
can require strategy reassessment after three negative datasets, but the user's
continuing search remains active and new data batches proceed under this recipe.

All new files stay in research/reports/.cache b2_dataset_search_20261019_v1 and
its focused tests, with an append to existing project history. Old source,
results, attachments and original dirty checkout remain unchanged.
