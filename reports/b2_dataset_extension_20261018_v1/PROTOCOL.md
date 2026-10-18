# B2 dataset extension — fixed development comparison

Registered before new training or inference on either candidate. Parent checkout:
`d2b4145c7c2b67ee580c2e9a1c68c11b083ad359`. The user requested finding candidates
and running the same conditions as the Jena/PeMS32 slide; this authorizes dataset
selection and the bounded comparison below. It does not request commit/push or
editing an existing presentation.

## Purpose and candidate choice

Determine whether the existing compressed-input B2 offers a useful accuracy,
allocated GPU tensor memory and whole-path latency choice on a second real
dataset. Success is not assumed. Both registered candidates are reported; no
candidate is replaced after viewing validation errors.

* Household: UCI individual household power consumption, existing hourly
  aggregation, seven physical measurement channels. A different domain from
  weather with related power/current/submeter measurements. K=ceil(7/4)=2.
* SML2010: UCI smart-house sensors, existing 15-minute grid, 21 sensor/control
  channels with Date/Time/Day-of-week excluded. Matches Jena's channel count and
  K=ceil(21/4)=6. Short duration and missing blocks may limit evidence.
* ETTm2 was considered: seven channels and local CSV, but a much longer series
  and no currently isolated approved prefix. Solar was considered as a physically
  shared-driver candidate, but no local asset was found. Neither is run here.

Selection uses availability/schema and training-only observability/rank
diagnostics, never new validation or test forecast scores. Candidate datasets
were used by older, different methods, so these are not previously untouched
datasets. This is a development comparison, not a new independent test claim.

Primary dataset references: [UCI Household](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption),
[UCI SML2010](https://archive.ics.uci.edu/dataset/274/sml2010),
[official ETT](https://github.com/zhouhaoyi/ETDataset),
[official LSTNet datasets](https://github.com/laiguokun/multivariate-time-series-data).
The new split below is our inherited chronological development protocol, not an
official UCI benchmark split or the original LSTNet/Informer evaluation task.

## Fixed data and scientific method

Read only the existing, separately stored FIT_TUNE physical/mask payload. Do not
open or hash raw ZIPs, EVAL inputs/targets, TEST files or full source arrays.
Validate the allowed payload binding against its existing contract. With the
existing full grid length N, use A_fit=[0,floor(.5N)),
A_stop=[floor(.5N),floor(.6N)), A60=[0,floor(.6N)), unused
M_cal=[floor(.6N),floor(.7N)), V_select=[floor(.7N),floor(.8N)). No numerical
rows beyond floor(.8N) are materialized. Preserve clocks, channels and masks.

L=512, H=48 for both candidates, original-channel outputs. Training origins have
stride 1, A_stop and V stride 48. No filtering by observed forecast values or
effects. Report missing targets, channel counts, actual origin coverage and any
unobservable channel. Targets are never filled. Causal history fill resets to
the fitted mean at each window start. Fixed per-channel population mean/std are
fit separately on A_fit for search and A60 for fresh refit; no V statistics.

B2 remains the existing P_B2 -> PCA-initialized trainable E(C->K) -> frozen
Chronos-Bolt-small F -> lag D plus original last value, RevIN OFF. F is the
pinned local FP32 47,718,016-parameter model. No new pretraining, architecture,
loss, mixed precision, quantization or compilation. AdamW lr .001,
weight_decay 0, gradient clip 1, effective batch 16, microbatch 4,
training F row microbatch 128, checkpoint_inputs False. Seeds 92601 and 92602.
Search checkpoints 0/.25/1/3 epochs; strict minimum A_stop observed equal-channel
macro MSE, earliest ties. All four B2 searches are sealed before fresh A60
refits and V evaluation. No cap extension or additional seed/K after V.

Raw requires no fit. The practical GPU alternative is the existing DirectG
channel-shared L->32->H GELU MLP innovation plus original last value, trained with
the same seeds, optimizer, batches, search schedule, A_stop selection and fresh
A60 refit. LoRA and CPU Ridge are not run in this extension. Therefore no
superiority over those alternatives is claimed. Selecting epoch zero and a
last checkpoint still improving are reported explicitly.

## Accuracy and execution constraints

Use independent FP64 equal-channel observed macro MSE and MAE on normalized
original-channel targets, plus per-channel counts. Preserve private predictions,
origins, scaler and model bindings; publish only aggregate evidence. Compare
seeds individually and their mean; two seeds do not establish significance.

For B=8, all F methods use the same per-call row cap B*K: Household 16 and SML
48, clamped to actual stream count. Raw has ceil(C/K) calls (four for both),
B2 one. Regenerate all V outputs under original effective row128 and the common
cap; require normalized predictions and metrics within atol1e-5/rtol1e-4.
Unequal final batches retain actual calls/stream accounting. No cost result is
valid without parity and finite observed output.

Each route is loaded in a fresh OS process. Timed path: physical past+boolean
mask -> causal fill -> fixed A60 scaler -> real model -> inverse scaler -> CPU
numpy. Window extraction, loading and whole-V parity are outside timing.
Synchronize CUDA before/after each trial; 10 warmups and 20 timed trials on the
same first eight V origins. Reset peak memory after warmup and per timed trial;
record loaded and peak allocated/reserved memory, raw trial rows, median/IQR/p95,
actual F calls and pinned F state. No cached predictions in timed execution.
Twenty repetitions are technical timing samples, not accuracy replications.
Memory is allocated/reserved tensor memory, not a system/process hard cap.
Timing uses one fixed origin batch and does not establish workload tail latency.

Existing Jena numbers may appear alongside the new results only with their
original identity and historical measurement context clearly identified. New
candidate MSE values are not compared numerically across different datasets.

## Execution, stopping and preservation

At most four B2 searches/refits and four DirectG searches/refits. One GPU process
at a time, no unrelated process termination or GPU reset. Reuse the pinned
Python3.11/OpenBLAS environment with two CPU/BLAS threads. Initial operating
estimate: one to two hours including setup, training, costs and verification;
update this estimate from the first completed training segment. This estimate
is not a user-imposed deadline and never justifies silently dropping work.
On interruption preserve attempts/checkpoints and resume the same identities;
on invalid data/parity stop the affected route and report the cause, without
changing the scientific recipe. No result-driven retries.

All new outputs use this namespace. Preserve the original dirty E: checkout,
all existing source/results/campaign ledgers and presentations. No commit/push.
Only a final append to the project's existing history is permitted outside this
namespace. Validate focused synthetic tests and actual data/model/cost receipts.
