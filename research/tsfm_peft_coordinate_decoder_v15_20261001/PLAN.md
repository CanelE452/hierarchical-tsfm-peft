# v15: Coordinate selection and an untied forecast-innovation decoder

Authority: the human delegated judgment and continued execution: "너가 직접 판단해서 쭉 진행하게 해줘", with the active goal "해결할수 있게 분석하고 해결해줘". This bounded contract is the assistant's prospective choice under that delegation, not a fabricated human approval of these exact parameters. It neither reopens v14 nor spends a prior campaign's technical reserve. Base/live main was886cda62e94070b6f3ec2bf07d1092f4d5c9c15f. No new server, paid service, download or global environment change.

## Why this comparison

The completed v14 component attribution did not find one P/Q replacement solving all external accuracy gaps. Jena's compressed main is a substantial source of difference, while Robin/Peacock also retain Q differences and Hog reverses by period/mask. All composed pooled MSE point estimates remain above full-MSE LoRA. This does not justify a larger temporal head or more old-family optimization.

One untested restriction is that past B models tie the input and output directions. A coordinate selector preserves actual univariate input rows, for which F0(XJ)=F0(X)J up to batch numerical differences. A dense PCA input does not have this property. An independent decoder can transfer the selected forecasts' innovation to unselected channels; a tied decoder cannot. We compare these two changes explicitly with a parameter- and selection-matched PCA-input decoder. No new K, loss, input-dependent gate, nonlinear head or dataset.

The strongest counterarguments are that forecast innovations need not share stable cross-channel structure, TRAIN-input PCA is not a forecast-innovation basis, the direct model may suffice, and uncompressed full LoRA may remain more accurate at practical microbatch costs. This is one finite test, not a guarantee of accuracy or originality.

## Standard principles and limits

AdaPTS (ICML2025) already uses learned latent input/output adapters; adapters and encoder/decoder learning are not new contributions. https://proceedings.mlr.press/v267/benechehab25a.html (publication/abstract rechecked; prior local method/code reading retained).

DEIM/Q-DEIM evaluates selected coordinates of a nonlinear function and reconstructs the whole vector. Drmac and Gugercin, SISC38(2),A631-A648,2016, DOI10.1137/15M1019271; author manuscript https://arxiv.org/pdf/1505.00370 , sections1.4/2.1 and Theorem2.1 read. The source's useful basis comes from nonlinear snapshots and its bound depends on approximation error and selection conditioning. Raw-input PCA does not inherit a forecast-error guarantee. Here QR selection is only a fixed input-space heuristic; the decoder is supervised forecast adaptation. The 2010 DEIM publication metadata/abstract was checked; its Rice report URL failed to fetch and is not claimed fully read. SciPy's official interpolative-decomposition documentation confirms actual-column rather than SVD-coordinate representations; this is not a new matrix-decomposition algorithm.

Existing NLinear S* is retained with its trained original-channel affine512->48 and last-value removal/restoration. The new method is a static output adapter of selected frozen TSFM forecasts; no new TSFM latent representation or new LoRA mathematics is claimed. The distinction from v11 B is an untied decoder and coordinate selection, not just a new name.

## Frozen data and parents

Use the exact v14/v13 TRAIN/VAL/testA/testB archives, masks, original channel order, time indices, TRAIN normalization/imputation and canonical pca_basis. Robin C17/K5 hourly, Jena C21/K6 ten-minute, Hog C32/K8 hourly, Peacock Education C13/K4 hourly. L512/H48 observations. Every dataset is already exposed development; no new confirmation claim or outcome-based dataset deletion.

S* is the actually selected DIRECT_NLINEAR of each dataset/seed92601/92602, not a latent linear substitute or v14 GELU parent. Its complete TRAIN/VAL caches exist in v14 parent_cache.json and are checked by hash/origin order. Reuse reference TEST forecasts for DIRECT,PCA_LEVEL,A,F0,FULL_MSE,NATIVE. Raw full-TRAIN F0 cache was not found; generate only the needed K-row native/PCA frozen forecasts under this new ledger, rather than assume it exists. Backbone revision remains772f3d25d38aec6d914c8949dab4462e2d46f5d8 and downloads remain0.

## Fixed model and input selectors

X:[B,512,C]. S*(X):[B,48,C]. U0:[C,K] is the recorded TRAIN PCA basis. Obtain pivot indices with CPU FP64 scipy.linalg.qr(U0.T, mode='economic', pivoting=True), take the first K in the returned deterministic pivot order, and save actual indices/basis/hash. No rank/selection-seed/performance sweep. Let J be the corresponding C-by-K coordinate selector.

For E=J (RAW_FREE) or E=U0 (PCA_FREE), define

  Z_E(X)=F0(X E)-S*(X) E,
  f_E,W(X)=S*(X)+Z_E(X) W,

where W:[K,C] is an independent bias-free channel decoder shared over origins/horizons. F0 includes native context scaling, the0.5 quantile and48-step crop. Compute S*(X)E, not S*(XE). Freeze F0, S*, E, all statistics/buffers. Do not include LEVEL's G or LoRA in the new models. Registered adapted coefficients K*C are85/126/256/52 respectively; S*'s historical24624 fitted parameters and complete backbone deployment bytes are reported separately.

W=0 is the exact direct-parent initial/fallback function. There is no initialization RNG or neural optimization. If the zero decoder is selected, deploy S alone without loading F0 and mark TSFM value unestablished. No approximate coefficient pruning.

Unfitted controls for each seed use the same generated features:
- RAW_TIED: E=J,W=J.T, exactly the corresponding B coordinate-selector special case with Gamma=I. Only selected channels receive F0.
- RAW_INTERP: E=J,W=solve(U0.T@J,U0.T) in FP64 then FP32. This is input-space Q-DEIM interpolation, not a supervised decoder. Record conditioning and WJ-I; do not claim an orthogonal P/Q split. If numerically rank deficient, retain a technical unavailable control and its cause without reselecting channels/K; do not secretly regularize it.
- PCA_TIED: E=U0,W=U0.T, existing B_FIXED_U form with Gamma=I.
These are fit0 references, not best-tuned competitors. W learned versus fixed and RAW_FREE versus PCA_FREE answer different questions.

## Exact supervised adaptation and selection

Use every recorded TRAIN origin, all48 horizons and the original observed-target mask. For each target channel c, fit W[:,c] in FP64 to Y_c-S*_c. Normalize each channel's squared objective by its observed TRAIN count n_c. With s_E = mean over all TRAIN origin/horizon rows of ||Z_E||^2/K, minimize

  (1/C) sum_c { ||Z_E,c W_c - (Y-S*)_c||^2/n_c + lambda*s_E*||W_c||^2 }.

For s_E=0 the unique declared policy is W=0; otherwise s_E is fixed from TRAIN only. No intercept, horizon-specific decoder, target filling, sample subsampling or new channel weights. Lambda candidates are[0,0.001,0.1]. They are a modest finite regularization comparison, not known optimal values. Lambda0 uses FP64 least squares/SVD on the observed design, not an unstable inverse; positive lambda uses the equivalent augmented least-squares objective. Share computed design/source receipts across lambda but count every independent lambda/seed/family/dataset fit separately. Internal C output solves are recorded, not counted as C separate tuning experiments.

Export W to FP32. Selection scores are full VAL masked macro MSE using FP32 exported coefficients and the fixed FP32 cached features. For each dataset/family compare mean two-seed VAL losses for three lambdas and the zero-decoder parent. Exact ties prefer parent, then lambda0,0.001,0.1. Select one common lambda or parent for both seeds. No TEST-based choice, coefficient sign clipping, extra penalty, iterative refinement of model definition, or fit extension.

48 basic coefficient fits =4datasets x2families x2parentseeds x3lambda. Neural fits0, optimizer updates0. A retry/restart solving coefficients again is another real fit, even if the linear algebra is cheap. Four technical recoveries give maximum52 attempts; they do not permit another candidate.

Check finite features/solutions, original frozen bytes, complete TRAIN counts, direct objective/regularized normal-equation residual, lambda0 minimal-norm diagnostics, FP64-to-FP32 prediction/loss difference, zero-parent parity, selector row order, output restoration, and cache/online parity. Do not demand all coefficients change or turn a normal negative result into a solver failure. Synthetic checks have no optimizer and are CPU-only. Use existing parity atol1e-5/rtol1e-4 and same-path replay atol1e-6/rtol0; report differences rather than enlarge tolerance to pass.

## Execution and evaluation

1. Write this plan, actual authority, immutable protocol/provenance, new ledger and reuse manifests; preserve v1-v14.
2. Prepare TRAIN-only selectors; verify data/parent cache bytes; check the new algebra on synthetic arrays without fitting real data.
3. Cache frozen TRAIN/VAL K-row F0 outputs for E=J,U0 once per dataset, not once per seed. Check raw selector/full original F0 parity on fixed VAL inputs; no full TRAIN uncompressed reference inference is needed.
4. Complete48 prescribed coefficient fits and all mean-seed VAL selections. Record W, all penalties, objectives, rank/condition, selected and adverse VAL results, exact cache/origin receipts. Seal all four datasets together before new TEST scoring.
5. Generate selected and three fixed-control TEST forecasts using the fixed selectors and selected W; replay existing reference predictions at identical origin/channel/mask order. All84 model instances (21/dataset) remain, including unfavorable/fallback cases: new4, fixedcontrols6, references11. Parent references are DIRECT2,PCA_LEVEL2,A2,F01,FULL_MSE2,NATIVE2.
6. Report primary MSE plus MAE/signed bias, full/period/seed/channel scores, pooled per-channel sums/counts then mean seed loss, never an ensemble or cross-dataset raw mean. Paired7-or42-origin blocks/2000draws/seed9262026 respect period boundaries and are conditional on exposed selected parents/models, not full selection uncertainty. Main contrasts RAW_FREE-PCA_FREE, RAW_FREE-DIRECT, learned-free versus its fixed controls, remaining external F0/FULL_MSE/NATIVE/A gaps. No causal claim that decoder learning or native-input preservation alone explains every difference.
7. Same-session costs and concrete method decision; source/numeric/artifact verification, limited necessary plots, ordinary publication.

## Fixed practical cost scope

All four datasets, independently of accuracy. Roles RAW_FREE2,PCA_FREE2,DIRECT2,A2,F01,FULL_MSE2,NATIVE2:13 instances/dataset. Full B1/B4 gives26 configs. All five uncompressed instances also get B1K/B4{4K,K}:15 extra configs. Six compressed new/A instances additionally get B4K:6 extra configs. Three balanced blocks plus F0 pre/post sentinels at each B gives153 GPU rows/dataset,max612. CPU DIRECT2 x B1/B4 x3blocks gives12/dataset,max48. Exact function/state/execution aliases (e.g. selected zero W) may share a measurement with an explicit alias receipt; do not report a repeated source as independent timing.

FP32,TF32off,CPU4threads,first24VAL origins,10warmups,3passes of the full24-origin workload,3balanced blocks; same pinned model/device and fullH48C outputs. Standardized CPU input to complete CPU forecast includes transfers, selector/feature/decoder/assembly and online F0/S; load/disk/hash/scoring/ledger are outside. No future prediction cache in timing. LoRA merge, export/full/chunk and direct CPU/GPU parity precede comparison. Full rows are split only along independent series, never time/horizon. Output-equal small-batch alternatives are provided to compressed and full models. Report peak allocated/reserved/resident/process where available, total parameter/buffer bytes, historical S fit plus new W adaptation, and incremental measured time separately. No historical timing ratio or C/K speed claim. Preserve all block/sentinel drift, no new profiler campaign.

## Separate bounded resource allocation

This new campaign's ceilings are52 real coefficient attempts (48basic+4technical), GPU3600seconds on the same local single GPU/one GPU job, CPU coefficient fitting plus numeric analysis1800seconds, CPU checks600seconds, new storage2GiB, downloads0, neural optimizer/synthetic optimizer sessions0. Read-only file/literature work and document writing are not coefficient fitting. Illustrative GPU allocation: frozen caches900s, prediction/parity240s,cost2400s,recovery60s; these are operating estimates, not measured guarantees. Existing v13 cost528 rows used1420GPU seconds and v14 caches820.967s provide scale only. Inspect initial throughput before overspending; report incomplete scope if the fixed total cannot cover it, do not drop an unfavorable dataset or auto-increase the cap. Cost2400s is within3600s, not additional.

Reserve every actual numerical/model/check job before running under a new ledger. Coefficient attempts are children of a CPU-analysis job and not double-counted in CPU seconds. Failures/re-solves count. No fit after the selection seal; post-TEST metric/restore repairs preserve first exposure and correct all affected counterparts, while substantive selection/model changes require a new decision. GPU conflict waits30min/incident,60mintotal, without stopping other projects. A live process must be verified before resuming; stale records are not proof of running work.

## Decisions, limits and artifacts

If RAW_FREE improves PCA_FREE and fixed raw controls while retaining a meaningful position against full LoRA/direct alternatives, keep the bounded coordinate+decoder hypothesis and state all accuracy/cost losses. If PCA_FREE is enough, do not credit coordinate selection. If zero W/direct is enough, record lack of TSFM benefit. If fixed interpolation/tied control is enough, do not credit learned decoder. If external alternatives remain preferable, reject this new adaptation as a general solution; do not add selection sweeps, new bases or gates. Different datasets may disagree; no TEST-built router or universal-win requirement. Normal failures do not justify more fits. This exposed development campaign cannot itself establish generalization, novelty or acceptance.

Artifacts: PLAN/AUTHORIZATION/protocol/provenance,STATUS, reuse/data/selector/cache manifests, coefficient run/result/receipt and local weights, selection/seal, forecast/evaluation and JSON/CSV contrasts, cost raw/summary/aliases/parity, METHOD_UPDATE/TOPIC_DECISION/verification and essential comparison figures. Raw/large weights/prediction arrays remain local. The goal is not marked achieved by document count or campaign completion. Publish only related new v15 files plus existing README/history with normal commit and origin/main push; verify liveSHA. No force/reset/branch/PR/hook bypass/unrelated dirty changes.
