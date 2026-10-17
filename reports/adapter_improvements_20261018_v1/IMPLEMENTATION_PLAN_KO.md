# Adapter improvements 2026-10-18 Implementation Plan

> For agentic workers: use the subagent-driven-development workflow. Read the full requested contract and this plan; implementation is followed by an independent task review and whole-campaign verification. Do not commit independently.

**Goal:** Execute the registered P/E/D improvements, preserve matched controls and negative results, independently verify predictions and whole-path costs, and publish the actual coverage to main.

**Architecture:** Reuse the frozen diagnostic artifacts and original fit/deployment/data APIs. New numerical modules and factories live only in the new namespace. A new sealed job graph, immutable receipts and persistent resource ledger bind execution; reporting code has a separate source seal.

**Tech Stack:** Existing Python 3.11 environment, NumPy/OpenBLAS (2 threads), PyTorch FP32 frozen Bolt, SciPy FP64 DCA/head fitting, pandas/matplotlib reporting.

**Spec:** User attachment `CLI_TSFM_Improvements_20261018_KO (1).txt`, SHA256 `55af78e3ade0d4cebcbc09f0f453b457afc7f748b2a2316486e4eadf8869e679`, 912 lines. The unsuffixed attachment is byte-identical. Original request is retained privately; section numbers below refer to that source.

## Global constraints

- Preserve old diagnostic code/reports/receipts and the dirty original checkout; work in the already attached clean isolated worktree at parent `2f413ac61aee7b7bd3639330193d792f1ae08bbc`.
- No new T or M_cal selection. Existing V exposure remains development use. No V-driven extension of candidates, seeds, kernels, K or grids.
- A_fit/A_stop selection, then fresh A60 refit including fresh scaler/PCA/DCA. CP0 is internal; epochs argument is [.25,1,3]. Training stride1, evaluation strideH, seeds92601/92602.
- Jena C21/L512/H48/K6; PeMS32 C32/L512/H12/K8. PeMS170/M5 large new training is excluded. M5 local assets exist, but a numeric task is unregistered.
- New campaign wall6h, GPU4h, final wall72m/GPU48m reserved; one GPU and one heavy CPU worker, BLAS2, new cache30GiB, downloads5GiB. Link prior campaign usage and enforce any stricter applicable inherited limits. Never reset a resumed campaign.
- Admit complete candidate/control bundles using recorded profiles; stop new training at80% or reserve boundary. Preserve partial optimizer/RNG/sampler/stage state and explicit resume commands.
- Source/checkpoint/data/mask/scaler/origin/F/precision bindings must prove reuse. Intervention differences are explicit; unmatched old results remain contextual.
- Accuracy thresholds/speed targets are UNREGISTERED. Separate execution, numerical validity, research evidence and publication status.
- Public files contain aggregates, code, receipts, figures and replay commands. Raw arrays, features, checkpoints and absolute asset paths remain private.
- Only explicit publication paths are staged. Normal main push is directly authorized by the user's active objective. No force/reset/hook bypass.

## Review focus

- A60 PE/scaler leaking into A_stop selection; tests must bind fitting scopes independently.
- Unequal horizon masks changing the decoder's global N_c penalty scale; compare augmented QR/SVD independently.
- Additional frontend initialization perturbing data order or invoking F twice; verify sampler and row counters.
- A DCA timeout/nonconvergence being silently replaced by PCA or retriggered at deployment; explicit solver state and immutable init cache are required.
- A green aggregate checker hiding absent predictions, incomplete pairs, stale source seals, private publication, or invalid whole-path precision/parity.

## Task 1: Read-only start/reuse/budget audit and runner contracts (root)

Files: `research/adapter_improvements_20261018_v1/{contracts,audit,run}.py`, dedicated runner/contract tests; public START_STATE, REUSE_AUDIT, EXPERIMENT_PLAN, RESOURCE_LEDGER, RESUME_QUEUE.

- [ ] Read latest remote, required existing contracts and actual producer call graph; hash protected historical files and actual reusable private assets without invoking old writers.
- [ ] Implement no-argument no-op and --help; modes audit/plan/execute/report/verify/all and documented root/plan/only arguments.
- [ ] Seal hypotheses, job IDs, candidate/control groups, recipes, private bindings, source hashes, tolerance probes and limits before scientific execution. Existing plans must validate and resume without resetting timestamps or ledger.
- [ ] Meaningful tests: no-op imports/side effects, tampered source/artifact rejection, dependency graph, exact --only matching, resource admission/resume accounting, partial-coverage report behavior.
- [ ] Independently inspect tests and actual CLI help/no-op behavior before admitting jobs.

## Task 2: Fixed-PE decoder family (decoder worker)

Files: `decoder.py`, `decoder_jobs.py`, decoder mathematical/mask/scope tests. Read spec§5 and original B03 producer/cache schema.

Interfaces: pure `horizon_basis(H)`, `fit_head(phi, delta, mask, arm, regularization, gamma=1, rms=None)`, `predict_head(phi, state)` returning innovation; immutable serializable state stores RMS/zero features/coefficients and fitting diagnostics. `execute_job(job,ctx)` consumes bound B03 cache inventory and writes only this campaign's private/public outputs.

- [ ] Test normalized linear basis, shared/linear/gain nesting, mask/zero-RMS/empty-channel behavior, global N_c/H regularization, independently computed penalty/objective/KKT and augmented QR/SVD.
- [ ] Reconstruct original shared/Dh A_fit grid(.01,.1,1) and original A60 chosen .01 only, preserving original metrics/receipt identities.
- [ ] Fit all40 registered settings per seed (lambda1e-4..1, gamma1/10/100 for linear/gain), select exact FP64 A_stop MSE then lambda/gamma order, seal before loading V predictions.
- [ ] Fresh D-only A60 refit for all four arms with A60 RMS; gain uses its own selected-lambda A60 shared base. Save prediction/target/mask/origin/channel bindings, FP64 rescore, coefficients, band/channel/origin sufficient statistics and norms.
- [ ] Enforce30m bound including20% verification. Prove new F calls0 on valid caches; no fallback to A60 selection if A_fit missing.
- [ ] Provide an exact inference head adapter for the same FP64 coefficients while frozen F stays FP32, suitable for serialization/reload/cost parity.

## Task 3: Local P and matched training integration (frontend worker)

Files: `frontends.py`, `training_jobs.py`, frontend/factory/gradient/reload tests. Existing shared science files remain unchanged.

Interfaces: `LocalFrontend(channels, split, seed)` with P1152 parameters; context manager `model_factory(...)` wraps the original fit factory across search/refit/reload/cost and always restores it. Training `execute_job(job,ctx)` uses the same registered recipe and independent sampler RNG.

- [ ] Implement exact causal replicate-padding kernels9/25 width32, split trailing mean25 and residual versus X/2 controls; zero only final1x1 kernels with paired identical initial tensors.
- [ ] Test CP0 B0 parity, prefix causality, future canary, impulse/ramp/channel alignment, R+T, two-step gradients into frontend/E/D through frozen F, unchanged F, origin×K logical rows and actual saved-bundle parity.
- [ ] Run complete Jena two-seed split/control pairs. Reuse matched B0/B2/PCA; contextualize mismatched originals.
- [ ] Profile and register breadth-first PeMS32 seed92601 split/control and PCA/DCA pairs independently of Jena V sign; seed92602 is conditional on all first-seed pairs and reserves. Do not silently use CPU-continuation PCA as fresh CUDA control.
- [ ] Save all common search endpoint results separately from selected-A60 V results, actual updates/exposures/K rows, fitting and validation scopes, preparation/search/refit/evaluation times and resumable checkpoints.

## Task 4: Official DCA and LIFT preparation (initialization worker)

Files: `initialization.py`, `lift.py` only if actually needed, DCA/refiner tests, METHOD_REFERENCES and official-source manifest.

Interfaces: deterministic split-bound `fit_dca_initialization(...)` returns status plus saved FP64/FP32 E and diagnostics; factory installs actual FP32 E and computes torch pinv at that dtype. Never optimize DCA in deployment loaders.

- [ ] Read pinned official DCA/LIFT sources, licenses/dependencies and requested primary references, recording source URLs/SHA and exact differences from paper reproduction.
- [ ] Preserve official DCA objective/covariance path while exposing L-BFGS-B maxiter200/maxfun1000/120s and optimizer status; fresh-leaf gradient wrapper prevents cached-gradient accumulation.
- [ ] Test Gaussian PI analytic case, finite-difference/repeat-gradient parity, SPD/rectification/sign checks, boundary canary, rankK and Procrustes invariance; install E32 then matching pinv/D zero lags.
- [ ] Fit A_fit and A60 separately with exact PCA baseline statistics/center/fill, T5/n_init1/defaults, deterministic scope cache shared across seeds. Nonconvergence blocks that arm explicitly.
- [ ] Run Jena B2 DCA versus matched PCA2seeds and registered PeMS32 minimum pairs through Task3 factory, preserving all initialization costs.
- [ ] Verify official LIFT external y_hat branch, full original-channel context path, no additional F rows, actual recipe/complex gradients and parameter bytes. Execute only if mandatory families and reserves are secured; pre-register and execute matching simple output-gain control if LIFT is admitted. Otherwise record precise prepared/unrun dependencies in RESUME_QUEUE.

## Task 5: Multi-K audit, whole-path costs and independent result verification (root + fresh reviewers)

Files: `costs.py`, `multik_audit.py`, `verification.py` as needed; focused tests plus relevant original regression suite.

- [ ] Audit actual existing multi-K parent/stage/updates/K/exposures/F rows/completion/checkpoint bytes/switch-time evidence; absent logs remain UNKNOWN. Verify lag-major prefix rule and separate shared/independent/posthoc/nested-PCA evidence.
- [ ] Measure actual physical-input→fill→normalize→P/E/F/D/refiner→inverse-scale→CPU path on CPU BLAS2 and GPU B1/B8,10warmup+20timed, median/IQR/p95, cold/warm and memory scopes. Isolate cost workers from heavy competing work.
- [ ] Reload the exact scored bundle and compare saved predictions/full-V and route output before cost_valid. Preserve native FP32 F with separately recorded FP64 head/cast branches.
- [ ] Independent verifier reloads actual prediction bindings and reconstructs channel/horizon/seed macro metrics and band sums; verifies matching signatures, all source/cache/selection seals, counts and complete paired coverage.
- [ ] Review numerical modules and integrated runner independently; resolve material findings without altering a sealed scientific identity in place.

## Task 6: Analysis, tables, figures, reproducibility and normal main publication (root)

Files: `reporting.py`, all spec§12 public files, figures F1–F5 PNG/SVG/source CSV, four PPT tables, root README link and existing daily history.

- [ ] Build RUNS/COMPARISONS/COST/CHANNEL_HORIZON_STATS/SAME_UPDATE/MULTIK schemas with all required IDs/precision/scopes/unknowns. Include Raw/Ridge/G-only/LoRA/B0/B2 and contextual native alternatives.
- [ ] Generate honest per-seed P/E/D and horizon conclusions, registered-target status and Pareto facets; preserve negative results and UNKNOWN training/cold fields. Correct misleading old Ridge dtype labels only in this new derivative package.
- [ ] Report partial groups/empty data without fabricating charts; include execution/reuse counts and exact RESUME_QUEUE commands/dependencies/estimates.
- [ ] Reproduce quantitative figures and tables into a separate temporary output root using public sufficient statistics, checking data/labels and actual rendered PNGs. Optional raw curves are private replay only.
- [ ] Verify old sealed package bytes unchanged, final tested scientific source seal, public manifests, explicit allowlist/type/size/privacy checks, actual staged delta and full index blobs.
- [ ] Fetch latest main, preserve other changes via normal integration, commit and normal push, then verify remote SHA and required report/figure paths. Final response follows spec§15 and accurately calls remaining work PARTIAL.
