> 공개 사본: 개인 경로·채팅 식별자와 링크를 정리했습니다. 원본 및 사본 해시는 감사 공개 폴더의 `publication_manifest.json`에 분리 기록합니다. 아래 상태·검산은 원래 감사/실행 시점의 기록이며, 이번 게시 검증이 아닙니다.

# Fixed Chronos-2 controls for LEVEL

Execution authority: the user's current request adopts REQUEST.txt and authorizes this finite campaign. This plan records the contract, rather than requesting another execution approval. No new fitting, optimizer updates, model selection, PCA statistics, data, periods, channels, or calibrator are authorized. Originals and unrelated changes remain read-only. Local deliverables only; this request does not explicitly authorize a new commit/push.

## Question and evidence scope

Does basic LEVEL remain a useful accuracy / inference-memory / throughput choice when native multivariate zero-shot Chronos-2 and Small are available under the same observed information? This is a practical comparison of different pretrained systems, not a causal attribution to compression or channel sharing. All three evaluation datasets were already exposed. Zero-shot means no adaptation to these tasks in this campaign; pretrained-data nonoverlap remains unverified.

## Confirmed local bindings

- Repository origin CanelE452/hierarchical-tsfm-peft, main HEAD 2154e7f2b0e9878164e77536b8deaa9f372ac760. Existing dirty/untracked paths are recorded in protected_before.json. The initial protected file hashes cover research, presentation, source, and existing history.
- input_contract.json binds original arrays, source contracts, masks, columns, TRAIN statistics, bases, timestamps, and origins. Robin 17/5 hourly, Peacock Education 13/4 hourly, Jena 21/6 at ten-minute intervals; L512/H48. TEST-A/B counts 40/40, 40/40, 365/365. VAL counts 30, 30, 371. No normalization refit. Standardized missing input values are original zero fills; original finite target masks are preserved, including observed zeros.
- reuse_manifest.json binds every actual checkpoint, restore source, stored prediction, array key, selection, and expected source score. Basic LEVEL uses v6 Robin, v7 Jena, v12 Peacock. MSE-LoRA uses selected v11 full_lora_mse Robin/Jena and v12 full_mse Peacock, never native-quantile LoRA. DIRECT uses original direct NLinear checkpoints. b is the fixed compression-only path. Peacock b's stored main_only_prediction is from an epoch0/steps0 A checkpoint with all LoRA B matrices exactly zero; it is counted once and not used as basic LEVEL. Peacock b/F0 have no matching stored VAL arrays; original forward parity is checked without claiming stored-VAL replay for them.
- model_manifest.json pins amazon/chronos-2 at 29ec3766d36d6f73f0696f85560a422f50e8498c (existing cache) and autogluon/chronos-2-small at ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a (111,750,017 downloaded bytes). Config/weight SHA256s and installed libraries are included. Installed chronos-forecasting 2.3.2 / Torch 2.10.0+cu128 / PEFT 0.21.0 are used without installation.

## Inference and controls

C2_MV / C2_SMALL_MV: CPU FP32 [B,512,C] becomes [B,C,512]; each forecast origin is a separate task with C targets. cross_learning=False, context_length512, prediction_length48, native row batch_size=B*C. Actual dataset group IDs and group-attention masks must be block diagonal by origin. No grouping across origins, no known future values or extra covariates. No native-group channel chunking.

C2_UNI / C2_SMALL_UNI: identical weights with [B*C,1,512] singleton tasks, so channels cannot share information. All four methods are fixed before evaluating TEST. Preserve native instance scaling including arcsinh and its inverse. Extract the actual 0.5 quantile: big index10 of21, Small index6 of13. Output remains in the existing TRAIN-standardized original-channel coordinates; do not apply another inverse transform.

Existing baselines restore original modules. LoRA deployment is a separate merged copy, verified against unmerged output. Bolt full/chunk paths retain every row, all512 context values, all48 outputs, and channel order. LEVEL forwards its complete encoder/backbone/decoder/residual path online.

## Preflight and seal

Use only TRAIN/VAL for preflight: axes, literal median, no future target, actual group IDs/masks, solo versus bundled origin, reordered origins, perturb other-origin history, UNI cross-channel invariance, MV cross-channel response (descriptive, not a required effect), finite frozen/eval/no-grad outputs, unchanged state, reload parity, original saved VAL replay when available, LoRA merge parity, and Bolt full/chunk parity. Same-path atol1e-6/rtol0; batch/merge atol1e-5/rtol1e-4. Do not enlarge tolerance to obtain a pass. Save actual maximum differences and violation counts.

Measure a small VAL sample to estimate all fixed evaluation and cost work before TEST. If completion exceeds the remaining cap, report the necessary scope/budget decision without silently deleting methods or datasets. After preflight passes, seal model/data/reuse hashes, code, grouping, point statistic, precision, metrics, cost grid, budget, and prior exposure. Original scientific definitions stay fixed after TEST. A concrete implementation/serialization bug may be repaired once per cause with the original failure and affected results retained; changed scientific contracts require a separate decision.

## Fixed evaluation

Four new methods × three datasets =12 units, 3,560 original-channel origin forecasts in total. Default task batch4; only origins may be reduced on a concrete memory failure. Reuse the 24 verified old stored predictions. Whole A/B: sum each channel's SSE/absolute/signed error and observed count across periods before the channel mean. Learned references report mean of the two seed losses, not averaged predictions. Report MSE, MAE, bias, period, seed, channel, origin and observed counts; never average different datasets' raw errors.

Paired block intervals: Robin7 / Peacock7 / Jena42 origins, 2,000 draws, seed9262026 and original dataset salt; resample each period separately and keep channels/methods paired. They are conditional on fixed models/seeds and exposed evaluation, not complete selection uncertainty or equivalence/noninferiority tests. Main comparisons LEVEL vs each MV, each MV vs UNI of the same checkpoint, and each MV vs F0/MSE-LoRA/DIRECT. No arbitrary gain threshold.

## Same-campaign cost

FP32, TF32off, CPU4 threads, same RTX4070. First24 chronological VAL origins. Each row:10 warmup calls and10 complete passes over24 origins,3 balanced order blocks, F0 before/after sentinels. CPU standardized input → grouping/transpose/H2D → full online forward → all original-channel H48 CPU output is inside timing. Model load/disk/scoring/hash/ledger/telemetry are outside. Keep per-pass distributions and unfavorable blocks. Report parameters/buffer bytes, resident allocation, peak allocated/reserved and available process memory separately. Newly zero-shot models have no task fitting; pretrained costs are not zero.

Per dataset GPU grid: LEVEL2seed×B1/B4=4; F0 one×(B1 fullC/K and B4 full4C/4K/K)=5; MSE-LoRA2×same=10; DIRECT2×B1/B4=4; two MV models×B1/B4=4. 27 conditions×3 datasets×3 blocks=243 rows, plus36 F0 sentinel rows=279 GPU rows. This gives42,210 wrapper calls (66,960 timed and7,380 warmup origin units). C2 has36 rows/5,760 pipeline calls. UNI is an accuracy auxiliary control, not an additional full cost grid. DIRECT CPU contributes36 rows/5,760 wrapper calls separately. Uncompressed channel chunks are never offered to native multivariate C2 groups. No compile, quantization, result cache, or altered precision.

## Hard operational limits and accounting

GPU one job at a time /10,800seconds total; CPU-only3,600seconds; download4GiB; new storage8GiB; fitting0. Each job is reserved in ledger.json before execution; failed/retry work counts. Parent GPU jobs and their subcalls are not counted twice. One retry after fixing each concrete technical cause. No other processes are terminated, no driver/power/global environment/server/service changes. Current external GPU evaluation must finish or bounded waiting must be reported before measurement. Budgets are operational caps, not completion predictions or academic acceptance criteria.

## Interpretation and deliverables

If MV or Small is more accurate and has lower relevant memory/higher throughput in this session, reduce LEVEL's practical claim. If LEVEL retains only a cost advantage, show the accuracy penalty and measured savings without inventing an allowable loss. If LEVEL wins accuracy on a dataset, limit it to that task and these fixed checkpoints. Same-checkpoint MV/UNI differences describe sharing under this inference contract; they are not a universal guarantee. Technical/budget incompleteness remains distinct from a method loss.

Deliver input/reuse/model manifests; source/seal/preflight; cached full predictions with public receipts; full evaluation CSV/JSON and paired intervals; cost raw repeats/grid/summaries; final verification and budget; Korean RESULT_REPORT.md; English presentation tables/captions; separate SLIDE_PATCHES.md with revised claims. Preserve original PPT/PDF. Own verification is not independent replication. Finish locally and report actual files and completion limits.

Official evidence: https://huggingface.co/amazon/chronos-2 ; https://huggingface.co/autogluon/chronos-2-small ; https://github.com/amazon-science/chronos-forecasting/tree/main/src/chronos/chronos2 . Actual installed source and explicit pinned revisions, rather than mutable documentation alone, bind the executed API. PyTorch benchmark/CUDA guidance informs synchronized timing; raw observations determine the conclusion.
