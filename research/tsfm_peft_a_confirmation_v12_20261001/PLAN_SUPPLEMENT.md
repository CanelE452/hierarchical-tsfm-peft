# V12 prospective confirmation supplement — 2026-10-01

The user delegated the missing data-contract decision and continued execution: “너가 직접 판단해서 쭉 진행하게 해줘”. AUTHORIZATION_SUPPLEMENT.txt records this prospective authorization. The missing earlier PLAN is not retroactively invented. Preserve original approval, request, protocol, old-data results and publication receipts.

## Objective and fixed scope

Confirm A = F_phi(X U0) U0^T + q*(X), using new TRAIN PCA and a newly fitted then frozen LEVEL residual predictor. Establish common-checkpoint Q contribution and A's accuracy/memory/throughput position against F0, full-channel point-MSE LoRA, native-quantile LoRA and direct NLinear. No guaranteed superiority, pretraining causal claim or universal generalization.

Choose at most two already-local public units using metadata, documented prior use and TRAIN-only eligibility. Prefer a new building group and an unused weather period. Never use forecast performance, TEST statistics, PCA explained variance or correlations for selection. Freeze exact sources/hashes, columns, periods, bounded exposure search and exclusions in confirmation_protocol.json before any fit. A new period is temporal holdout; a new building group is group confirmation. Unknown backbone overlap is reported. Download0.

Final prospective choice: one unit, BDG2 Peacock Education. Local Jena2018–2024 periods have documented prior use, so no fresh weather unit is claimed or run. New Office groups were exhausted/too small. Peacock Education has17 metadata candidates and no performance-use trace in the bounded record search. Keep sorted TRAIN-eligible channels (observation fraction≥0.95 and nonconstant), at most32; require at least8 for this group contract. If ineligible, report unavailability rather than performance-screening another site. Same source-naive hourly2016 precontext/TRAIN/VAL/TEST-A/B dates as the earlier BDG2 contract. Exact channel list/C/K is frozen after TRAIN-only eligibility. This is a new use-type/building-group confirmation in the same electricity domain, not new-domain generalization.

## Adaptation and selection

L512/H48 observations, K=ceil(C/4), TRAIN-only standardization/input replacement/PCA/adaptation. Preserve observed zeros and pre-imputation target masks. Primary observed channel-macro standardized MSE, secondary MAE/bias. No post-TEST selection of channels, periods, models, metrics or seeds.

Per unit: LEVEL LR1e-3 ×2 seeds; A/FULL_MSE/NATIVE/DIRECT each LR{1e-4,1e-3} ×2 seeds. Total18/unit,36 for two units. Seeds92601/92602. LEVEL: fixed PCA and shared bias-free G512→32→48 with zero output layer. A: selected same-seed LEVEL E/D/G fixed; zero-update internal q/v LoRA r8/alpha16/dropout0/bias none. Pin Chronos-Bolt-small772f3d25d38aec6d914c8949dab4462e2d46f5d8. One TSFM per model; no old-site fitted weights. LoRA families share same-seed initial adapter tensors; LR variants share initial state and origin order. DIRECT: shared affine512→48 with last subtraction/restoration, no TSFM/E/D/G.

AdamW wd0, clip1, batch4,512 phase-balanced TRAIN origins/epoch, max120, strict VAL nonimprovement6, plateau factor0.5/patience2/relative threshold1e-4. Step0 eligible. Minimum full VAL MSE per fit, exact tie earliest; family LR selected by two-seed best-VAL mean, tie1e-4. Native trains quantile loss but selects point-MSE. No automatic240 extension or additional LR/rank/K. Fit order: all parents, then seed/LR/unit/family[a,full_mse,native,direct]. Store initial/best/last, optimizer/scheduler/RNG, fixed TRAIN probe, actual updates/freeze checks, curve/result/receipt. Failed retries consume attempts.

## Q and joint TEST boundary

At each A checkpoint obtain MAIN_ONLY, MAIN_LAST and A_FULL from one main forward. FULL−MAIN is total Q contribution; FULL−LAST is learned temporal correction beyond persistence. E1 limits the claim to common-checkpoint contribution, adding no independent NO_Q fits and making no whole-policy equivalence claim. Record new TRAIN/VAL masks and relative-scheduler caveat before TEST. The old24 paired-backward allowance is not renewed.

All units' required fits/LR/checkpoints, hashes and planned roles must be jointly sealed before confirmation_test_exposure.json. TEST loader verifies seal; subsequent fitting prohibited. Complete all units/baselines regardless of early results. Later origins may use then-observed past, never online target fitting.

Pool period channel SSE/SAE/count, average channels then seed losses. No prediction ensemble or cross-unit raw score average. Save period/seed/channel/bias. P/Q diagnostics use common complete-target-vector subset and coverage separately from masked macro MSE. Conditional paired moving blocks: protocol block lengths,2000 resamples, no boundary crossing; intervals do not cover all selection uncertainty.

## Costs and budgets

Per unit11 instances: A2+LEVEL2+F0+FULL_MSE2+NATIVE2+DIRECT2. Uncompressed B1{C,K},B4{4C,4K,K}; A/LEVEL B1{K},B4{4K,K}; DIRECT B1/B4 GPU and separate CPU. Two units yield270 GPU rows and24 CPU rows, including F0 pre/post sentinels over3 blocks. First24 chronological VAL origins,FP32,TF32off,CPUthreads4,10 warmups and10 full24-origin passes per block. Save all repetitions/order/drift. Merge/full/chunk parity: atol1e-5/rtol1e-4; same-path1e-6/rtol0; Gram maxabs1e-5. No tolerance expansion to force a pass.

Measured time: standardized CPU input→transfer→online forward/chunk assembly→full original-channel H48 CPU output. Load/disk/hash/log/score outside latency, applicable time still charged to job. Separate B1 latency,B4 throughput,allocated/reserved/resident/process memory,deployed bytes and adaptation history. No old/new timing ratios,cached predictions,compile,quantization,new servers/global changes.

Original shared ledger caps remain: realattempt60,GPU28800s,CPUanalysis7200s,CPUcheck1800s,syntheticoptimizer6 sessions≤10 totalupdates/session,newstorage10GiB,download0. Consumed old work counts. Actual allocation36 base+4 technical recoveries; unused conditional headroom is not search permission. Historical rates suggest roughly2h fitting plus about1h evaluation/cost with uncertainty, not a promise. Re-estimate after first normal fits; never raise caps. Reserve before execution; failures/restarts count; GPU parent and nested fit not double-added. One neural fit at a time; no other-process termination. Conflictwait≤30min/incident,≤60min total.

With the one-unit choice, actual basic fits are18 plus at most4 technical recoveries (22 allocation within60 original hard cap); omitted second-unit headroom is not transferable to tuning. Planned cost is135 GPU rows and12 CPU rows,11 model instances and15 scoring roles. GPU timings contain32,400 total original-channel origins plus shape warmups; CPU contains2,880. Earlier two-unit counts above explain the original allocation, not additional authorized work. Approximate working allowance:0.5h restoration/checks,2h fitting/VAL,0.5h evaluation,1h cost, with the unspent original cap retained for genuine technical contingencies, not obligatory consumption.

## Autonomy and completion

Proceed through preparation, implementation checks, fixed fits, joint selection/evaluation, cost, review, figures and decision without repeated approval. Same-scope API/path/serialization/scoring errors can be repaired with original evidence/impact/diff preserved. Material method/data/budget changes remain outside scope. After TEST, reproducible scoring/restoration defects can be corrected across all affected comparisons; do not silently reselect or claim untouched confirmation.

Finish when fixed comparisons are interpreted or genuine access/budget/technical limits are reached. Positive/negative/mixed are valid conclusions. Restrict or retain A based on external alternatives and Q contrasts; no rescue through new modules/data. Update STATUS, data/run/selection/evaluation/cost, CLAIM_EVIDENCE,METHOD_UPDATE,TOPIC_DECISION and final checks. Ordinary related-files-only origin/main commits, verify live SHA. Prior/unrelated artifacts preserved; raw/weights/prediction arrays stay local. Self-checks are not independent replication.
