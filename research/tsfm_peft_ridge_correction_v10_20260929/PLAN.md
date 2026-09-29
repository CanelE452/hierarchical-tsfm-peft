# Proposed next repair — not yet approved or executed

## Decision supported by the new diagnostic

The saved Robin best-trained P heads reduce MSE on the identical 96-origin TRAIN probe for both parent seeds, but increase whole-VAL MSE. The final checkpoints reduce TRAIN-probe MSE further and increase VAL MSE further. Thus a disconnected optimizer is not the explanation, and simply extending these runs is not justified. This supports non-transfer of this learned correction; it does not prove a specific covariate shift, optimal training, or an irreducible accuracy limit. Diagnostic replay does not resolve the accuracy/generalization goal.

## One intervention and one matched control

Keep the selected frozen LEVEL parent f*, its TRAIN PCA P=UU^T, all normalization, C/K, and data contracts. Let deltaX=X-repeat(last(X),512). Replace only the newly added v9 correction head with a channel-shared, bias-free temporal map W in R^(48 x 512), fitted by ridge to TRAIN parent errors:

    RIDGE_P(X)   = f*(X) + W_P deltaX P
    RIDGE_RAW(X) = f*(X) + W_R deltaX

Each new head has 24,576 fitted coefficients; no bias, activation, channel-specific weights, extra backbone, or new future information. P/RAW have identical fitting and selection opportunity. W=0 is the exact parent reference, not a positive result. The P arm still cannot repair Q errors. The RAW arm can alter both spaces and is necessary to test whether preserving Q is itself too restrictive.

Fit the actual observed-target channel-macro loss, plus ridge:

    J(W) = (1/C) sum_c [sum_(o,h) m_ohc (f*_ohc+(W Z_o)_hc-y_ohc)^2 / N_c]
           + lambda ||W||_F^2 / H
    N_c = sum_(o,h) m_ohc; Z=deltaX P or deltaX; H=48.

Use all eligible TRAIN origins, not new validation/test targets. Solve the 48 weighted linear systems in CPU float64, with observations absent from the mask excluded. Cache one parent forecast per TRAIN/VAL origin and seed for fitting; deployment always runs the online parent and head. Do not reuse v1 `fit_masked_shared` unchanged: it includes bias and a different horizon-normalized objective. Retain sufficient-statistic/normal-equation residual checks, masked objective checks, saved-weight replay, finite coefficients, and P-output Q preservation. No synthetic optimizer session is necessary.

This changes temporal rank, regularization, and fitting together. It is a practical replacement/control, not a pure causal test of SGD or rank32. The observed TRAIN/VAL divergence motivates regularization, not a conclusion that more rank will improve accuracy. Ridge uses directional shrinkage, unlike the already-tested scalar alpha grid.

Ridge is established methodology, not a novelty claim. Official reference read: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html (objective/regularization). The previous v6 penalty values were 0.001 and 0.1; reuse these two numerical candidates with the explicitly different loss normalization above. No additional penalty/rank/LR search. The v5 FULL reserve was not executed; that historical non-execution is not negative evidence for this new fit.

## Fixed comparison and finite budget proposal

Include Robin and Jena at the outset, regardless of which is favorable. Both are exposed development data. Reuse the exact v9 manifests/parent seeds 92601/92602, TRAIN/VAL and evaluation origins/masks, and existing BASE_LEVEL/v9/v8/F0/LoRA/DIRECT predictions. No new data/site or independent-confirmation claim.

    Robin: P/RAW x 2 parent seeds x 2 penalties = 8 CPU fits
    Jena:  P/RAW x 2 parent seeds x 2 penalties = 8 CPU fits
    Technical recovery only                    = at most 2 attempts
    Total new real-data fits                   = at most 18 attempts
    New neural-network training                = 0

The two seeds identify different frozen learned parents, not stochastic ridge repetitions. Each data/parent/input/penalty fit is one attempt, including failed/interrupted/restarted fits; the 48 internal horizon solves are counted separately as solve work. Reserve each attempt before solving. No use of v9's closed ledger or unused allowance.

For each dataset and arm, select one penalty using the two-parent mean whole-VAL MSE, exact ties first 0.001 then 0.1. Also report the parent-reference VAL. If the selected nonzero arm is worse than the parent on VAL, any deploy-choice fallback must be labeled parent, while the nonzero ridge result is still reported. No post hoc alpha, channel deletion, period selection, or candidate replacement. Freeze both datasets' selections before evaluation; evaluate all planned arms, including unfavorable ones.

Report masked macro MSE/MAE, each seed and fixed period, paired channel differences, and remaining absolute/relative gaps to F0/LoRA/DIRECT. Aggregate periods by channel SSE/count before channel mean; average seed losses, not predictions. Reuse the v9 period-respecting bootstrap settings (Robin7/Jena42 origins, 2,000 draws), with the existing limitation that it does not include method-selection uncertainty. No new percent-improvement threshold or cross-dataset average.

[Proposed operating ceilings, not measured runtime estimates]: one local GPU, one job at a time; GPU job occupancy 1,800 seconds including inference/restoration/failure/conditional cost; CPU fit/analysis 1,800 seconds; CPU correctness checks 300 seconds; new storage 1 GiB; downloads0; synthetic optimizer sessions0. This proposal is separate from today's completed 44.31-second no-fit diagnostic. Parent forecast generation may dominate; process origins in chunks and do not store duplicate context tensors. Use actual early throughput to determine whether the remaining work fits the ceiling; do not raise it automatically or omit an unfavorable dataset to fit it.

Only if a nonzero correction retains interpretable value, measure the selected P/RAW and parent/F0/mergedLoRA/DIRECT under the existing same-session v9 VAL24/B1-B4 timing contract, with at most600 GPU seconds reserved inside the1,800 total. Preserve block variability and report online whole-model bytes/allocated/reserved. No benchmarking requirement if there is no useful new accuracy result; no speed claim without a valid measurement.

## Decisions and autonomy after approval

- Both datasets improve against parents and a useful alternative gap narrows: keep the correction as a development candidate, report matched RAW and remaining costs; no claim of broad generalization.
- Only RAW helps: stop describing strict P preservation as universally beneficial; retain the measured control result.
- Improvement only in TRAIN or one dataset: report restricted transfer and all counterexamples, without adding candidates.
- No material supported advantage, or simple alternatives remain preferable: reject this repair for the current scope. Do not search more penalties until a positive result appears.
- Reproducible code/solve/restoration errors may be repaired with original failure, hash and diff retained, within2 recovery attempts. Data-selection changes, additional methods, budget increases, or new datasets require a new decision.

After one approval: implement the separate runner, reserve jobs, fit, freeze VAL selections, compare all planned outputs, perform conditional cost, verify, and publish only related small artifacts to origin/main using ordinary commits. Preserve prior v1-v9 and unrelated dirty files. No raw/checkpoint/prediction-array publication, force push, new branch/repo, installations, global environment changes or termination of other projects. This is a proposal only; no such fits or publication have occurred.

The approval boundary is explicit in the original v9 APPROVAL.txt: `TEST 뒤 fit·계수·채널·기간·주지표를 변경하지 않는다.` and `새 PEFT·새 데이터·큰 gate·loss·K/rank 탐색·최종 다중데이터 확증으로 전환하지 않는다.` This new repair must not be smuggled into the completed v9 experiment.
