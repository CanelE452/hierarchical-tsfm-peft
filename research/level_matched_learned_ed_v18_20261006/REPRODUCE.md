# Reproduction and stopped execution

Current execution status is **BLOCKED_STOP_RESOURCE**. The full matched learned E/D accuracy/cost campaign has not run. Preflight performed zero optimizer updates and zero TEST predictions. Its saved canonical VAL predictions are step0 function checks, not trained-method improvements.

The adopted REQUEST.txt sets a cumulative GPU cap of 10,800 seconds and instructs stopping when the full scope is predicted to exceed it. The fixed estimator projected 11,421.533 seconds from recorded zero-update steps, whole canonical VAL measurements, all twelve fits reaching their 120-epoch maximum, and the later TEST/cost scope. It does not assume early stopping. It is a conservative planning projection, not observed convergence time. Actual preflight GPU job wall was 21.848 seconds.

Use the saved resource_stop.json, preflight.json, preflight_attempt01.json, preflight_initial_models.json and ledger.json to inspect the completed evidence. Official references and source revisions are in source_binding.json and PRIOR_ART.md. Parent artifact hashes and original channel/origin/mask contracts are in parent_reuse_manifest.json and data_contract_*.json.

Python sources are preserved for review. model_v18.py, train_v18.py and preflight_model_v18.py produced the bound preflight; fitting branches in train_v18.py and campaign_v18.py were not executed. Evaluation, paired comparisons, training-path cost, inference cost, and selected-checkpoint mechanism diagnostics require a completed joint TRAIN/VAL selection and remain unexecuted. No selection or performance result is fabricated to satisfy an output filename list.

Large data archives, checkpoints, predictions and the full private protected-workspace hash guard remain in .cache and are excluded from Git. The saved guard verifies 5,708 original files. Local reproduction needs those original cached artifacts and pinned Bolt weights; GitHub publishes source, small receipts, reports and figures.

The minimum continuation preserves the three datasets, K, original data/PCA contracts, two historical learning rates, two seeds, twelve-fit ceiling, six selected TEST predictions and all scientific selection/scoring rules. It first needs an explicit resource-budget amendment. A four-hour cumulative cap is a reviewable example; it has not been granted. Preserve this original failed preflight and its code hashes when recording any approved budget amendment. Reuse its completed parity/gradient evidence; never silently rewrite the old stop as PASS or launch duplicate completed controls.

This campaign is a structural learned E/D control inspired by official Linear E/D source. It is not a full AdaPTS framework reproduction, an independent evaluation, or evidence of publication novelty.

The post-stop [source review](SOURCE_REVIEW_KO.md) preserves the original ten preflight-bound artifacts and corrects only later cost/evaluation code. Its fixture receipts in source_review_checks.json describe isolated CPU checks, with no real backbone load, TEST forecast, fit, or added GPU job. The full completed-experiment verifier remains distinct from the stopped-attempt verifier; it must replay the eventual real selected artifacts before scientific completion can be reported. No source review lifts the existing execution gate.
