# v11 STATUS

Phase: approved v11 execution complete. All scheduled fitting, evaluation, GPU/CPU cost, reports and verification passed; artifact commit published and verified on origin/main.
Base/live main last verified: efc029f8aae619b499071967d6b4764cb55fb802.

- Neural fits: 40/40 complete. Coefficient fits: 12/12 complete. Technical fitting retries: 0/8. Total real attempts: 52/60.
- Synthetic optimizer checking: one session, six updates. Actual Bolt, masked aggregation, selected restart/restore and merge/export/chunk checks passed. Original failed checks and inference retry remain in REPAIRS.md and ledger.json.
- All 57 scheduled development prediction instances are evaluated. All S/U/LR/checkpoint/Gamma choices precede the recorded joint exposure. No fitting or reselection followed evaluation.
- GPU cost: 621/621 rows complete, one campaign session. CPU direct-NLinear cost: 36/36 complete. All measured models keep full online H48 original-channel outputs.
- report_values.json, report_family_comparison.csv/json, adaptation_cost.csv/json and figures exist. final_checks.json passed own saved-array/provenance/contract checks; it is not independent reproduction.
- No active research job remains. Check actual PID and ledger before resuming; completed labels must not be rerun.
- GPU charged time: 12,797.223 seconds (213.29 minutes) / 28,800 seconds. final_budget.json records final CPU/storage totals and the disclosed 0.778-second pre-reservation render import failure.

Decision: retain A conditionally around Jena's internal-adaptation gain and observed accuracy-memory-throughput choice. B learned U has a real Jena internal effect, including an increment over fixed U plus Gamma, but simple mixture explains part of the gain. Full MSE LoRA remains lowest combined MSE on all three datasets, native LoRA lowest MAE. Robin timing/accuracy variability and Hog MAE losses remain. See TOPIC_DECISION.md.

Independent confirmation, universal accuracy recovery and broad generality are not established by this exposed-development design. No new dataset, K, rank, LoRA variant or combined A+B method was added. The next proposed decision is whether to add separately trained compressed LoRA without Q in a later authorized study; it has not been executed here.

Publication: artifact commit deac143a79993c7a2d0b9202953ace0d8c8b2cd1 was pushed and verified live on origin/main. publication_receipt.json records that verification; an ordinary metadata-only follow-up publishes this receipt and completed status. Only v11, its related result index and the 2026-09-30 history entry were published. Existing v1-v10, original failures/seals and unrelated dirty paths are preserved. No raw, checkpoint or prediction arrays are included.
