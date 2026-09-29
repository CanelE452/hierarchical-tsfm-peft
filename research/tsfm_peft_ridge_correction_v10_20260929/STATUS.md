# Accuracy complete; cost grid stopped within its approved ceiling

Starting main:4c98b9228a9751542b2ff7bd61521c247911f946, live origin/main matched. Existing unrelated untracked files and completed v1-v9 are preserved. PLAN.md is the exact approved NEXT_REPAIR.md; APPROVAL.txt records the user's approval and context.

Current phase: numerical work closed; final artifact verification passed; authorized publication pending. All16 CPU ridge fits and both datasets' evaluations are complete. Both datasets' finite mean-parent VAL selections were sealed before evaluation. All8 selected online VAL restorations passed; no fit or model-selection change followed exposure. These are previously exposed development datasets, not independent confirmation.

Budget ceilings: basic16 CPU fits, maximum18 including technical recoveries; GPU1800s including optional cost600s; CPU fitting/analysis1800s; checks300s; storage1GiB; downloads0; synthetic optimizer0. Used16 fits with no fit recovery or optimizer update. Closed totals in `budget_final.json`: GPU880.003s (14.667min), CPU fitting/analysis87.980s, checks89.425s, storage256,985,013bytes at verification. No active jobs remain in the ledger or live Python process check. This does not reuse the closed v9 budget or count the earlier44.31-second no-fit diagnostic in this round.

Scientific decision: reject nonzero Robin ridge (P MSE+1.825% versus parent; both seeds and periods worsen). Jena P improves parent13.907% and matched RAW2.025%, but its0.2265% improvement over v9 is uncertain and LoRA MSE loss16.069% remains. The accuracy/generalization objective is not solved. Stop further tuning of this shared-linear correction family in this approved round; see TOPIC_DECISION.md.

Cost: Jena only; all11 parity checks passed,68/78 planned rows completed. Own-job guard stopped it at586.663seconds within the600-second subset ceiling. Preserve partial results; final throughput/latency/memory ranking is unconfirmed. Accuracy results remain valid. The first Jena evaluation writer failure and reuse-only repair are preserved in REPAIRS.md and ledger.json.

Verification: `final_checks.json` is PASS_ARTIFACTS_WITH_DOCUMENTED_LIMITATIONS. Checks include all16 fit specifications/weights, exact VAL selection, both selections preceding exposure, stored prediction hashes and reported arithmetic, budget closure, no public raw/weights/arrays, and preservation of1178 prior research files. This is execution-team verification, not independent replication or scientific success.

Publication: pending normal origin/main push. Only diagnostic/v10 artifacts, related README index and2026-09-29 history are included; raw, weights, prediction arrays and unrelated dirty paths are excluded.
