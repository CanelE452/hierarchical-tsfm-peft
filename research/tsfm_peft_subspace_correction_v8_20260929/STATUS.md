# V8 status

Approved execution. Base main/live remote: 0c9a08f22a9f01820859aa0e8648fc1601a21444.

Current: all eight fits, both VAL selections, fixed Robin/Jena evaluation, saved-result analysis and the comparison figure are complete. No v8 fit started after either TEST exposure. All fits stopped at strict patience before the 120epoch cap; selected-checkpoint replay error was zero and fixed TRAIN probe losses decreased. This is valid training evidence, not proof of optimal convergence or scientific superiority. No training/GPU process remains active for v8.

Budget: Robin4 + Jena4 = 8/12 real attempts; failed/retried real fits0, unused reserve4 will not be spent. GPU work occupancy537.482797 seconds (8.958047 minutes)/7200 seconds includes training and evaluation; nested fit times are not added again. Synthetic1/6 sessions used exactly10updates. CPU check limit900 seconds; storage5GiB; downloads0. Exact CPU verification time and final storage are recorded in final_checks.json and publication_receipt.json; ledger.json remains the authoritative execution account. Every experiment/check was reserved before execution; only the root coordinator launched jobs.

Decision: no default replacement of prior LEVEL and no claim that accuracy loss or generality is solved. Jena pq_split MSE0.279714 improves on prior LEVEL0.324359 and matched RAW640.291395, leaving a conditional P/Q correction result. Robin pq_split0.279019 does not establish an improvement over prior LEVEL0.275196. LoRA gaps remain on both. Both datasets are exposed development data. See TOPIC_DECISION.md and analysis01/analysis_summary.json.

Cost: explicitly not run, according to cost_decision.json. Jena's positive methodological evidence is retained, but the approved Robin-only cost grid would not change the Robin decision and cannot establish Jena deployment value. No new timing, inference-memory or Pareto claim is made. The cost implementation being present does not imply a completed benchmark.

Completion records: final_checks.json records the actual verification outcome, and publication_receipt.json records the artifact commit and live origin/main observation. Check those records rather than treating this status text as a PASS or push receipt. Remaining decision is whether to pursue a separate, bounded development of the retained-space prediction gap; no further candidate or unused dataset was started in v8.

Known limits: P correction is a hypothesis, not an established cause or fix. Q rank remains32. Robin/Jena are exposed development data; improvements cannot establish universal transfer. v1–v7 and six unrelated untracked baseline paths are preserved. Other Python processes belong to Algorithmic-Trading and are not stopped.

Pre-evaluation correction: contract_correction01.json preserves the original copied v7 cost metadata and per-dataset TEST guard. The guard now blocks any new v8 fit after either dataset's evaluation starts; cost metadata matches the approved narrow Robin grid. Per-run source receipts show three Robin fits using the preserved original hashes, then the fourth Robin and all Jena fits using the corrected hashes. The original first-fit-only description was too narrow and is corrected in contract_correction02.json. All fit specs/model/numeric training logic/data were unchanged and no TEST had been evaluated. No refit was needed.

Verification repairs: CPU checks01/02 failures are preserved. They concerned the reused Robin initialization manifest (v6 rather than v7) and the too-narrow source-history mapping above. These are verifier/provenance repairs, not failed real fits; final_checks.json records the repaired check's actual outcome. CPU checking time includes the failed attempts.
