# V9 status

Approved via user instruction to treat the two supplied documents as /plan and /goal. Base main/live origin is61fbbff0959294eb996bf6ae37a39dab7df2bfb3.

Current: all eight real fits, both VAL/checkpoint/alpha selections, both fixed TEST evaluations, saved-prediction analysis, figures and conditional Jena deployment measurements are complete. Both datasets are exposed; no further fitting or selection is allowed. Numerical model/train/runtime/protocol code remained fixed across fits. Robin cost was explicitly skipped under the approved conditional rule. The final verification state is recorded in final_checks.json; live origin publication is recorded in publication_receipt.json. Completion of computation does not imply publication until that receipt exists.

Limits: base8/max12 real attempts, GPU7200seconds, synthetic6sessions at most10updates each, CPUcheck900seconds, storage5GiB, downloads0. Exact jobs/PIDs/usage are in ledger.json. Root alone launches computational jobs.

Uncertainty: Q preservation does not ensure lower future error. Parent and alpha selection reuse VAL; Robin/Jena are exposed development data. Both datasets and both heads are required independent of interim performance. v1–v8 and unrelated dirty paths are preserved.

Checks: `prepare_reuse01` passed parent/data/prediction hashes and copied the data contracts byte-identically. Preservation covers 1,050 prior tracked research files. `synthetic_models01` failed before updates because its comparison reference did not use the same frozen inference conditions; the original and correction are in `checker_correction01.json`. `synthetic_models02` passed all ten synthetic updates, initial equality, frozen parent, P/Q, checkpoint and restart checks. No tolerance or model/training rule changed. Both sessions remain counted (2/6).

Accuracy: Robin STAGED_P selects step0 and alpha0 and equals its parent (MSE0.275196). Jena STAGED_P MSE0.279884 is13.71% below parent and2.24% below matched RAW, but remains16.33% above LoRA and2.15% above DIRECT_NLINEAR. It does not improve on v8 JOINT_PQ by point estimate. Parent fallback is not new-head efficacy, and the two exposed datasets do not establish broad generality.

Cost repair: the first Jena full-VAL replay failed the fixed1e-6 absolute tolerance for P seed92602 before timing began. A reserved VAL-only diagnostic found normal no_grad restore/forward bitwise equal to the saved output, whereas inference_mode over restore and forward differed by1.1920929e-6 at10 of373968 elements. The common cost context was aligned to no_grad for every method. All29 configurations then passed parity;99 timing rows completed. The original failure/source/partial output and correction are retained; no tolerance, numerical model, fit, selection or TEST result changed.

Cost finding: Jena STAGED_P B4 throughput325.52 origins/s and allocated217.97MiB trade accuracy against full LoRA224.80/281.44. M24 LoRA reaches216.89MiB at88.45origins/s; DIRECT_NLINEAR is much cheaper/faster and has lower point MSE. V9 is retained as a conditional secondary correction/ablation, not a default or main-method replacement. See TOPIC_DECISION.md and METHOD_UPDATE.md.

Usage at computational close: real8/12attempt, parent refits0, synthetic2/6sessions (0+10updates), GPU1131.394/7200seconds including failed parity and diagnostic. Before final document verification CPU checks used20.214/900seconds and storage was53,180,339bytes/5GiB; these latter counters continue through verification/publication and the closed ledger/publication receipt gives final values. No downloads, parent/raw duplication or reserve-fit spending occurred.

Delivery sequence: saved-artifact verification, related-files-only ordinary commit/origin main push, then live SHA confirmation and publication receipt. No further scientific execution is needed. Read actual PID/heartbeat and receipts before resumption; never duplicate a running job or reopen TEST. Remaining research decision: whether to continue the paper with the building-electricity contribution explicitly limited to conditional accuracy–cost tradeoffs; general accuracy recovery remains unsupported.
