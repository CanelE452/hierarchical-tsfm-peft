# v11 implementation repairs

## 2026-09-30 — zero-variance native scaling backward

Evidence: checks/actual_bolt_checks01.json, GPU job actual_bolt_checks01, optimizer updates 0. Actual Robin/92601 TRAIN context has finite active U gradient; exactly constant context has finite forecast but nonfinite U gradient. Original failed source is preserved by ledger source_snapshots.

Cause traced to installed Chronos InstanceNorm: sqrt(variance=0) is evaluated before replacing scale=0 with epsilon. Its backward can evaluate zero times infinite derivative. V11 now substitutes a finite sqrt input only on the exactly-zero branch, then restores the same native epsilon. Positive-variance computation and inverse transform remain unchanged. The local instance method is applied to every v11 TSFM family, without changing installed packages or pretrained weights.

Verification planned in actual_bolt_checks02: exact normalization/loc/scale equality against original class for TRAIN, constant, near-constant contexts; full parent/F0 parity; finite U gradients. No tolerance change. First failure remains invalid check evidence, not a negative method result. This repair adds no optimizer fitting attempt; its GPU time remains charged.

Verification: checks/actual_bolt_checks02.json completed all six dataset/seed cases. Native normalization forward values/location/scale exactly equal on TRAIN, zero-constant and near-constant contexts; repaired U gradients finite. Parent, initial F0, merge, export, row-chunk, and Gamma identities pass fixed tolerances. No tolerance or model weight changed.

## 2026-09-30 — cost observer outside timed intervals

Read-only review found the general Job heartbeat thread could write the ledger and inspect storage during a timed forward pass. No v11 cost measurement had run. The cost entry point now stops and joins its own observer before any timing, and explicitly records progress/checks limits before and after each complete timed pass. The runtime used by the active neural job is unchanged. Raw forward timing still includes transfer, dispatch, full output and synchronization, while metadata/ledger work stays outside. This is a measurement-boundary repair, not a deployment speed improvement.

## 2026-09-30 — report input schemas and immutable cost provenance

Pre-execution review found three reporting defects: a 36-run expectation omitted the four Hog DIRECT fits from the complete 40-run selection; the direct-selection reader required a newer TEST flag absent from the original immutable direct selection; and Gamma time provenance hashed a live ledger after reading it. The report reader now expects all 40 canonical configurations, explicitly accepts the old `test_used: false` direct schema, and preserves the exact completed coefficient-job subset as an immutable adaptation ledger snapshot. No source selection or experimental result was rewritten. `report_inputs05.json` verifies actual parent/native receipts, both reused DIRECT four-fit searches, the legacy selection reader and the approved run-ID mapping without model execution or new scoring.

## 2026-09-30 — transient Windows ledger read failure during inference

`evaluation_predict01` stopped with PermissionError while opening ledger.json in budget_snapshot, after Robin and ten Jena prediction records were complete. The process exited 1; its original log, failed ledger job, source snapshots and completed prediction receipts are retained. The traceback is a file-open failure, not a model, mask, score or selection failure. A concurrent atomic replacement is a plausible source of the transient contention, but the exact OS holder was not established.

The JSON reader now uses the same bounded PermissionError-only retry schedule as the existing atomic writer (six attempts, total backoff 1.55 seconds). Missing files, invalid JSON, persistent permission errors and hash mismatches still fail. No file permission, package, model, tolerance, data, coefficient or sealed selection is changed. No other process is terminated. The repaired prediction job resumes only receipt-complete outputs with the original selection seal; a partially in-memory model output is recomputed. This is a zero-fit inference retry, with all failed and resumed GPU job time charged. Original first TEST exposure remains unchanged.

## 2026-09-30 — presentation-only plot repair

Visual inspection found the first cost plot legend overlapping tick labels and the large direct-NLinear throughput compressing the TSFM points. The plot now uses a labeled logarithmic throughput axis and a separate legend margin. Original PNG/SVG and report_values remain unchanged; figure_revision01.json links the new v2 render to identical numeric inputs. A first ad-hoc rendering command imported receipt from runtime_v11 instead of report_v11 and failed before job reservation (observed wall time 0.7775688s, no model/score/optimizer). That failure is retained in figure_revision01_failed.json; the corrected render is in the reserved CPU check session figure_layout_revision01. The known pre-reservation overhead is separately included in final resource accounting, not hidden as a fit or synthetic step.
