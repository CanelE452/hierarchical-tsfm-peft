# K-tau rule implementation plan

> **For agentic workers:** execute the approved specification in this namespace, with independent fixture and result review. The latest direct user Step 1/G1/G5 amendment supersedes the original downloaded clauses.

**Goal:** evaluate the fixed training-row PCA variance rule K_tau(.9) on four datasets and ten fixed seeds without selecting K on V.

**Architecture:** `data.py` admits the same historical pre-T tasks; `runner.py` adapts only registration and per-job selection binding while the existing fit/refit/FP64 core stays unchanged. `spectrum.py` refuses to compute before a committed byte-bound preregistration. `execute.py` runs one owned GPU worker at a time. `analysis.py` derives decisions and figures from public tables.

**Tech Stack:** existing local Python3.11/PyTorch2.10+cu128, NumPy, Matplotlib, RTX4070. No installation or download.

**Spec:** the original `k_tau_rule_executor_instructions.md`, its SHA in PREREGISTRATION.json, and the exact user-approved amendment copied into PREREGISTRATION.md/JSON.

## Global constraints

- tau .9; .7/.8/.95 are descriptive K values only; seeds92601–92610.
- jena/ETTh1/ETTh2_safe17400/appliances; L512/H48; RevIN off; FP32; AdamW .001/weight_decay0/clip1; effective16/micro4; F rows128.
- Checkpoints0/.25/1/3; strict A_stop minimum with earlier tie; A60 fresh refit; fixed DEFAULT128 comparison policy. Existing core sources remain byte-identical.
- All new files stay in the three campaign namespaces. Old gated audit records remain available. No T evaluation or numeric T access.
- Fixed 24300-second active budget, including a conservative preparation charge. The explicit old G1 stop/user-wait interval is disclosed separately, never represented as GPU work.
- No K/seed/epoch changes after V. No optional follow-up experiment after REPLAN.

## Review focus

Uncommitted preregistration must reject actual PCA calls; tuple/list JSON identities must survive retries; missing/changed prefix hashes must fail before loading; incomplete fixed-seed coverage must never yield ADOPT; G4 replay and baseline-comparison jobs must remain separate from the ten main seeds.

## Tasks and interfaces

1. [x] Audit prior PCA artifacts/scalars and Jena's allowed M_fit spectrum. Recalculate historical individual-run V aggregates and fixed eligible set. Preserve before hashes.
2. [x] Implement `data.build_context` / `fingerprint_context`, scoped `runner.run_one`, guarded `spectrum.compute_spectra`, controller `execute`, and public-only `analysis.build_report`. Only the new registration/orchestration layer changes.
3. [x] Add explicit synthetic fixtures covering committed bytes, source mutation, eigenvalue/tau boundaries, seed coverage, masked prefix bounds, selection seals/ties, retries, G4 fallback comparisons, budget rejection and decisions.
4. [ ] Freeze and commit the preregistration before any real spectrum. Record the actual commit/time and verify its bytes. Run the four registered minimum-K seed92601 G4 fallbacks required by incomplete original runtime fingerprints.
5. [ ] Call the same fit_statistics→initialize_pca functions on A_fit/A60; validate old scalar agreement within1e-6. Save all spectra/auxiliary tau values; use only A_fit tau.9 for the main plan.
6. [ ] Seal the complete run plan and historical upper-bound budget before main training. Reprofile the first unseen K; retain all failures/negative seeds, one retry maximum, no concurrent GPU jobs.
7. [ ] Complete fixed-seed results, verify saved-array/core receipts, compute median/linear quartiles/loss/decision mechanically, render and inspect both PNG/SVG figures, compare old hashes and publish the final local report. The only requested Git publication step is the preregistration commit; no remote push is inferred for this campaign.
