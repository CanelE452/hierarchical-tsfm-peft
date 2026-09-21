# Hierarchical PEFT screen v1

Status: SEALED DESIGN; main training requires a matching PROTOCOL_SEAL.json and passing preflight.

The sole execution contract is `docs/00_MASTER_CLI_hierarchical_tsfm_peft_new_pc_20260921.txt`.
The three design-history documents are immutable context, not additional experiments.
User clarification authorizes the pre-created empty GitHub repository; it was made private.

Milestones:
- [x] Read MASTER completely; verify ZIP/master identity and repository ownership.
- [x] Extract and hash immutable source documents; bootstrap private repository.
- [x] Create isolated Python 3.11 environment; validate CUDA and freeze dependencies.
- [x] Download official Nixtla data and verify all timestamps, schema, sums and support graph.
- [x] Implement and verify Chronos F0, LoRA, adapters, permissions and reconciliation.
- [x] Seal remaining numerical implementation constants before inspecting results.
- [ ] Execute Labour Stage A within 16 fits / 8,192 main updates.
- [ ] Apply continuation gate; TourismLarge only if permitted.
- [ ] Publish verified Korean report and decision with scoped commits.

Failures are classified as implementation failure, data failure, resource block, or scientific negative result.
No main fit starts until every MASTER smoke gate passes.

## Numerical choices fixed before any main fit or TEST forecast

- FP32; CUDA matmul and cuDNN TF32 disabled. Frozen backbone stays in eval mode,
  including its pretrained dropout; all arms use identical deterministic dropout behavior.
- Adapter RMS is `sqrt(mean(x**2) + 1e-12)` over the 512 feature dimensions;
  epsilon is `1e-6`, stateless layer norm epsilon `1e-6`, cap is `0.1 * stopgrad(rms(h))`.
  The context RMS ratio is detached. GELU uses PyTorch's exact default.
- TRAIN seasonal denominator floor is `max(1e-8, mean(positive TRAIN scales) * 1e-8)`;
  all-zero TRAIN scales use 1 as the reference mean. No future value enters this rule.
- Per-series RMSSE is sqrt(mean squared errors over all evaluated origins/horizons / TRAIN scale).
  Each official level averages its series RMSSEs; primary equally averages level means.
  TRAIN loss averages scaled squared error using the same level weights before the square root.
- Origins have stride 1. One optimizer update uses all nodes at one origin.
  The 512-origin schedule is sampled uniformly with replacement using NumPy default_rng(seed),
  identical across arms and LRs for a seed. Node chunks of 64 preserve the all-node context/loss.
- Calibration residual columns are origin-major then horizon-major (13 x 12 = 156 columns).
  Overlapping calendar targets are retained as distinct forecast errors; no TEST residual is used.
  Official Nixtla MinTrace mint_shrink default ridge is unchanged; no custom covariance code.
- Each checkpoint is evaluated on all 13 CALIBRATION and 13 VALIDATION origins.
  LR/checkpoint tie order is validation primary, lower LR, earlier checkpoint.
  All 12 selected Stage A models are fixed and persisted before TEST scoring starts.
- SeasonalNaive(12) and AutoETS(12) use the same 48 past months at each origin and official
  StatsForecast forecast methods. Their CALIBRATION predictions fit the same reconciler options.
- The informal MASTER phrase 'no serious bottom-level harm' is operationalized conservatively:
  no selected seed may worsen MinT bottom RMSSE by over 5% against SELF or POOL.
  Continuation additionally requires mean unreconciled primary gains against both controls,
  the MASTER MinT mean gains, and favorable MinT direction in both repeat seeds.
  Differences below 1e-8 are treated as numerical ties. These are project budget rules,
  not scientific significance thresholds. No threshold is changed after outcomes.
- All 555 TourismLarge nodes remain. Duplicate supports are explained by official tag metadata;
  only strict support inclusion defines edges, with no equal-support edge, merge, or tie-break.
- Smoke updates are separately counted. A failed main process cannot automatically repeat fits;
  spent updates and failure evidence must first be reviewed. No CPU screen fallback.
