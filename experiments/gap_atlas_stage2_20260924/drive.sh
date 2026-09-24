#!/usr/bin/env bash
# Gap atlas Stage 2 driver: runs every remaining step to the final report, without stopping at
# "training finished". Each step is judged by its artefact, not by an exit code alone.
set -u
PY="E:/CODING/proj/hierarchical-tsfm-peft/.venv/Scripts/python.exe"
HERE="E:/CODING/proj/hierarchical-tsfm-peft/experiments/gap_atlas_stage2_20260924"
OUT="E:/CODING/proj/hierarchical-tsfm-peft/results/gap_atlas_stage2_20260924"
CACHE="E:/CODING/proj/hierarchical-tsfm-peft/.cache/gap_atlas_stage2_20260924"
export PYTHONUTF8=1 HF_HUB_DISABLE_PROGRESS_BARS=1
cd "$HERE" || exit 1

say() { echo "[$(date +%H:%M:%S)] $*"; }
die() { say "FAILED at $*"; exit 1; }

# STEP 0 — wait for any training started outside this driver to release the GPU. pgrep does not see
# Windows processes from Git Bash, so the card's own memory reading is the signal.
while true; do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ -z "$USED" ] && break
  [ "$USED" -lt 2000 ] && break
  say "waiting for the GPU to free up (${USED} MiB in use)"
  sleep 60
done
say "GPU free; starting"

# STEP 1 — train and select every arm in every feasible configuration and period
# The deep-learning arm runs in its own process: transformers sets the matmul precision through the
# legacy API during a Chronos fit and lightning uses the new one, which PyTorch 2.10 refuses to mix.
say "STEP 1a Chronos arms"
for stage in f0 lora full; do
  say "  stage $stage"
  "$PY" run.py --config all --period all --only $stage || die "run.py --only $stage"
done
say "STEP 1b deep-learning arm (separate process)"
"$PY" run.py --config all --period all --only dl || die "run.py --only dl"
[ -f "$CACHE/selection.json" ] || die "selection.json missing"
"$PY" - <<'EOF' || exit 1
import ga, sys
sel = ga.read_json(ga.CACHE/'selection.json')
need = [f"{c['id']}_{p}" for c in ga.CONFIGS if ga.geometry(c)['feasible'] for p in ('P1','P2')]
missing = [k for k in need if 'ACH' not in sel.get(k, {})]
if missing: print('INCOMPLETE selection:', missing); sys.exit(1)
print('selection complete for', len(need), 'config-period pairs')
EOF

# STEP 2 — seal, then score the evaluation windows
say "STEP 2 seal"
"$PY" seal.py || die "seal.py"
[ -f "$OUT/SEAL.json" ] || die "SEAL.json missing"
say "STEP 3 score"
"$PY" score.py || die "score.py"
[ -f "$OUT/GAP_TABLE.csv" ] || die "GAP_TABLE.csv missing"
[ -f "$OUT/STABILITY.csv" ] || die "STABILITY.csv missing"

# STEP 4 — localization for every configuration judged STABLE_GAP (criterion fixed in score.py)
say "STEP 4 localization"
STABLE=$("$PY" - <<'EOF'
import csv, ga
with (ga.OUT/'STABILITY.csv').open(encoding='utf-8') as f:
    print(' '.join(r['id'] for r in csv.DictReader(f)
                   if r['verdict'].startswith('STABLE_GAP') and r['role'] == 'candidate'))
EOF
)
for cid in $STABLE; do
  for period in P1 P2; do
    say "  localize $cid $period"
    "$PY" localize.py "$cid" --period "$period" || die "localize.py $cid $period"
  done
done
"$PY" hypotheses_doc.py || die "hypotheses_doc.py"

# STEP 5 — independent recomputation, figures, report
say "STEP 5 verify"
"$PY" verify_recompute.py || die "verify_recompute.py"
if [ -f figures.py ]; then
  if [ -n "$STABLE" ]; then FIGARGS=""; else FIGARGS="--no-localization"; fi
  "$PY" figures.py $FIGARGS || say "figures.py failed (reported, not fatal)"
fi
"$PY" report.py || die "report.py"
[ -f "$OUT/REPORT_KO.md" ] || die "REPORT_KO.md missing"

# STEP 6 — verdict and key numbers, not just "done"
say "STEP 6 verdict"
"$PY" - <<'EOF'
import csv, ga
fd = (ga.OUT/'FINAL_DECISION.md').read_text(encoding='utf-8').splitlines()
print('VERDICT:', fd[0].lstrip('# ').strip())
with (ga.OUT/'STABILITY.csv').open(encoding='utf-8') as f:
    for r in csv.DictReader(f):
        print(f"  {r['id']:3s} {r['config']:32s} {r['role']:9s} G_P1 {r['G_P1'] or '-':>9s} "
              f"G_P2 {r['G_P2'] or '-':>9s}  {r['verdict']}")
v = ga.read_json(ga.OUT/'VERIFY_RECOMPUTE.json')
print(f"recompute max diff {v['max_abs_difference']:.3e} over {v['n_checks']} checks, pass {v['pass_']}")
led = ga.read_json(ga.CACHE/'fit_ledger.json')
print(f"fits {len(led)}, training minutes {sum(x.get('seconds',0) for x in led.values())/60:.0f}")
EOF
say "ALL DONE"
