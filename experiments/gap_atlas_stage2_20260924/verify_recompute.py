#!/usr/bin/env python
"""Independent recomputation of GAP_TABLE and STABILITY from the cached per-series terms, numpy only
(contract section 7: maximum difference <= 1e-9)."""
from __future__ import annotations
import csv, math, time
from pathlib import Path
import numpy as np
import ga

TERMS = ga.CACHE/'terms'
TOL = 1e-9


def crps(num, den, idx=None):
    if idx is None: idx = np.arange(len(den))
    d = float(den[idx].sum())
    if d <= 0: return float('nan')
    return float(np.mean(num[idx].sum(axis=0)/d))


def rows(name):
    with (ga.OUT/name).open(encoding='utf-8') as f: return list(csv.DictReader(f))


def fnum(x):
    try: return float(x)
    except (TypeError, ValueError): return float('nan')


def main():
    gap_rows = rows('GAP_TABLE.csv')
    stab_rows = rows('STABILITY.csv')
    checks = {}
    cache = {}

    def terms(cid, period, arm):
        key = (cid, period, arm)
        if key not in cache:
            z = np.load(TERMS/f'{cid}_{period}_{arm}.npz')
            cache[key] = (z['num'], z['den'])      # unit-level terms, see ga.per_unit_terms
        return cache[key]

    for r in gap_rows:
        cid, period, arm = r['id'], r['period'], r['arm']
        num, den = terms(cid, period, arm)
        recomputed = crps(num, den)
        checks[f'{cid}_{period}_{arm}_crps'] = dict(recomputed=recomputed, reported=fnum(r['crps']))
        nf, df = terms(cid, period, 'F0')
        g = 1.0 - recomputed/crps(nf, df) if crps(nf, df) else float('nan')
        checks[f'{cid}_{period}_{arm}_gap'] = dict(recomputed=float(g), reported=fnum(r['G_crps']))

    for r in stab_rows:
        if r['verdict'] == 'INFEASIBLE': continue
        for period, col in (('P1', 'G_P1'), ('P2', 'G_P2')):
            ach = r['ach_P1'] if period == 'P1' else r['ach_P2']
            nf, df = terms(r['id'], period, 'F0')
            na, da = terms(r['id'], period, ach)
            g = 1.0 - crps(na, da)/crps(nf, df)
            checks[f"{r['id']}_{period}_stability_gap"] = dict(recomputed=float(g), reported=fnum(r[col]))

    for v in checks.values():
        v['abs_difference'] = abs(v['recomputed'] - v['reported']) if np.isfinite(v['reported']) else float('nan')
    diffs = [v['abs_difference'] for v in checks.values() if np.isfinite(v['abs_difference'])]
    res = dict(utc=time.time(), tolerance=TOL, checks=checks, n_checks=len(checks),
               max_abs_difference=float(np.max(diffs)) if diffs else float('nan'),
               note='the reported CRPS is computed from the same per-series terms in float64; the '
                    'gluonts float32 aggregate is reported beside it in GAP_TABLE.csv as crps_official')
    res['pass_'] = bool(res['max_abs_difference'] <= TOL)
    ga.write_json(ga.OUT/'VERIFY_RECOMPUTE.json', res)
    print('verify max abs difference', res['max_abs_difference'], 'over', len(checks), 'checks; pass', res['pass_'])
    # how far the float32 gluonts aggregate sits from the float64 recomputation, reported not gated
    off = [abs(fnum(r['crps']) - fnum(r['crps_official'])) for r in gap_rows if r['crps_official']]
    if off: print(f'gluonts float32 aggregate differs by at most {max(off):.3e}')


if __name__ == '__main__':
    main()
