#!/usr/bin/env python
"""L1-L7 localization for configurations that came out STABLE_GAP (contract 6 S2c).

These are descriptive tables computed on the evaluation windows after scoring; nothing here feeds
back into a selection. Series attributes are measured on the training segment only.
"""
from __future__ import annotations
import csv, sys
import numpy as np
import ga
from periods import Period

OUTDIR = ga.OUT/'LOCALIZATION'
Q = ga.QUANTILES


def _forecast_cache(cfg, period_name, arm, sel):
    """Recompute the forecasts needed for the step/variate/phase breakdowns."""
    import score
    fc, test, per = score._forecasts_for(cfg, period_name, arm, sel)
    return fc, list(test), per


def _loss_tensor(forecasts, labels):
    """Quantile loss and |y| per (entry, step, variate), averaged over quantile levels."""
    num, den = [], []
    for fc, lab in zip(forecasts, labels):
        y = np.asarray(lab['target'], dtype=float)
        if y.ndim == 2: y = y.T
        y = y[:, None] if y.ndim == 1 else y
        acc = np.zeros_like(y)
        for q in Q:
            p = np.asarray(fc.quantile(q), dtype=float)
            p = p[:, None] if p.ndim == 1 else p
            acc += 2*np.abs((p - y)*((p >= y).astype(float) - q))
        num.append(acc/len(Q)); den.append(np.abs(y))
    return np.stack(num), np.stack(den)       # [entries, H, variates]


def series_attributes(cfg, period_name):
    """Contract L3 attributes, measured on the training segment only."""
    per = Period(cfg, period_name)
    rows = []
    for i, e in enumerate(per.training_series()):
        t = np.asarray(e['target'], dtype=float)
        t = t[None, :] if t.ndim == 1 else t
        flat = t.reshape(-1)
        finite = flat[np.isfinite(flat)]
        level = float(np.mean(finite)) if len(finite) else float('nan')
        sd = float(np.std(finite)) if len(finite) else float('nan')
        cv = sd/abs(level) if level not in (0.0,) and np.isfinite(level) else float('nan')
        zero_fraction = float(np.mean(finite == 0)) if len(finite) else float('nan')
        season = _day_steps(per.freq)
        burst = _burst_fraction(t, season)
        rows.append(dict(series=i, level=level, cv=cv, zero_fraction=zero_fraction,
                         burst_fraction=burst, daily_strength=_daily_strength(t, season),
                         spectral_entropy=_spectral_entropy(t)))
    return rows


def _day_steps(freq) -> int:
    from gluonts.time_feature import get_seasonality
    s = int(get_seasonality(str(freq)))
    return max(s, 2)


def _burst_fraction(t: np.ndarray, window: int) -> float:
    """Share of points above twice the trailing one-day 95th percentile (contract L3)."""
    out = []
    for v in t:
        x = v[np.isfinite(v)]
        if len(x) <= window: out.append(float('nan')); continue
        thr = np.array([2*np.quantile(x[max(0, i-window):i], 0.95) for i in range(window, len(x))])
        out.append(float(np.mean(x[window:] > thr)))
    return float(np.nanmean(out))


def _daily_strength(t: np.ndarray, window: int) -> float:
    """Autocorrelation at the daily lag, averaged over variates."""
    vals = []
    for v in t:
        x = v[np.isfinite(v)]
        if len(x) <= window + 2: continue
        a, b = x[:-window] - x[:-window].mean(), x[window:] - x[window:].mean()
        d = np.sqrt((a @ a)*(b @ b))
        if d > 0: vals.append(float(a @ b/d))
    return float(np.mean(vals)) if vals else float('nan')


def _spectral_entropy(t: np.ndarray) -> float:
    vals = []
    for v in t:
        x = v[np.isfinite(v)]
        if len(x) < 16: continue
        x = x - x.mean()
        p = np.abs(np.fft.rfft(x*np.hanning(len(x))))**2
        p = p[1:]
        s = p.sum()
        if s <= 0: continue
        p = p/s; p = p[p > 0]
        vals.append(float(-np.sum(p*np.log(p))/np.log(len(p))))
    return float(np.mean(vals)) if vals else float('nan')


def _write(name, rows):
    if not rows: return
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with (OUTDIR/name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('  wrote', name, len(rows), 'rows', flush=True)


def localize(cfg, period_name='P2'):
    sel = ga.read_json(ga.CACHE/'selection.json')[f"{cfg['id']}_{period_name}"]
    ach_arm = sel['ACH']['arm']
    per = Period(cfg, period_name)
    tag = f"{cfg['id']}_{period_name}"
    packs = {}
    for arm in ('F0', ach_arm):
        fc, labels, _ = _forecast_cache(cfg, period_name, arm, sel)
        packs[arm] = _loss_tensor(fc, labels)
    (nf0, df0), (na, da) = packs['F0'], packs[ach_arm]
    W, H = per.W, per.H

    # L1 per horizon step
    rows = []
    for h in range(H):
        f0 = nf0[:, h, :].sum()/df0[:, h, :].sum()
        ac = na[:, h, :].sum()/da[:, h, :].sum()
        rows.append(dict(step=h+1, crps_f0=f0, crps_ach=ac, ratio=ac/f0 if f0 else float('nan'),
                         gap=ga.gap(f0, ac)))
    _write(f'L1_horizon_{tag}.csv', rows)

    # L2 per variate
    rows = []
    for v in range(nf0.shape[2]):
        f0 = nf0[:, :, v].sum()/df0[:, :, v].sum()
        ac = na[:, :, v].sum()/da[:, :, v].sum()
        rows.append(dict(variate=v, crps_f0=f0, crps_ach=ac, gap=ga.gap(f0, ac)))
    _write(f'L2_variate_{tag}.csv', rows)

    # L3 per series, with training-segment attributes
    attrs = {r['series']: r for r in series_attributes(cfg, period_name)}
    n_series = nf0.shape[0]//W
    rows = []
    for i in range(n_series):
        sl = slice(i*W, (i+1)*W)
        f0 = nf0[sl].sum()/df0[sl].sum(); ac = na[sl].sum()/da[sl].sum()
        rows.append(dict(series=i, crps_f0=f0, crps_ach=ac, gap=ga.gap(f0, ac),
                         weight=float(df0[sl].sum()/df0.sum()),
                         absolute_gain=float(nf0[sl].sum() - na[sl].sum()),
                         **{k: v for k, v in attrs.get(i, {}).items() if k != 'series'}))
    rows.sort(key=lambda r: -r['absolute_gain'])
    total_gain = sum(r['absolute_gain'] for r in rows)
    top10 = rows[:max(1, len(rows)//10)]
    _write(f'L3_series_{tag}.csv', rows)
    share = dict(config=cfg['key'], period=period_name, series=len(rows),
                 top10pct_share_of_total_gain=float(sum(r['absolute_gain'] for r in top10)/total_gain)
                 if total_gain else float('nan'))

    # L3b terciles of each attribute
    trows = []
    for attr in ('level', 'cv', 'zero_fraction', 'burst_fraction', 'daily_strength', 'spectral_entropy'):
        vals = np.array([r.get(attr, np.nan) for r in rows], dtype=float)
        ok = np.isfinite(vals)
        if ok.sum() < 6: continue
        lo, hi = np.nanquantile(vals[ok], [1/3, 2/3])
        for name, sel_mask in (('bottom', vals <= lo), ('middle', (vals > lo) & (vals < hi)), ('top', vals >= hi)):
            m = sel_mask & ok
            if not m.any(): continue
            idx = np.flatnonzero(m)
            f0 = sum(rows[j]['crps_f0']*rows[j]['weight'] for j in idx)
            trows.append(dict(attribute=attr, tercile=name, series=int(m.sum()),
                              median_gap=float(np.median([rows[j]['gap'] for j in idx])),
                              share_of_total_gain=float(sum(rows[j]['absolute_gain'] for j in idx)/total_gain)
                              if total_gain else float('nan')))
    _write(f'L3b_terciles_{tag}.csv', trows)

    # L4 hour of day and weekday, L5 regime
    starts = [lab for lab in range(0)]      # placeholder, filled below
    fc, labels, _ = _forecast_cache(cfg, period_name, 'F0', sel)
    times = []
    for lab in labels:
        start = lab['start']
        times.append([(start + k).to_timestamp() for k in range(H)])
    hours = np.array([[ts.hour for ts in row] for row in times])
    dows = np.array([[ts.dayofweek for ts in row] for row in times])
    rows = []
    for h in range(24):
        m = hours == h
        if not m.any(): continue
        f0 = nf0[m].sum()/df0[m].sum(); ac = na[m].sum()/da[m].sum()
        rows.append(dict(hour=h, points=int(m.sum()), crps_f0=f0, crps_ach=ac, gap=ga.gap(f0, ac)))
    _write(f'L4_hour_{tag}.csv', rows)
    rows = []
    for d in range(7):
        m = dows == d
        if not m.any(): continue
        f0 = nf0[m].sum()/df0[m].sum(); ac = na[m].sum()/da[m].sum()
        rows.append(dict(weekday=d, points=int(m.sum()), crps_f0=f0, crps_ach=ac, gap=ga.gap(f0, ac)))
    _write(f'L4_weekday_{tag}.csv', rows)

    # L5 regime by the label level relative to each series' training median
    med = {}
    for i, e in enumerate(per.training_series()):
        t = np.asarray(e['target'], dtype=float)
        t = t[None, :] if t.ndim == 1 else t
        med[i] = np.nanmedian(t, axis=1)
    y_all = []
    for k, lab in enumerate(labels):
        y = np.asarray(lab['target'], dtype=float)
        if y.ndim == 2: y = y.T
        y = y[:, None] if y.ndim == 1 else y
        y_all.append(y/np.where(med[k//W] == 0, np.nan, med[k//W])[None, :])
    ratio = np.stack(y_all)
    rows = []
    for name, m in (('idle (<= 0.25x median)', ratio <= 0.25),
                    ('normal', (ratio > 0.25) & (ratio < 2.0)),
                    ('burst (>= 2x median)', ratio >= 2.0)):
        m = m & np.isfinite(ratio)
        if not m.any(): continue
        f0 = nf0[m].sum()/df0[m].sum(); ac = na[m].sum()/da[m].sum()
        rows.append(dict(regime=name, points=int(m.sum()), crps_f0=f0, crps_ach=ac, gap=ga.gap(f0, ac)))
    _write(f'L5_regime_{tag}.csv', rows)

    # L6 calibration of the 80% interval, L7 point versus distribution
    rows = []
    for arm in ('F0', ach_arm):
        fcs, labs, _ = _forecast_cache(cfg, period_name, arm, sel)
        cov, width, wid_rel = [], [], []
        for f, lab in zip(fcs, labs):
            y = np.asarray(lab['target'], dtype=float)
            if y.ndim == 2: y = y.T
            lo = np.asarray(f.quantile(0.1), dtype=float); hi = np.asarray(f.quantile(0.9), dtype=float)
            if y.ndim == 1: lo, hi = lo.reshape(y.shape), hi.reshape(y.shape)
            m = np.isfinite(y)
            cov.append(float(np.mean(((y >= lo) & (y <= hi))[m])))
            width.append(float(np.mean((hi - lo)[m])))
            scale = np.nanmean(np.abs(y[m])) or np.nan
            wid_rel.append(float(np.mean((hi - lo)[m])/scale))
        rows.append(dict(arm=arm, nominal_coverage=0.8, empirical_coverage=float(np.mean(cov)),
                         mean_width=float(np.mean(width)), mean_width_over_mean_abs_y=float(np.nanmean(wid_rel))))
    _write(f'L6_calibration_{tag}.csv', rows)
    ga.write_json(OUTDIR/f'L3_share_{tag}.json', share)
    print(f'  top 10% of series hold {share["top10pct_share_of_total_gain"]:.3f} of the total gain', flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('config'); p.add_argument('--period', default='P2')
    a = p.parse_args()
    localize(ga.BY_ID[a.config], a.period)
