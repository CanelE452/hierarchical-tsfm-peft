#!/usr/bin/env python
"""Evaluation-window scoring (contract 5.4, 6 S2a/S2b). Refuses to run without a matching SEAL.json.

For every feasible configuration and period this scores F0, the selected LORA and FULL checkpoints,
both deep-learning models and seasonal naive, then forms the gap against F0 and the stability verdict.
Per-series CRPS terms are cached so the bootstrap, the localization tables and the independent
recomputation all read the same numbers.
"""
from __future__ import annotations
import argparse, csv, sys, time
import numpy as np
import ga
import arms
from periods import Period

sys.path.insert(0, str(ga.HERE))
TERMS = ga.CACHE/'terms'
TERMS.mkdir(parents=True, exist_ok=True)
G_THRESHOLD = 0.10                      # contract 6 S2a/S2b


def _seal_or_die():
    path = ga.OUT/'SEAL.json'
    if not path.exists(): raise SystemExit('REFUSED: SEAL.json is missing (contract 6, gate 10.3)')
    seal = ga.read_json(path)
    import seal as seal_mod
    if seal['selection_digest'] != seal_mod.selection_digest():
        raise SystemExit('REFUSED: the sealed selection digest does not match the current artifacts')
    return seal


def _forecasts_for(cfg, period_name, arm, sel):
    per = Period(cfg, period_name)
    test = per.test_data()
    if arm == 'F0':
        pipe = ga.pipeline(dtype='float32')
    elif arm in ('LORA', 'FULL'):
        pipe = ga.pipeline(ga.ROOT/sel[arm]['selected_path'], dtype='float32')
    elif arm.startswith('DL-'):
        from neuralforecast import NeuralForecast
        nf = NeuralForecast.load(str(ga.ROOT/sel[arm]['selected_path']))
        fc = arms.dl_forecast(nf, arm[3:], per.entries, per.freq, per.H, per.W,
                              drop_from_end=per.drop, target_dim=per.target_dim)
        return fc, test, per
    elif arm == 'SNAIVE':
        from gluonts.time_feature import get_seasonality
        from gift_eval_statsforecast import SeasonalNaivePredictor
        multivariate = per.target_dim > 1
        ds_u = ga.dataset(cfg, to_univariate=multivariate)
        # rebuild the period on the univariate view so the seasonal naive arm sees the same windows
        per_u = Period(dict(cfg), period_name)
        per_u.entries = list(ds_u.gluonts_dataset)
        per_u.target_dim = 1
        test_u = per_u.test_data()
        season = get_seasonality(per.freq)
        pred = SeasonalNaivePredictor(per.H, season_length=season, freq=per.freq,
                                      quantile_levels=ga.QUANTILES, batch_size=512)
        fc = list(pred.predict(test_u.input))
        return fc, test_u, per_u
    else:
        raise ValueError(arm)
    ctx = sel.get(arm, {}).get('selected_context') if arm in ('LORA', 'FULL') else None
    fc = ga.forecast_all(ga.Chronos2Predictor(pipe, prediction_length=per.H, context_length=ctx),
                         test, per.W)
    del pipe
    import torch; torch.cuda.empty_cache()
    return fc, test, per


def score_arm(cfg, period_name, arm, sel):
    dest = TERMS/f"{cfg['id']}_{period_name}_{arm}.npz"
    if dest.exists():
        z = np.load(dest)
        return dict(crps=float(z['crps']), mase=float(z['mase']), num=z['num'], den=z['den'],
                    series_of=z['series_of'], day_of=z['day_of'],
                    crps_official=float(z['crps_official']), seconds=float(z['seconds']))
    el = ga.timer()
    fc, test, per = _forecasts_for(cfg, period_name, arm, sel)
    official = ga.evaluate(fc, test, per.freq).reset_index(drop=True).to_dict('records')[0]
    num, den, series_of, day_of = ga.per_unit_terms(fc, test, per.W, steps_per_day(per.freq))
    crps = ga.crps_from_terms(num, den)
    secs = el()
    np.savez_compressed(dest, num=num, den=den, series_of=series_of, day_of=day_of, crps=crps,
                        mase=float(official['MASE[0.5]']),
                        crps_official=float(official['mean_weighted_sum_quantile_loss']), seconds=secs)
    print(f"  {cfg['id']}_{period_name}_{arm}: CRPS {crps:.5f} MASE {official['MASE[0.5]']:.4f} ({secs:.0f}s)",
          flush=True)
    return dict(crps=crps, mase=float(official['MASE[0.5]']), num=num, den=den,
                series_of=series_of, day_of=day_of,
                crps_official=float(official['mean_weighted_sum_quantile_loss']), seconds=secs)


def steps_per_day(freq) -> int:
    """How many steps make a day at this frequency (contract 5.4 block length)."""
    import pandas as pd
    off = pd.tseries.frequencies.to_offset(str(freq))
    per_day = pd.Timedelta('1D')/pd.Timedelta(off.nanos, unit='ns') if getattr(off, 'nanos', None)         else pd.Timedelta('1D')/pd.Timedelta(off)
    return max(int(round(float(per_day))), 1)


def run(only_config=None):
    seal = _seal_or_die()
    sel_all = ga.read_json(ga.CACHE/'selection.json')
    rows, stability = [], []
    for cfg in ga.CONFIGS:
        if only_config and cfg['id'] != only_config: continue
        if not ga.geometry(cfg)['feasible']:
            stability.append(dict(id=cfg['id'], config=cfg['key'], role=cfg['role'],
                                  verdict='INFEASIBLE', G_P1='', G_P2='', ci_P1='', ci_P2='',
                                  ach_P1='', ach_P2='', note='contract 5.2 feasibility'))
            continue
        gaps = {}
        for period_name in ('P1', 'P2'):
            sel = sel_all[f"{cfg['id']}_{period_name}"]
            ach_arm = sel['ACH']['arm']
            scored = {}
            for arm in ['F0', 'LORA', 'FULL', 'DL-PatchTST', 'DL-NHITS', 'SNAIVE']:
                scored[arm] = score_arm(cfg, period_name, arm, sel)
            f0 = scored['F0']
            per = Period(cfg, period_name)
            units, unit_kind = ga.unit_blocks(f0['series_of'], f0['day_of'], per.series_count)
            for arm, s in scored.items():
                g = ga.gap(f0['crps'], s['crps'])
                if arm in ('SNAIVE', 'F0') or len(s['den']) != len(f0['den']):
                    boot = dict(value=g, ci=[float('nan')]*2, n_units=len(units))
                else:
                    boot = ga.bootstrap_gap_terms(f0['num'], f0['den'], s['num'], s['den'], blocks=units)
                rows.append(dict(id=cfg['id'], config=cfg['key'], role=cfg['role'], period=period_name,
                                 arm=arm, is_ach=(arm == ach_arm), crps=s['crps'],
                                 crps_official=s['crps_official'], mase=s['mase'],
                                 G_crps=g, G_crps_ci_low=boot['ci'][0], G_crps_ci_high=boot['ci'][1],
                                 G_mase=ga.gap(f0['mase'], s['mase']), n_units=boot['n_units'],
                                 unit=unit_kind, val_crps=sel.get(arm, {}).get('val_crps', ''),
                                 selected=sel.get(arm, {}).get('selected_checkpoint', ''),
                                 seconds=round(s['seconds'], 1)))
            ach = scored[ach_arm]
            boot = ga.bootstrap_gap_terms(f0['num'], f0['den'], ach['num'], ach['den'], blocks=units)
            gaps[period_name] = dict(G=boot['value'], ci=boot['ci'], ach=ach_arm,
                                     f0_crps=f0['crps'], ach_crps=ach['crps'])
            print(f"{cfg['id']} {period_name}: ACH {ach_arm} G {boot['value']:+.4f} "
                  f"CI [{boot['ci'][0]:+.4f}, {boot['ci'][1]:+.4f}]", flush=True)
        p1, p2 = gaps['P1'], gaps['P2']
        ok1 = p1['G'] >= G_THRESHOLD and p1['ci'][0] > 0
        ok2 = p2['G'] >= G_THRESHOLD and p2['ci'][0] > 0
        verdict = 'STABLE_GAP' if (ok1 and ok2) else ('UNSTABLE_GAP' if (ok1 or ok2) else 'NO_GAP')
        if cfg['role'] == 'control':
            verdict += ' / CONTROL_' + ('FAIL' if (p1['G'] >= G_THRESHOLD or p2['G'] >= G_THRESHOLD) else 'OK')
        stability.append(dict(id=cfg['id'], config=cfg['key'], role=cfg['role'], verdict=verdict,
                              G_P1=p1['G'], G_P2=p2['G'],      # full precision: VERIFY_RECOMPUTE
                                                               # compares against these to 1e-9
                              ci_P1=f"[{p1['ci'][0]:.4f}, {p1['ci'][1]:.4f}]",
                              ci_P2=f"[{p2['ci'][0]:.4f}, {p2['ci'][1]:.4f}]",
                              ach_P1=p1['ach'], ach_P2=p2['ach'],
                              note=f"threshold G >= {G_THRESHOLD} in both periods with CI lower bound > 0"))
    _write(ga.OUT/'GAP_TABLE.csv', rows)
    _write(ga.OUT/'STABILITY.csv', stability)
    print('GAP_TABLE.csv rows', len(rows), '| STABILITY.csv rows', len(stability))


def _write(path, rows):
    if not rows: return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', default=None)
    a = p.parse_args(); run(a.config)
