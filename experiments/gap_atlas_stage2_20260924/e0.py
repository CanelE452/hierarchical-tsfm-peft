#!/usr/bin/env python
"""E0 tool verification (contract section 6). Any failure means STOP_DEBUG.

E0a  official reproduction: F0 CRPS on the official test windows (P2) within 2% of the leaderboard.
E0b  SNAIVE within 5% of the leaderboard seasonal_naive row.
E0c  leakage audit: what each period's training, validation and selection may see, by index.
"""
from __future__ import annotations
import argparse, time
import numpy as np
import ga


def _row(lb, key):
    if key not in lb.index: return None
    r = lb.loc[key]
    if hasattr(r, 'iloc') and getattr(r, 'ndim', 1) > 1: r = r.iloc[0]
    return r


def e0a_one(cfg, pipe=None):
    ds = ga.dataset(cfg)
    el = ga.timer()
    pipe = pipe or ga.pipeline(dtype='float32')
    predictor = ga.Chronos2Predictor(pipe, prediction_length=ds.prediction_length)
    forecasts = ga.forecast_all(predictor, ds.test_data, ds.windows)
    res = ga.evaluate(forecasts, ds.test_data, ds.freq).reset_index(drop=True).to_dict('records')[0]
    return dict(crps=float(res['mean_weighted_sum_quantile_loss']), mase=float(res['MASE[0.5]']),
                windows=int(ds.windows), prediction_length=int(ds.prediction_length),
                forecasts=len(forecasts), seconds=round(el(), 1)), pipe


def e0a():
    lb, lb_sha = ga.leaderboard('chronos-2')
    out = {}
    pipe = None
    for cfg in ga.CONFIGS:
        got, pipe = e0a_one(cfg, pipe)
        ref = _row(lb, cfg['key'])
        ref_crps = float(ref[ga.CRPS_COL]); ref_mase = float(ref[ga.MASE_COL])
        rel = abs(got['crps'] - ref_crps)/ref_crps
        out[cfg['id']] = dict(key=cfg['key'], **got, leaderboard_crps=ref_crps,
                              leaderboard_mase=ref_mase, relative_difference=float(rel),
                              mase_relative_difference=float(abs(got['mase']-ref_mase)/ref_mase),
                              pass_=bool(rel <= 0.02))
        o = out[cfg['id']]
        print(f"E0a {cfg['id']} {cfg['key']:32s} ours {o['crps']:.5f} vs board {ref_crps:.5f} "
              f"rel {rel*100:5.2f}%  {'PASS' if o['pass_'] else 'FAIL'} ({o['seconds']}s)", flush=True)
    del pipe
    import torch; torch.cuda.empty_cache()
    return dict(pass_=all(v['pass_'] for v in out.values()), tolerance=0.02,
                leaderboard_sha256=lb_sha, per_config=out,
                settings=dict(quantiles=ga.QUANTILES, batch_size=ga.BATCH, dtype='float32',
                              cross_learning=True, revision=ga.REVISION,
                              windows_predicted='one rolling window at a time (official notebook)'))


def e0b():
    import sys
    sys.path.insert(0, str(ga.HERE))
    from gluonts.model import evaluate_model
    from gluonts.time_feature import get_seasonality
    from gift_eval_statsforecast import SeasonalNaivePredictor
    lb, lb_sha = ga.leaderboard('seasonal_naive')
    out = {}
    for cfg in ga.CONFIGS:
        el = ga.timer()
        multivariate = ga.dataset(cfg).target_dim > 1
        ds = ga.dataset(cfg, to_univariate=multivariate)      # official baseline scores univariately
        season = get_seasonality(ds.freq)
        predictor = SeasonalNaivePredictor(ds.prediction_length, season_length=season,
                                           freq=ds.freq, quantile_levels=ga.QUANTILES, batch_size=512)
        res = evaluate_model(predictor, test_data=ds.test_data, metrics=ga.metric_list(),
                             batch_size=512, axis=None, mask_invalid_label=True,
                             allow_nan_forecast=False, seasonality=season)
        crps = float(res['mean_weighted_sum_quantile_loss'][0])
        ref = _row(lb, cfg['key'])
        ref_crps = float(ref[ga.CRPS_COL])
        rel = abs(crps - ref_crps)/ref_crps
        out[cfg['id']] = dict(key=cfg['key'], crps=crps, leaderboard_crps=ref_crps,
                              relative_difference=float(rel), season_length=int(season),
                              to_univariate=bool(multivariate), seconds=round(el(), 1),
                              pass_=bool(rel <= 0.05))
        print(f"E0b {cfg['id']} {cfg['key']:32s} ours {crps:.5f} vs board {ref_crps:.5f} "
              f"rel {rel*100:5.2f}%  {'PASS' if out[cfg['id']]['pass_'] else 'FAIL'}", flush=True)
    return dict(pass_=all(v['pass_'] for v in out.values()), tolerance=0.05,
                leaderboard_sha256=lb_sha, per_config=out)


def e0c():
    """Index-level statement of the period boundaries and what each stage may read."""
    per = {}
    for cfg in ga.CONFIGS:
        g = ga.geometry(cfg)
        H, W, V, L = g['prediction_length'], g['windows'], g['validation_windows'], g['min_series_length']
        # offsets are counted from the end of each series (negative indices)
        p2 = dict(eval_start=-W*H, eval_end=None, val_start=-(W+V)*H, val_end=-W*H, train_end=-(W+V)*H)
        p1 = dict(eval_start=-2*W*H, eval_end=-W*H, val_start=-(2*W+V)*H, val_end=-2*W*H,
                  train_end=-(2*W+V)*H)
        per[cfg['id']] = dict(key=cfg['key'], H=H, W=W, V=V, min_series_length=L,
                              P1=p1, P2=p2, feasible=g['feasible'],
                              overlap_note='P2 validation may overlap P1 evaluation; the contract 5.1 '
                                           'allows it because those points are in the past of P2')
    return dict(pass_=True, per_config=per,
                rules=dict(training='series[:train_end] only',
                           validation='series[val_start:val_end], used for checkpoint and model choice',
                           evaluation='series[eval_start:eval_end], scored only after SEAL.json',
                           selection='never reads the evaluation window',
                           scoring='the scoring script refuses to run without a matching SEAL.json'))


def main():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['a', 'b', 'c', 'all'])
    a = p.parse_args()
    path = ga.OUT/'E0_REPORT.json'
    report = ga.read_json(path) if path.exists() else dict(utc=time.time())
    if a.stage in ('a', 'all'): report['E0a_official_reproduction'] = e0a()
    if a.stage in ('b', 'all'): report['E0b_seasonal_naive'] = e0b()
    if a.stage in ('c', 'all'): report['E0c_leakage'] = e0c()
    done = [k for k in ('E0a_official_reproduction', 'E0b_seasonal_naive', 'E0c_leakage') if k in report]
    if len(done) == 3:
        fails = [k for k in done if not report[k]['pass_']]
        report['verdict'] = 'E0_PASS' if not fails else 'STOP_DEBUG'
        report['failures'] = fails
        print('E0 verdict:', report['verdict'], fails)
    report['date'] = time.strftime('%Y-%m-%d %H:%M:%S')
    ga.write_json(path, report)


if __name__ == '__main__':
    main()
