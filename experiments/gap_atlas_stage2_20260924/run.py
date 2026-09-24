#!/usr/bin/env python
"""Train and select the arms of contract 5.3 inside each period. Selection uses the validation block
only; the evaluation window is never read here. Results go to .cache/.../selection.json, which SEAL
hashes before score.py is allowed to touch the evaluation windows."""
from __future__ import annotations
import argparse, time
import numpy as np
import ga
import arms
from periods import Period

SEL = ga.CACHE/'selection.json'


def feasible_configs():
    return [c for c in ga.CONFIGS if ga.geometry(c)['feasible']]


def _load(path=SEL):
    return ga.read_json(path) if path.exists() else {}


def _save(obj, path=SEL):
    ga.write_json(path, obj)


def f0_validation(cfg, period_name):
    """F0 on the validation block, so that the ACH choice compares like with like."""
    per = Period(cfg, period_name)
    val = per.validation_data()
    pipe = ga.pipeline(dtype='float32')
    fc = ga.forecast_all(ga.Chronos2Predictor(pipe, prediction_length=per.H), val, per.V)
    res = ga.evaluate(fc, val, per.freq).reset_index(drop=True).to_dict('records')[0]
    del pipe
    import torch; torch.cuda.empty_cache()
    return float(res['mean_weighted_sum_quantile_loss'])


def dl_validation(cfg, period_name, model_name):
    per = Period(cfg, period_name)
    nf = arms.fit_dl(cfg, period_name, model_name)
    val = per.validation_data()
    fc = arms.dl_forecast(nf, model_name, per.entries, per.freq, per.H, per.V,
                          drop_from_end=per.drop + per.W*per.H, target_dim=per.target_dim)
    res = ga.evaluate(fc, val, per.freq).reset_index(drop=True).to_dict('records')[0]
    path = ga.CACHE/'fits'/f"{cfg['id']}_{period_name}_DL-{model_name}"
    path.mkdir(parents=True, exist_ok=True)
    nf.save(str(path), overwrite=True, save_dataset=False)
    return dict(arm=f'DL-{model_name}', val_crps=float(res['mean_weighted_sum_quantile_loss']),
                selected_path=str(path.relative_to(ga.ROOT)))


def one(cfg, period_name, only=None):
    state = _load()
    key = f"{cfg['id']}_{period_name}"
    rec = state.get(key, {})
    if only in (None, 'f0') and 'F0_val_crps' not in rec:
        el = ga.timer(); rec['F0_val_crps'] = f0_validation(cfg, period_name)
        print(f'{key} F0 val {rec["F0_val_crps"]:.5f} ({el():.0f}s)', flush=True)
        state[key] = rec; _save(state)
    for mode in ('lora', 'full'):
        if only not in (None, mode): continue
        if mode.upper() in rec: continue
        arms.fit_chronos(cfg, period_name, mode)
        rec[mode.upper()] = arms.score_chronos_checkpoints(cfg, period_name, mode)
        state[key] = rec; _save(state)
    for model_name in ('PatchTST', 'NHITS'):
        if only not in (None, 'dl'): continue
        if f'DL-{model_name}' in rec: continue
        rec[f'DL-{model_name}'] = dl_validation(cfg, period_name, model_name)
        state[key] = rec; _save(state)
    # ACH: the best of LORA, FULL, DL by validation CRPS (contract 5.3)
    # 'ACH' is the summary of the choice, never a candidate for it
    cands = {k: v['val_crps'] for k, v in rec.items()
             if k != 'ACH' and isinstance(v, dict) and 'val_crps' in v}
    if cands:
        best = min(cands, key=cands.get)
        rec['ACH'] = dict(arm=best, val_crps=cands[best], candidates=cands)
        print(f'{key} ACH = {best} (val {cands[best]:.5f}) among {", ".join(f"{k} {v:.5f}" for k, v in cands.items())}',
              flush=True)
    state[key] = rec; _save(state)
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='all')
    p.add_argument('--period', default='all', choices=['P1', 'P2', 'all'])
    p.add_argument('--only', default=None, choices=['f0', 'lora', 'full', 'dl'])
    a = p.parse_args()
    cfgs = feasible_configs() if a.config == 'all' else [ga.BY_ID[a.config]]
    periods = ['P1', 'P2'] if a.period == 'all' else [a.period]
    t0 = time.perf_counter()
    for cfg in cfgs:
        for name in periods:
            one(cfg, name, only=a.only)
    steps, fits = arms.optimizer_updates()
    print(f'done in {(time.perf_counter()-t0)/60:.1f} min; ledger {fits} fits, {steps} optimizer steps')


if __name__ == '__main__':
    main()
