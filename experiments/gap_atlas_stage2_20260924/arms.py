#!/usr/bin/env python
"""Arms of contract section 5.3, trained and selected separately inside each period.

F0      Chronos-2 zero-shot with the official notebook settings
LORA    Chronos-2 fit(finetune_mode='lora'), checkpoints every 250 steps
FULL    Chronos-2 fit(finetune_mode='full'), learning rate 1e-6
DL      neuralforecast PatchTST and NHITS, early stopping on the validation block
SNAIVE  statsforecast seasonal naive through the official wrapper (report only)

Selection uses the period's validation block only. Nothing here touches the evaluation window; the
evaluation window is scored by score.py after SEAL.json exists.
"""
from __future__ import annotations
import json, shutil, time
from pathlib import Path
import numpy as np
import ga
from periods import Period

CKPTS = [250, 500, 750, 1000]
FIT_COMMON = dict(num_steps=ga.FIT_STEPS, batch_size=32, validation_inputs=None,
                  context_length=ga.CONTEXT_FIT)
LEDGER = ga.CACHE/'fit_ledger.json'


def _ledger_add(key, rec):
    led = ga.read_json(LEDGER) if LEDGER.exists() else {}
    led[key] = rec
    ga.write_json(LEDGER, led)


def optimizer_updates():
    led = ga.read_json(LEDGER) if LEDGER.exists() else {}
    return sum(v.get('steps', 0) for v in led.values()), len(led)


def _inputs(series_entries):
    """fit and predict take the same formats; keep multivariate sources multivariate."""
    return [{'target': np.asarray(e['target'], dtype=np.float32)} for e in series_entries]


# ----------------------------------------------------------------- Chronos-2 arms
def fit_chronos(cfg, period_name, mode: str):
    """One fit per (configuration, period, mode) with checkpoints at 250/500/750/1000."""
    key = f"{cfg['id']}_{period_name}_{mode}"
    out_dir = ga.CACHE/'fits'/key
    led = ga.read_json(LEDGER) if LEDGER.exists() else {}
    if key in led and (out_dir/'done.json').exists():
        return out_dir
    if out_dir.exists(): shutil.rmtree(out_dir)
    import torch
    per = Period(cfg, period_name)
    series = _inputs(per.training_series())
    pipe = ga.pipeline(dtype='float32')
    torch.cuda.reset_peak_memory_stats(); el = ga.timer()
    torch.manual_seed(0)
    kw = dict(FIT_COMMON)
    kw.update(finetune_mode=mode, learning_rate=1e-5 if mode == 'lora' else 1e-6)
    if mode == 'lora': kw['lora_config'] = None
    # chronos sets save_total_limit=1 by default (chronos/chronos2/pipeline.py:285), which deletes the
    # intermediate checkpoints the contract selects among; keep all four.
    pipe.fit(series, prediction_length=per.H, output_dir=str(out_dir),
             save_strategy='steps', save_steps=ga.CKPT_EVERY, save_total_limit=len(CKPTS),
             seed=0, **kw)
    secs, peak = el(), torch.cuda.max_memory_allocated()/2**30
    del pipe; torch.cuda.empty_cache()
    ckpts = sorted(p.name for p in out_dir.glob('checkpoint-*'))
    (out_dir/'done.json').write_text(json.dumps(dict(checkpoints=ckpts)), encoding='utf-8')
    _ledger_add(key, dict(config=cfg['key'], period=period_name, mode=mode, steps=ga.FIT_STEPS,
                          seconds=round(secs, 1), peak_vram_gib=round(float(peak), 2),
                          series=len(series), prediction_length=per.H,
                          learning_rate=kw['learning_rate'], batch_size=kw['batch_size'],
                          checkpoints=ckpts, path=str(out_dir.relative_to(ga.ROOT))))
    print(f'fit {key}: {secs:.0f}s peak {peak:.2f} GiB ckpts {len(ckpts)}', flush=True)
    return out_dir


def score_chronos_checkpoints(cfg, period_name, mode: str):
    """Validation CRPS for each checkpoint; the chosen one is recorded, nothing is scored on TEST."""
    out_dir = ga.CACHE/'fits'/f"{cfg['id']}_{period_name}_{mode}"
    per = Period(cfg, period_name)
    val = per.validation_data()
    def val_crps(path, ctx):
        pipe = ga.pipeline(path, dtype='float32')
        pred = ga.Chronos2Predictor(pipe, prediction_length=per.H, context_length=ctx)
        fc = ga.forecast_all(pred, val, per.V)
        res = ga.evaluate(fc, val, per.freq).reset_index(drop=True).to_dict('records')[0]
        del pipe
        import torch; torch.cuda.empty_cache()
        return float(res['mean_weighted_sum_quantile_loss'])

    scores = {}
    for name in sorted(p.name for p in out_dir.glob('checkpoint-*')):
        step = int(name.split('-')[1])
        if step not in CKPTS: continue
        scores[step] = val_crps(out_dir/name, ga.CONTEXT_FIT)
        print(f'  val {cfg["id"]}_{period_name}_{mode} ckpt {step} ctx {ga.CONTEXT_FIT}: {scores[step]:.5f}',
              flush=True)
    best = min(scores, key=scores.get)
    # the arms are fitted at CONTEXT_FIT while F0 predicts at the package default; which context the
    # fitted model should predict with is decided on the validation block, never on the test windows
    by_ctx = {ga.CONTEXT_FIT: scores[best]}
    if ga.CONTEXT_MODEL_DEFAULT != ga.CONTEXT_FIT:
        by_ctx[ga.CONTEXT_MODEL_DEFAULT] = val_crps(out_dir/f'checkpoint-{best}', ga.CONTEXT_MODEL_DEFAULT)
        print(f'  val {cfg["id"]}_{period_name}_{mode} ckpt {best} ctx {ga.CONTEXT_MODEL_DEFAULT}: '
              f'{by_ctx[ga.CONTEXT_MODEL_DEFAULT]:.5f}', flush=True)
    best_ctx = min(by_ctx, key=by_ctx.get)
    return dict(arm=mode.upper(), val_scores=scores, selected_checkpoint=best,
                selected_path=str((out_dir/f'checkpoint-{best}').relative_to(ga.ROOT)),
                val_scores_by_context=by_ctx, selected_context=best_ctx,
                val_crps=by_ctx[best_ctx])


# ----------------------------------------------------------------- deep-learning arm
def impute(y: np.ndarray) -> np.ndarray:
    """Last-value imputation, the same policy the official GIFT-Eval statistical baseline applies
    (LastValueImputation in gift_eval_statsforecast.py). neuralforecast refuses missing values while
    Chronos-2 consumes them natively, so this is only applied to the deep-learning arm and is recorded
    as a deviation: on these datasets 16-20% of the training points are missing."""
    y = np.asarray(y, dtype=np.float64).copy()
    bad = ~np.isfinite(y)
    if bad.all(): return np.zeros_like(y)
    idx = np.where(~bad, np.arange(len(y)), 0)
    np.maximum.accumulate(idx, out=idx)
    y = y[idx]
    first = int(np.argmax(~bad))
    y[:first] = y[first]                      # leading gap takes the first observed value
    return y


def _long_frame(entries, freq):
    import pandas as pd
    rows = []
    for i, e in enumerate(entries):
        t = np.asarray(e['target'], dtype=np.float32)
        t = t[None, :] if t.ndim == 1 else t
        start = e['start']
        for v in range(t.shape[0]):
            rows.append(pd.DataFrame(dict(
                unique_id=f'{i}_{v}',
                ds=pd.date_range(start=start.to_timestamp(), periods=t.shape[1], freq=start.freq),
                y=impute(t[v]))))
    return pd.concat(rows, ignore_index=True)


def fit_dl(cfg, period_name, model_name: str, max_steps=1000):
    """PatchTST or NHITS on the period's training segment, early stopping on the validation block.
    neuralforecast is univariate here: multivariate sources are expanded to one series per variate,
    which is what the leaderboard's own deep-learning entries do."""
    import torch
    from neuralforecast import NeuralForecast
    from neuralforecast.models import PatchTST, NHITS
    from neuralforecast.losses.pytorch import MQLoss
    key = f"{cfg['id']}_{period_name}_DL-{model_name}"
    per = Period(cfg, period_name)
    df = _long_frame(per.training_series(), per.freq)
    input_size = min(5*per.H, 512)
    cls = dict(PatchTST=PatchTST, NHITS=NHITS)[model_name]
    el = ga.timer()
    model = cls(h=per.H, input_size=input_size, loss=MQLoss(quantiles=ga.QUANTILES),
                max_steps=max_steps, val_check_steps=100, early_stop_patience_steps=3,
                scaler_type='standard', random_seed=0, enable_progress_bar=False,
                logger=False, enable_checkpointing=False)
    nf = NeuralForecast(models=[model], freq=per.freq)
    nf.fit(df=df, val_size=per.V*per.H, verbose=False)
    secs = el()
    _ledger_add(key, dict(config=cfg['key'], period=period_name, mode=f'DL-{model_name}',
                          steps=max_steps, seconds=round(secs, 1), series=int(df.unique_id.nunique()),
                          input_size=int(input_size), prediction_length=per.H,
                          note='neuralforecast defaults except h, input_size, loss, max_steps, '
                               'early stopping on the validation block'))
    print(f'fit {key}: {secs:.0f}s', flush=True)
    return nf


QCOLS = ['-lo-80.0', '-lo-60.0', '-lo-40.0', '-lo-20.0', '-median',
         '-hi-20.0', '-hi-40.0', '-hi-60.0', '-hi-80.0']       # MQLoss order == quantiles 0.1 .. 0.9


def dl_forecast(nf, model_name, entries, freq, H, n_windows, drop_from_end, target_dim):
    """Roll the fitted model over the period's windows, feeding only values before each window.
    Returns gluonts forecasts in the same flat order the official test_data uses (series-major)."""
    from gluonts.model.forecast import QuantileForecast
    per_window = {}
    for w in range(n_windows):
        cut = drop_from_end + (n_windows - w)*H
        hist = [dict(e, target=np.asarray(e['target'])[..., :np.asarray(e['target']).shape[-1]-cut])
                for e in entries]
        pred = nf.predict(df=_long_frame(hist, freq)).reset_index()
        arrays = {}
        for uid, grp in pred.groupby('unique_id', sort=False):
            grp = grp.sort_values('ds')
            arrays[uid] = np.stack([grp[f'{model_name}{c}'].to_numpy() for c in QCOLS], axis=0)
        start = {i: hist[i]['start'] + np.asarray(hist[i]['target']).shape[-1] for i in range(len(hist))}
        per_window[w] = (arrays, start)
    out = []
    for i in range(len(entries)):
        for w in range(n_windows):
            arrays, start = per_window[w]
            if target_dim == 1:
                fa = arrays[f'{i}_0']
            else:
                fa = np.stack([arrays[f'{i}_{v}'] for v in range(target_dim)], axis=-1)
            out.append(QuantileForecast(forecast_arrays=fa,
                                        forecast_keys=[str(q) for q in ga.QUANTILES],
                                        start_date=start[i]))
    return out
