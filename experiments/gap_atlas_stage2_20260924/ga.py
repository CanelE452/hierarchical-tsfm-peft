#!/usr/bin/env python
"""Core library for gap atlas Stage 2 (contract 00_MASTER_CLI_gap_atlas_stage2_20260924.txt).

The official GIFT-Eval loader is vendored as gift_eval_data.py (byte-identical to
SalesforceAIResearch/gift-eval src/gift_eval/data.py) because installing the published package
would pin numpy 1.26 / datasets 2.17 and break this environment. Evaluation follows the official
notebook notebooks/chronos-2.ipynb: quantiles 0.1-0.9, cross-learning on, float32, batch 100,
one rolling window at a time.
"""
from __future__ import annotations
import hashlib, json, os, time, urllib.request
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'results'/'gap_atlas_stage2_20260924'
CACHE = ROOT/'.cache'/'gap_atlas_stage2_20260924'
DATA = ROOT/'data'/'gift_eval'
SCRATCH = Path(r'C:/Users/User/AppData/Local/Temp/claude/E--CODING-proj-hierarchical-tsfm-peft/68839535-3971-4221-82f1-10ee16fe17b4/scratchpad/gatlas')
for p in (OUT, CACHE, SCRATCH): p.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('GIFT_EVAL', str(DATA))

REVISION = '29ec3766d36d6f73f0696f85560a422f50e8498c'
MODEL = 'amazon/chronos-2'
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
BATCH = 100                      # official notebook; cross-learning results depend on this
# Contract 5.2 writes the feasibility rule as "context length + 5H" without giving a number. The model's
# own default is 8192 (config.json chronos_config.context_length), but fitting at 8192 peaks at
# 12.6 GiB on this 12 GiB card at the contract's batch size of 32 and thrashes (2.6-3.3 s/step versus
# 0.40 s/step at 2048, measured by bench_context.py). Fits therefore run at 2048, and the feasibility
# rule uses that same number so the two stay consistent. The feasible set is the same as at 512.
CONTEXT_FIT = 2048
CONTEXT_MODEL_DEFAULT = 8192
FIT_STEPS = 1000
CKPT_EVERY = 250
GITHUB = 'https://raw.githubusercontent.com/SalesforceAIResearch/gift-eval/main'

# contract section 4: four candidates then three negative controls
CONFIGS = [
    dict(id='C1', role='candidate', hf='bitbrains_rnd/5T',          term='medium', key='bitbrains_rnd/5T/medium'),
    dict(id='C2', role='candidate', hf='bitbrains_rnd/H',           term='short',  key='bitbrains_rnd/H/short'),
    dict(id='C3', role='candidate', hf='bitbrains_fast_storage/H',  term='short',  key='bitbrains_fast_storage/H/short'),
    dict(id='C4', role='candidate', hf='bizitobs_l2c/5T',           term='medium', key='bizitobs_l2c/5T/medium'),
    dict(id='N1', role='control',   hf='bitbrains_rnd/5T',          term='short',  key='bitbrains_rnd/5T/short'),
    dict(id='N2', role='control',   hf='bizitobs_l2c/H',            term='medium', key='bizitobs_l2c/H/medium'),
    dict(id='N3', role='control',   hf='electricity/H',             term='short',  key='electricity/H/short'),
]
BY_ID = {c['id']: c for c in CONFIGS}


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float)+'\n', encoding='utf-8')


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=float).encode()).hexdigest()


# ----------------------------------------------------------------- leaderboard reference
def leaderboard(model: str):
    """results/<model>/all_results.csv from the gift-eval repository, cached with its hash."""
    dest = CACHE/'leaderboard'/f'{model}.csv'
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(f'{GITHUB}/results/{model}/all_results.csv',
                                     headers={'User-Agent': 'Mozilla/5.0 (research; gap-atlas)'})
        with urllib.request.urlopen(req, timeout=60) as r: dest.write_bytes(r.read())
    import pandas as pd
    return pd.read_csv(dest).set_index('dataset'), sha(dest)


CRPS_COL = 'eval_metrics/mean_weighted_sum_quantile_loss'
MASE_COL = 'eval_metrics/MASE[0.5]'


# ----------------------------------------------------------------- official dataset and periods
def dataset(cfg, to_univariate=False):
    from gift_eval_data import Dataset
    return Dataset(name=cfg['hf'], term=cfg['term'], to_univariate=to_univariate)


def geometry(cfg):
    """Official window geometry plus the contract's P1/P2 split (section 5.1-5.2)."""
    ds = dataset(cfg)
    H, W = ds.prediction_length, ds.windows
    V = min(3, W)                                     # validation block, contract 5.1
    minlen = int(ds._min_series_length)
    # P2 = last W windows; P1 = the W windows before it; each period keeps its own validation block
    train_len_p1 = minlen - (2*W + V)*H
    train_len_p2 = minlen - (W + V)*H
    need = CONTEXT_FIT + 5*H
    return dict(id=cfg['id'], key=cfg['key'], role=cfg['role'], freq=str(ds.freq),
                prediction_length=H, windows=W, validation_windows=V,
                series=int(len(ds.hf_dataset)), target_dim=int(ds.target_dim),
                min_series_length=minlen, sum_series_length=int(ds.sum_series_length),
                train_length_P1=int(train_len_p1), train_length_P2=int(train_len_p2),
                required_train_length=int(need),
                required_train_length_strict=int(CONTEXT_MODEL_DEFAULT + 5*H),
                feasible=bool(train_len_p1 >= need),
                feasible_strict=bool(train_len_p1 >= CONTEXT_MODEL_DEFAULT + 5*H),
                note='P1 needs 2W+V windows held out; P2 needs W+V. INFEASIBLE if the P1 training '
                     'segment is shorter than context + 5H (contract 5.2)')


# ----------------------------------------------------------------- official predictor (notebook)
class Chronos2Predictor:
    """Copy of notebooks/chronos-2.ipynb Chronos2Predictor. `predict_batches_jointly` was renamed to
    `cross_learning` in chronos-forecasting 2.3.2 (pipeline.py:570-577 maps the old name)."""

    def __init__(self, pipeline, prediction_length: int, batch_size: int = BATCH,
                 quantile_levels=QUANTILES, cross_learning: bool = True,
                 context_length: int | None = None):
        self.pipeline = pipeline
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.quantile_levels = list(quantile_levels)
        self.cross_learning = cross_learning
        self.context_length = context_length      # None keeps the official default (F0 always does)

    def predict(self, test_data_input):
        import torch
        from gluonts.model.forecast import QuantileForecast
        inputs = [{'target': item['target']} for item in test_data_input]
        is_univariate = inputs[0]['target'].ndim == 1
        bs = self.batch_size
        while True:
            try:
                extra = {} if self.context_length is None else dict(context_length=self.context_length)
                quantiles, _ = self.pipeline.predict_quantiles(
                    inputs=inputs, prediction_length=self.prediction_length,
                    quantile_levels=self.quantile_levels, batch_size=bs,
                    cross_learning=self.cross_learning, **extra)
                break
            except torch.cuda.OutOfMemoryError:
                bs //= 2
                if bs < 1: raise
                print(f'  OOM -> batch_size {bs}', flush=True)
        q = torch.stack(quantiles).permute(0, 3, 2, 1).float().cpu().numpy()
        if is_univariate: q = q.squeeze(-1)
        assert q.shape[1] == len(self.quantile_levels) and q.shape[2] == self.prediction_length
        out = []
        for arr, ts in zip(q, test_data_input):
            out.append(QuantileForecast(forecast_arrays=arr,
                                        forecast_keys=[str(x) for x in self.quantile_levels],
                                        start_date=ts['start'] + len(ts['target'])))
        return out


def pipeline(finetuned: Path | None = None, dtype='float32'):
    """chronos already folds a LoRA adapter into the base weights when it loads one
    (chronos/chronos2/pipeline.py:1177 calls merge_and_unload), so a fitted arm and F0 run the same
    architecture; nothing extra is needed here."""
    import torch
    from chronos import Chronos2Pipeline
    kw = dict(device_map='cuda', torch_dtype=getattr(torch, dtype))
    if finetuned is None:
        kw['revision'] = REVISION
    else:
        ensure_pinned_base(finetuned)
    return Chronos2Pipeline.from_pretrained(str(finetuned) if finetuned else MODEL, **kw)


def ensure_pinned_base(model_dir: Path):
    """`fit` saves an adapter whose config has revision null, which would resolve the base model to
    the hub's current main (peft/auto.py:115,156). Pin it to the revision F0 uses."""
    cfg_path = Path(model_dir)/'adapter_config.json'
    if not cfg_path.exists(): return                     # full finetune saves a complete model
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    if cfg.get('base_model_name_or_path') != MODEL:
        raise SystemExit(f'UNEXPECTED_BASE {model_dir}: {cfg.get("base_model_name_or_path")}')
    if cfg.get('revision') != REVISION:
        cfg['revision'] = REVISION
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding='utf-8')


# ----------------------------------------------------------------- official metric computation
def metric_list():
    from gluonts.ev.metrics import (MAE, MAPE, MASE, MSE, MSIS, ND, NRMSE, RMSE, SMAPE,
                                    MeanWeightedSumQuantileLoss)
    return [MSE(forecast_type='mean'), MSE(forecast_type=0.5), MAE(), MASE(), MAPE(), SMAPE(),
            MSIS(), RMSE(), NRMSE(), ND(), MeanWeightedSumQuantileLoss(quantile_levels=QUANTILES)]


def evaluate(forecasts, test_data, freq, axis=None):
    from gluonts.model import evaluate_forecasts
    from gluonts.time_feature import get_seasonality
    return evaluate_forecasts(forecasts, test_data=test_data, metrics=metric_list(),
                              batch_size=1024, axis=axis, mask_invalid_label=True,
                              allow_nan_forecast=False, seasonality=get_seasonality(freq))


def forecast_all(predictor, test_data, n_windows: int):
    """Predict one rolling window at a time so cross-learning cannot see other windows of the same
    series (official notebook comment 'Avoid cross batch leakage of rolling evaluation')."""
    entries = list(test_data.input)          # materialised once; islice per window re-generates it
    per_window = [list(predictor.predict(entries[w::n_windows])) for w in range(n_windows)]
    return [item for items in zip(*per_window) for item in items]


# ----------------------------------------------------------------- scores and uncertainty
def per_unit_terms(forecasts, test_data, n_windows: int, steps_per_day: int, quantiles=QUANTILES):
    """Numerator and denominator of the official CRPS, split by (series, day block) so the aggregate can
    be bootstrapped either over series or over time blocks (contract 5.4). gluonts computes
    mean_q[ sum(2|err*(1{pred>=y}-q)|) / sum(|y|) ] (ev/metrics.py, ev/stats.py) and masks labels that
    are not finite, matching mask_invalid_label=True.

    Returns num[u, q], den[u], series_of[u], day_of[u].
    """
    labels = [lab for _, lab in test_data]
    assert len(labels) == len(forecasts), f'{len(labels)} labels vs {len(forecasts)} forecasts'
    steps_per_day = max(int(steps_per_day), 1)
    num, den, series_of, day_of = [], [], [], []
    for k, (fc, lab) in enumerate(zip(forecasts, labels)):
        i = k//n_windows                                   # series-major order, W windows per series
        y = np.asarray(lab['target'], dtype=float)
        if y.ndim == 2: y = y.T                            # [variates, H] -> [H, variates]
        y = y[:, None] if y.ndim == 1 else y
        mask = np.isfinite(y)
        per_q = np.empty((y.shape[0], len(quantiles)))
        for j, q in enumerate(quantiles):
            p = np.asarray(fc.quantile(q), dtype=float)
            p = p[:, None] if p.ndim == 1 else p
            per_q[:, j] = np.where(mask, 2*np.abs((p - y)*((p >= y).astype(float) - q)), 0.0).sum(axis=1)
        abs_y = np.where(mask, np.abs(y), 0.0).sum(axis=1)
        for d0 in range(0, y.shape[0], steps_per_day):
            sl = slice(d0, d0 + steps_per_day)
            num.append(per_q[sl].sum(axis=0)); den.append(float(abs_y[sl].sum()))
            series_of.append(i); day_of.append((k % n_windows)*((y.shape[0]+steps_per_day-1)//steps_per_day)
                                               + d0//steps_per_day)
    return np.array(num), np.array(den), np.array(series_of), np.array(day_of)


def unit_blocks(series_of: np.ndarray, day_of: np.ndarray, n_series: int):
    """Contract 5.4: resample series when there are at least five, otherwise one-day time blocks."""
    if n_series >= 5:
        return [np.flatnonzero(series_of == i) for i in np.unique(series_of)], 'series'
    return [np.flatnonzero(day_of == d) for d in np.unique(day_of)], 'one-day time blocks'


def crps_from_terms(num: np.ndarray, den: np.ndarray, idx=None) -> float:
    """Aggregate CRPS from unit terms: mean over quantiles of sum(quantile loss)/sum(|y|)."""
    if idx is None: idx = np.arange(len(den))
    d = float(den[idx].sum())
    if d <= 0: return float('nan')
    return float(np.mean(num[idx].sum(axis=0)/d))


def bootstrap_gap_terms(num_f0, den_f0, num_ach, den_ach, blocks, draws=2000, seed=20260924):
    """Bootstrap the gap by resampling whole blocks (series, or one-day time blocks)."""
    rng = np.random.default_rng(seed)
    n = len(blocks)
    if n < 2:
        return dict(value=gap(crps_from_terms(num_f0, den_f0), crps_from_terms(num_ach, den_ach)),
                    n_units=n, ci=[float('nan'), float('nan')],
                    note='too few resampling units for an interval')
    stats = np.empty(draws)
    for d in range(draws):
        pick = np.concatenate([blocks[j] for j in rng.integers(n, size=n)])
        stats[d] = gap(crps_from_terms(num_f0, den_f0, pick), crps_from_terms(num_ach, den_ach, pick))
    return dict(value=gap(crps_from_terms(num_f0, den_f0), crps_from_terms(num_ach, den_ach)),
                n_units=int(n),
                ci=[float(np.nanquantile(stats, .025)), float(np.nanquantile(stats, .975))])


def gap(score_f0: float, score_ach: float, noise_floor: float = 1e-12):
    """G = (CRPS_F0 - CRPS_ACH)/CRPS_F0; positive means training helped (contract 5.4).
    Returns nan when the denominator is at noise level (gate 5)."""
    if not np.isfinite(score_f0) or abs(score_f0) <= noise_floor: return float('nan')
    return float((score_f0 - score_ach)/score_f0)


def timer():
    t0 = time.perf_counter()
    return lambda: time.perf_counter() - t0
