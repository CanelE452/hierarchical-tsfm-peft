"""Bounded, TRAIN/VAL-only primitives for the approved past-only diagnosis."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import sys
import time

import numpy as np
import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / '.cache' / HERE.name
SEEDS = [92601, 92602]
L, H = 512, 48
LAMBDAS = [1e-4, 1e-2, 1., 1e2, 1e4]
DIRECT = ROOT / 'research/professor_compressor_direct_20261011'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def ahash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def jsonable(x):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value, replace=False):
    path = Path(path).resolve()
    if not (path.is_relative_to(HERE) or path.is_relative_to(CACHE)):
        raise ValueError('Writes outside the new campaign are forbidden')
    path.parent.mkdir(parents=True, exist_ok=True)
    s = json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if replace:
        tmp = path.with_suffix(f'.{os.getpid()}.tmp')
        tmp.write_text(s, encoding='utf-8')
        for attempt in range(10):
            try:
                tmp.replace(path)
                break
            except PermissionError:
                if attempt==9:
                    raise
                time.sleep(.05)
    else:
        with path.open('x', encoding='utf-8') as f:
            f.write(s)


def select_origins(values, maximum):
    values = np.asarray(values, dtype=np.int64)
    if len(values) <= maximum:
        return values
    return values[np.linspace(0, len(values)-1, maximum).astype(int)]


def access_guard():
    protected = set()
    for dataset in ('jena', 'robin'):
        c = read(ROOT / f'research/level_matched_learned_ed_v18_20261006/data_contract_{dataset}.json')
        protected.add((ROOT / c['test']['path']).resolve())
    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if path in protected:
                raise PermissionError('Protected TEST access denied')
    sys.addaudithook(audit)
    return [str(p) for p in protected]


class BudgetHeld(RuntimeError):
    pass


class Budget:
    def __init__(self, phase, gpu=False, label=''):
        self.phase, self.gpu, self.label = phase, gpu, label
        self.path = HERE / 'resources' / f'{os.getpid()}_{time.time_ns()}.json'
        self.start = time.perf_counter()
        self.rss = 0
        self.proc = psutil.Process()
        self.record = {'phase': phase, 'gpu': gpu, 'label': label, 'pid': os.getpid(),
                       'started_unix': time.time(), 'wall_seconds': 0., 'status': 'RUNNING_CONFIRMED',
                       'sampled_peak_rss': 0, 'peak_allocated_vram': 0, 'peak_reserved_vram': 0}
        self.check()

    def check(self):
        self.record['wall_seconds'] = time.perf_counter()-self.start
        self.rss = max(self.rss, self.proc.memory_info().rss)
        self.record['sampled_peak_rss'] = self.rss
        total = self.record['wall_seconds']
        for p in (HERE / 'resources').glob('*.json'):
            if p == self.path:
                continue
            r = read(p)
            if r['gpu'] == self.gpu and (not self.gpu or r['phase'] == self.phase):
                total += r['wall_seconds']
        if self.gpu:
            import torch
            self.record['peak_allocated_vram'] = torch.cuda.max_memory_allocated()
            self.record['peak_reserved_vram'] = torch.cuda.max_memory_reserved()
        write(self.path, self.record, replace=True)
        if total >= (5400 if self.gpu else 3600):
            raise BudgetHeld(f'Campaign phase/CPU wall exhausted: {total:.3f}s')
        if self.rss > 8*1024**3 or self.record['peak_allocated_vram'] > 8*1024**3:
            raise BudgetHeld('Existing 8GiB process RAM/allocated VRAM protection')

    def __enter__(self):
        return self

    def __exit__(self, kind, value, tb):
        self.record['wall_seconds'] = time.perf_counter()-self.start
        self.record['status'] = 'COMPLETE' if kind is None else 'FAILED_PRESERVED'
        self.record['failure_reason'] = None if value is None else str(value)
        write(self.path, self.record, replace=True)


def load_data(dataset, frozen=False, sizes=(256, 64, None)):
    cpath = ROOT / f'research/level_matched_learned_ed_v18_20261006/data_contract_{dataset}.json'
    c = read(cpath)
    path = ROOT / c['trainval']['path']
    if sha(path) != c['trainval']['sha256']:
        raise ValueError('TRAIN/VAL source bytes differ')
    with np.load(path, allow_pickle=False) as z:
        p = {k: z[k] for k in ('x', 'finite', 'mean', 'std', 'times', 'columns', 'val_origins')}
    physical = (p['x'] * p['std'] + p['mean']).astype(np.float32)
    physical[~p['finite']] = np.nan
    train = c['bounds']['train']
    cut = train[0] + int(.8*(train[1]-train[0]))
    fit, tune, val = (train[0], cut), (cut, train[1]), tuple(c['bounds']['val'])
    stats_end = train[1] if frozen else cut
    mean = np.nanmean(physical[train[0]:stats_end], axis=0).astype(np.float32)
    std = np.nanstd(physical[train[0]:stats_end], axis=0).astype(np.float32)
    std = np.where(std > 1e-6, std, 1.).astype(np.float32)
    if not np.isfinite(mean).all():
        raise ValueError('No observed fitting value in a channel')
    filled = np.empty_like(physical)
    last = mean.copy()
    for t in range(len(physical)):
        observed = p['finite'][t]
        last[observed] = physical[t, observed]
        filled[t] = last
    x = ((filled-mean)/std).astype(np.float32)
    origins = {'fit': select_origins(np.arange(fit[0]+L, fit[1]-H+1), sizes[0]),
               'tune': select_origins(np.arange(tune[0], tune[1]-H+1), sizes[1]),
               'val': p['val_origins'].astype(np.int64)}
    if sizes[2] is not None:
        origins['val'] = select_origins(origins['val'], sizes[2])
    for split, oo in origins.items():
        bounds = {'fit': fit, 'tune': tune, 'val': val}[split]
        if not len(oo) or np.any(oo < bounds[0]) or np.any(oo+H > bounds[1]) or np.any(oo-L < 0):
            raise ValueError(f'Illegal target/history bounds: {split}')
    return {'dataset': dataset, 'x': x, 'physical': physical, 'filled': filled, 'mask': p['finite'],
            'mean': mean, 'std': std, 'origins': origins, 'fit_bounds': fit, 'tune_bounds': tune,
            'val_bounds': val, 'columns': p['columns'].tolist(), 'times': p['times'],
            'interval_seconds': c['interval_seconds'], 'source': str(path), 'source_hash': sha(path),
            'normalization_fit_scope': 'TRAIN_ALL_FROZEN_DIAGNOSTIC' if frozen else 'FIT_ONLY',
            'raw_recovery_note': 'FP32 inverse of stored observed TRAIN/VAL; rounding retained; missing mask preserved'}


def windows(data, split, target=False):
    return np.stack([data['x'][o:o+H] if target else data['x'][o-L:o]
                     for o in data['origins'][split]]).astype(np.float32)


def masks(data, split, target=True):
    return np.stack([data['mask'][o:o+H] if target else data['mask'][o-L:o]
                     for o in data['origins'][split]])


def score(pred, target, mask):
    if pred.shape != target.shape or mask.shape != target.shape:
        raise ValueError('Shape mismatch')
    if not np.isfinite(pred).all() or not np.isfinite(target[mask]).all():
        raise ValueError('Non-finite science values')
    e = np.where(mask, np.asarray(pred, dtype=np.float64)-target, 0.)
    count = mask.sum(axis=(0, 1))
    cmse = np.divide((e*e).sum((0, 1)), count, out=np.zeros_like(count, dtype=float), where=count>0)
    cmae = np.divide(np.abs(e).sum((0, 1)), count, out=np.zeros_like(count, dtype=float), where=count>0)
    result = {'mse': float(cmse[count>0].mean()), 'mae': float(cmae[count>0].mean()),
              'per_channel_mse': cmse.tolist(), 'per_channel_mae': cmae.tolist(),
              'observed_counts': count.tolist(), 'horizon_metrics': {}}
    if pred.shape[1] == H:
        for name, lo, hi in [('1_8',0,8),('9_24',8,24),('25_48',24,48)]:
            result['horizon_metrics'][name] = score(pred[:,lo:hi],target[:,lo:hi],mask[:,lo:hi])
    return result


def physical_score(data, prediction, split):
    target = windows(data, split, True)
    mask = masks(data, split)
    result = score(prediction*data['std']+data['mean'], target*data['std']+data['mean'], mask)
    result['normalized'] = score(prediction, target, mask)
    result['scale_note'] = 'Physical channel macro within dataset; heterogeneous channel units, never pooled across datasets'
    return result


class Ridge:
    def __init__(self, lam):
        self.lam = lam

    def fit(self, x, y, valid=None):
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        if valid is not None and not np.all(valid):
            raise ValueError('Use one observed-output group at a time for missing target fits')
        self.mean = x.mean(0)
        self.std = np.maximum(x.std(0), 1e-12)
        z = (x-self.mean)/self.std
        self.ym = y.mean(0)
        self.train = z
        if z.shape[1] <= len(z):
            self.coef = np.linalg.solve(z.T@z + len(z)*self.lam*np.eye(z.shape[1]), z.T@(y-self.ym))
            self.dual = None
        else:
            self.coef = None
            self.dual = np.linalg.solve(z@z.T + len(z)*self.lam*np.eye(len(z)), y-self.ym)
        return self

    def predict(self, x):
        z = (np.asarray(x)-self.mean)/self.std
        return z@self.coef+self.ym if self.coef is not None else z@self.train.T@self.dual+self.ym


def ridge_grouped(xfit, yfit, mfit, xtune, ytune, mtune, xeval, score_scale=None, budget=None):
    shape = yfit.shape[1:]
    yf, mf = yfit.reshape(len(yfit),-1), mfit.reshape(len(yfit),-1)
    groups = {}
    for j in range(yf.shape[1]):
        key = mf[:,j].tobytes()
        groups.setdefault(key, []).append(j)
    options = []
    for lam in LAMBDAS:
        pt = np.zeros_like(ytune.reshape(len(ytune),-1), dtype=float)
        pe = np.zeros((len(xeval), yf.shape[1]))
        fits = []
        for js in groups.values():
            rows = mf[:,js[0]]
            if rows.sum() < 2:
                raise ValueError('Insufficient observed FIT outputs')
            model = Ridge(lam).fit(xfit[rows], yf[rows][:,js])
            pt[:,js] = model.predict(xtune)
            pe[:,js] = model.predict(xeval)
            fits.append((js, model))
        pt = pt.reshape(ytune.shape)
        scale = 1 if score_scale is None else np.asarray(score_scale)
        s = score(pt*scale, ytune*scale, mtune)['mse']
        options.append((s, lam, pe.reshape((len(xeval),)+shape), fits))
        if budget:
            budget.check()
    chosen = min(range(len(options)), key=lambda i: options[i][0])
    return options[chosen][2], {'lambda': options[chosen][1], 'tune_scores': [v[0] for v in options],
            'ridge_definition': 'mean output SSE + lambda ||w||^2; unpenalized intercept; alpha=n_observed*lambda',
            'feature_scaler_scope': 'FIT_ONLY_PER_OBSERVED_OUTPUT_GROUP', 'fit_groups': len(groups)}, options[chosen][3]
