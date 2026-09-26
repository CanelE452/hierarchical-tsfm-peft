import argparse
import json
import time

import numpy as np
from threadpoolctl import threadpool_limits

from model import pca_basis
from run import CACHE, HERE, digest, load_data, phase_sample, save_json, score_arrays, source_receipt


def windows(values, origins):
    x = np.stack([values[o - 512:o] for o in origins])
    y = np.stack([values[o:o + 48] for o in origins])
    return x, y


def ridge_fit(x, y, penalty):
    x = x.transpose(0, 2, 1).reshape(-1, 512).astype(np.float64)
    y = y.transpose(0, 2, 1).reshape(-1, 48).astype(np.float64)
    xmean, ymean = x.mean(0), y.mean(0)
    xc, yc = x - xmean, y - ymean
    covariance = xc.T @ xc / len(x)
    cross = xc.T @ yc / len(x)
    values, vectors = np.linalg.eigh(covariance)
    inv = 1 / np.maximum(values + penalty, 1e-9)
    weight = (vectors * inv) @ (vectors.T @ cross)
    return {'weight': weight, 'bias': ymean - xmean @ weight}


def ridge_predict(x, fit):
    return np.einsum('blc,lh->bhc', x, fit['weight']) + fit['bias'][None, :, None]


def compute(dataset='electricity'):
    if dataset != 'electricity':
        raise ValueError('Protected baseline fitting requires the final sealed protocol')
    out = HERE / 'linear_baselines.json'
    if out.exists():
        raise RuntimeError('Do not overwrite CPU baseline results')
    started = time.perf_counter()
    data = load_data(dataset)
    origins = phase_sample(data['train_origins'], 92601, 1)
    x, y = windows(data['x'], origins)
    # All development channels are fully observed; no masked ridge substitution.
    if not data['finite'][:int(data['train_end'])].all():
        raise ValueError('Closed-form baseline requires explicit missing-target handling')
    basis = pca_basis(data['x'][int(data['train_start']):int(data['train_end'])], 8)
    zx, zy = x @ basis, y @ basis
    rx, ry = x - zx @ basis.T, y - zy @ basis.T
    splits = {key: windows(data['x'], data[f'{key}_origins']) for key in ('val', 'dev')}
    records = []
    local = CACHE / 'baselines'
    local.mkdir(parents=True, exist_ok=True)
    for penalty in (0.001, 0.1, 10.0):
        common = ridge_fit(zx, zy, penalty)
        residual = ridge_fit(rx, ry, penalty)
        direct = ridge_fit(x, y, penalty)
        for arm in ('factor_linear', 'shared_linear'):
            scores = {}
            for split, (sx, sy) in splits.items():
                pred = (ridge_predict(sx @ basis, common) @ basis.T + ridge_predict(sx - (sx @ basis) @ basis.T, residual)) if arm == 'factor_linear' else ridge_predict(sx, direct)
                mask = np.stack([data['finite'][o:o + 48] for o in data[f'{split}_origins']])
                scores[split] = score_arrays(pred, sy, mask)
                np.savez_compressed(local / f'{arm}_{penalty:g}_{split}.npz', prediction=pred.astype(np.float32), origins=data[f'{split}_origins'])
            record = {'arm': arm, 'ridge_penalty': penalty, 'scores': scores}
            records.append(record)
        np.savez_compressed(local / f'linear_weights_{penalty:g}.npz', basis=basis, common_weight=common['weight'], common_bias=common['bias'], residual_weight=residual['weight'], residual_bias=residual['bias'], direct_weight=direct['weight'], direct_bias=direct['bias'])
    seasonal = {}
    for split, (sx, sy) in splits.items():
        pred = np.tile(sx[:, -24:, :], (1, 2, 1))
        mask = np.stack([data['finite'][o:o + 48] for o in data[f'{split}_origins']])
        seasonal[split] = score_arrays(pred, sy, mask)
        np.savez_compressed(local / f'seasonal_{split}.npz', prediction=pred, origins=data[f'{split}_origins'])
    selected = {arm: min([r for r in records if r['arm'] == arm], key=lambda r: r['scores']['val']['mse']) for arm in ('factor_linear', 'shared_linear')}
    save_json(out, {'status': 'complete', 'dataset': dataset, 'source': source_receipt(), 'data_sha256': digest(data['_path']), 'training_origin_count': len(origins), 'selection': 'minimum validation MSE, never dev', 'fits': 9, 'ridge_penalties': [0.001, 0.1, 10.0], 'records': records, 'selected': selected, 'seasonal': seasonal, 'elapsed_s': time.perf_counter() - started, 'scope': 'fixed train-PCA factor plus idiosyncratic linear ridge; shared temporal map, not official AdaPTS nor an optimized dynamic factor model; development only'})
    print(json.dumps({'elapsed_s': time.perf_counter() - started, 'selected': {a: {'penalty': r['ridge_penalty'], 'val_mse': r['scores']['val']['mse'], 'dev_mse': r['scores']['dev']['mse']} for a, r in selected.items()}, 'seasonal_dev': seasonal['dev']['mse']}), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        compute()
