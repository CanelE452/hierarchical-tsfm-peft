"""Masked residual Ridge probes with total-channel-count penalty scaling."""
from collections import defaultdict
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from research.raw_context_multiscale_20261018_v1.calibration import fit_alpha_diagnostic
from . import common


def summary(x):
    n, length, channels = x.shape
    if length != 512:
        raise ValueError('Registered 512-to-33 summaries required')
    return np.concatenate((x.reshape(n, 32, 16, channels).mean(2), x[:, -1:]), axis=1).astype(np.float64)


def ridge_fit(features, residual, mask, penalty, kind):
    """Fit SSE/N_c + lambda*||w||^2; b is unpenalized.

    Each output (h,c) has its own observed rows but shares the denominator
    N_c=sum_(i,h) mask[i,h,c]. Hence its normal-equation penalty is lambda*N_c,
    including when horizon observation counts differ.
    """
    f = np.asarray(features, np.float64)
    r = np.asarray(residual, np.float64)
    n, horizon, channels = r.shape
    flat = f.transpose(0, 2, 1).reshape(n, -1)
    if kind == 'ALL':
        matrices = [flat]
    elif kind == 'OWN':
        matrices = [f[:, :, c] for c in range(channels)]
    else:
        raise ValueError(kind)
    means = [a.mean(0) for a in matrices]
    scales = [np.where(a.std(0) > 1e-12, a.std(0), 1.) for a in matrices]
    z = [(a-mu)/sd for a, mu, sd in zip(matrices, means, scales)]
    counts = mask.sum((0, 1))
    groups = defaultdict(list)
    for c in range(channels):
        for h in range(horizon):
            observed = np.ascontiguousarray(mask[:, h, c])
            key = (0 if kind == 'ALL' else c, int(counts[c]), observed.tobytes())
            groups[key].append((h, c))
    dimension = z[0].shape[1]
    weights = np.zeros((horizon, channels, dimension), np.float64)
    intercept = np.zeros((horizon, channels), np.float64)
    factorizations = 0
    for (matrix, denominator, encoded_mask), outputs in groups.items():
        observed = np.frombuffer(encoded_mask, dtype=np.bool_)
        if not observed.any():
            continue
        design = z[matrix][observed]
        xmean = design.mean(0)
        centered = design-xmean
        targets = np.stack([r[observed, h, c] for h, c in outputs], axis=1)
        ymean = targets.mean(0)
        gram = centered.T @ centered
        gram.flat[::dimension+1] += float(penalty)*denominator
        factor = cho_factor(gram, lower=True, check_finite=True)
        beta = cho_solve(factor, centered.T @ (targets-ymean), check_finite=True)
        b = ymean-xmean @ beta
        for j, (h, c) in enumerate(outputs):
            weights[h, c] = beta[:, j]
            intercept[h, c] = b[j]
        factorizations += 1
    return dict(kind=kind, penalty=float(penalty), means=np.stack(means),
                scales=np.stack(scales), weights=weights, intercept=intercept,
                factorizations=factorizations, total_channel_counts=counts)


def ridge_predict(model, features):
    f = np.asarray(features, np.float64)
    n, _, channels = f.shape
    if model['kind'] == 'ALL':
        z = (f.transpose(0, 2, 1).reshape(n, -1)-model['means'][0])/model['scales'][0]
        return np.einsum('nd,hcd->nhc', z, model['weights'], optimize=True)+model['intercept']
    out = np.empty((n, *model['intercept'].shape), np.float64)
    for c in range(channels):
        z = (f[:, :, c]-model['means'][c])/model['scales'][c]
        out[:, :, c] = z @ model['weights'][:, c].T + model['intercept'][:, c]
    return out


def run(data, dataset):
    out = common.CACHE/'probes'/dataset
    out.mkdir(parents=True, exist_ok=True)
    f = summary(data['x'])
    fit, select = (data['split'][k] for k in ('fit_indices', 'select_indices'))
    y, mask, g = data['y'], data['mask'], data['g']
    r = np.where(mask, y-g, 0).astype(np.float64)
    rows = []
    selected = {}
    for kind in ('ZERO', 'MEAN', 'OWN', 'ALL'):
        began = time.perf_counter()
        if kind == 'ZERO':
            candidates = [(None, np.zeros_like(g, dtype=np.float64), None)]
        elif kind == 'MEAN':
            mean = np.divide(r[fit].sum(0), mask[fit].sum(0), out=np.zeros(r.shape[1:]), where=mask[fit].sum(0)>0)
            candidates = [(None, np.broadcast_to(mean, g.shape), dict(mean=mean))]
        else:
            candidates = []
            for lam in (.01, .1, 1., 10.):
                model = ridge_fit(f[fit], r[fit], mask[fit], lam, kind)
                correction = ridge_predict(model, f)
                candidates.append((lam, correction, model))
        best = min(candidates, key=lambda row: (common.score(g[select]+row[1][select], y[select], mask[select])['mse'], -(row[0] or 0)))
        for lam, correction, model in candidates:
            for split, idx in (('M_fit', fit), ('M_select', select)):
                rows.append(dict(dataset=dataset, method=kind, fit_split='M_fit', report_split=split,
                    penalty=lam, selected=lam == best[0], **common.score(g[idx]+correction[idx], y[idx], mask[idx])))
        if kind in ('OWN', 'ALL'):
            final = ridge_fit(f, r, mask, best[0], kind)
        elif kind == 'MEAN':
            final = dict(mean=np.divide(r.sum(0), mask.sum(0), out=np.zeros(r.shape[1:]), where=mask.sum(0)>0))
        else:
            final = {}
        np.savez_compressed(out/(kind+'.npz'), **final)
        np.savez_compressed(out/(kind+'_search_predictions.npz'), correction=best[1].astype(np.float32))
        selected[kind] = dict(penalty=best[0], M_select=common.score(g[select]+best[1][select], y[select], mask[select]),
                              fit_refit_seconds=time.perf_counter()-began,
                              factorization_count=final.get('factorizations'))
        common.event('PROBE', dataset=dataset, method=kind, **selected[kind])
    # Descriptive geometry uses M_fit histories; missing targets are excluded.
    values = data['x'][fit].reshape(-1, data['x'].shape[-1]).astype(np.float64)
    scales = np.maximum(values.std(0), 1e-12)
    covariance = np.cov((values-values.mean(0))/scales, rowvar=False)
    spectrum = np.maximum(np.linalg.eigvalsh(covariance)[::-1], 0.)
    common.atomic(out/'DIAGNOSTICS.json', dict(spectrum=spectrum.tolist(),
        cumulative_variance=(spectrum.cumsum()/spectrum.sum()).tolist(),
        covariance_scope='M_fit past windows, causal within-window forward-fill, zero when no earlier observed value; overlapping history repetitions retained',
        filled_values_are_not_evaluation_targets=True,
        squared_error_by_channel=np.square(r[select]).sum((0,1)).tolist(),
        absolute_error_by_channel=np.abs(r[select]).sum((0,1)).tolist()))
    common.atomic(out/'RESULT.json', dict(dataset=dataset, selected=selected, rows=rows, new_V_used=False))
    return rows


def alpha(data, dataset, seed):
    result = fit_alpha_diagnostic(data['g'], data['s'], data['y'], data['mask'],
                                 data['split']['fit_indices'], data['split']['select_indices'])
    common.atomic(common.CACHE/'probes'/dataset/f'alpha_s{seed}.json', result)
    return result
