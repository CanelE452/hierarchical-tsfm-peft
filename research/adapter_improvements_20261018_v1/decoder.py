"""Four registered fixed-PE readouts; all fitting and predictions use FP64."""
from dataclasses import dataclass
import types

import numpy as np
from scipy import linalg


ARMS = ('D_SHARED', 'D_HORIZON', 'D_LINEAR', 'D_GAIN')
LAMBDAS = (.0001, .001, .01, .1, 1.)
GAMMAS = (1., 10., 100.)


def horizon_basis(horizon):
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError('H must be a positive integer')
    if horizon == 1:
        return np.zeros(1, np.float64)
    return (np.arange(1, horizon+1, dtype=np.float64)-(horizon+1)/2)/np.sqrt((horizon*horizon-1)/12)


def registered_grid():
    return [dict(arm=arm, lambda_value=lam, gamma=gamma)
            for arm in ARMS for lam in LAMBDAS
            for gamma in (GAMMAS if arm in ('D_LINEAR', 'D_GAIN') else (1.,))]


def select_trial(trials):
    if not trials or not all(np.isfinite(t['MSE']) for t in trials):
        raise ValueError('Selection requires finite unrounded FP64 scores')
    return min(trials, key=lambda t: (t['MSE'], t['lambda_value'], t.get('gamma', 1.)))


@dataclass
class HeadStatistics:
    rms: np.ndarray
    zero: np.ndarray
    counts: np.ndarray
    horizon_counts: np.ndarray
    gram: np.ndarray
    rhs: np.ndarray
    yy: np.ndarray
    qr: list
    qty: list
    horizon: int
    features: int
    channels: int


def prepare_statistics(phi, delta, mask, rms=None, check=None):
    """Direct Gram statistics and independent QR compression of observed rows.

    Identical channel masks share a raw-design QR; no Gram-derived square root
    is used for the independent augmented least-squares verification.
    """
    x, y, observed = np.asarray(phi, np.float64), np.asarray(delta, np.float64), np.asarray(mask)
    if x.ndim != 3 or y.ndim != 3 or observed.shape != y.shape or x.shape[:2] != y.shape[:2]:
        raise ValueError('Expected phi[N,H,P], delta/mask[N,H,C]')
    if observed.dtype != np.bool_ or not np.isfinite(x).all() or not np.isfinite(y[observed]).all():
        raise ValueError('Nonfinite observed data or nonboolean mask')
    n, h, p = x.shape
    c = y.shape[-1]
    counts = observed.sum((0, 1), dtype=np.int64)
    if np.any(counts == 0):
        raise ValueError('BLOCKED_DATA_CONTRACT: registered channel has N_c=0')
    measured = np.sqrt(np.mean(np.square(x), axis=(0, 1)))
    zero = measured == 0
    scale = measured.copy() if rms is None else np.asarray(rms, np.float64).copy()
    if scale.shape != (p,) or not np.isfinite(scale).all() or np.any(scale < 0) or np.any((scale == 0) & ~zero):
        raise ValueError('Invalid fitting-split RMS')
    scale[zero] = 1.
    x = x/scale
    gram = np.zeros((h, c, p, p), np.float64)
    rhs = np.zeros((h, c, p), np.float64)
    yy = np.zeros((h, c), np.float64)
    qr, qty = [], []
    for index in range(h):
        if check:
            check()
        factors, projections, groups = [None]*c, [None]*c, {}
        for channel in range(c):
            active = observed[:, index, channel]
            groups.setdefault(active.tobytes(), []).append(channel)
        for channels in groups.values():
            active = observed[:, index, channels[0]]
            design = x[:, index][active]
            target = y[:, index, channels][active]
            if not len(design):
                q, r = np.zeros((0, 0)), np.zeros((0, p))
            else:
                q, r = linalg.qr(design, mode='economic', check_finite=False)
            g, b, t = design.T@design, design.T@target, q.T@target
            for column, channel in enumerate(channels):
                gram[index, channel] = g
                rhs[index, channel] = b[:, column]
                yy[index, channel] = np.dot(target[:, column], target[:, column])
                factors[channel], projections[channel] = r, t[:, column]
        qr.append(factors)
        qty.append(projections)
    return HeadStatistics(scale, zero, counts, observed.sum(0), gram, rhs, yy, qr, qty, h, p, c)


def _solve(gram, rhs, penalty, design, target):
    system = gram + np.diag(penalty)
    try:
        result = linalg.solve(system, rhs, assume_a='pos', check_finite=False)
    except linalg.LinAlgError:
        result = None
    augmented = np.vstack((design, np.diag(np.sqrt(penalty))))
    independent, _, rank, singular = linalg.lstsq(augmented, np.r_[target, np.zeros(len(penalty))],
                                                lapack_driver='gelsd', check_finite=False)
    if result is None:
        result = independent
    residual = np.linalg.norm(system@result-rhs)/max(1., np.linalg.norm(rhs))
    discrepancy = np.linalg.norm(independent-result)/max(1., np.linalg.norm(independent))
    if not np.isfinite(result).all() or residual > 1e-8 or discrepancy > 2e-8:
        raise ValueError('Decoder solve failed normal-equation/independent augmented-SVD check')
    return result, float(residual), float(discrepancy), int(rank)


def fit_statistics(stats, arm, regularization, gamma=1.):
    if arm not in ARMS or not np.isfinite(regularization) or regularization <= 0 or not np.isfinite(gamma) or gamma <= 0:
        raise ValueError('Registered arm requires positive lambda and gamma')
    h, p, c = stats.horizon, stats.features, stats.channels
    requested = arm
    if h == 1:
        arm = 'D_SHARED'
    g = horizon_basis(h)
    if arm == 'D_GAIN':
        base = fit_statistics(stats, 'D_SHARED', regularization)
        coefficients = np.asarray(base['coefficients'])
        gains = np.tile([1., 0.], (c, 1))
    else:
        coefficients = np.zeros((h, p, c) if arm == 'D_HORIZON' else (2, p, c) if arm == 'D_LINEAR' else (p, c))
        gains = None
    max_residual = max_difference = 0.
    ranks, zero_shared, channel_data, channel_penalties = [], [], [], []
    for channel in range(c):
        gram_h, rhs_h = stats.gram[:, channel], stats.rhs[:, channel]
        nc = int(stats.counts[channel])
        sse, penalty_value = float(stats.yy[:, channel].sum()), 0.
        if arm == 'D_GAIN' and not np.any(coefficients[:, channel]):
            zero_shared.append(channel)
            channel_data.append(sse/nc)
            channel_penalties.append(0.)
            ranks.append(0)
            continue
        for index in (range(h) if arm == 'D_HORIZON' else (None,)):
            if index is not None:
                gram, rhs = gram_h[index], rhs_h[index]
                design, target = stats.qr[index][channel], stats.qty[index][channel]
                penalty = np.full(p, regularization*nc/h)
            else:
                raw_design = np.vstack([stats.qr[i][channel] for i in range(h)])
                target = np.concatenate([stats.qty[i][channel] for i in range(h)])
                if arm == 'D_SHARED':
                    gram, rhs, design = gram_h.sum(0), rhs_h.sum(0), raw_design
                    penalty = np.full(p, regularization*nc)
                elif arm == 'D_LINEAR':
                    cross = np.einsum('h,hpq->pq', g, gram_h)
                    gram = np.block([[gram_h.sum(0), cross], [cross, np.einsum('h,hpq->pq', g*g, gram_h)]])
                    rhs = np.r_[rhs_h.sum(0), np.einsum('h,hp->p', g, rhs_h)]
                    design = np.vstack([np.c_[stats.qr[i][channel], g[i]*stats.qr[i][channel]] for i in range(h)])
                    penalty = regularization*nc*np.r_[np.ones(p), np.full(p, gamma)]
                else:
                    d = coefficients[:, channel]
                    gd = np.einsum('p,hpq,q->h', d, gram_h, d)
                    ry = rhs_h@d
                    gram = np.array([[gd.sum(), g@gd], [g@gd, (g*g)@gd]])
                    rhs = np.array([ry.sum(), g@ry])
                    design = np.vstack([np.c_[stats.qr[i][channel]@d, g[i]*(stats.qr[i][channel]@d)] for i in range(h)])
                    penalty = regularization*nc*np.dot(d, d)*np.array([1., gamma])
            fitted, residual, discrepancy, rank = _solve(gram, rhs, penalty, design, target)
            max_residual, max_difference = max(max_residual, residual), max(max_difference, discrepancy)
            ranks.append(rank)
            sse += float(fitted@gram@fitted-2*fitted@rhs)
            penalty_value += float(np.dot(penalty*fitted, fitted)/nc)
            if arm == 'D_HORIZON':
                fitted[stats.zero] = 0
                coefficients[index, :, channel] = fitted
            elif arm == 'D_LINEAR':
                fitted.reshape(2, p)[:, stats.zero] = 0
                coefficients[:, :, channel] = fitted.reshape(2, p)
            elif arm == 'D_GAIN':
                gains[channel] = fitted
            else:
                fitted[stats.zero] = 0
                coefficients[:, channel] = fitted
        channel_data.append(sse/nc)
        channel_penalties.append(penalty_value)
    data_term, penalty_value = float(np.mean(channel_data)), float(np.mean(channel_penalties))
    return dict(schema_version=1, arm=arm, requested_arm=requested, H=h, P=p, C=c,
                lambda_value=float(regularization), gamma=float(gamma),
                rms=stats.rms.tolist(), zero_features=stats.zero.tolist(), coefficients=coefficients.tolist(),
                gain=None if gains is None else gains.tolist(),
                identifiability=dict(gain_zero_shared_channels=zero_shared, H1_shared_reclassification=h == 1,
                                     empty_horizon_channels=np.argwhere(stats.horizon_counts == 0).tolist()),
                diagnostics=dict(data_term=data_term, penalty=penalty_value, objective=data_term+penalty_value,
                                 channel_data_term=channel_data, channel_penalty=channel_penalties,
                                 counts=stats.counts.tolist(), horizon_counts=stats.horizon_counts.tolist(),
                                 normal_equation_relative_residual=max_residual,
                                 augmented_svd_relative_difference=max_difference, augmented_ranks=ranks,
                                 independent_design='raw observed-design QR; augmented gelsd SVD',
                                 gain_estimation='in-sample two-stage; same-lambda fitting-only shared base' if gains is not None else None))


def fit_head(phi, delta, mask, arm, regularization, gamma=1., rms=None):
    return fit_statistics(prepare_statistics(phi, delta, mask, rms), arm, regularization, gamma)


def predict_head(phi, state):
    x = np.asarray(phi, np.float64)
    if x.ndim != 3 or x.shape[1:] != (state['H'], state['P']):
        raise ValueError('Prediction feature axes differ from fitted [H,P]')
    x = x/np.asarray(state['rms'])
    d, arm = np.asarray(state['coefficients']), state['arm']
    if arm == 'D_HORIZON':
        return np.einsum('nhp,hpc->nhc', x, d)
    if arm == 'D_LINEAR':
        return x@d[0] + horizon_basis(state['H'])[None, :, None]*(x@d[1])
    value = x@d
    if arm == 'D_GAIN':
        gain = np.asarray(state['gain'])
        value *= gain[None, None, :, 0] + horizon_basis(state['H'])[None, :, None]*gain[None, None, :, 1]
    return value


def install_head(model, state):
    """Install before strict state loading; PE/F stay FP32, head stays FP64."""
    import torch
    from torch import nn

    class FrozenReadout(nn.Module):
        def __init__(self):
            super().__init__()
            self.arm = state['arm']
            self.coefficients = nn.Parameter(torch.tensor(state['coefficients'], dtype=torch.float64), requires_grad=False)
            self.register_buffer('rms', torch.tensor(state['rms'], dtype=torch.float64))
            self.register_buffer('basis', torch.tensor(horizon_basis(state['H']), dtype=torch.float64))
            if state['gain'] is not None:
                self.gain = nn.Parameter(torch.tensor(state['gain'], dtype=torch.float64), requires_grad=False)

        def forward(self, features):
            x, d = features.to(torch.float64)/self.rms, self.coefficients
            if self.arm == 'D_HORIZON':
                return torch.einsum('nhp,hpc->nhc', x, d)
            if self.arm == 'D_LINEAR':
                return x@d[0] + self.basis[None, :, None]*(x@d[1])
            value = x@d
            if self.arm == 'D_GAIN':
                value = value*(self.gain[None, None, :, 0] + self.basis[None, :, None]*self.gain[None, None, :, 1])
            return value

    if model.horizon != state['H'] or model.channels != state['C'] or model.latent_k*7 != state['P']:
        raise ValueError('Head and PE configuration mismatch')
    device = model.E.device
    model.requires_grad_(False)
    if hasattr(model, 'D'):
        delattr(model, 'D')
    model.add_module('fitted_decoder', FrozenReadout().to(device))

    def decode(self, x, z, future, latent_k=None):
        k = self._k(latent_k)
        if k != self.latent_k or tuple(z.shape) != (x.shape[0], self.length, k) or tuple(future.shape) != (x.shape[0], self.horizon, k):
            raise ValueError('Exact decoder requires its registered K and [B,H,K] future')
        joined = torch.cat((z, future), 1)
        features = torch.cat([joined[:, self.length-lag:self.length-lag+self.horizon]-z[:, -1-lag, None]
                              for lag in (0, 1, 2, 4, 8, 16, 32)], -1)
        return x[:, -1:].to(torch.float64) + self.fitted_decoder(features)

    model.decode = types.MethodType(decode, model)
    return model
