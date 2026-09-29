"""Synthetic non-optimizer checks. The root runner reserves CPU time first."""
import io

import numpy as np
import torch
from torch import nn

from model_ridge import RidgeAdapter
from ridge_solver import build_statistics, objective_from_statistics, predict_correction, solve_ridge


def masked_macro(prediction, target, mask):
    error = np.zeros(target.shape, dtype=np.float64)
    np.subtract(prediction, target, out=error, where=mask)
    return float(np.mean(np.sum(error ** 2, axis=(0, 1)) / mask.sum(axis=(0, 1))))


class SyntheticParent(nn.Module):
    def __init__(self, basis):
        super().__init__()
        channels, latent = basis.shape
        self.encoder = nn.Linear(channels, latent, bias=False)
        self.decoder = nn.Linear(latent, channels, bias=False)
        with torch.no_grad():
            self.encoder.weight.copy_(torch.as_tensor(basis.T))
            self.decoder.weight.copy_(torch.as_tensor(basis))

    def forward(self, x):
        return x[:, -1:, :].expand(-1, 48, -1)


def run_checks():
    rng = np.random.default_rng(91010)
    n, length, horizon, channels = 31, 9, 4, 5
    x, target = rng.normal(size=(n, length, channels)), rng.normal(size=(n, horizon, channels))
    mask = rng.random(target.shape) > .17
    mask[:, 1, :] = mask[:, 0, :]
    mask[:, 3, :] = True
    target[~mask] = np.nan
    parents = {92601: rng.normal(size=target.shape), 92602: rng.normal(size=target.shape)}
    basis = np.linalg.qr(rng.normal(size=(channels, 2)))[0]
    output = []
    for family in ('ridge_p', 'ridge_raw'):
        blocks = lambda: ((np.arange(start, min(start + 7, n)), x[start:start + 7]) for start in range(0, n, 7))
        stats = build_statistics(blocks, target, mask, parents, family=family, basis=basis)
        assert len(stats.horizon_groups) == 3
        for key in parents:
            for penalty in (.001, .1):
                weight, receipt = solve_ridge(stats, key, penalty)
                correction = predict_correction(x, weight, family=family, basis=basis)
                actual = masked_macro(parents[key] + correction, target, mask)
                assert abs(actual - receipt['masked_macro_mse']) < 1e-10
                assert abs(actual + penalty * np.sum(weight ** 2) / horizon - receipt['objective']) < 1e-10
                assert receipt['max_relative_normal_equation_residual'] < 1e-8
                zero = objective_from_statistics(stats, key, np.zeros_like(weight), penalty)
                assert zero['objective'] >= receipt['objective'] - 1e-10
                z = x - x[:, -1:, :]
                if family == 'ridge_p':
                    z = (z @ basis) @ basis.T
                examples = z.transpose(0, 2, 1).reshape(-1, length)
                row_weights = np.tile(1 / (channels * mask.sum(axis=(0, 1))), n)
                for h in range(horizon):
                    observed = mask[:, h, :].reshape(-1)
                    xx = examples[observed]
                    ww = row_weights[observed]
                    yy = (target[:, h, :] - parents[key][:, h, :]).reshape(-1)[observed]
                    direct = np.linalg.solve((xx.T * ww) @ xx + penalty / horizon * np.eye(length), xx.T @ (ww * yy))
                    assert np.allclose(direct, weight[h], atol=1e-10, rtol=1e-10)
                output.append({'family': family, 'parent': key, 'penalty': penalty,
                               'normal_equation_residual': receipt['max_relative_normal_equation_residual'],
                               'objective_direct_match': True})
    shape_checks = []
    for channels, latent in ((17, 5), (21, 6)):
        basis = np.linalg.qr(rng.normal(size=(channels, latent)))[0].astype(np.float32)
        parent = SyntheticParent(basis)
        context = torch.tensor(rng.normal(size=(3, 512, channels)), dtype=torch.float32)
        weight = torch.tensor(rng.normal(scale=.001, size=(48, 512)), dtype=torch.float32)
        for family in ('ridge_p', 'ridge_raw'):
            model = RidgeAdapter(parent, weight, family).eval()
            with torch.no_grad():
                parts, predicted = model.parts(context), model(context)
                assert predicted.shape == (3, 48, channels)
                assert not any(p.requires_grad for p in model.parameters())
                if family == 'ridge_p':
                    projection = torch.eye(channels) - torch.tensor(basis) @ torch.tensor(basis.T)
                    maximum = float((parts['correction'] @ projection).abs().max())
                    assert maximum < 1e-5
                else:
                    maximum = None
                zero = RidgeAdapter(parent, torch.zeros_like(weight), family)
                assert torch.equal(zero(context), parent(context))
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                buffer.seek(0)
                restored = RidgeAdapter(SyntheticParent(basis), torch.zeros_like(weight), family).eval()
                restored.load_state_dict(torch.load(buffer, weights_only=True))
                assert torch.equal(restored(context), predicted)
            shape_checks.append({'channels': channels, 'latent': latent, 'family': family,
                                 'q_correction_max_absolute': maximum, 'state_roundtrip_exact': True,
                                 'initial_parent_exact': True, 'fitted_coefficients': model.weight.numel()})
    return {'status': 'passed', 'optimizer_updates': 0, 'data': 'synthetic only',
            'small_solver_checks': output, 'approved_shape_checks': shape_checks}
