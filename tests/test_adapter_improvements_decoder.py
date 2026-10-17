import importlib
import json

import numpy as np
import pytest
from scipy.linalg import lstsq


def impl():
    return importlib.import_module('research.adapter_improvements_20261018_v1.decoder')


def sample():
    rng = np.random.default_rng(22)
    x = rng.normal(size=(11, 4, 3))
    x[..., 2] = 0
    y = rng.normal(size=(11, 4, 2))
    mask = rng.random(y.shape) > .25
    mask[:, 2, 0] = False
    return x, y, mask


@pytest.mark.parametrize('arm', ['D_SHARED', 'D_HORIZON', 'D_LINEAR', 'D_GAIN'])
def test_masked_global_count_equations_against_raw_augmented_svd(arm):
    m = impl()
    phi, y, mask = sample()
    lam, gamma = .13, 10.
    state = m.fit_head(phi, y, mask, arm, lam, gamma)
    state = json.loads(json.dumps(state, allow_nan=False))
    x = phi / np.asarray(state['rms'])
    g = (np.arange(1, 5) - 2.5) / np.sqrt(15 / 12)
    expected = np.zeros_like(y)
    for c in range(2):
        nc = mask[..., c].sum()
        base = None
        if arm == 'D_GAIN':
            xx = x[mask[..., c]]
            yy = y[..., c][mask[..., c]]
            base = lstsq(np.vstack((xx, np.sqrt(lam*nc)*np.eye(3))),
                         np.r_[yy, np.zeros(3)], lapack_driver='gelsd')[0]
        for h in (range(4) if arm == 'D_HORIZON' else [None]):
            active = mask[..., c] if h is None else mask[:, h, c]
            xx = x[active] if h is None else x[:, h][active]
            yy = y[..., c][active] if h is None else y[:, h, c][active]
            if arm == 'D_LINEAR':
                gg = np.broadcast_to(g[None, :], mask.shape[:2])[active]
                xx = np.c_[xx, gg[:, None]*xx]
                penalty = lam*nc*np.r_[np.ones(3), np.full(3, gamma)]
            elif arm == 'D_GAIN':
                gg = np.broadcast_to(g[None, :], mask.shape[:2])[active]
                v = xx @ base
                xx = np.c_[v, gg*v]
                penalty = lam*nc*np.dot(base, base)*np.array([1., gamma])
            else:
                penalty = np.full(3, lam*nc/(4 if h is not None else 1))
            coefficient = lstsq(np.vstack((xx, np.diag(np.sqrt(penalty)))),
                                np.r_[yy, np.zeros(len(penalty))], lapack_driver='gelsd')[0]
            if arm == 'D_HORIZON':
                expected[:, h, c] = x[:, h] @ coefficient
            elif arm == 'D_LINEAR':
                expected[..., c] = x @ coefficient[:3] + g[None, :]*(x @ coefficient[3:])
            elif arm == 'D_GAIN':
                expected[..., c] = (x @ base)*(coefficient[0] + g[None, :]*coefficient[1])
            else:
                expected[..., c] = x @ coefficient
    np.testing.assert_allclose(m.predict_head(phi, state), expected, atol=2e-12, rtol=2e-12)
    assert state['zero_features'] == [False, False, True]
    assert state['diagnostics']['normal_equation_relative_residual'] < 1e-10
    assert state['diagnostics']['augmented_svd_relative_difference'] < 1e-10
    pred = m.predict_head(phi, state)
    data_term = np.mean([np.square((pred-y)[..., c][mask[..., c]]).mean() for c in range(2)])
    coefficients = np.asarray(state['coefficients'])
    if arm == 'D_SHARED':
        penalty = lam*np.sum(coefficients**2)/2
    elif arm == 'D_HORIZON':
        penalty = lam*np.sum(coefficients**2)/(2*4)
    elif arm == 'D_LINEAR':
        penalty = lam*(np.sum(coefficients[0]**2)+gamma*np.sum(coefficients[1]**2))/2
    else:
        gain = np.asarray(state['gain'])
        penalty = lam*np.sum(np.sum(coefficients**2, axis=0)*(gain[:, 0]**2+gamma*gain[:, 1]**2))/2
    assert state['diagnostics']['data_term'] == pytest.approx(data_term, abs=1e-13)
    assert state['diagnostics']['penalty'] == pytest.approx(penalty, abs=1e-13)
    assert state['diagnostics']['objective'] == pytest.approx(data_term + state['diagnostics']['penalty'])


def test_basis_nesting_penalized_objective_and_degenerate_gain():
    m = impl()
    g = m.horizon_basis(48)
    assert np.mean(g) == pytest.approx(0, abs=1e-15)
    assert np.mean(g*g) == pytest.approx(1, abs=1e-15)
    phi, y, mask = sample()
    shared = m.fit_head(phi, y, mask, 'D_SHARED', .1)
    linear = m.fit_head(phi, y, mask, 'D_LINEAR', .1)
    gain = m.fit_head(phi, y, mask, 'D_GAIN', .1)
    full = m.fit_head(phi, y, mask, 'D_HORIZON', .1)
    assert linear['diagnostics']['objective'] <= shared['diagnostics']['objective'] + 1e-12
    assert linear['diagnostics']['objective'] <= gain['diagnostics']['objective'] + 1e-12
    assert full['diagnostics']['objective'] <= linear['diagnostics']['objective'] + 1e-12
    linear['coefficients'] = [shared['coefficients'], np.zeros_like(shared['coefficients']).tolist()]
    gain['gain'] = [[1., 0.], [1., 0.]]
    np.testing.assert_allclose(m.predict_head(phi, shared), m.predict_head(phi, linear), atol=1e-15)
    np.testing.assert_allclose(m.predict_head(phi, shared), m.predict_head(phi, gain), atol=1e-15)
    zero = m.fit_head(phi*0, y, mask, 'D_GAIN', .1)
    assert zero['gain'] == [[1., 0.], [1., 0.]]
    assert zero['identifiability']['gain_zero_shared_channels'] == [0, 1]
    assert m.fit_head(phi[:, :1], y[:, :1], mask[:, :1], 'D_LINEAR', .1)['arm'] == 'D_SHARED'


def test_empty_channel_blocks_without_silently_changing_macro_channel_set():
    phi, y, mask = sample()
    mask[..., 1] = False
    with pytest.raises(ValueError, match='BLOCKED_DATA_CONTRACT'):
        impl().fit_head(phi, y, mask, 'D_SHARED', .1)


def test_exact_ties_are_not_rounded_and_registered_grid_is_40():
    m = impl()
    assert len(m.registered_grid()) == 40
    trials = [{'MSE': .2, 'lambda_value': .01, 'gamma': 1},
              {'MSE': np.nextafter(.2, 0), 'lambda_value': 1., 'gamma': 100}]
    assert m.select_trial(trials)['lambda_value'] == 1.
    trials[1]['MSE'] = .2
    assert m.select_trial(trials)['lambda_value'] == .01


def test_installed_fp64_head_lag_major_reload_with_fp32_backbone():
    import sys
    from pathlib import Path
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'research/informs_broad_validation_20261015_v1'))
    from networks import ForecastAdapter
    m = impl()
    rng = np.random.default_rng(19)
    state = m.fit_head(rng.normal(size=(12, 4, 14)), rng.normal(size=(12, 4, 3)),
                       np.ones((12, 4, 3), bool), 'D_LINEAR', .1)
    def create():
        return ForecastAdapter('B0', 40, 4, 3, 2, np.zeros(3), np.eye(3)[:, :2])
    model = m.install_head(create(), state)
    x = torch.randn(2, 40, 3)
    calls = []
    def frozen(z):
        calls.append(z.dtype)
        return z[:, -4:] + .3
    out = model(x, frozen)
    z = model.encode(x)
    u = z[:, -4:] + .3
    joined = torch.cat((z, u), 1)
    phi = torch.cat([joined[:, 40-lag:44-lag] - z[:, -1-lag, None] for lag in (0, 1, 2, 4, 8, 16, 32)], -1)
    np.testing.assert_allclose(out.detach().numpy(), x[:, -1:].numpy() + m.predict_head(phi.detach().numpy(), state), rtol=2e-14, atol=2e-14)
    assert out.dtype == torch.float64 and calls == [torch.float32]
    assert model.E.dtype == torch.float32
    assert not any(parameter.requires_grad for parameter in model.parameters())
    other = m.install_head(create(), json.loads(json.dumps(state)))
    other.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(other(x, frozen), out, rtol=0, atol=0)
    assert sum(p.numel() for p in model.fitted_decoder.parameters()) == 84
