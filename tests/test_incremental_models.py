import copy

import torch

from research.informs_broad_validation_20261015_v1.networks import ForecastAdapter, module_state_hash
from research.informs_broad_validation_20261015_v1.training import macro_loss
from research.incremental_latent_adaptation_20261019_v1.models import make, lag_features, matched_hidden


def parent():
    torch.manual_seed(3)
    return ForecastAdapter('B2', 64, 5, 4, 2, torch.zeros(4), torch.eye(4))


def foundation(z):
    return z[:, -1:] + z[:, -5:].mean(1, keepdim=True).expand(-1, 5, -1)*.3


def test_anchor_parity_zero_and_later_input_gradients():
    p = parent()
    x = torch.randn(3, 64, 4)
    z = p.encode(x)
    phi = lag_features(z, foundation(z))
    torch.testing.assert_close(p.decode(x, z, foundation(z))-x[:, -1:], phi @ p.D.reshape(14, 4))
    for method in ('RESIDUAL_F', 'RESIDUAL_Q', 'RESIDUAL_RAW_MLP', 'READOUT_ONLY'):
        m = make(p, method, 92601)
        assert torch.equal(m(x, foundation), torch.zeros(3, 5, 4))
    m = make(p, 'RESIDUAL_F', 92601)
    (m(x, foundation)-torch.randn(3, 5, 4)).square().mean().backward()
    assert torch.count_nonzero(m.E.grad) == 0
    with torch.no_grad():
        m.D.add_(torch.randn_like(m.D)*.01)
    m.zero_grad(set_to_none=True)
    m(x, foundation).square().mean().backward()
    assert m.E.grad.abs().sum() > 0
    assert sum(v.grad.abs().sum() for v in m.front.parameters() if v.grad is not None) > 0


def test_residual_loss_and_gradient_match_masked_channel_macro():
    torch.manual_seed(2)
    y, g = torch.randn(3, 5, 4, dtype=torch.float64), torch.randn(3, 5, 4, dtype=torch.float64)
    mask = torch.rand(3, 5, 4) > .3
    counts = mask.sum((0, 1))
    c = torch.randn_like(y, requires_grad=True)
    a = macro_loss(g+c, y, mask, counts, 3, 3)
    b = macro_loss(c, y-g, mask, counts, 3, 3)
    torch.testing.assert_close(a, b, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(torch.autograd.grad(a, c)[0], torch.autograd.grad(b, c)[0], atol=1e-14, rtol=1e-14)
    manual = torch.stack([((g+c-y)[..., k][mask[..., k]]**2).mean() for k in range(4)]).mean()
    torch.testing.assert_close(a, manual)


def test_readout_frozen_parent_and_future_canary():
    p = parent()
    p.requires_grad_(False)
    before = module_state_hash(p)
    m = make(p, 'READOUT_ONLY', 92601)
    x = torch.randn(2, 64, 4)
    phi = m.features(x, foundation).detach()
    with torch.no_grad():
        m.D.normal_(std=.01)
    torch.testing.assert_close(m(x, foundation), m.from_features(phi))
    assert not m.E.requires_grad and all(not p.requires_grad for p in m.front.parameters())
    assert module_state_hash(p) == before
    adapted = make(p, 'RESIDUAL_F', 92601)
    assert all(v.requires_grad for v in adapted.front.parameters())
    # The inference signature contains no future target or mask.
    y, mask = torch.randn(2, 5, 4), torch.rand(2, 5, 4) > .5
    a = m(x, foundation)
    y.fill_(1e9)
    mask.logical_not_()
    assert torch.equal(a, m(x, foundation))


def test_parameter_matching_is_analytic_and_ties_choose_smaller():
    for n in (1, 100, 1040, 5000):
        d = matched_hidden(64, 5, n)
        assert d == min(range(1, 100), key=lambda k: (abs(70*k+5-n), k))
    assert matched_hidden(64, 5, 110) == 1
