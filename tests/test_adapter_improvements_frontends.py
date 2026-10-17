import importlib
import io
import sys
from pathlib import Path

import pytest
import torch
from torch import nn


def impl():
    try:
        return importlib.import_module('research.adapter_improvements_20261018_v1.frontends')
    except ModuleNotFoundError:
        pytest.fail('Registered local frontend is not implemented')


def backbone_modules():
    path = Path(__file__).resolve().parents[1] / 'research/informs_broad_validation_20261015_v1'
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return importlib.import_module('networks')


def test_local_frontend_exact_count_cp0_and_seed_pair_without_rng_changes():
    m = impl()
    torch.manual_seed(82)
    state = torch.get_rng_state().clone()
    split = m.LocalFrontend(3, split=True, seed=92601)
    control = m.LocalFrontend(3, split=False, seed=92601)
    assert torch.equal(state, torch.get_rng_state())
    assert sum(p.numel() for p in split.parameters()) == 1152
    for key, value in split.state_dict().items():
        assert torch.equal(value, control.state_dict()[key])
    assert split.short.input.weight.abs().sum() > 0
    assert split.trend.input.weight.abs().sum() > 0
    assert split.short.output.weight.count_nonzero() == 0
    assert split.trend.output.weight.count_nonzero() == 0
    x = torch.randn(2, 45, 3)
    assert torch.equal(split(x), x)
    assert torch.equal(control(x), x)


def test_trailing_mean_replicates_first_value_with_exact_alignment_and_decomposition():
    m = impl()
    x = torch.tensor([[[2., 20.], [4., 40.], [9., 90.], [13., 130.]]], dtype=torch.float64)
    expected = torch.tensor([[[2., 20.], [8/3, 80/3], [5., 50.], [26/3, 260/3]]], dtype=torch.float64)
    torch.testing.assert_close(m.trailing_mean(x, 3), expected, rtol=1e-15, atol=1e-15)
    frontend = m.LocalFrontend(2, split=True)
    residual, trend = frontend.components(x)
    torch.testing.assert_close(residual + trend, x, rtol=1e-15, atol=1e-15)
    left, right = m.LocalFrontend(2, split=False).components(x)
    assert torch.equal(left, x / 2) and torch.equal(right, x / 2)


@pytest.mark.parametrize('split', [True, False])
def test_nonzero_frontend_is_prefix_causal_and_channel_shared(split):
    frontend = impl().LocalFrontend(3, split=split, seed=92601).double()
    with torch.no_grad():
        frontend.short.output.weight.fill_(.03)
        frontend.trend.output.weight.fill_(.02)
    x = torch.arange(45, dtype=torch.float64)[None, :, None].expand(2, -1, 3).clone()
    x[:, :, 1] += 100
    x[:, :, 2] *= -1
    x[0, 12, 0] += 7
    changed = x.clone()
    changed[:, 26:] = torch.randn_like(changed[:, 26:]) * 1000
    assert torch.equal(frontend(x)[:, :26], frontend(changed)[:, :26])
    perm = [2, 0, 1]
    torch.testing.assert_close(frontend(x[:, :, perm]), frontend(x)[:, :, perm], rtol=0, atol=0)
    one = frontend(x[:1])
    torch.testing.assert_close(one, frontend(x)[:1], rtol=1e-14, atol=1e-14)


def test_ramp_impulse_convolution_uses_only_left_padding_and_preserves_length():
    frontend = impl().LocalFrontend(2, split=False).double()
    with torch.no_grad():
        frontend.short.input.weight.zero_()
        frontend.short.input.weight[0, 0, 0] = 1
        frontend.short.output.weight.zero_()
        frontend.short.output.weight[0, 0, 0] = 1
    x = torch.arange(40, dtype=torch.float64)[None, :, None].repeat(1, 1, 2)
    x[:, :, 1] = 0
    x[0, 11, 1] = 1
    branch = torch.cat([x[:, :1].expand(-1, 8, -1), x[:, :-8]], 1) / 2
    expected = x + torch.nn.functional.gelu(branch)
    torch.testing.assert_close(frontend(x), expected, rtol=0, atol=0)


class FrozenSynthetic(nn.Module):
    def __init__(self, horizon=3):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.2), requires_grad=False)
        self.horizon = horizon
        self.streams = 0

    def forward(self, x):
        self.streams += x.shape[0] * x.shape[-1]
        trend = x[:, -1:] - x[:, -2:-1]
        return self.scale * (x[:, -1:] + trend * torch.arange(1, self.horizon + 1, device=x.device)[None, :, None])


def test_two_step_gradients_cp0_parity_frozen_backbone_and_serialized_bundle():
    frontend = impl()
    networks = backbone_modules()
    torch.manual_seed(91)
    basis = torch.linalg.qr(torch.randn(3, 2))[0]
    b0 = networks.ForecastAdapter('B0', 40, 3, 3, 2, torch.zeros(3), basis)
    model = networks.ForecastAdapter('B0', 40, 3, 3, 2, torch.zeros(3), basis)
    model.front = frontend.LocalFrontend(3, split=True, seed=92601)
    f = FrozenSynthetic()
    f_state = {k: v.clone() for k, v in f.state_dict().items()}
    x = torch.randn(2, 40, 3)
    assert torch.equal(model(x, f), b0(x, f))
    f.streams = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x, f)
        prediction.square().mean().backward()
        assert model.E.grad.abs().sum() > 0 and model.D.grad.abs().sum() > 0
        assert model.front.short.output.weight.grad.abs().sum() > 0
        assert model.front.trend.output.weight.grad.abs().sum() > 0
        for branch in (model.front.short, model.front.trend):
            assert bool(branch.input.weight.grad.abs().sum() > 0) == bool(step)
        assert f.scale.grad is None
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    assert f.streams == 2 * x.shape[0] * 2
    assert all(torch.equal(v, f.state_dict()[k]) for k, v in f_state.items())
    saved = io.BytesIO()
    torch.save(model.state_dict(), saved)
    saved.seek(0)
    restored = networks.ForecastAdapter('B0', 40, 3, 3, 2, torch.zeros(3), basis)
    restored.front = frontend.LocalFrontend(3, split=True, seed=92601)
    restored.load_state_dict(torch.load(saved, weights_only=True), strict=True)
    assert torch.equal(restored(x, f), model(x, f))


def test_future_outside_supplied_context_cannot_affect_forecast():
    frontend = impl()
    networks = backbone_modules()
    model = networks.ForecastAdapter('B0', 40, 3, 3, 2, torch.zeros(3), torch.eye(3)[:, :2])
    model.front = frontend.LocalFrontend(3)
    stream = torch.randn(60, 3)
    origin = 45
    f = FrozenSynthetic()
    prediction = model(stream[origin-40:origin][None], f)
    stream[origin:] += 100000
    assert torch.equal(prediction, model(stream[origin-40:origin][None], f))
