import importlib
from pathlib import Path

import numpy as np
import pytest
import torch


def impl():
    try:
        return importlib.import_module('research.adapter_improvements_20261018_v1.lift')
    except ModuleNotFoundError:
        pytest.fail('Official LIFT preparation is not implemented')


@pytest.fixture
def official():
    path = Path(__file__).resolve().parents[1] / '.cache/adapter_improvements_20261018_v1/official'
    if not (path/'LIFT/models/LIFT.py').is_file():
        pytest.skip('Pinned official LIFT source is an explicit private replay dependency')
    return path


def test_external_final_forecast_bypasses_backbone_and_complex_gradients(official):
    m = impl()
    class Counter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def forward(self, x):
            self.calls += 1
            return x[:, -1:].expand(-1, 8, -1)
    backbone = Counter()
    refiner = m.make_official_refiner(official, length=32, horizon=8, channels=5,
                                    n_leaders=4, states=20, seed=12, backbone=backbone)
    x = torch.tensor(np.random.default_rng(9).normal(size=(2, 32, 5)), dtype=torch.float32)
    final = backbone(x).clone().requires_grad_()
    pred = refiner(x, final)
    assert backbone.calls == 1
    assert pred.shape == final.shape and torch.isfinite(pred).all()
    pred.square().mean().backward()
    assert final.grad is not None and torch.isfinite(final.grad).all()
    complex_parameters = [p for p in refiner.parameters() if p.is_complex()]
    assert complex_parameters
    for p in complex_parameters:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    inventory = m.parameter_inventory(refiner)
    assert inventory['bytes'] == sum(p.numel()*p.element_size() for p in refiner.parameters())
    assert inventory['real_degrees_of_freedom'] > inventory['parameter_elements']


def test_original_channels_normalization_rng_and_future_canary(official):
    m = impl()
    torch.manual_seed(928)
    before = torch.get_rng_state().clone()
    refiner = m.make_official_refiner(official, length=32, horizon=8, channels=5, seed=12)
    assert torch.equal(before, torch.get_rng_state())
    stream = torch.randn(2, 50, 5)
    y_hat = stream[:, 31:32].expand(-1, 8, -1).clone()
    pred = refiner(stream[:, :32], y_hat)
    stream[:, 32:] = 1e10
    torch.testing.assert_close(refiner(stream[:, :32], y_hat), pred, rtol=0, atol=0)
    official_module = importlib.import_module(refiner.model.__class__.__module__)
    history = stream[:, :32].transpose(1, 2)
    offset = torch.arange(5).reshape(1, 5, 1)*10
    normalized, mean, std = official_module.instance_norm(history, -1)
    torch.testing.assert_close(normalized, (history-mean)/std)
    torch.testing.assert_close(official_module.instance_norm(history+offset, -1)[0], normalized, rtol=1e-4, atol=1e-4)
    received = []
    hook = refiner.model.lead_refiner.factory.register_forward_pre_hook(lambda module, args: received.append(args[0].detach().clone()))
    try:
        refiner(stream[:, :32], y_hat)
    finally:
        hook.remove()
    torch.testing.assert_close(received[0], history, rtol=0, atol=0)
    with pytest.raises(ValueError, match='original.channel'):
        refiner(stream[:, :32, :2], y_hat[..., :2])
