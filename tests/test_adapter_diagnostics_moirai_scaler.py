"""CPU-only regression for the pinned Moirai scaler's input-autograd blocker.

Reference operations follow Apache-2.0 uni2ts at cfd46d4510ed8896f263116f32928eede05b0a75,
src/uni2ts/module/packed_scaler.py. No model or optional uni2ts import is needed.
"""

import pytest
import torch

from research.adapter_diagnostics_20261017_v2.functional_scaler import SafePackedStdScaler


def _official_scaler(target, observed, sample, variate, correction=1, minimum_scale=1e-5):
    target = target.double()
    same = (sample.unsqueeze(-1) == sample.unsqueeze(-2)) & (
        variate.unsqueeze(-1) == variate.unsqueeze(-2))
    counts = (same * observed.sum(-1).unsqueeze(-2)).sum(-1, keepdim=True)
    def divide(numerator, denominator):
        return numerator / torch.where(denominator == 0, 1.0, denominator)
    loc = divide((same * (target * observed).sum(-1).unsqueeze(-2)).sum(-1, keepdim=True), counts)
    variance = (same * (((target-loc)**2) * observed).sum(-1).unsqueeze(-2)).sum(-1, keepdim=True)
    scale = torch.sqrt(divide(variance, counts-correction) + minimum_scale)
    loc[sample == 0] = 0
    scale[sample == 0] = 1
    return loc.float(), scale.float()


def _inputs():
    generator = torch.Generator().manual_seed(92601)
    target = torch.randn(2, 6, 4, generator=generator)
    observed = torch.rand(2, 6, 4, generator=generator) > .2
    sample = torch.tensor([[1, 1, 1, 2, 2, 0], [1, 1, 1, 1, 0, 0]])
    variate = torch.tensor([[0, 0, 1, 0, 0, 0], [0, 0, 1, 1, 0, 0]])
    return target, observed, sample, variate


def test_official_scaler_reproduces_inplace_backward_failure_without_model():
    target, observed, sample, variate = _inputs()
    target.requires_grad_()
    loc, scale = _official_scaler(target, observed, sample, variate)
    with pytest.raises(RuntimeError, match="inplace operation"):
        ((target-loc)/scale).square().mean().backward()


@pytest.mark.parametrize("kind", ["random", "constant", "all_missing", "singleton", "no_padding"])
def test_safe_scaler_preserves_official_forward_and_finite_input_gradients(kind):
    target, observed, sample, variate = _inputs()
    if kind == "constant":
        target.fill_(5.25)
    elif kind == "all_missing":
        observed.zero_()
    elif kind == "singleton":
        observed.zero_()
        observed[:, 0, 0] = True
    elif kind == "no_padding":
        sample.fill_(1)
    expected = _official_scaler(target, observed, sample, variate)
    target.requires_grad_()
    scaler = SafePackedStdScaler(correction=1, minimum_scale=1e-5)
    actual = scaler(target, observed, sample, variate)
    for lhs, rhs in zip(actual, expected):
        torch.testing.assert_close(lhs, rhs, atol=0, rtol=0)
    assert scaler.state_dict() == {}
    assert list(scaler.parameters()) == []
    ((target-actual[0])/actual[1]).square().mean().backward()
    assert torch.isfinite(target.grad).all()
    if kind in ("random", "no_padding"):
        assert target.grad.abs().sum() > 0
