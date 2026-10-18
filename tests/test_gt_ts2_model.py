import copy
import importlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from research.professor_compressor_direct_20261011.models_gt import (
    GTConvAE, temporal_downsample, temporal_upsample,
)


@pytest.fixture
def model_module():
    try:
        return importlib.import_module("research.gt_ts2_fastcheck_20261018_v1.model")
    except ModuleNotFoundError as error:
        if error.name in {"research.gt_ts2_fastcheck_20261018_v1",
                          "research.gt_ts2_fastcheck_20261018_v1.model"}:
            pytest.fail("The gradient-preserving GT/TS2 forecast model is missing")
        raise


def positive_gt():
    gt = GTConvAE(torch.eye(21))
    with torch.no_grad():
        for layer in [*gt.encoder, *gt.decoder]:
            layer.weight.zero_()
            layer.weight[:, 0, 0] = torch.tensor([1., .02, .01, .005, .002])
        gt.encoder[-1].weight[:, 0, 1] = .01
    return gt


class RepeatFoundation:
    def forecast(self, latent, horizon):
        return latent[:, -1:].expand(-1, horizon, -1, -1)


class TinyBolt(nn.Module):
    def __init__(self):
        super().__init__()
        from chronos.chronos_bolt import InstanceNorm
        self.instance_norm = InstanceNorm()
        self.weight = nn.Parameter(torch.tensor(2.))
        self.chronos_config = SimpleNamespace(
            quantiles=[.1, .5, .9], prediction_length=12, context_length=8,
        )

    def forward(self, context):
        context = context[:, -self.chronos_config.context_length:]
        normalized, loc_scale = self.instance_norm(context)
        median = (normalized.mean(-1, keepdim=True)
                  + .5 * normalized[:, -1:] * self.weight)
        prediction = torch.stack([median - 1, median, median + 1], dim=1)
        prediction = prediction.expand(-1, -1, 12)
        prediction = self.instance_norm.inverse(
            prediction, tuple(value.unsqueeze(1) for value in loc_scale),
        )
        return SimpleNamespace(quantile_preds=prediction)


@pytest.fixture
def fake_loader(monkeypatch):
    import chronos

    def load(directory, **kwargs):
        assert kwargs == dict(device_map="cpu", torch_dtype=torch.float32,
                              local_files_only=True)
        return SimpleNamespace(model=TinyBolt(), model_context_length=8)

    monkeypatch.setattr(chronos.ChronosBoltPipeline, "from_pretrained", load)


@pytest.mark.parametrize("length", [512, 1024, 2048])
def test_route_preserves_native_time_shapes(model_module, length):
    gt = positive_gt()
    route = model_module.ForecastRoute(gt, RepeatFoundation())
    x = torch.linspace(.1, 1., length)[None, :, None].expand(1, -1, 21)
    assert gt.encode(x).shape == (1, length // 4, 21, 2)
    assert route.forecast(x).shape == (1, 48, 21)
    assert sum(parameter.numel() for parameter in route.parameters()) == 244
    assert all(parameter.requires_grad for parameter in gt.parameters())
    assert not gt.product.graph.requires_grad


def test_pooling_matches_four_ticks_and_native_right_edge(model_module):
    raw = torch.arange(8, dtype=torch.float32)[None, :, None].expand(1, -1, 320)
    pooled = model_module.ForecastRoute.pool_features(raw)
    torch.testing.assert_close(pooled[0, :, 0], torch.tensor([1.5, 5.5]))
    ticks = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1, 1)
    torch.testing.assert_close(temporal_downsample(temporal_downsample(ticks)).flatten(),
                               torch.tensor([3., 7.]))
    upsampled = temporal_upsample(temporal_upsample(torch.tensor([[[[2.]], [[5.]]]])))
    torch.testing.assert_close(upsampled.flatten(), torch.tensor([0., 0., 0., 2., 0., 0., 0., 5.]))


def test_native_decoder_keeps_history_at_forecast_boundary(model_module):
    gt = positive_gt()
    with torch.no_grad():
        gt.product.s00.zero_()
        gt.product.s10.fill_(1.)
        gt.product.s01.zero_()
        gt.product.s11.zero_()
        for layer in gt.decoder:
            layer.weight.zero_()
            layer.weight[1, 0, 0] = 1.
    history = torch.zeros(1, 2, 21, 2)
    history[0, :, :, 0] = torch.tensor([1., 2.])[:, None]
    future = torch.zeros(1, 12, 21, 2)
    future[0, :, :, 0] = torch.arange(3., 15.)[:, None]
    result = gt.decode_future(history, future, horizon=48)
    torch.testing.assert_close(result[0, :12, 0],
                               torch.tensor([0., 0., 2., 0., 0., 0., 3., 0., 0., 0., 4., 0.]))
    assert not torch.equal(result, gt.decode(future))
    route = model_module.ForecastRoute(gt, RepeatFoundation())
    x = torch.ones(1, 512, 21)
    latent = gt.encode(x)
    expected = gt.decode_future(latent, latent[:, -1:].expand(-1, 12, -1, -1), 48)
    torch.testing.assert_close(route.forecast(x), expected, rtol=0, atol=0)


def test_zero_bridge_matches_parent_and_does_not_advance_rng(model_module):
    gt = positive_gt()
    rng = torch.get_rng_state().clone()
    fused = model_module.ForecastRoute(copy.deepcopy(gt), RepeatFoundation(),
                                      torch.zeros(320), torch.ones(320))
    assert torch.equal(rng, torch.get_rng_state())
    assert fused.W.bias is None and fused.W.weight.numel() == 13440
    assert torch.count_nonzero(fused.W.weight) == 0
    x = torch.linspace(.1, 2., 512)[None, :, None].expand(1, -1, 21)
    features = torch.ones(1, 128, 320)
    baseline = model_module.ForecastRoute(gt, RepeatFoundation())
    torch.testing.assert_close(fused.forecast(x, features), baseline.forecast(x), rtol=0, atol=0)


def test_bridge_uses_fixed_feature_statistics_and_constant_dimensions(model_module):
    gt = positive_gt()
    route = model_module.ForecastRoute(gt, RepeatFoundation(),
                                      torch.full((320,), 2.), torch.zeros(320))
    with torch.no_grad():
        route.W.weight.zero_()
        route.W.weight[0, 0] = 1.
    assert torch.equal(route.feature_std, torch.ones(320))
    x = torch.ones(1, 512, 21)
    features = torch.full((1, 128, 320), 5.)
    latent = gt.encode(x)
    latent[:, :, 0, 0] += 3.
    expected = gt.decode_future(latent, latent[:, -1:].expand(-1, 12, -1, -1), 48)
    torch.testing.assert_close(route.forecast(x, features), expected)


def test_encoder_bridge_and_shared_product_receive_gradient_and_branch_changes_output(model_module):
    gt = positive_gt()
    gt.requires_grad_(False)
    route = model_module.ForecastRoute(gt, RepeatFoundation(), torch.zeros(320), torch.ones(320))
    x = torch.linspace(.1, .8, 512)[None, :, None].expand(1, -1, 21)
    features = torch.ones(1, 128, 320)
    optimizer = torch.optim.Adam(route.parameters(), lr=1e-4)
    route.forecast(x, features).square().mean().backward()
    for name in ["encoder.0.weight", "encoder.1.weight", "decoder.0.weight",
                 "decoder.1.weight", "product.s00", "product.s01", "product.s10", "product.s11"]:
        gradient = dict(gt.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all(), name
        assert torch.count_nonzero(gradient), name
    assert route.W.weight.grad is not None and torch.count_nonzero(route.W.weight.grad)
    assert torch.isfinite(route.W.weight.grad).all()
    optimizer.step()
    assert not torch.equal(route.forecast(x, features), route.forecast(x))
    torch.testing.assert_close(gt.product.graph, torch.eye(21), rtol=0, atol=0)


@pytest.mark.parametrize("arcsinh", [False, True])
def test_safe_norm_preserves_official_forward_and_constant_gradient(model_module, arcsinh):
    from chronos.chronos_bolt import InstanceNorm
    native = InstanceNorm(use_arcsinh=arcsinh)
    safe = model_module.SafeInstanceNorm(use_arcsinh=arcsinh)
    x = torch.tensor([[2., 2., 2., 2.], [-2., -1., 0., 3.]])
    actual, statistics = safe(x)
    expected, native_statistics = native(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_statistic, expected_statistic in zip(statistics, native_statistics):
        torch.testing.assert_close(actual_statistic, expected_statistic, rtol=0, atol=0)
    torch.testing.assert_close(safe(x, statistics)[0], native(x, native_statistics)[0], rtol=0, atol=0)
    constant = torch.full((2, 16), 3., requires_grad=True)
    scaled, (location, scale) = safe(constant)
    (scaled.square().sum() + location.sum() + scale.sum()).backward()
    assert torch.isfinite(constant.grad).all()
    assert torch.count_nonzero(constant.grad)


def test_foundation_microbatch_keeps_coordinate_order_gradient_and_frozen_state(model_module, fake_loader, tmp_path):
    foundation = model_module.FrozenFoundation(tmp_path, device="cpu", row_microbatch=3)
    assert list(tmp_path.iterdir()) == []
    latent = torch.arange(2 * 13 * 2 * 2, dtype=torch.float32).reshape(2, 13, 2, 2).requires_grad_()
    forecast = foundation.forecast(latent, horizon=5)
    expected = latent[:, -1:].expand(-1, 5, -1, -1)
    torch.testing.assert_close(forecast, expected, rtol=1e-6, atol=1e-5)
    assert foundation.calls == 3 and foundation.streams == 8
    assert foundation.requested_context_length == 13 and foundation.actual_context_length == 8
    assert foundation.max_context == 8
    assert foundation.context_records[-1]["row_sizes"] == [3, 3, 2]
    forecast.square().mean().backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all()
    assert torch.count_nonzero(latent.grad[:, -8:])
    assert torch.count_nonzero(latent.grad[:, :-8]) == 0
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in foundation.model.parameters())
    assert foundation.verify_unchanged()["unchanged"]
    assert not foundation.model.training


def test_foundation_row_splitting_and_batch_permutation_are_invariant(model_module, fake_loader, tmp_path):
    foundation = model_module.FrozenFoundation(tmp_path, device="cpu", row_microbatch=1)
    latent = torch.arange(8 * 8 * 21 * 2, dtype=torch.float32).reshape(8, 8, 21, 2) / 100
    individual = torch.cat([foundation.forecast(window[None], 12) for window in latent])
    foundation.row_microbatch = 128
    batched = foundation.forecast(latent, 12)
    torch.testing.assert_close(batched, individual)
    order = torch.tensor([5, 2, 7, 0, 4, 1, 6, 3])
    channel_order = torch.arange(20, -1, -1)
    shuffled = foundation.forecast(latent[order][:, :, channel_order], 12)
    torch.testing.assert_close(shuffled, batched[order][:, :, channel_order])


def test_frozen_foundation_remains_fixed_during_route_training(model_module, fake_loader, tmp_path):
    foundation = model_module.FrozenFoundation(tmp_path, device="cpu", row_microbatch=32)
    route = model_module.ForecastRoute(positive_gt(), foundation, torch.zeros(320), torch.ones(320))
    route.train()
    x = torch.ones(1, 512, 21)
    optimizer = torch.optim.Adam(route.parameters(), lr=1e-4)
    route.forecast(x, torch.ones(1, 128, 320)).square().mean().backward()
    optimizer.step()
    assert torch.count_nonzero(route.W.weight)
    assert foundation.verify_unchanged() == {"unchanged": True, "parameter_gradients_absent": True}
    assert not foundation.model.training
    assert all("foundation" not in name for name in route.state_dict())


def test_adapter_checkpoint_roundtrip_excludes_foundation_weights(model_module, fake_loader, tmp_path):
    foundation = model_module.FrozenFoundation(tmp_path, device="cpu")
    route = model_module.ForecastRoute(positive_gt(), foundation, torch.ones(320), torch.full((320,), 2.))
    with torch.no_grad():
        route.W.weight.fill_(.001)
    state = route.adapter_state()
    assert not any("foundation" in name for name in state)
    restored = model_module.ForecastRoute(positive_gt(), foundation, torch.zeros(320), torch.ones(320))
    restored.load_adapter_state(state)
    x = torch.ones(1, 512, 21)
    features = torch.ones(1, 128, 320)
    torch.testing.assert_close(restored.forecast(x, features), route.forecast(x, features), rtol=0, atol=0)
    with torch.no_grad():
        route.W.weight.zero_()
    assert torch.count_nonzero(state["W.weight"])


def test_fresh_gt_initialization_is_rng_isolated(model_module):
    rng = torch.get_rng_state().clone()
    first = model_module.create_gt(torch.eye(21), seed=92601)
    assert torch.equal(rng, torch.get_rng_state())
    second = model_module.create_gt(torch.eye(21), seed=92601)
    assert all(torch.equal(value, second.state_dict()[key]) for key, value in first.state_dict().items())


def test_fresh_gt_does_not_train_a_gradient_enabled_graph(model_module):
    graph = torch.eye(21, requires_grad=True)
    gt = model_module.create_gt(graph, seed=92601)
    assert not gt.product.graph.requires_grad
    assert not gt.product.graph.grad_fn


@pytest.mark.parametrize("x", [torch.ones(1, 513, 21), torch.ones(1, 512, 20),
                               torch.ones(0, 512, 21), torch.full((1, 512, 21), float("nan"))])
def test_route_rejects_invalid_or_nonfinite_history(model_module, x):
    route = model_module.ForecastRoute(positive_gt(), RepeatFoundation())
    with pytest.raises(ValueError):
        route.forecast(x)


@pytest.mark.parametrize("features", [torch.ones(1, 512, 320), torch.ones(1, 128, 319),
                                      torch.full((1, 128, 320), float("inf"))])
def test_route_rejects_invalid_pooled_features(model_module, features):
    route = model_module.ForecastRoute(positive_gt(), RepeatFoundation(), torch.zeros(320), torch.ones(320))
    with pytest.raises(ValueError):
        route.forecast(torch.ones(1, 512, 21), features)


def test_route_rejects_unregistered_features_or_invalid_statistics(model_module):
    with pytest.raises(ValueError):
        model_module.ForecastRoute(positive_gt(), RepeatFoundation(), torch.zeros(320))
    with pytest.raises(ValueError):
        model_module.ForecastRoute(positive_gt(), RepeatFoundation(), torch.zeros(320), -torch.ones(320))
    route = model_module.ForecastRoute(positive_gt(), RepeatFoundation())
    with pytest.raises(ValueError):
        route.forecast(torch.ones(1, 512, 21), torch.ones(1, 128, 320))
    with pytest.raises(ValueError):
        model_module.ForecastRoute.pool_features(torch.ones(1, 513, 320))


@pytest.mark.parametrize("latent,horizon", [(torch.ones(1, 8), 5), (torch.ones(1, 8, 2), 13),
                                            (torch.ones(1, 8, 2), 0),
                                            (torch.full((1, 8, 2), float("inf")), 5)])
def test_foundation_rejects_invalid_or_nonfinite_latent(model_module, fake_loader, tmp_path, latent, horizon):
    foundation = model_module.FrozenFoundation(tmp_path, device="cpu")
    with pytest.raises(ValueError):
        foundation.forecast(latent, horizon)
