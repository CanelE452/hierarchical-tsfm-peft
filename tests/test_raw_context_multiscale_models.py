import copy
import importlib
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from research.informs_broad_validation_20261015_v1.networks import ForecastAdapter, LAGS
from research.informs_broad_validation_20261015_v1.training import FastWindows
from research.raw_context_multiscale_20261018_v1 import models


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def base_model(channels=4, latent_k=4, length=64, horizon=8, nonidentity=False):
    torch.manual_seed(92601)
    basis = torch.randn(channels, latent_k)
    base = ForecastAdapter("B2", length, horizon, channels, latent_k,
                           torch.arange(channels).float() / 10, basis)
    if nonidentity:
        with torch.no_grad():
            for name, parameter in base.front.named_parameters():
                if ".output." in name:
                    parameter.normal_(std=.015)
    return base


class FrozenRows(nn.Module):
    """Independent deterministic rows, supporting both input lengths."""
    def __init__(self, horizon=8):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.7), requires_grad=False)
        self.horizon = horizon
        self.seen = []

    def forward(self, z):
        if z.requires_grad:
            z.retain_grad()
        self.seen.append(z)
        offsets = torch.arange(self.horizon, device=z.device, dtype=z.dtype)[None, :, None]
        return (z[:, -1:] * self.weight + .2 * z.mean(dim=1, keepdim=True)
                + .03 * z[:, -2:-1].tanh() * offsets)


@pytest.mark.parametrize("channels,k,count", [(21, 12, 36725), (32, 8, 37472)])
def test_conversion_preserves_registered_parameter_counts_states_and_rng(channels, k, count):
    base = base_model(channels, k, length=512, horizon=48 if channels == 21 else 12)
    before = {key: value.clone() for key, value in base.state_dict().items()}
    parameter_ids = {name: id(value) for name, value in base.named_parameters()}
    rng = torch.get_rng_state().clone()
    assert sum(value.numel() for value in base.parameters()) == count
    for mode in ("S", "A", "B"):
        converted = models.convert_model(base, mode)
        assert converted is base and isinstance(base, ForecastAdapter)
        assert parameter_ids == {name: id(value) for name, value in base.named_parameters()}
        assert list(base.state_dict()) == list(before)
        assert all(torch.equal(value, base.state_dict()[key]) for key, value in before.items())
        assert torch.equal(rng, torch.get_rng_state())
        assert sum(value.numel() for value in base.parameters()) == count
        torch.testing.assert_close(base.D[0], torch.linalg.pinv(base.E), atol=0, rtol=0)
        assert torch.count_nonzero(base.D[1:]) == 0


def test_cp0_s_and_a_equal_original_and_keep_pca_column_order():
    base = base_model()
    x = torch.randn(3, 64, 4)
    foundation = FrozenRows()
    expected = base(x, foundation)
    s = models.convert_model(copy.deepcopy(base), "S")
    a = models.convert_model(copy.deepcopy(base), "A")
    torch.testing.assert_close(s(x, foundation), expected, atol=0, rtol=0)
    torch.testing.assert_close(a(x, foundation), expected, atol=models.ATOL, rtol=models.RTOL)
    raw, context = a.latent_branches(x)
    torch.testing.assert_close(raw, (x - base.center) @ base.E[:, :2], atol=0, rtol=0)
    torch.testing.assert_close(context, (x - base.center) @ base.E[:, 2:], atol=0, rtol=0)
    assert torch.equal(s.E, a.E) and torch.equal(s.D, a.D)


def test_a_raw_half_bypasses_p_and_context_half_retains_p():
    a = models.convert_model(base_model(nonidentity=True), "A")
    x = torch.randn(2, 64, 4)
    raw, context = a.latent_branches(x)
    expected_raw = (x - a.center) @ a.E[:, :2]
    expected_context = (a.front(x) - a.center) @ a.E[:, 2:]
    torch.testing.assert_close(raw, expected_raw, atol=0, rtol=0)
    torch.testing.assert_close(context, expected_context, atol=0, rtol=0)
    assert not torch.allclose(context, (x - a.center) @ a.E[:, 2:])


def test_ramp_impulse_constants_and_single_channel_block_time_helpers():
    ramp = torch.arange(512).float()[None, :, None]
    pooled = models.block_mean(ramp)
    torch.testing.assert_close(pooled[0, :, 0], torch.arange(.5, 512, 2), atol=0, rtol=0)
    restored = models.repeat_time(pooled)
    assert restored.shape == ramp.shape
    assert restored[0, -2:, 0].tolist() == [510.5, 510.5]
    assert not torch.equal(restored, ramp)
    impulse = torch.zeros(2, 512, 1)
    impulse[0, 1, 0] = 1
    impulse[1, 510, 0] = 2
    coarse = models.block_mean(impulse)
    assert coarse[0, 0, 0] == .5 and coarse[1, 255, 0] == 1
    assert torch.count_nonzero(coarse) == 2
    torch.testing.assert_close(models.repeat_time(models.block_mean(torch.full((2, 512, 1), 7.))),
                               torch.full((2, 512, 1), 7.), atol=0, rtol=0)
    assert models.block_mean(ramp, 1) is ramp
    assert models.repeat_time(ramp, 1) is ramp


@pytest.mark.parametrize("horizon,coarse_horizon", [(48, 24), (12, 6)])
def test_original_and_coarse_time_shapes_and_future_block_offsets(horizon, coarse_horizon):
    b = models.convert_model(base_model(length=512, horizon=horizon), "B")
    x = torch.randn(2, 512, 4)
    f = FrozenRows(horizon)
    past, future = b.forecast_latents(x, f)
    assert [tuple(value.shape) for value in f.seen] == [(2, 512, 2), (2, 256, 2)]
    assert past.shape == (2, 512, 4) and future.shape == (2, horizon, 4)
    assert torch.equal(past[:, 0::2, 2:], past[:, 1::2, 2:])
    assert torch.equal(future[:, 0::2, 2:], future[:, 1::2, 2:])
    contract = models.time_contract(512, horizon)
    assert contract["coarse_context"] == 256
    assert contract["coarse_horizon"] == coarse_horizon
    assert contract["future_offset_blocks"] == tuple((2 * j, 2 * j + 1) for j in range(coarse_horizon))
    assert contract["decoder_lags"] == LAGS and contract["anchor_count"] == 1


def test_coarse_future_repeats_blocks_without_shift_or_new_interpolation():
    b = models.convert_model(base_model(length=512, horizon=48), "B")
    x = torch.randn(1, 512, 4)
    seen = []

    def f(z):
        seen.append(z.shape)
        return torch.arange(48).float()[None, :, None].expand(len(z), -1, z.shape[-1])

    _, future = b.forecast_latents(x, f)
    assert seen == [torch.Size((1, 512, 2)), torch.Size((1, 256, 2))]
    assert future[0, :, 2].tolist() == [float(j // 2) for j in range(48)]
    assert future[0, :, 0].tolist() == [float(j) for j in range(48)]


def test_decoder_is_lag_major_original_grid_and_adds_anchor_once():
    b = models.convert_model(base_model(length=512, horizon=48), "B")
    x = torch.arange(512).float()[None, :, None].expand(1, -1, 4).clone()
    with torch.no_grad():
        b.E.copy_(torch.eye(4))
        b.center.zero_()
        b.D.zero_()
        b.D[-1, 2, 1] = 3
        b.D[1, 0, 3] = 2
    f = FrozenRows(48)
    z, u = b.forecast_latents(x, f)
    joined = torch.cat((z, u), dim=1)
    expected = x[:, -1:].expand(-1, 48, -1).clone()
    for h in range(48):
        for lag_index, lag in enumerate(LAGS):
            expected[:, h] += (joined[:, 512 + h - lag] - z[:, 511 - lag]) @ b.D[lag_index]
    torch.testing.assert_close(b(x, f), expected, atol=models.ATOL, rtol=models.RTOL)
    assert (joined[0, 480, 2] - z[0, 479, 2]).item() == 2
    assert (joined[0, 448, 2] - z[0, 447, 2]).item() == 2
    with torch.no_grad():
        b.D.zero_()
    torch.testing.assert_close(b(x, f), x[:, -1:].expand(-1, 48, -1), atol=0, rtol=0)


@pytest.mark.parametrize("mode,scale,schedule", [("A", 2, "two_call"), ("B", 1, "native")])
def test_a_native_two_call_and_b_scale_one_output_input_parameter_gradient_parity(mode, scale, schedule):
    a = models.convert_model(base_model(nonidentity=True), "A")
    control = models.convert_model(copy.deepcopy(a), mode, scale=scale, schedule=schedule)
    x_a = torch.randn(3, 64, 4, requires_grad=True)
    x_control = x_a.detach().clone().requires_grad_()
    f_a, f_control = FrozenRows(), FrozenRows()
    target = torch.randn(3, 8, 4)
    out_a, out_control = a(x_a, f_a), control(x_control, f_control)
    torch.testing.assert_close(out_a, out_control, atol=models.ATOL, rtol=models.RTOL)
    (out_a - target).square().mean().backward()
    (out_control - target).square().mean().backward()
    torch.testing.assert_close(x_a.grad, x_control.grad, atol=models.ATOL, rtol=models.RTOL)
    for (name_a, parameter_a), (name_b, parameter_b) in zip(a.named_parameters(), control.named_parameters()):
        assert name_a == name_b and parameter_a.grad is not None and parameter_b.grad is not None
        torch.testing.assert_close(parameter_a.grad, parameter_b.grad, atol=models.ATOL, rtol=models.RTOL)
    assert len(f_a.seen) == 1 and len(f_control.seen) == 2


@pytest.mark.parametrize("mode", ["A", "B"])
def test_live_frozen_foundation_input_gradients_and_trainable_weights_change(mode):
    model = models.convert_model(base_model(nonidentity=True), mode)
    f = FrozenRows()
    frozen_before = f.weight.detach().clone()
    x = torch.randn(2, 64, 4, requires_grad=True)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0)
    model(x, f).square().mean().backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0
               for value in f.seen)
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for name, value in model.named_parameters():
        assert value.grad is not None and torch.isfinite(value.grad).all(), name
    optimizer.step()
    for prefix in ("E", "D", "front."):
        assert any(not torch.equal(before[name], value) for name, value in model.named_parameters()
                   if name.startswith(prefix))
    assert f.weight.grad is None and torch.equal(f.weight, frozen_before)
    assert not any(id(f.weight) == id(value) for group in optimizer.param_groups for value in group["params"])


def test_cp0_does_not_require_nonzero_hidden_gradient_on_first_backward():
    model = models.convert_model(base_model(), "B")
    model(torch.randn(2, 64, 4), FrozenRows()).square().mean().backward()
    assert torch.count_nonzero(model.front.mixer.temporal.input.weight.grad) == 0
    assert model.front.mixer.temporal.output.weight.grad.abs().sum() > 0
    assert model.E.grad.abs().sum() > 0 and model.D.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["S", "A", "B"])
def test_batch_permutation_individual_windows_and_future_canary(mode):
    model = models.convert_model(base_model(nonidentity=True), mode)
    f = FrozenRows()
    x = torch.randn(4, 64, 4) * torch.arange(1, 5)[:, None, None]
    together = model(x, f)
    individually = torch.cat([model(value[None], f) for value in x])
    torch.testing.assert_close(together, individually, atol=models.ATOL, rtol=models.RTOL)
    order = torch.tensor([3, 0, 2, 1])
    torch.testing.assert_close(model(x[order], f), together[order], atol=models.ATOL, rtol=models.RTOL)
    values = np.random.default_rng(21).normal(size=(96, 4)).astype(np.float32)
    task = dict(physical=values.copy(), observed=np.ones_like(values, bool), L=64, H=8)
    first, first_target, _ = FastWindows(task, np.zeros(4), np.ones(4), "cpu").get(np.array([64]))
    task["physical"][64:] = 1e9
    second, second_target, _ = FastWindows(task, np.zeros(4), np.ones(4), "cpu").get(np.array([64]))
    assert torch.equal(first, second) and not torch.equal(first_target, second_target)
    torch.testing.assert_close(model(first, f), model(second, f), atol=0, rtol=0)


def factory_context(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    broad = root / "research/informs_broad_validation_20261015_v1"
    monkeypatch.syspath_prepend(str(broad))
    modules = {key: importlib.import_module(name) for key, name in
               (("fit", "fit_base"), ("training", "training"), ("data", "data_contract"))}
    return dict(repo=root, modules=modules, initialization_events=[])


def factory_task():
    values = np.random.default_rng(73).normal(size=(144, 4)).astype(np.float32)
    return dict(physical=values, observed=np.ones_like(values, bool), L=64, H=8,
                columns=["a", "b", "c", "d"], data_sha256="synthetic",
                split_ranges=dict(A_fit=(0, 80), A_stop=(80, 96), M_cal=(96, 112),
                                  V_select=(112, 128), T_final=(128, 144)))


def test_factory_reuses_seeded_pca_total_pinv_and_restores_all_hooks(monkeypatch):
    ctx, task = factory_context(monkeypatch), factory_task()
    fit, data = ctx["modules"]["fit"], ctx["modules"]["data"]
    before = fit.create_model, fit._source_contract, data.split_ranges
    spec = dict(family="B2", latent_k=2, seed=92601, revin=False, device="cpu")
    original, _ = fit.create_model(spec, task, 80, FrozenRows())
    original_rng = torch.get_rng_state().clone()
    expected = {key: value.clone() for key, value in original.state_dict().items()}
    with models.model_factory(ctx) as create:
        for mode in ("S", "A", "B"):
            model, norm = create(dict(spec, scientific_identity=dict(raw_context_mode=mode)), task, 80, FrozenRows())
            assert isinstance(model, type(original))
            assert all(torch.equal(value, model.state_dict()[key]) for key, value in expected.items())
            assert torch.equal(torch.get_rng_state(), original_rng)
            assert norm["raw_context"]["mode"] == mode
            assert norm["raw_context"]["original_channel_anchor_count"] == 1
            assert fit.create_model is create
        _, sources = fit._source_contract()
        assert "research/raw_context_multiscale_20261018_v1/models.py" in sources
        assert "research/b2_revin_20261018_v1/revin.py" in sources
        assert data.split_ranges(task)["A_fit"] == (0, 80)
        for key in ("P_initial_sha256", "E_initial_sha256", "D_initial_sha256", "center_sha256", "scaler_sha256"):
            assert len({event[key] for event in ctx["initialization_events"]}) == 1
    assert (fit.create_model, fit._source_contract, data.split_ranges) == before
    standalone, _ = models.create_model(dict(spec, scientific_identity=dict(raw_context_mode="B")),
                                       task, 80, FrozenRows(), modules=ctx["modules"])
    assert standalone.raw_context_mode == "B"
    assert (fit.create_model, fit._source_contract, data.split_ranges) == before


@pytest.mark.parametrize("identity", [
    {},
    {"source": "historical-parent"},
    "94862114ee8d43b02e03c8e4fb5facdabfeb10d52da41b6150e221a793fb1563",
])
def test_factory_legacy_s_identity_keeps_inherited_train_only_path(monkeypatch, identity):
    ctx, task = factory_context(monkeypatch), factory_task()
    fit = ctx["modules"]["fit"]
    spec = dict(family="B2", latent_k=2, seed=92601, revin=False, device="cpu",
                scientific_identity=identity)
    inherited_create = fit.create_model
    inherited, _ = inherited_create(spec, task, 80, FrozenRows())
    inherited_rng = torch.get_rng_state().clone()
    inherited_state = {key: value.clone() for key, value in inherited.state_dict().items()}
    forwarded = []

    def record_create(internal, *args):
        forwarded.append(copy.deepcopy(internal))
        return inherited_create(internal, *args)

    monkeypatch.setattr(fit, "create_model", record_create)
    x = torch.from_numpy(task["physical"][0:64].copy())[None]
    expected = inherited(x, FrozenRows())
    with models.model_factory(ctx) as create:
        model, norm = create(spec, task, 80, FrozenRows())
        assert forwarded == [spec]
        assert type(model) is type(inherited)
        assert not isinstance(model, models._RawContextMixin)
        assert "raw_context" not in norm
        assert list(model.state_dict()) == list(inherited_state)
        assert all(torch.equal(value, model.state_dict()[key]) for key, value in inherited_state.items())
        assert torch.equal(torch.get_rng_state(), inherited_rng)
        torch.testing.assert_close(model(x, FrozenRows()), expected, atol=0, rtol=0)
    assert fit.create_model is record_create


@pytest.mark.parametrize("identity", [
    None, [], 17, "not-a-legacy-fingerprint", "0" * 63,
    {"raw_context_scale": 2},
    {"raw_context_mode": None}, {"raw_context_mode": "bad"},
])
def test_factory_malformed_identity_fails_before_seeded_model_creation(monkeypatch, identity):
    ctx = factory_context(monkeypatch)
    fit = ctx["modules"]["fit"]
    original = fit.create_model
    spec = dict(family="B2", latent_k=2, seed=92601, revin=False, device="cpu",
                scientific_identity=identity)
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="identity|mode"):
        with models.model_factory(ctx) as create:
            create(spec, factory_task(), 80, FrozenRows())
    assert fit.create_model is original
    assert ctx["initialization_events"] == []
    assert torch.equal(torch.get_rng_state(), rng)


def test_factory_restores_hooks_on_failure_and_rejects_bad_source_or_revin(monkeypatch):
    ctx = factory_context(monkeypatch)
    fit, data = ctx["modules"]["fit"], ctx["modules"]["data"]
    before = fit.create_model, fit._source_contract, data.split_ranges
    spec = dict(family="B2", latent_k=2, seed=92601, revin=False, device="cpu",
                scientific_identity=dict(raw_context_mode="B"))
    with pytest.raises(ValueError, match="mode"):
        with models.model_factory(ctx) as create:
            create(dict(spec, scientific_identity=dict(raw_context_mode="bad")), factory_task(), 80, FrozenRows())
    assert (fit.create_model, fit._source_contract, data.split_ranges) == before
    with pytest.raises(ValueError, match="RevIN OFF"):
        with models.model_factory(ctx) as create:
            create(dict(spec, revin=True), factory_task(), 80, FrozenRows())
    assert (fit.create_model, fit._source_contract, data.split_ranges) == before
    ctx["scientific_source_hashes"] = {"research/raw_context_multiscale_20261018_v1/models.py": "0" * 64}
    with pytest.raises(ValueError, match="source"):
        with models.model_factory(ctx):
            pass
    assert (fit.create_model, fit._source_contract, data.split_ranges) == before


def test_fixed_k_validation_and_foundation_output_shape_errors():
    with pytest.raises(ValueError, match="even"):
        models.convert_model(base_model(latent_k=3), "A")
    with pytest.raises(ValueError, match="Horizon"):
        models.convert_model(base_model(horizon=5), "B")
    with pytest.raises(ValueError, match="scales"):
        models.block_mean(torch.ones(1, 64, 1), 4)
    with pytest.raises(ValueError, match="divisible"):
        models.block_mean(torch.ones(1, 63, 1))
    model = models.convert_model(base_model(), "B")
    with pytest.raises(ValueError, match="fixed total K"):
        model(torch.randn(1, 64, 4), FrozenRows(), latent_k=2)
    with pytest.raises(ValueError, match="Foundation"):
        model(torch.randn(1, 64, 4), lambda z: z[:, :1])
    with pytest.raises(ValueError, match="Foundation"):
        model(torch.randn(1, 64, 4), lambda z: np.ones((1, 8, 2)))
