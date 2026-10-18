import importlib
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from research.informs_broad_validation_20261015_v1.networks import ForecastAdapter
from research.informs_broad_validation_20261015_v1.training import FastWindows


def revin_module():
    try:
        return importlib.import_module("research.b2_revin_20261018_v1.revin")
    except ModuleNotFoundError:
        pytest.fail("The registered B2 outer RevIN implementation is missing")


def base_model(family="B2", channels=3):
    torch.manual_seed(92601)
    return ForecastAdapter(family, 64, 5, channels, 2, np.zeros(channels, np.float32),
                           np.eye(channels, 2, dtype=np.float32))


def foundation(z):
    return z[:, -1:].expand(-1, 5, -1) + .25 * z[:, -6:-1]


def test_off_returns_original_model_without_state_or_rng_changes():
    module = revin_module()
    base = base_model()
    rng = torch.get_rng_state().clone()
    state = {key: value.clone() for key, value in base.state_dict().items()}
    wrapped = module.wrap_model(base, False)
    assert wrapped is base
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, wrapped.state_dict()[key]) for key, value in state.items())
    x = torch.arange(2 * 64 * 3, dtype=torch.float32).reshape(2, 64, 3) / 100
    assert torch.equal(wrapped(x, foundation, 2), base(x, foundation, 2))


def test_roundtrip_uses_detached_per_call_biased_statistics_and_official_affine_inverse():
    module = revin_module()
    revin = module.OuterRevIN(3)
    with torch.no_grad():
        revin.affine_weight.copy_(torch.tensor([.75, 1.25, 2.]))
        revin.affine_bias.copy_(torch.tensor([.5, -1., 2.]))
    x = torch.arange(2 * 7 * 3, dtype=torch.float32).reshape(2, 7, 3).requires_grad_()
    normalized, context = revin.normalize(x)
    mean = x.mean(1, keepdim=True).detach()
    std = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5).detach()
    expected = (x - mean) / std * revin.affine_weight + revin.affine_bias
    torch.testing.assert_close(normalized, expected, rtol=0, atol=0)
    assert not context.mean.requires_grad and not context.stdev.requires_grad
    restored = revin.inverse(normalized, context)
    official = (normalized - revin.affine_bias) / (revin.affine_weight + 1e-10) * std + mean
    torch.testing.assert_close(restored, official, rtol=0, atol=0)
    torch.testing.assert_close(restored, x, rtol=2e-6, atol=2e-6)


def test_wrapper_inverse_includes_anchor_once_before_global_inverse():
    module = revin_module()
    base = base_model()
    wrapped = module.wrap_model(base, True)
    x = torch.randn(2, 64, 3) * torch.tensor([2., 4., 6.]) + torch.tensor([3., -7., 12.])
    normalized, context = wrapped.revin.normalize(x)
    expected = wrapped.revin.inverse(base(normalized, foundation, 2), context)
    actual = wrapped(x, foundation, 2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with torch.no_grad():
        base.D.zero_()
    torch.testing.assert_close(wrapped(x, foundation, 2), x[:, -1:].expand(-1, 5, -1),
                               rtol=2e-6, atol=2e-6)


def test_batch_order_and_interleaved_windows_keep_own_statistics():
    module = revin_module()
    wrapped = module.wrap_model(base_model(), True)
    x = torch.randn(8, 64, 3) * torch.arange(1, 9)[:, None, None] + torch.arange(8)[:, None, None]
    batched = wrapped(x, foundation, 2)
    individually = torch.cat([wrapped(window[None], foundation, 2) for window in x])
    torch.testing.assert_close(batched, individually, rtol=2e-6, atol=2e-6)
    order = torch.tensor([5, 2, 7, 0, 4, 1, 6, 3])
    torch.testing.assert_close(wrapped(x[order], foundation, 2), batched[order], rtol=2e-6, atol=2e-6)
    a, context_a = wrapped.revin.normalize(x[:1])
    b, context_b = wrapped.revin.normalize(x[1:2])
    torch.testing.assert_close(wrapped.revin.inverse(a, context_a), x[:1], rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(wrapped.revin.inverse(b, context_b), x[1:2], rtol=2e-6, atol=2e-6)


def test_constant_and_causally_filled_windows_are_finite():
    module = revin_module()
    physical = np.tile(np.array([[3., 7., -2.]], np.float32), (75, 1))
    observed = np.ones_like(physical, dtype=bool)
    observed[9:15, 1] = False
    observed[:70, 2] = False
    task = dict(physical=physical, observed=observed, L=64, H=5)
    windows = FastWindows(task, np.zeros(3, np.float32), np.ones(3, np.float32), "cpu")
    x, _, _ = windows.get(np.array([64, 65], np.int64))
    wrapped = module.wrap_model(base_model(), True)
    assert torch.isfinite(wrapped(x, foundation, 2)).all()
    assert torch.equal(x[:, :, 2], torch.zeros_like(x[:, :, 2]))
    normalized, context = wrapped.revin.normalize(torch.full((1, 64, 3), 7.))
    assert torch.isfinite(normalized).all()
    torch.testing.assert_close(context.stdev, torch.full((1, 1, 3), 1e-5 ** .5))


def test_frozen_foundation_keeps_input_gradient_to_encoder_and_revin():
    module = revin_module()
    wrapped = module.wrap_model(base_model(), True)
    frozen = nn.Linear(64, 5, bias=False).requires_grad_(False).eval()
    seen = []

    def frozen_f(z):
        z.retain_grad()
        seen.append(z)
        return frozen(z.transpose(1, 2)).transpose(1, 2)

    x = torch.randn(4, 64, 3).requires_grad_()
    loss = (wrapped(x, frozen_f, 2) - torch.randn(4, 5, 3)).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert seen[0].grad is not None and seen[0].grad.abs().sum() > 0
    assert wrapped.E.grad is not None and wrapped.E.grad.abs().sum() > 0
    assert wrapped.revin.affine_weight.grad is not None and wrapped.revin.affine_weight.grad.abs().sum() > 0
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert frozen.weight.grad is None and not frozen.training
    assert all(key.startswith(("base.", "revin.")) for key in wrapped.state_dict())
    assert len(list(wrapped.parameters())) == len(list(wrapped.base.parameters())) + 2
    assert sum(value.numel() for value in wrapped.parameters()) - sum(
        value.numel() for value in wrapped.base.parameters()) == 2 * wrapped.channels


def test_wrapper_rejects_nonboolean_enabled():
    module = revin_module()
    with pytest.raises((TypeError, ValueError), match="bool"):
        module.wrap_model(base_model(), "OFF")


def scientific_modules(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    campaign = root / "research/informs_broad_validation_20261015_v1"
    monkeypatch.syspath_prepend(str(campaign))
    modules = {key: importlib.import_module(name) for key, name in
               (("fit", "fit_base"), ("training", "training"), ("data", "data_contract"))}
    assert all(Path(value.__file__).resolve().parent == campaign for value in modules.values())
    return dict(repo=root, modules=modules, initialization_events=[])


def small_task():
    rng = np.random.default_rng(92601)
    physical = rng.normal(size=(120, 3)).astype(np.float32)
    return dict(physical=physical, observed=np.ones_like(physical, dtype=bool), L=64, H=5,
                columns=["a", "b", "c"], data_sha256="fixture-data",
                split_ranges=dict(A_fit=(0, 75), A_stop=(75, 90), M_cal=(90, 100),
                                  V_select=(100, 120), T_final=(120, 150)))


def test_factory_preserves_base_initialization_and_restores_global_hooks(monkeypatch):
    module = revin_module()
    assert hasattr(module, "model_factory"), "Scoped RevIN model factory is missing"
    ctx = scientific_modules(monkeypatch)
    fit, data = ctx["modules"]["fit"], ctx["modules"]["data"]
    original_create, original_sources, original_splits = fit.create_model, fit._source_contract, data.split_ranges
    task = small_task()
    spec = dict(family="B2", seed=92601, latent_k=2, device="cpu", revin=False)
    original, _ = original_create(spec, task, 75, foundation)
    expected = {key: value.clone() for key, value in original.state_dict().items()}
    with module.model_factory(ctx) as create:
        off, norm_off = create(spec, task, 75, foundation)
        on, norm_on = create(dict(spec, revin=True), task, 75, foundation)
        assert all(torch.equal(off.state_dict()[key], value) for key, value in expected.items())
        assert all(torch.equal(on.base.state_dict()[key], value) for key, value in expected.items())
        assert norm_off["revin"]["enabled"] is False and norm_on["revin"]["enabled"] is True
        assert data.split_ranges(task) == {key: tuple(value) for key, value in task["split_ranges"].items()}
        digest, sources = fit._source_contract()
        assert digest != original_sources()[0]
        assert "research/b2_revin_20261018_v1/revin.py" in sources
        assert len(ctx["initialization_events"]) == 2
        for key in ("P_initial_sha256", "E_initial_sha256", "D_initial_sha256", "scaler_sha256", "center_sha256"):
            assert ctx["initialization_events"][0][key] == ctx["initialization_events"][1][key]
        with pytest.raises(ValueError, match="family"):
            create(dict(spec, family="ADAPTS"), task, 75, foundation)
        with pytest.raises((TypeError, ValueError), match="bool"):
            create(dict(spec, revin="ON"), task, 75, foundation)
    assert fit.create_model is original_create and fit._source_contract is original_sources
    assert data.split_ranges is original_splits


def test_factory_rejects_changed_registered_source(monkeypatch):
    module = revin_module()
    assert hasattr(module, "model_factory"), "Scoped RevIN model factory is missing"
    ctx = scientific_modules(monkeypatch)
    ctx["scientific_source_hashes"] = {"research/b2_revin_20261018_v1/revin.py": "0" * 64}
    with pytest.raises(ValueError, match="source"):
        with module.model_factory(ctx):
            pass


def test_factory_hooks_restore_when_creation_raises(monkeypatch):
    module = revin_module()
    ctx = scientific_modules(monkeypatch)
    fit, data = ctx["modules"]["fit"], ctx["modules"]["data"]
    before = fit.create_model, fit._source_contract, data.split_ranges
    with pytest.raises(ValueError, match="family"):
        with module.model_factory(ctx) as create:
            create(dict(family="ADAPTS", revin=True), small_task(), 75, foundation)
    assert (fit.create_model, fit._source_contract, data.split_ranges) == before


class FakeFrozenFoundation:
    def __init__(self):
        self.initial_hash = "fixture-frozen-foundation"
        self.horizon = 5
        self.row_microbatch = 128
        self.streams = self.row_calls = self.physical_streams = self.physical_calls = 0

    def __call__(self, z):
        self.streams += len(z) * z.shape[2]
        self.physical_streams += len(z) * z.shape[2]
        self.row_calls += 1
        self.physical_calls += 1
        return foundation(z)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("tamper", [None, "policy", "source"])
def test_serialized_whole_physical_route_matches_training_prediction(monkeypatch, tmp_path, enabled, batch, tamper):
    module = revin_module()
    try:
        deployment = importlib.import_module("research.b2_revin_20261018_v1.deployment")
    except ModuleNotFoundError:
        pytest.fail("The serialized whole-path RevIN deployment route is missing")
    ctx = scientific_modules(monkeypatch)
    ctx["modules"]["deployment"] = importlib.import_module("deployment")
    fit, training, data = (ctx["modules"][key] for key in ("fit", "training", "data"))
    task = small_task()
    frozen = FakeFrozenFoundation()
    spec = dict(family="B2", run_id="fixture", seed=92601, latent_k=2, device="cpu", revin=enabled)
    with module.model_factory(ctx) as create:
        model, norm = create(spec, task, 90, frozen)
    state = training.cpu_state(model)
    bundle = dict(model=state, normalization=norm, spec=spec,
                  model_state_sha=training.fingerprint({key: data.array_sha256(value.numpy()) for key, value in state.items()}),
                  contract=dict(data_sha=task["data_sha256"], model_sha=frozen.initial_hash,
                                scaler_sha=fit.scaler_fingerprint(norm["mean"], norm["std"])))
    if tamper == "policy":
        bundle["normalization"]["revin"]["eps"] = .01
    elif tamper == "source":
        bundle["normalization"]["new_scientific_source_hashes"] = {"wrong_source": "0" * 64}
    bundle_path = tmp_path / "selected_bundle.pt"
    torch.save(bundle, bundle_path)
    route_spec = dict(spec, artifact_cache_relative_path=bundle_path.name,
                      artifact_sha256=hashlib.sha256(bundle_path.read_bytes()).hexdigest())
    windows = FastWindows(task, norm["mean"], norm["std"], "cpu")
    origins = np.arange(100, 100 + batch, dtype=np.int64)
    x, _, _ = windows.get(origins)
    expected = (model(x, frozen, 2).double() * torch.tensor(norm["std"]).double()
                + torch.tensor(norm["mean"]).double()).detach().numpy()
    pool = ctx["modules"]["deployment"].FoundationPool(None, "cpu", factory=lambda *args: frozen)
    physical = np.stack([task["physical"][origin-64:origin] for origin in origins])
    mask = np.ones_like(physical, dtype=bool)
    try:
        if tamper is not None:
            with pytest.raises(ValueError, match="RevIN"):
                deployment.load_route(route_spec, task, tmp_path, ctx=ctx, device="cpu", foundation_pool=pool)
            return
        route = deployment.load_route(route_spec, task, tmp_path, ctx=ctx, device="cpu", foundation_pool=pool)
        actual = route.predict_physical(physical, mask, measure_stages=True)
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
        assert route.last_receipt["F_logical_streams"] == 2 * batch
        assert route.last_receipt["whole_seconds"] >= 0
        if enabled:
            assert "RevIN_norm" in route.last_receipt["stage_seconds"]
            assert "RevIN_inverse" in route.last_receipt["stage_seconds"]
            assert isinstance(route.model, module.B2RevIN)
        else:
            assert "RevIN_norm" not in route.last_receipt["stage_seconds"]
            assert not isinstance(route.model, module.B2RevIN)
        route.close()
    finally:
        pool.close()
