"""Source-only tests: tiny doubles exercise routing without pretrained weights."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from research.adapter_diagnostics_20261017_v2.backbones import (
    Chronos2NativeRaw,
    Moirai2Independent,
    assert_optimizer_excludes_backbone,
    backbone_state_hash,
    routing_sentinel,
)


class TensorForecast(nn.Module):
    def __init__(self, length=40, horizon=12):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.25))
        self.module = SimpleNamespace(
            quantile_levels=[0.1, 0.5, 0.9], patch_size=16, num_predict_token=4
        )
        self.hparams = SimpleNamespace(context_length=length, prediction_length=horizon,
                                       target_dim=1, feat_dynamic_real_dim=0,
                                       past_feat_dynamic_real_dim=0)
        self.last_input = None

    def forward(self, past_target, past_observed_target, past_is_pad):
        assert past_target.shape[-1] == 1
        assert not past_is_pad.any()
        self.last_input = past_target.detach().clone()
        value = past_target.mean(1).squeeze(-1) * self.weight
        base = value[:, None, None].expand(-1, 3, self.hparams.prediction_length)
        return base + torch.tensor([-2.0, 0.0, 5.0])[None, :, None]


class NativeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.chronos_config = SimpleNamespace(quantiles=[0.1, 0.2, 0.5, 0.8],
            output_patch_size=16, max_output_patches=64, context_length=8192)
        self.last_groups = None

    def forward(self, context, context_mask, group_ids, num_output_patches):
        self.last_groups = group_ids.clone()
        row = torch.nanmean(context, -1)
        joint = torch.stack([row[group_ids == g].mean() for g in group_ids])
        value = (row + joint) * self.weight
        base = value[:, None, None].expand(-1, 4, 16 * num_output_patches)
        return SimpleNamespace(quantile_preds=base + torch.arange(4)[None, :, None])


def test_moirai_shape_order_crop_median_and_independence():
    forecast = TensorForecast()
    backbone = Moirai2Independent(forecast, length=40, horizon=12)
    x = torch.arange(2 * 40 * 3).reshape(2, 40, 3).float()
    y = backbone(x)
    assert y.shape == (2, 12, 3)
    torch.testing.assert_close(y[:, 0], x.mean(1) * 1.25)
    torch.testing.assert_close(forecast.last_input[:, :, 0], x.transpose(1, 2).reshape(6, 40))
    assert backbone.streams == 6
    assert routing_sentinel(backbone, x, atol=0, rtol=0)["passed"]


def test_moirai_frozen_eval_input_and_encoder_autograd_save_load(tmp_path):
    backbone = Moirai2Independent(TensorForecast(), length=40, horizon=12)
    backbone.train()
    assert not backbone.forecast.training
    assert not any(p.requires_grad for p in backbone.parameters())
    encoder = nn.Parameter(torch.tensor([[0.7, 0.2], [0.3, 0.8]]))
    optimizer = torch.optim.AdamW([encoder], lr=0.01)
    assert_optimizer_excludes_backbone(optimizer, backbone)
    before = backbone_state_hash(backbone)
    x = torch.randn(2, 40, 2)
    z = x @ encoder
    z.retain_grad()
    backbone(z).square().mean().backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
    assert torch.isfinite(encoder.grad).all() and encoder.grad.abs().sum() > 0
    optimizer.step()
    assert before == backbone_state_hash(backbone)
    path = tmp_path / "source_double.pt"
    torch.save(backbone.state_dict(), path)
    restored = Moirai2Independent(TensorForecast(), length=40, horizon=12)
    restored.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(restored(z.detach()), backbone(z.detach()), atol=0, rtol=0)
    with pytest.raises(ValueError, match="optimizer"):
        assert_optimizer_excludes_backbone(torch.optim.SGD(backbone.parameters(), lr=0.1), backbone)


def test_chronos_origin_groups_mask_and_config_median():
    model = NativeModel()
    backbone = Chronos2NativeRaw(model, length=40, horizon=12)
    x = torch.randn(3, 40, 4)
    y = backbone(x)
    assert y.shape == (3, 12, 4)
    assert model.last_groups.tolist() == [0] * 4 + [1] * 4 + [2] * 4
    means = x.mean(1)
    torch.testing.assert_close(y[:, 0], (means + means.mean(-1, keepdim=True)) * 2 + 2)
    assert backbone.median_index == 2
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[:, 1, 0] = False
    other = x.clone()
    other[:, 1, 0] = 1e10
    torch.testing.assert_close(backbone(x, mask), backbone(other, mask), atol=0, rtol=0)
    assert routing_sentinel(backbone, x, atol=1e-6, rtol=1e-6)["passed"]


def test_chronos_native_raw_has_no_adapter_autograd_and_roundtrip(tmp_path):
    backbone = Chronos2NativeRaw(NativeModel(), length=40, horizon=48)
    x = torch.randn(2, 40, 3, requires_grad=True)
    assert backbone(x).shape == (2, 48, 3)
    assert not backbone(x).requires_grad
    path = tmp_path / "native_double.pt"
    torch.save(backbone.state_dict(), path)
    restored = Chronos2NativeRaw(NativeModel(), length=40, horizon=48)
    restored.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(restored(x), backbone(x), atol=0, rtol=0)


def test_validation_rejects_bad_horizon_masks_and_missing_median():
    with pytest.raises(ValueError, match="64"):
        Moirai2Independent(TensorForecast(horizon=65), length=40, horizon=65)
    backbone = Moirai2Independent(TensorForecast(), length=40, horizon=12)
    with pytest.raises(ValueError, match="shape"):
        backbone(torch.ones(1, 39, 2))
    with pytest.raises(ValueError, match="bool"):
        backbone(torch.ones(1, 40, 2), torch.ones(1, 40, 2))
    broken = NativeModel()
    broken.chronos_config.quantiles = [0.1, 0.2, 0.9]
    with pytest.raises(ValueError, match="0.5"):
        Chronos2NativeRaw(broken, length=40, horizon=12)


def test_download_reserves_before_transfer_and_rejects_bad_weights(tmp_path, monkeypatch):
    import hashlib
    import io
    import time
    from research.adapter_diagnostics_20261017_v2 import backbone_jobs
    events = []
    resources = SimpleNamespace(record_download=lambda *args: events.append("reserve"))
    monkeypatch.setattr(backbone_jobs, "_resources", lambda ctx: resources)
    def response(*args, **kwargs):
        events.append("transfer")
        return io.BytesIO(b"model")
    monkeypatch.setattr(backbone_jobs.urllib.request, "urlopen", response)
    ctx = dict(cache=tmp_path, deadline=time.time() + 60)
    target = tmp_path / "model.safetensors"
    with pytest.raises(ValueError, match="SHA256"):
        backbone_jobs._download(ctx, "https://example.invalid/model", target, size=5,
                                expected_sha="0" * 64)
    assert events == ["reserve", "transfer"]
    assert not target.exists()
    assert not list(tmp_path.glob("*.tmp"))
    backbone_jobs._download(ctx, "https://example.invalid/model", target, size=5,
                            expected_sha=hashlib.sha256(b"model").hexdigest())
    assert target.read_bytes() == b"model"


def test_routing_sentinel_rejects_origin_leakage():
    class LeakingBackbone(nn.Module):
        routing = "native_multivariate_per_origin"
        def forward(self, x):
            return x.mean().expand(x.shape[0], 12, x.shape[-1])
    with pytest.raises(AssertionError):
        routing_sentinel(LeakingBackbone(), torch.randn(2, 40, 3), atol=1e-6, rtol=1e-6)


def test_prefix_training_preserves_full_task_splits_and_restores_binding():
    from research.adapter_diagnostics_20261017_v2.backbone_jobs import sealed_split_ranges
    original = lambda task, fold="primary": {"A_fit": [0, len(task["physical"]) // 2]}
    module = SimpleNamespace(split_ranges=original)
    task = dict(physical=[None] * 80, data_sha256="inherited_original_100_rows",
                split_ranges={"A_fit": [0, 50], "A_stop": [50, 60], "V_select": [70, 80]})
    assert module.split_ranges(task)["A_fit"][1] == 40
    with sealed_split_ranges(module, task):
        assert module.split_ranges(task)["A_fit"][1] == 50
        with pytest.raises(ValueError, match="unregistered"):
            module.split_ranges(dict(task, data_sha256="different"))
    assert module.split_ranges is original


def test_frozen_state_assertion_detects_mutated_parameter():
    backbone = Moirai2Independent(TensorForecast(), length=40, horizon=12)
    backbone.assert_frozen_unchanged()
    with torch.no_grad():
        backbone.forecast.weight.add_(1)
    with pytest.raises(ValueError, match="state changed"):
        backbone.assert_frozen_unchanged()


def test_moirai_parity_fixture_preserves_raw_nan_mask_with_mutating_imputer():
    import numpy as np
    from research.adapter_diagnostics_20261017_v2.backbone_jobs import _moirai_mask_fixture
    raw = np.arange(512, dtype=np.float32)[:, None]
    raw[1:4] = np.nan
    def in_place_imputer(values):
        values[np.isnan(values)] = 0
        return values
    filled, observed, official_input = _moirai_mask_fixture(raw, in_place_imputer)
    assert int(np.isnan(raw).sum()) == 3
    assert int(np.isnan(official_input).sum()) == 3
    assert int((~observed).sum()) == 3
    assert np.isfinite(filled).all()
    assert not np.shares_memory(filled, official_input)
    np.testing.assert_array_equal(observed, ~np.isnan(official_input))


def test_chronos_official_pipeline_inputs_are_detached_cpu_before_batching():
    from research.adapter_diagnostics_20261017_v2.backbone_jobs import _chronos_official_prediction
    events = []
    values = torch.arange(2 * 40 * 3).reshape(2, 40, 3).float()
    class DeviceInput:
        device = torch.device("cpu")
        def detach(self):
            events.append("detach")
            return self
        def cpu(self):
            events.append("cpu")
            return values
    class OfficialPipeline:
        def predict(self, inputs, **kwargs):
            events.append("predict")
            assert all(v.device.type == "cpu" and v.shape == (3, 40) for v in inputs)
            assert kwargs["cross_learning"] is False
            return [torch.arange(4)[None, :, None].expand(3, 4, 12).float() for _ in inputs]
    result = _chronos_official_prediction(OfficialPipeline(), DeviceInput(),
                                         length=40, horizon=12, median_index=2)
    assert events == ["detach", "cpu", "predict"]
    assert result.shape == (2, 12, 3)
    assert (result == 2).all()


@pytest.mark.parametrize("exposures", [0, 9])
def test_sampler_order_replay_supports_cp0_and_verifies_saved_state(exposures):
    import numpy as np
    from research.informs_broad_validation_20261015_v1.training import PermutationStream
    from research.adapter_diagnostics_20261017_v2.backbone_jobs import _verified_sampler_order
    origins = np.arange(8, dtype=np.int64)
    sampler = PermutationStream(origins, 92601)
    expected = sampler.draw(exposures) if exposures else np.empty(0, dtype=np.int64)
    state = sampler.state_dict()
    actual = _verified_sampler_order(PermutationStream, origins, 92601, state)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.int64
    with pytest.raises(ValueError, match="sampler"):
        _verified_sampler_order(PermutationStream, origins, 92601, dict(state, position=99))


def test_ridge_cost_metadata_matches_actual_fp64_solver_route(tmp_path, monkeypatch):
    import importlib
    import sys
    import numpy as np
    from research.informs_broad_validation_20261015_v1 import data_contract
    from research.adapter_diagnostics_20261017_v2 import backbone_costs
    monkeypatch.setitem(sys.modules, "data_contract", data_contract)
    ridge = importlib.import_module("research.informs_broad_validation_20261015_v1.ridge_baseline")
    original_import = backbone_costs.importlib.import_module
    monkeypatch.setattr(backbone_costs.importlib, "import_module",
                        lambda name: ridge if name == "ridge_baseline" else original_import(name))
    artifact = tmp_path / "ridge/primary_jena_E1_RAW_RIDGE_k0_s92601/artifact.npz"
    artifact.parent.mkdir(parents=True)
    mean, std = np.zeros(1, np.float32), np.ones(1, np.float32)
    np.savez(artifact, mean=mean, std=std, coefficients=np.zeros((1, 33, 48), np.float64))
    modules = dict(old_cache=tmp_path, data=SimpleNamespace(fit_statistics=lambda *args: (mean, std, None)))
    route = backbone_costs._load({}, modules, {}, "Ridge")
    history = np.ones((1, 512, 1), np.float32)
    assert ridge.predict_normalized(ridge.load_artifact(artifact), history).dtype == np.float64
    assert route["core_dtype"] == "float64"
    assert route["dtype"].startswith("FP64 Ridge core")
    np.testing.assert_array_equal(route["predict"](history, np.ones_like(history, dtype=bool)), np.ones((1, 48, 1)))
