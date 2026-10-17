"""Pinned pretrained tensor paths; importing this module never downloads a model.

Moirai consumes finite, already causally imputed histories. Its observed mask
is passed to the official Tensor forward without replacing the caller's fill.
Chronos2 missing entries become NaN before instance normalization, matching
the official inference path. No future covariates enter either comparison.
"""

from __future__ import annotations

import hashlib
import math

import torch
from torch import nn


MOIRAI_REPOSITORY = "Salesforce/moirai-2.0-R-small"
MOIRAI_REVISION = "30f43ff08c8494f4943ae1521e9d4e94a0fbb389"
CHRONOS2_REPOSITORY = "autogluon/chronos-2-small"
CHRONOS2_REVISION = "ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a"


def _median_index(quantiles):
    indices = [i for i, value in enumerate(quantiles) if math.isclose(float(value), 0.5, abs_tol=1e-8)]
    if len(indices) != 1:
        raise ValueError("model quantiles must contain exactly one 0.5 median")
    return indices[0]


def backbone_state_hash(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def assert_optimizer_excludes_backbone(optimizer, backbone):
    frozen = {id(p) for p in backbone.parameters()}
    if any(id(p) in frozen for group in optimizer.param_groups for p in group["params"]):
        raise ValueError("optimizer contains frozen backbone parameters")


def _validate_input(x, observed_mask, length):
    if x.ndim != 3 or x.shape[1] != length or x.shape[0] < 1 or x.shape[2] < 1:
        raise ValueError(f"input shape must be [B,{length},C] with positive B,C")
    if not x.is_floating_point():
        raise ValueError("history must be floating point")
    if observed_mask is None:
        observed_mask = torch.isfinite(x)
    if observed_mask.shape != x.shape or observed_mask.dtype != torch.bool:
        raise ValueError("observed mask must have input shape and bool dtype")
    if observed_mask.device != x.device:
        raise ValueError("observed mask must share input device")
    if not torch.isfinite(x[observed_mask]).all():
        raise ValueError("observed history contains nonfinite values")
    return observed_mask


class _FrozenBackbone(nn.Module):
    def _freeze(self):
        self.requires_grad_(False)
        self.eval()
        self.streams = self.row_calls = self.physical_streams = self.physical_calls = 0
        self.initial_hash = backbone_state_hash(self)

    def train(self, mode=True):
        # Adapter training must never enable dropout in the pretrained foundation.
        return super().train(False)

    def _count(self, rows):
        self.streams += rows
        self.physical_streams += rows
        self.row_calls += 1
        self.physical_calls += 1

    def assert_frozen_unchanged(self):
        if backbone_state_hash(self) != self.initial_hash:
            raise ValueError("frozen backbone state changed")
        if self.training or any(p.requires_grad for p in self.parameters()):
            raise ValueError("frozen backbone train/eval or requires_grad contract changed")


class Moirai2Independent(_FrozenBackbone):
    """[B,L,K] -> independent [B*K,L,1] -> official quantiles -> [B,H,K]."""

    routing = "independent_univariate"
    point_statistic = "quantile_0.5"

    def __init__(self, forecast, *, length=512, horizon=48):
        super().__init__()
        self.forecast = forecast
        self.length, self.horizon = int(length), int(horizon)
        self.quantiles = tuple(forecast.module.quantile_levels)
        self.median_index = _median_index(self.quantiles)
        native_horizon = forecast.module.patch_size * forecast.module.num_predict_token
        if not 1 <= self.horizon <= native_horizon:
            raise ValueError(f"single native forecast requires 1<=H<={native_horizon}")
        hp = forecast.hparams
        if (hp.context_length != self.length or hp.prediction_length != self.horizon
                or hp.target_dim != 1 or hp.feat_dynamic_real_dim != 0
                or hp.past_feat_dynamic_real_dim != 0):
            raise ValueError("official forecast configuration differs from independent tensor contract")
        self._freeze()

    def forward(self, x, observed_mask=None):
        mask = _validate_input(x, observed_mask, self.length)
        if not torch.isfinite(x).all():
            raise ValueError("Moirai requires finite causally imputed history, including missing entries")
        batch, _, channels = x.shape
        past = x.transpose(1, 2).reshape(batch * channels, self.length, 1)
        observed = mask.transpose(1, 2).reshape_as(past)
        is_pad = torch.zeros((batch * channels, self.length), device=x.device, dtype=torch.bool)
        # Public Tensor forward retains d(loss)/d(input); forecast.predict uses NumPy/no_grad.
        quantiles = self.forecast(past_target=past, past_observed_target=observed,
                                  past_is_pad=is_pad)
        expected = (batch * channels, len(self.quantiles), self.horizon)
        if tuple(quantiles.shape) != expected:
            raise ValueError(f"official Moirai quantile shape {tuple(quantiles.shape)} != {expected}")
        self._count(batch * channels)
        point = quantiles[:, self.median_index, :self.horizon]
        return point.reshape(batch, channels, self.horizon).transpose(1, 2).contiguous()


class Chronos2NativeRaw(_FrozenBackbone):
    """Native multivariate Raw with one isolated attention group per origin."""

    routing = "native_multivariate_per_origin"
    point_statistic = "quantile_0.5"

    def __init__(self, model, *, length=512, horizon=48):
        super().__init__()
        self.model = model
        self.length, self.horizon = int(length), int(horizon)
        config = model.chronos_config
        self.quantiles = tuple(config.quantiles)
        self.median_index = _median_index(self.quantiles)
        self.output_patch_size = int(config.output_patch_size)
        self.output_patches = math.ceil(self.horizon / self.output_patch_size)
        if not 1 <= self.output_patches <= config.max_output_patches:
            raise ValueError("horizon exceeds native Chronos2 output patches")
        if self.length > config.context_length:
            raise ValueError("input length exceeds configured context; silent truncation is forbidden")
        self._freeze()

    @torch.no_grad()
    def forward(self, x, observed_mask=None):
        mask = _validate_input(x, observed_mask, self.length)
        batch, _, channels = x.shape
        context = x.transpose(1, 2).reshape(batch * channels, self.length)
        flat_mask = mask.transpose(1, 2).reshape_as(context)
        context = context.masked_fill(~flat_mask, float("nan"))
        group_ids = torch.arange(batch, device=x.device).repeat_interleave(channels)
        quantiles = self.model(context=context, context_mask=flat_mask, group_ids=group_ids,
                               num_output_patches=self.output_patches).quantile_preds
        expected = (batch * channels, len(self.quantiles), self.output_patches * self.output_patch_size)
        if tuple(quantiles.shape) != expected:
            raise ValueError(f"official Chronos2 quantile shape {tuple(quantiles.shape)} != {expected}")
        self._count(batch * channels)
        point = quantiles[:, self.median_index, :self.horizon]
        return point.reshape(batch, channels, self.horizon).transpose(1, 2).contiguous()


def load_backbone(name, *, device="cpu", length=512, horizon=48,
                  cache_dir=None, local_files_only=True, model_path=None):
    """Explicit loader; defaults to local cache and never runs during import."""
    if name == "moirai2":
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

        module = Moirai2Module.from_pretrained(model_path or MOIRAI_REPOSITORY, revision=MOIRAI_REVISION,
            cache_dir=cache_dir, local_files_only=local_files_only)
        forecast = Moirai2Forecast(module=module, prediction_length=horizon, context_length=length,
            target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
        return Moirai2Independent(forecast.to(device), length=length, horizon=horizon)
    if name == "chronos2":
        from chronos import BaseChronosPipeline

        pipeline = BaseChronosPipeline.from_pretrained(model_path or CHRONOS2_REPOSITORY, revision=CHRONOS2_REVISION,
            device_map=device, cache_dir=cache_dir, local_files_only=local_files_only,
            torch_dtype=torch.float32)
        return Chronos2NativeRaw(pipeline.model, length=length, horizon=horizon)
    raise ValueError(f"unknown backbone {name!r}")


@torch.no_grad()
def routing_sentinel(backbone, x, *, atol, rtol):
    """Perturb a different origin and test split/merge; never inspects targets."""
    if x.shape[0] < 2:
        raise ValueError("routing sentinel requires at least two origins")
    merged = backbone(x)
    split = torch.cat([backbone(part) for part in x.split(1)], dim=0)
    torch.testing.assert_close(merged, split, atol=atol, rtol=rtol)
    changed = x.clone()
    changed[1] = changed[1] * -3 + 123.0
    perturbed = backbone(changed)
    torch.testing.assert_close(merged[0], perturbed[0], atol=atol, rtol=rtol)
    isolated_channel = None
    if backbone.routing == "independent_univariate" and x.shape[-1] > 1:
        changed = x.clone()
        changed[0, :, 1] = changed[0, :, 1] * 7 - 91
        torch.testing.assert_close(merged[0, :, 0], backbone(changed)[0, :, 0], atol=atol, rtol=rtol)
        isolated_channel = True
    return {"passed": True, "origin_isolation": True, "split_merge": True,
            "channel_isolation": isolated_channel, "atol": atol, "rtol": rtol,
            "max_split_merge_abs_error": float((merged - split).abs().max().cpu())}
