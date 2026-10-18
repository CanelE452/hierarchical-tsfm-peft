"""Native GT forecasts and a zero-initialized TS2 residual bridge."""

from __future__ import annotations

import hashlib
from numbers import Integral

import torch
from torch import nn
from chronos.chronos_bolt import InstanceNorm

from research.professor_compressor_direct_20261011.models_gt import GTConvAE


class ZeroSafeSqrt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, variance):
        scale = variance.sqrt()
        ctx.save_for_backward(variance, scale)
        return scale

    @staticmethod
    def backward(ctx, upstream):
        variance, scale = ctx.saved_tensors
        positive = variance > 0
        denominator = torch.where(positive, scale, torch.ones_like(scale))
        return torch.where(positive, upstream / (2 * denominator), torch.zeros_like(upstream))


class SafeInstanceNorm(InstanceNorm):
    """The existing temporal-campaign normalization, with identical forward values."""

    def forward(self, x, loc_scale=None):
        dtype = x.dtype
        x = x.float()
        if loc_scale is None:
            loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.)
            variance = (x - loc).square().nanmean(dim=-1, keepdim=True)
            scale = torch.nan_to_num(ZeroSafeSqrt.apply(variance), nan=1.)
            scale = torch.where(scale == 0, self.eps, scale)
        else:
            loc, scale = loc_scale
        y = (x - loc) / scale
        if self.use_arcsinh:
            y = torch.arcsinh(y)
        return y.to(dtype), (loc, scale)


def _state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class FrozenFoundation:
    """Load Bolt without legacy provenance writes and retain gradients to its inputs."""

    def __init__(self, model_directory, device="cuda", row_microbatch=128):
        from chronos import ChronosBoltPipeline

        if not isinstance(row_microbatch, Integral) or isinstance(row_microbatch, bool) or row_microbatch < 1:
            raise ValueError("row_microbatch must be a positive integer")
        self.pipeline = ChronosBoltPipeline.from_pretrained(
            str(model_directory), device_map=device, torch_dtype=torch.float32,
            local_files_only=True,
        )
        self.model = self.pipeline.model.eval().requires_grad_(False)
        original = self.model.instance_norm
        self.model.instance_norm = SafeInstanceNorm(eps=original.eps, use_arcsinh=original.use_arcsinh)
        self.quantiles = list(self.model.chronos_config.quantiles)
        self.median = self.quantiles.index(.5)
        self.native_horizon = int(self.model.chronos_config.prediction_length)
        self.max_context = int(self.model.chronos_config.context_length)
        self.row_microbatch = int(row_microbatch)
        self.initial_hash = _state_hash(self.model)
        self.calls = 0
        self.streams = 0
        self.requested_context_length = None
        self.actual_context_length = None
        self.context_records = []

    def forecast(self, latent, horizon):
        if (latent.ndim < 3 or any(size < 1 for size in latent.shape)
                or not latent.is_floating_point() or not torch.isfinite(latent).all()):
            raise ValueError("Expected a finite floating [B,time,features...] latent")
        if (not isinstance(horizon, Integral) or isinstance(horizon, bool)
                or not 1 <= horizon <= self.native_horizon):
            raise ValueError("horizon must be a positive integer within Bolt's native horizon")
        if (not isinstance(self.row_microbatch, Integral) or isinstance(self.row_microbatch, bool)
                or self.row_microbatch < 1):
            raise ValueError("row_microbatch must be a positive integer")
        batch, length = latent.shape[:2]
        feature_shape = latent.shape[2:]
        rows = latent.reshape(batch, length, -1).transpose(1, 2).reshape(-1, length)
        self.requested_context_length = length
        self.actual_context_length = min(length, self.max_context)
        rows = rows[:, -self.actual_context_length:]
        predictions = []
        row_sizes = []
        self.model.eval()
        for start in range(0, len(rows), self.row_microbatch):
            chunk = rows[start:start + self.row_microbatch]
            prediction = self.model(context=chunk).quantile_preds
            predictions.append(prediction[:, self.median, :horizon])
            self.calls += 1
            self.streams += len(chunk)
            row_sizes.append(len(chunk))
        self.context_records.append({
            "requested_context_length": length,
            "actual_context_length": self.actual_context_length,
            "streams": len(rows), "row_sizes": row_sizes,
        })
        combined = torch.cat(predictions)
        if not torch.isfinite(combined).all():
            raise ValueError("Bolt produced a non-finite forecast")
        return combined.reshape(batch, -1, horizon).transpose(1, 2).reshape(batch, horizon, *feature_shape)

    def verify_unchanged(self):
        return {
            "unchanged": _state_hash(self.model) == self.initial_hash,
            "parameter_gradients_absent": all(parameter.grad is None for parameter in self.model.parameters()),
        }


def create_gt(graph, seed, device="cpu"):
    """Initialize a full spatial GT without consuming the caller's CPU RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        model = GTConvAE(torch.as_tensor(graph, dtype=torch.float32, device="cpu").detach())
    return model.to(device)


class ForecastRoute(nn.Module):
    """GT E/F/native D, optionally augmented by fixed pooled TS2 features."""

    def __init__(self, gt, foundation, feature_mean=None, feature_std=None):
        super().__init__()
        if gt.channels != 21 or gt.no_spatial or gt.latent_features != 2 or gt.time_reduction != 4:
            raise ValueError("This route requires the full spatial Jena GT with 21 channels")
        if (feature_mean is None) != (feature_std is None):
            raise ValueError("Both fixed TRAIN feature_mean and feature_std are required")
        self.gt = gt.requires_grad_(True)
        self.gt.product.graph = self.gt.product.graph.detach()
        # F is shared and frozen; it must not enter adapter parameters/checkpoints.
        object.__setattr__(self, "foundation", foundation)
        foundation_model = getattr(foundation, "model", foundation)
        if isinstance(foundation_model, nn.Module):
            foundation_model.eval().requires_grad_(False)
        self.W = None
        mean = std = None
        if feature_mean is not None:
            reference = gt.product.graph
            mean = torch.as_tensor(feature_mean, dtype=reference.dtype, device=reference.device).detach().clone()
            std = torch.as_tensor(feature_std, dtype=reference.dtype, device=reference.device).detach().clone()
            if (mean.shape != (320,) or std.shape != (320,) or not torch.isfinite(mean).all()
                    or not torch.isfinite(std).all() or (std < 0).any()):
                raise ValueError("Fixed TRAIN feature statistics must be finite 320-vectors with nonnegative std")
            std = torch.where(std == 0, torch.ones_like(std), std)
            with torch.random.fork_rng(devices=[]):
                self.W = nn.Linear(320, 42, bias=False)
            self.W = self.W.to(reference)
            nn.init.zeros_(self.W.weight)
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_std", std)

    @staticmethod
    def pool_features(representations):
        if (representations.ndim != 3 or representations.shape[-1] != 320
                or any(size < 1 for size in representations.shape) or representations.shape[1] % 4
                or not representations.is_floating_point() or not torch.isfinite(representations).all()):
            raise ValueError("TS2 representations must be finite [B,L,320] with complete four-tick blocks")
        batch, length, features = representations.shape
        return representations.reshape(batch, length // 4, 4, features).mean(dim=2)

    def forecast(self, x, features=None):
        if (x.ndim != 3 or x.shape[-1] != 21 or any(size < 1 for size in x.shape)
                or x.shape[1] % 4 or not x.is_floating_point() or not torch.isfinite(x).all()):
            raise ValueError("History must be a finite normalized [B,L,21] with complete four-tick blocks")
        latent = self.gt.encode(x)
        if features is not None:
            if (self.W is None or features.shape != (x.shape[0], x.shape[1] // 4, 320)
                    or not features.is_floating_point() or not torch.isfinite(features).all()):
                raise ValueError("Fusion requires fixed TRAIN statistics and finite pooled [B,L/4,320] features")
            standardized = (features.detach().to(latent) - self.feature_mean) / self.feature_std
            latent = latent + self.W(standardized).reshape_as(latent)
        future = self.foundation.forecast(latent, horizon=12)
        if future.shape != (x.shape[0], 12, 21, 2) or not torch.isfinite(future).all():
            raise ValueError("Foundation must return a finite [B,12,21,2] forecast")
        forecast = self.gt.decode_future(latent, future, horizon=48)
        if not torch.isfinite(forecast).all():
            raise ValueError("GT decoder produced a non-finite forecast")
        return forecast

    def forward(self, x, features=None):
        return self.forecast(x, features)

    def adapter_state(self):
        return {name: value.detach().cpu().clone() for name, value in self.state_dict().items()}

    def load_adapter_state(self, state):
        graph = state.get("gt.product.graph")
        if graph is None or not torch.equal(graph.to(self.gt.product.graph), self.gt.product.graph):
            raise ValueError("Adapter checkpoint must preserve the fixed spatial graph")
        return self.load_state_dict(state, strict=True)
