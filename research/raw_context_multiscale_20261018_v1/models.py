"""Fixed raw/context splits with one shared, gradient-enabled foundation.

Conversion preserves the existing ForecastAdapter's parameter objects and state
keys. The foundation is supplied per forward and is never copied or registered
as a child of this adapter. All decoder lags remain on the original time grid.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path

import torch

from research.informs_broad_validation_20261015_v1.networks import ForecastAdapter, LAGS
from research.b2_revin_20261018_v1 import revin


ATOL = 1e-5
RTOL = 1e-4


def _scale(value):
    if type(value) is not int or value not in (1, 2):
        raise ValueError("Only registered integer scales 1 and 2 are allowed")
    return value


def block_mean(x, scale=2):
    """Average disjoint blocks within each supplied [B,L,K] history."""
    scale = _scale(scale)
    if x.ndim != 3 or x.shape[1] % scale:
        raise ValueError("Block mean requires [B,L,K] with L divisible by scale")
    if scale == 1:
        return x
    batch, length, channels = x.shape
    return x.reshape(batch, length // scale, scale, channels).mean(dim=2)


def repeat_time(x, scale=2):
    """Restore block j at exactly original-grid offsets sj .. sj+s-1."""
    scale = _scale(scale)
    if x.ndim != 3:
        raise ValueError("Time repeat requires [B,L,K]")
    return x if scale == 1 else x.repeat_interleave(scale, dim=1)


def future_block_offsets(horizon, scale=2):
    scale = _scale(scale)
    if type(horizon) is not int or horizon < 1 or horizon % scale:
        raise ValueError("Horizon must be a positive integer divisible by scale")
    return tuple(tuple(range(start, start + scale)) for start in range(0, horizon, scale))


def time_contract(length, horizon, scale=2):
    scale = _scale(scale)
    if type(length) is not int or length < 1 or length % scale:
        raise ValueError("Length must be a positive integer divisible by scale")
    return dict(original_context=length, original_horizon=horizon,
                fine_context=length, fine_horizon=horizon,
                coarse_context=length // scale, coarse_horizon=horizon // scale,
                future_offset_blocks=future_block_offsets(horizon, scale),
                decoder_lags=LAGS, decoder_time_grid="original", anchor_count=1)


class _RawContextMixin:
    def _k(self, value):
        k = super()._k(value)
        if self.raw_context_mode != "S" and k != self.kmax:
            raise ValueError("Raw/context methods require the fixed total K")
        return k

    def latent_branches(self, x, latent_k=None):
        """Return raw and context F inputs; context is coarse only in B."""
        k = self._k(latent_k)
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.length, self.channels):
            raise ValueError(f"Expected [B,{self.length},{self.channels}]")
        if self.raw_context_mode == "S":
            raise ValueError("S has one all-context latent input")
        half = k // 2
        raw = (x - self.center) @ self.E[:, :half]
        context = (self.front(x) - self.center) @ self.E[:, half:k]
        if self.raw_context_mode == "B":
            context = block_mean(context, self.raw_context_scale)
        return raw, context

    def encode(self, x, latent_k=None):
        if self.raw_context_mode == "S":
            return super().encode(x, latent_k)
        raw, context = self.latent_branches(x, latent_k)
        return torch.cat((raw, repeat_time(context, self.raw_context_scale)), dim=-1)

    @staticmethod
    def _future(foundation_callable, z, horizon):
        future = foundation_callable(z)
        if (not isinstance(future, torch.Tensor) or future.ndim != 3
                or future.shape[0] != z.shape[0] or future.shape[2] != z.shape[2]
                or future.shape[1] < horizon):
            raise ValueError("Foundation callable must return [B,at_least_H,K]")
        return future[:, :horizon]

    def forecast_latents(self, x, foundation_callable, latent_k=None):
        """Return original-grid (past, future) before the inherited lag decoder."""
        k = self._k(latent_k)
        if self.raw_context_mode == "S":
            z = self.encode(x, k)
            return z, self._future(foundation_callable, z, self.horizon)
        raw, context = self.latent_branches(x, k)
        z = torch.cat((raw, repeat_time(context, self.raw_context_scale)), dim=-1)
        if self.raw_context_mode == "A" and self.raw_context_schedule == "native":
            return z, self._future(foundation_callable, z, self.horizon)
        raw_future = self._future(foundation_callable, raw, self.horizon)
        context_future = self._future(foundation_callable, context,
                                      self.horizon // self.raw_context_scale)
        future = torch.cat((raw_future, repeat_time(context_future, self.raw_context_scale)), dim=-1)
        return z, future

    def forward(self, x, foundation_callable, latent_k=None):
        k = self._k(latent_k)
        past, future = self.forecast_latents(x, foundation_callable, k)
        return self.decode(x, past, future, k)


class RawContextForecastAdapter(_RawContextMixin, ForecastAdapter):
    """A ForecastAdapter converted without allocating or initializing weights."""


_CONVERSION_CLASSES = {ForecastAdapter: RawContextForecastAdapter}


def convert_model(base, mode="S", scale=2, schedule="native"):
    """Convert an existing B2 in place, preserving state keys, tensors and RNG.

    The old loader imports ``networks`` by its short module name. A matching
    subclass of that exact ForecastAdapter type preserves its identity too.
    """
    scale = _scale(scale)
    if mode not in ("S", "A", "B"):
        raise ValueError("Raw/context mode must be S, A or B")
    if schedule not in ("native", "two_call") or (mode == "S" and schedule != "native"):
        raise ValueError("Schedule must be native, or A/B two_call")
    if (not isinstance(base, (ForecastAdapter, _RawContextMixin))
            and (type(base).__name__ != "ForecastAdapter"
                 or type(base).__module__ != "networks")):
        raise TypeError("Conversion requires an existing ForecastAdapter")
    if base.front.kind != "B2" or base.encoder_kind != "pointwise":
        raise ValueError("Raw/context conversion requires the existing pointwise B2")
    if mode != "S" and (base.kmax % 2 or base.k_grid not in (None, (base.kmax,))):
        raise ValueError("Raw/context methods require even, fixed total K")
    if mode == "B":
        time_contract(base.length, base.horizon, scale)
    if not isinstance(base, _RawContextMixin):
        base_type = type(base)
        if base_type not in _CONVERSION_CLASSES:
            _CONVERSION_CLASSES[base_type] = type(
                "RawContextForecastAdapter", (_RawContextMixin, base_type),
                {"__module__": __name__})
        base.__class__ = _CONVERSION_CLASSES[base_type]
    base.raw_context_mode = mode
    base.raw_context_scale = scale if mode == "B" else 1
    base.raw_context_schedule = schedule
    return base


def scientific_sources(ctx):
    """Bind new model bytes alongside the existing safe normalization route."""
    sources = revin.scientific_sources(ctx)
    path = Path(__file__).resolve()
    relative = path.relative_to(Path(ctx["repo"]).resolve()).as_posix()
    sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return sources


def _mode_options(spec):
    identity = spec.get("scientific_identity", {})
    if isinstance(identity, str):
        if len(identity) == 64 and all(value in "0123456789abcdef" for value in identity):
            return None
        raise ValueError("Legacy scientific_identity must be a SHA256 fingerprint")
    if not isinstance(identity, dict):
        raise ValueError("scientific_identity must be a dictionary or legacy SHA256 fingerprint")
    mode = identity.get("raw_context_mode")
    if mode is None and not any(isinstance(key, str) and key.startswith("raw_context_") for key in identity):
        return None
    if mode not in ("S", "A", "B"):
        raise ValueError("Raw/context identity requires mode S, A or B")
    if spec["family"] != "B2" or spec.get("revin", False) is not False:
        raise ValueError("Raw/context methods require B2 with outer RevIN OFF")
    return dict(mode=mode, scale=identity.get("raw_context_scale", 2),
                schedule=identity.get("raw_context_schedule", "native"))


@contextmanager
def model_factory(ctx):
    """Scope conversion around the existing fit/reload factory; restore all hooks."""
    from research.adapter_diagnostics_20261017_v2 import existing_jobs as existing
    modules = ctx["modules"] if "modules" in ctx else existing._modules(ctx)
    factory_ctx = dict(ctx, modules=modules, scientific_source_hashes=scientific_sources(ctx))
    fit = modules["fit"]
    with revin.model_factory(factory_ctx) as base_create:
        def create(spec, task, fit_end, foundation):
            options = _mode_options(spec)
            internal = dict(spec, revin=spec.get("revin", False))
            base, norm = base_create(internal, task, fit_end, foundation)
            if options is None:
                return base, norm
            model = convert_model(base, **options)
            norm["raw_context"] = dict(
                mode=model.raw_context_mode, scale=model.raw_context_scale,
                schedule=model.raw_context_schedule, total_K=model.kmax,
                raw_columns=[0, model.kmax // 2] if model.raw_context_mode != "S" else None,
                context_columns=[model.kmax // 2, model.kmax] if model.raw_context_mode != "S" else [0, model.kmax],
                PCA_column_order="unchanged", decoder_lags=list(LAGS),
                decoder_time_grid="original", original_channel_anchor_count=1,
                foundation_instances=1,
                foundation_logical_calls=1 if model.raw_context_mode == "S" or (
                    model.raw_context_mode == "A" and model.raw_context_schedule == "native") else 2)
            return model, norm

        fit.create_model = create
        try:
            yield create
        finally:
            fit.create_model = base_create


def create_model(spec, task, fit_end, foundation, *, modules):
    """Standalone fresh seeded construction using the existing PCA/fit contract."""
    ctx = dict(repo=Path(__file__).resolve().parents[2], modules=modules)
    with model_factory(ctx) as create:
        return create(spec, task, fit_end, foundation)
