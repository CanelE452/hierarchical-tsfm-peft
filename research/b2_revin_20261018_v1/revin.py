"""Original-channel outer RevIN with explicit statistics for each call."""
from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class RevINContext:
    mean: torch.Tensor
    stdev: torch.Tensor


class OuterRevIN(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.eps = float(eps)
        self.affine_weight = nn.Parameter(torch.ones(channels))
        self.affine_bias = nn.Parameter(torch.zeros(channels))

    def normalize(self, x):
        context = RevINContext(x.mean(dim=1, keepdim=True).detach(),
                              torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach())
        normalized = (x - context.mean) / context.stdev
        return normalized * self.affine_weight + self.affine_bias, context

    def inverse(self, prediction, context):
        prediction = (prediction - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
        return prediction * context.stdev + context.mean


class B2RevIN(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.revin = OuterRevIN(base.channels).to(device=base.E.device, dtype=base.E.dtype)

    @property
    def E(self):
        return self.base.E

    @property
    def D(self):
        return self.base.D

    @property
    def front(self):
        return self.base.front

    @property
    def center(self):
        return self.base.center

    @property
    def pca_center(self):
        return self.base.pca_center

    @property
    def length(self):
        return self.base.length

    @property
    def horizon(self):
        return self.base.horizon

    @property
    def channels(self):
        return self.base.channels

    @property
    def latent_k(self):
        return self.base.latent_k

    @property
    def kmax(self):
        return self.base.kmax

    @property
    def k_grid(self):
        return self.base.k_grid

    def forward(self, x, foundation_callable, latent_k=None):
        window, context = self.revin.normalize(x)
        prediction = self.base(window, foundation_callable, latent_k)
        return self.revin.inverse(prediction, context)


def wrap_model(model, enabled):
    if type(enabled) is not bool:
        raise TypeError("RevIN enabled must be bool")
    return B2RevIN(model) if enabled else model


def revin_policy(enabled):
    return dict(enabled=enabled, eps=1e-5, affine=True, variance_unbiased=False,
                statistics="per_window_original_channel_detached_mean_std",
                affine_inverse_denominator="gamma_plus_eps_squared",
                location="after_fixed_fill_and_global_zscore_before_P",
                inverse_scope="entire_anchored_forecast_once_before_global_physical_inverse")


def scientific_sources(ctx):
    repo = Path(ctx["repo"]).resolve()
    here = Path(__file__).resolve().parent
    actual = {path.relative_to(repo).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for name in ("revin.py", "deployment.py") if (path := here / name).is_file()}
    for relative, expected in ctx.get("scientific_source_hashes", {}).items():
        path = (repo / relative).resolve()
        if not path.is_relative_to(repo) or not path.is_file():
            raise ValueError("Registered scientific source is outside the repository or missing")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError("Registered scientific source bytes changed: " + relative)
        actual[relative] = digest
    return actual


def _registered_splits(task, fold="primary"):
    if fold != "primary" or "split_ranges" not in task:
        raise ValueError("Registered primary split boundaries required")
    names = ("A_fit", "A_stop", "M_cal", "V_select", "T_final")
    bounds = {key: tuple(map(int, value)) for key, value in task["split_ranges"].items()}
    if set(bounds) != set(names) or bounds["A_fit"][0] != 0:
        raise ValueError("Incomplete registered split boundaries")
    for index, name in enumerate(names):
        lo, hi = bounds[name]
        if lo >= hi or (index and lo != bounds[names[index-1]][1]):
            raise ValueError("Invalid chronological split boundaries")
    return bounds


@contextmanager
def model_factory(ctx):
    from research.adapter_diagnostics_20261017_v2 import existing_jobs as existing
    modules = ctx["modules"] if "modules" in ctx else existing._modules(ctx)
    fit, training, data = modules["fit"], modules["training"], modules["data"]
    original_create, original_sources, original_splits = fit.create_model, fit._source_contract, data.split_ranges
    source_files = scientific_sources(ctx)

    def check_sources():
        if scientific_sources(ctx) != source_files:
            raise ValueError("Scientific source bytes changed inside RevIN factory scope")

    def source_contract():
        check_sources()
        _, base_files = original_sources()
        combined = dict(base_files, **source_files)
        return training.fingerprint(combined), combined

    def create(spec, task, fit_end, foundation):
        check_sources()
        if spec["family"] not in ("B2", "B0"):
            raise ValueError("Unregistered RevIN family: " + str(spec["family"]))
        enabled = spec["revin"]
        if type(enabled) is not bool:
            raise TypeError("RevIN spec['revin'] must be bool")
        base, normalization = original_create(spec, task, fit_end, foundation)
        normalization["revin"] = revin_policy(enabled)
        normalization["new_scientific_source_hashes"] = source_files.copy()
        if "initialization_events" in ctx:
            ctx["initialization_events"].append(dict(
                family=spec["family"], seed=int(spec["seed"]), revin=enabled, fit_end=int(fit_end),
                P_initial_sha256=training.fingerprint({key: data.array_sha256(value.detach().cpu().numpy())
                                                       for key, value in base.front.state_dict().items()}),
                E_initial_sha256=data.array_sha256(base.E.detach().cpu().numpy()),
                D_initial_sha256=data.array_sha256(base.D.detach().cpu().numpy()),
                center_sha256=data.array_sha256(normalization["pca_center"]),
                scaler_sha256=fit.scaler_fingerprint(normalization["mean"], normalization["std"])))
        return wrap_model(base, enabled), normalization

    fit.create_model, fit._source_contract, data.split_ranges = create, source_contract, _registered_splits
    try:
        yield create
    finally:
        fit.create_model, fit._source_contract, data.split_ranges = original_create, original_sources, original_splits
