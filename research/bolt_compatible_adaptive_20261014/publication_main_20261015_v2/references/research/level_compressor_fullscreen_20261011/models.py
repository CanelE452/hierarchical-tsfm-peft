"""Fixed A--H compressor models; the frozen forecaster is deliberately absent."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EPSILON = 1e-4
BURN_IN = 16
GT_ORDER = 2
GT_RHO = 0.01


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_OLD = _load("_fullscreen_previous_encoders", ROOT / "research/level_encoder_contract_20261010/encoders.py")


class SharedTemporalMap(nn.Module):
    """Channel-separate, weight-shared 512 -> 32 -> 48 correction with carry."""

    def __init__(self, seed: int, rank: int = 32):
        super().__init__()
        self.down = nn.Linear(512, rank, bias=False)
        self.up = nn.Linear(rank, 48, bias=False)
        generator = torch.Generator().manual_seed(int(seed) + 401)
        with torch.no_grad():
            self.down.weight.copy_(torch.randn(self.down.weight.shape, generator=generator) / np.sqrt(512))
            self.up.weight.zero_()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        last = z[:, -1:, :]
        centered = z - last
        correction = self.up(self.down(centered.transpose(1, 2))).transpose(1, 2)
        return last + correction


class JointTemporalMap(nn.Module):
    """All K histories may contribute to every future latent coordinate."""

    def __init__(self, latent: int, seed: int, rank: int = 32):
        super().__init__()
        self.latent = int(latent)
        self.down = nn.Linear(512 * latent, rank, bias=False)
        self.up = nn.Linear(rank, 48 * latent, bias=False)
        generator = torch.Generator().manual_seed(int(seed) + 401)
        with torch.no_grad():
            self.down.weight.copy_(torch.randn(self.down.weight.shape, generator=generator) / np.sqrt(512 * latent))
            self.up.weight.zero_()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        last = z[:, -1:, :]
        correction = self.up(self.down((z - last).flatten(1))).reshape(z.shape[0], 48, self.latent)
        return last + correction


class CausalAutoencoder(nn.Module):
    def __init__(self, structure: str, channels: int, latent: int, seed: int, arm: str):
        super().__init__()
        family = "MLP_POINT" if structure.upper() in ("MLP", "POINT_MLP", "MLP_POINT") else "TCN_TIME"
        self.encoder = _OLD.build_encoder(family, channels, latent, seed)
        self.decoder = _OLD.build_decoder(channels, latent, seed, lag=8)
        self.channels = int(channels)
        self.latent = int(latent)
        self.arm = normalize_arm(arm)
        self.encoder_lag = self.encoder.encoder_lag
        if self.arm == "JOINT_1":
            self.aux_predictor = JointTemporalMap(latent, seed)
        elif self.arm == "REC":
            self.aux_predictor = None
        else:
            self.aux_predictor = SharedTemporalMap(seed)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode(x)

    def encode_future(self, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        return self.encode(torch.cat([past, future], dim=1))[:, past.shape[1]:]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder.decode(z)

    def decode_future(self, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        return self.decoder.decode_future(past, future)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class ProductShift(nn.Module):
    """Eq.1 exactly, implemented without materializing the CT x CT matrix."""

    def __init__(self, graph, arm: str):
        super().__init__()
        self.register_buffer("graph", torch.as_tensor(graph, dtype=torch.float32).contiguous())
        self.arm = str(arm).lower()
        cartesian = torch.tensor([0., 1., 1., 0.], dtype=torch.float32)
        if self.arm in ("fixed_cartesian", "cartesian", "fixed"):
            self.register_buffer("coefficients", cartesian)
        elif self.arm in ("no_spatialmessage", "no_spatial", "no_spatial_message"):
            self.coefficients = nn.Parameter(torch.tensor([0., 0., 1., 0.]))
        elif self.arm in ("learned_product", "learned", "gt_learned_product"):
            self.coefficients = nn.Parameter(cartesian)
        else:
            raise ValueError(f"Unknown graph product arm: {arm}")

    def effective_coefficients(self) -> torch.Tensor:
        if self.arm in ("no_spatialmessage", "no_spatial", "no_spatial_message"):
            return self.coefficients * self.coefficients.new_tensor([1., 0., 1., 0.])
        return self.coefficients

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s00, s01, s10, s11 = self.effective_coefficients().unbind()
        temporal = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        spatial = torch.einsum("btcf,dc->btdf", x, self.graph)
        joint = torch.cat([torch.zeros_like(spatial[:, :1]), spatial[:, :-1]], dim=1)
        return s00 * x + s01 * spatial + s10 * temporal + s11 * joint


class PolynomialGTLayer(nn.Module):
    """Eq.2--3 feature bank, order q=2 for the new bounded prototype."""

    def __init__(self, inputs: int, outputs: int, seed: int, order: int = GT_ORDER):
        super().__init__()
        self.order = int(order)
        generator = torch.Generator().manual_seed(int(seed))
        scale = np.sqrt(2. / ((order + 1) * inputs + outputs))
        self.weight = nn.Parameter(torch.randn(order + 1, inputs, outputs, generator=generator) * scale)

    def forward(self, x: torch.Tensor, shift: ProductShift) -> torch.Tensor:
        current = x
        out = torch.einsum("btcg,gf->btcf", current, self.weight[0])
        for degree in range(1, self.order + 1):
            current = shift(current)
            out = out + torch.einsum("btcg,gf->btcf", current, self.weight[degree])
        return out


class GraphTimeAutoencoder(nn.Module):
    family = "GTCONVAE_INSPIRED"
    encoder_lag = 4

    def __init__(self, channels: int, latent: int, seed: int, arm: str, graph):
        super().__init__()
        if graph is None or np.shape(graph) != (channels, channels):
            raise ValueError("Graph-time adapter requires a TRAIN-only C x C spatial graph")
        self.channels = int(channels)
        self.latent = int(latent)
        self.arm = str(arm).lower()
        self.product = ProductShift(graph, arm)
        self.enc1 = PolynomialGTLayer(1, 8, seed + 501)
        self.enc2 = PolynomialGTLayer(8, 4, seed + 502)
        self.enc_projection = nn.Linear(channels * 4, latent, bias=False)
        self.dec_projection = nn.Linear(latent, channels * 4, bias=False)
        self.dec1 = PolynomialGTLayer(4, 8, seed + 503)
        self.dec2 = PolynomialGTLayer(8, 1, seed + 504)
        self.aux_predictor = None
        generator = torch.Generator().manual_seed(int(seed) + 505)
        with torch.no_grad():
            self.enc_projection.weight.copy_(torch.randn(self.enc_projection.weight.shape, generator=generator) / np.sqrt(channels * 4))
            self.dec_projection.weight.copy_(self.enc_projection.weight.T)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = torch.tanh(self.enc1(x.unsqueeze(-1), self.product))
        features = torch.tanh(self.enc2(features, self.product))
        return self.enc_projection(features.flatten(2))

    def encode_future(self, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        return self.encode(torch.cat([past, future], dim=1))[:, past.shape[1]:]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        features = self.dec_projection(z).reshape(z.shape[0], z.shape[1], self.channels, 4)
        features = torch.tanh(self.dec1(features, self.product))
        return self.dec2(features, self.product).squeeze(-1)

    def decode_future(self, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        return self.decode(torch.cat([past[:, -4:], future], dim=1))[:, min(4, past.shape[1]):]

    def product_penalty(self) -> torch.Tensor:
        if not self.product.coefficients.requires_grad:
            return self.product.coefficients.new_zeros(())
        return self.product.effective_coefficients().abs().sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def normalize_arm(arm: str) -> str:
    aliases = {
        "C0": "REC", "C1": "GUARDED_STOPGRAD", "C2": "SELF_0p1",
        "C3": "SELF_1", "C4": "JOINT_1", "MLP_REC": "REC",
        "MLP_GUARDED": "GUARDED_STOPGRAD", "MLP_GUARDED_STOPGRAD": "GUARDED_STOPGRAD",
        "MLP_SELF_1": "SELF_1", "GUARDED": "GUARDED_STOPGRAD",
        "SELF_0P1": "SELF_0p1",
    }
    normalized = aliases.get(str(arm).upper(), str(arm).upper())
    if normalized not in ("REC", "GUARDED_STOPGRAD", "SELF_0p1", "SELF_1", "JOINT_1"):
        raise ValueError(f"Unknown C objective: {arm}")
    return normalized


def build_model(structure: str, C: int, K: int, seed: int, arm: str = "REC", graph=None) -> nn.Module:
    if str(structure).upper() in ("GT", "GTCONVAE", "GTCONVAE_INSPIRED", "GRAPH_TIME"):
        return GraphTimeAutoencoder(C, K, seed, arm, graph)
    if str(structure).upper() not in ("MLP", "POINT_MLP", "MLP_POINT", "TCN", "TCN_TIME", "CAUSAL_TCN"):
        raise ValueError(f"Unknown encoder structure: {structure}")
    return CausalAutoencoder(structure, C, K, seed, arm)


def losses(model: nn.Module, x: torch.Tensor, y: torch.Tensor, mx: torch.Tensor, my: torch.Tensor, arm: str | None = None) -> dict[str, torch.Tensor]:
    """C loss: all arms read the same TRAIN past/future, with exact mask moments."""
    combined = torch.cat([x, y], dim=1)
    observed = torch.cat([mx, my], dim=1).bool()
    z = model.encode(combined)
    reconstructed = model.decode(z)
    score_mask = observed.clone()
    score_mask[:, :BURN_IN] = False
    rec_errors = (reconstructed - combined).square()
    count = score_mask.sum()
    if int(count.detach().cpu()) == 0:
        raise ValueError("No observed reconstruction targets after the fixed burn-in")
    rec = rec_errors[score_mask].mean()
    valid_rows = score_mask.any(dim=-1)
    values = z[valid_rows]
    centered = values - values.mean(dim=0, keepdim=True)
    variance = centered.square().mean(dim=0)
    covariance = centered.T @ centered / values.shape[0]
    zero = rec.new_zeros(())
    selected = getattr(model, "arm", "REC") if arm is None else arm
    if isinstance(model, GraphTimeAutoencoder):
        product_l1 = model.product_penalty()
        return {
            "total": rec + GT_RHO * product_l1, "rec": rec, "var": zero, "cov": zero,
            "pred": zero, "product_l1": product_l1, "valid_latent_rows": valid_rows.sum(),
            "imputed_channel_fraction": (~observed[valid_rows]).float().mean(),
            "min_latent_variance": variance.min(), "mean_latent_variance": variance.mean(),
        }
    selected = normalize_arm(selected)
    var_loss = torch.relu(1. - torch.sqrt(variance + EPSILON)).mean()
    if z.shape[-1] == 1:
        cov_loss = zero
    else:
        cov_loss = (covariance.square().sum() - covariance.diag().square().sum()) / (z.shape[-1] * (z.shape[-1] - 1))
    pred_loss = zero
    if selected != "REC":
        zp = z[:, :x.shape[1]]
        zf = z[:, x.shape[1]:]
        denominator = variance + EPSILON
        if selected == "GUARDED_STOPGRAD":
            zp, zf, denominator = zp.detach(), zf.detach(), denominator.detach()
        predicted = model.aux_predictor(zp)
        valid_future = my.bool().any(dim=-1).unsqueeze(-1).expand_as(zf)
        if int(valid_future.sum().detach().cpu()) == 0:
            raise ValueError("No observed TRAIN future row for the auxiliary target")
        pred_loss = (((predicted - zf).square() / denominator)[valid_future]).mean()
    if selected == "REC":
        var_loss, cov_loss = zero, zero
        beta = 0.
    else:
        beta = 0.1 if selected == "SELF_0p1" else 1.
    return {
        "total": rec + var_loss + 0.01 * cov_loss + beta * pred_loss,
        "rec": rec, "var": var_loss, "cov": cov_loss, "pred": pred_loss,
        "product_l1": zero, "valid_latent_rows": valid_rows.sum(),
        "imputed_channel_fraction": (~observed[valid_rows]).float().mean(),
        "min_latent_variance": variance.min(), "mean_latent_variance": variance.mean(),
    }


def build_ts2vec(C: int, K: int, seed: int, device: str = "cpu", size: str = "small"):
    wrapper = _load("_fullscreen_previous_ts2vec_adapter", ROOT / "research/level_k_tradeoff_20261011/ts2vec_adapter.py")
    hidden, depth = (16, 3) if str(size).lower() == "small" else (64, 10)
    configuration = {
        "hidden_dims": hidden, "depth": depth, "learning_rate": 0.001,
        "batch_size": 32, "max_train_length": 512, "temporal_unit": 0, "n_iters": 2880,
    }
    adapter = wrapper.NativeTs2VecAdapter(C, K, seed, device=device, configuration=configuration)
    adapter.family = f"TS2VEC_CAUSAL_{str(size).upper()}"
    return adapter


class DCAAdapter(nn.Module):
    family = "OFFICIAL_DCA"
    encoder_lag = 0

    def __init__(self, mean, coefficients):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float64))
        self.register_buffer("coefficients", torch.as_tensor(coefficients, dtype=torch.float64))
        self.channels, self.latent = self.coefficients.shape
        self.aux_predictor = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        coefficients = self.coefficients.to(device=x.device, dtype=x.dtype)
        return (x - mean) @ coefficients

    def encode_future(self, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        return self.encode(future)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.coefficients.to(device=z.device, dtype=z.dtype).T + self.mean.to(device=z.device, dtype=z.dtype)

    def decode_future(self, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        return self.decode(future)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def official_dca_module():
    vendor = HERE / "vendor/dca"
    receipt = json.loads((vendor / "provenance.json").read_text(encoding="utf-8"))
    for item in receipt["files"]:
        if hashlib.sha256((vendor / item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"DCA upstream source seal mismatch: {item['path']}")
    alias = "_fullscreen_pinned_dca"
    if alias not in sys.modules:
        spec = importlib.util.spec_from_file_location(alias, vendor / "__init__.py", submodule_search_locations=[str(vendor)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
    return sys.modules[alias + ".dca"], receipt


def fit_dca(values: np.ndarray, K: int, seed: int, T: int = 8, maxiter: int = 200, maxfun: int = 400):
    """Official objective and optimizer; only the public optimizer resource cap is added."""
    upstream, provenance = official_dca_module()
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or len(values) <= 2 * T:
        raise ValueError("DCA requires a finite TRAIN-only time-by-channel sequence longer than2T")
    estimator = upstream.DynamicalComponentsAnalysis(
        d=int(K), T=int(T), init="random_ortho", n_init=1, stride=1,
        chunk_cov_estimate=None, tol=1e-6, ortho_lambda=10., verbose=False,
        block_toeplitz=False, method="toeplitzify", device="cpu", dtype=torch.float64,
        rng_or_seed=int(seed),
    )
    native_minimize = upstream.minimize
    optimizations = []

    def bounded_minimize(*args, **kwargs):
        options = dict(kwargs.get("options", {}))
        options.update(maxiter=int(maxiter), maxfun=int(maxfun))
        kwargs["options"] = options
        result = native_minimize(*args, **kwargs)
        optimizations.append({
            "success": bool(result.success), "status": int(result.status), "message": str(result.message),
            "iterations": int(result.nit), "function_evaluations": int(result.nfev),
            "gradient_evaluations": int(result.njev), "objective": float(result.fun),
            "gradient_infinity_norm": float(np.max(np.abs(result.jac))), "options": options,
        })
        return result

    wall_started, cpu_started = time.perf_counter(), time.process_time()
    with patch.object(upstream, "minimize", bounded_minimize):
        estimator.fit(values, regularization=None, reg_ops=None)
    coefficients = np.asarray(estimator.coef_, dtype=np.float64)
    if coefficients.shape != (values.shape[1], K) or not np.isfinite(coefficients).all():
        raise RuntimeError("Official DCA returned an invalid or rank-deficient projection")
    model = DCAAdapter(estimator.mean_, coefficients)
    predictive_information = estimator.score()
    if isinstance(predictive_information, torch.Tensor):
        predictive_information = predictive_information.detach().cpu().numpy()
    receipt = {
        "upstream_commit": provenance["commit"], "data_shape": list(values.shape),
        "data_sha256": hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest(),
        "T": int(T), "K": int(K), "seed": int(seed), "initialization": "random_ortho", "n_init": 1,
        "optimizer": "official scipy L-BFGS-B", "tolerance_ftol": 1e-6, "ortho_lambda": 10.,
        "maxiter": int(maxiter), "maxfun": int(maxfun), "dtype": "float64", "device": "cpu",
        "covariance_method": "official toeplitzify with native nonpositive-spectrum rectification epsilon1e-6",
        "covariance_regularization": None, "optimizations": optimizations,
        "train_predictive_information": float(predictive_information),
        "wall_seconds": time.perf_counter() - wall_started, "process_cpu_seconds": time.process_time() - cpu_started,
        "fit_parameter_count": int(values.shape[1] * K), "trainable_parameters_after_fit": 0,
        "F_backward_calls": 0, "F_forward_calls": 0,
        "scope": "bounded official DCA fit; capped nonconvergence is retained, never described as the global optimum",
    }
    return model, receipt


def native_temporal_downsample(features: torch.Tensor, rate: int = 2) -> torch.Tensor:
    """Eq.4 mean-summary fixture, excluded from the length-preserving main path."""
    if features.shape[1] % rate:
        raise ValueError("Native sampling fixture requires T divisible by rate")
    return features.reshape(features.shape[0], features.shape[1] // rate, rate, *features.shape[2:]).mean(dim=2)


def native_temporal_upsample(features: torch.Tensor, rate: int = 2) -> torch.Tensor:
    """Eq.8: zero insertion on one-based timestamps t=k*r."""
    output = features.new_zeros((features.shape[0], features.shape[1] * rate, *features.shape[2:]))
    output[:, rate - 1::rate] = features
    return output
