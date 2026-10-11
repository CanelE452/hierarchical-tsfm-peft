"""PCA-free encoder families for the same-decoder contract study."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


FAMILIES = ("LINEAR_POINT", "LINEAR_TIME", "MLP_POINT", "TCN_TIME", "GRAPH_TIME_NL")


def seeded_qr(channels: int, latent: int, seed: int) -> torch.Tensor:
    if latent > channels:
        raise ValueError("latent dimension must not exceed channel dimension for QR initialization")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    raw = torch.randn(channels, latent, generator=generator, dtype=torch.float32)
    q, r = torch.linalg.qr(raw, mode="reduced")
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return (q * signs).contiguous()


def _normal_(parameter: torch.Tensor, seed: int, scale: float) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    values = torch.randn(parameter.shape, generator=generator, dtype=torch.float32) * float(scale)
    with torch.no_grad():
        parameter.copy_(values.to(parameter.device))


def causal_fir_apply(x: torch.Tensor, taps: torch.Tensor, prefix: torch.Tensor | None = None) -> torch.Tensor:
    if x.ndim != 3 or taps.ndim != 3:
        raise ValueError("x must be [B,T,I] and taps must be [lag,I,O]")
    lag_count, in_channels, out_channels = taps.shape
    if x.shape[2] != in_channels:
        raise ValueError("tap input channels do not match x")
    q = lag_count - 1
    if prefix is None:
        prefix = x.new_zeros((x.shape[0], q, in_channels))
    else:
        if prefix.ndim != 3 or prefix.shape[0] != x.shape[0] or prefix.shape[2] != in_channels:
            raise ValueError("prefix must be [B,T_prefix,I] with matching batch/channels")
        prefix = prefix[:, -q:] if q else prefix[:, :0]
        if prefix.shape[1] < q:
            pad = x.new_zeros((x.shape[0], q - prefix.shape[1], in_channels))
            prefix = torch.cat([pad, prefix], dim=1)
    full = torch.cat([prefix, x], dim=1)
    out = x.new_zeros((x.shape[0], x.shape[1], out_channels))
    for lag in range(lag_count):
        source = full[:, q - lag : q - lag + x.shape[1], :]
        out = out + torch.einsum("bti,io->bto", source, taps[lag])
    return out


def decode_future_with_taps(z_history: torch.Tensor, z_future: torch.Tensor, taps: torch.Tensor) -> torch.Tensor:
    q = taps.shape[0] - 1
    return causal_fir_apply(z_future, taps, prefix=z_history[:, -q:] if q else z_history[:, :0])


class LinearPointEncoder(nn.Module):
    family = "LINEAR_POINT"
    encoder_lag = 0

    def __init__(self, channels: int, latent: int, seed: int):
        super().__init__()
        self.weight = nn.Parameter(seeded_qr(channels, latent, seed))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("btc,ck->btk", x, self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


class LinearTimeEncoder(nn.Module):
    family = "LINEAR_TIME"
    encoder_lag = 8

    def __init__(self, channels: int, latent: int, seed: int, lag: int = 8):
        super().__init__()
        taps = torch.zeros(lag + 1, channels, latent, dtype=torch.float32)
        taps[0] = seeded_qr(channels, latent, seed)
        self.taps = nn.Parameter(taps)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return causal_fir_apply(x, self.taps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


class MLPPointEncoder(nn.Module):
    family = "MLP_POINT"
    encoder_lag = 0

    def __init__(self, channels: int, latent: int, seed: int):
        super().__init__()
        width = int(round(16 * (9 * channels + latent) / (channels + latent)))
        self.width = width
        self.fc1 = nn.Linear(channels, width, bias=False)
        self.fc2 = nn.Linear(width, latent, bias=False)
        _normal_(self.fc1.weight, seed + 101, math.sqrt(2.0 / (channels + width)))
        _normal_(self.fc2.weight, seed + 102, math.sqrt(2.0 / (width + latent)))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.tanh(self.fc1(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


class TCNTimeEncoder(nn.Module):
    family = "TCN_TIME"
    encoder_lag = 8

    def __init__(self, channels: int, latent: int, seed: int, hidden: int = 16, kernel_size: int = 9):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv1d(channels, hidden, kernel_size=kernel_size, bias=False)
        self.proj = nn.Linear(hidden, latent, bias=False)
        _normal_(self.conv.weight, seed + 201, math.sqrt(2.0 / (channels * kernel_size + hidden)))
        _normal_(self.proj.weight, seed + 202, math.sqrt(2.0 / (hidden + latent)))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        y = F.pad(x.transpose(1, 2), (self.kernel_size - 1, 0))
        y = self.conv(y).transpose(1, 2)
        return self.proj(torch.tanh(y))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


class GraphTimeNLEncoder(nn.Module):
    family = "GRAPH_TIME_NL"
    encoder_lag = 8

    def __init__(self, channels: int, latent: int, seed: int, graph: np.ndarray | torch.Tensor, features: int = 4, lag: int = 8):
        super().__init__()
        if graph is None:
            graph = np.zeros((channels, channels), dtype=np.float32)
        graph_tensor = torch.as_tensor(graph, dtype=torch.float32)
        if graph_tensor.shape != (channels, channels):
            raise ValueError(f"graph shape {tuple(graph_tensor.shape)} does not match {(channels, channels)}")
        self.channels = int(channels)
        self.features = int(features)
        self.lag = int(lag)
        self.register_buffer("graph", graph_tensor.contiguous())
        self.alpha = nn.Parameter(torch.empty(lag + 1, features, dtype=torch.float32))
        self.beta = nn.Parameter(torch.empty(lag + 1, features, dtype=torch.float32))
        self.out = nn.Parameter(torch.empty(channels * features, latent, dtype=torch.float32))
        _normal_(self.alpha, seed + 301, 1.0 / math.sqrt(lag + 1))
        _normal_(self.beta, seed + 302, 1.0 / math.sqrt(lag + 1))
        _normal_(self.out, seed + 303, math.sqrt(2.0 / (channels * features + latent)))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        padded = F.pad(x.transpose(1, 2), (self.lag, 0)).transpose(1, 2)
        features = x.new_zeros((x.shape[0], x.shape[1], self.channels, self.features))
        for lag in range(self.lag + 1):
            own = padded[:, self.lag - lag : self.lag - lag + x.shape[1], :]
            neighbor = torch.einsum("btc,dc->btd", own, self.graph)
            features = features + own.unsqueeze(-1) * self.alpha[lag] + neighbor.unsqueeze(-1) * self.beta[lag]
        return torch.einsum("btd,dk->btk", torch.tanh(features).flatten(2), self.out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


class CausalFIRDecoder(nn.Module):
    def __init__(self, channels: int, latent: int, seed: int, lag: int = 8):
        super().__init__()
        taps = torch.zeros(lag + 1, latent, channels, dtype=torch.float32)
        taps[0] = seeded_qr(channels, latent, seed).T
        self.taps = nn.Parameter(taps)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return causal_fir_apply(z, self.taps)

    def decode_future(self, z_history: torch.Tensor, z_future: torch.Tensor) -> torch.Tensor:
        return decode_future_with_taps(z_history, z_future, self.taps)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode(z)


def build_encoder(
    family: str,
    channels: int,
    latent: int,
    seed: int,
    graph: np.ndarray | torch.Tensor | None = None,
) -> nn.Module:
    family = str(family).upper()
    if family == "LINEAR_POINT":
        return LinearPointEncoder(channels, latent, seed)
    if family == "LINEAR_TIME":
        return LinearTimeEncoder(channels, latent, seed)
    if family == "MLP_POINT":
        return MLPPointEncoder(channels, latent, seed)
    if family == "TCN_TIME":
        return TCNTimeEncoder(channels, latent, seed)
    if family == "GRAPH_TIME_NL":
        return GraphTimeNLEncoder(channels, latent, seed, graph)
    raise ValueError(f"Unknown encoder family: {family}")


def build_decoder(channels: int, latent: int, seed: int, lag: int = 8) -> CausalFIRDecoder:
    return CausalFIRDecoder(channels, latent, seed, lag=lag)


def current_linear_map(model: nn.Module) -> torch.Tensor:
    if isinstance(model, LinearPointEncoder):
        return model.weight
    if isinstance(model, LinearTimeEncoder):
        return model.taps[0]
    raise TypeError(f"{type(model).__name__} does not expose a linear current tap")


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
