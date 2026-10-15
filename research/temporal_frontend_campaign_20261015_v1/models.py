"""Small campaign variants, not paper benchmark reproductions.

Official source mapping, read at pinned revisions on 2026-10-15:
Google TSMixer: alternating temporal and feature residual mixing, then L->H.
https://github.com/google-research/google-research/blob/5b09c22d73a9d35eb6c5d2a99b95677a45053466/tsmixer/tsmixer_basic/models/tsmixer.py
TimeMixer: mean-pooled scales, symmetric moving-average decomposition,
fine->coarse seasonal/residual and coarse->fine trend mixing in PDM; FMM
uses separate per-scale future heads.
https://github.com/kwuking/TimeMixer/blob/e24610583b36fdd8c76cc17a8df4e65759a5f460/models/TimeMixer.py
https://github.com/kwuking/TimeMixer/blob/e24610583b36fdd8c76cc17a8df4e65759a5f460/layers/Autoformer_EncDec.py

Variations: one block, GELU, width-limited temporal MLPs, no dropout, no
learned embedding/RevIN, and a shared feature MLP in the original C space.
B2 replaces Google's full L->L temporal dense with L->hidden->L. B4 front
replaces FMM with zero-initialized residual heads onto the original L grid;
its direct counterpart retains per-scale L_s->H heads. These choices do
not reproduce the papers' training setups, full architectures, or results.

Every operation sees only the supplied history [origin-L, origin), but can
use that entire window. These representations are NOT prefix causal.
The symmetric B4 smoother differs from R1's trailing-only smoother.
Pooling uses nonoverlapping blocks starting at history row zero. Learned
front heads map each scale onto original row indices 0..L-1; coarse rows
are never concatenated as if they had the original sampling interval.

CP0 front outputs equal their input. Only final residual-output linears
are zero initialized; their input linears and all cross-scale/core layers
are nonzero initialized. On the first backward, output layers can receive
nonzero gradients while earlier layers have zero gradients through the
zero output weights. After one nonzero output-layer update the interior
can receive gradients. B2's two zero outputs sit on separate residual
branches, not in a chain. B3/B4 have one zero output per parallel scale.
"""

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def parameter_count(module: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


class _MLP(nn.Module):
    def __init__(self, n_in: int, hidden: int, n_out: int, zero_output: bool = False):
        super().__init__()
        self.input = nn.Linear(n_in, hidden)
        self.output = nn.Linear(hidden, n_out)
        if zero_output:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, x):
        return self.output(F.gelu(self.input(x)))


class _AlternatingMixer(nn.Module):
    def __init__(self, length: int, channels: int, hidden: int):
        super().__init__()
        self.temporal_norm = nn.LayerNorm((length, channels), elementwise_affine=False)
        self.feature_norm = nn.LayerNorm((length, channels), elementwise_affine=False)
        self.temporal = _MLP(length, hidden, length, zero_output=True)
        self.feature = _MLP(channels, hidden, channels, zero_output=True)

    def forward(self, x):
        temporal = self.temporal(self.temporal_norm(x).transpose(1, 2)).transpose(1, 2)
        y = x + temporal
        return y + self.feature(self.feature_norm(y))


class _MultiscalePast(nn.Module):
    def __init__(self, length, channels, hidden, scales, kernel,
                 decomposition, seasonal_mix, trend_mix):
        super().__init__()
        self.scales = tuple(scales)
        self.lengths = tuple(length // s for s in self.scales)
        self.kernel = kernel
        self.decomposition = decomposition
        self.channel = _MLP(channels, hidden, channels)
        self.down = nn.ModuleList(
            _MLP(a, hidden, b) for a, b in zip(self.lengths, self.lengths[1:])
        ) if seasonal_mix else nn.ModuleList()
        self.up = nn.ModuleList(
            _MLP(b, hidden, a) for a, b in zip(self.lengths, self.lengths[1:])
        ) if trend_mix else nn.ModuleList()

    def scale_inputs(self, x):
        return [x if s == 1 else F.avg_pool1d(x.transpose(1, 2), s, s).transpose(1, 2)
                for s in self.scales]

    def decompose(self, x):
        if not self.decomposition:
            # Both directional paths retain real input and live parameters.
            return 0.5 * x, 0.5 * x
        padding = (self.kernel - 1) // 2
        trend = F.avg_pool1d(F.pad(x.transpose(1, 2), (padding, padding), mode="replicate"),
                             self.kernel, stride=1).transpose(1, 2)
        return x - trend, trend

    def forward(self, x):
        seasonal, trends = [], []
        for value in self.scale_inputs(x):
            residual, trend = self.decompose(value)
            seasonal.append((residual + self.channel(residual)).transpose(1, 2))
            trends.append((trend + self.channel(trend)).transpose(1, 2))
        if self.down:
            for i, down in enumerate(self.down):
                seasonal[i + 1] = seasonal[i + 1] + down(seasonal[i])
        if self.up:
            for i in reversed(range(len(self.up))):
                trends[i] = trends[i] + self.up[i](trends[i + 1])
        return [a + b for a, b in zip(seasonal, trends)]


def _validate_configuration(length, channels, hidden, scales, kernel):
    if length < 1 or channels < 1 or hidden < 1:
        raise ValueError("length, channels and hidden must be positive")
    if (not scales or scales[0] != 1 or tuple(scales) != tuple(sorted(set(scales)))
            or any(not isinstance(s, int) or s < 1 or length % s for s in scales)):
        raise ValueError("scales must be increasing integer divisors of length, starting with 1")
    if not isinstance(kernel, int) or kernel < 1 or kernel % 2 == 0:
        raise ValueError("kernel must be a positive odd integer")


def _validate_input(x, length, channels):
    if x.ndim != 3 or x.shape[1:] != (length, channels):
        raise ValueError(f"expected [batch, {length}, {channels}], got {tuple(x.shape)}")


class TemporalFront(nn.Module):
    """Past->past front: [B,L,C] -> [B,L,C], preserving original row indices.

    B0: identity. B1: channel-shared temporal residual. B2: alternating
    temporal/feature residuals. B3/B4: identical-capacity bidirectional
    multiscale mixing, with raw half-signals / residual-trend decomposition.

    B4 decomposition=False is exactly the B3 structural control. The two
    *_mix=False options remove that direction's cross-scale MLPs while
    preserving the unmixed component at every scale. Use fresh matched
    fitting for ablations; toggling after a fit is not a refit comparison.
    """

    def __init__(self, kind: str, length: int = 512, channels: int = 1,
                 hidden: int = 32, scales: Sequence[int] = (1, 2, 4), kernel: int = 25,
                 *, decomposition: bool = True, seasonal_mix: bool = True,
                 trend_mix: bool = True):
        super().__init__()
        _validate_configuration(length, channels, hidden, scales, kernel)
        if kind not in ("B0", "B1", "B2", "B3", "B4"):
            raise ValueError(f"unknown temporal-front kind {kind}")
        self.kind, self.length, self.channels = kind, length, channels
        self.hidden, self.scales, self.kernel = hidden, tuple(scales), kernel
        if kind == "B1":
            self.temporal = _MLP(length, hidden, length, zero_output=True)
        elif kind == "B2":
            self.mixer = _AlternatingMixer(length, channels, hidden)
        elif kind in ("B3", "B4"):
            self.multiscale = _MultiscalePast(
                length, channels, hidden, scales, kernel,
                decomposition=kind == "B4" and decomposition,
                seasonal_mix=seasonal_mix, trend_mix=trend_mix,
            )
            self.fusion = nn.ModuleList(
                _MLP(n, hidden, length, zero_output=True) for n in self.multiscale.lengths
            )

    def forward(self, x):
        _validate_input(x, self.length, self.channels)
        if self.kind == "B0":
            return x
        if self.kind == "B1":
            return x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        if self.kind == "B2":
            return self.mixer(x)
        values = self.multiscale(x)
        delta = sum(head(value) for head, value in zip(self.fusion, values)) / len(values)
        return x + delta.transpose(1, 2)


class DirectMixer(nn.Module):
    """Small direct forecasting variations: [B,L,C] -> [B,H,C].

    B2 uses the alternating mixer then an L->H head. B4 uses the same PDM
    core with one L_s->H head per scale, followed by their sum (FMM
    principle). Heads are normally initialized: direct CP0 is not an
    identity forecast, and B4's core can receive first-backward gradients.
    Neither model includes the papers' full embedding/normalization setup.
    """

    def __init__(self, kind: str, length: int = 512, horizon: int = 48,
                 channels: int = 1, hidden: int = 32,
                 scales: Sequence[int] = (1, 2, 4), kernel: int = 25,
                 *, decomposition: bool = True, seasonal_mix: bool = True,
                 trend_mix: bool = True):
        super().__init__()
        _validate_configuration(length, channels, hidden, scales, kernel)
        if kind not in ("B2", "B4") or horizon < 1:
            raise ValueError("DirectMixer requires kind B2/B4 and a positive horizon")
        self.kind, self.length, self.horizon, self.channels = kind, length, horizon, channels
        if kind == "B2":
            self.front = TemporalFront(kind, length, channels, hidden, scales, kernel)
            self.head = nn.Linear(length, horizon)
        else:
            self.multiscale = _MultiscalePast(
                length, channels, hidden, scales, kernel, decomposition, seasonal_mix, trend_mix
            )
            self.heads = nn.ModuleList(nn.Linear(n, horizon) for n in self.multiscale.lengths)

    def forward(self, x):
        _validate_input(x, self.length, self.channels)
        if self.kind == "B2":
            return self.head(self.front(x).transpose(1, 2)).transpose(1, 2)
        values = self.multiscale(x)
        return sum(head(value) for head, value in zip(self.heads, values)).transpose(1, 2)
