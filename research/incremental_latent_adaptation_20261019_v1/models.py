"""Correction-only paths; the caller adds the frozen G exactly once."""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from research.informs_broad_validation_20261015_v1.networks import LAGS


METHODS = ('RESIDUAL_F', 'RESIDUAL_Q', 'RESIDUAL_RAW_MLP', 'READOUT_ONLY')


def lag_features(z, future):
    length = z.shape[1]
    horizon = future.shape[1]
    joined = torch.cat((z, future), dim=1)
    return torch.cat([joined[:, length-lag:length-lag+horizon] - z[:, -1-lag, None]
                      for lag in LAGS], dim=-1)


def matched_hidden(length, horizon, target):
    unit = length + horizon + 1
    exact = (target-horizon)/unit
    candidates = {max(1, math.floor(exact)), max(1, math.ceil(exact))}
    return min(candidates, key=lambda d: (abs(unit*d+horizon-target), d))


class TimeMLP(nn.Module):
    def __init__(self, length, horizon, hidden):
        super().__init__()
        self.input = nn.Linear(length, hidden)
        self.output = nn.Linear(hidden, horizon)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        return self.output(F.gelu(self.input(x.transpose(1, 2)))).transpose(1, 2)


class ResidualLatent(nn.Module):
    def __init__(self, parent, method):
        super().__init__()
        if method not in ('RESIDUAL_F', 'RESIDUAL_Q', 'READOUT_ONLY'):
            raise ValueError(method)
        self.method = method
        self.length, self.horizon, self.channels = parent.length, parent.horizon, parent.channels
        self.k = parent.kmax
        self.front = copy.deepcopy(parent.front)
        self.front.requires_grad_(method != 'READOUT_ONLY')
        self.register_buffer('center', parent.center.detach().clone())
        self.E = nn.Parameter(parent.E.detach().clone())
        self.D = nn.Parameter(torch.zeros_like(parent.D))
        self.q = TimeMLP(self.length, self.horizon, 32) if method == 'RESIDUAL_Q' else None
        if method == 'READOUT_ONLY':
            self.front.requires_grad_(False)
            self.E.requires_grad_(False)

    def encode(self, x):
        return (self.front(x)-self.center) @ self.E

    def features(self, x, foundation=None):
        z = self.encode(x)
        if self.q is None:
            u = foundation(z)[:, :self.horizon]
        else:
            u = z[:, -1:] + self.q(z-z[:, -1:])
        return lag_features(z, u)

    def from_features(self, phi):
        return phi @ self.D.reshape(7*self.k, self.channels)

    def forward(self, x, foundation=None):
        return self.from_features(self.features(x, foundation))


class ResidualRawMLP(nn.Module):
    def __init__(self, parent, target_parameters):
        super().__init__()
        self.method = 'RESIDUAL_RAW_MLP'
        self.length, self.horizon, self.channels = parent.length, parent.horizon, parent.channels
        self.k = 0
        self.hidden = matched_hidden(self.length, self.horizon, target_parameters)
        self.mlp = TimeMLP(self.length, self.horizon, self.hidden)

    def forward(self, x, foundation=None):
        return self.mlp(x-x[:, -1:])


def make(parent, method, seed):
    if method not in METHODS:
        raise ValueError(method)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        if method == 'RESIDUAL_RAW_MLP':
            count = sum(p.numel() for p in parent.front.parameters()) + parent.E.numel() + parent.D.numel()
            return ResidualRawMLP(parent, count)
        return ResidualLatent(parent, method)


def parameter_counts(model):
    return dict(total_parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                buffers=sum(p.numel() for p in model.buffers()))
