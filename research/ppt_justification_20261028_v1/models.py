"""Registered F replacement and separate anchor/lag decoder interventions."""
from contextlib import contextmanager

import torch
from torch import nn


METHODS = ('NO_F_LINEAR', 'NO_F_LINEAR_NO_P', 'NO_ANCHOR', 'A5_LAG0')
LAGS = (0, 1, 2, 4, 8, 16, 32)
_original_factory = None


class DirectNoFoundation(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.adapter = original
        self.head = nn.Linear(original.length, original.horizon, bias=True).to(
            device=original.E.device, dtype=original.E.dtype)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.weight[:, -1] = 1.
            self.head.bias.zero_()
        self.method_id = 'NO_F_LINEAR'

    def forward(self, x, foundation=None, latent_k=None):
        z = self.adapter.encode(x, latent_k)
        future = self.head(z.transpose(1, 2)).transpose(1, 2)
        return self.adapter.decode(x, z, future, latent_k)


class DecoderIntervention(nn.Module):
    def __init__(self, original, method):
        super().__init__()
        if method not in ('NO_ANCHOR', 'A5_LAG0'):
            raise ValueError('Unknown decoder intervention')
        self.front, self.E = original.front, original.E
        self.register_buffer('center', original.center)
        self.length, self.horizon = original.length, original.horizon
        self.channels, self.latent_k = original.channels, original.latent_k
        self.kmax, self.k_grid = original.kmax, original.k_grid
        self.encoder_kind = original.encoder_kind
        self.method_id = method
        if method == 'NO_ANCHOR':
            self.D = original.D
        else:
            self.D0 = nn.Parameter(original.D[0].detach().clone())

    def _k(self, latent_k=None):
        k = self.latent_k if latent_k is None else latent_k
        if type(k) is not int or k != self.latent_k:
            raise ValueError('One registered fixed K is required')
        return k

    def encode(self, x, latent_k=None):
        self._k(latent_k)
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.length, self.channels):
            raise ValueError('Expected registered [B,L,C] history')
        return (self.front(x) - self.center) @ self.E

    def decode(self, x, z, future, latent_k=None):
        k = self._k(latent_k)
        if tuple(z.shape) != (x.shape[0], self.length, k) or tuple(future.shape) != (x.shape[0], self.horizon, k):
            raise ValueError('Latent history/future shapes differ')
        if self.method_id == 'A5_LAG0':
            return x[:, -1:] + (future - z[:, -1:]) @ self.D0
        joined = torch.cat((z, future), 1)
        features = torch.cat([joined[:, self.length-lag:self.length-lag+self.horizon] - z[:, -1-lag, None]
                              for lag in LAGS], -1)
        return features @ self.D.reshape(7*k, self.channels)

    def forward(self, x, foundation, latent_k=None):
        z = self.encode(x, latent_k)
        return self.decode(x, z, foundation(z), latent_k)


def create_model(spec, task, fit_end, foundation):
    if _original_factory is None or spec['method_id'] not in METHODS:
        raise ValueError('Registered model creation requires patch_core')
    original, normalization = _original_factory(dict(spec, family='B2'), task, fit_end, foundation)
    if spec['method_id'].startswith('NO_F_LINEAR'):
        if spec['method_id'] == 'NO_F_LINEAR_NO_P':
            original.front = nn.Identity()
        model = DirectNoFoundation(original)
        model.method_id = spec['method_id']
    else:
        model = DecoderIntervention(original, spec['method_id'])
    return model, normalization


def predict_model(model, x, foundation, family, latent_k=None):
    return model(x, foundation, latent_k)


@contextmanager
def evaluation_scope(model, *, deterministic=False):
    modes = {module: module.training for module in model.modules()}
    try:
        model.eval()
        with torch.no_grad():
            yield model
    finally:
        for module, training in modes.items():
            module.training = training


@contextmanager
def patch_core(fit_module):
    global _original_factory
    if _original_factory is not None:
        raise RuntimeError('Model hooks already installed')
    originals = fit_module.create_model, fit_module.predict_model
    _original_factory = originals[0]
    fit_module.create_model, fit_module.predict_model = create_model, predict_model
    try:
        yield
    finally:
        fit_module.create_model, fit_module.predict_model = originals
        _original_factory = None
