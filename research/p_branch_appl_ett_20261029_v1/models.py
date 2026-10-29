"""Remove one complete P residual branch without reinitializing the other."""
from contextlib import contextmanager

import torch
from torch import nn


METHODS = ('P_TIME_ONLY', 'P_CHANNEL_ONLY')
_original_factory = None


class SingleBranchMixer(nn.Module):
    def __init__(self, original, method):
        super().__init__()
        if method not in METHODS:
            raise ValueError('Unknown P branch')
        self.method = method
        if method == 'P_TIME_ONLY':
            self.temporal_norm = original.temporal_norm
            self.temporal = original.temporal
        else:
            self.feature_norm = original.feature_norm
            self.feature = original.feature

    def forward(self, x):
        if self.method == 'P_TIME_ONLY':
            return x + self.temporal(self.temporal_norm(x).transpose(1, 2)).transpose(1, 2)
        return x + self.feature(self.feature_norm(x))


def create_model(spec, task, fit_end, foundation):
    if _original_factory is None:
        raise RuntimeError('Use patch_core before creating a branch model')
    method = spec['method_id']
    if method not in METHODS:
        raise ValueError('Only registered P branches may train')
    model, normalization = _original_factory(dict(spec, family='B2'), task, fit_end, foundation)
    model.front.mixer = SingleBranchMixer(model.front.mixer, method)
    model.method_id = method
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
        raise RuntimeError('P branch model hooks already installed')
    originals = fit_module.create_model, fit_module.predict_model
    _original_factory = originals[0]
    fit_module.create_model, fit_module.predict_model = create_model, predict_model
    try:
        yield
    finally:
        fit_module.create_model, fit_module.predict_model = originals
        _original_factory = None
