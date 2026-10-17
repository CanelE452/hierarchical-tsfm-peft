"""Preparation-only bridge to the unchanged, pinned official LIFT refiner.

The source cache is private; its pinned tree has no LICENSE file. This module
does not vendor the upstream implementation or claim a paper reproduction.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import torch
from torch import nn

from .initialization import MANIFEST, _sha


LIFT_COMMIT = '79cf8f157bb9a616c9733daf1e9c2cfefdc98c94'


def _official_module(source_root):
    root = Path(source_root)
    root = root / 'LIFT' if (root/'LIFT').is_dir() else root
    source = next(item for item in json.loads(MANIFEST.read_text(encoding='utf-8'))['sources'] if item['name'] == 'LIFT')
    if source['commit'] != LIFT_COMMIT:
        raise ValueError('Official LIFT source commit mismatch')
    required = {'models/LIFT.py', 'models/DLinear.py', 'util/lead_estimate.py'}
    for item in source['files']:
        if item['path'] in required:
            path = root/item['path']
            if not path.is_file() or _sha(path) != item['sha256']:
                raise ValueError('Official LIFT source hash mismatch: '+item['path'])
            required.remove(item['path'])
    if required:
        raise ValueError('Official LIFT source manifest incomplete')
    name = '_adapter_lift_' + LIFT_COMMIT + '_' + hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:12]
    if name+'.models.LIFT' in sys.modules:
        return sys.modules[name+'.models.LIFT']
    for suffix, directory in [('', root), ('.models', root/'models'), ('.util', root/'util')]:
        package = ModuleType(name+suffix)
        package.__path__ = [str(directory)]
        sys.modules[name+suffix] = package
    lead = importlib.import_module(name+'.util.lead_estimate')
    previous = {key: sys.modules.get(key) for key in ['util', 'util.lead_estimate']}
    try:
        # The upstream absolute import is temporarily bound without editing it.
        sys.modules['util'] = sys.modules[name+'.util']
        sys.modules['util.lead_estimate'] = lead
        return importlib.import_module(name+'.models.LIFT')
    finally:
        for key, value in previous.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


class _ExternalOnly(nn.Module):
    def forward(self, x):
        raise RuntimeError('Provide the anchored final base prediction to LIFT')


class ExternalPredictionRefiner(nn.Module):
    def __init__(self, model, length, horizon, channels):
        super().__init__()
        self.model = model
        self.length, self.horizon, self.channels = length, horizon, channels

    def forward(self, context, final_prediction):
        if (context.ndim != 3 or tuple(context.shape[1:]) != (self.length, self.channels)
                or final_prediction.shape != (context.shape[0], self.horizon, self.channels)):
            raise ValueError('LIFT requires full original-channel history and anchored final forecast')
        if context.dtype != torch.float32 or final_prediction.dtype != torch.float32:
            raise ValueError('Official complex64 LIFT preparation requires FP32 inputs')
        return self.model(context, y_hat=final_prediction)


def make_official_refiner(source_root, *, length, horizon, channels, n_leaders=4,
                         states=20, temperature=1., seed=0, backbone=None):
    module = _official_module(source_root)
    config = SimpleNamespace(seq_len=length, pred_len=horizon, enc_in=channels,
                             in_dim=1, leader_num=n_leaders, state_num=states,
                             temperature=temperature, local_rank=-1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = module.Model(_ExternalOnly() if backbone is None else backbone, config)
    return ExternalPredictionRefiner(model, length, horizon, channels)


def parameter_inventory(module):
    parameters = list(module.parameters())
    return dict(parameter_elements=sum(p.numel() for p in parameters),
                complex_elements=sum(p.numel() for p in parameters if p.is_complex()),
                real_degrees_of_freedom=sum(p.numel()*(2 if p.is_complex() else 1) for p in parameters),
                bytes=sum(p.numel()*p.element_size() for p in parameters))
