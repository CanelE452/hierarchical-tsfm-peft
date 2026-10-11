"""Unmodified official TS2Vec with isolated imports and direct-study export rules."""
from pathlib import Path
import hashlib
import importlib.util
import json
import random
import sys
import types

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VENDOR = ROOT / 'research' / 'level_k_tradeoff_20261011' / 'vendor' / 'ts2vec'
REVISION = 'b0088e14a99706c05451316dc6db8d3da9351163'
RIDGE_ALPHAS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
_PREFIX = '_professor_direct_official_ts2vec'


def vendor_provenance():
    receipt = json.loads((VENDOR / 'provenance.json').read_text(encoding='utf-8-sig'))
    if receipt['commit'] != REVISION:
        raise RuntimeError('Official TS2Vec vendor revision changed')
    checks = []
    for item in receipt['files']:
        path = VENDOR / item['path']
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != item['sha256']:
            raise RuntimeError(f'Official source bytes changed: {item["path"]}')
        checks.append({**item, 'current_sha256': actual, 'matches': True})
    return {**receipt, 'files': checks, 'vendor_directory': str(VENDOR)}


def _load_module(name, path, package=False):
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)] if package else None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def official_class():
    name = _PREFIX + '.ts2vec'
    if name in sys.modules:
        return sys.modules[name].TS2Vec
    vendor_provenance()
    parent = types.ModuleType(_PREFIX)
    parent.__path__ = [str(VENDOR)]
    sys.modules[_PREFIX] = parent
    package = _load_module(_PREFIX + '.models', VENDOR / 'models' / '__init__.py', True)
    losses = _load_module(_PREFIX + '.models.losses', VENDOR / 'models' / 'losses.py')
    utils = _load_module(_PREFIX + '.utils', VENDOR / 'utils.py')
    # Upstream uses absolute imports. Temporarily bind those exact modules during
    # source execution, then restore every previous binding. No source is edited.
    replacements = {'models': package, 'models.losses': losses, 'utils': utils}
    prior = {key: sys.modules.get(key) for key in replacements}
    try:
        sys.modules.update(replacements)
        return _load_module(name, VENDOR / 'ts2vec.py').TS2Vec
    finally:
        for key, value in prior.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


def create_ts2(channels, dim, seed, device='cuda', after_iter_callback=None,
               batch_size=8, max_train_length=3000):
    if dim not in (16, 320):
        raise ValueError('This direct protocol declares only output_dims 16 and 320')
    # Match official utils.init_dl_program seed offsets without setting CUDA for
    # CPU checks. Algorithms/crop/mask/SWA stay in the original TS2Vec class.
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    if str(device).startswith('cuda'):
        torch.cuda.manual_seed(seed + 3)
    return official_class()(input_dims=channels, output_dims=dim, hidden_dims=64,
                            depth=10, device=device, lr=.001, batch_size=batch_size,
                            max_train_length=max_train_length, temporal_unit=0,
                            after_iter_callback=after_iter_callback)


def fit_default_iterations(train_data):
    return 200 if train_data.size <= 100000 else 600


def train_ts2(train_data, dim, seed, device='cuda', after_iter_callback=None,
              batch_size=8, max_train_length=3000):
    if train_data.ndim != 3:
        raise ValueError('Official fit expects [instance, contiguous time, channel]')
    model = create_ts2(train_data.shape[-1], dim, seed, device,
                       after_iter_callback, batch_size, max_train_length)
    losses = model.fit(train_data.copy(), verbose=False)
    return model, losses


def export_offline(model, past, batch_size=8):
    """[B,L,C] past-only windows -> [B,L,d], no rolling or extra future."""
    return model.encode(np.asarray(past, dtype=np.float32).copy(),
                        causal=False, sliding_length=None, batch_size=batch_size)


def export_last_causal(model, past, batch_size=256, padding=200):
    """Last tick of official causal=True, sliding_length=1, padding=200.

    Evaluate only that final roll instead of redundantly computing every earlier
    timestamp. The final official window contains the last 201 observed ticks,
    left-NaN-padded for short input, and its output slice is [200:201].
    """
    values = np.asarray(past, dtype=np.float32)
    if values.ndim != 3 or not values.shape[1]:
        raise ValueError('Causal native-head input must be nonempty [B,T,C]')
    length = padding + 1
    if values.shape[1] < length:
        values = np.pad(values, ((0, 0), (length-values.shape[1], 0), (0, 0)),
                        constant_values=np.nan)
    else:
        values = values[:, -length:]
    previous_mode = model.net.training
    model.net.eval()
    result = []
    try:
        with torch.no_grad():
            for start in range(0, len(values), batch_size):
                batch = torch.from_numpy(values[start:start+batch_size].copy())
                result.append(model._eval_with_pooling(
                    batch, slicing=slice(padding, padding+1)).numpy()[:, 0])
    finally:
        model.net.train(previous_mode)
    return np.concatenate(result, axis=0)


def freeze_encoder(model):
    model._net.eval()
    model.net.eval()
    for network in (model._net, model.net):
        for parameter in network.parameters():
            parameter.requires_grad_(False)


class ReconstructionDecoder(nn.Module):
    """New timestamp-wise reconstruction decoder; absent from original TS2Vec."""
    def __init__(self, dim, channels):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(),
                                    nn.Linear(128, channels))

    def forward(self, representation):
        return self.layers(representation)


def model_configuration(model):
    return {'input_dims': model._net.input_dims, 'output_dims': model._net.output_dims,
            'hidden_dims': model._net.hidden_dims,
            'depth': len(model._net.feature_extractor.net)-1,
            'lr': model.lr, 'batch_size': model.batch_size,
            'max_train_length': model.max_train_length,
            'temporal_unit': model.temporal_unit, 'n_iters': model.n_iters,
            'n_epochs': model.n_epochs,
            'online_parameters': sum(p.numel() for p in model._net.parameters()),
            'swa_parameters': sum(p.numel() for p in model.net.parameters()),
            'swa_n_averaged': int(model.net.n_averaged.item()),
            'online_trainable_parameters': sum(p.numel() for p in model._net.parameters() if p.requires_grad),
            'swa_trainable_flag_parameters': sum(p.numel() for p in model.net.parameters() if p.requires_grad),
            'source_revision': REVISION}
