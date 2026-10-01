"""Frozen selected DIRECT and F0 with a fitted channel decoder."""
import copy
import sys

import numpy as np
import torch
from torch import nn

from runtime_v15 import HERE, ROOT, DATASETS, SEEDS, artifact, import_file, load_data, read_json

FAMILIES = ('raw_free', 'pca_free', 'raw_tied', 'raw_interp', 'pca_tied')


def reference_row(dataset, family, seed=None):
    rows = read_json(HERE / 'reuse_manifest.json')['datasets'][dataset]['references']
    found = [row for row in rows if row['family'] == family and
             (family == 'f0' or row['seed'] == seed)]
    if len(found) != 1:
        raise ValueError('Expected one exact frozen reference: ' + repr((dataset, family, seed)))
    return copy.deepcopy(found[0])


def reference_api():
    previous = ROOT / 'research/tsfm_peft_group_basis_v13_20261001'
    if str(previous) not in sys.path:
        sys.path.append(str(previous))
    factory = import_file(previous / 'cost_v13.py', 'v15_reference_factory')
    factory.previous_api()
    return factory


def load_reference(dataset, family, seed=None, device='cuda'):
    row = reference_row(dataset, family, seed)
    for name in ('checkpoint', 'source_evaluation'):
        if row.get(name):
            artifact(row[name]['path'], row[name]['sha256'])
    factory = reference_api()
    model, deploy = factory.load_original(row, load_data(dataset), device)
    model.requires_grad_(False).eval()
    return model, deploy, row


def load_direct(dataset, seed, device='cuda'):
    model, _, row = load_reference(dataset, 'direct', seed, device)
    return model, row


def load_f0(dataset, device='cuda'):
    model, _, row = load_reference(dataset, 'f0', None, device)
    if not callable(getattr(model, 'backbone_point', None)):
        raise TypeError('Pinned F0 wrapper must expose the native median/crop point path')
    return model, row


def selector(dataset, family):
    if family not in FAMILIES:
        raise ValueError('Unknown fixed v15 family')
    value = read_json(HERE / 'selector_manifest.json')['datasets'][dataset]
    c, k, _ = DATASETS[dataset]
    if (value['C'], value['K']) != (c, k):
        raise ValueError('Selector dimensions changed')
    space = 'raw' if family.startswith('raw_') else 'pca'
    e = np.asarray(value['E_' + space], dtype=np.float32)
    if e.shape != (c, k) or not np.isfinite(e).all():
        raise ValueError('Invalid fixed selector')
    if family.endswith('_tied'):
        w = e.T.copy()
    elif family == 'raw_interp':
        if value['W_interp'] is None:
            raise ValueError('The fixed interpolation control is unavailable; do not reselect coordinates')
        w = np.asarray(value['W_interp'], dtype=np.float32)
    else:
        w = None
    return e, w


class SelectorDecoder(nn.Module):
    def __init__(self, direct, backbone, E, W, *, dataset=None, family=None, seed=None, basis=None, sources=None):
        super().__init__()
        e = torch.as_tensor(E, dtype=torch.float32, device='cpu').clone()
        w = torch.as_tensor(W, dtype=torch.float32, device='cpu').clone()
        if e.ndim != 2 or w.shape != (e.shape[1], e.shape[0]):
            raise ValueError('Expected E[C,K] and W[K,C]')
        if not torch.isfinite(e).all() or not torch.isfinite(w).all():
            raise ValueError('Nonfinite encoder or fitted decoder')
        self.channels, self.latent = e.shape
        self.zero_decoder = bool(torch.count_nonzero(w) == 0)
        if self.zero_decoder and backbone is not None:
            raise ValueError('Exact parent fallback must not retain an unused TSFM')
        if not self.zero_decoder and backbone is None:
            raise ValueError('A nonzero decoder requires the frozen point forecaster')
        self.direct = direct.requires_grad_(False).eval()
        self.backbone = backbone
        if isinstance(backbone, nn.Module):
            backbone.requires_grad_(False).eval()
        self.register_buffer('E', e)
        self.register_buffer('W', w)
        if basis is not None:
            self.register_buffer('basis0', torch.as_tensor(basis, dtype=torch.float32).clone())
        self.dataset, self.family, self.seed = dataset, family, seed
        self.spec = {'dataset': dataset, 'family': family, 'seed': seed}
        self.sources = copy.deepcopy(sources or {})
        self._chunk_rows = None
        self.deployed = True
        self.requires_grad_(False)

    @property
    def chunk_rows(self):
        return self._chunk_rows

    @chunk_rows.setter
    def chunk_rows(self, value):
        if value is not None and (int(value) != value or value <= 0):
            raise ValueError('chunk_rows must be a positive integer or None')
        self._chunk_rows = None if value is None else int(value)
        if self.backbone is not None:
            if not hasattr(self.backbone, 'chunk_rows') and value is not None:
                raise TypeError('Point callable does not expose row chunking')
            self.backbone.chunk_rows = self._chunk_rows

    def train(self, mode=True):
        super().train(mode)
        self.direct.eval()
        if isinstance(self.backbone, nn.Module):
            self.backbone.eval()
        return self

    def latent_point(self, x):
        if self.backbone is None:
            raise ValueError('A parent-only deployment has no TSFM')
        z = x @ self.E
        point = getattr(self.backbone, 'backbone_point', self.backbone)
        return point(z)

    def forecast_from_cached(self, direct_prediction, latent_prediction=None):
        if direct_prediction.ndim != 3 or tuple(direct_prediction.shape[1:]) != (48, self.channels):
            raise ValueError('Expected complete DIRECT output [B,48,C]')
        if self.zero_decoder:
            return direct_prediction
        if latent_prediction is None or tuple(latent_prediction.shape) != (len(direct_prediction), 48, self.latent):
            raise ValueError('Expected latent point output [B,48,K]')
        return direct_prediction + (latent_prediction - direct_prediction @ self.E) @ self.W

    def forward(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (512, self.channels):
            raise ValueError('Expected full original-channel input [B,512,C]')
        direct = self.direct(x)
        return self.forecast_from_cached(direct, None if self.zero_decoder else self.latent_point(x))


def build_model(dataset, family, seed, W=None, device='cuda'):
    if dataset not in DATASETS or seed not in SEEDS:
        raise ValueError('Dataset/seed outside the fixed v15 contract')
    e, fixed_w = selector(dataset, family)
    if fixed_w is not None:
        if W is not None and not np.array_equal(np.asarray(W, dtype=np.float32), fixed_w):
            raise ValueError('A fit-zero control cannot change its fixed decoder')
        W = fixed_w
    elif W is None:
        raise ValueError('A free-decoder deployment requires its selected FP32 W')
    w = np.asarray(W, dtype=np.float32)
    if w.shape != (e.shape[1], e.shape[0]) or not np.isfinite(w).all():
        raise ValueError('Invalid selected decoder')
    direct, direct_row = load_direct(dataset, seed, device)
    backbone, f0_row = (None, None) if not np.count_nonzero(w) else load_f0(dataset, device)
    basis = read_json(HERE / 'selector_manifest.json')['datasets'][dataset]['E_pca']
    model = SelectorDecoder(direct, backbone, e, w, dataset=dataset, family=family, seed=seed,
                            basis=basis, sources={'direct': direct_row, 'f0': f0_row})
    return model.to(device).eval()


def deployment_copy(model):
    return copy.deepcopy(model).requires_grad_(False).eval()


def parity(actual, expected, atol=1e-5, rtol=1e-4):
    a = torch.as_tensor(actual).detach().cpu().double()
    b = torch.as_tensor(expected).detach().cpu().double()
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Parity requires same-shape finite complete outputs')
    difference = (a - b).abs()
    bad = difference > atol + rtol * b.abs()
    return {'passed': not bool(bad.any()), 'max_absolute': float(difference.max()),
            'max_relative': float((difference / b.abs().clamp_min(1e-12)).max()),
            'relative_denominator_floor': 1e-12,
            'violating_elements': int(bad.sum()), 'elements': a.numel(), 'atol': atol, 'rtol': rtol}


@torch.no_grad()
def check_cached_online(model, x, direct_prediction, latent_prediction=None):
    return parity(model(x), model.forecast_from_cached(direct_prediction, latent_prediction))


@torch.no_grad()
def check_chunk(model, x, chunk_rows):
    previous = model.chunk_rows
    try:
        model.chunk_rows = None
        full = model(x)
        model.chunk_rows = chunk_rows
        return parity(model(x), full)
    finally:
        model.chunk_rows = previous


@torch.no_grad()
def check_zero(model, x):
    if not model.zero_decoder or model.backbone is not None:
        raise ValueError('Zero check requires actual no-TSFM deployment')
    return parity(model(x), model.direct(x), atol=0, rtol=0)
