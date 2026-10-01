"""Frozen selected parents plus one matched raw-history correction head."""
import copy
import sys
from pathlib import Path

import torch
from torch import nn

from runtime_v14 import HERE, ROOT, DATASETS, FAMILIES, artifact, digest, import_file, load_data, read_json

old = import_file(ROOT / 'research/tsfm_peft_internal_vs_subspace_v11_20260930/model_v11.py', 'v14_pinned_model_primitives')
tensor_hashes, state_digest = old.tensor_hashes, old.state_digest
SCHEMA = 'v14_frozen_parent_correction_1'


def parent_role(family):
    if family not in FAMILIES:
        raise ValueError('Unknown v14 family')
    return 'direct' if family == 'direct_gelu' else 'level'


def parent_reference(dataset, role, seed):
    row = copy.deepcopy(read_json(HERE / 'data_manifest.json')['datasets'][dataset]['parents'][role][str(seed)])
    if row['dataset'] != dataset or row['seed'] != seed:
        raise ValueError('Frozen parent identity differs')
    if row['family'] != ('direct' if role == 'direct' else 'pca_level'):
        raise ValueError('Only the original selected PCA_LEVEL/DIRECT parents are permitted')
    return row


def load_parent(dataset, role, seed, device='cuda'):
    row = parent_reference(dataset, role, seed)
    previous = ROOT / 'research/tsfm_peft_group_basis_v13_20261001'
    if str(previous) not in sys.path:
        sys.path.append(str(previous))
    factory = import_file(previous / 'cost_v13.py', 'v14_pinned_parent_factory')
    model, _ = factory.load_original(row, load_data(dataset), device)
    model.requires_grad_(False).eval()
    return model, row


def initial_g(dataset, seed):
    entry = read_json(HERE / 'data_manifest.json')['datasets'][dataset]['initial_g'][str(seed)]
    receipt = artifact(entry['path'], entry['sha256'])
    saved = torch.load(receipt['path'], map_location='cpu', weights_only=False)
    state = saved.get('state', saved)
    names = ('residual.down.weight', 'residual.up.weight')
    initial = {name: state[name].detach().cpu().clone() for name in names}
    if initial[names[0]].shape != (32, 512) or initial[names[1]].shape != (48, 32):
        raise ValueError('Original untrained G has unexpected dimensions')
    if torch.count_nonzero(initial[names[1]]):
        raise ValueError('New head must start with the original untrained zero up tensor')
    return initial, receipt


class CorrectionHead(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.down = nn.Linear(512, 32, bias=False)
        self.up = nn.Linear(32, 48, bias=False)
        self.activation = nn.GELU(approximate='none') if activation == 'gelu' else nn.Identity()

    def forward(self, centered):
        return self.up(self.activation(self.down(centered.transpose(1, 2)))).transpose(1, 2)


class CorrectionModel(nn.Module):
    def __init__(self, dataset, family, seed, basis, initial, parent=None,
                 initial_receipt=None, parent_receipt=None):
        super().__init__()
        if family not in FAMILIES:
            raise ValueError('Only the three frozen v14 families are permitted')
        self.dataset, self.family, self.seed = dataset, family, int(seed)
        self.channels, self.latent, _ = DATASETS[dataset]
        basis = torch.as_tensor(basis, dtype=torch.float32).cpu().clone()
        if basis.shape != (self.channels, self.latent):
            raise ValueError('Original PCA basis dimensions differ')
        if not torch.allclose(basis.T @ basis, torch.eye(self.latent), atol=1e-5, rtol=0):
            raise ValueError('Original PCA basis must be orthonormal')
        self.register_buffer('basis0', basis)
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(0)
            self.residual = CorrectionHead('linear' if family == 'level_linear' else 'gelu')
        with torch.no_grad():
            self.residual.down.weight.copy_(initial['residual.down.weight'])
            self.residual.up.weight.copy_(initial['residual.up.weight'])
        self.initial_receipt = copy.deepcopy(initial_receipt)
        self.parent_receipt = copy.deepcopy(parent_receipt)
        self._chunk_rows = None
        self.spec = {'dataset': dataset, 'family': family, 'seed': self.seed, 'channels': self.channels,
                     'latent': self.latent, 'context': 512, 'horizon': 48, 'parent_role': parent_role(family)}
        if parent is not None:
            self.parent = parent.requires_grad_(False).eval()

    @property
    def chunk_rows(self):
        return self._chunk_rows

    @chunk_rows.setter
    def chunk_rows(self, value):
        if value is not None and value <= 0:
            raise ValueError('Chunk rows must be positive')
        self._chunk_rows = value
        if hasattr(self, 'parent'):
            self.parent.chunk_rows = value

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'parent'):
            self.parent.eval()
        return self

    def parent_prediction(self, x):
        if not hasattr(self, 'parent'):
            raise RuntimeError('Cached head model has no online parent')
        with torch.no_grad():
            return self.parent(x).detach()

    def correction(self, x):
        return self.residual(x.detach() - x[:, -1:, :].detach())

    def forecast_from_parent(self, x, prediction):
        if tuple(x.shape[1:]) != (512, self.channels) or tuple(prediction.shape) != (len(x), 48, self.channels):
            raise ValueError('Expected B x 512 x C inputs and B x 48 x C parent predictions')
        return prediction.detach() + self.correction(x)

    def components(self, x):
        return self.parent_prediction(x), self.correction(x)

    def forward(self, x):
        return self.forecast_from_parent(x, self.parent_prediction(x))

    def adapter_state(self):
        return {name: value.detach().cpu().clone() for name, value in self.state_dict().items()
                if not name.startswith('parent.')}

    def restore_adapter(self, state):
        if state.keys() != self.adapter_state().keys():
            raise ValueError('V14 head checkpoint tensor names differ')
        self.load_state_dict(state, strict=False)

    def model_config(self):
        return {'schema': SCHEMA, **self.spec, 'activation': 'identity' if self.family == 'level_linear' else 'exact GELU',
                'head': 'shared bias-free 512-32-48', 'head_input': 'X minus last(X), no added level',
                'trainable_parameters': 17920, 'parent_frozen': True, 'precision': 'float32',
                'cache': 'TRAIN/VAL full parent predictions only; online parent once at deployment'}


def build_model(dataset, family, seed, device='cuda', load_parent=True):
    data = load_data(dataset)
    initial, receipt = initial_g(dataset, seed)
    role = parent_role(family)
    if load_parent:
        parent, parent_receipt = globals()['load_parent'](dataset, role, seed, device)
    else:
        parent, parent_receipt = None, parent_reference(dataset, role, seed)
    return CorrectionModel(dataset, family, seed, data['pca_basis'], initial, parent,
                           receipt, parent_receipt).to(device)


def make_checkpoint(model, spec, **extra):
    return {'schema': SCHEMA, 'spec': copy.deepcopy(spec), 'basis': model.basis0.detach().cpu().clone(),
            'state': model.adapter_state(), 'model_config': model.model_config(), 'parent': model.parent_receipt,
            'initial_g': model.initial_receipt, 'data_manifest_sha256': digest(HERE / 'data_manifest.json'), **extra}


def restore_model(checkpoint, device='cuda', load_parent=True):
    saved = torch.load(Path(checkpoint), map_location='cpu', weights_only=False)
    if saved['schema'] != SCHEMA or saved['data_manifest_sha256'] != digest(HERE / 'data_manifest.json'):
        raise ValueError('V14 checkpoint schema/data binding changed')
    spec = saved['spec']
    model = build_model(spec['dataset'], spec['family'], spec['seed'], device, load_parent)
    if model.initial_receipt != saved['initial_g'] or model.parent_receipt != saved['parent']:
        raise ValueError('Original initial head or selected frozen parent receipt changed')
    if not torch.equal(saved['basis'], model.basis0.cpu()):
        raise ValueError('Saved original PCA basis differs')
    model.restore_adapter(saved['state'])
    if model.model_config() != saved['model_config']:
        raise ValueError('Restored configuration differs')
    return model.eval()


def deployment_copy(model):
    if not hasattr(model, 'parent'):
        raise ValueError('Deployment must include the online frozen parent')
    return copy.deepcopy(model).eval()
