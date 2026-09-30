"""Frozen F0 over a fixed group basis, with the prescribed shared linear G."""
import copy
from pathlib import Path

import torch
from torch import nn

from runtime_v13 import HERE, ROOT, DATASETS, artifact, digest, import_file, load_data, read_json

old = import_file(ROOT / 'research/tsfm_peft_internal_vs_subspace_v11_20260930/model_v11.py', 'v13_pinned_model_primitives')
REVISION, SNAPSHOT = old.REVISION, old.SNAPSHOT
SCHEMA = 'v13_group_basis_model_1'
FAMILIES = ('group_res', 'group_raw')
tensor_hashes, state_digest = old.tensor_hashes, old.state_digest


def initial_g(dataset, seed):
    entry = read_json(HERE / 'data_manifest.json')['datasets'][dataset]['initial_g'][str(seed)]
    receipt = artifact(entry['path'], entry['sha256'])
    saved = torch.load(receipt['path'], map_location='cpu', weights_only=False)
    state = saved.get('state', saved)
    names = ('residual.down.weight', 'residual.up.weight')
    result = {name: state[name].detach().cpu().clone() for name in names}
    if result[names[0]].shape != (32, 512) or result[names[1]].shape != (48, 32):
        raise ValueError('Original LEVEL G has unexpected dimensions')
    if torch.count_nonzero(result[names[1]]):
        raise ValueError('Only original untrained, zero-output LEVEL G is allowed')
    return result, receipt


class GroupModel(nn.Module):
    def __init__(self, dataset, family, seed, basis, initial, backbone=None, initial_receipt=None):
        super().__init__()
        if family not in FAMILIES:
            raise ValueError('Only the two frozen v13 families are allowed')
        self.dataset, self.family, self.seed = dataset, family, int(seed)
        self.channels, self.latent, _ = DATASETS[dataset]
        basis = torch.as_tensor(basis, dtype=torch.float32).cpu().clone()
        if basis.shape != (self.channels, self.latent):
            raise ValueError('Group basis dimensions differ from the protocol')
        if not torch.allclose(basis.T @ basis, torch.eye(self.latent), atol=1e-5, rtol=0):
            raise ValueError('Group basis is not orthonormal')
        if (basis < 0).any() or not torch.all((basis > 0).sum(dim=1) == 1):
            raise ValueError('Basis must have disjoint nonnegative groups covering every channel')
        self.register_buffer('basis0', basis)
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(0)
            self.encoder = nn.Linear(self.channels, self.latent, bias=False)
            self.decoder = nn.Linear(self.latent, self.channels, bias=False)
            self.residual = old.TemporalResidual()
        with torch.no_grad():
            self.encoder.weight.copy_(basis.T)
            self.decoder.weight.copy_(basis)
            self.residual.down.weight.copy_(initial['residual.down.weight'])
            self.residual.up.weight.copy_(initial['residual.up.weight'])
        self.encoder.requires_grad_(False)
        self.decoder.requires_grad_(False)
        self.initial_receipt = copy.deepcopy(initial_receipt)
        self.chunk_rows = None
        self.spec = {'dataset': dataset, 'family': family, 'seed': self.seed,
                     'channels': self.channels, 'latent': self.latent, 'context': 512, 'horizon': 48}
        if backbone is not None:
            old.install_finite_constant_backward(backbone)
            self.backbone = backbone.requires_grad_(False).eval()
            self.median = list(backbone.chronos_config.quantiles).index(.5)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'backbone'):
            self.backbone.eval()
        return self

    def backbone_point(self, z):
        if not hasattr(self, 'backbone'):
            raise RuntimeError('Cached training model has no online backbone')
        batch, length, latent = z.shape
        rows = z.transpose(1, 2).reshape(batch * latent, length)
        if self.chunk_rows is not None and (self.chunk_rows <= 0 or torch.is_grad_enabled()):
            raise ValueError('Row chunks require positive size and inference context')
        chunks = rows.split(self.chunk_rows) if self.chunk_rows else (rows,)
        points = []
        for chunk in chunks:
            output = self.backbone(context=chunk)
            points.append(output.quantile_preds[:, self.median, :48].clone(memory_format=torch.contiguous_format))
            del output
        return torch.cat(points).reshape(batch, latent, 48).transpose(1, 2)

    def main_prediction(self, x):
        with torch.no_grad():
            return self.decoder(self.backbone_point(self.encoder(x)))

    def correction(self, x):
        with torch.no_grad():
            residual = x - self.decoder(self.encoder(x))
            last = residual[:, -1:, :].detach()
            centered = residual - last if self.family == 'group_res' else x - x[:, -1:, :].detach()
        return last.expand(-1, 48, -1) + self.residual(centered)

    def forecast_from_main(self, x, main):
        if tuple(x.shape[1:]) != (512, self.channels) or tuple(main.shape) != (len(x), 48, self.channels):
            raise ValueError('Cached input/main axes differ')
        return main.detach() + self.correction(x)

    def components(self, x):
        if tuple(x.shape[1:]) != (512, self.channels):
            raise ValueError('Expected B x 512 x C input')
        return self.main_prediction(x), self.correction(x)

    def forward(self, x):
        main, correction = self.components(x)
        return main + correction

    def adapter_state(self):
        return {name: value.detach().cpu().clone() for name, value in self.state_dict().items()
                if not name.startswith('backbone.')}

    def restore_adapter(self, state):
        if state.keys() != self.adapter_state().keys():
            raise ValueError('V13 checkpoint tensor names differ')
        self.load_state_dict(state, strict=False)

    def model_config(self):
        return {'schema': SCHEMA, **self.spec, 'basis': 'fixed disjoint normalized group indicators',
                'G': 'shared bias-free activation-free 512-32-48', 'trainable_parameters': 17920,
                'pretrained_revision': REVISION, 'backbone_frozen': True, 'precision': 'float32',
                'cache': 'training only; deployment invokes online F0'}


def _backbone():
    from chronos import ChronosBoltPipeline
    return ChronosBoltPipeline.from_pretrained(str(SNAPSHOT), device_map='cpu', torch_dtype=torch.float32).model


def build_model(dataset, family, seed, device='cuda', load_backbone=True):
    data = load_data(dataset, include_test=False)
    initial, receipt = initial_g(dataset, seed)
    return GroupModel(dataset, family, seed, data['basis'], initial,
                      _backbone() if load_backbone else None, receipt).to(device)


def make_checkpoint(model, spec, **extra):
    return {'schema': SCHEMA, 'spec': copy.deepcopy(spec), 'basis': model.basis0.detach().cpu().clone(),
            'state': model.adapter_state(), 'model_config': model.model_config(),
            'initial_g': model.initial_receipt, 'data_manifest_sha256': digest(HERE / 'data_manifest.json'), **extra}


def restore_model(checkpoint, device='cuda', load_backbone=True):
    saved = torch.load(Path(checkpoint), map_location='cpu', weights_only=False)
    if saved['schema'] != SCHEMA or saved['data_manifest_sha256'] != digest(HERE / 'data_manifest.json'):
        raise ValueError('V13 checkpoint schema/data binding changed')
    spec = saved['spec']
    initial, receipt = initial_g(spec['dataset'], spec['seed'])
    if receipt != saved['initial_g']:
        raise ValueError('Original untrained G receipt changed')
    data = load_data(spec['dataset'], include_test=False)
    if not torch.equal(saved['basis'], torch.as_tensor(data['basis'])):
        raise ValueError('Saved group basis differs from the data contract')
    model = GroupModel(spec['dataset'], spec['family'], spec['seed'], saved['basis'], initial,
                       _backbone() if load_backbone else None, receipt).to(device)
    model.restore_adapter(saved['state'])
    if model.model_config() != saved['model_config']:
        raise ValueError('Restored model configuration differs')
    return model.eval()


def deployment_copy(model):
    if not hasattr(model, 'backbone'):
        raise ValueError('A deployment copy must retain online F0')
    return copy.deepcopy(model).eval()
