"""Bounded v5 modifications; construction never starts training or evaluation."""
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / '.cache/huggingface/models--amazon--chronos-bolt-small/snapshots/772f3d25d38aec6d914c8949dab4462e2d46f5d8'
REVISION = '772f3d25d38aec6d914c8949dab4462e2d46f5d8'
DATASETS = {'hog': (32, 8, 'fixed_ed'), 'bull': (16, 4, 'fixed_ed'),
            'electricity': (32, 8, 'current')}


def tensor_hashes(state):
    return {key: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for key, value in state.items()}


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def state_digest(state):
    value = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        value.update(key.encode())
        value.update(str(tensor.dtype).encode())
        value.update(str(tuple(tensor.shape)).encode())
        value.update(tensor.numpy().tobytes())
    return value.hexdigest()


def nested_basis(train, old_basis, new_rank):
    """Keep existing PCA axes/signs exactly and add TRAIN-only orthogonal axes."""
    old = np.asarray(old_basis, dtype=np.float64)
    if old.ndim != 2 or not old.shape[1] < new_rank <= old.shape[0]:
        raise ValueError('Nested PCA requires old K < new K <= C')
    if not np.allclose(old.T @ old, np.eye(old.shape[1]), atol=1e-5, rtol=0):
        raise ValueError('Parent PCA columns are not orthonormal')
    x = np.asarray(train, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != old.shape[0] or not np.isfinite(x).all():
        raise ValueError('Nested PCA requires finite standardized TRAIN rows')
    x = x - x.mean(0, keepdims=True)
    # Complete QR supplies a basis of the complement without changing saved axes.
    complement = np.linalg.qr(old, mode='complete')[0][:, old.shape[1]:]
    residual_coordinates = x @ complement
    _, vectors = np.linalg.eigh(residual_coordinates.T @ residual_coordinates / len(x))
    added = complement @ vectors[:, -(new_rank-old.shape[1]):][:, ::-1]
    for column in range(added.shape[1]):
        if added[np.argmax(np.abs(added[:, column])), column] < 0:
            added[:, column] *= -1
    result = np.concatenate([np.asarray(old_basis, dtype=np.float32), added.astype(np.float32)], axis=1)
    if not np.array_equal(result[:, :old.shape[1]], np.asarray(old_basis, dtype=np.float32)):
        raise AssertionError('Nested PCA changed the original columns')
    if not np.allclose(result.T @ result, np.eye(new_rank), atol=1e-5, rtol=0):
        raise AssertionError('Nested PCA is not orthonormal')
    return result


class TemporalResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Linear(512, 32, bias=False)
        self.up = nn.Linear(32, 48, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.down(x.transpose(1, 2))).transpose(1, 2)


class FullTemporalResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(512, 48, bias=False)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        return self.linear(x.transpose(1, 2)).transpose(1, 2)


def initial_state(basis, seed, method, base_state=None, base_latent=None):
    if method not in ('level', 'full', 'k'):
        raise ValueError(method)
    basis = torch.as_tensor(basis, dtype=torch.float32, device='cpu')
    channels, latent = basis.shape
    # Always consume the base E/D construction stream before G; K expansion
    # cannot silently alter its initial G weights.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(int(seed))
        nn.Linear(channels, base_latent or latent, bias=False)
        nn.Linear(base_latent or latent, channels, bias=False)
        residual = TemporalResidual()
    down = residual.down.weight.detach().clone()
    up = residual.up.weight.detach().clone()
    if base_state is not None:
        down = torch.as_tensor(base_state['residual.down.weight']).detach().cpu().float().clone()
        up = torch.as_tensor(base_state['residual.up.weight']).detach().cpu().float().clone()
    if tuple(down.shape) != (32, 512) or tuple(up.shape) != (48, 32) or torch.count_nonzero(up):
        raise ValueError('G must be an untrained rank32, zero-output state')
    result = {'encoder.weight': basis.T.contiguous().clone(), 'decoder.weight': basis.clone()}
    if method == 'full':
        result['residual.linear.weight'] = up @ down
    else:
        result.update({'residual.down.weight': down, 'residual.up.weight': up})
    return result


class RecoveryAdapter(nn.Module):
    def __init__(self, backbone, spec, basis, initial):
        super().__init__()
        self.spec = dict(spec)
        self.method = spec['method']
        self.variant = spec['variant']
        self.ed_mode = spec['ed_mode']
        if self.method not in ('level', 'full', 'k') or self.variant not in ('res', 'raw'):
            raise ValueError('Only the approved single modifications and matched variants are allowed')
        if self.ed_mode not in ('current', 'fixed_ed'):
            raise ValueError(self.ed_mode)
        channels, latent = np.shape(basis)
        if (channels, latent) != (int(spec['channels']), int(spec['latent'])):
            raise ValueError('Basis/spec shape mismatch')
        self.channels, self.latent = channels, latent
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.median = list(backbone.chronos_config.quantiles).index(0.5)
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(0)
            self.encoder = nn.Linear(channels, latent, bias=False)
            self.decoder = nn.Linear(latent, channels, bias=False)
            self.residual = FullTemporalResidual() if self.method == 'full' else TemporalResidual()
        self.restore_adapter(initial)
        if self.ed_mode == 'fixed_ed':
            self.encoder.requires_grad_(False)
            self.decoder.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def backbone_point(self, x):
        batch, length, channels = x.shape
        out = self.backbone(context=x.transpose(1, 2).reshape(batch * channels, length))
        return out.quantile_preds[:, self.median, :48].reshape(batch, channels, 48).transpose(1, 2)

    def parts(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (512, self.channels):
            raise ValueError('Expected [batch, 512, C] input')
        if self.ed_mode == 'fixed_ed':
            with torch.no_grad():
                z = self.encoder(x)
                main = self.decoder(self.backbone_point(z))
                reconstruction = self.decoder(z)
        else:
            z = self.encoder(x)
            main = self.decoder(self.backbone_point(z))
            reconstruction = self.decoder(z)
        residual = x - reconstruction
        if self.method == 'level':
            residual_last = residual[:, -1:, :].detach()
            correction_input = (residual - residual_last if self.variant == 'res'
                                else x - x[:, -1:, :].detach())
            persistence = residual_last.expand(-1, 48, -1)
        else:
            correction_input = residual if self.variant == 'res' else x
            persistence = torch.zeros_like(main)
        learned = self.residual(correction_input)
        return {'main': main, 'reconstruction': reconstruction, 'residual_history': residual,
                'correction_input': correction_input, 'persistence': persistence, 'learned': learned}

    def components(self, x):
        parts = self.parts(x)
        return parts['main'], parts['persistence'] + parts['learned']

    def forward(self, x):
        main, correction = self.components(x)
        return main + correction

    def adapter_names(self):
        return {name for name, _ in self.named_parameters() if not name.startswith('backbone.')}

    def adapter_state(self):
        return {name: value.detach().cpu().clone() for name, value in self.state_dict().items()
                if name in self.adapter_names()}

    def restore_adapter(self, state):
        if set(state) != self.adapter_names():
            raise ValueError('Adapter checkpoint names do not match the selected modification')
        self.load_state_dict(state, strict=False)

    def model_config(self):
        return {'schema_version': 1, 'model': 'RecoveryAdapter', 'method': self.method,
                'variant': self.variant, 'ed_mode': self.ed_mode, 'channels': self.channels,
                'latent': self.latent, 'context': 512, 'horizon': 48, 'precision': 'float32',
                'G': 'Linear(512,48,bias=False)' if self.method == 'full' else '512-32-48,bias=False,activation=None',
                'level_statistic': 'last_input_value' if self.method == 'level' else None,
                'level_detach': self.method == 'level', 'pretrained_revision': REVISION}


def build_with_backbone(spec, basis, initial, backbone):
    return RecoveryAdapter(backbone, spec, basis, initial)


def build_model(spec, basis, initial, device='cpu'):
    from chronos import ChronosBoltPipeline
    backbone = ChronosBoltPipeline.from_pretrained(str(SNAPSHOT), device_map='cpu', torch_dtype=torch.float32).model
    return build_with_backbone(spec, basis, initial, backbone).to(device)


def restore_model(checkpoint_path, device='cpu'):
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if file_hash(saved['initial_path']) != saved['initial_sha256']:
        raise ValueError('Shared initialization file changed')
    initial = torch.load(saved['initial_path'], map_location='cpu', weights_only=False)
    if tensor_hashes(initial['state']) != initial['tensor_hashes']:
        raise ValueError('Shared initialization tensors changed')
    if not torch.equal(torch.as_tensor(saved['basis']), torch.as_tensor(initial['basis'])):
        raise ValueError('Saved basis differs from shared initialization')
    model = build_model(saved['spec'], saved['basis'], initial['state'], device=device)
    model.restore_adapter(saved['state'])
    if model.model_config() != saved['model_config']:
        raise ValueError('Saved configuration changed')
    return model.eval()


def macro_loss(pred, target, mask):
    if pred.shape != target.shape or pred.shape != mask.shape or pred.ndim != 3:
        raise ValueError('prediction, target and mask must share [batch,H,C] shape')
    valid_mask = mask.bool()
    count = valid_mask.sum((0, 1))
    valid_channels = count > 0
    if not bool(valid_channels.any()):
        raise ValueError('No observed target in batch')
    error = torch.where(valid_mask, pred - target, torch.zeros_like(pred))
    return (error.square().sum((0, 1))[valid_channels] / count[valid_channels]).mean()
