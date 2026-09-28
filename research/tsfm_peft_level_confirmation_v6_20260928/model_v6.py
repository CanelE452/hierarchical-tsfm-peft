"""Fixed Robin LEVEL comparisons; no checkpoint construction at import time."""
import copy
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
REVISION = '772f3d25d38aec6d914c8949dab4462e2d46f5d8'
SNAPSHOT = ROOT / '.cache/huggingface/models--amazon--chronos-bolt-small/snapshots' / REVISION
NEURAL_FAMILIES = ('level_res', 'old_res', 'level_raw', 'level_linear_res', 'lora')
REFERENCE_FAMILIES = ('f0', 'compress', 'level_only')


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def tensor_hashes(state):
    return {key: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for key, value in state.items()}


def state_digest(state):
    value = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        value.update(name.encode())
        value.update(str(tensor.dtype).encode())
        value.update(str(tuple(tensor.shape)).encode())
        value.update(tensor.numpy().tobytes())
    return value.hexdigest()


class TemporalResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Linear(512, 32, bias=False)
        self.up = nn.Linear(32, 48, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.down(x.transpose(1, 2))).transpose(1, 2)


def initial_edg(basis, seed):
    basis = torch.as_tensor(basis, dtype=torch.float32, device='cpu')
    if tuple(basis.shape) != (17, 5):
        raise ValueError('Approved Robin PCA shape is [17,5]')
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(int(seed))
        nn.Linear(17, 5, bias=False)
        nn.Linear(5, 17, bias=False)
        residual = TemporalResidual()
    return {'encoder.weight': basis.T.contiguous().clone(), 'decoder.weight': basis.clone(),
            'residual.down.weight': residual.down.weight.detach().clone(),
            'residual.up.weight': residual.up.weight.detach().clone()}


class FixedAdapter(nn.Module):
    def __init__(self, spec, basis, initial, backbone=None):
        super().__init__()
        self.spec = copy.deepcopy(spec)
        self.family = spec['family']
        self.channels, self.latent = 17, 5
        if self.family not in NEURAL_FAMILIES + REFERENCE_FAMILIES:
            raise ValueError(self.family)
        if spec.get('channels', 17) != 17 or spec.get('latent', 5) != 5 or spec.get('ed_mode') != 'fixed_ed':
            raise ValueError('Only approved Robin C17/K5 FIXED_ED is allowed')
        if np.shape(basis) != (17, 5):
            raise ValueError('Basis shape differs from the data contract')
        if self.family == 'level_linear_res':
            if backbone is not None:
                raise ValueError('Linear deployment must not retain a TSFM')
        else:
            if backbone is None:
                raise ValueError('TSFM family requires the pinned frozen backbone')
            self.backbone = backbone
            self.backbone.requires_grad_(False)
            self.median = list(backbone.chronos_config.quantiles).index(0.5)
            if self.family == 'lora':
                if str(ROOT / 'src') not in sys.path:
                    sys.path.insert(0, str(ROOT / 'src'))
                from hier_peft.lora import attach_lora
                self.backbone, _ = attach_lora(self.backbone)
            self.backbone.eval()
        if self.family not in ('f0', 'lora'):
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(0)
                self.encoder = nn.Linear(17, 5, bias=False)
                self.decoder = nn.Linear(5, 17, bias=False)
                if self.family not in ('compress', 'level_only'):
                    self.residual = TemporalResidual()
            names = {name for name, _ in self.named_parameters() if not name.startswith('backbone.')}
            self.load_state_dict({key: initial[key] for key in names}, strict=False)
            if hasattr(self, 'residual') and torch.count_nonzero(self.residual.up.weight):
                raise ValueError('G must begin at literal zero output')
            self.encoder.requires_grad_(False)
            self.decoder.requires_grad_(False)
            if self.family == 'level_linear_res':
                with torch.random.fork_rng(devices=[]):
                    torch.default_generator.manual_seed(int(spec['seed']) + 3_000_000)
                    self.temporal = nn.Linear(512, 48, bias=True, dtype=torch.float32)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'backbone'):
            self.backbone.eval()
        return self

    def backbone_point(self, x):
        batch, length, channels = x.shape
        output = self.backbone(context=x.transpose(1, 2).reshape(batch * channels, length))
        return output.quantile_preds[:, self.median, :48].reshape(batch, channels, 48).transpose(1, 2)

    def normalize_context(self, z):
        original_dtype = z.dtype
        z = z.float()
        loc = torch.nan_to_num(torch.nanmean(z, dim=-1, keepdim=True), nan=0.0)
        variance = (z - loc).square().nanmean(dim=-1, keepdim=True)
        zero_variance = variance == 0
        safe_variance = torch.where(zero_variance, torch.ones_like(variance), variance)
        scale = torch.nan_to_num(safe_variance.sqrt(), nan=1.0)
        scale = torch.where(zero_variance, 1e-5, scale)
        normalized = (z - loc) / scale
        normalized = torch.where(torch.isnan(z), 0.0, normalized)
        return normalized.to(original_dtype), loc, scale

    def temporal_point(self, z):
        normalized, loc, scale = self.normalize_context(z.transpose(1, 2))
        return (self.temporal(normalized).float() * scale + loc).transpose(1, 2)

    def parts(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (512, 17):
            raise ValueError('Expected input [batch,512,17]')
        if self.family in ('f0', 'lora'):
            main = self.backbone_point(x)
            return {'main': main, 'persistence': torch.zeros_like(main), 'learned': torch.zeros_like(main)}
        with torch.no_grad():
            z = self.encoder(x)
            reconstruction = self.decoder(z)
        if self.family == 'level_linear_res':
            main = self.decoder(self.temporal_point(z))
        else:
            with torch.no_grad():
                main = self.decoder(self.backbone_point(z))
        residual = x - reconstruction
        persistence = torch.zeros_like(main)
        correction_input = residual
        if self.family in ('level_res', 'level_raw', 'level_linear_res', 'level_only'):
            last = residual[:, -1:, :].detach()
            persistence = last.expand(-1, 48, -1)
            correction_input = (x - x[:, -1:, :].detach() if self.family == 'level_raw'
                                else residual - last)
        learned = self.residual(correction_input) if hasattr(self, 'residual') else torch.zeros_like(main)
        return {'main': main, 'persistence': persistence, 'learned': learned,
                'reconstruction': reconstruction, 'residual_history': residual, 'correction_input': correction_input}

    def components(self, x):
        parts = self.parts(x)
        return parts['main'], parts['persistence'] + parts['learned']

    def forward(self, x):
        main, correction = self.components(x)
        return main + correction

    def native_loss(self, x, target, mask):
        if self.family != 'lora':
            raise ValueError('Native quantile objective is reserved for LoRA')
        batch, length, channels = x.shape
        return self.backbone(context=x.transpose(1, 2).reshape(batch * channels, length),
                             target=target.transpose(1, 2).reshape(batch * channels, 48).clone(),
                             target_mask=mask.transpose(1, 2).reshape(batch * channels, 48)).loss

    def adapter_names(self):
        return {name for name, parameter in self.named_parameters()
                if not name.startswith('backbone.') or parameter.requires_grad}

    def adapter_state(self):
        names = self.adapter_names()
        return {key: value.detach().cpu().clone() for key, value in self.state_dict().items() if key in names}

    def restore_adapter(self, state):
        if set(state) != self.adapter_names():
            raise ValueError('Checkpoint must contain every adapter, including T, and only those tensors')
        self.load_state_dict(state, strict=False)

    def model_config(self):
        return {'schema_version': 1, 'family': self.family, 'channels': 17, 'latent': 5,
                'ed_mode': 'fixed_ed', 'context': 512, 'horizon': 48, 'precision': 'float32',
                'G': '512-32-48,bias=False,activation=None' if hasattr(self, 'residual') else None,
                'level_statistic': 'last_input_value_detached' if self.family.startswith('level_') else None,
                'temporal_normalization': 'context_mean_population_std_no_detach_eps1e-5' if hasattr(self, 'temporal') else None,
                'temporal_seed': int(self.spec['seed']) + 3_000_000 if hasattr(self, 'temporal') else None,
                'pretrained_revision': None if self.family == 'level_linear_res' else REVISION}


def build_with_backbone(spec, basis, initial, backbone):
    return FixedAdapter(spec, basis, initial, backbone)


def build_model(spec, basis, initial, device='cpu'):
    backbone = None
    if spec['family'] != 'level_linear_res':
        from chronos import ChronosBoltPipeline
        backbone = ChronosBoltPipeline.from_pretrained(str(SNAPSHOT), device_map='cpu', torch_dtype=torch.float32).model
    return FixedAdapter(spec, basis, initial, backbone).to(device)


def restore_model(checkpoint_path, device='cpu'):
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if file_hash(saved['initial_path']) != saved['initial_sha256']:
        raise ValueError('Shared initialization file changed')
    initial = torch.load(saved['initial_path'], map_location='cpu', weights_only=False)
    if tensor_hashes(initial['state']) != initial['tensor_hashes']:
        raise ValueError('Shared initialization tensors changed')
    if not torch.equal(torch.as_tensor(saved['basis']), torch.as_tensor(initial['basis'])):
        raise ValueError('Checkpoint basis differs from initial PCA')
    model = build_model(saved['spec'], saved['basis'], initial['state'], device=device)
    model.restore_adapter(saved['state'])
    if saved['model_config'] != model.model_config():
        raise ValueError('Checkpoint configuration changed')
    return model.eval()


def macro_loss(pred, target, mask):
    if pred.shape != target.shape or pred.shape != mask.shape or pred.ndim != 3:
        raise ValueError('Prediction/target/mask must share [batch,H,C] axes')
    observed = mask.bool()
    count = observed.sum((0, 1))
    valid = count > 0
    if not bool(valid.any()):
        raise ValueError('No observed target in batch')
    error = torch.where(observed, pred - target, torch.zeros_like(pred))
    return (error.square().sum((0, 1))[valid] / count[valid]).mean()
