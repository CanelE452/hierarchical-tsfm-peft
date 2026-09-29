"""Approved fixed-PCA P/Q correction and parameter-matched RAW64 adapters."""
import copy
import hashlib
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
REVISION = '772f3d25d38aec6d914c8949dab4462e2d46f5d8'
SNAPSHOT = ROOT / '.cache/huggingface/models--amazon--chronos-bolt-small/snapshots' / REVISION
DATASETS = {'robin': (17, 5), 'jena': (21, 6)}
NEURAL_FAMILIES = ('pq_split', 'raw64')
REFERENCE_FAMILIES = ()
RETAINED_SEED_OFFSET = 1000003


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
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
    def __init__(self, rank=32):
        super().__init__()
        if rank not in (32, 64):
            raise ValueError('Only the approved ranks 32 and 64 are supported')
        self.down = nn.Linear(512, rank, bias=False)
        self.up = nn.Linear(rank, 48, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.down(x.transpose(1, 2))).transpose(1, 2)


def initial_edg(basis, seed):
    basis = torch.as_tensor(basis, dtype=torch.float32, device='cpu')
    if tuple(basis.shape) not in DATASETS.values():
        raise ValueError('Only approved C17/K5 or C21/K6 PCA shapes are supported')
    channels, latent = basis.shape
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(int(seed))
        nn.Linear(channels, latent, bias=False)
        nn.Linear(latent, channels, bias=False)
        residual_q = TemporalResidual(32)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(int(seed) + RETAINED_SEED_OFFSET)
        retained = TemporalResidual(32)
    return {'encoder.weight': basis.T.contiguous().clone(), 'decoder.weight': basis.clone(),
            'residual_q.down.weight': residual_q.down.weight.detach().clone(),
            'residual_q.up.weight': residual_q.up.weight.detach().clone(),
            'retained.down.weight': retained.down.weight.detach().clone(),
            'retained.up.weight': retained.up.weight.detach().clone()}


def initial_raw(initial):
    return {'encoder.weight': initial['encoder.weight'].clone(),
            'decoder.weight': initial['decoder.weight'].clone(),
            'residual.down.weight': torch.cat((initial['residual_q.down.weight'], initial['retained.down.weight']), dim=0),
            'residual.up.weight': torch.cat((initial['residual_q.up.weight'], initial['retained.up.weight']), dim=1)}


class FixedAdapter(nn.Module):
    def __init__(self, spec, basis, initial, backbone=None):
        super().__init__()
        self.spec = copy.deepcopy(spec)
        self.family = spec['family']
        self.dataset = spec['dataset']
        self.channels, self.latent = DATASETS[self.dataset]
        if self.family not in NEURAL_FAMILIES or spec['ed_mode'] != 'fixed_ed':
            raise ValueError('Only fixed-PCA pq_split and raw64 are approved')
        if (spec['channels'], spec['latent']) != (self.channels, self.latent):
            raise ValueError('Model shape differs from the approved dataset contract')
        basis = torch.as_tensor(basis, dtype=torch.float32, device='cpu')
        if tuple(basis.shape) != (self.channels, self.latent):
            raise ValueError('Basis shape differs from the dataset contract')
        required = {'encoder.weight', 'decoder.weight', 'residual_q.down.weight',
                    'residual_q.up.weight', 'retained.down.weight', 'retained.up.weight'}
        if set(initial) != required:
            raise ValueError('Initialization must contain the six common untrained P/Q tensors')
        if not torch.equal(initial['encoder.weight'], basis.T) or not torch.equal(initial['decoder.weight'], basis):
            raise ValueError('Initial encoder/decoder must equal the recorded TRAIN PCA basis')
        if any(torch.count_nonzero(initial[name]) for name in ('residual_q.up.weight', 'retained.up.weight')):
            raise ValueError('Both common output layers must begin at literal zero')
        if backbone is None:
            raise ValueError('Both v8 families require the pinned frozen backbone')
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.median = list(backbone.chronos_config.quantiles).index(0.5)
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(0)
            self.encoder = nn.Linear(self.channels, self.latent, bias=False)
            self.decoder = nn.Linear(self.latent, self.channels, bias=False)
            if self.family == 'pq_split':
                self.residual_q = TemporalResidual(32)
                self.retained = TemporalResidual(32)
            else:
                self.residual = TemporalResidual(64)
        self.restore_adapter(initial if self.family == 'pq_split' else initial_raw(initial))
        self.encoder.requires_grad_(False)
        self.decoder.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def backbone_point(self, x):
        batch, length, channels = x.shape
        output = self.backbone(context=x.transpose(1, 2).reshape(batch * channels, length))
        return output.quantile_preds[:, self.median, :48].reshape(batch, channels, 48).transpose(1, 2)

    def parts(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (512, self.channels):
            raise ValueError('Expected approved [B,512,C] input')
        with torch.no_grad():
            z = self.encoder(x)
            reconstruction = self.decoder(z)
            main = self.decoder(self.backbone_point(z))
        residual = x - reconstruction
        last = residual[:, -1:, :].detach()
        persistence = last.expand(-1, 48, -1)
        if self.family == 'pq_split':
            learned_q = self.residual_q(residual - last)
            learned_p = self.retained(reconstruction - reconstruction[:, -1:, :].detach())
            learned = learned_q + learned_p
        else:
            learned = self.residual(x - x[:, -1:, :].detach())
            learned_q = learned_p = None
        return {'main': main, 'persistence': persistence, 'learned': learned,
                'residual_history': residual, 'learned_q': learned_q, 'learned_p': learned_p}

    def components(self, x):
        parts = self.parts(x)
        return parts['main'], parts['persistence'] + parts['learned']

    def forward(self, x):
        main, correction = self.components(x)
        return main + correction

    def adapter_names(self):
        return {name for name, _ in self.named_parameters() if not name.startswith('backbone.')}

    def adapter_state(self):
        names = self.adapter_names()
        return {key: value.detach().cpu().clone() for key, value in self.state_dict().items() if key in names}

    def restore_adapter(self, state):
        if set(state) != self.adapter_names():
            raise ValueError('Checkpoint must contain exactly this family\'s adapter tensors')
        self.load_state_dict(state, strict=False)

    def model_config(self):
        split = self.family == 'pq_split'
        return {'schema_version': 1, 'dataset': self.dataset, 'family': self.family,
                'channels': self.channels, 'latent': self.latent, 'ed_mode': 'fixed_ed',
                'context': 512, 'horizon': 48, 'precision': 'float32',
                'G': 'two independent shared 512-32-48' if split else 'shared 512-64-48',
                'bias': False, 'activation': None, 'trainable_parameters': 35840,
                'retained_input': 'full original-channel PCA reconstruction difference' if split else None,
                'level_statistic': 'last_reconstruction_residual_detached',
                'input_normalization': 'frozen TRAIN standardization plus native Bolt scaling',
                'retained_initial_seed_offset': RETAINED_SEED_OFFSET,
                'raw_initialization': 'concatenate common Q/P down rows and zero up columns',
                'pretrained_revision': REVISION}


def build_with_backbone(spec, basis, initial, backbone):
    return FixedAdapter(spec, basis, initial, backbone)


def build_model(spec, basis, initial, device='cpu'):
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
        raise ValueError('Checkpoint basis differs from the initial data receipt')
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
