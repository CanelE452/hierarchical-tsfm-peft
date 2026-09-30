"""Approved v11 internal LoRA and tied orthogonal-subspace models.

Importing this module does not load a backbone, run a model, or create files.
"""
import copy
import hashlib
import importlib.util
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from torch import nn
from torch.nn.utils.parametrizations import orthogonal

ROOT = Path(__file__).resolve().parents[2]
REVISION = '772f3d25d38aec6d914c8949dab4462e2d46f5d8'
SNAPSHOT = ROOT / '.cache/huggingface/models--amazon--chronos-bolt-small/snapshots' / REVISION
DATASETS = {'robin': (17, 5), 'jena': (21, 6), 'hog': (32, 8)}
LORA_FAMILIES = ('a_p_lora_qfixed', 'full_lora_mse', 'lora_native')
B_FAMILIES = ('b_fixed_u', 'b_learned_u', 'b_fixed_gamma', 'b_learned_gamma')
FAMILIES = LORA_FAMILIES + B_FAMILIES + ('direct_nlinear', 'base_level', 'f0')


def stable_instance_norm_forward(self, x, loc_scale=None):
    original_dtype = x.dtype
    x = x.to(torch.float32)
    if loc_scale is None:
        loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
        variance = (x - loc).square().nanmean(dim=-1, keepdim=True)
        zero = variance == 0
        # Native sqrt(0) followed by where produces 0*inf in backward.
        # Select a finite sqrt input on this constant branch before sqrt;
        # retain the native epsilon and every forward output value.
        safe_variance = torch.where(zero, torch.ones_like(variance), variance)
        scale = torch.nan_to_num(safe_variance.sqrt(), nan=1.0)
        scale = torch.where(zero, self.eps, scale)
    else:
        loc, scale = loc_scale
    scaled = (x - loc) / scale
    if self.use_arcsinh:
        scaled = torch.arcsinh(scaled)
    return scaled.to(original_dtype), (loc, scale)


def install_finite_constant_backward(backbone):
    if hasattr(backbone, 'instance_norm'):
        backbone.instance_norm.forward = MethodType(stable_instance_norm_forward, backbone.instance_norm)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_hashes(state):
    return {name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, value in state.items()}


def state_digest(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _attach_lora(backbone):
    path = ROOT / 'src/hier_peft/lora.py'
    spec = importlib.util.spec_from_file_location('v11_pinned_lora_source', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.attach_lora(backbone)


def _source_checkpoint(sources, role):
    key = role + '_checkpoint'
    if not sources.get(key):
        raise ValueError(f'Missing frozen source: {key}')
    path = Path(sources[key]).resolve()
    actual_hash = file_hash(path)
    expected_hash = sources.get(role + '_sha256') or sources.get(key + '_sha256')
    if expected_hash is not None and actual_hash != expected_hash:
        raise ValueError(f'Frozen source hash differs: {path}')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    return saved, {'path': str(path), 'sha256': actual_hash}


def _check_dataset(saved, dataset, seed=None):
    if saved.get('spec', {}).get('dataset') != dataset:
        raise ValueError('Frozen source dataset differs from the requested dataset')
    if seed is not None and int(saved['spec'].get('seed', -1)) != int(seed):
        raise ValueError('Frozen source seed differs from the requested paired seed')
    revision = saved.get('model_config', {}).get('pretrained_revision')
    if revision not in (None, REVISION):
        raise ValueError('Frozen source uses another backbone revision')


class TemporalResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Linear(512, 32, bias=False)
        self.up = nn.Linear(32, 48, bias=False)

    def forward(self, x):
        return self.up(self.down(x.transpose(1, 2))).transpose(1, 2)


class DirectNLinear(nn.Module):
    def __init__(self, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(int(seed))
            self.temporal = nn.Linear(512, 48, bias=True, dtype=torch.float32)

    def forward(self, x):
        last = x[:, -1:, :].detach()
        return self.temporal((x - last).transpose(1, 2)).transpose(1, 2) + last


class OrthogonalDirections(nn.Module):
    def __init__(self, basis, seed):
        super().__init__()
        self.weight = nn.Parameter(basis.detach().cpu().float().clone())
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(int(seed))
            orthogonal(self, 'weight', orthogonal_map='householder', use_trivialization=True)


class V11Model(nn.Module):
    def __init__(self, spec, basis, sources=None, backbone=None):
        super().__init__()
        self.spec = copy.deepcopy(spec)
        self.family = str(spec['family']).lower()
        self.dataset = spec['dataset']
        if self.family not in FAMILIES or self.dataset not in DATASETS:
            raise ValueError('Model family/dataset is outside the approved v11 contract')
        self.channels, self.latent = DATASETS[self.dataset]
        if spec.get('channels', self.channels) != self.channels:
            raise ValueError('Channel count differs from the approved data contract')
        allowed_latent = (0, self.latent) if self.family == 'direct_nlinear' else (self.latent,)
        if spec.get('latent', self.latent) not in allowed_latent:
            raise ValueError('K differs from the approved data contract')
        basis = torch.as_tensor(basis, dtype=torch.float32, device='cpu').clone()
        if tuple(basis.shape) != (self.channels, self.latent):
            raise ValueError('PCA basis shape differs from the data contract')
        if not torch.allclose(basis.T @ basis, torch.eye(self.latent), atol=1e-5, rtol=0):
            raise ValueError('PCA basis is not orthonormal within the fixed tolerance')
        self.register_buffer('basis0', basis)
        self.sources = copy.deepcopy(sources or {})
        self.source_receipts = {}
        self.chunk_rows = None
        self.prune_zero_gamma = False
        self.deployed = False
        seed = int(spec.get('seed', 92601))

        if self.family == 'direct_nlinear':
            if backbone is not None:
                raise ValueError('DIRECT_NLINEAR must not load a TSFM')
            direct = DirectNLinear(seed)
            self.temporal = direct.temporal
            return

        if backbone is None:
            raise ValueError('This family requires the pinned backbone')
        install_finite_constant_backward(backbone)
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.median = list(backbone.chronos_config.quantiles).index(0.5)
        if self.family in LORA_FAMILIES:
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(seed)
                self.backbone, self.lora_registration = _attach_lora(self.backbone)
            if self.family == 'lora_native':
                saved, receipt = _source_checkpoint(self.sources, 'native')
                _check_dataset(saved, self.dataset, seed)
                state = saved['state']
                if not state or not all(name.startswith('backbone.') and 'lora_' in name for name in state):
                    raise ValueError('Native LoRA source contains unexpected adapter variables')
                expected = {'backbone.' + name for name, _ in self.backbone.named_parameters() if 'lora_' in name}
                if set(state) != expected:
                    raise ValueError('Native LoRA targets differ from the pinned adapter')
                self.load_state_dict(state, strict=False)
                self.source_receipts['native'] = receipt
                self.backbone.requires_grad_(False)
        self.backbone.eval()

        if self.family in ('a_p_lora_qfixed', 'base_level'):
            saved, receipt = _source_checkpoint(self.sources, 'level')
            _check_dataset(saved, self.dataset, seed)
            parent_spec = saved['spec']
            level_parent = (parent_spec.get('family') == 'level_res' or
                            (parent_spec.get('method') == 'level' and parent_spec.get('variant') == 'res'))
            if not level_parent or parent_spec.get('ed_mode') != 'fixed_ed':
                raise ValueError('A requires the trained fixed-PCA LEVEL_RES parent')
            state = saved['state']
            names = {'encoder.weight', 'decoder.weight', 'residual.down.weight', 'residual.up.weight'}
            if set(state) != names:
                raise ValueError('LEVEL source must contain exactly fixed E/D and the trained rank32 G')
            if not torch.equal(torch.as_tensor(saved['basis']).float(), basis):
                raise ValueError('LEVEL source PCA basis differs from this data contract')
            if not torch.equal(state['encoder.weight'], basis.T) or not torch.equal(state['decoder.weight'], basis):
                raise ValueError('LEVEL source E/D is not the saved fixed PCA')
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(0)
                self.encoder = nn.Linear(self.channels, self.latent, bias=False)
                self.decoder = nn.Linear(self.latent, self.channels, bias=False)
                self.residual = TemporalResidual()
            self.load_state_dict(state, strict=False)
            self.encoder.requires_grad_(False)
            self.decoder.requires_grad_(False)
            self.residual.requires_grad_(False)
            self.source_receipts['level'] = receipt

        if self.family in B_FAMILIES:
            saved, receipt = _source_checkpoint(self.sources, 'direct')
            _check_dataset(saved, self.dataset, seed)
            if not torch.equal(torch.as_tensor(saved['basis']).float(), basis):
                raise ValueError('DIRECT source data PCA receipt differs from this data contract')
            if saved['spec'].get('family', '').lower() != 'direct_nlinear':
                raise ValueError('B requires the selected direct raw-channel NLinear')
            self.direct = DirectNLinear(seed)
            state = saved['state']
            if saved.get('schema_version') == 11:
                state = {key: value for key, value in state.items() if key.startswith('temporal.')}
            if set(state) != {'temporal.weight', 'temporal.bias'}:
                raise ValueError('DIRECT source is not a shared affine 512-to-48 model')
            self.direct.load_state_dict(state, strict=True)
            self.direct.requires_grad_(False)
            self.source_receipts['direct'] = receipt
            if self.family in ('b_learned_u', 'b_learned_gamma'):
                self.directions = OrthogonalDirections(basis, seed)
                if self.family == 'b_learned_gamma':
                    self.directions.requires_grad_(False)
            else:
                self.register_buffer('fixed_u', basis.clone())
            self.register_buffer('gamma', torch.ones(self.latent, dtype=torch.float32))

    @property
    def U(self):
        if hasattr(self, 'directions'):
            return self.directions.weight
        if hasattr(self, 'fixed_u'):
            return self.fixed_u
        return self.basis0

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'backbone'):
            self.backbone.eval()
        if hasattr(self, 'direct'):
            self.direct.eval()
        return self

    def _check_input(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (512, self.channels):
            raise ValueError(f'Expected input [B,512,{self.channels}]')

    def backbone_point(self, x):
        batch, length, channels = x.shape
        rows = x.transpose(1, 2).reshape(batch * channels, length)
        chunk = self.chunk_rows
        if chunk is not None:
            if int(chunk) <= 0:
                raise ValueError('chunk_rows must be positive or None')
            if torch.is_grad_enabled():
                raise ValueError('Chunked execution is reserved for inference under no_grad/inference_mode')
        chunks = rows.split(int(chunk)) if chunk is not None else (rows,)
        points = []
        for part in chunks:
            output = self.backbone(context=part)
            points.append(output.quantile_preds[:, self.median, :48].clone(memory_format=torch.contiguous_format))
            del output
        point = torch.cat(points, dim=0) if len(points) > 1 else points[0]
        return point.reshape(batch, channels, 48).transpose(1, 2)

    def q_prediction(self, x):
        if self.family not in ('a_p_lora_qfixed', 'base_level'):
            raise ValueError('Only A and BASE_LEVEL own the frozen LEVEL Q predictor')
        with torch.no_grad():
            residual = x - self.decoder(self.encoder(x))
            last = residual[:, -1:, :].detach()
            return last.expand(-1, 48, -1) + self.residual(residual - last)

    def direct_prediction(self, x):
        if self.family == 'direct_nlinear':
            last = x[:, -1:, :].detach()
            return self.temporal((x - last).transpose(1, 2)).transpose(1, 2) + last
        if self.family in B_FAMILIES:
            with torch.no_grad():
                return self.direct(x)
        raise ValueError('This family does not have an NLinear prediction')

    def mixture_features(self, x):
        if self.family not in B_FAMILIES:
            raise ValueError('Gamma features are defined only for B')
        self._check_input(x)
        with torch.no_grad():
            s = self.direct(x)
        u = self.U
        latent = self.backbone_point(x @ u)
        delta = latent - s @ u
        return {'S': s, 'latent': latent, 'latent_delta': delta, 'U': u}

    def set_gamma(self, gamma):
        if self.family not in B_FAMILIES:
            raise ValueError('Only B has Gamma coefficients')
        value = torch.as_tensor(gamma, dtype=self.gamma.dtype, device=self.gamma.device)
        if value.shape != self.gamma.shape or not torch.isfinite(value).all() or ((value < 0) | (value > 1)).any():
            raise ValueError('Gamma must be a finite K-vector within [0,1]')
        with torch.no_grad():
            self.gamma.copy_(value)

    def components(self, x):
        self._check_input(x)
        if self.family == 'direct_nlinear':
            value = self.direct_prediction(x)
            return value, torch.zeros_like(value)
        if self.family in ('f0', 'full_lora_mse', 'lora_native'):
            value = self.backbone_point(x)
            return value, torch.zeros_like(value)
        if self.family in ('a_p_lora_qfixed', 'base_level'):
            with torch.no_grad():
                z = self.encoder(x)
            # Only the Q calculation is detached: the A LoRA forward remains in autograd.
            main = self.decoder(self.backbone_point(z))
            return main, self.q_prediction(x)
        if self.prune_zero_gamma:
            if torch.is_grad_enabled():
                raise ValueError('Exact-zero Gamma pruning is reserved for inference')
            with torch.no_grad():
                s = self.direct(x)
            active = self.gamma != 0
            if not bool(active.any()):
                return s, torch.zeros_like(s)
            u = self.U[:, active]
            delta = self.backbone_point(x @ u) - s @ u
            return s, (delta * self.gamma[active]) @ u.T
        parts = self.mixture_features(x)
        return parts['S'], (parts['latent_delta'] * self.gamma) @ parts['U'].T

    def forward(self, x):
        main, correction = self.components(x)
        return main + correction

    def adapter_names(self):
        return {name for name in self.state_dict()
                if not name.startswith('backbone.') or 'lora_' in name}

    def adapter_state(self):
        names = self.adapter_names()
        return {name: value.detach().cpu().clone() for name, value in self.state_dict().items() if name in names}

    def restore_adapter(self, state):
        if set(state) != self.adapter_names():
            missing = sorted(self.adapter_names() - set(state))
            extra = sorted(set(state) - self.adapter_names())
            raise ValueError(f'Adapter state differs; missing={missing}, extra={extra}')
        self.load_state_dict(state, strict=False)

    def trainable_state(self):
        return {name: value.detach().cpu().clone() for name, value in self.named_parameters() if value.requires_grad}

    def frozen_state_digest(self):
        trainable = {name for name, value in self.named_parameters() if value.requires_grad}
        return state_digest({name: value for name, value in self.state_dict().items() if name not in trainable})

    def model_config(self):
        return {'schema_version': 11, 'family': self.family, 'dataset': self.dataset,
                'channels': self.channels, 'latent': self.latent, 'context': 512, 'horizon': 48,
                'precision': 'float32', 'pretrained_revision': None if self.family == 'direct_nlinear' else REVISION,
                'point': 'native inverse-scaled quantile 0.5, first 48 observations',
                'lora': {'r': 8, 'alpha': 16, 'dropout': 0, 'targets': ['q', 'v'], 'bias': 'none'}
                if self.family in LORA_FAMILIES else None,
                'orthogonal': {'map': 'householder', 'use_trivialization': True, 'tied': True}
                if self.family in B_FAMILIES else None,
                'G': 'shared 512-32-48, bias=False, frozen trained LEVEL' if self.family in ('a_p_lora_qfixed', 'base_level') else None,
                'S': 'shared Linear(512,48,bias=True), detached last subtraction/restoration'
                if self.family in B_FAMILIES or self.family == 'direct_nlinear' else None}


def build_model(spec, basis, initial=None, device='cpu', sources=None, backbone=None):
    sources = sources if sources is not None else (initial or {}).get('sources', {})
    if spec['family'].lower() != 'direct_nlinear' and backbone is None:
        from chronos import ChronosBoltPipeline
        backbone = ChronosBoltPipeline.from_pretrained(str(SNAPSHOT), device_map='cpu', torch_dtype=torch.float32).model
    model = V11Model(spec, basis, sources=sources, backbone=backbone)
    if initial is not None:
        state = initial.get('state', initial)
        if set(state) == model.adapter_names():
            model.restore_adapter(state)
        else:
            if model.family in ('a_p_lora_qfixed', 'full_lora_mse'):
                expected = {name for name in model.adapter_names() if name.startswith('backbone.')}
            elif model.family == 'b_learned_u':
                expected = {name for name in model.adapter_names() if name.startswith('directions.')}
            elif model.family == 'direct_nlinear':
                expected = {'temporal.weight', 'temporal.bias'}
            else:
                raise ValueError('This family requires a complete adapter initialization')
            if set(state) != expected:
                raise ValueError('Shared initial state contains the wrong variables for this family')
            model.load_state_dict(state, strict=False)
        if initial.get('tensor_hashes') is not None and tensor_hashes(state) != initial['tensor_hashes']:
            raise ValueError('Initial state tensor hashes changed')
    return model.to(device)


def generate_initial(spec, basis, sources=None, backbone=None):
    """Return only the shared initial tensors; no parent or S checkpoint is needed."""
    family, seed = spec['family'].lower(), int(spec['seed'])
    if family == 'direct_nlinear':
        return {name: value.detach().cpu().clone() for name, value in DirectNLinear(seed).state_dict().items()}
    if family in ('b_learned_u', 'b_learned_gamma'):
        directions = OrthogonalDirections(torch.as_tensor(basis, dtype=torch.float32), seed)
        return {'directions.' + name: value.detach().cpu().clone() for name, value in directions.state_dict().items()}
    if family in ('a_p_lora_qfixed', 'full_lora_mse'):
        if backbone is None:
            from chronos import ChronosBoltPipeline
            backbone = ChronosBoltPipeline.from_pretrained(str(SNAPSHOT), device_map='cpu', torch_dtype=torch.float32).model
        backbone.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            backbone, _ = _attach_lora(backbone)
        return {'backbone.' + name: value.detach().cpu().clone()
                for name, value in backbone.state_dict().items() if 'lora_' in name}
    raise ValueError('Shared initial states are generated only for a trainable v11 family')


def make_checkpoint(model, initial_reference=None, extra=None):
    if model.deployed:
        raise ValueError('Save the original training model, not a merged deployment copy')
    saved = {'schema_version': 11, 'spec': copy.deepcopy(model.spec),
             'basis': model.basis0.detach().cpu().clone(), 'sources': copy.deepcopy(model.sources),
             'source_receipts': copy.deepcopy(model.source_receipts), 'state': model.adapter_state(),
             'model_config': model.model_config(), 'initial_reference': copy.deepcopy(initial_reference)}
    if extra:
        collisions = set(extra) & set(saved)
        if collisions - {'spec'} or ('spec' in collisions and extra['spec'] != saved['spec']):
            raise ValueError(f'Checkpoint extra cannot change model fields: {sorted(collisions)}')
        saved.update(extra)
    return saved


def restore_model(checkpoint_path, device='cpu', deploy=False):
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if saved.get('schema_version') != 11:
        raise ValueError('Use source paths to import v4-v7 parents; restore_model accepts v11 checkpoints')
    for receipt in saved.get('source_receipts', {}).values():
        if file_hash(receipt['path']) != receipt['sha256']:
            raise ValueError('A frozen source changed after checkpoint creation')
    reference = saved.get('initial_reference')
    if isinstance(reference, dict) and reference.get('path') and reference.get('sha256'):
        if file_hash(reference['path']) != reference['sha256']:
            raise ValueError('Initial state source changed after checkpoint creation')
    model = build_model(saved['spec'], saved['basis'], sources=saved['sources'], device=device)
    model.restore_adapter(saved['state'])
    if saved['model_config'] != model.model_config():
        raise ValueError('Checkpoint model definition changed')
    return deployment_copy(model) if deploy else model.eval()


def deployment_copy(model, prune_zero_gamma=True):
    deployed = copy.deepcopy(model).eval()
    if deployed.family in LORA_FAMILIES:
        deployed.backbone = deployed.backbone.merge_and_unload(safe_merge=True)
    if hasattr(deployed, 'directions'):
        u = deployed.directions.weight.detach().clone()
        del deployed.directions
        deployed.register_buffer('fixed_u', u)
    deployed.requires_grad_(False)
    deployed.prune_zero_gamma = bool(prune_zero_gamma and deployed.family in B_FAMILIES)
    if deployed.prune_zero_gamma and not bool((deployed.gamma != 0).any()):
        del deployed.backbone
    deployed.deployed = True
    return deployed


def restore_legacy_level(checkpoint_path, device='cpu'):
    """Use the original version's restore function for authoritative parent parity."""
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    dataset = saved.get('spec', {}).get('dataset')
    locations = {
        'hog': 'tsfm_peft_accuracy_recovery_v5_20260927/model_v5.py',
        'robin': 'tsfm_peft_level_confirmation_v6_20260928/model_v6.py',
        'jena': 'tsfm_peft_practical_controls_v7_20260928/model_v7.py',
    }
    if dataset not in locations:
        raise ValueError('Only the three approved LEVEL parents can be restored')
    path = ROOT / 'research' / locations[dataset]
    module_spec = importlib.util.spec_from_file_location('v11_original_level_' + dataset, path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module.restore_model(checkpoint_path, device=device)


def restore_reused_direct(checkpoint_path, device='cpu'):
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if saved.get('schema_version') == 11:
        return restore_model(checkpoint_path, device=device)
    spec = saved['spec']
    if spec.get('family') != 'direct_nlinear' or spec.get('ed_mode') != 'none':
        raise ValueError('The reused checkpoint must be direct raw-channel NLinear')
    if saved.get('initial_path') and saved.get('initial_sha256'):
        if file_hash(saved['initial_path']) != saved['initial_sha256']:
            raise ValueError('Reused DIRECT initialization receipt changed')
    model = build_model(spec, saved['basis'], initial=saved['state'], device=device)
    model.source_receipts['direct'] = {'path': str(Path(checkpoint_path).resolve()), 'sha256': file_hash(checkpoint_path)}
    return model.eval()


def macro_loss(prediction, target, mask):
    if prediction.shape != target.shape or prediction.shape != mask.shape or prediction.ndim != 3:
        raise ValueError('Prediction, target and mask must share [B,H,C] axes')
    observed = mask.bool()
    counts = observed.sum((0, 1))
    valid = counts > 0
    if not bool(valid.any()):
        raise ValueError('No observed target in this batch')
    error = torch.where(observed, prediction - target, torch.zeros_like(prediction))
    return (error.square().sum((0, 1))[valid] / counts[valid]).mean()
