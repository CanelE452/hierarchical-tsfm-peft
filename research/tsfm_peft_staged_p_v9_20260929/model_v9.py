"""Frozen selected LEVEL parents with one matched temporal correction head."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REVISION = '772f3d25d38aec6d914c8949dab4462e2d46f5d8'
DATASETS = {'robin': (17, 5), 'jena': (21, 6)}
NEURAL_FAMILIES = ('staged_p', 'staged_raw')
ALPHAS = (0., .25, .5, .75, 1.)
PARENT_SOURCES = {
    'robin': ROOT / 'research/tsfm_peft_level_confirmation_v6_20260928/model_v6.py',
    'jena': ROOT / 'research/tsfm_peft_practical_controls_v7_20260928/model_v7.py'}


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def file_hash(path):
    value = hashlib.sha256()
    with resolve(path).open('rb') as handle:
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


def checked_path(receipt):
    path = resolve(receipt['path'])
    if file_hash(path) != receipt['sha256']:
        raise ValueError('Referenced artifact changed: ' + str(path))
    return path


def parent_receipt_for(dataset, seed):
    if dataset not in DATASETS or int(seed) not in (92601, 92602):
        raise ValueError('Only the approved dataset/seed pairs are supported')
    manifest = json.loads((HERE / 'reuse_manifest.json').read_text(encoding='utf-8'))
    receipt = copy.deepcopy(manifest['parents'][dataset][str(seed)])
    for name in ('checkpoint', 'model_source', 'result', 'head_initial'):
        checked_path(receipt[name])
    if resolve(receipt['model_source']['path']).resolve() != PARENT_SOURCES[dataset].resolve():
        raise ValueError('Parent module is not the approved dataset-specific source')
    return receipt


def initial_head(dataset, seed):
    receipt = parent_receipt_for(dataset, seed)
    initial = torch.load(checked_path(receipt['head_initial']), map_location='cpu', weights_only=False)
    if initial['epoch'] != 0 or initial['steps'] != 0:
        raise ValueError('The new head must use untrained v8 P initialization')
    if tensor_hashes(initial['state']) != initial['tensor_hashes']:
        raise ValueError('V8 initialization tensors changed')
    output = {f'head.{name}.weight': initial['state'][f'retained.{name}.weight'].clone()
              for name in ('down', 'up')}
    if torch.count_nonzero(output['head.up.weight']):
        raise ValueError('New head output layer must start at zero')
    return output


def _load_parent(spec, receipt, device):
    path = checked_path(receipt['model_source'])
    checked_path(receipt['result'])
    checkpoint = checked_path(receipt['checkpoint'])
    name = '_v9_parent_' + spec['dataset'] + '_' + receipt['model_source']['sha256'][:16]
    if name not in sys.modules:
        module_spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[name] = module
        module_spec.loader.exec_module(module)
    parent = sys.modules[name].restore_model(checkpoint, device=device)
    if (parent.family != 'level_res' or parent.spec['dataset'] != spec['dataset']
            or parent.spec['seed'] != spec['seed'] or parent.spec['ed_mode'] != 'fixed_ed'):
        raise ValueError('Restored parent differs from the fixed selected LEVEL dataset/seed')
    if parent.model_config()['pretrained_revision'] != REVISION:
        raise ValueError('Parent backbone revision changed')
    return parent


class TemporalHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Linear(512, 32, bias=False, dtype=torch.float32)
        self.up = nn.Linear(32, 48, bias=False, dtype=torch.float32)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.down(x.transpose(1, 2))).transpose(1, 2)


class StagedAdapter(nn.Module):
    def __init__(self, spec, basis, initial, parent, parent_receipt, alpha=1.):
        super().__init__()
        self.spec = copy.deepcopy(spec)
        self.family, self.dataset = spec['family'], spec['dataset']
        self.channels, self.latent = DATASETS[self.dataset]
        if self.family not in NEURAL_FAMILIES or spec['ed_mode'] != 'fixed_ed':
            raise ValueError('Only the approved staged P/RAW fixed-parent families are allowed')
        if (spec['channels'], spec['latent']) != (self.channels, self.latent):
            raise ValueError('Dataset dimensions changed')
        basis = torch.as_tensor(basis, dtype=torch.float32, device='cpu')
        if tuple(basis.shape) != (self.channels, self.latent):
            raise ValueError('PCA basis shape differs from the dataset')
        if not torch.equal(parent.encoder.weight.detach().cpu(), basis.T) or not torch.equal(parent.decoder.weight.detach().cpu(), basis):
            raise ValueError('Parent E/D differ from the frozen TRAIN PCA basis')
        self.parent = parent
        self.parent.requires_grad_(False)
        self.parent.eval()
        self.parent_receipt = copy.deepcopy(parent_receipt)
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(0)
            self.head = TemporalHead()
        self.restore_adapter(initial)
        if torch.count_nonzero(self.head.up.weight):
            raise ValueError('Initial head must preserve the trained parent exactly')
        self.set_alpha(alpha)

    def set_alpha(self, alpha):
        if float(alpha) not in ALPHAS:
            raise ValueError('Alpha must be one of the five predeclared constant values')
        self.alpha = float(alpha)
        return self

    def train(self, mode=True):
        super().train(mode)
        self.parent.eval()
        return self

    def parts(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (512, self.channels):
            raise ValueError('Expected approved [B,512,C] input')
        with torch.no_grad():
            parent_output = self.parent(x)
            if self.family == 'staged_p':
                reconstruction = self.parent.decoder(self.parent.encoder(x))
                head_input = reconstruction - reconstruction[:, -1:, :]
            else:
                head_input = x - x[:, -1:, :]
        correction = self.head(head_input)
        return {'parent_output': parent_output, 'head_input': head_input,
                'correction': correction, 'scaled_correction': self.alpha * correction}

    def forward(self, x):
        parts = self.parts(x)
        return parts['parent_output'] + parts['scaled_correction']

    def adapter_names(self):
        return {'head.down.weight', 'head.up.weight'}

    def adapter_state(self):
        return {key: value.detach().cpu().clone() for key, value in self.state_dict().items()
                if key in self.adapter_names()}

    def restore_adapter(self, state):
        if set(state) != self.adapter_names():
            raise ValueError('Checkpoint must contain exactly the new head tensors')
        self.load_state_dict(state, strict=False)

    def model_config(self):
        return {'schema_version': 1, 'dataset': self.dataset, 'family': self.family,
                'channels': self.channels, 'latent': self.latent, 'ed_mode': 'fixed_ed',
                'context': 512, 'horizon': 48, 'precision': 'float32',
                'head': 'shared 512-32-48,bias=False,activation=None',
                'trainable_parameters': 17920, 'cumulative_fitted_adapter_parameters': 35840,
                'input': 'PCA reconstruction minus its last value' if self.family == 'staged_p' else 'input minus its last value',
                'parent_id': self.parent_receipt['id'],
                'parent_checkpoint_sha256': self.parent_receipt['checkpoint']['sha256'],
                'parent_model_source_sha256': self.parent_receipt['model_source']['sha256'],
                'alpha': self.alpha, 'pretrained_revision': REVISION,
                'normalization': 'unchanged parent TRAIN standardization and native Bolt scaling'}


def build_with_parent(spec, basis, initial, parent, parent_receipt, alpha=1.):
    return StagedAdapter(spec, basis, initial, parent, parent_receipt, alpha)


def build_model(spec, basis, initial, device='cpu'):
    receipt = parent_receipt_for(spec['dataset'], spec['seed'])
    parent = _load_parent(spec, receipt, device)
    return StagedAdapter(spec, basis, initial, parent, receipt).to(device)


def restore_model(checkpoint_path, device='cpu'):
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    spec = saved['spec']
    receipt = parent_receipt_for(spec['dataset'], spec['seed'])
    if saved['parent_receipt'] != receipt:
        raise ValueError('Saved parent dependency differs from the pinned reuse manifest')
    if file_hash(saved['initial_path']) != saved['initial_sha256']:
        raise ValueError('Shared head initialization file changed')
    initial = torch.load(resolve(saved['initial_path']), map_location='cpu', weights_only=False)
    if tensor_hashes(initial['state']) != initial['tensor_hashes']:
        raise ValueError('Shared head initialization tensors changed')
    if not torch.equal(torch.as_tensor(saved['basis']), torch.as_tensor(initial['basis'])):
        raise ValueError('Saved basis differs from head initialization receipt')
    parent = _load_parent(spec, receipt, device)
    model = StagedAdapter(spec, saved['basis'], initial['state'], parent, receipt, saved['alpha']).to(device)
    model.restore_adapter(saved['state'])
    if saved['model_config'] != model.model_config():
        raise ValueError('Checkpoint model configuration changed')
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
