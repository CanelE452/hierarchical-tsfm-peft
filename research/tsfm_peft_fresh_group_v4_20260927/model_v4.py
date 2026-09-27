"""Approved Hog models; old wrappers are imported under explicit unique names."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V2 = ROOT / 'research/tsfm_peft_followup_v2_20260926'
V3 = ROOT / 'research/tsfm_peft_backbone_value_v3_20260927'
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))


def import_file(name, path):
    if name in sys.modules:
        module = sys.modules[name]
        assert Path(module.__file__).resolve() == Path(path).resolve()
        return module
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


V2_MODEL = import_file('_fresh_v4_original_v2_model', V2 / 'model.py')
V3_MODEL = import_file('_fresh_v4_original_v3_linear_model', V3 / 'model.py')
macro_loss = V3_MODEL.macro_loss


def tensor_hashes(state):
    return {key: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for key, value in state.items()}


def initial_edg(basis, seed):
    basis = torch.as_tensor(basis, dtype=torch.float32, device='cpu')
    channels, latent = basis.shape
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(int(seed))
        encoder = nn.Linear(channels, latent, bias=False, device='cpu')
        decoder = nn.Linear(latent, channels, bias=False, device='cpu')
        residual = V3_MODEL.TemporalResidual()
    return {'encoder.weight': basis.T.contiguous().clone(),
            'decoder.weight': basis.clone(),
            'residual.down.weight': residual.down.weight.detach().clone(),
            'residual.up.weight': residual.up.weight.detach().clone()}


def configuration(spec):
    return {'schema_version': 1, 'family': spec['family'], 'arm': spec['arm'],
            'ed_mode': spec['ed_mode'], 'latent': int(spec['latent']),
            'context': 512, 'horizon': 48, 'channels': 32, 'precision': 'float32',
            'residual_rank': 32, 'normalization': spec.get('normalization', 'context'),
            'seed': spec.get('seed'), 'temporal_seed': None if spec.get('seed') is None else int(spec['seed']) + 3_000_000,
            'pretrained_revision': None if spec['family'] == 'linear_res' else '772f3d25d38aec6d914c8949dab4462e2d46f5d8',
            'reused_modules': {'tsfm': str(V2 / 'model.py'), 'linear': str(V3 / 'model.py')}}


class TSFMAdapter(V2_MODEL.ForecastAdapter):
    def __init__(self, backbone, spec, basis, initial_state):
        self.spec = copy.deepcopy(spec)
        super().__init__(backbone, spec['arm'], basis, residual_rank=32, ed_mode=spec['ed_mode'])
        if spec['arm'] in ('compress', 'residual', 'raw_bypass'):
            names = self.adapter_names()
            self.restore_adapter({key: initial_state[key] for key in names})

    def model_config(self):
        return configuration(self.spec)


def build_with_backbone(spec, basis, initial_state, backbone):
    """Synthetic-check entry point: caller provides the non-pretrained fake model."""
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(int(spec.get('seed') or 92601))
        return TSFMAdapter(backbone, spec, basis, initial_state)


def build_model(spec, basis, initial_state, device='cpu'):
    if spec['family'] == 'linear_res':
        model = V3_MODEL.LinearResidual(spec, initial_state)
    else:
        from chronos import ChronosBoltPipeline
        backbone = ChronosBoltPipeline.from_pretrained(
            str(V2_MODEL.SNAPSHOT), device_map='cpu', torch_dtype=torch.float32).model
        model = build_with_backbone(spec, basis, initial_state, backbone)
    return model.to(device)


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def restore_model(checkpoint_path, device='cpu'):
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    path = Path(saved['initial_path'])
    if file_hash(path) != saved['initial_sha256']:
        raise ValueError('Shared initial checkpoint hash changed')
    initial = torch.load(path, map_location='cpu', weights_only=False)
    if tensor_hashes(initial['state']) != initial['tensor_hashes']:
        raise ValueError('Shared initial tensor hashes changed')
    if not torch.equal(torch.as_tensor(saved['basis']), torch.as_tensor(initial['basis'])):
        raise ValueError('Checkpoint basis differs from shared initialization')
    model = build_model(saved['spec'], saved['basis'], initial['state'], device=device)
    model.restore_adapter(saved['state'])
    if model.model_config() != saved['model_config']:
        raise ValueError('Checkpoint model configuration changed')
    return model.eval()
