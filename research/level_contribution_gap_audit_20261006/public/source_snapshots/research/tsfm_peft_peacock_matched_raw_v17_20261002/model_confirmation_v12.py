# TEST 노출 후 보충 대조 (v12 확인 실험 아님)
"""Fixed-recipe v12 confirmation models; importing does not load a backbone."""
import copy
from pathlib import Path

import torch
from torch import nn

from runtime_confirmation_v12 import HERE, artifact, import_file, load_data

ROOT = HERE.parents[1]
OLD_MODEL = ROOT / 'research/tsfm_peft_internal_vs_subspace_v11_20260930/model_v11.py'
old = import_file(OLD_MODEL, 'confirmation_v12_pinned_model_primitives')
REVISION, SNAPSHOT = old.REVISION, old.SNAPSHOT
SCHEMA = 'v17_post_exposure_matched_raw_1'
FAMILIES = ('level', 'a', 'full_mse', 'native', 'direct', 'f0')
LORA_FAMILIES = ('a', 'full_mse', 'native')
tensor_hashes, state_digest = old.tensor_hashes, old.state_digest


def _parent_checkpoint(parent, dataset, seed):
    if parent is None:
        raise ValueError('A requires the selected new-unit LEVEL parent')
    if isinstance(parent, dict):
        path = parent.get('path') or parent.get('checkpoint') or parent.get('level_checkpoint')
        expected = parent.get('sha256') or parent.get('checkpoint_sha256') or parent.get('level_sha256')
    else:
        path, expected = parent, None
    receipt = artifact(path, expected)
    saved = torch.load(receipt['path'], map_location='cpu', weights_only=False)
    if saved.get('schema') != SCHEMA or saved['spec']['family'] != 'level':
        raise ValueError('A parent must be a new confirmation LEVEL checkpoint')
    if saved['spec']['dataset'] != dataset or int(saved['spec']['seed']) != int(seed):
        raise ValueError('LEVEL parent dataset/seed differs')
    return saved, receipt


class ConfirmationModel(nn.Module):
    def __init__(self, dataset, family, seed, basis, backbone=None, parent=None):
        super().__init__()
        if family not in FAMILIES:
            raise ValueError(f'Unapproved confirmation family: {family}')
        self.dataset, self.family, self.seed = dataset, family, int(seed)
        basis = torch.as_tensor(basis, dtype=torch.float32, device='cpu').clone()
        if basis.ndim != 2:
            raise ValueError('TRAIN PCA must be a matrix')
        self.channels, self.latent = basis.shape
        if self.latent != (self.channels + 3) // 4:
            raise ValueError('Confirmation requires K=ceil(C/4)')
        if not torch.allclose(basis.T @ basis, torch.eye(self.latent), atol=1e-5, rtol=0):
            raise ValueError('TRAIN PCA is not orthonormal at the fixed tolerance')
        self.register_buffer('basis0', basis)
        self.spec = {'dataset': dataset, 'family': family, 'seed': self.seed,
                     'channels': self.channels, 'latent': self.latent,
                     'context': 512, 'horizon': 48}
        self.chunk_rows, self.deployed, self.parent_receipt = None, False, None
        if family == 'direct':
            if backbone is not None:
                raise ValueError('Direct NLinear must not load a TSFM')
            self.temporal = old.DirectNLinear(self.seed).temporal
            return
        if backbone is None:
            raise ValueError('TSFM family requires the pinned backbone')
        old.install_finite_constant_backward(backbone)
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.median = list(backbone.chronos_config.quantiles).index(0.5)
        if family in LORA_FAMILIES:
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(self.seed)
                self.backbone, self.lora_registration = old._attach_lora(self.backbone)
        self.backbone.eval()
        if family in ('level', 'a'):
            # Preserve the v7 E, D, G random-draw order and literal-zero G output.
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(self.seed)
                self.encoder = nn.Linear(self.channels, self.latent, bias=False)
                self.decoder = nn.Linear(self.latent, self.channels, bias=False)
                self.residual = old.TemporalResidual()
                nn.init.zeros_(self.residual.up.weight)
            with torch.no_grad():
                self.encoder.weight.copy_(basis.T)
                self.decoder.weight.copy_(basis)
            self.encoder.requires_grad_(False)
            self.decoder.requires_grad_(False)
            if family == 'a':
                saved, self.parent_receipt = _parent_checkpoint(parent, dataset, self.seed)
                if not torch.equal(saved['basis'], basis):
                    raise ValueError('A and LEVEL parent TRAIN PCA differ')
                names = {'basis0', 'encoder.weight', 'decoder.weight',
                         'residual.down.weight', 'residual.up.weight'}
                if set(saved['state']) != names:
                    raise ValueError('LEVEL parent has unexpected tensors')
                self.load_state_dict(saved['state'], strict=False)
                self.residual.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'backbone'):
            self.backbone.eval()
        return self

    def backbone_point(self, x):
        batch, length, channels = x.shape
        rows = x.transpose(1, 2).reshape(batch * channels, length)
        if self.chunk_rows is not None:
            if self.chunk_rows <= 0 or torch.is_grad_enabled():
                raise ValueError('Positive row chunking is for inference only')
            chunks = rows.split(int(self.chunk_rows))
        else:
            chunks = (rows,)
        predictions = []
        for chunk in chunks:
            output = self.backbone(context=chunk)
            predictions.append(output.quantile_preds[:, self.median, :48].clone(memory_format=torch.contiguous_format))
            del output
        point = torch.cat(predictions) if len(predictions) > 1 else predictions[0]
        return point.reshape(batch, channels, 48).transpose(1, 2)

    def q_prediction(self, x):
        if self.family not in ('level', 'a'):
            raise ValueError('Only LEVEL and A own Q')
        residual = x - self.decoder(self.encoder(x))
        last = residual[:, -1:, :].detach()
        return last.expand(-1, 48, -1) + self.residual(x - x[:, -1:, :].detach())

    def components(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (512, self.channels):
            raise ValueError('Expected [B,512,C] input')
        if self.family == 'direct':
            last = x[:, -1:, :].detach()
            value = self.temporal((x - last).transpose(1, 2)).transpose(1, 2) + last
            return value, torch.zeros_like(value)
        if self.family in ('f0', 'full_mse', 'native'):
            value = self.backbone_point(x)
            return value, torch.zeros_like(value)
        if self.family == 'level':
            with torch.no_grad():
                main = self.decoder(self.backbone_point(self.encoder(x)))
            return main, self.q_prediction(x)
        with torch.no_grad():
            z = self.encoder(x)
            q = self.q_prediction(x)
        main = self.decoder(self.backbone_point(z))
        return main, q

    def forward(self, x):
        main, q = self.components(x)
        return main + q

    def native_loss(self, x, target, mask):
        if self.family != 'native':
            raise ValueError('Native quantile training is reserved for native LoRA')
        batch, length, channels = x.shape
        return self.backbone(context=x.transpose(1, 2).reshape(batch * channels, length),
                             target=target.transpose(1, 2).reshape(batch * channels, 48).clone(),
                             target_mask=mask.transpose(1, 2).reshape(batch * channels, 48)).loss

    def adapter_state(self):
        return {name: value.detach().cpu().clone() for name, value in self.state_dict().items()
                if not name.startswith('backbone.') or 'lora_' in name}

    def restore_adapter(self, state):
        if state.keys() != self.adapter_state().keys():
            raise ValueError('Checkpoint adapter tensor names differ')
        self.load_state_dict(state, strict=False)

    def model_config(self):
        return {'schema': SCHEMA, **self.spec, 'pretrained_revision': None if self.family == 'direct' else REVISION,
                'precision': 'float32', 'point': 'native inverse-scaled median, first 48 observations',
                'G': 'shared bias-free 512-32-48; fixed PCA; RAW-last(X) G input; residual level restored'
                if self.family in ('level', 'a') else None,
                'lora': {'r': 8, 'alpha': 16, 'dropout': 0, 'targets': ['q', 'v'], 'bias': 'none'}
                if self.family in LORA_FAMILIES else None,
                'direct': 'shared Linear(512,48,bias=True); last centering' if self.family == 'direct' else None}


def _build(dataset, family, seed, basis, device, parent):
    backbone = None
    if family != 'direct':
        from chronos import ChronosBoltPipeline
        backbone = ChronosBoltPipeline.from_pretrained(str(SNAPSHOT), device_map='cpu', torch_dtype=torch.float32).model
    return ConfirmationModel(dataset, family, seed, basis, backbone, parent).to(device)


def build_model(dataset, family, seed, device='cuda', parent=None):
    data = load_data(dataset, include_test=False)
    if family == 'a' and parent is None:
        from runtime_confirmation_v12 import sources_for
        parent = sources_for(dataset, seed)
    return _build(dataset, family, seed, data['basis'], device, parent)


def make_checkpoint(model, spec, initial_reference=None, **extra):
    if model.deployed:
        raise ValueError('Save unmerged training state')
    return {'schema': SCHEMA, 'spec': copy.deepcopy(spec), 'basis': model.basis0.detach().cpu().clone(),
            'state': model.adapter_state(), 'model_config': model.model_config(),
            'parent': copy.deepcopy(model.parent_receipt), 'initial_reference': copy.deepcopy(initial_reference), **extra}


def restore_model(checkpoint, device='cuda'):
    saved = torch.load(Path(checkpoint), map_location='cpu', weights_only=False)
    if saved.get('schema') != SCHEMA:
        raise ValueError('Not a v12 confirmation checkpoint')
    reference = saved.get('initial_reference')
    if reference:
        artifact(reference['path'], reference['sha256'])
    spec = saved['spec']
    model = _build(spec['dataset'], spec['family'], spec['seed'], saved['basis'], device, saved['parent'])
    model.restore_adapter(saved['state'])
    if model.model_config() != saved['model_config']:
        raise ValueError('Saved model configuration changed')
    return model.eval()


def deployment_copy(model):
    deployed = copy.deepcopy(model).eval()
    if deployed.family in LORA_FAMILIES:
        deployed.backbone = deployed.backbone.merge_and_unload(safe_merge=True)
    deployed.requires_grad_(False)
    deployed.deployed = True
    return deployed
