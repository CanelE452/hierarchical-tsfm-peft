"""All-channel coarse TSFM with a frozen DIRECT fine prediction."""
import copy
import sys
from pathlib import Path

import torch
from torch import nn

from runtime_v16 import HERE, ROOT, DATASETS, SEEDS, artifact, digest, import_file, load_data, read_json

OLD = ROOT / 'research/tsfm_peft_internal_vs_subspace_v11_20260930'
old = import_file(OLD / 'model_v11.py', 'v16_pinned_model_primitives')
tensor_hashes, state_digest = old.tensor_hashes, old.state_digest
SCHEMA = 'v16_all_channel_temporal_allocation_1'
STRIDE, CONTEXT, HORIZON, COARSE_HORIZON = 4, 512, 48, 12


def average4(value):
    if value.ndim != 3 or value.shape[1] % STRIDE:
        raise ValueError('Expected [B,n,C] with n divisible by four')
    b, n, c = value.shape
    return value.reshape(b, n // STRIDE, STRIDE, c).mean(dim=2)


def repeat4(value):
    if value.ndim != 3:
        raise ValueError('Expected [B,n,C] coarse values')
    return value.repeat_interleave(STRIDE, dim=1)


def fine_component(value):
    return value - repeat4(average4(value))


def parent_reference(dataset, seed):
    row = copy.deepcopy(read_json(HERE / 'data_manifest.json')['datasets'][dataset]['parents']['direct'][str(seed)])
    if row['dataset'] != dataset or row['family'] != 'direct' or row['seed'] != seed:
        raise ValueError('Only the selected original-channel DIRECT parent is permitted')
    return row


def load_direct(dataset, seed, device='cpu'):
    row = parent_reference(dataset, seed)
    artifact(row['checkpoint']['path'], row['checkpoint']['sha256'])
    previous = ROOT / 'research/tsfm_peft_group_basis_v13_20261001'
    if str(previous) not in sys.path:
        sys.path.append(str(previous))
    factory = import_file(previous / 'cost_v13.py', 'v16_original_parent_factory')
    model, _ = factory.load_original(row, load_data(dataset), device)
    if sum(p.numel() for p in model.parameters()) != 24624:
        raise ValueError('Selected DIRECT is not the shared affine 512-to-48 parent')
    return model.requires_grad_(False).eval(), row


def restore_initial(model):
    entry = read_json(HERE / 'initial_manifest.json')['datasets'][model.dataset][str(model.seed)]
    source_receipt = artifact(entry['receipt']['path'], entry['receipt']['sha256'])
    source = read_json(source_receipt['path'])
    if source != entry['source'] or source['kind'] != 'lora':
        raise ValueError('Historical initial-LoRA source record changed')
    path = Path(source['path'])
    if not path.is_file():
        state = model.adapter_state()
        receipt = {'kind': 'lora', 'dataset': model.dataset, 'seed': model.seed,
                   'tensor_hashes': tensor_hashes(state), 'source_receipt': source_receipt,
                   'historical_tensor_reuse': False,
                   'reason': 'Historical tensor file absent; deterministic pinned attachment initialization, not claimed identical'}
    else:
        stored = artifact(path, source['sha256'])
        saved = torch.load(stored['path'], map_location='cpu', weights_only=False)
        if saved['dataset'] != model.dataset or saved['seed'] != model.seed or saved['kind'] != 'lora':
            raise ValueError('Historical initializer dataset/seed/kind changed')
        state = saved['state']
        if tensor_hashes(state) != source['tensor_hashes']:
            raise ValueError('Historical initial adapter tensor hashes changed')
        model.restore_adapter(state)
        receipt = {**stored, 'kind': 'lora', 'dataset': model.dataset, 'seed': model.seed,
                   'tensor_hashes': tensor_hashes(state), 'source_receipt': source_receipt,
                   'historical_tensor_reuse': True}
    if any(torch.count_nonzero(value) for name, value in state.items() if 'lora_B.' in name):
        raise ValueError('Historical LoRA initializer is not a zero-update adapter')
    return receipt


class TemporalAllocationModel(nn.Module):
    def __init__(self, dataset, seed, direct, backbone, adapt=True, parent_receipt=None):
        super().__init__()
        if dataset not in DATASETS or seed not in SEEDS:
            raise ValueError('Dataset/seed outside the fixed v16 contract')
        self.dataset, self.seed, self.adapt = dataset, int(seed), bool(adapt)
        self.channels = DATASETS[dataset][0]
        self.family = 'temporal_lora' if adapt else 'temporal_f0'
        self.spec = {'dataset': dataset, 'family': self.family, 'seed': self.seed}
        self.direct = direct.requires_grad_(False).eval()
        old.install_finite_constant_backward(backbone)
        self.backbone = backbone.requires_grad_(False)
        config = backbone.chronos_config
        if (config.prediction_length != 64 or config.input_patch_size != 16
                or config.input_patch_stride != 16 or not config.use_reg_token):
            raise ValueError('Pinned Bolt native temporal configuration changed')
        self.median = list(config.quantiles).index(.5)
        self.lora_registration = {}
        if adapt:
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(self.seed)
                self.backbone, self.lora_registration = old._attach_lora(self.backbone)
        self.backbone.eval()
        self.parent_receipt = copy.deepcopy(parent_receipt)
        self.initial_receipt = None
        self.chunk_rows = None
        self.deployed = False

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        self.direct.eval()
        return self

    def _check_input(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (CONTEXT, self.channels):
            raise ValueError('Expected original-channel input [B,512,C]')

    def coarse_point(self, x):
        self._check_input(x)
        context = average4(x)
        b, length, c = context.shape
        rows = context.transpose(1, 2).reshape(b * c, length)
        if self.chunk_rows is not None:
            if int(self.chunk_rows) != self.chunk_rows or self.chunk_rows <= 0 or torch.is_grad_enabled():
                raise ValueError('Positive row chunks are reserved for inference without gradients')
            chunks = rows.split(int(self.chunk_rows))
        else:
            chunks = (rows,)
        outputs = []
        for part in chunks:
            output = self.backbone(context=part)
            quantiles = output.quantile_preds
            if quantiles.ndim != 3 or quantiles.shape[0] != len(part) or quantiles.shape[-1] != 64:
                raise ValueError('Native quantile output axes/length changed')
            outputs.append(quantiles[:, self.median, :COARSE_HORIZON].clone(memory_format=torch.contiguous_format))
            del quantiles, output
        points = torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]
        return points.reshape(b, c, COARSE_HORIZON).transpose(1, 2)

    def q_prediction(self, x):
        self._check_input(x)
        with torch.no_grad():
            prediction = self.direct(x)
            if tuple(prediction.shape) != (len(x), HORIZON, self.channels):
                raise ValueError('Frozen DIRECT must forecast all 48 original-channel values')
            return fine_component(prediction)

    def components(self, x):
        return repeat4(self.coarse_point(x)), self.q_prediction(x)

    def forward(self, x):
        coarse, fine = self.components(x)
        return coarse + fine

    def adapter_state(self):
        return {name: value.detach().cpu().clone() for name, value in self.state_dict().items()
                if name.startswith('backbone.') and 'lora_' in name}

    def restore_adapter(self, state):
        expected = self.adapter_state()
        if state.keys() != expected.keys() or any(state[name].shape != expected[name].shape for name in state):
            raise ValueError('Saved state must contain exactly the matching LoRA tensors')
        if any(not torch.isfinite(value).all() for value in state.values()):
            raise ValueError('Nonfinite adapter tensor')
        self.load_state_dict(state, strict=False)

    def frozen_state_digest(self):
        trainable = {name for name, value in self.named_parameters() if value.requires_grad}
        return state_digest({name: value for name, value in self.state_dict().items() if name not in trainable})

    def model_config(self):
        return {'schema': SCHEMA, **self.spec, 'channels': self.channels, 'context': CONTEXT, 'horizon': HORIZON,
                'temporal_stride': STRIDE, 'coarse_context': 128, 'coarse_horizon': COARSE_HORIZON,
                'pretrained_revision': old.REVISION, 'precision': 'float32',
                'grouping': 'Four consecutive positions relative to each origin; never global-bin resampling',
                'point': 'Native inverse-scaled median; first 12 of 64 outputs; each repeated four times',
                'fine': 'Frozen selected DIRECT(X) minus block-constant projection of that prediction',
                'parent_fitted_parameters': 24624, 'new_registered_parameters': 294912 if self.adapt else 0,
                'lora': {'r': 8, 'alpha': 16, 'dropout': 0, 'targets': ['q', 'v'], 'bias': 'none'} if self.adapt else None,
                'new_channel_compression': False, 'initial_is_original_f0_or_direct': False}


def build_model(dataset, seed, device='cuda', adapt=True):
    from chronos import ChronosBoltPipeline
    if not old.SNAPSHOT.is_dir():
        raise FileNotFoundError('Pinned local backbone missing; downloading is outside the v16 contract')
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(int(seed))
        direct, parent = load_direct(dataset, seed, device='cpu')
        backbone = ChronosBoltPipeline.from_pretrained(str(old.SNAPSHOT), device_map='cpu', torch_dtype=torch.float32).model
        model = TemporalAllocationModel(dataset, seed, direct, backbone, adapt=adapt, parent_receipt=parent)
        if adapt:
            model.initial_receipt = restore_initial(model)
            registered = {name: p.numel() for name, p in model.named_parameters() if p.requires_grad}
            if sum(registered.values()) != 294912 or not all(name.startswith('backbone.') and 'lora_' in name for name in registered):
                raise ValueError('Only the expected LoRA tensors may be trainable')
    return model.to(device).eval()


def make_checkpoint(model, spec, **extra):
    if model.deployed or not model.adapt:
        raise ValueError('Only the unmerged trainable temporal-LoRA model is checkpointed')
    if any(spec[key] != model.spec[key] for key in ('dataset', 'family', 'seed')):
        raise ValueError('Checkpoint spec differs from the actual model')
    saved = {'schema': SCHEMA, 'spec': copy.deepcopy(spec), 'state': model.adapter_state(),
             'initial_receipt': copy.deepcopy(model.initial_receipt), 'parent_receipt': copy.deepcopy(model.parent_receipt),
             'model_config': model.model_config(), 'data_manifest_sha256': digest(HERE / 'data_manifest.json')}
    if set(saved) & set(extra):
        raise ValueError('Checkpoint metadata cannot override the model/source contract')
    saved.update(extra)
    return saved


def restore_model(path, device='cuda'):
    saved = torch.load(Path(path), map_location='cpu', weights_only=False)
    if saved['schema'] != SCHEMA or saved['data_manifest_sha256'] != digest(HERE / 'data_manifest.json'):
        raise ValueError('Temporal checkpoint schema/data contract changed')
    spec = saved['spec']
    if spec['family'] != 'temporal_lora':
        raise ValueError('Unexpected checkpoint family')
    model = build_model(spec['dataset'], spec['seed'], device=device, adapt=True)
    if model.initial_receipt != saved['initial_receipt'] or model.parent_receipt != saved['parent_receipt']:
        raise ValueError('Historical initializer or frozen parent changed')
    if model.model_config() != saved['model_config']:
        raise ValueError('Saved temporal forecasting definition changed')
    model.restore_adapter(saved['state'])
    return model.eval()


def deployment_copy(model):
    if model.deployed:
        raise ValueError('Export an unmerged source model only')
    deployed = copy.deepcopy(model).eval()
    if deployed.adapt:
        deployed.backbone = deployed.backbone.merge_and_unload(safe_merge=True)
    deployed.requires_grad_(False)
    deployed.deployed = True
    return deployed
