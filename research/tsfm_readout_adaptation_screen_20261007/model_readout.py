"""Output affine and the original Bolt output block on a frozen prefix."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
ARMS = ('OUTPUT_AFFINE', 'NATIVE_HEAD')
HEAD_PARAMETERS = ('hidden_layer.weight', 'hidden_layer.bias', 'output_layer.weight',
                   'output_layer.bias', 'residual_layer.weight', 'residual_layer.bias')


def load_backbone(dataset, device='cuda'):
    manifest_path = ROOT / 'research/level_bolt_backbone_scaling_v1_20261006/reuse_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    rows = [row for row in manifest['units'][dataset]['rows'] if row['method'] == 'F0']
    if len(rows) != 1:
        raise ValueError('Expected one exact F0 parent for ' + dataset)
    module_name = 'runtime_v18'
    if module_name not in sys.modules:
        path = ROOT / 'research/level_matched_learned_ed_v18_20261006/runtime_v18.py'
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    parent = sys.modules[module_name].load_original(rows[0], device=device, deploy=True)
    backbone = parent.backbone
    backbone.eval().requires_grad_(False)
    norm_forward = backbone.instance_norm.forward
    backbone.readout_parent_receipt = {
        'dataset': dataset, 'id': rows[0]['id'], 'row': rows[0],
        'instance_norm_forward': norm_forward.__module__ + '.' + norm_forward.__qualname__,
        'instance_norm_inverse': backbone.instance_norm.inverse.__module__ + '.' + backbone.instance_norm.inverse.__qualname__,
    }
    return backbone


def compute_loss(pred, target, mask, channel_count=None):
    if pred.ndim != 3 or pred.shape[1] != 48 or pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError('Prediction/target/mask must share [B,48,C] axes')
    observed = mask.bool()
    count = observed.sum((0, 1)) if channel_count is None else channel_count
    if count.shape != (pred.shape[-1],) or count.device != pred.device:
        raise ValueError('Effective-batch counts must share the prediction channel axis and device')
    valid = count > 0
    if not bool(valid.any()):
        raise ValueError('No observed target in effective batch')
    error = torch.where(observed, pred - target, torch.zeros_like(pred))
    return (error.square().sum((0, 1))[valid] / count[valid]).mean()


class Readout(nn.Module):
    def __init__(self, backbone, channels, arm='OUTPUT_AFFINE'):
        super().__init__()
        if arm not in ARMS or channels <= 0:
            raise ValueError('Unknown readout arm or channel count')
        self.backbone, self.channels, self.arm = backbone, channels, arm
        quantiles = list(backbone.chronos_config.quantiles)
        self.median = quantiles.index(.5)
        self.num_quantiles = len(quantiles)
        self.native_horizon = backbone.chronos_config.prediction_length
        if self.num_quantiles != 9 or self.median != 4 or self.native_horizon != 64:
            raise ValueError('The pinned Bolt-small quantile/output contract changed')
        self.backbone.requires_grad_(False)
        if arm == 'OUTPUT_AFFINE':
            reference = next(backbone.parameters())
            self.weight = nn.Parameter(torch.eye(channels, dtype=reference.dtype, device=reference.device))
            self.bias = nn.Parameter(torch.zeros(channels, dtype=reference.dtype, device=reference.device))
            self.trainable_names = ('weight', 'bias')
        else:
            head = backbone.output_patch_embedding
            if set(dict(head.named_parameters())) != set(HEAD_PARAMETERS):
                raise ValueError('The original forecast-output block must contain its six native parameter tensors')
            head.requires_grad_(True)
            self.trainable_names = tuple('backbone.output_patch_embedding.' + name for name in HEAD_PARAMETERS)
        self.train(False)
        actual = {name for name, tensor in self.named_parameters() if tensor.requires_grad}
        if actual != set(self.trainable_names):
            raise ValueError('Trainable parameters differ from the explicit readout allowlist')

    def train(self, mode=True):
        super().train(False)
        return self

    def _native_point(self, h, loc, scale):
        batch = h.shape[0]
        rows = batch * self.channels
        raw = self.backbone.output_patch_embedding(h.reshape(rows, 1, 512))
        native = self.backbone.instance_norm.inverse(
            raw.reshape(rows, self.num_quantiles * self.native_horizon),
            (loc.reshape(rows, 1), scale.reshape(rows, 1)))
        native = native.reshape(rows, self.num_quantiles, self.native_horizon)
        return native[:, self.median, :48].reshape(batch, self.channels, 48).transpose(1, 2).contiguous()

    @torch.inference_mode(False)
    @torch.no_grad()
    def features(self, x, include_point=True):
        if x.ndim != 3 or x.shape[1:] != (512, self.channels):
            raise ValueError('Context must have canonical [B,512,C] axes')
        batch = x.shape[0]
        rows = x.transpose(1, 2).reshape(batch * self.channels, 512)
        encoded, loc_scale, input_embeds, attention_mask = self.backbone.encode(context=rows)
        decoded = self.backbone.decode(input_embeds, attention_mask, encoded)
        if decoded.shape != (batch * self.channels, 1, 512):
            raise ValueError('Frozen decoder feature shape differs from [B*C,1,512]')
        loc, scale = loc_scale
        result = {'h': decoded.reshape(batch, self.channels, 512).detach(),
                  'loc': loc.reshape(batch, self.channels, 1).detach(),
                  'scale': scale.reshape(batch, self.channels, 1).detach()}
        if include_point:
            result['point'] = self._native_point(result['h'], result['loc'], result['scale']).detach()
        return result

    def from_features(self, features):
        if self.arm == 'OUTPUT_AFFINE':
            point = features['point']
            if point.ndim != 3 or point.shape[1:] != (48, self.channels):
                raise ValueError('Cached point forecasts must have [B,48,C] axes')
            return point @ self.weight + self.bias
        h, loc, scale = (features[key] for key in ('h', 'loc', 'scale'))
        if h.ndim != 3 or h.shape[1:] != (self.channels, 512) or loc.shape != (*h.shape[:2], 1) or scale.shape != loc.shape:
            raise ValueError('Cached native-head features/norm shapes changed')
        return self._native_point(h, loc, scale)

    def forward(self, x):
        return self.from_features(self.features(x, include_point=self.arm == 'OUTPUT_AFFINE'))

    def trainable_state(self):
        parameters = dict(self.named_parameters())
        return {name: parameters[name].detach().cpu().clone() for name in self.trainable_names}

    def load_trainable_state(self, state):
        if set(state) != set(self.trainable_names):
            raise ValueError('Checkpoint keys differ from the readout allowlist')
        parameters = dict(self.named_parameters())
        for name in self.trainable_names:
            saved, parameter = state[name], parameters[name]
            if not isinstance(saved, torch.Tensor) or saved.shape != parameter.shape or saved.dtype != parameter.dtype:
                raise ValueError('Readout checkpoint tensor shape/dtype changed: ' + name)
        with torch.no_grad():
            for name in self.trainable_names:
                parameters[name].copy_(state[name])

    def frozen_hash(self):
        digest = hashlib.sha256()
        allowed = set(self.trainable_names)
        parameters = [(name, value) for name, value in self.named_parameters(remove_duplicate=False) if name not in allowed]
        buffers = list(self.named_buffers(remove_duplicate=False))
        for kind, entries in (('parameter', parameters), ('buffer', buffers)):
            for name, value in sorted(entries):
                value = value.detach().cpu().contiguous()
                digest.update((kind + ':' + name).encode())
                digest.update(str(value.dtype).encode())
                digest.update(str(tuple(value.shape)).encode())
                digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()
