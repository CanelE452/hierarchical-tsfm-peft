"""Online frozen LEVEL parent plus one fitted bias-free temporal ridge map."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class RidgeAdapter(nn.Module):
    def __init__(self, parent, weight, family):
        super().__init__()
        if family not in ('ridge_p', 'ridge_raw'):
            raise ValueError('Only the approved P and RAW inputs are allowed')
        weight = torch.as_tensor(weight, dtype=torch.float32)
        if weight.shape != (48, 512) or not torch.isfinite(weight).all():
            raise ValueError('Approved deployment weight is finite FP32 [48,512]')
        self.parent, self.family = parent, family
        self.parent.requires_grad_(False)
        self.parent.eval()
        self.channels = int(parent.encoder.in_features)
        self.latent = int(parent.encoder.out_features)
        self.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)

    def train(self, mode=True):
        super().train(mode)
        self.parent.eval()
        return self

    def parts(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (512, self.channels):
            raise ValueError('Input must match the parent [B,512,C] contract')
        with torch.no_grad():
            parent_output = self.parent(x)
            delta = x - x[:, -1:, :]
            head_input = self.parent.decoder(self.parent.encoder(delta)) if self.family == 'ridge_p' else delta
            correction = F.linear(head_input.transpose(1, 2), self.weight).transpose(1, 2)
        return {'parent_output': parent_output, 'head_input': head_input, 'correction': correction}

    def forward(self, x):
        parts = self.parts(x)
        return parts['parent_output'] + parts['correction']

    def model_config(self):
        return {'family': self.family, 'channels': self.channels, 'latent': self.latent,
                'context': 512, 'horizon': 48, 'bias': False, 'precision': 'float32',
                'fitted_coefficients': self.weight.numel(), 'parent_frozen': True,
                'deployment_requires_grad_parameters': 0}


def restore_model(checkpoint_path, device='cpu'):
    from runtime_ridge import load_parent, parent_receipt_for

    saved = torch.load(Path(checkpoint_path), map_location='cpu', weights_only=False)
    required = {'dataset', 'seed', 'family', 'penalty', 'weight', 'basis', 'parent_receipt'}
    if not required.issubset(saved):
        raise ValueError('Incomplete ridge checkpoint dependency contract')
    if saved['parent_receipt'] != parent_receipt_for(saved['dataset'], int(saved['seed'])):
        raise ValueError('Checkpoint parent receipt differs from the pinned manifest')
    parent = load_parent(saved['dataset'], int(saved['seed']), device=device)
    basis = torch.as_tensor(saved['basis'], dtype=torch.float32)
    if (not torch.equal(parent.encoder.weight.detach().cpu(), basis.T)
            or not torch.equal(parent.decoder.weight.detach().cpu(), basis)):
        raise ValueError('Checkpoint basis differs from the pinned frozen parent')
    model = RidgeAdapter(parent, saved['weight'], saved['family']).to(device)
    model.spec = {key: saved[key] for key in ('dataset', 'seed', 'family', 'penalty')}
    model.parent_receipt = saved['parent_receipt']
    return model.eval()
