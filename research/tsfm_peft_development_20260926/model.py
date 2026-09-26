from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / '.cache/huggingface/models--amazon--chronos-bolt-small/snapshots/772f3d25d38aec6d914c8949dab4462e2d46f5d8'


def pca_basis(train, rank):
    centered = train - train.mean(axis=0, keepdims=True)
    _, vectors = np.linalg.eigh(centered.T @ centered / len(centered))
    return vectors[:, -rank:][:, ::-1].copy().astype(np.float32)


class TemporalResidual(nn.Module):
    def __init__(self, context=512, horizon=48, rank=8):
        super().__init__()
        self.down = nn.Linear(context, rank, bias=False)
        self.up = nn.Linear(rank, horizon, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.down(x.transpose(1, 2))).transpose(1, 2)


class ForecastAdapter(nn.Module):
    def __init__(self, backbone, arm, basis, context=512, horizon=48, residual_rank=8):
        super().__init__()
        self.backbone = backbone
        self.arm = arm
        self.horizon = horizon
        self.median = list(backbone.chronos_config.quantiles).index(0.5)
        for p in backbone.parameters():
            p.requires_grad_(False)
        if arm == 'lora':
            from hier_peft.lora import attach_lora
            self.backbone, _ = attach_lora(backbone)
        elif arm in ('compress', 'residual', 'raw_bypass'):
            channels, rank = basis.shape
            self.encoder = nn.Linear(channels, rank, bias=False)
            self.decoder = nn.Linear(rank, channels, bias=False)
            with torch.no_grad():
                self.encoder.weight.copy_(torch.as_tensor(basis.T))
                self.decoder.weight.copy_(torch.as_tensor(basis))
            if arm != 'compress':
                self.residual = TemporalResidual(context, horizon, residual_rank)
        elif arm != 'f0':
            raise ValueError(arm)
        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def backbone_point(self, x):
        b, length, c = x.shape
        out = self.backbone(context=x.transpose(1, 2).reshape(b * c, length))
        return out.quantile_preds[:, self.median, :self.horizon].reshape(b, c, self.horizon).transpose(1, 2)

    def forward(self, x):
        if self.arm in ('f0', 'lora'):
            return self.backbone_point(x)
        z = self.encoder(x)
        pred = self.decoder(self.backbone_point(z))
        if self.arm == 'residual':
            pred = pred + self.residual(x - self.decoder(z))
        elif self.arm == 'raw_bypass':
            pred = pred + self.residual(x)
        return pred

    def native_loss(self, x, target, mask):
        b, length, c = x.shape
        return self.backbone(
            context=x.transpose(1, 2).reshape(b * c, length),
            target=target.transpose(1, 2).reshape(b * c, self.horizon).clone(),
            target_mask=mask.transpose(1, 2).reshape(b * c, self.horizon),
        ).loss

    def adapter_state(self):
        trainable = {name for name, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if k in trainable}

    def restore_adapter(self, state):
        names = {k for k, p in self.named_parameters() if p.requires_grad}
        if set(state) != names:
            raise ValueError('checkpoint trainable names do not match')
        self.load_state_dict(state, strict=False)


def make_model(arm, basis, device='cuda', residual_rank=8):
    from chronos import ChronosBoltPipeline
    backbone = ChronosBoltPipeline.from_pretrained(str(SNAPSHOT), device_map='cpu', torch_dtype=torch.float32).model
    return ForecastAdapter(backbone, arm, basis, residual_rank=residual_rank).to(device)


def macro_loss(pred, target, mask, channel_count=None):
    squared = (pred - target).square() * mask
    count = mask.sum(dim=(0, 1)) if channel_count is None else channel_count
    valid = count > 0
    return (squared.sum(dim=(0, 1))[valid] / count[valid]).mean()
