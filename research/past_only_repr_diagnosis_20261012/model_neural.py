"""Small common encoder; no TS2Vec or GTConvAE reproduction claim."""
import torch
from torch import nn
from torch.nn import functional as F


class TinyEncoder(nn.Module):
    dilations = (1, 2, 4, 8, 16)
    receptive_field = 1 + 2 * sum(dilations)

    def __init__(self, channels, causal=True):
        super().__init__()
        self.channels, self.causal = channels, causal
        self.convs = nn.ModuleList([nn.Conv1d(channels if i == 0 else 32, 32, 3, dilation=d)
                                    for i, d in enumerate(self.dilations)])
        self.norms = nn.ModuleList([nn.LayerNorm(32) for _ in self.dilations])
        self.output = nn.Linear(32, 16)

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.channels:
            raise ValueError('Encoder input must be [batch,time,channels]')
        for conv, norm, dilation in zip(self.convs, self.norms, self.dilations):
            padding = (2*dilation, 0) if self.causal else (dilation, dilation)
            x = conv(F.pad(x.transpose(1, 2), padding)).transpose(1, 2)
            x = F.gelu(norm(x))
        return self.output(x)


class ReconstructionProbe(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(16, 128), nn.ReLU(), nn.Linear(128, channels))

    def forward(self, z):
        return self.layers(z)
