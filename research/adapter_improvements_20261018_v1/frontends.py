"""Registered causal local input alternatives; no foundation or data I/O."""
import torch
from torch import nn
from torch.nn import functional as F


def trailing_mean(x, kernel_size=25):
    """Trailing mean on [B,L,C], extending only the first supplied sample."""
    if x.ndim != 3 or x.shape[1] < 1 or kernel_size < 1:
        raise ValueError('Expected nonempty [B,L,C] and positive kernel')
    return F.avg_pool1d(F.pad(x.transpose(1, 2), (kernel_size-1, 0), mode='replicate'),
                        kernel_size, stride=1).transpose(1, 2)


class _LocalBranch(nn.Module):
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel
        self.input = nn.Conv1d(1, 32, kernel, bias=False)
        self.output = nn.Conv1d(32, 1, 1, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, x):
        batch, length, channels = x.shape
        rows = x.transpose(1, 2).reshape(batch * channels, 1, length)
        result = self.output(F.gelu(self.input(F.pad(rows, (self.kernel-1, 0), mode='replicate'))))
        return result.reshape(batch, channels, length).transpose(1, 2)


class LocalFrontend(nn.Module):
    """P_SPLIT or its identical-capacity raw-half-input control (1152 params)."""
    def __init__(self, channels, split=True, seed=92601):
        super().__init__()
        if not isinstance(channels, int) or channels < 1:
            raise ValueError('Positive channel count required')
        self.channels = channels
        self.split = bool(split)
        # Candidate and control draw exactly the same two branch tensors while
        # leaving the sampler/global initialization stream untouched.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed))
            self.short = _LocalBranch(9)
            self.trend = _LocalBranch(25)

    def components(self, x):
        if x.ndim != 3 or x.shape[-1] != self.channels or x.shape[1] < 1:
            raise ValueError('Expected nonempty [B,L,registered C]')
        if self.split:
            trend = trailing_mean(x, 25)
            return x - trend, trend
        return x / 2, x / 2

    def forward(self, x):
        residual, trend = self.components(x)
        return x + self.short(residual) + self.trend(trend)
