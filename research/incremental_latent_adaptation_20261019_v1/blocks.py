"""Two disjoint residual latent blocks with an immutable A60 front and center."""
import copy
import hashlib

import torch
from torch import nn

from research.informs_broad_validation_20261015_v1.training import macro_loss
from research.incremental_latent_adaptation_20261019_v1.models import lag_features


METHODS = ('SEQ_BLOCKS', 'JOINT_FULL', 'JOINT_PREFIX')


def frozen_state_hash(model):
    digest = hashlib.sha256()
    values = [(name, value) for name, value in model.named_parameters() if not value.requires_grad]
    values.extend(model.named_buffers())
    for name, value in sorted(values):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class LatentBlocks(nn.Module):
    def __init__(self, parent, method):
        super().__init__()
        if method not in METHODS or parent.kmax % 2:
            raise ValueError('A registered C method and two equal blocks are required')
        self.method = method
        self.length, self.horizon, self.channels = parent.length, parent.horizon, parent.channels
        self.k = self.kmax = parent.kmax
        self.block_sizes = (self.k//2, self.k//2)
        self.front = copy.deepcopy(parent.front).requires_grad_(False).eval()
        self.register_buffer('center', parent.center.detach().clone())
        self.E_blocks = nn.ParameterList(nn.Parameter(value.detach().clone())
                                         for value in parent.E.split(self.block_sizes, dim=1))
        self.D_blocks = nn.ParameterList(nn.Parameter(torch.zeros_like(value))
                                         for value in parent.D.split(self.block_sizes, dim=1))
        self.stage = None
        self.correction_only = False
        self.prefix_k = self.k
        if method == 'SEQ_BLOCKS':
            self.set_stage(0)

    def train(self, mode=True):
        super().train(mode)
        self.front.eval()
        return self

    def set_stage(self, stage, *, correction_only=False):
        if stage not in (None, 0, 1):
            raise ValueError('A C stage must be 0, 1, or joint')
        self.stage = stage
        self.correction_only = bool(correction_only)
        for parameter in self.parameters():
            parameter.grad = None
        for index in range(2):
            self.E_blocks[index].requires_grad_(stage is None or stage == index)
            self.D_blocks[index].requires_grad_(stage is None or stage == index)
        self.prefix_k = self.block_sizes[0] if stage == 0 else self.k
        return self

    def _parts(self, x, foundation, indices, block_calls=False):
        if not indices:
            return []
        if foundation is None:
            raise ValueError('A live foundation callable is required for nonzero prefixes')
        past = self.front(x) - self.center
        latent = [past @ self.E_blocks[index] for index in indices]
        if block_calls:
            future = [foundation(z)[:, :self.horizon] for z in latent]
        else:
            joined = foundation(torch.cat(latent, dim=-1))[:, :self.horizon]
            future = joined.split([self.block_sizes[index] for index in indices], dim=-1)
        return [lag_features(z, u) @ self.D_blocks[index].reshape(7*self.block_sizes[index], self.channels)
                for index, z, u in zip(indices, latent, future)]

    def correction_for_prefix(self, x, prefix_k, foundation=None, *, block_calls=False):
        if prefix_k not in (0, self.block_sizes[0], self.k):
            raise ValueError('Only the registered zero, half, and full prefixes are supported')
        if prefix_k == 0:
            return x.new_zeros((len(x), self.horizon, self.channels))
        count = 1 if prefix_k == self.block_sizes[0] else 2
        values = self._parts(x, foundation, list(range(count)), block_calls)
        return values[0] if count == 1 else values[0] + values[1]

    def forward(self, x, foundation=None):
        if self.correction_only:
            return self._parts(x, foundation, [self.stage])[0]
        return self.correction_for_prefix(x, self.prefix_k, foundation)

    def training_loss(self, x, g, target, mask, foundation, counts, n_fit, batch_rows):
        if self.method == 'JOINT_PREFIX':
            first, second = self._parts(x, foundation, [0, 1])
            return (.5 * macro_loss(g+first, target, mask, counts, n_fit, batch_rows)
                    + .5 * macro_loss(g+first+second, target, mask, counts, n_fit, batch_rows))
        return macro_loss(g+self(x, foundation), target, mask, counts, n_fit, batch_rows)


def restore(parent, path, *, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved['schema'] != 'incremental_C_final_model_v1':
        raise ValueError('A final registered C model is required')
    model = LatentBlocks(parent, saved['method'])
    model.load_state_dict(saved['model'], strict=True)
    model.set_stage(None)
    model.prefix_k = int(saved['selected_prefix'])
    return model.to(device).eval().requires_grad_(False)
