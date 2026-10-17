"""Parameterless functional form of the pinned Moirai PackedStdScaler.

Derived from Salesforce uni2ts (Apache-2.0), commit
cfd46d4510ed8896f263116f32928eede05b0a75, module/packed_scaler.py.
All statistics, FP64 accumulation, correction and epsilon are preserved.
Only padding writes become out-of-place torch.where operations, preserving
the sqrt result needed by input autograd. No pretrained parameter changes.
"""

import torch
from torch import nn
from einops import reduce


def _safe_div(numerator, denominator):
    return numerator / torch.where(denominator == 0, 1.0, denominator)


class SafePackedStdScaler(nn.Module):
    def __init__(self, correction=1, minimum_scale=1e-5):
        super().__init__()
        self.correction = correction
        self.minimum_scale = minimum_scale

    def forward(self, target, observed_mask=None, sample_id=None, variate_id=None):
        if observed_mask is None:
            observed_mask = torch.ones_like(target, dtype=torch.bool)
        if sample_id is None:
            sample_id = torch.zeros(target.shape[:-1], dtype=torch.long, device=target.device)
        if variate_id is None:
            variate_id = torch.zeros(target.shape[:-1], dtype=torch.long, device=target.device)
        loc, scale = self._get_loc_scale(target.double(), observed_mask, sample_id, variate_id)
        return loc.float(), scale.float()

    def _get_loc_scale(self, target, observed_mask, sample_id, variate_id):
        id_mask = torch.logical_and(
            torch.eq(sample_id.unsqueeze(-1), sample_id.unsqueeze(-2)),
            torch.eq(variate_id.unsqueeze(-1), variate_id.unsqueeze(-2)),
        )
        tobs = reduce(
            id_mask * reduce(observed_mask, "... seq dim -> ... 1 seq", "sum"),
            "... seq1 seq2 -> ... seq1 1", "sum",
        )
        loc = reduce(
            id_mask * reduce(target * observed_mask, "... seq dim -> ... 1 seq", "sum"),
            "... seq1 seq2 -> ... seq1 1", "sum",
        )
        loc = _safe_div(loc, tobs)
        var = reduce(
            id_mask * reduce(((target - loc) ** 2) * observed_mask,
                             "... seq dim -> ... 1 seq", "sum"),
            "... seq1 seq2 -> ... seq1 1", "sum",
        )
        var = _safe_div(var, tobs - self.correction)
        scale = torch.sqrt(var + self.minimum_scale)
        padding = (sample_id == 0).unsqueeze(-1)
        loc = torch.where(padding, torch.zeros_like(loc), loc)
        scale = torch.where(padding, torch.ones_like(scale), scale)
        return loc, scale
