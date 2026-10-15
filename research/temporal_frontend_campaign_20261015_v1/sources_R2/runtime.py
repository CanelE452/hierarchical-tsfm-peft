"""New-contract runtime. No legacy experiment controller or data loader imports."""
import os
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from chronos.chronos_bolt import InstanceNorm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / '.cache' / HERE.name
MODEL = ROOT / '.cache/huggingface/models--amazon--chronos-bolt-small/snapshots/772f3d25d38aec6d914c8949dab4462e2d46f5d8'
L, H = 512, 48
LAGS = (0, 1, 2, 4, 8, 16, 32)
SEEDS = (92601, 92602)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ahash(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def plain(x):
    if isinstance(x, dict): return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [plain(v) for v in x]
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Path): return str(x.relative_to(ROOT)) if x.is_relative_to(ROOT) else str(x)
    return x


def write(path, value):
    path = Path(path)
    assert path.resolve().is_relative_to(HERE) or path.resolve().is_relative_to(CACHE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(plain(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def event(**row):
    row = dict(time_unix=time.time(), **plain(row))
    with (HERE / 'EVENTS.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(row, ensure_ascii=False), flush=True)


def save(path, value):
    path = Path(path)
    assert path.resolve().is_relative_to(CACHE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f: torch.save(value, f)


def clone(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def state_hash(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class ZeroSafeSqrt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, variance):
        scale = variance.sqrt(); ctx.save_for_backward(variance, scale)
        return scale

    @staticmethod
    def backward(ctx, upstream):
        variance, scale = ctx.saved_tensors
        positive = variance > 0
        denominator = torch.where(positive, scale, torch.ones_like(scale))
        return torch.where(positive, upstream / (2 * denominator), torch.zeros_like(upstream))


class SafeInstanceNorm(InstanceNorm):
    def forward(self, x, loc_scale=None):
        dtype = x.dtype; x = x.float()
        if loc_scale is None:
            loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.)
            variance = (x-loc).square().nanmean(dim=-1, keepdim=True)
            scale = torch.nan_to_num(ZeroSafeSqrt.apply(variance), nan=1.)
            scale = torch.where(scale == 0, self.eps, scale)
        else: loc, scale = loc_scale
        y = (x-loc)/scale
        if self.use_arcsinh: y = torch.arcsinh(y)
        return y.to(dtype), (loc, scale)


class Bolt:
    def __init__(self, device='cuda'):
        from chronos import ChronosBoltPipeline
        self.pipeline = ChronosBoltPipeline.from_pretrained(str(MODEL), device_map=device,
            torch_dtype=torch.float32, local_files_only=True)
        self.model = self.pipeline.model.eval().requires_grad_(False)
        old = self.model.instance_norm
        self.model.instance_norm = SafeInstanceNorm(eps=old.eps, use_arcsinh=old.use_arcsinh)
        self.median = list(self.model.chronos_config.quantiles).index(.5)
        self.initial_hash = state_hash(self.model)
        self.calls = self.streams = 0
        self.row_microbatch = None

    def __call__(self, x, horizon=H):
        b, length, c = x.shape
        assert horizon <= self.model.chronos_config.prediction_length
        rows = x.transpose(1, 2).reshape(b*c, length)
        size = self.row_microbatch or len(rows)
        parts = []
        for i in range(0, len(rows), size):
            parts.append(self.model(context=rows[i:i+size]).quantile_preds[:, self.median, :horizon])
            self.calls += 1; self.streams += len(rows[i:i+size])
        return torch.cat(parts).reshape(b, c, horizon).transpose(1, 2)


def lag_features(z, f):
    length = z.shape[1]; horizon = f.shape[1]
    joined = torch.cat((z, f), 1)
    return torch.cat([joined[:, length-lag:length-lag+horizon]-z[:, -1-lag, None] for lag in LAGS], -1)


class Adapter(nn.Module):
    def __init__(self, front, center, basis):
        super().__init__(); self.front = front
        self.register_buffer('center', torch.as_tensor(center.copy(), dtype=torch.float32))
        self.E = nn.Parameter(torch.as_tensor(basis.copy(), dtype=torch.float32))
        r, c = basis.shape[1], basis.shape[0]
        d = torch.zeros(len(LAGS)*r, c)
        d[:r] = torch.as_tensor(np.linalg.pinv(basis).astype(np.float32))
        self.D = nn.Parameter(d)

    def encode(self, x): return (self.front(x)-self.center) @ self.E
    def decode(self, x, z, f): return x[:, -1:, :] + lag_features(z, f) @ self.D
    def forward(self, x, bolt):
        z = self.encode(x)
        return self.decode(x, z, bolt(z))


class Residual(nn.Module):
    def __init__(self, hidden=32):
        super().__init__(); self.down = nn.Linear(L, hidden); self.up = nn.Linear(hidden, H)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)

    def forward(self, x):
        delta = (x-x[:, -1:]).transpose(1, 2)
        return self.up(torch.nn.functional.gelu(self.down(delta))).transpose(1, 2)


def pca(x, r):
    c = x.shape[-1]
    if r == c: return np.zeros(c, np.float32), np.eye(c, dtype=np.float32)
    a = np.asarray(x, np.float64); center = a.mean(0)
    _, vectors = np.linalg.eigh((a-center).T @ (a-center))
    basis = vectors[:, -r:][:, ::-1].copy()
    for j in range(r):
        if basis[np.argmax(np.abs(basis[:, j])), j] < 0: basis[:, j] *= -1
    return center.astype(np.float32), basis.astype(np.float32)


def score(pred, y, mask):
    e = pred.astype(np.float64)-y.astype(np.float64)
    counts = mask.sum((0, 1)); active = counts > 0
    mse = np.where(mask, e*e, 0).sum((0, 1))/np.maximum(counts, 1)
    mae = np.where(mask, np.abs(e), 0).sum((0, 1))/np.maximum(counts, 1)
    return dict(mse=float(mse[active].mean()), mae=float(mae[active].mean()),
        channel_mse=mse.tolist(), channel_mae=mae.tolist(), counts=counts.tolist(), n_observed=int(counts.sum()))


def loss(pred, y, mask, counts=None, n=None):
    if counts is None:
        counts = mask.sum((0, 1)); n = len(y)
    active = counts > 0
    sse = torch.where(mask, (pred-y).square(), 0.).sum((0, 1))
    return n/len(y)*(sse/counts.clamp_min(1))[active].mean()


def schedule(n, updates, seed):
    # Stratification covers the complete permitted temporal range before repeats.
    rng = np.random.default_rng(seed)
    count = updates*4
    if count <= n:
        ids = np.floor((np.arange(count)+.5)*n/count).astype(np.int64)
        rng.shuffle(ids)
    else:
        ids = np.concatenate([rng.permutation(n) for _ in range((count+n-1)//n)])[:count]
    return ids.reshape(updates, 4)


@torch.no_grad()
def predict(model, bolt, x, batch=8):
    out = []
    for i in range(0, len(x), batch):
        v = torch.as_tensor(x[i:i+batch], device='cuda')
        out.append(model(v, bolt).cpu().numpy())
    return np.concatenate(out)
