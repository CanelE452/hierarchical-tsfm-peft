"""Small campaign variants, not paper benchmark reproductions.

Official source mapping, read at pinned revisions on 2026-10-15:
Google TSMixer: alternating temporal and feature residual mixing, then L->H.
https://github.com/google-research/google-research/blob/5b09c22d73a9d35eb6c5d2a99b95677a45053466/tsmixer/tsmixer_basic/models/tsmixer.py
TimeMixer: mean-pooled scales, symmetric moving-average decomposition,
fine->coarse seasonal/residual and coarse->fine trend mixing in PDM; FMM
uses separate per-scale future heads.
https://github.com/kwuking/TimeMixer/blob/e24610583b36fdd8c76cc17a8df4e65759a5f460/models/TimeMixer.py
https://github.com/kwuking/TimeMixer/blob/e24610583b36fdd8c76cc17a8df4e65759a5f460/layers/Autoformer_EncDec.py

Variations: one block, GELU, width-limited temporal MLPs, no dropout, no
learned embedding/RevIN, and a shared feature MLP in the original C space.
B2 replaces Google's full L->L temporal dense with L->hidden->L. B4 front
replaces FMM with zero-initialized residual heads onto the original L grid;
its direct counterpart retains per-scale L_s->H heads. These choices do
not reproduce the papers' training setups, full architectures, or results.

Every operation sees only the supplied history [origin-L, origin), but can
use that entire window. These representations are NOT prefix causal.
The symmetric B4 smoother differs from R1's trailing-only smoother.
Pooling uses nonoverlapping blocks starting at history row zero. Learned
front heads map each scale onto original row indices 0..L-1; coarse rows
are never concatenated as if they had the original sampling interval.

CP0 front outputs equal their input. Only final residual-output linears
are zero initialized; their input linears and all cross-scale/core layers
are nonzero initialized. On the first backward, output layers can receive
nonzero gradients while earlier layers have zero gradients through the
zero output weights. After one nonzero output-layer update the interior
can receive gradients. B2's two zero outputs sit on separate residual
branches, not in a chain. B3/B4 have one zero output per parallel scale.
"""

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def parameter_count(module: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


class _MLP(nn.Module):
    def __init__(self, n_in: int, hidden: int, n_out: int, zero_output: bool = False):
        super().__init__()
        self.input = nn.Linear(n_in, hidden)
        self.output = nn.Linear(hidden, n_out)
        if zero_output:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, x):
        return self.output(F.gelu(self.input(x)))


class _AlternatingMixer(nn.Module):
    def __init__(self, length: int, channels: int, hidden: int):
        super().__init__()
        self.temporal_norm = nn.LayerNorm((length, channels), elementwise_affine=False)
        self.feature_norm = nn.LayerNorm((length, channels), elementwise_affine=False)
        self.temporal = _MLP(length, hidden, length, zero_output=True)
        self.feature = _MLP(channels, hidden, channels, zero_output=True)

    def forward(self, x):
        temporal = self.temporal(self.temporal_norm(x).transpose(1, 2)).transpose(1, 2)
        y = x + temporal
        return y + self.feature(self.feature_norm(y))


class _MultiscalePast(nn.Module):
    def __init__(self, length, channels, hidden, scales, kernel,
                 decomposition, seasonal_mix, trend_mix):
        super().__init__()
        self.scales = tuple(scales)
        self.lengths = tuple(length // s for s in self.scales)
        self.kernel = kernel
        self.decomposition = decomposition
        self.channel = _MLP(channels, hidden, channels)
        self.down = nn.ModuleList(
            _MLP(a, hidden, b) for a, b in zip(self.lengths, self.lengths[1:])
        ) if seasonal_mix else nn.ModuleList()
        self.up = nn.ModuleList(
            _MLP(b, hidden, a) for a, b in zip(self.lengths, self.lengths[1:])
        ) if trend_mix else nn.ModuleList()

    def scale_inputs(self, x):
        return [x if s == 1 else F.avg_pool1d(x.transpose(1, 2), s, s).transpose(1, 2)
                for s in self.scales]

    def decompose(self, x):
        if not self.decomposition:
            # Both directional paths retain real input and live parameters.
            return 0.5 * x, 0.5 * x
        padding = (self.kernel - 1) // 2
        trend = F.avg_pool1d(F.pad(x.transpose(1, 2), (padding, padding), mode="replicate"),
                             self.kernel, stride=1).transpose(1, 2)
        return x - trend, trend

    def forward(self, x):
        seasonal, trends = [], []
        for value in self.scale_inputs(x):
            residual, trend = self.decompose(value)
            seasonal.append((residual + self.channel(residual)).transpose(1, 2))
            trends.append((trend + self.channel(trend)).transpose(1, 2))
        if self.down:
            for i, down in enumerate(self.down):
                seasonal[i + 1] = seasonal[i + 1] + down(seasonal[i])
        if self.up:
            for i in reversed(range(len(self.up))):
                trends[i] = trends[i] + self.up[i](trends[i + 1])
        return [a + b for a, b in zip(seasonal, trends)]


def _validate_configuration(length, channels, hidden, scales, kernel):
    if length < 1 or channels < 1 or hidden < 1:
        raise ValueError("length, channels and hidden must be positive")
    if (not scales or scales[0] != 1 or tuple(scales) != tuple(sorted(set(scales)))
            or any(not isinstance(s, int) or s < 1 or length % s for s in scales)):
        raise ValueError("scales must be increasing integer divisors of length, starting with 1")
    if not isinstance(kernel, int) or kernel < 1 or kernel % 2 == 0:
        raise ValueError("kernel must be a positive odd integer")


def _validate_input(x, length, channels):
    if x.ndim != 3 or x.shape[1:] != (length, channels):
        raise ValueError(f"expected [batch, {length}, {channels}], got {tuple(x.shape)}")


class TemporalFront(nn.Module):
    """Past->past front: [B,L,C] -> [B,L,C], preserving original row indices.

    B0: identity. B1: channel-shared temporal residual. B2: alternating
    temporal/feature residuals. B3/B4: identical-capacity bidirectional
    multiscale mixing, with raw half-signals / residual-trend decomposition.

    B4 decomposition=False is exactly the B3 structural control. The two
    *_mix=False options remove that direction's cross-scale MLPs while
    preserving the unmixed component at every scale. Use fresh matched
    fitting for ablations; toggling after a fit is not a refit comparison.
    """

    def __init__(self, kind: str, length: int = 512, channels: int = 1,
                 hidden: int = 32, scales: Sequence[int] = (1, 2, 4), kernel: int = 25,
                 *, decomposition: bool = True, seasonal_mix: bool = True,
                 trend_mix: bool = True):
        super().__init__()
        _validate_configuration(length, channels, hidden, scales, kernel)
        if kind not in ("B0", "B1", "B2", "B3", "B4"):
            raise ValueError(f"unknown temporal-front kind {kind}")
        self.kind, self.length, self.channels = kind, length, channels
        self.hidden, self.scales, self.kernel = hidden, tuple(scales), kernel
        if kind == "B1":
            self.temporal = _MLP(length, hidden, length, zero_output=True)
        elif kind == "B2":
            self.mixer = _AlternatingMixer(length, channels, hidden)
        elif kind in ("B3", "B4"):
            self.multiscale = _MultiscalePast(
                length, channels, hidden, scales, kernel,
                decomposition=kind == "B4" and decomposition,
                seasonal_mix=seasonal_mix, trend_mix=trend_mix,
            )
            self.fusion = nn.ModuleList(
                _MLP(n, hidden, length, zero_output=True) for n in self.multiscale.lengths
            )

    def forward(self, x):
        _validate_input(x, self.length, self.channels)
        if self.kind == "B0":
            return x
        if self.kind == "B1":
            return x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        if self.kind == "B2":
            return self.mixer(x)
        values = self.multiscale(x)
        delta = sum(head(value) for head, value in zip(self.fusion, values)) / len(values)
        return x + delta.transpose(1, 2)


class DirectMixer(nn.Module):
    """Small direct forecasting variations: [B,L,C] -> [B,H,C].

    B2 uses the alternating mixer then an L->H head. B4 uses the same PDM
    core with one L_s->H head per scale, followed by their sum (FMM
    principle). Heads are normally initialized: direct CP0 is not an
    identity forecast, and B4's core can receive first-backward gradients.
    Neither model includes the papers' full embedding/normalization setup.
    """

    def __init__(self, kind: str, length: int = 512, horizon: int = 48,
                 channels: int = 1, hidden: int = 32,
                 scales: Sequence[int] = (1, 2, 4), kernel: int = 25,
                 *, decomposition: bool = True, seasonal_mix: bool = True,
                 trend_mix: bool = True):
        super().__init__()
        _validate_configuration(length, channels, hidden, scales, kernel)
        if kind not in ("B2", "B4") or horizon < 1:
            raise ValueError("DirectMixer requires kind B2/B4 and a positive horizon")
        self.kind, self.length, self.horizon, self.channels = kind, length, horizon, channels
        if kind == "B2":
            self.front = TemporalFront(kind, length, channels, hidden, scales, kernel)
            self.head = nn.Linear(length, horizon)
        else:
            self.multiscale = _MultiscalePast(
                length, channels, hidden, scales, kernel, decomposition, seasonal_mix, trend_mix
            )
            self.heads = nn.ModuleList(nn.Linear(n, horizon) for n in self.multiscale.lengths)

    def forward(self, x):
        _validate_input(x, self.length, self.channels)
        if self.kind == "B2":
            return self.head(self.front(x).transpose(1, 2)).transpose(1, 2)
        values = self.multiscale(x)
        return sum(head(value) for head, value in zip(self.heads, values)).transpose(1, 2)


# INFORMS additions. The TemporalFront/DirectMixer definitions above retain the
# explicitly documented campaign variations and original official source pins.
LAGS = (0, 1, 2, 4, 8, 16, 32)
ENCODER_LAGS = {"causal7": LAGS, "bidirectional7": (-16, -8, -4, 0, 4, 8, 16)}


class ForecastAdapter(nn.Module):
    """Original-grid P/E -> callable([B,L,K]) -> anchored lag7 D.

    D is lag-major [7,Kmax,C]. Only the requested prefix is passed to F.
    Lag encoders pad the supplied standardized history with zero; no extra
    pre-window samples are read. Causal encoders require the identity front.
    """
    def __init__(self, front_kind, length, horizon, channels, latent_k,
                 pca_center, basis, encoder_kind="pointwise", k_grid=None):
        super().__init__()
        if horizon < 1 or length <= max(LAGS) or not 1 <= latent_k <= channels:
            raise ValueError("Require positive H, L>32 and 1<=Kmax<=C")
        if encoder_kind not in ("pointwise", "causal7", "bidirectional7"):
            raise ValueError("Unknown encoder kind")
        if encoder_kind != "pointwise" and front_kind != "B0":
            raise ValueError("Lag encoder comparisons require B0; a full-window front is not prefix causal")
        self.front = TemporalFront(front_kind, length=length, channels=channels)
        self.length, self.horizon, self.channels = int(length), int(horizon), int(channels)
        self.latent_k = self.kmax = int(latent_k)
        self.encoder_kind = encoder_kind
        self.k_grid = tuple(sorted(set(k_grid))) if k_grid is not None else None
        if self.k_grid is not None and (not self.k_grid or any(not isinstance(k,int) or not 1<=k<=latent_k for k in self.k_grid)):
            raise ValueError("Invalid registered K grid")
        center = torch.as_tensor(pca_center, dtype=torch.float32).detach().clone()
        vectors = torch.as_tensor(basis, dtype=torch.float32).detach().clone()
        if center.shape != (channels,) or vectors.ndim != 2 or vectors.shape[0] != channels or vectors.shape[1] < latent_k:
            raise ValueError("PCA center/basis shapes do not match C/Kmax")
        if not torch.isfinite(center).all() or not torch.isfinite(vectors).all():
            raise ValueError("Nonfinite PCA initialization")
        self.register_buffer("center", center)
        self.E = nn.Parameter(vectors[:, :latent_k].contiguous())
        d = torch.zeros(7, latent_k, channels)
        d[0] = torch.linalg.pinv(self.E.detach())
        self.D = nn.Parameter(d)
        if encoder_kind != "pointwise":
            self.lag_weights = nn.Parameter(torch.zeros(6, channels, latent_k))

    @property
    def pca_center(self):
        return self.center

    def _k(self, value):
        value = self.latent_k if value is None else value
        if not isinstance(value, int) or not 1 <= value <= self.kmax or (self.k_grid is not None and value not in self.k_grid):
            raise ValueError("K must be a registered prefix within Kmax")
        return value

    def encode(self, x, latent_k=None):
        _validate_input(x, self.length, self.channels)
        k = self._k(latent_k)
        past = self.front(x)
        z = (past-self.center) @ self.E[:, :k]
        if self.encoder_kind != "pointwise":
            other_lags = [lag for lag in ENCODER_LAGS[self.encoder_kind] if lag != 0]
            for index, lag in enumerate(other_lags):
                if lag > 0:
                    shifted = F.pad(past.transpose(1,2), (lag,0))[:, :, :self.length].transpose(1,2)
                else:
                    shifted = F.pad(past.transpose(1,2), (0,-lag))[:, :, -lag:-lag+self.length].transpose(1,2)
                z = z + (shifted-self.center) @ self.lag_weights[index, :, :k]
        return z

    def decode(self, x, z, future, latent_k=None):
        k = self._k(latent_k)
        if tuple(z.shape) != (x.shape[0],self.length,k) or tuple(future.shape) != (x.shape[0],self.horizon,k):
            raise ValueError("Foundation output must be [B,H,requested K]")
        joined = torch.cat((z,future),1)
        features = torch.cat([joined[:,self.length-lag:self.length-lag+self.horizon]-z[:, -1-lag, None]
                              for lag in LAGS], -1)
        return x[:, -1:] + features @ self.D[:, :k, :].reshape(7*k,self.channels)

    def forward(self, x, foundation_callable, latent_k=None):
        k = self._k(latent_k)
        z = self.encode(x, k)
        future = foundation_callable(z)
        if not isinstance(future, torch.Tensor):
            raise ValueError("Foundation callable must return a tensor")
        return self.decode(x,z,future,k)


class DirectG(nn.Module):
    """Channel-shared L->hidden->H innovation plus original current value."""
    def __init__(self, length, horizon, hidden=32):
        super().__init__()
        if min(length,horizon,hidden)<1:
            raise ValueError("Positive L/H/hidden required")
        self.length,self.horizon=int(length),int(horizon)
        self.input = nn.Linear(length,hidden)
        self.output = nn.Linear(hidden,horizon)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,x):
        if x.ndim!=3 or x.shape[1]!=self.length:
            raise ValueError("Expected [B,L,C] history")
        current=x[:, -1:]
        delta=self.output(F.gelu(self.input((x-current).transpose(1,2)))).transpose(1,2)
        return current+delta


class ConditionalHead(nn.Module):
    """g + H(past-current,g-current,b-g); control substitutes actual past H.

    Parent forecast tensors are detached. Zero output initialization preserves g
    only at initialization; it is not a post-training non-degradation guarantee.
    """
    def __init__(self,length,horizon,hidden=32):
        super().__init__()
        if min(length,horizon,hidden)<1 or length<horizon:
            raise ValueError("Require positive dimensions and L>=H")
        self.length,self.horizon=int(length),int(horizon)
        self.input=nn.Linear(length+2*horizon,hidden)
        self.output=nn.Linear(hidden,horizon)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,x,g,b,control=False):
        if x.ndim!=3 or x.shape[1]!=self.length or tuple(g.shape)!=(x.shape[0],self.horizon,x.shape[2]) or b.shape!=g.shape:
            raise ValueError("History/parent forecast shapes disagree")
        g,b=g.detach(),b.detach()
        current=x[:, -1:]
        condition=x[:, -self.horizon:]-current if control else b-g
        features=torch.cat(((x-current).transpose(1,2),(g-current).transpose(1,2),condition.transpose(1,2)),-1)
        return g+self.output(F.gelu(self.input(features))).transpose(1,2)

    def forecastforward(self,x,g,b,control=False):
        return self.forward(x,g,b,control=control)


class AdaPTSLinearPort(nn.Module):
    """Deterministic LinearAutoEncoder port, RevIN=False, patch size=1.

    Source: abenechehab/AdaPTS @ 8bf57c7ee3b97bfd3f1852ad8dc8d0695a806278,
    src/adapts/adapters.py LinearAutoEncoder and MultichannelProjector;
    src/adapts/adapts.py adapter_supervised_fine_tuning. This preserves its
    affine pointwise encoder/decoder, not the current anchor/lag7 decoder.
    Root supplies the new macro loss, chronology and differentiable Bolt.

    MIT License. Copyright (c) 2025 Abdelhakim Benechehab.
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
    IN THE SOFTWARE.
    """
    def __init__(self,length,horizon,channels,latent_k):
        super().__init__()
        if min(length,horizon,channels,latent_k)<1 or latent_k>channels:
            raise ValueError("Invalid L/H/C/K")
        self.length,self.horizon,self.channels,self.latent_k=length,horizon,channels,latent_k
        self.encoder=nn.Linear(channels,latent_k)
        self.decoder=nn.Linear(latent_k,channels)

    def forward(self,x,foundation_callable):
        _validate_input(x,self.length,self.channels)
        future=foundation_callable(self.encoder(x))
        if not isinstance(future,torch.Tensor) or tuple(future.shape)!=(x.shape[0],self.horizon,self.latent_k):
            raise ValueError("Foundation output must be [B,H,K]")
        return self.decoder(future)


def module_state_hash(module):
    """Byte-exact CPU digest; handles BF16 and scalar buffers without casting."""
    import hashlib
    result=hashlib.sha256()
    for key,value in module.state_dict().items():
        result.update(key.encode());result.update(str(value.dtype).encode());result.update(str(tuple(value.shape)).encode())
        result.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


class _LowRankUpdate(nn.Module):
    def __init__(self,linear,rank,alpha):
        super().__init__()
        self.A=nn.Parameter(torch.empty(rank,linear.in_features,device=linear.weight.device,dtype=linear.weight.dtype))
        self.B=nn.Parameter(torch.zeros(linear.out_features,rank,device=linear.weight.device,dtype=linear.weight.dtype))
        nn.init.kaiming_uniform_(self.A,a=5**.5)
        self.scale=float(alpha)/rank

    def forward(self,x):
        return F.linear(F.linear(x,self.A),self.B)*self.scale


class LoRAQvAdapter(nn.Module):
    """Adapter-only q/v state; caller retains and invokes the frozen base.

    Merge removes hooks. Unmerge/close copy saved original target weights,
    avoiding FP32 subtract-roundoff. Close also restores original grad flags.
    """
    def __init__(self,base_model,rank=8,alpha=16.0,dropout=0.0):
        super().__init__()
        if not isinstance(rank,int) or rank<1 or alpha<=0 or dropout!=0:
            raise ValueError("Require positive rank/alpha and dropout=0")
        targets=[(name,module) for name,module in base_model.named_modules()
                 if name.split('.')[-1] in ('q','v','q_proj','v_proj') and isinstance(module,nn.Linear)]
        if not targets:raise ValueError("No q/v linear targets found")
        if any(parameter.grad is not None for parameter in base_model.parameters()):
            raise ValueError("Base has existing gradients")
        object.__setattr__(self,'_base',base_model)
        self.base_hash=module_state_hash(base_model)
        self.target_names=tuple(name for name,_ in targets)
        self._targets=targets
        self._originals=[module.weight.detach().clone() for _,module in targets]
        self._grad_flags=[(parameter,parameter.requires_grad) for parameter in base_model.parameters()]
        self._handles=[];self._merged=False;self._closed=False
        self.adapters=nn.ModuleDict({str(i):_LowRankUpdate(module,rank,alpha) for i,(_,module) in enumerate(targets)})
        try:
            for parameter,_ in self._grad_flags:parameter.requires_grad_(False)
            self._attach_hooks()
        except Exception:
            self.close()
            raise

    def _attach_hooks(self):
        if self._handles:return
        for i,(_,module) in enumerate(self._targets):
            def hook(module,args,output,index=str(i)):
                return output+self.adapters[index](args[0])
            self._handles.append(module.register_forward_hook(hook))

    def _remove_hooks(self):
        for handle in self._handles:handle.remove()
        self._handles=[]

    def adapter_parameters(self):
        return self.adapters.parameters()

    def adapter_state_dict(self):
        return {key:value.detach().clone() for key,value in self.state_dict().items()}

    def load_adapter_state_dict(self,state):
        if self._closed or self._merged:raise RuntimeError("Load requires an open unmerged adapter")
        self.load_state_dict(state,strict=True)

    def merge(self):
        if self._closed:raise RuntimeError("Adapter is closed")
        if not self._merged:
            with torch.no_grad():
                for i,((_,module),original) in enumerate(zip(self._targets,self._originals)):
                    if not torch.equal(module.weight,original):raise RuntimeError("Base target weight changed")
                    update=self.adapters[str(i)]
                    module.weight.copy_(original+(update.B@update.A)*update.scale)
            self._remove_hooks();self._merged=True
        return self

    def unmerge(self):
        if self._closed:raise RuntimeError("Adapter is closed")
        if self._merged:
            with torch.no_grad():
                for (_,module),original in zip(self._targets,self._originals):module.weight.copy_(original)
            self._merged=False;self._attach_hooks()
            if module_state_hash(self._base)!=self.base_hash:raise RuntimeError("Base hash changed outside LoRA targets")
        return self

    def close(self):
        if self._closed:return
        self._remove_hooks()
        with torch.no_grad():
            for (_,module),original in zip(self._targets,self._originals):module.weight.copy_(original)
        for parameter,flag in self._grad_flags:parameter.requires_grad_(flag)
        self._merged=False;self._closed=True
        if module_state_hash(self._base)!=self.base_hash:raise RuntimeError("Base hash restoration failed")

    def __enter__(self):return self

    def __exit__(self,exc_type,exc_value,traceback):
        self.close()
        return False
