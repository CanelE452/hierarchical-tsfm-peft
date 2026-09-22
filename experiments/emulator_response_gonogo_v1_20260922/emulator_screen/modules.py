"""Module surgery on real Chronos-2: compression targets, activation-aware factors, emulators,
canonical LoRA, probes, TOP3 prefix, transfer and the EMLoC-style correction port."""
from __future__ import annotations
import hashlib, math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .common import (CFG, SET, BASECFG, DEVICE, SCALING, CACHE, rc, autocast, sync, load_base,
                     lora_targets, freeze_stochastic, tensor_batch, Blocked)


def get(model, path):
    return model.get_submodule(path)


def put(model, path, module):
    parent, _, name = path.rpartition('.')
    setattr(model.get_submodule(parent) if parent else model, name, module)


def compression_targets(model):
    """Every real nn.Linear inside the encoder blocks (q/k/v/o of both attentions, MLP wi/wo)."""
    return [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and n.startswith('encoder.block.')]


def canonical_map(model):
    names = sorted(lora_targets(model))
    return names


def top3_names(names, first=SET['top3_first_trainable_block']):
    keep = []
    for n in names:
        if n.startswith('encoder.block.'):
            if int(n.split('.')[2]) >= first: keep.append(n)
        else:
            keep.append(n)
    return keep


def backbone_digest(model):
    """Hash of every non-LoRA tensor under canonical (pre-adapter) names."""
    h = hashlib.sha256(); items = []
    for n, t in list(model.named_parameters()) + list(model.named_buffers()):
        if '.lora_' in n or 'probe_' in n: continue
        items.append((n.replace('.base_layer', ''), t))
    for n, t in sorted(items, key=lambda z: z[0]):
        h.update(n.encode()); h.update(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- activation statistics and factors
def collect_second_moments(fc, model, panel, origins, targets):
    """Uncentered X^T X / n per target module input, from the ORIGINAL model on TRAIN origins, no future y."""
    acc = {n: None for n in targets}; count = {n: 0 for n in targets}; hooks = []
    for n in targets:
        def pre(mod, inp, name=n):
            with torch.autocast('cuda', enabled=False):
                x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                xtx = x.T @ x
            acc[name] = xtx if acc[name] is None else acc[name] + xtx
            count[name] += x.shape[0]
        hooks.append(get(model, n).register_forward_pre_hook(pre))
    try:
        with torch.no_grad():
            for o in origins:
                x, _, g = tensor_batch(panel, [o])
                with autocast(DEVICE, BASECFG['precision']): fc(x, g, panel.horizon)
    finally:
        for h in hooks: h.remove()
    return {n: acc[n]/count[n] for n in targets}, count


def make_factors(model, moments, targets, clock):
    """Timed part = rank choice + weighted SVD only; the reconstruction diagnostic is outside the clock."""
    factors, excluded, info = {}, {}, {}
    for n in targets:
        w = get(model, n).weight.detach().float()
        out_f, in_f = w.shape
        with clock.part('svd_factorization'):
            try:
                r = rc.factor_rank(out_f, in_f, CFG['compression_weight_entry_fraction'])
            except ValueError as e:
                excluded[n] = str(e); continue
            left, right = rc.activation_factors(w, moments[n].to(w.device), r, CFG['svd_relative_eigen_floor'])
        c = moments[n].to(w.device).double()
        err = (w.double() - left.double() @ right.double())
        rel = float(torch.sqrt(torch.einsum('oi,ij,oj->', err, c, err))/torch.sqrt(torch.einsum('oi,ij,oj->', w.double(), c, w.double())))
        factors[n] = (left.detach().cpu(), right.detach().cpu())
        info[n] = dict(out=out_f, inp=in_f, rank=r, dense_entries=out_f*in_f, factor_entries=r*(out_f+in_f),
                       weighted_relative_error=rel)
    return factors, excluded, info


class ProbeLinear(nn.Module):
    """Frozen probe branch s*B*A with a Python sign in {0,+1,-1}; never trained, never exported."""
    def __init__(self, base_layer, scaling):
        super().__init__()
        self.base_layer = base_layer
        self.scaling = float(scaling)
        self.sign = 0.
        self.register_buffer('probe_A', None); self.register_buffer('probe_B', None)
        self.in_features, self.out_features = base_layer.in_features, base_layer.out_features

    def set_probe(self, a, b):
        dev = next(iter(self.base_layer.buffers()), None)
        dev = dev.device if dev is not None else next(self.base_layer.parameters()).device
        self.probe_A = a.to(dev, torch.float32); self.probe_B = b.to(dev, torch.float32)

    def forward(self, x):
        y = self.base_layer(x)
        if self.sign == 0.: return y
        return y + self.sign*self.scaling*F.linear(F.linear(x, self.probe_A), self.probe_B)


class FrozenFactorizedLinear(nn.Module):
    """Same factored map as reference_core.FactorizedLinear with gamma frozen: exp(gamma) is folded into left once.

    Still true factored storage (two GEMMs, no dense reconstruction). Avoids the per-call BF16->FP32 promotion
    of `low * gamma.exp()` that would slow only the emulator arms (measured ~10% per step before sealing).
    """
    def __init__(self, left, right, gamma=None, bias=None):
        super().__init__()
        scaled = left if gamma is None else left*gamma.exp()[None, :]
        self.register_buffer('left', scaled.detach().clone())
        self.register_buffer('right', right.detach().clone())
        self.register_buffer('bias', None if bias is None else bias.detach().clone())
        self.out_features, self.in_features = left.shape[0], right.shape[1]

    def forward(self, x):
        return F.linear(F.linear(x, self.right), self.left, self.bias)


def build_emulator(factors, gammas=None, device=DEVICE, trainable_gamma=False):
    """Load the pinned model on CPU, replace eligible Linear layers by true factorized layers, then move to GPU.

    trainable_gamma=True -> reference_core.FactorizedLinear (calibration only); otherwise the frozen folded form
    used by every main fit, E0 cache and emulator diagnostic. The dense originals never reach the GPU.
    """
    model, load_s = load_base(BASECFG, 'cpu')
    for n, (left, right) in factors.items():
        old = get(model, n); bias = None if old.bias is None else old.bias.detach()
        if trainable_gamma:
            new = rc.FactorizedLinear(left, right, bias)
            if gammas is not None:
                with torch.no_grad(): new.gamma.copy_(gammas[n])
        else:
            new = FrozenFactorizedLinear(left, right, None if gammas is None else gammas[n], bias)
        put(model, n, new)
    model.requires_grad_(False); freeze_stochastic(model)
    model.to(device)
    return model, load_s


def load_factors():
    blob = torch.load(CACHE/'emulator'/'factors.pt', map_location='cpu', weights_only=True)
    return {n: (v['left'], v['right']) for n, v in blob.items()}


def load_gammas(kind):
    if kind == 'svd': return None
    return torch.load(CACHE/'emulator'/f'gamma_{kind}.pt', map_location='cpu', weights_only=True)


def factorized_modules(model):
    """Calibratable (gamma) factor layers only."""
    return {n: m for n, m in model.named_modules() if isinstance(m, rc.FactorizedLinear)}


def any_factor_modules(model):
    return {n: m for n, m in model.named_modules() if isinstance(m, (rc.FactorizedLinear, FrozenFactorizedLinear))}


def emulator_dense_weight(left, right, gamma=None):
    """Audit/correction only (never used in the emulator forward)."""
    if gamma is None: return left @ right
    return (left*gamma.exp()[None, :]) @ right


# ---------------------------------------------------------------- canonical LoRA
def canonical_init(seed, shapes):
    """Per-seed canonical LoRA init on original module paths: A ~ U(+-1/sqrt(in)) (PEFT/kaiming(a=sqrt5)), B = 0."""
    g = torch.Generator().manual_seed(int(seed)); state = {}
    for n in sorted(shapes):
        out_f, in_f = shapes[n]; bound = 1/math.sqrt(in_f)
        state[n+'.lora_A.weight'] = (torch.rand(CFG['lora_rank'], in_f, generator=g)*2-1)*bound
        state[n+'.lora_B.weight'] = torch.zeros(out_f, CFG['lora_rank'])
    return state


def attach_adapters(model, names):
    for n in names:
        put(model, n, rc.AdapterLinear(get(model, n), CFG['lora_rank'], CFG['lora_alpha']))
    return model


def lora_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if '.lora_' in n}


@torch.no_grad()
def load_lora(model, state, zero_missing=True):
    params = {n: p for n, p in model.named_parameters() if '.lora_' in n}
    missing = set(params) - set(state)
    if not set(state) <= set(params): raise Blocked('LORA_MAPPING_UNKNOWN_KEYS')
    for n, p in params.items():
        if n in state: p.copy_(state[n].to(p.device, p.dtype))
        elif zero_missing: p.zero_()
    return sorted(missing)


def set_trainable_lora(model, names):
    model.requires_grad_(False)
    keep = set()
    for n in names:
        ad = get(model, n)
        ad.lora_A.weight.requires_grad_(True); ad.lora_B.weight.requires_grad_(True)
        keep |= {n+'.lora_A.weight', n+'.lora_B.weight'}
    actual = {n for n, p in model.named_parameters() if p.requires_grad}
    if actual != keep: raise Blocked('TRAINABLE_SET_MISMATCH')
    return sorted(actual)


def loraplus_optimizer(model):
    aa, bb = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if '.lora_A.' in n: aa.append(p)
        elif '.lora_B.' in n: bb.append(p)
        else: raise Blocked('LORAPLUS_UNCLASSIFIED '+n)
    return torch.optim.AdamW([{'params': aa, 'lr': CFG['lr_A'], 'group_name': 'A'},
                              {'params': bb, 'lr': CFG['lr_B'], 'group_name': 'B'}],
                             betas=(.9, .999), eps=1e-8, weight_decay=CFG['weight_decay'])


def check_loraplus_official(model):
    from peft.optimizers import create_loraplus_optimizer
    ours = loraplus_optimizer(model)
    ref = create_loraplus_optimizer(model=model, optimizer_cls=torch.optim.AdamW, lr=CFG['lr_A'],
                                    loraplus_lr_ratio=CFG['lr_B']/CFG['lr_A'], betas=(.9, .999), eps=1e-8,
                                    weight_decay=CFG['weight_decay'])
    mapping = lambda opt: {id(p): float(g['lr']) for g in opt.param_groups for p in g['params']}
    if mapping(ours) != mapping(ref): raise Blocked('LORAPLUS_OFFICIAL_GROUP_MISMATCH')
    return dict(official_lr_mapping_equal=True, ratio=CFG['lr_B']/CFG['lr_A'], groups=len(ours.param_groups))


# ---------------------------------------------------------------- F_TOP3 prefix and checkpointing
def install_top3_prefix(model, first=SET['top3_first_trainable_block']):
    """Blocks < first: no_grad and detached boundary output; blocks >= first: non-reentrant checkpointing."""
    from torch.utils.checkpoint import checkpoint
    for i, block in enumerate(model.encoder.block):
        original = block.forward
        if i < first:
            def call(*args, _f=original, **kwargs):
                with torch.no_grad():
                    out = _f(*args, **kwargs)
                return type(out)(hidden_states=out.hidden_states.detach(),
                                 time_self_attn_weights=out.time_self_attn_weights,
                                 group_self_attn_weights=out.group_self_attn_weights)
        else:
            def call(*args, _f=original, **kwargs):
                if torch.is_grad_enabled(): return checkpoint(_f, *args, use_reentrant=False, **kwargs)
                return _f(*args, **kwargs)
        block.forward = call


# ---------------------------------------------------------------- probes
def make_probes(seed, names, shapes, weight_norms):
    """Two Gaussian directions x two relative scales -> 4 probes, index = 2*direction + scale_index."""
    g = torch.Generator().manual_seed(int(seed)); dirs = []
    for _ in range(CFG['probe_independent_directions']):
        d = {}
        for n in sorted(names):
            out_f, in_f = shapes[n]
            d[n] = (torch.randn(CFG['lora_rank'], in_f, generator=g), torch.randn(out_f, CFG['lora_rank'], generator=g))
        dirs.append(d)
    probes = []
    for d in dirs:
        for rel in CFG['probe_relative_weight_norms']:
            p = {}
            for n, (a, b) in d.items():
                eff = SCALING*torch.linalg.matrix_norm(b.double() @ a.double())
                p[n] = (a.clone(), (b.double()*(rel*weight_norms[n]/eff)).float())
            probes.append(p)
    return probes


def attach_probes(model, names):
    for n in names: put(model, n, ProbeLinear(get(model, n), SCALING))


def set_probe(model, probe):
    for n, (a, b) in probe.items(): get(model, n).set_probe(a, b)


def set_sign(model, names, sign):
    for n in names: get(model, n).sign = float(sign)


# ---------------------------------------------------------------- EMLoC correction port
@torch.no_grad()
def official_clamp(B_, correction, L):
    """Verbatim math of EMLoC loraCorrection.clamp (commit 332d70e), shapes [r, out]."""
    b = torch.abs(B_).mean(dim=1, keepdim=True)
    c = torch.abs(correction).mean(dim=1, keepdim=True)
    ratio = c/b
    scale = torch.where(ratio > L, L/ratio, torch.ones_like(ratio))
    return B_ - correction*scale


@torch.no_grad()
def official_get_corrected(A_peft, B_peft, w_svd, w_full, L, dtype):
    """Verbatim math of EMLoC getCorrectedLoRA with bias-free linear modules; implicitly assumes scaling 1."""
    A = A_peft.transpose(0, 1); B = B_peft.transpose(0, 1)
    A_, Sa, Va = torch.linalg.svd(A.float(), full_matrices=False)
    B_ = Sa.unsqueeze(1)*torch.matmul(Va, B.float())
    A_ = A_.to(dtype); B_ = B_.to(dtype)
    correction = F.linear(A_.transpose(0, 1), w_full.to(dtype)) - F.linear(A_.transpose(0, 1), w_svd.to(dtype))
    B_ = official_clamp(B_, correction, L)
    return A_.transpose(0, 1), B_.transpose(0, 1)


@torch.no_grad()
def corrected_state(state, names, full_model, factors, gammas=None, limit=None):
    """Per-checkpoint copy: EMLoC-style correction on every module; non-factorized modules keep W_em = W_full."""
    limit = CFG['emloc_correction_limit'] if limit is None else limit
    out = {}
    for n in names:
        a = state[n+'.lora_A.weight'].to(DEVICE).float(); b = state[n+'.lora_B.weight'].to(DEVICE).float()
        w_full = get(full_model, n).base_layer.weight.detach().float()
        if n in factors:
            left, right = factors[n]
            w_em = emulator_dense_weight(left.to(DEVICE).float(), right.to(DEVICE).float(),
                                         None if gammas is None else gammas[n].to(DEVICE).float())
        else:
            w_em = w_full
        a2, b2 = rc.emloc_correct(w_full, w_em, a, b, scaling=SCALING, correction_limit=limit)
        out[n+'.lora_A.weight'] = a2.cpu(); out[n+'.lora_B.weight'] = b2.cpu()
    return out
