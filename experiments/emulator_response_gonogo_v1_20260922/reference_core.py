"""Small, CPU-testable components for a Chronos emulator go/no-go screen.

Not a Chronos runner. Integrate with the existing gap_screen only after real-model
parity checks. Tensor convention: weight[out, in], LoRA = scaling * B @ A.
Algorithm references and the scope of the EMLoC port are in MASTER_CLI.txt.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Callable
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _finite(x: Tensor, name: str) -> None:
    if not bool(torch.isfinite(x).all()):
        raise ValueError(f"nonfinite {name}")


def factor_rank(out_features: int, in_features: int, keep_fraction: float = .5) -> int:
    if min(out_features, in_features) < 2 or not 0 < keep_fraction < 1:
        raise ValueError("invalid shape or keep_fraction")
    r = max(1, int(keep_fraction * out_features * in_features / (out_features + in_features)))
    if r * (out_features + in_features) >= out_features * in_features:
        raise ValueError("factorization would not save weight entries")
    return min(r, out_features, in_features)


@torch.no_grad()
def activation_factors(weight: Tensor, second_moment: Tensor, rank: int,
                       relative_floor: float = 1e-7) -> tuple[Tensor, Tensor]:
    """Uncentered activation-aware SVD. Returns left[out,r], right[r,in].

    Minimize ||(W - L R) C^(1/2)||_F by a weighted SVD with a positive
    eigenvalue floor. C must come ONLY from TRAIN input activations.
    FP64 is useful for unit tests; production FP32 work must be timed.
    """
    if weight.ndim != 2 or second_moment.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError("shape mismatch")
    if not 1 <= rank <= min(weight.shape):
        raise ValueError("invalid rank")
    _finite(weight, "weight"); _finite(second_moment, "second moment")
    dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
    w = weight.to(dtype)
    c = second_moment.to(device=w.device, dtype=dtype)
    c = (c + c.T) * .5
    mean_eig = c.trace() / c.shape[0]
    if float(mean_eig) <= 0:
        raise ValueError("no usable activation variance")
    eig, vec = torch.linalg.eigh(c / mean_eig)
    eig = eig.clamp_min(relative_floor) * mean_eig
    root = vec * eig.sqrt()[None, :]
    inv_root = (vec / eig.sqrt()[None, :]).T
    u, s, vh = torch.linalg.svd(w @ root, full_matrices=False)
    sroot = s[:rank].sqrt()
    left = u[:, :rank] * sroot[None, :]
    right = sroot[:, None] * (vh[:rank] @ inv_root)
    return left.to(weight.dtype), right.to(weight.dtype)


class FactorizedLinear(nn.Module):
    """True factored storage, NOT a dense reconstructed weight in forward.

    gamma exists only for limited emulator calibration. Freeze it before the
    downstream LoRA fit; NEVER export gamma or these factors to the full model.
    """
    def __init__(self, left: Tensor, right: Tensor, bias: Tensor | None = None):
        super().__init__()
        if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
            raise ValueError("incompatible factors")
        self.out_features, self.in_features = left.shape[0], right.shape[1]
        self.register_buffer("left", left.detach().clone())
        self.register_buffer("right", right.detach().clone())
        self.register_buffer("bias", None if bias is None else bias.detach().clone())
        self.gamma = nn.Parameter(left.new_zeros(left.shape[1]), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        low = F.linear(x, self.right)
        return F.linear(low * self.gamma.exp(), self.left, self.bias)

    @torch.no_grad()
    def dense_weight_for_audit_only(self) -> Tensor:
        return (self.left * self.gamma.exp()[None, :]) @ self.right

    def set_calibration(self, enabled: bool) -> None:
        self.gamma.requires_grad_(enabled)

    @torch.no_grad()
    def clamp_calibration(self) -> None:
        self.gamma.clamp_(-math.log(2), math.log(2))


class AdapterLinear(nn.Module):
    """Same LoRA input/output interface on full or factored frozen bases."""
    def __init__(self, base_layer: nn.Module, rank: int = 8, alpha: float = 16.):
        super().__init__()
        self.base_layer = base_layer
        base_layer.requires_grad_(False)
        self.in_features = int(base_layer.in_features)
        self.out_features = int(base_layer.out_features)
        self.scaling = float(alpha / rank)
        prototype = next(iter(base_layer.parameters()), None)
        if prototype is None:
            prototype = next(iter(base_layer.buffers()))
        kwargs = dict(device=prototype.device, dtype=prototype.dtype)
        self.lora_A = nn.Linear(self.in_features, rank, bias=False, **kwargs)
        self.lora_B = nn.Linear(rank, self.out_features, bias=False, **kwargs)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.base_layer(x) + self.scaling * self.lora_B(self.lora_A(x))


def anchored_raw(full_zero: Tensor, emulator_now: Tensor, emulator_zero: Tensor) -> Tensor:
    """Raw-unit forecast, with stop-gradient only on cached zero forecasts."""
    if full_zero.shape != emulator_now.shape or full_zero.shape != emulator_zero.shape:
        raise ValueError("forecast shape mismatch")
    return full_zero.detach() + (emulator_now - emulator_zero.detach())


def raw_to_native(raw: Tensor, loc: Tensor, scale: Tensor, arcsinh: bool = True) -> Tensor:
    if bool((scale <= 0).any()):
        raise ValueError("nonpositive scale")
    z = (raw.float() - loc[:, None, :]) / scale[:, None, :]
    return z.asinh() if arcsinh else z


def native_loss_from_raw(raw: Tensor, target: Tensor, loc: Tensor, scale: Tensor,
                         quantiles: Tensor, arcsinh: bool = True) -> Tensor:
    """Mean H / sum Q / mean series, for whole output patches.

    target poison/NaN is removed BEFORE evaluating the loss. This local formula
    still requires a real-model parity test against the pinned Chronos version.
    """
    q = raw_to_native(raw, loc, scale, arcsinh)
    good = torch.isfinite(target)
    safe = torch.where(good, target, loc.expand_as(target))
    y = (safe.float() - loc) / scale
    if arcsinh: y = y.asinh()
    e = y[:, None, :] - q
    tau = quantiles.to(q)[None, :, None]
    per = 2 * torch.maximum(tau * e, (tau - 1) * e)
    return torch.where(good[:, None, :], per, 0.).mean(-1).sum(-1).mean()


def weighted_mse(delta: Tensor, sigma: Tensor) -> Tensor:
    """delta: [series, quantile, horizon], sigma: [series]."""
    if delta.ndim != 3 or sigma.shape != (delta.shape[0],):
        raise ValueError("response tensor / scale shape mismatch")
    if bool((sigma <= 0).any()): raise ValueError("invalid TRAIN sigma")
    return ((delta.float() / sigma.to(delta).float()[:, None, None]) ** 2).mean()


def response_loss(e_plus: Tensor, e_minus: Tensor, f_plus: Tensor, f_minus: Tensor,
                  sigma: Tensor, eps: float = 1e-8) -> Tensor:
    """Symmetric finite response, not an exact full Jacobian match."""
    df = ((f_plus - f_minus) * .5).detach()
    de = (e_plus - e_minus) * .5
    return weighted_mse(de - df, sigma) / (weighted_mse(df, sigma).detach() + eps)


def value_loss(e_zero: Tensor, f_zero: Tensor, initial_error: Tensor,
               sigma: Tensor, eps: float = 1e-8) -> Tensor:
    return weighted_mse(e_zero - f_zero.detach(), sigma) / (initial_error.detach() + eps)


@torch.no_grad()
def emloc_correct(weight_full: Tensor, weight_emulator: Tensor, a: Tensor, b: Tensor,
                  scaling: float = 2., correction_limit: float = 3.) -> tuple[Tensor, Tensor]:
    """EMLoC-style LoRA correction with explicit PEFT scaling.

    Based on official getCorrectedLoRA/clamp equations, but this is our tensor
    implementation, not a reproduction of the original VLM experiments.
    Clamp the mean absolute correction of EACH LoRA basis vector. Infinite L
    gives exact matching of linear maps on span(A.T). No bias mismatch allowed.
    """
    if weight_full.shape != weight_emulator.shape:
        raise ValueError("base weights incompatible")
    if weight_full.shape != (b.shape[0], a.shape[1]) or b.shape[1] != a.shape[0]:
        raise ValueError("LoRA shape mismatch")
    if scaling <= 0 or correction_limit < 0:
        raise ValueError("invalid scale / limit")
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError("nonfinite LoRA")
    if torch.count_nonzero(b) == 0:
        return a.clone(), b.clone()  # identity initial forecast; avoid 0/0.
    dtype = torch.float64 if a.dtype == torch.float64 else torch.float32
    aa, bb = a.to(dtype), b.to(dtype)
    u, singular, vh = torch.linalg.svd(aa.T, full_matrices=False)
    a_new = u.T
    b_effective = scaling * (bb @ vh.T) * singular[None, :]
    difference = (weight_full.to(dtype) - weight_emulator.to(dtype)) @ u
    if math.isinf(correction_limit):
        correction = difference
    else:
        bnorm = b_effective.abs().mean(0, keepdim=True)
        dnorm = difference.abs().mean(0, keepdim=True)
        tiny = torch.finfo(dtype).tiny
        factor = torch.minimum(torch.ones_like(dnorm), correction_limit * bnorm / dnorm.clamp_min(tiny))
        correction = difference * factor
    return a_new.to(a.dtype), ((b_effective - correction) / scaling).to(b.dtype)


@dataclass(frozen=True)
class FixedBudgetResult:
    arm: str
    seed: int
    selected_step: int
    test_error: float
    full_loop_seconds: float
    preprocessing_seconds: float
    transfer_seconds: float

    @property
    def total_seconds(self) -> float:
        # full_loop includes ALL checkpoints through the fixed budget, not a
        # retrospective estimate up to the hindsight-best checkpoint.
        return self.preprocessing_seconds + self.full_loop_seconds + self.transfer_seconds


def common_quality_target(full_validation: list[float], f0_validation: float,
                          tolerance: float = .005) -> tuple[float, bool]:
    if not full_validation or not all(math.isfinite(v) for v in full_validation):
        raise ValueError("invalid validation reference")
    target = min(full_validation) * (1 + tolerance)
    return target, f0_validation <= target


def outcome(full: list[FixedBudgetResult], candidate: list[FixedBudgetResult],
            controls: dict[str, list[FixedBudgetResult]], min_speedup: float = .20,
            quality_margin: float = .005, min_response_gain: float = .003) -> dict:
    """Project investment screen, NOT significance/equivalence or publication PASS.

    controls must include VALUE_DELTA. Report all points. No test-driven refits.
    """
    seeds = sorted(x.seed for x in full)
    if len(seeds) != 2 or sorted(x.seed for x in candidate) != seeds:
        raise ValueError("exactly two matched seeds required")
    for a, points in controls.items():
        if sorted(x.seed for x in points) != seeds: raise ValueError(f"bad seeds: {a}")
    if "VALUE_DELTA" not in controls: raise ValueError("missing value-matched control")
    mean = lambda pp, attr: sum(getattr(p, attr) for p in pp) / len(pp)
    ef, ec = mean(full, "test_error"), mean(candidate, "test_error")
    tf, tc = mean(full, "total_seconds"), mean(candidate, "total_seconds")
    if min(ef, ec, tf, tc) <= 0: raise ValueError("invalid error / cost")
    ev = mean(controls["VALUE_DELTA"], "test_error")
    cg = (ev - ec) / ev
    by_seed = lambda pp: {p.seed: p for p in pp}
    fc, cc, vc = by_seed(full), by_seed(candidate), by_seed(controls["VALUE_DELTA"])
    both_quality = all(cc[s].test_error <= (1 + quality_margin) * fc[s].test_error for s in seeds)
    both_faster = all(cc[s].total_seconds < fc[s].total_seconds for s in seeds)
    both_response = all(cc[s].test_error < vc[s].test_error for s in seeds)
    # A control with no worse quality and no greater cost already dominates it.
    dominated = [a for a, pp in controls.items()
                 if mean(pp,"test_error") <= ec and mean(pp,"total_seconds") <= tc]
    keep_quality = ec <= ef * (1 + quality_margin)
    save_cost = tc <= tf * (1 - min_speedup)
    added_value = cg >= min_response_gain and both_response
    if dominated:
        status = "GO_STANDARD_ONLY" if any(
            mean(controls[a],"test_error") <= ef*(1+quality_margin)
            and mean(controls[a],"total_seconds") <= tf*(1-min_speedup) for a in dominated
        ) else "NO_GO_CURRENT_IMPLEMENTATION"
    elif keep_quality and save_cost and added_value and both_quality and both_faster:
        status = "GO_METHOD_SCREEN"
    elif not keep_quality or not save_cost:
        status = "NO_GO_CURRENT_IMPLEMENTATION"
    else:
        status = "HOLD_NO_AUTO_RESCUE"
    return dict(status=status, relative_quality_loss_pct=100*(ec/ef-1),
                total_time_saved_pct=100*(1-tc/tf), response_gain_pct=100*cg,
                quality_ok=keep_quality, total_time_ok=save_cost,
                response_added_value=added_value, dominating_controls=dominated,
                scope="one-development-panel, two seeds, fixed budget; not inferential equivalence")
