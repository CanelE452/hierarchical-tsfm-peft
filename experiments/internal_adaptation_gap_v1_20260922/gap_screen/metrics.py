from __future__ import annotations
import numpy as np

def score(q: np.ndarray, y: np.ndarray, sigma: np.ndarray, levels: np.ndarray) -> tuple[dict, np.ndarray]:
    """q=N,C,Q,H, y=N,C,H. Sort only for evaluation; also report raw crossing."""
    q = np.asarray(q, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64); levels = np.asarray(levels, dtype=np.float64)
    if q.ndim != 4 or y.shape != (q.shape[0], q.shape[1], q.shape[3]): raise ValueError("bad forecast dimensions")
    if sigma.shape != (q.shape[1],) or np.any(sigma <= 0): raise ValueError("bad TRAIN scales")
    if not np.isfinite(q).all() or not np.isfinite(y).all(): raise ValueError("nonfinite score inputs")
    crossing = float((np.diff(q, axis=2) < 0).mean())
    ordered = np.sort(q, axis=2)
    e = y[:, :, None] - ordered
    rho = 2 * np.maximum(e * levels[None,None,:,None], e * (levels[None,None,:,None]-1))
    per = (rho / sigma[None,:,None,None]).mean(axis=(2,3))
    med = ordered[:,:,int(np.argmin(abs(levels-.5)))]
    lo = ordered[:,:,int(np.argmin(abs(levels-.1)))]; hi = ordered[:,:,int(np.argmin(abs(levels-.9)))]
    return dict(primary=float(per.mean()), nmae=float((abs(med-y)/sigma[None,:,None]).mean()),
                raw_mae=float(abs(med-y).mean()), nrmse=float(np.sqrt((((med-y)/sigma[None,:,None])**2).mean(axis=(0,2))).mean()),
                coverage80=float(((y>=lo)&(y<=hi)).mean()), width80=float(((hi-lo)/sigma[None,:,None]).mean()),
                crossing=crossing, per_channel=per.mean(0).tolist()), per

def scalar_primary(q, y, sigma, levels) -> float:
    """Deliberately separate loop implementation for small independent replays."""
    total = []
    for c in range(y.shape[1]):
        losses = []
        for n in range(y.shape[0]):
            for h in range(y.shape[2]):
                vals = sorted(float(v) for v in q[n,c,:,h])
                for z, tau in zip(vals, levels):
                    e = float(y[n,c,h]) - z
                    losses.append(2*(float(tau)*e if e>=0 else (float(tau)-1)*e)/float(sigma[c]))
        total.append(sum(losses)/len(losses))
    return sum(total)/len(total)

def gain(candidate: float, baseline: float) -> float:
    if baseline <= 0 or not np.isfinite(baseline): raise ValueError("invalid baseline")
    return 100*(baseline-candidate)/baseline

def paired_bootstrap(c: np.ndarray, b: np.ndarray, repeats=2000, block=7, seed=92603, labels=None) -> dict:
    """Inputs seed,origin,channel. Blocks are aligned time blocks, not seed resampling."""
    c = np.asarray(c); b = np.asarray(b)
    if c.shape != b.shape or c.ndim != 3: raise ValueError("unpaired scores")
    n = c.shape[1]; rng=np.random.default_rng(seed); stats=[]
    blocks=[np.flatnonzero(np.asarray(labels)==v) for v in np.unique(labels)] if labels is not None else [np.arange(i,min(i+block,n)) for i in range(0,n,block)]
    for _ in range(repeats):
        ix=np.concatenate([blocks[j] for j in rng.integers(len(blocks),size=len(blocks))])
        stats.append(gain(float(c[:,ix].mean()),float(b[:,ix].mean())))
    return dict(gain_pct=gain(float(c.mean()),float(b.mean())),
                ci95=np.quantile(stats,[.025,.975]).tolist(), seed_gains=[gain(x.mean(),z.mean()) for x,z in zip(c,b)],
                blocks=len(blocks), block_origins=block, independent_seed_population_claim=False)

def choose_checkpoint(curves: list[dict]) -> dict:
    return min(curves, key=lambda r: (r['val_primary'],r['step']))

def best_under_budget(curves: list[dict], budget: float) -> dict | None:
    eligible=[r for r in curves if r['cold_ready_s']<=budget]
    return choose_checkpoint(eligible) if eligible else None

def first_target(curves: list[dict], threshold: float) -> dict:
    by_step=sorted(curves,key=lambda r:r['step']); prev=None
    for r in by_step:
        if r['val_primary'] <= threshold:
            return dict(status='NO_ADAPTATION_HEADROOM' if r['step']==0 else 'REACHED_AT_RECORDED_CHECKPOINT',
                        step=r['step'], time_s=r['cold_ready_s'], lower_time_s=prev['cold_ready_s'] if prev else 0,
                        lower_step=prev['step'] if prev else 0, interpolated=False)
        prev=r
    return dict(status='CENSORED',step=None,time_s=None,lower_time_s=by_step[-1]['cold_ready_s'],interpolated=False)

def stable_seed_hash(text: str) -> int:
    import hashlib
    return int(hashlib.sha256(text.encode()).hexdigest()[:8],16)
