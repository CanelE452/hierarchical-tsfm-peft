"""Sequential execution of the approved temporal-front campaign."""
import argparse
import copy
import json
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
import runtime as rt
from models import TemporalFront, DirectMixer, parameter_count


class Windows:
    def __init__(self, data, fold):
        import data as dio
        self.data, self.fold = data, fold
        self.mean, self.std, self.count = dio.fit_statistics(data, fold['bounds']['A'])
        observed = data['observed']
        values = np.where(observed, (data['physical']-self.mean)/self.std, 0).astype(np.float32)
        previous = np.maximum.accumulate(np.where(observed, np.arange(len(values))[:, None], -1), axis=0)
        self.values = torch.as_tensor(values, device='cuda')
        self.mask = torch.as_tensor(observed, device='cuda')
        self.previous = torch.as_tensor(previous, device='cuda')
        self.channels = values.shape[1]
        self.cidx = torch.arange(self.channels, device='cuda')
        self.origins = {s: np.asarray(v, np.int64) for s, v in fold['origins'].items()}
        # Fixed complete-A loss denominators, including repeated target exposures.
        cum = np.vstack((np.zeros((1, self.channels), np.int64), np.cumsum(observed, 0)))
        origins = self.origins['A']
        self.counts = torch.as_tensor((cum[origins+rt.H]-cum[origins]).sum(0), device='cuda')
        self.eval_A = origins[np.unique(np.linspace(0, len(origins)-1, min(32, len(origins))).astype(int))]

    def get(self, origins, length=rt.L):
        o = torch.as_tensor(origins, device='cuda', dtype=torch.long)
        t = o[:, None]+torch.arange(-length, 0, device='cuda')[None]
        prev = self.previous[t]
        x = self.values[prev.clamp_min(0), self.cidx]
        x = torch.where(prev >= (o-length)[:, None, None], x, 0.)
        ty = o[:, None]+torch.arange(rt.H, device='cuda')[None]
        return x, self.values[ty], self.mask[ty]

    def truth(self, origins):
        o = np.asarray(origins)
        rows = o[:, None]+np.arange(rt.H)[None]
        mask = self.data['observed'][rows]
        y = np.where(mask, (self.data['physical'][rows]-self.mean)/self.std, 0).astype(np.float32)
        return y, mask

    def basis(self, r):
        lo, hi = self.fold['bounds']['A']
        # Causal filled A rows only; no B/C information in PCA.
        p = self.previous[lo:hi]
        filled = torch.where(p >= lo, self.values[p.clamp_min(0), self.cidx], 0.)
        return rt.pca(filled.cpu().numpy(), r)


@torch.no_grad()
def infer(model, bolt, win, origins, batch=8, length=512):
    out = []
    for start in range(0, len(origins), batch):
        x, _, _ = win.get(origins[start:start+batch], length)
        out.append(model(x, bolt).cpu().numpy())
    return np.concatenate(out)


def assess(model, bolt, win, split, origins=None):
    origins = win.origins[split] if origins is None else origins
    pred = infer(model, bolt, win, origins)
    y, mask = win.truth(origins)
    return rt.score(pred, y, mask), pred


def still_decreasing(checkpoints):
    values = [r['B']['mse'] for r in checkpoints[-3:]]
    return len(values) == 3 and values[0] > values[1] > values[2]


class Raw(nn.Module):
    def forward(self, x, bolt): return bolt(x)


class Current(nn.Module):
    def forward(self, x, bolt): return x[:, -1:].expand(-1, rt.H, -1)


class Direct(nn.Module):
    def __init__(self, kind, c):
        super().__init__(); self.net = DirectMixer(kind, channels=c)
    def forward(self, x, bolt): return self.net(x)


class WithG(nn.Module):
    def __init__(self, base):
        super().__init__(); self.base = base.requires_grad_(False); self.g = rt.Residual()
    def forward(self, x, bolt):
        with torch.no_grad(): value = self.base(x, bolt)
        return value+self.g(x)


class Native(nn.Module):
    def __init__(self, base):
        super().__init__(); self.base = base.requires_grad_(False)
        self.head = nn.Linear(rt.L, rt.H)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
    def forward(self, x, bolt):
        z = self.base.encode(x)
        f = z[:, -1:] + self.head((z-z[:, -1:]).transpose(1, 2)).transpose(1, 2)
        return self.base.decode(x, z, f)


def train(spec, model, bolt, win, updates=600, carry=None, lr=.001):
    name = spec['id']; model = model.cuda().eval()
    parameters = [p for p in model.parameters() if p.requires_grad]
    assert parameters
    assert all(not p.requires_grad and p.grad is None for p in bolt.model.parameters())
    assert not {id(p) for p in parameters}.intersection(id(p) for p in bolt.model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.)
    begin, curve, checkpoints = 0, [], []
    if carry:
        begin = carry['updates']; optimizer.load_state_dict(carry['optimizer'])
        curve = carry['curve']; checkpoints = carry['checkpoints']
    order = rt.schedule(len(win.origins['A']), 2400, spec['seed'])
    # Identical prefix order across candidates and conditional extensions.
    cp = set([0, 100, 200, 400, 600, 800, 1000, 1200, 1600, 2000, 2400])
    best = min(checkpoints, key=lambda v: (v['B']['mse'], v['update'])) if checkpoints else None
    started = time.perf_counter()
    for step in range(begin, updates+1):
        if step in cp and (not checkpoints or step > checkpoints[-1]['update']):
            a, _ = assess(model, bolt, win, 'A', win.eval_A)
            b, prediction = assess(model, bolt, win, 'B')
            row = dict(update=step, A=a, B=b, state_sha256=rt.state_hash(model))
            checkpoints.append(row)
            rt.save(rt.CACHE/'checkpoints'/name/f'cp{step}.pt', rt.clone(model))
            if best is None or (b['mse'], step) < (best['B']['mse'], best['update']):
                best = row
            rt.event(stage='CHECKPOINT', id=name, update=step, A_mse=a['mse'], B_mse=b['mse'])
        if step == updates: break
        o = win.origins['A'][order[step]]
        x, y, mask = win.get(o)
        optimizer.zero_grad(set_to_none=True)
        t = time.perf_counter()
        pred = model(x, bolt)
        loss = rt.loss(pred, y, mask, win.counts, len(win.origins['A']))
        if not torch.isfinite(loss): raise FloatingPointError('NONFINITE_LOSS')
        loss.backward()
        norms = {n: float(p.grad.norm()) for n, p in model.named_parameters() if p.grad is not None}
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        if not torch.isfinite(norm): raise FloatingPointError('NONFINITE_GRADIENT')
        optimizer.step()
        if not all(bool(torch.isfinite(p).all()) for p in parameters): raise FloatingPointError('NONFINITE_PARAMETER')
        torch.cuda.synchronize()
        curve.append(dict(update=step+1, loss=float(loss.detach()), gradient_norm=float(norm),
            clipped=bool(norm > 1), gradient_by_parameter=norms if step in (0, 1, 19, 99, 599) else {},
            seconds=time.perf_counter()-t))
        if (step+1) % 100 == 0: rt.event(stage='UPDATE', id=name, update=step+1, loss=curve[-1]['loss'])
    elapsed = time.perf_counter()-started
    last_state = rt.clone(model)
    carry = dict(updates=updates, optimizer=optimizer.state_dict(), curve=curve, checkpoints=checkpoints)
    rt.save(rt.CACHE/'training'/name/f'last_{updates}.pt', dict(state=last_state, carry=carry))
    selected_state = torch.load(rt.CACHE/'checkpoints'/name/f"cp{best['update']}.pt", weights_only=True)
    model.load_state_dict(selected_state)
    touched = np.unique(order[:updates].reshape(-1))
    used_origins = win.origins['A'][touched]
    input_times = np.unique((used_origins[:, None]+np.arange(-rt.L, 0)[None]).reshape(-1))
    target_times = np.unique((used_origins[:, None]+np.arange(rt.H)[None]).reshape(-1))
    result = dict(**spec, updates=updates, selected_update=best['update'], selected_B=best['B'],
        checkpoints=checkpoints, learning_curve=curve, parameter_count=sum(p.numel() for p in parameters),
        optimizer='AdamW', learning_rate=lr, batch=4, clipping_fraction=float(np.mean([r['clipped'] for r in curve])),
        update_seconds_median=float(np.median([r['seconds'] for r in curve])), segment_wall_seconds=elapsed,
        origin_schedule_sha256=rt.ahash(order[:updates]), legal_A_origins=len(win.origins['A']),
        distinct_A_origins=len(touched), unique_input_times=len(input_times), unique_target_times=len(target_times),
        unique_true_observed_target_entries=int(win.data['observed'][target_times].sum()),
        repeated_origin_exposures=int(updates*4-len(touched)), selected_state_sha256=rt.state_hash(model),
        status='UNDERTRAINED_OR_UNRESOLVED' if updates == 2400 and
            (best['update'] == 2400 or still_decreasing(checkpoints)) else 'COMPLETE_VALID',
        reused=False, protected_TEST_accessed=False, C_scored=False)
    rt.write(rt.HERE/'fits'/f'{name}_{updates}.json', result)
    return result, last_state, carry


def adapter(kind, win, r, seed, **kwargs):
    torch.manual_seed(seed)
    center, basis = win.basis(r)
    return rt.Adapter(TemporalFront(kind, channels=win.channels, **kwargs), center, basis).cuda()


def dataset_windows(name, fold='primary'):
    import data as dio
    d = dio.load_dataset(name)
    f = dio.make_fold(d, fold=fold)
    return Windows(d, f)


def probe():
    bolt = rt.Bolt(); win = dataset_windows('beijing')
    rows = []
    for r in (2, win.channels):
        for kind in ('B0', 'B1', 'B2', 'B3', 'B4'):
            model = adapter(kind, win, r, rt.SEEDS[0])
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0.)
            seconds, diagnostics = [], []
            for step, ids in enumerate(rt.schedule(len(win.origins['A']), 20, rt.SEEDS[0])):
                x, y, mask = win.get(win.origins['A'][ids]); optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize(); t = time.perf_counter()
                pred = model(x, bolt); value = rt.loss(pred, y, mask, win.counts, len(win.origins['A']))
                value.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                assert torch.isfinite(norm)
                diagnostics.append({n: float(p.grad.norm()) for n, p in model.named_parameters() if p.grad is not None})
                optimizer.step(); torch.cuda.synchronize(); seconds.append(time.perf_counter()-t)
            rows.append(dict(kind=kind, r=r, updates=20, median_seconds=float(np.median(seconds[2:])),
                parameters=parameter_count(model), first_two_gradients=diagnostics[:2],
                peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved()))
            rt.event(stage='PROBE', **{k:v for k,v in rows[-1].items() if k != 'first_two_gradients'})
            del model, optimizer; torch.cuda.empty_cache()
    assert rt.state_hash(bolt.model) == bolt.initial_hash
    rt.write(rt.HERE/'PROBE.json', dict(rows=rows, F_unchanged=True, C_scored=False))


def core():
    bolt = rt.Bolt(); results = []
    for dataset, k in (('beijing', 2), ('jena', 5)):
        win = dataset_windows(dataset)
        for r in (win.channels, k):
            for seed in rt.SEEDS:
                group = []
                for kind in ('B0', 'B1', 'B2', 'B3', 'B4'):
                    name = f'primary_{dataset}_{kind}_r{r}_s{seed}'
                    spec = dict(id=name, block='R2', dataset=dataset, fold='primary', kind=kind, r=r, seed=seed)
                    model = adapter(kind, win, r, seed)
                    try:
                        result, last, carry = train(spec, model, bolt, win)
                        group.append([spec, model, result, last, carry])
                    except Exception as e:
                        rt.write(rt.HERE/'failures'/f'{name}.json', dict(**spec, status='INVALID', error=str(e), traceback=traceback.format_exc()))
                        rt.event(stage='PATH_FAILURE', id=name, error=str(e))
                # A final-best candidate triggers an equal extension for this whole matched group.
                for endpoint in (1200, 2400):
                    prior = endpoint//2
                    if not any(v[2]['selected_update'] == prior or still_decreasing(v[4]['checkpoints']) for v in group): break
                    rt.event(stage='MATCHED_EXTENSION', dataset=dataset, r=r, seed=seed, endpoint=endpoint)
                    for item in group:
                        spec, model, old, last, carry = item
                        model.load_state_dict(last)
                        result, last, carry = train(spec, model, bolt, win, endpoint, carry)
                        item[2:] = [result, last, carry]
                for spec, model, result, _, _ in group:
                    rt.save(rt.CACHE/'selected'/f"{spec['id']}.pt", rt.clone(model))
                    results.append({k: v for k, v in result.items() if k not in ('learning_curve', 'checkpoints')})
                del group; torch.cuda.empty_cache()
        del win; torch.cuda.empty_cache()
    assert rt.state_hash(bolt.model) == bolt.initial_hash
    rt.write(rt.HERE/'R2_SELECTIONS.json', dict(results=results, F_unchanged=True, C_scored=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['probe', 'core'])
    args = parser.parse_args(); torch.set_num_threads(2); torch.set_num_interop_threads(1)
    rt.CACHE.mkdir(parents=True, exist_ok=True)
    rt.event(stage='PHASE_START', phase=args.phase)
    try:
        globals()[args.phase]()
        rt.event(stage='PHASE_COMPLETE', phase=args.phase)
    except Exception as error:
        rt.event(stage='PHASE_FAILURE', phase=args.phase, error=str(error), traceback=traceback.format_exc())
        raise
