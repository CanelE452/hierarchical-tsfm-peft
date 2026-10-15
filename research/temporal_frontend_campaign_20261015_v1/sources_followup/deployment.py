"""Rebuild selected routes and time the physical CPU-to-CPU deployment path."""
import json
import time
import numpy as np
import torch
from torch import nn
import runtime as rt
import campaign as cp
from models import TemporalFront
from transforms import r1_predict


class R1(nn.Module):
    def __init__(self, variant, period):
        super().__init__(); self.variant, self.period = variant, period
    def forward(self, x, bolt): return r1_predict(x, bolt, self.variant, period=self.period)


class CropRaw(nn.Module):
    def __init__(self,length):super().__init__();self.length=length
    def forward(self,x,bolt):return bolt(x[:,-self.length:])


class Ridge(nn.Module):
    def __init__(self, weights):
        super().__init__(); self.register_buffer('weights', torch.as_tensor(weights, dtype=torch.float32))
    def forward(self, x, bolt):
        from native_ridge import direct_features
        delta = torch.einsum('bcp,cph->bhc', direct_features(x), self.weights)
        return x[:, -1:]+delta


class Swap(nn.Module):
    def __init__(self, base, weights):
        super().__init__(); self.base = base.requires_grad_(False)
        self.register_buffer('weights', torch.as_tensor(weights, dtype=torch.float32))
    def forward(self, x, bolt):
        from native_ridge import swap_predict
        z = self.base.encode(x)
        return swap_predict(z, self.base.D, self.weights, current=x[:, -1])


class LoRAPair(nn.Module):
    def __init__(self, layer):
        super().__init__(); self.down = nn.Linear(layer.in_features, 8, bias=False)
        self.up = nn.Linear(8, layer.out_features, bias=False)
        nn.init.zeros_(self.up.weight)
    def forward(self, x): return 2.*self.up(self.down(x))


class LoRA(nn.Module):
    """The existing q/v,r8,alpha16,dropout0 configuration as removable hooks."""
    def __init__(self, bolt):
        super().__init__(); self.parts = nn.ModuleList(); self.handles = []
        layers = [m for name, m in bolt.model.named_modules()
                  if name.rsplit('.', 1)[-1] in ('q', 'v') and isinstance(m, nn.Linear)]
        try:
            for layer in layers:
                pair = LoRAPair(layer).cuda(); self.parts.append(pair)
                self.handles.append(layer.register_forward_hook(lambda module, args, output, p=pair: output+p(args[0])))
            assert sum(p.numel() for p in self.parameters()) == 294912
        except Exception:
            self.close(); raise
    def forward(self, x, bolt): return bolt(x)
    def close(self):
        for h in self.handles: h.remove()
        self.handles = []


def build(spec, win, bolt):
    kind = spec['type']
    if kind == 'RAW': model = cp.Raw()
    elif kind == 'CURRENT': model = cp.Current()
    elif kind == 'R1': model = R1(spec['variant'], win.data['period_from_existing_metadata'])
    elif kind == 'ADAPTER':
        model = cp.adapter(spec['kind'], win, spec['r'], spec['seed'], **spec.get('front_kwargs', {}))
    elif kind == 'DIRECT':
        torch.manual_seed(spec['seed']); model = cp.Direct(spec['kind'], win.channels)
    elif kind == 'G':
        base = build(spec['base'], win, bolt)
        torch.manual_seed(spec['seed']); model = cp.WithG(base)
    elif kind in ('RIDGE', 'SWAP'):
        weights = np.load(rt.CACHE/spec['weights'], allow_pickle=False)
        model = Ridge(weights) if kind == 'RIDGE' else Swap(build(spec['base'], win, bolt), weights)
    elif kind == 'LORA':
        torch.manual_seed(spec['seed']); model = LoRA(bolt)
    else: raise ValueError(kind)
    if spec.get('checkpoint'):
        try: model.load_state_dict(torch.load(rt.CACHE/spec['checkpoint'], weights_only=True))
        except Exception:
            for m in model.modules():
                if isinstance(m, LoRA): m.close()
            raise
    return model.cuda().eval()


class Route:
    def __init__(self, spec, win, bolt):
        self.spec, self.win, self.bolt = spec, win, bolt
        self.model = build(spec, win, bolt)
        self.last_stages = {}
        storages = {v.untyped_storage().data_ptr(): v.untyped_storage().nbytes()
                    for v in list(self.model.parameters())+list(self.model.buffers())}
        self.active_parameter_bytes = sum(storages.values())
        self.old_microbatch = bolt.row_microbatch
        bolt.row_microbatch = spec.get('row_microbatch')

    def close(self):
        for m in self.model.modules():
            if isinstance(m, LoRA): m.close()
        self.bolt.row_microbatch = self.old_microbatch
        self.model.cpu()

    @torch.no_grad()
    def predict_physical(self, physical, observed, mean, std, instrument=True):
        stages = {}; start = time.perf_counter()
        required=self.spec.get('length',512)
        if physical.shape[1]<required:raise ValueError('INSUFFICIENT_PHYSICAL_PAST_FOR_ROUTE')
        physical=physical[:,-required:];observed=observed[:,-required:]
        b, length, c = physical.shape
        index = np.maximum.accumulate(np.where(observed, np.arange(length)[None, :, None], -1), axis=1)
        filled = np.take_along_axis(physical, index.clip(0), axis=1)
        filled = np.where(index >= 0, filled, mean)
        x = torch.as_tensor(((filled-mean)/std).astype(np.float32), device='cuda')
        torch.cuda.synchronize(); stages['preprocessing_seconds'] = time.perf_counter()-start
        t = time.perf_counter()
        base = self.model.base if isinstance(self.model, cp.WithG) else self.model
        if isinstance(base, rt.Adapter):
            z = base.encode(x); torch.cuda.synchronize(); stages['PE_seconds'] = time.perf_counter()-t
            t = time.perf_counter(); f = self.bolt(z); torch.cuda.synchronize(); stages['F_seconds'] = time.perf_counter()-t
            t = time.perf_counter(); pred = base.decode(x, z, f)
        elif isinstance(base, (cp.Raw, LoRA)):
            stages['PE_seconds'] = 0.; pred = base(x, self.bolt)
            torch.cuda.synchronize(); stages['F_seconds'] = time.perf_counter()-t; t = time.perf_counter()
        else:
            f_seconds = [0.]
            def timed_bolt(value, horizon=rt.H):
                torch.cuda.synchronize(); start_f = time.perf_counter()
                result = self.bolt(value, horizon=horizon)
                torch.cuda.synchronize(); f_seconds[0] += time.perf_counter()-start_f
                return result
            pred = base(x, timed_bolt if isinstance(base, R1) else self.bolt)
            torch.cuda.synchronize()
            stages['PE_seconds'] = time.perf_counter()-t-f_seconds[0]
            stages['F_seconds'] = f_seconds[0]
            t = time.perf_counter()
        if isinstance(self.model, cp.WithG): pred = pred+self.model.g(x)
        torch.cuda.synchronize(); stages['DG_seconds'] = time.perf_counter()-t
        t = time.perf_counter(); out = pred.cpu().numpy()*std+mean
        torch.cuda.synchronize(); stages['return_seconds'] = time.perf_counter()-t
        self.last_stages = stages
        return out
