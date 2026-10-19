"""Actual physical-input deployment paths without cached forecasts/features."""
import hashlib
import math
from pathlib import Path
import time

import numpy as np
import torch

from research.raw_context_multiscale_20261018_v1 import common as inherited
from . import blocks, common, models, probes


METHODS = ('G','S','RAW','ALPHA_fitted',*models.METHODS,*blocks.METHODS,'ZERO','MEAN','OWN','ALL')
F_METHODS = ('S','RAW','ALPHA_fitted','RESIDUAL_F','READOUT_ONLY',*blocks.METHODS)


def c_prefix_metadata(spec,task):
    """Resolve a C prefix from sealed light metadata without loading weights."""
    allowed=(0,task['K']//2,task['K'])
    selected=spec.get('artifact_selected_prefix')
    source='request' if selected is not None else None
    if spec.get('state_path'):
        seal=Path(spec['state_path']).parent/'SELECTION_SEAL.json'
        if seal.is_file():
            if spec.get('selection_seal_sha256') and common.sha(seal)!=spec['selection_seal_sha256']:
                raise ValueError('C selection seal identity changed')
            actual=common.read(seal)['selected_prefix']
            if selected is not None and selected!=actual:
                raise ValueError('C selection metadata disagrees with original artifact selection')
            selected,source=actual,'SELECTION_SEAL.json'
    executed=spec.get('prefix',selected)
    if type(executed) is not int or executed not in allowed:
        raise ValueError('C prefix must be registered zero, half, or full K; default requires sealed selection metadata')
    if selected is not None and (type(selected) is not int or selected not in allowed):
        raise ValueError('Original C artifact selection prefix is invalid')
    return dict(executed_prefix=executed,artifact_selected_prefix=selected,
                artifact_selected_prefix_source=source,
                prefix_override_requested='prefix' in spec,
                prefix_is_selected=None if selected is None else executed==selected)


def assert_v_sealed(report=None):
    path = Path(common.REPORT if report is None else report)/'C_DECISIONS.json'
    if not path.is_file() or common.read(path).get('status') != 'SEALED_BEFORE_V':
        raise ValueError('All B/C decisions must be sealed before V numeric access')
    return dict(reference=str(path),sha256=common.sha(path),status='SEALED_BEFORE_V')


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _artifact(spec):
    path = Path(spec['state_path']).resolve()
    if not path.is_file() or not spec.get('state_sha256') or common.sha(path) != spec['state_sha256']:
        raise ValueError('Exact final artifact file SHA256 required')
    return path


class Route:
    """A single loaded deployment method with one lazy shared foundation."""
    def __init__(self, spec, parents, *, ctx=None, task=None, device='cuda', foundation_pool=None):
        started = time.perf_counter()
        self.spec, self.method, self.device = dict(spec),spec['method'],str(device)
        if self.method not in METHODS:
            raise ValueError('Unregistered deployment method: '+self.method)
        self.task = task if task is not None else parents['task']
        self.mean,self.std = np.asarray(parents['mean'],np.float32),np.asarray(parents['std'],np.float32)
        self.ctx,self.pool = ctx,foundation_pool
        self.model = self.g_model = self.s_model = None
        self.ridge = None
        self._closed = False
        self.last_receipt = {}
        self.c_prefix_info=c_prefix_metadata(spec,self.task) if self.method in blocks.METHODS else {}
        self.artifact_selected_prefix=self.c_prefix_info.get('artifact_selected_prefix')
        self.k = (self.c_prefix_info['executed_prefix'] if self.c_prefix_info else
                  int(self.spec.get('prefix',len(self.task['columns']) if self.method=='RAW' else self.task['K'])))
        if not 0 <= self.k <= (len(self.task['columns']) if self.method=='RAW' else self.task['K']):
            raise ValueError('Prefix outside registered K')
        self.alpha = float(spec.get('alpha',.5))
        if self.method == 'ALPHA_fitted' and (not math.isfinite(self.alpha) or not 0 <= self.alpha <= 1):
            raise ValueError('Sealed global alpha must be in [0,1]')
        use_g = self.method not in ('S','RAW') and not (self.method=='ALPHA_fitted' and self.alpha==1)
        use_s = self.method == 'S' or (self.method=='ALPHA_fitted' and self.alpha!=0)
        if use_g:
            self.g_model = parents['g_model'].to(self.device).eval().requires_grad_(False)
        if use_s:
            self.s_model = parents['s_model'].to(self.device).eval().requires_grad_(False)
        if self.method in models.METHODS and self.k:
            if self.k != self.task['K']:
                raise ValueError('Phase B residual paths require their full registered K')
            self.model = models.make(parents['s_model'],self.method,int(spec['seed']))
            saved = torch.load(_artifact(spec),map_location='cpu',weights_only=False)
            state = saved['model'] if isinstance(saved,dict) and 'model' in saved else saved
            self.model.load_state_dict(state,strict=True)
            if spec.get('model_state_sha256') and state_hash(self.model) != spec['model_state_sha256']:
                raise ValueError('Final correction model state identity changed')
            self.model.to(self.device).eval().requires_grad_(False)
        if self.method in blocks.METHODS and self.k:
            self.model=blocks.restore(parents['s_model'],_artifact(spec),device=self.device)
            if self.model.method!=self.method:
                raise ValueError('C artifact method differs from requested method')
            selected=int(self.model.prefix_k)
            if self.artifact_selected_prefix is not None and selected!=self.artifact_selected_prefix:
                raise ValueError('C artifact differs from sealed selection metadata')
            self.artifact_selected_prefix=selected
            self.c_prefix_info.update(artifact_selected_prefix=selected,
                prefix_is_selected=self.k==selected)
            if spec.get('model_state_sha256') and state_hash(self.model)!=spec['model_state_sha256']:
                raise ValueError('Final C model state identity changed')
            self.model.prefix_k=self.k
        if self.method in ('MEAN','OWN','ALL'):
            if self.device != 'cpu':
                raise ValueError('NumPy residual probes have a separate CPU cost scope')
            with np.load(_artifact(spec),allow_pickle=False) as archive:
                self.ridge = {key:archive[key].copy() for key in archive.files}
            if self.method != 'MEAN':
                self.ridge['kind'] = str(self.ridge['kind'].item())
                if self.ridge['kind'] != self.method:
                    raise ValueError('Probe artifact kind differs')
        self.mean_t = torch.as_tensor(self.mean,device=self.device)
        self.std_t = torch.as_tensor(self.std,device=self.device)
        self.model_load_seconds = time.perf_counter()-started

    @property
    def needs_foundation(self):
        return self.method in F_METHODS and not (self.method=='ALPHA_fitted' and self.alpha==0) and self.k>0

    def warm_load(self):
        if not self.needs_foundation:
            return
        if self.pool is not None and self.pool.foundation is not None:
            return
        started = time.perf_counter()
        if self.pool is None:
            if self.ctx is None:
                raise ValueError('Exact foundation context required')
            legacy = self.ctx['modules']['deployment']
            self.pool = legacy.FoundationPool(self.ctx['asset']/self.ctx['modules']['legacy'].MODEL_REL,self.device)
        foundation = self.pool.get(self.task['H'])
        if self.ctx is not None and foundation.initial_hash != self.ctx['modules']['legacy'].F_SHA:
            raise ValueError('Foundation tensor identity changed')
        foundation.row_microbatch,foundation.checkpoint_inputs = 128,False
        self.model_load_seconds += time.perf_counter()-started

    def _counters(self):
        if self.pool is None:
            return dict(streams=0,physical_streams=0,physical_calls=0)
        return self.pool.counters()

    def _foundation(self,z):
        self.warm_load()
        foundation = self.pool.get(self.task['H'])
        before = self._counters()
        output = foundation(z)
        after = self._counters()
        config = getattr(getattr(foundation,'model',None),'chronos_config',None)
        context_limit = getattr(config,'context_length',None)
        self._trace.append(dict(input_shape=list(z.shape),input_length=int(z.shape[1]),
            actual_input_length=min(int(z.shape[1]),int(context_limit)) if context_limit else int(z.shape[1]),
            requested_horizon=self.task['H'],actual_head_horizon=getattr(config,'prediction_length',None),
            returned_horizon=int(output.shape[1]),row_microbatch=128,
            logical_rows=after['streams']-before['streams'],
            physical_rows=after['physical_streams']-before['physical_streams'],
            physical_calls=after['physical_calls']-before['physical_calls']))
        return output

    def _preprocess(self,physical,observed):
        if observed.all():
            filled = physical.copy()
        else:
            from research.informs_broad_validation_20261015_v1.data_contract import causal_forward_fill
            filled = np.stack([causal_forward_fill(x,m,self.mean) for x,m in zip(physical,observed)])
        normalized = ((filled-self.mean)/self.std).astype(np.float32)
        return torch.as_tensor(normalized,device=self.device)

    @torch.no_grad()
    def predict_physical(self,physical,observed):
        if self._closed:
            raise RuntimeError('Deployment route is closed')
        if self.device.startswith('cuda'):
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        physical,observed = np.asarray(physical,np.float32),np.asarray(observed)
        axes = (self.task['L'],len(self.task['columns']))
        if physical.ndim!=3 or physical.shape[1:]!=axes or observed.shape!=physical.shape or observed.dtype!=np.bool_:
            raise ValueError('Physical history and original bool past mask must be [B,L,C]')
        if not np.isfinite(physical[observed]).all():
            raise ValueError('Observed physical history must be finite')
        self._trace = []
        before = self._counters()
        x = self._preprocess(physical,observed)
        if self.method == 'RAW':
            prediction = self._foundation(x)
        elif self.method == 'S':
            prediction = self.s_model(x,self._foundation,self.k)
        elif self.method == 'ALPHA_fitted':
            g = self.g_model(x) if self.g_model is not None else None
            s = self.s_model(x,self._foundation,self.k) if self.s_model is not None else None
            prediction = s if g is None else g if s is None else g+self.alpha*(s-g)
        else:
            prediction = self.g_model(x)
            if self.method in (*models.METHODS,*blocks.METHODS) and self.k:
                prediction = prediction+self.model(x,self._foundation if self.needs_foundation else None)
            elif self.method == 'MEAN':
                prediction = prediction.double()+torch.as_tensor(self.ridge['mean'],dtype=torch.float64)
            elif self.method in ('OWN','ALL'):
                correction = probes.ridge_predict(self.ridge,probes.summary(x.cpu().numpy()))
                prediction = prediction.double()+torch.from_numpy(correction)
        result = (prediction.double()*self.std_t.double()+self.mean_t.double()).cpu().numpy()
        if self.device.startswith('cuda'):
            torch.cuda.synchronize(self.device)
        after = self._counters()
        if result.shape != (len(x),self.task['H'],len(self.task['columns'])) or not np.isfinite(result).all():
            raise ValueError('Physical forecast axes/values invalid')
        self.last_receipt = dict(whole_seconds=time.perf_counter()-started,
            F_logical_rows=after['streams']-before['streams'],
            F_physical_rows=after['physical_streams']-before['physical_streams'],
            F_calls=after['physical_calls']-before['physical_calls'],F_invocations=self._trace,
            F_instances_created=0 if self.pool is None else self.pool.instances_created,
            cached_forecast_or_features_reads=0,
            selected_prefix=self.k if self.method in (*F_METHODS,'RESIDUAL_Q') else 0,
            **self.c_prefix_info)
        if self.method in blocks.METHODS:
            self.last_receipt.update(
                G_only_without_F_resident=self.k==0 and (self.pool is None or self.pool.foundation is None),
                F_weights_fixed_across_nonzero_prefixes=True,
                unselected_E_D_weights_may_remain_resident=True)
        return result

    def resident_state(self):
        modules = [m for m in (self.g_model,self.s_model,self.model) if m is not None]
        if self.pool is not None and self.pool.foundation is not None:
            modules.append(getattr(self.pool.foundation,'model',self.pool.foundation))
        seen,parameters,buffers,byte_counts = set(),0,0,{'cpu':0,'cuda':0}
        for module in modules:
            for parameter,values in ((True,module.parameters()),(False,module.buffers())):
                for value in values:
                    identity = (str(value.device),value.data_ptr(),value.numel())
                    if identity in seen:
                        continue
                    seen.add(identity)
                    parameters += value.numel() if parameter else 0
                    buffers += value.numel() if not parameter else 0
                    byte_counts[value.device.type] += value.numel()*value.element_size()
        if self.ridge:
            byte_counts['cpu'] += sum(v.nbytes for v in self.ridge.values() if isinstance(v,np.ndarray))
        return dict(resident_parameters=parameters,resident_buffers=buffers,
                    resident_weights_CPU_MiB=byte_counts['cpu']/1024**2,
                    resident_weights_GPU_MiB=byte_counts['cuda']/1024**2,
                    F_instances_created=0 if self.pool is None else self.pool.instances_created,
                    F_resident=self.pool is not None and self.pool.foundation is not None,
                    F_weight_memory_reduced_by_nonzero_prefix=False if self.method in blocks.METHODS and self.k>0 else None,
                    unselected_E_D_weights_may_remain_resident=self.method in blocks.METHODS and self.k>0)

    def close(self):
        if self.pool is not None:
            self.pool.close()
        self.model = self.g_model = self.s_model = None
        self.ridge = None
        self._closed = True
