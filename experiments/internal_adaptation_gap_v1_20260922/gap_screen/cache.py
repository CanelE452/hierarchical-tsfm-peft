from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import torch
from .io import Blocked, read_json, write_json, digest_object, sha
from .model import Predictor, autocast, sync

class Features:
    def __init__(self,path: Path):
        self.path=path; self.receipt=read_json(path/'receipt.json')
        if not self.receipt.get('complete'): raise Blocked('PARTIAL_FEATURE_CACHE')
        self.origins=np.load(path/'origins.npy'); self.lookup={int(o):i for i,o in enumerate(self.origins)}
        self.arrays={k:np.load(path/(k+'.npy'),mmap_mode='r') for k in ['hidden','base_norm','loc','scale']}
    def batch(self,origins,device):
        ii=[self.lookup[int(o)] for o in origins]; out=[]
        for k in ['hidden','base_norm','loc','scale']:
            a=self.arrays[k][ii]; a=a.reshape(-1,*a.shape[2:])
            out.append(torch.as_tensor(np.array(a,copy=True),device=device))
        return out


def build_cache(base,panel,config,path,roles,device,model_fingerprint):
    oo=np.unique(np.concatenate([panel.origins[r] for r in roles]))
    contract=dict(model_fingerprint=model_fingerprint,data_sha=panel.metadata['values_sha256'],
                  context=panel.context,horizon=panel.horizon,columns=panel.columns,
                  origins_sha=digest_object(oo.tolist()),dtype='float32 storage; common autocast on forward',
                  precision=config['precision'],groups='one complete multivariate origin per forward')
    receipt_path=path/'receipt.json'
    if receipt_path.exists():
        old=read_json(receipt_path)
        if old['contract']!=contract or not old['complete']: raise Blocked('CACHE_CONTRACT_CHANGED')
        for n,h in old['sha256'].items():
            if sha(path/n)!=h: raise Blocked('CACHE_HASH_MISMATCH')
        return Features(path)
    if path.exists() and any(path.iterdir()): raise Blocked('PARTIAL_CACHE: automatic overwrite prohibited')
    path.mkdir(parents=True,exist_ok=True); start=time.perf_counter(); c=len(panel.columns)
    k=panel.horizon//base.chronos_config.output_patch_size
    shapes={'hidden':(len(oo),c,k,base.model_dim),'base_norm':(len(oo),c,base.num_quantiles,panel.horizon),
            'loc':(len(oo),c,1),'scale':(len(oo),c,1)}
    arrays={name:np.lib.format.open_memmap(path/(name+'.npy'),mode='w+',dtype='float32',shape=shape) for name,shape in shapes.items()}
    np.save(path/'origins.npy',oo); wrapper=Predictor(base,'F0',config,config['seeds'][0]); compute=0.
    if device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
    for i,o in enumerate(oo):
        x,_,ids=panel.batch([o]); sync(device); t=time.perf_counter()
        with torch.no_grad(),autocast(device,config['precision']):
            features=wrapper.encode(torch.as_tensor(x,device=device),torch.as_tensor(ids,device=device),panel.horizon)
        sync(device); compute+=time.perf_counter()-t
        for name,z in zip(['hidden','base_norm','loc','scale'],features): arrays[name][i]=z.float().cpu().numpy()
        if i%100==0: print(f'CACHE {panel.name} {i+1}/{len(oo)}',flush=True)
    for a in arrays.values(): a.flush()
    elapsed=time.perf_counter()-start
    files={p.name:sha(p) for p in path.glob('*.npy')}
    receipt=dict(complete=True,contract=contract,cache_build_s=elapsed,forward_compute_s=compute,
                 peak_allocated_bytes=int(torch.cuda.max_memory_allocated()) if device.startswith('cuda') else None,
                 bytes=sum(p.stat().st_size for p in path.glob('*.npy')),sha256=files,
                 target_values_used_for_features=False,roles=roles)
    write_json(receipt_path,receipt); return Features(path)
