"""Explicit window coordinates, full-origin epochs and restartable optimization."""
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def verify_contract(expected,actual):
    for key in ("source_sha","config_sha","data_sha","model_sha","scaler_sha"):
        if expected[key]!=actual[key]:raise ValueError("Resume/cache contract changed: "+key)


class PermutationStream:
    def __init__(self,origins,seed):
        self.origins=np.asarray(origins,np.int64)
        if not len(self.origins):raise ValueError("No legal training origins")
        self.rng=np.random.default_rng(seed)
        self.order=self.rng.permutation(len(self.origins));self.position=0
        self.exposures=0;self.epochs_completed=0

    def draw(self,count):
        pieces=[];remaining=int(count)
        while remaining:
            take=min(remaining,len(self.order)-self.position)
            pieces.append(self.origins[self.order[self.position:self.position+take]])
            self.position+=take;self.exposures+=take;remaining-=take
            if self.position==len(self.order):
                self.epochs_completed+=1;self.order=self.rng.permutation(len(self.origins));self.position=0
        return np.concatenate(pieces)

    def state_dict(self):
        return dict(order=self.order.tolist(),position=self.position,exposures=self.exposures,
            epochs_completed=self.epochs_completed,rng=copy.deepcopy(self.rng.bit_generator.state),
            origins_sha=hashlib.sha256(self.origins.tobytes()).hexdigest())

    def load_state_dict(self,state):
        if state['origins_sha']!=hashlib.sha256(self.origins.tobytes()).hexdigest():
            raise ValueError("Training origins changed")
        self.order=np.asarray(state['order'],np.int64);self.position=state['position']
        self.exposures=state['exposures'];self.epochs_completed=state['epochs_completed']
        self.rng.bit_generator.state=copy.deepcopy(state['rng'])


class FastWindows:
    def __init__(self,task,mean,std,device='cuda'):
        self.L,self.H=task['L'],task['H'];self.channels=task['physical'].shape[1]
        observed=np.asarray(task['observed'],bool)
        normalized=np.where(observed,(task['physical']-mean)/std,0).astype(np.float32)
        previous=np.maximum.accumulate(np.where(observed,np.arange(len(observed))[:,None],-1),axis=0)
        self.values=torch.as_tensor(normalized,device=device)
        self.observed=torch.as_tensor(observed,device=device)
        self.previous=torch.as_tensor(previous,device=device)
        self.device=device
        self.channel_index=torch.arange(self.channels,device=device)

    def get(self,origins):
        o=torch.as_tensor(origins,device=self.device,dtype=torch.long)
        rows=o[:,None]+torch.arange(-self.L,0,device=self.device)[None]
        prev=self.previous[rows]
        x=self.values[prev.clamp(min=0),self.channel_index]
        x=torch.where(prev>=(o-self.L)[:,None,None],x,0)
        future=o[:,None]+torch.arange(self.H,device=self.device)[None]
        return x,self.values[future],self.observed[future]

    def target_counts(self,origins):
        o=torch.as_tensor(origins,device=self.device,dtype=torch.long)
        cumulative=torch.cat((torch.zeros(1,self.channels,device=self.device,dtype=torch.long),self.observed.cumsum(0)),0)
        return (cumulative[o+self.H]-cumulative[o]).sum(0)


def macro_loss(prediction,target,observed,channel_counts,n_origins,batch_origins):
    counts=channel_counts.to(prediction.device)
    active=counts>0
    if not active.any():raise ValueError("No observed training target")
    sse=torch.where(observed,(prediction-target).square(),0).sum((0,1))
    return ((sse[active]/counts[active])*float(n_origins)/batch_origins).mean()


def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def checkpoint_bundle(model,optimizer,sampler,contract,updates,extra=None):
    return dict(model=cpu_state(model),optimizer=optimizer.state_dict(),sampler=sampler.state_dict(),
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if any(p.is_cuda for p in model.parameters()) else [],
        contract=contract,updates=updates,extra=extra or {})


def restore_bundle(bundle,model,optimizer,sampler,contract):
    verify_contract(contract,bundle['contract'])
    model.load_state_dict(bundle['model']);optimizer.load_state_dict(bundle['optimizer']);sampler.load_state_dict(bundle['sampler'])
    torch.set_rng_state(bundle['torch_rng'])
    if bundle['cuda_rng']:torch.cuda.set_rng_state_all(bundle['cuda_rng'])
    return bundle['updates']


def train_segment(model,windows,train_origins,sampler,optimizer,end_exposures,predict,
                  *,effective_batch=16,microbatch=4,starting_updates=0,progress_path=None,
                  budget_deadline=None,k_grid=None,starting_k_exposures=None):
    model.train();counts=windows.target_counts(train_origins);updates=starting_updates
    started=time.perf_counter();clipped=0;grad_norms=[];losses=[]
    k_exposures=dict(starting_k_exposures or {})
    while sampler.exposures<end_exposures:
        if budget_deadline is not None and time.time()>=budget_deadline:
            return dict(status='BUDGET_LIMIT',updates=updates,exposures=sampler.exposures,
                        segment_seconds=time.perf_counter()-started,k_exposures=k_exposures)
        batch=sampler.draw(min(effective_batch,end_exposures-sampler.exposures))
        optimizer.zero_grad(set_to_none=True);value=0.
        k=None if not k_grid else int(k_grid[updates%len(k_grid)])
        for lo in range(0,len(batch),microbatch):
            x,y,m=windows.get(batch[lo:lo+microbatch])
            prediction=predict(x,k)
            if prediction.shape!=y.shape:raise ValueError('Original-channel output shape mismatch')
            loss=macro_loss(prediction,y,m,counts,len(train_origins),len(batch))
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite original-future loss')
            loss.backward();value+=loss.detach().item()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient norm')
        clipped+=int(norm>1);grad_norms.append(float(norm));optimizer.step();updates+=1;losses.append(value)
        if k is not None:k_exposures[str(k)]=k_exposures.get(str(k),0)+len(batch)
        if progress_path and (updates%10==0 or sampler.exposures==end_exposures):
            record=dict(status='TRAINING',updates=updates,exposures=sampler.exposures,
                epoch=sampler.exposures/len(train_origins),target_exposures=end_exposures,
                seconds=time.perf_counter()-started,last_loss=value,k_exposures=k_exposures)
            Path(progress_path).write_text(json.dumps(record),encoding='utf-8')
    if torch.cuda.is_available() and str(windows.device).startswith('cuda'):torch.cuda.synchronize()
    return dict(status='SEGMENT_COMPLETE',updates=updates,exposures=sampler.exposures,
        actual_epochs=sampler.exposures/len(train_origins),segment_seconds=time.perf_counter()-started,
        optimizer_updates_this_segment=len(losses),clipped_steps=clipped,
        gradient_norm_min=min(grad_norms) if grad_norms else None,
        gradient_norm_max=max(grad_norms) if grad_norms else None,
        mean_observed_training_batch_loss=float(np.mean(losses)) if losses else None,k_exposures=k_exposures)
