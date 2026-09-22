from __future__ import annotations
import copy, hashlib, math, time
from contextlib import nullcontext
from pathlib import Path
import torch
from torch import nn
from .io import Blocked


def seed_all(seed):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def configure_runtime(device: str, precision: str):
    torch.set_num_threads(4)
    if device.startswith('cuda'):
        if not torch.cuda.is_available(): raise Blocked('NO_CUDA: main real-model training is prohibited on CPU')
        if precision=='bf16' and not torch.cuda.is_bf16_supported(): raise Blocked('BF16_NOT_SUPPORTED: choose a common dtype before sealing')
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)


def autocast(device: str, precision: str):
    return torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False) if device.startswith('cuda') and precision=='bf16' else nullcontext()


def sync(device):
    if str(device).startswith('cuda'): torch.cuda.synchronize()


def raw_inverse(norm,loc,scale,use_arcsinh):
    n=norm.float(); n=n.sinh() if use_arcsinh else n
    return n*scale[:,None,:]+loc[:,None,:]


def normalized_loss(norm,y,loc,scale,levels,use_arcsinh):
    """Chronos-2 native loss: mean horizon, sum quantiles, mean series. No future covariates."""
    target=(y.float()-loc)/scale
    if use_arcsinh: target=target.asinh()
    mask=torch.isfinite(target); target=torch.where(mask,target,0.)
    pad=norm.shape[-1]-target.shape[-1]
    if pad<0: raise ValueError('horizon exceeds output')
    if pad:
        target=torch.nn.functional.pad(target,(0,pad)); mask=torch.nn.functional.pad(mask,(0,pad),value=False)
    e=target[:,None]-norm.float()
    loss=2*torch.maximum(levels[None,:,None]*e,(levels[None,:,None]-1)*e)
    return (loss*mask[:,None]).mean(-1).sum(-1).mean()


def to_quantiles(z,nq,patch):
    return z.reshape(z.shape[0],z.shape[1],nq,patch).permute(0,2,1,3).flatten(2).float()


def freeze_stochastic(base):
    base.eval(); base.config.dropout_rate=0.
    for m in base.modules():
        if isinstance(m,nn.Dropout): m.p=0.
        if hasattr(m,'dropout') and isinstance(m.dropout,(float,int)): m.dropout=0.
        if hasattr(m,'config') and hasattr(m.config,'dropout_rate'): m.config.dropout_rate=0.


def fingerprint(model, frozen_only=False):
    h=hashlib.sha256()
    items=[(n,p) for n,p in model.named_parameters() if not frozen_only or not p.requires_grad]
    items+=list(model.named_buffers())
    for n,p in sorted(items):
        h.update(n.encode()); h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def trainable_state(model): return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}


def restore(model,state):
    ps={n:p for n,p in model.named_parameters() if p.requires_grad}
    if ps.keys()!=state.keys(): raise Blocked('CHECKPOINT_PARAMETER_CONTRACT_CHANGED')
    with torch.no_grad():
        for n,p in ps.items(): p.copy_(state[n].to(p.device))


def lora_targets(base):
    return [n for n,m in base.named_modules() if isinstance(m,nn.Linear) and
            (any(n.endswith('self_attention.'+x) for x in ('q','k','v','o')) or n=='output_patch_embedding.output_layer')]


def enable_checkpointing(base):
    """Non-reentrant checkpoints support trainable LoRA even when first inputs are frozen."""
    from torch.utils.checkpoint import checkpoint
    for block in base.encoder.block:
        original=block.forward
        def call(*args,_f=original,**kwargs):
            if torch.is_grad_enabled(): return checkpoint(_f,*args,use_reentrant=False,**kwargs)
            return _f(*args,**kwargs)
        block.forward=call


def load_base(config,device,model_dir=None):
    from chronos import Chronos2Pipeline
    start=time.perf_counter()
    kwargs=dict(device_map=device,torch_dtype=torch.float32)
    if model_dir:
        pipeline=Chronos2Pipeline.from_pretrained(str(model_dir),local_files_only=True,**kwargs)
    else:
        pipeline=Chronos2Pipeline.from_pretrained(config['backbone'],revision=config['model_revision'],**kwargs)
    base=pipeline.model; base.requires_grad_(False); freeze_stochastic(base)
    shape=config['model_shape_contract']
    actual=dict(d_model=base.model_dim,num_layers=len(base.encoder.block),num_quantiles=base.num_quantiles,
                output_patch_size=base.chronos_config.output_patch_size)
    for key,value in actual.items():
        if value!=shape[key]: raise Blocked(f'MODEL_SHAPE_CHANGED {key}: {value} vs {shape[key]}')
    return base,time.perf_counter()-start


class Predictor(nn.Module):
    def __init__(self,base,arm,config,seed,inject=True):
        super().__init__(); self.base=base; self.arm=arm; self.config_contract=config
        seed_all(seed); self.base.requires_grad_(False); freeze_stochastic(base)
        self.nq=base.num_quantiles; self.patch=base.chronos_config.output_patch_size
        self.arcsinh=base.chronos_config.use_arcsinh
        self.target_names=[]
        if arm in ('LORA','LORAPLUS'):
            self.target_names=lora_targets(base)
            if inject:
                from peft import LoraConfig,inject_adapter_in_model
                inject_adapter_in_model(LoraConfig(r=config['lora_rank'],lora_alpha=config['lora_alpha'],
                    lora_dropout=0.,bias='none',target_modules=self.target_names),base)
            if config.get('gradient_checkpointing'): enable_checkpointing(base)
        self.eval()
    def encode(self,x,groups,horizon):
        count=math.ceil(horizon/self.patch)
        enc,(loc,scale),_,_=self.base.encode(context=x,group_ids=groups,num_output_patches=count)
        h=enc.last_hidden_state[:,-count:]
        q=to_quantiles(self.base.output_patch_embedding(h),self.nq,self.patch)
        return h,q,loc,scale
    def forward(self,x,groups,horizon):
        _,q,loc,scale=self.encode(x,groups,horizon)
        return q,raw_inverse(q,loc,scale,self.arcsinh)[...,:horizon],loc,scale


class CachedHead(nn.Module):
    """Original S1 output adaptation family; no newly proposed architecture."""
    def __init__(self,base,kind,width,seed):
        super().__init__(); seed_all(seed); self.kind=kind; self.nq=base.num_quantiles
        self.patch=base.chronos_config.output_patch_size; self.arcsinh=base.chronos_config.use_arcsinh
        self.register_buffer('levels',base.quantiles.detach().clone())
        if kind=='mlp':
            self.head=nn.Sequential(nn.Linear(base.model_dim,width),nn.ReLU(),nn.Linear(width,self.nq*self.patch))
            nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
        elif kind=='full':
            self.head=copy.deepcopy(base.output_patch_embedding); self.head.requires_grad_(True)
        else: raise ValueError(kind)
        self.eval()
    def forward(self,hidden,base_norm,loc,scale,horizon):
        delta=to_quantiles(self.head(hidden),self.nq,self.patch)
        q=base_norm.float()+delta if self.kind=='mlp' else delta
        return q,raw_inverse(q,loc,scale,self.arcsinh)[...,:horizon],loc,scale


def build_optimizer(model,arm,config):
    ps=[p for p in model.parameters() if p.requires_grad]
    options=dict(betas=(.9,.999),eps=1e-8,weight_decay=0.)
    if arm!='LORAPLUS':
        return torch.optim.AdamW(ps,lr=config['head_learning_rate'] if arm=='HEAD' else config['lora_learning_rate'],**options)
    # For this contract the only trainables are LoRA A/B, with no embeddings/biases.
    aa=[]; bb=[]
    for n,p in model.named_parameters():
        if not p.requires_grad: continue
        if 'lora_A' in n: aa.append(p)
        elif 'lora_B' in n: bb.append(p)
        else: raise Blocked('LORAPLUS_UNCLASSIFIED_PARAMETER '+n)
    if not aa or not bb: raise Blocked('LORAPLUS_EMPTY_GROUP')
    lr=config['loraplus_a_learning_rate']
    return torch.optim.AdamW([{'params':aa,'lr':lr,'group_name':'A'},
                             {'params':bb,'lr':lr*config['loraplus_ratio'],'group_name':'B'}],**options)


def check_loraplus_reference(model,config):
    """A/B learning-rate mapping must agree with official PEFT optimizer construction."""
    from peft.optimizers import create_loraplus_optimizer
    ours=build_optimizer(model,'LORAPLUS',config)
    ref=create_loraplus_optimizer(model=model,optimizer_cls=torch.optim.AdamW,
        lr=config['loraplus_a_learning_rate'],loraplus_lr_ratio=config['loraplus_ratio'],
        betas=(.9,.999),eps=1e-8,weight_decay=0.)
    def mapping(opt): return {id(p):float(g['lr']) for g in opt.param_groups for p in g['params']}
    if mapping(ours)!=mapping(ref): raise Blocked('LORAPLUS_OFFICIAL_GROUP_MISMATCH')
    return {'official_lr_mapping_equal':True,'ratio':config['loraplus_ratio']}
