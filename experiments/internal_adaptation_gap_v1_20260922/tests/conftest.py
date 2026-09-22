from __future__ import annotations
import copy,sys,types
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gap_screen import model as gm

class TinyConfig(SimpleNamespace):
    def to_dict(self):return dict(self.__dict__)

class TinyAttention(nn.Module):
    def __init__(self,d):
        super().__init__()
        for key in ('q','k','v','o'):setattr(self,key,nn.Linear(d,d,bias=False))
    def forward(self,x):return self.o(torch.tanh(self.q(x)+self.k(x)+self.v(x)))

class TinyBlock(nn.Module):
    def __init__(self,d):super().__init__();self.self_attention=TinyAttention(d)
    def forward(self,x):return x+self.self_attention(x)

class TinyOutput(nn.Module):
    def __init__(self,d,nout):
        super().__init__();self.output_layer=nn.Linear(d,nout);self.residual_layer=nn.Linear(d,nout)
    def forward(self,x):return self.output_layer(x)+self.residual_layer(x)

class TinyBase(nn.Module):
    """A test double for API/control-flow checks; NOT Chronos and NOT forecasting evidence."""
    def __init__(self):
        super().__init__();self.model_dim=8;self.num_quantiles=3
        self.config=TinyConfig(dropout_rate=0.,d_model=8,num_layers=1)
        self.chronos_config=SimpleNamespace(output_patch_size=2,input_patch_size=2,use_arcsinh=True,context_length=256)
        self.register_buffer('quantiles',torch.tensor([.1,.5,.9]))
        self.input_patch_embedding=nn.Linear(2,8)
        self.encoder=nn.Module();self.encoder.block=nn.ModuleList([TinyBlock(8)])
        self.output_patch_embedding=TinyOutput(8,6)
    def encode(self,context,group_ids,num_output_patches,**ignored):
        loc=context.mean(-1,keepdim=True);sc=context.std(-1,unbiased=False,keepdim=True).clamp_min(.01)
        n=((context-loc)/sc).asinh();z=self.input_patch_embedding(n.reshape(len(context),-1,2))
        same=(group_ids[:,None]==group_ids[None,:]).to(z)
        pooled=torch.einsum('ij,jtd->itd',same,z)/same.sum(-1)[:,None,None]
        z=z+.1*pooled
        fut=z.mean(1,keepdim=True).expand(-1,num_output_patches,-1)
        z=torch.cat([z,fut],1)
        for b in self.encoder.block:z=b(z)
        return SimpleNamespace(last_hidden_state=z),(loc,sc),None,None
    def forward(self,context,group_ids,num_output_patches,future_target=None,**kw):
        enc,(loc,sc),_,_=self.encode(context,group_ids,num_output_patches)
        q=gm.to_quantiles(self.output_patch_embedding(enc.last_hidden_state[:,-num_output_patches:]),3,2)
        raw=gm.raw_inverse(q,loc,sc,True)
        loss=None if future_target is None else gm.normalized_loss(q,future_target,loc,sc,self.quantiles,True)
        return SimpleNamespace(quantile_preds=raw,loss=loss)

class FakeLinear(nn.Module):
    def __init__(self,linear,r,alpha):
        super().__init__();self.base_layer=linear
        self.lora_A=nn.ModuleDict({'default':nn.Linear(linear.in_features,r,bias=False)})
        self.lora_B=nn.ModuleDict({'default':nn.Linear(r,linear.out_features,bias=False)})
        nn.init.zeros_(self.lora_B['default'].weight);self.scale=alpha/r
    def forward(self,x):return self.base_layer(x)+self.lora_B['default'](self.lora_A['default'](x))*self.scale

def fake_inject(conf,base):
    for name in conf.target_modules:
        parent_name,attr=name.rsplit('.',1);parent=base.get_submodule(parent_name)
        setattr(parent,attr,FakeLinear(getattr(parent,attr),conf.r,conf.lora_alpha))
    return base

@pytest.fixture
def fake_peft(monkeypatch):
    mod=types.ModuleType('peft');mod.LoraConfig=lambda **kw:SimpleNamespace(**kw);mod.inject_adapter_in_model=fake_inject
    monkeypatch.setitem(sys.modules,'peft',mod)

@pytest.fixture
def cfg():
    from gap_screen.io import read_json
    c=read_json(Path(__file__).parents[1]/'config.json');c=copy.deepcopy(c)
    c.update(max_steps=4,checkpoints=[0,2,4],accumulate_origins=2,precision='fp32',gradient_checkpointing=False,
        max_main_updates=48,max_smoke_updates=12,lora_rank=2,lora_alpha=4,head_mlp_width=8,
        minimum_train_origins=4,minimum_validation_origins=2,minimum_test_origins=2,
        bootstrap_repetitions=40,min_free_ram_gib=0.)
    c['model_shape_contract'].update(d_model=8,num_layers=1,num_quantiles=3,output_patch_size=2,lora_modules=5,lora_parameters=156)
    return c

@pytest.fixture
def tiny_base():
    torch.manual_seed(77);return TinyBase()
