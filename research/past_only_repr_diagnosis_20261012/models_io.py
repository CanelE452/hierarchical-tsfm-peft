"""Read-only old checkpoints, exact common-start exports, local frozen Bolt."""
import sys
import time
import numpy as np
import torch
from torch import nn
from common import *

FOLLOWUP = ROOT / 'research/professor_compressor_followup_20261012_run01'
sys.path.insert(0,str(DIRECT))
sys.path.insert(0,str(FOLLOWUP))
from ts2_export_followup import load_frozen_encoder
from ts2_adapter import ReconstructionDecoder
from common_direct import state_digest
from models_gt import GTConvAE


def load_ts2(dataset, dim, seed, device='cpu'):
    model, record = load_frozen_encoder(dataset, dim, seed, device)
    model.net.eval().requires_grad_(False)
    return model.net.module, record


def load_existing_decoder(dataset, dim, seed, device='cpu'):
    from ts2_export_followup import canonical_encoder_result
    r,_ = canonical_encoder_result(dataset,dim,seed)
    record = r['training']['decoder_reconstruction']
    path = ROOT/record['best_checkpoint']
    if sha(path) != record['best_checkpoint_sha256']:
        raise ValueError('Existing decoder bytes differ')
    decoder = ReconstructionDecoder(dim, r['training']['encoder']['configuration']['input_dims']).to(device)
    decoder.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state_dict'])
    decoder.eval().requires_grad_(False)
    return decoder, {'checkpoint':str(path),'checkpoint_hash':sha(path),'historical_selection':'VAL'}


def load_gt(asset, device='cpu'):
    path = Path(asset['path'])
    if sha(path) != asset['checkpoint_hash']:
        raise ValueError('GT checkpoint changed after preflight')
    saved = torch.load(path,map_location='cpu',weights_only=True)
    state = saved['state_dict']
    model = GTConvAE(state['product.graph'],no_spatial=asset['job']['no_spatial'])
    model.load_state_dict(state)
    return model.to(device).eval().requires_grad_(False)


@torch.no_grad()
def encode_full(encoder, values, device='cpu', batch=8, budget=None):
    result=[]
    for start in range(0,len(values),batch):
        t=torch.as_tensor(values[start:start+batch],device=device,dtype=torch.float32)
        result.append(encoder(t.clone(),mask='all_true').cpu().numpy())
        if budget:
            budget.check()
    return np.concatenate(result)


@torch.no_grad()
def encode_prefix(encoder, values, device='cpu', batch=8, first_tick=0, budget=None):
    """Each group has exactly the same true length; there is no right padding."""
    result=np.empty((len(values),values.shape[1]-first_tick,encoder.output_dims),np.float32)
    for start in range(0,len(values),batch):
        t=torch.as_tensor(values[start:start+batch],device=device,dtype=torch.float32)
        for end in range(first_tick+1,values.shape[1]+1):
            result[start:start+len(t),end-first_tick-1]=encoder(t[:,:end].clone(),mask='all_true')[:,-1].cpu().numpy()
            if budget and end%32==0:
                budget.check()
    return result


class Bolt:
    def __init__(self, device='cpu'):
        from chronos import ChronosBoltPipeline
        self.device=device
        self.directory=ROOT/'.cache/huggingface/models--amazon--chronos-bolt-small/snapshots/772f3d25d38aec6d914c8949dab4462e2d46f5d8'
        self.pipeline=ChronosBoltPipeline.from_pretrained(str(self.directory),device_map=device,torch_dtype=torch.float32,local_files_only=True)
        self.model=self.pipeline.model.eval().requires_grad_(False)
        self.median=list(self.model.chronos_config.quantiles).index(.5)
        self.initial=state_digest(self.model)

    @torch.no_grad()
    def forecast(self, latent, horizon=H, budget=None):
        shape=latent.shape
        t=torch.as_tensor(latent,device=self.device,dtype=torch.float32).reshape(shape[0],shape[1],-1)
        rows=t.transpose(1,2).reshape(-1,shape[1])
        result=[]
        for start in range(0,len(rows),128):
            result.append(self.model(context=rows[start:start+128]).quantile_preds[:,self.median,:horizon])
            if budget:
                budget.check()
        return torch.cat(result).reshape(shape[0],-1,horizon).transpose(1,2).reshape(shape[0],horizon,*shape[2:]).cpu().numpy()

    @torch.no_grad()
    def parity(self):
        ramp=torch.linspace(-2,3,512).reshape(1,512,1).numpy()
        ours=self.forecast(ramp)
        official=self.pipeline.predict(torch.from_numpy(ramp[:,:,0]),prediction_length=H)[:,self.median].numpy()
        error=float(np.max(np.abs(ours[:,:,0]-official)))
        if error != 0:
            raise ValueError('Bolt official median parity failed')
        return {'max_abs':error,'weights_frozen':True,'parameters':sum(p.numel() for p in self.model.parameters())}


def masked_loss(pred,target,mask):
    return torch.where(mask,(pred-target).square(),0.).sum()/mask.sum().clamp_min(1)
