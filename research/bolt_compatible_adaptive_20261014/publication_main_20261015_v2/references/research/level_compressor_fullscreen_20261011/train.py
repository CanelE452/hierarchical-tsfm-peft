"""Fresh fixed-update training. The foundation model is never constructed here."""
from pathlib import Path
import argparse
import json
import time

import numpy as np
import torch
from torch import nn

from common import HERE, CACHE, Tracker, save_json, read_json, windows, state_hash, array_hash, sha
import models


def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def sample_tensors(data, origins, device):
    x = windows(data, origins)
    y = windows(data, origins, future=True)
    mx = windows(data, origins, mask=True)
    my = windows(data, origins, future=True, mask=True)
    # Historical context remains an input; fitting supervision is restricted to the declared fit interval.
    low, high = data['fit_bounds']
    for i, o in enumerate(origins):
        t = np.arange(int(o)-512, int(o))
        mx[i] &= ((t >= low) & (t < high))[:, None]
    return tuple(torch.as_tensor(v, device=device) for v in (x, y, mx, my))


def train_encoder(job, data, tracker, device='cuda'):
    folder = CACHE/'checkpoints'/job['id']
    folder.mkdir(parents=True, exist_ok=True)
    final_path = folder/'final.pt'
    receipt_path = folder/'receipt.json'
    if final_path.exists() and receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt['checkpoint_sha256'] != sha(final_path) or receipt['data_contract_hash'] != data['contract_hash']:
            raise RuntimeError('stale encoder cache')
        model = models.build_model(job['structure'], data['x'].shape[1], job['K'], job['seed'], job['arm'], data.get('graph'))
        model.load_state_dict(torch.load(final_path,map_location='cpu',weights_only=False)['state'])
        return model.to(device).eval(), receipt
    if any(folder.iterdir()):
        raise RuntimeError('partial fit exists: explicit separate attempt required; no overwrite')
    torch.manual_seed(job['seed'])
    model = models.build_model(job['structure'], data['x'].shape[1], job['K'], job['seed'], job['arm'], data.get('graph')).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=.001)
    pool = np.asarray(data['train_fit_origins'], dtype=np.int64)
    rng = np.random.default_rng(92603)
    indices = rng.integers(0,len(pool),size=(2880,32),dtype=np.int32)
    sample_path = folder/'sample_indices.npz'
    np.savez_compressed(sample_path,indices=indices,origins=pool)
    curves=[]; best_score=float('inf'); best_step=0
    train_start=time.perf_counter(); cpu_start=time.process_time()
    with tracker.stage('encoder_train',gpu=device.startswith('cuda'),job=job['id']):
        for step in range(2881):
            if step:
                model.train()
                tensors=sample_tensors(data,pool[indices[step-1]],device)
                loss=models.losses(model,*tensors,arm=job['arm'])
                if not torch.isfinite(loss['total']):
                    raise RuntimeError('NaN/Inf loss')
                optimizer.zero_grad(set_to_none=True)
                loss['total'].backward()
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(grad):
                    raise RuntimeError('NaN/Inf gradient')
                optimizer.step()
            if step % 96 == 0:
                model.eval()
                with torch.no_grad():
                    eval_origins=np.asarray(data['selection_origins'])
                    if len(eval_origins)>32:
                        eval_origins=eval_origins[np.linspace(0,len(eval_origins)-1,32,dtype=int)]
                    ev=sample_tensors(data,eval_origins,device)
                    # Selection labels are held out TRAIN, not fit supervision; restore their true masks.
                    ev=(ev[0],ev[1],torch.as_tensor(windows(data,eval_origins,mask=True),device=device),ev[3])
                    val_loss=models.losses(model,*ev,arm=job['arm'])
                row={'step':step,'selection':{k:float(v.detach().cpu()) for k,v in val_loss.items()},
                     'train':{k:float(v.detach().cpu()) for k,v in loss.items()} if step else None}
                curves.append(row)
                score=row['selection']['total']
                if score<best_score:
                    best_score=score;best_step=step
                    torch.save({'state':cpu_state(model),'step':step,'criterion':'TRAIN_INTERNAL total objective'},folder/'internal_best.pt')
                save_json(folder/'curve.json',curves)
                tracker.check(job=job['id'],stage='encoder_train',updates=step,planned_updates=2880)
                print(json.dumps({'job':job['id'],'updates':step,'selection_rec':row['selection']['rec']}),flush=True)
            if step in (0,960,1920,2880):
                torch.save({'state':cpu_state(model),'step':step},folder/f'step_{step}.pt')
        torch.save({'state':cpu_state(model),'step':2880,'optimizer':optimizer.state_dict(),
                    'rng_numpy':rng.bit_generator.state,'rng_torch':torch.get_rng_state(),'job':job},final_path)
    model.eval()
    receipt={'job':job,'actual_optimizer_updates':2880,'actual_fit_starts':1,
             'trainable_params':sum(p.numel() for p in model.parameters() if p.requires_grad),
             'resident_params':sum(p.numel() for p in model.parameters()),
             'F_forward_calls':0,'F_backward_calls':0,'data_contract_hash':data['contract_hash'],
             'sample_indices_sha256':sha(sample_path),'pool_origins_hash':array_hash(pool),
             'checkpoint_sha256':sha(final_path),'encoder_state_hash':state_hash(model),
             'wall_seconds':time.perf_counter()-train_start,'cpu_seconds':time.process_time()-cpu_start,
             'primary_checkpoint':'final2880','secondary_internal_best_step':best_step,
             'checkpoint_path':final_path,'curve_path':folder/'curve.json'}
    save_json(receipt_path,receipt)
    return model,receipt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from data_contract import load_dataset
    manifest=read_json(HERE/'RUN_MANIFEST.json')
    job=next(j for j in manifest['primary_jobs']+manifest['additional_jobs'] if j['id']==args.job)
    data=load_dataset(job['data'],fold=job.get('fold'))
    model,receipt=train_encoder(job,data,Tracker())
    print(json.dumps({'complete':args.job,'actual_updates':receipt['actual_optimizer_updates'],
                      'wall_seconds':receipt['wall_seconds']}),flush=True)


if __name__=='__main__':
    main()
