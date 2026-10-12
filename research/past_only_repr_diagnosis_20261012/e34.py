"""Matched exact-prefix exports and frozen-D predictor diagnosis, no E refit."""
import argparse
import copy
import gc
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from common import *
from models_io import *

DEVICE='cuda'
ENCODER_BATCH=32


def synchronize():
    if DEVICE=='cuda':
        torch.cuda.synchronize()


@torch.no_grad()
def decode(decoder, z, device=None, batch=8):
    device=device or DEVICE
    return np.concatenate([decoder(torch.as_tensor(z[s:s+batch],device=device,dtype=torch.float32)).cpu().numpy()
                           for s in range(0,len(z),batch)])


def fit_decoder(initial, z, data, seed, budget, folder):
    device=DEVICE
    decoder=copy.deepcopy(initial).to(device)
    opt=torch.optim.Adam(decoder.parameters(),lr=1e-3)
    rng=np.random.default_rng(seed+33)
    targets={s:windows(data,s) for s in z}
    observed={s:masks(data,s,False) for s in z}
    for split,oo in data['origins'].items():
        lower={'fit':data['fit_bounds'][0],'tune':data['tune_bounds'][0],'val':data['val_bounds'][0]}[split]
        observed[split] &= (oo[:,None]-L+np.arange(L))[...,None]>=lower
    train=torch.from_numpy(z['fit']).to(device)
    target=torch.from_numpy(targets['fit']).to(device)
    mask=torch.from_numpy(observed['fit']).to(device)
    curve=[];best=float('inf');selected=None
    for step in range(601):
        if step%100==0:
            decoder.eval()
            p=decode(decoder,z['tune'])
            m=score(p*data['std'],targets['tune']*data['std'],observed['tune'])
            curve.append({'update':step,'tune_rec_mse':m['mse'],'tune_rec_mae':m['mae']})
            state={k:v.detach().cpu().clone() for k,v in decoder.state_dict().items()}
            torch.save({'state_dict':state,'update':step},folder/f'D_{step}.pt')
            if m['mse']<best:
                best=m['mse'];selected=step;best_state=state
            budget.check()
        if step==600:
            break
        decoder.train()
        index=rng.choice(len(train),size=min(8,len(train)),replace=False)
        opt.zero_grad(set_to_none=True)
        loss=masked_loss(decoder(train[index]),target[index],mask[index])
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite D training')
        loss.backward();opt.step()
        if step%20==0:
            budget.check()
    decoder.load_state_dict(best_state);decoder.eval().requires_grad_(False)
    reconstruction=decode(decoder,z['val'])
    rec=score(reconstruction*data['std']+data['mean'],targets['val']*data['std']+data['mean'],observed['val'])
    rec['normalized']=score(reconstruction,targets['val'],observed['val'])
    info={'updates':600,'actual_batch':min(8,len(train)),'seed':seed,'batch_order_seed':seed+33,
          'initial_state_hash':state_digest(initial),'selected_update':selected,'selection_boundary':selected==600,
          'selected_D_hash':state_digest(decoder),'curve':curve,'reconstruction':rec,
          'reconstruction_target_mask':'original observed timestamps within each scored split; earlier history context unscored',
          'selection':'TUNE observed physical reconstruction macroMSE at 0/100 updates'}
    return decoder,info,reconstruction


def save_path(folder,name,pred,data,extra=None):
    arrays=folder/f'{name}.npz'
    np.savez_compressed(arrays,prediction=pred,target=windows(data,'val',True),mask=masks(data,'val'),
                        origins=data['origins']['val'],mean=data['mean'],std=data['std'])
    return {'method':name,'prediction_path':str(arrays),'prediction_hash':sha(arrays),
            'metrics':physical_score(data,pred,'val'),**(extra or {})}


def latent_lag(z,channel=None):
    lagged=z[:,-1-np.arange(33)]
    return lagged.reshape(len(z),-1) if channel is None else lagged[:,:,channel]


def latent_ridge(z,ref,data,decoder,joint,budget,folder):
    dim=z['fit'].shape[-1]
    groups=[(None,np.arange(H*dim))] if joint else [(c,np.arange(c,H*dim,dim)) for c in range(dim)]
    yf=ref['fit'].reshape(len(ref['fit']),-1)
    valid={s:np.repeat(masks(data,s).all(-1,keepdims=True),dim,-1) for s in ('fit','tune','val')}
    mf=valid['fit'].reshape(len(yf),-1)
    by_mask=[]
    for channel,indices in groups:
        masks_group={}
        for j in indices:
            masks_group.setdefault(mf[:,j].tobytes(),[]).append(int(j))
        by_mask.extend((channel,js,mf[:,js[0]]) for js in masks_group.values())
    candidates=[];arrays=[]
    for lam in LAMBDAS:
        predicted={s:np.zeros_like(ref[s],dtype=np.float64) for s in ('tune','val')}
        saved={}
        for group,(channel,js,rows) in enumerate(by_mask):
            model=Ridge(lam).fit(latent_lag(z['fit'],channel)[rows],yf[rows][:,js])
            for s in predicted:
                predicted[s].reshape(len(predicted[s]),-1)[:,js]=model.predict(latent_lag(z[s],channel))
            saved[f'group{group}_mean']=model.mean
            saved[f'group{group}_std']=model.std
            saved[f'group{group}_ym']=model.ym
            saved[f'group{group}_train']=model.train
            saved[f'group{group}_coef' if model.coef is not None else f'group{group}_dual']=model.coef if model.coef is not None else model.dual
            saved[f'group{group}_outputs']=np.asarray(js)
            budget.check()
        pt=decode(decoder,predicted['tune'])
        pe=decode(decoder,predicted['val'])
        metric=physical_score(data,pt,'tune')
        candidates.append(metric['mse'])
        arrays.append((pe,predicted['val'],saved))
    k=int(np.argmin(candidates));name='LATENT_JOINT_RIDGE' if joint else 'LATENT_SELF_RIDGE'
    np.savez_compressed(folder/f'{name}_FIT_STATE.npz',**arrays[k][2])
    latent_metric=score(arrays[k][1],ref['val'],valid['val'])
    row=save_path(folder,name,arrays[k][0],data,{'lambda':LAMBDAS[k],'tune_candidate_mse':candidates,
                    'latent_metrics':latent_metric,'latent_mask_hash':ahash(valid['val']),'fixed_D_hash':state_digest(decoder),
                    'FIT_latent_target_scope':'same-a prefix; current timestamp all raw channels observed; no eval labels in fit',
                    'supervised':True,'features':33*(dim if joint else 1)})
    np.savez_compressed(folder/f'{name}_LATENT_PREDICTION.npz',prediction=arrays[k][1],mask=valid['val'])
    return row


def benchmark(encoder,decoder,bolt,data,mode,budget):
    result=[]
    origins=data['origins']['val']
    raw=np.stack([data['physical'][o-L:o] for o in origins])
    observed=masks(data,'val',False)
    initial=np.stack([data['filled'][o-L-1] if o-L>0 else data['mean'] for o in origins])
    for n in (1,32):
        selected=np.arange(n)%len(origins)
        raw_batch,mask_batch,initial_batch=raw[selected],observed[selected],initial[selected]
        if DEVICE=='cuda':
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        baseline_allocated=torch.cuda.memory_allocated() if DEVICE=='cuda' else 0
        samples=[]
        for run in range(25):
            budget.check();synchronize();start=time.perf_counter()
            filled=np.empty_like(raw_batch);last=initial_batch.copy()
            for t in range(L):
                last[mask_batch[:,t]]=raw_batch[:,t][mask_batch[:,t]]
                filled[:,t]=last
            v=((filled-data['mean'])/data['std']).astype(np.float32)
            z=encode_full(encoder,v,DEVICE,batch=ENCODER_BATCH) if mode=='FULL' else encode_prefix(encoder,v,DEVICE,batch=ENCODER_BATCH)
            p=decode(decoder,bolt.forecast(z));p.copy()
            synchronize();elapsed=time.perf_counter()-start
            if run>=5:
                samples.append(elapsed)
            budget.check()
        result.append({'batch':n,'unique_inputs':min(len(origins),n),'repeated_inputs':max(0,n-len(origins)),
                       'warmup':5,'timed':20,'samples_seconds':samples,'p50_seconds':float(np.median(samples)),
                       'end_to_end':'causal ffill + frozen normalization + exact export + Bolt + D + CPU output; load/disk/guards excluded',
                       'frozen_imputation_initial_state':'carried physical state at a-1; matches old preprocessing; not counted as extra512observations',
                       'device':DEVICE,'baseline_allocated_vram':baseline_allocated,'peak_scope':'case-reset, persistent process model footprint',
                       'peak_allocated_vram':torch.cuda.max_memory_allocated() if DEVICE=='cuda' else 0,
                       'peak_reserved_vram':torch.cuda.max_memory_reserved() if DEVICE=='cuda' else 0})
    return result


def profile(budget):
    data=load_data('jena',frozen=True,sizes=(64,32,32))
    encoder,record=load_ts2('jena',16,SEEDS[0],DEVICE)
    x=windows(data,'fit')[:32]
    timings=[]
    for actual_batch,length in [(b,k) for b in (1,32) for k in (1,64,256,512,560)]:
        t=torch.as_tensor(np.concatenate((x,x[:,:48]),1)[:actual_batch,:length],device=DEVICE)
        samples=[]
        for k in range(4):
            synchronize();start=time.perf_counter()
            with torch.no_grad():
                encoder(t.clone(),mask='all_true')[:,-1].cpu()
            synchronize()
            if k:
                samples.append(time.perf_counter()-start)
        timings.append({'length':length,'batch':actual_batch,'p50_seconds':float(np.median(samples))})
        budget.check()
    b1=float(np.mean([v['p50_seconds'] for v in timings if v['batch']==1 and v['length']<=512]))
    b32=float(np.mean([v['p50_seconds'] for v in timings if v['batch']==32 and v['length']<=512]))
    # Include both exports,48 reference evaluations, D fits,F and mandatory25xB1/B32prefix timings.
    benchmark_time=25*512*(b1+b32)
    estimated_subset_pair=2*(560*4*b32+benchmark_time+120)*2
    remaining=(5400 if DEVICE=='cuda' else 3600)-sum(read(p)['wall_seconds'] for p in (HERE/'resources').glob('*.json') if read(p)['gpu']==(DEVICE=='cuda') and (DEVICE!='cuda' or read(p)['phase']==1))
    # Unmeasured d320 bundles are explicitly held; this initial pair is chosen by measured cost only.
    scope='DIAG_SUBSET_JENA_D16_PAIR'
    decision={'status':'COST_ONLY_SCOPE_FROZEN','timings':timings,'scope':scope,
              'estimated_subset_Jena_d16_pair_seconds':estimated_subset_pair,'pair_fits_remaining':estimated_subset_pair<remaining*.85,
              'remaining_numeric_seconds':remaining,'science_metrics_seen':False,'checkpoint':record,'device':DEVICE,'encoder_batch':ENCODER_BATCH,
              'uncertainty':'Twofold margin includes D/ridge/F overhead; unmeasured d320/Robin held; timing forecast not guarantee',
              'mandatory_benchmark_may_dominate':True}
    write(HERE/f'E34_COST_SCOPE_{DEVICE}.json',decision)
    del encoder;gc.collect()
    if DEVICE=='cuda':
        torch.cuda.empty_cache()
    return decision


def run_one(ds,dim,seed,scope,bolt,budget,tag,bench=True):
    source_start={p.name:sha(p) for p in [Path(__file__),HERE/'models_io.py',HERE/'common.py']}
    sizes=(256,64,None) if scope=='FULL_VAL_ALL8' else (64,32,32)
    data=load_data(ds,frozen=True,sizes=sizes)
    folder=CACHE/'e34'/f'{tag}_{ds}_{dim}_{seed}'
    folder.mkdir(parents=True,exist_ok=False)
    encoder,record=load_ts2(ds,dim,seed,DEVICE)
    before=state_digest(encoder)
    xs={s:windows(data,s) for s in ('fit','tune','val')}
    zfull={s:encode_full(encoder,x,DEVICE,batch=ENCODER_BATCH,budget=budget) for s,x in xs.items()}
    zprefix={s:encode_prefix(encoder,x,DEVICE,batch=ENCODER_BATCH,budget=budget) for s,x in xs.items()}
    last_diffs={s:float(np.max(np.abs(zfull[s][:,-1]-zprefix[s][:,-1]))) for s in xs}
    if any(v>2e-5 for v in last_diffs.values()):
        raise ValueError('FULL last vs same-a PREFIX last failed')
    torch.manual_seed(seed+100)
    initial=ReconstructionDecoder(dim,len(data['columns']))
    e3rows=[];decs={};decoder_info={}
    for mode,z in [('FULL',zfull),('PREFIX',zprefix)]:
        dest=folder/mode;dest.mkdir()
        decoder,di,rec=fit_decoder(initial,z,data,seed,budget,dest)
        decs[mode]=decoder;decoder_info[mode]=di
        f=bolt.forecast(z['val'],budget=budget)
        p=decode(decoder,f)
        row=save_path(dest,mode+'_BOLT_REC_D',p,data,{'decoder':di,'fixed_E_hash':before,'checkpoint_hash':record['checkpoint_sha256']})
        recmask=masks(data,'val',False)
        recmask &= (data['origins']['val'][:,None]-L+np.arange(L))[...,None]>=data['val_bounds'][0]
        np.savez_compressed(dest/'RECONSTRUCTION.npz',prediction=rec,target=xs['val'],mask=recmask,mean=data['mean'],std=data['std'])
        np.savez_compressed(dest/'EXPORTS.npz',**{s:z[s] for s in z})
        if bench:
            row['end_to_end_cost']=benchmark(encoder,decoder,bolt,data,mode,budget)
        e3rows.append(row)
        budget.check()
    decoder=decs['PREFIX'];dest=folder/'E4';dest.mkdir()
    refs={}
    for s in xs:
        extended=np.stack([data['x'][o-L:o+H] for o in data['origins'][s]])
        refs[s]=encode_prefix(encoder,extended,DEVICE,batch=ENCODER_BATCH,first_tick=L,budget=budget)
    diagnostic=dest/'REFERENCE_LATENT_DIAGNOSTIC.npz'
    np.savez_compressed(diagnostic,**refs)
    rows=[]
    pp=np.repeat(zprefix['val'][:,-1:],H,1)
    latent_mask=np.repeat(masks(data,'val').all(-1,keepdims=True),dim,-1)
    np.savez_compressed(dest/'LATENT_PERSISTENCE_LATENT.npz',prediction=pp,mask=latent_mask,reference=refs['val'])
    rows.append(save_path(dest,'LATENT_PERSISTENCE',decode(decoder,pp),data,{'fixed_D_hash':state_digest(decoder),
                   'latent_metrics':score(pp,refs['val'],latent_mask),'latent_mask_hash':ahash(latent_mask)}))
    rows.append(latent_ridge(zprefix,refs,data,decoder,False,budget,dest))
    rows.append(latent_ridge(zprefix,refs,data,decoder,True,budget,dest))
    fb=bolt.forecast(zprefix['val'],budget=budget)
    np.savez_compressed(dest/'BOLT_LATENT_PREDICTION.npz',prediction=fb,mask=latent_mask,reference=refs['val'])
    rows.append(save_path(dest,'BOLT_LATENT',decode(decoder,fb),data,{'fixed_D_hash':state_digest(decoder),
                      'latent_metrics':score(fb,refs['val'],latent_mask),'latent_mask_hash':ahash(latent_mask),
                      'information_difference':'Bolt uses512latentticks, ridge uses33; frozenpretraining vsFITsupervision'}))
    rows.append(save_path(dest,'REFERENCE_LATENT_DIAGNOSTIC',decode(decoder,refs['val']),data,
                         {'DEPLOYABLE':False,'reference_path':str(diagnostic),'not_lower_bound':True,'fixed_D_hash':state_digest(decoder)}))
    pred,info,fits=ridge_grouped(latent_lag(zprefix['fit']),windows(data,'fit',True),masks(data,'fit'),
                                latent_lag(zprefix['tune']),windows(data,'tune',True),masks(data,'tune'),
                                latent_lag(zprefix['val']),data['std'],budget)
    rows.append(save_path(dest,'DIRECT_RAW_RIDGE',pred,data,{'ridge':info,'input':'same PREFIX latent lag0..32; outputoriginalchannels; noD'}))
    fit_arrays={}
    for i,(js,model) in enumerate(fits):
        fit_arrays.update({f'g{i}_outputs':np.asarray(js),f'g{i}_mean':model.mean,f'g{i}_std':model.std,f'g{i}_ym':model.ym,
                          f'g{i}_train':model.train,f'g{i}_coef' if model.coef is not None else f'g{i}_dual':model.coef if model.coef is not None else model.dual})
    np.savez_compressed(dest/'DIRECT_RAW_RIDGE_FIT_STATE.npz',**fit_arrays)
    if state_digest(encoder)!=before:
        raise ValueError('Frozen E changed')
    if sha(data['source'])!=data['source_hash'] or sha(record['checkpoint'])!=record['checkpoint_sha256'] or any(sha(HERE/k)!=v for k,v in source_start.items()):
        raise ValueError('Data/checkpoint source bytes changed during run')
    result={'status':'COMPLETE_UNVERIFIED','experiment_id':'E3_E4','dataset':ds,'dim':dim,'seed':seed,'scope':scope,
            'folder':str(folder),'checkpoint_record':record,'E3':e3rows,'E4':rows,'E_hash_before_after':before,
            'FULL_PREFIX_last_deltas':last_diffs,'FULL_PREFIX_all_deltas':{s:float(np.max(np.abs(zfull[s]-zprefix[s]))) for s in xs},
            'fit_bounds':data['fit_bounds'],'tune_bounds':data['tune_bounds'],'eval_bounds':data['val_bounds'],
            'origin_hashes':{s:ahash(data['origins'][s]) for s in xs},'normalization_fit_scope':data['normalization_fit_scope'],
            'source_hashes':{p.name:sha(p) for p in [Path(__file__),HERE/'models_io.py',HERE/'common.py']},
            'safety_gate_sha':sha(HERE/'E1_GATE.json'),'E3_same_D_initialization':decoder_info['FULL']['initial_state_hash']==decoder_info['PREFIX']['initial_state_hash'],
            'E4_reference_isolation':'Reference future only separate diagnostic and FIT/TUNE latent labels; never Bolt input',
            'device':DEVICE,'encoder_batch':ENCODER_BATCH,'supervised_origin_counts':{s:len(v) for s,v in data['origins'].items()},
            'unique_target_timestamps':{s:len(np.unique(v[:,None]+np.arange(H))) for s,v in data['origins'].items()},
            'parameter_counts':{'E':sum(p.numel() for p in encoder.parameters()),'D':sum(p.numel() for p in decoder.parameters())},
            'source_hashes_at_start':source_start,
            'remaining_uncertainties':['historicallyexposedE','repeatedVAL','two seeds not independent generalization','boundaryselectedD notconvergence','informationcapacity mismatch Bolt/ridge','frozen global imputation initial state may contain earlier past at a-1']}
    write(HERE/'e34'/f'{tag}_{ds}_{dim}_{seed}.json',result)
    del encoder,decoder,decs;gc.collect()
    if DEVICE=='cuda':
        torch.cuda.empty_cache()
    return result


def main():
    global DEVICE
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default='run01');parser.add_argument('--profile-only',action='store_true')
    parser.add_argument('--run',action='store_true');parser.add_argument('--device',choices=('cpu','cuda'),default='cuda');args=parser.parse_args();DEVICE=args.device
    access_guard();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not read(HERE/'E1_GATE.json')['safety_pass']:
        raise ValueError('E1 gate')
    # Root serializes jobs and verifies existing controller/lease terminal before starting this script.
    with threadpool_limits(2),Budget(1,DEVICE=='cuda','E3_E4_'+args.tag) as budget:
        path=HERE/f'E34_COST_SCOPE_{DEVICE}.json'
        decision=read(path) if path.exists() else profile(budget)
        if args.profile_only:
            print(json.dumps(decision,default=jsonable));return
        if not args.run:
            raise ValueError('Explicit --run required')
        if not decision['pair_fits_remaining']:
            raise BudgetHeld('Measured balanced Jena pair does not fit current available device budget')
        bolt=Bolt(DEVICE);parity=bolt.parity()
        jobs=[('jena',16,s) for s in SEEDS]
        if decision['scope']!='DIAG_SUBSET_JENA_D16_PAIR':
            jobs += [('jena',320,s) for s in SEEDS]+[(d,k,s) for d in ('robin',) for k in (16,320) for s in SEEDS]
        results=[]
        for ds,dim,seed in jobs:
            results.append(run_one(ds,dim,seed,decision['scope'],bolt,budget,args.tag))
            print(f'E3_E4 {ds} d{dim} seed{seed} COMPLETE_SAVED',flush=True)
        if state_digest(bolt.model)!=bolt.initial:
            raise ValueError('Bolt weights changed')
        write(HERE/'e34'/f'SUMMARY_{args.tag}.json',{'status':'COMPLETE_UNVERIFIED','jobs':jobs,
            'result_paths':[str(HERE/'e34'/f'{args.tag}_{d}_{k}_{s}.json') for d,k,s in jobs],
            'cost_scope':decision,'Bolt_parity':parity,'Bolt_unchanged':True,'GPU_ledger':str(budget.path)})


if __name__=='__main__':
    main()
