"""Finite B2 mechanism jobs. Default invocation performs no asset or model IO."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

BOUNDS = dict(A_fit=[0,10482], A_stop=[10482,12579], M_cal=[12579,14675], V_select=[14675,16772])
MODEL_REL = '.cache/huggingface/models--amazon--chronos-bolt-small/snapshots/772f3d25d38aec6d914c8949dab4462e2d46f5d8'
F_SHA = 'c2b4edf7db22d3f18e58ca79299a5c7e864e9b8f277e4e29e2f88d07dbb35874'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def bootstrap(args):
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key] = '2'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['INFORMS_ASSET_ROOT'] = str(args.asset)
    os.environ['INFORMS_CACHE_ROOT'] = str(args.asset/'.cache/informs_broad_validation_20261015_v1')
    campaign = args.repo/'research/informs_broad_validation_20261015_v1'
    sys.path.insert(0, str(campaign))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    import numpy as np
    import fit_base as fit
    import training
    import data_contract as data
    import deployment
    import evaluation
    import psutil
    context = dict(torch=torch,np=np,fit=fit,training=training,data=data,deployment=deployment,
                   evaluation=evaluation,psutil=psutil,campaign=campaign)
    private = args.asset/'.cache/informs_b2_why_where_20261017'
    public = args.repo/'reports/informs_b2_why_where_20261017'
    context.update(private=private,public=public,old_cache=args.asset/'.cache/informs_broad_validation_20261015_v1')
    scope = read(private/'CAMPAIGN_v1.json')
    assert scope['base_commit']=='a5eac7f889c72cd53e2f9bb418b5f2d2c5ae3065'
    assert scope['CPU_threads_per_worker']==2 and scope['new_GPU_seconds']==0
    context['deadline'] = read(private/'EXECUTION_BOUNDS_v1.json')['deadline_unix']
    if time.time() >= context['deadline']:
        raise TimeoutError('New finite campaign execution boundary reached')
    context['scope_sha256'] = sha(private/'CAMPAIGN_v1.json')
    return context


def load_jena(args, context):
    source = context['campaign']/'priority_b2_validation_20261016_v1/safe_jena_prefix.py'
    loader = module('b2_mechanism_jena_prefix', source)
    contract = read(context['campaign']/'DATA_CONTRACT.json')['assets']['jena']
    task, access = loader.load_jena_prefix(args.asset, contract)
    assert task['materialized_rows']==16772 and task['L']==512 and task['H']==48
    return task, access


def old_run(context, seed, family):
    if seed == 92601:
        namespace='postT_V_only_cpu_lag_20261016_v1'
        identifier=f'postT_cpu_jena_E3_{family}_k6_s92601_v1'
    elif seed == 92602:
        namespace='priority_core_followup_20261016_v1'
        identifier=f'priority_seed2_jena_{family}_k6_s92602_v1'
    else:
        raise ValueError('Unregistered seed')
    return context['old_cache']/namespace/'runs'/identifier


def spec_for(seed, family='B2'):
    return dict(run_id=f'mechanism_jena_{family}_k6_s{seed}_v1',dataset='jena',family=family,
                latent_k=6,seed=seed,fold='primary',epochs=[.25,1.,3.],lr=.001,device='cpu',
                microbatch=4,effective_batch=16,F_row_microbatch=128,checkpoint_inputs=False)


def instrument(context, run_private):
    """Preserve optimization math; observe completed steps and save safety copies."""
    training, torch, psutil = (context[k] for k in ('training','torch','psutil'))
    original = training.train_segment
    sequence = [0]
    def wrapped(model, windows, train_origins, sampler, optimizer, end_exposures, predict, **kwargs):
        sequence[0] += 1
        serial = sequence[0]
        steps = [kwargs.get('starting_updates',0)]
        started=time.perf_counter()
        step=optimizer.step
        safety = run_private/'safety'/f'segment_{serial:02d}'
        safety.mkdir(parents=True,exist_ok=False)
        def observed_step(*a,**kw):
            value=step(*a,**kw)
            steps[0]+=1
            rss=psutil.Process().memory_info().rss
            current=dict(updates=steps[0],exposures=sampler.exposures,target_exposures=end_exposures,
                         segment_seconds=time.perf_counter()-started,CPU_threads=2,RSS_bytes=rss)
            if steps[0]%10==0 or rss>3*1024**3:
                temp=safety/'progress.tmp'
                temp.write_text(json.dumps(current),encoding='utf-8')
                temp.replace(safety/'progress.json')
            if steps[0]%100==0 or rss>3*1024**3:
                payload=training.checkpoint_bundle(model,optimizer,sampler,{},steps[0],dict(
                    diagnostic_safety_snapshot=True,root_scope_sha256=context['scope_sha256'],
                    not_a_selected_model=True,**current))
                with (safety/f'optimizer_update_{steps[0]:06d}.pt').open('xb') as f:torch.save(payload,f)
            if rss>3*1024**3:
                raise MemoryError('Worker RSS3GiB exceeded; optimizer/sampler safety snapshot preserved')
            return value
        optimizer.step=observed_step
        try:
            return original(model,windows,train_origins,sampler,optimizer,end_exposures,predict,**kwargs)
        finally:
            optimizer.step=step
    training.train_segment=wrapped


def score(model, normalization, task, origins, foundation, context, metric_norm=None, save=None):
    np,torch,training,fit=context['np'],context['torch'],context['training'],context['fit']
    model.eval()
    windows=training.FastWindows(task,normalization['mean'],normalization['std'],'cpu')
    canonical = normalization if metric_norm is None else metric_norm
    metric_windows = windows if metric_norm is None else training.FastWindows(task,canonical['mean'],canonical['std'],'cpu')
    sse=np.zeros(model.channels,np.float64);sae=sse.copy();counts=np.zeros(model.channels,np.int64)
    predictions=[];truths=[];masks=[]
    started=time.perf_counter()
    with torch.no_grad():
        for lo in range(0,len(origins),16):
            if time.time()>=context['deadline']:raise TimeoutError('Evaluation boundary')
            selected=origins[lo:lo+16]
            x,_,mask=windows.get(selected)
            p=fit.predict_model(model,x,foundation,'B2',model.latent_k)
            _,y,_=metric_windows.get(selected)
            if metric_norm is not None:
                mean=torch.as_tensor(normalization['mean']);std=torch.as_tensor(normalization['std'])
                cm=torch.as_tensor(canonical['mean']);cs=torch.as_tensor(canonical['std'])
                p=(p*std+mean-cm)/cs
            p,y,m=p.cpu().numpy(),y.cpu().numpy(),mask.cpu().numpy()
            assert p.shape==y.shape==m.shape and np.isfinite(p[m]).all()
            error=p.astype(np.float64)-y.astype(np.float64)
            sse+=np.where(m,error*error,0).sum((0,1),dtype=np.float64)
            sae+=np.where(m,np.abs(error),0).sum((0,1),dtype=np.float64)
            counts+=m.sum((0,1),dtype=np.int64)
            if save is not None:
                predictions.append(p);truths.append(y);masks.append(m)
            if context['psutil'].Process().memory_info().rss>3*1024**3:raise MemoryError('Evaluation RSS limit')
    active=counts>0
    result=dict(MSE=math.fsum((sse[active]/counts[active]).tolist())/int(active.sum()),
                MAE=math.fsum((sae[active]/counts[active]).tolist())/int(active.sum()),
                observed=int(counts.sum()),origins=len(origins),channels=model.channels,
                channel_sse=sse.tolist(),channel_sae=sae.tolist(),channel_counts=counts.tolist(),
                seconds=time.perf_counter()-started,scaler_sha256=fit.scaler_fingerprint(canonical['mean'],canonical['std']))
    if save is not None:
        arrays=dict(pred_normalized=np.concatenate(predictions),target_normalized=np.concatenate(truths),
                    observed=np.concatenate(masks),origins=origins,channel_ids=np.asarray(task['columns'],str))
        with Path(save).open('xb') as f:np.savez_compressed(f,**arrays)
        result['private_prediction_archive_sha256']=sha(save)
        result['array_fingerprints']={k:context['evaluation'].array_fingerprint(v) for k,v in arrays.items()}
    return result


def selected_bundle(seed,family,context):
    if family=='B2_CAP_POINTWISE':
        return context['private']/'runs'/f'cap_s{seed}'/'selected_bundle.pt'
    return old_run(context,seed,family)/'selected_bundle.pt'


def verify_bundle(bundle,seed,family,task,normalization,context):
    internal='B2' if family=='B2_CAP_POINTWISE' else family
    assert bundle['spec']['seed']==seed and bundle['spec']['family']==internal
    assert bundle['spec']['latent_k']==6 and bundle['spec']['device']=='cpu'
    contract=bundle['contract']
    assert contract['data_sha']==task['data_sha256'] and contract['model_sha']==F_SHA
    assert contract['source_sha']==context['fit']._source_contract()[0]
    assert contract['scaler_sha']==context['fit'].scaler_fingerprint(normalization['mean'],normalization['std'])
    if family=='B2_CAP_POINTWISE':assert bundle['architecture']=='B2_CAP_POINTWISE'


def make_selected(task,seed,family,F,context):
    torch,fit=context['torch'],context['fit']
    path=selected_bundle(seed,family,context)
    bundle=torch.load(path,map_location='cpu',weights_only=False)
    if family=='B2_CAP_POINTWISE':
        import cap_control
        model,norm=cap_control.create_cap(spec_for(seed),task,12579,F)
    else:
        model,norm=fit.create_model(spec_for(seed,family),task,12579,F)
    for key in ('mean','std','basis','pca_center'):
        assert context['np'].array_equal(norm[key],bundle['normalization'][key]),key
    verify_bundle(bundle,seed,family,task,norm,context)
    model.load_state_dict(bundle['model'],strict=True)
    return model,norm,bundle,path


def train_cap(args,context,task,F):
    import cap_control
    private=context['private']/'runs'/f'cap_s{args.seed}'
    public=context['public']/'runs'/f'cap_s{args.seed}'
    instrument(context,private)
    result=cap_control.fit_cap(args.seed,task,F,private,public,context['deadline'])
    if result['status']!='COMPLETE_STAGES':return {'status':'PARTIAL','details':result.get('phase')}
    model,norm,bundle,path=make_selected(task,args.seed,'B2_CAP_POINTWISE',F,context)
    origins=context['data'].origins(BOUNDS['V_select'],512,48,48)
    metrics=score(model,norm,task,origins,F,context,save=private/'selected_V.npz')
    metrics.update(seed=args.seed,model='B2_CAP_POINTWISE',dataset='jena',C=21,K=6,
                   selected_epoch=result['selected_epoch'],artifact_sha256=sha(path),fit='fresh_A60',
                   selection='A_stop_search_0_.25_1_3',eval='repeated_development_V43',same_update=False,
                   params=sum(p.numel() for p in model.parameters() if p.requires_grad),device='cpu',
                   status='COMPLETE_SELECTED_V',convergence_proven=False)
    new(public/'SELECTED_V_RESULT.json',metrics)
    return dict(status='COMPLETE_SELECTED_V',seed=args.seed,model='B2_CAP_POINTWISE',MSE=metrics['MSE'],MAE=metrics['MAE'])


def evaluate_search(args,context,task,F):
    import cap_control
    torch,fit,data=context['torch'],context['fit'],context['data']
    public=context['public']/'same_update'/f's{args.seed}'
    private=context['private']/'same_update'/f's{args.seed}'
    public.mkdir(parents=True,exist_ok=False);private.mkdir(parents=True,exist_ok=False)
    canonical_mean,canonical_std,_=data.fit_statistics(task,12579)
    canonical=dict(mean=canonical_mean,std=canonical_std)
    rows=[]
    for family in ('B0','B2','B2_CAP_POINTWISE'):
        source=(context['private']/'runs'/f'cap_s{args.seed}') if family=='B2_CAP_POINTWISE' else old_run(context,args.seed,family)
        for index,epoch in enumerate((0.,.25,1.,3.)):
            path=source/'search'/f'checkpoint_{index:06d}.pt'
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            expected_exposures=(0,2481,9923,29769)[index]
            expected_updates=(0,156,622,1863)[index]
            assert checkpoint['sampler']['exposures']==expected_exposures
            assert checkpoint['updates']==expected_updates
            if family=='B2_CAP_POINTWISE':model,norm=cap_control.create_cap(spec_for(args.seed),task,10482,F)
            else:model,norm=fit.create_model(spec_for(args.seed,family),task,10482,F)
            public_source=context['public']/'runs'/f'cap_s{args.seed}'/'SOURCE_CONTRACT.json' if family=='B2_CAP_POINTWISE' else context['campaign']/source.relative_to(context['old_cache'])/'SOURCE_CONTRACT.json'
            declared=read(public_source)
            for key in ('source_sha','config_sha','data_sha','model_sha'):
                assert checkpoint['contract'][key]==declared[key],key
            assert checkpoint['contract']['source_sha']==fit._source_contract()[0]
            assert checkpoint['contract']['data_sha']==task['data_sha256'] and checkpoint['contract']['model_sha']==F_SHA
            assert checkpoint['contract']['scaler_sha']==fit.scaler_fingerprint(norm['mean'],norm['std'])
            model.load_state_dict(checkpoint['model'],strict=True)
            for split in ('A_fit','A_stop','V_select'):
                bounds=BOUNDS[split]
                origins=data.origins(bounds,512,48,1 if split=='A_fit' else 48)
                archive=private/f'{family}_e{index}_V.npz' if split=='V_select' else None
                metrics=score(model,norm,task,origins,F,context,metric_norm=canonical if split=='V_select' else None,save=archive)
                value=dict(seed=args.seed,model=family,dataset='jena',C=21,K=6,epoch=epoch,
                           updates=expected_updates,exposure=expected_exposures,same_update=True,
                           params=sum(p.numel() for p in model.parameters() if p.requires_grad),
                           fit='A_fit_search',selection='NOT_SELECTED_COMMON_ENDPOINT',eval=split,
                           metric_scale='canonical_A60' if split=='V_select' else 'search_A50',
                           loss_kind='checkpoint_full_A_fit_diagnostic' if split=='A_fit' else 'evaluation',
                           checkpoint_sha256=sha(path),device='cpu',status='COMPLETE',**metrics)
                new(public/f'{family}_e{index}_{split}.json',value)
                rows.append({k:v for k,v in value.items() if not isinstance(v,(dict,list))})
            del model,checkpoint
    with (public/'COMPARISONS.csv').open('x',encoding='utf-8',newline='') as f:
        keys=sorted(set().union(*(set(r) for r in rows)))
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    return dict(status='COMPLETE_SAME_UPDATE',seed=args.seed,rows=len(rows))


def resume_pems(args,context,F):
    import pems_resume
    task,access,old_plan=pems_resume.load_task(context['campaign'],context['old_cache'])
    family=args.family
    private=context['private']/'runs'/f'pems_{family}'
    public=context['public']/'runs'/f'pems_{family}'
    instrument(context,private)
    audit=read(context['private']/f'PEMS_RESTORE_AUDIT_{family}'/'RESULT.json')
    result=pems_resume.resume_stages(family,task,F,context['campaign'],context['old_cache'],private,public,context['deadline'],audit_result=audit)
    if result['status']!='COMPLETE_STAGES':return dict(status='PARTIAL',phase=result['phase'])
    bundle={k:result[k] for k in ('model','normalization','selected_epoch','spec','contract')}
    bundle.update(search_summary=result['search']['summary'],refit_summary=result['refit']['summary'],lineage=result['lineage'])
    bundle['model_state_sha']=context['training'].fingerprint({key:context['data'].array_sha256(value.numpy()) for key,value in bundle['model'].items()})
    with (private/'selected_bundle.pt').open('xb') as f:context['torch'].save(bundle,f)
    model,norm=context['fit'].create_model(result['spec'],task,10713,F)
    model.load_state_dict(result['model'],strict=True)
    origins=context['data'].origins(pems_resume.BOUNDS['V_select'],512,12,12)
    metrics=score(model,norm,task,origins,F,context,save=private/'selected_V.npz')
    metrics.update(model=family,seed=92601,dataset='pems08_C32',C=32,K=8,same_update=False,
                   selected_epoch=result['selected_epoch'],fit='fresh_A60_after_resumed_A50',
                   selection='original_A_stop',eval='repeated_development_V148',device='cpu',
                   params=sum(p.numel() for p in model.parameters() if p.requires_grad),
                   status='COMPLETE_SELECTED_V',convergence_proven=False,
                   updates=result['refit']['summary']['updates'],exposure=result['refit']['summary']['exposures'])
    new(public/'SELECTED_V_RESULT.json',metrics)
    new(private/'ACCESS_RECEIPT.json',access)
    return dict(status='COMPLETE_SELECTED_V',model=family,dataset='pems08_C32',MSE=metrics['MSE'],MAE=metrics['MAE'])


def probe(args,context,task):
    import linear_probe
    # create_model requires a non-None callable identity but encode never calls it.
    model,norm,bundle,path=make_selected(task,args.seed,args.family,object(),context)
    fit_origins=context['data'].origins((0,12579),512,48,1)
    eval_origins=context['data'].origins(BOUNDS['V_select'],512,48,48)
    return linear_probe.run_probe(task,model,norm,fit_origins,eval_origins,
        context['private']/'probes'/f'{args.family}_s{args.seed}',
        context['public']/'probes'/f'{args.family}_s{args.seed}',batch=32,
        windows_factory=context['training'].FastWindows,rss_limit=3*1024**3,deadline=context['deadline'],
        run_metadata=dict(seed=args.seed,model=args.family,dataset='jena',C=21,K=6,
            fit='A60',eval='repeated_development_V43',selected_epoch=bundle.get('selected_epoch',bundle.get('selected_search_epoch')),
            checkpoint_sha256=sha(path),scaler_sha256=context['fit'].scaler_fingerprint(norm['mean'],norm['std'])))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--job',choices=('train-cap','evaluate-search','resume-pems','probe'))
    parser.add_argument('--seed',type=int,choices=(92601,92602),default=92601)
    parser.add_argument('--family',choices=('B0','B2','B2_CAP_POINTWISE'),default='B2')
    parser.add_argument('--repo',type=Path)
    parser.add_argument('--asset',type=Path)
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(status='SOURCE_ONLY_NO_ASSET_READS',model_calls=0,writes=0)));return
    if None in (args.job,args.repo,args.asset):parser.error('Explicit job/repo/asset required')
    context=bootstrap(args)
    tag=f'{args.job}_{args.family}_{args.seed}'
    jobdir=context['private']/'jobs'/tag
    jobdir.mkdir(parents=True,exist_ok=False)
    started=time.time()
    new(jobdir/'STARTED.json',dict(pid=os.getpid(),started_unix=started,job=args.job,seed=args.seed,family=args.family,
        root_scope_sha256=context['scope_sha256'],runner_sha256=sha(__file__),CPU_threads=2,GPU_used=False))
    pool=None
    try:
        if args.job=='resume-pems':task=None
        else:
            task,access=load_jena(args,context)
            new(jobdir/'ACCESS_RECEIPT.json',access)
        if args.job=='probe':result=probe(args,context,task)
        else:
            pool=context['deployment'].FoundationPool(args.asset/MODEL_REL,'cpu')
            F=pool.get(12 if args.job=='resume-pems' else 48)
            assert F.initial_hash==F_SHA
            if args.job=='train-cap':result=train_cap(args,context,task,F)
            elif args.job=='evaluate-search':result=evaluate_search(args,context,task,F)
            else:result=resume_pems(args,context,F)
            F.assert_frozen_unchanged()
        new(jobdir/'RESULT.json',dict(status='JOB_COMPLETE',result=result,seconds=time.time()-started,
            scientific_hypothesis_supported_not_assumed=True,protected_T_numeric_access=0))
        print(json.dumps(dict(job=args.job,seed=args.seed,family=args.family,result=result),ensure_ascii=False,default=str))
    except BaseException as exc:
        new(jobdir/'RESULT.json',dict(status='PARTIAL_OR_FAILED',error_type=type(exc).__name__,error=str(exc),
            traceback=traceback.format_exc(),seconds=time.time()-started,protected_T_numeric_access=0))
        raise
    finally:
        if pool is not None:pool.close()


if __name__=='__main__':main()
