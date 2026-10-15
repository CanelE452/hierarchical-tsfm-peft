"""Cost precedes first C scoring; selection and cost receipts are mandatory."""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
import psutil
import runtime as rt
import campaign as cp
import deployment as dp
from cost_measure import measure_routes, measure_one
from transforms import r1_metadata


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def assert_no_training(check_exclusive=False):
    competing=[]
    for p in psutil.process_iter(['pid','cmdline']):
        if p.pid==psutil.Process().pid:continue
        cmd=' '.join(p.info['cmdline'] or [])
        if 'python' not in cmd.lower():continue
        if 'campaign.py core' in cmd or 'followup.py' in cmd:competing.append(p.pid)
        elif check_exclusive and 'cost_and_score.py exclusive --id' in cmd:competing.append(p.pid)
    if competing:raise RuntimeError(f'COMPETING_RESEARCH_PROCESS:{competing}')
    return competing


def uses_bolt(spec):
    if spec['type']=='G':return uses_bolt(spec['base'])
    return spec['type'] in ('RAW','R1','ADAPTER','LORA')


class NoBolt:
    def __init__(self):
        self.model=torch.nn.Module().cuda();self.model.register_buffer('device_marker',torch.empty(0,device='cuda'))
        self.initial_hash=rt.state_hash(self.model);self.row_microbatch=None
    def __call__(self,*args,**kwargs):raise RuntimeError('NO_F_ROUTE_CALLED_F')


def rows():
    lock=read(rt.HERE/'SELECTION_LOCK.json')
    assert lock['status']=='ALL_SELECTIONS_LOCKED_BEFORE_C' and not lock['C_scored']
    for name,digest in lock['source_hashes'].items():assert rt.sha(rt.HERE/name)==digest, 'SOURCE_CHANGED:'+name
    for name,digest in lock['artifact_hashes'].items():assert rt.sha(rt.CACHE/name)==digest,'SELECTED_ARTIFACT_CHANGED:'+name
    assert rt.sha(rt.HERE/'FOLD_STATISTICS.json')==lock['statistics_hash']
    return lock['routes']


def choose_cost_routes(routes):
    selected=[]
    for dataset in ('beijing','jena','robin'):
        pool=[r for r in routes if r['dataset']==dataset and r['fold']=='primary']
        for typ in ('RAW','CURRENT','G','DIRECT','SWAP','LORA','RIDGE'):
            found=[r for r in pool if r['type']==typ and r.get('length',512)==512]
            if typ in ('G','DIRECT','SWAP'):
                selected+=found
            elif found:selected.append(found[0])
        for kind in ('B0','B1','B2','B3','B4'):
            for rank in sorted(set(r['r'] for r in pool if r['type']=='ADAPTER' and r['kind']==kind and r.get('block','R2')=='R2')):
                selected.append(min([r for r in pool if r['type']=='ADAPTER' and r['kind']==kind and r['r']==rank and r.get('block','R2')=='R2'],
                    key=lambda v:v['selected_B']['mse']))
        for typ in ('ADAPTER',):
            if dataset=='robin':selected += [r for r in pool if r['type']==typ]
        selected += [r for r in pool if r['type']=='R1' or (r['type']=='RAW' and r.get('length',512)>512)]
        raw=next((r for r in pool if r['type']=='RAW' and r.get('length',512)==512),None)
        if raw:selected.append(next(r for r in pool if r['id']==raw['id']+'_micro4' and r.get('row_microbatch')==4))
    return list({r['id']:r for r in selected}.values())


def method_metadata(spec):
    typ=spec['type']
    if typ=='G':return dict(method_metadata(spec['base']),G=True)
    if typ=='SWAP':return dict(method_metadata(spec['base']),F='native_shared_3feature_head',G=False)
    if typ=='ADAPTER':return dict(P=spec['kind'],E='learned_pointwise_C_to_r',decoder='origin_reference_lag7',F='Bolt-small',G=False)
    if typ=='R1':return dict(P=spec['variant'],E='identity',decoder='R1_original_grid_restore',F='Bolt-small',G=False)
    if typ=='DIRECT':return dict(P=spec['kind'],E='none',decoder='direct_prediction_head',F='none',G=False)
    if typ=='RIDGE':return dict(P='block33',E='none',decoder='self_channel_linear_head',F='none',G=False)
    return dict(P='identity',E='identity',decoder='persistence' if typ=='CURRENT' else 'identity',
                F='none' if typ=='CURRENT' else 'Bolt-LoRA-qv8' if typ=='LORA' else 'Bolt-small',G=False)


def make_context(bolt,input_lengths=None):
    windows={}
    def window(spec):
        key=(spec['dataset'],spec.get('fold','primary'))
        if key not in windows:windows[key]=cp.dataset_windows(*key)
        return windows[key]
    def factory(spec):return dp.Route(spec,window(spec),bolt)
    def provider(dataset,batch):
        win=window(dict(dataset=dataset,fold='primary')); origins=win.origins['B']
        length=(input_lengths or {}).get(dataset,512)
        origins=origins[origins>=length]
        if not len(origins):raise ValueError('NO_COMMON_COST_ORIGINS')
        origins=np.resize(origins,batch)
        physical=np.stack([win.data['physical'][o-length:o] for o in origins])
        mask=np.stack([win.data['observed'][o-length:o] for o in origins])
        return physical,mask,win.mean,win.std
    return window,factory,provider


def parity(bolt,routes,window,factory,provider,output_name='ACTUAL_PARITY.json'):
    receipts=[]
    for dataset in ('beijing','jena','robin'):
        spec=next(r for r in routes if r['dataset']==dataset and r['type']=='RAW' and r.get('length',512)==512)
        win=window(spec)
        for batch in (1,8):
            physical,mask,mean,std=provider(dataset,batch)
            canonical=cp.infer(cp.Raw(),bolt,win,np.resize(win.origins['B'],batch),batch=batch)
            route=factory(spec); normal=route.predict_physical(physical,mask,mean,std);route.close()
            micro=factory(dict(**spec,row_microbatch=4));small=micro.predict_physical(physical,mask,mean,std);micro.close()
            previous=bolt.row_microbatch;bolt.row_microbatch=4
            canonical_micro=cp.infer(cp.Raw(),bolt,win,np.resize(win.origins['B'],batch),batch=batch)
            bolt.row_microbatch=previous
            valid=bool(np.allclose(canonical_micro,canonical,atol=1e-5,rtol=2e-6))
            checks=dict(microbatch=dict(max_abs_normalized=float(np.max(abs(canonical_micro-canonical))),pass_=valid))
            if not valid:raise AssertionError('RAW_NORMALIZED_MICROBATCH_PARITY_'+dataset)
            for label,value,reference_norm in [('physical_pipeline',normal,canonical),('physical_microbatch',small,canonical_micro)]:
                reference=(reference_norm*std+mean).astype(np.float32)
                difference=np.abs(value.astype(np.float64)-reference.astype(np.float64))/std
                # Compare actual physical FP32 outputs, allowing their deterministic quantization bound.
                tolerance=1e-5+2e-6*np.abs(reference_norm)+2*np.abs(np.spacing(reference))/std
                valid=bool(np.all(difference<=tolerance))
                checks[label]=dict(max_abs_normalized=float(difference.max()),pass_=valid,
                    physical_FP32_quantization_bound_used=True)
                if not valid:raise AssertionError('RAW_PHYSICAL_PIPELINE_PARITY_'+dataset+'_'+label)
            receipts.append(dict(dataset=dataset,batch=batch,checks=checks))
        # Rebuild actual saved CP0 for all identity-r=C temporal fronts.
        for kind in ('B0','B1','B2','B3','B4'):
            spec0=next((r for r in routes if r['dataset']==dataset and r['fold']=='primary' and r['type']=='ADAPTER'
                        and r.get('kind')==kind and r.get('r')==win.channels),None)
            if spec0 is None:continue
            cfg=dict(spec0,checkpoint=f"checkpoints/{spec0['id']}/cp0.pt")
            model=dp.build(cfg,win,bolt)
            a=cp.infer(model,bolt,win,win.origins['B'][:8]);b=cp.infer(cp.Raw(),bolt,win,win.origins['B'][:8])
            ok=bool(np.allclose(a,b,atol=1e-5,rtol=2e-6))
            receipts.append(dict(dataset=dataset,kind=kind,CP0_raw_parity=ok,max_abs=float(np.max(abs(a-b)))))
            if not ok:raise AssertionError('CP0_IDENTITY_PARITY')
    rt.write(rt.HERE/output_name,dict(receipts=receipts,status='PASS',C_scored=False,F_model_hash=bolt.initial_hash,
        statistics_sha256=rt.sha(rt.HERE/'FOLD_STATISTICS.json'),atol=1e-5,rtol=2e-6))


def cost():
    selected=choose_cost_routes(rows())
    # No other campaign training process is allowed during precision latency.
    competing=assert_no_training()
    selection_sha=rt.sha(rt.HERE/'SELECTION_LOCK.json')
    rt.write(rt.HERE/'COST_SELECTION.json',dict(routes=selected,selection_lock_sha256=selection_sha,
        time_unix=time.time(),training_competitors=competing,C_scored=False))
    bolt=rt.Bolt();parity(bolt,rows(),*make_context(bolt))
    lengths={d:max(r.get('length',512) for r in selected if r['dataset']==d) for d in ('beijing','jena','robin')}
    window,factory,provider=make_context(bolt,lengths)
    outputs=[]
    for dataset in ('beijing','jena','robin'):
        group=[r for r in selected if r['dataset']==dataset]
        result=measure_routes(group,factory,provider,bolt,rt.HERE/'cost'/dataset)
        outputs.append(result['metadata'])
        rt.event(stage='COST_GENERAL_COMPLETE',dataset=dataset,status=result['metadata']['status'])
    assert rt.state_hash(bolt.model)==bolt.initial_hash
    # Release this CUDA process before launching exclusive route processes externally.
    rt.write(rt.HERE/'GENERAL_COST_COMPLETE.json',dict(status='COMPLETE_VALID' if all(r['status']=='COMPLETE_VALID' for r in outputs) else 'INVALID',
        metadata=outputs,C_scored=False,F_unchanged=True,selection_lock_sha256=selection_sha,
        cost_selection_sha256=rt.sha(rt.HERE/'COST_SELECTION.json')))


def exclusive(identifier):
    assert_no_training(check_exclusive=True)
    selected=read(rt.HERE/'COST_SELECTION.json')['routes'];spec=next(r for r in selected if r['id']==identifier)
    lengths={d:max(r.get('length',512) for r in selected if r['dataset']==d) for d in ('beijing','jena','robin')}
    bolt=rt.Bolt() if uses_bolt(spec) else NoBolt();window,factory,provider=make_context(bolt,lengths)
    result=measure_one(spec,factory,provider,bolt,rt.HERE/'cost_exclusive'/identifier,exclusive=True)
    if spec['type']=='ADAPTER':
        win=window(spec);route=factory(spec)
        x,y,mask=win.get(win.origins['A'][:4]);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter();pred=route.model(x,bolt);loss=rt.loss(pred,y,mask);loss.backward();torch.cuda.synchronize()
        rt.write(rt.HERE/'cost_exclusive'/identifier/'TRAINING_GRADIENT_MEMORY.json',dict(
            diagnostic='Selected weights, one real A batch forward/backward, no optimizer update',batch=4,
            wall_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),F_parameters_frozen=all(not p.requires_grad for p in bolt.model.parameters()),
            F_parameter_gradients_none=all(p.grad is None for p in bolt.model.parameters()),
            E_gradient_norm=float(route.model.E.grad.norm()),D_gradient_norm=float(route.model.D.grad.norm()),optimizer_updates=0))
        route.close()
    assert rt.state_hash(bolt.model)==bolt.initial_hash
    rt.event(stage='EXCLUSIVE_COST_COMPLETE',id=identifier,status=result['metadata']['status'])


def exclusive_all():
    assert_no_training(check_exclusive=True)
    general=read(rt.HERE/'GENERAL_COST_COMPLETE.json');assert general['status']=='COMPLETE_VALID'
    assert general['selection_lock_sha256']==rt.sha(rt.HERE/'SELECTION_LOCK.json')
    assert general['cost_selection_sha256']==rt.sha(rt.HERE/'COST_SELECTION.json')
    selected=read(rt.HERE/'COST_SELECTION.json')['routes']
    # Memory claims use Raw, B0, selected B2/B4, matched G and the strongest direct/native alternatives.
    essential=[]
    for d in ('beijing','jena','robin'):
        pool=[r for r in selected if r['dataset']==d]
        for typ in ('RAW','G','DIRECT','SWAP','RIDGE','LORA'):
            essential += [r for r in pool if r['type']==typ]
        essential += [r for r in pool if r['type']=='ADAPTER' and r.get('kind') in ('B0','B2','B4')]
    essential=list({r['id']:r for r in essential}.values())
    outcomes=[]
    for spec in essential:
        folder=rt.HERE/'cost_exclusive'/spec['id']
        if folder.exists():raise FileExistsError(folder)
        result=subprocess.run([sys.executable,'-X','utf8',str(Path(__file__)),'exclusive','--id',spec['id']],check=False)
        meta=read(folder/'COST_RUN_METADATA.json') if (folder/'COST_RUN_METADATA.json').exists() else {}
        outcomes.append(dict(id=spec['id'],exit_code=result.returncode,status=meta.get('status','MISSING')))
    ok=all(r['exit_code']==0 and r['status']=='COMPLETE_VALID' for r in outcomes)
    flattened=[]
    for scope,base in [('general',rt.HERE/'cost'),('exclusive',rt.HERE/'cost_exclusive')]:
        for path in base.glob('*/COST_SUMMARY.json'):
            for row in read(path):
                statistics=row['statistics']
                flat=dict(id=row['route_id'],dataset=row['dataset'],batch=row['batch'],scope=scope,status=row['status'],
                    whole_seconds_median=statistics['whole_seconds']['median'],
                    whole_seconds_variance=statistics['whole_seconds']['population_variance'],
                    peak_allocated_bytes_median=statistics['peak_allocated_bytes']['median'],
                    peak_reserved_bytes_median=statistics['peak_reserved_bytes']['median'],
                    F_resident_bytes=row['F_resident_parameter_buffer_bytes'],
                    active_parameter_buffer_bytes=row['active_parameter_buffer_bytes'],
                    valid_trials=row['valid_trials'],measured_trials=row['measured_trials'])
                for stage in ('preprocessing','PE','F','DG','return'):
                    flat[stage+'_seconds_median']=statistics[stage+'_seconds']['median']
                flattened.append(flat)
    with (rt.HERE/'COST.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(flattened[0]));writer.writeheader();writer.writerows(flattened)
    rt.write(rt.HERE/'COST_LOCK.json',dict(status='COMPLETE_VALID' if ok else 'INCOMPLETE',general_cost=rt.sha(rt.HERE/'GENERAL_COST_COMPLETE.json'),
        selection_lock=general['selection_lock_sha256'],cost_selection_sha256=general['cost_selection_sha256'],
        exclusive=outcomes,time_unix=time.time(),C_scored=False))


def score():
    lock=read(rt.HERE/'COST_LOCK.json');assert lock['status']=='COMPLETE_VALID'
    assert lock['selection_lock']==rt.sha(rt.HERE/'SELECTION_LOCK.json')
    assert lock['cost_selection_sha256']==rt.sha(rt.HERE/'COST_SELECTION.json')
    assert lock['general_cost']==rt.sha(rt.HERE/'GENERAL_COST_COMPLETE.json')
    selected_routes=rows()
    rt.write(rt.HERE/'FIRST_C_SCORE.json',dict(time_unix=time.time(),selection_lock=lock['selection_lock'],cost_lock=rt.sha(rt.HERE/'COST_LOCK.json'),
        algorithmically_held_out_this_run=True,historically_unseen=False))
    bolt=rt.Bolt();window,_,_=make_context(bolt)
    result=[]; fail=[]; folder=rt.CACHE/'predictions';folder.mkdir(exist_ok=False)
    for spec in selected_routes:
        model=None
        previous_microbatch=bolt.row_microbatch
        bolt.row_microbatch=spec.get('row_microbatch')
        try:
            win=window(spec);model=dp.build(spec,win,bolt); length=spec.get('length',512)
            archive={}
            for split in ('B','C'):
                origins=win.origins[split][win.origins[split]>=length]
                if not len(origins):continue
                prediction=cp.infer(model,bolt,win,origins,length=length)
                y,mask=win.truth(origins);metric=rt.score(prediction,y,mask)
                physical_target=np.where(mask,win.data['physical'][origins[:,None]+np.arange(rt.H)[None]],0).astype(np.float32)
                physical=rt.score(prediction*win.std+win.mean,physical_target,mask)
                name=spec['id'];cohort=rt.ahash(origins)
                meta=r1_metadata(spec['variant'],length,win.channels,period=win.data['period_from_existing_metadata']) if spec.get('variant') else {}
                architecture=method_metadata(spec)
                foundation_context=meta.get('foundation_context_length',length) if architecture['F'].startswith('Bolt') else 0
                record=dict(id=name,dataset=spec['dataset'],fold=spec['fold'],split=split,cohort=cohort,
                    method=spec['type'],kind=spec.get('kind',spec.get('variant','')),r=spec.get('r',win.channels),
                    seed=spec.get('seed',0),updates=spec.get('updates',0),selected_update=spec.get('selected_update',0),
                    mse=metric['mse'],mae=metric['mae'],n_observed=metric['n_observed'],n_origins=len(origins),
                    length=length,physical_past_length=length,F_context_length=foundation_context,
                    F_row_microbatch=spec.get('row_microbatch','full') if foundation_context else 'none',
                    horizon=48,mask='original_observed',scaler='fold_A_only',input='past_only',
                    **architecture,
                    grid='original',status='COMPLETE_VALID',training_status=spec.get('training_status','NOT_APPLICABLE'),
                    algorithmically_held_out_this_run=split=='C',historically_unseen=False)
                result.append(record)
                archive.update({f'{split}_pred':prediction,f'{split}_target':y,f'{split}_mask':mask,f'{split}_origins':origins,
                    f'{split}_target_physical':physical_target})
                rt.write(rt.HERE/'scores'/f'{name}_{split}.json',dict(**record,normalized=metric,physical_per_channel=physical,
                    horizon_metrics={label:rt.score(prediction[:,lo:hi],y[:,lo:hi],mask[:,lo:hi])
                        for label,lo,hi in [('1-12',0,12),('13-24',12,24),('25-48',24,48)]}))
                # Matched-lookback controls do not share scores from a different origin cohort.
                comparison_lengths=[]
                if spec.get('variant') in ('A12','A13'):comparison_lengths=[512]
                if spec.get('variant')=='A11':comparison_lengths=[512//win.data['period_from_existing_metadata']*win.data['period_from_existing_metadata']]
                for compare_length in comparison_lengths:
                    if spec.get('variant')=='A11':
                        rawpred=cp.infer(dp.CropRaw(compare_length),bolt,win,origins,length=512)
                    else:rawpred=cp.infer(cp.Raw(),bolt,win,origins,length=compare_length)
                    rawmetric=rt.score(rawpred,y,mask)
                    ref=dict(record,id=name+f'_matched_RawL{compare_length}',method='RAW',kind='MATCHED_COHORT',length=compare_length,
                        physical_past_length=512,F_context_length=compare_length,**method_metadata(dict(type='RAW')),
                        mse=rawmetric['mse'],mae=rawmetric['mae'])
                    result.append(ref)
                    archive[f'{split}_matched_RawL{compare_length}']=rawpred
                rt.event(stage='SCORE',id=name,split=split,mse=metric['mse'])
            with (folder/f"{spec['id']}.npz").open('xb') as f:np.savez_compressed(f,**archive)
        except Exception as error:
            fail.append(dict(id=spec['id'],status='INVALID',error=str(error)))
            rt.event(stage='SCORING_FAILURE',id=spec['id'],error=str(error))
        finally:
            if model is not None:
                for module in model.modules():
                    if isinstance(module,dp.LoRA):module.close()
            bolt.row_microbatch=previous_microbatch
            del model;torch.cuda.empty_cache()
    with (rt.HERE/'SCORES.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(result[0]));writer.writeheader();writer.writerows(result)
    unchanged=rt.state_hash(bolt.model)==bolt.initial_hash
    rt.write(rt.HERE/'SCORE_COMPLETE.json',dict(status='COMPLETE_VALID' if not fail and unchanged else 'INCOMPLETE',rows=len(result),failures=fail,
        F_unchanged=unchanged,time_unix=time.time(),scoring_F_forward_calls=bolt.calls,scoring_F_forecast_streams=bolt.streams))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['cost','exclusive','exclusive_all','score']);parser.add_argument('--id')
    args=parser.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if args.phase=='exclusive':exclusive(args.id)
    else:globals()[args.phase]()
